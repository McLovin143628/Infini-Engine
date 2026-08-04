//! `runtime_sim` — the player's actor-ticking loop (P9.3 item 2).
//!
//! This mirrors the editor's `inf_editor_core::simulate::SimSession` semantics
//! **without any editor crate**: it drives Blueprints + 2D physics over a plain
//! [`EcsWorld`] using only Ring-0 crates (`inf-ecs`, `inf-physics`,
//! `inf-blueprint`, `inf-input`, `inf-runtime`). The editor's Simulate and the
//! shipped player therefore run gameplay through the *same* interpreter + the
//! *same* physics bridge — preview == shipped, one more time.
//!
//! # The fixed step (deterministic, `Guid`-ordered)
//!
//! Each fixed step runs, in order (matching `SimSession::fixed_step`):
//!
//! 1. `bridge.sync_from_world` — mirror the ECS bodies/colliders into rapier.
//! 2. **Blueprint `Tick`** for every actor in `Guid` order, through a
//!    [`RuntimeHost`] whose `physics()` accessor is a real [`Physics2dHost`] over
//!    the [`PhysicsBridge2D`] and whose `input.*` reads the held-action set.
//! 3. `bridge.step(dt)` — advance the solver (dynamic bodies).
//! 4. `bridge.write_back` — dynamic poses → ECS `Transform`s; then propagate.
//!
//! # Differences from `SimSession`
//!
//! `SimSession` borrows the world from a `SceneDoc` and snapshots/restores it on
//! enter/exit (the editor must not leave an edit behind). The player *owns* its
//! [`EcsWorld`] and has nothing to restore, so [`RuntimeSim`] holds the world
//! directly and drops the snapshot machinery. The blueprint/physics/input host is
//! otherwise identical (the [`RuntimeHost`] below is a line-for-line analogue of
//! the editor's `SimHost`). A future refactor could hoist this into `inf-runtime`
//! as an alternative schedule (documented follow-up).
//!
//! # Entity identity
//!
//! Blueprints address entities as opaque `i64`s. Each actor is assigned a stable
//! `i64` in `Guid` order and its `entity` member variable is seeded with it, so
//! `vars::get("entity")` feeds the `physics2d.*` nodes; the host maps
//! `i64 → Guid → body handle`.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use glam::{DQuat, DVec2, DVec3};
use uuid::Uuid;

use inf_anim::state_machine::{SmContext, SmRuntime, StateMachine};
use inf_anim::{root_delta, AnimClip, Skeleton};
use inf_audio::{
    Attenuation, AttenuationModel, AudioAsset, AudioCommand, AudioEngine, Listener, PlayCommand,
};
use inf_blueprint::interp::{
    AudioHost, MoveResult2d, MoveResult3d, Physics2dHost, Physics3dHost, RayHit2d, RayHit3d,
};
use inf_blueprint::semantics::run_event;
use inf_blueprint::{ActorInstance, BlueprintClass, EventKind, Host, InterpDebug, RunError, Value};
use inf_ecs::components::{
    AnimPlayer, AnimStateMachine, AudioListener, AudioSource, CharacterController2D,
    CharacterController3D, Collider2D, Collider3D, ColliderShape2DKind, ColliderShape3DKind,
    DistanceModel, GlobalTransform, RootMotion, RootMotionMode, SmRuntimeState, Terrain, Transform,
};
use inf_ecs::{sim_snapshot, update_attachments, EcsWorld, Entity, Guid};
use inf_physics::d3::WaterEventKind3D;
use inf_physics::{
    CharacterMover2D, CharacterMover3D, ColliderShape2D, ColliderShape3D, ContactPhase,
    PhysicsBridge2D, PhysicsBridge3D,
};
use inf_runtime::FixedStep;
use inf_voxel::VoxelData;

// ── Wave 3 (MIRROR of inf_editor_core::simulate) ─────────────────────────────
// The event-dispatch cap + the sensor-overlap seam are duplicated field-for-field
// with the editor `SimSession` so the shipped player and the editor Simulate drain
// events identically (preview == shipped). See that module for the rationale.

/// The per-step cap on chained event dispatches (Wave 3) — the shipped mirror of
/// the editor `DISPATCH_ROUND_CAP`.
const DISPATCH_ROUND_CAP: u32 = 64;

/// Whether a trigger-volume (sensor) overlap began or ended this step — the
/// shipped mirror of `inf_editor_core::simulate::OverlapPhase`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OverlapPhase {
    /// The two sensor colliders started overlapping this step.
    Begin,
    /// They stopped overlapping this step.
    End,
}

/// A drained **sensor-pair** overlap for one fixed step: two entity `Guid`s
/// (canonical `a < b`) + phase — the shipped mirror of
/// `inf_editor_core::simulate::OverlapEvent`. [`RuntimeSim::drained_overlaps`]
/// exposes the per-step list; it is the seam trigger-volume gameplay + the
/// editor/runtime parity tests pin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OverlapEvent {
    /// The lower-`Guid` entity of the overlapping pair.
    pub a: Uuid,
    /// The higher-`Guid` entity of the overlapping pair.
    pub b: Uuid,
    /// Begin (started overlapping) or End (stopped).
    pub phase: OverlapPhase,
}

/// The set of currently-held actions/keys for one tick (analogue of
/// `SimSession::SimInput`). Rising edges (`just_pressed`) are derived by the
/// [`RuntimeSim`] from the previous tick's set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeInput {
    down: BTreeSet<String>,
}

impl RuntimeInput {
    /// An input state with the given actions/keys held.
    pub fn with_down<I, S>(keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            down: keys.into_iter().map(Into::into).collect(),
        }
    }

    /// Whether `key` is currently held.
    pub fn is_down(&self, key: &str) -> bool {
        self.down.contains(key)
    }

    /// Mark `key` held (builder-style).
    pub fn press(mut self, key: impl Into<String>) -> Self {
        self.down.insert(key.into());
        self
    }
}

/// A per-fixed-step hook the sim ticks after gameplay/physics/anim, given
/// mutable access to the world + the blueprint entity-id map + this step's input
/// (P14.5). The WASM mod loader implements it (`crate::mods::PlayerMods`); the
/// trait names no `wasmtime`/`inf-wasm-host` types so the sim stays buildable for
/// `wasm32` (the browser player, which loads no native mods).
pub trait ModHook {
    /// Advance mods one fixed step. `entities` is the live blueprint `i64 →
    /// Guid` map (mutable so a spawning mod can register new entities).
    fn tick(
        &mut self,
        world: &mut EcsWorld,
        entities: &mut BTreeMap<i64, Uuid>,
        dt: f64,
        input: &RuntimeInput,
    );
}

/// One actor's live class + instance state.
struct ActorState {
    class: BlueprintClass,
    instance: ActorInstance,
}

/// A resolved `.inf_anim` clip + its skeleton, registered for **root-motion**
/// extraction (P11.3) — the runtime mirror of the editor `SimSession`'s
/// `RootClip`.
struct RootClip {
    skeleton: Skeleton,
    clip: AnimClip,
}

/// A headless gameplay simulation over one owned [`EcsWorld`].
pub struct RuntimeSim {
    world: EcsWorld,
    bridge: PhysicsBridge2D,
    /// The 3D physics bridge (P11.3), driven alongside the 2D one for
    /// `physics3d.*` nodes + root-motion movers.
    bridge3d: PhysicsBridge3D,
    /// Clips resolvable for root motion, keyed by `.inf_anim` GUID (P11.3).
    clips: BTreeMap<Uuid, RootClip>,
    /// Resolvable `.inf_sm` state machines keyed by asset GUID (P11.2). Entities
    /// with an `AnimStateMachine` whose `sm` GUID resolves here step each tick.
    state_machines: BTreeMap<Uuid, StateMachine>,
    stepper: FixedStep,
    /// Actors keyed by `Guid` (deterministic iteration).
    actors: BTreeMap<Uuid, ActorState>,
    /// Blueprint `i64` entity id → its `Guid`.
    entities: BTreeMap<i64, Uuid>,
    /// Currently-held actions/keys.
    input: RuntimeInput,
    /// Held the previous tick (for rising-edge detection).
    prev_down: BTreeSet<String>,
    /// Rising edges pending this fixed step.
    just_pressed: BTreeSet<String>,
    /// Falling edges pending this fixed step (Wave 3 input release events) —
    /// MIRROR of the editor `SimSession::just_released`.
    just_released: BTreeSet<String>,
    /// Wave 3 event dispatchers (MIRROR of `SimSession::bindings`): `(source
    /// entity, event name) → {listener entity → handler custom-event name}`.
    bindings: BTreeMap<(i64, String), BTreeMap<i64, String>>,
    /// FIFO queue of pending `(target entity, event name)` dispatches — MIRROR of
    /// `SimSession::dispatch_queue`.
    dispatch_queue: VecDeque<(i64, String)>,
    /// Sensor-pair overlaps drained this fixed step (canonical `a < b`, sorted) —
    /// MIRROR of `SimSession::drained_overlaps`.
    drained_overlaps: Vec<OverlapEvent>,
    /// Accumulated `debug.print` output.
    logs: Vec<String>,
    /// Last `move_and_slide` grounded result per actor.
    grounded: BTreeMap<Uuid, bool>,
    /// P12.3 audio (the shipped mirror of the editor `SimSession`): the long-lived
    /// host `AudioEngine` (a device build enables `cpal` — see the crate manifest;
    /// headless/CI runs the no-device fallback consistently). Output-only, not sim
    /// state: systems enqueue `audio_cmds`, drained here after each step.
    audio: AudioEngine,
    /// Resolvable `.inf_audio` payloads keyed by asset GUID (seeded by the level
    /// loader). Decoded lazily on first play.
    audio_clips: BTreeMap<Uuid, AudioAsset>,
    /// The audio command queue: filled by autoplay + Blueprint `audio.*` nodes,
    /// drained into `audio` at the end of a step (the determinism seam).
    audio_cmds: Vec<AudioCommand>,
    /// Entity `Guid`s whose autoplay `AudioSource` has already started.
    audio_started: BTreeSet<Uuid>,
    /// Accumulated drained audio command stream (determinism telemetry / test seam).
    audio_log: Vec<AudioCommand>,
    /// Total fixed steps run.
    steps: u64,
    /// World-space translations one fixed step ago, for render interpolation.
    prev_positions: HashMap<Uuid, DVec3>,
    /// World-space translations at the current fixed step.
    cur_positions: HashMap<Uuid, DVec3>,
    /// Optional sandboxed WASM mods, ticked each fixed step (P14.5). `None` on
    /// the browser player + any run without `--mods`.
    mods: Option<Box<dyn ModHook>>,
    /// Camera-driven terrain streaming (P16.3b2). Empty for every world with no
    /// asset-backed `Terrain`, in which case every call below is a no-op and the
    /// step is bit-identical to the pre-P16.3b2 one.
    ///
    /// **The determinism seam.** Its `sync_sim` runs at the TOP of
    /// [`fixed_step`](Self::fixed_step) from sim state alone; its `sync_render`
    /// is only ever called by the frame loop
    /// ([`sync_render_terrain`](Self::sync_render_terrain)) and writes into a
    /// working set no entity references. See [`crate::terrain_stream`].
    terrain: crate::terrain_stream::TerrainStreaming,
    /// World-partition cell streaming (P16.5). Empty for every unpartitioned
    /// level, in which case every call below is a no-op and the step is
    /// bit-identical to the pre-P16.5 one.
    ///
    /// **The other determinism seam, and the stricter one.** Terrain residency
    /// only costs detail; cell residency decides which entities *exist*, so its
    /// wants come from sim entities alone (`StreamingSource`) and it has no
    /// render half at all. `sync_sim` runs at the very TOP of
    /// [`fixed_step`](Self::fixed_step), before even terrain, so a cell's
    /// entities are present before anything — including terrain's own observer
    /// scan — can look for them. See [`crate::cell_stream`].
    cells: crate::cell_stream::CellStreaming,
    /// The **simulation's own** voxel volumes, keyed by entity `Guid` (P21.2).
    ///
    /// Read by the `terrain.height_at` host seam so a character can stand on a
    /// cave floor where the heightfield has been carved through. Empty by default;
    /// seed via [`set_voxel_volumes`](Self::set_voxel_volumes).
    ///
    /// **THE DETERMINISM SEAM.** This is emphatically *not* the render host's
    /// store: that one is paged by a camera (`inf_voxel::VoxelVolumes::sync_camera`)
    /// and a fixed step must never be able to observe where anyone is looking. The
    /// separation is structural rather than conventional — the render store lives
    /// in the render host, nothing here holds a reference to it, and this map is
    /// filled once from level state. Verbatim the split `inf_terrain::wants`
    /// documents between `sync_sim` and `sync_render`, and it is paid for the same
    /// way: a volume wanted by both is held twice, on purpose.
    voxels: BTreeMap<Uuid, VoxelData>,
}

/// The obstruction gain (linear) applied to an occluded spatial source — a −12 dB
/// cut (P12.3); the shipped mirror of the editor constant.
const OCCLUSION_CUT_LINEAR: f64 = 0.251_188_643_150_958; // 10^(-12/20)

impl RuntimeSim {
    /// Build a runtime sim over `world`, ticking `actors` (each an entity `Guid`
    /// paired with the [`BlueprintClass`] to run on it). `gravity` is the world
    /// gravity handed to the physics bridge (the platformer sample uses
    /// [`DVec2::ZERO`] — the character applies its own gravity in the blueprint);
    /// `hz` is the fixed update rate.
    pub fn new(
        mut world: EcsWorld,
        actors: Vec<(Uuid, BlueprintClass)>,
        gravity: DVec2,
        hz: f64,
    ) -> Self {
        world.propagate();
        let bridge = PhysicsBridge2D::new(gravity);
        // P11.3: a 3D bridge alongside the 2D one (2D vertical gravity → world −Y).
        let bridge3d = PhysicsBridge3D::new(DVec3::new(0.0, gravity.y, 0.0));

        // Assign stable i64 ids in Guid order, seed the `entity` variable.
        let mut entities = BTreeMap::new();
        let mut states = BTreeMap::new();
        let mut sorted = actors;
        sorted.sort_by_key(|(g, _)| *g);
        for (idx, (guid, class)) in sorted.into_iter().enumerate() {
            let id = idx as i64 + 1;
            entities.insert(id, guid);
            let mut instance = ActorInstance::new(&class);
            instance.vars.insert("entity".into(), Value::Int(id));
            states.insert(guid, ActorState { class, instance });
        }

        let mut sim = Self {
            world,
            bridge,
            bridge3d,
            clips: BTreeMap::new(),
            state_machines: BTreeMap::new(),
            stepper: FixedStep::from_hz(hz),
            actors: states,
            entities,
            input: RuntimeInput::default(),
            prev_down: BTreeSet::new(),
            just_pressed: BTreeSet::new(),
            just_released: BTreeSet::new(),
            bindings: BTreeMap::new(),
            dispatch_queue: VecDeque::new(),
            drained_overlaps: Vec::new(),
            logs: Vec::new(),
            grounded: BTreeMap::new(),
            audio: AudioEngine::new(),
            audio_clips: BTreeMap::new(),
            audio_cmds: Vec::new(),
            audio_started: BTreeSet::new(),
            audio_log: Vec::new(),
            steps: 0,
            prev_positions: HashMap::new(),
            cur_positions: HashMap::new(),
            mods: None,
            terrain: crate::terrain_stream::TerrainStreaming::default(),
            cells: crate::cell_stream::CellStreaming::default(),
            voxels: BTreeMap::new(),
        };

        sim.bridge.sync_from_world(&sim.world);
        sim.bridge3d.sync_from_world(&sim.world); // P11.3 3D bridge
        sim.run_all(&EventKind::BeginPlay);
        sim.drain_dispatch(); // Wave 3: BeginPlay may dispatch custom events.
        sim.capture_positions();
        sim.prev_positions = sim.cur_positions.clone();
        sim
    }

    /// Register a `.inf_anim` clip (with its skeleton) for root motion (P11.3) —
    /// the runtime mirror of `SimSession::register_root_motion_clip`. The player's
    /// level loader seeds these from the loaded assets; a [`RootMotion`] entity
    /// whose clip isn't registered simply skips root motion.
    pub fn register_root_motion_clip(
        &mut self,
        clip_guid: Uuid,
        skeleton: Skeleton,
        clip: AnimClip,
    ) {
        self.clips.insert(clip_guid, RootClip { skeleton, clip });
    }

    /// Register (or replace) the resolvable `.inf_sm` state machines (P11.2) — the
    /// runtime mirror of `SimSession::set_state_machines`. The level loader seeds
    /// these from the loaded assets; an [`AnimStateMachine`] entity whose `sm`
    /// GUID isn't registered simply doesn't step.
    pub fn set_state_machines(&mut self, machines: BTreeMap<Uuid, StateMachine>) {
        self.state_machines = machines;
    }

    /// Register a resolvable `.inf_audio` clip payload by asset GUID (P12.3) — the
    /// runtime mirror of `SimSession::register_audio_clip`. Idempotent.
    pub fn register_audio_clip(&mut self, clip_guid: Uuid, clip: AudioAsset) {
        self.audio_clips.insert(clip_guid, clip);
    }

    /// Seed the resolvable `.inf_audio` payloads in bulk (level loader).
    pub fn set_audio_clips(&mut self, clips: BTreeMap<Uuid, AudioAsset>) {
        self.audio_clips = clips;
    }

    /// Seed the simulation's voxel volumes (P21.2), keyed by entity `Guid`.
    ///
    /// Each [`VoxelData`]'s anchor must already be its **world** anchor — the
    /// asset's own origin plus the volume entity's translation — because that is
    /// what makes `terrain.height_at` answer in world metres. [`resolve_voxel_volumes`]
    /// is the resolver that does it, and every caller should go through it rather
    /// than assembling the map by hand.
    ///
    /// Camera-free by construction: nothing about this map depends on a viewport,
    /// so a Simulate step answers the same whatever the editor camera is doing.
    pub fn set_voxel_volumes(&mut self, volumes: BTreeMap<Uuid, VoxelData>) {
        self.voxels = volumes;
    }

    /// Attach a per-fixed-step mod hook (the WASM mod loader). Ticked each fixed
    /// step after gameplay/physics/anim (P14.5).
    pub fn set_mods(&mut self, mods: Box<dyn ModHook>) {
        self.mods = Some(mods);
    }

    /// The live blueprint `i64 → Guid` entity map (read-only; tests + the mod
    /// adapter map opaque ids to entities through it).
    pub fn entity_map(&self) -> &BTreeMap<i64, Uuid> {
        &self.entities
    }

    /// Install a named-bus mixer on the audio engine (loaded from
    /// `.infinity/mixer.toml`).
    pub fn set_audio_mixer(&mut self, mixer: inf_audio::MixerConfig) {
        self.audio.set_mixer(mixer);
    }

    /// The accumulated audio command stream (P12.3): the deterministic play/stop/
    /// set sequence a headless test asserts against instead of device output.
    pub fn audio_command_log(&self) -> &[AudioCommand] {
        &self.audio_log
    }

    /// The owned world (read-only projection for the renderer / trace).
    pub fn world(&self) -> &EcsWorld {
        &self.world
    }

    /// The owned world, mutable (snapshotting for the trace needs `&mut`).
    pub fn world_mut(&mut self) -> &mut EcsWorld {
        &mut self.world
    }

    /// Total fixed steps advanced.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    // ── terrain streaming (P16.3b2) ───────────────────────────────────────

    /// Attach camera-driven terrain streaming (see [`crate::terrain_stream`]).
    /// Runs one immediate [`sync_sim`](crate::terrain_stream::TerrainStreaming::sync_sim)
    /// so the world is queryable before the first step (`BeginPlay` has already
    /// run by the time a caller reaches here, which is why the attach seam is
    /// separate from `new`).
    pub fn set_terrain_streaming(&mut self, streaming: crate::terrain_stream::TerrainStreaming) {
        self.terrain = streaming;
        self.terrain.sync_sim(&mut self.world);
    }

    /// The streamed terrains (render-resident data + counters). Read by the
    /// render projection and the diagnostics dump.
    pub fn terrain_streaming(&self) -> &crate::terrain_stream::TerrainStreaming {
        &self.terrain
    }

    // ── world-partition cell streaming (P16.5) ────────────────────────────

    /// Attach world-partition cell streaming (see [`crate::cell_stream`]).
    ///
    /// Runs one immediate [`sync_sim`](crate::cell_stream::CellStreaming::sync_sim)
    /// at step 0, so the cells around the streaming sources are already populated
    /// before the first fixed step — the same reason
    /// [`set_terrain_streaming`](Self::set_terrain_streaming) does.
    pub fn set_cell_streaming(&mut self, cells: crate::cell_stream::CellStreaming) {
        self.cells = cells;
        self.cells.sync_sim(&mut self.world, self.steps);
        // A cell activation spawned entities; terrain residency is derived from
        // (some of) them, so re-derive it before anything reads a height.
        self.terrain.sync_sim(&mut self.world);
    }

    /// The cell streamer (residency + counters + the activation trace). Read by
    /// the diagnostics dump, the debug overlay and the gates.
    pub fn cell_streaming(&self) -> &crate::cell_stream::CellStreaming {
        &self.cells
    }

    /// The `terrain.height_at` **host seam**, as the simulation sees it: the exact
    /// function the Blueprint node dispatches to (lowest-`Guid` non-empty
    /// [`Terrain`], its component-resident working set, entity-origin shifted).
    ///
    /// Exposed so the streaming gate can assert sim determinism against the real
    /// seam rather than a re-implementation of it.
    pub fn terrain_height_at(&self, x: f64, z: f64) -> f64 {
        terrain_height_at(&self.world, &self.voxels, x, z)
    }

    /// The level clock as the simulation sees it (UTC seconds since midnight);
    /// `0` when the level has no clock. The read half of the `sky.*` host seam,
    /// exposed so a gate can trace the clock without naming `bevy_ecs`.
    pub fn time_of_day_seconds(&self) -> f64 {
        inf_ecs::sky::time_of_day_seconds(&self.world)
    }

    /// The unit direction **toward the sun** the projected scene would carry, or
    /// the renderer's retired-constant default when the level has no clock. This
    /// is the value the PIE-vs-shipping sun trace compares, so it must come from
    /// the same `inf_ecs::sky` resolution the projectors use — not a second copy.
    pub fn sun_direction(&self) -> glam::DVec3 {
        inf_ecs::sky::resolve_sky(&self.world)
            .map(|s| s.sun)
            .unwrap_or_else(|| inf_render::DEFAULT_SUN_DIR.normalize().as_dvec3())
    }

    /// **The render-sync point.** Advance every streamed terrain's camera-driven
    /// cut, exactly once per frame, immediately before the scene projection.
    ///
    /// Deliberately *not* called from [`fixed_step`](Self::fixed_step): the sim
    /// must never observe the result. The windowed loop and the headless harness
    /// call it at the same place in their frame so a scripted camera path yields
    /// the same resident-set trace either way.
    pub fn sync_render_terrain(&mut self, camera_world: DVec3) {
        self.terrain.sync_render(camera_world);
    }

    /// Render interpolation factor in `[0, 1)` (how far into the next step the
    /// frame accumulator sits).
    pub fn alpha(&self) -> f64 {
        self.stepper.alpha()
    }

    /// The `debug.print` log accumulated so far.
    /// The 3D physics bridge, read-only (P20.2) — the seam a gate, a debug view
    /// or a tool reads water probes and swim latches through without being able
    /// to perturb the step that produced them.
    pub fn bridge3d(&self) -> &PhysicsBridge3D {
        &self.bridge3d
    }

    pub fn logs(&self) -> &[String] {
        &self.logs
    }

    /// A live member variable of an actor (tests / debug HUD).
    pub fn actor_var(&self, guid: Uuid, name: &str) -> Option<&Value> {
        self.actors.get(&guid).and_then(|a| a.instance.get(name))
    }

    /// Whether an actor was grounded at its last `move_and_slide`.
    pub fn is_grounded(&self, guid: Uuid) -> bool {
        self.grounded.get(&guid).copied().unwrap_or(false)
    }

    /// The trigger-volume (sensor) overlaps drained during the most recent fixed
    /// step (Wave 3) — the shipped mirror of `SimSession::drained_overlaps`. Each
    /// a canonical `a < b` entity-`Guid` pair + Begin/End phase, sorted ascending;
    /// rebuilt every step. The seam trigger-volume gameplay + the parity tests
    /// consume.
    pub fn drained_overlaps(&self) -> &[OverlapEvent] {
        &self.drained_overlaps
    }

    /// Interpolated world translation of `guid` for rendering: the previous and
    /// current fixed-step positions blended by [`alpha`](Self::alpha). Falls back
    /// to the current position when there is no history.
    pub fn interp_translation(&self, guid: Uuid, alpha: f64) -> Option<DVec3> {
        let cur = *self.cur_positions.get(&guid)?;
        let prev = self.prev_positions.get(&guid).copied().unwrap_or(cur);
        Some(prev.lerp(cur, alpha.clamp(0.0, 1.0)))
    }

    /// The camera focus point: the centroid of the actors' current positions
    /// (a simple follow target for the windowed player). `DVec3::ZERO` with no
    /// actors.
    pub fn camera_focus(&self) -> DVec3 {
        if self.cur_positions.is_empty() {
            return DVec3::ZERO;
        }
        let sum: DVec3 = self.cur_positions.values().copied().sum();
        sum / self.cur_positions.len() as f64
    }

    /// Advance by a frame's elapsed time via the fixed-step accumulator (0..N
    /// fixed steps, spiral-of-death guarded). `input` is this frame's held set.
    /// Returns how many fixed steps ran.
    pub fn run_frame(&mut self, frame_dt: f64, input: RuntimeInput) -> u32 {
        self.set_input(input);
        let n = self.stepper.accumulate(frame_dt);
        for _ in 0..n {
            self.fixed_step();
        }
        n
    }

    /// Run exactly one fixed step with the given input — the deterministic entry
    /// point tests and the headless trace script against.
    pub fn step_once(&mut self, input: RuntimeInput) {
        self.set_input(input);
        self.fixed_step();
    }

    /// bincode of the `Guid`-sorted sim snapshot — the per-step trace unit folded
    /// by the determinism harness (same shape `inf_runtime::replay` hashes).
    pub fn state_bytes(&mut self) -> Vec<u8> {
        let snap = sim_snapshot(&mut self.world);
        bincode::serde::encode_to_vec(&snap, bincode::config::standard())
            .expect("sim snapshot is always encodable")
    }

    // ── internal ──────────────────────────────────────────────────────────

    fn set_input(&mut self, input: RuntimeInput) {
        // MIRROR of `SimSession::set_input` — Wave 3 adds the falling edge.
        self.just_pressed = input.down.difference(&self.prev_down).cloned().collect();
        self.just_released = self.prev_down.difference(&input.down).cloned().collect();
        self.prev_down = input.down.clone();
        self.input = input;
    }

    fn fixed_step(&mut self) {
        let dt = self.stepper.fixed_dt();
        // 0a. World-partition CELL streaming (P16.5). Spawn/despawn the cells the
        //     sim's own `StreamingSource` entities want, in ascending cell order,
        //     BEFORE anything else — including terrain's observer scan, which has
        //     to see the entities a freshly-activated cell brought in. Camera-free
        //     by construction: `sync_sim` has no camera to be given.
        self.cells.sync_sim(&mut self.world, self.steps);
        // 0b. Terrain streaming, SIM side (P16.3b2). Level-0 pages around the sim's
        //    own entities, loaded synchronously in key order BEFORE anything in
        //    this step can query a height. Camera-free by construction — see
        //    `crate::terrain_stream` for why that separation is structural.
        self.terrain.sync_sim(&mut self.world);
        // ── P17.1 time of day ── advance the level clock ONCE per fixed step,
        //    before anything reads it, so blueprints, the projected sun, shadows,
        //    GI and audio all observe one consistent clock for the step. Pure IEEE
        //    add/mul/floor over the sim's own state (`inf_ecs::sky`), hence
        //    bit-identical across runs and across processes — which is what makes
        //    the sun-direction trace a replay- and PIE-vs-shipping gate. Frozen at
        //    `rate == 0` (the component default), and never called outside a
        //    fixed step, so an idle editor never moves the sun.
        inf_ecs::sky::advance_time_of_day(&mut self.world, dt);
        // The weather blend advances in the same slot, for the same reason
        // (P17.4): everything downstream — the projected clouds, the fog, the
        // precipitation, a Blueprint reading `sky.get_precipitation` — must
        // observe ONE weather state for the step. Inert unless a transition is
        // actually in flight on an enabled weather block.
        inf_ecs::sky::advance_weather(&mut self.world, dt);
        // 1. ECS → physics.
        self.bridge.sync_from_world(&self.world);
        self.bridge3d.sync_from_world(&self.world); // ── P11.3 3D bridge: sync ──
                                                    // ── P20.2 water forces ── buoyancy + hydrodynamic drag, between the sync
                                                    //    and the solver: after the sync because a body must be sampled where it
                                                    //    IS, and before the step because that is the step the forces belong to.
                                                    //    Also arms this step's enter/exit/splash, drained in the collision slot
                                                    //    below so the fixed step has ONE event point. One branch on a level with
                                                    //    no `Buoyancy` component. (MIRROR of `SimSession::fixed_step`.)
        self.bridge3d.apply_water_forces(dt);
        // ── Wave 3 input events ── (MIRROR of SimSession) fire Input(action) edges
        //    BEFORE the Tick pass, then drain any dispatches they queued.
        self.fire_input_events();
        self.drain_dispatch();
        // 2. Blueprint Tick for every actor (Guid order).
        let args: HashMap<String, Value> = [("dt".to_string(), Value::Float(dt))].into();
        self.run_all_with_args(&EventKind::Tick, &args);
        self.drain_dispatch(); // Wave 3: Tick may dispatch custom events.
                               // 3. Solver.
        self.bridge.step(dt);
        self.bridge3d.step(dt); // ── P11.3 3D bridge: step ──
                                // ── Wave 3 collision + overlap drain ── (MIRROR of SimSession) between the
                                //    solver and write-back: fire `Collision` events + collect OverlapEvents.
        self.drain_collisions();
        self.drain_dispatch();
        // 4. Physics → ECS.
        self.bridge.write_back(&mut self.world);
        self.bridge3d.write_back_into(&mut self.world); // ── P11.3 3D bridge: write-back ──
        self.world.propagate();
        // 5. Advance skeletal-animation play-heads (P11.1) — the same order-free,
        //    fixed-`dt` integration the editor Simulate tick runs (preview ==
        //    shipped). ── P11.3 root motion ── snapshot play-heads, advance, apply.
        let prev_ts = self.capture_root_motion_times();
        inf_ecs::anim::advance_anim_players(&mut self.world, dt);
        // ── P11.2 anim state machines ── (adjacent to the P11.3 root-motion apply
        //    above; kept separate). Step each `AnimStateMachine` against its
        //    actor's Blueprint variables.
        self.advance_state_machines(dt);
        self.apply_root_motion(&prev_ts);
        self.world.propagate();
        // ── P11.3 attachments ── entities ride their target's socket, post-anim.
        update_attachments(&mut self.world);
        self.world.propagate();
        // ── P14.5 WASM mods ── tick sandboxed mods against the world (after
        //    gameplay/physics/anim), then propagate their transform edits.
        self.tick_mods(dt);
        // ── P12.3 audio step ── last, observing this step's final transforms
        //    (preview == shipped: the same logic the editor SimSession runs).
        self.audio_step();
        // Roll interpolation history + rising edges.
        std::mem::swap(&mut self.prev_positions, &mut self.cur_positions);
        self.capture_positions();
        self.just_pressed.clear();
        self.steps += 1;
    }

    /// Tick attached WASM mods, then propagate their transform edits so
    /// mod-moved entities reach `GlobalTransform` for rendering/audio (P14.5).
    /// The hook is lifted out during its run so it can borrow the world + entity
    /// map mutably without aliasing `self.mods`.
    fn tick_mods(&mut self, dt: f64) {
        let Some(mut mods) = self.mods.take() else {
            return;
        };
        mods.tick(&mut self.world, &mut self.entities, dt, &self.input);
        self.mods = Some(mods);
        self.world.propagate();
    }

    /// The P12.3 audio step — the shipped mirror of `SimSession::audio_step`.
    fn audio_step(&mut self) {
        let listener = active_listener(&self.world);
        let listener_pos = listener
            .map(|l| l.position)
            .unwrap_or_else(|| self.audio.listener().position);
        if let Some(l) = listener {
            self.audio_cmds.push(AudioCommand::SetListener(l));
        }

        // Autoplay once per not-yet-started `AudioSource`, plus the set of live
        // emitter Guids (for despawn pruning below).
        let mut autoplay: Vec<(Uuid, AudioSource, DVec3)> = Vec::new();
        let mut live: BTreeSet<Uuid> = BTreeSet::new();
        for e in self.world.world().iter_entities() {
            let Some(guid) = e.get::<Guid>().map(|g| g.0) else {
                continue;
            };
            live.insert(guid);
            let Some(src) = e.get::<AudioSource>() else {
                continue;
            };
            if !src.autoplay {
                continue;
            }
            if self.audio_started.contains(&guid) {
                continue;
            }
            let pos = e
                .get::<GlobalTransform>()
                .map(|g| g.translation())
                .or_else(|| e.get::<Transform>().map(|t| t.translation.to_dvec3()))
                .unwrap_or(DVec3::ZERO);
            autoplay.push((guid, src.clone(), pos));
        }
        // Emit in Guid order (the iteration above is archetype order, not Guid
        // order — the deterministic contract is the sort, not the scan).
        autoplay.sort_by_key(|(guid, _, _)| *guid);
        for (guid, src, pos) in autoplay {
            self.audio_started.insert(guid);
            let mut cmd = play_command_for(guid_source_key(guid), &src, src.spatial.then_some(pos));
            if src.occlusion && src.spatial {
                cmd.occlusion_gain = self.occlusion_gain(listener_pos, pos);
            }
            self.audio_cmds.push(AudioCommand::Play(cmd));
        }

        // Despawn → Stop: any started emitter whose entity is gone this step gets a
        // Stop (ascending Guid order — `audio_started` is a BTreeSet) and is
        // forgotten, so a shipped game never leaks a voice per despawned emitter.
        let despawned: Vec<Uuid> = self
            .audio_started
            .iter()
            .filter(|g| !live.contains(*g))
            .copied()
            .collect();
        for guid in despawned {
            self.audio_started.remove(&guid);
            self.audio_cmds.push(AudioCommand::Stop {
                source: guid_source_key(guid),
            });
        }

        // Occlusion for Blueprint-queued Plays (source = actor entity id).
        let mut occ: Vec<(usize, DVec3)> = Vec::new();
        for (i, cmd) in self.audio_cmds.iter().enumerate() {
            if let AudioCommand::Play(p) = cmd {
                if let Some(pos) = p.position {
                    if let Some(guid) = self.entities.get(&(p.source as i64)) {
                        if audio_source_of(&self.world, *guid)
                            .map(|s| s.occlusion)
                            .unwrap_or(false)
                        {
                            occ.push((i, pos));
                        }
                    }
                }
            }
        }
        for (i, emitter) in occ {
            let gain = self.occlusion_gain(listener_pos, emitter);
            if let AudioCommand::Play(p) = &mut self.audio_cmds[i] {
                p.occlusion_gain = gain;
            }
        }

        let cmds = std::mem::take(&mut self.audio_cmds);
        self.audio_log.extend(cmds.iter().cloned());
        let clips = &self.audio_clips;
        self.audio
            .drain(&cmds, &|g| clips.get(&g).and_then(|a| a.decode().ok()));
        // Host-side reap of naturally-finished voices (device bookkeeping only —
        // not sim state, so the command stream above is untouched).
        self.audio.reap();
    }

    /// One occlusion raycast from `listener` toward `emitter` (the shipped mirror
    /// of the editor helper).
    fn occlusion_gain(&mut self, listener: DVec3, emitter: DVec3) -> f64 {
        let delta = emitter - listener;
        let dist = delta.length();
        if dist < 1e-6 {
            return 1.0;
        }
        let dir = delta / dist;
        match self.bridge3d.world_mut().cast_ray(listener, dir, dist) {
            Some(hit) => {
                let hit_dist = (hit.point - listener).length();
                if hit_dist + 1e-3 < dist {
                    OCCLUSION_CUT_LINEAR
                } else {
                    1.0
                }
            }
            None => 1.0,
        }
    }

    /// Step every entity's [`AnimStateMachine`] (P11.2) whose `sm` GUID resolves
    /// in [`state_machines`](Self::state_machines) — the runtime mirror of
    /// `SimSession::advance_state_machines` (preview == shipped). Conditions +
    /// blend params read the entity's actor Blueprint variables; an entity with no
    /// actor gets an empty variable set (params default `0`). Order-independent →
    /// deterministic.
    fn advance_state_machines(&mut self, dt: f64) {
        if self.state_machines.is_empty() {
            return;
        }
        let mut targets: Vec<(Entity, Uuid, Uuid, SmRuntimeState)> = Vec::new();
        {
            let w = self.world.world_mut();
            let mut q = w.query::<(Entity, &Guid, &AnimStateMachine)>();
            for (e, g, asm) in q.iter(w) {
                if let Some(sm_guid) = asm.sm {
                    targets.push((e, g.0, sm_guid, asm.runtime));
                }
            }
        }
        for (entity, guid, sm_guid, rt_state) in targets {
            let Some(machine) = self.state_machines.get(&sm_guid) else {
                continue;
            };
            let vars = self
                .actors
                .get(&guid)
                .map(|a| var_snapshot(&a.instance))
                .unwrap_or_default();
            let mut rt = to_anim_runtime(rt_state);
            {
                let lookup = |name: &str| vars.get(name).copied();
                let ctx = SmContext::new(&lookup);
                rt.advance(machine, &ctx, dt);
            }
            if let Some(mut asm) = self.world.world_mut().get_mut::<AnimStateMachine>(entity) {
                asm.runtime = from_anim_runtime(rt);
            }
        }
    }

    /// Snapshot the play-head `t` of every root-motion-driven playing entity
    /// before the anim advance (P11.3) — the runtime mirror of
    /// `SimSession::capture_root_motion_times`.
    fn capture_root_motion_times(&mut self) -> BTreeMap<Uuid, f64> {
        let mut out = BTreeMap::new();
        let w = self.world.world_mut();
        let mut q = w.query::<(&inf_ecs::Guid, &RootMotion, &AnimPlayer)>();
        for (g, rm, ap) in q.iter(w) {
            if rm.mode == RootMotionMode::ApplyToEntity && ap.playing {
                out.insert(g.0, ap.t);
            }
        }
        out
    }

    /// Apply each root-motion entity's clip root delta to its `Transform` (P11.3)
    /// — the runtime mirror of `SimSession::apply_root_motion`.
    fn apply_root_motion(&mut self, prev_ts: &BTreeMap<Uuid, f64>) {
        if prev_ts.is_empty() {
            return;
        }
        let mut work: Vec<(Uuid, Entity, f64, f64, bool, Uuid, bool)> = Vec::new();
        for (&guid, &prev_t) in prev_ts {
            let Some(entity) = self.world.entity_of(guid) else {
                continue;
            };
            let w = self.world.world();
            let Some(ap) = w.get::<AnimPlayer>(entity) else {
                continue;
            };
            let Some(clip) = ap.clip else {
                continue;
            };
            let has_cc = w.get::<CharacterController3D>(entity).is_some();
            work.push((guid, entity, prev_t, ap.t, ap.looping, clip, has_cc));
        }

        let mut changed = false;
        for (guid, entity, prev_t, cur_t, looping, clip_guid, has_cc) in work {
            let Some(rc) = self.clips.get(&clip_guid) else {
                continue;
            };
            let d = root_delta(&rc.clip, &rc.skeleton, prev_t as f32, cur_t as f32, looping);
            if d.translation == glam::Vec3::ZERO && d.yaw == 0.0 {
                continue;
            }
            let t = self
                .world
                .world()
                .get::<Transform>(entity)
                .copied()
                .unwrap_or(Transform::IDENTITY);
            let yaw_rad = t.rotation.y.to_radians();
            let local = DVec3::new(d.translation.x as f64, 0.0, d.translation.z as f64);
            let world_delta = DQuat::from_rotation_y(yaw_rad) * local;
            let new_yaw_deg = t.rotation.y + d.yaw.to_degrees() as f64;
            let pos = t.translation.to_dvec3();

            let new_pos = if has_cc {
                let mover = build_mover3d(&self.world, guid);
                let exclude = self.bridge3d.collider_of(guid);
                let res =
                    self.bridge3d
                        .world_mut()
                        .move_character(&mover, pos, world_delta, exclude);
                let np = pos + res.translation;
                if let Some(body) = self.bridge3d.body_of(guid) {
                    self.bridge3d.world_mut().set_body_translation(body, np);
                }
                self.grounded.insert(guid, res.grounded);
                np
            } else {
                pos + world_delta
            };

            if let Some(mut tr) = self.world.world_mut().get_mut::<Transform>(entity) {
                tr.translation.x = new_pos.x;
                tr.translation.y = new_pos.y;
                tr.translation.z = new_pos.z;
                tr.rotation.y = new_yaw_deg;
                changed = true;
            }
        }
        if changed {
            self.world.mark_dirty();
        }
    }

    /// Capture every actor's world translation into `cur_positions`.
    fn capture_positions(&mut self) {
        self.cur_positions.clear();
        let guids: Vec<Uuid> = self.actors.keys().copied().collect();
        for guid in guids {
            if let Some(entity) = self.world.entity_of(guid) {
                if let Some(p) = self.world.world_translation(entity) {
                    self.cur_positions.insert(guid, p);
                }
            }
        }
    }

    fn run_all(&mut self, event: &EventKind) {
        let args = HashMap::new();
        self.run_all_with_args(event, &args);
    }

    /// Fire `event` on every actor in `Guid` order (each via [`run_on_guid`]) —
    /// MIRROR of `SimSession::run_all_with_args`.
    fn run_all_with_args(&mut self, event: &EventKind, args: &HashMap<String, Value>) {
        let guids: Vec<Uuid> = self.actors.keys().copied().collect();
        for guid in guids {
            self.run_on_guid(guid, event, args);
        }
    }

    /// Fire `event` on the single actor `guid` through a fresh [`RuntimeHost`] —
    /// MIRROR of `SimSession::run_on_guid`. The actor's `entity` id is threaded in
    /// as `current_entity` so `event::bind` knows the calling listener.
    fn run_on_guid(&mut self, guid: Uuid, event: &EventKind, args: &HashMap<String, Value>) {
        let Some(mut state) = self.actors.remove(&guid) else {
            return;
        };
        let current_entity = match state.instance.get("entity") {
            Some(Value::Int(i)) => *i,
            _ => 0,
        };
        {
            let mut host = RuntimeHost {
                bridge: &mut self.bridge,
                bridge3d: &mut self.bridge3d,
                world: &mut self.world,
                input: &self.input,
                just_pressed: &self.just_pressed,
                entities: &self.entities,
                logs: &mut self.logs,
                grounded: &mut self.grounded,
                audio_cmds: &mut self.audio_cmds,
                current_entity,
                bindings: &mut self.bindings,
                dispatch_queue: &mut self.dispatch_queue,
                dt: self.stepper.fixed_dt(),
                voxels: &self.voxels,
            };
            if let Err(e) = run_event(
                &state.class,
                &mut state.instance,
                event,
                args,
                &mut host,
                &InterpDebug::default(),
            ) {
                self.logs.push(format!("{}: {e}", event.key()));
            }
        }
        self.actors.insert(guid, state);
    }

    /// Fire `event` on whatever actor owns blueprint entity id `entity_id`, if any
    /// — MIRROR of `SimSession::fire_on_entity`.
    fn fire_on_entity(&mut self, entity_id: i64, event: &EventKind, args: &HashMap<String, Value>) {
        if let Some(&guid) = self.entities.get(&entity_id) {
            self.run_on_guid(guid, event, args);
        }
    }

    /// The blueprint `i64` entity id assigned to `guid`, if it is a mapped entity
    /// — MIRROR of `SimSession::entity_id_of`.
    fn entity_id_of(&self, guid: Uuid) -> Option<i64> {
        self.entities
            .iter()
            .find(|(_, g)| **g == guid)
            .map(|(id, _)| *id)
    }

    /// Fire this step's `Input(action)` events (Wave 3) — MIRROR of
    /// `SimSession::fire_input_events`: presses first (ascending), then releases.
    fn fire_input_events(&mut self) {
        if self.just_pressed.is_empty() && self.just_released.is_empty() {
            return;
        }
        let pressed: Vec<String> = self.just_pressed.iter().cloned().collect();
        let released: Vec<String> = self.just_released.iter().cloned().collect();
        for action in pressed {
            let args: HashMap<String, Value> = [("pressed".to_string(), Value::Bool(true))].into();
            self.run_all_with_args(&EventKind::Input(action), &args);
        }
        for action in released {
            let args: HashMap<String, Value> = [("pressed".to_string(), Value::Bool(false))].into();
            self.run_all_with_args(&EventKind::Input(action), &args);
        }
    }

    /// Drain this step's 2D then 3D contact events into Blueprint `Collision`
    /// events + the sensor `drained_overlaps` list — MIRROR of
    /// `SimSession::drain_collisions` (preview == shipped).
    fn drain_collisions(&mut self) {
        self.drained_overlaps.clear();
        let events2d = self.bridge.world_mut().drain_contact_events();
        let events3d = self.bridge3d.world_mut().drain_contact_events();
        let mut resolved: Vec<(Uuid, Uuid, ContactPhase, bool)> = Vec::new();
        for ev in &events2d {
            if let (Some(a), Some(b)) = (
                self.bridge.guid_of_collider(ev.collider_a),
                self.bridge.guid_of_collider(ev.collider_b),
            ) {
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                resolved.push((lo, hi, ev.phase, ev.sensor));
            }
        }
        for ev in &events3d {
            if let (Some(a), Some(b)) = (
                self.bridge3d.guid_of_collider(ev.collider_a),
                self.bridge3d.guid_of_collider(ev.collider_b),
            ) {
                let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                resolved.push((lo, hi, ev.phase, ev.sensor));
            }
        }

        // (b) Sensor-pair overlaps → the public `drained_overlaps` list.
        for &(a, b, phase, sensor) in &resolved {
            if !sensor {
                continue;
            }
            self.drained_overlaps.push(OverlapEvent {
                a,
                b,
                phase: match phase {
                    ContactPhase::Started => OverlapPhase::Begin,
                    ContactPhase::Stopped => OverlapPhase::End,
                },
            });
        }
        self.drained_overlaps.sort();
        self.drained_overlaps.dedup();

        // (a) Blueprint `Collision` events — Started only, sensors INCLUDED.
        let mut pairs: Vec<(Uuid, Uuid)> = resolved
            .iter()
            .filter(|(_, _, phase, _)| *phase == ContactPhase::Started)
            .map(|&(a, b, _, _)| (a, b))
            .collect();
        pairs.sort();
        pairs.dedup();
        let mut fires: Vec<(Uuid, i64)> = Vec::new();
        for (a, b) in pairs {
            if self.actors.contains_key(&a) {
                fires.push((a, self.entity_id_of(b).unwrap_or(0)));
            }
            if self.actors.contains_key(&b) {
                fires.push((b, self.entity_id_of(a).unwrap_or(0)));
            }
        }
        for (guid, other_id) in fires {
            let args: HashMap<String, Value> = [("other".to_string(), Value::Int(other_id))].into();
            self.run_on_guid(guid, &EventKind::Collision, &args);
        }
        self.drain_water_events();
    }

    /// Fire this step's water crossings on their actors (P20.2) — the MIRROR of
    /// `SimSession::drain_water_events`.
    ///
    /// Drained in the **collision slot**, and that is the design: a crossing is
    /// sensed from the same pre-step poses a contact is, so giving water its own
    /// event point would put two different "when did this happen" answers in one
    /// fixed step. The bridge already produced them in body-`Guid` order with
    /// `Enter`/`Exit` before the `Splash` that accompanies it, so this loop adds
    /// no ordering of its own.
    ///
    /// **The audio hook is the `audio.*` kit called from these handlers**, not a
    /// new command type: the P12.3 doctrine is that the audio stream is a pure
    /// function of sim state, water events *are* sim state, and a `Play` queued
    /// from an `On Splash` handler therefore lands in the command queue in this
    /// same deterministic order. Nothing else is needed, and inventing a
    /// `AudioCommand::Splash` would have made the engine pick the sound.
    fn drain_water_events(&mut self) {
        let events = self.bridge3d.drain_water_events();
        if events.is_empty() {
            return;
        }
        for ev in events {
            if !self.actors.contains_key(&ev.body) {
                continue;
            }
            let kind = match ev.kind {
                WaterEventKind3D::Enter => EventKind::WaterEnter,
                WaterEventKind3D::Exit => EventKind::WaterExit,
                WaterEventKind3D::Splash => EventKind::WaterSplash,
            };
            let args: HashMap<String, Value> = [
                (
                    "water".to_string(),
                    Value::Int(self.entity_id_of(ev.water).unwrap_or(0)),
                ),
                ("speed".to_string(), Value::Float(ev.speed_m_s)),
            ]
            .into();
            self.run_on_guid(ev.body, &kind, &args);
        }
    }

    /// Drain the FIFO dispatch queue (Wave 3) — MIRROR of
    /// `SimSession::drain_dispatch`: fire `Custom(name)` on the target then each
    /// bound listener's `Custom(handler)`; cap at [`DISPATCH_ROUND_CAP`].
    fn drain_dispatch(&mut self) {
        let mut rounds = 0u32;
        loop {
            if self.dispatch_queue.is_empty() {
                break;
            }
            if rounds >= DISPATCH_ROUND_CAP {
                self.logs.push(format!(
                    "event dispatch cap ({DISPATCH_ROUND_CAP}) exceeded; dropping {} pending",
                    self.dispatch_queue.len()
                ));
                self.dispatch_queue.clear();
                break;
            }
            let (target, name) = self.dispatch_queue.pop_front().unwrap();
            rounds += 1;
            let mut fires: Vec<(i64, String)> = vec![(target, name.clone())];
            if let Some(listeners) = self.bindings.get(&(target, name)) {
                for (listener, handler) in listeners {
                    fires.push((*listener, handler.clone()));
                }
            }
            for (entity_id, ev_name) in fires {
                let args = HashMap::new();
                self.fire_on_entity(entity_id, &EventKind::Custom(ev_name), &args);
            }
        }
    }
}

/// The engine [`Host`] a runtime tick runs against: routes `input.*` to the
/// held-action set, `debug.print` to the log, and exposes the physics world via
/// [`Host::physics`]. A line-for-line analogue of the editor `SimHost`.
struct RuntimeHost<'a> {
    bridge: &'a mut PhysicsBridge2D,
    /// The 3D physics bridge (P11.3), powering `physics3d.*` nodes.
    bridge3d: &'a mut PhysicsBridge3D,
    world: &'a mut EcsWorld,
    input: &'a RuntimeInput,
    just_pressed: &'a BTreeSet<String>,
    entities: &'a BTreeMap<i64, Uuid>,
    logs: &'a mut Vec<String>,
    grounded: &'a mut BTreeMap<Uuid, bool>,
    /// The P12.3 audio command sink: `audio.*` nodes enqueue here.
    audio_cmds: &'a mut Vec<AudioCommand>,
    /// The blueprint entity id of the actor currently running (Wave 3 MIRROR of
    /// `SimHost::current_entity`): the listener `event::bind`/`unbind` register.
    current_entity: i64,
    /// The sim's event-dispatcher bindings (Wave 3 MIRROR of `SimHost::bindings`).
    bindings: &'a mut BTreeMap<(i64, String), BTreeMap<i64, String>>,
    /// The sim's FIFO dispatch queue (Wave 3 MIRROR of `SimHost::dispatch_queue`).
    dispatch_queue: &'a mut VecDeque<(i64, String)>,
    /// The fixed timestep, seconds (P20.2). The swim transform is expressed as a
    /// *velocity* — a speed cap and a buoyancy-balance rate — while
    /// `move_and_slide` speaks displacement, so the host needs the step it is
    /// inside. Passed explicitly rather than read back off the physics world, so
    /// the number is the sim's own rather than whatever the solver was last
    /// stepped with.
    dt: f64,
    /// The simulation's own voxel volumes (P21.2), read by `terrain.height_at`.
    /// Borrowed from `RuntimeSim::voxels` — never from a render store, which is
    /// what keeps a camera out of a fixed step's answers.
    voxels: &'a BTreeMap<Uuid, VoxelData>,
}

use inf_ecs::components::WeatherPreset;

impl Host for RuntimeHost<'_> {
    fn call(&mut self, path: &[String], args: &[Value]) -> Result<Value, RunError> {
        match (
            path.first().map(String::as_str),
            path.get(1).map(String::as_str),
        ) {
            (Some("input"), Some("is_down")) => {
                Ok(Value::Bool(self.input.is_down(&arg_str(args, 0))))
            }
            (Some("input"), Some("just_pressed")) => {
                Ok(Value::Bool(self.just_pressed.contains(&arg_str(args, 0))))
            }
            // ── Wave 3 event dispatchers ── (MIRROR of SimHost) `dispatch.*` nodes.
            (Some("event"), Some("dispatch")) => {
                self.dispatch_queue
                    .push_back((arg_i64(args, 0), arg_str(args, 1)));
                Ok(Value::Unit)
            }
            (Some("event"), Some("bind")) => {
                let source = arg_i64(args, 0);
                let name = arg_str(args, 1);
                let handler = arg_str(args, 2);
                self.bindings
                    .entry((source, name))
                    .or_default()
                    .insert(self.current_entity, handler);
                Ok(Value::Unit)
            }
            (Some("event"), Some("unbind")) => {
                let source = arg_i64(args, 0);
                let name = arg_str(args, 1);
                let handler = arg_str(args, 2);
                if let Some(listeners) = self.bindings.get_mut(&(source, name)) {
                    if listeners.get(&self.current_entity) == Some(&handler) {
                        listeners.remove(&self.current_entity);
                    }
                }
                Ok(Value::Unit)
            }
            (Some("debug"), Some("print")) => {
                self.logs.push(arg_str(args, 0));
                Ok(Value::Unit)
            }
            // terrain.height_at(x, z) → world height at that XZ (P11.4) — the same
            // seam the editor SimHost exposes (preview == shipped): a 3D character
            // reads it to stay on a heightfield terrain (no physics collider).
            //
            // P21.2: it is the **combined** ground query. The heightfield answers
            // where it is still a heightfield; where a carve has holed it — or
            // where there is no terrain at all — the topmost voxel surface does, so
            // a character walking into a cave mouth gets the cave floor instead of
            // the `None` a holed bilinear cell produces. One Ring-0 rule
            // (`inf_voxel::ground_height_at`), read here and in the editor host.
            (Some("terrain"), Some("height_at")) => Ok(Value::Float(terrain_height_at(
                self.world,
                self.voxels,
                arg_f64(args, 0),
                arg_f64(args, 1),
            ))),
            // sky.* (P17.1) — the level clock, Blueprint-drivable. Four one-line
            // seams over `inf_ecs::sky`, shared verbatim with the editor SimHost
            // so preview == shipped by construction. Units: seconds for the clock,
            // a dimensionless multiplier for the rate.
            (Some("sky"), Some("get_time_of_day")) => {
                Ok(Value::Float(inf_ecs::sky::time_of_day_seconds(self.world)))
            }
            (Some("sky"), Some("set_time_of_day")) => {
                inf_ecs::sky::set_time_of_day_seconds(self.world, arg_f64(args, 0));
                Ok(Value::Unit)
            }
            (Some("sky"), Some("get_rate")) => {
                Ok(Value::Float(inf_ecs::sky::time_of_day_rate(self.world)))
            }
            (Some("sky"), Some("set_rate")) => {
                inf_ecs::sky::set_time_of_day_rate(self.world, arg_f64(args, 0));
                Ok(Value::Unit)
            }
            // sky.* weather (P17.4) — four more one-line seams over
            // `inf_ecs::sky`, shared verbatim by both hosts. The preset arrives
            // as a Str (the `input.is_down` precedent); an unparseable name is a
            // **no-op**, because a typo must never quietly produce a different
            // sky. `blend_seconds` is literal: 0 snaps, negative means "the
            // level's authored blend time".
            (Some("sky"), Some("set_weather")) => {
                if let Some(p) = WeatherPreset::parse(&arg_str(args, 0)) {
                    inf_ecs::sky::set_weather(self.world, p, arg_f64(args, 1));
                }
                Ok(Value::Unit)
            }
            (Some("sky"), Some("get_weather")) => Ok(Value::Str(
                inf_ecs::sky::weather_preset_name(self.world).to_string(),
            )),
            (Some("sky"), Some("get_precipitation")) => Ok(Value::Float(
                inf_ecs::sky::weather_precipitation(self.world),
            )),
            (Some("sky"), Some("get_wind_speed")) => {
                Ok(Value::Float(inf_ecs::sky::weather_wind_speed(self.world)))
            }
            // water.* (P20.2) — three pure queries against the **fixed step's own**
            // water index, shared verbatim with the editor SimHost so preview ==
            // shipped by construction. They read `inf_water`'s height query, the
            // same evaluator the buoyancy force used this step and the same one the
            // renderer draws — never render state, and never a camera.
            //
            // `surface_height` answers `0.0` where there is no water: the IR has no
            // optional Float, and the `terrain.height_at` precedent is a plain
            // default rather than a sentinel (0 is a plausible sea level).
            //
            // An id that names no entity answers "dry" rather than erroring: a
            // query is not an action, and a blueprint asking whether a despawned
            // actor is wet should get `false`, not a failed handler.
            (Some("water"), Some("is_in_water")) => Ok(Value::Bool(
                self.guid_of(arg_i64(args, 0))
                    .ok()
                    .and_then(|g| self.bridge3d.water_probe(g))
                    .is_some_and(|p| p.depth_m > 0.0),
            )),
            (Some("water"), Some("submerged_fraction")) => Ok(Value::Float(
                self.guid_of(arg_i64(args, 0))
                    .ok()
                    .and_then(|g| self.bridge3d.water_probe(g))
                    .map(|p| p.fraction)
                    .unwrap_or(0.0),
            )),
            (Some("water"), Some("surface_height")) => Ok(Value::Float(
                self.bridge3d
                    .water_surface_height(arg_f64(args, 0), arg_f64(args, 1))
                    .unwrap_or(0.0),
            )),
            // Unknown engine call: log it (matching the editor host) so a
            // partially-authored blueprint still runs rather than aborting.
            _ => {
                self.logs.push(path.join("::"));
                Ok(Value::Unit)
            }
        }
    }

    fn physics(&mut self) -> Option<&mut dyn Physics2dHost> {
        Some(self)
    }

    fn physics3d(&mut self) -> Option<&mut dyn Physics3dHost> {
        Some(self)
    }

    fn audio(&mut self) -> Option<&mut dyn AudioHost> {
        Some(self)
    }
}

/// `audio.*` nodes enqueue commands (P12.3) — the shipped mirror of the editor
/// `SimHost`'s `AudioHost` impl.
impl AudioHost for RuntimeHost<'_> {
    fn play(&mut self, entity: i64) -> Result<(), String> {
        let cmd = self.audio_play_command(entity)?;
        self.audio_cmds.push(AudioCommand::Play(cmd));
        Ok(())
    }
    fn stop(&mut self, entity: i64) -> Result<(), String> {
        self.audio_cmds.push(AudioCommand::Stop {
            source: entity as u64,
        });
        Ok(())
    }
    fn set_volume(&mut self, entity: i64, volume: f64) -> Result<(), String> {
        self.audio_cmds.push(AudioCommand::SetVolume {
            source: entity as u64,
            volume,
        });
        Ok(())
    }
    fn set_pitch(&mut self, entity: i64, pitch: f64) -> Result<(), String> {
        self.audio_cmds.push(AudioCommand::SetPitch {
            source: entity as u64,
            pitch,
        });
        Ok(())
    }
}

impl RuntimeHost<'_> {
    /// Build a [`PlayCommand`] from an entity's [`AudioSource`] (+ world pose).
    fn audio_play_command(&self, entity: i64) -> Result<PlayCommand, String> {
        let guid = self.guid_of(entity)?;
        let src = audio_source_of(self.world, guid)
            .ok_or_else(|| format!("entity {entity} has no AudioSource"))?;
        let pos = emitter_position(self.world, guid);
        Ok(play_command_for(
            entity as u64,
            &src,
            src.spatial.then_some(pos),
        ))
    }

    fn guid_of(&self, entity: i64) -> Result<Uuid, String> {
        self.entities
            .get(&entity)
            .copied()
            .ok_or_else(|| format!("no entity for id {entity}"))
    }

    /// Build a [`CharacterMover2D`] from an entity's `CharacterController2D` +
    /// `Collider2D`, defaulting to an upright capsule when absent.
    fn mover_for(&self, guid: Uuid) -> CharacterMover2D {
        let default_shape = ColliderShape2D::Capsule {
            half_height: 0.5,
            radius: 0.25,
        };
        let Some(entity) = self.world.entity_of(guid) else {
            return CharacterMover2D::new(default_shape);
        };
        let w = self.world.world();
        let shape = w
            .get::<Collider2D>(entity)
            .map(collider_shape)
            .unwrap_or(default_shape);
        let cc = w.get::<CharacterController2D>(entity).copied();
        let mut mover = CharacterMover2D::new(shape).up(DVec2::Y).slide(true);
        if let Some(cc) = cc {
            mover = mover
                .offset(cc.offset.max(1e-4))
                .max_slope_climb_angle(cc.max_slope_deg.to_radians())
                .snap_to_ground(if cc.snap_to_ground > 0.0 {
                    Some(cc.snap_to_ground)
                } else {
                    None
                });
        } else {
            mover = mover.offset(0.02);
        }
        mover
    }
}

impl Physics2dHost for RuntimeHost<'_> {
    fn move_and_slide(&mut self, entity: i64, motion: [f64; 2]) -> Result<MoveResult2d, String> {
        let guid = self.guid_of(entity)?;
        let body = self
            .bridge
            .body_of(guid)
            .ok_or_else(|| format!("entity {entity} has no physics body"))?;
        let pos = self
            .bridge
            .world()
            .body_translation(body)
            .ok_or("body vanished")?;
        let exclude = self.bridge.collider_of(guid);
        let mover = self.mover_for(guid);
        let result = self.bridge.world_mut().move_character(
            &mover,
            pos,
            DVec2::new(motion[0], motion[1]),
            exclude,
        );
        let new_pos = pos + result.translation;
        self.bridge.world_mut().set_body_translation(body, new_pos);
        if let Some(entity) = self.world.entity_of(guid) {
            if let Some(mut t) = self.world.world_mut().get_mut::<Transform>(entity) {
                t.translation.x = new_pos.x;
                t.translation.y = new_pos.y;
            }
            self.world.mark_dirty();
        }
        self.grounded.insert(guid, result.grounded);
        Ok(MoveResult2d {
            applied: result.translation.into(),
            grounded: result.grounded,
        })
    }

    fn is_grounded(&mut self, entity: i64) -> Result<bool, String> {
        let guid = self.guid_of(entity)?;
        let body = self
            .bridge
            .body_of(guid)
            .ok_or_else(|| format!("entity {entity} has no physics body"))?;
        let pos = self
            .bridge
            .world()
            .body_translation(body)
            .ok_or("body vanished")?;
        let exclude = self.bridge.collider_of(guid);
        let mover = self.mover_for(guid);
        let probe = self
            .bridge
            .world_mut()
            .move_character(&mover, pos, DVec2::ZERO, exclude);
        Ok(probe.grounded)
    }

    fn raycast(
        &mut self,
        origin: [f64; 2],
        dir: [f64; 2],
        max: f64,
    ) -> Result<Option<RayHit2d>, String> {
        let hit = self.bridge.world_mut().cast_ray(
            DVec2::new(origin[0], origin[1]),
            DVec2::new(dir[0], dir[1]),
            max,
        );
        Ok(hit.map(|h| RayHit2d {
            point: h.point.into(),
            normal: h.normal.into(),
        }))
    }

    fn set_velocity(&mut self, entity: i64, v: [f64; 2]) -> Result<(), String> {
        let guid = self.guid_of(entity)?;
        let body = self
            .bridge
            .body_of(guid)
            .ok_or_else(|| format!("entity {entity} has no physics body"))?;
        self.bridge
            .world_mut()
            .set_body_linvel(body, DVec2::new(v[0], v[1]));
        Ok(())
    }

    fn get_velocity(&mut self, entity: i64) -> Result<[f64; 2], String> {
        let guid = self.guid_of(entity)?;
        let body = self
            .bridge
            .body_of(guid)
            .ok_or_else(|| format!("entity {entity} has no physics body"))?;
        Ok(self
            .bridge
            .world()
            .body_linvel(body)
            .unwrap_or(DVec2::ZERO)
            .into())
    }

    fn apply_impulse(&mut self, entity: i64, v: [f64; 2]) -> Result<(), String> {
        let guid = self.guid_of(entity)?;
        let body = self
            .bridge
            .body_of(guid)
            .ok_or_else(|| format!("entity {entity} has no physics body"))?;
        self.bridge
            .world_mut()
            .apply_impulse(body, DVec2::new(v[0], v[1]));
        Ok(())
    }
}

impl Physics3dHost for RuntimeHost<'_> {
    fn move_and_slide(&mut self, entity: i64, motion: [f64; 3]) -> Result<MoveResult3d, String> {
        let guid = self.guid_of(entity)?;
        let body = self
            .bridge3d
            .body_of(guid)
            .ok_or_else(|| format!("entity {entity} has no physics body"))?;
        let pos = self
            .bridge3d
            .world()
            .body_translation(body)
            .ok_or("body vanished")?;
        let exclude = self.bridge3d.collider_of(guid);
        let mover = build_mover3d(self.world, guid);
        // ── P20.2 swim mode ── the mover HONOURS the water: a character deep
        //    enough to swim has its free-fall discarded, its vertical motion
        //    replaced by a buoyancy balance and its horizontal speed capped. The
        //    latch and the transform both live in `inf_physics::d3::water`, so
        //    this host and the editor's run the same thresholds rather than two
        //    copies of them. Inert — literally the identity — when not swimming.
        let dt = self.dt;
        self.bridge3d.update_swim(guid);
        let motion =
            self.bridge3d
                .apply_swim_motion(guid, DVec3::new(motion[0], motion[1], motion[2]), dt);
        let result = self
            .bridge3d
            .world_mut()
            .move_character(&mover, pos, motion, exclude);
        let new_pos = pos + result.translation;
        self.bridge3d
            .world_mut()
            .set_body_translation(body, new_pos);
        if let Some(entity) = self.world.entity_of(guid) {
            if let Some(mut t) = self.world.world_mut().get_mut::<Transform>(entity) {
                t.translation.x = new_pos.x;
                t.translation.y = new_pos.y;
                t.translation.z = new_pos.z;
            }
            self.world.mark_dirty();
        }
        self.grounded.insert(guid, result.grounded);
        Ok(MoveResult3d {
            applied: result.translation.into(),
            grounded: result.grounded,
        })
    }

    fn is_grounded(&mut self, entity: i64) -> Result<bool, String> {
        let guid = self.guid_of(entity)?;
        let body = self
            .bridge3d
            .body_of(guid)
            .ok_or_else(|| format!("entity {entity} has no physics body"))?;
        let pos = self
            .bridge3d
            .world()
            .body_translation(body)
            .ok_or("body vanished")?;
        let exclude = self.bridge3d.collider_of(guid);
        let mover = build_mover3d(self.world, guid);
        let probe = self
            .bridge3d
            .world_mut()
            .move_character(&mover, pos, DVec3::ZERO, exclude);
        Ok(probe.grounded)
    }

    fn raycast(
        &mut self,
        origin: [f64; 3],
        dir: [f64; 3],
        max: f64,
    ) -> Result<Option<RayHit3d>, String> {
        let hit = self.bridge3d.world_mut().cast_ray(
            DVec3::new(origin[0], origin[1], origin[2]),
            DVec3::new(dir[0], dir[1], dir[2]),
            max,
        );
        Ok(hit.map(|h| RayHit3d {
            point: h.point.into(),
            normal: h.normal.into(),
        }))
    }

    fn set_velocity(&mut self, entity: i64, v: [f64; 3]) -> Result<(), String> {
        let guid = self.guid_of(entity)?;
        let body = self
            .bridge3d
            .body_of(guid)
            .ok_or_else(|| format!("entity {entity} has no physics body"))?;
        self.bridge3d
            .world_mut()
            .set_body_linvel(body, DVec3::new(v[0], v[1], v[2]));
        Ok(())
    }

    fn get_velocity(&mut self, entity: i64) -> Result<[f64; 3], String> {
        let guid = self.guid_of(entity)?;
        let body = self
            .bridge3d
            .body_of(guid)
            .ok_or_else(|| format!("entity {entity} has no physics body"))?;
        Ok(self
            .bridge3d
            .world()
            .body_linvel(body)
            .unwrap_or(DVec3::ZERO)
            .into())
    }

    fn apply_impulse(&mut self, entity: i64, v: [f64; 3]) -> Result<(), String> {
        let guid = self.guid_of(entity)?;
        let body = self
            .bridge3d
            .body_of(guid)
            .ok_or_else(|| format!("entity {entity} has no physics body"))?;
        self.bridge3d
            .world_mut()
            .apply_impulse(body, DVec3::new(v[0], v[1], v[2]));
        Ok(())
    }
}

/// Build a [`CharacterMover3D`] from an entity's `CharacterController3D` +
/// `Collider3D` (the runtime mirror of the editor's `build_mover3d`). Shared by
/// the `physics3d.move_and_slide` host path and the root-motion applier.
fn build_mover3d(world: &EcsWorld, guid: Uuid) -> CharacterMover3D {
    let default_shape = ColliderShape3D::Capsule {
        half_height: 0.5,
        radius: 0.25,
    };
    let Some(entity) = world.entity_of(guid) else {
        return CharacterMover3D::new(default_shape);
    };
    let w = world.world();
    let shape = w
        .get::<Collider3D>(entity)
        .map(collider_shape3d)
        .unwrap_or(default_shape);
    let cc = w.get::<CharacterController3D>(entity).copied();
    let mut mover = CharacterMover3D::new(shape).up(DVec3::Y).slide(true);
    if let Some(cc) = cc {
        mover = mover
            .offset(cc.offset.max(1e-4))
            .max_slope_climb_angle(cc.max_slope_deg.to_radians())
            .snap_to_ground(if cc.snap_to_ground > 0.0 {
                Some(cc.snap_to_ground)
            } else {
                None
            });
    } else {
        mover = mover.offset(0.02);
    }
    mover
}

/// A `Collider3D` component's shape as the 3D physics-facade shape.
fn collider_shape3d(c: &Collider3D) -> ColliderShape3D {
    match c.shape_kind {
        ColliderShape3DKind::Box => ColliderShape3D::Box {
            half_extents: c.half_extents.to_dvec3(),
        },
        ColliderShape3DKind::Sphere => ColliderShape3D::Sphere { radius: c.radius },
        ColliderShape3DKind::Capsule => ColliderShape3D::Capsule {
            half_height: c.half_extents.y,
            radius: c.radius,
        },
    }
}

/// A `Collider2D` component's shape as the physics-facade shape.
fn collider_shape(c: &Collider2D) -> ColliderShape2D {
    match c.shape_kind {
        ColliderShape2DKind::Box => ColliderShape2D::Box {
            half_width: c.half_extents.x,
            half_height: c.half_extents.y,
        },
        ColliderShape2DKind::Circle => ColliderShape2D::Circle { radius: c.radius },
        ColliderShape2DKind::Capsule => ColliderShape2D::Capsule {
            half_height: c.half_extents.y,
            radius: c.radius,
        },
    }
}

fn arg_str(args: &[Value], i: usize) -> String {
    args.get(i)
        .and_then(|v| match v {
            Value::Str(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// Coerce a positional blueprint arg to `i64` (`Float` truncates; else `0`) — the
/// Wave 3 `event::*` entity-id coercion, MIRROR of `simulate::arg_i64`.
fn arg_i64(args: &[Value], i: usize) -> i64 {
    match args.get(i) {
        Some(Value::Int(n)) => *n,
        Some(Value::Float(f)) => *f as i64,
        _ => 0,
    }
}

/// Coerce a positional blueprint arg to `f64` (`Int` widens; else `0.0`).
fn arg_f64(args: &[Value], i: usize) -> f64 {
    match args.get(i) {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(n)) => *n as f64,
        _ => 0.0,
    }
}

/// Sample the world's terrain height at world `(x, z)` — the `terrain.height_at`
/// host seam (P11.4), byte-for-byte the editor `SimHost`'s (preview == shipped).
/// Uses the lowest-`Guid` non-empty [`Terrain`]; `0.0` with no terrain.
/// A stable audio source key for a scene-placed emitter (the `Guid`'s low bits) —
/// the shipped mirror of the editor helper.
fn guid_source_key(guid: Uuid) -> u64 {
    guid.as_u128() as u64
}

/// Read a clone of an entity's [`AudioSource`] by `Guid`.
fn audio_source_of(world: &EcsWorld, guid: Uuid) -> Option<AudioSource> {
    let e = world.entity_of(guid)?;
    world.world().get::<AudioSource>(e).cloned()
}

/// An entity's world-space emitter position (global transform, else local, else 0).
fn emitter_position(world: &EcsWorld, guid: Uuid) -> DVec3 {
    let Some(e) = world.entity_of(guid) else {
        return DVec3::ZERO;
    };
    world
        .world()
        .get::<GlobalTransform>(e)
        .map(|g| g.translation())
        .or_else(|| {
            world
                .world()
                .get::<Transform>(e)
                .map(|t| t.translation.to_dvec3())
        })
        .unwrap_or(DVec3::ZERO)
}

/// The first **active** [`AudioListener`] (Guid order), posed at the entity's
/// world position (default orientation — a documented P12.3 follow-up).
fn active_listener(world: &EcsWorld) -> Option<Listener> {
    let mut best: Option<(Uuid, DVec3)> = None;
    for e in world.world().iter_entities() {
        let Some(al) = e.get::<AudioListener>() else {
            continue;
        };
        if !al.active {
            continue;
        }
        let guid = e.get::<Guid>().map(|g| g.0).unwrap_or_else(Uuid::nil);
        let pos = e
            .get::<GlobalTransform>()
            .map(|g| g.translation())
            .or_else(|| e.get::<Transform>().map(|t| t.translation.to_dvec3()))
            .unwrap_or(DVec3::ZERO);
        if best.as_ref().map(|(g, _)| guid < *g).unwrap_or(true) {
            best = Some((guid, pos));
        }
    }
    best.map(|(_, position)| Listener {
        position,
        ..Listener::default()
    })
}

/// Build a [`PlayCommand`] from an [`AudioSource`] and an optional world position.
fn play_command_for(source_key: u64, src: &AudioSource, position: Option<DVec3>) -> PlayCommand {
    PlayCommand {
        source: source_key,
        clip: src.clip.unwrap_or_else(Uuid::nil),
        bus: src.bus.clone(),
        volume: src.volume,
        pitch: src.pitch,
        looping: src.looping,
        position,
        attenuation: attenuation_of(src),
        occlusion_gain: 1.0,
    }
}

/// Translate an [`AudioSource`]'s distance model + clamps into an
/// `inf_audio::Attenuation`.
fn attenuation_of(src: &AudioSource) -> Attenuation {
    let model = match src.distance_model {
        DistanceModel::Linear => AttenuationModel::Linear,
        DistanceModel::Inverse => AttenuationModel::Inverse,
        DistanceModel::Exponential => AttenuationModel::Exponential,
    };
    Attenuation {
        model,
        min_distance: src.min_distance,
        max_distance: src.max_distance,
        rolloff: src.rolloff.max(0.0),
    }
}

/// The **ground** height at world `(x, z)` — the `terrain.height_at` host seam.
///
/// Picks the lowest-`Guid` non-empty terrain, remembering only its entity + origin —
/// never a clone of the (multi-MB) heightfield. The component is re-fetched after
/// the scan (all EntityRef borrows released) and sampled in place.
///
/// P21.2: the answer is the **combined** ground query, not the heightfield alone.
/// A carved sample makes `TerrainData::height_at` return `None` through its whole
/// bilinear cell, and a character walking into a cave mouth would otherwise read
/// "no ground" at exactly the step that needs a floor — so the topmost voxel
/// surface answers there instead. The rule itself is one Ring-0 function
/// ([`inf_voxel::ground_height_at`]) precisely so the editor preview and the
/// shipped player cannot disagree about where the floor is.
///
/// `0.0` when nothing answers at all, unchanged from P11.4: the IR has no optional
/// Float, and this seam's documented default is a plain number rather than a
/// sentinel. (The P20.4 "missed picks reject, never y = 0" law is about *picks*,
/// which have a Result to reject into; this is a query, and a query that failed to
/// answer is not an action that failed.)
fn terrain_height_at(world: &EcsWorld, voxels: &BTreeMap<Uuid, VoxelData>, x: f64, z: f64) -> f64 {
    let w = world.world();
    let mut picked: Option<(Uuid, DVec3, Entity)> = None;
    for e in w.iter_entities() {
        let Some(guid) = e.get::<Guid>().map(|g| g.0) else {
            continue;
        };
        let Some(t) = e.get::<Terrain>() else {
            continue;
        };
        if t.data.is_empty() {
            continue;
        }
        let origin = e
            .get::<GlobalTransform>()
            .map(|g| g.translation())
            .or_else(|| e.get::<Transform>().map(|t| t.translation.to_dvec3()))
            .unwrap_or(DVec3::ZERO);
        if picked.as_ref().map(|(g, _, _)| guid < *g).unwrap_or(true) {
            picked = Some((guid, origin, e.id()));
        }
    }
    let picked = picked.and_then(|(_, origin, e)| w.get::<Terrain>(e).map(|t| (origin, t)));
    let (origin, terrain) = match &picked {
        Some((origin, t)) => (*origin, Some(&t.data)),
        // No terrain at all is not "no ground": a level may be nothing but caves,
        // and the voxel half still answers.
        None => (DVec3::ZERO, None),
    };
    inf_voxel::ground_height_at(terrain, origin, voxels, x, z).unwrap_or(0.0)
}

/// Resolve every `.inf_voxel` volume a `VoxelVolume` entity in `world` references,
/// keyed by the **entity's** `Guid` (P21.2). `resolve_voxel` yields the asset's raw
/// payload bytes by GUID (backed by the cooked pack / dev dir). A volume whose
/// bytes don't parse is skipped with a warning. Deterministic (`BTreeMap` / `Guid`
/// order). The caller seeds [`RuntimeSim::set_voxel_volumes`] — the shipped twin of
/// the editor Simulate's resolution, so preview == shipped.
///
/// Keyed by entity rather than by asset because two entities may reference one
/// `.inf_voxel` at two different transforms, and the world anchor `sim_volume`
/// folds in is per-entity. The transform is read **once, here**, which is the
/// honest limitation: a volume whose entity is moved during play keeps the anchor
/// it booted with, exactly as the physics bridge keeps the body it mirrored at
/// load. A moving cave is a P21.3 authoring concern and will re-seed.
pub fn resolve_voxel_volumes<H>(world: &EcsWorld, mut resolve_voxel: H) -> BTreeMap<Uuid, VoxelData>
where
    H: FnMut(Uuid) -> Option<Vec<u8>>,
{
    let w = world.world();
    // Walk once into a sorted list so the *load* order is the level's rather than
    // the ECS archetype layout's, and so no `EntityRef` borrow is held across the
    // byte read.
    let mut wants: Vec<(Uuid, Uuid, DVec3)> = w
        .iter_entities()
        .filter_map(|e| {
            let guid = e.get::<Guid>()?.0;
            let asset = e.get::<inf_ecs::components::VoxelVolume>()?.asset?;
            let translation = e
                .get::<GlobalTransform>()
                .map(|g| g.translation())
                .or_else(|| e.get::<Transform>().map(|t| t.translation.to_dvec3()))
                .unwrap_or(DVec3::ZERO);
            Some((guid, asset, translation))
        })
        .collect();
    wants.sort_by_key(|(g, _, _)| *g);
    let mut out: BTreeMap<Uuid, VoxelData> = BTreeMap::new();
    for (guid, asset, translation) in wants {
        let Some(bytes) = resolve_voxel(asset) else {
            continue;
        };
        match inf_voxel::sim_volume(&bytes, translation) {
            Ok(data) => {
                out.insert(guid, data);
            }
            Err(e) => tracing::warn!("inf-player: bad .inf_voxel {asset}: {e}"),
        }
    }
    out
}

// ── P11.2 state-machine glue: ECS POD ↔ inf-anim runtime + var snapshot ──────
// The editor `SimSession` duplicates these (preview == shipped) — see its docs
// for why `SmRuntimeState` is a POD mirror kept out of `inf-anim`'s dep of `inf-ecs`.

/// Convert the ECS component's transient runtime POD into the anim runtime.
fn to_anim_runtime(s: SmRuntimeState) -> SmRuntime {
    SmRuntime {
        current: s.current,
        prev: s.prev,
        prev_time: s.prev_time,
        fade_t: s.fade_t,
        fade_dur: s.fade_dur,
        state_time: s.state_time,
        started: s.started,
    }
}

/// Convert the advanced anim runtime back into the ECS component POD.
fn from_anim_runtime(r: SmRuntime) -> SmRuntimeState {
    SmRuntimeState {
        current: r.current,
        prev: r.prev,
        prev_time: r.prev_time,
        fade_t: r.fade_t,
        fade_dur: r.fade_dur,
        state_time: r.state_time,
        started: r.started,
    }
}

/// A `name → f64` snapshot of an actor's Blueprint variables for the state
/// machine's condition/param lookups (non-numeric values dropped; `Bool` → 1/0).
fn var_snapshot(instance: &ActorInstance) -> BTreeMap<String, f64> {
    instance
        .vars
        .iter()
        .filter_map(|(k, v)| value_as_f64(v).map(|f| (k.clone(), f)))
        .collect()
}

/// Coerce a Blueprint [`Value`] to `f64` for state-machine conditions/params.
fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Float(f) => Some(*f),
        Value::Int(i) => Some(*i as f64),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}
