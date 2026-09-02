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

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_anim::state_machine::StateMachine;
use inf_anim::{root_delta, AnimClip, Skeleton};
use inf_audio::{
    Attenuation, AttenuationModel, AudioAsset, AudioCommand, AudioEngine, Listener, PlayCommand,
};
use inf_blueprint::interp::{
    AudioHost, MoveResult2d, MoveResult3d, Physics2dHost, Physics3dHost, RayHit2d, RayHit3d,
};
use inf_blueprint::semantics::run_event;
use inf_blueprint::{ActorInstance, BlueprintClass, EventKind, Host, InterpDebug, RunError, Value};
use inf_core::BoundedLog;
use inf_ecs::components::{
    AnimPlayer, AudioListener, AudioSource, CharacterController2D, CharacterController3D,
    Collider2D, ColliderShape2DKind, Destructible, DistanceModel, GlobalTransform, RootMotion,
    RootMotionMode, Terrain, Transform, VoxelVolume,
};
use inf_ecs::{sim_snapshot, update_attachments, EcsWorld, Entity, Guid};
use inf_physics::d3::{
    DebrisBudget, DestroyedEvent, DestructOutcome, FractureAudit, FractureState, WaterEventKind3D,
};
use inf_physics::{
    CharacterMover2D, ColliderShape2D, ContactPhase, PhysicsBridge2D, PhysicsBridge3D, WorldGravity,
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

/// The set of currently-held actions/keys for one tick, plus this tick's
/// resolved **analog axes** (analogue of `SimSession::SimInput`). Rising edges
/// (`just_pressed`) are derived by the [`RuntimeSim`] from the previous tick's
/// set.
///
/// # The axes are P29.3, and they are why the movement component can exist
///
/// Until this wave the only thing that reached a fixed step was a set of action
/// NAMES: analog movement and mouse look had nowhere to arrive, so a character's
/// motion had to be a Blueprint handing `physics3d.move_and_slide` a finished
/// translation. A movement component that owns velocity needs an *intent*, and
/// an intent is analog.
///
/// Delta axes (mouse) arrive here already converted to **rates** by
/// [`InputState::axis_snapshot`](inf_input::InputState::axis_snapshot), so a
/// fixed step integrates `rate × dt` and the same gesture produces the same
/// rotation at any frame rate. `Eq` is gone with the `f32`s, deliberately: an
/// axis is a measurement and measurements compare approximately.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeInput {
    down: BTreeSet<String>,
    axes: BTreeMap<String, f32>,
}

impl RuntimeInput {
    /// An input state with the given actions/keys held and no axes.
    pub fn with_down<I, S>(keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            down: keys.into_iter().map(Into::into).collect(),
            axes: BTreeMap::new(),
        }
    }

    /// Attach this tick's resolved axes (builder-style).
    pub fn with_axes(mut self, axes: BTreeMap<String, f32>) -> Self {
        self.axes = axes;
        self
    }

    /// Whether `key` is currently held.
    pub fn is_down(&self, key: &str) -> bool {
        self.down.contains(key)
    }

    /// This tick's value for `axis`, or `0.0` if nothing bound it.
    pub fn axis(&self, axis: &str) -> f32 {
        self.axes.get(axis).copied().unwrap_or(0.0)
    }

    /// Mark `key` held (builder-style).
    pub fn press(mut self, key: impl Into<String>) -> Self {
        self.down.insert(key.into());
        self
    }

    /// Set one axis (builder-style) — the shape a test uses.
    pub fn axis_at(mut self, axis: impl Into<String>, value: f32) -> Self {
        self.axes.insert(axis.into(), value);
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
    /// Resolvable `.inf_skel` assets keyed by asset GUID (P24.1) — the runtime
    /// mirror of `SimSession::skeletons`. The skeleton a machine-driven
    /// `SkeletalMesh` poses against, plus its authored sockets.
    skeletons: BTreeMap<Uuid, inf_anim::SkeletonAsset>,
    /// Resolvable `.inf_anim` clips a state machine's states **play**, keyed by
    /// asset GUID (P24.1) — the runtime mirror of `SimSession::pose_clips`.
    /// Distinct from [`clips`](Self::clips), which keys a clip *with the skeleton
    /// it animates* because `root_delta` needs both.
    pose_clips: BTreeMap<Uuid, AnimClip>,
    /// Resolvable `.inf_cloth` garments keyed by asset GUID (P24.4) — the runtime
    /// mirror of `SimSession::cloths`. An entity whose `ClothSim.asset` resolves
    /// here simulates; one whose does not keeps its component and simulates
    /// nothing (`inf_ecs::cloth`'s rule 2).
    cloths: BTreeMap<Uuid, inf_anim::ClothAsset>,
    /// Resolvable `.inf_hair` hairstyles keyed by asset GUID (P24.4) - the
    /// runtime mirror of `SimSession::hairs`.
    hairs: BTreeMap<Uuid, inf_anim::HairAsset>,
    /// **How densely this session's hair DRAWS** (P24.4) - a render budget, and
    /// the only tier-derived number on the hair path.
    ///
    /// Deliberately not `ClothSim::quality`'s twin: a substep budget is folded
    /// into `state_bytes` and must therefore be content, while ribbon geometry is
    /// not folded at all and may therefore follow the machine. `inf-ecs`'
    /// `the_detail_draws_differently_and_traces_identically` is what keeps those
    /// two sentences true of the code.
    hair_detail: inf_anim::HairDetail,
    /// **The locomotion camera** (P29.6). A host owns it — it is never a
    /// component and never a resource, which is Ruling 4 kept literally — and it
    /// is stepped at the very end of the fixed step, AFTER everything that could
    /// move the character it follows.
    ///
    /// It is not in `state_bytes` and must never be: the camera reads the sim and
    /// never writes it, and `phase29_gate` asserts both halves.
    camera: inf_ecs::camera::LocomotionCamera,
    /// Whether the camera stepped this run at all — `None` on every level with no
    /// player-controlled character, which is every committed sample before this
    /// wave, so their view path is untouched.
    camera_subject: Option<Uuid>,
    stepper: FixedStep,
    /// Actors keyed by `Guid` (deterministic iteration).
    actors: BTreeMap<Uuid, ActorState>,
    /// Blueprint `i64` entity id → its `Guid`. Seeded in `Guid` order at
    /// entry and **grown by `engine.spawn`** (SCRIPT3).
    entities: BTreeMap<i64, Uuid>,
    /// Guids the running handler destroyed (SCRIPT3), drained into
    /// [`actors`](Self::actors) at the end of `run_on_guid` -- MIRROR of
    /// `SimSession::despawned`.
    despawned: Vec<Uuid>,
    /// Currently-held actions/keys.
    input: RuntimeInput,
    /// Held the previous tick (for rising-edge detection).
    prev_down: BTreeSet<String>,
    /// Rising edges pending this fixed step.
    just_pressed: BTreeSet<String>,
    /// Falling edges pending this fixed step (Wave 3 input release events) —
    /// MIRROR of the editor `SimSession::just_released`.
    just_released: BTreeSet<String>,
    /// **How long each action has been held, in SIM seconds** (I5) — MIRROR of
    /// `SimSession::holds`.
    ///
    /// Advanced by [`step_once`](Self::step_once) with the *fixed* `dt`, never
    /// with a frame time, which is what makes a press duration a function of the
    /// simulation rather than of the machine it ran on. It is what the C key's
    /// click-versus-long-press discrimination reads; see `inf_input::HoldClock`
    /// for why this is a second instance of the type the input layer also runs.
    holds: inf_input::HoldClock,
    /// The click/long-press threshold this session runs with, seconds (I5).
    ///
    /// A player setting, so it lives on the session rather than in a constant —
    /// but it reaches the sim through one setter and is guarded there, because
    /// a threshold arriving as a NaN would make every press classify as nothing
    /// at all.
    press_threshold_s: f64,
    /// **Whether the fixed step is frozen** (I5) — the in-game menu's pause.
    /// See [`set_sim_paused`](Self::set_sim_paused) for why it is here and not
    /// on the host.
    sim_paused: bool,
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
    ///
    /// **Bounded** (Hardening D). This is the shipped player, the only consumers
    /// are [`logs`](Self::logs) and the gates that read it, and the growth rate is
    /// reachable from authored content — a failed event dispatch pushes a
    /// formatted `String` *every tick*. See [`BoundedLog`] for why a ring and not
    /// a drain, and [`dropped_logs`](Self::dropped_logs) for the honest half.
    logs: BoundedLog<String>,
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
    ///
    /// **Bounded** (Hardening D): a listener command is enqueued at least once per
    /// fixed step, so at 60 Hz this grew by ~216 000 commands an hour in the
    /// SHIPPED player for a value only tests read.
    audio_log: BoundedLog<AudioCommand>,
    /// Total fixed steps run.
    steps: u64,
    /// World-space translations one fixed step ago, for render interpolation.
    ///
    /// **`BTreeMap`, and the container is the contract** (Wave C, L6.F1). These
    /// two were the only `HashMap`s in a struct that is otherwise `BTreeMap`
    /// throughout, and [`camera_centroid`] folds `cur_positions` *by iteration*.
    /// `HashMap`'s iteration order is a function of a per-process random seed, so
    /// a non-associative `f64` sum over it makes the committed camera a function
    /// of which process is running — see [`camera_centroid`] for what depends on
    /// that answer.
    prev_positions: BTreeMap<Uuid, DVec3>,
    /// World-space translations at the current fixed step. `BTreeMap` for the
    /// reason spelled out on [`Self::prev_positions`].
    cur_positions: BTreeMap<Uuid, DVec3>,
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
    /// The `.inf_pcg` graphs and the `.inf_biomes` palettes this level binds
    /// (island wave I7b), seeded by [`sim_from_built`](crate::sim_from_built).
    ///
    /// Read by the biome-scatter refresh below and by nothing else in the step.
    /// Empty for a level that binds no vegetation, in which case the refresh is
    /// one comparison and the step is bit-identical to its pre-I7b self.
    pcg: crate::level::PcgContext,
    /// The per-terrain memo the biome-scatter refresh keeps between steps
    /// (island wave I7b) — see [`crate::level::BiomeScatter`].
    ///
    /// **Not sim state that anything reads back.** `Terrain::biome_population`
    /// is `#[serde(skip)]` and reaches no state fold, no replay hash and no
    /// collider; it is what the projector draws. What makes it legal to compute
    /// inside the fixed step is that it is a pure function of the *resident*
    /// tiles, which are themselves sim state — so both hosts compute the same
    /// forest from the same drive, which is exactly what `island_gate` measures.
    biome_scatter: crate::level::BiomeScatter,
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
    /// The **simulation's own** fracture states (P22.3), keyed by the
    /// destructible actor's entity `Guid`.
    ///
    /// Empty until the caller seeds it after construction (the `voxels` map's
    /// rule, one phase later, and for the same reason: resolving an actor's
    /// `.inf_fracture` needs the built world to walk, and `BeginPlay` runs inside
    /// that construction). A `destruct.*` node on a `BeginPlay` handler therefore
    /// refuses with "no fracture data resident" — stated in the `destruct.*` kit
    /// docs rather than worked around.
    ///
    /// Sim-owned, never render-owned: a wall's rubble must not depend on where
    /// anyone was looking, in either direction.
    fractures: BTreeMap<Uuid, FractureState>,
    /// The level's debris limits (P22.3). Data rather than a constant physics
    /// reads for itself — see `inf_physics::d3::DEFAULT_DEBRIS_MAX_LIVE` for why
    /// a fixed step must not become a function of the graphics tier.
    debris_budget: DebrisBudget,
    /// This step's fracture audit — how many chunks the structural solve dropped,
    /// how many the budget reclaimed, how much debris is live. Read by gates.
    fracture_audit: FractureAudit,
    /// This step's gameplay report (I6) — doors moved, rounds fired, locks
    /// broken, bodies stopped. The `fracture_audit` shape one system along, and
    /// read for the same reason: it is the thing a gate asserts on.
    gameplay: inf_physics::d3::GameplayReport,
    /// This step's vehicle outcomes (island wave VEH1a) — one row per chassis,
    /// in `Guid` order: wheels grounded, suspension load, forward speed. The
    /// `fracture_audit` shape again, and read for the same reason: a gate that
    /// wants to know whether a car is DRIVING has to ask something other than
    /// "did the function get called". Empty on a level with no vehicle.
    vehicles: Vec<inf_physics::d3::VehicleOutcome>,
    /// This step's crowd counters (NPC1a) — how many agents each sim-LOD tier
    /// holds, and what materialized or re-tiered. The `fracture_audit` shape a
    /// third time, and read for the same reason: it is the thing a gate asserts
    /// on. All zeroes on a level with no population.
    crowd: inf_ecs::crowd::CrowdStats,
    /// This step's society counters (NPC1d) — how much of the level's own
    /// population has been derived, and what its network holds. MIRROR of
    /// `SimSession::society`.
    society: inf_ecs::society::SocietyStats,
    /// This step's traffic counters (VEH2b) — cars per tier, how many are
    /// driving, how many carry a driver and how many the traffic has let go of.
    /// The `crowd` shape one system along, and read for the same reason: a gate
    /// that wants to know whether the STREET is alive has to ask something
    /// other than "did the function get called". All zeroes on a level with no
    /// blocks. MIRROR of `SimSession::traffic`.
    traffic: inf_ecs::traffic::TrafficStats,
    /// The sim-LOD ladder's three radii, metres (NPC1a). Data rather than a
    /// constant `step_crowd` reads for itself, for `debris_budget`'s reason one
    /// system over: a level's own crowd block will set it, and the sweep
    /// instrument sets it to price the ladder against an all-`Full` control.
    /// Defaults to `inf_ecs::crowd::DEFAULT_CROWD_RADII`.
    crowd_radii: (f64, f64, f64),
    /// Whether [`fixed_step`](Self::fixed_step) marks its phases (island wave
    /// I4b). `false` on every shipped run; see [`crate::step_profile`] for why a
    /// stopwatch here cannot move the simulation.
    profiling: bool,
    /// The last profiled step's breakdown. All zeroes until
    /// [`set_step_profiling`](Self::set_step_profiling) is armed.
    step_profile: crate::step_profile::StepProfile,
}

/// The **committed camera fold**: the centroid of a set of world positions,
/// summed in ascending `Guid` order.
///
/// # Why this is a free function with its own name
///
/// It is the arithmetic behind [`RuntimeSim::camera_focus`], and
/// `inf_math::predict`'s module header names that method by hand as its
/// worked example of a *committed* pose — "a fold of actor positions, which is
/// a pure function of the committed input — it commits". Two things read the
/// answer and neither is cosmetic:
///
/// * the shipped player's camera, which drives the terrain-residency cut
///   (`terrain_stream::sync_render`'s want set), and
/// * [`inf_math::dead_reckon`], whose prediction is compared between hosts.
///
/// `f64` addition is not associative, so a fold's answer is a function of the
/// **order** it visits its terms in as well as of the terms. Until Wave C the
/// source was a `HashMap`, whose iteration order is seeded per process — so the
/// low bits of the camera, and therefore the residency cut and the prediction,
/// differed between two runs of the same build on the same machine. The
/// purity pin one crate over reads `predict.rs` and could never see it.
///
/// Extracted rather than left inline so that the order is a property something
/// can be *pointed at*: `the_committed_camera_folds_in_guid_order` in
/// `tests/committed_camera.rs` compares this against an independent
/// ascending-`Guid` sum and against a descending one, over terms chosen so the
/// two really do differ.
pub fn camera_centroid(positions: &BTreeMap<Uuid, DVec3>) -> DVec3 {
    if positions.is_empty() {
        return DVec3::ZERO;
    }
    // `BTreeMap::values()` is ascending by key, so the visit order is a function
    // of the Guid set alone.
    let sum: DVec3 = positions.values().copied().sum();
    sum / positions.len() as f64
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
    ///
    /// **A fixture's door.** One 2D vector means that vector in both dimensions
    /// ([`WorldGravity::from_2d`]), which is what every caller of this function
    /// meant before P29.7 and what a hand-built world still means. A **host** has
    /// a level with two authored fields and calls
    /// [`with_gravity`](Self::with_gravity) instead — see [`WorldGravity`].
    pub fn new(
        world: EcsWorld,
        actors: Vec<(Uuid, BlueprintClass)>,
        gravity: DVec2,
        hz: f64,
    ) -> Self {
        Self::with_gravity(world, actors, WorldGravity::from_2d(gravity), hz)
    }

    /// [`new`](Self::new) with **both** solvers' gravity — the door a host uses
    /// (P29.7).
    ///
    /// The 3D bridge is built from `gravity.d3`, i.e. from the level's authored
    /// `gravity_3d`. Before this wave it was built from `gravity_2d.y` and
    /// `gravity_3d` was read by nothing; [`WorldGravity`] carries the whole
    /// finding and the decision.
    pub fn with_gravity(
        mut world: EcsWorld,
        actors: Vec<(Uuid, BlueprintClass)>,
        gravity: WorldGravity,
        hz: f64,
    ) -> Self {
        world.propagate();
        let bridge = PhysicsBridge2D::new(gravity.d2);
        let bridge3d = PhysicsBridge3D::new(gravity.d3);

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
            skeletons: BTreeMap::new(),
            pose_clips: BTreeMap::new(),
            cloths: BTreeMap::new(),
            hairs: BTreeMap::new(),
            hair_detail: inf_anim::HairDetail::GUIDES,
            camera: inf_ecs::camera::LocomotionCamera::default(),
            camera_subject: None,
            stepper: FixedStep::from_hz(hz),
            actors: states,
            entities,
            despawned: Vec::new(),
            input: RuntimeInput::default(),
            prev_down: BTreeSet::new(),
            just_pressed: BTreeSet::new(),
            just_released: BTreeSet::new(),
            holds: inf_input::HoldClock::new(),
            press_threshold_s: inf_ecs::movement::DEFAULT_PRESS_THRESHOLD_S,
            sim_paused: false,
            bindings: BTreeMap::new(),
            dispatch_queue: VecDeque::new(),
            drained_overlaps: Vec::new(),
            logs: BoundedLog::default(),
            grounded: BTreeMap::new(),
            audio: AudioEngine::new(),
            audio_clips: BTreeMap::new(),
            audio_cmds: Vec::new(),
            audio_started: BTreeSet::new(),
            audio_log: BoundedLog::default(),
            steps: 0,
            prev_positions: BTreeMap::new(),
            cur_positions: BTreeMap::new(),
            mods: None,
            terrain: crate::terrain_stream::TerrainStreaming::default(),
            cells: crate::cell_stream::CellStreaming::default(),
            pcg: crate::level::PcgContext::default(),
            biome_scatter: crate::level::BiomeScatter::default(),
            fractures: BTreeMap::new(),
            debris_budget: DebrisBudget::default(),
            fracture_audit: FractureAudit::default(),
            crowd: inf_ecs::crowd::CrowdStats::default(),
            society: inf_ecs::society::SocietyStats::default(),
            traffic: inf_ecs::traffic::TrafficStats::default(),
            crowd_radii: inf_ecs::crowd::DEFAULT_CROWD_RADII,
            gameplay: inf_physics::d3::GameplayReport::default(),
            vehicles: Vec::new(),
            voxels: BTreeMap::new(),
            profiling: false,
            step_profile: crate::step_profile::StepProfile::default(),
        };

        sim.bridge.sync_from_world(&sim.world);
        // P11.3 3D bridge. The voxel map AND the P22.3 fracture map are still
        // empty here — the caller seeds
        // it after `new` returns (`attach_voxel_volumes`) — so a volume's chunk
        // colliders arrive on the first step's sync instead. That is also why a
        // `BeginPlay` handler cannot see a cave (see the P21.4 kit docs): the map
        // it would read does not exist yet.
        sim.bridge3d
            .sync_from_world_sim(&sim.world, &sim.voxels, &sim.fractures);
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

    /// Seed the resolvable `.inf_skel` assets (P24.1) — the runtime mirror of
    /// `SimSession::set_skeletons`.
    ///
    /// Without them every machine still advances and none publishes a pose, so a
    /// character falls back to its `AnimPlayer` (or its rest pose) — the
    /// pre-P24.1 behaviour, which is why a boot path that forgets this door does
    /// not crash and does not warn. `sim_from_built` is the one place that seeds
    /// it, for exactly that reason.
    pub fn set_skeletons(&mut self, skeletons: BTreeMap<Uuid, inf_anim::SkeletonAsset>) {
        self.skeletons = skeletons;
    }

    /// Seed the resolvable `.inf_anim` clips a state machine's states play
    /// (P24.1) — the runtime mirror of `SimSession::set_pose_clips`.
    pub fn set_pose_clips(&mut self, clips: BTreeMap<Uuid, AnimClip>) {
        self.pose_clips = clips;
    }

    /// Seed the resolvable `.inf_cloth` garments (P24.4) — the runtime mirror of
    /// `SimSession::set_cloths`.
    pub fn set_cloths(&mut self, cloths: BTreeMap<Uuid, inf_anim::ClothAsset>) {
        self.cloths = cloths;
    }

    /// The garments this sim can resolve (a read for tests and gates).
    pub fn cloths(&self) -> &BTreeMap<Uuid, inf_anim::ClothAsset> {
        &self.cloths
    }

    /// Seed the resolvable `.inf_hair` hairstyles (P24.4) - the runtime mirror of
    /// `SimSession::set_hairs`.
    pub fn set_hairs(&mut self, hairs: BTreeMap<Uuid, inf_anim::HairAsset>) {
        self.hairs = hairs;
    }

    /// The hairstyles this sim can resolve (a read for tests and gates).
    pub fn hairs(&self) -> &BTreeMap<Uuid, inf_anim::HairAsset> {
        &self.hairs
    }

    /// Set how densely hair draws (P24.4).
    ///
    /// The `set_debris_budget` seam, one system over: the *tier -> detail*
    /// mapping lives at the host (`inf_render::hair_detail_for`), because Ring 0
    /// must not know what a GPU is, and the value arrives here as data. A session
    /// nobody tells runs on `HairDetail::GUIDES`, which is what P24.4 v1 drew.
    pub fn set_hair_detail(&mut self, detail: inf_anim::HairDetail) {
        self.hair_detail = detail;
    }

    /// The hair detail this session draws at (a read for tests and gates).
    pub fn hair_detail(&self) -> inf_anim::HairDetail {
        self.hair_detail
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

    /// Seed the simulation's fracture states (P22.3), keyed by entity `Guid`.
    ///
    /// Built by `inf_physics::d3::resolve_fracture_states` — ONE Ring-0 resolver
    /// both hosts and the PIE seam call, so "which actors can break, and into
    /// what" is answered in one place. Camera-free by construction: the walk
    /// reads the world's own `Destructible` + `MeshRef`, never a render store.
    pub fn set_fractures(&mut self, fractures: BTreeMap<Uuid, FractureState>) {
        self.fractures = fractures;
    }

    /// Read-only view of the simulation's fracture states (P22.3).
    ///
    /// The map a `destruct.*` node writes and the render projector reads. Exposed
    /// so a gate can compare the *state* two runs produced — which chunks came
    /// off, and where they ended up — rather than only the floats a Blueprint
    /// happened to record. The two can disagree, and the state is the authority.
    pub fn fractures(&self) -> &BTreeMap<Uuid, FractureState> {
        &self.fractures
    }

    /// Set the level's debris limits (P22.3).
    ///
    /// **Nothing calls this yet**, and saying so is the point: the budget is data
    /// rather than a constant physics reads for itself (see
    /// `inf_physics::d3::DEFAULT_DEBRIS_MAX_LIVE` for why a fixed step must not
    /// become a function of the graphics tier), but the *tier → budget* mapping
    /// that would fill it in is P22.4's deliverable. Until then every level runs
    /// on `DebrisBudget::default()`. The seam exists so that mapping is a call
    /// site rather than a refactor; it is not yet a knob anyone has turned.
    pub fn set_debris_budget(&mut self, budget: DebrisBudget) {
        self.debris_budget = budget;
    }

    /// The most recent fixed step's fracture audit (P22.3).
    pub fn fracture_audit(&self) -> FractureAudit {
        self.fracture_audit
    }

    /// The most recent fixed step's crowd counters (NPC1a) — agents per
    /// sim-LOD tier, plus what materialized, dematerialized and re-tiered.
    ///
    /// All zeroes on a level with no population, which is how a gate tells
    /// "no crowd" from "a crowd that is not being tiered".
    pub fn crowd_stats(&self) -> inf_ecs::crowd::CrowdStats {
        self.crowd
    }

    /// **The most recent fixed step's society counters** (NPC1d) — how many of
    /// the level's own buildings have been folded into a walkable network, how
    /// many residents they imply, how many have a day, and the two numbers a
    /// gate asserts are zero (`salt_collisions`, `guid_refusals`).
    ///
    /// All zeroes on a level whose volumes offer no resident, which is how a
    /// gate tells "no society" from "a society that is not being derived".
    pub fn society_stats(&self) -> inf_ecs::society::SocietyStats {
        self.society
    }

    /// **The most recent fixed step's traffic counters** (VEH2b) — cars per
    /// tier, how many are on a leg of their day, how many carry an NPC driver,
    /// how many the traffic has let go of, and how many commuter routes are
    /// still waiting to be planned.
    ///
    /// All zeroes on a level with no blocks in it, which is how a gate tells
    /// "no streets" from "streets with nothing on them".
    pub fn traffic_stats(&self) -> inf_ecs::traffic::TrafficStats {
        self.traffic
    }

    /// **Install a crowd population** (NPC1a) — the door the sweep instrument
    /// and the island gate spawn test NPCs through.
    ///
    /// Records arrive tier-less; the next fixed step's
    /// [`inf_ecs::crowd::step_crowd`] materializes the ones the band wants. A
    /// caller cannot pre-decide a tier, which is what keeps the decision in one
    /// place.
    pub fn set_crowd_population(&mut self, records: BTreeMap<Uuid, inf_ecs::crowd::CrowdRecord>) {
        inf_ecs::crowd::set_population(&mut self.world, records);
    }

    /// **Retune the sim-LOD ladder** (NPC1a) — `(full_m, near_m, far_m)`.
    ///
    /// Refused unless the radii are finite and ascending, because a ladder that
    /// is not is not a ladder: `CrowdBand::from_anchors` would fail open to
    /// `Full` and the caller would think it had tightened something. Returns
    /// whether the change was taken.
    ///
    /// **Nothing in production calls this**, and saying so is the point — the
    /// `set_debris_budget` seam, one system over. It exists so the sweep
    /// instrument can price the ladder against an all-`Full` control on the same
    /// scene, and so a level's own crowd block is a call site rather than a
    /// refactor.
    pub fn set_crowd_radii(&mut self, radii: (f64, f64, f64)) -> bool {
        let (f, n, r) = radii;
        if !(f.is_finite() && n.is_finite() && r.is_finite()) || !(f <= n && n <= r) {
            return false;
        }
        self.crowd_radii = radii;
        true
    }

    /// The ladder this sim tiers with.
    pub fn crowd_radii(&self) -> (f64, f64, f64) {
        self.crowd_radii
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
    /// Read-only view of the simulation's voxel volumes (P21.4).
    ///
    /// The map a `voxel.*` node writes and `terrain.height_at` reads. Exposed so a
    /// gate can compare the *field* two runs produced rather than only the floats
    /// a Blueprint happened to record — the two can disagree, and the field is the
    /// authority.
    pub fn voxel_volumes(&self) -> &BTreeMap<Uuid, VoxelData> {
        &self.voxels
    }

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
        self.audio_log.as_slice()
    }

    /// How many audio commands fell off the front of
    /// [`audio_command_log`](Self::audio_command_log)'s ring (Hardening D).
    ///
    /// Non-zero means the slice is a **tail**, not the whole session's stream — a
    /// test that reasons about the first command must assert this is zero first.
    pub fn dropped_audio_commands(&self) -> u64 {
        self.audio_log.dropped()
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
        // The tiles that just became resident have vegetation on them, and the
        // attach happens after `RuntimeSim::new` — so without this the world is
        // queryable for a height and bare for a frame. Same reason the line
        // above exists.
        self.refresh_biome_scatter();
    }

    // ── the biome-bound population (island wave I7b) ──────────────────────

    /// Seed the graphs + palettes the biome-scatter refresh reads. Called by
    /// [`sim_from_built`](crate::sim_from_built), which every boot path goes
    /// through.
    pub fn set_pcg_context(&mut self, pcg: crate::level::PcgContext) {
        self.pcg = pcg;
    }

    /// The per-terrain biome-scatter memo — its counters are what a gate reads
    /// to tell "the refresh ran" from "the refresh was never wired".
    pub fn biome_scatter(&self) -> &crate::level::BiomeScatter {
        &self.biome_scatter
    }

    /// Re-derive every terrain's biome-bound population from the ground that is
    /// resident **now**. One comparison per terrain when nothing paged.
    fn refresh_biome_scatter(&mut self) -> usize {
        if self.pcg.binds_no_biome() {
            return 0;
        }
        crate::level::refresh_biome_bindings(
            &mut self.world,
            &self.pcg.biome_sets,
            &self.pcg.pcgs,
            &mut self.biome_scatter,
        )
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
    /// function the Blueprint node dispatches to (every non-empty [`Terrain`],
    /// each on its component-resident working set and shifted by its own entity
    /// origin, topmost surface that answers winning — the island phase's IB-15
    /// rule; see the free function below).
    ///
    /// Exposed so the streaming gate can assert sim determinism against the real
    /// seam rather than a re-implementation of it.
    ///
    /// `&mut` since Hardening Wave E: the seam resolves its terrains through an
    /// archetype-scoped query instead of a whole-world walk, and building one
    /// needs the world mutably. Nothing is cached, so a gate that traces this
    /// still traces the live simulation.
    pub fn terrain_height_at(&mut self, x: f64, z: f64) -> f64 {
        terrain_height_at(&mut self.world, &self.voxels, x, z)
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
        // Then the runtime hole mask (P21.4): a carve that opened a mouth in the
        // sim's heightfield has to reach the render streamer, or an asset-backed
        // terrain keeps drawing solid ground over it. `sim → render` only — the
        // camera pass above never reaches back.
        //
        // **After** `sync_render`, so the pin set follows *this* frame's cut and
        // stays bounded by it; pinning ahead of the cut let the set grow until it
        // hit the residency ceiling and silently stopped the terrain streaming.
        // Same ordering, same reason, as the voxel overlay's fourth act.
        self.terrain.overlay_sim_edits(&self.world);
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

    /// Mutable access to the 3D bridge, for a **scene query** (P21.4).
    ///
    /// `cast_ray` needs `&mut` because rapier's query pipeline updates itself
    /// lazily. Exposed so a gate can ask the *collider* world what it contains —
    /// which is the only way to tell a carve that reached the solver from one that
    /// only moved four floats a Blueprint recorded.
    pub fn bridge3d_mut(&mut self) -> &mut PhysicsBridge3D {
        &mut self.bridge3d
    }

    pub fn logs(&self) -> &[String] {
        self.logs.as_slice()
    }

    /// How many log lines fell off the front of [`logs`](Self::logs)'s ring
    /// (Hardening D). Non-zero means the slice is a tail, not the whole session.
    pub fn dropped_logs(&self) -> u64 {
        self.logs.dropped()
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
    ///
    /// The fold itself is [`camera_centroid`], which explains why the order it
    /// visits positions in is load-bearing rather than incidental.
    pub fn camera_focus(&self) -> DVec3 {
        camera_centroid(&self.cur_positions)
    }

    /// Advance by a frame's elapsed time via the fixed-step accumulator (0..N
    /// fixed steps, spiral-of-death guarded). `input` is this frame's held set.
    /// Returns how many fixed steps ran.
    pub fn run_frame(&mut self, frame_dt: f64, input: RuntimeInput) -> u32 {
        self.set_input(input);
        // **A paused sim accumulates nothing** (I5), rather than accumulating
        // and then declining to spend it: an accumulator that filled while a
        // menu was open would empty itself in one burst the moment it closed,
        // and the world would jump by however long the player took to read.
        if self.sim_paused {
            // Zero elapsed, and the answer is discarded because it is zero by
            // construction — the call is here so a paused frame still passes
            // through the accumulator's own clamp rather than leaving it in a
            // state no frame ever produced.
            let _ = self.stepper.accumulate(0.0);
            return 0;
        }
        let n = self.stepper.accumulate(frame_dt);
        for _ in 0..n {
            self.fixed_step();
        }
        n
    }

    /// Run exactly one fixed step with the given input — the deterministic entry
    /// point tests and the headless trace script against.
    ///
    /// **A paused sim takes the input and runs no step** (I5). The input is
    /// still taken, so the edge sets stay in the shape they would have been in
    /// — a step run immediately after the pause lifts must not see a stale
    /// press — and the world does not move, which is what makes a trace that
    /// opens the menu cost zero steps in both hosts.
    pub fn step_once(&mut self, input: RuntimeInput) {
        self.set_input(input);
        if self.sim_paused {
            return;
        }
        self.fixed_step();
    }

    /// The click/long-press threshold this session runs with, seconds (I5).
    pub fn press_threshold_s(&self) -> f64 {
        self.press_threshold_s
    }

    /// **Set the three built-in mixer buses' gains** (island wave I5) — the
    /// audio page's consumer.
    ///
    /// Guarded here rather than trusted from the caller, for the reason every
    /// numeric door in this wave is: a value can arrive from a settings file, a
    /// slider, a mod or a test. Non-finite takes `1.0` (an infinity is not "very
    /// loud", it is the absence of a gain) and finite clamps to `[0, 1]` — a
    /// bus gain above unity is a clip, and a settings slider is not the place to
    /// author one.
    ///
    /// **Not sim state**: the mixer is a property of the *listener*, two players
    /// with different volumes run the same simulation, and nothing here reaches
    /// a trace.
    pub fn set_bus_volumes(&mut self, master: f32, sfx: f32, music: f32) {
        let g = |v: f32| -> f64 {
            if v.is_finite() {
                f64::from(v.clamp(0.0, 1.0))
            } else {
                1.0
            }
        };
        self.audio.set_bus_volume(inf_audio::Bus::Master, g(master));
        self.audio.set_bus_volume(inf_audio::Bus::Sfx, g(sfx));
        self.audio.set_bus_volume(inf_audio::Bus::Music, g(music));
    }

    /// A built-in bus's gain — what the arm beside [`set_bus_volumes`](Self::set_bus_volumes)
    /// reads back.
    pub fn bus_volume(&self, bus: inf_audio::Bus) -> f64 {
        self.audio.bus_volume(bus)
    }

    /// **Whether the fixed step is frozen** (island wave I5).
    ///
    /// Set by a host when the in-game menu opens a single-player session — see
    /// `inf_ui::menu`'s ruling for why Tab pauses at all, and
    /// `MenuState::pauses_sim` for why it is a fact about the session rather
    /// than about the window.
    pub fn sim_paused(&self) -> bool {
        self.sim_paused
    }

    /// Freeze or resume it.
    ///
    /// **The pause lives on the SIM, not on the host**, and that is the whole
    /// point: a trace that opens the menu advances zero fixed steps, so the
    /// frames a player spends reading a table cost the simulation nothing and
    /// PIE stays byte-identical to shipping however long they take. A host-level
    /// pause would be invisible to `step_once`, which is the door every trace in
    /// this repository is scripted through.
    pub fn set_sim_paused(&mut self, paused: bool) {
        self.sim_paused = paused;
    }

    /// Set it — the controls settings' door into the sim (I5).
    ///
    /// **Guarded here** rather than trusted from the caller: a non-finite value
    /// falls back to [`inf_ecs::movement::DEFAULT_PRESS_THRESHOLD_S`] (an
    /// infinity is not "a very long press", it is a verb nobody can reach) and a
    /// finite one is clamped into
    /// [`inf_ecs::movement::PRESS_THRESHOLD_RANGE_S`]. The same rule the
    /// settings file's own door applies, restated at the sim's boundary because
    /// a value can arrive here from a mod, a test or a future network path that
    /// never passed through a settings file.
    pub fn set_press_threshold_s(&mut self, seconds: f64) {
        let (lo, hi) = inf_ecs::movement::PRESS_THRESHOLD_RANGE_S;
        self.press_threshold_s = if seconds.is_finite() {
            seconds.clamp(lo, hi)
        } else {
            inf_ecs::movement::DEFAULT_PRESS_THRESHOLD_S
        };
    }

    /// bincode of the `Guid`-sorted sim snapshot **followed by the surface
    /// deformation field's bytes and the evaluated poses'** — the per-step trace
    /// unit folded by the determinism harness (same shape `inf_runtime::replay`
    /// hashes).
    ///
    /// P24.1 appends the poses on exactly the P22.1 argument below: the machine
    /// now drives what is *drawn*, so the drawn pose is sim state, and appending
    /// it here makes every gate the engine already has cover it at once — the
    /// replay fold, `step_state_hash`, and the PIE `Frame::state_hash` the
    /// `PIE == shipping` arms compare. A level that poses nothing produces an
    /// empty vec, so every pre-P24.1 trace is byte-identical.
    ///
    /// P22.1 appends the field rather than tracing it separately, because that
    /// is what makes every gate the engine already has cover it at once: the
    /// replay fold, `step_state_hash`, and the PIE `Frame::state_hash` that the
    /// `PIE == shipping` arms compare all consume this one buffer. A level that
    /// deforms nothing has no field, so
    /// [`inf_ecs::deform::deform_state_bytes`] returns an empty vec and every
    /// pre-P22.1 trace is byte-identical.
    ///
    /// P24.4 appends the **simulated garments** last, on the same argument again:
    /// a settled coat is a pure function of the step history, so folding it here
    /// makes the replay fold, `step_state_hash` and the PIE `Frame::state_hash`
    /// cover cloth at once. The **position is frozen**: cloth bytes come after
    /// the pose bytes and nothing may be inserted before them, because every
    /// committed trace hash in the tree was taken over this concatenation in this
    /// order. A level with no `ClothSim` produces an empty vec, so every
    /// pre-P24.4 trace is byte-identical.
    ///
    /// This buffer is **hashed, never decoded**, which is why appending a second
    /// section needs no version and no reader change.
    pub fn state_bytes(&mut self) -> Vec<u8> {
        let snap = sim_snapshot(&mut self.world);
        let mut out = bincode::serde::encode_to_vec(&snap, bincode::config::standard())
            .expect("sim snapshot is always encodable");
        out.extend_from_slice(&inf_ecs::deform::deform_state_bytes(&self.world));
        out.extend_from_slice(&inf_ecs::pose::pose_state_bytes(&self.world));
        out.extend_from_slice(&inf_ecs::cloth::cloth_state_bytes(&self.world));
        out.extend_from_slice(&inf_ecs::hair::hair_state_bytes(&self.world));
        // I6 appends four sections on the same argument the four above rest on:
        // a door's angle, a bag's contents, a magazine's count and a body's
        // remaining joules are all **sim state**, so folding them here makes
        // every gate the engine already has — the replay fold, `step_state_hash`
        // and the PIE `Frame::state_hash` — cover gameplay at once.
        //
        // **The position is frozen**, exactly as cloth's is: these come after
        // the hair bytes and nothing may be inserted before them, because every
        // committed trace hash in the tree was taken over this concatenation in
        // this order. A level with no door, no bag, no weapon and nothing that
        // can be hurt produces four empty vecs, so every pre-I6 trace is
        // byte-identical.
        out.extend_from_slice(&inf_ecs::door::door_state_bytes(&self.world));
        out.extend_from_slice(&inf_ecs::item::item_state_bytes(&self.world));
        out.extend_from_slice(&inf_ecs::weapon::weapon_state_bytes(&self.world));
        out.extend_from_slice(&inf_ecs::weapon::health_state_bytes(&self.world));
        // NPC1a appends the crowd last, on the same argument the eight above
        // rest on, and with one extra of its own: a `Far` agent evaluates no
        // pose and a `Dormant` one has no entity, so **without this section the
        // sim-LOD decision would be invisible to every trace comparison in this
        // repository** — two hosts that tiered the same NPC differently would
        // compare equal at every step until one of them happened to run a pose
        // the other did not. 49 bytes an agent against a posed character's
        // 6 476: see `inf_ecs::crowd::crowd_state_bytes` for the arithmetic.
        //
        // **The position is frozen**, exactly as cloth's and I6's are: this
        // comes after the health bytes and nothing may be inserted before it,
        // because every committed trace hash in the tree was taken over this
        // concatenation in this order. A level with no population produces an
        // empty vec, so every pre-NPC1a trace is byte-identical.
        out.extend_from_slice(&inf_ecs::crowd::crowd_state_bytes(&self.world));
        // VEH2b appends the traffic, last, on the crowd's argument verbatim: a
        // `Near` car is not simulated and a `Dormant` one has no entity, so
        // **without this section the traffic tier decision would be invisible to
        // every trace comparison in this repository** — two hosts that tiered
        // the same car differently would compare equal until one of them
        // happened to solve a chassis the other did not. 60 bytes a car.
        //
        // The `taken` byte is in it for the same reason `rephase_m` is: it is
        // produced by the simulation (somebody got into a car), it decides
        // everything the traffic step does with that car afterwards, and two
        // hosts that disagreed about it have diverged.
        //
        // **The position is frozen**, exactly as the eight above it are. A level
        // with no streets produces an empty vec, so every pre-VEH2b trace is
        // byte-identical.
        out.extend_from_slice(&inf_ecs::traffic::traffic_state_bytes(&self.world));
        out
    }

    /// **Apply one inventory-panel verb** (I6) to the camera subject's bag.
    ///
    /// The one door the panel's decisions cross into the world through, so a UI
    /// cannot edit a bag by any other route — and `false` when there is no
    /// character, no bag or the slot is empty, which are all things a player can
    /// produce by pressing a key at the wrong moment.
    ///
    /// It is applied by the host between frames rather than inside the fixed
    /// step, and that is a deliberate bound worth stating: a drop lands on the
    /// frame the key was pressed rather than on a fixed step, so a trace that
    /// dropped something would depend on the frame rate. Every gate in this
    /// repository drives `step_once` directly and presses the verb through the
    /// same door, so the traces stay exact; a windowed player's own drop is the
    /// one place the frame clock touches gameplay, and closing it means a queue
    /// on `RuntimeSim` that the fixed step drains.
    pub fn apply_inventory_verb(&mut self, verb: inf_ui::InventoryVerb) -> bool {
        let Some(actor) = inf_ecs::movement::camera_subject(&self.world) else {
            return false;
        };
        match verb {
            inf_ui::InventoryVerb::Drop(slot) => {
                inf_ecs::item::drop_slot(&mut self.world, actor, slot, u32::MAX).is_some()
            }
            inf_ui::InventoryVerb::Move { from, to } => {
                let defs = inf_ecs::item::item_defs(&self.world)
                    .cloned()
                    .unwrap_or_default();
                let Some(e) = self.world.entity_of(actor) else {
                    return false;
                };
                match self
                    .world
                    .world_mut()
                    .get_mut::<inf_ecs::item::Inventory>(e)
                {
                    Some(mut inv) => inv.move_slot(&defs, from, to),
                    None => false,
                }
            }
            inf_ui::InventoryVerb::Equip(slot) => {
                let Some(e) = self.world.entity_of(actor) else {
                    return false;
                };
                let id = {
                    let w = self.world.world();
                    let Some(inv) = w.get::<inf_ecs::item::Inventory>(e) else {
                        return false;
                    };
                    match inv.slots.get(slot).and_then(|s| s.as_ref()) {
                        Some(s) => s.id.clone(),
                        None => return false,
                    }
                };
                inf_physics::d3::gameplay::equip_weapon(&mut self.world, actor, &id)
            }
        }
    }

    /// **What the last fixed step's gameplay did** (I6) — doors moved, rounds
    /// fired, locks broken, bodies stopped.
    ///
    /// A report rather than a log: it is the thing a gate asserts on, which is
    /// `FractureAudit`'s own shape one system along.
    pub fn gameplay(&self) -> &inf_physics::d3::GameplayReport {
        &self.gameplay
    }

    /// **What the last fixed step's vehicles did** (island wave VEH1a) — one row
    /// per chassis in `Guid` order.
    ///
    /// The `gameplay()` shape one phase along, and MIRRORED by
    /// `SimSession::vehicles`. A gate that wants to know a car is *driving* asks
    /// this; the state it should assert on is still the world.
    pub fn vehicles(&self) -> &[inf_physics::d3::VehicleOutcome] {
        &self.vehicles
    }

    /// The surface deformation field this sim has pressed into its terrain
    /// (P22.1), or `None` on a level where nothing has ever touched ground.
    pub fn deform_field(&self) -> Option<&inf_terrain::deform::DeformField> {
        inf_ecs::deform::deform_field(&self.world)
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
        // **The phase clock** (island wave I4b). Off by default and one branch
        // per mark when it is; see `crate::step_profile` for why a stopwatch in
        // this body reads no sim state, writes none, and changes no ordering.
        // Every mark charges the time since the previous one, so the phases tile
        // the step by construction rather than by an assertion somebody has to
        // remember to write.
        use crate::step_profile::phase;
        let mut clk = crate::step_profile::StepClock::start(self.profiling);
        // 0a. World-partition CELL streaming (P16.5). Spawn/despawn the cells the
        //     sim's own `StreamingSource` entities want, in ascending cell order,
        //     BEFORE anything else — including terrain's observer scan, which has
        //     to see the entities a freshly-activated cell brought in. Camera-free
        //     by construction: `sync_sim` has no camera to be given.
        self.cells.sync_sim(&mut self.world, self.steps);
        clk.mark(phase::CELL_STREAM);
        // 0b. Terrain streaming, SIM side (P16.3b2). Level-0 pages around the sim's
        //    own entities, loaded synchronously in key order BEFORE anything in
        //    this step can query a height. Camera-free by construction — see
        //    `crate::terrain_stream` for why that separation is structural.
        self.terrain.sync_sim(&mut self.world);
        clk.mark(phase::TERRAIN_STREAM);
        // 0c. **The vegetation on the ground that just arrived** (island wave
        //     I7b). After 0a and 0b, because its subject is exactly what those
        //     two made resident: a cell can bring a terrain entity, and the
        //     terrain streamer brings that terrain's tiles. A pure function of
        //     the resident set — never of arrival order or of who looked first
        //     (P21's law) — so both hosts grow the same forest from the same
        //     drive. One comparison per terrain on every step that paged
        //     nothing, which is almost all of them.
        self.refresh_biome_scatter();
        clk.mark(phase::BIOME_SCATTER);
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
        clk.mark(phase::SKY);
        // ── NPC1a the crowd ── decide every agent's sim-LOD tier, materialize
        //    or dematerialize it, and put it where its route says it is. ONE
        //    Ring-0 call (`inf_ecs::crowd`) rather than a loop spelled twice —
        //    the deform doctrine's shape — so the editor preview and the
        //    shipped player cannot tier the same NPC differently.
        //
        //    HERE, and not later: the physics sync below has to see this step's
        //    bodies (a `Far` agent has none), and `step_pose_evaluation` has to
        //    see this step's tiers. After the sky, because a schedule reads the
        //    clock (NPC1d) and the crowd is what will read the schedule.
        //
        //    Inert — one `contains_resource` branch, no allocation — on every
        //    level with no population, which is every level committed before
        //    this wave. (MIRROR of `SimSession::fixed_step`.)
        // ── NPC1d THE SOCIETY ── grow the level's own population before it is
        //    tiered. One Ring-0 door both hosts call, keyed on the level's own
        //    `PcgVolume::residents` and never on anything a host knows privately
        //    — the `set_crowd_population` seam finally has a PRODUCTION caller,
        //    and it is the level's own buildings.
        //
        //    Inert on every level whose volumes offer no resident, which is
        //    every level committed before this wave: one entity walk that finds
        //    nothing and inserts no resource.
        self.society = inf_ecs::society::sync_society(&mut self.world);
        clk.mark(phase::SOCIETY);
        self.crowd = inf_ecs::crowd::step_crowd_banded(&mut self.world, dt, self.crowd_radii);
        clk.mark(phase::CROWD);
        // ── VEH2b traffic ── the level's own carriageway, the tier every car
        //    takes, and the stick each steered car's driver is handed. HERE for
        //    the crowd's own two reasons: a car built this step must be mirrored
        //    by the sync below on this step, and the driver's INTENT must be
        //    written before `character move` reads it. Inert on a level with no
        //    blocks. (MIRROR of `SimSession::fixed_step`.)
        let (tw, tb) = (&mut self.world, &mut self.bridge3d);
        // MIRROR-BEGIN traffic_step
        self.traffic = inf_physics::d3::traffic::step_traffic(tw, tb, dt);
        // MIRROR-END traffic_step
        clk.mark(phase::TRAFFIC);
        // 1. ECS → physics.
        self.bridge.sync_from_world(&self.world);
        clk.mark(phase::PHYSICS2D_SYNC);
        // ── P22.3 fracture follow ── an INTACT destructible is a normal
        //    entity a Blueprint or a gizmo can move, so its placement tracks
        //    its transform right up until the first chunk comes off (after
        //    which the chunks are solver-owned and following would teleport
        //    settled rubble). Before the sync, because the sync reads the map
        //    while this writes it. (MIRROR of the other host's fixed step.)
        PhysicsBridge3D::follow_fractures(&self.world, &mut self.fractures);
        // ── P11.3 3D bridge: sync ── carrying the P21.4 voxel chunk colliders,
        //    so a runtime carve is something a body can fall into.
        self.bridge3d
            .sync_from_world_sim(&self.world, &self.voxels, &self.fractures);
        clk.mark(phase::PHYSICS3D_SYNC);
        // ── P20.2 water forces ── buoyancy + hydrodynamic drag, between the sync
        //    and the solver: after the sync because a body must be sampled where it
        //    IS, and before the step because that is the step the forces belong to.
        //    Also arms this step's enter/exit/splash, drained in the collision slot
        //    below so the fixed step has ONE event point. One branch on a level with
        //    no `Buoyancy` component. (MIRROR of `SimSession::fixed_step`.)
        self.bridge3d.apply_water_forces(dt);
        clk.mark(phase::WATER);
        // ── Wave 3 input events ── (MIRROR of SimSession) fire Input(action) edges
        //    BEFORE the Tick pass, then drain any dispatches they queued.
        self.fire_input_events();
        self.drain_dispatch();
        clk.mark(phase::INPUT_EVENTS);
        // 2. Blueprint Tick for every actor (Guid order).
        let args: HashMap<String, Value> = [("dt".to_string(), Value::Float(dt))].into();
        self.run_all_with_args(&EventKind::Tick, &args);
        self.drain_dispatch(); // Wave 3: Tick may dispatch custom events.
        clk.mark(phase::BLUEPRINT_TICK);
        // ── P29.3 character movement ── the ONE Ring-0 fixed step, called by
        //    both hosts, so the editor preview and the shipped player cannot
        //    integrate a character differently (the `step_pose_evaluation`
        //    shape). HERE, and not earlier: it runs AFTER the Blueprint Tick, so
        //    gameplay that set a character's intent this step is honoured this
        //    step; and BEFORE the solver, which is the slot
        //    `physics3d.move_and_slide` has always occupied. Inert (one empty
        //    query) on every level with no `CharacterMovement`.
        //
        //    Two calls rather than one because an intent is not input: a
        //    Blueprint, an AI or a replay can write one, and `apply_intent` is
        //    only the local player's path into the same field.
        //    **The press durations advance on the FIXED step** (I5), against
        //    this step's own held set, so "C has been down for 300 ms" is a
        //    fact about the simulation and not about the machine's frame rate.
        //    A frame that runs three fixed steps advances it three times; a
        //    frame that runs none advances it not at all, which is the same
        //    discipline every other integration in this body follows.
        //    (MIRROR of `SimSession::fixed_step`.)
        self.holds.advance(&self.input.down, dt);
        let threshold = self.press_threshold_s;
        let intent = inf_ecs::movement::MovementIntent::from_actions(
            |a| self.input.axis(a),
            |a| self.input.is_down(a),
            |a| self.just_pressed.contains(a),
            |a| self.just_released.contains(a),
            |a| self.holds.hold(a),
            threshold,
        );
        inf_ecs::movement::apply_intent(&mut self.world, &intent);
        inf_physics::d3::step_character_movement(&mut self.world, &mut self.bridge3d, dt);
        clk.mark(phase::CHARACTER_MOVE);
        // ── VEH1a the vehicle step ── the wheel rays, the model's forces and
        //    the visual wheel write, in the slot they have always run in: right
        //    after the character step (a driver's controls were written by it,
        //    from the same intent) and before the solver (the forces must land
        //    for the step they belong to). What is new is the ROW: P29.7 hid it
        //    inside `step_character_movement`'s last statement, where a car's
        //    milliseconds were charged to `character move` and could not be
        //    told from a crowd's.
        //
        //    `O(vehicles)` and an early return on a level with none.
        //    (MIRROR of `SimSession::fixed_step`, fenced and pinned.)
        let (w, b) = (&mut self.world, &mut self.bridge3d);
        // MIRROR-BEGIN vehicle_step
        self.vehicles = inf_physics::d3::step_vehicles(w, b, dt);
        // MIRROR-END vehicle_step
        clk.mark(phase::VEHICLE);
        // ── I6 gameplay ── doors swing, weapons cycle, health latches. HERE,
        //    between the character step and the solver, for two reasons that are
        //    both about one beat of latency:
        //
        //    * a door the E key opened THIS step starts moving this step, because
        //      `step_character_movement` is where the press is consumed; and
        //    * the leaf's collider is pushed into the bridge before
        //      `bridge3d.step`, so the solver this step runs sees the door where
        //      the state says it is rather than where it was last step.
        //
        //    Inert on a level with no door, no weapon and no inventory: three
        //    `try_query_filtered`s that answer `None`. (MIRROR of
        //    `SimSession::fixed_step`.)
        //
        //    **What it does NOT do is route damage at a destructible.** A shot
        //    that hits a wall comes back in `GameplayReport::destruct`, and the
        //    host spends it through its own `runtime_destruct_damage` — the same
        //    wrapper the `destruct.apply_damage` node uses, so the permission
        //    gate and the near-miss log are on every path into that door rather
        //    than on the Blueprint one only.
        let mut report = inf_physics::d3::step_gameplay(&mut self.world, &mut self.bridge3d, dt);
        for (entity, energy_j) in &report.destruct {
            runtime_destruct_damage(
                &mut self.bridge3d,
                &mut self.fractures,
                &self.world,
                &mut self.logs,
                *entity,
                *energy_j,
            );
        }
        // The impacts, as sound, through the P12 command queue — before the
        // report is stored, because the report is what a gate reads and the
        // queue is what a player hears.
        let hits = std::mem::take(&mut report.hits);
        self.fire_weapon_audio(&hits);
        report.hits = hits;
        self.gameplay = report;
        clk.mark(phase::GAMEPLAY);
        // 3. Solver.
        self.bridge.step(dt);
        self.bridge3d.step(dt); // ── P11.3 3D bridge: step ──
        clk.mark(phase::SOLVER);
        // ── Wave 3 collision + overlap drain ── (MIRROR of SimSession) between the
        //    solver and write-back: fire `Collision` events + collect OverlapEvents.
        self.drain_collisions();
        self.drain_dispatch();
        clk.mark(phase::COLLISION_DRAIN);
        // 4. Physics → ECS.
        self.bridge.write_back(&mut self.world);
        self.bridge3d.write_back_into(&mut self.world); // ── P11.3 3D bridge: write-back ──
        clk.mark(phase::WRITE_BACK);
        self.world.propagate();
        clk.mark(phase::PROPAGATE);
        // ── P22.1 surface deformation ── the ground remembers what stood on it.
        //    Here, and not earlier: a footprint's XZ is read off the transform
        //    the solver just wrote and `propagate` just settled, so the print
        //    lands where the body actually ended the step. ONE Ring-0 call
        //    (`inf_ecs::deform`) rather than a loop spelled twice — the sky
        //    advance's shape — so the editor preview and the shipped player
        //    cannot disagree about where a track goes. Inert (one empty vec, no
        //    allocation) on every level whose bodies never touch a terrain.
        //    (MIRROR of `SimSession::fixed_step`.)
        inf_ecs::deform::step_deformation(&mut self.world, dt);
        clk.mark(phase::DEFORMATION);
        // 5. Advance skeletal-animation play-heads (P11.1) — the same order-free,
        //    fixed-`dt` integration the editor Simulate tick runs (preview ==
        //    shipped). ── P11.3 root motion ── snapshot play-heads, advance, apply.
        let prev_ts = self.capture_root_motion_times();
        inf_ecs::anim::advance_anim_players(&mut self.world, dt);
        // ── SK1c: ROOT MOTION MOVES THE CHARACTER, THEN THE POSE IS EVALUATED
        //    AGAINST WHERE IT IS ──
        //
        //    These two lines used to be the other way round here and this way
        //    round in `SimSession::fixed_step`, and the SK1b audit carried the
        //    disagreement as an unmeasured LOW: the two hosts agreed on every
        //    committed trace, so nothing could say whether the passes commuted or
        //    whether no course had ever exercised both.
        //
        //    **They do not commute.** `step_pose_evaluation` reads the entity's
        //    `GlobalTransform` twice — `authored_ik_goals` inverts it to convert a
        //    world-space IK goal into model space, and `model_to_world` feeds the
        //    foot pass, the hand pass and the published feet — so which side of
        //    root motion the pose sits on decides which step's frame those are
        //    computed in. Measured on
        //    `both_hosts_pose_a_root_motion_driven_character_the_same_way`'s
        //    fixture at the SK1b head: **different bytes on every one of eight
        //    steps**, worst pose component 0.060 at step 2, while the transform
        //    itself agreed to the bit.
        //
        //    Root motion is *movement*, and every other movement in this engine
        //    happens before the pose — `step_character_movement` and
        //    `step_gameplay` both do, and `anim_bridge`'s own doc rests on it. So
        //    the player moves to the editor's order rather than the other way
        //    round, and the propagate between them is what makes the pose read
        //    THIS step's placement instead of last step's. Pinned as source text
        //    by `both_fixed_steps_move_the_root_before_the_pose`.
        self.apply_root_motion(&prev_ts);
        self.world.propagate();
        // ── P11.2 anim state machines ── Step each `AnimStateMachine` against
        //    its actor's Blueprint variables.
        self.advance_state_machines(dt);
        clk.mark(phase::ANIMATION);
        self.world.propagate();
        clk.mark(phase::PROPAGATE);
        // ── P11.3 attachments ── entities ride their target's socket, post-anim.
        update_attachments(&mut self.world);
        clk.mark(phase::ATTACHMENTS);
        self.world.propagate();
        clk.mark(phase::PROPAGATE);
        // ── P24.4 cloth ── garments fall on the body the pose just put them on.
        //    HERE, and not earlier: the capsules are read off the pose this step
        //    published and the model frame off a `GlobalTransform` the propagate
        //    above has settled, so a coat collides against THIS step's arm rather
        //    than the last one's. ONE Ring-0 call (`inf_ecs::cloth`) rather than a
        //    loop spelled twice — the deform doctrine's shape — so the editor
        //    preview and the shipped player cannot disagree about how a garment
        //    falls. Inert (one empty query, no allocation) on every level with no
        //    `ClothSim`. (MIRROR of `SimSession::fixed_step`.)
        self.step_cloth(dt);
        // -- P24.4 hair -- strands fall on the head the pose just put them on,
        //    in the same slot and for the same reasons as the garment above.
        //    (MIRROR of `SimSession::fixed_step`.)
        self.step_hair(dt);
        clk.mark(phase::CLOTH_HAIR);
        // ── P14.5 WASM mods ── tick sandboxed mods against the world (after
        //    gameplay/physics/anim), then propagate their transform edits.
        self.tick_mods(dt);
        clk.mark(phase::MODS);
        // ── P22.3 runtime destruction ── advance the fracture states: age the
        //    debris, run the structural solve, apply the level's budget, latch
        //    `Destroyed`. HERE, and not earlier: the support probes read where
        //    bodies actually ended the step, and a collapse decided now becomes
        //    chunk bodies on the NEXT sync — the one beat of latency that makes
        //    progressive collapse progressive rather than instantaneous. Inert
        //    (one `is_empty` branch) on every level with no destructible actor.
        //    (MIRROR of the other host's fixed step.)
        self.bridge3d.write_back_fractures(&mut self.fractures);
        let (audit, destroyed) =
            self.bridge3d
                .step_fractures(&mut self.fractures, dt, self.debris_budget);
        self.fracture_audit = audit;
        self.fire_destroyed(&destroyed);
        clk.mark(phase::DESTRUCTION);
        // ── P12.3 audio step ── last, observing this step's final transforms
        //    (preview == shipped: the same logic the editor SimSession runs).
        self.audio_step();
        clk.mark(phase::AUDIO);
        // ── P29.6 the locomotion camera ── LAST, and outside everything the
        //    trace folds: it reads where the character ended this step and writes
        //    nothing back. `step_locomotion_camera` is the ONE door, so the
        //    editor's Simulate and the shipped player cannot frame the same
        //    character differently. Inert (one `O(characters)` query that answers
        //    `None`) on every level with no player-controlled character.
        //    (MIRROR of `SimSession::fixed_step`.)
        self.step_camera(dt);
        clk.mark(phase::CAMERA);
        // Roll interpolation history + rising edges.
        std::mem::swap(&mut self.prev_positions, &mut self.cur_positions);
        self.capture_positions();
        self.just_pressed.clear();
        self.steps += 1;
        clk.mark(phase::POSITION_CAPTURE);
        if let Some(p) = clk.finish() {
            self.step_profile = p;
        }
    }

    /// Arm (or disarm) the fixed step's per-phase clock (island wave I4b).
    ///
    /// Off on every shipped run. See [`crate::step_profile`] for the cost and for
    /// why a profiled step and an unprofiled one produce byte-identical sim
    /// state.
    pub fn set_step_profiling(&mut self, on: bool) {
        self.profiling = on;
    }

    /// The last profiled step's per-phase breakdown — all zeroes until
    /// [`set_step_profiling`](Self::set_step_profiling) is armed.
    pub fn step_profile(&self) -> crate::step_profile::StepProfile {
        self.step_profile
    }

    /// Advance the locomotion camera against this step's world.
    ///
    /// The subject is re-resolved every step rather than latched, because a
    /// character can be spawned or despawned by gameplay and a camera pinned to a
    /// guid that has gone is a camera frozen in space.
    fn step_camera(&mut self, dt: f64) {
        self.camera_subject = inf_ecs::movement::camera_subject(&self.world);
        if let Some(subject) = self.camera_subject {
            inf_physics::d3::step_locomotion_camera(
                &self.world,
                &mut self.bridge3d,
                &mut self.camera,
                subject,
                dt,
            );
        }
    }

    /// The camera's pose this step, or `None` on a level with no
    /// player-controlled character (where the host keeps its own view).
    pub fn camera_pose(&self) -> Option<inf_ecs::camera::CameraPose> {
        self.camera_subject.map(|_| self.camera.pose)
    }

    /// The camera itself — for a host that wants to switch view mode or shoulder,
    /// and for the gate's camera trace.
    pub fn camera(&self) -> &inf_ecs::camera::LocomotionCamera {
        &self.camera
    }

    /// …and mutably, which is how a host toggles first person. Deliberately not a
    /// `set_view_mode` pair: the camera is a plain value and hiding it behind two
    /// setters would invite a third.
    pub fn camera_mut(&mut self) -> &mut inf_ecs::camera::LocomotionCamera {
        &mut self.camera
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
        // -- P29.4 footsteps -- the animation's own event markers, turned into
        //    voices. The DECISION is Ring 0 (`inf_ecs::anim_bridge::footstep_cues`
        //    is a pure function of this step's notifies and the clip's
        //    `Mask_FootstepSound` channel); this is only the mapping onto the
        //    queue, and it is the same six lines in the other host. Inert on
        //    every level whose clips carry no footstep markers.
        for cue in inf_ecs::anim_bridge::footstep_cues(&self.world) {
            let Some(src) = audio_source_of(&self.world, cue.source) else {
                continue;
            };
            let key = cue.source.as_u128() as u64;
            let pos = emitter_position(&self.world, cue.source);
            let mut cmd = play_command_for(key, &src, src.spatial.then_some(pos));
            cmd.volume *= cue.gain;
            self.audio_cmds.push(AudioCommand::Play(cmd));
        }
        let listener = active_listener(&self.world);
        let listener_pos = listener
            .map(|l| l.position)
            .unwrap_or_else(|| self.audio.listener().position);
        if let Some(l) = listener {
            self.audio_cmds.push(AudioCommand::SetListener(l));
        }

        // -- VEH1a the engine loop -- a looping spatial `Play` once, then a
        //    `SetPitch` and a `SetVolume` every step from THIS step's outcome.
        //    Zero new audio API: the three commands have existed since P12.3 and
        //    the decision (`inf_ecs::vehicle::engine_cue`) is Ring 0, so the
        //    stream stays a pure function of sim state — the P12 doctrine, met
        //    by a system that moves.
        //
        //    BEFORE the autoplay walk on purpose: that walk adds anything
        //    already in `audio_started` to `still_alive`, so a chassis latched
        //    here is not Stopped by the despawn sweep on the step it began, and
        //    an autoplay `AudioSource` on the same entity is skipped rather than
        //    firing a second `Play`.
        //
        //    Inert on a level with no vehicle, and on a vehicle with no
        //    `AudioSource`. (MIRROR of `SimSession::audio_step`.)
        let (vehicles, world, started, cmds) = (
            &self.vehicles,
            &self.world,
            &mut self.audio_started,
            &mut self.audio_cmds,
        );
        // MIRROR-BEGIN vehicle_engine_audio
        for out in vehicles {
            let Some(src) = audio_source_of(world, out.chassis) else {
                continue;
            };
            let key = guid_source_key(out.chassis);
            if started.insert(out.chassis) {
                let pos = emitter_position(world, out.chassis);
                cmds.push(AudioCommand::Play(play_command_for(
                    key,
                    &src,
                    src.spatial.then_some(pos),
                )));
            }
            let cue = inf_ecs::vehicle::engine_cue(out.revs, out.load, src.pitch, src.volume);
            cmds.push(AudioCommand::SetPitch {
                source: key,
                pitch: cue.pitch,
            });
            cmds.push(AudioCommand::SetVolume {
                source: key,
                volume: cue.volume,
            });
        }
        // MIRROR-END vehicle_engine_audio

        // Autoplay once per not-yet-started `AudioSource`, plus the **started**
        // emitters still alive (for despawn pruning below).
        //
        // `still_alive` used to be `live`: every guid in the world, inserted into
        // a fresh `BTreeSet` every fixed step — thirteen thousand node
        // allocations on a furnished town — so that a handful of started
        // emitters could each be asked whether they still existed (lens 3 P34).
        // The question is the same one, asked the other way round: an entity is
        // only interesting here if it is already in `audio_started`, which is
        // small by construction (one entry per emitter that has ever played).
        // Exactly the same source of truth as before — this walk — so no
        // reliance on the guid index agreeing with it.
        let mut autoplay: Vec<(Uuid, AudioSource, DVec3)> = Vec::new();
        let mut still_alive: BTreeSet<Uuid> = BTreeSet::new();
        for e in self.world.world().iter_entities() {
            let Some(guid) = e.get::<Guid>().map(|g| g.0) else {
                continue;
            };
            if self.audio_started.contains(&guid) {
                still_alive.insert(guid);
            }
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
            // Started THIS step, and seen alive by the walk above — so it
            // belongs in `still_alive` too, or the despawn sweep below would
            // Stop it on the very step it began and autoplay would re-fire it
            // for ever. (`runtime_autoplay_source_emits_one_deterministic_play_
            // command` measured exactly that: five Plays over five steps.)
            self.audio_started.insert(guid);
            still_alive.insert(guid);
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
            .filter(|g| !still_alive.contains(*g))
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

    /// Step every entity's [`AnimStateMachine`] (P11.2) **and evaluate the pose it
    /// lands in** (P24.1), through the ONE Ring-0 rule both hosts call
    /// ([`inf_ecs::pose::step_pose_evaluation`]) — the runtime mirror of
    /// `SimSession::advance_state_machines` (preview == shipped).
    ///
    /// The loop this used to spell inline is gone: it was byte-identical to the
    /// editor's, which is the shape `inf_ecs::deform` replaced. What is left is
    /// *which registries answer*, which is genuinely host-local — the player
    /// resolves them out of a cooked pack (or a PIE payload), the editor out of
    /// the project's asset DB.
    fn advance_state_machines(&mut self, dt: f64) {
        if self.state_machines.is_empty() {
            return;
        }
        // Split the borrow: the rule needs `&mut world` while the resolvers read
        // sibling fields of the same struct.
        let Self {
            world,
            state_machines,
            skeletons,
            pose_clips,
            actors,
            ..
        } = self;
        let (state_machines, skeletons, pose_clips, actors) =
            (&*state_machines, &*skeletons, &*pose_clips, &*actors);
        let machines = |g: Uuid| state_machines.get(&g);
        let skels = |g: Uuid| skeletons.get(&g);
        let clips = |c: inf_anim::ClipRef| pose_clips.get(&Uuid::from_bytes(c));
        let vars = |g: Uuid| {
            actors
                .get(&g)
                .map(|a| var_snapshot(&a.instance))
                .unwrap_or_default()
        };
        inf_ecs::pose::step_pose_evaluation(world, dt, &machines, &skels, &clips, &vars);
    }

    /// Advance every worn garment (P24.4) through the ONE Ring-0 rule both hosts
    /// call ([`inf_ecs::cloth::step_cloth_simulation`]) — the runtime mirror of
    /// `SimSession::step_cloth`.
    ///
    /// Only *which registries answer* is host-local: the player resolves them out
    /// of a cooked pack (or a `--level` dev dir), the editor out of the project's
    /// asset DB. The rule itself lives once.
    ///
    /// Returns immediately on a level with no resolvable garment, so a world that
    /// wears nothing pays one `is_empty` branch per step.
    fn step_cloth(&mut self, dt: f64) {
        if self.cloths.is_empty() {
            return;
        }
        let Self {
            world,
            cloths,
            skeletons,
            ..
        } = self;
        let (cloths, skeletons) = (&*cloths, &*skeletons);
        let garments = |g: Uuid| cloths.get(&g);
        let skels = |g: Uuid| skeletons.get(&g);
        inf_ecs::cloth::step_cloth_simulation(world, dt, &garments, &skels);
    }

    /// Advance every worn hairstyle (P24.4) through the ONE Ring-0 rule both hosts
    /// call ([`inf_ecs::hair::step_hair_simulation`]) - the runtime mirror of
    /// `SimSession::step_hair`.
    fn step_hair(&mut self, dt: f64) {
        if self.hairs.is_empty() {
            return;
        }
        let detail = self.hair_detail;
        let Self {
            world,
            hairs,
            skeletons,
            ..
        } = self;
        let (hairs, skeletons) = (&*hairs, &*skeletons);
        let styles = |g: Uuid| hairs.get(&g);
        let skels = |g: Uuid| skeletons.get(&g);
        inf_ecs::hair::step_hair_simulation(world, dt, &styles, &skels, detail);
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
            // Root motion is expressed in the character's facing frame → rotate
            // the ground-plane delta into world space through the ONE door both
            // fixed steps share. `DQuat::from_rotation_y` is `sin_cos` inside
            // glam, and this transform is folded into `state_bytes` (L6.F5).
            let world_delta = inf_anim::root_delta_world(t.rotation.y, d.translation);
            let new_yaw_deg = t.rotation.y + d.yaw.to_degrees() as f64;
            let pos = t.translation.to_dvec3();

            let new_pos = if has_cc {
                let mover = inf_physics::d3::mover_for(&self.world, guid);
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
        // `grounded` is the same shape and had no such rule (Hardening D): it is
        // written per `move_and_slide` and was never pruned, so in a world that
        // spawns and despawns characters — streamed cells, respawns, projectiles
        // that walk — it grew for the session. Its siblings are all pruned:
        // `audio_started` drops a despawned emitter in the audio step (and Stops
        // it), and `prev_positions`/`cur_positions` are rebuilt here every step.
        //
        // The live set is the world's, not the actor map's: a character need not
        // carry a blueprint to be moved by one.
        self.grounded.retain(|guid, _| {
            self.world
                .entity_of(*guid)
                .is_some_and(|e| self.world.world().get_entity(e).is_ok())
        });
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
                entities: &mut self.entities,
                despawned: &mut self.despawned,
                logs: &mut self.logs,
                grounded: &mut self.grounded,
                audio_cmds: &mut self.audio_cmds,
                current_entity,
                bindings: &mut self.bindings,
                dispatch_queue: &mut self.dispatch_queue,
                dt: self.stepper.fixed_dt(),
                voxels: &mut self.voxels,
                fractures: &mut self.fractures,
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
        // SCRIPT3 MIRROR of `SimSession::run_on_guid`: `engine.destroy` took an
        // entity out of the world; if it was an actor its handlers stop with it.
        // Drained AFTER the re-insert, so an actor that destroyed itself does
        // not come back.
        for g in std::mem::take(&mut self.despawned) {
            self.actors.remove(&g);
        }
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

    /// Fire this step's `Destroyed` events on their actors (P22.3) — the MIRROR
    /// of `SimSession::fire_destroyed`.
    ///
    /// Fired in the **fracture slot**, right after the structural solve that
    /// decided it, rather than in the collision slot: a collapse is sensed from
    /// the post-step poses (a chunk's support is a query against where things
    /// actually ended up), so putting it with the contacts would give one fixed
    /// step two different answers to "when did this happen". The bridge already
    /// produced the list in actor-`Guid` order, so this loop adds no ordering of
    /// its own.
    ///
    /// # The audio hook, and why it is a push rather than a handler call
    ///
    /// P20.2 deliberately did **not** give water events an implicit sound: the
    /// P12.3 doctrine is that the audio stream is a pure function of sim state,
    /// water events *are* sim state, and a `Play` queued from an `On Splash`
    /// handler lands in the command queue in the same deterministic order. That
    /// reasoning still holds — and it is not enough here, because the common case
    /// is different. A destructible wall usually has **no Blueprint at all**: it
    /// is a mesh, a `Destructible` and a collider. A handler-only route would make
    /// every wall in a level silent unless somebody scripted it one at a time.
    ///
    /// So an actor that carries an `AudioSource` plays it when it is destroyed,
    /// pushed straight onto the same `AudioCommandQueue` autoplay uses, in the
    /// same deterministic slot. An actor that also has an `On Destroyed` handler
    /// gets both, which is right: the component is the default and the handler is
    /// the override.
    ///
    /// **The clip is the emitter's own.** `AudioSource::clip` is what plays;
    /// there is no "destruction sound" slot on `Destructible` and there is not
    /// going to be one (the memo's §5 rule — a field describing the whole scene
    /// or a whole subsystem does not belong on this component). The
    /// asset-resolution gap `audio.play_oneshot` has is not solved here either:
    /// an emitter whose clip GUID is not in the sim's registered set is silent,
    /// exactly as it is for autoplay.
    fn fire_destroyed(&mut self, events: &[DestroyedEvent]) {
        if events.is_empty() {
            return;
        }
        for ev in events {
            // The audio push first, so it is queued whether or not the actor has
            // a handler — and so an actor that has both gets the component's
            // sound before whatever its handler queues.
            if let Some((src, pos)) = destroyed_emitter(&self.world, ev.entity) {
                let cmd =
                    play_command_for(guid_source_key(ev.entity), &src, src.spatial.then_some(pos));
                self.audio_cmds.push(AudioCommand::Play(cmd));
            }
            if !self.actors.contains_key(&ev.entity) {
                continue;
            }
            let args: std::collections::HashMap<String, Value> =
                [("chunks".to_string(), Value::Int(ev.detached as i64))].into();
            self.run_on_guid(ev.entity, &EventKind::Destroyed, &args);
        }
    }

    /// **This step's impacts, as sound** (island wave I6) — one `Play` per round
    /// that landed on something with an emitter, through the P12 command queue.
    ///
    /// MIRROR of the other host's, character for character, for the reason
    /// `fire_destroyed` is: a preview that made a different noise from the
    /// shipped build is a bug no compiler and no screenshot finds.
    ///
    /// **The clip is the TARGET's own `AudioSource`**, which is exactly the rule
    /// `destroyed_emitter` follows and it is the same argument: a wall usually
    /// has no Blueprint at all, so a handler-only route would make every wall
    /// silent. There is deliberately no "impact sound" slot on a weapon — P22's
    /// §5 refuses that class of field, and a shot into concrete and a shot into
    /// glass should differ by what was hit rather than by what fired.
    ///
    /// Positioned at the **hit**, not at the emitter: a round that struck the far
    /// end of a wall should be heard from there.
    fn fire_weapon_audio(&mut self, hits: &[inf_physics::d3::WeaponHit]) {
        if hits.is_empty() {
            return;
        }
        for hit in hits {
            let Some(target) = hit.target else {
                continue;
            };
            let Some((src, _)) = destroyed_emitter(&self.world, target) else {
                continue;
            };
            let cmd =
                play_command_for(guid_source_key(target), &src, src.spatial.then_some(hit.to));
            self.audio_cmds.push(AudioCommand::Play(cmd));
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
    /// The blueprint `i64 → Guid` map, **mutable since SCRIPT3** -- MIRROR of
    /// `SimHost::entities`: `engine.spawn` adds a row and `engine.destroy` takes
    /// one away, so a script-spawned entity is addressed through the one map an
    /// actor is. (`inf_mod::tick` has taken it mutably since P14.5 for the same
    /// reason.)
    entities: &'a mut BTreeMap<i64, Uuid>,
    /// Guids `engine.destroy` removed from the world during this handler --
    /// MIRROR of `SimHost::despawned`, drained by the sim after it finishes.
    despawned: &'a mut Vec<Uuid>,
    logs: &'a mut BoundedLog<String>,
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
    /// The simulation's own voxel volumes (P21.2), read by `terrain.height_at`
    /// and **written** by the P21.4 `voxel.carve_*`/`voxel.fill_*` nodes.
    /// Borrowed from `RuntimeSim::voxels` — never from a render store, which is
    /// what keeps a camera out of a fixed step's answers, in both directions: the
    /// carve a game commits must not depend on where anyone is looking either.
    voxels: &'a mut BTreeMap<Uuid, VoxelData>,
    /// The simulation's own fracture states (P22.3), keyed by the
    /// destructible actor's `Guid` and **written** by the `destruct.*`
    /// nodes. Borrowed from the sim's own map — never from a render
    /// store, for the reason the `voxels` field above states.
    fractures: &'a mut BTreeMap<Uuid, FractureState>,
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
            // ── the `engine.*` kit (wave SCRIPT3) ────────────────────────────
            //
            // The three verbs Phase 6 registered and, until this wave, NEITHER
            // host implemented: `the_engine_namespace_is_registered_and_
            // implemented_by_neither_host` measured them falling through to the
            // unknown-call arm below, logging their path and answering `Unit`.
            //
            // One Ring-0 rule each (`inf_ecs::prefab`), so this block and the
            // other host's diff to nothing — the `door.*` arrangement, for the
            // reason P22 keeps charging for: a spawn implemented twice is two
            // implementations that agree until they do not.
            //
            // The identity of what a spawn puts in the world is folded from the
            // prefab name and the place (never a counter), and the `i64` handle
            // is folded from that identity — so both hosts hand the same program
            // the same number, and the map an actor is addressed through gains a
            // row rather than growing a second addressing scheme.
            (Some("engine"), Some("spawn")) => {
                let prefab = arg_str(args, 0);
                // At the ACTING actor's own place: the node has exactly one
                // input and it is the prefab. An author who wants a point passes
                // one to `item.spawn_pickup`, or moves the spawner.
                let at = match self.guid_of(self.current_entity) {
                    Ok(g) => inf_ecs::Vec3d::from(emitter_position(self.world, g)),
                    Err(_) => inf_ecs::Vec3d::ZERO,
                };
                let (guid, handle) = inf_ecs::prefab::spawn_prefab(self.world, &prefab, at);
                self.entities.insert(handle, guid);
                Ok(Value::Int(handle))
            }
            (Some("engine"), Some("destroy")) => {
                let id = arg_i64(args, 0);
                // The rule answers the WHOLE SUBTREE that left the world, not a
                // bool — the SCRIPT3 audit's finding. Told only "something was
                // destroyed", this host would drop the root from the session's
                // actor map and leave every destroyed CHILD actor in it, ticking
                // against a world that no longer has its entity.
                let purged = match self.guid_of(id) {
                    Ok(g) => inf_ecs::prefab::destroy_entity(self.world, g),
                    Err(_) => Vec::new(),
                };
                if purged.is_empty() {
                    // A refusal is a VALUE (P21): an id that names nothing is
                    // something an author fixes by typing, not a reason to take
                    // the rest of the handler down.
                    self.logs
                        .push(format!("engine::destroy: no entity for id {id}"));
                } else {
                    // Every handle that named any of it stops resolving, and
                    // every ACTOR in it stops with its entity. The host cannot
                    // reach the session's actor map, so it records the guids and
                    // the session drains them after this handler finishes — the
                    // statement after an `engine.destroy(var.get("entity"))`
                    // still runs, which is the containment rule (A.7) rather
                    // than an exception to it.
                    self.entities.retain(|_, g| !purged.contains(g));
                    self.despawned.extend(purged);
                }
                Ok(Value::Unit)
            }
            (Some("engine"), Some("set_rotation")) => {
                let deg = arg_f64(args, 0);
                if let Ok(g) = self.guid_of(self.current_entity) {
                    inf_ecs::prefab::set_yaw_degrees(self.world, g, deg);
                }
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
            // ── the four SCRIPT2 sky reads ──
            //
            // `is_day` and `get_hour` come off the **same** resolution the
            // atmosphere and the crowd's daily schedules use, so a script, the
            // sky and an NPC's day cannot disagree about what time it is.
            // `resolve_sky` answers `None` on a level with no sky authority, and
            // the honest reading of "is it day" there is `false` rather than a
            // guess — the `water.surface_height` precedent.
            (Some("sky"), Some("is_day")) => Ok(Value::Bool(
                inf_ecs::sky::resolve_sky(self.world).is_some_and(|s| s.is_day()),
            )),
            (Some("sky"), Some("get_hour")) => {
                Ok(Value::Float(inf_ecs::sky::local_hour(self.world)))
            }
            (Some("sky"), Some("get_cloud_coverage")) => Ok(Value::Float(
                inf_ecs::sky::resolve_sky(self.world)
                    .map(|s| f64::from(s.weather().cloud_coverage))
                    .unwrap_or(0.0),
            )),
            (Some("sky"), Some("get_fog_density")) => Ok(Value::Float(
                inf_ecs::sky::resolve_sky(self.world)
                    .map(|s| f64::from(s.weather().fog_density))
                    .unwrap_or(0.0),
            )),
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
            // voxel.* (P21.4) — RUNTIME CARVING, shared verbatim with the editor
            // SimHost so preview == shipped by construction. The three actions run
            // one Ring-0 rule (`inf_voxel::runtime_carve`) against the sim's own
            // volume map plus the shared coupling rule against the sim's own
            // heightfield; the two queries read that same map.
            //
            // Every refusal — no volume, `runtime_carve` off, degenerate shape,
            // past the per-step sample ceiling — answers **0.0** and logs one
            // shared message, because a Blueprint node is not a transaction:
            // failing the handler would take down the rest of the Tick body for an
            // op the author fixes by typing a smaller radius. `0.0` also means "I
            // carved air", which is deliberate — the two are the same fact from
            // gameplay's side (no rock moved) and the log is where they differ.
            (Some("voxel"), Some("carve_sphere")) => {
                let op = inf_voxel::VoxelOp::carve(inf_voxel::VoxelShape::Sphere {
                    center: DVec3::new(arg_f64(args, 1), arg_f64(args, 2), arg_f64(args, 3)),
                    radius_m: arg_f64(args, 4),
                });
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Float(match entity {
                    Ok(guid) => runtime_voxel_op(
                        self.world,
                        self.voxels,
                        self.logs,
                        guid,
                        &op,
                        "voxel::carve_sphere",
                    ),
                    Err(_) => 0.0,
                }))
            }
            (Some("voxel"), Some("carve_box")) => {
                let op = inf_voxel::VoxelOp::carve(inf_voxel::VoxelShape::Box {
                    center: DVec3::new(arg_f64(args, 1), arg_f64(args, 2), arg_f64(args, 3)),
                    half_extents: DVec3::new(arg_f64(args, 4), arg_f64(args, 5), arg_f64(args, 6)),
                });
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Float(match entity {
                    Ok(guid) => runtime_voxel_op(
                        self.world,
                        self.voxels,
                        self.logs,
                        guid,
                        &op,
                        "voxel::carve_box",
                    ),
                    Err(_) => 0.0,
                }))
            }
            (Some("voxel"), Some("fill_sphere")) => {
                let op = inf_voxel::VoxelOp::fill(
                    inf_voxel::VoxelShape::Sphere {
                        center: DVec3::new(arg_f64(args, 1), arg_f64(args, 2), arg_f64(args, 3)),
                        radius_m: arg_f64(args, 4),
                    },
                    clamp_material(arg_i64(args, 5)),
                );
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Float(match entity {
                    Ok(guid) => runtime_voxel_op(
                        self.world,
                        self.voxels,
                        self.logs,
                        guid,
                        &op,
                        "voxel::fill_sphere",
                    ),
                    Err(_) => 0.0,
                }))
            }
            (Some("voxel"), Some("is_solid")) => Ok(Value::Bool(voxel_is_solid(
                self.voxels,
                DVec3::new(arg_f64(args, 0), arg_f64(args, 1), arg_f64(args, 2)),
            ))),
            (Some("voxel"), Some("ground_height")) => Ok(Value::Float(
                inf_voxel::topmost_voxel_surface(self.voxels, arg_f64(args, 0), arg_f64(args, 1))
                    .unwrap_or(0.0),
            )),
            // destruct.* (P22.3) — RUNTIME DESTRUCTION, shared verbatim with the
            // editor `SimHost` so preview == shipped by construction. The two actions run
            // one Ring-0 rule (`PhysicsBridge3D::runtime_destruct` /
            // `::radial_impulse`) against the sim's own fracture map; the two
            // queries read that same map.
            //
            // Every refusal — no `Destructible`, `runtime_destruct` off, no
            // fracture data resident, a non-positive energy — answers **0.0** and
            // logs one shared message, because a Blueprint node is not a
            // transaction: failing the handler would take down the rest of the
            // Tick body for a blow the author fixes by typing a bigger number.
            // `0.0` also means "I hit it and nothing came off", which is
            // deliberate — the two are the same fact from gameplay's side (no
            // rubble) and the log is where they differ. The `voxel.*` kit answers
            // the same way for the same reason.
            (Some("destruct"), Some("apply_damage")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Float(match entity {
                    Ok(guid) => runtime_destruct_damage(
                        self.bridge3d,
                        self.fractures,
                        self.world,
                        self.logs,
                        guid,
                        arg_f64(args, 1),
                    ),
                    Err(_) => 0.0,
                }))
            }
            (Some("destruct"), Some("radial_impulse")) => {
                // NOT gated by `runtime_destruct`: this breaks nothing. It pushes
                // dynamic bodies that already exist, and an actor whose
                // destruction is switched off still has a body an explosion
                // should move.
                let hit = self.bridge3d.radial_impulse(
                    DVec3::new(arg_f64(args, 0), arg_f64(args, 1), arg_f64(args, 2)),
                    arg_f64(args, 3),
                    arg_f64(args, 4),
                );
                Ok(Value::Float(hit as f64))
            }
            (Some("destruct"), Some("is_intact")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Bool(match entity {
                    Ok(guid) => PhysicsBridge3D::is_intact(self.fractures, guid),
                    Err(_) => false,
                }))
            }
            (Some("destruct"), Some("chunk_count")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Int(match entity {
                    Ok(guid) => PhysicsBridge3D::fracture_chunk_count(self.fractures, guid) as i64,
                    Err(_) => 0,
                }))
            }
            // ── the `item.*`, `door.*` and `health.*` kit (island wave I6) ──
            //
            // Nine arms, identical in both hosts, over the Ring-0 doors in
            // `inf_ecs::item`, `inf_ecs::door` and `inf_ecs::weapon`. This is the
            // authoring surface for gameplay content, and it is a Blueprint kit
            // rather than an asset for the reason `inf_blueprint::nodekit`'s
            // `gameplay_nodes` gives: `.inf_act` bytes are the only thing that
            // reaches Simulate, a PIE payload AND a cooked pack with no schema
            // move.
            //
            // Every one of them **reports rather than fails**: a malformed
            // catalogue, an unknown id or an entity that is not there is a thing
            // an author fixes by typing, not a reason to take the whole handler
            // down with it (the P21.4 law).
            (Some("item"), Some("define")) => {
                let text = arg_str(args, 0);
                let taken = match inf_ecs::item::item_defs_mut(self.world).merge_toml(&text) {
                    Ok(n) => n as i64,
                    Err(e) => {
                        self.logs.push(format!("item::define: {e}"));
                        0
                    }
                };
                Ok(Value::Int(taken))
            }
            (Some("item"), Some("spawn_pickup")) => {
                let id = arg_str(args, 0);
                let at = inf_ecs::Vec3d::new(arg_f64(args, 1), arg_f64(args, 2), arg_f64(args, 3));
                let count = arg_i64(args, 4).clamp(0, i64::from(u32::MAX)) as u32;
                // The identity is a pure function of the id and the place, so
                // two hosts running one trace put the same entity in the same
                // spot — a spawn keyed on a counter would depend on how many
                // times the graph had run.
                let guid = inf_ecs::item::authored_pickup_guid(&id, at);
                let ok =
                    inf_ecs::item::spawn_pickup(self.world, guid, &id, count.max(1), at).is_some();
                if !ok {
                    self.logs.push(format!(
                        "item::spawn_pickup: `{id}` is not in the catalogue (call Define Items first)"
                    ));
                }
                Ok(Value::Bool(ok))
            }
            (Some("item"), Some("give")) => {
                let id = arg_str(args, 1);
                let count = arg_i64(args, 2).clamp(0, i64::from(u32::MAX)) as u32;
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Int(match entity {
                    Ok(guid) => i64::from(inf_ecs::item::give(self.world, guid, &id, count.max(1))),
                    Err(_) => i64::from(count.max(1)),
                }))
            }
            (Some("item"), Some("equip")) => {
                let id = arg_str(args, 1);
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Bool(match entity {
                    Ok(guid) => inf_physics::d3::gameplay::equip_weapon(self.world, guid, &id),
                    Err(_) => false,
                }))
            }
            (Some("item"), Some("count")) => {
                let id = arg_str(args, 1);
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Int(match entity {
                    Ok(guid) => inf_ecs::item::inventory_of(self.world, guid)
                        .map(|inv| i64::from(inv.count_of(&id)))
                        .unwrap_or(0),
                    Err(_) => 0,
                }))
            }
            (Some("door"), Some("spawn")) => {
                let text = arg_str(args, 0);
                let hung = match inf_ecs::door::spawn_doors_from_toml(self.world, &text) {
                    Ok(n) => n as i64,
                    Err(e) => {
                        self.logs.push(format!("door::spawn: {e}"));
                        0
                    }
                };
                Ok(Value::Int(hung))
            }
            (Some("door"), Some("is_open")) => {
                Ok(Value::Bool(inf_physics::d3::door::is_open_near(
                    self.world,
                    DVec3::new(arg_f64(args, 0), arg_f64(args, 1), arg_f64(args, 2)),
                )))
            }
            (Some("health"), Some("set")) => {
                let joules = arg_f64(args, 1);
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Bool(match entity {
                    Ok(guid) => inf_ecs::weapon::give_health(self.world, guid, joules),
                    Err(_) => false,
                }))
            }
            (Some("health"), Some("get")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Float(match entity {
                    Ok(guid) => inf_ecs::weapon::health_of(self.world, guid)
                        .map(|h| h.joules)
                        .unwrap_or(0.0),
                    Err(_) => 0.0,
                }))
            }
            // ── the SCRIPT2 door verbs ──
            //
            // Four verbs, **one rule**: `d3::door::nearest` decides which leaf a
            // world point is about, and `is_open`, `is_locked`, `use` and `lock`
            // all ask it. A script that tests `door.is_open(x, y, z)` and then
            // calls `door.use(x, y, z)` is talking about the same leaf because
            // there is one resolution, not four that agree by hand (P22).
            //
            // The query point doubles as the character's FEET for `lock`, which
            // is what decides the face the bolt is offered from: a script locking
            // a door is standing where it says it is.
            (Some("door"), Some("is_locked")) => {
                Ok(Value::Bool(inf_physics::d3::door::is_locked_near(
                    self.world,
                    DVec3::new(arg_f64(args, 0), arg_f64(args, 1), arg_f64(args, 2)),
                )))
            }
            (Some("door"), Some("use")) => {
                let at = DVec3::new(arg_f64(args, 0), arg_f64(args, 1), arg_f64(args, 2));
                let guid = inf_physics::d3::door::nearest(self.world, at).map(|p| p.guid);
                Ok(Value::Bool(match guid {
                    Some(g) => inf_physics::d3::door::use_door(self.world, g, at).moved(),
                    None => false,
                }))
            }
            (Some("door"), Some("lock")) => {
                let at = DVec3::new(arg_f64(args, 0), arg_f64(args, 1), arg_f64(args, 2));
                let guid = inf_physics::d3::door::nearest(self.world, at).map(|p| p.guid);
                Ok(Value::Bool(match guid {
                    Some(g) => matches!(
                        inf_physics::d3::door::lock_door(self.world, g, at),
                        inf_ecs::door::DoorVerdict::Locked
                            | inf_ecs::door::DoorVerdict::Unlocked
                            | inf_ecs::door::DoorVerdict::LockedNow
                    ),
                    None => false,
                }))
            }
            // ── the SCRIPT2 health verbs ──
            //
            // `damage` goes through `weapon::damage_entity`, the door a bullet
            // uses, so a script and a rifle spend joules identically. Downing is
            // NOT marked here: the fixed step's own pass reads `newly_dead`, and
            // doing it in two places is two places that have to agree.
            (Some("health"), Some("damage")) => {
                let joules = arg_f64(args, 1);
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Float(match entity {
                    Ok(guid) => inf_ecs::weapon::damage_entity(self.world, guid, joules)
                        .map(|r| r.absorbed_j)
                        .unwrap_or(0.0),
                    Err(_) => 0.0,
                }))
            }
            (Some("health"), Some("fraction")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Float(match entity {
                    Ok(guid) => inf_ecs::weapon::health_of(self.world, guid)
                        .map(|h| h.fraction())
                        .unwrap_or(0.0),
                    Err(_) => 0.0,
                }))
            }
            (Some("health"), Some("is_downed")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Bool(match entity {
                    Ok(guid) => inf_ecs::weapon::is_downed(self.world, guid),
                    Err(_) => false,
                }))
            }
            // ── the SCRIPT2 crowd counts ──
            //
            // Four pure reads of the NPC arc's own resources, which gameplay
            // could not reach a single number of before this wave. Counts and not
            // agents: naming an individual needs a `CrowdClock` and a spatial
            // index the crowd does not keep, and that is priced rather than
            // smuggled in behind a count.
            (Some("crowd"), Some("population")) => Ok(Value::Int(
                inf_ecs::crowd::crowd_stats(self.world).total() as i64,
            )),
            (Some("crowd"), Some("blocked")) => Ok(Value::Int(
                inf_ecs::crowd::blocked_agents(self.world).len() as i64,
            )),
            (Some("crowd"), Some("homes")) => Ok(Value::Int(
                inf_ecs::society::society_stats(self.world).homes as i64,
            )),
            (Some("crowd"), Some("workplaces")) => Ok(Value::Int(
                inf_ecs::society::society_stats(self.world).works as i64,
            )),
            // ── the SCRIPT2 zone queries ──
            //
            // The mission-class primitive, and only the primitive: an
            // axis-aligned overlap on the 3D physics world plus the
            // collider→`Guid` map beside it. `zone.count` counts **entities**,
            // so an actor with three colliders in the box is one — a
            // `BTreeSet`, which also makes the count independent of the order
            // rapier returned the colliders in.
            (Some("zone"), Some("contains")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                let inside = match entity {
                    Ok(guid) => zone_guids(self.bridge3d, args, 1).contains(&guid),
                    Err(_) => false,
                };
                Ok(Value::Bool(inside))
            }
            (Some("zone"), Some("count")) => {
                Ok(Value::Int(zone_guids(self.bridge3d, args, 0).len() as i64))
            }
            // ── the `ik.*` kit (P24.3) ──
            //
            // Five arms, identical in both hosts, over the Ring-0 doors in
            // `inf_ecs::pose`. Every action edits the entity's AUTHORED
            // `IkTarget` by goal index — see `inf_blueprint::nodekit`'s `ik_nodes`
            // for why a Blueprint cannot name a chain — and reports `ok` rather
            // than failing its handler, because a bad goal index is something the
            // author fixes by typing a smaller number and not a reason to take
            // down the rest of the Tick body (the `voxel.*` ruling).
            //
            // The two queries are what make P24.2's `IkReport` reachable from
            // gameplay at all: before them the fixed step computed a reach error
            // that only a test could see.
            (Some("ik"), Some("set_goal")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Bool(match entity {
                    Ok(guid) => inf_ecs::pose::set_authored_goal_target(
                        self.world,
                        guid,
                        arg_i64(args, 1).max(0) as usize,
                        inf_ecs::math::Vec3d::new(
                            arg_f64(args, 2),
                            arg_f64(args, 3),
                            arg_f64(args, 4),
                        ),
                    ),
                    Err(_) => false,
                }))
            }
            (Some("ik"), Some("set_goal_weight")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Bool(match entity {
                    Ok(guid) => inf_ecs::pose::set_authored_goal_weight(
                        self.world,
                        guid,
                        arg_i64(args, 1).max(0) as usize,
                        arg_f64(args, 2) as f32,
                    ),
                    Err(_) => false,
                }))
            }
            (Some("ik"), Some("enable_goal")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Bool(match entity {
                    Ok(guid) => inf_ecs::pose::set_authored_goal_enabled(
                        self.world,
                        guid,
                        arg_i64(args, 1).max(0) as usize,
                        matches!(args.get(2), Some(Value::Bool(true))),
                    ),
                    Err(_) => false,
                }))
            }
            (Some("ik"), Some("reached")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Bool(match entity {
                    Ok(guid) => inf_ecs::pose::ik_reached(self.world, guid),
                    Err(_) => false,
                }))
            }
            (Some("ik"), Some("reach_error")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Float(match entity {
                    Ok(guid) => inf_ecs::pose::ik_reach_error(self.world, guid) as f64,
                    Err(_) => 0.0,
                }))
            }
            // -- the `anim.*` kit (P29.4) --
            //
            // Four arms, identical in both hosts, over the Ring-0 doors in
            // `inf_ecs::anim_bridge`. The rule is there and only the dispatch is
            // here, which is the `ik.*` shape and the reason the kit needed no
            // change to either fixed step: `step_pose_evaluation` reads the
            // bridge itself.
            //
            // Every one reports rather than failing its handler -- an entity
            // with no `AnimStateMachine` answers `false` and the rest of the
            // Tick body runs (the `voxel.*` ruling).
            (Some("anim"), Some("set_param")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Bool(match entity {
                    Ok(guid) => inf_ecs::anim_bridge::set_anim_param(
                        self.world,
                        guid,
                        &arg_str(args, 1),
                        arg_f64(args, 2),
                    ),
                    Err(_) => false,
                }))
            }
            (Some("anim"), Some("set_trigger")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Bool(match entity {
                    Ok(guid) => {
                        inf_ecs::anim_bridge::set_anim_trigger(self.world, guid, &arg_str(args, 1))
                    }
                    Err(_) => false,
                }))
            }
            (Some("anim"), Some("query_state")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Bool(match entity {
                    Ok(guid) => {
                        inf_ecs::anim_bridge::anim_state_is(self.world, guid, &arg_str(args, 1))
                    }
                    Err(_) => false,
                }))
            }
            (Some("anim"), Some("consume_notify")) => {
                let entity = self.guid_of(arg_i64(args, 0));
                Ok(Value::Bool(match entity {
                    Ok(guid) => inf_ecs::anim_bridge::consume_anim_notify(
                        self.world,
                        guid,
                        &arg_str(args, 1),
                    ),
                    Err(_) => false,
                }))
            }
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
        let mover = inf_physics::d3::mover_for(self.world, guid);
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
        let mover = inf_physics::d3::mover_for(self.world, guid);
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

// **`build_mover3d` and `collider_shape3d` moved to Ring 0 in P29.3** as
// `inf_physics::d3::mover_for`. They were a hand-maintained byte-identical pair,
// one copy in each host, and §13's risk register says why that is not good
// enough: a host-versus-host text compare cannot see a value that is wrong in
// BOTH. The mover is now built once, and the movement component's autostep and
// slope limit ride along for free rather than needing a third edit.

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

/// **The entities whose colliders overlap an axis-aligned box** — the one rule
/// behind `zone.contains` and `zone.count`, in both hosts.
///
/// `args[base..base + 6]` is `centre_xyz` then `half_extents_xyz`, in metres. A
/// negative half-extent is folded to its magnitude, because a box written
/// backwards is a typo rather than an empty region and an empty region is the
/// answer a designer would not be able to explain.
///
/// Returns a `BTreeSet`, so the answer is entity-shaped and order-free: an actor
/// with three colliders inside the box counts once, and rapier's collider order
/// cannot reach a committed count.
fn zone_guids(
    bridge: &mut PhysicsBridge3D,
    args: &[Value],
    base: usize,
) -> std::collections::BTreeSet<Uuid> {
    let c = DVec3::new(
        arg_f64(args, base),
        arg_f64(args, base + 1),
        arg_f64(args, base + 2),
    );
    let h = DVec3::new(
        arg_f64(args, base + 3).abs(),
        arg_f64(args, base + 4).abs(),
        arg_f64(args, base + 5).abs(),
    );
    if !c.is_finite() || !h.is_finite() {
        return std::collections::BTreeSet::new();
    }
    bridge
        .world_mut()
        .intersect_aabb(c - h, c + h)
        .into_iter()
        .filter_map(|id| bridge.guid_of_collider(id))
        .collect()
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

/// The **ground** height at world `(x, z)` — the `terrain.height_at` host seam
/// (P11.4), byte-for-byte the editor `SimHost`'s (preview == shipped). A
/// heightfield carries no physics collider, so a 3D character reads its height
/// here to stay grounded.
///
/// **POSITION-AWARE since the island phase (IB-15).** It used to pick the
/// lowest-`Guid` non-empty terrain and sample only that one, with no test of
/// whether it covered `(x, z)` — so over a second terrain it read `None` and a
/// character walking across the border fell to the `0.0` below. Every non-empty
/// terrain now goes to the Ring-0 rule with its origin and the topmost surface
/// that answers wins; ties go to the first in `Guid` order, which both hosts sort
/// by, so the answer is a function of the level and not of an archetype walk.
///
/// Only entities + origins are remembered during the scan, never a clone of a
/// (multi-MB) heightfield; the components are re-fetched afterwards (all
/// `EntityRef` borrows released) and sampled in place.
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
fn terrain_height_at(
    world: &mut EcsWorld,
    voxels: &BTreeMap<Uuid, VoxelData>,
    x: f64,
    z: f64,
) -> f64 {
    // ARCHETYPE-SCOPED, not a whole-world walk (Hardening Wave E). This is the
    // host seam behind a node designed to be called from actor `Tick` handlers,
    // so its cost is multiplied by the actor count on every fixed step — and it
    // used to visit **every entity in the world**, with two component lookups
    // each, to find the one or two that carry a `Terrain`. Measured on this
    // machine: 0.0137 ms/call over 1 000 entities, 0.0668 over 5 000, 0.2032
    // over 15 000 — linear in the world, for a query whose answer set is one.
    //
    // The answer is now POSITION-AWARE (island phase, IB-15). It used to be "the
    // lowest `Guid` among non-empty terrains", with no test of whether that
    // terrain covers `(x, z)` at all — so over a SECOND terrain the query sampled
    // the first one outside its authored extent, got `None`, and fell through to
    // the `0.0` below. A character walking from one terrain onto another dropped
    // to sea level at the border, and no committed scene placed two terrains
    // within a kilometre of each other, so nothing saw it. Every non-empty
    // terrain is now handed to the Ring-0 rule, which takes the topmost surface
    // that answers — see `inf_voxel::ground_height_at`.
    let mut query = world.world_mut().query::<(
        Entity,
        &Guid,
        &Terrain,
        Option<&GlobalTransform>,
        Option<&Transform>,
    )>();
    let mut found: Vec<(Uuid, Entity, DVec3)> = Vec::new();
    let w = world.world();
    for (entity, guid, t, global, local) in query.iter(w) {
        if t.data.is_empty() {
            continue;
        }
        let origin = global
            .map(|g| g.translation())
            .or_else(|| local.map(|t| t.translation.to_dvec3()))
            .unwrap_or(DVec3::ZERO);
        found.push((guid.0, entity, origin));
    }
    // `Guid` order, so the rule's tie-break is a function of the level rather
    // than of a bevy archetype walk.
    found.sort_unstable_by_key(|(g, _, _)| *g);
    let terrains: Vec<(&inf_ecs::TerrainData, DVec3)> = found
        .iter()
        .filter_map(|(_, e, o)| w.get::<Terrain>(*e).map(|t| (&t.data, *o)))
        .collect();
    // No terrain at all is not "no ground": a level may be nothing but caves,
    // and the voxel half still answers.
    inf_voxel::ground_height_at(&terrains, voxels, x, z).unwrap_or(0.0)
}

/// **One gameplay carve or fill** (P21.4) — the voxel half through the shared
/// Ring-0 rule, then the heightfield half through the shared coupling rule.
/// Returns the volume moved in **cubic metres**, `0.0` for every refusal.
///
/// Mirrored character-for-character in `inf_editor_core::simulate`, for the reason
/// `terrain_height_at` above is: a preview that dug a different hole from the
/// shipped build is a bug no compiler and no screenshot finds.
///
/// # The heightfield half, and what "sim-local" means
///
/// A game digging through the surface must open a mouth, or a player walks into a
/// cave that is not there. So a carve that reaches a terrain's height samples runs
/// [`inf_voxel::apply_surface_cut`] — the *same* exactly-invertible rule the
/// editor's carve brush runs — over **every** terrain in the world, in `Guid`
/// order.
///
/// That opens the mouth for **gameplay** (`terrain.height_at` stops answering, the
/// combined query falls to the cave floor, the physics bridge rebuilds the chunk
/// colliders). Making it *visible* is a second seam and was missing from the first
/// cut of P21.4: on an asset-backed terrain — the only kind that can carry a hole
/// mask, and therefore the configuration every carved level ships in — the render
/// side streams its own tiles out of the `.inf_terrain` and never saw this edit.
/// `TerrainStreaming::overlay_sim_edits` (player) pins the dirty tiles into the
/// render streamer, and `VoxelVolumes::overlay_sim` does the same for the chunks;
/// both run `sim → render` only.
///
/// The difference from the editor is that **nothing here is persisted, and that
/// changes which refusals apply.** The editor refuses to carve an *inline* terrain
/// (`CarveRefusal::InlineTerrain`) because scene schema v19 cannot carry a hole
/// mask, so saving would seal every mouth the author dug — the refusal protects a
/// document. A fixed step writes no document: the editor's Simulate world is a
/// `ScenePersist::Memory` snapshot and the player's is a loaded pack, and both die
/// with the session. So a runtime carve is allowed on any terrain, inline or
/// streamed, and the hole lives exactly as long as the play session does. A game
/// that wants craters to survive a reload needs a save system, which is not this
/// phase and is not silently half-built here.
fn runtime_voxel_op(
    world: &mut EcsWorld,
    voxels: &mut BTreeMap<Uuid, VoxelData>,
    logs: &mut BoundedLog<String>,
    entity: Uuid,
    op: &inf_voxel::VoxelOp,
    op_name: &str,
) -> f64 {
    // The component decides permission; a missing one is "no volume" rather than
    // "not permitted", because the two read very differently in a log.
    let flag = world
        .entity_of(entity)
        .and_then(|e| world.world().get::<VoxelVolume>(e))
        .map(|v| v.runtime_carve);
    let Some(permitted) = flag else {
        logs.push(
            inf_voxel::RuntimeCarveOutcome::NoVolume
                .refusal(op_name)
                .expect("NoVolume is a refusal"),
        );
        return 0.0;
    };
    let report = inf_voxel::runtime_carve(voxels, &entity, permitted, op);
    if let Some(msg) = report.outcome.refusal(op_name) {
        logs.push(msg);
        return 0.0;
    }
    let voxel_size_m = voxels
        .get(&entity)
        .map(|d| d.voxel_size_m())
        .unwrap_or(1.0_f64);

    // The heightfield half. Only when the field actually moved: an op that hit
    // nothing but air has nothing to open, and re-running the coupling would be a
    // no-op anyway (it is a pure function of the shape and the tile grid) — this
    // just declines to walk the sample grid for it.
    if report.touched > 0 {
        let open = matches!(op.kind, inf_voxel::VoxelOpKind::Carve);
        // Collect first: the walk borrows the world immutably and the cut needs it
        // mutably. `Guid` order, so two runs touch the tiles in one sequence.
        let mut terrains: Vec<(Uuid, DVec3)> = world
            .world()
            .iter_entities()
            .filter_map(|e| {
                let guid = e.get::<Guid>().map(|g| g.0)?;
                let t = e.get::<Terrain>()?;
                if t.data.is_empty() {
                    return None;
                }
                let origin = e
                    .get::<GlobalTransform>()
                    .map(|g| g.translation())
                    .or_else(|| e.get::<Transform>().map(|t| t.translation.to_dvec3()))
                    .unwrap_or(DVec3::ZERO);
                Some((guid, origin))
            })
            .collect();
        terrains.sort_by_key(|(g, _)| *g);
        for (guid, origin) in terrains {
            let Some(e) = world.entity_of(guid) else {
                continue;
            };
            if let Some(mut t) = world.world_mut().get_mut::<Terrain>(e) {
                inf_voxel::apply_surface_cut(&mut t.data, origin, &op.shape, open);
            }
        }
    }

    match op.kind {
        inf_voxel::VoxelOpKind::Carve => report.removed_m3(voxel_size_m),
        inf_voxel::VoxelOpKind::Fill { .. } => report.added_m3(voxel_size_m),
    }
}

/// Whether any of the sim's volumes has rock at a world point (`voxel.is_solid`).
/// Shared with the editor twin; deterministic (`BTreeMap` order, and the answer
/// The `AudioSource` and world position of a destroyed actor, or `None` when it
/// carries no emitter (P22.3). Shared shape with the runtime twin.
fn destroyed_emitter(world: &EcsWorld, entity: Uuid) -> Option<(AudioSource, DVec3)> {
    let e = world.entity_of(entity)?;
    let src = world.world().get::<AudioSource>(e)?.clone();
    let pos = world
        .world()
        .get::<GlobalTransform>(e)
        .map(|g| g.translation())
        .or_else(|| {
            world
                .world()
                .get::<Transform>(e)
                .map(|t| t.translation.to_dvec3())
        })
        .unwrap_or(DVec3::ZERO);
    Some((src, pos))
}

/// **One gameplay blow** (P22.3) — the destruction half through the shared
/// Ring-0 rule. Returns the energy **absorbed** in joules, `0.0` for every
/// refusal.
///
/// Mirrored character-for-character in `inf_editor_core::simulate`, for the reason
/// `terrain_height_at` and `runtime_voxel_op` are: a preview that broke a
/// different wall from the shipped build is a bug no compiler and no screenshot
/// finds.
///
/// The component decides permission and a missing one is "nothing to break"
/// rather than "not permitted", because the two read very differently in a log —
/// the `runtime_voxel_op` shape exactly.
fn runtime_destruct_damage(
    bridge3d: &mut PhysicsBridge3D,
    fractures: &mut BTreeMap<Uuid, FractureState>,
    world: &EcsWorld,
    logs: &mut BoundedLog<String>,
    entity: Uuid,
    energy_j: f64,
) -> f64 {
    let flag = world
        .entity_of(entity)
        .and_then(|e| world.world().get::<Destructible>(e))
        .map(|d| d.runtime_destruct);
    let Some(permitted) = flag else {
        logs.push(
            DestructOutcome::NoDestructible
                .refusal("destruct::apply_damage")
                .expect("NoDestructible is a refusal"),
        );
        return 0.0;
    };
    let report = bridge3d.runtime_destruct(fractures, entity, permitted, energy_j);
    if let Some(msg) = report.outcome.refusal("destruct::apply_damage") {
        logs.push(msg);
        return 0.0;
    }
    // **A NEAR MISS IS REPORTED** (P22.4 audit). `Applied` with zero joules
    // absorbed is a blow that could not break the cheapest bond set — damage is
    // not banked, so the energy is simply not spent. From outside it is
    // indistinguishable from a script that never ran: the level looks untouched
    // and the node returns a legal `0.0`. The phase-22 sample lost half a day to
    // exactly that (25 000 J against a 25 136 J chunk), so it says so, on the same
    // refusal-visibility principle the outcomes above are built on.
    if report.detached == 0 && report.energy_absorbed_j == 0.0 {
        logs.push(format!(
            "destruct::apply_damage: {energy_j} J was not enough to break the \
             cheapest bond set on that actor, so NOTHING was spent (damage is not \
             banked — a bigger single blow, not more small ones)"
        ));
    }
    report.energy_absorbed_j
}

/// is an `any`, so order cannot change it).
fn voxel_is_solid(voxels: &BTreeMap<Uuid, VoxelData>, p: DVec3) -> bool {
    voxels.values().any(|d| d.is_solid_at(p))
}

/// Clamp a Blueprint's `material` pin into the four splat layers a voxel sample
/// can carry. Out of range **saturates** rather than erroring or wrapping: a `7`
/// typed into a node is an author mistake, and filling with the last layer is a
/// visible wrong answer while wrapping to layer 3 would be an invisible one.
/// Shared with the editor twin.
fn clamp_material(m: i64) -> u8 {
    m.clamp(0, inf_voxel::MATERIAL_COUNT as i64 - 1) as u8
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
// P24.1: `to_anim_runtime` / `from_anim_runtime` used to live here, spelled
// identically in this file and in the editor's `simulate.rs` — a hand-maintained
// mirror pair for a struct-to-struct field copy. They are now
// `inf_ecs::pose::{to_anim_runtime, from_anim_runtime}`, called by the one Ring-0
// fixed-step rule both hosts share. `SmRuntimeState` itself stays a POD mirror:
// the component derives `Reflect` + serde and `inf_anim::SmRuntime` derives
// neither.

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
