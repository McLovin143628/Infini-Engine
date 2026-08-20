//! In-editor **Simulate** (P8.4): tick Blueprints + 2D physics together over the
//! live [`SceneDoc`], so a platformer plays in the viewport via the interpreter
//! (the Phase-8 gate) — no subprocess, no cook, no compiled dylib.
//!
//! # The tick (deterministic, fixed `dt`)
//!
//! [`SimSession`] drives a [`FixedStepper`]; each fixed step runs, **in this
//! order** (§2.5 determinism — every entity/actor pass is in `Guid` order):
//!
//! 1. `bridge.sync_from_world` — mirror the ECS `RigidBody2D`/`Collider2D`/
//!    `Transform`s into the rapier world (static/kinematic follow their
//!    Transform; dynamic is solver-owned).
//! 2. **Blueprint tick** — fire each actor's `Tick` handler through the
//!    interpreter. The engine [`Host`] is [`SimHost`]; its `physics()` accessor
//!    is a real [`Physics2dHost`] over the [`PhysicsBridge2D`], so a
//!    `physics2d.move_and_slide` node kinematically drives a
//!    `CharacterController2D` entity via [`CharacterMover2D`].
//! 3. `bridge.step(fixed_dt)` — advance the rapier solver (dynamic bodies).
//! 4. `bridge.write_back` — dynamic poses → ECS `Transform`s; then propagate.
//!
//! # Enter / exit
//!
//! [`SimSession::enter`] snapshots the world (the deterministic `.inf_lvl`
//! [`SceneFile`], the same bytes save/load uses), fires every actor's
//! `BeginPlay`, and returns the live session. [`SimSession::exit`] restores the
//! snapshot exactly — Simulate never leaves an edit behind.
//!
//! # Entity identity across the layers
//!
//! Blueprints address entities as opaque `i64`s (the IR has no `Guid` value).
//! The session assigns each actor a stable `i64` in `Guid` order and seeds its
//! `entity` member variable with it, so `var.get("entity")` feeds
//! `physics2d.*` nodes. [`SimHost`]/the adapter map `i64 → Guid → body handle`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use glam::{DVec2, DVec3};
use uuid::Uuid;

// P11.3 root motion: the pure root-delta extractor + the clip/skeleton it reads.
use inf_anim::state_machine::StateMachine;
use inf_anim::{root_delta, AnimClip, AnimClipAsset, Skeleton, SkeletonAsset, StateMachineAsset};
use inf_audio::{
    Attenuation, AttenuationModel, AudioAsset, AudioCommand, AudioEngine, Listener, PlayCommand,
};
use inf_blueprint::interp::{
    AudioHost, MoveResult2d, MoveResult3d, Physics2dHost, Physics3dHost, RayHit2d, RayHit3d,
};
use inf_blueprint::semantics::run_event;
use inf_blueprint::{
    ActorInstance, BlueprintClass, EventKind, Host, InterpDebug, RunError, Trace, Value,
};
use inf_core::BoundedLog;
use inf_ecs::components::{
    AnimPlayer, AnimStateMachine, AudioListener, AudioSource, CharacterController2D,
    CharacterController3D, Collider2D, ColliderShape2DKind, Destructible, DistanceModel,
    GlobalTransform, RootMotion, RootMotionMode, SkeletalMesh, Terrain, Transform, VoxelVolume,
};
use inf_ecs::{update_attachments, EcsWorld, Entity, Guid};
use inf_physics::d3::{
    DebrisBudget, DestroyedEvent, DestructOutcome, FractureAudit, FractureState, WaterEventKind3D,
};
use inf_physics::{
    CharacterMover2D, ColliderShape2D, ContactPhase, FixedStepper, PhysicsBridge2D,
    PhysicsBridge3D, WorldGravity,
};
use inf_voxel::VoxelData;

use crate::scene::serialize::{apply_to_doc, to_scene_file_for, SceneFile, ScenePersist};
use crate::scene::SceneDoc;

/// The default Simulate/physics rate (fixed updates per second).
pub const SIM_HZ: f64 = 60.0;

/// The per-step cap on chained event dispatches (Wave 3). Once this many
/// `dispatch.call`s have been processed in one drain, the remainder is dropped
/// with a log line — a deterministic guard against an unbounded dispatch cycle
/// (the event-graph analogue of the `flow.while` loop guard).
const DISPATCH_ROUND_CAP: u32 = 64;

/// Whether a trigger-volume (sensor) overlap began or ended this step — the
/// gameplay-facing phase of a drained [`OverlapEvent`] (Wave 3, E-P4 sim half).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OverlapPhase {
    /// The two sensor colliders started overlapping this step.
    Begin,
    /// They stopped overlapping this step.
    End,
}

/// A drained **sensor-pair** overlap for one fixed step: the two entity `Guid`s
/// (canonical `a < b`) and the phase. This is the seam trigger-volume gameplay +
/// the parity tests pin — [`SimSession::drained_overlaps`] exposes the list, which
/// is rebuilt (cleared then filled) every fixed step. Only sensor↔sensor and
/// sensor↔solid pairs where at least one collider is a sensor appear here; solid
/// contacts drive Blueprint `Collision` events instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct OverlapEvent {
    /// The lower-`Guid` entity of the overlapping pair.
    pub a: Uuid,
    /// The higher-`Guid` entity of the overlapping pair.
    pub b: Uuid,
    /// Begin (started overlapping) or End (stopped).
    pub phase: OverlapPhase,
}

/// A snapshot of the keyboard/action state for one tick: the set of currently
/// **held** keys/actions (e.g. `"left"`, `"jump"`) plus this tick's resolved
/// **analog axes**. Rising edges (`just_pressed`) are derived by the session
/// from the previous tick's set.
///
/// The axes are P29.3 and the MIRROR of `RuntimeInput`'s — see that type for why
/// a movement component that owns velocity cannot be driven by a set of names,
/// and for the delta-axes-arrive-as-rates rule
/// (`inf_input::InputState::axis_snapshot`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SimInput {
    down: BTreeSet<String>,
    axes: BTreeMap<String, f32>,
}

impl SimInput {
    /// An input state with the given keys/actions held down and no axes.
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

/// One actor's live class + instance state under Simulate.
struct ActorState {
    class: BlueprintClass,
    instance: ActorInstance,
}

/// A resolved `.inf_anim` clip + the skeleton it animates, registered for
/// **root-motion** extraction (P11.3). The sim can't reach the asset DB, so the
/// caller (the editor Simulate wiring) seeds these via
/// [`SimSession::register_root_motion_clip`]; a root-motion entity whose
/// [`AnimPlayer`] clip GUID isn't registered simply skips root motion this run.
struct RootClip {
    skeleton: Skeleton,
    clip: AnimClip,
}

/// The Simulate session's fracture states, shared with the viewport (P22.3).
///
/// **The editor twin of the sim→render fold P21.4 built for carves**, and it
/// exists for the same reason: a `destruct.*` node writes the *session's* state,
/// which the viewport cannot see, so without a channel a Blueprint would break a
/// wall, the colliders would swap, the shipped player would draw rubble — and the
/// editor would keep drawing the wall.
///
/// A **publish**, not a shared owner: Ring 2 copies the session's map in after
/// each tick and clears it when Simulate stops, so the authoritative state stays
/// where the fixed step can reach it and the viewport reads a snapshot. Lock
/// order everywhere: **document, then this** — the `voxel_store` rule.
pub type SharedFractures =
    std::sync::Arc<std::sync::Mutex<BTreeMap<Uuid, inf_physics::d3::FractureState>>>;

/// A fresh [`SharedFractures`] — the one constructor, so nobody has to spell the
/// `Arc<Mutex<…>>` out and no second wrapper shape can appear.
pub fn shared_fractures() -> SharedFractures {
    std::sync::Arc::new(std::sync::Mutex::new(BTreeMap::new()))
}

/// A live in-editor Simulate session over one [`SceneDoc`].
pub struct SimSession {
    bridge: PhysicsBridge2D,
    /// The 3D physics bridge (P11.3), driven alongside the 2D one so
    /// `physics3d.*` nodes + root-motion movers reach a rapier3d world.
    bridge3d: PhysicsBridge3D,
    /// Clips resolvable for root motion, keyed by `.inf_anim` GUID (P11.3).
    clips: BTreeMap<Uuid, RootClip>,
    stepper: FixedStepper,
    /// Actors keyed by `Guid` (deterministic iteration).
    actors: BTreeMap<Uuid, ActorState>,
    /// Blueprint `i64` entity id → its `Guid`.
    entities: BTreeMap<i64, Uuid>,
    /// The world state captured at [`enter`](Self::enter), restored on
    /// [`exit`](Self::exit).
    snapshot: SceneFile,
    /// Resolvable `.inf_sm` state machines keyed by asset GUID (P11.2). Empty by
    /// default; seed via [`set_state_machines`](Self::set_state_machines). Entities
    /// with an [`AnimStateMachine`] whose `sm` GUID resolves here are stepped each
    /// fixed tick.
    state_machines: BTreeMap<Uuid, StateMachine>,
    /// Resolvable `.inf_skel` assets keyed by asset GUID (P24.1) — the skeleton a
    /// machine-driven entity's [`SkeletalMesh`] names, plus its authored sockets.
    /// Seeded via [`set_skeletons`](Self::set_skeletons); an entity whose skeleton
    /// is absent still steps its machine and publishes no pose.
    skeletons: BTreeMap<Uuid, inf_anim::SkeletonAsset>,
    /// Resolvable `.inf_anim` clips a state machine's states **play**, keyed by
    /// asset GUID (P24.1). Distinct from [`clips`](Self::clips), which keys a clip
    /// *with the skeleton it animates* because `root_delta` needs both: the pose
    /// path takes its skeleton from the entity, so it needs the clip alone. The
    /// two overlap only when one clip is both an `AnimPlayer` clip on a
    /// root-motion entity and a machine state's motion.
    pose_clips: BTreeMap<Uuid, AnimClip>,
    /// Resolvable `.inf_cloth` garments keyed by asset GUID (P24.4) — the editor
    /// mirror of `RuntimeSim::cloths`. An entity whose `ClothSim.asset` resolves
    /// here simulates; one whose does not keeps its component and simulates
    /// nothing (`inf_ecs::cloth`'s rule 2).
    cloths: BTreeMap<Uuid, inf_anim::ClothAsset>,
    /// Resolvable `.inf_hair` hairstyles keyed by asset GUID (P24.4) - the editor
    /// mirror of `RuntimeSim::hairs`.
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
    /// **The locomotion camera** (P29.6). Owned by the session — never a
    /// component, never a resource, never serialized (Ruling 4) — and stepped at
    /// the very end of the fixed step through the same Ring-0 door the shipped
    /// player calls, so PIE and shipping frame the same character identically.
    camera: inf_ecs::camera::LocomotionCamera,
    /// The character the camera follows, re-resolved every step; `None` on every
    /// level with no player-controlled character.
    camera_subject: Option<Uuid>,
    /// **Tuning edits queued and not yet applied** (P29.5, pillar S4).
    ///
    /// Drained at the very top of [`fixed_step`](Self::fixed_step), before
    /// anything reads anything, so a tuned run is still a deterministic sequence
    /// of fixed steps rather than a run with an edit somewhere inside one.
    pending_tunes: Vec<(crate::tuning::Tune, crate::tuning::TuneScope)>,
    /// The [`TuneScope::Keep`](crate::tuning::TuneScope::Keep) tunes this session
    /// has applied, replayed onto the document by [`exit`](Self::exit) after the
    /// snapshot restore.
    kept_tunes: Vec<crate::tuning::Tune>,
    /// Currently-held keys/actions.
    input: SimInput,
    /// Keys/actions held the previous tick (for rising-edge detection).
    prev_down: BTreeSet<String>,
    /// Rising edges pending this fixed step (consumed after the first step).
    just_pressed: BTreeSet<String>,
    /// Falling edges pending this fixed step (Wave 3 input events): actions
    /// released since the previous tick (`prev_down − down`).
    just_released: BTreeSet<String>,
    /// Wave 3 event dispatchers: `(source entity, event name) → {listener entity
    /// → handler custom-event name}`. Deterministic (`BTreeMap` throughout).
    bindings: BTreeMap<(i64, String), BTreeMap<i64, String>>,
    /// FIFO queue of pending `(target entity, event name)` dispatches, drained
    /// after each event pass (Wave 3).
    dispatch_queue: VecDeque<(i64, String)>,
    /// Sensor-pair overlaps drained this fixed step (canonical `a < b`, sorted).
    /// Cleared and refilled every step; the trigger-volume gameplay seam.
    drained_overlaps: Vec<OverlapEvent>,
    /// Accumulated `debug.print` output (surfaced to the log panel).
    ///
    /// **Bounded** (Hardening D), for the reason `RuntimeSim::logs` is: a failed
    /// event dispatch pushes a formatted `String` every tick, so the growth rate
    /// is reachable from authored content. The sibling `debug_events` beside it
    /// has always been *drained* (`take_debug_events`); this one is read as a
    /// borrow by every caller, so it rings instead. See [`BoundedLog`].
    logs: BoundedLog<String>,
    /// Last `move_and_slide` grounded result per actor (a debug/telemetry read).
    grounded: BTreeMap<Uuid, bool>,
    /// P12.3 audio: the long-lived host `AudioEngine` (a no-device fallback in the
    /// editor/CI — output-only, non-sim state). Sim systems never touch it
    /// directly; they enqueue `audio_cmds`, drained here after each step.
    audio: AudioEngine,
    /// Resolvable `.inf_audio` payloads keyed by asset GUID (seeded via
    /// [`set_audio_clips`](Self::set_audio_clips)). Decoded lazily on first play.
    audio_clips: BTreeMap<Uuid, AudioAsset>,
    /// The audio command queue: filled by autoplay + Blueprint `audio.*` nodes
    /// during a step, drained into `audio` at the end (the determinism seam).
    audio_cmds: Vec<AudioCommand>,
    /// Entity `Guid`s whose autoplay `AudioSource` has already started (so autoplay
    /// enqueues exactly once, not every tick).
    audio_started: BTreeSet<Uuid>,
    /// Accumulated drained audio command stream (P12.3 determinism telemetry): the
    /// exact play/stop/set sequence, the observable a headless test asserts against
    /// (the deterministic-queue payoff — the command stream, not device output).
    ///
    /// **Bounded** (Hardening D): at least one listener command is enqueued per
    /// fixed step, so this grew for the whole session at the step rate.
    audio_log: BoundedLog<AudioCommand>,
    /// Total fixed steps run (a determinism/telemetry counter).
    steps: u64,
    /// B-P4 tier A′ seam: per-class debugger config (breakpoints + wire capture),
    /// keyed by [`BlueprintClass`] id. Empty by default — the shipped player never
    /// populates this (its `RuntimeSim` has no debug setter), so Simulate is the
    /// only path that can pause/inspect. Seeded via [`set_debug`](Self::set_debug).
    debug: BTreeMap<String, InterpDebug>,
    /// Debug hits/wire-values collected this step across every event pass, drained
    /// by [`take_debug_events`](Self::take_debug_events) after a step. Non-empty
    /// only when a class in `debug` carries breakpoints or wire capture.
    debug_events: Vec<SimDebugHit>,
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

/// One handler's debug observation under Simulate (B-P4 tier A′): the actor's
/// class + event, the handler fn, the breakpoint hits, and the captured wire
/// values.
///
/// **Honest limitation** — because Simulate runs hand-built `.inf_act` IR (there
/// is no graph→`.inf_act` authoring pipeline yet), these are keyed by the IR
/// `LocalId` (`local`), not a canvas `NodeId`. The `NodeId` half activates the
/// moment classes carry `LowerMap` graph provenance; until then the editor shows
/// the raw locals. The pause-on-hit wiring already works (any non-empty `hits`
/// pauses); only the node-level highlight awaits provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct SimDebugHit {
    /// The actor class id whose handler produced these observations.
    pub class_id: String,
    /// The event key that fired (`begin_play`, `tick`, `input:jump`, …).
    pub event: String,
    /// The handler's function name.
    pub fn_name: String,
    /// Breakpoint hits, as IR `LocalId` values, in order hit.
    pub hits: Vec<u32>,
    /// Latest captured value per wire, as `(LocalId, stringified value)`.
    pub wires: Vec<(u32, String)>,
}

/// The obstruction gain (linear) applied to an occluded spatial source — a −12 dB
/// cut (P12.3). Configurable constant; the sim's one raycast decides clear vs. cut.
const OCCLUSION_CUT_LINEAR: f64 = 0.251_188_643_150_958; // 10^(-12/20)

impl SimSession {
    /// Enter Simulate: snapshot the world, mirror it into a fresh physics world,
    /// seed each actor's `entity` variable, and fire every `BeginPlay`.
    ///
    /// `actors` pairs each blueprint-driven entity's `Guid` with the
    /// [`BlueprintClass`] to run on it. `gravity` is world units/s² (a
    /// side-scroller uses `(0, -9.81)`).
    ///
    /// **A fixture's door.** One 2D vector means that vector in both dimensions
    /// ([`WorldGravity::from_2d`]), which is what every caller of this function
    /// meant before P29.7. The **editor** has a document with two authored
    /// fields and calls [`enter_with_gravity`](Self::enter_with_gravity), which
    /// is the mirror of `RuntimeSim::with_gravity`.
    pub fn enter(
        doc: &mut SceneDoc,
        actors: Vec<(Uuid, BlueprintClass)>,
        gravity: DVec2,
        hz: f64,
    ) -> Self {
        Self::enter_with_gravity(doc, actors, WorldGravity::from_2d(gravity), hz)
    }

    /// **The document's own gravity**, by the same rule the shipped player reads
    /// a level with (P29.7).
    ///
    /// # The finding this closes
    ///
    /// The editor's Simulate passed `DVec2::ZERO` — a literal, in
    /// `commands/sim.rs`, under a comment saying a character applies its own
    /// gravity in its blueprint. That is true of a *character* and of nothing
    /// else: a dynamic body in the editor's Simulate therefore floated while the
    /// same level in the shipped player fell, and no gate could see it because
    /// until this wave no committed level with a dynamic body was ever played in
    /// both hosts. It is the same defect shape as the one `sim_from_payload`
    /// exists for — a boot path that forgets something does not crash, it agrees
    /// with itself.
    ///
    /// One function so the studio command, the gates and any test read the
    /// document the same way; `inf_player::level`'s two lines are the mirror.
    pub fn gravity_of(doc: &SceneDoc) -> WorldGravity {
        let s = doc.settings();
        WorldGravity::new(
            glam::DVec2::new(s.gravity_2d.x, s.gravity_2d.y),
            s.gravity_3d.to_dvec3(),
        )
    }

    /// [`enter`](Self::enter) with **both** solvers' gravity — the door the
    /// editor uses (P29.7).
    ///
    /// The 3D bridge is built from `gravity.d3`, i.e. from the document's
    /// authored `gravity_3d`. Before this wave it came from `gravity_2d.y` and
    /// `gravity_3d` was read by nothing; [`WorldGravity`] carries the finding and
    /// the decision, and this function is the editor half of the pair the shipped
    /// player's `RuntimeSim::with_gravity` is the other half of.
    pub fn enter_with_gravity(
        doc: &mut SceneDoc,
        actors: Vec<(Uuid, BlueprintClass)>,
        gravity: WorldGravity,
        hz: f64,
    ) -> Self {
        // ScenePersist::Memory, NOT the file projection (P16.4b): this snapshot
        // is applied straight back onto the same document on `exit`, so it must
        // carry a streamed terrain's unsaved working set and write-back marks.
        // The file projection strips them — Play → Stop would then delete every
        // unsaved terrain edit and leave the undo stack replaying height deltas
        // into tiles `revert_delta` recreates flat.
        let snapshot = to_scene_file_for(doc, ScenePersist::Memory);
        // ── P22.1 ── Simulate starts on UNDEFORMED ground. The deformation field
        //    is a bevy *resource*, and the `ScenePersist::Memory` snapshot above
        //    captures entities and components — resources are outside it by
        //    construction, so nothing restores this on `exit` and nothing would
        //    clear it on `enter` either. Both ends are therefore explicit:
        //    without the clear here, run 2 of a Simulate session would begin on
        //    run 1's footprints and its trace would not match the shipped
        //    player's, which starts from nothing every time.
        inf_ecs::deform::clear_deformation(doc.world_mut());
        // ── P24.1 ── …and on an UNPOSED character, for exactly the same reason:
        //    the evaluated-pose store is a resource too, so nothing above
        //    captures it and nothing below would clear it.
        inf_ecs::pose::clear_poses(doc.world_mut());
        // ── P24.4 ── …and on an UNFOLDED garment, third time the same reason.
        //    Without it run 2 would start from run 1's settled coat and its trace
        //    would not match the shipped player's, which seeds from rest.
        inf_ecs::cloth::clear_cloth(doc.world_mut());
        inf_ecs::hair::clear_hair(doc.world_mut());
        let bridge = PhysicsBridge2D::new(gravity.d2);
        // P11.3: a 3D bridge alongside the 2D one — built from the level's own
        // `gravity_3d` since P29.7 (a character still applies its own gravity
        // through `move_and_slide`; only a DYNAMIC body reads this number).
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

        let mut session = Self {
            bridge,
            bridge3d,
            clips: BTreeMap::new(),
            stepper: FixedStepper::from_hz(hz),
            actors: states,
            entities,
            snapshot,
            state_machines: BTreeMap::new(),
            skeletons: BTreeMap::new(),
            pose_clips: BTreeMap::new(),
            cloths: BTreeMap::new(),
            hairs: BTreeMap::new(),
            hair_detail: inf_anim::HairDetail::GUIDES,
            camera: inf_ecs::camera::LocomotionCamera::default(),
            camera_subject: None,
            pending_tunes: Vec::new(),
            kept_tunes: Vec::new(),
            input: SimInput::default(),
            prev_down: BTreeSet::new(),
            just_pressed: BTreeSet::new(),
            just_released: BTreeSet::new(),
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
            debug: BTreeMap::new(),
            debug_events: Vec::new(),
            fractures: BTreeMap::new(),
            debris_budget: DebrisBudget::default(),
            fracture_audit: FractureAudit::default(),
            voxels: BTreeMap::new(),
        };

        session.bridge.sync_from_world(doc.world());
        // P11.3 3D bridge. The voxel map AND the P22.3 fracture map are still
        // empty here — the caller seeds
        // it after `enter` returns — so a volume's chunk colliders arrive on the
        // first step's sync instead. That is also why a `BeginPlay` handler cannot
        // see a cave (see the P21.4 kit docs): the map it would read does not
        // exist yet.
        session
            .bridge3d
            .sync_from_world_sim(doc.world(), &session.voxels, &session.fractures);
        session.run_all(doc, &EventKind::BeginPlay);
        session.drain_dispatch(doc); // Wave 3: BeginPlay may dispatch custom events.
        session
    }

    /// Register a resolvable `.inf_audio` clip payload by asset GUID (P12.3), so
    /// an [`AudioSource`] referencing it can play. The editor Simulate wiring seeds
    /// these from the loaded assets; unregistered clips are a silent no-op.
    /// Idempotent.
    pub fn register_audio_clip(&mut self, clip_guid: Uuid, clip: AudioAsset) {
        self.audio_clips.insert(clip_guid, clip);
    }

    /// Seed the resolvable `.inf_audio` payloads in bulk (mirrors
    /// [`set_state_machines`](Self::set_state_machines)).
    pub fn set_audio_clips(&mut self, clips: BTreeMap<Uuid, AudioAsset>) {
        self.audio_clips = clips;
    }

    /// Install a named-bus [`MixerConfig`](inf_audio::MixerConfig) on the audio
    /// engine (loaded from `.infinity/mixer.toml` by the Ring-2 layer).
    pub fn set_audio_mixer(&mut self, mixer: inf_audio::MixerConfig) {
        self.audio.set_mixer(mixer);
    }

    /// Register a `.inf_anim` clip (with its skeleton) so a [`RootMotion`] entity
    /// playing that clip drives its `Transform` from the clip's root motion
    /// (P11.3). The editor Simulate wiring seeds these from the loaded assets;
    /// unregistered clips simply skip root motion. Idempotent (re-registering
    /// replaces).
    pub fn register_root_motion_clip(
        &mut self,
        clip_guid: Uuid,
        skeleton: Skeleton,
        clip: AnimClip,
    ) {
        self.clips.insert(clip_guid, RootClip { skeleton, clip });
    }

    /// Advance Simulate by a frame's elapsed time via the fixed-step
    /// accumulator: runs 0..N fixed steps (spiral-of-death guarded). `input` is
    /// the current held-key state.
    pub fn tick(&mut self, doc: &mut SceneDoc, frame_dt: f64, input: SimInput) {
        self.set_input(input);
        let n = self.stepper.accumulate(frame_dt);
        for _ in 0..n {
            self.fixed_step(doc);
        }
    }

    /// Run **exactly one** fixed step with the given input — the deterministic
    /// entry point headless tests script against (bypasses the wall-clock
    /// accumulator).
    pub fn step_once(&mut self, doc: &mut SceneDoc, input: SimInput) {
        self.set_input(input);
        self.fixed_step(doc);
    }

    /// **Queue a tuning edit** (P29.5, pillar S4).
    ///
    /// It applies at the top of the **next** fixed step and not before — see
    /// [`crate::tuning`] for why that is a contract and not a convenience, and
    /// for what [`TuneScope::Keep`](crate::tuning::TuneScope::Keep) does at
    /// `Stop`.
    ///
    /// Queueing always succeeds; *applying* answers whether it did anything, and
    /// that answer is a value nobody waits for. A tuning UI is a live surface
    /// over a world that is changing underneath it.
    pub fn tune(&mut self, tune: crate::tuning::Tune, scope: crate::tuning::TuneScope) {
        self.pending_tunes.push((tune, scope));
    }

    /// How many tunes are queued and not yet applied — the read a test uses to
    /// assert the "next step, not this one" half.
    pub fn pending_tunes(&self) -> usize {
        self.pending_tunes.len()
    }

    /// The `Keep`-scoped tunes applied so far, in order.
    pub fn kept_tunes(&self) -> &[crate::tuning::Tune] {
        &self.kept_tunes
    }

    /// Drain the queue onto the world. The **first** thing a fixed step does.
    fn apply_pending_tunes(&mut self, doc: &mut SceneDoc) {
        if self.pending_tunes.is_empty() {
            return;
        }
        for (tune, scope) in std::mem::take(&mut self.pending_tunes) {
            // ── P29.7 ── The two tunables that are not on the document: a
            //    vehicle lives on the physics bridge (a play-session thing, like
            //    a ragdoll's bodies) and the camera is a host-owned field here
            //    (Ruling 4: never a component and never a resource). Both are
            //    session-scoped by construction — there is no document field for
            //    a `Keep` to land on — and both answer with a value rather than
            //    failing, which is what `CameraTuning::set` already promised.
            let applied = match &tune {
                crate::tuning::Tune::Vehicle { guid, name, value } => self
                    .bridge3d
                    .vehicle_mut(*guid)
                    .map(|v| v.tune(name, *value))
                    .unwrap_or(false),
                crate::tuning::Tune::Camera { name, value } => self.camera.tuning.set(name, *value),
                _ => crate::tuning::apply_tune(doc, &tune),
            };
            if !applied {
                continue;
            }
            if scope == crate::tuning::TuneScope::Keep {
                self.kept_tunes.push(tune);
            }
        }
    }

    /// Exit Simulate: restore the world captured at [`enter`](Self::enter). The
    /// document is byte-for-byte what it was before play — **except** for the
    /// tunes the author asked to keep (P29.5), which are replayed onto it
    /// afterwards as ordinary undoable edits.
    pub fn exit(self, doc: &mut SceneDoc) {
        apply_to_doc(doc, &self.snapshot);
        // ── P29.5 pillar S4 ── AFTER the restore, deliberately: the snapshot is
        //    what the run started from, and a kept tune is a decision about the
        //    document taken during it. Applied through `edit_set_prop`, so it is
        //    one `Edit <field>` step on the history and `Ctrl+Z` takes it back —
        //    a tuning session must not be a change nobody can undo.
        crate::tuning::commit_kept(doc, &self.kept_tunes);
        // ── P22.1 ── **Simulate never leaves an edit behind** (this module's
        //    opening promise). The snapshot restores entities and components;
        //    the deformation field is a *resource*, so it is outside the snapshot
        //    and `apply_to_doc` cannot touch it. Without this line the authoring
        //    viewport keeps drawing the player's footprints after Stop — a
        //    Simulate artefact left on the author's document, which is exactly
        //    the class of bug the P21.4 "the render store IS the save's staging
        //    source" law was written about.
        inf_ecs::deform::clear_deformation(doc.world_mut());
        // ── P24.1 ── the evaluated pose goes with it: a stopped session must not
        //    leave the author's viewport drawing the last frame of a run's
        //    animation over the document's own rest pose.
        inf_ecs::pose::clear_poses(doc.world_mut());
        // ── P24.4 ── and the settled garment with it, for the same reason: a
        //    coat draped by a run is a Simulate artefact, not authored content.
        inf_ecs::cloth::clear_cloth(doc.world_mut());
        inf_ecs::hair::clear_hair(doc.world_mut());
    }

    /// Seed the resolvable `.inf_sm` state machines (P11.2). An entity carrying an
    /// [`AnimStateMachine`] whose `sm` GUID is present here is stepped each fixed
    /// tick against the actor's Blueprint variables.
    pub fn set_state_machines(&mut self, machines: BTreeMap<Uuid, StateMachine>) {
        self.state_machines = machines;
    }

    /// Seed the resolvable `.inf_skel` assets (P24.1) — the skeletons a
    /// machine-driven [`SkeletalMesh`] poses against, sockets included.
    ///
    /// Without them `advance_state_machines` still advances every machine, and
    /// publishes no pose at all: the drawn character falls back to its
    /// `AnimPlayer` (or its rest pose), which is exactly the pre-P24.1 behaviour.
    /// So a caller that forgets this door regresses rendering silently — which is
    /// why `resolve_anim_assets` returns skeletons and clips in the SAME seed
    /// tuple as the machines, and why the shipped player's builder resolves all
    /// three together.
    pub fn set_skeletons(&mut self, skeletons: BTreeMap<Uuid, inf_anim::SkeletonAsset>) {
        self.skeletons = skeletons;
    }

    /// Seed the resolvable `.inf_anim` clips a state machine's states play
    /// (P24.1). See [`pose_clips`](Self::pose_clips) for why this is not the
    /// root-motion registry.
    pub fn set_pose_clips(&mut self, clips: BTreeMap<Uuid, AnimClip>) {
        self.pose_clips = clips;
    }

    /// Seed the resolvable `.inf_cloth` garments (P24.4) — the editor mirror of
    /// `RuntimeSim::set_cloths`.
    pub fn set_cloths(&mut self, cloths: BTreeMap<Uuid, inf_anim::ClothAsset>) {
        self.cloths = cloths;
    }

    /// The garments this session can resolve (a read for tests and gates).
    pub fn cloths(&self) -> &BTreeMap<Uuid, inf_anim::ClothAsset> {
        &self.cloths
    }

    /// Seed the resolvable `.inf_hair` hairstyles (P24.4) - the editor mirror of
    /// `RuntimeSim::set_hairs`.
    pub fn set_hairs(&mut self, hairs: BTreeMap<Uuid, inf_anim::HairAsset>) {
        self.hairs = hairs;
    }

    /// The hairstyles this session can resolve (a read for tests and gates).
    pub fn hairs(&self) -> &BTreeMap<Uuid, inf_anim::HairAsset> {
        &self.hairs
    }

    /// Set how densely hair draws (P24.4).
    ///
    /// The `set_debris_budget` seam, one system over: the *tier -> detail*
    /// mapping lives at the host (`inf_render::hair_detail_for`), because Ring 0
    /// must not know what a GPU is, and the value arrives here as data. A session
    /// nobody tells runs on `HairDetail::GUIDES`, which is what P24.4 v1 drew.
    ///
    /// # No production caller yet (P24.4 audit F4)
    ///
    /// The windowed player calls its twin (`window.rs`, pinned by
    /// `the_windowed_player_applies_the_tier_hair_detail_to_pie_too`); **this one
    /// is called by tests only**, so the editor viewport's Simulate always draws
    /// `HairDetail::GUIDES` whatever the machine is. That is the same "nobody
    /// tells it" state `set_debris_budget` is in on this side, and there it is the
    /// P22.4 *ruling* (a budget that changes the sim must not be clamped in a
    /// preview) rather than an omission — hair's ruling is the opposite one, so
    /// the editor is the host that does not yet honour it. It costs correctness
    /// nothing (ribbons are not folded into `hair_state_bytes`, which
    /// `inf_ecs::hair`'s
    /// `the_detail_draws_differently_and_traces_identically` measures) and it
    /// costs the *preview* its fidelity on a Low-tier machine. Wiring it belongs
    /// with a tier read in `inf_viewport::host`, which is a `cfg`-gated file no
    /// Linux CI leg compiles. Ledgered in ROADMAP §12's P24.4 block.
    pub fn set_hair_detail(&mut self, detail: inf_anim::HairDetail) {
        self.hair_detail = detail;
    }

    /// The locomotion camera's pose this step, or `None` on a level with no
    /// player-controlled character.
    pub fn camera_pose(&self) -> Option<inf_ecs::camera::CameraPose> {
        self.camera_subject.map(|_| self.camera.pose)
    }

    /// The camera itself — the editor's viewport reads it, and the gate traces it.
    pub fn camera(&self) -> &inf_ecs::camera::LocomotionCamera {
        &self.camera
    }

    /// …and mutably: view mode, shoulder, and the tuning P29.5's door edits.
    pub fn camera_mut(&mut self) -> &mut inf_ecs::camera::LocomotionCamera {
        &mut self.camera
    }

    /// The hair detail this session draws at (a read for tests and gates).
    pub fn hair_detail(&self) -> inf_anim::HairDetail {
        self.hair_detail
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

    /// The surface deformation field this Simulate session has pressed into its
    /// terrain (P22.1), or `None` when nothing has touched ground.
    ///
    /// It lives on the **document's world**, not on this struct — see
    /// [`inf_ecs::deform`] for why. Exposed here so a parity gate can compare the
    /// *field* the editor preview produced against the shipped player's, which is
    /// the P22.4 PIE-==-shipping seed. (MIRROR of `RuntimeSim::deform_field`.)
    pub fn deform_field<'a>(
        &self,
        doc: &'a SceneDoc,
    ) -> Option<&'a inf_terrain::deform::DeformField> {
        inf_ecs::deform::deform_field(doc.world())
    }

    /// Install (or replace) the debugger config for a class (B-P4 tier A′): the
    /// breakpoints to pause on + whether to capture wire values. Applies from the
    /// next event pass. Passing `InterpDebug::default()` (empty) disables it.
    ///
    /// **Note** — for hand-built `.inf_act` classes the `breakpoints` are IR
    /// `LocalId`s (there is no `NodeId` map yet); once a class carries graph
    /// provenance the caller translates `NodeId → LocalId` before this call.
    pub fn set_debug(&mut self, class_id: impl Into<String>, debug: InterpDebug) {
        self.debug.insert(class_id.into(), debug);
    }

    /// Drain the debug hits/wire-values collected since the last drain (B-P4 tier
    /// A′). The Ring-2 `sim_tick`/`sim_step_fixed` commands call this after a step
    /// and, when non-empty, emit `sim://debug`. Empty unless a class is being
    /// debugged.
    pub fn take_debug_events(&mut self) -> Vec<SimDebugHit> {
        std::mem::take(&mut self.debug_events)
    }

    /// The `debug.print` log accumulated so far.
    pub fn logs(&self) -> &[String] {
        self.logs.as_slice()
    }

    /// How many log lines fell off the front of [`logs`](Self::logs)'s ring
    /// (Hardening D) — non-zero means the slice is a tail, not the whole session.
    /// MIRROR of `RuntimeSim::dropped_logs`.
    pub fn dropped_logs(&self) -> u64 {
        self.logs.dropped()
    }

    /// How many fixed steps have run.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// A live member variable of an actor (for tests / a debug HUD).
    pub fn actor_var(&self, guid: Uuid, name: &str) -> Option<&Value> {
        self.actors.get(&guid).and_then(|a| a.instance.get(name))
    }

    /// Whether an actor's character was grounded at its last `move_and_slide`
    /// (a telemetry read for tests / a debug HUD). `false` before its first move.
    pub fn is_grounded(&self, guid: Uuid) -> bool {
        self.grounded.get(&guid).copied().unwrap_or(false)
    }

    /// The trigger-volume (sensor) overlaps drained during the most recent fixed
    /// step (Wave 3, E-P4 sim half): each a canonical `a < b` entity-`Guid` pair
    /// with a Begin/End phase, sorted ascending. Rebuilt every step (cleared then
    /// filled), so read it right after a [`step_once`](Self::step_once). This is
    /// the seam trigger-volume gameplay + the editor/runtime parity tests consume.
    pub fn drained_overlaps(&self) -> &[OverlapEvent] {
        &self.drained_overlaps
    }

    // ── internal ──────────────────────────────────────────────────────────

    /// Latch the new input and compute rising **and** falling edges vs. the
    /// previous tick (Wave 3 adds the falling edge for `Input` release events).
    fn set_input(&mut self, input: SimInput) {
        self.just_pressed = input.down.difference(&self.prev_down).cloned().collect();
        self.just_released = self.prev_down.difference(&input.down).cloned().collect();
        self.prev_down = input.down.clone();
        self.input = input;
    }

    /// The four-phase fixed step (see the module docs).
    fn fixed_step(&mut self, doc: &mut SceneDoc) {
        let dt = self.stepper.fixed_dt();
        // ── P29.5 pillar S4 ── **tuning lands here and nowhere else.** Before
        //    the clock, before the physics sync, before a Blueprint reads a
        //    variable: a tuned run has to stay a deterministic sequence of fixed
        //    steps, and an edit that lands part-way through one is a step no
        //    replay can reproduce. See `crate::tuning`.
        self.apply_pending_tunes(doc);
        // ── P17.1 time of day ── advance the level clock ONCE per fixed step,
        //    before anything reads it, so blueprints, the projected sun, shadows,
        //    GI and audio all observe one consistent clock for the step. Pure IEEE
        //    add/mul/floor over the sim's own state (`inf_ecs::sky`), hence
        //    bit-identical across runs and across processes — which is what makes
        //    the sun-direction trace a replay- and PIE-vs-shipping gate. Frozen at
        //    `rate == 0` (the component default), and never called outside a
        //    fixed step, so an idle editor never moves the sun.
        inf_ecs::sky::advance_time_of_day(doc.world_mut(), dt);
        // The weather blend advances in the same slot, for the same reason
        // (P17.4): everything downstream — the projected clouds, the fog, the
        // precipitation, a Blueprint reading `sky.get_precipitation` — must
        // observe ONE weather state for the step. Inert unless a transition is
        // actually in flight on an enabled weather block.
        inf_ecs::sky::advance_weather(doc.world_mut(), dt);
        // 1. ECS → physics.
        self.bridge.sync_from_world(doc.world());
        // ── P22.3 fracture follow ── an INTACT destructible is a normal
        //    entity a Blueprint or a gizmo can move, so its placement tracks
        //    its transform right up until the first chunk comes off (after
        //    which the chunks are solver-owned and following would teleport
        //    settled rubble). Before the sync, because the sync reads the map
        //    while this writes it. (MIRROR of the other host's fixed step.)
        PhysicsBridge3D::follow_fractures(doc.world(), &mut self.fractures);
        // ── P11.3 3D bridge: sync ── carrying the P21.4 voxel chunk colliders,
        //    so a runtime carve is something a body can fall into.
        self.bridge3d
            .sync_from_world_sim(doc.world(), &self.voxels, &self.fractures);
        // ── P20.2 water forces ── buoyancy + hydrodynamic drag, between the
        //    sync and the solver: after the sync because a body must be sampled
        //    where it IS, and before the step because that is the step the forces
        //    belong to. Also arms this step's enter/exit/splash, drained in the
        //    collision slot below so the fixed step has ONE event point. One
        //    branch on a level with no `Buoyancy` component.
        //    (MIRROR of `RuntimeSim::fixed_step`.)
        self.bridge3d.apply_water_forces(dt);
        // ── Wave 3 input events ── fire Input(action) edges BEFORE the Tick pass,
        //    then drain any dispatches they queued.
        self.fire_input_events(doc);
        self.drain_dispatch(doc);
        // 2. Blueprint Tick for every actor (Guid order).
        let tick_args: BTreeMap<String, Value> = [("dt".to_string(), Value::Float(dt))].into();
        let args: std::collections::HashMap<String, Value> = tick_args.into_iter().collect();
        self.run_all_with_args(doc, &EventKind::Tick, &args);
        self.drain_dispatch(doc); // Wave 3: Tick may dispatch custom events.
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
        let intent = inf_ecs::movement::MovementIntent::from_actions(
            |a| self.input.axis(a),
            |a| self.input.is_down(a),
            |a| self.just_pressed.contains(a),
        );
        inf_ecs::movement::apply_intent(doc.world_mut(), &intent);
        inf_physics::d3::step_character_movement(doc.world_mut(), &mut self.bridge3d, dt);
        // 3. Solver.
        self.bridge.step(dt);
        self.bridge3d.step(dt); // ── P11.3 3D bridge: step ──
                                // ── Wave 3 collision + overlap drain ── between the solver and write-back:
                                //    fire Blueprint `Collision` events + collect sensor OverlapEvents.
        self.drain_collisions(doc);
        self.drain_dispatch(doc);
        // 4. Physics → ECS.
        self.bridge.write_back(doc.world_mut());
        self.bridge3d.write_back_into(doc.world_mut()); // ── P11.3 3D bridge: write-back ──
        doc.world_mut().propagate();
        // ── P22.1 surface deformation ── the ground remembers what stood on it.
        //    Here, and not earlier: a footprint's XZ is read off the transform
        //    the solver just wrote and `propagate` just settled, so the print
        //    lands where the body actually ended the step. ONE Ring-0 call
        //    (`inf_ecs::deform`) rather than a loop spelled twice — the sky
        //    advance's shape — so the editor preview and the shipped player
        //    cannot disagree about where a track goes. Inert (one empty vec, no
        //    allocation) on every level whose bodies never touch a terrain.
        //    (MIRROR of `RuntimeSim::fixed_step`.)
        inf_ecs::deform::step_deformation(doc.world_mut(), dt);
        // 5. Advance skeletal-animation play-heads (P11.1). Order-independent
        //    per-entity `t` integration → deterministic at the fixed `dt`.
        //    ── P11.3 root motion ── snapshot play-heads BEFORE advancing, advance,
        //    then apply each RootMotion entity's clip root delta to its Transform.
        let prev_ts = self.capture_root_motion_times(doc);
        inf_ecs::anim::advance_anim_players(doc.world_mut(), dt);
        self.apply_root_motion(doc, &prev_ts);
        doc.world_mut().propagate();
        // ── P11.2 anim state machines ── (P11.3 root-motion consumes poses just
        //    above this same marker; keep the two adjacent + separated.)
        //    Step each `AnimStateMachine` against its actor's Blueprint variables.
        self.advance_state_machines(doc, dt);
        // ── P11.3 attachments ── entities ride their target's socket, written
        //    post-anim-tick; propagate again so followers' own globals settle.
        update_attachments(doc.world_mut());
        doc.world_mut().propagate();
        // ── P24.4 cloth ── garments fall on the body the pose just put them on.
        //    HERE, and not earlier: the capsules are read off the pose this step
        //    published and the model frame off a `GlobalTransform` the propagate
        //    above has settled, so a coat collides against THIS step's arm rather
        //    than the last one's. ONE Ring-0 call (`inf_ecs::cloth`) rather than a
        //    loop spelled twice — the deform doctrine's shape — so the editor
        //    preview and the shipped player cannot disagree about how a garment
        //    falls. Inert (one empty query, no allocation) on every level with no
        //    `ClothSim`. (MIRROR of `RuntimeSim::fixed_step`.)
        self.step_cloth(doc, dt);
        // -- P24.4 hair -- strands fall on the head the pose just put them on,
        //    in the same slot and for the same reasons as the garment above.
        //    (MIRROR of `RuntimeSim::fixed_step`.)
        self.step_hair(doc, dt);
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
        self.fire_destroyed(doc, &destroyed);
        // ── P12.3 audio step ── last, so it observes this step's final transforms:
        //    pick the listener, enqueue autoplay, resolve occlusion, drain the queue.
        self.audio_step(doc);
        // ── P29.6 the locomotion camera ── LAST, and outside everything the trace
        //    folds: it reads where the character ended this step and writes
        //    nothing back. (MIRROR of `RuntimeSim::step_camera`.)
        self.camera_subject = inf_ecs::movement::camera_subject(doc.world());
        if let Some(subject) = self.camera_subject {
            inf_physics::d3::step_locomotion_camera(
                doc.world(),
                &mut self.bridge3d,
                &mut self.camera,
                subject,
                dt,
            );
        }
        // Per-actor caches keyed by `Guid` are pruned to the live world (Hardening
        // D) — MIRROR of `RuntimeSim::capture_positions`'s rule. `audio_started`
        // is dropped by the audio step above; `grounded` had no such rule and
        // grew for the session in a world that spawns and despawns characters.
        // The same predicate the player uses, character for character: the GUID
        // index alone is not quite enough, because an entity can also be
        // despawned through `world_mut()` without going past `EcsWorld::despawn`.
        self.grounded.retain(|guid, _| {
            doc.world()
                .entity_of(*guid)
                .is_some_and(|e| doc.world().world().get_entity(e).is_ok())
        });
        // Rising edges are one fixed step wide.
        self.just_pressed.clear();
        self.steps += 1;
    }

    /// Snapshot the play-head `t` of every root-motion-driven, playing entity
    /// **before** the anim advance, keyed by `Guid` (P11.3). The delta from these
    /// to the post-advance `t` is the root motion applied this step.
    fn capture_root_motion_times(&self, doc: &mut SceneDoc) -> BTreeMap<Uuid, f64> {
        let mut out = BTreeMap::new();
        let w = doc.world_mut().world_mut();
        let mut q = w.query::<(&Guid, &RootMotion, &AnimPlayer)>();
        for (g, rm, ap) in q.iter(w) {
            if rm.mode == RootMotionMode::ApplyToEntity && ap.playing {
                out.insert(g.0, ap.t);
            }
        }
        out
    }

    /// Apply each root-motion entity's clip root delta to its `Transform` (P11.3).
    ///
    /// For each entity captured in `prev_ts`: extract the root delta over
    /// `[prev_t, cur_t]` (via [`inf_anim::root_delta`]), rotate the ground-plane
    /// translation into world by the entity's current yaw, add the clip's yaw, and
    /// commit — **through the 3D character mover** (collision-resolved, grounded
    /// reported) when the entity is a [`CharacterController3D`], else as a raw
    /// transform add. Entities whose clip isn't registered skip silently.
    fn apply_root_motion(&mut self, doc: &mut SceneDoc, prev_ts: &BTreeMap<Uuid, f64>) {
        if prev_ts.is_empty() {
            return;
        }
        // Read pass: gather (guid, entity, prev_t, cur_t, looping, clip, has_cc).
        let mut work: Vec<(Uuid, Entity, f64, f64, bool, Uuid, bool)> = Vec::new();
        for (&guid, &prev_t) in prev_ts {
            let Some(entity) = doc.world().entity_of(guid) else {
                continue;
            };
            let w = doc.world().world();
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
            let t = doc
                .world()
                .world()
                .get::<Transform>(entity)
                .copied()
                .unwrap_or(Transform::IDENTITY);
            // Root motion is expressed in the character's facing frame → rotate the
            // ground-plane delta by the entity's current yaw into world space,
            // through the ONE door both fixed steps share.
            // `DQuat::from_rotation_y` is `sin_cos` inside glam, and this
            // transform is folded into `state_bytes` (L6.F5).
            let world_delta = inf_anim::root_delta_world(t.rotation.y, d.translation);
            let new_yaw_deg = t.rotation.y + d.yaw.to_degrees() as f64;
            let pos = t.translation.to_dvec3();

            let new_pos = if has_cc {
                // Drive the entity through the 3D mover so walls/steps/slopes apply.
                let mover = inf_physics::d3::mover_for(doc.world(), guid);
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

            if let Some(mut tr) = doc.world_mut().world_mut().get_mut::<Transform>(entity) {
                tr.translation.x = new_pos.x;
                tr.translation.y = new_pos.y;
                tr.translation.z = new_pos.z;
                tr.rotation.y = new_yaw_deg;
                changed = true;
            }
        }
        if changed {
            doc.world_mut().mark_dirty();
        }
    }

    /// Step every entity's [`AnimStateMachine`] (P11.2) **and evaluate the pose it
    /// lands in** (P24.1), through the ONE Ring-0 rule both hosts call
    /// ([`inf_ecs::pose::step_pose_evaluation`]).
    ///
    /// Everything this used to spell inline — the read pass, the runtime POD
    /// conversion, the `SmContext` seam, the write-back — now lives once, in
    /// Ring 0, for the reason `inf_ecs::deform` records: two byte-identical loops
    /// are two loops that can drift, and this one is the difference between the
    /// preview and the shipped build posing a character the same way. All that is
    /// left here is *which registries answer*, which is genuinely host-local (the
    /// editor resolves them out of the project's asset DB, the player out of a
    /// cooked pack).
    ///
    /// Transition conditions and blend-space params still read the entity's
    /// *actor* Blueprint variables; an entity with no actor gets an empty variable
    /// set, so every param defaults to `0` (documented, unchanged).
    fn advance_state_machines(&mut self, doc: &mut SceneDoc, dt: f64) {
        if self.state_machines.is_empty() {
            return;
        }
        let state_machines = &self.state_machines;
        let skeletons = &self.skeletons;
        let pose_clips = &self.pose_clips;
        let actors = &self.actors;
        let machines = |g: Uuid| state_machines.get(&g);
        let skels = |g: Uuid| skeletons.get(&g);
        let clips = |c: inf_anim::ClipRef| pose_clips.get(&Uuid::from_bytes(c));
        let vars = |g: Uuid| {
            actors
                .get(&g)
                .map(|a| var_snapshot(&a.instance))
                .unwrap_or_default()
        };
        inf_ecs::pose::step_pose_evaluation(doc.world_mut(), dt, &machines, &skels, &clips, &vars);
    }

    /// Advance every worn garment (P24.4) through the ONE Ring-0 rule both hosts
    /// call ([`inf_ecs::cloth::step_cloth_simulation`]) — the editor mirror of
    /// `RuntimeSim::step_cloth`.
    ///
    /// Only *which registries answer* is host-local: the editor resolves them out
    /// of the project's asset DB, the player out of a cooked pack (or a `--level`
    /// dev dir). The rule itself lives once.
    ///
    /// Returns immediately on a level with no resolvable garment, so a world that
    /// wears nothing pays one `is_empty` branch per step.
    fn step_cloth(&mut self, doc: &mut SceneDoc, dt: f64) {
        if self.cloths.is_empty() {
            return;
        }
        let cloths = &self.cloths;
        let skeletons = &self.skeletons;
        let garments = |g: Uuid| cloths.get(&g);
        let skels = |g: Uuid| skeletons.get(&g);
        inf_ecs::cloth::step_cloth_simulation(doc.world_mut(), dt, &garments, &skels);
    }

    /// Advance every worn hairstyle (P24.4) through the ONE Ring-0 rule both hosts
    /// call ([`inf_ecs::hair::step_hair_simulation`]) - the editor mirror of
    /// `RuntimeSim::step_hair`.
    fn step_hair(&mut self, doc: &mut SceneDoc, dt: f64) {
        if self.hairs.is_empty() {
            return;
        }
        let hairs = &self.hairs;
        let skeletons = &self.skeletons;
        let styles = |g: Uuid| hairs.get(&g);
        let skels = |g: Uuid| skeletons.get(&g);
        inf_ecs::hair::step_hair_simulation(doc.world_mut(), dt, &styles, &skels, self.hair_detail);
    }

    /// Fire `event` (no args) on every actor.
    fn run_all(&mut self, doc: &mut SceneDoc, event: &EventKind) {
        let args = std::collections::HashMap::new();
        self.run_all_with_args(doc, event, &args);
    }

    /// Fire `event` on every actor in `Guid` order (each via [`run_on_guid`]).
    fn run_all_with_args(
        &mut self,
        doc: &mut SceneDoc,
        event: &EventKind,
        args: &std::collections::HashMap<String, Value>,
    ) {
        let guids: Vec<Uuid> = self.actors.keys().copied().collect();
        for guid in guids {
            self.run_on_guid(doc, guid, event, args);
        }
    }

    /// Fire `event` on the single actor `guid` through a fresh [`SimHost`]. The
    /// actor is lifted out of the map during its run so the per-actor borrow
    /// doesn't alias the session's other fields; its `entity` id is threaded into
    /// the host as `current_entity` so `event::bind` knows the calling listener.
    fn run_on_guid(
        &mut self,
        doc: &mut SceneDoc,
        guid: Uuid,
        event: &EventKind,
        args: &std::collections::HashMap<String, Value>,
    ) {
        let Some(mut state) = self.actors.remove(&guid) else {
            return;
        };
        let current_entity = match state.instance.get("entity") {
            Some(Value::Int(i)) => *i,
            _ => 0,
        };
        // B-P4 tier A′: the class's debugger config (default when the class isn't
        // being debugged) + whether it observes anything, computed before the host
        // borrow. The config is cheap to clone (a HashSet + bool).
        let debug_cfg = self.debug.get(&state.class.id).cloned().unwrap_or_default();
        let debug_active = !debug_cfg.breakpoints.is_empty() || debug_cfg.capture_wires;
        let fn_name = state
            .class
            .handler(event)
            .map(|b| b.body.name.clone())
            .unwrap_or_default();
        let mut recorded: Option<SimDebugHit> = None;
        {
            let mut host = SimHost {
                bridge: &mut self.bridge,
                bridge3d: &mut self.bridge3d,
                world: doc.world_mut(),
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
                voxels: &mut self.voxels,
                fractures: &mut self.fractures,
            };
            match run_event(
                &state.class,
                &mut state.instance,
                event,
                args,
                &mut host,
                &debug_cfg,
            ) {
                Ok(trace) => {
                    if debug_active && (!trace.hits.is_empty() || !trace.wires.is_empty()) {
                        recorded = Some(SimDebugHit {
                            class_id: state.class.id.clone(),
                            event: event.key(),
                            fn_name,
                            hits: trace.hits.iter().map(|l| l.0).collect(),
                            wires: latest_wire_values(&trace),
                        });
                    }
                }
                Err(e) => self.logs.push(format!("{}: {e}", event.key())),
            }
        }
        if let Some(hit) = recorded {
            self.debug_events.push(hit);
        }
        self.actors.insert(guid, state);
    }

    /// Fire `event` on whatever actor owns blueprint entity id `entity_id`, if
    /// any (the dispatch drain's target/listener resolver).
    fn fire_on_entity(
        &mut self,
        doc: &mut SceneDoc,
        entity_id: i64,
        event: &EventKind,
        args: &std::collections::HashMap<String, Value>,
    ) {
        if let Some(&guid) = self.entities.get(&entity_id) {
            self.run_on_guid(doc, guid, event, args);
        }
    }

    /// The blueprint `i64` entity id assigned to `guid`, if it is a mapped entity.
    fn entity_id_of(&self, guid: Uuid) -> Option<i64> {
        self.entities
            .iter()
            .find(|(_, g)| **g == guid)
            .map(|(id, _)| *id)
    }

    /// Fire this step's `Input(action)` events (Wave 3): presses first, then
    /// releases, each set in `BTreeSet`-ascending action order, on every actor
    /// (handlers without a matching `Input` binding no-op). `pressed` carries
    /// `true` for a press edge, `false` for a release edge.
    fn fire_input_events(&mut self, doc: &mut SceneDoc) {
        if self.just_pressed.is_empty() && self.just_released.is_empty() {
            return;
        }
        let pressed: Vec<String> = self.just_pressed.iter().cloned().collect();
        let released: Vec<String> = self.just_released.iter().cloned().collect();
        for action in pressed {
            let args: std::collections::HashMap<String, Value> =
                [("pressed".to_string(), Value::Bool(true))].into();
            self.run_all_with_args(doc, &EventKind::Input(action), &args);
        }
        for action in released {
            let args: std::collections::HashMap<String, Value> =
                [("pressed".to_string(), Value::Bool(false))].into();
            self.run_all_with_args(doc, &EventKind::Input(action), &args);
        }
    }

    /// Drain this step's 2D **then** 3D contact events once, feeding two consumers
    /// (Wave 3): (a) Blueprint `Collision` events — `Started` phase only, sensors
    /// included — fired on each side that is an actor with the other side's entity
    /// id (0 if none), canonical `a ≤ b` and both sides ascending; (b) the
    /// `drained_overlaps` list — **sensor pairs only**, canonical `a < b`,
    /// Begin/End, sorted. Called between the solver step and write-back.
    fn drain_collisions(&mut self, doc: &mut SceneDoc) {
        self.drained_overlaps.clear();
        // Resolve both worlds' events to canonical (guid_a ≤ guid_b, phase, sensor)
        // tuples, dropping any pair with an untracked collider.
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

        // (a) Blueprint `Collision` events — Started only, sensors INCLUDED. Build
        // the fire list (canonical, sorted, both sides ascending) before firing so
        // no borrow of `resolved` outlives the actor runs.
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
            let args: std::collections::HashMap<String, Value> =
                [("other".to_string(), Value::Int(other_id))].into();
            self.run_on_guid(doc, guid, &EventKind::Collision, &args);
        }
        self.drain_water_events(doc);
    }

    /// Fire this step's water crossings on their actors (P20.2) — the MIRROR of
    /// `RuntimeSim::drain_water_events`.
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
    /// same deterministic order.
    fn drain_water_events(&mut self, doc: &mut SceneDoc) {
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
            let args: std::collections::HashMap<String, Value> = [
                (
                    "water".to_string(),
                    Value::Int(self.entity_id_of(ev.water).unwrap_or(0)),
                ),
                ("speed".to_string(), Value::Float(ev.speed_m_s)),
            ]
            .into();
            self.run_on_guid(doc, ev.body, &kind, &args);
        }
    }

    /// Fire this step's `Destroyed` events on their actors (P22.3) — the MIRROR
    /// of `RuntimeSim::fire_destroyed`.
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
    fn fire_destroyed(&mut self, doc: &mut SceneDoc, events: &[DestroyedEvent]) {
        if events.is_empty() {
            return;
        }
        for ev in events {
            // The audio push first, so it is queued whether or not the actor has
            // a handler — and so an actor that has both gets the component's
            // sound before whatever its handler queues.
            if let Some((src, pos)) = destroyed_emitter(doc.world(), ev.entity) {
                let cmd =
                    play_command_for(guid_source_key(ev.entity), &src, src.spatial.then_some(pos));
                self.audio_cmds.push(AudioCommand::Play(cmd));
            }
            if !self.actors.contains_key(&ev.entity) {
                continue;
            }
            let args: std::collections::HashMap<String, Value> =
                [("chunks".to_string(), Value::Int(ev.detached as i64))].into();
            self.run_on_guid(doc, ev.entity, &EventKind::Destroyed, &args);
        }
    }

    /// Drain the FIFO dispatch queue (Wave 3): for each popped `(target, name)`,
    /// fire `Custom(name)` on the target actor, then `Custom(handler)` on every
    /// listener bound to `(target, name)` in ascending listener-id order. Nested
    /// dispatches append to the queue and are processed in turn; once
    /// [`DISPATCH_ROUND_CAP`] dispatches have run the remainder is logged + dropped
    /// (a deterministic cycle guard). No-op when the queue is empty.
    fn drain_dispatch(&mut self, doc: &mut SceneDoc) {
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
            // Build the fire list under an immutable borrow of `bindings`, then
            // fire (each run may itself append to the queue).
            let mut fires: Vec<(i64, String)> = vec![(target, name.clone())];
            if let Some(listeners) = self.bindings.get(&(target, name)) {
                for (listener, handler) in listeners {
                    fires.push((*listener, handler.clone()));
                }
            }
            for (entity_id, ev_name) in fires {
                let args = std::collections::HashMap::new();
                self.fire_on_entity(doc, entity_id, &EventKind::Custom(ev_name), &args);
            }
        }
    }
}

impl SimSession {
    /// The accumulated audio command stream (P12.3): the deterministic play/stop/
    /// set sequence drained across every step. The headless command-stream test
    /// asserts against this rather than device output.
    pub fn audio_command_log(&self) -> &[AudioCommand] {
        self.audio_log.as_slice()
    }

    /// How many audio commands fell off the front of the ring (Hardening D).
    /// MIRROR of `RuntimeSim::dropped_audio_commands`.
    pub fn dropped_audio_commands(&self) -> u64 {
        self.audio_log.dropped()
    }

    /// The P12.3 audio step (runs last in a fixed step): pick the listener, enqueue
    /// autoplay sources once, resolve occlusion via one physics raycast per
    /// occlusion-enabled spatial source, then drain the queue into the host engine.
    fn audio_step(&mut self, doc: &mut SceneDoc) {
        // -- P29.4 footsteps -- the animation's own event markers, turned into
        //    voices. The DECISION is Ring 0 (`inf_ecs::anim_bridge::footstep_cues`
        //    is a pure function of this step's notifies and the clip's
        //    `Mask_FootstepSound` channel); this is only the mapping onto the
        //    queue, and it is the same six lines in the other host. Inert on
        //    every level whose clips carry no footstep markers.
        for cue in inf_ecs::anim_bridge::footstep_cues(doc.world()) {
            let Some(src) = audio_source_of(doc.world(), cue.source) else {
                continue;
            };
            let key = cue.source.as_u128() as u64;
            let pos = emitter_position(doc.world(), cue.source);
            let mut cmd = play_command_for(key, &src, src.spatial.then_some(pos));
            cmd.volume *= cue.gain;
            self.audio_cmds.push(AudioCommand::Play(cmd));
        }
        // 1. Listener: the first active `AudioListener` (Guid order); else keep the
        //    engine's current pose (default/origin). Orientation-from-transform is a
        //    documented follow-up — v1 uses the entity position with the default basis.
        let listener = active_listener(doc.world());
        let listener_pos = listener
            .map(|l| l.position)
            .unwrap_or_else(|| self.audio.listener().position);
        if let Some(l) = listener {
            self.audio_cmds.push(AudioCommand::SetListener(l));
        }

        // 2. Autoplay: enqueue a Play once per not-yet-started autoplay `AudioSource`
        //    (any entity, not just actors), in deterministic Guid order. Keyed by the
        //    Guid's low bits so a scene-placed emitter gets a stable source id.
        //
        //    `still_alive` rather than every guid in the world (lens 3 P34,
        //    Hardening Wave G) — the MIRROR of `RuntimeSim::audio_step`, whose
        //    doc carries the argument. Same walk, same source of truth, one
        //    lookup per entity instead of one insert.
        let mut autoplay: Vec<(Uuid, AudioSource, DVec3)> = Vec::new();
        let mut still_alive: BTreeSet<Uuid> = BTreeSet::new();
        for e in doc.world().world().iter_entities() {
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

        // 3. Occlusion for Blueprint-queued Plays (source = actor entity id): one
        //    raycast per spatial+occlusion source (targets collected under the
        //    immutable world borrow, then raycast).
        let mut occ: Vec<(usize, DVec3)> = Vec::new();
        for (i, cmd) in self.audio_cmds.iter().enumerate() {
            if let AudioCommand::Play(p) = cmd {
                if let Some(pos) = p.position {
                    if let Some(guid) = self.entities.get(&(p.source as i64)) {
                        if audio_source_of(doc.world(), *guid)
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

        // 4. Drain into the host engine (record the stream for the determinism test).
        let cmds = std::mem::take(&mut self.audio_cmds);
        self.audio_log.extend(cmds.iter().cloned());
        let clips = &self.audio_clips;
        self.audio
            .drain(&cmds, &|g| clips.get(&g).and_then(|a| a.decode().ok()));
        // Host-side reap of naturally-finished voices (device bookkeeping only —
        // not sim state, so the command stream above is untouched).
        self.audio.reap();
    }

    /// One occlusion raycast from `listener` toward `emitter`: a hit closer than the
    /// emitter obstructs the line of sight → the configured dB cut, else clear (1.0).
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
}

/// A stable audio source key for a scene-placed emitter (the `Guid`'s low 64
/// bits). Blueprint-driven sources instead key by the actor's `i64` entity id;
/// the two spaces don't collide in practice (an emitter uses one path).
fn guid_source_key(guid: Uuid) -> u64 {
    guid.as_u128() as u64
}

/// Read a clone of an entity's [`AudioSource`] by `Guid`, if present.
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

/// The first **active** [`AudioListener`] (Guid order), as an `inf_audio::Listener`
/// posed at the entity's world position (default orientation — orientation from the
/// transform basis is a documented P12.3 follow-up).
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

/// Build a [`PlayCommand`] from an [`AudioSource`] and an optional world position
/// (`Some` when spatial). Shared by autoplay + Blueprint `audio.play`.
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

/// The engine [`Host`] a Simulate tick runs against: routes `input.*` to the
/// held-key state, `debug.print` to the log, and exposes the physics world via
/// [`Host::physics`]. `vars::*` are handled one layer up by
/// [`ActorHost`](inf_blueprint::ActorHost).
struct SimHost<'a> {
    bridge: &'a mut PhysicsBridge2D,
    /// The 3D physics bridge (P11.3), powering `physics3d.*` nodes.
    bridge3d: &'a mut PhysicsBridge3D,
    world: &'a mut EcsWorld,
    input: &'a SimInput,
    just_pressed: &'a BTreeSet<String>,
    entities: &'a BTreeMap<i64, Uuid>,
    logs: &'a mut BoundedLog<String>,
    grounded: &'a mut BTreeMap<Uuid, bool>,
    /// The P12.3 audio command sink: `audio.*` nodes enqueue here; drained after
    /// the step in [`SimSession::audio_step`].
    audio_cmds: &'a mut Vec<AudioCommand>,
    /// The blueprint entity id of the actor currently running (Wave 3): the
    /// listener `event::bind`/`event::unbind` register against.
    current_entity: i64,
    /// The session's event-dispatcher bindings (Wave 3), mutated by
    /// `event::bind`/`event::unbind`.
    bindings: &'a mut BTreeMap<(i64, String), BTreeMap<i64, String>>,
    /// The session's FIFO dispatch queue (Wave 3); `event::dispatch` appends here.
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
    /// Borrowed from `SimSession::voxels` — never from a render store, which is
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

impl Host for SimHost<'_> {
    fn call(&mut self, path: &[String], args: &[Value]) -> Result<Value, RunError> {
        match (
            path.first().map(String::as_str),
            path.get(1).map(String::as_str),
        ) {
            (Some("input"), Some("is_down")) => {
                let key = arg_str(args, 0);
                Ok(Value::Bool(self.input.is_down(&key)))
            }
            (Some("input"), Some("just_pressed")) => {
                let key = arg_str(args, 0);
                Ok(Value::Bool(self.just_pressed.contains(&key)))
            }
            // ── Wave 3 event dispatchers ── `dispatch.*` nodes lower to these.
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
                    // Remove the caller's subscription iff the handler matches.
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
            // terrain.height_at(x, z) → the world height at that XZ (P11.4). The
            // seam a 3D character reads to stay on a heightfield terrain (which has
            // no physics collider); mirrored exactly in the shipped runtime host.
            //
            // P21.2: it is the **combined** ground query. The heightfield answers
            // where it is still a heightfield; where a carve has holed it — or
            // where there is no terrain at all — the topmost voxel surface does, so
            // a character walking into a cave mouth gets the cave floor instead of
            // the `None` a holed bilinear cell produces. One Ring-0 rule
            // (`inf_voxel::ground_height_at`), read here and in the shipped host.
            (Some("terrain"), Some("height_at")) => Ok(Value::Float(terrain_height_at(
                self.world,
                self.voxels,
                arg_f64(args, 0),
                arg_f64(args, 1),
            ))),
            // sky.* (P17.1) — the level clock, Blueprint-drivable. Four one-line
            // seams over `inf_ecs::sky`, shared verbatim with the shipped runtime
            // host so preview == shipped by construction. Units: seconds for the
            // clock, a dimensionless multiplier for the rate.
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
            // water index, shared verbatim with the shipped RuntimeHost so preview
            // == shipped by construction. They read `inf_water`'s height query, the
            // same evaluator the buoyancy force used this step and the same one the
            // renderer draws — never render state, and never a camera.
            //
            // `surface_height` answers `0.0` where there is no water: the IR has no
            // optional Float, and the `terrain.height_at` precedent is a plain
            // default rather than a sentinel (0 is a plausible sea level). An id
            // that names no entity answers "dry" rather than erroring: a query is
            // not an action.
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
            // voxel.* (P21.4) — RUNTIME CARVING, shared verbatim with the shipped
            // RuntimeHost so preview == shipped by construction. The three actions
            // run one Ring-0 rule (`inf_voxel::runtime_carve`) against the sim's
            // own volume map plus the shared coupling rule against the sim's own
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
            // shipped `RuntimeHost` so preview == shipped by construction. The two actions run
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
            // Unknown engine call: log it (matching the graph preview host) so a
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

/// `audio.*` nodes enqueue self-contained commands (P12.3): `play` reads the
/// entity's [`AudioSource`] to build a full [`PlayCommand`]; the setters address
/// the source's voice by its entity key. The queue is drained after the step.
impl AudioHost for SimHost<'_> {
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

impl SimHost<'_> {
    /// Build a [`PlayCommand`] from an entity's [`AudioSource`] (+ its world pose).
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

    /// Resolve a blueprint `i64` entity id to its `Guid`.
    fn guid_of(&self, entity: i64) -> Result<Uuid, String> {
        self.entities
            .get(&entity)
            .copied()
            .ok_or_else(|| format!("no entity for id {entity}"))
    }

    /// Build a [`CharacterMover2D`] from an entity's `CharacterController2D` +
    /// `Collider2D`. Falls back to a default upright capsule mover when the
    /// components are missing, so a bare actor still moves.
    fn mover_for(&self, guid: Uuid) -> CharacterMover2D {
        let Some(entity) = self.world.entity_of(guid) else {
            return CharacterMover2D::new(ColliderShape2D::Capsule {
                half_height: 0.5,
                radius: 0.25,
            });
        };
        let w = self.world.world();
        let shape =
            w.get::<Collider2D>(entity)
                .map(collider_shape)
                .unwrap_or(ColliderShape2D::Capsule {
                    half_height: 0.5,
                    radius: 0.25,
                });
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

impl Physics2dHost for SimHost<'_> {
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
        // Drive the kinematic body + the ECS transform to the resolved pose so
        // the next sync and any later query in this tick agree.
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
        // A zero-motion probe: the mover reports `grounded` without committing a
        // move — an accurate "on the ground right now?" independent of history.
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

impl Physics3dHost for SimHost<'_> {
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
        //    this host and the shipped runtime's run the same thresholds rather
        //    than two copies of them. Inert — literally the identity — when not
        //    swimming.
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

/// The latest captured value per wire in a [`Trace`], as `(LocalId, string)`
/// pairs sorted by local id (B-P4 tier A′). A wire re-evaluated in a loop keeps
/// only its most recent value.
fn latest_wire_values(trace: &Trace) -> Vec<(u32, String)> {
    let mut latest: BTreeMap<u32, String> = BTreeMap::new();
    for (lid, v) in &trace.wires {
        latest.insert(lid.0, debug_value_string(v));
    }
    latest.into_iter().collect()
}

/// Stringify a runtime [`Value`] for the debug wire inspector.
fn debug_value_string(v: &Value) -> String {
    match v {
        Value::Float(f) => format!("{f}"),
        Value::Int(i) => format!("{i}"),
        Value::Bool(b) => format!("{b}"),
        Value::Str(s) => format!("{s:?}"),
        Value::Unit => "()".to_string(),
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
/// entity-id / target argument coercion for the Wave 3 `event::*` host arms.
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

/// The **ground** height at world `(x, z)` — the `terrain.height_at` host seam
/// (P11.4), byte-for-byte the shipped runtime host's (preview == shipped). A
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
/// Mirrored character-for-character in `inf_player::runtime_sim`, for the reason
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
/// Shared with the runtime twin; deterministic (`BTreeMap` order, and the answer
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
/// Mirrored character-for-character in `inf_player::runtime_sim`, for the reason
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
/// Shared with the runtime twin.
fn clamp_material(m: i64) -> u8 {
    m.clamp(0, inf_voxel::MATERIAL_COUNT as i64 - 1) as u8
}

// ── P11.2 state-machine glue: ECS POD ↔ inf-anim runtime + var snapshot ──────
//
// The conversions are field-for-field (`SmRuntimeState` is a deliberate POD
// mirror of `inf_anim::SmRuntime`, kept in `inf-ecs` so that crate needs no
// `inf-anim` dep — see the `SmRuntimeState` docs). Duplicated in the shipped
// player's `runtime_sim` for the same "preview == shipped" reason `SimHost` is.

// P24.1: `to_anim_runtime` / `from_anim_runtime` used to live here, spelled
// identically in this file and in the player's `runtime_sim` — a hand-maintained
// mirror pair for a struct-to-struct field copy. They are now
// `inf_ecs::pose::{to_anim_runtime, from_anim_runtime}`, called by the one Ring-0
// fixed-step rule both hosts share.

/// A `name → f64` snapshot of an actor's Blueprint variables, for the state
/// machine's condition/param lookups. Non-numeric values (strings, unit, …) are
/// dropped; `Bool` maps to `1.0`/`0.0`.
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

/// A `Guid` marker used to look up entities during Simulate (re-export helper so
/// callers building an actor list don't reach into `inf_ecs` directly).
pub type ActorGuid = Guid;

/// The seed maps [`resolve_anim_assets`] produces: the `.inf_sm` state machines
/// (keyed by asset GUID), the root-motion `(clip GUID, skeleton, clip)` triples,
/// and — P24.1 — the `.inf_skel` assets and the clips a machine's states play.
///
/// A tuple, and a growing one, deliberately: every consumer seeds **all** of it
/// (`SimSession::set_state_machines` + `register_root_motion_clip` +
/// `set_skeletons` + `set_pose_clips`), so a caller that ignores an element fails
/// to compile rather than quietly running a session whose characters never move.
/// That is the same argument the PIE payload's positional tail records.
pub type AnimSeed = (
    BTreeMap<Uuid, StateMachine>,
    Vec<(Uuid, Skeleton, AnimClip)>,
    BTreeMap<Uuid, inf_anim::SkeletonAsset>,
    BTreeMap<Uuid, AnimClip>,
);

/// Resolve a scene's referenced P11 animation assets into the seed maps a
/// [`SimSession`] (and the shipped [`RuntimeSim`](../../inf_player/runtime_sim/index.html))
/// need (P11.4): the `.inf_sm` state machines it steps, and the root-motion
/// `(clip GUID, skeleton, clip)` triples it registers. `resolve_anim` reads an
/// anim asset's raw bytes by GUID (the caller backs it with the project asset DB /
/// the pack). A machine/clip whose bytes don't resolve is skipped; a clip whose
/// skeleton doesn't resolve is dropped from the *root-motion* set (root motion
/// needs it) but still reaches the pose set. Deterministic (`BTreeMap`/Guid
/// order). The caller seeds [`SimSession::set_state_machines`] +
/// [`SimSession::register_root_motion_clip`] + [`SimSession::set_skeletons`] +
/// [`SimSession::set_pose_clips`] from the result — the editor Simulate twin of
/// the player's `InfSceneWorldBuilder` anim resolution, so preview == shipped.
///
/// # The P24.1 transitive walk
///
/// Before this batch the walk stopped at the **directly referenced** GUIDs:
/// `SkeletalMesh.skeleton`, `AnimPlayer.clip`, `AnimStateMachine.sm`. That was
/// enough while the machine only ever advanced its own runtime state. Now the
/// machine evaluates a **pose**, which needs the clips its own states play — and
/// those are named inside the `.inf_sm`, not by any component. So each resolved
/// machine is walked for its motions ([`machine_clip_refs`]) and each named clip
/// is resolved through the same closure and the same dedupe. Without that hop a
/// machine-driven character would resolve its machine, step it correctly, and
/// draw its rest pose — the exact defect P24.1 exists to repair, one level up.
pub fn resolve_anim_assets<H>(doc: &SceneDoc, mut resolve_anim: H) -> AnimSeed
where
    H: FnMut(Uuid) -> Option<Vec<u8>>,
{
    use std::collections::btree_map::Entry;
    let mut machines: BTreeMap<Uuid, StateMachine> = BTreeMap::new();
    let mut clips: BTreeMap<Uuid, (Skeleton, AnimClip)> = BTreeMap::new();
    let mut skeletons: BTreeMap<Uuid, SkeletonAsset> = BTreeMap::new();
    let mut pose_clips: BTreeMap<Uuid, AnimClip> = BTreeMap::new();
    let world = doc.world();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let w = world.world();
        if let Some(sk_guid) = w.get::<SkeletalMesh>(e).and_then(|s| s.skeleton) {
            if let Entry::Vacant(v) = skeletons.entry(sk_guid) {
                if let Some(asset) =
                    resolve_anim(sk_guid).and_then(|b| decode_anim::<SkeletonAsset>(sk_guid, &b))
                {
                    v.insert(asset);
                }
            }
        }
        if let Some(sm_guid) = w.get::<AnimStateMachine>(e).and_then(|s| s.sm) {
            if let Entry::Vacant(v) = machines.entry(sm_guid) {
                if let Some(asset) = resolve_anim(sm_guid)
                    .and_then(|b| decode_anim::<StateMachineAsset>(sm_guid, &b))
                {
                    v.insert(asset.machine);
                }
            }
        }
        if let Some(clip_guid) = w.get::<AnimPlayer>(e).and_then(|p| p.clip) {
            if let Entry::Vacant(v) = clips.entry(clip_guid) {
                if let Some(ca) = resolve_anim(clip_guid)
                    .and_then(|b| decode_anim::<AnimClipAsset>(clip_guid, &b))
                {
                    if let Some(sk_guid) = ca.skeleton.map(Uuid::from_bytes) {
                        if let Some(sk) = resolve_anim(sk_guid)
                            .and_then(|b| decode_anim::<SkeletonAsset>(sk_guid, &b))
                        {
                            v.insert((sk.skeleton, ca.clip));
                        }
                    }
                }
            }
        }
    }
    // The transitive hop: every clip a resolved machine's states play (P24.1).
    let refs: Vec<Uuid> = machines
        .values()
        .flat_map(machine_clip_refs)
        .collect::<BTreeSet<Uuid>>()
        .into_iter()
        .collect();
    for clip_guid in refs {
        if let Entry::Vacant(v) = pose_clips.entry(clip_guid) {
            if let Some(ca) =
                resolve_anim(clip_guid).and_then(|b| decode_anim::<AnimClipAsset>(clip_guid, &b))
            {
                v.insert(ca.clip);
            }
        }
    }
    let root_clips = clips.into_iter().map(|(g, (s, c))| (g, s, c)).collect();
    (machines, root_clips, skeletons, pose_clips)
}

/// Decode one animation asset for a Simulate session, **saying so out loud when
/// it does not load** (P24.1 audit B1).
///
/// This used to be `inf_asset::decode::<T>(&b).ok()`, and the `.ok()` was the
/// whole defect: a `.inf_skel` written before the v2 `limits` tail fails to
/// decode (bincode is positional), the `None` propagates, the skeleton registry
/// stays empty, and the entity's state machine advances while publishing no pose.
/// The character silently stops animating and its sockets revert to the origin —
/// the exact behaviour P24.1 exists to remove, reintroduced through a discarded
/// error.
///
/// `inf_asset::decode` names that case [`AssetError::SchemaTooOld`] and carries
/// the remedy; this puts it in front of a human, on the editor's own
/// `tracing` → Output Log path. The asset is still skipped — one stale rig must
/// not stop a session starting — but skipping it is now a reported event rather
/// than a silence.
///
/// [`AssetError::SchemaTooOld`]: inf_asset::AssetError::SchemaTooOld
fn decode_anim<T: inf_asset::AssetPayload>(guid: Uuid, bytes: &[u8]) -> Option<T> {
    match inf_asset::decode::<T>(bytes) {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::warn!(
                "simulate: animation asset {guid} ({}) did not load and will not \
                 animate anything this session: {e}",
                T::KIND.slug()
            );
            None
        }
    }
}

/// Every `.inf_anim` GUID a state machine's states play, as `Uuid`s — the Ring-1
/// spelling of [`inf_anim::StateMachine::clip_refs`].
///
/// A thin conversion and nothing else. Its **two** consumers are both in this
/// crate — the PIE payload builder (`crate::pie::build_scene_payload`) and
/// `resolve_anim_assets` below; the cook is the third consumer of the *walk*, and
/// it calls the Ring-0 `clip_refs` directly because `inf-anim` is `uuid`-free by
/// design and the cook already speaks `[u8; 16]`. That is why the walk is Ring 0
/// and this is only a spelling: before P24.1 the cook closed the edge and the PIE
/// payload did not.
pub fn machine_clip_refs(machine: &StateMachine) -> BTreeSet<Uuid> {
    machine
        .clip_refs()
        .into_iter()
        .map(Uuid::from_bytes)
        .collect()
}

/// Resolve every `.inf_audio` clip an [`AudioSource`] in `doc` references, keyed by
/// asset GUID (P12.3). `resolve_audio` yields the asset's raw payload bytes by GUID
/// (backed by the project DB / pack). A clip whose bytes don't decode is skipped.
/// Deterministic (`BTreeMap`/Guid order). The caller seeds
/// [`SimSession::set_audio_clips`] — the editor Simulate twin of the player's
/// audio resolution, so preview == shipped.
pub fn resolve_audio_assets<H>(doc: &SceneDoc, mut resolve_audio: H) -> BTreeMap<Uuid, AudioAsset>
where
    H: FnMut(Uuid) -> Option<Vec<u8>>,
{
    use std::collections::btree_map::Entry;
    let mut clips: BTreeMap<Uuid, AudioAsset> = BTreeMap::new();
    let world = doc.world();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        if let Some(clip_guid) = world.world().get::<AudioSource>(e).and_then(|s| s.clip) {
            if let Entry::Vacant(v) = clips.entry(clip_guid) {
                if let Some(asset) =
                    resolve_audio(clip_guid).and_then(|b| decode_anim::<AudioAsset>(clip_guid, &b))
                {
                    v.insert(asset);
                }
            }
        }
    }
    clips
}

/// Resolve every `.inf_cloth` garment a [`ClothSim`] entity in `doc` references,
/// keyed by **asset** GUID (P24.4) — the editor Simulate twin of the player's
/// `InfSceneWorldBuilder::with_cloth_assets`, so preview == shipped.
///
/// `resolve_cloth` yields the asset's raw payload bytes by GUID (backed by the
/// project DB / the pack). A garment whose bytes do not decode is skipped with a
/// warning and its wearer simulates nothing, which is `inf_ecs::cloth`'s rule 2.
/// Deterministic (`BTreeMap` / document order).
///
/// **Keyed by asset, not by entity**, unlike [`resolve_voxel_volumes`]: two
/// characters can wear the same garment and each gets its own *state* while
/// sharing one *description*, so the description is resolved once. The per-entity
/// half is `ClothStateRes`, which lives in the sim world.
///
/// A `ClothSim` with `enabled == false` is still resolved. Disabling a garment is
/// a per-frame gameplay decision the fixed step makes, and re-loading an asset the
/// moment it is re-enabled would put a file read inside a fixed step.
pub fn resolve_cloth_assets<H>(
    doc: &SceneDoc,
    mut resolve_cloth: H,
) -> BTreeMap<Uuid, inf_anim::ClothAsset>
where
    H: FnMut(Uuid) -> Option<Vec<u8>>,
{
    use std::collections::btree_map::Entry;
    let mut out: BTreeMap<Uuid, inf_anim::ClothAsset> = BTreeMap::new();
    let world = doc.world();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let Some(asset_guid) = world
            .world()
            .get::<inf_ecs::components::ClothSim>(e)
            .and_then(|c| c.asset)
        else {
            continue;
        };
        if let Entry::Vacant(v) = out.entry(asset_guid) {
            if let Some(asset) = resolve_cloth(asset_guid)
                .and_then(|b| decode_anim::<inf_anim::ClothAsset>(asset_guid, &b))
            {
                v.insert(asset);
            }
        }
    }
    out
}

/// Resolve every `.inf_hair` hairstyle a [`HairGuides`] entity in `doc`
/// references, keyed by **asset** GUID (P24.4) - the twin of
/// [`resolve_cloth_assets`], with the same keying rule and the same reason.
pub fn resolve_hair_assets<H>(
    doc: &SceneDoc,
    mut resolve_hair: H,
) -> BTreeMap<Uuid, inf_anim::HairAsset>
where
    H: FnMut(Uuid) -> Option<Vec<u8>>,
{
    use std::collections::btree_map::Entry;
    let mut out: BTreeMap<Uuid, inf_anim::HairAsset> = BTreeMap::new();
    let world = doc.world();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let Some(asset_guid) = world
            .world()
            .get::<inf_ecs::components::HairGuides>(e)
            .and_then(|h| h.asset)
        else {
            continue;
        };
        if let Entry::Vacant(v) = out.entry(asset_guid) {
            if let Some(asset) = resolve_hair(asset_guid)
                .and_then(|b| decode_anim::<inf_anim::HairAsset>(asset_guid, &b))
            {
                v.insert(asset);
            }
        }
    }
    out
}

/// Resolve every `.inf_voxel` volume a [`VoxelVolume`] entity in `doc` references,
/// keyed by the **entity's** `Guid` (P21.2). `resolve_voxel` yields the asset's raw
/// payload bytes by GUID (backed by the project DB / pack). A volume whose bytes
/// don't parse is skipped with a warning. Deterministic (`BTreeMap` / document
/// order). The caller seeds [`SimSession::set_voxel_volumes`] — the editor Simulate
/// twin of the player's resolution, so preview == shipped.
///
/// Keyed by entity rather than by asset because two entities may reference one
/// `.inf_voxel` at two different transforms, and the world anchor `sim_volume`
/// folds in is per-entity. The transform is read **once, here**, which is the
/// honest limitation: a volume whose entity is moved during play keeps the anchor
/// it entered with, exactly as the physics bridge keeps the body it mirrored at
/// `enter`. A moving cave is a P21.3 authoring concern and will re-seed.
pub fn resolve_voxel_volumes<H>(doc: &SceneDoc, mut resolve_voxel: H) -> BTreeMap<Uuid, VoxelData>
where
    H: FnMut(Uuid) -> Option<Vec<u8>>,
{
    let mut out: BTreeMap<Uuid, VoxelData> = BTreeMap::new();
    let world = doc.world();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let w = world.world();
        let Some(asset) = w
            .get::<inf_ecs::components::VoxelVolume>(e)
            .and_then(|v| v.asset)
        else {
            continue;
        };
        let translation = w
            .get::<GlobalTransform>(e)
            .map(|g| g.translation())
            .or_else(|| w.get::<Transform>(e).map(|t| t.translation.to_dvec3()))
            .unwrap_or(DVec3::ZERO);
        let Some(bytes) = resolve_voxel(asset) else {
            continue;
        };
        match inf_voxel::sim_volume(&bytes, translation) {
            Ok(data) => {
                out.insert(guid, data);
            }
            Err(e) => tracing::warn!("inf-editor-core: bad .inf_voxel {asset}: {e}"),
        }
    }
    out
}

/// Fold the editor's **unsaved carves** into a resolved sim volume map — the
/// `ScenePersist::Memory` law (P16.4b), for voxels.
///
/// [`resolve_voxel_volumes`] reads `.inf_voxel` payloads off disk, so on its own
/// it hands Simulate the *last saved* cave. A streamed terrain does not behave
/// that way and must not: `SimSession::enter` snapshots the document with
/// [`ScenePersist::Memory`] precisely so a session sees the sculpts an author has
/// not saved yet. A carve is the same act on the other surface — and it is worse
/// unsaved, because a tunnel the author just dug is a tunnel they immediately
/// press Play to walk through, and the disk copy is still solid rock. The
/// difference is only that a terrain's working set lives in the `SceneDoc` (so
/// the snapshot carries it) while chunks cannot: scene schema v19 is frozen, so
/// the live store is a separate object and this is what reaches into it.
///
/// `live` yields the editor store's current [`VoxelData`] for an entity — in the
/// editor that is `EditorVoxelVolumes::slot(entity).data`.
///
/// # Why the DIRTY set, and why that is still camera-free
///
/// Only chunks the store marks dirty are copied, and dirty means "carved since
/// the last write-back" — a function of the **edit history**, never of residency:
/// `VoxelData::sync_residency` refuses to evict a dirty chunk and reports it as
/// `retained_dirty` (`sync_residency_never_evicts_a_dirty_chunk`), so a carve
/// cannot be paged out from under this. That is what keeps the determinism seam
/// on [`SimSession::set_voxel_volumes`] intact: the sim's map still does not
/// depend on where the editor camera is, it depends on what was dug.
///
/// A dirty key with **no resident chunk** is a deletion (see
/// `VoxelData::evict_chunk`'s note), and is replayed as one — a chunk the author
/// removed must not survive into Simulate just because the disk copy still has it.
///
/// Volumes the resolver could not produce are skipped rather than invented: a
/// carve has no world anchor without the asset it was cut into. So is a live
/// volume whose voxel size disagrees with the resolved one — those are two
/// different grids, and copying a chunk between them would place its samples
/// somewhere nobody carved.
///
/// Returns how many chunks were overlaid (0 = "the disk copy was already
/// current"), which is what a caller logs and a test asserts against.
pub fn overlay_unsaved_carves<'a, F>(volumes: &mut BTreeMap<Uuid, VoxelData>, mut live: F) -> usize
where
    F: FnMut(Uuid) -> Option<&'a VoxelData>,
{
    let mut applied = 0usize;
    for (&entity, sim) in volumes.iter_mut() {
        let Some(store) = live(entity) else { continue };
        if store.voxel_size_m() != sim.voxel_size_m() {
            tracing::warn!(
                "inf-editor-core: live voxel volume {entity} is {} m/voxel but the \
                 resolved one is {} m — unsaved carves NOT carried into Simulate",
                store.voxel_size_m(),
                sim.voxel_size_m()
            );
            continue;
        }
        for key in store.dirty_chunks() {
            match store.get_chunk(key) {
                Some(chunk) => sim.insert_resident_chunk(key, chunk.clone()),
                None => {
                    sim.evict_chunk(key);
                }
            }
            applied += 1;
        }
    }
    applied
}

#[cfg(test)]
mod voxel_sim_tests {
    //! P21.2 audit: an unsaved carve is standable in Simulate.

    use super::*;
    use inf_voxel::{ChunkKey, VoxelChunk, VoxelOp, VoxelShape};

    /// A 2 × 2 × 2-chunk block of solid rock at 0.5 m voxels, anchored at the
    /// world origin — the "saved" state both maps start from.
    fn solid_block() -> VoxelData {
        let mut v = VoxelData::new(0.5);
        for key in inf_voxel::chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
            v.insert_chunk(key, VoxelChunk::solid(1));
        }
        v.clear_dirty(); // it is what is on disk
        v
    }

    /// **THE M2 REGRESSION: Simulate must see the carve that has not been saved.**
    ///
    /// The ground query is the observable, because it is the one gameplay uses:
    /// standing over the carved column, the floor must drop to the cave floor. On
    /// the disk copy it is still the top of the rock.
    #[test]
    fn an_unsaved_carve_is_standable_in_simulate() {
        let entity = Uuid::from_u128(1);
        // What Simulate resolves off disk …
        let mut sim: BTreeMap<Uuid, VoxelData> = BTreeMap::new();
        sim.insert(entity, solid_block());
        // … and what the editor is actually holding: the same volume with a shaft
        // carved down through the middle of it, unsaved.
        let mut store = solid_block();
        let (report, _) = store.apply_op(&VoxelOp::carve(VoxelShape::Sphere {
            center: glam::DVec3::new(8.0, 12.0, 8.0),
            radius_m: 5.0,
        }));
        assert!(report.total_carved() > 0, "the fixture carved nothing");
        assert!(!store.dirty_chunks().is_empty());

        // Directly under the shaft: the carve breaches the rock's top face there,
        // so the topmost surface drops from the roof to the cave floor.
        let (x, z) = (8.0, 8.0);
        let saved = inf_voxel::voxel_surface_y_at(&sim[&entity], x, z).expect("rock has a top");
        let carved = inf_voxel::voxel_surface_y_at(&store, x, z).expect("the cave has a floor");
        assert!(
            carved < saved - 0.5,
            "the fixture must actually lower the ground ({carved} vs {saved})"
        );

        let applied = overlay_unsaved_carves(&mut sim, |e| (e == entity).then_some(&store));
        assert!(applied > 0, "no chunk was overlaid at all");
        assert_eq!(
            inf_voxel::voxel_surface_y_at(&sim[&entity], x, z),
            Some(carved),
            "Simulate is standing on the disk copy — the unsaved carve is invisible \
             to it, which is the P16.4b ScenePersist::Memory law broken for voxels"
        );
    }

    /// The overlay is a function of the **edit history**, not of residency: a
    /// carve that has been paged around still lands, and a clean volume changes
    /// nothing at all.
    #[test]
    fn the_overlay_is_camera_free_and_a_clean_store_is_a_no_op() {
        let entity = Uuid::from_u128(1);
        let clean = solid_block();
        let mut sim: BTreeMap<Uuid, VoxelData> = BTreeMap::new();
        sim.insert(entity, solid_block());
        let before = sim[&entity].clone();
        assert_eq!(overlay_unsaved_carves(&mut sim, |_| Some(&clean)), 0);
        assert_eq!(
            sim[&entity].chunk_count(),
            before.chunk_count(),
            "a store with nothing unsaved must not move the sim's volumes"
        );

        // A volume the resolver never produced is skipped rather than invented —
        // a carve with no asset behind it has no world anchor to hang off.
        let mut carved = solid_block();
        carved.apply_op(&VoxelOp::carve(VoxelShape::Sphere {
            center: glam::DVec3::new(8.0, 12.0, 8.0),
            radius_m: 5.0,
        }));
        let mut empty: BTreeMap<Uuid, VoxelData> = BTreeMap::new();
        assert_eq!(overlay_unsaved_carves(&mut empty, |_| Some(&carved)), 0);
        assert!(empty.is_empty());

        // A live volume on a different grid is refused: copying a chunk between
        // two voxel sizes would place its samples where nobody carved.
        let mut other = VoxelData::new(0.25);
        other.insert_chunk(ChunkKey::new(0, 0, 0), VoxelChunk::solid(1));
        assert!(!other.dirty_chunks().is_empty());
        let mut sim2: BTreeMap<Uuid, VoxelData> = BTreeMap::new();
        sim2.insert(entity, solid_block());
        assert_eq!(overlay_unsaved_carves(&mut sim2, |_| Some(&other)), 0);
    }
}

#[cfg(test)]
mod debug_tests {
    //! B-P4 tier A′: the Simulate debug seam (per-class `InterpDebug` +
    //! `take_debug_events`), exercised over the committed coyote platformer.
    use super::*;
    use crate::samples::{platformer_actors, platformer_scene};
    use inf_blueprint::LocalId;

    #[test]
    fn debug_config_collects_hits_and_wires() {
        let mut doc = platformer_scene();
        let actors = platformer_actors();
        let class_ids: Vec<String> = actors.iter().map(|(_, c)| c.id.clone()).collect();
        let mut session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);
        // Breakpoint on the player's `grounded` local (LocalId 2 in coyote_tick_fn)
        // + capture all wires, for every actor class.
        for id in &class_ids {
            session.set_debug(
                id.clone(),
                InterpDebug {
                    breakpoints: [LocalId(2)].into_iter().collect(),
                    capture_wires: true,
                },
            );
        }
        session.step_once(&mut doc, SimInput::with_down(["right"]));
        let events = session.take_debug_events();
        let ev = events
            .iter()
            .find(|e| e.event == "tick" && e.hits.contains(&2))
            .expect("a tick debug event that hit the `grounded` breakpoint");
        assert!(!ev.wires.is_empty(), "wire capture should record values");
        assert!(!ev.fn_name.is_empty(), "the handler fn name is recorded");
        // Draining is one-shot.
        assert!(
            session.take_debug_events().is_empty(),
            "take_debug_events drains"
        );
    }

    #[test]
    fn no_debug_config_collects_nothing() {
        // Without set_debug, stepping records no debug events (the player path is
        // debug-free by default; the shipped RuntimeSim has no setter at all).
        let mut doc = platformer_scene();
        let actors = platformer_actors();
        let mut session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);
        session.step_once(&mut doc, SimInput::with_down(["right"]));
        assert!(session.take_debug_events().is_empty());
    }
}
