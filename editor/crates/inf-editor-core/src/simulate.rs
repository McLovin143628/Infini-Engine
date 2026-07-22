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

use glam::{DQuat, DVec2, DVec3};
use uuid::Uuid;

// P11.3 root motion: the pure root-delta extractor + the clip/skeleton it reads.
use inf_anim::state_machine::{SmContext, SmRuntime, StateMachine};
use inf_anim::{root_delta, AnimClip, AnimClipAsset, Skeleton, SkeletonAsset, StateMachineAsset};
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
use inf_ecs::{update_attachments, EcsWorld, Entity, Guid};
use inf_physics::{
    CharacterMover2D, CharacterMover3D, ColliderShape2D, ColliderShape3D, ContactPhase,
    FixedStepper, PhysicsBridge2D, PhysicsBridge3D,
};

use crate::scene::serialize::{apply_to_doc, to_scene_file, SceneFile};
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
/// **held** keys/actions (e.g. `"left"`, `"jump"`). Rising edges
/// (`just_pressed`) are derived by the session from the previous tick's set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SimInput {
    down: BTreeSet<String>,
}

impl SimInput {
    /// An input state with the given keys/actions held down.
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
    logs: Vec<String>,
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
    audio_log: Vec<AudioCommand>,
    /// Total fixed steps run (a determinism/telemetry counter).
    steps: u64,
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
    pub fn enter(
        doc: &mut SceneDoc,
        actors: Vec<(Uuid, BlueprintClass)>,
        gravity: DVec2,
        hz: f64,
    ) -> Self {
        let snapshot = to_scene_file(doc);
        let bridge = PhysicsBridge2D::new(gravity);
        // P11.3: a 3D bridge alongside the 2D one. The 2D vertical gravity maps to
        // world −Y; a character applies its own gravity through move_and_slide.
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

        let mut session = Self {
            bridge,
            bridge3d,
            clips: BTreeMap::new(),
            stepper: FixedStepper::from_hz(hz),
            actors: states,
            entities,
            snapshot,
            state_machines: BTreeMap::new(),
            input: SimInput::default(),
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
        };

        session.bridge.sync_from_world(doc.world());
        session.bridge3d.sync_from_world(doc.world()); // P11.3 3D bridge
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

    /// Exit Simulate: restore the world captured at [`enter`](Self::enter). The
    /// document is byte-for-byte what it was before play.
    pub fn exit(self, doc: &mut SceneDoc) {
        apply_to_doc(doc, &self.snapshot);
    }

    /// Seed the resolvable `.inf_sm` state machines (P11.2). An entity carrying an
    /// [`AnimStateMachine`] whose `sm` GUID is present here is stepped each fixed
    /// tick against the actor's Blueprint variables.
    pub fn set_state_machines(&mut self, machines: BTreeMap<Uuid, StateMachine>) {
        self.state_machines = machines;
    }

    /// The `debug.print` log accumulated so far.
    pub fn logs(&self) -> &[String] {
        &self.logs
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
        // 1. ECS → physics.
        self.bridge.sync_from_world(doc.world());
        self.bridge3d.sync_from_world(doc.world()); // ── P11.3 3D bridge: sync ──
                                                    // ── Wave 3 input events ── fire Input(action) edges BEFORE the Tick pass,
                                                    //    then drain any dispatches they queued.
        self.fire_input_events(doc);
        self.drain_dispatch(doc);
        // 2. Blueprint Tick for every actor (Guid order).
        let tick_args: BTreeMap<String, Value> = [("dt".to_string(), Value::Float(dt))].into();
        let args: std::collections::HashMap<String, Value> = tick_args.into_iter().collect();
        self.run_all_with_args(doc, &EventKind::Tick, &args);
        self.drain_dispatch(doc); // Wave 3: Tick may dispatch custom events.
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
        // ── P12.3 audio step ── last, so it observes this step's final transforms:
        //    pick the listener, enqueue autoplay, resolve occlusion, drain the queue.
        self.audio_step(doc);
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
            // ground-plane delta by the entity's current yaw into world space.
            let yaw_rad = t.rotation.y.to_radians();
            let local = DVec3::new(d.translation.x as f64, 0.0, d.translation.z as f64);
            let world_delta = DQuat::from_rotation_y(yaw_rad) * local;
            let new_yaw_deg = t.rotation.y + d.yaw.to_degrees() as f64;
            let pos = t.translation.to_dvec3();

            let new_pos = if has_cc {
                // Drive the entity through the 3D mover so walls/steps/slopes apply.
                let mover = build_mover3d(doc.world(), guid);
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

    /// Step every entity's [`AnimStateMachine`] (P11.2) whose `sm` GUID resolves
    /// in [`state_machines`](Self::state_machines). The transition conditions and
    /// blend-space params read the entity's *actor* Blueprint variables (via the
    /// [`SmContext`] seam); an entity with no actor gets an empty variable set, so
    /// every param defaults to `0` (documented). Order-independent per entity →
    /// deterministic. Runtime state lives on the component; only `t`-like play
    /// state advances here (pose evaluation is render-time, the same placeholder
    /// gap as [`inf_ecs::components::SkeletalMesh`]).
    fn advance_state_machines(&mut self, doc: &mut SceneDoc, dt: f64) {
        if self.state_machines.is_empty() {
            return;
        }
        // Collect targets first so the mutable write-back doesn't overlap the read
        // query. `(entity, guid, sm_guid, runtime-snapshot)`.
        let mut targets: Vec<(Entity, Uuid, Uuid, SmRuntimeState)> = Vec::new();
        {
            let w = doc.world_mut().world_mut();
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
            if let Some(mut asm) = doc
                .world_mut()
                .world_mut()
                .get_mut::<AnimStateMachine>(entity)
            {
                asm.runtime = from_anim_runtime(rt);
            }
        }
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
        &self.audio_log
    }

    /// The P12.3 audio step (runs last in a fixed step): pick the listener, enqueue
    /// autoplay sources once, resolve occlusion via one physics raycast per
    /// occlusion-enabled spatial source, then drain the queue into the host engine.
    fn audio_step(&mut self, doc: &mut SceneDoc) {
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
        let mut autoplay: Vec<(Uuid, AudioSource, DVec3)> = Vec::new();
        let mut live: BTreeSet<Uuid> = BTreeSet::new();
        for e in doc.world().world().iter_entities() {
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
    logs: &'a mut Vec<String>,
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
}

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
            (Some("terrain"), Some("height_at")) => Ok(Value::Float(terrain_height_at(
                self.world,
                arg_f64(args, 0),
                arg_f64(args, 1),
            ))),
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
        let mover = build_mover3d(self.world, guid);
        let result = self.bridge3d.world_mut().move_character(
            &mover,
            pos,
            DVec3::new(motion[0], motion[1], motion[2]),
            exclude,
        );
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
/// `Collider3D`, defaulting to an upright capsule mover when absent (the `d3`
/// mirror of `SimHost::mover_for`). Shared by the `physics3d.move_and_slide` host
/// path and the root-motion applier.
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

/// Sample the world's terrain height at world `(x, z)` — the `terrain.height_at`
/// host seam (P11.4). Uses the **lowest-`Guid`** non-empty [`Terrain`] (a
/// heightfield carries no physics collider, so a 3D character reads its height
/// here to stay grounded); returns `0.0` with no terrain. Shared shape with the
/// shipped runtime host so preview == shipped. Deterministic (Guid-picked).
fn terrain_height_at(world: &EcsWorld, x: f64, z: f64) -> f64 {
    let w = world.world();
    // Pick the lowest-Guid non-empty terrain, remembering only its entity + origin —
    // never a clone of the (multi-MB) heightfield. The component is re-fetched after
    // the scan (all EntityRef borrows released) and sampled in place.
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
    match picked.and_then(|(_, origin, e)| w.get::<Terrain>(e).map(|t| (origin, t))) {
        Some((origin, t)) => t
            .data
            .height_at(DVec2::new(x - origin.x, z - origin.z))
            .map(|h| h + origin.y)
            .unwrap_or(0.0),
        None => 0.0,
    }
}

// ── P11.2 state-machine glue: ECS POD ↔ inf-anim runtime + var snapshot ──────
//
// The conversions are field-for-field (`SmRuntimeState` is a deliberate POD
// mirror of `inf_anim::SmRuntime`, kept in `inf-ecs` so that crate needs no
// `inf-anim` dep — see the `SmRuntimeState` docs). Duplicated in the shipped
// player's `runtime_sim` for the same "preview == shipped" reason `SimHost` is.

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
/// (keyed by asset GUID) and the root-motion `(clip GUID, skeleton, clip)` triples.
pub type AnimSeed = (
    BTreeMap<Uuid, StateMachine>,
    Vec<(Uuid, Skeleton, AnimClip)>,
);

/// Resolve a scene's referenced P11 animation assets into the seed maps a
/// [`SimSession`] (and the shipped [`RuntimeSim`](../../inf_player/runtime_sim/index.html))
/// need (P11.4): the `.inf_sm` state machines it steps, and the root-motion
/// `(clip GUID, skeleton, clip)` triples it registers. `resolve_anim` reads an
/// anim asset's raw bytes by GUID (the caller backs it with the project asset DB /
/// the pack). A machine/clip whose bytes don't resolve is skipped; a clip whose
/// skeleton doesn't resolve is dropped (root motion needs it). Deterministic
/// (`BTreeMap`/Guid order). The caller seeds
/// [`SimSession::set_state_machines`] + [`SimSession::register_root_motion_clip`]
/// from the result — the editor Simulate twin of the player's
/// `InfSceneWorldBuilder` anim resolution, so preview == shipped.
pub fn resolve_anim_assets<H>(doc: &SceneDoc, mut resolve_anim: H) -> AnimSeed
where
    H: FnMut(Uuid) -> Option<Vec<u8>>,
{
    use std::collections::btree_map::Entry;
    let mut machines: BTreeMap<Uuid, StateMachine> = BTreeMap::new();
    let mut clips: BTreeMap<Uuid, (Skeleton, AnimClip)> = BTreeMap::new();
    let world = doc.world();
    for &guid in doc.order() {
        let Some(e) = world.entity_of(guid) else {
            continue;
        };
        let w = world.world();
        if let Some(sm_guid) = w.get::<AnimStateMachine>(e).and_then(|s| s.sm) {
            if let Entry::Vacant(v) = machines.entry(sm_guid) {
                if let Some(asset) = resolve_anim(sm_guid)
                    .and_then(|b| inf_asset::decode::<StateMachineAsset>(&b).ok())
                {
                    v.insert(asset.machine);
                }
            }
        }
        if let Some(clip_guid) = w.get::<AnimPlayer>(e).and_then(|p| p.clip) {
            if let Entry::Vacant(v) = clips.entry(clip_guid) {
                if let Some(ca) = resolve_anim(clip_guid)
                    .and_then(|b| inf_asset::decode::<AnimClipAsset>(&b).ok())
                {
                    if let Some(sk) = ca
                        .skeleton
                        .map(Uuid::from_bytes)
                        .and_then(&mut resolve_anim)
                        .and_then(|b| inf_asset::decode::<SkeletonAsset>(&b).ok())
                    {
                        v.insert((sk.skeleton, ca.clip));
                    }
                }
            }
        }
    }
    let root_clips = clips.into_iter().map(|(g, (s, c))| (g, s, c)).collect();
    (machines, root_clips)
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
                    resolve_audio(clip_guid).and_then(|b| inf_asset::decode::<AudioAsset>(&b).ok())
                {
                    v.insert(asset);
                }
            }
        }
    }
    clips
}
