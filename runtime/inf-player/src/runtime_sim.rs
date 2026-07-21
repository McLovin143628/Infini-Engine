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

use std::collections::{BTreeMap, BTreeSet, HashMap};

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
use inf_physics::{
    CharacterMover2D, CharacterMover3D, ColliderShape2D, ColliderShape3D, PhysicsBridge2D,
    PhysicsBridge3D,
};
use inf_runtime::FixedStep;

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
        };

        sim.bridge.sync_from_world(&sim.world);
        sim.bridge3d.sync_from_world(&sim.world); // P11.3 3D bridge
        sim.run_all(&EventKind::BeginPlay);
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

    /// Render interpolation factor in `[0, 1)` (how far into the next step the
    /// frame accumulator sits).
    pub fn alpha(&self) -> f64 {
        self.stepper.alpha()
    }

    /// The `debug.print` log accumulated so far.
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
        self.just_pressed = input.down.difference(&self.prev_down).cloned().collect();
        self.prev_down = input.down.clone();
        self.input = input;
    }

    fn fixed_step(&mut self) {
        let dt = self.stepper.fixed_dt();
        // 1. ECS → physics.
        self.bridge.sync_from_world(&self.world);
        self.bridge3d.sync_from_world(&self.world); // ── P11.3 3D bridge: sync ──
                                                    // 2. Blueprint Tick for every actor (Guid order).
        let args: HashMap<String, Value> = [("dt".to_string(), Value::Float(dt))].into();
        self.run_all_with_args(&EventKind::Tick, &args);
        // 3. Solver.
        self.bridge.step(dt);
        self.bridge3d.step(dt); // ── P11.3 3D bridge: step ──
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
        // ── P12.3 audio step ── last, observing this step's final transforms
        //    (preview == shipped: the same logic the editor SimSession runs).
        self.audio_step();
        // Roll interpolation history + rising edges.
        std::mem::swap(&mut self.prev_positions, &mut self.cur_positions);
        self.capture_positions();
        self.just_pressed.clear();
        self.steps += 1;
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

        // Autoplay once per not-yet-started `AudioSource` (Guid order).
        let mut autoplay: Vec<(Uuid, AudioSource, DVec3)> = Vec::new();
        for e in self.world.world().iter_entities() {
            let Some(src) = e.get::<AudioSource>() else {
                continue;
            };
            if !src.autoplay {
                continue;
            }
            let Some(guid) = e.get::<Guid>().map(|g| g.0) else {
                continue;
            };
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
        for (guid, src, pos) in autoplay {
            self.audio_started.insert(guid);
            let mut cmd = play_command_for(guid_source_key(guid), &src, src.spatial.then_some(pos));
            if src.occlusion && src.spatial {
                cmd.occlusion_gain = self.occlusion_gain(listener_pos, pos);
            }
            self.audio_cmds.push(AudioCommand::Play(cmd));
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

    /// Fire `event` on every actor in `Guid` order, each through a fresh
    /// [`RuntimeHost`]. Actors are lifted out of the map during their run so the
    /// per-actor borrow doesn't alias the sim's other fields.
    fn run_all_with_args(&mut self, event: &EventKind, args: &HashMap<String, Value>) {
        let guids: Vec<Uuid> = self.actors.keys().copied().collect();
        for guid in guids {
            let Some(mut state) = self.actors.remove(&guid) else {
                continue;
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
}

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
            (Some("debug"), Some("print")) => {
                self.logs.push(arg_str(args, 0));
                Ok(Value::Unit)
            }
            // terrain.height_at(x, z) → world height at that XZ (P11.4) — the same
            // seam the editor SimHost exposes (preview == shipped): a 3D character
            // reads it to stay on a heightfield terrain (no physics collider).
            (Some("terrain"), Some("height_at")) => Ok(Value::Float(terrain_height_at(
                self.world,
                arg_f64(args, 0),
                arg_f64(args, 1),
            ))),
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

fn terrain_height_at(world: &EcsWorld, x: f64, z: f64) -> f64 {
    let w = world.world();
    let mut picked: Option<(Uuid, DVec3, inf_terrain::TerrainData)> = None;
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
            picked = Some((guid, origin, t.data.clone()));
        }
    }
    match picked {
        Some((_, origin, data)) => data
            .height_at(DVec2::new(x - origin.x, z - origin.z))
            .map(|h| h + origin.y)
            .unwrap_or(0.0),
        None => 0.0,
    }
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
