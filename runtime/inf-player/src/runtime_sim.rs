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

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_blueprint::interp::{MoveResult2d, Physics2dHost, RayHit2d};
use inf_blueprint::semantics::run_event;
use inf_blueprint::{ActorInstance, BlueprintClass, EventKind, Host, InterpDebug, RunError, Value};
use inf_ecs::components::{CharacterController2D, Collider2D, ColliderShape2DKind, Transform};
use inf_ecs::{sim_snapshot, EcsWorld};
use inf_physics::{CharacterMover2D, ColliderShape2D, PhysicsBridge2D};
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

/// A headless gameplay simulation over one owned [`EcsWorld`].
pub struct RuntimeSim {
    world: EcsWorld,
    bridge: PhysicsBridge2D,
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
    /// Total fixed steps run.
    steps: u64,
    /// World-space translations one fixed step ago, for render interpolation.
    prev_positions: HashMap<Uuid, DVec3>,
    /// World-space translations at the current fixed step.
    cur_positions: HashMap<Uuid, DVec3>,
}

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
            stepper: FixedStep::from_hz(hz),
            actors: states,
            entities,
            input: RuntimeInput::default(),
            prev_down: BTreeSet::new(),
            just_pressed: BTreeSet::new(),
            logs: Vec::new(),
            grounded: BTreeMap::new(),
            steps: 0,
            prev_positions: HashMap::new(),
            cur_positions: HashMap::new(),
        };

        sim.bridge.sync_from_world(&sim.world);
        sim.run_all(&EventKind::BeginPlay);
        sim.capture_positions();
        sim.prev_positions = sim.cur_positions.clone();
        sim
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
        // 2. Blueprint Tick for every actor (Guid order).
        let args: HashMap<String, Value> = [("dt".to_string(), Value::Float(dt))].into();
        self.run_all_with_args(&EventKind::Tick, &args);
        // 3. Solver.
        self.bridge.step(dt);
        // 4. Physics → ECS.
        self.bridge.write_back(&mut self.world);
        self.world.propagate();
        // Roll interpolation history + rising edges.
        std::mem::swap(&mut self.prev_positions, &mut self.cur_positions);
        self.capture_positions();
        self.just_pressed.clear();
        self.steps += 1;
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
                    world: &mut self.world,
                    input: &self.input,
                    just_pressed: &self.just_pressed,
                    entities: &self.entities,
                    logs: &mut self.logs,
                    grounded: &mut self.grounded,
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
    world: &'a mut EcsWorld,
    input: &'a RuntimeInput,
    just_pressed: &'a BTreeSet<String>,
    entities: &'a BTreeMap<i64, Uuid>,
    logs: &'a mut Vec<String>,
    grounded: &'a mut BTreeMap<Uuid, bool>,
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
}

impl RuntimeHost<'_> {
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
