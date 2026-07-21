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

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec2;
use uuid::Uuid;

use inf_blueprint::interp::{MoveResult2d, Physics2dHost, RayHit2d};
use inf_blueprint::semantics::run_event;
use inf_blueprint::{ActorInstance, BlueprintClass, EventKind, Host, InterpDebug, RunError, Value};
use inf_ecs::components::{CharacterController2D, Collider2D, ColliderShape2DKind, Transform};
use inf_ecs::{EcsWorld, Guid};
use inf_physics::{CharacterMover2D, ColliderShape2D, FixedStepper, PhysicsBridge2D};

use crate::scene::serialize::{apply_to_doc, to_scene_file, SceneFile};
use crate::scene::SceneDoc;

/// The default Simulate/physics rate (fixed updates per second).
pub const SIM_HZ: f64 = 60.0;

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

/// A live in-editor Simulate session over one [`SceneDoc`].
pub struct SimSession {
    bridge: PhysicsBridge2D,
    stepper: FixedStepper,
    /// Actors keyed by `Guid` (deterministic iteration).
    actors: BTreeMap<Uuid, ActorState>,
    /// Blueprint `i64` entity id → its `Guid`.
    entities: BTreeMap<i64, Uuid>,
    /// The world state captured at [`enter`](Self::enter), restored on
    /// [`exit`](Self::exit).
    snapshot: SceneFile,
    /// Currently-held keys/actions.
    input: SimInput,
    /// Keys/actions held the previous tick (for rising-edge detection).
    prev_down: BTreeSet<String>,
    /// Rising edges pending this fixed step (consumed after the first step).
    just_pressed: BTreeSet<String>,
    /// Accumulated `debug.print` output (surfaced to the log panel).
    logs: Vec<String>,
    /// Last `move_and_slide` grounded result per actor (a debug/telemetry read).
    grounded: BTreeMap<Uuid, bool>,
    /// Total fixed steps run (a determinism/telemetry counter).
    steps: u64,
}

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
            stepper: FixedStepper::from_hz(hz),
            actors: states,
            entities,
            snapshot,
            input: SimInput::default(),
            prev_down: BTreeSet::new(),
            just_pressed: BTreeSet::new(),
            logs: Vec::new(),
            grounded: BTreeMap::new(),
            steps: 0,
        };

        session.bridge.sync_from_world(doc.world());
        session.run_all(doc, &EventKind::BeginPlay);
        session
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

    // ── internal ──────────────────────────────────────────────────────────

    /// Latch the new input and compute rising edges vs. the previous tick.
    fn set_input(&mut self, input: SimInput) {
        self.just_pressed = input.down.difference(&self.prev_down).cloned().collect();
        self.prev_down = input.down.clone();
        self.input = input;
    }

    /// The four-phase fixed step (see the module docs).
    fn fixed_step(&mut self, doc: &mut SceneDoc) {
        let dt = self.stepper.fixed_dt();
        // 1. ECS → physics.
        self.bridge.sync_from_world(doc.world());
        // 2. Blueprint Tick for every actor (Guid order).
        let tick_args: BTreeMap<String, Value> = [("dt".to_string(), Value::Float(dt))].into();
        let args: std::collections::HashMap<String, Value> = tick_args.into_iter().collect();
        self.run_all_with_args(doc, &EventKind::Tick, &args);
        // 3. Solver.
        self.bridge.step(dt);
        // 4. Physics → ECS.
        self.bridge.write_back(doc.world_mut());
        doc.world_mut().propagate();
        // 5. Advance skeletal-animation play-heads (P11.1). Order-independent
        //    per-entity `t` integration → deterministic at the fixed `dt`.
        inf_ecs::anim::advance_anim_players(doc.world_mut(), dt);
        // Rising edges are one fixed step wide.
        self.just_pressed.clear();
        self.steps += 1;
    }

    /// Fire `event` (no args) on every actor.
    fn run_all(&mut self, doc: &mut SceneDoc, event: &EventKind) {
        let args = std::collections::HashMap::new();
        self.run_all_with_args(doc, event, &args);
    }

    /// Fire `event` on every actor in `Guid` order, each through a fresh
    /// [`SimHost`]. Actors are lifted out of the map during their run so the
    /// per-actor borrow doesn't alias the session's other fields.
    fn run_all_with_args(
        &mut self,
        doc: &mut SceneDoc,
        event: &EventKind,
        args: &std::collections::HashMap<String, Value>,
    ) {
        let guids: Vec<Uuid> = self.actors.keys().copied().collect();
        for guid in guids {
            let Some(mut state) = self.actors.remove(&guid) else {
                continue;
            };
            {
                let mut host = SimHost {
                    bridge: &mut self.bridge,
                    world: doc.world_mut(),
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

/// The engine [`Host`] a Simulate tick runs against: routes `input.*` to the
/// held-key state, `debug.print` to the log, and exposes the physics world via
/// [`Host::physics`]. `vars::*` are handled one layer up by
/// [`ActorHost`](inf_blueprint::ActorHost).
struct SimHost<'a> {
    bridge: &'a mut PhysicsBridge2D,
    world: &'a mut EcsWorld,
    input: &'a SimInput,
    just_pressed: &'a BTreeSet<String>,
    entities: &'a BTreeMap<i64, Uuid>,
    logs: &'a mut Vec<String>,
    grounded: &'a mut BTreeMap<Uuid, bool>,
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
            (Some("debug"), Some("print")) => {
                self.logs.push(arg_str(args, 0));
                Ok(Value::Unit)
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
}

impl SimHost<'_> {
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

/// A `Guid` marker used to look up entities during Simulate (re-export helper so
/// callers building an actor list don't reach into `inf_ecs` directly).
pub type ActorGuid = Guid;
