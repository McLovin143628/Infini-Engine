//! Wave 3 runtime parity: the shipped player's `runtime_sim` runs the identical
//! trigger-volume / blocking-volume / input-event / dispatcher scenes as the
//! editor `SimSession` (see `editor/crates/inf-editor-core/tests/simulate_volumes.rs`).
//! The two files build the same worlds with the same guids and assert the same
//! numbers — preview == shipped for event plumbing. Pure Ring-0, headless CI.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_blueprint::{
    BinOp, BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty, Value,
    Variable,
};
use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Transform, Volume, VolumeKind,
};
use inf_ecs::math::Vec3d;
use inf_ecs::{EcsWorld, Entity};
use inf_player::runtime_sim::{OverlapEvent, OverlapPhase, RuntimeInput, RuntimeSim};

// ── shared scene guids (identical to the editor twin so the pairs match) ───────
const SENSOR_GUID: u128 = 0x0000_0000_0000_0000_0000_0000_0000_0001;
const BODY_GUID: u128 = 0x0000_0000_0000_0000_0000_0000_0000_0002;
/// Pinned identically to the editor twin — this equality is the parity gate.
const EXPECTED_BEGIN_STEP: u64 = 40;

fn body_y(sim: &RuntimeSim, guid: Uuid) -> f64 {
    let e = sim.world().entity_of(guid).expect("entity");
    sim.world()
        .world()
        .get::<Transform>(e)
        .expect("transform")
        .translation
        .y
}

fn spawn(world: &mut EcsWorld, guid: Uuid, name: &str) -> Entity {
    world.spawn_with_guid(guid, name, None)
}

fn add_dynamic_cube(world: &mut EcsWorld, guid: Uuid, y: f64) {
    let e = spawn(world, guid, "Cube");
    let w = world.world_mut();
    w.entity_mut(e)
        .insert(Transform::from_translation(DVec3::new(0.0, y, 0.0)));
    w.entity_mut(e).insert(RigidBody3D {
        kind: BodyKind3D::Dynamic,
        ..Default::default()
    });
    w.entity_mut(e).insert(Collider3D {
        shape_kind: ColliderShape3DKind::Box,
        half_extents: Vec3d::new(0.5, 0.5, 0.5),
        ..Default::default()
    });
}

fn add_volume(world: &mut EcsWorld, guid: Uuid, sensor: bool) {
    let e = spawn(world, guid, "Volume");
    let w = world.world_mut();
    w.entity_mut(e)
        .insert(Transform::from_translation(DVec3::ZERO));
    w.entity_mut(e).insert(RigidBody3D {
        kind: BodyKind3D::Static,
        ..Default::default()
    });
    w.entity_mut(e).insert(Collider3D {
        shape_kind: ColliderShape3DKind::Box,
        half_extents: Vec3d::new(2.0, 0.5, 2.0),
        sensor,
        ..Default::default()
    });
    w.entity_mut(e).insert(Volume {
        kind: if sensor {
            VolumeKind::Trigger
        } else {
            VolumeKind::Blocking
        },
        ..Default::default()
    });
}

// ── blocking volume ────────────────────────────────────────────────────────────

#[test]
fn runtime_blocking_volume_rests_a_falling_body_on_top() {
    let mut world = EcsWorld::new();
    add_volume(&mut world, Uuid::from_u128(SENSOR_GUID), false);
    add_dynamic_cube(&mut world, Uuid::from_u128(BODY_GUID), 3.0);
    world.mark_dirty();

    let mut sim = RuntimeSim::new(world, vec![], DVec2::new(0.0, -9.81), 60.0);
    for _ in 0..180 {
        sim.step_once(RuntimeInput::default());
    }
    let y = body_y(&sim, Uuid::from_u128(BODY_GUID));
    assert!(
        (y - 1.0).abs() < 0.05,
        "blocking volume should rest the cube at y≈1.0, got {y}"
    );
}

// ── trigger (sensor) volume ────────────────────────────────────────────────────

#[test]
fn runtime_trigger_volume_reports_a_sensor_overlap_begin() {
    let mut world = EcsWorld::new();
    add_volume(&mut world, Uuid::from_u128(SENSOR_GUID), true);
    add_dynamic_cube(&mut world, Uuid::from_u128(BODY_GUID), 3.0);
    world.mark_dirty();

    let mut sim = RuntimeSim::new(world, vec![], DVec2::new(0.0, -9.81), 60.0);

    let mut begin: Option<(u64, Vec<OverlapEvent>)> = None;
    for _ in 0..240 {
        sim.step_once(RuntimeInput::default());
        let begins: Vec<OverlapEvent> = sim
            .drained_overlaps()
            .iter()
            .copied()
            .filter(|o| o.phase == OverlapPhase::Begin)
            .collect();
        if !begins.is_empty() {
            begin = Some((sim.steps(), begins));
            break;
        }
    }
    let (step, list) = begin.expect("no sensor overlap Begin observed in 240 steps");
    let a = Uuid::from_u128(SENSOR_GUID).min(Uuid::from_u128(BODY_GUID));
    let b = Uuid::from_u128(SENSOR_GUID).max(Uuid::from_u128(BODY_GUID));
    assert_eq!(
        list,
        vec![OverlapEvent {
            a,
            b,
            phase: OverlapPhase::Begin
        }]
    );
    assert_eq!(
        step, EXPECTED_BEGIN_STEP,
        "runtime begin step must match the editor twin"
    );
}

// ── input events ────────────────────────────────────────────────────────────────

fn jump_counter_class() -> BlueprintClass {
    let get = |name: &str| Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str(name.into()))],
    };
    let body = vec![Stmt::If {
        cond: Expr::Param("pressed".into()),
        then_body: vec![Stmt::ExprStmt(Expr::Call {
            path: vec!["vars".into(), "set".into()],
            args: vec![
                Expr::Lit(Lit::Str("count".into())),
                Expr::Binary(
                    BinOp::Add,
                    Box::new(get("count")),
                    Box::new(Expr::Lit(Lit::Int(1))),
                ),
            ],
        })],
        else_body: vec![],
    }];
    let mut class = BlueprintClass::new("act:jumper", "Jumper");
    class.variables = vec![Variable {
        name: "count".into(),
        ty: Ty::Int,
        default: Lit::Int(0),
        exposed: false,
    }];
    class.events = vec![EventBinding {
        event: EventKind::Input("jump".into()),
        body: BlueprintFn {
            id: "input:jump".into(),
            name: "input_jump".into(),
            params: vec![Param {
                name: "pressed".into(),
                ty: Ty::Bool,
            }],
            ret: Ty::Unit,
            body,
        },
    }];
    class
}

#[test]
fn runtime_input_event_press_fires_once_per_edge() {
    let actor = Uuid::from_u128(0x00A0_0001);
    let mut world = EcsWorld::new();
    spawn(&mut world, actor, "Jumper");
    world.mark_dirty();

    let mut sim = RuntimeSim::new(
        world,
        vec![(actor, jump_counter_class())],
        DVec2::ZERO,
        60.0,
    );
    let count = |s: &RuntimeSim| match s.actor_var(actor, "count") {
        Some(Value::Int(n)) => *n,
        _ => -1,
    };

    sim.step_once(RuntimeInput::with_down(["jump"]));
    assert_eq!(count(&sim), 1);
    sim.step_once(RuntimeInput::with_down(["jump"]));
    assert_eq!(count(&sim), 1, "held key does not re-fire");
    sim.step_once(RuntimeInput::default());
    assert_eq!(count(&sim), 1, "release edge does not increment");
    sim.step_once(RuntimeInput::with_down(["jump"]));
    assert_eq!(count(&sim), 2);
}

// ── dispatchers ────────────────────────────────────────────────────────────────

fn call_stmt(path: &[&str], args: Vec<Expr>) -> Stmt {
    Stmt::ExprStmt(Expr::Call {
        path: path.iter().map(|s| s.to_string()).collect(),
        args,
    })
}

fn incr_stmt(name: &str) -> Stmt {
    Stmt::ExprStmt(Expr::Call {
        path: vec!["vars".into(), "set".into()],
        args: vec![
            Expr::Lit(Lit::Str(name.into())),
            Expr::Binary(
                BinOp::Add,
                Box::new(Expr::Call {
                    path: vec!["vars".into(), "get".into()],
                    args: vec![Expr::Lit(Lit::Str(name.into()))],
                }),
                Box::new(Expr::Lit(Lit::Int(1))),
            ),
        ],
    })
}

fn int_var(name: &str) -> Variable {
    Variable {
        name: name.into(),
        ty: Ty::Int,
        default: Lit::Int(0),
        exposed: false,
    }
}

fn handler_fn(id: &str, name: &str, params: Vec<Param>, body: Vec<Stmt>) -> BlueprintFn {
    BlueprintFn {
        id: id.into(),
        name: name.into(),
        params,
        ret: Ty::Unit,
        body,
    }
}

fn emitter_class() -> BlueprintClass {
    let mut class = BlueprintClass::new("act:emitter", "Emitter");
    class.variables = vec![int_var("self_ping")];
    class.events = vec![
        EventBinding {
            event: EventKind::Tick,
            body: handler_fn(
                "tick",
                "tick",
                vec![Param {
                    name: "dt".into(),
                    ty: Ty::Float,
                }],
                vec![call_stmt(
                    &["event", "dispatch"],
                    vec![Expr::Lit(Lit::Int(1)), Expr::Lit(Lit::Str("ping".into()))],
                )],
            ),
        },
        EventBinding {
            event: EventKind::Custom("ping".into()),
            body: handler_fn(
                "custom:ping",
                "custom_ping",
                vec![],
                vec![incr_stmt("self_ping")],
            ),
        },
    ];
    class
}

fn listener_class() -> BlueprintClass {
    let mut class = BlueprintClass::new("act:listener", "Listener");
    class.variables = vec![int_var("reacted")];
    class.events = vec![
        EventBinding {
            event: EventKind::BeginPlay,
            body: handler_fn(
                "begin_play",
                "begin_play",
                vec![],
                vec![call_stmt(
                    &["event", "bind"],
                    vec![
                        Expr::Lit(Lit::Int(1)),
                        Expr::Lit(Lit::Str("ping".into())),
                        Expr::Lit(Lit::Str("react".into())),
                    ],
                )],
            ),
        },
        EventBinding {
            event: EventKind::Custom("react".into()),
            body: handler_fn(
                "custom:react",
                "custom_react",
                vec![],
                vec![incr_stmt("reacted")],
            ),
        },
    ];
    class
}

#[test]
fn runtime_dispatch_fires_target_and_bound_listener() {
    let emitter = Uuid::from_u128(0x00D0_0001);
    let listener = Uuid::from_u128(0x00D0_0002);
    let mut world = EcsWorld::new();
    spawn(&mut world, emitter, "Emitter");
    spawn(&mut world, listener, "Listener");
    world.mark_dirty();

    let mut sim = RuntimeSim::new(
        world,
        vec![(emitter, emitter_class()), (listener, listener_class())],
        DVec2::ZERO,
        60.0,
    );
    let getv = |s: &RuntimeSim, g: Uuid, n: &str| match s.actor_var(g, n) {
        Some(Value::Int(v)) => *v,
        _ => -1,
    };

    sim.step_once(RuntimeInput::default());
    assert_eq!(getv(&sim, emitter, "self_ping"), 1);
    assert_eq!(getv(&sim, listener, "reacted"), 1);
    sim.step_once(RuntimeInput::default());
    assert_eq!(getv(&sim, emitter, "self_ping"), 2);
    assert_eq!(getv(&sim, listener, "reacted"), 2);
}

fn looping_class() -> BlueprintClass {
    let mut class = BlueprintClass::new("act:looper", "Looper");
    class.variables = vec![int_var("n")];
    class.events = vec![
        EventBinding {
            event: EventKind::BeginPlay,
            body: handler_fn(
                "begin_play",
                "begin_play",
                vec![],
                vec![call_stmt(
                    &["event", "dispatch"],
                    vec![Expr::Lit(Lit::Int(1)), Expr::Lit(Lit::Str("loop".into()))],
                )],
            ),
        },
        EventBinding {
            event: EventKind::Custom("loop".into()),
            body: handler_fn(
                "custom:loop",
                "custom_loop",
                vec![],
                vec![
                    incr_stmt("n"),
                    call_stmt(
                        &["event", "dispatch"],
                        vec![Expr::Lit(Lit::Int(1)), Expr::Lit(Lit::Str("loop".into()))],
                    ),
                ],
            ),
        },
    ];
    class
}

#[test]
fn runtime_dispatch_cycle_is_capped_at_64() {
    let looper = Uuid::from_u128(0x00E0_0001);
    let mut world = EcsWorld::new();
    spawn(&mut world, looper, "Looper");
    world.mark_dirty();

    let sim = RuntimeSim::new(world, vec![(looper, looping_class())], DVec2::ZERO, 60.0);
    let n = match sim.actor_var(looper, "n") {
        Some(Value::Int(v)) => *v,
        _ => -1,
    };
    assert_eq!(n, 64, "the dispatch round cap fires exactly 64 times");
    assert!(
        sim.logs().iter().any(|l| l.contains("dispatch cap")),
        "the cap logs a deterministic drop line"
    );
}
