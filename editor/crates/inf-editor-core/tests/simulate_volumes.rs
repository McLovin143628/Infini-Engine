//! Wave 3 (E-P4 sim half + B-P3): end-to-end Simulate proofs for the
//! trigger-volume overlap drain, blocking volumes, input events, and the event
//! dispatchers — all through the real editor `SimSession` fixed step. The
//! **paired** runtime test `runtime/inf-player/tests/runtime_volumes.rs` builds
//! the identical scenes over `RuntimeSim`; the two assert the same numbers, which
//! is the preview == shipped parity gate for event plumbing.
//!
//! Scenes are built directly with `create_with_guid` + component inserts (the
//! platformer/character3d sample pattern). Volume entities carry a real `Volume`
//! component alongside their `Collider3D` (blocking = solid collider, trigger =
//! sensor), so this exercises the seam the volume-spawn agent's editor half emits.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_blueprint::{
    BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty, Variable,
};
use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Transform, Volume, VolumeKind,
};
use inf_ecs::math::Vec3d;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{OverlapEvent, OverlapPhase, SimInput, SimSession, SIM_HZ};

// ── shared scene guids (identical in the runtime twin so the pairs match) ──────
const SENSOR_GUID: u128 = 0x0000_0000_0000_0000_0000_0000_0000_0001;
const BODY_GUID: u128 = 0x0000_0000_0000_0000_0000_0000_0000_0002;

// ── helpers ────────────────────────────────────────────────────────────────────

macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

fn body_y(doc: &SceneDoc, guid: Uuid) -> f64 {
    let e = doc.entity_of(guid).expect("entity");
    doc.world()
        .world()
        .get::<Transform>(e)
        .expect("transform")
        .translation
        .y
}

/// A falling dynamic 1 m cube starting at `y`.
fn add_dynamic_cube(doc: &mut SceneDoc, guid: Uuid, y: f64) {
    doc.create_with_guid(guid, SpawnKind::Empty, "Cube", None);
    insert!(
        doc,
        guid,
        Transform::from_translation(DVec3::new(0.0, y, 0.0))
    );
    insert!(
        doc,
        guid,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        guid,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(0.5, 0.5, 0.5),
            ..Default::default()
        }
    );
}

/// A static volume box centred at the origin (top at y=0.5), `sensor` = trigger.
fn add_volume(doc: &mut SceneDoc, guid: Uuid, sensor: bool) {
    doc.create_with_guid(guid, SpawnKind::Empty, "Volume", None);
    insert!(doc, guid, Transform::from_translation(DVec3::ZERO));
    insert!(
        doc,
        guid,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        guid,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(2.0, 0.5, 2.0),
            sensor,
            ..Default::default()
        }
    );
    insert!(
        doc,
        guid,
        Volume {
            kind: if sensor {
                VolumeKind::Trigger
            } else {
                VolumeKind::Blocking
            },
            ..Default::default()
        }
    );
}

// ── deliverable: blocking volume stops a falling body at a known rest height ────

#[test]
fn blocking_volume_rests_a_falling_body_on_top() {
    let mut doc = SceneDoc::new();
    add_volume(&mut doc, Uuid::from_u128(SENSOR_GUID), false); // blocking
    add_dynamic_cube(&mut doc, Uuid::from_u128(BODY_GUID), 3.0);
    doc.world_mut().propagate();

    let mut session = SimSession::enter(&mut doc, vec![], DVec2::new(0.0, -9.81), SIM_HZ);
    for _ in 0..180 {
        session.step_once(&mut doc, SimInput::default());
    }
    // Cube (half 0.5) rests on the volume top (y=0.5) → centre ≈ 1.0. The same
    // constant is asserted in the runtime twin — that equality is the parity gate.
    let y = body_y(&doc, Uuid::from_u128(BODY_GUID));
    assert!(
        (y - 1.0).abs() < 0.05,
        "blocking volume should rest the cube at y≈1.0, got {y}"
    );
}

// ── deliverable: trigger (sensor) volume fires an OverlapEvent Begin ────────────

#[test]
fn trigger_volume_reports_a_sensor_overlap_begin() {
    let mut doc = SceneDoc::new();
    add_volume(&mut doc, Uuid::from_u128(SENSOR_GUID), true); // trigger/sensor
    add_dynamic_cube(&mut doc, Uuid::from_u128(BODY_GUID), 3.0);
    doc.world_mut().propagate();

    let mut session = SimSession::enter(&mut doc, vec![], DVec2::new(0.0, -9.81), SIM_HZ);

    // Step until the first Begin overlap; record the step index + the list.
    let (begin_step, list) = first_overlap_begin(&mut session, &mut doc);
    let a = Uuid::from_u128(SENSOR_GUID).min(Uuid::from_u128(BODY_GUID));
    let b = Uuid::from_u128(SENSOR_GUID).max(Uuid::from_u128(BODY_GUID));
    assert_eq!(
        list,
        vec![OverlapEvent {
            a,
            b,
            phase: OverlapPhase::Begin
        }],
        "the sensor overlap list must be the one canonical a<b Begin pair"
    );
    // The cube falls from y=3 through the sensor; the begin step is deterministic
    // (asserted identically in the runtime twin — the parity gate).
    assert!(begin_step > 0, "overlap should begin after some fall");
    assert_eq!(begin_step, EXPECTED_BEGIN_STEP, "begin step must be stable");
}

/// The fixed-step index (1-based) at which the sensor overlap begins. Pinned so
/// the runtime twin asserts the same value → editor/runtime parity.
const EXPECTED_BEGIN_STEP: u64 = 40;

/// Step until `drained_overlaps` first contains a `Begin`, returning
/// `(step_index, the overlap list at that step)`.
fn first_overlap_begin(session: &mut SimSession, doc: &mut SceneDoc) -> (u64, Vec<OverlapEvent>) {
    for _ in 0..240 {
        session.step_once(doc, SimInput::default());
        let begins: Vec<OverlapEvent> = session
            .drained_overlaps()
            .iter()
            .copied()
            .filter(|o| o.phase == OverlapPhase::Begin)
            .collect();
        if !begins.is_empty() {
            return (session.steps(), begins);
        }
    }
    panic!("no sensor overlap Begin observed in 240 steps");
}

// ── deliverable: input events fire deterministically ───────────────────────────

/// An actor whose `Input("jump")` press handler increments `count` (release is a
/// no-op — it branches on `pressed`).
fn jump_counter_class() -> BlueprintClass {
    let get = |name: &str| Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str(name.into()))],
    };
    let set = |name: &str, v: Expr| {
        Stmt::ExprStmt(Expr::Call {
            path: vec!["vars".into(), "set".into()],
            args: vec![Expr::Lit(Lit::Str(name.into())), v],
        })
    };
    let body = vec![Stmt::If {
        cond: Expr::Param("pressed".into()),
        then_body: vec![set(
            "count",
            Expr::Binary(
                inf_blueprint::BinOp::Add,
                Box::new(get("count")),
                Box::new(Expr::Lit(Lit::Int(1))),
            ),
        )],
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
fn input_event_press_fires_once_per_edge() {
    let actor = Uuid::from_u128(0x00A0_0001);
    let mut doc = SceneDoc::new();
    doc.create_with_guid(actor, SpawnKind::Empty, "Jumper", None);
    doc.world_mut().propagate();

    let mut session = SimSession::enter(
        &mut doc,
        vec![(actor, jump_counter_class())],
        DVec2::ZERO,
        SIM_HZ,
    );

    let count = |s: &SimSession| match s.actor_var(actor, "count") {
        Some(inf_blueprint::Value::Int(n)) => *n,
        _ => -1,
    };

    // Press edge → fires once.
    session.step_once(&mut doc, SimInput::with_down(["jump"]));
    assert_eq!(
        count(&session),
        1,
        "press edge fires the Input handler once"
    );
    // Held (no edge) → no additional fire.
    session.step_once(&mut doc, SimInput::with_down(["jump"]));
    assert_eq!(count(&session), 1, "held key does not re-fire");
    // Release edge → pressed=false branch is a no-op → still 1.
    session.step_once(&mut doc, SimInput::default());
    assert_eq!(count(&session), 1, "release edge does not increment");
    // Second press edge → fires again.
    session.step_once(&mut doc, SimInput::with_down(["jump"]));
    assert_eq!(count(&session), 2, "a fresh press edge fires again");
}

// ── deliverable: event dispatchers (bind / dispatch / listener / cap) ──────────

/// A dispatch integer literal call helper.
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
                inf_blueprint::BinOp::Add,
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

/// Emitter actor (entity id 1): on `Tick` dispatches `event::dispatch(1, "ping")`;
/// its own `Custom("ping")` handler increments `self_ping`.
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

/// Listener actor (entity id 2): on `BeginPlay` binds `(source=1, "ping") →
/// handler "react"`; `Custom("react")` increments `reacted`.
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
fn dispatch_fires_target_and_bound_listener() {
    // Guids ordered so the emitter is entity 1 and the listener entity 2.
    let emitter = Uuid::from_u128(0x00D0_0001);
    let listener = Uuid::from_u128(0x00D0_0002);
    let mut doc = SceneDoc::new();
    doc.create_with_guid(emitter, SpawnKind::Empty, "Emitter", None);
    doc.create_with_guid(listener, SpawnKind::Empty, "Listener", None);
    doc.world_mut().propagate();

    let mut session = SimSession::enter(
        &mut doc,
        vec![(emitter, emitter_class()), (listener, listener_class())],
        DVec2::ZERO,
        SIM_HZ,
    );
    let getv = |s: &SimSession, g: Uuid, n: &str| match s.actor_var(g, n) {
        Some(inf_blueprint::Value::Int(v)) => *v,
        _ => -1,
    };

    // One tick: emitter dispatches → its own ping handler + the bound listener fire.
    session.step_once(&mut doc, SimInput::default());
    assert_eq!(
        getv(&session, emitter, "self_ping"),
        1,
        "target's own Custom fires"
    );
    assert_eq!(
        getv(&session, listener, "reacted"),
        1,
        "bound listener's handler fires"
    );

    // Second tick: both increment again (binding persists).
    session.step_once(&mut doc, SimInput::default());
    assert_eq!(getv(&session, emitter, "self_ping"), 2);
    assert_eq!(getv(&session, listener, "reacted"), 2);
}

/// A self-perpetuating actor (entity id 1): `BeginPlay` dispatches `(1,"loop")`;
/// its `Custom("loop")` handler increments `n` **and** re-dispatches `(1,"loop")`,
/// so a single seed would spin forever without the round cap.
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
fn dispatch_cycle_is_capped_at_64() {
    let looper = Uuid::from_u128(0x00E0_0001);
    let mut doc = SceneDoc::new();
    doc.create_with_guid(looper, SpawnKind::Empty, "Looper", None);
    doc.world_mut().propagate();

    // BeginPlay seeds the cycle; the drain caps it at 64 dispatches.
    let session = SimSession::enter(
        &mut doc,
        vec![(looper, looping_class())],
        DVec2::ZERO,
        SIM_HZ,
    );
    let n = match session.actor_var(looper, "n") {
        Some(inf_blueprint::Value::Int(v)) => *v,
        _ => -1,
    };
    assert_eq!(n, 64, "the dispatch round cap fires exactly 64 times");
    assert!(
        session.logs().iter().any(|l| l.contains("dispatch cap")),
        "the cap logs a deterministic drop line"
    );
}
