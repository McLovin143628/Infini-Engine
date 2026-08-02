//! P20.2 **preview == shipped** for water physics, through the real editor
//! `SimSession` fixed step.
//!
//! The paired runtime test — `runtime/inf-player/tests/water_physics.rs`, the
//! `the_editor_and_the_runtime_agree_on_the_water` case — builds the *identical*
//! scene over `RuntimeSim` and asserts the **same pinned numbers**. That pairing
//! is the gate: the two hosts each have their own copy of the fixed-step
//! ordering, their own `Host::call` match arms and their own `move_and_slide`
//! path, so "they run the same physics" is a claim about two files and has to be
//! checked from the outside.
//!
//! What is pinned is deliberately **integer**: the step index a `WaterEnter` lands
//! on, and the handler counts. A settled `f64` position would pin the solver's
//! last bit on this machine and start failing on someone else's for a reason that
//! is not a bug; a step index is the same on any IEEE-conformant build and still
//! moves the moment the ordering, the thresholds or the force model do.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_blueprint::{
    BinOp, BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Stmt, Ty, Value,
    Variable,
};
use inf_ecs::components::{
    BodyKind3D, Buoyancy, Collider3D, ColliderShape3DKind, RigidBody3D, Transform, WaterBody,
    WaterKind,
};
use inf_ecs::math::Vec3d;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};

/// Shared with the runtime twin — the same guids build the same world.
pub const OCEAN_GUID: u128 = 0x2002_2001;
pub const CRATE_GUID: u128 = 0x2002_2002;

/// **PINNED IDENTICALLY IN THE RUNTIME TWIN.** The fixed step on which a 1 m
/// crate of density 500, released 5 m above a still sea, first reports
/// `WaterEnter`. Free fall to the waterline is `sqrt(2·4.5/9.81) ≈ 0.958 s`.
pub const EXPECTED_ENTER_STEP: usize = 58;
/// **PINNED IDENTICALLY IN THE RUNTIME TWIN.** Entries / exits / splashes over
/// 600 steps: it lands hard enough to splash, bounces clear once, and settles.
pub const EXPECTED_COUNTS: (i64, i64, i64) = (2, 1, 3);

macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

/// A Blueprint that counts what the water tells it. Character-for-character the
/// runtime twin's, because that is the point.
pub fn water_listener_class() -> BlueprintClass {
    let get = |name: &str| Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str(name.into()))],
    };
    let bump = |name: &str| {
        Stmt::ExprStmt(Expr::Call {
            path: vec!["vars".into(), "set".into()],
            args: vec![
                Expr::Lit(Lit::Str(name.into())),
                Expr::Binary(
                    BinOp::Add,
                    Box::new(get(name)),
                    Box::new(Expr::Lit(Lit::Int(1))),
                ),
            ],
        })
    };
    let counter = |name: &str| Variable {
        name: name.into(),
        ty: Ty::Int,
        default: Lit::Int(0),
        exposed: false,
    };
    let handler = |kind: EventKind, body: Vec<Stmt>| EventBinding {
        event: kind.clone(),
        body: BlueprintFn {
            id: kind.key(),
            name: kind.key(),
            params: kind.signature(),
            ret: Ty::Unit,
            body,
        },
    };
    let mut class = BlueprintClass::new("act:water-listener", "Water Listener");
    class.variables = vec![counter("enters"), counter("exits"), counter("splashes")];
    class.events = vec![
        handler(EventKind::WaterEnter, vec![bump("enters")]),
        handler(EventKind::WaterExit, vec![bump("exits")]),
        handler(EventKind::WaterSplash, vec![bump("splashes")]),
    ];
    class
}

/// The shared scene: a still ocean at y = 0 and a buoyant crate 5 m above it.
fn water_doc() -> SceneDoc {
    let mut doc = SceneDoc::new();
    let ocean = Uuid::from_u128(OCEAN_GUID);
    let krate = Uuid::from_u128(CRATE_GUID);

    doc.create_with_guid(ocean, SpawnKind::Empty, "Ocean", None);
    insert!(doc, ocean, Transform::IDENTITY);
    insert!(
        doc,
        ocean,
        WaterBody {
            kind: WaterKind::Ocean,
            level_m: 0.0,
            // Still: the height query is then exact, so the pinned step index is
            // about the physics rather than about which wave happened to be under
            // the crate.
            wave_amplitude_m: 0.0,
            ..WaterBody::default()
        }
    );

    doc.create_with_guid(krate, SpawnKind::Empty, "Crate", None);
    insert!(
        doc,
        krate,
        Transform::from_translation(DVec3::new(0.0, 5.0, 0.0))
    );
    insert!(
        doc,
        krate,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        krate,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::splat(0.5),
            density: 500.0,
            ..Default::default()
        }
    );
    insert!(doc, krate, Buoyancy::of_density(500.0));
    doc.world_mut().propagate();
    doc
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

#[test]
fn the_editor_floats_a_crate_and_hears_the_water() {
    let krate = Uuid::from_u128(CRATE_GUID);
    let mut doc = water_doc();
    let mut session = SimSession::enter(
        &mut doc,
        vec![(krate, water_listener_class())],
        DVec2::new(0.0, -9.81),
        SIM_HZ,
    );

    let var = |s: &SimSession, name: &str| match s.actor_var(krate, name) {
        Some(Value::Int(n)) => *n,
        other => panic!("{name} is {other:?}"),
    };

    let mut enter_step = None;
    for i in 0..600 {
        session.step_once(&mut doc, SimInput::default());
        if enter_step.is_none() && var(&session, "enters") > 0 {
            enter_step = Some(i);
        }
    }

    assert_eq!(
        enter_step,
        Some(EXPECTED_ENTER_STEP),
        "the crate met the water on a different step than the runtime twin"
    );
    assert_eq!(
        (
            var(&session, "enters"),
            var(&session, "exits"),
            var(&session, "splashes")
        ),
        EXPECTED_COUNTS,
        "the editor heard a different set of water events than the runtime twin"
    );

    // …and it really floats: a density-500 crate settles half submerged, i.e.
    // with its centre on the waterline. Given another 20 s to stop ringing — a
    // 5 m drop into water this lightly damped is a long, slow bob, and the point
    // here is WHERE it ends up, not how fast it gets there.
    for _ in 0..1_200 {
        session.step_once(&mut doc, SimInput::default());
    }
    let y = body_y(&doc, krate);
    assert!(
        (y - 0.0).abs() < 5e-3,
        "the crate settled at y = {y}, not on the 0 m waterline"
    );
}

/// The `water.*` Blueprint queries answer from the editor's own fixed-step water
/// index — the same seam the shipped host reads.
#[test]
fn the_editor_answers_the_water_queries() {
    let krate = Uuid::from_u128(CRATE_GUID);
    let mut doc = water_doc();

    // A Blueprint that stashes the three queries into member variables each tick.
    let entity = || Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str("entity".into()))],
    };
    let set = |name: &str, value: Expr| {
        Stmt::ExprStmt(Expr::Call {
            path: vec!["vars".into(), "set".into()],
            args: vec![Expr::Lit(Lit::Str(name.into())), value],
        })
    };
    let mut class = BlueprintClass::new("act:probe", "Probe");
    class.variables = vec![
        Variable {
            name: "wet".into(),
            ty: Ty::Bool,
            default: Lit::Bool(false),
            exposed: false,
        },
        Variable {
            name: "fraction".into(),
            ty: Ty::Float,
            default: Lit::Float(-1.0),
            exposed: false,
        },
        Variable {
            name: "height".into(),
            ty: Ty::Float,
            default: Lit::Float(-999.0),
            exposed: false,
        },
    ];
    class.events = vec![EventBinding {
        event: EventKind::Tick,
        body: BlueprintFn {
            id: "tick".into(),
            name: "tick".into(),
            params: EventKind::Tick.signature(),
            ret: Ty::Unit,
            body: vec![
                set(
                    "wet",
                    Expr::Call {
                        path: vec!["water".into(), "is_in_water".into()],
                        args: vec![entity()],
                    },
                ),
                set(
                    "fraction",
                    Expr::Call {
                        path: vec!["water".into(), "submerged_fraction".into()],
                        args: vec![entity()],
                    },
                ),
                set(
                    "height",
                    Expr::Call {
                        path: vec!["water".into(), "surface_height".into()],
                        args: vec![Expr::Lit(Lit::Float(0.0)), Expr::Lit(Lit::Float(0.0))],
                    },
                ),
            ],
        },
    }];

    let mut session = SimSession::enter(
        &mut doc,
        vec![(krate, class)],
        DVec2::new(0.0, -9.81),
        SIM_HZ,
    );
    // One step: the crate is still 5 m up, so it is dry and the fraction is zero…
    session.step_once(&mut doc, SimInput::default());
    assert_eq!(session.actor_var(krate, "wet"), Some(&Value::Bool(false)));
    assert_eq!(
        session.actor_var(krate, "fraction"),
        Some(&Value::Float(0.0))
    );
    // …but the ocean is still under it, at its authored level.
    assert_eq!(
        session.actor_var(krate, "height"),
        Some(&Value::Float(0.0)),
        "an ocean is over every point, so the surface height must answer"
    );

    for _ in 0..1_800 {
        session.step_once(&mut doc, SimInput::default());
    }
    assert_eq!(
        session.actor_var(krate, "wet"),
        Some(&Value::Bool(true)),
        "a floating crate is in the water"
    );
    match session.actor_var(krate, "fraction") {
        Some(Value::Float(f)) => assert!(
            (f - 0.5).abs() < 5e-3,
            "a density-500 crate settles half submerged, not at {f}"
        ),
        other => panic!("fraction is {other:?}"),
    }
}
