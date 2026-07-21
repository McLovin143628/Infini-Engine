//! End-to-end Simulate proofs for the **P11.3 3D character tools**: the in-sim 3D
//! character controller (blueprint `physics3d.move_and_slide` over rapier3d) and
//! **root motion** (a locomotion clip driving the entity), both through the real
//! `SimSession` fixed-step — no GPU, no Tauri, pure CI.
//!
//! Scenes are built directly with the same `create_with_guid` + component-insert
//! pattern the platformer sample uses.

use glam::{DVec3, Mat4};
use uuid::Uuid;

use inf_anim::root_motion::straight_line_clip;
use inf_anim::{AnimClip, Joint, JointTransform, Skeleton};
use inf_blueprint::{
    Binding, BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, LocalId, Param, Stmt,
    Ty,
};
use inf_ecs::components::{
    ActorClass, AnimPlayer, BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, RootMotion,
    Transform,
};
use inf_ecs::math::Vec3d;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};

// ── helpers ──────────────────────────────────────────────────────────────────

macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

fn pos_x(doc: &SceneDoc, guid: Uuid) -> f64 {
    let e = doc.entity_of(guid).expect("entity");
    doc.world()
        .world()
        .get::<Transform>(e)
        .expect("transform")
        .translation
        .x
}

/// A one-joint skeleton (just the root) — enough for root-motion extraction.
fn root_skeleton() -> Skeleton {
    Skeleton::new(vec![Joint {
        name: "root".into(),
        parent: None,
        inverse_bind: Mat4::IDENTITY.to_cols_array(),
        local_bind: JointTransform::IDENTITY,
    }])
    .unwrap()
}

/// A static floor box centred at (0,-0.5,0), top at y=0.
fn add_floor(doc: &mut SceneDoc, guid: Uuid) {
    doc.create_with_guid(guid, SpawnKind::Empty, "Floor", None);
    insert!(
        doc,
        guid,
        Transform::from_translation(DVec3::new(0.0, -0.5, 0.0))
    );
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
            half_extents: Vec3d::new(5.0, 0.5, 5.0),
            ..Default::default()
        }
    );
}

/// A static wall box centred at `x` (thin in X, tall in Y).
fn add_wall(doc: &mut SceneDoc, guid: Uuid, x: f64) {
    doc.create_with_guid(guid, SpawnKind::Empty, "Wall", None);
    insert!(
        doc,
        guid,
        Transform::from_translation(DVec3::new(x, 1.0, 0.0))
    );
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
            half_extents: Vec3d::new(0.2, 2.0, 2.0),
            ..Default::default()
        }
    );
}

/// A capsule character resting on the floor at `(x, 0.8, 0)` (bottom at y≈0).
fn add_character(doc: &mut SceneDoc, guid: Uuid, x: f64) {
    doc.create_with_guid(guid, SpawnKind::Empty, "Character", None);
    insert!(
        doc,
        guid,
        Transform::from_translation(DVec3::new(x, 0.8, 0.0))
    );
    insert!(
        doc,
        guid,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        guid,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(0.3, 0.5, 0.3),
            radius: 0.3,
            ..Default::default()
        }
    );
    insert!(
        doc,
        guid,
        inf_ecs::components::CharacterController3D::default()
    );
}

/// A locomotion `AnimPlayer` playing `clip_guid` (a 1 s, non-looping clip).
fn add_root_motion_player(doc: &mut SceneDoc, guid: Uuid, clip_guid: Uuid) {
    insert!(
        doc,
        guid,
        AnimPlayer {
            clip: Some(clip_guid),
            t: 0.0,
            speed: 1.0,
            looping: false,
            playing: true,
            duration: 1.0,
        }
    );
    insert!(doc, guid, RootMotion::apply());
}

// ── deliverable 3: 3D character controller (blueprint + ECS end-to-end) ────────

/// A blueprint class whose Tick moves the entity +X by 0.05 m each step while
/// "right" is held, via `physics3d.move_and_slide`.
fn walker_class() -> BlueprintClass {
    let call = |path: &[&str], args: Vec<Expr>| Expr::Call {
        path: path.iter().map(|s| s.to_string()).collect(),
        args,
    };
    let body = vec![
        Stmt::Let {
            id: LocalId(1),
            binding: Binding::Named("e".into()),
            ty: None,
            mutable: false,
            value: call(&["vars", "get"], vec![Expr::Lit(Lit::Str("entity".into()))]),
        },
        Stmt::If {
            cond: call(
                &["input", "is_down"],
                vec![Expr::Lit(Lit::Str("right".into()))],
            ),
            then_body: vec![Stmt::ExprStmt(call(
                &["physics3d", "move_and_slide"],
                vec![
                    Expr::Local(LocalId(1)),
                    Expr::Lit(Lit::Float(0.05)),
                    Expr::Lit(Lit::Float(0.0)),
                    Expr::Lit(Lit::Float(0.0)),
                ],
            ))],
            else_body: vec![],
        },
    ];
    let mut class = BlueprintClass::new("act:walker3d", "Walker 3D");
    class.events = vec![EventBinding {
        event: EventKind::Tick,
        body: BlueprintFn {
            id: "tick".into(),
            name: "tick".into(),
            params: vec![Param {
                name: "dt".into(),
                ty: Ty::Float,
            }],
            ret: Ty::Unit,
            body,
        },
    }];
    class
}

#[test]
fn character3d_walks_right_and_a_wall_stops_it() {
    let char_guid = Uuid::from_u128(0xC301_0001);
    let floor_guid = Uuid::from_u128(0xC301_0002);
    let wall_guid = Uuid::from_u128(0xC301_0003);

    let mut doc = SceneDoc::new();
    add_floor(&mut doc, floor_guid);
    add_character(&mut doc, char_guid, 0.0);
    add_wall(&mut doc, wall_guid, 2.0); // left face at x=1.8
    insert!(doc, char_guid, ActorClass(Uuid::from_u128(0xC301_00AC)));
    doc.world_mut().propagate();

    let mut session = SimSession::enter(
        &mut doc,
        vec![(char_guid, walker_class())],
        glam::DVec2::ZERO,
        SIM_HZ,
    );

    // Hold right: x rises off 0, then the wall halts it short of x≈1.5.
    for _ in 0..200 {
        session.step_once(&mut doc, SimInput::with_down(["right"]));
    }
    let x = pos_x(&doc, char_guid);
    assert!(x > 0.5, "character should have walked right, x={x}");
    assert!(
        x < 1.6,
        "the wall (left face x=1.8, minus the capsule radius) should stop it, x={x}"
    );
    assert!(
        session.is_grounded(char_guid),
        "character standing on the floor should be grounded"
    );
}

// ── deliverable 2: root motion ────────────────────────────────────────────────

#[test]
fn root_motion_moves_entity_one_metre_without_a_mover() {
    let ent = Uuid::from_u128(0xC302_0001);
    let clip_guid = Uuid::from_u128(0xC302_00A0);

    let mut doc = SceneDoc::new();
    doc.create_with_guid(ent, SpawnKind::Empty, "Mover", None);
    insert!(doc, ent, Transform::from_translation(DVec3::ZERO));
    add_root_motion_player(&mut doc, ent, clip_guid);
    doc.world_mut().propagate();

    let mut session = SimSession::enter(&mut doc, vec![], glam::DVec2::ZERO, SIM_HZ);
    // +1 m/s along +X over a 1 s clip.
    session.register_root_motion_clip(clip_guid, root_skeleton(), locomotion_clip());

    // 60 fixed steps ≈ 1 s → ~1 m of root motion.
    for _ in 0..60 {
        session.step_once(&mut doc, SimInput::default());
    }
    let x = pos_x(&doc, ent);
    assert!(
        (x - 1.0).abs() < 0.05,
        "expected ~1 m of root motion, x={x}"
    );
}

#[test]
fn root_motion_drives_a_character_through_the_mover() {
    let ent = Uuid::from_u128(0xC303_0001);
    let floor = Uuid::from_u128(0xC303_0002);
    let clip_guid = Uuid::from_u128(0xC303_00A0);

    let mut doc = SceneDoc::new();
    add_floor(&mut doc, floor);
    add_character(&mut doc, ent, 0.0);
    add_root_motion_player(&mut doc, ent, clip_guid);
    doc.world_mut().propagate();

    let mut session = SimSession::enter(&mut doc, vec![], glam::DVec2::ZERO, SIM_HZ);
    session.register_root_motion_clip(clip_guid, root_skeleton(), locomotion_clip());

    for _ in 0..60 {
        session.step_once(&mut doc, SimInput::default());
    }
    let x = pos_x(&doc, ent);
    // The open floor doesn't block horizontal motion → still ~1 m.
    assert!(
        (x - 1.0).abs() < 0.1,
        "root motion through the mover should still travel ~1 m, x={x}"
    );
}

#[test]
fn root_motion_is_blocked_by_a_wall() {
    let ent = Uuid::from_u128(0xC304_0001);
    let floor = Uuid::from_u128(0xC304_0002);
    let wall = Uuid::from_u128(0xC304_0003);
    let clip_guid = Uuid::from_u128(0xC304_00A0);

    let mut doc = SceneDoc::new();
    add_floor(&mut doc, floor);
    add_character(&mut doc, ent, 0.0);
    add_wall(&mut doc, wall, 0.8); // left face at x=0.6
    add_root_motion_player(&mut doc, ent, clip_guid);
    doc.world_mut().propagate();

    let mut session = SimSession::enter(&mut doc, vec![], glam::DVec2::ZERO, SIM_HZ);
    session.register_root_motion_clip(clip_guid, root_skeleton(), locomotion_clip());

    for _ in 0..60 {
        session.step_once(&mut doc, SimInput::default());
    }
    let x = pos_x(&doc, ent);
    // The wall (left face x=0.6, minus the capsule radius 0.3) halts the character
    // well short of the full 1 m the clip's root motion would otherwise travel.
    assert!(
        (0.0..0.9).contains(&x),
        "the wall should block root motion short of 1 m, x={x}"
    );
}

/// The shared locomotion clip: root translates +1 m/s along +X over 1 s.
fn locomotion_clip() -> AnimClip {
    straight_line_clip("walk", glam::Vec3::X, 1.0, 1.0)
}
