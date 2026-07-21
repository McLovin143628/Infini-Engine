//! P11.3 runtime parity: the shipped player's `runtime_sim` runs the same 3D
//! character tools (root motion + the 3D mover) as the editor Simulate — so
//! preview == shipped for character movement too. Pure Ring-0, headless CI.

use glam::{DVec2, DVec3, Mat4};
use uuid::Uuid;

use inf_anim::root_motion::straight_line_clip;
use inf_anim::{Joint, JointTransform, Skeleton};
use inf_ecs::components::{
    AnimPlayer, BodyKind3D, CharacterController3D, Collider3D, ColliderShape3DKind, RigidBody3D,
    RootMotion, Transform,
};
use inf_ecs::math::Vec3d;
use inf_ecs::EcsWorld;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

fn root_skeleton() -> Skeleton {
    Skeleton::new(vec![Joint {
        name: "root".into(),
        parent: None,
        inverse_bind: Mat4::IDENTITY.to_cols_array(),
        local_bind: JointTransform::IDENTITY,
    }])
    .unwrap()
}

fn pos_x(sim: &RuntimeSim, guid: Uuid) -> f64 {
    let e = sim.world().entity_of(guid).expect("entity");
    sim.world().world_translation(e).expect("world transform").x
}

#[test]
fn runtime_root_motion_moves_entity_one_metre() {
    let ent = Uuid::from_u128(0xD301_0001);
    let clip_guid = Uuid::from_u128(0xD301_00A0);

    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(ent, "Mover", None);
    world
        .world_mut()
        .entity_mut(e)
        .insert(Transform::from_translation(DVec3::ZERO));
    world.world_mut().entity_mut(e).insert(AnimPlayer {
        clip: Some(clip_guid),
        t: 0.0,
        speed: 1.0,
        looping: false,
        playing: true,
        duration: 1.0,
    });
    world.world_mut().entity_mut(e).insert(RootMotion::apply());
    world.mark_dirty();

    let mut sim = RuntimeSim::new(world, vec![], DVec2::ZERO, 60.0);
    sim.register_root_motion_clip(
        clip_guid,
        root_skeleton(),
        straight_line_clip("walk", glam::Vec3::X, 1.0, 1.0),
    );

    for _ in 0..60 {
        sim.step_once(RuntimeInput::default());
    }
    let x = pos_x(&sim, ent);
    assert!(
        (x - 1.0).abs() < 0.05,
        "expected ~1 m of root motion, x={x}"
    );
}

#[test]
fn runtime_root_motion_is_blocked_by_a_wall_through_the_mover() {
    let ent = Uuid::from_u128(0xD302_0001);
    let floor = Uuid::from_u128(0xD302_0002);
    let wall = Uuid::from_u128(0xD302_0003);
    let clip_guid = Uuid::from_u128(0xD302_00A0);

    let mut world = EcsWorld::new();

    // Floor.
    let f = world.spawn_with_guid(floor, "Floor", None);
    world
        .world_mut()
        .entity_mut(f)
        .insert(Transform::from_translation(DVec3::new(0.0, -0.5, 0.0)));
    world.world_mut().entity_mut(f).insert(RigidBody3D {
        kind: BodyKind3D::Static,
        ..Default::default()
    });
    world.world_mut().entity_mut(f).insert(Collider3D {
        shape_kind: ColliderShape3DKind::Box,
        half_extents: Vec3d::new(5.0, 0.5, 5.0),
        ..Default::default()
    });

    // Wall (left face x = 0.6).
    let wl = world.spawn_with_guid(wall, "Wall", None);
    world
        .world_mut()
        .entity_mut(wl)
        .insert(Transform::from_translation(DVec3::new(0.8, 1.0, 0.0)));
    world.world_mut().entity_mut(wl).insert(RigidBody3D {
        kind: BodyKind3D::Static,
        ..Default::default()
    });
    world.world_mut().entity_mut(wl).insert(Collider3D {
        shape_kind: ColliderShape3DKind::Box,
        half_extents: Vec3d::new(0.2, 2.0, 2.0),
        ..Default::default()
    });

    // Character.
    let c = world.spawn_with_guid(ent, "Character", None);
    world
        .world_mut()
        .entity_mut(c)
        .insert(Transform::from_translation(DVec3::new(0.0, 0.8, 0.0)));
    world.world_mut().entity_mut(c).insert(RigidBody3D {
        kind: BodyKind3D::Kinematic,
        ..Default::default()
    });
    world.world_mut().entity_mut(c).insert(Collider3D {
        shape_kind: ColliderShape3DKind::Capsule,
        half_extents: Vec3d::new(0.3, 0.5, 0.3),
        radius: 0.3,
        ..Default::default()
    });
    world
        .world_mut()
        .entity_mut(c)
        .insert(CharacterController3D::default());
    world.world_mut().entity_mut(c).insert(AnimPlayer {
        clip: Some(clip_guid),
        t: 0.0,
        speed: 1.0,
        looping: false,
        playing: true,
        duration: 1.0,
    });
    world.world_mut().entity_mut(c).insert(RootMotion::apply());
    world.mark_dirty();

    let mut sim = RuntimeSim::new(world, vec![], DVec2::ZERO, 60.0);
    sim.register_root_motion_clip(
        clip_guid,
        root_skeleton(),
        straight_line_clip("walk", glam::Vec3::X, 1.0, 1.0),
    );

    for _ in 0..60 {
        sim.step_once(RuntimeInput::default());
    }
    let x = pos_x(&sim, ent);
    assert!(
        (0.0..0.9).contains(&x),
        "the wall should block root motion short of 1 m, x={x}"
    );
}
