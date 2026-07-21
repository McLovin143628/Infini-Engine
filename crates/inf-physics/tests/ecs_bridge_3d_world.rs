//! End-to-end tests for the 3D ECS↔physics bridge over a real [`EcsWorld`]:
//! scene components → simulation → transforms, plus determinism over the ECS
//! layer. The `d3` mirror of `tests/ecs_bridge.rs` (which drives `PhysicsBridge2D`
//! from `RigidBody2D`/`Collider2D`), using `PhysicsBridge3D::sync_from_world` +
//! `write_back_into`.

use glam::DVec3;
use inf_ecs::components::{BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Transform};
use inf_ecs::{EcsWorld, Vec3d};
use inf_physics::PhysicsBridge3D;
use uuid::Uuid;

const DT: f64 = 1.0 / 60.0;

/// Insert the P9.1 physics components onto a spawned entity at `pos`.
fn add_body(
    w: &mut EcsWorld,
    entity: inf_ecs::Entity,
    rb: RigidBody3D,
    col: Collider3D,
    pos: DVec3,
) {
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(pos.x, pos.y, pos.z);
    w.world_mut().entity_mut(entity).insert((rb, col, t));
    w.mark_dirty();
}

fn floor_collider() -> Collider3D {
    Collider3D {
        shape_kind: ColliderShape3DKind::Box,
        half_extents: Vec3d::new(50.0, 0.5, 50.0),
        ..Default::default()
    }
}

fn ball_collider() -> Collider3D {
    Collider3D {
        shape_kind: ColliderShape3DKind::Sphere,
        radius: 0.5,
        friction: 0.8,
        ..Default::default()
    }
}

/// A world with a static floor at y=0 and a dynamic ball at y=10, plus a bridge.
fn build_drop_scene() -> (EcsWorld, PhysicsBridge3D, Uuid) {
    let mut w = EcsWorld::new();

    let floor = w.spawn_with_guid(Uuid::from_u128(1), "Floor", None);
    add_body(
        &mut w,
        floor,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        floor_collider(),
        DVec3::ZERO,
    );

    let ball_guid = Uuid::from_u128(2);
    let ball = w.spawn_with_guid(ball_guid, "Ball", None);
    add_body(
        &mut w,
        ball,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        },
        ball_collider(),
        DVec3::new(0.0, 10.0, 0.0),
    );
    w.propagate();

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&w);
    (w, bridge, ball_guid)
}

#[test]
fn dynamic_box_falls_and_rests_on_static_floor_via_components() {
    let (mut w, mut bridge, ball) = build_drop_scene();

    for _ in 0..180 {
        bridge.step(DT);
    }
    bridge.write_back_into(&mut w);
    w.propagate();

    let entity = w.entity_of(ball).unwrap();
    let t = w.world().get::<Transform>(entity).unwrap();
    // Floor top at y=0.5, ball radius 0.5 → rest centre near y=1.0.
    assert!(
        (t.translation.y - 1.0).abs() < 0.05,
        "ball should rest near y=1.0, got y={}",
        t.translation.y
    );
    // The ball fell straight down: X/Z barely moved.
    assert!(t.translation.x.abs() < 0.05 && t.translation.z.abs() < 0.05);
}

#[test]
fn kinematic_body_follows_externally_set_transform() {
    let mut w = EcsWorld::new();
    let guid = Uuid::from_u128(7);
    let e = w.spawn_with_guid(guid, "Platform", None);
    add_body(
        &mut w,
        e,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
        Collider3D::default(),
        DVec3::ZERO,
    );
    w.propagate();

    let mut bridge = PhysicsBridge3D::new(DVec3::ZERO);
    bridge.sync_from_world(&w);
    let body = bridge.body_of(guid).unwrap();

    // Externally move the entity's Transform, re-sync: the kinematic body tracks it.
    if let Some(mut t) = w.world_mut().get_mut::<Transform>(e) {
        t.translation = Vec3d::new(3.0, 4.0, 5.0);
    }
    w.propagate();
    bridge.sync_from_world(&w);
    bridge.step(DT);

    let p = bridge.world().body_translation(body).unwrap();
    assert!((p - DVec3::new(3.0, 4.0, 5.0)).length() < 1e-9, "got {p:?}");
}

#[test]
fn removing_the_component_despawns_the_body() {
    let (mut w, mut bridge, ball) = build_drop_scene();
    assert!(bridge.body_of(ball).is_some());

    // Despawn the ball entity, re-sync: its body is gone.
    let entity = w.entity_of(ball).unwrap();
    w.despawn(entity);
    bridge.sync_from_world(&w);
    assert!(bridge.body_of(ball).is_none(), "body should be despawned");

    // The floor body survives (1 body left).
    assert_eq!(bridge.world().body_ids().len(), 1);
}

#[test]
fn two_identical_scenes_step_byte_identical() {
    // Build two independent scenes/bridges from the same construction and step
    // both 300 times; the dynamic body's transform must be byte-identical.
    fn run() -> [u8; 24] {
        let (mut w, mut bridge, ball) = build_drop_scene();
        for _ in 0..300 {
            bridge.step(DT);
        }
        bridge.write_back_into(&mut w);
        w.propagate();
        let entity = w.entity_of(ball).unwrap();
        let t = w.world().get::<Transform>(entity).unwrap();
        let mut bytes = [0u8; 24];
        bytes[0..8].copy_from_slice(&t.translation.x.to_le_bytes());
        bytes[8..16].copy_from_slice(&t.translation.y.to_le_bytes());
        bytes[16..24].copy_from_slice(&t.translation.z.to_le_bytes());
        bytes
    }
    assert_eq!(
        run(),
        run(),
        "two identical ECS scenes diverged after 300 steps"
    );
}
