//! End-to-end tests for the 3D ECS↔physics bridge ([`PhysicsBridge3D`]): scene
//! snapshot → simulation → poses, plus determinism over the bridge layer. The
//! `d3` mirror of `tests/ecs_bridge.rs` — over the facade-local [`EntitySync3D`]
//! snapshot (the real `inf-ecs` 3D components land next batch; see the report).

use glam::{DQuat, DVec3};
use inf_physics::d3::{BodyDesc3D, ColliderDesc3D, EntitySync3D};
use inf_physics::{BodyKind3D, ColliderShape3D, PhysicsBridge3D};
use uuid::Uuid;

const DT: f64 = 1.0 / 60.0;

fn floor_collider() -> ColliderDesc3D {
    ColliderDesc3D::new(ColliderShape3D::Box {
        half_extents: DVec3::new(50.0, 0.5, 50.0),
    })
}

fn ball_collider() -> ColliderDesc3D {
    ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.5 }).friction(0.8)
}

fn entity(guid: u128, kind: BodyKind3D, collider: ColliderDesc3D, pos: DVec3) -> EntitySync3D {
    EntitySync3D {
        guid: Uuid::from_u128(guid),
        body: Some(BodyDesc3D {
            kind,
            ..Default::default()
        }),
        collider: Some(collider),
        translation: pos,
        rotation: DQuat::IDENTITY,
        joint: None,
    }
}

/// A snapshot with a static floor at y=0 and a dynamic ball at y=10.
fn drop_scene() -> Vec<EntitySync3D> {
    vec![
        entity(1, BodyKind3D::Static, floor_collider(), DVec3::ZERO),
        entity(
            2,
            BodyKind3D::Dynamic,
            ball_collider(),
            DVec3::new(0.0, 10.0, 0.0),
        ),
    ]
}

#[test]
fn dynamic_body_falls_and_rests_on_static_floor_via_snapshot() {
    let scene = drop_scene();
    let ball = Uuid::from_u128(2);
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync(&scene);

    for _ in 0..180 {
        bridge.step(DT);
    }

    let writes = bridge.write_back();
    // Only the dynamic ball is written back (the static floor never moved).
    assert_eq!(writes.len(), 1);
    let w = writes[0];
    assert_eq!(w.guid, ball);
    // Floor top at y=0.5, ball radius 0.5 → rest centre near y=1.0.
    assert!(
        (w.translation.y - 1.0).abs() < 0.05,
        "ball should rest near y=1.0, got y={}",
        w.translation.y
    );
    // Z preserved / untouched dimension stays put.
    assert!(w.translation.z.abs() < 0.05);
}

#[test]
fn kinematic_body_follows_externally_set_transform() {
    let guid = Uuid::from_u128(7);
    let mut snap = EntitySync3D {
        guid,
        body: Some(BodyDesc3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        }),
        collider: Some(ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::splat(0.5),
        })),
        translation: DVec3::ZERO,
        rotation: DQuat::IDENTITY,
        joint: None,
    };

    let mut bridge = PhysicsBridge3D::new(DVec3::ZERO);
    bridge.sync(std::slice::from_ref(&snap));
    let body = bridge.body_of(guid).unwrap();

    // Externally move the entity's transform, re-sync: the kinematic body tracks it.
    snap.translation = DVec3::new(3.0, 4.0, 5.0);
    bridge.sync(std::slice::from_ref(&snap));
    bridge.step(DT);

    let p = bridge.world().body_translation(body).unwrap();
    assert!((p - DVec3::new(3.0, 4.0, 5.0)).length() < 1e-9, "got {p:?}");
}

#[test]
fn removing_the_entity_despawns_the_body() {
    let scene = drop_scene();
    let ball = Uuid::from_u128(2);
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync(&scene);
    assert!(bridge.body_of(ball).is_some());

    // Re-sync with only the floor: the ball's body is gone.
    let floor_only = vec![scene[0].clone()];
    bridge.sync(&floor_only);
    assert!(bridge.body_of(ball).is_none(), "body should be despawned");
    assert_eq!(bridge.world().body_ids().len(), 1);
}

#[test]
fn snapshot_order_does_not_affect_result() {
    // The bridge sorts by Guid, so a shuffled snapshot yields the same handles
    // and the same simulation.
    fn run(reversed: bool) -> [u8; 24] {
        let mut scene = drop_scene();
        if reversed {
            scene.reverse();
        }
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync(&scene);
        for _ in 0..300 {
            bridge.step(DT);
        }
        let w = bridge.write_back();
        let ball = w.iter().find(|p| p.guid == Uuid::from_u128(2)).unwrap();
        let mut bytes = [0u8; 24];
        bytes[0..8].copy_from_slice(&ball.translation.x.to_le_bytes());
        bytes[8..16].copy_from_slice(&ball.translation.y.to_le_bytes());
        bytes[16..24].copy_from_slice(&ball.translation.z.to_le_bytes());
        bytes
    }
    assert_eq!(
        run(false),
        run(true),
        "snapshot order changed the simulation result"
    );
}

#[test]
fn two_identical_scenes_step_byte_identical() {
    fn run() -> [u8; 24] {
        let scene = drop_scene();
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync(&scene);
        for _ in 0..300 {
            bridge.step(DT);
        }
        let w = bridge.write_back();
        let ball = w.iter().find(|p| p.guid == Uuid::from_u128(2)).unwrap();
        let mut bytes = [0u8; 24];
        bytes[0..8].copy_from_slice(&ball.translation.x.to_le_bytes());
        bytes[8..16].copy_from_slice(&ball.translation.y.to_le_bytes());
        bytes[16..24].copy_from_slice(&ball.translation.z.to_le_bytes());
        bytes
    }
    assert_eq!(
        run(),
        run(),
        "two identical scenes diverged after 300 steps"
    );
}
