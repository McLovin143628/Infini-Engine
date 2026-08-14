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

// ── C4-30: a collider that will not build (hardening wave B) ────────────────

/// A trimesh whose index buffer names a vertex it does not have. parry refuses
/// it, deterministically — the same *shape* of refusal `FIX_INTERNAL_EDGES`
/// gives on the non-manifold output Surface-Nets voxel chunks and Voronoi
/// fracture hulls produce, which this repo's own seam-chord law measured at
/// ~10 % of meshes.
fn unbuildable_trimesh() -> ColliderDesc3D {
    ColliderDesc3D::new(ColliderShape3D::Trimesh {
        vertices: vec![DVec3::ZERO, DVec3::X, DVec3::Z],
        indices: vec![[0, 1, 9]],
    })
}

/// **THE walk-through-walls fix.**
///
/// `add_collider` returning `None` used to be followed by `r.col =
/// snap.collider.clone()` — the DESIRED descriptor recorded as the ACHIEVED
/// one. The next reconcile compared them, found nothing to do, and the entity
/// kept a rigid body with no collider. No log, no counter, no advisory: the
/// only symptom is a player walking through a cave wall.
///
/// This asserts the world (does the body have a collider?) and the ledger (was
/// the refusal counted and attributed?), and then that a **changed** shape is
/// retried — which is the case where retrying can actually succeed, and the
/// thing the latch destroyed.
#[test]
fn a_collider_that_will_not_build_is_never_recorded_as_built() {
    let guid = Uuid::from_u128(7);
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));

    let mut scene = vec![entity(
        7,
        BodyKind3D::Dynamic,
        unbuildable_trimesh(),
        DVec3::new(0.0, 5.0, 0.0),
    )];
    bridge.sync(&scene);

    // The world: a body, and nothing to collide with.
    assert!(bridge.body_of(guid).is_some(), "the body must still exist");
    assert!(
        bridge.collider_of(guid).is_none(),
        "parry refused this shape; a handle here would mean it did not"
    );
    // The ledger: counted once, attributed to the entity.
    assert_eq!(bridge.collider_refusals(), 1);
    assert_eq!(bridge.entities_missing_colliders(), vec![guid]);
    // And the record does not CLAIM a shape it does not have. This is the
    // assertion the finding is actually named for, and it needs its own
    // accessor: `col_refused` is what bounds the retry, so a record that latched
    // `col` as well would behave identically through every other path — an
    // untestable claim, which is the shape of defect this campaign already met
    // once ("an arm that observes only the endpoints cannot see a window
    // between them").
    assert!(
        bridge.attached_collider_desc(guid).is_none(),
        "the bridge records a collider descriptor as attached while the body has none"
    );

    // Re-syncing the SAME descriptor must not re-attempt it: `to_shared_checked`
    // is a pure function of the descriptor, so a retry is trimesh-topology cost
    // per fixed step for an answer that cannot change.
    for _ in 0..5 {
        bridge.sync(&scene);
    }
    assert_eq!(
        bridge.collider_refusals(),
        1,
        "the same refused shape was re-attempted every sync"
    );
    assert!(bridge.collider_of(guid).is_none());

    // A CHANGED shape is retried — the case the latch destroyed. This is what a
    // re-carve, a re-fracture or a terrain edit looks like from here.
    scene[0].collider = Some(ball_collider());
    bridge.sync(&scene);
    assert!(
        bridge.collider_of(guid).is_some(),
        "a repaired shape was never retried — the latch is still there"
    );
    assert_eq!(
        bridge.collider_refusals(),
        1,
        "the repair counted as a fail"
    );
    assert!(bridge.entities_missing_colliders().is_empty());

    // And it really collides: drop it onto a floor and it must stop.
    scene.push(entity(1, BodyKind3D::Static, floor_collider(), DVec3::ZERO));
    bridge.sync(&scene);
    for _ in 0..240 {
        bridge.step(DT);
    }
    let y = bridge
        .write_back()
        .into_iter()
        .find(|w| w.guid == guid)
        .expect("the dynamic body must write back")
        .translation
        .y;
    assert!(
        y > 0.4 && y < 1.5,
        "the repaired body fell to y = {y}; it has no collider after all"
    );
}

/// The control: a healthy level refuses nothing. Without this the assertions
/// above cannot tell "the counter works" from "the counter counts everything".
#[test]
fn a_healthy_scene_refuses_no_colliders() {
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync(&drop_scene());
    assert_eq!(bridge.collider_refusals(), 0);
    assert!(bridge.entities_missing_colliders().is_empty());
}
