//! **The reverse collider→`Guid` map stays correct when it stops being rebuilt**
//! (Hardening Wave E).
//!
//! `PhysicsBridge3D`/`PhysicsBridge2D` used to rebuild `collider_to_guid` from
//! scratch at the end of **every** sync — one `BTreeMap` insert per collider in
//! the level, sixty times a second, to reproduce a map that changes only when
//! something spawns, despawns or has its shape edited. It is now behind a dirty
//! flag set by the three sites that can move a collider handle.
//!
//! That trade is only sound if no fourth site exists, and a missed one is
//! invisible: the map would simply answer with yesterday's handle, and the only
//! consumer is the collision-event drain — so the symptom is a `Collision` event
//! fired on the wrong actor, occasionally, in a shipped build. There is no pixel
//! and no counter for that.
//!
//! So this file drives the three transitions on a live bridge and asserts the
//! map after each: a **rebuilt** collider (shape edited), a **new** entity, and a
//! **despawn**. Each also asserts the negative — the retired handle must stop
//! resolving — because "the new handle resolves" is satisfied by a map that
//! grew and never shed.
//!
//! Mutation-verified: removing any one of the three `collider_map_dirty = true`
//! assignments fails the arm that names it, and only that arm.

use glam::{DQuat, DVec2, DVec3};
use uuid::Uuid;

use inf_physics::d3::{BodyDesc3D, ColliderDesc3D, EntitySync3D};
use inf_physics::{BodyKind3D, ColliderShape3D, PhysicsBridge3D};

const A: u128 = 0x1E11_0001;
const B: u128 = 0x1E11_0002;

fn boxed(half: f64) -> ColliderDesc3D {
    ColliderDesc3D::new(ColliderShape3D::Box {
        half_extents: DVec3::splat(half),
    })
}

fn entity(guid: u128, collider: ColliderDesc3D, x: f64) -> EntitySync3D {
    EntitySync3D {
        guid: Uuid::from_u128(guid),
        body: Some(BodyDesc3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }),
        collider: Some(collider),
        translation: DVec3::new(x, 0.0, 0.0),
        rotation: DQuat::IDENTITY,
        joint: None,
    }
}

/// The bridge's own view of the map, as the collision drain reads it.
fn resolves(bridge: &PhysicsBridge3D, guid: u128) -> bool {
    let g = Uuid::from_u128(guid);
    bridge
        .collider_of(g)
        .and_then(|c| bridge.guid_of_collider(c))
        == Some(g)
}

#[test]
fn a_rebuilt_collider_is_reachable_through_the_reverse_map() {
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync(&[entity(A, boxed(0.5), 0.0)]);
    let before = bridge.collider_of(Uuid::from_u128(A)).expect("A has one");
    assert!(resolves(&bridge, A));

    // Edit the shape — the bridge removes the old collider and attaches a new
    // one, which is a new handle.
    bridge.sync(&[entity(A, boxed(1.5), 0.0)]);
    let after = bridge
        .collider_of(Uuid::from_u128(A))
        .expect("A still has one");
    assert_ne!(
        before, after,
        "the fixture must actually rebuild the collider, or this arm is vacuous"
    );
    assert_eq!(
        bridge.guid_of_collider(after),
        Some(Uuid::from_u128(A)),
        "the reverse map did not learn the rebuilt collider's handle"
    );
    assert_eq!(
        bridge.guid_of_collider(before),
        None,
        "the reverse map still resolves the retired handle"
    );
}

#[test]
fn a_new_entity_is_reachable_through_the_reverse_map() {
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync(&[entity(A, boxed(0.5), 0.0)]);
    assert!(bridge.collider_of(Uuid::from_u128(B)).is_none());

    bridge.sync(&[entity(A, boxed(0.5), 0.0), entity(B, boxed(0.5), 4.0)]);
    assert!(
        resolves(&bridge, B),
        "a newly spawned entity's collider does not resolve back to it"
    );
    assert!(resolves(&bridge, A), "the incumbent stopped resolving");
}

#[test]
fn a_despawned_entity_leaves_the_reverse_map() {
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync(&[entity(A, boxed(0.5), 0.0), entity(B, boxed(0.5), 4.0)]);
    let gone = bridge.collider_of(Uuid::from_u128(B)).expect("B has one");
    assert!(resolves(&bridge, B));

    bridge.sync(&[entity(A, boxed(0.5), 0.0)]);
    assert_eq!(
        bridge.guid_of_collider(gone),
        None,
        "a despawned entity's collider still resolves — an event drained after \
         this would fire `Collision` on an actor that no longer exists"
    );
    assert!(resolves(&bridge, A), "the survivor stopped resolving");
}

/// The 2D mirror carries the same three sites and the same flag; one arm over
/// the transition most likely to be forgotten (a despawn) keeps it honest.
#[test]
fn the_two_d_mirror_sheds_a_despawned_collider() {
    use inf_ecs::components::{
        BodyKind2D, Collider2D, ColliderShape2DKind, RigidBody2D, Transform,
    };
    use inf_ecs::{EcsWorld, Vec3d};
    use inf_physics::PhysicsBridge2D;

    let mut w = EcsWorld::new();
    let mut spawn = |w: &mut EcsWorld, guid: u128, x: f64| {
        let e = w.spawn_with_guid(Uuid::from_u128(guid), "Body", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(x, 0.0, 0.0);
        w.world_mut().entity_mut(e).insert((
            RigidBody2D {
                kind: BodyKind2D::Static,
                ..Default::default()
            },
            Collider2D {
                shape_kind: ColliderShape2DKind::Box,
                half_extents: inf_ecs::Vec2d::new(0.5, 0.5),
                ..Default::default()
            },
            t,
        ));
        w.mark_dirty();
        e
    };
    spawn(&mut w, A, 0.0);
    let b = spawn(&mut w, B, 4.0);

    let mut bridge = PhysicsBridge2D::new(DVec2::new(0.0, -9.81));
    bridge.sync_from_world(&w);
    let gone = bridge.collider_of(Uuid::from_u128(B)).expect("B has one");
    assert_eq!(bridge.guid_of_collider(gone), Some(Uuid::from_u128(B)));

    w.despawn(b);
    bridge.sync_from_world(&w);
    assert_eq!(
        bridge.guid_of_collider(gone),
        None,
        "the 2D mirror still resolves a despawned collider"
    );
    assert_eq!(
        bridge
            .collider_of(Uuid::from_u128(A))
            .and_then(|c| bridge.guid_of_collider(c)),
        Some(Uuid::from_u128(A)),
        "the survivor stopped resolving"
    );
}
