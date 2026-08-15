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
//!
//! **Round 2 (Hardening Wave H)** added a fourth transition the file did not
//! have: a collider **removed** while its entity lives on. Wave B's C4-30
//! retry bound suppressed it — see `removing_a_collider_removes_it_from_the_world`
//! — and no instrument in this repo could see the difference, because every
//! accessor answered from the same stale record that was wrong. Those arms
//! therefore assert **rapier's own state** (`contains_collider`, a point
//! query), pin that the 2D and 3D bridges answer the removal question the same
//! way, and pin that the C4-30 bound they restore is still bounded.

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

/// The same entity with its `Collider3D` component taken away — a rigid body
/// and nothing to collide with. This is what the gather at `d3/ecs.rs` emits
/// for an actor whose collider was disabled, turned into a ghost, or edited
/// out in the Details panel during Simulate.
fn body_only(guid: u128, x: f64) -> EntitySync3D {
    EntitySync3D {
        collider: None,
        ..entity(guid, boxed(0.5), x)
    }
}

/// A trimesh whose index buffer names a vertex it does not have — parry
/// refuses it deterministically. Borrowed from `ecs_bridge_3d.rs`'s C4-30
/// fixture so the two files exercise one refusal.
fn unbuildable() -> ColliderDesc3D {
    ColliderDesc3D::new(ColliderShape3D::Trimesh {
        vertices: vec![DVec3::ZERO, DVec3::X, DVec3::Z],
        indices: vec![[0, 1, 9]],
    })
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

// ── Round-2 finding B1: a REMOVED collider actually leaves the world ────────

/// **THE removal arm.** Wave B's C4-30 repair bounded the retry with a second
/// conjunct, `rec.col_refused != snap.collider`. On the ordinary success path
/// (`col = Some(A)`, `col_refused = None`) a component removal makes
/// `snap.collider = None`, so that conjunct reads `None != None` = **false**
/// and the whole predicate is false: `remove_collider` never runs, the shape
/// stays in rapier for the rest of the session, and no later sync notices
/// because the record still says `Some(A)`.
///
/// Nothing in this repo could see it. `collider_of` answers from the record,
/// the reverse map answers from the record, and both are consistent with the
/// stale state — so this asserts the **world**: rapier's own
/// `contains_collider`, and a point query at the shape's centre.
#[test]
fn removing_a_collider_removes_it_from_the_world() {
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync(&[entity(A, boxed(0.5), 0.0)]);
    let handle = bridge.collider_of(Uuid::from_u128(A)).expect("A has one");
    assert!(
        bridge.world().contains_collider(handle),
        "the fixture never built a collider, so this arm is vacuous"
    );
    assert!(
        !bridge.world_mut().intersect_point(DVec3::ZERO).is_empty(),
        "the fixture's box does not cover its own centre"
    );

    // Take the component away. The entity keeps its rigid body.
    bridge.sync(&[body_only(A, 0.0)]);

    assert!(
        !bridge.world().contains_collider(handle),
        "the collider component was removed and the shape is still in rapier — \
         a ghost/disabled-trigger/Details edit during Simulate leaves the world \
         solid where the level says it is empty"
    );
    assert!(
        bridge.collider_of(Uuid::from_u128(A)).is_none(),
        "the bridge still hands out a handle for a collider that is gone"
    );
    assert!(
        bridge.attached_collider_desc(Uuid::from_u128(A)).is_none(),
        "the record still claims an attached descriptor after the removal"
    );
    assert!(
        bridge.world_mut().intersect_point(DVec3::ZERO).is_empty(),
        "a point query still hits the removed shape"
    );
    assert_eq!(
        bridge.guid_of_collider(handle),
        None,
        "the reverse map still resolves the removed handle"
    );
    assert!(
        bridge.body_of(Uuid::from_u128(A)).is_some(),
        "removing the collider must not remove the body"
    );

    // And it comes back when the component does — the removal must not latch
    // either.
    bridge.sync(&[entity(A, boxed(0.5), 0.0)]);
    let again = bridge
        .collider_of(Uuid::from_u128(A))
        .expect("re-adding the component must re-attach a collider");
    assert!(bridge.world().contains_collider(again));
    assert!(resolves(&bridge, A));
}

/// The 2D mirror never had the extra conjunct (`rec_col != snap.col`), and its
/// comment says so. This pins that the two bridges answer the same question,
/// because both PIE and shipping drive the 3D one and every parity gate
/// compares two hosts that share it — so a 3D-only divergence is invisible to
/// all of them.
#[test]
fn the_two_bridges_agree_on_what_a_removal_means() {
    use inf_ecs::components::{
        BodyKind2D, Collider2D, ColliderShape2DKind, RigidBody2D, Transform,
    };
    use inf_ecs::EcsWorld;
    use inf_physics::PhysicsBridge2D;

    // 3D.
    let mut b3 = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    b3.sync(&[entity(A, boxed(0.5), 0.0)]);
    let h3 = b3.collider_of(Uuid::from_u128(A)).expect("3D has one");
    b3.sync(&[body_only(A, 0.0)]);
    let three_d_shed = !b3.world().contains_collider(h3);

    // 2D, through the ECS door the bridge actually reads.
    let mut w = EcsWorld::new();
    let e = w.spawn_with_guid(Uuid::from_u128(A), "Body", None);
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
        Transform::IDENTITY,
    ));
    w.mark_dirty();
    let mut b2 = PhysicsBridge2D::new(DVec2::new(0.0, -9.81));
    b2.sync_from_world(&w);
    let h2 = b2.collider_of(Uuid::from_u128(A)).expect("2D has one");
    w.world_mut().entity_mut(e).remove::<Collider2D>();
    w.mark_dirty();
    b2.sync_from_world(&w);
    let two_d_shed = !b2.world().contains_collider(h2);

    assert!(
        two_d_shed,
        "the 2D mirror kept a removed collider — this arm's control failed"
    );
    assert_eq!(
        three_d_shed, two_d_shed,
        "the 2D and 3D bridges disagree about what removing a collider means"
    );
}

/// Removing a collider that was **refused** clears the refusal ledger, so the
/// entity stops being reported as "asked for a collider and has none".
///
/// This is the other half of the one-term predicate: `decided` is
/// `col.or(col_refused)`, so `Some(refused) -> None` is a change and the
/// record is cleaned. Written as two conjuncts it was not, and
/// `entities_missing_colliders()` kept naming an entity that no longer wants
/// one.
#[test]
fn removing_a_refused_collider_clears_the_refusal() {
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    let mut scene = vec![entity(A, unbuildable(), 0.0)];
    bridge.sync(&scene);
    assert_eq!(bridge.collider_refusals(), 1);
    assert_eq!(
        bridge.entities_missing_colliders(),
        vec![Uuid::from_u128(A)],
        "the fixture's shape was accepted, so this arm is vacuous"
    );

    scene[0].collider = None;
    bridge.sync(&scene);
    assert!(
        bridge.entities_missing_colliders().is_empty(),
        "an entity with no collider component is still reported as missing one"
    );

    // The retry is still bounded: re-adding the same bad shape refuses once
    // more and then stops, exactly as C4-30 requires.
    scene[0].collider = Some(unbuildable());
    for _ in 0..5 {
        bridge.sync(&scene);
    }
    assert_eq!(
        bridge.collider_refusals(),
        2,
        "the refused shape is being re-attempted every sync — the C4-30 bound \
         was traded away, not repaired"
    );
}

/// Round-2 finding R2-3: the "already warned about this entity" set is pruned
/// with the entity.
///
/// Its keys are synthetic — one per voxel chunk, terrain tile or fracture
/// piece — so a session that streams a cave in and out accumulates one `Uuid`
/// per failed chunk per visit for ever. The five sibling maps in the same
/// struct are all retained against the live set.
#[test]
fn the_warned_set_is_pruned_with_its_entity() {
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    let mut scene = vec![entity(A, boxed(0.5), 0.0), entity(B, unbuildable(), 4.0)];
    bridge.sync(&scene);
    assert_eq!(
        bridge.warned_collider_count(),
        1,
        "the fixture's shape was accepted, so this arm is vacuous"
    );

    scene.pop();
    bridge.sync(&scene);
    assert_eq!(
        bridge.warned_collider_count(),
        0,
        "the warn set kept a despawned entity — it is the one map in this \
         struct that can only grow, and its keys are minted per chunk"
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
    let spawn = |w: &mut EcsWorld, guid: u128, x: f64| {
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
