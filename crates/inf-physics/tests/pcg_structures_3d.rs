//! **P19.5: a `PcgVolume`'s derived solids really do become rapier colliders.**
//!
//! Before this batch, procedurally scattered content was categorically
//! walk-through: a `ScatteredInstance` is not an entity, has no `Guid`, and
//! `PhysicsBridge3D::sync_from_world`'s walk keys on exactly that. P19.5 makes a
//! grammar-built building enterable by giving the volume a second, *derived*
//! cache — `PcgVolume::structures` — and walking it here.
//!
//! These tests exist because the claim "the buildings have colliders" is
//! otherwise unfalsifiable: the building layer's own tests prove a
//! `ScatteredSolid` list is produced, and nothing proved it reached the
//! simulation. Everything below is about the *bridge*, not about buildings.

use glam::{DQuat, DVec3};
use inf_ecs::components::{PcgVolume, ScatteredSolid, Transform};
use inf_ecs::{EcsWorld, Vec2d};
use inf_physics::d3::pcg_structure_guid;
use inf_physics::PhysicsBridge3D;
use uuid::Uuid;

const VOLUME: Uuid = Uuid::from_u128(0x1955);

fn solid(center: DVec3, half: DVec3) -> ScatteredSolid {
    ScatteredSolid {
        center,
        half_extents: half,
        rotation: DQuat::IDENTITY,
    }
}

/// A world holding one `PcgVolume` carrying `solids`.
fn world_with(solids: Vec<ScatteredSolid>) -> EcsWorld {
    let mut w = EcsWorld::new();
    let e = w.spawn_with_guid(VOLUME, "Lot", None);
    w.world_mut().entity_mut(e).insert((
        Transform::IDENTITY,
        PcgVolume {
            extent: Vec2d::splat(20.0),
            structures: solids,
            ..Default::default()
        },
    ));
    w.mark_dirty();
    w
}

fn set_structures(w: &mut EcsWorld, solids: Vec<ScatteredSolid>) {
    let e = w.entity_of(VOLUME).expect("the volume exists");
    if let Some(mut v) = w.world_mut().get_mut::<PcgVolume>(e) {
        // Through the setter, which bumps the change stamp — the only supported
        // way to write the cache, and what tells the bridge to rebuild.
        v.set_structures(solids);
    }
    w.mark_dirty();
}

/// **The headline**: two derived solids become two static colliders, at the
/// transforms the solids declare.
#[test]
fn a_volumes_solids_become_static_colliders() {
    let a = solid(DVec3::new(1.0, 1.5, 2.0), DVec3::new(0.1, 1.5, 1.2));
    let b = solid(DVec3::new(-4.0, 0.1, 7.5), DVec3::new(2.0, 0.1, 3.0));
    let w = world_with(vec![a, b]);
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&w);

    for (i, s) in [a, b].into_iter().enumerate() {
        let guid = pcg_structure_guid(VOLUME, i);
        let body = bridge
            .body_of(guid)
            .unwrap_or_else(|| panic!("structure {i} has no body"));
        assert!(
            bridge.collider_of(guid).is_some(),
            "structure {i} has no collider"
        );
        let at = bridge
            .world()
            .body_translation(body)
            .expect("the body is live");
        assert!(
            (at - s.center).length() < 1e-12,
            "structure {i} is at {at:?}, not {:?}",
            s.center
        );
    }

    // A wall does not fall over: every structure body is STATIC, so stepping the
    // world leaves it exactly where the plan put it.
    for _ in 0..30 {
        bridge.step(1.0 / 60.0);
    }
    let body = bridge.body_of(pcg_structure_guid(VOLUME, 0)).unwrap();
    let at = bridge.world().body_translation(body).unwrap();
    assert_eq!(at, a.center, "a structure moved under gravity");
}

/// **Re-evaluation is reconciled, not leaked.** Dropping a solid despawns its
/// body; the volume's own entity never had one and still does not.
#[test]
fn dropping_a_solid_despawns_its_collider() {
    let mut w = world_with(vec![
        solid(DVec3::ZERO, DVec3::splat(1.0)),
        solid(DVec3::new(5.0, 0.0, 0.0), DVec3::splat(1.0)),
    ]);
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&w);
    assert!(bridge.body_of(pcg_structure_guid(VOLUME, 1)).is_some());

    set_structures(&mut w, vec![solid(DVec3::ZERO, DVec3::splat(1.0))]);
    bridge.sync_from_world(&w);
    assert!(
        bridge.body_of(pcg_structure_guid(VOLUME, 0)).is_some(),
        "the surviving structure was despawned"
    );
    assert!(
        bridge.body_of(pcg_structure_guid(VOLUME, 1)).is_none(),
        "a dropped structure kept its collider — re-evaluating a volume would leak"
    );

    // Clearing the cache entirely leaves nothing behind.
    set_structures(&mut w, Vec::new());
    bridge.sync_from_world(&w);
    assert!(bridge.body_of(pcg_structure_guid(VOLUME, 0)).is_none());
    // …and the volume entity itself never became a body: it has no collider of
    // its own, only derived ones.
    assert!(bridge.body_of(VOLUME).is_none());
}

/// **The change stamp**: a sync that sees no change must neither rebuild the
/// colliders nor despawn them. This is the per-fixed-step cost fix — the bridge
/// runs at 60 Hz over the whole world, and a furnished town is ~13 000 immovable
/// boxes — and its failure mode is silent (stale or vanished colliders), so it
/// is pinned rather than assumed.
#[test]
fn an_unchanged_volume_is_retained_across_syncs() {
    let mut w = world_with(vec![
        solid(DVec3::ZERO, DVec3::splat(1.0)),
        solid(DVec3::new(5.0, 0.0, 0.0), DVec3::splat(1.0)),
    ]);
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&w);
    let handles: Vec<_> = (0..2)
        .map(|i| bridge.body_of(pcg_structure_guid(VOLUME, i)).expect("body"))
        .collect();

    // Twenty no-change syncs later, the SAME handles are still live: nothing was
    // despawned and nothing was rebuilt (a rebuild would mint new handles).
    for _ in 0..20 {
        bridge.sync_from_world(&w);
    }
    for (i, h) in handles.iter().enumerate() {
        let got = bridge
            .body_of(pcg_structure_guid(VOLUME, i))
            .expect("still live");
        assert_eq!(got, *h, "structure {i} was rebuilt on an unchanged sync");
    }

    // A real change is still picked up — the stamp gates work, it does not
    // disable it.
    set_structures(
        &mut w,
        vec![solid(DVec3::new(9.0, 1.0, 9.0), DVec3::splat(1.0))],
    );
    bridge.sync_from_world(&w);
    let moved = bridge.body_of(pcg_structure_guid(VOLUME, 0)).expect("body");
    let at = bridge.world().body_translation(moved).unwrap();
    assert_eq!(
        at,
        DVec3::new(9.0, 1.0, 9.0),
        "a changed volume was not resynced"
    );
    assert!(bridge.body_of(pcg_structure_guid(VOLUME, 1)).is_none());
}

/// The setter is what bumps the stamp; a raw assignment does not, which is why
/// it is documented as unsupported.
///
/// The stamp is drawn from a **process-global** counter (island wave I8a audit),
/// so what is asserted is that it moves and never repeats — never that it counts.
#[test]
fn the_setter_bumps_the_change_stamp() {
    let mut v = PcgVolume::default();
    assert_eq!(v.structures_gen, 0);
    v.set_structures(vec![solid(DVec3::ZERO, DVec3::splat(1.0))]);
    let first = v.structures_gen;
    assert_ne!(first, 0);
    assert_eq!(v.structures.len(), 1);
    v.set_structures(Vec::new());
    assert!(v.structures_gen > first);
    assert!(v.structures.is_empty());
    // …and a NEW volume — which is what a cell reactivation builds under the
    // same guid — never lands back on a stamp this one already used.
    let mut reborn = PcgVolume::default();
    reborn.set_structures(vec![solid(DVec3::ZERO, DVec3::splat(1.0))]);
    assert!(reborn.structures_gen > v.structures_gen);
}

/// **The mirror pin.** `inf_ecs::ScatteredSolid` and `inf_pcg::PcgCollider` are
/// the same three fields declared twice — `inf-ecs` must not depend on the whole
/// PCG runtime just to hold a result cache, which is the same reason
/// `ScatteredInstance` mirrors `PcgInstance`.
///
/// Two hand-written copies of a struct is a triplication hazard (the standing
/// `GpuLight` law): add a field to one and the two evaluation sites that convert
/// between them silently drop it. This is the test that notices — it converts in
/// both directions and requires the round trip to be lossless, which it cannot be
/// if either side grows.
#[test]
fn scattered_solid_and_pcg_collider_stay_in_sync() {
    let s = ScatteredSolid {
        center: DVec3::new(1.5, -2.0, 30.25),
        half_extents: DVec3::new(0.1, 1.75, 1.2),
        rotation: inf_pcg::grammar::yaw_onto(DVec3::X),
    };
    // ECS → PCG, exactly as the gate's enterability arm converts.
    let p = inf_pcg::PcgCollider {
        center: s.center,
        half_extents: s.half_extents,
        rotation: s.rotation,
    };
    // PCG → ECS, exactly as both evaluation sites convert.
    let back = ScatteredSolid {
        center: p.center,
        half_extents: p.half_extents,
        rotation: p.rotation,
    };
    assert_eq!(back, s, "the mirrored types round-trip losslessly");
    // Field-for-field, on bits — a `PartialEq` that ever gained a tolerance
    // would not catch a drifting mirror.
    assert_eq!(
        back.center.to_array().map(f64::to_bits),
        s.center.to_array().map(f64::to_bits)
    );
    assert_eq!(
        back.half_extents.to_array().map(f64::to_bits),
        s.half_extents.to_array().map(f64::to_bits)
    );
    assert_eq!(
        back.rotation.to_array().map(f64::to_bits),
        s.rotation.to_array().map(f64::to_bits)
    );
    // Both are three fields and no more: `..` in either literal above would stop
    // compiling if one grew, and this size check catches a field added to BOTH
    // (which the round trip alone would tolerate).
    assert_eq!(
        std::mem::size_of::<ScatteredSolid>(),
        std::mem::size_of::<inf_pcg::PcgCollider>(),
        "the mirrored types have diverged in shape"
    );
}

/// **The second mirror pin** (IB-2b): `inf_ecs::StructureGroup` and
/// `inf_pcg::building::StructureGroup` are the same five fields declared twice,
/// for the same dependency reason the pin above states.
///
/// Sharper than its sibling, because four of the five fields are `u32` and a
/// permutation of them **compiles on both sides**. So the round trip uses four
/// distinct values and checks each by name, and the size check catches a field
/// added to both.
#[test]
fn structure_group_stays_in_sync_with_its_pcg_twin() {
    let shell = ScatteredSolid {
        center: DVec3::new(-8.0, 4.5, 12.0),
        half_extents: DVec3::new(6.0, 9.0, 3.5),
        rotation: inf_pcg::grammar::yaw_onto(DVec3::Z),
    };
    // Four DIFFERENT numbers: equal ones would let any permutation pass.
    let e = inf_ecs::StructureGroup {
        shell,
        start: 11,
        len: 22,
        inst_start: 33,
        inst_len: 44,
    };
    let p = inf_pcg::StructureGroup {
        shell: inf_pcg::PcgCollider {
            center: e.shell.center,
            half_extents: e.shell.half_extents,
            rotation: e.shell.rotation,
        },
        start: e.start,
        len: e.len,
        inst_start: e.inst_start,
        inst_len: e.inst_len,
    };
    let back = inf_ecs::StructureGroup {
        shell: ScatteredSolid {
            center: p.shell.center,
            half_extents: p.shell.half_extents,
            rotation: p.shell.rotation,
        },
        start: p.start,
        len: p.len,
        inst_start: p.inst_start,
        inst_len: p.inst_len,
    };
    assert_eq!(back, e, "the mirrored groups round-trip losslessly");
    assert_eq!(back.range(), 11..33);
    assert_eq!(back.instance_range(), 33..77);
    assert_eq!(p.range(), back.range());
    assert_eq!(p.instance_range(), back.instance_range());
    assert_eq!(
        std::mem::size_of::<inf_ecs::StructureGroup>(),
        std::mem::size_of::<inf_pcg::StructureGroup>(),
        "the mirrored groups have diverged in shape"
    );
}

/// The synthetic identity is a pure function of `(volume, index)`, distinct
/// across both — a XOR-based derivation would alias two volumes whose GUIDs
/// differ only in the low bits, which is exactly what sequential test GUIDs are.
#[test]
fn structure_guids_are_pure_and_do_not_alias() {
    let a = Uuid::from_u128(1);
    let b = Uuid::from_u128(2);
    assert_eq!(pcg_structure_guid(a, 7), pcg_structure_guid(a, 7));
    assert_ne!(pcg_structure_guid(a, 7), pcg_structure_guid(a, 8));
    assert_ne!(pcg_structure_guid(a, 7), pcg_structure_guid(b, 7));
    // The classic aliasing failure: volume A's structure 1 must not be volume
    // B's structure 0.
    assert_ne!(pcg_structure_guid(a, 1), pcg_structure_guid(b, 0));
    // And it never collides with the volume's own identity.
    assert_ne!(pcg_structure_guid(a, 0), a);
    let mut seen = std::collections::BTreeSet::new();
    for v in 0..8u128 {
        for i in 0..64 {
            assert!(
                seen.insert(pcg_structure_guid(Uuid::from_u128(v), i)),
                "collision at volume {v} index {i}"
            );
        }
    }
}

/// **The point of the whole exercise**: a dynamic body dropped onto a
/// structure's top face comes to rest *on* it instead of falling through.
#[test]
fn a_falling_body_lands_on_a_derived_structure() {
    use inf_ecs::components::{BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D};

    // A 10 x 0.5 x 10 slab whose top face is y = 0.
    let mut w = world_with(vec![solid(
        DVec3::new(0.0, -0.25, 0.0),
        DVec3::new(5.0, 0.25, 5.0),
    )]);
    let ball = w.spawn_with_guid(Uuid::from_u128(0xBA11), "Ball", None);
    w.world_mut().entity_mut(ball).insert((
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Sphere,
            radius: 0.5,
            ..Default::default()
        },
        Transform {
            translation: inf_ecs::Vec3d::new(0.0, 6.0, 0.0),
            ..Transform::IDENTITY
        },
    ));
    w.mark_dirty();

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&w);
    let body = bridge.body_of(Uuid::from_u128(0xBA11)).expect("the ball");
    for _ in 0..240 {
        bridge.step(1.0 / 60.0);
    }
    let y = bridge.world().body_translation(body).unwrap().y;
    assert!(
        (y - 0.5).abs() < 0.1,
        "the ball settled at y={y}, not on the slab's face — it fell through a \
         derived structure, which is the pre-P19.5 behaviour this batch fixed"
    );
}
