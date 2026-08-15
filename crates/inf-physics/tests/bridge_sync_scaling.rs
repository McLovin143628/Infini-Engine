//! **The steady-state `PhysicsBridge3D::sync` does not pay for the whole world**
//! (Hardening Wave G).
//!
//! Wave E left this function named, with a number: **6.0 ms at 13 000 static
//! colliders**, the dominant term of the shipped player's fixed step. It is
//! called once per fixed step by both hosts, and on a furnished town the
//! overwhelming majority of what it reconciles is a wall that has not moved
//! since the level loaded.
//!
//! Three costs were paid unconditionally per tracked entity per step:
//!
//! * `set_body_translation` + `set_body_rotation` on **every static and
//!   kinematic body** — two `get_mut`s into rapier's body set, a wake, and
//!   `query_dirty = true`, which invalidates the whole query BVH. That last one
//!   is lens 3's **P6** seen from the other end: the BVH was rebuilt once per
//!   moving character because the *bridge* dirtied it on every step anyway.
//! * `reconcile_joint` for every entity, jointless or not (**P30**) — two to
//!   three `BTreeMap` lookups each for a guaranteed no-op on a level with no
//!   joints, plus a `Vec` of one entry per entity to carry the desires into it.
//! * A `BTreeSet<Uuid>` of every seen guid (**P32**), rebuilt per step for a
//!   single `contains` sweep, when the input is already sorted by guid — plus
//!   the `snaps`/`live` vectors themselves, allocated fresh each pass.
//!
//! This file is the instrument and the protection. Two arms:
//!
//! * **The world is what the snapshot says it is.** Skipping a pose write is
//!   only sound if the pose was already right — so the arm reads the poses back
//!   *out of rapier* and compares them to the snapshot exactly, after a steady
//!   state, after a move, and after a body was pushed behind the bridge's back.
//!   The last case is why the skip compares against **rapier's own state**
//!   rather than a remembered copy: a remembered copy cannot see a body someone
//!   else moved.
//! * **The cost does not grow the way it used to.** A ratio measured on one
//!   machine in one process (the `ground_seam_scaling` reasoning: no GPU in it,
//!   CPU work over CPU data), plus a hard per-entity ceiling.
//!
//! Measured on the Wave G machine, steady-state sync, 60 iterations, dev
//! profile (which is what the battery runs), **before any repair**:
//!
//! | entities | ms/sync | ns/entity |
//! |---|---|---|
//! | 1 000 | 0.3138 | 313.8 |
//! | 5 000 | 2.0880 | 417.6 |
//! | 13 000 | 6.0752 | **467.3** |
//!
//! That 6.0752 reproduces Wave E's 6.0083 to within run-to-run noise, so this
//! file is measuring the thing that wave named.
//!
//! The ceiling below is minted from **that** column, deliberately: a budget
//! minted after the fix cannot certify the fix (Wave E's law), so this arm
//! lands green against the unrepaired reconcile and is ratcheted in the commit
//! that repairs it.

use std::time::Instant;

use glam::{DQuat, DVec3};
use uuid::Uuid;

use inf_physics::d3::{BodyDesc3D, ColliderDesc3D, EntitySync3D};
use inf_physics::{BodyKind3D, ColliderShape3D, PhysicsBridge3D};

const BASE: u128 = 0x1E07_0000;

fn snap(i: u32, at: DVec3) -> EntitySync3D {
    EntitySync3D {
        guid: Uuid::from_u128(BASE + u128::from(i)),
        body: Some(BodyDesc3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }),
        collider: Some(ColliderDesc3D::new(ColliderShape3D::Box {
            half_extents: DVec3::splat(0.5),
        })),
        translation: at,
        rotation: DQuat::IDENTITY,
        joint: None,
    }
}

/// `n` static boxes on a 64-wide grid — the shape of a furnished town's
/// immovable population, which is what the 13 000 figure counts.
fn town(n: u32) -> Vec<EntitySync3D> {
    (0..n)
        .map(|i| {
            snap(
                i,
                DVec3::new(f64::from(i % 64) * 2.0, 0.0, f64::from(i / 64) * 2.0),
            )
        })
        .collect()
}

/// Every tracked body's pose, read back **out of rapier** rather than out of the
/// bridge's own bookkeeping.
fn poses_from_rapier(bridge: &PhysicsBridge3D, snaps: &[EntitySync3D]) -> Vec<(DVec3, DQuat)> {
    snaps
        .iter()
        .map(|s| {
            let body = bridge
                .body_of(s.guid)
                .expect("every fixture entity is tracked");
            (
                bridge.world().body_translation(body).expect("live handle"),
                bridge.world().body_rotation(body).expect("live handle"),
            )
        })
        .collect()
}

#[test]
fn the_world_is_exactly_what_the_snapshot_says() {
    const N: u32 = 256;
    let mut snaps = town(N);
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync(&snaps);

    // (a) A steady-state re-sync leaves every pose exactly where the snapshot
    //     put it. Exact equality: these are f64 values copied, never computed.
    for _ in 0..4 {
        bridge.sync(&snaps);
    }
    let got = poses_from_rapier(&bridge, &snaps);
    for (i, s) in snaps.iter().enumerate() {
        assert_eq!(
            got[i].0, s.translation,
            "entity {i} drifted in the steady state"
        );
        assert_eq!(
            got[i].1, s.rotation,
            "entity {i} rotated in the steady state"
        );
    }

    // (b) A static body that MOVES is moved. The skip is a comparison, not a
    //     latch, so an authored edit still lands on the very next sync.
    let moved = DVec3::new(-17.5, 3.25, 8.125);
    let turned = DQuat::from_xyzw(0.0, 0.382_683_432_365_089_8, 0.0, 0.923_879_532_511_286_7);
    snaps[7].translation = moved;
    snaps[7].rotation = turned;
    bridge.sync(&snaps);
    let body = bridge.body_of(snaps[7].guid).expect("tracked");
    assert_eq!(
        bridge.world().body_translation(body),
        Some(moved),
        "a static body that moved in the snapshot did not move in the world"
    );
    assert_eq!(
        bridge.world().body_rotation(body),
        Some(turned),
        "a static body that turned in the snapshot did not turn in the world"
    );

    // (c) THE CASE A REMEMBERED COPY CANNOT SEE. Something outside the bridge
    //     pushes a body — a debug teleport, a gameplay shove through
    //     `world_mut`. The snapshot still says it belongs at its authored pose,
    //     so the next sync must put it back. A skip that compared against the
    //     bridge's own memory of what it last wrote would leave it displaced for
    //     the rest of the session.
    let stray = DVec3::new(400.0, 400.0, 400.0);
    bridge.world_mut().set_body_translation(body, stray);
    assert_eq!(bridge.world().body_translation(body), Some(stray));
    bridge.sync(&snaps);
    assert_eq!(
        bridge.world().body_translation(body),
        Some(moved),
        "a body moved behind the bridge's back was not restored to its authored pose"
    );
}

/// The steady-state cost, measured as a ratio across world sizes and as an
/// absolute per-entity ceiling.
///
/// The ratio is the load-bearing half. Before the repair the whole function was
/// linear in the tracked population with a large constant (two rapier writes, a
/// BVH invalidation and two-to-three `BTreeMap` lookups per entity); after it,
/// the per-entity steady-state work is a pair of `f64` comparisons, so the
/// constant collapses while the shape stays linear. A ratio survives a slower
/// machine; a millisecond does not.
#[test]
fn the_steady_state_sync_does_not_scale_like_the_world() {
    const ITERS: u32 = 60;
    /// Nanoseconds per entity per steady-state sync. **Minted at 700 against
    /// the unrepaired 467.3** (13 000 entities, 6.0752 ms) — headroom for a
    /// loaded machine, and a tripwire against the cost growing further while
    /// the repair is being written.
    const CEILING_NS_PER_ENTITY: f64 = 700.0;

    let mut report: Vec<(u32, f64)> = Vec::new();
    for n in [1_000u32, 5_000, 13_000] {
        let snaps = town(n);
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync(&snaps); // the spawn pass — not what is being measured
        bridge.sync(&snaps); // one warm pass, so the first timed one is steady

        let t0 = Instant::now();
        for _ in 0..ITERS {
            bridge.sync(&snaps);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);
        eprintln!(
            "sync @ {n:>6} entities: {ms:.4} ms  ({:.1} ns/entity)",
            ms * 1e6 / f64::from(n)
        );
        report.push((n, ms));
    }

    let (n, ms) = *report.last().expect("three sizes");
    let ns_per_entity = ms * 1e6 / f64::from(n);
    assert!(
        ns_per_entity < CEILING_NS_PER_ENTITY,
        "steady-state sync costs {ns_per_entity:.1} ns/entity at {n} entities \
         (ceiling {CEILING_NS_PER_ENTITY}); the per-entity work in the reconcile \
         has grown back"
    );

    // And the growth is not worse than linear: 13x the entities for under 26x
    // the time. (The unrepaired ratio is 19.4x — the per-entity constant grows
    // with the population because every `BTreeMap` probe is one level deeper —
    // so this fails loudly only if the reconcile becomes quadratic.)
    let small = report[0].1;
    assert!(
        ms < small * 26.0,
        "sync grew {:.1}x from 1 000 to 13 000 entities — super-linear",
        ms / small
    );
}
