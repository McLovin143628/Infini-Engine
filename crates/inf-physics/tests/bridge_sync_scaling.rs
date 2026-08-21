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
//!   a `query_dirty = true` that invalidated the whole query BVH. That last one
//!   is lens 3's **P6** seen from the other end: the BVH was rebuilt once per
//!   moving character because the *bridge* dirtied it on every step anyway.
//!   *(Island wave I4b retired the flag for per-body marks; the skip this file
//!   measures is what keeps those marks empty on a town that has not moved.)*
//! * `reconcile_joint` for every entity, jointless or not (**P30**) — two to
//!   three `BTreeMap` lookups each for a guaranteed no-op on a level with no
//!   joints, plus a `Vec` of one entry per entity to carry the desires into it.
//! * A `BTreeSet<Uuid>` of every seen guid (**P32**), rebuilt per step for a
//!   single `contains` sweep, when the input is already sorted by guid — plus
//!   the `snaps`/`live` vectors themselves, allocated fresh each pass.
//!
//! This file is the instrument and the protection. Three arms:
//!
//! * **The world is what the snapshot says it is.** Skipping a pose write is
//!   only sound if the pose was already right — so the arm reads the poses back
//!   *out of rapier* and compares them to the snapshot exactly, after a steady
//!   state, after a move, and after a body was pushed behind the bridge's back.
//!   The last case is why the skip compares against **rapier's own state**
//!   rather than a remembered copy: a remembered copy cannot see a body someone
//!   else moved.
//! * **The skip fires**, said without a clock (island wave I4b): a steady-state
//!   sync must not mark one body as moved, and a moved wall must mark one. Added
//!   because this wave measured what the ceiling below no longer catches on its
//!   own — see that arm's docs.
//! * **The cost does not grow the way it used to.** In P26.5's budget classes:
//!   **WORLD** — a per-entity *scaling ratio* across world sizes, divided by the
//!   scaling of [`calibration_ns`] measured over the SAME two populations in the
//!   SAME process (the `ground_seam_scaling` reasoning: no GPU in it, CPU work
//!   over CPU data); and **CLOCK** — the absolute per-entity ceiling, converted
//!   into this machine's nanoseconds by that same control.
//!
//!   Both halves are that shape because both of the first two drafts were red on
//!   a runner with nothing wrong with the code: the ceiling unconditionally, at
//!   276.9 ns/entity on ubuntu against 93.2 here; then a FIXED growth ceiling of
//!   3x, at 3.32x on macOS against 1.39–1.62 here. A `BTreeMap` descent is
//!   `O(log n)` and its working set leaves cache between the two populations, so
//!   the growth is machine-sensitive too — dividing by the calibration's own
//!   growth is what cancels both.
//!
//!   **And a third red, which is why the control looks the way it does now**
//!   (island wave I4b). At `fc34632` the ubuntu runner read 318.8 / 412.8 /
//!   430.2 ns/entity against 57.6 / 82.1 / 97.8 here, on a `sync` path that the
//!   whole wave had not touched by one byte. The control could not see it: it
//!   read **98.6** ns/entry, 1.37x its reference, and normalized 430.2 to 314.1.
//!   The reason it could not see it is that the control and its subject did not
//!   live in the same part of the memory hierarchy — a `BTreeMap<Uuid, [f64; 4]>`
//!   at 13 000 entries is ~0.9 MB and stays in cache, while the reconcile's own
//!   working set (its tracked records, rapier's two arenas, the snapshot vector)
//!   is ~15 MB and does not. Measured here, with four threads streaming 64 MB
//!   each in the background: the subject inflates **2.6–3.1x** and that control
//!   **1.00–1.15x**. A control that barely moves while its subject triples is not a
//!   control, and the CI history says so in the sharpest possible way — between
//!   the two ubuntu readings the control got **faster** (108.7 -> 98.6 ns/entry
//!   at 13 000; 112.6 -> 67.1 at 1 000) and the whole battery got faster (204 s
//!   for 2 958 tests -> 163 s for 3 037) while the subject doubled.
//!
//!   [`calibration_ns`] therefore descends a map whose entries are the size of
//!   the records the reconcile descends, comparing the fields the skip compares.
//!   Over that same background load it inflates **2.9–4.1x** where the subject
//!   inflates 2.6–3.1x — the two now move together, which is the only property a
//!   divisor has to have.
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

/// **The skip fires** — as a fact about the world, not a number on a clock
/// (island wave I4b).
///
/// The CLOCK half below is a ratchet on the reconcile's *total* per-entity cost,
/// and this wave measured what it no longer catches: with the skip forced open,
/// a steady-state sync at 13 000 entities costs **108.9** ns/entity against
/// 92.3 repaired, because the two other costs the 467.3 figure contained
/// (`reconcile_joint` per entity, a `BTreeSet` of every guid per step) were
/// repaired in the same wave and the `query_dirty = true` that used to ride
/// along was retired in this one. A defect the ratchet would have caught by a
/// factor of five now moves it by 18 %, which is inside the headroom the
/// ceiling was given for a loaded runner.
///
/// So the skip gets an arm that has no clock in it at all.
/// `set_body_pose_if_moved` returns before it touches the body list behind
/// `PhysicsWorld3D::pending_query_marks`, and every static pose write goes
/// through it — so on a town that has not moved, that list must not grow by one
/// entry, and on a town where one wall moved, it must. Measured against the
/// mutation: 2 048 marks with the skip forced open (256 bodies x 4 syncs x the
/// two writes a pose is), against 0.
#[test]
fn a_steady_state_sync_marks_no_body_as_moved() {
    const N: u32 = 256;
    let mut snaps = town(N);
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync(&snaps);
    // Whatever the spawn pass left pending is the baseline; nothing here drains
    // it, because nothing here makes a query.
    let base = bridge.world().pending_query_marks().0;

    for _ in 0..4 {
        bridge.sync(&snaps);
    }
    assert_eq!(
        bridge.world().pending_query_marks().0,
        base,
        "a steady-state sync over {N} static bodies marked a body as moved — the \
         pose-write skip is not firing, and every one of those marks is a \
         `RigidBodySet::get_mut`, a wake and a stale query leaf paid sixty times \
         a second for a town that has not moved"
    );

    // ANTI-VACUITY: the same counter has to move when a wall really moves, or
    // the assertion above is satisfied by a counter that is simply dead.
    snaps[7].translation = DVec3::new(-17.5, 3.25, 8.125);
    bridge.sync(&snaps);
    assert!(
        bridge.world().pending_query_marks().0 > base,
        "a static body that moved in the snapshot left the query tree's mark list \
         untouched — the counter this arm reads is not the one the write path writes"
    );
}

/// **The reference workload both halves below are calibrated against**, at a
/// population of `n`.
///
/// Returns nanoseconds per entry for one pass, measured in **this** process, on
/// **this** machine, moments before the arm that uses it.
///
/// It is deliberately the steady-state sync's own dominant term and nothing
/// else: for each entry of a snapshot already in `guid` order, one descent
/// through a `BTreeMap` keyed by `Uuid` to the record the bridge kept, and then
/// the comparisons the skip is made of — pose against pose, collider descriptor
/// against collider descriptor, every one of them answering *equal*, which is
/// what a town that has not moved answers. That is what the reconcile does per
/// tracked entity once every skip has fired, so this converts a nanosecond on
/// the machine running it into a nanosecond on the machine the ceiling was
/// minted on.
///
/// # The value type is the whole point (island wave I4b)
///
/// This used to descend a `BTreeMap<Uuid, [f64; 4]>` — the right *shape* and
/// the wrong *footprint*. At 13 000 entries that tree is ~0.9 MB and stays in
/// cache on anything; the reconcile's own working set at the same population is
/// ~15 MB (its `BodyRecord`s, rapier's body and collider arenas, the snapshot
/// vector) and stays in cache on nothing. A divisor that lives one level of the
/// memory hierarchy above its subject cancels the machine's clock and none of
/// its memory system, so a runner whose cores are *faster* and whose memory is
/// slower or busier reads the control low and the subject high — which is
/// exactly the red that sent this file back for a third time. See the module
/// docs for the two ubuntu readings that state it in numbers.
///
/// So the entries here are [`EntitySync3D`] — 416 bytes, the size of what the
/// bridge really keys by guid — and the incoming side is the same `town` vector
/// the arm itself syncs. Same bytes per entry, same order, same compares. On
/// the calibrating machine, under four background threads streaming 64 MB each,
/// the subject's per-entity cost at 13 000 inflates 2.6–3.1x and this control's
/// 2.9–4.1x; the control it replaces inflated 1.00–1.15x.
///
/// **It is parameterised on `n` because the WORLD half needs its GROWTH**, not
/// just its speed — see the arm below. A `BTreeMap` descent is `O(log n)` per
/// lookup and its working set at 13 000 entries does not fit the caches that
/// hold it at 1 000, so this function's own cost per entry rises with `n` by
/// close to the amount the reconcile's does, on whatever machine is running it.
///
/// # Best of three, and which direction that errs in
///
/// The first pass is discarded (it faults the tree in) and the smallest of the
/// three that follow is returned. A microbenchmark that is preempted reads
/// **high**, and a divisor that reads high deflates the normalized cost and
/// *manufactures slack* — the failure mode this file has already been burned by
/// once, and named. Taking the minimum can only make the divisor smaller, so it
/// can only make both claims below stricter. It costs ~3 ms at 13 000 entries.
///
/// `black_box` on both ends: the whole loop is dead code otherwise.
fn calibration_ns(n: u32) -> f64 {
    use std::collections::BTreeMap;
    use std::hint::black_box;

    // The two sides the reconcile has: a snapshot in guid order, and the
    // records the bridge kept, keyed by guid.
    let want = town(n);
    let have: BTreeMap<Uuid, EntitySync3D> = want.iter().map(|s| (s.guid, s.clone())).collect();

    let mut best = f64::INFINITY;
    for pass in 0..4 {
        let t0 = Instant::now();
        let mut moved = 0u32;
        for s in &want {
            if let Some(rec) = have.get(black_box(&s.guid)) {
                if rec.translation != s.translation
                    || rec.rotation != s.rotation
                    || rec.collider != s.collider
                {
                    moved += 1;
                }
            }
        }
        let ns = t0.elapsed().as_secs_f64() * 1e9 / f64::from(n);
        black_box(moved);
        if pass > 0 {
            best = best.min(ns);
        }
    }
    best
}

/// The steady-state cost, measured as a **scaling ratio** (WORLD — asserted
/// everywhere) and as an **absolute per-entity ceiling** (CLOCK — asserted only
/// where the machine is in the class the number was measured on).
///
/// # Why the split (round 2, Hardening Wave H)
///
/// The first version of this arm asserted `< 200 ns/entity` unconditionally. It
/// is red on the ubuntu runner at **276.9**, against **93.2** on the machine it
/// was minted on — a three-times-slower shared CI box, with nothing wrong with
/// the code. That is the P26.5 budget class exactly: *a millisecond on one
/// machine is not a contract*, and the repair is that precedent's LOAD / WORLD /
/// CLOCK split.
///
/// * **WORLD** — the *scaling* property, measured **against the calibration's
///   own scaling**. The reconcile's per-entity cost may grow from 1 000 to
///   13 000 entities by at most `GROWTH_MARGIN` times as much as a bare
///   `BTreeMap` descent does over the same two populations, in the same
///   process, on the same machine.
///
///   **A fixed ratio ceiling here was wrong, and this arm's own failure is the
///   measurement.** The first cut of the split asserted a flat `< 3x`, measured
///   1.39–1.62 on the calibrating machine — and the macOS runner returned
///   **3.32x** (108.7 -> 360.9 ns/entity) with nothing reintroduced. Two
///   legitimate reasons, both of which this arm's own analysis already supplies
///   and neither of which is population-work:
///
///   * the dominant term is a **`BTreeMap` descent**, which is `O(log n)` per
///     entity — `log(13000)/log(1000)` is ~1.37x from the *size alone*, before
///     any machine effect; and
///   * the 13 000-entity working set is ~5 MB and falls out of a slow runner's
///     cache, while the 1 000-entity set (~400 KB) does not, so the memory
///     hierarchy inflates the ratio by a factor that is a property of the
///     **machine**, not of the code.
///
///   A number that mixes both cannot be a constant. Dividing by the
///   calibration's growth cancels them: every figure in the comparison shares
///   one machine, one cache hierarchy, one load, and one instant. What is left
///   is the only thing the arm was ever claiming — that the reconcile does not
///   scale *worse than a map lookup*, which is what a per-entity `contains`
///   over the seen set, a scan per contact or a rebuild of the reverse map
///   would make it do, without bound.
/// * **CLOCK** — the 200 ns/entity ratchet. Real, and still the honest number
///   for the class of machine it was measured on, so it is kept rather than
///   loosened to fit the slowest runner (which would retire it). It is asserted
///   after [`calibration_ns`] — the same descent plus the same compares the
///   reconcile is *made of*, over entries the same size, run in this process
///   moments earlier — has converted this machine's nanoseconds into that
///   machine's. `CALIBRATION_TOLERANCE` is no longer the class gate it was: the
///   conversion is what admits a slower machine, and the tolerance is only the
///   point past which a control reading is a *preemption* rather than a
///   measurement.
///
/// # Measured, dev profile (what the battery runs)
///
/// | entities | before the Wave G repair | after | island I4b |
/// |---|---|---|---|
/// | 1 000 | 313.8 ns/entity | 60.6 | 58.8 |
/// | 5 000 | 417.6 | 78.5 | 80.7 |
/// | 13 000 | **467.3** | **94.7** | **94.3** |
///
/// The third column is the same machine at `fc34632`, the tree the third red
/// was raised against — flat against Wave G's repair, which is half the reason
/// that red was read as a broken control rather than as a regression. (The
/// other half is that the ubuntu runner's own control got *faster* between the
/// two readings.)
///
/// # What the ceiling catches, measured rather than assumed
///
/// Two defects this file names, each restored on its own on the calibrating
/// machine at 13 000 entities: the P32 `BTreeSet` of every seen guid, rebuilt
/// per step for one `contains` sweep — **182.7** ns/entity; the pose-write skip
/// forced open — **108.9**; both at once — **193.7**. Against a repaired 94.3
/// and a ceiling of 200, only the first is close, and none of them is over.
///
/// That is stated rather than left implied because it is the honest reading of
/// this ratchet's reach: it is a bound on the reconcile's **total** per-entity
/// cost, worth roughly a doubling, and the individual repairs it was minted
/// beside are each guarded by an arm of their own —
/// `a_steady_state_sync_marks_no_body_as_moved` for the skip, the despawn
/// sweep's own arms for the merge. The ceiling is not the place to learn that
/// one of them came undone; it is the place to learn that the sum did.
///
/// Note what the ratio does and does not say: 13k/1k is **1.49 unrepaired** and
/// **1.56 repaired**, so the ratio is *not* the ratchet and never was — the
/// per-entity constant grows slightly with the population either way, because
/// every `BTreeMap` probe is one level deeper and the tracked records no longer
/// fit in cache. Collapsing the constant made that more visible, not less. The
/// ratio bound is a **superlinearity** bound; the ceiling is the ratchet. They
/// are two different claims, and this file states them separately instead of
/// resting both on one wall-clock number.
///
/// And those two sentences are exactly why the ratio needs a *measured*
/// divisor rather than a constant: "one level deeper" and "no longer fit in
/// cache" are both properties of the machine the numbers were taken on, and a
/// slower machine with a smaller cache pays more for both. `GROWTH_MARGIN`
/// divides them out.
///
/// # …and why the divisor has to be made of the same bytes (island wave I4b)
///
/// The sentence above is the whole finding of the third red, read one clause
/// too shallow. "A slower machine with a smaller cache pays more for both" is
/// only a thing a divisor can cancel **if the divisor's own working set is in
/// the same part of the memory hierarchy as its subject's**. It was not: a
/// `BTreeMap<Uuid, [f64; 4]>` at 13 000 entries is ~0.9 MB against the
/// reconcile's ~15 MB, so on a runner with fast cores and a busy memory system
/// the control read *low* while the subject read *high*, and the normalization
/// pointed the wrong way. [`calibration_ns`] now keys entries the size of the
/// records the reconcile keys. Measured on the calibrating machine over fifteen
/// runs, `subject / control` at 13 000 entities is **1.12–1.40** quiet and
/// **0.75–1.24** under a four-thread memory hog; the control it replaces read
/// **1.32–1.65** quiet and **4.10–5.07** loaded. The repair is that second
/// column collapsing onto the first.
#[test]
fn the_steady_state_sync_does_not_scale_like_the_world() {
    const ITERS: u32 = 60;
    /// Nanoseconds per entity per steady-state sync. Minted at **700** against
    /// the unrepaired **467.3** (13 000 entities, 6.0752 ms); **ratcheted to
    /// 200** against the repaired **93.2** (1.2116 ms) in the commit that
    /// repaired it. The headroom is for a loaded machine, and 200 is still less
    /// than half the cost this arm was born measuring.
    ///
    /// **CLOCK**: asserted after the control has converted this machine's
    /// nanoseconds into the calibrating machine's.
    ///
    /// **NOT raised by the I4b repair, deliberately.** The third red was a
    /// broken divisor, and a ceiling raised to cover a broken divisor is a
    /// retired ratchet wearing a number.
    const CEILING_NS_PER_ENTITY: f64 = 200.0;
    /// [`calibration_ns`] on the calibrating machine (dev profile), measured at
    /// 13 000 entries over **fifteen** runs: 71.4, 72.8, 73.1, 73.6, 73.7,
    /// 73.7, 73.8, 73.8, 74.0, 75.4, 76.4, 76.7, 77.4, 81.3, 87.0 ns/entry.
    /// The reference is the middle of that; the worst reading is 1.16x, which
    /// is a seventh of the tolerance below.
    ///
    /// Re-minted in the I4b repair because the control it names changed — the
    /// old **72** was the same descent over 32-byte values, and 72 of those
    /// nanoseconds were never the same quantity as 72 of these.
    const CALIBRATION_REF_NS: f64 = 75.0;
    /// How far from `CALIBRATION_REF_NS` a control reading may land and still
    /// be treated as a measurement of this machine rather than as a leg that
    /// was preempted.
    ///
    /// **It is not a machine-class gate any more** (island wave I4b). It was
    /// 1.6, and it was doing two jobs: admitting only machines near the
    /// calibrating one, *and* rejecting garbage. The first job is the
    /// normalization's, and it is only honest to hand it over now that the
    /// control shares its subject's footprint — a control that inflates with
    /// the subject is exactly what makes `raw / ratio` mean something on a
    /// machine three times slower. Left at 1.6, the new control would SKIP the
    /// CLOCK half on every runner that has ever reddened it (the loaded
    /// readings are 3.4–5.1x), which is a gate that cannot fail — the very
    /// thing the "gates must falsify" law forbids.
    ///
    /// 8.0 is the second job alone: past it, a control that is *made of* the
    /// subject's own bytes has read something the 60-iteration subject leg
    /// beside it cannot have escaped, so the reading is the scheduler's and not
    /// the machine's. Measured worst case on real hardware to date: 5.07x, with
    /// four threads streaming 64 MB each in the background.
    const CALIBRATION_TOLERANCE: f64 = 8.0;
    /// **WORLD**: how much faster than the CALIBRATION's own per-entity growth
    /// the reconcile's may grow, over the same two populations in the same
    /// process. See the function docs for why this is a ratio of ratios and not
    /// a constant.
    ///
    /// Measured on the calibrating machine over **fifteen** runs of the I4b
    /// control (`measured / calibration`): 1.06, 1.16, 1.17, 1.18, 1.22, 1.25,
    /// 1.26, 1.30, 1.32, 1.33, 1.34, 1.35, 1.36, 1.38, **1.41** — and 0.71,
    /// 1.18, 1.29 with the memory hog running. (The control this replaces
    /// measured 0.69–1.45 over fourteen; the band is the same width and better
    /// centred, because the divisor now leaves cache where the subject does.)
    /// `3.0` is a little over twice the worst of those, which is the headroom a
    /// shared runner under load needs — and it is still far below what the
    /// defect looks like: a term linear in the population inside the per-entity
    /// step makes the measured growth ~13x while the calibration's stays at its
    /// `log n`-plus-cache figure of ~1.3–1.7x, so the slack lands near **8x**,
    /// well clear of the margin in the other direction.
    ///
    /// Both readings below 1.0 are the divisor being noisy upward (one run saw
    /// a 2.43x calibration growth), which only *passes* a regression. Noise in
    /// the **small** leg is the unsafe direction — it deflates the divisor and
    /// manufactures slack (a CI run measured 427.5 ns/entry there against the
    /// 72 reference, a 0.16x "growth", 16.84x slack, on byte-untouched code) —
    /// so a divisor that invalidates itself is named and skipped below rather
    /// than divided by.
    const GROWTH_MARGIN: f64 = 3.0;
    /// The two populations the growth is measured across. Named so the arm and
    /// the calibration cannot drift apart about which sizes they compare.
    const GROWTH_SMALL: u32 = 1_000;
    const GROWTH_LARGE: u32 = 13_000;

    // Not measured on a paravirtual macOS runner at all. That environment has
    // gone red on this arm four times while `inf-physics` was byte-untouched —
    // the last two on *consecutive* runs, one per half, each half's control
    // blaming the opposite direction: the CLOCK trip read the workload 15% hot
    // (230.4 ns/entity) while its calibration read 0.99x reference, and the
    // next run's WORLD trip read the workload's raw growth at 2.63x — the last
    // green run's own figure — while the calibration's small leg read 427.5
    // ns/entry, a descent that "sped up" six-fold per entry as the tree grew
    // thirteen-fold. An environment that invalidates both controls in opposite
    // directions on identical bytes cannot support a ratio between two
    // microbenchmarks. The functional arms in this file still run there; real
    // macOS hardware still measures; Windows and Linux CI still assert both
    // halves.
    if cfg!(target_os = "macos") && std::env::var_os("CI").is_some() {
        eprintln!(
            "SKIP the measured halves: paravirtual macOS CI has invalidated both of \
             this arm's controls in opposite directions on byte-identical code. The \
             functional arms in this file still ran."
        );
        return;
    }

    // `(entities, ms/sync, control ns/entry)`. The control leg is taken
    // **inside** this loop, next to the subject leg it divides (island wave
    // I4b): a shared runner's memory system is busy in bursts, and two numbers
    // meant to cancel each other have to be taken in the same burst. The old
    // arrangement took all three subject legs and then both control legs, so
    // ~400 ms of scheduling could sit between a figure and its divisor.
    let mut report: Vec<(u32, f64, f64)> = Vec::new();
    for n in [GROWTH_SMALL, 5_000, GROWTH_LARGE] {
        let snaps = town(n);
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync(&snaps); // the spawn pass — not what is being measured
        bridge.sync(&snaps); // one warm pass, so the first timed one is steady

        let t0 = Instant::now();
        for _ in 0..ITERS {
            bridge.sync(&snaps);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let ms = elapsed * 1e3 / f64::from(ITERS);
        let calib = calibration_ns(n);
        eprintln!(
            "sync @ {n:>6} entities: {ms:.4} ms  ({:.1} ns/entity), control {calib:.1} ns/entry",
            ms * 1e6 / f64::from(n)
        );
        // ANTI-VACUITY, per leg: a ratio between two numbers that are really
        // timer resolution measures nothing. 60 syncs of 1 000 tracked entities
        // is ~4 ms on the fastest machine this has run on, and the coarsest
        // `Instant` in play is ~100 ns.
        assert!(
            elapsed > 1e-4,
            "the {n}-entity leg measured {elapsed:e} s over {ITERS} syncs — that is \
             timer resolution, not a cost, and every claim below would be noise"
        );
        report.push((n, ms, calib));
    }

    let (small_n, small_ms, calib_small) = report[0];
    let (n, ms, calib_large) = *report.last().expect("three sizes");
    let ns_per_entity = ms * 1e6 / f64::from(n);
    let small_ns_per_entity = small_ms * 1e6 / f64::from(small_n);

    // ── WORLD: the scaling shape, against this machine's own ────────────────
    //
    // Each control leg was taken beside its own subject leg above, so the ratio
    // of ratios is a single machine's answer under a single load.
    assert!(
        calib_small > 0.0
            && calib_small.is_finite()
            && calib_large > 0.0
            && calib_large.is_finite(),
        "the calibration measured {calib_small} / {calib_large} ns/entry — the \
         workload was optimized away, so neither claim below is a check"
    );

    let growth = ns_per_entity / small_ns_per_entity;
    let calib_growth = calib_large / calib_small;
    let slack = growth / calib_growth;
    eprintln!(
        "per-entity growth {small_n} -> {n}: {growth:.2}x against a calibration \
         growth of {calib_growth:.2}x ({calib_small:.1} -> {calib_large:.1} ns/entry) \
         = {slack:.2}x (margin {GROWTH_MARGIN}x)"
    );
    // The control must not invalidate itself: a BTreeMap descent cannot run
    // materially *faster* per entry at thirteen times the population, so a
    // growth below 0.8x means a calibration leg was preempted mid-measurement.
    // A garbage divisor proves nothing in either direction — the honest verdict
    // is a named skip of this half, with the raw growth printed so a real
    // regression is still visible in the log, and the CLOCK half below still
    // gets its own answer from the large leg alone.
    if calib_growth < 0.8 {
        eprintln!(
            "SKIP the WORLD half: the calibration invalidated itself \
             ({calib_small:.1} -> {calib_large:.1} ns/entry is a {calib_growth:.2}x \
             growth no descent exhibits — a leg was preempted). Raw workload growth \
             {growth:.2}x over {small_n} -> {n} entities."
        );
    } else {
        assert!(
            slack < GROWTH_MARGIN,
            "per-entity steady-state sync cost grew {growth:.2}x from {small_n} to {n} \
             entities ({small_ns_per_entity:.1} -> {ns_per_entity:.1} ns/entity) while the \
             same descent over records the same size, over the same two populations, on \
             this machine, in this process, grew {calib_growth:.2}x — {slack:.2}x more than it should, \
             against a margin of {GROWTH_MARGIN}x. Work proportional to the POPULATION \
             has been re-introduced inside the per-entity step: a `contains` over the \
             seen set, a scan per contact, a rebuild of the reverse map. (A slower \
             machine cannot fail this by being slow — the divisor is measured on it.)"
        );
    }

    // ── CLOCK: the absolute ratchet, in the calibrating machine's ns ────────
    let calib = calib_large;
    let ratio = calib / CALIBRATION_REF_NS;
    let calibrated = calib <= CALIBRATION_REF_NS * CALIBRATION_TOLERANCE;
    eprintln!(
        "calibration: {calib:.1} ns/entry vs {CALIBRATION_REF_NS} reference ({ratio:.2}x) \
         — the {CEILING_NS_PER_ENTITY} ns/entity ceiling is {}",
        if calibrated { "ASSERTED" } else { "SKIPPED" }
    );
    if !calibrated {
        eprintln!(
            "SKIP the CLOCK half: the control read {ratio:.2}x its reference — past the \
             point where a workload made of the subject's own bytes, run beside a \
             60-iteration subject leg that did not read {ratio:.2}x, is measuring this \
             machine rather than the scheduler. Measured {ns_per_entity:.1} ns/entity \
             at {n} entities."
        );
        return;
    }
    // Normalize by the calibration ratio before asserting: the SKIP text above
    // already states the law — a nanosecond on a 1.51x machine is not the
    // nanosecond the ceiling means — and the Wave-D-push red proved the binary
    // calibrated/skip gate admits machines whose noise exceeds the raw margin
    // (206.2 raw at 1.51x reddened a 200 ceiling; 136.6 normalized passes). A
    // machine FASTER than the reference divides by 1.0, never by its own speed,
    // so a fast runner cannot manufacture slack; a genuine 2x regression on the
    // 1.51x machine still normalizes to ~265 and fails.
    //
    // **And this line is only sound because the divisor is made of the same
    // bytes** (island wave I4b). With a control that stayed in cache while the
    // subject left it, `ratio` measured the machine's clock and `ns_per_entity`
    // measured its clock AND its memory system, so the division cancelled half
    // a quantity: 430.2 raw on a runner the control called 1.37x normalized to
    // 314.1 and reddened a green tree. The control's footprint is the fix, not
    // the arithmetic here, which is unchanged.
    let normalized = ns_per_entity / ratio.max(1.0);
    assert!(
        normalized < CEILING_NS_PER_ENTITY,
        "steady-state sync costs {ns_per_entity:.1} ns/entity raw, {normalized:.1} \
         calibration-normalized, at {n} entities (ceiling {CEILING_NS_PER_ENTITY}) \
         on a machine the calibration puts at {ratio:.2}x the reference; the \
         per-entity work in the reconcile has grown back"
    );
}

/// The three archetype early-outs (lens 3 P12) answer the same as the walks
/// they replaced — including in the case that makes them dangerous.
///
/// `has_component` is a claim about the archetype table, and an early-out built
/// on it is only sound if the pass it skips had nothing to do. The failure mode
/// is not "slower": it is a level whose last `PcgVolume` or last `Terrain` or
/// last 2D body was just deleted, whose colliders would then stay in the solver
/// for ever because the sweep that removes them was skipped. Each guard
/// therefore carries a second clause over its own tracked set, and this arm
/// drives exactly that transition.
#[test]
fn the_archetype_early_outs_still_reach_the_despawn_sweep() {
    use inf_ecs::components::{
        BodyKind2D, Collider2D, ColliderShape2DKind, RigidBody2D, Transform,
    };
    use inf_ecs::{EcsWorld, Vec3d};
    use inf_physics::PhysicsBridge2D;

    use glam::DVec2;

    let mut w = EcsWorld::new();
    assert!(
        !w.has_component::<RigidBody2D>(),
        "an empty world claims to carry a 2D body"
    );

    let e = w.spawn_with_guid(Uuid::from_u128(0x2D01), "Body", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(1.0, 2.0, 0.0);
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
    assert!(
        w.has_component::<RigidBody2D>(),
        "a world that just gained a 2D body says it has none — every 2D level \
         would silently stop simulating"
    );

    let mut bridge = PhysicsBridge2D::new(DVec2::new(0.0, -9.81));
    bridge.sync_from_world(&w);
    assert!(bridge.body_of(Uuid::from_u128(0x2D01)).is_some());

    // THE TRANSITION. The entity survives; only its physics components go. The
    // archetype table now says there is no 2D body anywhere — and the tracked
    // set says otherwise, which is the clause that has to win.
    w.world_mut()
        .entity_mut(e)
        .remove::<RigidBody2D>()
        .remove::<Collider2D>();
    w.mark_dirty();
    assert!(
        !w.has_component::<RigidBody2D>(),
        "the fixture did not actually empty the archetype, so this arm is vacuous"
    );
    bridge.sync_from_world(&w);
    assert!(
        bridge.body_of(Uuid::from_u128(0x2D01)).is_none(),
        "the last 2D body was deleted and its rapier body is still in the world"
    );

    // And from there the pass really is skipped: a further sync on a world with
    // no 2D archetype and no tracked body is a no-op, not a rebuild.
    bridge.sync_from_world(&w);
    assert!(bridge.body_of(Uuid::from_u128(0x2D01)).is_none());
}
