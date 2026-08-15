//! **The committed camera is a function of the world, not of the process**
//! (Hardening Wave C, L6.F1).
//!
//! `inf_math::predict`'s module header names `RuntimeSim::camera_focus` by hand
//! as its worked example of a *committed* pose — "a fold of actor positions,
//! which is a pure function of the committed input — it commits". That claim was
//! false: the fold summed `f64` over a `HashMap`'s values, and `HashMap`
//! iteration order is a function of a per-process random seed. `f64` addition is
//! not associative, so the camera's low bits were a function of *which process
//! was running*, and two consumers read those bits:
//!
//! * the shipped player's terrain-residency cut (`terrain_stream`'s want set),
//!   which a phase gate compares between hosts, and
//! * [`inf_math::dead_reckon`], whose whole module is pinned against libm for
//!   exactly this class of divergence.
//!
//! # Why the arm lives here and not beside the pin it belongs to
//!
//! `inf-math`'s purity pin (`the_predictor_reads_no_clock_and_holds_no_state`)
//! reads `predict.rs` and nothing else, and it *cannot* be extended to reach
//! this: `inf-player` depends on `inf-math`, not the other way round, so no test
//! in that crate can see this source at all. The pin is structurally blind, and
//! the answer is an arm on the other side of the dependency edge — here, beside
//! the fold.
//!
//! # Two arms, because one process cannot see the defect
//!
//! The failure is *cross-process* by construction: within one process the seed
//! is fixed, so a value assertion can be green while the property is false. That
//! is the P28.4 finding in miniature (a clock scaled to `1e-9` passed all 59
//! arms of `inf-math`). So:
//!
//! * `the_committed_camera_folds_in_guid_order` pins the ORDER against two
//!   independent sums, over terms chosen so ascending and descending really do
//!   disagree — it can tell orders apart, which a container check alone cannot;
//! * `the_position_maps_are_ordered_containers` pins the SOURCE, which is the
//!   only thing that can see a seed.

use std::collections::BTreeMap;

use glam::DVec3;
use inf_player::runtime_sim::camera_centroid;
use uuid::Uuid;

/// Three x-values whose `f64` sum genuinely depends on the order it is taken in.
///
/// `1e16` has an ulp of 2, so `1.0 + 1e16` rounds back to `1e16` and the `1.0`
/// is gone before the `-1e16` arrives; taken the other way the two large terms
/// cancel first and the `1.0` survives. Ascending answers `0`, descending
/// answers `1`. Without a spread like this the "order matters" arms below would
/// be vacuously green on any order at all.
const ORDER_SENSITIVE_X: [f64; 3] = [1.0, 1.0e16, -1.0e16];

fn guid(n: u128) -> Uuid {
    Uuid::from_u128(n)
}

/// A position set whose ascending-`Guid` order is exactly [`ORDER_SENSITIVE_X`].
fn sample_positions() -> Vec<(Uuid, DVec3)> {
    ORDER_SENSITIVE_X
        .iter()
        .enumerate()
        .map(|(i, x)| (guid(i as u128 + 1), DVec3::new(*x, 0.0, 0.0)))
        .collect()
}

fn bits(v: DVec3) -> (u64, u64, u64) {
    (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
}

/// The fold visits positions in ascending `Guid` order — pinned against an
/// independent sum, and shown to be able to tell that order from its reverse.
#[test]
fn the_committed_camera_folds_in_guid_order() {
    let sample = sample_positions();

    // The container normalizes insertion order: the same content inserted
    // forwards and backwards is the same map and therefore the same answer.
    let forward: BTreeMap<Uuid, DVec3> = sample.iter().copied().collect();
    let backward: BTreeMap<Uuid, DVec3> = sample.iter().rev().copied().collect();
    assert_eq!(
        bits(camera_centroid(&forward)),
        bits(camera_centroid(&backward)),
        "the camera fold answers differently depending on the order positions \
         were inserted in — the container is not normalizing"
    );

    // …and the order it settles on is ASCENDING, checked against a sum this test
    // takes for itself rather than against the code under test.
    let n = sample.len() as f64;
    let mut asc = sample.clone();
    asc.sort_by_key(|(g, _)| *g);
    let asc_sum: DVec3 = asc.iter().map(|(_, p)| *p).sum::<DVec3>() / n;
    assert_eq!(
        bits(camera_centroid(&forward)),
        bits(asc_sum),
        "the camera fold is not the ascending-Guid sum"
    );

    // **NOT VACUOUS**: with these terms the descending sum is a different
    // number, so the assertion above genuinely distinguishes one order from
    // another. If this ever stops holding the arm above has stopped meaning
    // anything and this line says so first.
    let desc_sum: DVec3 = asc.iter().rev().map(|(_, p)| *p).sum::<DVec3>() / n;
    assert_ne!(
        bits(asc_sum),
        bits(desc_sum),
        "ORDER_SENSITIVE_X no longer produces an order-dependent sum, so the \
         arms above would pass for any fold order at all"
    );

    // The empty case is a value, not a division by zero.
    assert_eq!(camera_centroid(&BTreeMap::new()), DVec3::ZERO);
}

/// **The source pin**: the two position maps are ordered containers.
///
/// A value arm cannot see this. `HashMap`'s seed is drawn once per process, so
/// two runs in one test binary agree and two runs on two machines need not —
/// which is precisely the P14 shape, and precisely what
/// `the_predictor_reads_no_clock_and_holds_no_state` was written for one crate
/// over. Built to falsify: the same predicate over a poisoned copy rejects it.
#[test]
fn the_position_maps_are_ordered_containers() {
    // Normalized: `core.autocrlf = true` checks `.rs` out CRLF on Windows, and a
    // newline-delimited search over CRLF source finds nothing (the P22 law).
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/runtime_sim.rs"),
    )
    .expect("the sim module is readable")
    .replace("\r\n", "\n");

    let declares = |field: &str| -> bool {
        src.contains(&format!("{field}: BTreeMap<Uuid, DVec3>,"))
            && !src.contains(&format!("{field}: HashMap<Uuid, DVec3>"))
    };
    for field in ["prev_positions", "cur_positions"] {
        assert!(
            declares(field),
            "`{field}` is not a `BTreeMap<Uuid, DVec3>` — a hashed container's \
             iteration order is seeded per process, and the camera fold over it \
             is a non-associative f64 sum (P14 / L6.F1)"
        );
    }

    // The fold really is over the ordered map, and it really is this crate's.
    assert!(
        src.contains("pub fn camera_centroid(positions: &BTreeMap<Uuid, DVec3>) -> DVec3"),
        "`camera_centroid` no longer takes the ordered map"
    );
    assert!(
        src.contains("camera_centroid(&self.cur_positions)"),
        "`camera_focus` no longer folds through `camera_centroid`"
    );

    // **Built to falsify** (the P22 law): the predicate must reject the source it
    // was written against, or it is a statement about a string that happens to
    // be green.
    let poisoned = src.replace(
        "cur_positions: BTreeMap<Uuid, DVec3>,",
        "cur_positions: HashMap<Uuid, DVec3>,",
    );
    assert!(
        !poisoned.contains("cur_positions: BTreeMap<Uuid, DVec3>,")
            && poisoned.contains("cur_positions: HashMap<Uuid, DVec3>"),
        "the pin cannot see a hashed position map even when one is put in front \
         of it"
    );
}
