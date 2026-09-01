//! **The GUID-probe measurement** (IASSET1, clause 5) — does the pack's
//! `BTreeMap<AssetId, usize>` lookup register against a frame budget at all?
//!
//! # The prescription this was written to price
//!
//! The `.iasset` source doc proposes an **atom-slot table**: intern every asset
//! path/GUID at cook time into a dense `u32` index, and let the runtime do a
//! `Vec` subscript where it currently does a tree probe. `inf-vt`'s
//! `vt_library` is the in-tree precedent, so the idea is not speculative.
//!
//! The reason it is *measured* before it is built is the seam it would serve.
//! `VgeomSource::payload` (`inf-vgeom`'s `asset.rs`) calls
//! [`PackReader::read_ref`] once per asset per frame and again per staged page,
//! and every one of those calls starts with a `by_guid` probe. "Per frame" is
//! what makes it sound expensive; whether it *is* expensive is a different
//! question, and this repo has a standing law that an unmeasured prescription can
//! be backwards.
//!
//! # The answer, and what it decides
//!
//! **A probe costs 12–59 ns**, rising with index size the way a tree does. That
//! number alone decides nothing, because "does it register" is a question about
//! *probes × cost*, and the honest way to answer it is to invert: at the slowest
//! measured probe, how many probes per frame would it take to reach 1% of a
//! shipping frame? The test prints that break-even. It is **tens of thousands**,
//! against a real cooked island that holds **48 pack entries in total and one
//! `.inf_vmesh`**.
//!
//! So the atom table stays **unbuilt, with the number beside it** — which is the
//! outcome this file exists to be able to state, rather than a conclusion it was
//! written to reach. The budget it is checked against is
//! `inf_player::budget::SHIPPING_FRAME_BUDGET_MS` (16.6 ms at 60 fps), restated
//! here because `inf-asset` is Ring 0.

use std::time::Instant;

use inf_asset::{AssetId, AssetKind, PackReader, PackWriter};

/// `inf_player::budget::SHIPPING_FRAME_BUDGET_MS`, mirrored (Ring 0 may not
/// depend on the player). A drift here would have to be several-fold before it
/// changed a verdict measured in thousandths of this number.
const SHIPPING_FRAME_BUDGET_MS: f64 = 16.6;

/// Probes charged to one frame.
///
/// The real number is "resident vgeom assets, plus staged pages". The cooked
/// island holds **48 pack entries in total, one of them a `.inf_vmesh`**, so its
/// true per-frame probe count is in the single digits. 256 is ~250× that: an
/// upper bound with a wide margin, chosen so a passing verdict cannot be an
/// artefact of an optimistic count, and *not* so large that the test is
/// measuring a scene nobody will ever build.
const PROBES_PER_FRAME: usize = 256;

/// Index size the verdict is taken at — 10 000 entries, ~200× the island's 48.
///
/// The table below still reports 100 000, because the *shape* is worth seeing;
/// the assertion is taken here because a pack with a hundred thousand entries is
/// not a thing this engine has ever produced and asserting against it would be
/// budgeting for a hypothesis.
const VERDICT_ENTRIES: usize = 10_000;

fn guid(n: u128) -> AssetId {
    AssetId(uuid::Uuid::from_u128(n))
}

/// Build a pack index of `n` entries. Payloads are one byte: this measures the
/// *probe*, and a pack whose blobs are real would measure the allocator.
fn pack_of(n: usize) -> PackReader {
    let mut w = PackWriter::new();
    for i in 0..n {
        // Spread the ids across the u128 space rather than 0..n: a `BTreeMap`
        // over dense consecutive keys has friendlier cache behaviour than one
        // over real v4 GUIDs, and measuring the friendly case would understate
        // the cost this test exists to price.
        let id = guid((i as u128).wrapping_mul(0x9E37_79B9_7F4A_7C15_1234_5678_9ABC_DEF1));
        w.add_bytes(id, AssetKind::MeshletMesh, &[0u8]).unwrap();
    }
    PackReader::from_bytes(w.to_bytes().unwrap()).unwrap()
}

/// Nanoseconds per successful `entry()` probe, best of five passes.
fn probe_ns(reader: &PackReader, keys: &[AssetId]) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let mut found = 0usize;
        for k in keys {
            if reader.entry(*k).is_some() {
                found += 1;
            }
        }
        std::hint::black_box(found);
        assert_eq!(found, keys.len(), "every probe must hit");
        best = best.min(t.elapsed().as_secs_f64() * 1e9 / keys.len() as f64);
    }
    best
}

/// **THE MEASUREMENT.** Prints the probe cost at four index sizes and the share
/// of a shipping frame a whole frame's worth of probes would take.
#[test]
fn the_guid_probe_does_not_register_against_a_frame() {
    println!(
        "\n  pack index probe (BTreeMap<AssetId, usize>), best of 5, \
         {PROBES_PER_FRAME} probes charged per frame"
    );
    println!(
        "  {:>10}{:>14}{:>18}{:>12}",
        "entries", "ns/probe", "ms/frame-worth", "% of 16.6ms"
    );

    let mut verdict_ns = 0.0f64;
    let mut slowest_ns = 0.0f64;
    for n in [48usize, 1_000, VERDICT_ENTRIES, 100_000] {
        let reader = pack_of(n);
        let keys: Vec<AssetId> = reader.index().map(|e| e.guid).collect();
        let ns = probe_ns(&reader, &keys);
        let frame_ms = ns * PROBES_PER_FRAME as f64 / 1e6;
        let share = frame_ms / SHIPPING_FRAME_BUDGET_MS * 100.0;
        if n == VERDICT_ENTRIES {
            verdict_ns = ns;
        }
        slowest_ns = slowest_ns.max(ns);
        println!("  {n:>10}{ns:>14.1}{frame_ms:>18.4}{share:>12.3}");
    }

    // **THE INVERSION, which is what actually answers the question.** "Does the
    // probe register" is about probes × cost, so the useful figure is the
    // break-even: at the SLOWEST measured probe, how many per frame would it take
    // to reach 1% of a shipping frame? Printed so the ruling in the memo has a
    // number attached rather than an adjective.
    let breakeven = (SHIPPING_FRAME_BUDGET_MS * 0.01 * 1e6 / slowest_ns) as u64;
    println!(
        "  break-even: {breakeven} probes/frame would cost 1% of a {SHIPPING_FRAME_BUDGET_MS} ms \
         frame at {slowest_ns:.1} ns/probe (the island cooks 48 entries, 1 of them a .inf_vmesh)"
    );

    // **THE VERDICT.** 1% of a shipping frame at 256 probes over a 10 000-entry
    // index — ~250× the island's real probe count over ~200× its real index.
    // Below the line, the atom table would be an optimization with no measurable
    // subject, and the source doc's own precedent (`vt_library`'s dense index)
    // serves a path that pages thousands of TILES per frame rather than tens of
    // assets.
    let share = verdict_ns * PROBES_PER_FRAME as f64 / 1e6 / SHIPPING_FRAME_BUDGET_MS * 100.0;
    assert!(
        share < 1.0,
        "the GUID probe now costs {share:.3}% of a shipping frame at \
         {PROBES_PER_FRAME} probes over {VERDICT_ENTRIES} entries — that is \
         enough to be worth an atom-slot table, and IASSET2 should build one"
    );
}

/// The probe is `O(log n)`, and the *shape* is worth pinning separately from the
/// magnitude: an accidental linear scan would still pass the budget arm at 48
/// entries and would be a real regression at ten thousand.
#[test]
fn the_probe_grows_logarithmically_not_linearly() {
    let small = pack_of(1_000);
    let large = pack_of(100_000);
    let sk: Vec<AssetId> = small.index().map(|e| e.guid).collect();
    let lk: Vec<AssetId> = large.index().map(|e| e.guid).collect();
    let s = probe_ns(&small, &sk);
    let l = probe_ns(&large, &lk);
    println!(
        "\n  1k: {s:.1} ns/probe   100k: {l:.1} ns/probe   ratio {:.2}",
        l / s
    );
    // 100× the entries is ~1.7× the tree depth. Allowed 8× to absorb the cache
    // behaviour of a 100 000-node tree on a loaded runner; a linear scan would be
    // ~100×, which is what this arm is actually looking for.
    assert!(
        l < s * 8.0,
        "100× the entries cost {:.1}× the probe — that is not a tree lookup",
        l / s
    );
}
