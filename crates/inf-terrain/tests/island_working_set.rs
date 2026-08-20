//! **IB-9 — the terrain working set, measured, and both budgets derived from it.**
//!
//! The AAA-readiness certification's finding: `TERRAIN_RESIDENT_BYTES_CEILING`
//! is 16 MiB and `StreamBudget::default().max_resident_tiles` was 1 024, which at
//! a 257² page is ≈ 264 MiB — *"16.4× apart, and the gate scene never approaches
//! either, so the two have never had to agree. At island scale the ratchet fires
//! first, and whoever is holding it will not know which number is the real one."*
//!
//! # The shape of the answer
//!
//! Neither number was ever derived, and the reason they could disagree is that
//! they were two independent guesses at one quantity: **how many pages a render
//! cut holds**. That is not a property of the world — a quadtree cut is a
//! camera-local ring structure, so its page count is a function of the *ladder*
//! (`RENDER_LOD0_RADIUS_TILES`, the hysteresis, the level count) and of nothing
//! else. It does not grow with the terrain.
//!
//! So this file measures the cut over a fast flythrough, at four world sizes and
//! at two page resolutions, and shows the peak grows by a **constant ring per LOD
//! level** while the catalog quadruples — `O(levels)`, not `O(pages)`.
//! `inf_terrain::stream::cut_page_bound` states that count as a rule, and
//! `StreamBudget::default()` and `resident_bytes_bound` are both read off it — so
//! the two numbers the certification found 16.4× apart are one number read twice,
//! and cannot drift again.
//!
//! # Why the working set is measured in PAGES
//!
//! A page is `tile_resolution² × 4` bytes whatever the world is, so pages are the
//! grid-independent quantity and bytes follow from them by multiplication. That
//! is also what makes the always-on arm affordable: an island-class **topology**
//! — 32 × 32 level-0 pages over 5 LOD levels, the certification's own 8192²
//! sample / 1 364-page row — costs a few MB at a 33² page and 360 MB at a 257²
//! one. The topology is what the cut sees;
//! `the_island_page_count_holds_at_a_shipping_page_size` runs the 257² version
//! under `#[ignore]` so the byte figure is reproducible on demand without putting
//! 360 MB into every battery on three runners.

use std::collections::BTreeSet;

use glam::{DVec2, DVec3};
use inf_terrain::residency::MemoryTileStore;
use inf_terrain::stream::{
    cut_page_bound, resident_bytes_bound, StreamBudget, TerrainStreamer, RENDER_LOD0_RADIUS_TILES,
};
use inf_terrain::wants::{RenderWantsParams, TileCatalog, TileGrid, TileIndex};
use inf_terrain::{TerrainTile, TileKey};

/// A store with `side × side` level-0 pages and a full pyramid down to ≤ 4 pages,
/// every sample flat.
///
/// Flat because the working set is a function of the camera and the ladder, never
/// of the heights — and a constant field is the cheapest way to say so.
fn island(side: i32, res: u32) -> (MemoryTileStore, u32) {
    let mut store = MemoryTileStore::new();
    let mut level = side;
    let mut lod = 0u32;
    loop {
        for tz in 0..level {
            for tx in 0..level {
                store
                    .insert(
                        TileKey {
                            lod,
                            coord: (tx, tz),
                        },
                        &TerrainTile::flat(res, DVec3::ZERO),
                    )
                    .expect("a fresh key encodes");
            }
        }
        if (level * level) as usize <= inf_terrain::pyramid::DEFAULT_MIN_PYRAMID_TILES {
            break;
        }
        level = (level + 1) / 2;
        lod += 1;
        if lod >= inf_terrain::pyramid::DEFAULT_MAX_PYRAMID_LEVELS {
            break;
        }
    }
    (store, lod + 1)
}

/// What a flythrough left resident at its worst.
struct Peak {
    pages: usize,
    bytes: u64,
    cuts: usize,
    levels: u32,
    catalog: usize,
}

/// Fly a diagonal across the world at a stride wider than a level-0 page, so the
/// cut is re-cut on nearly every sync — *at speed*, because a working set
/// measured while parked is a working set nobody has.
fn flythrough(
    store: &MemoryTileStore,
    side: i32,
    res: u32,
    mps: f64,
    budget: StreamBudget,
) -> Peak {
    let catalog = TileCatalog::from_store(store);
    let levels = TileIndex::max_lod(&catalog) + 1;
    let catalog_len = catalog.len();
    let grid = TileGrid::new(res, mps);
    let params =
        RenderWantsParams::geometric(RENDER_LOD0_RADIUS_TILES * grid.level0_span(), levels);
    let mut streamer = TerrainStreamer::new(grid, catalog, params, budget, res, mps);

    // **Each world's OWN diagonal.** The first version of this harness flew a
    // fixed 32-span distance whatever the world was, so the 16x16 row spent half
    // its path OUTSIDE its own terrain and reported a cut that was falling off an
    // edge. A working set measured over ground that is not there is not a working
    // set.
    let span = grid.level0_span();
    let extent = span * f64::from(side);
    const STEPS: u32 = 64;
    let mut peak = Peak {
        pages: 0,
        bytes: 0,
        cuts: 0,
        levels,
        catalog: catalog_len,
    };
    let mut seen: BTreeSet<Vec<TileKey>> = BTreeSet::new();
    for s in 0..=STEPS {
        let t = f64::from(s) / f64::from(STEPS);
        let at = DVec2::new(t * extent, t * extent);
        streamer.sync_render(at, store);
        let cut: Vec<TileKey> = streamer.cut().iter().copied().collect();
        peak.pages = peak.pages.max(cut.len());
        peak.bytes = peak.bytes.max(streamer.stats().bytes_resident);
        seen.insert(cut);
    }
    peak.cuts = seen.len();
    peak
}

/// A budget that clamps nothing — a clamped cut measures the clamp rather than
/// the cut, which is exactly the circularity IB-9 is about.
fn unclamped() -> StreamBudget {
    StreamBudget {
        max_resident_tiles: 0,
        max_loads_per_sync: usize::MAX,
    }
}

/// **The measurement, and the derivation both budgets now come from.**
#[test]
fn the_render_cut_is_camera_local_and_both_budgets_derive_from_it() {
    println!(
        "IB-9 render cut, ladder = {RENDER_LOD0_RADIUS_TILES} level-0 spans, \
         hysteresis {}:",
        inf_terrain::wants::DEFAULT_HYSTERESIS
    );
    let mut peaks = Vec::new();
    for (side, res, mps, label) in [
        (8, 33, 8.0, "8x8 pages (clips)"),
        (16, 33, 8.0, "16x16 pages"),
        (32, 33, 8.0, "32x32 pages (island class)"),
        (64, 33, 8.0, "64x64 pages"),
        (32, 65, 4.0, "32x32 pages, 65^2 page"),
    ] {
        let (store, _) = island(side, res);
        let p = flythrough(&store, side, res, mps, unclamped());
        println!(
            "  {label:<28} catalog {:>5} pages / {} levels -> PEAK CUT {:>4} pages, \
             {:>9} B resident, {} distinct cuts",
            p.catalog, p.levels, p.pages, p.bytes, p.cuts
        );
        peaks.push((label, p));
    }

    let deepest_levels = peaks.iter().map(|(_, p)| p.levels).max().unwrap_or(1);
    let bound = cut_page_bound(deepest_levels, RENDER_LOD0_RADIUS_TILES);
    let budget = StreamBudget::default();
    println!("  cut_page_bound({deepest_levels} levels) = {bound} pages");
    println!(
        "  StreamBudget::default().max_resident_tiles = {} pages \
         (= cut_page_bound({}), the deepest pyramid this engine imports)",
        budget.max_resident_tiles,
        inf_terrain::stream::DEFAULT_LADDER_LEVELS
    );
    for res in [33u32, 129, 257] {
        println!(
            "  resident_bytes_bound(budget, {res}^2 page) = {} B ({:.2} MiB)",
            resident_bytes_bound(budget.max_resident_tiles, res),
            resident_bytes_bound(budget.max_resident_tiles, res) as f64 / (1024.0 * 1024.0)
        );
    }

    // (1) Every measured peak is inside the rule for its own ladder depth.
    for (label, p) in &peaks {
        let b = cut_page_bound(p.levels, RENDER_LOD0_RADIUS_TILES);
        assert!(
            p.pages <= b,
            "{label}: the cut peaked at {} pages, over the {b} the rule allows \
             for {} levels — the rule is wrong or the cut grew",
            p.pages,
            p.levels
        );
    }

    // (2) The rule is TIGHT. A bound nothing approaches is a bound that cannot
    //     fail, and "the gate scene never approaches either" is half of what IB-9
    //     says is wrong with the pair this replaces.
    let deepest = peaks
        .iter()
        .max_by_key(|(_, p)| p.levels)
        .expect("four rows");
    let b = cut_page_bound(deepest.1.levels, RENDER_LOD0_RADIUS_TILES);
    assert!(
        deepest.1.pages * 2 >= b,
        // `deepest` is the row with the most LEVELS, which is the 64 x 64 world,
        // not the island-class 32 x 32 one — the message said "island-class"
        // until the I4 audit read it against the printed table.
        "the deepest measured cut ({} levels) peaked at {} pages against a \
         {b}-page rule — the rule is more than twice the thing it bounds, so \
         nothing can ever approach it",
        deepest.1.levels,
        deepest.1.pages
    );

    // (3) **THE LAW, MEASURED.** The cut is a camera-local ring structure, so a
    //     level contributes the same ring of pages however large the world is:
    //     the cut is O(LEVELS), not O(pages). Each row below quadruples the
    //     catalog; the cut must grow by a CONSTANT.
    //
    //     This is the claim the whole derivation rests on, and it is the one the
    //     first draft of this file got wrong — it asserted the peak was
    //     *identical* across world sizes, which is false: the peak grows, just
    //     logarithmically.
    let unclipped: Vec<(usize, usize)> = peaks[1..4]
        .iter()
        .map(|(_, p)| (p.catalog, p.pages))
        .collect();
    let steps: Vec<isize> = unclipped
        .windows(2)
        .map(|w| w[1].1 as isize - w[0].1 as isize)
        .collect();
    println!("  per-level ring: {steps:?} pages (catalog x4 per step)");
    for w in unclipped.windows(2) {
        // A pyramid over `n` level-0 pages holds `n + n/4 + n/16 + …`, so
        // quadrupling the world quadruples the catalog and adds the old
        // catalog's own top — 84 -> 340 -> 1364 -> 5460, which is `4n + 4`
        // rather than `4n` exactly.
        assert!(
            w[1].0 >= w[0].0 * 4,
            "each row must at least quadruple the catalog ({} -> {}), or the growth below is not being measured against a growing world",
            w[0].0,
            w[1].0
        );
    }
    let mean = steps.iter().sum::<isize>() as f64 / steps.len() as f64;
    for s in &steps {
        assert!(
            (*s as f64 - mean).abs() <= mean * 0.1,
            "the per-level ring is not constant: {steps:?} pages while the catalog quadrupled each step. The cut would then be a function of the world's SIZE, and no budget could be derived from the ladder."
        );
    }
    assert!(
        mean > 0.0,
        "the cut did not grow at all with the ladder — the rows are measuring one world"
    );
    // …and the growth really is logarithmic rather than proportional: four times
    // the pages must not be anywhere near four times the cut.
    let (first, last) = (unclipped[0].1, unclipped[unclipped.len() - 1].1);
    assert!(
        (last as f64) < first as f64 * 2.5,
        "the cut went from {first} to {last} pages while the catalog went up 16x — that is not logarithmic growth"
    );
    assert!(
        peaks[0].1.pages < unclipped[0].1,
        "the 8x8 world did not clip its own cut ({} against {}), so the ring above is being measured on worlds that may all be too small to fill it",
        peaks[0].1.pages,
        unclipped[0].1
    );

    // (4) …and the cut does not depend on the PAGE SIZE, which is what lets the
    //     always-on arm measure a cheap grid and the bytes be a multiplication.
    assert_eq!(
        peaks[4].1.pages, peaks[2].1.pages,
        "the peak cut changed with the page resolution ({} at 65^2 against {} at 33^2) — pages would not be the grid-independent unit and `resident_bytes_bound` could not be a multiplication",
        peaks[4].1.pages,
        peaks[2].1.pages
    );

    // (5) Anti-vacuity: the flythrough really flew. A run that published one cut
    //     measured a parked camera.
    assert!(
        deepest.1.cuts >= 8,
        "the flythrough published only {} distinct cuts — it did not move",
        deepest.1.cuts
    );

    // (6) And the two budgets AGREE, which is the finding closed: the default
    //     budget must be able to hold the deepest cut this engine can build, and
    //     must not be an order of magnitude past it.
    assert!(
        budget.max_resident_tiles >= bound,
        "the default budget ({} pages) cannot hold the deepest cut ({bound})",
        budget.max_resident_tiles
    );
    assert!(
        budget.max_resident_tiles <= bound * 4,
        "the default budget ({} pages) is more than 4x the deepest cut \
         ({bound}) — that is the 16.4x IB-9 found, wearing a smaller number",
        budget.max_resident_tiles
    );
}

/// **The shipping page size, and the byte figure it makes** — `#[ignore]`d because
/// it builds ≈ 360 MB of pages.
///
/// The always-on arm measures the island's page **topology** on a cheap grid;
/// this is the arm that says the same topology at the certification's own 257²
/// page really does cost what `resident_bytes_bound` multiplies out, over a real
/// store.
///
/// `cargo test -p inf-terrain --test island_working_set -- --ignored`
#[test]
#[ignore = "builds ~360 MB of pages: the shipping-page-size confirmation"]
fn the_island_page_count_holds_at_a_shipping_page_size() {
    let (store, _) = island(32, 257);
    let p = flythrough(&store, 32, 257, 1.0, unclamped());
    let bound = cut_page_bound(p.levels, RENDER_LOD0_RADIUS_TILES);
    println!(
        "IB-9 island class at 257^2: catalog {} pages / {} levels -> PEAK CUT {} pages, \
         {} B ({:.2} MiB) resident; bound {bound} pages = {} B ({:.2} MiB)",
        p.catalog,
        p.levels,
        p.pages,
        p.bytes,
        p.bytes as f64 / (1024.0 * 1024.0),
        resident_bytes_bound(bound, 257),
        resident_bytes_bound(bound, 257) as f64 / (1024.0 * 1024.0),
    );
    assert!(p.pages <= bound);
    assert!(
        p.bytes <= resident_bytes_bound(bound, 257),
        "the measured bytes {} exceed the derived bound {}",
        p.bytes,
        resident_bytes_bound(bound, 257)
    );
}
