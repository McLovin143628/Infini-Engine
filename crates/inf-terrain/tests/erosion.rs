//! Hydraulic + thermal erosion behaviour + mass accounting (P10.3a).
//!
//! These test the *algorithm*, not just plumbing: channels form, mass is
//! accounted, water is conserved in a closed box, thermal relaxes to the talus
//! angle, deposition raises a basin, holes are walls, and the whole thing is
//! byte-for-byte deterministic.

use glam::DVec2;
use inf_terrain::{
    erode, erode_terrain, erode_with, DataMapKind, ErosionParams, ErosionStats, HeightRegion,
    TerrainData,
};
use xxhash_rust::xxh3::Xxh3;

// ── helpers ──────────────────────────────────────────────────────────────────

/// A single tile authored from a world-height function, at metres-per-sample 1.
fn terrain_from<F: FnMut(f64, f64) -> f64>(res: u32, f: F) -> TerrainData {
    let mut t = TerrainData::new(res, 1.0);
    t.author_tile((0, 0), f);
    t
}

/// A radial cone of height `peak` centred in a `res`-sample tile, floored at 0.
fn cone(res: u32, peak: f64) -> TerrainData {
    let c = (res as f64 - 1.0) / 2.0;
    terrain_from(res, move |x, z| {
        let d = ((x - c).powi(2) + (z - c).powi(2)).sqrt();
        (peak * (1.0 - d / c)).max(0.0)
    })
}

/// xxh3-128 over a region's raw `f32` heights — byte identity of the field.
fn hash_region(r: &HeightRegion) -> u128 {
    let mut h = Xxh3::new();
    let (nx, nz) = r.dims();
    h.update(&nx.to_le_bytes());
    h.update(&nz.to_le_bytes());
    for v in r.heights() {
        h.update(&v.to_le_bytes());
    }
    h.digest128()
}

/// xxh3-128 over a whole terrain's tiles (mirrors the brush test's helper).
/// Covers the **erosion data maps** too (P19.1), so "byte-identical terrain"
/// after an undo means both layers, not just the heights.
fn hash_terrain(t: &TerrainData) -> u128 {
    let mut h = Xxh3::new();
    h.update(&(t.tile_count() as u64).to_le_bytes());
    for (&(x, z), tile) in t.tiles() {
        h.update(&x.to_le_bytes());
        h.update(&z.to_le_bytes());
        h.update(&tile.origin.y.to_le_bytes());
        for &v in tile.heights() {
            h.update(&v.to_le_bytes());
        }
        h.update(&(tile.maps_len() as u64).to_le_bytes());
        for texel in tile.maps() {
            for v in texel {
                h.update(&v.to_le_bytes());
            }
        }
    }
    h.digest128()
}

/// xxh3-128 over a region's raw data-map buffer — byte identity of all three
/// accumulators (P19.1).
fn hash_maps(r: &HeightRegion) -> u128 {
    let mut h = Xxh3::new();
    let (nx, nz) = r.dims();
    h.update(&nx.to_le_bytes());
    h.update(&nz.to_le_bytes());
    for texel in r.maps() {
        for v in texel {
            h.update(&v.to_le_bytes());
        }
    }
    h.digest128()
}

/// Sum one data-map channel over a region's authored cells.
fn map_total(r: &HeightRegion, kind: DataMapKind) -> f64 {
    let (nx, nz) = r.dims();
    let mut s = 0.0f64;
    for z in 0..nz {
        for x in 0..nx {
            if r.is_authored(x, z) {
                s += r.map(x, z, kind) as f64;
            }
        }
    }
    s
}

/// Max terrain slope (rise/run) over all authored cells and their 8-neighbours.
fn max_slope(r: &HeightRegion) -> f32 {
    let (nx, nz) = r.dims();
    let sqrt2 = std::f32::consts::SQRT_2;
    let mut m = 0.0f32;
    for z in 0..nz {
        for x in 0..nx {
            if !r.is_authored(x, z) {
                continue;
            }
            let h = r.height(x, z);
            for (dx, dz) in [
                (-1i32, 0i32),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
                (1, 1),
            ] {
                let nc = x as i32 + dx;
                let nr = z as i32 + dz;
                if nc < 0 || nr < 0 || nc >= nx as i32 || nr >= nz as i32 {
                    continue;
                }
                let (nc, nr) = (nc as u32, nr as u32);
                if !r.is_authored(nc, nr) {
                    continue;
                }
                let dist = if dx == 0 || dz == 0 { 1.0 } else { sqrt2 };
                let slope = (h - r.height(nc, nr)).abs() / dist;
                m = m.max(slope);
            }
        }
    }
    m
}

/// Σ of authored-cell height differences between two same-shaped regions.
fn height_sum_delta(before: &[f32], after: &[f32]) -> f64 {
    before
        .iter()
        .zip(after)
        .map(|(&a, &b)| (b - a) as f64)
        .sum()
}

// ── determinism ──────────────────────────────────────────────────────────────

#[test]
fn erosion_is_byte_identical_across_runs() {
    let t = cone(33, 40.0);
    let mut a = t.extract_region(DVec2::new(0.0, 0.0), DVec2::new(32.0, 32.0), 0);
    let mut b = t.extract_region(DVec2::new(0.0, 0.0), DVec2::new(32.0, 32.0), 0);
    // Exercise the seeded rain-variation fBm path too (still deterministic).
    let params = ErosionParams {
        rain_variation: 0.4,
        rain_seed: 99,
        ..Default::default()
    };
    let sa = erode(&mut a, &params, 120);
    let sb = erode(&mut b, &params, 120);
    assert_eq!(
        hash_region(&a),
        hash_region(&b),
        "erosion must be reproducible"
    );
    assert_eq!(sa, sb, "stats must be reproducible");
}

// ── params serde ─────────────────────────────────────────────────────────────

#[test]
fn params_serde_round_trip() {
    let p = ErosionParams {
        dt: 0.017,
        rain_rate: 0.033,
        rain_variation: 0.25,
        thermal_talus_deg: 41.0,
        thermal_every: 4,
        ..Default::default()
    };
    let json = serde_json::to_string(&p).unwrap();
    let back: ErosionParams = serde_json::from_str(&json).unwrap();
    assert_eq!(p, back);
}

// ── hydraulic erosion + mass accounting ──────────────────────────────────────

#[test]
fn single_peak_carves_channels_and_lowers_with_accounted_mass() {
    let t = cone(41, 60.0);
    let mut region = t.extract_region(DVec2::new(0.0, 0.0), DVec2::new(40.0, 40.0), 0);
    let before = region.heights().to_vec();

    let params = ErosionParams::default();
    let stats = erode(&mut region, &params, 500);
    let after = region.heights().to_vec();

    // The peak lowered.
    let peak_before = before[(20 * 41 + 20) as usize];
    let peak_after = after[(20 * 41 + 20) as usize];
    assert!(
        peak_after < peak_before,
        "peak should erode down: {peak_before} -> {peak_after}"
    );

    // Erosion happened and the surface net-eroded.
    assert!(stats.sediment_moved > 0.0, "some terrain must dissolve");
    assert!(stats.mass_delta < 0.0, "net surface should lose material");

    // The stats' mass_delta equals an independent measurement of the terrain
    // change (area = 1 m²), proving the accounting is faithful.
    let measured = height_sum_delta(&before, &after); // × l² = ×1
    assert!(
        (measured - stats.mass_delta).abs() <= 1e-3 * stats.mass_delta.abs().max(1.0),
        "measured {measured} vs stats {}",
        stats.mass_delta
    );

    // Channels: many off-peak cells changed height (not a uniform sink).
    let changed = before
        .iter()
        .zip(&after)
        .filter(|(a, b)| (**a - **b).abs() > 1e-3)
        .count();
    assert!(
        changed > 50,
        "erosion should reshape a wide area: {changed}"
    );
}

// ── closed-box water conservation ────────────────────────────────────────────

#[test]
fn closed_box_conserves_water() {
    // A hole ring (margin) walls the authored tile → a closed box.
    let t = terrain_from(16, |_, _| 0.0);
    let mut region = t.extract_region(DVec2::new(0.0, 0.0), DVec2::new(15.0, 15.0), 2);

    // Rain on, no evaporation, no thermal: water can only move, never leave.
    let params = ErosionParams {
        rain_rate: 0.02,
        evaporation: 0.0,
        thermal_rate: 0.0,
        ..Default::default()
    };
    let stats = erode(&mut region, &params, 80);

    assert!(stats.water_in > 0.0, "rain should add water");
    assert!(
        stats.water_out < 1e-4,
        "no water may leave a hole-walled box: {}",
        stats.water_out
    );
    // Every drop that fell is still present (flux clamp keeps d ≥ 0, no loss).
    let tol = 1e-3 * stats.water_in;
    assert!(
        (stats.water_present - stats.water_in).abs() <= tol,
        "water not conserved: in={} present={}",
        stats.water_in,
        stats.water_present
    );
}

// ── thermal relaxation ───────────────────────────────────────────────────────

#[test]
fn thermal_relaxes_oversteep_cliff_toward_talus() {
    // An over-steep cliff (a step), max slope ≫ tan(talus). A step relaxes into a
    // talus ramp with a cleanly non-increasing global max slope (unlike a singular
    // cone tip, whose discrete ring artifact can transiently nudge the max).
    let t = terrain_from(33, |x, _z| if x < 16.0 { 60.0 } else { 0.0 });
    let mut region = t.extract_region(DVec2::new(0.0, 0.0), DVec2::new(32.0, 32.0), 0);
    let before = region.heights().to_vec();
    let slope0 = max_slope(&region);

    // Isolate thermal: zero rain (⇒ hydraulic passes are inert).
    let params = ErosionParams {
        rain_rate: 0.0,
        thermal_talus_deg: 33.0,
        thermal_rate: 0.5,
        ..Default::default()
    };

    let mut prev = f32::INFINITY;
    let mut steps = 0u32;
    let stats = erode_with(&mut region, &params, 150, |_, r| {
        let s = max_slope(r);
        assert!(
            s <= prev + 1e-3,
            "max slope must not increase: {s} > {prev}"
        );
        prev = s;
        steps += 1;
    });

    assert_eq!(steps, 150, "callback runs once per step");
    let slope1 = max_slope(&region);
    assert!(
        slope1 < slope0 * 0.75,
        "thermal should markedly reduce the max slope: {slope0} -> {slope1}"
    );

    // Thermal conserves terrain mass.
    let measured = height_sum_delta(&before, region.heights());
    assert!(
        measured.abs() < 1e-2,
        "thermal must conserve mass, moved {measured}"
    );
    assert!(
        stats.mass_delta.abs() < 1e-2,
        "stats mass_delta ≈ 0 for pure thermal: {}",
        stats.mass_delta
    );
}

// ── deposition ───────────────────────────────────────────────────────────────

#[test]
fn sediment_laden_flow_deposits_and_raises_ground() {
    // A tall slope on the left draining onto a low flat shelf on the right.
    let res = 48;
    let t = terrain_from(res, |x, _z| {
        if x < 24.0 {
            40.0 - x * 1.4 // steep ramp down
        } else {
            4.0 // flat basin
        }
    });
    let mut region = t.extract_region(DVec2::new(0.0, 0.0), DVec2::new(47.0, 47.0), 0);
    let before = region.heights().to_vec();

    // Thermal OFF so the only thing that can *raise* a cell is deposition.
    let params = ErosionParams {
        thermal_rate: 0.0,
        ..Default::default()
    };
    let _ = erode(&mut region, &params, 500);
    let after = region.heights().to_vec();

    let max_rise = before
        .iter()
        .zip(&after)
        .map(|(&a, &b)| b - a)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_rise > 1e-3,
        "sediment should deposit and raise some ground: max rise {max_rise}"
    );
}

// ── holes are walls ──────────────────────────────────────────────────────────

#[test]
fn unauthored_holes_never_receive_material() {
    // Two authored tiles with a one-tile gap between them: the gap is holes.
    let res = 16;
    let mut t = TerrainData::new(res, 1.0);
    let hill = |x: f64, _z: f64| 30.0 + (x % 15.0) * 0.5; // sloped, so erosion acts
    t.author_tile((0, 0), hill);
    t.author_tile((2, 0), hill);

    // Region spans all three tile columns; the middle (tile (1,0)) is holes.
    let mut region = t.extract_region(DVec2::new(0.0, 0.0), DVec2::new(45.0, 15.0), 0);
    let before = region.heights().to_vec();
    let (nx, nz) = region.dims();

    // Full erosion (rain + thermal). Nothing may enter the holes.
    let stats = erode(&mut region, &ErosionParams::default(), 200);
    let after = region.heights().to_vec();

    let mut hole_count = 0;
    for z in 0..nz {
        for x in 0..nx {
            if !region.is_authored(x, z) {
                hole_count += 1;
                let idx = (z * nx + x) as usize;
                assert_eq!(after[idx], 0.0, "hole cell ({x},{z}) got material");
            }
        }
    }
    assert!(hole_count > 0, "test must actually contain holes");

    // And the run was non-vacuous: authored terrain changed.
    let changed = before.iter().zip(&after).any(|(a, b)| (a - b).abs() > 1e-3);
    assert!(changed, "erosion should have altered the authored tiles");
    // Faithful accounting even with holes present.
    let measured = height_sum_delta(&before, &after);
    assert!((measured - stats.mass_delta).abs() <= 1e-3 * stats.mass_delta.abs().max(1.0));
}

/// **The carved twin of `unauthored_holes_never_receive_material`.** A P21.2
/// hole is a hole to erosion for exactly the same reason an unauthored tile is:
/// there is no surface there. The distinction matters because the two arrive by
/// completely different routes — one by never authoring a tile, the other by
/// carving a cave out of one that is fully authored, fully painted and sitting in
/// the middle of the region — and only the second can be *surrounded* by material
/// with somewhere to flow from.
///
/// A single tile, sloped so erosion genuinely runs, with an interior patch
/// carved: the patch must come out of a full erosion bake with nothing in it.
#[test]
fn carved_holes_never_receive_material() {
    let res = 32;
    let mut t = TerrainData::new(res, 1.0);
    t.author_tile((0, 0), |x, _z| 30.0 + (x % 15.0) * 0.5);

    // Carve an interior patch — well inside the tile, so it is ringed by material
    // on all four sides and a leaky implementation has plenty to leak.
    let tile = t.get_tile_mut((0, 0)).unwrap();
    for j in 12..20 {
        for i in 12..20 {
            tile.set_hole(res, i, j, true);
        }
    }
    assert!(t.get_tile((0, 0)).unwrap().has_holes());

    // Stop one sample short of the shared far edge: sample 31 belongs to tile
    // (1, 0), which does not exist, and counting THAT as a hole would let the
    // exact-count assertion below pass without the carve.
    let mut region = t.extract_region(DVec2::new(0.0, 0.0), DVec2::new(30.0, 30.0), 0);
    let before = region.heights().to_vec();
    let (nx, nz) = region.dims();

    let stats = erode(&mut region, &ErosionParams::default(), 200);
    let after = region.heights().to_vec();

    let mut hole_count = 0;
    for z in 0..nz {
        for x in 0..nx {
            if !region.is_authored(x, z) {
                hole_count += 1;
                let idx = (z * nx + x) as usize;
                assert_eq!(after[idx], 0.0, "carved cell ({x},{z}) got material");
            }
        }
    }
    // The carve is the ONLY source of unauthored cells here (the tile is whole),
    // so this also pins that `extract_region` really did read the hole mask.
    assert_eq!(
        hole_count, 64,
        "the region must see exactly the 8×8 carved patch as unauthored"
    );

    // Non-vacuous: the surrounding, authored terrain really did erode.
    let changed = before.iter().zip(&after).any(|(a, b)| (a - b).abs() > 1e-3);
    assert!(changed, "erosion should have altered the authored terrain");
    let measured = height_sum_delta(&before, &after);
    assert!((measured - stats.mass_delta).abs() <= 1e-3 * stats.mass_delta.abs().max(1.0));

    // … and writing back cannot fill the cave in either: the write-back guard
    // reads the same mask, so the tile's holes survive a bake untouched.
    let delta = t.edit_region(DVec2::new(0.0, 0.0), DVec2::new(30.0, 30.0), 0, |r| {
        let _ = erode(r, &ErosionParams::default(), 50);
    });
    let _ = delta;
    let tile = t.get_tile((0, 0)).unwrap();
    for j in 12..20 {
        for i in 12..20 {
            assert!(
                tile.is_hole(res, i, j),
                "erosion healed the hole at ({i},{j})"
            );
        }
    }
}

// ── erode_terrain round-trip through the undo delta ──────────────────────────

#[test]
fn erode_terrain_delta_reverts_byte_identical() {
    let mut t = cone(41, 50.0);
    let h0 = hash_terrain(&t);

    let (delta, maps, stats) = erode_terrain(
        &mut t,
        DVec2::new(0.0, 0.0),
        DVec2::new(40.0, 40.0),
        0,
        &ErosionParams::default(),
        300,
    );
    assert!(!delta.is_empty(), "erosion should produce a delta");
    // P19.1: the bake's sibling data-map delta, and it really materialized the
    // never-eroded tiles it touched.
    assert!(!maps.is_empty(), "erosion should produce a data-map delta");
    assert!(!maps.materialized_tiles.is_empty());
    assert!(!t.data_maps_are_default(), "the maps must have landed");
    assert!(stats.mass_delta < 0.0);
    assert_ne!(hash_terrain(&t), h0, "terrain should have changed");

    // ONE undo step is both halves: heights AND maps, byte-identical.
    t.revert_delta(&delta);
    t.revert_data_map_delta(&maps);
    assert!(
        t.data_maps_are_default(),
        "undo must un-materialize the maps"
    );
    assert_eq!(
        hash_terrain(&t),
        h0,
        "reverting the delta must restore byte-identical terrain"
    );

    // Redo lands back on the eroded state.
    let eroded = {
        let mut t2 = cone(41, 50.0);
        erode_terrain(
            &mut t2,
            DVec2::new(0.0, 0.0),
            DVec2::new(40.0, 40.0),
            0,
            &ErosionParams::default(),
            300,
        );
        hash_terrain(&t2)
    };
    t.apply_delta(&delta);
    t.apply_data_map_delta(&maps);
    assert_eq!(hash_terrain(&t), eroded, "redo must reproduce the erosion");
}

// ── zero-step no-op ──────────────────────────────────────────────────────────

#[test]
fn zero_steps_is_a_no_op() {
    let t = cone(17, 20.0);
    let mut region = t.extract_region(DVec2::new(0.0, 0.0), DVec2::new(16.0, 16.0), 0);
    let h0 = hash_region(&region);
    let stats = erode(&mut region, &ErosionParams::default(), 0);
    assert_eq!(hash_region(&region), h0, "0 steps must not change terrain");
    assert_eq!(stats, ErosionStats::default(), "0 steps ⇒ zero accounting");
}

// ── local profiling smoke (ignored by default) ───────────────────────────────

#[test]
#[ignore = "bench-ish: run with --ignored for local profiling"]
fn bench_smoke_512_squared_200_steps() {
    let res = 512;
    let t = cone(res, 200.0);
    let mut region = t.extract_region(
        DVec2::new(0.0, 0.0),
        DVec2::new((res - 1) as f64, (res - 1) as f64),
        0,
    );
    let start = std::time::Instant::now();
    let stats = erode(&mut region, &ErosionParams::default(), 200);
    let dt = start.elapsed();
    println!(
        "erode {res}²×200 steps: {:?} ({:.1} Msteps·cell/s), mass_delta={:.2}",
        dt,
        (res as f64 * res as f64 * 200.0) / dt.as_secs_f64() / 1e6,
        stats.mass_delta
    );
}

// ── P19.1: erosion data maps ─────────────────────────────────────────────────

/// A V-shaped valley running along +Z: the terrain a flow map has an obvious
/// right answer for. Ridge crests at `x = 0` and `x = res−1`, floor at the middle
/// column, with a gentle downhill along Z so water actually runs.
fn ridge_and_valley(res: u32) -> TerrainData {
    let c = (res as f64 - 1.0) / 2.0;
    terrain_from(res, move |x, z| {
        20.0 * ((x - c).abs() / c) + (res as f64 - z) * 0.04
    })
}

fn eroded_region(res: u32, steps: u32) -> HeightRegion {
    let t = ridge_and_valley(res);
    let hi = (res - 1) as f64;
    let mut region = t.extract_region(DVec2::ZERO, DVec2::new(hi, hi), 0);
    let params = ErosionParams {
        rain_rate: 0.05,
        rain_variation: 0.3,
        ..ErosionParams::default()
    };
    erode(&mut region, &params, steps);
    region
}

/// **Accumulator determinism.** The three maps are a pure function of
/// `(region, params, steps)` — byte-identical across repeated runs, and
/// unaffected by how much ambient parallelism is available (the erosion loop is
/// serial by construction; this pins that it stays that way).
#[test]
fn data_maps_are_byte_identical_across_runs_and_pool_sizes() {
    let baseline = hash_maps(&eroded_region(48, 60));
    assert_eq!(baseline, hash_maps(&eroded_region(48, 60)), "run-to-run");
    for threads in [1usize, 2, 4, 8] {
        let pool = inf_core::job::JobPool::new(threads);
        let got = pool.install(|| hash_maps(&eroded_region(48, 60)));
        assert_eq!(got, baseline, "data maps moved at pool size {threads}");
    }
}

/// **Conservation.** The mass-accounting gate, extended to the maps: every metre
/// the hydraulic passes deposit or wear is exactly the height they moved, so
/// `Σ deposition − Σ wear` equals the net height change. (Thermal is excluded
/// from the maps precisely so this identity is exact — see the module docs.)
#[test]
fn deposition_minus_wear_equals_the_net_height_change() {
    let t = ridge_and_valley(48);
    let hi = 47.0;
    let mut region = t.extract_region(DVec2::ZERO, DVec2::new(hi, hi), 0);
    let before: Vec<f32> = region.heights().to_vec();
    let params = ErosionParams {
        rain_rate: 0.05,
        ..ErosionParams::default()
    };
    let stats = erode(&mut region, &params, 80);

    let depo = map_total(&region, DataMapKind::Deposition);
    let wear = map_total(&region, DataMapKind::Wear);
    assert!(wear > 0.0, "the run must actually erode something");
    assert!(depo > 0.0, "the run must actually deposit something");

    // `mass_delta` is a volume (Σ Δb · l²); the maps are metres, and l = 1 here,
    // so the two are directly comparable.
    let moved = height_sum_delta(&before, region.heights());
    let net = depo - wear;
    let tol = 1e-3 * (depo + wear).max(1.0);
    assert!(
        (net - moved).abs() < tol,
        "deposition − wear = {net} but the heights moved {moved} (tol {tol})"
    );
    // …and the same identity against the independently-computed stats.
    assert!(
        (net - stats.mass_delta).abs() < tol,
        "deposition − wear = {net} but stats.mass_delta = {} (tol {tol})",
        stats.mass_delta
    );
    // Wear is the same quantity the stats accumulate as `sediment_moved`, summed
    // in f64 there and f32 here.
    assert!(
        (wear - stats.sediment_moved).abs() < 1e-2 * stats.sediment_moved.max(1.0),
        "wear {wear} vs sediment_moved {}",
        stats.sediment_moved
    );
}

/// **Known values, not exact pixels.** On a V-shaped valley the water collects in
/// the floor, so the flow map's per-column profile must **peak at the valley** —
/// the pattern any flow-accumulation map has to show. Asserted as a shape (the
/// argmax column and a ratio against the typical column), never as pixel values,
/// because the channels themselves are chaotic.
#[test]
fn flow_concentrates_in_the_valley() {
    let res = 48u32;
    let region = eroded_region(res, 120);
    let mid = (res / 2) as usize;
    let (nx, nz) = region.dims();

    // Per-column flow over interior rows only: the open outer edge drains, which
    // is a different regime from the interior the claim is about.
    let columns: Vec<f64> = (0..nx)
        .map(|x| {
            (4..nz - 4)
                .map(|z| region.map(x, z, DataMapKind::Flow) as f64)
                .sum()
        })
        .collect();
    let argmax = columns
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap();
    // The far column is the *next* tile's local 0 — a hole here — so the right
    // crest is the last authored one.
    let right = nx as usize - 2;
    eprintln!(
        "flow columns: argmax={argmax} (valley={mid}) left={:.2} peak={:.2} right={:.2}",
        columns[0], columns[argmax], columns[right]
    );

    assert!(columns[mid] > 0.0, "the valley must carry water at all");
    assert!(
        argmax.abs_diff(mid) <= 2,
        "the flow peak is at column {argmax}, not the valley floor ({mid})"
    );
    // Twice the flow of either crest — the concentration, stated as a ratio the
    // physics guarantees rather than a pixel value the chaos does not.
    for (label, crest) in [("left", columns[0]), ("right", columns[right])] {
        assert!(
            columns[mid] > 2.0 * crest,
            "flow did not concentrate vs the {label} crest: valley={} crest={crest}",
            columns[mid]
        );
    }
    // …and the profile really is a single hill: each flank's midpoint sits
    // strictly between its crest and the valley.
    for (crest, flank) in [(0usize, mid / 2), (right, (mid + right) / 2)] {
        assert!(
            columns[crest] < columns[flank] && columns[flank] < columns[mid],
            "profile is not unimodal at {flank}: crest={} flank={} valley={}",
            columns[crest],
            columns[flank],
            columns[mid]
        );
    }

    // The other two maps carry the complementary story, and it is the textbook
    // one: capacity is `Kc · sin(tilt) · |vel|`, so the steep flank cycles far
    // more material than the flat floor — it **wears** hardest — while the
    // valley's net balance (deposition − wear) is much closer to break-even
    // because what the flank sheds settles there. Both are asserted, because a
    // data map with the two channels swapped would still pass a "the valley is
    // special" check on flow alone.
    let at = |x: u32, kind: DataMapKind| -> f64 {
        (4..nz - 4).map(|z| region.map(x, z, kind) as f64).sum()
    };
    let flank = nx / 4;
    let net = |x: u32| at(x, DataMapKind::Deposition) - at(x, DataMapKind::Wear);
    eprintln!(
        "flank: wear={:.1} depo={:.1} net={:.1} | valley: wear={:.1} depo={:.1} net={:.1}",
        at(flank, DataMapKind::Wear),
        at(flank, DataMapKind::Deposition),
        net(flank),
        at(mid as u32, DataMapKind::Wear),
        at(mid as u32, DataMapKind::Deposition),
        net(mid as u32)
    );
    assert!(
        at(flank, DataMapKind::Wear) > at(mid as u32, DataMapKind::Wear),
        "the steep flank must wear more than the valley floor: flank={} valley={}",
        at(flank, DataMapKind::Wear),
        at(mid as u32, DataMapKind::Wear)
    );
    assert!(
        net(flank) < 0.0,
        "the flank must be a net source: {}",
        net(flank)
    );
    assert!(
        net(mid as u32) > net(flank),
        "the valley must sit above the flank on net balance: valley={} flank={}",
        net(mid as u32),
        net(flank)
    );
}

/// **The ACCUMULATORS compose across bakes — the simulation does not.**
///
/// Because the maps are extracted from the tiles and added to, a second 40-step
/// bake leaves strictly more flow than the first and undoing it returns the
/// first bake's totals **exactly**. That is the whole claim, and it is
/// deliberately not "two 40s equal one 80": a bake ends by discarding its water,
/// sediment and flux, so the second run starts on dry ground — measured 34 %
/// below the single long run over the same step count. The heights diverge the
/// same way; it is a property of the model, not of the maps.
#[test]
fn a_second_bake_adds_to_the_first_but_does_not_replay_it() {
    let params = ErosionParams {
        rain_rate: 0.05,
        ..ErosionParams::default()
    };
    let mut t = ridge_and_valley(32);
    let bounds = (DVec2::ZERO, DVec2::new(31.0, 31.0));
    let (_, maps1, _) = erode_terrain(&mut t, bounds.0, bounds.1, 0, &params, 40);
    assert!(!maps1.is_empty());
    let after_one = total_flow(&t);
    assert!(after_one > 0.0);

    let (_, maps2, _) = erode_terrain(&mut t, bounds.0, bounds.1, 0, &params, 40);
    // The second bake found the maps already materialized, so it materialized
    // nothing — the sparse marker is not re-armed by a re-bake.
    assert!(
        maps2.materialized_tiles.is_empty(),
        "a second bake must not re-materialize"
    );
    let after_two = total_flow(&t);
    assert!(
        after_two > after_one,
        "the second bake did not accumulate: {after_one} → {after_two}"
    );

    // Undoing only the second bake returns to the first bake's totals exactly.
    t.revert_data_map_delta(&maps2);
    assert_eq!(
        total_flow(&t),
        after_one,
        "undo of bake 2 must restore bake 1 exactly"
    );

    // And the honest negative: two 40s are NOT one 80. The doc says ~34 % less
    // flow, so this pins the direction and the order of magnitude rather than
    // letting the claim rot into "roughly the same".
    let one_long = {
        let mut t = ridge_and_valley(32);
        erode_terrain(&mut t, bounds.0, bounds.1, 0, &params, 80);
        total_flow(&t)
    };
    let shortfall = (one_long - after_two) / one_long;
    eprintln!(
        "2x40 flow={after_two:.2} | 1x80 flow={one_long:.2} (shortfall {:.1}%)",
        shortfall * 100.0
    );
    assert!(
        shortfall > 0.1,
        "a re-bake starting on dry ground must move measurably less water than one \
         long run: 2×40={after_two} vs 1×80={one_long}"
    );
}

/// Total flow accumulated over every authored tile.
fn total_flow(t: &TerrainData) -> f64 {
    t.tiles()
        .flat_map(|(_, tile)| tile.maps().iter().map(|m| m[0] as f64))
        .sum()
}

/// **Holes never accumulate.** An unauthored cell is a wall: no flux crosses it,
/// nothing erodes or deposits on it, and its maps stay at the never-eroded
/// default.
#[test]
fn holes_carry_no_data_maps() {
    let mut t = TerrainData::new(17, 1.0);
    t.author_tile((0, 0), |x, z| 10.0 - (x + z) * 0.1);
    let mut region = t.extract_region(DVec2::new(-8.0, -8.0), DVec2::new(24.0, 24.0), 0);
    let (nx, nz) = region.dims();
    let mut holes = 0usize;
    erode(&mut region, &ErosionParams::default(), 60);
    for z in 0..nz {
        for x in 0..nx {
            if !region.is_authored(x, z) {
                holes += 1;
                for kind in DataMapKind::ALL {
                    assert_eq!(region.map(x, z, kind), 0.0, "hole ({x},{z}) accumulated");
                }
            }
        }
    }
    assert!(holes > 0, "the fixture must actually contain holes");
}

/// **Persistence round-trip.** A terrain carrying data maps survives both codecs
/// byte-identically, and re-encodes stably.
#[test]
fn data_maps_round_trip_through_both_codecs() {
    let params = ErosionParams {
        rain_rate: 0.05,
        ..ErosionParams::default()
    };
    let mut t = ridge_and_valley(24);
    erode_terrain(&mut t, DVec2::ZERO, DVec2::new(23.0, 23.0), 0, &params, 30);
    assert!(!t.data_maps_are_default());

    let json = serde_json::to_string(&t).unwrap();
    let back: TerrainData = serde_json::from_str(&json).unwrap();
    assert_eq!(hash_terrain(&back), hash_terrain(&t), "json round trip");
    assert_eq!(
        serde_json::to_string(&back).unwrap(),
        json,
        "json is stable"
    );

    let cfg = bincode::config::standard();
    let bytes = bincode::serde::encode_to_vec(&t, cfg).unwrap();
    let (back, _): (TerrainData, usize) = bincode::serde::decode_from_slice(&bytes, cfg).unwrap();
    assert_eq!(hash_terrain(&back), hash_terrain(&t), "bincode round trip");
    assert_eq!(
        bincode::serde::encode_to_vec(&back, cfg).unwrap(),
        bytes,
        "bincode re-encode is byte-identical"
    );
}

/// **The absolute gate on flow's unit.** Every other flow assertion is
/// *relative* — a profile shape, a CPU/GPU agreement, a run-to-run hash — and a
/// coherent mutation that drops the `dt` factor from the accumulation passes all
/// of them: both paths would simply count steps instead of integrating seconds,
/// and nothing that compares one flow number to another would notice.
///
/// What pins the unit is that flow is a **time integral of a volume rate**, so
/// at a fixed simulated duration it must not depend on how finely that duration
/// was sliced. Halve `dt`, double `steps`, and the total must land in the same
/// place — whereas the drop-`dt` mutation doubles it.
///
/// Erosion is switched off (`dissolving`/`deposition`/`thermal_rate` are
/// per-**step** fractions, not per-second rates, so they genuinely do change
/// under refinement) leaving pure rain-and-transport, which is exactly the
/// quantity flow measures.
///
/// Mutation-verified: replacing `dt * (total * k)` with `total * k` in
/// `erosion.rs` makes the halved-`dt` run come back at ~2.0× and this fails.
#[test]
fn flow_is_a_time_integral_not_a_step_count() {
    let base = ErosionParams {
        rain_rate: 0.05,
        dissolving: 0.0,
        deposition: 0.0,
        thermal_rate: 0.0,
        evaporation: 0.0,
        ..ErosionParams::default()
    };
    let simulated_seconds = 2.0f32;
    let mut totals = Vec::new();
    for slices in [100u32, 200, 400] {
        let dt = simulated_seconds / slices as f32;
        let t = ridge_and_valley(32);
        let mut region = t.extract_region(DVec2::ZERO, DVec2::new(31.0, 31.0), 0);
        erode(&mut region, &ErosionParams { dt, ..base }, slices);
        let total = map_total(&region, DataMapKind::Flow);
        eprintln!("dt={dt:.5} steps={slices} → Σflow={total:.6} m³");
        totals.push(total);
    }
    assert!(totals[0] > 0.0, "the fixture must actually move water");
    // Halving dt at fixed simulated time is a refinement, not a rescale: the
    // integral converges rather than doubling. 15% covers the first-order
    // discretization error of the explicit pipe integrator at these step sizes;
    // the mutation it exists to catch is off by 100%.
    for w in totals.windows(2) {
        let ratio = w[1] / w[0];
        assert!(
            (ratio - 1.0).abs() < 0.15,
            "flow is not a time integral — refining dt changed it by {:.1}% \
             ({:.6} → {:.6}); a step-count accumulator would show ~100%",
            (ratio - 1.0) * 100.0,
            w[0],
            w[1]
        );
    }

    // The other half of the unit claim: flow really is a VOLUME, so it scales
    // with the cell area. The same run on 2 m cells ships 4× the water per cell.
    let coarse = {
        let mut t = TerrainData::new(32, 2.0);
        let c = 31.0 / 2.0;
        t.author_tile((0, 0), |x, z| {
            20.0 * ((x / 2.0 - c).abs() / c) + (32.0 - z / 2.0) * 0.04
        });
        let mut region = t.extract_region(DVec2::ZERO, DVec2::new(62.0, 62.0), 0);
        erode(
            &mut region,
            &ErosionParams {
                dt: simulated_seconds / 100.0,
                ..base
            },
            100,
        );
        map_total(&region, DataMapKind::Flow)
    };
    eprintln!(
        "1 m cells Σflow={:.4} | 2 m cells Σflow={coarse:.4}",
        totals[0]
    );
    assert!(
        coarse > 2.0 * totals[0],
        "flow does not scale with cell area — {coarse} vs {}",
        totals[0]
    );
}

// ── the talus threshold (Hardening Wave C, L6.F7) ────────────────────────────

/// The portable talus threshold agrees with `f64::tan` across the whole range a
/// repose angle can meaningfully take.
///
/// A tolerance and not a bit compare, necessarily: `psin64`/`pcos64` are ~1e-7
/// polynomials and `tan` is libm, and the point is precisely that the
/// polynomials do not depend on which libm is linked. The bound is *relative*
/// because `tan` runs away near vertical — at 85° it is 11.4, and an absolute
/// tolerance there would say nothing about the 30–40° band the parameter
/// actually lives in.
#[test]
fn the_talus_threshold_matches_tan_without_libm() {
    let mut worst = 0.0_f64;
    // 0.5° steps up to 89°, which covers the documented 30–40° repose band and
    // then some. 90° and past is the degenerate branch, checked below.
    for i in 0..=178 {
        let deg = i as f32 * 0.5;
        let got = inf_terrain::erosion::talus_tan(deg);
        let want = (deg as f64).to_radians().tan();
        let rel = if want.abs() > 1.0 {
            (got - want).abs() / want.abs()
        } else {
            (got - want).abs()
        };
        if rel > worst {
            worst = rel;
        }
        assert!(rel < 1.0e-6, "at {deg}°: {got} vs {want} (rel {rel})");
    }
    // Not vacuous: the sweep really did exercise a spread of values, and the
    // agreement is real rather than both sides being zero.
    assert!(
        worst > 0.0,
        "the two answers were bit-identical everywhere, which \
         would mean the portable pair is not being used"
    );
    assert!(
        (inf_terrain::erosion::talus_tan(45.0) - 1.0).abs() < 1.0e-7,
        "tan(45°) is 1"
    );
    assert_eq!(inf_terrain::erosion::talus_tan(0.0), 0.0);
}

/// **At or past vertical there is no finite slope, and the answer says so.**
///
/// This is the one input where the portable form and `f64::tan` differ, and the
/// difference is a fix rather than a drift. At exactly 90° the range reduction
/// inside `pcos64` lands on `sin(π)` and returns exactly `-0.0`; an unguarded
/// division would answer `-INFINITY`, which reads as "every slope is
/// over-steep" and would liquefy the terrain in one step. `f64::tan` answered
/// `1.6e16` — the same "nothing moves", by luck rather than by rule.
#[test]
fn a_vertical_talus_angle_moves_nothing() {
    for deg in [90.0_f32, 90.5, 120.0, 180.0, f32::NAN] {
        let t = inf_terrain::erosion::talus_tan(deg);
        assert!(
            t == f64::INFINITY,
            "a repose angle of {deg}° answered {t}; anything finite and negative \
             makes every slope over-steep"
        );
    }
    // The behaviour that threshold buys, on the real pass: the over-steep cliff
    // `thermal_relaxes_oversteep_cliff_toward_talus` relaxes by 25% under a 33°
    // repose angle does not move ONE BIT under a vertical one.
    let t = terrain_from(33, |x, _z| if x < 16.0 { 60.0 } else { 0.0 });
    let mut region = t.extract_region(DVec2::new(0.0, 0.0), DVec2::new(32.0, 32.0), 0);
    let before = region.heights().to_vec();
    let params = ErosionParams {
        rain_rate: 0.0,
        thermal_talus_deg: 90.0,
        thermal_rate: 1.0,
        ..Default::default()
    };
    erode(&mut region, &params, 150);
    assert_eq!(
        region.heights(),
        before.as_slice(),
        "a vertical talus angle moved material"
    );

    // NOT VACUOUS: the same cliff under the default repose angle really does
    // move, so the equality above is a property of the threshold and not of an
    // inert fixture.
    let mut moving = t.extract_region(DVec2::new(0.0, 0.0), DVec2::new(32.0, 32.0), 0);
    erode(
        &mut moving,
        &ErosionParams {
            rain_rate: 0.0,
            thermal_talus_deg: 33.0,
            thermal_rate: 1.0,
            ..Default::default()
        },
        150,
    );
    assert_ne!(moving.heights(), before.as_slice());
}
