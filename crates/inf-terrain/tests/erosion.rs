//! Hydraulic + thermal erosion behaviour + mass accounting (P10.3a).
//!
//! These test the *algorithm*, not just plumbing: channels form, mass is
//! accounted, water is conserved in a closed box, thermal relaxes to the talus
//! angle, deposition raises a basin, holes are walls, and the whole thing is
//! byte-for-byte deterministic.

use glam::DVec2;
use inf_terrain::{
    erode, erode_terrain, erode_with, ErosionParams, ErosionStats, HeightRegion, TerrainData,
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
    }
    h.digest128()
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

// ── erode_terrain round-trip through the undo delta ──────────────────────────

#[test]
fn erode_terrain_delta_reverts_byte_identical() {
    let mut t = cone(41, 50.0);
    let h0 = hash_terrain(&t);

    let (delta, stats) = erode_terrain(
        &mut t,
        DVec2::new(0.0, 0.0),
        DVec2::new(40.0, 40.0),
        0,
        &ErosionParams::default(),
        300,
    );
    assert!(!delta.is_empty(), "erosion should produce a delta");
    assert!(stats.mass_delta < 0.0);
    assert_ne!(hash_terrain(&t), h0, "terrain should have changed");

    t.revert_delta(&delta);
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
