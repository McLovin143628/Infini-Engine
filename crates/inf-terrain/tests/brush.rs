//! Sculpt-brush behaviour + the undo contract (P10.2a).

use glam::{DVec2, DVec3};
use inf_terrain::{
    apply_brush, dab_positions, BrushOp, BrushParams, Falloff, FlattenTarget, Stroke, TerrainData,
};
use xxhash_rust::xxh3::Xxh3;

// ── helpers ──────────────────────────────────────────────────────────────────

/// A single flat tile of resolution `res` at metres-per-sample 1.
fn flat_terrain(res: u32) -> TerrainData {
    let mut t = TerrainData::new(res, 1.0);
    t.author_tile((0, 0), |_, _| 0.0);
    t
}

/// Two horizontally-adjacent flat tiles sharing an edge at `x = span`.
fn two_flat_tiles(res: u32) -> TerrainData {
    let mut t = TerrainData::new(res, 1.0);
    t.author_tile((0, 0), |_, _| 0.0);
    t.author_tile((1, 0), |_, _| 0.0);
    t
}

/// A content hash over the tile set (coords + `f32` heights + `f64` anchors) —
/// byte-identity of the whole heightfield, tile membership included.
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

fn height(t: &TerrainData, x: f64, z: f64) -> f64 {
    t.height_at(DVec2::new(x, z)).unwrap()
}

// ── falloff ──────────────────────────────────────────────────────────────────

#[test]
fn falloff_center_edge_beyond() {
    let r = 8.0;
    for f in [
        Falloff::Smooth,
        Falloff::Sphere,
        Falloff::Linear,
        Falloff::Sharp,
        Falloff::Plateau(0.5),
    ] {
        assert!((f.weight(0.0, r) - 1.0).abs() < 1e-12, "{f:?} centre != 1");
        assert_eq!(f.weight(r, r), 0.0, "{f:?} edge != 0");
        assert_eq!(f.weight(r + 1.0, r), 0.0, "{f:?} beyond != 0");
        assert_eq!(f.weight(1000.0, r), 0.0, "{f:?} far != 0");
    }
}

#[test]
fn falloff_monotonic_decreasing() {
    let r = 10.0;
    for f in [
        Falloff::Smooth,
        Falloff::Sphere,
        Falloff::Linear,
        Falloff::Sharp,
        Falloff::Plateau(0.3),
    ] {
        let mut prev = f.weight(0.0, r);
        let mut d = 0.1;
        while d <= r {
            let w = f.weight(d, r);
            assert!(
                w <= prev + 1e-9,
                "{f:?} not monotonic at d={d}: {w} > {prev}"
            );
            assert!((0.0..=1.0).contains(&w), "{f:?} out of [0,1] at d={d}: {w}");
            prev = w;
            d += 0.1;
        }
    }
}

#[test]
fn plateau_is_flat_within_fraction() {
    let f = Falloff::Plateau(0.5);
    let r = 10.0;
    // Everything within 0.5·r is full weight; past it, it falls off.
    assert_eq!(f.weight(0.0, r), 1.0);
    assert_eq!(f.weight(4.9, r), 1.0);
    assert_eq!(f.weight(5.0, r), 1.0);
    assert!(f.weight(7.5, r) < 1.0 && f.weight(7.5, r) > 0.0);
    assert_eq!(f.weight(r, r), 0.0);
}

// ── raise / lower ────────────────────────────────────────────────────────────

#[test]
fn raise_exact_at_center_and_feathered_edge() {
    let mut t = flat_terrain(16);
    let params = BrushParams {
        center: DVec2::new(8.0, 8.0),
        radius: 4.0,
        strength: 10.0,
        falloff: Falloff::Linear,
    };
    apply_brush(&mut t, BrushOp::Raise, params);
    // Centre: weight 1 → exactly +strength.
    assert!((height(&t, 8.0, 8.0) - 10.0).abs() < 1e-4);
    // 2 m out on a radius-4 linear brush: weight 0.5 → +5.
    assert!((height(&t, 10.0, 8.0) - 5.0).abs() < 1e-4);
    // Just outside the radius: untouched.
    assert!(height(&t, 12.001, 8.0).abs() < 1e-6);
}

#[test]
fn lower_is_the_signed_mirror_of_raise() {
    let mut t = flat_terrain(16);
    let params = BrushParams::new(DVec2::new(8.0, 8.0), 4.0, 10.0);
    apply_brush(&mut t, BrushOp::Lower, params);
    assert!(
        (height(&t, 8.0, 8.0) + 10.0).abs() < 1e-4,
        "centre should be -10"
    );
}

#[test]
fn raise_is_seamless_across_a_tile_boundary() {
    let res = 8;
    let mut t = two_flat_tiles(res);
    let span = t.tile_span(); // shared edge at x = span
    let params = BrushParams::new(DVec2::new(span, 3.0), 3.0, 5.0);
    apply_brush(&mut t, BrushOp::Raise, params);

    let a = t.get_tile((0, 0)).unwrap();
    let b = t.get_tile((1, 0)).unwrap();
    let mut any_nonzero = false;
    for j in 0..res {
        let left = a.sample(res, res - 1, j); // far edge of tile 0
        let right = b.sample(res, 0, j); // near edge of tile 1
        assert!(
            (left - right).abs() < 1e-6,
            "seam at j={j}: {left} vs {right}"
        );
        any_nonzero |= left.abs() > 1e-6;
    }
    assert!(any_nonzero, "brush should have reached the shared edge");
    // And the interpolated height is continuous across the seam.
    let l = height(&t, span - 1e-6, 3.0);
    let r = height(&t, span + 1e-6, 3.0);
    assert!((l - r).abs() < 1e-3, "height discontinuity {l} vs {r}");
}

// ── smooth ───────────────────────────────────────────────────────────────────

#[test]
fn smooth_flattens_a_spike() {
    let mut t = flat_terrain(16);
    // A single tall spike.
    t.get_tile_mut((0, 0)).unwrap().set_sample(16, 8, 8, 100.0);
    let before = height(&t, 8.0, 8.0);
    assert!((before - 100.0).abs() < 1e-4);

    let params = BrushParams::new(DVec2::new(8.0, 8.0), 5.0, 1.0);
    apply_brush(&mut t, BrushOp::Smooth { iterations: 20 }, params);

    let after = height(&t, 8.0, 8.0);
    assert!(
        after < 40.0,
        "spike should collapse toward its neighbours: {after}"
    );
    assert!(after > 0.0, "but stay above the surrounding flat: {after}");
}

#[test]
fn smooth_is_seamless_across_a_tile_boundary() {
    let res = 8;
    let mut t = two_flat_tiles(res);
    let span = t.tile_span();
    // Spike on the shared edge — written into BOTH tiles' edge samples.
    t.get_tile_mut((0, 0))
        .unwrap()
        .set_sample(res, res - 1, 4, 80.0);
    t.get_tile_mut((1, 0)).unwrap().set_sample(res, 0, 4, 80.0);

    let params = BrushParams::new(DVec2::new(span, 4.0), 4.0, 1.0);
    apply_brush(&mut t, BrushOp::Smooth { iterations: 12 }, params);

    let a = t.get_tile((0, 0)).unwrap();
    let b = t.get_tile((1, 0)).unwrap();
    for j in 0..res {
        let left = a.sample(res, res - 1, j);
        let right = b.sample(res, 0, j);
        assert!(
            (left - right).abs() < 1e-5,
            "seam at j={j}: {left} vs {right}"
        );
    }
    // The spike was smoothed down.
    assert!(a.sample(res, res - 1, 4) < 80.0);
}

// ── flatten ──────────────────────────────────────────────────────────────────

#[test]
fn flatten_converges_to_picked_height() {
    let mut t = flat_terrain(16);
    // Vary the surface so flatten has something to converge.
    t.get_tile_mut((0, 0)).unwrap().set_sample(16, 8, 8, 30.0);
    let params = BrushParams {
        center: DVec2::new(8.0, 8.0),
        radius: 5.0,
        strength: 1.0,
        // Plateau(1.0) = uniform full weight inside the disk.
        falloff: Falloff::Plateau(1.0),
    };
    apply_brush(
        &mut t,
        BrushOp::Flatten {
            target: FlattenTarget::PickedHeight(50.0),
        },
        params,
    );
    // Full weight + full strength → snaps to target across the disk.
    assert!((height(&t, 8.0, 8.0) - 50.0).abs() < 1e-3);
    assert!((height(&t, 10.0, 8.0) - 50.0).abs() < 1e-3);
}

#[test]
fn flatten_mean_pulls_extremes_together() {
    let mut t = flat_terrain(16);
    let tile = t.get_tile_mut((0, 0)).unwrap();
    tile.set_sample(16, 7, 8, 0.0);
    tile.set_sample(16, 8, 8, 20.0); // a high sample amid the flat
    let params = BrushParams {
        center: DVec2::new(8.0, 8.0),
        radius: 5.0,
        strength: 1.0,
        falloff: Falloff::Plateau(1.0),
    };
    apply_brush(
        &mut t,
        BrushOp::Flatten {
            target: FlattenTarget::Mean,
        },
        params,
    );
    // The high sample drops toward the (small) regional mean; the flat rises a bit.
    let high = height(&t, 8.0, 8.0);
    assert!(
        high < 20.0 && high > 0.0,
        "high sample should fall to the mean: {high}"
    );
}

// ── noise ────────────────────────────────────────────────────────────────────

fn noise_op() -> BrushOp {
    BrushOp::Noise {
        seed: 1234,
        frequency: 0.1,
        octaves: 4,
        amplitude: 8.0,
    }
}

#[test]
fn noise_is_deterministic_across_identical_strokes() {
    let mut a = flat_terrain(32);
    let mut b = flat_terrain(32);
    let params = BrushParams::new(DVec2::new(16.0, 16.0), 10.0, 1.0);
    apply_brush(&mut a, noise_op(), params);
    apply_brush(&mut b, noise_op(), params);
    assert_eq!(
        hash_terrain(&a),
        hash_terrain(&b),
        "noise must be reproducible"
    );
}

#[test]
fn noise_is_world_anchored_and_order_independent() {
    // Two non-overlapping dabs applied in opposite order land identically, because
    // the noise is a function of world position, not the stroke path.
    let dab_a = BrushParams::new(DVec2::new(6.0, 6.0), 3.0, 1.0);
    let dab_b = BrushParams::new(DVec2::new(24.0, 24.0), 3.0, 1.0);

    let mut t1 = flat_terrain(32);
    apply_brush(&mut t1, noise_op(), dab_a);
    apply_brush(&mut t1, noise_op(), dab_b);

    let mut t2 = flat_terrain(32);
    apply_brush(&mut t2, noise_op(), dab_b);
    apply_brush(&mut t2, noise_op(), dab_a);

    assert_eq!(hash_terrain(&t1), hash_terrain(&t2));
}

// ── auto-authoring ───────────────────────────────────────────────────────────

#[test]
fn raise_auto_authors_tiles_in_empty_space() {
    let mut t = TerrainData::new(16, 1.0);
    assert_eq!(t.tile_count(), 0);
    apply_brush(
        &mut t,
        BrushOp::Raise,
        BrushParams::new(DVec2::new(5.0, 5.0), 4.0, 10.0),
    );
    assert!(t.tile_count() > 0, "raise should author a tile");
    assert!(t.has_tile((0, 0)));
}

#[test]
fn zero_write_and_relative_ops_never_author_empty_tiles() {
    // Raise with zero strength writes nothing → no tile.
    let mut t = TerrainData::new(16, 1.0);
    apply_brush(
        &mut t,
        BrushOp::Raise,
        BrushParams::new(DVec2::new(5.0, 5.0), 4.0, 0.0),
    );
    assert_eq!(t.tile_count(), 0, "zero-strength raise must not author");

    // Smooth / Flatten over empty space never author.
    apply_brush(
        &mut t,
        BrushOp::Smooth { iterations: 4 },
        BrushParams::new(DVec2::new(5.0, 5.0), 4.0, 1.0),
    );
    apply_brush(
        &mut t,
        BrushOp::Flatten {
            target: FlattenTarget::PickedHeight(10.0),
        },
        BrushParams::new(DVec2::new(5.0, 5.0), 4.0, 1.0),
    );
    apply_brush(
        &mut t,
        BrushOp::Flatten {
            target: FlattenTarget::Mean,
        },
        BrushParams::new(DVec2::new(5.0, 5.0), 4.0, 1.0),
    );
    assert_eq!(t.tile_count(), 0, "relative ops must never author");
}

// ── undo contract ────────────────────────────────────────────────────────────

#[test]
fn delta_revert_restores_byte_identical_heights() {
    let mut t = flat_terrain(16);
    // Add some pre-existing relief so revert has real bytes to restore.
    t.get_tile_mut((0, 0)).unwrap().set_sample(16, 4, 4, 7.5);
    let h0 = hash_terrain(&t);

    // A raise that both edits the existing tile AND grows into new tiles.
    let delta = apply_brush(
        &mut t,
        BrushOp::Raise,
        BrushParams::new(DVec2::new(15.0, 15.0), 8.0, 12.0),
    );
    assert!(t.tile_count() > 1, "brush should have grown the terrain");
    assert_ne!(hash_terrain(&t), h0);

    t.revert_delta(&delta);
    assert_eq!(
        hash_terrain(&t),
        h0,
        "revert must restore byte-identical state"
    );
    assert_eq!(t.tile_count(), 1, "created tiles must be removed on revert");

    // Redo lands back on the post-stroke state exactly.
    let h1_before = {
        let mut t2 = flat_terrain(16);
        t2.get_tile_mut((0, 0)).unwrap().set_sample(16, 4, 4, 7.5);
        apply_brush(
            &mut t2,
            BrushOp::Raise,
            BrushParams::new(DVec2::new(15.0, 15.0), 8.0, 12.0),
        );
        hash_terrain(&t2)
    };
    t.apply_delta(&delta);
    assert_eq!(
        hash_terrain(&t),
        h1_before,
        "redo must reproduce the stroke"
    );
}

// ── stroke merge ─────────────────────────────────────────────────────────────

#[test]
fn stroke_merges_overlapping_dabs_with_first_before_final_after() {
    let mut t = flat_terrain(16);
    let h0 = hash_terrain(&t);

    let mut stroke = Stroke::begin();
    // Two overlapping raise dabs; their union raises the shared sample twice.
    stroke.add_dab(
        &mut t,
        BrushOp::Raise,
        BrushParams::new(DVec2::new(7.0, 8.0), 3.0, 5.0),
    );
    stroke.add_dab(
        &mut t,
        BrushOp::Raise,
        BrushParams::new(DVec2::new(9.0, 8.0), 3.0, 5.0),
    );
    let final_center = height(&t, 8.0, 8.0);
    assert!(
        final_center > 5.0,
        "overlap region raised by both dabs: {final_center}"
    );
    let h_final = hash_terrain(&t);

    let delta = stroke.finish(&t);
    // before = first touch (flat 0) → revert returns to the original.
    t.revert_delta(&delta);
    assert_eq!(
        hash_terrain(&t),
        h0,
        "stroke revert must restore the start (before = first touch)"
    );
    // after = final value → redo returns to the merged result.
    t.apply_delta(&delta);
    assert_eq!(
        hash_terrain(&t),
        h_final,
        "stroke redo must restore the final (after = final touch)"
    );
}

// ── region round-trip ────────────────────────────────────────────────────────

#[test]
fn region_extract_write_back_round_trips() {
    let mut t = TerrainData::new(16, 1.0);
    let f = |x: f64, z: f64| (x * 0.2).sin() * 5.0 + z * 0.1;
    t.author_tile((0, 0), f);
    t.author_tile((1, 0), f);
    let h0 = hash_terrain(&t);

    // Extract then write back unchanged: a no-op delta, heights untouched.
    let delta = t.edit_region(
        DVec2::new(2.0, 2.0),
        DVec2::new(20.0, 12.0),
        0,
        |_region| {},
    );
    assert!(
        delta.is_empty(),
        "an unmodified round-trip must produce no delta"
    );
    assert_eq!(hash_terrain(&t), h0, "round-trip must not perturb heights");
}

// ── dab spacing ──────────────────────────────────────────────────────────────

#[test]
fn dab_positions_are_evenly_spaced_along_the_path() {
    let path = [DVec2::new(0.0, 0.0), DVec2::new(10.0, 0.0)];
    let dabs = dab_positions(&path, 2.5);
    assert_eq!(dabs.len(), 5); // 0, 2.5, 5, 7.5, 10
    for (k, p) in dabs.iter().enumerate() {
        assert!((p.x - k as f64 * 2.5).abs() < 1e-9, "dab {k} at {p:?}");
        assert!(p.y.abs() < 1e-9);
    }
}

#[test]
fn dab_positions_span_a_corner_by_arc_length() {
    // An L-shaped path: spacing carries across the bend.
    let path = [
        DVec2::new(0.0, 0.0),
        DVec2::new(4.0, 0.0),
        DVec2::new(4.0, 4.0),
    ];
    let dabs = dab_positions(&path, 2.0);
    // Arc length 8 → dabs at 0,2,4,6,8 = 5 points, last at the far end.
    assert_eq!(dabs.len(), 5);
    let last = *dabs.last().unwrap();
    assert!((last - DVec2::new(4.0, 4.0)).length() < 1e-9);
}

#[test]
fn dab_positions_edge_cases() {
    assert!(dab_positions(&[], 1.0).is_empty());
    let single = [DVec2::new(3.0, 4.0)];
    assert_eq!(dab_positions(&single, 1.0), vec![DVec2::new(3.0, 4.0)]);
    // Non-positive spacing returns the raw path points.
    let path = [DVec2::new(0.0, 0.0), DVec2::new(1.0, 1.0)];
    assert_eq!(dab_positions(&path, 0.0), path.to_vec());
}

// ── determinism guard on DVec3 anchor usage (keeps origin.y in the hash path) ──

#[test]
fn sculpt_preserves_zero_anchor() {
    let mut t = flat_terrain(16);
    apply_brush(
        &mut t,
        BrushOp::Raise,
        BrushParams::new(DVec2::new(8.0, 8.0), 4.0, 3.0),
    );
    // The engine keeps origin.y == 0; the f32 offset carries the height.
    assert_eq!(
        t.get_tile((0, 0)).unwrap().origin,
        DVec3::new(0.0, 0.0, 0.0)
    );
}

// ── the hardening-B NaN arms (C4-35, C4-36) ─────────────────────────────────

/// **C4-36 — `Falloff::weight` was the standing NaN-blind-guard class, verbatim,
/// on the `.inf_terrain` write path.**
///
/// `radius <= 0.0` is false for NaN. `dist >= radius` is false for NaN. And
/// `(dist / radius).clamp(0.0, 1.0)` *propagates* NaN — `f64::clamp` returns
/// `self` when `self` is NaN — so the poison came out the other side as a
/// weight, which the caller multiplied a height by.
///
/// Un-fix mutation: delete the `is_finite` line at the top of `weight` and every
/// falloff below returns NaN.
#[test]
fn a_non_finite_brush_argument_weighs_nothing() {
    for f in [
        Falloff::Smooth,
        Falloff::Sphere,
        Falloff::Linear,
        Falloff::Sharp,
        Falloff::Plateau(0.5),
    ] {
        for (d, r) in [
            (f64::NAN, 4.0),
            (2.0, f64::NAN),
            (f64::NAN, f64::NAN),
            (f64::INFINITY, 4.0),
            (2.0, f64::INFINITY),
        ] {
            let w = f.weight(d, r);
            assert!(
                w.is_finite() && (0.0..=1.0).contains(&w),
                "{f:?}.weight({d}, {r}) = {w}"
            );
        }
        // The healthy contract is untouched: 1 at the centre, 0 at the radius.
        assert!((f.weight(0.0, 4.0) - 1.0).abs() < 1e-12);
        assert_eq!(f.weight(4.0, 4.0), 0.0);
    }
}

/// **C4-35 — a saturating sculpt used to write NaN into the committed
/// heightfield.**
///
/// `Raise` computes `old + strength · w` in f64 and narrows with `as f32`, which
/// **saturates to `f32::INFINITY`** rather than wrapping. A later `Smooth` dab
/// over the same tile computes `old + (mean − old) · blend` — `inf − inf` —
/// which is NaN; `Flatten` does the same. Every guard between there and the
/// tile was NaN-blind: `w <= 0.0` is false for NaN so nothing was skipped, and
/// `new_off != old_off` is ALWAYS true for NaN so every sample in the footprint
/// was committed. `encode_tile` then bincodes it with no finiteness check.
///
/// This asserts the WORLD — every stored sample of every tile — rather than the
/// call's return value.
#[test]
fn no_sculpt_can_put_a_non_finite_height_into_a_tile() {
    let mut t = flat_terrain(16);
    // A strength beyond f32's reach. The Tauri door bounds this now
    // (`SculptSettings::sanitized`), but Ring 0 has no door and the op is public.
    apply_brush(
        &mut t,
        BrushOp::Raise,
        BrushParams::new(DVec2::new(8.0, 8.0), 6.0, 1.0e300),
    );
    apply_brush(
        &mut t,
        BrushOp::Smooth { iterations: 2 },
        BrushParams::new(DVec2::new(8.0, 8.0), 6.0, 0.5),
    );
    apply_brush(
        &mut t,
        BrushOp::Flatten {
            target: FlattenTarget::Mean,
        },
        BrushParams::new(DVec2::new(8.0, 8.0), 6.0, 0.5),
    );
    // And the NaN arguments themselves, straight at the op.
    apply_brush(
        &mut t,
        BrushOp::Raise,
        BrushParams::new(DVec2::new(8.0, 8.0), f64::NAN, f64::NAN),
    );
    for (coord, tile) in t.tiles() {
        for (i, &v) in tile.heights().iter().enumerate() {
            assert!(
                v.is_finite(),
                "tile {coord:?} sample {i} is {v} — a non-finite height reached the heightfield"
            );
        }
    }
}
