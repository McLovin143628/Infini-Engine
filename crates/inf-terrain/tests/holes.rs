//! The **per-sample hole mask** (P21.2): the layer that lets a heightfield stop
//! being a heightfield in one place, so a voxel volume can put a cave there.
//!
//! Five properties, one file:
//!
//! 1. the mask is **byte-stable** — an un-carved tile is indistinguishable from a
//!    pre-P21.2 one, and a carve→heal cycle returns to exactly those bytes;
//! 2. the packing is **bits, not bytes**, and costs what the docs claim;
//! 3. every query obeys **one poison rule** — a holed corner removes its whole
//!    bilinear cell, in `height_at`, `normal_at`, `is_hole_at` and the raycast
//!    alike, so the walkable gap and the visible gap are the same gap;
//! 4. the sculpt brush **passes over** holes rather than re-growing them;
//! 5. [`HoleDelta`] undoes byte-identically, including collapsing a mask the
//!    carve materialized.

use glam::{DVec2, DVec3};
use inf_terrain::{
    hole_mask_bytes, raycast_terrain, BrushOp, BrushParams, HoleDeltaBuilder, TerrainData,
};

/// A single sloped tile with no holes — every test's starting point.
fn sloped(res: u32) -> TerrainData {
    let mut t = TerrainData::new(res, 1.0);
    t.author_tile((0, 0), |x, z| 10.0 + x * 0.25 - z * 0.1);
    t
}

/// The tile's `.inf_terrain` blob — the container that **carries holes**.
///
/// Deliberately not `TerrainData`'s own serde: that wire form is pinned at tile
/// generation 3 and drops the mask on purpose (the scene schema is frozen at
/// v19), so byte-identity through *it* would be satisfied by a carve that did
/// nothing at all. `inf-scene`'s `terrain_data_wire_is_pinned_at_generation_three`
/// is the test that pins that other, deliberate silence.
fn blob(t: &TerrainData) -> Vec<u8> {
    inf_terrain::asset::encode_tile(t.get_tile((0, 0)).unwrap()).unwrap()
}

// ── 1 + 2. representation and its price ─────────────────────────────────────

/// The sparse default is genuinely free, and the materialized mask is genuinely
/// packed. Both halves priced against the documented formula rather than a magic
/// number, so a change to the packing has to change the doc too.
#[test]
fn an_un_carved_tile_pays_nothing_and_a_carved_one_pays_bits() {
    let res = 32;
    let t = sloped(res);
    let tile = t.get_tile((0, 0)).unwrap();
    assert!(tile.holes_are_default());
    assert!(!tile.has_holes());
    assert_eq!(tile.holes_len(), 0);

    let mut carved = t.clone();
    carved
        .get_tile_mut((0, 0))
        .unwrap()
        .set_hole(res, 5, 7, true);
    let tile = carved.get_tile((0, 0)).unwrap();
    assert!(!tile.holes_are_default());
    assert!(tile.has_holes());
    assert_eq!(tile.holes_len(), hole_mask_bytes(res));
    assert_eq!(hole_mask_bytes(res), (res * res).div_ceil(8) as usize);
    // The whole point of packing: one bit, not one byte, per sample.
    assert_eq!(tile.holes_len() * 8, (res * res) as usize);

    // Exactly the addressed sample is holed, and its neighbours are not — a
    // shift-by-one bug in the bit index would light the wrong one.
    assert!(tile.is_hole(res, 5, 7));
    for (i, j) in [(4, 7), (6, 7), (5, 6), (5, 8)] {
        assert!(!tile.is_hole(res, i, j), "({i},{j}) should be untouched");
    }
}

/// **Byte stability, both directions.** An un-carved tile serializes exactly as
/// it did before the layer existed (the sparse-default promise), and carving
/// then healing returns to those same bytes — which is what makes carve → undo a
/// real round trip rather than a value-identical near-miss.
#[test]
fn carve_then_heal_is_byte_identical() {
    let res = 16;
    let t = sloped(res);
    let clean = blob(&t);

    let mut carved = t.clone();
    {
        let tile = carved.get_tile_mut((0, 0)).unwrap();
        for i in 3..9 {
            tile.set_hole(res, i, 4, true);
        }
    }
    // The blob really does grow, by exactly the packed mask.
    assert_eq!(blob(&carved).len(), clean.len() + hole_mask_bytes(res));

    // Clear every bit …
    {
        let tile = carved.get_tile_mut((0, 0)).unwrap();
        for i in 3..9 {
            tile.set_hole(res, i, 4, false);
        }
        // … and the buffer is still there until it is healed.
        assert!(!tile.holes_are_default());
        assert!(!tile.has_holes());
        assert!(tile.heal_holes(), "an all-zero mask must collapse");
        assert!(!tile.heal_holes(), "healing twice changes nothing");
    }
    assert_eq!(blob(&carved), clean, "carve → heal was not byte-identical");
}

/// Closing a hole that was never open must not *cost* the tile its default —
/// otherwise a fill brush sweeping un-carved ground would materialize a mask per
/// tile it passed over and every one of them would ride into the asset forever.
#[test]
fn clearing_a_hole_on_a_clean_tile_allocates_nothing() {
    let res = 8;
    let mut t = sloped(res);
    let tile = t.get_tile_mut((0, 0)).unwrap();
    for j in 0..res {
        for i in 0..res {
            tile.set_hole(res, i, j, false);
        }
    }
    assert!(tile.holes_are_default(), "a no-op fill materialized a mask");
}

// ── 3. the poison rule ──────────────────────────────────────────────────────

/// **One holed corner removes the whole cell**, and the removal is *exactly* one
/// cell wide in each direction — not the sample, not the tile.
///
/// The count is asserted, not sampled: a rule that poisoned too little would
/// leave walkable ground the clipmap does not draw, and one that poisoned too
/// much would eat the cave's rim.
#[test]
fn a_holed_corner_poisons_its_bilinear_cell_and_no_more() {
    let res = 16;
    let mut t = sloped(res);
    // Sample (8, 8) only.
    t.get_tile_mut((0, 0)).unwrap().set_hole(res, 8, 8, true);

    // Every cell that interpolates (8, 8) has its lower corner at (7..=8, 7..=8),
    // so the dead band is world x/z ∈ [7, 9] and everything outside is alive.
    for j in 0..res {
        for i in 0..res {
            let p = DVec2::new(i as f64, j as f64);
            // A lattice point at index `i` floors into the cell `[i, i+1]`, so
            // the four points that can see sample (8, 8) are (7..=8, 7..=8) —
            // and no others.
            let dead = (7..=8).contains(&i) && (7..=8).contains(&j);
            assert_eq!(t.height_at(p).is_none(), dead, "lattice point ({i},{j})");
        }
    }

    // Off-lattice, inside the cell, the poison is real and symmetric.
    for (dx, dz) in [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
        let p = DVec2::new(8.0 + dx, 8.0 + dz);
        assert!(t.height_at(p).is_none(), "cell around {p:?} survived");
        assert!(t.is_hole_at(p), "{p:?} should report as holed");
        assert!(t.normal_at(p).is_none(), "a hole has no normal");
    }
    // One cell further out is untouched.
    for (dx, dz) in [(-1.5, -1.5), (1.5, 1.5)] {
        let p = DVec2::new(8.0 + dx, 8.0 + dz);
        assert!(t.height_at(p).is_some(), "{p:?} was poisoned too far");
        assert!(!t.is_hole_at(p));
    }
}

/// `is_hole_at` is **not** `height_at(..).is_none()`: outside the authored
/// terrain there is no surface either, and a ground query has to tell "carved
/// through" from "nothing here" to know whether to go looking in a voxel volume.
#[test]
fn a_hole_is_distinguishable_from_the_edge_of_the_world() {
    let res = 16;
    let mut t = sloped(res);
    t.get_tile_mut((0, 0)).unwrap().set_hole(res, 4, 4, true);

    let carved = DVec2::new(4.0, 4.0);
    assert!(t.height_at(carved).is_none());
    assert!(t.is_hole_at(carved), "a carve must read as a hole");

    let outside = DVec2::new(-500.0, -500.0);
    assert!(t.height_at(outside).is_none());
    assert!(
        !t.is_hole_at(outside),
        "unauthored space is not a hole — nothing was carved there"
    );

    assert!(t.has_holes());
    assert!(!sloped(res).has_holes());
}

/// A ray fired straight down a carved cell **passes through** rather than
/// stopping on a surface the clipmap will not draw. The control — the same ray
/// one cell over — must hit, or the test proves nothing about the hole.
#[test]
fn a_raycast_passes_through_a_carved_cell() {
    let res = 16;
    let mut t = sloped(res);
    {
        let tile = t.get_tile_mut((0, 0)).unwrap();
        for j in 6..11 {
            for i in 6..11 {
                tile.set_hole(res, i, j, true);
            }
        }
    }
    let down = DVec3::new(0.0, -1.0, 0.0);
    let through = raycast_terrain(&t, DVec3::new(8.0, 60.0, 8.0), down, 200.0);
    assert!(
        through.is_none(),
        "the ray stopped inside a hole: {through:?}"
    );

    let control = raycast_terrain(&t, DVec3::new(2.0, 60.0, 2.0), down, 200.0);
    assert!(
        control.is_some(),
        "the control ray must hit, or the hole proves nothing"
    );
}

// ── 4. the brush passes over ────────────────────────────────────────────────

/// Sculpting across a cave mouth must not re-grow its roof: a holed sample is
/// skipped exactly like an unauthored one, and its stored height is left alone.
#[test]
fn the_sculpt_brush_never_writes_a_holed_sample() {
    let res = 24;
    let mut t = sloped(res);
    {
        let tile = t.get_tile_mut((0, 0)).unwrap();
        for j in 10..14 {
            for i in 10..14 {
                tile.set_hole(res, i, j, true);
            }
        }
    }
    let before: Vec<f32> = (10..14)
        .flat_map(|j| (10..14).map(move |i| (i, j)))
        .map(|(i, j)| t.get_tile((0, 0)).unwrap().sample(res, i, j))
        .collect();

    let delta = inf_terrain::apply_brush(
        &mut t,
        BrushOp::Raise,
        BrushParams::new(DVec2::new(12.0, 12.0), 8.0, 5.0),
    );
    assert!(
        !delta.is_empty(),
        "the brush must have moved the surrounding ground"
    );

    let after: Vec<f32> = (10..14)
        .flat_map(|j| (10..14).map(move |i| (i, j)))
        .map(|(i, j)| t.get_tile((0, 0)).unwrap().sample(res, i, j))
        .collect();
    assert_eq!(before, after, "the brush wrote into the hole");
    // The ring around it did move — the raise really did land.
    assert_ne!(
        t.get_tile((0, 0)).unwrap().sample(res, 8, 12),
        sloped(res).get_tile((0, 0)).unwrap().sample(res, 8, 12)
    );
}

// ── 5. the undo record ──────────────────────────────────────────────────────

/// [`HoleDelta`] applies and reverts byte-identically, and the revert **heals**:
/// a carve that materialized a mask undoes back to bytes a carve never touched.
#[test]
fn a_hole_delta_round_trips_byte_identically() {
    let res = 16;
    let mut t = sloped(res);
    let clean = blob(&t);

    let mut b = HoleDeltaBuilder::new();
    {
        let res_ = res;
        let tile = t.get_tile_mut((0, 0)).unwrap();
        for j in 5..9 {
            for i in 5..9 {
                b.touch((0, 0), i, j, tile.is_hole(res_, i, j));
                tile.set_hole(res_, i, j, true);
            }
        }
    }
    let delta = b.finalize(&t);
    assert!(!delta.is_empty());
    assert_eq!(delta.net_opened(), 16, "sixteen samples opened");
    let carved = blob(&t);
    assert_ne!(carved, clean);

    assert_eq!(t.revert_hole_delta(&delta), 0);
    assert!(
        t.get_tile((0, 0)).unwrap().holes_are_default(),
        "undo left the materialized mask behind"
    );
    assert_eq!(blob(&t), clean, "undo was not byte-identical");

    assert_eq!(t.apply_hole_delta(&delta), 0);
    assert_eq!(blob(&t), carved, "redo was not byte-identical");

    // A second revert is idempotent, and a wash produces no delta at all.
    assert_eq!(t.revert_hole_delta(&delta), 0);
    assert_eq!(t.revert_hole_delta(&delta), 0);
    assert_eq!(blob(&t), clean);

    let mut wash = HoleDeltaBuilder::new();
    wash.touch((0, 0), 1, 1, false);
    assert!(
        wash.finalize(&t).is_empty(),
        "a touch that changed nothing must not become an undo step"
    );
}

/// The heightfield delta and the hole delta are **independent layers**: undoing
/// a carve's holes must not disturb the heights, and vice versa. The editor
/// groups them into one transaction; that is a policy, and this is why it is
/// safe.
#[test]
fn the_two_deltas_do_not_interfere() {
    let res = 16;
    let mut t = sloped(res);

    let height_delta = inf_terrain::apply_brush(
        &mut t,
        BrushOp::Lower,
        BrushParams::new(DVec2::new(8.0, 8.0), 4.0, 2.0),
    );
    let after_sculpt = blob(&t);

    let mut b = HoleDeltaBuilder::new();
    {
        let tile = t.get_tile_mut((0, 0)).unwrap();
        b.touch((0, 0), 8, 8, false);
        tile.set_hole(res, 8, 8, true);
    }
    let hole_delta = b.finalize(&t);

    assert_eq!(t.revert_hole_delta(&hole_delta), 0);
    assert_eq!(
        blob(&t),
        after_sculpt,
        "reverting holes disturbed the heights"
    );

    t.revert_delta(&height_delta);
    assert_eq!(blob(&t), blob(&sloped(res)));
}

// ── 6. the GPU row-pack ──────────────────────────────────────────

/// [`pack_hole_rows`] is the *only* translation between the tile's dense bit
/// layout and the row-aligned one a fragment shader indexes, so it is pinned by
/// reading every bit back out through the row rule and comparing against the
/// tile itself — at a resolution that is deliberately **not** a multiple of 32,
/// which is where an off-by-a-row bug lives.
#[test]
fn the_gpu_row_pack_agrees_with_the_tile_bit_for_bit() {
    let res = 33;
    let mut t = TerrainData::new(res, 1.0);
    t.author_tile((0, 0), |_, _| 0.0);
    {
        let tile = t.get_tile_mut((0, 0)).unwrap();
        // A pattern that crosses word boundaries in both axes.
        for j in 0..res {
            for i in 0..res {
                if (i * 7 + j * 13) % 11 == 0 {
                    tile.set_hole(res, i, j, true);
                }
            }
        }
    }
    let tile = t.get_tile((0, 0)).unwrap();
    let packed = inf_terrain::pack_hole_rows(tile, res);
    let words = res.div_ceil(32) as usize;
    assert_eq!(packed.len(), words * res as usize);
    assert_eq!(words, 2, "a 33-sample row must cost two words");

    let mut holed = 0usize;
    for j in 0..res {
        for i in 0..res {
            let bit = packed[j as usize * words + (i >> 5) as usize] & (1u32 << (i & 31)) != 0;
            assert_eq!(bit, tile.is_hole(res, i, j), "({i},{j})");
            holed += usize::from(bit);
        }
    }
    assert!(
        holed > 50,
        "only {holed} holed samples — the pattern is too sparse"
    );

    // The sparse default packs to nothing at all — which is what lets the
    // renderer bind a four-byte sentinel for an un-carved tile.
    assert!(inf_terrain::pack_hole_rows(sloped(res).get_tile((0, 0)).unwrap(), res).is_empty());
}
