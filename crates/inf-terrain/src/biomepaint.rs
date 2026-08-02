//! **Biome painting** (P19.2): the biome-id twin of [`crate::splat`].
//!
//! Where a splat brush blends four per-sample weights, a biome brush writes a
//! single per-sample **id** — a categorical label, not a quantity. The delta /
//! stroke / undo shapes mirror the splat side exactly:
//!
//! * [`apply_paint_biome`] is the analogue of [`apply_paint`](crate::apply_paint):
//!   one dab returns a reversible [`BiomeDelta`].
//! * [`BiomeStroke`] merges the dabs of one mouse-down→up gesture into a single
//!   [`BiomeDelta`] (`before` = first touch, `after` = final).
//! * [`BiomeDelta`] / [`BiomePatch`] mirror [`SplatDelta`] / [`SplatPatch`]:
//!   dense per-tile sub-rect patches the editor's undo stack stores and
//!   [`TerrainData::revert_biome_delta`] replays.
//!
//! Painting **never authors tiles** (like Smooth/Flatten and like splat): ids are
//! only written where terrain already exists. It *does*
//! [materialize](crate::TerrainTile::ensure_biomes) a tile's sparse id buffer on
//! first touch; the delta records which tiles it materialized
//! (`materialized_tiles`), so undo drops them back to the byte-stable sparse
//! default — a biome stroke on fresh terrain undoes to byte-identical unpainted
//! space.
//!
//! # THE HARD-EDGE DECISION
//!
//! A biome boundary is **semantic**, so the brush writes a crisp edge: a sample
//! is either in the biome or it is not. What the falloff and the strength do is
//! decide *where that edge falls* —
//!
//! ```text
//! claimed  ⇔  falloff.weight(d, r) > 0  ∧  falloff.weight(d, r) ≥ 1 − strength
//! ```
//!
//! — so `strength = 1` claims the whole disk (a stamp), `strength = 0.5` claims
//! the half-weight contour (a smaller disk under a soft falloff, the whole disk
//! under [`Falloff::Plateau(1.0)`](crate::Falloff)), and `strength = 0` claims
//! nothing. Monotone in both parameters, pure, and every slider position does
//! something visible.
//!
//! **A "soft radius" that writes a majority vote was considered and rejected.**
//! Majority-vote smoothing over each sample's neighbourhood would make a dab's
//! result depend on ids the *previous* dabs wrote, so:
//!
//! * a stroke would stop being a function of its path — dab order would change
//!   the result, and the editor resamples a drag into dabs at ~⅓ radius, which is
//!   a timing-dependent count;
//! * re-applying the same dab would keep changing the terrain, breaking the
//!   `apply(revert(apply)) == apply` identity every delta in this crate relies on
//!   (redo must reproduce the post-stroke bytes exactly);
//! * and the visible effect — a ragged, half-eroded boundary — is not what
//!   "softer brush" means to someone painting a forest edge.
//!
//! Boundary *feathering* is a real requirement, but it belongs where it is
//! consumed: P19.3 blends between adjacent biomes' PCG graphs across a feather
//! width, reading the crisp ids stored here. Feathering the **storage** would
//! throw away the information that blend needs.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::brush::BrushParams;
use crate::data::TerrainData;
use crate::tile::{DEFAULT_BIOME, UNASSIGNED_BIOME};

/// Whether a brush at falloff `weight` and `strength` claims a sample for its
/// biome — the hard-edge rule, stated once (see the module docs).
///
/// Pure and total: `weight` outside `[0, 1]` and `strength` outside `[0, 1]`
/// clamp rather than misbehave.
#[inline]
pub fn biome_claims(weight: f64, strength: f64) -> bool {
    let s = strength.clamp(0.0, 1.0);
    if s <= 0.0 {
        return false;
    }
    let w = weight.clamp(0.0, 1.0);
    w > 0.0 && w >= 1.0 - s
}

// ── delta (undo record) ──────────────────────────────────────────────────────

/// A dense before/after patch over one tile's `[i0, i0+w) × [j0, j0+h)` sample
/// rectangle of biome ids. Mirrors [`SplatPatch`](crate::SplatPatch) at the biome
/// layer: `before`/`after` are row-major `u8`, length `w · h`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BiomePatch {
    pub coord: (i32, i32),
    pub i0: u32,
    pub j0: u32,
    pub w: u32,
    pub h: u32,
    pub before: Vec<u8>,
    pub after: Vec<u8>,
}

impl BiomePatch {
    #[inline]
    pub fn sample_count(&self) -> usize {
        (self.w * self.h) as usize
    }
}

/// The reversible record a biome-paint stroke returns — the biome analogue of
/// [`SplatDelta`](crate::SplatDelta). Apply it for redo, revert it for undo.
///
/// `materialized_tiles` records tiles whose biome buffer was on the **sparse
/// unpainted default** before the stroke and got materialized to paint into: on
/// revert they are dropped back, so the terrain re-serializes byte-identically to
/// its unpainted form.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BiomeDelta {
    /// One dense sub-rect patch per touched tile (deterministic tile order).
    pub patches: Vec<BiomePatch>,
    /// Tiles the stroke materialized from the sparse default (reset on revert).
    /// Sorted.
    pub materialized_tiles: Vec<(i32, i32)>,
}

impl BiomeDelta {
    /// `true` when the stroke changed no sample **and** materialized no tile —
    /// i.e. reverting it would do nothing at all.
    ///
    /// Both halves, following the P19.1 [`DataMapDelta`](crate::DataMapDelta)
    /// standard rather than the patch-only shortcut: `materialized_tiles` is what
    /// drops a buffer back to the sparse default, so it is real undo work even if
    /// the patch list were somehow empty.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty() && self.materialized_tiles.is_empty()
    }

    /// Total samples stored across all patches (bounding-rect inclusive).
    pub fn sample_count(&self) -> usize {
        self.patches.iter().map(BiomePatch::sample_count).sum()
    }

    /// Approximate heap footprint of the stored before/after buffers, in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.sample_count() * 2
    }
}

/// `coord → (i, j) → first-touch before id`.
type TouchedBiomes = BTreeMap<(i32, i32), BTreeMap<(u32, u32), u8>>;

/// Accumulates the *first-touch* `before` id per changed sample across one or
/// more dabs, then compresses the result into per-tile dense patches — the biome
/// twin of `SplatDeltaBuilder`.
#[derive(Default)]
pub(crate) struct BiomeDeltaBuilder {
    touched: TouchedBiomes,
    materialized: BTreeSet<(i32, i32)>,
}

impl BiomeDeltaBuilder {
    #[inline]
    fn touch(&mut self, coord: (i32, i32), i: u32, j: u32, before: u8) {
        self.touched
            .entry(coord)
            .or_default()
            .entry((i, j))
            .or_insert(before);
    }

    #[inline]
    fn mark_materialized(&mut self, coord: (i32, i32)) {
        self.materialized.insert(coord);
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.touched.is_empty() && self.materialized.is_empty()
    }

    /// Compress the accumulated touches into per-tile dense sub-rect patches,
    /// reading final (`after`) and filler values from `data`'s current tiles.
    fn finalize(&self, data: &TerrainData) -> BiomeDelta {
        let res = data.tile_resolution();
        let mut patches = Vec::with_capacity(self.touched.len());
        for (&coord, samples) in &self.touched {
            let Some(tile) = data.get_tile(coord) else {
                continue;
            };
            let mut i_min = u32::MAX;
            let mut i_max = 0u32;
            let mut j_min = u32::MAX;
            let mut j_max = 0u32;
            for &(i, j) in samples.keys() {
                i_min = i_min.min(i);
                i_max = i_max.max(i);
                j_min = j_min.min(j);
                j_max = j_max.max(j);
            }
            let w = i_max - i_min + 1;
            let h = j_max - j_min + 1;
            let mut before = vec![DEFAULT_BIOME; (w * h) as usize];
            let mut after = vec![DEFAULT_BIOME; (w * h) as usize];
            for row in 0..h {
                for col in 0..w {
                    let i = i_min + col;
                    let j = j_min + row;
                    let idx = (row * w + col) as usize;
                    let cur = tile.biome_sample(res, i, j);
                    after[idx] = cur;
                    before[idx] = samples.get(&(i, j)).copied().unwrap_or(cur);
                }
            }
            patches.push(BiomePatch {
                coord,
                i0: i_min,
                j0: j_min,
                w,
                h,
                before,
                after,
            });
        }
        BiomeDelta {
            patches,
            materialized_tiles: self.materialized.iter().copied().collect(),
        }
    }
}

// ── brush application ─────────────────────────────────────────────────────────

/// Apply one biome dab, writing `biome` into every claimed sample, and return the
/// reversible [`BiomeDelta`].
///
/// Passing [`UNASSIGNED_BIOME`] (`0`) is the **eraser** — a legitimate, useful
/// operation (unpaint this region), and the reason the write path never special-
/// cases the default value.
pub fn apply_paint_biome(data: &mut TerrainData, biome: u8, params: BrushParams) -> BiomeDelta {
    let mut builder = BiomeDeltaBuilder::default();
    apply_biome_op(data, biome, &params, &mut builder);
    builder.finalize(data)
}

/// Clamped `[lo, hi]` local-sample range along one axis intersecting
/// `[minw, maxw]` — identical to the splat/height brushes' axis clamp.
fn axis_range(minw: f64, maxw: f64, origin: f64, mps: f64, res: u32) -> Option<(u32, u32)> {
    let lo = ((minw - origin) / mps).ceil().max(0.0);
    let hi = ((maxw - origin) / mps).floor().min((res - 1) as f64);
    if lo > hi {
        None
    } else {
        Some((lo as u32, hi as u32))
    }
}

/// Drive the biome write across every **existing** tile the brush overlaps
/// (painting never authors tiles). A sample on a shared tile edge is written in
/// every tile that owns it — like the height and splat brushes — so a biome
/// boundary stays seamless across tile seams.
fn apply_biome_op(
    data: &mut TerrainData,
    biome: u8,
    params: &BrushParams,
    builder: &mut BiomeDeltaBuilder,
) {
    let (min, max) = params.aabb();
    let res = data.tile_resolution();
    let mps = data.meters_per_sample();
    let c0 = data.tile_coord_of(min.x, min.y);
    let c1 = data.tile_coord_of(max.x, max.y);
    let (cx, cz, radius, falloff, strength) = (
        params.center.x,
        params.center.y,
        params.radius,
        params.falloff,
        params.strength,
    );

    for tz in c0.1..=c1.1 {
        for tx in c0.0..=c1.0 {
            let coord = (tx, tz);
            if !data.has_tile(coord) {
                continue; // paint only touches existing terrain
            }
            let o = data.tile_origin_xz(coord);
            let Some((i_lo, i_hi)) = axis_range(min.x, max.x, o.x, mps, res) else {
                continue;
            };
            let Some((j_lo, j_hi)) = axis_range(min.y, max.y, o.y, mps, res) else {
                continue;
            };

            // Gather first (immutable read), then commit — the splat brush's
            // shape, so a dab that changes nothing never materializes a buffer.
            let was_default = data.get_tile(coord).is_some_and(|t| t.biomes_are_default());
            let writes: Vec<(u32, u32, u8)> = {
                let tile = data.get_tile(coord).expect("has_tile checked");
                let mut ws = Vec::new();
                for j in j_lo..=j_hi {
                    let wz = o.y + j as f64 * mps;
                    for i in i_lo..=i_hi {
                        let wx = o.x + i as f64 * mps;
                        let d = ((wx - cx).powi(2) + (wz - cz).powi(2)).sqrt();
                        if !biome_claims(falloff.weight(d, radius), strength) {
                            continue;
                        }
                        let old = tile.biome_sample(res, i, j);
                        if old != biome {
                            ws.push((i, j, old));
                        }
                    }
                }
                ws
            };

            if writes.is_empty() {
                continue;
            }
            if was_default {
                builder.mark_materialized(coord);
            }
            let tile = data.get_tile_mut(coord).expect("has_tile checked");
            for (i, j, before) in writes {
                builder.touch(coord, i, j, before);
                tile.set_biome_sample(res, i, j, biome);
            }
        }
    }
}

/// A biome-paint **stroke**: the accumulated dabs of one mouse-down→up gesture,
/// merged into a single [`BiomeDelta`] — the biome twin of
/// [`SplatStroke`](crate::SplatStroke).
///
/// The target biome is captured at [`begin`](Self::begin), so changing the
/// toolbar's selection mid-drag cannot corrupt an in-flight stroke.
#[derive(Default)]
pub struct BiomeStroke {
    builder: BiomeDeltaBuilder,
    biome: u8,
}

impl BiomeStroke {
    /// Begin an empty stroke painting `biome` ([`UNASSIGNED_BIOME`] erases).
    pub fn begin(biome: u8) -> Self {
        Self {
            builder: BiomeDeltaBuilder::default(),
            biome,
        }
    }

    /// `true` before any dab has changed a sample.
    pub fn is_empty(&self) -> bool {
        self.builder.is_empty()
    }

    /// The biome this stroke paints.
    pub fn biome(&self) -> u8 {
        self.biome
    }

    /// `true` when this stroke erases rather than assigns.
    pub fn is_eraser(&self) -> bool {
        self.biome == UNASSIGNED_BIOME
    }

    /// Apply one dab into the stroke, mutating `data` and accumulating the delta.
    pub fn add_dab(&mut self, data: &mut TerrainData, params: BrushParams) {
        apply_biome_op(data, self.biome, &params, &mut self.builder);
    }

    /// Finish the stroke, producing the merged [`BiomeDelta`]. Reads the final
    /// (`after`) ids from `data`.
    pub fn finish(self, data: &TerrainData) -> BiomeDelta {
        self.builder.finalize(data)
    }
}

impl TerrainData {
    /// Apply (redo) a [`BiomeDelta`]: materialize each targeted tile's biome
    /// buffer if needed, then write each patch's `after` ids. Byte-identical to
    /// the state right after the stroke that produced `delta`.
    pub fn apply_biome_delta(&mut self, delta: &BiomeDelta) {
        let res = self.tile_resolution();
        for patch in &delta.patches {
            let Some(tile) = self.get_tile_mut(patch.coord) else {
                continue;
            };
            tile.ensure_biomes(res);
            write_biome_patch(tile, res, patch, &patch.after);
        }
    }

    /// Revert (undo) a [`BiomeDelta`]: write each patch's `before` ids, then reset
    /// any tile the stroke materialized back to the sparse default — so the
    /// terrain returns byte-identical to before the stroke.
    pub fn revert_biome_delta(&mut self, delta: &BiomeDelta) {
        let res = self.tile_resolution();
        let materialized: BTreeSet<(i32, i32)> = delta.materialized_tiles.iter().copied().collect();
        for patch in &delta.patches {
            // A materialized tile is about to be cleared wholesale — skip writing
            // its (all-default) `before` buffer.
            if materialized.contains(&patch.coord) {
                continue;
            }
            let Some(tile) = self.get_tile_mut(patch.coord) else {
                continue;
            };
            write_biome_patch(tile, res, patch, &patch.before);
        }
        for &coord in &delta.materialized_tiles {
            if let Some(tile) = self.get_tile_mut(coord) {
                tile.clear_biomes();
            }
        }
    }

    /// `true` when **no** authored tile carries biome ids — i.e. nothing has ever
    /// been biome-painted, and the layer costs zero per-sample bytes.
    pub fn biomes_are_default(&self) -> bool {
        self.tiles().all(|(_, t)| t.biomes_are_default())
    }

    /// The biome id at world `(x, z)`, resolving the sparse default. `None`
    /// outside the authored extent.
    ///
    /// **Nearest sample, never interpolated** — an id is categorical, so the
    /// midpoint between biome 3 and biome 7 is one of the two, not "biome 5".
    pub fn biome_at(&self, world_xz: glam::DVec2) -> Option<u8> {
        let res = self.tile_resolution();
        let span = self.tile_span();
        let mps = self.meters_per_sample();
        // Same shared-edge resolution the height query uses: prefer the floored
        // tile, fall back to the previous tile's far edge exactly on a seam.
        let axis = |w: f64| -> [(i32, f64); 2] {
            let th = (w / span).floor() as i32;
            let u = (w - th as f64 * span) / mps;
            if u <= 1e-9 {
                [(th, u), (th - 1, (res - 1) as f64)]
            } else {
                [(th, u), (th, u)]
            }
        };
        let xs = axis(world_xz.x);
        let zs = axis(world_xz.y);
        for &(tx, u) in xs.iter().take(if xs[0].0 == xs[1].0 { 1 } else { 2 }) {
            for &(tz, v) in zs.iter().take(if zs[0].0 == zs[1].0 { 1 } else { 2 }) {
                if let Some(tile) = self.get_tile((tx, tz)) {
                    let i = u.round().clamp(0.0, (res - 1) as f64) as u32;
                    let j = v.round().clamp(0.0, (res - 1) as f64) as u32;
                    return Some(tile.biome_sample(res, i, j));
                }
            }
        }
        None
    }
}

/// Blit a `w · h` row-major id buffer into `tile`'s sample rectangle.
fn write_biome_patch(tile: &mut crate::TerrainTile, res: u32, patch: &BiomePatch, buf: &[u8]) {
    for row in 0..patch.h {
        for col in 0..patch.w {
            let idx = (row * patch.w + col) as usize;
            tile.set_biome_sample(res, patch.i0 + col, patch.j0 + row, buf[idx]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brush::{BrushParams, Falloff};
    use glam::DVec2;

    fn terrain() -> TerrainData {
        let mut t = TerrainData::new(8, 1.0);
        t.author_tile((0, 0), |_, _| 0.0);
        t
    }

    fn dab(t: &TerrainData, frac: f64, strength: f64) -> BrushParams {
        let span = t.tile_span();
        BrushParams::new(DVec2::new(span * 0.5, span * 0.5), span * frac, strength)
    }

    // ── the hard-edge rule ───────────────────────────────────────────────────

    /// The claim predicate is monotone in both arguments, total on garbage input,
    /// and never claims at zero strength.
    #[test]
    fn the_claim_rule_is_monotone_total_and_zero_strength_is_a_no_op() {
        // Zero (and negative) strength claims nothing, anywhere.
        for w in [0.0, 0.25, 0.5, 1.0, 7.0] {
            assert!(!biome_claims(w, 0.0), "w={w}");
            assert!(!biome_claims(w, -1.0), "w={w}");
        }
        // Zero weight (outside the radius) is never claimed, at any strength.
        for s in [0.01, 0.5, 1.0, 9.0] {
            assert!(!biome_claims(0.0, s), "s={s}");
        }
        // Full strength claims every in-radius sample — the stamp.
        for w in [1e-9, 0.1, 0.5, 1.0] {
            assert!(biome_claims(w, 1.0), "w={w}");
        }
        // Monotone in strength at fixed weight, and in weight at fixed strength.
        let mut claimed_before = false;
        for k in 0..=20 {
            let s = k as f64 / 20.0;
            let c = biome_claims(0.5, s);
            assert!(c >= claimed_before, "strength must be monotone at s={s}");
            claimed_before = c;
        }
        let mut claimed_before = false;
        for k in 0..=20 {
            let w = k as f64 / 20.0;
            let c = biome_claims(w, 0.5);
            assert!(c >= claimed_before, "weight must be monotone at w={w}");
            claimed_before = c;
        }
        // The documented contour: strength 0.5 claims exactly weight >= 0.5.
        assert!(biome_claims(0.5, 0.5));
        assert!(!biome_claims(0.4999, 0.5));
    }

    /// **The edge really is hard.** Under a smooth falloff at full strength the
    /// painted region is a solid disk of one id — no partial values, no gradient —
    /// and strength shrinks that disk instead of fading it.
    #[test]
    fn strength_shrinks_the_disk_instead_of_fading_it() {
        // A fine lattice, so the four contours land on distinct sample counts
        // rather than being quantized together (res 8 is too coarse to tell
        // strength 0.5 from 0.25 apart at all).
        let res = 33;
        let mut counts = Vec::new();
        for strength in [1.0, 0.75, 0.5, 0.25] {
            let mut t = TerrainData::new(res, 1.0);
            t.author_tile((0, 0), |_, _| 0.0);
            let p = dab(&t, 0.4, strength);
            apply_paint_biome(&mut t, 4, p);
            let tile = t.get_tile((0, 0)).unwrap();
            let mut n = 0;
            for j in 0..res {
                for i in 0..res {
                    let b = tile.biome_sample(res, i, j);
                    assert!(
                        b == 4 || b == UNASSIGNED_BIOME,
                        "a biome brush may only write its own id or leave the default, got {b}"
                    );
                    n += (b == 4) as usize;
                }
            }
            counts.push(n);
        }
        assert!(counts[0] > 0, "full strength must paint something");
        for k in 1..counts.len() {
            assert!(
                counts[k] < counts[k - 1],
                "lower strength must claim strictly fewer samples: {counts:?}"
            );
        }
    }

    /// A [`Falloff::Plateau(1.0)`] brush is flat, so *every* strength above zero
    /// stamps the whole disk — the falloff curve is what makes strength shrink it.
    #[test]
    fn a_flat_falloff_stamps_the_whole_disk_at_any_strength() {
        let mut sizes = Vec::new();
        for strength in [1.0, 0.5, 0.05] {
            let mut t = terrain();
            let mut p = dab(&t, 0.4, strength);
            p.falloff = Falloff::Plateau(1.0);
            let d = apply_paint_biome(&mut t, 2, p);
            sizes.push(d.sample_count());
        }
        assert!(sizes[0] > 0);
        assert!(
            sizes.iter().all(|&n| n == sizes[0]),
            "a flat brush must be strength-independent: {sizes:?}"
        );
    }

    // ── determinism, undo, redo ──────────────────────────────────────────────

    /// Painting is a pure function of terrain + biome + params: two runs of the
    /// same stroke on the same terrain give byte-identical results and equal
    /// deltas.
    #[test]
    fn paint_is_deterministic() {
        let run = || {
            let mut t = terrain();
            let p = dab(&t, 0.45, 0.8);
            let d = apply_paint_biome(&mut t, 9, p);
            (serde_json::to_string(&t).unwrap(), d)
        };
        let (a, da) = run();
        let (b, db) = run();
        assert_eq!(a, b, "the terrain bytes must be reproducible");
        assert_eq!(da, db, "the delta must be reproducible");
    }

    /// Undo is **byte-identical**, including dropping the materialized buffer back
    /// to the sparse default; redo reproduces the post-stroke bytes exactly.
    #[test]
    fn dab_then_revert_is_byte_identical_and_redo_round_trips() {
        let mut t = terrain();
        let before = serde_json::to_string(&t).unwrap();
        assert!(t.biomes_are_default());

        let p = dab(&t, 0.4, 1.0);
        let delta = apply_paint_biome(&mut t, 3, p);
        assert!(!delta.is_empty());
        assert!(delta.memory_bytes() > 0);
        assert!(!t.get_tile((0, 0)).unwrap().biomes_are_default());
        assert_eq!(delta.materialized_tiles, vec![(0, 0)]);
        let painted = serde_json::to_string(&t).unwrap();

        t.revert_biome_delta(&delta);
        assert!(t.get_tile((0, 0)).unwrap().biomes_are_default());
        assert_eq!(
            serde_json::to_string(&t).unwrap(),
            before,
            "undo must be exact"
        );

        t.apply_biome_delta(&delta);
        assert_eq!(
            serde_json::to_string(&t).unwrap(),
            painted,
            "redo must be exact"
        );
    }

    /// A stroke merges its dabs: `before` is the FIRST touch, `after` the final
    /// value, so one revert walks all the way back.
    #[test]
    fn a_stroke_merges_dabs_and_reverts_in_one_step() {
        let mut t = terrain();
        let before = serde_json::to_string(&t).unwrap();
        let span = t.tile_span();

        // First paint biome 1 over a wide area, then biome 2 over part of it, in
        // one stroke — the second dab's `before` must be biome 1, not the default.
        let mut stroke = BiomeStroke::begin(1);
        for k in 0..4 {
            let c = DVec2::new(span * 0.3 + k as f64, span * 0.5);
            stroke.add_dab(&mut t, BrushParams::new(c, 2.5, 1.0));
        }
        assert!(!stroke.is_empty());
        assert_eq!(stroke.biome(), 1);
        let first = stroke.finish(&t);

        let mut second = BiomeStroke::begin(2);
        second.add_dab(
            &mut t,
            BrushParams::new(DVec2::new(span * 0.3, span * 0.5), 1.5, 1.0),
        );
        let second = second.finish(&t);
        assert!(
            second.patches.iter().any(|p| p.before.contains(&1)),
            "the second stroke must record biome 1 as its `before`"
        );

        t.revert_biome_delta(&second);
        t.revert_biome_delta(&first);
        assert_eq!(serde_json::to_string(&t).unwrap(), before);
        assert!(t.biomes_are_default());
    }

    /// The eraser (`UNASSIGNED_BIOME`) is a first-class stroke: it reverts painted
    /// samples to the reserved default and undoes exactly.
    #[test]
    fn painting_the_reserved_id_erases() {
        let mut t = terrain();
        let p = dab(&t, 0.45, 1.0);
        let painted = apply_paint_biome(&mut t, 6, p);
        assert!(!painted.is_empty());
        let after_paint = serde_json::to_string(&t).unwrap();

        let mut eraser = BiomeStroke::begin(UNASSIGNED_BIOME);
        assert!(eraser.is_eraser());
        eraser.add_dab(&mut t, p);
        let erased = eraser.finish(&t);
        assert!(
            !erased.is_empty(),
            "erasing a painted region is a real edit"
        );
        assert_eq!(
            t.get_tile((0, 0)).unwrap().biome_sample(8, 4, 4),
            UNASSIGNED_BIOME
        );
        // The buffer stays materialized (the tile was already painted), so this is
        // NOT the sparse default — erasing is an edit, not a rewind.
        assert!(!t.get_tile((0, 0)).unwrap().biomes_are_default());
        assert!(
            erased.materialized_tiles.is_empty(),
            "an already-painted tile is not materialized again"
        );

        t.revert_biome_delta(&erased);
        assert_eq!(serde_json::to_string(&t).unwrap(), after_paint);
    }

    // ── invariants shared with the splat brush ───────────────────────────────

    #[test]
    fn paint_does_not_author_new_tiles() {
        let mut t = terrain();
        let n = t.tile_count();
        let delta = apply_paint_biome(
            &mut t,
            1,
            BrushParams::new(DVec2::new(10_000.0, 10_000.0), 5.0, 1.0),
        );
        assert!(delta.is_empty());
        assert_eq!(t.tile_count(), n, "biome paint must not author tiles");
        assert!(t.biomes_are_default(), "…and must not materialize anything");
    }

    /// A repainted-with-the-same-id dab changes nothing — so it records nothing
    /// and, crucially, never materializes a fresh buffer for a no-op.
    #[test]
    fn repainting_the_same_id_is_a_no_op() {
        let mut t = terrain();
        let p = dab(&t, 0.4, 1.0);
        assert!(!apply_paint_biome(&mut t, 5, p).is_empty());
        let bytes = serde_json::to_string(&t).unwrap();
        let again = apply_paint_biome(&mut t, 5, p);
        assert!(
            again.is_empty(),
            "a second identical dab must record nothing"
        );
        assert_eq!(serde_json::to_string(&t).unwrap(), bytes);
    }

    #[test]
    fn paint_across_a_tile_seam_is_seamless() {
        let mut t = TerrainData::new(8, 1.0);
        t.author_tile((0, 0), |_, _| 0.0);
        t.author_tile((1, 0), |_, _| 0.0);
        let span = t.tile_span();
        apply_paint_biome(
            &mut t,
            7,
            BrushParams::new(DVec2::new(span, span * 0.5), 3.0, 1.0),
        );
        let res = t.tile_resolution();
        for j in 0..res {
            let left = t.get_tile((0, 0)).unwrap().biome_sample(res, res - 1, j);
            let right = t.get_tile((1, 0)).unwrap().biome_sample(res, 0, j);
            assert_eq!(left, right, "seam mismatch at j={j}: {left} vs {right}");
        }
    }

    /// **Sparse is free**, priced in bytes rather than asserted: an unpainted
    /// terrain leaks no `biomes` key at all in the human-readable form and costs
    /// exactly one length byte per tile in bincode; a materialized tile costs
    /// exactly its dense `u8` buffer.
    #[test]
    fn the_unpainted_default_is_free() {
        let mut t = TerrainData::new(5, 1.0);
        t.author_tile((0, 0), |_, _| 0.0);
        t.author_tile((1, 0), |_, _| 0.0);
        assert!(t.biomes_are_default());
        let json = serde_json::to_string(&t).unwrap();
        assert!(!json.contains("biomes"), "unpainted terrain leaked: {json}");

        let cfg = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&t, cfg).unwrap();
        let mut painted = t.clone();
        painted.get_tile_mut((0, 0)).unwrap().ensure_biomes(5);
        let big = bincode::serde::encode_to_vec(&painted, cfg).unwrap();
        assert_eq!(
            big.len() - bytes.len(),
            5 * 5,
            "materializing one tile must cost exactly its dense u8 buffer"
        );
    }

    /// A biome edit dirties its tile so the P16.4b save write-back picks it up —
    /// the biome twin of the P19.1 data-map assertion. Biome ids ride inside the
    /// tile blob, so `TerrainEdits::from_dirty` carries them for free; asserting
    /// it here means a future back door that writes ids without `get_tile_mut`
    /// fails loudly instead of silently dropping saves.
    #[test]
    fn a_biome_edit_dirties_the_tile_for_write_back() {
        let mut t = terrain();
        t.clear_dirty();
        assert!(!t.has_dirty_tiles(), "the fixture starts clean");

        let p = dab(&t, 0.4, 1.0);
        apply_paint_biome(&mut t, 2, p);
        let dirty = t.dirty_tiles();
        assert!(!dirty.is_empty(), "a biome edit must schedule a write-back");

        let edits = crate::writeback::TerrainEdits::from_dirty(&t, &dirty);
        assert!(!edits.is_empty());
        assert!(
            edits
                .changed
                .values()
                .all(|tile| !tile.biomes_are_default()),
            "the staged tiles must carry the ids they were dirtied for"
        );
    }

    /// `biome_at` reads the nearest sample and never interpolates.
    #[test]
    fn biome_at_is_nearest_and_never_interpolates() {
        let mut t = terrain();
        t.get_tile_mut((0, 0))
            .unwrap()
            .set_biome_sample(8, 3, 3, 10);
        t.get_tile_mut((0, 0))
            .unwrap()
            .set_biome_sample(8, 4, 3, 20);
        // Sample centres.
        assert_eq!(t.biome_at(DVec2::new(3.0, 3.0)), Some(10));
        assert_eq!(t.biome_at(DVec2::new(4.0, 3.0)), Some(20));
        // Between them: one of the two, never a blend.
        let mid = t.biome_at(DVec2::new(3.4, 3.0)).unwrap();
        assert_eq!(mid, 10, "nearest, not averaged");
        assert_eq!(t.biome_at(DVec2::new(3.6, 3.0)), Some(20));
        // Off the terrain.
        assert_eq!(t.biome_at(DVec2::new(1e6, 1e6)), None);
        // An unpainted sample reads the reserved default, not None.
        assert_eq!(t.biome_at(DVec2::new(0.0, 0.0)), Some(UNASSIGNED_BIOME));
    }
}
