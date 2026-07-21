//! Splat-layer **painting** (P10.4): the weights-layer twin of the height
//! [`brush`](crate::brush)/[`delta`](crate::delta) machinery.
//!
//! Where a sculpt brush edits per-sample `f32` heights, a **paint** brush edits
//! per-sample `[u8; 4]` splat weights — the RGBA blend of the four terrain
//! material layers, normalized to sum ≈ 255. The shapes deliberately mirror the
//! height side one storey up:
//!
//! * [`apply_paint`] is the weights analogue of [`apply_brush`](crate::apply_brush):
//!   one dab returns a reversible [`SplatDelta`].
//! * [`SplatStroke`] is the analogue of [`Stroke`](crate::Stroke): it merges the
//!   dabs of one mouse-down→up gesture into a single [`SplatDelta`] (`before` =
//!   first touch, `after` = final).
//! * [`SplatDelta`] / [`SplatPatch`] mirror [`HeightDelta`](crate::HeightDelta) /
//!   [`TilePatch`](crate::TilePatch): dense per-tile sub-rect patches of `[u8; 4]`
//!   the editor's undo stack stores and [`TerrainData::revert_splat_delta`]
//!   replays.
//!
//! Painting **never authors tiles** (like Smooth/Flatten): weights are only
//! written where terrain already exists. It *does* [materialize] a tile's sparse
//! weight buffer on first touch; the delta records which tiles it materialized
//! (`materialized_tiles`), so undo can drop them back to the byte-stable sparse
//! default — a paint that touched a fresh terrain undoes to byte-identical
//! unpainted space.
//!
//! [materialize]: crate::TerrainTile::ensure_weights

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::brush::BrushParams;
use crate::data::TerrainData;
use crate::tile::DEFAULT_WEIGHT;

/// Number of splat layers packed into one `[u8; 4]` weight sample.
pub const SPLAT_LAYERS: usize = 4;

/// Paint one sample's splat weight toward `layer` by fractional `flow` (`0` = no
/// change, `1` = fully this layer), renormalizing the other channels so the four
/// weights keep summing to exactly `255`.
///
/// The other layers keep their **relative** proportions (a sample that was half
/// layer 1 / half layer 2 stays balanced between them as layer 0 is painted in).
/// Pure and deterministic — the paint math the brush and its tests share.
pub fn paint_weight(cur: [u8; 4], layer: u8, flow: f64) -> [u8; 4] {
    let l = (layer as usize) % SPLAT_LAYERS;
    let flow = flow.clamp(0.0, 1.0);
    if flow <= 0.0 {
        return cur;
    }
    const T: f64 = 255.0;
    let raw = [cur[0] as f64, cur[1] as f64, cur[2] as f64, cur[3] as f64];
    let sum: f64 = raw.iter().sum();
    // Normalize to T (defensive — inputs should already sum to ~255, but a hand-
    // authored or drifted sample must not skew the blend). All-zero → pure layer.
    let mut w = if sum > 0.0 {
        [
            raw[0] * T / sum,
            raw[1] * T / sum,
            raw[2] * T / sum,
            raw[3] * T / sum,
        ]
    } else {
        let mut z = [0.0; 4];
        z[l] = T;
        z
    };
    let new_l = w[l] + (T - w[l]) * flow;
    let others_sum = T - w[l];
    let remaining = (T - new_l).max(0.0);
    if others_sum > 0.0 {
        let scale = remaining / others_sum;
        for (k, wk) in w.iter_mut().enumerate() {
            if k != l {
                *wk *= scale;
            }
        }
    }
    w[l] = new_l;
    quantize_to_255(w, l)
}

/// Round four real weights to `u8` with an **exact** sum of `255`, absorbing the
/// rounding residual on `primary` (the painted layer — the channel whose intent
/// is authoritative). Guarantees the normalization invariant channel-for-channel.
fn quantize_to_255(w: [f64; 4], primary: usize) -> [u8; 4] {
    let mut out = [0i32; 4];
    for (k, wk) in w.iter().enumerate() {
        out[k] = wk.round().clamp(0.0, 255.0) as i32;
    }
    let total: i32 = out.iter().sum();
    out[primary] = (out[primary] + (255 - total)).clamp(0, 255);
    // If clamping the primary couldn't absorb the whole residual (e.g. it was
    // already 0/255), spread the rest across the channels deterministically.
    let mut total: i32 = out.iter().sum();
    let mut guard = 0;
    while total != 255 && guard < 16 {
        let idx = guard % SPLAT_LAYERS;
        let step = if total < 255 { 1 } else { -1 };
        let nv = (out[idx] + step).clamp(0, 255);
        if nv != out[idx] {
            out[idx] = nv;
            total += step;
        }
        guard += 1;
    }
    [out[0] as u8, out[1] as u8, out[2] as u8, out[3] as u8]
}

// ── delta (undo record) ──────────────────────────────────────────────────────

/// A dense before/after patch over one tile's `[i0, i0+w) × [j0, j0+h)` sample
/// rectangle of splat weights. Mirrors [`TilePatch`](crate::TilePatch) at the
/// weights layer: `before`/`after` are row-major `[u8; 4]`, length `w · h`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplatPatch {
    pub coord: (i32, i32),
    pub i0: u32,
    pub j0: u32,
    pub w: u32,
    pub h: u32,
    pub before: Vec<[u8; 4]>,
    pub after: Vec<[u8; 4]>,
}

impl SplatPatch {
    #[inline]
    pub fn sample_count(&self) -> usize {
        (self.w * self.h) as usize
    }
}

/// The reversible record a paint stroke returns — the weights analogue of
/// [`HeightDelta`](crate::HeightDelta). Apply it for redo, revert it for undo.
///
/// `materialized_tiles` records tiles whose weight buffer was on the **sparse
/// default** (uniform layer 0) before the stroke and got materialized to paint
/// into: on revert they are dropped back to the sparse default so the terrain
/// re-serializes byte-identically to its unpainted form.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SplatDelta {
    /// One dense sub-rect patch per touched tile (deterministic tile order).
    pub patches: Vec<SplatPatch>,
    /// Tiles the stroke materialized from the sparse default (reset on revert).
    /// Sorted.
    pub materialized_tiles: Vec<(i32, i32)>,
}

impl SplatDelta {
    /// `true` when the stroke changed no samples.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }

    /// Total samples stored across all patches (bounding-rect inclusive).
    pub fn sample_count(&self) -> usize {
        self.patches.iter().map(SplatPatch::sample_count).sum()
    }
}

/// `coord → (i, j) → first-touch before weight` — the sparse accumulator keyed
/// per tile then per sample.
type TouchedWeights = BTreeMap<(i32, i32), BTreeMap<(u32, u32), [u8; 4]>>;

/// Accumulates the *first-touch* `before` weight per changed sample across one or
/// more paint dabs, then compresses the result into per-tile dense patches — the
/// weights twin of [`DeltaBuilder`](crate::delta) internals.
#[derive(Default)]
pub(crate) struct SplatDeltaBuilder {
    touched: TouchedWeights,
    /// Tiles materialized from the sparse default during the stroke.
    materialized: BTreeSet<(i32, i32)>,
}

impl SplatDeltaBuilder {
    #[inline]
    fn touch(&mut self, coord: (i32, i32), i: u32, j: u32, before: [u8; 4]) {
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
        self.touched.is_empty()
    }

    /// Compress the accumulated touches into per-tile dense sub-rect patches,
    /// reading final (`after`) and filler values from `data`'s current tiles.
    fn finalize(&self, data: &TerrainData) -> SplatDelta {
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
            let mut before = vec![DEFAULT_WEIGHT; (w * h) as usize];
            let mut after = vec![DEFAULT_WEIGHT; (w * h) as usize];
            for row in 0..h {
                for col in 0..w {
                    let i = i_min + col;
                    let j = j_min + row;
                    let idx = (row * w + col) as usize;
                    let cur = tile.weight_sample(res, i, j);
                    after[idx] = cur;
                    before[idx] = samples.get(&(i, j)).copied().unwrap_or(cur);
                }
            }
            patches.push(SplatPatch {
                coord,
                i0: i_min,
                j0: j_min,
                w,
                h,
                before,
                after,
            });
        }
        SplatDelta {
            patches,
            materialized_tiles: self.materialized.iter().copied().collect(),
        }
    }
}

// ── brush application ─────────────────────────────────────────────────────────

/// Apply one paint dab to `layer` of `data`, returning the reversible
/// [`SplatDelta`]. The weights twin of [`apply_brush`](crate::apply_brush).
pub fn apply_paint(data: &mut TerrainData, layer: u8, params: BrushParams) -> SplatDelta {
    let mut builder = SplatDeltaBuilder::default();
    apply_paint_op(data, layer, &params, &mut builder);
    builder.finalize(data)
}

/// Clamped `[lo, hi]` local-sample range along one axis intersecting `[minw,
/// maxw]` (identical to the height brush's axis clamp).
fn axis_range(minw: f64, maxw: f64, origin: f64, mps: f64, res: u32) -> Option<(u32, u32)> {
    let lo = ((minw - origin) / mps).ceil().max(0.0);
    let hi = ((maxw - origin) / mps).floor().min((res - 1) as f64);
    if lo > hi {
        None
    } else {
        Some((lo as u32, hi as u32))
    }
}

/// Drive the paint op across every **existing** tile the brush overlaps (paint
/// never authors tiles). For each in-radius sample, flow the splat weight toward
/// `layer` by `falloff · strength`, renormalizing the other channels
/// ([`paint_weight`]). Seams: a sample on a shared tile edge is written in every
/// tile that owns it (like the height brush), so painting stays seamless across
/// tile boundaries.
fn apply_paint_op(
    data: &mut TerrainData,
    layer: u8,
    params: &BrushParams,
    builder: &mut SplatDeltaBuilder,
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

            // Gather the changed samples first (immutable read), then commit.
            let was_default = data
                .get_tile(coord)
                .is_some_and(|t| t.weights_are_default());
            let writes: Vec<(u32, u32, [u8; 4], [u8; 4])> = {
                let tile = data.get_tile(coord).expect("has_tile checked");
                let mut ws = Vec::new();
                for j in j_lo..=j_hi {
                    let wz = o.y + j as f64 * mps;
                    for i in i_lo..=i_hi {
                        let wx = o.x + i as f64 * mps;
                        let d = ((wx - cx).powi(2) + (wz - cz).powi(2)).sqrt();
                        let fw = falloff.weight(d, radius);
                        if fw <= 0.0 {
                            continue;
                        }
                        let old = tile.weight_sample(res, i, j);
                        let new = paint_weight(old, layer, fw * strength);
                        if new != old {
                            ws.push((i, j, old, new));
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
            for (i, j, before, new) in writes {
                builder.touch(coord, i, j, before);
                tile.set_weight_sample(res, i, j, new);
            }
        }
    }
}

/// A paint **stroke**: the accumulated dabs of one mouse-down→up gesture, merged
/// into a single [`SplatDelta`] — the weights twin of [`Stroke`](crate::Stroke).
#[derive(Default)]
pub struct SplatStroke {
    builder: SplatDeltaBuilder,
    layer: u8,
}

impl SplatStroke {
    /// Begin an empty stroke painting `layer`.
    pub fn begin(layer: u8) -> Self {
        Self {
            builder: SplatDeltaBuilder::default(),
            layer: layer % SPLAT_LAYERS as u8,
        }
    }

    /// `true` before any dab has changed a sample.
    pub fn is_empty(&self) -> bool {
        self.builder.is_empty()
    }

    /// The layer this stroke paints.
    pub fn layer(&self) -> u8 {
        self.layer
    }

    /// Apply one dab into the stroke, mutating `data` and accumulating the delta.
    pub fn add_dab(&mut self, data: &mut TerrainData, params: BrushParams) {
        apply_paint_op(data, self.layer, &params, &mut self.builder);
    }

    /// Finish the stroke, producing the merged [`SplatDelta`]. Reads the final
    /// (`after`) weights from `data`.
    pub fn finish(self, data: &TerrainData) -> SplatDelta {
        self.builder.finalize(data)
    }
}

impl TerrainData {
    /// Apply (redo) a [`SplatDelta`]: materialize each targeted tile's weight
    /// buffer if needed, then write each patch's `after` weights. Byte-identical
    /// to the state right after the stroke that produced `delta`.
    pub fn apply_splat_delta(&mut self, delta: &SplatDelta) {
        let res = self.tile_resolution();
        for patch in &delta.patches {
            let Some(tile) = self.get_tile_mut(patch.coord) else {
                continue;
            };
            tile.ensure_weights(res);
            write_splat_patch(tile, res, patch, &patch.after);
        }
    }

    /// Revert (undo) a [`SplatDelta`]: write each patch's `before` weights, then
    /// reset any tile the stroke materialized back to the sparse default — so the
    /// terrain returns byte-identical to before the stroke (an unpainted tile has
    /// no stored weights again).
    pub fn revert_splat_delta(&mut self, delta: &SplatDelta) {
        let res = self.tile_resolution();
        let materialized: BTreeSet<(i32, i32)> = delta.materialized_tiles.iter().copied().collect();
        for patch in &delta.patches {
            // A materialized tile is about to be cleared wholesale — skip writing
            // its `before` (all default) buffer.
            if materialized.contains(&patch.coord) {
                continue;
            }
            let Some(tile) = self.get_tile_mut(patch.coord) else {
                continue;
            };
            write_splat_patch(tile, res, patch, &patch.before);
        }
        for &coord in &delta.materialized_tiles {
            if let Some(tile) = self.get_tile_mut(coord) {
                tile.clear_weights();
            }
        }
    }
}

/// Blit a `w · h` row-major weight buffer into `tile`'s sample rectangle.
fn write_splat_patch(tile: &mut crate::TerrainTile, res: u32, patch: &SplatPatch, buf: &[[u8; 4]]) {
    for row in 0..patch.h {
        for col in 0..patch.w {
            let idx = (row * patch.w + col) as usize;
            tile.set_weight_sample(res, patch.i0 + col, patch.j0 + row, buf[idx]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brush::BrushParams;
    use glam::DVec2;

    fn painted_terrain() -> TerrainData {
        let mut t = TerrainData::new(8, 1.0);
        t.author_tile((0, 0), |_, _| 0.0);
        t
    }

    #[test]
    fn paint_full_flow_saturates_target_layer() {
        // Flow 1 → the sample becomes 100% the painted layer, others zero.
        assert_eq!(paint_weight(DEFAULT_WEIGHT, 1, 1.0), [0, 255, 0, 0]);
        assert_eq!(paint_weight([100, 100, 55, 0], 2, 1.0), [0, 0, 255, 0]);
    }

    #[test]
    fn paint_zero_flow_is_identity() {
        assert_eq!(paint_weight([64, 64, 64, 63], 3, 0.0), [64, 64, 64, 63]);
    }

    #[test]
    fn paint_partial_flow_renormalizes_to_255() {
        // Half flow onto layer 1 from a pure-layer-0 sample: sum stays 255.
        for flow in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let w = paint_weight(DEFAULT_WEIGHT, 1, flow);
            let sum: u32 = w.iter().map(|&c| c as u32).sum();
            assert_eq!(sum, 255, "flow {flow} broke normalization: {w:?}");
            // The painted channel rose, layer 0 fell.
            assert!(w[1] > 0 && w[0] < 255);
        }
    }

    #[test]
    fn paint_preserves_other_layer_ratios() {
        // Start 50/50 between layers 1 and 2, paint some layer 0: 1 and 2 keep
        // their 1:1 ratio as they shrink.
        let start = [0, 128, 127, 0];
        let w = paint_weight(start, 0, 0.4);
        assert_eq!(w.iter().map(|&c| c as u32).sum::<u32>(), 255);
        assert!(w[0] > 0);
        assert!((w[1] as i32 - w[2] as i32).abs() <= 1, "ratio drift: {w:?}");
    }

    #[test]
    fn paint_flow_is_clamped() {
        // Over-unit / negative flow clamps to [0, 1].
        assert_eq!(paint_weight(DEFAULT_WEIGHT, 1, 5.0), [0, 255, 0, 0]);
        assert_eq!(paint_weight([50, 50, 50, 105], 2, -3.0), [50, 50, 50, 105]);
    }

    #[test]
    fn paint_dab_then_revert_is_byte_identical() {
        let mut t = painted_terrain();
        let before = serde_json::to_string(&t).unwrap();
        let span = t.tile_span();
        let params = BrushParams::new(DVec2::new(span * 0.5, span * 0.5), span * 0.4, 1.0);
        let delta = apply_paint(&mut t, 1, params);
        assert!(!delta.is_empty(), "paint changed samples");
        // The tile materialized its weights.
        assert!(!t.get_tile((0, 0)).unwrap().weights_are_default());
        t.revert_splat_delta(&delta);
        // Byte-identical to before the stroke — the materialized buffer is gone.
        assert!(t.get_tile((0, 0)).unwrap().weights_are_default());
        assert_eq!(serde_json::to_string(&t).unwrap(), before);
    }

    #[test]
    fn paint_apply_revert_apply_round_trips() {
        let mut t = painted_terrain();
        let span = t.tile_span();
        let params = BrushParams::new(DVec2::new(span * 0.5, span * 0.5), span * 0.4, 0.7);
        let delta = apply_paint(&mut t, 2, params);
        let painted = serde_json::to_string(&t).unwrap();
        t.revert_splat_delta(&delta);
        t.apply_splat_delta(&delta);
        assert_eq!(
            serde_json::to_string(&t).unwrap(),
            painted,
            "redo != post-stroke"
        );
    }

    #[test]
    fn paint_stroke_merges_dabs() {
        let mut t = painted_terrain();
        let span = t.tile_span();
        let mut stroke = SplatStroke::begin(1);
        for k in 0..4 {
            let c = DVec2::new(span * 0.3 + k as f64, span * 0.5);
            stroke.add_dab(&mut t, BrushParams::new(c, 2.0, 0.5));
        }
        let delta = stroke.finish(&t);
        assert!(!delta.is_empty());
        // Revert takes us all the way back (merged before = first touch).
        t.revert_splat_delta(&delta);
        assert!(t.get_tile((0, 0)).unwrap().weights_are_default());
    }

    #[test]
    fn paint_across_tile_seam_is_seamless() {
        // Two tiles authored from one function; paint a dab centred on the shared
        // edge. The shared sample column must carry identical weights in both
        // tiles (the seam guarantee, mirroring the height brush).
        let mut t = TerrainData::new(8, 1.0);
        t.author_tile((0, 0), |_, _| 0.0);
        t.author_tile((1, 0), |_, _| 0.0);
        let span = t.tile_span();
        let params = BrushParams::new(DVec2::new(span, span * 0.5), 3.0, 1.0);
        apply_paint(&mut t, 1, params);
        let res = t.tile_resolution();
        // Far edge of tile (0,0) (i = res-1) == near edge of tile (1,0) (i = 0).
        for j in 0..res {
            let left = t.get_tile((0, 0)).unwrap().weight_sample(res, res - 1, j);
            let right = t.get_tile((1, 0)).unwrap().weight_sample(res, 0, j);
            assert_eq!(left, right, "seam mismatch at j={j}: {left:?} vs {right:?}");
        }
    }

    #[test]
    fn paint_does_not_author_new_tiles() {
        let mut t = painted_terrain();
        let n = t.tile_count();
        // Brush well outside the single authored tile.
        let params = BrushParams::new(DVec2::new(10_000.0, 10_000.0), 5.0, 1.0);
        let delta = apply_paint(&mut t, 1, params);
        assert!(delta.is_empty());
        assert_eq!(t.tile_count(), n, "paint must not author tiles");
    }
}
