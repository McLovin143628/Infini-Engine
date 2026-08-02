//! **Erosion data maps** (P19.1): the undo record and the export path for the
//! per-tile flow / deposition / wear layers.
//!
//! Erosion used to throw its own story away — the water flux, the sediment it
//! dropped and the material it dissolved existed only inside one bake's working
//! state and were gone the moment the heights were written back. P19.1 keeps
//! them: [`crate::erosion`] accumulates three per-cell scalars alongside the
//! heights, they persist on the tile ([`crate::TerrainTile::maps`]) exactly as
//! sparsely as the splat weights do, and they come back out as normalized
//! grayscale images — the Gaea-class *data maps* a texturing or scatter pass
//! wants ([`DataMapKind`] states each one's definition and SI unit).
//!
//! This module is the layer's twin of [`crate::delta`] / [`crate::splat`]:
//!
//! * [`DataMapPatch`] / [`DataMapDelta`] mirror [`SplatPatch`] / [`SplatDelta`] —
//!   dense per-tile sub-rect before/after patches the editor's undo stack stores
//!   and [`TerrainData::revert_data_map_delta`] replays;
//! * `materialized_tiles` mirrors the splat rule: a tile that was on the sparse
//!   **never-eroded** default when the bake touched it is dropped back to that
//!   default on undo, so an erosion bake on fresh terrain undoes to
//!   byte-identical, map-free space.
//!
//! # Normalization is a *view*, never the storage
//!
//! The stored accumulators are raw and monotone: a second bake **adds to** the
//! first's totals, and no bake can retroactively rescale an earlier one. (The
//! accumulators compose that way; the *simulation* does not — see the
//! [`erosion`](crate::erosion) module docs.) Every consumer that
//! needs `[0, 1]` (the PNG export here; the P19.3 PCG mask samplers later)
//! divides by a range it computes at read time — [`TerrainData::data_map_range`]
//! is that computation, published so a caller can report the divisor it used.
//!
//! [`SplatPatch`]: crate::SplatPatch
//! [`SplatDelta`]: crate::SplatDelta

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::data::TerrainData;
use crate::import::{encode_png16, HeightImage, TerrainError};
use crate::tile::{DataMapKind, TerrainTile, DATA_MAP_CHANNELS, DEFAULT_DATA_MAP};

// ── delta (undo record) ──────────────────────────────────────────────────────

/// A dense before/after patch over one tile's `[i0, i0+w) × [j0, j0+h)` sample
/// rectangle of erosion data maps. Mirrors [`TilePatch`](crate::TilePatch) at the
/// data-map layer: `before`/`after` are row-major `[f32; 3]`
/// (`[flow, deposition, wear]`), length `w · h`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataMapPatch {
    pub coord: (i32, i32),
    pub i0: u32,
    pub j0: u32,
    pub w: u32,
    pub h: u32,
    pub before: Vec<[f32; DATA_MAP_CHANNELS]>,
    pub after: Vec<[f32; DATA_MAP_CHANNELS]>,
}

impl DataMapPatch {
    #[inline]
    pub fn sample_count(&self) -> usize {
        (self.w * self.h) as usize
    }
}

/// The reversible record an erosion bake returns for its **data maps** — apply it
/// for redo, revert it for undo. The height half of the same bake is a
/// [`HeightDelta`](crate::HeightDelta); the editor records both in one undo
/// transaction.
///
/// `materialized_tiles` records tiles whose data-map buffer was on the sparse
/// **never-eroded** default before the bake: on revert they are dropped back to
/// it, so the terrain re-serializes byte-identically to its un-eroded form.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DataMapDelta {
    /// One dense sub-rect patch per touched tile (deterministic tile order).
    pub patches: Vec<DataMapPatch>,
    /// Tiles the bake materialized from the sparse default (reset on revert).
    /// Sorted.
    pub materialized_tiles: Vec<(i32, i32)>,
}

impl DataMapDelta {
    /// `true` when the bake changed no data-map sample **and** materialized no
    /// tile — i.e. reverting it would do nothing at all.
    ///
    /// Both halves, mirroring [`HeightDelta::is_empty`](crate::HeightDelta),
    /// because both halves have work to undo: `materialized_tiles` is what drops
    /// a buffer back to the sparse never-eroded default. The two lists move
    /// together today (`write_back_maps` only ever marks a tile it is about to
    /// touch), so this cannot currently disagree with the patch-only test — which
    /// is exactly why it is written as the invariant rather than as the shortcut
    /// that happens to hold.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty() && self.materialized_tiles.is_empty()
    }

    /// Total samples stored across all patches (bounding-rect inclusive).
    pub fn sample_count(&self) -> usize {
        self.patches.iter().map(DataMapPatch::sample_count).sum()
    }

    /// Approximate heap footprint of the stored before/after buffers, in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.sample_count() * 2 * DATA_MAP_CHANNELS * std::mem::size_of::<f32>()
    }
}

/// `coord → (i, j) → first-touch before texel`.
type TouchedMaps = BTreeMap<(i32, i32), BTreeMap<(u32, u32), [f32; DATA_MAP_CHANNELS]>>;

/// Accumulates the *first-touch* `before` texel per changed sample across a bake,
/// then compresses the result into per-tile dense patches — the data-map twin of
/// [`SplatDeltaBuilder`](crate::splat).
#[derive(Default)]
pub(crate) struct DataMapDeltaBuilder {
    touched: TouchedMaps,
    materialized: BTreeSet<(i32, i32)>,
}

impl DataMapDeltaBuilder {
    #[inline]
    pub(crate) fn touch(
        &mut self,
        coord: (i32, i32),
        i: u32,
        j: u32,
        before: [f32; DATA_MAP_CHANNELS],
    ) {
        self.touched
            .entry(coord)
            .or_default()
            .entry((i, j))
            .or_insert(before);
    }

    #[inline]
    pub(crate) fn mark_materialized(&mut self, coord: (i32, i32)) {
        self.materialized.insert(coord);
    }

    /// Compress the accumulated touches into per-tile dense sub-rect patches,
    /// reading final (`after`) and filler values from `data`'s current tiles.
    pub(crate) fn finalize(&self, data: &TerrainData) -> DataMapDelta {
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
            let mut before = vec![DEFAULT_DATA_MAP; (w * h) as usize];
            let mut after = vec![DEFAULT_DATA_MAP; (w * h) as usize];
            for row in 0..h {
                for col in 0..w {
                    let i = i_min + col;
                    let j = j_min + row;
                    let idx = (row * w + col) as usize;
                    let cur = tile.map_texel(res, i, j);
                    after[idx] = cur;
                    before[idx] = samples.get(&(i, j)).copied().unwrap_or(cur);
                }
            }
            patches.push(DataMapPatch {
                coord,
                i0: i_min,
                j0: j_min,
                w,
                h,
                before,
                after,
            });
        }
        DataMapDelta {
            patches,
            materialized_tiles: self.materialized.iter().copied().collect(),
        }
    }
}

impl TerrainData {
    /// Apply (redo) a [`DataMapDelta`]: materialize each targeted tile's data-map
    /// buffer if needed, then write each patch's `after` texels. Byte-identical to
    /// the state right after the bake that produced `delta`.
    pub fn apply_data_map_delta(&mut self, delta: &DataMapDelta) {
        let res = self.tile_resolution();
        for patch in &delta.patches {
            let Some(tile) = self.get_tile_mut(patch.coord) else {
                continue;
            };
            tile.ensure_maps(res);
            write_map_patch(tile, res, patch, &patch.after);
        }
    }

    /// Revert (undo) a [`DataMapDelta`]: write each patch's `before` texels, then
    /// reset any tile the bake materialized back to the sparse never-eroded
    /// default — so the terrain returns byte-identical to before the bake.
    pub fn revert_data_map_delta(&mut self, delta: &DataMapDelta) {
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
            write_map_patch(tile, res, patch, &patch.before);
        }
        for &coord in &delta.materialized_tiles {
            if let Some(tile) = self.get_tile_mut(coord) {
                tile.clear_maps();
            }
        }
    }
}

/// Blit a `w · h` row-major texel buffer into `tile`'s sample rectangle.
fn write_map_patch(
    tile: &mut TerrainTile,
    res: u32,
    patch: &DataMapPatch,
    buf: &[[f32; DATA_MAP_CHANNELS]],
) {
    for row in 0..patch.h {
        for col in 0..patch.w {
            let idx = (row * patch.w + col) as usize;
            tile.set_map_texel(res, patch.i0 + col, patch.j0 + row, buf[idx]);
        }
    }
}

// ── export ───────────────────────────────────────────────────────────────────

/// The rectangle of the terrain's global sample lattice an export covers.
///
/// Derived from the authored tile extent exactly like
/// [`TerrainData::to_height_image`]'s grid (shared tile edges, so
/// `width = ntx·(res−1)+1`), then optionally clipped to a world AABB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SampleGrid {
    min_tx: i32,
    min_tz: i32,
    max_tx: i32,
    max_tz: i32,
    /// Global sample index of the first exported column/row (0-based within the
    /// authored extent).
    gx0: i64,
    gz0: i64,
    width: u32,
    height: u32,
}

impl TerrainData {
    /// The exported sample rectangle for `region` (a world-XZ AABB, or the whole
    /// authored extent when `None`). `None` for an empty terrain or an empty clip.
    fn data_map_grid(&self, region: Option<(DVec2, DVec2)>) -> Option<SampleGrid> {
        let mut it = self.tiles();
        let (&(mut min_tx, mut min_tz), _) = it.next()?;
        let (mut max_tx, mut max_tz) = (min_tx, min_tz);
        for (&(tx, tz), _) in self.tiles() {
            min_tx = min_tx.min(tx);
            min_tz = min_tz.min(tz);
            max_tx = max_tx.max(tx);
            max_tz = max_tz.max(tz);
        }
        let res = self.tile_resolution();
        let cells = (res - 1) as i64;
        // Shared tile edges: `ntx` tiles span `ntx·(res−1)+1` samples.
        let full_w = (max_tx - min_tx + 1) as i64 * cells + 1;
        let full_h = (max_tz - min_tz + 1) as i64 * cells + 1;

        // World position of the extent's sample (0, 0), and the clip in samples.
        let mps = self.meters_per_sample();
        let origin = self.tile_origin_xz((min_tx, min_tz));
        let (mut gx0, mut gz0) = (0i64, 0i64);
        let (mut gx1, mut gz1) = (full_w - 1, full_h - 1);
        if let Some((min, max)) = region {
            let (min, max) = (min.min(max), min.max(max));
            gx0 = gx0.max(((min.x - origin.x) / mps).floor() as i64);
            gz0 = gz0.max(((min.y - origin.y) / mps).floor() as i64);
            gx1 = gx1.min(((max.x - origin.x) / mps).ceil() as i64);
            gz1 = gz1.min(((max.y - origin.y) / mps).ceil() as i64);
            if gx1 < gx0 || gz1 < gz0 {
                return None;
            }
        }
        Some(SampleGrid {
            min_tx,
            min_tz,
            max_tx,
            max_tz,
            gx0,
            gz0,
            width: (gx1 - gx0 + 1) as u32,
            height: (gz1 - gz0 + 1) as u32,
        })
    }

    /// The data-map value at global sample `(gx, gz)` of the authored extent,
    /// resolving the owning tile (holes and never-eroded tiles read `0`).
    fn data_map_at(&self, grid: &SampleGrid, kind: DataMapKind, gx: i64, gz: i64) -> f32 {
        let res = self.tile_resolution();
        let cells = (res - 1) as i64;
        // Clamp the far edge back onto the last authored column/row — shared-edge
        // equivalence, exactly as `to_height_image` does.
        let mut tx = grid.min_tx as i64 + gx.div_euclid(cells);
        let mut i = gx - (tx - grid.min_tx as i64) * cells;
        if tx > grid.max_tx as i64 {
            tx = grid.max_tx as i64;
            i = cells;
        }
        let mut tz = grid.min_tz as i64 + gz.div_euclid(cells);
        let mut j = gz - (tz - grid.min_tz as i64) * cells;
        if tz > grid.max_tz as i64 {
            tz = grid.max_tz as i64;
            j = cells;
        }
        self.get_tile((tx as i32, tz as i32))
            .map(|t| t.map_sample(res, kind, i as u32, j as u32))
            .unwrap_or(0.0)
    }

    /// The `(min, max)` of one data map over `region` (the whole authored extent
    /// when `None`) — the divisor a normalized view is derived with, published so
    /// a caller can report it. `None` for an empty terrain / empty clip.
    ///
    /// `min` is `0.0` in practice (the accumulators are non-negative), but it is
    /// measured rather than assumed so a future signed channel needs no new API.
    pub fn data_map_range(
        &self,
        kind: DataMapKind,
        region: Option<(DVec2, DVec2)>,
    ) -> Option<(f32, f32)> {
        let grid = self.data_map_grid(region)?;
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for gz in 0..grid.height as i64 {
            for gx in 0..grid.width as i64 {
                let v = self.data_map_at(&grid, kind, grid.gx0 + gx, grid.gz0 + gz);
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        lo.is_finite().then_some((lo, hi))
    }

    /// Render one data map as a **16-bit grayscale image**, normalized over
    /// `region`'s own `[min, max]` (the whole authored extent when `None`).
    ///
    /// The normalization is derived here and *not* stored — see the module docs.
    /// A perfectly uniform map (including an entirely un-eroded terrain, where
    /// `max == min == 0`) renders as solid black rather than dividing by zero.
    /// Returns the image plus the `(min, max)` it divided by, so the caller can
    /// state the mapping.
    pub fn to_data_map_image(
        &self,
        kind: DataMapKind,
        region: Option<(DVec2, DVec2)>,
    ) -> Option<(HeightImage, (f32, f32))> {
        let grid = self.data_map_grid(region)?;
        let (lo, hi) = self.data_map_range(kind, region)?;
        let span = (hi - lo) as f64;
        let mut samples = vec![0u16; (grid.width as usize) * (grid.height as usize)];
        for gz in 0..grid.height as i64 {
            for gx in 0..grid.width as i64 {
                let v = self.data_map_at(&grid, kind, grid.gx0 + gx, grid.gz0 + gz);
                let norm = if span > 0.0 {
                    ((v - lo) as f64 / span).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                samples[gz as usize * grid.width as usize + gx as usize] =
                    (norm * 65535.0).round() as u16;
            }
        }
        Some((
            HeightImage {
                width: grid.width,
                height: grid.height,
                samples,
            },
            (lo, hi),
        ))
    }

    /// Export one data map as 16-bit grayscale PNG bytes, returning the bytes and
    /// the `(min, max)` the normalization divided by. Deterministic: identical
    /// terrain ⇒ identical bytes.
    pub fn export_data_map(
        &self,
        kind: DataMapKind,
        region: Option<(DVec2, DVec2)>,
    ) -> Result<(Vec<u8>, (f32, f32)), TerrainError> {
        let (img, range) = self
            .to_data_map_image(kind, region)
            .ok_or(TerrainError::Empty)?;
        Ok((encode_png16(&img)?, range))
    }

    /// `true` when **no** authored tile carries data maps — i.e. erosion has never
    /// run on this terrain, and the maps cost zero per-sample bytes.
    pub fn data_maps_are_default(&self) -> bool {
        self.tiles().all(|(_, t)| t.maps_are_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::DataMapKind;

    fn terrain() -> TerrainData {
        let mut t = TerrainData::new(5, 1.0);
        t.author_tile((0, 0), |_, _| 0.0);
        t.author_tile((1, 0), |_, _| 0.0);
        t
    }

    /// The never-eroded default costs **zero per-sample bytes** — no `maps` key at
    /// all in the human-readable form, and exactly one length byte in bincode.
    #[test]
    fn the_never_eroded_default_is_free() {
        let t = terrain();
        assert!(t.data_maps_are_default());
        let json = serde_json::to_string(&t).unwrap();
        assert!(
            !json.contains("maps"),
            "un-eroded terrain leaked maps: {json}"
        );

        let cfg = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&t, cfg).unwrap();
        // One tile with maps materialized costs res² · 3 · 4 bytes more; an
        // un-eroded one costs the single zero-length count already inside `bytes`.
        let mut eroded = t.clone();
        eroded.get_tile_mut((0, 0)).unwrap().ensure_maps(5);
        let big = bincode::serde::encode_to_vec(&eroded, cfg).unwrap();
        assert_eq!(
            big.len() - bytes.len(),
            5 * 5 * DATA_MAP_CHANNELS * 4,
            "materializing one tile must cost exactly its dense buffer"
        );
    }

    /// Apply → revert → apply round-trips, and a revert of a bake that
    /// materialized the buffer returns byte-identical un-eroded terrain.
    #[test]
    fn delta_round_trips_and_undo_is_byte_identical() {
        let mut t = terrain();
        let before = serde_json::to_string(&t).unwrap();

        let mut builder = DataMapDeltaBuilder::default();
        builder.mark_materialized((0, 0));
        for j in 0..3 {
            for i in 0..3 {
                builder.touch((0, 0), i, j, DEFAULT_DATA_MAP);
                t.get_tile_mut((0, 0)).unwrap().set_map_texel(
                    5,
                    i,
                    j,
                    [i as f32, j as f32, (i + j) as f32],
                );
            }
        }
        let delta = builder.finalize(&t);
        assert!(!delta.is_empty());
        assert_eq!(delta.sample_count(), 9);
        assert!(delta.memory_bytes() > 0);
        let after = serde_json::to_string(&t).unwrap();

        t.revert_data_map_delta(&delta);
        assert!(t.get_tile((0, 0)).unwrap().maps_are_default());
        assert_eq!(
            serde_json::to_string(&t).unwrap(),
            before,
            "undo must be exact"
        );

        t.apply_data_map_delta(&delta);
        assert_eq!(
            serde_json::to_string(&t).unwrap(),
            after,
            "redo must be exact"
        );
    }

    /// Export normalizes over the measured range and is deterministic; a uniform
    /// (un-eroded) map is solid black rather than a divide-by-zero.
    #[test]
    fn export_normalizes_and_is_deterministic() {
        let mut t = terrain();
        assert_eq!(t.data_map_range(DataMapKind::Flow, None), Some((0.0, 0.0)));
        let (flat, _) = t.export_data_map(DataMapKind::Flow, None).unwrap();
        let (flat2, _) = t.export_data_map(DataMapKind::Flow, None).unwrap();
        assert_eq!(flat, flat2, "export is a pure function");

        t.get_tile_mut((0, 0))
            .unwrap()
            .set_map_texel(5, 2, 2, [4.0, 0.0, 0.0]);
        assert_eq!(t.data_map_range(DataMapKind::Flow, None), Some((0.0, 4.0)));
        let (img, (lo, hi)) = t.to_data_map_image(DataMapKind::Flow, None).unwrap();
        assert_eq!((lo, hi), (0.0, 4.0));
        // Two tiles of res 5 share an edge ⇒ 9 × 5 samples.
        assert_eq!((img.width, img.height), (9, 5));
        assert_eq!(
            img.samples[2 * img.width as usize + 2],
            65535,
            "the peak maps to full white"
        );
        assert_eq!(img.samples[0], 0);
    }

    /// **A map-only edit still schedules a write-back.** Data maps ride the tile
    /// blob, so a bake that moved a map but no height must still mark the tile
    /// dirty — otherwise the P16.4b save would decide the terrain was clean and
    /// the accumulated maps would be lost on reload.
    ///
    /// The seam that makes this work is `write_back_maps` reaching for
    /// `get_tile_mut`, which is what stamps and dirties; asserting it here means
    /// a future "optimization" that writes maps through an immutable back door
    /// fails loudly instead of silently dropping saves.
    #[test]
    fn a_map_only_edit_dirties_the_tile_for_write_back() {
        let mut t = terrain();
        t.clear_dirty();
        assert!(!t.has_dirty_tiles(), "the fixture starts clean");

        // Edit the region's MAPS only — the heights come back untouched.
        let span = t.tile_span();
        let (heights, maps) =
            t.edit_region_with_maps(DVec2::ZERO, DVec2::new(span, span), 0, |region| {
                let (nx, nz) = region.dims();
                for r in 0..nz {
                    for c in 0..nx {
                        region.set_map(c, r, DataMapKind::Flow, 1.5);
                    }
                }
            });
        assert!(heights.is_empty(), "no height moved");
        assert!(!maps.is_empty(), "the maps did move");

        let dirty = t.dirty_tiles();
        assert!(
            !dirty.is_empty(),
            "a map-only edit must still schedule a write-back"
        );
        // …and the write-back stager really picks it up, carrying the maps.
        let edits = crate::writeback::TerrainEdits::from_dirty(&t, &dirty);
        assert!(!edits.is_empty());
        assert!(
            edits.changed.values().all(|tile| !tile.maps_are_default()),
            "the staged tiles must carry the maps they were dirtied for"
        );
    }

    /// A region clip really clips, and stays inside the authored extent.
    #[test]
    fn export_region_clips_the_grid() {
        let t = terrain();
        let (img, _) = t
            .to_data_map_image(
                DataMapKind::Wear,
                Some((DVec2::new(0.0, 0.0), DVec2::new(3.0, 2.0))),
            )
            .unwrap();
        assert_eq!((img.width, img.height), (4, 3));
        // A clip entirely off the terrain yields nothing rather than a stripe.
        assert!(t
            .to_data_map_image(
                DataMapKind::Wear,
                Some((DVec2::new(-500.0, -500.0), DVec2::new(-400.0, -400.0)))
            )
            .is_none());
    }
}
