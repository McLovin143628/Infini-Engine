//! The **undo contract** for sculpting: a sparse, tile-partitioned before/after
//! record of the height samples a brush stroke changed.
//!
//! ## Representation & memory rationale
//!
//! A stroke is stored as a set of **per-tile dense sub-rect patches**
//! ([`TilePatch`]): for every tile the stroke touched, the axis-aligned bounding
//! rectangle of the changed samples, with a flat `before`/`after` `f32` buffer for
//! that rectangle. This is deliberately *not* a `Vec<(tile, index, before, after)>`
//! of individual samples.
//!
//! Consider the worst case the design must survive — a 4096 m-radius brush at 1 m
//! resolution touches an ~8192 × 8192 sample disk (~5.3 × 10⁷ samples):
//!
//! * **Sparse per-sample**: each entry needs a tile key + sample index + two
//!   `f32` (~20 B) → ~1 GB, and every write chases a hash/BTree entry.
//! * **Dense sub-rect**: two `f32` per sample (8 B) in flat buffers → ~430 MB,
//!   and apply/revert is a straight slice copy into the tile — cache-friendly and
//!   allocation-light.
//!
//! A round brush fills a disk, so a tile's touched samples nearly fill their
//! bounding rectangle; the dense form's only waste is the ≤ 21 % of the bounding
//! box outside the disk (stored as `before == after` no-ops), which is far cheaper
//! than per-sample coordinate overhead. This is also exactly the shape the
//! editor's undo stack stores (mirroring [`EditCommand::SetTiles`]'s sparse
//! touched-cell record, one level down at the height-sample layer).
//!
//! `created_tiles` records tiles the stroke **authored from nothing** so that
//! [`TerrainData::revert_delta`] can remove them again — a raise that grew the
//! terrain undoes back to byte-identical empty space.
//!
//! [`EditCommand::SetTiles`]: (editor `inf-editor-core::scene::undo`)

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::data::TerrainData;

/// A dense before/after patch over one tile's `[i0, i0+w) × [j0, j0+h)` sample
/// rectangle. `before`/`after` are row-major `f32` **offsets** (tile-local, the
/// same space as [`crate::TerrainTile::heights`]), length `w · h`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TilePatch {
    /// Tile grid coordinate this patch belongs to.
    pub coord: (i32, i32),
    /// Left sample column of the rectangle.
    pub i0: u32,
    /// Top sample row of the rectangle.
    pub j0: u32,
    /// Rectangle width in samples.
    pub w: u32,
    /// Rectangle height in samples.
    pub h: u32,
    /// Pre-edit `f32` offsets (row-major, `w · h`).
    pub before: Vec<f32>,
    /// Post-edit `f32` offsets (row-major, `w · h`).
    pub after: Vec<f32>,
}

impl TilePatch {
    /// Number of samples in this patch.
    #[inline]
    pub fn sample_count(&self) -> usize {
        (self.w * self.h) as usize
    }
}

/// The reversible record a brush stroke returns — apply it for redo, revert it
/// for undo. Empty (no patches, no created tiles) when the stroke changed nothing.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HeightDelta {
    /// One dense sub-rect patch per touched tile (deterministic tile order).
    pub patches: Vec<TilePatch>,
    /// Tiles this stroke created from nothing (removed on revert). Sorted.
    pub created_tiles: Vec<(i32, i32)>,
}

impl HeightDelta {
    /// `true` when the stroke changed no samples and authored no tiles.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty() && self.created_tiles.is_empty()
    }

    /// Total samples stored across all patches (bounding-rect inclusive).
    pub fn sample_count(&self) -> usize {
        self.patches.iter().map(TilePatch::sample_count).sum()
    }

    /// Approximate heap footprint of the stored before/after buffers, in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.sample_count() * 2 * std::mem::size_of::<f32>()
    }
}

/// Accumulates the *first-touch* `before` value per changed sample across one or
/// more brush dabs, then compresses the result into per-tile dense patches.
///
/// A [`Stroke`](crate::Stroke) keeps one builder alive across every dab so that,
/// where dabs overlap, `before` is the value at the **first** touch and `after`
/// (read from the terrain at [`finalize`](DeltaBuilder::finalize) time) is the
/// **final** value — the merge contract the editor relies on.
#[derive(Default)]
pub(crate) struct DeltaBuilder {
    /// `coord → (i, j) → first-touch before offset`.
    touched: BTreeMap<(i32, i32), BTreeMap<(u32, u32), f32>>,
    /// Tiles created from nothing during the stroke.
    created: BTreeSet<(i32, i32)>,
}

impl DeltaBuilder {
    /// Record that sample `(i, j)` of `coord` is about to change; keeps only the
    /// first `before` seen (so repeated dabs over the same sample merge correctly).
    #[inline]
    pub(crate) fn touch(&mut self, coord: (i32, i32), i: u32, j: u32, before: f32) {
        self.touched
            .entry(coord)
            .or_default()
            .entry((i, j))
            .or_insert(before);
    }

    /// Note that `coord` was authored from nothing during this stroke.
    #[inline]
    pub(crate) fn mark_created(&mut self, coord: (i32, i32)) {
        self.created.insert(coord);
    }

    /// `true` if nothing has been touched yet (used to skip empty finalize work).
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.touched.is_empty()
    }

    /// Compress the accumulated touches into per-tile dense sub-rect patches,
    /// reading final (`after`) and filler values from `data`'s current tiles.
    pub(crate) fn finalize(&self, data: &TerrainData) -> HeightDelta {
        let res = data.tile_resolution();
        let mut patches = Vec::with_capacity(self.touched.len());
        for (&coord, samples) in &self.touched {
            let Some(tile) = data.get_tile(coord) else {
                continue;
            };
            // Bounding rectangle of the touched samples.
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
            let mut before = vec![0.0f32; (w * h) as usize];
            let mut after = vec![0.0f32; (w * h) as usize];
            for row in 0..h {
                for col in 0..w {
                    let i = i_min + col;
                    let j = j_min + row;
                    let idx = (row * w + col) as usize;
                    // `after` is the tile's current value; `before` is the recorded
                    // first-touch value, or the current value for filler cells
                    // inside the bounding box that were never actually changed.
                    let cur = tile.sample(res, i, j);
                    after[idx] = cur;
                    before[idx] = samples.get(&(i, j)).copied().unwrap_or(cur);
                }
            }
            patches.push(TilePatch {
                coord,
                i0: i_min,
                j0: j_min,
                w,
                h,
                before,
                after,
            });
        }
        HeightDelta {
            patches,
            created_tiles: self.created.iter().copied().collect(),
        }
    }
}

impl TerrainData {
    /// Apply (redo) a [`HeightDelta`]: write each patch's `after` buffer,
    /// (re)creating any tile it targets. Byte-identical to the state right after
    /// the stroke that produced `delta`.
    pub fn apply_delta(&mut self, delta: &HeightDelta) {
        let res = self.tile_resolution();
        for patch in &delta.patches {
            let tile = self.get_or_create_tile(patch.coord);
            write_patch(tile, res, patch, &patch.after);
        }
    }

    /// Revert (undo) a [`HeightDelta`]: write each patch's `before` buffer, then
    /// remove the tiles the stroke created (now flat again) so the terrain returns
    /// byte-identical to before the stroke — including its tile set.
    pub fn revert_delta(&mut self, delta: &HeightDelta) {
        let res = self.tile_resolution();
        for patch in &delta.patches {
            // Skip a patch that targets a created tile we are about to remove — but
            // it is cheaper and correct to just write then remove.
            let tile = self.get_or_create_tile(patch.coord);
            write_patch(tile, res, patch, &patch.before);
        }
        for &coord in &delta.created_tiles {
            self.remove_tile(coord);
        }
    }
}

/// Blit a `w · h` row-major buffer into `tile`'s sample rectangle.
fn write_patch(tile: &mut crate::TerrainTile, res: u32, patch: &TilePatch, buf: &[f32]) {
    for row in 0..patch.h {
        for col in 0..patch.w {
            let idx = (row * patch.w + col) as usize;
            tile.set_sample(res, patch.i0 + col, patch.j0 + row, buf[idx]);
        }
    }
}

// ── the hole-mask undo record (P21.2) ───────────────────────────────────────

/// A dense before/after patch over one tile's `[i0, i0+w) × [j0, j0+h)` sample
/// rectangle of the **hole mask**. `before`/`after` are row-major `0`/`1` bytes,
/// length `w · h`.
///
/// ## Why bytes here and bits on the tile
///
/// [`TerrainTile`](crate::TerrainTile) packs its mask to a bit per sample because
/// it is a full `resolution²` layer that also goes to the GPU. A patch is bounded
/// by a **brush footprint**, so the same packing would buy kilobytes at most while
/// making every `apply`/`revert` a shift-and-mask loop and every `PartialEq` in a
/// test opaque. One plain byte per sample is the right trade at this size — the
/// same reasoning [`TilePatch`] applies to `f32` heights, one layer down.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HolePatch {
    /// Tile grid coordinate this patch belongs to.
    pub coord: (i32, i32),
    /// Left sample column of the rectangle.
    pub i0: u32,
    /// Top sample row of the rectangle.
    pub j0: u32,
    /// Rectangle width in samples.
    pub w: u32,
    /// Rectangle height in samples.
    pub h: u32,
    /// Pre-edit hole flags (row-major, `w · h`, `0` = surface, `1` = holed).
    pub before: Vec<u8>,
    /// Post-edit hole flags (row-major, `w · h`).
    pub after: Vec<u8>,
}

impl HolePatch {
    /// Number of samples in this patch.
    #[inline]
    pub fn sample_count(&self) -> usize {
        (self.w * self.h) as usize
    }
}

/// The reversible record a hole edit returns — apply it for redo, revert it for
/// undo. Empty when the edit opened and closed nothing.
///
/// This is the heightfield half of a P21.2 carve. The voxel half is an
/// `inf_voxel::VoxelDelta`, and the editor pairs them into **one** transaction so
/// a single undo puts back both the rock and the ground above it.
///
/// ## The healing invariant
///
/// After [`apply_hole_delta`](TerrainData::apply_hole_delta) or
/// [`revert_hole_delta`](TerrainData::revert_hole_delta), every touched tile is
/// [healed](crate::TerrainTile::heal_holes): its mask buffer exists **iff** some
/// sample is holed. That is what makes carve → undo byte-identical rather than
/// merely value-identical — an un-healed tile would carry a `resolution²/8` block
/// of zeros into every container forever.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HoleDelta {
    /// One dense sub-rect patch per touched tile (deterministic tile order).
    pub patches: Vec<HolePatch>,
}

impl HoleDelta {
    /// `true` when the edit changed no sample.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty()
    }

    /// Total samples stored across all patches (bounding-rect inclusive).
    pub fn sample_count(&self) -> usize {
        self.patches.iter().map(HolePatch::sample_count).sum()
    }

    /// How many samples this edit **opens** (surface → hole) minus how many it
    /// closes. Positive for a carve that broke through, negative for a fill that
    /// healed one, zero for a wash. The number the editor's readout quotes.
    pub fn net_opened(&self) -> i64 {
        let mut net = 0i64;
        for p in &self.patches {
            for (b, a) in p.before.iter().zip(&p.after) {
                net += i64::from(*a != 0) - i64::from(*b != 0);
            }
        }
        net
    }
}

/// Accumulates the *first-touch* `before` flag per changed sample across one or
/// more carve dabs, then compresses the result into per-tile dense patches.
///
/// The exact shape — and the exact merge contract — of [`DeltaBuilder`], one
/// layer over: `before` is the flag at the **first** touch and `after` is read
/// from the terrain at [`finalize`](HoleDeltaBuilder::finalize) time, so a stroke
/// whose dabs overlap still undoes in one step to where it started.
#[derive(Default)]
pub struct HoleDeltaBuilder {
    /// `coord → (i, j) → first-touch hole flag`.
    touched: BTreeMap<(i32, i32), BTreeMap<(u32, u32), bool>>,
}

impl HoleDeltaBuilder {
    /// A builder with nothing touched.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that sample `(i, j)` of `coord` is about to change; keeps only the
    /// first `before` seen.
    #[inline]
    pub fn touch(&mut self, coord: (i32, i32), i: u32, j: u32, before: bool) {
        self.touched
            .entry(coord)
            .or_default()
            .entry((i, j))
            .or_insert(before);
    }

    /// `true` if nothing has been touched yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.touched.is_empty()
    }

    /// Compress the accumulated touches into per-tile dense patches, reading the
    /// final (`after`) flags from `data`'s current tiles. Patches whose `before`
    /// and `after` agree everywhere are dropped, so a wash produces an empty
    /// delta rather than an undo step that does nothing.
    pub fn finalize(&self, data: &TerrainData) -> HoleDelta {
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
            let mut before = vec![0u8; (w * h) as usize];
            let mut after = vec![0u8; (w * h) as usize];
            let mut changed = false;
            for row in 0..h {
                for col in 0..w {
                    let i = i_min + col;
                    let j = j_min + row;
                    let idx = (row * w + col) as usize;
                    let cur = tile.is_hole(res, i, j);
                    after[idx] = u8::from(cur);
                    before[idx] = u8::from(samples.get(&(i, j)).copied().unwrap_or(cur));
                    changed |= before[idx] != after[idx];
                }
            }
            if changed {
                patches.push(HolePatch {
                    coord,
                    i0: i_min,
                    j0: j_min,
                    w,
                    h,
                    before,
                    after,
                });
            }
        }
        HoleDelta { patches }
    }
}

impl TerrainData {
    /// Apply (redo) a [`HoleDelta`]: write each patch's `after` flags, then heal.
    ///
    /// Returns how many patches were **skipped** because the terrain no longer
    /// holds their tile — a hole cannot exist without a heightfield to cut it
    /// from, so skipping is right, but a caller that ignores the number is
    /// silently replaying less than the record describes.
    ///
    /// `#[must_use]` for parity with the voxel half
    /// ([`VoxelData::apply_delta`](inf_voxel::VoxelData::apply_delta), whose
    /// dropped answer is exactly how a hole mask came to be replayed over rock
    /// nobody put back). The two halves of one carve must not have two different
    /// standards about a partial replay (P21.3 audit).
    #[must_use = "a non-zero skip count means the mask was only partly replayed"]
    pub fn apply_hole_delta(&mut self, delta: &HoleDelta) -> usize {
        self.write_hole_patches(delta, false)
    }

    /// Revert (undo) a [`HoleDelta`]: write each patch's `before` flags, then
    /// heal — so a carve that materialized a mask undoes back to a tile that
    /// serializes byte-identically to one nothing ever carved.
    ///
    /// Returns the skip count, like [`apply_hole_delta`](Self::apply_hole_delta).
    #[must_use = "a non-zero skip count means the mask was only partly reverted"]
    pub fn revert_hole_delta(&mut self, delta: &HoleDelta) -> usize {
        self.write_hole_patches(delta, true)
    }

    fn write_hole_patches(&mut self, delta: &HoleDelta, revert: bool) -> usize {
        let res = self.tile_resolution();
        let mut skipped = 0;
        for patch in &delta.patches {
            if self.get_tile(patch.coord).is_none() {
                skipped += 1;
                continue;
            }
            let buf = if revert { &patch.before } else { &patch.after };
            let tile = self.get_or_create_tile(patch.coord);
            for row in 0..patch.h {
                for col in 0..patch.w {
                    let idx = (row * patch.w + col) as usize;
                    tile.set_hole(res, patch.i0 + col, patch.j0 + row, buf[idx] != 0);
                }
            }
            tile.heal_holes();
        }
        skipped
    }
}
