//! **Write-back** (P16.4b): folding a session's terrain edits back into the
//! `.inf_terrain` they came from, without rebuilding the terrain.
//!
//! The editor edits an asset-backed terrain by paging level-0 tiles into the ECS
//! component's `TerrainData` and sculpting them there (see
//! `inf_editor_core::terrain_edit` for the authority decision). On save, a
//! handful of edited tiles have to become a **whole new asset image** — the
//! `.inf_terrain` layout is a header + a sorted directory + packed blobs, so
//! there is no such thing as patching one tile in place.
//!
//! [`rewrite_terrain_asset`] does that merge:
//!
//! 1. **Level 0** — every tile the store already holds is copied through as
//!    *bytes* (no decode, no re-encode), except the edited ones, which are
//!    encoded from the working set, and the deleted ones, which are dropped.
//! 2. **The pyramid** — only the **ancestors of edited tiles** are decimated
//!    again; every other coarse tile is copied through as bytes.
//! 3. The result is assembled by the same [`TerrainAssetBuilder`] a fresh build
//!    uses, so it is byte-deterministic in the same way.
//!
//! # Why the partial rebuild is byte-equal to a full one
//!
//! Not by testing it (though `partial_recompute_equals_a_full_rebuild` does), but
//! by construction, in three pieces:
//!
//! * **The level *shape*** — which levels exist, at what spacing, holding which
//!   coordinates — is a function of the level-0 **coordinate set** alone, because
//!   `build_pyramid`'s stop conditions are counts and its fine→coarse grouping is
//!   floor-halving. [`plan_pyramid`] computes exactly that, and `build_pyramid`
//!   is implemented on top of it rather than beside it.
//! * **The reduction kernel** is shared: both paths call
//!   [`downsample_block`], which needs only a block's four members. A full
//!   rebuild hands it the whole level; this hands it four tiles.
//! * **A coarse tile is a pure function of its 2 × 2 block.** So a coarse tile
//!   whose block is untouched is bit-identical to the one already in the store,
//!   and copying its bytes is not an optimization that *approximates* the rebuild
//!   — it is the rebuild's own answer, already computed.
//!
//! Dirtiness therefore propagates strictly upward: an edited level-0 tile marks
//! its parent, which marks *its* parent, and so on. A coarse tile is recomputed
//! when its block changed, when the store simply does not have it (a level the
//! terrain grew into, or ground the edit authored from nothing), or when the
//! plan places it at a level the store never had.
//!
//! # The RAM cost, stated plainly
//!
//! This is **whole-payload staging**: the source image is fully in memory (a
//! `FileTileStore` reads the loose file whole) and the rewritten image is
//! assembled into a second `Vec<u8>` before it is written. Peak cost is therefore
//! ≈ 2 × the asset, plus the edited tiles. That is the *same* limitation P16.4a's
//! chunked import documents from the other side, and it has the same fix — a
//! streaming rewriter that copies untouched blobs from the old file to the new
//! one on disk, never through memory. Tracked as the shared P16.4 follow-up; a
//! 16 k × 16 k source at 256² tiles is ~1 GB, which is survivable on an authoring
//! machine and not on a small one.

use std::collections::{BTreeMap, BTreeSet};

use crate::asset::{
    encode_tile, TerrainAsset, TerrainAssetBuilder, TerrainAssetError, TERRAIN_ASSET_SCHEMA_VERSION,
};
use crate::data::TerrainData;
use crate::pyramid::{downsample_block, plan_pyramid, PyramidOptions};
use crate::tile::{TerrainTile, TileKey};
use crate::TerrainAssetReader;

type Result<T> = std::result::Result<T, TerrainAssetError>;

/// The level-0 changes one save has to fold into a `.inf_terrain`.
///
/// Deliberately owned and level-0 only: the caller stages it under whatever lock
/// guards the document, then releases the lock and does the (slow, whole-payload)
/// rewrite from this alone.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TerrainEdits {
    /// Tiles whose contents changed, or that were authored from nothing.
    pub changed: BTreeMap<(i32, i32), TerrainTile>,
    /// Tiles deleted outright (dropped from the asset).
    pub removed: BTreeSet<(i32, i32)>,
}

impl TerrainEdits {
    /// `true` when there is nothing to write back.
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty() && self.removed.is_empty()
    }

    /// Number of level-0 tiles this write-back touches.
    pub fn len(&self) -> usize {
        self.changed.len() + self.removed.len()
    }

    /// Collect the edits a working set's **dirty set** describes.
    ///
    /// The dirty/resident split is what distinguishes the two cases: a dirty key
    /// that is still resident was *edited*, and a dirty key that is not is one
    /// [`TerrainData::remove_tile`] deleted (an authoring delete drops residency
    /// but keeps the write-back mark). Coarse keys are ignored — the pyramid is
    /// derived, never authored.
    pub fn from_dirty(data: &TerrainData, dirty: &[TileKey]) -> Self {
        let mut out = Self::default();
        for &key in dirty {
            if !key.is_lod0() {
                continue;
            }
            match data.get_tile(key.coord) {
                Some(tile) => {
                    out.changed.insert(key.coord, tile.clone());
                }
                None => {
                    out.removed.insert(key.coord);
                }
            }
        }
        out
    }

    /// Every level-0 coordinate this write-back mentions (changed **or** removed)
    /// — the seed of the upward dirty propagation, and the exact set of tiles a
    /// caller may un-mark once the rewrite lands.
    pub fn touched(&self) -> BTreeSet<(i32, i32)> {
        self.changed
            .keys()
            .copied()
            .chain(self.removed.iter().copied())
            .collect()
    }
}

/// Merge `edits` into `source` and return the new payload image, or `None` when
/// `edits` is empty (**no dirty tiles ⇒ no write at all** — a save that changed
/// no terrain must not touch the file, so a re-save cannot perturb its bytes or
/// its mtime).
///
/// `opts` must be the options the asset's pyramid was originally built with, or
/// the rewrite will re-plan the pyramid to a different shape (correct output,
/// but a needlessly total rebuild).
pub fn rewrite_terrain_asset<B: AsRef<[u8]>>(
    source: &TerrainAssetReader<B>,
    edits: &TerrainEdits,
    opts: PyramidOptions,
) -> Result<Option<TerrainAsset>> {
    if edits.is_empty() {
        return Ok(None);
    }
    let res = source.tile_resolution();
    let mps0 = source.meters_per_sample();

    // ── level 0: the store's tiles, overridden and extended by the edits ──
    let mut level0: BTreeSet<(i32, i32)> = source
        .keys()
        .filter(|k| k.is_lod0())
        .map(|k| k.coord)
        .collect();
    level0.extend(edits.changed.keys().copied());
    for c in &edits.removed {
        level0.remove(c);
    }

    // P16.6: the rewrite carries the options it re-planned with into the new
    // header, so the *next* write-back re-plans identically. `opts` is the source
    // header's own options when it recorded them (v2) and the caller's fallback
    // when it did not (v1) — either way, recording them ends the drift.
    let mut builder = TerrainAssetBuilder::new(res, mps0)
        .with_origin(source.origin())
        .with_pyramid(opts);
    for &coord in &level0 {
        let key = TileKey::lod0(coord);
        match edits.changed.get(&coord) {
            Some(tile) => builder.insert(key, tile)?,
            // Untouched: pass the canonical blob through verbatim. Encoding is
            // deterministic, so this is byte-identical to re-encoding it — and
            // it never decodes a tile the save has no reason to look at.
            None => builder.insert_bytes(key, passthrough(source, key)?)?,
        }
    }

    // ── the pyramid: recompute only the ancestors of the touched tiles ──
    let plan = plan_pyramid(&level0, mps0, opts);
    let mut dirty = edits.touched();
    // The finer level's tiles, for the blocks this level has to reduce again.
    // Level 1 reduces level 0, which lives in `edits.changed` + the store.
    let mut prev_recomputed: BTreeMap<(i32, i32), TerrainTile> = BTreeMap::new();

    for step in &plan {
        let src_lod = step.lod - 1;
        let src_mps = step.meters_per_sample * 0.5;
        dirty = dirty
            .iter()
            .map(|&(tx, tz)| (tx.div_euclid(2), tz.div_euclid(2)))
            .filter(|c| step.coords.contains(c))
            .collect();

        let mut recomputed: BTreeMap<(i32, i32), TerrainTile> = BTreeMap::new();
        for &coarse in &step.coords {
            let key = TileKey::new(step.lod, coarse);
            let stale = dirty.contains(&coarse) || source.entry(key).is_none();
            if !stale {
                // Untouched block ⇒ the store already holds the rebuild's answer.
                builder.insert_bytes(key, passthrough(source, key)?)?;
                continue;
            }
            // Materialize just this block's ≤ 4 members and run the shared kernel.
            let mut block: BTreeMap<(i32, i32), TerrainTile> = BTreeMap::new();
            for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let fine = (2 * coarse.0 + a, 2 * coarse.1 + b);
                if let Some(tile) = fine_tile(source, edits, &prev_recomputed, src_lod, fine)? {
                    block.insert(fine, tile);
                }
            }
            let tile = downsample_block(res, src_mps, coarse, &block);
            builder.insert(key, &tile)?;
            recomputed.insert(coarse, tile);
        }
        prev_recomputed = recomputed;
    }

    builder.build().map(Some)
}

/// A tile the rewrite is carrying over unchanged.
///
/// **Bytes when it can, values when it must.** The rewrite always emits the
/// *current* schema, so a byte-for-byte copy is only correct when the source is
/// already at that schema. An older payload's blobs hold an older tile layout
/// (P19.1 appended the data maps; bincode is positional), and copying them into a
/// v3 image would produce a file whose header promises a layout its blobs do not
/// have — the one failure mode that survives every validity check and only
/// surfaces as a corrupt tile on some later load. So an older source is
/// **transcoded**: decoded through its own version and re-encoded at the current
/// one. That is a one-time cost on the first save after an upgrade, and it is why
/// the migration needs no separate pass.
///
/// Every caller has already established that `source` holds `key` (it either came
/// out of the source's own directory or survived an `entry(key).is_some()`
/// check), so a miss here is a logic error in the merge — and it fails **loudly**
/// rather than writing a zero-length blob that would decode to nothing on the
/// next load.
fn passthrough<B: AsRef<[u8]>>(source: &TerrainAssetReader<B>, key: TileKey) -> Result<Vec<u8>> {
    if source.header().schema_version == TERRAIN_ASSET_SCHEMA_VERSION {
        return source.tile_bytes(key).map(<[u8]>::to_vec).ok_or_else(|| {
            TerrainAssetError::Malformed(format!(
                "write-back tried to pass through tile {key:?}, which the source asset does not hold"
            ))
        });
    }
    let tile = source.tile(key)?.ok_or_else(|| {
        TerrainAssetError::Malformed(format!(
            "write-back tried to pass through tile {key:?}, which the source asset does not hold"
        ))
    })?;
    encode_tile(&tile)
}

/// One tile of the level a coarse level reduces: the recomputed copy when this
/// rewrite produced one, else the edited copy (level 0 only), else the store's.
///
/// `None` means the block member simply does not exist — a coverage hole, which
/// [`downsample_block`] fills flat exactly as a full rebuild would.
fn fine_tile<B: AsRef<[u8]>>(
    source: &TerrainAssetReader<B>,
    edits: &TerrainEdits,
    prev_recomputed: &BTreeMap<(i32, i32), TerrainTile>,
    lod: u32,
    coord: (i32, i32),
) -> Result<Option<TerrainTile>> {
    if let Some(tile) = prev_recomputed.get(&coord) {
        return Ok(Some(tile.clone()));
    }
    if lod == 0 {
        if edits.removed.contains(&coord) {
            return Ok(None);
        }
        if let Some(tile) = edits.changed.get(&coord) {
            return Ok(Some(tile.clone()));
        }
    }
    source.tile(TileKey::new(lod, coord))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::build_terrain_asset;
    use crate::pyramid::build_pyramid;
    use crate::{apply_brush, BrushOp, BrushParams};
    use glam::DVec2;

    fn height_fn(x: f64, z: f64) -> f64 {
        x * 0.25 - z * 0.125 + x * z * 0.001 + 5.0
    }

    /// An `n × n` authored terrain (res 5, 2 m spacing ⇒ 8 m tiles), **clean** —
    /// the state a working set is in right after it pages in from a store, which
    /// is the only state a write-back is ever asked about.
    fn terrain(n: i32) -> TerrainData {
        let mut t = TerrainData::new(5, 2.0);
        for tz in 0..n {
            for tx in 0..n {
                t.author_tile((tx, tz), height_fn);
            }
        }
        t.clear_dirty();
        t
    }

    fn asset_of(t: &TerrainData) -> TerrainAsset {
        build_terrain_asset(
            t,
            &build_pyramid(t, PyramidOptions::default()),
            PyramidOptions::default(),
        )
        .unwrap()
    }

    /// Rewrite `source` with `edits`, and assert the result is **byte-identical**
    /// to a full `build_terrain_asset` over the equivalent whole terrain.
    fn assert_matches_full_rebuild(
        source: &TerrainAsset,
        full: &TerrainData,
        edits: &TerrainEdits,
    ) {
        let partial = rewrite_terrain_asset(&source.reader(), edits, PyramidOptions::default())
            .unwrap()
            .expect("edits are non-empty");
        let expect = asset_of(full);
        assert_eq!(
            partial.header().tile_count,
            expect.header().tile_count,
            "tile counts differ"
        );
        // Per-tile first: a mismatch names the tile instead of an offset.
        let (pr, er) = (partial.reader(), expect.reader());
        for key in er.keys() {
            assert_eq!(
                pr.tile_bytes(key),
                er.tile_bytes(key),
                "tile {key:?} differs from the full rebuild"
            );
        }
        assert_eq!(
            partial.as_bytes(),
            expect.as_bytes(),
            "the partial rebuild is not byte-equal to the full one"
        );
    }

    /// **THE GATE.** A partial pyramid recompute is byte-equal to a full
    /// `build_pyramid` rebuild — across an interior edit, a corner edit, a
    /// negative-coordinate edit, and several terrain sizes.
    #[test]
    fn partial_recompute_equals_a_full_rebuild() {
        for n in [5, 8, 9] {
            for &target in &[(0, 0), (3, 2), (n - 1, n - 1)] {
                let base = terrain(n);
                let source = asset_of(&base);

                let mut edited = base.clone();
                let center = edited.tile_origin_xz(target) + DVec2::splat(4.0);
                apply_brush(
                    &mut edited,
                    BrushOp::Raise,
                    BrushParams::new(center, 6.0, 3.0),
                );
                let dirty = edited.dirty_tiles();
                assert!(!dirty.is_empty(), "the brush must have dirtied something");
                let edits = TerrainEdits::from_dirty(&edited, &dirty);
                assert_matches_full_rebuild(&source, &edited, &edits);

                // …and the rewrite touched only what it had to: a coarse tile far
                // from the edit keeps the *identical bytes* it had in the source.
                let out =
                    rewrite_terrain_asset(&source.reader(), &edits, PyramidOptions::default())
                        .unwrap()
                        .unwrap();
                let (sr, or) = (source.reader(), out.reader());
                let mut untouched_coarse = 0;
                for key in sr.keys().filter(|k| !k.is_lod0()) {
                    let far = !edits
                        .touched()
                        .iter()
                        .any(|&(tx, tz)| ancestor_of(key, (tx, tz)));
                    if far && or.tile_bytes(key).is_some() {
                        assert_eq!(or.tile_bytes(key), sr.tile_bytes(key), "{key:?} moved");
                        untouched_coarse += 1;
                    }
                }
                assert!(
                    untouched_coarse > 0,
                    "nothing was reused — {n} @ {target:?}"
                );
            }
        }
    }

    /// Is `key` the level-`key.lod` ancestor of level-0 tile `fine`?
    fn ancestor_of(key: TileKey, fine: (i32, i32)) -> bool {
        let mut k = TileKey::lod0(fine);
        while k.lod < key.lod {
            k = k.parent();
        }
        k == key
    }

    /// Every op the editor exposes, folded back and compared to a full rebuild —
    /// including the neighbourhood ops, which write across tile seams.
    #[test]
    fn every_brush_op_writes_back_byte_equal() {
        let ops = [
            BrushOp::Raise,
            BrushOp::Lower,
            BrushOp::Smooth { iterations: 2 },
            BrushOp::Flatten {
                target: crate::FlattenTarget::Mean,
            },
            BrushOp::Noise {
                seed: 7,
                frequency: 0.05,
                octaves: 3,
                amplitude: 4.0,
            },
        ];
        for op in ops {
            let base = terrain(8);
            let source = asset_of(&base);
            let mut edited = base.clone();
            // Straddle a tile seam so several tiles (and several blocks) move.
            apply_brush(
                &mut edited,
                op,
                BrushParams::new(DVec2::new(16.0, 16.0), 10.0, 2.5),
            );
            let dirty = edited.dirty_tiles();
            assert!(!dirty.is_empty(), "{op:?} changed nothing");
            assert_matches_full_rebuild(
                &source,
                &edited,
                &TerrainEdits::from_dirty(&edited, &dirty),
            );
        }
    }

    /// An **authoring op past the authored extent** grows the asset: a tile that
    /// did not exist appears in the directory, and the pyramid grows with it.
    #[test]
    fn authoring_beyond_the_extent_extends_the_asset() {
        let base = terrain(8);
        let source = asset_of(&base);
        let before = source.reader().tile_count();

        let mut edited = base.clone();
        // Well outside the authored 8×8 grid (tiles span 8 m ⇒ the grid ends at 64).
        apply_brush(
            &mut edited,
            BrushOp::Raise,
            BrushParams::new(DVec2::new(80.0, 80.0), 6.0, 5.0),
        );
        let new_tiles: Vec<(i32, i32)> = edited
            .tiles()
            .map(|(&c, _)| c)
            .filter(|c| !base.has_tile(*c))
            .collect();
        assert!(!new_tiles.is_empty(), "Raise must author new ground");

        let edits = TerrainEdits::from_dirty(&edited, &edited.dirty_tiles());
        assert_matches_full_rebuild(&source, &edited, &edits);

        let out = rewrite_terrain_asset(&source.reader(), &edits, PyramidOptions::default())
            .unwrap()
            .unwrap();
        let r = out.reader();
        assert!(r.tile_count() > before, "the directory did not grow");
        for c in new_tiles {
            assert!(
                r.tile_bytes(TileKey::lod0(c)).is_some(),
                "new tile {c:?} missing from the rewritten directory"
            );
        }
    }

    /// A **deleted** tile leaves the asset, and its ancestors are re-decimated
    /// around the hole.
    #[test]
    fn a_removed_tile_leaves_the_asset() {
        let base = terrain(8);
        let source = asset_of(&base);
        let mut edited = base.clone();
        assert!(edited.remove_tile((2, 3)).is_some());

        let edits = TerrainEdits::from_dirty(&edited, &edited.dirty_tiles());
        assert_eq!(edits.removed, [(2, 3)].into_iter().collect());
        assert!(edits.changed.is_empty());
        assert_matches_full_rebuild(&source, &edited, &edits);

        let out = rewrite_terrain_asset(&source.reader(), &edits, PyramidOptions::default())
            .unwrap()
            .unwrap();
        assert!(out.reader().tile_bytes(TileKey::lod0((2, 3))).is_none());
    }

    /// The **no-dirty save** rule: nothing to write back ⇒ no payload at all, so
    /// the caller can skip touching the file entirely.
    #[test]
    fn an_empty_edit_set_produces_no_payload() {
        let source = asset_of(&terrain(8));
        let out = rewrite_terrain_asset(
            &source.reader(),
            &TerrainEdits::default(),
            PyramidOptions::default(),
        )
        .unwrap();
        assert!(out.is_none(), "an empty edit set must not rewrite anything");

        // A clean working set produces an empty edit set, so a plain save of an
        // unedited streamed terrain is a no-op by construction.
        let mut live = TerrainData::new(5, 2.0);
        live.sync_residency(&crate::tile_range(0, (0, 0), (2, 2)), &source.reader());
        assert!(!live.has_dirty_tiles(), "paging in is not an edit");
        assert!(TerrainEdits::from_dirty(&live, &live.dirty_tiles()).is_empty());
    }

    /// Round trip: rewrite → reopen → the edited heights are what the store now
    /// serves, and a second rewrite over the *same* edits is idempotent.
    #[test]
    fn a_rewritten_asset_serves_the_edited_tiles() {
        let base = terrain(8);
        let source = asset_of(&base);
        let mut edited = base.clone();
        let probe = DVec2::new(20.0, 20.0);
        apply_brush(
            &mut edited,
            BrushOp::Raise,
            BrushParams::new(probe, 8.0, 12.0),
        );
        let edits = TerrainEdits::from_dirty(&edited, &edited.dirty_tiles());
        let out = rewrite_terrain_asset(&source.reader(), &edits, PyramidOptions::default())
            .unwrap()
            .unwrap();

        // A FRESH working set over the rewritten bytes sees the edit.
        let reader = out.reader();
        let mut fresh = TerrainData::new(5, 2.0);
        fresh.sync_residency(&crate::tile_range(0, (0, 0), (7, 7)), &reader);
        assert_eq!(fresh.height_at(probe), edited.height_at(probe));
        assert_ne!(fresh.height_at(probe), base.height_at(probe));

        // Idempotent: rewriting the same asset with the same edits reproduces it.
        let again = rewrite_terrain_asset(&reader, &edits, PyramidOptions::default())
            .unwrap()
            .unwrap();
        assert_eq!(again.as_bytes(), out.as_bytes());
    }

    /// A terrain small enough to have no pyramid at all still round-trips (the
    /// `plan_pyramid` early-return path).
    #[test]
    fn a_pyramid_less_terrain_writes_back() {
        let base = terrain(2); // 4 tiles == min_tiles ⇒ no coarse levels
        let source = asset_of(&base);
        assert_eq!(source.reader().lod_levels(), 1);
        let mut edited = base.clone();
        apply_brush(
            &mut edited,
            BrushOp::Raise,
            BrushParams::new(DVec2::new(4.0, 4.0), 3.0, 2.0),
        );
        assert_matches_full_rebuild(
            &source,
            &edited,
            &TerrainEdits::from_dirty(&edited, &edited.dirty_tiles()),
        );
    }

    /// Growing past `min_tiles` grows the pyramid: a terrain with **no** coarse
    /// levels gains them when an authoring op pushes it over the threshold, and
    /// the result still matches a full rebuild exactly.
    #[test]
    fn an_edit_that_grows_the_pyramid_matches_a_full_rebuild() {
        let base = terrain(2); // 4 tiles, no pyramid
        let source = asset_of(&base);
        let mut edited = base.clone();
        for c in [(2, 0), (2, 1), (0, 2), (1, 2), (2, 2)] {
            edited.author_tile(c, height_fn);
        }
        let edits = TerrainEdits::from_dirty(&edited, &edited.dirty_tiles());
        let out = rewrite_terrain_asset(&source.reader(), &edits, PyramidOptions::default())
            .unwrap()
            .unwrap();
        assert!(
            out.reader().lod_levels() > 1,
            "9 tiles must now carry a pyramid"
        );
        assert_matches_full_rebuild(&source, &edited, &edits);
    }

    /// Negative tile coordinates group with floor semantics all the way up the
    /// partial recompute (the `div_euclid` rule, mirrored from `pyramid`).
    #[test]
    fn negative_coordinates_write_back_byte_equal() {
        let mut base = TerrainData::new(5, 2.0);
        for tz in -4..4 {
            for tx in -4..4 {
                base.author_tile((tx, tz), height_fn);
            }
        }
        let source = asset_of(&base);
        let mut edited = base.clone();
        apply_brush(
            &mut edited,
            BrushOp::Lower,
            BrushParams::new(DVec2::new(-20.0, -12.0), 9.0, 4.0),
        );
        assert_matches_full_rebuild(
            &source,
            &edited,
            &TerrainEdits::from_dirty(&edited, &edited.dirty_tiles()),
        );
    }

    /// Painting splat weights writes back too — the weights ride in the tile blob,
    /// and coarse tiles must still carry the sparse default.
    #[test]
    fn a_paint_stroke_writes_back_byte_equal() {
        let base = terrain(8);
        let source = asset_of(&base);
        let mut edited = base.clone();
        crate::apply_paint(
            &mut edited,
            2,
            BrushParams::new(DVec2::new(20.0, 20.0), 9.0, 1.0),
        );
        let dirty = edited.dirty_tiles();
        assert!(!dirty.is_empty());
        let edits = TerrainEdits::from_dirty(&edited, &dirty);
        assert_matches_full_rebuild(&source, &edited, &edits);
        let out = rewrite_terrain_asset(&source.reader(), &edits, PyramidOptions::default())
            .unwrap()
            .unwrap();
        for key in out.reader().keys().filter(|k| !k.is_lod0()) {
            assert!(
                out.reader()
                    .tile(key)
                    .unwrap()
                    .unwrap()
                    .weights_are_default(),
                "coarse {key:?} leaked weights"
            );
        }
    }

    /// Erosion data maps ride the tile blob exactly like the splat weights do
    /// (P19.1): a bake writes back byte-equal to a full rebuild, and the derived
    /// **coarse** levels stay on the never-eroded sparse default — they are
    /// streaming pages, not authored content.
    #[test]
    fn an_erosion_bake_writes_back_byte_equal() {
        let base = terrain(8);
        let source = asset_of(&base);
        let mut edited = base.clone();
        let params = crate::ErosionParams {
            rain_rate: 0.05,
            ..crate::ErosionParams::default()
        };
        let (_, maps, _) = crate::erode_terrain(
            &mut edited,
            DVec2::new(8.0, 8.0),
            DVec2::new(40.0, 40.0),
            0,
            &params,
            25,
        );
        assert!(!maps.is_empty(), "the bake must write data maps");
        assert!(!edited.data_maps_are_default());

        let dirty = edited.dirty_tiles();
        assert!(!dirty.is_empty());
        let edits = TerrainEdits::from_dirty(&edited, &dirty);
        assert_matches_full_rebuild(&source, &edited, &edits);

        let out = rewrite_terrain_asset(&source.reader(), &edits, PyramidOptions::default())
            .unwrap()
            .unwrap();
        let r = out.reader();
        let mut eroded_lod0 = 0;
        for key in r.keys() {
            let tile = r.tile(key).unwrap().unwrap();
            if key.is_lod0() {
                if !tile.maps_are_default() {
                    eroded_lod0 += 1;
                }
            } else {
                assert!(tile.maps_are_default(), "coarse {key:?} leaked data maps");
            }
        }
        assert!(eroded_lod0 > 0, "no level-0 tile carried its maps through");
    }
}
