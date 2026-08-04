//! Folding edits back into a `.inf_voxel` — **without decoding what did not
//! change**.
//!
//! A carved volume is a handful of dirty chunks against a payload that may hold
//! thousands. [`rewrite_voxel_asset`] therefore reads the source's directory,
//! copies every untouched chunk's blob **verbatim** (a slice, no decode, no
//! re-encode), re-encodes only the chunks that were edited, drops the ones that
//! were deleted, and rebuilds the container around them.
//!
//! Two properties make that safe, and both are asserted below:
//!
//! * **Verbatim really is verbatim.** An untouched chunk's blob comes out of the
//!   new payload byte-for-byte identical to the one that went in. That is not an
//!   optimization detail — it is what stops a save from silently re-encoding
//!   (and, on some future schema, *re-interpreting*) content the author never
//!   touched.
//! * **Nothing to do is `Ok(None)`, not an empty rewrite.** A save with no edits
//!   must not rewrite the file at all: rewriting would churn the content hash, the
//!   asset database, and a git diff, for bytes that are identical. The `None` is
//!   the caller's signal to skip the write entirely — the same shape, and the same
//!   reasoning, as `inf_terrain::rewrite_terrain_asset`.

use std::collections::{BTreeMap, BTreeSet};

use crate::asset::{VoxelAsset, VoxelAssetBuilder, VoxelAssetError, VoxelAssetReader};
use crate::chunk::{ChunkKey, VoxelChunk};
use crate::data::VoxelData;

/// The edits a save is folding into a `.inf_voxel`.
///
/// Deliberately explicit rather than "diff the two volumes": a streamed volume's
/// working set is only the chunks that were paged in, so a diff would read every
/// chunk that is merely *absent* as deleted and erase the parts of the cave the
/// author never visited.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VoxelEdits {
    /// Chunks whose contents changed (or that are new).
    pub changed: BTreeMap<ChunkKey, VoxelChunk>,
    /// Chunks the author deleted.
    pub removed: BTreeSet<ChunkKey>,
}

impl VoxelEdits {
    /// `true` when there is nothing to fold in.
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty() && self.removed.is_empty()
    }

    /// Stage the volume's **dirty** chunks (and drain the marks), which is what a
    /// save actually wants: exactly what was carved since the last one.
    ///
    /// A dirty key with no resident chunk is a deletion — that is precisely what
    /// [`VoxelData::remove_chunk`] leaves behind — so the two halves fall out of
    /// one walk.
    pub fn from_dirty(data: &mut VoxelData) -> Self {
        let mut edits = Self::default();
        for key in data.drain_dirty() {
            match data.get_chunk(key) {
                Some(c) => {
                    edits.changed.insert(key, c.clone());
                }
                None => {
                    edits.removed.insert(key);
                }
            }
        }
        edits
    }
}

/// Rewrite `source` with `edits` folded in.
///
/// Returns `Ok(None)` when `edits` is empty — **do not write the file** (see the
/// module docs). Otherwise returns the new payload image, ready for
/// [`write_voxel_asset`](crate::write_voxel_asset).
///
/// A chunk that is both changed and removed is **removed**: a deletion is the
/// later, more destructive intent, and resolving it here rather than at every call
/// site means the two sets can be built independently.
pub fn rewrite_voxel_asset<B: AsRef<[u8]>>(
    source: &VoxelAssetReader<B>,
    edits: &VoxelEdits,
) -> Result<Option<VoxelAsset>, VoxelAssetError> {
    if edits.is_empty() {
        return Ok(None);
    }

    let mut builder = VoxelAssetBuilder::new(source.voxel_size_m()).with_origin(source.origin());

    // Every chunk already in the payload, in directory order.
    for e in source.directory() {
        if edits.removed.contains(&e.key) {
            continue;
        }
        match edits.changed.get(&e.key) {
            // Edited: re-encode.
            Some(c) => builder.insert(e.key, c)?,
            // Untouched: the blob rides through as a slice. No decode, no
            // re-encode, no opportunity to reinterpret it.
            None => builder.insert_bytes(
                e.key,
                source
                    .chunk_bytes(e.key)
                    .expect("directory entry resolves")
                    .to_vec(),
            )?,
        }
    }
    // Chunks the edits authored that the source never had.
    for (&key, c) in &edits.changed {
        if edits.removed.contains(&key) || source.entry(key).is_some() {
            continue;
        }
        builder.insert(key, c)?;
    }

    builder.build().map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::build_voxel_asset;
    use crate::chunk::CHUNK_DIM;
    use crate::residency::chunk_range;
    use glam::DVec3;

    fn source_volume() -> VoxelData {
        let mut v = VoxelData::new(0.5).with_origin(DVec3::new(3.0, 0.0, -7.0));
        for key in chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 0, 1)) {
            let b = key.base_sample();
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|i, j, k| {
                    DVec3::new(
                        (b[0] + i as i32) as f64,
                        (b[1] + j as i32) as f64,
                        (b[2] + k as i32) as f64,
                    )
                    .distance(DVec3::new(16.0, 8.0, 16.0))
                        - 12.0
                }),
            );
        }
        v.clear_dirty();
        v
    }

    #[test]
    fn no_edits_means_no_rewrite() {
        let asset = build_voxel_asset(&source_volume()).unwrap();
        assert_eq!(
            rewrite_voxel_asset(&asset.reader(), &VoxelEdits::default()).unwrap(),
            None,
            "an untouched volume must not churn its bytes, its content hash or a \
             git diff"
        );
    }

    /// **The verbatim gate**: every chunk the edits did not name comes out of the
    /// rewrite byte-for-byte identical to the blob that went in.
    #[test]
    fn untouched_chunks_ride_through_verbatim() {
        let v = source_volume();
        let asset = build_voxel_asset(&v).unwrap();
        let src = asset.reader();

        let edited_key = ChunkKey::new(1, 0, 1);
        let mut edited = v.get_chunk(edited_key).unwrap().clone();
        edited.set_sample(2, 2, 2, -3.5);
        edited.set_material(2, 2, 2, 2);
        let edits = VoxelEdits {
            changed: BTreeMap::from([(edited_key, edited.clone())]),
            removed: BTreeSet::new(),
        };

        let out = rewrite_voxel_asset(&src, &edits).unwrap().unwrap();
        let dst = out.reader();
        assert_eq!(dst.chunk_count(), src.chunk_count());
        assert_eq!(dst.voxel_size_m(), src.voxel_size_m());
        assert_eq!(dst.origin(), src.origin());

        for key in src.keys() {
            if key == edited_key {
                assert_eq!(dst.chunk(key).unwrap().unwrap(), edited);
                assert_ne!(dst.chunk_bytes(key), src.chunk_bytes(key));
                continue;
            }
            assert_eq!(
                dst.chunk_bytes(key),
                src.chunk_bytes(key),
                "{key:?} was re-encoded despite never being edited"
            );
        }
        // The result is a real, valid payload that round-trips.
        assert_eq!(
            build_voxel_asset(&dst.to_voxel_data().unwrap())
                .unwrap()
                .as_bytes(),
            out.as_bytes()
        );
    }

    #[test]
    fn additions_and_deletions_both_land() {
        let v = source_volume();
        let asset = build_voxel_asset(&v).unwrap();
        let src = asset.reader();
        let before = src.chunk_count();

        let added = ChunkKey::new(5, 5, 5);
        let dropped = ChunkKey::new(0, 0, 0);
        let edits = VoxelEdits {
            changed: BTreeMap::from([(added, VoxelChunk::solid(1))]),
            removed: BTreeSet::from([dropped]),
        };
        let out = rewrite_voxel_asset(&src, &edits).unwrap().unwrap();
        let dst = out.reader();
        assert_eq!(dst.chunk_count(), before);
        assert!(dst.chunk_bytes(dropped).is_none());
        assert_eq!(dst.chunk(added).unwrap().unwrap(), VoxelChunk::solid(1));
        // The directory is still ascending after the churn (the binary search and
        // the byte-determinism guarantee both depend on it).
        let keys: Vec<_> = dst.keys().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    /// A chunk that is both changed and removed is removed — deletion wins, once,
    /// here, rather than at every call site.
    #[test]
    fn deletion_wins_over_a_simultaneous_change() {
        let v = source_volume();
        let asset = build_voxel_asset(&v).unwrap();
        let key = ChunkKey::new(1, 0, 0);
        let edits = VoxelEdits {
            changed: BTreeMap::from([(key, VoxelChunk::solid(2))]),
            removed: BTreeSet::from([key]),
        };
        let out = rewrite_voxel_asset(&asset.reader(), &edits)
            .unwrap()
            .unwrap();
        assert!(out.reader().chunk_bytes(key).is_none());
    }

    /// The save path end to end: carve, stage the dirty chunks, rewrite, reload —
    /// and the reloaded volume equals the carved one exactly.
    #[test]
    fn a_carve_round_trips_through_the_write_back_path() {
        let mut v = source_volume();
        let asset = build_voxel_asset(&v).unwrap();

        let (report, _) = v.apply_op(&crate::VoxelOp::carve(crate::VoxelShape::Sphere {
            center: v.grid_to_world(DVec3::new(16.0, 8.0, 16.0)),
            radius_m: 3.0,
        }));
        assert!(report.total_carved() > 0);
        assert!(v.has_dirty_chunks());

        let edits = VoxelEdits::from_dirty(&mut v);
        assert!(!edits.is_empty());
        assert!(edits.removed.is_empty());
        assert!(!v.has_dirty_chunks(), "staging drains the marks");

        let out = rewrite_voxel_asset(&asset.reader(), &edits)
            .unwrap()
            .unwrap();
        let back = out.reader().to_voxel_data().unwrap();
        assert_eq!(back, v, "the saved volume must equal the carved one");
        // …and saving again with nothing new changes nothing.
        assert_eq!(
            rewrite_voxel_asset(&out.reader(), &VoxelEdits::default()).unwrap(),
            None
        );
    }

    /// A deleted chunk reaches the asset through the same dirty walk — the case a
    /// "diff the resident sets" staging would get catastrophically wrong on a
    /// streamed volume.
    #[test]
    fn from_dirty_reports_a_deleted_chunk_as_removed() {
        let mut v = source_volume();
        let asset = build_voxel_asset(&v).unwrap();
        let key = ChunkKey::new(0, 0, 1);
        v.remove_chunk(key);

        let edits = VoxelEdits::from_dirty(&mut v);
        assert_eq!(edits.removed, BTreeSet::from([key]));
        assert!(edits.changed.is_empty());

        let out = rewrite_voxel_asset(&asset.reader(), &edits)
            .unwrap()
            .unwrap();
        assert!(out.reader().chunk_bytes(key).is_none());
        assert_eq!(out.reader().chunk_count(), asset.reader().chunk_count() - 1);
        // Every surviving chunk is still verbatim.
        for k in out.reader().keys() {
            assert_eq!(out.reader().chunk_bytes(k), asset.reader().chunk_bytes(k));
        }
        let _ = CHUNK_DIM;
    }
}
