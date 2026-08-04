//! The **undo contract** for carving: a sparse, chunk-partitioned before/after
//! record of the samples an op changed.
//!
//! ## Representation & memory rationale
//!
//! A carve is stored as a set of **per-chunk dense sub-box patches**
//! ([`ChunkPatch`]): for every chunk the op touched, the axis-aligned bounding box
//! of the changed samples, with flat `before`/`after` buffers for that box. This
//! is deliberately *not* a `Vec<(chunk, index, before, after)>` of individual
//! samples.
//!
//! Consider the worst case the design must survive — a 16 m-radius carve at 0.5 m
//! voxels touches a ~64³ sample ball (~2.7 × 10⁵ samples, and the band around it
//! pushes that toward 10⁶):
//!
//! * **Sparse per-sample**: each entry needs a chunk key + sample index + two
//!   `f32` + two `u8` (~26 B) → ~26 MB, and every write chases a `BTreeMap` entry.
//! * **Dense sub-box**: 10 bytes per sample in flat buffers → ~10 MB, and
//!   apply/revert is a straight strided copy into the chunk — cache-friendly and
//!   allocation-light.
//!
//! A sphere fills a ball, so a chunk's touched samples nearly fill their bounding
//! box; the dense form's only waste is the ~48 % of the box outside the ball
//! (stored as `before == after` no-ops), which is still cheaper than per-sample
//! coordinate overhead **and** is bounded — the box can never exceed one chunk,
//! because the patches are per chunk. That bound is the real reason for the split:
//! a single global bounding box around a long tunnel would be almost entirely
//! no-ops.
//!
//! `created_chunks` records chunks the op **materialized from nothing** so that
//! [`VoxelData::revert_delta`] can remove them again — a carve that grew the
//! volume undoes back to byte-identical empty space, which is what makes the
//! round-trip assertion in [`crate::ops`] a *byte* assertion rather than a
//! geometric one.
//!
//! This is one level down from the editor's `EditCommand::SetTiles`, exactly as
//! `inf_terrain::HeightDelta` is, and P21.3 will hang the undo stack off it.

use serde::{Deserialize, Serialize};

use crate::chunk::{ChunkKey, VoxelChunk};
use crate::data::VoxelData;

/// A dense before/after patch over one chunk's
/// `[i0, i0+w) × [j0, j0+h) × [k0, k0+d)` sample box.
///
/// The four buffers are row-major over that box with **x fastest, then y, then
/// z** — the chunk's own layout — and are each `w · h · d` long.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChunkPatch {
    /// Chunk this patch belongs to.
    pub key: ChunkKey,
    /// Lowest sample column of the box.
    pub i0: u32,
    /// Lowest sample row of the box.
    pub j0: u32,
    /// Lowest sample slice of the box.
    pub k0: u32,
    /// Box extent on x, in samples.
    pub w: u32,
    /// Box extent on y, in samples.
    pub h: u32,
    /// Box extent on z, in samples.
    pub d: u32,
    /// Pre-op signed distances.
    pub sdf_before: Vec<f32>,
    /// Post-op signed distances.
    pub sdf_after: Vec<f32>,
    /// Pre-op material indices.
    pub mat_before: Vec<u8>,
    /// Post-op material indices.
    pub mat_after: Vec<u8>,
}

impl ChunkPatch {
    /// Samples in this patch's box.
    #[inline]
    pub fn sample_count(&self) -> usize {
        (self.w as usize) * (self.h as usize) * (self.d as usize)
    }

    /// `true` when every buffer is the length the box declares — the invariant
    /// [`VoxelData::apply_delta`] relies on, checked rather than assumed because a
    /// patch can arrive from a deserialized undo stack.
    pub fn is_well_formed(&self) -> bool {
        let n = self.sample_count();
        n > 0
            && self.sdf_before.len() == n
            && self.sdf_after.len() == n
            && self.mat_before.len() == n
            && self.mat_after.len() == n
            && self.i0 + self.w <= crate::CHUNK_DIM
            && self.j0 + self.h <= crate::CHUNK_DIM
            && self.k0 + self.d <= crate::CHUNK_DIM
    }
}

/// The reversible record an op returns — apply it for redo, revert it for undo.
/// Empty (no patches, no created chunks) when the op changed nothing.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct VoxelDelta {
    /// One dense sub-box patch per touched chunk, in ascending key order.
    pub patches: Vec<ChunkPatch>,
    /// Chunks this op materialized from nothing (removed on revert). Ascending.
    pub created_chunks: Vec<ChunkKey>,
}

impl VoxelDelta {
    /// `true` when the op changed no samples and materialized no chunks.
    pub fn is_empty(&self) -> bool {
        self.patches.is_empty() && self.created_chunks.is_empty()
    }

    /// Total samples stored across all patches (bounding-box inclusive).
    pub fn sample_count(&self) -> usize {
        self.patches.iter().map(ChunkPatch::sample_count).sum()
    }

    /// Approximate heap footprint of the stored before/after buffers, in bytes:
    /// two `f32` plus two `u8` per sample.
    pub fn memory_bytes(&self) -> usize {
        self.sample_count() * (2 * std::mem::size_of::<f32>() + 2)
    }
}

impl VoxelData {
    /// Apply (redo) a [`VoxelDelta`]: write each patch's `after` buffers,
    /// (re)creating any chunk it targets. Byte-identical to the state right after
    /// the op that produced `delta`.
    pub fn apply_delta(&mut self, delta: &VoxelDelta) {
        for patch in &delta.patches {
            if !patch.is_well_formed() {
                continue;
            }
            let chunk = self.get_or_create_chunk(patch.key);
            write_patch(chunk, patch, &patch.sdf_after, &patch.mat_after);
        }
    }

    /// Revert (undo) a [`VoxelDelta`]: write each patch's `before` buffers, then
    /// remove the chunks the op created (now empty again) so the volume returns
    /// **byte-identical** to before the op — including its chunk set.
    pub fn revert_delta(&mut self, delta: &VoxelDelta) {
        for patch in &delta.patches {
            if !patch.is_well_formed() {
                continue;
            }
            let chunk = self.get_or_create_chunk(patch.key);
            write_patch(chunk, patch, &patch.sdf_before, &patch.mat_before);
        }
        for &key in &delta.created_chunks {
            self.remove_chunk(key);
        }
    }
}

/// Blit a patch's `w · h · d` buffers into `chunk`'s sample box.
///
/// Compacts the material buffer afterwards, so a chunk whose paint was undone
/// returns to the sparse form and therefore to its original bytes — the half of
/// "reverts byte-identically" that a naive blit would miss.
fn write_patch(chunk: &mut VoxelChunk, patch: &ChunkPatch, sdf: &[f32], mat: &[u8]) {
    for dk in 0..patch.d {
        for dj in 0..patch.h {
            for di in 0..patch.w {
                let idx = ((dk * patch.h + dj) * patch.w + di) as usize;
                let (i, j, k) = (patch.i0 + di, patch.j0 + dj, patch.k0 + dk);
                chunk.set_sample(i, j, k, sdf[idx]);
                chunk.set_material(i, j, k, mat[idx]);
            }
        }
    }
    chunk.compact_materials();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{CHUNK_DIM, EMPTY_SDF};

    fn patch(key: ChunkKey) -> ChunkPatch {
        ChunkPatch {
            key,
            i0: 1,
            j0: 2,
            k0: 3,
            w: 2,
            h: 2,
            d: 2,
            sdf_before: vec![EMPTY_SDF; 8],
            sdf_after: vec![-1.0; 8],
            mat_before: vec![0; 8],
            mat_after: vec![1; 8],
        }
    }

    #[test]
    fn a_delta_applies_and_reverts_to_byte_identical_bytes() {
        let key = ChunkKey::new(0, 0, 0);
        let mut v = VoxelData::new(0.5);
        v.insert_chunk(key, VoxelChunk::empty());
        let before = crate::asset::encode_chunk(v.get_chunk(key).unwrap()).unwrap();

        let delta = VoxelDelta {
            patches: vec![patch(key)],
            created_chunks: Vec::new(),
        };
        v.apply_delta(&delta);
        assert_eq!(v.get_chunk(key).unwrap().sample(1, 2, 3), -1.0);
        assert_eq!(v.get_chunk(key).unwrap().material(1, 2, 3), 1);
        assert_ne!(
            crate::asset::encode_chunk(v.get_chunk(key).unwrap()).unwrap(),
            before
        );

        v.revert_delta(&delta);
        assert_eq!(
            crate::asset::encode_chunk(v.get_chunk(key).unwrap()).unwrap(),
            before,
            "an undo must return the chunk to its exact bytes, sparse material \
             buffer included"
        );
        assert!(v.get_chunk(key).unwrap().materials_are_default());
    }

    /// A chunk the op materialized is **removed** on revert, so the volume's chunk
    /// set — not merely its samples — comes back.
    #[test]
    fn reverting_removes_the_chunks_the_op_created() {
        let key = ChunkKey::new(4, -2, 7);
        let mut v = VoxelData::new(0.5);
        assert!(v.is_empty());

        let delta = VoxelDelta {
            patches: vec![patch(key)],
            created_chunks: vec![key],
        };
        v.apply_delta(&delta);
        assert_eq!(v.chunk_count(), 1);
        v.revert_delta(&delta);
        assert!(
            v.is_empty(),
            "the volume must return to empty space, not to an empty chunk that \
             would still cost a directory entry and a blob"
        );
    }

    #[test]
    fn a_malformed_patch_is_skipped_rather_than_panicking() {
        let key = ChunkKey::new(0, 0, 0);
        let mut v = VoxelData::new(0.5);
        v.insert_chunk(key, VoxelChunk::empty());

        for bad in [
            ChunkPatch {
                sdf_after: vec![-1.0; 3], // short buffer
                ..patch(key)
            },
            ChunkPatch {
                i0: CHUNK_DIM - 1, // box runs off the chunk
                ..patch(key)
            },
            ChunkPatch {
                w: 0, // empty box
                ..patch(key)
            },
        ] {
            assert!(!bad.is_well_formed());
            let d = VoxelDelta {
                patches: vec![bad],
                created_chunks: Vec::new(),
            };
            v.apply_delta(&d);
            v.revert_delta(&d);
        }
        assert_eq!(v.get_chunk(key).unwrap().sample(1, 2, 3), EMPTY_SDF);
        assert!(patch(key).is_well_formed());
    }

    #[test]
    fn a_delta_serializes_and_reports_its_footprint() {
        let d = VoxelDelta {
            patches: vec![patch(ChunkKey::new(1, 0, 0))],
            created_chunks: vec![ChunkKey::new(1, 0, 0)],
        };
        assert!(!d.is_empty());
        assert_eq!(d.sample_count(), 8);
        assert_eq!(d.memory_bytes(), 8 * 10);
        assert!(VoxelDelta::default().is_empty());

        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(serde_json::from_str::<VoxelDelta>(&json).unwrap(), d);
        let cfg = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(&d, cfg).unwrap();
        let (back, _): (VoxelDelta, usize) =
            bincode::serde::decode_from_slice(&bytes, cfg).unwrap();
        assert_eq!(back, d);
    }
}
