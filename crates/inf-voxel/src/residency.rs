//! Chunk **residency**: which chunks are in memory right now, and where the cold
//! ones come from.
//!
//! # The contract
//!
//! * **[`ChunkStore`]** is the cold side — anything that can hand back a chunk's
//!   canonical blob bytes for a [`ChunkKey`]. Two implementations ship here: an
//!   in-memory [`MemoryChunkStore`] (tests, an editor scratch buffer) and the
//!   asset-backed [`VoxelAssetReader`](crate::VoxelAssetReader), which slices the
//!   bytes out of a `.inf_voxel` payload — a pack mapping included — with no
//!   decode and no copy.
//! * **The caller computes the wants.** [`VoxelData::sync_residency`] takes an
//!   explicit set of wanted keys and no policy at all — no camera, no radius, no
//!   budget — which is what keeps residency testable as a pure set operation.
//!   P21.1 said the *policy* was the hosts'; P21.2 corrected that and put it in
//!   Ring 0 too ([`crate::wants`]), because a radius and a dead band written once
//!   in the editor and once in the player is two policies and the shipped build
//!   would stream a cave differently from its preview. The split that survives is
//!   **policy vs. execution**, not Ring 0 vs. the hosts: `wants` decides, this
//!   executes, and the hosts supply only the eye position.
//! * **Absent ≡ non-resident.** A sample in a non-resident chunk reads as empty
//!   space, exactly like one that was never carved, so streaming can never
//!   *change* an answer — only make one available.
//! * **Per-chunk versions.** Every mutation stamps the chunk it touched from a
//!   single **process-global** monotone counter, so a GPU cache re-uploads a chunk
//!   when — and only when — its contents changed. See
//!   [`VoxelData::chunk_version`].
//! * **Dirty set.** The same mutations mark the chunk dirty; a caller drains the
//!   set to write edits back. `sync_residency` refuses to evict a dirty chunk —
//!   unsaved carving is never dropped on the floor.

use std::collections::{BTreeMap, BTreeSet};

use crate::asset::{decode_chunk_at, VoxelAssetReader, VOXEL_ASSET_SCHEMA_VERSION};
use crate::chunk::{ChunkKey, VoxelChunk};
use crate::data::VoxelData;

/// The cold side of residency: a source of chunk blobs keyed by [`ChunkKey`].
///
/// Implementors hand back the chunk's **canonical bincode blob** — the same bytes
/// a chunk writes into a `.inf_voxel` — borrowed for as long as the store, so an
/// asset-backed store can serve straight out of an mmap'd pack.
pub trait ChunkStore {
    /// The chunk's blob bytes, or `None` when the store has no such chunk.
    fn chunk_bytes(&self, key: ChunkKey) -> Option<&[u8]>;

    /// Every key the store can serve, in ascending [`ChunkKey`] order.
    fn chunk_keys(&self) -> Vec<ChunkKey>;

    /// Whether the store can serve `key`.
    fn contains_chunk(&self, key: ChunkKey) -> bool {
        self.chunk_bytes(key).is_some()
    }

    /// The `.inf_voxel` schema version the store's blobs were **written at** —
    /// which is what says how to decode them (bincode is positional; see
    /// [`decode_chunk_at`]).
    ///
    /// Defaults to the current version, which is right for every store that
    /// encodes its own blobs. An asset-backed store overrides it with its
    /// payload's own stamp, so an older `.inf_voxel` keeps paging in forever.
    fn chunk_schema_version(&self) -> u32 {
        VOXEL_ASSET_SCHEMA_VERSION
    }

    /// Decode a chunk, or `None` when absent. `Err` only for a corrupt blob.
    fn load_chunk(&self, key: ChunkKey) -> Result<Option<VoxelChunk>, String> {
        match self.chunk_bytes(key) {
            None => Ok(None),
            Some(bytes) => decode_chunk_at(bytes, self.chunk_schema_version()).map(Some),
        }
    }
}

/// An in-memory [`ChunkStore`] over pre-encoded blobs — a test fixture, an editor
/// scratch buffer, or the staging area a dirty drain flushes into.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MemoryChunkStore {
    chunks: BTreeMap<ChunkKey, Vec<u8>>,
}

impl MemoryChunkStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stage a chunk, encoding it to its canonical blob. Replaces any previous one.
    pub fn insert(&mut self, key: ChunkKey, chunk: &VoxelChunk) -> Result<(), String> {
        let bytes = crate::asset::encode_chunk(chunk).map_err(|e| e.to_string())?;
        self.chunks.insert(key, bytes);
        Ok(())
    }

    /// Stage already-encoded blob bytes.
    pub fn insert_bytes(&mut self, key: ChunkKey, bytes: Vec<u8>) {
        self.chunks.insert(key, bytes);
    }

    /// Drop a chunk.
    pub fn remove(&mut self, key: ChunkKey) -> Option<Vec<u8>> {
        self.chunks.remove(&key)
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Stage every resident chunk of `data` — the "spill the whole volume into a
    /// cold store" helper.
    pub fn stage(&mut self, data: &VoxelData) -> Result<(), String> {
        for (&key, chunk) in data.chunks() {
            self.insert(key, chunk)?;
        }
        Ok(())
    }
}

impl ChunkStore for MemoryChunkStore {
    fn chunk_bytes(&self, key: ChunkKey) -> Option<&[u8]> {
        self.chunks.get(&key).map(|v| v.as_slice())
    }

    fn chunk_keys(&self) -> Vec<ChunkKey> {
        self.chunks.keys().copied().collect()
    }
}

/// The asset-backed store: a `.inf_voxel` payload served in place.
///
/// `chunk_bytes` is a binary search plus a sub-slice of the payload — so when the
/// payload is itself a borrowed `PackReader::read_ref` slice of an mmap, a chunk
/// reaches the caller with **zero copies and zero decodes**.
impl<B: AsRef<[u8]>> ChunkStore for VoxelAssetReader<B> {
    fn chunk_bytes(&self, key: ChunkKey) -> Option<&[u8]> {
        VoxelAssetReader::chunk_bytes(self, key)
    }

    fn chunk_keys(&self) -> Vec<ChunkKey> {
        self.keys().collect()
    }

    /// The payload's own stamp — an older asset's blobs hold the older chunk
    /// layout, and this is what makes them keep paging in.
    fn chunk_schema_version(&self) -> u32 {
        self.header().schema_version
    }
}

/// What one [`VoxelData::sync_residency`] pass did.
///
/// Every list is in ascending [`ChunkKey`] order, so a report is a deterministic
/// function of `(resident set, wants, store)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidencyReport {
    /// Chunks brought in from the store.
    pub loaded: Vec<ChunkKey>,
    /// Chunks dropped because they were no longer wanted.
    pub evicted: Vec<ChunkKey>,
    /// Wanted chunks the store could not serve (a coverage hole, not an error).
    pub missing: Vec<ChunkKey>,
    /// Unwanted chunks that were **kept** because they carry unsaved edits. Drain
    /// the dirty set and write them back, then sync again to release them.
    pub retained_dirty: Vec<ChunkKey>,
    /// Chunks whose blob failed to decode, with the reason. The chunk is left
    /// non-resident; a corrupt page never becomes silent geometry.
    pub failed: Vec<(ChunkKey, String)>,
}

impl ResidencyReport {
    /// `true` when nothing moved (no load, no evict, no failure).
    pub fn is_noop(&self) -> bool {
        self.loaded.is_empty() && self.evicted.is_empty() && self.failed.is_empty()
    }

    /// Number of chunks that changed residency.
    pub fn churn(&self) -> usize {
        self.loaded.len() + self.evicted.len()
    }
}

/// The set of keys a rectangular chunk box wants — the shape a caller's
/// view-dependent selection usually reduces to. Inclusive on both ends and
/// order-insensitive in its bounds. (Still *caller* policy: this is arithmetic,
/// not a camera.)
pub fn chunk_range(min: ChunkKey, max: ChunkKey) -> BTreeSet<ChunkKey> {
    let (x0, x1) = (min.x.min(max.x), min.x.max(max.x));
    let (y0, y1) = (min.y.min(max.y), min.y.max(max.y));
    let (z0, z1) = (min.z.min(max.z), min.z.max(max.z));
    let mut out = BTreeSet::new();
    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                out.insert(ChunkKey::new(x, y, z));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::build_voxel_asset;

    fn volume() -> VoxelData {
        let mut v = VoxelData::new(0.5);
        for key in chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
            v.insert_chunk(key, VoxelChunk::solid(1));
        }
        v.clear_dirty();
        v
    }

    #[test]
    fn a_memory_store_serves_encoded_chunks() {
        let v = volume();
        let mut store = MemoryChunkStore::new();
        store.stage(&v).unwrap();
        assert_eq!(store.len(), 8);
        assert_eq!(store.chunk_keys().len(), 8);
        let key = ChunkKey::new(1, 0, 1);
        assert!(store.contains_chunk(key));
        assert_eq!(
            &store.load_chunk(key).unwrap().unwrap(),
            v.get_chunk(key).unwrap()
        );
        assert!(store.load_chunk(ChunkKey::new(9, 9, 9)).unwrap().is_none());
        assert!(store.remove(key).is_some());
        assert!(!store.contains_chunk(key));
    }

    #[test]
    fn an_asset_reader_is_a_chunk_store() {
        let v = volume();
        let asset = build_voxel_asset(&v).unwrap();
        let reader = asset.reader();
        let store: &dyn ChunkStore = &reader;
        assert_eq!(store.chunk_keys().len(), 8);
        assert_eq!(store.chunk_schema_version(), VOXEL_ASSET_SCHEMA_VERSION);
        assert!(store.contains_chunk(ChunkKey::new(0, 0, 0)));
        assert_eq!(
            &store.load_chunk(ChunkKey::new(1, 1, 1)).unwrap().unwrap(),
            v.get_chunk(ChunkKey::new(1, 1, 1)).unwrap()
        );
    }

    /// A corrupt blob is a **reported failure**, never a silently empty chunk —
    /// "a bad page never becomes silent geometry".
    #[test]
    fn a_corrupt_blob_fails_loudly_and_stays_non_resident() {
        let mut store = MemoryChunkStore::new();
        let key = ChunkKey::new(0, 0, 0);
        store.insert_bytes(key, vec![0xFF, 0xFF, 0xFF]);
        let mut live = VoxelData::new(0.5);
        let r = live.sync_residency(&chunk_range(key, key), &store);
        assert_eq!(r.failed.len(), 1);
        assert_eq!(r.failed[0].0, key);
        assert!(r.loaded.is_empty());
        assert!(!live.is_resident(key));
        assert!(!r.is_noop(), "a failure is not a no-op");
    }

    #[test]
    fn chunk_range_is_inclusive_and_normalizes() {
        let a = chunk_range(ChunkKey::new(1, -1, 0), ChunkKey::new(2, 0, 0));
        assert_eq!(a.len(), 4);
        assert!(a.contains(&ChunkKey::new(1, -1, 0)));
        assert!(a.contains(&ChunkKey::new(2, 0, 0)));
        assert_eq!(
            chunk_range(ChunkKey::new(2, 0, 0), ChunkKey::new(1, -1, 0)),
            a
        );
        // Ascending order is the BTreeSet's, which is what makes every report
        // deterministic.
        let keys: Vec<_> = a.iter().copied().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }

    #[test]
    fn request_chunks_never_evicts() {
        let v = volume();
        let mut store = MemoryChunkStore::new();
        store.stage(&v).unwrap();
        let mut live = VoxelData::new(0.5);

        live.request_chunks(
            &chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(0, 0, 0)),
            &store,
        );
        let r = live.request_chunks(
            &chunk_range(ChunkKey::new(1, 1, 1), ChunkKey::new(1, 1, 1)),
            &store,
        );
        assert_eq!(r.loaded, vec![ChunkKey::new(1, 1, 1)]);
        assert!(r.evicted.is_empty() && r.retained_dirty.is_empty());
        assert_eq!(live.chunk_count(), 2, "the earlier want must survive");
        assert_eq!(r.churn(), 1);
    }
}
