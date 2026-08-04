//! [`VoxelData`]: the **resident** chunk set of one voxel volume, anchored in
//! `f64` world space.
//!
//! Deliberately the same shape as `inf_terrain::TerrainData`, down to the
//! residency vocabulary and the process-global stamp allocator, because it is the
//! same problem: a sparse `BTreeMap` working set that a caller grows and shrinks
//! against a cold store, with per-page change stamps a GPU cache gates its
//! uploads on. Anything a reader already knows about terrain streaming transfers
//! here unchanged.

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;

use crate::chunk::{ChunkKey, VoxelChunk, CHUNK_DIM, DEFAULT_MATERIAL, EMPTY_SDF};
use crate::residency::{ChunkStore, ResidencyReport};

/// A sparse SDF voxel volume: the chunks that are in memory right now.
///
/// Chunk `(x, y, z)` owns global sample coordinates `x·CHUNK_DIM + i`; the world
/// position of global sample `(gx, gy, gz)` is
/// `origin + (gx, gy, gz) · voxel_size_m`. Absent chunks read as
/// [`EMPTY_SDF`](crate::EMPTY_SDF) — **non-resident is indistinguishable from
/// never-authored**, exactly as it is for terrain tiles, so streaming can never
/// *change* an answer, only make one available.
///
/// `PartialEq` compares only the **authored** state (scale, anchor, chunks). The
/// stamps and the dirty set are runtime caches; letting them into equality would
/// make a volume look changed for having been paged.
#[derive(Clone, Debug)]
pub struct VoxelData {
    voxel_size_m: f64,
    origin: DVec3,
    chunks: BTreeMap<ChunkKey, VoxelChunk>,
    /// Change stamps for **resident** chunks only — pruned on evict/remove, so
    /// the ledger is bounded by residency rather than by everything ever
    /// streamed. Drawn from [`NEXT_CHUNK_VERSION`].
    versions: BTreeMap<ChunkKey, u64>,
    dirty: BTreeSet<ChunkKey>,
}

/// The single monotone source **every** chunk stamp in the process is drawn from.
///
/// Deliberately global rather than per-[`VoxelData`], and the reasoning is
/// verbatim `inf_terrain`'s `NEXT_TILE_VERSION`, which was paid for in a shipped
/// bug: a per-instance counter makes two different volumes mint the *same*
/// stamps — two freshly loaded levels both walk their chunk `BTreeMap` in order
/// and hand out 1, 2, 3, … — and a consumer that caches by chunk key alone (the
/// renderer's GPU buffer cache does exactly that) would then see
/// "version 1 == cached version 1" after a File → Open and keep drawing the
/// **previous level's** cave walls wherever the two volumes' chunk coordinates
/// coincide.
///
/// One global counter makes a stamp unique across all `VoxelData` instances for
/// the life of the process, so **the same stamp always means the same content**.
/// Every documented property survives: it only ever increases (so a re-paged
/// chunk's stamp strictly exceeds any it held before), `0` is never a live stamp,
/// and the per-volume ledger is still pruned on evict. At one stamp per
/// nanosecond a `u64` lasts ~584 years; `Relaxed` is sufficient because the only
/// requirement is uniqueness, not ordering against other memory.
static NEXT_CHUNK_VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl PartialEq for VoxelData {
    /// Authored state only — see the type docs.
    fn eq(&self, other: &Self) -> bool {
        self.voxel_size_m == other.voxel_size_m
            && self.origin == other.origin
            && self.chunks == other.chunks
    }
}

impl Default for VoxelData {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl VoxelData {
    /// An empty volume at the given voxel scale in **metres** (clamped to a
    /// positive finite length — a zero or NaN scale would divide by zero in every
    /// world↔grid conversion, and it is reachable from the Details grid).
    pub fn new(voxel_size_m: f64) -> Self {
        Self {
            voxel_size_m: if voxel_size_m.is_finite() && voxel_size_m > 0.0 {
                voxel_size_m
            } else {
                0.5
            },
            origin: DVec3::ZERO,
            chunks: BTreeMap::new(),
            versions: BTreeMap::new(),
            dirty: BTreeSet::new(),
        }
    }

    /// Anchor the chunk grid at an explicit world position (the world point of
    /// global sample `(0, 0, 0)`).
    pub fn with_origin(mut self, origin: DVec3) -> Self {
        self.origin = if origin.is_finite() {
            origin
        } else {
            DVec3::ZERO
        };
        self
    }

    /// World metres per voxel.
    #[inline]
    pub fn voxel_size_m(&self) -> f64 {
        self.voxel_size_m
    }

    /// World position of global sample `(0, 0, 0)`.
    #[inline]
    pub fn origin(&self) -> DVec3 {
        self.origin
    }

    /// World edge length of one chunk: `CHUNK_DIM · voxel_size_m`.
    #[inline]
    pub fn chunk_span_m(&self) -> f64 {
        CHUNK_DIM as f64 * self.voxel_size_m
    }

    /// World position of `key`'s lowest owned sample.
    #[inline]
    pub fn chunk_origin_world(&self, key: ChunkKey) -> DVec3 {
        let b = key.base_sample();
        self.origin + DVec3::new(b[0] as f64, b[1] as f64, b[2] as f64) * self.voxel_size_m
    }

    /// World position of a (possibly fractional) global grid coordinate.
    #[inline]
    pub fn grid_to_world(&self, g: DVec3) -> DVec3 {
        self.origin + g * self.voxel_size_m
    }

    /// Global grid coordinate (fractional) of a world position.
    #[inline]
    pub fn world_to_grid(&self, w: DVec3) -> DVec3 {
        (w - self.origin) / self.voxel_size_m
    }

    /// Number of resident chunks.
    #[inline]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// `true` when no chunk is resident.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Iterate resident chunks in ascending [`ChunkKey`] order.
    pub fn chunks(&self) -> impl Iterator<Item = (&ChunkKey, &VoxelChunk)> {
        self.chunks.iter()
    }

    /// Every resident key, ascending.
    pub fn resident_keys(&self) -> Vec<ChunkKey> {
        self.chunks.keys().copied().collect()
    }

    /// Whether `key` is resident.
    #[inline]
    pub fn is_resident(&self, key: ChunkKey) -> bool {
        self.chunks.contains_key(&key)
    }

    /// The chunk-Y coordinates of every resident chunk in the **vertical chunk
    /// column** `(cx, cz)`, ascending.
    ///
    /// The bound a downward column scan needs (see [`crate::ground`]): a query
    /// that marched from some fixed sky height to some fixed floor would either
    /// miss a cave dug above it or burn thousands of samples on empty space, and a
    /// query that marched `min_y ..= max_y` would burn them on the *gap* between
    /// two chunks a sparse volume left a kilometre apart. Only the resident chunks
    /// can hold a surface, so only they are walked.
    ///
    /// `Ord` on [`ChunkKey`] is lexicographic `(x, y, z)`, so the `x == cx` slab is
    /// one contiguous range but the `z == cz` column inside it is not — hence the
    /// filter. Cost is therefore the slab, not the volume, and never the whole
    /// `i32` range.
    pub fn column_chunk_ys(&self, cx: i32, cz: i32) -> Vec<i32> {
        self.chunks
            .range(ChunkKey::new(cx, i32::MIN, i32::MIN)..=ChunkKey::new(cx, i32::MAX, i32::MAX))
            .filter(|(k, _)| k.z == cz)
            .map(|(k, _)| k.y)
            .collect()
    }

    pub fn get_chunk(&self, key: ChunkKey) -> Option<&VoxelChunk> {
        self.chunks.get(&key)
    }

    /// Mutable access to a resident chunk.
    ///
    /// **Touching is mutating**: the chunk's version is bumped and it is marked
    /// dirty on the way out, because a `&mut` handle cannot tell the caller's
    /// intent. Every carve, fill and delta path funnels through here (or
    /// [`get_or_create_chunk`](Self::get_or_create_chunk)), so "stamp only the
    /// chunks under the brush" falls out for free.
    pub fn get_chunk_mut(&mut self, key: ChunkKey) -> Option<&mut VoxelChunk> {
        if !self.chunks.contains_key(&key) {
            return None;
        }
        self.touch(key);
        self.chunks.get_mut(&key)
    }

    /// Get `key`, creating it **empty** if absent, and return a mutable handle
    /// (stamped + dirtied like [`get_chunk_mut`](Self::get_chunk_mut)).
    pub fn get_or_create_chunk(&mut self, key: ChunkKey) -> &mut VoxelChunk {
        self.chunks.entry(key).or_insert_with(VoxelChunk::empty);
        self.touch(key);
        self.chunks.get_mut(&key).expect("just inserted")
    }

    /// Mutable access that **does not stamp** — the one deliberate bypass of
    /// [`get_chunk_mut`](Self::get_chunk_mut)'s "touching is mutating" door.
    ///
    /// Reserved for changes that cannot alter what any reader sees, of which
    /// there is exactly one today: compacting a material buffer back to its sparse
    /// form after an op ([`crate::ops`]). Stamping there would re-stamp a chunk the
    /// op has already stamped, which costs a redundant re-mesh, and *not* stamping
    /// is safe because the sparse and dense forms of the same values are the same
    /// values. `pub(crate)`, so the door stays closed from outside.
    #[inline]
    pub(crate) fn raw_chunk_mut(&mut self, key: ChunkKey) -> Option<&mut VoxelChunk> {
        self.chunks.get_mut(&key)
    }

    /// Insert (or replace) an **authored** chunk: stamped *and* dirtied, because
    /// this is an edit. Streaming a page in goes through
    /// [`insert_resident_chunk`](Self::insert_resident_chunk) instead.
    pub fn insert_chunk(&mut self, key: ChunkKey, chunk: VoxelChunk) -> Option<VoxelChunk> {
        self.touch(key);
        self.chunks.insert(key, chunk)
    }

    /// Delete a chunk as an **authoring** act: it stops being resident and is
    /// scheduled for write-back (so the deletion reaches the `.inf_voxel`).
    pub fn remove_chunk(&mut self, key: ChunkKey) -> Option<VoxelChunk> {
        let had = self.chunks.remove(&key);
        if had.is_some() {
            self.versions.remove(&key);
            self.dirty.insert(key);
        }
        had
    }

    // ── sampling ────────────────────────────────────────────────────────────

    /// Signed distance (voxels) at a **global integer sample**, reading
    /// [`EMPTY_SDF`] where no chunk is resident.
    #[inline]
    pub fn sample_global(&self, gx: i32, gy: i32, gz: i32) -> f32 {
        let key = ChunkKey::of_sample(gx, gy, gz);
        match self.chunks.get(&key) {
            None => EMPTY_SDF,
            Some(c) => {
                let (i, j, k) = local_of(gx, gy, gz);
                c.sample(i, j, k)
            }
        }
    }

    /// Material index at a **global integer sample**, reading
    /// [`DEFAULT_MATERIAL`] where no chunk is resident.
    #[inline]
    pub fn material_global(&self, gx: i32, gy: i32, gz: i32) -> u8 {
        let key = ChunkKey::of_sample(gx, gy, gz);
        match self.chunks.get(&key) {
            None => DEFAULT_MATERIAL,
            Some(c) => {
                let (i, j, k) = local_of(gx, gy, gz);
                c.material(i, j, k)
            }
        }
    }

    /// Write a signed distance at a global sample, creating the chunk if needed.
    pub fn set_sample_global(&mut self, gx: i32, gy: i32, gz: i32, v: f32) {
        let key = ChunkKey::of_sample(gx, gy, gz);
        let (i, j, k) = local_of(gx, gy, gz);
        self.get_or_create_chunk(key).set_sample(i, j, k, v);
    }

    /// Write a material index at a global sample, creating the chunk if needed.
    pub fn set_material_global(&mut self, gx: i32, gy: i32, gz: i32, m: u8) {
        let key = ChunkKey::of_sample(gx, gy, gz);
        let (i, j, k) = local_of(gx, gy, gz);
        self.get_or_create_chunk(key).set_material(i, j, k, m);
    }

    /// **Trilinearly** interpolated signed distance, in **voxels**, at a world
    /// position — reading *across* chunk boundaries (unlike
    /// [`VoxelChunk::sdf_at_local`], which clamps inside one chunk).
    ///
    /// Multiply by [`voxel_size_m`](Self::voxel_size_m) for an approximate
    /// distance in metres. It is approximate on purpose: the stored field is a
    /// clamped band, not a globally exact distance, so the number is meaningful
    /// near a surface and saturates away from one.
    pub fn sdf_at(&self, world: DVec3) -> f64 {
        let g = self.world_to_grid(world);
        if !g.is_finite() {
            return EMPTY_SDF as f64;
        }
        let (i0, fx) = split_global(g.x);
        let (j0, fy) = split_global(g.y);
        let (k0, fz) = split_global(g.z);
        let s = |dx: i32, dy: i32, dz: i32| self.sample_global(i0 + dx, j0 + dy, k0 + dz) as f64;
        let c00 = s(0, 0, 0) * (1.0 - fx) + s(1, 0, 0) * fx;
        let c10 = s(0, 1, 0) * (1.0 - fx) + s(1, 1, 0) * fx;
        let c01 = s(0, 0, 1) * (1.0 - fx) + s(1, 0, 1) * fx;
        let c11 = s(0, 1, 1) * (1.0 - fx) + s(1, 1, 1) * fx;
        let c0 = c00 * (1.0 - fy) + c10 * fy;
        let c1 = c01 * (1.0 - fy) + c11 * fy;
        c0 * (1.0 - fz) + c1 * fz
    }

    /// `true` when `world` is inside solid (the sampled field is negative).
    pub fn is_solid_at(&self, world: DVec3) -> bool {
        self.sdf_at(world) < 0.0
    }

    // ── stamps + residency ──────────────────────────────────────────────────

    /// Stamp `key` from the process-global counter (paging a page in stamps
    /// without dirtying).
    #[inline]
    fn stamp(&mut self, key: ChunkKey) {
        let v = NEXT_CHUNK_VERSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        self.versions.insert(key, v);
    }

    /// Stamp `key` **and** mark it dirty — the funnel every *mutation* goes
    /// through.
    #[inline]
    fn touch(&mut self, key: ChunkKey) {
        self.stamp(key);
        self.dirty.insert(key);
    }

    /// A resident chunk's change stamp; **`0` for a chunk that is not resident**
    /// (never loaded, evicted, or deleted).
    ///
    /// Stamps come from one **process-global** counter ([`NEXT_CHUNK_VERSION`])
    /// that only ever increases, so:
    ///
    /// * a stamp strictly exceeds every stamp handed out before it — including
    ///   any this chunk held in an earlier residency, so a re-paged chunk can
    ///   never be mistaken for what a GPU cache still holds;
    /// * a stamp is unique across **every** `VoxelData` in the process, so a
    ///   key-addressed cache cannot serve one volume's chunk as another's after a
    ///   level switch;
    /// * `0` is never a live stamp, so a consumer treating "0 or changed" as
    ///   "re-upload" is conservatively correct for a non-resident chunk;
    /// * the ledger is **bounded by residency**.
    ///
    /// Stamps are therefore **not reproducible across runs** — they are runtime
    /// cache identity, never content. Nothing serialized, hashed or compared for
    /// equality may read one (`PartialEq` deliberately ignores them).
    #[inline]
    pub fn chunk_version(&self, key: ChunkKey) -> u64 {
        self.versions.get(&key).copied().unwrap_or(0)
    }

    /// How many chunks currently hold a change stamp — the bounded-ledger
    /// invariant made observable.
    pub fn version_ledger_len(&self) -> usize {
        self.versions.len()
    }

    /// Make `key` resident from `store`.
    ///
    /// `Ok(true)` when the chunk is resident afterwards. An **already resident**
    /// chunk is left untouched (never re-read over live edits) and reports `true`;
    /// a chunk the store cannot serve reports `false`. `Err` only for a corrupt
    /// blob — the chunk then stays non-resident.
    pub fn request_chunk<S: ChunkStore + ?Sized>(
        &mut self,
        key: ChunkKey,
        store: &S,
    ) -> Result<bool, String> {
        if self.is_resident(key) {
            return Ok(true);
        }
        let Some(chunk) = store.load_chunk(key)? else {
            return Ok(false);
        };
        self.insert_resident(key, chunk);
        Ok(true)
    }

    /// Make an **already-decoded** chunk resident — the seam a streamer that
    /// decodes off the main thread applies its batch through.
    ///
    /// Stamped (a consumer must see it as changed) but **not** dirtied: paging a
    /// page in is not an edit and schedules no write-back.
    pub fn insert_resident_chunk(&mut self, key: ChunkKey, chunk: VoxelChunk) {
        self.insert_resident(key, chunk);
    }

    fn insert_resident(&mut self, key: ChunkKey, chunk: VoxelChunk) {
        self.stamp(key);
        self.chunks.insert(key, chunk);
    }

    /// Drop `key` from residency, returning whether it was resident.
    ///
    /// A pure paging operation: it marks nothing dirty, and it **does not clear an
    /// existing dirty mark** either.
    ///
    /// # Evicting a dirty chunk DELETES it from the asset
    ///
    /// Stated plainly because the consequence is worse than "loses the edit". A
    /// dirty mark with no resident chunk is exactly how a *deletion* is
    /// represented ([`remove_chunk`](Self::remove_chunk) leaves the same pair), so
    /// [`VoxelEdits::from_dirty`](crate::VoxelEdits::from_dirty) reads the evicted
    /// key as `removed` and the next save **erases that chunk from the
    /// `.inf_voxel`** — the carved cave, not merely the unsaved carving.
    ///
    /// [`sync_residency`](Self::sync_residency) is the safe door: it refuses to
    /// evict a dirty chunk and reports it as `retained_dirty` instead. This
    /// function is the unsafe one, kept because a caller that has *just* written
    /// the chunk back (and cleared its mark) needs it. The behaviour mirrors
    /// `inf_terrain::TerrainData::evict_tile` deliberately — one paging vocabulary
    /// for both — and this note is the part that was missing there.
    pub fn evict_chunk(&mut self, key: ChunkKey) -> bool {
        let had = self.chunks.remove(&key).is_some();
        if had {
            self.versions.remove(&key);
        }
        had
    }

    /// Make every key in `wants` resident, **evicting nothing** — the additive
    /// half of [`sync_residency`](Self::sync_residency).
    ///
    /// An edit working set pages this way rather than through `sync_residency`
    /// because a carve footprint is a *different* set on every dab: reconciling
    /// against it would page the surrounding neighbourhood out and straight back
    /// in as the brush moved, one full round trip through the store per dab.
    ///
    /// **The reason it does NOT give is data loss**, and the correction is worth
    /// keeping because the wrong version is the plausible one. An earlier draft of
    /// this comment claimed reconciling "would evict the chunks the previous dab
    /// just edited" and leave an undo step writing `before` into chunks
    /// [`VoxelDelta`](crate::VoxelDelta) would have to recreate **empty**. It would
    /// not: every op path marks what it touches dirty
    /// ([`chunk_version`](Self::chunk_version)), and `sync_residency` refuses to
    /// evict a dirty chunk — it reports it as `retained_dirty` and leaves it
    /// resident, which `sync_residency_never_evicts_a_dirty_chunk` asserts against
    /// the most hostile want set there is (the empty one). Unsaved carving is
    /// never dropped on the floor; this door exists for churn, not for safety.
    pub fn request_chunks<S: ChunkStore + ?Sized>(
        &mut self,
        wants: &BTreeSet<ChunkKey>,
        store: &S,
    ) -> ResidencyReport {
        let mut report = ResidencyReport::default();
        for &key in wants {
            if self.is_resident(key) {
                continue;
            }
            match store.load_chunk(key) {
                Ok(Some(chunk)) => {
                    self.insert_resident(key, chunk);
                    report.loaded.push(key);
                }
                Ok(None) => report.missing.push(key),
                Err(e) => report.failed.push((key, e)),
            }
        }
        report
    }

    /// Reconcile residency against an explicit set of wanted keys.
    ///
    /// Loads every want the store can serve, drops every resident chunk that is
    /// no longer wanted — **except** dirty ones, which are retained and reported
    /// so their edits can be written back first. There is deliberately no camera
    /// and no budget policy here (see [`crate::residency`]): the caller computes
    /// the wants, this executes them.
    pub fn sync_residency<S: ChunkStore + ?Sized>(
        &mut self,
        wants: &BTreeSet<ChunkKey>,
        store: &S,
    ) -> ResidencyReport {
        let mut report = ResidencyReport::default();

        // Evict first, so a sliding window frees its trailing pages before paging
        // in the leading ones (peak residency stays at the window size).
        for key in self.resident_keys() {
            if wants.contains(&key) {
                continue;
            }
            if self.dirty.contains(&key) {
                report.retained_dirty.push(key);
                continue;
            }
            if self.evict_chunk(key) {
                report.evicted.push(key);
            }
        }

        for &key in wants {
            if self.is_resident(key) {
                continue;
            }
            match store.load_chunk(key) {
                Ok(Some(chunk)) => {
                    self.insert_resident(key, chunk);
                    report.loaded.push(key);
                }
                Ok(None) => report.missing.push(key),
                Err(e) => report.failed.push((key, e)),
            }
        }
        report
    }

    /// Chunks mutated since the last drain, ascending.
    pub fn dirty_chunks(&self) -> Vec<ChunkKey> {
        self.dirty.iter().copied().collect()
    }

    /// Whether any chunk awaits write-back.
    pub fn has_dirty_chunks(&self) -> bool {
        !self.dirty.is_empty()
    }

    /// Take the dirty set, leaving it empty — the write-back seam.
    pub fn drain_dirty(&mut self) -> Vec<ChunkKey> {
        std::mem::take(&mut self.dirty).into_iter().collect()
    }

    /// Clear the dirty set **without** reading it (a fresh load is not an edit).
    pub fn clear_dirty(&mut self) {
        self.dirty.clear();
    }

    /// Clear `key`'s write-back mark **only if** its stamp still equals
    /// `version` — the concurrent-edit guard for a write-back that ran with the
    /// document unlocked. `false` means "still dirty, deliberately".
    pub fn clear_dirty_if_unchanged(&mut self, key: ChunkKey, version: u64) -> bool {
        if self.chunk_version(key) != version {
            return false;
        }
        self.dirty.remove(&key)
    }
}

/// Chunk-local sample indices of a global sample coordinate.
#[inline]
pub(crate) fn local_of(gx: i32, gy: i32, gz: i32) -> (u32, u32, u32) {
    let n = CHUNK_DIM as i32;
    (
        gx.rem_euclid(n) as u32,
        gy.rem_euclid(n) as u32,
        gz.rem_euclid(n) as u32,
    )
}

/// Split a global grid coordinate into its floor sample and fraction. Saturates
/// at the `i32` range rather than wrapping — a position that far out reads as
/// empty space, which is what it is.
#[inline]
fn split_global(v: f64) -> (i32, f64) {
    let f = v
        .floor()
        .clamp(i32::MIN as f64 + 2.0, i32::MAX as f64 - 2.0);
    (f as i32, v - f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::SDF_BAND;
    use crate::residency::{chunk_range, MemoryChunkStore};

    fn sphere_volume() -> VoxelData {
        // A solid ball of radius 6 voxels centred on global sample (8, 8, 8),
        // spanning chunks (0,0,0) and its + neighbours.
        let mut v = VoxelData::new(0.5);
        for key in chunk_range(ChunkKey::new(-1, -1, -1), ChunkKey::new(1, 1, 1)) {
            let b = key.base_sample();
            let c = VoxelChunk::from_fn(|i, j, k| {
                let p = DVec3::new(
                    (b[0] + i as i32) as f64,
                    (b[1] + j as i32) as f64,
                    (b[2] + k as i32) as f64,
                );
                p.distance(DVec3::splat(8.0)) - 6.0
            });
            v.insert_chunk(key, c);
        }
        v.clear_dirty();
        v
    }

    #[test]
    fn world_and_grid_conversions_are_inverses() {
        let v = VoxelData::new(0.25).with_origin(DVec3::new(100.0, -8.0, 3.5));
        assert_eq!(v.voxel_size_m(), 0.25);
        assert_eq!(v.chunk_span_m(), 4.0);
        assert_eq!(v.grid_to_world(DVec3::ZERO), DVec3::new(100.0, -8.0, 3.5));
        assert_eq!(
            v.chunk_origin_world(ChunkKey::new(1, 0, -1)),
            DVec3::new(104.0, -8.0, -0.5)
        );
        for g in [
            DVec3::ZERO,
            DVec3::new(1.5, -3.25, 40.0),
            DVec3::new(-70.5, 12.0, 0.125),
        ] {
            let back = v.world_to_grid(v.grid_to_world(g));
            assert!((back - g).length() < 1e-9, "{g} → {back}");
        }
        // A degenerate scale falls back rather than dividing by zero downstream.
        assert_eq!(VoxelData::new(0.0).voxel_size_m(), 0.5);
        assert_eq!(VoxelData::new(f64::NAN).voxel_size_m(), 0.5);
        assert_eq!(
            VoxelData::new(1.0)
                .with_origin(DVec3::new(f64::NAN, 0.0, 0.0))
                .origin(),
            DVec3::ZERO
        );
    }

    #[test]
    fn absent_chunks_read_as_empty_space() {
        let v = VoxelData::new(1.0);
        assert!(v.is_empty());
        assert_eq!(v.sample_global(0, 0, 0), EMPTY_SDF);
        assert_eq!(v.sample_global(-500, 900, 3), EMPTY_SDF);
        assert_eq!(v.material_global(-500, 900, 3), DEFAULT_MATERIAL);
        assert!(!v.is_solid_at(DVec3::ZERO));
        assert_eq!(v.sdf_at(DVec3::ZERO), EMPTY_SDF as f64);
        // A non-finite probe is empty space, not a NaN that poisons a caller.
        assert_eq!(v.sdf_at(DVec3::splat(f64::NAN)), EMPTY_SDF as f64);
    }

    #[test]
    fn global_samples_route_to_the_right_chunk_including_negatives() {
        let mut v = VoxelData::new(1.0);
        v.set_sample_global(-1, -1, -1, -2.0);
        v.set_sample_global(0, 0, 0, -1.0);
        assert_eq!(v.chunk_count(), 2);
        assert!(v.is_resident(ChunkKey::new(-1, -1, -1)));
        assert!(v.is_resident(ChunkKey::new(0, 0, 0)));
        assert_eq!(v.sample_global(-1, -1, -1), -2.0);
        assert_eq!(v.sample_global(0, 0, 0), -1.0);
        // The two live at different local indices in their own chunks.
        let n = CHUNK_DIM - 1;
        assert_eq!(
            v.get_chunk(ChunkKey::new(-1, -1, -1))
                .unwrap()
                .sample(n, n, n),
            -2.0
        );
        assert_eq!(
            v.get_chunk(ChunkKey::new(0, 0, 0)).unwrap().sample(0, 0, 0),
            -1.0
        );
    }

    /// `sdf_at` interpolates **across** the chunk boundary, which is the whole
    /// difference from `VoxelChunk::sdf_at_local`. A field linear in x makes the
    /// claim exact and the failure mode (clamping at the seam) obvious.
    #[test]
    fn sdf_at_interpolates_across_chunk_boundaries() {
        let mut v = VoxelData::new(1.0);
        let f = |g: f64| 0.25 * g - 3.0;
        for key in [ChunkKey::new(0, 0, 0), ChunkKey::new(1, 0, 0)] {
            let b = key.base_sample();
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|i, _, _| f((b[0] + i as i32) as f64)),
            );
        }
        let n = CHUNK_DIM as f64;
        // Straddling the seam at global x = 16: both sides and the midpoint.
        for gx in [n - 1.0, n - 0.5, n, n + 0.5, n + 1.0] {
            let got = v.sdf_at(DVec3::new(gx, 0.0, 0.0));
            assert!(
                (got - f(gx)).abs() < 1e-5,
                "at global x={gx} got {got}, analytic {}",
                f(gx)
            );
        }
    }

    /// **The global-stamp gate** (the `inf-terrain` law, restated for chunks):
    /// two volumes paging the SAME chunk coordinates must mint disjoint stamps,
    /// or a key-addressed GPU cache serves the first volume's walls after a level
    /// switch.
    #[test]
    fn distinct_volumes_mint_disjoint_stamps() {
        let src = sphere_volume();
        let mut store = MemoryChunkStore::new();
        store.stage(&src).unwrap();
        let wants = chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1));

        let mut a = VoxelData::new(0.5);
        let mut b = VoxelData::new(0.5);
        a.sync_residency(&wants, &store);
        b.sync_residency(&wants, &store);

        let stamps_a: BTreeSet<u64> = wants.iter().map(|&k| a.chunk_version(k)).collect();
        let stamps_b: BTreeSet<u64> = wants.iter().map(|&k| b.chunk_version(k)).collect();
        assert_eq!(stamps_a.len(), wants.len(), "stamps unique within a volume");
        assert_eq!(stamps_b.len(), wants.len());
        assert!(
            stamps_a.is_disjoint(&stamps_b),
            "two volumes minted overlapping stamps {stamps_a:?} / {stamps_b:?} — a \
             key-addressed cache would serve stale chunks across a level switch"
        );
        for &key in &wants {
            assert!(b.chunk_version(key) > a.chunk_version(key));
            assert!(a.chunk_version(key) > 0, "a live stamp is never 0");
        }
    }

    #[test]
    fn the_version_ledger_is_bounded_and_strictly_increasing() {
        let src = sphere_volume();
        let mut store = MemoryChunkStore::new();
        store.stage(&src).unwrap();
        let mut live = VoxelData::new(0.5);

        let key = ChunkKey::new(0, 0, 0);
        assert_eq!(live.chunk_version(key), 0, "never-loaded ⇒ no stamp");

        let wants = chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(0, 0, 0));
        let mut last = 0;
        for _ in 0..4 {
            live.sync_residency(&wants, &store);
            let v = live.chunk_version(key);
            assert!(v > last, "reload stamp {v} must exceed the previous {last}");
            last = v;
            assert_eq!(live.version_ledger_len(), 1);

            live.sync_residency(&BTreeSet::new(), &store);
            assert_eq!(live.chunk_version(key), 0, "evicted ⇒ stamp pruned");
            assert_eq!(live.version_ledger_len(), 0, "ledger bounded by residency");
        }

        // A mutation stamps only the chunk under it.
        let wants = chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 0, 0));
        live.sync_residency(&wants, &store);
        let other = ChunkKey::new(1, 0, 0);
        let before_other = live.chunk_version(other);
        let before = live.chunk_version(key);
        live.get_chunk_mut(key).unwrap().set_sample(1, 1, 1, -1.0);
        assert!(live.chunk_version(key) > before);
        assert_eq!(live.chunk_version(other), before_other);
        assert_eq!(live.dirty_chunks(), vec![key]);

        // Paging in is NOT an edit …
        live.clear_dirty();
        live.evict_chunk(other);
        live.request_chunk(other, &store).unwrap();
        assert!(
            !live.has_dirty_chunks(),
            "streaming must not schedule a write-back"
        );
        // … but an authoring delete is.
        live.remove_chunk(other);
        assert_eq!(live.chunk_version(other), 0);
        assert!(live.dirty_chunks().contains(&other));
    }

    #[test]
    fn sync_residency_loads_evicts_and_reports() {
        let src = sphere_volume();
        let mut store = MemoryChunkStore::new();
        store.stage(&src).unwrap();
        let mut live = VoxelData::new(0.5);

        let wants = chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(0, 0, 1));
        let r = live.sync_residency(&wants, &store);
        assert_eq!(r.loaded.len(), 2);
        assert!(r.evicted.is_empty() && r.missing.is_empty());
        assert_eq!(live.chunk_count(), 2);

        // Idempotent.
        assert!(live.sync_residency(&wants, &store).is_noop());

        // Slide the window.
        let next = chunk_range(ChunkKey::new(0, 0, 1), ChunkKey::new(0, 0, 1));
        let r = live.sync_residency(&next, &store);
        assert_eq!(r.evicted, vec![ChunkKey::new(0, 0, 0)]);
        assert!(r.loaded.is_empty());
        assert_eq!(live.resident_keys(), vec![ChunkKey::new(0, 0, 1)]);

        // A want the store cannot serve is reported, not an error.
        let mut hole = next.clone();
        hole.insert(ChunkKey::new(40, 40, 40));
        let r = live.sync_residency(&hole, &store);
        assert_eq!(r.missing, vec![ChunkKey::new(40, 40, 40)]);
        assert!(!live.is_resident(ChunkKey::new(40, 40, 40)));
    }

    #[test]
    fn sync_residency_never_evicts_a_dirty_chunk() {
        let src = sphere_volume();
        let mut store = MemoryChunkStore::new();
        store.stage(&src).unwrap();
        let mut live = VoxelData::new(0.5);
        let wants = chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(0, 0, 1));
        live.sync_residency(&wants, &store);

        let key = ChunkKey::new(0, 0, 0);
        live.get_chunk_mut(key)
            .unwrap()
            .set_sample(2, 2, 2, -SDF_BAND);
        assert!(live.dirty_chunks().contains(&key));

        let r = live.sync_residency(&BTreeSet::new(), &store);
        assert_eq!(r.retained_dirty, vec![key]);
        assert_eq!(r.evicted, vec![ChunkKey::new(0, 0, 1)]);
        assert!(live.is_resident(key));
        assert_eq!(live.get_chunk(key).unwrap().sample(2, 2, 2), -SDF_BAND);

        // Write back + drain, then it releases.
        let drained = live.drain_dirty();
        assert_eq!(drained, vec![key]);
        for k in drained {
            store.insert(k, live.get_chunk(k).unwrap()).unwrap();
        }
        let r = live.sync_residency(&BTreeSet::new(), &store);
        assert_eq!(r.evicted, vec![key]);
        assert!(live.is_empty());
        assert_eq!(
            store.load_chunk(key).unwrap().unwrap().sample(2, 2, 2),
            -SDF_BAND
        );
    }

    /// The concurrent-edit guard: a write-back that ran with the document
    /// unlocked must not clear a mark for content that moved underneath it.
    #[test]
    fn clear_dirty_if_unchanged_refuses_a_restamped_chunk() {
        let mut v = VoxelData::new(1.0);
        let key = ChunkKey::new(0, 0, 0);
        v.get_or_create_chunk(key).set_sample(0, 0, 0, -1.0);
        let staged = v.chunk_version(key);
        // Nothing moved → the mark clears.
        assert!(v.clear_dirty_if_unchanged(key, staged));
        assert!(!v.has_dirty_chunks());

        v.get_chunk_mut(key).unwrap().set_sample(0, 0, 0, -2.0);
        // The staged stamp is now stale → the mark survives, deliberately.
        assert!(!v.clear_dirty_if_unchanged(key, staged));
        assert!(v.dirty_chunks().contains(&key));
    }

    /// Equality is **authored state only** — a volume that has merely been paged
    /// (or stamped) must not read as different content.
    #[test]
    fn equality_ignores_stamps_and_dirt() {
        let a = sphere_volume();
        let mut b = a.clone();
        assert_eq!(a, b);
        // Touch every chunk without changing a sample.
        for key in b.resident_keys() {
            let _ = b.get_chunk_mut(key);
        }
        assert!(b.has_dirty_chunks());
        assert_eq!(a, b, "stamps and dirt are runtime state, not content");
        // A real edit is a real difference.
        b.get_chunk_mut(ChunkKey::new(0, 0, 0))
            .unwrap()
            .set_sample(0, 0, 0, -1.0);
        assert_ne!(a, b);
    }
}
