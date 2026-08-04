//! [`VoxelVolumes`]: the per-entity working set both hosts hold, so they cannot
//! hold *different* ones.
//!
//! A host has one job here — hand over the `.inf_voxel` bytes for a volume's
//! asset — and that job is genuinely host-local (the editor reads a loose file
//! under the content root; the shipped player `read`s a pack entry). **Everything
//! after the bytes** — parsing, residency, meshing, invalidation, eviction — lives
//! here, in Ring 0, because a rule written twice is a rule that drifts, and the
//! last two phases' mirror gates exist to catch exactly that.
//!
//! # Residency is camera-driven (P21.2)
//!
//! P21.1 shipped this store paging every chunk of an asset at [`ensure`] time and
//! said so: a wants policy needed a camera, a budget and an eviction gate on both
//! hosts, which is the work P21.2 names. This is that work, and it landed as the
//! module docs promised — **one call replaced**, not a second residency
//! implementation.
//!
//! [`ensure`](VoxelVolumes::ensure) now **binds** a volume: it parses the payload,
//! keeps it as the cold store, indexes its chunk directory, and pages *nothing*.
//! [`sync_camera`](VoxelVolumes::sync_camera) then executes the policy from
//! [`crate::wants`] against every bound volume. The consequence is worth stating
//! because it is a real obligation on the two hosts: **a bound volume with no
//! `sync_camera` draws nothing.** Both hosts therefore call the two in the same
//! pre-pass, one after the other, and the source-text mirror gate
//! (`projector_mirror`) requires both calls on both sides.
//!
//! Why hold the payload at all, when P21.1's player registry deliberately did not?
//! Because streaming created the reader it lacked. A chunk blob is a **sub-slice**
//! of the payload — no decode, no copy — so the cold store is the payload, exactly
//! as `inf_terrain`'s `FileTileStore` is. The decoded set is a subset of it, which
//! is the whole point.
//!
//! [`ensure`]: VoxelVolumes::ensure

use std::collections::{BTreeMap, BTreeSet};

use glam::DVec3;

use crate::asset::VoxelAssetReader;
use crate::chunk::ChunkKey;
use crate::data::VoxelData;
use crate::mesh::VoxelMeshCache;
use crate::wants::{
    advance_wants, chunk_wants, clamp_wants, ChunkCatalog, ChunkGrid, VoxelStreamBudget,
    VoxelWantsParams,
};

/// One volume's loaded state: its cold store, its resident chunks and its meshed
/// surface.
#[derive(Debug, Clone)]
pub struct VolumeSlot {
    /// The `.inf_voxel` asset id these chunks came from (as a `u128`, so Ring 0
    /// needs no `uuid` dependency — both hosts hand over `Uuid::as_u128`).
    pub asset: u128,
    /// The resident chunks.
    pub data: VoxelData,
    /// The meshed surface, re-meshed only where the field moved.
    pub meshes: VoxelMeshCache,
    /// World translation of the entity carrying this volume, as the host last
    /// [`placed`](VoxelVolumes::place) it.
    ///
    /// Deliberately **not** folded into [`data`](Self::data)'s anchor: the
    /// projectors emit `chunk_origin_world(key) + translation`, so an anchor that
    /// already contained the translation would place every cave twice. It is
    /// folded into [`grid`](Self::grid) instead, which is the only thing that
    /// needs the world position — the camera distance a want is measured by.
    pub translation: DVec3,
    /// The cold side: the whole `.inf_voxel` payload, served in place. A chunk
    /// blob is a sub-slice of it, so paging costs a binary search and a decode of
    /// one 20 KB block.
    store: VoxelAssetReader<Vec<u8>>,
    /// Which chunks the payload can serve — parsed once at bind, never re-walked.
    catalog: ChunkCatalog,
    /// The world chunk grid wants are measured against (asset scale + asset anchor
    /// + [`translation`](Self::translation)).
    grid: ChunkGrid,
    /// The currently published want set — also the hysteresis state, which is why
    /// it is kept rather than recomputed from residency: a chunk the budget has
    /// not reached yet is *wanted* and not *resident*, and feeding residency back
    /// into the dead-band test would let the budget change the policy.
    wants: BTreeSet<ChunkKey>,
}

impl VolumeSlot {
    /// The world chunk grid this volume's wants are measured against.
    #[inline]
    pub fn grid(&self) -> ChunkGrid {
        self.grid
    }

    /// The chunks this volume's payload can serve (resident or not).
    #[inline]
    pub fn catalog(&self) -> &ChunkCatalog {
        &self.catalog
    }

    /// The currently published want set (ascending).
    #[inline]
    pub fn wants(&self) -> &BTreeSet<ChunkKey> {
        &self.wants
    }

    /// Wanted chunks that are not resident yet — the convergence backlog.
    pub fn pending_loads(&self) -> usize {
        self.wants
            .iter()
            .filter(|k| !self.data.is_resident(**k))
            .count()
    }
}

/// What one [`VoxelVolumes::sync_camera`] pass did, summed over every volume.
///
/// Counts rather than key lists: a sync can touch thousands of chunks across
/// several volumes, and the per-key detail already exists on
/// [`ResidencyReport`](crate::ResidencyReport) and
/// [`MeshSyncReport`](crate::MeshSyncReport) for a caller that wants it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VoxelStreamReport {
    /// Chunks brought in from a payload.
    pub loaded: usize,
    /// Chunks dropped because they left the want set.
    pub evicted: usize,
    /// Wanted chunks no payload could serve (a coverage hole, not an error).
    pub missing: usize,
    /// Chunks whose blob failed to decode, with the reason. The chunk is left
    /// non-resident — a corrupt page never becomes silent geometry — and the
    /// message rides out to the caller rather than to a `tracing` line here,
    /// because Ring 0 owns the *rule* and the two hosts own their logs. (It is
    /// also what keeps this crate's dependency list at six.)
    pub failed: Vec<(ChunkKey, String)>,
    /// Unwanted chunks **kept** because they carry unsaved edits (P21.3's carve
    /// write-back releases them).
    pub retained_dirty: usize,
    /// Chunk meshes rebuilt because something in their neighbourhood changed.
    pub remeshed: usize,
    /// Cached meshes dropped because their chunk left the volume.
    pub dropped: usize,
    /// Resident chunks across every volume after the pass.
    pub resident: usize,
    /// Whether any volume's want set had to be trimmed to the chunk budget.
    pub budget_clamped: bool,
}

impl VoxelStreamReport {
    /// `true` when nothing moved — no page in, no page out, no re-mesh, no
    /// failure. The property a steady camera must hold.
    pub fn is_noop(&self) -> bool {
        self.loaded == 0
            && self.evicted == 0
            && self.failed.is_empty()
            && self.remeshed == 0
            && self.dropped == 0
    }

    /// Chunks that changed residency.
    pub fn churn(&self) -> usize {
        self.loaded + self.evicted
    }

    /// A one-line human summary for the Output Log / a `stats` dump.
    pub fn summary(&self) -> String {
        format!(
            "voxel streaming: {} resident, {} loads / {} evictions / {} failed, \
             {} re-meshed / {} dropped{}",
            self.resident,
            self.loaded,
            self.evicted,
            self.failed.len(),
            self.remeshed,
            self.dropped,
            if self.budget_clamped {
                " [budget-clamped]"
            } else {
                ""
            }
        )
    }
}

/// Every voxel volume a host currently has loaded, keyed by the **entity**'s id.
///
/// Keyed by entity rather than by asset because two entities may reference one
/// asset at two different transforms, and each needs its own residency and its own
/// GPU cache slot — the same reasoning that makes `RenderTerrain::id` an entity
/// identity rather than an asset one.
#[derive(Debug, Clone, Default)]
pub struct VoxelVolumes {
    slots: BTreeMap<u128, VolumeSlot>,
}

impl VoxelVolumes {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `entity` is already loaded from `asset` — the cheap check a host
    /// makes **before** touching the filesystem or the pack, so a steady-state
    /// frame reads no bytes at all.
    pub fn is_bound(&self, entity: u128, asset: u128) -> bool {
        self.slots.get(&entity).map(|s| s.asset) == Some(asset)
    }

    /// **Bind** `entity`'s volume to a `.inf_voxel` payload image: parse it, index
    /// its chunk directory, keep it as the cold store — and page **no chunks**.
    /// [`sync_camera`](Self::sync_camera) is what pages them.
    ///
    /// A no-op returning `Ok(false)` when the entity is already bound to this
    /// asset — so a host may call it unconditionally and pay only the
    /// [`is_bound`](Self::is_bound) comparison. Re-binding to a *different* asset
    /// replaces the slot wholesale, which is what makes an asset swap in the
    /// Details panel take effect without a stale chunk surviving.
    ///
    /// `Err` for a payload that does not parse — a corrupt or foreign file is
    /// reported, never silently drawn as an empty cave.
    pub fn ensure(&mut self, entity: u128, asset: u128, bytes: &[u8]) -> Result<bool, String> {
        if self.is_bound(entity, asset) {
            return Ok(false);
        }
        let store = VoxelAssetReader::new(bytes.to_vec()).map_err(|e| e.to_string())?;
        let catalog = ChunkCatalog::from_store(&store);
        // The scale and the anchor are the ASSET's, never the component's — a host
        // that seeded them from `VoxelVolume::voxel_size_m` would scale chunks
        // against origins derived from the asset's and tear the volume apart in
        // every level where the two happen to disagree.
        let data = VoxelData::new(store.voxel_size_m()).with_origin(store.origin());
        // A rebind keeps whatever placement the host last pushed: the transform is
        // a property of the entity, not of the asset it points at.
        let translation = self
            .slots
            .get(&entity)
            .map(|s| s.translation)
            .unwrap_or(DVec3::ZERO);
        let grid = ChunkGrid::new(store.voxel_size_m(), store.origin() + translation);
        self.slots.insert(
            entity,
            VolumeSlot {
                asset,
                data,
                meshes: VoxelMeshCache::new(),
                translation,
                store,
                catalog,
                grid,
                wants: BTreeSet::new(),
            },
        );
        Ok(true)
    }

    /// Record where the host has placed `entity`'s volume in the world.
    ///
    /// Idempotent and cheap, so a host calls it every pre-pass beside
    /// [`ensure`](Self::ensure) rather than trying to detect a moved entity. It
    /// exists because residency is measured **in world space**: without it a cave
    /// authored around the asset origin but placed a kilometre away would page the
    /// chunks under the *asset's* anchor and leave the ones under the player
    /// missing — a hole in the world that no rendering code could explain.
    ///
    /// Returns whether the placement changed (a mover invalidates the want set;
    /// the next [`sync_camera`](Self::sync_camera) reconciles it).
    ///
    /// # A non-finite placement KEEPS THE LAST GOOD ONE
    ///
    /// A NaN must never reach the grid — every distance off a NaN anchor is NaN, so
    /// every want is undecidable and the volume simply stops paging. What to put
    /// there instead is a choice, and this one used to substitute the world origin:
    /// a single NaN frame from a host (a degenerate parent transform, a
    /// mid-teleport pose) then *teleported the cave to (0, 0, 0)* and paged the
    /// chunks there until the next good `place`.
    ///
    /// Holding the last good translation is strictly better and no bigger: a
    /// transform that went non-finite for a frame is a transform with no new
    /// information in it, and the volume stays where it was last known to be. It
    /// converges better across PIE and shipping too — the two hosts recover to the
    /// same place instead of to whichever origin-teleport each of them saw. And it
    /// is byte-identical for a *first* NaN: a fresh slot's translation is already
    /// `DVec3::ZERO`, which is exactly what the old substitution wrote.
    pub fn place(&mut self, entity: u128, translation: DVec3) -> bool {
        let Some(slot) = self.slots.get_mut(&entity) else {
            return false;
        };
        if !translation.is_finite() {
            return false;
        }
        if slot.translation == translation {
            return false;
        }
        slot.translation = translation;
        slot.grid = ChunkGrid::new(slot.data.voxel_size_m(), slot.data.origin() + translation);
        true
    }

    /// **The camera-driven residency pass.** Reconcile every bound volume against
    /// the want set [`crate::wants`] computes for `camera_world`, then re-mesh
    /// whatever moved.
    ///
    /// Three bounded steps per volume, each of them pure:
    ///
    /// 1. [`chunk_wants`] — the radius sphere, with hysteresis against the
    ///    volume's previously published set.
    /// 2. [`clamp_wants`] — trimmed farthest-first to
    ///    [`VoxelStreamBudget::max_resident_chunks`].
    /// 3. [`advance_wants`] — at most
    ///    [`VoxelStreamBudget::max_loads_per_sync`] not-yet-resident chunks enter
    ///    per pass, nearest first, so a teleporting camera costs many small syncs
    ///    rather than one hitch.
    ///
    /// Then [`VoxelData::sync_residency`] executes it and
    /// [`VoxelMeshCache::sync`] re-meshes exactly the chunks whose 3 × 3 × 3
    /// neighbourhood changed — which is what makes paging cost the boundary of the
    /// window and not the window.
    ///
    /// **Which budget this is held against:** a load, i.e.
    /// `inf_player::budget::LOAD_BUDGET_MS`, never `inf_core::FRAME_BUDGET_MS`.
    /// See [`VoxelStreamBudget`] for the full argument.
    ///
    /// Deterministic: the result is a function of `(camera, params, budget,
    /// payloads, previous state)` alone. Volumes are walked in ascending entity
    /// order and every set is a `BTreeSet`, so two runs of a scripted camera path
    /// produce the identical trace.
    pub fn sync_camera(
        &mut self,
        camera_world: DVec3,
        params: &VoxelWantsParams,
        budget: VoxelStreamBudget,
    ) -> VoxelStreamReport {
        let mut report = VoxelStreamReport::default();
        for slot in self.slots.values_mut() {
            let target = chunk_wants(camera_world, slot.grid, params, &slot.catalog, &slot.wants);
            let clamped = clamp_wants(&target, slot.grid, camera_world, budget.max_resident_chunks);
            report.budget_clamped |= clamped.len() < target.len();
            let next = advance_wants(
                &slot.wants,
                &clamped,
                slot.grid,
                camera_world,
                &|k| slot.data.is_resident(k),
                budget.max_loads_per_sync,
            );
            let residency = slot.data.sync_residency(&next, &slot.store);
            slot.wants = next;
            report.loaded += residency.loaded.len();
            report.evicted += residency.evicted.len();
            report.missing += residency.missing.len();
            report.retained_dirty += residency.retained_dirty.len();
            report.failed.extend(residency.failed);
            // Re-mesh unconditionally rather than only when residency moved: a
            // carve (P21.3) mutates chunks without changing which are resident,
            // and the cache's own source key is what decides whether anything is
            // rebuilt — a no-op sync walks the keys and rebuilds nothing.
            let mesh = slot.meshes.sync(&slot.data);
            report.remeshed += mesh.remeshed.len();
            report.dropped += mesh.dropped.len();
            report.resident += slot.data.chunk_count();
        }
        report
    }

    /// Re-mesh whatever moved in an already-loaded slot — the seam a carve tool
    /// (P21.3) and a gameplay carve node (P21.4) call after mutating
    /// [`VolumeSlot::data`]. A no-op on an untouched volume.
    pub fn resync(&mut self, entity: u128) -> bool {
        let Some(slot) = self.slots.get_mut(&entity) else {
            return false;
        };
        !slot.meshes.sync(&slot.data).is_noop()
    }

    /// A loaded volume's slot.
    pub fn get(&self, entity: u128) -> Option<&VolumeSlot> {
        self.slots.get(&entity)
    }

    /// A loaded volume's slot, mutably (the carve seam).
    pub fn get_mut(&mut self, entity: u128) -> Option<&mut VolumeSlot> {
        self.slots.get_mut(&entity)
    }

    /// Drop one entity's volume.
    pub fn remove(&mut self, entity: u128) -> Option<VolumeSlot> {
        self.slots.remove(&entity)
    }

    /// Drop every volume whose entity is no longer in `live` — the "the document
    /// changed" release, so a deleted entity's chunks and meshes do not outlive it.
    pub fn retain_only(&mut self, live: &BTreeSet<u128>) {
        self.slots.retain(|k, _| live.contains(k));
    }

    /// Drop everything (a level switch).
    pub fn clear(&mut self) {
        self.slots.clear();
    }

    /// Loaded volumes, ascending by entity id.
    pub fn iter(&self) -> impl Iterator<Item = (&u128, &VolumeSlot)> {
        self.slots.iter()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Total meshed triangles across every loaded volume — a budget/stat readout.
    pub fn triangle_count(&self) -> usize {
        self.slots.values().map(|s| s.meshes.triangle_count()).sum()
    }

    /// Total resident chunks across every loaded volume.
    pub fn chunk_count(&self) -> usize {
        self.slots.values().map(|s| s.data.chunk_count()).sum()
    }

    /// Wanted-but-not-yet-resident chunks across every loaded volume — the
    /// convergence backlog a debug readout prints.
    pub fn pending_loads(&self) -> usize {
        self.slots.values().map(|s| s.pending_loads()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::build_voxel_asset;
    use crate::chunk::VoxelChunk;
    use crate::residency::chunk_range;
    use glam::DVec3;

    /// A 2 × 2 × 2-chunk sphere payload at 0.5 m voxels (a 8 m chunk span), anchored
    /// at the world origin.
    fn payload(radius: f64) -> Vec<u8> {
        let mut v = VoxelData::new(0.5);
        for key in chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
            let b = key.base_sample();
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|i, j, k| {
                    DVec3::new(
                        (b[0] + i as i32) as f64,
                        (b[1] + j as i32) as f64,
                        (b[2] + k as i32) as f64,
                    )
                    .distance(DVec3::splat(16.0))
                        - radius
                }),
            );
        }
        build_voxel_asset(&v).unwrap().into_bytes()
    }

    /// A radius wide enough to want every chunk of `payload`.
    fn everything() -> VoxelWantsParams {
        VoxelWantsParams {
            radius_m: 1000.0,
            hysteresis: 0.0,
        }
    }

    #[test]
    fn ensure_binds_without_paging_and_sync_camera_pages() {
        let mut store = VoxelVolumes::new();
        assert!(store.is_empty());
        let bytes = payload(10.0);

        assert!(store.ensure(1, 100, &bytes).unwrap(), "first bind");
        assert!(store.is_bound(1, 100));
        assert_eq!(store.len(), 1);
        let slot = store.get(1).unwrap();
        assert_eq!(slot.asset, 100);
        assert_eq!(slot.catalog().len(), 8, "the directory is indexed at bind");
        assert_eq!(
            slot.data.chunk_count(),
            0,
            "binding must page NOTHING — that is the P21.2 change"
        );
        assert_eq!(slot.meshes.triangle_count(), 0);

        // A second bind with the same asset does nothing at all.
        assert!(!store.ensure(1, 100, &bytes).unwrap(), "already bound");
        assert!(!store.is_bound(1, 999));

        // …and the camera pass is what makes a cave exist.
        let r = store.sync_camera(
            DVec3::splat(8.0),
            &everything(),
            VoxelStreamBudget::default(),
        );
        assert_eq!(r.loaded, 8);
        assert_eq!(r.resident, 8);
        assert!(!r.is_noop());
        assert!(store.triangle_count() > 0);
        assert_eq!(store.chunk_count(), 8);
        assert_eq!(store.pending_loads(), 0);
    }

    /// **A no-op sync stays a no-op**: a steady camera moves no chunk and, more to
    /// the point, **no mesh stamp**. A stamp that moved would re-upload every
    /// cave's vertex buffer every frame while every pixel stayed identical, which
    /// nothing on screen would show.
    #[test]
    fn a_no_op_sync_moves_no_mesh_stamp() {
        let mut store = VoxelVolumes::new();
        store.ensure(1, 100, &payload(10.0)).unwrap();
        let cam = DVec3::splat(8.0);
        store.sync_camera(cam, &everything(), VoxelStreamBudget::default());

        let keys: Vec<ChunkKey> = store.get(1).unwrap().meshes.cached_keys().collect();
        assert!(!keys.is_empty());
        let before: Vec<u64> = keys
            .iter()
            .map(|k| store.get(1).unwrap().meshes.version(*k))
            .collect();
        assert!(
            store.get(1).unwrap().meshes.triangle_count() > 0,
            "the fixture must produce a real surface, or this proves nothing"
        );

        // Four consecutive steady passes.
        for pass in 0..4 {
            let r = store.sync_camera(cam, &everything(), VoxelStreamBudget::default());
            assert!(r.is_noop(), "pass {pass} was not a no-op: {r:?}");
            let now: Vec<u64> = keys
                .iter()
                .map(|k| store.get(1).unwrap().meshes.version(*k))
                .collect();
            assert_eq!(now, before, "pass {pass} moved a mesh stamp");
        }
    }

    /// **Paging moves mesh stamps only where the resident bitmask says.** The
    /// store-level twin of `mesh::evicting_a_non_maximal_neighbour_remeshes_the_
    /// chunks_that_could_see_it`: a chunk whose 3 × 3 × 3 neighbourhood gained or
    /// lost a page re-meshes; one that could not see the change does not.
    ///
    /// Without this a streaming volume would either keep a wall against samples
    /// that are no longer resident (too few re-meshes) or rebuild the whole cave
    /// on every step of the camera (too many) — and the second is the one that
    /// looks fine and costs everything.
    #[test]
    fn paging_remeshes_only_the_chunks_that_could_see_it() {
        // A 4 × 1 × 1 row of chunks so "in the neighbourhood" and "far away" are
        // easy to name: chunk 0 can see 1, and cannot see 3.
        let mut v = VoxelData::new(0.5);
        for x in 0..4 {
            let key = ChunkKey::new(x, 0, 0);
            let b = key.base_sample();
            // A slab of rock in the lower half of every chunk, so each has a real
            // surface and each neighbour's residency is visible in the seam.
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|_, j, _| (b[1] + j as i32) as f64 - 8.0),
            );
        }
        let bytes = build_voxel_asset(&v).unwrap().into_bytes();

        let mut store = VoxelVolumes::new();
        store.ensure(1, 100, &bytes).unwrap();
        // Camera inside chunk 0, radius wide enough for all four. Deliberately
        // NOT in the middle of the row: a centred camera is equidistant from both
        // ends, so tightening the radius would evict two chunks at once and the
        // "could not see it" half of the assertion would have nothing to stand on.
        let cam = DVec3::new(4.0, 4.0, 4.0);
        store.sync_camera(cam, &everything(), VoxelStreamBudget::default());
        assert_eq!(store.chunk_count(), 4);

        let far = ChunkKey::new(0, 0, 0); // two chunks away from the one evicted
        let near = ChunkKey::new(2, 0, 0); // the evicted chunk's neighbour
        let observer_tris = store
            .get(1)
            .unwrap()
            .meshes
            .get(near)
            .map(|m| m.triangle_count());
        assert!(
            observer_tris.is_some_and(|t| t > 0),
            "the observing chunk must carry a real mesh, or the assertion below is \
             vacuous"
        );
        let far_before = store.get(1).unwrap().meshes.version(far);
        let near_before = store.get(1).unwrap().meshes.version(near);

        // Shrink the radius so exactly chunk 3 leaves the want set. At 0.5 m
        // voxels a chunk spans 8 m, so from x = 4 the boxes sit at distances
        // 0 / 4 / 12 / 20 — and 13 m keeps three and drops the last.
        let tight = VoxelWantsParams {
            radius_m: 13.0,
            hysteresis: 0.0,
        };
        let r = store.sync_camera(cam, &tight, VoxelStreamBudget::default());
        assert_eq!(r.evicted, 1, "exactly one chunk left: {r:?}");
        assert_eq!(store.chunk_count(), 3);

        assert_ne!(
            store.get(1).unwrap().meshes.version(near),
            near_before,
            "the evicted chunk's neighbour must re-mesh — its seam now abuts air"
        );
        assert_eq!(
            store.get(1).unwrap().meshes.version(far),
            far_before,
            "a chunk two away could not see the eviction and must not re-mesh"
        );

        // Paging it back in is the same rule in reverse.
        let near_after_evict = store.get(1).unwrap().meshes.version(near);
        let far_after_evict = store.get(1).unwrap().meshes.version(far);
        let r = store.sync_camera(cam, &everything(), VoxelStreamBudget::default());
        assert_eq!(r.loaded, 1);
        assert_ne!(
            store.get(1).unwrap().meshes.version(near),
            near_after_evict,
            "the returning chunk's neighbour must re-mesh"
        );
        assert_eq!(
            store.get(1).unwrap().meshes.version(far),
            far_after_evict,
            "…and the chunk that could not see it still must not"
        );
    }

    /// The budget really bounds a pass, and the volume still converges.
    #[test]
    fn the_load_budget_bounds_a_pass_and_still_converges() {
        let mut store = VoxelVolumes::new();
        store.ensure(1, 100, &payload(10.0)).unwrap();
        let cam = DVec3::splat(8.0);
        let budget = VoxelStreamBudget {
            max_resident_chunks: 0,
            max_loads_per_sync: 3,
        };
        let r = store.sync_camera(cam, &everything(), budget);
        assert_eq!(r.loaded, 3, "the batch bound was not honoured: {r:?}");
        let mut passes = 1;
        while store.chunk_count() < 8 {
            store.sync_camera(cam, &everything(), budget);
            passes += 1;
            assert!(passes < 20, "did not converge");
        }
        assert!(
            passes > 1,
            "an unbounded pass would defeat the amortization"
        );

        // The residency ceiling trims the want set and says so.
        let mut small = VoxelVolumes::new();
        small.ensure(1, 100, &payload(10.0)).unwrap();
        let r = small.sync_camera(
            cam,
            &everything(),
            VoxelStreamBudget {
                max_resident_chunks: 4,
                max_loads_per_sync: 0,
            },
        );
        assert!(r.budget_clamped);
        assert_eq!(small.chunk_count(), 4);
    }

    /// Placement is world-space: a volume the host moved pages the chunks under
    /// the *player*, not under the asset's anchor.
    #[test]
    fn placement_moves_the_residency_window() {
        let mut store = VoxelVolumes::new();
        store.ensure(1, 100, &payload(10.0)).unwrap();
        let params = VoxelWantsParams {
            radius_m: 4.0,
            hysteresis: 0.0,
        };
        let cam = DVec3::new(1000.0, 8.0, 8.0);
        // Unplaced: the asset sits at the origin, so a camera a kilometre away
        // wants nothing.
        assert!(store
            .sync_camera(cam, &params, VoxelStreamBudget::default())
            .is_noop());
        assert_eq!(store.chunk_count(), 0);

        // Placed under the camera, the same pass pages.
        assert!(store.place(1, DVec3::new(1000.0, 0.0, 0.0)));
        assert!(!store.place(1, DVec3::new(1000.0, 0.0, 0.0)), "idempotent");
        let r = store.sync_camera(cam, &params, VoxelStreamBudget::default());
        assert!(r.loaded > 0, "{r:?}");
        // A non-finite placement KEEPS THE LAST GOOD ONE (P21.2 re-audit). The NaN
        // never reaches the grid — that is the point, since a NaN anchor makes every
        // distance NaN and every want undecidable — and the cave does not teleport
        // to the world origin either, which is what the earlier origin-substitution
        // did on a single bad frame from a host.
        assert!(
            !store.place(1, DVec3::splat(f64::NAN)),
            "a non-finite placement changed nothing, so it reports no change"
        );
        assert_eq!(
            store.get(1).unwrap().translation,
            DVec3::new(1000.0, 0.0, 0.0),
            "one NaN frame moved the volume — the last good placement must stand"
        );
        // …and the pages under it stay where they were: the refusal is a no-op all
        // the way down, not just in the recorded translation.
        let after = store.sync_camera(cam, &params, VoxelStreamBudget::default());
        assert!(
            after.is_noop(),
            "the NaN frame re-paged the volume: {after:?}"
        );
        // A volume that has never been placed is unaffected too — the first NaN is
        // byte-identical to the old behaviour, because a fresh slot is already at
        // the origin.
        let mut fresh = VoxelVolumes::new();
        fresh.ensure(1, 100, &payload(10.0)).unwrap();
        assert!(!fresh.place(1, DVec3::splat(f64::NAN)));
        assert_eq!(fresh.get(1).unwrap().translation, DVec3::ZERO);
        assert!(
            !store.place(999, DVec3::ZERO),
            "an unbound entity is not an error"
        );
    }

    /// Hysteresis survives the round trip through the slot: the published want set
    /// is what the dead band is measured against, so a hovering camera pages
    /// nothing.
    #[test]
    fn a_hovering_camera_pages_nothing() {
        let mut store = VoxelVolumes::new();
        store.ensure(1, 100, &payload(10.0)).unwrap();
        let params = VoxelWantsParams {
            radius_m: 30.0,
            hysteresis: 0.25,
        };
        let budget = VoxelStreamBudget::default();
        // Stand off along −X so the two chunk columns sit at 20 m and 28 m: the
        // near four are inside the take radius (22.5 m) and the far four are not,
        // which is what makes the dead band observable at all.
        let eye = DVec3::new(-20.0, 8.0, 8.0);
        for _ in 0..8 {
            store.sync_camera(eye, &params, budget);
        }
        let settled = store.chunk_count();
        assert_eq!(settled, 4, "the fixture must straddle the radius");
        // Jitter inside the dead band.
        for k in 0..12 {
            let d = if k % 2 == 0 { 1.0 } else { -1.0 };
            let r = store.sync_camera(eye + DVec3::new(d, 0.0, 0.0), &params, budget);
            assert!(r.is_noop(), "the hover paged: {r:?}");
        }
        assert_eq!(store.chunk_count(), settled);
    }

    /// Re-binding an entity to a **different** asset replaces the slot — the
    /// Details-panel asset swap, which must not leave a stale chunk behind.
    #[test]
    fn rebinding_to_another_asset_replaces_the_slot() {
        let mut store = VoxelVolumes::new();
        let cam = DVec3::splat(8.0);
        store.ensure(1, 100, &payload(10.0)).unwrap();
        store.sync_camera(cam, &everything(), VoxelStreamBudget::default());
        let big = store.get(1).unwrap().meshes.triangle_count();

        assert!(store.ensure(1, 200, &payload(4.0)).unwrap());
        assert_eq!(store.get(1).unwrap().asset, 200);
        assert!(store.is_bound(1, 200) && !store.is_bound(1, 100));
        assert_eq!(
            store.get(1).unwrap().data.chunk_count(),
            0,
            "a rebind is a fresh bind: nothing of the old residency survives"
        );
        store.sync_camera(cam, &everything(), VoxelStreamBudget::default());
        assert_ne!(store.get(1).unwrap().meshes.triangle_count(), big);
        assert_eq!(store.len(), 1, "the old slot must not survive beside it");
    }

    /// Two entities on the **same** asset get independent slots — each needs its
    /// own residency and its own GPU cache identity.
    #[test]
    fn two_entities_on_one_asset_get_independent_slots() {
        let mut store = VoxelVolumes::new();
        let bytes = payload(10.0);
        store.ensure(1, 100, &bytes).unwrap();
        store.ensure(2, 100, &bytes).unwrap();
        store.sync_camera(
            DVec3::splat(8.0),
            &everything(),
            VoxelStreamBudget::default(),
        );
        assert_eq!(store.len(), 2);
        assert!(store.is_bound(1, 100) && store.is_bound(2, 100));
        // …and their stamps are disjoint, so a key-addressed GPU cache cannot
        // serve one entity's chunks as the other's.
        let key = ChunkKey::new(0, 0, 0);
        assert_ne!(
            store.get(1).unwrap().data.chunk_version(key),
            store.get(2).unwrap().data.chunk_version(key)
        );
        assert_eq!(store.iter().count(), 2);
    }

    #[test]
    fn a_corrupt_payload_is_reported_never_drawn_as_an_empty_cave() {
        let mut store = VoxelVolumes::new();
        let err = store
            .ensure(1, 100, b"not a voxel asset at all")
            .unwrap_err();
        assert!(err.contains("magic") || err.contains("shorter"), "{err}");
        assert!(store.get(1).is_none());
        assert!(store.is_empty());
    }

    #[test]
    fn a_carve_resyncs_only_the_meshes_it_reached() {
        let mut store = VoxelVolumes::new();
        store.ensure(1, 100, &payload(10.0)).unwrap();
        store.sync_camera(
            DVec3::splat(8.0),
            &everything(),
            VoxelStreamBudget::default(),
        );
        let before = store.get(1).unwrap().meshes.triangle_count();

        // Nothing moved ⇒ nothing to re-mesh.
        assert!(!store.resync(1), "an untouched volume must re-mesh nothing");
        assert!(!store.resync(999), "an unloaded entity is not an error");

        let (report, _) = store
            .get_mut(1)
            .unwrap()
            .data
            .apply_op(&crate::VoxelOp::carve(crate::VoxelShape::Sphere {
                center: DVec3::splat(8.0),
                radius_m: 2.0,
            }));
        assert!(report.total_carved() > 0);
        assert!(store.resync(1), "a carve must re-mesh something");
        assert_ne!(store.get(1).unwrap().meshes.triangle_count(), before);
    }

    #[test]
    fn retain_only_and_clear_release_dead_volumes() {
        let mut store = VoxelVolumes::new();
        let bytes = payload(10.0);
        store.ensure(1, 100, &bytes).unwrap();
        store.ensure(2, 100, &bytes).unwrap();
        store.ensure(3, 100, &bytes).unwrap();

        store.retain_only(&BTreeSet::from([1, 3]));
        assert_eq!(store.len(), 2);
        assert!(store.get(2).is_none());
        assert!(store.remove(3).is_some());
        assert!(store.remove(3).is_none());

        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.triangle_count(), 0);
        assert_eq!(store.chunk_count(), 0);
    }

    /// The report's summary is a readout, not a claim — but an empty one would
    /// make the debug log useless, so it is pinned.
    #[test]
    fn the_report_summarizes_and_recognizes_a_no_op() {
        let r = VoxelStreamReport::default();
        assert!(r.is_noop() && r.churn() == 0);
        let r = VoxelStreamReport {
            loaded: 3,
            evicted: 1,
            remeshed: 4,
            resident: 12,
            budget_clamped: true,
            ..Default::default()
        };
        assert!(!r.is_noop());
        assert_eq!(r.churn(), 4);
        let s = r.summary();
        assert!(
            s.contains("12 resident") && s.contains("budget-clamped"),
            "{s}"
        );
    }
}
