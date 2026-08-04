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
//! # Residency, honestly
//!
//! [`ensure`](VoxelVolumes::ensure) pages **every** chunk of the asset resident.
//! That is deliberate and it is P21.1's stated scope: camera-driven residency is
//! P21.2, and the machinery for it is already built and tested one module over
//! ([`crate::residency`] — `sync_residency`, wants sets, dirty retention). Wiring
//! a *policy* through it needs a camera, a budget and an eviction gate on both
//! hosts, which is the work P21.2 names. Loading everything is the honest v1: a
//! carved cave system is tens of chunks, so it fits, and the moment it does not
//! the fix is to call the function that is already there rather than to write a
//! new one.
//!
//! What this is **not** is a second residency implementation. The slot holds a
//! real [`VoxelData`] with real stamps, and P21.2 replaces one call.

use std::collections::{BTreeMap, BTreeSet};

use crate::asset::VoxelAssetReader;
use crate::data::VoxelData;
use crate::mesh::VoxelMeshCache;

/// One volume's loaded state: its chunks and its meshed surface.
#[derive(Debug, Clone)]
pub struct VolumeSlot {
    /// The `.inf_voxel` asset id these chunks came from (as a `u128`, so Ring 0
    /// needs no `uuid` dependency — both hosts hand over `Uuid::as_u128`).
    pub asset: u128,
    /// The resident chunks.
    pub data: VoxelData,
    /// The meshed surface, re-meshed only where the field moved.
    pub meshes: VoxelMeshCache,
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

    /// Load (or reload) `entity`'s volume from a `.inf_voxel` payload image.
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
        let reader = VoxelAssetReader::new(bytes).map_err(|e| e.to_string())?;
        let data = reader.to_voxel_data().map_err(|e| e.to_string())?;
        let mut meshes = VoxelMeshCache::new();
        meshes.sync(&data);
        self.slots.insert(
            entity,
            VolumeSlot {
                asset,
                data,
                meshes,
            },
        );
        Ok(true)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::build_voxel_asset;
    use crate::chunk::{ChunkKey, VoxelChunk};
    use crate::residency::chunk_range;
    use glam::DVec3;

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

    #[test]
    fn ensure_loads_once_and_meshes_what_it_loaded() {
        let mut store = VoxelVolumes::new();
        assert!(store.is_empty());
        let bytes = payload(10.0);

        assert!(store.ensure(1, 100, &bytes).unwrap(), "first load");
        assert!(store.is_bound(1, 100));
        assert_eq!(store.len(), 1);
        let slot = store.get(1).unwrap();
        assert_eq!(slot.asset, 100);
        assert_eq!(slot.data.chunk_count(), 8);
        assert!(slot.meshes.triangle_count() > 0);
        assert_eq!(store.chunk_count(), 8);
        assert_eq!(store.triangle_count(), slot.meshes.triangle_count());

        // A second call with the same asset does nothing at all.
        assert!(!store.ensure(1, 100, &bytes).unwrap(), "already bound");
        assert!(!store.is_bound(1, 999));
    }

    /// Re-binding an entity to a **different** asset replaces the slot — the
    /// Details-panel asset swap, which must not leave a stale chunk behind.
    #[test]
    fn rebinding_to_another_asset_replaces_the_slot() {
        let mut store = VoxelVolumes::new();
        store.ensure(1, 100, &payload(10.0)).unwrap();
        let small = store.get(1).unwrap().meshes.triangle_count();

        assert!(store.ensure(1, 200, &payload(4.0)).unwrap());
        assert_eq!(store.get(1).unwrap().asset, 200);
        assert!(store.is_bound(1, 200) && !store.is_bound(1, 100));
        assert_ne!(store.get(1).unwrap().meshes.triangle_count(), small);
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

    /// `to_voxel_data` is the whole-volume door; the store uses it deliberately
    /// (see the module docs) and the chunk count proves it really did page
    /// everything rather than a camera-shaped subset.
    #[test]
    fn the_store_pages_the_whole_asset_by_design() {
        let bytes = payload(10.0);
        let declared = VoxelAssetReader::new(bytes.as_slice())
            .unwrap()
            .chunk_count();
        let mut store = VoxelVolumes::new();
        store.ensure(1, 100, &bytes).unwrap();
        assert_eq!(store.get(1).unwrap().data.chunk_count(), declared);
    }
}
