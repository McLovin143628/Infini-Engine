//! **The interactive viewport's voxel store** (P21.1): the loose-file half of
//! what `inf_player::voxel::VoxelRegistry` is for a cooked pack.
//!
//! # Why it lives in Ring 1 and not in the viewport host
//!
//! Verbatim the reasoning that put [`crate::terrain_stream`] and
//! [`crate::render_assets`] here: `inf_viewport::host` is
//! `#[cfg(any(windows, target_os = "macos"))]`, so anything written there is
//! invisible to the Linux CI leg — the leg most likely to be the one a
//! contributor's PR runs first. Keeping the index, the resolution rule and the
//! failure policy here (platform-neutral, GPU-free) means the tests below run on
//! all three OSes and the host is left with nothing but call sites.
//!
//! # What is here, and what deliberately is not
//!
//! Here: finding the `.inf_voxel` a `VoxelVolume.asset` names, reading it, and
//! handing the bytes to the Ring-0 [`inf_voxel::VoxelVolumes`] store — which owns
//! everything after the bytes (parsing, residency, meshing, invalidation), so the
//! editor and the shipped player cannot mesh the same field differently.
//!
//! Not here: **any camera policy**. The Ring-0 store pages a volume whole; the
//! view-dependent selection that would page it in parts is P21.2, and the
//! machinery for it is already built and tested in `inf_voxel::residency`.
//!
//! # Resolution is by sidecar GUID, and a miss rescans once
//!
//! A `.inf_voxel` carries its GUID in its `inf_asset` sidecar, exactly as a
//! `.inf_terrain` does, so the index is a walk of the content root. A miss
//! triggers **one** rescan before giving up — an asset written after the project
//! opened (a fresh carve saved by a tool) must resolve without a reopen, which is
//! the same gap `terrain_stream` closed at P16.4a.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use inf_ecs::components::VoxelVolume;
use inf_voxel::{VolumeSlot, VoxelVolumes};
use uuid::Uuid;

/// Depth cap on the content-root walk — deep enough for any content layout,
/// shallow enough that a symlink loop cannot hang project open. Mirrors
/// `terrain_stream`'s cap for the same reason.
const MAX_CONTENT_DEPTH: u32 = 16;

/// Every loose `.inf_voxel` under `dir`, keyed by the GUID in its sidecar.
///
/// A payload without a sidecar is skipped with a warning rather than guessed at:
/// the file name is not an identity, and inventing one would make a level's
/// `VoxelVolume.asset` resolve to whatever happened to be lying next to it.
pub fn voxel_paths_by_guid(dir: &Path) -> HashMap<Uuid, PathBuf> {
    let mut out = HashMap::new();
    let mut files = Vec::new();
    collect_voxel_files(dir, 0, &mut files);
    for p in files {
        match inf_asset::AssetSidecar::load(&p) {
            Ok(side) => {
                out.insert(side.guid.uuid(), p);
            }
            Err(_) => tracing::warn!(
                "inf-editor-core: .inf_voxel without a sidecar {}",
                p.display()
            ),
        }
    }
    out
}

fn collect_voxel_files(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    if depth > MAX_CONTENT_DEPTH {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    // Sorted, so the index is a deterministic function of the tree rather than of
    // the filesystem's enumeration order.
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            let hidden = path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.starts_with('.'));
            if !hidden {
                collect_voxel_files(&path, depth + 1, out);
            }
        } else if path.extension().and_then(|s| s.to_str()) == Some("inf_voxel") {
            out.push(path);
        }
    }
}

/// The editor's loaded voxel volumes, indexed over a project's content root.
#[derive(Default)]
pub struct EditorVoxelVolumes {
    /// Where loose `.inf_voxel` assets are looked up. `None` (the default)
    /// disables the whole store — a document with no volumes is unaffected either
    /// way, which is what keeps a viewport with no project byte-identical to its
    /// pre-P21.1 self.
    content_root: Option<PathBuf>,
    index: HashMap<Uuid, PathBuf>,
    volumes: VoxelVolumes,
    /// Assets that failed to resolve or load. Kept so the warning is logged
    /// **once** rather than on every frame of a projection that runs at 60 Hz —
    /// the difference between a diagnosable log and an unusable one.
    failed: HashSet<Uuid>,
}

impl EditorVoxelVolumes {
    pub fn new() -> Self {
        Self::default()
    }

    /// Point the store at a project's content root (or `None` to disable it).
    ///
    /// Drops every loaded volume: a different root is a different project, and a
    /// GUID that resolved under the old one says nothing about the new one.
    pub fn set_content_root(&mut self, root: Option<PathBuf>) {
        self.volumes.clear();
        self.failed.clear();
        self.index = match &root {
            Some(dir) => voxel_paths_by_guid(dir),
            None => HashMap::new(),
        };
        self.content_root = root;
    }

    /// Rebuild the index after the content database changed, **keeping** loaded
    /// volumes (their bytes are already in memory; a rewritten asset re-binds
    /// through [`ensure`](Self::ensure) when its GUID changes, and a rewrite under
    /// the same GUID is a P21.3 concern the carve path will drive explicitly).
    pub fn refresh_index(&mut self) {
        let Some(dir) = self.content_root.clone() else {
            return;
        };
        self.index = voxel_paths_by_guid(&dir);
        self.failed.clear();
    }

    pub fn content_root(&self) -> Option<&Path> {
        self.content_root.as_deref()
    }

    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    /// Make sure `entity`'s volume is loaded, and report whether anything is
    /// drawable afterwards.
    ///
    /// A volume with no `asset` releases whatever it had loaded and reports
    /// `false` — an author who cleared the reference must see the cave disappear,
    /// not keep the last one that resolved.
    pub fn ensure(&mut self, entity: Uuid, volume: &VoxelVolume) -> bool {
        let Some(asset) = volume.asset else {
            self.volumes.remove(entity.as_u128());
            return false;
        };
        if self.volumes.is_bound(entity.as_u128(), asset.as_u128()) {
            return true;
        }
        if self.content_root.is_none() || self.failed.contains(&asset) {
            return false;
        }
        // A miss rescans ONCE: an asset written after the project opened must
        // resolve without a reopen. `failed` then suppresses the retry storm.
        if !self.index.contains_key(&asset) {
            self.refresh_index();
        }
        let Some(path) = self.index.get(&asset).cloned() else {
            tracing::warn!("inf-editor-core: no .inf_voxel for asset {asset}");
            self.failed.insert(asset);
            return false;
        };
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("inf-editor-core: read {}: {e}", path.display());
                self.failed.insert(asset);
                return false;
            }
        };
        match self
            .volumes
            .ensure(entity.as_u128(), asset.as_u128(), &bytes)
        {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!("inf-editor-core: bad .inf_voxel {}: {e}", path.display());
                self.failed.insert(asset);
                false
            }
        }
    }

    /// The loaded slot for `entity` — its resident chunks and meshed surface.
    pub fn slot(&self, entity: Uuid) -> Option<&VolumeSlot> {
        self.volumes.get(entity.as_u128())
    }

    /// Release every volume whose entity is no longer in `live` (the document
    /// changed), so a deleted entity's chunks and meshes do not outlive it.
    pub fn retain_only(&mut self, live: impl IntoIterator<Item = Uuid>) {
        let keep: BTreeSet<u128> = live.into_iter().map(|g| g.as_u128()).collect();
        self.volumes.retain_only(&keep);
    }

    /// Drop everything (a document close / level switch).
    pub fn clear(&mut self) {
        self.volumes.clear();
        self.failed.clear();
    }

    pub fn len(&self) -> usize {
        self.volumes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.volumes.is_empty()
    }

    /// Meshed triangles across every loaded volume (a status readout).
    pub fn triangle_count(&self) -> usize {
        self.volumes.triangle_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use inf_voxel::{ChunkKey, VoxelChunk, VoxelData};

    /// A carved-out `.inf_voxel` written into `dir` with a sidecar, so the index
    /// can find it exactly as the editor's own writers would leave it.
    fn write_asset(dir: &Path, name: &str, guid: Uuid, radius: f64) {
        let mut v = VoxelData::new(0.5);
        for key in inf_voxel::chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
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
        let asset = inf_voxel::build_voxel_asset(&v).unwrap();
        std::fs::create_dir_all(dir).unwrap();
        let path = dir.join(name);
        let bytes = inf_voxel::write_voxel_asset(&path, &asset).unwrap();
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(guid),
            inf_asset::AssetKind::VoxelVolume,
            inf_asset::ContentHash::of(bytes),
        )
        .save(&path)
        .unwrap();
    }

    #[test]
    fn without_a_content_root_nothing_loads() {
        let mut s = EditorVoxelVolumes::new();
        assert!(s.content_root().is_none());
        assert!(!s.ensure(
            Uuid::from_u128(1),
            &VoxelVolume::from_asset(Uuid::from_u128(9))
        ));
        assert!(s.is_empty());
    }

    #[test]
    fn a_volume_with_no_asset_releases_what_it_had() {
        let dir = tempfile::tempdir().unwrap();
        let asset = Uuid::from_u128(0xA1);
        write_asset(dir.path(), "Cave.inf_voxel", asset, 10.0);
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));

        let e = Uuid::from_u128(1);
        assert!(s.ensure(e, &VoxelVolume::from_asset(asset)));
        assert_eq!(s.len(), 1);
        assert!(s.slot(e).is_some());
        assert!(s.triangle_count() > 0);

        // Clearing the reference must make the cave disappear, not keep the last
        // one that resolved.
        assert!(!s.ensure(e, &VoxelVolume::default()));
        assert!(s.slot(e).is_none());
        assert!(s.is_empty());
    }

    #[test]
    fn a_volume_loads_from_the_content_root_and_stays_loaded() {
        let dir = tempfile::tempdir().unwrap();
        let asset = Uuid::from_u128(0xB2);
        write_asset(&dir.path().join("Caves"), "Deep.inf_voxel", asset, 12.0);
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        assert_eq!(s.index_len(), 1, "the walk must reach subfolders");

        let e = Uuid::from_u128(7);
        let volume = VoxelVolume::from_asset(asset);
        assert!(s.ensure(e, &volume));
        let tris = s.slot(e).unwrap().meshes.triangle_count();
        assert!(tris > 0);
        // A second ensure is a no-op — no filesystem access, same slot.
        assert!(s.ensure(e, &volume));
        assert_eq!(s.slot(e).unwrap().meshes.triangle_count(), tris);
        assert_eq!(s.len(), 1);
    }

    /// An asset written **after** the project opened still resolves: the miss
    /// rescans once. This is the P16.4a gap, closed here at birth.
    #[test]
    fn an_asset_written_after_the_index_still_loads() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        assert_eq!(s.index_len(), 0);

        let asset = Uuid::from_u128(0xC3);
        write_asset(dir.path(), "Late.inf_voxel", asset, 10.0);
        // No reopen, no `set_content_root`: the miss rescans and finds it.
        assert!(s.ensure(Uuid::from_u128(1), &VoxelVolume::from_asset(asset)));
        assert_eq!(s.index_len(), 1);
    }

    /// An unresolvable asset is a warning and a `false`, **once** — a projection
    /// runs every frame, and a store that re-walked the content root each time
    /// would make the editor unusable and the log unreadable.
    #[test]
    fn an_unresolvable_asset_fails_quietly_after_the_first_try() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        let volume = VoxelVolume::from_asset(Uuid::from_u128(0xDEAD));
        for _ in 0..5 {
            assert!(!s.ensure(Uuid::from_u128(1), &volume));
        }
        assert!(s.is_empty());
        // …and a refresh clears the suppression, so a later import is picked up.
        write_asset(dir.path(), "Found.inf_voxel", Uuid::from_u128(0xDEAD), 10.0);
        s.refresh_index();
        assert!(s.ensure(Uuid::from_u128(1), &volume));
    }

    #[test]
    fn changing_the_content_root_drops_every_volume() {
        let dir = tempfile::tempdir().unwrap();
        let asset = Uuid::from_u128(0xE4);
        write_asset(dir.path(), "Cave.inf_voxel", asset, 10.0);
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        assert!(s.ensure(Uuid::from_u128(1), &VoxelVolume::from_asset(asset)));
        assert_eq!(s.len(), 1);

        let other = tempfile::tempdir().unwrap();
        s.set_content_root(Some(other.path().to_path_buf()));
        assert!(s.is_empty(), "a different root is a different project");
        assert_eq!(s.index_len(), 0);
    }

    #[test]
    fn retain_only_releases_dead_volumes() {
        let dir = tempfile::tempdir().unwrap();
        let asset = Uuid::from_u128(0xF5);
        write_asset(dir.path(), "Cave.inf_voxel", asset, 10.0);
        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        let volume = VoxelVolume::from_asset(asset);
        let (a, b) = (Uuid::from_u128(1), Uuid::from_u128(2));
        assert!(s.ensure(a, &volume) && s.ensure(b, &volume));
        assert_eq!(s.len(), 2);

        s.retain_only([a]);
        assert_eq!(s.len(), 1);
        assert!(s.slot(a).is_some() && s.slot(b).is_none());
        s.clear();
        assert!(s.is_empty());
    }

    /// A corrupt payload is reported and skipped — never drawn as an empty cave.
    #[test]
    fn a_corrupt_payload_is_reported_and_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let asset = Uuid::from_u128(0x0BAD);
        let path = dir.path().join("Broken.inf_voxel");
        std::fs::write(&path, b"this is not a voxel asset at all, not even close").unwrap();
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(asset),
            inf_asset::AssetKind::VoxelVolume,
            inf_asset::ContentHash::of(b"x"),
        )
        .save(&path)
        .unwrap();

        let mut s = EditorVoxelVolumes::new();
        s.set_content_root(Some(dir.path().to_path_buf()));
        assert_eq!(s.index_len(), 1, "it IS indexed — it just does not parse");
        assert!(!s.ensure(Uuid::from_u128(1), &VoxelVolume::from_asset(asset)));
        assert!(s.is_empty());
    }
}
