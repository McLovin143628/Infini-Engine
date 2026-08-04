//! **The shipped player's voxel asset source** (P21.1): the pack twin of the
//! editor's `inf_editor_core::voxel_store`.
//!
//! One job — hand over the `.inf_voxel` bytes a `VoxelVolume.asset` names — from
//! whichever world the player was launched with: a cooked pack (`--pack`), a
//! level's dev-directory sidecars (`--level`), or nothing at all (`--demo`).
//! **Everything after the bytes** — parsing, residency, meshing, invalidation —
//! lives in the Ring-0 [`inf_voxel::VoxelVolumes`] store the render host owns, so
//! the shipped build and the editor preview cannot mesh the same field
//! differently.
//!
//! # Bytes are read on demand and never held
//!
//! Unlike [`VmeshRegistry`](crate::vmesh::VmeshRegistry), which keeps an opened
//! `VgeomSource` per asset because a meshlet page is a *sub-slice* of the mapping,
//! this registry keeps only the *source* (a path map or the pack reader). A
//! `.inf_voxel` is decoded into a [`VoxelData`](inf_voxel::VoxelData) exactly once
//! per `(entity, asset)` binding, so caching the payload beside the decoded chunks
//! would be a second full copy of every cave in the level for no reader.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use inf_asset::{AssetKind, PackReader};
use uuid::Uuid;

/// Where a `.inf_voxel` payload comes from.
enum Source {
    /// No voxel content (the `--demo` / primitive-only worlds).
    None,
    /// Loose files under a dev content directory, indexed by sidecar GUID.
    Dir(HashMap<Uuid, PathBuf>),
    /// A cooked pack. Shared as an `Arc` with the other pack-backed stores, so
    /// the mapping is opened once however many kinds of asset the renderer reaches
    /// into it for (the P18.2 rule).
    Pack(Arc<PackReader>),
    /// Bytes that arrived over the PIE wire (P21.4). The editor has the project's
    /// asset DB and the player subprocess has no filesystem handle to it at all,
    /// so a PIE session's voxel source *is* the payload — see
    /// [`ScenePayload::voxels`](inf_runtime::pie::ScenePayload::voxels).
    Memory(HashMap<Uuid, Vec<u8>>),
}

/// Resolves a `VoxelVolume.asset` GUID to its `.inf_voxel` payload bytes.
pub struct VoxelRegistry {
    source: Source,
    /// How many voxel assets the source can serve (a startup log / stat).
    count: usize,
}

impl Default for VoxelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl VoxelRegistry {
    /// An inert registry — every lookup misses, so every `VoxelVolume` draws
    /// nothing. The `--demo` world's, and the pre-P21.1 behaviour of every world.
    pub fn new() -> Self {
        Self {
            source: Source::None,
            count: 0,
        }
    }

    /// Index the loose `.inf_voxel` files under a dev content directory by the
    /// GUID in their `inf_asset` sidecars (the `--level` path).
    ///
    /// A payload without a sidecar is skipped: the file name is not an identity,
    /// and guessing one would make a level's reference resolve to whatever
    /// happened to be lying next to it.
    pub fn from_dir(dir: &Path) -> Self {
        let mut files = Vec::new();
        collect(dir, 0, &mut files);
        let mut map = HashMap::new();
        for p in files {
            match inf_asset::AssetSidecar::load(&p) {
                Ok(side) => {
                    map.insert(side.guid.uuid(), p);
                }
                Err(_) => {
                    tracing::warn!("inf-player: .inf_voxel without a sidecar {}", p.display())
                }
            }
        }
        let count = map.len();
        Self {
            source: Source::Dir(map),
            count,
        }
    }

    /// Serve `.inf_voxel` payloads out of a cooked pack (the `--pack` path).
    pub fn from_pack(reader: Arc<PackReader>) -> Self {
        let count = reader
            .index()
            .filter(|e| e.kind == AssetKind::VoxelVolume)
            .count();
        Self {
            source: Source::Pack(reader),
            count,
        }
    }

    /// Serve `.inf_voxel` payloads carried by a PIE [`ScenePayload`] (P21.4).
    ///
    /// The editor resolved these out of the project's asset DB (or out of its own
    /// live carve store, for a volume with unsaved digs) and put them on the wire;
    /// the player has no other route to them, because `--pie` discards every
    /// content path argument. This is what makes the PIE half of the phase-21 gate
    /// a real comparison rather than two empty maps agreeing.
    ///
    /// [`ScenePayload`]: inf_runtime::pie::ScenePayload
    pub fn from_payload(voxels: &[(Uuid, Vec<u8>)]) -> Self {
        let map: HashMap<Uuid, Vec<u8>> = voxels.iter().cloned().collect();
        let count = map.len();
        Self {
            source: Source::Memory(map),
            count,
        }
    }

    /// Read `asset`'s payload bytes, or `None` when this world has no such asset.
    ///
    /// Called once per `(entity, asset)` binding — see the module docs on why
    /// nothing is cached here.
    pub fn load(&self, asset: Uuid) -> Option<Vec<u8>> {
        match &self.source {
            Source::None => None,
            Source::Memory(map) => map.get(&asset).cloned(),
            Source::Dir(map) => {
                let path = map.get(&asset)?;
                match std::fs::read(path) {
                    Ok(b) => Some(b),
                    Err(e) => {
                        tracing::warn!("inf-player: read {}: {e}", path.display());
                        None
                    }
                }
            }
            Source::Pack(reader) => match reader.read(inf_asset::AssetId(asset)) {
                Ok(b) => Some(b),
                Err(e) => {
                    tracing::warn!("inf-player: read .inf_voxel {asset} from pack: {e}");
                    None
                }
            },
        }
    }

    /// How many `.inf_voxel` assets this world can serve.
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Depth cap on the dev-directory walk (mirrors `VmeshRegistry::from_dir`'s).
const MAX_DEPTH: u32 = 16;

fn collect(dir: &Path, depth: u32, out: &mut Vec<PathBuf>) {
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            let hidden = path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.starts_with('.'));
            if !hidden {
                collect(&path, depth + 1, out);
            }
        } else if path.extension().and_then(|s| s.to_str()) == Some("inf_voxel") {
            out.push(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_voxel::{ChunkKey, VoxelChunk, VoxelData};

    fn asset_bytes() -> Vec<u8> {
        let mut v = VoxelData::new(0.5);
        for key in inf_voxel::chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(0, 0, 0)) {
            v.insert_chunk(key, VoxelChunk::solid(1));
        }
        inf_voxel::build_voxel_asset(&v).unwrap().into_bytes()
    }

    #[test]
    fn an_inert_registry_serves_nothing() {
        let r = VoxelRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert!(r.load(Uuid::from_u128(1)).is_none());
    }

    #[test]
    fn a_dev_directory_indexes_by_sidecar_guid() {
        let dir = tempfile::tempdir().unwrap();
        let guid = Uuid::from_u128(0xCAFE);
        let path = dir.path().join("Caves").join("Deep.inf_voxel");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let bytes = asset_bytes();
        std::fs::write(&path, &bytes).unwrap();
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(guid),
            AssetKind::VoxelVolume,
            inf_asset::ContentHash::of(&bytes),
        )
        .save(&path)
        .unwrap();
        // A payload with NO sidecar is skipped rather than guessed at.
        std::fs::write(dir.path().join("Orphan.inf_voxel"), &bytes).unwrap();

        let r = VoxelRegistry::from_dir(dir.path());
        assert_eq!(r.len(), 1, "the walk reaches subfolders and skips orphans");
        assert_eq!(r.load(guid).as_deref(), Some(bytes.as_slice()));
        assert!(r.load(Uuid::from_u128(0xBEEF)).is_none());
    }

    /// The PIE source (P21.4): the payload's bytes come back verbatim, and a GUID
    /// the payload did not carry misses rather than resolving to a neighbour.
    #[test]
    fn a_payload_serves_the_bytes_it_carried() {
        let guid = Uuid::from_u128(0x2104_0001);
        let bytes = asset_bytes();
        let r = VoxelRegistry::from_payload(&[(guid, bytes.clone())]);
        assert_eq!(r.len(), 1);
        assert!(!r.is_empty());
        assert_eq!(r.load(guid).as_deref(), Some(bytes.as_slice()));
        assert!(r.load(Uuid::from_u128(0x2104_0002)).is_none());
        // An empty payload is an INERT registry, so `attach_voxel_volumes`'
        // early-out keeps a voxel-free PIE session on exactly the path it had.
        assert!(VoxelRegistry::from_payload(&[]).is_empty());
    }

    #[test]
    fn a_pack_serves_voxel_payloads_verbatim() {
        let guid = inf_asset::AssetId(Uuid::from_u128(0xF00D));
        let bytes = asset_bytes();
        let mut w = inf_asset::PackWriter::new();
        w.add_bytes(guid, AssetKind::VoxelVolume, &bytes).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("game.inf_pack");
        w.write_to_file(&path).unwrap();

        let reader = Arc::new(PackReader::open(&path).unwrap());
        let r = VoxelRegistry::from_pack(reader);
        assert_eq!(r.len(), 1);
        assert_eq!(
            r.load(guid.uuid()).as_deref(),
            Some(bytes.as_slice()),
            "a streaming-class kind must come back byte-identical (uncompressed)"
        );
        assert!(r.load(Uuid::from_u128(1)).is_none());
    }
}
