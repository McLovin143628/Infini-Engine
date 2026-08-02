//! Runtime virtualized-geometry pick logic (P13.1b), **lazily indexed** (P18.2).
//!
//! A cooked pack carries **both** a mesh's authoring `.inf_mesh` and its
//! cook-derived `.inf_vmesh` (a meshlet LOD DAG), wired by a deterministic id:
//! the vmesh GUID is a fixed bijection of the mesh GUID ([`derived_vmesh_id`]), so
//! the runtime finds a mesh's virtualized form by **computing** the id — no side
//! index. When virtualized geometry is enabled *and* the derived vmesh is present,
//! the renderer draws it through the GPU meshlet path
//! ([`inf_render::cull_visible`] / the `vgeom` render pass); otherwise it falls
//! back to the classic mesh path (roadmap risk #3: the engine ships without
//! virtualized geometry).
//!
//! # What "lazy" means here, and why it matters
//!
//! Before P18.2 this registry **decoded every `.inf_vmesh` in the pack at load**:
//! a level with a thousand virtualized meshes paid a thousand full bincode
//! decodes and held every vertex of every LOD in RAM before the first frame. It
//! now holds a [`VgeomSource`] per asset — the header and page directory only, a
//! few hundred bytes — over the **mmap'd pack itself**, and the renderer's
//! streamer pages meshlets in and out of GPU pools against a byte budget as the
//! camera moves.
//!
//! The pack path shares one [`Arc<PackReader>`] across every source, so a page
//! fetch is a sub-slice of the mapping (`.inf_vmesh` cooks *uncompressed*
//! precisely so it can be — see `PackWriter::compresses_kind`). The loose-file
//! path (a dev-dir `--level` run, and the editor) reads the payload once and is
//! then identical. A **v1** payload — the bare bincode `VgeomMesh` every pack
//! cooked before P18.2 carries — is decoded once at open and re-laid-out into the
//! paged form, so an old pack keeps running with no second code path downstream.
//!
//! [`VmeshRegistry`] holds the resolved sources, and [`VmeshRegistry::pick`] is
//! the pick rule.
//!
//! ## Scope note (honest)
//!
//! The ECS `MeshRef` component's `asset` binding is what wires a scene mesh entity
//! to its derived vmesh; the editor's in-viewport half of that is P18.3.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use uuid::Uuid;

use inf_asset::{AssetId, AssetKind, PackReader};
use inf_vgeom::VgeomSource;

/// Derive the deterministic `.inf_vmesh` asset id for a mesh id. XOR with a
/// constant is a bijection, so distinct mesh ids always yield distinct vmesh ids.
///
/// **P18.3**: the salt used to be hand-copied here (so the shipped player would
/// not depend on the cook pipeline) with a drift test holding the two together.
/// It now lives in Ring 0 — [`inf_vgeom::VMESH_ID_SALT`], the crate that owns the
/// `.inf_vmesh` format — which the player already depends on, so the copy is gone
/// and the cook, the player and the editor read one constant.
pub fn derived_vmesh_id(mesh_id: Uuid) -> Uuid {
    inf_vgeom::derived_vmesh_id(AssetId(mesh_id)).uuid()
}

/// The indexed `.inf_vmesh` meshlet DAGs available to the player, keyed by vmesh
/// asset GUID. Loaded from a cooked pack ([`from_pack`](Self::from_pack)) or a
/// dev-dir ([`from_dir`](Self::from_dir)).
///
/// Holds **indexes, not geometry** — see the module docs.
#[derive(Default)]
pub struct VmeshRegistry {
    meshes: HashMap<Uuid, Arc<VgeomSource>>,
}

impl VmeshRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }

    /// Whether the vmesh with this asset GUID is indexed.
    pub fn contains(&self, vmesh_id: Uuid) -> bool {
        self.meshes.contains_key(&vmesh_id)
    }

    /// Register a source under its asset GUID (used by loaders + tests).
    pub fn insert(&mut self, vmesh_id: Uuid, source: Arc<VgeomSource>) {
        self.meshes.insert(vmesh_id, source);
    }

    /// Index an in-memory [`VgeomMesh`](inf_vgeom::VgeomMesh) — the door for tests
    /// and for a host that built the DAG rather than loading one.
    pub fn insert_mesh(
        &mut self,
        vmesh_id: Uuid,
        mesh: &inf_vgeom::VgeomMesh,
    ) -> Result<(), String> {
        self.meshes
            .insert(vmesh_id, Arc::new(VgeomSource::from_mesh(mesh)?));
        Ok(())
    }

    /// Index every `.inf_vmesh` ([`AssetKind::MeshletMesh`]) entry of a cooked
    /// pack, keyed by its asset GUID (the derived id the cook wrote).
    ///
    /// Parses each entry's header + page directory and nothing else; the payload
    /// stays in the mapping and is sliced page by page as the camera asks for
    /// detail. The `Arc<PackReader>` is shared by every source, so the mapping is
    /// opened once however many vmeshes the pack holds.
    ///
    /// An entry that fails to index is **skipped with a warning** rather than
    /// failing the load: one bad asset must not take a level down, and the entity
    /// referencing it falls back to the placeholder path exactly as an un-cooked
    /// mesh does.
    pub fn from_pack(reader: Arc<PackReader>) -> Result<Self, String> {
        let mut out = Self::new();
        let guids: Vec<AssetId> = reader
            .index()
            .filter(|e| e.kind == AssetKind::MeshletMesh)
            .map(|e| e.guid)
            .collect();
        for guid in guids {
            match VgeomSource::open_pack(reader.clone(), guid) {
                Ok(src) => {
                    out.meshes.insert(guid.uuid(), Arc::new(src));
                }
                Err(e) => tracing::warn!("inf-player: skipping vmesh {guid}: {e}"),
            }
        }
        Ok(out)
    }

    /// Index every `.inf_vmesh` in `dir` (non-recursive) keyed by its sidecar asset
    /// GUID — the dev-dir twin of [`from_pack`](Self::from_pack). Files without a
    /// readable sidecar/GUID or an indexable payload are skipped. Deterministic
    /// (path-sorted) iteration.
    pub fn from_dir(dir: &Path) -> Self {
        let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
            Ok(rd) => rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("inf_vmesh"))
                .collect(),
            Err(_) => return Self::new(),
        };
        files.sort();
        let mut out = Self::new();
        for p in files {
            let Ok(side) = inf_asset::AssetSidecar::load(&p) else {
                continue;
            };
            match std::fs::read(&p)
                .map_err(|e| e.to_string())
                .and_then(VgeomSource::from_payload)
            {
                Ok(src) => {
                    out.meshes.insert(side.guid.uuid(), Arc::new(src));
                }
                Err(e) => tracing::warn!("inf-player: bad .inf_vmesh {}: {e}", p.display()),
            }
        }
        out
    }

    /// The **pick rule**: when `enabled` and the vmesh derived from `mesh_id` is
    /// present, return `(vmesh asset id as u128, the paged source)` for the
    /// renderer's GPU meshlet path; otherwise `None` (⇒ the classic mesh path).
    pub fn pick(&self, mesh_id: Uuid, enabled: bool) -> Option<(u128, Arc<VgeomSource>)> {
        if !enabled {
            return None;
        }
        self.resolve(mesh_id)
    }

    /// Resolve `mesh_id` to its cook-derived `(vmesh asset id as u128, source)`
    /// **regardless of the render setting** (P13.4). The renderer's *tier* decides
    /// which path draws the resolved vgeom content — the GPU meshlet path (High) or
    /// the classic discrete-LOD fallback (Medium/Low) — so the scene content is the
    /// same either way and this resolver is enabled-agnostic. `None` when the mesh
    /// has no derived vmesh (an un-cooked / non-dense mesh ⇒ the placeholder path).
    pub fn resolve(&self, mesh_id: Uuid) -> Option<(u128, Arc<VgeomSource>)> {
        let vmesh_id = derived_vmesh_id(mesh_id);
        self.meshes
            .get(&vmesh_id)
            .map(|m| (vmesh_id.as_u128(), m.clone()))
    }

    /// Bytes every indexed vmesh would occupy in the meshlet pools if fully
    /// resident — the ceiling a streaming budget is measured against.
    pub fn total_resident_bytes(&self) -> u64 {
        self.meshes.values().map(|s| s.total_resident_bytes()).sum()
    }
}

/// Every `.inf_vmesh` in a cooked pack, keyed by asset GUID — the pack twin of
/// [`crate::level::PackLevelSource`]'s asset loaders.
pub fn load_vmeshes_from_pack(reader: Arc<PackReader>) -> Result<VmeshRegistry, String> {
    VmeshRegistry::from_pack(reader)
}

/// True if a pack contains the vmesh derived from `mesh_id` (the id-only presence
/// check the renderer uses before indexing — no decode).
pub fn pack_has_vmesh(reader: &PackReader, mesh_id: Uuid) -> bool {
    reader.contains(AssetId(derived_vmesh_id(mesh_id)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_vgeom::VgeomMesh;

    #[test]
    fn derived_id_is_a_bijection() {
        let a = Uuid::from_u128(0x11);
        let b = Uuid::from_u128(0x22);
        assert_ne!(derived_vmesh_id(a), derived_vmesh_id(b));
        // Involutive salt: applying twice returns the original (XOR bijection).
        assert_eq!(
            derived_vmesh_id(derived_vmesh_id(a)),
            a,
            "XOR salt is its own inverse"
        );
    }

    #[test]
    fn pick_respects_enabled_and_presence() {
        let mesh_id = Uuid::from_u128(0xDEAD);
        let vmesh_id = derived_vmesh_id(mesh_id);
        let mut reg = VmeshRegistry::new();
        // Absent → None even when enabled.
        assert!(reg.pick(mesh_id, true).is_none());
        // A trivial empty payload just to key the registry.
        reg.insert_mesh(vmesh_id, &empty_vmesh()).expect("index");
        // Disabled → classic path.
        assert!(reg.pick(mesh_id, false).is_none());
        // Enabled + present → picked, with the derived id.
        let (id, _) = reg.pick(mesh_id, true).expect("vmesh picked");
        assert_eq!(id, vmesh_id.as_u128());
        // A different mesh id (whose derived id is not registered) misses.
        assert!(reg.pick(Uuid::from_u128(0xBEEF), true).is_none());
    }

    fn empty_vmesh() -> VgeomMesh {
        VgeomMesh {
            schema_version: VgeomMesh::CURRENT_VERSION,
            vertices: Vec::new(),
            meshlets: Vec::new(),
            meshlet_vertices: Vec::new(),
            meshlet_triangles: Vec::new(),
            groups: Vec::new(),
            levels: Vec::new(),
            center: [0.0; 3],
            radius: 0.0,
        }
    }
}
