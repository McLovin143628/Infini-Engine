//! Runtime virtualized-geometry pick logic (P13.1b).
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
//! [`VmeshRegistry`] holds the resolved `.inf_vmesh` payloads (from a cooked pack
//! or a dev-dir), and [`VmeshRegistry::pick`] is the pick rule.
//!
//! ## Scope note (honest)
//!
//! The ECS `MeshRef` component is currently a primitive selector (it carries no
//! mesh-**asset** GUID — see `inf_ecs::components::MeshRef`), so wiring a scene
//! mesh entity to its derived vmesh needs the mesh-asset-in-viewport binding
//! that lands with the Phase 4→7 follow-up. This module delivers the resolver +
//! pick rule + registry (the runtime half); the render host consults it once that
//! binding exists.

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use uuid::Uuid;

use inf_asset::{AssetId, AssetKind, PackReader};
use inf_vgeom::VgeomMesh;

/// The fixed salt XORed into a mesh GUID to derive its `.inf_vmesh` GUID.
///
/// **Kept in sync with `inf_packager::cook::VMESH_ID_SALT`** (duplicated here so
/// the shipped player does not depend on the cook pipeline — the same pattern as
/// [`crate::level::PACK_FILE`]). A drift test asserts the two agree.
const VMESH_ID_SALT: u128 = 0x7635_4e56_4d45_5348_1f13_1a2b_3c4d_5e6f;

/// Derive the deterministic `.inf_vmesh` asset id for a mesh id. XOR with a
/// constant is a bijection, so distinct mesh ids always yield distinct vmesh ids;
/// mirrors `inf_packager::derived_vmesh_id`.
pub fn derived_vmesh_id(mesh_id: Uuid) -> Uuid {
    Uuid::from_u128(mesh_id.as_u128() ^ VMESH_ID_SALT)
}

/// The resolved `.inf_vmesh` meshlet DAGs available to the player, keyed by vmesh
/// asset GUID. Loaded from a cooked pack ([`from_pack`](Self::from_pack)) or a
/// dev-dir ([`from_dir`](Self::from_dir)).
#[derive(Default)]
pub struct VmeshRegistry {
    meshes: HashMap<Uuid, Arc<VgeomMesh>>,
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

    /// Whether the vmesh with this asset GUID is loaded.
    pub fn contains(&self, vmesh_id: Uuid) -> bool {
        self.meshes.contains_key(&vmesh_id)
    }

    /// Register a vmesh under its asset GUID (used by loaders + tests).
    pub fn insert(&mut self, vmesh_id: Uuid, mesh: Arc<VgeomMesh>) {
        self.meshes.insert(vmesh_id, mesh);
    }

    /// Load every `.inf_vmesh` ([`AssetKind::MeshletMesh`]) entry from a cooked
    /// pack, keyed by its asset GUID (the derived id the cook wrote).
    pub fn from_pack(reader: &PackReader) -> Result<Self, String> {
        let mut out = Self::new();
        for e in reader.index() {
            if e.kind != AssetKind::MeshletMesh {
                continue;
            }
            let bytes = reader
                .read(e.guid)
                .map_err(|err| format!("read vmesh {}: {err}", e.guid))?;
            let mesh: VgeomMesh = inf_asset::decode(&bytes)
                .map_err(|err| format!("decode vmesh {}: {err}", e.guid))?;
            out.meshes.insert(e.guid.uuid(), Arc::new(mesh));
        }
        Ok(out)
    }

    /// Read every `.inf_vmesh` in `dir` (non-recursive) keyed by its sidecar asset
    /// GUID — the dev-dir twin of [`from_pack`](Self::from_pack). Files without a
    /// readable sidecar/GUID or a decodable payload are skipped. Deterministic
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
            match std::fs::read(&p).map(|b| inf_asset::decode::<VgeomMesh>(&b)) {
                Ok(Ok(mesh)) => {
                    out.meshes.insert(side.guid.uuid(), Arc::new(mesh));
                }
                _ => tracing::warn!("inf-player: bad .inf_vmesh {}", p.display()),
            }
        }
        out
    }

    /// The **pick rule**: when `enabled` and the vmesh derived from `mesh_id` is
    /// present, return `(vmesh asset id as u128, shared meshlet DAG)` for the
    /// renderer's GPU meshlet path; otherwise `None` (⇒ the classic mesh path).
    pub fn pick(&self, mesh_id: Uuid, enabled: bool) -> Option<(u128, Arc<VgeomMesh>)> {
        if !enabled {
            return None;
        }
        self.resolve(mesh_id)
    }

    /// Resolve `mesh_id` to its cook-derived `(vmesh asset id as u128, meshlet DAG)`
    /// **regardless of the render setting** (P13.4). The renderer's *tier* decides
    /// which path draws the resolved vgeom content — the GPU meshlet path (High) or
    /// the classic discrete-LOD fallback (Medium/Low) — so the scene content is the
    /// same either way and this resolver is enabled-agnostic. `None` when the mesh
    /// has no derived vmesh (an un-cooked / non-dense mesh ⇒ the placeholder path).
    pub fn resolve(&self, mesh_id: Uuid) -> Option<(u128, Arc<VgeomMesh>)> {
        let vmesh_id = derived_vmesh_id(mesh_id);
        self.meshes
            .get(&vmesh_id)
            .map(|m| (vmesh_id.as_u128(), m.clone()))
    }
}

/// Every `.inf_vmesh` in a cooked pack, keyed by asset GUID — the pack twin of
/// [`crate::level::PackLevelSource`]'s asset loaders.
pub fn load_vmeshes_from_pack(reader: &PackReader) -> Result<VmeshRegistry, String> {
    VmeshRegistry::from_pack(reader)
}

/// True if a pack contains the vmesh derived from `mesh_id` (the id-only presence
/// check the renderer uses before uploading — no decode).
pub fn pack_has_vmesh(reader: &PackReader, mesh_id: Uuid) -> bool {
    reader.contains(AssetId(derived_vmesh_id(mesh_id)))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        reg.insert(vmesh_id, Arc::new(empty_vmesh()));
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
