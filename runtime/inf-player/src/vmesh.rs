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

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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
    /// Mesh ids already reported as having **no** derived DAG, so the refusal is
    /// stated once per asset per session rather than once per asset per frame
    /// (wave FIX2).
    ///
    /// This is where the placeholder cube used to be. A `MeshRef.asset` that
    /// misses here draws **nothing** on either host — that is the honest frame,
    /// because a 1 m box at the entity's transform is a claim about the world
    /// that no author made — and the reason it is not silent is this set.
    missing: Mutex<HashSet<Uuid>>,
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
    /// referencing it reaches the same stated miss ([`report_missing`](Self::report_missing))
    /// an un-cooked mesh does.
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
            out.open_file(side.guid.uuid(), &p);
        }
        out
    }

    /// Index the `.inf_vmesh` files a **PIE payload** names
    /// (`ScenePayload::vmesh_paths`, v13): `(derived vmesh asset guid, path)`.
    ///
    /// The payload half of [`from_dir`](Self::from_dir), and deliberately the
    /// same act: read the payload once, index its header + page directory, key it
    /// under the guid the producer named. What differs is only *which* files —
    /// `from_dir` takes every `.inf_vmesh` in a directory and reads its guid off
    /// a sidecar, while a payload names exactly the DAGs the level's rigid
    /// `MeshRef.asset`s resolve to and carries the guid itself (the editor's
    /// derived asset has a sidecar, but a player must not have to walk the
    /// author's whole content root to find four files).
    ///
    /// Entry order is the payload's, which is document order at the producer; the
    /// map is a `HashMap` keyed by guid, so nothing downstream can see it. A file
    /// that will not open is **skipped with a warning** — the `from_pack` rule:
    /// one bad asset must not take a level down, and the entity that named it
    /// falls to the same stated miss an absent one does.
    pub fn from_paths(paths: &[(Uuid, String)]) -> Self {
        let mut out = Self::new();
        for (guid, path) in paths {
            out.open_file(*guid, Path::new(path));
        }
        out
    }

    /// Read one loose `.inf_vmesh` payload and index it under `guid` — the ONE
    /// open rule the two loose-file doors ([`from_dir`](Self::from_dir) and
    /// [`from_paths`](Self::from_paths)) share.
    fn open_file(&mut self, guid: Uuid, path: &Path) {
        match std::fs::read(path)
            .map_err(|e| e.to_string())
            .and_then(VgeomSource::from_payload)
        {
            Ok(src) => {
                self.meshes.insert(guid, Arc::new(src));
            }
            Err(e) => tracing::warn!("inf-player: bad .inf_vmesh {}: {e}", path.display()),
        }
    }

    /// Every indexed vmesh asset guid, **sorted** — the set two hosts are
    /// compared on (`island_gate`'s registry arm, wave FIX2).
    ///
    /// Sorted rather than in insertion order because the underlying map is a
    /// `HashMap`: an unsorted answer would be a different sequence on two runs of
    /// the same build, and a gate over it would be measuring the hasher.
    pub fn registered_guids(&self) -> Vec<Uuid> {
        let mut out: Vec<Uuid> = self.meshes.keys().copied().collect();
        out.sort();
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
    /// has no derived vmesh (an un-cooked / non-dense mesh), which since wave FIX2
    /// means the entity draws NOTHING and says so once — see
    /// [`report_missing`](Self::report_missing).
    pub fn resolve(&self, mesh_id: Uuid) -> Option<(u128, Arc<VgeomSource>)> {
        let vmesh_id = derived_vmesh_id(mesh_id);
        match self.meshes.get(&vmesh_id) {
            Some(m) => Some((vmesh_id.as_u128(), m.clone())),
            None => {
                self.report_missing(mesh_id, vmesh_id);
                None
            }
        }
    }

    /// **Say what will not be drawn, once** (wave FIX2).
    ///
    /// Until this wave a `MeshRef.asset` with no derived DAG drew a 1 m
    /// placeholder cube, in the editor viewport, in PIE and in the shipped build
    /// alike. That is a claim about the world no author made, and on the island
    /// it hid four missing streets behind four boxes 2.7 km from the spawn. The
    /// draw is gone; this is what replaced it, and it is stated at the ONE seam
    /// both hosts resolve through so neither can be the quiet one.
    ///
    /// `error!` and not `warn!`: a level that names geometry the runtime cannot
    /// find is a broken build, not a tuning note. Once per mesh per session —
    /// this runs inside a per-frame projection over every entity.
    fn report_missing(&self, mesh_id: Uuid, vmesh_id: Uuid) {
        let Ok(mut seen) = self.missing.lock() else {
            return;
        };
        if !seen.insert(mesh_id) {
            return;
        }
        tracing::error!(
            "inf-player: mesh {mesh_id} has no derived meshlet DAG ({vmesh_id}), so it \
             DRAWS NOTHING — a cooked pack derives one for every mesh past [vgeom] \
             min_triangles, and a PIE session is handed the editor's own through the \
             payload's vmesh_paths; if this is a PIE session the derivation sweep has \
             not finished, and if it is a shipped build the cook's sub-threshold \
             advisory names the mesh"
        );
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

    /// **The PIE door indexes what the payload names, under the guid it names**
    /// (wave FIX2).
    ///
    /// The whole of `from_paths`' contract in one arm: a file the payload names
    /// is opened and keyed under the *carried* guid — which is the DERIVED id, so
    /// `resolve` on the MESH id finds it — and a file the payload names that is
    /// not there is skipped rather than taking the level down.
    ///
    /// Non-vacuous: the registry answers `None` for this mesh before the payload
    /// is read, so the assertion below is the path route working and not a
    /// registry that was already full.
    #[test]
    fn a_payload_path_is_indexed_under_the_guid_the_payload_names() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mesh_id = Uuid::from_u128(0xF1_2201_0002);
        let vmesh_id = derived_vmesh_id(mesh_id);

        // A real v2 paged image, through the same writer the editor and the cook
        // both use — an invented byte string would only prove the error path.
        let payload =
            inf_vgeom::build_vgeom_asset(&empty_vmesh(), &inf_vgeom::ClusterTextureSet::none())
                .expect("the image writes")
                .into_bytes();
        let file = dir.path().join("Roads.inf_vmesh");
        std::fs::write(&file, &payload).expect("the payload writes");

        assert!(
            VmeshRegistry::new().resolve(mesh_id).is_none(),
            "the fixture resolves before anything is loaded"
        );

        let reg = VmeshRegistry::from_paths(&[(vmesh_id, file.to_string_lossy().to_string())]);
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.registered_guids(), vec![vmesh_id]);
        let (id, _) = reg
            .resolve(mesh_id)
            .expect("the MESH id resolves through the derived one");
        assert_eq!(id, vmesh_id.as_u128());

        // A named file that is not there is skipped, not fatal — the `from_pack`
        // rule — and the entity that named it reaches the stated miss instead.
        let missing = dir.path().join("Gone.inf_vmesh");
        let reg = VmeshRegistry::from_paths(&[
            (vmesh_id, file.to_string_lossy().to_string()),
            (
                Uuid::from_u128(0xDEAD),
                missing.to_string_lossy().to_string(),
            ),
        ]);
        assert_eq!(
            reg.len(),
            1,
            "an unreadable entry took the whole level down"
        );
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
            meshlet_materials: Vec::new(),
        }
    }
}
