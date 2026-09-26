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
    /// **Each mesh's drawn SECTIONS** (wave VEH3f.2a), found by computing their
    /// ids (`inf_mesh::section_mesh_id`) the first time a mesh is asked about
    /// and remembered: `(slot, vmesh id)` in slot order, empty for a mesh drawn
    /// whole -- which is every mesh an importer did not section.
    sections: Mutex<SectionCache>,
    /// **The crumpled DAGs** (wave VEH3f.2b) -- one per `(DAG, quantized dent)`
    /// a car's hull has been drawn with, built on first sight by
    /// [`dented`](Self::dented) and kept while it is still drawn.
    dents: Mutex<DentCache>,
}

/// `(slot, vmesh id)` per mesh, slot order -- see `VmeshRegistry::sections`.
type SectionCache = HashMap<Uuid, Arc<[(u32, Uuid)]>>;

/// One crumpled variant: its drawn id, its DAG and what the crumple did.
#[derive(Clone)]
pub struct DentedDag {
    /// The id the renderer streams it under -- never a real asset's.
    pub id: u128,
    /// The crumpled DAG.
    pub source: Arc<VgeomSource>,
    /// The vertex census of the crumple (`inf_ecs::bodywork::MeshDent`).
    pub census: inf_ecs::bodywork::MeshDent,
}

/// `(DAG id, dent key)` -> the variant, and how many were ever built.
#[derive(Default)]
struct DentCache {
    map: HashMap<(u128, [i32; 7]), DentedDag>,
    builds: u64,
}

/// **How finely a drawn dent is quantized** -- a new crumpled DAG is built when
/// the dent moves by more than this, metres. Half a centimetre: finer than the
/// eye reads on a car at street distance, coarse enough that a car being
/// shunted along a kerb builds a handful of variants and not one a step.
pub const DENT_DRAW_QUANTUM_M: f64 = 0.005;

/// How many crumpled DAGs the registry keeps before it forgets the oldest.
pub const MAX_DENTED_DAGS: usize = 48;

/// One drawn section of a sectioned mesh: its slot, and its DAG.
pub type VmeshSection = (u32, u128, Arc<VgeomSource>);

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

    /// **The crumpled form of a hull's DAG** (wave VEH3f.2b) -- `source`
    /// (drawn under `id`) with every vertex pushed by
    /// [`inf_ecs::bodywork::dent_mesh_positions`] for a dent of `depth_m`
    /// along `dir` (chassis frame) at the instance's `scale`.
    ///
    /// Every LOD level of the DAG shares its vertex buffer, and a meshlet
    /// simplifier keeps the vertices it keeps where they were, so displacing
    /// the one buffer crumples every level alike -- nothing is rebuilt but the
    /// image. Each meshlet's bounding sphere grows by the deepest push (the
    /// cull must not lose a vertex that moved out of its sphere) and its
    /// normal cone is retired (a crumple turns faces).
    ///
    /// The dent is quantized to [`DENT_DRAW_QUANTUM_M`] and the direction to
    /// 1/256, so a variant is built once per visible change and then reused.
    /// `None` when the payload cannot be read, and then the car draws its
    /// undented body -- which is what it did before this wave.
    pub fn dented(
        &self,
        id: u128,
        source: &Arc<VgeomSource>,
        scale: [f64; 3],
        dir: inf_ecs::math::Vec3d,
        depth_m: f64,
    ) -> Option<DentedDag> {
        let q = |v: f64, step: f64| (v / step).round() as i32;
        let key = [
            q(depth_m, DENT_DRAW_QUANTUM_M),
            q(dir.x, 1.0 / 256.0),
            q(dir.y, 1.0 / 256.0),
            q(dir.z, 1.0 / 256.0),
            q(scale[0], 1e-3),
            q(scale[1], 1e-3),
            q(scale[2], 1e-3),
        ];
        if key[0] <= 0 {
            return None;
        }
        if let Some(d) = self.dents.lock().ok()?.map.get(&(id, key)) {
            return Some(d.clone());
        }
        let depth = key[0] as f64 * DENT_DRAW_QUANTUM_M;
        let dir = inf_ecs::math::Vec3d::new(
            key[1] as f64 / 256.0,
            key[2] as f64 / 256.0,
            key[3] as f64 / 256.0,
        );
        let payload = source.payload()?;
        let reader = inf_vgeom::VgeomAssetReader::new(payload.as_ref()).ok()?;
        let mut mesh = reader.to_mesh().ok()?;
        drop(payload);
        let mut pos: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
        let census = inf_ecs::bodywork::dent_mesh_positions(&mut pos, scale, dir, depth);
        for (v, p) in mesh.vertices.iter_mut().zip(&pos) {
            v.position = *p;
        }
        let min_scale = scale
            .iter()
            .fold(f64::INFINITY, |m, s| m.min(s.abs()))
            .max(1e-9);
        let grow = (census.max_m / min_scale) as f32;
        for m in mesh.meshlets.iter_mut() {
            m.radius += grow;
            m.cone_cutoff = 1.0;
        }
        mesh.radius += grow;
        let built = Arc::new(VgeomSource::from_mesh(&mesh).ok()?);
        // A drawn id no real asset has: the DAG's own id, mixed with the key.
        let mut h: u128 = id ^ 0x5645_4833_4632_4244_454e_5400_0000_0000;
        for k in key {
            h = h.rotate_left(29) ^ (k as u32 as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
        let out = DentedDag {
            id: h,
            source: built,
            census,
        };
        let mut c = self.dents.lock().ok()?;
        if c.map.len() >= MAX_DENTED_DAGS {
            c.map.clear();
        }
        c.builds += 1;
        c.map.insert((id, key), out.clone());
        Some(out)
    }

    /// How many crumpled DAGs have ever been built -- the engagement count a
    /// gate reads to know the crumple path ran at all.
    pub fn dent_builds(&self) -> u64 {
        self.dents.lock().map(|c| c.builds).unwrap_or(0)
    }

    /// The crumpled DAGs held right now, `(drawn id, census)`, id order.
    pub fn dent_census(&self) -> Vec<(u128, inf_ecs::bodywork::MeshDent)> {
        let mut out: Vec<(u128, inf_ecs::bodywork::MeshDent)> = self
            .dents
            .lock()
            .map(|c| c.map.values().map(|d| (d.id, d.census)).collect())
            .unwrap_or_default();
        out.sort_by_key(|(id, _)| *id);
        out
    }

    /// Whether the vmesh with this asset GUID is indexed.
    pub fn contains(&self, vmesh_id: Uuid) -> bool {
        self.meshes.contains_key(&vmesh_id)
    }

    /// Register a source under its asset GUID (used by loaders + tests).
    pub fn insert(&mut self, vmesh_id: Uuid, source: Arc<VgeomSource>) {
        self.meshes.insert(vmesh_id, source);
        // A new DAG can be some mesh's section: forget what was found.
        if let Ok(mut c) = self.sections.lock() {
            c.clear();
        }
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
        if let Ok(mut c) = self.sections.lock() {
            c.clear();
        }
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
    /// referencing it reaches the same stated miss (`report_missing`) an un-cooked
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
    /// means the entity draws NOTHING and says so once — see `report_missing`.
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

    /// **The drawn sections of `mesh_id`** (wave VEH3f.2a), slot order -- empty
    /// for a mesh drawn whole. See `inf_mesh::section` for the arrangement; a
    /// section whose DAG is absent is simply not listed (the parent is then
    /// drawn whole, which is what a checkout without the local art does).
    pub fn sections(&self, mesh_id: Uuid) -> Vec<VmeshSection> {
        let ids: Arc<[(u32, Uuid)]> = {
            let Ok(mut cache) = self.sections.lock() else {
                return Vec::new();
            };
            cache
                .entry(mesh_id)
                .or_insert_with(|| {
                    (0..inf_mesh::MAX_SECTIONS)
                        .filter_map(|s| {
                            let sid = inf_mesh::section_mesh_id(AssetId(mesh_id), s).uuid();
                            let v = derived_vmesh_id(sid);
                            self.meshes.contains_key(&v).then_some((s, v))
                        })
                        .collect::<Vec<_>>()
                        .into()
                })
                .clone()
        };
        ids.iter()
            .filter_map(|(s, v)| self.meshes.get(v).map(|m| (*s, v.as_u128(), m.clone())))
            .collect()
    }

    /// **Say what will not be drawn, once** (wave FIX2).
    ///
    /// Until this wave a `MeshRef.asset` with no derived DAG drew a 1 m
    /// placeholder cube, in the editor viewport, in PIE and in the shipped build
    /// alike. That is a claim about the world no author made, and on the island
    /// it hid four missing streets behind four boxes 2.7 km from the spawn. The
    /// draw is gone; this is what replaced it.
    ///
    /// **This is the PLAYER's seam, not a shared one** (FIX2 audit corrects the
    /// sentence that stood here). The editor viewport does not reach this
    /// function at all: `inf_viewport::EngineHost` resolves through
    /// `EditorRenderAssets::resolve_vgeom` → `open_vgeom`, which warns for a
    /// STALE derivation and returns `None` in silence for an ABSENT one. So a
    /// mesh with no DAG draws nothing on both hosts and only one of them says so.
    ///
    /// That asymmetry is defensible and is the reason it is written down rather
    /// than closed: in the editor an absent DAG is TRANSIENT — the project-open
    /// sweep derives one from a single triangle for every mesh in the project, so
    /// a line per mesh would fire for the whole content root every session and
    /// stop being read. In a player it is terminal: nothing is going to derive
    /// one, and the frame is missing geometry until the project is re-cooked.
    ///
    /// `error!` and not `warn!`: a level that names geometry the runtime cannot
    /// find is a broken build, not a tuning note. Once per mesh per session —
    /// this runs inside a per-frame projection over every entity, and the
    /// once-ness is measured: 300 projections over two missing meshes emit two
    /// lines (FIX2 audit).
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

    /// **The dev-directory door indexes what is beside the level, under the guid
    /// its SIDECAR names** (FIX2 audit).
    ///
    /// [`VmeshRegistry::from_dir`] is what a `--level` boot builds its registry
    /// from, and it had no arm anywhere in the tree — measured: breaking the
    /// `open_file` the two loose-file doors share reddened
    /// `a_payload_path_is_indexed_under_the_guid_the_payload_names` and **nothing
    /// else in the whole workspace**. So wave FIX2's "one door" was fenced from
    /// the `from_paths` side only, and the shipped side of the same rewrite was
    /// asserted by reading it.
    ///
    /// Three facts, and each is a different way the door has been wrong before:
    /// the guid comes off the SIDECAR (not off the filename or the payload), a
    /// `.inf_vmesh` with no readable sidecar is skipped rather than fatal, and
    /// `resolve` finds the entry from the MESH id — the derived-id rule
    /// `from_paths` also obeys, which is what makes them one lookup rule.
    #[test]
    fn a_dev_directory_is_indexed_under_the_guid_each_sidecar_names() {
        let dir = tempfile::tempdir().expect("tempdir");
        let payload =
            inf_vgeom::build_vgeom_asset(&empty_vmesh(), &inf_vgeom::ClusterTextureSet::none())
                .expect("the image writes")
                .into_bytes();

        let mut want: Vec<Uuid> = Vec::new();
        for (stem, mesh) in [
            ("Roads", Uuid::from_u128(0xF1_2202_0001)),
            ("Kerbs", Uuid::from_u128(0xF1_2202_0002)),
        ] {
            let vmesh_id = derived_vmesh_id(mesh);
            let file = dir.path().join(format!("{stem}.inf_vmesh"));
            std::fs::write(&file, &payload).expect("the payload writes");
            inf_asset::AssetSidecar::new(
                inf_asset::AssetId(vmesh_id),
                inf_asset::AssetKind::MeshletMesh,
                inf_asset::ContentHash(0),
            )
            .save(&file)
            .expect("the sidecar writes");
            want.push(vmesh_id);
        }
        // …and one with no sidecar at all, which must be skipped and must not
        // take the other two down with it.
        std::fs::write(dir.path().join("Orphan.inf_vmesh"), &payload).expect("writes");

        let reg = VmeshRegistry::from_dir(dir.path());
        assert_eq!(reg.len(), 2, "a sidecar-less file took the directory down");
        want.sort();
        assert_eq!(reg.registered_guids(), want, "keyed by something else");
        for mesh in [
            Uuid::from_u128(0xF1_2202_0001),
            Uuid::from_u128(0xF1_2202_0002),
        ] {
            let (id, _) = reg
                .resolve(mesh)
                .expect("the MESH id resolves through the derived one");
            assert_eq!(id, derived_vmesh_id(mesh).as_u128());
        }
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
