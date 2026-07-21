//! Cook orchestration (P9.2, deliverable 2).
//!
//! [`cook`] turns a project's loose Content directory into a shippable build:
//! one content-addressed `.inf_pack` plus a `manifest.toml`. The stages:
//!
//! 1. **Open** the project + **scan** its asset database.
//! 2. **Resolve roots.** Explicit `--roots`, else the default set: every level
//!    plus every script asset (`.inf_act`/`.inf_fn`). Levels are the entry
//!    points; scripts are kept as implicit default roots so a project with
//!    library functions or not-yet-bound classes still ships its gameplay logic.
//!    Data assets enter only through a root's dependency closure.
//! 3. **Close** over forward dependency edges (BFS) to pull in referenced assets:
//!    the explicit sidecar `dependencies` **and** a level's persisted per-entity
//!    refs — its `actor` blueprint bindings (schema v3) and its `PcgVolume.graph`
//!    scatter-graph refs (schema v4) — real level→asset edges, so an
//!    explicit-roots cook of just a level still ships its bound classes + PCG
//!    graphs. Unreferenced strays are dropped.
//! 4. **Compile blueprints** — decode + migrate + statically validate every
//!    `.inf_act`/`.inf_fn` IR; a broken graph fails the cook with a
//!    handler-anchored error.
//! 5. **Rewrite levels for runtime** — decode each `.inf_lvl` with the Ring-0
//!    [`inf_scene`] reader (validating it) and re-encode to the current runtime
//!    schema (upgrading legacy v1 levels to v2).
//! 6. **Derive virtualized geometry** (P13.1) — for every `.inf_mesh` in the
//!    closure whose triangle count is at least
//!    [`VgeomCookOptions::min_triangles`], build a meshlet LOD DAG
//!    ([`inf_vgeom::build_vgeom`]) and pack it beside the mesh as an `.inf_vmesh`
//!    ([`AssetKind::MeshletMesh`]). The source `.inf_mesh` stays authoring-clean;
//!    the render-optimized form is *derived* at cook (roadmap P13.1). The derived
//!    asset's GUID is a deterministic function of the mesh GUID
//!    ([`derived_vmesh_id`]) so both cooks and the runtime agree without an index:
//!    the next-wave renderer, when virtualized geometry is enabled, computes the
//!    vmesh id from a mesh's id and prefers it if present in the pack, else falls
//!    back to the classic `.inf_mesh` LOD path (roadmap risk #3). The build is
//!    deterministic (`meshopt` + `inf_core::parallel_map`'s in-order collect), so
//!    the vmesh derivation preserves the cook's byte-identical guarantee.
//! 7. **Pack + manifest** — write `content.inf_pack` (sorted, zstd, deterministic)
//!    and a deterministic `manifest.toml`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use inf_asset::{AssetDb, AssetId, AssetKind, PackWriter};
use inf_blueprint::{BlueprintClass, BlueprintLibrary};
use inf_project::Project;

use crate::blueprint::{validate_class, validate_library};
use crate::error::{CookError, Result};
use crate::manifest::{CookManifest, MANIFEST_FILE, MANIFEST_SCHEMA_VERSION};

/// The default pack file name written into the output directory.
pub const DEFAULT_PACK_NAME: &str = "content.inf_pack";

/// Options controlling a cook.
#[derive(Debug, Clone, Default)]
pub struct CookOptions {
    /// Explicit root asset GUIDs. `None` uses the default roots (all levels +
    /// all script assets).
    pub roots: Option<Vec<AssetId>>,
    /// Pack file name (defaults to [`DEFAULT_PACK_NAME`]).
    pub pack_name: Option<String>,
    /// Virtualized-geometry (`.inf_vmesh`) derivation controls.
    pub vgeom: VgeomCookOptions,
}

/// Controls the cook's virtualized-geometry derivation (P13.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VgeomCookOptions {
    /// Whether to derive `.inf_vmesh` meshlet DAGs from cooked meshes.
    pub enabled: bool,
    /// Only meshes with at least this many triangles get a derived DAG (below it
    /// the classic single-mesh path is cheaper than a virtualized one).
    pub min_triangles: usize,
}

impl Default for VgeomCookOptions {
    fn default() -> Self {
        // Enabled by default, but only for meshes dense enough to benefit — small
        // props stay on the classic path (and the derivation is a no-op for them).
        Self {
            enabled: true,
            min_triangles: 2048,
        }
    }
}

/// The fixed salt XORed into a mesh GUID to derive its `.inf_vmesh` GUID.
///
/// XOR with a constant is a bijection, so distinct mesh ids always yield distinct
/// vmesh ids; the salt makes a collision with any *authored* asset id vanishingly
/// unlikely (and the cook guards the remaining case). This lets the runtime find
/// a mesh's virtualized form by computing the id — no side index needed.
const VMESH_ID_SALT: u128 = 0x7635_4e56_4d45_5348_1f13_1a2b_3c4d_5e6f;

/// Derive the deterministic `.inf_vmesh` asset id for a given mesh id.
pub fn derived_vmesh_id(mesh_id: AssetId) -> AssetId {
    AssetId(uuid::Uuid::from_u128(
        mesh_id.uuid().as_u128() ^ VMESH_ID_SALT,
    ))
}

/// The outcome of a successful cook.
#[derive(Debug, Clone)]
pub struct CookReport {
    pub project_name: String,
    pub engine_version: String,
    /// Output directory the build was written to.
    pub out_dir: PathBuf,
    /// The written pack file.
    pub pack_path: PathBuf,
    /// The written manifest file.
    pub manifest_path: PathBuf,
    /// Total assets packed.
    pub asset_count: usize,
    /// Per-kind counts (keyed by kind slug, sorted).
    pub kinds: BTreeMap<String, usize>,
    /// Size of the written pack in bytes.
    pub pack_bytes: u64,
    /// Level GUIDs in the pack (sorted).
    pub levels: Vec<AssetId>,
    /// The primary level (lowest GUID), if any.
    pub root_level: Option<AssetId>,
    /// How many blueprint assets were validated.
    pub blueprints_validated: usize,
    /// How many levels were rewritten to the runtime schema.
    pub levels_rewritten: usize,
    /// How many `.inf_vmesh` meshlet DAGs were derived from meshes (P13.1).
    pub meshlet_meshes_derived: usize,
    /// Non-fatal advisories (e.g. "no levels").
    pub warnings: Vec<String>,
}

impl CookReport {
    /// A human-readable multi-line summary for CLI output.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Cooked \"{}\" (engine {})\n",
            self.project_name, self.engine_version
        ));
        s.push_str(&format!(
            "  {} → {} ({} bytes)\n",
            self.pack_path
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default(),
            self.out_dir.display(),
            self.pack_bytes
        ));
        s.push_str(&format!("  {} assets:\n", self.asset_count));
        for (kind, n) in &self.kinds {
            s.push_str(&format!("    {kind:<18} {n}\n"));
        }
        s.push_str(&format!(
            "  {} blueprint(s) validated, {} level(s) rewritten for runtime\n",
            self.blueprints_validated, self.levels_rewritten
        ));
        if self.meshlet_meshes_derived > 0 {
            s.push_str(&format!(
                "  {} meshlet DAG(s) derived (.inf_vmesh)\n",
                self.meshlet_meshes_derived
            ));
        }
        match self.root_level {
            Some(l) => s.push_str(&format!("  root level: {l}\n")),
            None => s.push_str("  root level: (none)\n"),
        }
        for w in &self.warnings {
            s.push_str(&format!("  warning: {w}\n"));
        }
        s
    }
}

/// Cook `project_root` into `out_dir`, producing a pack + manifest.
pub fn cook(project_root: &Path, out_dir: &Path, opts: &CookOptions) -> Result<CookReport> {
    let project = Project::open(project_root)?;
    let mut db = AssetDb::new(project.content_root());
    db.scan()?;

    // ── 2. resolve roots ────────────────────────────────────────────────────
    let mut warnings = Vec::new();
    let roots: Vec<AssetId> = match &opts.roots {
        Some(explicit) => {
            for &r in explicit {
                if !db.contains(r) {
                    return Err(CookError::UnknownRoot(r));
                }
            }
            explicit.clone()
        }
        None => {
            let mut r: Vec<AssetId> = db
                .iter()
                .filter(|e| is_root_kind(e.kind()))
                .map(|e| e.id())
                .collect();
            r.sort();
            r
        }
    };

    // ── 3. dependency closure (BFS over forward edges) ──────────────────────
    let closure = dependency_closure(&db, &roots);

    // ── 4/5. compile blueprints + rewrite levels; collect cooked payloads ───
    let mut writer = PackWriter::new();
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    let mut levels: Vec<AssetId> = Vec::new();
    let mut blueprints_validated = 0usize;
    let mut levels_rewritten = 0usize;
    let mut meshlet_meshes_derived = 0usize;
    // Meshlet DAGs derived alongside their meshes, added after the closure so a
    // derived id never shadows a real closure asset processed later.
    let mut derived_vmeshes: Vec<(AssetId, Vec<u8>)> = Vec::new();

    for guid in &closure {
        let entry = db.get(*guid).ok_or(CookError::UnknownRoot(*guid))?.clone();
        let kind = entry.kind();
        if kind == AssetKind::Unknown {
            warnings.push(format!(
                "skipped unknown-kind asset at {}",
                entry.path.display()
            ));
            continue;
        }
        let raw = std::fs::read(&entry.path)?;

        let cooked: Vec<u8> = match kind {
            AssetKind::Level => {
                let level = inf_scene::decode(&raw).map_err(|source| CookError::Scene {
                    guid: *guid,
                    source,
                })?;
                levels.push(*guid);
                levels_rewritten += 1;
                inf_scene::encode(&level).map_err(|source| CookError::Scene {
                    guid: *guid,
                    source,
                })?
            }
            AssetKind::Blueprint => {
                let mut class: BlueprintClass =
                    serde_json::from_slice(&raw).map_err(|e| CookError::Blueprint {
                        guid: *guid,
                        class: entry.name.clone(),
                        handler: "<decode>".into(),
                        message: e.to_string(),
                    })?;
                if let Some(issue) = validate_class(&mut class) {
                    return Err(CookError::Blueprint {
                        guid: *guid,
                        class: class.name,
                        handler: issue.handler,
                        message: issue.message,
                    });
                }
                blueprints_validated += 1;
                raw
            }
            AssetKind::FunctionLib => {
                let mut lib: BlueprintLibrary =
                    serde_json::from_slice(&raw).map_err(|e| CookError::Blueprint {
                        guid: *guid,
                        class: entry.name.clone(),
                        handler: "<decode>".into(),
                        message: e.to_string(),
                    })?;
                if let Some(issue) = validate_library(&mut lib) {
                    return Err(CookError::Blueprint {
                        guid: *guid,
                        class: lib.name,
                        handler: issue.handler,
                        message: issue.message,
                    });
                }
                blueprints_validated += 1;
                raw
            }
            // Data assets ride through verbatim (already deterministic bincode).
            _ => raw,
        };

        writer.add_bytes(*guid, kind, &cooked)?;
        *kinds.entry(kind.slug().to_string()).or_default() += 1;

        // ── 6. derive virtualized geometry for dense meshes ─────────────────
        if kind == AssetKind::Mesh && opts.vgeom.enabled {
            // For a mesh `cooked == raw` (it rides through verbatim).
            if let Some(vmesh_bytes) = derive_vmesh(*guid, &cooked, opts.vgeom.min_triangles)? {
                let vmesh_id = derived_vmesh_id(*guid);
                // Guard the (astronomically unlikely) collision with a real asset
                // or an already-derived vmesh; skip with a warning rather than
                // corrupt the pack.
                if db.contains(vmesh_id) || derived_vmeshes.iter().any(|(id, _)| *id == vmesh_id) {
                    warnings.push(format!(
                        "skipped vmesh for mesh {guid}: derived id {vmesh_id} collides"
                    ));
                } else {
                    derived_vmeshes.push((vmesh_id, vmesh_bytes));
                }
            }
        }
    }

    // Pack the derived meshlet DAGs (order-independent — the writer sorts by GUID).
    for (vmesh_id, bytes) in &derived_vmeshes {
        writer.add_bytes(*vmesh_id, AssetKind::MeshletMesh, bytes)?;
        *kinds
            .entry(AssetKind::MeshletMesh.slug().to_string())
            .or_default() += 1;
        meshlet_meshes_derived += 1;
    }

    // ── 7. write pack + manifest ────────────────────────────────────────────
    std::fs::create_dir_all(out_dir)?;
    let pack_name = opts
        .pack_name
        .clone()
        .unwrap_or_else(|| DEFAULT_PACK_NAME.to_string());
    let pack_path = out_dir.join(&pack_name);
    writer.write_to_file(&pack_path)?;
    let pack_bytes = std::fs::metadata(&pack_path)?.len();

    levels.sort();
    let root_level = levels.first().copied();
    if levels.is_empty() {
        warnings.push("no levels in cook — the build has no boot scene".into());
    }

    let manifest = CookManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        engine_version: project.manifest.engine_version.clone(),
        project_name: project.manifest.name.clone(),
        packs: vec![pack_name],
        root_level: root_level.map(|a| a.uuid()),
        levels: levels.iter().map(|a| a.uuid()).collect(),
        asset_count: writer.len() as u32,
        kinds: kinds.iter().map(|(k, v)| (k.clone(), *v as u32)).collect(),
    };
    let manifest_path = out_dir.join(MANIFEST_FILE);
    std::fs::write(&manifest_path, manifest.to_toml()?)?;

    Ok(CookReport {
        project_name: project.manifest.name.clone(),
        engine_version: project.manifest.engine_version.clone(),
        out_dir: out_dir.to_path_buf(),
        pack_path,
        manifest_path,
        asset_count: writer.len(),
        kinds,
        pack_bytes,
        levels,
        root_level,
        blueprints_validated,
        levels_rewritten,
        meshlet_meshes_derived,
        warnings,
    })
}

/// Build the `.inf_vmesh` payload for a `.inf_mesh`, or `None` if the mesh is
/// below `min_triangles` (stays on the classic path) or has no geometry.
fn derive_vmesh(guid: AssetId, raw: &[u8], min_triangles: usize) -> Result<Option<Vec<u8>>> {
    let mesh: inf_mesh::MeshAsset = inf_asset::decode(raw).map_err(|e| CookError::Mesh {
        guid,
        message: e.to_string(),
    })?;
    if mesh.triangle_count() < min_triangles.max(1) {
        return Ok(None);
    }
    let (positions, normals, uvs, indices) = mesh.vgeom_streams();
    if indices.len() < 3 {
        return Ok(None);
    }
    let vgeom = inf_vgeom::build_vgeom(
        &positions,
        &normals,
        &uvs,
        &indices,
        inf_vgeom::BuildParams::default(),
    );
    let bytes = inf_asset::encode(&vgeom).map_err(|e| CookError::Mesh {
        guid,
        message: e.to_string(),
    })?;
    Ok(Some(bytes))
}

/// Kinds that are cook roots by default: levels (entry points) and script assets
/// (gameplay logic, always shipped — see the module docs).
fn is_root_kind(kind: AssetKind) -> bool {
    matches!(
        kind,
        AssetKind::Level | AssetKind::Blueprint | AssetKind::FunctionLib
    )
}

/// The transitive closure of `roots` over forward dependency edges, returned
/// sorted (deterministic packing order).
///
/// Two edge sources are followed: the explicit sidecar `dependencies`
/// ([`AssetDb::references_of`]) and — for `.inf_lvl` levels — the **persisted
/// per-entity `actor` bindings** (P9.5), which form a real level→blueprint edge
/// so an explicit-roots cook (`--roots <level>`) still ships the bound classes.
fn dependency_closure(db: &AssetDb, roots: &[AssetId]) -> Vec<AssetId> {
    let mut seen: BTreeSet<AssetId> = BTreeSet::new();
    let mut queue: VecDeque<AssetId> = VecDeque::new();
    for &r in roots {
        if db.contains(r) && seen.insert(r) {
            queue.push_back(r);
        }
    }
    while let Some(id) = queue.pop_front() {
        if let Some(deps) = db.references_of(id) {
            for &dep in deps {
                if db.contains(dep) && seen.insert(dep) {
                    queue.push_back(dep);
                }
            }
        }
        for dep in asset_deps(db, id) {
            if db.contains(dep) && seen.insert(dep) {
                queue.push_back(dep);
            }
        }
    }
    seen.into_iter().collect()
}

/// The asset GUIDs an asset references through its persisted refs — the real
/// forward edges (beyond the explicit sidecar `dependencies`) the cook must close
/// over. Empty for kinds with no such refs, or an undecodable payload (decode
/// errors surface later in the real cook stage with a proper error).
///
/// * **Level** (`.inf_lvl`) — its persisted per-entity slots: `actor`
///   blueprint-class bindings (v3), `PcgVolume.graph` scatter refs (v4), and the
///   v5 animation-component refs: `SkeletalMesh.{skeleton, mesh}`,
///   `AnimPlayer.clip`, `AnimStateMachine.sm` (real level→anim edges, so an
///   explicit-roots cook of just a level ships its referenced anim assets).
/// * **StateMachine** (`.inf_sm`) — the clip GUIDs its states/blend-spaces play
///   (`Motion::Clip` + blend entries) **and** its skeleton ref. This closes the
///   `state-machine → clip` edge so the clips a machine plays ship in the pack.
/// * **AnimClip** (`.inf_anim`) — the skeleton GUID it was authored against.
fn asset_deps(db: &AssetDb, id: AssetId) -> Vec<AssetId> {
    let Some(entry) = db.get(id) else {
        return Vec::new();
    };
    let Ok(raw) = std::fs::read(&entry.path) else {
        return Vec::new();
    };
    match entry.kind() {
        AssetKind::Level => {
            let Ok(level) = inf_scene::decode(&raw) else {
                return Vec::new();
            };
            let mut deps: Vec<AssetId> = Vec::new();
            for e in &level.entities {
                deps.extend(e.actor.map(AssetId));
                deps.extend(e.pcg_volume.as_ref().and_then(|v| v.graph).map(AssetId));
                if let Some(sk) = &e.skeletal_mesh {
                    deps.extend(sk.skeleton.map(AssetId));
                    deps.extend(sk.mesh.map(AssetId));
                }
                deps.extend(e.anim_player.as_ref().and_then(|p| p.clip).map(AssetId));
                deps.extend(
                    e.anim_state_machine
                        .as_ref()
                        .and_then(|s| s.sm)
                        .map(AssetId),
                );
                // P12.4: an AudioSource pulls its referenced `.inf_audio` clip.
                deps.extend(e.audio_source.as_ref().and_then(|a| a.clip).map(AssetId));
            }
            deps
        }
        AssetKind::StateMachine => {
            let Ok(sm) = inf_asset::decode::<inf_anim::StateMachineAsset>(&raw) else {
                return Vec::new();
            };
            let mut deps: Vec<AssetId> = Vec::new();
            deps.extend(sm.skeleton.map(|b| AssetId(uuid::Uuid::from_bytes(b))));
            for st in &sm.machine.states {
                for clip in motion_clip_refs(&st.motion) {
                    deps.push(AssetId(uuid::Uuid::from_bytes(clip)));
                }
            }
            deps
        }
        AssetKind::AnimClip => {
            let Ok(clip) = inf_asset::decode::<inf_anim::AnimClipAsset>(&raw) else {
                return Vec::new();
            };
            clip.skeleton
                .map(|b| AssetId(uuid::Uuid::from_bytes(b)))
                .into_iter()
                .collect()
        }
        _ => Vec::new(),
    }
}

/// Every clip GUID a state's [`Motion`](inf_anim::state_machine::Motion) plays: a
/// single clip, or every entry of a 1D/2D blend space.
fn motion_clip_refs(motion: &inf_anim::state_machine::Motion) -> Vec<[u8; 16]> {
    use inf_anim::state_machine::Motion;
    match motion {
        Motion::Clip(c) => vec![*c],
        Motion::Blend1D(space) => space.entries.iter().map(|e| e.clip).collect(),
        Motion::Blend2D(space) => space.entries.iter().map(|e| e.clip).collect(),
    }
}
