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
//!    handler-anchored error. Streaming assets are structurally validated in the
//!    same stage: a `.inf_terrain`'s header + tile directory must parse (P16.3),
//!    because the runtime pages tiles by trusting that structure.
//! 5. **Rewrite levels for runtime** — decode each `.inf_lvl` with the Ring-0
//!    [`inf_scene`] reader (validating it) and re-encode to the current runtime
//!    schema (upgrading legacy v1 levels to v2).
//!
//!    **World partition** (P16.5): a level whose `settings.partition.enabled` is
//!    set does not ship its entities inline. They are binned by
//!    [`inf_scene::partition::partition_entities`] into a persistent cell plus a
//!    grid of streamed cells, written to a derived `.inf_part`
//!    ([`AssetKind::Partition`], stored **uncompressed** so the runtime slices one
//!    cell straight out of the mapping), and the cooked level keeps only its title
//!    and settings. Its GUID is a deterministic function of the level's
//!    ([`derived_partition_id`]) — the [`derived_vmesh_id`] precedent — so the
//!    runtime finds it with no side index. A level with partitioning **off** cooks
//!    byte-for-byte as it did before this existed.
//! 6. **Derive virtualized geometry** (P13.1) — for every `.inf_mesh` in the
//!    closure whose triangle count is at least
//!    [`VgeomCookOptions::min_triangles`], build a meshlet LOD DAG
//!    ([`inf_vgeom::build_vgeom`]), lay it out as the v2 **paged** `.inf_vmesh`
//!    image ([`inf_vgeom::build_vgeom_asset`] — a raw image, never
//!    `inf_asset::encode`, so a runtime can slice a meshlet page straight out of
//!    the mmap'd pack) and pack it beside the mesh as an `.inf_vmesh`
//!    ([`AssetKind::MeshletMesh`]). The source `.inf_mesh` stays authoring-clean;
//!    the render-optimized form is *derived* at cook (roadmap P13.1). The derived
//!    asset's GUID is a deterministic function of the mesh GUID
//!    ([`derived_vmesh_id`]) so both cooks and the runtime agree without an index:
//!    the next-wave renderer, when virtualized geometry is enabled, computes the
//!    vmesh id from a mesh's id and prefers it if present in the pack, else falls
//!    back to the classic `.inf_mesh` LOD path (roadmap risk #3). The build is
//!    deterministic (`meshopt` + `inf_core::parallel_map`'s in-order collect), so
//!    the vmesh derivation preserves the cook's byte-identical guarantee.
//! 7. **Derive fracture chunk sets** (P22.2) — for every `.inf_mesh` a
//!    [`Destructible`] entity somewhere in the cooked content points at, run the
//!    Ring-0 Voronoi pre-fracture ([`inf_mesh::fracture::fracture_mesh`]) and
//!    pack the resulting chunk set beside the mesh as an `.inf_fracture`
//!    ([`AssetKind::Fracture`]). Its GUID is a deterministic function of the mesh
//!    GUID ([`derived_fracture_id`]) — the [`derived_vmesh_id`] precedent again —
//!    so the runtime finds a mesh's chunks by computing the id.
//!
//!    **Only meshes something declares destructible are fractured**, unlike the
//!    vmesh derivation which runs on every mesh above a threshold. The reason is
//!    that a fracture is *not* a cheaper form of the same asset: it is a
//!    different asset, useful only to gameplay that breaks the thing, and
//!    fracturing every mesh in a project would spend cook time and ship bytes on
//!    chunk sets nothing can ever load. `Destructible` is exactly the declaration
//!    that makes one useful, so it is the trigger. See [`plan_fractures`] for why
//!    that plan is built *before* the parallel stage.
//! 8. **Pack + manifest** — write `content.inf_pack` (sorted, zstd, deterministic)
//!    and a deterministic `manifest.toml`.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use inf_asset::{AssetDb, AssetId, AssetKind, PackWriter};
use inf_blueprint::{BlueprintClass, BlueprintLibrary};
use inf_mesh::fracture::{self, FractureParams, FractureSkip};
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
    /// Fracture (`.inf_fracture`) derivation controls (P22.2).
    pub fracture: FractureCookOptions,
}

/// Controls the cook's fracture derivation (P22.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FractureCookOptions {
    /// Whether to derive `.inf_fracture` chunk sets for destructible meshes.
    ///
    /// On by default. Off is for a build that deliberately ships without
    /// destruction (a bandwidth-constrained tier, a bisect); it does **not**
    /// change what is authored, and re-enabling it re-derives byte-identical
    /// chunk sets because the fracture is a pure function of the mesh.
    pub enabled: bool,
}

impl Default for FractureCookOptions {
    fn default() -> Self {
        Self { enabled: true }
    }
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

/// Derive the deterministic `.inf_vmesh` asset id for a given mesh id.
///
/// XOR with a constant is a bijection, so distinct mesh ids always yield distinct
/// vmesh ids; the salt makes a collision with any *authored* asset id vanishingly
/// unlikely (and the cook guards the remaining case). This lets the runtime find
/// a mesh's virtualized form by computing the id — no side index needed.
///
/// **P18.3**: the salt itself moved to Ring 0 ([`inf_vgeom::VMESH_ID_SALT`], the
/// crate that owns the `.inf_vmesh` format) rather than being hand-copied here and
/// in the player. The editor derives `.inf_vmesh` too now, and a third copy — with
/// a third drift test holding it in place — is one past the point where the
/// duplication is defensible.
pub fn derived_vmesh_id(mesh_id: AssetId) -> AssetId {
    inf_vgeom::derived_vmesh_id(mesh_id)
}

/// The fixed salt XORed into a level GUID to derive its `.inf_part` GUID.
///
/// Same construction (and same reasoning) as [`VMESH_ID_SALT`]: XOR with a
/// constant is a bijection, so distinct levels always yield distinct partition
/// ids, and the salt makes a collision with an *authored* asset id vanishingly
/// unlikely — the cook guards the remaining case. The runtime finds a level's
/// partition by computing the id, so no side index has to ship or stay in sync.
const PARTITION_ID_SALT: u128 = 0x7016_0500_5041_5254_9d4e_2c7a_b31f_60e8;

/// Derive the deterministic `.inf_part` asset id for a given level id.
pub fn derived_partition_id(level_id: AssetId) -> AssetId {
    AssetId(uuid::Uuid::from_u128(
        level_id.uuid().as_u128() ^ PARTITION_ID_SALT,
    ))
}

/// Derive the deterministic `.inf_fracture` asset id for a given mesh id (P22.2).
///
/// Same construction and the same reasoning as [`PARTITION_ID_SALT`]'s, and like
/// [`derived_vmesh_id`] the salt itself lives in **Ring 0** — in `inf-mesh`, the
/// crate that owns the `.inf_fracture` format ([`inf_mesh::fracture::FRACTURE_ID_SALT`]).
/// That placement is deliberate and taken *before* the second copy exists rather
/// than after the third: P22.3's runtime will need to compute this id too, and
/// hand-copying a constant into the player is precisely the drift the vmesh salt
/// was moved to Ring 0 to end.
pub fn derived_fracture_id(mesh_id: AssetId) -> AssetId {
    fracture::derived_fracture_id(mesh_id)
}

/// What one `Destructible` asks the cook to build for a mesh.
///
/// The count is stored **as authored**; the clamp happens where the advisory is
/// raised, so the report can quote both numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FractureRequest {
    params: FractureParams,
    /// The entity that asked, for advisory anchoring.
    entity: uuid::Uuid,
    /// The level that entity lives in.
    level: AssetId,
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
    /// How many `.inf_matd` runtime material records the cook derived (P26.3b) —
    /// one per `.inf_mat` in the dependency closure.
    pub materials_derived: usize,
    /// How many `.inf_fracture` chunk sets were derived from destructible
    /// meshes (P22.2).
    pub fractures_derived: usize,
    /// Total level-1 chunks across those fracture sets — the number that
    /// predicts P22.3's rigid-body cost, which the asset count does not.
    pub fracture_chunks: usize,
    /// Chunks that were **asked for and not produced**, across those sets: a
    /// site whose Voronoi cell missed the solid yields nothing. Counted rather
    /// than inferred, because "12 requested, 5 shipped" is invisible in every
    /// other number the report carries.
    pub fracture_chunks_dropped: usize,
    /// How many `.inf_part` world partitions were built from levels (P16.5).
    pub partitions_built: usize,
    /// Total streamed grid cells across those partitions (the persistent cell is
    /// not counted — it is always resident, so it is not a streaming unit).
    pub partition_cells: usize,
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
        if self.fractures_derived > 0 {
            s.push_str(&format!(
                "  {} fracture chunk set(s) derived (.inf_fracture), {} chunk(s){}\n",
                self.fractures_derived,
                self.fracture_chunks,
                if self.fracture_chunks_dropped > 0 {
                    format!(
                        " ({} requested site(s) yielded nothing)",
                        self.fracture_chunks_dropped
                    )
                } else {
                    String::new()
                }
            ));
        }
        if self.partitions_built > 0 {
            s.push_str(&format!(
                "  {} world partition(s) built (.inf_part), {} streamed cell(s)\n",
                self.partitions_built, self.partition_cells
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

/// A single asset's owned cook input, read serially from the `AssetDb` then
/// handed to the parallel cook stage (so the parallel closure owns its data and
/// never borrows the DB).
enum CookInput {
    /// An unknown-kind asset that is skipped (carries its advisory message).
    Skipped(String),
    Asset {
        guid: AssetId,
        kind: AssetKind,
        /// Authoring name (used to anchor a decode error before the graph names
        /// itself).
        name: String,
        raw: Vec<u8>,
    },
}

/// The result of cooking one asset, folded back into the pack serially in
/// closure order.
enum CookOutput {
    Skipped(String),
    Cooked {
        guid: AssetId,
        kind: AssetKind,
        cooked: Vec<u8>,
        is_level: bool,
        is_blueprint: bool,
        /// A derived meshlet DAG `(vmesh_id, bytes)` for a dense mesh.
        vmesh: Option<(AssetId, Vec<u8>)>,
        /// A derived fracture chunk set `(fracture_id, bytes, chunks, requested)`
        /// for a mesh some `Destructible` names (P22.2).
        fracture: Option<(AssetId, Vec<u8>, usize, u32)>,
        /// A derived world partition `(part_id, bytes, streamed_cell_count)` for
        /// a partitioned level (P16.5).
        partition: Option<(AssetId, Vec<u8>, usize)>,
        /// A derived material record `(derived_id, bytes)` for a `.inf_mat`
        /// (P26.3b). Derived rather than carried because the shipped player
        /// cannot decode a `.inf_mat` at all.
        derived_material: Option<(AssetId, Vec<u8>)>,
        /// Cook advisories raised while cooking this asset (today: cross-cell
        /// references in a partitioned level). Folded into the report in closure
        /// order so it stays deterministic.
        advisories: Vec<String>,
    },
}

/// Cook one asset: decode/validate/re-encode + optional vmesh derivation. Pure
/// over its input (no shared mutable state), so it runs on the job pool and its
/// output is a deterministic function of the input bytes.
fn cook_one(
    input: CookInput,
    opts: &CookOptions,
    fractures: &BTreeMap<AssetId, FractureRequest>,
) -> Result<CookOutput> {
    let (guid, kind, name, raw) = match input {
        CookInput::Skipped(w) => return Ok(CookOutput::Skipped(w)),
        CookInput::Asset {
            guid,
            kind,
            name,
            raw,
        } => (guid, kind, name, raw),
    };

    let mut is_level = false;
    let mut is_blueprint = false;
    let mut partition: Option<(AssetId, Vec<u8>, usize)> = None;
    let mut advisories: Vec<String> = Vec::new();
    let cooked: Vec<u8> = match kind {
        AssetKind::Level => {
            let mut level =
                inf_scene::decode(&raw).map_err(|source| CookError::Scene { guid, source })?;
            is_level = true;
            // P20.1: water flows downhill. A river spline authored the wrong way
            // up a valley is a mistake the cook can SEE and the runtime can only
            // LOOK WRONG about — the `dangling_terrain_refs` shape of advisory.
            //
            // **BEFORE the partition branch, and that ordering is load-bearing.**
            // Partitioning MOVES the entities into the `.inf_part` and clears them
            // here, so anything reading `level.entities` afterwards reads an empty
            // list and reports nothing — silently, and on exactly the levels most
            // likely to hold a kilometre of river. Every future per-entity advisory
            // on this path belongs above the branch for the same reason.
            advisories.extend(uphill_rivers(guid, &level));
            if level.settings.partition.enabled {
                let (bytes, cells, notes) = build_partition(guid, &level)?;
                partition = Some((derived_partition_id(guid), bytes, cells));
                advisories.extend(notes);
                // The entities now live in the `.inf_part`. Shipping them here too
                // would double the level's bytes AND give the runtime two
                // authorities for the same world — so the cooked level keeps only
                // its title and settings, and `partition.enabled` is what tells the
                // player where its entities went.
                level.entities.clear();
            }
            inf_scene::encode(&level).map_err(|source| CookError::Scene { guid, source })?
        }
        AssetKind::Blueprint => {
            let mut class: BlueprintClass =
                serde_json::from_slice(&raw).map_err(|e| CookError::Blueprint {
                    guid,
                    class: name.clone(),
                    handler: "<decode>".into(),
                    message: e.to_string(),
                })?;
            if let Some(issue) = validate_class(&mut class) {
                return Err(CookError::Blueprint {
                    guid,
                    class: class.name,
                    handler: issue.handler,
                    message: issue.message,
                });
            }
            is_blueprint = true;
            raw
        }
        AssetKind::FunctionLib => {
            let mut lib: BlueprintLibrary =
                serde_json::from_slice(&raw).map_err(|e| CookError::Blueprint {
                    guid,
                    class: name.clone(),
                    handler: "<decode>".into(),
                    message: e.to_string(),
                })?;
            if let Some(issue) = validate_library(&mut lib) {
                return Err(CookError::Blueprint {
                    guid,
                    class: lib.name,
                    handler: issue.handler,
                    message: issue.message,
                });
            }
            is_blueprint = true;
            raw
        }
        // A `.inf_terrain` rides through verbatim — the payload IS the shipped
        // layout, and re-encoding it would be a no-op at best — but it is
        // **structurally validated** first (P16.3). The runtime pages tiles by
        // trusting a header + directory it checks once; a truncated, overlapping,
        // misaligned or accidentally bincode-framed asset must break the build
        // here rather than the player later. Header + directory only: this is
        // O(tile_count) and never decodes a tile.
        AssetKind::Terrain => {
            inf_terrain::TerrainAssetReader::new(raw.as_slice()).map_err(|e| {
                CookError::Terrain {
                    guid,
                    message: e.to_string(),
                }
            })?;
            raw
        }
        // A `.inf_voxel` rides through verbatim as well (P21.1) — same reasoning
        // as the terrain above, one dimension up. **Structurally validated**
        // first: header + chunk directory only, so it is O(chunk_count) and never
        // decodes a chunk. It is a streaming-class kind, so it cooks uncompressed
        // and the player sub-slices chunks straight out of the mapping — which
        // means a directory it cannot trust is a cave made of another chunk's
        // bytes, discovered by a player rather than by this line.
        AssetKind::VoxelVolume => {
            inf_voxel::VoxelAssetReader::new(raw.as_slice()).map_err(|e| {
                CookError::VoxelVolume {
                    guid,
                    message: e.to_string(),
                }
            })?;
            raw
        }
        // A `.inf_biomes` rides through verbatim too, but is **decoded** first
        // (P19.2). Unlike a terrain this is cheap — a short list of names — and
        // unlike a terrain it cannot be checked structurally at runtime: the
        // per-sample ids are already in the tiles, so an ambiguous vocabulary
        // surfaces as terrain that resolves to the wrong biome rather than as a
        // load failure. `BiomeSet::migrate` runs the whole validation.
        AssetKind::BiomeSet => {
            inf_asset::decode::<inf_terrain::BiomeSet>(&raw).map_err(|e| CookError::BiomeSet {
                guid,
                message: e.to_string(),
            })?;
            raw
        }
        // Data assets ride through verbatim (already deterministic bincode).
        _ => raw,
    };

    // ── derive the runtime material record for a `.inf_mat` (P26.3b) ───────
    //
    // Through `inf_material::derive_material_bytes` — the ONE door the PIE
    // payload builder and the editor projector also call — so the pack, the
    // preview and the viewport cannot flatten a material three ways. Keyed by
    // `derived_material_id`, the same salted id the payload uses, so a runtime
    // has one lookup rule rather than two.
    //
    // A `.inf_mat` this build cannot decode is an **advisory**, not a failure:
    // the level still ships, and every surface bound to that material renders
    // off its scalar attributes — which is exactly what a pre-v22 level did, and
    // is the permanent no-texture path rather than a hole.
    let derived_material = if kind == AssetKind::Material {
        match inf_material::derive_material_bytes(&cooked) {
            Ok(rec) => match inf_asset::encode(&rec) {
                Ok(bytes) => Some((inf_asset::derived_material_id(guid), bytes)),
                Err(e) => {
                    advisories.push(undecodable_material_advisory(guid, &name, &e.to_string()));
                    None
                }
            },
            Err(e) => {
                advisories.push(undecodable_material_advisory(guid, &name, &e.to_string()));
                None
            }
        }
    } else {
        None
    };

    // ── derive virtualized geometry for dense meshes (for a mesh cooked == raw) ─
    let vmesh = if kind == AssetKind::Mesh && opts.vgeom.enabled {
        let _span = tracing::info_span!("derive_vmesh", %guid).entered();
        match derive_vmesh(guid, &cooked, opts.vgeom.min_triangles)? {
            Ok(bytes) => Some((derived_vmesh_id(guid), bytes)),
            // A mesh with real geometry that the threshold turned away ships as a
            // placeholder cube — say so (P18.3 audit). An empty mesh says nothing:
            // a cube is the honest rendering of no geometry.
            Err(VmeshSkip::BelowThreshold { triangles, min }) => {
                advisories.push(sub_threshold_advisory(guid, &name, triangles, min));
                None
            }
            Err(VmeshSkip::NoGeometry) => None,
        }
    } else {
        None
    };

    // ── derive the fracture chunk set for a destructible mesh (P22.2) ───────
    //
    // Gated on the PLAN, not on the kind: a mesh nothing declares destructible
    // gets nothing, which is what keeps a project's cook from growing an
    // `.inf_fracture` for every prop in it.
    let fracture = if kind == AssetKind::Mesh && opts.fracture.enabled {
        match fractures.get(&guid) {
            Some(req) => {
                let _span = tracing::info_span!("derive_fracture", %guid).entered();
                let (params, clamp_note) = clamped_params(req, &name);
                advisories.extend(clamp_note);
                match derive_fracture(guid, &cooked, params)? {
                    Ok((bytes, chunks, requested)) => {
                        // A handful of dropped sites is normal; a large shortfall
                        // is a break coarser than the level claims, and nothing
                        // else in the pipeline can say so.
                        if (chunks as f64) < requested as f64 * CHUNK_YIELD_FLOOR {
                            advisories.push(chunk_yield_advisory(guid, &name, chunks, requested));
                        }
                        Some((derived_fracture_id(guid), bytes, chunks, requested))
                    }
                    Err(issue) => {
                        advisories.push(fracture_skip_advisory(guid, &name, req.level, issue));
                        None
                    }
                }
            }
            None => None,
        }
    } else {
        None
    };

    Ok(CookOutput::Cooked {
        guid,
        kind,
        cooked,
        is_level,
        is_blueprint,
        vmesh,
        fracture,
        partition,
        derived_material,
        advisories,
    })
}

/// Clamp a request's chunk count, returning the advisory when the clamp bit.
///
/// Pure, so the wording and the trigger are unit-tested — the
/// `sub_threshold_advisory` precedent.
fn clamped_params(req: &FractureRequest, name: &str) -> (FractureParams, Option<String>) {
    let asked = req.params.chunk_count;
    let clamped = fracture::clamp_chunk_count(asked);
    let params = FractureParams {
        chunk_count: clamped,
        ..req.params
    };
    if clamped == asked {
        return (params, None);
    }
    (
        params,
        Some(chunk_count_clamp_advisory(
            req.level, req.entity, name, asked, clamped,
        )),
    )
}

/// The advisory for a `Destructible` whose authored chunk count is out of range.
pub fn chunk_count_clamp_advisory(
    level: AssetId,
    entity: uuid::Uuid,
    mesh_name: &str,
    asked: u32,
    clamped: u32,
) -> String {
    format!(
        "level {level}: entity {entity} asks for {asked} fracture chunks of mesh \"{mesh_name}\", \
         outside the supported range [{min}, {max}] — the cook clamped it to {clamped}, so the \
         shipped break will not have the piece count the level says it does; author a \
         Destructible.chunk_count inside the range",
        min = fracture::MIN_CHUNK_COUNT,
        max = fracture::MAX_CHUNK_COUNT,
    )
}

/// Why the cook did not write a `.inf_fracture` for a mesh a `Destructible`
/// named.
///
/// Wider than [`FractureSkip`] because the cook knows one thing the Ring-0
/// fracture kernel deliberately does not: whether the physics facade can turn a
/// chunk's hull points into a collider.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FractureIssue {
    /// The geometry kernel declined — see [`FractureSkip`].
    Skipped(FractureSkip),
    /// A chunk's hull point set cannot become a `ColliderShape3D::ConvexHull`.
    Uncollidable { chunk: u32, volume_m3: f64 },
}

/// The advisory for a mesh the fracture declined (P22.2).
///
/// **Why this is worth a line of output.** A `Destructible` on a mesh that cannot
/// be fractured is the exact shape of defect the vmesh sub-threshold advisory
/// exists for: the level says the wall breaks, the cook silently ships no chunk
/// set, and P22.3 finds nothing to swap in — so the wall is simply indestructible,
/// in the build only, discovered by whoever shoots it. Unlike the vmesh case
/// there is no `NoGeometry` exemption, because here somebody *asked*: a
/// `Destructible` pointing at an empty mesh is always a mistake worth naming.
///
/// Pure, so the wording and the trigger are unit-tested.
pub fn fracture_skip_advisory(
    guid: AssetId,
    name: &str,
    level: AssetId,
    issue: FractureIssue,
) -> String {
    let skip = match issue {
        FractureIssue::Skipped(s) => s,
        FractureIssue::Uncollidable { chunk, volume_m3 } => {
            return uncollidable_chunk_advisory(guid, name, chunk as usize, volume_m3)
        }
    };
    let why = match skip {
        FractureSkip::NoGeometry => {
            "it has no triangles at all — there is nothing to break".to_string()
        }
        FractureSkip::Degenerate => {
            "its geometry is flat or collinear, so it bounds no volume and its chunks could \
             have no mass"
                .to_string()
        }
        FractureSkip::TooSmall { volume_m3 } => format!(
            "its convex hull measures {volume_m3:.3e} m3, below the {min:.0e} m3 worth \
             fracturing",
            min = fracture::MIN_FRACTURE_VOLUME_M3
        ),
        FractureSkip::TooFewChunks {
            produced,
            requested,
        } => format!(
            "the chunking yielded {produced} piece(s) of the {requested} asked for, which is \
             not a fracture — every site's Voronoi cell missed the solid, which happens when a \
             mesh is a sliver in its own bounding box"
        ),
    };
    format!(
        "mesh {guid} ({name}) is marked Destructible by a level {level} entity but was NOT \
         fractured: {why}. The shipped build will have no chunk set for it, so it cannot \
         break at all; give it real solid geometry, or remove the Destructible"
    )
}

/// The fraction of the requested chunk count below which a fracture is reported
/// as a shortfall.
///
/// A site whose Voronoi cell misses the solid produces no chunk, which is legal
/// and normal — a handful of drops on a lumpy mesh is not worth a line of
/// output. A *large* shortfall is: an author who asked for 24 pieces and ships 5
/// gets a break that reads as a cut, and nothing else in the pipeline can tell
/// them. Three quarters is the line, and the advisory quotes both numbers so the
/// judgement is visible rather than implied.
pub const CHUNK_YIELD_FLOOR: f64 = 0.75;

/// The advisory for a fracture that yielded materially fewer chunks than asked.
pub fn chunk_yield_advisory(guid: AssetId, name: &str, produced: usize, requested: u32) -> String {
    format!(
        "mesh {guid} ({name}) fractured into {produced} chunks of the {requested} its \
         Destructible asked for — sites whose Voronoi cells missed the solid produce nothing, \
         which happens when a mesh is long, thin or diagonal in its own bounding box. The \
         shipped break will be coarser than the level says; lower chunk_count to match, or \
         split the mesh into pieces that fill their bounds"
    )
}

/// The advisory for a chunk the physics facade would refuse to collide.
pub fn uncollidable_chunk_advisory(
    guid: AssetId,
    name: &str,
    chunk: usize,
    volume_m3: f64,
) -> String {
    format!(
        "mesh {guid} ({name}): fracture chunk {chunk} measures {volume_m3:.3e} m3 but its hull \
         point set cannot become a collider — a chunk with no collider falls through the world. \
         No .inf_fracture was written; report this with the mesh, it is an engine defect rather \
         than an authoring one"
    )
}

/// Build the `.inf_fracture` payload for a `.inf_mesh`, or the reason it was
/// skipped.
#[allow(clippy::type_complexity)]
fn derive_fracture(
    guid: AssetId,
    raw: &[u8],
    params: FractureParams,
) -> Result<std::result::Result<(Vec<u8>, usize, u32), FractureIssue>> {
    let mesh: inf_mesh::MeshAsset = inf_asset::decode(raw).map_err(|e| CookError::Mesh {
        guid,
        message: e.to_string(),
    })?;
    let asset = match fracture::fracture_mesh(&mesh, guid, params) {
        Ok(a) => a,
        Err(skip) => return Ok(Err(FractureIssue::Skipped(skip))),
    };
    // **The producer-side collider gate, actually called.**
    //
    // `inf_physics::d3::convex_hull_is_buildable` exists so a producer of hull
    // point sets can check before it writes bytes rather than discovering the
    // refusal in a player — and a gate that is cited but never invoked is worse
    // than no gate at all (the phantom-gate law). This is the invocation: it runs
    // the SAME `parry` path the collider build runs, so it is what stops the cook
    // shipping a chunk P22.3 cannot give a body.
    //
    // **The claim is bounded, and the bound is worth stating.** This checks the
    // chunk's raw `hull_points`; P22.3's `FractureState::chunk_collider` builds
    // its hull from those points *after* the actor's placement affine, so an
    // actor scaled to a millionth could still refuse at spawn where this passed.
    // "A yes here means the cook did not ship an unbuildable chunk set" is true;
    // "a yes here guarantees every runtime collider builds" is not, and the
    // difference is the actor's own transform.
    if let Some((chunk, volume_m3)) = inf_physics::d3::first_uncollidable_chunk(&asset) {
        return Ok(Err(FractureIssue::Uncollidable { chunk, volume_m3 }));
    }
    let chunks = asset.chunks.len();
    let requested = asset.requested_chunks;
    // A plain bincode payload through `inf_asset::encode`, NOT a raw image: a
    // `.inf_fracture` is loaded whole when an asset breaks (it is not
    // streaming-class — see `PackWriter::compresses_kind`), so there is nothing
    // to sub-slice and nothing that needs 16-byte alignment.
    let bytes = inf_asset::encode(&asset).map_err(|e| CookError::Mesh {
        guid,
        message: e.to_string(),
    })?;
    Ok(Ok((bytes, chunks, requested)))
}

/// Which meshes to fracture, and the parameters to fracture them with.
///
/// **Why this runs before the parallel stage.** [`cook_one`] is a pure function
/// of *one asset's* bytes, which is what lets the cook fan out and still be
/// byte-identical. "Is this mesh destructible" is not a fact about the mesh: it
/// lives in a *level*, on some other entity, in some other asset. So the plan is
/// built serially first — one extra decode per level, which is cheap beside the
/// re-encode the level cook already does — and handed to the closure as read-only
/// data.
///
/// Returns advisories for the two hazards visible here: a `Destructible` with no
/// mesh to break, and two `Destructible`s asking for different fractures of the
/// same mesh (only one `.inf_fracture` can exist per mesh, because its id is a
/// function of the mesh's). Deduplicated + sorted, like every other advisory
/// here, so the report stays deterministic.
fn plan_fractures(inputs: &[CookInput]) -> (BTreeMap<AssetId, FractureRequest>, Vec<String>) {
    let mut plan: BTreeMap<AssetId, FractureRequest> = BTreeMap::new();
    let mut notes: BTreeSet<String> = BTreeSet::new();
    for input in inputs {
        let CookInput::Asset {
            guid, kind, raw, ..
        } = input
        else {
            continue;
        };
        if *kind != AssetKind::Level {
            continue;
        }
        // An undecodable level is not this pass's problem to report: the real
        // cook stage fails the build on it with a proper error.
        let Ok(level) = inf_scene::decode(raw) else {
            continue;
        };
        for e in &level.entities {
            let Some(d) = e.destructible else {
                continue;
            };
            let Some(mesh_id) = e.mesh.as_ref().and_then(|m| m.asset).map(AssetId) else {
                notes.insert(format!(
                    "level {guid}: entity {} (\"{}\") is Destructible but references no mesh \
                     asset — a fracture is derived from the entity's own mesh, so there is \
                     nothing to break; give it a MeshRef pointing at an .inf_mesh, or remove \
                     the Destructible",
                    e.guid, e.name
                ));
                continue;
            };
            let req = FractureRequest {
                params: FractureParams {
                    seed: d.fracture_seed,
                    chunk_count: d.chunk_count,
                },
                entity: e.guid,
                level: *guid,
            };
            match plan.get(&mesh_id) {
                Some(first) if first.params != req.params => {
                    notes.insert(format!(
                        "mesh {mesh_id} is marked Destructible with two different fractures \
                         (seed {a_seed}/{a_count} chunks by entity {a_ent}, seed \
                         {b_seed}/{b_count} by entity {b_ent}) — exactly ONE .inf_fracture is \
                         derived per mesh, because its id is a function of the mesh's, so the \
                         first in cook order wins and the other entity will break the wrong \
                         way; give the variants matching parameters, or separate meshes",
                        a_seed = first.params.seed,
                        a_count = first.params.chunk_count,
                        a_ent = first.entity,
                        b_seed = req.params.seed,
                        b_count = req.params.chunk_count,
                        b_ent = req.entity,
                    ));
                }
                Some(_) => {}
                None => {
                    plan.insert(mesh_id, req);
                }
            }
        }
    }
    (plan, notes.into_iter().collect())
}

/// Bin a partitioned level's entities and build its `.inf_part` payload.
///
/// Returns the payload bytes, the number of **streamed** grid cells (the
/// persistent cell is not a streaming unit), and the cross-cell-reference
/// advisories.
///
/// Pure over `level`, like every other `cook_one` stage, so it runs on the job
/// pool and two cooks of one level are byte-identical.
fn build_partition(
    guid: AssetId,
    level: &inf_scene::RuntimeLevel,
) -> Result<(Vec<u8>, usize, Vec<String>)> {
    use inf_scene::partition;

    let settings = &level.settings.partition;
    let plan = partition::partition_entities(&level.entities, settings);
    let asset =
        partition::build_partition_asset(&plan, settings).map_err(|e| CookError::Partition {
            guid,
            message: e.to_string(),
        })?;

    // ── advisories ──
    //
    // Both of these are hazards the cook can *see* and the runtime can only
    // *suffer*, and neither is fixed up automatically: a fixup would make cell
    // residency depend on the reference graph / on which entities carry a script,
    // and a level's memory ceiling would then move every time gameplay was added.
    // So they are named, with the remedy, where they are cheap to fix — the
    // `dangling_terrain_refs` precedent.
    let mut advisories: Vec<String> = Vec::new();

    // (1) A cook-time reference from one cell to another is legal to author and
    //     legal to cook, but at runtime the target may simply not be spawned when
    //     the referrer is.
    advisories.extend(partition::cross_cell_refs(&plan).into_iter().map(|r| {
        format!(
            "level {guid}: entity {} in {} references entity {} in {} through `{}` — the \
             target may not be resident when the referrer is; mark it AlwaysLoaded, or put \
             both under one parent so they stream together",
            r.from, r.from_cell, r.to, r.to_cell, r.field
        )
    }));

    // (2) A Blueprint on a STREAMED entity never ticks: the runtime assigns actor
    //     ids once, at `RuntimeSim` construction, from the persistent cell. The
    //     mesh + `ActorClass` shape (an enemy, a door, a pickup) is the commonest
    //     gameplay actor there is and bins into a cell every time, so without this
    //     the only symptom is a level full of statues.
    advisories.extend(partition::streamed_actors(&plan).into_iter().map(|a| {
        format!(
            "level {guid}: entity {} in {} runs blueprint {} but is STREAMED — a blueprint \
             only ticks on an entity present when the sim is built, so this one will never \
             run; mark it AlwaysLoaded (or give it a StreamingSource) to put it in the \
             persistent cell",
            a.entity, a.cell, a.class
        )
    }));

    // (3) A TERRAIN on a streamed entity takes the ground with it. A heightfield
    //     spans kilometres from its entity origin, but the partitioner bins it by
    //     that origin, so it lands in one cell — and when the player walks out of
    //     that cell the floor of the world despawns. Enabling partitioning on a
    //     level that already has terrain produces this every time, with no symptom
    //     until somebody walks far enough.
    advisories.extend(partition::streamed_terrains(&plan).into_iter().map(|t| {
        let source = match t.asset {
            Some(a) => format!("streamed from .inf_terrain {a}"),
            None => "inline".to_string(),
        };
        format!(
            "level {guid}: terrain entity {} ({source}) landed in {} and is STREAMED — a \
             heightfield spans far more world than the cell holding its origin, so the ground \
             will DESPAWN once a streaming source leaves that cell; mark the terrain \
             AlwaysLoaded to keep it in the persistent cell",
            t.entity, t.cell
        )
    }));

    Ok((asset.into_bytes(), plan.cells.len(), advisories))
}

/// Cook `project_root` into `out_dir`, producing a pack + manifest.
pub fn cook(project_root: &Path, out_dir: &Path, opts: &CookOptions) -> Result<CookReport> {
    let _span = tracing::info_span!("cook").entered();
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
    // A level may name a terrain asset the project no longer has. The closure
    // simply cannot follow that edge, which would ship a level whose ground never
    // streams — silently. Say so.
    warnings.extend(dangling_terrain_refs(&db, &closure));
    warnings.extend(dangling_material_refs(&db, &closure));
    warnings.extend(voxel_scale_mismatches(&db, &closure));
    warnings.extend(see_through_pits(&db, &closure));
    warnings.extend(dangling_biome_set_refs(&db, &closure));
    warnings.extend(unresolvable_image_masks(&db, &closure));
    warnings.extend(dangling_grammar_modules(&db, &closure));
    warnings.extend(unfracturable_destructible_meshes(&db, &closure));
    warnings.extend(unbaked_destructible_populations(&db, &closure));

    // ── 4/5/6/7. blueprints + levels + vmesh + fracture ─────────────────────
    //
    // The per-asset CPU work (scene decode/re-encode, blueprint decode+validate,
    // meshlet-DAG derivation) is a **pure function of each asset's input bytes**,
    // and [`PackWriter`] stores into a GUID-keyed `BTreeMap` (sorting on write),
    // so we can fan the work across the Ring-0 job pool and then fold the results
    // back **serially, in closure order**, and still get a byte-identical pack
    // (the P9.2 determinism gate). We read each asset's bytes serially first (I/O
    // bound + needs the `AssetDb`), then hand owned inputs to the parallel stage,
    // then fold — `?`-ing on the first error in closure order preserves the
    // fail-fast, handler-anchored first-broken-blueprint contract.
    let inputs: Vec<CookInput> = {
        let _span = tracing::info_span!("cook_read", assets = closure.len()).entered();
        let mut inputs = Vec::with_capacity(closure.len());
        for guid in &closure {
            let entry = db.get(*guid).ok_or(CookError::UnknownRoot(*guid))?;
            let kind = entry.kind();
            if kind == AssetKind::Unknown {
                inputs.push(CookInput::Skipped(format!(
                    "skipped unknown-kind asset at {}",
                    entry.path.display()
                )));
                continue;
            }
            let raw = std::fs::read(&entry.path)?;
            inputs.push(CookInput::Asset {
                guid: *guid,
                kind,
                name: entry.name.clone(),
                raw,
            });
        }
        inputs
    };

    // Which meshes are destructible is a cross-asset fact (it lives on level
    // entities), so it is resolved serially here and handed to the pure per-asset
    // closure as read-only data — see `plan_fractures`.
    let (fracture_plan, fracture_notes) = if opts.fracture.enabled {
        let _span = tracing::info_span!("plan_fractures").entered();
        plan_fractures(&inputs)
    } else {
        (BTreeMap::new(), Vec::new())
    };
    warnings.extend(fracture_notes);

    // Parallel, deterministic (in-order) map: cook every asset on the job pool.
    let outputs: Vec<Result<CookOutput>> = {
        let _span = tracing::info_span!("cook_assets", assets = inputs.len()).entered();
        inf_core::parallel_map(inputs, |input| cook_one(input, opts, &fracture_plan))
    };

    // Serial fold in closure order → byte-identical pack + fail-fast first error.
    let mut writer = PackWriter::new();
    let mut kinds: BTreeMap<String, usize> = BTreeMap::new();
    let mut levels: Vec<AssetId> = Vec::new();
    let mut blueprints_validated = 0usize;
    let mut levels_rewritten = 0usize;
    let mut meshlet_meshes_derived = 0usize;
    let mut materials_derived = 0usize;
    let mut fractures_derived = 0usize;
    let mut fracture_chunks = 0usize;
    let mut fracture_chunks_dropped = 0usize;
    let mut partitions_built = 0usize;
    let mut partition_cells = 0usize;
    // Meshlet DAGs derived alongside their meshes, added after the closure so a
    // derived id never shadows a real closure asset processed later.
    let mut derived_vmeshes: Vec<(AssetId, Vec<u8>)> = Vec::new();
    let mut derived_materials: Vec<(AssetId, Vec<u8>)> = Vec::new();
    // Fracture chunk sets derived alongside their meshes, added after the closure
    // for the same reason.
    let mut derived_fractures: Vec<(AssetId, Vec<u8>, usize, u32)> = Vec::new();
    // World partitions derived alongside their levels, added after the closure for
    // exactly the same reason.
    let mut derived_partitions: Vec<(AssetId, Vec<u8>, usize)> = Vec::new();

    for output in outputs {
        match output? {
            CookOutput::Skipped(warning) => warnings.push(warning),
            CookOutput::Cooked {
                guid,
                kind,
                cooked,
                is_level,
                is_blueprint,
                vmesh,
                fracture,
                partition,
                derived_material,
                advisories,
            } => {
                writer.add_bytes(guid, kind, &cooked)?;
                *kinds.entry(kind.slug().to_string()).or_default() += 1;
                if is_level {
                    levels.push(guid);
                    levels_rewritten += 1;
                }
                if is_blueprint {
                    blueprints_validated += 1;
                }
                if let Some((vmesh_id, vmesh_bytes)) = vmesh {
                    // Guard the (astronomically unlikely) collision with a real
                    // asset or an already-derived vmesh; skip with a warning
                    // rather than corrupt the pack.
                    if db.contains(vmesh_id)
                        || derived_vmeshes.iter().any(|(id, _)| *id == vmesh_id)
                    {
                        warnings.push(format!(
                            "skipped vmesh for mesh {guid}: derived id {vmesh_id} collides"
                        ));
                    } else {
                        derived_vmeshes.push((vmesh_id, vmesh_bytes));
                    }
                }
                if let Some((frac_id, frac_bytes, chunks, requested)) = fracture {
                    // Same collision guard as the vmesh above, and the same
                    // degradation: a fracture that cannot be packed means the
                    // asset does not break, which is a warning rather than a
                    // reason to fail an otherwise-valid build.
                    if db.contains(frac_id)
                        || derived_fractures.iter().any(|(id, _, _, _)| *id == frac_id)
                    {
                        warnings.push(format!(
                            "skipped fracture for mesh {guid}: derived id {frac_id} collides"
                        ));
                    } else {
                        derived_fractures.push((frac_id, frac_bytes, chunks, requested));
                    }
                }
                // P26.3b: the derived material record. Same collision guard as
                // the vmesh above, and the same degradation — a record that
                // cannot be packed means every surface bound to that material
                // renders off its scalars, which is a warning rather than a
                // reason to fail an otherwise-valid build.
                if let Some((mat_id, mat_bytes)) = derived_material {
                    if db.contains(mat_id) || derived_materials.iter().any(|(id, _)| *id == mat_id)
                    {
                        warnings.push(format!(
                            "skipped derived material for {guid}: derived id {mat_id} collides"
                        ));
                    } else {
                        derived_materials.push((mat_id, mat_bytes));
                    }
                }
                warnings.extend(advisories);
                if let Some((part_id, part_bytes, cells)) = partition {
                    // A collision here would make the partition unreachable (the
                    // runtime looks it up by the derived id), and a partitioned
                    // level ships no entities of its own — so unlike the vmesh
                    // case, degrading to "skip it" would ship an empty world.
                    // Fail the build.
                    if db.contains(part_id)
                        || derived_partitions.iter().any(|(id, _, _)| *id == part_id)
                    {
                        return Err(CookError::Partition {
                            guid,
                            message: format!("derived partition id {part_id} collides"),
                        });
                    }
                    derived_partitions.push((part_id, part_bytes, cells));
                }
            }
        }
    }

    // Pack the derived material records (P26.3b). `AssetKind::DerivedMaterial` is
    // NOT streaming-class, so these compress like any other payload — a record is
    // a handful of GUIDs and scalars, read whole when a level loads.
    for (mat_id, bytes) in &derived_materials {
        writer.add_bytes(*mat_id, AssetKind::DerivedMaterial, bytes)?;
        *kinds
            .entry(AssetKind::DerivedMaterial.slug().to_string())
            .or_default() += 1;
        materials_derived += 1;
    }

    // Pack the derived meshlet DAGs (order-independent — the writer sorts by GUID).
    for (vmesh_id, bytes) in &derived_vmeshes {
        writer.add_bytes(*vmesh_id, AssetKind::MeshletMesh, bytes)?;
        *kinds
            .entry(AssetKind::MeshletMesh.slug().to_string())
            .or_default() += 1;
        meshlet_meshes_derived += 1;
    }

    // Pack the derived fracture chunk sets. `AssetKind::Fracture` is NOT
    // streaming-class, so these compress like any other payload — a chunk set is
    // read whole, at the instant its mesh breaks.
    for (frac_id, bytes, chunks, requested) in &derived_fractures {
        writer.add_bytes(*frac_id, AssetKind::Fracture, bytes)?;
        *kinds
            .entry(AssetKind::Fracture.slug().to_string())
            .or_default() += 1;
        fractures_derived += 1;
        fracture_chunks += chunks;
        fracture_chunks_dropped += (*requested as usize).saturating_sub(*chunks);
    }

    // Pack the derived world partitions. `AssetKind::Partition` is
    // streaming-class, so `PackWriter` stores these **uncompressed** and the
    // runtime slices one cell out of the mapping with no decode of the rest.
    for (part_id, bytes, cells) in &derived_partitions {
        writer.add_bytes(*part_id, AssetKind::Partition, bytes)?;
        *kinds
            .entry(AssetKind::Partition.slug().to_string())
            .or_default() += 1;
        partitions_built += 1;
        partition_cells += cells;
    }

    // ── 8. write pack + manifest ────────────────────────────────────────────
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
        materials_derived,
        fractures_derived,
        fracture_chunks,
        fracture_chunks_dropped,
        partitions_built,
        partition_cells,
        warnings,
    })
}

/// The advisory for a `.inf_mat` the cook could not flatten into its runtime
/// record (P26.3b).
///
/// **Why this is worth a line of output.** A shipped player cannot decode a
/// `.inf_mat` — that is the whole reason the derived record exists — so a
/// material that fails here ships as *nothing*, and every surface bound to it
/// renders off its per-instance scalar attributes. That is the same picture a
/// pre-v22 level draws, which is exactly what makes it invisible: the build
/// succeeds, the level loads, the geometry is right, and one prop is a flat
/// colour that nobody can tell from an authored flat colour. The editor viewport
/// meanwhile reads the project's content root directly and can still show it
/// textured, so the divergence is visible only to whoever compares the two.
///
/// Pure, so the wording and the trigger are unit-tested — the
/// `sub_threshold_advisory` / `dangling_terrain_refs` precedent.
pub fn undecodable_material_advisory(guid: AssetId, name: &str, detail: &str) -> String {
    format!(
        "material {guid} ({name}) did not decode ({detail}), so no .inf_matd was derived \
         and every surface bound to it renders off its SCALAR attributes in the shipped \
         build — textureless, and indistinguishable from an authored flat colour. \
         Re-import or re-save the material."
    )
}

/// The advisory for a `Material.asset` binding that names an asset the project
/// does not have (P26.3b) — the `dangling_terrain_refs` shape, for materials.
///
/// Raised from the closure rather than from `cook_one`, because a dangling ref is
/// a fact about the *level* and its database, and `cook_one` sees one asset's
/// bytes.
pub fn dangling_material_advisory(level: AssetId, entity: &str, missing: AssetId) -> String {
    format!(
        "level {level}: entity '{entity}' is bound to material {missing}, which is not in \
         the project — the surface ships with its scalar attributes only. Re-apply a \
         material, or clear the binding."
    )
}

/// Why a `.inf_mesh` did not get a derived `.inf_vmesh`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmeshSkip {
    /// No geometry at all (no triangles / fewer than 3 indices) — nothing to
    /// virtualize, and nothing to say about it.
    NoGeometry,
    /// Real geometry, but below [`VgeomCookOptions::min_triangles`].
    BelowThreshold { triangles: usize, min: usize },
}

/// The advisory for a mesh the cook declined to virtualize because it is small
/// (P18.3 audit).
///
/// **Why this is worth a line of output.** `RenderScene` has exactly one door for
/// real (non-primitive) geometry — virtualized geometry — so a mesh with no
/// `.inf_vmesh` renders as a **placeholder cube**, in the editor and in the
/// shipped build alike. Since P18.3 the editor derives from one triangle, so a
/// 500-triangle prop looks correct while it is being authored and ships as a cube.
/// That is the worst shape a defect can have: invisible until the build, and
/// invisible *in* the build until someone walks up to it. The threshold itself is
/// a defensible cost decision; leaving it silent is not.
///
/// Pure, so the wording and the trigger are unit-tested — the
/// `partition::streamed_actors` / `streamed_terrains` precedent.
pub fn sub_threshold_advisory(guid: AssetId, name: &str, triangles: usize, min: usize) -> String {
    format!(
        "mesh {guid} ({name}) has {triangles} triangles, below the virtualized-geometry          threshold of {min}, so no .inf_vmesh was derived — the shipped build renders it as a          PLACEHOLDER CUBE (the editor derives from one triangle, so it looks correct while you          author it). Lower [vgeom] min_triangles for this build, or merge the mesh into a          denser one."
    )
}

/// Classify a mesh the derivation declined, for [`sub_threshold_advisory`].
///
/// Pure over the decoded mesh so the rule — *real geometry, but under the bar* —
/// is testable without a cook.
fn classify_skip(triangles: usize, indices: usize, min: usize) -> VmeshSkip {
    if triangles == 0 || indices < 3 {
        VmeshSkip::NoGeometry
    } else {
        VmeshSkip::BelowThreshold {
            triangles,
            min: min.max(1),
        }
    }
}

/// Build the `.inf_vmesh` payload for a `.inf_mesh`, or the reason it was skipped
/// (below `min_triangles` — it stays on the classic path — or no geometry).
fn derive_vmesh(
    guid: AssetId,
    raw: &[u8],
    min_triangles: usize,
) -> Result<std::result::Result<Vec<u8>, VmeshSkip>> {
    let mesh: inf_mesh::MeshAsset = inf_asset::decode(raw).map_err(|e| CookError::Mesh {
        guid,
        message: e.to_string(),
    })?;
    let (positions, normals, uvs, indices) = mesh.vgeom_streams();
    if mesh.triangle_count() < min_triangles.max(1) || indices.len() < 3 {
        return Ok(Err(classify_skip(
            mesh.triangle_count(),
            indices.len(),
            min_triangles,
        )));
    }
    let vgeom = inf_vgeom::build_vgeom(
        &positions,
        &normals,
        &uvs,
        &indices,
        inf_vgeom::BuildParams::default(),
    );
    // P18.2: the packed payload is the **v2 paged image**, not `inf_asset::encode`
    // output. A bincode length prefix would shift every section off its 16-byte
    // boundary and defeat the whole layout — the same rule, and the same reason,
    // as `.inf_terrain`'s raw image (see `inf_vgeom::asset`). The image is a pure
    // function of the DAG, so the cook stays byte-identical run to run.
    let bytes = inf_vgeom::build_vgeom_asset(&vgeom)
        .map_err(|e| CookError::Mesh {
            guid,
            message: e.to_string(),
        })?
        .into_bytes();
    Ok(Ok(bytes))
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
///   explicit-roots cook of just a level ships its referenced anim assets), and
///   the v9 `Terrain.asset` streaming-terrain ref (P16.3).
/// * **StateMachine** (`.inf_sm`) — the clip GUIDs its states/blend-spaces play
///   (`Motion::Clip` + blend entries) **and** its skeleton ref. This closes the
///   `state-machine → clip` edge so the clips a machine plays ship in the pack.
/// * **AnimClip** (`.inf_anim`) — the skeleton GUID it was authored against.
/// * **BiomeSet** (`.inf_biomes`) — the `.inf_pcg` graphs its biomes scatter with
///   (P19.2's `pcg_graph`, the P19.3 hook). This closes the
///   `level → biome set → graph` chain, so an explicit-roots cook of just a level
///   ships the graphs its painted biomes will evaluate.
/// * **Pcg** (`.inf_pcg`) — the `.inf_mesh` assets its **grammar modules** place
///   (P19.4), closing `level → PcgVolume.graph → module mesh`. Grammar only; see
///   the arm for why a scatter kind's mesh is deliberately still not an edge.
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
                // P13.4: a MeshRef.asset pulls its `.inf_mesh` into the closure, so
                // the cook packs the mesh AND (for a dense mesh) derives + ships its
                // `.inf_vmesh` meshlet DAG beside it (the virtualized-geometry path).
                deps.extend(e.mesh.as_ref().and_then(|m| m.asset).map(AssetId));
                deps.extend(e.pcg_volume.as_ref().and_then(|v| v.graph).map(AssetId));
                // P16.3: a Terrain.asset pulls its `.inf_terrain` into the closure
                // so the cook packs the streaming tiles + LOD pyramid beside the
                // level. The entry is stored **uncompressed** (streaming-class,
                // `PackWriter::compresses_kind`) so the runtime pages individual
                // tiles out of the mapping.
                deps.extend(e.terrain.as_ref().and_then(|t| t.asset).map(AssetId));
                // P19.2: a Terrain.biome_set pulls its `.inf_biomes` into the
                // closure. The per-sample biome ids ride inside the tiles (they
                // need no edge of their own); this edge is what ships the
                // *vocabulary* those ids name — without it a cooked level's
                // terrain would carry ids nothing can resolve, and P19.3's
                // per-biome PCG dispatch would find no graphs at all.
                deps.extend(e.terrain.as_ref().and_then(|t| t.biome_set).map(AssetId));
                // P21.1: a VoxelVolume.asset pulls its `.inf_voxel` into the
                // closure — the caves, tunnels and excavations that locally
                // extend the heightfield. Streaming-class like the terrain, so
                // the entry is stored **uncompressed** and the runtime
                // sub-slices chunks straight out of the mapping. Without this
                // edge a shipped level would draw the ground with its holes and
                // none of what is behind them.
                deps.extend(e.voxel_volume.as_ref().and_then(|v| v.asset).map(AssetId));
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
                // P24.4: a ClothSim pulls its `.inf_cloth` garment. Without this
                // edge a shipped character wears a component nothing can resolve
                // and simulates no cloth at all — while the editor's Simulate,
                // which reads the project's content root directly, folds the coat
                // perfectly. That asymmetry is exactly the shape the P22 "one
                // door for three paths" law was written about, so the edge lands
                // in the same pass as every other component reference.
                deps.extend(e.cloth_sim.as_ref().and_then(|c| c.asset).map(AssetId));
                // P24.4: and a HairGuides pulls its `.inf_hair`, same edge, same
                // reason.
                deps.extend(e.hair_guides.as_ref().and_then(|h| h.asset).map(AssetId));
                // P26.3b (scene v22): a `Material.asset` pulls its `.inf_mat`
                // into the closure — and the `AssetKind::Material` arm below
                // pulls that material's `.inf_tex` maps in after it. Without
                // this edge a shipped level would name a material nothing can
                // resolve and every surface would render off its scalar
                // attributes, silently, while the editor — reading the project's
                // content root directly — showed it textured. That asymmetry is
                // the shape the P22 "one door for three paths" law is about, so
                // the edge lands in the same pass as every other component
                // reference.
                deps.extend(e.material.as_ref().and_then(|m| m.asset).map(AssetId));
            }
            deps
        }
        // P26.3b: a `.inf_mat` references the `.inf_tex` maps it samples, and it
        // is the ONLY thing that does — a level names the material, the material
        // names the textures. The edge is read through the same
        // `derive_material_bytes` door the derived record is written with, so the
        // set the cook PACKS and the set the runtime ASKS FOR cannot differ.
        AssetKind::Material => match inf_material::derive_material_bytes(&raw) {
            Ok(d) => d.texture_dependencies(),
            Err(_) => Vec::new(),
        },
        AssetKind::StateMachine => {
            let Ok(sm) = inf_asset::decode::<inf_anim::StateMachineAsset>(&raw) else {
                return Vec::new();
            };
            let mut deps: Vec<AssetId> = Vec::new();
            deps.extend(sm.skeleton.map(|b| AssetId(uuid::Uuid::from_bytes(b))));
            // P24.1: through the Ring-0 walk, not a local copy. The PIE payload
            // builder closes the same edge from the same function, because this
            // one closed it and that one did not — a divergence with no symptom
            // until the machine started driving the drawn pose.
            deps.extend(
                sm.machine
                    .clip_refs()
                    .into_iter()
                    .map(|c| AssetId(uuid::Uuid::from_bytes(c))),
            );
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
        // P19.2: a `.inf_biomes` references the `.inf_pcg` graph each of its
        // biomes scatters with. It was empty when this arm was written — nothing
        // populated `pcg_graph` until P19.3 bound them, and it does now — which is
        // exactly why the edge was derived here and then rather than waiting for
        // the first graph to appear: every other referencing
        // kind re-derives its edges from the payload, and a `biome_set` whose
        // graphs the cook could not reach would ship a level whose biomes
        // evaluate to nothing, silently, the same failure `Terrain.biome_set`
        // itself has an advisory for.
        AssetKind::BiomeSet => {
            let Ok(set) = inf_asset::decode::<inf_terrain::BiomeSet>(&raw) else {
                return Vec::new();
            };
            set.dependencies()
        }
        // P19.4: a `.inf_pcg` references the `.inf_mesh` assets its **grammar
        // modules** place. Without this edge a cooked level's fence would expand
        // to a full derivation naming meshes the pack does not contain — a hole
        // in a building, discovered by a player.
        //
        // **The edge is grammar-only, deliberately.** A scatter rule's
        // `PcgKind.mesh` is an older hole with a different shape: it has been a
        // blank-tolerant palette slot since P10.5, a document can carry
        // thousands of them, and the renderer still draws every scattered
        // instance as a placeholder cube — so closing it would change what every
        // existing project packs, for bytes nothing currently reads. A grammar
        // module is named in authored text and is the only thing that makes a
        // wall a wall. The scatter half is a stated remainder, not an oversight.
        AssetKind::Pcg => grammar_module_refs(&raw).into_iter().map(AssetId).collect(),
        _ => Vec::new(),
    }
}

/// The mesh GUIDs every grammar module in a `.inf_pcg` payload names, sorted and
/// deduplicated (P19.4).
///
/// Reads the payload's **authored graph** — the source of truth every evaluation
/// site re-lowers — and asks it for the palettes its `grammar.rules` nodes
/// declare. A document-only (v1) payload carries no graph and so contributes
/// nothing.
///
/// **Deliberately NOT driven off the lowered passes.** Lowering has five ways to
/// give up before a pass exists, and every one of them is an ordinary
/// mid-authoring state (a Span pin not yet dragged, most obviously). Taking the
/// edge from the lowered passes would let a graph that plainly declares its
/// meshes ship without them, and without the advisory that would have said so.
/// [`inf_pcg::grammar_mesh_refs`] carries the full argument and the
/// over-inclusiveness it trades for.
fn grammar_module_refs(raw: &[u8]) -> Vec<uuid::Uuid> {
    let Ok(payload) = inf_pcg::PcgAssetPayload::decode(raw) else {
        return Vec::new();
    };
    let Some(graph) = payload.graph() else {
        return Vec::new();
    };
    inf_pcg::grammar_mesh_refs(&graph, &inf_pcg::pcg_registry())
}
/// Every `Material.asset` binding in the closure's levels that names an asset the
/// project does not have (P26.3b) — the [`dangling_terrain_refs`] shape, for
/// materials.
///
/// A dangling binding is not a build failure: the surface ships with its scalar
/// attributes, which is the permanent no-texture path and exactly what a pre-v22
/// level did. It IS a level that quietly stopped being textured, which is the
/// class of hazard the advisory doctrine exists for. Deduplicated + sorted, like
/// every other advisory here, so the report stays deterministic.
fn dangling_material_refs(db: &AssetDb, closure: &[AssetId]) -> Vec<String> {
    let mut missing: BTreeSet<(AssetId, String, AssetId)> = BTreeSet::new();
    for &id in closure {
        let Some(entry) = db.get(id) else { continue };
        if entry.kind() != AssetKind::Level {
            continue;
        }
        let Ok(raw) = std::fs::read(&entry.path) else {
            continue;
        };
        let Ok(level) = inf_scene::decode(&raw) else {
            continue;
        };
        for e in &level.entities {
            if let Some(asset) = e.material.as_ref().and_then(|m| m.asset) {
                if !db.contains(AssetId(asset)) {
                    missing.insert((id, e.name.clone(), AssetId(asset)));
                }
            }
        }
    }
    missing
        .into_iter()
        .map(|(level, name, asset)| dangling_material_advisory(level, &name, asset))
        .collect()
}

/// Advisory: `Terrain.asset` references, in the levels being cooked, that name an
/// asset the project database does not have (P16.3).
///
/// [`dependency_closure`] can only follow edges it can resolve, so a dangling
/// terrain ref is skipped — which would otherwise ship a level whose terrain never
/// streams with no sign anything was wrong. Non-fatal (the level is still valid;
/// the inline `data` remains authoritative when the asset is absent), so it is a
/// warning rather than a [`CookError`], and it is deduplicated + sorted so the
/// report stays deterministic.
fn dangling_terrain_refs(db: &AssetDb, closure: &[AssetId]) -> Vec<String> {
    // `(level, asset, is_voxel)` — ONE set, so the report stays deduplicated and
    // sorted across both kinds rather than concatenating two orderings.
    let mut missing: BTreeSet<(AssetId, AssetId, bool)> = BTreeSet::new();
    for &id in closure {
        let Some(entry) = db.get(id) else { continue };
        if entry.kind() != AssetKind::Level {
            continue;
        }
        let Ok(raw) = std::fs::read(&entry.path) else {
            continue;
        };
        let Ok(level) = inf_scene::decode(&raw) else {
            continue;
        };
        for e in &level.entities {
            if let Some(asset) = e.terrain.as_ref().and_then(|t| t.asset) {
                if !db.contains(AssetId(asset)) {
                    missing.insert((id, AssetId(asset), false));
                }
            }
            // P21.1: the same advisory for a `VoxelVolume.asset`, for the same
            // reason. A dangling reference is not a build failure — the level
            // still loads and plays — but it IS a level whose caves silently do
            // not exist, which is precisely the class of hazard an advisory is
            // for, and the class this cook has been burned by before.
            if let Some(asset) = e.voxel_volume.as_ref().and_then(|v| v.asset) {
                if !db.contains(AssetId(asset)) {
                    missing.insert((id, AssetId(asset), true));
                }
            }
        }
    }
    missing
        .into_iter()
        .map(|(level, asset, is_voxel)| {
            let (kind, pages) = if is_voxel {
                ("voxel", "chunks")
            } else {
                ("terrain", "tiles")
            };
            format!("level {level} references missing {kind} asset {asset}; its {pages} will not stream")
        })
        .collect()
}

/// **The seventh advisory of this shape (P22.2): a `Destructible` whose mesh
/// never reaches the cook.**
///
/// `Destructible` names no asset of its own — the fracture is derived from the
/// entity's own `MeshRef` — which is what makes the component cheap and what
/// makes this failure invisible. There are three ways for the mesh to be absent
/// and **all of them cook clean**: the GUID resolves to nothing (a deleted
/// asset), it resolves to something that is not a mesh, or it is simply outside
/// this build's closure (an explicit `--roots` cook). In every case the level
/// ships, the pack verifies, the player boots, and a wall the level says is
/// destructible cannot break — with `warnings` empty.
///
/// So it is said, in the same place and the same shape as the six advisories
/// above it. Deduplicated + sorted (one `BTreeSet`), like every other advisory
/// here, so the report stays deterministic.
fn unfracturable_destructible_meshes(db: &AssetDb, closure: &[AssetId]) -> Vec<String> {
    /// Why the mesh is unreachable — carried so the advisory can name the fix
    /// rather than the symptom.
    #[derive(PartialEq, Eq, PartialOrd, Ord)]
    enum Why {
        Missing,
        NotAMesh,
    }
    let mut bad: BTreeSet<(AssetId, uuid::Uuid, AssetId, Why)> = BTreeSet::new();
    for &id in closure {
        let Some(entry) = db.get(id) else { continue };
        if entry.kind() != AssetKind::Level {
            continue;
        }
        let Ok(raw) = std::fs::read(&entry.path) else {
            continue;
        };
        let Ok(level) = inf_scene::decode(&raw) else {
            continue;
        };
        for e in &level.entities {
            if e.destructible.is_none() {
                continue;
            }
            // "No MeshRef at all" is `plan_fractures`' advisory, not this one —
            // saying it twice is the noise the advisory law forbids.
            let Some(mesh) = e.mesh.as_ref().and_then(|m| m.asset).map(AssetId) else {
                continue;
            };
            let why = match db.get(mesh) {
                None => Why::Missing,
                Some(m) if m.kind() != AssetKind::Mesh => Why::NotAMesh,
                Some(_) => continue,
            };
            bad.insert((id, e.guid, mesh, why));
        }
    }
    bad.into_iter()
        .map(|(level, entity, mesh, why)| {
            let (what, fix) = match why {
                Why::Missing => (
                    format!("names mesh asset {mesh}, which this project does not have"),
                    "re-import the mesh, or clear the entity's MeshRef",
                ),
                Why::NotAMesh => (
                    format!("names asset {mesh}, which is not an .inf_mesh"),
                    "point the entity's MeshRef at a real mesh",
                ),
            };
            format!(
                "level {level}: entity {entity} is Destructible and {what} — no .inf_fracture \
                 can be derived, so the shipped build has no chunk set for it and the entity \
                 CANNOT BREAK, silently; {fix}"
            )
        })
        .collect()
}

/// The sentence one unbaked destructible population gets.
///
/// A pure function so the wording is unit-testable, like every other advisory in
/// this file, and it names a fix **the author can actually perform**. The first
/// version prescribed `inf_editor_core::bake`, which is a library call with no
/// editor command behind it — advice nobody reading a cook report can take, and
/// the advisory law's own requirement failed on a technicality of wording.
pub fn unbaked_population_advisory(
    level: AssetId,
    entity: uuid::Uuid,
    mesh: Option<AssetId>,
) -> String {
    let names = match mesh {
        Some(m) => format!("its MeshRef names mesh {m}, and ONLY that mesh will fracture"),
        None => "its MeshRef names no mesh asset at all, so nothing will fracture".to_string(),
    };
    format!(
        "level {level}: entity {entity} is Destructible and also carries a PcgVolume — a \
         scattered population is INSTANCES, not geometry, so {names}. Everything the \
         grammar placed stays standing and cannot be broken. Do one of two things: REMOVE \
         the Destructible from this entity (on a population it does nothing), or REPLACE \
         the population with a single mesh — import or model one and point the entity's \
         MeshRef at it. (Collapsing a grammar population into one mesh is implemented, but \
         only as a library call; there is no editor command for it yet, so it is not \
         something you can do from here.)"
    )
}

/// **P23.6: a `Destructible` on a still-scattered (unbaked) population.**
///
/// The P22 ruling, met from the other side: *a `Destructible` names no asset of
/// its own — the cook fractures the ONE mesh its actor's `MeshRef.asset` points
/// at* (`docs/memos/p22-strength.md` §5). The P19 grammar produces a
/// **population**: `ScatteredSolid`s placed as instances, with no merged mesh
/// anywhere in the pipeline. So an author who puts a `Destructible` on a grammar
/// building gets a cook that succeeds, a level that loads, a charge that goes
/// off — and a building that stands there, because the only thing that could
/// break is whatever primitive box its `MeshRef` happens to name.
///
/// Nothing in the pipeline fails, which is exactly why it needs a signpost: this
/// is the P16 advisory doctrine's own case, a silent hazard the author cannot
/// see. The fix now exists (P23.6's bake), so the advisory can name it.
///
/// Deliberately **not** conditioned on the population being non-empty: a
/// `PcgVolume`'s `structures` are `#[serde(skip)]`, recomputed from the graph on
/// load, so the persisted level says nothing about how many solids it will
/// produce. Asking would mean evaluating the grammar inside an advisory sweep.
fn unbaked_destructible_populations(db: &AssetDb, closure: &[AssetId]) -> Vec<String> {
    let mut bad: BTreeSet<(AssetId, uuid::Uuid, Option<AssetId>)> = BTreeSet::new();
    for &id in closure {
        let Some(entry) = db.get(id) else { continue };
        if entry.kind() != AssetKind::Level {
            continue;
        }
        let Ok(raw) = std::fs::read(&entry.path) else {
            continue;
        };
        let Ok(level) = inf_scene::decode(&raw) else {
            continue;
        };
        for e in &level.entities {
            if e.destructible.is_none() || e.pcg_volume.is_none() {
                continue;
            }
            bad.insert((
                id,
                e.guid,
                e.mesh.as_ref().and_then(|m| m.asset).map(AssetId),
            ));
        }
    }
    bad.into_iter()
        .map(|(level, entity, mesh)| unbaked_population_advisory(level, entity, mesh))
        .collect()
}

/// **P21.2 (a): a `VoxelVolume.voxel_size_m` that disagrees with its asset.**
///
/// The component carries a voxel scale and so does the `.inf_voxel` header, and
/// the asset's is the one that wins at runtime — the chunks were meshed at it.
/// A component that says something else is not a build failure (nothing breaks,
/// the volume loads and renders), it is a level whose *authored intent* about
/// scale is quietly discarded: whoever typed 0.25 into the Details panel gets a
/// cave built at 0.5 and no indication why.
///
/// So the advisory quotes **both measured numbers** and names the fix as an edit
/// to the component, because the asset's value is the one that cannot be wrong.
///
/// Compared exactly, not with a tolerance. Both numbers travel as `f64` from the
/// same authoring act, so any difference at all is a real disagreement — and a
/// tolerance here would be a threshold nobody could justify the width of.
fn voxel_scale_mismatches(db: &AssetDb, closure: &[AssetId]) -> Vec<String> {
    // `(level, volume asset, component scale bits, asset scale)`. Bits rather
    // than the `f64` so the key is `Ord` without a wrapper; the value is
    // reconstructed for the message.
    let mut hits: BTreeSet<(AssetId, AssetId, u64, u64)> = BTreeSet::new();
    for &id in closure {
        let Some(entry) = db.get(id) else { continue };
        if entry.kind() != AssetKind::Level {
            continue;
        }
        let Ok(raw) = std::fs::read(&entry.path) else {
            continue;
        };
        let Ok(level) = inf_scene::decode(&raw) else {
            continue;
        };
        for e in &level.entities {
            let Some(v) = e.voxel_volume.as_ref() else {
                continue;
            };
            let Some(asset) = v.asset else { continue };
            let Some(vol) = db.get(AssetId(asset)) else {
                // Missing entirely — `dangling_terrain_refs` already says so, and
                // saying it twice is exactly the noise the advisory law forbids.
                continue;
            };
            let Ok(bytes) = std::fs::read(&vol.path) else {
                continue;
            };
            let Ok(reader) = inf_voxel::VoxelAssetReader::new(bytes) else {
                // A corrupt payload fails the cook outright in the structural
                // validation pass; there is nothing for an advisory to add.
                continue;
            };
            if reader.voxel_size_m() != v.voxel_size_m {
                hits.insert((
                    id,
                    AssetId(asset),
                    v.voxel_size_m.to_bits(),
                    reader.voxel_size_m().to_bits(),
                ));
            }
        }
    }
    hits.into_iter()
        .map(|(level, asset, component, asset_scale)| {
            format!(
                "level {level}: voxel volume {asset} is built at {} m per voxel but its \
                 component says {} m; the asset wins at runtime — set VoxelVolume.voxel_size_m \
                 to {} m or rebuild the volume",
                f64::from_bits(asset_scale),
                f64::from_bits(component),
                f64::from_bits(asset_scale),
            )
        })
        .collect()
}

/// The world transform of a persisted entity: its own local affine composed with
/// every parent's, root-last.
///
/// A `.inf_lvl` stores only the **local** `Transform` — `GlobalTransform` is
/// computed by `inf_ecs::transform::propagate` over a live world and is never
/// serialized — so an advisory that wants to know where something actually is has
/// to walk the chain itself. Depth-guarded at 64, because a merge-mangled level
/// can contain a parent cycle and an advisory must never hang a build.
///
/// Shared by [`uphill_rivers`] and [`see_through_pits`]: two advisories reading a
/// level in two different frames is precisely how one of them ends up quoting
/// coordinates from a world that does not exist.
fn world_affine(
    level: &inf_scene::RuntimeLevel,
    start: &inf_scene::RuntimeEntity,
) -> glam::DAffine3 {
    let mut e = start;
    let mut affine = e.transform.affine();
    let mut depth = 0u32;
    while let Some(pg) = e.parent {
        depth += 1;
        if depth > 64 {
            break;
        }
        let Some(parent) = level.entity(pg) else {
            break;
        };
        affine = parent.transform.affine() * affine;
        e = parent;
    }
    affine
}

/// **P21.2 (b): a see-through pit — holed heightfield with nothing behind it.**
///
/// A P21.2 carve removes the heightfield surface at a set of samples so that the
/// voxel volume underneath shows through. If no volume covers those samples, what
/// shows through is the **sky** (or the inside of the level), from below and from
/// above, and gameplay falls through the world at exactly that spot.
///
/// This is the hazard that has no other alarm: the level loads, the terrain
/// streams, the cook succeeds, and the hole is invisible from any camera that
/// never looks at it. So the advisory quotes the measured sample count and the
/// tile it is in, and names the fix in the two forms it actually takes.
///
/// **It reads the `.inf_terrain` asset, not the level.** That is not an
/// implementation detail — it is the only place holes exist. `TerrainData`'s wire
/// form is pinned at tile generation 3 (the scene schema is frozen at v19), so a
/// hole mask cannot ride an `.inf_lvl` at all; see
/// `inf_terrain::TerrainTileFrozenV1`'s generation table. A terrain with no asset
/// therefore has no holes to report, and this walks nothing.
///
/// Coverage is tested at **chunk-column** granularity: a holed sample is covered
/// if some volume in the level has a resident chunk whose XZ footprint contains
/// it. Deliberately coarse in the *permissive* direction — a chunk that exists
/// over the hole might still be empty air inside — because the alternative is
/// decoding every chunk's SDF during a cook to catch a case the author almost
/// certainly did on purpose. An advisory that fires on correct caves stops being
/// read, which is the failure this whole class of check is trying to avoid.
///
/// # BOTH SIDES ARE PLACED IN THE WORLD (P21.2 audit)
///
/// The first cut compared the two footprints in **raw asset space**, so a level
/// whose cave sat at the origin in its `.inf_voxel` and a hundred metres away in
/// its level answered a question nobody asked. Both sides now go through
/// [`world_affine`], so a moved terrain, a moved volume, or either of them under a
/// moved or rotated *parent* is judged where the level actually puts it.
///
/// **Translation, and deliberately only translation** — because that is the whole
/// of what the runtime honours, and this advisory must describe the world that
/// ships rather than a more principled one that does not:
///
/// * a volume is placed by `inf_voxel::VoxelVolumes::place`, which offsets the
///   asset's own anchor by the entity translation (`ChunkGrid::new(voxel_size,
///   origin + translation)`), and the shipped player feeds it exactly
///   `GlobalTransform::translation()`;
/// * a terrain's world frame is `GlobalTransform::translation()` too — every
///   consumer (`ground_height_at`'s heightfield arm, the scatter and biome passes
///   in `inf_player::level`) takes the translation and nothing else.
///
/// So a *rotated* volume covers the same ground a same-place unrotated one does,
/// here and at runtime alike, and the fixtures below pin that rather than leave it
/// to be rediscovered. Composing the parent chain as a full affine and taking its
/// translation is still the right way to get there: a child of a rotated parent
/// really does move, and it moves by exactly this.
///
/// The frames, spelled out, because the two sides are genuinely different:
///
/// * a hole sample is at `terrain_translation.xz + coord · span + (i, j) · mps`.
///   The `.inf_terrain` header's own origin is **not** in it —
///   `TerrainData::tile_origin_xz` is `coord · span` and nothing else, so the
///   header value round-trips through a write-back and never places a tile.
/// * a chunk column is at `voxel_asset_origin.xz + volume_translation.xz +
///   base_sample · voxel_size`, which is the frame `VoxelData::world_to_grid`
///   answers in.
fn see_through_pits(db: &AssetDb, closure: &[AssetId]) -> Vec<String> {
    // `(level, terrain asset, tile x, tile z, holed samples)`.
    let mut hits: BTreeSet<(AssetId, AssetId, i32, i32, usize)> = BTreeSet::new();
    for &id in closure {
        let Some(entry) = db.get(id) else { continue };
        if entry.kind() != AssetKind::Level {
            continue;
        }
        let Ok(raw) = std::fs::read(&entry.path) else {
            continue;
        };
        let Ok(level) = inf_scene::decode(&raw) else {
            continue;
        };

        // Every voxel volume in the level, as covered XZ world rectangles. One
        // set for the whole level: a hole is covered if ANY volume covers it, and
        // which entity owns that volume is not the author's question here.
        let mut covered: Vec<(f64, f64, f64, f64)> = Vec::new();
        for e in &level.entities {
            let Some(asset) = e.voxel_volume.as_ref().and_then(|v| v.asset) else {
                continue;
            };
            let Some(vol) = db.get(AssetId(asset)) else {
                continue;
            };
            let Ok(bytes) = std::fs::read(&vol.path) else {
                continue;
            };
            let Ok(reader) = inf_voxel::VoxelAssetReader::new(bytes) else {
                continue;
            };
            let vs = reader.voxel_size_m();
            let dim = inf_voxel::CHUNK_DIM as f64;
            // Where the level puts this cave: the asset's own anchor offset by
            // the entity's WORLD translation, which is what `VoxelVolumes::place`
            // does to build the grid the runtime queries against.
            let o = reader.origin() + world_affine(&level, e).translation;
            for key in reader.keys() {
                let b = key.base_sample();
                let x0 = o.x + b[0] as f64 * vs;
                let z0 = o.z + b[2] as f64 * vs;
                covered.push((x0, z0, x0 + dim * vs, z0 + dim * vs));
            }
        }

        for e in &level.entities {
            let Some(t) = e.terrain.as_ref() else {
                continue;
            };
            let Some(asset) = t.asset else { continue };
            let Some(ta) = db.get(AssetId(asset)) else {
                continue;
            };
            let Ok(bytes) = std::fs::read(&ta.path) else {
                continue;
            };
            let Ok(reader) = inf_terrain::TerrainAssetReader::new(bytes) else {
                continue;
            };
            let res = reader.tile_resolution();
            let mps = reader.meters_per_sample();
            let span = (res as f64 - 1.0) * mps;
            // The heightfield's world frame is its entity's translation — see the
            // doc above for why the `.inf_terrain` header origin is not in it.
            let base = world_affine(&level, e).translation;
            for key in reader.keys().collect::<Vec<_>>() {
                // LOD \u2265 1 pages are downsampled heights only — the pyramid carries
                // no hole mask, so walking them would double-count nothing and
                // find nothing.
                if key.lod != 0 {
                    continue;
                }
                let Ok(Some(tile)) = reader.tile(key) else {
                    continue;
                };
                if !tile.has_holes() {
                    continue;
                }
                let ox = base.x + key.coord.0 as f64 * span;
                let oz = base.z + key.coord.1 as f64 * span;
                let mut naked = 0usize;
                for j in 0..res {
                    for i in 0..res {
                        if !tile.is_hole(res, i, j) {
                            continue;
                        }
                        let x = ox + i as f64 * mps;
                        let z = oz + j as f64 * mps;
                        if !covered
                            .iter()
                            .any(|&(x0, z0, x1, z1)| x >= x0 && x < x1 && z >= z0 && z < z1)
                        {
                            naked += 1;
                        }
                    }
                }
                if naked > 0 {
                    hits.insert((id, AssetId(asset), key.coord.0, key.coord.1, naked));
                }
            }
        }
    }
    hits.into_iter()
        .map(|(level, asset, tx, tz, naked)| {
            format!(
                "level {level}: terrain {asset} tile ({tx},{tz}) has {naked} holed sample(s) \
                 no voxel volume covers; the ground is see-through there — extend a volume \
                 over them, or fill the holes back in with the carve tool"
            )
        })
        .collect()
}

/// How much elevation a river must gain, in metres, before the cook says so.
///
/// **The Ring-0 constant, not a local copy** (P20.4). It moved to
/// [`inf_water::UPHILL_TOLERANCE_M`] the moment it acquired a second reader — the
/// editor's river tool, which re-runs both climb checks so the tool says what the
/// build will say. Two copies of a threshold that two different surfaces quote to
/// the author is exactly the drift a shared constant exists to prevent; the
/// argument for the *value* lives on the constant.
use inf_water::UPHILL_TOLERANCE_M as RIVER_UPHILL_TOLERANCE_M;

/// Advisory: rivers, in the levels being cooked, whose surface **gains
/// elevation** in the direction they flow (P20.1).
///
/// Non-fatal, like every other advisory here: the level is valid and cooks fine,
/// and a stylised game may genuinely want a river running up a hill. What it is
/// not is *accidental* — and an accidental one is invisible until somebody
/// notices the water is wrong, because nothing crashes and nothing is missing.
///
/// ## What is checked, and what is not
///
/// The **water surface** is the spline, so this reads the spline's own elevation
/// profile along arc length — the same [`inf_water::RiverPath`] the renderer and
/// the sim build, sampled at the same frames, so the advisory and the picture
/// agree. That is deliberately the strongest check available at cook time: it
/// needs no terrain at all, so it works for a level whose ground streams from an
/// `.inf_terrain` the cook never decodes.
///
/// The **authored bed** is checked too since P20.4, on the same terrain-free
/// terms: [`inf_water::RiverPath::bed_profile_from_depth`] lowers the surface by
/// the profile's depth taper, and a bed that *climbs* is a basin however cleanly
/// the surface falls (a river descending 2 m while its depth tapers from 5 m to
/// 0.5 m has a bed 2.5 m higher at the mouth than at the source). It is a second
/// advisory rather than a stronger version of the first, because they have
/// different remedies — one moves spline points, the other moves depths — and an
/// author told only "your river is wrong" would fix the wrong one.
///
/// What is still **not** checked here is the bed against the *ground*: "does this
/// river run inside the hill, or hang in the air over a gorge?" needs a
/// heightfield, and the answer lives in tile payloads inside a `.inf_terrain` the
/// cook validates structurally and never pages in. That check exists — it is
/// `inf_water::hydro::bed_conflicts`, over `RiverPath::bed_profile` — and it runs
/// in the P20.4 authoring tools, where the terrain is resident.
///
/// A river entity's transform is applied translation-and-rotation-wise through
/// [`Transform::affine`], and parent chains are followed, so a river under a
/// moved or rotated parent is judged where it actually is.
///
/// Deduplicated + sorted (by entity) so the report stays deterministic.
fn uphill_rivers(guid: AssetId, level: &inf_scene::RuntimeLevel) -> Vec<String> {
    use inf_ecs::components::{SplineInterp, WaterKind};

    let mut out: Vec<(uuid::Uuid, String)> = Vec::new();
    for e in &level.entities {
        let Some(water) = e.water_body else { continue };
        if water.kind != WaterKind::River {
            continue;
        }
        let Some(spline) = e.spline.as_ref() else {
            continue;
        };
        if spline.points.len() < 2 {
            continue;
        }
        let affine = world_affine(level, e);
        let points: Vec<glam::DVec3> = spline
            .points
            .iter()
            .map(|p| affine.transform_point3(p.to_dvec3()))
            .collect();
        let interp = match spline.interp {
            SplineInterp::Linear => inf_math::spline::SplineInterp::Linear,
            SplineInterp::CatmullRom => inf_math::spline::SplineInterp::CatmullRom,
        };
        // ONE sanitizer, in Ring 0 (P20.4). This call site used the RAW fields
        // until the P20.4 audit: a negative authored depth tapered the cook's bed
        // differently from the renderer's, the sim's and the tool's, which is
        // exactly the disagreement the editor's re-run of this check exists to
        // rule out.
        let profile = inf_water::RiverProfile::authored(
            water.river_width_start_m,
            water.river_width_end_m,
            water.river_depth_start_m,
            water.river_depth_end_m,
            water.river_flow_m_s,
        );
        let path = inf_water::RiverPath::from_points(&points, spline.closed, interp, &profile);
        if path.is_empty() {
            continue;
        }
        // A NEGATIVE flow speed reverses the river without re-authoring the
        // spline, so "downhill" reverses with it: read the profile backwards.
        let mut elevations = path.surface_profile();
        if water.river_flow_m_s < 0.0 {
            let total = path.length_m;
            elevations.reverse();
            for (s, _) in elevations.iter_mut() {
                *s = total - *s;
            }
        }
        // A CLOSED river cannot help gaining every metre it loses — it is a loop.
        // Advising on it would be advising on a circle, so it is skipped and said
        // so rather than silently filtered.
        if path.closed {
            continue;
        }
        // Two profiles, two advisories, one traversal (P20.4). The reversal above
        // has already been applied to `elevations`, so the bed is read the same
        // way round — a river whose flow was reversed must have BOTH its surface
        // and its bed judged in the direction the water actually goes.
        let mut bed = path.bed_profile_from_depth();
        if water.river_flow_m_s < 0.0 {
            let total = path.length_m;
            bed.reverse();
            for (s, _) in bed.iter_mut() {
                *s = total - *s;
            }
        }
        let surface_spans = inf_water::river::uphill_spans(&elevations, RIVER_UPHILL_TOLERANCE_M);
        let bed_spans = inf_water::river::uphill_spans(&bed, RIVER_UPHILL_TOLERANCE_M);
        if !surface_spans.is_empty() {
            let total: f64 = surface_spans.iter().map(|s| s.rise_m).sum();
            let worst = worst_span(&surface_spans);
            out.push((
                e.guid,
                format!(
                    "level {guid}: river entity {} climbs {total:.2} m across {} stretch(es) in \
                     the direction it flows (the worst gains {:.2} m over {:.1} m, a gradient of \
                     {:.1}%) — water does not flow uphill, so either re-order the spline points, \
                     lower them, or set a negative `river_flow_m_s` to reverse the flow",
                    e.guid,
                    surface_spans.len(),
                    worst.rise_m,
                    worst.length_m(),
                    worst.gradient() * 100.0,
                ),
            ));
        }
        if !bed_spans.is_empty() {
            let total: f64 = bed_spans.iter().map(|s| s.rise_m).sum();
            let worst = worst_span(&bed_spans);
            out.push((
                e.guid,
                format!(
                    "level {guid}: river entity {}'s BED climbs {total:.2} m across {} \
                     stretch(es) in the direction it flows (the worst gains {:.2} m over {:.1} \
                     m, a gradient of {:.1}%) — the surface can fall while the depth taper \
                     lifts the bed under it, which is a basin rather than a river; raise \
                     `river_depth_end_m`, lower `river_depth_start_m`, or drop the downstream \
                     spline points",
                    e.guid,
                    bed_spans.len(),
                    worst.rise_m,
                    worst.length_m(),
                    worst.gradient() * 100.0,
                ),
            ));
        }
    }
    // Stable, so an entity carrying BOTH a climbing surface and a climbing bed
    // keeps them in that order rather than in whichever the sort happened to
    // produce.
    out.sort_by_key(|(g, _)| *g);
    out.into_iter().map(|(_, m)| m).collect()
}

/// The worst climb in a non-empty span list — the one an advisory quotes.
fn worst_span(spans: &[inf_water::UphillSpan]) -> inf_water::UphillSpan {
    spans
        .iter()
        .max_by(|a, b| a.rise_m.total_cmp(&b.rise_m))
        .copied()
        .unwrap_or(spans[0])
}

/// Advisory: `Terrain.biome_set` references, in the levels being cooked, that
/// name an asset the project database does not have (P19.2).
///
/// The exact twin of [`dangling_terrain_refs`], and non-fatal for the same
/// reason: the level is still valid — its per-sample biome ids are stored on the
/// tiles and cook fine — but nothing can *resolve* them, so the biome overlay is
/// blank and P19.3's per-biome dispatch finds no graphs. That is a real silent
/// hole, and the closure cannot follow the edge to complain about it on its own.
/// Deduplicated + sorted so the report stays deterministic.
fn dangling_biome_set_refs(db: &AssetDb, closure: &[AssetId]) -> Vec<String> {
    let mut missing: BTreeSet<(AssetId, AssetId)> = BTreeSet::new();
    for &id in closure {
        let Some(entry) = db.get(id) else { continue };
        if entry.kind() != AssetKind::Level {
            continue;
        }
        let Ok(raw) = std::fs::read(&entry.path) else {
            continue;
        };
        let Ok(level) = inf_scene::decode(&raw) else {
            continue;
        };
        for e in &level.entities {
            if let Some(set) = e.terrain.as_ref().and_then(|t| t.biome_set) {
                if !db.contains(AssetId(set)) {
                    missing.insert((id, AssetId(set)));
                }
            }
        }
    }
    missing
        .into_iter()
        .map(|(level, set)| {
            format!(
                "level {level} references missing biome set {set}; its painted biome ids will \
                 not resolve"
            )
        })
        .collect()
}

/// `.inf_pcg` graphs in the closure that use a `mask.image` node (P19.3).
///
/// **This is a build-time advisory for a stated runtime gap, not a dangling
/// reference.** The editor resolves an image mask's texture through its live
/// asset database and lowers real pixels; the shipped and PIE players have no
/// such database on the evaluation path and lower the node to an **empty** mask,
/// which scores `0` everywhere. So a graph that uses one places *less* content
/// once shipped than it did in the preview — and it does so **silently**, because
/// failing closed is otherwise exactly the right behaviour and nothing downstream
/// can tell "masked out" from "authored empty".
///
/// The cook is the right place to say so: it is the last moment before the
/// difference becomes a shipped build, and it is the only place that can see both
/// the graph and the fact that it is being packaged. Non-fatal, because the level
/// is valid and an author may genuinely be mid-work.
///
/// Deduplicated + sorted, like every other advisory here, so the report stays
/// deterministic.
fn unresolvable_image_masks(db: &AssetDb, closure: &[AssetId]) -> Vec<String> {
    let mut found: BTreeSet<AssetId> = BTreeSet::new();
    for &id in closure {
        let Some(entry) = db.get(id) else { continue };
        if entry.kind() != AssetKind::Pcg {
            continue;
        }
        let Ok(raw) = std::fs::read(&entry.path) else {
            continue;
        };
        let Ok(payload) = inf_pcg::PcgAssetPayload::decode(&raw) else {
            continue;
        };
        let Some(graph) = payload.graph() else {
            continue;
        };
        if graph.nodes.values().any(|n| n.type_id == "mask.image") {
            found.insert(id);
        }
    }
    found
        .into_iter()
        .map(|id| {
            format!(
                "pcg graph {id} uses an Image Mask; the shipped and PIE players cannot resolve a \
                 mask texture at load, so that mask evaluates to zero and the graph places less \
                 than it does in the editor"
            )
        })
        .collect()
}

/// Advisory: grammar modules, in the `.inf_pcg` graphs being cooked, that name a
/// mesh the project database does not have (P19.4).
///
/// The dangling-reference twin of [`dangling_terrain_refs`], one asset kind
/// over. [`dependency_closure`] can only follow edges it can resolve, so a
/// module whose mesh GUID is a typo (or points at a deleted asset) is silently
/// skipped — and a grammar fails *quietly*: the derivation still runs, the slot
/// still consumes its span, and the wall simply has a piece missing. Non-fatal,
/// because the level is valid and an author may be mid-work; deduplicated and
/// sorted, like every other advisory here, so the report stays deterministic.
fn dangling_grammar_modules(db: &AssetDb, closure: &[AssetId]) -> Vec<String> {
    let mut missing: BTreeSet<(AssetId, AssetId)> = BTreeSet::new();
    for &id in closure {
        let Some(entry) = db.get(id) else { continue };
        if entry.kind() != AssetKind::Pcg {
            continue;
        }
        let Ok(raw) = std::fs::read(&entry.path) else {
            continue;
        };
        for mesh in grammar_module_refs(&raw) {
            if !db.contains(AssetId(mesh)) {
                missing.insert((id, AssetId(mesh)));
            }
        }
    }
    missing
        .into_iter()
        .map(|(pcg, mesh)| {
            format!(
                "pcg graph {pcg} declares a grammar module whose mesh {mesh} is not in the \
                 project; that module places nothing and its slot stays empty"
            )
        })
        .collect()
}

/// The sub-threshold vmesh advisory (P18.3 audit). Pure rules, so the trigger and
/// the wording are pinned without running a cook — the `partition::streamed_*`
/// precedent.
#[cfg(test)]
mod vmesh_advisory {
    use super::{
        classify_skip, sub_threshold_advisory, unbaked_population_advisory, AssetId, VmeshSkip,
    };

    /// Real geometry under the bar is the advisable case; an empty mesh is not.
    /// A cube IS the honest rendering of no geometry, so saying anything about it
    /// would be noise — and noise is how advisories stop being read.
    #[test]
    fn only_real_geometry_under_the_bar_is_advisable() {
        assert_eq!(
            classify_skip(500, 1500, 2048),
            VmeshSkip::BelowThreshold {
                triangles: 500,
                min: 2048
            }
        );
        assert_eq!(classify_skip(0, 0, 2048), VmeshSkip::NoGeometry);
        assert_eq!(classify_skip(0, 2, 2048), VmeshSkip::NoGeometry);
        // A zero/one threshold still means "everything", not "nothing".
        assert_eq!(
            classify_skip(3, 9, 0),
            VmeshSkip::BelowThreshold {
                triangles: 3,
                min: 1
            }
        );
    }

    /// The message has to carry all four things a reader needs: which asset, how
    /// big it is, what the bar was, and what to do. A warning that omits the
    /// remedy is a warning people learn to scroll past.
    #[test]
    fn the_advisory_names_the_asset_the_counts_and_the_remedy() {
        let guid = AssetId::new();
        let msg = sub_threshold_advisory(guid, "Barrel", 500, 2048);
        assert!(msg.contains(&guid.to_string()), "names the asset: {msg}");
        assert!(msg.contains("Barrel"), "names it readably: {msg}");
        assert!(msg.contains("500"), "states the triangle count: {msg}");
        assert!(msg.contains("2048"), "states the threshold: {msg}");
        assert!(msg.contains("min_triangles"), "states the remedy: {msg}");
        assert!(
            msg.contains("PLACEHOLDER CUBE"),
            "states the consequence — the part that makes it worth reading: {msg}"
        );
    }

    /// **P23.6 — the unbaked-population signpost.** Same bar as above: which
    /// level, which entity, what will actually happen, and the fix. The two arms
    /// differ in the *consequence*, which is the part an author acts on: with a
    /// mesh, one box breaks and the building does not; with none, nothing does.
    #[test]
    fn the_unbaked_population_advisory_names_a_fix_an_author_can_perform() {
        let level = AssetId::new();
        let entity = uuid::Uuid::from_u128(0x2306_00AB);
        let mesh = AssetId::new();

        let with = unbaked_population_advisory(level, entity, Some(mesh));
        assert!(with.contains(&level.to_string()), "names the level: {with}");
        assert!(
            with.contains(&entity.to_string()),
            "names the entity: {with}"
        );
        assert!(with.contains(&mesh.to_string()), "names the mesh: {with}");
        assert!(
            with.contains("ONLY that mesh will fracture"),
            "states what actually happens: {with}"
        );
        assert!(
            with.contains("cannot be broken"),
            "states the consequence: {with}"
        );
        // **The remedy must be something an author can DO.** The first version
        // prescribed `inf_editor_core::bake` — a library call with no editor
        // command behind it, which is a note to a programmer rather than advice.
        assert!(
            with.contains("REMOVE the Destructible") && with.contains("REPLACE"),
            "names two fixes an author can actually perform: {with}"
        );
        assert!(
            !with.contains("inf_editor_core") && !with.contains("::"),
            "the advisory prescribes a code path an author cannot invoke: {with}"
        );
        assert!(
            with.contains("no editor command for it yet"),
            "…and says why the obvious fix is not offered: {with}"
        );

        let without = unbaked_population_advisory(level, entity, None);
        assert!(
            without.contains("nothing will fracture"),
            "an entity with no mesh gets the harsher, truer sentence: {without}"
        );
        assert_ne!(
            with, without,
            "the two cases must not read identically — the reader's next action \
             differs"
        );
    }
}
