//! Ring-2 command surface for the PCG graph editor (ROADMAP P10.5b).
//!
//! Mirrors the Material `material_*` surface: an in-memory workspace of PCG
//! graph documents over the shared `inf-graph` substrate (with the PCG node
//! registry), per-graph undo journals, and CRUD/apply/undo/redo. The
//! PCG-specific commands are:
//!
//! * `pcg_compile` — lower the graph to the stable `PcgDocument` runtime model
//!   and surface node-anchored lowering diagnostics (no GPU/preview — PCG has no
//!   shader; the payoff is the scattered instances in the viewport);
//! * `pcg_save` — persist the authored graph into a `.inf_pcg` asset (the graph
//!   is the on-disk source of truth);
//! * `pcg_evaluate` — lower + evaluate over the scene's terrain and refresh the
//!   target `PcgVolume`'s instance cache, then bump the scene so the viewport
//!   re-projects (world://delta). Since P19.4 that includes the graph's
//!   **grammar passes**: the same `PcgVolume`, the same height provider, spans
//!   resolved from the live scene's `Spline` components, instances appended
//!   after the scatter;
//! * `pcg_evaluate_biomes` — the terrain-level sibling (P19.3): every painted
//!   biome's own graph over the region its id owns, merged into the terrain's
//!   `biome_population`. Everything downstream of a GUID is
//!   [`inf_pcg::BiomeBinding::from_set`], shared verbatim with the player's
//!   load-time pass (`inf_player::level::evaluate_biome_bindings`) — this layer
//!   owns only the *fetch*, which is the whole reason the two cannot drift.
//!
//! Every lowering here goes through [`AssetMasks`], so a `mask.image` node
//! resolves to the real `.inf_tex` pixels instead of failing closed.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use glam::{DVec2, DVec3};
use inf_asset::AssetId;
use inf_ecs::components::{
    GlobalTransform, PcgVolume, ScatteredInstance, Spline, Terrain, Transform,
};
// IB-1: the paging pre-pass walks entities by stable `Guid`, in the same
// spelling the player's mirror uses — `grammar_span_mirror` compares the two
// bodies character for character.
use inf_ecs::Guid;
use inf_graph::{
    apply_edits, compile::validate, GraphDoc, GraphEdit, GraphIssue, GraphJournal, Link, NodeDef,
    NodeId, NodeRegistry,
};
use inf_pcg::graph::{
    lower_graph_with, pcg_registry as build_pcg_registry, MaskSource, PcgSeverity,
};
use inf_pcg::height::{FnHeight, HeightProvider};
use inf_pcg::scatter::Region;
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use uuid::Uuid;

use super::assets::AssetState;
use super::scene::{emit_world_delta, SceneState};

/// **The rectangles procedural generation is about to ask about**, in world XZ,
/// deterministically ordered — one per [`PcgVolume`], plus one per [`Spline`].
///
/// MIRROR: `inf_player::level::pcg_regions_of`. The two hosts must ask for the
/// same ground or they get different worlds — and getting *different* worlds
/// silently is precisely what the island phase's IB-1 was.
fn pcg_regions_of(world: &inf_ecs::EcsWorld) -> Vec<(DVec2, DVec2)> {
    let mut out: Vec<(DVec2, DVec2)> = Vec::new();
    let mut ents: Vec<(Uuid, inf_ecs::Entity)> = {
        let w = world.world();
        let mut v: Vec<(Uuid, inf_ecs::Entity)> = w
            .iter_entities()
            .filter_map(|e| e.get::<Guid>().map(|g| (g.0, e.id())))
            .collect();
        v.sort_by_key(|(g, _)| *g);
        v
    };
    let w = world.world();
    for (_, e) in ents.drain(..) {
        let origin = w
            .get::<GlobalTransform>(e)
            .map(|g| g.translation())
            .unwrap_or(DVec3::ZERO);
        if let Some(v) = w.get::<PcgVolume>(e) {
            let c = DVec2::new(origin.x, origin.z);
            let half = DVec2::new(v.extent.x, v.extent.y);
            out.push((c - half, c + half));
        }
        if let Some(s) = w.get::<Spline>(e) {
            let mut lo = DVec2::splat(f64::INFINITY);
            let mut hi = DVec2::splat(f64::NEG_INFINITY);
            for p in &s.points {
                let wp = DVec2::new(origin.x + p.x, origin.z + p.z);
                lo = lo.min(wp);
                hi = hi.max(wp);
            }
            if lo.is_finite() && hi.is_finite() {
                out.push((lo, hi));
            }
        }
    }
    out
}

/// **Page a streamed terrain's heights before PCG asks for them** (island phase,
/// IB-1) — the editor half.
///
/// MIRROR: `inf_player::level::page_terrains_for_pcg`. The player pages from the
/// pack mapping (or the PIE payload's bytes); the editor pages from the
/// `.inf_terrain` files in the content root, which is the only difference between
/// them and is exactly the difference between the two hosts everywhere else.
///
/// **Without it the two hosts disagree.** Before this, both hosts fell back to
/// `Some(0.0)` over a streamed terrain and therefore *agreed on the wrong
/// answer*, which is why the PIE-equals-shipping gate never caught IB-1. Fixing
/// only the player would have replaced a shared wrong answer with two different
/// answers, which is worse.
fn page_terrains_for_pcg(
    world: &mut inf_ecs::EcsWorld,
    paths: &std::collections::HashMap<Uuid, std::path::PathBuf>,
) -> usize {
    if paths.is_empty() {
        return 0;
    }
    let regions = pcg_regions_of(world);
    if regions.is_empty() {
        return 0;
    }
    let targets: Vec<(Uuid, Uuid, DVec3)> = {
        let w = world.world();
        let mut v: Vec<(Uuid, Uuid, DVec3)> = w
            .iter_entities()
            .filter_map(|e| {
                let guid = e.get::<Guid>()?.0;
                let asset = e.get::<Terrain>()?.asset?;
                let origin = e
                    .get::<GlobalTransform>()
                    .map(|g| g.translation())
                    .unwrap_or(DVec3::ZERO);
                Some((guid, asset, origin))
            })
            .collect();
        v.sort_by_key(|(g, _, _)| *g);
        v
    };

    let mut paged = 0usize;
    for (entity, asset, origin) in targets {
        let Some(path) = paths.get(&asset) else {
            continue;
        };
        let store = match inf_terrain::open_file_tile_store(path) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("pcg: open {} for paging: {e}", path.display());
                continue;
            }
        };
        let header = *store.header();
        let Some(e) = world.entity_of(entity) else {
            continue;
        };
        let Some(mut t) = world.world_mut().get_mut::<Terrain>(e) else {
            continue;
        };
        let stale = t.data.tile_resolution() != header.tile_resolution
            || t.data.meters_per_sample() != header.meters_per_sample;
        if stale {
            if !t.data.is_empty() {
                continue;
            }
            t.data = inf_ecs::TerrainData::new(header.tile_resolution, header.meters_per_sample);
        }
        for (min, max) in &regions {
            let local_min = DVec2::new(min.x - origin.x, min.y - origin.z);
            let local_max = DVec2::new(max.x - origin.x, max.y - origin.z);
            let report =
                inf_terrain::residency::page_region(&mut t.data, &store, local_min, local_max);
            paged += report.loaded.len();
        }
    }
    if paged > 0 {
        tracing::info!("pcg: paged {paged} terrain tile(s) for evaluation");
    }
    paged
}

/// A [`VolumeOutput`](inf_pcg::VolumeOutput) in the ECS's own dependency-light
/// mirror types.
///
/// MIRROR: `inf_player::level::population_of`, character for character between
/// the `MIRROR-BEGIN`/`MIRROR-END` markers — see that function for why the body
/// cannot be hoisted into a shared crate (rings, and the P19.5 dependency-light
/// mirror ruling) and what specifically goes wrong when it drifts: `start` and
/// `inst_start` are both `u32`, so swapping them compiles and silently draws one
/// distant building with another's walls.
fn population_of(
    out: inf_pcg::VolumeOutput,
) -> (
    Vec<ScatteredInstance>,
    Vec<inf_ecs::components::ScatteredSolid>,
    Vec<inf_ecs::StructureGroup>,
) {
    // MIRROR-BEGIN population_of
    let instances = out
        .instances
        .into_iter()
        .map(|i| ScatteredInstance {
            position: i.pos,
            rotation: i.rotation,
            scale: i.scale,
            kind: i.kind_index,
        })
        .collect();
    let solids: Vec<inf_ecs::components::ScatteredSolid> = out
        .colliders
        .iter()
        .map(|s| inf_ecs::components::ScatteredSolid {
            center: s.center,
            half_extents: s.half_extents,
            rotation: s.rotation,
        })
        .collect();
    let groups = out
        .groups
        .iter()
        .map(|g| inf_ecs::StructureGroup {
            shell: inf_ecs::components::ScatteredSolid {
                center: g.shell.center,
                half_extents: g.shell.half_extents,
                rotation: g.shell.rotation,
            },
            start: g.start,
            len: g.len,
            inst_start: g.inst_start,
            inst_len: g.inst_len,
        })
        .collect();
    (instances, solids, groups)
    // MIRROR-END population_of
}

/// The editor's [`MaskSource`]: resolves a `mask.image` node's texture GUID to
/// the live project's `.inf_tex` pixels.
///
/// The lowerer holds no asset database (it is a pure function of a graph), so
/// the bitmap arrives through this seam. Decoding reuses
/// [`inf_material::TextureAsset::level_rgba8`] — the same CPU BC1/BC3 path the
/// thumbnailer draws texture previews with — rather than a second decoder.
///
/// Mip 0 is the mask: a `mask.image` is authored data, not a shading input, so
/// the author's own resolution is the one the density field should have. RGBA8 →
/// grayscale uses the standard Rec.709 luma weights, so a colour mask reads the
/// way an author's eye reads it.
struct AssetMasks<'a>(&'a AssetState);

/// Every entity's [`Spline`] resolved into **world space**, keyed by its stable
/// `Guid` — the P19.4 `grammar.spline` seam's fetch.
///
/// Unlike `mask.image`, this resolves identically in the editor, in PIE and in a
/// shipped build: a `Spline` is a persisted scene component in both codecs, so
/// every host has already built the entity by the time PCG evaluates. There is
/// no preview/shipping divergence here and therefore no cook advisory for it.
///
/// MIRROR: identical in `inf_player::level::collect_spline_paths`, pinned by
/// `inf-editor-core`'s `tests/grammar_span_mirror.rs`.
fn collect_spline_paths(
    world: &inf_ecs::EcsWorld,
) -> std::collections::HashMap<Uuid, inf_pcg::SplinePath> {
    let w = world.world();
    let mut out = std::collections::HashMap::new();
    for e in w.iter_entities() {
        let (Some(guid), Some(spline)) = (e.get::<inf_ecs::Guid>(), e.get::<Spline>()) else {
            continue;
        };
        let to_world = e
            .get::<GlobalTransform>()
            .map(|g| g.0)
            .unwrap_or(glam::DAffine3::IDENTITY);
        out.insert(
            guid.0,
            inf_pcg::SplinePath::from_local(
                spline.points.iter().map(|p| p.to_dvec3()),
                spline.closed,
                match spline.interp {
                    inf_ecs::components::SplineInterp::Linear => inf_pcg::SplineInterp::Linear,
                    inf_ecs::components::SplineInterp::CatmullRom => {
                        inf_pcg::SplineInterp::CatmullRom
                    }
                },
                to_world,
            ),
        );
    }
    out
}

impl MaskSource for AssetMasks<'_> {
    fn mask(&self, texture: Uuid) -> Option<(u32, u32, Vec<u8>)> {
        let bytes = self.0.load_texture_bytes(AssetId(texture))?;
        // `from_payload`, not `inf_asset::decode`: since P26.1 an imported
        // `.inf_tex` is a tiled image, and only the door sniffs both versions.
        let tex = inf_material::TextureAsset::from_payload(&bytes).ok()?;
        let mip = tex.mips.first()?;
        let (w, h) = (mip.width, mip.height);
        let rgba = tex.level_rgba8(0)?;
        let gray: Vec<u8> = rgba
            .chunks_exact(4)
            .map(|p| {
                let luma = 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
                luma.round().clamp(0.0, 255.0) as u8
            })
            .collect();
        // A truncated / mismatched mip would silently shift the mask by a row;
        // fail closed instead (the `NoMasks` contract).
        if gray.len() != (w as usize) * (h as usize) {
            return None;
        }
        Some((w, h, gray))
    }
}

/// In-memory PCG-graph workspace.
struct PcgStore {
    registry: NodeRegistry,
    docs: BTreeMap<String, GraphDoc>,
    journals: BTreeMap<String, GraphJournal>,
    counter: u32,
}

pub struct PcgState {
    inner: Mutex<PcgStore>,
}

impl Default for PcgState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(PcgStore {
                registry: build_pcg_registry(),
                docs: BTreeMap::new(),
                journals: BTreeMap::new(),
                counter: 0,
            }),
        }
    }
}

impl PcgState {
    fn with<R>(&self, f: impl FnOnce(&mut PcgStore) -> Result<R, String>) -> Result<R, String> {
        let mut store = self.inner.lock().map_err(|e| e.to_string())?;
        f(&mut store)
    }

    /// Drop a document + its undo journal from the workspace. Idempotent
    /// (closing an unknown id is a no-op).
    fn close(&self, id: &str) -> Result<(), String> {
        self.with(|s| {
            s.docs.remove(id);
            s.journals.remove(id);
            Ok(())
        })
    }
}

/// Result of applying an edit batch (mirrors `GraphApplyResult`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PcgApplyResult {
    pub issues: Vec<GraphIssue>,
    pub can_undo: bool,
    pub can_redo: bool,
    /// **How many of the submitted edits the backend REFUSED** (C4-44).
    ///
    /// Non-zero means the canvas and the graph disagree: the canvas is
    /// fully controlled, so it must re-derive from the backend rather than
    /// keep drawing a wire that does not exist.
    pub rejected: u32,
}

/// A lowering diagnostic anchored (best-effort) to a node.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PcgIssueDto {
    pub node: Option<u32>,
    pub severity: String,
    pub message: String,
}

/// Result of lowering a PCG graph to the runtime model.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PcgCompileResult {
    pub ok: bool,
    pub issues: Vec<PcgIssueDto>,
    /// How many rules the lowered document holds (v1: 0 or 1).
    pub rule_count: u32,
    /// A short human summary of the lowered document (for the panel).
    pub summary: String,
}

/// Result of evaluating a volume.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PcgEvaluateResult {
    /// Entity GUID the scatter was written to.
    pub entity: String,
    /// How many instances were placed.
    pub placed: u32,
    /// Lowering diagnostics (a compile error → `placed == 0`).
    pub issues: Vec<PcgIssueDto>,
    pub ok: bool,
}

/// Result of evaluating a terrain's **biome→PCG binding** (P19.3).
///
/// Deliberately not a `PcgEvaluateResult`: this command lowers no open document,
/// so it has no node-anchored diagnostics to report. What it *does* have is a
/// dispatch count and the blend width they ran under — the two numbers that tell
/// an author whether the thing they painted is what ran.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PcgBiomeResult {
    /// Terrain entity GUID the population was written to.
    pub entity: String,
    /// How many instances the merged population holds.
    pub placed: u32,
    /// How many biomes actually dispatched (biomes whose `pcg_graph` resolved).
    pub biomes: u32,
    /// The border blend width the dispatches ran under, in metres.
    pub feather: f64,
    pub ok: bool,
    /// A short human summary (or the reason nothing ran).
    pub message: String,
}

fn issue_dtos(issues: &[inf_pcg::graph::PcgGraphIssue]) -> Vec<PcgIssueDto> {
    issues
        .iter()
        .map(|i| PcgIssueDto {
            node: i.node,
            severity: match i.severity {
                PcgSeverity::Error => "error".into(),
                PcgSeverity::Warning => "warning".into(),
            },
            message: i.message.clone(),
        })
        .collect()
}

/// The PCG node palette.
#[tauri::command]
pub async fn pcg_registry(state: State<'_, PcgState>) -> Result<Vec<NodeDef>, String> {
    state.with(|s| Ok(s.registry.dtos()))
}

/// All open PCG documents.
#[tauri::command]
pub async fn pcg_list(state: State<'_, PcgState>) -> Result<Vec<GraphDoc>, String> {
    state.with(|s| Ok(s.docs.values().cloned().collect()))
}

/// Create a new PCG graph seeded with a valid `const → scatter → output` chain.
#[tauri::command]
pub async fn pcg_create(
    app: AppHandle,
    name: String,
    state: State<'_, PcgState>,
) -> Result<GraphDoc, String> {
    let doc = state.with(|s| {
        s.counter += 1;
        let id = format!("pcg:{}", s.counter);
        let mut graph = inf_graph::Graph::empty();
        // Allocate ids for the seed chain up front.
        let konst = NodeId(graph.next_id);
        let scatter = NodeId(graph.next_id + 1);
        let output = NodeId(graph.next_id + 2);
        apply_edits(
            &mut graph,
            &s.registry,
            &[
                GraphEdit::AddNode {
                    id: konst,
                    type_id: "const.density".into(),
                    x: 40.0,
                    y: 160.0,
                    params: Default::default(),
                },
                GraphEdit::AddNode {
                    id: scatter,
                    type_id: "scatter.scatter".into(),
                    x: 320.0,
                    y: 160.0,
                    params: Default::default(),
                },
                GraphEdit::AddNode {
                    id: output,
                    type_id: "output.pcg".into(),
                    x: 600.0,
                    y: 160.0,
                    params: Default::default(),
                },
                GraphEdit::Connect {
                    link: Link {
                        from: konst,
                        from_port: "out".into(),
                        to: scatter,
                        to_port: "density".into(),
                    },
                },
                GraphEdit::Connect {
                    link: Link {
                        from: scatter,
                        from_port: "out".into(),
                        to: output,
                        to_port: "scatter".into(),
                    },
                },
            ],
        );
        let doc = GraphDoc {
            id: id.clone(),
            name: if name.is_empty() {
                format!("PCG {}", s.counter)
            } else {
                name
            },
            graph,
            viewport: None,
            modified_ms: 0,
        };
        s.docs.insert(id.clone(), doc.clone());
        s.journals.insert(id, GraphJournal::new(64));
        Ok(doc)
    })?;
    let _ = app.emit("pcg://sync", doc.id.clone());
    Ok(doc)
}

/// Fetch one document.
#[tauri::command]
pub async fn pcg_get(id: String, state: State<'_, PcgState>) -> Result<GraphDoc, String> {
    state.with(|s| {
        s.docs
            .get(&id)
            .cloned()
            .ok_or_else(|| format!("no pcg `{id}`"))
    })
}

/// Close a document: free its graph + undo journal so open PCG graphs don't
/// accumulate for the life of the session. Called when the editing surface is
/// discarded.
#[tauri::command]
pub async fn pcg_close(id: String, state: State<'_, PcgState>) -> Result<(), String> {
    state.close(&id)
}

/// Apply an edit batch; records one undo entry and re-validates.
#[tauri::command]
pub async fn pcg_apply(
    app: AppHandle,
    id: String,
    edits: Vec<GraphEdit>,
    label: String,
    state: State<'_, PcgState>,
) -> Result<PcgApplyResult, String> {
    let result = state.with(|s| {
        let registry = &s.registry;
        let doc = s
            .docs
            .get_mut(&id)
            .ok_or_else(|| format!("no pcg `{id}`"))?;
        let journal = s.journals.get_mut(&id).ok_or("missing journal")?;
        // C4-44: `apply_edits` returns how many edits took effect, and all three
        // canvases discarded it. `apply_edit` answers `false` for a Connect that
        // would cycle, a missing node or an unknown param — so a fully-controlled
        // canvas kept drawing a wire the backend had rejected, `validate()`
        // reported nothing (the wire is not in the graph to be invalid), and the
        // command returned `Ok`.
        //
        // The snapshot is also taken BEFORE and recorded only when something
        // landed: journalling first left a dead undo step for a batch that
        // changed nothing.
        let before = doc.graph.clone();
        let applied = apply_edits(&mut doc.graph, registry, &edits);
        let rejected = edits.len() - applied;
        if applied > 0 {
            journal.record(label, before);
        }
        let issues = validate(&doc.graph, registry);
        Ok(PcgApplyResult {
            issues,
            can_undo: journal.can_undo(),
            can_redo: journal.can_redo(),
            rejected: rejected as u32,
        })
    })?;
    let _ = app.emit("pcg://sync", id);
    Ok(result)
}

/// Undo the last edit group.
#[tauri::command]
pub async fn pcg_undo(
    app: AppHandle,
    id: String,
    state: State<'_, PcgState>,
) -> Result<Option<GraphDoc>, String> {
    let doc = state.with(|s| {
        let doc = s
            .docs
            .get_mut(&id)
            .ok_or_else(|| format!("no pcg `{id}`"))?;
        let journal = s.journals.get_mut(&id).ok_or("missing journal")?;
        match journal.undo(doc.graph.clone()) {
            Some(prev) => {
                doc.graph = prev;
                Ok(Some(doc.clone()))
            }
            None => Ok(None),
        }
    })?;
    if doc.is_some() {
        let _ = app.emit("pcg://sync", id);
    }
    Ok(doc)
}

/// Redo the last undone edit group.
#[tauri::command]
pub async fn pcg_redo(
    app: AppHandle,
    id: String,
    state: State<'_, PcgState>,
) -> Result<Option<GraphDoc>, String> {
    let doc = state.with(|s| {
        let doc = s
            .docs
            .get_mut(&id)
            .ok_or_else(|| format!("no pcg `{id}`"))?;
        let journal = s.journals.get_mut(&id).ok_or("missing journal")?;
        match journal.redo(doc.graph.clone()) {
            Some(next) => {
                doc.graph = next;
                Ok(Some(doc.clone()))
            }
            None => Ok(None),
        }
    })?;
    if doc.is_some() {
        let _ = app.emit("pcg://sync", id);
    }
    Ok(doc)
}

/// Lower the PCG graph to the runtime model and surface node-anchored
/// diagnostics (the PCG analogue of `material_compile`, minus the preview).
#[tauri::command]
pub async fn pcg_compile(
    id: String,
    state: State<'_, PcgState>,
    assets: State<'_, AssetState>,
) -> Result<PcgCompileResult, String> {
    let masks = AssetMasks(&assets);
    state.with(|s| {
        let doc = s.docs.get(&id).ok_or_else(|| format!("no pcg `{id}`"))?;
        let lowered = lower_graph_with(&doc.graph, &s.registry, &masks);
        let rule_count: u32 = lowered
            .document
            .layers
            .iter()
            .map(|l| l.rules.len() as u32)
            .sum();
        let summary = match lowered
            .document
            .layers
            .first()
            .and_then(|l| l.rules.first())
        {
            Some(rule) => format!(
                "{} rule(s) · cell {} · density {} · {} kind(s)",
                rule_count,
                rule.scatter.cell_size,
                rule.scatter.base_density,
                rule.kinds.len()
            ),
            None => "no rules (graph does not lower to any scatter)".into(),
        };
        Ok(PcgCompileResult {
            ok: lowered.ok,
            issues: issue_dtos(&lowered.issues),
            rule_count,
            summary,
        })
    })
}

/// Persist the authored graph into a `.inf_pcg` asset under the project content
/// root (the graph is the source of truth; the lowered document rides along).
/// Returns the written file name.
///
/// **Through the asset door** (C4-21). This used to be a bare `fs::write` into
/// `<Content>/PCG/<slug>.inf_pcg`: a re-save clobbered the committed payload
/// non-atomically *and* wrote no sidecar, so the file was left for the watcher
/// to promote under a synthesized id derived from its content hash — a GUID
/// that changed on every edit, dangling every reference into the graph.
/// `write_asset_at` overwrites the same path atomically and keeps the GUID.
#[tauri::command]
pub async fn pcg_save(
    id: String,
    name: String,
    state: State<'_, PcgState>,
    assets: State<'_, AssetState>,
) -> Result<String, String> {
    let masks = AssetMasks(&assets);
    let (payload, file_name) = state.with(|s| {
        let doc = s.docs.get(&id).ok_or_else(|| format!("no pcg `{id}`"))?;
        let lowered = lower_graph_with(&doc.graph, &s.registry, &masks);
        let payload = inf_pcg::PcgAssetPayload::from_graph(&doc.graph, lowered.document);
        let base = if name.is_empty() { &doc.name } else { &name };
        let slug: String = base
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        Ok((payload, format!("{slug}.inf_pcg")))
    })?;

    // **Off the async workers** (round-2, the sm/pcg MED). Wave A's atomic-write
    // conversion deleted the `spawn_blocking` wrapper AND the comment that said
    // why ("keep it off the async workers"), so a disk write ran in this
    // `async fn`'s body under two mutexes — inverting the same campaign's
    // Wave-E rule two waves later. `project_handle` exists because
    // `with_project` borrows through `&State`, which is not `Send`.
    let project = assets.project_handle()?;
    tauri::async_runtime::spawn_blocking(move || {
        let mut proj = project.lock().map_err(|e| e.to_string())?;
        let dir = proj.content_dir("PCG").map_err(|e| e.to_string())?;
        proj.write_asset_at(&dir.join(&file_name), &payload, Vec::new(), None)
            .map_err(|e| e.to_string())?;
        Ok(file_name)
    })
    .await
    .map_err(|e| format!("asset write task failed to run: {e}"))?
}

/// Lower the PCG graph and evaluate it over the scene terrain into the target
/// `PcgVolume`'s instance cache, then bump the scene so the viewport re-projects.
///
/// Target resolution: the explicit `entity` GUID, else the first selected
/// entity, else the first entity that already carries a `PcgVolume`. If the
/// target has no `PcgVolume`, a default one is inserted (so "select an entity →
/// Evaluate" works without a spawn menu). Terrain: the first non-empty scene
/// `Terrain` drives height/slope; with none, a flat `y = 0` plane is used
/// (documented). The volume `seed` is folded into every rule's scatter seed.
#[tauri::command]
pub async fn pcg_evaluate(
    app: AppHandle,
    id: String,
    entity: Option<String>,
    state: State<'_, PcgState>,
    scene: State<'_, SceneState>,
    assets: State<'_, AssetState>,
) -> Result<PcgEvaluateResult, String> {
    // Lower under the store lock.
    let masks = AssetMasks(&assets);
    let (mut document, grammars, buildings, issues, ok) = state.with(|s| {
        let doc = s.docs.get(&id).ok_or_else(|| format!("no pcg `{id}`"))?;
        let lowered = lower_graph_with(&doc.graph, &s.registry, &masks);
        Ok((
            lowered.document,
            lowered.grammars,
            lowered.buildings,
            issue_dtos(&lowered.issues),
            lowered.ok,
        ))
    })?;

    if !ok {
        return Ok(PcgEvaluateResult {
            entity: entity.unwrap_or_default(),
            placed: 0,
            issues,
            ok: false,
        });
    }

    // OFF THE ASYNC WORKERS (Hardening Wave E). What follows is a full transform
    // propagation, a whole `TerrainData` clone, `inf_pcg::evaluate` over the
    // volume's region, and the grammar + building passes — seconds of CPU on a
    // real volume, in the body of an `async fn`, parking a Tokio worker that
    // every one of the editor's 239 commands shares. `git.rs`, `files.rs`,
    // `search.rs`, `shell.rs` and `package.rs` all state the rule for work far
    // smaller than this.
    //
    // THE DOC LOCK IS STILL HELD FOR THE WHOLE EVALUATION, deliberately: the
    // terrain it scatters over and the `PcgVolume` it writes back to are the
    // same document, and the file's own note (on `pcg_evaluate_biomes`) records
    // that a consistent snapshot beats a shorter hold. What changes is *who*
    // waits — a viewport frame that wants the document still does; the runtime
    // that serves everything else no longer does.
    // IB-1: where a streamed terrain's heights come from. Resolved OUTSIDE the
    // doc lock (a directory walk under it would block every viewport frame), and
    // handed in as an index exactly as the viewport's own streamer takes one.
    let terrain_paths = assets
        .content_root()
        .map(|root| inf_editor_core::terrain_stream::terrain_paths_by_guid(&root))
        .unwrap_or_default();
    let doc_handle = Arc::clone(&scene.doc);
    let entity_arg = entity.clone();
    let placed_entity = tauri::async_runtime::spawn_blocking(move || {
        let entity = entity_arg;
        let mut doc = doc_handle.lock().map_err(|e| e.to_string())?;
        doc.world_mut().propagate();
        // **Page the ground before asking about it** (island phase, IB-1). An
        // asset-backed terrain carries no inline tiles, so without this the
        // provider below took its `None` arm — a flat `Some(0.0)` — and the
        // editor previewed a scatter at sea level that the shipped build also
        // placed at sea level. The two hosts agreed, on the wrong world.
        page_terrains_for_pcg(doc.world_mut(), &terrain_paths);

        // Resolve the target entity GUID.
        let guid = match &entity {
            Some(s) => Uuid::parse_str(s).map_err(|e| e.to_string())?,
            None => doc
                .selection()
                .first()
                .copied()
                .or_else(|| {
                    doc.order().iter().copied().find(|&g| {
                        doc.entity_of(g)
                            .and_then(|e| doc.world().world().get::<PcgVolume>(e))
                            .is_some()
                    })
                })
                .ok_or("select an entity to scatter into (or add a PCG Volume)")?,
        };
        let e = doc
            .entity_of(guid)
            .ok_or("target entity no longer exists")?;

        // Ensure the volume exists.
        {
            let w = doc.world_mut().world_mut();
            if w.get::<PcgVolume>(e).is_none() {
                w.entity_mut(e).insert(PcgVolume::default());
            }
        }

        // Read owned params (region center + extent + seed).
        let (center, extent, seed) = {
            let w = doc.world().world();
            let vol = w.get::<PcgVolume>(e).ok_or("volume vanished")?;
            let center = w
                .get::<GlobalTransform>(e)
                .map(|g| g.translation())
                .or_else(|| w.get::<Transform>(e).map(|t| t.translation.to_dvec3()))
                .unwrap_or(DVec3::ZERO);
            (center, vol.extent, vol.seed)
        };

        // Clone the first non-empty terrain (data + world origin) for the height
        // provider. `None` ⇒ a flat y = 0 plane.
        let terrain: Option<(inf_ecs::TerrainData, DVec3)> =
            doc.order().iter().copied().find_map(|g| {
                let te = doc.entity_of(g)?;
                let w = doc.world().world();
                let t = w.get::<Terrain>(te)?;
                if t.data.is_empty() {
                    return None;
                }
                let origin = w
                    .get::<GlobalTransform>(te)
                    .map(|gt| gt.translation())
                    .unwrap_or(DVec3::ZERO);
                Some((t.data.clone(), origin))
            });

        // Fold the volume seed into every rule so distinct volumes differ.
        for layer in &mut document.layers {
            for rule in &mut layer.rules {
                rule.scatter.seed = rule.scatter.seed.wrapping_add(seed as u64);
            }
        }

        let region = Region::from_xz(
            center.x - extent.x,
            center.z - extent.y,
            center.x + extent.x,
            center.z + extent.y,
        );
        let provider: Box<dyn HeightProvider> = match terrain {
            Some((data, o)) => Box::new(FnHeight::new(move |x, z| {
                data.height_at(DVec2::new(x - o.x, z - o.z))
                    .map(|h| h + o.y)
            })),
            None => Box::new(FnHeight::new(|_, _| Some(0.0))),
        };

        let instances = inf_pcg::evaluate(&document, provider.as_ref(), region);
        // P19.4/P19.5: the graph's grammar and building passes run on the same
        // volume, against the same height provider, and append after the scatter
        // — grammars first, then buildings, a fixed order, so the cache is a
        // pure function of the content. Everything downstream of the resolved
        // spans is `inf_pcg::evaluate_grammars` / `inf_pcg::evaluate_buildings`,
        // shared verbatim with the player's load-time pass; this layer owns only
        // the fetch.
        let splines = collect_spline_paths(doc.world());
        let cx = inf_pcg::GrammarContext {
            entity: Some(guid),
            center,
            extent: DVec2::new(extent.x, extent.y),
            seed_offset: seed as u64,
        };
        let mut generated = inf_pcg::evaluate_grammars(&grammars, &splines, provider.as_ref(), &cx);
        generated.extend(inf_pcg::evaluate_buildings(
            &buildings,
            &splines,
            provider.as_ref(),
            &cx,
        ));
        // The join is `inf_pcg::compose_volume`, the same door the shipped
        // player's `evaluate_pcg_volumes_in` goes through — I3 made the ORDER
        // load-bearing (a `StructureGroup` carries index ranges into these very
        // lists), so it is stated once in Ring 0 rather than twice here.
        let (baked, solid, groups) = population_of(inf_pcg::compose_volume(instances, generated));
        let placed = baked.len() as u32;

        {
            let w = doc.world_mut().world_mut();
            if let Some(mut vol) = w.get_mut::<PcgVolume>(e) {
                vol.set_population(baked, solid, groups);
            }
        }
        doc.bump_version_for_runtime();
        Ok::<(Uuid, u32), String>((guid, placed))
    })
    .await
    .map_err(|e| format!("pcg_evaluate task failed to run: {e}"))??;

    // Re-emit the scene delta so the viewport re-projects the new instances.
    emit_world_delta(&app, &scene);

    Ok(PcgEvaluateResult {
        entity: placed_entity.0.to_string(),
        placed: placed_entity.1,
        issues,
        ok: true,
    })
}

/// Evaluate a terrain's **biome→PCG binding** (P19.3): every painted biome's own
/// `.inf_pcg` graph over the region its id owns, merged in ascending biome-id
/// order into the terrain's `biome_population`, then bump the scene so the
/// viewport re-projects.
///
/// The terrain-level **sibling** of [`pcg_evaluate`], not a replacement: a volume
/// scatters one graph over a box an author placed, a binding scatters many graphs
/// over the regions an author *painted*. Neither reads the other's output.
///
/// This is the editor half of the P19.3 parity gate. From the GUID onward it is
/// the player's `evaluate_biome_bindings` verbatim — the same
/// [`BiomeBinding::from_set`](inf_pcg::BiomeBinding::from_set), the same
/// [`DEFAULT_BIOME_FEATHER`](inf_pcg::DEFAULT_BIOME_FEATHER), the same
/// [`OffsetTerrain`](inf_pcg::OffsetTerrain) region off
/// [`xz_bounds`](inf_terrain::TerrainData::xz_bounds) — so all this command owns
/// is the *fetch* (an [`AssetState`] lookup here, a pack entry there). The one
/// deliberate difference is [`AssetMasks`]: the editor can resolve a
/// `mask.image` texture and the load-time pass (which has no asset database)
/// cannot, so a biome graph that uses one is richer here. Graphs without an
/// image mask — every graph the binding gate exercises — lower identically.
///
/// Target resolution: the explicit `entity` GUID, else the first entity in
/// creation order whose `Terrain` has both a biome set and authored tiles.
///
/// `biome_population` is a derived, never-persisted cache (`#[serde(skip)]`),
/// exactly like `PcgVolume::evaluated` — which is why re-running this on demand
/// and re-running it on level load can be compared at all.
#[tauri::command]
pub async fn pcg_evaluate_biomes(
    app: AppHandle,
    entity: Option<String>,
    scene: State<'_, SceneState>,
    assets: State<'_, AssetState>,
) -> Result<PcgBiomeResult, String> {
    let masks = AssetMasks(&assets);
    let registry = build_pcg_registry();
    let feather = inf_pcg::DEFAULT_BIOME_FEATHER;

    // The evaluation holds the scene lock throughout, exactly as `pcg_evaluate`
    // does: the terrain data it scatters over and the component it writes back to
    // are the same document, and a consistent snapshot beats a shorter hold.
    let outcome = {
        let mut doc = scene.doc.lock().map_err(|e| e.to_string())?;
        doc.world_mut().propagate();

        const NO_TERRAIN: &str = "no terrain with a biome set — assign one in the Terrain details";

        let guid = match &entity {
            Some(s) => Uuid::parse_str(s).map_err(|e| e.to_string())?,
            None => doc
                .order()
                .iter()
                .copied()
                .find(|&g| {
                    doc.entity_of(g)
                        .and_then(|e| doc.world().world().get::<Terrain>(e))
                        .is_some_and(|t| t.biome_set.is_some() && !t.data.is_empty())
                })
                .ok_or(NO_TERRAIN)?,
        };
        let e = doc
            .entity_of(guid)
            .ok_or("target entity no longer exists")?;

        // Own the terrain's tiles + its world origin + the set its ids name.
        let (data, origin, set_guid) = {
            let w = doc.world().world();
            let t = w.get::<Terrain>(e).ok_or(NO_TERRAIN)?;
            let set_guid = t.biome_set.ok_or(NO_TERRAIN)?;
            let origin = w
                .get::<GlobalTransform>(e)
                .map(|g| g.translation())
                .unwrap_or(DVec3::ZERO);
            (t.data.clone(), origin, set_guid)
        };

        let bytes = assets
            .load_biome_set_bytes(AssetId(set_guid))
            .ok_or_else(|| format!("biome set {set_guid} is not in this project's content"))?;
        let set: inf_terrain::BiomeSet = inf_asset::decode(&bytes)
            .map_err(|err| format!("decode biome set {set_guid}: {err}"))?;

        // The parity seam: which biomes dispatch, in what order, under which
        // feather is `from_set`, shared verbatim with the player.
        let binding = inf_pcg::BiomeBinding::from_set(&set, feather, |g| {
            let bytes = assets.load_pcg_bytes(AssetId(g))?;
            let payload = inf_pcg::PcgAssetPayload::decode(&bytes).ok()?;
            // The stored authored graph re-lowered when there is one (the graph
            // is the source of truth), else the stored lowered mirror — the
            // player's `lowered_document`, stated the same way.
            Some(match payload.graph() {
                Some(graph) => lower_graph_with(&graph, &registry, &masks).document,
                None => payload.document.clone(),
            })
        });
        let biomes = binding.graphs().len() as u32;

        if binding.is_empty() {
            // A set whose biomes name no resolvable graph is not a failure — it
            // is the cook's dangling-reference advisory. Mirroring the player,
            // which skips such a terrain, the existing population is left alone.
            PcgBiomeResult {
                entity: guid.to_string(),
                placed: 0,
                biomes: 0,
                feather,
                ok: false,
                message: format!(
                    "no biome in {} names a PCG graph that resolves — nothing was changed",
                    set.name
                ),
            }
        } else {
            // The terrain's own authored extent, in world space, is the region.
            let (min, max) = data
                .xz_bounds()
                .ok_or("the terrain has no authored tiles to populate")?;
            let region = Region::from_xz(
                min.x + origin.x,
                min.y + origin.z,
                max.x + origin.x,
                max.y + origin.z,
            );
            let fields = inf_pcg::OffsetTerrain::new(&data, origin);
            let provider = FnHeight::new(|x, z| fields.height_at(x, z));
            let baked: Vec<ScatteredInstance> = binding
                .evaluate(&provider, &fields, region)
                .iter()
                .map(|i| ScatteredInstance {
                    position: i.pos,
                    rotation: i.rotation,
                    scale: i.scale,
                    kind: i.kind_index,
                })
                .collect();
            let placed = baked.len() as u32;
            if let Some(mut t) = doc.world_mut().world_mut().get_mut::<Terrain>(e) {
                t.biome_population = baked;
            }
            doc.bump_version_for_runtime();
            PcgBiomeResult {
                entity: guid.to_string(),
                placed,
                biomes,
                feather,
                ok: true,
                message: format!(
                    "{placed} instance(s) from {biomes} biome(s) of {}",
                    set.name
                ),
            }
        }
    };

    // Re-emit the scene delta so the viewport re-projects the new population.
    if outcome.ok {
        emit_world_delta(&app, &scene);
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Insert a bare document + journal directly (mimics `pcg_create`, which
    /// needs a Tauri `State`/`AppHandle` we can't build in a unit test).
    fn seed(state: &PcgState, id: &str) {
        state
            .with(|s| {
                s.docs.insert(
                    id.to_string(),
                    GraphDoc {
                        id: id.to_string(),
                        name: "T".into(),
                        graph: inf_graph::Graph::empty(),
                        viewport: None,
                        modified_ms: 0,
                    },
                );
                s.journals.insert(id.to_string(), GraphJournal::new(8));
                Ok(())
            })
            .unwrap();
    }

    #[test]
    fn close_frees_the_doc_and_journal() {
        let state = PcgState::default();
        seed(&state, "pcg:1");
        state.close("pcg:1").unwrap();
        state
            .with(|s| {
                assert!(s.docs.is_empty(), "doc freed");
                assert!(s.journals.is_empty(), "journal freed");
                Ok(())
            })
            .unwrap();
    }
}
