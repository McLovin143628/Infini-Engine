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
//!   re-projects (world://delta).

use std::collections::BTreeMap;
use std::sync::Mutex;

use glam::{DVec2, DVec3};
use inf_ecs::components::{GlobalTransform, PcgVolume, ScatteredInstance, Terrain, Transform};
use inf_graph::{
    apply_edits, compile::validate, GraphDoc, GraphEdit, GraphIssue, GraphJournal, Link, NodeDef,
    NodeId, NodeRegistry,
};
use inf_pcg::graph::{lower_graph, pcg_registry as build_pcg_registry, PcgSeverity};
use inf_pcg::height::{FnHeight, HeightProvider};
use inf_pcg::scatter::Region;
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};
use uuid::Uuid;

use super::scene::{emit_world_delta, SceneState};

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
}

/// Result of applying an edit batch (mirrors `GraphApplyResult`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PcgApplyResult {
    pub issues: Vec<GraphIssue>,
    pub can_undo: bool,
    pub can_redo: bool,
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
        journal.record(label, doc.graph.clone());
        apply_edits(&mut doc.graph, registry, &edits);
        let issues = validate(&doc.graph, registry);
        Ok(PcgApplyResult {
            issues,
            can_undo: journal.can_undo(),
            can_redo: journal.can_redo(),
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
) -> Result<PcgCompileResult, String> {
    state.with(|s| {
        let doc = s.docs.get(&id).ok_or_else(|| format!("no pcg `{id}`"))?;
        let lowered = lower_graph(&doc.graph, &s.registry);
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
/// Written as a loose payload — the asset watcher rescans and registers it.
/// Returns the written file name.
#[tauri::command]
pub async fn pcg_save(
    id: String,
    name: String,
    state: State<'_, PcgState>,
    assets: State<'_, super::assets::AssetState>,
) -> Result<String, String> {
    let (bytes, file_name) = state.with(|s| {
        let doc = s.docs.get(&id).ok_or_else(|| format!("no pcg `{id}`"))?;
        let lowered = lower_graph(&doc.graph, &s.registry);
        let payload = inf_pcg::PcgAssetPayload::from_graph(&doc.graph, lowered.document);
        let bytes = payload.encode().map_err(|e| e.to_string())?;
        let base = if name.is_empty() { &doc.name } else { &name };
        let slug: String = base
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        Ok((bytes, format!("{slug}.inf_pcg")))
    })?;

    let root = assets
        .content_root()
        .ok_or("open a project before saving a PCG graph")?;
    let dir = root.join("PCG");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(&file_name), &bytes).map_err(|e| e.to_string())?;
    Ok(file_name)
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
) -> Result<PcgEvaluateResult, String> {
    // Lower under the store lock.
    let (mut document, issues, ok) = state.with(|s| {
        let doc = s.docs.get(&id).ok_or_else(|| format!("no pcg `{id}`"))?;
        let lowered = lower_graph(&doc.graph, &s.registry);
        Ok((lowered.document, issue_dtos(&lowered.issues), lowered.ok))
    })?;

    if !ok {
        return Ok(PcgEvaluateResult {
            entity: entity.unwrap_or_default(),
            placed: 0,
            issues,
            ok: false,
        });
    }

    let placed_entity = {
        let mut doc = scene.doc.lock().map_err(|e| e.to_string())?;
        doc.world_mut().propagate();

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
        let baked: Vec<ScatteredInstance> = instances
            .iter()
            .map(|i| ScatteredInstance {
                position: i.pos,
                rotation: i.rotation,
                scale: i.scale,
                kind: i.kind_index,
            })
            .collect();
        let placed = baked.len() as u32;

        {
            let w = doc.world_mut().world_mut();
            if let Some(mut vol) = w.get_mut::<PcgVolume>(e) {
                vol.evaluated = baked;
            }
        }
        doc.bump_version_for_runtime();
        (guid, placed)
    };

    // Re-emit the scene delta so the viewport re-projects the new instances.
    emit_world_delta(&app, &scene);

    Ok(PcgEvaluateResult {
        entity: placed_entity.0.to_string(),
        placed: placed_entity.1,
        issues,
        ok: true,
    })
}
