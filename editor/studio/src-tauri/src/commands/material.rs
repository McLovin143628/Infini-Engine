//! Ring-2 command surface for the Material graph editor (ROADMAP P7.2).
//!
//! Mirrors the Blueprint `graph_*` surface: an in-memory workspace of material
//! graph documents over the shared `inf-graph` substrate (with the material node
//! registry), per-graph undo journals, and the CRUD/apply/undo/redo commands.
//! The material-specific command is `material_compile`: it emits naga-validated
//! WGSL from the graph, maps any errors back onto nodes, and renders a live
//! preview sphere (offscreen) to a PNG data-URL.

use std::collections::BTreeMap;
use std::sync::Mutex;

use inf_editor_core::thumbnail::{encode_png, Thumbnailer};
use inf_graph::{
    apply_edits, compile::validate, GraphDoc, GraphEdit, GraphIssue, GraphJournal, NodeDef,
    NodeRegistry,
};
use inf_material::graph::MatSeverity;
use inf_material::{
    emit_texture_compute, emit_wgsl, material_registry as build_material_registry,
    TextureImportSettings,
};
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

/// In-memory material-graph workspace.
struct MaterialStore {
    registry: NodeRegistry,
    docs: BTreeMap<String, GraphDoc>,
    journals: BTreeMap<String, GraphJournal>,
    counter: u32,
}

pub struct MaterialState {
    inner: Mutex<MaterialStore>,
    /// Headless renderer for the live preview sphere (lazily builds a GPU).
    preview: Mutex<Thumbnailer>,
}

impl Default for MaterialState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(MaterialStore {
                registry: build_material_registry(),
                docs: BTreeMap::new(),
                journals: BTreeMap::new(),
                counter: 0,
            }),
            preview: Mutex::new(Thumbnailer::new(256)),
        }
    }
}

/// Result of applying an edit batch (mirrors `GraphApplyResult`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaterialApplyResult {
    pub issues: Vec<GraphIssue>,
    pub can_undo: bool,
    pub can_redo: bool,
}

/// A material-compile diagnostic anchored (best-effort) to a node.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MatIssueDto {
    pub node: Option<u32>,
    pub severity: String,
    pub message: String,
}

/// The material complexity-budget report (P13.2) mirrored for the frontend
/// stat strip. `*_over` flags are advisory (warnings), not compile errors.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComplexityDto {
    pub nodes: u32,
    pub slabs: u32,
    pub textures: u32,
    pub est_alu_ops: u32,
    pub max_nodes: u32,
    pub max_slabs: u32,
    pub max_textures: u32,
    pub max_alu_ops: u32,
    pub nodes_over: bool,
    pub slabs_over: bool,
    pub textures_over: bool,
    pub alu_over: bool,
    pub over_budget: bool,
}

impl From<inf_material::ComplexityReport> for ComplexityDto {
    fn from(r: inf_material::ComplexityReport) -> Self {
        Self {
            nodes: r.nodes,
            slabs: r.slabs,
            textures: r.textures,
            est_alu_ops: r.est_alu_ops,
            max_nodes: r.budget.max_nodes,
            max_slabs: r.budget.max_slabs,
            max_textures: r.budget.max_textures,
            max_alu_ops: r.budget.max_alu_ops,
            nodes_over: r.nodes_over(),
            slabs_over: r.slabs_over(),
            textures_over: r.textures_over(),
            alu_over: r.alu_over(),
            over_budget: r.over_budget(),
        }
    }
}

/// Result of compiling a material graph to WGSL + a preview render.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaterialCompileResult {
    /// The complete, naga-validated module ("View Generated WGSL").
    pub wgsl: String,
    /// Whether the graph compiled without errors.
    pub ok: bool,
    pub issues: Vec<MatIssueDto>,
    /// How many `tex.sample` slots the graph references.
    pub texture_count: u32,
    /// The preview sphere as a PNG data-URL (`None` if compile failed or no GPU).
    pub image: Option<String>,
    /// Advisory material-complexity budget report (P13.2).
    pub complexity: ComplexityDto,
}

impl MaterialState {
    fn with<R>(
        &self,
        f: impl FnOnce(&mut MaterialStore) -> Result<R, String>,
    ) -> Result<R, String> {
        let mut store = self.inner.lock().map_err(|e| e.to_string())?;
        f(&mut store)
    }

    /// Drop a document + its undo journal from the workspace. Idempotent
    /// (closing an unknown id is a no-op). The shared preview `Thumbnailer` is
    /// not per-doc, so it is left intact.
    fn close(&self, id: &str) -> Result<(), String> {
        self.with(|s| {
            s.docs.remove(id);
            s.journals.remove(id);
            Ok(())
        })
    }
}

/// The material node palette.
#[tauri::command]
pub async fn material_registry(state: State<'_, MaterialState>) -> Result<Vec<NodeDef>, String> {
    state.with(|s| Ok(s.registry.dtos()))
}

/// All open material documents.
#[tauri::command]
pub async fn material_list(state: State<'_, MaterialState>) -> Result<Vec<GraphDoc>, String> {
    state.with(|s| Ok(s.docs.values().cloned().collect()))
}

/// Create a new, empty material graph seeded with an output node.
#[tauri::command]
pub async fn material_create(
    app: AppHandle,
    name: String,
    state: State<'_, MaterialState>,
) -> Result<GraphDoc, String> {
    let doc = state.with(|s| {
        s.counter += 1;
        let id = format!("mat:{}", s.counter);
        let mut graph = inf_graph::Graph::empty();
        // Seed the mandatory Material Output sink so a fresh graph compiles.
        let out_id = inf_graph::NodeId(graph.next_id);
        apply_edits(
            &mut graph,
            &s.registry,
            &[GraphEdit::AddNode {
                id: out_id,
                type_id: "output.surface".into(),
                x: 420.0,
                y: 180.0,
                params: Default::default(),
            }],
        );
        let doc = GraphDoc {
            id: id.clone(),
            name: if name.is_empty() {
                format!("Material {}", s.counter)
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
    let _ = app.emit("material://sync", doc.id.clone());
    Ok(doc)
}

/// Fetch one document.
#[tauri::command]
pub async fn material_get(id: String, state: State<'_, MaterialState>) -> Result<GraphDoc, String> {
    state.with(|s| {
        s.docs
            .get(&id)
            .cloned()
            .ok_or_else(|| format!("no material `{id}`"))
    })
}

/// Close a document: free its graph + undo journal so open material graphs don't
/// accumulate for the life of the session. Called when the editing surface is
/// discarded.
#[tauri::command]
pub async fn material_close(id: String, state: State<'_, MaterialState>) -> Result<(), String> {
    state.close(&id)
}

/// Apply an edit batch; records one undo entry and re-validates.
#[tauri::command]
pub async fn material_apply(
    app: AppHandle,
    id: String,
    edits: Vec<GraphEdit>,
    label: String,
    state: State<'_, MaterialState>,
) -> Result<MaterialApplyResult, String> {
    let result = state.with(|s| {
        let registry = &s.registry;
        let doc = s
            .docs
            .get_mut(&id)
            .ok_or_else(|| format!("no material `{id}`"))?;
        let journal = s.journals.get_mut(&id).ok_or("missing journal")?;
        journal.record(label, doc.graph.clone());
        apply_edits(&mut doc.graph, registry, &edits);
        let issues = validate(&doc.graph, registry);
        Ok(MaterialApplyResult {
            issues,
            can_undo: journal.can_undo(),
            can_redo: journal.can_redo(),
        })
    })?;
    let _ = app.emit("material://sync", id);
    Ok(result)
}

/// Undo the last edit group.
#[tauri::command]
pub async fn material_undo(
    app: AppHandle,
    id: String,
    state: State<'_, MaterialState>,
) -> Result<Option<GraphDoc>, String> {
    let doc = state.with(|s| {
        let doc = s
            .docs
            .get_mut(&id)
            .ok_or_else(|| format!("no material `{id}`"))?;
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
        let _ = app.emit("material://sync", id);
    }
    Ok(doc)
}

/// Redo the last undone edit group.
#[tauri::command]
pub async fn material_redo(
    app: AppHandle,
    id: String,
    state: State<'_, MaterialState>,
) -> Result<Option<GraphDoc>, String> {
    let doc = state.with(|s| {
        let doc = s
            .docs
            .get_mut(&id)
            .ok_or_else(|| format!("no material `{id}`"))?;
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
        let _ = app.emit("material://sync", id);
    }
    Ok(doc)
}

/// Compile the material graph to WGSL (naga-validated), mapping errors back onto
/// nodes, and render a live preview sphere to a PNG data-URL.
#[tauri::command]
pub async fn material_compile(
    id: String,
    state: State<'_, MaterialState>,
) -> Result<MaterialCompileResult, String> {
    // Compile under the store lock, then render the preview outside it.
    let (wgsl, surface, ok, issues, tex_count, complexity) = state.with(|s| {
        let doc = s
            .docs
            .get(&id)
            .ok_or_else(|| format!("no material `{id}`"))?;
        let c = emit_wgsl(&doc.graph, &s.registry);
        let issues: Vec<MatIssueDto> = c
            .issues
            .iter()
            .map(|i| MatIssueDto {
                node: i.node,
                severity: match i.severity {
                    MatSeverity::Error => "error".into(),
                    MatSeverity::Warning => "warning".into(),
                },
                message: i.message.clone(),
            })
            .collect();
        Ok((
            c.module_wgsl,
            c.surface_wgsl,
            c.ok,
            issues,
            c.textures.len() as u32,
            ComplexityDto::from(c.complexity),
        ))
    })?;

    // A preview failure is a VALUE, not a crash and not silence (P23.2a audit —
    // M3). The composed module (this graph's surface + the preview wrapper) is
    // what wgpu compiles, and a symbol collision with the wrapper's `pv_*`
    // names would abort the editor on the default handler; `PreviewSession`
    // naga-validates it first and hands the error back. It rides the existing
    // diagnostics list as a warning — the GRAPH compiled (`ok` is true), only
    // its picture did not — with `node: None`, because nothing in the graph is
    // to blame in a way the canvas could highlight.
    let mut issues = issues;
    let image = if ok {
        let mut prev = state.preview.lock().map_err(|e| e.to_string())?;
        match prev.render_material_preview(&surface, tex_count, 256) {
            Ok(rgba) => rgba
                .and_then(|rgba| encode_png(256, &rgba).ok())
                .map(|png| format!("data:image/png;base64,{}", super::assets::base64(&png))),
            Err(msg) => {
                tracing::warn!("material_compile: {msg}");
                issues.push(MatIssueDto {
                    node: None,
                    severity: "warning".into(),
                    message: format!("preview failed: {msg}"),
                });
                None
            }
        }
    } else {
        None
    };

    Ok(MaterialCompileResult {
        wgsl,
        ok,
        issues,
        texture_count: tex_count,
        image,
        complexity,
    })
}

/// Bake the material graph to a `.inf_tex` texture asset via a compute pass
/// (P7.3). Evaluates the graph per texel into an offscreen storage texture and
/// writes the result as a new texture asset. Returns the new asset id.
#[tauri::command]
pub async fn material_bake(
    id: String,
    size: u32,
    name: String,
    state: State<'_, MaterialState>,
    assets: State<'_, super::assets::AssetState>,
) -> Result<String, String> {
    let (compute_wgsl, ok, tex_count) = state.with(|s| {
        let doc = s
            .docs
            .get(&id)
            .ok_or_else(|| format!("no material `{id}`"))?;
        let c = emit_texture_compute(&doc.graph, &s.registry);
        Ok((c.compute_wgsl, c.ok, c.texture_count))
    })?;
    if !ok {
        return Err("material has compile errors; fix them before baking".into());
    }
    let size = size.clamp(16, 2048);
    let rgba = {
        let mut prev = state.preview.lock().map_err(|e| e.to_string())?;
        prev.bake_texture(&compute_wgsl, tex_count, size)
            .ok_or("no GPU adapter available to bake")?
    };
    // A **v2 tiled** image (P26.1), not the bincode record: a texture the editor
    // produces from pixels goes down the same container every import does, so
    // there is one shape for the virtual-texture streamer to page.
    let tex = inf_material::build_tiled_texture(rgba, size, size, TextureImportSettings::default())
        .map_err(|e| e.to_string())?;
    let name = if name.is_empty() {
        "Baked Texture".to_string()
    } else {
        name
    };
    let asset_id = assets.write_texture_asset(&name, &tex)?;
    Ok(asset_id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Insert a bare document + journal directly (mimics `material_create`, which
    /// needs a Tauri `State`/`AppHandle` we can't build in a unit test).
    fn seed(state: &MaterialState, id: &str) {
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
        let state = MaterialState::default();
        seed(&state, "mat:1");
        state.close("mat:1").unwrap();
        state
            .with(|s| {
                assert!(s.docs.is_empty(), "doc freed");
                assert!(s.journals.is_empty(), "journal freed");
                Ok(())
            })
            .unwrap();
    }
}
