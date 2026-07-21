//! Scene / world commands (Phase 3): the editor↔world binding.
//!
//! The authoritative [`SceneDoc`] lives here behind an `Arc<Mutex<…>>` shared
//! with the native viewport thread (picking + gizmo writeback go through the
//! same document — the single source of truth). The frontend Outliner/Details
//! never touch the world directly: they call these commands and consume the
//! full [`SceneSnapshot`] (`scene_snapshot`) plus incremental `world://delta`
//! events after every mutation.

use std::sync::{Arc, Mutex};

use inf_editor_core::ipc::{DetailsDto, PropValueDto, SceneSnapshot, SpawnKind};
use inf_editor_core::scene::{details, diff, serialize, SceneDoc};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

/// Shared scene state: the document (also handed to the viewport thread) plus
/// the last snapshot we emitted, so mutations can ship a minimal delta.
pub struct SceneState {
    pub doc: Arc<Mutex<SceneDoc>>,
    last: Mutex<Option<SceneSnapshot>>,
}

impl Default for SceneState {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneState {
    pub fn new() -> Self {
        Self {
            doc: Arc::new(Mutex::new(SceneDoc::with_demo())),
            last: Mutex::new(None),
        }
    }
}

/// On startup, if a crash-recovery file survived (unclean exit), load it into
/// the shared document so no work is lost (P3.5.4).
pub fn recover_scene_on_boot(app: &AppHandle) {
    let Ok(dir) = app.path().app_data_dir() else {
        return;
    };
    if let Some(recovered) = serialize::take_recovery(&dir) {
        if let Some(state) = app.try_state::<SceneState>() {
            if let Ok(mut doc) = state.doc.lock() {
                *doc = recovered;
                tracing::info!("recovered unsaved scene from crash-recovery file");
            }
        }
    }
}

fn parse_guids(v: &[String]) -> Vec<Uuid> {
    v.iter().filter_map(|s| Uuid::parse_str(s).ok()).collect()
}

fn lock(doc: &Arc<Mutex<SceneDoc>>) -> Result<std::sync::MutexGuard<'_, SceneDoc>, String> {
    doc.lock().map_err(|e| e.to_string())
}

/// Recompute the snapshot and emit the delta against the last emitted one.
/// Emits globally (`app.emit`) so it reaches the main webview whether the caller
/// is a command or the viewport thread (a pick/gizmo edit, via the event sink).
pub fn emit_world_delta(app: &AppHandle, state: &SceneState) {
    let next = match lock(&state.doc) {
        Ok(mut doc) => doc.snapshot(),
        Err(_) => return,
    };
    let mut last = state.last.lock().expect("last snapshot lock");
    let delta = match last.as_ref() {
        Some(prev) => diff(prev, &next),
        None => diff(&empty_snapshot(), &next),
    };
    *last = Some(next);
    drop(last);
    if let Err(e) = app.emit("world://delta", delta) {
        tracing::warn!("world://delta emit failed: {e}");
    }
}

fn empty_snapshot() -> SceneSnapshot {
    SceneSnapshot {
        version: 0,
        roots: Vec::new(),
        nodes: Vec::new(),
        selection: Vec::new(),
        dirty: false,
        title: String::new(),
        can_undo: false,
        can_redo: false,
        undo_label: None,
        redo_label: None,
    }
}

// ── reads ────────────────────────────────────────────────────────────────

/// Full snapshot (Outliner load + resync). Seeds the delta baseline.
#[tauri::command]
pub async fn scene_snapshot(state: State<'_, SceneState>) -> Result<SceneSnapshot, String> {
    let snap = lock(&state.doc)?.snapshot();
    *state.last.lock().map_err(|e| e.to_string())? = Some(snap.clone());
    Ok(snap)
}

/// Details view of the current selection.
#[tauri::command]
pub async fn scene_details(state: State<'_, SceneState>) -> Result<DetailsDto, String> {
    let doc = lock(&state.doc)?;
    Ok(details::build(&doc))
}

// ── structural mutations ───────────────────────────────────────────────────

#[tauri::command]
pub async fn scene_create(
    app: AppHandle,
    state: State<'_, SceneState>,
    kind: SpawnKind,
    parent: Option<String>,
) -> Result<String, String> {
    let parent = parent.and_then(|s| Uuid::parse_str(&s).ok());
    let guid = {
        let mut doc = lock(&state.doc)?;
        let guid = doc.edit_create(kind, "", parent);
        doc.select(&[guid], false);
        guid
    };
    emit_world_delta(&app, &state);
    Ok(guid.to_string())
}

/// Spawn a dropped Content-Drawer asset into the scene (drag-to-viewport
/// handoff, P4.4). A real, selectable, saveable entity named after the asset is
/// created; it renders as a placeholder primitive today — resolving the mesh
/// asset to its imported geometry in the interactive viewport is the documented
/// Phase 4→7 follow-up (the thumbnailer already renders the real geometry).
#[tauri::command]
pub async fn scene_spawn_asset(
    app: AppHandle,
    state: State<'_, SceneState>,
    assets: State<'_, super::assets::AssetState>,
    asset_id: String,
) -> Result<String, String> {
    let id = asset_id
        .parse::<inf_asset::AssetId>()
        .map_err(|e| e.to_string())?;
    let name = assets.asset_name(id).unwrap_or_else(|| "Asset".to_string());
    let guid = {
        let mut doc = lock(&state.doc)?;
        let guid = doc.edit_create(SpawnKind::Cube, &name, None);
        doc.select(&[guid], false);
        guid
    };
    emit_world_delta(&app, &state);
    Ok(guid.to_string())
}

/// Apply a material asset's PBR parameters to entities (Content-Drawer
/// apply-by-drag / "Apply to Selection", P7.1). `targets` defaults to the
/// current selection. Returns how many entities were updated.
#[tauri::command]
pub async fn scene_apply_material(
    app: AppHandle,
    state: State<'_, SceneState>,
    assets: State<'_, super::assets::AssetState>,
    asset_id: String,
    targets: Option<Vec<String>>,
) -> Result<usize, String> {
    let id = asset_id
        .parse::<inf_asset::AssetId>()
        .map_err(|e| e.to_string())?;
    let mat = assets
        .load_material(id)
        .ok_or_else(|| "asset is not a material".to_string())?;
    let applied = {
        let mut doc = lock(&state.doc)?;
        let targets: Vec<Uuid> = match targets {
            Some(list) => list
                .iter()
                .filter_map(|s| Uuid::parse_str(s).ok())
                .collect(),
            None => doc.selection().to_vec(),
        };
        doc.edit_apply_material(
            &targets,
            mat.base_color,
            mat.metallic,
            mat.roughness,
            mat.emissive,
        )
    };
    if applied > 0 {
        emit_world_delta(&app, &state);
    }
    Ok(applied)
}

#[tauri::command]
pub async fn scene_delete(
    app: AppHandle,
    state: State<'_, SceneState>,
    guids: Vec<String>,
) -> Result<(), String> {
    lock(&state.doc)?.edit_delete(&parse_guids(&guids));
    emit_world_delta(&app, &state);
    Ok(())
}

#[tauri::command]
pub async fn scene_rename(
    app: AppHandle,
    state: State<'_, SceneState>,
    guid: String,
    name: String,
) -> Result<(), String> {
    if let Ok(g) = Uuid::parse_str(&guid) {
        lock(&state.doc)?.edit_rename(g, &name);
        emit_world_delta(&app, &state);
    }
    Ok(())
}

#[tauri::command]
pub async fn scene_reparent(
    app: AppHandle,
    state: State<'_, SceneState>,
    guid: String,
    parent: Option<String>,
) -> Result<bool, String> {
    let Ok(g) = Uuid::parse_str(&guid) else {
        return Ok(false);
    };
    let parent = parent.and_then(|s| Uuid::parse_str(&s).ok());
    let ok = lock(&state.doc)?.edit_reparent(g, parent);
    emit_world_delta(&app, &state);
    Ok(ok)
}

#[tauri::command]
pub async fn scene_set_visible(
    app: AppHandle,
    state: State<'_, SceneState>,
    guid: String,
    visible: bool,
) -> Result<(), String> {
    if let Ok(g) = Uuid::parse_str(&guid) {
        lock(&state.doc)?.edit_set_visible(g, visible);
        emit_world_delta(&app, &state);
    }
    Ok(())
}

#[tauri::command]
pub async fn scene_select(
    app: AppHandle,
    state: State<'_, SceneState>,
    guids: Vec<String>,
    additive: bool,
) -> Result<(), String> {
    lock(&state.doc)?.select(&parse_guids(&guids), additive);
    emit_world_delta(&app, &state);
    Ok(())
}

// ── property edits (Details) ───────────────────────────────────────────────

/// Set one field on every selected object in a single undo transaction.
#[tauri::command]
pub async fn scene_set_property(
    app: AppHandle,
    state: State<'_, SceneState>,
    guids: Vec<String>,
    type_path: String,
    field: String,
    value: PropValueDto,
) -> Result<DetailsDto, String> {
    let pv = details::from_dto(&value);
    let targets = parse_guids(&guids);
    let details = {
        let mut doc = lock(&state.doc)?;
        doc.begin_transaction(&format!("Edit {field}"));
        for g in &targets {
            doc.edit_set_prop(*g, &type_path, &field, &pv);
        }
        doc.commit_transaction();
        doc.details()
    };
    emit_world_delta(&app, &state);
    Ok(details)
}

/// Reset one field to its default on every selected object.
#[tauri::command]
pub async fn scene_reset_property(
    app: AppHandle,
    state: State<'_, SceneState>,
    guids: Vec<String>,
    type_path: String,
    field: String,
) -> Result<DetailsDto, String> {
    let targets = parse_guids(&guids);
    let details = {
        let mut doc = lock(&state.doc)?;
        doc.begin_transaction(&format!("Reset {field}"));
        for g in &targets {
            doc.edit_reset_prop(*g, &type_path, &field);
        }
        doc.commit_transaction();
        doc.details()
    };
    emit_world_delta(&app, &state);
    Ok(details)
}

// ── history ────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn scene_undo(app: AppHandle, state: State<'_, SceneState>) -> Result<(), String> {
    lock(&state.doc)?.undo();
    emit_world_delta(&app, &state);
    Ok(())
}

#[tauri::command]
pub async fn scene_redo(app: AppHandle, state: State<'_, SceneState>) -> Result<(), String> {
    lock(&state.doc)?.redo();
    emit_world_delta(&app, &state);
    Ok(())
}

// ── files ──────────────────────────────────────────────────────────────────

/// The app-data dir (created if needed) — where quicksaves + the recovery file
/// live until the P5 project system gives levels real project paths.
fn data_dir(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    let dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// Resolve a save path: the given one, or a default quicksave in the data dir.
fn resolve_path(app: &AppHandle, path: Option<String>) -> Result<std::path::PathBuf, String> {
    match path {
        Some(p) if !p.is_empty() => Ok(std::path::PathBuf::from(p)),
        _ => Ok(data_dir(app)?.join("quicksave.inf_lvl")),
    }
}

#[tauri::command]
pub async fn scene_save(
    app: AppHandle,
    state: State<'_, SceneState>,
    path: Option<String>,
) -> Result<(), String> {
    let path = resolve_path(&app, path)?;
    {
        let mut doc = lock(&state.doc)?;
        serialize::save(&doc, &path, None)?;
        doc.mark_saved();
    }
    // A clean save invalidates the crash-recovery file.
    if let Ok(dir) = data_dir(&app) {
        serialize::clear_recovery(&dir);
    }
    emit_world_delta(&app, &state);
    Ok(())
}

#[tauri::command]
pub async fn scene_open(
    app: AppHandle,
    state: State<'_, SceneState>,
    path: Option<String>,
) -> Result<SceneSnapshot, String> {
    let path = resolve_path(&app, path)?;
    let loaded = serialize::load(&path)?;
    let snap = {
        let mut doc = lock(&state.doc)?;
        *doc = loaded;
        doc.snapshot()
    };
    emit_world_delta(&app, &state);
    Ok(snap)
}

/// Write the crash-recovery file (frontend calls this on a debounce while
/// there are unsaved changes; P3.5.4).
#[tauri::command]
pub async fn scene_autosave(app: AppHandle, state: State<'_, SceneState>) -> Result<(), String> {
    let dir = data_dir(&app)?;
    let doc = lock(&state.doc)?;
    if doc.is_dirty() {
        serialize::write_recovery(&doc, &dir)?;
    }
    Ok(())
}

#[tauri::command]
pub async fn scene_new(
    app: AppHandle,
    state: State<'_, SceneState>,
) -> Result<SceneSnapshot, String> {
    let snap = {
        let mut doc = lock(&state.doc)?;
        *doc = SceneDoc::new();
        doc.snapshot()
    };
    emit_world_delta(&app, &state);
    Ok(snap)
}
