//! Scene / world commands (Phase 3): the editor↔world binding.
//!
//! The authoritative [`SceneDoc`] lives here behind an `Arc<Mutex<…>>` shared
//! with the native viewport thread (picking + gizmo writeback go through the
//! same document — the single source of truth). The frontend Outliner/Details
//! never touch the world directly: they call these commands and consume the
//! full [`SceneSnapshot`] (`scene_snapshot`) plus incremental `world://delta`
//! events after every mutation.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use inf_editor_core::ipc::{
    AddableComponentDto, DetailsDto, LevelSettingsDto, PropValueDto, SaveResultDto, SceneSnapshot,
    SpawnKind, TilemapCellDto, TilemapDto,
};
use inf_editor_core::scene::serialize::EntityRecord;
use inf_editor_core::scene::{details, diff, serialize, tilemap, SceneDoc};
use inf_editor_core::terrain_edit;
use inf_editor_core::voxel_edit;
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

/// Shared scene state: the document (also handed to the viewport thread) plus
/// the last snapshot we emitted, so mutations can ship a minimal delta. Also
/// holds the entity clipboard (copy / cut → paste) and the path the current
/// level last saved to (so a plain Save overwrites it and Save As can seed its
/// dialog).
pub struct SceneState {
    pub doc: Arc<Mutex<SceneDoc>>,
    last: Mutex<Option<SceneSnapshot>>,
    /// Backend-held entity clipboard — a forest of records (parents-first). Does
    /// not cross a process boundary, so it needn't serialize.
    clipboard: Mutex<Vec<EntityRecord>>,
    /// The `.inf_lvl` path the level was last opened from / saved to (`None` for
    /// an untitled or freshly-`New`ed scene). A Save-with-explicit-path records
    /// it; a plain Save (no path) writes here when set, else the quicksave.
    current_level_path: Mutex<Option<PathBuf>>,
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
            clipboard: Mutex::new(Vec::new()),
            current_level_path: Mutex::new(None),
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

/// `pub(super)` since P24.5: the New Character wizard's command spawns its actor
/// through `SceneDoc::edit_create_character` and then emits the delta, which is
/// exactly what `scene_create` does one module over. A second lock helper would
/// be a second place the poison-to-`String` mapping could drift.
pub(super) fn lock(
    doc: &Arc<Mutex<SceneDoc>>,
) -> Result<std::sync::MutexGuard<'_, SceneDoc>, String> {
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
    // Hold `last` across the compare-store so two emitters (a command thread and
    // the viewport thread firing `WorldChanged`) can't interleave and ship a
    // backward delta. The snapshot's monotonically-increasing `version` is the
    // guard: if what we snapshotted is STRICTLY older than the last emitted
    // state, a newer emit already won the race — drop this one, leaving `last`
    // un-regressed. (Equal versions still emit: a save clears `dirty` without
    // bumping the version, and that flag change must reach the frontend.
    // `scene_open` / `scene_new` reset `last` to `None`, so a fresh,
    // lower-versioned document still emits its full delta.)
    let mut last = state.last.lock().expect("last snapshot lock");
    if let Some(prev) = last.as_ref() {
        if next.version < prev.version {
            return;
        }
    }
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
    // P26.3b audit: the binding names the ROOT `.inf_mat`, not necessarily the
    // asset that was dropped. Apply-by-drag accepts a `.inf_mati` (the Content
    // Drawer routes `material_instance` here, and so does the instance editor's
    // "Apply to Selection"), and a binding naming an instance resolves nowhere —
    // no `.inf_matd` is derived for it, its maps never enter the pack closure,
    // the PIE loader's kind check misses it, and nothing raises an advisory
    // because the asset IS in the project. `MatOverrides` carries no texture
    // slots, so an instance's maps ARE its parent's: the scalars below still come
    // from the resolved instance, and the binding names the material they were
    // resolved against.
    let binding = assets.material_binding_id(id);
    let applied = {
        let mut doc = lock(&state.doc)?;
        let targets: Vec<Uuid> = match targets {
            Some(list) => list
                .iter()
                .filter_map(|s| Uuid::parse_str(s).ok())
                .collect(),
            None => doc.selection().to_vec(),
        };
        // Map the inf-material blend enum onto the ECS `BlendMode` variant name
        // (inf-material has no inf-ecs dep — the mapping lives here, R-P5).
        let blend = match mat.blend {
            inf_material::MatBlend::Opaque => "Opaque",
            inf_material::MatBlend::Masked => "Masked",
            inf_material::MatBlend::Translucent => "Translucent",
        };
        doc.edit_apply_material(
            &targets,
            // P26.3b: the binding scene v22 persists. This is the ONE caller
            // that knows which `.inf_mat` the scalars below came from, which is
            // exactly why the reference was being thrown away here. `None` when
            // the drop was a material instance whose chain is broken — no
            // binding at all beats one that resolves nowhere.
            binding.map(|b| b.uuid()),
            mat.base_color,
            mat.metallic,
            mat.roughness,
            mat.emissive,
            blend,
            mat.alpha_cutoff,
        )
    };
    if applied > 0 {
        emit_world_delta(&app, &state);
    }
    Ok(applied)
}

/// Bind a `.inf_act` blueprint-class asset to entities (Content-Drawer drag a
/// blueprint onto an entity / "Bind Actor", P9.5). Sets the entity's `ActorClass`
/// link to the asset GUID as one undo step; `targets` defaults to the current
/// selection. Returns how many entities were (re)bound.
#[tauri::command]
pub async fn scene_apply_actor(
    app: AppHandle,
    state: State<'_, SceneState>,
    asset_id: String,
    targets: Option<Vec<String>>,
) -> Result<usize, String> {
    let id = asset_id
        .parse::<inf_asset::AssetId>()
        .map_err(|e| e.to_string())?;
    let applied = {
        let mut doc = lock(&state.doc)?;
        let targets: Vec<Uuid> = match targets {
            Some(list) => list
                .iter()
                .filter_map(|s| Uuid::parse_str(s).ok())
                .collect(),
            None => doc.selection().to_vec(),
        };
        doc.edit_apply_actor(&targets, id.uuid())
    };
    if applied > 0 {
        emit_world_delta(&app, &state);
    }
    Ok(applied)
}

/// Apply a texture's sprite-sheet slice to entities (Sprite Sheet panel "Apply
/// to Selection", P8.2a). Sets `Sprite.texture` + `Sprite.atlas_rect` (and,
/// when `resize_to_slice`, `Sprite.size` from the slice's pixel aspect) on the
/// targets (or the current selection) as one undo transaction. `slice` is a
/// resolved slice name (grid cells are index-named "0","1",…; manual rects use
/// their names). Returns how many entities were updated.
#[tauri::command]
pub async fn scene_apply_sprite_slice(
    app: AppHandle,
    state: State<'_, SceneState>,
    assets: State<'_, super::assets::AssetState>,
    asset_id: String,
    slice: String,
    targets: Option<Vec<String>>,
    resize_to_slice: bool,
) -> Result<usize, String> {
    let id = asset_id
        .parse::<inf_asset::AssetId>()
        .map_err(|e| e.to_string())?;
    let (slices, w, h) = assets.read_sprite_slices(id)?;
    let resolved = slices.resolve(w, h);
    let hit = resolved
        .iter()
        .find(|s| s.name == slice)
        .ok_or_else(|| format!("no slice `{slice}` in this sprite sheet"))?;

    // Opt-in resize: a unit-height quad whose width matches the slice's aspect.
    let size = if resize_to_slice {
        let ph = hit.px[3].max(1) as f64;
        Some([hit.px[2] as f64 / ph, 1.0])
    } else {
        None
    };
    let uv_min = hit.uv_min;
    let uv_max = hit.uv_max;

    let applied = {
        let mut doc = lock(&state.doc)?;
        let targets: Vec<Uuid> = match targets {
            Some(list) => list
                .iter()
                .filter_map(|s| Uuid::parse_str(s).ok())
                .collect(),
            None => doc.selection().to_vec(),
        };
        doc.edit_apply_sprite_slice(&targets, Some(id.uuid()), uv_min, uv_max, size)
    };
    if applied > 0 {
        emit_world_delta(&app, &state);
    }
    Ok(applied)
}

/// Read an entity's tilemap for the paint panel (P8.2b). `None` when the entity
/// has no `Tilemap`. When the tilemap's atlas texture carries a P8.2a
/// sprite-sheet grid, the palette dimensions are taken from that grid so the
/// palette matches the sliced atlas.
#[tauri::command]
pub async fn tilemap_get(
    state: State<'_, SceneState>,
    assets: State<'_, super::assets::AssetState>,
    entity: String,
) -> Result<Option<TilemapDto>, String> {
    let guid = Uuid::parse_str(&entity).map_err(|e| e.to_string())?;
    let mut dto = {
        let doc = lock(&state.doc)?;
        tilemap::build_dto(&doc, guid)
    };
    if let Some(d) = dto.as_mut() {
        if let Some(tex) = &d.texture {
            if let Ok(id) = tex.parse::<inf_asset::AssetId>() {
                if let Ok((slices, _, _)) = assets.read_sprite_slices(id) {
                    if let Some(g) = slices.grid {
                        d.palette_cols = g.columns.max(1);
                        d.palette_rows = g.rows.max(1);
                    }
                }
            }
        }
    }
    Ok(dto)
}

/// Paint one stroke (a mouse-down→up batch, or a bounded flood-fill) onto an
/// entity's tilemap as a **single** undo step (P8.2b). `cells` carries the new
/// 1-based atlas index per cell (`0` erases). Returns how many cells actually
/// changed; emits `world://delta` when non-zero so other views re-sync.
#[tauri::command]
pub async fn tilemap_paint(
    app: AppHandle,
    state: State<'_, SceneState>,
    entity: String,
    cells: Vec<TilemapCellDto>,
) -> Result<usize, String> {
    let guid = Uuid::parse_str(&entity).map_err(|e| e.to_string())?;
    let batch: Vec<(i32, i32, u32)> = cells.iter().map(|c| (c.x, c.y, c.tile)).collect();
    let n = lock(&state.doc)?.edit_paint_tiles(guid, &batch);
    if n > 0 {
        emit_world_delta(&app, &state);
    }
    Ok(n)
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

/// Resolve the targets for a duplicate/copy/cut command: the given guids, or the
/// current selection when omitted (mirrors `scene_apply_material`).
fn targets_or_selection(doc: &SceneDoc, guids: Option<Vec<String>>) -> Vec<Uuid> {
    match guids {
        Some(list) => parse_guids(&list),
        None => doc.selection().to_vec(),
    }
}

/// Duplicate the selected (or explicitly-given) roots as one undo step: fresh
/// GUIDs, internal hierarchy preserved, each copy a sibling, root names
/// " Copy"-suffixed. Selects and returns the new root GUIDs.
#[tauri::command]
pub async fn scene_duplicate(
    app: AppHandle,
    state: State<'_, SceneState>,
    guids: Option<Vec<String>>,
) -> Result<Vec<String>, String> {
    let new_roots = {
        let mut doc = lock(&state.doc)?;
        let targets = targets_or_selection(&doc, guids);
        if targets.is_empty() {
            return Ok(Vec::new());
        }
        let new_roots = doc.edit_duplicate(&targets);
        if !new_roots.is_empty() {
            doc.select(&new_roots, false);
        }
        new_roots
    };
    if !new_roots.is_empty() {
        emit_world_delta(&app, &state);
    }
    Ok(new_roots.iter().map(|g| g.to_string()).collect())
}

/// Copy the selected (or given) roots into the backend clipboard. Returns how
/// many entity records were captured (subtree-expanded, nested-selection deduped).
#[tauri::command]
pub async fn scene_copy(
    state: State<'_, SceneState>,
    guids: Option<Vec<String>>,
) -> Result<usize, String> {
    let records = {
        let doc = lock(&state.doc)?;
        let targets = targets_or_selection(&doc, guids);
        doc.collect_subtree_records(&targets)
    };
    let n = records.len();
    *state.clipboard.lock().map_err(|e| e.to_string())? = records;
    Ok(n)
}

/// Cut = copy into the clipboard + delete the originals (the delete is the
/// undoable half — one Delete step). Returns how many records were captured.
#[tauri::command]
pub async fn scene_cut(
    app: AppHandle,
    state: State<'_, SceneState>,
    guids: Option<Vec<String>>,
) -> Result<usize, String> {
    let n = {
        let mut doc = lock(&state.doc)?;
        let targets = targets_or_selection(&doc, guids);
        let records = doc.collect_subtree_records(&targets);
        if records.is_empty() {
            return Ok(0);
        }
        let n = records.len();
        *state.clipboard.lock().map_err(|e| e.to_string())? = records;
        doc.edit_delete(&targets);
        n
    };
    emit_world_delta(&app, &state);
    Ok(n)
}

/// Paste the clipboard's records with fresh GUIDs as one undo step; selects and
/// returns the new root GUIDs (empty when the clipboard is empty).
#[tauri::command]
pub async fn scene_paste(
    app: AppHandle,
    state: State<'_, SceneState>,
) -> Result<Vec<String>, String> {
    let records = state.clipboard.lock().map_err(|e| e.to_string())?.clone();
    if records.is_empty() {
        return Ok(Vec::new());
    }
    let new_roots = {
        let mut doc = lock(&state.doc)?;
        let new_roots = doc.edit_paste_records(&records);
        if !new_roots.is_empty() {
            doc.select(&new_roots, false);
        }
        new_roots
    };
    if !new_roots.is_empty() {
        emit_world_delta(&app, &state);
    }
    Ok(new_roots.iter().map(|g| g.to_string()).collect())
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

// ── add / remove component (E-P1) ──────────────────────────────────────────

/// The components the user may add via "+ Add Component" (registry order).
#[tauri::command]
pub async fn scene_list_addable_components(
    state: State<'_, SceneState>,
) -> Result<Vec<AddableComponentDto>, String> {
    let doc = lock(&state.doc)?;
    Ok(doc
        .world()
        .registry()
        .addable()
        .map(|c| AddableComponentDto {
            type_path: c.type_path.to_string(),
            display: c.display.to_string(),
        })
        .collect())
}

/// Add a `Default` instance of `type_path` to every target (selected or given)
/// entity in one undo transaction. Returns the refreshed Details view.
#[tauri::command]
pub async fn scene_add_component(
    app: AppHandle,
    state: State<'_, SceneState>,
    guids: Option<Vec<String>>,
    type_path: String,
) -> Result<DetailsDto, String> {
    let details = {
        let mut doc = lock(&state.doc)?;
        let targets = targets_or_selection(&doc, guids);
        doc.begin_transaction("Add Component");
        for g in &targets {
            doc.edit_add_component(*g, &type_path);
        }
        doc.commit_transaction();
        doc.details()
    };
    emit_world_delta(&app, &state);
    Ok(details)
}

/// Remove `type_path` from a single entity (one undo step). Returns the refreshed
/// Details view.
#[tauri::command]
pub async fn scene_remove_component(
    app: AppHandle,
    state: State<'_, SceneState>,
    guid: String,
    type_path: String,
) -> Result<DetailsDto, String> {
    let g = Uuid::parse_str(&guid).map_err(|e| e.to_string())?;
    let details = {
        let mut doc = lock(&state.doc)?;
        doc.edit_remove_component(g, &type_path);
        doc.details()
    };
    emit_world_delta(&app, &state);
    Ok(details)
}

/// The default value for a single element of the list at `path` on `type_path`'s
/// component — powers the ListField "add element" button. Errors if the path is
/// not a list.
#[tauri::command]
pub async fn scene_list_default(
    state: State<'_, SceneState>,
    type_path: String,
    path: String,
) -> Result<PropValueDto, String> {
    let doc = lock(&state.doc)?;
    doc.list_default(&type_path, &path)
        .map(|v| details::to_dto(&v))
        .ok_or_else(|| format!("{path} on {type_path} is not a list"))
}

// ── world settings (R-P4) ───────────────────────────────────────────────────

/// The level's file-level settings (gravity / sim rate / the persisted render
/// block) for the editable World Settings panel.
#[tauri::command]
pub async fn scene_get_settings(state: State<'_, SceneState>) -> Result<LevelSettingsDto, String> {
    let doc = lock(&state.doc)?;
    Ok(LevelSettingsDto::from_doc(&doc))
}

/// Replace the level settings as one undo step (World Settings panel; the
/// frontend debounces the calls). Emits `world://delta` so the panel + any
/// listener re-sync. The host applies the render block to the live viewport on
/// the resulting version bump, and re-projects the sun on the same bump (P17.1).
#[tauri::command]
pub async fn scene_set_settings(
    app: AppHandle,
    state: State<'_, SceneState>,
    settings: LevelSettingsDto,
) -> Result<(), String> {
    {
        let mut doc = lock(&state.doc)?;
        doc.edit_settings(settings.to_settings());
        // The time-of-day half lives on the sky-authority ENTITY, not in the
        // settings record, so it is a separate (also single-step) edit. `present`
        // rides through as the **create** flag: the panel sets it on any time-of-day
        // row (opting the level into a dynamic sun), and leaves it `false` when it
        // is merely echoing back the previewed defaults — so editing gravity can
        // never conjure a sun as a side effect. The guard itself lives in
        // `SceneDoc::edit_time_of_day`, where it is unit-testable.
        doc.edit_time_of_day(
            settings.time_of_day.to_component(),
            settings.time_of_day.present,
        );
        // The atmosphere (P17.2) rides the same authority entity and the same
        // create guard. It overlays the level's CURRENT component rather than
        // rebuilding one, so the `Color` fields the DTO deliberately omits (sun,
        // moon and the three gradient colours, all edited in Details) survive a
        // fog-slider drag.
        let base = doc.sky_atmosphere().unwrap_or_default();
        doc.edit_sky_atmosphere(
            settings.atmosphere.to_component(base),
            settings.atmosphere.present,
        );
        // The weather block writes third, onto whatever the atmosphere edit just
        // left: both project the SAME component, so re-reading the base here is
        // what stops the second write from resurrecting the first's old values.
        let base = doc.sky_atmosphere().unwrap_or_default();
        doc.edit_weather(
            settings.weather.to_component(base),
            settings.weather.present,
        );
    }
    emit_world_delta(&app, &state);
    Ok(())
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

/// Resolve where a Save writes and whether it should update the stored
/// current-level path. Precedence: an explicit path wins (and BECOMES the new
/// current, i.e. Save As); otherwise the stored current path (a plain Save
/// overwrites the level in place); otherwise the quicksave fallback. Returns
/// `(target, new_current)` where `new_current` is `Some` only when the explicit
/// path should replace the stored one. Pure (no `AppHandle`) so it is unit-tested.
fn resolve_save_target(
    explicit: Option<&std::path::Path>,
    current: Option<&std::path::Path>,
    quicksave: &std::path::Path,
) -> (PathBuf, Option<PathBuf>) {
    match explicit {
        Some(p) => (p.to_path_buf(), Some(p.to_path_buf())),
        None => (current.unwrap_or(quicksave).to_path_buf(), None),
    }
}

/// What a poisoned voxel working set means to a save, said once so the save path,
/// autosave and the tests quote the same sentence.
const POISONED_VOXEL_STORE_SAVE_FAILURE: &str =
    "Carve edits could NOT be staged: the voxel working set is poisoned — a thread panicked \
     while holding it. The level itself was written, but no .inf_voxel was updated, so every \
     cave carved this session is still only in memory. The crash-recovery file has been kept.";

/// Stage the voxel half of a save out of the shared working set (P21.2 audit).
///
/// **Three outcomes, and the middle one is why this is a function.** No viewport
/// (headless, or a build with no embedding backend) means no carve tool and
/// nothing to stage — the honest empty answer. A readable store stages its dirty
/// chunks. A **poisoned** store is neither: it is a save that cannot see the
/// edits it is supposed to write, and folding it into the headless arm — which is
/// what `match … { Some(Ok(s)) => …, _ => Vec::new() }` did — made a save that
/// silently dropped every carve report complete success and then delete the
/// crash-recovery file that was their last copy.
fn stage_voxels(
    volumes: Option<&inf_editor_core::voxel_store::SharedVoxelVolumes>,
) -> Result<Vec<voxel_edit::StagedVoxelEdit>, String> {
    match volumes {
        None => Ok(Vec::new()),
        Some(v) => match v.lock() {
            Ok(store) => Ok(voxel_edit::stage_voxel_edits(&store)),
            Err(_) => Err(POISONED_VOXEL_STORE_SAVE_FAILURE.to_string()),
        },
    }
}

/// Whether carve edits survived the save and the crash-recovery file must
/// therefore be kept (P21.2 audit).
///
/// **A poisoned store answers "still dirty".** It is the terrain twin's rule —
/// `has_unsaved_terrain_edits` is read through `lock(&state.doc)?`, which aborts
/// the save outright on a poisoned document — expressed the only way this seam
/// can: the question is "may the recovery file be deleted?", and an unreadable
/// store cannot answer *yes* to it. `unwrap_or(false)` said yes, so a poisoned
/// mutex deleted the only surviving copy of the session's carves.
fn voxels_still_dirty(volumes: Option<&inf_editor_core::voxel_store::SharedVoxelVolumes>) -> bool {
    volumes.is_some_and(|v| v.lock().map_or(true, |s| s.has_unsaved_edits()))
}

/// Mark the document saved — **after** a successful write, and only if it is
/// still the document that was written (C4-3/F2).
///
/// The version guard matters because the encode and the disk write deliberately
/// happen at different times, with the doc lock released in between so the
/// viewport can keep rendering. An edit that lands in that window is genuinely
/// *not* in the bytes on disk, so marking the document clean would lose it at
/// the next close-without-prompt exactly as the pre-write `mark_saved` did.
///
/// `version` moves on *any* mutation, including the non-dirtying
/// `write_prop_preview` a sequencer scrub uses, so the guard is conservative in
/// one direction: a scrub command landing on another thread during the exact
/// window of the write would leave the document dirty after a successful save.
/// That is the safe direction — an extra recovery write and a live Save button,
/// never a lost edit — and the alternative (an authored-edit counter beside
/// `version`) is new document state for a case that costs nothing when it fires.
/// `scene_save`'s body has no `.await` between the encode and this call, so the
/// realistic mutator in that window is the viewport thread, whose gizmo
/// writeback is a genuine authored edit and exactly what this must catch.
fn mark_saved_if_unchanged(state: &SceneState, saved_version: u64) -> Result<(), String> {
    let mut doc = lock(&state.doc)?;
    if doc.version() == saved_version {
        doc.mark_saved();
    } else {
        tracing::info!(
            "scene_save: the document changed while it was being written (v{saved_version} → \
             v{}), so it stays dirty — the save is complete, it is just no longer the whole \
             document",
            doc.version()
        );
    }
    Ok(())
}

#[tauri::command]
pub async fn scene_save(
    app: AppHandle,
    state: State<'_, SceneState>,
    path: Option<String>,
) -> Result<SaveResultDto, String> {
    // Route the target: an explicit path (Save As) writes there and becomes the
    // current level; a plain Save (no path) overwrites the current level, else
    // the quicksave fallback.
    let explicit = match &path {
        Some(p) if !p.is_empty() => Some(std::path::PathBuf::from(p)),
        _ => None,
    };
    let quicksave = data_dir(&app)?.join("quicksave.inf_lvl");
    let (target, new_current) = {
        let current = state.current_level_path.lock().map_err(|e| e.to_string())?;
        resolve_save_target(explicit.as_deref(), current.as_deref(), &quicksave)
    };
    // Encode under the doc lock (with any live sequencer scrub rolled back to the
    // authored values), then write the files AFTER releasing the lock — the
    // viewport locks the same doc every frame, so disk IO must not be held under it.
    //
    // Streamed-terrain edits are staged in the same window and written in the same
    // "outside the lock" phase (P16.4b): a `.inf_terrain` rewrite is a
    // whole-payload operation and must never happen under the doc lock.
    //
    // **`mark_saved` runs AFTER the write, never before** — the `skel.rs` rule
    // (C4-3/F2). It used to run here, inside the encode block, so a write that
    // failed returned `Err` with the document already clean: the dirty indicator
    // lied, closing discarded the work without a prompt, and — worst — the
    // autosave gate below (`!doc.is_dirty()`) stopped writing the crash-recovery
    // file for work that then existed nowhere on disk.
    //
    // The version stamped here is what the mark is checked against afterwards,
    // so an edit that lands between the encode and the write leaves the document
    // dirty (it is: the bytes on disk do not contain it) instead of being
    // marked saved by a save that never saw it.
    let (enc, staged, saved_version) = {
        let mut doc = lock(&state.doc)?;
        let enc = doc.with_authored_scene(|d| serialize::encode_scene(d, None))?;
        let staged = terrain_edit::stage_terrain_edits(&doc);
        let version = doc.version();
        (enc, staged, version)
    };
    // The voxel half is staged at the same "save begins" moment, out of the
    // shared store the viewport carves into (schema v19 is frozen, so the chunks
    // are NOT in the document).
    //
    // **THE LOCK RULE: never hold the document and the volumes at the same
    // time** (P23.2a audit). The doc guard above ends with its block, so this
    // takes the store with nothing else held.
    //
    // The rule is *no overlap*, NOT an acquisition order, and the distinction is
    // load-bearing: the three sites that touch both genuinely differ in which
    // they take first — this one and `overlay_sim_carves` read the document
    // first, `scene_autosave` reads the store first — and that is only safe, and
    // only checkable site-by-site, because none of them ever holds two. An
    // earlier comment here called it "document first, volumes second", which
    // described this site and would have led a future author to "fix"
    // `scene_autosave` into the very overlap the rule forbids.
    //
    // `overlay_sim_carves` used to be a real exception (store held ACROSS the
    // document, i.e. the inverse order, both live for a whole loop — a genuine
    // two-lock deadlock shape). It now snapshots its bindings under the document
    // and releases before touching the store, so there is no exception left.
    //
    // **Resolved WITHOUT a viewport** (P23.2a — the hoist): the store is the
    // process's, held by `SharedStores`, so a save stages the author's carves
    // whether one viewport is open, three are, or none is. Before the hoist it
    // hung off whichever `ViewportHandle` happened to exist.
    let voxel_volumes = app
        .try_state::<crate::commands::SharedStores>()
        .map(|s| s.voxel_volumes());
    let staged_voxels = stage_voxels(voxel_volumes.as_ref());
    // The write, and only then the mark. A failed write returns here with the
    // document still dirty, so the indicator, the close prompt and the
    // crash-recovery autosave all keep telling the truth (C4-3/F2).
    serialize::write_encoded(&enc, &target)?;
    mark_saved_if_unchanged(&state, saved_version)?;

    // Fold this session's sculpt/paint into the `.inf_terrain` assets, then clear
    // the write-back marks for the ones that really landed and re-point the
    // viewport's streams at the rewritten files. A failure here surfaces AFTER the
    // level is on disk and leaves the terrain edits dirty, so the next save retries
    // them rather than losing them.
    let mut result = SaveResultDto {
        path: target.to_string_lossy().to_string(),
        terrain_assets_written: 0,
        terrain_tiles_written: 0,
        terrain_failures: Vec::new(),
        voxel_assets_written: 0,
        voxel_chunks_written: 0,
        voxel_warnings: Vec::new(),
    };
    if !staged.is_empty() {
        let content_root = app
            .try_state::<crate::commands::ProjectState>()
            .and_then(|p| p.current_content_root());
        match content_root {
            Some(root) => {
                let report = terrain_edit::write_terrain_edits(&staged, &root);
                if !report.is_empty() {
                    let mut doc = lock(&state.doc)?;
                    terrain_edit::mark_terrain_edits_saved(&mut doc, &report);
                    drop(doc);
                    tracing::info!("{}", report.summary());
                }
                result.terrain_assets_written = report.written.len() as u32;
                result.terrain_tiles_written = report.tiles() as u32;
                result.terrain_failures = report
                    .unwritten
                    .iter()
                    .map(|u| format!("terrain {}: {}", u.entity, u.reason))
                    .collect();
                // Broadcast (P23.2a): the `.inf_terrain` on disk changed, so
                // every viewport streaming it must reopen — one left on the
                // pre-save bytes would draw ground the file no longer has.
                if !report.written.is_empty() {
                    if let Some(viewport) = app.try_state::<crate::commands::ViewportState>() {
                        viewport.reload_terrain_stores(crate::commands::Target::All);
                    }
                }
            }
            None => {
                let msg = format!(
                    "{} terrain(s) have unsaved edits but no project is open, so there is no \
                     content root to write the .inf_terrain into — the edits stay in memory",
                    staged.len()
                );
                tracing::warn!("scene_save: {msg}");
                result.terrain_failures.push(msg);
            }
        }
    }
    // A working set nobody can read is a save failure, not an empty stage. It is
    // reported rather than returned because the level bytes are already on disk
    // and the document is already marked saved: bailing here with `?` would leave
    // a document that believes it was written by a save that never wrote it.
    let staged_voxels = match staged_voxels {
        Ok(staged) => staged,
        Err(msg) => {
            tracing::error!("scene_save: {msg}");
            result.voxel_warnings.push(msg);
            Vec::new()
        }
    };
    // Fold this session's carves into the `.inf_voxel` assets, then clear the
    // marks for the ones that really landed and re-walk the viewport's index.
    //
    // **After** the terrain flush, deliberately. Neither write reads the other,
    // so this is not a data dependency — it is a reporting one: a cave and the
    // mouth above it are ONE edit to the author, and a save that announced the
    // rock written while the ground failed would be describing half a cut.
    if !staged_voxels.is_empty() {
        let content_root = app
            .try_state::<crate::commands::ProjectState>()
            .and_then(|p| p.current_content_root());
        match content_root {
            Some(root) => {
                let report = voxel_edit::write_voxel_edits(&staged_voxels, &root);
                if !report.is_empty() {
                    match voxel_volumes.as_ref().map(|v| v.lock()) {
                        Some(Ok(mut store)) => {
                            voxel_edit::mark_voxel_edits_saved(&mut store, &report);
                        }
                        // Poisoned in the window between staging and marking (the
                        // rewrite runs with both locks released, so the viewport
                        // thread can panic inside it). The marks stay set, which
                        // is the safe direction — the next save rewrites the same
                        // chunks — but it must be said, not swallowed.
                        Some(Err(_)) => {
                            let msg = format!(
                                "{} volume(s) were written but their write-back marks could \
                                 not be cleared: the voxel working set was poisoned during \
                                 the save. The next save will rewrite them.",
                                report.written.len()
                            );
                            tracing::error!("scene_save: {msg}");
                            result.voxel_warnings.push(msg);
                        }
                        None => {}
                    }
                    tracing::info!("{}", report.summary());
                }
                result.voxel_assets_written = report.written.len() as u32;
                result.voxel_chunks_written = report.chunks() as u32;
                result.voxel_warnings.extend(
                    report
                        .unwritten
                        .iter()
                        .map(|u| format!("volume {}: {}", u.entity, u.reason)),
                );
                if !report.written.is_empty() {
                    if let Some(viewport) = app.try_state::<crate::commands::ViewportState>() {
                        // The terrain twin: a re-walk, not a clear. Loaded volumes
                        // keep their chunks and meshes, so Ctrl+S never blinks a
                        // cave — what changes is which GUIDs resolve. Broadcast,
                        // for the terrain twin's reason (P23.2a).
                        viewport.reload_voxel_stores(crate::commands::Target::All);
                    }
                }
            }
            None => {
                let msg = format!(
                    "{} voxel volume(s) have unsaved carve edits but no project is open, so \
                     there is no content root to write the .inf_voxel into — the edits stay \
                     in memory",
                    staged_voxels.len()
                );
                tracing::warn!("scene_save: {msg}");
                result.voxel_warnings.push(msg);
            }
        }
    }
    // **The inline-terrain hole advisory, on EVERY save** (P21.2) — not only when
    // something was carved this session. The state it detects is one a document
    // *arrives* in (an older session, a hand-edited level, a terrain converted
    // back to inline after a carve), and the save is the moment the mouths are
    // actually lost: an inline terrain's tiles ride the `.inf_lvl`, whose frozen
    // v19 wire form drops the hole mask by construction. Gating it on a carve
    // having happened would mean the one save that silently seals a level is the
    // one that says nothing.
    {
        let doc = lock(&state.doc)?;
        for line in voxel_edit::inline_hole_warnings(&doc) {
            tracing::warn!("scene_save: {line}");
            result.voxel_warnings.push(line);
        }
    }
    // A clean save invalidates the crash-recovery file — but ONLY when the save
    // really was clean (P16.4b audit). Terrain edits that could not be written are
    // still in memory and nowhere else, so dropping the recovery file would leave
    // them with no backstop at all; keeping it (and the note beside it) is the
    // whole point of the note.
    if let Ok(dir) = data_dir(&app) {
        let terrain_still_dirty = lock(&state.doc)?.has_unsaved_terrain_edits();
        // The voxel twin, and the same starvation condition: carve edits that
        // could not be written live only in the shared store, so the recovery
        // file and its note are their only backstop (P21.2).
        let voxels_still_dirty = voxels_still_dirty(voxel_volumes.as_ref());
        if terrain_still_dirty || voxels_still_dirty {
            tracing::warn!(
                "scene_save: unsaved terrain/carve edits survived the save — keeping the \
                 crash-recovery file and its note"
            );
        } else if let Err(msg) = serialize::clear_recovery(&dir) {
            // A recovery file that outlives the save superseding it is restored
            // on the next boot over the newer content (C4-45). `clear_recovery`
            // has already stamped it superseded, so that cannot happen silently
            // — but the failure is reported rather than swallowed. It goes to the
            // Output Log rather than into `SaveResultDto`: the save itself
            // succeeded, and the DTO's two warning lists name the two asset
            // classes a save writes (terrain, voxels), neither of which this is.
            tracing::error!("scene_save: {msg}");
        }
    }
    // Save As adopts the written path as the current level.
    if let Some(p) = new_current {
        *state.current_level_path.lock().map_err(|e| e.to_string())? = Some(p);
    }
    emit_world_delta(&app, &state);
    Ok(result)
}

/// The path the current level last opened from / saved to (`null` when
/// untitled). The Save As dialog seeds its default path from this.
#[tauri::command]
pub async fn scene_current_path(state: State<'_, SceneState>) -> Result<Option<String>, String> {
    Ok(state
        .current_level_path
        .lock()
        .map_err(|e| e.to_string())?
        .clone()
        .map(|p| p.to_string_lossy().to_string()))
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
    // The document was replaced wholesale, so every terrain stream is keyed on
    // the OLD document's entity GUIDs: dead memory holding a whole `.inf_terrain`
    // payload, plus any tile it pinned for an unsaved edit (which nothing would
    // ever unpin). Release them (P16.4b) — in EVERY viewport (P23.2a): the
    // streams are keyed on that document whichever window opened them, so a
    // Primary-only clear would leak a second viewport's whole payload.
    if let Some(viewport) = app.try_state::<crate::commands::ViewportState>() {
        viewport.clear_streams(crate::commands::Target::All);
    }
    // The opened file becomes the current level (a later plain Save overwrites it).
    *state.current_level_path.lock().map_err(|e| e.to_string())? = Some(path);
    // A fresh document restarts the version counter (lower than the doc we just
    // replaced), so drop the stale baseline — otherwise the version-monotonic
    // guard in `emit_world_delta` would suppress this scene's full delta.
    *state.last.lock().map_err(|e| e.to_string())? = None;
    emit_world_delta(&app, &state);
    Ok(snap)
}

/// Write the crash-recovery file (frontend calls this on a debounce while
/// there are unsaved changes; P3.5.4).
#[tauri::command]
pub async fn scene_autosave(app: AppHandle, state: State<'_, SceneState>) -> Result<(), String> {
    let dir = data_dir(&app)?;
    // The carve half of the same "clean document, unsaved assets" state (P21.2).
    //
    // **THE LOCK RULE: never hold the document and the volumes at the same
    // time** (P23.2a audit; stated in full at `scene_save`). This site takes the
    // store FIRST — the guard dies with the `and_then` closure, entirely before
    // the doc lock below — which is fine precisely because the rule is about
    // overlap and not about order. Do not "align" it with the other sites by
    // moving it inside the doc block: that would create the overlap.
    // The store is the process's (P23.2a), so this answers the same whether a
    // viewport is attached or not: with none, it is empty and there is no note.
    let voxel_note = app
        .try_state::<crate::commands::SharedStores>()
        .map(|s| s.voxel_volumes())
        .and_then(|v| match v.lock() {
            Ok(store) => voxel_edit::unsaved_voxel_note(&store),
            // Not `None` (P21.2 audit). `None` here means "no unsaved carves",
            // which on a clean document short-circuits autosave entirely — so a
            // poisoned store would suppress the one mechanism that records the
            // carves existed. An unreadable store must assume the worst.
            Err(_) => Some(POISONED_VOXEL_STORE_SAVE_FAILURE.to_string()),
        });
    // Encode under the lock (authored values if a scrub is live), write outside
    // it — and both **off the async workers** (Hardening Wave E). This fires on
    // a frontend `setInterval` every five seconds for the life of the session,
    // and on a dirty document it bincodes the WHOLE level and then writes it to
    // disk. Doing that in the body of an `async fn` parks a Tokio worker that
    // all 239 commands share, five seconds apart, for ever.
    //
    // The doc lock still covers the encode and nothing else — it has to: the
    // payload is a function of the document, and taking a snapshot to encode
    // outside the lock would be a second whole copy of the level to save one
    // encode's worth of hold. The `voxel_note` above stays where it is: THE
    // LOCK RULE (never hold the document and the volumes at once) is about
    // overlap, and moving the store read into this closure would create it.
    let doc_handle = Arc::clone(&state.doc);
    tauri::async_runtime::spawn_blocking(move || {
        let (payload, terrain_note) = {
            let mut doc = lock(&doc_handle)?;
            // A document can be CLEAN and still carry unsaved terrain edits — a save
            // whose `.inf_terrain` write failed marks the level saved but leaves the
            // tiles dirty. Gating autosave on `is_dirty()` alone would then starve the
            // one mechanism that records those edits ever existed (P16.4b audit).
            // `voxel_note` is exactly that condition for a `.inf_voxel`.
            //
            // The *level* half of the same hazard is closed at the source since
            // C4-3/F2: `scene_save` no longer marks the document saved before its
            // write, so a save whose `.inf_lvl` write failed leaves the document
            // dirty and this gate re-arms the recovery file by itself. It used to
            // do the opposite — mark clean, fail, and stop autosaving work that
            // existed nowhere on disk.
            if !doc.is_dirty() && !doc.has_unsaved_terrain_edits() && voxel_note.is_none() {
                return Ok(());
            }
            let payload =
                doc.with_authored_scene(|d| serialize::encode(&serialize::to_scene_file(d)))?;
            (payload, terrain_edit::unsaved_terrain_note(&doc))
        };
        serialize::write_recovery_bytes(&payload, &dir)?;
        // Autosave deliberately does NOT rewrite `.inf_terrain` or `.inf_voxel`
        // assets — asset writes are explicit (P16.4b, P21.2). Record that unsaved
        // terrain and carve edits existed instead, so a recovery can say honestly
        // that the ground and the caves it restores are the last SAVED assets
        // rather than pretending the level is whole.
        //
        // ONE note file carrying both halves, not two: a recovery consumes and
        // warns exactly once, and a second file is a second thing to forget.
        let note = match (terrain_note, voxel_note) {
            (Some(t), Some(v)) => Some(format!("{t}\n{v}")),
            (t, v) => t.or(v),
        };
        serialize::write_recovery_terrain_note(&dir, note.as_deref())?;
        Ok(())
    })
    .await
    .map_err(|e| format!("scene_autosave task failed to run: {e}"))?
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
    // The document was replaced wholesale, so every terrain stream is keyed on
    // the OLD document's entity GUIDs: dead memory holding a whole `.inf_terrain`
    // payload, plus any tile it pinned for an unsaved edit (which nothing would
    // ever unpin). Release them (P16.4b) — in EVERY viewport (P23.2a): the
    // streams are keyed on that document whichever window opened them, so a
    // Primary-only clear would leak a second viewport's whole payload.
    if let Some(viewport) = app.try_state::<crate::commands::ViewportState>() {
        viewport.clear_streams(crate::commands::Target::All);
    }
    // A brand-new scene is untitled — forget any current-level path.
    *state.current_level_path.lock().map_err(|e| e.to_string())? = None;
    // Reset the delta baseline: the fresh document's version counter restarts
    // below the replaced one, which the version-monotonic guard would otherwise
    // treat as a stale (backward) emit and drop.
    *state.last.lock().map_err(|e| e.to_string())? = None;
    emit_world_delta(&app, &state);
    Ok(snap)
}

#[cfg(test)]
mod tests {
    use super::resolve_save_target;
    use super::{
        mark_saved_if_unchanged, stage_voxels, voxels_still_dirty, SceneState,
        POISONED_VOXEL_STORE_SAVE_FAILURE,
    };
    use inf_editor_core::ipc::SpawnKind;
    use inf_editor_core::voxel_store::{shared_volumes, SharedVoxelVolumes};
    use std::path::{Path, PathBuf};

    #[test]
    fn save_target_precedence_explicit_current_quicksave() {
        let quicksave = Path::new("/data/quicksave.inf_lvl");
        let current = Path::new("/proj/Levels/main.inf_lvl");
        let explicit = Path::new("/proj/Levels/copy.inf_lvl");

        // Save As: an explicit path wins AND becomes the new current level.
        let (target, new_current) = resolve_save_target(Some(explicit), Some(current), quicksave);
        assert_eq!(target, explicit.to_path_buf());
        assert_eq!(new_current, Some(explicit.to_path_buf()));

        // Plain Save with a current level set: overwrite it, don't change current.
        let (target, new_current) = resolve_save_target(None, Some(current), quicksave);
        assert_eq!(target, current.to_path_buf());
        assert_eq!(new_current, None);

        // Plain Save with no current level (fresh/untitled): fall back to quicksave.
        let (target, new_current): (PathBuf, _) = resolve_save_target(None, None, quicksave);
        assert_eq!(target, quicksave.to_path_buf());
        assert_eq!(new_current, None);
    }

    /// A [`SharedVoxelVolumes`] whose mutex is **poisoned** — a thread panicked
    /// holding the guard, which is exactly what a viewport-thread panic mid-carve
    /// leaves behind (the render loop's `catch_unwind` nets are around the render
    /// and the pointer-interaction block; a panic anywhere else on that thread
    /// poisons every lock it held).
    fn poisoned_store() -> SharedVoxelVolumes {
        let volumes = shared_volumes();
        let inner = volumes.clone();
        // The panic message is expected output — the test harness prints it, and
        // that is cheaper than installing a process-global panic hook that other
        // tests in this binary would race against.
        let _ = std::thread::spawn(move || {
            let _guard = inner.lock().expect("a fresh mutex is not poisoned");
            panic!("deliberate: poisoning the voxel working set");
        })
        .join();
        assert!(volumes.lock().is_err(), "the fixture must really poison it");
        volumes
    }

    /// **THE B2 GATE.** A poisoned voxel working set must make the save *report a
    /// failure*, and must never be read as "nothing was carved".
    ///
    /// Both halves matter and they used to fail together: staging folded
    /// `Some(Err(poison))` into the no-viewport arm (so every carve was silently
    /// dropped) and the recovery check answered `unwrap_or(false)` (so the save
    /// reported success and deleted the crash-recovery file — the only remaining
    /// copy of the chunks it had just failed to read).
    #[test]
    fn a_poisoned_voxel_store_fails_the_save_and_keeps_the_recovery_file() {
        let volumes = poisoned_store();

        let err = stage_voxels(Some(&volumes)).expect_err("a poisoned store staged silently");
        assert_eq!(err, POISONED_VOXEL_STORE_SAVE_FAILURE);
        assert!(
            err.contains("crash-recovery"),
            "the failure line must say the backstop was kept: {err}"
        );

        assert!(
            voxels_still_dirty(Some(&volumes)),
            "a poisoned store answered `not dirty`, which is what deletes the recovery file"
        );
    }

    /// The two honest answers, so the gate above cannot pass by reporting failure
    /// unconditionally: no viewport stages nothing and is not dirty, and a
    /// readable empty store is the same.
    #[test]
    fn a_readable_or_absent_store_stages_cleanly() {
        assert!(stage_voxels(None)
            .expect("no viewport is not a save failure")
            .is_empty());
        assert!(!voxels_still_dirty(None));

        let volumes = shared_volumes();
        assert!(stage_voxels(Some(&volumes))
            .expect("an empty store stages nothing")
            .is_empty());
        assert!(!voxels_still_dirty(Some(&volumes)));
    }

    /// **The save path resolves its store WITHOUT a viewport** (P23.2a — the
    /// hoist).
    ///
    /// Before the hoist the carve store hung off a `ViewportHandle`, so this
    /// exact call — "give me what the author carved" — could only be answered by
    /// a window that exists. `SharedStores` is constructed here with no viewport,
    /// no window and no GPU anywhere in the process, and staging runs off it
    /// straight through, which is the whole claim.
    ///
    /// It also pins the None-semantics preservation: the pre-hoist `None` ("no
    /// viewport ⇒ no carve edits ⇒ nothing to write") and the post-hoist EMPTY
    /// store must produce the *same* staged result and the *same* dirty answer,
    /// asserted against each other rather than each against a literal.
    #[test]
    fn staging_resolves_the_carve_store_with_no_viewport_attached() {
        let stores = crate::commands::SharedStores::default();
        let volumes = stores.voxel_volumes();

        let staged = stage_voxels(Some(&volumes)).expect("no viewport is not a save failure");
        assert_eq!(
            staged.len(),
            stage_voxels(None).expect("the pre-hoist answer").len(),
            "an unattached process store must stage exactly what the old `None` did"
        );
        assert_eq!(
            voxels_still_dirty(Some(&volumes)),
            voxels_still_dirty(None),
            "…and must answer the recovery-file question the same way"
        );
    }

    /// **`mark_saved` runs after `write_encoded`, never before** (C4-3/F2).
    ///
    /// A source gate, because a `#[tauri::command]` cannot be driven from a test
    /// (the reason `save_mesh_session`'s logic lives in Ring 1) and *ordering*
    /// is the whole rule. It reads `scene_save`'s own body and asserts three
    /// things: the write is there, the mark is there, and the write comes
    /// first — plus that no bare `mark_saved()` call has crept back into the
    /// body, which is exactly the shape that was there before.
    ///
    /// `.rs` is `text eol=lf` in `.gitattributes`, so the indices this compares
    /// are the same on every checkout (the P22.4 lesson).
    #[test]
    fn scene_save_marks_the_document_saved_only_after_the_write() {
        const SRC: &str = include_str!("scene.rs");
        let start = SRC
            .find("pub async fn scene_save(")
            .expect("scene_save moved or was renamed");
        let body = &SRC[start..];
        let end = body
            .find("\n}\n")
            .expect("scene_save's body has no closing brace at column 0");
        let body = &body[..end];

        let write = body
            .find("serialize::write_encoded(")
            .expect("scene_save no longer writes the level through write_encoded");
        let mark = body
            .find("mark_saved_if_unchanged(")
            .expect("scene_save no longer marks the document saved");
        assert!(
            write < mark,
            "mark_saved runs before the write — a failed save would report success, \
             disarm the close prompt and starve the crash-recovery autosave"
        );
        assert!(
            !body.contains("doc.mark_saved()"),
            "a bare mark_saved() is back in scene_save; the marking must go through \
             mark_saved_if_unchanged, after the write"
        );
    }

    /// The version guard inside that marking: a document edited while it was
    /// being written stays dirty, because the bytes on disk do not contain the
    /// edit. Both directions, so "never mark anything" cannot pass.
    #[test]
    fn a_document_edited_during_its_own_save_stays_dirty() {
        let state = SceneState::new();

        // Unchanged: the mark lands.
        {
            let mut doc = state.doc.lock().unwrap();
            doc.edit_create(SpawnKind::Cube, "A", None);
            assert!(doc.is_dirty());
        }
        let version = state.doc.lock().unwrap().version();
        mark_saved_if_unchanged(&state, version).unwrap();
        assert!(
            !state.doc.lock().unwrap().is_dirty(),
            "a save of the current document must mark it clean"
        );

        // Changed under the write: the mark is withheld.
        {
            let mut doc = state.doc.lock().unwrap();
            doc.edit_create(SpawnKind::Cone, "B", None);
        }
        mark_saved_if_unchanged(&state, version).unwrap();
        assert!(
            state.doc.lock().unwrap().is_dirty(),
            "an edit that landed after the encode was marked saved by a save that \
             never saw it"
        );
    }
}
