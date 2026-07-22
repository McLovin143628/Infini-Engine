//! Named content-collection commands (E-P8).
//!
//! Collections are user-defined, **persisted** groupings of assets — the durable
//! successor to the frontend-only Favorites. They live at
//! `<project_root>/.infinity/collections.toml` (the same directory as
//! `inf_audio`'s `mixer.toml`, resolved as the parent of the content root), and
//! are NOT dependency edges (deleting an asset never asks about collections; a
//! now-dangling member is pruned on the next load instead).
//!
//! Every mutation saves the file and emits `collections://changed`, which the
//! Content Drawer re-fetches on (mirroring `assets://changed`).

use std::path::PathBuf;

use inf_editor_core::assets::collections::CollectionsFile;
use inf_editor_core::ipc::CollectionDto;
use tauri::{AppHandle, Emitter, State};

use super::assets::AssetState;

/// The project root that holds `.infinity/collections.toml` — the parent of the
/// asset content root (mirrors where `mixer.toml` lives). Falls back to the
/// content root itself when it has no parent.
fn project_root(assets: &AssetState) -> Result<PathBuf, String> {
    let content = assets.content_root().ok_or("assets not initialized")?;
    Ok(content.parent().map(|p| p.to_path_buf()).unwrap_or(content))
}

/// Load the collections file and prune ids whose assets no longer exist,
/// persisting the pruned file when anything was dropped. Returns the project
/// root (for a subsequent save) alongside the pruned file.
fn load_pruned(assets: &AssetState) -> Result<(PathBuf, CollectionsFile), String> {
    let root = project_root(assets)?;
    let mut file = CollectionsFile::load_or_default(&root)?;
    let removed = assets.with_project(|p| Ok(file.prune_against_db(p.db())))?;
    if removed > 0 {
        file.save(&root)?;
    }
    Ok((root, file))
}

fn to_dtos(file: &CollectionsFile) -> Vec<CollectionDto> {
    let mut out: Vec<CollectionDto> = file
        .collections
        .iter()
        .map(|c| CollectionDto {
            name: c.name.clone(),
            ids: c.ids.iter().map(|id| id.to_string()).collect(),
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn emit_changed(app: &AppHandle) {
    let _ = app.emit("collections://changed", ());
}

fn parse_id(s: &str) -> Result<inf_asset::AssetId, String> {
    s.parse::<inf_asset::AssetId>()
        .map_err(|e| format!("bad asset id: {e}"))
}

/// List every collection (dangling ids pruned + persisted).
#[tauri::command]
pub async fn collections_list(assets: State<'_, AssetState>) -> Result<Vec<CollectionDto>, String> {
    let (_root, file) = load_pruned(&assets)?;
    Ok(to_dtos(&file))
}

/// Create a new (empty) collection. Errors on an empty or duplicate name.
#[tauri::command]
pub async fn collections_create(
    app: AppHandle,
    name: String,
    assets: State<'_, AssetState>,
) -> Result<Vec<CollectionDto>, String> {
    let (root, mut file) = load_pruned(&assets)?;
    file.create(&name)?;
    file.save(&root)?;
    emit_changed(&app);
    Ok(to_dtos(&file))
}

/// Rename a collection. Errors if `old` is missing or `new` is empty/duplicate.
#[tauri::command]
pub async fn collections_rename(
    app: AppHandle,
    old: String,
    new: String,
    assets: State<'_, AssetState>,
) -> Result<Vec<CollectionDto>, String> {
    let (root, mut file) = load_pruned(&assets)?;
    file.rename(&old, &new)?;
    file.save(&root)?;
    emit_changed(&app);
    Ok(to_dtos(&file))
}

/// Delete a collection. Errors if it does not exist.
#[tauri::command]
pub async fn collections_delete(
    app: AppHandle,
    name: String,
    assets: State<'_, AssetState>,
) -> Result<Vec<CollectionDto>, String> {
    let (root, mut file) = load_pruned(&assets)?;
    file.delete(&name)?;
    file.save(&root)?;
    emit_changed(&app);
    Ok(to_dtos(&file))
}

/// Add an asset to a collection (deduped, insertion-ordered).
#[tauri::command]
pub async fn collections_add(
    app: AppHandle,
    name: String,
    id: String,
    assets: State<'_, AssetState>,
) -> Result<Vec<CollectionDto>, String> {
    let asset_id = parse_id(&id)?;
    let (root, mut file) = load_pruned(&assets)?;
    file.add(&name, asset_id)?;
    file.save(&root)?;
    emit_changed(&app);
    Ok(to_dtos(&file))
}

/// Remove an asset from a collection (a missing id is a no-op).
#[tauri::command]
pub async fn collections_remove(
    app: AppHandle,
    name: String,
    id: String,
    assets: State<'_, AssetState>,
) -> Result<Vec<CollectionDto>, String> {
    let asset_id = parse_id(&id)?;
    let (root, mut file) = load_pruned(&assets)?;
    file.remove(&name, asset_id)?;
    file.save(&root)?;
    emit_changed(&app);
    Ok(to_dtos(&file))
}
