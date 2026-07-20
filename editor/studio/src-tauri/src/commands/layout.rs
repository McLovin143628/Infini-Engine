//! Dock-layout persistence commands (ROADMAP P1.2.5). Thin Tauri shims over
//! `inf_editor_core::layouts::LayoutStore` — all real logic (name validation,
//! atomic writes, listing) lives in Ring 1 where it is headless-tested.

use inf_editor_core::ipc::LayoutSummary;
use inf_editor_core::layouts::LayoutStore;
use tauri::Manager;

fn store(app: &tauri::AppHandle) -> Result<LayoutStore, String> {
    let dir = app
        .path()
        .app_config_dir()
        .map_err(|e| format!("resolve app config dir: {e}"))?
        .join("layouts");
    Ok(LayoutStore::new(dir))
}

/// Persist a named layout preset (opaque JSON owned by the dock store).
#[tauri::command]
pub async fn layout_save(app: tauri::AppHandle, name: String, json: String) -> Result<(), String> {
    store(&app)?.save(&name, &json)
}

/// Load a named layout preset; `None` when it doesn't exist.
#[tauri::command]
pub async fn layout_load(app: tauri::AppHandle, name: String) -> Result<Option<String>, String> {
    store(&app)?.load(&name)
}

/// List saved layout presets, sorted by name.
#[tauri::command]
pub async fn layout_list(app: tauri::AppHandle) -> Result<Vec<LayoutSummary>, String> {
    store(&app)?.list()
}

/// Delete a preset. Returns whether it existed.
#[tauri::command]
pub async fn layout_delete(app: tauri::AppHandle, name: String) -> Result<bool, String> {
    store(&app)?.delete(&name)
}
