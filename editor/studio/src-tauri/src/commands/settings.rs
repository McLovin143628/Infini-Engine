//! Settings commands — per-project (P8.2c) and app-level (Wave E batch A).
//!
//! *Per project*: a tiny file (`<content-root>/.infinity/settings.toml`,
//! `inf-editor-core::project_settings`) holding the pixels-per-unit used by 2D
//! pixel snapping. Mirrors the sorting-layer commands' shape: the project root
//! comes from `AssetState::content_root()`, normalized + written deterministically.
//!
//! *App level*: `editor-settings.toml` beside the layout presets in the app
//! config directory (`inf-editor-core::editor_settings`). **Ring 2 resolves the
//! directory** — Ring 1 never names `app_config_dir` (the `LayoutStore`
//! precedent, `commands/layout.rs`).

use inf_editor_core::editor_settings::EditorSettings;
use inf_editor_core::ipc::ProjectSettingsDto;
use inf_editor_core::project_settings::ProjectSettings;
use tauri::{Manager, State};

use super::assets::AssetState;

/// Where the app-level preferences live: `<app config>/editor-settings.toml`.
fn config_dir(app: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
    app.path()
        .app_config_dir()
        .map_err(|e| format!("resolve app config dir: {e}"))
}

/// The current project's editor settings (defaults when none saved yet).
#[tauri::command]
pub async fn project_settings_get(
    assets: State<'_, AssetState>,
) -> Result<ProjectSettingsDto, String> {
    let root = assets.content_root().ok_or("assets not initialized")?;
    // Unreadable is an error, not the default (C4-38).
    let s = ProjectSettings::load_or_default(&root)?;
    Ok(ProjectSettingsDto {
        pixels_per_unit: s.pixels_per_unit,
    })
}

/// Persist the editor settings; returns the normalized values.
#[tauri::command]
pub async fn project_settings_set(
    assets: State<'_, AssetState>,
    settings: ProjectSettingsDto,
) -> Result<ProjectSettingsDto, String> {
    let root = assets.content_root().ok_or("assets not initialized")?;
    let mut s = ProjectSettings {
        pixels_per_unit: settings.pixels_per_unit,
    };
    s.normalize();
    s.save(&root)?;
    Ok(ProjectSettingsDto {
        pixels_per_unit: s.pixels_per_unit,
    })
}

/// The app-level editor preferences (defaults when none saved yet).
///
/// A corrupt file is an ERROR, not the defaults (C4-38) — the frontend surfaces
/// the message and keeps its in-memory values, so a typo in a hand-edited file
/// never silently wipes a user's theme and keybindings.
#[tauri::command]
pub async fn editor_settings_get(app: tauri::AppHandle) -> Result<EditorSettings, String> {
    EditorSettings::load_or_default(&config_dir(&app)?)
}

/// Persist the app-level editor preferences; returns the NORMALIZED values.
///
/// The frontend applies what comes back, not what it sent — that is how a
/// clamped or NaN-guarded field reaches the UI instead of diverging from disk.
#[tauri::command]
pub async fn editor_settings_set(
    app: tauri::AppHandle,
    settings: EditorSettings,
) -> Result<EditorSettings, String> {
    let dir = config_dir(&app)?;
    let mut s = settings;
    // `migrate` runs on the way IN too: a frontend that echoed back a version it
    // read from a newer file must not be able to write it into an older one.
    s.migrate()?;
    s.normalize();
    s.save(&dir)?;
    Ok(s)
}
