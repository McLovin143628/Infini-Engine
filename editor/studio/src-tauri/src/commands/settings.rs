//! Per-project editor-settings commands (P8.2c).
//!
//! A tiny per-project settings file (`<content-root>/.infinity/settings.toml`,
//! `inf-editor-core::project_settings`) holding the pixels-per-unit used by 2D
//! pixel snapping. Mirrors the sorting-layer commands' shape: the project root
//! comes from `AssetState::content_root()`, normalized + written deterministically.

use inf_editor_core::ipc::ProjectSettingsDto;
use inf_editor_core::project_settings::ProjectSettings;
use tauri::State;

use super::assets::AssetState;

/// The current project's editor settings (defaults when none saved yet).
#[tauri::command]
pub async fn project_settings_get(
    assets: State<'_, AssetState>,
) -> Result<ProjectSettingsDto, String> {
    let root = assets.content_root().ok_or("assets not initialized")?;
    let s = ProjectSettings::load(&root);
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
