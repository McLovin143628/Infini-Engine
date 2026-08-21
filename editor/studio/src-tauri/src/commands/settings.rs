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
///
/// Two files, deliberately: `pixels_per_unit` is editor state under the content
/// root's `.infinity/`, and `anim_blend` is in **`inf.toml`** because the cook
/// reads it (see [`ProjectSettingsDto::anim_blend`]).
#[tauri::command]
pub async fn project_settings_get(
    assets: State<'_, AssetState>,
    project: State<'_, super::project::ProjectState>,
) -> Result<ProjectSettingsDto, String> {
    let root = assets.content_root().ok_or("assets not initialized")?;
    // Unreadable is an error, not the default (C4-38).
    let s = ProjectSettings::load_or_default(&root)?;
    Ok(ProjectSettingsDto {
        pixels_per_unit: s.pixels_per_unit,
        anim_blend: Some(project_anim_blend(&project)),
    })
}

/// Persist the editor settings; returns the normalized values.
#[tauri::command]
pub async fn project_settings_set(
    assets: State<'_, AssetState>,
    project: State<'_, super::project::ProjectState>,
    scene: State<'_, super::scene::SceneState>,
    settings: ProjectSettingsDto,
) -> Result<ProjectSettingsDto, String> {
    let root = assets.content_root().ok_or("assets not initialized")?;
    let mut s = ProjectSettings {
        pixels_per_unit: settings.pixels_per_unit,
    };
    s.normalize();
    s.save(&root)?;

    // **The blend mode's writer** — `inf.toml`, then the live world.
    //
    // Both, because they are two halves of one promise. `inf.toml` is what the
    // cook copies into `manifest.toml` and the shipped player applies; the live
    // world is what PIE snapshots into `ScenePayload::blend_mode`. Writing only
    // the file would leave the editor previewing the OLD mode until the project
    // was reopened, which is the same "preview one blend, ship another" defect
    // one step smaller.
    //
    // `None` is a partial update and touches neither — see the DTO's note.
    if let Some(asked) = settings.anim_blend.as_deref() {
        let anim_blend =
            inf_project::anim_blend_name(inf_project::anim_blend_wire(asked)).to_string();
        if let Some(project_root) = project.current_root() {
            let mut m = inf_project::ProjectManifest::load(&project_root)
                .map_err(|e| format!("read inf.toml: {e}"))?;
            if m.anim_blend != anim_blend {
                m.anim_blend = anim_blend.clone();
                m.save(&project_root)
                    .map_err(|e| format!("write inf.toml: {e}"))?;
            }
        }
        let mut doc = scene.doc.lock().map_err(|e| e.to_string())?;
        let mode = match inf_project::anim_blend_wire(&anim_blend) {
            1 => inf_anim::SmBlendMode::CrossFade,
            _ => inf_anim::SmBlendMode::Inertialize,
        };
        inf_ecs::pose::set_blend_mode(doc.world_mut(), mode);
    }
    Ok(ProjectSettingsDto {
        pixels_per_unit: s.pixels_per_unit,
        anim_blend: Some(project_anim_blend(&project)),
    })
}

/// The open project's `anim_blend`, or the engine default when nothing is open.
fn project_anim_blend(project: &super::project::ProjectState) -> String {
    project
        .current_root()
        .and_then(|root| inf_project::ProjectManifest::load(&root).ok())
        .map(|m| m.anim_blend)
        .unwrap_or_else(|| inf_project::ANIM_BLEND_INERTIALIZE.to_string())
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

/// **The GAME's binding table** as the preferences panel renders it (island wave
/// I5), resolved against the overrides the panel is holding.
///
/// A projection rather than a second table: the same rows, in the same order,
/// from the same `inf_ui::bindings` source the in-game dialog reads. The
/// overrides come *from the caller* rather than from the file, so a panel with
/// an unsaved edit renders what the player is looking at.
#[tauri::command]
pub async fn game_bindings_table(
    overrides: std::collections::BTreeMap<String, String>,
) -> Result<Vec<inf_editor_core::ipc::GameBindingRowDto>, String> {
    Ok(inf_editor_core::editor_settings::game_binding_table(
        &overrides,
    ))
}

/// **Bind a key to a row**, through the one apply rule.
///
/// With `swap = false` a taken key changes nothing and comes back with the rows
/// that own it, so the panel can ask; with `swap = true` the key is taken off
/// exactly those rows — the token, not their whole binding — and bound here.
/// The identical rule the in-game dialog's capture runs.
#[tauri::command]
pub async fn game_binding_apply(
    overrides: std::collections::BTreeMap<String, String>,
    row: String,
    token: String,
    swap: bool,
) -> Result<inf_editor_core::ipc::GameBindingApplyDto, String> {
    Ok(inf_editor_core::editor_settings::game_binding_apply(
        &overrides, &row, &token, swap,
    ))
}
