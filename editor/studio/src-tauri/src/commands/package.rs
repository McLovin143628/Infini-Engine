//! Package / cook command (P9.2 item 3, P9.5 editor surface).
//!
//! [`project_package`] runs [`inf_packager::cook`] against the currently-open
//! project (its root resolved from [`ProjectState`]) and returns a
//! [`PackageResultDto`] projection of the [`inf_packager::CookReport`], or a
//! structured [`PackageErrorDto`] on failure (blueprint failures carry the
//! class + handler anchor).
//!
//! Cooking does real filesystem + CPU work (scan, dependency closure, blueprint
//! validation, level rewrite, zstd pack write), so it runs on a **blocking**
//! task via [`tauri::async_runtime::spawn_blocking`] rather than tying up an
//! async-runtime worker. Start/finish is broadcast on the `package://state`
//! event (a boolean running flag) for any global listener.
//!
//! **Follow-ups (honest scope):** per-stage progress is not surfaced — the
//! `cook` API exposes no progress callback, so we emit start/finish only and do
//! not fake intermediate progress. Full per-platform *bundling* (`inf export`,
//! P9.5) is not yet available in `inf-packager` (no `bundle` module at time of
//! writing); this wires the cook only, so the dialog produces a `.inf_pack` +
//! manifest. Wiring the dialog to full export lands when that module exists.

use std::path::PathBuf;

use inf_asset::AssetId;
use inf_editor_core::ipc::{PackageErrorDto, PackageKindCountDto, PackageResultDto};
use inf_packager::{cook, CookError, CookOptions, CookReport};
use tauri::{AppHandle, Emitter, State};

use super::project::ProjectState;

/// Forward-slash a path for display / TS interop (mirrors the project commands).
fn slashed(path: &std::path::Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Project a successful [`CookReport`] into its wire DTO.
fn report_to_dto(report: &CookReport) -> PackageResultDto {
    PackageResultDto {
        project_name: report.project_name.clone(),
        engine_version: report.engine_version.clone(),
        out_dir: slashed(&report.out_dir),
        pack_path: slashed(&report.pack_path),
        manifest_path: slashed(&report.manifest_path),
        asset_count: report.asset_count as u32,
        kinds: report
            .kinds
            .iter()
            .map(|(kind, count)| PackageKindCountDto {
                kind: kind.clone(),
                count: *count as u32,
            })
            .collect(),
        pack_bytes: report.pack_bytes,
        levels: report.levels.iter().map(|l| l.to_string()).collect(),
        root_level: report.root_level.map(|l| l.to_string()),
        blueprints_validated: report.blueprints_validated as u32,
        levels_rewritten: report.levels_rewritten as u32,
        warnings: report.warnings.clone(),
    }
}

/// Map a [`CookError`] onto the structured wire error, anchoring blueprint
/// failures to their class + handler.
fn cook_error_to_dto(err: CookError) -> PackageErrorDto {
    let mut dto = PackageErrorDto {
        class: "internal".into(),
        message: err.to_string(),
        blueprint_class: None,
        handler: None,
        guid: None,
    };
    match err {
        CookError::Blueprint {
            guid,
            class,
            handler,
            message,
        } => {
            dto.class = "blueprint".into();
            dto.message = message;
            dto.blueprint_class = Some(class);
            dto.handler = Some(handler);
            dto.guid = Some(guid.to_string());
        }
        CookError::Scene { guid, source } => {
            dto.class = "scene".into();
            dto.message = source.to_string();
            dto.guid = Some(guid.to_string());
        }
        CookError::UnknownRoot(guid) => {
            dto.class = "unknown_root".into();
            dto.guid = Some(guid.to_string());
        }
        CookError::Export(message) => {
            dto.class = "export".into();
            dto.message = message;
        }
        CookError::Mod(message) => {
            dto.class = "mod".into();
            dto.message = message;
        }
        CookError::Mesh { guid, message } => {
            dto.class = "mesh".into();
            dto.message = message;
            dto.guid = Some(guid.to_string());
        }
        CookError::Terrain { guid, message } => {
            dto.class = "terrain".into();
            dto.message = message;
            dto.guid = Some(guid.to_string());
        }
        CookError::Partition { guid, message } => {
            dto.class = "partition".into();
            dto.message = message;
            dto.guid = Some(guid.to_string());
        }
        CookError::Project(_) => dto.class = "project".into(),
        CookError::Asset(_) => dto.class = "asset".into(),
        CookError::Io(_) => dto.class = "io".into(),
        CookError::Toml(_) => dto.class = "manifest".into(),
    }
    dto
}

/// Cook the open project into `out_dir` (default `<project>/Build`), packing
/// `roots` (GUID strings) or the default root set when omitted.
#[tauri::command]
pub async fn project_package(
    app: AppHandle,
    project: State<'_, ProjectState>,
    out_dir: Option<String>,
    roots: Option<Vec<String>>,
) -> Result<PackageResultDto, PackageErrorDto> {
    // Resolve the open project's root.
    let root = project.current_root().ok_or_else(|| PackageErrorDto {
        class: "no_project".into(),
        message: "No project is open. Open a project before packaging.".into(),
        blueprint_class: None,
        handler: None,
        guid: None,
    })?;

    // Output directory: explicit (non-empty) or the default `<project>/Build`.
    let out = match out_dir {
        Some(s) if !s.trim().is_empty() => PathBuf::from(s.trim()),
        _ => root.join("Build"),
    };

    // Parse explicit root GUIDs, if any.
    let parsed_roots = match roots {
        Some(list) => {
            let mut ids = Vec::with_capacity(list.len());
            for s in list {
                let id = s.parse::<AssetId>().map_err(|e| PackageErrorDto {
                    class: "bad_root".into(),
                    message: format!("invalid root GUID `{s}`: {e}"),
                    blueprint_class: None,
                    handler: None,
                    guid: Some(s.clone()),
                })?;
                ids.push(id);
            }
            Some(ids)
        }
        None => None,
    };

    let opts = CookOptions {
        roots: parsed_roots,
        pack_name: None,
        ..Default::default()
    };

    // Cook on a blocking task; announce running state to any global listener.
    let _ = app.emit("package://state", true);
    let cook_root = root.clone();
    let cook_out = out.clone();
    let joined =
        tauri::async_runtime::spawn_blocking(move || cook(&cook_root, &cook_out, &opts)).await;
    let _ = app.emit("package://state", false);

    match joined {
        Ok(Ok(report)) => Ok(report_to_dto(&report)),
        Ok(Err(err)) => Err(cook_error_to_dto(err)),
        Err(join_err) => Err(PackageErrorDto {
            class: "internal".into(),
            message: format!("cook task failed to run: {join_err}"),
            blueprint_class: None,
            handler: None,
            guid: None,
        }),
    }
}
