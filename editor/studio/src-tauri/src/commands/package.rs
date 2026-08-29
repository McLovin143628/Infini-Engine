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
        // B10 == R2.F8: the ship/no-ship verdict, which `inf cook` has honoured
        // since C4-40 and this door did not. Every entry is also in `warnings`
        // (that is `CookReport`'s own contract), so the dialog separates them
        // rather than counting them twice.
        blocking: report.blocking.clone(),
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
        // SCRIPT1b: an InfiniScript that would not compile. Its own class, with
        // the diagnostics as the message: a script's anchor is a LINE AND A
        // COLUMN, and folding it into `blueprint` would have rendered it under a
        // handler field it does not have.
        CookError::Script {
            guid,
            name,
            diagnostics,
        } => {
            dto.class = "script".into();
            dto.message = diagnostics;
            dto.blueprint_class = Some(name);
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
        // P21.1: a `.inf_voxel` that failed its structural check. Its own class,
        // like `terrain`, so the Package dialog can point at the asset rather than
        // reporting an internal error the author cannot act on.
        CookError::VoxelVolume { guid, message } => {
            dto.class = "voxel_volume".into();
            dto.message = message;
            dto.guid = Some(guid.to_string());
        }
        CookError::BiomeSet { guid, message } => {
            dto.class = "biome_set".into();
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
    //
    // Round-2 finding B13: an explicit `out_dir` must be ABSOLUTE. It arrives
    // from the webview, and a relative one resolved against the *editor
    // process's* working directory — which is wherever the app happened to be
    // launched from, so `"../.."` wrote a whole cooked build somewhere nobody
    // named. Exporting to an arbitrary absolute directory stays legal on
    // purpose: putting a build on the Desktop is what this door is for, and
    // `package.rs::project_package` is in `paths.rs`'s `READ_ANYWHERE` table
    // with that reason written down.
    let out = match out_dir {
        Some(s) if !s.trim().is_empty() => {
            let p = PathBuf::from(s.trim());
            if !p.is_absolute() {
                return Err(PackageErrorDto {
                    class: "bad_out_dir".into(),
                    message: format!(
                        "output directory {:?} is relative; it would resolve against the \
                         editor's working directory rather than anything you named. Give \
                         an absolute path, or leave it empty for <project>/Build.",
                        s.trim()
                    ),
                    blueprint_class: None,
                    handler: None,
                    guid: None,
                });
            }
            p
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A cook report with the two advisory lists set and everything else
    /// trivial.
    ///
    /// **Constructed field by field on purpose.** `CookReport` has no `Default`,
    /// so adding a field to it breaks this test — and that is the class of
    /// finding B10 is: a fact the cook computed, that the editor's projection
    /// silently did not carry. The next such field gets looked at.
    fn report(warnings: &[&str], blocking: &[&str]) -> CookReport {
        CookReport {
            project_name: "Demo".into(),
            engine_version: "0.1.0".into(),
            out_dir: PathBuf::from("C:/proj/Build"),
            pack_path: PathBuf::from("C:/proj/Build/demo.inf_pack"),
            manifest_path: PathBuf::from("C:/proj/Build/manifest.toml"),
            asset_count: 12,
            kinds: Default::default(),
            pack_bytes: 4096,
            levels: Vec::new(),
            root_level: None,
            blueprints_validated: 0,
            levels_rewritten: 0,
            meshlet_meshes_derived: 0,
            materials_derived: 0,
            fractures_derived: 0,
            fracture_chunks: 0,
            fracture_chunks_dropped: 0,
            partitions_built: 0,
            partition_cells: 0,
            warnings: warnings.iter().map(|s| s.to_string()).collect(),
            blocking: blocking.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// **Round-2 finding B10 == R2.F8.** `inf cook` exits non-zero on
    /// `has_blocking` and prints "this build must not ship"; this projection
    /// dropped the list, and `PackageResultDto` had no slot for it — so the
    /// Package dialog, the door an author actually uses, showed those same
    /// strings inside a yellow "N warnings" list under a success panel.
    #[test]
    fn the_dto_carries_the_ship_verdict() {
        let no_boot = "no boot level: the runtime has nothing to load";
        let dangling = "a material binding is dangling";
        let dto = report_to_dto(&report(&[no_boot, dangling], &[no_boot]));

        assert_eq!(
            dto.blocking,
            vec![no_boot.to_string()],
            "the cook's ship/no-ship decision must reach the editor — it is the one \
             question `inf cook`'s exit code asks"
        );
        // `CookReport`'s own contract: a blocking entry is also a warning. Both
        // lists cross the wire whole; the dialog is what splits them, so it can
        // show each sentence once.
        assert_eq!(dto.warnings.len(), 2);
        assert!(dto.warnings.contains(&no_boot.to_string()));
    }

    /// The other direction, so the field cannot be hard-wired to "something".
    #[test]
    fn a_shippable_cook_reports_nothing_blocking() {
        let dto = report_to_dto(&report(&["a material binding is dangling"], &[]));
        assert!(
            dto.blocking.is_empty(),
            "an ordinary advisory must not read as a refusal — a dialog that cries \
             wolf is how an author learns to ignore the one that matters"
        );
        assert_eq!(dto.warnings.len(), 1);
    }
}
