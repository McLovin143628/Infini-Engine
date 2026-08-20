//! **The GIS import wizard's backend** (IB-3, Ring 2).
//!
//! Three commands, and none of them decides anything: the probe is
//! `inf_gis::probe`, the suggestion is `GisImportSettingsDto::suggested`, and
//! the import is `inf_editor_core::gis::run_import` — the same function the
//! `inf gis` CLI's library half calls, so the wizard and the headless pipeline
//! are two front ends onto one importer rather than two importers that agree by
//! inspection.
//!
//! Every one runs on `spawn_blocking`: reading a 50 000-feature Shapefile is
//! hundreds of milliseconds of parsing and reprojection, and a Tauri command
//! that does it on the async runtime's thread freezes the shell.

use std::path::PathBuf;

use tauri::{AppHandle, State};

use inf_editor_core::ipc::{GisImportResultDto, GisImportSettingsDto, GisProbeDto};

use super::assets::AssetState;
use super::scene::SceneState;

/// The `.prj`-and-fields preview: what is in a file, before importing it.
///
/// `source_crs` is the author's override; empty reads the `.prj` sidecar, and a
/// file with neither is a refusal naming the remedy rather than a guess. The
/// level's own anchor CRS rides back on the DTO so the wizard can show both —
/// "the file says zone 10 and your world is anchored in zone 11" is the question
/// an author needs asked.
#[tauri::command]
pub async fn gis_probe(
    path: String,
    source_crs: String,
    scene: State<'_, SceneState>,
) -> Result<GisProbeDto, String> {
    let level_anchor_crs = {
        let doc = scene.doc.lock().map_err(|e| e.to_string())?;
        let a = doc.geo();
        a.enabled.then(|| a.crs.clone())
    };
    tauri::async_runtime::spawn_blocking(move || {
        inf_editor_core::gis::probe_dto(&PathBuf::from(&path), &source_crs, level_anchor_crs)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// The settings the wizard should open on for a probed source: the layer kind
/// its geometry suggests, and the author's own remembered entity cap (IB-14).
#[tauri::command]
pub async fn gis_suggested_settings(
    probe: GisProbeDto,
    max_entities: usize,
) -> Result<GisImportSettingsDto, String> {
    Ok(GisImportSettingsDto::suggested(&probe, max_entities.max(1)))
}

/// **Import.** Spawns the layer, and whichever of the road surface, the
/// land-cover paint and the footprint bake the author asked for.
#[tauri::command]
pub async fn gis_import(
    app: AppHandle,
    path: String,
    settings: GisImportSettingsDto,
    assets: State<'_, AssetState>,
    scene: State<'_, SceneState>,
) -> Result<GisImportResultDto, String> {
    let project = assets.project_handle()?;
    let doc = scene.doc.clone();
    let p = PathBuf::from(&path);
    // **Both locks are taken on the blocking thread, in this order** — project
    // then document — because that is the order every other write path takes
    // them (`viewport_drop`, the terrain spawn) and two orders is a deadlock
    // waiting for a slow import to meet a slow save.
    let result = tauri::async_runtime::spawn_blocking(move || -> Result<_, String> {
        let mut proj = project.lock().map_err(|e| e.to_string())?;
        let mut doc = doc.lock().map_err(|e| e.to_string())?;
        inf_editor_core::gis::run_import(&mut proj, &mut doc, &p, &settings)
    })
    .await
    .map_err(|e| e.to_string())??;

    // The import both created entities and (possibly) wrote assets, so both
    // projections have to be told.
    super::scene::emit_world_delta(&app, &scene);
    super::assets::emit_changed(&app, &assets);
    Ok(result)
}
