//! Terrain commands: the erosion bake (P10.3b) and the **Terrain Import wizard**
//! (P16.4a).
//!
//! # Erosion bake
//!
//! `terrain_erode` runs the GPU hydraulic + thermal erosion compute pipeline
//! (CPU reference fallback when no adapter is present — the same code path the
//! parity test uses) over an entity's terrain and commits the result as ONE
//! undoable `SculptTerrain` height delta, then bumps the scene version + emits
//! `world://delta` so the viewport re-uploads and the Outliner re-syncs.
//!
//! DETERMINISM: eroded terrain is DATA — the delta is stored in the `.inf_lvl`,
//! so reloading a level is byte-exact on any machine. Only the *bake action*
//! itself varies by GPU adapter (GPU `f32` vs the partly-`f64` CPU reference);
//! the CPU path is the deterministic reference. See `inf_editor_core::erosion_gpu`.
//!
//! # Terrain import
//!
//! Three commands back the wizard, and none of them decode a pixel on the
//! command thread:
//!
//! * `terrain_probe_heightmap` reads the file's **header** and returns its
//!   dimensions plus suggested settings — instant on a 16 k source.
//! * `terrain_import_plan` recomputes the world a settings block would produce
//!   (metres of extent, tile counts) so the wizard's readback is never stale.
//!   Pure arithmetic, no IO.
//! * `terrain_import` queues the chunked import on the **existing asset import
//!   queue**, so progress arrives on the established `assets://import` channel
//!   and the produced asset lands in the same database the Content Drawer reads.
//!   `terrain_import_cancel` stops an in-flight job.
//!
//! `terrain_spawn_streamed` then puts a `Terrain` entity referencing the new
//! asset into the scene as one undoable edit — the "walk it immediately" step.

use std::path::PathBuf;
use std::sync::Mutex;

use glam::DVec2;
use inf_editor_core::assets::terrain_import;
use inf_editor_core::erosion_gpu::ErosionHost;
use inf_editor_core::ipc::{
    ErosionParamsDto, ErosionReportDto, HeightmapProbeDto, TerrainImportPlanDto,
    TerrainImportResultDto, TerrainImportSettingsDto,
};
use inf_terrain::HeightmapGrid;
use tauri::{AppHandle, State};
use uuid::Uuid;

use super::assets::AssetState;
use super::scene::{emit_world_delta, SceneState};

/// Holds the lazily-created erosion GPU host (shared across bakes).
#[derive(Default)]
pub struct ErosionState {
    host: Mutex<ErosionHost>,
}

/// Bake `steps` of erosion onto `entity`'s terrain. `region` is an optional
/// terrain-local world AABB `[min_x, min_z, max_x, max_z]`; `None` erodes the
/// whole authored terrain. Commits one undoable height delta and returns the
/// bake report (cells changed, net mass delta, whether the GPU ran).
#[tauri::command]
pub async fn terrain_erode(
    app: AppHandle,
    scene: State<'_, SceneState>,
    erosion: State<'_, ErosionState>,
    entity: String,
    params: ErosionParamsDto,
    steps: u32,
    region: Option<Vec<f64>>,
) -> Result<ErosionReportDto, String> {
    let guid = Uuid::parse_str(&entity).map_err(|e| e.to_string())?;
    let steps = steps.clamp(1, 2000);
    let region = match region {
        Some(v) if v.len() == 4 => Some((DVec2::new(v[0], v[1]), DVec2::new(v[2], v[3]))),
        _ => None,
    };

    let outcome = {
        let mut doc = scene.doc.lock().map_err(|e| e.to_string())?;
        let mut host = erosion.host.lock().map_err(|e| e.to_string())?;
        host.bake(&mut doc, guid, &params, steps, region)
            .ok_or_else(|| "selected entity has no terrain to erode".to_string())?
    };

    // Version was bumped inside the bake; ship the delta so the UI re-syncs.
    emit_world_delta(&app, &scene);

    Ok(ErosionReportDto {
        cells_changed: outcome.cells_changed as u32,
        mass_delta: outcome.mass_delta,
        sediment_moved: outcome.sediment_moved,
        used_gpu: outcome.used_gpu,
        steps,
    })
}

// ── the import wizard (P16.4a) ──────────────────────────────────────────────

/// Read a heightmap's header (no pixel decode) and return its shape plus the
/// settings the wizard should open with.
#[tauri::command]
pub async fn terrain_probe_heightmap(path: String) -> Result<HeightmapProbeDto, String> {
    let source = PathBuf::from(&path);
    // Header IO is trivial, but it is still IO — keep it off the async workers.
    tauri::async_runtime::spawn_blocking(move || {
        let probe = terrain_import::probe(&source).map_err(|e| e.to_string())?;
        let suggested = terrain_import::suggested_settings(&probe);
        Ok(HeightmapProbeDto {
            path: source.to_string_lossy().into_owned(),
            format: probe.format.label().to_string(),
            width: probe.width,
            height: probe.height,
            bit_depth: probe.bit_depth,
            float_samples: probe.float_samples,
            channel: probe.channel.clone(),
            suggested: TerrainImportSettingsDto::from_settings(&suggested),
        })
    })
    .await
    .map_err(|e| format!("terrain_probe_heightmap task failed to run: {e}"))?
}

/// The world a `width × height` source would become under `settings` — the
/// wizard's live extent readback. Pure arithmetic; no file is touched.
#[tauri::command]
pub async fn terrain_import_plan(
    width: u32,
    height: u32,
    settings: TerrainImportSettingsDto,
) -> Result<TerrainImportPlanDto, String> {
    let s = settings.to_settings();
    let import = s.to_import(width, height);
    let grid = HeightmapGrid::new(width, height, &import);
    let (x, z) = s.world_extent(width, height);
    Ok(TerrainImportPlanDto {
        extent_x_m: x,
        extent_z_m: z,
        tiles_x: grid.ntx.max(0) as u32,
        tiles_z: grid.ntz.max(0) as u32,
        tiles: grid.tile_count() as u64,
    })
}

/// Queue a chunked heightmap import. Returns the job id; progress arrives on
/// `assets://import` (phase `"progress"` carries `done`/`total` tiles).
#[tauri::command]
pub async fn terrain_import(
    path: String,
    settings: TerrainImportSettingsDto,
    name: Option<String>,
    state: State<'_, AssetState>,
) -> Result<u64, String> {
    state.submit_terrain_import(PathBuf::from(&path), settings.to_settings(), name)
}

/// Ask an in-flight terrain import to stop. The job still reports terminally
/// (`phase: "failed"` with a cancellation message), so the wizard's state machine
/// needs no special case. `false` for an unknown or already-finished job.
#[tauri::command]
pub async fn terrain_import_cancel(job: u64, state: State<'_, AssetState>) -> Result<bool, String> {
    state.cancel_import(job)
}

/// Details of a finished terrain asset, for the wizard's done state.
///
/// Read back off the **payload's own directory**, not the settings that produced
/// it: once the asset exists it is the authority on what it contains. `width`/
/// `height` are therefore the *lattice's* sample extent (the source rounded up to
/// whole tiles), which is what the terrain actually covers.
#[tauri::command]
pub async fn terrain_asset_info(
    asset_id: String,
    state: State<'_, AssetState>,
) -> Result<TerrainImportResultDto, String> {
    let id = asset_id
        .parse::<inf_asset::AssetId>()
        .map_err(|e| e.to_string())?;
    state.with_project(|proj| {
        let entry = proj
            .db()
            .get(id)
            .ok_or_else(|| format!("unknown asset {asset_id}"))?;
        if entry.kind() != inf_asset::AssetKind::Terrain {
            return Err(format!("asset {asset_id} is not a terrain"));
        }
        let name = entry.name.clone();
        let payload = inf_terrain::read_terrain_asset(&entry.path).map_err(|e| e.to_string())?;
        let bytes = payload.as_bytes().len() as u64;
        let reader = payload.reader();
        let res = reader.tile_resolution();
        let mps = reader.meters_per_sample();
        // The level-0 lattice, read back off the directory (the asset is the
        // source of truth once it exists — no settings round-trip needed).
        let lod0: Vec<(i32, i32)> = reader
            .keys()
            .filter(|k| k.is_lod0())
            .map(|k| k.coord)
            .collect();
        let (mut min_x, mut max_x, mut min_z, mut max_z) = (0, 0, 0, 0);
        for (i, &(x, z)) in lod0.iter().enumerate() {
            if i == 0 {
                (min_x, max_x, min_z, max_z) = (x, x, z, z);
            }
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
        }
        let tiles_x = (max_x - min_x + 1).max(0) as u32;
        let tiles_z = (max_z - min_z + 1).max(0) as u32;
        let cells = (res.max(2) - 1) as f64;
        Ok(TerrainImportResultDto {
            asset: asset_id.clone(),
            name,
            width: (tiles_x as f64 * cells) as u32 + 1,
            height: (tiles_z as f64 * cells) as u32 + 1,
            tiles_x,
            tiles_z,
            tiles: reader.tile_count() as u64,
            lod_levels: reader.lod_levels(),
            extent_x_m: tiles_x as f64 * cells * mps,
            extent_z_m: tiles_z as f64 * cells * mps,
            bytes,
        })
    })
}

/// Spawn a `Terrain` entity that **streams** from `asset_id` — no tiles in the
/// document, the whole heightfield paged from the `.inf_terrain` by the editor
/// streamer. One undoable edit (the standard `SceneDoc` command path), so Ctrl+Z
/// removes it like any other spawn. Returns the new entity GUID.
#[tauri::command]
pub async fn terrain_spawn_streamed(
    app: AppHandle,
    scene: State<'_, SceneState>,
    assets: State<'_, AssetState>,
    asset_id: String,
) -> Result<String, String> {
    let id = asset_id
        .parse::<inf_asset::AssetId>()
        .map_err(|e| e.to_string())?;
    // Read the asset's own grid configuration so the component agrees with the
    // pages it will be handed (resolution + spacing are asset-wide facts).
    let (name, tile_resolution, meters_per_sample) = assets.with_project(|proj| {
        let entry = proj
            .db()
            .get(id)
            .ok_or_else(|| format!("unknown asset {asset_id}"))?;
        if entry.kind() != inf_asset::AssetKind::Terrain {
            return Err(format!("asset {asset_id} is not a terrain"));
        }
        let payload = inf_terrain::read_terrain_asset(&entry.path).map_err(|e| e.to_string())?;
        let header = *payload.header();
        Ok((
            entry.name.clone(),
            header.tile_resolution,
            header.meters_per_sample,
        ))
    })?;

    let guid = {
        let mut doc = scene.doc.lock().map_err(|e| e.to_string())?;
        let guid =
            doc.edit_create_streamed_terrain(&name, id.uuid(), tile_resolution, meters_per_sample);
        doc.select(&[guid], false);
        guid
    };
    emit_world_delta(&app, &scene);
    tracing::info!("terrain: spawned streamed terrain {guid} from {asset_id}");
    Ok(guid.to_string())
}
