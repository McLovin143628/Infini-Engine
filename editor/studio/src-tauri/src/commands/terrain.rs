//! Terrain erosion bake command (ROADMAP P10.3b).
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

use std::sync::Mutex;

use glam::DVec2;
use inf_editor_core::erosion_gpu::ErosionHost;
use inf_editor_core::ipc::{ErosionParamsDto, ErosionReportDto};
use tauri::{AppHandle, State};
use uuid::Uuid;

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
