//! **Hydrology authoring commands** (P20.4) — the Ring-2 shims over
//! `inf_editor_core::hydro` and the `SceneDoc::edit_*` water mutations.
//!
//! Five commands, and between them they are the water tool's non-viewport half:
//!
//! * `water_defaults` — what a click at a world point suggests, including the
//!   **biome water hint** (P19.2's `BiomeDef::water_hint`, whose first reader
//!   this is). The command resolves the `.inf_biomes` asset, because Ring 1 does
//!   not reach into the asset database and the viewport thread cannot either.
//! * `water_lake_preview` — where a level lands on the ground inside a rectangle.
//! * `water_create_lake` / `water_create_river` / `water_append_river_point` —
//!   the placements, each ONE undo step, for callers that are not the viewport
//!   (the palette, a script, a test). The native viewport performs the same
//!   edits directly on the document it shares, so there is one implementation
//!   and two entry points rather than two implementations.
//! * `water_set_river_profile` — the per-river width/depth/flow edit.
//! * `water_river_report` — the verdict: the two cook advisories re-run plus the
//!   terrain-aware bed conflicts only the editor can make.
//!
//! Every mutation ends with `emit_world_delta`, so the Outliner and the viewport
//! see the change on the established channel.

use inf_asset::AssetId;
use inf_editor_core::hydro;
use inf_editor_core::ipc::{
    LakePreviewDto, RiverBedConflictDto, RiverClimbDto, RiverReportDto, WaterDefaultsDto,
};
use inf_terrain::BiomeSet;
use tauri::{AppHandle, State};
use uuid::Uuid;

use super::assets::AssetState;
use super::scene::{emit_world_delta, SceneState};

/// Tolerance the tool's climb checks use, metres — **the cook's own constant**,
/// re-exported through Ring 1 rather than restated.
///
/// A tool with a smaller value would nag about rivers the build accepts; a larger
/// one would let a build advisory arrive as a surprise at package time. Since
/// P20.4 there is one definition (`inf_water::UPHILL_TOLERANCE_M`) and both the
/// cook and this command read it, so there is nothing to keep in step.
use inf_editor_core::hydro::UPHILL_TOLERANCE_M as RIVER_UPHILL_TOLERANCE_M;

/// Resolve the `.inf_biomes` set the level's **first bound terrain** names.
///
/// First-bound rather than "the terrain under the cursor", and that is a stated
/// v1 simplification: `terrain_biomes` — the biome tool's own vocabulary query —
/// resolves the same way, so the water tool and the paint tool agree about which
/// vocabulary is in play. A multi-terrain level whose terrains bind *different*
/// sets would take the first one's hints everywhere; per-terrain resolution is a
/// `hydro::water_defaults` signature change away and is ledgered rather than
/// half-built.
///
/// Total by design: a level with no terrain, no bound set, or a set that fails to
/// decode answers `None`, which the defaults treat as "no hint" rather than as an
/// error. This runs on every click.
fn bound_biome_set(scene: &SceneState, assets: &AssetState) -> Option<BiomeSet> {
    let set = {
        let doc = scene.doc.lock().ok()?;
        let guid = doc
            .order()
            .iter()
            .copied()
            .find(|g| doc.terrain_biome_set(*g).is_some())?;
        doc.terrain_biome_set(guid)?
    };
    assets
        .with_project(|proj| Ok(inf_editor_core::assets::biome_set::get(proj, AssetId(set)).ok()))
        .ok()
        .flatten()
}

/// What the water tool suggests for a click at a world point (P20.4).
#[tauri::command]
pub async fn water_defaults(
    x: f64,
    z: f64,
    scene: State<'_, SceneState>,
    assets: State<'_, AssetState>,
) -> Result<WaterDefaultsDto, String> {
    let set = bound_biome_set(&scene, &assets);
    let doc = scene.doc.lock().map_err(|e| e.to_string())?;
    let d = hydro::water_defaults(&doc, set.as_ref(), x, z);
    Ok(WaterDefaultsDto {
        level_m: d.level_m,
        river_width_m: d.river_width_m,
        river_depth_m: d.river_depth_m,
        ground_m: d.ground_m,
        biome_id: d.biome_id,
        biome_name: d.biome_name,
        from_biome_hint: d.from_biome_hint,
    })
}

/// Where a still-water level lands on the terrain inside a rectangle (P20.4).
#[tauri::command]
pub async fn water_lake_preview(
    center_x: f64,
    center_z: f64,
    half_x: f64,
    half_z: f64,
    level_m: f64,
    resolution: u32,
    scene: State<'_, SceneState>,
) -> Result<LakePreviewDto, String> {
    let doc = scene.doc.lock().map_err(|e| e.to_string())?;
    let p = hydro::lake_preview(
        &doc,
        glam::DVec2::new(center_x, center_z),
        glam::DVec2::new(half_x, half_z),
        level_m,
        resolution,
    );
    Ok(LakePreviewDto {
        level_m: p.level_m,
        covered_fraction: p.covered_fraction,
        max_depth_m: p.max_depth_m,
        mean_depth_m: p.mean_depth_m,
        samples: p.samples,
        known: p.known,
        waterline: p
            .waterline
            .iter()
            .flat_map(|[a, b]| [a.x, a.y, b.x, b.y])
            .collect(),
    })
}

/// Place a lake — ONE undo step. Returns the new entity's GUID.
///
/// Flat scalars rather than a DTO, matching `water_lake_preview` right above it:
/// the two are the same rectangle asked two questions, and giving one a struct
/// and the other six numbers would make the pair harder to read than the lint it
/// silences.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn water_create_lake(
    app: AppHandle,
    center_x: f64,
    center_y: f64,
    center_z: f64,
    half_x: f64,
    half_z: f64,
    level_m: f64,
    scene: State<'_, SceneState>,
) -> Result<String, String> {
    let guid = {
        let mut doc = scene.doc.lock().map_err(|e| e.to_string())?;
        doc.edit_create_lake(
            "Lake",
            glam::DVec3::new(center_x, center_y, center_z),
            inf_ecs::Vec2d::new(half_x, half_z),
            level_m,
        )
    };
    emit_world_delta(&app, &scene);
    Ok(guid.to_string())
}

/// Place a river from world-space control points — ONE undo step. `points` is a
/// flat `[x, y, z, …]` array (the `LakePreviewDto::waterline` convention: one
/// array of numbers rather than an array of triples).
#[tauri::command]
pub async fn water_create_river(
    app: AppHandle,
    points: Vec<f64>,
    width_m: f64,
    depth_m: f64,
    flow_m_s: f64,
    scene: State<'_, SceneState>,
) -> Result<String, String> {
    if points.len() < 6 || !points.len().is_multiple_of(3) {
        return Err("a river needs at least two [x, y, z] control points".into());
    }
    let pts: Vec<glam::DVec3> = points
        .chunks_exact(3)
        .map(|c| glam::DVec3::new(c[0], c[1], c[2]))
        .collect();
    let guid = {
        let mut doc = scene.doc.lock().map_err(|e| e.to_string())?;
        doc.edit_create_river("River", &pts, width_m, depth_m, flow_m_s)
    };
    emit_world_delta(&app, &scene);
    Ok(guid.to_string())
}

/// Append a world-space control point to a river's centreline — ONE undo step.
/// Returns whether anything changed (`false` for an entity with no `Spline`).
#[tauri::command]
pub async fn water_append_river_point(
    app: AppHandle,
    entity: String,
    x: f64,
    y: f64,
    z: f64,
    scene: State<'_, SceneState>,
) -> Result<bool, String> {
    let guid = Uuid::parse_str(&entity).map_err(|e| e.to_string())?;
    let changed = {
        let mut doc = scene.doc.lock().map_err(|e| e.to_string())?;
        doc.edit_append_spline_point(guid, glam::DVec3::new(x, y, z))
    };
    if changed {
        emit_world_delta(&app, &scene);
    }
    Ok(changed)
}

/// Rewrite a river's width / depth / flow profile — ONE undo step (P20.4).
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn water_set_river_profile(
    app: AppHandle,
    entity: String,
    width_start_m: f64,
    width_end_m: f64,
    depth_start_m: f64,
    depth_end_m: f64,
    flow_m_s: f64,
    scene: State<'_, SceneState>,
) -> Result<bool, String> {
    let guid = Uuid::parse_str(&entity).map_err(|e| e.to_string())?;
    let changed = {
        let mut doc = scene.doc.lock().map_err(|e| e.to_string())?;
        doc.edit_set_river_profile(
            guid,
            width_start_m,
            width_end_m,
            depth_start_m,
            depth_end_m,
            flow_m_s,
        )
    };
    if changed {
        emit_world_delta(&app, &scene);
    }
    Ok(changed)
}

/// The tool's verdict on one river (P20.4).
#[tauri::command]
pub async fn water_river_report(
    entity: String,
    scene: State<'_, SceneState>,
) -> Result<RiverReportDto, String> {
    let guid = Uuid::parse_str(&entity).map_err(|e| e.to_string())?;
    let doc = scene.doc.lock().map_err(|e| e.to_string())?;
    let r = hydro::river_report(&doc, guid, RIVER_UPHILL_TOLERANCE_M);
    let climb = |s: &hydro::UphillSpan| RiverClimbDto {
        from_s: s.from_s,
        to_s: s.to_s,
        rise_m: s.rise_m,
        gradient: s.gradient(),
    };
    Ok(RiverReportDto {
        entity: guid.to_string(),
        length_m: r.length_m,
        points: r.points,
        fall_m: r.fall_m,
        surface_climbs: r.surface_climbs.iter().map(climb).collect(),
        bed_climbs: r.bed_climbs.iter().map(climb).collect(),
        bed_conflicts: r
            .bed_conflicts
            .iter()
            .map(|c| RiverBedConflictDto {
                issue: c.issue.id().to_string(),
                from_s: c.from_s,
                to_s: c.to_s,
                worst_m: c.worst_m,
                worst_x: c.worst_xz.x,
                worst_z: c.worst_xz.y,
            })
            .collect(),
        sampled_frames: r.sampled_frames,
        total_frames: r.total_frames,
    })
}
