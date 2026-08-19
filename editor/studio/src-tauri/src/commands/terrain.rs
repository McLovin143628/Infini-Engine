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
//!
//! # Biome binding (P19.2)
//!
//! `terrain_biomes` reads the vocabulary the Biome tool paints with (which
//! `.inf_biomes` a terrain names, plus every set in the project for the picker)
//! and `terrain_set_biome_set` rebinds it as one undo step. Both end by pushing
//! the **overlay palettes** to the viewport — see [`push_biome_palettes`].

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use glam::DVec2;
use inf_asset::AssetId;
use inf_editor_core::assets::{biome_set, terrain_import};
use inf_editor_core::erosion_gpu::ErosionHost;
use inf_editor_core::ipc::{
    DataMapExportDto, ErosionParamsDto, ErosionReportDto, HeightmapProbeDto, TerrainBiomesDto,
    TerrainImportPlanDto, TerrainImportResultDto, TerrainImportSettingsDto,
};
use inf_editor_core::scene::SceneDoc;
use inf_terrain::HeightmapGrid;
use tauri::{AppHandle, Manager, State};
use uuid::Uuid;

use super::assets::{biome_def_dto, AssetState};
use super::scene::{emit_world_delta, SceneState};

/// Holds the lazily-created erosion GPU host (shared across bakes).
///
/// `Arc` rather than a bare `Mutex` so [`terrain_erode`] can move a handle into
/// `spawn_blocking` (Hardening Wave E) — see that command for why a bake may not
/// run on an async worker.
#[derive(Default)]
pub struct ErosionState {
    host: Arc<Mutex<ErosionHost>>,
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

    // OFF THE ASYNC WORKERS (Hardening Wave E). A bake is up to 2 000 solver
    // steps over the whole authored heightfield — and on the first call it also
    // creates a headless `GpuContext`, adapter request included. Running that in
    // the body of an `async fn` parks a Tokio worker for its whole duration, and
    // every one of the editor's 239 commands is `async`: the viewport's own
    // per-frame IPC, the autosave tick and the asset watcher all queue behind it.
    // `terrain_probe_heightmap` in this same file states the rule for a *header
    // read*; this is the heaviest thing the module does.
    //
    // The DOC LOCK IS STILL HELD FOR THE WHOLE BAKE, deliberately and
    // unavoidably: `ErosionHost::bake` takes `&mut SceneDoc` — it reads the
    // heightfield, erodes it and commits the undo step through the document — so
    // there is no snapshot to take. What changes is *who* waits: a viewport
    // frame that wants the document still blocks, but the async runtime that
    // serves every other command no longer does. Shortening the hold itself
    // needs `bake` split into "read the region / solve / commit", which is a
    // Ring-1 redesign and not this wave's repair.
    let doc = Arc::clone(&scene.doc);
    let host = Arc::clone(&erosion.host);
    let outcome = tauri::async_runtime::spawn_blocking(move || {
        let mut doc = doc.lock().map_err(|e| e.to_string())?;
        let mut host = host.lock().map_err(|e| e.to_string())?;
        host.bake(&mut doc, guid, &params, steps, region)
            .ok_or_else(|| "selected entity has no terrain to erode".to_string())
    })
    .await
    .map_err(|e| format!("terrain_erode task failed to run: {e}"))??;

    // Version was bumped inside the bake; ship the delta so the UI re-syncs.
    emit_world_delta(&app, &scene);

    Ok(ErosionReportDto {
        cells_changed: outcome.cells_changed as u32,
        map_cells_changed: outcome.map_cells_changed as u32,
        mass_delta: outcome.mass_delta,
        sediment_moved: outcome.sediment_moved,
        used_gpu: outcome.used_gpu,
        steps,
    })
}

/// Export one of the terrain's **erosion data maps** (P19.1) as a 16-bit
/// grayscale PNG under `<content root>/DataMaps/`.
///
/// `map` is `"flow"` / `"deposition"` / `"wear"`; `region` is the same optional
/// terrain-local world AABB `[min_x, min_z, max_x, max_z]` the bake takes.
///
/// The destination is derived, not chosen: writing into the project's own content
/// root keeps this to a read-only file dialog's worth of capability (there is no
/// save-dialog permission in the editor's Tauri capability set) and lands the PNG
/// where the asset watcher will pick it up as importable content. The returned
/// path is what the UI reports.
///
/// The image is normalized over the exported region's own `[min, max]`, which the
/// report states: the stored accumulators are raw and are never rescaled.
#[tauri::command]
pub async fn terrain_export_data_map(
    scene: State<'_, SceneState>,
    assets: State<'_, AssetState>,
    entity: String,
    map: String,
    region: Option<Vec<f64>>,
) -> Result<DataMapExportDto, String> {
    let guid = Uuid::parse_str(&entity).map_err(|e| e.to_string())?;
    let kind = inf_terrain::DataMapKind::from_label(&map)
        .ok_or_else(|| format!("unknown data map {map:?} (expected flow/deposition/wear)"))?;
    let region = match region {
        Some(v) if v.len() == 4 => Some((DVec2::new(v[0], v[1]), DVec2::new(v[2], v[3]))),
        _ => None,
    };
    let root = assets
        .content_root()
        .ok_or_else(|| "no project is open".to_string())?;

    let (png, (min, max), width, height, name) = {
        let doc = scene.doc.lock().map_err(|e| e.to_string())?;
        let (data, _) = doc
            .terrain_data_and_origin(guid)
            .ok_or_else(|| "selected entity has no terrain".to_string())?;
        let (img, range) = data
            .to_data_map_image(kind, region)
            .ok_or_else(|| "the terrain has no authored tiles to export".to_string())?;
        let png = inf_terrain::encode_png16(&img).map_err(|e| e.to_string())?;
        let name = doc.display_name(guid);
        (png, range, img.width, img.height, name)
    };

    let dir = root.join("DataMaps");
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let path = dir.join(format!("{}_{}.png", sanitize(&name), kind.label()));
    let bytes = png.len() as u32;
    // Atomic (C4-29): the file name is deterministic, so a re-export overwrites
    // a committed PNG inside the content root that the watcher has registered as
    // an asset — a torn one is a broken asset, not a missing file.
    inf_asset::write_atomically(&path, &png)
        .map_err(|e| format!("write {}: {e}", path.display()))?;

    Ok(DataMapExportDto {
        map: kind.label().to_string(),
        path: path.to_string_lossy().into_owned(),
        width,
        height,
        bytes,
        min,
        max,
        unit: kind.unit().to_string(),
    })
}

/// A file-name-safe form of an entity name (the export's only naming input).
fn sanitize(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "Terrain".into()
    } else {
        cleaned
    }
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
            absolute_samples: probe.absolute_samples,
            channel: probe.channel.clone(),
            geo: probe
                .geo
                .as_ref()
                .map(inf_editor_core::ipc::GeoReferenceDto::from_meta),
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
///
/// **The open level's geo-anchor rides along** (Wave G). It is read here, on the
/// command thread, and captured into the job: the build runs for minutes with no
/// lock held, so looking the anchor up later would make where a terrain lands
/// depend on whether the author touched World Settings in the meantime. Without
/// this the wizard's "place using the source's georeferencing" switch is inert —
/// the importer has nothing to subtract and every asset lands at the world
/// origin, which is exactly the state the Wave-G audit found it in.
#[tauri::command]
pub async fn terrain_import(
    path: String,
    settings: TerrainImportSettingsDto,
    name: Option<String>,
    state: State<'_, AssetState>,
    scene: State<'_, super::scene::SceneState>,
) -> Result<u64, String> {
    let anchor = {
        let doc = scene.doc.lock().map_err(|e| e.to_string())?;
        let a = doc.geo().clone();
        a.enabled.then_some(a)
    };
    state.submit_terrain_import(PathBuf::from(&path), settings.to_settings(), anchor, name)
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

// ── biome binding + the overlay palette push (P19.2) ─────────────────────────

/// Every terrain entity in the document, in creation order.
///
/// [`SceneDoc::streamed_terrain_entities`] lists only *asset-backed* terrains,
/// and an inline terrain paints biomes exactly like a streamed one — so this
/// filters the document's creation order by "carries a `Terrain`", probed through
/// the existing [`SceneDoc::terrain_data_and_origin`] accessor (a borrow, not a
/// copy) rather than by adding a method to Ring 1.
fn terrain_entities(doc: &SceneDoc) -> Vec<Uuid> {
    doc.order()
        .iter()
        .copied()
        .filter(|&g| doc.terrain_data_and_origin(g).is_some())
        .collect()
}

/// Re-push every terrain's biome **overlay palette** to the viewport.
///
/// The viewport keeps one palette per terrain entity, and it has no way to
/// resolve a `.inf_biomes` itself (Ring 1's asset project lives on this side), so
/// every path that can change a binding or a set's colours has to re-push:
/// `terrain_set_biome_set`, `asset_save_biome_set`, and `terrain_biomes` (cheap,
/// and it means opening a level re-syncs the palettes on the toolbar's first poll
/// without inventing an event for it).
///
/// An unbound — or missing, or unloadable — set pushes an **empty** palette,
/// which clears the viewport's entry. Anything else would leave the Biomes view
/// mode tinting with a vocabulary the terrain no longer names.
///
/// Three locks are reachable here (document, asset project, viewport handle) and
/// none is ever held across a call into another: the document read is collected
/// first, the asset project is entered once for the distinct sets, and only then
/// does the viewport get written.
pub(super) fn push_biome_palettes(app: &AppHandle, assets: &AssetState) {
    let Some(scene) = app.try_state::<SceneState>() else {
        return;
    };
    let bindings: Vec<(Uuid, Option<Uuid>)> = {
        let Ok(doc) = scene.doc.lock() else { return };
        terrain_entities(&doc)
            .into_iter()
            .map(|g| (g, doc.terrain_biome_set(g)))
            .collect()
    };
    if bindings.is_empty() {
        return;
    }

    // Resolve each DISTINCT bound set once, under a single short asset-lock hold
    // (several terrains commonly share one set).
    let mut palettes: HashMap<Uuid, Vec<[f32; 4]>> = HashMap::new();
    // P20.4: the id-indexed water-level hints ride the SAME resolution. A second
    // pass over the same assets, taken under a second lock, is how two
    // projections of one set come to disagree about which set they are.
    let mut hints: HashMap<Uuid, Vec<Option<f64>>> = HashMap::new();
    let wanted: Vec<Uuid> = bindings.iter().filter_map(|&(_, set)| set).collect();
    if !wanted.is_empty() {
        let _ = assets.with_project(|proj| {
            for set in wanted {
                if palettes.contains_key(&set) {
                    continue;
                }
                let loaded = biome_set::get(proj, AssetId(set)).ok();
                palettes.insert(
                    set,
                    loaded.as_ref().map(|s| s.palette()).unwrap_or_default(),
                );
                hints.insert(
                    set,
                    loaded.as_ref().map(|s| s.water_hints()).unwrap_or_default(),
                );
            }
            Ok(())
        });
    }

    let Some(viewport) = app.try_state::<super::ViewportState>() else {
        return;
    };
    // **Broadcast** (P23.2a): a terrain's biome vocabulary and its water hints
    // are properties of the LEVEL, so every viewport tints and resolves the same
    // way. A Primary-only push would give a second viewport the Biomes view mode
    // with an empty palette — silently untinted rather than visibly wrong.
    for (entity, set) in bindings {
        let palette = set
            .and_then(|s| palettes.get(&s))
            .cloned()
            .unwrap_or_default();
        viewport.set_biome_palette(super::Target::All, entity, palette);
        let hint = set.and_then(|s| hints.get(&s)).cloned().unwrap_or_default();
        viewport.set_water_hints(super::Target::All, entity, hint);
    }
}

/// The biome vocabulary the viewport toolbar paints with, for one terrain.
///
/// `entity` names a terrain; `None` resolves to the **first** terrain in the
/// level (creation order), which is what the toolbar wants — it polls this
/// without a selection.
///
/// Deliberately total: a level with no terrain returns `null`, and a terrain with
/// no bound set (or a set that has been deleted, or fails to decode) returns a
/// DTO with `biome_set: null` and an empty `biomes` list. This command is polled
/// every time the tool is entered, and "there is no vocabulary here" is an answer,
/// not a failure — only a malformed entity GUID is an error.
#[tauri::command]
pub async fn terrain_biomes(
    app: AppHandle,
    entity: Option<String>,
    scene: State<'_, SceneState>,
    assets: State<'_, AssetState>,
) -> Result<Option<TerrainBiomesDto>, String> {
    let requested = match entity.as_deref() {
        Some(s) => Some(Uuid::parse_str(s).map_err(|e| e.to_string())?),
        None => None,
    };
    let resolved = {
        let doc = scene.doc.lock().map_err(|e| e.to_string())?;
        let terrains = terrain_entities(&doc);
        let guid = match requested {
            Some(g) => terrains.into_iter().find(|&t| t == g),
            None => terrains.into_iter().next(),
        };
        guid.map(|g| (g, doc.terrain_biome_set(g)))
    };
    let Some((guid, bound)) = resolved else {
        return Ok(None);
    };

    let dto = assets.with_project(|proj| {
        let available = biome_set::list(proj)
            .into_iter()
            .map(|(id, name)| (id.to_string(), name))
            .collect();
        // The set's DISPLAY name is the asset entry's, not the payload's: renaming
        // the asset is what an author sees in the drawer, and the picker below has
        // to agree with it.
        let loaded = bound.and_then(|s| {
            let id = AssetId(s);
            let set = biome_set::get(proj, id).ok()?;
            let name = proj.db().get(id).map(|e| e.name.clone())?;
            Some((id, name, set))
        });
        Ok(match loaded {
            Some((id, name, set)) => TerrainBiomesDto {
                entity: guid.to_string(),
                biome_set: Some(id.to_string()),
                biome_set_name: name,
                biomes: set.biomes.iter().map(biome_def_dto).collect(),
                available,
            },
            None => TerrainBiomesDto {
                entity: guid.to_string(),
                biome_set: None,
                biome_set_name: String::new(),
                biomes: Vec::new(),
                available,
            },
        })
    })?;

    // Both locks are released; keep the viewport's palettes in step with what we
    // just reported (this is the path that re-syncs them after a level open).
    push_biome_palettes(&app, &assets);
    Ok(Some(dto))
}

/// Bind (or clear, with `asset: null`) the `.inf_biomes` set a terrain's painted
/// biome ids name — ONE undo step. Returns whether anything changed (rebinding to
/// the same set records nothing).
#[tauri::command]
pub async fn terrain_set_biome_set(
    app: AppHandle,
    entity: String,
    asset: Option<String>,
    scene: State<'_, SceneState>,
    assets: State<'_, AssetState>,
) -> Result<bool, String> {
    let guid = Uuid::parse_str(&entity).map_err(|e| e.to_string())?;
    let set = match asset.as_deref() {
        Some(s) => Some(s.parse::<AssetId>().map_err(|e| e.to_string())?.uuid()),
        None => None,
    };
    let changed = {
        let mut doc = scene.doc.lock().map_err(|e| e.to_string())?;
        doc.edit_set_terrain_biome_set(guid, set)
    };
    if changed {
        emit_world_delta(&app, &scene);
    }
    // Push unconditionally: a rebind that recorded nothing can still be the call
    // that first arms an entity the viewport has no palette for.
    push_biome_palettes(&app, &assets);
    Ok(changed)
}
