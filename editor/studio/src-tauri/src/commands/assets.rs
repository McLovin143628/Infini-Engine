//! Asset / Content Drawer commands (Phase 4).
//!
//! The [`AssetProject`] (asset DB + import cache over the project content root)
//! lives here behind an `Arc<Mutex<…>>` shared with the background import
//! worker. Commands mutate it and emit `assets://changed` (a version bump the
//! frontend re-fetches on); a background tick drains import-job progress
//! (`assets://import`) and the file watcher (external edits → rescan → changed).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use inf_asset::{AssetChange, AssetId, AssetWatcher};
use inf_editor_core::assets::{data, snapshot, AssetProject, ImportProgress, ImportQueue};
use inf_editor_core::ipc::{
    AssetChanged, AssetRefDto, AssetSnapshot, DataAssetDto, DeleteResult, ImportEventDto,
};
use inf_editor_core::thumbnail::{ThumbnailCache, Thumbnailer};
use inf_material::MaterialAsset;
use tauri::{AppHandle, Emitter, Manager, State};

/// The debounce for the content watcher, and the background tick interval.
const WATCH_DEBOUNCE: Duration = Duration::from_millis(250);
const TICK: Duration = Duration::from_millis(120);

/// Lazily-initialized asset subsystem (set up in [`init_assets_on_boot`]).
#[derive(Default)]
pub struct AssetState {
    inner: Mutex<Option<AssetInner>>,
}

struct AssetInner {
    project: Arc<Mutex<AssetProject>>,
    queue: ImportQueue,
    watcher: Option<AssetWatcher>,
    thumb: Thumbnailer,
    cache: ThumbnailCache,
}

impl AssetState {
    /// The display name of an asset (for the drag-to-scene placeholder). Public
    /// so the scene command can name a dropped-asset entity.
    pub fn asset_name(&self, id: AssetId) -> Option<String> {
        let guard = self.inner.lock().ok()?;
        let inner = guard.as_ref()?;
        let proj = inner.project.lock().ok()?;
        proj.db().get(id).map(|e| e.name.clone())
    }

    fn with_project<R>(
        &self,
        f: impl FnOnce(&mut AssetProject) -> Result<R, String>,
    ) -> Result<R, String> {
        let guard = self.inner.lock().map_err(|e| e.to_string())?;
        let inner = guard.as_ref().ok_or("assets not initialized")?;
        let mut proj = inner.project.lock().map_err(|e| e.to_string())?;
        f(&mut proj)
    }

    /// Rebuild the asset subsystem against a new content root (on project
    /// open/switch). The old import worker + watcher stop when their `AssetInner`
    /// is dropped; the shared background tick picks up the new one. Emits
    /// `assets://changed` so the Content Drawer re-syncs.
    pub fn reroot(&self, app: &AppHandle, content_root: PathBuf) {
        match build_inner(app, &content_root) {
            Some(inner) => {
                if let Ok(mut guard) = self.inner.lock() {
                    *guard = Some(inner);
                }
                tracing::info!("asset system re-rooted to {}", content_root.display());
            }
            None => tracing::error!("asset re-root failed for {}", content_root.display()),
        }
        emit_changed(app, self);
    }
}

/// Build a fresh [`AssetInner`] rooted at `content_root`: open the project (seed
/// starter content if empty), spawn the import worker + file watcher, and open
/// the (content-hash-keyed, shared) thumbnail cache. `None` on failure.
fn build_inner(app: &AppHandle, content_root: &std::path::Path) -> Option<AssetInner> {
    let project = match AssetProject::open(content_root) {
        Ok(p) => Arc::new(Mutex::new(p)),
        Err(e) => {
            tracing::error!("asset project open failed: {e}");
            return None;
        }
    };
    seed_starter_content(&project, content_root);

    let queue = ImportQueue::spawn(project.clone());
    let watcher = AssetWatcher::watch(content_root, WATCH_DEBOUNCE)
        .map_err(|e| tracing::warn!("asset watcher: {e}"))
        .ok();
    let cache_dir = app.path().app_data_dir().ok()?.join("thumbnails");
    let cache = match ThumbnailCache::open(cache_dir) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("thumbnail cache: {e}");
            return None;
        }
    };
    Some(AssetInner {
        project,
        queue,
        watcher,
        thumb: Thumbnailer::default(),
        cache,
    })
}

/// Boot the asset subsystem at the default content root (`<app_data>/Content`),
/// so the app works before any project is opened. Opening a project re-roots it
/// via [`AssetState::reroot`]. Starts the background progress/rescan tick once.
pub fn init_assets_on_boot(app: &AppHandle) {
    let Ok(base) = app.path().app_data_dir() else {
        tracing::warn!("no app data dir; asset system disabled");
        return;
    };
    let root = base.join("Content");
    if let Some(inner) = build_inner(app, &root) {
        if let Some(state) = app.try_state::<AssetState>() {
            *state.inner.lock().expect("asset state") = Some(inner);
        }
    }
    spawn_tick(app.clone());
    tracing::info!("asset system ready ({})", root.display());
}

/// Seed a few starter material assets on a fresh (empty) content root so the
/// Content Drawer isn't blank on first run.
fn seed_starter_content(project: &Arc<Mutex<AssetProject>>, root: &std::path::Path) {
    let Ok(mut proj) = project.lock() else { return };
    if !proj.db().is_empty() {
        return;
    }
    let _ = root;
    let Ok(dir) = proj.content_dir("Materials") else {
        return;
    };
    let starters = [
        ("DefaultLit", [0.8, 0.8, 0.8, 1.0], 0.0, 0.5),
        ("Metal", [0.9, 0.9, 0.92, 1.0], 1.0, 0.25),
        ("Rubber", [0.1, 0.1, 0.1, 1.0], 0.0, 0.9),
    ];
    for (name, base_color, metallic, roughness) in starters {
        let mat = MaterialAsset {
            base_color,
            metallic,
            roughness,
            ..Default::default()
        };
        let _ = proj.write_asset(&dir, name, &mat, None, vec![], None);
    }
}

/// The background tick: drain import progress → `assets://import`, drain the
/// file watcher → rescan → `assets://changed`.
fn spawn_tick(app: AppHandle) {
    std::thread::Builder::new()
        .name("asset-tick".into())
        .spawn(move || loop {
            std::thread::sleep(TICK);
            let Some(state) = app.try_state::<AssetState>() else {
                continue;
            };
            let mut changed = false;
            let version;
            {
                let mut guard = match state.inner.lock() {
                    Ok(g) => g,
                    Err(_) => continue,
                };
                let Some(inner) = guard.as_mut() else {
                    continue;
                };
                for ev in inner.queue.poll() {
                    if matches!(ev, ImportProgress::Finished { .. }) {
                        changed = true;
                    }
                    let _ = app.emit("assets://import", import_event(&ev));
                }
                if let Some(w) = &inner.watcher {
                    let batch = w.drain();
                    if !batch.is_empty() {
                        if let Ok(mut proj) = inner.project.lock() {
                            for c in batch {
                                match c {
                                    AssetChange::Upserted(p) => {
                                        let _ = proj.db_mut().rescan_path(&p);
                                    }
                                    AssetChange::Removed(p) => {
                                        proj.db_mut().remove_path(&p);
                                    }
                                }
                            }
                            proj.bump();
                            changed = true;
                        }
                    }
                }
                version = match inner.project.lock() {
                    Ok(proj) => proj.version(),
                    Err(_) => 0,
                };
            }
            if changed {
                let _ = app.emit("assets://changed", AssetChanged { version });
            }
        })
        .expect("spawn asset tick");
}

fn import_event(ev: &ImportProgress) -> ImportEventDto {
    match ev {
        ImportProgress::Started { id, source } => ImportEventDto {
            job: *id,
            source: source.to_string_lossy().into_owned(),
            phase: "started".into(),
            produced: vec![],
            primary: None,
            cached: false,
            error: None,
        },
        ImportProgress::Finished {
            id,
            source,
            produced,
            primary,
            cached,
        } => ImportEventDto {
            job: *id,
            source: source.to_string_lossy().into_owned(),
            phase: "finished".into(),
            produced: produced.iter().map(|a| a.to_string()).collect(),
            primary: primary.map(|a| a.to_string()),
            cached: *cached,
            error: None,
        },
        ImportProgress::Failed { id, source, error } => ImportEventDto {
            job: *id,
            source: source.to_string_lossy().into_owned(),
            phase: "failed".into(),
            produced: vec![],
            primary: None,
            cached: false,
            error: Some(error.clone()),
        },
    }
}

fn emit_changed(app: &AppHandle, state: &AssetState) {
    let version = state.with_project(|p| Ok(p.version())).unwrap_or_default();
    let _ = app.emit("assets://changed", AssetChanged { version });
}

fn parse_id(s: &str) -> Result<AssetId, String> {
    s.parse::<AssetId>()
        .map_err(|e| format!("bad asset id: {e}"))
}

// ── reads ──────────────────────────────────────────────────────────────────

/// The full content snapshot (Content Drawer load + resync).
#[tauri::command]
pub async fn assets_snapshot(state: State<'_, AssetState>) -> Result<AssetSnapshot, String> {
    state.with_project(|p| Ok(snapshot::build(p)))
}

/// The assets that reference `id` (the "Show References" context action).
#[tauri::command]
pub async fn asset_references(
    id: String,
    state: State<'_, AssetState>,
) -> Result<Vec<AssetRefDto>, String> {
    let id = parse_id(&id)?;
    state.with_project(|p| {
        Ok(p.referenced_by(id)
            .into_iter()
            .map(|r| snapshot::ref_dto(p, r))
            .collect())
    })
}

/// A rendered thumbnail as a PNG data URL, or `null` if the kind has no preview
/// (or a 3D kind with no GPU adapter). Cached on disk by content hash.
#[tauri::command]
pub async fn asset_thumbnail(
    id: String,
    state: State<'_, AssetState>,
) -> Result<Option<String>, String> {
    let id = parse_id(&id)?;
    let mut guard = state.inner.lock().map_err(|e| e.to_string())?;
    let inner = guard.as_mut().ok_or("assets not initialized")?;
    let path = {
        let proj = inner.project.lock().map_err(|e| e.to_string())?;
        inner.cache.get_or_render(&proj, id, &mut inner.thumb)
    };
    let Some(path) = path else { return Ok(None) };
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    Ok(Some(format!("data:image/png;base64,{}", base64(&bytes))))
}

// ── mutations ───────────────────────────────────────────────────────────────

/// Import external files into a destination folder (default "Imported"). Returns
/// the queued job ids; progress arrives on `assets://import`.
#[tauri::command]
pub async fn asset_import(
    sources: Vec<String>,
    dest_folder: Option<String>,
    state: State<'_, AssetState>,
) -> Result<Vec<u64>, String> {
    let sub = dest_folder.unwrap_or_else(|| "Imported".into());
    let mut guard = state.inner.lock().map_err(|e| e.to_string())?;
    let inner = guard.as_mut().ok_or("assets not initialized")?;
    let dest = {
        let proj = inner.project.lock().map_err(|e| e.to_string())?;
        proj.content_dir(&sub).map_err(|e| e.to_string())?
    };
    let jobs = sources
        .into_iter()
        .map(|s| inner.queue.submit(PathBuf::from(s), dest.clone()))
        .collect();
    Ok(jobs)
}

/// Create a new authored asset: "mat" (Material), or a data kind
/// "struct"/"enum"/"table" (P4.5). Returns the new GUID.
#[tauri::command]
pub async fn asset_create(
    app: AppHandle,
    kind: String,
    folder: Option<String>,
    name: Option<String>,
    state: State<'_, AssetState>,
) -> Result<String, String> {
    let default_folder = match kind.as_str() {
        "mat" => "Materials",
        _ => "Data",
    };
    let sub = folder.unwrap_or_else(|| default_folder.to_string());
    let name = name.unwrap_or_else(|| match kind.as_str() {
        "mat" => "New Material".into(),
        "struct" => "NewStruct".into(),
        "enum" => "NewEnum".into(),
        "table" => "NewTable".into(),
        _ => "New Asset".into(),
    });
    let id = state.with_project(|p| {
        let dir = p.content_dir(&sub).map_err(|e| e.to_string())?;
        match kind.as_str() {
            "mat" => p
                .write_asset(&dir, &name, &MaterialAsset::default(), None, vec![], None)
                .map_err(|e| e.to_string()),
            "struct" | "enum" | "table" => {
                data::create_default(p, &kind, &dir, &name).map_err(|e| e.to_string())
            }
            other => Err(format!("cannot create asset kind: {other}")),
        }
    })?;
    emit_changed(&app, &state);
    Ok(id.to_string())
}

/// Load a data asset (struct/enum/table) for editing. `null` if not a data kind.
#[tauri::command]
pub async fn asset_data(
    id: String,
    state: State<'_, AssetState>,
) -> Result<Option<DataAssetDto>, String> {
    let id = parse_id(&id)?;
    state.with_project(|p| data::to_dto(p, id).map_err(|e| e.to_string()))
}

/// Save an edited data asset.
#[tauri::command]
pub async fn asset_data_save(
    app: AppHandle,
    data_asset: DataAssetDto,
    state: State<'_, AssetState>,
) -> Result<(), String> {
    state.with_project(|p| data::save_dto(p, &data_asset).map_err(|e| e.to_string()))?;
    emit_changed(&app, &state);
    Ok(())
}

/// Import a CSV/JSON file into an existing table asset, replacing its contents.
#[tauri::command]
pub async fn asset_table_import(
    app: AppHandle,
    id: String,
    source: String,
    state: State<'_, AssetState>,
) -> Result<(), String> {
    let id = parse_id(&id)?;
    state.with_project(|p| {
        data::import_table_into(p, id, &PathBuf::from(&source)).map_err(|e| e.to_string())
    })?;
    emit_changed(&app, &state);
    Ok(())
}

/// The generated Rust source for a struct/enum asset (codegen preview).
#[tauri::command]
pub async fn asset_rust_source(id: String, state: State<'_, AssetState>) -> Result<String, String> {
    let id = parse_id(&id)?;
    state.with_project(|p| data::rust_source(p, id).map_err(|e| e.to_string()))
}

/// Delete an asset. When still referenced (and `force` is false) nothing is
/// deleted and the referrers are returned so the UI can warn.
#[tauri::command]
pub async fn asset_delete(
    app: AppHandle,
    id: String,
    force: bool,
    state: State<'_, AssetState>,
) -> Result<DeleteResult, String> {
    let id = parse_id(&id)?;
    let result = state.with_project(|p| {
        let blockers = p.delete(id, force).map_err(|e| e.to_string())?;
        Ok(if blockers.is_empty() {
            DeleteResult {
                deleted: true,
                blockers: vec![],
            }
        } else {
            DeleteResult {
                deleted: false,
                blockers: blockers
                    .into_iter()
                    .map(|b| snapshot::ref_dto(p, b))
                    .collect(),
            }
        })
    })?;
    if result.deleted {
        emit_changed(&app, &state);
    }
    Ok(result)
}

/// Rename an asset.
#[tauri::command]
pub async fn asset_rename(
    app: AppHandle,
    id: String,
    name: String,
    state: State<'_, AssetState>,
) -> Result<(), String> {
    let id = parse_id(&id)?;
    state.with_project(|p| p.rename(id, &name).map_err(|e| e.to_string()))?;
    emit_changed(&app, &state);
    Ok(())
}

/// Duplicate an asset (fresh GUID). Returns the new id.
#[tauri::command]
pub async fn asset_duplicate(
    app: AppHandle,
    id: String,
    state: State<'_, AssetState>,
) -> Result<String, String> {
    let id = parse_id(&id)?;
    let new_id = state.with_project(|p| p.duplicate(id).map_err(|e| e.to_string()))?;
    emit_changed(&app, &state);
    Ok(new_id.to_string())
}

/// Replace an asset's tags.
#[tauri::command]
pub async fn asset_set_tags(
    app: AppHandle,
    id: String,
    tags: Vec<String>,
    state: State<'_, AssetState>,
) -> Result<(), String> {
    let id = parse_id(&id)?;
    state.with_project(|p| p.set_tags(id, tags).map_err(|e| e.to_string()))?;
    emit_changed(&app, &state);
    Ok(())
}

/// Standard base64 (no line breaks) for thumbnail data URLs — avoids a dep.
fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | b2 as u32;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            T[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            T[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
