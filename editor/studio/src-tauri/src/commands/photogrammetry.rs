//! **The capture wizard's Ring-2 door** (P25.4): photographs to a standard
//! asset, in nine commands and one event channel.
//!
//! Every rule lives in [`inf_editor_core::capture`] (see its module docs);
//! everything here is the string-to-enum hop at the wire, the two `State`
//! lookups, the offscreen preview and one event emission per side effect — the
//! typed-IPC law, the shape `commands::character` already is.
//!
//! # One session per process, and why that is not a limitation
//!
//! A capture holds decoded photographs, six depth maps and a fused mesh — tens
//! of megabytes that live for the length of a wizard. Two of them at once is not
//! a thing the dialog can show, so [`PhotogrammetrySession`] refuses a second
//! run by name rather than queueing it, and this state holds exactly one.
//!
//! # The progress tick, and why it is its own thread
//!
//! The session's worker publishes onto an in-process channel; a webview cannot
//! read one. So a tick thread drains it and re-emits on
//! `photogrammetry://progress`, on the asset tick's own two rules: it holds the
//! session mutex only for the drain, and it never blocks on anything a run is
//! holding. It is spawned on the first `photo_start` and lives for the process,
//! because a wizard opened twice must not get two of them.
//!
//! # The preview is two images, and the reason is written down
//!
//! `photo_preview` renders the finished **geometry** through
//! [`Thumbnailer::render_geometry_view`] — the same offscreen door the Model
//! Editor draws an edit mesh with, with the same neutral surface — and hands the
//! baked **base colour** back beside it, decoded on the CPU. Binding a real
//! texture inside `PreviewSession` is the standing P7 follow-up its own docs
//! name; inventing a second textured offscreen path for one wizard would be a
//! second answer to "what does a preview look like". Both halves degrade
//! independently: no adapter loses the geometry and keeps the atlas.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use inf_core::job::JobPool;
use inf_editor_core::capture::{CaptureIssue, CaptureState, PhotogrammetrySession, SCAN_FOLDER};
use inf_editor_core::dcc::DCC_SURFACE_WGSL;
use inf_editor_core::ipc::{
    CaptureImportDto, CaptureIssueDto, CapturePreviewDto, CaptureProgressDto, CaptureResultDto,
    CaptureSettingsDto, CaptureStatusDto,
};
use inf_editor_core::thumbnail::{encode_png_fast, texture_preview_rgba, Thumbnailer};
use tauri::{AppHandle, Emitter, Manager as _, State};

use super::assets::{emit_changed, AssetState};

/// The preview edge length. 256², the P23.2a measurement: the render is
/// sub-millisecond there and the base64 transport is the real cost.
const PREVIEW_SIZE: u32 = 256;

/// How often the tick drains the session's progress channel.
const TICK: std::time::Duration = std::time::Duration::from_millis(120);

/// The one capture session, its worker pool and its offscreen renderer.
pub struct PhotogrammetryState {
    session: Mutex<PhotogrammetrySession>,
    /// This wizard's own renderer, like `DccState`'s. Separate because a
    /// `Thumbnailer` owns its `PreviewSession`, and that session owns the
    /// uploaded geometry — sharing one with the Model Editor would have the two
    /// overwrite each other's vertex buffer on every alternation.
    preview: Mutex<Thumbnailer>,
    /// **The capture's own workers — the process's SECOND [`JobPool`], and the
    /// ruling that allows one.**
    ///
    /// P7.0's doctrine is a single process-wide [`inf_core::job::global`] pool
    /// and the reason it gives for it is "no oversubscription".
    /// [`JobPool::new`]'s own docs carve out the exception this takes —
    /// "construct one explicitly for a subsystem that wants an isolated pool" —
    /// and a reconstruction is what that exception is for: a **minutes-long**
    /// batch, on the only path in the editor that has one, which would otherwise
    /// sit in front of every `parallel_map` the editor makes while it runs (the
    /// glTF importer's among them). This is the only explicit pool in the editor.
    ///
    /// The thread COUNT is the part that costs something, and it is a choice
    /// rather than an oversight. Both pools are sized to
    /// `available_parallelism`, so while a capture and an import are running at
    /// once the process has **2N runnable workers**. The alternative to
    /// oversubscribing for the length of one batch is starving the editor for
    /// the length of one batch, and an editor operation is seconds where a
    /// capture is minutes. Nothing here has been measured, which is exactly why
    /// it should not be *tuned* without measuring: a smaller capture pool would
    /// be a number somebody guessed, and P23's law covers prescriptions like
    /// that one too.
    pool: Arc<JobPool>,
    /// Whether the progress tick has been spawned.
    ticking: AtomicBool,
}

impl Default for PhotogrammetryState {
    fn default() -> Self {
        Self {
            session: Mutex::new(PhotogrammetrySession::new()),
            preview: Mutex::new(Thumbnailer::new(PREVIEW_SIZE)),
            pool: Arc::new(JobPool::with_available_parallelism()),
            ticking: AtomicBool::new(false),
        }
    }
}

impl PhotogrammetryState {
    fn with<T>(
        &self,
        f: impl FnOnce(&mut PhotogrammetrySession) -> Result<T, String>,
    ) -> Result<T, String> {
        let mut guard = self.session.lock().map_err(|e| e.to_string())?;
        f(&mut guard)
    }
}

/// The whole panel state, read off the session.
fn status(session: &PhotogrammetrySession) -> CaptureStatusDto {
    let state = session.state();
    let settings = CaptureSettingsDto::from_config(session.config());
    // Which of the three places a finding lives in is a rule, so it lives in
    // Ring 1: `findings` is the pre-flight's before a run, the product's after
    // one, and a FAILED run's own — which this used to drop on the floor at the
    // one moment they matter most.
    let issues: Vec<CaptureIssueDto> = session
        .findings()
        .iter()
        .map(CaptureIssueDto::from_issue)
        .collect();
    let result = match state {
        CaptureState::Ready | CaptureState::Imported => {
            session.with_product(|p| CaptureResultDto::from_product(p, settings.metres_per_unit))
        }
        _ => None,
    };
    CaptureStatusDto {
        state: state.name().to_string(),
        stage: match state {
            CaptureState::Running(stage) => Some(stage.name().to_string()),
            _ => None,
        },
        run: session.run_id(),
        photos: session
            .photos()
            .iter()
            .map(|p| inf_editor_core::ipc::PhotoEntryDto {
                path: p.path.display().to_string(),
                name: p.name.clone(),
                width: p.width,
                height: p.height,
                error: p.error.clone(),
            })
            .collect(),
        settings,
        issues,
        result,
        error: session.error(),
        folder: SCAN_FOLDER.to_string(),
    }
}

/// The current panel state. Cheap; the dialog polls it after every command.
#[tauri::command]
pub async fn photo_status(
    state: State<'_, PhotogrammetryState>,
) -> Result<CaptureStatusDto, String> {
    state.with(|s| Ok(status(s)))
}

/// **Load a set of photographs**, replacing whatever was loaded, and run the
/// pre-flight over them. Reconstructs nothing.
#[tauri::command]
pub async fn photo_load(
    paths: Vec<String>,
    state: State<'_, PhotogrammetryState>,
) -> Result<CaptureStatusDto, String> {
    let paths: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    state.with(|s| {
        // The findings are read back off `status` a line later, so the return
        // value is dropped and the REFUSAL is not: a load during a run is
        // `Busy`, by name, exactly as a second `photo_start` is.
        s.load_photos(&paths).map_err(|e| e.to_string())?;
        Ok(status(s))
    })
}

/// Replace the settings. Takes effect on the next run or re-finish.
#[tauri::command]
pub async fn photo_set_settings(
    settings: CaptureSettingsDto,
    state: State<'_, PhotogrammetryState>,
) -> Result<CaptureStatusDto, String> {
    state.with(|s| {
        let cfg = settings.to_config(s.config());
        s.set_config(cfg);
        Ok(status(s))
    })
}

/// **Start a reconstruction.** Returns the run id; progress arrives on
/// `photogrammetry://progress`.
#[tauri::command]
pub async fn photo_start(
    app: AppHandle,
    state: State<'_, PhotogrammetryState>,
) -> Result<CaptureStatusDto, String> {
    let pool = state.pool.clone();
    state.with(|s| s.start(pool).map_err(|e| e.to_string()))?;
    spawn_tick(&app);
    state.with(|s| Ok(status(s)))
}

/// **Re-run the finish stage alone** over the reconstruction in hand — how the
/// scale step, the triangle budget and the atlas size are changed without paying
/// for structure from motion and the dense solve again.
#[tauri::command]
pub async fn photo_refinish(
    app: AppHandle,
    state: State<'_, PhotogrammetryState>,
) -> Result<CaptureStatusDto, String> {
    let pool = state.pool.clone();
    state.with(|s| s.refinish(pool).map_err(|e| e.to_string()))?;
    spawn_tick(&app);
    state.with(|s| Ok(status(s)))
}

/// Ask an in-flight run to stop. `false` when nothing is running.
///
/// It stops **between stages** — the granularity is documented on the session
/// and said in the panel, because a Cancel that looks instant and is not is
/// worse than one that says what it costs. `true` means the flag was set, not
/// that the run will stop: `Finish` is the last automatic stage, so a Cancel
/// that lands inside it finds nothing left to skip and the run completes into
/// `Ready`. Nothing is written either way, which is the guarantee.
#[tauri::command]
pub async fn photo_cancel(state: State<'_, PhotogrammetryState>) -> Result<bool, String> {
    state.with(|s| Ok(s.cancel()))
}

/// Forget the photographs, the reconstruction and the state.
#[tauri::command]
pub async fn photo_reset(
    state: State<'_, PhotogrammetryState>,
) -> Result<CaptureStatusDto, String> {
    state.with(|s| {
        s.reset();
        Ok(status(s))
    })
}

/// **Preview the finished scan** — the geometry through the offscreen path, the
/// baked base colour beside it.
///
/// `yaw`/`pitch` orbit the framing camera the mesh's own bounds fit. Both images
/// are optional and independent: no GPU adapter loses the render and keeps the
/// atlas.
#[tauri::command]
pub async fn photo_preview(
    yaw: f32,
    pitch: f32,
    size: Option<u32>,
    state: State<'_, PhotogrammetryState>,
) -> Result<CapturePreviewDto, String> {
    let size = size.unwrap_or(PREVIEW_SIZE).clamp(64, 1024);
    // Everything the render needs, copied out under a short session lock: the
    // render itself must not hold it, or an orbit would block the tick that is
    // reporting the run behind it.
    let job = state.with(|s| {
        s.with_product(|p| {
            let (verts, indices) = p.preview_geometry();
            (
                verts,
                indices,
                p.finished.mesh.bounds,
                p.finished.albedo.clone(),
                p.finished.report.final_triangles as u64,
            )
        })
        .ok_or_else(|| "there is no finished reconstruction to preview".to_string())
    })?;
    let (verts, indices, bounds, albedo, stamp) = job;

    // The base colour is CPU-decoded, so it is there on every machine.
    let atlas = {
        let rgba = texture_preview_rgba(&albedo, size);
        encode_png_fast(size, &rgba)
            .ok()
            .map(|png| format!("data:image/png;base64,{}", super::assets::base64(&png)))
    };

    let mut view = inf_editor_core::dcc::frame(bounds);
    view.yaw_deg = yaw;
    view.pitch_deg = pitch;
    let rendered = {
        let mut prev = state.preview.lock().map_err(|e| e.to_string())?;
        prev.render_geometry_view(&verts, &indices, stamp, DCC_SURFACE_WGSL, size, view)
    };
    let (geometry, error) = match rendered {
        Ok(Some(rgba)) => match encode_png_fast(size, &rgba) {
            Ok(png) => (
                Some(format!(
                    "data:image/png;base64,{}",
                    super::assets::base64(&png)
                )),
                None,
            ),
            Err(e) => (None, Some(e)),
        },
        Ok(None) => (
            None,
            Some("no GPU preview available on this machine".into()),
        ),
        Err(message) => {
            tracing::warn!("photo_preview: {message}");
            (None, Some(message))
        }
    };
    Ok(CapturePreviewDto {
        geometry,
        albedo: atlas,
        error,
        size,
    })
}

/// **Import the finished scan** into the open project under `Content/Scans`,
/// then tell the Content Drawer.
///
/// The five assets are written or none of them are — the session inherits
/// `write_finished`'s rollback. `assets://changed` fires after the write, so the
/// drawer knows about every GUID this returns before anything can reference one.
#[tauri::command]
pub async fn photo_import(
    app: AppHandle,
    name: String,
    photo: State<'_, PhotogrammetryState>,
    assets: State<'_, AssetState>,
) -> Result<CaptureImportDto, String> {
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err("a scan needs a name".into());
    }
    // **Lock order: the capture session, then the asset project.** It is the
    // only nesting in this module and the only one in the app that involves
    // either — the progress tick takes the session alone, and every asset
    // command takes the project alone — so stating it here is what keeps it
    // true.
    let (ids, notes) = photo.with(|session| {
        let ids = assets
            .with_project(|project| session.import(project, &name).map_err(|e| e.to_string()))?;
        // The notes are read AFTER the write, because the placeholder-cube one
        // is added by it.
        let notes = session
            .with_product(|p| {
                p.issues
                    .iter()
                    .filter(|i| matches!(i, CaptureIssue::NoMeshletDag))
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok((ids, notes))
    })?;
    emit_changed(&app, &assets);
    Ok(CaptureImportDto {
        mesh: ids.mesh.to_string(),
        albedo: ids.albedo.to_string(),
        normal: ids.normal.to_string(),
        orm: ids.orm.to_string(),
        material: ids.material.to_string(),
        folder: SCAN_FOLDER.to_string(),
        name,
        notes,
    })
}

/// Drain the session's progress channel onto `photogrammetry://progress`.
///
/// Spawned once, on the first run, and lives for the process. The two rules the
/// asset tick is written to apply here for the same reasons: the session mutex
/// is held only for the drain, and nothing in the loop can block on what a run
/// is holding.
fn spawn_tick(app: &AppHandle) {
    let Some(state) = app.try_state::<PhotogrammetryState>() else {
        return;
    };
    if state.ticking.swap(true, Ordering::SeqCst) {
        return;
    }
    let app = app.clone();
    let _ = std::thread::Builder::new()
        .name("photogrammetry-tick".into())
        .spawn(move || loop {
            std::thread::sleep(TICK);
            let Some(state) = app.try_state::<PhotogrammetryState>() else {
                continue;
            };
            let events = match state.session.lock() {
                Ok(session) => session.drain(),
                Err(_) => continue,
            };
            for event in &events {
                let _ = app.emit(
                    "photogrammetry://progress",
                    CaptureProgressDto::from_progress(event),
                );
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_editor_core::capture::CaptureConfig;

    /// The panel's whole state is a projection of the session and decides
    /// nothing: an idle session shows its pre-flight, and a session with a
    /// blocking pre-flight says so on the wire.
    #[test]
    fn an_idle_session_reports_its_preflight_and_no_result() {
        let session = PhotogrammetrySession::new();
        let dto = status(&session);
        assert_eq!(dto.state, "idle");
        assert_eq!(dto.stage, None);
        assert!(dto.result.is_none());
        assert!(dto.photos.is_empty());
        assert_eq!(dto.folder, SCAN_FOLDER);
        // No photographs is a blocking finding, and it crosses the wire as one.
        assert!(
            dto.issues.iter().any(|i| i.severity == "blocking"),
            "{:?}",
            dto.issues
        );
        assert!(dto.error.is_none());
    }

    /// A settings edit is a round trip through the Ring-1 config and back, so
    /// the panel can never show a number the next run will not use.
    #[test]
    fn the_settings_the_panel_sends_are_the_settings_it_reads_back() {
        let mut session = PhotogrammetrySession::new();
        let mut dto = CaptureSettingsDto::from_config(&CaptureConfig::default());
        dto.metres_per_unit = 0.125;
        dto.target_triangles = 8_000;
        dto.camera.focal_ratio = 0.9375;
        session.set_config(dto.to_config(session.config()));
        let read_back = status(&session).settings;
        assert_eq!(read_back, dto);
        // …and a scale that is not a scale is a blocking finding, before a run.
        let mut bad = dto;
        bad.metres_per_unit = 0.0;
        session.set_config(bad.to_config(session.config()));
        let issues = status(&session).issues;
        assert!(
            issues
                .iter()
                .any(|i| i.severity == "blocking" && i.message.contains("not a scale")),
            "{issues:?}"
        );
    }

    /// C4-44: `focal_ratio = 0` divides the projection by zero — `0.0/0.0` is
    /// NaN at the principal point — and those observations feed triangulation,
    /// the SfM poses and the finished `.inf_mesh`. The wizard's own
    /// `metres_per_unit`, two fields away, has been validated since P25.4;
    /// these three were copied through unchecked.
    #[test]
    fn an_impossible_lens_is_refused_rather_than_copied_through() {
        let mut session = PhotogrammetrySession::new();
        let good = CaptureSettingsDto::from_config(&CaptureConfig::default());
        let before = session.config().camera;

        for bad in [
            (0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (f64::NAN, 0.0, 0.0),
            (0.9375, f64::INFINITY, 0.0),
            (0.9375, 0.0, f64::NAN),
        ] {
            let mut dto = good.clone();
            dto.camera.focal_ratio = bad.0;
            dto.camera.k1 = bad.1;
            dto.camera.k2 = bad.2;
            session.set_config(dto.to_config(session.config()));
            assert_eq!(
                session.config().camera,
                before,
                "the lens {bad:?} was copied into the config"
            );
        }

        // …and a usable one still lands, so the guard is not simply "refuse".
        let mut ok = good;
        ok.camera.focal_ratio = 0.9375;
        session.set_config(ok.to_config(session.config()));
        assert_eq!(session.config().camera.focal_ratio, 0.9375);
    }
}
