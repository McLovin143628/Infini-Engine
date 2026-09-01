//! Project commands (P5.5): create / open / switch projects.
//!
//! The open project lives here behind an `Arc<Mutex<…>>`. Opening (or creating)
//! a project re-roots the asset database to `<project>/Content`
//! ([`AssetState::reroot`]), records it in the recent list, and emits
//! `project://changed` so the frontend leaves the start screen and re-syncs.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use inf_editor_core::ipc::{ProjectInfoDto, ProjectTemplateDto, RecentProjectDto};
use inf_project::{Project, ProjectTemplate, RecentProjects};
use tauri::{AppHandle, Emitter, Manager, State};

use super::assets::AssetState;

/// The currently-open project (none until one is opened).
#[derive(Default)]
pub struct ProjectState {
    current: Arc<Mutex<Option<Project>>>,
}

impl ProjectState {
    /// The open project's root path, if any. Used by sibling command modules
    /// (e.g. the packager) that cook against the open project.
    pub fn current_root(&self) -> Option<PathBuf> {
        self.current
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|p| p.root.clone()))
    }

    /// The open project's `Content` directory. Pushed to the viewport's terrain
    /// streamer on project open **and** on viewport attach, since either can
    /// happen first (P16.4a).
    pub fn current_content_root(&self) -> Option<PathBuf> {
        self.current
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|p| p.content_root()))
    }

    /// The open project's levels root (`<project>/Content/Levels`), if any.
    ///
    /// One reader: `scene_save`'s no-path fallback, which used to write a level
    /// into app-data — outside the project, invisible to the asset database and
    /// to the cook. See `commands::scene::quicksave_path` and the IB-7 ruling on
    /// [`inf_project::Project::levels_root`].
    pub fn current_levels_root(&self) -> Option<PathBuf> {
        self.current
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|p| p.levels_root()))
    }
}

fn info(p: &Project) -> ProjectInfoDto {
    ProjectInfoDto {
        name: p.name().to_string(),
        root: p.root.to_string_lossy().replace('\\', "/"),
        content_dir: p.manifest.content_dir.clone(),
        levels_dir: p.manifest.levels_dir.clone(),
        template: p.manifest.template.clone(),
    }
}

/// Shared open flow: re-root assets, record recent, set current, announce.
fn apply_open(
    app: &AppHandle,
    state: &ProjectState,
    project: Project,
) -> Result<ProjectInfoDto, String> {
    let dto = info(&project);
    let content_root = project.content_root();

    // PUBLISH FIRST. `viewport_attach` reads `state.current` to catch up on a
    // project that opened before it existed; if the push below happened while
    // `current` was still `None`, an attach landing in that window would read
    // `None`, and the push it raced would already be gone — a terrain that never
    // streams until the next project switch. Setting `current` first makes the
    // two paths idempotent instead of ordered.
    if let Ok(cfg) = app.path().app_config_dir() {
        let _ = RecentProjects::push(&cfg, &project);
    }
    *state.current.lock().map_err(|e| e.to_string())? = Some(project);

    // Re-root the asset database to this project's content dir.
    if let Some(assets) = app.try_state::<AssetState>() {
        assets.reroot(app, content_root.clone());
    }
    // …and point the viewport's terrain streamer at the same directory, so a
    // `Terrain.asset` in the level resolves to a loose `.inf_terrain` and starts
    // paging (P16.4a). Switching projects re-points it, which drops every live
    // stream — the previous project's pages can never be served here.
    // **Broadcast** (P23.2a): the content root is a property of the open
    // project, so every viewport streams from it. A viewport left pointing at
    // the previous project would keep serving its pages.
    if let Some(viewport) = app.try_state::<super::ViewportState>() {
        viewport.set_content_root(super::Target::All, Some(content_root));
    }
    // …and drop every open Model Editor session (round-2 finding R2.F15). A
    // `MeshSession` is keyed on an `AssetId` **in the database re-rooted two
    // statements above**, so a document left open across a switch holds a whole
    // base mesh + op journal + checkpoints against a project that is gone. The
    // asset store and the viewports are re-pointed here; this one was not, and
    // the panel's own unmount effect never runs for a dock tab that stays open.
    if let Some(dcc) = app.try_state::<super::dcc::DccState>() {
        let dropped = dcc.clear_all();
        if dropped > 0 {
            tracing::info!(
                dropped,
                "closed Model Editor sessions belonging to the previous project"
            );
        }
    }
    let _ = app.emit("project://changed", dto.clone());
    tracing::info!("project opened: {}", dto.name);
    Ok(dto)
}

/// The available first-run templates.
#[tauri::command]
pub async fn project_templates() -> Result<Vec<ProjectTemplateDto>, String> {
    Ok(ProjectTemplate::all()
        .iter()
        .map(|t| ProjectTemplateDto {
            slug: t.slug().to_string(),
            label: t.label().to_string(),
            description: t.description().to_string(),
        })
        .collect())
}

/// The recent-projects list (pruned of roots that no longer exist).
#[tauri::command]
pub async fn project_recent(app: AppHandle) -> Result<Vec<RecentProjectDto>, String> {
    let cfg = app.path().app_config_dir().map_err(|e| e.to_string())?;
    // A list that exists but cannot be read is an error, not an empty list
    // (C4-38/F14): reporting it empty is how the Start Screen would have shown
    // "no recent projects" one moment before `push` wrote that emptiness back
    // over the real file.
    let mut list = RecentProjects::load_or_default(&cfg).map_err(|e| e.to_string())?;
    list.prune_missing();
    Ok(list
        .entries
        .into_iter()
        .map(|e| RecentProjectDto {
            name: e.name,
            path: e.path,
        })
        .collect())
}

/// The currently-open project, if any.
#[tauri::command]
pub async fn project_current(
    state: State<'_, ProjectState>,
) -> Result<Option<ProjectInfoDto>, String> {
    let cur = state.current.lock().map_err(|e| e.to_string())?;
    Ok(cur.as_ref().map(info))
}

/// **The level this project boots** (wave GTA1): the path of the `.inf_lvl` a
/// cooked build would start in, or `null` when the project has none.
///
/// # One rule, and it is the cook's
///
/// `inf_packager::cook` picks the root level by sorting the level assets it
/// found and taking the first (`levels.sort(); levels.first()`), so a build
/// starts in the LOWEST-GUID level. This reads the same asset database and
/// applies the same rule, which is the only way "the level the editor opens" and
/// "the level the build boots" can be the same sentence. Sorting by filename
/// instead would agree by luck on a one-level project and disagree the day a
/// second one arrives.
///
/// Opening it is the frontend's decision, not this command's: a command that
/// replaced the open document would throw away an author's unsaved work on a
/// menu click.
#[tauri::command]
pub async fn project_boot_level(assets: State<'_, AssetState>) -> Result<Option<String>, String> {
    assets.with_project(|proj| {
        let mut levels: Vec<_> = proj
            .db()
            .by_kind(inf_asset::AssetKind::Level)
            .map(|e| (e.id(), e.path.clone()))
            .collect();
        levels.sort_by_key(|(id, _)| *id);
        Ok(levels
            .first()
            .map(|(_, p)| p.to_string_lossy().replace('\\', "/")))
    })
}

/// Scaffold a new project under `parent` from `template` and open it.
#[tauri::command]
pub async fn project_new(
    app: AppHandle,
    parent: String,
    name: String,
    template: String,
    state: State<'_, ProjectState>,
) -> Result<ProjectInfoDto, String> {
    let tmpl = ProjectTemplate::from_slug(&template).unwrap_or(ProjectTemplate::Blank3d);
    let project =
        Project::create(&PathBuf::from(&parent), &name, tmpl).map_err(|e| e.to_string())?;
    apply_open(&app, &state, project)
}

/// Open an existing project at `root`.
#[tauri::command]
pub async fn project_open(
    app: AppHandle,
    root: String,
    state: State<'_, ProjectState>,
) -> Result<ProjectInfoDto, String> {
    let project = Project::open(PathBuf::from(&root)).map_err(|e| e.to_string())?;
    apply_open(&app, &state, project)
}

/// Close the current project (returns to the start screen). The frontend applies
/// the cleared state from the command result; no event is emitted (the
/// `project://changed` channel carries an opened project, never a close).
#[tauri::command]
pub async fn project_close(app: AppHandle, state: State<'_, ProjectState>) -> Result<(), String> {
    *state.current.lock().map_err(|e| e.to_string())? = None;
    // Stop streaming the closed project's terrain pages — in EVERY viewport
    // (P23.2a): one left streaming would outlive the project it belongs to.
    if let Some(viewport) = app.try_state::<super::ViewportState>() {
        viewport.set_content_root(super::Target::All, None);
    }
    Ok(())
}
