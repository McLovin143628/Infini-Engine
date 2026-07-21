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
    // Re-root the asset database to this project's content dir.
    if let Some(assets) = app.try_state::<AssetState>() {
        assets.reroot(app, project.content_root());
    }
    // Record in the recent list.
    if let Ok(cfg) = app.path().app_config_dir() {
        let _ = RecentProjects::push(&cfg, &project);
    }
    let dto = info(&project);
    *state.current.lock().map_err(|e| e.to_string())? = Some(project);
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
    let mut list = RecentProjects::load(&cfg);
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
pub async fn project_close(state: State<'_, ProjectState>) -> Result<(), String> {
    *state.current.lock().map_err(|e| e.to_string())? = None;
    Ok(())
}
