//! Project commands (P5.5): create / open / switch projects.
//!
//! The open project lives here behind an `Arc<Mutex<…>>`. Opening (or creating)
//! a project re-roots the asset database to `<project>/Content`
//! ([`AssetState::reroot`]), records it in the recent list, and emits
//! `project://changed` so the frontend leaves the start screen and re-syncs.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use inf_editor_core::editor_settings::EditorSettings;
use inf_editor_core::ipc::{ProjectBootDto, ProjectInfoDto, ProjectTemplateDto, RecentProjectDto};
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
        pin_boot_project(&cfg, &project.root);
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

/// **Remember this project as the one to open next launch** (wave CERT1).
///
/// Rung 2 of [`inf_project::boot::resolve`]. Written on every successful open,
/// so the plain meaning of the pin is *the last project you opened* — the
/// behaviour every other editor on this machine already has, and the reason the
/// showcase rung below it only fires for someone who has never opened anything.
///
/// **Best effort, deliberately.** A settings directory that cannot be written
/// must not turn opening a project into an error; the consequence is that the
/// next launch falls one rung further down, which is exactly the outcome an
/// author who has never had a pin already lives with. It is logged rather than
/// swallowed silently.
fn pin_boot_project(cfg: &std::path::Path, root: &std::path::Path) {
    let mut settings = match EditorSettings::load_or_default(cfg) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("boot project not pinned (settings unreadable): {e}");
            return;
        }
    };
    let pin = root.to_string_lossy().to_string();
    if settings.boot_project == pin {
        return;
    }
    settings.boot_project = pin;
    settings.normalize();
    if let Err(e) = settings.save(cfg) {
        tracing::warn!("boot project not pinned: {e}");
    }
}

/// **Open the project the application boots with, when it was launched with
/// none** (wave CERT1) — or `null` for the start screen.
///
/// # The rule
///
/// [`inf_project::boot::resolve`] and nothing else: `INF_BOOT_PROJECT`, then the
/// `boot_project` pin (the last project opened), then the showcase island
/// discovered beside the checkout, then nothing. Ring 2 supplies the three
/// inputs the rule cannot read for itself — the environment, the settings file
/// under `app_config_dir`, and the running executable's directory — and Ring 0
/// decides. That split is why the whole ordering is unit-tested against a temp
/// directory rather than against this machine.
///
/// # Why this is a command and not a `.setup()` hook
///
/// `apply_open` ends in `app.emit("project://changed")`, which is what makes the
/// Content Drawer re-sync and the boot level open. During `.setup()` there is no
/// webview to receive it. So the frontend asks for this once, on mount, when
/// `project_current` came back `null` — and a project opened this way takes
/// exactly the same path as one opened from the start screen.
///
/// **It refuses to act when a project is already open**, because the frontend
/// asking twice (a re-mount, a second window) must not re-root the asset
/// database under a session that is already running.
#[tauri::command]
pub async fn project_boot_default(
    app: AppHandle,
    state: State<'_, ProjectState>,
) -> Result<Option<ProjectBootDto>, String> {
    if state.current.lock().map_err(|e| e.to_string())?.is_some() {
        return Ok(None);
    }
    let env = std::env::var(inf_project::BOOT_PROJECT_ENV).ok();
    let pinned = app
        .path()
        .app_config_dir()
        .ok()
        .and_then(|cfg| EditorSettings::load_or_default(&cfg).ok())
        .map(|s| s.boot_project)
        .unwrap_or_default();
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(std::path::Path::to_path_buf));
    let Some(choice) =
        inf_project::resolve_boot_project(env.as_deref(), &pinned, exe_dir.as_deref())
    else {
        return Ok(None);
    };
    // A root that resolves and then fails to OPEN is not an error either: a
    // half-written `inf.toml` beside the checkout must not stop the application
    // starting. The start screen is the fallback, as it is for every other rung.
    let project = match Project::open(choice.root.clone()) {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!("boot project {} did not open: {e}", choice.root.display());
            return Ok(None);
        }
    };
    let project = apply_open(&app, &state, project)?;
    Ok(Some(ProjectBootDto {
        project,
        source: choice.source.phrase().to_string(),
    }))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The pin is written, and **nothing else in the file is disturbed**.
    ///
    /// The second half is the one worth an arm: `pin_boot_project` is called on
    /// every project open, so a bug that round-tripped the settings through a
    /// default would silently wipe an author's theme and keybindings the first
    /// time they opened a project.
    #[test]
    fn opening_a_project_pins_it_without_disturbing_the_settings_around_it() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join("config");
        std::fs::create_dir_all(&cfg).unwrap();

        let mut before = EditorSettings::default();
        before.theme_id = "a-theme-the-author-chose".to_string();
        before.autosave_interval_s = 17.0;
        before
            .keybindings
            .insert("Ctrl+K".to_string(), "some.command".to_string());
        before.save(&cfg).unwrap();

        let root = tmp.path().join("Vancouver Island");
        pin_boot_project(&cfg, &root);

        let after = EditorSettings::load_or_default(&cfg).unwrap();
        assert_eq!(after.boot_project, root.to_string_lossy());
        assert_eq!(after.theme_id, "a-theme-the-author-chose");
        assert_eq!(after.autosave_interval_s, 17.0);
        assert_eq!(
            after.keybindings.get("Ctrl+K").map(String::as_str),
            Some("some.command")
        );
    }

    /// A settings directory that cannot be written does not make opening a
    /// project fail — the pin is best-effort, and the consequence is one rung
    /// further down at the next launch.
    #[test]
    fn a_pin_that_cannot_be_written_is_not_an_error() {
        let missing = std::path::Path::new("this-directory-does-not-exist-cert1");
        pin_boot_project(missing, std::path::Path::new("anywhere"));
        // Reached: `pin_boot_project` returns `()` and must not panic.
    }
}
