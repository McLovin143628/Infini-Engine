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

/// **Remember this project as the one to open next launch** (wave CERT1) —
/// the AUTOMATIC pin, rung 4.
///
/// Written on every successful open, so its plain meaning is *the last project
/// you opened*. Since the CERT1 audit's ruling it sits **below** the showcase:
/// visiting a project is not the same act as choosing one, and the rung exists
/// for a machine where `inf island build` has never run.
///
/// **A DELIBERATE pin is left alone.** An author who used Preferences ▸ "Make
/// this project the default" must be able to open a scratch project without
/// silently losing that choice — one string carries both pins, so the only way
/// to keep the decision is not to overwrite it. Nothing is lost by skipping the
/// write: rung 2 answers first, so rung 4's value is never read while the flag
/// is set.
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
    if settings.boot_project_deliberate {
        return;
    }
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

/// **Write the boot pin and say what it is** — the one door Preferences' two
/// actions share (CERT1 audit ruling).
///
/// `root` empty clears the pin entirely, which is "reset to the showcase"; a
/// path sets it and marks it DELIBERATE, which is "make this project the
/// default". The answer is the phrase for the rung that will win at the *next*
/// launch, resolved through [`inf_project::boot::resolve`] itself rather than
/// assumed — because "reset to the showcase" only reaches the showcase on a
/// machine that has one, and a dialog that said otherwise would be lying on
/// every other machine.
fn set_boot_pin(app: &AppHandle, root: &str) -> Result<String, String> {
    let cfg = app.path().app_config_dir().map_err(|e| e.to_string())?;
    let mut settings = EditorSettings::load_or_default(&cfg).map_err(|e| e.to_string())?;
    settings.boot_project = root.to_string();
    settings.boot_project_deliberate = !root.is_empty();
    settings.normalize();
    settings.save(&cfg).map_err(|e| e.to_string())?;
    Ok(boot_phrase(&settings))
}

/// What `inf_project::boot::resolve` would answer next launch, as a phrase.
///
/// Ring 2 supplies the environment and the executable's directory; Ring 0
/// decides. Split out so the two commands and their arms read the same rule the
/// launch does, rather than a second description of it.
fn boot_phrase(settings: &EditorSettings) -> String {
    let env = std::env::var(inf_project::BOOT_PROJECT_ENV).ok();
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(std::path::Path::to_path_buf));
    match inf_project::resolve_boot_project(
        env.as_deref(),
        &settings.boot_project,
        settings.boot_project_deliberate,
        exe_dir.as_deref(),
    ) {
        Some(b) => b.source.phrase().to_string(),
        None => NOTHING_BOOTS.to_string(),
    }
}

/// The phrase for "no rung answers" — the start screen.
///
/// A constant rather than a literal in two places, because it is the string the
/// Preferences row prints and the arm below asserts.
pub const NOTHING_BOOTS: &str = "nothing — the start screen";

/// **Make the open project the one the application boots on** (CERT1 audit
/// ruling) — the DELIBERATE pin, rung 2, above the showcase.
///
/// Answers the phrase for what will boot next launch, so the caller can print a
/// sentence without a second copy of the rule.
#[tauri::command]
pub async fn project_set_default(
    app: AppHandle,
    state: State<'_, ProjectState>,
) -> Result<String, String> {
    let root = {
        let cur = state.current.lock().map_err(|e| e.to_string())?;
        match cur.as_ref() {
            Some(p) => p.root.to_string_lossy().to_string(),
            None => return Err("no project is open".to_string()),
        }
    };
    set_boot_pin(&app, &root)
}

/// **Forget the deliberate default**, so the showcase (or, failing that, the
/// last project opened) answers again. The other half of the ruling: a choice a
/// reader cannot undo is not a preference, it is a trap.
#[tauri::command]
pub async fn project_clear_default(app: AppHandle) -> Result<String, String> {
    set_boot_pin(&app, "")
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
    let settings = app
        .path()
        .app_config_dir()
        .ok()
        .and_then(|cfg| EditorSettings::load_or_default(&cfg).ok())
        .unwrap_or_default();
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(std::path::Path::to_path_buf));
    let Some(choice) = inf_project::resolve_boot_project(
        env.as_deref(),
        &settings.boot_project,
        settings.boot_project_deliberate,
        exe_dir.as_deref(),
    ) else {
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

        let mut before = EditorSettings {
            theme_id: "a-theme-the-author-chose".to_string(),
            autosave_interval_s: 17.0,
            ..EditorSettings::default()
        };
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
    ///
    /// # The first version of this arm was worse than useless
    ///
    /// It passed a bare relative name, `this-directory-does-not-exist-cert1`,
    /// on the assumption that a missing directory cannot be written to.
    /// `EditorSettings::save` **creates** it — so the arm exercised the SUCCESS
    /// path while claiming to exercise the failure one, and it created that
    /// directory next to the running test's working directory, which is
    /// `editor/studio/src-tauri`. It was committed to the repository before the
    /// CRLF diff sweep noticed a file nobody had written.
    ///
    /// A path whose PARENT IS A FILE cannot be created on any platform, which is
    /// the shape a "cannot be written" arm actually needs. And the arm now
    /// asserts the outcome rather than merely surviving: nothing is created, and
    /// the settings that could not be reached are still absent afterwards.
    /// **(d) A DELIBERATE pin survives an open, and reset clears it** (CERT1
    /// audit ruling).
    ///
    /// The two halves are one arm because they are one property: the pin is a
    /// DECISION, so a visit must not overwrite it and an explicit reset must.
    /// Split, either half passes on a `set_boot_pin` that writes nothing.
    #[test]
    fn an_open_does_not_overwrite_a_deliberate_pin_and_a_reset_does_clear_it() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join("config");
        std::fs::create_dir_all(&cfg).unwrap();

        // An author made the island the default, from Preferences.
        let island = tmp.path().join("island-build").join("project");
        let mut chosen = EditorSettings::default();
        chosen.boot_project = island.to_string_lossy().to_string();
        chosen.boot_project_deliberate = true;
        chosen.save(&cfg).unwrap();

        // …then opens a scratch project. `apply_open` pins on every open.
        pin_boot_project(&cfg, &tmp.path().join("scratch"));

        let after = EditorSettings::load_or_default(&cfg).unwrap();
        assert_eq!(
            after.boot_project,
            island.to_string_lossy(),
            "opening another project overwrote a DELIBERATE pin — the choice an \
             author made in Preferences must survive a visit, or the flag is \
             decoration"
        );
        assert!(after.boot_project_deliberate, "the pin lost its intent");

        // The reset half, through the same door the command uses.
        let mut reset = after;
        reset.boot_project = String::new();
        reset.boot_project_deliberate = false;
        reset.save(&cfg).unwrap();
        let cleared = EditorSettings::load_or_default(&cfg).unwrap();
        assert!(cleared.boot_project.is_empty());
        assert!(!cleared.boot_project_deliberate);

        // …and NOW an open pins again, because there is no decision to protect.
        pin_boot_project(&cfg, &tmp.path().join("scratch"));
        let visited = EditorSettings::load_or_default(&cfg).unwrap();
        assert_eq!(
            visited.boot_project,
            tmp.path().join("scratch").to_string_lossy()
        );
        assert!(
            !visited.boot_project_deliberate,
            "an open marked its own pin deliberate — a visit is not a decision"
        );
    }

    #[test]
    fn a_pin_that_cannot_be_written_is_not_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let blocker = tmp.path().join("a-file-not-a-directory");
        std::fs::write(&blocker, b"x").unwrap();
        let unwritable = blocker.join("config");

        pin_boot_project(&unwritable, std::path::Path::new("anywhere"));

        assert!(
            !unwritable.exists(),
            "the pin created a directory under a FILE, so this arm is not testing \
             an unwritable path"
        );
        assert!(
            blocker.is_file(),
            "the blocker stopped being a file, so the path above became writable"
        );
    }
}
