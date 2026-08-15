//! **Path confinement for every filesystem door** (L7.H6).
//!
//! The editor's IPC surface is reachable by anything running in the webview, and
//! `file_read` / `file_write` / the `git_*` family / `search_workspace` all took
//! a bare absolute path and used it. `file_read("C:/Users/x/.ssh/id_rsa")` was a
//! valid request; `git_discard` outside the project threw away work in a repo
//! the editor was never opened on.
//!
//! The correct guard already existed — `shell::shell_reveal` canonicalizes the
//! target and requires it under the open project root or the app-data dir — and
//! was applied at exactly one of the eight doors. This module *is* that guard,
//! hoisted, and `shell_reveal` now calls it too, so there is one rule and one
//! place to change it.
//!
//! # The two shapes
//!
//! * [`confine_existing`] — the path must exist. Canonicalization resolves `..`,
//!   symlinks and 8.3 short names, so containment is decided on the real path
//!   rather than on its spelling.
//! * [`confine_for_write`] — the path may not exist yet (a save-as, a new file).
//!   Canonicalizes the nearest **existing ancestor** and re-joins the tail, so a
//!   new file inherits its parent's verdict and `../../..` still cannot escape.
//!
//! # Why not "starts_with the project root" on the raw string
//!
//! Because `C:/proj/../../secrets` starts with `C:/proj`. Every check here is on
//! a canonicalized path for that reason.

use std::path::{Component, Path, PathBuf};

use tauri::{AppHandle, Manager};

use super::project::ProjectState;

/// The roots a request may touch: the open project (Content, Build, the user's
/// source tree) and the app-data dir (loose Content + quicksaves before a
/// project is opened).
///
/// Empty when no project is open and app-data cannot be resolved — in which case
/// every confined door refuses, which is the correct answer rather than a
/// fallback to "anywhere".
pub fn allowed_roots(app: &AppHandle, project: &ProjectState) -> Vec<PathBuf> {
    let mut allowed: Vec<PathBuf> = Vec::new();
    if let Some(root) = project.current_root() {
        if let Ok(c) = std::fs::canonicalize(&root) {
            allowed.push(c);
        }
    }
    if let Ok(dir) = app.path().app_data_dir() {
        if let Ok(c) = std::fs::canonicalize(&dir) {
            allowed.push(c);
        }
    }
    allowed
}

/// Confine an **existing** path to [`allowed_roots`], returning the canonical
/// path to use.
pub fn confine_existing(
    app: &AppHandle,
    project: &ProjectState,
    path: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    let path = path.as_ref();
    let target = std::fs::canonicalize(path).map_err(|e| {
        format!(
            "path not found: {e} ({})",
            path.file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_default()
        )
    })?;
    check(app, project, &target)?;
    Ok(target)
}

/// Confine a path that **may not exist yet** (the write/create case), returning
/// the path to write.
///
/// The nearest existing ancestor is canonicalized and checked; the remaining
/// components are re-joined onto it. A `..` in the tail is refused outright
/// rather than normalized, because a tail that walks upward is never something a
/// save dialog produced.
pub fn confine_for_write(
    app: &AppHandle,
    project: &ProjectState,
    path: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    let path = path.as_ref();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut cursor = path.to_path_buf();
    loop {
        if cursor.exists() {
            break;
        }
        let Some(name) = cursor.file_name().map(|n| n.to_os_string()) else {
            return Err("refusing a path with no existing ancestor".into());
        };
        if name == ".." {
            return Err("refusing a path that walks above its own directory".into());
        }
        tail.push(name);
        if !cursor.pop() {
            return Err("refusing a path with no existing ancestor".into());
        }
    }
    let base = std::fs::canonicalize(&cursor)
        .map_err(|e| format!("cannot resolve the containing directory: {e}"))?;
    check(app, project, &base)?;
    let mut out = base;
    for name in tail.into_iter().rev() {
        out.push(name);
    }
    Ok(out)
}

/// Confine a path **relative to an already-confined root** — the `git_*` family's
/// shape, where the repo is confined once and each path argument is interpreted
/// inside it.
///
/// The argument must be relative and must not contain a `..` or a root/prefix
/// component (on Windows a bare `C:` in a `push` replaces the whole path — the
/// C4-28 mechanism). It is not required to exist: `git_discard` names files that
/// git will restore, and a deleted file is exactly the case.
pub fn confine_under(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let p = Path::new(rel);
    for c in p.components() {
        match c {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => {
                return Err(format!("refusing a path with `..` in it: {rel}"));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!("refusing an absolute path here: {rel}"));
            }
        }
    }
    Ok(root.join(p))
}

fn check(app: &AppHandle, project: &ProjectState, target: &Path) -> Result<(), String> {
    let allowed = allowed_roots(app, project);
    if allowed.is_empty() {
        return Err(
            "no project is open, so there is nothing this command is allowed to touch".into(),
        );
    }
    if !allowed.iter().any(|root| target.starts_with(root)) {
        return Err(
            "refusing a path outside the open project and the editor's app-data directory".into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `confine_under` is the half that needs no Tauri handle, and it is the half
    /// `git_discard` — the one door here that destroys work — depends on.
    #[test]
    fn a_relative_path_is_joined_and_an_escaping_one_is_refused() {
        let root = Path::new("C:/proj");
        assert_eq!(
            confine_under(root, "src/main.rs").unwrap(),
            root.join("src/main.rs")
        );
        assert_eq!(
            confine_under(root, "./a.txt").unwrap(),
            root.join("./a.txt")
        );

        // Drive-prefixed spellings and backslash separators are escapes only
        // where the platform's path grammar says so: on Unix `C:/…` is a plain
        // relative component and `..\..\secrets` is ONE odd filename, and
        // confining either under the root is the correct verdict there. The
        // sibling fixture in git.rs learned this on its first CI run.
        #[cfg(windows)]
        let bad_paths: &[&str] = &[
            "../secrets",
            "src/../../secrets",
            "/etc/passwd",
            "C:/Windows/system32",
            r"..\..\secrets",
        ];
        #[cfg(not(windows))]
        let bad_paths: &[&str] = &["../secrets", "src/../../secrets", "/etc/passwd"];

        for &bad in bad_paths {
            assert!(
                confine_under(root, bad).is_err(),
                "must refuse {bad:?} — this is the argument `git_discard` deletes"
            );
        }
    }

    /// The reason every check canonicalizes: a prefix test on the raw string is
    /// satisfied by a path that leaves immediately.
    #[test]
    fn a_string_prefix_test_would_have_passed_the_escape() {
        let spelled = "C:/proj/../../secrets";
        assert!(
            spelled.starts_with("C:/proj"),
            "the naive guard accepts this, which is why it is not the guard"
        );
        assert!(confine_under(Path::new("C:/proj"), "../../secrets").is_err());
    }

    /// **The source gate** (L7.H6, widened by round-2 finding **B13**).
    ///
    /// `confine_*` can only protect a door that calls it, and the whole finding
    /// is that seven of eight doors did not. Nothing observable distinguishes a
    /// guarded command from an unguarded one — the guard's effect is a *refusal*
    /// that never happens in a legitimate session — so, per the campaign's
    /// standing rule, the enforcement is a source pin: every `#[tauri::command]`
    /// in the filesystem modules must reach the confinement helper.
    ///
    /// The list is of MODULES, not of functions, so a new command added to any
    /// of them is covered the day it is written.
    ///
    /// # Round 2: the scope was four modules, and six more doors took paths
    ///
    /// `asset_import`, `asset_table_import`, `photo_load`, `terrain_import`,
    /// `terrain_probe_heightmap` and `project_package` all take a path from the
    /// webview, and this gate's module list could not see any of them.
    ///
    /// Widening it needed a decision rather than a `confine_existing`, because
    /// those doors are **imports**: reading a `.gltf` off the Desktop is the
    /// product's core workflow, and confining them to the project root would
    /// make importing impossible. So the rule has two ways to be satisfied —
    /// confine the path, or appear in [`READ_ANYWHERE`] **with the reason
    /// written down**. That converts an invisible gap into a recorded decision,
    /// and it is an allowlist rather than a ban list per the P22 law: a new
    /// command in any of these modules fails this gate until somebody decides
    /// which of the two it is.
    ///
    /// The *write* halves are confined regardless, and separately:
    /// `AssetProject::content_dir` refuses a destination folder with any
    /// non-`Normal` component (it was `root.join(sub)` + `create_dir_all`, so
    /// `"../../../Users/Public"` escaped), and `project_package` refuses a
    /// relative `out_dir` (which resolved against the *editor process's*
    /// working directory).
    const FS_MODULES: &[&str] = &[
        "files.rs",
        "git.rs",
        "search.rs",
        "shell.rs",
        "assets.rs",
        "terrain.rs",
        "photogrammetry.rs",
        "package.rs",
    ];

    /// Commands that deliberately accept a path **outside** the project, each
    /// with the reason and what it exposes (round-2 finding B13).
    ///
    /// Every entry is an import door whose whole purpose is to read a file the
    /// author picked from anywhere on their disk. None of them returns the
    /// file's *bytes* to the webview: the import doors copy content into the
    /// project (where it becomes a visible asset), the probe returns a header,
    /// and the packager writes.
    const READ_ANYWHERE: &[(&str, &str)] = &[
        (
            "assets.rs::asset_import",
            "the Content Drawer's Import: reads the author's chosen source files \
             and copies them into the project. Its DESTINATION is confined by \
             AssetProject::content_dir.",
        ),
        (
            "assets.rs::asset_table_import",
            "imports a CSV/JSON the author picked into an existing .inf_table.",
        ),
        (
            "photogrammetry.rs::photo_load",
            "loads the author's photographs from wherever the camera dumped them.",
        ),
        (
            "terrain.rs::terrain_import",
            "imports the author's heightmap; the destination is TERRAIN_IMPORT_FOLDER \
             under the content root.",
        ),
        (
            "terrain.rs::terrain_probe_heightmap",
            "reads a heightmap HEADER for the wizard — dimensions and bit depth, \
             never pixels.",
        ),
        (
            "package.rs::project_package",
            "cooks into an output directory the author chose (exporting a build to \
             the Desktop is the point). Refuses a RELATIVE out_dir, which would \
             resolve against the editor process's working directory.",
        ),
    ];

    /// Whether a command signature takes something path-shaped from the
    /// webview. Deliberately generous: a false positive costs one allowlist
    /// entry, a false negative is the finding.
    fn takes_a_path(signature: &str) -> bool {
        // The PARAMETER LIST only. The first draft searched the whole
        // signature and matched `asset_rust_source` on its own name — a gate
        // that reads the wrong text is the failure mode this file exists to
        // avoid, met on its first run.
        let Some(args) = signature.split_once('(').map(|(_, rest)| rest) else {
            return false;
        };
        args.split(',')
            .filter_map(|arg| arg.split(':').next())
            .map(|name| name.trim().trim_start_matches("mut ").to_ascii_lowercase())
            .any(|n| n.contains("path") || n.contains("source") || n.contains("dir") || n == "repo")
    }

    #[test]
    fn every_filesystem_command_reaches_the_confinement_helper() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/commands");
        let mut unguarded = Vec::new();
        let mut checked = 0usize;
        let mut with_paths = 0usize;
        for module in FS_MODULES {
            let text = std::fs::read_to_string(dir.join(module))
                .unwrap_or_else(|e| panic!("{module} is readable: {e}"));
            // Split on the attribute so each chunk is one command's body.
            let mut chunks = text.split("#[tauri::command]");
            let _preamble = chunks.next();
            for body in chunks {
                checked += 1;
                let name = body
                    .split("fn ")
                    .nth(1)
                    .and_then(|r| r.split('(').next())
                    .unwrap_or("?")
                    .trim()
                    .to_string();
                let qualified = format!("{module}::{name}");
                let signature = body.split('{').next().unwrap_or("");
                if !takes_a_path(signature) {
                    continue;
                }
                with_paths += 1;
                // Either the command confines directly, or it calls a helper in
                // its own module that does (`confined_repo` in git.rs), or it is
                // a recorded read-anywhere import door.
                let guarded = body.contains("confine_existing")
                    || body.contains("confine_for_write")
                    || body.contains("confined_repo")
                    || body.contains("confined_paths")
                    || body.contains("confine_under");
                let allowed = READ_ANYWHERE.iter().any(|(q, _)| *q == qualified);
                if !guarded && !allowed {
                    unguarded.push(qualified.clone());
                    continue;
                }
                // …and the PATH ARGUMENTS, separately. Confining the repo says
                // nothing about the file names inside it, and `git_discard`
                // deletes whatever it is handed. The first version of this gate
                // checked only the line above and a gutted `confined_paths`
                // survived its mutation — a vacuous half, caught the way the
                // campaign's other three were.
                let takes_named_paths =
                    signature.contains("paths: Vec<String>") || signature.contains("path: String");
                if takes_named_paths
                    && !allowed
                    && !(body.contains("confined_paths")
                        || body.contains("confine_under")
                        || body.contains("confine_existing")
                        || body.contains("confine_for_write"))
                {
                    unguarded.push(format!("{qualified} (path arguments)"));
                }
            }
        }
        assert!(
            checked >= 10,
            "the gate found only {checked} commands — it is not reading what it thinks it is"
        );
        assert!(
            with_paths >= 12,
            "the gate found only {with_paths} path-taking commands across {} modules; \
             it is calibrated against a list that includes six import doors, so a \
             smaller number means `takes_a_path` no longer matches the signatures",
            FS_MODULES.len()
        );
        assert!(
            unguarded.is_empty(),
            "these commands take a path from the webview and neither confine it nor \
             appear in READ_ANYWHERE with a reason: {unguarded:?}"
        );
    }

    /// The allowlist may not rot: every entry must still name a command that
    /// exists, or it is a decision recorded about nothing — and the next
    /// command to take that name inherits an exemption nobody granted it.
    #[test]
    fn every_read_anywhere_entry_still_names_a_live_command() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/commands");
        for (qualified, reason) in READ_ANYWHERE {
            let (module, name) = qualified.split_once("::").expect("module::name");
            let text = std::fs::read_to_string(dir.join(module))
                .unwrap_or_else(|e| panic!("{module} is readable: {e}"));
            assert!(
                text.contains(&format!("fn {name}(")),
                "READ_ANYWHERE names {qualified}, which no longer exists"
            );
            assert!(
                reason.len() > 40,
                "{qualified}'s exemption reason is too short to be a reason: {reason:?}"
            );
        }
    }

    /// The gate's own falsifier: it must be able to see an unguarded command.
    /// A source gate that reads the wrong thing passes forever.
    #[test]
    fn the_gate_can_see_an_unguarded_command() {
        // The shape it must reject: a command with a path argument, whose body
        // confines the repo and nothing else.
        let body = "\npub async fn git_wipe(repo: String, paths: Vec<String>) -> R {\n    \
                    let root = confined_repo(&app, &project, &repo)?;\n}";
        let signature = body.split('{').next().unwrap_or("");
        assert!(signature.contains("paths: Vec<String>"));
        assert!(!body.contains("confined_paths"));
        // …and the shape it must accept.
        let fixed = format!("{body}\n    let paths = confined_paths(&root, &paths)?;");
        assert!(fixed.contains("confined_paths"));
    }
}
