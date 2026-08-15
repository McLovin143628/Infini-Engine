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

    /// **The source gate** (L7.H6).
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
    #[test]
    fn every_filesystem_command_reaches_the_confinement_helper() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/commands");
        let mut unguarded = Vec::new();
        let mut checked = 0usize;
        for module in ["files.rs", "git.rs", "search.rs", "shell.rs"] {
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
                // Either the command confines directly, or it calls a helper in
                // its own module that does (`confined_repo` in git.rs).
                let guarded = body.contains("confine_existing")
                    || body.contains("confine_for_write")
                    || body.contains("confined_repo");
                if !guarded {
                    unguarded.push(format!("{module}::{name}"));
                    continue;
                }
                // …and the PATH ARGUMENTS, separately. Confining the repo says
                // nothing about the file names inside it, and `git_discard`
                // deletes whatever it is handed. The first version of this gate
                // checked only the line above and a gutted `confined_paths`
                // survived its mutation — a vacuous half, caught the way the
                // campaign's other three were.
                let signature = body.split('{').next().unwrap_or("");
                let takes_paths =
                    signature.contains("paths: Vec<String>") || signature.contains("path: String");
                if takes_paths
                    && !(body.contains("confined_paths")
                        || body.contains("confine_under")
                        || body.contains("confine_existing")
                        || body.contains("confine_for_write"))
                {
                    unguarded.push(format!("{module}::{name} (path arguments)"));
                }
            }
        }
        assert!(
            checked >= 10,
            "the gate found only {checked} commands — it is not reading what it thinks it is"
        );
        assert!(
            unguarded.is_empty(),
            "these filesystem commands take a path from the webview and never confine it: {unguarded:?}"
        );
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
