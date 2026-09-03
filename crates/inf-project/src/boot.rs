//! **Which project the application opens when it is launched with none** (wave
//! CERT1).
//!
//! # What this closes
//!
//! Before this module the editor had no answer at all. `ProjectState::current`
//! started `None` on every cold launch, nothing persisted which project had been
//! open, and the only project knowledge that survived a restart was the recent
//! list — which the start screen renders as *buttons a human presses*. The
//! showcase this engine exists to show off therefore took, at best, one launch
//! and one click, and on a fresh profile it took a file dialog and a path the
//! author had to know.
//!
//! The certification's ruling is that the island **is** the default document:
//! *"this starter level is meant to be the starter/default level for the
//! application to really show off what it can do"*. So the question "which
//! project" needs a rule, and the rule needs to be one a reader can predict.
//!
//! # The rungs, in order
//!
//! 1. **[`BOOT_PROJECT_ENV`]** — an absolute project root in the environment.
//!    The `INF_PLAYER_BIN` precedent: a machine-level override that a harness, a
//!    packaging script or a developer can set without touching a settings file.
//! 2. **The pin** — `EditorSettings::boot_project`, which the editor writes on
//!    every successful project open. So *the last project you opened* is the one
//!    that comes back, which is what every editor on this machine already does,
//!    and an author who pins one deliberately outranks the showcase for ever.
//! 3. **The showcase** — [`SHOWCASE_RELATIVE`], found by walking up from a start
//!    directory (the running executable's). This is the rung that makes the
//!    island the default *on a machine where it has been built*, with no
//!    settings file and no click.
//! 4. **Nothing.** The start screen, exactly as before.
//!
//! # Why the showcase is DISCOVERED and never hard-coded
//!
//! The island's heavy halves are gigabytes and are deliberately outside the
//! repository — `<checkout>/../island-build/project`, gitignored, built by
//! `inf island build`. An absolute path compiled into the engine would be a
//! statement about one person's disk; a path relative to the *executable* is a
//! statement about the layout `inf island build` itself produces. On a machine
//! where the island has never been built no rung resolves, [`resolve`] answers
//! `None`, and the application opens the start screen the way it always has.
//!
//! # Purity
//!
//! Everything here is a pure function of its arguments plus `Path::is_file`.
//! There is no environment read, no settings read and no directory walk that
//! begins anywhere but an argument — so the whole rule is testable against a
//! temp directory, and the Ring-2 command is the only thing that knows where an
//! `AppHandle` keeps its config.

use std::path::{Path, PathBuf};

use crate::manifest::PROJECT_FILE;

/// The environment variable that outranks every other rung — an absolute path
/// to a project root (the directory holding `inf.toml`).
///
/// Named for the `INF_PLAYER_BIN` precedent in `inf_editor_core::pie`: one
/// variable, one path, no parsing.
pub const BOOT_PROJECT_ENV: &str = "INF_BOOT_PROJECT";

/// Where `inf island build` puts the showcase, relative to the directory that
/// holds the engine checkout: `island-build/project`.
///
/// This is the recipe's own `[source] cache` convention seen from one level up —
/// `samples/island/island.toml` writes its project to `../../../island-build/
/// project` relative to itself, which is `<checkout>/../island-build/project`.
pub const SHOWCASE_RELATIVE: [&str; 2] = ["island-build", "project"];

/// How many ancestors of the start directory [`find_showcase`] examines.
///
/// A dev build's executable sits at `<checkout>/target/debug/inf-studio.exe`, so
/// the showcase's holder is **four** ancestors up (`debug`, `target`,
/// `<checkout>`, then its parent). Eight leaves room for a `target/<triple>/
/// release/` layout and for a bundle nested a level or two deeper, and it is
/// bounded so the walk cannot climb to the volume root on a machine where the
/// showcase does not exist.
pub const SHOWCASE_SEARCH_DEPTH: usize = 8;

/// Which rung of [`resolve`] answered.
///
/// Carried out of the resolution rather than inferred from the path, because
/// the status line the editor prints says *why* a project opened and a reader
/// who cannot tell "you pinned this" from "this is the showcase" cannot act on
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootSource {
    /// [`BOOT_PROJECT_ENV`] named it.
    Environment,
    /// The editor's own `boot_project` pin — the last project opened, or one
    /// set deliberately.
    Pinned,
    /// Discovered beside the checkout by [`find_showcase`].
    Showcase,
}

impl BootSource {
    /// A short human phrase for a status line.
    pub fn phrase(self) -> &'static str {
        match self {
            BootSource::Environment => "from INF_BOOT_PROJECT",
            BootSource::Pinned => "the last project you opened",
            BootSource::Showcase => "the showcase island",
        }
    }
}

/// A resolved boot project: where it is, and which rung named it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootProject {
    /// The project root — the directory holding `inf.toml`.
    pub root: PathBuf,
    /// The rung that answered.
    pub source: BootSource,
}

/// Does `root` hold an `inf.toml`?
///
/// The same test [`crate::Project::find`] walks up looking for, factored out so
/// every rung of the resolution asks the identical question. A rung that
/// answered a path with no manifest would hand the editor a root
/// `Project::open` is about to refuse.
pub fn is_project_root(root: &Path) -> bool {
    root.join(PROJECT_FILE).is_file()
}

/// Walk up from `start`, at most [`SHOWCASE_SEARCH_DEPTH`] ancestors, for a
/// directory holding [`SHOWCASE_RELATIVE`] with an `inf.toml` in it.
///
/// `start` itself is the first candidate's holder, so an executable that sits
/// *beside* `island-build/` is found without climbing.
pub fn find_showcase(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    for _ in 0..=SHOWCASE_SEARCH_DEPTH {
        let d = dir?;
        let mut candidate = d.to_path_buf();
        for seg in SHOWCASE_RELATIVE {
            candidate.push(seg);
        }
        if is_project_root(&candidate) {
            return Some(candidate);
        }
        dir = d.parent();
    }
    None
}

/// Resolve the project the application should open, or `None` for the start
/// screen.
///
/// * `env` — the value of [`BOOT_PROJECT_ENV`], if it is set.
/// * `pinned` — `EditorSettings::boot_project`; **empty means unset**, which is
///   how a `String` carries "no pin" through TOML without a `None` the format
///   cannot write and without a `skip_serializing_if` this repository has been
///   bitten by three times.
/// * `start` — where to begin the showcase walk (the running executable's
///   directory). `None` skips that rung entirely, which is what a caller with no
///   executable path — a test, or a host that would rather not guess — wants.
///
/// A rung whose path does not hold an `inf.toml` is **skipped, not fatal**: a
/// pin left behind by a project the author has since deleted must not stop the
/// application booting, and it must not stop the showcase below it answering.
pub fn resolve(env: Option<&str>, pinned: &str, start: Option<&Path>) -> Option<BootProject> {
    for (candidate, source) in [
        (env.unwrap_or("").trim(), BootSource::Environment),
        (pinned.trim(), BootSource::Pinned),
    ] {
        if candidate.is_empty() {
            continue;
        }
        let root = PathBuf::from(candidate);
        if is_project_root(&root) {
            return Some(BootProject { root, source });
        }
    }
    let root = find_showcase(start?)?;
    Some(BootProject {
        root,
        source: BootSource::Showcase,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A project root with a real (minimal) manifest in it.
    fn project_at(dir: &Path) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        crate::ProjectManifest::new("Fixture", "blank-3d")
            .save(dir)
            .unwrap();
        dir.to_path_buf()
    }

    #[test]
    fn a_directory_without_a_manifest_is_not_a_project_root() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(!is_project_root(tmp.path()));
        project_at(tmp.path());
        assert!(is_project_root(tmp.path()));
    }

    #[test]
    fn the_showcase_is_found_by_walking_up_from_a_dev_executable() {
        let tmp = tempfile::tempdir().unwrap();
        // The dev layout the walk exists for: <holder>/<checkout>/target/debug.
        let holder = tmp.path();
        let exe_dir = holder.join("infinity_engine").join("target").join("debug");
        std::fs::create_dir_all(&exe_dir).unwrap();
        let showcase = project_at(&holder.join("island-build").join("project"));

        assert_eq!(find_showcase(&exe_dir), Some(showcase));
    }

    #[test]
    fn a_showcase_that_was_never_built_is_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let exe_dir = tmp
            .path()
            .join("infinity_engine")
            .join("target")
            .join("debug");
        std::fs::create_dir_all(&exe_dir).unwrap();
        // The directory exists but nothing built a project into it.
        std::fs::create_dir_all(tmp.path().join("island-build").join("project")).unwrap();

        assert_eq!(find_showcase(&exe_dir), None);
        assert_eq!(resolve(None, "", Some(&exe_dir)), None);
    }

    #[test]
    fn the_walk_is_bounded_exactly_where_it_says() {
        let tmp = tempfile::tempdir().unwrap();
        project_at(&tmp.path().join("island-build").join("project"));

        // The holder is the Nth ancestor of a start N levels below it, and the
        // walk examines the start plus SHOWCASE_SEARCH_DEPTH ancestors — so N
        // levels down reaches and N + 1 does not. Both halves are asserted,
        // because a bound only one side of which is checked is a bound that can
        // be off by one in the direction nobody looked.
        let mut at_bound = tmp.path().to_path_buf();
        for i in 0..SHOWCASE_SEARCH_DEPTH {
            at_bound.push(format!("d{i}"));
        }
        let past_bound = at_bound.join("one-too-far");
        std::fs::create_dir_all(&past_bound).unwrap();

        assert!(
            find_showcase(&at_bound).is_some(),
            "the walk stopped short of its own bound"
        );
        assert_eq!(
            find_showcase(&past_bound),
            None,
            "the walk climbed past its bound"
        );
    }

    #[test]
    fn the_environment_outranks_the_pin_and_the_pin_outranks_the_showcase() {
        let tmp = tempfile::tempdir().unwrap();
        let from_env = project_at(&tmp.path().join("env"));
        let pinned = project_at(&tmp.path().join("pinned"));
        let exe_dir = tmp
            .path()
            .join("infinity_engine")
            .join("target")
            .join("debug");
        std::fs::create_dir_all(&exe_dir).unwrap();
        let showcase = project_at(&tmp.path().join("island-build").join("project"));

        let env = from_env.to_string_lossy().to_string();
        let pin = pinned.to_string_lossy().to_string();

        assert_eq!(
            resolve(Some(&env), &pin, Some(&exe_dir)),
            Some(BootProject {
                root: from_env,
                source: BootSource::Environment
            })
        );
        assert_eq!(
            resolve(None, &pin, Some(&exe_dir)),
            Some(BootProject {
                root: pinned,
                source: BootSource::Pinned
            })
        );
        assert_eq!(
            resolve(None, "", Some(&exe_dir)),
            Some(BootProject {
                root: showcase,
                source: BootSource::Showcase
            })
        );
        assert_eq!(resolve(None, "", None), None, "no start, no showcase rung");
    }

    #[test]
    fn a_rung_pointing_at_a_deleted_project_is_skipped_not_fatal() {
        let tmp = tempfile::tempdir().unwrap();
        let exe_dir = tmp
            .path()
            .join("infinity_engine")
            .join("target")
            .join("debug");
        std::fs::create_dir_all(&exe_dir).unwrap();
        let showcase = project_at(&tmp.path().join("island-build").join("project"));
        let gone = tmp.path().join("deleted").to_string_lossy().to_string();

        // Both upper rungs name nothing that exists; the showcase still answers.
        assert_eq!(
            resolve(Some(&gone), &gone, Some(&exe_dir)),
            Some(BootProject {
                root: showcase,
                source: BootSource::Showcase
            })
        );
    }

    #[test]
    fn whitespace_is_not_a_pin() {
        let tmp = tempfile::tempdir().unwrap();
        let exe_dir = tmp.path().join("a").join("b").join("c");
        std::fs::create_dir_all(&exe_dir).unwrap();
        assert_eq!(resolve(Some("   "), "\t", Some(&exe_dir)), None);
    }
}
