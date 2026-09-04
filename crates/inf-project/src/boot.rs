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
//! 2. **A DELIBERATE pin** — `EditorSettings::boot_project` with
//!    `boot_project_deliberate` set: a project an author explicitly made the
//!    default, from Preferences. An explicit choice outranks everything but the
//!    environment, and nothing but another explicit choice takes it away.
//! 3. **The showcase** — [`SHOWCASE_RELATIVE`], found by walking up from a start
//!    directory (the running executable's). This is the rung that makes the
//!    island the default *on a machine where it has been built*, with no
//!    settings file and no click.
//! 4. **An AUTOMATIC pin** — the same field with the flag clear: the last
//!    project opened, which the editor writes on every successful open. It sits
//!    BELOW the showcase, which is the whole of the CERT1 audit's ruling.
//! 5. **Nothing.** The start screen, exactly as before.
//!
//! # Why the automatic pin is below the showcase (CERT1 audit ruling)
//!
//! The first version of this module had one pin at rung 2, written by every
//! open. The consequence, measured by the audit rather than argued: **the first
//! time an author opened any other project, the showcase stopped being what the
//! application booted on, for ever, and no UI showed or cleared it.** The owner's
//! sentence is that the island *is* the default level, and "the last thing you
//! opened is the default" is a different sentence.
//!
//! So the pin is split by INTENT rather than by value. A visit is a visit and
//! ranks below the showcase; a decision is a decision and ranks above it. The
//! automatic rung is kept, below, because it is what a machine with **no**
//! showcase wants — there, rung 3 answers `None` and "reopen what I had" is
//! still the right behaviour.
//!
//! One string and one flag rather than two strings, because the two pins are
//! never both interesting: rung 2 answering means rung 4 is never read. The
//! consequence is stated where it is caused — `pin_boot_project` leaves a
//! DELIBERATE pin alone, so opening a scratch project cannot quietly overwrite
//! the choice an author made in Preferences.
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
    /// A **deliberate** pin — a project an author made the default from
    /// Preferences. Rung 2, above the showcase.
    Pinned,
    /// Discovered beside the checkout by [`find_showcase`]. Rung 3.
    Showcase,
    /// An **automatic** pin — the last project opened, written by every
    /// successful open. Rung 4, BELOW the showcase (the CERT1 audit's ruling),
    /// so it answers only on a machine where the showcase was never built.
    LastOpened,
}

impl BootSource {
    /// A short human phrase for a status line.
    ///
    /// Each one has to finish the sentence *"opened Vancouver Island — …"*, and
    /// the two pins have to read differently, because "you chose this" and
    /// "this is where you were" are the two states this wave's ruling separated.
    pub fn phrase(self) -> &'static str {
        match self {
            BootSource::Environment => "from INF_BOOT_PROJECT",
            BootSource::Pinned => "the project you made the default",
            BootSource::Showcase => "the showcase island",
            BootSource::LastOpened => "the last project you opened",
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
/// * `deliberate` — `EditorSettings::boot_project_deliberate`. `true` puts the
///   pin at rung 2, **above** the showcase; `false` puts it at rung 4, below it.
///   That one bit is the whole of the CERT1 audit's ruling: a project an author
///   chose outranks the showcase, and a project they merely visited does not.
/// * `start` — where to begin the showcase walk (the running executable's
///   directory). `None` skips that rung entirely, which is what a caller with no
///   executable path — a test, or a host that would rather not guess — wants.
///
/// A rung whose path does not hold an `inf.toml` is **skipped, not fatal**: a
/// pin left behind by a project the author has since deleted must not stop the
/// application booting, and it must not stop the showcase below it answering.
pub fn resolve(
    env: Option<&str>,
    pinned: &str,
    deliberate: bool,
    start: Option<&Path>,
) -> Option<BootProject> {
    // Each rung is one `answer` call and the ORDER of the calls is the rule, so
    // the ruling reads as a sequence rather than hiding in a loop: the pin
    // appears once, above or below the showcase, and never twice.
    let pin = pinned.trim();
    let env = env.unwrap_or("").trim();
    let answer = |candidate: &str, source: BootSource| -> Option<BootProject> {
        if candidate.is_empty() {
            return None;
        }
        let root = PathBuf::from(candidate);
        is_project_root(&root).then_some(BootProject { root, source })
    };

    if let Some(found) = answer(env, BootSource::Environment) {
        return Some(found);
    }
    if deliberate {
        if let Some(found) = answer(pin, BootSource::Pinned) {
            return Some(found);
        }
    }
    if let Some(dir) = start {
        if let Some(root) = find_showcase(dir) {
            return Some(BootProject {
                root,
                source: BootSource::Showcase,
            });
        }
    }
    match deliberate {
        // A deliberate pin has already been tried; trying it again as an
        // automatic one would make the flag meaningless on a machine with no
        // showcase, where both rungs would answer the same path.
        true => None,
        false => answer(pin, BootSource::LastOpened),
    }
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
        assert_eq!(resolve(None, "", false, Some(&exe_dir)), None);
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

    /// The four-rung world: an env project, a pinned project, an executable
    /// four levels below a holder, and a showcase in that holder.
    struct Rungs {
        _tmp: tempfile::TempDir,
        env: String,
        env_root: PathBuf,
        pin: String,
        pin_root: PathBuf,
        exe_dir: PathBuf,
        showcase: PathBuf,
    }

    fn rungs() -> Rungs {
        let tmp = tempfile::tempdir().unwrap();
        let env_root = project_at(&tmp.path().join("env"));
        let pin_root = project_at(&tmp.path().join("pinned"));
        let exe_dir = tmp
            .path()
            .join("infinity_engine")
            .join("target")
            .join("debug");
        std::fs::create_dir_all(&exe_dir).unwrap();
        let showcase = project_at(&tmp.path().join("island-build").join("project"));
        Rungs {
            env: env_root.to_string_lossy().to_string(),
            env_root,
            pin: pin_root.to_string_lossy().to_string(),
            pin_root,
            exe_dir,
            showcase,
            _tmp: tmp,
        }
    }

    #[test]
    fn the_environment_outranks_every_pin() {
        let r = rungs();
        for deliberate in [false, true] {
            assert_eq!(
                resolve(Some(&r.env), &r.pin, deliberate, Some(&r.exe_dir)),
                Some(BootProject {
                    root: r.env_root.clone(),
                    source: BootSource::Environment
                }),
                "INF_BOOT_PROJECT stopped outranking a {} pin",
                match deliberate {
                    true => "deliberate",
                    false => "automatic",
                }
            );
        }
    }

    /// **(a) A DELIBERATE pin beats the showcase.** The author chose it.
    #[test]
    fn a_deliberate_pin_outranks_the_showcase() {
        let r = rungs();
        assert_eq!(
            resolve(None, &r.pin, true, Some(&r.exe_dir)),
            Some(BootProject {
                root: r.pin_root.clone(),
                source: BootSource::Pinned
            }),
            "a project the author made the default no longer wins — an explicit \
             choice must outrank the showcase, or Preferences does nothing"
        );
    }

    /// **(b) An AUTOMATIC pin does NOT beat the showcase**, and this is the whole
    /// of the CERT1 audit's ruling. Before it there was one pin at rung 2 written
    /// by every open, so the first other project an author opened took the
    /// showcase's place for ever.
    #[test]
    fn an_automatic_pin_does_not_outrank_the_showcase() {
        let r = rungs();
        assert_eq!(
            resolve(None, &r.pin, false, Some(&r.exe_dir)),
            Some(BootProject {
                root: r.showcase.clone(),
                source: BootSource::Showcase
            }),
            "the last project opened took the showcase's place — that is the \
             defect this rung order exists to close"
        );
    }

    /// **(c) An AUTOMATIC pin still answers when there is no showcase.** The
    /// rung is demoted, not deleted: on a machine where `inf island build` has
    /// never run, "reopen what I had" is still what an author wants.
    #[test]
    fn an_automatic_pin_answers_when_no_showcase_was_ever_built() {
        let tmp = tempfile::tempdir().unwrap();
        let pinned = project_at(&tmp.path().join("pinned"));
        let exe_dir = tmp
            .path()
            .join("infinity_engine")
            .join("target")
            .join("debug");
        std::fs::create_dir_all(&exe_dir).unwrap();
        let pin = pinned.to_string_lossy().to_string();

        assert_eq!(
            resolve(None, &pin, false, Some(&exe_dir)),
            Some(BootProject {
                root: pinned,
                source: BootSource::LastOpened
            }),
            "with no showcase to answer, the last project opened must still come \
             back — demoting the rung must not delete it"
        );
        // …and the two pins are distinguishable in the answer, because the status
        // line has to say WHY.
        assert_ne!(BootSource::LastOpened.phrase(), BootSource::Pinned.phrase());
    }

    #[test]
    fn the_showcase_answers_when_nothing_is_pinned_at_all() {
        let r = rungs();
        assert_eq!(
            resolve(None, "", false, Some(&r.exe_dir)),
            Some(BootProject {
                root: r.showcase.clone(),
                source: BootSource::Showcase
            })
        );
        assert_eq!(
            resolve(None, "", false, None),
            None,
            "no start, no showcase rung"
        );
    }

    /// A deliberate pin naming a project that no longer exists must not be
    /// retried as an automatic one below the showcase: the flag would stop
    /// meaning anything on a machine with no showcase, where both rungs would
    /// answer the same path.
    #[test]
    fn a_deliberate_pin_is_not_retried_as_an_automatic_one() {
        let tmp = tempfile::tempdir().unwrap();
        let exe_dir = tmp.path().join("a").join("b");
        std::fs::create_dir_all(&exe_dir).unwrap();
        let real = project_at(&tmp.path().join("still-here"));
        let pin = real.to_string_lossy().to_string();

        // No showcase anywhere: a deliberate pin answers at rung 2, and the same
        // path as an automatic pin answers at rung 4 — with a different source.
        assert_eq!(
            resolve(None, &pin, true, Some(&exe_dir)).map(|b| b.source),
            Some(BootSource::Pinned)
        );
        assert_eq!(
            resolve(None, &pin, false, Some(&exe_dir)).map(|b| b.source),
            Some(BootSource::LastOpened)
        );
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

        // Both upper rungs name nothing that exists; the showcase still answers,
        // whichever kind of pin it was.
        for deliberate in [false, true] {
            assert_eq!(
                resolve(Some(&gone), &gone, deliberate, Some(&exe_dir)),
                Some(BootProject {
                    root: showcase.clone(),
                    source: BootSource::Showcase
                })
            );
        }
    }

    #[test]
    fn whitespace_is_not_a_pin() {
        let tmp = tempfile::tempdir().unwrap();
        let exe_dir = tmp.path().join("a").join("b").join("c");
        std::fs::create_dir_all(&exe_dir).unwrap();
        for deliberate in [false, true] {
            assert_eq!(resolve(Some("   "), "\t", deliberate, Some(&exe_dir)), None);
        }
    }
}
