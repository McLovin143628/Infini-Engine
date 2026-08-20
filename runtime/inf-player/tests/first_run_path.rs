//! **`inf new` → `inf cook` → the shipped player runs, for every template.**
//!
//! The inversion of the AAA-readiness certification's IB-7, which measured this:
//!
//! | `inf new --template` | levels scaffolded | `inf cook` |
//! |---|---|---|
//! | `blank-3d` | 0 | blocked — "no levels in cook — the build has no boot scene" |
//! | `2d-platformer` | 0 | blocked, same |
//! | `first-person` | 1 (`Levels/Main.inf_lvl`) | **blocked, same** |
//! | `hybrid-2.5d` | 1 (`Levels/Main.inf_lvl`) | **blocked, same** |
//!
//! Four of four. The first thing anyone does with this engine was a dead end,
//! for two independent reasons that each hid the other: two templates scaffolded
//! no level at all, and the two that did scaffolded it into `<root>/Levels/` —
//! a directory `inf cook` never opens, because the cook reads the *content* root
//! and nothing else.
//!
//! # Why CI did not see it
//!
//! Because the cook-and-run smoke did not use `inf new`. It hand-wrote an
//! `inf.toml`, copied a committed sample into `Content/`, and cooked that — the
//! shape of a project nobody creates. The real first-run path had no arm at all,
//! which is the failure mode this file exists to end and the reason it runs the
//! **real** scaffold rather than a fixture that resembles one.
//!
//! # What each leg proves
//!
//! 1. **`Project::create`** — the door `inf new` and the editor's New Project
//!    dialog both go through — puts a boot scene where the cook looks.
//! 2. **`cook`** resolves a root level and raises no blocking advisory.
//! 3. **The shipped `inf-player` binary** boots the pack, runs 300 frames and
//!    exits 0 — the same three flags the certification's soak used, so the
//!    numbers are comparable.
//!
//! Leg 3 is a subprocess on purpose. An in-process `run_headless` would share
//! this test's working directory, its logger and its global pools; what a studio
//! ships is the executable, and the certification's finding was about what a
//! studio gets.

use std::path::{Path, PathBuf};
use std::process::Command;

use inf_packager::{cook, CookOptions};
use inf_project::{Project, ProjectTemplate};

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
}

/// The frame count the certification's soak used, so a regression here is
/// comparable to the number in the memo.
const RUN_FRAMES: &str = "300";

/// `new` → `cook` → `run`, for one template. Returns the cooked level count so
/// the caller can assert the population rather than only the exit status.
fn first_run(tmp: &Path, name: &str, template: ProjectTemplate) -> usize {
    // ── 1. inf new ──────────────────────────────────────────────────────────
    let project = Project::create(tmp, name, template)
        .unwrap_or_else(|e| panic!("{}: inf new failed: {e}", template.slug()));

    // The boot scene is inside the content root — the whole IB-7 ruling, checked
    // at the filesystem rather than inferred from the manifest.
    let starter = template.starter_level();
    let level = project.levels_root().join(starter.file_name);
    assert!(
        level.is_file(),
        "{}: no boot scene at {}",
        template.slug(),
        level.display()
    );
    assert!(
        level.starts_with(project.content_root()),
        "{}: the boot scene at {} is outside the content root, so the cook \
         cannot see it — this is the IB-7 defect exactly",
        template.slug(),
        level.display()
    );
    assert!(
        project
            .levels_root()
            .join(format!("{}.toml", starter.file_name))
            .is_file(),
        "{}: the boot scene has no sidecar, so its GUID is content-derived",
        template.slug()
    );

    // ── 2. inf cook ─────────────────────────────────────────────────────────
    let out = project.root.join("Build");
    let report = cook(&project.root, &out, &CookOptions::default())
        .unwrap_or_else(|e| panic!("{}: inf cook failed: {e}", template.slug()));
    assert!(
        !report.has_blocking(),
        "{}: cook raised {} blocking advisory(ies): {:?}",
        template.slug(),
        report.blocking.len(),
        report.blocking
    );
    assert!(
        report.root_level.is_some(),
        "{}: the pack has no boot scene",
        template.slug()
    );
    assert!(
        report.pack_path.is_file(),
        "{}: no pack was written",
        template.slug()
    );

    // ── 3. the shipped player runs it ───────────────────────────────────────
    let output = Command::new(player_bin())
        .arg("--pack")
        .arg(&out)
        .args(["--headless", "--run-frames", RUN_FRAMES, "--assert-exit"])
        .output()
        .unwrap_or_else(|e| panic!("{}: spawning the player failed: {e}", template.slug()));
    assert!(
        output.status.success(),
        "{}: the shipped player refused the pack it was just handed.\n\
         --- stdout ---\n{}\n--- stderr ---\n{}",
        template.slug(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    report.levels.len()
}

#[test]
fn every_template_scaffolds_cooks_and_runs() {
    let tmp = tempfile::tempdir().unwrap();
    let templates = ProjectTemplate::all();
    assert_eq!(templates.len(), 4, "the four templates the memo measured");

    for (i, &t) in templates.iter().enumerate() {
        // A name per template with no '.' in it — `sanitize_dir` turns '.' into
        // '-', so a project named after `hybrid-2.5d` would land in a directory
        // spelled differently from its own slug.
        let levels = first_run(tmp.path(), &format!("FirstRun{i}"), t);
        assert!(
            levels >= 1,
            "{}: cooked {levels} levels",
            ProjectTemplate::all()[i].slug()
        );
        eprintln!(
            "IB-7 first run: {:<14} → cooked {levels} level(s), player exit 0 over {RUN_FRAMES} frames",
            t.slug()
        );
    }
}

/// **The pre-ruling layout is reported, not silently ignored** — the migration
/// half of IB-7.
///
/// A project authored before levels became content has its levels at
/// `<root>/Levels/`, which is now nowhere. The cook's answer used to be *"no
/// levels in cook — the build has no boot scene"*, which names the symptom and
/// not one word of the cause, and an author who had just looked at a directory
/// full of `.inf_lvl` files would reasonably conclude the cook was broken.
///
/// The advisory is a warning rather than a blocking one deliberately: it also
/// fires for a project that cooks *fine* and has one stale level stranded
/// outside, which is the case where a silent skip would ship the wrong world.
#[test]
fn a_pre_ruling_project_is_told_where_its_levels_went() {
    let tmp = tempfile::tempdir().unwrap();
    let project = Project::create(tmp.path(), "Legacy", ProjectTemplate::Blank3d).unwrap();

    // Re-create the pre-ruling layout: move the boot scene back to <root>/Levels.
    let starter = ProjectTemplate::Blank3d.starter_level();
    let legacy = project.legacy_levels_root();
    std::fs::create_dir_all(&legacy).unwrap();
    let from = project.levels_root().join(starter.file_name);
    std::fs::rename(&from, legacy.join(starter.file_name)).unwrap();
    std::fs::rename(
        project
            .levels_root()
            .join(format!("{}.toml", starter.file_name)),
        legacy.join(format!("{}.toml", starter.file_name)),
    )
    .unwrap();

    let out = project.root.join("Build");
    let report = cook(&project.root, &out, &CookOptions::default()).expect("cook runs");

    // The cook is blocked, as it always was…
    assert!(
        report.has_blocking(),
        "a project with no level under Content/ must still be blocked"
    );
    // …and now it says WHY, naming the directory, the file and the remedy.
    let advisory = report
        .warnings
        .iter()
        .find(|w| w.contains("OUTSIDE the content root"))
        .unwrap_or_else(|| {
            panic!(
                "no stranded-levels advisory in {:?} — a pre-ruling project is \
                 told its build has no boot scene and nothing about where its \
                 levels are",
                report.warnings
            )
        });
    assert!(
        advisory.contains(starter.file_name),
        "the advisory does not name the stranded file: {advisory}"
    );
    assert!(
        advisory.contains("Levels are content"),
        "the advisory does not name the remedy: {advisory}"
    );
    eprintln!("IB-7 migration advisory: {advisory}");
}

/// **The advisory fires on a project that cooks successfully**, which is the
/// case an "only when there are no levels" check would miss.
///
/// A studio mid-migration has some levels moved and some not. The cook succeeds,
/// the pack boots, and one of the author's levels is simply absent from the
/// build — the failure mode with no symptom, which is what the cook-advisory
/// doctrine exists for.
#[test]
fn a_partly_migrated_project_still_hears_about_the_rest() {
    let tmp = tempfile::tempdir().unwrap();
    let project = Project::create(tmp.path(), "HalfMoved", ProjectTemplate::FirstPerson).unwrap();

    // The boot scene stays where it belongs; a SECOND level is stranded.
    let starter = ProjectTemplate::FirstPerson.starter_level();
    let legacy = project.legacy_levels_root();
    std::fs::create_dir_all(&legacy).unwrap();
    std::fs::copy(
        project.levels_root().join(starter.file_name),
        legacy.join("Rooftops.inf_lvl"),
    )
    .unwrap();

    let out = project.root.join("Build");
    let report = cook(&project.root, &out, &CookOptions::default()).expect("cook runs");

    assert!(
        !report.has_blocking(),
        "this project has a boot scene, so nothing should block: {:?}",
        report.blocking
    );
    let advisory = report
        .warnings
        .iter()
        .find(|w| w.contains("OUTSIDE the content root"))
        .unwrap_or_else(|| {
            panic!(
                "a successful cook silently left a level out of the build: {:?}",
                report.warnings
            )
        });
    assert!(
        advisory.contains("Rooftops.inf_lvl"),
        "the advisory does not name the stranded level: {advisory}"
    );
    // ANTI-VACUITY: the advisory is about the stranded one and not about the
    // boot scene, which cooked correctly.
    assert!(
        report.root_level.is_some(),
        "the boot scene under Content/ must still have cooked"
    );
}

/// **A project whose content root IS the project root hears nothing** — the
/// false-alarm guard.
///
/// `content_dir = "."` puts `<root>/Levels/` *inside* the content root, and the
/// cook's scan is recursive, so those levels are packed. The first version of the
/// guard compared the two paths for equality and would have called every one of
/// them stranded, for ever, while the cook was shipping them.
#[test]
fn a_content_root_that_contains_the_legacy_directory_raises_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("Flat");
    std::fs::create_dir_all(root.join("Levels")).unwrap();
    let mut manifest = inf_project::ProjectManifest::new("Flat", "blank-3d");
    manifest.content_dir = ".".into();
    manifest.save(&root).unwrap();

    // A real level, in the directory that would look "stranded" under the naive
    // rule but is in fact inside the content root.
    let starter = ProjectTemplate::Blank3d.starter_level();
    std::fs::write(root.join("Levels").join(starter.file_name), starter.payload).unwrap();
    std::fs::write(
        root.join("Levels")
            .join(format!("{}.toml", starter.file_name)),
        starter.sidecar,
    )
    .unwrap();

    let out = root.join("Build");
    let report = cook(&root, &out, &CookOptions::default()).expect("cook runs");
    assert!(
        report.root_level.is_some(),
        "the cook did not find a level that is inside its own content root"
    );
    assert!(
        !report
            .warnings
            .iter()
            .any(|w| w.contains("OUTSIDE the content root")),
        "a level the cook PACKED was reported as stranded: {:?}",
        report.warnings
    );
}
