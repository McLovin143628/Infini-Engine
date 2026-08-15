//! **The cook's exit code** (C4-40) and **the edges it could not read** (C4-41).
//!
//! Both findings are the same shape: the cook already knew, said so in a line of
//! `warnings`, and returned success. `inf cook` printed the line and exited 0, so
//! CI cooked an unbootable pack — or one missing the content its levels name —
//! and reported green.
//!
//! These arms are over the **classification**, not the wording: a message can be
//! reworded, and `CookReport::blocking` is the decision.

use std::path::{Path, PathBuf};

use inf_asset::{AssetKind, AssetSidecar, ContentHash};
use inf_packager::{cook, CookOptions};
use inf_project::ProjectManifest;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// A project with the platformer sample's level + blueprint (the CI smoke's
/// content) — a cook that must come back clean.
fn make_platformer_project(root: &Path) {
    ProjectManifest::new("Blocking Fixture", "2d-platformer")
        .save(root)
        .unwrap();
    let content = root.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let sample = workspace_root().join("samples/platformer-2d");
    for f in [
        "Platformer.inf_lvl",
        "Coyote.inf_act",
        "Coyote.inf_act.toml",
    ] {
        std::fs::copy(sample.join(f), content.join(f)).unwrap();
    }
}

/// The control: the committed sample cooks with **nothing** blocking. Without
/// this arm, a `blocking` that answered "everything" would pass every assertion
/// below — and it is also the local statement of the CI smoke's premise, since
/// `inf cook` now turns this list into the exit code.
#[test]
fn the_committed_sample_cooks_with_nothing_blocking() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_platformer_project(&proj);
    let report = cook(&proj, &dir.path().join("out"), &CookOptions::default()).expect("cook");
    assert!(
        !report.has_blocking(),
        "the sample the CI smoke cooks must exit 0: {:?}",
        report.blocking
    );
    assert!(
        report.warnings.is_empty(),
        "and it carries no advisories at all: {:?}",
        report.warnings
    );
}

/// C4-40: a pack with no level cannot boot. The cook still succeeds — the pack
/// is written and is a legitimate artefact — but the verdict is not "success".
#[test]
fn a_cook_with_no_boot_scene_is_blocking() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    ProjectManifest::new("Empty", "empty").save(&proj).unwrap();
    std::fs::create_dir_all(proj.join("Content")).unwrap();

    let report = cook(&proj, &dir.path().join("out"), &CookOptions::default()).expect("cook");
    assert!(report.root_level.is_none());
    assert!(
        report.has_blocking(),
        "a build with no boot scene must not ship"
    );
    assert!(
        report.blocking.iter().any(|w| w.contains("no boot scene")),
        "{:?}",
        report.blocking
    );
    // Blocking advisories are a SUBSET of warnings, so a reader of the old field
    // still sees everything.
    for b in &report.blocking {
        assert!(report.warnings.contains(b), "blocking not in warnings: {b}");
    }
    // …and the rendering says so, at error level.
    let rendered = report.render();
    assert!(rendered.contains("error: "), "{rendered}");
    assert!(rendered.contains("must not ship"), "{rendered}");
}

/// C4-41: an asset whose payload cannot be decoded contributes **no forward
/// edges**, so everything it references silently leaves the pack. The arm parks
/// a corrupt `.inf_pcg` in the closure and asserts the cook says so.
///
/// A `.inf_pcg` is the right probe because its edge walk has two failure modes
/// (an undecodable payload and an unparseable authored graph) and one legitimate
/// empty answer (a v1 document-only payload), which the fix must not confuse.
#[test]
fn an_undecodable_payload_in_the_closure_is_blocking() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_platformer_project(&proj);
    let content = proj.join("Content");

    // A blueprint is a cook ROOT, so this file is in the closure by construction
    // — the point is the *edge walk*, not reachability. Give the corrupt payload
    // a kind whose edges the cook derives.
    let path = content.join("Broken.inf_pcg");
    std::fs::write(&path, b"not a pcg payload at all").unwrap();
    let mut side = AssetSidecar::new(
        inf_asset::AssetId::new(),
        AssetKind::Pcg,
        ContentHash::of(b"not a pcg payload at all"),
    );
    side.dependencies = Vec::new();
    side.save(&path).unwrap();

    // Bind it into the closure the only way a cook follows: make the blueprint's
    // sidecar declare it as a dependency.
    let bp = content.join("Coyote.inf_act");
    let mut bp_side = AssetSidecar::load(&bp).expect("the sample sidecar loads");
    bp_side.dependencies = vec![side.guid];
    bp_side.save(&bp).unwrap();

    let report = cook(&proj, &dir.path().join("out"), &CookOptions::default()).expect("cook");
    assert!(
        report.has_blocking(),
        "an unreadable edge must not ship silently: {:?}",
        report.warnings
    );
    let msg = report
        .blocking
        .iter()
        .find(|w| w.contains("dependency edges"))
        .unwrap_or_else(|| panic!("no C4-41 advisory in {:?}", report.blocking));
    assert!(msg.contains("Broken"), "names the asset: {msg}");
    assert!(msg.contains("MISSING from the pack"), "{msg}");
}

/// The distinction the fix turns on: a **v1 document-only** `.inf_pcg` carries no
/// authored graph, and that is not a defect. Without this arm the fix could be
/// "report everything that answered empty", which would block every project that
/// predates P10.5b.
#[test]
fn a_graphless_pcg_payload_is_not_an_unreadable_edge() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_platformer_project(&proj);
    let content = proj.join("Content");

    let payload = inf_pcg::PcgAssetPayload::new(inf_pcg::PcgDocument::default());
    let bytes = payload.encode().expect("v1 payload encodes");
    let path = content.join("Plain.inf_pcg");
    std::fs::write(&path, &bytes).unwrap();
    let mut side = AssetSidecar::new(
        inf_asset::AssetId::new(),
        AssetKind::Pcg,
        ContentHash::of(&bytes),
    );
    side.dependencies = Vec::new();
    side.save(&path).unwrap();

    let bp = content.join("Coyote.inf_act");
    let mut bp_side = AssetSidecar::load(&bp).expect("the sample sidecar loads");
    bp_side.dependencies = vec![side.guid];
    bp_side.save(&bp).unwrap();

    let report = cook(&proj, &dir.path().join("out"), &CookOptions::default()).expect("cook");
    assert!(
        !report.has_blocking(),
        "a payload with no graph is absent, not broken: {:?}",
        report.blocking
    );
}
