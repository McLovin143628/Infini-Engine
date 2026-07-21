//! End-to-end cook tests (P9.2): cook the committed platformer sample into a
//! temp dir, assert determinism, blueprint validation, and closure filtering.

use std::path::{Path, PathBuf};

use inf_asset::PackReader;
use inf_packager::{cook, CookError, CookManifest, CookOptions, DEFAULT_PACK_NAME, MANIFEST_FILE};
use inf_project::ProjectManifest;

/// The workspace root, from this crate at `runtime/inf-packager`.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Scaffold a minimal project in `root` with the platformer sample copied into
/// its Content directory. Returns the project root.
fn make_platformer_project(root: &Path) {
    ProjectManifest::new("Platformer Sample", "2d-platformer")
        .save(root)
        .unwrap();
    let content = root.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let sample = workspace_root().join("samples/platformer-2d");
    for f in ["Platformer.inf_lvl", "Coyote.inf_act"] {
        std::fs::copy(sample.join(f), content.join(f)).unwrap();
    }
}

#[test]
fn cooks_the_platformer_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_platformer_project(&proj);
    let out = dir.path().join("out");

    let report = cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");

    // The pack + manifest exist.
    let pack_path = out.join(DEFAULT_PACK_NAME);
    assert!(pack_path.exists(), "pack written");
    assert!(out.join(MANIFEST_FILE).exists(), "manifest written");

    // The level + the blueprint were packed.
    assert_eq!(report.asset_count, 2, "level + blueprint");
    assert_eq!(report.kinds.get("level"), Some(&1));
    assert_eq!(report.kinds.get("blueprint"), Some(&1));
    assert_eq!(report.blueprints_validated, 1);
    assert_eq!(report.levels_rewritten, 1);
    assert_eq!(report.levels.len(), 1);
    assert!(report.root_level.is_some());
    assert!(report.pack_bytes > 0);

    // The pack reads back, and the root level decodes to the platformer.
    let reader = PackReader::open(&pack_path).unwrap();
    assert_eq!(reader.len(), 2);
    let root = report.root_level.unwrap();
    let level_bytes = reader.read(root).expect("root level in pack");
    let level = inf_scene::decode(&level_bytes).expect("cooked level decodes");
    assert_eq!(level.title, "Platformer 2D");
    assert_eq!(level.len(), 5);

    // The manifest names the pack + root level.
    let manifest =
        CookManifest::from_toml(&std::fs::read_to_string(out.join(MANIFEST_FILE)).unwrap())
            .unwrap();
    assert_eq!(manifest.packs, vec![DEFAULT_PACK_NAME.to_string()]);
    assert_eq!(manifest.root_level, Some(root.uuid()));
    assert_eq!(manifest.asset_count, 2);
    assert_eq!(manifest.project_name, "Platformer Sample");
}

#[test]
fn cook_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_platformer_project(&proj);

    let out_a = dir.path().join("a");
    let out_b = dir.path().join("b");
    cook(&proj, &out_a, &CookOptions::default()).unwrap();
    cook(&proj, &out_b, &CookOptions::default()).unwrap();

    let pack_a = std::fs::read(out_a.join(DEFAULT_PACK_NAME)).unwrap();
    let pack_b = std::fs::read(out_b.join(DEFAULT_PACK_NAME)).unwrap();
    assert_eq!(pack_a, pack_b, "two cooks → byte-identical pack");

    let man_a = std::fs::read(out_a.join(MANIFEST_FILE)).unwrap();
    let man_b = std::fs::read(out_b.join(MANIFEST_FILE)).unwrap();
    assert_eq!(man_a, man_b, "two cooks → byte-identical manifest");
}

#[test]
fn a_broken_blueprint_fails_with_a_handler_anchored_error() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    ProjectManifest::new("Broken", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    // A `.inf_act` whose Tick body references an undefined local (n99).
    let broken = r#"{
        "schema_version": 1,
        "id": "act:broken",
        "name": "Broken Actor",
        "events": [
            { "event": "Tick",
              "body": { "id": "tick", "name": "tick",
                        "params": [{ "name": "dt", "ty": "Float" }],
                        "ret": "Unit",
                        "body": [ { "ExprStmt": { "Local": 99 } } ] } }
        ]
    }"#;
    std::fs::write(content.join("Broken.inf_act"), broken).unwrap();

    let out = proj.join("out");
    let err = cook(&proj, &out, &CookOptions::default()).unwrap_err();
    match err {
        CookError::Blueprint {
            class,
            handler,
            message,
            ..
        } => {
            assert_eq!(class, "Broken Actor");
            assert_eq!(handler, "tick");
            assert!(message.contains("n99"), "message: {message}");
        }
        other => panic!("expected a Blueprint error, got {other:?}"),
    }
}

#[test]
fn dependency_closure_excludes_an_unreferenced_stray() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_platformer_project(&proj);

    // A stray texture that no level/blueprint references (synthesized sidecar,
    // no dependency edges, not a root kind) → must not be packed.
    std::fs::write(
        proj.join("Content").join("Stray.inf_tex"),
        b"a stray unreferenced texture payload nobody depends on",
    )
    .unwrap();

    let out = proj.join("out");
    let report = cook(&proj, &out, &CookOptions::default()).unwrap();

    // Still just the level + blueprint; the texture was dropped.
    assert_eq!(report.asset_count, 2, "stray excluded from the closure");
    assert!(
        !report.kinds.contains_key("texture"),
        "no texture in the pack"
    );
}
