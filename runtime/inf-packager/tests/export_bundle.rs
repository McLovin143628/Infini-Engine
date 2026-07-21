//! P9.5 deliverable 2 (cook side): `export` assembles the runnable bundle layout.
//!
//! Uses a **dummy** player binary (a plain file) so this test stays in the cook
//! crate and fast; the *run* of a real exported player lives in `inf-player`'s
//! `bundle` test (it needs the built player binary).

use std::path::{Path, PathBuf};

use inf_packager::{export, CookError, ExportOptions};
use inf_project::ProjectManifest;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn scaffold_platformer(proj: &Path) {
    ProjectManifest::new("My Game", "2d-platformer")
        .save(proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let sample = workspace_root().join("samples/platformer-2d");
    for f in ["Platformer.inf_lvl", "Coyote.inf_act"] {
        std::fs::copy(sample.join(f), content.join(f)).unwrap();
    }
}

#[test]
fn export_writes_bundle_with_renamed_exe_pack_manifest_and_config() {
    let tmp = tempfile::tempdir().unwrap();
    let proj = tmp.path().join("proj");
    scaffold_platformer(&proj);

    // A stand-in "player" binary to copy in.
    let fake_player = tmp.path().join("inf-player-fake");
    std::fs::write(&fake_player, b"#!not-a-real-binary").unwrap();

    let out = tmp.path().join("export");
    let opts = ExportOptions {
        out_dir: Some(out.clone()),
        player_bin: Some(fake_player.clone()),
        ..Default::default()
    };
    let report = export(&proj, &opts).expect("export succeeds");

    let bundle = out.join("My Game");
    assert_eq!(report.bundle_dir, bundle);
    // Exe renamed to the project (+ host exe suffix).
    assert_eq!(
        report.exe_path,
        bundle.join(format!("My Game{}", std::env::consts::EXE_SUFFIX))
    );
    assert!(report.exe_path.is_file());
    // The copied bytes are the player source's.
    assert_eq!(
        std::fs::read(&report.exe_path).unwrap(),
        b"#!not-a-real-binary"
    );

    assert!(bundle.join("content.inf_pack").is_file());
    assert!(bundle.join("manifest.toml").is_file());
    assert!(bundle.join("docs").join("PACKAGING.txt").is_file());

    // player.toml is valid TOML pointing at the pack + carrying window defaults.
    let cfg = std::fs::read_to_string(&report.config_path).unwrap();
    let parsed: toml::Value = toml::from_str(&cfg).unwrap();
    assert_eq!(parsed["pack"].as_str(), Some("content.inf_pack"));
    assert_eq!(parsed["title"].as_str(), Some("My Game"));
    assert_eq!(parsed["width"].as_integer(), Some(1280));

    // The cook actually packed the level + blueprint.
    assert_eq!(report.cook.asset_count, 2);
}

#[test]
fn export_errors_when_player_bin_missing() {
    let tmp = tempfile::tempdir().unwrap();
    let proj = tmp.path().join("proj");
    scaffold_platformer(&proj);

    let opts = ExportOptions {
        out_dir: Some(tmp.path().join("export")),
        player_bin: Some(tmp.path().join("does-not-exist")),
        ..Default::default()
    };
    let err = export(&proj, &opts).unwrap_err();
    assert!(matches!(err, CookError::Export(_)), "got {err:?}");
}
