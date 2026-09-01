//! P9.5 deliverable 2: desktop export, end to end.
//!
//! Export the platformer into a runnable bundle (passing the **debug** player as
//! `--player-bin` so the test never pays a release build), assert the folder
//! layout, then **run the exported executable** headlessly from inside the bundle
//! — it must load its own `player.toml` + `content.ipack` and exit 0.

use std::path::{Path, PathBuf};
use std::process::Command;

use inf_packager::{export, ExportOptions};
use inf_project::ProjectManifest;

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn scaffold_platformer(proj: &Path) {
    ProjectManifest::new("Platformer Sample", "2d-platformer")
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
fn export_bundle_layout_and_runs_headless() {
    let tmp = tempfile::tempdir().unwrap();
    let proj = tmp.path().join("proj");
    scaffold_platformer(&proj);

    let out = tmp.path().join("export");
    let opts = ExportOptions {
        out_dir: Some(out.clone()),
        player_bin: Some(player_bin()),
        ..Default::default()
    };
    let report = export(&proj, &opts).expect("export succeeds");

    // ── layout ──────────────────────────────────────────────────────────────
    let bundle = out.join("Platformer Sample");
    assert_eq!(report.bundle_dir, bundle);
    assert!(report.exe_path.is_file(), "renamed player exe present");
    assert_eq!(
        report.exe_path.file_stem().unwrap().to_str().unwrap(),
        "Platformer Sample",
        "exe renamed to the project"
    );
    assert!(bundle.join("content.ipack").is_file(), "pack bundled");
    assert!(bundle.join("manifest.toml").is_file(), "manifest bundled");
    assert!(
        bundle.join("player.toml").is_file(),
        "launch config written"
    );
    assert!(
        bundle.join("docs").join("PACKAGING.txt").is_file(),
        "packaging note written"
    );

    // player.toml points at the bundled pack + carries the window title.
    let cfg = std::fs::read_to_string(bundle.join("player.toml")).unwrap();
    assert!(cfg.contains("pack = \"content.ipack\""), "config: {cfg}");
    assert!(cfg.contains("Platformer Sample"), "config: {cfg}");

    // ── run the exported exe headless (loads its own player.toml + pack) ──────
    let output = Command::new(&report.exe_path)
        .args(["--headless", "--run-frames", "60", "--assert-exit"])
        .current_dir(&bundle)
        .output()
        .expect("exported player runs");
    assert!(
        output.status.success(),
        "exported game must exit 0; stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("final-state-hash:"),
        "the run printed a determinism hash; stdout:\n{stdout}"
    );
    // The label came from the cooked level (proving it booted the bundled pack,
    // not the built-in demo).
    assert!(
        stdout.contains("Platformer 2D"),
        "booted the bundled level; stdout:\n{stdout}"
    );
}
