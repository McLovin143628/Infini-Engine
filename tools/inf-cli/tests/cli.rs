//! CLI smoke tests, run against the built `inf` binary (cargo provides its path
//! via `CARGO_BIN_EXE_inf`) — the same pattern the player's `pie.rs` uses. The
//! cook logic itself is unit/integration-tested in `inf-packager`; these prove
//! the command wiring, exit codes, and `pack ls`.

use std::path::{Path, PathBuf};
use std::process::Command;

fn inf() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf"))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn write_project(root: &Path, template: &str) {
    std::fs::create_dir_all(root.join("Content")).unwrap();
    std::fs::write(
        root.join("inf.toml"),
        format!(
            "schema_version = 1\nname = \"CliTest\"\nengine_version = \"0.1.0\"\ntemplate = \"{template}\"\n"
        ),
    )
    .unwrap();
}

#[test]
fn cook_then_pack_ls_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    write_project(&proj, "2d-platformer");
    let sample = workspace_root().join("samples/platformer-2d");
    for f in ["Platformer.inf_lvl", "Coyote.inf_act"] {
        std::fs::copy(sample.join(f), proj.join("Content").join(f)).unwrap();
    }

    let out = Command::new(inf())
        .args(["cook", "--project"])
        .arg(&proj)
        .output()
        .expect("inf cook runs");
    assert!(out.status.success(), "cook exit: {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("2 assets"), "report: {stdout}");

    let pack = proj.join("Build").join("content.inf_pack");
    assert!(pack.exists(), "pack written");

    let ls = Command::new(inf())
        .args(["pack", "ls"])
        .arg(&pack)
        .output()
        .expect("inf pack ls runs");
    assert!(ls.status.success());
    let ls_out = String::from_utf8_lossy(&ls.stdout);
    assert!(ls_out.contains("level"), "ls: {ls_out}");
    assert!(ls_out.contains("blueprint"), "ls: {ls_out}");
}

#[test]
fn export_bundles_with_an_explicit_player_bin() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    write_project(&proj, "2d-platformer");
    let sample = workspace_root().join("samples/platformer-2d");
    for f in ["Platformer.inf_lvl", "Coyote.inf_act"] {
        std::fs::copy(sample.join(f), proj.join("Content").join(f)).unwrap();
    }
    // A stand-in player binary so the CLI export path doesn't release-build.
    let fake = dir.path().join("player.bin");
    std::fs::write(&fake, b"fake").unwrap();
    let out = dir.path().join("export");

    let res = Command::new(inf())
        .args(["export", "--project"])
        .arg(&proj)
        .arg("--out")
        .arg(&out)
        .arg("--player-bin")
        .arg(&fake)
        .output()
        .expect("inf export runs");
    assert!(res.status.success(), "export exit: {:?}", res.status);

    let bundle = out.join("CliTest");
    assert!(bundle.join("content.inf_pack").exists(), "pack in bundle");
    assert!(
        bundle.join("player.toml").exists(),
        "launch config in bundle"
    );
    let exe = bundle.join(format!("CliTest{}", std::env::consts::EXE_SUFFIX));
    assert!(exe.exists(), "renamed exe in bundle");
}

/// **C4-40 at the process level** (the round-3 spot-check).
///
/// C4-40 is the cook that produced an unbootable pack and exited 0, so CI
/// shipped it green. The fix is two lines in `cmd_cook` — `if
/// report.has_blocking() { ExitCode::FAILURE }` — and **nothing failed when
/// they were reverted.** The suite covered the `Err(e)` path (a broken
/// blueprint, "cook failed" on stderr), which is a different branch entirely:
/// `cook` returns `Ok(report)` here, the report renders, and the only thing
/// that distinguishes success from failure is the exit code. Round 2 closed the
/// same gap on the *editor's* side (B10 == R2.F8, the Package dialog's verdict);
/// this is the CLI's, and it is the one CI actually runs.
///
/// A project with content but **no levels** is the blocking advisory that needs
/// no fixture to manufacture: `cook.rs` raises "no levels in cook — the build
/// has no boot scene", and everything else about the run succeeds. So the arm
/// distinguishes the two failure modes rather than just "non-zero": stdout must
/// carry the rendered blocking line, and stderr must NOT say "cook failed",
/// because that would mean the cook errored and this test had drifted onto the
/// branch the other one already covers.
#[test]
fn cook_exits_nonzero_on_a_blocking_advisory_it_reported_as_success() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    write_project(&proj, "blank-3d");
    // Content, but no `.inf_lvl` — a cook that succeeds and must not ship.
    let sample = workspace_root().join("samples/platformer-2d");
    std::fs::copy(
        sample.join("Coyote.inf_act"),
        proj.join("Content").join("Coyote.inf_act"),
    )
    .unwrap();

    let out = Command::new(inf())
        .args(["cook", "--project"])
        .arg(&proj)
        .output()
        .expect("inf cook runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        stderr.is_empty() || !stderr.contains("cook failed"),
        "this arm exists for the Ok(report) branch; the cook itself failed: {stderr}"
    );
    assert!(
        stdout.contains("blocking issue"),
        "the report must say the build must not ship: {stdout}"
    );
    assert!(
        !out.status.success(),
        "a cook with a blocking advisory exited {:?} — CI would ship it green",
        out.status.code()
    );
}

#[test]
fn cook_fails_nonzero_on_a_broken_blueprint() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    write_project(&proj, "blank-3d");
    std::fs::write(
        proj.join("Content").join("Broken.inf_act"),
        r#"{ "schema_version": 1, "id": "act:b", "name": "B",
             "events": [ { "event": "Tick",
               "body": { "id": "t", "name": "t",
                         "params": [{ "name": "dt", "ty": "Float" }],
                         "ret": "Unit",
                         "body": [ { "ExprStmt": { "Local": 42 } } ] } } ] }"#,
    )
    .unwrap();

    let out = Command::new(inf())
        .args(["cook", "--project"])
        .arg(&proj)
        .output()
        .expect("inf cook runs");
    assert!(!out.status.success(), "broken cook must exit nonzero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("cook failed"), "stderr: {stderr}");
}
