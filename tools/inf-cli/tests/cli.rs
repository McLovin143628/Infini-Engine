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
