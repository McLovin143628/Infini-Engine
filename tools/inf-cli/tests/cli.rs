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

// ── the GIS door (IB-3) ─────────────────────────────────────────────────────

/// The fixture both front ends import: two roads and a two-metre stub, in WGS 84
/// degrees near Vancouver.
const ROADS_GEOJSON: &str = r#"{"type":"FeatureCollection","features":[
  {"type":"Feature","properties":{"name":"Main St","lanes":4,"road_type":"arterial"},
   "geometry":{"type":"LineString","coordinates":[[-123.12,49.28],[-123.11,49.28],[-123.11,49.29]]}},
  {"type":"Feature","properties":{"name":"Elm","lanes":2},
   "geometry":{"type":"LineString","coordinates":[[-123.11,49.28],[-123.105,49.275]]}},
  {"type":"Feature","properties":{"name":"Stub"},
   "geometry":{"type":"LineString","coordinates":[[-123.12,49.28],[-123.119975,49.28]]}}
]}"#;

fn write_roads(dir: &Path) -> PathBuf {
    let p = dir.join("roads.geojson");
    std::fs::write(&p, ROADS_GEOJSON).unwrap();
    p
}

/// The anchor both front ends transform into — Vancouver, UTM zone 10N.
const ANCHOR: &str = "EPSG:32610,491000,5459000,0";

fn field(stdout: &str, key: &str) -> String {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(&format!("{key}: ")))
        .unwrap_or_else(|| panic!("no {key:?} line in:\n{stdout}"))
        .trim()
        .to_string()
}

/// **THE ONE-DOOR PROOF.**
///
/// The editor's wizard and `inf gis` are two front ends onto one importer, and
/// this is the arm that makes that falsifiable rather than asserted: the real
/// `inf` binary imports a fixture in its own process, and the same request is
/// built in *this* process through `inf_gis::import_layer` — the call the Ring-2
/// command makes. Every count must agree, and so must the plan's digest, which
/// folds every entity's name, kind and coordinate BIT PATTERN.
///
/// Un-fix mutations that this catches: applying the cap in the CLI instead of at
/// the door; naming an entity differently; reading a different attribute for the
/// stream channel; reordering the plan.
#[test]
fn the_cli_and_the_library_import_the_same_fixture_identically() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_roads(dir.path());

    let out = Command::new(inf())
        .args(["gis", "plan"])
        .arg(&path)
        .args(["--kind", "roads", "--crs", "EPSG:4326", "--anchor", ANCHOR])
        .output()
        .expect("inf gis plan runs");
    assert!(
        out.status.success(),
        "plan exit {:?}: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();

    // The same request, in-process, through the door the wizard uses.
    let anchor = inf_gis::anchor_at("EPSG:32610", 491_000.0, 5_459_000.0, 0.0, "unknown").unwrap();
    let mut req = inf_gis::ImportRequest::new(&path, inf_gis::LayerKind::Roads);
    req.source_crs = Some("EPSG:4326".into());
    let lib = inf_gis::import_layer(&req, &anchor).unwrap();

    assert_eq!(field(&stdout, "entities"), lib.plan.count().to_string());
    assert_eq!(field(&stdout, "too-short"), lib.plan.too_short.to_string());
    assert_eq!(field(&stdout, "unusable"), lib.plan.unusable.to_string());
    assert_eq!(field(&stdout, "truncated"), lib.plan.truncated.to_string());
    assert_eq!(field(&stdout, "cap"), lib.plan.cap.to_string());
    assert_eq!(
        field(&stdout, "digest"),
        format!("{:016x}", lib.plan.digest()),
        "the CLI's plan and the library's must be the SAME plan, coordinate bit \
         for coordinate bit — this is what 'one import door' means\n{stdout}"
    );
    // …and the fixture is not vacuous: two roads survive, the 2 m stub does not.
    assert_eq!(lib.plan.count(), 2, "{:?}", lib.plan);
    assert_eq!(lib.plan.too_short, 1);
}

/// The cap is visible, raisable, and a truncated import **fails the pipeline**
/// (IB-14) rather than quietly shipping a city with a hard edge.
#[test]
fn the_entity_cap_is_reported_raisable_and_blocking() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_roads(dir.path());

    let capped = Command::new(inf())
        .args(["gis", "plan"])
        .arg(&path)
        .args([
            "--kind",
            "roads",
            "--crs",
            "EPSG:4326",
            "--anchor",
            ANCHOR,
            "--max",
            "1",
        ])
        .output()
        .unwrap();
    assert!(
        !capped.status.success(),
        "a truncated import must exit NONZERO — a pipeline that ships 1 of 2 roads \
         without stopping is the defect IB-14 names"
    );
    let stdout = String::from_utf8_lossy(&capped.stdout);
    assert_eq!(field(&stdout, "truncated"), "1");
    let stderr = String::from_utf8_lossy(&capped.stderr);
    assert!(
        stderr.contains("--max 2"),
        "the remedy must name the number to raise it to: {stderr}"
    );

    // Raised, it takes the whole layer and exits zero.
    let whole = Command::new(inf())
        .args(["gis", "plan"])
        .arg(&path)
        .args([
            "--kind",
            "roads",
            "--crs",
            "EPSG:4326",
            "--anchor",
            ANCHOR,
            "--max",
            "2",
        ])
        .output()
        .unwrap();
    assert!(whole.status.success());
    assert_eq!(
        field(&String::from_utf8_lossy(&whole.stdout), "truncated"),
        "0"
    );
}

/// `inf gis info` reads a file before the level has an anchor — which is the
/// point of the probe — and suggests the CRS to anchor in.
#[test]
fn gis_info_reads_a_file_and_suggests_an_anchor() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_roads(dir.path());
    let out = Command::new(inf())
        .args(["gis", "info"])
        .arg(&path)
        .args(["--crs", "EPSG:4326"])
        .output()
        .expect("inf gis info runs");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("3 features"), "{s}");
    assert!(s.contains("3 polylines"), "{s}");
    assert!(
        s.contains("suggested anchor CRS: EPSG:32610"),
        "Vancouver is UTM zone 10 north: {s}"
    );
    assert!(s.contains("road_type"), "the fields are listed: {s}");
    assert!(s.contains("lanes"), "{s}");

    // No CRS and no .prj is a refusal with the remedy, not a guess.
    let bare = Command::new(inf())
        .args(["gis", "info"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(!bare.status.success());
    let e = String::from_utf8_lossy(&bare.stderr);
    assert!(
        e.contains("no .prj sidecar") && e.contains("EPSG:4326"),
        "{e}"
    );
}

/// A GIS import needs an anchor, and without one the CLI says so rather than
/// importing a city at the world origin.
#[test]
fn a_plan_with_no_anchor_is_refused_with_its_remedy() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_roads(dir.path());
    let out = Command::new(inf())
        .args(["gis", "plan"])
        .arg(&path)
        .args(["--crs", "EPSG:4326"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let e = String::from_utf8_lossy(&out.stderr);
    assert!(e.contains("--anchor") && e.contains("--level"), "{e}");

    // Web Mercator stays refused AS AN ANCHOR, with its number.
    let merc = Command::new(inf())
        .args(["gis", "plan"])
        .arg(&path)
        .args([
            "--crs",
            "EPSG:4326",
            "--anchor",
            "EPSG:3857,-13704000,6318000,0",
        ])
        .output()
        .unwrap();
    assert!(!merc.status.success());
    let e = String::from_utf8_lossy(&merc.stderr);
    assert!(
        e.contains("geographic") || e.contains("1.5") || e.contains("Web Mercator"),
        "{e}"
    );
}
