//! **The cook path for InfiniScript** (SCRIPT1b clause 3): a `.infini` under
//! `Content/` becomes a `BlueprintClass` in the pack, its asset edges enter the
//! closure, and every way it can be wrong is a value.
//!
//! # The ship decision this file gates
//!
//! The brief asked for both doors priced and one taken. Taken: **the cook lowers
//! a script and packs the IR for the shipped interpreter.**
//!
//! * it is what the engine already does with a `.inf_act` — the cook ships the
//!   IR as data and PIE and the shipped player both interpret it, so a script
//!   riding the same road costs no new mechanism and inherits the PIE ==
//!   shipping arms that road already has;
//! * the transpile-to-Rust door writes into the **user's own crate**
//!   (`inf_editor_core::blueprint_source` → `<project>/src/blueprints/`), and
//!   the shipped `inf-player` is a *prebuilt generic binary that loads packs* —
//!   it does not link a project crate at all. So transpiling changes nothing
//!   about what ships until somebody prices a per-project player build, which is
//!   a build-system decision this wave did not take;
//! * and after SCRIPT1b's crown gate the choice costs no correctness, because
//!   interpreted and compiled are now *measured* equal over a compiled program
//!   rather than over four hand-written mirrors.
//!
//! The transpile door stays where it is — the Code tab, and the WASM mod cook —
//! and SCRIPT3 takes the per-script-class decision with measurements, as the memo
//! says. What is decided here is the **default**, and it is: pack the IR.

use std::path::{Path, PathBuf};

use inf_asset::{AssetId, AssetKind, PackReader};
use inf_blueprint::BlueprintClass;
use inf_packager::{cook, CookError, CookOptions, DEFAULT_PACK_NAME};
use inf_project::ProjectManifest;

const GATE: &str = "\
actor \"Gate\"

var open_frac: float = 0.0
var speed: float = 0.5

on begin_play()
  debug.print(\"gate armed\")
end

on tick(dt)
  local step = speed * dt
  open_frac = open_frac + step
  if open_frac > 1.0 then
    open_frac = 1.0
  end
  engine.set_rotation(open_frac * 90.0)
end
";

fn scaffold(root: &Path, files: &[(&str, &[u8])]) {
    ProjectManifest::new("Scripted", "blank-3d")
        .save(root)
        .unwrap();
    let content = root.join("Content");
    std::fs::create_dir_all(content.join("Levels")).unwrap();
    // A boot level, so the cook is not blocked for having none.
    //
    // The BLANK one, deliberately: the platformer's level carries an `actor`
    // binding, which is its own edge into the closure — a gate for the script's
    // edge that used a level already pulling the same asset in would pass with
    // the script's arm deleted.
    let root = workspace_root();
    std::fs::copy(
        root.join("templates/blank-3d/Blank.inf_lvl"),
        content.join("Levels/Main.inf_lvl"),
    )
    .unwrap();
    std::fs::copy(
        root.join("templates/blank-3d/Blank.inf_lvl.toml"),
        content.join("Levels/Main.inf_lvl.toml"),
    )
    .unwrap();
    for (rel, bytes) in files {
        let p = content.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn packed_script(out: &Path) -> (AssetId, BlueprintClass) {
    let reader = PackReader::open(&out.join(DEFAULT_PACK_NAME)).expect("the pack opens");
    let entry = reader
        .index()
        .find(|e| e.kind == AssetKind::Script)
        .expect("the pack carries a script entry");
    let bytes = reader.read(entry.guid).expect("the blob reads");
    let class = serde_json::from_slice::<BlueprintClass>(&bytes)
        .expect("a cooked script is a BlueprintClass in pretty JSON");
    (entry.guid, class)
}

/// **The clause, end to end.** A `.infini` under `Content/Scripts/` is a cook
/// root, it lowers, and what lands in the pack is a class the *player's own*
/// decoder reads.
#[test]
fn a_script_under_content_cooks_into_a_class_the_player_can_read() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    scaffold(&proj, &[("Scripts/Gate.infini", GATE.as_bytes())]);
    let out = dir.path().join("out");

    let report = cook(&proj, &out, &CookOptions::default()).expect("the cook succeeds");
    println!("{}", report.render());
    assert_eq!(report.kinds.get("script"), Some(&1), "{:?}", report.kinds);
    assert!(!report.has_blocking(), "{:?}", report.blocking);

    let (_guid, class) = packed_script(&out);
    assert_eq!(class.name, "Gate");
    assert_eq!(class.variables.len(), 2);
    assert_eq!(class.events.len(), 2, "begin_play and tick");

    // "The player can boot it" is asserted where the player lives —
    // `runtime/inf-player/tests/script_gameplay_gate.rs`, which reads this same
    // pack back through `PackLevelSource` and runs it. `inf-packager` cannot
    // dev-depend on `inf-player` (the player already dev-depends on the cook),
    // and a claim about the player asserted here would be a claim about
    // `serde_json` anyway.
}

/// **The same program cooks to the same class on any host** — the cross-host
/// determinism law, extended from the lowering to the cooked artifact.
///
/// # And the one thing no reader can normalise, found by writing this arm
///
/// The lexer is insensitive to CRLF, a lone CR and a leading byte-order mark, so
/// what a script *means* is a pure function of the text however an editor saved
/// it. The cooked class's **id** is not: it is the asset GUID, and a `.infini`
/// with no committed sidecar gets a GUID synthesised from its **content hash** —
/// a hash over bytes, which nothing downstream can normalise after the fact.
///
/// So this arm asserts the strong half (the program is identical) and
/// *measures* the weak half rather than asserting an equality that would be a
/// lie. The remedy is a `.gitattributes` rule, which is where the CRLF law has
/// always lived for files whose bytes are their identity — pinned by the arm
/// below.
#[test]
fn the_same_program_cooks_to_the_same_class_on_any_host() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a");
    scaffold(&a, &[("Scripts/Gate.infini", GATE.as_bytes())]);
    let out_a = dir.path().join("out_a");
    cook(&a, &out_a, &CookOptions::default()).expect("cook a");
    let (guid_a, mut class_a) = packed_script(&out_a);

    // The same script as a Windows editor would have saved it.
    let mut windows = "\u{feff}".as_bytes().to_vec();
    windows.extend_from_slice(GATE.replace('\n', "\r\n").as_bytes());
    assert_ne!(windows, GATE.as_bytes(), "the fixture must really differ");
    let b = dir.path().join("b");
    scaffold(&b, &[("Scripts/Gate.infini", &windows)]);
    let out_b = dir.path().join("out_b");
    cook(&b, &out_b, &CookOptions::default()).expect("cook b");
    let (guid_b, mut class_b) = packed_script(&out_b);

    println!("unix guid {guid_a}, windows guid {guid_b}");
    assert_ne!(
        guid_a, guid_b,
        "if these ever agree, the synthesised GUID stopped being the content \
         hash and this arm's whole reasoning — and the .gitattributes rule it \
         justifies — needs re-deriving"
    );
    // The PROGRAM — every variable, every handler, every statement — is
    // identical. Only the identity the database gave the file differs.
    class_a.id = String::new();
    class_b.id = String::new();
    let bytes_a = serde_json::to_vec_pretty(&class_a).unwrap();
    let bytes_b = serde_json::to_vec_pretty(&class_b).unwrap();
    assert_eq!(
        bytes_a, bytes_b,
        "CRLF and a byte-order mark changed the cooked PROGRAM — the \
         determinism law reaches the cook, not only the lowering"
    );
    println!("the cooked class is {} bytes on both hosts", bytes_a.len());
    assert!(bytes_a.len() > 200, "a class this small proves nothing");

    // …and a second cook of the SAME project is byte-identical end to end,
    // which is the property `cook_determinism` holds for every other kind.
    let out_again = dir.path().join("out_again");
    cook(&a, &out_again, &CookOptions::default()).expect("cook a again");
    assert_eq!(
        std::fs::read(out_a.join(DEFAULT_PACK_NAME)).unwrap(),
        std::fs::read(out_again.join(DEFAULT_PACK_NAME)).unwrap(),
        "two cooks of one project must be byte-identical"
    );
}

/// The `.gitattributes` rule the arm above derives, asserted where it is
/// reasoned about rather than left as a comment somebody deletes.
#[test]
fn committed_scripts_are_pinned_against_line_ending_conversion() {
    let attrs = std::fs::read_to_string(workspace_root().join(".gitattributes"))
        .expect(".gitattributes is committed");
    assert!(
        attrs.lines().any(|l| l.trim() == "*.infini -text"),
        "a committed .infini whose bytes git may rewrite gets a different asset \
         GUID on Windows than on Linux"
    );
}

/// A script that does not parse **fails the build with its line and column**,
/// not with `<decode>`.
#[test]
fn a_script_that_does_not_parse_fails_the_build_by_line_and_column() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    scaffold(
        &proj,
        &[(
            "Scripts/Broken.infini",
            b"on tick(dt)\n  engine.set_rotation(\nend\n",
        )],
    );
    let out = dir.path().join("out");
    match cook(&proj, &out, &CookOptions::default()) {
        Err(CookError::Script {
            name, diagnostics, ..
        }) => {
            println!("the cook said: {name}\n{diagnostics}");
            assert_eq!(name, "Broken");
            assert!(diagnostics.contains("error:"), "{diagnostics}");
            assert!(
                diagnostics.contains(":1:")
                    || diagnostics.contains(":2:")
                    || diagnostics.contains(":3:"),
                "a script's anchor is a line and a column: {diagnostics}"
            );
        }
        other => panic!("expected a script refusal, got {other:?}"),
    }
}

/// A `.infini` holding bytes that are not UTF-8 fails the build **through the
/// file door**, naming the byte offset — the SCRIPT1a audit's routed item, met
/// on the cook path.
#[test]
fn a_script_that_is_not_utf8_fails_the_build_by_byte_offset() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    scaffold(
        &proj,
        &[(
            "Scripts/Latin1.infini",
            b"actor \"Caf\xe9\"\non tick(dt)\nend\n",
        )],
    );
    let out = dir.path().join("out");
    match cook(&proj, &out, &CookOptions::default()) {
        Err(CookError::Script { diagnostics, .. }) => {
            println!("the cook said:\n{diagnostics}");
            assert!(diagnostics.contains("not valid UTF-8"), "{diagnostics}");
            assert!(diagnostics.contains("byte 10"), "{diagnostics}");
        }
        other => panic!("expected a script refusal, got {other:?}"),
    }
}

/// **`i64::MIN` previews and cooks — and says so.** SCRIPT1a pinned that the
/// transpiler refuses the literal and routed the reporting here: an advisory,
/// naming the handler and the remedy, not a stack trace.
#[test]
fn the_int_min_literal_cooks_with_an_advisory() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let src = "on begin_play()\n  var.set(\"floor\", -9223372036854775808)\nend\n";
    scaffold(&proj, &[("Scripts/Floor.infini", src.as_bytes())]);
    let out = dir.path().join("out");
    let report = cook(&proj, &out, &CookOptions::default()).expect("it COOKS");
    println!("{}", report.render());
    let hit = report
        .warnings
        .iter()
        .find(|w| w.contains("9223372036854775808"))
        .unwrap_or_else(|| panic!("no i64::MIN advisory in {:?}", report.warnings));
    assert!(hit.contains("begin_play"), "{hit}");
    assert!(hit.contains("CANNOT be transpiled"), "{hit}");
    assert!(
        !report.has_blocking(),
        "the literal runs — it must not block a build: {:?}",
        report.blocking
    );
    // …and the packed class really holds it, so the advisory is about a program
    // that exists rather than about a parse that failed.
    let (_, class) = packed_script(&out);
    assert_eq!(class.events.len(), 1);

    // The control: the same script one below the minimum draws no advisory.
    let ok = dir.path().join("ok");
    scaffold(
        &ok,
        &[(
            "Scripts/Floor.infini",
            "on begin_play()\n  var.set(\"floor\", -9223372036854775807)\nend\n".as_bytes(),
        )],
    );
    let out2 = dir.path().join("out2");
    let clean = cook(&ok, &out2, &CookOptions::default()).expect("cooks");
    assert!(
        !clean
            .warnings
            .iter()
            .any(|w| w.contains("9223372036854775")),
        "{:?}",
        clean.warnings
    );
}

/// **THE SK1c BLOCKER-4 ARM.** A script that names an asset pulls it into the
/// cook closure — the edge `asset_deps` did not have for *any* IR-carrying kind
/// until this wave.
#[test]
fn a_script_named_asset_enters_the_cook_closure() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    // The platformer's committed actor, copied in with its sidecar so it keeps
    // its GUID — and named by a script, by file name.
    let sample = workspace_root().join("samples/platformer-2d");
    let act = std::fs::read(sample.join("Coyote.inf_act")).unwrap();
    let side = std::fs::read(sample.join("Coyote.inf_act.toml")).unwrap();
    let src = "on begin_play()\n  engine.spawn(\"Coyote\")\nend\n";
    scaffold(
        &proj,
        &[
            ("Scripts/Spawner.infini", src.as_bytes()),
            ("Coyote.inf_act", &act),
            ("Coyote.inf_act.toml", &side),
        ],
    );
    let out = dir.path().join("out");
    let report = cook(
        &proj,
        &out,
        &CookOptions {
            // **Explicit roots**, so the actor can only reach the pack THROUGH
            // the script's edge. Without this the actor is a root in its own
            // right (`is_root_kind`) and the arm would pass with no edge at all
            // — a vacuous gate of exactly the shape this house keeps repairing.
            roots: Some(roots(&proj)),
            ..Default::default()
        },
    )
    .expect("the cook succeeds");
    println!("{}", report.render());
    assert!(!report.has_blocking(), "{:?}", report.blocking);
    assert_eq!(
        report.kinds.get("blueprint"),
        Some(&1),
        "the named actor must be in the pack — SK1c's whole finding: {:?}",
        report.kinds
    );

    // The falsifier: the same cook with the name misspelled packs the script
    // alone and says so, BLOCKING.
    let bad = dir.path().join("bad");
    scaffold(
        &bad,
        &[
            (
                "Scripts/Spawner.infini",
                "on begin_play()\n  engine.spawn(\"Coyot\")\nend\n".as_bytes(),
            ),
            ("Coyote.inf_act", &act),
            ("Coyote.inf_act.toml", &side),
        ],
    );
    let out2 = dir.path().join("out2");
    let report2 = cook(
        &bad,
        &out2,
        &CookOptions {
            roots: Some(roots(&bad)),
            ..Default::default()
        },
    )
    .expect("a dangling name does not stop the cook running");
    println!("{}", report2.render());
    assert_eq!(report2.kinds.get("blueprint"), None, "{:?}", report2.kinds);
    assert!(
        report2.blocking.iter().any(|b| b.contains("`Coyot`")),
        "a name that resolves to nothing must BLOCK and name itself: {:?}",
        report2.blocking
    );
}

/// The **script and the boot level** as explicit cook roots.
///
/// The level is here so the build is not blocked for having no boot scene; the
/// blueprint deliberately is NOT, so the only way it can reach the pack is the
/// script's edge. Without that restriction the actor is a root in its own right
/// (`is_root_kind`) and the gate would pass with no edge at all — the vacuous
/// shape this house keeps repairing.
fn roots(proj: &Path) -> Vec<AssetId> {
    let mut db = inf_asset::AssetDb::new(proj.join("Content"));
    db.scan().expect("scan");
    let mut out: Vec<AssetId> = Vec::new();
    for kind in [AssetKind::Script, AssetKind::Level] {
        out.push(
            db.by_kind(kind)
                .next()
                .unwrap_or_else(|| panic!("no {kind:?} in the database"))
                .id(),
        );
    }
    out
}

/// **The extension is spelled once.** `inf_script::SCRIPT_EXT` and
/// `AssetKind::Script`'s row are two tables in two crates that a rename would
/// otherwise split silently — the compiler would still build, and a `.infini`
/// would simply stop being an asset.
#[test]
fn the_asset_database_and_the_compiler_agree_about_what_a_script_is() {
    assert_eq!(AssetKind::Script.extension(), Some(inf_script::SCRIPT_EXT));
    assert_eq!(
        AssetKind::from_extension(inf_script::SCRIPT_EXT),
        AssetKind::Script
    );
    assert!(inf_script::is_script_path(Path::new("a.infini")));
    assert_eq!(
        AssetKind::from_path(Path::new("a.infini")),
        AssetKind::Script
    );
}
