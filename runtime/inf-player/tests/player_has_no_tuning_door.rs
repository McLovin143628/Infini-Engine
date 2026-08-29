//! **The shipped player has no tuning door** (P29.5, pillar S4's other half).
//!
//! Live tuning is an **editor** capability. §13's PIE == shipping discipline says
//! the two hosts run the same simulation from the same document, so a host that
//! can be nudged while the other cannot is not a faster workflow — it is a second
//! simulation wearing the first one's name, and every parity gate in this
//! repository would go on passing while it happened.
//!
//! **SCRIPT1b extends it to hot reload**, which is the same capability wearing a
//! different name: `SimSession::reload_class` swaps a recompiled InfiniScript
//! into a running editor Simulate, and `RuntimeSim` has no such door and must
//! not grow one. A shipped player's classes come out of a pack, once, at load.
//!
//! So the rule is: `inf_editor_core::tuning` is Ring 1, `SimSession` is the only
//! thing that holds a queue of tunes *or of classes*, and the player links
//! neither. Three
//! assertions, and the third is the one that makes the first two mean anything:
//!
//! 1. `inf-editor-core` is a **dev-dependency** of `inf-player` and not a
//!    dependency — so the shipped binary cannot call the door even in principle;
//! 2. no source file the player ships names a tune;
//! 3. **the predicate can see one** — the same search over the editor's own
//!    files finds every token, so a zero above is a zero and not a typo.
//!
//! Point 3 is this repository's own vacuity law, and it is not theoretical here:
//! a gate that greps for `TuneScope` in a tree where the type is spelled
//! `TuningScope` reports a clean player for ever.

use std::path::{Path, PathBuf};

/// The spellings that would mean the player had a tuning door.
///
/// Each is a *type or function* name rather than a word: `tune` on its own
/// appears in prose all over this tree, and a gate that fired on prose would be
/// turned off within a week.
const TOKENS: [&str; 8] = [
    "TuneScope",
    "apply_tune",
    "pending_tunes",
    "kept_tunes",
    "inf_editor_core::tuning",
    // ── SCRIPT1b ── **and no hot-reload door either**, for exactly the same
    //    reason. A host that can have its gameplay code swapped while the other
    //    cannot is a second simulation wearing the first one's name, and every
    //    parity gate here would go on passing while it happened. Hot reload is
    //    an EDITOR capability: the shipped player's classes come out of a pack,
    //    once, at load.
    "reload_class",
    "pending_classes",
    "ScriptSwap",
];

fn repo_root() -> PathBuf {
    // `runtime/inf-player` → the workspace root.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Every `.rs` file under `dir`, recursively, sorted — determinism, and so a
/// failure names the same file twice in a row.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// `(file, token)` for every hit.
fn hits(files: &[PathBuf]) -> Vec<(String, &'static str)> {
    let mut out = Vec::new();
    for f in files {
        let Ok(text) = std::fs::read_to_string(f) else {
            continue;
        };
        for t in TOKENS {
            if text.contains(t) {
                out.push((f.display().to_string(), t));
            }
        }
    }
    out
}

/// **The player ships no tuning door**, and the search that says so can see one.
#[test]
fn no_shipped_player_source_names_a_tuning_door() {
    let root = repo_root();
    let player_src = root.join("runtime").join("inf-player").join("src");
    let files = rust_files(&player_src);
    assert!(
        files.len() > 10,
        "the player's source tree has {} files — this gate is looking at the wrong place ({})",
        files.len(),
        player_src.display()
    );
    let found = hits(&files);
    assert!(
        found.is_empty(),
        "the shipped player names a tuning door: {found:?}"
    );

    // **The vacuity check.** The same predicate over the editor's own files must
    // find EVERY token — otherwise the zero above is about a spelling nobody
    // uses rather than about a door nobody opened.
    let editor = root
        .join("editor")
        .join("crates")
        .join("inf-editor-core")
        .join("src");
    let editor_hits = hits(&rust_files(&editor));
    let mut seen: std::collections::BTreeSet<&str> = editor_hits.iter().map(|(_, t)| *t).collect();
    // `inf_editor_core::tuning` is how a *consumer* names the module; inside the
    // crate it is `crate::tuning`, so that one is checked against the studio's
    // Ring-2 caller instead — where it is spelled exactly as a player would have
    // to spell it.
    seen.insert("inf_editor_core::tuning");
    let studio = root
        .join("editor")
        .join("studio")
        .join("src-tauri")
        .join("src");
    let studio_hits: std::collections::BTreeSet<&str> =
        hits(&rust_files(&studio)).iter().map(|(_, t)| *t).collect();
    assert!(
        studio_hits.contains("inf_editor_core::tuning"),
        "Ring 2 does not name the tuning module, so this gate's consumer spelling is stale"
    );
    for t in TOKENS {
        assert!(
            seen.contains(t),
            "`{t}` is not found anywhere in the editor either — this gate is searching for a spelling that does not exist"
        );
    }
}

/// **The player cannot link the doors**: `inf-editor-core` and `inf-script` are
/// dev-dependencies.
///
/// The source search above is about what is *written*; this is about what is
/// *reachable*. A player that gained a normal dependency on either crate would
/// pass the search on the day it did.
///
/// `inf-script` joined at SCRIPT1b, and the SCRIPT1b audit is why it is in this
/// loop rather than only in a comment. The manifest's own note says *"the
/// SHIPPED player reads the class the cook lowered, never a script, so it links
/// no lexer and no parser (`player_has_no_tuning_door` keeps that honest)"* —
/// and this gate was hard-coded to one crate name, so it did not. The claim the
/// pack carries a lowered `BlueprintClass` and never `.infini` text is exactly
/// the claim "the parser is not linked", and it is worth a line of code rather
/// than a sentence.
#[test]
fn the_player_depends_on_the_editor_crate_only_for_tests() {
    let manifest = std::fs::read_to_string(
        repo_root()
            .join("runtime")
            .join("inf-player")
            .join("Cargo.toml"),
    )
    .expect("the player's manifest");

    for crate_name in ["inf-editor-core", "inf-script"] {
        // Split into sections at top-level `[table]` headers, keeping the header.
        let mut section = String::from("[package]");
        let mut in_deps = Vec::new();
        let mut in_dev = false;
        let mut dev_names_it = false;
        for line in manifest.lines() {
            let t = line.trim();
            if t.starts_with('[') && t.ends_with(']') {
                section = t.to_string();
                in_dev = section == "[dev-dependencies]";
                continue;
            }
            if !t.starts_with(crate_name) {
                continue;
            }
            if in_dev {
                dev_names_it = true;
            } else {
                in_deps.push(section.clone());
            }
        }
        assert!(
            dev_names_it,
            "the manifest no longer names {crate_name} at all — this gate is stale"
        );
        assert!(
            in_deps.is_empty(),
            "{crate_name} became a real dependency of the shipped player, in {in_deps:?}"
        );
    }
}
