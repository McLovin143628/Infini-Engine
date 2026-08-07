//! **The fixture builds once, deterministically** (P24.3).
//!
//! # The defect
//!
//! `loading_is_content_addressed` failed macOS CI twice with
//! `copy fixture dylib … No such file or directory`. Two earlier fixes narrowed
//! the window and neither closed it:
//!
//! * **`d01c3e8`** gave each test process a private stash to load the dylib
//!   *from*, so a later rebuild cannot tear a file the process already has open.
//!   That protects the load; it does nothing for the copy that has not happened.
//! * **`f2828e3`** gave the fixtures a dedicated `target/hotreload-fixtures`, so
//!   the test binaries' own (differently feature-unified) builds stop forcing a
//!   relink. Its commit message says the in-process `OnceLock` "never helped: the
//!   contention is across processes" — and then leaves the cross-process
//!   contention to cargo's build lock.
//!
//! **What is left, precisely.** nextest runs every test in its own process, so
//! the three tests in `reload.rs` are three processes each running the whole of
//! `fixture_dylibs()`. Cargo's build-directory lock serializes their *builds* and
//! is **released when `cargo build` exits** — before `fs::copy` runs. Cargo
//! publishes the artifact by "uplifting" it out of `deps/`, and uplift is
//! `remove_file(dst)` then `hard_link(src, dst)`: not atomic, and performed by a
//! cargo the copying process is no longer synchronised with. Process A can
//! therefore be mid-`fs::copy` of `debug/libfoo.dylib` at the moment process B's
//! cargo unlinks it. `ENOENT`.
//!
//! # The fix, and why it is CLOSED rather than narrowed
//!
//! `.config/nextest.toml` puts those tests in a **test group with
//! `max-threads = 1`**. At most one member of the group runs at a time, and the
//! group's members are the only writers of that directory — so a copy can never
//! overlap a relink, because there is never a second process to relink. That is a
//! removal of the concurrency, not a widening of the window: no sleep, no retry,
//! no probability left in it.
//!
//! Both runners are covered, by two different mechanisms and deliberately so:
//!
//! * under **nextest** (what CI runs) the group serializes the *processes*;
//! * under plain **`cargo test`** the three tests are *threads in one process*
//!   and `reload.rs`'s `OnceLock` already collapses them to a single build.
//!
//! # Why this file exists
//!
//! A config file is invisible to every test in the repository, so a new
//! fixture-building test would be added *outside* the group and would flake
//! exactly as `reload.rs` did — the config would be right and the coverage
//! silently wrong. The gate below reads the config **and the workspace**, and
//! fails when a test file that shells out to `cargo build` is not in a group.
//! That is the forcing function; the config alone is only a fix.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The workspace root, reachable from this crate at `crates/inf-hotreload`.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("workspace root")
}

fn config() -> String {
    // Normalized: `.gitattributes` pins `*.toml`-adjacent sources to LF, but a
    // reader that assumed it would be the third time this repository was bitten
    // by reading a checked-out file byte-for-byte on Windows.
    std::fs::read_to_string(workspace_root().join(".config/nextest.toml"))
        .expect("`.config/nextest.toml` is present — the fixture race fix lives there")
        .replace("\r\n", "\n")
}

/// The two groups exist, are **serialized**, and their filters name the binaries
/// that actually build a fixture.
///
/// Asserted on the file's text rather than by running nextest, because a test
/// that shells out to `cargo nextest show-config` cannot run on a machine with no
/// nextest installed — which is every `cargo test` invocation in
/// `docs/CONTRIBUTING.md`.
#[test]
fn the_fixture_groups_are_serialized_and_aimed_at_the_right_binaries() {
    let cfg = config();
    for group in ["hotreload-fixtures", "wasm-mod-fixtures"] {
        let decl = format!("{group} = {{ max-threads = 1 }}");
        assert!(
            cfg.contains(&decl),
            "`{group}` is not declared with `max-threads = 1`. Anything above 1 \
             lets two processes into the same build directory again, which is the \
             whole of the race: `{decl}` not found in\n{cfg}"
        );
        assert!(
            cfg.contains(&format!("test-group = '{group}'")),
            "`{group}` is declared and nothing is assigned to it — a group with no \
             members serializes nothing"
        );
    }
    // The filters, quoted, so a rename of either binary fails here instead of
    // silently emptying its group.
    for filter in [
        "filter = 'package(inf-hotreload) and binary(reload)'",
        "filter = 'binary(spinner_e2e) or binary(mods_e2e)'",
    ] {
        assert!(cfg.contains(filter), "missing override: {filter}");
    }
}

/// **The forcing function.** Every integration test in the workspace that builds
/// a cargo fixture must be inside a test group.
///
/// Without this, the config is a fix for the three tests that were flaking and an
/// invitation for the fourth. The scan is deliberately over *source text*: a test
/// that spawns `cargo build` names the subcommand as a string literal, and one
/// that goes through `inf_packager::mods::build_mod_wasm` names that. Both
/// spellings produce an artifact somebody then copies, which is the shape that
/// races.
///
/// A new fixture-building test fails this by name, with the remedy in the
/// message. Adding it to a group is the fix; adding it to `KNOWN` is not an
/// option, because there is no `KNOWN` — the P23 ruling that a ban list
/// enumerates what you thought of.
#[test]
fn every_fixture_building_test_binary_is_in_a_group() {
    let root = workspace_root();
    let cfg = config();
    let mut offenders: Vec<String> = Vec::new();
    let mut scanned = 0usize;
    let mut found: BTreeSet<String> = BTreeSet::new();

    for dir in ["crates", "editor/crates", "runtime", "tools"] {
        let base = root.join(dir);
        if !base.exists() {
            continue;
        }
        for krate in std::fs::read_dir(&base).expect("read crate dir").flatten() {
            let tests = krate.path().join("tests");
            if !tests.is_dir() {
                continue;
            }
            for entry in std::fs::read_dir(&tests).expect("read tests dir").flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                // **The detector cannot detect itself.** This file quotes every
                // spelling it looks for, so it matches its own scan — and it
                // builds nothing. Excluded by identity (`file!()`), not by name
                // in a list: an exception list is the ban this gate replaced.
                if p.file_name().and_then(|s| s.to_str())
                    == Path::new(file!()).file_name().and_then(|s| s.to_str())
                {
                    continue;
                }
                scanned += 1;
                let src = match std::fs::read_to_string(&p) {
                    Ok(s) => s.replace("\r\n", "\n"),
                    Err(_) => continue,
                };
                // Two spellings of "this test produces a cargo artifact":
                // spawning the subcommand, or going through the packager's door.
                let builds = (src.contains("Command::new(cargo)")
                    || src.contains("Command::new(\"cargo\")")
                    || src.contains("Command::new(env!(\"CARGO\"))"))
                    && src.contains("\"build\"")
                    || src.contains("build_mod_wasm(");
                if !builds {
                    continue;
                }
                let stem = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                found.insert(stem.clone());
                if !cfg.contains(&format!("binary({stem})")) {
                    offenders.push(format!("{}::{stem}", dir));
                }
            }
        }
    }

    // Anti-vacuity FIRST: a scan that found nothing would pass this test while
    // proving nothing at all, which is the exact shape of a check that hides a
    // real intrusion (the P19 lesson).
    assert!(
        scanned > 20,
        "the scan walked only {scanned} test files — it is looking in the wrong \
         place and its verdict is meaningless"
    );
    assert!(
        found.contains("reload"),
        "the scan did not identify `reload.rs` as a fixture-building test, and \
         that is the file the whole race was measured on — the detection is \
         broken, not the config. Found: {found:?}"
    );
    assert!(
        found.len() >= 2,
        "only one fixture-building test found ({found:?}); the spinner pair is \
         built the same way and the scan should see it"
    );

    assert!(
        offenders.is_empty(),
        "these test binaries build a cargo fixture and are NOT in a nextest test \
         group: {offenders:?}\n\nUnder nextest each of their tests is its own \
         process, so N of them run one `cargo build` each against a shared build \
         directory. Cargo's lock serializes the builds and is released BEFORE the \
         artifact is copied, so one process can unlink-and-relink the uplifted \
         file another is mid-copy of (`No such file or directory`). Add the \
         binary to an existing group in `.config/nextest.toml`, or give it a new \
         one with `max-threads = 1`."
    );
}
