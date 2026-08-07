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

/// **The forcing function.** Every test binary that can cause a cargo build into
/// the shared target directory must be inside a test group.
///
/// # It was an enumeration, which is the failure mode it exists to prevent (F14)
///
/// The first cut listed five literal spellings of "spawns cargo" and searched
/// `tests/*.rs` for them. That is the ban-list shape the P24.1 allowlist lesson
/// is about, one level up: it can only see hazards somebody thought to spell, and
/// it missed a real one. `inf-packager`'s `export_bundle` spawns nothing itself —
/// it calls `inf_packager::export`, declared in `bundle.rs`, which
/// release-builds `inf-player` into the shared `target/release/` and then reads
/// the artifact. Byte for byte the hotreload race, in a crate whose tests never
/// mention `Command::new`.
///
/// So the classification is by **call site**, in two passes:
///
///  1. scan every crate's `src/` and `tests/` for a cargo spawn, and collect the
///     `pub fn`s declared in the files that have one — those are the doors
///     through which a test can reach a build;
///  2. a test binary is a hazard if it spawns directly **or** calls one of those
///     doors.
///
/// Still source text, and still bounded: it follows one hop, not a call graph, so
/// a test reaching a builder through two layers of indirection would be missed.
/// That bound is stated rather than papered over — closing it properly needs a
/// real call graph, which is a different tool. One hop is what found the miss.
#[test]
fn every_fixture_building_test_binary_is_in_a_group() {
    let root = workspace_root();
    let cfg = config();
    let mut offenders: Vec<String> = Vec::new();
    let mut scanned = 0usize;
    let mut found: BTreeSet<String> = BTreeSet::new();

    // Pass 1: the doors, each QUALIFIED BY ITS CRATE.
    //
    // An unqualified name is useless here: the doors include `export`, `render`
    // and `build`, and matching those as bare calls flagged fifteen unrelated
    // test binaries (measured). A door is only reachable from another crate as
    // `inf_packager::export(...)` or through a `use inf_packager::…` import, so
    // the crate is part of the key.
    let mut doors: BTreeSet<(String, String)> = BTreeSet::new();
    let mut spawning_files = 0usize;
    for dir in ["crates", "editor/crates", "runtime", "tools"] {
        let base = root.join(dir);
        if !base.exists() {
            continue;
        }
        for krate in std::fs::read_dir(&base).expect("read crate dir").flatten() {
            let krate_ident = krate
                .file_name()
                .to_string_lossy()
                .replace('-', "_")
                .to_string();
            for sub in ["src", "tests"] {
                let d = krate.path().join(sub);
                if !d.is_dir() {
                    continue;
                }
                for entry in walk(&d) {
                    let Ok(src) = std::fs::read_to_string(&entry) else {
                        continue;
                    };
                    let src = src.replace("\r\n", "\n");
                    if !(src.contains("Command::new") && src.contains("\"build\"")) {
                        continue;
                    }
                    spawning_files += 1;
                    for l in src.lines() {
                        // **Module-level `pub fn` only** — no `trim()`. A spawning
                        // file's `#[cfg(test)] mod tests` declares helpers called
                        // `log`, `render` and `is_down`, and treating those as
                        // doors flagged five unrelated `inf-player` binaries
                        // (measured). A door has to be callable from another
                        // crate, which means column zero.
                        if let Some(rest) = l.strip_prefix("pub fn ") {
                            if let Some(name) = rest.split(['(', '<', ' ']).next() {
                                if !name.is_empty() {
                                    doors.insert((krate_ident.clone(), name.to_string()));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        spawning_files >= 3,
        "only {spawning_files} files in the workspace spawn a cargo build — the \
         scan is looking at the wrong place and every verdict below is vacuous"
    );

    // Pass 2: the test binaries.
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
            for p in walk(&tests) {
                if p.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                // The detector cannot detect itself: this file quotes every
                // spelling it looks for and builds nothing. Excluded by
                // `file!()` identity, not by a name in a list.
                if p.file_name().and_then(|s| s.to_str())
                    == Path::new(file!()).file_name().and_then(|s| s.to_str())
                {
                    continue;
                }
                scanned += 1;
                let Ok(src) = std::fs::read_to_string(&p) else {
                    continue;
                };
                let src = src.replace("\r\n", "\n");
                let direct = src.contains("Command::new") && src.contains("\"build\"");
                // Two reachable forms, and both are needed: `export_bundle`
                // calls `export(..)` after `use inf_packager::{export, ..}`, so a
                // qualified-path-only rule misses it — measured. The import form
                // is safe now that doors are module-level `pub fn`s only; with
                // test-module helpers in the set it flagged fifteen binaries on
                // names like `log` and `render`.
                let via_door = doors.iter().any(|(k, d)| {
                    src.contains(&format!("{k}::{d}("))
                        || (src.contains(&format!("use {k}::")) && src.contains(&format!("{d}(")))
                });
                if !(direct || via_door) {
                    continue;
                }
                let stem = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                found.insert(stem.clone());
                if !cfg.contains(&format!("binary({stem})")) {
                    offenders.push(format!("{dir}::{stem}"));
                }
            }
        }
    }

    // Anti-vacuity FIRST.
    assert!(
        scanned > 20,
        "the scan walked only {scanned} test files — it is looking in the wrong \
         place and its verdict is meaningless"
    );
    for must in ["reload", "export_bundle"] {
        assert!(
            found.contains(must),
            "the scan did not identify `{must}` as reaching a cargo build. \
             `reload` spawns directly; `export_bundle` reaches it through \
             `inf_packager::export` and is the miss that made this a call-site \
             classification rather than a list of spellings. Found: {found:?}"
        );
    }

    assert!(
        offenders.is_empty(),
        "these test binaries can cause a cargo build into the shared target \
         directory and are NOT in a nextest test group: {offenders:?}\n\nUnder \
         nextest each of their tests is its own process, so N of them run one \
         build each against a shared directory. Cargo's lock serializes the \
         builds and is released BEFORE the artifact is copied, so one process can \
         unlink-and-relink the uplifted file another is mid-copy of (`No such \
         file or directory`). Add the binary to an existing group in \
         `.config/nextest.toml`, or give it a new one with `max-threads = 1`."
    );
}

/// Every `.rs` file under `dir`, recursively.
fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else if p.extension().and_then(|x| x.to_str()) == Some("rs") {
            out.push(p);
        }
    }
    out
}
