//! Editor startup budget: headless project-open (asset scan) time (P15.1 / §8).
//!
//! Copies a committed sample project's content into a temp dir and times a full
//! asset-database scan (the dominant cost of opening a project), asserting it is
//! under a generous bound. A regression tripwire, not a perf claim — the budget
//! only ratchets **down** per §8.

use std::path::{Path, PathBuf};

use inf_asset::AssetDb;

/// Hard budget for a headless project-open scan, in milliseconds. **RATCHET RULE
/// (§8): only ever DECREASE this.**
const OPEN_BUDGET_MS: f64 = 5000.0;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

/// Copy every file in `src` (flat) into `dst`.
fn copy_flat(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            std::fs::copy(entry.path(), dst.join(entry.file_name())).unwrap();
        }
    }
}

#[test]
fn project_open_scan_under_budget() {
    // The richest committed sample (skeleton + anims + state machine + level).
    let sample = workspace_root().join("samples").join("character-demo");
    if !sample.is_dir() {
        eprintln!("SKIP startup_budget: sample project not found at {sample:?}");
        return;
    }

    // Copy into a temp content dir so the scan (which may synthesize sidecars)
    // never touches the committed samples.
    let tmp = tempfile::tempdir().unwrap();
    let content = tmp.path().join("Content");
    copy_flat(&sample, &content);

    let start = std::time::Instant::now();
    let mut db = AssetDb::new(&content);
    let count = db.scan().expect("asset scan succeeds");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    assert!(count > 0, "the sample project has scannable assets");
    eprintln!("startup: scanned {count} assets in {elapsed_ms:.1} ms (budget {OPEN_BUDGET_MS} ms)");
    assert!(
        elapsed_ms < OPEN_BUDGET_MS,
        "project-open scan {elapsed_ms:.1} ms exceeded the {OPEN_BUDGET_MS} ms budget \
         (the §8 budget only ratchets DOWN — investigate the regression, do not raise it)"
    );
}
