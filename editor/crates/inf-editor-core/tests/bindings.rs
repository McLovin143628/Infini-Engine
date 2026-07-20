//! ts-rs export harness (ROADMAP P0.1.5).
//!
//! Running this test (re)generates the committed TypeScript bindings under
//! `editor/studio/src/bindings/`. CI's bindings-drift job reruns it and fails
//! on `git diff`, so the committed bindings can never lag the Rust types.
//! Every new type in `inf_editor_core::ipc` must be added to `ROOTS` here.

use std::path::Path;

use inf_editor_core::ipc::{LayoutSummary, LogLine, ViewportDrop, ViewportRect};
use ts_rs::{Config, TS};

#[test]
fn export_bindings() {
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../studio/src/bindings");
    let cfg = Config::new().with_out_dir(out);

    // export_all also exports each root's transitive dependencies.
    ViewportRect::export_all(&cfg).expect("export ViewportRect");
    ViewportDrop::export_all(&cfg).expect("export ViewportDrop");
    LogLine::export_all(&cfg).expect("export LogLine");
    LayoutSummary::export_all(&cfg).expect("export LayoutSummary");
}
