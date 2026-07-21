//! Diagnostics commands (P15): a memory-budget readout + the editor crash hook.
//!
//! `editor_diagnostics` returns a [`MemoryReport`] the frontend can adopt for a
//! status-bar readout later (backend-only for now — a documented handoff); it
//! also logs a one-line summary so the `stats` console command / Output Log shows
//! it today. [`install_crash_hook`] installs a Rust panic hook that writes a
//! structured, timestamped crash file. There is **no** telemetry upload — opt-in
//! only (a config flag + docs), never silent (see `docs/profiling.md`).

use std::path::PathBuf;
use std::sync::OnceLock;

use inf_editor_core::diagnostics::{write_crash_report, CrashReport, MemoryReport};
use tauri::State;

use super::SceneState;

/// Return a memory-budget report for the active scene document. Renderer/runtime
/// cache fields are left at their defaults (the viewport runs on its own thread;
/// wiring its texture/vgeom/audio byte counters through is the documented Ring-2
/// handoff). Also logs a one-line summary for the Output Log / `stats` command.
#[tauri::command]
pub async fn editor_diagnostics(scene: State<'_, SceneState>) -> Result<MemoryReport, String> {
    let doc = scene
        .doc
        .lock()
        .map_err(|_| "scene lock poisoned".to_string())?;
    let report = MemoryReport::for_scene(&doc);
    tracing::info!("{}", report.summary());
    Ok(report)
}

// ── crash reporter (P15.2) ───────────────────────────────────────────────────

/// Where crash files are written. Filled with the real app-data crash dir once
/// the Tauri app is up (see [`set_crash_dir`]); before that (and if resolution
/// fails) it falls back to a temp dir so a very-early panic still leaves a trail.
static CRASH_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Point the crash reporter at `dir` (called from `setup` with the app-data
/// crash directory). First call wins.
pub fn set_crash_dir(dir: PathBuf) {
    let _ = CRASH_DIR.set(dir);
}

fn crash_dir() -> PathBuf {
    CRASH_DIR
        .get()
        .cloned()
        .unwrap_or_else(|| std::env::temp_dir().join("InfinityEngine").join("crashes"))
}

/// Install the editor's crash-reporting panic hook. Safe to call once; chains to
/// the previous hook so normal panic printing/aborting is preserved.
pub fn install_crash_hook() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    if INSTALLED.set(()).is_err() {
        return;
    }
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let report = CrashReport::from_panic(
            "inf-studio",
            env!("CARGO_PKG_VERSION"),
            info,
            None,
            crate::logging::recent_lines(),
        );
        match write_crash_report(&crash_dir(), &report) {
            Ok(path) => eprintln!("inf-studio crash report written to {}", path.display()),
            Err(e) => eprintln!("inf-studio: failed to write crash report: {e}"),
        }
        default(info);
    }));
}
