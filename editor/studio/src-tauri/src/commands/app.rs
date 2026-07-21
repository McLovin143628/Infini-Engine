//! App-level commands: version info, lifecycle.

/// Returns the editor version string shown in the About dialog / status bar.
#[tauri::command]
pub async fn app_version() -> Result<String, String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}

/// A multi-line build banner: engine version + git short hash embedded at build
/// time (see `build.rs`). Surfaced by the Help ▸ About dialog. Returns a plain
/// string (no new ts-rs binding needed).
#[tauri::command]
pub async fn app_build_info() -> Result<String, String> {
    let version = env!("CARGO_PKG_VERSION");
    let git = option_env!("INF_GIT_HASH").unwrap_or("unknown");
    Ok(format!(
        "Infinity Engine {version}\ncommit {git}\nnative Rust core · Tauri v2 editor"
    ))
}
