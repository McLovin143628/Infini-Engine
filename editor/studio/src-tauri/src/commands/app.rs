//! App-level commands: version info, lifecycle.

/// Returns the editor version string shown in the About dialog / status bar.
#[tauri::command]
pub async fn app_version() -> Result<String, String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}
