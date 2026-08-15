//! Native "Reveal in Explorer" (editor seams).
//!
//! The single OS shell-out the editor performs: reveal a file or folder in the
//! platform file browser (Explorer / Finder / the default file manager). Every
//! request is validated — the (canonicalized) path must exist AND live under an
//! allowed root (the open project root, or the app-data dir where loose Content
//! and quicksaves live) — so a compromised frontend can't shell-open arbitrary
//! locations. The spawn is fire-and-forget (Explorer returns non-zero even on
//! success), wrapped in `spawn_blocking` so it never blocks the async runtime.

use std::path::{Path, PathBuf};

use tauri::{AppHandle, State};

use super::paths::confine_existing;
use super::project::ProjectState;

/// Reveal `path` in the OS file browser — selecting the entry when it is a file,
/// opening the folder when it is a directory. Rejects (with a clear message) any
/// path that doesn't exist or lands outside the open project / app-data roots.
#[tauri::command]
pub async fn shell_reveal(
    app: AppHandle,
    project: State<'_, ProjectState>,
    path: String,
) -> Result<(), String> {
    // The guard this function used to spell out inline. It is now
    // `super::paths::confine_existing`, applied at every filesystem door in the
    // IPC surface rather than at this one (L7.H6).
    let target = confine_existing(&app, &project, &path)?;

    let is_dir = target.is_dir();
    tauri::async_runtime::spawn_blocking(move || reveal_in_os(&target, is_dir))
        .await
        .map_err(|e| e.to_string())?
}

/// Strip the Windows `\\?\` verbatim prefix that `canonicalize` adds — Explorer
/// doesn't understand extended-length paths. A no-op off Windows / for
/// non-verbatim paths.
#[cfg(target_os = "windows")]
fn display_path(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        // Leave UNC verbatim (`\\?\UNC\…`) alone; only plain drive paths convert.
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => path.to_path_buf(),
    }
}

/// Spawn the platform file-browser reveal. Fire-and-forget: we `spawn` (not
/// `output`) because Explorer/xdg-open often exit non-zero even on success —
/// only a failure to *launch* is reported.
fn reveal_in_os(path: &Path, is_dir: bool) -> Result<(), String> {
    use std::process::Command;

    #[cfg(target_os = "windows")]
    {
        let path = display_path(path);
        let mut cmd = Command::new("explorer");
        if is_dir {
            cmd.arg(&path);
        } else {
            // `/select,<file>` opens the parent and highlights the file. It must
            // be one combined argument.
            cmd.arg(format!("/select,{}", path.display()));
        }
        cmd.spawn().map_err(|e| format!("explorer failed: {e}"))?;
    }

    #[cfg(target_os = "macos")]
    {
        let mut cmd = Command::new("open");
        if is_dir {
            cmd.arg(path);
        } else {
            cmd.arg("-R").arg(path); // reveal + select in Finder
        }
        cmd.spawn().map_err(|e| format!("open failed: {e}"))?;
    }

    #[cfg(target_os = "linux")]
    {
        // No portable "select the file" verb; open the containing folder.
        let dir = if is_dir {
            path.to_path_buf()
        } else {
            path.parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| path.to_path_buf())
        };
        Command::new("xdg-open")
            .arg(dir)
            .spawn()
            .map_err(|e| format!("xdg-open failed: {e}"))?;
    }

    Ok(())
}
