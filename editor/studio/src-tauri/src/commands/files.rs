//! Project file surface (P5.4): read/write text files + a gitignore-aware
//! file listing for the explorer, search, and (P5.1) the code editor.
//!
//! Paths are absolute (resolved against the project root on the frontend);
//! `list_project_files` returns paths relative to the walked root.

use std::path::Path;

use ignore::WalkBuilder;
use inf_editor_core::ipc::FileEntryDto;
use tauri::{AppHandle, State};

use super::paths::{confine_existing, confine_for_write};
use super::project::ProjectState;

/// Max text-file size we'll hand to the editor (bytes).
const MAX_FILE: u64 = 5 * 1024 * 1024;
/// Max entries returned by a single listing.
const MAX_ENTRIES: usize = 8000;

/// Read a UTF-8 text file. Errors on binary or over-large files.
///
/// **Confined** (L7.H6): the path must resolve inside the open project or the
/// editor's app-data dir. It took a bare absolute path before, so
/// `file_read("~/.ssh/id_rsa")` was a valid IPC request from the webview.
#[tauri::command]
pub async fn file_read(
    app: AppHandle,
    project: State<'_, ProjectState>,
    path: String,
) -> Result<String, String> {
    let path = confine_existing(&app, &project, &path)?;
    // Reading up to MAX_FILE of a possibly-slow disk is blocking IO — keep it
    // off the async workers (mirrors package.rs).
    tauri::async_runtime::spawn_blocking(move || {
        let shown = path.display().to_string();
        let meta = std::fs::metadata(&path).map_err(|e| format!("stat {shown}: {e}"))?;
        if meta.len() > MAX_FILE {
            return Err(format!("file is larger than {} MB", MAX_FILE / 1024 / 1024));
        }
        let bytes = std::fs::read(&path).map_err(|e| format!("read {shown}: {e}"))?;
        String::from_utf8(bytes).map_err(|_| "not a UTF-8 text file".to_string())
    })
    .await
    .map_err(|e| format!("file_read task failed to run: {e}"))?
}

/// Write a UTF-8 text file (creating parent dirs).
///
/// **Atomic** (C4-13). This is the IDE panel's Ctrl+S, over the user's own
/// `.rs`/`.toml`, and unlike a level there is no schema check downstream to
/// notice the damage: a truncated `.rs` fails to compile with a message about
/// the wrong thing, and a truncated `Cargo.toml` may still parse as
/// valid-but-wrong.
/// **Confined** (L7.H6) through the may-not-exist-yet door: a new file inherits
/// its parent directory's verdict, and a tail containing `..` is refused.
#[tauri::command]
pub async fn file_write(
    app: AppHandle,
    project: State<'_, ProjectState>,
    path: String,
    content: String,
) -> Result<(), String> {
    let path = confine_for_write(&app, &project, &path)?;
    tauri::async_runtime::spawn_blocking(move || {
        let shown = path.display().to_string();
        inf_asset::write_atomically(&path, content).map_err(|e| format!("write {shown}: {e}"))
    })
    .await
    .map_err(|e| format!("file_write task failed to run: {e}"))?
}

/// List the project's files (gitignore-aware; skips `target/`, `.git/`, …),
/// as paths relative to `root`. Capped at [`MAX_ENTRIES`].
#[tauri::command]
pub async fn list_project_files(
    app: AppHandle,
    project: State<'_, ProjectState>,
    root: String,
) -> Result<Vec<FileEntryDto>, String> {
    let root = confine_existing(&app, &project, &root)?;
    // Walking the whole project tree is blocking IO — run it off the async workers.
    tauri::async_runtime::spawn_blocking(move || {
        let root_path: &Path = root.as_ref();
        if !root_path.is_dir() {
            return Err(format!("{} is not a directory", root_path.display()));
        }
        let mut out = Vec::new();
        let walker = WalkBuilder::new(root_path)
            .hidden(false) // show dotfiles; .gitignore still applies
            .git_ignore(true)
            .git_global(false)
            .filter_entry(|e| e.file_name() != ".git")
            .build();

        for dent in walker.flatten() {
            if out.len() >= MAX_ENTRIES {
                tracing::warn!("list_project_files capped at {MAX_ENTRIES} entries");
                break;
            }
            let path = dent.path();
            if path == root_path {
                continue;
            }
            let Ok(rel) = path.strip_prefix(root_path) else {
                continue;
            };
            let is_dir = dent.file_type().map(|t| t.is_dir()).unwrap_or(false);
            out.push(FileEntryDto {
                path: rel.to_string_lossy().replace('\\', "/"),
                name: dent.file_name().to_string_lossy().into_owned(),
                is_dir,
            });
        }
        out.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(out)
    })
    .await
    .map_err(|e| format!("list_project_files task failed to run: {e}"))?
}
