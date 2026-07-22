//! Project file surface (P5.4): read/write text files + a gitignore-aware
//! file listing for the explorer, search, and (P5.1) the code editor.
//!
//! Paths are absolute (resolved against the project root on the frontend);
//! `list_project_files` returns paths relative to the walked root.

use std::path::Path;

use ignore::WalkBuilder;
use inf_editor_core::ipc::FileEntryDto;

/// Max text-file size we'll hand to the editor (bytes).
const MAX_FILE: u64 = 5 * 1024 * 1024;
/// Max entries returned by a single listing.
const MAX_ENTRIES: usize = 8000;

/// Read a UTF-8 text file. Errors on binary or over-large files.
#[tauri::command]
pub async fn file_read(path: String) -> Result<String, String> {
    // Reading up to MAX_FILE of a possibly-slow disk is blocking IO — keep it
    // off the async workers (mirrors package.rs).
    tauri::async_runtime::spawn_blocking(move || {
        let meta = std::fs::metadata(&path).map_err(|e| format!("stat {path}: {e}"))?;
        if meta.len() > MAX_FILE {
            return Err(format!("file is larger than {} MB", MAX_FILE / 1024 / 1024));
        }
        let bytes = std::fs::read(&path).map_err(|e| format!("read {path}: {e}"))?;
        String::from_utf8(bytes).map_err(|_| "not a UTF-8 text file".to_string())
    })
    .await
    .map_err(|e| format!("file_read task failed to run: {e}"))?
}

/// Write a UTF-8 text file (creating parent dirs).
#[tauri::command]
pub async fn file_write(path: String, content: String) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        if let Some(parent) = Path::new(&path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
        }
        std::fs::write(&path, content).map_err(|e| format!("write {path}: {e}"))
    })
    .await
    .map_err(|e| format!("file_write task failed to run: {e}"))?
}

/// List the project's files (gitignore-aware; skips `target/`, `.git/`, …),
/// as paths relative to `root`. Capped at [`MAX_ENTRIES`].
#[tauri::command]
pub async fn list_project_files(root: String) -> Result<Vec<FileEntryDto>, String> {
    // Walking the whole project tree is blocking IO — run it off the async workers.
    tauri::async_runtime::spawn_blocking(move || {
        let root_path = Path::new(&root);
        if !root_path.is_dir() {
            return Err(format!("{root} is not a directory"));
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
