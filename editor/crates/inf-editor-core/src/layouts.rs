//! Named dock-layout persistence (ROADMAP P1.2.5).
//!
//! Ring 1, Tauri-free: the storage directory is injected (the studio app
//! passes its per-app config dir), so everything here is headless-testable.
//! Layout documents are opaque JSON strings owned by the frontend dock
//! store — this layer only guarantees safe names, atomic writes, and
//! stable listing.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::UNIX_EPOCH;

use crate::ipc::LayoutSummary;

const LAYOUT_EXT: &str = "json";

/// File-backed store of named layout presets, one `<name>.json` per layout.
#[derive(Debug, Clone)]
pub struct LayoutStore {
    dir: PathBuf,
}

impl LayoutStore {
    /// A store rooted at `dir` (created lazily on first save).
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Persist `json` under `name`, atomically (temp file + rename — a
    /// concurrent reader never observes a torn write; see the Spike C
    /// shadow-copy race for why the temp name must be unique per writer).
    pub fn save(&self, name: &str, json: &str) -> Result<(), String> {
        let path = self.path_for(name)?;
        fs::create_dir_all(&self.dir)
            .map_err(|e| format!("create layout dir {}: {e}", self.dir.display()))?;
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let tmp = self.dir.join(format!(
            ".{name}.{}-{}.tmp",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&tmp, json).map_err(|e| format!("write {}: {e}", tmp.display()))?;
        // Windows rename fails if the target exists — remove first. The
        // brief gap is fine: layouts are single-user, last-writer-wins.
        if path.exists() {
            fs::remove_file(&path).map_err(|e| format!("replace {}: {e}", path.display()))?;
        }
        fs::rename(&tmp, &path).map_err(|e| format!("rename to {}: {e}", path.display()))?;
        Ok(())
    }

    /// Load a layout's JSON. `Ok(None)` when the preset doesn't exist.
    pub fn load(&self, name: &str) -> Result<Option<String>, String> {
        let path = self.path_for(name)?;
        match fs::read_to_string(&path) {
            Ok(json) => Ok(Some(json)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(format!("read {}: {e}", path.display())),
        }
    }

    /// All saved presets, sorted by name.
    pub fn list(&self) -> Result<Vec<LayoutSummary>, String> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(format!("read layout dir {}: {e}", self.dir.display())),
        };
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| format!("read layout dir entry: {e}"))?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some(LAYOUT_EXT) {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            // Skip temp files and anything that wouldn't round-trip.
            if validate_name(stem).is_err() {
                continue;
            }
            let modified_ms = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as f64)
                .unwrap_or(0.0);
            out.push(LayoutSummary {
                name: stem.to_string(),
                modified_ms,
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    /// Delete a preset. Returns whether it existed.
    pub fn delete(&self, name: &str) -> Result<bool, String> {
        let path = self.path_for(name)?;
        match fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(format!("delete {}: {e}", path.display())),
        }
    }

    fn path_for(&self, name: &str) -> Result<PathBuf, String> {
        validate_name(name)?;
        Ok(self.dir.join(format!("{name}.{LAYOUT_EXT}")))
    }
}

/// Layout names become file names: 1–64 chars of `[A-Za-z0-9 _-]`, no
/// leading/trailing space. Everything else (path separators, dots, unicode
/// tricks) is rejected outright rather than escaped.
pub fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 64 {
        return Err(format!(
            "layout name must be 1-64 characters (got {})",
            name.len()
        ));
    }
    if name.starts_with(' ') || name.ends_with(' ') {
        return Err("layout name cannot start or end with a space".into());
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == ' ' || c == '_' || c == '-')
    {
        return Err(format!("layout name has unsupported characters: {name:?}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (tempfile::TempDir, LayoutStore) {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = LayoutStore::new(dir.path().join("layouts"));
        (dir, store)
    }

    #[test]
    fn save_load_roundtrip_and_overwrite() {
        let (_guard, store) = store();
        store.save("Default", r#"{"v":1}"#).unwrap();
        assert_eq!(
            store.load("Default").unwrap().as_deref(),
            Some(r#"{"v":1}"#)
        );
        store.save("Default", r#"{"v":2}"#).unwrap();
        assert_eq!(
            store.load("Default").unwrap().as_deref(),
            Some(r#"{"v":2}"#)
        );
    }

    #[test]
    fn load_missing_is_none_and_list_empty_dir_is_empty() {
        let (_guard, store) = store();
        assert_eq!(store.load("Nope").unwrap(), None);
        assert!(store.list().unwrap().is_empty());
    }

    #[test]
    fn list_returns_sorted_names_and_skips_foreign_files() {
        let (_guard, store) = store();
        store.save("b layout", "{}").unwrap();
        store.save("A-Layout", "{}").unwrap();
        fs::write(store.dir().join("notes.txt"), "x").unwrap();
        fs::write(store.dir().join(".hidden.json"), "x").unwrap();
        let names: Vec<_> = store.list().unwrap().into_iter().map(|l| l.name).collect();
        assert_eq!(names, vec!["A-Layout", "b layout"]);
    }

    #[test]
    fn delete_reports_existence() {
        let (_guard, store) = store();
        store.save("Gone", "{}").unwrap();
        assert!(store.delete("Gone").unwrap());
        assert!(!store.delete("Gone").unwrap());
    }

    #[test]
    fn rejects_hostile_names() {
        let (_guard, store) = store();
        for bad in [
            "",
            "../evil",
            "a/b",
            "a\\b",
            "con.",
            "name\u{202e}",
            &"x".repeat(65),
            " pad",
        ] {
            assert!(store.save(bad, "{}").is_err(), "accepted {bad:?}");
        }
    }
}
