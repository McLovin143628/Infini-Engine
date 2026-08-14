//! Named dock-layout persistence (ROADMAP P1.2.5).
//!
//! Ring 1, Tauri-free: the storage directory is injected (the studio app
//! passes its per-app config dir), so everything here is headless-testable.
//! Layout documents are opaque JSON strings owned by the frontend dock
//! store — this layer only guarantees safe names, atomic writes, and
//! stable listing.

use std::fs;
use std::path::{Path, PathBuf};
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
    ///
    /// # The delete-then-rename that made the doc above false
    ///
    /// This used to remove the destination before renaming onto it (C4-23),
    /// justified by "Windows rename fails if the target exists". That is not
    /// true: `std::fs::rename` asks for `MOVEFILE_REPLACE_EXISTING`, and five
    /// other writers in this repo — the pack, the vmesh, the sidecar, the
    /// `.inf_voxel` and the `.inf_terrain` — have relied on rename-over-existing
    /// on Windows since P16. The cost of the belief was real: between the delete
    /// and the rename the preset did **not exist**, and a failed rename lost it
    /// permanently while reporting "rename to …".
    pub fn save(&self, name: &str, json: &str) -> Result<(), String> {
        let path = self.path_for(name)?;
        inf_asset::write_atomically(&path, json)
            .map_err(|e| format!("write {}: {e}", path.display()))
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

    /// The doc's standing promise — "a concurrent reader never observes a torn
    /// write" — asserted, plus no temp litter for `list` to filter.
    ///
    /// **This arm does not falsify C4-23 and is not claimed to.** Measured: the
    /// delete-then-rename shape still passes it, because the gap it opens is
    /// between two calls *inside* `save` and every observation here happens
    /// after `save` returns (and on Windows a handle opened with
    /// `FILE_SHARE_DELETE` survives the unlink besides). The falsifier for the
    /// deleted lines is
    /// `saving_a_preset_never_deletes_its_destination_first` below, with
    /// `inf_asset::atomic::rename_over_an_existing_file_replaces_it` supplying
    /// the premise those lines doubted.
    #[test]
    fn overwriting_a_preset_never_makes_it_disappear() {
        use std::io::Read;
        let (_guard, store) = store();
        store.save("Default", r#"{"v":1}"#).unwrap();
        let path = store.dir().join("Default.json");

        let mut live = fs::File::open(&path).unwrap();
        store.save("Default", r#"{"v":2}"#).unwrap();
        assert!(path.exists(), "the preset vanished during the overwrite");

        let mut seen = String::new();
        live.read_to_string(&mut seen).unwrap();
        assert_eq!(
            seen, r#"{"v":1}"#,
            "a concurrent reader observed the rewrite"
        );
        assert_eq!(fs::read_to_string(&path).unwrap(), r#"{"v":2}"#);

        // …and no temp litter, which `list` would otherwise have to filter.
        let strays: Vec<_> = fs::read_dir(store.dir())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(strays.is_empty(), "left temp files behind: {strays:?}");
    }

    /// **`save` never removes its destination** (C4-23) — the falsifier.
    ///
    /// A source gate, because the state being ruled out exists only *between*
    /// two syscalls inside `save` and no observation from outside it is
    /// deterministic. The property is exactly a source property: there must be
    /// no delete on the write path at all.
    ///
    /// `.rs` is `text eol=lf` in `.gitattributes`, so this reads the same on
    /// every checkout (the P22.4 lesson).
    #[test]
    fn saving_a_preset_never_deletes_its_destination_first() {
        const SRC: &str = include_str!("layouts.rs");
        let start = SRC
            .find("pub fn save(&self, name: &str, json: &str)")
            .expect("LayoutStore::save moved or was renamed");
        let body = &SRC[start..];
        let end = body
            .find("\n    }\n")
            .expect("save's body has no closing brace at its indent");
        let body = &body[..end];

        assert!(
            body.contains("write_atomically"),
            "save no longer goes through the one atomic-write door"
        );
        assert!(
            !body.contains("remove_file"),
            "save deletes its destination again — between that call and the \
             rename the preset does not exist, and a failed rename loses it"
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
