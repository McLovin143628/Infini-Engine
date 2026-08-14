//! The recent-projects list (editor start screen / File → Open Recent).
//!
//! Persisted as JSON in the editor's config dir; most-recent first, deduped by
//! path, capped. Stored as plain data (no engine deps) so the CLI and editor
//! share it.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::project::Project;

/// The recent list file name.
pub const RECENT_FILE: &str = "recent-projects.json";

/// Max entries retained.
pub const MAX_RECENT: usize = 12;

/// The recent list's own format version.
///
/// **A JSON editor config, not a wire format** — nothing decodes it
/// positionally, no `.inf_*` asset carries it and no pack ships it, so this
/// number is free to move and does not touch the schema ladder. It exists for
/// the reason every sibling config in the repo has one: so a file written by a
/// future build is *refused* rather than silently reinterpreted (F14).
pub const RECENT_SCHEMA_VERSION: u32 = 1;

fn default_schema_version() -> u32 {
    RECENT_SCHEMA_VERSION
}

/// One remembered project.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecentProject {
    pub name: String,
    /// Absolute project root (forward-slashed for stable JSON).
    pub path: String,
}

/// The persisted list, most-recent first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentProjects {
    /// See [`RECENT_SCHEMA_VERSION`]. Defaulted on read so every list written
    /// before the field existed still loads — they are all v1 by definition.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub entries: Vec<RecentProject>,
}

impl Default for RecentProjects {
    fn default() -> Self {
        Self {
            schema_version: RECENT_SCHEMA_VERSION,
            entries: Vec::new(),
        }
    }
}

impl RecentProjects {
    fn file(dir: &Path) -> PathBuf {
        dir.join(RECENT_FILE)
    }

    /// Load from `dir`.
    ///
    /// **Absent is empty; unreadable is an error** (C4-38/F14). This was
    /// `unwrap_or_default()`, so a truncated or hand-mangled list silently
    /// became an empty one — and `push` writes it straight back, which is how a
    /// single interrupted write erased every remembered project permanently. The
    /// idiom is `inf_audio::MixerConfig::load_or_default`'s.
    pub fn load_or_default(dir: &Path) -> Result<Self> {
        let path = Self::file(dir);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(e.into()),
        };
        let list: Self = serde_json::from_str(&text).map_err(|e| {
            crate::error::ProjectError::Other(format!(
                "the recent-projects list {} exists but cannot be read ({e}); it is left \
                 untouched rather than overwritten — repair or delete it",
                path.display()
            ))
        })?;
        if list.schema_version > RECENT_SCHEMA_VERSION {
            return Err(crate::error::ProjectError::Other(format!(
                "the recent-projects list {} was written by a newer build (v{} > v{}); it is \
                 left untouched",
                path.display(),
                list.schema_version,
                RECENT_SCHEMA_VERSION
            )));
        }
        Ok(list)
    }

    /// Persist to `dir`, **atomically** — a torn write silently empties the list
    /// (C4-45).
    pub fn save(&self, dir: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::error::ProjectError::Other(format!("recent json: {e}")))?;
        inf_asset::write_atomically(&Self::file(dir), json)?;
        Ok(())
    }

    /// Move `project` to the front (dedup by path), cap the list, and persist.
    pub fn push(dir: &Path, project: &Project) -> Result<Self> {
        let mut list = Self::load_or_default(dir)?;
        let path = project.root.to_string_lossy().replace('\\', "/");
        list.entries.retain(|e| e.path != path);
        list.entries.insert(
            0,
            RecentProject {
                name: project.name().to_string(),
                path,
            },
        );
        list.entries.truncate(MAX_RECENT);
        list.save(dir)?;
        Ok(list)
    }

    /// Drop entries whose path no longer exists on disk (stale roots).
    pub fn prune_missing(&mut self) {
        self.entries
            .retain(|e| Path::new(&e.path).join("inf.toml").exists());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::ProjectTemplate;

    #[test]
    fn push_dedupes_and_orders_most_recent_first() {
        let cfg = tempfile::tempdir().unwrap();
        let workspace = tempfile::tempdir().unwrap();
        let a = Project::create(workspace.path(), "A", ProjectTemplate::Blank3d).unwrap();
        let b = Project::create(workspace.path(), "B", ProjectTemplate::Blank3d).unwrap();

        RecentProjects::push(cfg.path(), &a).unwrap();
        RecentProjects::push(cfg.path(), &b).unwrap();
        let list = RecentProjects::push(cfg.path(), &a).unwrap(); // A again → front

        assert_eq!(list.entries.len(), 2, "deduped");
        assert_eq!(list.entries[0].name, "A");
        assert_eq!(list.entries[1].name, "B");

        // Reload from disk.
        let reloaded = RecentProjects::load_or_default(cfg.path()).unwrap();
        assert_eq!(reloaded.entries[0].name, "A");
    }

    #[test]
    fn prune_drops_missing_roots() {
        let mut list = RecentProjects {
            entries: vec![RecentProject {
                name: "Gone".into(),
                path: "/nope/does/not/exist".into(),
            }],
            ..Default::default()
        };
        list.prune_missing();
        assert!(list.entries.is_empty());
    }

    /// **A corrupt list is refused; an absent one is empty** (C4-38/F14) — and
    /// the corrupt file is left alone, because `push` writing an empty list back
    /// over it is what erased every remembered project permanently.
    #[test]
    fn a_corrupt_list_is_refused_while_an_absent_one_is_empty() {
        let absent = tempfile::tempdir().unwrap();
        assert!(RecentProjects::load_or_default(absent.path())
            .unwrap()
            .entries
            .is_empty());

        let cfg = tempfile::tempdir().unwrap();
        let path = cfg.path().join(RECENT_FILE);
        let damaged = b"{\"entries\": [{\"name\": \"A\",".to_vec(); // truncated mid-write
        std::fs::write(&path, &damaged).unwrap();
        let err = RecentProjects::load_or_default(cfg.path())
            .expect_err("a damaged list must not read as an empty one");
        assert!(err.to_string().contains("cannot be read"), "{err}");
        assert_eq!(std::fs::read(&path).unwrap(), damaged);
    }

    /// A list written **before** the version field existed still loads, and one
    /// written by a newer build is refused rather than reinterpreted (F14).
    #[test]
    fn the_version_field_defaults_on_read_and_refuses_the_future() {
        let cfg = tempfile::tempdir().unwrap();
        let path = cfg.path().join(RECENT_FILE);

        std::fs::write(&path, b"{\"entries\": []}").unwrap();
        assert_eq!(
            RecentProjects::load_or_default(cfg.path())
                .expect("a pre-version list still loads")
                .schema_version,
            RECENT_SCHEMA_VERSION
        );

        let future = format!("{{\"schema_version\": {}, \"entries\": []}}", RECENT_SCHEMA_VERSION + 1);
        std::fs::write(&path, future.as_bytes()).unwrap();
        let err = RecentProjects::load_or_default(cfg.path())
            .expect_err("a newer list must be refused, not reinterpreted");
        assert!(err.to_string().contains("newer build"), "{err}");
    }
}
