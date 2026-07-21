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

/// One remembered project.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecentProject {
    pub name: String,
    /// Absolute project root (forward-slashed for stable JSON).
    pub path: String,
}

/// The persisted list, most-recent first.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecentProjects {
    pub entries: Vec<RecentProject>,
}

impl RecentProjects {
    fn file(dir: &Path) -> PathBuf {
        dir.join(RECENT_FILE)
    }

    /// Load from `dir` (missing/corrupt → empty).
    pub fn load(dir: &Path) -> Self {
        match std::fs::read_to_string(Self::file(dir)) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Persist to `dir`.
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::error::ProjectError::Other(format!("recent json: {e}")))?;
        std::fs::write(Self::file(dir), json)?;
        Ok(())
    }

    /// Move `project` to the front (dedup by path), cap the list, and persist.
    pub fn push(dir: &Path, project: &Project) -> Result<Self> {
        let mut list = Self::load(dir);
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
        let reloaded = RecentProjects::load(cfg.path());
        assert_eq!(reloaded.entries[0].name, "A");
    }

    #[test]
    fn prune_drops_missing_roots() {
        let mut list = RecentProjects {
            entries: vec![RecentProject {
                name: "Gone".into(),
                path: "/nope/does/not/exist".into(),
            }],
        };
        list.prune_missing();
        assert!(list.entries.is_empty());
    }
}
