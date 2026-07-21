//! An opened project: its root + manifest, and the derived content/levels roots.

use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::manifest::{ProjectManifest, PROJECT_FILE};
use crate::template::{scaffold, ProjectTemplate};

/// A loaded project.
#[derive(Debug, Clone)]
pub struct Project {
    pub root: PathBuf,
    pub manifest: ProjectManifest,
}

impl Project {
    /// Open the project rooted at `root` (must contain `inf.toml`). Ensures the
    /// content + levels directories exist.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let manifest = ProjectManifest::load(&root)?;
        let project = Self { root, manifest };
        std::fs::create_dir_all(project.content_root())?;
        std::fs::create_dir_all(project.levels_root())?;
        Ok(project)
    }

    /// Scaffold a new project under `parent` and open it.
    pub fn create(parent: &Path, name: &str, template: ProjectTemplate) -> Result<Self> {
        let s = scaffold(parent, name, template)?;
        Self::open(s.root)
    }

    /// Walk up from `start` looking for the nearest `inf.toml`; returns its root.
    pub fn find(start: &Path) -> Option<PathBuf> {
        let mut dir = Some(start);
        while let Some(d) = dir {
            if d.join(PROJECT_FILE).is_file() {
                return Some(d.to_path_buf());
            }
            dir = d.parent();
        }
        None
    }

    pub fn name(&self) -> &str {
        &self.manifest.name
    }

    /// Absolute content root (`<root>/<content_dir>`).
    pub fn content_root(&self) -> PathBuf {
        self.root.join(&self.manifest.content_dir)
    }

    /// Absolute levels root (`<root>/<levels_dir>`).
    pub fn levels_root(&self) -> PathBuf {
        self.root.join(&self.manifest.levels_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_then_open_and_find() {
        let parent = tempfile::tempdir().unwrap();
        let p = Project::create(parent.path(), "Game A", ProjectTemplate::Blank3d).unwrap();
        assert_eq!(p.name(), "Game A");
        assert!(p.content_root().is_dir());
        assert!(p.levels_root().is_dir());

        // Re-open by root.
        let reopened = Project::open(&p.root).unwrap();
        assert_eq!(reopened.manifest, p.manifest);

        // find() from a nested dir walks up to the root.
        let nested = p.content_root();
        assert_eq!(Project::find(&nested), Some(p.root.clone()));
    }

    #[test]
    fn find_returns_none_outside_a_project() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(Project::find(dir.path()), None);
    }
}
