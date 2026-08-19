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

    /// Absolute levels root — **`<root>/<content_dir>/<levels_dir>`**.
    ///
    /// # Levels are content (the island phase's IB-7 ruling)
    ///
    /// Until this ruling `levels_dir` was resolved against the *project* root, so
    /// `inf new` scaffolded a boot level into `<root>/Levels/` and `inf cook` —
    /// which opens `<root>/Content/` and nothing else — refused every one of the
    /// four templates with *"no levels in cook — the build has no boot scene"*.
    /// The first thing anyone did with the engine was a dead end, and CI never
    /// saw it because the cook-and-run smoke hand-writes an `inf.toml` instead of
    /// running `inf new`.
    ///
    /// Two fixes were available: teach the cook a second root, or put levels
    /// where the cook already looks. This is the second, because a level **is**
    /// content — it is an `AssetKind::Level` with a GUID, a sidecar, a content
    /// hash and a dependency closure, indistinguishable in the asset database
    /// from the meshes and materials it references. A parallel root would have
    /// meant two scan paths, two dedupe indices and two answers to
    /// "what does this project contain".
    ///
    /// `AssetDb::scan` already recurses, so `Content/Levels/` needs nothing from
    /// the cook: it is found the day it exists.
    ///
    /// **The one migration.** A project authored before the ruling has real
    /// levels at `<root>/Levels/`, which is now nowhere. It is not silently
    /// ignored — `inf_packager::stranded_levels_advisory` names the files and the
    /// remedy, and a cook that would otherwise be blocked for having no boot
    /// scene says so in the same breath.
    pub fn levels_root(&self) -> PathBuf {
        self.content_root().join(&self.manifest.levels_dir)
    }

    /// The **pre-ruling** levels directory (`<root>/<levels_dir>`) — where a
    /// project scaffolded before IB-7 put its boot level.
    ///
    /// Nothing writes here. It exists so the cook can look, find stranded
    /// content and say so; a migration that cannot see the old location can only
    /// report an absence.
    pub fn legacy_levels_root(&self) -> PathBuf {
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
        // IB-7: the levels root is INSIDE the content root, which is the only
        // directory the cook opens.
        assert!(
            p.levels_root().starts_with(p.content_root()),
            "levels must live under Content/ or `inf cook` cannot see them"
        );
        assert_eq!(p.levels_root(), p.root.join("Content").join("Levels"));
        // …and the pre-ruling location is left alone rather than created.
        assert_eq!(p.legacy_levels_root(), p.root.join("Levels"));
        assert!(!p.legacy_levels_root().exists());

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
