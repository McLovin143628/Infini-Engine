//! The `inf.toml` project manifest.
//!
//! Every Infinity Engine project is rooted at a directory containing an
//! `inf.toml` — the human-readable, git-diffable descriptor the editor, the CLI
//! (`inf new`/`inf cook`), and the runtime all read to locate a project's
//! content, engine version, and template lineage.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{ProjectError, Result};

/// The manifest file name at a project root.
pub const PROJECT_FILE: &str = "inf.toml";

/// Current manifest schema version.
pub const SCHEMA_VERSION: u32 = 1;

/// The `inf.toml` contents. Field order is fixed and `toml` sorts map keys, so
/// re-emission is byte-stable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectManifest {
    pub schema_version: u32,
    /// Display name.
    pub name: String,
    /// The engine version this project was created/last-opened with.
    pub engine_version: String,
    /// The template slug this project was scaffolded from.
    pub template: String,
    /// Content root relative to the project (default `"Content"`).
    #[serde(default = "default_content_dir")]
    pub content_dir: String,
    /// Levels root relative to the project (default `"Levels"`).
    #[serde(default = "default_levels_dir")]
    pub levels_dir: String,
}

fn default_content_dir() -> String {
    "Content".to_string()
}
fn default_levels_dir() -> String {
    "Levels".to_string()
}

impl ProjectManifest {
    /// A fresh manifest for a new project.
    pub fn new(name: impl Into<String>, template: impl Into<String>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            name: name.into(),
            engine_version: env!("CARGO_PKG_VERSION").to_string(),
            template: template.into(),
            content_dir: default_content_dir(),
            levels_dir: default_levels_dir(),
        }
    }

    /// Parse from TOML text, rejecting newer-than-current schemas.
    pub fn from_toml(text: &str) -> Result<Self> {
        let m: ProjectManifest = toml::from_str(text)?;
        if m.schema_version > SCHEMA_VERSION {
            return Err(ProjectError::SchemaTooNew {
                found: m.schema_version,
                current: SCHEMA_VERSION,
            });
        }
        Ok(m)
    }

    /// Serialize to deterministic pretty TOML.
    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Read the manifest at `<root>/inf.toml`.
    ///
    /// The `exists()` pre-check is gone (C4-25): it was a TOCTOU window, and
    /// `NotFound` from the read says the same thing without one.
    pub fn load(root: &Path) -> Result<Self> {
        let path = root.join(PROJECT_FILE);
        match std::fs::read_to_string(&path) {
            Ok(text) => Self::from_toml(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(ProjectError::NoManifest(PROJECT_FILE.to_string()))
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Write the manifest to `<root>/inf.toml`, **atomically**.
    ///
    /// `Project::open` requires this file, so a truncated one makes the project
    /// unopenable (C4-25) — and `RecentProjects::prune_missing` only checks that
    /// it *exists*, so the entry stays in the list and keeps failing. It is a
    /// permanent single point of failure for the whole project.
    pub fn save(&self, root: &Path) -> Result<()> {
        inf_asset::write_atomically(&root.join(PROJECT_FILE), self.to_toml()?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_is_deterministic() {
        let m = ProjectManifest::new("My Game", "blank-3d");
        let t1 = m.to_toml().unwrap();
        assert_eq!(t1, m.to_toml().unwrap());
        assert_eq!(ProjectManifest::from_toml(&t1).unwrap(), m);
    }

    #[test]
    fn missing_optional_fields_default() {
        let text = r#"
            schema_version = 1
            name = "X"
            engine_version = "0.1.0"
            template = "blank-3d"
        "#;
        let m = ProjectManifest::from_toml(text).unwrap();
        assert_eq!(m.content_dir, "Content");
        assert_eq!(m.levels_dir, "Levels");
    }

    #[test]
    fn rejects_newer_schema() {
        let text = "schema_version = 999\nname=\"X\"\nengine_version=\"0.1\"\ntemplate=\"t\"\n";
        assert!(matches!(
            ProjectManifest::from_toml(text),
            Err(ProjectError::SchemaTooNew { .. })
        ));
    }

    #[test]
    fn save_load_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let m = ProjectManifest::new("Disk", "blank-3d");
        m.save(dir.path()).unwrap();
        assert!(dir.path().join(PROJECT_FILE).exists());
        assert_eq!(ProjectManifest::load(dir.path()).unwrap(), m);
    }

    #[test]
    fn load_missing_errors() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            ProjectManifest::load(dir.path()),
            Err(ProjectError::NoManifest(_))
        ));
    }
}
