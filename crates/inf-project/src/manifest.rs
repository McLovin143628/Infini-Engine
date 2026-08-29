//! The `inf.toml` project manifest.
//!
//! Every Infini Engine project is rooted at a directory containing an
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
    /// Levels root relative to the **content root** (default `"Levels"`), so the
    /// resolved directory is `<root>/<content_dir>/<levels_dir>` — see
    /// [`crate::Project::levels_root`] for the ruling and its one migration.
    ///
    /// **This field's bytes did not change when the ruling landed, only what they
    /// resolve to.** A pre-ruling `inf.toml` says `levels_dir = "Levels"` and a
    /// post-ruling one says the same thing; the manifest schema therefore does
    /// **not** move, and a project written by an older engine opens without a
    /// migration. What moves is the directory the engine points at, and the
    /// content an author already has under `<root>/Levels/` is what the cook's
    /// `stranded levels` advisory is for.
    #[serde(default = "default_levels_dir")]
    pub levels_dir: String,
    /// **The session-default animation blend mode** — `"inertialize"` (the
    /// engine's own default) or `"crossfade"`.
    ///
    /// # Why a project setting, and why here
    ///
    /// `ScenePayload::blend_mode` has round-tripped and been applied since the
    /// island's schema window, and until now **nothing could set it**: no panel,
    /// no command, and the blueprint kit declines it by name. So the preview
    /// default was `Inertialize` in practice and the *cooked* path — which
    /// carries no `ScenePayload` at all — applied none, which meant a project
    /// that ever gained a way to change it would preview one blend and ship
    /// another.
    ///
    /// A per-transition mode is authored in `.inf_sm` and ships with the asset;
    /// this is what an *inheriting* transition inherits, which is a property of
    /// the whole game rather than of one machine or one level. `inf.toml` is
    /// where the cook can read it (`inf-packager` depends on this crate; it does
    /// not depend on the editor), and it is the file an author can put in
    /// version control.
    ///
    /// **This costs no schema move**, for the reason [`Self::levels_dir`]'s note
    /// spells out: TOML is name-keyed and the field defaults, so an older
    /// `inf.toml` opens unchanged and an older engine ignores the key.
    #[serde(default = "default_anim_blend")]
    pub anim_blend: String,
}

fn default_content_dir() -> String {
    "Content".to_string()
}
fn default_levels_dir() -> String {
    "Levels".to_string()
}
fn default_anim_blend() -> String {
    ANIM_BLEND_INERTIALIZE.to_string()
}

/// The engine's own default blend mode — `SmBlendMode::Inertialize`, wire 0.
pub const ANIM_BLEND_INERTIALIZE: &str = "inertialize";
/// `SmBlendMode::CrossFade`, wire 1.
pub const ANIM_BLEND_CROSSFADE: &str = "crossfade";

/// The `SmBlendMode` **wire discriminant** a blend-mode name spells.
///
/// One function, because this name crosses three boundaries — `inf.toml`, the
/// cook's `manifest.toml`, and the shipped player's boot — and three spellings
/// of the mapping is how a project previews one blend and ships another. An
/// unknown name is the engine's default rather than an error: a manifest written
/// by a newer engine must not stop an older one booting.
pub fn anim_blend_wire(name: &str) -> u8 {
    match name.trim().to_ascii_lowercase().as_str() {
        ANIM_BLEND_CROSSFADE => 1,
        _ => 0,
    }
}

/// The inverse of [`anim_blend_wire`].
pub fn anim_blend_name(wire: u8) -> &'static str {
    match wire {
        1 => ANIM_BLEND_CROSSFADE,
        _ => ANIM_BLEND_INERTIALIZE,
    }
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
            anim_blend: default_anim_blend(),
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
