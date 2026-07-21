//! The exported game's boot config: `player.toml` beside the executable (P9.5
//! desktop export).
//!
//! `inf export` renames the release `inf-player` to the project name and drops a
//! `player.toml` next to it that points at the bundled `content.inf_pack`. When
//! the player starts with **no** explicit world flag (`--demo`/`--level`/`--pack`)
//! it looks for this file beside its own executable and boots that pack — so a
//! double-clicked exported game runs its own content with no arguments.
//!
//! Shape (all fields but `pack` optional):
//!
//! ```toml
//! schema_version = 1
//! pack   = "content.inf_pack"   # relative to this file
//! title  = "My Game"            # window title (windowed mode)
//! width  = 1280
//! height = 720
//! ```

use std::path::{Path, PathBuf};

/// The boot-config file name written beside the exported executable.
pub const PLAYER_CONFIG_FILE: &str = "player.toml";

/// Current `player.toml` schema version.
pub const CONFIG_SCHEMA_VERSION: u32 = 1;

/// A resolved boot config: the pack path is made absolute (joined onto the config
/// file's directory) so the caller can open it regardless of the process cwd.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerConfig {
    /// Absolute path to the pack directory-or-file to boot.
    pub pack: PathBuf,
    /// Window title override (windowed mode).
    pub title: Option<String>,
    /// Window width override.
    pub width: Option<u32>,
    /// Window height override.
    pub height: Option<u32>,
}

/// The on-disk `player.toml` layout.
#[derive(Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct PlayerConfigFile {
    #[serde(default = "default_schema")]
    pub schema_version: u32,
    #[serde(default)]
    pub pack: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

fn default_schema() -> u32 {
    CONFIG_SCHEMA_VERSION
}

impl PlayerConfigFile {
    /// Serialize to deterministic pretty TOML (the export writer).
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}

/// Load `player.toml` from `dir`, resolving the pack path against `dir`. Returns
/// `None` if the file is absent, malformed, or has no `pack` field.
pub fn load_from_dir(dir: &Path) -> Option<PlayerConfig> {
    let path = dir.join(PLAYER_CONFIG_FILE);
    let text = std::fs::read_to_string(&path).ok()?;
    let file: PlayerConfigFile = toml::from_str(&text).ok()?;
    if file.pack.trim().is_empty() {
        return None;
    }
    Some(PlayerConfig {
        pack: dir.join(&file.pack),
        title: file.title,
        width: file.width,
        height: file.height,
    })
}

/// Load `player.toml` from the directory holding the running executable.
pub fn load_beside_exe() -> Option<PlayerConfig> {
    let exe = std::env::current_exe().ok()?;
    load_from_dir(exe.parent()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_resolves_pack_relative_to_dir() {
        let dir = tempfile::tempdir().unwrap();
        let file = PlayerConfigFile {
            schema_version: CONFIG_SCHEMA_VERSION,
            pack: "content.inf_pack".into(),
            title: Some("My Game".into()),
            width: Some(1024),
            height: Some(768),
        };
        std::fs::write(dir.path().join(PLAYER_CONFIG_FILE), file.to_toml().unwrap()).unwrap();

        let cfg = load_from_dir(dir.path()).expect("config loads");
        assert_eq!(cfg.pack, dir.path().join("content.inf_pack"));
        assert_eq!(cfg.title.as_deref(), Some("My Game"));
        assert_eq!(cfg.width, Some(1024));
    }

    #[test]
    fn absent_or_empty_config_is_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_from_dir(dir.path()).is_none());
        std::fs::write(dir.path().join(PLAYER_CONFIG_FILE), "pack = \"\"\n").unwrap();
        assert!(load_from_dir(dir.path()).is_none());
    }
}
