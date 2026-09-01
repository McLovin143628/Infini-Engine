//! The exported game's boot config: `player.toml` beside the executable (P9.5
//! desktop export).
//!
//! `inf export` renames the release `inf-player` to the project name and drops a
//! `player.toml` next to it that points at the bundled `content.ipack`. When
//! the player starts with **no** explicit world flag (`--demo`/`--level`/`--pack`)
//! it looks for this file beside its own executable and boots that pack — so a
//! double-clicked exported game runs its own content with no arguments.
//!
//! Shape (all fields but `pack` optional):
//!
//! ```toml
//! schema_version = 1
//! pack   = "content.ipack"   # relative to this file
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
/// **`Default` writes a config the loader refuses** (round-2, the MED
/// cluster). `#[derive(Default)]` gives `schema_version = 0`, and `load` now
/// refuses anything below `CONFIG_SCHEMA_VERSION` — so
/// `PlayerConfigFile::default()` round-tripped through the writer is a bundle
/// that will not boot. Latent, because nothing in the tree calls it; `pub`, so
/// it is a door somebody will. The hand-written impl uses the same
/// `default_schema()` the serde attribute does, so the two cannot drift.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
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

impl Default for PlayerConfigFile {
    fn default() -> Self {
        Self {
            schema_version: default_schema(),
            pack: String::new(),
            title: None,
            width: None,
            height: None,
        }
    }
}

impl PlayerConfigFile {
    /// Serialize to deterministic pretty TOML (the export writer).
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}

/// Why a `player.toml` that **exists** could not be honoured.
///
/// Absence is not an error — it is `Ok(None)` from [`load_from_dir`]. Every
/// variant here means the file is there and the shipped game must **refuse to
/// boot** rather than fall through to the built-in demo (C4-33): a customer who
/// hand-edits a title into a syntax error would otherwise double-click their
/// game and watch the engine's own platformer start, exit 0.
#[derive(Debug)]
pub enum ConfigError {
    /// The file exists but could not be read (permissions, a locked handle, IO).
    Read(std::io::Error),
    /// The file exists but is not valid TOML, or does not match the schema.
    Parse(String),
    /// `schema_version` is not one this build understands.
    ///
    /// `found == 0` is the torn/zero-filled case: the field is `default`ed to
    /// [`CONFIG_SCHEMA_VERSION`] when *absent*, so a literal zero can only come
    /// from a file that says so.
    UnknownSchema {
        /// The version the file declares.
        found: u32,
        /// The newest version this build can read.
        supported: u32,
    },
    /// The file exists and parses but names no pack.
    NoPack,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read(e) => write!(f, "could not read {PLAYER_CONFIG_FILE}: {e}"),
            Self::Parse(e) => write!(f, "{PLAYER_CONFIG_FILE} is malformed: {e}"),
            Self::UnknownSchema { found, supported } => write!(
                f,
                "{PLAYER_CONFIG_FILE} declares schema_version {found}; this build reads 1..={supported}"
            ),
            Self::NoPack => write!(f, "{PLAYER_CONFIG_FILE} names no `pack`"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Load `player.toml` from `dir`, resolving the pack path against `dir`.
///
/// `Ok(None)` means **absent** — the only case where booting something else is
/// correct. Every other failure is an [`ConfigError`], because the file being
/// there is the customer saying "boot my game".
pub fn load_from_dir(dir: &Path) -> Result<Option<PlayerConfig>, ConfigError> {
    let path = dir.join(PLAYER_CONFIG_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(ConfigError::Read(e)),
    };
    let file: PlayerConfigFile =
        toml::from_str(&text).map_err(|e| ConfigError::Parse(e.to_string()))?;
    if file.schema_version == 0 || file.schema_version > CONFIG_SCHEMA_VERSION {
        return Err(ConfigError::UnknownSchema {
            found: file.schema_version,
            supported: CONFIG_SCHEMA_VERSION,
        });
    }
    if file.pack.trim().is_empty() {
        return Err(ConfigError::NoPack);
    }
    Ok(Some(PlayerConfig {
        pack: dir.join(&file.pack),
        title: file.title,
        width: file.width,
        height: file.height,
    }))
}

/// Load `player.toml` from the directory holding the running executable.
///
/// An unresolvable executable path is treated as "absent" — there is no file to
/// have an opinion about.
pub fn load_beside_exe() -> Result<Option<PlayerConfig>, ConfigError> {
    let Ok(exe) = std::env::current_exe() else {
        return Ok(None);
    };
    let Some(dir) = exe.parent() else {
        return Ok(None);
    };
    load_from_dir(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_resolves_pack_relative_to_dir() {
        let dir = tempfile::tempdir().unwrap();
        let file = PlayerConfigFile {
            schema_version: CONFIG_SCHEMA_VERSION,
            pack: "content.ipack".into(),
            title: Some("My Game".into()),
            width: Some(1024),
            height: Some(768),
        };
        std::fs::write(dir.path().join(PLAYER_CONFIG_FILE), file.to_toml().unwrap()).unwrap();

        let cfg = load_from_dir(dir.path())
            .expect("config loads")
            .expect("config present");
        assert_eq!(cfg.pack, dir.path().join("content.ipack"));
        assert_eq!(cfg.title.as_deref(), Some("My Game"));
        assert_eq!(cfg.width, Some(1024));
    }

    #[test]
    fn absent_config_is_ok_none() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_from_dir(dir.path())
            .expect("absent is not an error")
            .is_none());
    }

    /// C4-33: a `player.toml` that exists and is broken must **refuse**, not
    /// fall through to `WorldChoice::Demo`. Each arm below was the same `None`
    /// before the repair, which is why one assertion per shape is the arm.
    #[test]
    fn a_present_but_broken_config_refuses_rather_than_defaulting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(PLAYER_CONFIG_FILE);

        // Malformed TOML — the hand-edited-title case.
        std::fs::write(&path, "title = \"My Game\npack = \"content.ipack\"\n").unwrap();
        assert!(matches!(
            load_from_dir(dir.path()),
            Err(ConfigError::Parse(_))
        ));

        // Parses, but names no pack.
        std::fs::write(&path, "pack = \"\"\n").unwrap();
        assert!(matches!(
            load_from_dir(dir.path()),
            Err(ConfigError::NoPack)
        ));

        // A zero-filled / torn file: `schema_version = 0` is only reachable from
        // a file that says zero, because absence defaults to the current version.
        std::fs::write(&path, "schema_version = 0\npack = \"content.ipack\"\n").unwrap();
        assert!(matches!(
            load_from_dir(dir.path()),
            Err(ConfigError::UnknownSchema { found: 0, .. })
        ));

        // Newer than this build.
        std::fs::write(&path, "schema_version = 99\npack = \"content.ipack\"\n").unwrap();
        assert!(matches!(
            load_from_dir(dir.path()),
            Err(ConfigError::UnknownSchema { found: 99, .. })
        ));

        // A wrong-typed field is a schema mismatch, not a silent default.
        std::fs::write(&path, "pack = \"content.ipack\"\nwidth = \"wide\"\n").unwrap();
        assert!(matches!(
            load_from_dir(dir.path()),
            Err(ConfigError::Parse(_))
        ));
    }

    /// Every refusal must say which file and why — the message is the whole
    /// remedy for a customer holding a bundle that will not start.
    #[test]
    fn every_refusal_names_the_file() {
        for e in [
            ConfigError::Parse("expected `=`".into()),
            ConfigError::NoPack,
            ConfigError::UnknownSchema {
                found: 9,
                supported: CONFIG_SCHEMA_VERSION,
            },
        ] {
            assert!(e.to_string().contains(PLAYER_CONFIG_FILE), "{e}");
        }
    }
}
