//! Per-project editor settings (P8.2c).
//!
//! A tiny TOML under the content root's `.infinity/` directory, mirroring the
//! sorting-layer registry's deterministic persistence (`sorting.rs`). Today it
//! holds only the **pixels-per-unit** used by 2D pixel snapping, but it is the
//! home for future per-project editor preferences.
//!
//! Persisted as `<root>/.infinity/settings.toml` (deterministic, stable field
//! order — the sidecar byte-determinism discipline). The `root` the caller
//! passes is the project content root.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Default pixels-per-unit: a sprite authored at 100 px maps to 1 world unit.
pub const DEFAULT_PIXELS_PER_UNIT: f32 = 100.0;

/// Per-project editor settings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ProjectSettings {
    /// Pixels-per-unit for 2D pixel snapping (translate snaps to `1/ppu` world
    /// units when pixel snap is on).
    #[serde(default = "default_ppu")]
    pub pixels_per_unit: f32,
}

fn default_ppu() -> f32 {
    DEFAULT_PIXELS_PER_UNIT
}

impl Default for ProjectSettings {
    fn default() -> Self {
        Self {
            pixels_per_unit: DEFAULT_PIXELS_PER_UNIT,
        }
    }
}

/// `<root>/.infinity/settings.toml`.
fn settings_path(root: &Path) -> PathBuf {
    root.join(".infinity").join("settings.toml")
}

impl ProjectSettings {
    /// Load the settings for the project rooted at `root`, or the default when
    /// none has been saved / the file is unparseable.
    pub fn load(root: &Path) -> Self {
        match std::fs::read_to_string(settings_path(root)) {
            Ok(text) => toml::from_str(&text).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Clamp to a usable range: a non-finite or non-positive PPU falls back to
    /// the default (pixel snap divides by it).
    pub fn normalize(&mut self) {
        if !self.pixels_per_unit.is_finite() || self.pixels_per_unit <= 0.0 {
            self.pixels_per_unit = DEFAULT_PIXELS_PER_UNIT;
        }
    }

    /// Deterministic TOML for the (normalized) settings.
    pub fn to_toml(&self) -> Result<String, String> {
        let mut copy = *self;
        copy.normalize();
        toml::to_string_pretty(&copy).map_err(|e| e.to_string())
    }

    /// Write the settings under `root` (creating `.infinity/` if needed).
    pub fn save(&self, root: &Path) -> Result<(), String> {
        let path = settings_path(root);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&path, self.to_toml()?).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_100_ppu() {
        assert_eq!(ProjectSettings::default().pixels_per_unit, 100.0);
    }

    #[test]
    fn normalize_rejects_nonpositive_and_nan() {
        let mut s = ProjectSettings {
            pixels_per_unit: 0.0,
        };
        s.normalize();
        assert_eq!(s.pixels_per_unit, DEFAULT_PIXELS_PER_UNIT);
        let mut n = ProjectSettings {
            pixels_per_unit: f32::NAN,
        };
        n.normalize();
        assert_eq!(n.pixels_per_unit, DEFAULT_PIXELS_PER_UNIT);
    }

    #[test]
    fn save_load_round_trips_and_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let s = ProjectSettings {
            pixels_per_unit: 32.0,
        };
        s.save(dir.path()).unwrap();
        assert_eq!(s.to_toml().unwrap(), s.to_toml().unwrap());
        assert_eq!(ProjectSettings::load(dir.path()), s);
    }

    #[test]
    fn load_missing_returns_default() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            ProjectSettings::load(dir.path()),
            ProjectSettings::default()
        );
    }
}
