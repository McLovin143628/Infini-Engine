//! Project-level sorting-layer registry (P8.2a).
//!
//! Sprites draw back-to-front by a coarse **sorting layer** (an `i32` on the
//! [`inf_ecs::Sprite`] component) then a fine `order`. This registry names those
//! integer layers so the editor can present them as a friendly ordered list
//! ("Background", "Default", "Foreground", "UI") while the component keeps the
//! raw `i32`.
//!
//! It is per-project metadata, persisted deterministically as
//! `<root>/.infinity/sorting_layers.toml` (mirroring the sidecar
//! byte-determinism discipline: sorted-by-key TOML, stable field order). The
//! `root` the caller passes is the project content root.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One named sorting layer: a stable `i32` id and its display name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SortingLayer {
    pub id: i32,
    pub name: String,
}

/// The ordered set of named sorting layers for a project. Order is meaningful
/// (it is the drawing order the list presents), but persistence is deterministic
/// for a given in-memory value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SortingLayers {
    pub layers: Vec<SortingLayer>,
}

impl Default for SortingLayers {
    fn default() -> Self {
        Self {
            layers: vec![SortingLayer {
                id: 0,
                name: "Default".to_string(),
            }],
        }
    }
}

/// `<root>/.infinity/sorting_layers.toml`.
fn layers_path(root: &Path) -> PathBuf {
    root.join(".infinity").join("sorting_layers.toml")
}

impl SortingLayers {
    /// Load the registry for the project rooted at `root`, or the default
    /// (a single "Default" layer at id 0) when none has been saved yet.
    pub fn load(root: &Path) -> Self {
        match std::fs::read_to_string(layers_path(root)) {
            Ok(text) => toml::from_str(&text).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Canonicalize before persistence: drop empty-named layers, de-duplicate
    /// ids (first wins), and guarantee at least the id-0 "Default" layer so the
    /// registry is never empty.
    pub fn normalize(&mut self) {
        let mut seen = std::collections::BTreeSet::new();
        self.layers
            .retain(|l| !l.name.trim().is_empty() && seen.insert(l.id));
        if self.layers.is_empty() {
            *self = Self::default();
        }
    }

    /// Deterministic TOML for the (normalized) registry.
    pub fn to_toml(&self) -> Result<String, String> {
        let mut copy = self.clone();
        copy.normalize();
        toml::to_string_pretty(&copy).map_err(|e| e.to_string())
    }

    /// Write the registry under `root` (creating `.infinity/` if needed).
    pub fn save(&self, root: &Path) -> Result<(), String> {
        let path = layers_path(root);
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
    fn default_has_one_default_layer() {
        let l = SortingLayers::default();
        assert_eq!(l.layers.len(), 1);
        assert_eq!(l.layers[0].id, 0);
        assert_eq!(l.layers[0].name, "Default");
    }

    #[test]
    fn normalize_dedups_and_drops_empty() {
        let mut l = SortingLayers {
            layers: vec![
                SortingLayer {
                    id: -1,
                    name: "BG".into(),
                },
                SortingLayer {
                    id: 0,
                    name: "".into(),
                }, // dropped (empty)
                SortingLayer {
                    id: -1,
                    name: "Dup".into(),
                }, // dropped (dup id)
                SortingLayer {
                    id: 5,
                    name: "UI".into(),
                },
            ],
        };
        l.normalize();
        assert_eq!(l.layers.len(), 2);
        assert_eq!(l.layers[0].name, "BG");
        assert_eq!(l.layers[1].name, "UI");
    }

    #[test]
    fn empty_after_normalize_falls_back_to_default() {
        let mut l = SortingLayers { layers: vec![] };
        l.normalize();
        assert_eq!(l, SortingLayers::default());
    }

    #[test]
    fn save_load_round_trips_and_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let l = SortingLayers {
            layers: vec![
                SortingLayer {
                    id: -10,
                    name: "Background".into(),
                },
                SortingLayer {
                    id: 0,
                    name: "Default".into(),
                },
                SortingLayer {
                    id: 10,
                    name: "Foreground".into(),
                },
            ],
        };
        l.save(dir.path()).unwrap();
        let a = l.to_toml().unwrap();
        let b = l.to_toml().unwrap();
        assert_eq!(a, b, "re-emit is byte-identical");
        let back = SortingLayers::load(dir.path());
        assert_eq!(back, l);
    }

    #[test]
    fn load_missing_returns_default() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(SortingLayers::load(dir.path()), SortingLayers::default());
    }
}
