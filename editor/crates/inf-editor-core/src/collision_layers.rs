//! Project-level collision-layer name registry (P12.1).
//!
//! A [`Collider2D`/`Collider3D`](inf_ecs::components::Collider3D) carries two raw
//! `u32` bitmasks — `collision_memberships` and `collision_filter` — over 32
//! collision layers. This registry names those 32 bits per-project so the editor
//! can eventually present a friendly checkbox list ("Player", "Enemy", "Terrain",
//! "Trigger", …) instead of a hex number. The Details grid still shows the raw
//! `u32` today; a named-bitmask widget driven by this registry is the documented
//! follow-up.
//!
//! It is the exact sibling of the P8.2a sorting-layer registry
//! ([`crate::sorting`]): per-project metadata persisted deterministically as
//! `<root>/.infinity/collision_layers.toml` (sorted-by-bit TOML, stable field
//! order). The `root` the caller passes is the project content root.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// The number of collision-layer bits (matches the `u32` mask width).
pub const LAYER_COUNT: u8 = 32;

/// One named collision layer: a bit index in `0..32` and its display name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollisionLayer {
    /// Bit index in `0..32` (bit `n` ⇔ mask value `1 << n`).
    pub bit: u8,
    pub name: String,
}

/// The set of named collision layers for a project. Order is the presentation
/// order; persistence is deterministic for a given in-memory value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollisionLayers {
    pub layers: Vec<CollisionLayer>,
}

impl Default for CollisionLayers {
    fn default() -> Self {
        Self {
            layers: vec![CollisionLayer {
                bit: 0,
                name: "Default".to_string(),
            }],
        }
    }
}

/// `<root>/.infinity/collision_layers.toml`.
fn layers_path(root: &Path) -> PathBuf {
    root.join(".infinity").join("collision_layers.toml")
}

impl CollisionLayers {
    /// Load the registry for the project rooted at `root`, or the default
    /// (a single "Default" layer at bit 0) when none has been saved yet.
    pub fn load(root: &Path) -> Self {
        match std::fs::read_to_string(layers_path(root)) {
            Ok(text) => toml::from_str(&text).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Canonicalize before persistence: drop empty-named or out-of-range layers,
    /// de-duplicate bits (first wins), sort by bit, and guarantee at least the
    /// bit-0 "Default" layer so the registry is never empty.
    pub fn normalize(&mut self) {
        let mut seen = std::collections::BTreeSet::new();
        self.layers
            .retain(|l| l.bit < LAYER_COUNT && !l.name.trim().is_empty() && seen.insert(l.bit));
        self.layers.sort_by_key(|l| l.bit);
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
        let l = CollisionLayers::default();
        assert_eq!(l.layers.len(), 1);
        assert_eq!(l.layers[0].bit, 0);
        assert_eq!(l.layers[0].name, "Default");
    }

    #[test]
    fn normalize_dedups_sorts_and_drops_invalid() {
        let mut l = CollisionLayers {
            layers: vec![
                CollisionLayer {
                    bit: 5,
                    name: "UI".into(),
                },
                CollisionLayer {
                    bit: 1,
                    name: "".into(),
                }, // dropped (empty)
                CollisionLayer {
                    bit: 40,
                    name: "OOB".into(),
                }, // dropped (out of range)
                CollisionLayer {
                    bit: 5,
                    name: "Dup".into(),
                }, // dropped (dup bit)
                CollisionLayer {
                    bit: 2,
                    name: "Enemy".into(),
                },
            ],
        };
        l.normalize();
        assert_eq!(l.layers.len(), 2);
        // Sorted by bit.
        assert_eq!(l.layers[0].bit, 2);
        assert_eq!(l.layers[0].name, "Enemy");
        assert_eq!(l.layers[1].bit, 5);
    }

    #[test]
    fn empty_after_normalize_falls_back_to_default() {
        let mut l = CollisionLayers { layers: vec![] };
        l.normalize();
        assert_eq!(l, CollisionLayers::default());
    }

    #[test]
    fn save_load_round_trips_and_is_deterministic() {
        let dir = tempfile::tempdir().unwrap();
        let l = CollisionLayers {
            layers: vec![
                CollisionLayer {
                    bit: 0,
                    name: "Default".into(),
                },
                CollisionLayer {
                    bit: 1,
                    name: "Player".into(),
                },
                CollisionLayer {
                    bit: 2,
                    name: "Enemy".into(),
                },
            ],
        };
        l.save(dir.path()).unwrap();
        assert_eq!(l.to_toml().unwrap(), l.to_toml().unwrap());
        assert_eq!(CollisionLayers::load(dir.path()), l);
    }

    #[test]
    fn load_missing_returns_default() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            CollisionLayers::load(dir.path()),
            CollisionLayers::default()
        );
    }
}
