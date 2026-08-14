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
    ///
    /// **Absent is the default; unreadable is an error** (C4-38). This was
    /// `unwrap_or_default()`, and the write-back is what made it severe: a
    /// merge-conflicted `collision_layers.toml` silently became the single
    /// Default layer, the dialog showed that, and `collision_layers_set` wrote
    /// it back — destroying every named layer in the project while live
    /// `Collider.collision_memberships` bits kept referencing them. The correct
    /// idiom already existed twice in the repo (`inf_audio`'s mixer,
    /// `assets::collections`); this predates it.
    pub fn load_or_default(root: &Path) -> Result<Self, String> {
        let path = layers_path(root);
        match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str(&text).map_err(|e| {
                format!(
                    "{} exists but cannot be read ({e}); it is left untouched rather than                      replaced by defaults — repair or delete it",
                    path.display()
                )
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("read {}: {e}", path.display())),
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
    ///
    /// Atomic (C4-24): a crash mid-write truncates the file, and a truncated
    /// registry loads as the single Default layer — which silently re-maps every
    /// layer id in the project while live `Collider.collision_memberships` bits
    /// keep referencing the names that are gone.
    pub fn save(&self, root: &Path) -> Result<(), String> {
        let path = layers_path(root);
        inf_asset::write_atomically(&path, self.to_toml()?).map_err(|e| e.to_string())
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
        assert_eq!(CollisionLayers::load_or_default(dir.path()).unwrap(), l);
    }

    #[test]
    fn load_missing_returns_default() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            CollisionLayers::load_or_default(dir.path()).unwrap(),
            CollisionLayers::default()
        );
    }

    /// **A corrupt registry is refused; an absent one is the default** (C4-38)
    /// — the two arms side by side, because the defect was that the code had
    /// one. The corrupt file must also survive: reporting defaults is only half
    /// the harm, `collision_layers_set` writing them back is the other half.
    #[test]
    fn a_corrupt_registry_is_refused_while_an_absent_one_is_the_default() {
        let dir = tempfile::tempdir().unwrap();
        let good = CollisionLayers {
            layers: vec![
                CollisionLayer {
                    bit: 0,
                    name: "Default".into(),
                },
                CollisionLayer {
                    bit: 3,
                    name: "Enemy".into(),
                },
            ],
        };
        good.save(dir.path()).unwrap();

        // A merge conflict, which is how this file actually gets damaged.
        let path = layers_path(dir.path());
        let damaged = b"<<<<<<< HEAD
[[layers]]
bit = 0
=======
"
        .to_vec();
        std::fs::write(&path, &damaged).unwrap();

        let err = CollisionLayers::load_or_default(dir.path())
            .expect_err("a damaged registry must not read as the default one");
        assert!(err.contains("cannot be read"), "{err}");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            damaged,
            "the damaged registry was modified"
        );
    }
}
