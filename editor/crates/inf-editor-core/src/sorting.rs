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
    ///
    /// **Absent is the default; unreadable is an error** (C4-38) —
    /// `CollisionLayers::load_or_default`'s twin, with the same write-back
    /// consequence: every authored sorting id in the project re-maps.
    pub fn load_or_default(root: &Path) -> Result<Self, String> {
        let path = layers_path(root);
        match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str(&text).map_err(|e| {
                format!(
                    "{} exists but cannot be read ({e}); it is left untouched rather than \
                     replaced by defaults — repair or delete it",
                    path.display()
                )
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("read {}: {e}", path.display())),
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
    ///
    /// Atomic (C4-24) — `CollisionLayers::save`'s twin, and the same
    /// consequence: a truncated file loads as the single Default layer and every
    /// authored sorting id in the project re-maps.
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
        let back = SortingLayers::load_or_default(dir.path()).unwrap();
        assert_eq!(back, l);
    }

    #[test]
    fn load_missing_returns_default() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            SortingLayers::load_or_default(dir.path()).unwrap(),
            SortingLayers::default()
        );
    }

    /// **A corrupt registry is refused; an absent one is the default** (C4-38),
    /// and the corrupt file survives untouched.
    #[test]
    fn a_corrupt_registry_is_refused_while_an_absent_one_is_the_default() {
        let dir = tempfile::tempdir().unwrap();
        SortingLayers::default().save(dir.path()).unwrap();
        let path = layers_path(dir.path());
        let damaged = b"[[layers]]
id = \"not a number\"
"
        .to_vec();
        std::fs::write(&path, &damaged).unwrap();

        let err = SortingLayers::load_or_default(dir.path())
            .expect_err("a damaged registry must not read as the default one");
        assert!(err.contains("cannot be read"), "{err}");
        assert_eq!(std::fs::read(&path).unwrap(), damaged);
    }
}
