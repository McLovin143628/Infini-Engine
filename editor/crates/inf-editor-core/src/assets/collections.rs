//! Named content collections (E-P8): user-defined, **persisted** groupings of
//! assets, the durable successor to the frontend-only Favorites.
//!
//! A collection is a name + an ordered list of asset GUIDs. Collections are NOT
//! dependency edges — they carry no delete-guard semantics (deliberate: deleting
//! an asset never asks about collections). Dangling ids (an asset a collection
//! points at was deleted) are pruned on load against the live [`AssetDb`].
//!
//! Persisted deterministically at `<project_root>/.infinity/collections.toml`
//! (the same directory as `inf_audio`'s `mixer.toml`), with a `schema_version`
//! guard that accepts equal-or-older and rejects newer — mirroring the mixer /
//! asset-payload migrate discipline. On save the collections are sorted by name
//! (unique keys → stable order); each collection's ids keep insertion order.

use std::path::{Path, PathBuf};

use inf_asset::{AssetDb, AssetId};
use serde::{Deserialize, Serialize};

/// The current [`CollectionsFile`] schema version. Bumped when the on-disk shape
/// changes; equal-or-older is accepted, newer is rejected.
pub const COLLECTIONS_SCHEMA_VERSION: u32 = 1;

/// Relative path (under a project root) of the persisted collections file.
pub const COLLECTIONS_REL_PATH: &str = ".infinity/collections.toml";

/// One named collection: a unique display name + an ordered set of asset GUIDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Collection {
    /// Unique (case-sensitive) display name.
    pub name: String,
    /// Member asset GUIDs, in insertion order (deduped).
    #[serde(default)]
    pub ids: Vec<AssetId>,
}

/// The persisted set of named collections.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectionsFile {
    /// On-disk schema version (see [`COLLECTIONS_SCHEMA_VERSION`]).
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// The collections. Sorted by name on save.
    #[serde(default)]
    pub collections: Vec<Collection>,
}

fn default_schema_version() -> u32 {
    COLLECTIONS_SCHEMA_VERSION
}

impl Default for CollectionsFile {
    fn default() -> Self {
        Self {
            schema_version: COLLECTIONS_SCHEMA_VERSION,
            collections: Vec::new(),
        }
    }
}

impl CollectionsFile {
    /// The collections path under a project root (`<root>/.infinity/collections.toml`).
    pub fn path_in(root: &Path) -> PathBuf {
        root.join(COLLECTIONS_REL_PATH)
    }

    /// Validate + accept an equal-or-older file (rejects a newer schema),
    /// mirroring the mixer / asset-payload migrate guard.
    pub fn migrate(self) -> Result<Self, String> {
        if self.schema_version > COLLECTIONS_SCHEMA_VERSION {
            return Err(format!(
                "collections.toml schema v{} is newer than supported v{COLLECTIONS_SCHEMA_VERSION}",
                self.schema_version
            ));
        }
        Ok(self)
    }

    /// Deserialize from a TOML string (with the migrate guard).
    pub fn from_toml_str(s: &str) -> Result<Self, String> {
        toml::from_str::<CollectionsFile>(s)
            .map_err(|e| e.to_string())?
            .migrate()
    }

    /// Serialize to a deterministic pretty-TOML string (collections sorted by name).
    pub fn to_toml_string(&self) -> Result<String, String> {
        let mut copy = self.clone();
        copy.sort();
        toml::to_string_pretty(&copy).map_err(|e| e.to_string())
    }

    /// Load from `<root>/.infinity/collections.toml`, or the empty [`Default`]
    /// when absent. A parse error is surfaced (not swallowed).
    pub fn load_or_default(root: &Path) -> Result<Self, String> {
        let path = Self::path_in(root);
        match std::fs::read_to_string(&path) {
            Ok(s) => Self::from_toml_str(&s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("read {}: {e}", path.display())),
        }
    }

    /// Write to `<root>/.infinity/collections.toml`, creating `.infinity/`.
    /// Deterministic: collections are sorted by name (via [`to_toml_string`]).
    ///
    /// [`to_toml_string`]: Self::to_toml_string
    /// Atomic (C4-24), like the rest of the `.infinity/` family.
    pub fn save(&self, root: &Path) -> Result<(), String> {
        let path = Self::path_in(root);
        inf_asset::write_atomically(&path, self.to_toml_string()?)
            .map_err(|e| format!("write {}: {e}", path.display()))
    }

    /// Sort collections by name in place (unique names → stable order). Ids keep
    /// their insertion order.
    pub fn sort(&mut self) {
        self.collections.sort_by(|a, b| a.name.cmp(&b.name));
    }

    fn find(&self, name: &str) -> Option<usize> {
        self.collections.iter().position(|c| c.name == name)
    }

    /// Create a new (empty) collection. Errors on an empty or duplicate name.
    pub fn create(&mut self, name: &str) -> Result<(), String> {
        let name = name.trim();
        if name.is_empty() {
            return Err("collection name cannot be empty".into());
        }
        if self.find(name).is_some() {
            return Err(format!("a collection named \"{name}\" already exists"));
        }
        self.collections.push(Collection {
            name: name.to_string(),
            ids: Vec::new(),
        });
        Ok(())
    }

    /// Rename a collection. Errors if `old` is missing, or `new` is empty /
    /// duplicates another collection.
    pub fn rename(&mut self, old: &str, new: &str) -> Result<(), String> {
        let new = new.trim();
        if new.is_empty() {
            return Err("collection name cannot be empty".into());
        }
        if old != new && self.find(new).is_some() {
            return Err(format!("a collection named \"{new}\" already exists"));
        }
        let idx = self
            .find(old)
            .ok_or_else(|| format!("no collection named \"{old}\""))?;
        self.collections[idx].name = new.to_string();
        Ok(())
    }

    /// Delete a collection. Errors if it does not exist.
    pub fn delete(&mut self, name: &str) -> Result<(), String> {
        let idx = self
            .find(name)
            .ok_or_else(|| format!("no collection named \"{name}\""))?;
        self.collections.remove(idx);
        Ok(())
    }

    /// Add an asset to a collection (deduped, insertion-ordered). Errors if the
    /// collection does not exist.
    pub fn add(&mut self, name: &str, id: AssetId) -> Result<(), String> {
        let idx = self
            .find(name)
            .ok_or_else(|| format!("no collection named \"{name}\""))?;
        let ids = &mut self.collections[idx].ids;
        if !ids.contains(&id) {
            ids.push(id);
        }
        Ok(())
    }

    /// Remove an asset from a collection. Errors if the collection does not
    /// exist (a missing id is a no-op).
    pub fn remove(&mut self, name: &str, id: AssetId) -> Result<(), String> {
        let idx = self
            .find(name)
            .ok_or_else(|| format!("no collection named \"{name}\""))?;
        self.collections[idx].ids.retain(|x| *x != id);
        Ok(())
    }

    /// Drop ids whose assets no longer exist (per the `exists` predicate),
    /// returning how many were removed. Called on load so a deleted asset never
    /// lingers as a dangling collection member — collections are not dependency
    /// edges, so the delete command does not know about them.
    pub fn prune_missing(&mut self, exists: impl Fn(AssetId) -> bool) -> usize {
        let mut removed = 0usize;
        for c in &mut self.collections {
            let before = c.ids.len();
            c.ids.retain(|id| exists(*id));
            removed += before - c.ids.len();
        }
        removed
    }

    /// Convenience: prune against a live [`AssetDb`] (the on-load caller).
    pub fn prune_against_db(&mut self, db: &AssetDb) -> usize {
        self.prune_missing(|id| db.contains(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn create_rejects_empty_and_duplicate() {
        let mut f = CollectionsFile::default();
        f.create("Props").unwrap();
        assert!(f.create("Props").is_err(), "duplicate rejected");
        assert!(f.create("   ").is_err(), "empty rejected");
        assert_eq!(f.collections.len(), 1);
    }

    #[test]
    fn add_dedupes_and_keeps_insertion_order() {
        let mut f = CollectionsFile::default();
        f.create("Props").unwrap();
        let a = AssetId::new();
        let b = AssetId::new();
        f.add("Props", a).unwrap();
        f.add("Props", b).unwrap();
        f.add("Props", a).unwrap(); // dupe → no-op
        assert_eq!(f.collections[0].ids, vec![a, b]);
    }

    #[test]
    fn rename_and_delete() {
        let mut f = CollectionsFile::default();
        f.create("A").unwrap();
        f.create("B").unwrap();
        assert!(f.rename("A", "B").is_err(), "collides with B");
        f.rename("A", "C").unwrap();
        assert!(f.find("C").is_some());
        f.delete("C").unwrap();
        assert!(f.find("C").is_none());
        assert!(f.delete("C").is_err(), "already gone");
    }

    #[test]
    fn round_trips_and_is_deterministic_sorted_by_name() {
        let mut f = CollectionsFile::default();
        // Insert out of order; save must sort by name.
        f.create("Zed").unwrap();
        f.create("Alpha").unwrap();
        let a = AssetId::new();
        f.add("Zed", a).unwrap();

        let s = f.to_toml_string().unwrap();
        // Alpha comes before Zed in the serialized form.
        let ai = s.find("Alpha").unwrap();
        let zi = s.find("Zed").unwrap();
        assert!(ai < zi, "collections sorted by name on save");

        let back = CollectionsFile::from_toml_str(&s).unwrap();
        assert_eq!(back.to_toml_string().unwrap(), s, "stable re-serialize");
        // The Zed member survives the round-trip.
        assert_eq!(
            back.collections
                .iter()
                .find(|c| c.name == "Zed")
                .unwrap()
                .ids,
            vec![a]
        );
    }

    #[test]
    fn save_load_round_trips_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let mut f = CollectionsFile::default();
        f.create("Props").unwrap();
        f.add("Props", AssetId::new()).unwrap();
        f.save(dir.path()).unwrap();
        let mut loaded = CollectionsFile::load_or_default(dir.path()).unwrap();
        loaded.sort();
        f.sort();
        assert_eq!(loaded, f);
    }

    #[test]
    fn load_missing_is_default_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            CollectionsFile::load_or_default(dir.path()).unwrap(),
            CollectionsFile::default()
        );
    }

    #[test]
    fn prune_drops_dangling_ids() {
        let mut f = CollectionsFile::default();
        f.create("Props").unwrap();
        let live = AssetId::new();
        let dead = AssetId::new();
        f.add("Props", live).unwrap();
        f.add("Props", dead).unwrap();

        let present: HashSet<AssetId> = [live].into_iter().collect();
        let removed = f.prune_missing(|id| present.contains(&id));
        assert_eq!(removed, 1);
        assert_eq!(f.collections[0].ids, vec![live], "only the live id remains");
    }

    #[test]
    fn migrate_rejects_newer_schema() {
        let f = CollectionsFile {
            schema_version: COLLECTIONS_SCHEMA_VERSION + 1,
            collections: vec![],
        };
        assert!(f.migrate().is_err());
    }
}
