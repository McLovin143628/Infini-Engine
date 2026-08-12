//! The asset database: a GUID-keyed registry with a live dependency graph.
//!
//! The database is an **in-memory index** over the sidecars found under a
//! content root. It answers the questions the editor asks constantly:
//!   * by GUID / by path / by kind — the Content Drawer's listing + selection;
//!   * *what does this reference* (forward deps) — cook + load ordering;
//!   * *what references this* (reverse deps) — the "delete-with-references
//!     warns" safety check (ROADMAP Phase 4 gate);
//!   * *is this content already imported* (content-hash index) — dedupe.
//!
//! Mutations to disk (writing sidecars) are explicit (`persist`); the database
//! keeps memory and its reverse-edge cache consistent as entries come and go.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::error::{AssetError, Result};
use crate::hash::ContentHash;
use crate::id::AssetId;
use crate::kind::AssetKind;
use crate::sidecar::{is_sidecar, AssetSidecar};

/// One registered asset: its sidecar metadata plus where its payload lives.
#[derive(Debug, Clone, PartialEq)]
pub struct AssetEntry {
    /// Sidecar metadata (guid, kind, hash, deps, tags, source, import).
    pub sidecar: AssetSidecar,
    /// Absolute path to the payload file.
    pub path: PathBuf,
    /// Display name (file stem).
    pub name: String,
}

impl AssetEntry {
    pub fn id(&self) -> AssetId {
        self.sidecar.guid
    }
    pub fn kind(&self) -> AssetKind {
        self.sidecar.kind
    }
    pub fn content_hash(&self) -> ContentHash {
        self.sidecar.content_hash
    }
}

/// The registry + dependency graph.
#[derive(Debug, Default)]
pub struct AssetDb {
    root: PathBuf,
    by_id: HashMap<AssetId, AssetEntry>,
    by_path: HashMap<PathBuf, AssetId>,
    by_hash: HashMap<ContentHash, HashSet<AssetId>>,
    /// Reverse dependency edges: `dep -> {assets that reference dep}`.
    reverse: HashMap<AssetId, BTreeSet<AssetId>>,
}

impl AssetDb {
    /// A database rooted at the project content directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            ..Default::default()
        }
    }

    /// The content root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Number of registered assets.
    pub fn len(&self) -> usize {
        self.by_id.len()
    }
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    // ── lookups ───────────────────────────────────────────────────────────

    pub fn get(&self, id: AssetId) -> Option<&AssetEntry> {
        self.by_id.get(&id)
    }

    pub fn get_by_path(&self, path: &Path) -> Option<&AssetEntry> {
        let key = normalize(path);
        self.by_path.get(&key).and_then(|id| self.by_id.get(id))
    }

    pub fn contains(&self, id: AssetId) -> bool {
        self.by_id.contains_key(&id)
    }

    /// All entries, unordered.
    pub fn iter(&self) -> impl Iterator<Item = &AssetEntry> {
        self.by_id.values()
    }

    /// All entries of a given kind.
    pub fn by_kind(&self, kind: AssetKind) -> impl Iterator<Item = &AssetEntry> {
        self.by_id.values().filter(move |e| e.kind() == kind)
    }

    /// Every asset whose payload hashes to `hash` (usually 0 or 1; >1 means the
    /// same content was imported twice — a dedupe opportunity).
    pub fn by_content_hash(&self, hash: ContentHash) -> Vec<AssetId> {
        self.by_hash
            .get(&hash)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    // ── dependency queries ────────────────────────────────────────────────

    /// Forward edges: the assets `id` references. `None` if `id` is unknown.
    pub fn references_of(&self, id: AssetId) -> Option<&[AssetId]> {
        self.by_id
            .get(&id)
            .map(|e| e.sidecar.dependencies.as_slice())
    }

    /// Reverse edges: the assets that reference `id`. Empty if none (or unknown).
    /// This is the query that powers "delete-with-references warns".
    pub fn referenced_by(&self, id: AssetId) -> Vec<AssetId> {
        self.reverse
            .get(&id)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    /// True if deleting `id` would leave dangling references.
    pub fn has_referrers(&self, id: AssetId) -> bool {
        self.reverse.get(&id).is_some_and(|s| !s.is_empty())
    }

    // ── mutation ──────────────────────────────────────────────────────────

    /// Register (or replace) an entry, keeping every index and the reverse-edge
    /// cache consistent.
    pub fn insert(&mut self, mut entry: AssetEntry) {
        entry.sidecar.normalize();
        entry.path = normalize(&entry.path);
        let id = entry.id();

        // Remove any prior state for this id (path/hash/edges may have changed).
        if self.by_id.contains_key(&id) {
            self.detach(id);
        }

        self.by_path.insert(entry.path.clone(), id);
        self.by_hash
            .entry(entry.content_hash())
            .or_default()
            .insert(id);
        for &dep in &entry.sidecar.dependencies {
            self.reverse.entry(dep).or_default().insert(id);
        }
        self.by_id.insert(id, entry);
    }

    /// Remove an asset from the index (does not touch disk).
    pub fn remove(&mut self, id: AssetId) -> Option<AssetEntry> {
        if !self.by_id.contains_key(&id) {
            return None;
        }
        self.detach(id);
        self.by_id.remove(&id)
    }

    /// Detach `id` from the path/hash/reverse indices without dropping the entry
    /// itself (used by both `insert`-replace and `remove`).
    fn detach(&mut self, id: AssetId) {
        if let Some(entry) = self.by_id.get(&id) {
            self.by_path.remove(&entry.path);
            if let Some(set) = self.by_hash.get_mut(&entry.content_hash()) {
                set.remove(&id);
                if set.is_empty() {
                    self.by_hash.remove(&entry.content_hash());
                }
            }
            let deps = entry.sidecar.dependencies.clone();
            for dep in deps {
                if let Some(set) = self.reverse.get_mut(&dep) {
                    set.remove(&id);
                    if set.is_empty() {
                        self.reverse.remove(&dep);
                    }
                }
            }
        }
    }

    /// Replace an asset's dependency list, updating reverse edges. Returns the
    /// old list. Errors if `id` is unknown.
    pub fn set_dependencies(&mut self, id: AssetId, deps: Vec<AssetId>) -> Result<Vec<AssetId>> {
        let entry = self.by_id.get(&id).ok_or(AssetError::UnknownAsset(id))?;
        let old = entry.sidecar.dependencies.clone();
        // Re-insert with the new deps to reuse the consistency machinery.
        let mut updated = entry.clone();
        updated.sidecar.dependencies = deps;
        self.insert(updated);
        Ok(old)
    }

    /// Replace an asset's tags. Errors if `id` is unknown.
    pub fn set_tags(&mut self, id: AssetId, tags: Vec<String>) -> Result<()> {
        let entry = self
            .by_id
            .get_mut(&id)
            .ok_or(AssetError::UnknownAsset(id))?;
        entry.sidecar.tags = tags;
        entry.sidecar.normalize();
        Ok(())
    }

    /// Write an asset's sidecar to disk (memory → disk).
    pub fn persist(&self, id: AssetId) -> Result<()> {
        let entry = self.by_id.get(&id).ok_or(AssetError::UnknownAsset(id))?;
        entry.sidecar.save(&entry.path)
    }

    // ── scanning ──────────────────────────────────────────────────────────

    /// Rebuild the whole index from disk by walking the content root.
    pub fn scan(&mut self) -> Result<usize> {
        self.by_id.clear();
        self.by_path.clear();
        self.by_hash.clear();
        self.reverse.clear();
        let root = self.root.clone();
        let mut count = 0;
        if root.exists() {
            self.scan_dir(&root, &mut count)?;
        }
        Ok(count)
    }

    fn scan_dir(&mut self, dir: &Path, count: &mut usize) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let ft = entry.file_type()?;
            if ft.is_dir() {
                self.scan_dir(&path, count)?;
            } else if ft.is_file() && !is_sidecar(&path) {
                if let Some(e) = read_entry(&path)? {
                    self.insert(e);
                    *count += 1;
                }
            }
        }
        Ok(())
    }

    /// Re-read a single asset at `path` (on a watch event). Returns the id if it
    /// is a recognized, sidecar-bearing asset.
    pub fn rescan_path(&mut self, path: &Path) -> Result<Option<AssetId>> {
        if is_sidecar(path) {
            return Ok(None);
        }
        match read_entry(path)? {
            Some(e) => {
                let id = e.id();
                self.insert(e);
                Ok(Some(id))
            }
            None => Ok(None),
        }
    }

    /// Drop the asset registered at `path`, if any (on a remove event). Returns
    /// the id that was removed.
    pub fn remove_path(&mut self, path: &Path) -> Option<AssetId> {
        let id = *self.by_path.get(&normalize(path))?;
        self.remove(id);
        Some(id)
    }
}

/// Read one payload path into an [`AssetEntry`], loading its sidecar. Recognized
/// `.inf_*` payloads without a sidecar are still surfaced (with a synthesized
/// sidecar) so nothing under the content root goes invisible; unrecognized files
/// are ignored.
fn read_entry(path: &Path) -> Result<Option<AssetEntry>> {
    let kind = AssetKind::from_path(path);
    if kind == AssetKind::Unknown {
        return Ok(None);
    }
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("asset")
        .to_string();

    let sidecar = match AssetSidecar::load(path) {
        Ok(s) => s,
        Err(_) => {
            // No/invalid sidecar: synthesize one from the payload so the asset is
            // still browsable. The GUID is derived deterministically from the
            // content hash so a missing sidecar doesn't churn ids across scans.
            let bytes = std::fs::read(path)?;
            let hash = ContentHash::of(&bytes);
            let declared = declared_sidecar(path);
            // A sidecar this crate cannot parse may still name the asset's own
            // GUID, and a `.inf_lvl`'s does. Honouring it is not a nicety: the
            // content-hash fallback below makes a level's asset id **churn with
            // its contents**, so every save re-registers the level under a new id
            // and every edge into it goes stale. Same key-not-schema reading as
            // the dependencies below.
            let guid = declared
                .0
                .unwrap_or_else(|| AssetId(uuid::Uuid::from_u128(hash.0)));
            let mut side = AssetSidecar::new(guid, kind, hash);
            // …but a sidecar this crate cannot parse may still DECLARE its edges,
            // and a `.inf_lvl`'s does (P26.4).
            //
            // A level's sidecar is written by `inf_editor_core::scene::serialize`
            // to its own schema — title, entity count, a 64-bit content hash —
            // so it has never parsed as an `AssetSidecar` and every level has
            // therefore been an asset with **no outgoing edges**. Which made
            // `has_referrers` blind to level → material, level → mesh, level →
            // terrain and level → cloth all at once: deleting a `.inf_mat` a
            // level binds warned about nothing, and P26.4 is what makes that
            // deletion visible (the surface loses its maps).
            //
            // Read as a TOML *key* rather than as a scene concept, so this crate
            // learns nothing about levels: any payload whose sidecar carries a
            // `dependencies` array of GUID strings gets those edges, whatever
            // else that file is.
            side.dependencies = declared.1;
            side.normalize();
            side
        }
    };
    Ok(Some(AssetEntry {
        sidecar,
        path: normalize(path),
        name,
    }))
}

/// What a sidecar this crate could **not** parse as an [`AssetSidecar`] still
/// declares: its `guid`, and its `dependencies` (P26.4).
///
/// Deliberately schema-blind — it parses the file as a generic TOML table and
/// reads two keys, so `inf-asset` learns nothing about the schemas other crates
/// write. A malformed entry is skipped rather than failing the scan: a sidecar
/// that already failed to parse once has earned no more trust than "take what is
/// legible from it".
fn declared_sidecar(payload_path: &Path) -> (Option<AssetId>, Vec<AssetId>) {
    let Ok(text) = std::fs::read_to_string(crate::sidecar::sidecar_path(payload_path)) else {
        return (None, Vec::new());
    };
    let Ok(table) = text.parse::<toml::Table>() else {
        return (None, Vec::new());
    };
    let id = |v: &toml::Value| v.as_str().and_then(|s| uuid::Uuid::parse_str(s).ok());
    let guid = table.get("guid").and_then(id).map(AssetId);
    let deps = table
        .get("dependencies")
        .and_then(|v| v.as_array())
        .map(|l| l.iter().filter_map(id).map(AssetId).collect())
        .unwrap_or_default();
    (guid, deps)
}

/// Normalize a path for use as a map key: canonicalize when the file exists
/// (resolves `..`, symlinks, and case on Windows), else return it as-is.
fn normalize(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: AssetId, kind: AssetKind, deps: Vec<AssetId>) -> AssetEntry {
        let mut sc = AssetSidecar::new(id, kind, ContentHash::of(id.to_string().as_bytes()));
        sc.dependencies = deps;
        AssetEntry {
            sidecar: sc,
            path: PathBuf::from(format!("/content/{id}.{}", kind.extension().unwrap())),
            name: id.to_string(),
        }
    }

    #[test]
    fn reverse_edges_track_references() {
        let mut db = AssetDb::new("/content");
        let tex = AssetId::new();
        let mat = AssetId::new();
        let mesh = AssetId::new();
        db.insert(entry(tex, AssetKind::Texture, vec![]));
        db.insert(entry(mat, AssetKind::Material, vec![tex]));
        db.insert(entry(mesh, AssetKind::Mesh, vec![mat]));

        // material references texture; mesh references material.
        assert_eq!(db.referenced_by(tex), vec![mat]);
        assert_eq!(db.referenced_by(mat), vec![mesh]);
        assert!(db.referenced_by(mesh).is_empty());
        assert!(db.has_referrers(tex));
        assert!(!db.has_referrers(mesh));
        assert_eq!(db.references_of(mat).unwrap(), &[tex]);
    }

    #[test]
    fn removing_an_asset_clears_its_reverse_edges() {
        let mut db = AssetDb::new("/content");
        let tex = AssetId::new();
        let mat = AssetId::new();
        db.insert(entry(tex, AssetKind::Texture, vec![]));
        db.insert(entry(mat, AssetKind::Material, vec![tex]));
        assert!(db.has_referrers(tex));
        db.remove(mat);
        assert!(!db.has_referrers(tex), "referrer gone → no reverse edge");
    }

    #[test]
    fn set_dependencies_rewires_reverse_edges() {
        let mut db = AssetDb::new("/content");
        let a = AssetId::new();
        let b = AssetId::new();
        let mat = AssetId::new();
        db.insert(entry(a, AssetKind::Texture, vec![]));
        db.insert(entry(b, AssetKind::Texture, vec![]));
        db.insert(entry(mat, AssetKind::Material, vec![a]));
        assert_eq!(db.referenced_by(a), vec![mat]);
        db.set_dependencies(mat, vec![b]).unwrap();
        assert!(db.referenced_by(a).is_empty(), "old edge removed");
        assert_eq!(db.referenced_by(b), vec![mat], "new edge added");
    }

    #[test]
    fn content_hash_index_finds_duplicates() {
        let mut db = AssetDb::new("/content");
        let a = AssetId::new();
        let b = AssetId::new();
        // Force identical hashes.
        let h = ContentHash::of(b"same");
        for id in [a, b] {
            let mut sc = AssetSidecar::new(id, AssetKind::Texture, h);
            sc.normalize();
            db.insert(AssetEntry {
                sidecar: sc,
                path: PathBuf::from(format!("/content/{id}.inf_tex")),
                name: id.to_string(),
            });
        }
        let mut dupes = db.by_content_hash(h);
        dupes.sort();
        let mut want = vec![a, b];
        want.sort();
        assert_eq!(dupes, want);
    }

    #[test]
    fn scan_reads_assets_and_synthesizes_missing_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir_all(root.join("meshes")).unwrap();

        // One asset with a proper sidecar.
        let with = root.join("meshes/Hero.inf_mesh");
        std::fs::write(&with, b"hero-bytes").unwrap();
        let id = AssetId::new();
        AssetSidecar::new(id, AssetKind::Mesh, ContentHash::of(b"hero-bytes"))
            .save(&with)
            .unwrap();

        // One asset with NO sidecar (should be synthesized, not skipped).
        let without = root.join("Loose.inf_tex");
        std::fs::write(&without, b"loose-bytes").unwrap();

        // A non-asset file (ignored).
        std::fs::write(root.join("readme.txt"), b"nope").unwrap();

        let mut db = AssetDb::new(root);
        let n = db.scan().unwrap();
        assert_eq!(n, 2, "two assets, txt ignored");
        assert!(db.get(id).is_some());
        assert!(db.get_by_path(&without).is_some());
    }
}
