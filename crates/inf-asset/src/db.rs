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
    /// **A sidecar file is beside this payload and could not be read** — so
    /// [`sidecar`](Self::sidecar) is a stand-in this crate invented, not what is
    /// on disk (C4-39).
    ///
    /// Distinct from "there is no sidecar", which is the ordinary case a
    /// synthesized entry exists for: a payload with no sidecar loses nothing
    /// when the synthesized one is written out, while a payload whose sidecar is
    /// merely *unparseable* still has a real source path, import settings, tags
    /// and dependency list on disk that the stand-in would destroy.
    /// [`AssetDb::persist`] refuses to write over one.
    pub sidecar_unreadable: bool,
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
    /// Two files under the root claiming one GUID, found by the last
    /// [`scan`](Self::scan) (P26.5). See [`IdCollision`].
    collisions: Vec<IdCollision>,
    /// Payloads whose sidecar is present but unreadable, found by the last
    /// [`scan`](Self::scan) (C4-39). Advisory text, ready to show.
    sidecar_advisories: Vec<String>,
}

/// **Two payloads under one content root declaring the same GUID** (P26.5).
///
/// The path P26.4 opened and did not test. `read_entry` prefers a sidecar's own
/// `guid` over the content hash for any payload whose sidecar this crate cannot
/// parse — which it must, because deriving a `.inf_lvl`'s id from its content
/// hash made that id **churn with every save** and every edge into the level go
/// stale. The cost is that two files can now name one asset, and the realistic
/// trigger is not exotic: copy a `.inf_lvl` beside its `.toml` in a file
/// manager.
///
/// # The ruling: FIRST WINS, and the second is reported
///
/// Not last-writer. Three reasons, and the first is the one that decides it:
///
/// * **last-writer was not even deterministic.** `scan_dir` walked `read_dir`,
///   whose order is the filesystem's, so *which* of two colliding files owned
///   the id was a property of the volume. The walk is path-sorted now (the
///   `content_paths_by_guid` rule, which this crate should have had first), and
///   on a sorted walk "first" is a fact about the directory rather than about
///   the disk;
/// * **a copy must not steal an original's identity.** Every edge in the graph
///   names the id, so last-writer silently re-points a level's whole dependency
///   set at a duplicate the author made by accident;
/// * **and it must not vanish, either.** The loser keeps its path and its
///   bytes; what it loses is the id, and it is *named* here so a caller can say
///   so. An asset dropped in silence is the failure mode the synthesized-sidecar
///   rule exists to avoid.
///
/// An **exact** copy — identical bytes and no declared guid — collided before
/// this and still does, at the content-hash id, which is correct: identical
/// content is one asset. This type is for the case where the ids agree and the
/// content does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdCollision {
    /// The contested GUID.
    pub id: AssetId,
    /// The payload that keeps it — the first in the sorted walk.
    pub kept: PathBuf,
    /// The payload that was refused it.
    pub refused: PathBuf,
}

impl std::fmt::Display for IdCollision {
    /// The P16 advisory shape: asset, consequence, remedy.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} and {} both declare asset {} — the second is not registered and \
             nothing can reference it; give it its own guid (or delete the copy)",
            self.kept.display(),
            self.refused.display(),
            self.id.uuid()
        )
    }
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
    ///
    /// Normalizes `entry.path` first, because a caller that built the entry by
    /// hand may hand in a relative or symlinked path and every index here is
    /// keyed on the canonical one. The **scan** path does not go through here —
    /// see [`insert_normalized`](Self::insert_normalized).
    pub fn insert(&mut self, mut entry: AssetEntry) {
        entry.path = normalize(&entry.path);
        self.insert_normalized(entry);
    }

    /// [`insert`](Self::insert) for an entry whose `path` is **already**
    /// canonical (Hardening Wave E).
    ///
    /// `read_entry` — the only producer the scan and the watcher use — already
    /// calls `normalize` when it builds the entry, and `insert` called it a
    /// second time on the result. `normalize` is a `fs::canonicalize`, which on
    /// Windows is a `CreateFileW` + `GetFinalPathNameByHandleW` + `CloseHandle`
    /// per call: **53.8 us measured on this machine**, so a 10 000-asset scan
    /// paid 20 000 file opens where 10 000 do — about **0.54 s of pure
    /// duplicate syscall per full scan**, and a share of it again on every watch
    /// event.
    ///
    /// It is a separate door rather than a flag because "this path is already
    /// canonical" is a fact about the caller, and the one caller that knows it
    /// is the one that made it true two statements earlier.
    fn insert_normalized(&mut self, mut entry: AssetEntry) {
        entry.sidecar.normalize();
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
    ///
    /// **Refuses to overwrite a sidecar this crate could not read** (C4-39). The
    /// in-memory sidecar for such an entry is a stand-in — a fresh id-or-hash,
    /// no source path, no import settings, no tags — and writing it out is how a
    /// TOML syntax error or a merge-conflict marker turned into permanent data
    /// loss, one `set_tags` later. The file stays as it is until a human repairs
    /// it, at which point the next scan reads the real thing.
    pub fn persist(&self, id: AssetId) -> Result<()> {
        let entry = self.by_id.get(&id).ok_or(AssetError::UnknownAsset(id))?;
        if entry.sidecar_unreadable {
            return Err(AssetError::SidecarUnreadable {
                path: crate::sidecar::sidecar_path(&entry.path)
                    .display()
                    .to_string(),
            });
        }
        entry.sidecar.save(&entry.path)
    }

    // ── scanning ──────────────────────────────────────────────────────────

    /// Rebuild the whole index from disk by walking the content root.
    pub fn scan(&mut self) -> Result<usize> {
        self.by_id.clear();
        self.by_path.clear();
        self.by_hash.clear();
        self.reverse.clear();
        self.collisions.clear();
        self.sidecar_advisories.clear();
        let root = self.root.clone();
        let mut count = 0;
        if root.exists() {
            self.scan_dir(&root, &mut count)?;
        }
        Ok(count)
    }

    /// Walk `dir`, **path-sorted** (P26.5).
    ///
    /// The sort is not cosmetic: two payloads can declare one GUID since P26.4
    /// (see [`IdCollision`]), and which of them owns it has to be a fact about
    /// the directory rather than about `read_dir`'s order, which is the
    /// filesystem's. `inf_editor_core::render_assets::content_paths_by_guid`
    /// has sorted for the same reason since P18.3; this crate had not.
    fn scan_dir(&mut self, dir: &Path, count: &mut usize) -> Result<()> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        paths.sort();
        for path in paths {
            if path.is_dir() {
                self.scan_dir(&path, count)?;
            } else if !is_sidecar(&path) {
                let mut advisories = std::mem::take(&mut self.sidecar_advisories);
                let entry = read_entry(&path, &mut advisories);
                self.sidecar_advisories = advisories;
                if let Some(e) = entry? {
                    // FIRST WINS. The incumbent keeps the id and the intruder is
                    // named, rather than the intruder silently re-pointing every
                    // edge in the graph at itself — see `IdCollision`.
                    let id = e.id();
                    if let Some(prior) = self.by_id.get(&id) {
                        if prior.path != e.path {
                            self.collisions.push(IdCollision {
                                id,
                                kept: prior.path.clone(),
                                refused: e.path.clone(),
                            });
                            continue;
                        }
                    }
                    // `read_entry` normalized the path when it built the entry.
                    self.insert_normalized(e);
                    *count += 1;
                }
            }
        }
        Ok(())
    }

    /// GUID collisions the last [`scan`](Self::scan) found — empty on a healthy
    /// content root, which is the state a caller should expect and therefore the
    /// state worth being able to check.
    pub fn collisions(&self) -> &[IdCollision] {
        &self.collisions
    }

    /// Payloads whose sidecar exists but could not be read, found by the last
    /// [`scan`](Self::scan) (C4-39).
    ///
    /// A sibling of [`collisions`](Self::collisions), and it exists for the same
    /// reason: the scan already *knew* something was wrong with the content root
    /// and said nothing, so a TOML syntax error, a merge-conflict marker or a
    /// permission error was indistinguishable from a missing file — right up
    /// until a `persist` wrote the stand-in over the real one.
    pub fn sidecar_advisories(&self) -> &[String] {
        &self.sidecar_advisories
    }

    /// Re-read a single asset at `path` (on a watch event). Returns the id if it
    /// is a recognized, sidecar-bearing asset.
    pub fn rescan_path(&mut self, path: &Path) -> Result<Option<AssetId>> {
        if is_sidecar(path) {
            return Ok(None);
        }
        // Drop any advisory this path already has before re-reading it, so a
        // watcher that fires a hundred times on one broken sidecar leaves one
        // line and not a hundred. The list is then bounded by the number of
        // damaged files, exactly as `collisions` is bounded by the number of
        // colliding ones — a full `scan` clears it outright.
        let mut advisories = std::mem::take(&mut self.sidecar_advisories);
        let prefix = format!("{}: ", path.display());
        advisories.retain(|a| !a.starts_with(&prefix));
        let entry = read_entry(path, &mut advisories);
        self.sidecar_advisories = advisories;
        match entry? {
            Some(e) => {
                let id = e.id();
                // As in `scan`: `read_entry` already normalized this path.
                self.insert_normalized(e);
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
fn read_entry(path: &Path, advisories: &mut Vec<String>) -> Result<Option<AssetEntry>> {
    let kind = AssetKind::from_path(path);
    if kind == AssetKind::Unknown {
        return Ok(None);
    }
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("asset")
        .to_string();

    // **Absent and unreadable are different facts** (C4-39). Both still produce
    // an entry — a payload that vanishes from the Content Drawer because its
    // metadata is damaged is the failure this synthesis exists to avoid — but
    // only the absent case produces one that may be written back out.
    let mut sidecar_unreadable = false;
    let sidecar = match AssetSidecar::load(path) {
        Ok(s) => s,
        Err(e) => {
            if !is_not_found(&e) {
                sidecar_unreadable = true;
                advisories.push(format!(
                    "{}: the sidecar beside it exists but cannot be read ({e}); it is listed \
                     under a stand-in identity and nothing will be written over the file — \
                     repair or delete it, then rescan",
                    path.display()
                ));
            }
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
        sidecar_unreadable,
    }))
}

/// True when this error means "the sidecar is not there" rather than "the
/// sidecar is there and this crate cannot make sense of it".
fn is_not_found(e: &AssetError) -> bool {
    matches!(e, AssetError::Io(io) if io.kind() == std::io::ErrorKind::NotFound)
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

/// Normalize a path for use as a map key.
///
/// # The key must not depend on whether the leaf exists
///
/// This used to be `canonicalize(path).unwrap_or(path)`, which produced **two
/// different keys for one file** depending on the moment it was asked (C4-39).
/// `remove_path` is called by the watcher with the raw path of a file that has
/// *just been deleted* — where `canonicalize` necessarily fails — so it hashed
/// the un-canonicalized path, missed the canonicalized key the scan had
/// inserted, and the entry never left the index: an externally deleted asset
/// kept listing in the Content Drawer and kept counting as a live referrer in
/// `has_referrers`, so nothing it referenced could be deleted either.
///
/// The fix keeps canonicalization — it is what resolves `..`, symlinks and case
/// on Windows, and a purely lexical key would treat `C:\Content` and
/// `c:\content` as two assets — but applies it to the deepest **existing**
/// ancestor and re-joins the rest lexically. A file and its own grave therefore
/// share a key, because the directory they live in is what gets canonicalized.
fn normalize(path: &Path) -> PathBuf {
    if let Ok(canonical) = std::fs::canonicalize(path) {
        return canonical;
    }
    let mut tail = Vec::new();
    let mut cursor = path;
    while let Some(parent) = cursor.parent() {
        // `file_name()` is `None` for `..` and for a root/prefix; neither can be
        // canonicalized away safely, so give up and key the path as written.
        let Some(name) = cursor.file_name() else {
            return path.to_path_buf();
        };
        tail.push(name.to_os_string());
        if let Ok(canonical) = std::fs::canonicalize(parent) {
            let mut out = canonical;
            for name in tail.iter().rev() {
                out.push(name);
            }
            return out;
        }
        cursor = parent;
    }
    path.to_path_buf()
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
            sidecar_unreadable: false,
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
                sidecar_unreadable: false,
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
    /// **A sidecar this crate cannot parse still declares two things** (P26.4):
    /// the asset's own `guid` and its `dependencies` — and one written before the
    /// field existed still scans, keeps its id, and simply has no edges.
    ///
    /// Both behaviours shipped without a unit arm (the P26.4 audit). The gate
    /// that covers them lives in `inf-player`'s `phase26_gate`, needs a cooked
    /// project, and exercises only the PRESENT field: nothing anywhere loaded a
    /// level sidecar written before this batch, and every committed
    /// `.inf_lvl.toml` under `samples/` and `templates/` is exactly that.
    /// Measured: deleting `#[serde(default)]` from the writer's field survived
    /// `-p inf-editor-core` entirely.
    ///
    /// The id half is the defect the batch found rather than a convenience: the
    /// content-hash fallback made a level's asset id churn with its CONTENTS, so
    /// every save re-registered it under a new one and every edge into it went
    /// stale — which is asserted here directly, by rewriting the payload.
    #[test]
    fn an_unparsable_sidecar_still_declares_its_guid_and_its_edges() {
        let dir = tempfile::tempdir().expect("tempdir");
        let level = dir.path().join("A.inf_lvl");
        let guid = AssetId::new();
        let mat = AssetId::new();
        let write = |payload: &[u8], deps: Option<AssetId>| {
            std::fs::write(&level, payload).unwrap();
            let deps = match deps {
                Some(d) => format!(
                    "dependencies = [\"{d}\"]
"
                ),
                None => String::new(),
            };
            // The LEVEL's own schema, which `AssetSidecar::load` cannot parse (no
            // `kind`, and a `content_hash` of a different shape) — the whole
            // reason `read_entry` synthesizes one.
            std::fs::write(
                format!("{}.toml", level.display()),
                format!(
                    "schema_version = 22
guid = \"{guid}\"
title = \"A\"
                     entity_count = 3
content_hash = \"0\"
{deps}"
                ),
            )
            .unwrap();
        };

        write(b"payload-one", Some(mat));
        let mut db = AssetDb::new(dir.path());
        assert_eq!(db.scan().expect("scan"), 1);
        assert!(db.contains(guid), "the declared guid was not honoured");
        assert_eq!(
            db.references_of(guid),
            Some(&[mat][..]),
            "the level's declared edge was not read"
        );
        assert_eq!(
            db.referenced_by(mat),
            vec![guid],
            "the declared edge did not reach the delete-with-references guard"
        );

        // THE ID DOES NOT CHURN WITH THE CONTENT. Before P26.4 it was
        // `Uuid::from_u128(content_hash)`, so every save minted a new one.
        write(b"payload-two-which-is-a-different-length", Some(mat));
        let mut moved = AssetDb::new(dir.path());
        moved.scan().expect("scan");
        assert!(
            moved.contains(guid),
            "the level's asset id moved when its contents did, so every edge into \
             it goes stale on the next save"
        );

        // MIGRATION: a sidecar written before the field existed still scans,
        // keeps its id, and declares no edges — the status quo ante, not an error.
        write(b"payload-three", None);
        let mut old = AssetDb::new(dir.path());
        assert_eq!(old.scan().expect("a field-less sidecar still scans"), 1);
        assert!(old.contains(guid), "the migration lost the declared guid");
        assert!(
            old.referenced_by(mat).is_empty(),
            "a sidecar with no `dependencies` key invented an edge"
        );

        // …and a MALFORMED key is skipped rather than failing the scan, which is
        // what `declared_sidecar`'s doc promises ("a sidecar that already failed
        // to parse once has earned no more trust than 'take what is legible from
        // it'"). Measured: replacing the `as_array` guard with an `expect`
        // survived every other arm in this crate.
        std::fs::write(&level, b"payload-four").unwrap();
        std::fs::write(
            format!("{}.toml", level.display()),
            format!(
                "schema_version = 22\nguid = \"{guid}\"\ntitle = \"A\"\n\
                 entity_count = 3\ncontent_hash = \"0\"\ndependencies = \"nope\"\n"
            ),
        )
        .unwrap();
        let mut bad = AssetDb::new(dir.path());
        assert_eq!(
            bad.scan()
                .expect("a malformed dependencies key still scans"),
            1
        );
        assert!(bad.contains(guid));
        assert!(bad.referenced_by(mat).is_empty());
    }

    /// **An unreadable sidecar is advised and never written over; an absent one
    /// is simply synthesized** (C4-39) — the two arms side by side, because the
    /// defect is that the code had one.
    ///
    /// The unreadable half is where real data lives: a sidecar with a TOML
    /// syntax error still carries the asset's guid, source path, import settings
    /// and tags, and `persist` used to write a freshly-invented stand-in over all
    /// of it the first time anything called `set_tags`.
    #[test]
    fn an_unreadable_sidecar_is_advised_and_never_written_over() {
        // (a) ABSENT: no advisory, and persisting materializes the sidecar.
        let bare = tempfile::tempdir().unwrap();
        let payload = bare.path().join("Loose.inf_mesh");
        std::fs::write(&payload, b"payload").unwrap();
        let mut db = AssetDb::new(bare.path());
        assert_eq!(db.scan().unwrap(), 1);
        assert!(
            db.sidecar_advisories().is_empty(),
            "a missing sidecar is not a damaged one"
        );
        let id = db.iter().next().unwrap().id();
        assert!(!db.get(id).unwrap().sidecar_unreadable);
        db.set_tags(id, vec!["ok".into()]).unwrap();
        db.persist(id)
            .expect("a synthesized sidecar for a payload that has none may be written");
        assert!(crate::sidecar::sidecar_path(&payload).exists());

        // (b) UNREADABLE: advised, flagged, and the file is left exactly alone.
        let damaged = tempfile::tempdir().unwrap();
        let payload = damaged.path().join("Real.inf_mesh");
        std::fs::write(&payload, b"payload").unwrap();
        let side = crate::sidecar::sidecar_path(&payload);
        // A merge-conflict marker: legible enough to exist, not to parse.
        let bytes = b"<<<<<<< HEAD\nschema_version = 1\n=======\n".to_vec();
        std::fs::write(&side, &bytes).unwrap();

        let mut db = AssetDb::new(damaged.path());
        assert_eq!(db.scan().unwrap(), 1, "the payload must stay browsable");
        assert_eq!(
            db.sidecar_advisories().len(),
            1,
            "the scan knew and said nothing: {:?}",
            db.sidecar_advisories()
        );
        assert!(db.sidecar_advisories()[0].contains("Real.inf_mesh"));
        let id = db.iter().next().unwrap().id();
        assert!(db.get(id).unwrap().sidecar_unreadable);

        db.set_tags(id, vec!["tag".into()]).unwrap();
        let err = db
            .persist(id)
            .expect_err("the stand-in must not be written over the real sidecar");
        assert!(matches!(err, AssetError::SidecarUnreadable { .. }), "{err}");
        assert_eq!(
            std::fs::read(&side).unwrap(),
            bytes,
            "the damaged sidecar was modified"
        );

        // A rescan re-raises it (the advisory is a property of the content root,
        // not a one-shot at boot) — and does not stack: a watcher firing
        // repeatedly on one broken sidecar must leave one line, not a hundred.
        for _ in 0..8 {
            db.rescan_path(&payload).unwrap();
        }
        assert_eq!(
            db.sidecar_advisories().len(),
            1,
            "advisories accumulated per watcher event: {:?}",
            db.sidecar_advisories()
        );
    }

    /// **A deleted asset leaves the index** (C4-39, the `by_path` half).
    ///
    /// The watcher calls `remove_path` with the raw path of a file that has just
    /// been deleted, where `canonicalize` necessarily fails — so the key it
    /// computed never matched the one the scan inserted, the entry stayed
    /// forever, and it kept counting as a live referrer in `has_referrers`.
    #[test]
    fn removing_a_file_that_is_already_gone_still_leaves_the_index() {
        let dir = tempfile::tempdir().unwrap();
        // A path with a `.` segment, so the raw and canonical spellings differ
        // in more than existence alone.
        let sub = dir.path().join("Meshes");
        std::fs::create_dir_all(&sub).unwrap();
        let payload = sub.join("Prop.inf_mesh");
        std::fs::write(&payload, b"payload").unwrap();

        let mut db = AssetDb::new(dir.path());
        assert_eq!(db.scan().unwrap(), 1);
        let id = db.iter().next().unwrap().id();

        // Delete first, then report — the watcher's real order.
        std::fs::remove_file(&payload).unwrap();
        std::fs::remove_file(crate::sidecar::sidecar_path(&payload)).ok();
        let indirect = dir.path().join("Meshes").join(".").join("Prop.inf_mesh");
        assert_eq!(
            db.remove_path(&indirect),
            Some(id),
            "an externally deleted asset never left the index"
        );
        assert!(!db.contains(id));
        assert!(db.get_by_path(&payload).is_none());
    }

    /// **Two files claiming one GUID: the first wins, and the second is named**
    /// (P26.5) — the id-collision path the P26.4 audit opened and left untested.
    ///
    /// Its own words: *"Honouring a declared guid opens an id-collision path …
    /// Two payloads declaring one guid therefore collide: `by_id` keeps the last
    /// scanned and `by_path` keeps both paths pointing at it. An exact file copy
    /// already collided before the batch (identical hash → identical id);
    /// same-guid/different-content is new, and untested. The realistic trigger is
    /// a filesystem copy of a `.inf_lvl` beside its `.toml`."*
    ///
    /// The ruling and why it is not last-writer are in [`IdCollision`]. What this
    /// asserts is all three halves of it: the incumbent keeps the id, the
    /// intruder is reported by name, and the answer does not depend on which file
    /// the filesystem hands over first — the copy is written **first** on disk
    /// and named so it sorts **second**, so an unsorted walk and a
    /// last-writer rule each produce a different answer from this one.
    #[test]
    fn two_files_declaring_one_guid_keep_the_first_and_name_the_second() {
        let dir = tempfile::tempdir().unwrap();
        let guid = AssetId(uuid::Uuid::from_u128(0x2605_0000_0000_0001));
        let level_side = |g: AssetId, title: &str| {
            format!(
                "schema_version = 22\nguid = \"{}\"\ntitle = \"{title}\"\n\
                 entity_count = 1\ncontent_hash = \"0\"\n",
                g.uuid()
            )
        };
        // Written in the order a copy really happens — the duplicate exists
        // first here — and named so the ORIGINAL sorts first. An unsorted walk
        // would give either answer depending on the volume.
        let dup = dir.path().join("Zebra copy.inf_lvl");
        std::fs::write(&dup, b"the-copy").unwrap();
        std::fs::write(format!("{}.toml", dup.display()), level_side(guid, "Copy")).unwrap();
        let original = dir.path().join("Alpha.inf_lvl");
        std::fs::write(&original, b"the-original").unwrap();
        std::fs::write(
            format!("{}.toml", original.display()),
            level_side(guid, "Original"),
        )
        .unwrap();

        let mut db = AssetDb::new(dir.path());
        // Only ONE asset is registered, and `scan` counts what it registered.
        assert_eq!(db.scan().unwrap(), 1);
        let entry = db.get(guid).expect("the guid is registered");
        assert!(
            entry.path.ends_with("Alpha.inf_lvl"),
            "the copy took the original's identity: {}",
            entry.path.display()
        );

        // The intruder is NAMED, not swallowed.
        assert_eq!(db.collisions().len(), 1, "{:?}", db.collisions());
        let c = &db.collisions()[0];
        assert_eq!(c.id, guid);
        assert!(c.kept.ends_with("Alpha.inf_lvl"));
        assert!(c.refused.ends_with("Zebra copy.inf_lvl"));
        // The advisory names the asset, the consequence and the remedy — the P16
        // shape, checked rather than assumed.
        let said = c.to_string();
        assert!(said.contains("Alpha.inf_lvl") && said.contains("Zebra copy.inf_lvl"));
        assert!(said.contains(&guid.uuid().to_string()));
        assert!(said.contains("not registered") && said.contains("own guid"));

        // ANTI-VACUITY: a healthy root reports NOTHING, so the list above is a
        // measurement of the collision and not of a scan that always complains.
        let clean = tempfile::tempdir().unwrap();
        let solo = clean.path().join("Alpha.inf_lvl");
        std::fs::write(&solo, b"the-original").unwrap();
        std::fs::write(format!("{}.toml", solo.display()), level_side(guid, "Only")).unwrap();
        let mut ok = AssetDb::new(clean.path());
        assert_eq!(ok.scan().unwrap(), 1);
        assert!(ok.collisions().is_empty());

        // …and an EXACT copy — same bytes, no declared guid — is still ONE asset
        // at the content-hash id, which is correct and is not what this ruling
        // changed. It is reported as a collision all the same, because two paths
        // did claim one id and a caller asking "is my content root healthy" is
        // entitled to know the second file exists.
        let twins = tempfile::tempdir().unwrap();
        std::fs::write(twins.path().join("A.inf_mesh"), b"identical").unwrap();
        std::fs::write(twins.path().join("B.inf_mesh"), b"identical").unwrap();
        let mut t = AssetDb::new(twins.path());
        assert_eq!(t.scan().unwrap(), 1);
        assert_eq!(t.collisions().len(), 1);
        assert!(t.collisions()[0].kept.ends_with("A.inf_mesh"));
    }

    /// **The walk is path-sorted**, which is what makes "first" mean anything.
    ///
    /// Asserted through the collision above rather than by exposing the order:
    /// four files, two pairs, and the winner of each pair is the one that sorts
    /// first.
    ///
    /// # What this arm cannot see, and where the other half is
    ///
    /// The original claim here was that *"a `read_dir` order would have to agree
    /// with `sort()` twice by chance"*. On **NTFS it agrees every time**: a
    /// directory is a name-ordered B-tree and `read_dir` walks it in that order,
    /// so this arm is satisfied by an unsorted walk on Windows — measured, by
    /// deleting `paths.sort()` and watching every `inf-asset` arm stay green.
    /// ext4's htree hands back hash order and APFS insertion order, so the
    /// Linux CI leg does bite; the developer machine and the Windows leg do not.
    /// That is the P25 one-platform law turned around: not a bound that reddens
    /// CI on one platform, but a gate that goes *vacuous* on one.
    ///
    /// So the rule has a source pin beside it
    /// ([`the_scan_walk_sorts_before_it_reads`]), on the `projector_mirror`
    /// pattern for a rule the local machine cannot execute against.
    #[test]
    fn the_scan_walk_is_path_sorted() {
        let dir = tempfile::tempdir().unwrap();
        let side = |g: AssetId| {
            format!(
                "schema_version = 22\nguid = \"{}\"\ntitle = \"T\"\n\
                 entity_count = 1\ncontent_hash = \"0\"\n",
                g.uuid()
            )
        };
        let mut want = Vec::new();
        for (n, (first, second)) in [("aaa", "zzz"), ("bbb", "yyy")].iter().enumerate() {
            let g = AssetId(uuid::Uuid::from_u128(0x2605_0001_0000_0000 + n as u128));
            for name in [second, first] {
                let p = dir.path().join(format!("{name}.inf_lvl"));
                std::fs::write(&p, format!("payload-{name}")).unwrap();
                std::fs::write(format!("{}.toml", p.display()), side(g)).unwrap();
            }
            want.push((g, format!("{first}.inf_lvl")));
        }
        let mut db = AssetDb::new(dir.path());
        assert_eq!(db.scan().unwrap(), 2);
        assert_eq!(db.collisions().len(), 2);
        for (g, first) in want {
            assert!(
                db.get(g).expect("registered").path.ends_with(&first),
                "{g:?} went to the wrong file"
            );
        }
    }

    /// **`scan_dir` sorts before it reads** — the half of the ruling above that
    /// a name-ordered filesystem hides (P26.5 audit).
    ///
    /// `IdCollision`'s whole argument for *first wins* is that "first" is a fact
    /// about the directory and not about the volume, and the behavioural arm
    /// above cannot falsify the sort on NTFS. This can: it reads the function's
    /// own source, comment-stripped (the P26.1 finding that a claim in a comment
    /// satisfied a gate whose message said the opposite), and CRLF-normalized
    /// (the P22 law — `.rs` is read by tests).
    ///
    /// A source pin rather than a behavioural one because the behaviour is the
    /// filesystem's, and this machine's filesystem always agrees with the
    /// answer. It is the same shape as `projector_mirror`'s pins on the viewport
    /// host, and for the same reason: the rule is real, the local machine cannot
    /// be made to break it.
    #[test]
    fn the_scan_walk_sorts_before_it_reads() {
        let src = include_str!("db.rs").replace("\r\n", "\n");
        let body = src
            .split_once("fn scan_dir(")
            .expect("`scan_dir` moved; this gate scopes on its name")
            .1
            .split_once("\n    }\n")
            .expect("`scan_dir` has no end")
            .0;
        let code: String = body
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains("paths.sort()"),
            "`scan_dir` does not sort its entries, so which of two payloads \
             declaring one guid keeps it is the filesystem's answer and not the \
             directory's — invisible on NTFS, which enumerates in name order \
             anyway:\n{code}"
        );
        // …and the sort happens BEFORE anything is read, or it is a sort of a
        // list nobody walks.
        let sort_at = code.find("paths.sort()").expect("checked above");
        let read_at = code
            .find("read_entry(")
            .expect("`scan_dir` no longer reads entries");
        assert!(
            sort_at < read_at,
            "`scan_dir` reads entries before sorting them"
        );
        // ANTI-VACUITY: the strip left the code, not only the comments.
        assert!(
            code.contains("self.scan_dir("),
            "the source filter ate the function this gate reads"
        );
    }
}
