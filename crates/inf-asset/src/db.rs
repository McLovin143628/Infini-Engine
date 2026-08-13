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
    /// Two files under the root claiming one GUID, found by the last
    /// [`scan`](Self::scan) (P26.5). See [`IdCollision`].
    collisions: Vec<IdCollision>,
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
        self.collisions.clear();
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
                if let Some(e) = read_entry(&path)? {
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
                    self.insert(e);
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
            "the level's asset id moved when its contents did, so every edge into              it goes stale on the next save"
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
    /// first — a `read_dir` order would have to agree with `sort()` twice by
    /// chance.
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
}
