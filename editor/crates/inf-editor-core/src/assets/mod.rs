//! The editor's asset project: the live [`AssetDb`] over a content root, plus
//! the import orchestrator ([`import`]) and the async import queue ([`queue`]).
//!
//! This is the Ring-1 layer that ties the Ring-0 pieces together — it owns the
//! database, writes assets in the dual-format (payload + sidecar), maintains the
//! dependency edges the importer discovers, and enforces the
//! delete-with-references safety check that is the Phase 4 gate.

pub mod biome_set;
pub mod collections;
pub mod data;
pub mod import;
pub mod material_instance;
pub mod queue;
pub mod snapshot;
pub mod sprite_sheet;
pub mod table_import;
pub mod terrain_import;
pub mod vmesh;

use std::path::{Path, PathBuf};

use inf_asset::{
    AssetDb, AssetEntry, AssetError, AssetId, AssetKind, AssetPayload, AssetSidecar, ContentHash,
    ImportCache, Result,
};

pub use import::ImportOutcome;
pub use queue::{ImportProgress, ImportQueue};
pub use terrain_import::{TerrainImportOutcome, TerrainImportSettings, TERRAIN_IMPORT_FOLDER};
pub use vmesh::{ensure_vmesh, VmeshDerivation};

/// A content project rooted at a directory: the asset database + import cache.
pub struct AssetProject {
    root: PathBuf,
    db: AssetDb,
    cache: ImportCache,
    /// Monotonic content version — bumped by every mutation so the frontend can
    /// detect a change and re-fetch the snapshot.
    version: u64,
}

impl AssetProject {
    /// Open (scanning) the project at `root`, creating the content dir and the
    /// import cache under `<root>/.inf/import-cache` if needed.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        let mut db = AssetDb::new(&root);
        db.scan()?;
        let cache = ImportCache::open(root.join(".inf").join("import-cache"))?;
        Ok(Self {
            root,
            db,
            cache,
            version: 1,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn db(&self) -> &AssetDb {
        &self.db
    }
    pub fn db_mut(&mut self) -> &mut AssetDb {
        &mut self.db
    }

    /// The current content version.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Bump the content version (called by every mutation).
    pub fn bump(&mut self) {
        self.version += 1;
    }

    /// Rescan the whole content root from disk.
    pub fn rescan(&mut self) -> Result<usize> {
        let n = self.db.scan()?;
        self.bump();
        Ok(n)
    }

    /// Absolute default import destination folder (created if missing).
    pub fn content_dir(&self, sub: &str) -> Result<PathBuf> {
        let dir = self.root.join(sub);
        std::fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    // ── writing assets (dual-format) ──────────────────────────────────────

    /// Write a payload + sidecar under `dir`, register it, and return its id.
    /// `dir` should be inside the content root.
    pub fn write_asset<T: AssetPayload>(
        &mut self,
        dir: &Path,
        name: &str,
        payload: &T,
        source: Option<String>,
        dependencies: Vec<AssetId>,
        import: Option<toml::Table>,
    ) -> Result<AssetId> {
        let bytes = inf_asset::encode(payload)?;
        let hash = ContentHash::of(&bytes);
        let id = AssetId::new();
        let ext = T::KIND.extension().expect("payload kinds have extensions");
        let path = unique_path(dir, name, ext)?;
        std::fs::create_dir_all(dir)?;
        std::fs::write(&path, &bytes)?;

        let mut sidecar = AssetSidecar::new(id, T::KIND, hash);
        sidecar.source = source;
        sidecar.dependencies = dependencies;
        sidecar.import = import;
        sidecar.save(&path)?;

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(name)
            .to_string();
        self.db.insert(AssetEntry {
            sidecar,
            path,
            name,
        });
        self.bump();
        Ok(id)
    }

    /// Register an asset whose payload a **specialized writer** already put on
    /// disk, instead of the generic dual-format [`write_asset`](Self::write_asset).
    ///
    /// Exactly one kind needs this today: `.inf_terrain` (P16.3/P16.4). Its
    /// payload is a raw, 16-byte-aligned image that must go through
    /// [`inf_terrain::write_terrain_asset`] — the generic path would frame it with
    /// a bincode length prefix and silently knock every tile off its alignment, so
    /// `TerrainAsset` deliberately implements neither `AssetPayload` nor `serde`
    /// and the generic door fails to compile. This is the matching door: the
    /// caller writes the bytes, hands us their hash, and the database gets its
    /// sidecar exactly as it would for any other asset.
    ///
    /// Pass `reuse` to keep an existing GUID (a reimport rewriting its own file).
    ///
    /// # The half-written-pair problem
    ///
    /// The payload is already on disk when this is called, so a failure to write
    /// the sidecar would leave a payload **with no sidecar** — which the content
    /// watcher later promotes into the database under a *freshly minted* GUID, so
    /// the level's `Terrain.asset` reference silently dangles. Two things prevent
    /// that: the sidecar goes down temp + rename (never half-written, and a
    /// reimport's existing sidecar survives a failure intact), and if it fails for
    /// a **new** asset the payload is removed again, so the pair is all-or-nothing.
    ///
    /// A failed *reimport* keeps its payload — the previous bytes are already gone
    /// and its sidecar still names the right GUID; only `content_hash` is stale
    /// until the next successful reimport, which is a re-derivable inconsistency
    /// rather than a dangling reference.
    pub fn register_written_asset(
        &mut self,
        path: PathBuf,
        kind: AssetKind,
        content_hash: ContentHash,
        source: Option<String>,
        import: Option<toml::Table>,
        reuse: Option<AssetId>,
    ) -> Result<AssetId> {
        let id = reuse.unwrap_or_default();
        let mut sidecar = AssetSidecar::new(id, kind, content_hash);
        sidecar.source = source;
        sidecar.import = import;
        if let Some(existing) = reuse.and_then(|r| self.db.get(r)) {
            sidecar.tags = existing.sidecar.tags.clone();
        }
        if let Err(e) = save_sidecar_atomically(&sidecar, &path) {
            if reuse.is_none() {
                let _ = std::fs::remove_file(&path);
            }
            return Err(e);
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Asset")
            .to_string();
        self.db.insert(AssetEntry {
            sidecar,
            path,
            name,
        });
        self.bump();
        Ok(id)
    }

    /// A collision-free payload path under `dir` — the same naming the generic
    /// writer uses, exposed for [`register_written_asset`](Self::register_written_asset).
    pub fn unique_asset_path(&self, dir: &Path, name: &str, ext: &str) -> Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        unique_path(dir, name, ext)
    }

    // ── mutation ──────────────────────────────────────────────────────────

    /// Rename an asset (moves both payload + sidecar on disk).
    pub fn rename(&mut self, id: AssetId, new_name: &str) -> Result<()> {
        let entry = self.db.get(id).ok_or(AssetError::UnknownAsset(id))?;
        let old_path = entry.path.clone();
        let ext = entry.kind().extension().unwrap_or("bin");
        let dir = old_path.parent().unwrap_or(Path::new("."));
        let new_path = unique_path(dir, new_name, ext)?;

        std::fs::rename(&old_path, &new_path)?;
        let old_side = inf_asset::sidecar_path(&old_path);
        if old_side.exists() {
            std::fs::rename(old_side, inf_asset::sidecar_path(&new_path))?;
        }
        // Re-register under the new path.
        let mut entry = self.db.remove(id).unwrap();
        entry.path = new_path.clone();
        entry.name = new_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(new_name)
            .to_string();
        self.db.insert(entry);
        self.bump();
        Ok(())
    }

    /// Duplicate an asset (fresh GUID, same payload + dependencies).
    pub fn duplicate(&mut self, id: AssetId) -> Result<AssetId> {
        let entry = self.db.get(id).ok_or(AssetError::UnknownAsset(id))?;
        let bytes = std::fs::read(&entry.path)?;
        let hash = ContentHash::of(&bytes);
        let kind = entry.kind();
        let ext = kind.extension().unwrap_or("bin");
        let dir = entry.path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let base = format!("{}_Copy", entry.name);
        let deps = entry.sidecar.dependencies.clone();
        let source = entry.sidecar.source.clone();
        let import = entry.sidecar.import.clone();

        let new_id = AssetId::new();
        let path = unique_path(&dir, &base, ext)?;
        std::fs::write(&path, &bytes)?;
        let mut sidecar = AssetSidecar::new(new_id, kind, hash);
        sidecar.dependencies = deps;
        sidecar.source = source;
        sidecar.import = import;
        sidecar.save(&path)?;
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&base)
            .to_string();
        self.db.insert(AssetEntry {
            sidecar,
            path,
            name,
        });
        self.bump();
        Ok(new_id)
    }

    /// Replace an asset's sidecar metadata in the DB and on disk, re-deriving
    /// dependency edges from it. Bumps the content version. Used by the
    /// sprite-sheet slicer to persist slices into the `import` table without
    /// touching the payload (P8.2a).
    pub fn replace_sidecar(&mut self, id: AssetId, sidecar: AssetSidecar) -> Result<()> {
        let entry = self.db.get(id).ok_or(AssetError::UnknownAsset(id))?;
        let path = entry.path.clone();
        let name = entry.name.clone();
        self.db.insert(AssetEntry {
            sidecar,
            path,
            name,
        });
        self.db.persist(id)?;
        self.bump();
        Ok(())
    }

    /// Set an asset's tags (persisted).
    pub fn set_tags(&mut self, id: AssetId, tags: Vec<String>) -> Result<()> {
        self.db.set_tags(id, tags)?;
        self.db.persist(id)?;
        self.bump();
        Ok(())
    }

    /// Delete an asset. Unless `force`, refuses (with the list of referrers)
    /// when other assets still reference it — the delete-with-references guard.
    ///
    /// Deleting a `.inf_mesh` also removes its **editor-derived** `.inf_vmesh`
    /// (P18.3). That artifact carries no dependency edge on purpose — an edge
    /// would make every mesh undeletable, since its own derived form would count
    /// as a referrer — so nothing else would ever collect it, and the viewport
    /// resolves a mesh's DAG by *computing* its id, which means a surviving
    /// artifact keeps drawing geometry for an asset the user just deleted.
    pub fn delete(&mut self, id: AssetId, force: bool) -> Result<Vec<AssetId>> {
        let referrers = self.db.referenced_by(id);
        if !force && !referrers.is_empty() {
            return Ok(referrers); // caller warns; nothing deleted
        }
        let was_mesh = self.db.get(id).map(|e| e.kind()) == Some(AssetKind::Mesh);
        if let Some(entry) = self.db.remove(id) {
            let _ = std::fs::remove_file(&entry.path);
            let side = inf_asset::sidecar_path(&entry.path);
            let _ = std::fs::remove_file(side);
            self.bump();
        }
        if was_mesh {
            vmesh::remove_derived_vmesh(self, id);
        }
        Ok(Vec::new()) // empty = deleted
    }

    /// The assets that reference `id` (reverse deps) — the warning payload.
    pub fn referenced_by(&self, id: AssetId) -> Vec<AssetId> {
        self.db.referenced_by(id)
    }

    // ── payload read / rewrite (data-asset editing, P4.5) ─────────────────

    /// Decode an existing asset's payload as `T`.
    pub fn load_payload<T: AssetPayload>(&self, id: AssetId) -> Result<T> {
        let entry = self.db.get(id).ok_or(AssetError::UnknownAsset(id))?;
        let bytes = std::fs::read(&entry.path)?;
        inf_asset::decode(&bytes)
    }

    /// Rewrite an existing asset's payload in place (data-asset editors save
    /// through this), updating the content hash + dependency edges.
    pub fn rewrite_payload<T: AssetPayload>(
        &mut self,
        id: AssetId,
        payload: &T,
        dependencies: Vec<AssetId>,
    ) -> Result<()> {
        let path = self
            .db
            .get(id)
            .ok_or(AssetError::UnknownAsset(id))?
            .path
            .clone();
        let bytes = inf_asset::encode(payload)?;
        let hash = ContentHash::of(&bytes);
        std::fs::write(&path, &bytes)?;

        let name = self.db.get(id).unwrap().name.clone();
        let mut sidecar = self.db.get(id).unwrap().sidecar.clone();
        sidecar.content_hash = hash;
        sidecar.dependencies = dependencies;
        sidecar.save(&path)?;
        self.db.insert(AssetEntry {
            sidecar,
            path,
            name,
        });
        self.bump();
        Ok(())
    }

    // ── import (delegates to the orchestrator, using the cache) ───────────

    /// Import one external source file into `dest_dir`.
    pub fn import_file(&mut self, source: &Path, dest_dir: &Path) -> Result<ImportOutcome> {
        import::import_file(self, source, dest_dir)
    }

    pub(crate) fn cache_mut(&mut self) -> &mut ImportCache {
        &mut self.cache
    }
}

/// Write `sidecar` beside `payload_path` **atomically** (temp file + rename),
/// the same discipline `inf_terrain::write_terrain_asset` and
/// `PackWriter::write_to_file` use.
///
/// A plain `fs::write` can leave a truncated TOML behind on a full disk or a
/// crash, and truncating an existing sidecar is worse than not writing one: the
/// asset's GUID would be lost while its payload stayed. Renaming over the target
/// makes the sidecar either wholly old or wholly new. A failed attempt cleans up
/// its own temp file rather than leaving litter in the content root.
fn save_sidecar_atomically(sidecar: &AssetSidecar, payload_path: &Path) -> Result<()> {
    if let Some(parent) = payload_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let final_path = inf_asset::sidecar_path(payload_path);
    let tmp = {
        let mut s = final_path.as_os_str().to_os_string();
        s.push(format!(".{}.tmp", std::process::id()));
        PathBuf::from(s)
    };
    let text = sidecar.to_toml()?;
    if let Err(e) = std::fs::write(&tmp, text).and_then(|()| std::fs::rename(&tmp, &final_path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(())
}

/// A collision-free payload path: `<dir>/<name>.<ext>`, `_1`, `_2`, …
fn unique_path(dir: &Path, name: &str, ext: &str) -> Result<PathBuf> {
    let safe = sanitize(name);
    let mut candidate = dir.join(format!("{safe}.{ext}"));
    let mut n = 1;
    while candidate.exists() {
        candidate = dir.join(format!("{safe}_{n}.{ext}"));
        n += 1;
    }
    Ok(candidate)
}

/// Reduce a display name to a filesystem-safe stem.
fn sanitize(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = s.trim_matches('_');
    if trimmed.is_empty() {
        "Asset".to_string()
    } else {
        trimmed.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_material::MaterialAsset;

    #[test]
    fn write_delete_with_reference_guard() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let mats = proj.content_dir("materials").unwrap();

        // A texture and a material that references it.
        let tex = inf_material::texture_from_rgba8(vec![255; 4 * 4 * 4], 4, 4, Default::default())
            .unwrap();
        let tex_id = proj
            .write_asset(&mats, "Albedo", &tex, None, vec![], None)
            .unwrap();
        let mat = MaterialAsset {
            base_color_texture: Some(tex_id),
            ..Default::default()
        };
        let mat_id = proj
            .write_asset(&mats, "Mat", &mat, None, mat.texture_dependencies(), None)
            .unwrap();

        // Deleting the texture is refused (material references it).
        let blockers = proj.delete(tex_id, false).unwrap();
        assert_eq!(blockers, vec![mat_id], "delete-with-references warns");
        assert!(proj.db().contains(tex_id), "not actually deleted");

        // Delete the material first, then the texture is free.
        assert!(proj.delete(mat_id, false).unwrap().is_empty());
        assert!(proj.delete(tex_id, false).unwrap().is_empty());
        assert!(!proj.db().contains(tex_id));
    }

    #[test]
    fn rename_moves_payload_and_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let d = proj.content_dir("m").unwrap();
        let mat = MaterialAsset::default();
        let id = proj
            .write_asset(&d, "Old", &mat, None, vec![], None)
            .unwrap();
        proj.rename(id, "New").unwrap();
        let e = proj.db().get(id).unwrap();
        assert_eq!(e.name, "New");
        assert!(e.path.exists());
        assert!(inf_asset::sidecar_path(&e.path).exists());
    }

    /// A payload with no sidecar is worse than no asset at all: the content
    /// watcher later promotes it under a **freshly minted** GUID, so any level
    /// that referenced the intended one dangles silently. The pair must therefore
    /// be all-or-nothing (P16.4a audit).
    #[test]
    fn a_failed_sidecar_write_leaves_no_orphan_payload() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let content = proj.content_dir("Terrain").unwrap();
        let payload = content.join("World.inf_terrain");
        std::fs::write(&payload, b"payload bytes").unwrap();

        // Make the sidecar write fail by parking a DIRECTORY where its file goes.
        let side = inf_asset::sidecar_path(&payload);
        std::fs::create_dir_all(&side).unwrap();

        let err = proj
            .register_written_asset(
                payload.clone(),
                AssetKind::Terrain,
                ContentHash::of(b"payload bytes"),
                None,
                None,
                None,
            )
            .unwrap_err();
        assert!(!err.to_string().is_empty());
        assert!(
            !payload.exists(),
            "the payload survived a failed sidecar write — it would be adopted \
             under a random GUID by the watcher"
        );
        assert!(proj.db().is_empty(), "nothing may be registered");
        // No temp litter beside the asset either.
        let strays: Vec<_> = std::fs::read_dir(&content)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
            .collect();
        assert!(strays.is_empty(), "left temp files: {strays:?}");
    }

    /// The sidecar goes down temp + rename, so a **reimport** whose sidecar write
    /// fails keeps the previous, still-valid sidecar rather than a truncated one —
    /// the GUID is never lost.
    #[test]
    fn a_failed_reimport_sidecar_write_keeps_the_previous_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let content = proj.content_dir("Terrain").unwrap();
        let payload = content.join("World.inf_terrain");
        std::fs::write(&payload, b"v1").unwrap();
        let id = proj
            .register_written_asset(
                payload.clone(),
                AssetKind::Terrain,
                ContentHash::of(b"v1"),
                None,
                None,
                None,
            )
            .unwrap();
        let side = inf_asset::sidecar_path(&payload);
        let original = std::fs::read_to_string(&side).unwrap();

        // Now make the *rename* fail by replacing the sidecar with a directory.
        std::fs::remove_file(&side).unwrap();
        std::fs::create_dir_all(&side).unwrap();
        std::fs::write(&payload, b"v2").unwrap();
        assert!(proj
            .register_written_asset(
                payload.clone(),
                AssetKind::Terrain,
                ContentHash::of(b"v2"),
                None,
                None,
                Some(id),
            )
            .is_err());
        // A reimport keeps its payload (the old bytes are already gone) and the
        // database still knows the asset under its original GUID.
        assert!(payload.exists());
        assert!(proj.db().contains(id));
        // Put the real sidecar back and confirm it was never truncated.
        std::fs::remove_dir(&side).unwrap();
        std::fs::write(&side, &original).unwrap();
        assert_eq!(
            inf_asset::AssetSidecar::load(&payload).unwrap().guid,
            id,
            "the GUID must survive a failed sidecar write"
        );
    }

    #[test]
    fn reopen_rescans_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let id = {
            let mut proj = AssetProject::open(dir.path()).unwrap();
            let d = proj.content_dir("m").unwrap();
            proj.write_asset(&d, "Keep", &MaterialAsset::default(), None, vec![], None)
                .unwrap()
        };
        // Fresh open scans the sidecar back in.
        let proj = AssetProject::open(dir.path()).unwrap();
        assert!(proj.db().contains(id), "persisted asset reloads");
    }
}
