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
        report_sidecar_advisories(&db);
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
        report_sidecar_advisories(&self.db);
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
    ///
    /// # `dir` is taken VERBATIM
    ///
    /// It is **not** joined onto [`root`](Self::root) — it is passed to
    /// `std::fs::create_dir_all` and `unique_path` as given. So a caller must
    /// hand over an absolute path (or one already rooted with
    /// [`content_dir`](Self::content_dir)); a relative `dir` resolves against
    /// the *process's* working directory, and `""` writes into it. That is not
    /// hypothetical: the first run of P25.3's photogrammetry gate left five
    /// `Scan.inf_*` files in the crate it lives in. Written here rather than in
    /// the report that found it, because this is where a caller meets the
    /// parameter.
    ///
    /// # The half-written pair, here too
    ///
    /// The payload lands before the sidecar, so a sidecar that cannot be written
    /// would leave a payload with **no sidecar** — which the content watcher
    /// later promotes into the database under a *freshly minted* GUID, so every
    /// reference the caller was about to write points at nothing. That is the
    /// hazard [`register_written_asset`](Self::register_written_asset) has
    /// spelled out since P16.3, and the rule is the same one: if the sidecar
    /// fails, the payload is removed again, so the pair is all-or-nothing and a
    /// failed write leaves the content root as it found it.
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
        // Atomic (C4-27): a crash between a plain payload write and its sidecar
        // leaves a *truncated* payload that the next scan promotes into the
        // database with a synthesized sidecar — an asset that looks legitimate
        // and decodes as `schema_version = 0` with every field zero.
        inf_asset::write_atomically(&path, &bytes)?;

        let mut sidecar = AssetSidecar::new(id, T::KIND, hash);
        sidecar.source = source;
        sidecar.dependencies = dependencies;
        sidecar.import = import;
        if let Err(e) = sidecar.save(&path) {
            let _ = std::fs::remove_file(&path);
            return Err(e);
        }

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(name)
            .to_string();
        self.db.insert(AssetEntry {
            sidecar,
            path,
            name,
            // The sidecar was written by this call, so it is on disk and legible.
            sidecar_unreadable: false,
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
    /// Pass `reuse` to keep an existing GUID (a reimport rewriting its own file,
    /// or a derivation whose id *is* a function of its source). Without it a
    /// fresh GUID is minted, exactly as [`write_asset`](Self::write_asset) does
    /// — this used to be `reuse.unwrap_or_default()`, and `AssetId::default()`
    /// is `AssetId::NIL`, the "no asset" sentinel. Every asset written through
    /// this door with no `reuse` therefore shared one id: the second one to
    /// arrive replaced the first in the database and left its payload orphaned
    /// on disk. It stayed invisible while `.inf_terrain` was the only such
    /// caller, because a project has one terrain and a reimport passes `reuse`;
    /// P26.1 routed every imported *texture* through here, and a glTF brings
    /// several at once.
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
        // Spelled as a `match` on purpose. `reuse.unwrap_or_else(AssetId::new)`
        // is what this wants to say, and `clippy::unwrap_or_default` rewrites it
        // to `unwrap_or_default()` — because the lint assumes `T::new()` and
        // `T::default()` agree, which for `AssetId` is precisely the assumption
        // that produced the defect: `default()` is the NIL "no asset" sentinel
        // and `new()` mints a v4 UUID. Taking clippy's advice here reintroduces
        // it.
        let id = match reuse {
            Some(existing) => existing,
            None => AssetId::new(),
        };
        let mut sidecar = AssetSidecar::new(id, kind, content_hash);
        sidecar.source = source;
        sidecar.import = import;
        if let Some(existing) = reuse.and_then(|r| self.db.get(r)) {
            sidecar.tags = existing.sidecar.tags.clone();
        }
        if let Err(e) = sidecar.save(&path) {
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
            // The sidecar was written by this call, so it is on disk and legible.
            sidecar_unreadable: false,
        });
        self.bump();
        Ok(id)
    }

    /// Write `payload` to an **exact** path, overwriting whatever asset is
    /// registered there and keeping its GUID, or creating a new one.
    ///
    /// The door for an editor whose save target is a *deterministic file name*
    /// rather than a fresh unique one — the graph editors, which slug the
    /// document's name into `<Content>/PCG/<Name>.inf_pcg` and re-save over it
    /// (C4-21). Those two saves used to be a bare `fs::write` into the content
    /// root: non-atomic, and with **no sidecar at all**, so the payload was left
    /// for the watcher to promote under a *synthesized* id derived from its
    /// content hash — an asset GUID that churned with every edit, so every
    /// reference into the graph went stale on each save.
    ///
    /// [`write_asset`](Self::write_asset) cannot serve them: it calls
    /// `unique_path`, so the second save of "Forest" would land in `Forest_1`.
    pub fn write_asset_at<T: AssetPayload>(
        &mut self,
        path: &Path,
        payload: &T,
        dependencies: Vec<AssetId>,
    ) -> Result<AssetId> {
        // A different *kind* already registered at this exact path cannot be
        // rewritten in place — the payload codec differs — so it is evicted
        // rather than left in `by_id` pointing at bytes that are no longer its
        // own. (`by_path` would be re-keyed by the insert below either way; the
        // stale `by_id` entry is what would linger.)
        match self.db.get_by_path(path).map(|e| (e.id(), e.kind())) {
            Some((id, kind)) if kind == T::KIND => {
                self.rewrite_payload(id, payload, dependencies)?;
                return Ok(id);
            }
            Some((id, _)) => {
                self.db.remove(id);
            }
            None => {}
        }
        let bytes = inf_asset::encode(payload)?;
        let hash = ContentHash::of(&bytes);
        let id = AssetId::new();
        inf_asset::write_atomically(path, &bytes)?;
        let mut sidecar = AssetSidecar::new(id, T::KIND, hash);
        sidecar.dependencies = dependencies;
        if let Err(e) = sidecar.save(path) {
            let _ = std::fs::remove_file(path);
            return Err(e);
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("Asset")
            .to_string();
        self.db.insert(AssetEntry {
            sidecar,
            path: path.to_path_buf(),
            name,
            // The sidecar was written by this call, so it is on disk and legible.
            sidecar_unreadable: false,
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
    ///
    /// # The sidecar moves first, and a failure rolls it back
    ///
    /// The pair used to move payload-then-sidecar (C4-22/F10), so a failure or a
    /// crash between the two left the payload at the new path with **no
    /// sidecar** and an orphan sidecar at the old one — the exact state this
    /// module documents as the GUID-churn hazard at `write_asset` and
    /// `register_written_asset`: the watcher promotes the sidecar-less payload
    /// under a freshly minted id and every edge into the asset dangles.
    ///
    /// Moving the sidecar first inverts the exposure into the harmless one — a
    /// sidecar at the new path with no payload beside it is not an asset and no
    /// scan adopts it — and a failed payload move puts the sidecar back, so the
    /// on-disk pair is always whole at one path or the other.
    ///
    /// The old `exists()` pre-check is gone with it: it was a TOCTOU window, and
    /// `NotFound` from the rename says the same thing without one.
    pub fn rename(&mut self, id: AssetId, new_name: &str) -> Result<()> {
        let entry = self.db.get(id).ok_or(AssetError::UnknownAsset(id))?;
        let old_path = entry.path.clone();
        let ext = entry.kind().extension().unwrap_or("bin");
        let dir = old_path.parent().unwrap_or(Path::new("."));
        let new_path = unique_path(dir, new_name, ext)?;

        let old_side = inf_asset::sidecar_path(&old_path);
        let new_side = inf_asset::sidecar_path(&new_path);
        let moved_sidecar = match std::fs::rename(&old_side, &new_side) {
            Ok(()) => true,
            // A payload with no sidecar is a state this database already handles
            // (it synthesizes one), so an absent sidecar is not a reason to
            // refuse the rename — but any other error is.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
            Err(e) => return Err(e.into()),
        };
        if let Err(e) = std::fs::rename(&old_path, &new_path) {
            if moved_sidecar {
                let _ = std::fs::rename(&new_side, &old_side);
            }
            return Err(e.into());
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
        inf_asset::write_atomically(&path, &bytes)?; // C4-27, as `write_asset`
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
            // The sidecar was written by this call, so it is on disk and legible.
            sidecar_unreadable: false,
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
        // The flag rides with the entry, not with the metadata being written
        // into it: an asset whose sidecar on disk is unreadable still is one,
        // and `persist` below is exactly the call that must refuse (C4-39).
        let sidecar_unreadable = entry.sidecar_unreadable;
        self.db.insert(AssetEntry {
            sidecar,
            path,
            name,
            sidecar_unreadable,
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
            // A removal that fails is reported, not swallowed (C4-45): the
            // command answers "deleted" while the payload is still on disk, so
            // the next scan re-registers it — under a *synthesized* sidecar if
            // the sidecar half is the one that went — and the asset reappears
            // with a different GUID than every reference to it names.
            for path in [entry.path.clone(), inf_asset::sidecar_path(&entry.path)] {
                if let Err(e) = std::fs::remove_file(&path) {
                    if e.kind() != std::io::ErrorKind::NotFound {
                        tracing::error!(
                            "deleting asset {id}: could not remove {} ({e}); it is out of the \
                             database but still on disk, and the next content scan will \
                             re-register it",
                            path.display()
                        );
                    }
                }
            }
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
    ///
    /// **Not for `.inf_tex`** — a texture payload has two container versions
    /// since P26.1 and only one of them is bincode. Use
    /// [`load_texture`](Self::load_texture), which sniffs.
    pub fn load_payload<T: AssetPayload>(&self, id: AssetId) -> Result<T> {
        let entry = self.db.get(id).ok_or(AssetError::UnknownAsset(id))?;
        let bytes = std::fs::read(&entry.path)?;
        inf_asset::decode(&bytes)
    }

    /// Decode a `.inf_tex` payload of **either container version** (P26.1) into
    /// the whole-levels record — the door every consumer that wants mips goes
    /// through, in place of [`load_payload`](Self::load_payload).
    ///
    /// A consumer that wants one *tile* at a time reads the bytes and builds an
    /// `inf_material::TiledTextureReader` instead; that is what P26.2's residency
    /// does.
    pub fn load_texture(&self, id: AssetId) -> Result<inf_material::TextureAsset> {
        let entry = self.db.get(id).ok_or(AssetError::UnknownAsset(id))?;
        let bytes = std::fs::read(&entry.path)?;
        inf_material::TextureAsset::from_payload(&bytes)
            .map_err(|e| AssetError::Import(e.to_string()))
    }

    /// Write a **v2 tiled** `.inf_tex` image (P26.1) and register it, the way
    /// [`write_asset`](Self::write_asset) writes a bincode payload.
    ///
    /// A tiled image must never go through the generic door: `inf_asset::encode`
    /// would frame it with a bincode length prefix and knock every tile off its
    /// 16-byte boundary, silently. `TiledTextureImage` implements no
    /// `AssetPayload`, so that mistake does not compile — this is the matching
    /// writer, on the `.inf_terrain` pattern
    /// ([`register_written_asset`](Self::register_written_asset)).
    ///
    /// A texture has no dependencies, which is why this takes none.
    pub fn write_tiled_texture(
        &mut self,
        dir: &Path,
        name: &str,
        image: &inf_material::TiledTextureImage,
        source: Option<String>,
        import: Option<toml::Table>,
    ) -> Result<AssetId> {
        let bytes = image.as_bytes();
        let hash = ContentHash::of(bytes);
        let path = self.unique_asset_path(dir, name, "inf_tex")?;
        inf_asset::write_atomically(&path, bytes)?; // C4-27, as `write_asset`
        self.register_written_asset(
            path,
            inf_material::TiledTextureImage::KIND,
            hash,
            source,
            import,
            None,
        )
    }

    /// Rewrite an existing asset's payload (data-asset editors save through
    /// this), updating the content hash + dependency edges.
    ///
    /// **All-or-nothing** (C4-12), like its sibling
    /// [`write_asset`](Self::write_asset). This is the save door for every
    /// data-asset editor and, via `dcc::save_mesh_session`, for the Model
    /// Editor; a failure here maps to `SaveError::Write`, whose user-facing text
    /// ends *"Nothing on disk changed."* That sentence was false in exactly the
    /// disk-full / interrupted case it exists for — the payload was already
    /// truncated. Temp+rename is what makes the message true.
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
        inf_asset::write_atomically(&path, &bytes)?;

        let name = self.db.get(id).unwrap().name.clone();
        let mut sidecar = self.db.get(id).unwrap().sidecar.clone();
        sidecar.content_hash = hash;
        sidecar.dependencies = dependencies;
        sidecar.save(&path)?;
        self.db.insert(AssetEntry {
            sidecar,
            path,
            name,
            // The sidecar was written by this call, so it is on disk and legible.
            sidecar_unreadable: false,
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

/// Say out loud what the scan found (C4-39).
///
/// [`AssetDb::sidecar_advisories`] is the record; this is its surface. A
/// payload whose sidecar exists but cannot be read is listed under a stand-in
/// identity, so the Content Drawer shows it and nothing looks wrong — the whole
/// point of the advisory is that the scan already *knew* and said nothing. It
/// goes to the Output Log rather than into the snapshot DTO because that costs
/// no wire change and the condition needs a human, not a UI affordance.
fn report_sidecar_advisories(db: &AssetDb) {
    for line in db.sidecar_advisories() {
        tracing::warn!("content scan: {line}");
    }
    // **`AssetDb::collisions()`'s first production consumer** (P26.5's record,
    // Wave A's hand-off). The scan has detected two payloads claiming one GUID
    // since P26.5 and refused the second — correctly, since silently
    // re-pointing every edge in the graph at the intruder is the worse answer —
    // and then said nothing, so the symptom an author meets is an asset that is
    // in the folder and not in the drawer. A record nothing reads is not a
    // report.
    // `IdCollision` already writes itself in the P16 advisory shape (asset,
    // consequence, remedy) — the sentence existed and had no reader.
    for c in db.collisions() {
        tracing::warn!("content scan: {c}");
    }
}

/// The value an import records in its sidecar's `source` field — **`None` when
/// the file lives outside the project root** (L7.H7).
///
/// The sidecar is deterministic TOML committed beside the payload. Recording an
/// absolute path writes a developer's machine layout, and on Windows their OS
/// **username**, into every other checkout of the project: a data leak in
/// version-controlled content. It has never fired in this repo only because
/// every committed sample is built programmatically rather than imported.
///
/// The capability given up is small and was already unreliable: every re-import
/// door joins the recorded value onto the project root
/// (`terrain_import::reimport_terrain`), so an outside path only resolved by
/// accident of `Path::join`'s absolute-replaces rule, and moved the moment the
/// project did. `None` makes the re-import door say "this asset has no import
/// source" instead — a refusal rather than a leak. Copy the file under the
/// project to keep re-import.
///
/// One door for every importer: the generic one, the mesh container, and the
/// terrain wizard each had their own copy of the old `rel_source`.
pub(crate) fn sidecar_source(project: &AssetProject, source: &Path) -> Option<String> {
    source
        .strip_prefix(project.root())
        .ok()
        .map(|rel| rel.to_string_lossy().replace('\\', "/"))
}

/// The advisory an import raises when [`sidecar_source`] declines to record the
/// path — so the lost re-import is *said*, not merely absent.
pub(crate) fn outside_root_advisory(source: &Path) -> String {
    let name = source
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| "the source file".to_string());
    format!(
        "\"{name}\" was imported from outside the project, so no source path is recorded \
         (a sidecar is committed content and an absolute path would carry this machine's \
         layout into every checkout) — re-import is unavailable until a copy lives under \
         the project"
    )
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

    /// **Two assets through the specialized door are two assets** (P26.1 audit).
    ///
    /// `reuse.unwrap_or_default()` looks like "mint one" and is not:
    /// `AssetId::default()` is `AssetId::NIL`, the *"no asset"* sentinel. So
    /// every asset written here without a `reuse` shared one GUID — the second
    /// to arrive replaced the first in the database and left its payload
    /// orphaned on disk, silently, with the import reporting success.
    ///
    /// It hid for two phases because `.inf_terrain` was the only caller that
    /// ever passes `None` (a project has one terrain, a reimport passes `reuse`,
    /// and a `.inf_vmesh` derivation always passes its computed id). P26.1 sent
    /// every imported texture through it, and one glTF brings several.
    ///
    /// Asserted kind-agnostically at the door rather than only on textures,
    /// because the defect was never about textures.
    #[test]
    fn the_specialized_writer_mints_a_fresh_guid_per_asset() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let content = proj.content_dir("Terrain").unwrap();
        let mut ids = Vec::new();
        for name in ["North", "South", "East"] {
            let payload = content.join(format!("{name}.inf_terrain"));
            std::fs::write(&payload, name.as_bytes()).unwrap();
            ids.push(
                proj.register_written_asset(
                    payload,
                    AssetKind::Terrain,
                    ContentHash::of(name.as_bytes()),
                    None,
                    None,
                    None,
                )
                .unwrap(),
            );
        }
        for (i, id) in ids.iter().enumerate() {
            assert!(!id.is_nil(), "asset {i} was registered under the NIL guid");
        }
        assert_eq!(
            ids.iter().collect::<std::collections::BTreeSet<_>>().len(),
            3,
            "the specialized writer handed out one guid for three assets"
        );
        // …and all three are in the database, rather than the last one having
        // evicted the other two.
        for id in &ids {
            assert!(proj.db().contains(*id));
        }
        assert_eq!(
            proj.db()
                .iter()
                .filter(|e| e.kind() == AssetKind::Terrain)
                .count(),
            3
        );
        // `reuse` still pins the guid — the reimport and derivation contract.
        let payload = content.join("North.inf_terrain");
        std::fs::write(&payload, b"v2").unwrap();
        assert_eq!(
            proj.register_written_asset(
                payload,
                AssetKind::Terrain,
                ContentHash::of(b"v2"),
                None,
                None,
                Some(ids[0]),
            )
            .unwrap(),
            ids[0]
        );
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

    /// **`rewrite_payload` is all-or-nothing** (C4-12/F9) — the save door for
    /// every data-asset editor and, through `save_mesh_session`, for the Model
    /// Editor. Its failure reports *"Nothing on disk changed."*; a reader
    /// holding the old payload open proves that sentence.
    #[test]
    fn rewriting_a_payload_never_exposes_a_torn_one() {
        use std::io::Read;
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let d = proj.content_dir("tables").unwrap();

        let mut table = inf_asset::StructAsset::new("Doc");
        let id = proj
            .write_asset(&d, "Doc", &table, None, vec![], None)
            .unwrap();
        let path = proj.db().get(id).unwrap().path.clone();
        let before = std::fs::read(&path).unwrap();

        let mut live = std::fs::File::open(&path).unwrap();
        for i in 0..64 {
            table.fields.push(inf_asset::FieldDef {
                name: format!("field_{i}"),
                ty: inf_asset::FieldType::Int,
            });
        }
        proj.rewrite_payload(id, &table, vec![]).unwrap();

        let mut seen = Vec::new();
        live.read_to_end(&mut seen).unwrap();
        assert_eq!(seen, before, "a live reader observed the rewrite");
        assert_ne!(std::fs::read(&path).unwrap(), before);
        assert!(no_temp_litter(&d), "left temp files behind");
    }

    /// **A rename moves the sidecar first and rolls it back on failure**
    /// (C4-22/F10). The exposure a failure leaves must be "a sidecar with no
    /// payload" (which no scan adopts) and never "a payload with no sidecar"
    /// (which the watcher promotes under a fresh GUID, dangling every edge).
    #[test]
    fn a_rename_that_cannot_finish_leaves_the_pair_whole() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let d = proj.content_dir("m").unwrap();
        let id = proj
            .write_asset(&d, "Before", &MaterialAsset::default(), None, vec![], None)
            .unwrap();
        let old_path = proj.db().get(id).unwrap().path.clone();
        let old_side = inf_asset::sidecar_path(&old_path);

        // (a) The SIDECAR move is the one that fails. Park a directory at
        // `After.inf_mat.toml` — `rename` onto a non-empty directory fails on
        // every platform, and `unique_path` does not look at the `.toml`, so the
        // payload destination stays free.
        let blocked = d.join("After.inf_mat.toml");
        std::fs::create_dir_all(&blocked).unwrap();
        std::fs::write(blocked.join("occupant"), b"x").unwrap();

        assert!(
            proj.rename(id, "After").is_err(),
            "the rename must not report success"
        );
        assert!(
            old_path.exists(),
            "the payload moved out from under its sidecar — the watcher would \
             promote it under a fresh GUID"
        );
        assert!(!d.join("After.inf_mat").exists());
        assert_eq!(inf_asset::AssetSidecar::load(&old_path).unwrap().guid, id);
        std::fs::remove_file(blocked.join("occupant")).unwrap();
        std::fs::remove_dir(&blocked).unwrap();

        // (b) The PAYLOAD move is the one that fails, *after* the sidecar has
        // already moved: the rollback path. The payload is removed behind the
        // database's back, so its rename comes back `NotFound`.
        std::fs::remove_file(&old_path).unwrap();
        assert!(proj.rename(id, "After").is_err());
        assert!(
            old_side.exists(),
            "the sidecar was not rolled back to the payload's path"
        );
        assert!(!d.join("After.inf_mat.toml").exists());

        // Unblocked, the rename lands and both halves move together.
        std::fs::write(&old_path, b"payload").unwrap();
        proj.rename(id, "After").unwrap();
        let new_path = proj.db().get(id).unwrap().path.clone();
        assert!(!old_path.exists() && !old_side.exists());
        assert_eq!(inf_asset::AssetSidecar::load(&new_path).unwrap().guid, id);
    }

    /// **`write_asset_at` overwrites in place and keeps the GUID** (C4-21) —
    /// the deterministic-path door the graph editors save through. The second
    /// save must land on the same file with the same id, not on `Name_1` and
    /// not under a fresh one.
    #[test]
    fn a_deterministic_path_save_keeps_one_file_and_one_guid() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let d = proj.content_dir("graphs").unwrap();
        let path = d.join("Forest.inf_struct");

        let first = proj
            .write_asset_at(&path, &inf_asset::StructAsset::new("Forest"), vec![])
            .unwrap();
        assert!(
            inf_asset::sidecar_path(&path).exists(),
            "no sidecar written"
        );

        let mut edited = inf_asset::StructAsset::new("Forest");
        edited.name = "Forest v2".into();
        let second = proj.write_asset_at(&path, &edited, vec![]).unwrap();
        assert_eq!(first, second, "the graph's GUID churned on re-save");
        assert_eq!(
            std::fs::read_dir(&d)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("inf_struct"))
                .count(),
            1,
            "the re-save forked a second file"
        );
        assert_eq!(
            proj.load_payload::<inf_asset::StructAsset>(first)
                .unwrap()
                .name,
            "Forest v2"
        );
    }

    fn no_temp_litter(dir: &Path) -> bool {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .all(|e| !e.file_name().to_string_lossy().ends_with(".tmp"))
    }
}
