//! The **Terrain Import** job (P16.4a): a huge heightmap becomes a streamed
//! `.inf_terrain` asset the editor can walk immediately.
//!
//! Ring 1 owns the orchestration; the decoding, tiling and pyramid all live in
//! [`inf_terrain::chunked`]. What is added here is everything a *project* needs:
//!
//! * **Header-only probing** ([`probe`]) so the wizard can show the source's real
//!   dimensions and a suggested extent before a single pixel is decoded.
//! * **Settings that survive** — the whole [`TerrainImportSettings`] block is
//!   written into the asset's sidecar `import` table, so [`reimport`] re-runs the
//!   import with exactly the choices the user made the first time (the sidecar
//!   contract: "reimport must honor these").
//! * **Atomic, cancellable writes.** The payload goes through
//!   [`inf_terrain::write_terrain_asset`] — the ONE sanctioned `.inf_terrain`
//!   writer — which is temp-file + rename, so a cancelled or failed import leaves
//!   the content root exactly as it found it. There is no intermediate spill file
//!   to clean up: the pipeline streams source rows straight into the payload
//!   builder.
//!
//! ## Why not `import_file`
//!
//! The generic importer routes by extension and a `.png` is far more often a
//! texture than a heightmap. Terrain import is therefore its own explicit job,
//! reached from the wizard rather than from a drag-and-drop of an image.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use inf_asset::{AssetError, AssetId, AssetKind, ContentHash, Result};
use inf_terrain::{
    ChunkedImportOptions, HeightMode, HeightmapGrid, HeightmapImport, HeightmapProbe,
    ImportProgress, PyramidOptions,
};
use serde::{Deserialize, Serialize};

use super::AssetProject;

/// Default content sub-folder terrain imports land in.
pub const TERRAIN_IMPORT_FOLDER: &str = "Terrain";

/// Everything the wizard decides, and everything a reimport needs.
///
/// Serialized verbatim into the sidecar's `import` table (deterministic TOML, like
/// every other importer's settings block), so it round-trips through the file and
/// a later reimport is bit-identical to the first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerrainImportSettings {
    /// Samples per tile side of the produced terrain.
    pub tile_resolution: u32,
    /// World metres between adjacent samples. `≫ 1` is first-class: an 8 m
    /// spacing turns a 16 k source into 131 km of world.
    pub meters_per_sample: f64,
    /// Elevation the source's `0` maps to (normalized mode only).
    pub min_height: f64,
    /// Elevation the source's full scale maps to (normalized mode only).
    pub max_height: f64,
    /// Take the decoded float as **absolute metres** instead of normalizing.
    /// Float sources (EXR) only.
    #[serde(default)]
    pub float_meters: bool,
    /// Straddle the world origin instead of growing into `+X/+Z`.
    #[serde(default)]
    pub center: bool,
    /// Maximum coarse LOD levels to generate.
    #[serde(default = "default_max_levels")]
    pub max_pyramid_levels: u32,
    /// Stop generating levels once one holds at most this many tiles.
    #[serde(default = "default_min_tiles")]
    pub min_pyramid_tiles: usize,
}

fn default_max_levels() -> u32 {
    PyramidOptions::default().max_levels
}
fn default_min_tiles() -> usize {
    PyramidOptions::default().min_tiles
}

impl Default for TerrainImportSettings {
    fn default() -> Self {
        Self {
            tile_resolution: inf_terrain::DEFAULT_TILE_RESOLUTION,
            meters_per_sample: 1.0,
            min_height: 0.0,
            max_height: 1000.0,
            float_meters: false,
            center: true,
            max_pyramid_levels: default_max_levels(),
            min_pyramid_tiles: default_min_tiles(),
        }
    }
}

impl TerrainImportSettings {
    /// The Ring-0 settings these map onto for a `width × height` source.
    ///
    /// Centring is resolved **here**, against the probed dimensions, into an
    /// integral tile origin — so the value that reaches Ring 0 is exact and a
    /// reimport of the same file re-lands on the same tiles.
    pub fn to_import(&self, width: u32, height: u32) -> HeightmapImport {
        let base = HeightmapImport {
            tile_resolution: self.tile_resolution.max(2),
            meters_per_sample: self.meters_per_sample,
            min_height: self.min_height,
            max_height: self.max_height,
            mode: if self.float_meters {
                HeightMode::FloatMeters
            } else {
                HeightMode::Normalized
            },
            tile_origin: (0, 0),
        };
        let tile_origin = if self.center {
            HeightmapGrid::centered_origin(width, height, &base)
        } else {
            (0, 0)
        };
        HeightmapImport {
            tile_origin,
            ..base
        }
    }

    /// The pyramid knobs.
    pub fn pyramid(&self) -> PyramidOptions {
        PyramidOptions {
            max_levels: self.max_pyramid_levels,
            min_tiles: self.min_pyramid_tiles,
        }
    }

    /// The real-world span (metres) this import will cover.
    pub fn world_extent(&self, width: u32, height: u32) -> (f64, f64) {
        let e = self.to_import(width, height).world_extent(width, height);
        (e.x, e.y)
    }

    /// C4-45: `None` means the sidecar records no import settings, and
    /// `reimport_terrain` then refuses with "no recorded import settings" — the
    /// wizard's every choice, lost with no stated cause.
    fn to_table(&self) -> Option<toml::Table> {
        match toml::Value::try_from(self) {
            Ok(toml::Value::Table(t)) => Some(t),
            Ok(other) => {
                tracing::warn!(
                    "terrain import settings serialized as {} rather than a table; \
                     re-import will refuse",
                    other.type_str()
                );
                None
            }
            Err(e) => {
                tracing::warn!(
                    "terrain import settings will not serialize ({e}); re-import will refuse"
                );
                None
            }
        }
    }

    /// `None` means the recorded block is not these settings — an older schema,
    /// or a hand-edited sidecar. Said out loud for the same reason.
    fn from_table(table: &toml::Table) -> Option<Self> {
        match toml::Value::Table(table.clone()).try_into() {
            Ok(v) => Some(v),
            Err(e) => {
                tracing::warn!(
                    "a terrain sidecar's recorded import settings will not read back ({e}); \
                     re-import will refuse rather than guess"
                );
                None
            }
        }
    }
}

/// Header-only probe of a heightmap file — no pixels decoded.
pub fn probe(path: &Path) -> Result<HeightmapProbe> {
    inf_terrain::probe_heightmap(path).map_err(|e| AssetError::Import(e.to_string()))
}

/// Settings the wizard opens with for a given source.
///
/// The sampling suggestion is a **size tier**, not a computation: a small
/// heightmap is usually a hand-authored prop terrain (1 m samples), a
/// 1 k–4 k source a level (4 m), and anything larger a landscape the whole point
/// of which is tens of kilometres (8 m). Every value is editable in the wizard —
/// this only decides what the first screen shows.
pub fn suggested_settings(probe: &HeightmapProbe) -> TerrainImportSettings {
    let longest = probe.width.max(probe.height);
    let meters_per_sample = if longest >= 4096 {
        8.0
    } else if longest >= 1024 {
        4.0
    } else {
        1.0
    };
    TerrainImportSettings {
        meters_per_sample,
        // Float-metres stays OFF by default even for a float EXR: a normalized
        // `[0, 1]` EXR is just as common as an absolute-elevation one, and
        // guessing wrong the other way silently flattens the terrain to a metre.
        // The wizard offers the toggle whenever `probe.float_samples`.
        float_meters: false,
        ..Default::default()
    }
}

/// What a terrain import produced.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainImportOutcome {
    /// The `.inf_terrain` asset GUID.
    pub asset: AssetId,
    /// Where the payload landed.
    pub path: PathBuf,
    /// Source sample dimensions.
    pub width: u32,
    pub height: u32,
    /// Level-0 tiles across / down.
    pub tiles_x: i32,
    pub tiles_z: i32,
    /// Tiles across all levels.
    pub tiles: usize,
    /// LOD levels present (`1` = level 0 only).
    pub lod_levels: u32,
    /// Real-world span in metres (SI — the wizard divides by 1000 for its km
    /// readback, and nothing else ever scales it).
    pub extent_m: (f64, f64),
    /// Payload size on disk.
    pub bytes: u64,
    /// **Non-fatal advisories** (round-2 finding R2.F7).
    ///
    /// L7.H7 stopped the sidecar recording an absolute source path — a sidecar
    /// is committed content and must not carry this machine's layout — and the
    /// *advisory* half of that fix reached the mesh and texture importers and
    /// not this one. Here it matters most: a heightmap normally lives outside
    /// the project, so `source: None` is the common case, and the author's only
    /// notice was `reimport` refusing with "no import source" at some later
    /// session.
    pub advisories: Vec<String>,
}

/// A cancellation token shared with the caller (the Ring-2 command layer holds a
/// clone so a wizard "Cancel" can stop an in-flight import).
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// A built, in-memory `.inf_terrain` payload that has not been committed to a
/// project yet — the output of the **lock-free build phase**.
///
/// # Why the phases are split
///
/// Building is the whole import: probe, decode, tile, pyramid, serialize. It can
/// run for minutes on a 16 k source and it needs **no project state at all** —
/// only the source path and the settings. Committing is a handful of filesystem
/// and database operations.
///
/// Holding the shared `AssetProject` mutex across the build would freeze every
/// asset command, the background progress tick, and the wizard's own Cancel for
/// the duration of the import. So [`build`] takes no project, and [`commit`]
/// takes it for the few milliseconds it actually needs. The worker in
/// [`super::queue`] is the caller that matters, and its test
/// (`a_terrain_import_progresses_and_cancels_while_the_project_is_locked`) pins
/// exactly this property.
pub struct TerrainBuild {
    asset: inf_terrain::TerrainAsset,
    report: inf_terrain::ImportReport,
    source: PathBuf,
    settings: TerrainImportSettings,
}

impl TerrainBuild {
    /// The source that was decoded.
    pub fn source(&self) -> &Path {
        &self.source
    }
    /// Tiles across all levels.
    pub fn tiles(&self) -> usize {
        self.report.tiles
    }
    /// Payload size the commit will write.
    pub fn bytes(&self) -> usize {
        self.asset.as_bytes().len()
    }
}

/// **Phase 1 — build.** Decode `source` into a `.inf_terrain` payload. Touches no
/// project state, takes no locks, and can be cancelled at any tile row.
pub fn build(
    source: &Path,
    settings: &TerrainImportSettings,
    progress: &mut dyn FnMut(ImportProgress),
    cancel: &CancelToken,
) -> Result<TerrainBuild> {
    let probe = inf_terrain::probe_heightmap(source)
        .map_err(|e| AssetError::Import(format!("{}: {e}", source.display())))?;
    let import = settings.to_import(probe.width, probe.height);
    let opts = ChunkedImportOptions {
        pyramid: settings.pyramid(),
    };
    let (asset, report) =
        inf_terrain::import_heightmap(source, import, opts, progress, &|| cancel.is_cancelled())
            .map_err(|e| AssetError::Import(format!("{}: {e}", source.display())))?;
    Ok(TerrainBuild {
        asset,
        report,
        source: source.to_path_buf(),
        settings: settings.clone(),
    })
}

/// **Phase 2 — commit.** Write the built payload into `project` and register it.
///
/// `name` defaults to the source's file stem; `reuse` keeps an existing GUID (a
/// reimport rewriting its own file). Short-lived: everything expensive already
/// happened in [`build`].
///
/// **The last cancellation check lives here**, before anything is written: a job
/// cancelled after its build finished must register nothing, or a retry would
/// find `World` taken and land a stray `World_1` beside the abandoned asset.
pub fn commit(
    project: &mut AssetProject,
    built: TerrainBuild,
    name: Option<&str>,
    reuse: Option<AssetId>,
    cancel: &CancelToken,
) -> Result<TerrainImportOutcome> {
    if cancel.is_cancelled() {
        return Err(AssetError::Import("import cancelled".into()));
    }
    let path = match reuse.and_then(|id| project.db().get(id).map(|e| e.path.clone())) {
        Some(existing) => existing,
        None => {
            let dir = project.content_dir(TERRAIN_IMPORT_FOLDER)?;
            let stem = name
                .map(|s| s.to_string())
                .unwrap_or_else(|| file_stem(&built.source));
            project.unique_asset_path(&dir, &stem, "inf_terrain")?
        }
    };

    // The ONE sanctioned `.inf_terrain` writer: raw image, temp + rename. It
    // hands back the bytes it wrote so the sidecar hash covers exactly them.
    let bytes = inf_terrain::write_terrain_asset(&path, &built.asset)?;
    let hash = ContentHash::of(bytes);
    let size = bytes.len() as u64;
    // L7.H7: `None` when the heightmap lives outside the project — the sidecar
    // is committed content and must not carry this machine's paths.
    //
    // R2.F7: and SAY SO. This is the one importer where an outside-the-project
    // source is the norm, and it recorded `None` in silence; the author found
    // out at some later session, from `reimport` refusing with "no import
    // source". `outside_root_advisories` is the same door the mesh and texture
    // importers raise it through, so the wording cannot drift.
    let advisories = super::import::outside_root_advisories(project, &built.source);
    let rel_source = super::sidecar_source(project, &built.source);
    let id = project.register_written_asset(
        path,
        AssetKind::Terrain,
        hash,
        rel_source,
        built.settings.to_table(),
        reuse,
    )?;
    // The database normalizes paths on insert; report the stored one so a
    // reimport (which reads its path back out of the database) agrees with the
    // original import's outcome.
    let path = project
        .db()
        .get(id)
        .map(|e| e.path.clone())
        .ok_or(AssetError::UnknownAsset(id))?;

    Ok(TerrainImportOutcome {
        asset: id,
        path,
        width: built.report.probe.width,
        height: built.report.probe.height,
        tiles_x: built.report.grid.ntx,
        tiles_z: built.report.grid.ntz,
        tiles: built.report.tiles,
        lod_levels: built.report.lod_levels,
        extent_m: (built.report.extent.x, built.report.extent.y),
        bytes: size,
        advisories,
    })
}

/// Import `source` into `project` as a new `.inf_terrain` asset — [`build`] then
/// [`commit`], for callers that are not sharing the project with anything.
///
/// `name` defaults to the source's file stem; `progress` is called as tile rows
/// complete. Cancellation leaves nothing behind — the payload is only written
/// once the whole import succeeded, and that write is atomic.
pub fn import_terrain(
    project: &mut AssetProject,
    source: &Path,
    settings: &TerrainImportSettings,
    name: Option<&str>,
    progress: &mut dyn FnMut(ImportProgress),
    cancel: &CancelToken,
) -> Result<TerrainImportOutcome> {
    let built = build(source, settings, progress, cancel)?;
    commit(project, built, name, None, cancel)
}

/// Re-run the import that produced `asset`, honoring the settings stored in its
/// sidecar and rewriting the same file under the same GUID.
///
/// Fails when the asset has no recorded source or no settings block — a terrain
/// authored in-editor has nothing to reimport *from*.
pub fn reimport(
    project: &mut AssetProject,
    asset: AssetId,
    progress: &mut dyn FnMut(ImportProgress),
    cancel: &CancelToken,
) -> Result<TerrainImportOutcome> {
    let entry = project
        .db()
        .get(asset)
        .ok_or(AssetError::UnknownAsset(asset))?;
    if entry.kind() != AssetKind::Terrain {
        return Err(AssetError::Import(format!(
            "asset {asset} is not a terrain"
        )));
    }
    let source = entry
        .sidecar
        .source
        .clone()
        .ok_or_else(|| AssetError::Import("terrain asset has no import source".into()))?;
    let settings = entry
        .sidecar
        .import
        .as_ref()
        .and_then(TerrainImportSettings::from_table)
        .ok_or_else(|| {
            AssetError::Import("terrain asset has no recorded import settings".into())
        })?;
    let source = project.root().join(source);
    let built = build(&source, &settings, progress, cancel)?;
    commit(project, built, None, Some(asset), cancel)
}

fn file_stem(source: &Path) -> String {
    source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Terrain")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_terrain::{encode_png16, HeightImage};

    /// A deterministic 16-bit heightmap on disk.
    fn write_png(dir: &Path, name: &str, w: u32, h: u32) -> PathBuf {
        let samples = (0..w as u64 * h as u64)
            .map(|i| {
                let (x, y) = (i % w as u64, i / w as u64);
                ((x * 6367 + y * 2749 + x * y * 13) % 65536) as u16
            })
            .collect();
        let png = encode_png16(&HeightImage {
            width: w,
            height: h,
            samples,
        })
        .unwrap();
        let path = dir.join(name);
        std::fs::write(&path, png).unwrap();
        path
    }

    fn settings() -> TerrainImportSettings {
        TerrainImportSettings {
            tile_resolution: 33,
            meters_per_sample: 8.0,
            min_height: -200.0,
            max_height: 1800.0,
            center: true,
            ..Default::default()
        }
    }

    #[test]
    fn probing_reads_the_header_only() {
        let src = tempfile::tempdir().unwrap();
        let png = write_png(src.path(), "heights.png", 129, 65);
        let p = probe(&png).unwrap();
        assert_eq!((p.width, p.height), (129, 65));
        assert_eq!(p.bit_depth, 16);
        assert!(!p.float_samples);
        // The suggestion is a size tier and stays metric.
        let s = suggested_settings(&p);
        assert_eq!(s.meters_per_sample, 1.0);
        assert!(!s.float_meters);
    }

    /// The end-to-end job: a source big enough for **two pyramid levels and many
    /// row bands** but small enough for CI (129² samples at resolution 33 →
    /// 4 × 4 level-0 tiles, 128 source rows through the band assembler).
    #[test]
    fn importing_produces_a_registered_streamable_asset() {
        let src = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let png = write_png(src.path(), "World.png", 129, 129);
        let mut proj = AssetProject::open(proj_dir.path()).unwrap();

        let mut ticks = 0usize;
        let out = import_terrain(
            &mut proj,
            &png,
            &settings(),
            None,
            &mut |_| ticks += 1,
            &CancelToken::new(),
        )
        .unwrap();

        assert!(ticks > 1, "progress never ticked");
        assert_eq!((out.width, out.height), (129, 129));
        assert_eq!((out.tiles_x, out.tiles_z), (4, 4));
        assert!(out.lod_levels >= 2, "no pyramid: {out:?}");
        // 129 samples at 8 m = 1024 m across — SI metres, no scale factor.
        assert_eq!(out.extent_m, (1024.0, 1024.0));

        // Registered, on disk, and readable through the streaming reader.
        let entry = proj.db().get(out.asset).expect("registered");
        assert_eq!(entry.kind(), AssetKind::Terrain);
        assert_eq!(entry.name, "World");
        assert!(out.path.exists());
        assert!(inf_asset::sidecar_path(&out.path).exists());
        let payload = inf_terrain::read_terrain_asset(&out.path).unwrap();
        assert_eq!(payload.reader().tile_count(), out.tiles);
        assert_eq!(payload.reader().tile_resolution(), 33);
        assert_eq!(payload.reader().meters_per_sample(), 8.0);
        // Centred: the lattice straddles the origin.
        assert!(payload
            .reader()
            .keys()
            .any(|k| k.is_lod0() && k.coord.0 < 0 && k.coord.1 < 0));
        // The sidecar hash covers the bytes actually written.
        assert_eq!(
            entry.sidecar.content_hash,
            ContentHash::of(&std::fs::read(&out.path).unwrap())
        );
    }

    #[test]
    fn reimport_respects_the_recorded_settings() {
        let src = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let png = write_png(src.path(), "World.png", 65, 65);
        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        // **Inside the project** (L7.H7): a heightmap outside it records no
        // source path, so a reimport refuses by name. That is the deliberate
        // trade and it has its own arm below.
        let png = {
            let inside = proj_dir.path().join("Source");
            std::fs::create_dir_all(&inside).unwrap();
            let dest = inside.join(png.file_name().unwrap());
            std::fs::copy(&png, &dest).unwrap();
            dest
        };
        let s = TerrainImportSettings {
            tile_resolution: 17,
            meters_per_sample: 12.5,
            min_height: -5.0,
            max_height: 95.0,
            center: false,
            ..Default::default()
        };
        let first =
            import_terrain(&mut proj, &png, &s, None, &mut |_| {}, &CancelToken::new()).unwrap();
        let first_bytes = std::fs::read(&first.path).unwrap();

        // The settings really are in the sidecar…
        let table = proj
            .db()
            .get(first.asset)
            .unwrap()
            .sidecar
            .import
            .clone()
            .expect("import settings persisted");
        assert_eq!(TerrainImportSettings::from_table(&table).unwrap(), s);

        // …and a reimport (which is told nothing) reproduces them byte-for-byte,
        // in place, under the same GUID.
        let again = reimport(&mut proj, first.asset, &mut |_| {}, &CancelToken::new()).unwrap();
        assert_eq!(again.asset, first.asset);
        assert_eq!(again.path, first.path);
        assert_eq!(std::fs::read(&again.path).unwrap(), first_bytes);
        assert_eq!(proj.db().len(), 1, "reimport must not duplicate the asset");
    }

    /// L7.H7's other half, for the terrain wizard: a heightmap outside the
    /// project records no source, so `reimport` refuses **by name** rather than
    /// guessing a path. The refusal is the point — the alternative was a sidecar
    /// carrying this machine's layout into every checkout.
    #[test]
    fn a_heightmap_outside_the_project_records_no_source_and_refuses_reimport() {
        let outside = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let png = write_png(outside.path(), "Away.png", 129, 129);
        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        let s = TerrainImportSettings::default();
        let built =
            import_terrain(&mut proj, &png, &s, None, &mut |_| {}, &CancelToken::new()).unwrap();

        assert_eq!(
            proj.db().get(built.asset).unwrap().sidecar.source,
            None,
            "the sidecar records a path from outside the project"
        );
        // **R2.F7: and it has to SAY SO.** The refusal above is the only notice
        // the author used to get, and it arrives at whatever later session they
        // try to re-import in. The advisory arrives now, on the import that
        // caused it, through the same door the mesh and texture importers use.
        assert_eq!(built.advisories.len(), 1, "{:?}", built.advisories);
        assert!(
            built.advisories[0].contains("outside the project")
                && built.advisories[0].contains("re-import"),
            "the advisory must name the consequence, not just the fact: {:?}",
            built.advisories
        );

        let err = reimport(&mut proj, built.asset, &mut |_| {}, &CancelToken::new())
            .expect_err("reimport must refuse rather than guess");
        assert!(
            err.to_string().contains("no import source"),
            "the refusal must say why: {err}"
        );
    }

    /// The other direction: an import from INSIDE the project records its source
    /// and must raise nothing. An advisory that always fires is noise, and an
    /// author who learns to ignore the badge is back where R2.F7 started.
    #[test]
    fn a_heightmap_inside_the_project_raises_no_advisory() {
        let proj_dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        let inside = proj_dir.path().join("Source");
        std::fs::create_dir_all(&inside).unwrap();
        let png = write_png(&inside, "Home.png", 65, 65);

        let built = import_terrain(
            &mut proj,
            &png,
            &TerrainImportSettings::default(),
            None,
            &mut |_| {},
            &CancelToken::new(),
        )
        .unwrap();

        assert!(
            proj.db().get(built.asset).unwrap().sidecar.source.is_some(),
            "the source is recordable, so it is recorded"
        );
        assert!(built.advisories.is_empty(), "{:?}", built.advisories);
    }

    #[test]
    fn cancelling_leaves_the_content_root_untouched() {
        let src = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let png = write_png(src.path(), "World.png", 129, 129);
        let mut proj = AssetProject::open(proj_dir.path()).unwrap();

        let token = CancelToken::new();
        token.cancel();
        let err =
            import_terrain(&mut proj, &png, &settings(), None, &mut |_| {}, &token).unwrap_err();
        assert!(err.to_string().contains("cancelled"), "got {err}");

        assert!(
            proj.db().is_empty(),
            "a cancelled import registered an asset"
        );
        // No payload, no sidecar, and no temp litter beside them.
        let dir = proj.root().join(TERRAIN_IMPORT_FOLDER);
        let strays: Vec<_> = std::fs::read_dir(&dir)
            .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).collect())
            .unwrap_or_default();
        assert!(strays.is_empty(), "cancelled import left {strays:?}");
    }

    /// **The cancel-then-retry gate (P16.4a audit).** A job cancelled *after* its
    /// build finished must still register nothing — otherwise the abandoned job
    /// takes the name `World`, and the user's retry lands a stray `World_1` beside
    /// an asset nothing references.
    #[test]
    fn a_job_cancelled_after_its_build_registers_nothing_and_frees_the_name() {
        let src = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let png = write_png(src.path(), "World.png", 65, 65);
        let mut proj = AssetProject::open(proj_dir.path()).unwrap();

        // Build with a live token, then cancel — exactly the race the worker has
        // between its two phases.
        let token = CancelToken::new();
        let built = build(&png, &settings(), &mut |_| {}, &token).unwrap();
        token.cancel();
        let err = commit(&mut proj, built, None, None, &token).unwrap_err();
        assert!(err.to_string().contains("cancelled"), "got {err}");
        assert!(
            proj.db().is_empty(),
            "a cancelled commit registered an asset"
        );
        assert!(
            !proj_dir.path().join("Terrain/World.inf_terrain").exists(),
            "a cancelled commit wrote a payload"
        );

        // The retry gets the original name, not `World_1`.
        let retry = import_terrain(
            &mut proj,
            &png,
            &settings(),
            None,
            &mut |_| {},
            &CancelToken::new(),
        )
        .unwrap();
        assert_eq!(proj.db().len(), 1, "exactly one asset after the retry");
        assert_eq!(proj.db().get(retry.asset).unwrap().name, "World");
    }

    /// The build phase must not need the project at all — that is what lets the
    /// worker run a multi-minute import without freezing every asset command.
    #[test]
    fn the_build_phase_needs_no_project() {
        let src = tempfile::tempdir().unwrap();
        let png = write_png(src.path(), "World.png", 65, 65);
        let built = build(&png, &settings(), &mut |_| {}, &CancelToken::new()).unwrap();
        assert!(built.tiles() > 0 && built.bytes() > 0);
        assert_eq!(built.source(), png.as_path());
    }

    #[test]
    fn bad_settings_fail_before_anything_is_written() {
        let src = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let png = write_png(src.path(), "World.png", 33, 33);
        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        // Float-metres on a PNG is refused, not reinterpreted as 0..65535 m.
        let bad = TerrainImportSettings {
            float_meters: true,
            ..settings()
        };
        assert!(import_terrain(
            &mut proj,
            &png,
            &bad,
            None,
            &mut |_| {},
            &CancelToken::new()
        )
        .is_err());
        assert!(proj.db().is_empty());
    }

    /// The perf-pass smoke: a real 16 k × 16 k import at 8 m/sample — 131 km of
    /// world, ~4 200 level-0 pages, six coarse levels. `#[ignore]`d because it
    /// writes ~1 GB and takes minutes; run it with
    /// `cargo test -p inf-editor-core -- --ignored huge_heightmap` when profiling
    /// the importer, which is when the memory shape actually needs measuring
    /// (the structural bound is asserted on every CI run by
    /// `inf_terrain::chunked`'s `the_pipeline_never_holds_more_than_its_documented_bound`).
    #[test]
    #[ignore = "perf pass: writes ~1 GB and takes minutes"]
    fn huge_heightmap_16k_imports() {
        let src = tempfile::tempdir().unwrap();
        let proj_dir = tempfile::tempdir().unwrap();
        let png = write_png(src.path(), "Huge.png", 16385, 16385);
        let mut proj = AssetProject::open(proj_dir.path()).unwrap();
        let out = import_terrain(
            &mut proj,
            &png,
            &TerrainImportSettings {
                tile_resolution: 257,
                meters_per_sample: 8.0,
                min_height: 0.0,
                max_height: 4000.0,
                center: true,
                ..Default::default()
            },
            None,
            &mut |_| {},
            &CancelToken::new(),
        )
        .unwrap();
        assert_eq!((out.tiles_x, out.tiles_z), (64, 64));
        assert_eq!(out.extent_m, (131072.0, 131072.0));
        assert!(out.lod_levels >= 5, "{out:?}");
    }
}
