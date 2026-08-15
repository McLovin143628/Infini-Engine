//! **Chunked huge-heightmap import** (P16.4a): a 16 k × 16 k source becomes a
//! streamed `.inf_terrain` without the sample grid ever existing.
//!
//! # The pipeline
//!
//! ```text
//!  file ──decode_rows──▶ one source row at a time (f64, source domain)
//!            │
//!            ▼
//!   BandAssembler        holds ONE lattice row of tiles (`ntx` pages) and fills
//!            │           their rows as the source rows arrive; a band is final
//!            │           the moment its last shared-edge row lands
//!            ▼
//!   LevelAccumulator ×N  each holds ≤ 2 finished rows of its input level and
//!            │           emits a coarse row the moment a 2 × 2 sibling pair is
//!            │           complete (the pyramid, built as we go)
//!            ▼
//!   TerrainAssetBuilder  every finished tile is encoded to its canonical blob
//!                        and staged by `TileKey`
//! ```
//!
//! # The memory bound
//!
//! Between source rows the pipeline holds
//!
//! ```text
//!   2 · ntx                     the assembler (two bands overlap for exactly the
//!                               one shared-edge row that ends the lower band)
//! + Σ over levels 2 · width(L)  each accumulator's ≤ 2 pending input rows
//! ```
//!
//! tiles — [`ImportReport::live_tile_bound`], asserted against the measured
//! high-water ([`ImportReport::peak_live_tiles`]) by
//! `the_pipeline_never_holds_more_than_its_documented_bound`. Since
//! `width(L) = ceil(width(L−1) / 2)`, the whole chain sums to under `6 · ntx`
//! pages — **O(one tile row)**, independent of the source's height. A 16 k × 16 k
//! import at `tile_resolution = 256` holds ~65 pages per row, ~100 MB, where the
//! sample grid alone would be 1 GB.
//!
//! The one thing that *does* scale with the output is
//! [`TerrainAssetBuilder`]'s staged blobs: `write_terrain_asset` is a whole-image
//! atomic writer by contract (the module doc's "exactly one writer" rule), so the
//! payload is assembled before it is renamed into place. That is the size of the
//! file being produced, not of the source, and a spill-to-temp payload writer is
//! the documented follow-up for multi-gigabyte terrains.
//!
//! # Determinism
//!
//! The output is **byte-identical** to the whole-image path
//! ([`TerrainData::from_height_image`] + [`build_pyramid`] +
//! [`build_terrain_asset`]) — the `chunked_matches_whole_image_byte_for_byte`
//! gate — and byte-identical for any job-pool size, because the only parallel step
//! is [`inf_core::job::parallel_map_ref`], the deterministic in-order pure map
//! (P7.0), and everything downstream of it lands in a `BTreeMap` keyed by
//! [`TileKey`]. Rows may even arrive out of order (a `Decreasing`/`Unspecified`
//! EXR): every stage here is keyed, never appended.

use std::collections::BTreeMap;
use std::path::Path;

use glam::{DVec2, DVec3};
use inf_core::job::JobPool;

use crate::asset::{encode_tile, TerrainAsset, TerrainAssetBuilder};
use crate::import::{
    decode_rows, probe_reader, HeightmapGrid, HeightmapImport, HeightmapProbe, TerrainError,
};
use crate::pyramid::{downsample_tiles, PyramidOptions};
use crate::tile::{TerrainTile, TileKey};

/// Knobs for [`import_heightmap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChunkedImportOptions {
    /// LOD pyramid generation, exactly as [`crate::build_pyramid`] would apply it
    /// to the finished terrain.
    pub pyramid: PyramidOptions,
}

/// A progress tick: how many tiles of the whole import are final.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImportProgress {
    /// Tiles written so far, across every level.
    pub tiles_done: u64,
    /// Tiles the finished asset will hold, known before decoding starts.
    pub tiles_total: u64,
    /// The level the tiles just staged belong to (`0` = the authored level).
    pub lod: u32,
}

/// What an import produced, for the caller's log line and the wizard's done state.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportReport {
    /// What the source file said about itself.
    pub probe: HeightmapProbe,
    /// The level-0 tile lattice.
    pub grid: HeightmapGrid,
    /// Tiles across all levels.
    pub tiles: usize,
    /// LOD levels present (`1` = level 0 only).
    pub lod_levels: u32,
    /// Real-world span of the source, in metres (SI, per the units doctrine).
    pub extent: DVec2,
    /// Highest number of tile pages held in the pipeline at once.
    pub peak_live_tiles: usize,
    /// The bound `peak_live_tiles` is asserted against (see the module docs).
    pub live_tile_bound: usize,
}

/// Import a heightmap **file** into a `.inf_terrain` payload, chunked.
///
/// `progress` is called after every finished tile row; `cancel` is polled at the
/// same points and turns the import into [`TerrainError::Cancelled`] — the caller
/// is left with nothing partially written, because the payload is only handed back
/// on success and [`crate::write_terrain_asset`] is atomic.
pub fn import_heightmap(
    path: &Path,
    import: HeightmapImport,
    opts: ChunkedImportOptions,
    progress: &mut dyn FnMut(ImportProgress),
    cancel: &dyn Fn() -> bool,
) -> Result<(TerrainAsset, ImportReport), TerrainError> {
    import_heightmap_in(inf_core::global(), path, import, opts, progress, cancel)
}

/// [`import_heightmap`] on a caller-supplied job pool. The payload is independent
/// of the pool's thread count — this is the seam the determinism guard uses (the
/// same shape as `inf_pcg::scatter_region_in`).
pub fn import_heightmap_in(
    pool: &JobPool,
    path: &Path,
    import: HeightmapImport,
    opts: ChunkedImportOptions,
    progress: &mut dyn FnMut(ImportProgress),
    cancel: &dyn Fn() -> bool,
) -> Result<(TerrainAsset, ImportReport), TerrainError> {
    let file = std::fs::File::open(path)?;
    import_heightmap_reader_in(
        pool,
        std::io::BufReader::new(file),
        import,
        opts,
        progress,
        cancel,
    )
}

/// [`import_heightmap`] over any seekable source (tests import from memory).
pub fn import_heightmap_reader<R: std::io::BufRead + std::io::Seek>(
    src: R,
    import: HeightmapImport,
    opts: ChunkedImportOptions,
    progress: &mut dyn FnMut(ImportProgress),
    cancel: &dyn Fn() -> bool,
) -> Result<(TerrainAsset, ImportReport), TerrainError> {
    import_heightmap_reader_in(inf_core::global(), src, import, opts, progress, cancel)
}

/// [`import_heightmap_reader`] on a caller-supplied job pool.
pub fn import_heightmap_reader_in<R: std::io::BufRead + std::io::Seek>(
    pool: &JobPool,
    mut src: R,
    import: HeightmapImport,
    opts: ChunkedImportOptions,
    progress: &mut dyn FnMut(ImportProgress),
    cancel: &dyn Fn() -> bool,
) -> Result<(TerrainAsset, ImportReport), TerrainError> {
    import.validate(None)?;
    // The lattice needs the source's dimensions before the first row arrives, so
    // the header is read first — header-only, no pixels, no allocation.
    let probe = probe_reader(&mut src)?;
    import.validate(Some(&probe))?;
    let mut build = Build::new(pool, import, opts, &probe, progress, cancel);
    let decoded = decode_rows(src, &mut |y, row| build.push_source_row(y as i32, row))?;
    if (decoded.width, decoded.height) != (probe.width, probe.height) {
        return Err(TerrainError::Image(
            "the heightmap's header and its pixel data disagree about the size".into(),
        ));
    }
    build.finish(probe)
}

// ── the pyramid plan ────────────────────────────────────────────────────────

/// The rectangle of tile coordinates one level covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LevelShape {
    tx_min: i32,
    tx_max: i32,
    tz_min: i32,
    tz_max: i32,
}

impl LevelShape {
    fn count(self) -> usize {
        self.width() * self.depth()
    }
    fn width(self) -> usize {
        (self.tx_max - self.tx_min + 1).max(0) as usize
    }
    fn depth(self) -> usize {
        (self.tz_max - self.tz_min + 1).max(0) as usize
    }
    /// The coarse rectangle one level up — floor-halved on every side, matching
    /// [`downsample_tiles`]'s `div_euclid(2)` grouping (so negative coordinates
    /// group the same way).
    fn parent(self) -> Self {
        Self {
            tx_min: self.tx_min.div_euclid(2),
            tx_max: self.tx_max.div_euclid(2),
            tz_min: self.tz_min.div_euclid(2),
            tz_max: self.tz_max.div_euclid(2),
        }
    }
}

/// The coarse levels an import will produce.
///
/// A **mirror of [`crate::build_pyramid`]'s stop rule** expressed on counts alone,
/// which is what lets the pyramid be built incrementally: the streaming build must
/// know how many levels it owes before it has seen a single tile. The mirror is
/// pinned by `the_plan_matches_build_pyramid`.
fn plan_levels(level0: LevelShape, opts: PyramidOptions) -> Vec<LevelShape> {
    let mut out = Vec::new();
    if level0.count() <= opts.min_tiles || opts.max_levels == 0 {
        return out;
    }
    let mut cur = level0.parent();
    for lod in 1..=opts.max_levels {
        let stop = cur.count() <= opts.min_tiles || lod == opts.max_levels;
        out.push(cur);
        if stop {
            break;
        }
        cur = cur.parent();
    }
    out
}

// ── the band assembler ──────────────────────────────────────────────────────

/// One lattice row of tiles being filled in from source rows.
struct Band {
    /// `ntx` height buffers of `res²` samples.
    tiles: Vec<Vec<f32>>,
    /// Which of the band's `res` rows have landed.
    rows_seen: Vec<bool>,
    seen: u32,
}

/// Fills one lattice row of tiles at a time from source rows.
///
/// Order-agnostic: a row is routed by its index, so a `Decreasing`/`Unspecified`
/// EXR works, and a monotone source (every PNG, and the standard EXR line order)
/// never holds more than two bands.
struct BandAssembler {
    grid: HeightmapGrid,
    import: HeightmapImport,
    bands: BTreeMap<i32, Band>,
    /// Source row mapped to metres once per row and shared by every tile in it —
    /// the *same* arithmetic the whole-image tiler applies, in the same order.
    mapped: Vec<f32>,
}

impl BandAssembler {
    fn new(grid: HeightmapGrid, import: HeightmapImport) -> Self {
        Self {
            grid,
            import,
            bands: BTreeMap::new(),
            mapped: Vec::new(),
        }
    }

    /// Tiles currently held.
    fn live_tiles(&self) -> usize {
        self.bands.len() * self.grid.ntx as usize
    }

    /// Feed one source row (`gz` may run past the source's last row — those rows
    /// clamp onto it, exactly as the whole-image tiler's `sample_at` does).
    /// Returns any bands the row completed, keyed by lattice row.
    fn push_row(&mut self, gz: i32, row: &[f64]) -> Vec<(i32, Vec<TerrainTile>)> {
        let res = self.grid.resolution;
        let cells = self.grid.cells;
        let width = self.grid.width as usize;

        // Map once: `import.map_sample(v) as f32` is the whole-image path's
        // per-sample expression, so the two agree bit-for-bit.
        self.mapped.clear();
        self.mapped
            .extend(row.iter().map(|&v| self.import.map_sample(v) as f32));
        let last_col = width.saturating_sub(1);

        let mut done = Vec::new();
        for tz in band_rows_of(gz, cells, self.grid.ntz) {
            let j = (gz - tz * cells) as u32;
            let ntx = self.grid.ntx as usize;
            let band = self.bands.entry(tz).or_insert_with(|| Band {
                tiles: vec![vec![0.0f32; (res * res) as usize]; ntx],
                rows_seen: vec![false; res as usize],
                seen: 0,
            });
            if band.rows_seen[j as usize] {
                continue; // a duplicate row in a malformed source
            }
            band.rows_seen[j as usize] = true;
            band.seen += 1;
            let base = (j * res) as usize;
            for (x, tile) in band.tiles.iter_mut().enumerate() {
                let gx0 = x as i32 * cells;
                for i in 0..res as usize {
                    let gx = (gx0 as usize + i).min(last_col);
                    tile[base + i] = self.mapped[gx];
                }
            }
            if band.seen == res {
                let band = self.bands.remove(&tz).expect("band present");
                done.push((tz, self.finish_band(tz, band)));
            }
        }
        done
    }

    /// Turn a complete band into real tiles (world-anchored, `origin.y = 0` —
    /// the `f32` offsets carry the whole height, matching `author_tile` and the
    /// whole-image tiler).
    fn finish_band(&self, tz: i32, band: Band) -> Vec<TerrainTile> {
        let res = self.grid.resolution;
        let span = (res as f64 - 1.0) * self.import.meters_per_sample;
        band.tiles
            .into_iter()
            .enumerate()
            .map(|(x, heights)| {
                let (cx, cz) = self.grid.coord(x as i32, tz);
                let origin = DVec3::new(cx as f64 * span, 0.0, cz as f64 * span);
                TerrainTile::from_heights(res, origin, heights).expect("res² heights")
            })
            .collect()
    }
}

/// Which lattice rows a source row belongs to.
///
/// Tiles share edges, so the row at a tile boundary (`gz % cells == 0`) is both
/// the *last* row of the band below and the *first* row of the band above.
fn band_rows_of(gz: i32, cells: i32, ntz: i32) -> impl Iterator<Item = i32> {
    let primary = gz.div_euclid(cells);
    let shared = (gz % cells == 0).then_some(primary - 1);
    shared
        .into_iter()
        .chain(std::iter::once(primary))
        .filter(move |tz| *tz >= 0 && *tz < ntz)
}

// ── the incremental pyramid ─────────────────────────────────────────────────

/// One level's worth of tiles, keyed by grid coordinate — what a finished row of
/// any level is, and what a coarse row comes back as.
type TileRow = BTreeMap<(i32, i32), TerrainTile>;

/// Buffers finished rows of one level until a 2 × 2 sibling pair is complete,
/// then decimates it into one coarse row.
struct LevelAccumulator {
    /// Metres per sample of the level this consumes.
    fine_mps: f64,
    /// Row range of the level this consumes (so a coarse row knows when no
    /// further sibling can arrive).
    fine_tz_min: i32,
    fine_tz_max: i32,
    rows: BTreeMap<i32, TileRow>,
}

impl LevelAccumulator {
    fn live_tiles(&self) -> usize {
        self.rows.values().map(BTreeMap::len).sum()
    }

    /// Accept a finished input row; return the coarse row it completed, if any.
    fn push(&mut self, res: u32, tz: i32, tiles: TileRow) -> Option<(i32, TileRow)> {
        self.rows.insert(tz, tiles);
        let cz = tz.div_euclid(2);
        let siblings: Vec<i32> = [2 * cz, 2 * cz + 1]
            .into_iter()
            .filter(|t| *t >= self.fine_tz_min && *t <= self.fine_tz_max)
            .collect();
        if !siblings.iter().all(|t| self.rows.contains_key(t)) {
            return None;
        }
        // Exactly the two rows the coarse row decimates — `downsample_tiles` sees
        // the same neighbourhood a whole-terrain pyramid build would give it for
        // this block, so the coarse tiles are bit-identical.
        let mut block: TileRow = BTreeMap::new();
        for t in siblings {
            if let Some(row) = self.rows.remove(&t) {
                block.extend(row);
            }
        }
        Some((cz, downsample_tiles(res, self.fine_mps, &block)))
    }
}

// ── the driver ──────────────────────────────────────────────────────────────

struct Build<'a> {
    pool: &'a JobPool,
    import: HeightmapImport,
    grid: HeightmapGrid,
    plan: Vec<LevelShape>,
    assembler: BandAssembler,
    accs: Vec<LevelAccumulator>,
    builder: TerrainAssetBuilder,
    tiles_total: u64,
    tiles_done: u64,
    peak_live: usize,
    /// The source's last row, kept so the rows past it can clamp onto it.
    last_row: Vec<f64>,
    progress: &'a mut dyn FnMut(ImportProgress),
    cancel: &'a dyn Fn() -> bool,
}

impl<'a> Build<'a> {
    /// Shape the whole pipeline from the probe: the lattice, the level plan (and
    /// therefore the tile total the progress bar needs), and one accumulator per
    /// coarse level — all before a single pixel is decoded.
    fn new(
        pool: &'a JobPool,
        import: HeightmapImport,
        opts: ChunkedImportOptions,
        probe: &HeightmapProbe,
        progress: &'a mut dyn FnMut(ImportProgress),
        cancel: &'a dyn Fn() -> bool,
    ) -> Self {
        let grid = HeightmapGrid::new(probe.width, probe.height, &import);
        let level0 = LevelShape {
            tx_min: grid.tile_origin.0,
            tx_max: grid.tile_origin.0 + grid.ntx - 1,
            tz_min: grid.tile_origin.1,
            tz_max: grid.tile_origin.1 + grid.ntz - 1,
        };
        let plan = plan_levels(level0, opts.pyramid);
        let tiles_total = (level0.count() + plan.iter().map(|s| s.count()).sum::<usize>()) as u64;

        let mut fine = level0;
        let mut mps = import.meters_per_sample;
        let accs = plan
            .iter()
            .map(|coarse| {
                let acc = LevelAccumulator {
                    fine_mps: mps,
                    fine_tz_min: fine.tz_min,
                    fine_tz_max: fine.tz_max,
                    rows: BTreeMap::new(),
                };
                fine = *coarse;
                mps *= 2.0;
                acc
            })
            .collect();

        Self {
            pool,
            import,
            grid,
            plan,
            assembler: BandAssembler::new(grid, import),
            accs,
            // P16.6: record the options this import's pyramid was built with in
            // the v2 header, so a later write-back re-plans to the same shape.
            builder: TerrainAssetBuilder::new(import.resolution(), import.meters_per_sample)
                .with_pyramid(opts.pyramid),
            tiles_total,
            tiles_done: 0,
            peak_live: 0,
            last_row: Vec::new(),
            progress,
            cancel,
        }
    }

    fn check_cancel(&self) -> Result<(), TerrainError> {
        if (self.cancel)() {
            return Err(TerrainError::Cancelled);
        }
        Ok(())
    }

    /// Feed one decoded source row.
    fn push_source_row(&mut self, gz: i32, row: &[f64]) -> Result<(), TerrainError> {
        self.check_cancel()?;
        if gz == self.grid.height as i32 - 1 {
            self.last_row.clear();
            self.last_row.extend_from_slice(row);
        }
        for (tz, tiles) in self.assembler.push_row(gz, row) {
            self.emit_level0_row(tz, tiles)?;
        }
        self.observe_live();
        Ok(())
    }

    /// Key a finished band by world tile coordinate and hand it to the cascade.
    fn emit_level0_row(&mut self, tz: i32, tiles: Vec<TerrainTile>) -> Result<(), TerrainError> {
        let (ox, oz) = self.grid.tile_origin;
        let map: TileRow = tiles
            .into_iter()
            .enumerate()
            .map(|(x, t)| ((ox + x as i32, oz + tz), t))
            .collect();
        self.emit_map(0, oz + tz, map)
    }

    fn emit_map(&mut self, level: usize, tz: i32, tiles: TileRow) -> Result<(), TerrainError> {
        self.check_cancel()?;
        self.stage(level as u32, &tiles)?;
        if level < self.accs.len() {
            let res = self.grid.resolution;
            if let Some((cz, coarse)) = self.accs[level].push(res, tz, tiles) {
                self.emit_map(level + 1, cz, coarse)?;
            }
        }
        Ok(())
    }

    /// Encode a finished row's tiles to their canonical blobs and stage them.
    ///
    /// [`inf_core::job::parallel_map_ref`] is the deterministic in-order pure map
    /// (P7.0): the blob vector is a function of the tile slice alone, so the
    /// staged bytes — and therefore the payload — are identical for any pool size.
    fn stage(&mut self, lod: u32, tiles: &TileRow) -> Result<(), TerrainError> {
        if tiles.is_empty() {
            return Ok(());
        }
        let keys: Vec<(i32, i32)> = tiles.keys().copied().collect();
        let list: Vec<&TerrainTile> = tiles.values().collect();
        let blobs = self.pool.parallel_map_ref(&list, |t| encode_tile(t));
        for (coord, blob) in keys.into_iter().zip(blobs) {
            let blob = blob.map_err(|e| TerrainError::Image(e.to_string()))?;
            self.builder
                .insert_bytes(TileKey::new(lod, coord), blob)
                .map_err(|e| TerrainError::Image(e.to_string()))?;
        }
        self.tiles_done += tiles.len() as u64;
        (self.progress)(ImportProgress {
            tiles_done: self.tiles_done,
            tiles_total: self.tiles_total,
            lod,
        });
        Ok(())
    }

    fn observe_live(&mut self) {
        let live = self.assembler.live_tiles()
            + self
                .accs
                .iter()
                .map(LevelAccumulator::live_tiles)
                .sum::<usize>();
        self.peak_live = self.peak_live.max(live);
    }

    /// The documented high-water bound (see the module docs).
    fn live_tile_bound(&self) -> usize {
        let mut bound = 2 * self.grid.ntx as usize; // the assembler's two bands
        let mut fine_width = self.grid.ntx as usize;
        for coarse in &self.plan {
            bound += 2 * fine_width; // that level's accumulator
            fine_width = coarse.width();
        }
        bound
    }

    fn finish(
        mut self,
        probe: HeightmapProbe,
    ) -> Result<(TerrainAsset, ImportReport), TerrainError> {
        // Rows past the source's last one clamp onto it (the whole-image tiler's
        // `sample_at` clamp, expressed as extra rows).
        let last = std::mem::take(&mut self.last_row);
        if !last.is_empty() {
            for gz in probe.height as i32..=self.grid.last_source_row() {
                self.check_cancel()?;
                for (tz, tiles) in self.assembler.push_row(gz, &last) {
                    self.emit_level0_row(tz, tiles)?;
                }
            }
        }
        self.observe_live();
        if !self.assembler.bands.is_empty() {
            return Err(TerrainError::Image(format!(
                "{} tile band(s) never completed — the source is truncated",
                self.assembler.bands.len()
            )));
        }
        for acc in &self.accs {
            if !acc.rows.is_empty() {
                return Err(TerrainError::Image(
                    "the pyramid has rows with no sibling — the plan and the lattice disagree"
                        .into(),
                ));
            }
        }
        let tiles = self.builder.len();
        let lod_levels = 1 + self.plan.len() as u32;
        let live_tile_bound = self.live_tile_bound();
        let report = ImportReport {
            extent: self.import.world_extent(probe.width, probe.height),
            probe,
            grid: self.grid,
            tiles,
            lod_levels,
            peak_live_tiles: self.peak_live,
            live_tile_bound,
        };
        let asset = self
            .builder
            .build()
            .map_err(|e| TerrainError::Image(e.to_string()))?;
        Ok((asset, report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::{encode_png16, HeightImage, HeightMode};
    use crate::pyramid::build_pyramid;
    use crate::TerrainData;

    fn noop_progress() -> impl FnMut(ImportProgress) {
        |_| {}
    }

    fn never_cancel() -> impl Fn() -> bool {
        || false
    }

    /// A deterministic, non-degenerate 16-bit test image (no `std` trig — the P14
    /// bit-portability law applies to anything whose bytes get compared).
    fn synthetic(width: u32, height: u32) -> HeightImage {
        let samples = (0..width as u64 * height as u64)
            .map(|i| {
                let x = i % width as u64;
                let y = i / width as u64;
                (((x * 6367 + y * 2749 + x * y * 13) % 65536) as u16).rotate_left(1)
            })
            .collect();
        HeightImage {
            width,
            height,
            samples,
        }
    }

    fn png_of(width: u32, height: u32) -> Vec<u8> {
        encode_png16(&synthetic(width, height)).unwrap()
    }

    /// The whole-image reference: decode everything, tile it, pyramid it, pack it.
    fn reference_asset(
        bytes: &[u8],
        import: HeightmapImport,
        opts: ChunkedImportOptions,
    ) -> Vec<u8> {
        let data = TerrainData::from_height_image(bytes, import).unwrap();
        let pyramid = build_pyramid(&data, opts.pyramid);
        crate::asset::build_terrain_asset(&data, &pyramid, opts.pyramid)
            .unwrap()
            .into_bytes()
    }

    fn chunked_asset(
        bytes: &[u8],
        import: HeightmapImport,
        opts: ChunkedImportOptions,
    ) -> (Vec<u8>, ImportReport) {
        let (asset, report) = import_heightmap_reader(
            std::io::Cursor::new(bytes.to_vec()),
            import,
            opts,
            &mut noop_progress(),
            &never_cancel(),
        )
        .unwrap();
        (asset.into_bytes(), report)
    }

    #[test]
    fn the_plan_matches_build_pyramid() {
        // The streaming build must know its level count before it sees a tile;
        // this pins the count-only mirror against the real thing.
        for (ntx, ntz) in [(1, 1), (2, 2), (3, 5), (8, 8), (9, 4), (16, 16)] {
            for origin in [(0, 0), (-4, -3), (5, -9)] {
                let mut data = TerrainData::new(5, 1.0);
                for z in 0..ntz {
                    for x in 0..ntx {
                        data.author_tile((origin.0 + x, origin.1 + z), |x, z| x * 0.5 - z * 0.25);
                    }
                }
                let opts = PyramidOptions::default();
                let real = build_pyramid(&data, opts);
                let plan = plan_levels(
                    LevelShape {
                        tx_min: origin.0,
                        tx_max: origin.0 + ntx - 1,
                        tz_min: origin.1,
                        tz_max: origin.1 + ntz - 1,
                    },
                    opts,
                );
                assert_eq!(
                    plan.len(),
                    real.len(),
                    "{ntx}x{ntz} @ {origin:?} level count"
                );
                for (shape, level) in plan.iter().zip(&real) {
                    assert_eq!(shape.count(), level.tiles.len(), "lod {} size", level.lod);
                }
            }
        }
    }

    #[test]
    fn chunked_matches_whole_image_byte_for_byte() {
        // THE DETERMINISM GATE. Sizes chosen to exercise: an exact tile fit, a
        // partial edge tile in x, in z, and in both, and enough tiles for two
        // coarse levels.
        let import = HeightmapImport {
            tile_resolution: 5,
            meters_per_sample: 3.0,
            min_height: -120.0,
            max_height: 880.0,
            ..Default::default()
        };
        let opts = ChunkedImportOptions::default();
        for (w, h) in [(9, 9), (17, 17), (18, 17), (17, 18), (19, 23), (33, 33)] {
            let png = png_of(w, h);
            let (chunked, report) = chunked_asset(&png, import, opts);
            let reference = reference_asset(&png, import, opts);
            assert_eq!(
                chunked, reference,
                "{w}x{h}: chunked import diverged from the whole-image path"
            );
            assert_eq!(
                report.tiles,
                report.grid.tile_count() + coarse_of(&reference)
            );
        }
    }

    /// Tiles in the reference beyond level 0 (read back off the payload header).
    fn coarse_of(bytes: &[u8]) -> usize {
        let r = crate::asset::TerrainAssetReader::new(bytes).unwrap();
        r.directory().iter().filter(|e| !e.key.is_lod0()).count()
    }

    #[test]
    fn a_negative_tile_origin_is_byte_identical_too() {
        // Centring on the world origin puts half the lattice in the negative
        // quadrant, where the pyramid's floor-halving is the thing that can go
        // wrong (a truncating `/ 2` would merge -1 with 0).
        let import = HeightmapImport {
            tile_resolution: 5,
            meters_per_sample: 2.0,
            min_height: 0.0,
            max_height: 500.0,
            tile_origin: (-4, -3),
            ..Default::default()
        };
        let opts = ChunkedImportOptions::default();
        let png = png_of(33, 29);
        let (chunked, _) = chunked_asset(&png, import, opts);
        assert_eq!(chunked, reference_asset(&png, import, opts));
    }

    #[test]
    fn the_output_does_not_depend_on_the_job_pool_size() {
        // `parallel_map_ref` is a pure in-order map, so a 1-thread and an
        // N-thread pool must stage identical bytes.
        let import = HeightmapImport {
            tile_resolution: 5,
            meters_per_sample: 1.0,
            min_height: 0.0,
            max_height: 100.0,
            ..Default::default()
        };
        let png = png_of(33, 33);
        let opts = ChunkedImportOptions::default();
        let baseline = chunked_asset(&png, import, opts).0;
        for threads in [1usize, 2, 3, 7] {
            let pool = JobPool::new(threads);
            let (asset, _) = import_heightmap_reader_in(
                &pool,
                std::io::Cursor::new(png.clone()),
                import,
                opts,
                &mut noop_progress(),
                &never_cancel(),
            )
            .unwrap();
            assert_eq!(
                asset.into_bytes(),
                baseline,
                "pool of {threads} thread(s) diverged"
            );
        }
    }

    #[test]
    fn the_pipeline_never_holds_more_than_its_documented_bound() {
        // The memory shape, asserted structurally: the high-water page count of
        // the band assembler + the pyramid accumulators, not RSS.
        let import = HeightmapImport {
            tile_resolution: 5,
            meters_per_sample: 1.0,
            min_height: 0.0,
            max_height: 100.0,
            ..Default::default()
        };
        let png = png_of(65, 65); // 16x16 tiles => 3 coarse levels
        let (_, report) = chunked_asset(&png, import, ChunkedImportOptions::default());
        assert!(report.lod_levels >= 3, "need a real pyramid: {report:?}");
        assert!(
            report.peak_live_tiles <= report.live_tile_bound,
            "held {} pages, bound is {}",
            report.peak_live_tiles,
            report.live_tile_bound
        );
        // …and the bound really is O(one tile row), not O(image).
        assert!(
            report.live_tile_bound < report.grid.tile_count(),
            "the bound ({}) is not below the tile count ({})",
            report.live_tile_bound,
            report.grid.tile_count()
        );
    }

    #[test]
    fn progress_counts_every_tile_exactly_once() {
        let import = HeightmapImport {
            tile_resolution: 5,
            meters_per_sample: 1.0,
            min_height: 0.0,
            max_height: 10.0,
            ..Default::default()
        };
        let png = png_of(33, 33);
        let mut ticks: Vec<ImportProgress> = Vec::new();
        let (asset, report) = import_heightmap_reader(
            std::io::Cursor::new(png),
            import,
            ChunkedImportOptions::default(),
            &mut |p| ticks.push(p),
            &never_cancel(),
        )
        .unwrap();
        assert!(!ticks.is_empty());
        let last = ticks.last().unwrap();
        assert_eq!(last.tiles_done, last.tiles_total);
        assert_eq!(last.tiles_total as usize, report.tiles);
        assert_eq!(asset.reader().tile_count(), report.tiles);
        // Monotone and never over-counting.
        assert!(ticks.windows(2).all(|w| w[0].tiles_done < w[1].tiles_done));
    }

    #[test]
    fn cancellation_stops_the_import_and_yields_nothing() {
        let import = HeightmapImport {
            tile_resolution: 5,
            ..Default::default()
        };
        let png = png_of(33, 33);
        let err = import_heightmap_reader(
            std::io::Cursor::new(png),
            import,
            ChunkedImportOptions::default(),
            &mut noop_progress(),
            &|| true,
        )
        .unwrap_err();
        assert!(matches!(err, TerrainError::Cancelled), "got {err}");
    }

    #[test]
    fn float_metres_exr_round_trips_known_values() {
        // A float EXR imported as ABSOLUTE metres: the decoded value IS the
        // height, with no normalization anywhere in the path (the P10 follow-up).
        let (w, h) = (9u32, 9u32);
        let height_at = |x: u32, z: u32| (x as f32) * 12.5 - (z as f32) * 3.25 - 500.0;
        let exr = write_test_exr(w, h, &height_at);
        let import = HeightmapImport {
            tile_resolution: 5,
            meters_per_sample: 4.0,
            mode: HeightMode::FloatMeters,
            ..Default::default()
        };
        let (asset, report) = import_heightmap_reader(
            std::io::Cursor::new(exr.clone()),
            import,
            ChunkedImportOptions::default(),
            &mut noop_progress(),
            &never_cancel(),
        )
        .unwrap();
        assert_eq!((report.probe.width, report.probe.height), (w, h));
        assert!(report.probe.float_samples);
        let r = asset.reader();
        for (tx, tz) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            let tile = r.tile(TileKey::lod0((tx, tz))).unwrap().unwrap();
            for j in 0..5u32 {
                for i in 0..5u32 {
                    let (gx, gz) = (tx as u32 * 4 + i, tz as u32 * 4 + j);
                    assert_eq!(
                        tile.world_height(5, i, j),
                        height_at(gx, gz) as f64,
                        "tile ({tx},{tz}) sample ({i},{j})"
                    );
                }
            }
        }
        // …and the chunked EXR path is byte-identical to the whole-image one.
        assert_eq!(
            asset.as_bytes(),
            reference_asset(&exr, import, ChunkedImportOptions::default())
        );
    }

    #[test]
    fn exr_import_exercises_multiple_blocks_and_partial_edge_tiles() {
        // 67x53 at resolution 9 => 9x7 tiles with a partial edge tile on both
        // axes, and enough scanlines that the EXR arrives in several blocks.
        let (w, h) = (67u32, 53u32);
        let exr = write_test_exr(w, h, &|x, z| ((x * 37 + z * 11) % 1000) as f32 / 1000.0);
        let import = HeightmapImport {
            tile_resolution: 9,
            meters_per_sample: 5.0,
            min_height: -50.0,
            max_height: 250.0,
            ..Default::default()
        };
        let opts = ChunkedImportOptions::default();
        let (chunked, report) = chunked_asset(&exr, import, opts);
        assert_eq!((report.grid.ntx, report.grid.ntz), (9, 7));
        assert_eq!(chunked, reference_asset(&exr, import, opts));
        assert!(report.peak_live_tiles <= report.live_tile_bound);
    }

    /// **Round-2 finding B3: a NaN from a third-party EXR must not reach
    /// committed terrain.**
    ///
    /// NaN is the *conventional* no-data value in a float height EXR, and this
    /// decoder passed f16/f32 channel samples through verbatim.
    /// `HeightmapImport::map_sample` looks like the guard and is not:
    /// `Normalized` does `v.clamp(0.0, 1.0)`, and Rust's `clamp` returns
    /// `self` when neither comparison fires — which is what a NaN does to both
    /// — while `FloatMeters`, which the wizard offers whenever the source is
    /// float, returns `v` untouched. Downstream is `.inf_terrain`, rapier
    /// heightfields, `terrain.height_at`, the clipmap and erosion; and
    /// `height_bounds` uses `f32::min`/`max`, which ignore NaN, so the tile's
    /// own bounds look perfectly healthy the whole way.
    ///
    /// Both tilers are driven, because the finding is precisely that they
    /// share one decoder and neither had the check.
    #[test]
    fn a_non_finite_exr_sample_is_refused_by_both_tilers() {
        let import = HeightmapImport {
            tile_resolution: 5,
            meters_per_sample: 1.0,
            min_height: 0.0,
            max_height: 100.0,
            mode: HeightMode::FloatMeters,
            ..Default::default()
        };
        let opts = ChunkedImportOptions::default();

        // The control: the same generator with every sample finite imports
        // through both paths, so a refusal below cannot be "EXR is broken".
        let clean = write_test_exr(9, 7, &|x, z| (x + z) as f32);
        assert!(TerrainData::from_height_image(&clean, import).is_ok());
        assert!(import_heightmap_reader(
            std::io::Cursor::new(clean.clone()),
            import,
            opts,
            &mut noop_progress(),
            &never_cancel(),
        )
        .is_ok());

        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let bytes = write_test_exr(9, 7, &move |x, z| {
                if (x, z) == (4, 3) {
                    poison
                } else {
                    (x + z) as f32
                }
            });

            // The whole-image tiler.
            let e = TerrainData::from_height_image(&bytes, import)
                .expect_err("a {poison} sample tiled into a terrain");
            let msg = e.to_string();
            assert!(
                msg.contains("(4, 3)") && msg.contains("finite"),
                "the refusal must name the pixel: {msg}"
            );

            // The chunked tiler — the one a 16 k import actually uses.
            let e = import_heightmap_reader(
                std::io::Cursor::new(bytes.clone()),
                import,
                opts,
                &mut noop_progress(),
                &never_cancel(),
            )
            .expect_err("the chunked tiler accepted a non-finite sample");
            assert!(
                e.to_string().contains("finite"),
                "the two tilers disagree about a non-finite sample: {e}"
            );
        }
    }

    /// The second line, one layer down: `TerrainTile::set_sample` drops a
    /// non-finite height rather than storing it, exactly as its sibling
    /// `HeightRegion::set_height` has since C4-35.
    ///
    /// Seven writers reach this door without crossing `decode_rows` — the
    /// brush, the delta replay, the pyramid fold, the analytic generators —
    /// and `encode_tile` bincodes whatever is in the buffer with no check of
    /// its own.
    #[test]
    fn a_tile_refuses_to_store_a_non_finite_height() {
        let mut tile = crate::TerrainTile::flat(4, glam::DVec3::ZERO);
        tile.set_sample(4, 1, 1, 12.5);
        assert_eq!(tile.sample(4, 1, 1), 12.5);
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            tile.set_sample(4, 1, 1, poison);
            assert_eq!(
                tile.sample(4, 1, 1),
                12.5,
                "a {poison} height was stored into a tile that is about to be bincoded into a committed .inf_terrain"
            );
        }
    }

    /// A single-channel (`Y`) `f32` scanline EXR — the shape a heightmap export
    /// takes.
    fn write_test_exr(width: u32, height: u32, f: &dyn Fn(u32, u32) -> f32) -> Vec<u8> {
        use exr::prelude::*;
        let pixels: Vec<f32> = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x, y)))
            .map(|(x, y)| f(x, y))
            .collect();
        let w = width as usize;
        let channel = SpecificChannels::build()
            .with_channel("Y")
            .with_pixel_fn(|p: Vec2<usize>| (pixels[p.y() * w + p.x()],));
        let image = Image::from_channels((width as usize, height as usize), channel);
        let mut out = std::io::Cursor::new(Vec::new());
        image
            .write()
            .non_parallel()
            .to_buffered(&mut out)
            .expect("write exr");
        out.into_inner()
    }
}
