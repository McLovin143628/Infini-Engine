//! Heightmap import/export: 16-bit PNG and EXR in, 16-bit PNG out.
//!
//! # One decoder, two tilers (P16.4)
//!
//! Every heightmap in this engine is decoded by exactly one thing:
//! [`decode_rows`], a **row-at-a-time** reader over a `Read + Seek` source. It
//! yields each source row as `f64` samples in the source's own domain
//! (`value / 65535` for a 16-bit PNG, the raw float for an EXR) and never holds
//! more than a row plus the codec's own band.
//!
//! Two tilers consume it:
//!
//! * [`TerrainData::from_height_image`] — the **whole-image** path. Collects the
//!   full sample grid, then tiles it. Simple, and the byte-identity reference the
//!   chunked path is measured against.
//! * [`crate::chunked::import_heightmap`] — the **chunked** path. Tiles rows into
//!   pages as they arrive, so a 16 k × 16 k source never materializes its grid.
//!
//! Because they share the decoder *and* the sample→metres mapping
//! ([`HeightmapImport::map_sample`]), the only thing that can differ between them
//! is the tiling strategy — which is exactly what the determinism gate
//! (`chunked_matches_whole_image_byte_for_byte`) pins.
//!
//! # Height mapping
//!
//! [`HeightMode::Normalized`] is the World-Machine / Gaea convention: the source's
//! full range maps onto `[min_height, max_height]` metres. [`HeightMode::FloatMeters`]
//! (EXR only) takes the decoded float as **absolute metres** with no scaling at
//! all — the mode a DEM / real-world elevation export needs, and the P10 follow-up
//! this closes.
//!
//! # What the decoder accepts
//!
//! * **PNG**: non-interlaced **grayscale**, 8- or 16-bit. Colour and interlaced
//!   PNGs are rejected with a message rather than silently luma-averaged: a
//!   heightmap is a scalar field, and matching `image`'s luma weights bit-for-bit
//!   across two code paths is a parity trap nobody would notice failing.
//! * **EXR**: any scanline or tiled layout, `f16` / `f32` / `u32` samples, read
//!   through the `exr` crate's **sequential block** decompressor — the chunked
//!   seam. The `Y` channel is preferred, then `R`, then channel 0.

use std::io::{BufRead, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use glam::{DVec2, DVec3};
use image::{DynamicImage, ImageBuffer, Luma};

use crate::data::TerrainData;

/// How a decoded sample becomes a world height in metres.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeightMode {
    /// The source's full range maps onto `[min_height, max_height]` — the
    /// World-Machine / Gaea convention. Values are clamped to `[0, 1]` first, so
    /// an out-of-range EXR float saturates rather than escaping the stated extent.
    #[default]
    Normalized,
    /// The decoded float **is** the height in metres (no normalization, no
    /// clamping). Float sources only.
    FloatMeters,
}

/// Import settings mapping a decoded heightmap onto world metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightmapImport {
    /// Samples per tile side of the produced terrain.
    pub tile_resolution: u32,
    /// World units between samples.
    pub meters_per_sample: f64,
    /// Elevation (metres) the normalized value `0.0` maps to.
    /// Unused in [`HeightMode::FloatMeters`].
    pub min_height: f64,
    /// Elevation (metres) the normalized value `1.0` maps to.
    /// Unused in [`HeightMode::FloatMeters`].
    pub max_height: f64,
    /// How samples map onto metres.
    pub mode: HeightMode,
    /// Grid coordinate of the tile holding the source's top-left sample.
    ///
    /// `(0, 0)` puts the heightmap's corner on the world origin and grows into
    /// `+X/+Z`; a negative origin (what the wizard's "centre on world origin"
    /// computes) straddles the origin instead. Tile coordinates are integral, so
    /// centring never introduces a fractional offset that would make two imports
    /// of the same file disagree.
    pub tile_origin: (i32, i32),
}

impl Default for HeightmapImport {
    fn default() -> Self {
        Self {
            tile_resolution: crate::DEFAULT_TILE_RESOLUTION,
            meters_per_sample: crate::DEFAULT_METERS_PER_SAMPLE,
            min_height: 0.0,
            max_height: 1.0,
            mode: HeightMode::Normalized,
            tile_origin: (0, 0),
        }
    }
}

impl HeightmapImport {
    /// Map one decoded sample (source domain) onto world metres.
    ///
    /// **The single mapping**: both tilers call this, so a whole-image and a
    /// chunked import of the same file cannot disagree about a height.
    #[inline]
    pub fn map_sample(&self, v: f64) -> f64 {
        match self.mode {
            HeightMode::Normalized => {
                self.min_height + v.clamp(0.0, 1.0) * (self.max_height - self.min_height)
            }
            HeightMode::FloatMeters => v,
        }
    }

    /// Samples per tile side, floored at the 2 the tiling math needs.
    #[inline]
    pub fn resolution(&self) -> u32 {
        self.tile_resolution.max(2)
    }

    /// Reject settings that cannot produce a terrain, before any decoding starts.
    ///
    /// `probe` is checked too when supplied: float-metres is meaningless on an
    /// integer source (a PNG's samples are counts, not elevations), so it is
    /// refused rather than silently reinterpreted as `0..65535` metres.
    pub fn validate(&self, probe: Option<&HeightmapProbe>) -> Result<(), TerrainError> {
        if self.tile_resolution < 2 {
            return Err(TerrainError::Settings(format!(
                "tile_resolution {} is below the minimum of 2",
                self.tile_resolution
            )));
        }
        if !(self.meters_per_sample.is_finite() && self.meters_per_sample > 0.0) {
            return Err(TerrainError::Settings(format!(
                "meters_per_sample {} is not a positive finite length",
                self.meters_per_sample
            )));
        }
        match self.mode {
            HeightMode::Normalized => {
                if !(self.min_height.is_finite() && self.max_height.is_finite()) {
                    return Err(TerrainError::Settings(
                        "the height range must be finite".into(),
                    ));
                }
                if self.max_height <= self.min_height {
                    return Err(TerrainError::Settings(format!(
                        "height range [{}, {}] is empty or inverted",
                        self.min_height, self.max_height
                    )));
                }
            }
            HeightMode::FloatMeters => {
                if let Some(p) = probe {
                    if !p.float_samples {
                        return Err(TerrainError::Settings(format!(
                            "float-metres mode needs a float source; {} carries integer samples",
                            p.format.label()
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// The real-world span (metres) a `width × height` source covers at this
    /// sampling density: `(w − 1) · mps` by `(h − 1) · mps` — the wizard's
    /// "extent" readback. SI metres, per the units doctrine.
    #[inline]
    pub fn world_extent(&self, width: u32, height: u32) -> DVec2 {
        DVec2::new(
            (width.max(1) - 1) as f64 * self.meters_per_sample,
            (height.max(1) - 1) as f64 * self.meters_per_sample,
        )
    }
}

/// The tile lattice a `width × height` source maps onto.
///
/// Tiles share edges (a tile's last row/column is the next tile's first), so a
/// `W × H` grid tiled into `cells = res − 1`-cell tiles needs
/// `ceil((W − 1) / cells)` of them across.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeightmapGrid {
    /// Samples per tile side.
    pub resolution: u32,
    /// World/tile cell span in samples (`resolution − 1`).
    pub cells: i32,
    /// Source width/height in samples.
    pub width: u32,
    pub height: u32,
    /// Tiles across / down.
    pub ntx: i32,
    pub ntz: i32,
    /// Grid coordinate of tile `(0, 0)` of this lattice.
    pub tile_origin: (i32, i32),
}

impl HeightmapGrid {
    /// The lattice `import` maps a `width × height` source onto.
    pub fn new(width: u32, height: u32, import: &HeightmapImport) -> Self {
        let resolution = import.resolution();
        let cells = (resolution - 1) as i32;
        let ntx = (((width as i32 - 1).max(0) + cells - 1) / cells).max(1);
        let ntz = (((height as i32 - 1).max(0) + cells - 1) / cells).max(1);
        Self {
            resolution,
            cells,
            width,
            height,
            ntx,
            ntz,
            tile_origin: import.tile_origin,
        }
    }

    /// Total level-0 tiles.
    #[inline]
    pub fn tile_count(&self) -> usize {
        self.ntx as usize * self.ntz as usize
    }

    /// The world grid coordinate of lattice tile `(x, z)` (`0`-based).
    #[inline]
    pub fn coord(&self, x: i32, z: i32) -> (i32, i32) {
        (self.tile_origin.0 + x, self.tile_origin.1 + z)
    }

    /// The last global source row any tile needs (`ntz · cells`); rows past
    /// `height − 1` clamp onto the final row.
    #[inline]
    pub fn last_source_row(&self) -> i32 {
        self.ntz * self.cells
    }

    /// The tile origin that centres the lattice on the world origin.
    ///
    /// Integral, floored: an odd tile count leaves the extra tile on the `+`
    /// side. Deterministic, so re-importing the same file re-lands identically.
    pub fn centered_origin(width: u32, height: u32, import: &HeightmapImport) -> (i32, i32) {
        let g = Self::new(
            width,
            height,
            &HeightmapImport {
                tile_origin: (0, 0),
                ..*import
            },
        );
        (-(g.ntx / 2), -(g.ntz / 2))
    }
}

/// A reconstructed 16-bit grayscale height image (row-major, `width · height`).
#[derive(Clone, Debug, PartialEq)]
pub struct HeightImage {
    pub width: u32,
    pub height: u32,
    pub samples: Vec<u16>,
}

/// Errors from heightmap import/export.
#[derive(Debug, thiserror::Error)]
pub enum TerrainError {
    #[error("image decode/encode failed: {0}")]
    Image(String),
    #[error("empty or zero-sized heightmap")]
    Empty,
    #[error("io: {0}")]
    Io(String),
    #[error("unsupported heightmap: {0}")]
    Unsupported(String),
    #[error("invalid import settings: {0}")]
    Settings(String),
    #[error("import cancelled")]
    Cancelled,
}

impl From<std::io::Error> for TerrainError {
    fn from(e: std::io::Error) -> Self {
        TerrainError::Io(e.to_string())
    }
}

// ── probing ─────────────────────────────────────────────────────────────────

/// The container a heightmap is stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeightmapFormat {
    /// PNG, grayscale, 8- or 16-bit.
    Png,
    /// OpenEXR.
    Exr,
}

impl HeightmapFormat {
    /// A short label for messages and the wizard.
    pub const fn label(self) -> &'static str {
        match self {
            HeightmapFormat::Png => "PNG",
            HeightmapFormat::Exr => "EXR",
        }
    }
}

/// What a heightmap file says about itself, read from its **header only** — no
/// pixel data is decoded, so probing a 16 k × 16 k source is instant and costs no
/// memory. The wizard's first screen.
#[derive(Clone, Debug, PartialEq)]
pub struct HeightmapProbe {
    pub format: HeightmapFormat,
    pub width: u32,
    pub height: u32,
    /// Bits per sample of the source (8/16 for PNG, 16/32 for EXR).
    pub bit_depth: u32,
    /// `true` when samples are floats — i.e. when [`HeightMode::FloatMeters`] is
    /// available.
    pub float_samples: bool,
    /// The channel the decoder will read (EXR channel name, or `"gray"`).
    pub channel: String,
}

/// The eight-byte PNG signature.
const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
/// The four-byte OpenEXR magic (`0x76 0x2f 0x31 0x01`).
const EXR_MAGIC: [u8; 4] = [0x76, 0x2f, 0x31, 0x01];

/// Sniff the container from the leading bytes, rewinding the source **before and
/// after** — a source may already have been probed once (the chunked importer
/// reads the header, then re-reads the file to decode it).
fn sniff<R: BufRead + Seek>(src: &mut R) -> Result<HeightmapFormat, TerrainError> {
    src.seek(SeekFrom::Start(0))?;
    let mut head = [0u8; 8];
    let n = read_up_to(src, &mut head)?;
    src.seek(SeekFrom::Start(0))?;
    if n >= 8 && head == PNG_MAGIC {
        return Ok(HeightmapFormat::Png);
    }
    if n >= 4 && head[..4] == EXR_MAGIC {
        return Ok(HeightmapFormat::Exr);
    }
    Err(TerrainError::Unsupported(
        "not a PNG or EXR heightmap (unrecognized file signature)".into(),
    ))
}

fn read_up_to<R: Read>(src: &mut R, buf: &mut [u8]) -> Result<usize, TerrainError> {
    let mut filled = 0;
    while filled < buf.len() {
        match src.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}

/// Probe a heightmap **file** without decoding it.
pub fn probe_heightmap(path: &Path) -> Result<HeightmapProbe, TerrainError> {
    let file = std::fs::File::open(path)?;
    probe_reader(std::io::BufReader::new(file))
}

/// Probe a heightmap already in memory.
pub fn probe_heightmap_bytes(bytes: &[u8]) -> Result<HeightmapProbe, TerrainError> {
    probe_reader(Cursor::new(bytes))
}

/// Probe any seekable source.
pub fn probe_reader<R: BufRead + Seek>(mut src: R) -> Result<HeightmapProbe, TerrainError> {
    let probe = match sniff(&mut src)? {
        HeightmapFormat::Png => probe_png(src)?,
        HeightmapFormat::Exr => probe_exr(src)?,
    };
    if probe.width == 0 || probe.height == 0 {
        return Err(TerrainError::Empty);
    }
    Ok(probe)
}

fn probe_png<R: BufRead + Seek>(src: R) -> Result<HeightmapProbe, TerrainError> {
    let mut decoder = png::Decoder::new(src);
    let info = decoder
        .read_header_info()
        .map_err(|e| TerrainError::Image(format!("png header: {e}")))?;
    let (w, h) = (info.width, info.height);
    let depth = png_bit_depth(info.bit_depth);
    check_png_shape(info.color_type, info.bit_depth, info.interlaced)?;
    Ok(HeightmapProbe {
        format: HeightmapFormat::Png,
        width: w,
        height: h,
        bit_depth: depth,
        float_samples: false,
        channel: "gray".into(),
    })
}

fn png_bit_depth(d: png::BitDepth) -> u32 {
    match d {
        png::BitDepth::One => 1,
        png::BitDepth::Two => 2,
        png::BitDepth::Four => 4,
        png::BitDepth::Eight => 8,
        png::BitDepth::Sixteen => 16,
    }
}

/// The PNG shapes the row decoder handles — see the module docs for why colour is
/// refused rather than luma-averaged.
fn check_png_shape(
    color: png::ColorType,
    depth: png::BitDepth,
    interlaced: bool,
) -> Result<(), TerrainError> {
    if interlaced {
        return Err(TerrainError::Unsupported(
            "interlaced PNG heightmaps cannot be streamed row-by-row — re-save \
             the file without Adam7 interlacing"
                .into(),
        ));
    }
    if color != png::ColorType::Grayscale {
        return Err(TerrainError::Unsupported(format!(
            "PNG heightmaps must be grayscale (found {color:?}) — re-export as \
             16-bit grayscale"
        )));
    }
    if !matches!(depth, png::BitDepth::Eight | png::BitDepth::Sixteen) {
        return Err(TerrainError::Unsupported(format!(
            "PNG heightmaps must be 8- or 16-bit (found {depth:?})"
        )));
    }
    Ok(())
}

fn probe_exr<R: BufRead + Seek>(src: R) -> Result<HeightmapProbe, TerrainError> {
    let reader = exr::block::read(src, false).map_err(|e| exr_err("header", &e))?;
    let header = reader
        .headers()
        .first()
        .ok_or_else(|| TerrainError::Unsupported("EXR file has no layers".into()))?;
    let size = header.layer_size;
    let (channel, sample_type) = pick_exr_channel(&header.channels)?;
    let bit_depth = match sample_type {
        exr::meta::attribute::SampleType::F16 => 16,
        exr::meta::attribute::SampleType::F32 => 32,
        exr::meta::attribute::SampleType::U32 => 32,
    };
    let float_samples = !matches!(sample_type, exr::meta::attribute::SampleType::U32);
    Ok(HeightmapProbe {
        format: HeightmapFormat::Exr,
        width: size.0 as u32,
        height: size.1 as u32,
        bit_depth,
        float_samples,
        channel,
    })
}

/// Which EXR channel carries the elevation: `Y` (luminance — what a height export
/// writes), else `R`, else channel 0. Returned by name so the wizard can show it.
fn pick_exr_channel(
    channels: &exr::meta::attribute::ChannelList,
) -> Result<(String, exr::meta::attribute::SampleType), TerrainError> {
    let list = &channels.list;
    if list.is_empty() {
        return Err(TerrainError::Unsupported(
            "EXR layer has no channels".into(),
        ));
    }
    let find = |name: &str| list.iter().position(|c| c.name.to_string() == name);
    let idx = find("Y").or_else(|| find("R")).unwrap_or(0);
    let c = &list[idx];
    Ok((c.name.to_string(), c.sample_type))
}

fn exr_err(what: &str, e: &exr::error::Error) -> TerrainError {
    TerrainError::Image(format!("exr {what}: {e}"))
}

// ── the one row decoder ─────────────────────────────────────────────────────

/// The row sink [`decode_rows`] drives: `(y, samples)` in the source's own
/// domain, with `samples.len() == width`. Returning an error aborts the decode
/// (that is how cancellation and a downstream failure stop a 16 k read).
pub type RowSink<'a> = &'a mut dyn FnMut(u32, &[f64]) -> Result<(), TerrainError>;

/// Decode a heightmap **one row at a time**, in the source's own sample domain.
///
/// `on_row(y, samples)` is called once per source row with `samples.len() ==
/// width`. Rows arrive in increasing `y` for every PNG and for the standard EXR
/// line order; a `Decreasing`/`Unspecified` EXR may deliver them in another order,
/// which is why every consumer here is order-agnostic.
///
/// Returns the probe it read on the way in. Never allocates more than the codec's
/// own band plus one row.
pub fn decode_rows<R: BufRead + Seek>(
    mut src: R,
    on_row: RowSink<'_>,
) -> Result<HeightmapProbe, TerrainError> {
    let format = sniff(&mut src)?;
    let probe = match format {
        HeightmapFormat::Png => decode_rows_png(src, on_row)?,
        HeightmapFormat::Exr => decode_rows_exr(src, on_row)?,
    };
    if probe.width == 0 || probe.height == 0 {
        return Err(TerrainError::Empty);
    }
    Ok(probe)
}

fn decode_rows_png<R: BufRead + Seek>(
    src: R,
    on_row: RowSink<'_>,
) -> Result<HeightmapProbe, TerrainError> {
    let decoder = png::Decoder::new(src);
    let mut reader = decoder
        .read_info()
        .map_err(|e| TerrainError::Image(format!("png: {e}")))?;
    let (width, height, depth) = {
        let info = reader.info();
        check_png_shape(info.color_type, info.bit_depth, info.interlaced)?;
        (info.width, info.height, info.bit_depth)
    };
    if width == 0 || height == 0 {
        return Err(TerrainError::Empty);
    }

    let mut scratch = vec![0.0f64; width as usize];
    let mut y = 0u32;
    while let Some(row) = reader
        .next_row()
        .map_err(|e| TerrainError::Image(format!("png row {y}: {e}")))?
    {
        let bytes = row.data();
        match depth {
            png::BitDepth::Sixteen => {
                // PNG stores 16-bit samples big-endian; `/ 65535` is the unit
                // domain both tilers map from.
                for (i, s) in scratch.iter_mut().enumerate() {
                    let b = i * 2;
                    let v = u16::from_be_bytes([bytes[b], bytes[b + 1]]);
                    *s = v as f64 / 65535.0;
                }
            }
            _ => {
                for (i, s) in scratch.iter_mut().enumerate() {
                    *s = bytes[i] as f64 / 255.0;
                }
            }
        }
        on_row(y, &scratch)?;
        y += 1;
        if y == height {
            break;
        }
    }
    if y != height {
        return Err(TerrainError::Image(format!(
            "png ended after {y} of {height} rows"
        )));
    }
    Ok(HeightmapProbe {
        format: HeightmapFormat::Png,
        width,
        height,
        bit_depth: png_bit_depth(depth),
        float_samples: false,
        channel: "gray".into(),
    })
}

/// Stream an EXR through the `exr` crate's **sequential block** decompressor.
///
/// Each decompressed block is a horizontal band (scanline files) or a tile; its
/// lines are scattered into a small map of partially-filled rows, and a row is
/// handed on the moment its full width is covered. Peak cost is one block plus
/// the rows of that block — never the image.
fn decode_rows_exr<R: BufRead + Seek>(
    src: R,
    on_row: RowSink<'_>,
) -> Result<HeightmapProbe, TerrainError> {
    use exr::block::chunk::TileCoordinates;
    use exr::math::Vec2;
    use exr::meta::attribute::SampleType;

    let reader = exr::block::read(src, false).map_err(|e| exr_err("header", &e))?;
    let header = reader
        .headers()
        .first()
        .ok_or_else(|| TerrainError::Unsupported("EXR file has no layers".into()))?;
    let size = header.layer_size;
    let (width, height) = (size.0, size.1);
    if width == 0 || height == 0 {
        return Err(TerrainError::Empty);
    }
    let channels = header.channels.clone();
    let (channel_name, sample_type) = pick_exr_channel(&channels)?;
    let channel_index = channels
        .list
        .iter()
        .position(|c| c.name.to_string() == channel_name)
        .unwrap_or(0);

    // Only the largest resolution level of the first layer: a mip-mapped EXR
    // carries coarser copies of the same pixels, and blending them into the same
    // row map would silently corrupt the heightfield.
    let chunks = reader
        .filter_chunks(false, |_meta, tile: TileCoordinates, block| {
            block.layer == 0 && tile.level_index == Vec2(0, 0)
        })
        .map_err(|e| exr_err("chunk table", &e))?;

    let mut rows = PartialRows::new(width);
    let mut err: Option<TerrainError> = None;
    let mut scratch_f16: Vec<half::f16> = Vec::new();
    let mut scratch_f32: Vec<f32> = Vec::new();
    let mut scratch_u32: Vec<u32> = Vec::new();

    use exr::block::reader::ChunksReader;
    let outcome = chunks.decompress_sequential(false, |_meta, block| {
        if err.is_some() {
            return Ok(());
        }
        if block.index.layer != 0 || block.index.level != Vec2(0, 0) {
            return Ok(());
        }
        for line in block.lines(&channels) {
            if line.location.channel != channel_index {
                continue;
            }
            let n = line.location.sample_count;
            let x0 = line.location.position.0;
            let y = line.location.position.1;
            if y >= height {
                continue;
            }
            let read = match sample_type {
                SampleType::F16 => {
                    scratch_f16.resize(n, half::f16::ZERO);
                    line.read_samples_into_slice(&mut scratch_f16[..n])
                        .map(|()| rows.write(y, x0, scratch_f16[..n].iter().map(|v| v.to_f64())))
                }
                SampleType::F32 => {
                    scratch_f32.resize(n, 0.0);
                    line.read_samples_into_slice(&mut scratch_f32[..n])
                        .map(|()| rows.write(y, x0, scratch_f32[..n].iter().map(|&v| v as f64)))
                }
                SampleType::U32 => {
                    scratch_u32.resize(n, 0);
                    line.read_samples_into_slice(&mut scratch_u32[..n])
                        .map(|()| {
                            rows.write(
                                y,
                                x0,
                                scratch_u32[..n].iter().map(|&v| v as f64 / u32::MAX as f64),
                            )
                        })
                }
            };
            match read {
                Ok(true) => {
                    let row = rows.take(y).expect("a complete row is present");
                    if let Err(e) = on_row(y as u32, &row) {
                        err = Some(e);
                        return Ok(());
                    }
                    rows.recycle(row);
                }
                Ok(false) => {}
                Err(e) => {
                    err = Some(exr_err("line", &e));
                    return Ok(());
                }
            }
        }
        Ok(())
    });
    if let Some(e) = err {
        return Err(e);
    }
    outcome.map_err(|e| exr_err("blocks", &e))?;
    if !rows.is_drained() {
        return Err(TerrainError::Image(format!(
            "exr left {} incomplete row(s) — the file does not cover its data window",
            rows.len()
        )));
    }

    Ok(HeightmapProbe {
        format: HeightmapFormat::Exr,
        width: width as u32,
        height: height as u32,
        bit_depth: match sample_type {
            SampleType::F16 => 16,
            _ => 32,
        },
        float_samples: !matches!(sample_type, SampleType::U32),
        channel: channel_name,
    })
}

/// Rows being filled in from EXR line fragments. A row leaves as soon as its
/// full width is covered, so a scanline file never holds more than one block's
/// worth and a tiled file never more than one tile-row's.
struct PartialRows {
    width: usize,
    open: std::collections::BTreeMap<usize, (Vec<f64>, usize)>,
    pool: Vec<Vec<f64>>,
}

impl PartialRows {
    fn new(width: usize) -> Self {
        Self {
            width,
            open: std::collections::BTreeMap::new(),
            pool: Vec::new(),
        }
    }

    /// Write `values` starting at column `x0` of row `y`; `true` when the row is
    /// now complete.
    fn write(&mut self, y: usize, x0: usize, values: impl Iterator<Item = f64>) -> bool {
        let width = self.width;
        let fresh = self.pool.pop();
        let (row, filled) = self
            .open
            .entry(y)
            .or_insert_with(|| (fresh.unwrap_or_else(|| vec![0.0; width]), 0));
        row.resize(width, 0.0);
        let mut written = 0usize;
        for (i, v) in values.enumerate() {
            let x = x0 + i;
            if x < width {
                row[x] = v;
                written += 1;
            }
        }
        *filled += written;
        *filled >= width
    }

    fn take(&mut self, y: usize) -> Option<Vec<f64>> {
        self.open.remove(&y).map(|(row, _)| row)
    }

    fn recycle(&mut self, row: Vec<f64>) {
        if self.pool.len() < 4 {
            self.pool.push(row);
        }
    }

    fn is_drained(&self) -> bool {
        self.open.is_empty()
    }

    fn len(&self) -> usize {
        self.open.len()
    }
}

// ── the whole-image tiler ───────────────────────────────────────────────────

impl TerrainData {
    /// Decode a `PNG`/`EXR` heightmap **whole** and page it into a [`TerrainData`].
    ///
    /// The `W × H` sample grid is tiled into `resolution × resolution` pages with
    /// **shared edges** (a tile's last row/column is the next tile's first), so the
    /// result is seamless; indices past the source clamp onto its edge. Samples map
    /// through [`HeightmapImport::map_sample`] (stored as the tile-local `f32`
    /// offset, `origin.y = 0`).
    ///
    /// This is the byte-identity **reference** for
    /// [`crate::chunked::import_heightmap`]: same decoder, same mapping, batch
    /// tiling instead of streaming. Use the chunked path for anything large — this
    /// one holds the whole `W · H` sample grid.
    pub fn from_height_image(bytes: &[u8], import: HeightmapImport) -> Result<Self, TerrainError> {
        import.validate(None)?;
        let mut samples: Vec<f64> = Vec::new();
        let mut width = 0usize;
        let probe = decode_rows(Cursor::new(bytes), &mut |_y, row| {
            width = row.len();
            samples.extend_from_slice(row);
            Ok(())
        })?;
        import.validate(Some(&probe))?;
        let (w, h) = (probe.width, probe.height);
        if w == 0 || h == 0 || samples.is_empty() {
            return Err(TerrainError::Empty);
        }

        let grid = HeightmapGrid::new(w, h, &import);
        let res = grid.resolution;
        let mut data = TerrainData::new(res, import.meters_per_sample);
        let sample_at = |gx: i32, gz: i32| -> f64 {
            let gx = gx.clamp(0, w as i32 - 1) as usize;
            let gz = gz.clamp(0, h as i32 - 1) as usize;
            import.map_sample(samples[gz * width + gx])
        };
        for tz in 0..grid.ntz {
            for tx in 0..grid.ntx {
                let coord = grid.coord(tx, tz);
                let o = data.tile_origin_xz(coord);
                let tile = data.get_or_create_tile(coord);
                tile.origin = DVec3::new(o.x, 0.0, o.y);
                for j in 0..res {
                    for i in 0..res {
                        let gx = tx * grid.cells + i as i32;
                        let gz = tz * grid.cells + j as i32;
                        tile.set_sample(res, i, j, sample_at(gx, gz) as f32);
                    }
                }
            }
        }
        data.clear_dirty();
        Ok(data)
    }

    /// Reconstruct the global 16-bit sample grid from the authored tiles, mapping
    /// world heights back through `[min_height, max_height]` → `[0, 65535]`.
    ///
    /// The grid spans the bounding rectangle of authored tile coordinates with
    /// shared edges (`width = ntx·(res−1)+1`). Unauthored holes read as height `0`.
    /// Returns `None` for an empty terrain.
    pub fn to_height_image(&self, min_height: f64, max_height: f64) -> Option<HeightImage> {
        let mut it = self.tiles();
        let (&(mut min_tx, mut min_tz), _) = it.next()?;
        let (mut max_tx, mut max_tz) = (min_tx, min_tz);
        for (&(tx, tz), _) in self.tiles() {
            min_tx = min_tx.min(tx);
            min_tz = min_tz.min(tz);
            max_tx = max_tx.max(tx);
            max_tz = max_tz.max(tz);
        }
        let res = self.tile_resolution();
        let cells = (res - 1) as i32;
        let ntx = max_tx - min_tx + 1;
        let ntz = max_tz - min_tz + 1;
        let width = (ntx * cells + 1) as u32;
        let height = (ntz * cells + 1) as u32;
        let span = (max_height - min_height).abs().max(f64::MIN_POSITIVE);

        let mut samples = vec![0u16; (width * height) as usize];
        for gz in 0..height as i32 {
            for gx in 0..width as i32 {
                // Locate the owning tile + local sample (clamp the far edge back
                // onto the last authored column/row — shared-edge equivalence).
                let mut tx = min_tx + gx / cells;
                let mut i = gx - (tx - min_tx) * cells;
                if tx > max_tx {
                    tx = max_tx;
                    i = cells;
                }
                let mut tz = min_tz + gz / cells;
                let mut j = gz - (tz - min_tz) * cells;
                if tz > max_tz {
                    tz = max_tz;
                    j = cells;
                }
                let world_h = self
                    .get_tile((tx, tz))
                    .map(|t| t.world_height(res, i as u32, j as u32))
                    .unwrap_or(0.0);
                let norm = ((world_h - min_height) / span).clamp(0.0, 1.0);
                samples[(gz as u32 * width + gx as u32) as usize] = (norm * 65535.0).round() as u16;
            }
        }
        Some(HeightImage {
            width,
            height,
            samples,
        })
    }

    /// Export the terrain to 16-bit PNG bytes over `[min_height, max_height]`.
    pub fn export_png16(&self, min_height: f64, max_height: f64) -> Result<Vec<u8>, TerrainError> {
        let img = self
            .to_height_image(min_height, max_height)
            .ok_or(TerrainError::Empty)?;
        encode_png16(&img)
    }
}

/// Encode a 16-bit grayscale [`HeightImage`] to PNG bytes.
pub fn encode_png16(img: &HeightImage) -> Result<Vec<u8>, TerrainError> {
    if img.width == 0 || img.height == 0 || img.samples.len() != (img.width * img.height) as usize {
        return Err(TerrainError::Empty);
    }
    let buf: ImageBuffer<Luma<u16>, Vec<u16>> =
        ImageBuffer::from_raw(img.width, img.height, img.samples.clone())
            .ok_or(TerrainError::Empty)?;
    let mut out = Vec::new();
    DynamicImage::ImageLuma16(buf)
        .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| TerrainError::Image(e.to_string()))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(width: u32, height: u32) -> HeightImage {
        let samples = (0..width * height)
            .map(|i| ((i as u64 * 7919) % 65536) as u16)
            .collect();
        HeightImage {
            width,
            height,
            samples,
        }
    }

    #[test]
    fn probe_reads_png_headers_without_decoding() {
        let png = encode_png16(&ramp(37, 11)).unwrap();
        let p = probe_heightmap_bytes(&png).unwrap();
        assert_eq!(p.format, HeightmapFormat::Png);
        assert_eq!((p.width, p.height), (37, 11));
        assert_eq!(p.bit_depth, 16);
        assert!(!p.float_samples);
    }

    #[test]
    fn png_rows_decode_in_order_and_in_the_unit_domain() {
        let img = ramp(9, 4);
        let png = encode_png16(&img).unwrap();
        let mut seen = Vec::new();
        let probe = decode_rows(Cursor::new(&png), &mut |y, row| {
            seen.push((y, row.to_vec()));
            Ok(())
        })
        .unwrap();
        assert_eq!(probe.height, 4);
        assert_eq!(seen.len(), 4);
        for (y, row) in &seen {
            assert_eq!(row.len(), 9);
            for (x, v) in row.iter().enumerate() {
                let expect = img.samples[(*y * 9 + x as u32) as usize] as f64 / 65535.0;
                assert_eq!(*v, expect, "row {y} col {x}");
            }
        }
        assert_eq!(
            seen.iter().map(|(y, _)| *y).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    /// The **8-bit grayscale arm** of the PNG row decoder. `v / 255` is the same
    /// unit value `image`'s `to_luma16` would have produced (it widens by ×257,
    /// and `257·v / 65535 == v / 255` exactly), so an 8-bit source maps onto the
    /// height range identically through either path — with a quarter of the
    /// levels, which is exactly why 16-bit is what the wizard asks for.
    #[test]
    fn eight_bit_grayscale_pngs_decode_on_the_same_unit_scale() {
        let mut buf: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(4, 3);
        for (i, p) in buf.pixels_mut().enumerate() {
            *p = Luma([(i as u8) * 20]);
        }
        let mut png = Vec::new();
        DynamicImage::ImageLuma8(buf)
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();

        let probe = probe_heightmap_bytes(&png).unwrap();
        assert_eq!(probe.bit_depth, 8);
        assert_eq!((probe.width, probe.height), (4, 3));

        let mut seen = Vec::new();
        decode_rows(Cursor::new(&png), &mut |_y, row| {
            seen.extend_from_slice(row);
            Ok(())
        })
        .unwrap();
        for (i, v) in seen.iter().enumerate() {
            let raw = (i as u8) * 20;
            assert_eq!(*v, raw as f64 / 255.0, "sample {i}");
            // …and that is bit-for-bit the 16-bit widening `image` applies.
            assert_eq!(
                *v,
                (raw as u16 * 257) as f64 / 65535.0,
                "sample {i} widened"
            );
        }
        // Through a real import it lands on the stated range.
        let import = HeightmapImport {
            tile_resolution: 2,
            meters_per_sample: 1.0,
            min_height: 0.0,
            max_height: 255.0,
            ..Default::default()
        };
        let t = TerrainData::from_height_image(&png, import).unwrap();
        assert_eq!(t.get_tile((0, 0)).unwrap().sample(2, 0, 0), 0.0);
        assert_eq!(t.get_tile((0, 0)).unwrap().sample(2, 1, 0), 20.0);
    }

    #[test]
    fn colour_and_interlaced_pngs_are_refused_not_luma_averaged() {
        // An RGB PNG: a heightmap is a scalar field, so this is a user error we
        // name rather than a silent weighted average.
        let buf: ImageBuffer<image::Rgb<u8>, Vec<u8>> = ImageBuffer::new(4, 4);
        let mut rgb = Vec::new();
        DynamicImage::ImageRgb8(buf)
            .write_to(&mut Cursor::new(&mut rgb), image::ImageFormat::Png)
            .unwrap();
        let err = probe_heightmap_bytes(&rgb).unwrap_err();
        assert!(
            matches!(&err, TerrainError::Unsupported(m) if m.contains("grayscale")),
            "got {err}"
        );
    }

    #[test]
    fn settings_are_validated_before_any_decode() {
        let base = HeightmapImport::default();
        assert!(HeightmapImport {
            tile_resolution: 1,
            ..base
        }
        .validate(None)
        .is_err());
        assert!(HeightmapImport {
            meters_per_sample: 0.0,
            ..base
        }
        .validate(None)
        .is_err());
        assert!(HeightmapImport {
            min_height: 10.0,
            max_height: 10.0,
            ..base
        }
        .validate(None)
        .is_err());
        // Float-metres on an integer source is refused, not reinterpreted.
        let png_probe = HeightmapProbe {
            format: HeightmapFormat::Png,
            width: 8,
            height: 8,
            bit_depth: 16,
            float_samples: false,
            channel: "gray".into(),
        };
        assert!(HeightmapImport {
            mode: HeightMode::FloatMeters,
            ..base
        }
        .validate(Some(&png_probe))
        .is_err());
    }

    #[test]
    fn the_lattice_covers_the_source_with_shared_edges() {
        let import = HeightmapImport {
            tile_resolution: 5,
            ..Default::default()
        };
        // 9x9 is exactly 2x2 tiles of resolution 5 (cells = 4).
        let g = HeightmapGrid::new(9, 9, &import);
        assert_eq!((g.ntx, g.ntz), (2, 2));
        // 10x9 needs a third column that only partly covers the source.
        let g = HeightmapGrid::new(10, 9, &import);
        assert_eq!((g.ntx, g.ntz), (3, 2));
        assert_eq!(g.last_source_row(), 8);
        // Centring is integral and leaves the extra tile on the + side.
        assert_eq!(HeightmapGrid::centered_origin(9, 9, &import), (-1, -1));
        assert_eq!(HeightmapGrid::centered_origin(13, 9, &import), (-1, -1));
        assert_eq!(HeightmapGrid::centered_origin(17, 17, &import), (-2, -2));
    }

    #[test]
    fn the_negative_tile_origin_shifts_the_world_position() {
        let img = ramp(9, 9);
        let png = encode_png16(&img).unwrap();
        let base = HeightmapImport {
            tile_resolution: 5,
            meters_per_sample: 2.0,
            min_height: 0.0,
            max_height: 100.0,
            ..Default::default()
        };
        let at_origin = TerrainData::from_height_image(&png, base).unwrap();
        let centered = TerrainData::from_height_image(
            &png,
            HeightmapImport {
                tile_origin: (-1, -1),
                ..base
            },
        )
        .unwrap();
        assert_eq!(at_origin.tile_count(), centered.tile_count());
        // The same source sample lands at the same height, shifted by one tile.
        let span = at_origin.tile_span();
        let a = at_origin.get_tile((0, 0)).unwrap();
        let b = centered.get_tile((-1, -1)).unwrap();
        assert_eq!(a.heights(), b.heights());
        assert_eq!(b.origin.x, -span);
        assert_eq!(b.origin.z, -span);
    }

    #[test]
    fn float_metres_mode_keeps_absolute_values() {
        let import = HeightmapImport {
            mode: HeightMode::FloatMeters,
            min_height: -1000.0,
            max_height: 1000.0,
            ..Default::default()
        };
        assert_eq!(import.map_sample(1234.5), 1234.5);
        assert_eq!(import.map_sample(-42.0), -42.0);
        // Normalized clamps out-of-range floats instead of escaping the extent.
        let norm = HeightmapImport {
            mode: HeightMode::Normalized,
            min_height: 0.0,
            max_height: 100.0,
            ..Default::default()
        };
        assert_eq!(norm.map_sample(0.5), 50.0);
        assert_eq!(norm.map_sample(3.0), 100.0);
        assert_eq!(norm.map_sample(-3.0), 0.0);
    }

    #[test]
    fn world_extent_is_metric() {
        let import = HeightmapImport {
            meters_per_sample: 8.0,
            ..Default::default()
        };
        // 2049 samples at 8 m = 16 384 m = 16.384 km across.
        assert_eq!(import.world_extent(2049, 1025).x, 16384.0);
        assert_eq!(import.world_extent(2049, 1025).y, 8192.0);
    }
}
