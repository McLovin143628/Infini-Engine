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
                    // `absolute_samples`, not `float_samples` — see the field's
                    // own docs. A 16-bit integer GeoTIFF DEM carries absolute
                    // metres and must be allowed into this mode; a 16-bit PNG
                    // carries counts and must not.
                    if !p.absolute_samples {
                        return Err(TerrainError::Settings(format!(
                            "float-metres mode needs a source whose samples are \
                             absolute elevations; {} carries scaled integer samples, \
                             which mean nothing without a stated height range",
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
    ///
    /// **The tile counts are derived in `i64`** (C4-17). `(width as i32 - 1)` on
    /// a source that legally declares `i32::MAX` (or more — PNG width is a `u32`
    /// field) wrapped negative, `.max(0)` pulled it to zero, and `.max(1)`
    /// handed back **one tile**: a two-billion-sample source became a single
    /// tile with no error, reachable from `probe` alone, which reads an 8-byte
    /// IHDR and no pixels. `check_png_shape` now refuses those dimensions
    /// outright; this is the arithmetic behind that door.
    pub fn new(width: u32, height: u32, import: &HeightmapImport) -> Self {
        let resolution = import.resolution();
        let cells = (resolution - 1) as i32;
        let tiles = |n: u32| -> i32 {
            let cells = i64::from(cells);
            let span = (i64::from(n) - 1).max(0);
            (((span + cells - 1) / cells).max(1)).min(i64::from(i32::MAX)) as i32
        };
        let ntx = tiles(width);
        let ntz = tiles(height);
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
    /// TIFF, including **GeoTIFF** — the format real elevation data ships in
    /// (Wave G). See [`crate::geotiff`] for the georeferencing half.
    Tiff,
}

impl HeightmapFormat {
    /// A short label for messages and the wizard.
    pub const fn label(self) -> &'static str {
        match self {
            HeightmapFormat::Png => "PNG",
            HeightmapFormat::Exr => "EXR",
            HeightmapFormat::Tiff => "TIFF",
        }
    }
}

// ── the nodata policy door (Wave G) ─────────────────────────────────────────

/// What to do with a sample the source says it has no data for.
///
/// # Why this door has to exist
///
/// Real-world DEMs are **full** of no-data: ocean beyond the survey, the ragged
/// edge of a flight line, LiDAR voids under water and dense canopy, cloud
/// shadow. It arrives three ways — as `NaN` in a float raster, as a declared
/// sentinel (`-9999`, `-32768`, `-3.4e38`) in the GeoTIFF `GDAL_NODATA` tag, or
/// as an undeclared sentinel the publisher simply knows about.
///
/// Before Wave G this engine refused every one of them. The refusal was right,
/// and its reasoning still is:
///
/// > *"a no-data pixel means the author's source does not cover that ground, and
/// > this engine has no representation for a hole in a heightfield"*
///
/// But "refuse" as the **only** behaviour means the very first real GeoTIFF
/// anyone imports stops dead at the first ocean pixel, and the author's only
/// recourse is to go and edit the DEM. So the refusal becomes the *default* of a
/// policy the author can change, and — this is the part that matters — the
/// choice is **recorded in the sidecar**, so re-importing the same file
/// reproduces the same terrain rather than depending on what somebody clicked.
///
/// # The two rules that fall out
///
/// 1. **The policy runs BEFORE the finiteness door, not instead of it.** A
///    non-finite sample that survives the policy is still a bug and must still
///    refuse. Substituting first and checking second is what keeps the guarantee
///    that nothing non-finite is ever written into an `.inf_terrain`.
/// 2. **The policy is sidecar data, not wire data.** The `.inf_terrain` sidecar
///    is self-describing TOML, so recording it costs **no schema bump**.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum NodataPolicy {
    /// Refuse the import, naming the first offending sample. **The default**,
    /// and the behaviour every import had before Wave G.
    #[default]
    Refuse,
    /// Substitute a stated elevation, in the source's own vertical units.
    ///
    /// `Clamp(0.0)` — sea level — is the common answer for a coastal DEM whose
    /// no-data is ocean, which is why [`NodataPolicy::sea_level`] names it.
    Clamp(f64),
    /// Fill a void from the valid samples on either side of it **within its own
    /// row**, linearly, for runs up to `max_span` samples wide. Wider runs
    /// refuse.
    ///
    /// # Why "within its own row", stated plainly
    ///
    /// This decoder is row-at-a-time by design — that is the whole reason a
    /// 16 k × 16 k source imports in ~100 MB instead of 1 GB (the P16.4a memory
    /// bound). A true nearest-valid-neighbour fill needs the rows above and
    /// below, which means buffering the image and giving that bound up.
    ///
    /// A row-wise fill is what a streaming decoder can honestly do. It closes
    /// the case it is meant for — the scattered small voids that pepper LiDAR —
    /// and `max_span` is what stops it from smearing a coastline across an ocean
    /// it cannot see the far side of. A 2-D fill is a named remainder, not a
    /// pretence.
    FillRow { max_span: u32 },
}

impl NodataPolicy {
    /// [`NodataPolicy::Clamp`] at sea level — the usual answer for a coastal DEM.
    pub const fn sea_level() -> Self {
        NodataPolicy::Clamp(0.0)
    }

    /// A stable label for the sidecar and the wizard.
    pub fn label(self) -> String {
        match self {
            NodataPolicy::Refuse => "refuse".into(),
            NodataPolicy::Clamp(v) => format!("clamp:{v}"),
            NodataPolicy::FillRow { max_span } => format!("fill-row:{max_span}"),
        }
    }

    /// Parse a label back. Returns `None` for an unrecognised spelling, so a
    /// sidecar written by a newer build is refused rather than silently
    /// downgraded to `Refuse` — which would change the terrain.
    pub fn from_label(s: &str) -> Option<Self> {
        let t = s.trim();
        if t.eq_ignore_ascii_case("refuse") {
            return Some(NodataPolicy::Refuse);
        }
        if let Some(v) = t.strip_prefix("clamp:") {
            return v
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .map(NodataPolicy::Clamp);
        }
        if let Some(v) = t.strip_prefix("fill-row:") {
            return v
                .trim()
                .parse::<u32>()
                .ok()
                .map(|max_span| NodataPolicy::FillRow { max_span });
        }
        None
    }
}

/// The nodata policy plus the sentinel it acts on.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct NodataHandling {
    pub policy: NodataPolicy,
    /// The value that means "no data" in the source's own domain.
    ///
    /// Read from `GDAL_NODATA` when the file declares one; an author may also
    /// state it for a file that does not. `NaN` is *always* treated as no-data
    /// whether or not a sentinel is set, because that is what a float raster
    /// conventionally means by it.
    pub sentinel: Option<f64>,
}

impl NodataHandling {
    /// Nothing to do — the pre-Wave-G behaviour, and what a source with no
    /// no-data gets.
    pub const NONE: Self = Self {
        policy: NodataPolicy::Refuse,
        sentinel: None,
    };

    /// Whether a sample is no-data.
    ///
    /// Non-finite is always no-data. A declared sentinel matches on a
    /// **relative** tolerance rather than exact equality: a sentinel like
    /// `-3.4028234663852886e+38` reaches us as text through `GDAL_NODATA` and as
    /// an `f32` through the pixels, and the two round-trips need not land on the
    /// same double. An exact compare would silently miss every one of them.
    #[inline]
    pub fn is_nodata(&self, v: f64) -> bool {
        if !v.is_finite() {
            return true;
        }
        match self.sentinel {
            Some(s) if s == v => true,
            Some(s) => {
                let scale = s.abs().max(v.abs());
                scale > 0.0 && (s - v).abs() / scale < 1e-9
            }
            None => false,
        }
    }

    /// `true` when this handling can actually change a sample.
    pub fn is_active(&self) -> bool {
        !matches!(self.policy, NodataPolicy::Refuse)
    }
}

/// What the nodata policy did, for the import report and the cook advisory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodataReport {
    /// Samples replaced by the policy.
    pub substituted: u64,
    /// Runs of consecutive no-data filled by [`NodataPolicy::FillRow`].
    pub filled_runs: u64,
    /// The widest run seen, in samples. Reported because an author who set
    /// `max_span` to 64 and imported a DEM whose widest void is 63 got away with
    /// it by one sample, and should be told.
    pub widest_run: u32,
}

impl NodataReport {
    /// `true` when the policy touched anything — the trigger for the cook
    /// advisory (the P16 law: a silent hazard gets named).
    pub fn engaged(&self) -> bool {
        self.substituted > 0
    }
}

/// Apply the nodata policy to one decoded row, in place.
///
/// Runs before the finiteness door — see [`NodataPolicy`] for why that ordering
/// is load-bearing.
fn apply_nodata(
    handling: &NodataHandling,
    report: &mut NodataReport,
    y: u32,
    row: &mut [f64],
) -> Result<(), TerrainError> {
    if row.is_empty() {
        return Ok(());
    }
    match handling.policy {
        NodataPolicy::Refuse => Ok(()),
        NodataPolicy::Clamp(v) => {
            for s in row.iter_mut() {
                if handling.is_nodata(*s) {
                    *s = v;
                    report.substituted += 1;
                }
            }
            Ok(())
        }
        NodataPolicy::FillRow { max_span } => {
            let n = row.len();
            let mut i = 0usize;
            while i < n {
                if !handling.is_nodata(row[i]) {
                    i += 1;
                    continue;
                }
                let start = i;
                while i < n && handling.is_nodata(row[i]) {
                    i += 1;
                }
                let end = i; // exclusive
                let span = (end - start) as u32;
                report.widest_run = report.widest_run.max(span);
                if span > max_span {
                    return Err(TerrainError::Image(format!(
                        "row {y} has a run of {span} no-data samples starting at \
                         column {start}, wider than the {max_span}-sample limit this \
                         import's fill policy allows. Filling it would invent {span} \
                         samples of terrain from two endpoints that are {span} \
                         samples apart — raise the limit if that is really what you \
                         want, or choose a clamp elevation instead so the gap is \
                         flat and honest rather than a smear."
                    )));
                }
                // Endpoints. A run touching the row's edge has only one, and
                // extends it flat rather than inventing a slope.
                let left = (start > 0).then(|| row[start - 1]);
                let right = (end < n).then(|| row[end]);
                let (a, b) = match (left, right) {
                    (Some(a), Some(b)) => (a, b),
                    (Some(a), None) => (a, a),
                    (None, Some(b)) => (b, b),
                    (None, None) => {
                        return Err(TerrainError::Image(format!(
                            "row {y} is entirely no-data ({n} samples), so there is \
                             nothing in it to fill from. A DEM whose rows are wholly \
                             absent needs a clamp elevation, not a fill."
                        )))
                    }
                };
                for (k, slot) in row[start..end].iter_mut().enumerate() {
                    let t = (k + 1) as f64 / (span + 1) as f64;
                    *slot = a + (b - a) * t;
                }
                report.substituted += u64::from(span);
                report.filled_runs += 1;
            }
            Ok(())
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
    /// `true` when samples are IEEE floats.
    pub float_samples: bool,
    /// `true` when the samples are **absolute elevations**, so
    /// [`HeightMode::FloatMeters`] is meaningful.
    ///
    /// # Why this is not the same question as `float_samples`
    ///
    /// It used to be: a PNG's samples are counts on a `0..65535` scale and mean
    /// nothing without a stated height range, while an EXR float is an elevation,
    /// so "is it a float" and "is it an elevation" coincided and one flag served
    /// both.
    ///
    /// GeoTIFF breaks that. A 16-bit **integer** DEM — which is what SRTM and a
    /// great many national products ship as — stores absolute metres in a
    /// `u16`/`i16`. It is not a float and it *is* an elevation. Reading it in
    /// normalized mode would map the ocean-to-summit range of the file onto
    /// whatever height range the wizard happened to show, which is a landscape
    /// with the right shape and the wrong scale — the failure that looks like an
    /// authoring choice.
    ///
    /// So the two questions are separated. `float_samples` stays what it always
    /// meant (and keeps driving the wizard's "float" readout); this is what
    /// [`HeightmapImport::validate`] actually asks about.
    pub absolute_samples: bool,
    /// The channel the decoder will read (EXR channel name, or `"gray"`).
    pub channel: String,
    /// GeoTIFF georeferencing, when the source carried any (Wave G).
    ///
    /// Derived from the header, **never persisted** — the same rule the rest of
    /// this struct follows.
    pub geo: Option<crate::geotiff::GeoTiffMeta>,
}

/// The eight-byte PNG signature.
const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
/// The four-byte OpenEXR magic (`0x76 0x2f 0x31 0x01`).
const EXR_MAGIC: [u8; 4] = [0x76, 0x2f, 0x31, 0x01];
/// Classic TIFF, little- and big-endian: `II*\0` and `MM\0*`.
const TIFF_MAGIC_LE: [u8; 4] = [b'I', b'I', 42, 0];
const TIFF_MAGIC_BE: [u8; 4] = [b'M', b'M', 0, 42];
/// **BigTIFF** — version 43 instead of 42, for files past 4 GB. Real
/// state-wide 1 m DEMs cross that line routinely, so the signature is
/// recognised on purpose: a BigTIFF must be refused *as a BigTIFF*, naming the
/// remedy, rather than as "not a PNG or EXR heightmap".
const BIGTIFF_MAGIC_LE: [u8; 4] = [b'I', b'I', 43, 0];
const BIGTIFF_MAGIC_BE: [u8; 4] = [b'M', b'M', 0, 43];

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
    if n >= 4 && (head[..4] == TIFF_MAGIC_LE || head[..4] == TIFF_MAGIC_BE) {
        return Ok(HeightmapFormat::Tiff);
    }
    if n >= 4 && (head[..4] == BIGTIFF_MAGIC_LE || head[..4] == BIGTIFF_MAGIC_BE) {
        return Err(TerrainError::Unsupported(
            "this is a BigTIFF (the >4 GB TIFF variant, version 43). This engine's \
             TIFF reader handles classic TIFF only. A state-wide 1 m DEM is often \
             published this way — cut the area you need out of it first, with \
             `gdal_translate -projwin <west> <north> <east> <south> in.tif out.tif`, \
             which also gets you a file this importer can stream without holding \
             the whole thing."
                .into(),
        ));
    }
    Err(TerrainError::Unsupported(
        "not a PNG, EXR or TIFF heightmap (unrecognized file signature)".into(),
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
        HeightmapFormat::Tiff => probe_tiff(src)?,
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
    check_dimensions(w, h)?;
    Ok(HeightmapProbe {
        format: HeightmapFormat::Png,
        width: w,
        height: h,
        bit_depth: depth,
        float_samples: false,
        // A PNG's samples are counts on a 0..65535 scale, not elevations.
        absolute_samples: false,
        channel: "gray".into(),
        geo: None,
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
/// The largest heightmap dimension the importer will accept, in samples.
///
/// 1 048 576 samples on a side is 2 TB of 16-bit source and a lattice of
/// millions of tiles — comfortably past anything real, which is the point: the
/// number is a ceiling that keeps every downstream `i32`/`u32` product exact,
/// not a capability claim. Above it the file is refused with its own dimensions
/// named, which is a better answer than a silent one-tile import (C4-17).
const MAX_HEIGHTMAP_SIDE: u32 = 1 << 20;

/// Refuse a source whose declared dimensions the lattice arithmetic cannot
/// carry (C4-17).
///
/// Reachable from `probe` alone: a PNG's `IHDR` is eight bytes and declares two
/// `u32`s, so this costs nothing and fires before a single pixel is read.
fn check_dimensions(width: u32, height: u32) -> Result<(), TerrainError> {
    if width == 0 || height == 0 {
        return Err(TerrainError::Unsupported(format!(
            "heightmap declares a {width}×{height} image"
        )));
    }
    if width > MAX_HEIGHTMAP_SIDE || height > MAX_HEIGHTMAP_SIDE {
        return Err(TerrainError::Unsupported(format!(
            "heightmap is {width}×{height}; this importer accepts up to \
             {MAX_HEIGHTMAP_SIDE} samples on a side"
        )));
    }
    Ok(())
}

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
    check_dimensions(size.0 as u32, size.1 as u32)?;
    Ok(HeightmapProbe {
        format: HeightmapFormat::Exr,
        width: size.0 as u32,
        height: size.1 as u32,
        bit_depth,
        float_samples,
        // An EXR float channel IS an elevation; a u32 channel is a count.
        absolute_samples: float_samples,
        channel,
        geo: None,
    })
}

// ── TIFF / GeoTIFF (Wave G) ─────────────────────────────────────────────────

/// What one TIFF sample type is, reduced to what this decoder needs to know.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TiffSampleKind {
    bits: u32,
    float: bool,
    /// `true` when the raw sample value is an absolute elevation rather than a
    /// count needing a height range — see [`HeightmapProbe::absolute_samples`].
    absolute: bool,
}

/// Read the header facts a TIFF probe needs, without decoding a pixel.
fn tiff_header<R: BufRead + Seek>(
    src: R,
) -> Result<
    (
        tiff::decoder::Decoder<R>,
        u32,
        u32,
        TiffSampleKind,
        crate::geotiff::GeoTiffMeta,
    ),
    TerrainError,
> {
    use tiff::tags::{SampleFormat, Tag};

    let mut decoder = tiff::decoder::Decoder::new(src)
        // Real DEMs are large; the default limits are sized for web images and
        // would refuse a 16 k source that this importer streams comfortably.
        .map_err(|e| tiff_err("header", &e))?
        .with_limits(tiff::decoder::Limits::unlimited());
    let (width, height) = decoder
        .dimensions()
        .map_err(|e| tiff_err("dimensions", &e))?;
    check_dimensions(width, height)?;

    let spp = decoder.get_tag_u32(Tag::SamplesPerPixel).unwrap_or(1);
    if spp != 1 {
        return Err(TerrainError::Unsupported(format!(
            "TIFF heightmaps must carry one sample per pixel (this one has {spp}) \
             — a heightmap is a scalar field, and averaging colour channels into \
             an elevation is a guess this importer will not make. If this is an \
             RGB terrarium elevation tile, it is a different format with its own \
             door; otherwise re-export the elevation band on its own with \
             `gdal_translate -b 1 in.tif out.tif`."
        )));
    }
    let bits = decoder.get_tag_u32(Tag::BitsPerSample).unwrap_or(8);
    let format = decoder
        .find_tag_unsigned::<u16>(Tag::SampleFormat)
        .ok()
        .flatten()
        .map(SampleFormat::from_u16_exhaustive)
        .unwrap_or(SampleFormat::Uint);

    let geo = crate::geotiff::read_meta(&mut decoder)?;

    let float = matches!(format, SampleFormat::IEEEFP);
    // What makes a TIFF sample an ELEVATION rather than a count:
    //   * it is a float — nobody stores normalized 0..1 heights as f32 in a TIFF;
    //   * or it is SIGNED — a signed integer raster is an elevation, since the
    //     negative half would be meaningless as a count;
    //   * or the file is georeferenced — a GeoTIFF with a pixel scale and a tie
    //     point is a DEM, whatever its sample type.
    // An unsigned, ungeoreferenced TIFF is treated like a PNG: a count.
    let absolute = float || matches!(format, SampleFormat::Int) || geo.is_georeferenced();
    Ok((
        decoder,
        width,
        height,
        TiffSampleKind {
            bits,
            float,
            absolute,
        },
        geo,
    ))
}

fn probe_tiff<R: BufRead + Seek>(src: R) -> Result<HeightmapProbe, TerrainError> {
    let (_decoder, width, height, kind, geo) = tiff_header(src)?;
    Ok(HeightmapProbe {
        format: HeightmapFormat::Tiff,
        width,
        height,
        bit_depth: kind.bits,
        float_samples: kind.float,
        absolute_samples: kind.absolute,
        channel: "gray".into(),
        geo: Some(geo),
    })
}

fn tiff_err(what: &str, e: &tiff::TiffError) -> TerrainError {
    let msg = e.to_string();
    // image-tiff has NO LERC and NO JPEG2000 decoder, and ArcGIS — which is what
    // most US government portals are built on — produces LERC-compressed COGs
    // routinely. "unsupported compression" would leave an author with nowhere to
    // go; naming the remedy is the whole difference.
    if msg.contains("ompression") || msg.contains("LERC") || msg.contains("Lerc") {
        return TerrainError::Unsupported(format!(
            "this TIFF uses a compression this engine cannot read ({msg}). The \
             pure-Rust TIFF reader handles uncompressed, Deflate/zip, LZW and \
             PackBits. It does NOT handle LERC or JPEG2000 — and ArcGIS, which \
             most government elevation portals run on, produces LERC-compressed \
             files by default. Re-save it once on your machine with \
             `gdal_translate -co COMPRESS=DEFLATE in.tif out.tif` and it will \
             import; nothing about the elevations changes."
        ));
    }
    TerrainError::Image(format!("tiff {what}: {msg}"))
}

/// Stream a TIFF **one chunk at a time** — the strip or tile row the format
/// itself is built out of.
///
/// # The memory bound survives the new format
///
/// This is the whole reason `tiff` was the crate chosen: `read_chunk` +
/// `chunk_dimensions` is the same chunk-at-a-time seam that `png::Reader::next_row`
/// and `exr`'s `decompress_sequential` gave us, so the P16.4a guarantee — a
/// 16 k × 16 k source imports inside a live set of `< 6·ntx` tiles rather than
/// materialising a 1 GB grid — holds for GeoTIFF exactly as it does for the
/// other two. Peak cost here is one chunk row: for a striped file that is a
/// handful of source rows, for a tiled file it is one tile row.
///
/// # Elevations arrive in metres
///
/// The vertical unit conversion ([`crate::geotiff::VerticalUnits`]) is applied
/// here and **only** here. A source in feet hands this function feet and hands
/// its caller metres.
fn decode_rows_tiff<R: BufRead + Seek>(
    src: R,
    on_row: RowSink<'_>,
) -> Result<HeightmapProbe, TerrainError> {
    use tiff::decoder::DecodingResult;

    let (mut decoder, width, height, kind, geo) = tiff_header(src)?;
    if width == 0 || height == 0 {
        return Err(TerrainError::Empty);
    }
    let w = width as usize;
    let vscale = geo.vertical_scale();
    // Applying a scale of exactly 1.0 would still cost a multiply per sample on
    // the overwhelmingly common metric path, and — more importantly — would turn
    // an exact integer elevation into `x * 1.0`, which is the same value but
    // makes the "no conversion happened" case indistinguishable from the
    // "conversion happened to be identity" case in a debugger.
    let scaled = vscale != 1.0;

    let (chunk_w, chunk_h) = decoder.chunk_dimensions();
    if chunk_w == 0 || chunk_h == 0 {
        return Err(TerrainError::Image(
            "this TIFF declares a zero-sized chunk layout".into(),
        ));
    }
    let chunks_across = width.div_ceil(chunk_w);
    let chunks_down = height.div_ceil(chunk_h);
    let chunk_count = (chunks_across as u64) * (chunks_down as u64);

    // One row of source samples, reused. This plus one chunk row is the peak.
    let mut row = vec![0.0f64; w];
    // Which columns of the current row band have been written, so a short or
    // missing chunk cannot leave stale values in it (the C4-18 lesson, met in a
    // second decoder).
    let mut band: Vec<f64> = Vec::new();
    let mut band_covered: Vec<bool> = Vec::new();

    for cy in 0..chunks_down {
        let band_y0 = cy * chunk_h;
        let band_rows = chunk_h.min(height - band_y0) as usize;
        band.clear();
        band.resize(band_rows * w, 0.0);
        band_covered.clear();
        band_covered.resize(band_rows * w, false);

        for cx in 0..chunks_across {
            let index = cy * chunks_across + cx;
            if u64::from(index) >= chunk_count {
                break;
            }
            let (data_w, data_h) = decoder.chunk_data_dimensions(index);
            let result = decoder
                .read_chunk(index)
                .map_err(|e| tiff_err(&format!("chunk {index}"), &e))?;
            let dw = data_w as usize;
            let dh = data_h as usize;
            let x0 = (cx * chunk_w) as usize;

            // One closure per sample type, so the match is paid once per chunk
            // rather than once per sample.
            macro_rules! scatter {
                ($vals:expr, $conv:expr) => {{
                    let vals = $vals;
                    for j in 0..dh {
                        let gy = j;
                        if gy >= band_rows {
                            break;
                        }
                        for i in 0..dw {
                            let gx = x0 + i;
                            if gx >= w {
                                break;
                            }
                            let raw = match vals.get(j * dw + i) {
                                Some(v) => $conv(*v),
                                None => continue,
                            };
                            band[gy * w + gx] = if scaled { raw * vscale } else { raw };
                            band_covered[gy * w + gx] = true;
                        }
                    }
                }};
            }

            match result {
                DecodingResult::U8(v) => scatter!(v, |x: u8| x as f64),
                DecodingResult::U16(v) => scatter!(v, |x: u16| x as f64),
                DecodingResult::U32(v) => scatter!(v, |x: u32| x as f64),
                DecodingResult::U64(v) => scatter!(v, |x: u64| x as f64),
                DecodingResult::I8(v) => scatter!(v, |x: i8| x as f64),
                DecodingResult::I16(v) => scatter!(v, |x: i16| x as f64),
                DecodingResult::I32(v) => scatter!(v, |x: i32| x as f64),
                DecodingResult::I64(v) => scatter!(v, |x: i64| x as f64),
                DecodingResult::F16(v) => scatter!(v, |x: half::f16| x.to_f64()),
                DecodingResult::F32(v) => scatter!(v, |x: f32| x as f64),
                DecodingResult::F64(v) => scatter!(v, |x: f64| x),
            }
        }

        for j in 0..band_rows {
            let y = band_y0 + j as u32;
            if let Some(x) = band_covered[j * w..(j + 1) * w].iter().position(|c| !c) {
                return Err(TerrainError::Image(format!(
                    "the TIFF's chunks do not cover sample ({x}, {y}) — the file \
                     declares a {width}x{height} image but its strip/tile table \
                     leaves part of it unwritten"
                )));
            }
            row.copy_from_slice(&band[j * w..(j + 1) * w]);
            on_row(y, &row)?;
        }
    }

    Ok(HeightmapProbe {
        format: HeightmapFormat::Tiff,
        width,
        height,
        bit_depth: kind.bits,
        float_samples: kind.float,
        absolute_samples: kind.absolute,
        channel: "gray".into(),
        geo: Some(geo),
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
    src: R,
    on_row: RowSink<'_>,
) -> Result<HeightmapProbe, TerrainError> {
    let mut report = NodataReport::default();
    decode_rows_with(src, &NodataHandling::NONE, &mut report, on_row)
}

/// [`decode_rows`], with an explicit **nodata policy** (Wave G).
///
/// The policy runs on each decoded row **before** the finiteness door — see
/// [`NodataPolicy`] for why that ordering is the load-bearing part. `report`
/// accumulates what the policy actually did, which is what the import turns into
/// a cook advisory.
pub fn decode_rows_with<R: BufRead + Seek>(
    mut src: R,
    nodata: &NodataHandling,
    report: &mut NodataReport,
    on_row: RowSink<'_>,
) -> Result<HeightmapProbe, TerrainError> {
    let format = sniff(&mut src)?;
    // **The finiteness door** (round-2 finding B3). Every row crosses it, so
    // both tilers — the whole-image one at `:891` and the chunked one — are
    // covered by one check, and neither can disagree with the other about it.
    //
    // NaN is the *conventional* no-data value in a float height EXR, and this
    // decoder passes f16/f32 channel samples through verbatim. Downstream,
    // `HeightmapImport::map_sample` looks like a guard and is not: `Normalized`
    // does `v.clamp(0.0, 1.0)`, and Rust's `clamp` returns `self` when neither
    // comparison fires, which is exactly what a NaN does to both of them; and
    // `FloatMeters` — which the wizard offers whenever the source is float —
    // returns `v` untouched. From there it is `TerrainTile::set_sample` and a
    // committed `.inf_terrain`, then rapier heightfields, `terrain.height_at`,
    // the clipmap upload and erosion. `height_bounds` uses `f32::min`/`max`,
    // which ignore NaN, so the tile's own bounds look perfectly healthy —
    // the mechanism `crate::validate`'s module doc names.
    //
    // Refused rather than substituted: a no-data pixel means the author's
    // source does not cover that ground, and this engine has no representation
    // for a hole in a heightfield (holes are the voxel layer's, P21.2). Naming
    // the pixel is the only answer that lets them fix it in the exporter.
    //
    // **Wave G — the policy runs first.** Real DEMs are full of no-data, and
    // before Wave G the door above was the only behaviour, so the first real
    // GeoTIFF anyone imported stopped at its first ocean pixel. `apply_nodata`
    // substitutes according to an author-chosen, sidecar-recorded policy; the
    // finiteness check then runs on the result, so a non-finite sample that the
    // policy did NOT explain is still a bug and still refuses.
    let mut scratch: Vec<f64> = Vec::new();
    let mut finite_rows = |y: u32, row: &[f64]| -> Result<(), TerrainError> {
        let row: &[f64] = if nodata.is_active() {
            scratch.clear();
            scratch.extend_from_slice(row);
            apply_nodata(nodata, report, y, &mut scratch)?;
            &scratch
        } else {
            row
        };
        if let Some(x) = row.iter().position(|v| !v.is_finite()) {
            return Err(TerrainError::Image(format!(
                "sample at ({x}, {y}) is not a finite number (NaN or infinity) — \
                 in a float heightmap that usually means 'no data', and a terrain \
                 has no way to be absent at one sample. If this source is a real \
                 DEM with voids or ocean in it, choose a no-data policy for the \
                 import (clamp to an elevation, or fill small gaps) rather than \
                 editing the file."
            )));
        }
        on_row(y, row)
    };
    let probe = match format {
        HeightmapFormat::Png => decode_rows_png(src, &mut finite_rows)?,
        HeightmapFormat::Exr => decode_rows_exr(src, &mut finite_rows)?,
        HeightmapFormat::Tiff => decode_rows_tiff(src, &mut finite_rows)?,
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
        check_dimensions(info.width, info.height)?;
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
        absolute_samples: false,
        channel: "gray".into(),
        geo: None,
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
        absolute_samples: !matches!(sample_type, SampleType::U32),
        channel: channel_name,
        geo: None,
    })
}

/// Rows being filled in from EXR line fragments. A row leaves as soon as its
/// full width is covered, so a scanline file never holds more than one block's
/// worth and a tiled file never more than one tile-row's.
/// Which columns of one open row have actually been written (C4-18).
///
/// A bitmask plus its popcount, kept incrementally so `write` stays `O(values)`
/// rather than re-scanning the row.
struct RowCoverage {
    mask: Vec<u64>,
    covered: usize,
}

struct PartialRows {
    width: usize,
    open: std::collections::BTreeMap<usize, (Vec<f64>, RowCoverage)>,
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
    /// now **covered**.
    ///
    /// # Coverage, not a write count (C4-18)
    ///
    /// This used to accumulate `filled += written` and call the row done at
    /// `filled >= width`. An EXR with **overlapping** chunks — duplicate or
    /// re-sent line fragments, which `filter_chunks` does not filter (it selects
    /// by layer and level only) — drove that counter to `width` while some
    /// columns had never been touched at all. Those columns kept the pooled or
    /// zero-initialized value they were created with, and were persisted into
    /// the `.inf_terrain` as elevations.
    ///
    /// A per-column bitmask answers the question the row actually needs — *has
    /// every column been written* — rather than a proxy for it, and a second
    /// write to an already-covered column costs nothing but is not counted
    /// twice.
    fn write(&mut self, y: usize, x0: usize, values: impl Iterator<Item = f64>) -> bool {
        let width = self.width;
        let words = width.div_ceil(64);
        let fresh = self.pool.pop();
        let entry = self.open.entry(y).or_insert_with(|| {
            (
                fresh.unwrap_or_else(|| vec![0.0; width]),
                RowCoverage {
                    mask: vec![0u64; words],
                    covered: 0,
                },
            )
        });
        let (row, cov) = entry;
        row.resize(width, 0.0);
        cov.mask.resize(words, 0);
        for (i, v) in values.enumerate() {
            let x = x0 + i;
            if x < width {
                row[x] = v;
                let (w, b) = (x / 64, x % 64);
                if cov.mask[w] & (1u64 << b) == 0 {
                    cov.mask[w] |= 1u64 << b;
                    cov.covered += 1;
                }
            }
        }
        cov.covered >= width
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
            absolute_samples: false,
            channel: "gray".into(),
            geo: None,
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

    /// **C4-17 — a source that declares more samples than the lattice
    /// arithmetic can carry.**
    ///
    /// `(width as i32 - 1)` on a PNG legally declaring `i32::MAX` wrapped
    /// negative, `.max(0)` pulled it to zero, and `.max(1)` handed back **one
    /// tile** — a two-billion-sample source silently became a single tile, from
    /// nothing more than an 8-byte IHDR that `probe` reads without touching a
    /// pixel.
    ///
    /// Un-fix mutation: restore the `i32` expression in `HeightmapGrid::new` and
    /// the one-tile assertion below fails.
    #[test]
    fn an_absurd_source_size_never_collapses_to_a_single_tile() {
        let import = HeightmapImport::default();
        // Refused at the door, by name and with the dimensions in the message.
        let e = check_dimensions(i32::MAX as u32, 1024)
            .unwrap_err()
            .to_string();
        assert!(e.contains("2147483647"), "{e}");
        assert!(check_dimensions(0, 1024).is_err());
        assert!(check_dimensions(1024, 0).is_err());
        assert!(check_dimensions(MAX_HEIGHTMAP_SIDE, MAX_HEIGHTMAP_SIDE).is_ok());

        // And the arithmetic behind that door does not wrap either: the lattice
        // for a huge source is huge, not one tile.
        let cells = (import.resolution() - 1) as i64;
        for w in [u32::MAX, i32::MAX as u32, MAX_HEIGHTMAP_SIDE] {
            let g = HeightmapGrid::new(w, 1024, &import);
            let want = (((w as i64 - 1) + cells - 1) / cells).max(1);
            assert_eq!(
                g.ntx as i64, want,
                "a {w}-sample source produced {} tiles across",
                g.ntx
            );
            assert!(
                g.ntx > 1,
                "a {w}-sample source collapsed to {} tiles",
                g.ntx
            );
        }
        // Ordinary sizes are untouched.
        let g = HeightmapGrid::new(2049, 1025, &import);
        assert_eq!(g.ntx, HeightmapGrid::new(2049, 1025, &import).ntx);
        assert!(g.ntx >= 1 && g.ntz >= 1);
    }

    // ── Wave G: GeoTIFF ─────────────────────────────────────────────────────

    use crate::geotiff::VerticalUnits;
    use tiff::encoder::colortype;
    use tiff::encoder::TiffEncoder;
    use tiff::tags::Tag;

    /// How a test fixture is georeferenced.
    #[derive(Default, Clone)]
    struct GeoTags {
        pixel_scale: Option<[f64; 3]>,
        tiepoint: Option<[f64; 6]>,
        transformation: Option<[f64; 16]>,
        /// `(key id, value)` pairs written with `location == 0`.
        keys: Vec<(u16, u16)>,
        nodata: Option<String>,
        rows_per_strip: Option<u32>,
    }

    /// Build a real f32 GeoTIFF in memory.
    fn geotiff_f32(width: u32, height: u32, samples: &[f32], geo: &GeoTags) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut enc = TiffEncoder::new(Cursor::new(&mut buf)).expect("tiff encoder");
            let mut img = enc
                .new_image::<colortype::Gray32Float>(width, height)
                .expect("new image");
            if let Some(n) = geo.rows_per_strip {
                img.rows_per_strip(n).expect("rows per strip");
            }
            write_geo_tags(img.encoder(), geo);
            img.write_data(samples).expect("write data");
        }
        buf
    }

    /// Build a real i16 GeoTIFF — the shape SRTM and most national DEMs ship in.
    fn geotiff_i16(width: u32, height: u32, samples: &[i16], geo: &GeoTags) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut enc = TiffEncoder::new(Cursor::new(&mut buf)).expect("tiff encoder");
            let mut img = enc
                .new_image::<colortype::GrayI16>(width, height)
                .expect("new image");
            if let Some(n) = geo.rows_per_strip {
                img.rows_per_strip(n).expect("rows per strip");
            }
            write_geo_tags(img.encoder(), geo);
            img.write_data(samples).expect("write data");
        }
        buf
    }

    fn write_geo_tags<W: std::io::Write + Seek, K: tiff::encoder::TiffKind>(
        dir: &mut tiff::encoder::DirectoryEncoder<'_, W, K>,
        geo: &GeoTags,
    ) {
        if let Some(s) = geo.pixel_scale {
            dir.write_tag(Tag::Unknown(33550), &s[..]).expect("scale");
        }
        if let Some(t) = geo.tiepoint {
            dir.write_tag(Tag::Unknown(33922), &t[..]).expect("tie");
        }
        if let Some(m) = geo.transformation {
            dir.write_tag(Tag::Unknown(34264), &m[..]).expect("xform");
        }
        if !geo.keys.is_empty() {
            let mut d: Vec<u16> = vec![1, 1, 0, geo.keys.len() as u16];
            for (id, value) in &geo.keys {
                d.extend_from_slice(&[*id, 0, 1, *value]);
            }
            dir.write_tag(Tag::Unknown(34735), &d[..]).expect("geokeys");
        }
        if let Some(n) = &geo.nodata {
            dir.write_tag(Tag::Unknown(42113), n.as_str())
                .expect("nodata");
        }
    }

    fn utm10n() -> Vec<(u16, u16)> {
        vec![(1024, 1), (3072, 32610), (3076, 9001)]
    }

    fn read_all(bytes: &[u8]) -> (HeightmapProbe, Vec<Vec<f64>>) {
        let mut rows = Vec::new();
        let probe = decode_rows(Cursor::new(bytes), &mut |y, row| {
            if rows.len() <= y as usize {
                rows.resize(y as usize + 1, Vec::new());
            }
            rows[y as usize] = row.to_vec();
            Ok(())
        })
        .expect("decode");
        (probe, rows)
    }

    /// The whole Batch G-A door: a georeferenced float DEM probes, reports its
    /// geotransform, and decodes to its own elevations.
    #[test]
    fn a_georeferenced_float_geotiff_probes_and_decodes() {
        let (w, h) = (8u32, 5u32);
        let samples: Vec<f32> = (0..w * h).map(|i| i as f32 * 3.25 - 20.0).collect();
        let geo = GeoTags {
            pixel_scale: Some([10.0, 10.0, 0.0]),
            tiepoint: Some([0.0, 0.0, 0.0, 491_000.0, 5_459_000.0, 0.0]),
            keys: utm10n(),
            ..Default::default()
        };
        let bytes = geotiff_f32(w, h, &samples, &geo);

        // The magic is sniffed as TIFF.
        let probe = probe_heightmap_bytes(&bytes).expect("probe");
        assert_eq!(probe.format, HeightmapFormat::Tiff);
        assert_eq!((probe.width, probe.height), (w, h));
        assert_eq!(probe.bit_depth, 32);
        assert!(probe.float_samples);
        assert!(
            probe.absolute_samples,
            "a float DEM carries absolute elevations"
        );

        // …and the georeferencing came back.
        let g = probe.geo.as_ref().expect("georeferenced");
        assert!(g.is_georeferenced());
        assert_eq!(g.meters_per_sample, Some(10.0));
        assert_eq!(g.origin, Some((491_000.0, 5_459_000.0, 0.0)));
        assert_eq!(g.epsg, Some(32610));
        assert!(!g.crs_is_geographic);
        assert_eq!(g.vertical_units, VerticalUnits::Metre);
        assert_eq!(g.vertical_scale(), 1.0);

        // Every sample decodes to its own value, in order.
        let (probe2, rows) = read_all(&bytes);
        assert_eq!(probe2.width, probe.width);
        assert_eq!(rows.len(), h as usize);
        for (y, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), w as usize, "row {y}");
            for (x, v) in row.iter().enumerate() {
                let want = samples[y * w as usize + x] as f64;
                assert_eq!(*v, want, "sample ({x}, {y})");
            }
        }

        // And it lands on a real terrain through the whole-image tiler.
        let import = HeightmapImport {
            tile_resolution: 5,
            meters_per_sample: 10.0,
            mode: HeightMode::FloatMeters,
            ..Default::default()
        };
        let t = TerrainData::from_height_image(&bytes, import).expect("import");
        assert!(t.tile_count() >= 1);
        assert_eq!(t.get_tile((0, 0)).unwrap().sample(5, 0, 0), -20.0);
    }

    /// **A 16-bit INTEGER GeoTIFF is a DEM.** This is the case the old
    /// `float_samples` gate would have refused, and refusing it would have shut
    /// out SRTM and most national elevation products.
    ///
    /// Un-fix mutation: point `validate` back at `float_samples` and the
    /// float-metres import below fails.
    #[test]
    fn an_integer_geotiff_dem_carries_absolute_elevations() {
        let (w, h) = (6u32, 4u32);
        // Real elevations, including below sea level.
        let samples: Vec<i16> = (0..w * h).map(|i| i as i16 * 40 - 100).collect();
        let geo = GeoTags {
            pixel_scale: Some([30.0, 30.0, 0.0]),
            tiepoint: Some([0.0, 0.0, 0.0, 500_000.0, 4_000_000.0, 0.0]),
            keys: utm10n(),
            ..Default::default()
        };
        let bytes = geotiff_i16(w, h, &samples, &geo);

        let probe = probe_heightmap_bytes(&bytes).unwrap();
        assert_eq!(probe.bit_depth, 16);
        assert!(!probe.float_samples, "an i16 raster is not float");
        assert!(
            probe.absolute_samples,
            "…but a SIGNED integer raster IS elevations — that is the distinction \
             `absolute_samples` exists to draw"
        );

        // Float-metres mode is therefore available, and is validated as such.
        let import = HeightmapImport {
            tile_resolution: 4,
            meters_per_sample: 30.0,
            mode: HeightMode::FloatMeters,
            ..Default::default()
        };
        assert!(import.validate(Some(&probe)).is_ok());

        // The negative elevations survive as negative metres.
        let (_, rows) = read_all(&bytes);
        assert_eq!(rows[0][0], -100.0);
        assert_eq!(rows[0][1], -60.0);

        // A PNG, by contrast, is still refused in that mode — the old behaviour
        // is intact for the format it was written for.
        let png = encode_png16(&ramp(4, 4)).unwrap();
        let png_probe = probe_heightmap_bytes(&png).unwrap();
        assert!(!png_probe.absolute_samples);
        assert!(import.validate(Some(&png_probe)).is_err());
    }

    /// **Feet become metres exactly once**, at the decoder, and a landscape in
    /// feet does not import 3.28x too flat.
    ///
    /// Un-fix mutation: drop the `vscale` multiply in `decode_rows_tiff` and the
    /// 1000 ft summit comes back as 1000 m.
    #[test]
    fn a_dem_in_feet_imports_in_metres() {
        let (w, h) = (4u32, 2u32);
        // A 1000-foot summit — 304.8 m.
        let samples: Vec<f32> = vec![0.0, 1000.0, 500.0, -100.0, 0.0, 0.0, 0.0, 0.0];
        let mut keys = utm10n();
        keys.push((4099, 9002)); // vertical unit: international foot
        let geo = GeoTags {
            pixel_scale: Some([10.0, 10.0, 0.0]),
            tiepoint: Some([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            keys,
            ..Default::default()
        };
        let bytes = geotiff_f32(w, h, &samples, &geo);

        let probe = probe_heightmap_bytes(&bytes).unwrap();
        assert_eq!(
            probe.geo.as_ref().unwrap().vertical_units,
            VerticalUnits::Foot
        );

        let (_, rows) = read_all(&bytes);
        assert!(
            (rows[0][1] - 304.8).abs() < 1e-6,
            "1000 ft must import as 304.8 m, got {}",
            rows[0][1]
        );
        assert!((rows[0][2] - 152.4).abs() < 1e-6);
        assert!((rows[0][3] + 30.48).abs() < 1e-6, "negatives scale too");
        assert_eq!(rows[0][0], 0.0, "sea level is sea level in any unit");

        // The US survey foot is a DIFFERENT unit and is not silently the same.
        let mut keys = utm10n();
        keys.push((4099, 9003));
        let us = geotiff_f32(
            w,
            h,
            &samples,
            &GeoTags {
                keys,
                nodata: None,
                transformation: None,
                ..geo.clone()
            },
        );
        let (_, us_rows) = read_all(&us);
        assert!(
            us_rows[0][1] > 304.8,
            "the US survey foot is the longer one"
        );
        assert!(
            (us_rows[0][1] - 304.800_609_6).abs() < 1e-5,
            "got {}",
            us_rows[0][1]
        );

        // A horizontal foot unit with NO vertical key implies foot elevations —
        // the inference that keeps a foot-based state plane from importing flat.
        let flat_keys = vec![(1024, 1), (3072, 32610), (3076, 9002)];
        let inferred = geotiff_f32(
            w,
            h,
            &samples,
            &GeoTags {
                keys: flat_keys,
                ..geo.clone()
            },
        );
        let p = probe_heightmap_bytes(&inferred).unwrap();
        assert_eq!(
            p.geo.as_ref().unwrap().vertical_units,
            VerticalUnits::Foot,
            "a foot-based CRS with no vertical key must imply foot elevations"
        );
    }

    /// **The nodata policy door.** The default still refuses (naming the fix),
    /// and each policy does what it says.
    #[test]
    fn the_nodata_policy_decides_what_happens_to_a_void() {
        let (w, h) = (6u32, 2u32);
        // A row with a two-sample void in the middle, declared as -9999.
        let samples: Vec<f32> = vec![
            10.0, 20.0, -9999.0, -9999.0, 50.0, 60.0, //
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ];
        let geo = GeoTags {
            pixel_scale: Some([10.0, 10.0, 0.0]),
            tiepoint: Some([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            keys: utm10n(),
            nodata: Some("-9999".into()),
            ..Default::default()
        };
        let bytes = geotiff_f32(w, h, &samples, &geo);

        // The sentinel is read off the ASCII tag.
        let probe = probe_heightmap_bytes(&bytes).unwrap();
        assert_eq!(probe.geo.as_ref().unwrap().nodata, Some(-9999.0));

        let handling = |policy| NodataHandling {
            policy,
            sentinel: Some(-9999.0),
        };
        let run = |h: NodataHandling| -> Result<(Vec<Vec<f64>>, NodataReport), TerrainError> {
            let mut rows: Vec<Vec<f64>> = Vec::new();
            let mut report = NodataReport::default();
            decode_rows_with(Cursor::new(&bytes), &h, &mut report, &mut |y, row| {
                if rows.len() <= y as usize {
                    rows.resize(y as usize + 1, Vec::new());
                }
                rows[y as usize] = row.to_vec();
                Ok(())
            })?;
            Ok((rows, report))
        };

        // Refuse: the default, and it survives — a -9999 is finite, so this one
        // imports as a -9999 m pit rather than erroring. That is the honest
        // behaviour for an UNDECLARED sentinel, and it is why the policy exists.
        let (rows, report) = run(handling(NodataPolicy::Refuse)).unwrap();
        assert_eq!(rows[0][2], -9999.0);
        assert!(!report.engaged(), "Refuse substitutes nothing");

        // Clamp: the void becomes the stated elevation.
        let (rows, report) = run(handling(NodataPolicy::sea_level())).unwrap();
        assert_eq!(rows[0][2], 0.0);
        assert_eq!(rows[0][3], 0.0);
        assert_eq!(rows[0][1], 20.0, "valid samples are untouched");
        assert_eq!(report.substituted, 2);
        assert!(
            report.engaged(),
            "and the cook advisory has something to say"
        );

        // Fill: the void is interpolated between its neighbours.
        let (rows, report) = run(handling(NodataPolicy::FillRow { max_span: 4 })).unwrap();
        assert!(
            (rows[0][2] - 30.0).abs() < 1e-9 && (rows[0][3] - 40.0).abs() < 1e-9,
            "a 2-wide void between 20 and 50 fills to 30 and 40, got {:?}",
            &rows[0]
        );
        assert_eq!(report.substituted, 2);
        assert_eq!(report.filled_runs, 1);
        assert_eq!(report.widest_run, 2);

        // …and a void wider than the limit REFUSES rather than smearing.
        let e = run(handling(NodataPolicy::FillRow { max_span: 1 }))
            .unwrap_err()
            .to_string();
        assert!(e.contains("run of 2"), "{e}");
        assert!(
            e.contains("clamp"),
            "the refusal must name the alternative: {e}"
        );
    }

    /// NaN is no-data whether or not a sentinel is declared — and the refusal
    /// now tells the author about the policy instead of just naming the pixel.
    #[test]
    fn a_nan_void_is_handled_by_the_policy_and_named_without_one() {
        let (w, h) = (4u32, 1u32);
        let samples: Vec<f32> = vec![10.0, f32::NAN, f32::NAN, 40.0];
        let geo = GeoTags {
            pixel_scale: Some([1.0, 1.0, 0.0]),
            tiepoint: Some([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            keys: utm10n(),
            ..Default::default()
        };
        let bytes = geotiff_f32(w, h, &samples, &geo);

        // With no policy the finiteness door fires, and now points at the fix.
        let e = decode_rows(Cursor::new(&bytes), &mut |_, _| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(e.contains("not a finite number"), "{e}");
        assert!(
            e.contains("no-data policy"),
            "the refusal must now name the policy as the remedy: {e}"
        );

        // With a clamp, the NaNs become sea level — no sentinel needed, because
        // NaN is always no-data.
        let mut rows: Vec<Vec<f64>> = Vec::new();
        let mut report = NodataReport::default();
        decode_rows_with(
            Cursor::new(&bytes),
            &NodataHandling {
                policy: NodataPolicy::sea_level(),
                sentinel: None,
            },
            &mut report,
            &mut |_, row| {
                rows.push(row.to_vec());
                Ok(())
            },
        )
        .expect("a clamp policy handles NaN with no sentinel declared");
        assert_eq!(rows[0], vec![10.0, 0.0, 0.0, 40.0]);
        assert_eq!(report.substituted, 2);
    }

    /// The two rasters that are refused rather than distorted, and the two
    /// compression/format cases that get a remedy rather than a shrug.
    #[test]
    fn distorting_rasters_are_refused_with_a_named_remedy() {
        let samples: Vec<f32> = vec![0.0; 16];

        // Non-square pixels would stretch the world along one axis.
        let aniso = geotiff_f32(
            4,
            4,
            &samples,
            &GeoTags {
                pixel_scale: Some([10.0, 25.0, 0.0]),
                tiepoint: Some([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                keys: utm10n(),
                ..Default::default()
            },
        );
        let e = probe_heightmap_bytes(&aniso).unwrap_err().to_string();
        assert!(e.contains("not square"), "{e}");
        assert!(e.contains("10") && e.contains("25"), "{e}");
        assert!(e.contains("gdalwarp"), "{e}");

        // A rotated transformation cannot be honoured without resampling.
        let mut m = [0.0f64; 16];
        m[0] = 10.0;
        m[1] = 2.5; // shear
        m[5] = -10.0;
        let rotated = geotiff_f32(
            4,
            4,
            &samples,
            &GeoTags {
                transformation: Some(m),
                keys: utm10n(),
                ..Default::default()
            },
        );
        let e = probe_heightmap_bytes(&rotated).unwrap_err().to_string();
        assert!(e.contains("rotated") || e.contains("sheared"), "{e}");
        assert!(e.contains("north-up"), "{e}");

        // An unknown vertical unit is refused rather than assumed metric.
        let mut keys = utm10n();
        keys.push((4099, 9036)); // kilometres
        let km = geotiff_f32(
            4,
            4,
            &samples,
            &GeoTags {
                pixel_scale: Some([10.0, 10.0, 0.0]),
                tiepoint: Some([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                keys,
                ..Default::default()
            },
        );
        let e = probe_heightmap_bytes(&km).unwrap_err().to_string();
        assert!(
            e.contains("9036"),
            "the refusal must name the unit code: {e}"
        );
        assert!(e.contains("vertical unit"), "{e}");

        // A BigTIFF is refused AS a BigTIFF, with the cutting remedy — not as
        // "unrecognized file signature".
        let mut big = vec![b'I', b'I', 43, 0];
        big.extend_from_slice(&[0u8; 32]);
        let e = probe_heightmap_bytes(&big).unwrap_err().to_string();
        assert!(e.contains("BigTIFF"), "{e}");
        assert!(e.contains("gdal_translate"), "{e}");
    }

    /// **The memory bound survives the new format.** A striped TIFF is read one
    /// chunk row at a time, and the decoder never holds the image.
    ///
    /// The assertion is on the DECODER's own chunk layout rather than on a
    /// process memory figure: what matters is that the file really is being read
    /// in many chunks and that each row arrives exactly once, in order.
    #[test]
    fn a_striped_geotiff_streams_one_chunk_row_at_a_time() {
        let (w, h) = (64u32, 64u32);
        let samples: Vec<f32> = (0..w * h).map(|i| (i % 997) as f32).collect();
        let bytes = geotiff_f32(
            w,
            h,
            &samples,
            &GeoTags {
                pixel_scale: Some([1.0, 1.0, 0.0]),
                tiepoint: Some([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                keys: utm10n(),
                // Four rows per strip: sixteen strips, so the chunked path is
                // genuinely exercised rather than degenerating to one chunk.
                rows_per_strip: Some(4),
                ..Default::default()
            },
        );

        let mut seen: Vec<u32> = Vec::new();
        let mut peak_row_len = 0usize;
        let probe = decode_rows(Cursor::new(&bytes), &mut |y, row| {
            seen.push(y);
            peak_row_len = peak_row_len.max(row.len());
            // Every sample is its own value — no chunk boundary smeared.
            for (x, v) in row.iter().enumerate() {
                assert_eq!(
                    *v,
                    samples[(y as usize) * w as usize + x] as f64,
                    "sample ({x}, {y}) crossed a chunk boundary wrong"
                );
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(probe.width, w);
        assert_eq!(seen.len(), h as usize, "every row arrives exactly once");
        assert_eq!(
            seen,
            (0..h).collect::<Vec<_>>(),
            "rows must arrive in order, once each"
        );
        assert_eq!(peak_row_len, w as usize, "a row is a row, never the image");
        // (The whole-image-vs-chunked BYTE identity for GeoTIFF is asserted by
        // the real determinism gate in `crate::chunked` — extended to cover this
        // format rather than re-implemented here.)
    }

    /// A plain (ungeoreferenced) TIFF is still a heightmap — importing one is
    /// legitimate and must not require a geotransform.
    #[test]
    fn a_plain_tiff_is_an_ordinary_heightmap() {
        let (w, h) = (4u32, 3u32);
        let samples: Vec<f32> = (0..w * h).map(|i| i as f32 / 100.0).collect();
        let bytes = geotiff_f32(w, h, &samples, &GeoTags::default());
        let probe = probe_heightmap_bytes(&bytes).unwrap();
        assert_eq!(probe.format, HeightmapFormat::Tiff);
        let g = probe.geo.as_ref().unwrap();
        assert!(!g.is_georeferenced(), "no tags, no georeference");
        assert_eq!(g.epsg, None);
        assert_eq!(g.meters_per_sample, None);
        // It still decodes. (Compared with a tolerance, not exactly: the source
        // samples are f32, so `1.0/100.0` reaches us as the nearest f32 widened
        // to f64 — 0.009 999 999 776…, which is the right answer and not 0.01.)
        let (_, rows) = read_all(&bytes);
        assert!((rows[0][1] - 0.01).abs() < 1e-7, "got {}", rows[0][1]);

        // Multi-sample TIFFs are refused by name — a heightmap is scalar.
        let mut buf = Vec::new();
        {
            let mut enc = TiffEncoder::new(Cursor::new(&mut buf)).unwrap();
            enc.write_image::<colortype::RGB8>(2, 2, &[0u8; 12])
                .unwrap();
        }
        let e = probe_heightmap_bytes(&buf).unwrap_err().to_string();
        assert!(e.contains("one sample per pixel"), "{e}");
        assert!(e.contains("gdal_translate"), "{e}");
    }

    /// The tiepoint's raster offset is honoured rather than assumed to be zero.
    #[test]
    fn a_tiepoint_at_the_raster_centre_places_the_corner_correctly() {
        let (w, h) = (11u32, 11u32);
        let samples: Vec<f32> = vec![0.0; (w * h) as usize];
        // Tie raster pixel (5, 5) to model (1000, 2000) at 10 m pixels. The
        // top-left sample is therefore 50 m west and 50 m north of that.
        let bytes = geotiff_f32(
            w,
            h,
            &samples,
            &GeoTags {
                pixel_scale: Some([10.0, 10.0, 0.0]),
                tiepoint: Some([5.0, 5.0, 0.0, 1000.0, 2000.0, 0.0]),
                keys: utm10n(),
                ..Default::default()
            },
        );
        let g = probe_heightmap_bytes(&bytes).unwrap().geo.unwrap();
        assert_eq!(
            g.origin,
            Some((950.0, 2050.0, 0.0)),
            "the tie's raster offset must be walked back to the top-left sample"
        );
    }

    /// A geographic GeoTIFF reports its pixel size in DEGREES, so nobody reads
    /// 0.000278 as a quarter of a millimetre.
    #[test]
    fn a_geographic_geotiff_says_its_pixels_are_degrees() {
        let (w, h) = (4u32, 4u32);
        let samples: Vec<f32> = vec![0.0; 16];
        let bytes = geotiff_f32(
            w,
            h,
            &samples,
            &GeoTags {
                pixel_scale: Some([0.000_277_8, 0.000_277_8, 0.0]),
                tiepoint: Some([0.0, 0.0, 0.0, -123.5, 49.5, 0.0]),
                // model type 2 == geographic; EPSG:4326.
                keys: vec![(1024, 2), (2048, 4326)],
                ..Default::default()
            },
        );
        let g = probe_heightmap_bytes(&bytes).unwrap().geo.unwrap();
        assert!(g.crs_is_geographic);
        assert_eq!(g.epsg, Some(4326));
        assert_eq!(g.degrees_per_sample(), Some(0.000_277_8));
    }

    /// The nodata policy label round-trips, and an unrecognised one is refused
    /// rather than silently downgraded to `Refuse` — which would change the
    /// terrain a re-import produces.
    #[test]
    fn nodata_policy_labels_round_trip_and_refuse_the_unknown() {
        for p in [
            NodataPolicy::Refuse,
            NodataPolicy::Clamp(0.0),
            NodataPolicy::Clamp(-12.5),
            NodataPolicy::FillRow { max_span: 64 },
        ] {
            let label = p.label();
            assert_eq!(
                NodataPolicy::from_label(&label),
                Some(p),
                "{label:?} did not round-trip"
            );
        }
        assert_eq!(NodataPolicy::sea_level(), NodataPolicy::Clamp(0.0));
        assert_eq!(NodataPolicy::default(), NodataPolicy::Refuse);
        assert_eq!(NodataPolicy::from_label("smear-outward"), None);
        assert_eq!(NodataPolicy::from_label("clamp:nan"), None);
        assert_eq!(NodataPolicy::from_label("clamp:"), None);
    }

    /// The sentinel match is RELATIVE, because a float sentinel reaches us twice
    /// by two routes that need not round to the same double.
    #[test]
    fn the_nodata_sentinel_matches_across_a_float_round_trip() {
        let h = NodataHandling {
            policy: NodataPolicy::sea_level(),
            sentinel: Some(-3.402_823_466_385_288_6e38),
        };
        // The f32 minimum widened to f64 — what the pixels actually carry.
        assert!(h.is_nodata(f32::MIN as f64), "the f32 sentinel must match");
        assert!(h.is_nodata(-3.402_823_466_385_288_6e38));
        // A real elevation does not.
        assert!(!h.is_nodata(-9999.0));
        assert!(!h.is_nodata(0.0));
        // Non-finite is always no-data, sentinel or not.
        assert!(h.is_nodata(f64::NAN));
        assert!(NodataHandling::NONE.is_nodata(f64::NAN));
        assert!(!NodataHandling::NONE.is_nodata(-9999.0));
        // An exact integer sentinel still matches exactly.
        let h = NodataHandling {
            policy: NodataPolicy::sea_level(),
            sentinel: Some(-9999.0),
        };
        assert!(h.is_nodata(-9999.0));
        assert!(!h.is_nodata(-9998.0));
    }

    /// **C4-18 — `PartialRows` counted writes, not coverage.**
    ///
    /// An EXR with overlapping chunks (duplicate or re-sent line fragments, which
    /// `filter_chunks` does not filter — it selects by layer and level only)
    /// drove the old `filled += written` counter to `width` while some columns
    /// had never been touched. Those columns kept the pooled or zero-initialized
    /// value and were persisted into the `.inf_terrain` as elevations.
    ///
    /// Un-fix mutation: restore the counter and the first assertion fails —
    /// eight overlapping writes of the same four columns declare a 16-wide row
    /// complete.
    #[test]
    fn overlapping_row_fragments_never_declare_a_row_covered() {
        let mut rows = PartialRows::new(16);
        // Eight writes of columns 0..4 — 32 values into a 16-wide row.
        for _ in 0..8 {
            assert!(
                !rows.write(0, 0, [1.0, 2.0, 3.0, 4.0].into_iter()),
                "the row was declared covered by re-writing four columns"
            );
        }
        // Covering the rest completes it, exactly once.
        assert!(rows.write(0, 4, (4..16).map(|i| i as f64)));
        let row = rows.take(0).unwrap();
        assert_eq!(row.len(), 16);
        assert_eq!(row[0], 1.0);
        assert_eq!(row[15], 15.0);

        // A row written straight through in one go still completes on the last
        // value, and not before it.
        let mut rows = PartialRows::new(16);
        for x in 0..15 {
            assert!(!rows.write(1, x, std::iter::once(x as f64)));
        }
        assert!(rows.write(1, 15, std::iter::once(15.0)));
    }
}
