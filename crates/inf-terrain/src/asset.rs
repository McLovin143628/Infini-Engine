//! The `.inf_terrain` **streaming asset** (P16.3): terrain leaves the `.inf_lvl`
//! blob for a random-access, page-at-a-time container.
//!
//! ```text
//! ┌ header (v2/v3/v4: 128 B; v1: 64 B — all little-endian) ───────────┐
//! │  magic         [u8; 8]   b"INFTERRN"                              │
//! │  schema_ver    u32       TERRAIN_ASSET_SCHEMA_VERSION             │
//! │  tile_res      u32       samples per tile side (every level)      │
//! │  mps           f64       level-0 metres per sample                │
//! │  origin        [f64; 3]  world anchor of tile (0,0) sample (0,0)  │
//! │  lod_levels    u32       levels present (1 = level 0 only)        │
//! │  tile_count    u32       directory entries                        │
//! │  blob_base     u64       absolute offset of the blob section      │
//! │ ── v2+ only, offset 64 ───────────────────────────────────────── │
//! │  pyr_max_lvls  u32       PyramidOptions::max_levels               │
//! │  pyr_min_tiles u32       PyramidOptions::min_tiles                │
//! │  reserved      [u8; 56]  zeros (room for v5 without a re-length)  │
//! ├ tile directory (tile_count × 32 B, sorted by TileKey) ────────────┤
//! │  lod u32 · tx i32 · tz i32 · reserved u32 · offset u64 · len u64  │
//! ├ blob section (each tile's bincode bytes, 16-byte-aligned) ────────┤
//! │  … tile blobs, zero-padded up to each 16-byte boundary …          │
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Schema v3 / v4: tile blobs grow layers (P19.1, P19.2)
//!
//! The header has not changed since v2 — v3 and v4 are byte-for-byte v2's
//! 128-byte layout — but the *tile blob* has, twice: v3 appended the sparse
//! erosion [`maps`](TerrainTile::maps) layer (flow / deposition / wear, P19.1) and
//! v4 appended the sparse [`biomes`](TerrainTile::biomes) layer (per-sample biome
//! ids, P19.2). bincode is **positional**, so each is a wire-format change even
//! though the new field is `#[serde(default)]`: a v2 blob has three fields where
//! this build reads five, and decoding it as the grown tile would run off the end
//! of the blob.
//!
//! The version is therefore what selects the wire type ([`decode_tile_at`]): a
//! payload stamped **≤ 2** decodes through the frozen [`TerrainTileFrozenV1`], a
//! **v3** one through [`TerrainTileFrozenV2`], and a **v4** one as the current
//! tile — each lifting the layers it cannot carry to their sparse defaults, which
//! is exactly what a payload of that vintage meant. So **v1..v3 payloads load
//! forever**, and the only thing a rewrite costs an untouched terrain is one
//! zero-length count per layer per tile blob. The header length still comes from
//! [`header_len`] alone, which is why v3 and v4 could reuse v2's without
//! ambiguity: the *schema version* remains the single source of truth for how to
//! read everything after the magic.
//!
//! # Schema v2: the pyramid options are recorded (P16.6)
//!
//! v1's header stopped at `blob_base`, so the options a terrain's LOD pyramid was
//! *built* with were nowhere on disk. The write-back path
//! ([`crate::writeback::rewrite_terrain_asset`]) has to re-plan the pyramid to
//! fold a sculpt back in, and with nothing recorded it could only re-plan with the
//! **defaults** — silently reshaping an asset imported with non-default wizard
//! knobs on its first save. (Inferring the options back out was considered and
//! rejected: the two stop conditions are indistinguishable after the fact, and
//! every inference rule that preserves a capped asset's depth also refuses to
//! deepen a terrain that genuinely grew.)
//!
//! v2 writes them into 8 bytes of header, and the read side reports
//! [`TerrainAssetHeader::pyramid`] as `Some` for a v2 asset and **`None` for a
//! v1 one** — *unknown*, not "the defaults". That distinction is the whole value:
//! a rewrite of a v1 asset still falls back to the defaults (there is nothing
//! better to do), but it can now say so, and a rewrite of a v2 asset cannot
//! reshape anything. The forward lift is the established asset-level pattern: the
//! header length is a function of the schema version, so a v1 payload keeps
//! loading, byte for byte, forever.
//!
//! # Why this shape
//!
//! * **Random access without decoding the terrain.** The directory is a flat,
//!   sorted array — a consumer binary-searches a `(coord, lod)` and gets an
//!   `(offset, len)` into the payload. A cooked-pack consumer takes
//!   [`PackReader::read_ref`](inf_asset::PackReader::read_ref) (the entry is
//!   **uncompressed** by [`PackWriter::compresses_kind`](inf_asset::PackWriter::compresses_kind),
//!   so it is a borrowed slice of the mapping) and sub-slices it; a loose-file
//!   consumer seeks to the very same offsets. One layout, both paths.
//! * **16-byte-aligned tiles.** Blob offsets are multiples of
//!   [`TILE_ALIGN`] — the same constant, and the same reasoning, as
//!   [`inf_asset::BLOB_ALIGN`]: a pack v2 blob starts 16-byte aligned *and* every
//!   pack backing is 16-byte aligned at its base, so a tile slice sub-sliced out
//!   of one is aligned by address too, and can go straight to a GPU upload or a
//!   `[f32]` view. (Alignment inside the payload is unconditional; address
//!   alignment, as always, is the conjunction with the backing's.)
//! * **Byte-deterministic.** The directory is emitted in `BTreeMap` (`TileKey`)
//!   order, padding is deterministic zeros, and each tile blob is the *same*
//!   `bincode` a tile writes into an `.inf_lvl` — so two builds of one terrain are
//!   byte-identical and the two persistence paths agree.
//!
//! # There is exactly one writer: [`write_terrain_asset`]
//!
//! The bytes on disk — and in a `.inf_pack` — are the **raw image**
//! ([`TerrainAsset::as_bytes`]). They are *not* `inf_asset::encode` output: a
//! bincode length prefix would shift every tile off its 16-byte boundary and
//! defeat the entire point of the layout, silently, in a way only a GPU upload on
//! some other machine would notice.
//!
//! So the generic door is **closed at compile time**: [`TerrainAsset`]
//! deliberately implements neither [`AssetPayload`](inf_asset::AssetPayload) nor
//! `Serialize`/`Deserialize`, which makes `inf_asset::encode`,
//! `Project::write_asset` and `Project::rewrite_payload` reject it with a type
//! error rather than write a subtly wrong file. The kind and schema version it
//! still owes the database live as inherent consts ([`TerrainAsset::KIND`],
//! [`TerrainAsset::SCHEMA_VERSION`]).
//!
//! **P16.4 note for the terrain import wizard**: emit assets through
//! [`write_terrain_asset`], never a generic asset-writing helper. It writes the
//! image atomically (temp + rename, like `PackWriter::write_to_file`) and stamps
//! the sidecar's content hash over the same bytes the cook will pack.

use std::collections::BTreeMap;

use glam::DVec3;
use inf_asset::AssetKind;

use crate::data::TerrainData;
use crate::pyramid::{PyramidLevel, PyramidOptions};
use crate::tile::{TerrainTile, TerrainTileFrozenV1, TerrainTileFrozenV2, TileKey};

/// Magic at the head of every `.inf_terrain` payload.
pub const TERRAIN_ASSET_MAGIC: [u8; 8] = *b"INFTERRN";

/// Current `.inf_terrain` payload schema version (**4** since P19.2 — tile blobs
/// carry the per-sample biome ids; **3** since P19.1 — the erosion data maps;
/// **2** since P16.6 — the header records the pyramid options; see the module
/// docs).
pub const TERRAIN_ASSET_SCHEMA_VERSION: u32 = 4;

/// Tile blobs start on multiples of this many bytes (see the module docs).
pub const TILE_ALIGN: u64 = 16;

/// Bytes of the **v1** fixed header (no pyramid options).
pub const HEADER_LEN_V1: u64 = 64;

/// Bytes of the **v2** fixed header: v1's fields plus the pyramid options and 56
/// reserved zero bytes. A multiple of [`TILE_ALIGN`], like v1, so the directory
/// and every blob stay aligned without padding.
pub const HEADER_LEN_V2: u64 = 128;

/// Bytes of the **v3** fixed header. P19.1 changed the *tile blob* layout, not
/// the header, so this is deliberately v2's length — stated as its own constant
/// so the version→length table below reads as a table rather than as a fall-through.
pub const HEADER_LEN_V3: u64 = HEADER_LEN_V2;

/// Bytes of the **v4** fixed header. P19.2, like P19.1, changed the *tile blob*
/// layout and not the header — same reasoning as [`HEADER_LEN_V3`].
pub const HEADER_LEN_V4: u64 = HEADER_LEN_V2;

/// Bytes of the fixed header **this build writes** (the current schema's).
pub const HEADER_LEN: u64 = HEADER_LEN_V4;

/// Bytes of the fixed header of a payload at `schema_version`.
///
/// The one place the version→length mapping lives, so the writer, the parser and
/// the `blob_base` validation cannot disagree about where the directory starts.
/// An unknown (future) version is rejected before this is ever asked.
#[inline]
pub const fn header_len(schema_version: u32) -> u64 {
    match schema_version {
        0 | 1 => HEADER_LEN_V1,
        2 => HEADER_LEN_V2,
        3 => HEADER_LEN_V3,
        _ => HEADER_LEN_V4,
    }
}

/// Bytes of one tile-directory entry.
pub const DIR_ENTRY_LEN: u64 = 32;

/// A failure building or reading a `.inf_terrain` payload.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TerrainAssetError {
    #[error("terrain asset is shorter than its fixed header")]
    TooShort,
    #[error("not an .inf_terrain payload (bad magic)")]
    BadMagic,
    #[error("terrain asset schema v{found} is newer than this build (v{current})")]
    SchemaTooNew { found: u32, current: u32 },
    #[error("terrain asset is malformed: {0}")]
    Malformed(String),
    #[error("terrain tile {lod}:({tx},{tz}) failed to decode: {message}")]
    TileDecode {
        lod: u32,
        tx: i32,
        tz: i32,
        message: String,
    },
    #[error("terrain tile {lod}:({tx},{tz}) added twice")]
    DuplicateTile { lod: u32, tx: i32, tz: i32 },
}

type Result<T> = std::result::Result<T, TerrainAssetError>;

/// Round `n` up to the next multiple of [`TILE_ALIGN`].
#[inline]
fn align_up(n: u64) -> u64 {
    n.next_multiple_of(TILE_ALIGN)
}

/// The bincode configuration tile blobs use — the **same** `standard()` the
/// `.inf_lvl` path uses, so a tile's bytes are identical in both containers.
fn tile_config() -> impl bincode::config::Config {
    bincode::config::standard()
}

/// Encode one tile to its canonical blob bytes.
pub fn encode_tile(tile: &TerrainTile) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(tile, tile_config())
        .map_err(|e| TerrainAssetError::Malformed(format!("tile encode: {e}")))
}

/// Decode one tile from its canonical blob bytes, at the **current** schema.
///
/// Prefer [`decode_tile_at`] anywhere the payload's own version is in hand — a
/// blob from an older asset has a shorter layout and this would run off its end
/// (loudly: bincode reports an unexpected end, never a silent wrong tile).
pub fn decode_tile(bytes: &[u8]) -> std::result::Result<TerrainTile, String> {
    decode_tile_at(bytes, TERRAIN_ASSET_SCHEMA_VERSION)
}

/// Decode one tile blob written at `schema_version`.
///
/// The version is the **only** thing that says which tile wire layout the bytes
/// hold (bincode carries no field names or count), so this is the single place
/// the mapping lives — the asset-container column of [`TerrainTileFrozenV1`]'s
/// generation table:
///
/// | payload schema | tile layout |
/// |---|---|
/// | ≤ 2 | [`TerrainTileFrozenV1`] — origin + heights + weights, lifted with empty maps and biomes |
/// | 3 | [`TerrainTileFrozenV2`] — the above plus the P19.1 data maps, lifted with empty biomes |
/// | ≥ 4 | [`TerrainTile`] — the above plus the sparse P19.2 biome ids |
pub fn decode_tile_at(
    bytes: &[u8],
    schema_version: u32,
) -> std::result::Result<TerrainTile, String> {
    if schema_version <= 2 {
        return bincode::serde::decode_from_slice::<TerrainTileFrozenV1, _>(bytes, tile_config())
            .map(|(t, _)| t.into_current())
            .map_err(|e| e.to_string());
    }
    if schema_version == 3 {
        return bincode::serde::decode_from_slice::<TerrainTileFrozenV2, _>(bytes, tile_config())
            .map(|(t, _)| t.into_current())
            .map_err(|e| e.to_string());
    }
    bincode::serde::decode_from_slice(bytes, tile_config())
        .map(|(t, _)| t)
        .map_err(|e| e.to_string())
}

// ── header ──────────────────────────────────────────────────────────────────

/// The decoded fixed header of a `.inf_terrain` payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainAssetHeader {
    /// Payload schema version.
    pub schema_version: u32,
    /// Samples per tile side, at **every** LOD level.
    pub tile_resolution: u32,
    /// Level-0 world units between adjacent samples (level `n` is `mps · 2ⁿ`).
    pub meters_per_sample: f64,
    /// World anchor of tile `(0, 0)`'s sample `(0, 0)`.
    pub origin: DVec3,
    /// Number of LOD levels present (`1` = level 0 only).
    pub lod_levels: u32,
    /// Number of tile-directory entries.
    pub tile_count: u32,
    /// Absolute offset of the blob section within the payload.
    pub blob_base: u64,
    /// The [`PyramidOptions`] this asset's coarse levels were built with
    /// (schema v2+), or `None` for a **v1** payload, which did not record them.
    ///
    /// `None` means *unknown*, deliberately — not "the defaults". A write-back has
    /// to fall back to the defaults either way, but only the `None` case can
    /// actually reshape an asset, so only the `None` case warns.
    pub pyramid: Option<PyramidOptions>,
}

/// One tile-directory entry: where a tile's blob lives inside the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileDirEntry {
    /// Which tile this is.
    pub key: TileKey,
    /// Absolute offset of the blob **within the payload** (a multiple of
    /// [`TILE_ALIGN`]) — usable directly as a `seek` target on a loose file.
    pub offset: u64,
    /// Blob length in bytes (unpadded).
    pub len: u64,
}

// ── builder ─────────────────────────────────────────────────────────────────

/// Assembles a `.inf_terrain` payload tile by tile.
///
/// Tiles land in a `BTreeMap` keyed by [`TileKey`], so insertion order cannot
/// affect the output — [`build`](Self::build) is a pure function of the tile set.
#[derive(Debug, Clone)]
pub struct TerrainAssetBuilder {
    tile_resolution: u32,
    meters_per_sample: f64,
    origin: DVec3,
    pyramid: PyramidOptions,
    tiles: BTreeMap<TileKey, Vec<u8>>,
}

impl TerrainAssetBuilder {
    /// A builder for a terrain of the given level-0 configuration, anchored at the
    /// world origin.
    pub fn new(tile_resolution: u32, meters_per_sample: f64) -> Self {
        Self {
            tile_resolution: tile_resolution.max(2),
            meters_per_sample: if meters_per_sample > 0.0 {
                meters_per_sample
            } else {
                crate::DEFAULT_METERS_PER_SAMPLE
            },
            origin: DVec3::ZERO,
            pyramid: PyramidOptions::default(),
            tiles: BTreeMap::new(),
        }
    }

    /// Anchor the tile grid at an explicit world origin (reserved for world
    /// partitioning; a [`TerrainData`] grid is anchored at zero today).
    pub fn with_origin(mut self, origin: DVec3) -> Self {
        self.origin = origin;
        self
    }

    /// Record the [`PyramidOptions`] the coarse levels being staged were built
    /// with (schema v2 header; P16.6).
    ///
    /// **Set this to the options actually used.** The value is what a later
    /// write-back re-plans with, so a wrong one costs a needless total pyramid
    /// rebuild and an asset reshaped away from the shape its author chose — which
    /// is precisely the failure v2 exists to end. Defaults to
    /// [`PyramidOptions::default`], which is correct for every caller that used
    /// the defaults to build.
    pub fn with_pyramid(mut self, pyramid: PyramidOptions) -> Self {
        self.pyramid = pyramid;
        self
    }

    /// Add a tile, encoding it to its canonical blob. Errors on a duplicate key.
    pub fn insert(&mut self, key: TileKey, tile: &TerrainTile) -> Result<()> {
        self.insert_bytes(key, encode_tile(tile)?)
    }

    /// Add a tile from already-encoded blob bytes (a re-pack that never decodes).
    pub fn insert_bytes(&mut self, key: TileKey, bytes: Vec<u8>) -> Result<()> {
        if self.tiles.insert(key, bytes).is_some() {
            return Err(TerrainAssetError::DuplicateTile {
                lod: key.lod,
                tx: key.coord.0,
                tz: key.coord.1,
            });
        }
        Ok(())
    }

    /// Number of tiles staged.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// Serialize the payload image.
    pub fn build(self) -> Result<TerrainAsset> {
        let count = u32::try_from(self.tiles.len())
            .map_err(|_| TerrainAssetError::Malformed("more than u32::MAX tiles".into()))?;
        let lod_levels = self
            .tiles
            .keys()
            .map(|k| k.lod + 1)
            .max()
            .unwrap_or(1)
            .max(1);

        // Offsets first: the blob section starts after the header + full directory
        // (both already multiples of TILE_ALIGN, so no padding is ever needed
        // there — `align_up` states the invariant rather than relying on it), and
        // every tile after it starts on the next aligned boundary.
        let blob_base = align_up(HEADER_LEN + DIR_ENTRY_LEN * count as u64);
        // The options are u32 on disk; `min_tiles` is a usize in the API and a
        // count in practice, so a saturating narrow is exact for every value a
        // real pyramid can have and monotone for the absurd ones.
        let pyr_max_levels = self.pyramid.max_levels;
        let pyr_min_tiles = u32::try_from(self.pyramid.min_tiles).unwrap_or(u32::MAX);
        let mut offsets = Vec::with_capacity(self.tiles.len());
        let mut offset = blob_base;
        for blob in self.tiles.values() {
            offsets.push(offset);
            offset = align_up(offset + blob.len() as u64);
        }
        let total = offset;

        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&TERRAIN_ASSET_MAGIC);
        out.extend_from_slice(&TERRAIN_ASSET_SCHEMA_VERSION.to_le_bytes());
        out.extend_from_slice(&self.tile_resolution.to_le_bytes());
        out.extend_from_slice(&self.meters_per_sample.to_le_bytes());
        for v in self.origin.to_array() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&lod_levels.to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&blob_base.to_le_bytes());
        debug_assert_eq!(out.len() as u64, HEADER_LEN_V1);
        // ── v2 tail: the pyramid options + reserved zeros ──
        out.extend_from_slice(&pyr_max_levels.to_le_bytes());
        out.extend_from_slice(&pyr_min_tiles.to_le_bytes());
        out.resize(HEADER_LEN as usize, 0);
        debug_assert_eq!(out.len() as u64, HEADER_LEN);

        for ((key, blob), &off) in self.tiles.iter().zip(&offsets) {
            out.extend_from_slice(&key.lod.to_le_bytes());
            out.extend_from_slice(&key.coord.0.to_le_bytes());
            out.extend_from_slice(&key.coord.1.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes()); // reserved, always zero
            out.extend_from_slice(&off.to_le_bytes());
            out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        }
        debug_assert_eq!(out.len() as u64, HEADER_LEN + DIR_ENTRY_LEN * count as u64);

        // Blob section, zero-padded up to each offset (and to the end, so the
        // payload length is itself a multiple of TILE_ALIGN).
        for (blob, &off) in self.tiles.values().zip(&offsets) {
            out.resize(off as usize, 0);
            out.extend_from_slice(blob);
        }
        out.resize(total as usize, 0);

        TerrainAsset::from_bytes(out)
    }
}

/// Build a `.inf_terrain` payload from an authored terrain plus its coarse LOD
/// levels ([`crate::pyramid::build_pyramid`]).
///
/// Level 0 is `data`'s own tiles; each [`PyramidLevel`] contributes its own level.
/// Anchored at the world origin — see [`TerrainAssetBuilder::with_origin`].
///
/// `opts` must be **the options `pyramid` was built with**: they are recorded in
/// the v2 header so a later write-back re-plans to the same shape (P16.6). It is
/// an explicit parameter rather than a default precisely so the pyramid and the
/// options recorded beside it cannot drift apart at a call site.
pub fn build_terrain_asset(
    data: &TerrainData,
    pyramid: &[PyramidLevel],
    opts: PyramidOptions,
) -> Result<TerrainAsset> {
    let mut b = TerrainAssetBuilder::new(data.tile_resolution(), data.meters_per_sample())
        .with_pyramid(opts);
    for (&coord, tile) in data.tiles() {
        b.insert(TileKey::lod0(coord), tile)?;
    }
    for level in pyramid {
        for (&coord, tile) in &level.tiles {
            b.insert(TileKey::new(level.lod, coord), tile)?;
        }
    }
    b.build()
}

// ── the payload ─────────────────────────────────────────────────────────────

/// A validated `.inf_terrain` payload image.
///
/// Owns the bytes; [`reader`](Self::reader) borrows a random-access view over
/// them. See the module docs for why [`as_bytes`](Self::as_bytes) — not
/// `inf_asset::encode` — is what goes on disk and into a pack.
#[derive(Clone, PartialEq)]
pub struct TerrainAsset {
    bytes: Vec<u8>,
    header: TerrainAssetHeader,
}

impl TerrainAsset {
    /// The asset kind this payload is written as. An inherent const, not an
    /// [`AssetPayload`](inf_asset::AssetPayload) impl — see the module docs: the
    /// generic write path would frame these bytes and misalign every tile, so it
    /// is closed at compile time and the kind is published here instead.
    pub const KIND: AssetKind = AssetKind::Terrain;

    /// The current payload schema version (mirrors
    /// [`TERRAIN_ASSET_SCHEMA_VERSION`], published beside [`KIND`](Self::KIND)).
    pub const SCHEMA_VERSION: u32 = TERRAIN_ASSET_SCHEMA_VERSION;

    /// Validate and take ownership of a payload image.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let header = parse(&bytes)?.0;
        Ok(Self { bytes, header })
    }

    /// The canonical payload bytes — what is written to `*.inf_terrain` and packed.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume into the payload bytes.
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// The decoded header.
    #[inline]
    pub fn header(&self) -> &TerrainAssetHeader {
        &self.header
    }

    /// A random-access view over the payload.
    pub fn reader(&self) -> TerrainAssetReader<&[u8]> {
        // Already validated in `from_bytes`; re-parsing cannot fail.
        TerrainAssetReader::new(self.bytes.as_slice()).expect("validated payload")
    }
}

impl std::fmt::Debug for TerrainAsset {
    /// Summarizes; never dumps the (possibly gigabyte) tile blobs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TerrainAsset")
            .field("schema_version", &self.header.schema_version)
            .field("tile_resolution", &self.header.tile_resolution)
            .field("meters_per_sample", &self.header.meters_per_sample)
            .field("lod_levels", &self.header.lod_levels)
            .field("tile_count", &self.header.tile_count)
            .field("pyramid", &self.header.pyramid)
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

/// Write a `.inf_terrain` payload to `path` — **the one sanctioned loose-file
/// writer** (see the module docs).
///
/// Writes the raw image, never a framed encoding, and does it **atomically**: the
/// bytes go to a sibling temp file that is then renamed over `path`, exactly like
/// [`PackWriter::write_to_file`](inf_asset::PackWriter::write_to_file) and for the
/// same reason — a re-import routinely targets a file some other process may still
/// have mapped, and truncating it under a live mapping is the mutation an mmap's
/// safety contract forbids. A failed write cleans up its temp file rather than
/// leaving litter beside the asset.
///
/// Returns the bytes written, so a caller can hash them into the sidecar without
/// re-reading the file.
pub fn write_terrain_asset<'a>(
    path: &std::path::Path,
    asset: &'a TerrainAsset,
) -> std::io::Result<&'a [u8]> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = asset.as_bytes();
    let tmp = {
        let mut s = path.as_os_str().to_os_string();
        s.push(format!(".{}.tmp", std::process::id()));
        std::path::PathBuf::from(s)
    };
    if let Err(err) = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path)) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    Ok(bytes)
}

/// Read a `.inf_terrain` payload from `path`, validating it.
pub fn read_terrain_asset(path: &std::path::Path) -> std::io::Result<TerrainAsset> {
    let bytes = std::fs::read(path)?;
    TerrainAsset::from_bytes(bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ── reader ──────────────────────────────────────────────────────────────────

/// Random access over a `.inf_terrain` payload.
///
/// Generic over the byte source so the *same* reader serves every backing: an
/// owned `Vec<u8>` (a loose file read), a `&[u8]` borrowed from a
/// [`PackReader::read_ref`](inf_asset::PackReader::read_ref) mapping, or a `Cow`
/// straight off it. [`TerrainAssetView`] is the borrowed shorthand.
///
/// The header + directory are parsed and validated once at construction; every
/// [`tile_bytes`](Self::tile_bytes) after that is a binary search plus a slice —
/// no decode, no allocation.
#[derive(Debug, Clone)]
pub struct TerrainAssetReader<B> {
    bytes: B,
    header: TerrainAssetHeader,
    dir: Vec<TileDirEntry>,
}

/// A [`TerrainAssetReader`] borrowing its bytes (the pack-mapping case).
pub type TerrainAssetView<'a> = TerrainAssetReader<&'a [u8]>;

impl<B: AsRef<[u8]>> TerrainAssetReader<B> {
    /// Parse + validate a payload. Rejects a bad magic, a newer schema, an
    /// out-of-bounds or misaligned blob, and an out-of-order directory (which
    /// would break both the binary search and the byte-determinism guarantee).
    pub fn new(bytes: B) -> Result<Self> {
        let (header, dir) = parse(bytes.as_ref())?;
        Ok(Self { bytes, header, dir })
    }

    /// The whole payload image.
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    /// The decoded header.
    #[inline]
    pub fn header(&self) -> &TerrainAssetHeader {
        &self.header
    }

    /// Samples per tile side.
    #[inline]
    pub fn tile_resolution(&self) -> u32 {
        self.header.tile_resolution
    }

    /// Level-0 metres per sample.
    #[inline]
    pub fn meters_per_sample(&self) -> f64 {
        self.header.meters_per_sample
    }

    /// Metres per sample at `lod` (`level0 · 2^lod`).
    #[inline]
    pub fn meters_per_sample_at(&self, lod: u32) -> f64 {
        self.header.meters_per_sample * (1u64 << lod.min(63)) as f64
    }

    /// World anchor of tile `(0, 0)`'s sample `(0, 0)`.
    #[inline]
    pub fn origin(&self) -> DVec3 {
        self.header.origin
    }

    /// Number of LOD levels present.
    #[inline]
    pub fn lod_levels(&self) -> u32 {
        self.header.lod_levels
    }

    /// The [`PyramidOptions`] the coarse levels were built with, or `None` for a
    /// **v1** payload that did not record them (P16.6).
    #[inline]
    pub fn pyramid_options(&self) -> Option<PyramidOptions> {
        self.header.pyramid
    }

    /// Number of tiles across all levels.
    #[inline]
    pub fn tile_count(&self) -> usize {
        self.dir.len()
    }
    pub fn is_empty(&self) -> bool {
        self.dir.is_empty()
    }

    /// The tile directory, in [`TileKey`] order (LOD-major) — the seam a
    /// loose-file consumer seeks with.
    #[inline]
    pub fn directory(&self) -> &[TileDirEntry] {
        &self.dir
    }

    /// Every key present, in directory order.
    pub fn keys(&self) -> impl Iterator<Item = TileKey> + '_ {
        self.dir.iter().map(|e| e.key)
    }

    /// The directory entry for `key`.
    pub fn entry(&self, key: TileKey) -> Option<&TileDirEntry> {
        self.dir
            .binary_search_by(|e| e.key.cmp(&key))
            .ok()
            .map(|i| &self.dir[i])
    }

    /// A tile's blob, **borrowed from the payload** — 16-byte aligned within it.
    pub fn tile_bytes(&self, key: TileKey) -> Option<&[u8]> {
        let e = self.entry(key)?;
        // Bounds + alignment were validated in `new`.
        Some(&self.bytes.as_ref()[e.offset as usize..(e.offset + e.len) as usize])
    }

    /// Decode a tile, through **this payload's own** schema version — so a v1/v2
    /// asset's tiles come back with empty data maps rather than a decode error.
    pub fn tile(&self, key: TileKey) -> Result<Option<TerrainTile>> {
        let Some(bytes) = self.tile_bytes(key) else {
            return Ok(None);
        };
        decode_tile_at(bytes, self.header.schema_version)
            .map(Some)
            .map_err(|message| TerrainAssetError::TileDecode {
                lod: key.lod,
                tx: key.coord.0,
                tz: key.coord.1,
                message,
            })
    }

    /// Rebuild a [`TerrainData`] from the asset's **level-0** tiles (the authored
    /// detail level). Coarse levels are a streaming/render concern and are not
    /// part of the editable heightfield — reach them through
    /// [`tile`](Self::tile) or the residency layer.
    pub fn to_terrain_data(&self) -> Result<TerrainData> {
        let mut data = TerrainData::new(self.tile_resolution(), self.meters_per_sample());
        for e in self.dir.iter().filter(|e| e.key.is_lod0()) {
            let tile = self.tile(e.key)?.expect("directory entry resolves");
            let _ = data.insert_tile(e.key.coord, tile);
        }
        data.clear_dirty();
        Ok(data)
    }
}

/// Parse + validate the header and directory of a payload image.
fn parse(data: &[u8]) -> Result<(TerrainAssetHeader, Vec<TileDirEntry>)> {
    if (data.len() as u64) < HEADER_LEN_V1 {
        return Err(TerrainAssetError::TooShort);
    }
    if data[0..8] != TERRAIN_ASSET_MAGIC {
        return Err(TerrainAssetError::BadMagic);
    }
    let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
    let f64_at = |o: usize| f64::from_le_bytes(data[o..o + 8].try_into().unwrap());

    let schema_version = u32_at(8);
    if schema_version > TERRAIN_ASSET_SCHEMA_VERSION {
        return Err(TerrainAssetError::SchemaTooNew {
            found: schema_version,
            current: TERRAIN_ASSET_SCHEMA_VERSION,
        });
    }
    let tile_resolution = u32_at(12);
    if tile_resolution < 2 {
        return Err(TerrainAssetError::Malformed(format!(
            "tile_resolution {tile_resolution} < 2"
        )));
    }
    let meters_per_sample = f64_at(16);
    if !(meters_per_sample.is_finite() && meters_per_sample > 0.0) {
        return Err(TerrainAssetError::Malformed(format!(
            "meters_per_sample {meters_per_sample} is not a positive finite length"
        )));
    }
    let origin = DVec3::new(f64_at(24), f64_at(32), f64_at(40));
    if !origin.is_finite() {
        return Err(TerrainAssetError::Malformed("origin is not finite".into()));
    }
    let lod_levels = u32_at(48);
    let tile_count = u32_at(52);
    let blob_base = u64_at(56);

    // The header's length — and therefore where the directory starts — is a pure
    // function of the schema version (see [`header_len`]).
    let hlen = header_len(schema_version);
    if (data.len() as u64) < hlen {
        return Err(TerrainAssetError::TooShort);
    }
    let pyramid = (schema_version >= 2).then(|| PyramidOptions {
        max_levels: u32_at(64),
        min_tiles: u32_at(68) as usize,
    });

    let dir_end = hlen + DIR_ENTRY_LEN * tile_count as u64;
    if (data.len() as u64) < dir_end {
        return Err(TerrainAssetError::Malformed(
            "truncated in the tile directory".into(),
        ));
    }
    if blob_base != align_up(dir_end) || blob_base > data.len() as u64 {
        return Err(TerrainAssetError::Malformed(format!(
            "blob_base {blob_base} does not follow the directory"
        )));
    }

    let mut dir = Vec::with_capacity(tile_count as usize);
    let mut prev: Option<TileKey> = None;
    // Blobs are laid out in directory order, so each must start at or after the
    // previous one's end. This is the **overlap** check: two entries sharing bytes
    // would let one tile's edit corrupt another, and a reader has no way to notice.
    let mut prev_end = blob_base;
    for i in 0..tile_count as u64 {
        let base = (hlen + DIR_ENTRY_LEN * i) as usize;
        let key = TileKey {
            lod: u32::from_le_bytes(data[base..base + 4].try_into().unwrap()),
            coord: (
                i32::from_le_bytes(data[base + 4..base + 8].try_into().unwrap()),
                i32::from_le_bytes(data[base + 8..base + 12].try_into().unwrap()),
            ),
        };
        let offset = u64::from_le_bytes(data[base + 16..base + 24].try_into().unwrap());
        let len = u64::from_le_bytes(data[base + 24..base + 32].try_into().unwrap());
        if let Some(p) = prev {
            if key <= p {
                return Err(TerrainAssetError::Malformed(
                    "tile directory is not strictly ascending".into(),
                ));
            }
        }
        prev = Some(key);
        if key.lod >= lod_levels {
            return Err(TerrainAssetError::Malformed(format!(
                "tile lod {} outside the declared {lod_levels} level(s)",
                key.lod
            )));
        }
        if offset % TILE_ALIGN != 0 {
            return Err(TerrainAssetError::Malformed(format!(
                "tile blob at {offset} is not {TILE_ALIGN}-byte aligned"
            )));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| TerrainAssetError::Malformed("tile length overflow".into()))?;
        if offset < blob_base || end > data.len() as u64 {
            return Err(TerrainAssetError::Malformed(format!(
                "tile blob [{offset}, {end}) out of bounds"
            )));
        }
        if offset < prev_end {
            return Err(TerrainAssetError::Malformed(format!(
                "tile blob [{offset}, {end}) overlaps the previous blob (ends at {prev_end})"
            )));
        }
        prev_end = end;
        dir.push(TileDirEntry { key, offset, len });
    }

    Ok((
        TerrainAssetHeader {
            schema_version,
            tile_resolution,
            meters_per_sample,
            origin,
            lod_levels,
            tile_count,
            blob_base,
            pyramid,
        },
        dir,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyramid::{build_pyramid, PyramidOptions};

    fn height_fn(x: f64, z: f64) -> f64 {
        x * 0.25 - z * 0.125 + x * z * 0.001 + 5.0
    }

    /// A 4 × 4-tile terrain with one painted tile (so the blobs exercise both the
    /// sparse and the materialized weight form).
    fn sample_terrain() -> TerrainData {
        let mut t = TerrainData::new(5, 2.0);
        for tz in 0..4 {
            for tx in 0..4 {
                t.author_tile((tx, tz), height_fn);
            }
        }
        t.get_tile_mut((1, 1))
            .unwrap()
            .set_weight_sample(5, 2, 3, [10, 20, 30, 195]);
        t
    }

    fn sample_asset() -> TerrainAsset {
        let t = sample_terrain();
        let p = build_pyramid(&t, PyramidOptions::default());
        build_terrain_asset(&t, &p, PyramidOptions::default()).unwrap()
    }

    // ── schema v2: the pyramid options in the header (P16.6) ─────────────────

    /// Re-encode `asset`'s tiles as a **v1** payload image — the pre-P16.6 layout,
    /// byte for byte (64-byte header, no pyramid options).
    ///
    /// This is the v1 *fixture*: hand-built rather than committed, because the
    /// bytes are a pure function of the tiles and a committed blob would only be a
    /// less-inspectable copy of this loop. What it pins is the real thing — a v1
    /// image, produced exactly as the shipped v1 writer produced one, must keep
    /// loading forever.
    fn v1_image(asset: &TerrainAsset) -> Vec<u8> {
        let r = asset.reader();
        let count = r.tile_count() as u32;
        let blob_base = (HEADER_LEN_V1 + DIR_ENTRY_LEN * count as u64).next_multiple_of(TILE_ALIGN);
        let mut offsets = Vec::with_capacity(count as usize);
        let mut offset = blob_base;
        for e in r.directory() {
            offsets.push(offset);
            offset = (offset + e.len).next_multiple_of(TILE_ALIGN);
        }
        let total = offset;

        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&TERRAIN_ASSET_MAGIC);
        out.extend_from_slice(&1u32.to_le_bytes()); // schema v1
        out.extend_from_slice(&r.tile_resolution().to_le_bytes());
        out.extend_from_slice(&r.meters_per_sample().to_le_bytes());
        for v in r.origin().to_array() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&r.lod_levels().to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&blob_base.to_le_bytes());
        assert_eq!(out.len() as u64, HEADER_LEN_V1);
        for (e, &off) in r.directory().iter().zip(&offsets) {
            out.extend_from_slice(&e.key.lod.to_le_bytes());
            out.extend_from_slice(&e.key.coord.0.to_le_bytes());
            out.extend_from_slice(&e.key.coord.1.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&off.to_le_bytes());
            out.extend_from_slice(&e.len.to_le_bytes());
        }
        for (e, &off) in r.directory().iter().zip(&offsets) {
            out.resize(off as usize, 0);
            out.extend_from_slice(r.tile_bytes(e.key).unwrap());
        }
        out.resize(total as usize, 0);
        out
    }

    /// The header this build writes is v4, its length is still 128, and the
    /// pyramid options it was built with come back out of it.
    #[test]
    fn v2_records_the_pyramid_options() {
        let t = sample_terrain();
        let opts = PyramidOptions {
            max_levels: 3,
            min_tiles: 2,
        };
        let p = build_pyramid(&t, opts);
        let asset = build_terrain_asset(&t, &p, opts).unwrap();
        let r = asset.reader();
        assert_eq!(r.header().schema_version, TERRAIN_ASSET_SCHEMA_VERSION);
        assert_eq!(r.pyramid_options(), Some(opts));
        // The directory starts after a 128-byte header, and the reserved tail is
        // zeros (so a v5 that fills it can be told from a v4 that did not).
        // P19.1 and P19.2 kept the header length: v3 and v4 changed the *tile
        // blob*, not this.
        assert_eq!(header_len(2), HEADER_LEN_V2);
        assert_eq!(header_len(3), HEADER_LEN_V3);
        assert_eq!(header_len(4), HEADER_LEN_V4);
        assert_eq!(HEADER_LEN_V3, HEADER_LEN_V2);
        assert_eq!(HEADER_LEN_V4, HEADER_LEN_V2);
        assert!(asset.as_bytes()[72..HEADER_LEN_V2 as usize]
            .iter()
            .all(|&b| b == 0));
        assert_eq!(
            r.header().blob_base,
            (HEADER_LEN_V2 + DIR_ENTRY_LEN * r.tile_count() as u64).next_multiple_of(TILE_ALIGN)
        );
    }

    // ── schema v3 / v4: the tile blob grows layers (P19.1, P19.2) ───────────

    /// Re-encode `asset`'s tiles as a **legacy** payload image at `version` — the
    /// pre-P19.1 (`2`) or pre-P19.2 (`3`) layout, byte for byte: the same 128-byte
    /// header stamped `version`, and tile blobs written through that generation's
    /// frozen wire type.
    ///
    /// Hand-built for the same reason `v1_image` is: the bytes are a pure function
    /// of the tiles, and a committed blob would only be a less-inspectable copy of
    /// this loop. One helper for both generations, so the two ladder tests cannot
    /// drift apart.
    fn legacy_image(asset: &TerrainAsset, version: u32) -> Vec<u8> {
        assert!(
            (2..=3).contains(&version),
            "legacy_image writes generation-1 (v2) or generation-2 (v3) blobs"
        );
        let r = asset.reader();
        let count = r.tile_count() as u32;
        // Down-blessed blobs (the maps dropped) — the exact bytes the v2 writer
        // produced. They can differ in *length* from the v3 ones, so offsets are
        // recomputed rather than copied.
        let blobs: Vec<Vec<u8>> = r
            .directory()
            .iter()
            .map(|e| {
                let tile = r.tile(e.key).unwrap().unwrap();
                if version == 2 {
                    bincode::serde::encode_to_vec(
                        TerrainTileFrozenV1::from_current(&tile),
                        tile_config(),
                    )
                    .unwrap()
                } else {
                    bincode::serde::encode_to_vec(
                        TerrainTileFrozenV2::from_current(&tile),
                        tile_config(),
                    )
                    .unwrap()
                }
            })
            .collect();
        let blob_base = (HEADER_LEN_V2 + DIR_ENTRY_LEN * count as u64).next_multiple_of(TILE_ALIGN);
        let mut offsets = Vec::with_capacity(count as usize);
        let mut offset = blob_base;
        for b in &blobs {
            offsets.push(offset);
            offset = (offset + b.len() as u64).next_multiple_of(TILE_ALIGN);
        }
        let total = offset;

        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&TERRAIN_ASSET_MAGIC);
        out.extend_from_slice(&version.to_le_bytes()); // the legacy schema
        out.extend_from_slice(&r.tile_resolution().to_le_bytes());
        out.extend_from_slice(&r.meters_per_sample().to_le_bytes());
        for v in r.origin().to_array() {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&r.lod_levels().to_le_bytes());
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&blob_base.to_le_bytes());
        let opts = r.pyramid_options().unwrap_or_default();
        out.extend_from_slice(&opts.max_levels.to_le_bytes());
        out.extend_from_slice(&(opts.min_tiles as u32).to_le_bytes());
        out.resize(HEADER_LEN_V2 as usize, 0);
        for (e, (&off, b)) in r.directory().iter().zip(offsets.iter().zip(&blobs)) {
            out.extend_from_slice(&e.key.lod.to_le_bytes());
            out.extend_from_slice(&e.key.coord.0.to_le_bytes());
            out.extend_from_slice(&e.key.coord.1.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&off.to_le_bytes());
            out.extend_from_slice(&(b.len() as u64).to_le_bytes());
        }
        for (&off, b) in offsets.iter().zip(&blobs) {
            out.resize(off as usize, 0);
            out.extend_from_slice(b);
        }
        out.resize(total as usize, 0);
        out
    }

    /// The pre-P19.1 image (generation-1 tile blobs).
    fn v2_image(asset: &TerrainAsset) -> Vec<u8> {
        legacy_image(asset, 2)
    }

    /// The pre-P19.2 image (generation-2 tile blobs — maps, no biomes).
    fn v3_image(asset: &TerrainAsset) -> Vec<u8> {
        legacy_image(asset, 3)
    }

    /// **A v2 payload loads forever** — every tile decodes through the frozen
    /// pre-P19.1 wire type and comes back with the never-eroded default maps.
    ///
    /// This is the load-bearing half of the P19.1 asset bump: bincode is
    /// positional, so a v2 blob fed to the grown tile would run off its end. The
    /// version — not a guess, not a length heuristic — is what selects the layout.
    #[test]
    fn a_v2_payload_still_loads_with_default_data_maps() {
        // A terrain whose maps are genuinely populated, so "the maps came back
        // empty" cannot pass by accident.
        let mut t = sample_terrain();
        t.get_tile_mut((2, 2))
            .unwrap()
            .set_map_texel(5, 1, 1, [3.0, 2.0, 1.0]);
        let p = build_pyramid(&t, PyramidOptions::default());
        let asset = build_terrain_asset(&t, &p, PyramidOptions::default()).unwrap();
        assert_eq!(
            asset.reader().header().schema_version,
            TERRAIN_ASSET_SCHEMA_VERSION
        );
        assert!(!asset
            .reader()
            .tile(TileKey::lod0((2, 2)))
            .unwrap()
            .unwrap()
            .maps_are_default());

        let v2 = v2_image(&asset);
        assert_eq!(v2[8..12], 2u32.to_le_bytes(), "the fixture must be v2");
        let back = TerrainAsset::from_bytes(v2).expect("a v2 payload must still load");
        let old = back.reader();
        assert_eq!(old.header().schema_version, 2);
        assert_eq!(old.tile_count(), asset.reader().tile_count());
        for key in old.keys() {
            let tile = old.tile(key).unwrap().expect("directory entry resolves");
            assert!(tile.maps_are_default(), "{key:?} conjured data maps");
            assert!(tile.biomes_are_default(), "{key:?} conjured biome ids");
            // Heights and weights survive verbatim through the frozen type.
            let new = asset.reader().tile(key).unwrap().unwrap();
            assert_eq!(tile.heights(), new.heights(), "{key:?} heights moved");
            assert_eq!(tile.weights(), new.weights(), "{key:?} weights moved");
        }

        // The raw blobs really are shorter — the current form appends one length
        // byte per untouched layer per tile (maps at v3, biomes at v4), and a
        // dense buffer for the eroded one.
        let key = TileKey::lod0((0, 0));
        assert_eq!(
            asset.reader().tile_bytes(key).unwrap().len(),
            old.tile_bytes(key).unwrap().len() + 2,
            "an untouched tile costs exactly one extra byte per layer added since v2"
        );
    }

    /// **A v3 payload loads forever** — the generation-2 ladder rung. Every tile
    /// decodes through [`TerrainTileFrozenV2`], **keeps its erosion data maps**
    /// (unlike the v2 rung, which cannot carry them) and comes back with the
    /// unpainted default biome ids.
    ///
    /// The two-rung structure is the point: "old bytes load" is not one claim but
    /// one per generation, and only a per-rung test can catch a `decode_tile_at`
    /// that routes v3 through the wrong wire type.
    #[test]
    fn a_v3_payload_still_loads_with_default_biomes() {
        // Populate BOTH new layers, so neither "it came back empty" can pass by
        // accident and the maps' survival is a real assertion.
        let mut t = sample_terrain();
        t.get_tile_mut((2, 2))
            .unwrap()
            .set_map_texel(5, 1, 1, [3.0, 2.0, 1.0]);
        t.get_tile_mut((2, 2)).unwrap().set_biome_sample(5, 1, 1, 7);
        let p = build_pyramid(&t, PyramidOptions::default());
        let asset = build_terrain_asset(&t, &p, PyramidOptions::default()).unwrap();
        assert!(!asset
            .reader()
            .tile(TileKey::lod0((2, 2)))
            .unwrap()
            .unwrap()
            .biomes_are_default());

        let v3 = v3_image(&asset);
        assert_eq!(v3[8..12], 3u32.to_le_bytes(), "the fixture must be v3");
        let back = TerrainAsset::from_bytes(v3).expect("a v3 payload must still load");
        let old = back.reader();
        assert_eq!(old.header().schema_version, 3);
        assert_eq!(old.tile_count(), asset.reader().tile_count());
        for key in old.keys() {
            let tile = old.tile(key).unwrap().expect("directory entry resolves");
            assert!(tile.biomes_are_default(), "{key:?} conjured biome ids");
            let new = asset.reader().tile(key).unwrap().unwrap();
            assert_eq!(tile.heights(), new.heights(), "{key:?} heights moved");
            assert_eq!(tile.weights(), new.weights(), "{key:?} weights moved");
            assert_eq!(tile.maps(), new.maps(), "{key:?} lost its data maps");
        }

        // The wire cost of v4 over v3, priced: one zero-length count per tile.
        let key = TileKey::lod0((0, 0));
        assert_eq!(
            asset.reader().tile_bytes(key).unwrap().len(),
            old.tile_bytes(key).unwrap().len() + 1,
            "an unpainted tile costs exactly one extra byte at v4"
        );
    }

    /// A **write-back over a v3 source transcodes** too — the generation-2 twin
    /// of `a_v2_source_is_transcoded_on_write_back`, and the rung that would
    /// silently pass if the rewrite copied blobs it "knew" were already current.
    #[test]
    fn a_v3_source_is_transcoded_on_write_back() {
        let mut t = sample_terrain();
        t.get_tile_mut((1, 1))
            .unwrap()
            .set_map_texel(5, 2, 2, [4.0, 1.0, 0.5]);
        let p = build_pyramid(&t, PyramidOptions::default());
        let current = build_terrain_asset(&t, &p, PyramidOptions::default()).unwrap();
        let old = TerrainAsset::from_bytes(v3_image(&current)).unwrap();

        let mut edited = t.clone();
        edited
            .get_tile_mut((0, 0))
            .unwrap()
            .set_biome_sample(5, 0, 0, 3);
        let mut edits = crate::writeback::TerrainEdits::default();
        edits
            .changed
            .insert((0, 0), edited.get_tile((0, 0)).unwrap().clone());

        let out = crate::writeback::rewrite_terrain_asset(
            &old.reader(),
            &edits,
            PyramidOptions::default(),
        )
        .unwrap()
        .unwrap();
        let r = out.reader();
        assert_eq!(r.header().schema_version, TERRAIN_ASSET_SCHEMA_VERSION);
        for key in r.keys() {
            // A passed-through v3 blob would fail here with an unexpected end.
            let tile = decode_tile(r.tile_bytes(key).unwrap())
                .unwrap_or_else(|e| panic!("{key:?} did not transcode: {e}"));
            if key == TileKey::lod0((0, 0)) {
                assert_eq!(tile.biome_sample(5, 0, 0), 3);
            } else if key.is_lod0() {
                assert!(tile.biomes_are_default(), "{key:?} conjured biome ids");
            } else {
                // A coarse page over the painted block reduces the ids (P19.3);
                // one over unpainted ground still costs nothing.
                assert!(
                    tile.biomes_are_default() || tile.biome_sample(5, 0, 0) == 3,
                    "{key:?} invented a biome id the fine level never had"
                );
            }
            // The v3 source's data maps rode through the transcode untouched.
            if key == TileKey::lod0((1, 1)) {
                assert_eq!(tile.map_texel(5, 2, 2), [4.0, 1.0, 0.5]);
            }
        }
    }

    /// A **write-back over a v2 source transcodes** rather than copying bytes: the
    /// rewritten image is stamped at the current schema and every tile in it —
    /// including the ones the edit never touched — decodes there.
    ///
    /// The failure this pins is the quiet one: a current header over v2 blobs
    /// passes every structural check in `parse` and only surfaces as a corrupt
    /// tile on some later load, on some other machine.
    #[test]
    fn a_v2_source_is_transcoded_on_write_back() {
        let t = sample_terrain();
        let p = build_pyramid(&t, PyramidOptions::default());
        let current = build_terrain_asset(&t, &p, PyramidOptions::default()).unwrap();
        let old = TerrainAsset::from_bytes(v2_image(&current)).unwrap();

        // Edit exactly one tile.
        let mut edited = t.clone();
        edited
            .get_tile_mut((0, 0))
            .unwrap()
            .set_map_texel(5, 0, 0, [9.0, 0.0, 0.0]);
        let mut edits = crate::writeback::TerrainEdits::default();
        edits
            .changed
            .insert((0, 0), edited.get_tile((0, 0)).unwrap().clone());

        let out = crate::writeback::rewrite_terrain_asset(
            &old.reader(),
            &edits,
            PyramidOptions::default(),
        )
        .unwrap()
        .unwrap();
        let r = out.reader();
        assert_eq!(r.header().schema_version, TERRAIN_ASSET_SCHEMA_VERSION);
        for key in r.keys() {
            // Decoding at the *current* schema must succeed for every tile — a
            // passed-through v2 blob would fail here with an unexpected end.
            let tile = decode_tile(r.tile_bytes(key).unwrap())
                .unwrap_or_else(|e| panic!("{key:?} did not transcode: {e}"));
            if key == TileKey::lod0((0, 0)) {
                assert_eq!(tile.map_texel(5, 0, 0), [9.0, 0.0, 0.0]);
            } else if key.is_lod0() {
                assert!(tile.maps_are_default(), "{key:?} conjured data maps");
            } else {
                // A coarse page over the eroded block sums the maps (P19.3); the
                // near corner decimates, so it is the fine value verbatim.
                assert!(
                    tile.maps_are_default() || tile.map_texel(5, 0, 0) == [9.0, 0.0, 0.0],
                    "{key:?} invented a data-map value the fine level never had"
                );
            }
        }
    }

    /// **Byte-identity for v2 rebuilds.** Two builds of one terrain with the same
    /// options are byte-identical; a build with *different* options differs only
    /// where it must (the 8 recorded bytes) when the pyramid shape is unchanged.
    #[test]
    fn v2_rebuilds_are_byte_identical() {
        let t = sample_terrain();
        let opts = PyramidOptions::default();
        let p = build_pyramid(&t, opts);
        let a = build_terrain_asset(&t, &p, opts).unwrap();
        let b = build_terrain_asset(&t, &p, opts).unwrap();
        assert_eq!(a.as_bytes(), b.as_bytes(), "a rebuild moved bytes");

        // Same tiles, different recorded options ⇒ same length, same directory,
        // same blobs — and exactly the two option words differ.
        let other = PyramidOptions {
            max_levels: 5,
            min_tiles: 3,
        };
        let c = TerrainAssetBuilder::new(t.tile_resolution(), t.meters_per_sample())
            .with_pyramid(other);
        let mut c = c;
        for (&coord, tile) in t.tiles() {
            c.insert(TileKey::lod0(coord), tile).unwrap();
        }
        for level in &p {
            for (&coord, tile) in &level.tiles {
                c.insert(TileKey::new(level.lod, coord), tile).unwrap();
            }
        }
        let c = c.build().unwrap();
        assert_eq!(a.as_bytes().len(), c.as_bytes().len());
        let differing: Vec<usize> = a
            .as_bytes()
            .iter()
            .zip(c.as_bytes())
            .enumerate()
            .filter(|(_, (x, y))| x != y)
            .map(|(i, _)| i)
            .collect();
        assert!(
            differing.iter().all(|&i| (64..72).contains(&i)),
            "options changed bytes outside the header slot: {differing:?}"
        );
        assert_eq!(c.reader().pyramid_options(), Some(other));
    }

    /// **A v1 payload loads forever**, and reports its pyramid options as
    /// *unknown* rather than as the defaults — the distinction the write-back
    /// warning is built on.
    #[test]
    fn a_v1_payload_still_loads_and_reports_unknown_options() {
        let asset = sample_asset();
        let v1 = v1_image(&asset);
        assert_eq!(v1[8..12], 1u32.to_le_bytes(), "the fixture must be v1");
        assert_eq!(header_len(1), HEADER_LEN_V1);

        let back = TerrainAsset::from_bytes(v1.clone()).expect("a v1 payload must still load");
        let (old, new) = (back.reader(), asset.reader());
        assert_eq!(old.header().schema_version, 1);
        assert_eq!(old.pyramid_options(), None, "v1 options are UNKNOWN");
        assert_eq!(new.pyramid_options(), Some(PyramidOptions::default()));

        // Every tile is byte-identical through the older container.
        assert_eq!(old.tile_count(), new.tile_count());
        assert_eq!(old.tile_resolution(), new.tile_resolution());
        assert_eq!(old.meters_per_sample(), new.meters_per_sample());
        assert_eq!(old.lod_levels(), new.lod_levels());
        for key in new.keys() {
            assert_eq!(old.tile_bytes(key), new.tile_bytes(key), "{key:?}");
        }
        // …and a v1 image is genuinely 64 bytes shorter in its header.
        assert_eq!(
            back.reader().header().blob_base + (v1.len() as u64 - back.reader().header().blob_base),
            v1.len() as u64
        );
        assert_eq!(
            new.header().blob_base - old.header().blob_base,
            HEADER_LEN_V2 - HEADER_LEN_V1
        );

        // A truncated v1 header is still rejected, and a v1 image truncated to
        // v1-header length (no directory) is a legal empty terrain, not a v2 read
        // that walks off the end.
        assert_eq!(
            TerrainAsset::from_bytes(v1[..40].to_vec()).unwrap_err(),
            TerrainAssetError::TooShort
        );
    }

    /// A payload claiming a **future** schema is refused rather than mis-parsed.
    #[test]
    fn a_newer_schema_is_refused() {
        let mut bytes = sample_asset().into_bytes();
        bytes[8..12].copy_from_slice(&(TERRAIN_ASSET_SCHEMA_VERSION + 1).to_le_bytes());
        assert_eq!(
            TerrainAsset::from_bytes(bytes).unwrap_err(),
            TerrainAssetError::SchemaTooNew {
                found: TERRAIN_ASSET_SCHEMA_VERSION + 1,
                current: TERRAIN_ASSET_SCHEMA_VERSION,
            }
        );
    }

    #[test]
    fn header_round_trips_and_counts_levels() {
        let t = sample_terrain();
        let p = build_pyramid(&t, PyramidOptions::default());
        let asset = build_terrain_asset(&t, &p, PyramidOptions::default()).unwrap();
        let r = asset.reader();
        assert_eq!(r.header().schema_version, TERRAIN_ASSET_SCHEMA_VERSION);
        assert_eq!(r.tile_resolution(), 5);
        assert_eq!(r.meters_per_sample(), 2.0);
        assert_eq!(r.origin(), DVec3::ZERO);
        assert_eq!(r.lod_levels(), 1 + p.len() as u32);
        assert_eq!(
            r.tile_count(),
            t.tile_count() + p.iter().map(|l| l.tiles.len()).sum::<usize>()
        );
        assert_eq!(r.meters_per_sample_at(0), 2.0);
        assert_eq!(r.meters_per_sample_at(2), 8.0);
    }

    #[test]
    fn every_tile_round_trips_bit_identically() {
        let t = sample_terrain();
        let p = build_pyramid(&t, PyramidOptions::default());
        let asset = build_terrain_asset(&t, &p, PyramidOptions::default()).unwrap();
        let r = asset.reader();
        for (&coord, tile) in t.tiles() {
            let back = r.tile(TileKey::lod0(coord)).unwrap().expect("lod0 present");
            assert_eq!(&back, tile, "level-0 tile {coord:?}");
            // Bytes, not just values: the blob is the tile's canonical bincode.
            assert_eq!(
                r.tile_bytes(TileKey::lod0(coord)).unwrap(),
                encode_tile(tile).unwrap()
            );
        }
        for level in &p {
            for (&coord, tile) in &level.tiles {
                let key = TileKey::new(level.lod, coord);
                assert_eq!(&r.tile(key).unwrap().unwrap(), tile, "lod {}", level.lod);
            }
        }
        assert!(r.tile(TileKey::lod0((99, 99))).unwrap().is_none());
        assert!(r.tile_bytes(TileKey::new(99, (0, 0))).is_none());
    }

    #[test]
    fn every_tile_blob_is_16_byte_aligned_inside_the_payload() {
        let asset = sample_asset();
        let r = asset.reader();
        assert!(!r.is_empty());
        for e in r.directory() {
            assert_eq!(e.offset % TILE_ALIGN, 0, "{:?} misaligned", e.key);
            assert!(e.offset >= r.header().blob_base);
            assert!(e.offset + e.len <= asset.as_bytes().len() as u64);
        }
        // The header + directory are themselves multiples of the alignment, so the
        // blob section needs no leading padding at all.
        assert_eq!(HEADER_LEN % TILE_ALIGN, 0);
        assert_eq!(DIR_ENTRY_LEN % TILE_ALIGN, 0);
        assert_eq!(
            r.header().blob_base,
            HEADER_LEN + DIR_ENTRY_LEN * r.tile_count() as u64
        );
        // And the payload as a whole ends on a boundary.
        assert_eq!(asset.as_bytes().len() as u64 % TILE_ALIGN, 0);
    }

    #[test]
    fn rebuilds_are_byte_identical() {
        let a = sample_asset();
        let b = sample_asset();
        assert_eq!(a.as_bytes(), b.as_bytes(), "the build is a pure function");

        // Insertion order cannot leak into the bytes (BTreeMap-ordered directory).
        let t = sample_terrain();
        let p = build_pyramid(&t, PyramidOptions::default());
        let mut fwd = TerrainAssetBuilder::new(5, 2.0);
        let mut rev = TerrainAssetBuilder::new(5, 2.0);
        let mut keys: Vec<(TileKey, &TerrainTile)> = t
            .tiles()
            .map(|(&c, tile)| (TileKey::lod0(c), tile))
            .chain(
                p.iter()
                    .flat_map(|l| l.tiles.iter().map(|(&c, ti)| (TileKey::new(l.lod, c), ti))),
            )
            .collect();
        for (k, tile) in &keys {
            fwd.insert(*k, tile).unwrap();
        }
        keys.reverse();
        for (k, tile) in &keys {
            rev.insert(*k, tile).unwrap();
        }
        assert_eq!(fwd.build().unwrap(), rev.build().unwrap());
    }

    #[test]
    fn directory_is_lod_major_and_sorted() {
        let asset = sample_asset();
        let r = asset.reader();
        let keys: Vec<TileKey> = r.keys().collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]), "strictly ascending");
        // A whole LOD level is one contiguous run (the clipmap-ring locality claim).
        let lods: Vec<u32> = keys.iter().map(|k| k.lod).collect();
        assert!(lods.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn to_terrain_data_recovers_the_authored_level() {
        let t = sample_terrain();
        let p = build_pyramid(&t, PyramidOptions::default());
        let asset = build_terrain_asset(&t, &p, PyramidOptions::default()).unwrap();
        let mut back = asset.reader().to_terrain_data().unwrap();
        assert_eq!(back, t, "level-0 tiles rebuild the authored terrain");
        assert_eq!(back.tile_count(), t.tile_count());
        assert!(back.drain_dirty().is_empty(), "a fresh load is not dirty");
    }

    #[test]
    fn empty_terrain_builds_a_valid_header_only_payload() {
        let t = TerrainData::new(8, 1.0);
        let asset = build_terrain_asset(&t, &[], PyramidOptions::default()).unwrap();
        let r = asset.reader();
        assert_eq!(r.tile_count(), 0);
        assert_eq!(r.lod_levels(), 1);
        assert_eq!(asset.as_bytes().len() as u64, HEADER_LEN);
    }

    #[test]
    fn duplicate_tiles_are_rejected() {
        let mut b = TerrainAssetBuilder::new(4, 1.0);
        let tile = TerrainTile::flat(4, DVec3::ZERO);
        b.insert(TileKey::lod0((0, 0)), &tile).unwrap();
        assert_eq!(
            b.insert(TileKey::lod0((0, 0)), &tile),
            Err(TerrainAssetError::DuplicateTile {
                lod: 0,
                tx: 0,
                tz: 0
            })
        );
    }

    #[test]
    fn corrupt_payloads_are_rejected_not_trusted() {
        let asset = sample_asset();
        assert_eq!(
            TerrainAsset::from_bytes(vec![0u8; 8]).unwrap_err(),
            TerrainAssetError::TooShort
        );
        let mut bad = asset.as_bytes().to_vec();
        bad[0] = b'X';
        assert_eq!(
            TerrainAsset::from_bytes(bad).unwrap_err(),
            TerrainAssetError::BadMagic
        );
        // A future schema is refused rather than misread.
        let mut future = asset.as_bytes().to_vec();
        future[8..12].copy_from_slice(&(TERRAIN_ASSET_SCHEMA_VERSION + 7).to_le_bytes());
        assert!(matches!(
            TerrainAsset::from_bytes(future).unwrap_err(),
            TerrainAssetError::SchemaTooNew { .. }
        ));
        // A blob pointing past the end.
        let mut oob = asset.as_bytes().to_vec();
        oob[HEADER_LEN as usize + 16..HEADER_LEN as usize + 24]
            .copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            TerrainAsset::from_bytes(oob).unwrap_err(),
            TerrainAssetError::Malformed(_)
        ));
        // A misaligned blob offset.
        let mut mis = asset.as_bytes().to_vec();
        let off = u64::from_le_bytes(
            mis[HEADER_LEN as usize + 16..HEADER_LEN as usize + 24]
                .try_into()
                .unwrap(),
        );
        mis[HEADER_LEN as usize + 16..HEADER_LEN as usize + 24]
            .copy_from_slice(&(off + 1).to_le_bytes());
        assert!(matches!(
            TerrainAsset::from_bytes(mis).unwrap_err(),
            TerrainAssetError::Malformed(_)
        ));
        // Truncation inside the directory.
        assert!(matches!(
            TerrainAsset::from_bytes(asset.as_bytes()[..HEADER_LEN as usize + 8].to_vec())
                .unwrap_err(),
            TerrainAssetError::Malformed(_)
        ));

        // Two directory entries sharing bytes: one tile's blob would alias
        // another's, so a reader can never see them as independent pages.
        let r = asset.reader();
        assert!(r.tile_count() >= 2, "need two tiles to overlap them");
        let (first, second) = (r.directory()[0], r.directory()[1]);
        let mut overlap = asset.as_bytes().to_vec();
        let len_at = HEADER_LEN as usize + 24;
        overlap[len_at..len_at + 8]
            .copy_from_slice(&(second.offset - first.offset + 1).to_le_bytes());
        let err = TerrainAsset::from_bytes(overlap).unwrap_err();
        assert!(
            matches!(&err, TerrainAssetError::Malformed(m) if m.contains("overlaps")),
            "expected an overlap rejection, got {err}"
        );
    }

    /// The **fail-loud divergence** regression (P16.3 audit).
    ///
    /// A `.inf_terrain` is the raw image; a generic `inf_asset::encode` would frame
    /// it with a bincode length prefix and knock every tile off its 16-byte
    /// boundary. That door is closed at compile time — `TerrainAsset` implements
    /// neither `AssetPayload` nor `Serialize`/`Deserialize` — but a future refactor
    /// could re-open it, so both directions are pinned here as *loud* failures
    /// rather than quiet wrong answers.
    #[test]
    fn framed_and_raw_forms_are_not_interchangeable() {
        let asset = sample_asset();
        let image = asset.as_bytes();

        // Direction 1 — framed bytes fed to the reader: rejected outright, never
        // misread as a payload that happens to start with a length varint.
        let framed = bincode::serde::encode_to_vec(image, bincode::config::standard()).unwrap();
        assert_ne!(framed, image, "framing really does change the bytes");
        assert_eq!(
            TerrainAssetReader::new(framed.as_slice()).unwrap_err(),
            TerrainAssetError::BadMagic,
            "a framed payload must fail the magic check, not be parsed"
        );

        // Direction 2 — the raw image fed to a generic bincode decode: it does NOT
        // reproduce the image (it reads the magic's first byte as a length), which
        // is exactly the silent corruption the closed door prevents.
        let generic: std::result::Result<(Vec<u8>, usize), _> =
            bincode::serde::decode_from_slice(image, bincode::config::standard());
        match generic {
            Err(_) => {}
            Ok((decoded, _)) => assert_ne!(
                decoded, image,
                "a generic decode must never round-trip the raw image"
            ),
        }

        // The sanctioned path round-trips exactly, and publishes the kind/version
        // the database needs without a generic serde impl.
        assert_eq!(TerrainAsset::KIND, AssetKind::Terrain);
        assert_eq!(TerrainAsset::SCHEMA_VERSION, TERRAIN_ASSET_SCHEMA_VERSION);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/World.inf_terrain");
        let written = write_terrain_asset(&path, &asset).unwrap();
        assert_eq!(written, image, "the writer writes the image verbatim");
        assert_eq!(std::fs::read(&path).unwrap(), image);
        assert_eq!(read_terrain_asset(&path).unwrap().as_bytes(), image);
        // No temp litter beside the asset.
        let strays: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
            .collect();
        assert!(
            strays.is_empty(),
            "atomic write left temp files: {strays:?}"
        );

        // A corrupt file surfaces as an IO error, not a panic or a silent empty.
        std::fs::write(&path, b"not a terrain asset at all").unwrap();
        assert!(read_terrain_asset(&path).is_err());
    }
}
