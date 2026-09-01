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
//! │  reserved      [u8; 56]  zeros (room for v6 without a re-length)  │
//! ├ tile directory (tile_count × 32 B, sorted by TileKey) ────────────┤
//! │  lod u32 · tx i32 · tz i32 · codec u8 + rsv[3] · offset u64 ·     │
//! │  stored_len u64                                                   │
//! ├ blob section (each tile's stored bytes, 16-byte-aligned) ─────────┤
//! │  … tile blobs, zero-padded up to each 16-byte boundary …          │
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Schema v6: per-tile compression (IASSET1)
//!
//! v6 changes the **container**, not the tile blob — the mirror image of v3/v4/v5,
//! which changed the blob and not the container. The directory entry's
//! always-zero `reserved u32` becomes a [`BlockCodec`] byte plus three zero bytes,
//! `len` keeps meaning *stored* bytes, and a tile whose codec is not
//! [`BlockCodec::Raw`] holds an
//! [`inf_asset::block`](inf_asset::block)-framed blob instead of bare bincode.
//! Reading is unchanged in shape: binary-search the directory, take the blob,
//! [`decode_block`] it (a borrow when raw), `bincode`-decode the result.
//!
//! **Terrain tiles were never cast in place**, which is the fact that makes this
//! legal where it is not legal for `.inf_vmesh` or `.inf_tex`. Every page-in has
//! always run the blob through `bincode` into a [`TerrainTile`], so the borrowed
//! slice was an input to a decoder, never a `bytemuck` cast — a decompress in
//! front of that decode is an addition to an existing cost, and the "zero-copy
//! streaming" the raw layout bought was zero-*copy*, not zero-*work*.
//!
//! **The downgrade story.** A v6 payload written with [`BlockCodec::Raw`] on every
//! tile is byte-identical to the v5 image of the same tiles **except for the four
//! bytes of `schema_version`** (the codec byte lands where v5's reserved zero
//! already was) — `v6_all_raw_is_v5_plus_four_bytes` asserts exactly that. A v1–v5
//! payload keeps loading for ever, because the parser refuses a non-zero
//! `reserved` on those versions rather than reading it as a codec: an old asset
//! has no codec byte, and no old asset can be mistaken for one that does. There is
//! no reverse migration and none is needed — the loose editor asset is what an
//! author edits, and the cook rebuilds the compressed one from it every time.
//!
//! **Where compression is applied.** The editor's writer keeps emitting
//! [`BlockCodec::Raw`]: a loose `.inf_terrain` is being authored, its tiles are
//! rewritten on every sculpt write-back, and compressing them would tax the
//! editor to shrink a file nobody ships. The **cook** transcodes with
//! [`recompress_terrain_asset`] on the way into the pack. So the editor and the
//! shipped game hold two different images of one terrain that decode to
//! bit-identical tiles — which is what keeps PIE (reading the loose asset by path)
//! and shipping (reading the cooked one) simulating the same world.
//!
//! # Schema v3 / v4 / v5: tile blobs grow layers (P19.1, P19.2, P21.2)
//!
//! The header has not changed since v2 — v3, v4 and v5 are byte-for-byte v2's
//! 128-byte layout — but the *tile blob* has, three times: v3 appended the sparse
//! erosion [`maps`](TerrainTile::maps) layer (flow / deposition / wear, P19.1), v4
//! appended the sparse [`biomes`](TerrainTile::biomes) layer (per-sample biome
//! ids, P19.2), and v5 appended the sparse [`holes`](TerrainTile::holes) layer
//! (the packed per-sample hole mask, P21.2). bincode is **positional**, so each is
//! a wire-format change even though the new field is `#[serde(default)]`: a v2
//! blob has three fields where this build reads six, and decoding it as the grown
//! tile would run off the end of the blob.
//!
//! The version is therefore what selects the wire type ([`decode_tile_at`]): a
//! payload stamped **≤ 2** decodes through the frozen [`TerrainTileFrozenV1`], a
//! **v3** one through [`TerrainTileFrozenV2`], a **v4** one through
//! [`TerrainTileFrozenV3`], and a **v5** one as the current tile — each lifting
//! the layers it cannot carry to their sparse defaults, which is exactly what a
//! payload of that vintage meant. So **v1..v4 payloads load forever**, and the
//! only thing a rewrite costs an untouched terrain is one zero-length count per
//! layer per tile blob. The header length still comes from [`header_len`] alone,
//! which is why v3, v4 and v5 could reuse v2's without ambiguity: the *schema
//! version* remains the single source of truth for how to read everything after
//! the magic.
//!
//! **This is the container that carries holes.** The other one — `.inf_lvl`, via
//! `TerrainData`'s wire form — is pinned at tile generation 3 because the scene
//! schema was frozen at v19; see the *THE EMPTY CELL* section of
//! [`TerrainTileFrozenV1`]'s generation table. So a terrain that has been carved
//! wants a `.inf_terrain` asset behind it — and a carved terrain that has none
//! reaches no cook at all to be warned about, because the mask was already gone
//! when the level was written. The cook's P21.2 advisory (`see_through_pits`)
//! therefore reads *this* container, and reports the hazard that does survive:
//! holes with no voxel volume behind them.
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
//! The bytes on disk — and in a `.ipack` — are the **raw image**
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

use std::borrow::Cow;
use std::collections::BTreeMap;

use glam::DVec3;
use inf_asset::block::{decode_block, encode_block, BlockCodec};
use inf_asset::AssetKind;

use crate::data::TerrainData;
use crate::pyramid::{PyramidLevel, PyramidOptions};
use crate::tile::{
    TerrainTile, TerrainTileFrozenV1, TerrainTileFrozenV2, TerrainTileFrozenV3, TileKey,
};

/// Magic at the head of every `.inf_terrain` payload.
pub const TERRAIN_ASSET_MAGIC: [u8; 8] = *b"INFTERRN";

/// Current `.inf_terrain` payload schema version (**6** since IASSET1 — the
/// directory carries a per-tile [`BlockCodec`]; **5** since P21.2 — tile blobs
/// carry the packed per-sample hole mask; **4** since P19.2 — the biome ids;
/// **3** since P19.1 — the erosion data maps; **2** since P16.6 — the header
/// records the pyramid options; see the module docs).
pub const TERRAIN_ASSET_SCHEMA_VERSION: u32 = 6;

/// The first schema version whose directory entries carry a codec byte.
///
/// Below it byte 12 of an entry is v1's `reserved u32` and **must be zero** —
/// which is what makes "a v5 asset has no codec" a checked fact rather than an
/// assumption (see [`parse`]).
pub const FIRST_CODEC_SCHEMA_VERSION: u32 = 6;

/// Tile blobs start on multiples of this many bytes (see the module docs).
pub const TILE_ALIGN: u64 = 16;

/// **The codec the cook transcodes `.inf_terrain` tiles to** (IASSET1), chosen
/// by measurement over the real island's 1 064 DEM tiles at 257².
///
/// The bake-off is `tests/block_codec_bakeoff.rs`; the full table is in
/// `docs/memos/iasset1-block-compression.md`. Measured (Windows, test profile
/// with optimizations), whole 549 879 456 B asset:
///
/// | codec | ship ratio | lod-0 decode | 16 serialized | encode (whole) |
/// |---|---|---|---|---|
/// | lz4 | 0.4442 | 0.099 ms | 1.59 ms | 0.4 s |
/// | deflate | 0.3567 | 0.749 ms | **11.99 ms** | 8.8 s |
/// | **zstd** | **0.3505** | **0.168 ms** | **2.69 ms** | 3.3 s |
///
/// **zstd wins on every axis at once**, which is not the answer the arc brief
/// expected and is why it was measured: it has the best ratio *and* is 4.5×
/// faster to decode than DEFLATE *and* 2.7× faster to encode, *and* it is the
/// codec already in the tree. DEFLATE is the one that fails — 16 level-0 tiles
/// decompressed serially is 11.99 ms against
/// `inf_player::budget::STREAMED_STEP_BUDGET_MS`'s 4.0.
///
/// **The one cost, measured rather than assumed**: `Zstd` is the C `zstd`
/// natively and the pure-Rust `ruzstd` in a browser, and ruzstd decodes the same
/// tile in **1.224 ms — 7.3× slower**. `lz4_flex` and `miniz_oxide` are one
/// implementation on every target and have no such split. A web-targeted cook
/// should therefore pass `BlockCodec::Lz4`, which is why this is a *default* and
/// `CookOptions::terrain_codec` exists.
///
/// **Every codec in [`BlockCodec`] must stay decodable** whatever this constant
/// says: a pack cooked with the knob turned is opened by the same reader.
pub const COOK_TILE_CODEC: BlockCodec = BlockCodec::Zstd;

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

/// Bytes of the **v5** fixed header. P21.2's hole mask is, once again, a *tile
/// blob* change only — and v2 left 56 reserved bytes precisely so a version that
/// needs no new header field costs no re-length. Same reasoning as
/// [`HEADER_LEN_V3`]; `v5_needed_no_header_re_length` is the check (it was cited
/// here before it was written — the P21.2 audit wrote it).
pub const HEADER_LEN_V5: u64 = HEADER_LEN_V2;

/// Bytes of the **v6** fixed header. IASSET1's per-tile codec lives in the
/// *directory entry*, in bytes v1 already reserved, so once again the header does
/// not move — same reasoning as [`HEADER_LEN_V3`].
pub const HEADER_LEN_V6: u64 = HEADER_LEN_V2;

/// Bytes of the fixed header **this build writes** (the current schema's).
pub const HEADER_LEN: u64 = HEADER_LEN_V6;

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
        4 => HEADER_LEN_V4,
        5 => HEADER_LEN_V5,
        _ => HEADER_LEN_V6,
    }
}

/// The largest `tile_resolution` a `.inf_terrain` may declare (IASSET1 audit).
///
/// **This exists because the block ceiling is derived from a header field.**
/// [`tile_raw_ceiling`] turns `tile_resolution` into the bound
/// [`decode_block`] refuses a length claim against — and `tile_resolution` is
/// read out of the same doctored file the claim came from. With only the old
/// `>= 2` check, a header declaring `u32::MAX` samples per side mints a ceiling
/// of `usize::MAX`, and "bound the claim before it becomes an allocation" bounds
/// nothing: a forty-byte block asks for a terabyte and gets it.
///
/// 2049 is `2^11 + 1` — the shape a terrain tile takes — and **8× the largest
/// resolution this engine has ever authored** (the island's 257; the tree's
/// fixtures top out there). One tile at 2049² is 16.8 MB of `f32` heights before
/// any other layer, which is already far past what a 4 ms streaming step can
/// page; the bound refuses files nothing here can produce and nothing sane would
/// want, and in exchange the worst-case decompression allocation is a knowable
/// ~256 MiB instead of unbounded.
///
/// Enforced in **both** directions, on the P23.6 rule that a writer must not
/// manufacture a file its own reader rejects: [`parse`] refuses a payload above
/// it and [`TerrainAssetBuilder::build`] refuses to write one.
pub const MAX_TILE_RESOLUTION: u32 = 2049;

/// A ceiling on how many bytes one tile blob can decompress to, at
/// `tile_resolution` samples per side.
///
/// A compressed block's declared length is written by whoever made the file, so
/// it is a claim that has to be bounded **before** it becomes an allocation
/// ([`decode_block`]). This is the bound, and it is a property of the container
/// rather than a guess: a [`TerrainTile`] carries at most one `f32` height, one
/// RGBA weight, three `f32` data-map channels, one biome id and one hole bit per
/// sample — ~21.2 bytes — so 64 gives better than 3× headroom for `bincode`'s
/// framing and any layer a future schema appends, while still refusing the
/// gigabyte claim a doctored asset makes.
///
/// **The resolution is clamped to [`MAX_TILE_RESOLUTION`] first**, so this
/// function is safe to call with a number that has not been validated — a
/// ceiling derived from an attacker's header is the one way a ceiling stops
/// being one. A payload past the cap never reaches here anyway ([`parse`]
/// refuses it); the clamp is the belt to that braces.
#[inline]
pub const fn tile_raw_ceiling(tile_resolution: u32) -> usize {
    /// Generous per-sample bound; see the doc comment for the real ~21.2.
    const BYTES_PER_SAMPLE: usize = 64;
    /// Slack for the origin, the five length prefixes and alignment.
    const SLACK: usize = 64 * 1024;
    let res = if tile_resolution > MAX_TILE_RESOLUTION {
        MAX_TILE_RESOLUTION
    } else {
        tile_resolution
    } as usize;
    res.saturating_mul(res)
        .saturating_mul(BYTES_PER_SAMPLE)
        .saturating_add(SLACK)
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
/// | ≤ 2 | [`TerrainTileFrozenV1`] — origin + heights + weights, lifted with empty maps, biomes and holes |
/// | 3 | [`TerrainTileFrozenV2`] — the above plus the P19.1 data maps, lifted with empty biomes and holes |
/// | 4 | [`TerrainTileFrozenV3`] — the above plus the P19.2 biome ids, lifted with an empty hole mask |
/// | ≥ 5 | [`TerrainTile`] — the above plus the sparse P21.2 hole mask |
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
    if schema_version == 4 {
        return bincode::serde::decode_from_slice::<TerrainTileFrozenV3, _>(bytes, tile_config())
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
    /// **Stored** blob length in bytes (unpadded) — the bytes on disk, which for
    /// a compressed tile is smaller than what it decodes to.
    pub len: u64,
    /// How the blob is stored (schema v6+; always [`BlockCodec::Raw`] below it).
    pub codec: BlockCodec,
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
    codec: BlockCodec,
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
            // RAW by default, deliberately: this builder writes the LOOSE asset
            // an author edits, whose tiles are rewritten on every sculpt
            // write-back. Compression belongs to the cook
            // ([`recompress_terrain_asset`]), which runs once per ship.
            codec: BlockCodec::Raw,
            tiles: BTreeMap::new(),
        }
    }

    /// Compress every staged tile under `codec` (IASSET1).
    ///
    /// Per-tile and independent: a tile that would inflate is stored
    /// [`Raw`](BlockCodec::Raw) and its directory entry says so, so the payload
    /// can never be larger than the uncompressed one. The default is
    /// [`Raw`](BlockCodec::Raw) — see [`new`](Self::new).
    pub fn with_codec(mut self, codec: BlockCodec) -> Self {
        self.codec = codec;
        self
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
        // The P23.6 rule: a writer must not manufacture a file its own reader
        // rejects. `parse` refuses a resolution past the cap because the block
        // ceiling is derived from it (see `MAX_TILE_RESOLUTION`), so this refuses
        // it here rather than writing a payload nothing can open.
        if self.tile_resolution > MAX_TILE_RESOLUTION {
            return Err(TerrainAssetError::Malformed(format!(
                "tile_resolution {} is past the {MAX_TILE_RESOLUTION} maximum a \
                 `.inf_terrain` may declare",
                self.tile_resolution
            )));
        }
        let count = u32::try_from(self.tiles.len())
            .map_err(|_| TerrainAssetError::Malformed("more than u32::MAX tiles".into()))?;
        let lod_levels = self
            .tiles
            .keys()
            .map(|k| k.lod + 1)
            .max()
            .unwrap_or(1)
            .max(1);

        // Compress each tile independently (IASSET1). `encode_block` reports the
        // codec it actually used, which is `Raw` for a tile compression would
        // inflate — so the directory always states what happened, and the payload
        // can never be larger than the uncompressed one would have been.
        let stored: Vec<(BlockCodec, Vec<u8>)> = self
            .tiles
            .values()
            .map(|blob| encode_block(self.codec, blob))
            .collect::<std::result::Result<_, _>>()
            .map_err(|e: inf_asset::AssetError| {
                TerrainAssetError::Malformed(format!("tile compression: {e}"))
            })?;

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
        for (_, blob) in &stored {
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

        for ((key, (codec, blob)), &off) in self.tiles.keys().zip(&stored).zip(&offsets) {
            out.extend_from_slice(&key.lod.to_le_bytes());
            out.extend_from_slice(&key.coord.0.to_le_bytes());
            out.extend_from_slice(&key.coord.1.to_le_bytes());
            // v6: the codec byte, then three bytes v1 reserved and every version
            // since has written as zero. A raw tile writes 0 here, which is what
            // makes an all-raw v6 image byte-identical to the v5 one.
            out.push(codec.code());
            out.extend_from_slice(&[0u8; 3]);
            out.extend_from_slice(&off.to_le_bytes());
            out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        }
        debug_assert_eq!(out.len() as u64, HEADER_LEN + DIR_ENTRY_LEN * count as u64);

        // Blob section, zero-padded up to each offset (and to the end, so the
        // payload length is itself a multiple of TILE_ALIGN).
        for ((_, blob), &off) in stored.iter().zip(&offsets) {
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
/// The mechanism comes from [`inf_asset::write_atomically`] rather than being
/// re-spelled here, which also gives this door the *counter* half of the Spike C
/// temp-name law: the name used to be pid-only, so two threads of one process
/// writing two terrains at once could pick one temp path and interleave into
/// each other's rename.
///
/// Returns the bytes written, so a caller can hash them into the sidecar without
/// re-reading the file.
pub fn write_terrain_asset<'a>(
    path: &std::path::Path,
    asset: &'a TerrainAsset,
) -> std::io::Result<&'a [u8]> {
    let bytes = asset.as_bytes();
    inf_asset::write_atomically(path, bytes)?;
    Ok(bytes)
}

/// What one [`recompress_terrain_asset`] pass did to a payload — the cook's
/// per-asset row in the before/after table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RecompressReport {
    /// Tiles in the payload.
    pub tiles: usize,
    /// Tiles that actually ended up compressed (the rest inflated and stayed
    /// [`Raw`](BlockCodec::Raw)).
    pub tiles_compressed: usize,
    /// Payload bytes before.
    pub bytes_before: u64,
    /// Payload bytes after.
    pub bytes_after: u64,
}

impl RecompressReport {
    /// `after / before` — 1.0 when nothing was won.
    pub fn ratio(&self) -> f64 {
        if self.bytes_before == 0 {
            1.0
        } else {
            self.bytes_after as f64 / self.bytes_before as f64
        }
    }
}

/// **Transcode a `.inf_terrain` payload to per-tile `codec`** — the cook's
/// compile step for a streaming terrain (IASSET1).
///
/// Rebuilds the container at the current schema with every tile's *stored* bytes
/// re-framed under `codec`. Losslessness is by construction rather than by
/// assertion: the tile blobs are carried through as **opaque bytes**, never
/// decoded and re-encoded, so this cannot change a height, a weight or a hole
/// even if `TerrainTile`'s wire form were wrong. What it may change is the
/// schema stamp — a v1–v5 payload is lifted to v6, which is legal precisely
/// because the *tile blob* layout is untouched and the version is what selects
/// the blob's wire type ([`decode_tile_at`]).
///
/// Passing [`BlockCodec::Raw`] is the inverse transcode, and it is not a no-op
/// worth removing: it is how a test proves the round trip, and how a build that
/// wants the old ship size gets it back with one option.
///
/// # The one thing this is NOT bit-preserving about, and why it is safe
///
/// A **v1** payload did not record its [`PyramidOptions`], and
/// [`TerrainAssetHeader::pyramid`] reports that as `None` — *unknown*, which the
/// P16.6 docs are emphatic is not the same as "the defaults". Lifting such a
/// payload to v6 has nowhere to put "unknown", so the rebuilt header carries
/// [`PyramidOptions::default`] and the distinction is lost.
///
/// That is harmless **here** and would not be elsewhere, so the reason is worth
/// stating rather than assuming. The only consumer of the None/Some distinction
/// is the editor's write-back re-plan (`inf_editor_core::terrain_edit`), which
/// reads the **loose** `.inf_terrain` an author edits — a file this function
/// never touches. Nothing reads `pyramid_options` out of a cooked pack. If
/// something ever does, `a_v1_payload_transcodes_forward_without_decoding_its_tiles`
/// pins the behaviour it would be reading.
pub fn recompress_terrain_asset(
    payload: &[u8],
    codec: BlockCodec,
) -> Result<(Vec<u8>, RecompressReport)> {
    let reader = TerrainAssetReader::new(payload)?;
    let mut b = TerrainAssetBuilder::new(reader.tile_resolution(), reader.meters_per_sample())
        .with_origin(reader.origin())
        .with_codec(codec);
    if let Some(p) = reader.pyramid_options() {
        b = b.with_pyramid(p);
    }
    for e in reader.directory() {
        // The blob, decompressed if it already was compressed and borrowed if it
        // was not — so a re-cook of an already-compressed pack transcodes rather
        // than double-framing.
        let blob = reader
            .tile_bytes(e.key)
            .ok_or_else(|| TerrainAssetError::TileDecode {
                lod: e.key.lod,
                tx: e.key.coord.0,
                tz: e.key.coord.1,
                message: "stored block did not decode".into(),
            })?;
        b.insert_bytes(e.key, blob.into_owned())?;
    }
    let asset = b.build()?;
    let after = TerrainAssetReader::new(asset.as_bytes())?;
    let report = RecompressReport {
        tiles: after.tile_count(),
        tiles_compressed: after
            .directory()
            .iter()
            .filter(|e| e.codec != BlockCodec::Raw)
            .count(),
        bytes_before: payload.len() as u64,
        bytes_after: asset.as_bytes().len() as u64,
    };
    Ok((asset.into_bytes(), report))
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

    /// A tile's **stored** bytes, exactly as they sit in the payload — borrowed,
    /// 16-byte aligned within it, and still compressed if the directory says so.
    ///
    /// This is the re-pack seam ([`recompress_terrain_asset`], a write-back that
    /// carries a tile through untouched) and the seam a size report measures. A
    /// consumer that wants the *tile* wants [`tile_bytes`](Self::tile_bytes).
    pub fn stored_bytes(&self, key: TileKey) -> Option<&[u8]> {
        let e = self.entry(key)?;
        // Bounds + alignment were validated in `new`.
        Some(&self.bytes.as_ref()[e.offset as usize..(e.offset + e.len) as usize])
    }

    /// A tile's **canonical bincode blob** — borrowed straight out of the payload
    /// for a [`Raw`](BlockCodec::Raw) tile, decompressed into an owned buffer for
    /// a compressed one (schema v6, IASSET1).
    ///
    /// The `Cow` is the whole IASSET1 seam in one return type: a raw container
    /// keeps the zero-copy read it has had since P16.3, and a compressed one pays
    /// exactly one allocation + one decompress for the one tile that was asked
    /// for. `None` for an absent tile; `None` for a corrupt block too — a
    /// directory that lies is an absent tile, never silent geometry (the caller
    /// that wants the reason uses [`tile`](Self::tile)).
    pub fn tile_bytes(&self, key: TileKey) -> Option<Cow<'_, [u8]>> {
        let e = self.entry(key)?;
        let stored = &self.bytes.as_ref()[e.offset as usize..(e.offset + e.len) as usize];
        decode_block(
            e.codec,
            stored,
            tile_raw_ceiling(self.header.tile_resolution),
        )
        .ok()
    }

    /// Decode a tile, through **this payload's own** schema version — so a v1/v2
    /// asset's tiles come back with empty data maps rather than a decode error.
    pub fn tile(&self, key: TileKey) -> Result<Option<TerrainTile>> {
        let Some(e) = self.entry(key) else {
            return Ok(None);
        };
        let stored = &self.bytes.as_ref()[e.offset as usize..(e.offset + e.len) as usize];
        let bytes = decode_block(
            e.codec,
            stored,
            tile_raw_ceiling(self.header.tile_resolution),
        )
        .map_err(|message| TerrainAssetError::TileDecode {
            lod: key.lod,
            tx: key.coord.0,
            tz: key.coord.1,
            message: message.to_string(),
        })?;
        decode_tile_at(&bytes, self.header.schema_version)
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
    // The upper bound is not cosmetic: `tile_raw_ceiling` derives the bound a
    // compressed block's length CLAIM is refused against from this very field, so
    // an unbounded resolution mints an unbounded ceiling and the claim is not
    // bounded at all. See `MAX_TILE_RESOLUTION`.
    if tile_resolution > MAX_TILE_RESOLUTION {
        return Err(TerrainAssetError::Malformed(format!(
            "tile_resolution {tile_resolution} is past the {MAX_TILE_RESOLUTION} \
             maximum — a header that names its own block ceiling is not a ceiling"
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
        // Bytes 12..16: v1's `reserved u32`, reinterpreted at v6 as a codec byte
        // plus three still-reserved zeros. Below v6 the WHOLE word must be zero —
        // an old asset has no codec, so this refuses to *read* one out of bytes
        // that never meant anything, which is what makes "v1–v5 load for ever"
        // a checked fact instead of a hope.
        let codec = if schema_version >= FIRST_CODEC_SCHEMA_VERSION {
            if data[base + 13..base + 16] != [0, 0, 0] {
                return Err(TerrainAssetError::Malformed(
                    "tile directory entry has a non-zero reserved tail".into(),
                ));
            }
            BlockCodec::from_code(data[base + 12]).ok_or_else(|| {
                TerrainAssetError::Malformed(format!(
                    "tile block codec {} is not one this build knows",
                    data[base + 12]
                ))
            })?
        } else {
            if data[base + 12..base + 16] != [0, 0, 0, 0] {
                return Err(TerrainAssetError::Malformed(format!(
                    "a schema v{schema_version} tile directory entry has a non-zero \
                     reserved word (v{FIRST_CODEC_SCHEMA_VERSION} is the first with a codec)"
                )));
            }
            BlockCodec::Raw
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
        dir.push(TileDirEntry {
            key,
            offset,
            len,
            codec,
        });
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
            out.extend_from_slice(&r.tile_bytes(e.key).unwrap());
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

    /// **The v5 and v6 header claims, actually checked** (P21.2 audit; extended
    /// at IASSET1). [`HEADER_LEN_V5`] says P21.2's hole mask cost no header
    /// re-length and cited this test by name; the test did not exist, and a cited
    /// gate that does not exist is worse than no claim. [`HEADER_LEN_V6`] makes
    /// the same claim for a *directory* change, and it is checked here for the
    /// same reason.
    ///
    /// What it pins is what the claims actually rest on: `header_len(5)` and
    /// `header_len(6)` both resolve to the v2 length, this build writes v6 and
    /// writes that length, and the reserved tail is **still all zeros** — the 56
    /// bytes v2 set aside are unclaimed, so the next version that genuinely needs
    /// a header field can take them and be told apart from every version that did
    /// not.
    #[test]
    fn neither_v5_nor_v6_needed_a_header_re_length() {
        assert_eq!(TERRAIN_ASSET_SCHEMA_VERSION, 6, "this test is about v5/v6");
        assert_eq!(header_len(5), HEADER_LEN_V2);
        assert_eq!(header_len(6), HEADER_LEN_V2);
        assert_eq!(HEADER_LEN_V5, HEADER_LEN_V2);
        assert_eq!(HEADER_LEN_V6, HEADER_LEN_V2);
        assert_eq!(HEADER_LEN, HEADER_LEN_V6);
        // The catch-all arm answers with the current length. Not a forward-compat
        // promise — an unknown version is rejected before `header_len` is asked —
        // just the statement that v6 is the arm every future number lands in until
        // one of them claims the reserved tail and adds its own.
        assert_eq!(header_len(7), HEADER_LEN_V6);

        let t = sample_terrain();
        let opts = PyramidOptions::default();
        let asset = build_terrain_asset(&t, &build_pyramid(&t, opts), opts).unwrap();
        assert_eq!(asset.reader().header().schema_version, 6);
        assert!(
            asset.as_bytes()[72..HEADER_LEN_V6 as usize]
                .iter()
                .all(|&b| b == 0),
            "v6 spent part of the reserved tail — then it is not a tile-blob-only \
             change, and HEADER_LEN_V6's doc is wrong"
        );
        assert_eq!(
            asset.reader().header().blob_base,
            (HEADER_LEN_V5 + DIR_ENTRY_LEN * asset.reader().tile_count() as u64)
                .next_multiple_of(TILE_ALIGN),
            "the directory does not start where a 128-byte header would put it"
        );
    }

    // ── schema v6: per-tile block compression (IASSET1) ─────────────────────

    /// Re-stamp a v6 image's `schema_version` as `5`, leaving every other byte
    /// alone — the "what would v5 have written?" fixture for an ALL-RAW asset.
    fn restamp(image: &[u8], version: u32) -> Vec<u8> {
        let mut out = image.to_vec();
        out[8..12].copy_from_slice(&version.to_le_bytes());
        out
    }

    /// **The downgrade story, as bytes.** A v6 payload with every tile
    /// [`BlockCodec::Raw`] differs from the v5 image of the same tiles in exactly
    /// the four bytes of `schema_version` — because the codec byte lands where v5
    /// already wrote a reserved zero.
    ///
    /// That is what makes "the editor keeps writing what it wrote" a fact rather
    /// than a hope, and it is also the cheapest possible arm of the lossless
    /// claim: for the raw case there is nothing to be lossy *with*.
    #[test]
    fn v6_all_raw_is_v5_plus_four_bytes() {
        let asset = sample_asset(); // built with the default Raw codec
        let image = asset.as_bytes();
        assert_eq!(image[8..12], 6u32.to_le_bytes(), "this build writes v6");
        for e in asset.reader().directory() {
            assert_eq!(e.codec, BlockCodec::Raw, "the loose writer stays raw");
        }
        let as_v5 = restamp(image, 5);
        // Byte-identical apart from the version word, and it still parses as v5.
        assert_eq!(image[..8], as_v5[..8]);
        assert_eq!(image[12..], as_v5[12..]);
        let v5 = TerrainAssetReader::new(as_v5.as_slice()).unwrap();
        assert_eq!(v5.header().schema_version, 5);
        for e in asset.reader().directory() {
            assert_eq!(
                v5.tile_bytes(e.key).unwrap(),
                asset.reader().tile_bytes(e.key).unwrap()
            );
        }
    }

    /// A pre-v6 payload whose reserved word is **not** zero is refused, rather
    /// than having a codec read out of bytes that never meant one.
    ///
    /// This is what turns "v1–v5 load for ever" into a checked property: the only
    /// way an old asset could be misread as a compressed one is if the parser
    /// guessed, and it does not.
    #[test]
    fn a_pre_v6_reserved_word_must_be_zero() {
        let asset = sample_asset();
        let mut image = restamp(asset.as_bytes(), 5);
        let dir0 = HEADER_LEN_V2 as usize;
        image[dir0 + 12] = 1; // would be `Lz4` at v6
        let err = TerrainAssetReader::new(image.as_slice()).unwrap_err();
        assert!(
            matches!(&err, TerrainAssetError::Malformed(m) if m.contains("reserved word")),
            "{err}"
        );
    }

    /// A v6 payload naming a codec this build does not know is refused with the
    /// code in the message — never silently read as raw, which would hand the
    /// tile decoder a compressed frame and blame `bincode`.
    #[test]
    fn an_unknown_codec_is_refused_by_name() {
        let asset = sample_asset();
        let mut image = asset.as_bytes().to_vec();
        image[HEADER_LEN_V2 as usize + 12] = 200;
        let err = TerrainAssetReader::new(image.as_slice()).unwrap_err();
        assert!(
            matches!(&err, TerrainAssetError::Malformed(m) if m.contains("codec 200")),
            "{err}"
        );
    }

    /// **The lossless claim, per codec**: a transcode to any codec and back to
    /// raw reproduces the original image byte for byte, and every tile decodes to
    /// the same `TerrainTile` in between.
    #[test]
    fn recompress_round_trips_every_codec_byte_for_byte() {
        let asset = sample_asset();
        let raw_image = asset.as_bytes().to_vec();
        for codec in BlockCodec::ALL {
            let (packed, report) = recompress_terrain_asset(&raw_image, codec).unwrap();
            assert_eq!(report.tiles, asset.reader().tile_count());
            let r = TerrainAssetReader::new(packed.as_slice()).unwrap();
            assert_eq!(r.header().schema_version, TERRAIN_ASSET_SCHEMA_VERSION);
            assert_eq!(r.tile_count(), asset.reader().tile_count());
            for e in asset.reader().directory() {
                assert_eq!(
                    r.tile(e.key).unwrap(),
                    asset.reader().tile(e.key).unwrap(),
                    "{codec:?} changed tile {:?}",
                    e.key
                );
                assert_eq!(
                    r.tile_bytes(e.key).unwrap(),
                    asset.reader().tile_bytes(e.key).unwrap(),
                    "{codec:?} changed tile {:?}'s blob",
                    e.key
                );
            }
            // …and back. The inverse transcode is the byte-level arm.
            let (back, _) = recompress_terrain_asset(&packed, BlockCodec::Raw).unwrap();
            assert_eq!(back, raw_image, "{codec:?} did not round-trip to raw");
        }
    }

    /// A transcode is **deterministic**: same input, same codec, same bytes. The
    /// cook's byte-identical-rebuild gate reaches through this.
    #[test]
    fn recompress_is_byte_deterministic() {
        let asset = sample_asset();
        for codec in BlockCodec::ALL {
            let (a, _) = recompress_terrain_asset(asset.as_bytes(), codec).unwrap();
            let (b, _) = recompress_terrain_asset(asset.as_bytes(), codec).unwrap();
            assert_eq!(a, b, "{codec:?}");
        }
    }

    /// A **v1** payload transcodes forward: the container is lifted to v6 while
    /// the tile blobs are carried through untouched, so the tiles that come back
    /// out are the ones a v1 reader would have produced — empty maps, biomes and
    /// holes included.
    ///
    /// This is the arm that says the cook may compress an asset from any vintage
    /// without decoding it, which is the whole reason `recompress_terrain_asset`
    /// moves bytes rather than tiles.
    #[test]
    fn a_v1_payload_transcodes_forward_without_decoding_its_tiles() {
        let asset = sample_asset();
        let v1 = v1_image(&asset);
        let (packed, report) = recompress_terrain_asset(&v1, BlockCodec::Deflate).unwrap();
        assert_eq!(report.tiles, asset.reader().tile_count());
        let r = TerrainAssetReader::new(packed.as_slice()).unwrap();
        assert_eq!(r.header().schema_version, TERRAIN_ASSET_SCHEMA_VERSION);
        let v1r = TerrainAssetReader::new(v1.as_slice()).unwrap();
        for e in v1r.directory() {
            assert_eq!(v1r.tile_bytes(e.key).unwrap(), r.tile_bytes(e.key).unwrap());
        }
        // The ONE thing the lift is not bit-preserving about, pinned rather than
        // discovered: a v1 payload's pyramid options were *unknown*, and the
        // lifted header has nowhere to say so, so it says "the defaults". See
        // `recompress_terrain_asset`'s docs for why that is unreachable by the
        // only consumer of the distinction — and why this assertion is what a
        // future pack-side consumer would run into.
        assert_eq!(v1r.pyramid_options(), None, "a v1 payload records nothing");
        assert_eq!(r.pyramid_options(), Some(PyramidOptions::default()));
    }

    /// Compression is per-tile and independent: a tile that would inflate keeps
    /// [`BlockCodec::Raw`] and says so, so a payload can never grow.
    #[test]
    fn an_incompressible_tile_stays_raw_and_the_payload_never_grows() {
        let asset = sample_asset();
        for codec in BlockCodec::ALL {
            let (packed, report) = recompress_terrain_asset(asset.as_bytes(), codec).unwrap();
            assert!(
                report.bytes_after <= report.bytes_before,
                "{codec:?}: {} -> {}",
                report.bytes_before,
                report.bytes_after
            );
            assert!(packed.len() <= asset.as_bytes().len(), "{codec:?}");
            let r = TerrainAssetReader::new(packed.as_slice()).unwrap();
            for e in r.directory() {
                assert!(e.codec == BlockCodec::Raw || e.codec == codec, "{codec:?}");
            }
        }
    }

    /// A compressed tile decompresses through the **store** seam too — the path
    /// `sync_sim`/`sync_render` actually take — and the raw one still borrows.
    #[test]
    fn the_tile_store_seam_borrows_raw_and_owns_compressed() {
        use crate::residency::TileStore;
        // `sample_terrain`'s 5² tiles are below `block::MIN_COMPRESSIBLE`, so a
        // 33² one — still tiny, but past the threshold — is what makes the
        // "owned" half of this test non-vacuous.
        let mut t = TerrainData::new(33, 1.0);
        for tz in 0..2 {
            for tx in 0..2 {
                t.author_tile((tx, tz), height_fn);
            }
        }
        let opts = PyramidOptions::default();
        let asset = build_terrain_asset(&t, &build_pyramid(&t, opts), opts).unwrap();
        let key = asset.reader().directory()[0].key;
        assert!(matches!(
            TileStore::tile_bytes(&asset.reader(), key).unwrap(),
            Cow::Borrowed(_)
        ));

        let (packed, _) = recompress_terrain_asset(asset.as_bytes(), BlockCodec::Deflate).unwrap();
        let r = TerrainAssetReader::new(packed).unwrap();
        // The sample terrain's tiles are large enough to compress, so at least
        // one of them must have actually taken the owned path — otherwise this
        // test would pass vacuously on a corpus where nothing compressed.
        let compressed = r
            .directory()
            .iter()
            .find(|e| e.codec != BlockCodec::Raw)
            .expect("the sample terrain has at least one compressible tile");
        assert!(matches!(
            TileStore::tile_bytes(&r, compressed.key).unwrap(),
            Cow::Owned(_)
        ));
        assert_eq!(
            r.load_tile(compressed.key).unwrap().unwrap(),
            asset.reader().tile(compressed.key).unwrap().unwrap()
        );
    }

    /// **Membership does not decompress** (IASSET1 audit).
    ///
    /// `TileStore::contains_tile`'s default is `tile_bytes(key).is_some()`, which
    /// was free while `tile_bytes` was a sub-slice and is a whole tile's
    /// decompress now that it is a `Cow`. `TerrainAssetReader` overrides it with
    /// the directory's own binary search, and the only way to *observe* the
    /// difference — the only shape where the two answers differ — is a block the
    /// directory holds and the decoder refuses. So that is what this builds: a
    /// compressed tile whose length prefix is over-written with a small, in-range
    /// lie. `contains_tile` still says yes (the directory does hold it) while the
    /// bytes are gone, which is exactly the proof that the fetch never ran.
    ///
    /// A false answer here means the override was deleted and the fetch is back.
    #[test]
    fn membership_asks_the_directory_and_never_the_decoder() {
        use crate::residency::TileStore;
        let mut t = TerrainData::new(33, 1.0);
        for tz in 0..2 {
            for tx in 0..2 {
                t.author_tile((tx, tz), height_fn);
            }
        }
        let opts = PyramidOptions::default();
        let asset = build_terrain_asset(&t, &build_pyramid(&t, opts), opts).unwrap();
        let (mut packed, _) =
            recompress_terrain_asset(asset.as_bytes(), BlockCodec::Deflate).unwrap();

        let victim = {
            let r = TerrainAssetReader::new(packed.as_slice()).unwrap();
            *r.directory()
                .iter()
                .find(|e| e.codec != BlockCodec::Raw)
                .expect("at least one tile compressed")
        };
        // A length claim that is under the ceiling and wrong — the strict half of
        // `decode_block`, and a corruption the *parser* cannot see.
        let at = victim.offset as usize;
        packed[at..at + 8].copy_from_slice(&7u64.to_le_bytes());

        let r = TerrainAssetReader::new(packed.as_slice()).expect("the directory still parses");
        assert!(
            TileStore::contains_tile(&r, victim.key),
            "membership went through the decoder"
        );
        assert!(
            TileStore::tile_bytes(&r, victim.key).is_none(),
            "the fixture is vacuous — the doctored block still decoded"
        );
        // The reader's own door still reports the reason rather than an absence.
        assert!(r.tile(victim.key).is_err(), "a corrupt tile is an Err");
        // …and an absent key is still absent.
        assert!(!TileStore::contains_tile(
            &r,
            TileKey::new(0, (9_999, 9_999))
        ));
    }

    /// The decompression ceiling is a **container** property, and it refuses the
    /// gigabyte claim a doctored asset makes.
    #[test]
    fn the_block_ceiling_is_derived_from_the_tile_resolution() {
        // 257² samples × 64 B + slack — comfortably above the ~21.2 B/sample a
        // fully-materialized tile actually needs, and nowhere near a gigabyte.
        let c = tile_raw_ceiling(257);
        assert!(c > 257 * 257 * 22, "the ceiling must clear a real tile");
        assert!(c < 8 << 20, "…and must still be a ceiling: {c}");
        assert_eq!(tile_raw_ceiling(0), 64 * 1024, "no overflow at the edges");
    }

    /// **A header may not mint its own ceiling** (IASSET1 audit).
    ///
    /// [`tile_raw_ceiling`] derives the bound a compressed block's length claim
    /// is refused against *from a header field*, and the header came out of the
    /// same file as the claim. Unbounded, `tile_resolution = u32::MAX` gives a
    /// ceiling of `usize::MAX` and the bound bounds nothing — a forty-byte block
    /// asks for a terabyte and `decode_block` hands the allocator the number.
    ///
    /// So: the parser refuses the header, the builder refuses to write one (a
    /// writer must not manufacture a file its own reader rejects), and the
    /// ceiling function clamps even when called with a number nobody validated.
    #[test]
    fn a_doctored_tile_resolution_cannot_mint_its_own_block_ceiling() {
        // The clamp, first — this is what makes the function safe standalone.
        assert_eq!(
            tile_raw_ceiling(u32::MAX),
            tile_raw_ceiling(MAX_TILE_RESOLUTION)
        );
        assert!(
            tile_raw_ceiling(u32::MAX) < 300 << 20,
            "the worst-case allocation must be a knowable number: {}",
            tile_raw_ceiling(u32::MAX)
        );

        // The parser. A real image with the resolution word over-written is the
        // fixture, so nothing but that field differs from a payload that loads.
        let asset = sample_asset();
        let mut image = asset.as_bytes().to_vec();
        image[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = TerrainAssetReader::new(image.as_slice()).unwrap_err();
        assert!(
            matches!(&err, TerrainAssetError::Malformed(m) if m.contains("past the")),
            "{err}"
        );
        // …and one byte under the cap still parses, so the arm is not vacuous
        // for the wrong reason.
        image[12..16].copy_from_slice(&MAX_TILE_RESOLUTION.to_le_bytes());
        assert!(TerrainAssetReader::new(image.as_slice()).is_ok());

        // The writer, on the same number.
        let over = TerrainAssetBuilder::new(MAX_TILE_RESOLUTION + 1, 1.0).build();
        assert!(
            matches!(&over, Err(TerrainAssetError::Malformed(m)) if m.contains("past the")),
            "{over:?}"
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
    /// this loop. One helper for every generation, so the ladder tests cannot
    /// drift apart.
    fn legacy_image(asset: &TerrainAsset, version: u32) -> Vec<u8> {
        assert!(
            (2..=4).contains(&version),
            "legacy_image writes generation-1 (v2), -2 (v3) or -3 (v4) blobs"
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
                match version {
                    2 => bincode::serde::encode_to_vec(
                        TerrainTileFrozenV1::from_current(&tile),
                        tile_config(),
                    )
                    .unwrap(),
                    3 => bincode::serde::encode_to_vec(
                        TerrainTileFrozenV2::from_current(&tile),
                        tile_config(),
                    )
                    .unwrap(),
                    _ => bincode::serde::encode_to_vec(
                        TerrainTileFrozenV3::from_current(&tile),
                        tile_config(),
                    )
                    .unwrap(),
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

    /// The pre-P21.2 image (generation-3 tile blobs — biomes, no hole mask).
    fn v4_image(asset: &TerrainAsset) -> Vec<u8> {
        legacy_image(asset, 4)
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
            old.tile_bytes(key).unwrap().len() + 3,
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

        // The wire cost of v4 + v5 over v3, priced: one zero-length count per
        // layer per tile.
        let key = TileKey::lod0((0, 0));
        assert_eq!(
            asset.reader().tile_bytes(key).unwrap().len(),
            old.tile_bytes(key).unwrap().len() + 2,
            "an unpainted, un-carved tile costs exactly one extra byte per layer \
             added since v3"
        );
    }

    /// **A v4 payload loads forever** — the generation-3 ladder rung. Every tile
    /// decodes through [`TerrainTileFrozenV3`], **keeps its data maps and biome
    /// ids**, and comes back with the un-carved default hole mask.
    ///
    /// The third rung, written to the same shape as the first two for the same
    /// reason: "old bytes load" is one claim per generation, and only a per-rung
    /// test catches a `decode_tile_at` that routes v4 through the wrong wire type
    /// — which, positionally, would read the biome count as a hole count and then
    /// run off the end of the blob.
    #[test]
    fn a_v4_payload_still_loads_with_default_holes() {
        // Populate all three post-v2 layers, so nothing can pass by coming back
        // empty, and carve a hole so the *dropped* layer is a real one too.
        let mut t = sample_terrain();
        {
            let tile = t.get_tile_mut((2, 2)).unwrap();
            tile.set_map_texel(5, 1, 1, [3.0, 2.0, 1.0]);
            tile.set_biome_sample(5, 1, 1, 7);
            tile.set_hole(5, 3, 3, true);
        }
        let p = build_pyramid(&t, PyramidOptions::default());
        let asset = build_terrain_asset(&t, &p, PyramidOptions::default()).unwrap();
        assert!(asset
            .reader()
            .tile(TileKey::lod0((2, 2)))
            .unwrap()
            .unwrap()
            .has_holes());

        let v4 = v4_image(&asset);
        assert_eq!(v4[8..12], 4u32.to_le_bytes(), "the fixture must be v4");
        let back = TerrainAsset::from_bytes(v4).expect("a v4 payload must still load");
        let old = back.reader();
        assert_eq!(old.header().schema_version, 4);
        assert_eq!(old.tile_count(), asset.reader().tile_count());
        for key in old.keys() {
            let tile = old.tile(key).unwrap().expect("directory entry resolves");
            assert!(tile.holes_are_default(), "{key:?} conjured a hole mask");
            let new = asset.reader().tile(key).unwrap().unwrap();
            assert_eq!(tile.heights(), new.heights(), "{key:?} heights moved");
            assert_eq!(tile.weights(), new.weights(), "{key:?} weights moved");
            assert_eq!(tile.maps(), new.maps(), "{key:?} lost its data maps");
            assert_eq!(tile.biomes(), new.biomes(), "{key:?} lost its biome ids");
        }

        // The wire cost of v5 over v4 on an UN-carved tile: one zero-length count.
        let key = TileKey::lod0((0, 0));
        assert_eq!(
            asset.reader().tile_bytes(key).unwrap().len(),
            old.tile_bytes(key).unwrap().len() + 1,
            "an un-carved tile costs exactly one extra byte at v5"
        );
        // … and on a carved one, the packed mask — res² bits, not res² bytes.
        let carved = TileKey::lod0((2, 2));
        assert_eq!(
            asset.reader().tile_bytes(carved).unwrap().len(),
            old.tile_bytes(carved).unwrap().len() + 1 + crate::hole_mask_bytes(5),
            "a carved tile costs its packed mask and nothing more"
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
            let tile = decode_tile(&r.tile_bytes(key).unwrap())
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
            let tile = decode_tile(&r.tile_bytes(key).unwrap())
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
