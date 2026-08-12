//! The `.inf_tex` **v2 tiled container**, on the READ side: the header, the two
//! directories, and one tile as a borrowed slice.
//!
//! # Why this lives here and the writer does not (P26.3)
//!
//! P26.1 built the container inside `inf-material`, which is where the *writer*
//! belongs — tiling needs the mip chain, the format choice and the BC encoder,
//! and all three are import-time concerns. P26.2 then inverted the dependency
//! (`inf-material` → `inf-vt`) precisely so that binding virtual texturing into
//! the lit passes could never drag `image` + `naga` into a shipped player.
//!
//! P26.3 is that binding, and it makes the omission concrete: **the player has
//! to read tiles, and the player does not link `inf-material`** (it is a *dev*
//! dependency of `inf-render`). So the read half moves down to the crate the
//! player already links, leaving the write half exactly where it was:
//!
//! | stays in `inf-material` | moves here |
//! |---|---|
//! | `build_tiled_texture`, `lift_texture_asset` (the one writer) | the header + directory **parse** |
//! | `TiledTextureImage` (the payload it produces) | [`TiledTextureReader`] and [`tile`](TiledTextureReader::tile) |
//! | `level_bytes` / `to_texture_asset` (the v1 view, `TextureAsset`) | [`vt_desc`](TiledTextureReader::vt_desc) |
//! | the BC **encoder** (`inf_material::bc`) | the BC **decoder** ([`decode_bc1`] / [`decode_bc3`]) |
//!
//! The decoder is the one item that deviates from "move the parser, not the BC
//! code", and it deviates for a reason that is measured rather than tidy: the
//! transcode tier — an adapter without `TEXTURE_COMPRESSION_BC`, which is every
//! mobile target — pages [`TiledTextureReader::tile_rgba8`] into an
//! [`Rgba8`](crate::PageFormat::Rgba8) pool. Leaving the decoder up in
//! `inf-material` would have left that tier reachable in the editor and
//! unreachable in the shipped player: the exact split the P26.1 audit found in
//! the capability clamp, arriving a second time. The encoder stays put, and its
//! no-floating-point source gate with it.
//!
//! Everything `inf_material::tiles` used to export is still exported from there,
//! by re-export, so no call site moved.
//!
//! # The layout
//!
//! ```text
//! ┌ header (128 B, little-endian) ────────────────────────────────────┐
//! │  magic        [u8; 8]  b"INFVTEX\0"                               │
//! │  schema_ver   u32      TEX_ASSET_SCHEMA_VERSION (2)               │
//! │  width · height  u32   the VIRTUAL extent (mip 0)                 │
//! │  format       u32      0 Rgba8 · 1 Bc1 · 2 Bc3 (freeze-pinned)    │
//! │  flags        u32      bit 0 = srgb                               │
//! │  tile_size    u32      PAYLOAD texels per tile side (128)         │
//! │  border       u32      border texels per side (4)                 │
//! │  mip_count · tile_count  u32                                      │
//! │  tile_bytes   u32      bytes of ONE stored tile (uniform)         │
//! │  mip_dir_off · tile_dir_off · tile_base · total_len  (u64 each)   │
//! │  reserved     [u8; 48] zeros (room for v3 without a re-length)    │
//! ├ mip directory (mip_count × 32 B, FINEST FIRST) ───────────────────┤
//! │  width u32 · height u32 · tiles_x u32 · tiles_y u32 ·             │
//! │  first_tile u32 · tile_count u32 · pad[8]                         │
//! ├ tile directory (tile_count × 32 B, sorted by (mip, y, x)) ────────┤
//! │  mip u32 · x u32 · y u32 · pad u32 · offset u64 · len u64         │
//! ├ tile blobs ───────────────────────────────────────────────────────┤
//! │  … each 16-byte aligned, zero-padded up to the next boundary …    │
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Because there is exactly one writer, the layout is **pinned rather than
//! merely described**: [`parse`] requires the payload to be exactly `total_len`
//! bytes, `total_len` to be exactly `tile_base + stride × tile_count`, and tile
//! `n`'s blob to sit at exactly `tile_base + stride × n`. A tile's byte offset is
//! therefore *arithmetic*, and a directory can neither describe more tiles than
//! the file stores nor point two entries at one blob. (Those three equalities are
//! the P26.1 audit's third finding; the measurement that motivated them — a
//! 584 KiB payload that asked `level_bytes` for 1 GiB — is recorded there.)
//!
//! # The tile, and the arithmetic of its border
//!
//! A tile carries [`TILE_SIZE`] = 128 texels of **payload** per side plus a
//! [`TILE_BORDER`] = 4 texel ring on every side, so filtering a texel at the
//! payload edge reads real neighbours from the same mip rather than whatever page
//! happens to sit beside it in the atlas. 136 = 4 × 34, so a stored tile is
//! exactly 34 × 34 BC blocks and its block grid **is** the level's own block grid
//! offset by whole blocks.

use crate::address::{TileCoord, VtMipDesc, VtTextureDesc};
use crate::pool::PageFormat;

/// Magic at the head of every v2 `.inf_tex` payload.
pub const TEX_ASSET_MAGIC: [u8; 8] = *b"INFVTEX\0";

/// Current `.inf_tex` **container** schema version.
///
/// v1 is the bare bincode `TextureAsset` (no magic) and keeps loading forever;
/// v2 is the tiled image `inf_material::tiles` writes. This versions the
/// *container* and is independent of `TextureAsset::CURRENT_VERSION`, which
/// versions the in-memory record.
pub const TEX_ASSET_SCHEMA_VERSION: u32 = 2;

/// Tile blobs start on multiples of this many bytes — the same constant, and the
/// same reasoning, as `inf_asset::BLOB_ALIGN` and `.inf_vmesh`'s `SECTION_ALIGN`.
pub const SECTION_ALIGN: u64 = 16;

/// Bytes of the fixed header.
pub const HEADER_LEN: u64 = 128;

/// Bytes of one mip-directory entry.
pub const MIP_ENTRY_LEN: u64 = 32;

/// Bytes of one tile-directory entry.
pub const TILE_ENTRY_LEN: u64 = 32;

/// **Payload** texels per tile side. The addressable unit of a virtual texture.
pub const TILE_SIZE: u32 = 128;

/// Border texels baked on **each** side of a tile's payload, so filtering across
/// a tile edge never needs the neighbouring tile to be resident. A multiple of 4
/// because a BC block is 4×4 (see the module docs).
pub const TILE_BORDER: u32 = 4;

/// Texels per side of a tile **as stored**: `border + payload + border`.
pub const STORED_TILE_SIZE: u32 = TILE_SIZE + 2 * TILE_BORDER;

// The two invariants every other piece of arithmetic here assumes.
const _: () = assert!(TILE_BORDER.is_multiple_of(4));
const _: () = assert!(TILE_SIZE.is_multiple_of(4));
const _: () = assert!(STORED_TILE_SIZE.is_multiple_of(4));

/// A failure reading a v2 `.inf_tex` payload.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TiledTextureError {
    #[error("payload is shorter than the fixed header")]
    TooShort,
    #[error("bad .inf_tex v2 magic")]
    BadMagic,
    #[error("payload schema v{found} is newer than this build's v{current}")]
    SchemaTooNew { found: u32, current: u32 },
    #[error("malformed tiled .inf_tex payload: {0}")]
    Malformed(String),
    #[error("tile {index} is out of bounds or misaligned")]
    TileOutOfBounds { index: usize },
}

type Result<T> = std::result::Result<T, TiledTextureError>;

/// Round `n` up to the next multiple of [`SECTION_ALIGN`].
///
/// Public because the one **writer** (`inf_material::tiles::tile_levels`) lays
/// its blobs out on exactly this stride, and a second copy of the rule up there
/// is a second thing that can stop agreeing with [`parse`].
#[inline]
pub const fn align_up(n: u64) -> u64 {
    n.next_multiple_of(SECTION_ALIGN)
}

/// The **freeze-pinned** wire code of a storage format. Append-only: a code is
/// never reused and never renumbered, because it is written into shipped files.
pub fn format_code(f: PageFormat) -> u32 {
    match f {
        PageFormat::Rgba8 => 0,
        PageFormat::Bc1 => 1,
        PageFormat::Bc3 => 2,
    }
}

/// The inverse of [`format_code`]; `None` for a code this build does not know.
pub fn format_from_code(c: u32) -> Option<PageFormat> {
    match c {
        0 => Some(PageFormat::Rgba8),
        1 => Some(PageFormat::Bc1),
        2 => Some(PageFormat::Bc3),
        _ => None,
    }
}

/// Bytes one **stored tile** occupies in `format` — uniform across the whole
/// image, which is what makes a physical page a fixed-size slot.
pub fn stored_tile_bytes(format: PageFormat) -> usize {
    format.page_bytes(STORED_TILE_SIZE) as usize
}

/// The decoded fixed header of a v2 `.inf_tex` payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiledTextureHeader {
    pub schema_version: u32,
    /// The virtual extent (mip 0).
    pub width: u32,
    pub height: u32,
    pub format: PageFormat,
    /// Whether the data is sRGB-encoded (base colour) vs linear (normal/data).
    pub srgb: bool,
    /// Payload texels per tile side ([`TILE_SIZE`] as written).
    pub tile_size: u32,
    /// Border texels per side ([`TILE_BORDER`] as written).
    pub border: u32,
    pub mip_count: u32,
    pub tile_count: u32,
    /// Bytes of one stored tile (uniform).
    pub tile_bytes: u32,
    pub mip_dir_off: u64,
    pub tile_dir_off: u64,
    /// Absolute offset of the first tile blob.
    pub tile_base: u64,
    /// Payload length as written.
    pub total_len: u64,
}

impl TiledTextureHeader {
    /// Texels per side of a stored tile, as this payload declares them.
    #[inline]
    pub fn stored_tile_size(&self) -> u32 {
        self.tile_size + 2 * self.border
    }
}

/// One mip level's grid — everything a streamer needs about a level **before**
/// any of its bytes are touched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TexMipEntry {
    /// The level's virtual extent.
    pub width: u32,
    pub height: u32,
    /// Tiles across / down (`ceil(width / tile_size)` etc.).
    pub tiles_x: u32,
    pub tiles_y: u32,
    /// Index of this level's first tile in the tile directory.
    pub first_tile: u32,
    /// `tiles_x * tiles_y`.
    pub tile_count: u32,
}

/// One tile-directory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TexTileEntry {
    pub mip: u32,
    pub x: u32,
    pub y: u32,
    /// Absolute byte offset of the blob (16-byte aligned).
    pub offset: u64,
    /// Blob length in bytes.
    pub len: u64,
}

/// Random access over a v2 `.inf_tex` payload.
///
/// Generic over the byte source so one reader serves every backing: an owned
/// `Vec<u8>` (a loose file), a `&[u8]` borrowed from `PackReader::read_ref`'s
/// mapping, or a newtype over whatever a host holds the payload in. The header +
/// directories are parsed once at construction; every [`tile`](Self::tile) after
/// that is one sub-slice.
#[derive(Debug, Clone)]
pub struct TiledTextureReader<B> {
    bytes: B,
    header: TiledTextureHeader,
    mips: Vec<TexMipEntry>,
    tiles: Vec<TexTileEntry>,
}

/// A [`TiledTextureReader`] borrowing its bytes (the pack-mapping case).
pub type TiledTextureView<'a> = TiledTextureReader<&'a [u8]>;

impl<B: AsRef<[u8]>> TiledTextureReader<B> {
    /// Parse + validate a payload image.
    pub fn new(bytes: B) -> Result<Self> {
        let (header, mips, tiles) = parse(bytes.as_ref())?;
        Ok(Self {
            bytes,
            header,
            mips,
            tiles,
        })
    }

    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    #[inline]
    pub fn header(&self) -> &TiledTextureHeader {
        &self.header
    }

    /// The mip directory, **finest first**.
    #[inline]
    pub fn mips(&self) -> &[TexMipEntry] {
        &self.mips
    }

    /// **The residency door**: this container's address space, as
    /// [`VtResidency::register_texture`](crate::VtResidency::register_texture)
    /// wants it.
    ///
    /// The whole of what crosses the seam between the file and the streamer —
    /// the tile geometry and the grid of every level, and no bytes. What the
    /// streamer asks for *back* is one tile, through [`tile`](Self::tile) (or
    /// [`tile_rgba8`](Self::tile_rgba8) when the adapter has no BC): the same
    /// door, one format decision earlier.
    pub fn vt_desc(&self) -> VtTextureDesc {
        VtTextureDesc {
            tile_size: self.header.tile_size,
            border: self.header.border,
            srgb: self.header.srgb,
            mips: self
                .mips
                .iter()
                .map(|m| VtMipDesc {
                    width: m.width,
                    height: m.height,
                    tiles_x: m.tiles_x,
                    tiles_y: m.tiles_y,
                })
                .collect(),
        }
    }

    /// The tile directory, sorted by `(mip, y, x)`.
    #[inline]
    pub fn tiles(&self) -> &[TexTileEntry] {
        &self.tiles
    }

    /// The directory index of `(mip, x, y)`, or `None` if it is outside the grid.
    pub fn tile_index(&self, mip: u32, x: u32, y: u32) -> Option<usize> {
        let m = self.mips.get(mip as usize)?;
        if x >= m.tiles_x || y >= m.tiles_y {
            return None;
        }
        Some((m.first_tile + y * m.tiles_x + x) as usize)
    }

    /// **The tile read door** (what the residency pages): the stored blob of
    /// `(mip, x, y)` as a borrowed slice, whose offset is 16-byte aligned inside
    /// the payload.
    ///
    /// The bytes are `stored_tile_size²` texels in
    /// [`TiledTextureHeader::format`] — payload plus border ring — ready to
    /// `write_texture` into a physical page with no decode and no copy.
    pub fn tile(&self, mip: u32, x: u32, y: u32) -> Option<&[u8]> {
        let e = self.tiles.get(self.tile_index(mip, x, y)?)?;
        // Bounds were validated in `parse`.
        Some(&self.bytes.as_ref()[e.offset as usize..(e.offset + e.len) as usize])
    }

    /// [`tile`](Self::tile) by address.
    pub fn tile_at(&self, at: TileCoord) -> Option<&[u8]> {
        self.tile(at.mip, at.x, at.y)
    }

    /// **The CPU transcode door**: one stored tile decoded to RGBA8, still
    /// `stored_tile_size²` with its border ring intact.
    ///
    /// This is the fallback an adapter without `TEXTURE_COMPRESSION_BC` takes —
    /// the *same* residency door, one page-format decision earlier.
    pub fn tile_rgba8(&self, mip: u32, x: u32, y: u32) -> Option<Vec<u8>> {
        let blob = self.tile(mip, x, y)?;
        let n = self.header.stored_tile_size();
        Some(match self.header.format {
            PageFormat::Rgba8 => blob.to_vec(),
            PageFormat::Bc1 => decode_bc1(blob, n, n),
            PageFormat::Bc3 => decode_bc3(blob, n, n),
        })
    }
}

/// Parse + validate the header and both directories of a payload image.
pub fn parse(data: &[u8]) -> Result<(TiledTextureHeader, Vec<TexMipEntry>, Vec<TexTileEntry>)> {
    if (data.len() as u64) < HEADER_LEN {
        return Err(TiledTextureError::TooShort);
    }
    if data[0..8] != TEX_ASSET_MAGIC {
        return Err(TiledTextureError::BadMagic);
    }
    let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(data[o..o + 8].try_into().unwrap());

    let schema_version = u32_at(8);
    if schema_version > TEX_ASSET_SCHEMA_VERSION {
        return Err(TiledTextureError::SchemaTooNew {
            found: schema_version,
            current: TEX_ASSET_SCHEMA_VERSION,
        });
    }
    let format = format_from_code(u32_at(20)).ok_or_else(|| {
        TiledTextureError::Malformed(format!("unknown format code {}", u32_at(20)))
    })?;
    let header = TiledTextureHeader {
        schema_version,
        width: u32_at(12),
        height: u32_at(16),
        format,
        srgb: u32_at(24) & 1 != 0,
        tile_size: u32_at(28),
        border: u32_at(32),
        mip_count: u32_at(36),
        tile_count: u32_at(40),
        tile_bytes: u32_at(44),
        mip_dir_off: u64_at(48),
        tile_dir_off: u64_at(56),
        tile_base: u64_at(64),
        total_len: u64_at(72),
    };
    if header.tile_size == 0
        || !header.tile_size.is_multiple_of(4)
        || !header.border.is_multiple_of(4)
    {
        return Err(TiledTextureError::Malformed(format!(
            "tile geometry {}+2×{} is not a whole number of BC blocks",
            header.tile_size, header.border
        )));
    }
    let stored = header.stored_tile_size();
    if u64::from(header.tile_bytes) != format.page_bytes(stored) {
        return Err(TiledTextureError::Malformed(format!(
            "tile_bytes {} does not match a {stored}² {:?} tile",
            header.tile_bytes, format
        )));
    }
    // Every record the header claims has to be *stored*, so the payload length
    // bounds the counts from above — the `.inf_vmesh` discipline: a doctored
    // header must never make a `Vec::with_capacity` ask the allocator for
    // gigabytes from a file the process merely opened.
    let payload_len = data.len() as u64;
    let dir_end = HEADER_LEN
        + MIP_ENTRY_LEN * header.mip_count as u64
        + TILE_ENTRY_LEN * header.tile_count as u64;
    if payload_len < dir_end {
        return Err(TiledTextureError::TooShort);
    }
    if header.mip_dir_off != HEADER_LEN
        || header.tile_dir_off != HEADER_LEN + MIP_ENTRY_LEN * header.mip_count as u64
        || header.tile_base < header.tile_dir_off + TILE_ENTRY_LEN * header.tile_count as u64
        || !header.tile_base.is_multiple_of(SECTION_ALIGN)
    {
        return Err(TiledTextureError::Malformed(
            "directory offsets do not describe this payload".into(),
        ));
    }
    // **The payload must CONTAIN the tiles it declares**, and be exactly as long
    // as it says it is. Bounding each tile's own range (below) is not the same
    // claim: every entry may legally point at the *same* blob, and then a
    // directory — 32 B per tile — describes 73 984 B of tile each. Measured
    // before this check existed: a 584 KiB payload whose 16 384 entries all
    // aliased one blob made `level_bytes` ask for 1 GiB, 1 794× amplification,
    // from a file the thumbnailer opens the moment it appears in the Content
    // Drawer. Pinning `total_len` to the uniform-stride layout collapses it to
    // ~1×, because a level's texels can never outweigh the tiles that store
    // them. Free to pin: v2 has exactly one writer, and a v3 with a different
    // layout is refused above as `SchemaTooNew`.
    let stride = align_up(header.tile_bytes as u64);
    let laid_out = stride
        .checked_mul(header.tile_count as u64)
        .and_then(|n| header.tile_base.checked_add(n));
    if header.mip_count == 0
        || laid_out != Some(header.total_len)
        || header.total_len != payload_len
    {
        return Err(TiledTextureError::Malformed(format!(
            "a {payload_len}-byte payload does not hold {} tiles of {} B from {}",
            header.tile_count, header.tile_bytes, header.tile_base
        )));
    }

    let mut mips = Vec::with_capacity(header.mip_count as usize);
    let mut next_tile = 0u32;
    for i in 0..header.mip_count as usize {
        let b = HEADER_LEN as usize + i * MIP_ENTRY_LEN as usize;
        let e = TexMipEntry {
            width: u32_at(b),
            height: u32_at(b + 4),
            tiles_x: u32_at(b + 8),
            tiles_y: u32_at(b + 12),
            first_tile: u32_at(b + 16),
            tile_count: u32_at(b + 20),
        };
        if e.tiles_x == 0
            || e.tiles_y == 0
            || e.tiles_x != e.width.div_ceil(header.tile_size).max(1)
            || e.tiles_y != e.height.div_ceil(header.tile_size).max(1)
            || e.tile_count != e.tiles_x.saturating_mul(e.tiles_y)
            || e.first_tile != next_tile
        {
            return Err(TiledTextureError::Malformed(format!(
                "mip {i}'s grid does not tile its {}×{} extent",
                e.width, e.height
            )));
        }
        next_tile = e
            .first_tile
            .checked_add(e.tile_count)
            .ok_or_else(|| TiledTextureError::Malformed("tile block overflows".into()))?;
        mips.push(e);
    }
    if next_tile != header.tile_count {
        return Err(TiledTextureError::Malformed(
            "the mip directory does not account for exactly the header's tiles".into(),
        ));
    }
    // The header's extent IS mip 0's. Two fields carry the virtual extent and
    // nothing made them agree: `to_texture_asset` reads the header's, while
    // `level_bytes` reads the directory's, so a payload could hand a consumer a
    // `TextureAsset` whose `width` disagreed with `mips[0].width` — which is
    // what the sprite-sheet slicer cuts cells with.
    if (mips[0].width, mips[0].height) != (header.width, header.height) {
        return Err(TiledTextureError::Malformed(format!(
            "the header's {}×{} extent is not mip 0's {}×{}",
            header.width, header.height, mips[0].width, mips[0].height
        )));
    }

    let mut tiles = Vec::with_capacity(header.tile_count as usize);
    let end = payload_len;
    for index in 0..header.tile_count as usize {
        let b = header.tile_dir_off as usize + index * TILE_ENTRY_LEN as usize;
        let e = TexTileEntry {
            mip: u32_at(b),
            x: u32_at(b + 4),
            y: u32_at(b + 8),
            offset: u64_at(b + 16),
            len: u64_at(b + 24),
        };
        let m = mips
            .get(e.mip as usize)
            .ok_or(TiledTextureError::TileOutOfBounds { index })?;
        // The directory's order IS its index: entry n must be the tile the
        // (mip, y, x) walk puts at n. A reader computes an index arithmetically
        // (`tile_index`), so a directory that merely *contains* the right entries
        // in the wrong order would hand back the wrong bytes silently.
        if m.first_tile + e.y * m.tiles_x + e.x != index as u32
            || e.x >= m.tiles_x
            || e.y >= m.tiles_y
        {
            return Err(TiledTextureError::Malformed(format!(
                "tile directory entry {index} is labelled (mip {}, {}, {})",
                e.mip, e.x, e.y
            )));
        }
        // …and so is its BLOB. The same argument the label check above makes,
        // applied to the other half of the entry: entry n's offset must be the
        // one the uniform stride puts at n, not merely some 16-aligned offset
        // inside the payload. Without it a permuted or self-aliased directory
        // parses and hands back the wrong tile's bytes silently, and the hot path
        // — which computes a byte offset arithmetically rather than reading 32
        // bytes of directory per request — would disagree with the file it is
        // reading. `end` is `total_len`, pinned above.
        if e.len != header.tile_bytes as u64
            || e.offset != header.tile_base + stride * index as u64
            || e.offset.checked_add(e.len).is_none_or(|x| x > end)
        {
            return Err(TiledTextureError::TileOutOfBounds { index });
        }
        tiles.push(e);
    }
    Ok((header, mips, tiles))
}

/// Whether `bytes` look like a v2 tiled image (as opposed to a v1 bincode
/// `TextureAsset`, whose first four bytes are `schema_version = 1`).
pub fn is_v2(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && bytes[0..8] == TEX_ASSET_MAGIC
}

/// The format `bytes` **stores** its pages in, without building a reader
/// (P26.4).
///
/// A pool holds one page size for every tile of every texture in it, so a host
/// registering a level's textures has to know what it is about to be handed
/// *before* it can plan the pool. This is that question and only that question:
/// it parses the header and throws the directories away, so asking it of every
/// bound texture at load costs one pass over a few hundred bytes each.
///
/// `None` for anything that is not a readable v2 container — which is the same
/// answer [`TiledTextureReader::new`] will give the registration a moment later,
/// through [`VtRefusal`](../../inf_render/vt_library/enum.VtRefusal.html), so a
/// caller does not have to decide what a refusal means twice.
pub fn stored_page_format(bytes: &[u8]) -> Option<PageFormat> {
    parse(bytes).ok().map(|(h, _, _)| h.format)
}

// ── BC decoders ─────────────────────────────────────────────────────────────
//
// Moved down from `inf_material::bc` with the reader (see the module docs): the
// transcode tier is the shipped player's arm on an adapter without
// `TEXTURE_COMPRESSION_BC`, and the player does not link `inf-material`. The
// ENCODER stays there, with its no-floating-point source gate.

/// Decode a BC1 image to RGBA8 (opaque). Handles both the 4-color (c0 > c1) and
/// 3-color+transparent (c0 ≤ c1) block modes.
pub fn decode_bc1(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![0u8; (width * height * 4) as usize];
    let mut p = 0;
    for by in 0..height.div_ceil(4) {
        for bx in 0..width.div_ceil(4) {
            if p + 8 > data.len() {
                return out;
            }
            let (colors, _) = decode_color_block(&data[p..p + 8]);
            p += 8;
            blit_block(&mut out, width, height, bx, by, &colors, None);
        }
    }
    out
}

/// Decode a BC3 image to RGBA8 (with alpha).
pub fn decode_bc3(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![0u8; (width * height * 4) as usize];
    let mut p = 0;
    for by in 0..height.div_ceil(4) {
        for bx in 0..width.div_ceil(4) {
            if p + 16 > data.len() {
                return out;
            }
            let alpha = decode_alpha_block(&data[p..p + 8]);
            let (colors, _) = decode_color_block(&data[p + 8..p + 16]);
            p += 16;
            blit_block(&mut out, width, height, bx, by, &colors, Some(&alpha));
        }
    }
    out
}

fn from_565(c: u16) -> [u8; 3] {
    let r = ((c >> 11) & 0x1F) as u8;
    let g = ((c >> 5) & 0x3F) as u8;
    let b = (c & 0x1F) as u8;
    [
        (r << 3) | (r >> 2),
        (g << 2) | (g >> 4),
        (b << 3) | (b >> 2),
    ]
}

fn lerp(a: [u8; 3], b: [u8; 3], num: u8, den: u8) -> [u8; 3] {
    let mut out = [0u8; 3];
    for i in 0..3 {
        let x = a[i] as u32 * (den - num) as u32 + b[i] as u32 * num as u32;
        out[i] = (x / den as u32) as u8;
    }
    out
}

fn decode_color_block(block: &[u8]) -> ([[u8; 3]; 16], bool) {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let idxs = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let e0 = from_565(c0);
    let e1 = from_565(c1);
    let pal = if c0 > c1 {
        [e0, e1, lerp(e0, e1, 1, 3), lerp(e0, e1, 2, 3)]
    } else {
        [e0, e1, lerp(e0, e1, 1, 2), [0, 0, 0]]
    };
    let mut out = [[0u8; 3]; 16];
    for (n, o) in out.iter_mut().enumerate() {
        *o = pal[((idxs >> (2 * n)) & 0x3) as usize];
    }
    (out, c0 <= c1)
}

fn decode_alpha_block(block: &[u8]) -> [u8; 16] {
    let a0 = block[0];
    let a1 = block[1];
    let mut palette = [0u8; 8];
    palette[0] = a0;
    palette[1] = a1;
    if a0 > a1 {
        for i in 1..7u32 {
            palette[(i + 1) as usize] = (((7 - i) * a0 as u32 + i * a1 as u32) / 7) as u8;
        }
    } else {
        for i in 1..5u32 {
            palette[(i + 1) as usize] = (((5 - i) * a0 as u32 + i * a1 as u32) / 5) as u8;
        }
        palette[6] = 0;
        palette[7] = 255;
    }
    let mut bits: u64 = 0;
    for (b, &byte) in block[2..8].iter().enumerate() {
        bits |= (byte as u64) << (8 * b);
    }
    let mut out = [0u8; 16];
    for (n, o) in out.iter_mut().enumerate() {
        *o = palette[((bits >> (3 * n)) & 0x7) as usize];
    }
    out
}

fn blit_block(
    out: &mut [u8],
    width: u32,
    height: u32,
    bx: u32,
    by: u32,
    colors: &[[u8; 3]; 16],
    alpha: Option<&[u8; 16]>,
) {
    for j in 0..4u32 {
        for i in 0..4u32 {
            let x = bx * 4 + i;
            let y = by * 4 + j;
            if x >= width || y >= height {
                continue;
            }
            let n = (j * 4 + i) as usize;
            let c = colors[n];
            let o = ((y * width + x) * 4) as usize;
            out[o] = c[0];
            out[o + 1] = c[1];
            out[o + 2] = c[2];
            out[o + 3] = alpha.map(|a| a[n]).unwrap_or(255);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The container never touches a float, decoders included, and that is a
    /// LAW** — the encoder's P26.1 gate, followed to where P26.3 moved its other
    /// half.
    ///
    /// `inf_material::bc`'s gate scopes over `bc.rs` and could never see this
    /// file. The claim it makes reaches here all the same, and on the arm that
    /// matters most: `tile_rgba8` transcodes a BC page for an adapter without
    /// `TEXTURE_COMPRESSION_BC` — every mobile target — and that RGBA8 page is
    /// what the atlas holds and the shader samples. One `as f32` in
    /// `decode_color_block`'s palette interpolation and the same tile becomes a
    /// different image on a phone than on a desktop, with nothing comparing the
    /// two: the P14 trig law's shape, one tier down.
    ///
    /// Scoped to the code ABOVE this test module, on the encoder gate's pattern —
    /// a ban list has to be able to name the tokens it bans, and the fixtures
    /// below legitimately use floats.
    #[test]
    fn the_container_never_touches_a_float() {
        let whole = include_str!("container.rs");
        let marker = "#[cfg(test)]";
        let (code, _) = whole
            .split_once(marker)
            .expect("the test module marker moved; this gate scopes on it");
        let code: String = code
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in ["f32", "f64", "sqrt", "powf", "powi", "as f"] {
            assert!(
                !code.contains(banned),
                "`{banned}` reached the container: a decoded RGBA8 page is no \
                 longer the same bytes on every target, so a transcoding adapter \
                 samples a different image from a BC one"
            );
        }
        // A bare `0.5` is an f64 with no annotation to grep for.
        let chars: Vec<char> = code.chars().collect();
        assert!(
            !chars
                .windows(3)
                .any(|w| w[0].is_ascii_digit() && w[1] == '.' && w[2].is_ascii_digit()),
            "a decimal literal reached the container"
        );
        // Anti-vacuity: the filter left the code behind, not only the comments —
        // both the parser and the decoders, because a filter that ate one half
        // would leave the other and still pass.
        assert!(
            code.contains("pub fn parse(") && code.contains("fn decode_color_block"),
            "the source filter ate the container"
        );
    }

    /// The format code is written into shipped files, so it is a wire contract
    /// and not an enum's discriminant: pinned by value, both directions.
    #[test]
    fn the_format_code_is_freeze_pinned_both_ways() {
        for (f, c) in [
            (PageFormat::Rgba8, 0u32),
            (PageFormat::Bc1, 1),
            (PageFormat::Bc3, 2),
        ] {
            assert_eq!(format_code(f), c, "{f:?}");
            assert_eq!(format_from_code(c), Some(f));
        }
        assert_eq!(format_from_code(3), None);
    }

    /// The stored tile is a whole number of BC blocks in every format, and the
    /// number the header carries is [`PageFormat::page_bytes`]'s — the two are
    /// one arithmetic, not two that agree today.
    #[test]
    fn a_stored_tile_is_34_blocks_a_side() {
        assert_eq!(STORED_TILE_SIZE, 136);
        assert_eq!(STORED_TILE_SIZE / 4, 34);
        assert_eq!(stored_tile_bytes(PageFormat::Bc1), 34 * 34 * 8);
        assert_eq!(stored_tile_bytes(PageFormat::Bc3), 34 * 34 * 16);
        assert_eq!(stored_tile_bytes(PageFormat::Rgba8), 136 * 136 * 4);
    }

    /// A payload that is not a v2 image is refused **by name** at the door,
    /// rather than mis-parsed into a plausible header.
    #[test]
    fn a_non_v2_payload_is_refused_by_name() {
        assert_eq!(parse(&[]), Err(TiledTextureError::TooShort));
        assert_eq!(parse(&[0u8; 4]), Err(TiledTextureError::TooShort));
        let mut v1 = vec![0u8; HEADER_LEN as usize];
        v1[0] = 1; // a bincode `TextureAsset`'s schema_version = 1
        assert_eq!(parse(&v1), Err(TiledTextureError::BadMagic));
        assert!(!is_v2(&v1));

        let mut too_new = vec![0u8; HEADER_LEN as usize];
        too_new[0..8].copy_from_slice(&TEX_ASSET_MAGIC);
        too_new[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(
            is_v2(&too_new),
            "the magic is v2's regardless of the version"
        );
        assert_eq!(
            parse(&too_new),
            Err(TiledTextureError::SchemaTooNew {
                found: 99,
                current: TEX_ASSET_SCHEMA_VERSION,
            })
        );
    }

    /// The decoders moved with the reader, so they need an arm here rather than
    /// only in the crate they left: a BC1 block of one solid colour decodes to
    /// that colour in every texel, and a BC3 block carries its alpha.
    #[test]
    fn the_decoders_came_across_intact() {
        // BC1: c0 == c1 == white in 565, all indices 0.
        let mut block = vec![0u8; 8];
        block[0..2].copy_from_slice(&0xFFFFu16.to_le_bytes());
        block[2..4].copy_from_slice(&0xFFFFu16.to_le_bytes());
        let rgba = decode_bc1(&block, 4, 4);
        assert_eq!(rgba.len(), 4 * 4 * 4);
        assert!(rgba.chunks_exact(4).all(|p| p == [255, 255, 255, 255]));

        // BC3: alpha endpoints 0/0 (a0 == a1 ⇒ the 6-value palette, index 0 = 0),
        // colour black.
        let mut bc3 = vec![0u8; 16];
        bc3[0] = 0;
        bc3[1] = 0;
        let rgba = decode_bc3(&bc3, 4, 4);
        assert!(rgba.chunks_exact(4).all(|p| p == [0, 0, 0, 0]));
    }
}
