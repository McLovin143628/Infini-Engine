//! The `.inf_tex` **v2 tiled container** (P26.1): a texture leaves its single
//! bincode blob for a random-access, tile-at-a-time image.
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
//! # Why a sectioned v2 and not a range-read of v1
//!
//! v1 is `inf_asset::encode(&TextureAsset)` — one bincode stream whose `mips`
//! field is a length prefix followed by **varint-packed** records, so there is no
//! byte offset for "the tile at (3, 2) of mip 1" that does not require decoding
//! everything before it. Virtual texturing needs exactly that byte offset, on the
//! hot path, per frame. This is the same shape, and the same reasoning, as
//! [`inf_vgeom`'s `.inf_vmesh` v2](../../inf_vgeom/asset/index.html) and
//! `.inf_terrain` (P16.3): header + sorted directory + 16-byte-aligned blobs,
//! cooked uncompressed so a resident tile is a borrowed slice of an mmap.
//!
//! **v1 keeps loading forever.** [`TextureAsset::from_payload`] sniffs the magic;
//! a payload without it is decoded as a v1 `TextureAsset`. A bincode
//! `TextureAsset` begins with `schema_version: u32 = 1` (`01 00 00 00`), which
//! cannot collide with `b"INFVTEX\0"`.
//!
//! # There is exactly one writer: [`TiledTextureImage::as_bytes`]
//!
//! Like `.inf_vmesh` and `.inf_terrain`, the bytes on disk and in a `.inf_pack`
//! are the **raw image**, never `inf_asset::encode` output — a bincode length
//! prefix would shift every tile off its 16-byte boundary and defeat the whole
//! layout, silently. So [`TiledTextureImage`] deliberately implements neither
//! `AssetPayload` nor `Serialize`, which makes the generic asset-writing doors a
//! type error rather than a subtly wrong file.
//!
//! Because there is one writer, the v2 layout is **pinned rather than merely
//! described**: `parse` requires the payload to be exactly `total_len` bytes,
//! `total_len` to be exactly `tile_base + stride × tile_count`, and tile `n`'s
//! blob to sit at exactly `tile_base + stride × n`. So a tile's byte offset is
//! *arithmetic* — P26.2's hot path never has to read 32 bytes of directory to
//! find one — and a directory can neither describe more tiles than the file
//! stores nor point two entries at one blob. A v3 that lays its bytes out
//! differently changes `schema_ver` and is refused here as
//! [`TiledTextureError::SchemaTooNew`], so pinning v2 costs nothing.
//!
//! # The tile, and the arithmetic of its border
//!
//! A tile carries [`TILE_SIZE`] = 128 texels of **payload** per side. Filtering a
//! texel at the payload edge reads its neighbours, which live in the *next* tile
//! — possibly not resident, possibly at a different atlas address — so each tile
//! also stores a **border ring** of [`TILE_BORDER`] = 4 texels on every side,
//! baked at cook time from the neighbouring texels of the same mip:
//!
//! ```text
//!   stored tile side = border + TILE_SIZE + border = 4 + 128 + 4 = 136 texels
//! ```
//!
//! The border width is 4 and not 1 or 2 because **a BC block is 4×4**: a stored
//! tile is compressed as one block grid, so its side must be a multiple of 4 or
//! the block grid would not tile it. 136 = 4 × 34, so a stored tile is exactly
//! **34 × 34 blocks** — 9 248 B in BC1, 18 496 B in BC3, 73 984 B as RGBA8.
//!
//! That choice buys a property the whole v1-compatibility story rests on. Stored
//! texel `i` of tile `tx` is source texel `tx·128 + i − 4`, so stored block `b`
//! begins at source texel `tx·128 + 4b − 4`, which is a multiple of 4 for every
//! `b`. **The stored block grid is the level's own block grid, offset by whole
//! blocks** — stored block `b ∈ [1, 33)` *is* level block `tx·32 + (b − 1)`, byte
//! for byte. Reconstructing a whole level from tiles is therefore a re-gather of
//! block bytes with no decode and no re-compression, and it reproduces exactly
//! what v1 would have written from the same source.
//!
//! # Clamp padding, at the border and at the edge
//!
//! One rule covers both: a stored texel whose source coordinate falls outside the
//! mip level is taken from the nearest texel that is inside
//! (`clamp(x, 0, w−1)`). That is what fills the border ring of a tile on the
//! level's boundary, and it is what fills the right/bottom remainder of a tile
//! when the level is not a multiple of 128 — the same edge-clamp rule
//! [`crate::bc`] already applies to partial blocks.
//!
//! # The pyramid
//!
//! Every mip of the full chain (down to 1×1, exactly the levels v1 stores) is
//! tiled at `ceil(w/128) × ceil(h/128)` tiles. Once a level fits inside one tile
//! that is one tile, and it stays one tile all the way down. **Every stored tile
//! is the same size**, which is what lets a physical page pool be a flat array of
//! interchangeable slots (P26.2). The cost is the tail: a 1×1 mip still occupies
//! a 136² tile, so a full chain adds ~8 tail tiles (~74 KB in BC1) to any
//! texture. Packing the tail mips into a single shared page is the documented
//! follow-up; it is a size optimisation, not a correctness one.

use inf_asset::AssetKind;

use crate::bc;
use crate::error::MaterialError;
use crate::texture::{TextureAsset, TextureFormat, TextureMip};

/// Magic at the head of every v2 `.inf_tex` payload.
pub const TEX_ASSET_MAGIC: [u8; 8] = *b"INFVTEX\0";

/// Current `.inf_tex` **container** schema version.
///
/// v1 is the bare bincode [`TextureAsset`] (no magic) and keeps loading forever;
/// v2 is the tiled image this module writes. This versions the *container* and is
/// independent of [`TextureAsset::CURRENT_VERSION`], which versions the in-memory
/// record and is unchanged by this batch.
pub const TEX_ASSET_SCHEMA_VERSION: u32 = 2;

/// Tile blobs start on multiples of this many bytes — the same constant, and the
/// same reasoning, as [`inf_asset::BLOB_ALIGN`] and `.inf_vmesh`'s `SECTION_ALIGN`.
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

// The two invariants every other piece of arithmetic in this module assumes.
const _: () = assert!(TILE_BORDER.is_multiple_of(4));
const _: () = assert!(TILE_SIZE.is_multiple_of(4));
const _: () = assert!(STORED_TILE_SIZE.is_multiple_of(4));

/// A failure building or reading a v2 `.inf_tex` payload.
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
#[inline]
const fn align_up(n: u64) -> u64 {
    n.next_multiple_of(SECTION_ALIGN)
}

/// The **freeze-pinned** wire code of a storage format. Append-only: a code is
/// never reused and never renumbered, because it is written into shipped files.
fn format_code(f: TextureFormat) -> u32 {
    match f {
        TextureFormat::Rgba8 => 0,
        TextureFormat::Bc1 => 1,
        TextureFormat::Bc3 => 2,
    }
}

fn format_from_code(c: u32) -> Option<TextureFormat> {
    match c {
        0 => Some(TextureFormat::Rgba8),
        1 => Some(TextureFormat::Bc1),
        2 => Some(TextureFormat::Bc3),
        _ => None,
    }
}

/// Bytes one **stored tile** occupies in `format` — uniform across the whole
/// image, which is what makes a physical page a fixed-size slot.
pub fn stored_tile_bytes(format: TextureFormat) -> usize {
    format.level_size(STORED_TILE_SIZE, STORED_TILE_SIZE)
}

/// The decoded fixed header of a v2 `.inf_tex` payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TiledTextureHeader {
    pub schema_version: u32,
    /// The virtual extent (mip 0).
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
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

/// Address of one tile in a virtual texture — the key P26.2's residency table is
/// built on.
///
/// **It lives in [`inf_vt`], and is re-exported here.** The address of a tile is
/// more primitive than the file that stores it, so it belongs to the crate that
/// owns the virtual address space; this crate consumes it. The direction is not a
/// preference — `inf-vt` must never name `inf-material` (which pulls `image` and
/// `naga`, neither of which a shipped player links to sample a texture), and
/// making `inf-material` the *upper* crate turns that mistake into a dependency
/// cycle instead of a review question. See `inf_vt`'s crate docs.
///
/// Its derived `Ord` is `(mip, x, y)` while the tile directory is sorted
/// `(mip, y, x)`; [`TileCoord::payload_order`] is the key to sort a request set
/// on. Pinned against a real directory by
/// [`tests::the_tile_coord_order_is_not_the_payload_order`].
pub use inf_vt::TileCoord;

// ── builder ─────────────────────────────────────────────────────────────────

/// Lay an RGBA8 source out as a v2 tiled image.
///
/// Pure and byte-deterministic: the mip chain, the format choice, the tile order
/// and the padding are all functions of `(rgba, width, height, settings)` alone,
/// so two builds of one source are byte-identical (the cook's guarantee, and the
/// reason a re-import produces the same content hash).
pub fn build_tiled_texture(
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    settings: crate::texture::TextureImportSettings,
) -> std::result::Result<TiledTextureImage, MaterialError> {
    let levels = crate::texture::rgba_mip_chain(rgba, width, height, settings.generate_mips)?;
    let format = crate::texture::choose_format(settings.compression, &levels[0].2);
    Ok(tile_levels(&levels, format, settings.srgb))
}

/// Lay an **already-imported v1 [`TextureAsset`]** out as a v2 tiled image — the
/// lift path, the twin of `VgeomSource::from_payload`'s v1 branch.
///
/// Every payload written before this batch is a v1 record, and the residency door
/// must be able to page one. Rather than teach the streamer a second shape, a v1
/// asset is lifted here: each level is CPU-decoded to RGBA8 (the door
/// [`TextureAsset::level_rgba8`] already is, for the thumbnailer) and re-tiled in
/// **the asset's own format**.
///
/// **Honest cost:** for a BC asset that is a decode→re-encode round trip, so a
/// lifted tile is not guaranteed bit-identical to the v1 block it came from
/// (bounding-box endpoints are re-derived from already-quantised colours). It is
/// stable — the same input always gives the same output — and it is only ever
/// reached by content that predates the tiled container. Content imported from
/// here on is tiled from the *original* pixels and never takes this path.
pub fn lift_texture_asset(
    tex: &TextureAsset,
) -> std::result::Result<TiledTextureImage, MaterialError> {
    if tex.mips.is_empty() {
        return Err(MaterialError::Image("texture has no mip levels".into()));
    }
    let mut levels: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(tex.mips.len());
    for (i, mip) in tex.mips.iter().enumerate() {
        let rgba = tex
            .level_rgba8(i)
            .ok_or_else(|| MaterialError::Image(format!("mip {i} will not decode")))?;
        if rgba.len() < (mip.width as usize * mip.height as usize * 4) {
            return Err(MaterialError::Image(format!("mip {i} is truncated")));
        }
        levels.push((mip.width, mip.height, rgba));
    }
    Ok(tile_levels(&levels, tex.format, tex.srgb))
}

/// The one tiler. `levels` are RGBA8, largest first.
fn tile_levels(
    levels: &[(u32, u32, Vec<u8>)],
    format: TextureFormat,
    srgb: bool,
) -> TiledTextureImage {
    let stored = STORED_TILE_SIZE;
    let tile_bytes = stored_tile_bytes(format);

    // ── 1. The grids, finest first ──
    let mut mips: Vec<TexMipEntry> = Vec::with_capacity(levels.len());
    let mut first_tile = 0u32;
    for &(w, h, _) in levels {
        let tiles_x = w.div_ceil(TILE_SIZE).max(1);
        let tiles_y = h.div_ceil(TILE_SIZE).max(1);
        mips.push(TexMipEntry {
            width: w,
            height: h,
            tiles_x,
            tiles_y,
            first_tile,
            tile_count: tiles_x * tiles_y,
        });
        first_tile += tiles_x * tiles_y;
    }
    let tile_count = first_tile as usize;

    // ── 2. Offsets ──
    let mip_dir_off = HEADER_LEN;
    let tile_dir_off = mip_dir_off + MIP_ENTRY_LEN * mips.len() as u64;
    let tile_base = align_up(tile_dir_off + TILE_ENTRY_LEN * tile_count as u64);
    let stride = align_up(tile_bytes as u64);
    let total_len = tile_base + stride * tile_count as u64;

    // ── 3. Emit the header ──
    let mut out: Vec<u8> = Vec::with_capacity(total_len as usize);
    out.extend_from_slice(&TEX_ASSET_MAGIC);
    out.extend_from_slice(&TEX_ASSET_SCHEMA_VERSION.to_le_bytes());
    out.extend_from_slice(&levels[0].0.to_le_bytes());
    out.extend_from_slice(&levels[0].1.to_le_bytes());
    out.extend_from_slice(&format_code(format).to_le_bytes());
    out.extend_from_slice(&u32::from(srgb).to_le_bytes());
    out.extend_from_slice(&TILE_SIZE.to_le_bytes());
    out.extend_from_slice(&TILE_BORDER.to_le_bytes());
    out.extend_from_slice(&(mips.len() as u32).to_le_bytes());
    out.extend_from_slice(&(tile_count as u32).to_le_bytes());
    out.extend_from_slice(&(tile_bytes as u32).to_le_bytes());
    debug_assert_eq!(out.len(), 48, "the u64 lane starts at 48");
    out.extend_from_slice(&mip_dir_off.to_le_bytes());
    out.extend_from_slice(&tile_dir_off.to_le_bytes());
    out.extend_from_slice(&tile_base.to_le_bytes());
    out.extend_from_slice(&total_len.to_le_bytes());
    debug_assert!(out.len() as u64 <= HEADER_LEN);
    out.resize(HEADER_LEN as usize, 0);

    // ── 4. The mip directory ──
    for m in &mips {
        out.extend_from_slice(&m.width.to_le_bytes());
        out.extend_from_slice(&m.height.to_le_bytes());
        out.extend_from_slice(&m.tiles_x.to_le_bytes());
        out.extend_from_slice(&m.tiles_y.to_le_bytes());
        out.extend_from_slice(&m.first_tile.to_le_bytes());
        out.extend_from_slice(&m.tile_count.to_le_bytes());
        out.extend_from_slice(&[0u8; 8]);
    }
    debug_assert_eq!(out.len() as u64, tile_dir_off);

    // ── 5. The tile directory, sorted by (mip, y, x) ──
    for (mip, m) in mips.iter().enumerate() {
        for y in 0..m.tiles_y {
            for x in 0..m.tiles_x {
                let index = m.first_tile + y * m.tiles_x + x;
                out.extend_from_slice(&(mip as u32).to_le_bytes());
                out.extend_from_slice(&x.to_le_bytes());
                out.extend_from_slice(&y.to_le_bytes());
                out.extend_from_slice(&0u32.to_le_bytes());
                out.extend_from_slice(&(tile_base + stride * index as u64).to_le_bytes());
                out.extend_from_slice(&(tile_bytes as u64).to_le_bytes());
            }
        }
    }
    out.resize(tile_base as usize, 0);

    // ── 6. The blobs, in directory order ──
    for (mip, m) in mips.iter().enumerate() {
        let (lw, lh, ref rgba) = levels[mip];
        for y in 0..m.tiles_y {
            for x in 0..m.tiles_x {
                let gathered = gather_stored_tile(rgba, lw, lh, x, y);
                let blob = match format {
                    TextureFormat::Rgba8 => gathered,
                    TextureFormat::Bc1 => bc::compress_bc1(&gathered, stored, stored),
                    TextureFormat::Bc3 => bc::compress_bc3(&gathered, stored, stored),
                };
                debug_assert_eq!(blob.len(), tile_bytes);
                out.extend_from_slice(&blob);
                out.resize(
                    (tile_base + stride * (m.first_tile + y * m.tiles_x + x) as u64 + stride)
                        as usize,
                    0,
                );
            }
        }
    }
    debug_assert_eq!(out.len() as u64, total_len);

    TiledTextureImage { bytes: out }
}

/// Gather one stored tile's `STORED_TILE_SIZE²` RGBA8 texels out of a level,
/// clamping every coordinate that falls outside it (the border ring and the
/// right/bottom remainder alike — see the module docs).
fn gather_stored_tile(rgba: &[u8], lw: u32, lh: u32, tx: u32, ty: u32) -> Vec<u8> {
    let stored = STORED_TILE_SIZE as usize;
    let mut out = vec![0u8; stored * stored * 4];
    let x0 = tx * TILE_SIZE;
    let y0 = ty * TILE_SIZE;
    for j in 0..stored {
        let sy = (y0 as i64 + j as i64 - TILE_BORDER as i64).clamp(0, lh as i64 - 1) as u32;
        for i in 0..stored {
            let sx = (x0 as i64 + i as i64 - TILE_BORDER as i64).clamp(0, lw as i64 - 1) as u32;
            let si = ((sy * lw + sx) * 4) as usize;
            let di = (j * stored + i) * 4;
            out[di..di + 4].copy_from_slice(&rgba[si..si + 4]);
        }
    }
    out
}

// ── the image ───────────────────────────────────────────────────────────────

/// A validated v2 `.inf_tex` payload image. Owns its bytes; this — not
/// `inf_asset::encode` — is what goes on disk and into a pack.
#[derive(Clone, PartialEq, Eq)]
pub struct TiledTextureImage {
    bytes: Vec<u8>,
}

impl TiledTextureImage {
    /// The asset kind + container schema version this image owes the database (it
    /// implements no `AssetPayload` on purpose — see the module docs).
    pub const KIND: AssetKind = AssetKind::Texture;
    pub const SCHEMA_VERSION: u32 = TEX_ASSET_SCHEMA_VERSION;

    /// Validate + adopt already-serialized bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        parse(&bytes)?;
        Ok(Self { bytes })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// A random-access view over the image.
    pub fn reader(&self) -> TiledTextureView<'_> {
        TiledTextureReader::new(self.bytes.as_slice()).expect("image validated at construction")
    }
}

impl std::fmt::Debug for TiledTextureImage {
    /// Summarizes; never dumps the (possibly hundred-MB) payload.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = self.reader();
        f.debug_struct("TiledTextureImage")
            .field("bytes", &self.bytes.len())
            .field("extent", &(r.header().width, r.header().height))
            .field("format", &r.header().format)
            .field("mips", &r.mips().len())
            .field("tiles", &r.tiles().len())
            .finish()
    }
}

// ── reader ──────────────────────────────────────────────────────────────────

/// Random access over a v2 `.inf_tex` payload.
///
/// Generic over the byte source so one reader serves every backing: an owned
/// `Vec<u8>` (a loose file), a `&[u8]` borrowed from
/// [`PackReader::read_ref`](inf_asset::PackReader::read_ref)'s mapping, or a `Cow`
/// straight off it. The header + directories are parsed once at construction;
/// every [`tile`](Self::tile) after that is one sub-slice.
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
    /// [`inf_vt::VtResidency::register_texture`] wants it.
    ///
    /// The whole of what crosses the seam between the file and the streamer —
    /// the tile geometry and the grid of every level, and no bytes. What the
    /// streamer asks for *back* is one tile, through [`tile`](Self::tile) (or
    /// [`tile_rgba8`](Self::tile_rgba8) when the adapter has no BC): the same
    /// door, one format decision earlier.
    pub fn vt_desc(&self) -> inf_vt::VtTextureDesc {
        inf_vt::VtTextureDesc {
            tile_size: self.header.tile_size,
            border: self.header.border,
            srgb: self.header.srgb,
            mips: self
                .mips
                .iter()
                .map(|m| inf_vt::VtMipDesc {
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

    /// **The tile read door** (what P26.2's residency pages): the stored blob of
    /// `(mip, x, y)` as a borrowed slice, whose offset is 16-byte aligned inside
    /// the payload.
    ///
    /// The bytes are `STORED_TILE_SIZE²` texels in [`TiledTextureHeader::format`]
    /// — payload plus border ring — ready to `write_texture` into a physical page
    /// with no decode and no copy.
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
    /// `STORED_TILE_SIZE²` with its border ring intact.
    ///
    /// This is the fallback an adapter without `TEXTURE_COMPRESSION_BC` takes —
    /// the *same* residency door, one page-format decision earlier. It reuses
    /// [`crate::bc`]'s decoders, which the thumbnailer has depended on since P4.
    pub fn tile_rgba8(&self, mip: u32, x: u32, y: u32) -> Option<Vec<u8>> {
        let blob = self.tile(mip, x, y)?;
        let n = self.header.stored_tile_size();
        Some(match self.header.format {
            TextureFormat::Rgba8 => blob.to_vec(),
            TextureFormat::Bc1 => bc::decode_bc1(blob, n, n),
            TextureFormat::Bc3 => bc::decode_bc3(blob, n, n),
        })
    }

    /// One whole mip level, **borders stripped**, in the stored format — the
    /// bytes v1 keeps in `TextureMip::data`.
    ///
    /// A re-gather, not a re-encode: a payload block of a tile *is* a block of
    /// the level (see the module docs), so this copies block bytes straight
    /// across for BC and texel rows for RGBA8.
    pub fn level_bytes(&self, mip: u32) -> Option<Vec<u8>> {
        let m = *self.mips.get(mip as usize)?;
        let stored = self.header.stored_tile_size();
        let border = self.header.border;
        let tile_size = self.header.tile_size;
        match self.header.format {
            TextureFormat::Rgba8 => {
                let mut out = vec![0u8; (m.width as usize) * (m.height as usize) * 4];
                for ty in 0..m.tiles_y {
                    for tx in 0..m.tiles_x {
                        let blob = self.tile(mip, tx, ty)?;
                        for j in 0..tile_size {
                            let dy = ty * tile_size + j;
                            if dy >= m.height {
                                break;
                            }
                            let sj = (j + border) as usize;
                            for i in 0..tile_size {
                                let dx = tx * tile_size + i;
                                if dx >= m.width {
                                    break;
                                }
                                let si = (sj * stored as usize + (i + border) as usize) * 4;
                                let di = ((dy * m.width + dx) * 4) as usize;
                                out[di..di + 4].copy_from_slice(&blob[si..si + 4]);
                            }
                        }
                    }
                }
                Some(out)
            }
            f => {
                let block = if f == TextureFormat::Bc3 { 16usize } else { 8 };
                let lbx = m.width.div_ceil(4);
                let lby = m.height.div_ceil(4);
                let sbx = (stored / 4) as usize; // blocks per stored tile row
                let border_blocks = (border / 4) as usize;
                let tile_blocks = (tile_size / 4) as usize;
                let mut out = vec![0u8; lbx as usize * lby as usize * block];
                for by in 0..lby {
                    for bx in 0..lbx {
                        let (tx, ty) = (bx as usize / tile_blocks, by as usize / tile_blocks);
                        let blob = self.tile(mip, tx as u32, ty as u32)?;
                        let si = ((by as usize % tile_blocks + border_blocks) * sbx
                            + (bx as usize % tile_blocks + border_blocks))
                            * block;
                        let di = (by as usize * lbx as usize + bx as usize) * block;
                        out[di..di + block].copy_from_slice(&blob[si..si + block]);
                    }
                }
                Some(out)
            }
        }
    }

    /// The **v1 view**: the whole texture rebuilt as the record every consumer
    /// written before P26.1 already knows how to read.
    ///
    /// Byte-identical to what v1 import would have written from the same source
    /// (the block-grid property in the module docs), which is what lets the
    /// thumbnailer, the sprite-sheet slicer and the PCG mask reader keep their
    /// existing code with one call changed.
    pub fn to_texture_asset(&self) -> Option<TextureAsset> {
        let mips = (0..self.mips.len() as u32)
            .map(|l| {
                let m = self.mips[l as usize];
                Some(TextureMip {
                    width: m.width,
                    height: m.height,
                    data: self.level_bytes(l)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(TextureAsset {
            schema_version: TextureAsset::CURRENT_VERSION,
            width: self.header.width,
            height: self.header.height,
            format: self.header.format,
            srgb: self.header.srgb,
            mips,
        })
    }
}

/// Parse + validate the header and both directories of a payload image.
fn parse(data: &[u8]) -> Result<(TiledTextureHeader, Vec<TexMipEntry>, Vec<TexTileEntry>)> {
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
    if header.tile_bytes as usize != format.level_size(stored, stored) {
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
        // parses and hands back the wrong tile's bytes silently, and P26.2's
        // hot path — which computes a byte offset arithmetically rather than
        // reading 32 bytes of directory per request — would disagree with the
        // file it is reading. `end` is `total_len`, pinned above.
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

/// The page format a container's tiles upload as **when the adapter can take
/// them whole**.
///
/// The other arm — an adapter without `TEXTURE_COMPRESSION_BC` — does not use
/// this: it pages [`TiledTextureReader::tile_rgba8`] into an
/// [`inf_vt::PageFormat::Rgba8`] pool, at 8× the bytes for BC1 and 4× for BC3.
/// So the two enums are not mirrors of one another and cannot drift: this one
/// says what is *on disk*, [`inf_vt::PageFormat`] says what the *pool* is, and
/// on that arm they deliberately differ.
impl From<TextureFormat> for inf_vt::PageFormat {
    fn from(f: TextureFormat) -> Self {
        match f {
            TextureFormat::Rgba8 => inf_vt::PageFormat::Rgba8,
            TextureFormat::Bc1 => inf_vt::PageFormat::Bc1,
            TextureFormat::Bc3 => inf_vt::PageFormat::Bc3,
        }
    }
}

/// Whether `bytes` look like a v2 tiled image (as opposed to a v1 bincode
/// [`TextureAsset`], whose first four bytes are `schema_version = 1`).
pub fn is_v2(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && bytes[0..8] == TEX_ASSET_MAGIC
}

/// Decode a `.inf_tex` payload of **either** version into the v1 record.
///
/// The one door every consumer that wants whole levels goes through; the
/// tile-at-a-time readers use [`TiledTextureReader`] directly.
pub fn decode_texture_payload(bytes: &[u8]) -> std::result::Result<TextureAsset, MaterialError> {
    if is_v2(bytes) {
        let r = TiledTextureReader::new(bytes).map_err(|e| MaterialError::Image(e.to_string()))?;
        r.to_texture_asset()
            .ok_or_else(|| MaterialError::Image("tiled .inf_tex will not reconstruct".into()))
    } else {
        inf_asset::decode::<TextureAsset>(bytes).map_err(|e| MaterialError::Image(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture::{texture_from_rgba8, TextureCompression, TextureImportSettings};

    /// **The asymmetric fixture.** Four distinct corner colours plus a per-texel
    /// gradient, so a transposed tile coordinate, a mirrored border or a swapped
    /// row/column stride cannot pass by symmetry. Deliberately NOT a checkerboard:
    /// a checkerboard is invariant under exactly the transforms this must catch.
    fn corners(w: u32, h: u32) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let left = x * 2 < w;
                let top = y * 2 < h;
                let base: [u8; 3] = match (left, top) {
                    (true, true) => [220, 20, 20],    // TL red
                    (false, true) => [20, 200, 20],   // TR green
                    (true, false) => [20, 20, 210],   // BL blue
                    (false, false) => [230, 210, 30], // BR yellow
                };
                // A small gradient so no two texels inside a quadrant are equal.
                let g = ((x % 8) as u8) << 1;
                v.extend_from_slice(&[
                    base[0].saturating_sub(g),
                    base[1].saturating_sub(g),
                    base[2].saturating_sub(g),
                    255,
                ]);
            }
        }
        v
    }

    fn settings(c: TextureCompression) -> TextureImportSettings {
        TextureImportSettings {
            srgb: true,
            generate_mips: true,
            compression: c,
        }
    }

    #[test]
    fn the_tile_geometry_is_a_whole_number_of_bc_blocks() {
        assert_eq!(STORED_TILE_SIZE, 136);
        assert_eq!(STORED_TILE_SIZE % 4, 0);
        assert_eq!(STORED_TILE_SIZE / 4, 34);
        assert_eq!(stored_tile_bytes(TextureFormat::Bc1), 34 * 34 * 8);
        assert_eq!(stored_tile_bytes(TextureFormat::Bc3), 34 * 34 * 16);
        assert_eq!(stored_tile_bytes(TextureFormat::Rgba8), 136 * 136 * 4);
        // Every stored tile is 16-byte aligned by its own size, so the blob
        // stride never has to pad — the property the atlas slot count rests on.
        for f in [TextureFormat::Rgba8, TextureFormat::Bc1, TextureFormat::Bc3] {
            assert_eq!(stored_tile_bytes(f) % SECTION_ALIGN as usize, 0, "{f:?}");
        }
    }

    /// The wire codes are written into shipped files: append-only, never
    /// renumbered.
    #[test]
    fn format_codes_are_freeze_pinned() {
        assert_eq!(format_code(TextureFormat::Rgba8), 0);
        assert_eq!(format_code(TextureFormat::Bc1), 1);
        assert_eq!(format_code(TextureFormat::Bc3), 2);
        for f in [TextureFormat::Rgba8, TextureFormat::Bc1, TextureFormat::Bc3] {
            assert_eq!(format_from_code(format_code(f)), Some(f));
        }
        assert_eq!(format_from_code(3), None);
    }

    #[test]
    fn the_grid_tiles_every_level_of_the_full_pyramid() {
        let img = build_tiled_texture(
            corners(320, 192),
            320,
            192,
            settings(TextureCompression::Bc1),
        )
        .unwrap();
        let r = img.reader();
        assert_eq!(r.header().width, 320);
        assert_eq!(r.header().height, 192);
        // 320,192 → 160,96 → 80,48 → 40,24 → 20,12 → 10,6 → 5,3 → 2,1 → 1,1
        assert_eq!(r.mips().len(), 9);
        assert_eq!(r.mips()[0].tiles_x, 3);
        assert_eq!(r.mips()[0].tiles_y, 2);
        assert_eq!(r.mips()[1].tiles_x, 2);
        assert_eq!(r.mips()[1].tiles_y, 1);
        for m in r.mips().iter().skip(2) {
            assert_eq!((m.tiles_x, m.tiles_y), (1, 1), "a small level is one tile");
        }
        assert_eq!(r.tiles().len(), 6 + 2 + 7);
        assert_eq!(r.header().tile_count as usize, r.tiles().len());
    }

    #[test]
    fn every_tile_blob_is_aligned_sorted_and_the_declared_size() {
        let img = build_tiled_texture(
            corners(300, 260),
            300,
            260,
            settings(TextureCompression::Bc3),
        )
        .unwrap();
        let r = img.reader();
        let mut previous: Option<(u32, u32, u32)> = None;
        for (i, e) in r.tiles().iter().enumerate() {
            assert_eq!(e.offset % SECTION_ALIGN, 0, "tile {i} is misaligned");
            assert_eq!(e.len, r.header().tile_bytes as u64);
            let key = (e.mip, e.y, e.x);
            if let Some(p) = previous {
                assert!(
                    p < key,
                    "tile directory is not sorted at {i}: {p:?} !< {key:?}"
                );
            }
            previous = Some(key);
            assert_eq!(r.tile_index(e.mip, e.x, e.y), Some(i));
            assert_eq!(r.tile(e.mip, e.x, e.y).unwrap().len(), e.len as usize);
        }
        assert_eq!(r.tile(0, 99, 0), None, "outside the grid");
        assert_eq!(r.tile(99, 0, 0), None, "outside the pyramid");
    }

    /// **The load-bearing claim**: a v2 image reconstructs, byte for byte, the
    /// v1 record the same source would have produced — for every level and every
    /// format. Everything downstream (the thumbnailer, the slicer, the PCG mask)
    /// rests on this being exact rather than approximate.
    #[test]
    fn the_v1_view_is_byte_identical_to_v1() {
        for c in [
            TextureCompression::None,
            TextureCompression::Bc1,
            TextureCompression::Bc3,
        ] {
            for (w, h) in [(320u32, 192u32), (300, 260), (64, 64), (5, 3), (1, 1)] {
                let s = settings(c);
                let v1 = texture_from_rgba8(corners(w, h), w, h, s).unwrap();
                let v2 = build_tiled_texture(corners(w, h), w, h, s).unwrap();
                let view = v2.reader().to_texture_asset().expect("reconstructs");
                assert_eq!(view, v1, "{c:?} {w}×{h}");
            }
        }
    }

    /// **The extents the matrix above does not reach** (P26.1 audit). Every one
    /// of them is a place where the tiling arithmetic could differ from itself,
    /// and none of `{320×192, 300×260, 64×64, 5×3, 1×1}` visits any of them:
    ///
    /// * `128` — exactly one tile, so *every* border texel on all four sides is
    ///   clamp padding and none of it is a neighbour;
    /// * `129` — one texel past, so the second tile is 1 texel of payload and 135
    ///   of border, and `ceil(129/4) = 33` blocks means level block 32 lives in
    ///   tile 1 (the `bx / 32` index has to land inside `tiles_x`);
    /// * `4`, `3`, `2` — levels at and *below* the 4-texel border ring, where a
    ///   stored texel's clamped source coordinate is the same one for most of
    ///   the row;
    /// * `8192×4` — a 64×1 grid whose chain runs 14 levels because the `≥1`
    ///   floor pins the short axis while `ceil` keeps halving the long one.
    ///
    /// The load-bearing assertion is byte identity with v1, because it subsumes
    /// the block-grid-offset premise the whole compatibility story rests on:
    /// stored block `b` of tile `tx` re-gathers into level block `tx·32 + b − 1`
    /// only if the offset is right — *including* at tile 0, where the stored
    /// block begins at source texel −4 and is pure clamp.
    #[test]
    fn the_v1_view_is_byte_identical_at_the_awkward_extents() {
        for (w, h) in [
            (128u32, 128u32),
            (129, 129),
            (129, 3),
            (255, 128),
            (4, 4),
            (3, 3),
            (2, 2),
            (2, 1),
            (1, 7),
        ] {
            for c in [
                TextureCompression::None,
                TextureCompression::Bc1,
                TextureCompression::Bc3,
            ] {
                let s = settings(c);
                let v1 = texture_from_rgba8(corners(w, h), w, h, s).unwrap();
                let v2 = build_tiled_texture(corners(w, h), w, h, s).unwrap();
                let r = v2.reader();
                assert_eq!(
                    r.to_texture_asset().expect("reconstructs"),
                    v1,
                    "{c:?} {w}×{h}"
                );
                assert_eq!(r.mips().len(), v1.mips.len(), "{c:?} {w}×{h}: level count");
                for (i, m) in r.mips().iter().enumerate() {
                    assert_eq!(
                        (m.width, m.height),
                        (v1.mips[i].width, v1.mips[i].height),
                        "{c:?} {w}×{h}: mip {i}"
                    );
                    assert_eq!(m.tiles_x, m.width.div_ceil(TILE_SIZE).max(1));
                    assert_eq!(m.tiles_y, m.height.div_ceil(TILE_SIZE).max(1));
                    // The last tile of the level is where the directory says.
                    assert_eq!(
                        r.tile_index(i as u32, m.tiles_x - 1, m.tiles_y - 1),
                        Some((m.first_tile + m.tile_count - 1) as usize),
                        "{c:?} {w}×{h}: mip {i}'s last tile"
                    );
                }
            }
        }

        // The non-square extreme, BC1 only: an 8192×4 RGBA8 pyramid is 10 MB of
        // tile for no coverage the BC1 one does not already give.
        let (w, h) = (8192u32, 4u32);
        let s = settings(TextureCompression::Bc1);
        let v1 = texture_from_rgba8(corners(w, h), w, h, s).unwrap();
        let r_owned = build_tiled_texture(corners(w, h), w, h, s).unwrap();
        let r = r_owned.reader();
        assert_eq!((r.mips()[0].tiles_x, r.mips()[0].tiles_y), (64, 1));
        assert_eq!(r.mips().len(), 14, "the ≥1 floor holds the short axis at 1");
        assert_eq!(r.tiles().len(), 64 + 32 + 16 + 8 + 4 + 2 + 1 + 7);
        assert_eq!(r.to_texture_asset().unwrap(), v1);
        // …and it is the shape the aspect-ratio advisory exists to warn about.
        assert_eq!(crate::texture::texture_import_advisories(w, h).len(), 1);
    }

    /// The same claim through the public door, on the bytes a consumer actually
    /// holds — and the v1 sniff in both directions.
    #[test]
    fn the_decode_door_takes_both_versions() {
        let s = settings(TextureCompression::Bc1);
        let v1 = texture_from_rgba8(corners(160, 160), 160, 160, s).unwrap();
        let v1_bytes = inf_asset::encode(&v1).unwrap();
        let v2_bytes = build_tiled_texture(corners(160, 160), 160, 160, s)
            .unwrap()
            .into_bytes();

        assert!(!is_v2(&v1_bytes), "a bincode payload must not sniff as v2");
        assert!(is_v2(&v2_bytes));
        assert_eq!(decode_texture_payload(&v1_bytes).unwrap(), v1);
        assert_eq!(decode_texture_payload(&v2_bytes).unwrap(), v1);
    }

    /// Two builds of one source are byte-identical — no timestamps, no map
    /// iteration, deterministic padding. A re-import must produce the same
    /// content hash or the import cache is a lie.
    #[test]
    fn the_encode_is_a_pure_function_of_its_source() {
        let s = settings(TextureCompression::Auto);
        let a = build_tiled_texture(corners(300, 260), 300, 260, s).unwrap();
        let b = build_tiled_texture(corners(300, 260), 300, 260, s).unwrap();
        assert_eq!(a.as_bytes(), b.as_bytes());
        // And re-parsing the bytes yields the same image.
        let c = TiledTextureImage::from_bytes(a.as_bytes().to_vec()).unwrap();
        assert_eq!(c, a);
    }

    /// **Per-corner, per-tile.** A transposed tile coordinate would put the
    /// green corner where the blue one belongs; a symmetric fixture would never
    /// notice. Asserted on the decoded PAYLOAD centre of each corner tile, so
    /// the border ring cannot mask a mistake either.
    #[test]
    fn each_corner_tile_holds_its_own_corner() {
        let (w, h) = (320u32, 192u32);
        let img =
            build_tiled_texture(corners(w, h), w, h, settings(TextureCompression::None)).unwrap();
        let r = img.reader();
        let m = r.mips()[0];
        assert_eq!((m.tiles_x, m.tiles_y), (3, 2));
        let centre = |tx: u32, ty: u32| -> [u8; 3] {
            let t = r.tile_rgba8(0, tx, ty).unwrap();
            let n = STORED_TILE_SIZE as usize;
            // The payload centre: border + 64 in both axes.
            let i = (((TILE_BORDER + 64) as usize) * n + (TILE_BORDER + 64) as usize) * 4;
            [t[i], t[i + 1], t[i + 2]]
        };
        let dominant = |c: [u8; 3]| -> usize { (0..3).max_by_key(|&i| c[i]).unwrap() };
        // (0,0) is inside the red top-left quadrant; (2,0) the green top-right;
        // (0,1) the blue bottom-left; (2,1) the yellow bottom-right (R≈G > B).
        assert_eq!(dominant(centre(0, 0)), 0, "top-left tile is not red");
        assert_eq!(dominant(centre(2, 0)), 1, "top-right tile is not green");
        assert_eq!(dominant(centre(0, 1)), 2, "bottom-left tile is not blue");
        let br = centre(2, 1);
        assert!(
            br[0] > 150 && br[1] > 150 && br[2] < 80,
            "bottom-right tile is not yellow: {br:?}"
        );
        // Anti-vacuity: the four corners are genuinely different from each other.
        let all = [centre(0, 0), centre(2, 0), centre(0, 1), centre(2, 1)];
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(all[i], all[j], "corners {i} and {j} are the same colour");
            }
        }
    }

    /// The border ring is the **neighbour's** texels, not a repeat of the tile's
    /// own edge — which is the entire reason it exists. Measured on the seam
    /// between tile (0,0) and tile (1,0) of an uncompressed image, where the
    /// comparison is exact.
    #[test]
    fn the_border_ring_carries_the_neighbouring_texels() {
        let (w, h) = (320u32, 192u32);
        let src = corners(w, h);
        let img =
            build_tiled_texture(src.clone(), w, h, settings(TextureCompression::None)).unwrap();
        let r = img.reader();
        let n = STORED_TILE_SIZE as usize;
        let t0 = r.tile_rgba8(0, 0, 0).unwrap();
        let at = |t: &[u8], i: u32, j: u32| -> [u8; 4] {
            let o = ((j as usize) * n + i as usize) * 4;
            [t[o], t[o + 1], t[o + 2], t[o + 3]]
        };
        let source = |x: u32, y: u32| -> [u8; 4] {
            let o = ((y * w + x) * 4) as usize;
            [src[o], src[o + 1], src[o + 2], src[o + 3]]
        };
        // Tile (0,0)'s RIGHT border columns are source columns 128..132 — the
        // first four columns of tile (1,0)'s payload.
        for k in 0..TILE_BORDER {
            let stored_i = TILE_BORDER + TILE_SIZE + k;
            assert_eq!(
                at(&t0, stored_i, TILE_BORDER + 40),
                source(TILE_SIZE + k, 40),
                "right border column {k} is not the neighbour's texel"
            );
        }
        // And it is NOT a repeat of the tile's own last payload column.
        assert_ne!(
            at(&t0, TILE_BORDER + TILE_SIZE, TILE_BORDER + 40),
            at(&t0, TILE_BORDER + TILE_SIZE - 1, TILE_BORDER + 40),
            "the border merely repeats the edge — nothing was baked"
        );
        // On the level's own boundary the rule is clamp: tile (0,0)'s LEFT
        // border is column 0 repeated.
        for k in 0..TILE_BORDER {
            assert_eq!(
                at(&t0, k, TILE_BORDER + 40),
                source(0, 40),
                "left border is not clamped"
            );
        }
    }

    /// A lifted v1 asset is a legal tiled image with the same geometry — and for
    /// an uncompressed one it is exact, which is what proves the lift is a
    /// re-layout rather than a re-render.
    #[test]
    fn a_v1_asset_lifts_into_the_tiled_container() {
        let s = settings(TextureCompression::None);
        let v1 = texture_from_rgba8(corners(300, 260), 300, 260, s).unwrap();
        let lifted = lift_texture_asset(&v1).unwrap();
        assert_eq!(lifted.reader().to_texture_asset().unwrap(), v1);
        // Identical to tiling the original pixels directly.
        let direct = build_tiled_texture(corners(300, 260), 300, 260, s).unwrap();
        assert_eq!(lifted.as_bytes(), direct.as_bytes());
        // Deterministic for the BC case too, even though it is a re-encode.
        let s = settings(TextureCompression::Bc1);
        let v1 = texture_from_rgba8(corners(160, 160), 160, 160, s).unwrap();
        let a = lift_texture_asset(&v1).unwrap();
        let b = lift_texture_asset(&v1).unwrap();
        assert_eq!(a.as_bytes(), b.as_bytes());
        assert_eq!(a.reader().header().format, TextureFormat::Bc1);
    }

    #[test]
    fn parse_refuses_a_doctored_payload() {
        let img = build_tiled_texture(
            corners(160, 160),
            160,
            160,
            settings(TextureCompression::Bc1),
        )
        .unwrap();
        let good = img.into_bytes();

        assert_eq!(
            TiledTextureImage::from_bytes(vec![0u8; 8]).unwrap_err(),
            TiledTextureError::TooShort
        );
        let mut bad = good.clone();
        bad[3] = b'X';
        assert_eq!(
            TiledTextureImage::from_bytes(bad).unwrap_err(),
            TiledTextureError::BadMagic
        );
        let mut bad = good.clone();
        bad[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert_eq!(
            TiledTextureImage::from_bytes(bad).unwrap_err(),
            TiledTextureError::SchemaTooNew {
                found: 99,
                current: 2
            }
        );
        // A tile offset dragged off its 16-byte boundary.
        let mut bad = good.clone();
        let dir = u64::from_le_bytes(bad[56..64].try_into().unwrap()) as usize;
        let off = u64::from_le_bytes(bad[dir + 16..dir + 24].try_into().unwrap());
        bad[dir + 16..dir + 24].copy_from_slice(&(off + 1).to_le_bytes());
        assert_eq!(
            TiledTextureImage::from_bytes(bad).unwrap_err(),
            TiledTextureError::TileOutOfBounds { index: 0 }
        );
        // A mip grid that does not tile its own extent.
        let mut bad = good.clone();
        bad[HEADER_LEN as usize + 8..HEADER_LEN as usize + 12].copy_from_slice(&7u32.to_le_bytes());
        assert!(matches!(
            TiledTextureImage::from_bytes(bad).unwrap_err(),
            TiledTextureError::Malformed(_)
        ));
        // An unknown format code.
        let mut bad = good.clone();
        bad[20..24].copy_from_slice(&9u32.to_le_bytes());
        assert!(matches!(
            TiledTextureImage::from_bytes(bad).unwrap_err(),
            TiledTextureError::Malformed(_)
        ));
        // Truncation past the directory.
        let mut bad = good.clone();
        bad.truncate(good.len() - 16);
        assert!(TiledTextureImage::from_bytes(bad).is_err());
        // The good one still parses (anti-vacuity: the doctoring is what failed).
        assert!(TiledTextureImage::from_bytes(good).is_ok());
    }

    /// **The payload must contain the tiles it declares** (P26.1 audit).
    ///
    /// Bounding each tile's own range is a weaker claim than it looks: every
    /// entry may point at the *same* blob, and then a 32-byte directory entry
    /// describes 73 984 B of tile. Measured before the check existed: 16 384
    /// aliased entries in a 584 KiB payload made `level_bytes` ask the allocator
    /// for 1 GiB — 1 794× — from a file the thumbnailer opens the moment it
    /// appears in the Content Drawer. Three lies, all now refused.
    #[test]
    fn parse_refuses_a_payload_that_does_not_hold_its_own_tiles() {
        let good = build_tiled_texture(
            corners(160, 160),
            160,
            160,
            settings(TextureCompression::Bc1),
        )
        .unwrap()
        .into_bytes();
        let hdr_u64 = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());

        // (a) `total_len` that disagrees with the bytes actually present — in
        // either direction, so neither an over- nor an under-claim survives.
        for lie in [1u64 << 40, 16] {
            let mut bad = good.clone();
            bad[72..80].copy_from_slice(&lie.to_le_bytes());
            assert!(
                matches!(
                    TiledTextureImage::from_bytes(bad).unwrap_err(),
                    TiledTextureError::Malformed(_)
                ),
                "total_len = {lie} was accepted"
            );
        }
        // (b) Trailing bytes. The payload is a whole file / a whole pack entry,
        // never a prefix of one, so anything past `total_len` means this is not
        // the image it says it is.
        let mut bad = good.clone();
        bad.extend_from_slice(&[0xAB; 4096]);
        assert!(matches!(
            TiledTextureImage::from_bytes(bad).unwrap_err(),
            TiledTextureError::Malformed(_)
        ));
        // (c) Every entry aliased onto the first blob. The labels are still
        // perfectly ordered, which is exactly why the label check cannot see it.
        let mut bad = good.clone();
        let dir = hdr_u64(&bad, 56) as usize;
        let base = hdr_u64(&bad, 64);
        let n = u32::from_le_bytes(bad[40..44].try_into().unwrap()) as usize;
        for i in 0..n {
            let o = dir + i * 32 + 16;
            bad[o..o + 8].copy_from_slice(&base.to_le_bytes());
        }
        assert_eq!(
            TiledTextureImage::from_bytes(bad).unwrap_err(),
            TiledTextureError::TileOutOfBounds { index: 1 },
            "a self-aliased tile directory was accepted"
        );
        // Anti-vacuity: the undoctored payload still parses.
        assert!(TiledTextureImage::from_bytes(good).is_ok());
    }

    /// The header's extent and mip 0's are two fields carrying one fact, and
    /// nothing made them agree: `to_texture_asset` reads the header's while
    /// `level_bytes` reads the directory's, so a payload could hand back a
    /// `TextureAsset` whose `width` is not the width of `mips[0]` — which is
    /// what the sprite-sheet slicer cuts cells with.
    #[test]
    fn parse_refuses_a_header_extent_that_is_not_mip_zero() {
        let good = build_tiled_texture(
            corners(160, 160),
            160,
            160,
            settings(TextureCompression::Bc1),
        )
        .unwrap()
        .into_bytes();
        for (off, value) in [(12usize, 65535u32), (16, 65535), (12, 159), (16, 161)] {
            let mut bad = good.clone();
            bad[off..off + 4].copy_from_slice(&value.to_le_bytes());
            assert!(
                matches!(
                    TiledTextureImage::from_bytes(bad).unwrap_err(),
                    TiledTextureError::Malformed(_)
                ),
                "extent field at {off} set to {value} was accepted"
            );
        }
        assert!(TiledTextureImage::from_bytes(good).is_ok());
    }

    /// Truncation at **every section boundary**, not only past the end. Each one
    /// is a different arm of `parse` reaching for bytes that are not there, and
    /// a payload arrives from a file or a pack mapping — both of which can be
    /// short.
    #[test]
    fn parse_refuses_truncation_at_every_section_boundary() {
        let good = build_tiled_texture(
            corners(300, 260),
            300,
            260,
            settings(TextureCompression::Bc3),
        )
        .unwrap()
        .into_bytes();
        let tile_dir = u64::from_le_bytes(good[56..64].try_into().unwrap()) as usize;
        let tile_base = u64::from_le_bytes(good[64..72].try_into().unwrap()) as usize;
        let tile_bytes = u32::from_le_bytes(good[44..48].try_into().unwrap()) as usize;
        for (what, len) in [
            ("nothing at all", 0usize),
            ("shorter than the magic", 4),
            ("the magic alone", 8),
            ("one byte short of the header", HEADER_LEN as usize - 1),
            ("the header alone", HEADER_LEN as usize),
            (
                "mid mip directory",
                HEADER_LEN as usize + MIP_ENTRY_LEN as usize / 2,
            ),
            ("the mip directory alone", tile_dir),
            ("mid tile directory", tile_dir + TILE_ENTRY_LEN as usize + 8),
            ("the directories alone", tile_base),
            ("one tile short", good.len() - tile_bytes),
            ("one byte short", good.len() - 1),
        ] {
            let mut bad = good.clone();
            bad.truncate(len);
            assert!(
                TiledTextureImage::from_bytes(bad).is_err(),
                "{what} ({len} B) parsed as a whole image"
            );
        }
        assert!(TiledTextureImage::from_bytes(good).is_ok());
    }

    /// The tail of the pyramid is measured rather than assumed — the honest cost
    /// of a uniform page size, recorded where a future packing pass will find it.
    #[test]
    fn the_uniform_tail_cost_is_what_the_docs_claim() {
        let img = build_tiled_texture(
            corners(256, 256),
            256,
            256,
            settings(TextureCompression::Bc1),
        )
        .unwrap();
        let r = img.reader();
        // 256² → 4 tiles at mip 0, 1 at mip 1, then 7 single-tile tail levels.
        assert_eq!(r.mips().len(), 9);
        let tail: usize = r.mips()[2..].iter().map(|m| m.tile_count as usize).sum();
        assert_eq!(tail, 7);
        assert_eq!(tail * stored_tile_bytes(TextureFormat::Bc1), 64_736);
    }

    /// **A `TileCoord` does not sort the way the payload does** (P26.1 audit).
    ///
    /// The derived `Ord` is `(mip, x, y)` because that is the field order; the
    /// directory is sorted `(mip, y, x)` because that is row-major. A residency
    /// table keyed on `TileCoord` therefore iterates in an order that is *not*
    /// the order its bytes lie in, which turns one sequential read into a
    /// scatter — a quiet cost with no symptom. Recorded here rather than fixed
    /// by renaming a field, and [`TileCoord::payload_order`] is the key to sort
    /// on when the order that matters is where the bytes are.
    #[test]
    fn the_tile_coord_order_is_not_the_payload_order() {
        let img = build_tiled_texture(
            corners(320, 192),
            320,
            192,
            settings(TextureCompression::Bc1),
        )
        .unwrap();
        let r = img.reader();
        let coords: Vec<TileCoord> = r
            .tiles()
            .iter()
            .map(|e| TileCoord::new(e.mip, e.x, e.y))
            .collect();
        // The directory is in payload order by construction (asserted in
        // `every_tile_blob_is_aligned_sorted_and_the_declared_size`).
        let mut derived = coords.clone();
        derived.sort();
        assert_ne!(
            derived, coords,
            "the derived order happens to match the payload order on this grid — \
             pick a grid where it does not, or the trap this pins has moved"
        );
        // …and sorting on `payload_order` reproduces the directory exactly.
        let mut by_payload = coords.clone();
        by_payload.sort_by_key(|c| c.payload_order());
        assert_eq!(by_payload, coords);
        // Every address round-trips to its own directory index.
        for (i, c) in coords.iter().enumerate() {
            assert_eq!(r.tile_index(c.mip, c.x, c.y), Some(i));
            assert_eq!(r.tile_at(*c), r.tile(c.mip, c.x, c.y));
        }
    }

    /// **What the container costs on disk, divided by something** (P26.1 audit).
    ///
    /// The tail cost above is a real number with no denominator, and the ship-size
    /// note on `PackWriter::compresses_kind` names the wrong cost: it blames the
    /// lost zstd frame and says "BC payloads barely compress", which is true and
    /// is not where the bytes go. The bytes go into the *border ring and the
    /// uniform tail*, and they are paid by BC textures too.
    ///
    /// Measured here so a future packing pass has a target and a regression has a
    /// tripwire. The shape of the curve is the finding: the overhead is a
    /// **small-texture** phenomenon and it is severe there — a 128² BC1 texture,
    /// the size of a UI icon or a decal, is nearly seven times its v1 payload,
    /// because eight of its nine levels are one 136² tile each. At 1k and up it
    /// is the ~1.2× the border ring alone implies (136²/128² = 1.13, plus the
    /// tail).
    #[test]
    fn the_container_costs_more_bytes_than_v1_and_this_is_how_many() {
        let measure = |w: u32, h: u32, c: TextureCompression| -> (usize, usize) {
            let s = settings(c);
            let v1 =
                inf_asset::encode(&texture_from_rgba8(corners(w, h), w, h, s).unwrap()).unwrap();
            let v2 = build_tiled_texture(corners(w, h), w, h, s).unwrap();
            (v1.len(), v2.as_bytes().len())
        };
        // Pinned exactly: a level's byte count is a function of its extent alone,
        // so these do not depend on the fixture's pixels.
        assert_eq!(measure(128, 128, TextureCompression::Bc1), (10_972, 74_624));
        assert_eq!(
            measure(256, 256, TextureCompression::Bc1),
            (43_753, 111_776)
        );
        assert_eq!(
            measure(1024, 1024, TextureCompression::Bc1),
            (699_135, 854_240)
        );
        assert_eq!(
            measure(1024, 1024, TextureCompression::None),
            (5_592_483, 6_809_952)
        );
        // …and the claim the numbers are here to support, as a claim.
        let ratio = |w, h, c| {
            let (a, b) = measure(w, h, c);
            b as f64 / a as f64
        };
        assert!(
            ratio(128, 128, TextureCompression::Bc1) > 6.5,
            "the small-texture tail got cheaper — re-write the ledger, do not \
             re-bless the number"
        );
        assert!(ratio(2048, 2048, TextureCompression::Bc1) < 1.2);
        // The cost is NOT the lost zstd frame: an uncompressed-format texture and
        // a BC one grow by the same factor, because both pay the same geometry.
        let (a, b) = (
            ratio(1024, 1024, TextureCompression::Bc1),
            ratio(1024, 1024, TextureCompression::None),
        );
        assert!((a - b).abs() < 0.01, "{a} vs {b}");
    }
}
