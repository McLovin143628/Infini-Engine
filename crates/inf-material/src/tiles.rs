//! The `.inf_tex` **v2 tiled container** (P26.1): a texture leaves its single
//! bincode blob for a random-access, tile-at-a-time image.
//!
//! ```text
//! ┌ header (128 B, little-endian) ────────────────────────────────────┐
//! │  magic        [u8; 8]  b"INFVTEX\0"                               │
//! │  schema_ver   u32      the LOWEST version this payload needs      │
//! │  width · height  u32   the VIRTUAL extent (mip 0)                 │
//! │  format       u32      0 Rgba8·1 Bc1·2 Bc3·3 Bc5·4 RGBA16F (v3)  │
//! │  flags        u32      bit 0 = srgb                               │
//! │  tile_size    u32      PAYLOAD texels per tile side (128)         │
//! │  border       u32      border texels per side (4)                 │
//! │  mip_count · tile_count  u32                                      │
//! │  tile_bytes   u32      bytes of ONE stored tile (uniform)         │
//! │  mip_dir_off · tile_dir_off · tile_base · total_len  (u64 each)   │
//! │  reserved     [u8; 48] zeros (room for v4 without a re-length)    │
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
//! Like `.inf_vmesh` and `.inf_terrain`, the bytes on disk and in a `.ipack`
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

/// **The read half of this container lives in [`inf_vt::container`]** (P26.3) and
/// is re-exported here, so every call site that named `inf_material::tiles::…`
/// still does.
///
/// The split is not a tidy-up: a shipped player samples virtual textures, so it
/// must *read* tiles, and **it does not link this crate** — `inf-material` is a
/// *dev* dependency of `inf-render`. Measured rather than asserted:
/// `cargo tree -p inf-player -e normal` shows `inf-vt` and no `inf-material`.
///
/// What that buys is that this crate's own code — the BC *encoder*, the
/// naga-validating material compiler, the `image`-backed importer — stays out of
/// a shipped build. It is **not** that `image` and `naga` are absent from a
/// player: they are there anyway, `image` through `gltf`/`inf-terrain` and `naga`
/// through `wgpu-core`. The claim worth defending is the dependency *direction*,
/// which makes "`inf-vt` reaches for the importer" a compile error rather than a
/// review question.
///
/// The writer below — which needs the mip chain, the format choice and the BC
/// encoder — stays. See
/// [`inf_vt::container`]'s module docs for the table of what went and what
/// stayed, and for the one deliberate deviation (the BC *decoder* went with the
/// reader; the encoder did not).
pub use inf_vt::container::{
    align_up, format_code, format_from_code, is_v2, min_schema_version, parse, TexMipEntry,
    TexTileEntry, TiledTextureError, TiledTextureHeader, TiledTextureReader, TiledTextureView,
    HEADER_LEN, MIP_ENTRY_LEN, SECTION_ALIGN, STORED_TILE_SIZE, TEX_ASSET_MAGIC,
    TEX_ASSET_SCHEMA_VERSION, TILE_BORDER, TILE_ENTRY_LEN, TILE_SIZE,
};

type Result<T> = std::result::Result<T, TiledTextureError>;

/// Bytes one **stored tile** occupies in `format` — uniform across the whole
/// image, which is what makes a physical page a fixed-size slot.
///
/// The `TextureFormat`-flavoured face of [`inf_vt::stored_tile_bytes`]: the
/// arithmetic is one function, and this is the spelling the importer speaks in.
pub fn stored_tile_bytes(format: TextureFormat) -> usize {
    inf_vt::stored_tile_bytes(format.into())
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

/// Lay an **RGBA16F** source (four LE halves a texel) out as a tiled image
/// (Wave T) — the float twin of [`build_tiled_texture`].
///
/// The format is `Rgba16F` regardless of `settings.compression`, for the reason
/// `crate::texture::texture_from_rgba16f` states: no block format in this
/// container can hold a value outside `[0, 1]`, so honouring a compression
/// request would silently undo the whole point.
pub fn build_tiled_texture_rgba16f(
    halfs: Vec<u8>,
    width: u32,
    height: u32,
    settings: crate::texture::TextureImportSettings,
) -> std::result::Result<TiledTextureImage, MaterialError> {
    let levels = crate::texture::rgba16f_mip_chain(halfs, width, height, settings.generate_mips)?;
    Ok(tile_levels(&levels, TextureFormat::Rgba16F, false))
}

/// Decode `bytes` and lay it out as a tiled image, taking the **float** path
/// when the source is float and `settings.hdr` asks for it (Wave T).
///
/// The one door an importer should call: it is where `source_is_float` and
/// `hdr` meet, so no call site has to remember to check both. Returns the image
/// and the [`crate::texture::hdr_import_advisory`] the decision earned, if any —
/// a value rather than a log line, on the house advisory doctrine.
pub fn build_tiled_texture_from_bytes(
    bytes: &[u8],
    settings: crate::texture::TextureImportSettings,
) -> std::result::Result<(TiledTextureImage, Option<String>), MaterialError> {
    let float_source = crate::texture::source_is_float(bytes);
    let kept = float_source && settings.hdr;
    let advisory = crate::texture::hdr_import_advisory(float_source, kept);
    let image = if kept {
        let (halfs, w, h) = crate::texture::decode_image_rgba16f(bytes)?;
        build_tiled_texture_rgba16f(halfs, w, h, settings)?
    } else {
        let (rgba, w, h) = crate::texture::decode_image_rgba8(bytes)?;
        build_tiled_texture(rgba, w, h, settings)?
    };
    Ok((image, advisory))
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
        // **A float level is carried, never decoded** (Wave T). `level_rgba8`
        // clamps to `[0, 1]` for previews; taking that door here would make a
        // lift the very flattening the float format exists to end.
        let texels = if tex.format.is_float() {
            mip.data.clone()
        } else {
            tex.level_rgba8(i)
                .ok_or_else(|| MaterialError::Image(format!("mip {i} will not decode")))?
        };
        let want = tex.format.texel_bytes().unwrap_or(4);
        if texels.len() < (mip.width as usize * mip.height as usize * want) {
            return Err(MaterialError::Image(format!("mip {i} is truncated")));
        }
        levels.push((mip.width, mip.height, texels));
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
    // **The LOWEST version that can express this format, not the newest this
    // build knows** (Wave T). A BC1 albedo therefore still stamps v2 and is
    // byte-identical to what it was before Wave T — no content hash moves, no
    // import cache is invalidated, no `.ipack` stops reproducing — while a
    // BC5 or RGBA16F payload stamps v3 and is refused by name by a build that
    // cannot read its format code. See `inf_vt::container::min_schema_version`.
    out.extend_from_slice(&min_schema_version(format.into()).to_le_bytes());
    out.extend_from_slice(&levels[0].0.to_le_bytes());
    out.extend_from_slice(&levels[0].1.to_le_bytes());
    out.extend_from_slice(&format_code(format.into()).to_le_bytes());
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
    // Bytes a source texel occupies. Every format but the float one gathers from
    // an RGBA8 level; `Rgba16F` gathers from four halves. The clamp rule, the
    // border ring and the tile order are identical either way — which is why
    // this is a stride and not a second tiler.
    let src_texel = format.texel_bytes().unwrap_or(4);
    for (mip, m) in mips.iter().enumerate() {
        let (lw, lh, ref rgba) = levels[mip];
        for y in 0..m.tiles_y {
            for x in 0..m.tiles_x {
                let gathered = gather_stored_tile(rgba, lw, lh, x, y, src_texel);
                let blob = match format {
                    TextureFormat::Rgba8 | TextureFormat::Rgba16F => gathered,
                    TextureFormat::Bc1 => bc::compress_bc1(&gathered, stored, stored),
                    TextureFormat::Bc3 => bc::compress_bc3(&gathered, stored, stored),
                    TextureFormat::Bc5 => bc::compress_bc5(&gathered, stored, stored),
                    TextureFormat::Bc7 => bc::compress_bc7(&gathered, stored, stored),
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
fn gather_stored_tile(rgba: &[u8], lw: u32, lh: u32, tx: u32, ty: u32, texel: usize) -> Vec<u8> {
    let stored = STORED_TILE_SIZE as usize;
    let mut out = vec![0u8; stored * stored * texel];
    let x0 = tx * TILE_SIZE;
    let y0 = ty * TILE_SIZE;
    for j in 0..stored {
        let sy = (y0 as i64 + j as i64 - TILE_BORDER as i64).clamp(0, lh as i64 - 1) as u32;
        for i in 0..stored {
            let sx = (x0 as i64 + i as i64 - TILE_BORDER as i64).clamp(0, lw as i64 - 1) as u32;
            let si = (sy as usize * lw as usize + sx as usize) * texel;
            let di = (j * stored + i) * texel;
            out[di..di + texel].copy_from_slice(&rgba[si..si + texel]);
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

// ── the v1 view ─────────────────────────────────────────────────────────────

/// The **v1-facing** half of a [`TiledTextureReader`]: whole levels, and the
/// `TextureAsset` record every consumer written before P26.1 already reads.
///
/// An extension trait rather than inherent methods because the reader itself now
/// lives in [`inf_vt::container`] (P26.3) and `TextureAsset` — with its `image`
/// decode path — must not follow it down there. Nothing else moved: the two
/// bodies are the ones P26.1 wrote, verbatim.
pub trait TiledTextureExt {
    /// One whole mip level, **borders stripped**, in the stored format — the
    /// bytes v1 keeps in `TextureMip::data`.
    fn level_bytes(&self, mip: u32) -> Option<Vec<u8>>;

    /// The whole texture rebuilt as the v1 record.
    fn to_texture_asset(&self) -> Option<TextureAsset>;
}

impl<B: AsRef<[u8]>> TiledTextureExt for TiledTextureReader<B> {
    /// A re-gather, not a re-encode: a payload block of a tile *is* a block of
    /// the level (see the module docs), so this copies block bytes straight
    /// across for BC and texel rows for RGBA8.
    fn level_bytes(&self, mip: u32) -> Option<Vec<u8>> {
        let m = *self.mips().get(mip as usize)?;
        let stored = self.header().stored_tile_size();
        let border = self.header().border;
        let tile_size = self.header().tile_size;
        let fmt = TextureFormat::from(self.header().format);
        match fmt.texel_bytes() {
            // The uncompressed arm, at whichever texel width the format has —
            // 4 for RGBA8, 8 for RGBA16F (Wave T). One loop, one stride.
            Some(texel) => {
                let mut out = vec![0u8; (m.width as usize) * (m.height as usize) * texel];
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
                                let si = (sj * stored as usize + (i + border) as usize) * texel;
                                let di = (dy as usize * m.width as usize + dx as usize) * texel;
                                out[di..di + texel].copy_from_slice(&blob[si..si + texel]);
                            }
                        }
                    }
                }
                Some(out)
            }
            None => {
                let block = if fmt == TextureFormat::Bc1 {
                    8usize
                } else {
                    16
                };
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

    /// Byte-identical to what v1 import would have written from the same source
    /// (the block-grid property in the module docs), which is what lets the
    /// thumbnailer, the sprite-sheet slicer and the PCG mask reader keep their
    /// existing code with one call changed.
    fn to_texture_asset(&self) -> Option<TextureAsset> {
        let mips = (0..self.mips().len() as u32)
            .map(|l| {
                let m = self.mips()[l as usize];
                Some(TextureMip {
                    width: m.width,
                    height: m.height,
                    data: self.level_bytes(l)?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        Some(TextureAsset {
            schema_version: TextureAsset::CURRENT_VERSION,
            width: self.header().width,
            height: self.header().height,
            format: self.header().format.into(),
            srgb: self.header().srgb,
            mips,
        })
    }
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
            TextureFormat::Bc5 => inf_vt::PageFormat::Bc5,
            TextureFormat::Rgba16F => inf_vt::PageFormat::Rgba16F,
            TextureFormat::Bc7 => inf_vt::PageFormat::Bc7,
        }
    }
}

/// The inverse, needed since P26.3 put the container's header in `inf-vt`: the
/// header now carries a [`inf_vt::PageFormat`] (the crate that reads it knows no
/// `TextureFormat`), and the v1 view above has to name the format the importer
/// speaks in.
///
/// **This is a total bijection and the enums must stay one**: `PageFormat` says
/// what a *pool* is and `TextureFormat` what a *file* is, and on the transcode
/// arm they legitimately differ — but the three storage formats they enumerate
/// are the same three, which is what makes both directions total. Pinned by
/// [`tests::the_two_format_enums_are_one_bijection`].
impl From<inf_vt::PageFormat> for TextureFormat {
    fn from(f: inf_vt::PageFormat) -> Self {
        match f {
            inf_vt::PageFormat::Rgba8 => TextureFormat::Rgba8,
            inf_vt::PageFormat::Bc1 => TextureFormat::Bc1,
            inf_vt::PageFormat::Bc3 => TextureFormat::Bc3,
            inf_vt::PageFormat::Bc5 => TextureFormat::Bc5,
            inf_vt::PageFormat::Rgba16F => TextureFormat::Rgba16F,
            inf_vt::PageFormat::Bc7 => TextureFormat::Bc7,
        }
    }
}

/// Every storage format, for a sweep to iterate — one list, so a format added to
/// the enum and forgotten here fails the bijection gate rather than being missed
/// by every test at once.
pub const ALL_TEXTURE_FORMATS: [TextureFormat; 6] = [
    TextureFormat::Rgba8,
    TextureFormat::Bc1,
    TextureFormat::Bc3,
    TextureFormat::Bc5,
    TextureFormat::Rgba16F,
    TextureFormat::Bc7,
];

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
            hdr: false,
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
    ///
    /// Reached here through the [`TextureFormat`] → [`inf_vt::PageFormat`]
    /// conversion the importer speaks in, which is the whole of what P26.3's move
    /// changed at this call site.
    #[test]
    fn format_codes_are_freeze_pinned() {
        assert_eq!(format_code(TextureFormat::Rgba8.into()), 0);
        assert_eq!(format_code(TextureFormat::Bc1.into()), 1);
        assert_eq!(format_code(TextureFormat::Bc3.into()), 2);
        // Wave T APPENDED 3 and 4 — the three codes above did not move, which is
        // the whole content of "freeze-pinned".
        assert_eq!(format_code(TextureFormat::Bc5.into()), 3);
        assert_eq!(format_code(TextureFormat::Rgba16F.into()), 4);
        // …and wave IASSET2 APPENDED 5, leaving the five above where they were.
        assert_eq!(format_code(TextureFormat::Bc7.into()), 5);
        for f in ALL_TEXTURE_FORMATS {
            assert_eq!(format_from_code(format_code(f.into())), Some(f.into()));
        }
        assert_eq!(format_from_code(6), None);
        // …and the version a format demands is the version the writer stamps.
        // A build that predates Wave T refuses codes 3 and 4 by *version*, not
        // by guessing at the code — and every older format still stamps v2, so
        // no existing texture's bytes moved. Wave IASSET2's v4 window says the
        // same thing about code 5.
        for f in [TextureFormat::Rgba8, TextureFormat::Bc1, TextureFormat::Bc3] {
            assert_eq!(min_schema_version(f.into()), 2, "{f:?}");
        }
        for f in [TextureFormat::Bc5, TextureFormat::Rgba16F] {
            assert_eq!(min_schema_version(f.into()), 3, "{f:?}");
        }
        assert_eq!(min_schema_version(TextureFormat::Bc7.into()), 4);
    }

    /// **The v3 bump moved no existing byte** (Wave T).
    ///
    /// The one claim that makes a container version cheap: a texture in a
    /// pre-Wave-T format is the same file it was, version word included, so no
    /// content hash moves, no import cache is invalidated and no `.ipack`
    /// stops reproducing. Asserted on the header word rather than on a memory of
    /// the decision — and on both axes, because "the version is 2" and "a v3
    /// format actually stamps 3" are two different mistakes.
    #[test]
    fn a_pre_wave_t_format_still_writes_a_v2_container() {
        for c in [
            TextureCompression::None,
            TextureCompression::Bc1,
            TextureCompression::Bc3,
        ] {
            let img = build_tiled_texture(corners(160, 160), 160, 160, settings(c)).unwrap();
            let v = u32::from_le_bytes(img.as_bytes()[8..12].try_into().unwrap());
            assert_eq!(v, 2, "{c:?} stamped v{v}; every pre-Wave-T byte must stand");
        }
        let bc5 = build_tiled_texture(
            corners(160, 160),
            160,
            160,
            settings(TextureCompression::Bc5),
        )
        .unwrap();
        assert_eq!(
            u32::from_le_bytes(bc5.as_bytes()[8..12].try_into().unwrap()),
            3
        );
        let hdr = build_tiled_texture_rgba16f(
            vec![0u8; 160 * 160 * 8],
            160,
            160,
            settings(TextureCompression::None),
        )
        .unwrap();
        assert_eq!(
            u32::from_le_bytes(hdr.as_bytes()[8..12].try_into().unwrap()),
            3
        );
    }

    /// **A v2 stamp over a v3 format code is refused** — the version and the
    /// format lane are one contract (Wave T).
    ///
    /// Without this the file would be one no build in the world but this one
    /// could read: every pre-Wave-T reader refuses the code, and this reader
    /// would accept a version claim its own writer would never make.
    #[test]
    fn a_v3_format_stamped_v2_is_refused() {
        let mut bad = build_tiled_texture(
            corners(160, 160),
            160,
            160,
            settings(TextureCompression::Bc5),
        )
        .unwrap()
        .into_bytes();
        assert!(TiledTextureImage::from_bytes(bad.clone()).is_ok());
        bad[8..12].copy_from_slice(&2u32.to_le_bytes());
        let e = TiledTextureImage::from_bytes(bad).unwrap_err();
        assert!(
            matches!(&e, TiledTextureError::Malformed(m) if m.contains("needs container v3")),
            "{e:?}"
        );
    }

    /// **The downgrade bless**: every container this build can write, read back
    /// through the public door, for every format including the two Wave T added.
    #[test]
    fn every_format_round_trips_through_the_container() {
        for c in [
            TextureCompression::None,
            TextureCompression::Bc1,
            TextureCompression::Bc3,
            TextureCompression::Bc5,
            TextureCompression::Auto,
        ] {
            for (w, h) in [(320u32, 192u32), (129, 129), (4, 4), (1, 1)] {
                let img = build_tiled_texture(corners(w, h), w, h, settings(c)).unwrap();
                let r = img.reader();
                let v1 = r.to_texture_asset().expect("reconstructs");
                assert_eq!((v1.width, v1.height), (w, h), "{c:?} {w}×{h}");
                assert_eq!(
                    v1.mips[0].data.len(),
                    v1.format.level_size(w, h),
                    "{c:?} {w}×{h}: level 0 is not its own declared size"
                );
                // …and the payload decodes through the ONE door every consumer
                // that wants whole levels uses.
                assert_eq!(decode_texture_payload(img.as_bytes()).unwrap(), v1);
            }
        }
        // The float container, whose levels are 8 bytes a texel.
        let halfs: Vec<u8> = (0..(64 * 64 * 4))
            .flat_map(|i| ((i % 251) as u16).to_le_bytes())
            .collect();
        let img =
            build_tiled_texture_rgba16f(halfs, 64, 64, settings(TextureCompression::None)).unwrap();
        let v1 = img.reader().to_texture_asset().expect("reconstructs");
        assert_eq!(v1.format, TextureFormat::Rgba16F);
        assert_eq!(v1.mips[0].data.len(), 64 * 64 * 8);
        assert_eq!(v1.mips.len(), 7);
    }

    /// The two format enums are **one bijection**, both ways and totally — the
    /// property P26.3's split rests on, since the container's header now carries
    /// a [`inf_vt::PageFormat`] and the v1 view has to name a [`TextureFormat`].
    ///
    /// A `From` in one direction alone would let the pair drift the day either
    /// enum grows a variant: the new one would convert *out* and have nowhere to
    /// come back to, and the `impl` that had to be written would be the notice.
    #[test]
    fn the_two_format_enums_are_one_bijection() {
        for f in [TextureFormat::Rgba8, TextureFormat::Bc1, TextureFormat::Bc3] {
            let page: inf_vt::PageFormat = f.into();
            assert_eq!(TextureFormat::from(page), f, "{f:?} did not round-trip");
        }
        for p in [
            inf_vt::PageFormat::Rgba8,
            inf_vt::PageFormat::Bc1,
            inf_vt::PageFormat::Bc3,
        ] {
            let f: TextureFormat = p.into();
            assert_eq!(inf_vt::PageFormat::from(f), p, "{p:?} did not round-trip");
        }
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
        // …and it is the shape BOTH shape advisories exist to warn about: 2048:1
        // is a strip that stores mostly clamp padding (P26.1), and its levels
        // are almost all smaller than one tile, so it also pays the uniform
        // page's tail cost (P26.5). Two sentences, and they are about two
        // different costs of the same extent.
        let said = crate::texture::texture_import_advisories(w, h);
        assert_eq!(said.len(), 2, "{said:?}");
        assert!(said[0].contains("2048:1"), "{said:?}");
        assert!(said[1].contains("tiled .inf_tex"), "{said:?}");
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
        assert_eq!(a.reader().header().format, TextureFormat::Bc1.into());
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
                current: TEX_ASSET_SCHEMA_VERSION
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
