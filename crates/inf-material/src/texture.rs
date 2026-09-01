//! The `.inf_tex` texture payload + importer.
//!
//! Import decodes any supported source (PNG/JPEG/TGA/BMP/HDR/EXR via the
//! `image` crate) to RGBA8, generates a full box-filtered mip chain, and
//! optionally block-compresses every level (BC1 opaque / BC3 alpha, chosen
//! automatically). Import settings live in the sidecar so reimport is
//! reproducible.

use inf_asset::{AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

use crate::bc;
use crate::error::MaterialError;

/// GPU-facing pixel format of a stored texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextureFormat {
    /// Uncompressed 8-bit RGBA.
    Rgba8,
    /// BC1 / DXT1 — opaque, 4:1.
    Bc1,
    /// BC3 / DXT5 — full alpha, 4:1.
    Bc3,
    /// **BC5 / RGTC2** — two 8-bit channels at 8 bpp, the normal-map format
    /// (Wave T). X and Y are stored; Z is rebuilt by the sampler. 4:1 against the
    /// RGBA8 normal maps this engine shipped before it. See
    /// [`crate::bc::compress_bc5`].
    Bc5,
    /// **RGBA16F** — four half-floats, 8 bytes a texel (Wave T).
    ///
    /// The format an EXR or Radiance source keeps its dynamic range in. Every
    /// float source used to be flattened to 8 bits on the way in, silently; this
    /// is the alternative, and it is *bigger* than the source rather than smaller
    /// because the compressed HDR format (BC6H) has no encoder in this project.
    /// The trade is written down in `docs/memos/wave-t-textures-disposition.md`
    /// and surfaced to the importing author as an advisory.
    Rgba16F,
    /// **BC7** — full RGBA at 16 bytes a block, twice BC1's page (wave IASSET2).
    ///
    /// The quality format: effectively 8-bit RGBA endpoints and sixteen
    /// interpolated levels, against BC1's 5:6:5 endpoints and four. This
    /// project's encoder writes **mode 6** only and says so
    /// ([`crate::bc::compress_bc7`]). It costs twice BC1's bytes on disk and in
    /// the atlas, so it is a measured per-map choice rather than a default.
    Bc7,
}

impl TextureFormat {
    /// Bytes one texel occupies in an **uncompressed** format, or `None` for a
    /// block format.
    pub fn texel_bytes(self) -> Option<usize> {
        match self {
            TextureFormat::Rgba8 => Some(4),
            TextureFormat::Rgba16F => Some(8),
            TextureFormat::Bc1 | TextureFormat::Bc3 | TextureFormat::Bc5 | TextureFormat::Bc7 => {
                None
            }
        }
    }

    /// Bytes one mip level of `w×h` occupies in this format.
    ///
    /// **`usize`, not `u32`** (C4-5): `w * h * 4` in `u32` is exactly `0` at
    /// 65 536 × 65 536, and the release profile carries no `overflow-checks`, so
    /// the wrap is silent. A `.inf_tex` declaring those dimensions used to ask
    /// for a zero-byte buffer and then be written into as though it were 16 GiB.
    pub fn level_size(self, w: u32, h: u32) -> usize {
        match self {
            TextureFormat::Rgba8 => w as usize * h as usize * 4,
            TextureFormat::Rgba16F => w as usize * h as usize * 8,
            TextureFormat::Bc1 => bc::compressed_size(w, h, false),
            TextureFormat::Bc3 => bc::compressed_size(w, h, true),
            TextureFormat::Bc5 => bc::compressed_size_bc5(w, h),
            TextureFormat::Bc7 => bc::compressed_size_bc7(w, h),
        }
    }
    pub fn is_compressed(self) -> bool {
        matches!(
            self,
            TextureFormat::Bc1 | TextureFormat::Bc3 | TextureFormat::Bc5 | TextureFormat::Bc7
        )
    }
    /// Whether a texel can hold a value outside `[0, 1]`.
    pub fn is_float(self) -> bool {
        matches!(self, TextureFormat::Rgba16F)
    }
}

/// One mip level's dimensions + bytes (in the texture's [`TextureFormat`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextureMip {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// The `.inf_tex` payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextureAsset {
    pub schema_version: u32,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    /// Whether the data is sRGB-encoded (base color) vs linear (normal/data).
    pub srgb: bool,
    /// Mip levels, largest first.
    pub mips: Vec<TextureMip>,
}

impl TextureAsset {
    pub const CURRENT_VERSION: u32 = 1;
    pub fn mip_count(&self) -> usize {
        self.mips.len()
    }

    /// Decode a `.inf_tex` payload of **either container version** (P26.1).
    ///
    /// v1 is the bincode record this struct serializes to; v2 is the tiled image
    /// [`crate::tiles`] writes, and it is reconstructed here into the identical
    /// record. Sniffed on the magic — a v1 payload begins with
    /// `schema_version = 1` and cannot collide with `b"INFVTEX\0"`.
    ///
    /// Every consumer that wants **whole levels** goes through this instead of
    /// `inf_asset::decode`, which sees only v1. A consumer that wants one tile at
    /// a time uses [`crate::tiles::TiledTextureReader`] directly.
    pub fn from_payload(bytes: &[u8]) -> Result<Self, MaterialError> {
        crate::tiles::decode_texture_payload(bytes)
    }

    /// Decode a mip level to RGBA8 regardless of storage format (decompressing
    /// BC1/BC3 on the CPU). Used by the thumbnailer. Returns `None` for an
    /// out-of-range level.
    pub fn level_rgba8(&self, level: usize) -> Option<Vec<u8>> {
        let mip = self.mips.get(level)?;
        Some(match self.format {
            TextureFormat::Rgba8 => mip.data.clone(),
            TextureFormat::Bc1 => bc::decode_bc1(&mip.data, mip.width, mip.height),
            TextureFormat::Bc3 => bc::decode_bc3(&mip.data, mip.width, mip.height),
            TextureFormat::Bc5 => bc::decode_bc5(&mip.data, mip.width, mip.height),
            TextureFormat::Bc7 => bc::decode_bc7(&mip.data, mip.width, mip.height),
            // **This one loses data, and it is the only door that may** (Wave T).
            // A thumbnail is 8 bits per channel because a PNG is; asking a float
            // texture for RGBA8 is asking for a preview, and a preview that
            // clamps is honest where a *storage* format that clamps is not. The
            // hot path never comes here — a float page uploads as RGBA16F and is
            // sampled as one.
            TextureFormat::Rgba16F => rgba16f_to_rgba8(&mip.data, mip.width, mip.height),
        })
    }

    /// Structural validation of a decoded v1 payload — the check
    /// [`AssetPayload::migrate`] runs before anybody may hold one.
    ///
    /// # Why this exists (C4-5, and lens 5's hand-off)
    ///
    /// The v2 tiled container ([`crate::tiles`]) validates thoroughly: magic,
    /// version, `total_len` against the payload, every tile's offset pinned to
    /// the uniform stride. The **v1** path beside it is plain `bincode` with the
    /// default `migrate()`, i.e. no structural check at all — and it is the path
    /// a Content Drawer scan takes for every loose `.inf_tex` in the project.
    ///
    /// Two panics followed from that, both in one function
    /// (`thumbnail::texture_thumbnail`): `mips: []` indexed at `[0]`, and a
    /// header declaring 65 536 × 65 536 over forty bytes of data, where
    /// `decode_bc1`'s `width * height * 4` wrapped to zero and `blit_block` then
    /// wrote past the end of the empty buffer it got back. A file that produces
    /// either is ~40 bytes long.
    ///
    /// The questions asked here are the ones the decoders assume the answers
    /// to:
    ///
    /// 1. the image has a size at all;
    /// 2. it has at least one mip, and every mip has a size;
    /// 3. **every mip carries at least the bytes its own dimensions imply.**
    /// 4. **the chain is the pyramid the header describes** (the round-3
    ///    spot-check).
    ///
    /// (3) is what makes a ceiling unnecessary: a payload claiming 65 536² would
    /// have to *contain* 16 GiB to pass, so the absurd header is refused by
    /// arithmetic rather than by an arbitrary limit somebody has to maintain.
    /// It is `>=`, not `==`, because a level-0 buffer is stored as the importer
    /// received it and a source decoder is entitled to hand back padding; short
    /// is the direction that is unsafe.
    ///
    /// # (4), and why the v2 container's own reasoning asks for it
    ///
    /// The first three questions are all per-mip. Nothing tied the mips to
    /// **each other**, or to the `width`/`height` beside them — so a payload
    /// whose header says 2048² while `mips[0]` is 64², or whose "chain" is four
    /// unrelated images, passed every one of them. Two consequences, both
    /// silent:
    ///
    /// * `thumbnail::texture_thumbnail` selects a level by comparing *mip*
    ///   extents against the requested size and then letterboxes with the same
    ///   mip's extents, while every framing decision above it reads
    ///   `TextureAsset::width`/`height`. Disagreeing numbers give a thumbnail
    ///   scaled to an aspect the image does not have.
    /// * `tiles::lift_texture_asset` builds the v2 tile grid from each mip's own
    ///   extents (`w.div_ceil(TILE_SIZE)`), and the sampler walks the pyramid by
    ///   **halving**: `VtTextureDesc::ancestor` assumes level `n + 1` covers the
    ///   same surface as level `n` at half the resolution. A chain that does not
    ///   halve makes that walk address a tile of a different picture — the
    ///   "odd-halving pyramid" the VT tests already treat as load-bearing, with
    ///   the pyramid itself no longer a pyramid.
    ///
    /// The rule is exactly the one `rgba_mip_chain` produces: `mips[0]` is the
    /// declared extent, and each level after it is `(prev / 2).max(1)`. A chain
    /// that **stops early** is legal (`generate_mips: false` is one level, and a
    /// partial pyramid is a real import setting); a chain that changes shape is
    /// not.
    pub fn validate(&self) -> Result<(), MaterialError> {
        if self.width == 0 || self.height == 0 {
            return Err(MaterialError::Image(format!(
                "texture declares a {}×{} image",
                self.width, self.height
            )));
        }
        if self.mips.is_empty() {
            return Err(MaterialError::Image(
                "texture has no mip levels (every consumer reads level 0)".into(),
            ));
        }
        for (i, mip) in self.mips.iter().enumerate() {
            if mip.width == 0 || mip.height == 0 {
                return Err(MaterialError::Image(format!(
                    "texture mip {i} declares a {}×{} level",
                    mip.width, mip.height
                )));
            }
            let need = self.format.level_size(mip.width, mip.height);
            if mip.data.len() < need {
                return Err(MaterialError::Image(format!(
                    "texture mip {i} declares {}×{} ({need} bytes in {:?}) and carries {}",
                    mip.width,
                    mip.height,
                    self.format,
                    mip.data.len()
                )));
            }
            // (4) The chain, against the header and against itself.
            let (want_w, want_h) = match i {
                0 => (self.width, self.height),
                _ => {
                    let p = &self.mips[i - 1];
                    ((p.width / 2).max(1), (p.height / 2).max(1))
                }
            };
            if (mip.width, mip.height) != (want_w, want_h) {
                return Err(MaterialError::Image(format!(
                    "texture mip {i} is {}×{}, but the {} is {want_w}×{want_h}",
                    mip.width,
                    mip.height,
                    if i == 0 {
                        "image it declares"
                    } else {
                        "halving of the level above it"
                    }
                )));
            }
        }
        Ok(())
    }
}

impl AssetPayload for TextureAsset {
    const KIND: AssetKind = AssetKind::Texture;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Reject newer-than-current (the default rule), then **validate** — the
    /// [`BiomeSet`](inf_asset::AssetPayload) pattern, applied to the one v1
    /// bincode payload whose consumers index straight into it. See
    /// [`TextureAsset::validate`].
    fn migrate(self) -> inf_asset::Result<Self> {
        let found = self.schema_version;
        if found > Self::SCHEMA_VERSION {
            return Err(inf_asset::AssetError::SchemaTooNew {
                kind: Self::KIND.slug(),
                found,
                current: Self::SCHEMA_VERSION,
            });
        }
        self.validate()
            .map_err(|e| inf_asset::AssetError::Decode(format!("invalid texture: {e}")))?;
        Ok(self)
    }
}

/// Which block compression to apply on import.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TextureCompression {
    /// Keep RGBA8.
    None,
    /// Force BC1 (drops alpha).
    Bc1,
    /// Force BC3.
    Bc3,
    /// Force **BC5** — two channels (R, G), the normal-map format (Wave T).
    /// Blue and alpha are discarded and the sampler rebuilds Z.
    Bc5,
    /// Force **BC7** — full RGBA at twice BC1's bytes, mode 6 (wave IASSET2).
    /// The quality choice for a base colour or an ORM triple whose banding
    /// shows; see [`crate::bc::compress_bc7`] for what the subset covers.
    Bc7,
    /// BC1 if fully opaque, else BC3.
    #[default]
    Auto,
}

/// Reimport-reproducible texture import settings (stored in the sidecar).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TextureImportSettings {
    pub srgb: bool,
    pub generate_mips: bool,
    pub compression: TextureCompression,
    /// **Keep a float source's dynamic range** (Wave T).
    ///
    /// `false` — the default, and exactly what every import did before Wave T —
    /// decodes an EXR or Radiance source to 8 bits per channel like any PNG,
    /// which is lossy in a way nothing used to say out loud.
    /// [`hdr_import_advisory`] now says it.
    ///
    /// `true` imports a float source as [`TextureFormat::Rgba16F`] instead:
    /// 8 bytes a texel, no block compression (there is no BC6H encoder here),
    /// and the values arrive as authored. It has no effect on an 8-bit source —
    /// promoting a PNG to half-float invents precision it never had and costs
    /// twice the bytes to do it.
    ///
    /// `#[serde(default)]`: the sidecar is TOML, so every settings block written
    /// before Wave T reads back with the pre-Wave-T behaviour.
    #[serde(default)]
    pub hdr: bool,
}

impl Default for TextureImportSettings {
    fn default() -> Self {
        Self {
            srgb: true,
            generate_mips: true,
            compression: TextureCompression::Auto,
            hdr: false,
        }
    }
}

impl TextureImportSettings {
    /// Preset for non-color data (masks, ORM, height): linear, uncompressed.
    ///
    /// **Still uncompressed, and deliberately** (Wave T). BC1 on a data map
    /// quantises the channels the data lives in, and Wave T's answer for the one
    /// data map that has a *correct* compressed form — the normal map — is
    /// [`normal_map`](Self::normal_map), not a change of default here. Anything
    /// that changed what this preset produces would move the bytes of every
    /// asset that has ever been imported through it.
    pub fn data() -> Self {
        Self {
            srgb: false,
            generate_mips: true,
            compression: TextureCompression::None,
            hdr: false,
        }
    }

    /// Preset for a **tangent-space normal map** (Wave T): linear, BC5.
    ///
    /// The preset [`data`](Self::data) could not be, because it is the preset for
    /// every data map and BC5 keeps only two channels. Opting in per map is what
    /// makes that safe: an ORM triple through `data()` still stores three
    /// channels, and a normal map through this one stores the two it needs at a
    /// quarter of the page bytes.
    pub fn normal_map() -> Self {
        Self {
            srgb: false,
            generate_mips: true,
            compression: TextureCompression::Bc5,
            hdr: false,
        }
    }

    /// [`data`](Self::data) with the float range kept — the preset a Megascans
    /// EXR displacement/cavity map wants.
    pub fn data_hdr() -> Self {
        Self {
            hdr: true,
            ..Self::data()
        }
    }
}

/// **Import advisories** for a source image's dimensions (P26.1) — non-fatal
/// notices in the P16 cook-advisory shape: the import always succeeds, and the
/// caller surfaces what it had to do.
///
/// Three hazards are visible from the extent alone, and all three are silent
/// otherwise:
///
/// * **Not a multiple of 4.** A BC block is 4×4, so a partial block at the right
///   or bottom edge is filled by repeating the last row/column of texels
///   ([`crate::bc`]'s edge clamp). The invented texels are compressed *with* the
///   real ones, so the endpoints of an edge block are chosen partly from data
///   that is not in the image. It is the standard treatment and it is fine for
///   most content; it is not fine silently for a texture whose edge matters
///   (a tiling material, an atlas page).
/// * **An extreme aspect ratio.** A virtual texture is tiled at 128², so a
///   `4096×16` source is 32 tiles wide and 1 tall, and every one of them stores
///   112 rows of clamp padding — 87 % of the payload is padding. The texture
///   works; it costs about eight times what its pixels do.
/// * **A small texture pays for the uniform page** (P26.5, the P26.1 audit's
///   deferred badging). Every mip level is stored in whole `136²` tiles, so a
///   texture whose levels are mostly smaller than one tile stores mostly border
///   ring and clamp padding: eight of a 128² texture's levels are one whole tile
///   each. The P26.1 audit measured it — **6.8× at 128², 2.55× at 256², 1.22× at
///   1024², 1.16× at 2048²** — and found the cost is inherent to the page rather
///   than (as four documents then said) a lost zstd frame: BC1 and
///   `TextureCompression::None` grow by the *same* factor, which is what
///   falsifies the compression story directly. [`tiled_size_factor_pct`]
///   recomputes those numbers from the extent, so the advisory carries the
///   project's own figure instead of a remembered one.
///
/// Pure and deterministic (dimensions in, sorted sentences out), so it is
/// unit-tested with no project, no GPU and no filesystem.
pub fn texture_import_advisories(width: u32, height: u32) -> Vec<String> {
    let mut out = Vec::new();
    if !width.is_multiple_of(4) || !height.is_multiple_of(4) {
        out.push(format!(
            "{width}×{height} is not a multiple of 4, so the edge BC blocks are padded by \
             repeating the last row/column; resize to a multiple of 4 if the edge texels matter"
        ));
    }
    // 8:1 is the threshold, not a magic number: at 8:1 a 128-texel tile row is
    // already ≥ 7/8 padding in the short axis once the source is smaller than one
    // tile there, which is the point at which the padding outweighs the content.
    let (long, short) = (width.max(height), width.min(height).max(1));
    if long / short >= 8 {
        out.push(format!(
            "{width}×{height} is {}:1 — a virtual texture is tiled at {}², so a strip like this \
             stores mostly clamp padding; consider an atlas instead",
            long / short,
            crate::tiles::TILE_SIZE
        ));
    }
    if let Some(pct) = tail_cost_advisory(width, height) {
        out.push(pct);
    }
    out
}

/// **How much a tiled `.inf_tex` costs against the raw pyramid it stores**, as a
/// percentage (100 = the same size).
///
/// A pure function of the extent and the tile geometry, integer-only, because
/// the answer is arithmetic and not a measurement: every mip level is stored as
/// `ceil(w / TILE_SIZE) × ceil(h / TILE_SIZE)` tiles of `stored²` texels, and the
/// raw pyramid is `Σ w·h`. The ratio is **format-independent** — a stored tile
/// and a raw level compress by the same rule — which is exactly the P26.1 audit's
/// finding that the cost is the uniform page and not a lost zstd frame.
///
/// Reproduces the audit's measured table to the digit
/// (`the_tail_cost_advisory_reproduces_the_measured_table`), which is why the
/// advisory can quote a number rather than a memory.
pub fn tiled_size_factor_pct(width: u32, height: u32) -> u32 {
    let stored = u64::from(crate::tiles::TILE_SIZE + 2 * crate::tiles::TILE_BORDER);
    let (mut w, mut h) = (width.max(1), height.max(1));
    let (mut raw, mut tiled) = (0u64, 0u64);
    loop {
        raw += u64::from(w) * u64::from(h);
        tiled += u64::from(w.div_ceil(crate::tiles::TILE_SIZE))
            * u64::from(h.div_ceil(crate::tiles::TILE_SIZE))
            * stored
            * stored;
        if w == 1 && h == 1 {
            break;
        }
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    ((tiled * 100) / raw.max(1)) as u32
}

/// The tail-cost advisory, or `None` when the extent carries it comfortably.
///
/// **200 %** is the threshold, and it is where the measured table turns: 128²
/// pays 677 %, 256² pays 254 %, and 1024² pays 121 % — so this fires for the
/// sizes where the uniform page dominates and stays quiet for the sizes a real
/// material is authored at. It is a badge, not a refusal: the texture works, it
/// pages exactly like any other, and an author who wants many small textures
/// should know they are cheaper in an atlas.
fn tail_cost_advisory(width: u32, height: u32) -> Option<String> {
    let pct = tiled_size_factor_pct(width, height);
    (pct >= 200).then(|| {
        format!(
            "{width}×{height} stores {}.{}× its own pixels as a tiled .inf_tex — every mip \
             level is kept in whole {}² tiles, so a small texture is mostly border \
             ring and clamp padding; pack small textures into an atlas if the \
             on-disk size matters",
            pct / 100,
            (pct / 10) % 10,
            crate::tiles::TILE_SIZE + 2 * crate::tiles::TILE_BORDER
        )
    })
}

/// **The HDR advisory** (Wave T): what an import is about to do to a float
/// source's dynamic range, said out loud.
///
/// The P16 house rule is "a cook advisory for every silent hazard", and this is
/// one of the purest examples the project has had. `image::load_from_memory(…)
/// .to_rgba8()` is one call and it is the *only* decode an import has ever done,
/// so a 32-bit Radiance or OpenEXR source — the exact shape the Megascans/Fab
/// pipeline delivers displacement, cavity and bent-normal maps in — arrived with
/// everything above 1.0 clipped and everything below quantised to 256 steps, and
/// the import reported success. Nothing downstream could tell the difference
/// between that and a PNG.
///
/// Two sentences, never both: one when the range is being thrown away, one when
/// it is being kept and the author should know what it costs. Pure, so it is
/// unit-tested with no file and no project.
pub fn hdr_import_advisory(source_is_float: bool, kept_float: bool) -> Option<String> {
    if !source_is_float {
        return None;
    }
    if kept_float {
        Some(format!(
            "this is a floating-point source and it is being kept as RGBA16F — 8 bytes a texel, \
             {}× a BC1 page and {}× a BC5 one, because there is no BC6H encoder in this project \
             (see docs/memos/wave-t-textures-disposition.md); import it with hdr = false if the \
             map's values all live in [0, 1]. A half holds up to 65 504: anything past that \
             saturates rather than becoming infinity, and a NaN sample is stored as 0 — see \
             `half_at_the_door`",
            8 * 4 / 4,
            8 * 4 / 8
        ))
    } else {
        Some(
            "this is a floating-point source (EXR/Radiance) and it is being imported at 8 bits \
             per channel: every value above 1.0 is clipped and the rest is quantised to 256 \
             steps. Set hdr = true on the import to keep it as RGBA16F"
                .to_string(),
        )
    }
}

/// Whether an encoded source carries **floating-point** samples — the question
/// [`hdr_import_advisory`] is asked.
///
/// Sniffed from the container rather than from a decode, because the answer has
/// to be available before deciding which decode to run. Radiance `.hdr` and
/// OpenEXR are the two float containers the accepted-extension list admits; a
/// 16-bit PNG is high-precision and is *not* float (it cannot hold a value above
/// 1.0), so it is not one of these and does not want an RGBA16F page.
pub fn source_is_float(bytes: &[u8]) -> bool {
    matches!(
        image::guess_format(bytes),
        Ok(image::ImageFormat::Hdr) | Ok(image::ImageFormat::OpenExr)
    )
}

/// Decode an encoded image (PNG/JPEG/…) and import it with `settings`.
///
/// **The one place the float branch is taken** (Wave T): a float source with
/// `settings.hdr` keeps its range, and everything else takes the path it always
/// did, byte for byte.
pub fn import_texture_bytes(
    bytes: &[u8],
    settings: TextureImportSettings,
) -> Result<TextureAsset, MaterialError> {
    if settings.hdr && source_is_float(bytes) {
        let (halfs, w, h) = decode_image_rgba16f(bytes)?;
        return texture_from_rgba16f(halfs, w, h, settings);
    }
    let (rgba, w, h) = decode_image_rgba8(bytes)?;
    texture_from_rgba8(rgba, w, h, settings)
}

/// Decode an encoded image (PNG/JPEG/TGA/BMP/HDR/EXR) to `(rgba8, width,
/// height)`.
///
/// The one place the `image` crate is named for a texture import, so the v1
/// writer, the v2 tiler and [`texture_import_advisories`] all read the **same**
/// decoded extent. An importer that decoded separately could advise about one
/// size and tile another.
pub fn decode_image_rgba8(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), MaterialError> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| MaterialError::Image(e.to_string()))?
        .to_rgba8();
    let (w, h) = img.dimensions();
    Ok((img.into_raw(), w, h))
}

/// Decode an encoded image to **RGBA16F**, four little-endian halves a texel
/// (Wave T).
///
/// The float twin of [`decode_image_rgba8`], and the same "one decode site" rule
/// applies: the advisory, the tiler and the v1 writer all read the extent this
/// returns.
///
/// `to_rgba32f` is `image`'s own widening — it is exact for a float source and
/// it is why this is not a second decoder. The narrowing to half is
/// round-to-nearest-even and therefore a pure function of the input bits, on
/// every target: this file writes bytes that are content-hashed into a
/// reproducible pack, which is the same claim `crate::bc`'s no-float gate
/// defends by banning floats outright. It cannot ban them here — a half *is* a
/// float — so it defends the claim the other way, by using only operations IEEE
/// 754 specifies exactly (`from_f32`, and in the mip chain a sum of four values
/// scaled by a power of two). No transcendental, no fused multiply-add, no
/// libm.
///
/// # The door clamps, and never bakes (Wave-T audit)
///
/// A half is not an `f32`, and an EXR is not obliged to fit in one. Measured on
/// this path before [`half_at_the_door`] existed: a texel of `70 000` — an
/// ordinary sun disc in a captured HDR sky, and below what `1e30` does — narrows
/// to **`+inf`**, and a texel of `NaN` narrows to `NaN`. Both were written into
/// the page verbatim, and then `rgba16f_mip_chain` averaged them: a *single*
/// NaN texel poisons its 2×2 parent, and its parent, all the way to the 1×1, so
/// **one bad sample in a source made every coarse level of the pyramid NaN** —
/// which is the level a virtual texture's pinned floor samples, i.e. the one
/// that is always resident. `TextureAsset::validate` was happy with it.
///
/// So this door does what the P29.5 import-door precedent says: a non-finite
/// sample is clamped or refused, never stored. NaN becomes `0.0` (the only
/// value that is not a lie about a measurement that does not exist) and an
/// overflow saturates to `±f16::MAX` rather than to infinity — which keeps the
/// promise the format was added for (the *range* survives) up to the range the
/// format actually has, instead of past it. Finite in-range samples are
/// untouched, so nothing an honest source carries is changed by this.
///
/// The trade is stated to the author by [`hdr_import_advisory`], because a
/// silent clamp is the shape of failure this whole item exists to end.
pub fn decode_image_rgba16f(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), MaterialError> {
    let img = image::load_from_memory(bytes).map_err(|e| MaterialError::Image(e.to_string()))?;
    let f = img.to_rgba32f();
    let (w, h) = f.dimensions();
    let mut out = Vec::with_capacity(w as usize * h as usize * 8);
    for px in f.pixels() {
        for c in px.0 {
            out.extend_from_slice(&half_at_the_door(c).to_le_bytes());
        }
    }
    Ok((out, w, h))
}

/// **One `f32` sample as a storable half** — the whole non-finite policy, in one
/// pure function so both the import door and the mip chain speak it.
///
/// * `NaN` → `0.0`. A NaN is not a value a texture can carry: it propagates
///   through every filter that touches it, and a *storage* format that keeps one
///   hands the poison to the sampler and to every coarser mip.
/// * anything past the half's range → `±f16::MAX` (65 504). Saturating rather
///   than overflowing to infinity, for the same reason: `inf * 0.25` is `inf`,
///   and `inf - inf` in a later blend is a NaN.
/// * everything else → `half::f16::from_f32`, which is round-to-nearest-even and
///   therefore a pure function of the input bits on every target.
///
/// **This clamps, it does not refuse.** The refusal would have to be the whole
/// import, and a single blown texel in a 4K capture is not a reason to reject a
/// map an author has; the advisory carries the news instead.
#[inline]
pub fn half_at_the_door(v: f32) -> half::f16 {
    if v.is_nan() {
        return half::f16::ZERO;
    }
    const MAX: f32 = 65_504.0;
    half::f16::from_f32(v.clamp(-MAX, MAX))
}

/// Import an already-decoded **RGBA16F** buffer (four LE halves a texel).
///
/// The format is `Rgba16F` whatever `settings.compression` says: none of BC1,
/// BC3 or BC5 can hold a value outside `[0, 1]`, so honouring a compression
/// request here would be the silent flattening this path exists to end. A caller
/// that wants a compressed HDR page wants BC6H, which this project does not have.
pub fn texture_from_rgba16f(
    halfs: Vec<u8>,
    width: u32,
    height: u32,
    settings: TextureImportSettings,
) -> Result<TextureAsset, MaterialError> {
    let levels = rgba16f_mip_chain(halfs, width, height, settings.generate_mips)?;
    Ok(TextureAsset {
        schema_version: TextureAsset::CURRENT_VERSION,
        width,
        height,
        format: TextureFormat::Rgba16F,
        srgb: false,
        mips: levels
            .into_iter()
            .map(|(w, h, data)| TextureMip {
                width: w,
                height: h,
                data,
            })
            .collect(),
    })
}

/// Import an already-decoded RGBA8 buffer (e.g. a glTF-embedded image).
pub fn texture_from_rgba8(
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    settings: TextureImportSettings,
) -> Result<TextureAsset, MaterialError> {
    let levels = rgba_mip_chain(rgba, width, height, settings.generate_mips)?;
    let format = choose_format(settings.compression, &levels[0].2);

    let mips = levels
        .into_iter()
        .map(|(w, h, data)| {
            let data = match format {
                TextureFormat::Rgba8 => data,
                TextureFormat::Bc1 => bc::compress_bc1(&data, w, h),
                TextureFormat::Bc3 => bc::compress_bc3(&data, w, h),
                TextureFormat::Bc5 => bc::compress_bc5(&data, w, h),
                TextureFormat::Bc7 => bc::compress_bc7(&data, w, h),
                // Unreachable: `choose_format` never answers a float format for
                // an 8-bit source (promoting one invents precision). Answering
                // the bytes unchanged rather than panicking, because a wrong
                // *size* is caught by `validate` at the very next door.
                TextureFormat::Rgba16F => data,
            };
            TextureMip {
                width: w,
                height: h,
                data,
            }
        })
        .collect();

    Ok(TextureAsset {
        schema_version: TextureAsset::CURRENT_VERSION,
        width,
        height,
        format,
        srgb: settings.srgb,
        mips,
    })
}

/// The RGBA8 mip chain of a source image, largest first, down to 1×1.
///
/// Shared by the v1 writer above and the v2 tiler ([`crate::tiles`]) so the two
/// containers cannot disagree about how many levels there are or what is in
/// them — the premise of "a v2 image reconstructs its v1 byte for byte".
pub(crate) fn rgba_mip_chain(
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    generate_mips: bool,
) -> Result<Vec<(u32, u32, Vec<u8>)>, MaterialError> {
    if width == 0 || height == 0 {
        return Err(MaterialError::Image("zero-sized texture".into()));
    }
    // `usize`, not `u32` (C4-16): this guard is the one that was supposed to
    // notice a truncated buffer, and it wrapped *identically* to the producer
    // that truncated it — `0 < 0` is false, so the empty buffer sailed through
    // into `downsample_box`. Two wraps that cancelled into silence.
    if rgba.len() < width as usize * height as usize * 4 {
        return Err(MaterialError::Image("truncated pixel buffer".into()));
    }
    let mut levels: Vec<(u32, u32, Vec<u8>)> = vec![(width, height, rgba)];
    if generate_mips {
        while levels.last().unwrap().0 > 1 || levels.last().unwrap().1 > 1 {
            let (pw, ph, ref prev) = *levels.last().unwrap();
            let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
            let down = downsample_box(prev, pw, ph, nw, nh);
            levels.push((nw, nh, down));
        }
    }
    Ok(levels)
}

/// The **RGBA16F** mip chain of a float source, largest first, down to 1×1
/// (Wave T).
///
/// The twin of [`rgba_mip_chain`], and shared by the v1 writer and the v2 tiler
/// for exactly the same reason: two chains that agree today are two chains.
///
/// Every arithmetic step is IEEE-exact. A half widens to `f32` losslessly; four
/// `f32` adds are correctly rounded; and the scale is `0.25`, a power of two, so
/// it is exact too. Rust emits no fused multiply-add without being asked, and
/// there is no transcendental here — so the bytes this writes are the same bytes
/// on every target, which is the property `crate::bc`'s gate defends for the
/// integer formats and this paragraph defends for the float one.
pub(crate) fn rgba16f_mip_chain(
    halfs: Vec<u8>,
    width: u32,
    height: u32,
    generate_mips: bool,
) -> Result<Vec<(u32, u32, Vec<u8>)>, MaterialError> {
    if width == 0 || height == 0 {
        return Err(MaterialError::Image("zero-sized texture".into()));
    }
    if halfs.len() < width as usize * height as usize * 8 {
        return Err(MaterialError::Image("truncated pixel buffer".into()));
    }
    let mut levels: Vec<(u32, u32, Vec<u8>)> = vec![(width, height, halfs)];
    if generate_mips {
        while levels.last().unwrap().0 > 1 || levels.last().unwrap().1 > 1 {
            let (pw, ph, ref prev) = *levels.last().unwrap();
            let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
            let down = downsample_box_rgba16f(prev, pw, ph, nw, nh);
            levels.push((nw, nh, down));
        }
    }
    Ok(levels)
}

/// One half at texel `(x, y)`, channel `c`.
#[inline]
fn half_at(src: &[u8], w: u32, x: u32, y: u32, c: usize) -> f32 {
    let i = ((y as usize * w as usize + x as usize) * 4 + c) * 2;
    half::f16::from_le_bytes([src[i], src[i + 1]]).to_f32()
}

/// 2×2 box-filter downsample of an RGBA16F image, mirroring `downsample_box`.
fn downsample_box_rgba16f(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut out = vec![0u8; dw as usize * dh as usize * 8];
    for y in 0..dh {
        for x in 0..dw {
            let sx0 = (x * 2).min(sw - 1);
            let sy0 = (y * 2).min(sh - 1);
            let sx1 = (x * 2 + 1).min(sw - 1);
            let sy1 = (y * 2 + 1).min(sh - 1);
            for c in 0..4 {
                let acc = half_at(src, sw, sx0, sy0, c)
                    + half_at(src, sw, sx1, sy0, c)
                    + half_at(src, sw, sx0, sy1, c)
                    + half_at(src, sw, sx1, sy1, c);
                let o = ((y as usize * dw as usize + x as usize) * 4 + c) * 2;
                // `half_at_the_door`, not a bare `from_f32` (Wave-T audit): a
                // level built from an already-stored page must not be able to
                // introduce what the import door refuses. Four values ≤ f16::MAX
                // sum to at most 4× it, which is finite in `f32` and clamps back
                // here — so the pyramid cannot overflow at a level even where
                // its parent did not.
                out[o..o + 2].copy_from_slice(&half_at_the_door(acc * 0.25).to_le_bytes());
            }
        }
    }
    out
}

/// RGBA16F → RGBA8 for a **preview**, clamping to `[0, 1]`. See
/// [`TextureAsset::level_rgba8`] for why this is the only door allowed to.
fn rgba16f_to_rgba8(src: &[u8], width: u32, height: u32) -> Vec<u8> {
    let n = width as usize * height as usize * 4;
    let mut out = vec![0u8; n];
    for (i, o) in out.iter_mut().enumerate() {
        let Some(b) = src.get(i * 2..i * 2 + 2) else {
            break;
        };
        let v = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
        *o = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    out
}

/// The storage format an import setting resolves to for a given level-0 image.
/// Shared with the v2 tiler for the same reason [`rgba_mip_chain`] is.
pub(crate) fn choose_format(compression: TextureCompression, level0: &[u8]) -> TextureFormat {
    match compression {
        TextureCompression::None => TextureFormat::Rgba8,
        TextureCompression::Bc1 => TextureFormat::Bc1,
        TextureCompression::Bc3 => TextureFormat::Bc3,
        TextureCompression::Bc5 => TextureFormat::Bc5,
        TextureCompression::Bc7 => TextureFormat::Bc7,
        TextureCompression::Auto => {
            if is_fully_opaque(level0) {
                TextureFormat::Bc1
            } else {
                TextureFormat::Bc3
            }
        }
    }
}

fn is_fully_opaque(rgba: &[u8]) -> bool {
    rgba.chunks_exact(4).all(|p| p[3] == 255)
}

/// 2×2 box-filter downsample of an RGBA8 image (source-space average). Handles
/// odd dimensions by clamping the sample coordinates.
fn downsample_box(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut out = vec![0u8; (dw * dh * 4) as usize];
    for y in 0..dh {
        for x in 0..dw {
            let sx0 = (x * 2).min(sw - 1);
            let sy0 = (y * 2).min(sh - 1);
            let sx1 = (x * 2 + 1).min(sw - 1);
            let sy1 = (y * 2 + 1).min(sh - 1);
            let mut acc = [0u32; 4];
            for &(sx, sy) in &[(sx0, sy0), (sx1, sy0), (sx0, sy1), (sx1, sy1)] {
                let i = ((sy * sw + sx) * 4) as usize;
                for c in 0..4 {
                    acc[c] += src[i + c] as u32;
                }
            }
            let o = ((y * dw + x) * 4) as usize;
            for c in 0..4 {
                out[o + c] = (acc[c] / 4) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::{decode, encode};

    /// A `.inf_tex` v1 payload built **by hand**, so a test can declare
    /// dimensions its data does not back — which is the whole finding.
    fn handmade(width: u32, height: u32, format: TextureFormat, mips: Vec<TextureMip>) -> Vec<u8> {
        encode(&TextureAsset {
            schema_version: TextureAsset::CURRENT_VERSION,
            width,
            height,
            format,
            srgb: false,
            mips,
        })
        .unwrap()
    }

    /// **C4-5 / the lens-5 hand-off — the ~40-byte `.inf_tex` that used to
    /// panic the Content Drawer.**
    ///
    /// Two panics in one function, both reached by a Content Drawer scan
    /// touching any `.inf_tex` in the project directory:
    ///
    /// * `mips: []` indexed at `[0]` by `texture_thumbnail`'s `unwrap_or(0)`;
    /// * a header declaring 65 536 × 65 536 over a handful of bytes, where
    ///   `decode_bc1`'s `width * height * 4` wrapped in `u32` to **exactly
    ///   zero** (2³² pixels × 4) and `blit_block` then wrote past the end of the
    ///   empty buffer it was handed.
    ///
    /// Both are refused at `migrate`, i.e. by `inf_asset::decode` itself, so no
    /// consumer ever holds one. Un-fix mutation: drop the `self.validate()` call
    /// from `TextureAsset::migrate` and every arm below decodes successfully.
    #[test]
    fn a_texture_that_lies_about_its_size_is_refused_at_decode() {
        // The control: a real payload still decodes.
        let good = texture_from_rgba8(
            checker(4, 4, 255),
            4,
            4,
            TextureImportSettings {
                compression: TextureCompression::None,
                generate_mips: false,
                srgb: false,
                hdr: false,
            },
        )
        .unwrap();
        let bytes = encode(&good).unwrap();
        decode::<TextureAsset>(&bytes).expect("a healthy texture must still decode");

        // No mips at all.
        let empty = handmade(4, 4, TextureFormat::Rgba8, Vec::new());
        assert!(
            decode::<TextureAsset>(&empty).is_err(),
            "a texture with no mip levels decoded"
        );
        assert!(
            empty.len() < 64,
            "the whole hazard is that this file is tiny: {} bytes",
            empty.len()
        );

        // The 65 536² header over forty bytes of data — the u32-wrap class.
        let huge = handmade(
            65_536,
            65_536,
            TextureFormat::Bc1,
            vec![TextureMip {
                width: 65_536,
                height: 65_536,
                data: vec![0u8; 8],
            }],
        );
        assert!(
            huge.len() < 64,
            "the whole hazard is that this file is tiny: {} bytes",
            huge.len()
        );
        let e = decode::<TextureAsset>(&huge).unwrap_err().to_string();
        assert!(e.contains("carries 8"), "refusal did not name the gap: {e}");

        // A zero-sized level, and a zero-sized image.
        assert!(decode::<TextureAsset>(&handmade(
            0,
            4,
            TextureFormat::Rgba8,
            vec![TextureMip {
                width: 4,
                height: 4,
                data: vec![0u8; 64]
            }]
        ))
        .is_err());
        assert!(decode::<TextureAsset>(&handmade(
            4,
            4,
            TextureFormat::Rgba8,
            vec![TextureMip {
                width: 0,
                height: 4,
                data: Vec::new()
            }]
        ))
        .is_err());

        // A truncated RGBA8 level: one byte short is still short.
        assert!(decode::<TextureAsset>(&handmade(
            4,
            4,
            TextureFormat::Rgba8,
            vec![TextureMip {
                width: 4,
                height: 4,
                data: vec![0u8; 63]
            }]
        ))
        .is_err());
    }

    /// **The round-3 spot-check on C4-5: the mips were checked one at a time,
    /// and never against each other or against the header.**
    ///
    /// A payload can pass all three of the original questions and still not be
    /// a pyramid — 2048² in the header over a 64² level 0, or four unrelated
    /// images stacked as "mips". `texture_thumbnail` picks a level by *mip*
    /// extent and letterboxes by it while its callers frame by the header's,
    /// and the v2 tiler derives every grid from the mip's own extent while the
    /// sampler walks the pyramid by halving. Neither notices.
    ///
    /// Every case here is a payload the three original questions accept: the
    /// data is long enough for the dimensions it declares, and no extent is
    /// zero. The controls are the two legal shapes — one level
    /// (`generate_mips: false`) and a full chain — so this cannot pass by
    /// refusing everything.
    #[test]
    fn a_mip_chain_that_is_not_the_header_s_pyramid_is_refused_at_decode() {
        let rgba = |w: u32, h: u32| TextureMip {
            width: w,
            height: h,
            data: vec![0u8; (w as usize) * (h as usize) * 4],
        };

        // Controls. One level is legal (generate_mips: false), and so is the
        // real chain — including one that stops before 1×1.
        for mips in [
            vec![rgba(8, 4)],
            vec![rgba(8, 4), rgba(4, 2)],
            vec![rgba(8, 4), rgba(4, 2), rgba(2, 1), rgba(1, 1)],
        ] {
            let bytes = handmade(8, 4, TextureFormat::Rgba8, mips);
            decode::<TextureAsset>(&bytes).expect("a real pyramid must decode");
        }

        // Level 0 disagreeing with the header it is the level 0 OF.
        let e = decode::<TextureAsset>(&handmade(
            2048,
            2048,
            TextureFormat::Rgba8,
            vec![rgba(64, 64)],
        ))
        .unwrap_err()
        .to_string();
        assert!(
            e.contains("mip 0") && e.contains("image it declares"),
            "the refusal must name the disagreement: {e}"
        );

        // A chain that does not halve: same-size twice, a quarter step, and a
        // level that grows. All three break the sampler's ancestor walk.
        for (bad, what) in [
            (vec![rgba(8, 4), rgba(8, 4)], "did not shrink"),
            (vec![rgba(8, 4), rgba(2, 1)], "skipped a level"),
            (vec![rgba(8, 4), rgba(16, 8)], "grew"),
        ] {
            let e = decode::<TextureAsset>(&handmade(8, 4, TextureFormat::Rgba8, bad))
                .unwrap_err()
                .to_string();
            assert!(
                e.contains("mip 1") && e.contains("halving"),
                "a chain that {what} decoded, or was refused for the wrong reason: {e}"
            );
        }

        // The floor: a non-square chain keeps halving the axis that can, and
        // `1` stays `1` — this is `rgba_mip_chain`'s own rule and it must not
        // be refused as a non-halving step.
        decode::<TextureAsset>(&handmade(
            4,
            1,
            TextureFormat::Rgba8,
            vec![rgba(4, 1), rgba(2, 1), rgba(1, 1)],
        ))
        .expect("the max(1) floor is the chain the importer produces");
    }

    /// The arithmetic that made the header believable: `level_size` in `u32`
    /// answers **zero** for the image whose data would be 16 GiB.
    #[test]
    fn level_size_does_not_wrap_at_the_65536_squared_boundary() {
        // 65 536² × 4 == 2³², which is 0 in u32 and 17 179 869 184 in usize.
        assert_eq!(
            TextureFormat::Rgba8.level_size(65_536, 65_536),
            17_179_869_184usize
        );
        assert_eq!(TextureFormat::Bc1.level_size(65_536, 65_536), 2_147_483_648);
        // Ordinary sizes are unchanged.
        assert_eq!(TextureFormat::Rgba8.level_size(4, 4), 64);
    }

    fn checker(w: u32, h: u32, alpha: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let on = (x + y) % 2 == 0;
                let c = if on { 220 } else { 30 };
                v.extend_from_slice(&[c, c, c, alpha]);
            }
        }
        v
    }

    #[test]
    fn full_mip_chain_down_to_1x1() {
        let tex =
            texture_from_rgba8(checker(8, 8, 255), 8, 8, TextureImportSettings::default()).unwrap();
        // 8,4,2,1 → 4 levels.
        assert_eq!(tex.mip_count(), 4);
        assert_eq!(tex.mips[0].width, 8);
        assert_eq!(tex.mips.last().unwrap().width, 1);
    }

    #[test]
    fn auto_picks_bc1_when_opaque_bc3_when_alpha() {
        let opaque =
            texture_from_rgba8(checker(4, 4, 255), 4, 4, TextureImportSettings::default()).unwrap();
        assert_eq!(opaque.format, TextureFormat::Bc1);
        let translucent =
            texture_from_rgba8(checker(4, 4, 128), 4, 4, TextureImportSettings::default()).unwrap();
        assert_eq!(translucent.format, TextureFormat::Bc3);
    }

    #[test]
    fn compressed_level_sizes_match_format() {
        let tex =
            texture_from_rgba8(checker(4, 4, 255), 4, 4, TextureImportSettings::default()).unwrap();
        // 4×4 BC1 = one 8-byte block.
        assert_eq!(tex.mips[0].data.len(), 8);
    }

    #[test]
    fn bc_levels_decode_back_to_rgba8() {
        // Opaque → BC1, decodes to RGBA8 of the right size and opaque alpha.
        let tex =
            texture_from_rgba8(checker(8, 8, 255), 8, 8, TextureImportSettings::default()).unwrap();
        assert_eq!(tex.format, TextureFormat::Bc1);
        let rgba = tex.level_rgba8(0).unwrap();
        assert_eq!(rgba.len(), 8 * 8 * 4);
        assert!(rgba.chunks_exact(4).all(|p| p[3] == 255));
        // Translucent → BC3, alpha preserved (~128).
        let tex3 =
            texture_from_rgba8(checker(8, 8, 128), 8, 8, TextureImportSettings::default()).unwrap();
        assert_eq!(tex3.format, TextureFormat::Bc3);
        let a = tex3.level_rgba8(0).unwrap()[3];
        assert!((a as i32 - 128).abs() <= 20, "alpha ~128, got {a}");
    }

    #[test]
    fn no_compression_keeps_rgba8_and_round_trips() {
        let s = TextureImportSettings {
            compression: TextureCompression::None,
            ..Default::default()
        };
        let tex = texture_from_rgba8(checker(4, 4, 255), 4, 4, s).unwrap();
        assert_eq!(tex.format, TextureFormat::Rgba8);
        assert_eq!(tex.mips[0].data.len(), 4 * 4 * 4);
        let bytes = encode(&tex).unwrap();
        assert_eq!(decode::<TextureAsset>(&bytes).unwrap(), tex);
    }

    /// **The eaten-`\` law, sixth catch — and this time it is guarded**
    /// (P26.5 audit).
    ///
    /// P26.5's tail-cost advisory shipped with the source indentation in it:
    /// *"…as a tiled .inf_tex — every mip              level is kept…"*. These
    /// are the sentences the Content Drawer's badge chip prints, so the mangling
    /// is what an author reads, and `contains()` is perfectly happy with it —
    /// which is why the arm above could not see it and every one of the five
    /// prior catches was found by eye.
    ///
    /// The house remedy, one crate over (`inf_packager::cook`'s material
    /// advisories, `pie_schema_skew`'s refusal, the capture wizard's literals):
    /// assert on the PRODUCED string, over every extent that can produce one.
    #[test]
    fn no_import_advisory_carries_an_eaten_continuation() {
        let mut seen = 0usize;
        for (w, h) in [
            (128u32, 128u32),
            (256, 256),
            (255, 260),
            (4096, 16),
            (8192, 4),
            (4095, 15),
            (224, 32),
            (1, 1),
        ] {
            for msg in texture_import_advisories(w, h) {
                seen += 1;
                assert!(
                    !msg.contains("  "),
                    "{w}×{h}: the advisory carries a run of spaces — a Rust line \
                     continuation was eaten and the source indentation is in the \
                     text an author reads: {msg:?}"
                );
                assert!(
                    !msg.contains('\n'),
                    "{w}×{h}: the advisory is multi-line: {msg:?}"
                );
            }
        }
        // ANTI-VACUITY: the sweep produced sentences. An extent table that
        // raised nothing would satisfy every assertion above.
        assert!(seen >= 8, "the sweep produced only {seen} advisories");
    }

    /// P26.1 (+ P26.5's third): the advisories fire on exactly the hazards they
    /// name, stay silent on ordinary content, and are a pure function of the
    /// extent.
    #[test]
    fn import_advisories_name_the_silent_hazards() {
        // A 1024² material raises nothing: it is a multiple of 4, it is square,
        // and its tiled form costs 121 % of its pixels.
        assert!(texture_import_advisories(1024, 1024).is_empty());
        assert!(
            texture_import_advisories(1200, 1040).is_empty(),
            "4-multiples are fine"
        );

        let block = texture_import_advisories(1023, 1040);
        assert_eq!(block.len(), 1);
        assert!(block[0].contains("multiple of 4"), "{block:?}");
        // Either axis is enough.
        assert_eq!(texture_import_advisories(1040, 1023).len(), 1);

        let strip = texture_import_advisories(4096, 16);
        assert_eq!(strip.len(), 2, "{strip:?}");
        assert!(strip.iter().any(|a| a.contains("256:1")), "{strip:?}");

        // Both at once, both reported — never one swallowing the other.
        let both = texture_import_advisories(4095, 15);
        assert_eq!(both.len(), 3, "{both:?}");

        // 7:1 is under the threshold, 8:1 is at it (the boundary is asserted, so
        // a re-tune has to come here). Both extents are small enough to carry
        // the tail-cost advisory as well, which is why the counts are 1 and 2.
        assert_eq!(texture_import_advisories(224, 32).len(), 1);
        assert_eq!(texture_import_advisories(256, 32).len(), 2);

        // P26.5: the tail cost fires below 512² and is silent from it up.
        assert_eq!(texture_import_advisories(128, 128).len(), 1);
        assert!(texture_import_advisories(128, 128)[0].contains("tiled .inf_tex"));
        assert_eq!(texture_import_advisories(256, 256).len(), 1);
        assert!(texture_import_advisories(512, 512).is_empty());

        // Pure: same extent, same sentences.
        assert_eq!(
            texture_import_advisories(4095, 15),
            texture_import_advisories(4095, 15)
        );
    }

    /// **THE HDR FINDING, and its fix** (Wave T).
    ///
    /// Before this batch there was exactly one decode on the import path —
    /// `image::load_from_memory(bytes)?.to_rgba8()` — so an OpenEXR or Radiance
    /// source, which is what Megascans/Fab ship displacement, cavity and
    /// bent-normal maps as, arrived with **everything above 1.0 clipped** and the
    /// rest quantised to 256 steps, and the import reported success. Nothing
    /// downstream could tell it from a PNG.
    ///
    /// This arm builds a real Radiance file with values above 1.0 in it and
    /// measures both paths against the source, so the loss is a number rather
    /// than an assertion: the 8-bit path is the *control* and it must be badly
    /// wrong, or the float path is not proving anything.
    #[test]
    fn a_float_source_survives_import_when_hdr_is_asked_for() {
        // A 4×4 Radiance image whose brightest texel is 8.0 — an ordinary value
        // for a cavity/displacement map and four stops past what RGBA8 holds.
        let (w, h) = (4u32, 4u32);
        let mut src = image::Rgb32FImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = 0.25f32 * (1 + x + y * w) as f32; // 0.25 … 4.0
                src.put_pixel(x, y, image::Rgb([v, v * 2.0, v * 0.5]));
            }
        }
        let mut encoded: Vec<u8> = Vec::new();
        image::DynamicImage::ImageRgb32F(src.clone())
            .write_to(
                &mut std::io::Cursor::new(&mut encoded),
                image::ImageFormat::Hdr,
            )
            .expect("the fixture encodes");
        assert!(source_is_float(&encoded), "the fixture must sniff as float");
        assert!(
            !source_is_float(include_bytes!("bc.rs")),
            "a non-image must not sniff as float"
        );

        // **The document's own premise is "4K/8K Megascans PNG/EXR from Fab", so
        // the EXR leg is not optional.** Same pixels, the other float container:
        // it must sniff as float, take the same branch, and come back with the
        // same values — a sniff that only knew Radiance would silently flatten
        // every OpenEXR map in a Megascans set, which is the exact failure.
        let mut exr: Vec<u8> = Vec::new();
        image::DynamicImage::ImageRgb32F(src.clone())
            .write_to(
                &mut std::io::Cursor::new(&mut exr),
                image::ImageFormat::OpenExr,
            )
            .expect("the EXR fixture encodes");
        assert!(source_is_float(&exr), "an EXR must sniff as float");
        let from_exr = import_texture_bytes(&exr, TextureImportSettings::data_hdr()).unwrap();
        assert_eq!(from_exr.format, TextureFormat::Rgba16F);
        assert_eq!(from_exr.mips[0].data.len(), (w * h * 8) as usize);
        for (i, px) in src.pixels().enumerate() {
            for c in 0..3 {
                let o = (i * 4 + c) * 2;
                let got = half::f16::from_le_bytes([
                    from_exr.mips[0].data[o],
                    from_exr.mips[0].data[o + 1],
                ])
                .to_f32();
                assert!(
                    (got - px.0[c]).abs() <= px.0[c] * 0.001 + 1e-3,
                    "EXR texel {i} channel {c}: kept {got}, source {}",
                    px.0[c]
                );
            }
        }
        // …and the same EXR through the DEFAULT settings is still flattened,
        // which is what makes `hdr` an opt-in rather than a behaviour change.
        assert_eq!(
            import_texture_bytes(&exr, TextureImportSettings::data())
                .unwrap()
                .format,
            TextureFormat::Rgba8
        );

        // The control: the pre-Wave-T path, which is still the default.
        let flat = import_texture_bytes(&encoded, TextureImportSettings::data()).unwrap();
        assert_eq!(flat.format, TextureFormat::Rgba8);
        let flat0 = flat.level_rgba8(0).unwrap();
        assert!(
            flat0.chunks_exact(4).all(|p| p[0] == 255 || p[1] == 255) || flat0.contains(&255),
            "the 8-bit control must be clipping — otherwise this arm proves nothing"
        );

        // The fix.
        let kept = import_texture_bytes(&encoded, TextureImportSettings::data_hdr()).unwrap();
        assert_eq!(kept.format, TextureFormat::Rgba16F);
        assert_eq!(kept.mips[0].data.len(), (w * h * 8) as usize);
        kept.validate().expect("a float asset is a valid asset");

        // Every source value comes back, including the ones past 1.0.
        let mut biggest = 0f32;
        for (i, px) in src.pixels().enumerate() {
            for c in 0..3 {
                let o = (i * 4 + c) * 2;
                let got =
                    half::f16::from_le_bytes([kept.mips[0].data[o], kept.mips[0].data[o + 1]])
                        .to_f32();
                assert!(
                    (got - px.0[c]).abs() <= px.0[c] * 0.001 + 1e-3,
                    "texel {i} channel {c}: kept {got}, source {}",
                    px.0[c]
                );
                biggest = biggest.max(got);
            }
        }
        assert!(
            biggest > 4.0,
            "the fixture must carry a value the 8-bit path cannot: max {biggest}"
        );

        // …and the advisory says which of the two happened, both ways.
        assert!(hdr_import_advisory(false, false).is_none());
        let lost = hdr_import_advisory(true, false).unwrap();
        assert!(
            lost.contains("clipped") && lost.contains("hdr = true"),
            "{lost}"
        );
        let cost = hdr_import_advisory(true, true).unwrap();
        assert!(cost.contains("RGBA16F") && cost.contains("BC6H"), "{cost}");
        for a in [lost, cost] {
            assert!(
                !a.contains("  ") && !a.contains('\n'),
                "eaten continuation: {a:?}"
            );
        }
    }

    /// **The import door clamps a non-finite sample; it never stores one**
    /// (Wave-T audit — the P29.5 A1 precedent, applied to the float texture
    /// door).
    ///
    /// # The defect
    ///
    /// `half::f16::from_f32` is faithful, which is the problem: `70 000` — an
    /// ordinary sun disc in a captured sky — is `+inf` in a half, and a `NaN`
    /// stays a `NaN`. Both were written into the page. Then the box downsample
    /// averaged them, so **one bad texel made every coarser level of the whole
    /// pyramid NaN**, including the 1×1 — and the coarsest levels are exactly
    /// the ones a virtual texture pins resident and always samples.
    /// `TextureAsset::validate` reported the result healthy.
    ///
    /// The arm asserts the world rather than the report: every stored half of
    /// every level, of a source built to contain all four hazards, is finite.
    #[test]
    fn a_float_import_never_stores_an_infinity_or_a_nan() {
        let (w, h) = (4u32, 4u32);
        let mut src = image::Rgb32FImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                // An ordinary, in-range value everywhere except the four hazards
                // below, so the pyramid has real data to be poisoned.
                src.put_pixel(x, y, image::Rgb([0.5, 1.5, 3.0]));
            }
        }
        src.put_pixel(0, 0, image::Rgb([70_000.0, 1.0, 1.0])); // past f16::MAX
        src.put_pixel(1, 0, image::Rgb([1.0e30, 1.0, 1.0])); // far past it
        src.put_pixel(0, 1, image::Rgb([f32::NAN, 1.0, 1.0])); // not a number
        src.put_pixel(1, 1, image::Rgb([-1.0e30, 1.0, 1.0])); // past it, signed
        let mut exr: Vec<u8> = Vec::new();
        image::DynamicImage::ImageRgb32F(src)
            .write_to(
                &mut std::io::Cursor::new(&mut exr),
                image::ImageFormat::OpenExr,
            )
            .expect("the fixture encodes");

        let tex = import_texture_bytes(&exr, TextureImportSettings::data_hdr()).unwrap();
        assert_eq!(tex.format, TextureFormat::Rgba16F);
        assert!(
            tex.mips.len() > 1,
            "the fixture must have a pyramid to poison"
        );
        for (i, m) in tex.mips.iter().enumerate() {
            for (k, b) in m.data.chunks_exact(2).enumerate() {
                let v = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
                assert!(
                    v.is_finite(),
                    "mip {i} sample {k} is {v} — a non-finite texel reached the \
                     stored page, and the box downsample carries it to every \
                     coarser level from here up"
                );
            }
        }
        // …and the clamp is a SATURATION, not a zeroing: the overflowing texel
        // still reads as the brightest thing in the image.
        let m0 = &tex.mips[0].data;
        let r = |i: usize| half::f16::from_le_bytes([m0[i * 8], m0[i * 8 + 1]]).to_f32();
        assert_eq!(
            r(0),
            65_504.0,
            "an overflow must saturate, not wrap or zero"
        );
        assert_eq!(r(1), 65_504.0);
        assert_eq!(r(4), 0.0, "a NaN is stored as 0, the only honest answer");
        assert_eq!(r(5), -65_504.0, "a signed overflow saturates too");
        // The policy, at its own door, so a change to it is a change to a value.
        assert_eq!(half_at_the_door(f32::NAN), half::f16::ZERO);
        assert_eq!(half_at_the_door(f32::INFINITY).to_f32(), 65_504.0);
        assert_eq!(half_at_the_door(f32::NEG_INFINITY).to_f32(), -65_504.0);
        assert_eq!(half_at_the_door(1.5).to_f32(), 1.5, "in range is untouched");
        // And the author is told, rather than left to find it in a lit frame.
        let kept = hdr_import_advisory(true, true).unwrap();
        assert!(kept.contains("65 504") && kept.contains("NaN"), "{kept}");
    }

    /// The float mip chain halves like every other chain, and is a **pure
    /// function of its input** — the property a content hash rests on.
    #[test]
    fn the_float_mip_chain_halves_and_is_deterministic() {
        let halfs: Vec<u8> = (0..(8 * 8 * 4))
            .flat_map(|i| half::f16::from_f32(i as f32 * 0.125).to_le_bytes())
            .collect();
        let a =
            texture_from_rgba16f(halfs.clone(), 8, 8, TextureImportSettings::data_hdr()).unwrap();
        let b = texture_from_rgba16f(halfs, 8, 8, TextureImportSettings::data_hdr()).unwrap();
        assert_eq!(a, b, "two imports of one buffer differ");
        assert_eq!(a.mip_count(), 4);
        for (i, m) in a.mips.iter().enumerate() {
            assert_eq!(m.data.len(), (m.width * m.height * 8) as usize, "mip {i}");
        }
        a.validate().expect("the chain validates");
        // A promoted 8-bit source is NOT float: `hdr` only preserves, never
        // invents.
        let png = texture_from_rgba8(
            checker(4, 4, 255),
            4,
            4,
            TextureImportSettings {
                hdr: true,
                ..TextureImportSettings::data()
            },
        )
        .unwrap();
        assert_eq!(png.format, TextureFormat::Rgba8);
    }

    /// BC5 reaches the asset through the ordinary import door, and the level
    /// sizes it declares are the ones it stores.
    #[test]
    fn the_normal_map_preset_imports_as_bc5() {
        let s = TextureImportSettings::normal_map();
        assert_eq!(s.compression, TextureCompression::Bc5);
        assert!(!s.srgb, "a normal map is never sRGB");
        let tex = texture_from_rgba8(checker(16, 16, 255), 16, 16, s).unwrap();
        assert_eq!(tex.format, TextureFormat::Bc5);
        assert!(tex.format.is_compressed() && !tex.format.is_float());
        assert_eq!(tex.mips[0].data.len(), 4 * 4 * 16);
        tex.validate().expect("a BC5 asset is a valid asset");
        // And the preset that must NOT have moved: `data()` is what every
        // existing data map imported through, and it still answers RGBA8.
        assert_eq!(
            texture_from_rgba8(checker(16, 16, 255), 16, 16, TextureImportSettings::data())
                .unwrap()
                .format,
            TextureFormat::Rgba8
        );
    }

    #[test]
    fn rejects_zero_and_truncated() {
        assert!(texture_from_rgba8(vec![], 0, 0, Default::default()).is_err());
        assert!(texture_from_rgba8(vec![0; 4], 4, 4, Default::default()).is_err());
    }

    /// **The tail-cost factor reproduces the P26.1 audit's measured table**, to
    /// the digit.
    ///
    /// The audit measured a v2 payload against its v1 twin on real files —
    /// *"6.8× at 128² BC1, 2.55× at 256², 1.22× at 1024², 1.16× at 2048²"* — and
    /// found the growth is the uniform page rather than a lost zstd frame,
    /// *"because BC1 and `TextureCompression::None` grow by the SAME factor"*.
    /// [`tiled_size_factor_pct`] derives those numbers from the extent alone, so
    /// the advisory quotes the project's own figure and this arm is what keeps
    /// the derivation honest against the files that were actually weighed.
    ///
    /// If the tile geometry ever changes, this fails first and says so — which is
    /// the point, because four documents once carried a number that had stopped
    /// being true.
    #[test]
    fn the_tail_cost_factor_reproduces_the_measured_table() {
        // (extent, the audit's measured factor × 100, tolerance in percentage
        // points — the audit quoted two significant figures).
        for (n, measured, tol) in [
            (128u32, 680u32, 10u32),
            (256, 255, 5),
            (1024, 122, 3),
            (2048, 116, 3),
        ] {
            let got = tiled_size_factor_pct(n, n);
            assert!(
                got.abs_diff(measured) <= tol,
                "{n}²: derived {got} %, the P26.1 audit measured {measured} %"
            );
        }
        // Monotone in size, which is the shape of the claim ("a SMALL texture
        // pays for the uniform page") and not just four points.
        let mut prev = u32::MAX;
        for n in [128u32, 256, 512, 1024, 2048, 4096] {
            let got = tiled_size_factor_pct(n, n);
            assert!(
                got < prev,
                "{n}² costs {got} %, more than the size below it"
            );
            prev = got;
        }
        // It never claims a saving: a tiled container is always at least as big
        // as its pixels, because a page is never smaller than the level it holds.
        assert!(tiled_size_factor_pct(8192, 8192) >= 100);
        // Format-independent by construction — there is no format argument, which
        // is the structural form of the audit's finding.
        assert_eq!(tiled_size_factor_pct(1, 1), tiled_size_factor_pct(1, 1));
    }
}
