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
}

impl TextureFormat {
    /// Bytes one mip level of `w×h` occupies in this format.
    pub fn level_size(self, w: u32, h: u32) -> usize {
        match self {
            TextureFormat::Rgba8 => (w * h * 4) as usize,
            TextureFormat::Bc1 => bc::compressed_size(w, h, false),
            TextureFormat::Bc3 => bc::compressed_size(w, h, true),
        }
    }
    pub fn is_compressed(self) -> bool {
        !matches!(self, TextureFormat::Rgba8)
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
        })
    }
}

impl AssetPayload for TextureAsset {
    const KIND: AssetKind = AssetKind::Texture;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
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
}

impl Default for TextureImportSettings {
    fn default() -> Self {
        Self {
            srgb: true,
            generate_mips: true,
            compression: TextureCompression::Auto,
        }
    }
}

impl TextureImportSettings {
    /// Preset for non-color data (normal maps, masks): linear, uncompressed by
    /// default (BC introduces artifacts in normals; a BC5 path is future work).
    pub fn data() -> Self {
        Self {
            srgb: false,
            generate_mips: true,
            compression: TextureCompression::None,
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

/// Decode an encoded image (PNG/JPEG/…) and import it with `settings`.
pub fn import_texture_bytes(
    bytes: &[u8],
    settings: TextureImportSettings,
) -> Result<TextureAsset, MaterialError> {
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
    if rgba.len() < (width * height * 4) as usize {
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

/// The storage format an import setting resolves to for a given level-0 image.
/// Shared with the v2 tiler for the same reason [`rgba_mip_chain`] is.
pub(crate) fn choose_format(compression: TextureCompression, level0: &[u8]) -> TextureFormat {
    match compression {
        TextureCompression::None => TextureFormat::Rgba8,
        TextureCompression::Bc1 => TextureFormat::Bc1,
        TextureCompression::Bc3 => TextureFormat::Bc3,
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
