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
/// Two hazards are visible from the extent alone, and both are silent otherwise:
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
    out
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

    /// P26.1: the advisories fire on exactly the two hazards they name, stay
    /// silent on ordinary content, and are a pure function of the extent.
    #[test]
    fn import_advisories_name_the_two_silent_hazards() {
        assert!(texture_import_advisories(256, 256).is_empty());
        assert!(
            texture_import_advisories(300, 260).is_empty(),
            "4-multiples are fine"
        );

        let block = texture_import_advisories(255, 260);
        assert_eq!(block.len(), 1);
        assert!(block[0].contains("multiple of 4"), "{block:?}");
        // Either axis is enough.
        assert_eq!(texture_import_advisories(260, 255).len(), 1);

        let strip = texture_import_advisories(4096, 16);
        assert_eq!(strip.len(), 1);
        assert!(strip[0].contains("256:1"), "{strip:?}");

        // Both at once, both reported — never one swallowing the other.
        let both = texture_import_advisories(4095, 15);
        assert_eq!(both.len(), 2, "{both:?}");

        // 7:1 is under the threshold, 8:1 is at it (the boundary is asserted, so
        // a re-tune has to come here).
        assert!(texture_import_advisories(224, 32).is_empty());
        assert_eq!(texture_import_advisories(256, 32).len(), 1);

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
}
