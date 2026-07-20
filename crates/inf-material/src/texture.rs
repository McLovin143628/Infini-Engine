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

/// Decode an encoded image (PNG/JPEG/…) and import it with `settings`.
pub fn import_texture_bytes(
    bytes: &[u8],
    settings: TextureImportSettings,
) -> Result<TextureAsset, MaterialError> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| MaterialError::Image(e.to_string()))?
        .to_rgba8();
    let (w, h) = img.dimensions();
    texture_from_rgba8(img.into_raw(), w, h, settings)
}

/// Import an already-decoded RGBA8 buffer (e.g. a glTF-embedded image).
pub fn texture_from_rgba8(
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    settings: TextureImportSettings,
) -> Result<TextureAsset, MaterialError> {
    if width == 0 || height == 0 {
        return Err(MaterialError::Image("zero-sized texture".into()));
    }
    if rgba.len() < (width * height * 4) as usize {
        return Err(MaterialError::Image("truncated pixel buffer".into()));
    }

    // Build the RGBA8 mip chain first (compression is applied per level after).
    let mut levels: Vec<(u32, u32, Vec<u8>)> = vec![(width, height, rgba)];
    if settings.generate_mips {
        while levels.last().unwrap().0 > 1 || levels.last().unwrap().1 > 1 {
            let (pw, ph, ref prev) = *levels.last().unwrap();
            let (nw, nh) = ((pw / 2).max(1), (ph / 2).max(1));
            let down = downsample_box(prev, pw, ph, nw, nh);
            levels.push((nw, nh, down));
        }
    }

    let format = match settings.compression {
        TextureCompression::None => TextureFormat::Rgba8,
        TextureCompression::Bc1 => TextureFormat::Bc1,
        TextureCompression::Bc3 => TextureFormat::Bc3,
        TextureCompression::Auto => {
            if is_fully_opaque(&levels[0].2) {
                TextureFormat::Bc1
            } else {
                TextureFormat::Bc3
            }
        }
    };

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

    #[test]
    fn rejects_zero_and_truncated() {
        assert!(texture_from_rgba8(vec![], 0, 0, Default::default()).is_err());
        assert!(texture_from_rgba8(vec![0; 4], 4, 4, Default::default()).is_err());
    }
}
