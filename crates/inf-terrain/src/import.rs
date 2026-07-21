//! Heightmap import/export: 16-bit PNG and EXR in, 16-bit PNG out.
//!
//! Both formats decode through one `image` path (`DynamicImage::to_luma16`), so a
//! `PNG16` file maps 1:1 and an `EXR` float file is imported normalized-clamped to
//! `[0, 1]`. The `[min_height, max_height]` range maps the normalized `[0, 1]`
//! sample onto metres — the World-Machine / Gaea convention where the full 16-bit
//! range spans the terrain's elevation extent. Absolute-range EXR (raw float
//! metres) is a documented follow-up.

use std::io::Cursor;

use glam::DVec3;
use image::{DynamicImage, ImageBuffer, Luma};

use crate::data::TerrainData;

/// Import settings mapping a decoded heightmap onto world metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightmapImport {
    /// Samples per tile side of the produced terrain.
    pub tile_resolution: u32,
    /// World units between samples.
    pub meters_per_sample: f64,
    /// Elevation (metres) the normalized value `0.0` maps to.
    pub min_height: f64,
    /// Elevation (metres) the normalized value `1.0` maps to.
    pub max_height: f64,
}

impl Default for HeightmapImport {
    fn default() -> Self {
        Self {
            tile_resolution: crate::DEFAULT_TILE_RESOLUTION,
            meters_per_sample: crate::DEFAULT_METERS_PER_SAMPLE,
            min_height: 0.0,
            max_height: 1.0,
        }
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
}

impl TerrainData {
    /// Decode a `PNG16`/`EXR` heightmap and page it into a [`TerrainData`].
    ///
    /// The `W × H` sample grid is tiled into `resolution × resolution` pages with
    /// **shared edges** (a tile's last row/column is the next tile's first), so the
    /// result is seamless. Sample values are `value / 65535` mapped onto
    /// `[min_height, max_height]` (stored as the tile-local `f32` offset, `origin.y
    /// = 0`).
    pub fn from_height_image(bytes: &[u8], import: HeightmapImport) -> Result<Self, TerrainError> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| TerrainError::Image(e.to_string()))?
            .to_luma16();
        let (w, h) = img.dimensions();
        if w == 0 || h == 0 {
            return Err(TerrainError::Empty);
        }
        let raw = img.into_raw();
        let res = import.tile_resolution.max(2);
        let cells = (res - 1) as i32; // world/tile cell span in samples
        let span_m = import.max_height - import.min_height;

        // A W×H grid tiled into `cells`-cell tiles needs ceil((dim-1)/cells) tiles.
        let ntx = (((w as i32 - 1).max(0) + cells - 1) / cells).max(1);
        let ntz = (((h as i32 - 1).max(0) + cells - 1) / cells).max(1);

        let mut data = TerrainData::new(res, import.meters_per_sample);
        let sample_at = |gx: i32, gz: i32| -> f64 {
            let gx = gx.clamp(0, w as i32 - 1) as u32;
            let gz = gz.clamp(0, h as i32 - 1) as u32;
            let v = raw[(gz * w + gx) as usize] as f64 / 65535.0;
            import.min_height + v * span_m
        };
        for tz in 0..ntz {
            for tx in 0..ntx {
                let coord = (tx, tz);
                let o = data.tile_origin_xz(coord);
                let tile = data.get_or_create_tile(coord);
                tile.origin = DVec3::new(o.x, 0.0, o.y);
                for j in 0..res {
                    for i in 0..res {
                        let gx = tx * cells + i as i32;
                        let gz = tz * cells + j as i32;
                        tile.set_sample(res, i, j, sample_at(gx, gz) as f32);
                    }
                }
            }
        }
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
