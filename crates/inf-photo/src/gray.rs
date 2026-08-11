//! Grayscale images and the scale pyramid the detector runs on.
//!
//! Photogrammetry does not need colour. Every classical feature detector works
//! on intensity, and carrying three channels through the pyramid triples the
//! memory for nothing — so a photograph becomes an 8-bit [`GrayImage`] the
//! moment it is decoded, and colour is re-read from the original file later,
//! when P25.3 bakes albedo.
//!
//! # Determinism
//!
//! * **Luma conversion** is the `image` crate's `to_luma8`, which is an integer
//!   Rec.601 weighting — the same bytes on every platform.
//! * **The blur is integer**: a separable `[1, 4, 6, 4, 1] / 16` kernel with
//!   clamp-to-edge, evaluated in `u32`. Both passes are accumulated unnormalised
//!   and divided once at the end — `(acc + 128) / 256`, a round-to-nearest over
//!   the two `/16` passes together, so the intermediate is exact rather than
//!   rounded twice. No float rounding mode can touch it.
//! * **Downsampling is bilinear in `f64`** with `round()` (round-half-away-from-
//!   zero, an exact operation) at the end.
//!
//! # Why blur before describing
//!
//! A BRIEF-class descriptor compares single pixels, which makes it *very*
//! sensitive to sensor noise and to the sub-pixel phase of a resample. Blurring
//! the level once, before sampling, is what makes the comparison stable — it is
//! part of the descriptor's definition, not an optional pre-filter, so
//! [`Pyramid`] carries the blurred level beside the sharp one and the detector
//! uses the sharp one while the descriptor uses the blurred one.

use crate::PhotoError;

/// An 8-bit single-channel image, row-major, origin top-left.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrayImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl GrayImage {
    /// An all-black image.
    pub fn new(width: u32, height: u32) -> Result<Self, PhotoError> {
        if width == 0 || height == 0 {
            return Err(PhotoError::EmptyImage { width, height });
        }
        Ok(Self {
            width,
            height,
            pixels: vec![0u8; width as usize * height as usize],
        })
    }

    /// Wrap an existing row-major luma buffer.
    pub fn from_luma(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, PhotoError> {
        if width == 0 || height == 0 {
            return Err(PhotoError::EmptyImage { width, height });
        }
        let expected = width as usize * height as usize;
        if pixels.len() != expected {
            return Err(PhotoError::PixelCount {
                width,
                height,
                expected,
                given: pixels.len(),
            });
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    /// Decode a photograph from disk and convert it to grayscale.
    ///
    /// This is the crate's only filesystem door. Formats are whatever the
    /// workspace `image` pin enables (PNG, JPEG, TGA, BMP, HDR, EXR).
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, PhotoError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| PhotoError::Read {
            path: path.display().to_string(),
            source,
        })?;
        let decoded = image::load_from_memory(&bytes).map_err(|e| PhotoError::Decode {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        let luma = decoded.to_luma8();
        let (width, height) = (luma.width(), luma.height());
        Self::from_luma(width, height, luma.into_raw())
    }

    /// Width in pixels.
    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in pixels.
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// The raw row-major buffer.
    #[inline]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// The pixel at `(x, y)`, unchecked against the bounds the caller has
    /// already established. Panics in debug builds if out of range.
    #[inline]
    pub fn at(&self, x: u32, y: u32) -> u8 {
        debug_assert!(x < self.width && y < self.height);
        self.pixels[y as usize * self.width as usize + x as usize]
    }

    /// The pixel at `(x, y)` with **clamp-to-edge** addressing.
    #[inline]
    pub fn clamped(&self, x: i64, y: i64) -> u8 {
        let x = x.clamp(0, self.width as i64 - 1) as u32;
        let y = y.clamp(0, self.height as i64 - 1) as u32;
        self.at(x, y)
    }

    /// Write a pixel.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, v: u8) {
        debug_assert!(x < self.width && y < self.height);
        let w = self.width as usize;
        self.pixels[y as usize * w + x as usize] = v;
    }

    /// A separable `[1, 4, 6, 4, 1] / 16` blur, in integers, clamp-to-edge.
    pub fn blur_5x5(&self) -> GrayImage {
        const K: [u32; 5] = [1, 4, 6, 4, 1];
        let w = self.width as i64;
        let h = self.height as i64;
        let mut horizontal = vec![0u16; self.pixels.len()];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0u32;
                for (i, k) in K.iter().enumerate() {
                    acc += k * self.clamped(x + i as i64 - 2, y) as u32;
                }
                horizontal[(y * w + x) as usize] = acc as u16;
            }
        }
        let mut out = vec![0u8; self.pixels.len()];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0u32;
                for (i, k) in K.iter().enumerate() {
                    let yy = (y + i as i64 - 2).clamp(0, h - 1);
                    acc += k * horizontal[(yy * w + x) as usize] as u32;
                }
                // Two passes of a /16 kernel: divide by 256, round to nearest.
                out[(y * w + x) as usize] = ((acc + 128) / 256).min(255) as u8;
            }
        }
        GrayImage {
            width: self.width,
            height: self.height,
            pixels: out,
        }
    }

    /// Bilinear resample to `(width, height)`.
    ///
    /// Samples on the **pixel-centre** convention: destination centre
    /// `(x + 0.5) / dst_w` maps to source `(x + 0.5) * src_w / dst_w - 0.5`.
    /// Getting that half-pixel wrong biases every feature's level-0 coordinate
    /// by half a pixel per octave, which shows up as a systematic pose error and
    /// nowhere else.
    pub fn resample(&self, width: u32, height: u32) -> Result<GrayImage, PhotoError> {
        if width == 0 || height == 0 {
            return Err(PhotoError::EmptyImage { width, height });
        }
        let sx = self.width as f64 / width as f64;
        let sy = self.height as f64 / height as f64;
        let mut out = vec![0u8; width as usize * height as usize];
        for y in 0..height {
            let fy = (y as f64 + 0.5) * sy - 0.5;
            let y0 = fy.floor();
            let ty = fy - y0;
            let y0 = y0 as i64;
            for x in 0..width {
                let fx = (x as f64 + 0.5) * sx - 0.5;
                let x0 = fx.floor();
                let tx = fx - x0;
                let x0 = x0 as i64;
                let p00 = self.clamped(x0, y0) as f64;
                let p10 = self.clamped(x0 + 1, y0) as f64;
                let p01 = self.clamped(x0, y0 + 1) as f64;
                let p11 = self.clamped(x0 + 1, y0 + 1) as f64;
                let top = p00 + (p10 - p00) * tx;
                let bottom = p01 + (p11 - p01) * tx;
                let v = top + (bottom - top) * ty;
                out[y as usize * width as usize + x as usize] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        Ok(GrayImage {
            width,
            height,
            pixels: out,
        })
    }
}

/// One level of a scale pyramid.
#[derive(Clone, Debug)]
pub struct PyramidLevel {
    /// The resampled image, used for detection.
    pub image: GrayImage,
    /// The same image blurred, used for descriptor sampling.
    pub blurred: GrayImage,
    /// Level-0 pixels per level pixel: multiply a level coordinate by this to
    /// get a level-0 coordinate. Level 0 has a scale of exactly `1.0`.
    pub scale: f64,
}

/// A multi-scale pyramid: successively smaller resamples of one photograph.
#[derive(Clone, Debug)]
pub struct Pyramid {
    /// Coarsening levels, level 0 first.
    pub levels: Vec<PyramidLevel>,
}

impl Pyramid {
    /// Build `levels` levels, each `factor` times smaller in each axis than the
    /// one before. A level whose smaller side would fall under `min_side` is not
    /// built, so a small photograph simply gets a shorter pyramid rather than a
    /// degenerate one.
    pub fn build(
        image: &GrayImage,
        levels: u32,
        factor: f64,
        min_side: u32,
    ) -> Result<Self, PhotoError> {
        assert!(factor > 1.0, "a pyramid factor must be greater than one");
        let mut out = Vec::new();
        let mut scale = 1.0f64;
        for level in 0..levels.max(1) {
            let (w, h) = if level == 0 {
                (image.width(), image.height())
            } else {
                (
                    (image.width() as f64 / scale).round().max(1.0) as u32,
                    (image.height() as f64 / scale).round().max(1.0) as u32,
                )
            };
            // Level 0 is never dropped: a photograph too small for the pyramid
            // is still a photograph, and refusing it here would be a refusal
            // with no name attached to it.
            if level > 0 && w.min(h) < min_side {
                break;
            }
            let resampled = if level == 0 {
                image.clone()
            } else {
                image.resample(w, h)?
            };
            // The *effective* scale is the realised ratio, not the requested
            // one: `round()` above means a 320-wide level 1 at factor 1.2 is
            // 267 px, and 320/267 is not 1.2. Using the requested factor here
            // would put every level-1 feature a fraction of a pixel out.
            let effective = image.width() as f64 / w as f64;
            let blurred = resampled.blur_5x5();
            out.push(PyramidLevel {
                image: resampled,
                blurred,
                scale: effective,
            });
            scale *= factor;
        }
        assert!(!out.is_empty(), "a pyramid always has its base level");
        Ok(Pyramid { levels: out })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: u32, h: u32) -> GrayImage {
        let mut img = GrayImage::new(w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, ((x * 7 + y * 13) % 256) as u8);
            }
        }
        img
    }

    #[test]
    fn empty_and_mismatched_buffers_are_refused() {
        assert!(matches!(
            GrayImage::new(0, 4),
            Err(PhotoError::EmptyImage { .. })
        ));
        assert!(matches!(
            GrayImage::from_luma(4, 4, vec![0; 15]),
            Err(PhotoError::PixelCount {
                expected: 16,
                given: 15,
                ..
            })
        ));
    }

    #[test]
    fn clamped_addressing_never_wraps() {
        let img = ramp(4, 3);
        assert_eq!(img.clamped(-5, -5), img.at(0, 0));
        assert_eq!(img.clamped(100, 100), img.at(3, 2));
        assert_eq!(img.clamped(2, -1), img.at(2, 0));
    }

    #[test]
    fn blur_preserves_a_constant_image_exactly() {
        let mut img = GrayImage::new(9, 7).unwrap();
        for y in 0..7 {
            for x in 0..9 {
                img.set(x, y, 137);
            }
        }
        let b = img.blur_5x5();
        for y in 0..7 {
            for x in 0..9 {
                assert_eq!(b.at(x, y), 137, "blur shifted a flat field at ({x},{y})");
            }
        }
    }

    #[test]
    fn blur_actually_blurs() {
        let mut img = GrayImage::new(9, 9).unwrap();
        img.set(4, 4, 255);
        let b = img.blur_5x5();
        // 6*6/256 of the impulse stays at the centre; the ring is non-zero.
        assert_eq!(b.at(4, 4), ((36 * 255 + 128) / 256) as u8);
        assert!(b.at(3, 4) > 0 && b.at(5, 4) > 0 && b.at(4, 3) > 0 && b.at(4, 5) > 0);
        assert_eq!(b.at(0, 0), 0, "the impulse leaked past the kernel radius");
    }

    #[test]
    fn resample_keeps_a_flat_field_flat_and_halves_dimensions() {
        let mut img = GrayImage::new(16, 12).unwrap();
        for y in 0..12 {
            for x in 0..16 {
                img.set(x, y, 200);
            }
        }
        let half = img.resample(8, 6).unwrap();
        assert_eq!((half.width(), half.height()), (8, 6));
        assert!(half.pixels().iter().all(|&p| p == 200));
    }

    #[test]
    fn resample_centres_are_not_biased() {
        // A horizontal ramp resampled 2:1 must land on the average of each pair,
        // which is what the half-pixel convention buys. An off-by-half maps
        // destination 0 to source 0 and shifts the whole image.
        let mut img = GrayImage::new(8, 1).unwrap();
        for x in 0..8 {
            img.set(x, 0, (x * 20) as u8);
        }
        let half = img.resample(4, 1).unwrap();
        for x in 0..4 {
            let want = ((x * 2) * 20 + (x * 2 + 1) * 20) as f64 / 2.0;
            assert!(
                (half.at(x, 0) as f64 - want).abs() <= 0.5,
                "resample bias at {x}: {} vs {want}",
                half.at(x, 0)
            );
        }
    }

    #[test]
    fn resample_is_bit_identical_across_calls() {
        let img = ramp(37, 29);
        let a = img.resample(21, 17).unwrap();
        let b = img.resample(21, 17).unwrap();
        assert_eq!(a.pixels(), b.pixels());
    }

    #[test]
    fn the_pyramid_reports_the_scale_it_actually_realised() {
        let img = ramp(320, 240);
        let p = Pyramid::build(&img, 4, 1.2, 32).unwrap();
        assert_eq!(p.levels.len(), 4);
        assert_eq!(p.levels[0].scale, 1.0);
        for lvl in &p.levels {
            let realised = 320.0 / lvl.image.width() as f64;
            assert_eq!(
                lvl.scale, realised,
                "scale {} is not the realised ratio {realised}",
                lvl.scale
            );
        }
        assert!(p.levels[3].image.width() < p.levels[0].image.width());
    }

    #[test]
    fn the_pyramid_stops_rather_than_degenerating() {
        let img = ramp(40, 40);
        let p = Pyramid::build(&img, 8, 2.0, 16).unwrap();
        // 40 -> 20 -> 10 (below 16, dropped): two levels.
        assert_eq!(p.levels.len(), 2, "levels: {:?}", p.levels.len());
        assert!(p.levels.iter().all(|l| l.image.width() >= 16));
        // Even a tiny image keeps its base level.
        let tiny = ramp(8, 8);
        assert_eq!(Pyramid::build(&tiny, 4, 1.2, 16).unwrap().levels.len(), 1);
    }
}
