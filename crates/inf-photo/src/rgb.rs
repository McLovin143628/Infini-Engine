//! **Colour**, re-read from the original photograph — the sibling
//! [`crate::gray`] promised.
//!
//! # Why colour arrives second
//!
//! [`GrayImage`]'s module docs state the rule: photogrammetry does not need
//! colour. Every classical detector works on intensity, and carrying three
//! channels through the pyramid triples the memory for nothing. So a photograph
//! becomes 8-bit luma the moment it is decoded, poses and depth are solved on
//! luma, **and colour is re-read from the original file later** — when P25.3
//! bakes albedo, which is the only stage in the whole pipeline that wants it.
//!
//! This module is that "later". It is a decoder and a container and nothing
//! else: no pyramid, no blur, no resample. A bake samples an [`RgbImage`] at
//! projected pixel coordinates and that is the entire contract.
//!
//! # One luma rule, in one place
//!
//! [`RgbImage::to_gray`] exists so a caller that has already decoded a file for
//! its colour does not decode it a second time for its intensity — the shape
//! P25.4's wizard wants, where a dropped photograph is read once. It routes
//! through the **same** `image` crate conversion [`GrayImage::from_bytes`] uses
//! rather than spelling a weighting out again, because two integer luma
//! formulae that agree today are two that can disagree after a dependency bump,
//! and the feature detector would then find its corners in a slightly different
//! place depending on which door the caller came through.
//!
//! `the_two_doors_to_luma_agree` measures that on a round-tripped PNG rather
//! than trusting the sentence above.
//!
//! # Colour space, stated once
//!
//! These bytes are **whatever the file holds** — for an ordinary photograph,
//! sRGB-encoded. Nothing here linearizes, and nothing here should: the albedo
//! bake blends observations of one surface across views, and a blend of sRGB
//! bytes is what the P4 texture convention (`srgb: true` on a base-colour
//! `.inf_tex`) expects to receive. Any stage that needs *linear* radiance — the
//! de-lighting solve is the one that does — converts explicitly at its own door
//! and says so there.

use crate::gray::GrayImage;
use crate::PhotoError;

/// An 8-bit **three-channel** image, row-major, origin top-left, interleaved
/// `[r, g, b]`.
///
/// Interleaved rather than planar because every consumer reads all three
/// channels of one pixel at once (a bake projects a point and wants its colour),
/// and because it is the layout `image` decodes into, so wrapping a decode is a
/// move rather than a shuffle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbImage {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl RgbImage {
    /// Channels per pixel. Named because `* 3` appears in every bounds check
    /// and an off-by-one there is an image that decodes green.
    pub const CHANNELS: usize = 3;

    /// An all-black image.
    pub fn new(width: u32, height: u32) -> Result<Self, PhotoError> {
        if width == 0 || height == 0 {
            return Err(PhotoError::EmptyImage { width, height });
        }
        Ok(Self {
            width,
            height,
            pixels: vec![0u8; width as usize * height as usize * Self::CHANNELS],
        })
    }

    /// Wrap an existing row-major interleaved RGB buffer.
    pub fn from_rgb(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, PhotoError> {
        if width == 0 || height == 0 {
            return Err(PhotoError::EmptyImage { width, height });
        }
        let expected = width as usize * height as usize * Self::CHANNELS;
        if pixels.len() != expected {
            return Err(PhotoError::PixelCount {
                width,
                height,
                // `PixelCount` counts **bytes** for this type, and the message
                // says so by carrying the byte figure rather than the pixel one:
                // a buffer three bytes short of an RGB image is not "one pixel
                // short", and reporting it as a pixel count would send the
                // reader looking for a dimension error.
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

    /// Decode an in-memory photograph as RGB.
    ///
    /// Split out from [`RgbImage::load`] so the round-trip agreement with
    /// [`GrayImage`] can be measured without a filesystem — a test that writes
    /// a temporary file to check a colour conversion is a test that fails on a
    /// read-only checkout for a reason that has nothing to do with colour.
    pub fn from_bytes(bytes: &[u8], name: &str) -> Result<Self, PhotoError> {
        let decoded = image::load_from_memory(bytes).map_err(|e| PhotoError::Decode {
            path: name.to_string(),
            message: e.to_string(),
        })?;
        let rgb = decoded.to_rgb8();
        let (width, height) = (rgb.width(), rgb.height());
        Self::from_rgb(width, height, rgb.into_raw())
    }

    /// Decode a photograph from disk in colour.
    ///
    /// The sibling of [`GrayImage::load`], and formats are the same set: what
    /// the workspace `image` pin enables (PNG, JPEG, TGA, BMP, HDR, EXR). A
    /// float source (HDR, EXR) is tone-mapped to 8 bits by `image`'s own
    /// conversion — this crate does not own a tone mapper and does not pretend
    /// to.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, PhotoError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| PhotoError::Read {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_bytes(&bytes, &path.display().to_string())
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

    /// The raw interleaved buffer, `width * height * 3` bytes.
    #[inline]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// The pixel at `(x, y)`. Panics in debug builds if out of range.
    #[inline]
    pub fn at(&self, x: u32, y: u32) -> [u8; 3] {
        debug_assert!(x < self.width && y < self.height);
        let i = (y as usize * self.width as usize + x as usize) * Self::CHANNELS;
        [self.pixels[i], self.pixels[i + 1], self.pixels[i + 2]]
    }

    /// The pixel at `(x, y)` with **clamp-to-edge** addressing, the same rule
    /// [`GrayImage::clamped`] uses, so a projection that lands a hair outside
    /// the frame reads the border rather than refusing.
    #[inline]
    pub fn clamped(&self, x: i64, y: i64) -> [u8; 3] {
        let x = x.clamp(0, self.width as i64 - 1) as u32;
        let y = y.clamp(0, self.height as i64 - 1) as u32;
        self.at(x, y)
    }

    /// Write a pixel.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, v: [u8; 3]) {
        debug_assert!(x < self.width && y < self.height);
        let i = (y as usize * self.width as usize + x as usize) * Self::CHANNELS;
        self.pixels[i] = v[0];
        self.pixels[i + 1] = v[1];
        self.pixels[i + 2] = v[2];
    }

    /// Convert to luma through the **same** `image` conversion the gray decoder
    /// uses.
    ///
    /// Not a hand-written weighting. See the module docs: the detector's corner
    /// positions depend on which luma a caller handed it, so the pipeline gets
    /// exactly one answer to "what is the intensity of this pixel".
    pub fn to_gray(&self) -> GrayImage {
        let buffer = image::RgbImage::from_raw(self.width, self.height, self.pixels.clone())
            .expect("from_rgb checked the buffer length, which is the only thing from_raw checks");
        let luma = image::DynamicImage::ImageRgb8(buffer).to_luma8();
        GrayImage::from_luma(self.width, self.height, luma.into_raw())
            .expect("a luma image derived from a non-empty RGB one has the same dimensions")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: u32, h: u32) -> RgbImage {
        let mut img = RgbImage::new(w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                img.set(
                    x,
                    y,
                    [
                        ((x * 7) % 256) as u8,
                        ((y * 13) % 256) as u8,
                        ((x * 3 + y * 5) % 256) as u8,
                    ],
                );
            }
        }
        img
    }

    fn png_bytes(img: &RgbImage) -> Vec<u8> {
        let buffer = image::RgbImage::from_raw(img.width(), img.height(), img.pixels().to_vec())
            .expect("valid buffer");
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(buffer)
            .write_to(&mut out, image::ImageFormat::Png)
            .expect("png encode");
        out.into_inner()
    }

    #[test]
    fn empty_and_mismatched_buffers_are_refused() {
        assert!(matches!(
            RgbImage::new(0, 4),
            Err(PhotoError::EmptyImage { .. })
        ));
        assert!(matches!(
            RgbImage::new(4, 0),
            Err(PhotoError::EmptyImage { .. })
        ));
        // Three bytes short of a 2x2 RGB image: 12 expected, 9 given.
        assert!(matches!(
            RgbImage::from_rgb(2, 2, vec![0; 9]),
            Err(PhotoError::PixelCount {
                expected: 12,
                given: 9,
                ..
            })
        ));
    }

    #[test]
    fn channels_do_not_bleed_into_each_other() {
        // The failure this catches is an index computed with `* 2` or without
        // the channel stride: it does not crash, it produces a green image.
        let mut img = RgbImage::new(3, 2).unwrap();
        img.set(2, 1, [10, 20, 30]);
        img.set(0, 0, [1, 2, 3]);
        assert_eq!(img.at(2, 1), [10, 20, 30]);
        assert_eq!(img.at(0, 0), [1, 2, 3]);
        assert_eq!(img.at(1, 0), [0, 0, 0], "a neighbour was written");
        assert_eq!(img.at(1, 1), [0, 0, 0], "a neighbour was written");
    }

    #[test]
    fn clamped_addressing_never_wraps() {
        let img = ramp(4, 3);
        assert_eq!(img.clamped(-5, -5), img.at(0, 0));
        assert_eq!(img.clamped(100, 100), img.at(3, 2));
        assert_eq!(img.clamped(2, -1), img.at(2, 0));
    }

    #[test]
    fn a_png_round_trips_byte_for_byte() {
        let img = ramp(17, 11);
        let decoded = RgbImage::from_bytes(&png_bytes(&img), "ramp.png").unwrap();
        assert_eq!(decoded.width(), 17);
        assert_eq!(decoded.height(), 11);
        assert_eq!(decoded.pixels(), img.pixels());
    }

    #[test]
    fn the_two_doors_to_luma_agree() {
        // The module's claim, measured: decoding a file as gray and decoding it
        // as colour then converting must give the same bytes, or the detector
        // finds its corners somewhere else depending on the caller's route.
        let img = ramp(23, 19);
        let bytes = png_bytes(&img);
        let direct = GrayImage::from_bytes(&bytes, "ramp.png").unwrap();
        let via_colour = RgbImage::from_bytes(&bytes, "ramp.png").unwrap().to_gray();
        assert_eq!(
            direct.pixels(),
            via_colour.pixels(),
            "the gray decoder and RgbImage::to_gray disagree about luma"
        );
    }

    #[test]
    fn a_decode_failure_names_the_file() {
        let err = RgbImage::from_bytes(b"not an image at all", "broken.jpg").unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("broken.jpg"),
            "the refusal does not name the file: {message}"
        );
    }
}
