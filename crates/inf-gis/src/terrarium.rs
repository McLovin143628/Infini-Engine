//! Terrarium RGB elevation codec, and "bytes in, elevations out" (ported from
//! GeoCanvas).
//!
//! # Provenance
//!
//! Adapted from `geocanvas-core/src/ops/terrain/encode.rs` and `global.rs`. The
//! latter's own header is the reason it was worth porting: *"This module has NO
//! network access and NO GDAL: bytes come in (a 256×256 terrarium PNG), numbers
//! come out (elevations in meters)."* That is exactly the shape this engine
//! wants — a pure function with no I/O policy baked into it, so the editor can
//! feed it a downloaded file, a test can feed it a synthetic one, and neither
//! path is a special case of the other.
//!
//! # The encoding
//!
//! Terrarium (Mapzen / AWS Terrain Tiles) packs one elevation into an RGB
//! triplet:
//!
//! ```text
//! v' = elevation + 32768
//! r  = ⌊v' / 256⌋      g = ⌊v'⌋ mod 256      b = ⌊frac(v') · 256⌋
//! ```
//!
//! — 1/256 m precision over ±32 768 m, which comfortably spans the Mariana
//! Trench to Everest with 4 mm of quantisation. Sea level encodes as
//! `(128, 0, 0)`.
//!
//! # Why this earns its place
//!
//! It makes **real Earth elevation importable with zero new dependencies**. The
//! tiles are ordinary PNGs and this engine already decodes PNG a row at a time
//! (`inf_terrain::import::decode_rows`), so the entire path from "a global DEM
//! source" to "an `.inf_terrain`" is code we already had plus the ten lines of
//! [`decode_elevation`]. It is the fastest route from nothing to a real-world
//! island, and it needs no GeoTIFF, no CRS table and no projection library —
//! Web-Mercator tile bounds are closed-form ([`crate::tilemath`]).
//!
//! # The one thing to be careful about
//!
//! `(128, 0, 0)` means *sea level*, and it is also what a **missing tile**
//! decodes to. Mapping libraries rely on that (a hole renders flat rather than
//! as a cliff). An importer must not: "the ocean here" and "we never fetched
//! this tile" are different facts, and silently treating the second as the first
//! builds a flat plain where a mountain was. [`decode_tile_rgb`] therefore
//! reports what it decoded and leaves the policy to the caller, and the nodata
//! policy at the terrain door is where that decision is actually recorded.

use crate::GisError;

/// Elevation (metres) → a terrarium RGB triplet.
///
/// Non-finite input encodes as 0 m and the representable range clamps at the u8
/// edges — a lossy but *bounded* answer, which is the right one for an encoder
/// (the refusing door is on the import side, where the author can act on it).
pub fn encode_elevation(elevation: f64) -> [u8; 3] {
    let v = if elevation.is_finite() {
        elevation
    } else {
        0.0
    };
    // Clamp just inside the encodable range so the floor below stays in u8.
    let shifted = (v + 32768.0).clamp(0.0, 65535.999);
    let whole = shifted.floor();
    let r = (whole / 256.0).floor() as u8;
    let g = (whole as u32 % 256) as u8;
    let b = ((shifted - whole) * 256.0).floor().clamp(0.0, 255.0) as u8;
    [r, g, b]
}

/// A terrarium RGB triplet → elevation in metres. The exact inverse of
/// [`encode_elevation`] for every value the encoder produced.
#[inline]
pub fn decode_elevation(rgb: [u8; 3]) -> f64 {
    f64::from(rgb[0]) * 256.0 + f64::from(rgb[1]) + f64::from(rgb[2]) / 256.0 - 32768.0
}

/// The elevation a *missing* terrarium tile decodes to (0 m — see the module
/// docs on why that is a hazard rather than a convenience).
pub const MISSING_TILE_ELEVATION_M: f64 = 0.0;

/// What one decoded tile turned out to contain.
#[derive(Clone, Debug, PartialEq)]
pub struct TerrariumTile {
    /// Row-major elevations in metres, `width · height` of them. Row 0 is the
    /// tile's **north** edge (the XYZ convention).
    pub elevations: Vec<f64>,
    pub width: u32,
    pub height: u32,
    /// How many samples decoded to exactly sea level.
    ///
    /// Reported rather than acted on: a tile that is 100% sea level is either
    /// genuine ocean or a tile that was never fetched, and this crate refuses to
    /// guess which. See [`TerrariumTile::is_uniformly_sea_level`].
    pub sea_level_samples: usize,
}

impl TerrariumTile {
    /// The elevation at `(x, y)`, or `None` outside the tile.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> Option<f64> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.elevations.get((y * self.width + x) as usize).copied()
    }

    /// `true` when every sample is exactly sea level — the signature a **missing
    /// tile** shares with genuine open ocean.
    ///
    /// The caller decides what that means. An importer covering a coastline
    /// should accept it; an importer covering a mountain range should treat it as
    /// a fetch that failed, and either way the decision belongs where the extent
    /// is known, not here.
    pub fn is_uniformly_sea_level(&self) -> bool {
        !self.elevations.is_empty() && self.sea_level_samples == self.elevations.len()
    }

    /// `(min, max)` elevation, or `None` for an empty tile.
    pub fn range(&self) -> Option<(f64, f64)> {
        let mut it = self.elevations.iter().copied();
        let first = it.next()?;
        Some(it.fold((first, first), |(lo, hi), v| (lo.min(v), hi.max(v))))
    }
}

/// Decode a raw RGB(A) buffer of terrarium samples into elevations.
///
/// `channels` is 3 or 4 (an alpha channel is ignored). This is the pure core —
/// no PNG, no I/O — so a caller that already has pixels does not pay for a
/// decoder.
pub fn decode_tile_rgb(
    rgb: &[u8],
    width: u32,
    height: u32,
    channels: usize,
) -> Result<TerrariumTile, GisError> {
    if !(3..=4).contains(&channels) {
        return Err(GisError::Format(format!(
            "a terrarium tile is RGB or RGBA; this one claims {channels} channels"
        )));
    }
    let want = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| GisError::Format(format!("terrarium tile {width}x{height} overflows")))?;
    if rgb.len() != want {
        return Err(GisError::Format(format!(
            "a {width}x{height} terrarium tile with {channels} channels needs \
             {want} bytes; this buffer has {}",
            rgb.len()
        )));
    }
    let mut elevations = Vec::with_capacity((width as usize) * (height as usize));
    let mut sea_level_samples = 0usize;
    for px in rgb.chunks_exact(channels) {
        let v = decode_elevation([px[0], px[1], px[2]]);
        if v == MISSING_TILE_ELEVATION_M {
            sea_level_samples += 1;
        }
        elevations.push(v);
    }
    Ok(TerrariumTile {
        elevations,
        width,
        height,
        sea_level_samples,
    })
}

/// Decode terrarium **PNG bytes** into elevations — the `global.rs` contract,
/// "bytes in, numbers out".
///
/// Uses the same `png` crate `inf_terrain::import` decodes heightmaps with, so
/// this adds no decoder to the tree.
pub fn decode_tile_png(bytes: &[u8]) -> Result<TerrariumTile, GisError> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| GisError::Format(format!("terrarium tile PNG header: {e}")))?;
    let info = reader.info();
    let (width, height) = (info.width, info.height);
    let channels = match info.color_type {
        png::ColorType::Rgb => 3usize,
        png::ColorType::Rgba => 4,
        other => {
            return Err(GisError::Format(format!(
                "a terrarium tile packs elevation into R, G and B, so it must be an \
                 RGB or RGBA PNG (this one is {other:?}). A grayscale PNG is an \
                 ordinary heightmap — import it through the terrain heightmap door \
                 instead."
            )))
        }
    };
    if info.bit_depth != png::BitDepth::Eight {
        return Err(GisError::Format(format!(
            "a terrarium tile is 8 bits per channel by definition (this one is \
             {:?}) — the encoding uses the three bytes as a fixed-point number",
            info.bit_depth
        )));
    }
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let frame = reader
        .next_frame(&mut buf)
        .map_err(|e| GisError::Format(format!("terrarium tile PNG data: {e}")))?;
    decode_tile_rgb(&buf[..frame.buffer_size()], width, height, channels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference triplets, re-derived from the published encoding rather than
    /// copied from the port source, so this test would catch a transcription
    /// error in the port itself.
    #[test]
    fn known_elevations_encode_to_their_reference_triplets() {
        // 0 m: v' = 32768 = 128*256 + 0, fraction 0.
        assert_eq!(encode_elevation(0.0), [128, 0, 0], "sea level");
        // 100.5 m: v' = 32868.5 = 128*256 + 100, fraction .5 -> 128.
        assert_eq!(encode_elevation(100.5), [128, 100, 128]);
        // -32.25: v' = 32735.75 = 127*256 + 223, fraction .75 -> 192.
        assert_eq!(encode_elevation(-32.25), [127, 223, 192]);
        // Everest, 8848: v' = 41616 = 162*256 + 144.
        assert_eq!(encode_elevation(8848.0), [162, 144, 0]);
        // And the decoder inverts each one exactly.
        for (v, rgb) in [
            (0.0, [128u8, 0, 0]),
            (100.5, [128, 100, 128]),
            (-32.25, [127, 223, 192]),
            (8848.0, [162, 144, 0]),
        ] {
            assert_eq!(decode_elevation(rgb), v, "{rgb:?}");
        }
    }

    #[test]
    fn the_codec_round_trips_within_its_quantisation() {
        for v in [0.0, 100.5, -32.25, 8848.0, -415.0, 0.004, 4810.9, -10_994.0] {
            let back = decode_elevation(encode_elevation(v));
            assert!(
                (back - v).abs() < 1.0 / 256.0 + 1e-9,
                "{v} m round-tripped to {back} m, outside the 1/256 m quantisation"
            );
        }
    }

    #[test]
    fn degenerate_elevations_clamp_rather_than_wrapping() {
        assert_eq!(encode_elevation(f64::NAN), [128, 0, 0]);
        assert_eq!(encode_elevation(f64::INFINITY), [128, 0, 0]);
        assert_eq!(encode_elevation(f64::NEG_INFINITY), [128, 0, 0]);
        // Past the representable range it saturates; it does not wrap to the
        // opposite extreme, which would turn a mountain into a trench.
        assert_eq!(encode_elevation(-1e9), [0, 0, 0]);
        assert_eq!(encode_elevation(1e9), [255, 255, 255]);
    }

    /// A tile decodes with a real gradient, and the sea-level/missing-tile
    /// ambiguity is REPORTED rather than resolved.
    #[test]
    fn a_tile_decodes_and_reports_the_missing_tile_ambiguity() {
        // A 4x4 ramp: distinct elevations, so nothing here is uniform.
        let mut rgb = Vec::new();
        for i in 0..16 {
            rgb.extend_from_slice(&encode_elevation(i as f64 * 12.5 - 20.0));
        }
        let t = decode_tile_rgb(&rgb, 4, 4, 3).unwrap();
        assert_eq!(t.width, 4);
        assert_eq!(t.elevations.len(), 16);
        assert_eq!(t.get(0, 0), Some(-20.0));
        assert_eq!(t.get(3, 3), Some(15.0 * 12.5 - 20.0));
        assert_eq!(t.get(4, 0), None, "outside the tile");
        let (lo, hi) = t.range().unwrap();
        assert_eq!((lo, hi), (-20.0, 167.5));
        assert!(!t.is_uniformly_sea_level());

        // An all-sea-level tile is FLAGGED, not corrected. This is the "missing
        // tile and open ocean decode identically" hazard.
        let flat: Vec<u8> = std::iter::repeat_n([128u8, 0, 0], 16).flatten().collect();
        let t = decode_tile_rgb(&flat, 4, 4, 3).unwrap();
        assert!(t.is_uniformly_sea_level());
        assert_eq!(t.sea_level_samples, 16);
        assert_eq!(t.range(), Some((0.0, 0.0)));

        // RGBA works and ignores alpha.
        let mut rgba = Vec::new();
        for i in 0..16 {
            rgba.extend_from_slice(&encode_elevation(i as f64));
            rgba.push(0xAB);
        }
        let t = decode_tile_rgb(&rgba, 4, 4, 4).unwrap();
        assert_eq!(t.get(1, 0), Some(1.0));
    }

    #[test]
    fn a_malformed_buffer_is_refused_with_its_own_numbers() {
        let e = decode_tile_rgb(&[0; 10], 4, 4, 3).unwrap_err().to_string();
        assert!(e.contains("48") && e.contains("10"), "{e}");
        assert!(
            decode_tile_rgb(&[0; 16], 4, 4, 1).is_err(),
            "1 channel is not terrarium"
        );
        assert!(decode_tile_rgb(&[0; 16], 4, 4, 5).is_err());
    }

    /// The PNG door, end to end, plus its two refusals — a terrarium tile that
    /// is really a grayscale heightmap is a mistake worth naming.
    #[test]
    fn the_png_door_decodes_and_refuses_the_wrong_kind_of_png() {
        // Build a real RGB PNG whose pixels are terrarium-encoded.
        let mut rgb = Vec::new();
        for i in 0..64 {
            rgb.extend_from_slice(&encode_elevation(i as f64 * 3.25 - 50.0));
        }
        let mut png_bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(std::io::Cursor::new(&mut png_bytes), 8, 8);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&rgb).unwrap();
        }
        let t = decode_tile_png(&png_bytes).expect("the RGB terrarium tile decodes");
        assert_eq!((t.width, t.height), (8, 8));
        assert_eq!(t.get(0, 0), Some(-50.0));
        assert_eq!(t.get(7, 7), Some(63.0 * 3.25 - 50.0));

        // A grayscale PNG is an ordinary heightmap; refuse it and say where it
        // should go instead.
        let mut gray = Vec::new();
        {
            let mut enc = png::Encoder::new(std::io::Cursor::new(&mut gray), 4, 4);
            enc.set_color(png::ColorType::Grayscale);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&[0u8; 16]).unwrap();
        }
        let e = decode_tile_png(&gray).unwrap_err().to_string();
        assert!(e.contains("Grayscale"), "{e}");
        assert!(
            e.contains("heightmap door"),
            "the refusal must point at the right door: {e}"
        );

        // Not a PNG at all.
        assert!(decode_tile_png(b"not a png").is_err());
    }
}
