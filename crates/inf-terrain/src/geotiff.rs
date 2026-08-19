//! GeoTIFF georeferencing: the tags that say where a DEM is and what its
//! numbers mean (Wave G).
//!
//! # What a GeoTIFF is
//!
//! An ordinary TIFF plus a handful of private tags that answer three questions:
//! *where on Earth is pixel (0, 0)?*, *how big is a pixel?*, and *in what
//! coordinate system?* The owner's document's recipe (`raster_size`,
//! `geo_transform`, `raster_band(1)`, `no_data_value`, a windowed read, pixel →
//! world by `t[0] + x·t[1]`) is exactly right about **what to read**. Only the
//! library was wrong: it prescribes `gdal`, a binding over a C++ library that
//! drags forty shared objects onto a three-OS CI, and this reads the same tags
//! with the pure-Rust `tiff` crate instead.
//!
//! # The tags, and what each maps onto
//!
//! | tag | name | engine field |
//! |---|---|---|
//! | 33550 | `ModelPixelScale` `[sx, sy, sz]` | `meters_per_sample` (with the `sx == sy` check) |
//! | 33922 | `ModelTiepoint` `(i,j,k, x,y,z)` | the model position of a raster pixel → `TerrainAssetHeader::origin` |
//! | 34264 | `ModelTransformation` (full 4×4) | **refused** — a rotated raster needs resampling we do not build |
//! | 34735/6/7 | `GeoKeyDirectory` + params | the source CRS, and the linear/vertical units |
//! | 42113 | `GDAL_NODATA` (ASCII!) | the sentinel the nodata policy acts on |
//!
//! # Three things that are refused rather than guessed
//!
//! **Rotated and sheared rasters.** `ModelTransformation` can express an
//! arbitrary affine map, and honouring one means resampling the raster onto a
//! north-up lattice — a real feature, not a small one. A rotated DEM is refused
//! by name, telling the author to reproject north-up in their own tool, because
//! the alternative is importing it as though it were north-up and building an
//! island rotated a few degrees off from every vector layer that lands on it.
//!
//! **Anisotropic pixels.** `sx != sy` means the raster's ground samples are not
//! square, and this engine's terrain lattice is. Silently averaging the two
//! would stretch the world along one axis by their ratio. Refused with both
//! numbers named.
//!
//! **A unit we do not recognise.** See [`VerticalUnits`]: guessing "probably
//! metres" on a DEM in feet builds a landscape 3.28× too flat.
//!
//! # The vertical unit conversion happens exactly ONCE
//!
//! Feet become metres in [`GeoTiffMeta::vertical_scale`], applied inside the
//! TIFF row decoder and nowhere else. That is the units doctrine's whole point:
//! a scale factor that exists in two places is a scale factor that will be
//! applied twice or zero times, and the symptom (a landscape 10.8× too tall, or
//! 3.28× too flat) looks like an authoring mistake rather than a bug.

use tiff::decoder::Decoder;
use tiff::tags::Tag;

use crate::import::TerrainError;

/// `ModelPixelScale` — three doubles, the ground size of one pixel.
const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
/// `ModelTiepoint` — six doubles per tie, `(i, j, k, x, y, z)`.
const TAG_MODEL_TIEPOINT: u16 = 33922;
/// `ModelTransformation` — sixteen doubles, a full 4×4 affine.
const TAG_MODEL_TRANSFORMATION: u16 = 34264;
/// `GeoKeyDirectory` — the geo-key block.
const TAG_GEO_KEY_DIRECTORY: u16 = 34735;
/// `GeoDoubleParams` — doubles the key directory indexes into.
const TAG_GEO_DOUBLE_PARAMS: u16 = 34736;
/// `GeoAsciiParams` — ASCII the key directory indexes into.
const TAG_GEO_ASCII_PARAMS: u16 = 34737;
/// `GDAL_NODATA` — **an ASCII tag**, not a numeric one. The sentinel is written
/// as text (`"-9999"`, `"-3.4028234663852886e+38"`) and must be parsed.
const TAG_GDAL_NODATA: u16 = 42113;

/// Geo-key: 1 = projected, 2 = geographic, 3 = geocentric.
const KEY_MODEL_TYPE: u16 = 1024;
/// Geo-key: the EPSG code of a geographic CRS.
const KEY_GEOGRAPHIC_TYPE: u16 = 2048;
/// Geo-key: the EPSG code of a projected CRS.
const KEY_PROJECTED_CS_TYPE: u16 = 3072;
/// Geo-key: the horizontal linear unit.
const KEY_PROJ_LINEAR_UNITS: u16 = 3076;
/// Geo-key: the vertical unit.
const KEY_VERTICAL_UNITS: u16 = 4099;

/// EPSG unit codes this engine understands.
const UNIT_METRE: u16 = 9001;
const UNIT_FOOT: u16 = 9002;
const UNIT_US_SURVEY_FOOT: u16 = 9003;

/// A length unit a DEM's elevations may be in.
///
/// # Why this cannot be defaulted quietly
///
/// A great many published DEMs — most older US state and county products — are
/// in **feet**, and a great many are in metres, and the file says which. Reading
/// a foot DEM as metres builds a landscape 3.28 times too flat: a 1 000 ft
/// mountain becomes a 305 m hill. Nothing about the result looks broken; it just
/// looks like the author scaled it wrong. So the unit is read, converted once,
/// and an *unrecognised* code is refused rather than assumed.
///
/// The two feet are genuinely different units and the difference is not
/// negligible at survey scale: the international foot is exactly 0.3048 m, the
/// US survey foot is 1200/3937 m ≈ 0.304 800 6. Over a 50 km island that is
/// 30 cm — irrelevant to a game, but free to get right, and getting it wrong
/// would be a silent inaccuracy in a system whose whole point is fidelity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VerticalUnits {
    #[default]
    Metre,
    Foot,
    UsSurveyFoot,
}

impl VerticalUnits {
    /// Metres per unit.
    pub fn to_meters(self) -> f64 {
        match self {
            VerticalUnits::Metre => 1.0,
            VerticalUnits::Foot => 0.3048,
            // 1200/3937 exactly, written as the division so the constant cannot
            // be mis-transcribed.
            VerticalUnits::UsSurveyFoot => 1200.0 / 3937.0,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            VerticalUnits::Metre => "metres",
            VerticalUnits::Foot => "international feet",
            VerticalUnits::UsSurveyFoot => "US survey feet",
        }
    }

    /// Map an EPSG unit code, or `None` for one we do not handle.
    pub fn from_epsg(code: u16) -> Option<Self> {
        match code {
            UNIT_METRE => Some(VerticalUnits::Metre),
            UNIT_FOOT => Some(VerticalUnits::Foot),
            UNIT_US_SURVEY_FOOT => Some(VerticalUnits::UsSurveyFoot),
            _ => None,
        }
    }
}

/// What a GeoTIFF says about where it is and what its numbers mean.
///
/// Everything here is **derived from the file's header**, never persisted — the
/// same rule `HeightmapProbe` follows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeoTiffMeta {
    /// Ground size of one pixel, metres, when the file declares one and it is
    /// square.
    pub meters_per_sample: Option<f64>,
    /// The model coordinate (easting, northing, height) of the raster's
    /// **top-left corner sample**, after the tiepoint's raster offset is undone.
    pub origin: Option<(f64, f64, f64)>,
    /// The EPSG code the file names, when it names one we can read.
    pub epsg: Option<u32>,
    /// `true` when the file's CRS is geographic (degrees), which makes
    /// `meters_per_sample` meaningless as written — see
    /// [`GeoTiffMeta::degrees_per_sample`].
    pub crs_is_geographic: bool,
    /// The nodata sentinel from `GDAL_NODATA`.
    pub nodata: Option<f64>,
    /// The vertical unit of the elevation samples.
    pub vertical_units: VerticalUnits,
    /// The horizontal linear unit, for the record.
    pub linear_units: VerticalUnits,
    /// A human description of the CRS from `GeoAsciiParams`, for the wizard.
    pub crs_name: Option<String>,
}

impl GeoTiffMeta {
    /// Metres per elevation unit — **the one place feet become metres**.
    pub fn vertical_scale(&self) -> f64 {
        self.vertical_units.to_meters()
    }

    /// `true` when the file carried enough to place itself on Earth.
    pub fn is_georeferenced(&self) -> bool {
        self.origin.is_some() && self.meters_per_sample.is_some()
    }

    /// Where this DEM's top-left sample sits in **world metres**, given the
    /// level's geo-anchor — the value that finally fills the `.inf_terrain`
    /// header's `origin` field.
    ///
    /// Returns `None` when the file is not georeferenced, when the anchor is
    /// disabled, or when the source's CRS is not the anchor's. That last one
    /// matters: this is a **subtraction, not a reprojection**. Reprojecting a
    /// raster means resampling it onto a new lattice, which changes the
    /// elevations rather than placing them, and this engine does not do that —
    /// so a DEM in a different CRS from the level is refused here and the author
    /// is told to reproject it in their own tool. Silently subtracting one CRS's
    /// origin from another's coordinates would land the terrain hundreds of
    /// kilometres away with nothing to show for it.
    ///
    /// The elevation component is deliberately **zero**: a heightfield tile
    /// carries its own sample heights, and the anchor's height datum is already
    /// applied to those by the import's height mapping. Putting it here as well
    /// would apply it twice.
    pub fn world_origin(&self, anchor: &inf_math::geo::GeoAnchor) -> Option<glam::DVec3> {
        if !anchor.enabled || !self.is_georeferenced() {
            return None;
        }
        if !self.crs_matches(&anchor.crs) {
            return None;
        }
        let (e, n, _) = self.origin?;
        let w = anchor.world_from_projected(e, n, anchor.origin_height_m);
        w.is_finite().then_some(glam::DVec3::new(w.x, 0.0, w.z))
    }

    /// Whether this file's CRS is the one `spec` names.
    ///
    /// Compares EPSG codes when both sides have one. A file with no CRS key is
    /// taken at the author's word — plenty of real DEMs omit the key while being
    /// perfectly well placed, and refusing those would be worse than trusting
    /// the author who picked the anchor.
    pub fn crs_matches(&self, spec: &str) -> bool {
        let Some(mine) = self.epsg else {
            return true;
        };
        // Parse "EPSG:32610" / "32610"; a proj4 string carries no code to
        // compare, so it is likewise taken at the author's word.
        let theirs = {
            let t = spec.trim();
            if t.contains('+') {
                return true;
            }
            t.strip_prefix("EPSG:")
                .or_else(|| t.strip_prefix("epsg:"))
                .unwrap_or(t)
                .trim()
                .parse::<u32>()
                .ok()
        };
        theirs.is_none_or(|c| c == mine)
    }

    /// The pixel size in **degrees**, when the source is in a geographic CRS.
    ///
    /// A DEM in EPSG:4326 has a pixel scale like `0.000 277 8` — one arc-second,
    /// not 0.28 mm. Treating that as metres builds an island a few centimetres
    /// across, which is at least an obvious failure; naming it is better.
    pub fn degrees_per_sample(&self) -> Option<f64> {
        self.crs_is_geographic.then_some(self.meters_per_sample?)
    }
}

/// One geo-key entry from the key directory.
struct GeoKey {
    id: u16,
    location: u16,
    count: u16,
    value: u16,
}

/// Parse the `GeoKeyDirectory` block.
///
/// Layout: four `SHORT`s of header (directory version, key revision, minor
/// revision, key count), then `count` entries of four `SHORT`s each —
/// `(key id, tag location, value count, value or offset)`. A `location` of 0
/// means the fourth short **is** the value; otherwise it indexes into tag 34736
/// (doubles) or 34737 (ASCII).
fn parse_geo_keys(raw: &[u16]) -> Vec<GeoKey> {
    if raw.len() < 4 {
        return Vec::new();
    }
    let count = raw[3] as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let base = 4 + i * 4;
        if base + 3 >= raw.len() {
            break;
        }
        out.push(GeoKey {
            id: raw[base],
            location: raw[base + 1],
            count: raw[base + 2],
            value: raw[base + 3],
        });
    }
    out
}

/// Read the georeferencing tags from an open decoder.
///
/// A TIFF with none of them is not an error — it is an ordinary heightmap, and
/// importing one as such is legitimate. The returned [`GeoTiffMeta`] simply
/// reports [`is_georeferenced`](GeoTiffMeta::is_georeferenced) as `false`.
pub fn read_meta<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> Result<GeoTiffMeta, TerrainError> {
    let mut meta = GeoTiffMeta::default();

    // A full 4x4 model transformation is refused before anything else is read:
    // whatever else the file says, we cannot honour a rotated raster, and
    // finding that out after reporting a plausible pixel size would be worse.
    if let Ok(Some(v)) = decoder.find_tag(Tag::Unknown(TAG_MODEL_TRANSFORMATION)) {
        if let Ok(m) = v.into_f64_vec() {
            if m.len() >= 16 {
                // Rotation/shear live in the off-diagonal terms. A north-up
                // raster has both at exactly zero.
                let (rot_a, rot_b) = (m[1], m[4]);
                if rot_a != 0.0 || rot_b != 0.0 {
                    return Err(TerrainError::Unsupported(format!(
                        "this GeoTIFF carries a rotated or sheared ModelTransformation \
                         (off-diagonal terms {rot_a} and {rot_b}); this importer needs \
                         a north-up raster. Reproject it north-up first — \
                         `gdalwarp -t_srs <its own CRS> in.tif out.tif` does it — \
                         because honouring the rotation means resampling the \
                         elevations onto a new lattice, which would change the data \
                         rather than just place it."
                    )));
                }
                // A non-rotated 4x4 carries the same information as the
                // scale + tiepoint pair, so read it as those.
                meta.meters_per_sample = check_square_pixels(m[0], -m[5])?;
                meta.origin = Some((m[3], m[7], 0.0));
            }
        }
    }

    // ModelPixelScale: [sx, sy, sz]. sy is positive and means "downward", since
    // a north-up raster's rows march SOUTH.
    if let Ok(Some(v)) = decoder.find_tag(Tag::Unknown(TAG_MODEL_PIXEL_SCALE)) {
        if let Ok(s) = v.into_f64_vec() {
            if s.len() >= 2 {
                meta.meters_per_sample = check_square_pixels(s[0], s[1])?;
            }
        }
    }

    // ModelTiepoint: (i, j, k, x, y, z) — raster point (i,j,k) is at model
    // (x,y,z). Almost always (0,0,0)->(easting, northing, 0), but the raster
    // offset is honoured rather than assumed, since a tie at the raster's centre
    // is legal and would otherwise place the DEM half its own width away.
    if let Ok(Some(v)) = decoder.find_tag(Tag::Unknown(TAG_MODEL_TIEPOINT)) {
        if let Ok(t) = v.into_f64_vec() {
            if t.len() >= 6 {
                let (i, j, _k, x, y, z) = (t[0], t[1], t[2], t[3], t[4], t[5]);
                let scale = meta.meters_per_sample.unwrap_or(1.0);
                // Walk back from the tie point to the raster's top-left sample:
                // -i pixels east, +j pixels north (rows march south).
                meta.origin = Some((x - i * scale, y + j * scale, z));
            }
        }
    }

    // Geo keys: CRS identity and units.
    if let Ok(Some(v)) = decoder.find_tag(Tag::Unknown(TAG_GEO_KEY_DIRECTORY)) {
        if let Ok(raw) = v.into_u16_vec() {
            let doubles = decoder
                .find_tag(Tag::Unknown(TAG_GEO_DOUBLE_PARAMS))
                .ok()
                .flatten()
                .and_then(|v| v.into_f64_vec().ok())
                .unwrap_or_default();
            let ascii = decoder
                .find_tag(Tag::Unknown(TAG_GEO_ASCII_PARAMS))
                .ok()
                .flatten()
                .and_then(|v| v.into_string().ok())
                .unwrap_or_default();
            let _ = &doubles;

            for key in parse_geo_keys(&raw) {
                match key.id {
                    KEY_MODEL_TYPE if key.location == 0 => {
                        // 2 == geographic (degrees).
                        meta.crs_is_geographic = key.value == 2;
                    }
                    KEY_PROJECTED_CS_TYPE if key.location == 0 && key.value != 0 => {
                        meta.epsg = Some(u32::from(key.value));
                    }
                    KEY_GEOGRAPHIC_TYPE if key.location == 0 && key.value != 0 => {
                        // Only used when no projected code was found — a
                        // projected file names both, and the projected one wins.
                        meta.epsg.get_or_insert(u32::from(key.value));
                    }
                    KEY_PROJ_LINEAR_UNITS if key.location == 0 => {
                        meta.linear_units =
                            VerticalUnits::from_epsg(key.value).ok_or_else(|| {
                                TerrainError::Unsupported(format!(
                                    "this GeoTIFF declares horizontal linear unit EPSG:{} \
                                 (geo-key 3076), which this importer does not know. It \
                                 handles metres (9001), international feet (9002) and \
                                 US survey feet (9003). Guessing would scale the whole \
                                 island by an unknown factor, so it is refused instead.",
                                    key.value
                                ))
                            })?;
                    }
                    KEY_VERTICAL_UNITS if key.location == 0 => {
                        meta.vertical_units =
                            VerticalUnits::from_epsg(key.value).ok_or_else(|| {
                                TerrainError::Unsupported(format!(
                                    "this GeoTIFF declares vertical unit EPSG:{} \
                                     (geo-key 4099), which this importer does not know. \
                                     It handles metres (9001), international feet (9002) \
                                     and US survey feet (9003). Reading elevations in an \
                                     unknown unit as metres would make the terrain \
                                     silently the wrong height, so it is refused.",
                                    key.value
                                ))
                            })?;
                    }
                    KEY_PROJECTED_CS_TYPE | KEY_GEOGRAPHIC_TYPE
                        if key.location == TAG_GEO_ASCII_PARAMS =>
                    {
                        let start = key.value as usize;
                        let end = (start + key.count as usize).min(ascii.len());
                        if start < end {
                            meta.crs_name =
                                Some(ascii[start..end].trim_end_matches('|').trim().to_string());
                        }
                    }
                    _ => {}
                }
            }
            // When the vertical unit is absent (the common case), the horizontal
            // one is the best available statement about the file's units. A DEM
            // in a foot-based state plane stores foot elevations essentially
            // always; assuming metres there is the exact 3.28x error this whole
            // module exists to prevent.
            if !has_key(&raw, KEY_VERTICAL_UNITS) && meta.linear_units != VerticalUnits::Metre {
                meta.vertical_units = meta.linear_units;
            }
        }
    }

    // GDAL_NODATA is ASCII — the sentinel is written as TEXT.
    if let Ok(Some(v)) = decoder.find_tag(Tag::Unknown(TAG_GDAL_NODATA)) {
        if let Ok(s) = v.into_string() {
            let t = s.trim().trim_end_matches('\0').trim();
            // `nan` is a legal spelling and means the same thing the float
            // sentinel does; parsing it to a NaN would make every comparison
            // against it false, so it is normalised away here and the finiteness
            // door handles those samples instead.
            if !t.is_empty() && !t.eq_ignore_ascii_case("nan") {
                meta.nodata = t.parse::<f64>().ok().filter(|v| v.is_finite());
            }
        }
    }

    Ok(meta)
}

fn has_key(raw: &[u16], id: u16) -> bool {
    parse_geo_keys(raw).iter().any(|k| k.id == id)
}

/// Refuse a raster whose pixels are not square. See the module docs.
fn check_square_pixels(sx: f64, sy: f64) -> Result<Option<f64>, TerrainError> {
    let (sx, sy) = (sx.abs(), sy.abs());
    if !(sx.is_finite() && sy.is_finite()) || sx <= 0.0 || sy <= 0.0 {
        return Ok(None);
    }
    // A relative tolerance, not an absolute one: real files carry values like
    // 0.9999999999 vs 1.0 from a round trip through a decimal representation,
    // and refusing those would reject most of the world's DEMs.
    let rel = (sx - sy).abs() / sx.max(sy);
    if rel > 1e-6 {
        return Err(TerrainError::Unsupported(format!(
            "this GeoTIFF's pixels are not square: {sx} by {sy} ground units. This \
             engine's terrain is a square lattice, so importing it would stretch \
             the world along one axis by a factor of {:.4}. Resample it to square \
             pixels first — `gdalwarp -tr {s} {s} in.tif out.tif` — rather than \
             letting the importer average the two and quietly distort the result.",
            sx.max(sy) / sx.min(sy),
            s = sx.min(sy)
        )));
    }
    Ok(Some((sx + sy) * 0.5))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_units_convert_once_and_exactly() {
        assert_eq!(VerticalUnits::Metre.to_meters(), 1.0);
        assert_eq!(VerticalUnits::Foot.to_meters(), 0.3048);
        // The US survey foot is 1200/3937, which is NOT 0.3048.
        let us = VerticalUnits::UsSurveyFoot.to_meters();
        assert!(
            us > 0.3048,
            "the US survey foot is longer than the international one"
        );
        assert!((us - 0.304_800_609_601_219_2).abs() < 1e-15, "got {us}");
        // The difference is 2 ppm — 10 cm over a 50 km island. Small, and free.
        let over_50km = 50_000.0 * (us / 0.3048 - 1.0);
        assert!(
            (0.05..0.2).contains(&over_50km),
            "the two feet differ by {over_50km} m over 50 km"
        );

        assert_eq!(VerticalUnits::from_epsg(9001), Some(VerticalUnits::Metre));
        assert_eq!(VerticalUnits::from_epsg(9002), Some(VerticalUnits::Foot));
        assert_eq!(
            VerticalUnits::from_epsg(9003),
            Some(VerticalUnits::UsSurveyFoot)
        );
        // An unknown unit is NOT defaulted to metres — that is the whole point.
        assert_eq!(
            VerticalUnits::from_epsg(9036),
            None,
            "kilometres must not pass as metres"
        );
        assert_eq!(VerticalUnits::from_epsg(0), None);
    }

    /// Non-square pixels are refused with both numbers and a remedy, and the
    /// tolerance is relative so real round-tripped values still pass.
    #[test]
    fn anisotropic_pixels_are_refused_but_float_noise_is_not() {
        let e = check_square_pixels(10.0, 20.0).unwrap_err().to_string();
        assert!(e.contains("10") && e.contains("20"), "{e}");
        assert!(
            e.contains("gdalwarp"),
            "the refusal must name a remedy: {e}"
        );
        assert!(e.contains('2'), "and quantify the distortion: {e}");

        // A value that round-tripped through a decimal text form is fine.
        assert_eq!(
            check_square_pixels(1.000_000_000_1, 1.0).unwrap(),
            Some(1.000_000_000_05)
        );
        assert!(check_square_pixels(30.0, 30.0).unwrap().is_some());
        // A tenth of a percent is a real distortion and is refused.
        assert!(check_square_pixels(30.0, 30.03).is_err());
        // Degenerate scales report "no scale" rather than failing — the file
        // simply is not georeferenced.
        assert_eq!(check_square_pixels(0.0, 0.0).unwrap(), None);
        assert_eq!(check_square_pixels(f64::NAN, 1.0).unwrap(), None);
    }

    /// The geo-key directory's own layout: a 4-short header then 4-short
    /// entries.
    #[test]
    fn geo_keys_parse_from_their_wire_layout() {
        // version 1, revision 1.0, 2 keys.
        let raw = vec![
            1,
            1,
            0,
            2, //
            KEY_MODEL_TYPE,
            0,
            1,
            1, // projected
            KEY_PROJECTED_CS_TYPE,
            0,
            1,
            32610, // UTM 10N
        ];
        let keys = parse_geo_keys(&raw);
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].id, KEY_MODEL_TYPE);
        assert_eq!(keys[0].value, 1);
        assert_eq!(keys[1].value, 32610);
        assert!(has_key(&raw, KEY_PROJECTED_CS_TYPE));
        assert!(!has_key(&raw, KEY_VERTICAL_UNITS));

        // A truncated directory yields what it can rather than panicking — real
        // files are written by many tools and some of them are wrong.
        let truncated = vec![1, 1, 0, 5, KEY_MODEL_TYPE, 0, 1, 1];
        assert_eq!(parse_geo_keys(&truncated).len(), 1);
        assert!(parse_geo_keys(&[1, 1]).is_empty());
        assert!(parse_geo_keys(&[]).is_empty());
    }

    /// **The header's `origin` finally means something.** A georeferenced DEM
    /// lands at its real offset from the level's anchor rather than at the world
    /// origin — which is what makes a terrain and a road network from the same
    /// survey line up.
    ///
    /// Un-fix mutation: return `DVec3::ZERO` here and the offset assertions fail
    /// while every other terrain test stays green, which is exactly why this one
    /// exists.
    #[test]
    fn a_georeferenced_dem_lands_at_its_offset_from_the_anchor() {
        use inf_math::geo::GeoAnchor;
        let anchor = GeoAnchor {
            enabled: true,
            crs: "EPSG:32610".into(),
            origin_easting_m: 491_000.0,
            origin_northing_m: 5_459_000.0,
            origin_height_m: 0.0,
            origin_latitude_deg: 49.28,
            origin_longitude_deg: -123.12,
            grid_convergence_deg: 0.0,
            vertical_datum: "EGM2008".into(),
        };
        // A DEM whose top-left sample is 2 km east and 3 km NORTH of the anchor.
        let meta = GeoTiffMeta {
            meters_per_sample: Some(10.0),
            origin: Some((493_000.0, 5_462_000.0, 0.0)),
            epsg: Some(32610),
            ..Default::default()
        };
        let w = meta.world_origin(&anchor).expect("a placed origin");
        assert_eq!(w.x, 2000.0, "2 km east is +2000 X");
        // North is -Z (the solar frame), so 3 km north is -3000 Z.
        assert_eq!(w.z, -3000.0, "3 km north must be -3000 Z, got {}", w.z);
        assert_eq!(
            w.y, 0.0,
            "the height datum belongs to the samples, not the header"
        );

        // A DEM sitting exactly on the anchor lands on the world origin — the
        // case every pre-Wave-G import silently assumed.
        let on_anchor = GeoTiffMeta {
            origin: Some((491_000.0, 5_459_000.0, 0.0)),
            ..meta.clone()
        };
        assert_eq!(on_anchor.world_origin(&anchor), Some(glam::DVec3::ZERO));

        // Refusals: no anchor, no georeference, wrong CRS.
        assert_eq!(meta.world_origin(&GeoAnchor::default()), None);
        assert_eq!(
            GeoTiffMeta {
                origin: None,
                ..meta.clone()
            }
            .world_origin(&anchor),
            None,
            "an ungeoreferenced file has no offset to give"
        );
        let elsewhere = GeoTiffMeta {
            epsg: Some(32611),
            ..meta.clone()
        };
        assert_eq!(
            elsewhere.world_origin(&anchor),
            None,
            "a DEM in a different UTM zone must NOT be placed by subtraction — \
             that would land it hundreds of km away"
        );

        // A file that names no CRS is taken at the author's word, because plenty
        // of real, correctly-placed DEMs omit the key.
        let unnamed = GeoTiffMeta {
            epsg: None,
            ..meta.clone()
        };
        assert_eq!(unnamed.world_origin(&anchor).map(|w| w.x), Some(2000.0));
        assert!(meta.crs_matches("EPSG:32610"));
        assert!(meta.crs_matches("32610"));
        assert!(!meta.crs_matches("EPSG:32611"));
        assert!(
            meta.crs_matches("+proj=utm +zone=10 +datum=WGS84 +units=m"),
            "a proj4 string carries no code to compare against"
        );
    }

    #[test]
    fn meta_reports_what_it_can_and_admits_what_it_cannot() {
        let bare = GeoTiffMeta::default();
        assert!(
            !bare.is_georeferenced(),
            "a plain TIFF is not georeferenced"
        );
        assert_eq!(bare.vertical_scale(), 1.0);
        assert_eq!(bare.degrees_per_sample(), None);

        let geo = GeoTiffMeta {
            meters_per_sample: Some(1.0),
            origin: Some((491_000.0, 5_459_000.0, 0.0)),
            epsg: Some(32610),
            vertical_units: VerticalUnits::Foot,
            ..Default::default()
        };
        assert!(geo.is_georeferenced());
        assert_eq!(geo.vertical_scale(), 0.3048);

        // A geographic source reports its pixel size as DEGREES, so nobody reads
        // 0.000278 as a quarter of a millimetre.
        let wgs = GeoTiffMeta {
            meters_per_sample: Some(0.000_277_8),
            crs_is_geographic: true,
            ..Default::default()
        };
        assert_eq!(wgs.degrees_per_sample(), Some(0.000_277_8));
    }
}
