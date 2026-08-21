//! The elevation source: which tiles the world needs, where they live, and how
//! to ask one for a height.
//!
//! # Destination-driven, and why that is the whole design
//!
//! The world is a square in a projected metric CRS; the source is a pyramid of
//! Web-Mercator tiles. Those two lattices do not line up and never will. There
//! are two ways to get from one to the other:
//!
//! * **forward** — walk the source's pixels, project each into the world, write
//!   it there. This is what a naive port does and it is wrong twice: the
//!   projected pixels land between destination samples (so the result is a
//!   scatter with holes to fill), and a Mercator pixel is 1.53× wider than a
//!   ground metre at this latitude (so the density of writes is a function of
//!   latitude).
//! * **inverse** — walk the *destination's* samples, ask each one where it came
//!   from, read there. Every output sample is written exactly once, the
//!   filtering happens in the source's own space where the pixels are square,
//!   and there is nothing to fill in afterwards.
//!
//! This module is the second one. `inf_gis::crs::Transform::to_source` is the
//! door it asks (added by this wave for exactly this reason, so that the
//! inverse of a projection is the projection library's own inverse and not a
//! locally-fitted approximation).
//!
//! # The two hazards, both named
//!
//! **A missing tile and open ocean decode identically.** Terrarium's `(128, 0,
//! 0)` is sea level, and it is also what a hole renders as. [`TileMosaic`]
//! counts uniformly-sea-level tiles and reports them; the carve step is what
//! turns "no data" into ocean, deliberately and once.
//!
//! **A 404 body is 299 bytes of XML and a cache will keep it.** The dataset this
//! recipe names serves terrarium to z15 and answers `NoSuchKey` above it. A
//! cached error page decodes as "not a PNG", which is why
//! [`TileMosaic::load`] refuses it by name with the remedy rather than treating
//! an undecodable tile as absent.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use inf_gis::terrarium::TerrariumTile;
use inf_gis::tilemath;

use crate::recipe::IslandRecipe;
use crate::IslandError;

/// The elevation range a terrarium sample may plausibly carry, in metres.
///
/// # Why a range and not just a finiteness check
///
/// The codec's own floor is `(0, 0, 0)` → **−32 768 m**, and the shipped island's
/// source really does contain it: the first full build reported a source range of
/// `(-32768.0, 1239.0)`. That is not a trench, it is a black pixel — a tile the
/// provider filled rather than surveyed — and it is *finite*, so every guard this
/// repository already has waves it through. Inside a coastline it would carve a
/// thirty-two-kilometre pit with no symptom other than a hole in the world.
///
/// The Mariana Trench is −10 935 m and Everest is 8 849 m; ±12 000 covers Earth
/// with room and admits nothing that could be one of those two encodings.
pub const PLAUSIBLE_ELEVATION_M: std::ops::RangeInclusive<f64> = -12_000.0..=12_000.0;

/// One XYZ tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileId {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

/// Which tiles an island's extent needs, and what that costs.
#[derive(Clone, Debug, PartialEq)]
pub struct TilePlan {
    pub zoom: u8,
    /// Ascending, so the plan is a value two runs can compare.
    pub tiles: Vec<TileId>,
    /// `(min, max)` longitude the extent reaches, degrees.
    pub lon: (f64, f64),
    /// `(min, max)` latitude the extent reaches, degrees.
    pub lat: (f64, f64),
    /// True ground metres per source pixel at the extent's centre latitude —
    /// **not** the inflated Mercator figure, which would tell an author their
    /// DEM is 1.53× coarser than it is.
    pub ground_m_per_px: f64,
    /// The world grid's own sample pitch, for the comparison an author actually
    /// wants: how much of the detail in the terrain is survey and how much is
    /// design.
    pub grid_m_per_sample: f64,
}

impl TilePlan {
    /// How many tiles the plan names.
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// `true` when the plan names no tile at all.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// How many world samples one source pixel has to cover.
    ///
    /// Above 1.0 the terrain grid is finer than the survey and the difference is
    /// interpolation plus whatever the design puts there — which is a true and
    /// useful thing to print, and a false and dangerous thing to leave unsaid.
    pub fn upsample_ratio(&self) -> f64 {
        if self.grid_m_per_sample <= 0.0 {
            return 0.0;
        }
        self.ground_m_per_px / self.grid_m_per_sample
    }

    /// The tiles the plan names that the cache does not have.
    pub fn missing_in(&self, cache_dir: &Path) -> Vec<TileId> {
        self.tiles
            .iter()
            .copied()
            .filter(|t| {
                let p = cache_path(cache_dir, *t);
                !std::fs::metadata(&p).map(|m| m.len() > 0).unwrap_or(false)
            })
            .collect()
    }
}

/// Where a tile's bytes live under a cache directory.
///
/// The ordinary XYZ layout (`<cache>/terrarium/<z>/<x>/<y>.png`) on purpose: a
/// directory laid out this way can be filled by anything — this build, a
/// download manager, a colleague's zip — and the fixture's committed bytes are
/// the same shape as a fetched cache rather than a special case.
pub fn cache_path(cache_dir: &Path, t: TileId) -> PathBuf {
    cache_dir
        .join("terrarium")
        .join(t.z.to_string())
        .join(t.x.to_string())
        .join(format!("{}.png", t.y))
}

/// A tile's URL from the recipe's template.
pub fn tile_url(template: &str, t: TileId) -> String {
    template
        .replace("{z}", &t.z.to_string())
        .replace("{x}", &t.x.to_string())
        .replace("{y}", &t.y.to_string())
}

/// How many points around the world square's perimeter the plan inverts.
///
/// A projected rectangle is not a lat/lon rectangle — the north edge of a UTM
/// square bows — so taking the four corners alone under-covers the middle of
/// each edge. 64 per side is far past the bow at any island scale and costs 256
/// projections once.
const PERIMETER_SAMPLES: u32 = 64;

/// Which source tiles this recipe's extent needs.
///
/// Pure apart from the projection: no file is opened and no byte is fetched, so
/// `inf island plan` can print the download before an author commits to it.
pub fn plan_tiles(recipe: &IslandRecipe) -> Result<TilePlan, IslandError> {
    let anchor = recipe.anchor()?;
    let tf = inf_gis::Transform::new("EPSG:4326", &anchor)?;
    let half = recipe.grid.half_extent_m();

    let (mut lon0, mut lon1) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut lat0, mut lat1) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..=PERIMETER_SAMPLES {
        let f = f64::from(i) / f64::from(PERIMETER_SAMPLES);
        let t = -half + 2.0 * half * f;
        for p in [
            glam::DVec3::new(t, 0.0, -half),
            glam::DVec3::new(t, 0.0, half),
            glam::DVec3::new(-half, 0.0, t),
            glam::DVec3::new(half, 0.0, t),
        ] {
            let (lon, lat, _) = tf.to_source(p)?;
            lon0 = lon0.min(lon);
            lon1 = lon1.max(lon);
            lat0 = lat0.min(lat);
            lat1 = lat1.max(lat);
        }
    }

    let z = recipe.source.zoom;
    let (x0, y1) = tilemath::tile_of_lonlat(z, lon0, lat0).ok_or_else(|| {
        IslandError::Settings(format!(
            "the extent's south-west corner ({lon0}, {lat0}) has no tile at zoom {z}"
        ))
    })?;
    let (x1, y0) = tilemath::tile_of_lonlat(z, lon1, lat1).ok_or_else(|| {
        IslandError::Settings(format!(
            "the extent's north-east corner ({lon1}, {lat1}) has no tile at zoom {z}"
        ))
    })?;

    // The margin exists so the bilinear read at the very edge of the world has a
    // neighbour to interpolate against; without it the last row of samples reads
    // a clamped pixel and the island's border is subtly flat.
    let m = recipe.source.tile_margin;
    let last = tilemath::tiles_at_zoom(z).saturating_sub(1) as u32;
    let (x0, x1) = (x0.saturating_sub(m), x1.saturating_add(m).min(last));
    let (y0, y1) = (y0.saturating_sub(m), y1.saturating_add(m).min(last));

    let mut tiles = Vec::with_capacity(((x1 - x0 + 1) as usize) * ((y1 - y0 + 1) as usize));
    for x in x0..=x1 {
        for y in y0..=y1 {
            tiles.push(TileId { z, x, y });
        }
    }
    tiles.sort();

    let mid_lat = 0.5 * (lat0 + lat1);
    let ground = tilemath::ground_resolution_m(z, mid_lat).ok_or_else(|| {
        IslandError::Settings(format!(
            "the extent's centre latitude {mid_lat} has no resolution"
        ))
    })?;

    Ok(TilePlan {
        zoom: z,
        tiles,
        lon: (lon0, lon1),
        lat: (lat0, lat1),
        ground_m_per_px: ground,
        grid_m_per_sample: recipe.grid.meters_per_sample,
    })
}

/// A decoded mosaic of source tiles, answering elevations at geodetic positions.
#[derive(Clone, Debug)]
pub struct TileMosaic {
    zoom: u8,
    tiles: BTreeMap<(u32, u32), TerrariumTile>,
    /// Tiles the plan named whose every sample is exactly sea level.
    ///
    /// Reported, never corrected: a tile that is 100 % sea level is either
    /// genuine open ocean or a fetch that never happened, and only the extent
    /// knows which. The carve step is where that decision is taken.
    sea_level_tiles: BTreeSet<(u32, u32)>,
    /// How many samples across the whole mosaic decoded to exactly sea level.
    sea_level_samples: usize,
    /// How many decoded to an elevation Earth does not have. Counted rather than
    /// silently dropped — a provider that fills a tile is a fact about the
    /// source and an author should hear it.
    implausible_samples: usize,
    /// Total samples decoded.
    samples: usize,
    lo: f64,
    hi: f64,
}

impl TileMosaic {
    /// Decode every tile the plan names out of the cache.
    ///
    /// Memory: one `f64` per source pixel. At the island's own plan (132 tiles at
    /// z15) that is 8.6 M samples — 69 MB, which is why the mosaic is decoded
    /// once and the destination walk reads it, rather than the other way round.
    pub fn load(plan: &TilePlan, cache_dir: &Path) -> Result<Self, IslandError> {
        let mut tiles = BTreeMap::new();
        for t in &plan.tiles {
            let path = cache_path(cache_dir, *t);
            let bytes = std::fs::read(&path).map_err(|_| IslandError::MissingTile {
                z: t.z,
                x: t.x,
                y: t.y,
                cache: cache_dir.display().to_string(),
            })?;
            let tile =
                inf_gis::terrarium::decode_tile_png(&bytes).map_err(|e| IslandError::BadTile {
                    z: t.z,
                    x: t.x,
                    y: t.y,
                    path: path.display().to_string(),
                    message: e.to_string(),
                })?;
            tiles.insert((t.x, t.y), tile);
        }
        Ok(Self::from_tiles(plan.zoom, tiles))
    }

    /// Build a mosaic straight from decoded tiles — the seam a test uses so it
    /// does not need a filesystem, and the one [`TileMosaic::load`] goes through,
    /// so the summary numbers cannot differ between the two.
    pub fn from_tiles(zoom: u8, tiles: BTreeMap<(u32, u32), TerrariumTile>) -> Self {
        let mut sea_level_tiles = BTreeSet::new();
        let mut sea_level_samples = 0usize;
        let mut implausible_samples = 0usize;
        let mut samples = 0usize;
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for (k, t) in &tiles {
            if t.is_uniformly_sea_level() {
                sea_level_tiles.insert(*k);
            }
            sea_level_samples += t.sea_level_samples;
            samples += t.elevations.len();
            // The range is over PLAUSIBLE samples only: a mosaic that reported
            // −32 768 m as its floor would be reporting the codec, not the
            // ground, and every number derived from it downstream would inherit
            // that.
            for v in &t.elevations {
                if PLAUSIBLE_ELEVATION_M.contains(v) {
                    lo = lo.min(*v);
                    hi = hi.max(*v);
                } else {
                    implausible_samples += 1;
                }
            }
        }
        Self {
            zoom,
            tiles,
            sea_level_tiles,
            sea_level_samples,
            implausible_samples,
            samples,
            lo,
            hi,
        }
    }

    /// How many source samples carry an elevation Earth does not have.
    pub fn implausible_samples(&self) -> usize {
        self.implausible_samples
    }

    /// The tiles that are entirely sea level.
    pub fn sea_level_tiles(&self) -> &BTreeSet<(u32, u32)> {
        &self.sea_level_tiles
    }

    /// The fraction of source samples that decoded to exactly sea level.
    pub fn sea_level_fraction(&self) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        self.sea_level_samples as f64 / self.samples as f64
    }

    /// The `(min, max)` **plausible** elevation the source carries, or `None`
    /// when it carries none.
    pub fn range(&self) -> Option<(f64, f64)> {
        (self.lo.is_finite() && self.hi.is_finite()).then_some((self.lo, self.hi))
    }

    /// How many tiles are loaded.
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// One source pixel's elevation, in **global pixel coordinates** at this
    /// mosaic's zoom. `None` outside the loaded tiles.
    fn pixel(&self, gx: i64, gy: i64) -> Option<f64> {
        if gx < 0 || gy < 0 {
            return None;
        }
        let px = tilemath::TILE_PX as i64;
        let t = self.tiles.get(&((gx / px) as u32, (gy / px) as u32))?;
        // A provider may serve 512s; the tile knows its own width.
        let scale = t.width as i64 / px;
        let (ix, iy) = if scale > 1 {
            ((gx % px) * scale, (gy % px) * scale)
        } else {
            (gx % px, gy % px)
        };
        // **The implausible-elevation door.** A filled black pixel decodes to
        // −32 768 m, which is finite and therefore invisible to every other
        // guard. `None` here makes it nodata, which the carve turns into ocean —
        // the one policy this pipeline already has for "the source does not
        // cover this ground". See [`PLAUSIBLE_ELEVATION_M`].
        t.get(ix as u32, iy as u32)
            .filter(|v| PLAUSIBLE_ELEVATION_M.contains(v))
    }

    /// Bilinear elevation at a geodetic position, or `None` off the mosaic.
    pub fn elevation_at(&self, lon_deg: f64, lat_deg: f64) -> Option<f64> {
        let (mx, my) = tilemath::lonlat_to_mercator(lon_deg, lat_deg)?;
        let n = tilemath::tiles_at_zoom(self.zoom) as f64 * tilemath::TILE_PX as f64;
        let px = (mx + tilemath::MERC_HALF_WORLD) / tilemath::MERC_WORLD * n;
        let py = (tilemath::MERC_HALF_WORLD - my) / tilemath::MERC_WORLD * n;
        self.elevation_at_pixel(px, py)
    }

    /// Bilinear elevation at **global source pixel coordinates**, or `None` off
    /// the mosaic.
    ///
    /// This is the door the island's sample walk uses, because its projection
    /// lattice tabulates pixel coordinates directly and asking through
    /// longitude and latitude would undo that.
    ///
    /// The half-pixel offset is not decoration: a terrarium sample is the value
    /// *of the pixel's area*, so its representative point is the pixel centre.
    /// Interpolating against pixel corners shifts the whole terrain half a source
    /// pixel north-west — 1.6 m at this recipe's zoom, which is a metre and a
    /// half of coastline in the wrong place.
    pub fn elevation_at_pixel(&self, px: f64, py: f64) -> Option<f64> {
        if !(px.is_finite() && py.is_finite()) {
            return None;
        }
        let px = px - 0.5;
        let py = py - 0.5;
        let x0 = px.floor();
        let y0 = py.floor();
        let fx = px - x0;
        let fy = py - y0;
        let (x0, y0) = (x0 as i64, y0 as i64);
        let a = self.pixel(x0, y0)?;
        let b = self.pixel(x0 + 1, y0)?;
        let c = self.pixel(x0, y0 + 1)?;
        let d = self.pixel(x0 + 1, y0 + 1)?;
        let top = a + (b - a) * fx;
        let bot = c + (d - c) * fx;
        Some(top + (bot - top) * fy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_gis::terrarium::encode_elevation;

    /// A synthetic tile whose elevation is a known function of its own pixel, so
    /// a sampler that reads the wrong pixel reads a wrong NUMBER rather than a
    /// plausible one.
    fn ramp_tile(x: u32, y: u32) -> TerrariumTile {
        let px = tilemath::TILE_PX as u32;
        let mut rgb = Vec::with_capacity((px * px * 3) as usize);
        for j in 0..px {
            for i in 0..px {
                // Global pixel index folded into a metre value.
                let gx = f64::from(x * px + i);
                let gy = f64::from(y * px + j);
                rgb.extend_from_slice(&encode_elevation(gx * 0.25 + gy * 0.0625));
            }
        }
        inf_gis::terrarium::decode_tile_rgb(&rgb, px, px, 3).unwrap()
    }

    fn flat_tile(v: f64) -> TerrariumTile {
        let px = tilemath::TILE_PX as u32;
        let rgb: Vec<u8> = std::iter::repeat_n(encode_elevation(v), (px * px) as usize)
            .flatten()
            .collect();
        inf_gis::terrarium::decode_tile_rgb(&rgb, px, px, 3).unwrap()
    }

    fn recipe() -> IslandRecipe {
        IslandRecipe::parse(
            &crate::recipe::tests::tiny_recipe_text(),
            Path::new("/tmp/island"),
        )
        .unwrap()
    }

    #[test]
    fn a_plan_covers_the_extent_with_a_margin_and_prices_the_source() {
        let r = recipe();
        let plan = plan_tiles(&r).expect("the tiny recipe plans");
        assert_eq!(plan.zoom, 14);
        assert!(!plan.is_empty());
        assert_eq!(plan.len(), plan.tiles.len());

        // Ascending and unique — a plan is a value two runs compare.
        let mut sorted = plan.tiles.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted, plan.tiles, "the plan must be sorted and unique");

        // The extent's own corners are inside the planned tile rectangle, and the
        // margin puts at least one tile beyond each edge.
        let (x0, x1) = (
            plan.tiles.iter().map(|t| t.x).min().unwrap(),
            plan.tiles.iter().map(|t| t.x).max().unwrap(),
        );
        let (y0, y1) = (
            plan.tiles.iter().map(|t| t.y).min().unwrap(),
            plan.tiles.iter().map(|t| t.y).max().unwrap(),
        );
        let (cx0, cy1) = tilemath::tile_of_lonlat(14, plan.lon.0, plan.lat.0).unwrap();
        let (cx1, cy0) = tilemath::tile_of_lonlat(14, plan.lon.1, plan.lat.1).unwrap();
        assert!(
            x0 < cx0 && x1 > cx1,
            "the margin must reach past the extent"
        );
        assert!(y0 < cy0 && y1 > cy1);

        // The source is priced in TRUE ground metres, and the honest comparison
        // against the grid is printed rather than implied.
        let eq = tilemath::ground_resolution_m(14, 0.0).unwrap();
        assert!(
            plan.ground_m_per_px < eq * 0.7,
            "at 49 N a z14 pixel covers much less ground than at the equator \
             ({} vs {eq}) — if these are equal the cos(lat) term is missing",
            plan.ground_m_per_px
        );
        assert_eq!(plan.grid_m_per_sample, 2.0);
        println!(
            "PLAN  z{} {} tiles  source {:.3} m/px  grid {:.3} m/sample  upsample {:.2}x",
            plan.zoom,
            plan.len(),
            plan.ground_m_per_px,
            plan.grid_m_per_sample,
            plan.upsample_ratio()
        );
        assert!((plan.upsample_ratio() - plan.ground_m_per_px / 2.0).abs() < 1e-12);
    }

    /// The perimeter walk is not decoration: a projected square's north edge bows,
    /// so the four corners alone under-cover it.
    ///
    /// Measured here rather than asserted from theory — the assertion is that the
    /// *widest* latitude on the north edge is strictly north of both north
    /// corners.
    #[test]
    fn a_projected_square_is_not_a_lat_lon_rectangle() {
        let mut r = recipe();
        // A big square makes the bow measurable; a 512 m one does not.
        r.grid.tiles = 200; // 200 x 256 m = 51.2 km
        let anchor = r.anchor().unwrap();
        let tf = inf_gis::Transform::new("EPSG:4326", &anchor).unwrap();
        let half = r.grid.half_extent_m();
        let corner_n = [-half, half]
            .iter()
            .map(|&x| tf.to_source(glam::DVec3::new(x, 0.0, -half)).unwrap().1)
            .fold(f64::NEG_INFINITY, f64::max);
        let mid_n = tf.to_source(glam::DVec3::new(0.0, 0.0, -half)).unwrap().1;
        let bow_m = (mid_n - corner_n) * 111_320.0;
        println!("NORTH-EDGE BOW at 51.2 km: {bow_m:.1} m");
        assert!(
            bow_m > 1.0,
            "the north edge's middle is only {bow_m} m from its corners; if this \
             is zero the perimeter walk is measuring nothing"
        );
        // And the plan's own latitude bound reaches it.
        let plan = plan_tiles(&r).unwrap();
        assert!(plan.lat.1 >= mid_n - 1e-12);
    }

    #[test]
    fn cache_paths_and_urls_are_the_ordinary_xyz_layout() {
        let t = TileId {
            z: 15,
            x: 5179,
            y: 11205,
        };
        let p = cache_path(Path::new("/c"), t);
        let s = p.to_string_lossy().replace('\\', "/");
        assert!(s.ends_with("/c/terrarium/15/5179/11205.png"), "{s}");
        assert_eq!(
            tile_url("https://h/{z}/{x}/{y}.png", t),
            "https://h/15/5179/11205.png"
        );
    }

    #[test]
    fn the_mosaic_reads_the_pixel_it_was_asked_for() {
        // Two tiles side by side at z2, so the arithmetic is checkable by hand.
        let mut tiles = BTreeMap::new();
        tiles.insert((1, 1), ramp_tile(1, 1));
        tiles.insert((2, 1), ramp_tile(2, 1));
        let m = TileMosaic::from_tiles(2, tiles);
        assert_eq!(m.tile_count(), 2);
        assert!(m.sea_level_tiles().is_empty());

        // The centre of global pixel (300, 300) must read that pixel's own value.
        let px = tilemath::TILE_PX as f64;
        let n = 4.0 * px;
        let want = 300.0 * 0.25 + 300.0 * 0.0625;
        let mx = (300.5 / n) * tilemath::MERC_WORLD - tilemath::MERC_HALF_WORLD;
        let my = tilemath::MERC_HALF_WORLD - (300.5 / n) * tilemath::MERC_WORLD;
        let (lon, lat) = tilemath::mercator_to_lonlat(mx, my).unwrap();
        let got = m.elevation_at(lon, lat).expect("inside the mosaic");
        assert!(
            (got - want).abs() < 1e-6,
            "pixel (300,300) is {want} m; the sampler read {got} m — a half-pixel \
             error here shifts the whole terrain"
        );

        // Halfway between two pixels is the mean of the two.
        let mx = (301.0 / n) * tilemath::MERC_WORLD - tilemath::MERC_HALF_WORLD;
        let (lon2, _) = tilemath::mercator_to_lonlat(mx, my).unwrap();
        let mid = m.elevation_at(lon2, lat).unwrap();
        let want_mid = 0.5 * (want + (301.0 * 0.25 + 300.0 * 0.0625));
        assert!((mid - want_mid).abs() < 1e-6, "{mid} vs {want_mid}");

        // Off the mosaic is None, not a plausible zero.
        assert_eq!(m.elevation_at(179.0, 0.0), None);
        assert_eq!(m.elevation_at(f64::NAN, 0.0), None);
    }

    /// The missing-tile / open-ocean ambiguity is REPORTED and not resolved here.
    #[test]
    fn a_sea_level_tile_is_flagged_rather_than_corrected() {
        let mut tiles = BTreeMap::new();
        tiles.insert((1, 1), flat_tile(0.0));
        tiles.insert((2, 1), ramp_tile(2, 1));
        let m = TileMosaic::from_tiles(2, tiles);
        assert_eq!(m.sea_level_tiles().len(), 1);
        assert!(m.sea_level_tiles().contains(&(1, 1)));
        // The fraction counts SAMPLES, which is what an extent-level policy reads.
        let px = tilemath::TILE_PX as usize;
        assert!(
            (m.sea_level_fraction() - 0.5).abs() < 0.01,
            "one flat tile of two is about half the samples, got {}",
            m.sea_level_fraction()
        );
        assert_eq!(m.range().map(|r| r.0), Some(0.0));
        // The ramp tile at (2, 1) tops out at its own far corner — computed from
        // the fixture's own rule rather than guessed at.
        let want = f64::from(3 * px as u32 - 1) * 0.25 + f64::from(2 * px as u32 - 1) * 0.0625;
        assert!(
            (m.range().unwrap().1 - want).abs() < 1e-9,
            "the mosaic's range is {:?} and the ramp's own corner is {want}",
            m.range()
        );
    }

    /// **A BLACK PIXEL IS FINITE**, and that is what makes it dangerous.
    ///
    /// The shipped island's own source carries `(0, 0, 0)` samples, which decode
    /// to −32 768 m. Every guard in this repository checks finiteness; none of
    /// them can see this. Un-fix mutation: delete the `.filter(...)` in `pixel`
    /// and the sampler below returns −32 768 instead of `None`.
    #[test]
    fn an_implausible_elevation_is_nodata_rather_than_a_thirty_two_kilometre_pit() {
        let px = tilemath::TILE_PX as u32;
        // A tile that is half real ground and half the codec's own floor.
        let mut rgb = Vec::new();
        for j in 0..px {
            for _ in 0..px {
                if j < px / 2 {
                    rgb.extend_from_slice(&encode_elevation(140.0));
                } else {
                    rgb.extend_from_slice(&[0u8, 0, 0]);
                }
            }
        }
        let tile = inf_gis::terrarium::decode_tile_rgb(&rgb, px, px, 3).unwrap();
        // The codec itself is happy with it — the hazard is real and it is here.
        assert_eq!(tile.range(), Some((-32768.0, 140.0)));
        assert!(tile.range().unwrap().0.is_finite(), "and it is FINITE");

        let mut tiles = BTreeMap::new();
        tiles.insert((1, 1), tile);
        let m = TileMosaic::from_tiles(2, tiles);
        assert_eq!(
            m.implausible_samples(),
            (px * px / 2) as usize,
            "half the tile is the codec's floor"
        );
        assert_eq!(
            m.range(),
            Some((140.0, 140.0)),
            "the mosaic reports the GROUND's range, not the codec's"
        );

        // And the sampler answers nodata there, which the carve turns into ocean.
        let n = 4.0 * tilemath::TILE_PX as f64;
        let probe = |gy: f64| {
            let mx = (300.5 / n) * tilemath::MERC_WORLD - tilemath::MERC_HALF_WORLD;
            let my = tilemath::MERC_HALF_WORLD - (gy / n) * tilemath::MERC_WORLD;
            let (lon, lat) = tilemath::mercator_to_lonlat(mx, my).unwrap();
            m.elevation_at(lon, lat)
        };
        assert_eq!(probe(300.5), Some(140.0), "the real half answers");
        assert_eq!(probe(420.5), None, "the filled half is nodata");
        assert_eq!(
            crate::shape::carve_sample(probe(420.5), 5_000.0, 0.0, 60.0, 500.0, 32.0),
            0.0,
            "and nodata inland becomes sea level, not a pit"
        );
        // The bound admits Earth and refuses both encodings that are not.
        assert!(
            PLAUSIBLE_ELEVATION_M.contains(&-10_935.0),
            "the Mariana Trench"
        );
        assert!(PLAUSIBLE_ELEVATION_M.contains(&8_849.0), "Everest");
        assert!(!PLAUSIBLE_ELEVATION_M.contains(&-32_768.0));
        assert!(!PLAUSIBLE_ELEVATION_M.contains(&32_767.996));
    }

    #[test]
    fn a_missing_or_corrupt_cache_entry_is_refused_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let plan = TilePlan {
            zoom: 15,
            tiles: vec![TileId { z: 15, x: 1, y: 2 }],
            lon: (0.0, 1.0),
            lat: (0.0, 1.0),
            ground_m_per_px: 3.0,
            grid_m_per_sample: 1.0,
        };
        assert_eq!(plan.missing_in(dir.path()).len(), 1);
        let e = TileMosaic::load(&plan, dir.path()).unwrap_err().to_string();
        assert!(e.contains("15/1/2") && e.contains("--offline"), "{e}");

        // A 404 body cached as a tile is refused, not treated as absent — the
        // hazard that would otherwise build a flat plain where a mountain is.
        let p = cache_path(dir.path(), plan.tiles[0]);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(
            &p,
            b"<?xml version=\"1.0\"?><Error><Code>NoSuchKey</Code></Error>",
        )
        .unwrap();
        assert!(plan.missing_in(dir.path()).is_empty(), "it is present");
        let e = TileMosaic::load(&plan, dir.path()).unwrap_err().to_string();
        assert!(
            e.contains("not a terrarium PNG") && e.contains("flat plain"),
            "{e}"
        );
    }
}
