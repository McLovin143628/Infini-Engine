//! The world grid, the projection lattice, and the sample walk that puts real
//! elevation on it.

use glam::{DVec2, DVec3};
use inf_terrain::{TerrainData, TerrainTile};

use crate::recipe::IslandRecipe;
use crate::shape::{carve_sample, flatten_sample, smooth01, Coastline, Field, SegmentIndex};
use crate::source::TileMosaic;
use crate::IslandError;

/// How far apart the projection lattice's control points sit, in world metres.
///
/// **This is a measured number, not a preference.** The UTM → geodetic map is
/// smooth: over a 64 m step its second-order term is ~`tan(lat)/R²·h²/8`, which
/// at 49 N and 64 m is about `1.5e-11` degrees — a *nanometre* on the ground.
/// The arm `the_projection_lattice_agrees_with_the_projection` measures it
/// against the real inverse rather than trusting that arithmetic, and it is what
/// would go red if the lattice were ever coarsened past what the map can carry.
///
/// What it buys: the island's 51 million samples cost 51 million bilinear reads
/// instead of 51 million `proj4rs` inversions.
pub const PROJECTION_LATTICE_M: f64 = 64.0;

/// The world → source-pixel map, tabulated.
///
/// Two fields, holding **global source pixel coordinates** at the mosaic's zoom.
/// Folding the whole chain (world → UTM → geodetic → Mercator → pixel) into one
/// pair of tables is what keeps the per-sample cost to two bilinears, and it is
/// also the only place in this crate that touches a transcendental function.
#[derive(Clone, Debug)]
pub struct ProjectionLattice {
    px: Field,
    py: Field,
}

impl ProjectionLattice {
    /// Tabulate the map over `[min, max]` at [`PROJECTION_LATTICE_M`].
    pub fn build(
        tf: &inf_gis::Transform,
        zoom: u8,
        min: DVec2,
        max: DVec2,
    ) -> Result<Self, IslandError> {
        // One lattice cell of slack each way, so a sample exactly on the world's
        // border still interpolates between two control points rather than
        // clamping onto one.
        let pad = DVec2::splat(PROJECTION_LATTICE_M);
        let mut px = Field::new(min - pad, max + pad, PROJECTION_LATTICE_M);
        let mut py = Field::new(min - pad, max + pad, PROJECTION_LATTICE_M);
        let (nx, nz) = px.dims();
        let n = inf_gis::tilemath::tiles_at_zoom(zoom) as f64 * inf_gis::tilemath::TILE_PX as f64;
        for j in 0..nz {
            for i in 0..nx {
                let p = px.position(i, j);
                let (lon, lat, _) = tf.to_source(DVec3::new(p.x, 0.0, p.y))?;
                let (mx, my) =
                    inf_gis::tilemath::lonlat_to_mercator(lon, lat).ok_or_else(|| {
                        IslandError::Settings(format!(
                            "the world position ({}, {}) inverts to ({lon}, {lat}), which \
                         has no Mercator coordinate",
                            p.x, p.y
                        ))
                    })?;
                let gx =
                    (mx + inf_gis::tilemath::MERC_HALF_WORLD) / inf_gis::tilemath::MERC_WORLD * n;
                let gy =
                    (inf_gis::tilemath::MERC_HALF_WORLD - my) / inf_gis::tilemath::MERC_WORLD * n;
                px.set(i, j, gx);
                py.set(i, j, gy);
            }
        }
        Ok(Self { px, py })
    }

    /// Global source pixel coordinates for a world XZ position.
    #[inline]
    pub fn pixel_at(&self, p: DVec2) -> (f64, f64) {
        (self.px.at(p), self.py.at(p))
    }

    /// How many control points the lattice holds.
    pub fn control_points(&self) -> usize {
        let (nx, nz) = self.px.dims();
        nx * nz
    }
}

/// The world grid's geometry — the tile coordinate range and the world bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IslandGrid {
    /// Samples per tile edge.
    pub tile_resolution: u32,
    /// World metres between samples.
    pub meters_per_sample: f64,
    /// Level-0 tiles per axis.
    pub tiles: u32,
    /// The lowest tile coordinate on both axes.
    pub tile0: i32,
}

impl IslandGrid {
    /// The grid a recipe describes, centred on the world origin.
    ///
    /// Centred rather than corner-anchored: the player start, the floating
    /// origin and the partition's own cell (0, 0) all live near the world
    /// origin, and a 50 km² world hung off one corner spends its whole east half
    /// at seven kilometres before the first rebase.
    pub fn of(recipe: &IslandRecipe) -> Self {
        Self {
            tile_resolution: recipe.grid.tile_resolution,
            meters_per_sample: recipe.grid.meters_per_sample,
            tiles: recipe.grid.tiles,
            tile0: -((recipe.grid.tiles as i32) / 2),
        }
    }

    /// One tile's world span.
    pub fn tile_span_m(&self) -> f64 {
        f64::from(self.tile_resolution.saturating_sub(1)) * self.meters_per_sample
    }

    /// `(min, max)` world XZ the grid covers.
    pub fn bounds(&self) -> (DVec2, DVec2) {
        let span = self.tile_span_m();
        let lo = f64::from(self.tile0) * span;
        let hi = lo + f64::from(self.tiles) * span;
        (DVec2::splat(lo), DVec2::splat(hi))
    }

    /// Every level-0 tile coordinate, in `(tz, tx)` row-major order.
    pub fn tile_coords(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        let t0 = self.tile0;
        let n = self.tiles as i32;
        (0..n).flat_map(move |dz| (0..n).map(move |dx| (t0 + dx, t0 + dz)))
    }

    /// An empty `TerrainData` with this grid's geometry.
    pub fn empty_data(&self) -> TerrainData {
        TerrainData::new(self.tile_resolution, self.meters_per_sample)
    }
}

/// Everything the carve needs that is not a height.
#[derive(Debug)]
pub struct CarvePlan<'a> {
    pub coast: &'a Coastline,
    /// `(centre, radius, datum)` per site pad.
    pub pads: Vec<(DVec2, f64, f64)>,
    /// The road corridor, if the design has one yet.
    pub corridor: Option<&'a SegmentIndex>,
    /// Half the levelled corridor's width, metres — where the easing reaches
    /// zero and the terrain is its own again.
    pub corridor_half_m: f64,
    /// **Half the corridor's FLAT plateau**, metres (wave ROAD1).
    ///
    /// # The easing used to start at the centreline, and the road sat in a bowl
    ///
    /// The levelling was `w = 1 − smooth01(d / half)`: full at the centreline and
    /// tapering from there, so the ground under the road's own edge was only
    /// partly eased toward the design height. On the island's 11.2 m corridor a
    /// 14 m carriageway's edge sat at `w = 0.32` — sixty-eight per cent of the
    /// terrain's natural cross-slope survived under a surface a road is graded
    /// flat across.
    ///
    /// Everything the road *draws* now sits on a plateau at exactly the route's
    /// height (`w = 1`), and the smoothstep eases from the plateau's edge out to
    /// [`corridor_half_m`](Self::corridor_half_m) — which is what a cut-and-fill
    /// batter is.
    ///
    /// **And it is what makes the road conform to the terrain the renderer
    /// DRAWS.** The clipmap morphs a vertex toward a bilinear on a lattice twice
    /// as coarse, and a locally planar surface decimates to itself: on a plateau,
    /// the coarse height equals the fine height, so `mix(h_fine, h_coarse, m)` is
    /// the same number at every morph factor and the ground under the road stops
    /// moving with the camera. `road1_gate`'s three-distance table is the
    /// measurement, and it says the graded road degrades **less** under the morph
    /// than a conforming one: +0.0388 m and +0.1626 m off its own baseline
    /// against +0.0893 m and +0.1687 m.
    ///
    /// # A WIDER plateau is worse, and that was measured rather than assumed
    ///
    /// The obvious next step is to floor this at one of the renderer's own coarse
    /// cells — 8 m at ring 0, 16 m at ring 1 — so the decimation's samples land
    /// inside the flat rather than on its batter. **It was tried at 16 m and it
    /// is worse**: a grade-limited router builds switchbacks, several limbs of
    /// one road pass within a plateau's width of each other at different heights,
    /// and the levelling takes the nearest — so a wide plateau is not a flat, it
    /// is a **staircase**, and each step is a feature the decimation then
    /// smooths. The fixture's worst road vertex went from 0.889 m off the drawn
    /// terrain (conforming, no plateau) to 1.51 m at the road's own built
    /// half-width, and no better at 16 m.
    ///
    /// So the plateau is the road's own built half-width and no more, the gate
    /// reads the MEAN, and the switchback is a carried item rather than a
    /// number hidden behind a percentile.
    pub corridor_flat_m: f64,
}

/// What the sample walk found.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SampleStats {
    pub samples: u64,
    /// Samples whose source had no elevation to give.
    pub nodata: u64,
    /// Samples inside the coastline.
    pub land: u64,
    /// Samples the road corridor levelled.
    pub corridor: u64,
    /// Samples a site pad moved.
    pub pad: u64,
    /// Samples **inside** the coastline whose carved ground is still under the
    /// waterline.
    ///
    /// A real inlet the design kept, or a design that put its shore across a
    /// valley. The two are indistinguishable from inside the loop, so the number
    /// is reported rather than acted on — and it is why the island's own floor
    /// is not the same quantity as its sea floor.
    pub submerged_land: u64,
    pub lo_m: f64,
    pub hi_m: f64,
    /// The lowest sample **outside** the coastline — the sea floor proper.
    pub sea_floor_m: f64,
    /// The highest point on **land**, which is the peak a ledger quotes (the
    /// overall maximum could be a sea-floor artefact and is not the same claim).
    pub peak_land_m: f64,
    /// Land above sea level, in square metres.
    pub land_area_m2: f64,
}

impl SampleStats {
    fn new() -> Self {
        Self {
            lo_m: f64::INFINITY,
            hi_m: f64::NEG_INFINITY,
            sea_floor_m: f64::INFINITY,
            peak_land_m: f64::NEG_INFINITY,
            ..Default::default()
        }
    }
}

/// Walk the destination grid, sampling real elevation and carving the design
/// into it.
///
/// This is `BuildStep::Sample` and `BuildStep::Carve` in one pass over the
/// samples, deliberately: they are two decisions about one number, and splitting
/// them would mean materialising a 51-million-sample intermediate whose only
/// purpose is to be overwritten.
pub fn sample_terrain(
    recipe: &IslandRecipe,
    mosaic: &TileMosaic,
    lattice: &ProjectionLattice,
    plan: &CarvePlan<'_>,
) -> (TerrainData, SampleStats) {
    let grid = IslandGrid::of(recipe);
    let mut data = grid.empty_data();
    let mut st = SampleStats::new();
    let sea = recipe.sea;
    let cell_area = recipe.grid.meters_per_sample * recipe.grid.meters_per_sample;

    for coord in grid.tile_coords() {
        // `author_tile` walks the tile's own samples and hands each one its world
        // position, so the shared-edge rule (a tile's last row IS its
        // neighbour's first) is the terrain crate's to keep, not this one's.
        let mut tile_nodata = 0u64;
        let mut tile_land = 0u64;
        let mut tile_corr = 0u64;
        let mut tile_pad = 0u64;
        let mut tile_lo = f64::INFINITY;
        let mut tile_hi = f64::NEG_INFINITY;
        let mut tile_sea_floor = f64::INFINITY;
        let mut tile_submerged = 0u64;
        let mut tile_peak = f64::NEG_INFINITY;
        let mut tile_land_cells = 0u64;
        let mut tile_samples = 0u64;

        data.author_tile(coord, |x, z| {
            let p = DVec2::new(x, z);
            let (gx, gy) = lattice.pixel_at(p);
            let src = mosaic.elevation_at_pixel(gx, gy);
            if src.is_none() {
                tile_nodata += 1;
            }
            let d = plan.coast.distance_at(p);
            if d > 0.0 {
                tile_land += 1;
            }
            let mut h = carve_sample(
                src,
                d,
                sea.level_m,
                sea.shelf_depth_m,
                sea.shelf_width_m,
                sea.beach_width_m,
            );

            // Site pads, in recipe order. A sample inside two pads is moved by
            // both, nearest last — which is a design decision an author can see
            // (the pads overlap) rather than an ordering accident.
            //
            // **A PAD DOES NOT BUILD LAND.** A city site near a shore has a
            // radius that reaches past it, and a pad that flattened toward the
            // site's own datum out there would raise the sea floor to thirteen
            // metres and hand the island a rectangular headland nobody designed.
            // The coastline is the authority on where the land ends; the pad only
            // levels what is already inside it.
            if d > 0.0 {
                for (c, r, datum) in &plan.pads {
                    let dist = (p - *c).length();
                    if dist < *r {
                        let before = h;
                        h = flatten_sample(h, dist, *r, *datum);
                        if (h - before).abs() > 1e-9 {
                            tile_pad += 1;
                        }
                    }
                }
            }

            // The road corridor. Levelled ACROSS the road only: the centreline's
            // own profile is what it is eased toward, and that profile is the
            // route's, which was planned against a grade bound. Levelling ALONG
            // as well would flatten the switchbacks the grade bound produced.
            if let Some(idx) = plan.corridor {
                if plan.corridor_half_m > 0.0 {
                    if let Some(n) = idx.nearest(p) {
                        if n.distance_m < plan.corridor_half_m {
                            // **A plateau, then a batter** (wave ROAD1). Inside
                            // the flat the ground IS the route's design height;
                            // outside it the smoothstep eases back to the
                            // terrain's own over what is left of the corridor.
                            // See `CarvePlan::corridor_flat_m`.
                            let flat = plan.corridor_flat_m.min(plan.corridor_half_m);
                            let w = if n.distance_m <= flat {
                                1.0
                            } else {
                                let span = plan.corridor_half_m - flat;
                                if span > 0.0 {
                                    1.0 - smooth01((n.distance_m - flat) / span)
                                } else {
                                    0.0
                                }
                            };
                            let before = h;
                            h = h + (n.height_m - h) * w;
                            if (h - before).abs() > 1e-9 {
                                tile_corr += 1;
                            }
                        }
                    }
                }
            }

            tile_samples += 1;
            tile_lo = tile_lo.min(h);
            tile_hi = tile_hi.max(h);
            if d <= 0.0 {
                tile_sea_floor = tile_sea_floor.min(h);
            } else if h <= sea.level_m {
                tile_submerged += 1;
            }
            if h > sea.level_m {
                tile_land_cells += 1;
                tile_peak = tile_peak.max(h);
            }
            h
        });

        st.samples += tile_samples;
        st.nodata += tile_nodata;
        st.land += tile_land;
        st.corridor += tile_corr;
        st.pad += tile_pad;
        st.lo_m = st.lo_m.min(tile_lo);
        st.hi_m = st.hi_m.max(tile_hi);
        st.sea_floor_m = st.sea_floor_m.min(tile_sea_floor);
        st.submerged_land += tile_submerged;
        st.peak_land_m = st.peak_land_m.max(tile_peak);
        st.land_area_m2 += tile_land_cells as f64 * cell_area;
    }
    (data, st)
}

/// A coarse height grid over the world, for the derivations that cannot afford
/// the full lattice.
///
/// # Why the derivations get their own grid
///
/// Flow accumulation needs every cell sorted by height, and a depression fill
/// needs a priority queue over the same set. At the island's 51 million samples
/// that is 600 MB of index-and-key on top of the terrain itself. At the 8 m
/// pitch a stream is actually authored at, it is six.
///
/// The honest cost is stated where it is paid: a channel narrower than the pitch
/// cannot be found, which is why the pitch is a recipe-adjacent constant and the
/// stream layer is a *design* artifact that an author may edit rather than an
/// oracle.
#[derive(Clone, Debug)]
pub struct CoarseHeights {
    pub min: DVec2,
    pub pitch: f64,
    pub nx: usize,
    pub nz: usize,
    pub h: Vec<f32>,
    /// `false` where the terrain had nothing to say.
    pub known: Vec<bool>,
}

impl CoarseHeights {
    /// Sample a terrain onto a coarse lattice.
    pub fn of(data: &TerrainData, min: DVec2, max: DVec2, pitch: f64) -> Self {
        let pitch = if pitch.is_finite() && pitch > 0.0 {
            pitch
        } else {
            8.0
        };
        let nx = ((max.x - min.x) / pitch).ceil() as usize + 1;
        let nz = ((max.y - min.y) / pitch).ceil() as usize + 1;
        let mut h = vec![0.0f32; nx * nz];
        let mut known = vec![false; nx * nz];
        for j in 0..nz {
            for i in 0..nx {
                let p = DVec2::new(min.x + i as f64 * pitch, min.y + j as f64 * pitch);
                if let Some(v) = data.height_at(p) {
                    h[j * nx + i] = v as f32;
                    known[j * nx + i] = true;
                }
            }
        }
        Self {
            min,
            pitch,
            nx,
            nz,
            h,
            known,
        }
    }

    /// The world position of cell `(i, j)`.
    #[inline]
    pub fn position(&self, i: usize, j: usize) -> DVec2 {
        DVec2::new(
            self.min.x + i as f64 * self.pitch,
            self.min.y + j as f64 * self.pitch,
        )
    }

    /// Cell count.
    #[inline]
    pub fn len(&self) -> usize {
        self.nx * self.nz
    }

    /// `true` when the lattice holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Height at a cell.
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f32 {
        self.h[j * self.nx + i]
    }

    /// The steepest slope at a cell, in degrees, by central difference.
    ///
    /// `atan` is not portable, so this answers the **tangent's** angle through
    /// `inf_math::portable::patan2_64`, which is (the portable trig door this
    /// repository already pins for exactly this reason).
    pub fn slope_deg(&self, i: usize, j: usize) -> f64 {
        let ip = (i + 1).min(self.nx - 1);
        let im = i.saturating_sub(1);
        let jp = (j + 1).min(self.nz - 1);
        let jm = j.saturating_sub(1);
        let dx =
            f64::from(self.at(ip, j) - self.at(im, j)) / ((ip - im).max(1) as f64 * self.pitch);
        let dz =
            f64::from(self.at(i, jp) - self.at(i, jm)) / ((jp - jm).max(1) as f64 * self.pitch);
        let g = (dx * dx + dz * dz).sqrt();
        inf_math::portable::patan2_64(g, 1.0).to_degrees()
    }
}

/// Write a terrain's tiles into an asset, with its pyramid.
///
/// The three-line idiom every sample generator in this repository uses, with the
/// origin the island's own: the header's `origin` is what makes a georeferenced
/// terrain land where the survey says it does, and it costs no format bump
/// because the bytes were always there.
pub fn build_asset(
    data: &TerrainData,
    origin: DVec3,
    opts: inf_terrain::PyramidOptions,
) -> Result<(inf_terrain::TerrainAsset, Vec<inf_terrain::PyramidLevel>), IslandError> {
    let pyramid = inf_terrain::build_pyramid(data, opts);
    let mut b =
        inf_terrain::TerrainAssetBuilder::new(data.tile_resolution(), data.meters_per_sample())
            .with_origin(origin)
            .with_pyramid(opts);
    for (coord, tile) in data.tiles() {
        b.insert(inf_terrain::TileKey::lod0(*coord), tile)
            .map_err(|e| IslandError::Terrain(e.to_string()))?;
    }
    for level in &pyramid {
        for (coord, tile) in &level.tiles {
            b.insert(inf_terrain::TileKey::new(level.lod, *coord), tile)
                .map_err(|e| IslandError::Terrain(e.to_string()))?;
        }
    }
    let asset = b.build().map_err(|e| IslandError::Terrain(e.to_string()))?;
    Ok((asset, pyramid))
}

/// Write biome ids into a terrain's level-0 tiles.
///
/// Straight onto the tiles rather than through `BiomeFill`, because the
/// classifier answers **per sample** rather than per polygon — and the fill's own
/// door is a polygon fill. The design masks *do* go through `BiomeFill`; see
/// [`crate::biome`].
pub fn stamp_biomes(data: &mut TerrainData, mut f: impl FnMut(DVec2) -> u8) -> u64 {
    let res = data.tile_resolution();
    let coords: Vec<(i32, i32)> = data.tiles().map(|(c, _)| *c).collect();
    let mut written = 0u64;
    for c in coords {
        let origin = data.tile_origin_xz(c);
        let mps = data.meters_per_sample();
        let Some(tile) = data.get_tile_mut(c) else {
            continue;
        };
        for j in 0..res {
            for i in 0..res {
                let p = DVec2::new(origin.x + f64::from(i) * mps, origin.y + f64::from(j) * mps);
                let id = f(p);
                if id != inf_terrain::UNASSIGNED_BIOME {
                    tile.set_biome_sample(res, i, j, id);
                    written += 1;
                }
            }
        }
    }
    written
}

/// A tile of exactly one height — the fixture helper the tests share.
#[doc(hidden)]
pub fn flat_tile(res: u32, origin: DVec3, h: f32) -> TerrainTile {
    TerrainTile::from_heights(res, origin, vec![h; (res * res) as usize])
        .expect("a full height buffer builds a tile")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe::tests::tiny_recipe_text;
    use std::path::Path;

    fn recipe() -> IslandRecipe {
        IslandRecipe::parse(&tiny_recipe_text(), Path::new("/tmp/island")).unwrap()
    }

    #[test]
    fn the_grid_is_centred_on_the_world_origin() {
        let mut r = recipe();
        r.grid.tiles = 28;
        r.grid.tile_resolution = 257;
        r.grid.meters_per_sample = 1.0;
        let g = IslandGrid::of(&r);
        assert_eq!(g.tile_span_m(), 256.0);
        assert_eq!(g.tile0, -14);
        let (lo, hi) = g.bounds();
        assert_eq!(lo, DVec2::splat(-3584.0));
        assert_eq!(hi, DVec2::splat(3584.0));
        assert_eq!(g.tile_coords().count(), 784);
        // Every coordinate is inside the bounds it claims.
        for (tx, tz) in g.tile_coords() {
            assert!((-14..14).contains(&tx) && (-14..14).contains(&tz));
        }
        // The world's own half extent and the grid's agree.
        assert_eq!(hi.x, r.grid.half_extent_m());
    }

    /// **The lattice is an approximation and this is its error, measured.**
    ///
    /// Un-fix mutation: raise `PROJECTION_LATTICE_M` to 4 096 and the worst error
    /// below leaves the tolerance.
    #[test]
    fn the_projection_lattice_agrees_with_the_projection() {
        let mut r = recipe();
        r.grid.tiles = 28;
        r.grid.tile_resolution = 257;
        r.grid.meters_per_sample = 1.0;
        let anchor = r.anchor().unwrap();
        let tf = inf_gis::Transform::new("EPSG:4326", &anchor).unwrap();
        let (min, max) = IslandGrid::of(&r).bounds();
        let lat = ProjectionLattice::build(&tf, 15, min, max).unwrap();
        assert!(lat.control_points() > 100);

        let n = inf_gis::tilemath::tiles_at_zoom(15) as f64 * inf_gis::tilemath::TILE_PX as f64;
        let mut worst_px = 0.0f64;
        // A deterministic sweep at a pitch that is coprime with the lattice, so
        // the samples land between control points rather than on them.
        let step = 37.0;
        let mut k = 0;
        let mut x = min.x;
        while x <= max.x {
            let mut z = min.y;
            while z <= max.y {
                let p = DVec2::new(x, z);
                let (gx, gy) = lat.pixel_at(p);
                let (lon, la, _) = tf.to_source(DVec3::new(x, 0.0, z)).unwrap();
                let (mx, my) = inf_gis::tilemath::lonlat_to_mercator(lon, la).unwrap();
                let want_x =
                    (mx + inf_gis::tilemath::MERC_HALF_WORLD) / inf_gis::tilemath::MERC_WORLD * n;
                let want_y =
                    (inf_gis::tilemath::MERC_HALF_WORLD - my) / inf_gis::tilemath::MERC_WORLD * n;
                worst_px = worst_px.max((gx - want_x).abs()).max((gy - want_y).abs());
                k += 1;
                z += step;
            }
            x += step;
        }
        // A z15 pixel is ~3.1 m of ground at this latitude, so express the error
        // in the unit that matters.
        let ground =
            inf_gis::tilemath::ground_resolution_m(15, anchor.origin_latitude_deg).unwrap();
        let worst_m = worst_px * ground;
        println!(
            "PROJECTION LATTICE at {PROJECTION_LATTICE_M} m: {k} probes, worst \
             {worst_px:.3e} px = {worst_m:.3e} m of ground"
        );
        assert!(k > 10_000, "the sweep visited only {k} positions");
        assert!(
            worst_m < 0.01,
            "the lattice is {worst_m} m off the projection it stands in for"
        );
    }

    #[test]
    fn the_sample_walk_carves_an_island_out_of_a_uniform_source() {
        // A source that is 500 m everywhere: nothing about the result below can
        // come from the source's shape, so every feature is the carve's.
        let px = inf_gis::tilemath::TILE_PX as u32;
        let rgb: Vec<u8> = std::iter::repeat_n(
            inf_gis::terrarium::encode_elevation(500.0),
            (px * px) as usize,
        )
        .flatten()
        .collect();
        let tile = inf_gis::terrarium::decode_tile_rgb(&rgb, px, px, 3).unwrap();

        let mut r = recipe();
        r.grid.tiles = 4;
        r.grid.tile_resolution = 65;
        r.grid.meters_per_sample = 4.0;
        r.sea.beach_width_m = 24.0;
        let g = IslandGrid::of(&r);
        let (min, max) = g.bounds();
        assert_eq!(max.x - min.x, 1024.0);

        let anchor = r.anchor().unwrap();
        let tf = inf_gis::Transform::new("EPSG:4326", &anchor).unwrap();
        let plan = crate::source::plan_tiles(&r).unwrap();
        let mut tiles = std::collections::BTreeMap::new();
        for t in &plan.tiles {
            tiles.insert((t.x, t.y), tile.clone());
        }
        let mosaic = TileMosaic::from_tiles(plan.zoom, tiles);
        let lattice = ProjectionLattice::build(&tf, plan.zoom, min, max).unwrap();

        // A 300 m-radius disc of coast, as a 64-gon.
        let ring: Vec<DVec2> = (0..64)
            .map(|k| {
                let a = std::f64::consts::TAU * f64::from(k) / 64.0;
                DVec2::new(
                    300.0 * inf_math::portable::pcos64(a),
                    300.0 * inf_math::portable::psin64(a),
                )
            })
            .collect();
        let coast = Coastline::new(
            vec![ring],
            min,
            max,
            Coastline::field_pitch_m(r.sea.beach_width_m),
        );
        let carve = CarvePlan {
            coast: &coast,
            pads: vec![(DVec2::ZERO, 120.0, 480.0)],
            corridor: None,
            corridor_half_m: 0.0,
            corridor_flat_m: 0.0,
        };
        let (data, st) = sample_terrain(&r, &mosaic, &lattice, &carve);

        assert_eq!(st.samples, 4 * 4 * 65 * 65);
        assert_eq!(st.nodata, 0, "the mosaic covers the whole extent");
        assert!(st.land > 0 && st.land < st.samples, "some land, some sea");
        assert!(st.pad > 0, "the pad moved samples");

        // THE ISLAND PROPERTY: dry in the middle, wet at the edges, everywhere.
        assert!(data.height_at(DVec2::ZERO).unwrap() > 0.0);
        for (x, z) in [(500.0, 0.0), (-500.0, 0.0), (0.0, 500.0), (0.0, -500.0)] {
            let h = data.height_at(DVec2::new(x, z)).unwrap();
            assert!(h < 0.0, "({x},{z}) should be sea, got {h} m");
        }
        // The shore is at the waterline to within one sample of the ring.
        let shore = data.height_at(DVec2::new(300.0, 0.0)).unwrap();
        assert!(
            shore.abs() < 3.0,
            "the shore is at {shore} m, not the waterline"
        );
        // The shelf bottoms out at the recipe's own depth.
        assert!(
            (st.lo_m + r.sea.shelf_depth_m).abs() < 1.0,
            "the sea floor is {} m, the recipe says {}",
            st.lo_m,
            -r.sea.shelf_depth_m
        );
        // And the pad is at its datum, not at the source's 500 m.
        assert!(
            (data.height_at(DVec2::ZERO).unwrap() - 480.0).abs() < 1.0,
            "the pad datum did not take"
        );
        println!(
            "CARVE: {} samples, {} land, {} pad, {:.1}..{:.1} m, peak-on-land {:.1} m, \
             land {:.3} km2",
            st.samples,
            st.land,
            st.pad,
            st.lo_m,
            st.hi_m,
            st.peak_land_m,
            st.land_area_m2 / 1e6
        );
        assert!(
            st.land_area_m2 > 0.20e6 && st.land_area_m2 < 0.30e6,
            "{}",
            st.land_area_m2
        );
    }

    #[test]
    fn a_coarse_grid_reads_the_terrain_and_measures_its_slope() {
        let mut data = TerrainData::new(33, 2.0);
        // A plane at 10 % grade along +X.
        data.author_tile((0, 0), |x, _z| x * 0.1);
        data.author_tile((1, 0), |x, _z| x * 0.1);
        let ch = CoarseHeights::of(&data, DVec2::ZERO, DVec2::new(120.0, 60.0), 8.0);
        assert!(!ch.is_empty());
        assert!(ch.known[0]);
        let want = inf_math::portable::patan2_64(0.1, 1.0).to_degrees();
        for (i, j) in [(2usize, 2usize), (5, 3), (8, 4)] {
            let s = ch.slope_deg(i, j);
            assert!(
                (s - want).abs() < 0.2,
                "a 10 % grade is {want:.3} deg; the coarse grid says {s:.3}"
            );
        }
        assert!((f64::from(ch.at(5, 2)) - ch.position(5, 2).x * 0.1).abs() < 1e-3);
        // Off the terrain is unknown rather than zero.
        let off = CoarseHeights::of(
            &data,
            DVec2::new(5_000.0, 0.0),
            DVec2::new(5_100.0, 50.0),
            8.0,
        );
        assert!(off.known.iter().all(|k| !k));
    }

    #[test]
    fn an_asset_carries_the_origin_and_every_level_of_its_pyramid() {
        let mut data = TerrainData::new(17, 1.0);
        for tz in 0..8 {
            for tx in 0..8 {
                data.author_tile((tx, tz), |x, z| (x + z) * 0.01);
            }
        }
        let origin = DVec3::new(492_600.0, 0.0, 5_465_600.0);
        let opts = inf_terrain::PyramidOptions::default();
        let (asset, pyramid) = build_asset(&data, origin, opts).unwrap();
        let r = asset.reader();
        assert_eq!(
            r.origin(),
            origin,
            "the header must carry the survey's origin"
        );
        assert_eq!(r.tile_resolution(), 17);
        assert_eq!(r.pyramid_options(), Some(opts));
        let want: usize = 64 + pyramid.iter().map(|l| l.tiles.len()).sum::<usize>();
        assert_eq!(
            r.tile_count(),
            want,
            "every level-0 tile and every coarse one"
        );
        assert!(r.lod_levels() >= 2, "an 8x8 world has a ladder");
        // And it round-trips through the reader's own decode.
        let t = r.tile(inf_terrain::TileKey::lod0((0, 0))).unwrap().unwrap();
        assert_eq!(t.heights().len(), 17 * 17);

        // Two builds of one terrain are byte-identical — the property the whole
        // recipe rests on.
        let (again, _) = build_asset(&data, origin, opts).unwrap();
        assert_eq!(asset.as_bytes(), again.as_bytes());
    }

    #[test]
    fn stamping_biomes_writes_only_what_the_classifier_named() {
        let mut data = TerrainData::new(9, 1.0);
        data.author_tile((0, 0), |_, _| 0.0);
        data.author_tile((1, 0), |_, _| 0.0);
        // Half the world gets biome 3; the rest is left unassigned.
        let n = stamp_biomes(&mut data, |p| {
            if p.x < 8.0 {
                3
            } else {
                inf_terrain::UNASSIGNED_BIOME
            }
        });
        assert!(n > 0 && n < 2 * 81, "{n} of 162 written");
        assert_eq!(data.biome_at(DVec2::new(1.0, 1.0)), Some(3));
        assert_eq!(
            data.biome_at(DVec2::new(12.0, 1.0)),
            Some(inf_terrain::UNASSIGNED_BIOME)
        );
        assert!(!data.biomes_are_default(), "something was painted");
    }
}
