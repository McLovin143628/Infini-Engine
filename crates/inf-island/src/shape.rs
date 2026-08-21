//! The designed shape: the coastline, the sea shelf, the beach, and the site
//! flats.
//!
//! # This is the step that makes it an island
//!
//! The source is a piece of the North Shore. It is not an island and no amount
//! of sampling will make it one. The coastline is **authored** — one closed
//! polygon in the anchor's own CRS, committed beside the recipe — and this
//! module is what turns "here is some real ground" into "here is a landmass with
//! a shore, a shelf and a beach".
//!
//! # Why a field and not a per-sample polygon test
//!
//! A signed-distance query against an N-vertex polygon is `O(N)`, and the island
//! has 51 million samples. At the committed coastline's 200-odd edges that is
//! ten billion edge tests — minutes of work for a quantity that varies over
//! hundreds of metres.
//!
//! [`Field`] rasterizes it once onto a coarse lattice and interpolates. The
//! lattice pitch is **derived from the narrowest feature the field is used for**
//! (the beach band), not chosen: a field coarser than the thing it shapes puts
//! the shoreline in the wrong place, and one finer costs time for a resolution
//! nothing reads. [`Coastline::field_pitch_m`] carries that arithmetic and
//! `the_field_pitch_follows_the_narrowest_feature` is what says so.
//!
//! # Portability
//!
//! Everything here is `+ - * /`, comparisons and `sqrt`. `sqrt` is exact in
//! IEEE-754 and therefore bit-identical everywhere; the transcendental family is
//! not, and none of it appears (`tests/portable_math_law.rs`). That is what lets
//! the carved ground — and so the streams and lakes derived from it — be the
//! same on any machine that starts from the same sampled heights.

use glam::DVec2;

/// The classic cubic smoothstep on `[0, 1]`, clamped.
///
/// Polynomial on purpose: it is the only ease this crate uses, and a
/// trigonometric one would put the carve on the wrong side of the portability
/// law for a shape nobody could tell apart.
#[inline]
pub fn smooth01(t: f64) -> f64 {
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    t * t * (3.0 - 2.0 * t)
}

/// A coarse scalar field over the world's XZ plane, read by bilinear
/// interpolation.
///
/// Row-major, `nx × nz` samples, sample `(i, j)` at
/// `(min.x + i·pitch, min.y + j·pitch)`. `min`/`max` are world metres; `y` of a
/// [`DVec2`] here is world **Z**.
#[derive(Clone, Debug, PartialEq)]
pub struct Field {
    min: DVec2,
    pitch: f64,
    nx: usize,
    nz: usize,
    v: Vec<f64>,
}

impl Field {
    /// An all-zero field covering `[min, max]` at `pitch` metres.
    pub fn new(min: DVec2, max: DVec2, pitch: f64) -> Self {
        let pitch = if pitch.is_finite() && pitch > 0.0 {
            pitch
        } else {
            1.0
        };
        let span = (max - min).max(DVec2::ZERO);
        let nx = (span.x / pitch).ceil() as usize + 2;
        let nz = (span.y / pitch).ceil() as usize + 2;
        Self {
            min,
            pitch,
            nx,
            nz,
            v: vec![0.0; nx * nz],
        }
    }

    /// Fill every sample from a pure function of its world position.
    pub fn fill(&mut self, mut f: impl FnMut(DVec2) -> f64) {
        for j in 0..self.nz {
            for i in 0..self.nx {
                let p = self.position(i, j);
                self.v[j * self.nx + i] = f(p);
            }
        }
    }

    /// The world position of sample `(i, j)`.
    pub fn position(&self, i: usize, j: usize) -> DVec2 {
        DVec2::new(
            self.min.x + i as f64 * self.pitch,
            self.min.y + j as f64 * self.pitch,
        )
    }

    /// `(nx, nz)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.nx, self.nz)
    }

    /// The lattice pitch in metres.
    pub fn pitch_m(&self) -> f64 {
        self.pitch
    }

    /// Write one sample. Out-of-range indices are ignored rather than panicking:
    /// a filler that walks its own `dims()` cannot be out of range, and one that
    /// does not should not take the process down mid-build.
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        if i < self.nx && j < self.nz {
            self.v[j * self.nx + i] = v;
        }
    }

    /// The stored sample, clamped to the lattice at the edges.
    #[inline]
    pub fn sample(&self, i: i64, j: i64) -> f64 {
        let i = i.clamp(0, self.nx as i64 - 1) as usize;
        let j = j.clamp(0, self.nz as i64 - 1) as usize;
        self.v[j * self.nx + i]
    }

    /// Bilinear value at a world position. Clamped at the edges rather than
    /// refused, because the carve asks past the world's own border by design.
    #[inline]
    pub fn at(&self, p: DVec2) -> f64 {
        if !(p.x.is_finite() && p.y.is_finite()) {
            return 0.0;
        }
        let fx = (p.x - self.min.x) / self.pitch;
        let fz = (p.y - self.min.y) / self.pitch;
        let i0 = fx.floor();
        let j0 = fz.floor();
        let tx = fx - i0;
        let tz = fz - j0;
        let (i0, j0) = (i0 as i64, j0 as i64);
        let a = self.sample(i0, j0);
        let b = self.sample(i0 + 1, j0);
        let c = self.sample(i0, j0 + 1);
        let d = self.sample(i0 + 1, j0 + 1);
        let top = a + (b - a) * tx;
        let bot = c + (d - c) * tx;
        top + (bot - top) * tz
    }
}

/// The designed shore: rings in world XZ, plus the distance field derived from
/// them.
#[derive(Clone, Debug)]
pub struct Coastline {
    /// Closed rings, world XZ, **not** carrying a repeated closing vertex.
    rings: Vec<Vec<DVec2>>,
    sdf: Field,
}

/// What the carve did, in numbers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShapeStats {
    /// Samples inside the coastline.
    pub land_samples: u64,
    /// Samples outside it.
    pub sea_samples: u64,
    /// Samples inside the coastline whose carved ground is still below sea level
    /// — a real inlet the design kept, or a mistake the design made. Reported
    /// either way, because the two look identical from inside the loop.
    pub submerged_land_samples: u64,
    /// Samples inside the beach band.
    pub beach_samples: u64,
    /// Land samples that had no source elevation at all.
    pub nodata_samples: u64,
    /// Highest carved sample, metres.
    pub peak_m: f64,
    /// Lowest carved sample, metres.
    pub floor_m: f64,
}

impl Coastline {
    /// Build a coastline from world-space rings.
    ///
    /// `pitch_m` is the distance-field lattice; use
    /// [`Coastline::field_pitch_m`] unless you have a reason not to.
    pub fn new(rings: Vec<Vec<DVec2>>, min: DVec2, max: DVec2, pitch_m: f64) -> Self {
        let rings: Vec<Vec<DVec2>> = rings.into_iter().filter(|r| r.len() >= 3).collect();
        let mut sdf = Field::new(min, max, pitch_m);
        let r = &rings;
        sdf.fill(|p| signed_distance(r, p));
        Self { rings, sdf }
    }

    /// The lattice pitch a coastline should be rasterized at, given the
    /// narrowest feature it shapes.
    ///
    /// A quarter of the beach band, floored at 2 m and capped at 8 m. The floor
    /// is because a metre-pitch field over a 50 km² world is 2.5 GB; the cap is
    /// because past 8 m the shelf ramp — hundreds of metres wide — is the only
    /// thing left and it does not need the samples.
    pub fn field_pitch_m(beach_width_m: f64) -> f64 {
        if !beach_width_m.is_finite() || beach_width_m <= 0.0 {
            return 8.0;
        }
        (beach_width_m * 0.25).clamp(2.0, 8.0)
    }

    /// Signed distance to the shore in metres, **positive inland**.
    #[inline]
    pub fn distance_at(&self, p: DVec2) -> f64 {
        self.sdf.at(p)
    }

    /// `true` when a world position is on land.
    #[inline]
    pub fn is_land(&self, p: DVec2) -> bool {
        self.distance_at(p) > 0.0
    }

    /// The rings, for a report or a re-export.
    pub fn rings(&self) -> &[Vec<DVec2>] {
        &self.rings
    }

    /// The distance field, for a caller that wants its lattice.
    pub fn field(&self) -> &Field {
        &self.sdf
    }

    /// The shore's total length in metres — the coastline number the ledger
    /// prints.
    pub fn perimeter_m(&self) -> f64 {
        let mut total = 0.0;
        for r in &self.rings {
            for i in 0..r.len() {
                total += (r[(i + 1) % r.len()] - r[i]).length();
            }
        }
        total
    }

    /// The land area the rings enclose, in square metres.
    ///
    /// The shoelace over every ring, summed with sign, so a ring wound the other
    /// way subtracts — which is how a lagoon inside the island is spelled.
    pub fn area_m2(&self) -> f64 {
        let mut a = 0.0;
        for r in &self.rings {
            let mut s = 0.0;
            for i in 0..r.len() {
                let p = r[i];
                let q = r[(i + 1) % r.len()];
                s += p.x * q.y - q.x * p.y;
            }
            a += s * 0.5;
        }
        a.abs()
    }
}

/// Signed distance from `p` to a set of closed rings, positive inside.
///
/// Inside/outside by the half-open crossing rule — the same one
/// `inf_terrain::BiomeFill` uses, so a sample on a shared edge belongs to
/// exactly one ring and the shore has no seam.
fn signed_distance(rings: &[Vec<DVec2>], p: DVec2) -> f64 {
    let mut best = f64::INFINITY;
    let mut inside = false;
    for r in rings {
        for i in 0..r.len() {
            let a = r[i];
            let b = r[(i + 1) % r.len()];
            best = best.min(distance_to_segment(p, a, b));
            if (a.y > p.y) != (b.y > p.y) {
                let t = (p.y - a.y) / (b.y - a.y);
                if p.x < a.x + t * (b.x - a.x) {
                    inside = !inside;
                }
            }
        }
    }
    if !best.is_finite() {
        return 0.0;
    }
    if inside {
        best
    } else {
        -best
    }
}

#[inline]
fn distance_to_segment(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 <= 0.0 {
        return (p - a).length();
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    (p - (a + ab * t)).length()
}

/// The carve, as a pure function of one sample.
///
/// * `source` — the sampled elevation, or `None` where the source had nothing.
/// * `d` — signed distance to the shore, positive inland.
///
/// The two branches, and the reason each is what it is:
///
/// * **Outside** the shore the source is discarded entirely and the sea floor is
///   a shelf: sea level at the shore, falling to `shelf_depth_m` over
///   `shelf_width_m`. Discarding rather than blending is the point — the ground
///   out there is a real mountain that the design says is not there.
/// * **Inside**, the ground rises from exactly sea level at the shore to the
///   source's own elevation over `beach_width_m`. One rule gives both a beach
///   and a cliff: where the land behind the shore is low the ramp is gentle sand,
///   and where it is a 200 m bluff the same ramp is a bluff. A shore under a
///   mountain gets a cliff, which is what a shore under a mountain is.
///
/// `None` source inland becomes sea level — the **nodata → ocean** policy, taken
/// here, once, where the extent is known, exactly as
/// `inf_gis::terrarium`'s own header says it must be.
#[inline]
pub fn carve_sample(
    source: Option<f64>,
    d: f64,
    sea_level_m: f64,
    shelf_depth_m: f64,
    shelf_width_m: f64,
    beach_width_m: f64,
) -> f64 {
    if d <= 0.0 {
        let t = if shelf_width_m > 0.0 {
            smooth01(-d / shelf_width_m)
        } else {
            1.0
        };
        return sea_level_m - shelf_depth_m * t;
    }
    let h = match source {
        Some(v) if v.is_finite() => v,
        _ => sea_level_m,
    };
    if beach_width_m <= 0.0 {
        return h;
    }
    let t = smooth01(d / beach_width_m);
    sea_level_m + (h - sea_level_m) * t
}

/// One vertex of a polyline, carrying the elevation the polyline wants there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vertex3 {
    /// World XZ.
    pub xz: DVec2,
    /// The elevation this polyline asks the ground for.
    pub y: f64,
}

/// A bucketed index over polyline segments, for "what is the nearest centreline
/// and what height does it want here".
///
/// # Why an index at all
///
/// The road corridor and the stream channels are both a *distance to the nearest
/// segment* query, evaluated over a field of millions of cells against thousands
/// of segments. Brute force is the product of those two numbers and it is
/// billions. The bucket grid makes each query `O(segments within one bucket
/// ring)`, which on a road network is a handful.
///
/// The bucket pitch is the **query reach**, so a query never has to look further
/// than the 3 × 3 ring around its own bucket. That is not an optimisation
/// choice — it is what makes the ring sufficient, and a pitch smaller than the
/// reach would silently miss segments.
#[derive(Clone, Debug)]
pub struct SegmentIndex {
    /// `(a, b)` endpoints, world XZ + the wanted elevation at each end.
    segs: Vec<(Vertex3, Vertex3)>,
    /// Per segment, the polyline it came from — so a caller can tell two rivers
    /// apart at a confluence.
    owner: Vec<usize>,
    buckets: std::collections::BTreeMap<(i64, i64), Vec<u32>>,
    pitch: f64,
    reach: f64,
}

/// What the nearest segment says at a position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NearestSegment {
    /// Distance to the centreline, metres.
    pub distance_m: f64,
    /// The elevation the centreline wants at the closest point.
    pub height_m: f64,
    /// Which polyline it belongs to.
    pub owner: usize,
}

impl SegmentIndex {
    /// Build an index over polylines, answering out to `reach` metres.
    pub fn new(lines: &[Vec<Vertex3>], reach: f64) -> Self {
        let reach = if reach.is_finite() && reach > 0.0 {
            reach
        } else {
            1.0
        };
        let mut segs = Vec::new();
        let mut owner = Vec::new();
        for (li, line) in lines.iter().enumerate() {
            for w in line.windows(2) {
                if w[0].xz.distance_squared(w[1].xz) <= 0.0 {
                    continue;
                }
                segs.push((w[0], w[1]));
                owner.push(li);
            }
        }
        let mut buckets: std::collections::BTreeMap<(i64, i64), Vec<u32>> = Default::default();
        for (i, (a, b)) in segs.iter().enumerate() {
            let (x0, x1) = (a.xz.x.min(b.xz.x) - reach, a.xz.x.max(b.xz.x) + reach);
            let (z0, z1) = (a.xz.y.min(b.xz.y) - reach, a.xz.y.max(b.xz.y) + reach);
            let (bx0, bx1) = ((x0 / reach).floor() as i64, (x1 / reach).floor() as i64);
            let (bz0, bz1) = ((z0 / reach).floor() as i64, (z1 / reach).floor() as i64);
            for bz in bz0..=bz1 {
                for bx in bx0..=bx1 {
                    buckets.entry((bx, bz)).or_default().push(i as u32);
                }
            }
        }
        Self {
            segs,
            owner,
            buckets,
            pitch: reach,
            reach,
        }
    }

    /// How many segments the index holds.
    pub fn len(&self) -> usize {
        self.segs.len()
    }

    /// `true` when there is nothing to be near.
    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }

    /// The index's reach in metres — past this it always answers `None`.
    pub fn reach_m(&self) -> f64 {
        self.reach
    }

    /// The nearest segment within the reach, or `None`.
    pub fn nearest(&self, p: DVec2) -> Option<NearestSegment> {
        if !(p.x.is_finite() && p.y.is_finite()) {
            return None;
        }
        let bx = (p.x / self.pitch).floor() as i64;
        let bz = (p.y / self.pitch).floor() as i64;
        let mut best: Option<NearestSegment> = None;
        for dz in -1..=1 {
            for dx in -1..=1 {
                let Some(ids) = self.buckets.get(&(bx + dx, bz + dz)) else {
                    continue;
                };
                for &i in ids {
                    let (a, b) = self.segs[i as usize];
                    let ab = b.xz - a.xz;
                    let len2 = ab.length_squared();
                    let t = if len2 > 0.0 {
                        ((p - a.xz).dot(ab) / len2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let d = (p - (a.xz + ab * t)).length();
                    if d > self.reach {
                        continue;
                    }
                    if best.is_none_or(|n| d < n.distance_m) {
                        best = Some(NearestSegment {
                            distance_m: d,
                            height_m: a.y + (b.y - a.y) * t,
                            owner: self.owner[i as usize],
                        });
                    }
                }
            }
        }
        best
    }
}

/// Flatten the ground toward a target over a radius — the site pad.
///
/// A city site on a 12 % slope is a city site nobody can build on, so the recipe
/// names a radius and the ground inside it is eased toward the site's own datum.
/// The ease is `smooth01` from the centre out, so the pad meets the hillside with
/// a matched gradient instead of a step — a step would be a wall around every
/// town.
#[inline]
pub fn flatten_sample(h: f64, dist_m: f64, radius_m: f64, target_m: f64) -> f64 {
    if radius_m <= 0.0 || !dist_m.is_finite() || dist_m >= radius_m {
        return h;
    }
    // 1 at the centre, 0 at the rim.
    let w = 1.0 - smooth01(dist_m / radius_m);
    target_m + (h - target_m) * (1.0 - w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(half: f64) -> Vec<Vec<DVec2>> {
        vec![vec![
            DVec2::new(-half, -half),
            DVec2::new(half, -half),
            DVec2::new(half, half),
            DVec2::new(-half, half),
        ]]
    }

    #[test]
    fn smoothstep_is_the_polynomial_one_and_clamps() {
        assert_eq!(smooth01(0.0), 0.0);
        assert_eq!(smooth01(1.0), 1.0);
        assert_eq!(smooth01(0.5), 0.5);
        assert_eq!(smooth01(-3.0), 0.0);
        assert_eq!(smooth01(7.0), 1.0);
        assert_eq!(smooth01(f64::NAN), 0.0);
        // Its derivative vanishes at both ends — the property that makes the
        // shelf meet the shore without a crease.
        let e = 1e-6;
        assert!(smooth01(e) < e * 1e-3);
        assert!(1.0 - smooth01(1.0 - e) < e * 1e-3);
    }

    #[test]
    fn a_field_interpolates_a_plane_exactly() {
        let mut f = Field::new(DVec2::splat(-100.0), DVec2::splat(100.0), 10.0);
        f.fill(|p| 3.0 * p.x - 2.0 * p.y + 5.0);
        for (x, z) in [(0.0, 0.0), (13.7, -44.2), (-99.0, 99.0), (5.5, 5.5)] {
            let want = 3.0 * x - 2.0 * z + 5.0;
            let got = f.at(DVec2::new(x, z));
            assert!(
                (got - want).abs() < 1e-9,
                "bilinear must be exact on a plane: ({x},{z}) -> {got}, want {want}"
            );
        }
        assert!((f.at(DVec2::new(1e9, 0.0)) - f.at(DVec2::new(1e9, 0.0))).abs() < 1e-12);
        assert_eq!(f.at(DVec2::new(f64::NAN, 0.0)), 0.0);
        assert_eq!(f.pitch_m(), 10.0);
    }

    /// The signed distance is signed, and it is a DISTANCE — measured against
    /// hand-computable positions on a square.
    #[test]
    fn the_signed_distance_is_positive_inland_and_metric() {
        let r = square(100.0);
        assert_eq!(
            signed_distance(&r, DVec2::ZERO),
            100.0,
            "centre of a 200 m square"
        );
        assert_eq!(signed_distance(&r, DVec2::new(90.0, 0.0)), 10.0);
        assert_eq!(signed_distance(&r, DVec2::new(110.0, 0.0)), -10.0);
        assert_eq!(
            signed_distance(&r, DVec2::new(100.0, 0.0)),
            0.0,
            "on the shore"
        );
        // A corner: 3-4-5.
        let d = signed_distance(&r, DVec2::new(103.0, 104.0));
        assert!((d + 5.0).abs() < 1e-9, "outside a corner by 5 m, got {d}");
    }

    #[test]
    fn the_field_pitch_follows_the_narrowest_feature() {
        assert_eq!(Coastline::field_pitch_m(30.0), 7.5);
        assert_eq!(Coastline::field_pitch_m(4.0), 2.0, "floored");
        assert_eq!(Coastline::field_pitch_m(1_000.0), 8.0, "capped");
        assert_eq!(Coastline::field_pitch_m(0.0), 8.0);
        assert_eq!(Coastline::field_pitch_m(f64::NAN), 8.0);

        // And the floor is a memory decision, priced: a 1 m field over the
        // shipped island is what it is refusing.
        let island_m = 7_168.0_f64;
        let cells_at_1m = (island_m * island_m) / 1.0;
        let bytes = cells_at_1m * 8.0;
        assert!(
            bytes > 4.0e8,
            "a 1 m field over the island is {bytes} bytes — if this is small the \
             floor is guarding nothing"
        );
        println!(
            "FIELD PITCH: 1 m over the island would be {:.2} GB",
            bytes / 1e9
        );
    }

    #[test]
    fn a_coastline_reports_its_own_shore_and_area() {
        let c = Coastline::new(
            square(100.0),
            DVec2::splat(-200.0),
            DVec2::splat(200.0),
            4.0,
        );
        assert_eq!(c.rings().len(), 1);
        assert_eq!(c.perimeter_m(), 800.0);
        assert_eq!(c.area_m2(), 40_000.0);
        assert!(c.is_land(DVec2::ZERO));
        assert!(!c.is_land(DVec2::new(150.0, 0.0)));
        // The field agrees with the exact query where the lattice lands on it.
        assert!((c.distance_at(DVec2::ZERO) - 100.0).abs() < 1e-9);
        // A degenerate ring is dropped rather than dividing by zero.
        let empty = Coastline::new(
            vec![vec![DVec2::ZERO, DVec2::X]],
            DVec2::splat(-1.0),
            DVec2::splat(1.0),
            1.0,
        );
        assert!(empty.rings().is_empty());
        assert_eq!(empty.perimeter_m(), 0.0);
    }

    /// The carve's two branches, and the property that makes it an island: the
    /// sea floor is BELOW sea level everywhere outside, whatever the source said.
    #[test]
    fn the_carve_sinks_the_sea_and_lands_the_shore_at_the_waterline() {
        let (sea, depth, shelf, beach) = (0.0, 60.0, 400.0, 30.0);
        // Outside: the source is a 900 m mountain and it is gone.
        assert_eq!(
            carve_sample(Some(900.0), 0.0, sea, depth, shelf, beach),
            0.0
        );
        let at100 = carve_sample(Some(900.0), -100.0, sea, depth, shelf, beach);
        assert!(
            at100 < 0.0 && at100 > -depth,
            "100 m out is on the ramp: {at100}"
        );
        assert_eq!(
            carve_sample(Some(900.0), -shelf, sea, depth, shelf, beach),
            -depth,
            "past the shelf width it is flat sea floor"
        );
        assert_eq!(
            carve_sample(Some(900.0), -10_000.0, sea, depth, shelf, beach),
            -depth
        );

        // Inside: exactly sea level at the shore, the source's own height past
        // the beach, monotone between.
        assert!(
            carve_sample(Some(40.0), 1e-12, sea, depth, shelf, beach).abs() < 1e-12,
            "a sample a picometre inland must be at the waterline"
        );
        assert!((carve_sample(Some(40.0), beach, sea, depth, shelf, beach) - 40.0).abs() < 1e-12);
        assert!((carve_sample(Some(40.0), 500.0, sea, depth, shelf, beach) - 40.0).abs() < 1e-12);
        let mut prev = -1.0;
        for k in 0..=30 {
            let h = carve_sample(Some(40.0), f64::from(k), sea, depth, shelf, beach);
            assert!(h >= prev, "the beach must not fall going inland");
            prev = h;
        }

        // ONE rule, two outcomes: a low hinterland is sand, a high one is a bluff.
        let sand = carve_sample(Some(4.0), 15.0, sea, depth, shelf, beach);
        let bluff = carve_sample(Some(200.0), 15.0, sea, depth, shelf, beach);
        assert!(sand < 3.0 && bluff > 90.0, "sand {sand}, bluff {bluff}");
        println!(
            "SHORE at 15 m inland: 4 m hinterland -> {sand:.2} m (sand), \
             200 m hinterland -> {bluff:.2} m (cliff)"
        );

        // nodata inland becomes ocean, not a flat plain at whatever zero means.
        assert_eq!(carve_sample(None, 500.0, sea, depth, shelf, beach), sea);
        assert_eq!(
            carve_sample(Some(f64::NAN), 500.0, sea, depth, shelf, beach),
            sea
        );

        // A non-zero sea level moves everything with it.
        assert_eq!(
            carve_sample(Some(900.0), -shelf, 12.0, depth, shelf, beach),
            12.0 - depth
        );
        assert_eq!(
            carve_sample(Some(40.0), 1e-12, 12.0, depth, shelf, beach),
            12.0
        );
    }

    /// The bucket index answers what brute force answers, and its reach is a
    /// refusal rather than a preference.
    ///
    /// Un-fix mutation: shrink the ring from 3 × 3 to the query's own bucket and
    /// the agreement below breaks on every position near a bucket edge.
    #[test]
    fn the_segment_index_agrees_with_brute_force_inside_its_reach() {
        // Two polylines: a diagonal and a zigzag, with heights that vary along
        // them so a wrong segment gives a wrong HEIGHT and not just a wrong
        // distance.
        let a: Vec<Vertex3> = (0..40)
            .map(|k| Vertex3 {
                xz: DVec2::new(f64::from(k) * 10.0, f64::from(k) * 7.0),
                y: f64::from(k) * 3.0,
            })
            .collect();
        let b: Vec<Vertex3> = (0..25)
            .map(|k| Vertex3 {
                xz: DVec2::new(400.0 - f64::from(k) * 9.0, f64::from(k) * 11.0),
                y: 500.0 - f64::from(k) * 4.0,
            })
            .collect();
        let reach = 25.0;
        let idx = SegmentIndex::new(&[a.clone(), b.clone()], reach);
        assert_eq!(idx.len(), 39 + 24);
        assert!(!idx.is_empty());
        assert_eq!(idx.reach_m(), reach);

        let brute = |p: DVec2| -> Option<NearestSegment> {
            let mut best: Option<NearestSegment> = None;
            for (li, line) in [&a, &b].iter().enumerate() {
                for w in line.windows(2) {
                    let ab = w[1].xz - w[0].xz;
                    let t = ((p - w[0].xz).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
                    let d = (p - (w[0].xz + ab * t)).length();
                    if d <= reach && best.is_none_or(|n| d < n.distance_m) {
                        best = Some(NearestSegment {
                            distance_m: d,
                            height_m: w[0].y + (w[1].y - w[0].y) * t,
                            owner: li,
                        });
                    }
                }
            }
            best
        };

        let mut hits = 0;
        let mut misses = 0;
        // A deterministic sweep — no RNG, so a failure is reproducible.
        for i in 0..90 {
            for j in 0..90 {
                let p = DVec2::new(f64::from(i) * 5.0 - 20.0, f64::from(j) * 4.0 - 20.0);
                match (idx.nearest(p), brute(p)) {
                    (None, None) => misses += 1,
                    (Some(x), Some(y)) => {
                        hits += 1;
                        assert!(
                            (x.distance_m - y.distance_m).abs() < 1e-9,
                            "at {p:?}: {} vs {}",
                            x.distance_m,
                            y.distance_m
                        );
                        assert!((x.height_m - y.height_m).abs() < 1e-9, "height at {p:?}");
                        assert_eq!(x.owner, y.owner, "owner at {p:?}");
                    }
                    (l, r) => panic!("disagreement at {p:?}: {l:?} vs {r:?}"),
                }
            }
        }
        println!("SEGMENT INDEX: {hits} hits, {misses} out-of-reach, all agreeing");
        assert!(
            hits > 500,
            "only {hits} positions were within reach — the sweep is vacuous"
        );
        assert!(
            misses > 500,
            "only {misses} were outside — the reach is not being tested"
        );

        // Degenerate input does not divide by zero.
        let dup = SegmentIndex::new(
            &[vec![
                Vertex3 {
                    xz: DVec2::ZERO,
                    y: 0.0,
                },
                Vertex3 {
                    xz: DVec2::ZERO,
                    y: 9.0,
                },
            ]],
            10.0,
        );
        assert!(dup.is_empty());
        assert_eq!(dup.nearest(DVec2::ZERO), None);
        assert_eq!(idx.nearest(DVec2::new(f64::NAN, 0.0)), None);
    }

    #[test]
    fn a_site_pad_flattens_to_its_datum_and_meets_the_hill_without_a_step() {
        let (r, target) = (200.0, 50.0);
        assert_eq!(
            flatten_sample(300.0, 0.0, r, target),
            target,
            "dead centre is the datum"
        );
        assert_eq!(
            flatten_sample(300.0, r, r, target),
            300.0,
            "at the rim, untouched"
        );
        assert_eq!(flatten_sample(300.0, 1e9, r, target), 300.0);
        assert_eq!(
            flatten_sample(300.0, 0.0, 0.0, target),
            300.0,
            "no radius, no pad"
        );
        // Continuous at the rim: the step across the boundary is under a
        // millimetre, so the pad has no wall around it.
        let inside = flatten_sample(300.0, r - 1e-6, r, target);
        assert!(
            (inside - 300.0).abs() < 1e-3,
            "a step of {} m at the rim",
            inside - 300.0
        );
        // Monotone from datum to hillside.
        let mut prev = target - 1.0;
        for k in 0..=200 {
            let h = flatten_sample(300.0, f64::from(k), r, target);
            assert!(h >= prev - 1e-9, "the pad must rise outward");
            prev = h;
        }
    }
}
