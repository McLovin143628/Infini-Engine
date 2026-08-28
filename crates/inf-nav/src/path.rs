//! The path: a polyline with its arc length pre-computed, and the four
//! questions anything walking one asks.
//!
//! # Why arc length and not "the next waypoint"
//!
//! A crowd agent's place has to be a **pure function of its route and a
//! scalar**, because that is what lets a `Far` agent cost nothing: no body, no
//! controller, no integration, no history — one multiply for the distance
//! travelled and one interpolation for where that is. Parameterizing on arc
//! length is what makes the scalar meaningful (a metre is a metre anywhere along
//! the chain, which a normalized `t` is not on an uneven polyline) and it is
//! what makes a *speed* in m/s mean what it says.
//!
//! # Shared, because a thousand agents walk one street
//!
//! [`NavPath`] is an `Arc` over its data. A population walking one route holds
//! one copy of it, cloning a `CrowdRoute` is a refcount bump, and comparing two
//! is a pointer compare that falls through to the contents. That is the same
//! argument NPC1b made for the joint palette, one system over.

use std::sync::Arc;

use glam::{DVec2, DVec3};

/// Distance below which two consecutive points are one point, metres.
///
/// A source polyline routinely repeats a vertex — a road layer digitises the
/// same junction twice, a room's centre coincides with its doorway on a
/// one-metre closet — and a zero-length segment would make the interpolation
/// divide by zero. A millimetre is below every geometry this engine authors and
/// above every rounding it does.
pub const COINCIDENT_M: f64 = 1.0e-3;

#[derive(Debug, PartialEq)]
struct PathData {
    points: Vec<DVec3>,
    /// `cum[i]` is the arc length from the start to `points[i]`; `cum[0] == 0`
    /// and `cum.len() == points.len()`.
    cum: Vec<f64>,
}

/// A polyline in world metres, with its arc length.
///
/// Always holds at least one point: [`NavPath::new`] over an empty sequence
/// answers a path standing at the origin, because a route that is a value cannot
/// have a constructor that fails.
#[derive(Clone, Debug, PartialEq)]
pub struct NavPath(Arc<PathData>);

/// Where a point falls on a path — [`NavPath::project`]'s answer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathProjection {
    /// Arc length of the nearest point on the chain, metres.
    pub s_m: f64,
    /// Distance from the queried point to it, metres.
    pub distance_m: f64,
    /// Index of the segment it fell on — `points[leg] → points[leg + 1]`.
    pub leg: usize,
}

impl NavPath {
    /// A path that stands still at `p`.
    pub fn single(p: DVec3) -> Self {
        Self(Arc::new(PathData {
            points: vec![p],
            cum: vec![0.0],
        }))
    }

    /// A path through `points`, with coincident and non-finite ones dropped.
    ///
    /// Dropping rather than refusing, for [`link`](crate::NavGraph::link)'s
    /// reason: every producer of a polyline in this tree already filters its
    /// own source, and a second filter here that could *fail* would be a second
    /// opinion about the same data.
    pub fn new(points: impl IntoIterator<Item = DVec3>) -> Self {
        let mut kept: Vec<DVec3> = Vec::new();
        for p in points {
            if !p.is_finite() {
                continue;
            }
            match kept.last() {
                Some(last) => {
                    let d = p - *last;
                    if (d.x * d.x + d.y * d.y + d.z * d.z).sqrt() > COINCIDENT_M {
                        kept.push(p);
                    }
                }
                None => kept.push(p),
            }
        }
        if kept.is_empty() {
            return Self::single(DVec3::ZERO);
        }
        let mut cum = Vec::with_capacity(kept.len());
        cum.push(0.0);
        let mut total = 0.0;
        for w in kept.windows(2) {
            let d = w[1] - w[0];
            total += (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
            cum.push(total);
        }
        Self(Arc::new(PathData { points: kept, cum }))
    }

    /// The points, in travel order. Never empty.
    pub fn points(&self) -> &[DVec3] {
        &self.0.points
    }

    /// Total arc length, metres. Zero for a stand.
    pub fn length_m(&self) -> f64 {
        *self.0.cum.last().unwrap_or(&0.0)
    }

    /// Whether this path goes nowhere — one point, or a chain of coincident ones
    /// that collapsed to one.
    pub fn is_stand(&self) -> bool {
        self.0.points.len() < 2
    }

    /// The segment index `s_m` falls on, `0 ..= points.len() - 2`.
    pub fn leg_at(&self, s_m: f64) -> usize {
        let n = self.0.points.len();
        if n < 2 {
            return 0;
        }
        match self.0.cum.binary_search_by(|c| c.total_cmp(&s_m)) {
            Ok(i) => i.min(n - 2),
            Err(i) => i.saturating_sub(1).min(n - 2),
        }
    }

    /// **Where `s_m` metres along the chain is**, clamped to both ends.
    ///
    /// For a two-point path this is exactly `from + (to - from) * (s / len)`,
    /// which is what NPC1a's `CrowdRoute::position_at` computed — the same
    /// three multiplies in the same order, so a population that walked a
    /// straight route before this wave walks it to the bit.
    pub fn position_at(&self, s_m: f64) -> DVec3 {
        let pts = &self.0.points;
        if pts.len() < 2 || !s_m.is_finite() {
            return pts[0];
        }
        let total = self.length_m();
        if s_m <= 0.0 {
            return pts[0];
        }
        if s_m >= total {
            return pts[pts.len() - 1];
        }
        let leg = self.leg_at(s_m);
        let (a, b) = (pts[leg], pts[leg + 1]);
        let run = self.0.cum[leg + 1] - self.0.cum[leg];
        if run <= 0.0 {
            return a;
        }
        a + (b - a) * ((s_m - self.0.cum[leg]) / run)
    }

    /// The unit direction of travel at `s_m`, or `+Z` for a stand.
    pub fn direction_at(&self, s_m: f64) -> DVec3 {
        let pts = &self.0.points;
        if pts.len() < 2 {
            return DVec3::Z;
        }
        let leg = self.leg_at(s_m.clamp(0.0, self.length_m()));
        let d = pts[leg + 1] - pts[leg];
        let len = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
        if len > 0.0 {
            d / len
        } else {
            DVec3::Z
        }
    }

    /// **The nearest point on the chain to `p`**, as an arc length.
    ///
    /// Walks every segment — `O(points)` — because a path this engine hands an
    /// agent is a street, a corridor or a road link, which is tens of points and
    /// not thousands, and a windowed search would need a *previous* answer,
    /// which is history a pure function must not carry.
    pub fn project(&self, p: DVec3) -> PathProjection {
        let pts = &self.0.points;
        if pts.len() < 2 {
            let d = p - pts[0];
            return PathProjection {
                s_m: 0.0,
                distance_m: (d.x * d.x + d.y * d.y + d.z * d.z).sqrt(),
                leg: 0,
            };
        }
        let mut best = PathProjection {
            s_m: 0.0,
            distance_m: f64::INFINITY,
            leg: 0,
        };
        for leg in 0..pts.len() - 1 {
            let (a, b) = (pts[leg], pts[leg + 1]);
            let d = b - a;
            let len2 = d.x * d.x + d.y * d.y + d.z * d.z;
            let t = if len2 > 0.0 {
                (((p - a).dot(d)) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let on = a + d * t;
            let e = p - on;
            let dist = (e.x * e.x + e.y * e.y + e.z * e.z).sqrt();
            if dist < best.distance_m {
                let run = self.0.cum[leg + 1] - self.0.cum[leg];
                best = PathProjection {
                    s_m: self.0.cum[leg] + run * t,
                    distance_m: dist,
                    leg,
                };
            }
        }
        best
    }

    /// The chain re-cut at a fixed spacing, ends and corners kept.
    ///
    /// The corners are kept because a resample that dropped them would round off
    /// a right-angled street grid; the spacing is what gives
    /// [`snapped`](Self::snapped) somewhere to ask the ground.
    pub fn resampled(&self, spacing_m: f64) -> Self {
        let pts = &self.0.points;
        if pts.len() < 2 || !spacing_m.is_finite() || spacing_m <= 0.0 {
            return self.clone();
        }
        let mut out: Vec<DVec3> = Vec::new();
        for leg in 0..pts.len() - 1 {
            let (a, b) = (pts[leg], pts[leg + 1]);
            let run = self.0.cum[leg + 1] - self.0.cum[leg];
            out.push(a);
            if run > spacing_m {
                // Integer step count, so the samples on a segment are a function
                // of its length alone — a `while s < run` accumulation would put
                // a different number of them on two segments that differ in the
                // last ulp.
                let n = (run / spacing_m) as usize;
                for k in 1..=n {
                    let t = k as f64 * spacing_m / run;
                    if t < 1.0 {
                        out.push(a + (b - a) * t);
                    }
                }
            }
        }
        out.push(pts[pts.len() - 1]);
        Self::new(out)
    }

    /// **The chain put on the ground**: every point's Y replaced by the height
    /// the sampler answers there, plus `lift_m`.
    ///
    /// A sampler that answers `None` — a point over a hole, or over a terrain
    /// tile that is not resident — leaves that point's authored Y alone. That is
    /// the honest fallback rather than a hidden one: a road spine already
    /// carries the height it was *surveyed* at and a building's floor already
    /// carries `floor_y`, so the authored Y is a real answer and not a zero.
    ///
    /// Snapping happens **once, where the route is built**, and never per step.
    /// A per-step ground query would make an agent's Y a function of terrain
    /// residency, which is streaming state, and a position that depends on what
    /// has paged in is a position two hosts can disagree about.
    pub fn snapped(&self, lift_m: f64, mut height: impl FnMut(DVec2) -> Option<f64>) -> Self {
        let out: Vec<DVec3> = self
            .0
            .points
            .iter()
            .map(|p| match height(DVec2::new(p.x, p.z)) {
                Some(y) if y.is_finite() => DVec3::new(p.x, y + lift_m, p.z),
                _ => *p,
            })
            .collect();
        Self::new(out)
    }

    /// The chain reversed — the same street walked the other way.
    pub fn reversed(&self) -> Self {
        let mut pts = self.0.points.clone();
        pts.reverse();
        Self::new(pts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, z: f64) -> DVec3 {
        DVec3::new(x, 0.0, z)
    }

    #[test]
    fn a_path_of_nothing_is_a_stand_at_the_origin() {
        let path = NavPath::new(std::iter::empty());
        assert!(path.is_stand());
        assert_eq!(path.length_m(), 0.0);
        assert_eq!(path.position_at(17.0), DVec3::ZERO);
    }

    #[test]
    fn coincident_and_non_finite_points_are_dropped() {
        let path = NavPath::new([
            p(0.0, 0.0),
            p(0.0, 0.0),
            DVec3::new(f64::NAN, 0.0, 0.0),
            p(10.0, 0.0),
        ]);
        assert_eq!(path.points().len(), 2);
        assert_eq!(path.length_m(), 10.0);
    }

    /// **The bit-identity that keeps NPC1a's population where it was**: a
    /// two-point path interpolates exactly the way `CrowdRoute::position_at`
    /// did before this wave.
    #[test]
    fn a_two_point_path_interpolates_exactly_as_the_npc1a_route_did() {
        let (from, to) = (p(3.0, -7.0), DVec3::new(41.0, 12.0, 5.5));
        let path = NavPath::new([from, to]);
        let d = to - from;
        let len = (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
        for k in 0..64 {
            let s = len * k as f64 / 63.0;
            let want = from + d * (s / len);
            let got = path.position_at(s);
            assert_eq!(got.x.to_bits(), want.x.to_bits(), "x at s = {s}");
            assert_eq!(got.y.to_bits(), want.y.to_bits(), "y at s = {s}");
            assert_eq!(got.z.to_bits(), want.z.to_bits(), "z at s = {s}");
        }
    }

    #[test]
    fn position_clamps_at_both_ends_and_walks_the_corners() {
        let path = NavPath::new([p(0.0, 0.0), p(10.0, 0.0), p(10.0, 10.0)]);
        assert_eq!(path.length_m(), 20.0);
        assert_eq!(path.position_at(-5.0), p(0.0, 0.0));
        assert_eq!(path.position_at(10.0), p(10.0, 0.0));
        assert_eq!(path.position_at(15.0), p(10.0, 5.0));
        assert_eq!(path.position_at(1e9), p(10.0, 10.0));
    }

    #[test]
    fn project_finds_the_arc_length_of_the_nearest_point() {
        let path = NavPath::new([p(0.0, 0.0), p(10.0, 0.0), p(10.0, 10.0)]);
        let pr = path.project(p(4.0, 3.0));
        assert_eq!(pr.leg, 0);
        assert!((pr.s_m - 4.0).abs() < 1e-9, "{}", pr.s_m);
        assert!((pr.distance_m - 3.0).abs() < 1e-9);
        let pr = path.project(p(13.0, 6.0));
        assert_eq!(pr.leg, 1);
        assert!((pr.s_m - 16.0).abs() < 1e-9, "{}", pr.s_m);
        // …and a projection round-trips through `position_at`.
        let back = path.position_at(pr.s_m);
        assert!((back - p(10.0, 6.0)).length() < 1e-9);
    }

    #[test]
    fn a_resample_keeps_the_corners_and_hits_the_spacing() {
        let path = NavPath::new([p(0.0, 0.0), p(10.0, 0.0), p(10.0, 10.0)]).resampled(2.5);
        assert!(path.points().contains(&p(10.0, 0.0)), "the corner is gone");
        assert_eq!(path.points().len(), 9);
        assert!((path.length_m() - 20.0).abs() < 1e-9);
    }

    /// A sampler that has nothing to say leaves the authored height alone — a
    /// road spine's surveyed Y is an answer, and zero is not.
    #[test]
    fn a_snap_over_a_hole_keeps_the_authored_height() {
        let path = NavPath::new([DVec3::new(0.0, 55.0, 0.0), DVec3::new(10.0, 60.0, 0.0)]);
        let snapped = path.snapped(0.1, |xz| if xz.x < 5.0 { Some(7.0) } else { None });
        assert_eq!(snapped.points()[0], DVec3::new(0.0, 7.1, 0.0));
        assert_eq!(snapped.points()[1], DVec3::new(10.0, 60.0, 0.0));
    }

    #[test]
    fn a_shared_path_is_one_allocation() {
        let a = NavPath::new([p(0.0, 0.0), p(10.0, 0.0)]);
        let b = a.clone();
        assert!(Arc::ptr_eq(&a.0, &b.0));
        assert_eq!(a, b);
    }
}
