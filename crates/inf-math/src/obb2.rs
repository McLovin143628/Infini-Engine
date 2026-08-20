//! Planar convex hulls and **oriented minimum-area rectangles** (IB-6).
//!
//! # Why the engine needed this
//!
//! Every footprint in the tree collapsed to an axis-aligned box. A GIS building
//! footprint, a surveyed parcel, a hand-drawn lot: each became
//! `Rect2 { min, max }` over its own XZ bounds, so a city whose streets are not
//! on a compass grid — Vancouver's West End and downtown are both rotated — sat
//! on parcels that were not its own. The certification calls it IB-6 and the
//! Wave-G memo calls it "a deep change [that] deserves its own sub-phase".
//!
//! # Deterministic, owned, and trig-free
//!
//! * **Owned**, because the workspace had no 2-D hull and no oriented box at
//!   all: `inf-mesh`'s hull is 3-D triangle faces for fracture, `inf-anim`'s is
//!   the boundary of a Delaunay triangulation for blend spaces, and neither is
//!   the shape a lot wants.
//! * **Trig-free**, because a lot's orientation reaches committed content
//!   through the grammar's placements, and the P14 law bans `atan2`/`sin`/`cos`
//!   there — they are not bit-portable. Everything here is `+ - * /` and
//!   `sqrt` (through `normalize`), which are IEEE-exact operations.
//! * **Canonical**, so the answer is a function of the point set and not of
//!   which hull vertex the walk started at. Two facts do it: the rectangle's
//!   local `+X` is always its **longer** axis, and the basis direction is
//!   sign-normalised into the half-plane `u.x > 0` (ties broken by `u.y > 0`).
//!   Without that, a square lot has four equally-minimal answers and a rebuild
//!   can pick a different one.

use glam::DVec2;

/// The convex hull of a point set, counter-clockwise, without its repeated
/// first point.
///
/// Andrew's monotone chain: sort, then walk a lower and an upper hull popping
/// any vertex that is not a strict left turn. `O(n log n)`, exact for the
/// predicate it uses (a cross product of differences), and stable — a set
/// presented in a different order gives the same hull, which is the property a
/// cooked asset needs.
///
/// Non-finite points are dropped rather than propagated: a NaN loses every
/// comparison, so leaving one in makes the sort's order depend on the input
/// order and the hull's shape depend on where the NaN landed.
pub fn convex_hull_2d(points: &[DVec2]) -> Vec<DVec2> {
    let mut pts: Vec<DVec2> = points.iter().copied().filter(|p| p.is_finite()).collect();
    if pts.len() < 3 {
        pts.dedup();
        return pts;
    }
    // `total_cmp`, not `partial_cmp().unwrap()` — the panic this workspace has
    // already paid for.
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }

    let cross = |o: DVec2, a: DVec2, b: DVec2| (a - o).perp_dot(b - o);
    let mut hull: Vec<DVec2> = Vec::with_capacity(pts.len() * 2);
    for &p in &pts {
        while hull.len() >= 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower = hull.len() + 1;
    for &p in pts.iter().rev() {
        while hull.len() >= lower && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

/// An oriented rectangle on a plane: a centre, half-extents, and the unit
/// direction of its own local `+X`.
///
/// `half.x` is always the **longer** half-extent — see the module docs on why
/// that is canonical rather than cosmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MinAreaRect {
    pub center: DVec2,
    /// Half-extents in the rectangle's own frame. `half.x >= half.y`.
    pub half: DVec2,
    /// Unit direction of local `+X` in the plane's frame.
    pub u: DVec2,
}

impl MinAreaRect {
    /// The rectangle's local `+Y` direction.
    #[inline]
    pub fn v(&self) -> DVec2 {
        DVec2::new(-self.u.y, self.u.x)
    }

    pub fn area(&self) -> f64 {
        4.0 * self.half.x * self.half.y
    }

    /// A plane-frame point in the rectangle's own coordinates.
    #[inline]
    pub fn to_local(&self, p: DVec2) -> DVec2 {
        let d = p - self.center;
        DVec2::new(d.dot(self.u), d.dot(self.v()))
    }

    /// A rectangle-frame point back in the plane's frame.
    #[inline]
    pub fn to_world(&self, p: DVec2) -> DVec2 {
        self.center + self.u * p.x + self.v() * p.y
    }

    /// The four corners, counter-clockwise from local `(-x, -y)`.
    pub fn corners(&self) -> [DVec2; 4] {
        [
            self.to_world(DVec2::new(-self.half.x, -self.half.y)),
            self.to_world(DVec2::new(self.half.x, -self.half.y)),
            self.to_world(DVec2::new(self.half.x, self.half.y)),
            self.to_world(DVec2::new(-self.half.x, self.half.y)),
        ]
    }

    /// Whether `p` is inside, within `slack` metres.
    pub fn contains(&self, p: DVec2, slack: f64) -> bool {
        let l = self.to_local(p);
        l.x.abs() <= self.half.x + slack && l.y.abs() <= self.half.y + slack
    }
}

/// Sign-normalise a direction into the canonical half-plane.
fn canonical_dir(u: DVec2) -> DVec2 {
    if u.x < 0.0 || (u.x == 0.0 && u.y < 0.0) {
        -u
    } else {
        u
    }
}

/// **The oriented minimum-area rectangle** enclosing a point set.
///
/// Rotating calipers over the convex hull, using the theorem that the
/// minimum-area enclosing rectangle has a side **collinear with a hull edge** —
/// so the search is over `h` candidate orientations rather than a continuum,
/// and the answer is exact rather than optimised.
///
/// Returns `None` when fewer than three distinct finite points remain, and a
/// zero-width rectangle when they are collinear: a lot with no area is a lot,
/// and the caller's `is_positive` check is the right place to reject it.
pub fn min_area_rect(points: &[DVec2]) -> Option<MinAreaRect> {
    let hull = convex_hull_2d(points);
    if hull.is_empty() {
        return None;
    }
    if hull.len() < 3 {
        // Degenerate: a point or a segment. Report it rather than refusing, so
        // the caller sees a zero-area lot instead of a missing one.
        let a = hull[0];
        let b = *hull.last().unwrap();
        let d = b - a;
        let u = if d.length_squared() > 0.0 {
            canonical_dir(d.normalize())
        } else {
            DVec2::X
        };
        return Some(MinAreaRect {
            center: (a + b) * 0.5,
            half: DVec2::new(d.length() * 0.5, 0.0),
            u,
        });
    }

    let mut best: Option<(f64, DVec2, DVec2, DVec2)> = None; // (area, u, lo, hi) in u/v
    for i in 0..hull.len() {
        let e = hull[(i + 1) % hull.len()] - hull[i];
        if e.length_squared() <= 0.0 {
            continue;
        }
        let u = e.normalize();
        let v = DVec2::new(-u.y, u.x);
        let mut lo = DVec2::splat(f64::INFINITY);
        let mut hi = DVec2::splat(f64::NEG_INFINITY);
        for p in &hull {
            let q = DVec2::new(p.dot(u), p.dot(v));
            lo = lo.min(q);
            hi = hi.max(q);
        }
        let ext = hi - lo;
        let area = ext.x * ext.y;
        // `<` and not `<=`, so the FIRST minimal edge wins and the walk's start
        // does not decide a tie. Combined with the canonicalisation below, a
        // square's four equally-minimal orientations collapse to one answer.
        if best.is_none_or(|(a, _, _, _)| area < a) {
            best = Some((area, u, lo, hi));
        }
    }
    let (_, u, lo, hi) = best?;
    let v = DVec2::new(-u.y, u.x);
    let mid = (lo + hi) * 0.5;
    let center = u * mid.x + v * mid.y;
    let mut half = (hi - lo) * 0.5;
    let mut u = u;
    // Canonical: local +X is the LONGER axis, so "the long side of the lot" is
    // one fact rather than two, and the floor-plate slicer's own longer-axis
    // rule reads the same way it always did.
    if half.y > half.x {
        half = DVec2::new(half.y, half.x);
        u = v;
    }
    Some(MinAreaRect {
        center,
        half,
        u: canonical_dir(u),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rot(p: DVec2, c: DVec2, s: f64) -> DVec2 {
        // A rotation by a direction, not an angle — the same trig-free
        // construction the callers use.
        let d = DVec2::new(s, (1.0 - s * s).sqrt());
        DVec2::new(p.x * d.x - p.y * d.y, p.x * d.y + p.y * d.x) + c
    }

    #[test]
    fn the_hull_of_a_square_with_interior_points_is_the_square() {
        let mut pts = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(10.0, 0.0),
            DVec2::new(10.0, 10.0),
            DVec2::new(0.0, 10.0),
        ];
        for i in 1..9 {
            pts.push(DVec2::new(i as f64, i as f64));
            pts.push(DVec2::new(5.0, i as f64));
        }
        let h = convex_hull_2d(&pts);
        assert_eq!(h.len(), 4, "{h:?}");
        for c in [
            DVec2::new(0.0, 0.0),
            DVec2::new(10.0, 0.0),
            DVec2::new(10.0, 10.0),
            DVec2::new(0.0, 10.0),
        ] {
            assert!(h.contains(&c), "{c:?} missing from {h:?}");
        }
        // Order-independent: the same set shuffled gives the same hull.
        let mut shuffled = pts.clone();
        shuffled.reverse();
        assert_eq!(convex_hull_2d(&shuffled), h);
        // Collinear points do not become hull vertices.
        let line = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(1.0, 0.0),
            DVec2::new(2.0, 0.0),
        ];
        assert_eq!(convex_hull_2d(&line).len(), 2);
        // A NaN is dropped rather than deciding the sort.
        let mut with_nan = pts.clone();
        with_nan.push(DVec2::new(f64::NAN, 3.0));
        assert_eq!(convex_hull_2d(&with_nan), h);
    }

    /// **The rectangle is the lot's, not the axes'.**
    ///
    /// A 30 × 10 lot rotated off the compass has an axis-aligned bounding box of
    /// ~28 × ~24 = 690 m² and a minimum-area rectangle of exactly 300 m². The
    /// arm asserts the AABB's own number too, because a gate against a cheaper
    /// alternative has to price the alternative.
    #[test]
    fn a_rotated_lot_gets_its_own_rectangle_not_the_axes() {
        let c = DVec2::new(100.0, -40.0);
        // sin = 0.6, cos = 0.8 — a 3-4-5 rotation, exact in binary.
        let corners: Vec<DVec2> = [
            DVec2::new(-15.0, -5.0),
            DVec2::new(15.0, -5.0),
            DVec2::new(15.0, 5.0),
            DVec2::new(-15.0, 5.0),
        ]
        .iter()
        .map(|p| rot(*p, c, 0.6))
        .collect();

        let r = min_area_rect(&corners).unwrap();
        assert!(
            (r.area() - 300.0).abs() < 1e-9,
            "the lot is 30 x 10 = 300 m2 however it is turned; got {}",
            r.area()
        );
        assert!((r.half.x - 15.0).abs() < 1e-9 && (r.half.y - 5.0).abs() < 1e-9);
        assert!((r.center - c).length() < 1e-9, "{:?}", r.center);
        // The basis is the lot's long side, turned by the fixture's own rotation
        // (cos 0.6, sin 0.8).
        assert!(
            (r.u - DVec2::new(0.6, 0.8)).length() < 1e-9,
            "the basis must be the long side, got {:?}",
            r.u
        );
        // Every corner is on the rectangle's boundary.
        for p in &corners {
            let l = r.to_local(*p);
            assert!(
                (l.x.abs() - 15.0).abs() < 1e-9 && (l.y.abs() - 5.0).abs() < 1e-9,
                "{l:?}"
            );
        }

        // THE ALTERNATIVE, PRICED: the axis-aligned box of the same lot.
        let mut lo = DVec2::splat(f64::INFINITY);
        let mut hi = DVec2::splat(f64::NEG_INFINITY);
        for p in &corners {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let aabb = (hi.x - lo.x) * (hi.y - lo.y);
        assert!(
            aabb > 600.0,
            "the axis-aligned box of this lot is {aabb:.1} m2 against the oriented \
             300 — if that ratio is not large this fixture is not rotated"
        );
        println!(
            "IB-6: oriented {:.1} m2 vs axis-aligned {aabb:.1} m2",
            r.area()
        );
    }

    /// The answer is a function of the point set. A square has four equally
    /// minimal orientations and must always come back as the same one.
    #[test]
    fn the_rectangle_is_canonical_under_reordering_and_ties() {
        let sq = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(8.0, 0.0),
            DVec2::new(8.0, 8.0),
            DVec2::new(0.0, 8.0),
        ];
        let a = min_area_rect(&sq).unwrap();
        for shift in 1..4 {
            let mut rotated = sq.clone();
            rotated.rotate_left(shift);
            assert_eq!(min_area_rect(&rotated).unwrap(), a, "shift {shift}");
        }
        let mut rev = sq.clone();
        rev.reverse();
        assert_eq!(min_area_rect(&rev).unwrap(), a);
        assert_eq!(a.u, DVec2::X);
        assert!(a.half.x >= a.half.y, "local +X is the longer axis");

        // Degenerates report rather than refuse.
        assert!(min_area_rect(&[]).is_none());
        let seg = min_area_rect(&[DVec2::ZERO, DVec2::new(4.0, 3.0)]).unwrap();
        assert!((seg.half.x - 2.5).abs() < 1e-12 && seg.half.y == 0.0);
        assert_eq!(seg.area(), 0.0);
    }

    /// An L-shaped footprint — the case a bounding box is worst at — still gets
    /// the rectangle that actually encloses it, and every input point is inside.
    #[test]
    fn a_concave_footprint_is_enclosed_by_its_own_rectangle() {
        let l = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(20.0, 0.0),
            DVec2::new(20.0, 6.0),
            DVec2::new(8.0, 6.0),
            DVec2::new(8.0, 18.0),
            DVec2::new(0.0, 18.0),
        ];
        let c = DVec2::new(-30.0, 12.0);
        let turned: Vec<DVec2> = l.iter().map(|p| rot(*p, c, 0.6)).collect();
        let r = min_area_rect(&turned).unwrap();
        for p in &turned {
            assert!(r.contains(*p, 1e-9), "{p:?} escaped {r:?}");
        }
        // …and it is snug: at least two of the four sides touch a point.
        let touching = turned
            .iter()
            .filter(|p| {
                let l = r.to_local(**p);
                (l.x.abs() - r.half.x).abs() < 1e-9 || (l.y.abs() - r.half.y).abs() < 1e-9
            })
            .count();
        assert!(touching >= 4, "only {touching} points touch the rectangle");
    }
}
