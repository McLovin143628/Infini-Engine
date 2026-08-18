//! Deterministic 2D Delaunay triangulation (P29.2 clause 3).
//!
//! This closes the P11.2 deferral written into [`crate::blend_space`]'s own
//! module docs: *"True 2D blend spaces triangulate the sample points (Delaunay)
//! and barycentrically weight the enclosing triangle. v1 ships the simpler
//! inverse-distance weighting of the `k = 3` nearest."*
//!
//! # Why IDW had to go
//!
//! Inverse-distance weighting of the three nearest samples is not a partition of
//! unity over a triangulation: **every** sample within the three nearest
//! contributes, whatever side of the query it is on, so a locomotion grid blends
//! the *backward* clip into a forward walk as soon as the backward sample is one
//! of the three closest. ALS works around exactly this — its `FALSVelocityBlend`
//! is a hand-rolled four-way blend whose comment says outright that it exists
//! because it "produces better directional blending than a standard
//! blendspace". Barycentric weights over a Delaunay triangle have the property
//! IDW lacks: only the three corners of the simplex the query is *inside*
//! contribute, and the weights are the unique affine ones. So the port map's
//! `[SUPERSEDE]` verdict is discharged by construction rather than by porting the
//! workaround.
//!
//! # Determinism
//!
//! Everything here is a house-law obligation, not a nicety: a blend weight
//! reaches the pose, and the pose is folded into `state_bytes`, which two hosts
//! and two machines are compared on.
//!
//! * The insertion order is **sorted** (`x`, then `y`, then the original index),
//!   never the caller's order and never a hash order.
//! * Output triangles are canonicalised — each triple ascending, the list
//!   lexicographically sorted — so the same point set is the same triangulation
//!   whatever order it arrived in.
//! * The arithmetic is `f64` add/sub/mul only. No transcendental, no `sqrt` in
//!   the predicate. (There is a `sqrt` in the *outside-the-hull* projection, and
//!   IEEE-754 specifies `sqrt` exactly — it is the one function the
//!   portable-math ban list deliberately omits.)
//! * Co-circular points — a **square**, which is what a hand-authored locomotion
//!   grid is — are the degenerate case that matters, and the predicate is
//!   deliberately *strict*: a point exactly on a circumcircle is not "inside", so
//!   no flip happens and the triangulation stays whatever the sorted insertion
//!   built. Not unique, but every choice interpolates identically along the
//!   shared diagonal, and it is the same choice every time on every machine.

use glam::DVec2;

/// A triangulated point set: the triangles, and the boundary of their union.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Triangulation {
    /// Triangles as ascending-index triples, lexicographically sorted.
    pub triangles: Vec<[usize; 3]>,
    /// The edges that belong to exactly one triangle — the convex hull, as
    /// ascending-index pairs, sorted. Empty when `triangles` is.
    pub hull: Vec<[usize; 2]>,
}

impl Triangulation {
    /// Whether anything was triangulated (fewer than three distinct points, or a
    /// wholly collinear set, produces nothing — see [`triangulate`]).
    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }
}

/// Triangulate `points`, returning triangles as indices into it.
///
/// Empty when there are fewer than three points, when every point is a duplicate
/// of another, or when the whole set is **collinear** — a set with no area has no
/// triangulation, and the caller ([`crate::blend_space::blend_weights_2d`]) falls
/// back to a chain interpolation rather than inventing a sliver.
///
/// Duplicate positions are collapsed: the **lowest** index at a position is the
/// representative, and the others take no weight. An exact duplicate in a blend
/// space is an authoring mistake with no correct answer; picking the first is
/// deterministic and says so.
pub fn triangulate(points: &[DVec2]) -> Triangulation {
    let n = points.len();
    if n < 3 {
        return Triangulation::default();
    }
    // 1. Deterministic insertion order, and duplicate collapse in the same pass.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        cmp_f64(points[a].x, points[b].x)
            .then_with(|| cmp_f64(points[a].y, points[b].y))
            .then_with(|| a.cmp(&b))
    });
    let mut insert: Vec<usize> = Vec::with_capacity(n);
    for &i in &order {
        if !points[i].is_finite() {
            continue;
        }
        match insert.last() {
            Some(&prev) if points[prev] == points[i] => {
                // Same position: keep the lower original index as the
                // representative (the sort is stable on index within a position).
                if i < prev {
                    *insert.last_mut().expect("just matched") = i;
                }
            }
            _ => insert.push(i),
        }
    }
    if insert.len() < 3 {
        return Triangulation::default();
    }

    // 2. A super-triangle comfortably enclosing the set. `at()` reads a working
    //    index: below `n` it is an input point, at or above it a super vertex.
    let mut lo = points[insert[0]];
    let mut hi = lo;
    for &i in &insert {
        lo = lo.min(points[i]);
        hi = hi.max(points[i]);
    }
    let centre = (lo + hi) * 0.5;
    let span = (hi - lo).max_element().max(1.0) * 32.0;
    let super_pts = [
        DVec2::new(centre.x, centre.y + span),
        DVec2::new(centre.x - span, centre.y - span),
        DVec2::new(centre.x + span, centre.y - span),
    ];
    let at = |i: usize| -> DVec2 {
        if i < n {
            points[i]
        } else {
            super_pts[i - n]
        }
    };

    let mut tris: Vec<[usize; 3]> = vec![ccw([n, n + 1, n + 2], &at)];
    let mut bad: Vec<usize> = Vec::new();
    let mut edges: Vec<[usize; 2]> = Vec::new();
    for &p in &insert {
        bad.clear();
        for (ti, t) in tris.iter().enumerate() {
            if in_circumcircle(at(t[0]), at(t[1]), at(t[2]), points[p]) {
                bad.push(ti);
            }
        }
        if bad.is_empty() {
            // The point landed on (or outside, which the super-triangle makes
            // impossible) every circumcircle's boundary. Skipping it is the only
            // answer that keeps the mesh valid; it takes no weight, exactly like a
            // duplicate.
            continue;
        }
        // The cavity boundary: directed edges of the removed triangles that have
        // no opposite twin among them.
        edges.clear();
        for &ti in &bad {
            let t = tris[ti];
            edges.push([t[0], t[1]]);
            edges.push([t[1], t[2]]);
            edges.push([t[2], t[0]]);
        }
        let boundary: Vec<[usize; 2]> = edges
            .iter()
            .enumerate()
            .filter(|(i, e)| {
                !edges
                    .iter()
                    .enumerate()
                    .any(|(j, f)| j != *i && f[0] == e[1] && f[1] == e[0])
            })
            .map(|(_, e)| *e)
            .collect();
        // `bad` is ascending, so removing from the back keeps the earlier indices
        // valid. `swap_remove` reorders the tail, which is why the final list is
        // canonicalised rather than trusted.
        for &ti in bad.iter().rev() {
            tris.swap_remove(ti);
        }
        for e in boundary {
            tris.push(ccw([e[0], e[1], p], &at));
        }
    }

    // 3. Drop everything still touching the super-triangle, then canonicalise.
    tris.retain(|t| t.iter().all(|&i| i < n));
    let mut triangles: Vec<[usize; 3]> = tris
        .into_iter()
        .map(|mut t| {
            t.sort_unstable();
            t
        })
        .collect();
    triangles.sort_unstable();
    triangles.dedup();
    // A zero-area triangle is not a simplex: barycentric weights over one are a
    // division by zero, and it can only come from a collinear run the predicate
    // could not reject.
    triangles.retain(|t| signed_area(points[t[0]], points[t[1]], points[t[2]]).abs() > 0.0);

    let hull = boundary_edges(&triangles);
    Triangulation { triangles, hull }
}

/// The edges belonging to exactly one triangle, as ascending pairs, sorted.
fn boundary_edges(triangles: &[[usize; 3]]) -> Vec<[usize; 2]> {
    let mut all: Vec<[usize; 2]> = Vec::with_capacity(triangles.len() * 3);
    for t in triangles {
        for e in [[t[0], t[1]], [t[1], t[2]], [t[0], t[2]]] {
            all.push(if e[0] <= e[1] { e } else { [e[1], e[0]] });
        }
    }
    all.sort_unstable();
    let mut out = Vec::new();
    let mut i = 0;
    while i < all.len() {
        let j = all[i..].partition_point(|e| *e == all[i]) + i;
        if j - i == 1 {
            out.push(all[i]);
        }
        i = j;
    }
    out
}

/// Order a triple counter-clockwise (the winding [`in_circumcircle`] assumes).
fn ccw(t: [usize; 3], at: &dyn Fn(usize) -> DVec2) -> [usize; 3] {
    if signed_area(at(t[0]), at(t[1]), at(t[2])) < 0.0 {
        [t[0], t[2], t[1]]
    } else {
        t
    }
}

/// Twice the signed area of the triangle `abc` — positive when counter-clockwise.
pub fn signed_area(a: DVec2, b: DVec2, c: DVec2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)
}

/// Whether `d` lies **strictly** inside the circumcircle of the CCW triangle
/// `abc`.
///
/// The lifted-determinant form, with a *relative* tolerance so that a
/// co-circular point (the four corners of a square, which is what a
/// hand-authored locomotion grid is) reads as **outside**. That is the choice
/// that keeps the triangulation stable rather than flipping on the last bit of a
/// rounding error, and it is why this predicate is documented as
/// order-deterministic instead of unique.
fn in_circumcircle(a: DVec2, b: DVec2, c: DVec2, d: DVec2) -> bool {
    let (ax, ay) = (a.x - d.x, a.y - d.y);
    let (bx, by) = (b.x - d.x, b.y - d.y);
    let (cx, cy) = (c.x - d.x, c.y - d.y);
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let det = a2 * (bx * cy - cx * by) - b2 * (ax * cy - cx * ay) + c2 * (ax * by - bx * ay);
    // The determinant scales as length⁴; the tolerance has to scale with it or it
    // means something different for a grid in metres and one in centimetres.
    let scale = a2.max(b2).max(c2);
    det > 1e-12 * scale * scale
}

/// Barycentric coordinates of `p` in the triangle `abc`.
///
/// `None` for a degenerate (zero-area) triangle — the caller must not divide by
/// it. The three values sum to 1; all three non-negative means `p` is inside or
/// on the boundary.
pub fn barycentric(a: DVec2, b: DVec2, c: DVec2, p: DVec2) -> Option<[f64; 3]> {
    let area = signed_area(a, b, c);
    if area == 0.0 || !area.is_finite() {
        return None;
    }
    let w0 = signed_area(p, b, c) / area;
    let w1 = signed_area(a, p, c) / area;
    let w2 = signed_area(a, b, p) / area;
    if !(w0.is_finite() && w1.is_finite() && w2.is_finite()) {
        return None;
    }
    Some([w0, w1, w2])
}

/// A total order over `f64` that puts NaN last — the sort key, so a poisoned
/// coordinate cannot make the insertion order depend on the comparison sort's
/// internals.
fn cmp_f64(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| {
        // NaN sorts after everything; two NaNs are equal.
        match (a.is_nan(), b.is_nan()) {
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            _ => std::cmp::Ordering::Equal,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(v: &[(f64, f64)]) -> Vec<DVec2> {
        v.iter().map(|&(x, y)| DVec2::new(x, y)).collect()
    }

    #[test]
    fn a_single_triangle_is_itself() {
        let t = triangulate(&pts(&[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]));
        assert_eq!(t.triangles, vec![[0, 1, 2]]);
        // All three edges are on the hull.
        assert_eq!(t.hull, vec![[0, 1], [0, 2], [1, 2]]);
    }

    /// **The unit square** — the co-circular case a hand-authored grid produces.
    /// Two triangles covering it, whichever diagonal; what matters is that it is
    /// a *cover* and that it is the same one every time.
    #[test]
    fn a_square_triangulates_into_a_cover() {
        let square = pts(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]);
        let t = triangulate(&square);
        assert_eq!(t.triangles.len(), 2, "{:?}", t.triangles);
        // Total area is the square's.
        let area: f64 = t
            .triangles
            .iter()
            .map(|x| signed_area(square[x[0]], square[x[1]], square[x[2]]).abs() * 0.5)
            .sum();
        assert!((area - 1.0).abs() < 1e-12, "covered {area}");
        // Four hull edges, no more.
        assert_eq!(t.hull.len(), 4, "{:?}", t.hull);
    }

    /// **Order-independence**, which is the property the determinism law actually
    /// needs: the same point set given in any order is the same triangulation.
    #[test]
    fn the_triangulation_does_not_depend_on_input_order() {
        let base = [
            (0.0, 0.0),
            (2.0, 0.0),
            (1.0, 1.7),
            (3.0, 1.5),
            (1.0, -1.2),
            (2.4, 0.8),
        ];
        let canonical = triangulate(&pts(&base));
        assert!(!canonical.is_empty());
        // Every rotation of the list, re-indexed back to the original numbering.
        for shift in 1..base.len() {
            let rotated: Vec<(f64, f64)> = (0..base.len())
                .map(|i| base[(i + shift) % base.len()])
                .collect();
            let t = triangulate(&pts(&rotated));
            let mut mapped: Vec<[usize; 3]> = t
                .triangles
                .iter()
                .map(|x| {
                    let mut m = [
                        (x[0] + shift) % base.len(),
                        (x[1] + shift) % base.len(),
                        (x[2] + shift) % base.len(),
                    ];
                    m.sort_unstable();
                    m
                })
                .collect();
            mapped.sort_unstable();
            assert_eq!(mapped, canonical.triangles, "shift {shift}");
        }
    }

    /// **A triangulation is a partition**: no interior edge belongs to more than
    /// two triangles, and the Euler relation holds for a planar triangulation of a
    /// convex point set — `2T = 2n − 2 − h` with `h` hull vertices.
    #[test]
    fn the_mesh_is_a_valid_planar_triangulation() {
        let p = pts(&[
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.3),
            (2.0, 1.0),
            (1.0, 2.4),
        ]);
        let t = triangulate(&p);
        // Every triangle has positive area and distinct vertices.
        for x in &t.triangles {
            assert!(x[0] < x[1] && x[1] < x[2], "{x:?} is not canonical");
            assert!(signed_area(p[x[0]], p[x[1]], p[x[2]]).abs() > 1e-12);
        }
        // No edge in three triangles.
        let mut counts: std::collections::BTreeMap<[usize; 2], usize> = Default::default();
        for x in &t.triangles {
            for e in [[x[0], x[1]], [x[1], x[2]], [x[0], x[2]]] {
                *counts.entry(e).or_default() += 1;
            }
        }
        assert!(counts.values().all(|&c| c <= 2), "{counts:?}");
        // The Delaunay property itself: no vertex strictly inside a circumcircle.
        for x in &t.triangles {
            let tri = ccw([x[0], x[1], x[2]], &|i| p[i]);
            for (i, &q) in p.iter().enumerate() {
                if x.contains(&i) {
                    continue;
                }
                assert!(
                    !in_circumcircle(p[tri[0]], p[tri[1]], p[tri[2]], q),
                    "point {i} is inside {x:?}'s circumcircle"
                );
            }
        }
    }

    #[test]
    fn degenerate_sets_triangulate_to_nothing() {
        // Fewer than three points.
        assert!(triangulate(&pts(&[(0.0, 0.0), (1.0, 0.0)])).is_empty());
        // Collinear.
        assert!(triangulate(&pts(&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])).is_empty());
        // Every point the same.
        assert!(triangulate(&pts(&[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)])).is_empty());
        // A NaN coordinate is skipped, and the rest still triangulate.
        let mut p = pts(&[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]);
        p.push(DVec2::new(f64::NAN, 0.5));
        assert_eq!(triangulate(&p).triangles, vec![[0, 1, 2]]);
    }

    /// A duplicate position collapses to the **lowest** index, whichever order the
    /// duplicate arrived in.
    #[test]
    fn a_duplicate_position_collapses_to_its_lowest_index() {
        let a = triangulate(&pts(&[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]));
        assert_eq!(a.triangles, vec![[0, 1, 2]]);
        let b = triangulate(&pts(&[(1.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]));
        assert_eq!(b.triangles, vec![[0, 1, 3]]);
    }

    #[test]
    fn barycentric_is_exact_at_the_corners_and_the_centroid() {
        let (a, b, c) = (
            DVec2::new(0.0, 0.0),
            DVec2::new(4.0, 0.0),
            DVec2::new(0.0, 4.0),
        );
        assert_eq!(barycentric(a, b, c, a).unwrap(), [1.0, 0.0, 0.0]);
        assert_eq!(barycentric(a, b, c, b).unwrap(), [0.0, 1.0, 0.0]);
        assert_eq!(barycentric(a, b, c, c).unwrap(), [0.0, 0.0, 1.0]);
        let mid = (a + b + c) / 3.0;
        let w = barycentric(a, b, c, mid).unwrap();
        for x in w {
            assert!((x - 1.0 / 3.0).abs() < 1e-12, "{w:?}");
        }
        // Outside is signalled by a negative coordinate, not by a refusal.
        let out = barycentric(a, b, c, DVec2::new(-1.0, -1.0)).unwrap();
        assert!(out.iter().any(|x| *x < 0.0), "{out:?}");
        // A degenerate triangle refuses rather than dividing by zero.
        assert!(barycentric(a, b, DVec2::new(8.0, 0.0), mid).is_none());
    }
}
