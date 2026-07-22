//! Control-point splines: pure f64 curve evaluation for camera rails, patrol
//! routes, and procedural placement (E-P5).
//!
//! # Ring-0 eval seam
//!
//! This module is the deterministic, pure-`f64` math backing the ECS `Spline`
//! component (`inf_ecs::components::Spline`). It never depends on the ECS — it
//! operates on plain [`glam::DVec3`] control points and a local [`SplineInterp`]
//! (callers map `inf_ecs::components::SplineInterp` onto it, avoiding a
//! dependency cycle). Everything here is a pure function of its inputs — no
//! wall-clock, no `rand`, no global state — so evaluating the same spline twice
//! yields byte-identical results (house determinism law).
//!
//! # Parameterization
//!
//! The global parameter `t ∈ [0, 1]` spans the **whole** spline. With `n`
//! control points the curve is split into `seg_count` segments:
//!
//! * **open** (`closed == false`): `seg_count = n - 1` — the curve runs from the
//!   first point (`t == 0`) to the last (`t == 1`);
//! * **closed** (`closed == true`): `seg_count = n` — the final segment wraps the
//!   last point back to the first, so `eval(0) == eval(1)`.
//!
//! `t` is clamped to `[0, 1]`. A global `t` maps to a segment index `seg` and a
//! local parameter `u ∈ [0, 1]` by `scaled = t * seg_count`, `seg =
//! floor(scaled)` (clamped to `seg_count - 1`), `u = scaled - seg`. At `t == 1`
//! this yields `seg == seg_count - 1`, `u == 1`, i.e. the exact final knot.
//!
//! # Catmull-Rom variant
//!
//! The smooth interpolation is the **uniform** (not centripetal / chordal)
//! Catmull-Rom spline — the classic `0.5 · [...]` cubic with unit knot spacing.
//! Uniform is chosen for v1 because it is the cheapest, is exactly
//! reproducible, and passes through every control point at the knots; its known
//! weakness (cusps/overshoot when control points are unevenly spaced) is
//! acceptable for editor rails and is a documented candidate for a centripetal
//! upgrade later. The endpoint / wrap conventions for the phantom control points
//! either side of a segment are:
//!
//! * **closed**: indices wrap modulo `n` (`P0 = points[seg-1 mod n]`, etc.), so
//!   the loop is C¹-continuous across the seam;
//! * **open**: indices **clamp** to `[0, n-1]` — the phantom endpoint before the
//!   first point duplicates the first point, and the one past the last
//!   duplicates the last. (Duplicating, rather than reflecting, keeps the ends
//!   from overshooting outward.)
//!
//! # Edge cases (NaN-free)
//!
//! * `0` control points → [`DVec3::ZERO`];
//! * `1` control point → that point (for every `t`);
//! * a closed spline with `< 2` points degenerates to the single-point / zero
//!   case above.
//!
//! No evaluation path divides by a possibly-zero quantity, so results are always
//! finite for finite inputs.
//!
//! # Documented follow-ups (the eval API is the seam)
//!
//! * per-point viewport **gizmo dragging** (points are edited through the Details
//!   List editor today);
//! * baking a spline to a renderable **mesh** (tube / ribbon);
//! * a **Blueprint eval node** (`spline.eval` / `spline.eval_at_distance`) — it
//!   would call straight into these pure functions.

use glam::DVec3;

/// How a spline's control points are interpolated. A local mirror of
/// `inf_ecs::components::SplineInterp` (kept here so Ring-0 math carries no ECS
/// dependency); callers map one onto the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SplineInterp {
    /// Straight segments between successive control points.
    Linear,
    /// Uniform Catmull-Rom — a smooth C¹ curve through every control point.
    #[default]
    CatmullRom,
}

/// Number of segments spanning the whole spline (see the module docs).
#[inline]
fn segment_count(n: usize, closed: bool) -> usize {
    if closed {
        n
    } else {
        n.saturating_sub(1)
    }
}

/// Map a global `t ∈ [0, 1]` to `(segment, local u)` for a spline with
/// `seg_count` segments. `t` is clamped; the returned `seg` is in
/// `0..seg_count` and `u` in `[0, 1]`.
#[inline]
fn locate(t: f64, seg_count: usize) -> (usize, f64) {
    debug_assert!(seg_count >= 1);
    let t = t.clamp(0.0, 1.0);
    let scaled = t * seg_count as f64;
    let mut seg = scaled.floor() as usize;
    if seg >= seg_count {
        // t == 1.0 (or FP just past): land on the final segment at u == 1.
        seg = seg_count - 1;
        return (seg, 1.0);
    }
    (seg, scaled - seg as f64)
}

/// Control-point access for a **closed** spline: wrap the index modulo `n`.
#[inline]
fn wrap(points: &[DVec3], i: isize) -> DVec3 {
    let n = points.len() as isize;
    points[i.rem_euclid(n) as usize]
}

/// Control-point access for an **open** spline: clamp the index to `[0, n-1]`
/// (duplicating the endpoints for the phantom control points).
#[inline]
fn clamp_idx(points: &[DVec3], i: isize) -> DVec3 {
    let n = points.len() as isize;
    points[i.clamp(0, n - 1) as usize]
}

/// Evaluate the spline with the given interpolation at global `t ∈ [0, 1]`.
/// Convenience dispatcher over [`eval_linear`] / [`eval_catmull_rom`].
pub fn eval(points: &[DVec3], closed: bool, interp: SplineInterp, t: f64) -> DVec3 {
    match interp {
        SplineInterp::Linear => eval_linear(points, closed, t),
        SplineInterp::CatmullRom => eval_catmull_rom(points, closed, t),
    }
}

/// Piecewise-linear evaluation at global `t ∈ [0, 1]` (see the module docs for
/// the segment mapping and edge cases).
pub fn eval_linear(points: &[DVec3], closed: bool, t: f64) -> DVec3 {
    let n = points.len();
    if n == 0 {
        return DVec3::ZERO;
    }
    if n == 1 {
        return points[0];
    }
    let seg_count = segment_count(n, closed);
    if seg_count == 0 {
        return points[0];
    }
    let (seg, u) = locate(t, seg_count);
    let a = points[seg];
    let b = if closed {
        wrap(points, seg as isize + 1)
    } else {
        points[seg + 1]
    };
    a.lerp(b, u)
}

/// Uniform Catmull-Rom evaluation at global `t ∈ [0, 1]` (see the module docs
/// for the variant, endpoint/wrap conventions, and edge cases). Passes exactly
/// through each control point at the knots.
pub fn eval_catmull_rom(points: &[DVec3], closed: bool, t: f64) -> DVec3 {
    let n = points.len();
    if n == 0 {
        return DVec3::ZERO;
    }
    if n == 1 {
        return points[0];
    }
    let seg_count = segment_count(n, closed);
    if seg_count == 0 {
        return points[0];
    }
    let (seg, u) = locate(t, seg_count);
    let s = seg as isize;
    // The four control points around this segment (P1→P2 is the drawn span).
    let (p0, p1, p2, p3) = if closed {
        (
            wrap(points, s - 1),
            wrap(points, s),
            wrap(points, s + 1),
            wrap(points, s + 2),
        )
    } else {
        (
            clamp_idx(points, s - 1),
            clamp_idx(points, s),
            clamp_idx(points, s + 1),
            clamp_idx(points, s + 2),
        )
    };
    catmull_rom_segment(p0, p1, p2, p3, u)
}

/// Uniform Catmull-Rom basis for one segment `P1 → P2` (`u ∈ [0, 1]`), with the
/// neighbouring control points `P0`/`P3` shaping the tangents.
#[inline]
fn catmull_rom_segment(p0: DVec3, p1: DVec3, p2: DVec3, p3: DVec3, u: f64) -> DVec3 {
    let u2 = u * u;
    let u3 = u2 * u;
    // 0.5 · [ 2P1 + (-P0+P2)u + (2P0-5P1+4P2-P3)u² + (-P0+3P1-3P2+P3)u³ ]
    (p1 * 2.0
        + (p2 - p0) * u
        + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * u2
        + (p1 * 3.0 - p2 * 3.0 + p3 - p0) * u3)
        * 0.5
}

/// One entry of an arc-length lookup table: a global parameter `t` and the
/// cumulative curve length from `t == 0` up to it.
pub type ArcLenSample = (f64, f64);

/// Build a monotonic arc-length lookup table by sampling the curve
/// `samples_per_seg` times per segment and accumulating chord lengths. The
/// returned table starts at `(0.0, 0.0)` and ends at `(1.0, total_length)`; its
/// second column is non-decreasing (a chord length is never negative), so it can
/// be binary-searched / linearly inverted by [`eval_at_distance`].
///
/// `samples_per_seg` is clamped to at least `1`. Degenerate splines (`< 2`
/// points, or a closed spline that collapses to a point) yield
/// `[(0.0, 0.0), (1.0, 0.0)]` — a zero-length, still-valid table.
pub fn arc_length_lut(
    points: &[DVec3],
    closed: bool,
    interp: SplineInterp,
    samples_per_seg: usize,
) -> Vec<ArcLenSample> {
    let n = points.len();
    let seg_count = if n >= 2 { segment_count(n, closed) } else { 0 };
    if seg_count == 0 {
        return vec![(0.0, 0.0), (1.0, 0.0)];
    }
    let per = samples_per_seg.max(1);
    let steps = seg_count * per;
    let mut out = Vec::with_capacity(steps + 1);
    let mut cumulative = 0.0;
    let mut prev = eval(points, closed, interp, 0.0);
    out.push((0.0, 0.0));
    for i in 1..=steps {
        let t = i as f64 / steps as f64;
        let p = eval(points, closed, interp, t);
        cumulative += prev.distance(p);
        out.push((t, cumulative));
        prev = p;
    }
    out
}

/// Total arc length of `lut` (its last cumulative value), or `0.0` for an empty
/// table.
#[inline]
pub fn lut_length(lut: &[ArcLenSample]) -> f64 {
    lut.last().map(|&(_, len)| len).unwrap_or(0.0)
}

/// Evaluate the spline at an absolute arc-length `distance` (world units) from
/// the start, using a precomputed [`arc_length_lut`]. `distance` is clamped to
/// `[0, total_length]`. Inverts the LUT by locating the bracketing samples and
/// linearly interpolating their `t`, then evaluates the curve there — so points
/// come out approximately evenly spaced along the curve (constant-speed
/// traversal, exact in the limit of a dense LUT).
///
/// A zero-length span between two LUT entries is handled without dividing by
/// zero (the lower `t` is used), keeping the result finite.
pub fn eval_at_distance(
    points: &[DVec3],
    closed: bool,
    interp: SplineInterp,
    lut: &[ArcLenSample],
    distance: f64,
) -> DVec3 {
    if lut.is_empty() {
        return eval(points, closed, interp, 0.0);
    }
    let total = lut_length(lut);
    let d = distance.clamp(0.0, total);
    // Find the first LUT entry whose cumulative length reaches `d`.
    let t = invert_lut(lut, d);
    eval(points, closed, interp, t)
}

/// Map an arc-length `d ∈ [0, total]` back to a global `t` by linear
/// interpolation within the bracketing [`arc_length_lut`] entries.
fn invert_lut(lut: &[ArcLenSample], d: f64) -> f64 {
    // Locate the segment [i-1, i] whose cumulative length brackets `d`.
    for w in 1..lut.len() {
        let (t0, l0) = lut[w - 1];
        let (t1, l1) = lut[w];
        if d <= l1 {
            let span = l1 - l0;
            if span <= 0.0 {
                return t0; // zero-length span: no division
            }
            let f = (d - l0) / span;
            return t0 + (t1 - t0) * f;
        }
    }
    lut.last().map(|&(t, _)| t).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(v: &[[f64; 3]]) -> Vec<DVec3> {
        v.iter().map(|a| DVec3::new(a[0], a[1], a[2])).collect()
    }

    const EPS: f64 = 1e-9;

    #[test]
    fn open_endpoints_are_exact() {
        let p = pts(&[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [4.0, 0.0, 3.0]]);
        for interp in [SplineInterp::Linear, SplineInterp::CatmullRom] {
            assert!((eval(&p, false, interp, 0.0) - p[0]).length() < EPS);
            assert!((eval(&p, false, interp, 1.0) - p[2]).length() < EPS);
        }
    }

    #[test]
    fn linear_midpoint_is_exact() {
        // Two points: the global midpoint is their average.
        let p = pts(&[[0.0, 0.0, 0.0], [4.0, 2.0, -6.0]]);
        let mid = eval_linear(&p, false, 0.5);
        assert!((mid - DVec3::new(2.0, 1.0, -3.0)).length() < EPS);
    }

    #[test]
    fn linear_segment_boundary_hits_middle_point() {
        // Three evenly parameterized points: t == 0.5 lands on the middle knot.
        let p = pts(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
        assert!((eval_linear(&p, false, 0.5) - p[1]).length() < EPS);
    }

    #[test]
    fn catmull_rom_passes_through_control_points() {
        let p = pts(&[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [3.0, -1.0, 1.0],
            [5.0, 0.0, 2.0],
        ]);
        // Knots are at t = i / (n-1) for an open spline.
        let n = p.len();
        for (i, cp) in p.iter().enumerate() {
            let t = i as f64 / (n - 1) as f64;
            let got = eval_catmull_rom(&p, false, t);
            assert!(
                (got - *cp).length() < 1e-9,
                "knot {i}: got {got:?}, want {cp:?}"
            );
        }
    }

    #[test]
    fn closed_wraps_continuously() {
        let p = pts(&[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
        ]);
        for interp in [SplineInterp::Linear, SplineInterp::CatmullRom] {
            let a = eval(&p, true, interp, 0.0);
            let b = eval(&p, true, interp, 1.0);
            assert!(
                (a - b).length() < EPS,
                "{interp:?}: eval(0)={a:?} eval(1)={b:?}"
            );
            // The seam is genuine: a point just before t == 1 approaches p[0].
            let near = eval(&p, true, interp, 1.0 - 1e-6);
            assert!((near - p[0]).length() < 1e-3);
        }
    }

    #[test]
    fn closed_catmull_rom_hits_every_knot() {
        let p = pts(&[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 4.0],
            [0.0, 0.0, 4.0],
        ]);
        let n = p.len();
        for (i, cp) in p.iter().enumerate() {
            let t = i as f64 / n as f64; // closed: seg_count == n
            let got = eval_catmull_rom(&p, true, t);
            assert!((got - *cp).length() < 1e-9, "closed knot {i}");
        }
    }

    #[test]
    fn edge_cases_are_nan_free() {
        // Zero points → ZERO.
        assert_eq!(eval(&[], false, SplineInterp::CatmullRom, 0.3), DVec3::ZERO);
        assert_eq!(eval(&[], true, SplineInterp::Linear, 0.7), DVec3::ZERO);
        // One point → that point, for any t and mode.
        let one = pts(&[[1.0, 2.0, 3.0]]);
        for t in [0.0, 0.5, 1.0] {
            for closed in [false, true] {
                let got = eval(&one, closed, SplineInterp::CatmullRom, t);
                assert!((got - one[0]).length() < EPS);
                assert!(got.is_finite());
            }
        }
    }

    #[test]
    fn lut_is_monotonic_and_ends_at_total() {
        let p = pts(&[
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [3.0, -1.0, 1.0],
            [5.0, 0.0, 2.0],
        ]);
        let lut = arc_length_lut(&p, false, SplineInterp::CatmullRom, 16);
        assert_eq!(lut.first().unwrap().0, 0.0);
        assert_eq!(lut.first().unwrap().1, 0.0);
        assert!((lut.last().unwrap().0 - 1.0).abs() < EPS);
        for w in 1..lut.len() {
            assert!(lut[w].1 >= lut[w - 1].1, "cumulative length decreased");
            assert!(lut[w].0 >= lut[w - 1].0, "t decreased");
            assert!(lut[w].1.is_finite());
        }
        assert!(lut_length(&lut) > 0.0);
    }

    #[test]
    fn linear_lut_length_matches_chord_sum() {
        // A straight-line 3-4-5 shape: total length is exactly the chord sum.
        let p = pts(&[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]]);
        let lut = arc_length_lut(&p, false, SplineInterp::Linear, 8);
        assert!((lut_length(&lut) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn eval_at_distance_endpoints_and_clamp() {
        let p = pts(&[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]]);
        let lut = arc_length_lut(&p, false, SplineInterp::Linear, 8);
        let total = lut_length(&lut);
        let start = eval_at_distance(&p, false, SplineInterp::Linear, &lut, 0.0);
        let end = eval_at_distance(&p, false, SplineInterp::Linear, &lut, total);
        assert!((start - p[0]).length() < EPS);
        assert!((end - p[2]).length() < EPS);
        // Halfway by distance (3.5) is 0.5 up the second (vertical) leg.
        let mid = eval_at_distance(&p, false, SplineInterp::Linear, &lut, 3.5);
        assert!((mid - DVec3::new(3.0, 0.5, 0.0)).length() < 1e-6);
        // Over-range clamps to the end; negative clamps to the start.
        let over = eval_at_distance(&p, false, SplineInterp::Linear, &lut, total + 100.0);
        assert!((over - p[2]).length() < EPS);
        let under = eval_at_distance(&p, false, SplineInterp::Linear, &lut, -5.0);
        assert!((under - p[0]).length() < EPS);
    }

    #[test]
    fn deterministic_repeat() {
        let p = pts(&[
            [0.0, 0.0, 0.0],
            [1.2, 2.3, -0.7],
            [3.1, -1.4, 1.9],
            [5.0, 0.5, 2.2],
        ]);
        for interp in [SplineInterp::Linear, SplineInterp::CatmullRom] {
            for closed in [false, true] {
                let a: Vec<_> = (0..=100)
                    .map(|i| eval(&p, closed, interp, i as f64 / 100.0))
                    .collect();
                let b: Vec<_> = (0..=100)
                    .map(|i| eval(&p, closed, interp, i as f64 / 100.0))
                    .collect();
                assert_eq!(a, b, "eval not deterministic ({interp:?}, closed={closed})");
                let l1 = arc_length_lut(&p, closed, interp, 16);
                let l2 = arc_length_lut(&p, closed, interp, 16);
                assert_eq!(l1, l2, "lut not deterministic");
            }
        }
    }
}
