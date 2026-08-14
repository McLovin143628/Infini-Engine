//! Spans: the 1-D domains a grammar expands along, and the frames it places on.
//!
//! A [`Span`] is an **arc-length-parameterized polyline** in world space. Every
//! source reduces to one:
//!
//! * a **spline** — sampled through [`inf_math::spline`]'s Catmull-Rom /linear
//!   evaluator at the same `t` sequence its own arc-length LUT uses, so the
//!   polyline's chord sums *are* the LUT (asserted, not assumed);
//! * a footprint **perimeter** — four edge spans plus four corner frames;
//! * footprint **rows** — N parallel spans across the rectangle.
//!
//! # Why a polyline and not the curve itself
//!
//! Arc length on a cubic has no closed form, so *any* implementation samples.
//! Making the sampled polyline the domain — rather than sampling only to build a
//! lookup table and then evaluating the curve again — means the length a slot is
//! measured against and the position it is placed at come from the **same**
//! points. A layout can therefore fill the span exactly (see
//! [`expand`](super::expand)) instead of exactly filling a length that the
//! placement then disagrees with.
//!
//! # Bit-portability
//!
//! Everything here uses only IEEE-754 add, subtract, multiply, divide and
//! `sqrt`, all of which are exactly specified by the standard and identical on
//! every target. In particular there is **no `atan2`** on the orientation path:
//! [`Frame::rotation`] builds the yaw quaternion from a half-angle identity
//! (`cos(θ/2) = √((1+cosθ)/2)`), which needs only the components of the unit
//! tangent. This is the P14 law (`f32`/`f64` `std` trig is not bit-identical
//! across platforms) applied to committed, deterministic placement data. The
//! only trigonometry on the whole grammar path is a module's authored euler
//! rotation, which goes through [`inf_math::portable::psin64`] /
//! [`pcos64`](inf_math::portable::pcos64).

use glam::{DQuat, DVec2, DVec3};
use inf_math::spline;
use uuid::Uuid;

/// Re-exported so a caller building a [`SplinePath`] needs `inf-pcg` alone —
/// the two Ring-2 evaluation sites map an ECS `SplineInterp` onto this one and
/// should not have to name `inf-math` to do it.
pub use inf_math::spline::SplineInterp;

/// The number of polyline samples a spline segment is cut into by default.
///
/// Sixteen is the same density `inf_math`'s own arc-length LUT tests use, and at
/// a 10 m knot spacing it puts a sample every ~60 cm — well under the smallest
/// module anybody lays along a fence.
pub const DEFAULT_SPLINE_SAMPLES: usize = 16;

/// An upper bound on the polyline samples one span may hold, so a spline with a
/// pathological control-point count cannot make expansion unbounded.
pub const MAX_SPAN_SAMPLES: usize = 1 << 16;

/// A spline resolved into **world space**: what a [`SplineSource`] hands back.
///
/// The control points are already transformed by the owning entity's
/// `GlobalTransform`; nothing downstream knows about entities.
#[derive(Debug, Clone, PartialEq)]
pub struct SplinePath {
    pub points: Vec<DVec3>,
    pub closed: bool,
    pub interp: SplineInterp,
}

impl SplinePath {
    /// A world-space path from local control points and an affine placement.
    ///
    /// The one function both evaluation sites (the editor command and the
    /// player's load pass) use to turn an ECS `Spline` into a span source, so
    /// the two cannot disagree about whether points are local or world.
    pub fn from_local(
        points: impl IntoIterator<Item = DVec3>,
        closed: bool,
        interp: SplineInterp,
        to_world: glam::DAffine3,
    ) -> Self {
        Self {
            points: points
                .into_iter()
                .map(|p| to_world.transform_point3(p))
                .collect(),
            closed,
            interp,
        }
    }
}

/// Resolves a spline entity's GUID into its world-space path.
///
/// The same seam shape as [`MaskSource`](crate::graph::MaskSource): the grammar
/// runtime is a pure function of its inputs and holds no world, so the caller
/// supplies the lookup. **Unlike `mask.image`, this seam is available on every
/// path** — a `Spline` is a persisted scene component in both codecs, so the
/// editor, PIE and the shipped player all resolve it from the world they already
/// built. There is no preview/shipping divergence here and therefore no cook
/// advisory for it.
pub trait SplineSource {
    /// The world-space path for `guid`, or `None` when the entity is absent or
    /// carries no spline.
    fn spline(&self, guid: Uuid) -> Option<SplinePath>;
}

/// The empty spline source: nothing resolves, so a `grammar.spline` node
/// produces no span and places nothing (fails closed).
pub struct NoSplines;

impl SplineSource for NoSplines {
    fn spline(&self, _guid: Uuid) -> Option<SplinePath> {
        None
    }
}

impl SplineSource for std::collections::HashMap<Uuid, SplinePath> {
    fn spline(&self, guid: Uuid) -> Option<SplinePath> {
        self.get(&guid).cloned()
    }
}

impl<T: SplineSource + ?Sized> SplineSource for &T {
    fn spline(&self, guid: Uuid) -> Option<SplinePath> {
        (**self).spline(guid)
    }
}

/// A placement frame on a span: a world position and the unit direction of
/// travel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub position: DVec3,
    /// Unit tangent (direction of travel). `+Z` of the placed module maps onto
    /// this.
    pub tangent: DVec3,
}

impl Frame {
    /// The module orientation for this frame: a **yaw-only** rotation taking
    /// `+Z` onto the horizontal component of the tangent.
    ///
    /// Yaw-only keeps modules upright, which is what a fence, a wall and a
    /// façade bay all want; a tangent that climbs simply leans the *span*, not
    /// the posts. A perfectly vertical tangent has no yaw and returns identity.
    ///
    /// Built from the half-angle identity rather than `atan2` — see the module
    /// docs on portability. With `d = (sinθ, cosθ)` the unit horizontal tangent:
    /// `c = cos(θ/2) = √((1 + cosθ)/2)` and `s = sin(θ/2) = sinθ / 2c`, giving
    /// the quaternion `(0, s, 0, c)`, which is unit by construction
    /// (`s² + c² = (1−cosθ)/2 + (1+cosθ)/2 = 1`).
    pub fn rotation(&self) -> DQuat {
        yaw_onto(self.tangent)
    }
}

/// `v > 0.0`, written once so the **NaN-rejecting** form reads as intent rather
/// than as a negated comparison.
///
/// Every length, extent and weight on the grammar path goes through this: a NaN
/// span length must behave like a zero one (place nothing) rather than like a
/// positive one, and `!(v > 0.0)` is the only spelling that does that. Naming it
/// keeps the meaning visible — and keeps clippy's `neg_cmp_op_on_partial_ord`
/// from being suppressed one `allow` at a time.
#[inline]
pub(crate) fn positive(v: f64) -> bool {
    v > 0.0
}

/// The yaw-only quaternion taking `+Z` onto the horizontal part of `dir`.
pub fn yaw_onto(dir: DVec3) -> DQuat {
    let len2 = dir.x * dir.x + dir.z * dir.z;
    if !positive(len2) {
        return DQuat::IDENTITY;
    }
    let inv = 1.0 / len2.sqrt();
    let (sin_t, cos_t) = (dir.x * inv, dir.z * inv);
    let c2 = (1.0 + cos_t) * 0.5;
    if !positive(c2) {
        // θ = π: facing −Z. The half-angle formula divides by zero here, so the
        // answer is written out.
        return DQuat::from_xyzw(0.0, 1.0, 0.0, 0.0);
    }
    let c = c2.sqrt();
    DQuat::from_xyzw(0.0, sin_t / (2.0 * c), 0.0, c)
}

/// A world-space euler-degree rotation (`YXZ`, matching
/// `inf_ecs::components::Transform`) built from **bit-portable** trigonometry.
///
/// Two details that are load-bearing rather than defensive:
///
/// * **A zero angle short-circuits to the exact identity.** `pcos64(0)` is the
///   polynomial's value `0.999_999_943_741_051_1` — short of `1` by
///   **5.63e-8** — which is accurate enough to place a rotated module and *not*
///   accurate enough for the overwhelmingly common `rot 0,0,0`, where anything
///   but an exact identity would make an unrotated fence post carry a residual
///   tilt.
/// * Each axis quaternion is **normalized**, because a ~1e-7 sine is not exactly
///   on the unit sphere and three of them compose.
pub fn euler_deg_to_quat(deg: DVec3) -> DQuat {
    let axis = |degrees: f64, ax: DVec3| {
        if degrees == 0.0 {
            return DQuat::IDENTITY;
        }
        let h = degrees.to_radians() * 0.5;
        let (s, c) = (inf_math::portable::psin64(h), inf_math::portable::pcos64(h));
        DQuat::from_xyzw(ax.x * s, ax.y * s, ax.z * s, c).normalize()
    };
    axis(deg.y, DVec3::Y) * axis(deg.x, DVec3::X) * axis(deg.z, DVec3::Z)
}

/// One 1-D domain: a world-space polyline with its cumulative arc lengths.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    points: Vec<DVec3>,
    /// `cum[i]` is the arc length from `points[0]` to `points[i]`; `cum[0] == 0`
    /// and `cum.len() == points.len()`. Non-decreasing by construction (a chord
    /// length is never negative).
    cum: Vec<f64>,
    closed: bool,
}

impl Span {
    /// A span from an ordered polyline. A `closed` span repeats its first point
    /// at the end, so the closing chord is part of the domain.
    ///
    /// Consecutive duplicate points are kept (they cost a zero-length segment
    /// and nothing else); a span of fewer than two points is **degenerate** and
    /// places nothing.
    pub fn from_points(points: impl IntoIterator<Item = DVec3>, closed: bool) -> Self {
        let mut points: Vec<DVec3> = points.into_iter().collect();
        if closed {
            if let Some(&first) = points.first() {
                if points.len() >= 2 {
                    points.push(first);
                }
            }
        }
        let mut cum = Vec::with_capacity(points.len());
        let mut acc = 0.0;
        for (i, p) in points.iter().enumerate() {
            if i > 0 {
                acc += points[i - 1].distance(*p);
            }
            cum.push(acc);
        }
        Self {
            points,
            cum,
            closed,
        }
    }

    /// A span sampling `path` at `samples_per_segment` steps per spline segment.
    ///
    /// The `t` sequence is exactly `inf_math::spline::arc_length_lut`'s
    /// (`i / steps` for `i` in `0..=steps`), so this polyline's cumulative
    /// lengths are bit-identical to that LUT's second column — pinned by
    /// `span_length_matches_the_shared_arc_length_lut`.
    pub fn from_spline(path: &SplinePath, samples_per_segment: usize) -> Self {
        let n = path.points.len();
        if n < 2 {
            return Self::from_points(path.points.iter().copied(), false);
        }
        let seg_count = if path.closed { n } else { n - 1 };
        let per = samples_per_segment.max(1);
        let steps = (seg_count * per).min(MAX_SPAN_SAMPLES);
        let pts = (0..=steps).map(|i| {
            spline::eval(
                &path.points,
                path.closed,
                path.interp,
                i as f64 / steps as f64,
            )
        });
        // `closed` is already expressed by the sampled points wrapping to the
        // start, so the polyline is built open and simply *is* a loop.
        let mut span = Self::from_points(pts, false);
        span.closed = path.closed;
        span
    }

    /// Total arc length (`0` for a degenerate span).
    pub fn length(&self) -> f64 {
        self.cum.last().copied().unwrap_or(0.0)
    }

    /// `true` when the span carries no usable length.
    pub fn is_degenerate(&self) -> bool {
        self.points.len() < 2 || !positive(self.length())
    }

    /// Whether the domain loops back on itself (documentation for consumers; the
    /// polyline already contains the closing chord).
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// The sampled polyline.
    pub fn points(&self) -> &[DVec3] {
        &self.points
    }

    /// The frame at arc length `s`, clamped to `[0, length()]`.
    ///
    /// The interpolation is written `a·(1−t) + b·t` rather than `a + (b−a)·t`
    /// **so the endpoints are exact**: at `t == 0` it is `a` bit-for-bit and at
    /// `t == 1` it is `b`. That is what lets a slot boundary landing exactly on
    /// the span's end place a module exactly at the span's last point.
    pub fn frame_at(&self, s: f64) -> Frame {
        if self.points.len() < 2 {
            return Frame {
                position: self.points.first().copied().unwrap_or(DVec3::ZERO),
                tangent: DVec3::Z,
            };
        }
        let total = self.length();
        // **Two panics in two adjacent lines** (C4-19), both from a NaN reaching
        // an API that asserts an ordering:
        //
        // * `s.clamp(0.0, total)` panics when `total` is NaN (`f64::clamp`
        //   asserts `min <= max`), and `total` is `cum.last()`, which is NaN if
        //   any control point of the source spline is;
        // * `partial_cmp(..).unwrap()` panics when either side is NaN — and this
        //   was the ONLY `partial_cmp().unwrap()` in production code tree-wide,
        //   a lone straggler in a house that otherwise uses `total_cmp`.
        //
        // Both reachable from a PCG/grammar asset. A span with no usable length
        // has no frames to hand out, so answer with the first point rather than
        // inventing one.
        if !(total.is_finite() && s.is_finite()) {
            return Frame {
                position: self.points[0],
                tangent: DVec3::Z,
            };
        }
        let s = s.clamp(0.0, total);
        // The last index whose cumulative length does not exceed `s`, capped so
        // there is always a following point. `total_cmp` is a TOTAL order — it
        // cannot fail to compare, which is the whole reason the house uses it.
        let i = match self.cum.binary_search_by(|c| c.total_cmp(&s)) {
            Ok(mut i) => {
                // Duplicate cumulative values (zero-length segments) — take the
                // first, so the frame is a pure function of `s`.
                while i > 0 && self.cum[i - 1] == s {
                    i -= 1;
                }
                i.min(self.points.len() - 2)
            }
            Err(i) => i.saturating_sub(1).min(self.points.len() - 2),
        };
        let (a, b) = (self.points[i], self.points[i + 1]);
        let seg = self.cum[i + 1] - self.cum[i];
        let t = if seg > 0.0 {
            (s - self.cum[i]) / seg
        } else {
            0.0
        };
        let position = a * (1.0 - t) + b * t;
        let delta = b - a;
        let tangent = if delta.length_squared() > 0.0 {
            delta.normalize()
        } else {
            // A zero-length segment carries no direction; borrow the nearest
            // non-degenerate one so a duplicated control point does not spin a
            // module to identity.
            self.nearest_direction(i)
        };
        Frame { position, tangent }
    }

    /// The direction of the closest non-degenerate segment to index `i`
    /// (searching forward first, then back), or `+Z`.
    fn nearest_direction(&self, i: usize) -> DVec3 {
        for j in i + 1..self.points.len().saturating_sub(1) {
            let d = self.points[j + 1] - self.points[j];
            if d.length_squared() > 0.0 {
                return d.normalize();
            }
        }
        for j in (0..i).rev() {
            let d = self.points[j + 1] - self.points[j];
            if d.length_squared() > 0.0 {
                return d.normalize();
            }
        }
        DVec3::Z
    }
}

/// Which axis a [`footprint_rows`] span runs along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowAxis {
    /// Rows run along world X, stacked along Z.
    X,
    /// Rows run along world Z, stacked along X.
    Z,
}

/// A set of spans plus any frames placed *outside* the grammar — today, a
/// footprint's corners.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpanSet {
    /// The domains the grammar expands along, in a fixed order.
    pub spans: Vec<Span>,
    /// Frames a corner module is stamped on, in a fixed order. Empty for every
    /// source but a perimeter.
    pub corners: Vec<Frame>,
}

impl SpanSet {
    /// A set holding one span.
    pub fn one(span: Span) -> Self {
        Self {
            spans: vec![span],
            corners: Vec::new(),
        }
    }

    /// `true` when there is nothing to expand along and no corner to stamp.
    pub fn is_empty(&self) -> bool {
        self.corners.is_empty() && self.spans.iter().all(|s| s.is_degenerate())
    }
}

/// The four corners of an XZ rectangle centred on `center`, in
/// `(−x,−z) → (+x,−z) → (+x,+z) → (−x,+z)` order — counter-clockwise seen from
/// above with `+Z` toward the viewer, and clockwise in the engine's left-handed
/// screen sense. The order is fixed because it decides which wall is edge 0.
fn rect_corners(center: DVec3, size: DVec2) -> [DVec3; 4] {
    let (hx, hz) = (size.x * 0.5, size.y * 0.5);
    [
        DVec3::new(center.x - hx, center.y, center.z - hz),
        DVec3::new(center.x + hx, center.y, center.z - hz),
        DVec3::new(center.x + hx, center.y, center.z + hz),
        DVec3::new(center.x - hx, center.y, center.z + hz),
    ]
}

/// A rectangle's perimeter as **four independent edge spans**, plus one frame
/// per corner.
///
/// # The v1 corner rule, stated
///
/// A perimeter is not one span but four, because a wall grammar wants to start
/// and end at a corner — `Wall -> Post Panel* Post` should put a post at each
/// end of each side, not run a single expansion around the loop and land its
/// only two posts on one arbitrary corner.
///
/// `corner_size` is the length each corner reserves. Every edge is **inset by
/// `corner_size / 2` at both ends**, so a corner module of that footprint sits
/// in the gap and abuts the two walls it joins. With `corner_size == 0` the
/// edges run corner to corner and the corner frames are still emitted (a
/// grammar with no corner module simply ignores them). An edge shorter than its
/// two insets degenerates to nothing rather than folding back on itself.
///
/// A corner frame's tangent is the direction of the **outgoing** edge, so a
/// corner piece modelled to turn left is oriented consistently at all four.
pub fn footprint_perimeter(center: DVec3, size: DVec2, corner_size: f64) -> SpanSet {
    let corners = rect_corners(center, size);
    let inset = (corner_size * 0.5).max(0.0);
    let mut spans = Vec::with_capacity(4);
    let mut frames = Vec::with_capacity(4);
    for i in 0..4 {
        let a = corners[i];
        let b = corners[(i + 1) % 4];
        let delta = b - a;
        let len = delta.length();
        let dir = if len > 0.0 { delta / len } else { DVec3::Z };
        frames.push(Frame {
            position: a,
            tangent: dir,
        });
        if len > 2.0 * inset {
            spans.push(Span::from_points([a + dir * inset, b - dir * inset], false));
        } else {
            // Degenerate on purpose: the corners alone consume this side.
            spans.push(Span::from_points([a, a], false));
        }
    }
    SpanSet {
        spans,
        corners: frames,
    }
}

/// A rectangle cut into `rows` evenly spaced parallel spans — the domain for
/// façade bays, floor joists, seating rows, and (in P19.5) a floor plate's
/// partition lines.
///
/// Row `j` sits at the **centre** of its band, so `rows == 1` runs down the
/// middle and no row ever lies on the boundary. Rows are ordered by ascending
/// cross-axis coordinate, and each runs in the positive direction of its own
/// axis.
pub fn footprint_rows(center: DVec3, size: DVec2, rows: u32, axis: RowAxis) -> SpanSet {
    if rows == 0 || !positive(size.x) || !positive(size.y) {
        return SpanSet::default();
    }
    let (hx, hz) = (size.x * 0.5, size.y * 0.5);
    let spans = (0..rows)
        .map(|j| {
            let f = (j as f64 + 0.5) / rows as f64;
            match axis {
                RowAxis::X => {
                    let z = center.z - hz + f * size.y;
                    Span::from_points(
                        [
                            DVec3::new(center.x - hx, center.y, z),
                            DVec3::new(center.x + hx, center.y, z),
                        ],
                        false,
                    )
                }
                RowAxis::Z => {
                    let x = center.x - hx + f * size.x;
                    Span::from_points(
                        [
                            DVec3::new(x, center.y, center.z - hz),
                            DVec3::new(x, center.y, center.z + hz),
                        ],
                        false,
                    )
                }
            }
        })
        .collect();
    SpanSet {
        spans,
        corners: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn path(points: &[[f64; 3]], closed: bool, interp: SplineInterp) -> SplinePath {
        SplinePath {
            points: points
                .iter()
                .map(|p| DVec3::new(p[0], p[1], p[2]))
                .collect(),
            closed,
            interp,
        }
    }

    /// **C4-19 — two panics in two adjacent lines.**
    ///
    /// `s.clamp(0.0, total)` panics when `total` is NaN (`f64::clamp` asserts
    /// `min <= max`), and `total` is `cum.last()` — NaN if any control point of
    /// the source spline is. `binary_search_by(|c| c.partial_cmp(&s).unwrap())`
    /// panics when either side is NaN, and it was the **only**
    /// `partial_cmp().unwrap()` in production code tree-wide.
    ///
    /// Both reachable from a PCG/grammar asset. Un-fix mutation: restore either
    /// spelling and this test panics rather than failing.
    #[test]
    fn a_span_built_over_a_nan_never_panics_and_never_hands_out_a_nan() {
        let s = Span::from_points(
            [
                DVec3::ZERO,
                DVec3::new(f64::NAN, 0.0, 0.0),
                DVec3::new(3.0, 4.0, 0.0),
            ],
            false,
        );
        for probe in [0.0f64, 1.0, -5.0, 1e9, f64::NAN, f64::INFINITY] {
            let f = s.frame_at(probe);
            assert!(
                f.tangent.is_finite(),
                "frame_at({probe}) tangent is {:?}",
                f.tangent
            );
        }
        // A finite span is unaffected: the healthy path still measures exactly.
        let ok = Span::from_points([DVec3::ZERO, DVec3::new(3.0, 0.0, 0.0)], false);
        assert_eq!(ok.length(), 3.0);
        assert_eq!(ok.frame_at(3.0).position, DVec3::new(3.0, 0.0, 0.0));
        // ... and a NaN query against a healthy span answers rather than panics.
        assert!(ok.frame_at(f64::NAN).position.is_finite());
    }

    #[test]
    fn a_straight_polyline_measures_and_places_exactly() {
        let s = Span::from_points(
            [
                DVec3::ZERO,
                DVec3::new(3.0, 0.0, 0.0),
                DVec3::new(3.0, 4.0, 0.0),
            ],
            false,
        );
        assert_eq!(s.length(), 7.0);
        // Endpoints are EXACT, not approximate — the property the lerp form buys.
        assert_eq!(s.frame_at(0.0).position, DVec3::ZERO);
        assert_eq!(s.frame_at(7.0).position, DVec3::new(3.0, 4.0, 0.0));
        assert_eq!(s.frame_at(3.0).position, DVec3::new(3.0, 0.0, 0.0));
        assert_eq!(s.frame_at(5.0).position, DVec3::new(3.0, 2.0, 0.0));
        // Clamping, both ways.
        assert_eq!(s.frame_at(-10.0).position, DVec3::ZERO);
        assert_eq!(s.frame_at(1e9).position, DVec3::new(3.0, 4.0, 0.0));
        assert_eq!(s.frame_at(1.0).tangent, DVec3::X);
    }

    /// **The span IS the LUT.** The polyline a spline span samples has exactly
    /// the cumulative lengths `inf_math::spline::arc_length_lut` computes at the
    /// same density — bit for bit, so the length a layout fills and the position
    /// it places at can never be two different curves.
    #[test]
    fn span_length_matches_the_shared_arc_length_lut() {
        for closed in [false, true] {
            for interp in [SplineInterp::Linear, SplineInterp::CatmullRom] {
                let p = path(
                    &[
                        [0.0, 0.0, 0.0],
                        [10.0, 0.0, 2.0],
                        [18.0, 1.0, 9.0],
                        [4.0, 0.0, 14.0],
                    ],
                    closed,
                    interp,
                );
                let span = Span::from_spline(&p, 16);
                let lut = spline::arc_length_lut(&p.points, closed, interp, 16);
                assert_eq!(span.points().len(), lut.len());
                assert_eq!(
                    span.length().to_bits(),
                    spline::lut_length(&lut).to_bits(),
                    "span length != lut length ({interp:?}, closed={closed})"
                );
            }
        }
    }

    /// Arc-length accuracy against an analytic figure: a closed Catmull-Rom
    /// through the four axis points of a circle of radius `r` is not a circle,
    /// but a **linear** closed span through 64 points on it is within a
    /// chord-error of `2πr`.
    #[test]
    fn arc_length_tracks_an_analytic_circle() {
        const R: f64 = 10.0;
        const N: usize = 64;
        let pts: Vec<DVec3> = (0..N)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / N as f64;
                DVec3::new(R * a.cos(), 0.0, R * a.sin())
            })
            .collect();
        let span = Span::from_spline(
            &SplinePath {
                points: pts,
                closed: true,
                interp: SplineInterp::Linear,
            },
            1,
        );
        let exact = std::f64::consts::TAU * R;
        // An N-gon inscribed in a circle is shorter by O(1/N²); at N = 64 that
        // is ~0.04 %.
        let rel = (exact - span.length()) / exact;
        assert!(rel > 0.0 && rel < 1e-3, "relative error {rel}");
    }

    #[test]
    fn degenerate_spans_are_nan_free() {
        for span in [
            Span::from_points([], false),
            Span::from_points([DVec3::ONE], false),
            Span::from_points([DVec3::ONE, DVec3::ONE], false),
            Span::from_points([DVec3::ONE], true),
            Span::from_spline(
                &SplinePath {
                    points: vec![],
                    closed: false,
                    interp: SplineInterp::CatmullRom,
                },
                8,
            ),
        ] {
            assert!(span.is_degenerate());
            assert_eq!(span.length(), 0.0);
            let f = span.frame_at(0.5);
            assert!(f.position.is_finite() && f.tangent.is_finite());
            assert!(f.rotation().is_finite());
        }
    }

    #[test]
    fn a_closed_span_wraps_back_to_its_start() {
        let s = Span::from_points(
            [
                DVec3::ZERO,
                DVec3::new(4.0, 0.0, 0.0),
                DVec3::new(4.0, 0.0, 4.0),
                DVec3::new(0.0, 0.0, 4.0),
            ],
            true,
        );
        assert!(s.is_closed());
        assert_eq!(s.length(), 16.0);
        assert_eq!(s.frame_at(16.0).position, DVec3::ZERO);
    }

    /// A duplicated control point makes a zero-length segment; the frame there
    /// must still carry a real direction rather than snapping to identity.
    #[test]
    fn zero_length_segments_borrow_a_direction() {
        let s = Span::from_points(
            [
                DVec3::ZERO,
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(2.0, 0.0, 0.0),
            ],
            false,
        );
        assert_eq!(s.length(), 2.0);
        let f = s.frame_at(1.0);
        assert_eq!(f.position, DVec3::new(1.0, 0.0, 0.0));
        assert_eq!(f.tangent, DVec3::X);
    }

    // ── orientation ─────────────────────────────────────────────────────────

    /// The portable yaw agrees with the trigonometric answer everywhere, and is
    /// exact at the axes.
    #[test]
    fn yaw_matches_from_rotation_y_without_trig() {
        assert_eq!(yaw_onto(DVec3::Z), DQuat::IDENTITY);
        for step in 0..720 {
            let theta = std::f64::consts::TAU * step as f64 / 720.0;
            let dir = DVec3::new(theta.sin(), 0.0, theta.cos());
            let want = DQuat::from_rotation_y(theta);
            let got = yaw_onto(dir);
            // Quaternions double-cover: compare the rotated basis instead.
            let (a, b) = (got * DVec3::Z, want * DVec3::Z);
            assert!((a - b).length() < 1e-9, "θ={theta}: {a:?} vs {b:?}");
            assert!((got.length() - 1.0).abs() < 1e-12, "not unit at θ={theta}");
        }
        // A vertical tangent has no yaw; the antiparallel case is exact.
        assert_eq!(yaw_onto(DVec3::Y), DQuat::IDENTITY);
        let back = yaw_onto(DVec3::NEG_Z) * DVec3::Z;
        assert!((back - DVec3::NEG_Z).length() < 1e-12, "{back:?}");
        // Magnitude does not matter, only direction.
        assert_eq!(yaw_onto(DVec3::X * 7.0), yaw_onto(DVec3::X));
    }

    /// The module euler composition matches glam's `YXZ` — the convention
    /// `inf_ecs::components::Transform` uses — while going through the portable
    /// polynomials rather than `std` trig.
    #[test]
    fn portable_euler_matches_the_transform_convention() {
        // An unrotated module is EXACTLY unrotated. The size of the error the
        // short-circuit avoids is asserted rather than described, so the prose
        // on `euler_deg_to_quat` cannot drift from the polynomial.
        let raw_cos0 = inf_math::portable::pcos64(0.0);
        assert!(raw_cos0 < 1.0, "the short-circuit would be pointless");
        assert!(
            (1.0 - raw_cos0 - 5.63e-8).abs() < 1e-10,
            "pcos64(0) is {raw_cos0}; the documented 5.63e-8 shortfall moved"
        );
        assert_eq!(euler_deg_to_quat(DVec3::ZERO), DQuat::IDENTITY);
        assert_eq!(
            euler_deg_to_quat(DVec3::new(0.0, 90.0, 0.0)).x,
            0.0,
            "a pure yaw must not leak into the other axes"
        );
        for deg in [
            DVec3::ZERO,
            DVec3::new(0.0, 90.0, 0.0),
            DVec3::new(30.0, -45.0, 15.0),
            DVec3::new(-80.0, 200.0, 12.5),
        ] {
            let want = DQuat::from_euler(
                glam::EulerRot::YXZ,
                deg.y.to_radians(),
                deg.x.to_radians(),
                deg.z.to_radians(),
            );
            let got = euler_deg_to_quat(deg);
            assert!((got.length() - 1.0).abs() < 1e-12, "{deg:?} is not unit");
            for probe in [DVec3::X, DVec3::Y, DVec3::Z] {
                assert!(
                    (got * probe - want * probe).length() < 1e-6,
                    "{deg:?}: {:?} vs {:?}",
                    got * probe,
                    want * probe
                );
            }
        }
    }

    // ── footprints ──────────────────────────────────────────────────────────

    #[test]
    fn a_perimeter_is_four_edges_and_four_corners() {
        let set = footprint_perimeter(DVec3::ZERO, DVec2::new(10.0, 6.0), 0.0);
        assert_eq!(set.spans.len(), 4);
        assert_eq!(set.corners.len(), 4);
        let lens: Vec<f64> = set.spans.iter().map(|s| s.length()).collect();
        assert_eq!(lens, vec![10.0, 6.0, 10.0, 6.0]);
        // Corners sit on the rectangle, in the documented order.
        assert_eq!(set.corners[0].position, DVec3::new(-5.0, 0.0, -3.0));
        assert_eq!(set.corners[1].position, DVec3::new(5.0, 0.0, -3.0));
        assert_eq!(set.corners[2].position, DVec3::new(5.0, 0.0, 3.0));
        assert_eq!(set.corners[3].position, DVec3::new(-5.0, 0.0, 3.0));
        // Each corner faces along the edge that leaves it.
        assert_eq!(set.corners[0].tangent, DVec3::X);
        assert_eq!(set.corners[1].tangent, DVec3::Z);
        assert_eq!(set.corners[2].tangent, DVec3::NEG_X);
        assert_eq!(set.corners[3].tangent, DVec3::NEG_Z);
    }

    #[test]
    fn corner_size_insets_every_edge_at_both_ends() {
        let set = footprint_perimeter(DVec3::ZERO, DVec2::new(10.0, 6.0), 1.0);
        let lens: Vec<f64> = set.spans.iter().map(|s| s.length()).collect();
        assert_eq!(lens, vec![9.0, 5.0, 9.0, 5.0]);
        assert_eq!(
            set.spans[0].frame_at(0.0).position,
            DVec3::new(-4.5, 0.0, -3.0)
        );
        // Perimeter conservation: 4 edges + 4 corners covers the whole rectangle.
        let total: f64 = lens.iter().sum();
        assert_eq!(total + 4.0 * 1.0, 2.0 * (10.0 + 6.0));
        // An edge that cannot host its own insets degenerates rather than
        // folding back on itself.
        let tight = footprint_perimeter(DVec3::ZERO, DVec2::new(10.0, 1.0), 4.0);
        assert!(tight.spans[1].is_degenerate());
        assert_eq!(tight.spans[0].length(), 6.0);
    }

    #[test]
    fn rows_are_centred_in_their_bands() {
        let set = footprint_rows(DVec3::ZERO, DVec2::new(8.0, 6.0), 3, RowAxis::X);
        assert_eq!(set.spans.len(), 3);
        assert!(set.corners.is_empty());
        let z: Vec<f64> = set
            .spans
            .iter()
            .map(|s| s.frame_at(0.0).position.z)
            .collect();
        assert_eq!(z, vec![-2.0, 0.0, 2.0]);
        for s in &set.spans {
            assert_eq!(s.length(), 8.0);
            assert_eq!(s.frame_at(0.0).tangent, DVec3::X);
        }
        // One row runs down the middle.
        let one = footprint_rows(DVec3::ZERO, DVec2::new(8.0, 6.0), 1, RowAxis::X);
        assert_eq!(one.spans[0].frame_at(0.0).position.z, 0.0);
        // The Z axis stacks along X instead.
        let across = footprint_rows(DVec3::ZERO, DVec2::new(8.0, 6.0), 2, RowAxis::Z);
        assert_eq!(across.spans[0].frame_at(0.0).tangent, DVec3::Z);
        assert_eq!(across.spans[0].length(), 6.0);
        // Degenerate inputs are empty, not panicking.
        assert!(footprint_rows(DVec3::ZERO, DVec2::new(8.0, 6.0), 0, RowAxis::X).is_empty());
        assert!(footprint_rows(DVec3::ZERO, DVec2::ZERO, 4, RowAxis::X).is_empty());
    }

    #[test]
    fn a_footprint_is_placed_at_its_centre_and_height() {
        let c = DVec3::new(100.0, 5.0, -40.0);
        let set = footprint_perimeter(c, DVec2::new(4.0, 4.0), 0.0);
        for f in &set.corners {
            assert_eq!(f.position.y, 5.0);
            assert!((f.position.x - 100.0).abs() == 2.0);
        }
        let rows = footprint_rows(c, DVec2::new(4.0, 4.0), 2, RowAxis::X);
        assert_eq!(
            rows.spans[0].frame_at(0.0).position,
            DVec3::new(98.0, 5.0, -41.0)
        );
    }

    #[test]
    fn spline_paths_transform_local_points_into_the_world() {
        let to_world = glam::DAffine3::from_rotation_translation(
            DQuat::from_rotation_y(std::f64::consts::FRAC_PI_2),
            DVec3::new(10.0, 2.0, 0.0),
        );
        let p = SplinePath::from_local(
            [DVec3::ZERO, DVec3::new(0.0, 0.0, 4.0)],
            false,
            SplineInterp::Linear,
            to_world,
        );
        assert!((p.points[0] - DVec3::new(10.0, 2.0, 0.0)).length() < EPS);
        assert!((p.points[1] - DVec3::new(14.0, 2.0, 0.0)).length() < EPS);
    }

    #[test]
    fn the_sample_budget_is_capped() {
        let pts: Vec<DVec3> = (0..5000).map(|i| DVec3::new(i as f64, 0.0, 0.0)).collect();
        let span = Span::from_spline(
            &SplinePath {
                points: pts,
                closed: false,
                interp: SplineInterp::Linear,
            },
            64,
        );
        assert_eq!(span.points().len(), MAX_SPAN_SAMPLES + 1);
    }

    #[test]
    fn no_splines_resolves_nothing_and_a_map_resolves_by_guid() {
        let guid = Uuid::from_u128(7);
        assert!(NoSplines.spline(guid).is_none());
        let mut map = std::collections::HashMap::new();
        map.insert(
            guid,
            path(
                &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                false,
                SplineInterp::Linear,
            ),
        );
        assert!(map.spline(guid).is_some());
        assert!(map.spline(Uuid::from_u128(8)).is_none());
    }
}
