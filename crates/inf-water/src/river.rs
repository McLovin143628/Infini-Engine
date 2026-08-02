//! Spline rivers: arc-length frames, width/depth profiles, and the downhill
//! validation the cook advises on.
//!
//! # The centreline is a spline, the river is a ribbon
//!
//! A river entity carries a [`Spline`](inf_math::spline) — the same control-point
//! curve camera rails and P19.4's grammar spans ride — and a [`RiverProfile`]. The
//! two together give a **ribbon**: a sequence of [`RiverFrame`]s spaced evenly by
//! **arc length** (never by the spline parameter `t`, which bunches up on tight
//! curves), each carrying a centre, a flow tangent, a horizontal across-vector,
//! and the width and depth interpolated along the profile.
//!
//! Even arc-length spacing is the same choice P19.4 made for grammar spans, and
//! for the same reason: everything downstream — the water surface tessellation,
//! the flow speed, the foam banding, a buoyant body's query — is a function of
//! *distance along the river*, and a frame list spaced in `t` would make all of
//! them stretch and squash with the control-point layout.
//!
//! # Frames, and why they do not twist
//!
//! The across-vector is `normalize(tangent × up)` — recomputed per frame from the
//! world up, **not** parallel-transported along the curve. Parallel transport
//! accumulates roll: a closed loop generally does not come back to the frame it
//! started from (the holonomy of the curve), which for a river shows up as a
//! ribbon that is visibly banked at the seam. Deriving the across-vector from the
//! world up costs the ability to bank a river — which water does not do — and buys
//! exact continuity on closed splines, which it must have. The one degenerate case
//! (a vertically-flowing "river", i.e. a waterfall) falls back to the previous
//! frame's across-vector; a waterfall is P20.4's problem, not v1's.
//!
//! # Determinism
//!
//! Everything here is a pure function of the control points, the flags and the
//! profile — no wall clock, no RNG, and no `std` trigonometry (the arc-length LUT
//! and the Catmull-Rom basis are IEEE add/mul only, and `sqrt` is exact). Two
//! builds of the same river are bit-identical, which is what lets the editor and
//! the shipped player draw and simulate the same ribbon.

use glam::{DVec2, DVec3};
use inf_math::spline::{self, ArcLenSample, SplineInterp};

/// World up. Rivers are horizontal ribbons; there is no per-river up axis.
const UP: DVec3 = DVec3::Y;

/// Below this the tangent is treated as vertical and the across-vector is carried
/// over from the previous frame rather than divided into existence.
const DEGENERATE_CROSS_EPS: f64 = 1e-9;

/// Default arc-length samples per spline segment when a caller does not say.
/// Matches the density [`spline::arc_length_lut`] is built at, so the frame
/// spacing and the length measurement agree.
pub const DEFAULT_SAMPLES_PER_SEGMENT: usize = 16;

/// The authored cross-section of a river: how wide and how deep it is at each
/// end, and how fast it flows.
///
/// Width and depth are interpolated **linearly in arc length** between the two
/// ends — the simplest profile that lets a stream widen into a river, and the one
/// an author can predict. A keyframed profile (a `Vec<(s, width, depth)>`) is the
/// obvious v2 and needs no change here beyond the interpolator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverProfile {
    /// Full width at the start of the spline, metres.
    pub width_start_m: f64,
    /// Full width at the end, metres.
    pub width_end_m: f64,
    /// Depth from the water surface to the bed at the start, metres. Drives the
    /// absorption tint and the shallow-water foam band.
    pub depth_start_m: f64,
    /// Depth at the end, metres.
    pub depth_end_m: f64,
    /// Surface flow speed along the tangent, m/s (SI). Positive flows from `t=0`
    /// toward `t=1`; negative reverses the river without re-authoring the spline.
    pub flow_speed_m_s: f64,
}

impl Default for RiverProfile {
    /// A modest stream: 6 m wide, 1.5 m deep, ambling at 1 m/s.
    fn default() -> Self {
        Self {
            width_start_m: 6.0,
            width_end_m: 6.0,
            depth_start_m: 1.5,
            depth_end_m: 1.5,
            flow_speed_m_s: 1.0,
        }
    }
}

/// One sample of the river's centreline, with its local frame and profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverFrame {
    /// World-space centre of the water surface here.
    pub center: DVec3,
    /// Unit flow direction (the spline tangent, pointing `t=0 → t=1`).
    pub tangent: DVec3,
    /// Unit horizontal across-vector (`tangent × up`, normalized). The left bank
    /// is `center − right·width/2`.
    pub right: DVec3,
    /// Arc length from the start of the spline, metres.
    pub s: f64,
    /// Full width here, metres.
    pub width_m: f64,
    /// Depth to the bed here, metres.
    pub depth_m: f64,
}

/// A built river: its frames, its length, and the flow it carries.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RiverPath {
    /// Frames in flow order, evenly spaced by arc length. `frames[0].s == 0` and
    /// `frames.last().s == length_m`.
    pub frames: Vec<RiverFrame>,
    /// Total centreline length, metres.
    pub length_m: f64,
    /// Whether the spline loops (the last frame coincides with the first).
    pub closed: bool,
    /// Surface flow speed, m/s (copied from the profile so a sampler needs only
    /// the path).
    pub flow_speed_m_s: f64,
}

/// Where a world point falls relative to a river.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiverSample {
    /// Arc length of the nearest centreline point, metres.
    pub s: f64,
    /// Signed lateral offset from the centreline along `right`, metres.
    pub lateral_m: f64,
    /// Water-surface elevation at the nearest centreline point, metres.
    pub surface_y: f64,
    /// Full width here, metres.
    pub width_m: f64,
    /// Depth to the bed here, metres.
    pub depth_m: f64,
    /// Unit flow direction here (horizontal component; `y` is kept so a steep
    /// river still reports the slope it flows down).
    pub tangent: DVec3,
}

impl RiverSample {
    /// `0` at the centreline, `1` at either bank, `>1` outside the river.
    #[inline]
    pub fn bank_fraction(&self) -> f64 {
        if self.width_m <= 0.0 {
            f64::INFINITY
        } else {
            2.0 * self.lateral_m.abs() / self.width_m
        }
    }

    /// Whether the sampled point is inside the ribbon.
    #[inline]
    pub fn inside(&self) -> bool {
        self.bank_fraction() <= 1.0
    }
}

impl RiverPath {
    /// Build a river from spline control points (already in **world** space) and a
    /// profile.
    ///
    /// `samples_per_segment` controls how finely the centreline is sampled; the
    /// frame count is `segments · samples_per_segment` (+1 for the open end).
    /// Degenerate inputs (fewer than two points, or a spline that collapses to a
    /// point) yield an empty path rather than a NaN-riddled one.
    pub fn build(
        points: &[DVec3],
        closed: bool,
        interp: SplineInterp,
        profile: &RiverProfile,
        samples_per_segment: usize,
    ) -> Self {
        if points.len() < 2 {
            return Self::default();
        }
        let per = samples_per_segment.max(1);
        let lut = spline::arc_length_lut(points, closed, interp, per);
        let length = spline::lut_length(&lut);
        if !length.is_finite() || length <= 0.0 {
            return Self::default();
        }
        // One frame per LUT step, so the frames and the length measurement are
        // sampled at exactly the same density.
        let steps = lut.len().saturating_sub(1).max(1);
        // The finite-difference half-step for the tangent. Half a frame spacing:
        // small enough to track a tight curve, large enough that the difference
        // does not vanish into the f64 noise floor of two nearby evaluations.
        let h = 0.5 * length / steps as f64;

        let mut frames: Vec<RiverFrame> = Vec::with_capacity(steps + 1);
        let mut prev_right: Option<DVec3> = None;
        for i in 0..=steps {
            let s = length * i as f64 / steps as f64;
            let center = at(points, closed, interp, &lut, s);
            let tangent = tangent_at(points, closed, interp, &lut, s, h, length, closed);
            let right = cross_at(tangent, prev_right);
            prev_right = Some(right);
            let u = s / length;
            frames.push(RiverFrame {
                center,
                tangent,
                right,
                s,
                width_m: lerp(profile.width_start_m, profile.width_end_m, u).max(0.0),
                depth_m: lerp(profile.depth_start_m, profile.depth_end_m, u).max(0.0),
            });
        }
        // A closed river's last frame IS its first, position and frame alike, so
        // the ribbon has no seam. (`eval(1) == eval(0)` already holds; the frame
        // vectors are made identical too, because the two one-sided finite
        // differences at the wrap are not literally the same expression.)
        if closed && frames.len() > 1 {
            let first = frames[0];
            if let Some(last) = frames.last_mut() {
                last.center = first.center;
                last.tangent = first.tangent;
                last.right = first.right;
                last.width_m = first.width_m;
                last.depth_m = first.depth_m;
            }
        }
        Self {
            frames,
            length_m: length,
            closed,
            flow_speed_m_s: profile.flow_speed_m_s,
        }
    }

    /// Build with [`DEFAULT_SAMPLES_PER_SEGMENT`].
    pub fn from_points(
        points: &[DVec3],
        closed: bool,
        interp: SplineInterp,
        profile: &RiverProfile,
    ) -> Self {
        Self::build(points, closed, interp, profile, DEFAULT_SAMPLES_PER_SEGMENT)
    }

    /// Whether this path carries no geometry.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.frames.len() < 2
    }

    /// Widest point on the river, metres — the bound a spatial query or a
    /// bounding volume needs.
    pub fn max_width_m(&self) -> f64 {
        self.frames.iter().fold(0.0f64, |a, f| a.max(f.width_m))
    }

    /// Project a world XZ point onto the centreline.
    ///
    /// Returns the nearest point's arc length, the signed lateral offset, and the
    /// profile there — **whether or not** the point is inside the banks, so a
    /// caller can build a shore falloff from `bank_fraction()` rather than a
    /// binary in/out. `None` only for an empty path.
    ///
    /// Cost is `O(frames)`: an exhaustive scan over the centreline polyline. That
    /// is deliberate for v1 — it is exact, order-independent and trivially
    /// deterministic, and a river is a few hundred frames. A spatial index is the
    /// documented follow-up if P20.2 ever queries thousands of bodies per step.
    pub fn sample(&self, p: DVec2) -> Option<RiverSample> {
        if self.is_empty() {
            return None;
        }
        let mut best: Option<(f64, usize, f64)> = None; // (dist², segment, u)
        for (i, pair) in self.frames.windows(2).enumerate() {
            let (a, b) = (pair[0], pair[1]);
            let a2 = DVec2::new(a.center.x, a.center.z);
            let b2 = DVec2::new(b.center.x, b.center.z);
            let ab = b2 - a2;
            let len2 = ab.length_squared();
            let u = if len2 > 0.0 {
                ((p - a2).dot(ab) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let closest = a2 + ab * u;
            let d2 = (p - closest).length_squared();
            if best.is_none_or(|(bd, _, _)| d2 < bd) {
                best = Some((d2, i, u));
            }
        }
        let (_, i, u) = best?;
        let a = self.frames[i];
        let b = self.frames[i + 1];
        let center = a.center.lerp(b.center, u);
        let tangent = norm_or(a.tangent.lerp(b.tangent, u), a.tangent);
        let right = norm_or(a.right.lerp(b.right, u), a.right);
        let lateral = (p - DVec2::new(center.x, center.z)).dot(DVec2::new(right.x, right.z));
        Some(RiverSample {
            s: lerp(a.s, b.s, u),
            lateral_m: lateral,
            surface_y: center.y,
            width_m: lerp(a.width_m, b.width_m, u),
            depth_m: lerp(a.depth_m, b.depth_m, u),
            tangent,
        })
    }

    /// Surface **flow velocity** at a world XZ point, m/s — the tangent scaled by
    /// the profile's speed, or `None` outside the banks.
    ///
    /// This is the P20.2 seam for drag: a buoyant body inside a river is pushed
    /// downstream at (a fraction of) this. Horizontal only; a river's vertical
    /// velocity is a rendering detail, not a force.
    pub fn flow_at(&self, p: DVec2) -> Option<DVec2> {
        let s = self.sample(p)?;
        if !s.inside() {
            return None;
        }
        let t = DVec2::new(s.tangent.x, s.tangent.z);
        let len = t.length();
        if len <= 0.0 {
            return Some(DVec2::ZERO);
        }
        Some(t / len * self.flow_speed_m_s)
    }

    /// The elevation profile the downhill validation reads: `(arc length, y)` per
    /// frame.
    pub fn surface_profile(&self) -> Vec<(f64, f64)> {
        self.frames.iter().map(|f| (f.s, f.center.y)).collect()
    }

    /// The **bed** elevation profile, sampled from a terrain height function:
    /// `(arc length, terrain height)` for every frame the function answers for.
    ///
    /// Frames over a hole in the terrain (no authored tile) are skipped rather
    /// than defaulted to zero — a river crossing an unloaded region must not be
    /// reported as plunging to sea level.
    pub fn bed_profile(&self, height_at: impl Fn(DVec2) -> Option<f64>) -> Vec<(f64, f64)> {
        self.frames
            .iter()
            .filter_map(|f| height_at(DVec2::new(f.center.x, f.center.z)).map(|h| (f.s, h)))
            .collect()
    }
}

/// A stretch of river that gains elevation in the direction it flows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UphillSpan {
    /// Arc length where the climb starts, metres.
    pub from_s: f64,
    /// Arc length where it ends, metres.
    pub to_s: f64,
    /// Total elevation gained across the span, metres.
    pub rise_m: f64,
}

impl UphillSpan {
    /// Length of the climb, metres.
    #[inline]
    pub fn length_m(&self) -> f64 {
        self.to_s - self.from_s
    }

    /// Mean gradient of the climb (rise over run), dimensionless. `0` for a
    /// zero-length span.
    #[inline]
    pub fn gradient(&self) -> f64 {
        let l = self.length_m();
        if l > 0.0 {
            self.rise_m / l
        } else {
            0.0
        }
    }
}

/// Contiguous stretches of an elevation profile that **rise**, merged, with
/// climbs of less than `tolerance_m` ignored.
///
/// Water flows downhill. A river authored to run the wrong way up a valley is a
/// mistake the cook can *see* and the runtime can only *look wrong* about, which
/// is exactly the shape of thing the advisory pattern exists for (the
/// `dangling_terrain_refs` precedent): named, with the remedy, where it is cheap
/// to fix.
///
/// `tolerance_m` exists because every profile this is fed is a **resampling**,
/// not the authored data, and a resampling wobbles. Which wobble depends on the
/// caller: [`RiverPath::surface_profile`] carries Catmull-Rom overshoot between
/// knots plus arc-length quantization (centimetres); [`RiverPath::bed_profile`]
/// additionally carries bilinear heightfield noise where the curve crosses a tile
/// diagonal (millimetres). Reporting either as "your river flows uphill" is how an
/// advisory stops being read.
///
/// The tolerance is applied to the **total rise of a merged span**, not to each
/// step, so a long, gentle, genuinely-wrong climb is still caught however small
/// its individual steps are. What it does **not** catch is a *sawtooth* — a
/// climb broken into sub-tolerance rises by intervening falls — because each span
/// closes at the fall. That is a real gap, and it is a deliberate one: the
/// alternative is a net-elevation test, which fires on every river that crosses a
/// ridge on its way down a valley, i.e. on correct content. Pinned by
/// `a_sawtooth_climb_escapes_the_per_span_tolerance` so it is a known property
/// rather than a surprise.
pub fn uphill_spans(profile: &[(f64, f64)], tolerance_m: f64) -> Vec<UphillSpan> {
    let tol = if tolerance_m.is_finite() {
        tolerance_m.max(0.0)
    } else {
        0.0
    };
    let mut out = Vec::new();
    let mut open: Option<UphillSpan> = None;
    for w in profile.windows(2) {
        let ((s0, y0), (s1, y1)) = (w[0], w[1]);
        if y1 > y0 {
            match open.as_mut() {
                Some(span) => {
                    span.to_s = s1;
                    span.rise_m += y1 - y0;
                }
                None => {
                    open = Some(UphillSpan {
                        from_s: s0,
                        to_s: s1,
                        rise_m: y1 - y0,
                    })
                }
            }
        } else if let Some(span) = open.take() {
            if span.rise_m >= tol {
                out.push(span);
            }
        }
    }
    if let Some(span) = open {
        if span.rise_m >= tol {
            out.push(span);
        }
    }
    out
}

// ── internals ───────────────────────────────────────────────────────────────

#[inline]
fn lerp(a: f64, b: f64, u: f64) -> f64 {
    a + (b - a) * u.clamp(0.0, 1.0)
}

#[inline]
fn at(points: &[DVec3], closed: bool, interp: SplineInterp, lut: &[ArcLenSample], s: f64) -> DVec3 {
    spline::eval_at_distance(points, closed, interp, lut, s)
}

/// Unit tangent at arc length `s`, by central difference where there is room and
/// a one-sided difference at an open spline's ends. A **closed** spline wraps, so
/// it is always central.
#[allow(clippy::too_many_arguments)]
fn tangent_at(
    points: &[DVec3],
    closed: bool,
    interp: SplineInterp,
    lut: &[ArcLenSample],
    s: f64,
    h: f64,
    length: f64,
    wrap: bool,
) -> DVec3 {
    let (a, b) = if wrap {
        let lo = wrap_s(s - h, length);
        let hi = wrap_s(s + h, length);
        (
            at(points, closed, interp, lut, lo),
            at(points, closed, interp, lut, hi),
        )
    } else {
        let lo = (s - h).max(0.0);
        let hi = (s + h).min(length);
        (
            at(points, closed, interp, lut, lo),
            at(points, closed, interp, lut, hi),
        )
    };
    norm_or(b - a, DVec3::X)
}

#[inline]
fn wrap_s(s: f64, length: f64) -> f64 {
    if length <= 0.0 {
        0.0
    } else {
        s.rem_euclid(length)
    }
}

/// The horizontal across-vector for a tangent, carrying the previous one over
/// when the tangent is (near-)vertical. See the module docs for why this is not
/// parallel transport.
fn cross_at(tangent: DVec3, prev: Option<DVec3>) -> DVec3 {
    let c = tangent.cross(UP);
    if c.length_squared() > DEGENERATE_CROSS_EPS {
        c.normalize()
    } else {
        prev.unwrap_or(DVec3::X)
    }
}

/// Normalize, falling back to `fallback` for a degenerate vector — never a NaN.
#[inline]
fn norm_or(v: DVec3, fallback: DVec3) -> DVec3 {
    let len = v.length();
    if len > 0.0 && len.is_finite() {
        v / len
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(v: &[[f64; 3]]) -> Vec<DVec3> {
        v.iter().map(|a| DVec3::new(a[0], a[1], a[2])).collect()
    }

    fn straight() -> Vec<DVec3> {
        pts(&[
            [0.0, 10.0, 0.0],
            [50.0, 8.0, 0.0],
            [100.0, 6.0, 0.0],
            [150.0, 4.0, 0.0],
        ])
    }

    #[test]
    fn frames_are_evenly_spaced_by_arc_length() {
        let path = RiverPath::from_points(
            &straight(),
            false,
            SplineInterp::CatmullRom,
            &RiverProfile::default(),
        );
        assert!(path.frames.len() > 8);
        assert_eq!(path.frames[0].s, 0.0);
        assert!((path.frames.last().unwrap().s - path.length_m).abs() < 1e-9);
        let step = path.length_m / (path.frames.len() - 1) as f64;
        for w in path.frames.windows(2) {
            assert!((w[1].s - w[0].s - step).abs() < 1e-9, "uneven spacing");
            // …and the *geometry* is evenly spaced too, which is the property that
            // matters (a LUT that reported even `s` for uneven points would be a
            // lie the frame list would then propagate everywhere).
            let d = w[0].center.distance(w[1].center);
            assert!((d - step).abs() < step * 0.05, "chord {d} vs step {step}");
        }
    }

    #[test]
    fn width_and_depth_follow_the_profile() {
        let profile = RiverProfile {
            width_start_m: 4.0,
            width_end_m: 12.0,
            depth_start_m: 1.0,
            depth_end_m: 3.0,
            flow_speed_m_s: 2.0,
        };
        let path = RiverPath::from_points(&straight(), false, SplineInterp::Linear, &profile);
        assert!((path.frames[0].width_m - 4.0).abs() < 1e-9);
        assert!((path.frames.last().unwrap().width_m - 12.0).abs() < 1e-9);
        assert!((path.frames[0].depth_m - 1.0).abs() < 1e-9);
        assert!((path.frames.last().unwrap().depth_m - 3.0).abs() < 1e-9);
        assert!((path.max_width_m() - 12.0).abs() < 1e-9);
        // Monotone in between (a linear profile in arc length).
        for w in path.frames.windows(2) {
            assert!(w[1].width_m >= w[0].width_m - 1e-12);
        }
    }

    #[test]
    fn frames_are_orthonormal_and_continuous() {
        let curvy = pts(&[
            [0.0, 5.0, 0.0],
            [20.0, 4.8, 5.0],
            [25.0, 4.6, 25.0],
            [5.0, 4.4, 40.0],
            [-20.0, 4.2, 30.0],
        ]);
        let path = RiverPath::from_points(
            &curvy,
            false,
            SplineInterp::CatmullRom,
            &RiverProfile::default(),
        );
        for f in &path.frames {
            assert!((f.tangent.length() - 1.0).abs() < 1e-9);
            assert!((f.right.length() - 1.0).abs() < 1e-9);
            assert!(f.tangent.dot(f.right).abs() < 1e-6, "frame not orthogonal");
            assert!(f.right.y.abs() < 1e-9, "the across-vector must stay level");
        }
        // Continuity: no frame flips relative to its neighbour, even on the
        // tight inside of the curve.
        for w in path.frames.windows(2) {
            assert!(
                w[0].tangent.dot(w[1].tangent) > 0.7,
                "tangent jumped between adjacent frames"
            );
            assert!(
                w[0].right.dot(w[1].right) > 0.7,
                "across-vector jumped between adjacent frames"
            );
        }
    }

    /// A closed river has no seam: the last frame *is* the first, so the ribbon
    /// closes exactly rather than to within a finite difference.
    #[test]
    fn a_closed_river_closes_exactly() {
        let loop_pts = pts(&[
            [0.0, 3.0, 0.0],
            [40.0, 3.0, 0.0],
            [40.0, 3.0, 40.0],
            [0.0, 3.0, 40.0],
        ]);
        let path = RiverPath::from_points(
            &loop_pts,
            true,
            SplineInterp::CatmullRom,
            &RiverProfile::default(),
        );
        assert!(path.closed);
        let first = path.frames[0];
        let last = *path.frames.last().unwrap();
        assert_eq!(first.center, last.center);
        assert_eq!(first.tangent, last.tangent);
        assert_eq!(first.right, last.right);
        // …and the frame really did wrap rather than degenerate: the tangent at
        // the seam continues around the loop.
        let second = path.frames[1];
        assert!(first.tangent.dot(second.tangent) > 0.9);
    }

    /// The tight-curve case the frame construction is really for: a hairpin. The
    /// across-vector must stay level and continuous through the reversal.
    #[test]
    fn a_hairpin_keeps_its_frames() {
        let hairpin = pts(&[
            [0.0, 2.0, 0.0],
            [30.0, 2.0, 0.0],
            [34.0, 2.0, 4.0],
            [30.0, 2.0, 8.0],
            [0.0, 2.0, 8.0],
        ]);
        let path = RiverPath::build(
            &hairpin,
            false,
            SplineInterp::CatmullRom,
            &RiverProfile::default(),
            32,
        );
        for f in &path.frames {
            assert!(f.right.y.abs() < 1e-9);
            assert!(f.tangent.is_finite() && f.right.is_finite());
        }
        // The tangent *does* reverse across the hairpin (that is the point) —
        // but never within one frame step.
        let ends = path.frames[0]
            .tangent
            .dot(path.frames.last().unwrap().tangent);
        assert!(ends < 0.0, "the hairpin did not turn around");
        for w in path.frames.windows(2) {
            assert!(w[0].tangent.dot(w[1].tangent) > 0.5, "tangent snapped");
        }
    }

    #[test]
    fn sampling_locates_the_bank_and_the_centreline() {
        let profile = RiverProfile {
            width_start_m: 10.0,
            width_end_m: 10.0,
            ..RiverProfile::default()
        };
        // A straight river along +X at y = 6, running through z = 0.
        let path = RiverPath::from_points(
            &pts(&[[0.0, 6.0, 0.0], [100.0, 6.0, 0.0]]),
            false,
            SplineInterp::Linear,
            &profile,
        );
        let mid = path.sample(DVec2::new(50.0, 0.0)).unwrap();
        assert!(mid.lateral_m.abs() < 1e-9);
        assert!(mid.inside());
        assert!((mid.surface_y - 6.0).abs() < 1e-9);
        assert!((mid.s - 50.0).abs() < 1e-6);

        // Exactly on the bank.
        let bank = path.sample(DVec2::new(50.0, 5.0)).unwrap();
        assert!((bank.bank_fraction() - 1.0).abs() < 1e-9);
        assert!(bank.inside());
        // Outside.
        let out = path.sample(DVec2::new(50.0, 9.0)).unwrap();
        assert!(!out.inside());
        assert!((out.bank_fraction() - 1.8).abs() < 1e-9);
        // The sign of the lateral offset is consistent with `right`.
        let l = path.sample(DVec2::new(50.0, -3.0)).unwrap();
        assert_eq!(l.lateral_m.signum(), -bank.lateral_m.signum());
    }

    #[test]
    fn flow_follows_the_tangent_and_stops_at_the_bank() {
        let path = RiverPath::from_points(
            &pts(&[[0.0, 1.0, 0.0], [100.0, 1.0, 0.0]]),
            false,
            SplineInterp::Linear,
            &RiverProfile {
                flow_speed_m_s: 2.5,
                width_start_m: 8.0,
                width_end_m: 8.0,
                ..RiverProfile::default()
            },
        );
        let v = path.flow_at(DVec2::new(50.0, 1.0)).unwrap();
        assert!((v.x - 2.5).abs() < 1e-6, "flow {v:?}");
        assert!(v.y.abs() < 1e-6);
        assert!(path.flow_at(DVec2::new(50.0, 20.0)).is_none());
        // A negative speed reverses the river without re-authoring the spline.
        let back = RiverPath::from_points(
            &pts(&[[0.0, 1.0, 0.0], [100.0, 1.0, 0.0]]),
            false,
            SplineInterp::Linear,
            &RiverProfile {
                flow_speed_m_s: -2.5,
                width_start_m: 8.0,
                width_end_m: 8.0,
                ..RiverProfile::default()
            },
        );
        assert!(back.flow_at(DVec2::new(50.0, 0.0)).unwrap().x < 0.0);
    }

    #[test]
    fn degenerate_rivers_are_empty_not_nan() {
        for (p, closed) in [
            (pts(&[]), false),
            (pts(&[[1.0, 2.0, 3.0]]), false),
            (pts(&[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), false),
        ] {
            let path = RiverPath::from_points(
                &p,
                closed,
                SplineInterp::CatmullRom,
                &RiverProfile::default(),
            );
            assert!(path.is_empty(), "{p:?}");
            assert_eq!(path.length_m, 0.0);
            assert!(path.sample(DVec2::ZERO).is_none());
            assert!(path.flow_at(DVec2::ZERO).is_none());
        }
    }

    #[test]
    fn building_is_deterministic() {
        let a = RiverPath::from_points(
            &straight(),
            false,
            SplineInterp::CatmullRom,
            &RiverProfile::default(),
        );
        let b = RiverPath::from_points(
            &straight(),
            false,
            SplineInterp::CatmullRom,
            &RiverProfile::default(),
        );
        assert_eq!(a, b);
    }

    // ── downhill validation ──────────────────────────────────────────────

    #[test]
    fn a_downhill_river_reports_nothing() {
        let path = RiverPath::from_points(
            &straight(),
            false,
            SplineInterp::Linear,
            &RiverProfile::default(),
        );
        assert!(uphill_spans(&path.surface_profile(), 0.05).is_empty());
    }

    #[test]
    fn an_uphill_stretch_is_reported_with_its_rise() {
        // Down, then up 3 m, then down again.
        let p = pts(&[
            [0.0, 10.0, 0.0],
            [50.0, 8.0, 0.0],
            [100.0, 11.0, 0.0],
            [150.0, 5.0, 0.0],
        ]);
        let path =
            RiverPath::from_points(&p, false, SplineInterp::Linear, &RiverProfile::default());
        let spans = uphill_spans(&path.surface_profile(), 0.05);
        assert_eq!(spans.len(), 1, "{spans:?}");
        let s = spans[0];
        // Within a frame spacing of the authored 3 m: the peak knot falls between
        // two arc-length frames, so a sampled profile measures slightly less than
        // the exact control-point rise. That is the same discretization the cook
        // advisory sees, which is why the tolerance is here and not hidden.
        assert!((s.rise_m - 3.0).abs() < 0.05, "rise {}", s.rise_m);
        assert!(s.from_s > 0.0 && s.to_s <= path.length_m + 1e-9);
        assert!(s.length_m() > 0.0);
        assert!(s.gradient() > 0.0);
    }

    #[test]
    fn the_tolerance_suppresses_sampling_wobble_but_not_a_real_climb() {
        // A 1 mm wobble on an otherwise-falling profile.
        let wobble: Vec<(f64, f64)> = (0..40)
            .map(|i| {
                let s = i as f64;
                let bump = if i % 2 == 0 { 0.02 } else { 0.0 };
                (s, 10.0 - s * 0.01 + bump)
            })
            .collect();
        assert!(uphill_spans(&wobble, 0.05).is_empty(), "wobble reported");
        assert!(!uphill_spans(&wobble, 0.0).is_empty(), "the wobble is real");

        // A long, gentle, genuinely-wrong climb: every step is smaller than the
        // tolerance, and the merged span is not.
        let creep: Vec<(f64, f64)> = (0..200).map(|i| (i as f64, i as f64 * 0.01)).collect();
        let spans = uphill_spans(&creep, 0.05);
        assert_eq!(spans.len(), 1);
        assert!((spans[0].rise_m - 1.99).abs() < 1e-9, "{:?}", spans[0]);
    }

    #[test]
    fn multiple_climbs_are_reported_separately() {
        let profile = vec![
            (0.0, 10.0),
            (1.0, 9.0),
            (2.0, 10.0), // +1
            (3.0, 8.0),
            (4.0, 6.0),
            (5.0, 8.5), // +2.5
            (6.0, 8.5), // flat closes the span
            (7.0, 7.0),
        ];
        let spans = uphill_spans(&profile, 0.1);
        assert_eq!(spans.len(), 2, "{spans:?}");
        assert!((spans[0].rise_m - 1.0).abs() < 1e-9);
        assert!((spans[1].rise_m - 2.5).abs() < 1e-9);
    }

    /// **The documented escape**, pinned so it stays documented.
    ///
    /// A sawtooth — sub-tolerance rises separated by falls — climbs a long way in
    /// total and is reported by nothing, because every span closes at the fall.
    /// The alternative (a net-elevation test) would fire on every river that
    /// crosses a ridge on its way down a valley, so this is the lesser gap, and it
    /// is a gap rather than a bug only because it is written down.
    #[test]
    fn a_sawtooth_climb_escapes_the_per_span_tolerance() {
        // +0.4 / -0.1, twenty times: a net climb of 6 m in 0.4 m bites.
        let mut profile = vec![(0.0, 0.0)];
        let mut y = 0.0;
        for i in 0..20 {
            y += 0.4;
            profile.push((i as f64 * 2.0 + 1.0, y));
            y -= 0.1;
            profile.push((i as f64 * 2.0 + 2.0, y));
        }
        let net = profile.last().unwrap().1 - profile[0].1;
        assert!(net > 5.0, "the fixture must really climb: {net}");
        assert!(
            uphill_spans(&profile, 0.5).is_empty(),
            "the sawtooth escape closed — if this is deliberate, update the docs              on `uphill_spans` and the cook's tolerance constant in the same commit"
        );
        // …and each individual span really is under the tolerance, which is WHY it
        // escapes — not because the merging is broken.
        let spans = uphill_spans(&profile, 0.0);
        assert_eq!(spans.len(), 20);
        assert!(spans.iter().all(|s| s.rise_m < 0.5));
    }

    #[test]
    fn the_bed_profile_skips_terrain_holes() {
        let path = RiverPath::from_points(
            &pts(&[[0.0, 5.0, 0.0], [100.0, 5.0, 0.0]]),
            false,
            SplineInterp::Linear,
            &RiverProfile::default(),
        );
        // A terrain with a hole over the middle third.
        let bed = path.bed_profile(|p| {
            if (33.0..66.0).contains(&p.x) {
                None
            } else {
                Some(1.0 + p.x * 0.01)
            }
        });
        assert!(!bed.is_empty());
        assert!(bed.len() < path.frames.len(), "the hole was not skipped");
        assert!(bed.iter().all(|&(_, y)| y.is_finite()));
        // The full profile is uphill (the terrain rises with +X) and the hole
        // does not fabricate a plunge to zero.
        let spans = uphill_spans(&bed, 0.05);
        assert_eq!(spans.len(), 1, "{spans:?}");
    }
}
