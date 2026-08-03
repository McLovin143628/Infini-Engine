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

/// How far past an **open** river's end a point may still count as inside,
/// metres (P20.4).
///
/// The mouth plane is **inclusive**: a query exactly at the last frame answers
/// "inside", the same way [`RiverSample::bank_fraction`] `== 1.0` — exactly on
/// the bank — does. It needs a tolerance because the frames are a *resampling*
/// of the spline: the last one lands on the authored endpoint to within the
/// arc-length LUT's inversion error rather than exactly on it, so without one,
/// whether the water reaches its own mouth would depend on the last bit of that
/// inversion. A micrometre is four orders of magnitude below the finest authored
/// geometry (frames are spaced in metres) and several above the error.
pub const RIVER_END_TOLERANCE_M: f64 = 1e-6;

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

impl RiverProfile {
    /// **The one place authored river numbers are sanitized** (P20.4).
    ///
    /// Widths and depths are clamped to `>= 0` — a negative width is not a
    /// mirrored river, it is a ribbon whose banks have swapped, and a negative
    /// depth puts the bed above the surface. The flow speed is **not** clamped:
    /// a negative one reverses the river without re-authoring its spline, which
    /// is a documented `WaterBody` feature.
    ///
    /// Every consumer of a `WaterBody`'s river fields goes through here — both
    /// scene projectors, `PhysicsBridge3D`'s surface builder, the cook's
    /// advisories and the editor's river report. Before P20.4 the cook built its
    /// profile from the raw fields while everyone else clamped, so a negative
    /// authored depth produced a *different taper* in the build from the one the
    /// tool and the renderer showed — which breaks the one thing the tool's
    /// re-run of the cook checks is for: saying what the build will say.
    pub fn authored(
        width_start_m: f64,
        width_end_m: f64,
        depth_start_m: f64,
        depth_end_m: f64,
        flow_speed_m_s: f64,
    ) -> Self {
        Self {
            width_start_m: width_start_m.max(0.0),
            width_end_m: width_end_m.max(0.0),
            depth_start_m: depth_start_m.max(0.0),
            depth_end_m: depth_end_m.max(0.0),
            flow_speed_m_s,
        }
    }
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
    /// How far **past an end** of an open river the point lies, metres —
    /// measured along the end segment's own direction, and never negative
    /// (P20.4).
    ///
    /// `0` for every point whose projection lands inside the centreline's
    /// arc-length span, and **always `0` for a closed path**, which has no ends
    /// to be past. Positive only beyond the first frame or beyond the last one.
    ///
    /// This exists because [`s`](Self::s) cannot express it: `s` is clamped to
    /// `[0, length_m]` by construction, so a point thirty metres downstream of
    /// the mouth and a point exactly at the mouth report the same arc length.
    /// Without a second number, the only test [`inside`](Self::inside) could
    /// make was the *lateral* one — which is precisely how a boat came to float
    /// over dry land past a river's mouth (the P20.3 ledger's entry, closed
    /// here).
    pub beyond_m: f64,
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
    ///
    /// **Two bounds, not one** (P20.4): across the banks
    /// ([`bank_fraction`](Self::bank_fraction)) *and* along the centreline
    /// ([`beyond_m`](Self::beyond_m)). A ribbon is a bounded surface; testing
    /// only the lateral offset made an open river an infinite strip in the
    /// direction it points, so buoyancy, drag, swim and the water events all
    /// fired past its mouth for as far as the lateral test kept passing.
    ///
    /// Both bounds are inclusive at the edge — exactly on a bank is wet, exactly
    /// at the mouth is wet — see [`RIVER_END_TOLERANCE_M`] for why the second
    /// one carries a tolerance and the first does not.
    #[inline]
    pub fn inside(&self) -> bool {
        self.bank_fraction() <= 1.0 && self.beyond_m <= RIVER_END_TOLERANCE_M
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
    ///
    /// # The arc-length bound (P20.4)
    ///
    /// The projection onto each segment is **clamped**, so a point past the
    /// mouth necessarily lands on the last frame and reports `s == length_m`.
    /// The *unclamped* parameter of the winning segment is therefore kept, and
    /// the overshoot beyond the first or last segment of an **open** path is
    /// reported as [`RiverSample::beyond_m`]. It is only ever non-zero on the
    /// two end segments: a point beyond a hairpin's tip that happens to be
    /// nearest an interior segment is genuinely beside *that* stretch of river,
    /// and the lateral test is the right one for it.
    pub fn sample(&self, p: DVec2) -> Option<RiverSample> {
        if self.is_empty() {
            return None;
        }
        // (dist², segment, clamped u, raw u, segment length)
        let mut best: Option<(f64, usize, f64, f64, f64)> = None;
        for (i, pair) in self.frames.windows(2).enumerate() {
            let (a, b) = (pair[0], pair[1]);
            let a2 = DVec2::new(a.center.x, a.center.z);
            let b2 = DVec2::new(b.center.x, b.center.z);
            let ab = b2 - a2;
            let len2 = ab.length_squared();
            let raw = if len2 > 0.0 {
                (p - a2).dot(ab) / len2
            } else {
                0.0
            };
            let u = raw.clamp(0.0, 1.0);
            let closest = a2 + ab * u;
            let d2 = (p - closest).length_squared();
            if best.is_none_or(|(bd, _, _, _, _)| d2 < bd) {
                best = Some((d2, i, u, raw, len2.sqrt()));
            }
        }
        let (_, i, u, raw, seg_len) = best?;
        let a = self.frames[i];
        let b = self.frames[i + 1];
        let center = a.center.lerp(b.center, u);
        let tangent = norm_or(a.tangent.lerp(b.tangent, u), a.tangent);
        let right = norm_or(a.right.lerp(b.right, u), a.right);
        let lateral = (p - DVec2::new(center.x, center.z)).dot(DVec2::new(right.x, right.z));
        // A closed path wraps: its "ends" are joined, so there is nothing to be
        // past and the overshoot is identically zero.
        let last = self.frames.len() - 2;
        let beyond_m = if self.closed {
            0.0
        } else if i == 0 && raw < 0.0 {
            -raw * seg_len
        } else if i == last && raw > 1.0 {
            (raw - 1.0) * seg_len
        } else {
            0.0
        };
        Some(RiverSample {
            s: lerp(a.s, b.s, u),
            lateral_m: lateral,
            surface_y: center.y,
            width_m: lerp(a.width_m, b.width_m, u),
            depth_m: lerp(a.depth_m, b.depth_m, u),
            tangent,
            beyond_m,
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

    /// The **authored bed** profile: `(arc length, surface − depth)` per frame
    /// (P20.4).
    ///
    /// This is the bed the *author* described — the water surface lowered by the
    /// [`RiverProfile`]'s depth taper — and it is a different question from
    /// [`bed_profile`](Self::bed_profile), which asks the *terrain* how high the
    /// ground actually is. Neither is derived from the other, exactly as the
    /// shader's screen-space shore and `shore::shore_distance`'s world-space one
    /// are not.
    ///
    /// It is what the **cook** can check, because it needs no terrain at all: a
    /// river that descends 2 m over its length while its depth tapers from 5 m
    /// to 0.5 m has a bed that *climbs* 2.5 m, which is a basin, not a river —
    /// and nothing at runtime says so, because the surface still slopes the right
    /// way and the water still renders. Feed it to [`uphill_spans`] exactly like
    /// the surface profile.
    pub fn bed_profile_from_depth(&self) -> Vec<(f64, f64)> {
        self.frames
            .iter()
            .map(|f| (f.s, f.center.y - f.depth_m))
            .collect()
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

/// Flow accumulation, in **m³**, that drives the foam boost to its full value
/// (P20.4).
///
/// The unit is [`DataMapKind::Flow`](inf_terrain)'s own: the P19.1 erosion pass
/// integrates `dt · outflow` over the whole bake, so the map is a *volume* of
/// water that left each cell, peaking in the channels the water carved. The
/// value matches `mask.flow`'s default `max` in the PCG node kit (1000 m³), and
/// deliberately so — a flow value that reads as "a real channel" to a scatter
/// mask should read as one to a river, and two different ceilings for the same
/// map would be two different opinions about the same terrain.
pub const FLOW_FOAM_REFERENCE_M3: f64 = 1000.0;

/// How much a fully-channelled cell multiplies a river's flow foam by, on top of
/// the authored amount (P20.4).
///
/// `0.6` — a rapid over a carved channel foams a little over half again as much
/// as the same river crossing a plain. Chosen to be *visible* and not
/// *dominant*: the authored `foam_flow_m_s` is still what decides whether a river
/// foams at all.
pub const FLOW_FOAM_GAIN: f64 = 0.6;

/// The foam multiplier a river frame takes from the terrain's P19.1 flow map.
///
/// **It can only ever ADD foam.** The identity is `1.0`, returned for a frame
/// over terrain that was never eroded, over a hole in the heightfield, or over
/// no terrain at all — so wiring the flow map in changes *nothing* about content
/// that has no flow map, which is what let this ship without moving a single
/// golden. The upper bound is `1 + FLOW_FOAM_GAIN`.
///
/// A subtraction was the other candidate ("a river off-channel is glassy") and
/// was rejected: it makes the absence of a bake — the default state of every
/// terrain in the engine — into a visible change to every river already
/// authored, which is a migration disguised as a feature.
///
/// Pure, monotone, allocation-free and `f64`; no trig, so the P14 portability law
/// is satisfied trivially.
#[inline]
pub fn flow_foam_gain(flow_m3: f64) -> f64 {
    if !flow_m3.is_finite() || flow_m3 <= 0.0 {
        return 1.0;
    }
    let t = (flow_m3 / FLOW_FOAM_REFERENCE_M3).min(1.0);
    1.0 + FLOW_FOAM_GAIN * t
}

/// How much elevation a river must gain, in metres, before anything says so.
///
/// **One value, in Ring 0, because two callers need it and they must agree**:
/// the cook's advisory (`inf_packager::cook`) and the editor's river tool
/// (`inf_editor_core::hydro` via the `water_river_report` command). A tool with a
/// smaller tolerance would nag about rivers the build accepts; a larger one would
/// let a build advisory arrive as a surprise at package time. It lived in the
/// cook alone until P20.4 gave it a second reader.
///
/// **What it actually bounds.** Every profile it is applied to is a *resampling*
/// of the authored curve, not the authored data, and a resampling wobbles:
///
/// * **Catmull-Rom overshoot** — a smooth curve through strictly descending
///   control points can still bulge upward between knots, so a polyline an author
///   would call monotone shows centimetre rises;
/// * **arc-length quantization** — frames land at even distances, not on knots,
///   so the sampled extrema sit slightly off the real ones.
///
/// Half a metre is comfortably above both and comfortably below anything an
/// author would call a rise. It is a **merged-span** tolerance, applied to a
/// contiguous climb's total (see [`uphill_spans`]), so a long gentle ascent made
/// of individually tiny steps is still caught —
/// `a_sawtooth_climb_escapes_the_per_span_tolerance` documents the case it does
/// not catch.
///
/// The editor's *terrain-aware* checks use a larger one
/// (`inf_editor_core::hydro::BED_TOLERANCE_M`), because those additionally sample
/// a bilinear heightfield along a curve that crosses tile diagonals.
pub const UPHILL_TOLERANCE_M: f64 = 0.5;

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

    /// The one sanitizer: widths and depths floor at zero, the flow keeps its
    /// sign, and a clean profile passes through untouched.
    #[test]
    fn authored_profiles_are_clamped_in_exactly_one_place() {
        let p = RiverProfile::authored(-4.0, 12.0, -1.0, 3.0, -2.5);
        assert_eq!(p.width_start_m, 0.0);
        assert_eq!(p.width_end_m, 12.0);
        assert_eq!(p.depth_start_m, 0.0);
        assert_eq!(p.depth_end_m, 3.0);
        assert_eq!(p.flow_speed_m_s, -2.5, "a reversed river is not a mistake");
        // Anti-vacuity: a valid profile is passed through unchanged.
        let ok = RiverProfile::authored(6.0, 14.0, 1.2, 2.0, 2.2);
        assert_eq!(
            ok,
            RiverProfile {
                width_start_m: 6.0,
                width_end_m: 14.0,
                depth_start_m: 1.2,
                depth_end_m: 2.0,
                flow_speed_m_s: 2.2,
            }
        );
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

    /// **The river-mouth bug, closed (P20.4).**
    ///
    /// Before the arc-length bound, `inside()` tested only the lateral offset and
    /// `sample` clamped to the end segment — so every point on the centreline's
    /// extension, to infinity, answered "inside, at the mouth's level". A boat
    /// thirty metres past the mouth floated; swim and the water events fired over
    /// dry land.
    #[test]
    fn a_point_past_an_open_rivers_mouth_is_outside() {
        let path = RiverPath::from_points(
            &pts(&[[0.0, 5.0, 0.0], [100.0, 5.0, 0.0]]),
            false,
            SplineInterp::Linear,
            &RiverProfile {
                width_start_m: 10.0,
                width_end_m: 10.0,
                ..RiverProfile::default()
            },
        );
        // Thirty metres downstream of the mouth, dead on the centreline: the
        // lateral test passes and the arc-length one does not.
        let past = path.sample(DVec2::new(130.0, 0.0)).unwrap();
        assert!(past.bank_fraction() <= 1.0, "the lateral test still passes");
        assert!((past.beyond_m - 30.0).abs() < 1e-6, "{}", past.beyond_m);
        assert!(!past.inside(), "a boat past the mouth must not float");
        // …and the same distance BEFORE the source.
        let before = path.sample(DVec2::new(-30.0, 0.0)).unwrap();
        assert!((before.beyond_m - 30.0).abs() < 1e-6, "{}", before.beyond_m);
        assert!(!before.inside());

        // ANTI-VACUITY: the river did not get shorter. The mouth itself, and a
        // point a centimetre inside it, are still wet.
        let mouth = path.sample(DVec2::new(100.0, 0.0)).unwrap();
        assert_eq!(mouth.beyond_m, 0.0);
        assert!(mouth.inside(), "the mouth plane is inclusive");
        let inside = path.sample(DVec2::new(99.99, 2.0)).unwrap();
        assert!(inside.inside());
        assert_eq!(inside.beyond_m, 0.0);
        // The whole interior is unaffected.
        for i in 0..=100 {
            let s = path.sample(DVec2::new(i as f64, 0.0)).unwrap();
            assert!(s.inside(), "interior point {i} went dry");
        }
    }

    /// A **closed** path has no ends to be past, so the bound is inert on it —
    /// every point still answers by its lateral offset alone, and the loop's seam
    /// is not a wall.
    #[test]
    fn a_closed_river_has_no_mouth_to_be_past() {
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
            &RiverProfile {
                width_start_m: 8.0,
                width_end_m: 8.0,
                ..RiverProfile::default()
            },
        );
        // Walk the whole loop: nothing on it is ever "beyond" anything.
        for i in 0..200 {
            let s = path.length_m * i as f64 / 200.0;
            let f = &path.frames[(i * (path.frames.len() - 1) / 200).min(path.frames.len() - 1)];
            let p = DVec2::new(f.center.x, f.center.z);
            let smp = path.sample(p).unwrap();
            assert_eq!(smp.beyond_m, 0.0, "at s={s}");
            assert!(smp.inside());
        }
        // …and the centre of the loop, which is off the ribbon, is still out by
        // the LATERAL test — the bound did not become the only one.
        let centre = path.sample(DVec2::new(20.0, 20.0)).unwrap();
        assert_eq!(centre.beyond_m, 0.0);
        assert!(!centre.inside());
    }

    /// The bound is only ever applied on the two END segments. A hairpin's far
    /// arm passes back beside its own tip, and a point there is *beside a real
    /// stretch of river* — it must stay wet.
    #[test]
    fn a_hairpin_stays_wet_beside_its_own_far_arm() {
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
            &RiverProfile {
                width_start_m: 6.0,
                width_end_m: 6.0,
                ..RiverProfile::default()
            },
            32,
        );
        // A point on the RETURN arm, well past where the outbound arm's x ends —
        // nearest an interior segment, so no end bound applies.
        let smp = path.sample(DVec2::new(20.0, 8.0)).unwrap();
        assert_eq!(smp.beyond_m, 0.0);
        assert!(smp.inside(), "the return arm went dry");
        // Past the actual mouth (the far end of the return arm, at x ≈ 0, z = 8).
        let past = path.sample(DVec2::new(-25.0, 8.0)).unwrap();
        assert!(past.beyond_m > 20.0, "{}", past.beyond_m);
        assert!(!past.inside());
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

    /// **The authored-bed advisory (P20.4).** A river whose surface falls the
    /// whole way can still have a bed that climbs, because the depth taper is
    /// authored independently of the elevation — and nothing at runtime says so.
    #[test]
    fn a_tapering_depth_can_make_the_bed_climb_under_a_falling_surface() {
        // Surface: 10 m → 8 m over 150 m (falls 2 m). Depth: 5 m → 0.5 m.
        // Bed: 5 m → 7.5 m. It climbs 2.5 m.
        let gentle = pts(&[
            [0.0, 10.0, 0.0],
            [50.0, 9.33, 0.0],
            [100.0, 8.67, 0.0],
            [150.0, 8.0, 0.0],
        ]);
        let path = RiverPath::from_points(
            &gentle,
            false,
            SplineInterp::Linear,
            &RiverProfile {
                depth_start_m: 5.0,
                depth_end_m: 0.5,
                ..RiverProfile::default()
            },
        );
        // The SURFACE is clean — this is exactly the case the P20.1 advisory
        // cannot see.
        assert!(uphill_spans(&path.surface_profile(), 0.5).is_empty());
        let bed = path.bed_profile_from_depth();
        assert_eq!(bed.len(), path.frames.len());
        assert!((bed[0].1 - 5.0).abs() < 1e-9, "{:?}", bed[0]);
        assert!((bed.last().unwrap().1 - 7.5).abs() < 1e-9);
        let spans = uphill_spans(&bed, 0.5);
        assert_eq!(spans.len(), 1, "{spans:?}");
        assert!((spans[0].rise_m - 2.5).abs() < 0.01, "{:?}", spans[0]);

        // ANTI-VACUITY: the same river with a CONSTANT depth has a bed that
        // falls exactly as its surface does, and is reported by nothing.
        let level = RiverPath::from_points(
            &gentle,
            false,
            SplineInterp::Linear,
            &RiverProfile {
                depth_start_m: 2.0,
                depth_end_m: 2.0,
                ..RiverProfile::default()
            },
        );
        assert!(uphill_spans(&level.bed_profile_from_depth(), 0.5).is_empty());
    }

    /// The flow-map foam gain: identity where there is no flow, bounded above,
    /// monotone, and never *less* than the authored amount.
    #[test]
    fn the_flow_foam_gain_only_ever_adds() {
        assert_eq!(flow_foam_gain(0.0), 1.0);
        assert_eq!(flow_foam_gain(-5.0), 1.0, "a negative map is not a rebate");
        // A non-finite map value is not data; it reads as "no flow here" rather
        // than as an infinite rapid.
        assert_eq!(flow_foam_gain(f64::NAN), 1.0);
        assert_eq!(flow_foam_gain(f64::INFINITY), 1.0);
        // Saturates at the reference and never climbs past the bound.
        assert!((flow_foam_gain(FLOW_FOAM_REFERENCE_M3) - (1.0 + FLOW_FOAM_GAIN)).abs() < 1e-12);
        assert!((flow_foam_gain(1e9) - (1.0 + FLOW_FOAM_GAIN)).abs() < 1e-12);
        // Monotone non-decreasing, and strictly above 1 for any real flow.
        let mut prev = 1.0;
        for i in 0..=200 {
            let g = flow_foam_gain(i as f64 * 10.0);
            assert!(g >= prev - 1e-15, "not monotone at {i}");
            assert!((1.0..=1.0 + FLOW_FOAM_GAIN + 1e-12).contains(&g));
            prev = g;
        }
        assert!(flow_foam_gain(1.0) > 1.0, "any real flow adds something");
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
