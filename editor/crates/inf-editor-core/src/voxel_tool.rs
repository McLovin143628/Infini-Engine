//! **The dig tools' policy** (P21.3): the pure rules that turn an author's
//! gesture into the shapes a dig cuts.
//!
//! # Why this is a Ring-1 module and not viewport code
//!
//! Verbatim the argument that put [`crate::terrain_stream`],
//! [`crate::render_assets`] and [`crate::voxel_store`] here, and the one the
//! P21.2 audit's **M11** ledger item names: `inf_viewport::host` is
//! `#[cfg(any(windows, target_os = "macos"))]`, so anything written there is
//! invisible to the Linux CI leg — the one a contributor's PR usually runs
//! first. A rule that decides *what geometry a click commits* is the last thing
//! that should be untested on a whole platform.
//!
//! So the shapes live here, pure and unit-tested, and the host keeps the input
//! plumbing: resolve a world point, call one of these, hand the result to
//! [`SceneDoc::edit_dig`](crate::scene::SceneDoc::edit_dig).
//!
//! # The three cuts, and what each one is for
//!
//! | cut | gesture | shape |
//! |---|---|---|
//! | **Box** | press-drag a rectangle on the surface | one axis-aligned [`VoxelShape::Box`] — the foundation-pit / parking-garage primitive |
//! | **Trench** | click waypoints, Ctrl+click to commit | one [`VoxelShape::Trench`] per leg — the utility-trench / road-cut primitive |
//! | **Brush** | press-drag freehand | a sphere per dab, or a *column* to grade in dig-to-depth mode |
//!
//! All three share the one rule that makes a pit read as a pit rather than as a
//! buried box: **the cut is open to the sky**. Its top is
//! [`BOX_CUT_TOP_MARGIN_M`] above the *highest* ground it spans, not above the
//! point the author happened to click, so a pit dragged across a slope has no
//! lid of surviving hillside over its uphill half. Its floor is the same margin
//! below the *lowest* ground it spans, so "3 m deep" means three metres below
//! grade everywhere rather than three metres below one corner.
//!
//! # Units
//!
//! World **metres**, `f64`, `y` up (architecture rules 3 and 6). Nothing here
//! reads the clock, the camera or the document — every function is a pure
//! function of its arguments, which is what lets the tests below be the gate
//! rather than a hardware pass.

use glam::DVec3;
use inf_voxel::VoxelShape;

/// How far above the highest ground a cut's top sits, metres.
///
/// The mouth rule ([`inf_voxel::coupling`]) opens a heightfield sample only when
/// the cut's signed distance there is **strictly** negative, so a box whose top
/// landed exactly on the highest sample would leave that sample closed — one
/// stubborn pixel of ground on the pit's rim. A quarter metre is comfortably
/// past any float slop and small enough that the cut does not visibly overshoot.
pub const BOX_CUT_TOP_MARGIN_M: f64 = 0.25;

/// Probe lattice resolution per axis for [`box_cut_plan`]'s surface scan.
///
/// **The documented limit of the pit's "open to the sky" rule.** The plan reads
/// the ground at `BOX_CUT_PROBES²` points across the rectangle; terrain that
/// rises between two probes by more than [`BOX_CUT_TOP_MARGIN_M`] can keep a
/// sliver of surface over the pit there. At 33 probes a 30 m pit samples every
/// 94 cm — finer than the metre-per-sample heightfields this engine authors — so
/// the case needs a deliberately spiky terrain to reach, and the fix is to
/// re-drag rather than to fail.
pub const BOX_CUT_PROBES: u32 = 33;

/// The smallest extent a cut may have on any axis, metres.
///
/// A press-and-release with no drag is a click, not a pit; the alternative is a
/// tool that digs a millimetre-wide slot every time an author misses a
/// selection.
pub const MIN_CUT_EXTENT_M: f64 = 0.05;

/// A **box cut** resolved against the ground it spans — a foundation pit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxCutPlan {
    /// The one op the cut makes.
    pub shape: VoxelShape,
    /// Pit floor, world metres — `depth_m` below the **lowest** ground spanned.
    pub floor_y: f64,
    /// Pit top, world metres — [`BOX_CUT_TOP_MARGIN_M`] above the **highest**.
    pub top_y: f64,
    /// Footprint along X, metres.
    pub size_x_m: f64,
    /// Footprint along Z, metres.
    pub size_z_m: f64,
}

impl BoxCutPlan {
    /// Footprint area, m².
    pub fn area_m2(&self) -> f64 {
        self.size_x_m * self.size_z_m
    }

    /// The pit's full vertical extent, metres (floor to top, margin included).
    pub fn height_m(&self) -> f64 {
        self.top_y - self.floor_y
    }
}

/// Resolve a **box cut** from the two corners of a surface drag.
///
/// `a` and `b` are world points on the ground (the drag's anchor and the
/// cursor); only their XZ decides the footprint. `surface(x, z)` answers the
/// ground height there, or `None` where this level has no terrain — which is a
/// legal answer, not a failure: a pit dragged over a hole, over the edge of the
/// world, or in a level with no terrain at all falls back to the two picks'
/// own heights, and those are points the author demonstrably aimed at.
///
/// `None` when the drag is smaller than [`MIN_CUT_EXTENT_M`] on either axis —
/// a click rather than a pit.
///
/// See the module docs for the open-to-the-sky rule and its documented limit.
pub fn box_cut_plan(
    a: DVec3,
    b: DVec3,
    depth_m: f64,
    surface: impl Fn(f64, f64) -> Option<f64>,
) -> Option<BoxCutPlan> {
    if !(a.is_finite() && b.is_finite() && depth_m.is_finite()) {
        return None;
    }
    let (lo_x, hi_x) = (a.x.min(b.x), a.x.max(b.x));
    let (lo_z, hi_z) = (a.z.min(b.z), a.z.max(b.z));
    let size_x_m = hi_x - lo_x;
    let size_z_m = hi_z - lo_z;
    if size_x_m < MIN_CUT_EXTENT_M || size_z_m < MIN_CUT_EXTENT_M {
        return None;
    }
    // Seeded with the two picks so the pit always contains the ground the author
    // actually aimed at, whatever the probe lattice finds.
    let mut min_h = a.y.min(b.y);
    let mut max_h = a.y.max(b.y);
    let steps = BOX_CUT_PROBES.max(2) - 1;
    for j in 0..=steps {
        let z = lo_z + size_z_m * j as f64 / steps as f64;
        for i in 0..=steps {
            let x = lo_x + size_x_m * i as f64 / steps as f64;
            let Some(h) = surface(x, z) else {
                continue;
            };
            if !h.is_finite() {
                continue;
            }
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }
    }
    let floor_y = min_h - depth_m.max(0.0);
    let top_y = max_h + BOX_CUT_TOP_MARGIN_M;
    let shape = VoxelShape::Box {
        center: DVec3::new(
            (lo_x + hi_x) * 0.5,
            (floor_y + top_y) * 0.5,
            (lo_z + hi_z) * 0.5,
        ),
        half_extents: DVec3::new(size_x_m * 0.5, (top_y - floor_y) * 0.5, size_z_m * 0.5),
    };
    if !shape.is_valid() {
        return None;
    }
    Some(BoxCutPlan {
        shape,
        floor_y,
        top_y,
        size_x_m,
        size_z_m,
    })
}

/// Resolve a **spline trench** from a path of surface waypoints: one swept
/// rectangle per leg.
///
/// Each leg's cross-section is `2·half_width_m` across and spans from `depth_m`
/// below the waypoints to [`BOX_CUT_TOP_MARGIN_M`] above them, so the trench is
/// open along its whole length.
///
/// **The section is perpendicular to the run, not to gravity.** A leg that dives
/// tilts its floor with it, which is what a road cut or a sewer fall actually
/// looks like — and it means a steeply-diving leg is slightly shallower measured
/// vertically than `depth_m`, by the cosine of its pitch. Documented rather than
/// corrected: correcting it would make the trench's walls non-planar at every
/// waypoint, and an author dragging a gradient wants a constant section.
///
/// Degenerate legs (a repeated waypoint) are dropped rather than emitted as
/// invalid shapes, so a double-click in the middle of a path costs nothing.
///
/// # The miter allowance
///
/// Two swept rectangles meeting at a bend leave a wedge of un-cut ground on the
/// **outside** of the corner, because each leg's end cap is square to its own
/// run. Every leg is therefore extended by `half_width_m` past both waypoints,
/// which closes the wedge for any deviation up to a right angle (the uncovered
/// overhang is `half_width · tan(θ/2)`, and `tan(45°) = 1`). Bends sharper than
/// that keep a small notch on the outside — a documented limit, not a silent
/// one, and the fix is a waypoint rather than a different primitive. The same
/// allowance is what stops the run looking clipped at the first and last click.
pub fn trench_shapes(path: &[DVec3], half_width_m: f64, depth_m: f64) -> Vec<VoxelShape> {
    let mut out = Vec::new();
    if path.len() < 2 || half_width_m.is_nan() || half_width_m <= 0.0 || half_width_m.is_infinite()
    {
        return out;
    }
    let depth = depth_m.max(0.0);
    let half_height_m = (depth + BOX_CUT_TOP_MARGIN_M) * 0.5;
    // The centreline sits half-way between the floor and the top, which is
    // `(margin − depth)/2` from the surface the author clicked.
    let shift = DVec3::new(0.0, (BOX_CUT_TOP_MARGIN_M - depth) * 0.5, 0.0);
    for seg in path.windows(2) {
        let (a, b) = (seg[0] + shift, seg[1] + shift);
        let run = b - a;
        let len = run.length();
        if len.is_nan() || len <= 0.0 || !run.is_finite() {
            continue;
        }
        let over = run / len * half_width_m;
        let shape = VoxelShape::Trench {
            a: a - over,
            b: b + over,
            half_width_m,
            half_height_m,
        };
        if shape.is_valid() {
            out.push(shape);
        }
    }
    out
}

/// The shape one **brush** dab makes at a surface pick.
///
/// Two modes, one function, because the brush is one tool:
///
/// * the default is a ball centred `depth_m` below the surface — P21.2's carve
///   brush, unchanged, and the right shape for hollowing a cave;
/// * **dig-to-depth** replaces it with a *column* from `depth_m` below the
///   surface to [`BOX_CUT_TOP_MARGIN_M`] above it — the "excavate to grade"
///   brush, which is what freehand digging of a pit actually wants. Every dab
///   reaches daylight, so a stroke leaves a trench with no roof rather than a
///   string of buried bubbles.
///
/// `surface` is the raw ground pick (not the sunk centre), because the column
/// needs to know where daylight is.
pub fn brush_dab_shape(
    surface: DVec3,
    radius_m: f64,
    depth_m: f64,
    dig_to_depth: bool,
) -> VoxelShape {
    let radius_m = radius_m.max(0.0);
    let depth = depth_m.max(0.0);
    if !dig_to_depth {
        return VoxelShape::Sphere {
            center: surface - DVec3::new(0.0, depth, 0.0),
            radius_m,
        };
    }
    let floor_y = surface.y - depth;
    let top_y = surface.y + BOX_CUT_TOP_MARGIN_M;
    VoxelShape::Box {
        center: DVec3::new(surface.x, (floor_y + top_y) * 0.5, surface.z),
        half_extents: DVec3::new(radius_m, (top_y - floor_y) * 0.5, radius_m),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-flat deterministic ground: polynomial, never trigonometric (this
    /// workspace's `std`-trig ban reaches into its fixtures — a fixture that
    /// drifts by a bit between platforms is a gate that fails on one of them).
    fn ground(x: f64, z: f64) -> Option<f64> {
        Some(4.0 + x * 0.25 - x * x * 0.01 + z * 0.1)
    }

    /// **The open-to-the-sky rule.** A pit dragged across rising ground is
    /// deeper than the drag's own span: its top clears the HIGHEST ground it
    /// covers and its floor is `depth` below the LOWEST.
    #[test]
    fn a_pit_spans_from_below_the_lowest_ground_to_above_the_highest() {
        let a = DVec3::new(0.0, ground(0.0, 0.0).unwrap(), 0.0);
        let b = DVec3::new(10.0, ground(10.0, 6.0).unwrap(), 6.0);
        let plan = box_cut_plan(a, b, 3.0, ground).expect("a real drag");

        // Independently: the extremes of the ground over the rectangle.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for j in 0..=200 {
            for i in 0..=200 {
                let h = ground(i as f64 * 0.05, j as f64 * 0.03).unwrap();
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        assert!(
            (plan.floor_y - (lo - 3.0)).abs() < 0.05,
            "floor {} vs {}",
            plan.floor_y,
            lo - 3.0
        );
        assert!(
            plan.top_y >= hi + BOX_CUT_TOP_MARGIN_M - 1e-9,
            "top {} does not clear the highest ground {hi}",
            plan.top_y
        );
        assert_eq!(plan.size_x_m, 10.0);
        assert_eq!(plan.size_z_m, 6.0);
        assert_eq!(plan.area_m2(), 60.0);
        assert!(plan.height_m() > 3.0);

        // The shape covers both corners' surface points — on the boundary,
        // because the rectangle's corners are exactly where the author dragged
        // to (the mouth rule opens strictly-inside samples, and the interior of
        // the rectangle is what a mouth is made of).
        for p in [a, b] {
            assert!(
                plan.shape.distance_m(p) <= 0.0,
                "the pit does not reach its own corner pick {p:?}"
            );
        }
        // …and the ground in the MIDDLE of the drag is strictly inside, even
        // though it is higher than either corner pick's own height.
        let mid = DVec3::new(5.0, ground(5.0, 3.0).unwrap(), 3.0);
        assert!(
            plan.shape.distance_m(mid) < 0.0,
            "the pit has a lid of surviving hillside over its middle"
        );
    }

    /// The drag's corners may arrive in any order, and the plan is the same pit.
    #[test]
    fn a_pit_is_the_same_whichever_corner_was_dragged_first() {
        let a = DVec3::new(-4.0, 3.0, 7.0);
        let b = DVec3::new(6.0, 5.0, -2.0);
        let one = box_cut_plan(a, b, 2.0, ground).unwrap();
        let two = box_cut_plan(b, a, 2.0, ground).unwrap();
        assert_eq!(one, two);
        // …and it is deterministic run to run.
        assert_eq!(one, box_cut_plan(a, b, 2.0, ground).unwrap());
    }

    /// A level with **no terrain** still digs: the two picks are the ground.
    #[test]
    fn a_pit_with_no_ground_under_it_falls_back_to_the_picks() {
        let a = DVec3::new(0.0, 10.0, 0.0);
        let b = DVec3::new(4.0, 12.0, 4.0);
        let plan = box_cut_plan(a, b, 1.5, |_, _| None).expect("still a pit");
        assert_eq!(plan.floor_y, 10.0 - 1.5);
        assert_eq!(plan.top_y, 12.0 + BOX_CUT_TOP_MARGIN_M);
        // A NaN from a broken height query is skipped, not folded in.
        let nanny = box_cut_plan(a, b, 1.5, |_, _| Some(f64::NAN)).expect("a pit");
        assert_eq!(nanny, plan);
    }

    /// A click (or a degenerate drag) is not a pit.
    #[test]
    fn a_click_digs_nothing() {
        let a = DVec3::new(1.0, 2.0, 3.0);
        assert!(box_cut_plan(a, a, 4.0, ground).is_none());
        assert!(box_cut_plan(a, a + DVec3::new(0.01, 0.0, 9.0), 4.0, ground).is_none());
        assert!(box_cut_plan(a, DVec3::splat(f64::NAN), 4.0, ground).is_none());
        assert!(box_cut_plan(a, a + DVec3::splat(5.0), f64::NAN, ground).is_none());
        // A zero-depth drag is still a pit — it skims the surface, which is a
        // legal (and useful) way to punch a mouth without hollowing anything.
        let flat = box_cut_plan(a, a + DVec3::new(5.0, 0.0, 5.0), 0.0, ground).unwrap();
        assert!(flat.height_m() > 0.0);
    }

    /// A trench is one swept rectangle per leg, each containing its own
    /// waypoints, open at the top and `depth_m` deep below them.
    #[test]
    fn a_trench_is_one_open_section_per_leg() {
        let path = [
            DVec3::new(0.0, 5.0, 0.0),
            DVec3::new(10.0, 5.0, 0.0),
            DVec3::new(10.0, 5.0, 10.0),
        ];
        let shapes = trench_shapes(&path, 1.5, 2.0);
        assert_eq!(shapes.len(), 2);
        for (n, s) in shapes.iter().enumerate() {
            assert!(s.is_valid(), "leg {n}");
            // The waypoints' own surface points are inside (so the mouth opens
            // along the whole run) …
            assert!(s.distance_m(path[n]) < 0.0, "leg {n} start");
            assert!(s.distance_m(path[n + 1]) < 0.0, "leg {n} end");
            // … the floor is `depth` down …
            let mid = (path[n] + path[n + 1]) * 0.5;
            assert!(
                s.distance_m(mid - DVec3::new(0.0, 1.9, 0.0)) < 0.0,
                "leg {n}"
            );
            assert!(
                s.distance_m(mid - DVec3::new(0.0, 2.6, 0.0)) > 0.0,
                "leg {n}"
            );
            // … and above the margin is open air.
            assert!(
                s.distance_m(mid + DVec3::new(0.0, 0.5, 0.0)) > 0.0,
                "leg {n}"
            );
        }
        // The trench really is `2·half_width` across, measured across the run.
        let across = path[0] + DVec3::new(0.0, 0.0, 1.4);
        assert!(shapes[0].distance_m(across) < 0.0);
        let outside = path[0] + DVec3::new(0.0, 0.0, 1.6);
        assert!(shapes[0].distance_m(outside) > 0.0);
    }

    /// Degenerate input produces no legs rather than invalid ones.
    #[test]
    fn a_degenerate_trench_path_produces_no_legs() {
        let p = DVec3::new(1.0, 2.0, 3.0);
        assert!(trench_shapes(&[], 1.0, 1.0).is_empty());
        assert!(trench_shapes(&[p], 1.0, 1.0).is_empty());
        assert!(
            trench_shapes(&[p, p], 1.0, 1.0).is_empty(),
            "a repeated point"
        );
        assert!(
            trench_shapes(&[p, p + DVec3::X], 0.0, 1.0).is_empty(),
            "no width"
        );
        // A path with one repeated point in the middle keeps its real legs.
        let legs = trench_shapes(&[p, p, p + DVec3::X * 5.0], 1.0, 1.0);
        assert_eq!(legs.len(), 1);
    }

    /// The brush's two modes: a ball at depth, or a column to grade.
    #[test]
    fn the_brush_digs_a_ball_or_a_column_to_grade() {
        let surface = DVec3::new(2.0, 8.0, -3.0);

        let ball = brush_dab_shape(surface, 1.5, 4.0, false);
        assert!(matches!(ball, VoxelShape::Sphere { .. }));
        assert_eq!(ball.center_m(), DVec3::new(2.0, 4.0, -3.0));
        assert!(
            ball.distance_m(surface) > 0.0,
            "a ball 4 m down must not reach the surface"
        );

        let column = brush_dab_shape(surface, 1.5, 4.0, true);
        assert!(matches!(column, VoxelShape::Box { .. }));
        // Open at the top …
        assert!(column.distance_m(surface) < 0.0);
        assert!(column.distance_m(surface + DVec3::new(0.0, 0.5, 0.0)) > 0.0);
        // … down to grade …
        assert!(column.distance_m(surface - DVec3::new(0.0, 3.9, 0.0)) < 0.0);
        assert!(column.distance_m(surface - DVec3::new(0.0, 4.6, 0.0)) > 0.0);
        // … and `radius` wide, not `radius` in a sphere's sense.
        assert!(column.distance_m(surface + DVec3::new(1.4, 0.0, 1.4)) < 0.0);
        assert!(column.distance_m(surface + DVec3::new(1.6, 0.0, 0.0)) > 0.0);

        // Degenerate settings stay valid-or-refused rather than producing NaNs.
        assert!(!brush_dab_shape(surface, 0.0, 1.0, false).is_valid());
        assert!(!brush_dab_shape(surface, 0.0, 1.0, true).is_valid());
        assert!(brush_dab_shape(surface, 1.0, -5.0, true).is_valid());
    }
}
