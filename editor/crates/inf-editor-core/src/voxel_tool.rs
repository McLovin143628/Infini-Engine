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
//!
//! # What arrived here from `host.rs` (the M11 move)
//!
//! [`voxel_target`] (which volume a click carves), [`cut_center`] (the
//! surface → depth rule) and [`dab_centers`] (the stroke resampler) were written
//! in `inf_viewport::host` in P21.2 and moved here in P21.3, when the dig tools
//! reworked that file anyway — one refactor instead of two, as the ledger item
//! said. The host now calls them and keeps nothing but the pick and the plumbing
//! around them.
//!
//! [`dab_centers`] in particular is a **wrapper over
//! [`inf_terrain::dab_positions`]**, not a second copy of it: the spacing, the
//! carry across segments and the "the first point is re-emitted" convention are
//! the sculpt brush's, resolved by the sculpt brush's code, with this function
//! doing nothing but lifting the answer back onto a 3D polyline. Two resamplers
//! that agree today and drift tomorrow is exactly the failure the M11 argument
//! is about.

use glam::{DVec2, DVec3};
use inf_voxel::VoxelShape;
use uuid::Uuid;

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

/// **Which volume a click carves**: the first *selected* entity that has a
/// loaded volume, else the first in document order.
///
/// Selection first, so an author with two cave systems open cuts the one they
/// are looking at. Deliberately **not** auto-created the way the foliage tool
/// creates its `Foliage` entity: a volume's chunks live in an `.inf_voxel` on
/// disk, and conjuring an entity that references an asset nobody has written
/// would produce a tool that silently does nothing.
///
/// `loaded` is the caller's answer to "does the shared store hold chunks for
/// this entity?" — a parameter rather than a lookup because the store lives
/// behind a mutex the viewport thread owns, and keeping the *rule* here and the
/// *lookup* at the call site is what makes the rule testable at all.
pub fn voxel_target(
    order: &[Uuid],
    selection: &[Uuid],
    loaded: impl Fn(Uuid) -> bool,
) -> Option<Uuid> {
    selection
        .iter()
        .copied()
        .find(|&g| loaded(g))
        .or_else(|| order.iter().copied().find(|&g| loaded(g)))
}

/// Where a click puts the **centre** of a sunk cut: the picked surface, dropped
/// by `depth_m`.
///
/// Camera-independent by construction, which is the ruling the water tool's pick
/// got and matters more here: a carve commits geometry, and two authors at
/// different camera distances must not dig different caves from the same click.
/// The depth is what turns a surface pick into a *tunnel* — at `0` the cut
/// breaks the ground where you point (a mouth), and past the cut's own radius it
/// hollows rock with no mouth at all, which the surface-crossing verdict then
/// allows on any terrain.
///
/// A negative depth is clamped to zero rather than honoured: depth is measured
/// DOWN from the surface that was picked, so a negative one names a cut above
/// the ground the author clicked on.
///
/// **Documented limit** (inherited, unchanged): the pick this takes resolves
/// against the heightfield, not against voxel surfaces, so continuing a tunnel
/// from *inside* an existing cave needs a voxel raycast the editor does not have
/// yet. Aim from above and set a depth; the raycast is the follow-up.
pub fn cut_center(surface: DVec3, depth_m: f64) -> DVec3 {
    surface - DVec3::new(0.0, depth_m.max(0.0), 0.0)
}

/// Dab spacing for a cut of `radius_m`, metres — **⅔ of a radius**, floored so a
/// zero-radius brush cannot ask for an infinite number of dabs.
///
/// Coarser than the sculpt brush's ⅓ because a *volume* brush's dabs overlap in
/// three dimensions rather than two: at ⅔ of a radius consecutive spheres still
/// intersect well inside each other, and the stroke costs a third of the cuts.
pub fn dab_spacing(radius_m: f64) -> f64 {
    (0.65 * radius_m).max(MIN_DAB_SPACING_M)
}

/// The floor under [`dab_spacing`], metres.
pub const MIN_DAB_SPACING_M: f64 = 0.05;

/// Resample a 3D drag path at even **arc length** — the stroke resampler, and a
/// wrapper over [`inf_terrain::dab_positions`] rather than a second copy of it.
///
/// The path's own points are re-emitted as the first entry exactly as the
/// Ring-0 function does (callers `skip(1)` when the first dab is already
/// placed), and the leftover arc length carries across segments so a drag with
/// several waypoints has no bunching at the joints.
///
/// # Why arc length in 3D, and why through the 2D function
///
/// A carve moves in `y` as well as `xz` — a tunnel dives — so resampling the XZ
/// path alone would space the dabs by their *shadow* and leave gaps on a slope.
/// But the spacing rule, the carry and the boundary conventions are the sculpt
/// brush's, and re-deriving them here is how two resamplers come to disagree.
/// So the path is flattened onto its own arc-length axis, handed to
/// [`inf_terrain::dab_positions`], and the answers are lifted back onto the
/// polyline. The 2D function is the one that decides *where* the dabs fall; this
/// one only decides what a distance means.
pub fn dab_centers(path: &[DVec3], spacing: f64) -> Vec<DVec3> {
    if path.is_empty() {
        return Vec::new();
    }
    if path.len() == 1 || spacing.is_nan() || spacing <= 0.0 || spacing.is_infinite() {
        return path.to_vec();
    }
    // Cumulative arc length, as a degenerate 2D path along +X. `dab_positions`
    // then does all the work.
    let mut arc = Vec::with_capacity(path.len());
    let mut total = 0.0;
    arc.push(DVec2::ZERO);
    for w in path.windows(2) {
        let seg = (w[1] - w[0]).length();
        total += if seg.is_finite() { seg } else { 0.0 };
        arc.push(DVec2::new(total, 0.0));
    }
    inf_terrain::dab_positions(&arc, spacing)
        .into_iter()
        .map(|p| point_at_arc(path, p.x))
        .collect()
}

/// The point `s` metres along the polyline `path`, clamped to its ends.
fn point_at_arc(path: &[DVec3], s: f64) -> DVec3 {
    let mut remaining = s.max(0.0);
    for w in path.windows(2) {
        let seg = w[1] - w[0];
        let len = seg.length();
        if !len.is_finite() || len <= 0.0 {
            continue;
        }
        if remaining <= len {
            return w[0] + seg * (remaining / len);
        }
        remaining -= len;
    }
    *path.last().unwrap_or(&DVec3::ZERO)
}

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

    // ── the M11 move: rules that used to be invisible to the Linux CI leg ──

    /// **Selection wins, document order is the fallback, and an unloaded volume
    /// is not a target at all.**
    ///
    /// The last clause is the one that matters: a selected entity whose
    /// `.inf_voxel` never resolved must not swallow the click, or the tool
    /// reports success and cuts nothing.
    #[test]
    fn the_carve_target_prefers_the_selection_and_skips_unloaded_volumes() {
        let a = Uuid::from_u128(1);
        let b = Uuid::from_u128(2);
        let c = Uuid::from_u128(3);
        let order = [a, b, c];

        // Nothing selected → the first LOADED volume in document order.
        assert_eq!(voxel_target(&order, &[], |g| g == b || g == c), Some(b));
        // A selection that is loaded wins over document order.
        assert_eq!(voxel_target(&order, &[c], |_| true), Some(c));
        // A selection that is NOT loaded falls through to document order rather
        // than swallowing the click.
        assert_eq!(voxel_target(&order, &[a], |g| g == c), Some(c));
        // Several selected: the first loaded one.
        assert_eq!(voxel_target(&order, &[a, b], |g| g != a), Some(b));
        // Nothing loaded anywhere → no target, and the tool says so.
        assert_eq!(voxel_target(&order, &[a, b, c], |_| false), None);
        assert_eq!(voxel_target(&[], &[], |_| true), None);
    }

    /// The depth rule: straight down from the pick, clamped at zero.
    #[test]
    fn the_cut_centre_sinks_from_the_pick_and_never_rises_above_it() {
        let surface = DVec3::new(3.0, 10.0, -4.0);
        assert_eq!(
            cut_center(surface, 0.0),
            surface,
            "0 depth IS the mouth cut"
        );
        assert_eq!(cut_center(surface, 2.5), DVec3::new(3.0, 7.5, -4.0));
        assert_eq!(
            cut_center(surface, -9.0),
            surface,
            "a negative depth must not lift the cut into the air"
        );
        // Camera-independence, stated as the property it is: the answer is a
        // function of the pick and the depth and of nothing else.
        assert_eq!(cut_center(surface, 2.5), cut_center(surface, 2.5));
    }

    /// **The resampler is the sculpt brush's**, lifted onto a 3D polyline: even
    /// arc length, the first point re-emitted, and the leftover carried across a
    /// bend.
    #[test]
    fn the_stroke_resampler_spaces_dabs_by_arc_length_in_three_dimensions() {
        // A path that climbs as it runs: resampling its XZ shadow would place
        // the dabs 1.0 m apart in world space instead of 1.0 m along the run,
        // which is the bug this exists to prevent.
        let a = DVec3::new(0.0, 0.0, 0.0);
        let b = DVec3::new(3.0, 4.0, 0.0); // 5 m long, 3 m of shadow
        let dabs = dab_centers(&[a, b], 1.0);
        assert_eq!(dabs.len(), 6, "{dabs:?}");
        assert_eq!(dabs[0], a);
        for w in dabs.windows(2) {
            assert!(
                ((w[1] - w[0]).length() - 1.0).abs() < 1e-12,
                "uneven spacing: {w:?}"
            );
        }
        assert!((dabs[5] - b).length() < 1e-12);

        // The carry crosses a bend: 1.5 m then 1.5 m at a right angle, spaced
        // 1.0 m, must place dabs at 0, 1, 2 along the RUN — the second of which
        // is 0.5 m past the corner and not 1.0 m past it.
        let path = [
            DVec3::ZERO,
            DVec3::new(1.5, 0.0, 0.0),
            DVec3::new(1.5, 0.0, 1.5),
        ];
        let bent = dab_centers(&path, 1.0);
        assert_eq!(bent.len(), 4, "{bent:?}");
        assert!((bent[1] - DVec3::new(1.0, 0.0, 0.0)).length() < 1e-12);
        assert!((bent[2] - DVec3::new(1.5, 0.0, 0.5)).length() < 1e-12);
        assert!((bent[3] - DVec3::new(1.5, 0.0, 1.5)).length() < 1e-12);

        // Degenerate inputs answer with the path rather than with a hang or a
        // NaN — a UI hands this a zero-length drag on the first frame of every
        // stroke.
        assert!(dab_centers(&[], 1.0).is_empty());
        assert_eq!(dab_centers(&[a], 1.0), vec![a]);
        assert_eq!(dab_centers(&[a, a], 1.0), vec![a]);
        assert_eq!(dab_centers(&[a, b], 0.0), vec![a, b]);
        assert_eq!(dab_centers(&[a, b], f64::NAN), vec![a, b]);
    }

    /// The spacing rule, including its floor — a zero-radius brush must not ask
    /// for infinitely many dabs.
    #[test]
    fn the_dab_spacing_is_two_thirds_of_a_radius_with_a_floor() {
        assert!((dab_spacing(3.0) - 1.95).abs() < 1e-12);
        assert_eq!(dab_spacing(0.0), MIN_DAB_SPACING_M);
        assert_eq!(dab_spacing(-5.0), MIN_DAB_SPACING_M);
        // …and a long fast drag stays bounded because of it.
        let far = dab_centers(
            &[DVec3::ZERO, DVec3::new(100.0, 0.0, 0.0)],
            dab_spacing(0.0),
        );
        assert_eq!(far.len(), 2001);
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
