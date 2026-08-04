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
//! lid of surviving hillside over its uphill half. Its floor is `depth` below
//! the *lowest* ground it spans, so "3 m deep" means three metres below grade
//! everywhere rather than three metres below one corner.
//!
//! The box and the trench each read that ground themselves, through a `surface`
//! closure the caller supplies ([`box_cut_plan`], [`trench_shapes`]); the brush
//! gets it from the pick under each dab. The trench's half of that arrived in
//! the P21.3 audit round — it had the docs and not the rule, and cut a run
//! roofed by any ridge between two waypoints.
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

/// The ground-probe **pitch**, metres — how finely a cut reads the surface it
/// has to clear.
///
/// # Pitch, not count (P21.3 audit)
///
/// This was a fixed *count* (33 per axis) documented as though it were a pitch,
/// which is only the same thing at one size: a 200 m pit probed every 6.25 m
/// misses a 0.6 m ridge and puts its roof 2.75 m too low, while the doc promised
/// "94 cm". A pitch keeps the **promise** — *the cut clears any ground feature
/// wider than [`GROUND_PROBE_PITCH_M`]* — at every size an author can drag, and
/// the count follows from the drag rather than the other way round.
///
/// Half a metre is finer than the metre-per-sample heightfields this engine
/// authors, so on a level whose terrain the editor produced the scan is exact:
/// every heightfield sample under the cut is probed at least once.
pub const GROUND_PROBE_PITCH_M: f64 = 0.5;

/// Ceiling on probes per axis, so a kilometre-wide drag cannot ask for a
/// million-point scan.
///
/// **This is where the guarantee above stops, and it is the honest statement of
/// the limit**: past `MAX_GROUND_PROBES · GROUND_PROBE_PITCH_M` = 64 m on an
/// axis the pitch coarsens in proportion, and a feature narrower than the
/// coarsened pitch can be missed. A 200 m pit therefore probes every 1.56 m, not
/// every 6.25 m as the fixed count gave — four times finer, and still a stated
/// limit rather than a promise.
pub const MAX_GROUND_PROBES: u32 = 129;

/// Ground extremes over an axis-aligned rectangle, seeded with the caller's own
/// picks so a cut always contains the points the author actually aimed at.
///
/// Returns `(lowest, highest)`. Probes at [`GROUND_PROBE_PITCH_M`] up to
/// [`MAX_GROUND_PROBES`] per axis; `None` answers (no terrain, a hole, past the
/// world's edge) are skipped rather than folded in, which is what lets a pit be
/// dragged over a cave mouth or off the edge of the world.
pub fn ground_extremes(
    lo: DVec2,
    hi: DVec2,
    seed: (f64, f64),
    surface: &impl Fn(f64, f64) -> Option<f64>,
) -> (f64, f64) {
    let (mut min_h, mut max_h) = seed;
    let (nx, nz) = (probe_steps(hi.x - lo.x), probe_steps(hi.y - lo.y));
    for j in 0..=nz {
        let z = lo.y + (hi.y - lo.y) * j as f64 / nz as f64;
        for i in 0..=nx {
            let x = lo.x + (hi.x - lo.x) * i as f64 / nx as f64;
            let Some(h) = surface(x, z) else { continue };
            if !h.is_finite() {
                continue;
            }
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }
    }
    (min_h, max_h)
}

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
    dab_centers_capped(path, spacing, usize::MAX)
}

/// [`dab_centers`], **bounded before it allocates** — the door an interactive
/// brush uses (P21.3 audit).
///
/// A per-frame `.take(n)` on the full list is a *filter*, not a bound: the whole
/// list is built first. At the [`MIN_DAB_SPACING_M`] floor a drag whose pick
/// landed far away — and `pick_world_point` admits a ray parameter out to 1e6 —
/// asks for two million points, eighty megabytes and twenty-seven milliseconds,
/// every frame, to keep thirty-two of them. So the *path* is trimmed to
/// `max_dabs · spacing` of arc length first, and the resampler never sees the
/// rest. The remainder rides to the next frame from the last dab actually
/// placed, exactly as it did before, so a long drag is still continuous.
///
/// Returns at most `max_dabs + 1` points (the first is the path's own start,
/// which callers `skip`).
pub fn dab_centers_capped(path: &[DVec3], spacing: f64, max_dabs: usize) -> Vec<DVec3> {
    if path.is_empty() {
        return Vec::new();
    }
    if path.len() == 1 || spacing.is_nan() || spacing <= 0.0 || spacing.is_infinite() {
        return path.to_vec();
    }
    // Trim the path to the arc length the cap allows, BEFORE resampling it.
    let total: f64 = path
        .windows(2)
        .map(|w| (w[1] - w[0]).length())
        .filter(|d| d.is_finite())
        .sum();
    let allowed = (max_dabs as f64).saturating_mul_f64(spacing);
    if allowed < total {
        let end = point_at_arc(path, allowed);
        let mut trimmed: Vec<DVec3> = Vec::new();
        let mut walked = 0.0;
        trimmed.push(path[0]);
        for w in path.windows(2) {
            let seg = (w[1] - w[0]).length();
            if !seg.is_finite() || seg <= 0.0 {
                continue;
            }
            if walked + seg >= allowed {
                break;
            }
            walked += seg;
            trimmed.push(w[1]);
        }
        trimmed.push(end);
        return dab_centers_unbounded(&trimmed, spacing);
    }
    dab_centers_unbounded(path, spacing)
}

/// A helper so `usize::MAX * spacing` cannot become `inf` and then `NaN`.
trait SaturatingMulF64 {
    fn saturating_mul_f64(self, rhs: f64) -> f64;
}
impl SaturatingMulF64 for f64 {
    fn saturating_mul_f64(self, rhs: f64) -> f64 {
        let v = self * rhs;
        if v.is_finite() {
            v
        } else {
            f64::MAX
        }
    }
}

fn dab_centers_unbounded(path: &[DVec3], spacing: f64) -> Vec<DVec3> {
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
    let (min_h, max_h) = ground_extremes(
        DVec2::new(lo_x, lo_z),
        DVec2::new(hi_x, hi_z),
        (a.y.min(b.y), a.y.max(b.y)),
        &surface,
    );
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
/// rectangle per leg, **each one open to the sky over the ground it spans**.
///
/// # The sky rule, which this had to earn (P21.3 audit)
///
/// The first version took no `surface` closure at all: each leg's roof sat
/// `BOX_CUT_TOP_MARGIN_M` above the straight chord between its two waypoints,
/// which is fine on a plane and wrong on anything else. A 1.5 m ridge mid-run
/// left 11 of 51 ground samples outside the cut, so the coupling never punched
/// their holes and the trench came out roofed by the hillside it was supposed
/// to open. Three doc blocks and a toolbar tooltip promised a rule the code did
/// not implement.
///
/// So each leg now reads the ground **over its own footprint** — a rotated
/// rectangle, probed along the run and across the width at
/// [`GROUND_PROBE_PITCH_M`] — and spans from `depth_m` below the lowest of it to
/// [`BOX_CUT_TOP_MARGIN_M`] above the highest. Exactly the box cut's rule, in
/// the leg's own frame.
///
/// # Legs are HORIZONTAL, and that is the design
///
/// The vertical span above is an absolute world range, so each leg's centreline
/// is level and only its *yaw* follows the path. Three things fall out, all of
/// them wanted:
///
/// * a trench is an **open cut from the surface**, so it follows the ground by
///   spanning it, not by tilting into it — the diving primitive is the tunnel;
/// * the "a diving leg is shallower vertically by the cosine of its pitch"
///   caveat this function used to carry simply stops existing;
/// * and so does a real bug behind it — a *vertical* centreline shift applied to
///   a pitched leg leaks into the along-run axis, which put the first waypoint
///   **outside its own leg** (`sdf(A) = +0.33` on a 45°, 4 m leg).
///
/// A long run across a big elevation change therefore becomes a tall box and
/// over-digs its low end, exactly as the box cut does over the same ground; the
/// gesture that fixes it is another waypoint, which is the gesture an author
/// already has.
///
/// Degenerate legs (a repeated waypoint) are dropped rather than emitted as
/// invalid shapes, so a double-click in the middle of a path costs nothing.
///
/// # The miter allowance
///
/// Two swept rectangles meeting at a bend leave a wedge of un-cut ground on the
/// **outside** of the corner, because each leg's end cap is square to its own
/// run. Every leg is therefore extended by `half_width_m` past both waypoints.
/// Measured over deviations from 10 to 170 degrees, the uncut wedge is
/// **0.0000 m**: the allowance closes the corner completely, not merely up to a
/// right angle as an earlier note here guessed. It is also what stops the run
/// looking clipped at the first and last click.
pub fn trench_shapes(
    path: &[DVec3],
    half_width_m: f64,
    depth_m: f64,
    surface: impl Fn(f64, f64) -> Option<f64>,
) -> Vec<VoxelShape> {
    let mut out = Vec::new();
    if path.len() < 2 || half_width_m.is_nan() || half_width_m <= 0.0 || half_width_m.is_infinite()
    {
        return out;
    }
    let depth = depth_m.max(0.0);
    for seg in path.windows(2) {
        let (p, q) = (seg[0], seg[1]);
        if !(p.is_finite() && q.is_finite()) {
            continue;
        }
        // Yaw only: the leg's direction is its HORIZONTAL run.
        let run = DVec2::new(q.x - p.x, q.z - p.z);
        let len = run.length();
        if len.is_nan() || len <= 0.0 || !len.is_finite() {
            continue;
        }
        let dir = run / len;
        let over = dir * half_width_m;
        let a2 = DVec2::new(p.x, p.z) - over;
        let b2 = DVec2::new(q.x, q.z) + over;
        let (min_h, max_h) =
            leg_ground_extremes(a2, b2, half_width_m, (p.y.min(q.y), p.y.max(q.y)), &surface);
        let floor_y = min_h - depth;
        let top_y = max_h + BOX_CUT_TOP_MARGIN_M;
        let mid_y = (floor_y + top_y) * 0.5;
        let shape = VoxelShape::Trench {
            a: DVec3::new(a2.x, mid_y, a2.y),
            b: DVec3::new(b2.x, mid_y, b2.y),
            half_width_m,
            half_height_m: (top_y - floor_y) * 0.5,
        };
        if shape.is_valid() {
            out.push(shape);
        }
    }
    out
}

/// Resolve a **spline tunnel** from a path of waypoints: one swept sphere
/// (capsule) per leg — a round bore at depth.
///
/// Capsules and not spheres: a chain of spheres at waypoint spacing leaves gaps
/// between the beads, and `VoxelShape::Capsule` exists precisely for this.
///
/// The trench's twin, and its opposite in every way that matters: a tunnel's
/// waypoints are its **centreline** (the caller has already sunk them by the
/// tool's depth), it dives freely, and it reads no ground at all — a bore below
/// the surface needs no sky rule, and one that breaks out of a hillside is
/// handled by the surface-cut verdict rather than by its shape.
///
/// Moved out of `inf_viewport::host` in the P21.3 audit round (the M11 item's
/// last inline shape): what geometry a Ctrl+click commits is a rule, and rules
/// do not live in a `#[cfg]`-gated module.
pub fn tunnel_shapes(path: &[DVec3], radius_m: f64) -> Vec<VoxelShape> {
    let mut out = Vec::new();
    if path.len() < 2 || radius_m.is_nan() || radius_m <= 0.0 || radius_m.is_infinite() {
        return out;
    }
    for seg in path.windows(2) {
        let shape = VoxelShape::Capsule {
            a: seg[0],
            b: seg[1],
            radius_m,
        };
        if shape.is_valid() {
            out.push(shape);
        }
    }
    out
}

/// Resample a **2D** drag path at even arc length, bounded before it allocates —
/// the terrain brushes' resampler.
///
/// [`dab_centers_capped`]'s twin for the sculpt / splat / biome strokes, which
/// work in terrain-local XZ. It lives beside its 3D sibling so there is **one**
/// stroke-resampling policy in the workspace rather than one per tool: both wrap
/// [`inf_terrain::dab_positions`], both carry the leftover across segments, and
/// both bound the path rather than filtering the result.
///
/// The terrain brushes had **no per-frame cap at all** before the P21.3 audit —
/// only the carve brush did — so a sculpt drag whose pick landed far away built
/// the whole list every frame and laid every dab in it.
pub fn dab_centers_2d_capped(path: &[DVec2], spacing: f64, max_dabs: usize) -> Vec<DVec2> {
    if path.is_empty() {
        return Vec::new();
    }
    if path.len() == 1 || spacing.is_nan() || spacing <= 0.0 || spacing.is_infinite() {
        return path.to_vec();
    }
    let total: f64 = path
        .windows(2)
        .map(|w| (w[1] - w[0]).length())
        .filter(|d| d.is_finite())
        .sum();
    let allowed = {
        let v = max_dabs as f64 * spacing;
        if v.is_finite() {
            v
        } else {
            f64::MAX
        }
    };
    if allowed < total {
        // Trim to the allowed arc length, then resample the short path.
        let mut walked = 0.0;
        let mut trimmed: Vec<DVec2> = vec![path[0]];
        let mut end = path[0];
        for w in path.windows(2) {
            let seg = (w[1] - w[0]).length();
            if !seg.is_finite() || seg <= 0.0 {
                continue;
            }
            if walked + seg >= allowed {
                end = w[0] + (w[1] - w[0]) / seg * (allowed - walked);
                break;
            }
            walked += seg;
            trimmed.push(w[1]);
            end = w[1];
        }
        trimmed.push(end);
        return inf_terrain::dab_positions(&trimmed, spacing);
    }
    inf_terrain::dab_positions(path, spacing)
}

/// Ground extremes over one trench leg's **rotated** footprint: the rectangle
/// `a2 -> b2` widened by `half_width_m` on both sides.
///
/// Probed in the leg's own frame rather than over its world AABB. The AABB of a
/// diagonal leg is up to twice its area, and folding a hill *beside* the trench
/// into `max_h` would raise the roof (harmless) while folding a valley beside it
/// into `min_h` would deepen the floor along the whole run (not harmless).
fn leg_ground_extremes(
    a2: DVec2,
    b2: DVec2,
    half_width_m: f64,
    seed: (f64, f64),
    surface: &impl Fn(f64, f64) -> Option<f64>,
) -> (f64, f64) {
    let (mut min_h, mut max_h) = seed;
    let run = b2 - a2;
    let len = run.length();
    if len.is_nan() || len <= 0.0 || len.is_infinite() {
        return (min_h, max_h);
    }
    let dir = run / len;
    let perp = DVec2::new(-dir.y, dir.x);
    let (na, nw) = (probe_steps(len), probe_steps(half_width_m * 2.0));
    for i in 0..=na {
        let along = a2 + dir * (len * i as f64 / na as f64);
        for j in 0..=nw {
            let s = -half_width_m + 2.0 * half_width_m * j as f64 / nw as f64;
            let p = along + perp * s;
            let Some(h) = surface(p.x, p.y) else { continue };
            if !h.is_finite() {
                continue;
            }
            min_h = min_h.min(h);
            max_h = max_h.max(h);
        }
    }
    (min_h, max_h)
}

/// Probe intervals across `span` metres: one per [`GROUND_PROBE_PITCH_M`], never
/// more than [`MAX_GROUND_PROBES`] - 1 and never fewer than one.
fn probe_steps(span: f64) -> u32 {
    let n = (span / GROUND_PROBE_PITCH_M).ceil();
    if !n.is_finite() || n < 1.0 {
        1
    } else {
        (n as u32).min(MAX_GROUND_PROBES - 1)
    }
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

    /// A deterministic ground with a **RIDGE and a HOLLOW inside** the fixtures'
    /// rectangles — polynomial, never trigonometric (this workspace's `std`-trig
    /// ban reaches into its fixtures; a fixture that drifts by a bit between
    /// platforms is a gate that fails on one of them).
    ///
    /// # Why the old fixture made the gate vacuous (P21.3 audit, B4)
    ///
    /// It was `4 + 0.25x − 0.01x² + 0.1z`, whose turning point is `x = 12.5` —
    /// **outside** the `[0, 10] × [0, 6]` rectangle the test dragged. Monotone
    /// over its own domain, so its extremes were the two corners, which
    /// `box_cut_plan` already seeds before it probes anything. The whole probe
    /// loop could be deleted, or `BOX_CUT_PROBES` dropped from 33 to 2, and the
    /// test still passed — while its own comment claimed it was checking that
    /// the middle of the drag is higher than either corner.
    ///
    /// This one has a genuine interior maximum at `(5, 3)` and a genuine
    /// interior minimum at `(2, 4.5)`, both strictly inside the rectangle and
    /// both invisible from the corners. `RIDGE_XZ` / `HOLLOW_XZ` name them so a
    /// reader can check the claim without solving the polynomial.
    fn ground(x: f64, z: f64) -> Option<f64> {
        // A bump at (5, 3) and a dip at (2, 4.5), on a gentle background slope.
        let bump = 3.0 * bell(x - 5.0, 2.5) * bell(z - 3.0, 2.0);
        let dip = -2.0 * bell(x - 2.0, 1.5) * bell(z - 4.5, 1.5);
        Some(4.0 + x * 0.05 + z * 0.03 + bump + dip)
    }

    /// A smooth, compactly-supported bump: `(1 − (t/r)²)²` inside `|t| < r`, else
    /// `0`. Polynomial, so it is bit-identical everywhere.
    fn bell(t: f64, r: f64) -> f64 {
        let u = t / r;
        if u.abs() >= 1.0 {
            0.0
        } else {
            let v = 1.0 - u * u;
            v * v
        }
    }

    /// The fixture's interior maximum, and its interior minimum.
    const RIDGE_XZ: (f64, f64) = (5.0, 3.0);
    const HOLLOW_XZ: (f64, f64) = (2.0, 4.5);

    /// **The open-to-the-sky rule.** A pit dragged across rising ground is
    /// deeper than the drag's own span: its top clears the HIGHEST ground it
    /// covers and its floor is `depth` below the LOWEST.
    #[test]
    fn a_pit_spans_from_below_the_lowest_ground_to_above_the_highest() {
        let a = DVec3::new(0.0, ground(0.0, 0.0).unwrap(), 0.0);
        let b = DVec3::new(10.0, ground(10.0, 6.0).unwrap(), 6.0);
        let plan = box_cut_plan(a, b, 3.0, ground).expect("a real drag");

        // The fixture must actually have interior extrema, or everything below
        // is a test of two corners (the P21.3 audit's B4).
        let ridge_h = ground(RIDGE_XZ.0, RIDGE_XZ.1).unwrap();
        let hollow_h = ground(HOLLOW_XZ.0, HOLLOW_XZ.1).unwrap();
        assert!(
            ridge_h > a.y.max(b.y) + 1.0,
            "the fixture ridge ({ridge_h:.2}) is not above both corner picks — \
             seeding alone would find it"
        );
        assert!(
            hollow_h < a.y.min(b.y) - 0.5,
            "the fixture hollow ({hollow_h:.2}) is not below both corner picks"
        );

        // Independently: the extremes of the ground over the rectangle, on a
        // lattice far finer than the one under test.
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for j in 0..=600 {
            for i in 0..=1000 {
                let h = ground(i as f64 * 0.01, j as f64 * 0.01).unwrap();
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        assert!(
            (plan.floor_y - (lo - 3.0)).abs() < 0.05,
            "floor {} vs {} — the probe did not find the HOLLOW",
            plan.floor_y,
            lo - 3.0
        );
        assert!(
            plan.top_y >= hi + BOX_CUT_TOP_MARGIN_M - 0.05,
            "top {} does not clear the highest ground {hi} — the probe did not find the RIDGE",
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
        // …and the ground ON THE RIDGE is strictly inside, though it is a metre
        // above either corner pick and nothing but a probe could have found it.
        let crest = DVec3::new(RIDGE_XZ.0, ridge_h, RIDGE_XZ.1);
        assert!(
            plan.shape.distance_m(crest) < 0.0,
            "the pit has a lid of surviving hillside over its ridge"
        );
        // …and the floor really is `depth` below the HOLLOW, not below a corner.
        assert!(
            plan.floor_y < hollow_h - 3.0 + 0.05,
            "the floor stops short of 3 m below the hollow ({hollow_h:.2})"
        );
    }

    /// **B4's mutation guard, stated as a test.** The probe lattice has to be
    /// fine enough to *find* the fixture's ridge, and the fixture's ridge has to
    /// be narrow enough that a coarse lattice misses it — so a coarsened pitch
    /// really does produce a shallower roof.
    ///
    /// Without this the "pitch, not count" change would be untested: a rule that
    /// probes at 0.5 m and one that probes at 6 m both pass every other
    /// assertion in this file on a wide enough bump.
    #[test]
    fn a_coarse_probe_lattice_would_miss_the_ridge() {
        let a = DVec3::new(0.0, ground(0.0, 0.0).unwrap(), 0.0);
        let b = DVec3::new(10.0, ground(10.0, 6.0).unwrap(), 6.0);
        let fine = box_cut_plan(a, b, 3.0, ground).unwrap();

        // The same rectangle read at the pitch the old fixed COUNT gave a 200 m
        // pit (6.25 m): three probes across this drag, none of them on the
        // ridge.
        let coarse = {
            let (mut lo, mut hi) = (a.y.min(b.y), a.y.max(b.y));
            for j in 0..=1 {
                for i in 0..=1 {
                    let h = ground(i as f64 * 10.0, j as f64 * 6.0).unwrap();
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            hi
        };
        assert!(
            fine.top_y > coarse + BOX_CUT_TOP_MARGIN_M + 0.5,
            "a corners-only read ({coarse:.2}) reaches the same roof as the probed one \
             ({:.2}) — the fixture's ridge is invisible to the probe and B4 is back",
            fine.top_y
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
        let flat = |_: f64, _: f64| Some(5.0);
        let path = [
            DVec3::new(0.0, 5.0, 0.0),
            DVec3::new(10.0, 5.0, 0.0),
            DVec3::new(10.0, 5.0, 10.0),
        ];
        let shapes = trench_shapes(&path, 1.5, 2.0, flat);
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

    /// **THE TRENCH SKY RULE** (P21.3 audit, B2). A ridge between two waypoints
    /// must be inside the cut, or the coupling never punches its holes and the
    /// trench comes out roofed by the hillside it was supposed to open.
    ///
    /// The measurement the audit made: with the roof pinned to the straight
    /// chord, 11 of 51 ground samples along a run over a 1.5 m ridge fell
    /// outside the leg. This asserts **zero**, and then asserts that the
    /// chord-roofed version really would have failed — so the gate cannot pass
    /// by the ridge being too small to matter.
    #[test]
    fn a_trench_clears_a_ridge_between_its_waypoints() {
        // Flat at y = 5 except for a 1.5 m ridge across the middle of the run.
        let ridge = |x: f64, _z: f64| Some(5.0 + 1.5 * bell(x - 5.0, 2.0));
        let a = DVec3::new(0.0, 5.0, 0.0);
        let b = DVec3::new(10.0, 5.0, 0.0);
        let legs = trench_shapes(&[a, b], 1.0, 2.0, ridge);
        assert_eq!(legs.len(), 1);
        let leg = legs[0];

        let mut outside = 0;
        let mut samples = 0;
        for i in 0..=50 {
            let x = i as f64 * 0.2;
            let p = DVec3::new(x, ridge(x, 0.0).unwrap(), 0.0);
            samples += 1;
            if leg.distance_m(p) >= 0.0 {
                outside += 1;
            }
        }
        assert_eq!(
            outside, 0,
            "{outside}/{samples} ground samples are outside the trench — a roofed run"
        );

        // Non-vacuity: the OLD rule (roof `margin` above the chord between the
        // waypoints, both at y = 5) really does leave the crest outside.
        let crest = DVec3::new(5.0, ridge(5.0, 0.0).unwrap(), 0.0);
        assert!(
            crest.y > 5.0 + BOX_CUT_TOP_MARGIN_M,
            "the fixture ridge ({:.2}) does not even clear the old roof — this proves nothing",
            crest.y
        );
        // …and the floor is still `depth` below the LOWEST ground, not below the
        // ridge: a trench does not get shallower because it crossed a hill.
        assert!(
            leg.distance_m(DVec3::new(0.0, 5.0 - 1.9, 0.0)) < 0.0,
            "the floor rose with the ridge"
        );
    }

    /// **Legs are horizontal, and the first waypoint is inside its own leg**
    /// (P21.3 audit). A vertical centreline shift on a pitched leg leaked into
    /// the along-run axis and pushed the start cap past the waypoint —
    /// `sdf(A) = +0.33` on a 45°, 4 m leg. Yaw-only legs cannot express that.
    #[test]
    fn a_steep_run_still_contains_its_own_waypoints() {
        // A 45° run: 4 m across, 4 m down.
        let a = DVec3::new(0.0, 10.0, 0.0);
        let b = DVec3::new(4.0, 6.0, 0.0);
        let ramp = |x: f64, _z: f64| Some(10.0 - x.clamp(0.0, 4.0));
        let legs = trench_shapes(&[a, b], 1.0, 2.0, ramp);
        assert_eq!(legs.len(), 1);
        for (label, p) in [("start", a), ("end", b)] {
            assert!(
                legs[0].distance_m(p) < 0.0,
                "the {label} waypoint is OUTSIDE its own leg (sdf = {:.3})",
                legs[0].distance_m(p)
            );
        }
        // …and the whole ramp between them is covered, not just its ends.
        for i in 0..=40 {
            let x = i as f64 * 0.1;
            let p = DVec3::new(x, ramp(x, 0.0).unwrap(), 0.0);
            assert!(legs[0].distance_m(p) < 0.0, "the ramp at x={x} is roofed");
        }
    }

    /// **The miter allowance closes the corner completely** — measured, not
    /// assumed (P21.3 audit).
    ///
    /// An earlier note here guessed the allowance only covered bends up to a
    /// right angle and documented a "small notch" past that. The audit measured
    /// 0.0000 m of uncut wedge from 10 degrees to 170; this pins it, so the
    /// guess cannot come back and neither can a regression that makes it true.
    ///
    /// The sweep is over the bend's OUTSIDE, which is where a wedge would be:
    /// for each deviation, a dense fan of points on the corner's outer side is
    /// tested against the union of the two legs.
    #[test]
    fn the_miter_allowance_leaves_no_uncut_wedge_at_any_bend() {
        let flat = |_: f64, _: f64| Some(0.0);
        let half_width = 1.0;
        // Deviations from a gentle 10 degrees to a hairpin 170, as directions
        // built without `std` trig (a unit vector from a rational slope).
        let dirs: [(f64, f64); 9] = [
            (0.985, 0.174),
            (0.940, 0.342),
            (0.866, 0.500),
            (0.707, 0.707),
            (0.500, 0.866),
            (0.174, 0.985),
            (-0.342, 0.940),
            (-0.766, 0.643),
            (-0.985, 0.174),
        ];
        for (n, (dx, dz)) in dirs.into_iter().enumerate() {
            let corner = DVec3::new(0.0, 0.0, 0.0);
            let a = DVec3::new(-8.0, 0.0, 0.0);
            let len = (dx * dx + dz * dz).sqrt();
            let b = DVec3::new(dx / len * 8.0, 0.0, dz / len * 8.0);
            let legs = trench_shapes(&[a, corner, b], half_width, 2.0, flat);
            assert_eq!(legs.len(), 2, "case {n}");

            // Every point of the corner's *inner cut* — the region a single
            // continuous trench of this width would occupy near the bend — must
            // be inside at least one leg. Sampled densely on the surface plane.
            let mut uncovered = 0;
            let mut sampled = 0;
            for i in -40..=40 {
                for j in -40..=40 {
                    let p = DVec3::new(i as f64 * 0.05, 0.0, j as f64 * 0.05);
                    // Inside the swept region iff within half_width of either
                    // centreline segment (that is what "a trench of this width
                    // along this path" means).
                    let near = |s: DVec3, e: DVec3| {
                        let d = e - s;
                        let t = ((p - s).dot(d) / d.length_squared()).clamp(0.0, 1.0);
                        (p - (s + d * t)).length() <= half_width
                    };
                    if !(near(a, corner) || near(corner, b)) {
                        continue;
                    }
                    sampled += 1;
                    if legs.iter().all(|l| l.distance_m(p) > 0.0) {
                        uncovered += 1;
                    }
                }
            }
            assert!(
                sampled > 100,
                "case {n}: only {sampled} points in the corner"
            );
            assert_eq!(
                uncovered, 0,
                "case {n}: {uncovered}/{sampled} points of the bend are uncut — the miter \
                 allowance does not close this corner"
            );
        }
    }

    /// Degenerate input produces no legs rather than invalid ones.
    #[test]
    fn a_degenerate_trench_path_produces_no_legs() {
        let flat = |_: f64, _: f64| Some(2.0);
        let p = DVec3::new(1.0, 2.0, 3.0);
        assert!(trench_shapes(&[], 1.0, 1.0, flat).is_empty());
        assert!(trench_shapes(&[p], 1.0, 1.0, flat).is_empty());
        assert!(
            trench_shapes(&[p, p], 1.0, 1.0, flat).is_empty(),
            "a repeated point"
        );
        assert!(
            trench_shapes(&[p, p + DVec3::X], 0.0, 1.0, flat).is_empty(),
            "no width"
        );
        // A purely VERTICAL leg has no horizontal run and is dropped — a trench
        // is an open cut, and a shaft is the tunnel tool's job.
        assert!(
            trench_shapes(&[p, p + DVec3::Y * 5.0], 1.0, 1.0, flat).is_empty(),
            "a vertical leg"
        );
        // A path with one repeated point in the middle keeps its real legs.
        let legs = trench_shapes(&[p, p, p + DVec3::X * 5.0], 1.0, 1.0, flat);
        assert_eq!(legs.len(), 1);
        // A level with no terrain at all still cuts: the waypoints are the
        // ground, exactly as the box cut falls back.
        let none = trench_shapes(&[p, p + DVec3::X * 5.0], 1.0, 1.0, |_, _| None);
        assert_eq!(none.len(), 1);
        assert!(none[0].distance_m(p) < 0.0);
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
        // Camera-independence, stated as the property it actually is: the
        // horizontal position is the PICK's, untouched — a depth rule that
        // leaked into x or z would move the cave sideways with the aim.
        // (`cut_center(x) == cut_center(x)` was the previous line here, which
        // asserts that a pure function is pure — P21.3 audit.)
        for d in [0.0, 0.5, 3.0, 40.0] {
            let c = cut_center(surface, d);
            assert_eq!(
                (c.x, c.z),
                (surface.x, surface.z),
                "depth {d} moved the cut sideways"
            );
            assert!((surface.y - c.y - d).abs() < 1e-12, "depth {d}");
        }
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

    /// **The cap is a BOUND, not a filter** (P21.3 audit). A pick a hundred
    /// kilometres away must not materialize two million points to keep
    /// thirty-two, and the ones it keeps must be the FIRST thirty-two — the drag
    /// continues from the last one placed on the next frame, so a sample of the
    /// whole run would leave a gap in the middle of the cut.
    #[test]
    fn the_dab_cap_trims_the_path_before_it_resamples() {
        let a = DVec3::ZERO;
        let far = DVec3::new(100_000.0, 0.0, 0.0);
        let capped = dab_centers_capped(&[a, far], 0.05, 32);
        assert!(
            capped.len() <= 33,
            "{} points materialized for a 32-dab budget",
            capped.len()
        );
        assert_eq!(capped[0], a);
        for (i, p) in capped.iter().enumerate() {
            assert!(
                (p.x - i as f64 * 0.05).abs() < 1e-9,
                "dab {i} is at {} — the cap resampled the whole run instead of trimming it",
                p.x
            );
        }
        // A path that fits under the cap is untouched by it.
        let short = [a, DVec3::new(1.0, 0.0, 0.0)];
        assert_eq!(
            dab_centers_capped(&short, 0.25, 32),
            dab_centers(&short, 0.25)
        );
        // A multi-leg path is trimmed mid-leg, not dropped at a waypoint.
        let bent = [a, DVec3::new(1.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 100.0)];
        let cut = dab_centers_capped(&bent, 0.25, 8);
        assert!(cut.len() <= 9, "{}", cut.len());
        assert!(cut.last().unwrap().z > 0.0, "the trim stopped at the bend");
        // Degenerate caps answer sanely rather than panicking or spinning.
        assert!(dab_centers_capped(&short, 0.25, 0).len() <= 2);
        assert!(!dab_centers_capped(&[a, far], 0.05, usize::MAX).is_empty());
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
