//! **IB-12 — the locomotion camera at partition scale** (island wave I4).
//!
//! P29's disposition row 23, relayed verbatim by the AAA-readiness
//! certification: `axis_independent_lag` *"unrotates absolute world positions
//! rather than the delta — algebraically origin-independent, not so in floating
//! point at partition scale. It wants a floating-origin-aware camera, which is a
//! streaming-scale question and belongs with the island's 50 km²."*
//!
//! Every camera arm in the tree before this file runs within a few metres of
//! `(0, 0, 0)` — which is precisely why none of them could see the defect. These
//! run the same camera at 50 km and at 500 km, and compare it against itself at
//! the origin.
//!
//! # What "continuity across a rebase" means for a camera
//!
//! `inf_math::FloatingOrigin` is a *render* anchor: nothing in the sim, and
//! nothing in `LocomotionCamera`, ever reads it. So a rebase cannot move the
//! camera by touching it — it can only move the camera if the camera's own
//! arithmetic is a function of how far the world is from zero. That is the thing
//! measured here, and it is the honest reading of the finding: the pose must be
//! **translation-equivariant**, so that the picture a rebase produces is the same
//! picture wherever the anchor lands.

use inf_ecs::camera::{
    axis_independent_lag, axis_independent_lag_absolute, CameraInput, LocomotionCamera,
};
use inf_ecs::components::{Gait, MovementMode, RotationMode};
use inf_ecs::math::Vec3d;
use inf_math::{FloatingOrigin, ORIGIN_SNAP, REBASE_DISTANCE};

/// Lag speeds that differ per axis, so the rotation genuinely matters — equal
/// speeds would make the frame irrelevant and the arm vacuous.
fn speeds() -> Vec3d {
    Vec3d::new(12.0, 6.0, 3.0)
}

const DT: f64 = 1.0 / 60.0;
/// A yaw that is not a multiple of 90°, so both sine and cosine are irrational
/// and the rotation cannot be exact by luck.
const YAW: f64 = 37.0;

/// Run `steps` of the lag over a character walking `speed_mps` north-east, with
/// everything offset by `anchor`, and answer the pose **relative to the anchor**.
fn run(lag: fn(Vec3d, Vec3d, f64, Vec3d, f64) -> Vec3d, anchor: Vec3d, steps: usize) -> Vec3d {
    let mut pivot = Vec3d::new(anchor.x, anchor.y, anchor.z);
    for s in 0..steps {
        // 6 m/s, the sprint the camera's `z` lag is tuned for, plus a metre of
        // vertical so the y axis is exercised too.
        let t = s as f64 * DT * 6.0;
        let target = Vec3d::new(
            anchor.x + t * std::f64::consts::FRAC_1_SQRT_2,
            anchor.y + (t * 0.1),
            anchor.z + t * std::f64::consts::FRAC_1_SQRT_2,
        );
        pivot = lag(pivot, target, YAW, speeds(), DT);
    }
    Vec3d::new(pivot.x - anchor.x, pivot.y - anchor.y, pivot.z - anchor.z)
}

fn err(a: Vec3d, b: Vec3d) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
}

/// **The measurement, and the alternative it rejects, priced in the same run.**
///
/// The claim is *translation equivariance*: the same relative motion must
/// produce the same relative pose wherever in the world it happens. The delta
/// form is exact — the anchor is added back untouched, so every arithmetic step
/// is on a small quantity. The absolute form is not, and the error grows with the
/// distance from zero.
#[test]
fn the_lag_is_origin_independent_at_partition_scale() {
    const STEPS: usize = 600; // ten seconds at 60 Hz
    let at_origin_delta = run(axis_independent_lag, Vec3d::ZERO, STEPS);
    let at_origin_abs = run(axis_independent_lag_absolute, Vec3d::ZERO, STEPS);

    println!("IB-12 lag, {STEPS} steps at {YAW}° yaw, error against the same run at the origin:");
    println!("  anchor delta form absolute form (pre-IB-12)");
    let mut worst_delta = 0.0f64;
    let mut worst_abs = 0.0f64;
    for (label, anchor) in [
        ("1 km", Vec3d::new(1_000.0, 0.0, 1_000.0)),
        ("50 km (the island)", Vec3d::new(50_000.0, 0.0, 50_000.0)),
        ("500 km", Vec3d::new(500_000.0, 0.0, 500_000.0)),
    ] {
        let d = err(run(axis_independent_lag, anchor, STEPS), at_origin_delta);
        let a = err(
            run(axis_independent_lag_absolute, anchor, STEPS),
            at_origin_abs,
        );
        println!("  {label:<18} {d:>12.3e} m {a:>12.3e} m");
        worst_delta = worst_delta.max(d);
        worst_abs = worst_abs.max(a);
    }

    // (1) The delta form is SUB-MICRON at half a million metres.
    //
    //     Not exactly zero, and the honest reason is worth writing down: the
    //     rotation is now on a small quantity, but `pivot` itself is still an
    //     absolute world position, so `current.x + world.x` loses ULPs at the
    //     anchor's own magnitude. Closing that last part means a camera whose
    //     state is origin-relative, which is a different change — and at 7e-10 m
    //     it is orders below anything a pixel can show.
    assert!(
        worst_delta < 1.0e-6,
        "the delta form drifted {worst_delta:.3e} m at partition scale, past a \
         micron — the anchor is reaching further into the arithmetic than the \
         final addition"
    );

    // (2) THE PRICE OF THE ALTERNATIVE, and the anti-vacuity clause in one: the
    //     form this replaced must be measurably worse, or clause (1) would be a
    //     statement about f64 rather than about the fix, and IB-12's premise
    //     would be wrong.
    let ratio = worst_abs / worst_delta.max(f64::MIN_POSITIVE);
    println!(
        "  worst: delta form {worst_delta:.3e} m, absolute form {worst_abs:.3e} m \
         -> {ratio:.3e}x better"
    );
    assert!(
        ratio > 1.0e4,
        "the absolute form was only {ratio:.1e}x worse than the delta form — the \
         defect IB-12 names is not measurable here, so this arm proves nothing"
    );
}

/// **A rebase mid-lag moves nothing.**
///
/// The camera never reads `FloatingOrigin` — a rebase is a render anchor moving —
/// so the honest statement of "continuity across a rebase" is that the pose is a
/// function of world positions alone, and the render-local pose therefore steps by
/// exactly the origin's own step and by nothing else.
///
/// This drives a real [`LocomotionCamera`] rather than the bare function, so it
/// covers the seeding latch and the settings blend too, and it forces a rebase in
/// the middle by walking past [`REBASE_DISTANCE`].
#[test]
fn a_rebase_mid_lag_moves_the_camera_by_exactly_the_origin_and_no_more() {
    let mut cam = LocomotionCamera::default();
    let mut origin = FloatingOrigin::new(inf_ecs::math::Vec3d::ZERO.to_dvec3());
    let mut rebases = 0usize;
    let mut worst_render_jump = 0.0f64;
    let mut prev_render: Option<glam::DVec3> = None;
    let mut prev_world: Option<Vec3d> = None;

    // Walk 3 km east at 6 m/s: three crossings of the 1024 m rebase threshold.
    const STEPS: usize = 30_000;
    for s in 0..STEPS {
        let t = s as f64 * DT * 6.0;
        let pivot = Vec3d::new(t, 1.5, 0.0);
        cam.advance(
            &CameraInput {
                pivot_target: pivot,
                aim_yaw_deg: YAW,
                aim_pitch_deg: -8.0,
                rotation_mode: RotationMode::default(),
                gait: Gait::default(),
                mode: MovementMode::default(),
            },
            DT,
        );
        let world = cam.desired;
        let moved = origin.maybe_rebase(glam::DVec3::new(world.x, world.y, world.z));
        rebases += usize::from(moved);
        let render = origin.to_render(glam::DVec3::new(world.x, world.y, world.z));
        if let (Some(pr), Some(pw)) = (prev_render, prev_world) {
            // The render-local step must equal the world step minus the origin's
            // own step. A rebase moves the origin by a multiple of ORIGIN_SNAP,
            // so the render pose jumps by that and by nothing else.
            let world_step = glam::DVec3::new(world.x - pw.x, world.y - pw.y, world.z - pw.z);
            let render_step = glam::DVec3::new(
                f64::from(render.x) - pr.x,
                f64::from(render.y) - pr.y,
                f64::from(render.z) - pr.z,
            );
            // `to_render` is f32, so the comparison carries a float-precision
            // tolerance rather than being exact — at a 1024 m origin distance an
            // f32 metre is ~1e-4 m.
            //
            // **The snap is only forgiven on the step that rebased** (the I4
            // audit). Rounding every step to the nearest multiple of `ORIGIN_SNAP`
            // forgives a jump of exactly 10 m — or 20, or 30 — on *any* step, so
            // a camera that teleported by one snap on an ordinary frame passed
            // this arm unremarked. The origin only moves when `maybe_rebase` says
            // it did, and that is the only step allowed to be a multiple of the
            // snap.
            let unexplained = (render_step - world_step).length();
            let residue = match moved {
                true => (unexplained - (unexplained / ORIGIN_SNAP).round() * ORIGIN_SNAP).abs(),
                false => unexplained,
            };
            worst_render_jump = worst_render_jump.max(residue);
        }
        prev_render = Some(glam::DVec3::new(
            f64::from(render.x),
            f64::from(render.y),
            f64::from(render.z),
        ));
        prev_world = Some(world);
    }

    println!(
        "IB-12 rebase: {rebases} rebases over {STEPS} steps ({REBASE_DISTANCE} m \
         threshold, {ORIGIN_SNAP} m snap); worst unexplained render step \
         {worst_render_jump:.6} m"
    );
    assert!(
        rebases >= 2,
        "only {rebases} rebases in {STEPS} steps — the walk did not cross the \
         threshold and this arm measured a stationary origin"
    );
    assert!(
        worst_render_jump < 1.0e-2,
        "the render-local camera moved {worst_render_jump:.6} m that neither the \
         world step nor the origin's own snap accounts for — the pose is a \
         function of the floating origin, which is IB-12"
    );
}

/// **The production call site uses the delta form, and there is exactly one.**
///
/// A fix that leaves its predecessor reachable from the shipped path is not a
/// fix. `axis_independent_lag_absolute` exists for the price-the-alternative arm
/// above and may have no caller in `src/`; the delta form must have exactly one,
/// in `LocomotionCamera::advance`.
///
/// Pinned as **counts over the source**, per the I1-audit law that a `contains`
/// needle which is a prefix of a declaration can never fail.
#[test]
fn the_shipped_camera_calls_the_delta_form_and_only_that() {
    let whole = include_str!("../src/camera.rs");
    // **Production only.** `src/camera.rs`'s own `#[cfg(test)] mod tests` calls
    // the lag eight times; counting those would make "the shipped camera calls
    // it" true of a camera that had stopped calling it.
    let src = match whole.find("#[cfg(test)]") {
        Some(i) => &whole[..i],
        None => whole,
    };
    let calls = |needle: &str| src.matches(needle).count();
    // Two declarations (`pub fn axis_independent_lag(` and
    // `..._absolute(`) plus the one production call.
    let delta_decl = calls("pub fn axis_independent_lag(");
    let abs_decl = calls("pub fn axis_independent_lag_absolute(");
    let delta_calls = calls("axis_independent_lag(") - delta_decl;
    let abs_calls = calls("axis_independent_lag_absolute(") - abs_decl;
    println!(
        "IB-12 call sites in inf_ecs::camera: delta {delta_calls}, absolute \
         {abs_calls} (declarations {delta_decl}/{abs_decl})"
    );
    assert_eq!(delta_decl, 1, "the delta form must be declared once");
    assert_eq!(abs_decl, 1, "the priced alternative must be declared once");
    assert_eq!(
        abs_calls, 0,
        "`axis_independent_lag_absolute` is called {abs_calls} time(s) inside \
         `src/` — the pre-IB-12 spelling is reachable from the shipped camera"
    );
    assert_eq!(
        delta_calls, 1,
        "nothing in `src/` calls `axis_independent_lag` — the camera stopped \
         lagging and every arm above is measuring a free function"
    );
}

/// **A partition handoff at speed does not move the camera** (IB-12's second half).
///
/// A world-partition cell is `inf_scene::DEFAULT_CELL_SIZE_M` (256 m) across, and
/// a handoff is what happens when the subject crosses one at speed: the cell
/// streamer despawns one cell's entities and spawns another's, inside a single
/// fixed step. The camera has no cell of its own and reads no streamer — its
/// subject is **re-resolved every step** (`inf_ecs::movement::camera_subject`,
/// the minimum `Guid` among player-controlled characters) rather than latched,
/// precisely so a despawned subject does not leave a camera pinned to a guid that
/// has gone.
///
/// The hazard that leaves is the **gap**: a step in which the subject is missing
/// (its cell has gone and its replacement has not arrived), during which
/// `inf_physics::d3::step_locomotion_camera` returns `None` and the camera holds
/// its pose. The step after it presents a target two steps' worth away. This arm
/// drives exactly that — a sprint across four cell boundaries with a one-step gap
/// at each — and requires the recovery to stay inside the lag's own per-step
/// bound, i.e. that a handoff is absorbed by the lag rather than showing as a cut.
#[test]
fn a_partition_handoff_at_speed_is_absorbed_by_the_lag() {
    /// `inf_scene::DEFAULT_CELL_SIZE_M`, named here rather than imported —
    /// `inf-ecs` does not depend on `inf-scene`, and `inf_ecs::band`'s own docs
    /// already reference the number the same way. It is safe as a literal
    /// precisely because **nothing below is asserted against it**: it decides how
    /// OFTEN the gap fires and never what the gap costs, so a drift makes this
    /// arm cross more or fewer boundaries and cannot make it pass wrongly.
    const CELL_M: f64 = 256.0;
    const SPEED: f64 = 20.0; // a vehicle, not a jog: 0.333 m per fixed step
    let mut cam = LocomotionCamera::default();
    let input = |p: Vec3d| CameraInput {
        pivot_target: p,
        aim_yaw_deg: YAW,
        aim_pitch_deg: -8.0,
        rotation_mode: RotationMode::default(),
        gait: Gait::default(),
        mode: MovementMode::default(),
    };

    // Settle first, so the seeding snap is not what is measured.
    for s in 0..120 {
        cam.advance(&input(Vec3d::new(s as f64 * DT * SPEED, 1.5, 0.0)), DT);
    }

    let mut worst_normal = 0.0f64;
    let mut worst_after_gap = 0.0f64;
    let mut gaps = 0usize;
    let mut prev = cam.pose.position;
    let mut last_cell = ((120.0 * DT * SPEED) / CELL_M).floor() as i64;
    // Whether the PREVIOUS step was the gap. The first version of this arm
    // re-derived "was there a gap" from the cell index, which is true only *on*
    // the crossing step — the step it `continue`s past — so the branch never ran
    // and `worst_after_gap` stayed 0.000000. A check that cannot fire is not a
    // check; carrying the fact forward is.
    let mut just_gapped = false;
    for s in 120..6_000 {
        let x = s as f64 * DT * SPEED;
        let cell = (x / CELL_M).floor() as i64;
        let crossing = cell != last_cell;
        last_cell = cell;
        if crossing {
            // THE GAP: the subject is not resolvable this step, so the host does
            // not advance the camera at all. This is what
            // `a_camera_with_nothing_to_follow_holds_its_pose` covers, driven at
            // speed and across a boundary.
            gaps += 1;
            just_gapped = true;
            continue;
        }
        cam.advance(&input(Vec3d::new(x, 1.5, 0.0)), DT);
        let p = cam.pose.position;
        let step =
            ((p.x - prev.x).powi(2) + (p.y - prev.y).powi(2) + (p.z - prev.z).powi(2)).sqrt();
        // The step immediately after a gap covers two frames of subject motion.
        if just_gapped {
            worst_after_gap = worst_after_gap.max(step);
        } else {
            worst_normal = worst_normal.max(step);
        }
        just_gapped = false;
        prev = p;
    }

    // **THE ALTERNATIVE, PRICED.** A cut is the whole two frames of subject
    // motion arriving in one step, which at this speed is exactly `2 · dt · v`.
    // The number matters because the arm's first ceiling was `3 x` the steady
    // step and the steady step is 0.377 m, not "a few centimetres" as its own
    // comment claimed — so a full cut at 0.667 m is **1.77x**, comfortably inside
    // a 3x ceiling. The arm named a defect it could not see (the I4 audit).
    let cut = 2.0 * DT * SPEED;
    println!(
        "IB-12 handoff: {gaps} cell crossings at {SPEED} m/s over {CELL_M} m cells; \
         worst camera step {worst_normal:.6} m normally, {worst_after_gap:.6} m \
         across a handoff ({:.2}x); a CUT would be {cut:.6} m ({:.2}x)",
        worst_after_gap / worst_normal.max(f64::MIN_POSITIVE),
        cut / worst_normal.max(f64::MIN_POSITIVE),
    );
    assert!(
        gaps >= 4,
        "only {gaps} cell crossings — the run did not cross enough boundaries to \
         be a handoff test"
    );
    assert!(
        worst_normal > 0.0,
        "the camera never moved, so the comparison below is vacuous"
    );
    assert!(
        worst_after_gap > 0.0,
        "no step was ever recorded as following a handoff — the branch that measures the recovery never ran, which is how the first version of this arm reported 0.000000 m and passed"
    );
    // A handoff may cost at most one extra step of catch-up, and the ceiling has
    // to sit **below the cut it names** or it cannot see one. Measured: 0.385 m
    // across a handoff against a 0.377 m steady step (1.02x) and a 0.667 m cut
    // (1.77x). 1.5x is halfway between the measurement and the defect, which is
    // where a falsifying threshold goes.
    assert!(
        worst_after_gap <= worst_normal * 1.5,
        "the camera moved {worst_after_gap:.6} m in the step after a partition \
         handoff against a steady {worst_normal:.6} m — the handoff is no longer \
         being absorbed by the lag, and a full cut here would be {cut:.6} m"
    );
    assert!(
        worst_normal * 1.5 < cut,
        "the ceiling above ({:.6} m) is not below the cut it exists to catch \
         ({cut:.6} m), so a handoff that arrived as a cut would satisfy it — which \
         is what a 3x ceiling did before the I4 audit measured the steady step",
        worst_normal * 1.5
    );
}
