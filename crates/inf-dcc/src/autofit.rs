//! **Auto-fit** (P24.2): put a template skeleton *inside* an arbitrary mesh.
//!
//! `inf_anim::build_template` makes a rig from proportions. This makes a rig from
//! a **model** — bounding analysis, a symmetry plane, then per-joint refinement
//! against the surface — so a character an author imported or sculpted gets a
//! skeleton without anyone placing a bone by hand.
//!
//! # Closest points, not an SDF
//!
//! The P24 plan said "SDF-based template-to-mesh fitting" and the survey found
//! that the engine's SDF cannot do it: `inf_voxel`'s field is a **±4-voxel narrow
//! band** with no gradient exposed, so it can answer "is this near a surface"
//! over a quantized lattice and nothing more — no distance at range, no direction
//! to descend. What a fit actually needs is *this point's nearest surface and the
//! chord through it*, which is a closest-point query, which is
//! [`crate::bvh`]. Building an SDF just to sample it would be a lossy
//! reimplementation of the query underneath it.
//!
//! # The three stages, and what each one can refuse
//!
//! 1. **Bounds.** The mesh's extents give the template its height (and a
//!    multi-girdle body its length). A degenerate extent is
//!    [`FitError::Degenerate`] — there is no such thing as a rig for a plane.
//! 2. **The symmetry plane.** A character is bilaterally symmetric about some
//!    `x = c`, and that plane is where the spine goes. `c` is found by scoring a
//!    fixed ladder of candidates — mirror a sample of the surface across each and
//!    measure how far the mirrored points land from the surface — and the best
//!    score is *reported*, so "we found a plane" is a number rather than a claim.
//!    Past [`SYMMETRY_TOLERANCE`] of the diagonal it is
//!    [`FitError::NoSymmetryPlane`]: a mesh with no symmetry is not a mistake to
//!    round off, it is a mesh this cannot fit.
//! 3. **Per-joint medial refinement.** A joint moves toward the middle of the
//!    chord through it — from the nearest surface point, through the joint, out
//!    the other side — which is a cheap estimate of the medial axis and is what
//!    puts a shoulder inside a shoulder rather than inside the air beside it. A
//!    step that would take the joint *out* of the mesh is refused and counted;
//!    the joint keeps its previous position, so refinement can only improve
//!    containment, never break it.
//!
//! # Determinism
//!
//! Every loop is an integer range, every candidate ladder is a fixed count, every
//! comparison that picks a winner uses `total_cmp` with an index tie-break, and
//! the geometry is `f64` throughout. **No trigonometry** (the P14 law): the fit
//! places joints by translation only, so the generated bind pose keeps
//! `build_template`'s identity rotations and its exact-negation inverse binds.
//!
//! `f32` appears in exactly one function, [`emit_joints`], which is the wire —
//! `inf_anim`'s skeleton is `f32` by its own documented decision, and the
//! conversion happens once, at the edge, the discipline `inf_anim::template`
//! records. `autofit_converts_to_f32_once_at_the_wire` in the crate's determinism
//! gate is what keeps that true.
//!
//! # What this does NOT do, and is ledgered
//!
//! * **It does not rotate a joint.** Bones are placed, not aimed, so a character
//!   modelled in a pose other than a symmetric rest stance gets a rig whose bind
//!   pose is upright while its mesh is not. Aiming needs a rotation on the bind
//!   pose, which is a `sin`-free but genuinely new step.
//! * **It assumes +Y up and the mesh centred on its own symmetry plane in x.**
//!   Both are the engine's conventions (architecture rule 6's frame), not
//!   detections.
//! * **It does not weight anything.** The rig lands rigid — every vertex on
//!   joint 0 — and `crate::heat` is what fills the channel in.

use glam::{DVec3, Quat, Vec3};
use inf_anim::skeleton::{Joint, JointTransform, Skeleton};
use inf_anim::template::{build_template, BodyParams, BodyPlan, TemplateError};
use inf_anim::SkeletonAsset;

use crate::bvh::Bvh;

/// How many candidate symmetry planes are scored.
///
/// A fixed count, not a search: a bisection or a descent would terminate on a
/// floating comparison and make the plane depend on rounding, which is the same
/// class of hazard as a non-portable `sin` on committed content. 64 candidates
/// across the model's width put the ladder's spacing at well under a centimetre
/// on a human-sized mesh.
pub const SYMMETRY_CANDIDATES: usize = 64;

/// At most this many surface samples are mirrored when scoring a plane.
///
/// Every triangle would be exact and quadratic; a deterministic stride over the
/// triangle list is a uniform sample of the surface in the order the mesh stores
/// it, and 512 of them settle the score to far finer than the ladder's spacing.
pub const SYMMETRY_SAMPLES: usize = 512;

/// The worst symmetry score a fit will accept, as a fraction of the mesh's
/// diagonal.
///
/// The score is the mean distance a mirrored surface sample lands from the
/// surface. On a truly symmetric model it is ~0; at 2 % of the diagonal the model
/// is asymmetric enough that the spine would be placed through one shoulder.
pub const SYMMETRY_TOLERANCE: f64 = 0.02;

/// How far a joint moves toward its medial estimate per iteration.
///
/// Damped rather than snapped: the medial estimate is a chord midpoint, which
/// overshoots near a limb's end where the chord runs along the limb rather than
/// across it. Half-steps converge in a handful of iterations and never
/// oscillate.
pub const REFINE_DAMPING: f64 = 0.5;

/// Feet are planted this far above the mesh's lowest point, as a fraction of its
/// height.
///
/// Not zero: a joint exactly at `min.y` is *on* the surface, where "inside" is
/// the one question a containment test cannot answer. 2 % of a 1.75 m character
/// is 3.5 cm — inside the foot, and still on the ground by any measure a gate
/// would use.
pub const FOOT_CLEARANCE: f64 = 0.02;

/// What to fit, and how hard.
#[derive(Debug, Clone, PartialEq)]
pub struct FitOptions {
    /// The body plan to generate.
    pub plan: BodyPlan,
    /// Proportions. `height_m` and (for a multi-girdle plan) `body_length_m` are
    /// **overwritten** from the mesh — those are the two the model answers — and
    /// every other ratio is the caller's.
    pub params: BodyParams,
    /// Medial refinement passes. Zero is legal and means "place by proportions
    /// alone", which is what the anti-vacuity arm of the gate uses.
    pub refine_iterations: u32,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            plan: BodyPlan::Biped,
            params: BodyParams::default(),
            refine_iterations: 4,
        }
    }
}

/// What the fit found, as numbers rather than as a verdict.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitReport {
    /// The mesh's height, which became the template's (metres).
    pub height_m: f64,
    /// Where the symmetry plane landed (`x = symmetry_x`, metres).
    pub symmetry_x: f64,
    /// The winning plane's mean mirrored-sample distance, **as a fraction of the
    /// mesh diagonal** — comparable across models, and the number
    /// [`SYMMETRY_TOLERANCE`] is checked against.
    pub symmetry_score: f64,
    /// Joints in the generated skeleton.
    pub joints: usize,
    /// Joints inside the mesh when the fit finished. Equal to `joints` on a model
    /// this can fit; less means the mesh has parts the template's proportions do
    /// not reach, and it is reported rather than refused because a partial fit is
    /// still a better starting point than none.
    pub joints_inside: usize,
    /// Refinement steps refused because they would have moved a joint out of the
    /// mesh.
    pub steps_rejected: usize,
}

/// Why a fit refused.
/// `Clone` is deliberately absent: `TemplateError` is not `Clone` (it carries a
/// `String` reading of the parameter it refused), and a refusal is read once at
/// the point it happens.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum FitError {
    /// The mesh has no volume to put a skeleton in.
    #[error("the mesh cannot carry a skeleton: {what} is {value}")]
    Degenerate { what: &'static str, value: f64 },
    /// No candidate plane was symmetric enough.
    #[error(
        "no symmetry plane found: the best candidate mirrors the surface onto \
         itself with a mean error of {best:.4} of the model's diagonal, past the \
         {tolerance:.4} a bilateral body has"
    )]
    NoSymmetryPlane { best: f64, tolerance: f64 },
    /// The template generator refused the derived proportions.
    #[error("the template refused the proportions taken from the mesh: {0}")]
    Template(#[from] TemplateError),
}

/// Fit `opts.plan` inside `bvh`.
///
/// Returns the skeleton and the numbers behind it. See the module docs for the
/// three stages and what each can refuse.
pub fn fit_template(bvh: &Bvh, opts: &FitOptions) -> Result<(SkeletonAsset, FitReport), FitError> {
    // ── 1. bounds ──────────────────────────────────────────────────────────
    let Some((min, max)) = bvh.bounds() else {
        return Err(FitError::Degenerate {
            what: "the triangle count",
            value: 0.0,
        });
    };
    let ext = max - min;
    if !ext.is_finite() {
        return Err(FitError::Degenerate {
            what: "an extent",
            value: f64::NAN,
        });
    }
    for (what, v) in [
        ("the height", ext.y),
        ("the width", ext.x),
        ("the depth", ext.z),
    ] {
        if v <= 1e-6 {
            return Err(FitError::Degenerate { what, value: v });
        }
    }
    let diagonal = ext.length();

    // ── 2. the symmetry plane ──────────────────────────────────────────────
    let samples = sample_points(bvh);
    let (symmetry_x, score) = best_symmetry_plane(bvh, &samples, min.x, max.x, diagonal);
    if score > SYMMETRY_TOLERANCE {
        return Err(FitError::NoSymmetryPlane {
            best: score,
            tolerance: SYMMETRY_TOLERANCE,
        });
    }

    // ── 3. the template, sized by the mesh ─────────────────────────────────
    let mut params = opts.params;
    params.height_m = ext.y;
    if opts.plan.legs() > 2 {
        // A quadruped's torso runs along the model's depth. 70 % of it, because
        // the head and the tail hang off the ends of the girdle span rather than
        // inside it.
        params.body_length_m = ext.z * 0.7;
    }
    let asset = build_template(opts.plan, &params)?;
    let skeleton = &asset.skeleton;

    // The bind pose is pure translation (`build_template` builds it from axis
    // offsets and nothing else), so a joint's global bind position is the sum of
    // its ancestors' local translations — no matrix, no divide.
    let mut globals: Vec<DVec3> = Vec::with_capacity(skeleton.len());
    for joint in skeleton.joints() {
        let local = DVec3::new(
            joint.local_bind.translation[0] as f64,
            joint.local_bind.translation[1] as f64,
            joint.local_bind.translation[2] as f64,
        );
        let base = joint
            .parent
            .map(|p| globals[p as usize])
            .unwrap_or(DVec3::ZERO);
        globals.push(base + local);
    }

    // Place: the template stands with its feet at y = 0 on the x = 0 plane, so
    // the offset is the symmetry plane, the mesh's floor, and the mesh's centre
    // of depth.
    let centre_z = 0.5 * (min.z + max.z);
    let offset = DVec3::new(symmetry_x, min.y, centre_z);
    for g in &mut globals {
        *g += offset;
    }

    // ── 4. medial refinement ───────────────────────────────────────────────
    let ground = min.y + FOOT_CLEARANCE * ext.y;
    // Which joints are planted on the ground. **The role table first** (SK1a):
    // the mannequin's `foot_*` is an ankle and its `ik_foot_*` is a marker, and
    // the prefix rule cannot tell a `Foot` from a `Ball` on any rig that has both.
    // The prefix stays as the fallback for a rig with no table.
    let roles = asset.role_index();
    let is_foot: Vec<bool> = skeleton
        .joints()
        .iter()
        .enumerate()
        .map(|(i, j)| match roles.kind_of(i as u16) {
            Some(k) => k == inf_anim::BoneRoleKind::Foot,
            None => j.name.starts_with("foot_"),
        })
        .collect();
    // Feet are planted before refinement so the chord they are refined against is
    // the one through the ankle rather than through the floor.
    for (i, g) in globals.iter_mut().enumerate() {
        if is_foot[i] {
            g.y = ground;
        }
    }

    let mut steps_rejected = 0usize;
    for _ in 0..opts.refine_iterations {
        for i in 0..globals.len() {
            let p = globals[i];
            let Some(next) = medial_step(bvh, p) else {
                continue;
            };
            let mut candidate = p + (next - p) * REFINE_DAMPING;
            if is_foot[i] {
                candidate.y = ground; // a foot stays on the ground
            }
            if bvh.contains(candidate) {
                globals[i] = candidate;
            } else {
                steps_rejected += 1;
            }
        }
        symmetrize(skeleton, &mut globals, symmetry_x);
    }

    let joints_inside = globals.iter().filter(|g| bvh.contains(**g)).count();
    let joints = skeleton.len();
    let fitted = emit_joints(skeleton, &globals);
    // A generator bug, kept as a value so it can never be a panic in an
    // author's editor — `build_template` makes the same call for the same
    // reason, and this reuses its variant rather than inventing a second.
    let skeleton = Skeleton::new(fitted).map_err(TemplateError::Invalid)?;
    let mut out = SkeletonAsset::with_sockets(skeleton, asset.sockets.clone());
    out.limits = asset.limits.clone();
    // **The side tables ride through the fit** (SK1a). The fit MOVES joints and
    // never adds, removes or reorders one — `emit_joints` maps the input list
    // one-to-one — so every table keyed on a joint index is still true of the
    // result. Dropping them was the shape of a real defect: a fitted mannequin
    // arrived with no role table, `build_locomotion` fell back to its name rule,
    // and the wizard refused the rig it had just generated with a message about
    // `upper_leg_l`.
    out.roles = asset.roles.clone();
    out.twists = asset.twists.clone();
    out.ik_follow = asset.ik_follow.clone();
    out.grips = asset.grips.clone();

    Ok((
        out,
        FitReport {
            height_m: ext.y,
            symmetry_x,
            symmetry_score: score,
            joints,
            joints_inside,
            steps_rejected,
        },
    ))
}

/// A deterministic surface sample: every `stride`-th triangle's centroid, at most
/// [`SYMMETRY_SAMPLES`] of them.
fn sample_points(bvh: &Bvh) -> Vec<DVec3> {
    let tris = bvh.triangles();
    if tris.is_empty() {
        return Vec::new();
    }
    let stride = tris.len().div_ceil(SYMMETRY_SAMPLES).max(1);
    tris.iter()
        .step_by(stride)
        .map(|t| (t.a + t.b + t.c) / 3.0)
        .collect()
}

/// Score a fixed ladder of candidate planes and return `(x, score)` for the best.
///
/// The score is the mean distance a mirrored sample lands from the surface,
/// divided by the diagonal so it is comparable across model sizes. Ties go to the
/// lower candidate index, so a perfectly symmetric ladder (a sphere, where every
/// plane through the centre scores zero) still has one answer.
fn best_symmetry_plane(
    bvh: &Bvh,
    samples: &[DVec3],
    min_x: f64,
    max_x: f64,
    diagonal: f64,
) -> (f64, f64) {
    if samples.is_empty() || diagonal <= 0.0 {
        return (0.5 * (min_x + max_x), f64::INFINITY);
    }
    let mut best = (0.5 * (min_x + max_x), f64::INFINITY);
    for k in 0..SYMMETRY_CANDIDATES {
        // Cell centres of a uniform ladder: an integer numerator over an integer
        // denominator, so the candidate set is a pure function of the bounds.
        let t = (2 * k + 1) as f64 / (2 * SYMMETRY_CANDIDATES) as f64;
        let c = min_x + (max_x - min_x) * t;
        let mut total = 0.0;
        for s in samples {
            let mirrored = DVec3::new(2.0 * c - s.x, s.y, s.z);
            total += bvh
                .closest_point(mirrored)
                .map(|h| h.distance)
                .unwrap_or(f64::INFINITY);
        }
        let score = total / samples.len() as f64 / diagonal;
        if score.total_cmp(&best.1).is_lt() {
            best = (c, score);
        }
    }
    best
}

/// The midpoint of the chord that runs **into the mesh** from the surface point
/// nearest `p`.
///
/// One rule for both cases, and the case split is only which way "into" is:
///
/// * `p` **inside** — the chord enters at the nearest surface point, passes
///   through `p`, and leaves the far side. Its midpoint is a medial estimate, and
///   moving toward it is what centres a joint in a limb.
/// * `p` **outside** — the medial estimate through `p` is meaningless (the chord
///   runs away from the mesh), so the direction flips: the chord enters at the
///   nearest surface point heading *toward* the interior, and its midpoint pulls
///   the joint in. Without this an outside joint is stuck outside for ever —
///   every candidate step fails the containment check, which is what
///   `an_overhanging_joint_is_pulled_into_the_mesh` measures.
///
/// `None` when `p` sits exactly on the surface (no direction to leave along) or
/// when the chord has no far side (an open mesh, where the ray escapes).
fn medial_step(bvh: &Bvh, p: DVec3) -> Option<DVec3> {
    let near = bvh.closest_point(p)?;
    let outward = p - near.point;
    let len = outward.length();
    // `<=` rather than `!(> )`: `length()` is `sqrt` of a sum of squares, so it
    // is NaN only if the input was, and a NaN position is already a
    // `Violation::NonFinitePosition` upstream.
    if len <= 1e-9 {
        return None;
    }
    // Inward from the surface, whichever side `p` is on.
    let dir = if bvh.contains(p) {
        outward / len
    } else {
        -outward / len
    };
    // Fired from the surface point itself; `RAY_EPSILON` is what stops it
    // re-hitting the triangle it started on.
    let far = bvh.first_hit(near.point, dir)?;
    Some(near.point + dir * (far.t * 0.5))
}

/// Average each left/right joint pair across the symmetry plane, and pull every
/// unpaired joint onto it.
///
/// Refinement is per-joint and the mesh is only *approximately* symmetric, so
/// without this a left elbow and a right elbow drift apart by a few millimetres —
/// which is invisible in a viewport and fatal to a mirror-modelling workflow and
/// to any gate that compares the two sides.
fn symmetrize(skeleton: &Skeleton, globals: &mut [DVec3], symmetry_x: f64) {
    let joints = skeleton.joints();
    for (i, j) in joints.iter().enumerate() {
        let Some(mate) = mirror_name(&j.name) else {
            // Unpaired: it is on the midline by construction (hips, spine, head).
            globals[i].x = symmetry_x;
            continue;
        };
        let Some(k) = skeleton.index_of(&mate).map(|x| x as usize) else {
            continue;
        };
        if k < i {
            continue; // handled when the pair's lower index came round
        }
        let (a, b) = (globals[i], globals[k]);
        // Mirror b onto a's side, average, and write the pair back symmetrically.
        let b_mirrored = DVec3::new(2.0 * symmetry_x - b.x, b.y, b.z);
        let mean = 0.5 * (a + b_mirrored);
        globals[i] = mean;
        globals[k] = DVec3::new(2.0 * symmetry_x - mean.x, mean.y, mean.z);
    }
}

/// The name of a joint's mirror, or `None` when it has no side.
///
/// The template's vocabulary is a `_l` / `_r` suffix, optionally followed by a
/// girdle index (`_l0`, `_r1`) — see `inf_anim::template::leg_suffix`. Both
/// spellings are handled here, and nothing else is: an unrecognised name is
/// midline, which is the conservative answer.
fn mirror_name(name: &str) -> Option<String> {
    let (stem, tail) = name.rsplit_once('_')?;
    let mut chars = tail.chars();
    let side = chars.next()?;
    let index: String = chars.collect();
    if !index.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let other = match side {
        'l' => 'r',
        'r' => 'l',
        _ => return None,
    };
    Some(format!("{stem}_{other}{index}"))
}

/// **The wire.** Turn refined `f64` global positions back into `inf_anim`'s
/// `f32` joints — the crate's only `f32`, in one function, converting once.
///
/// The local bind is the difference from the parent's global, the rotation stays
/// identity and the scale unit (the fit translates and never aims), and the
/// inverse bind is the exact negation of the global translation — the same
/// closed form `inf_anim::template` uses, for the same reason: a bind pose with
/// identity rotation and unit scale has one, and `Mat4::inverse` would put a
/// divide on a path that does not need one.
fn emit_joints(skeleton: &Skeleton, globals: &[DVec3]) -> Vec<Joint> {
    skeleton
        .joints()
        .iter()
        .enumerate()
        .map(|(i, j)| {
            let base = j.parent.map(|p| globals[p as usize]).unwrap_or(DVec3::ZERO);
            let local = globals[i] - base;
            let local32 = Vec3::new(local.x as f32, local.y as f32, local.z as f32);
            let global32 = Vec3::new(
                globals[i].x as f32,
                globals[i].y as f32,
                globals[i].z as f32,
            );
            Joint {
                name: j.name.clone(),
                parent: j.parent,
                inverse_bind: glam::Mat4::from_translation(-global32).to_cols_array(),
                local_bind: JointTransform::from_trs(local32, Quat::IDENTITY, Vec3::ONE),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::Tri;

    /// **The fixture's plan is the twenty-joint rig, and the production default is
    /// not** (SK1a).
    ///
    /// `FitOptions::default()` is the mannequin, because the wizard's default is,
    /// and that is the right production answer. It is the wrong *fixture*: the
    /// tube man below is 40 cm across, and a 161-bone rig in a T-pose puts both
    /// forearms, both hands, every finger and both `ik_hand_*` handles outside it
    /// before the fit has done anything at all. A containment test on that
    /// measures the tube, not the fit.
    fn fixture_opts() -> FitOptions {
        FitOptions {
            plan: BodyPlan::BipedCanonical,
            ..FitOptions::default()
        }
    }

    /// **A tube man**: a torso, a head, two arms and two legs, as overlapping
    /// boxes.
    ///
    /// Overlapping deliberately — that is what a modelled character is, and it is
    /// the case a parity containment test gets wrong (see [`crate::bvh`]). Sized
    /// as a 1.8 m adult so the template's default proportions are the ones under
    /// test rather than a shape invented to suit them.
    fn tube_man() -> Bvh {
        let mut tris: Vec<Tri> = Vec::new();
        let mut part = |min: [f64; 3], max: [f64; 3]| {
            tris.extend(crate::bvh::tests::box_tris(
                DVec3::from_array(min),
                DVec3::from_array(max),
            ));
        };
        // Torso: hips (0.95 m) to shoulders (1.48 m).
        part([-0.20, 0.90, -0.12], [0.20, 1.50, 0.12]);
        // Head + neck.
        part([-0.10, 1.48, -0.10], [0.10, 1.80, 0.10]);
        // Arms, hanging from the shoulders to the wrists. They reach mid-thigh
        // (0.66 m on a 1.8 m body) because that is where a human's do and where
        // `BodyParams::default`'s `arm_length_ratio` of 0.42 puts the template's:
        // an arm box that stopped at the hips would put the hand 3 cm below it,
        // and the test would be measuring the fixture.
        part([-0.30, 0.66, -0.08], [-0.16, 1.48, 0.08]);
        part([0.16, 0.66, -0.08], [0.30, 1.48, 0.08]);
        // Legs, hips to the floor.
        part([-0.18, 0.0, -0.09], [-0.02, 0.98, 0.09]);
        part([0.02, 0.0, -0.09], [0.18, 0.98, 0.09]);
        Bvh::new(tris)
    }

    fn globals_of(asset: &SkeletonAsset) -> Vec<DVec3> {
        let mut out: Vec<DVec3> = Vec::new();
        for j in asset.skeleton.joints() {
            let l = DVec3::new(
                j.local_bind.translation[0] as f64,
                j.local_bind.translation[1] as f64,
                j.local_bind.translation[2] as f64,
            );
            let base = j.parent.map(|p| out[p as usize]).unwrap_or(DVec3::ZERO);
            out.push(base + l);
        }
        out
    }

    /// **THE GATE.** Every joint of a fitted biped is inside the mesh, the feet
    /// are on the ground, and the hands are in the sleeves.
    #[test]
    fn a_fitted_biped_lands_entirely_inside_the_mesh() {
        let bvh = tube_man();
        let (asset, report) = fit_template(&bvh, &fixture_opts()).expect("the fit succeeds");

        assert!(
            (report.height_m - 1.80).abs() < 1e-9,
            "the template takes the MESH's height: {report:?}"
        );
        assert!(
            report.symmetry_score < SYMMETRY_TOLERANCE,
            "the plane must be a real one: {report:?}"
        );
        assert!(
            report.symmetry_x.abs() < 0.02,
            "the tube man is symmetric about x = 0: {report:?}"
        );

        let globals = globals_of(&asset);
        assert_eq!(globals.len(), report.joints);
        let outside: Vec<&str> = asset
            .skeleton
            .joints()
            .iter()
            .zip(&globals)
            .filter(|(_, g)| !bvh.contains(**g))
            .map(|(j, _)| j.name.as_str())
            .collect();
        assert!(
            outside.is_empty(),
            "these joints landed outside the mesh: {outside:?}"
        );
        assert_eq!(report.joints_inside, report.joints);

        // Feet on the ground (the mesh's floor is y = 0).
        for name in ["foot_l", "foot_r"] {
            let i = asset.skeleton.index_of(name).expect("a biped has feet") as usize;
            assert!(
                globals[i].y < 0.06 * report.height_m,
                "{name} is at y = {} on a {} m body",
                globals[i].y,
                report.height_m
            );
        }
        // Hands in the sleeves: out past the torso's half-width, inside the arm.
        for (name, sign) in [("hand_l", -1.0), ("hand_r", 1.0)] {
            let i = asset.skeleton.index_of(name).expect("a biped has hands") as usize;
            let x = globals[i].x * sign;
            assert!(
                x > 0.15,
                "{name} is at x = {}, which is inside the torso rather than the \
                 arm",
                globals[i].x
            );
        }
        // Left and right agree, exactly, after `symmetrize`.
        for (l, r) in [("hand_l", "hand_r"), ("foot_l", "foot_r")] {
            let (a, b) = (
                globals[asset.skeleton.index_of(l).unwrap() as usize],
                globals[asset.skeleton.index_of(r).unwrap() as usize],
            );
            assert!(
                (a.x + b.x - 2.0 * report.symmetry_x).abs() < 1e-6
                    && (a.y - b.y).abs() < 1e-6
                    && (a.z - b.z).abs() < 1e-6,
                "{l} {a:?} and {r} {b:?} are not mirror images"
            );
        }
    }

    /// **ANTI-VACUITY.** A deliberately mis-scaled template — the same fit with a
    /// height the mesh did not give it — puts joints outside, so the containment
    /// assertion above is a test and not a tautology.
    #[test]
    fn a_mis_scaled_fit_fails_the_containment_check() {
        let bvh = tube_man();
        let (good, _) = fit_template(&bvh, &fixture_opts()).unwrap();
        let good_outside = globals_of(&good)
            .iter()
            .filter(|g| !bvh.contains(**g))
            .count();
        assert_eq!(good_outside, 0, "the control fit is clean");

        // Build the SAME template at 2.6 m and stand it on the same floor: its
        // shoulders are above the tube man's head and its hips are in his chest.
        let params = BodyParams {
            height_m: 2.6,
            ..Default::default()
        };
        let tall = build_template(BodyPlan::BipedCanonical, &params).unwrap();
        let outside = globals_of(&tall)
            .iter()
            .filter(|g| !bvh.contains(**g))
            .count();
        assert!(
            outside > 4,
            "a 2.6 m rig standing in a 1.8 m body must put joints outside it; \
             only {outside} of {} are out, so the containment check cannot fail",
            tall.skeleton.len()
        );
    }

    /// Refinement can only improve containment: a step that would leave the mesh
    /// is refused and counted, so `joints_inside` never falls with iterations.
    #[test]
    fn refinement_never_moves_a_joint_out_of_the_mesh() {
        let bvh = tube_man();
        let mut inside = Vec::new();
        for iterations in [0u32, 1, 2, 4, 8] {
            let opts = FitOptions {
                plan: BodyPlan::BipedCanonical,
                refine_iterations: iterations,
                ..Default::default()
            };
            let (_, report) = fit_template(&bvh, &opts).expect("fits");
            inside.push(report.joints_inside);
        }
        assert!(
            inside.windows(2).all(|w| w[1] >= w[0]),
            "containment fell as refinement ran: {inside:?}"
        );
    }

    /// **Refinement pulls an outside joint IN**, which is the half of
    /// [`medial_step`] that only exists because the first cut did not have it.
    ///
    /// Measured on a body whose arms stop above where the template's proportions
    /// put its hands (3 cm short — the exact shape the first `tube_man` had, and
    /// the reason this rule was written). With no refinement the hands are out;
    /// with it they are in.
    #[test]
    fn an_overhanging_joint_is_pulled_into_the_mesh() {
        let mut tris: Vec<Tri> = Vec::new();
        let mut part = |min: [f64; 3], max: [f64; 3]| {
            tris.extend(crate::bvh::tests::box_tris(
                DVec3::from_array(min),
                DVec3::from_array(max),
            ));
        };
        part([-0.20, 0.90, -0.12], [0.20, 1.50, 0.12]);
        part([-0.10, 1.48, -0.10], [0.10, 1.80, 0.10]);
        // Short arms: they stop at 0.75, and a 1.8 m template's hands land at
        // 0.72.
        part([-0.30, 0.75, -0.08], [-0.16, 1.48, 0.08]);
        part([0.16, 0.75, -0.08], [0.30, 1.48, 0.08]);
        part([-0.18, 0.0, -0.09], [-0.02, 0.98, 0.09]);
        part([0.02, 0.0, -0.09], [0.18, 0.98, 0.09]);
        let short_armed = Bvh::new(tris);

        let raw = FitOptions {
            plan: BodyPlan::BipedCanonical,
            refine_iterations: 0,
            ..Default::default()
        };
        let (_, unrefined) = fit_template(&short_armed, &raw).expect("fits");
        assert!(
            unrefined.joints_inside < unrefined.joints,
            "the fixture must start with a joint outside, or this proves nothing"
        );
        let (_, refined) = fit_template(&short_armed, &fixture_opts()).expect("fits");
        assert_eq!(
            refined.joints_inside, refined.joints,
            "refinement must pull the overhanging joints in (was \
             {}/{} unrefined)",
            unrefined.joints_inside, unrefined.joints
        );
    }

    #[test]
    fn two_fits_are_identical() {
        let bvh = tube_man();
        let a = fit_template(&bvh, &fixture_opts()).unwrap();
        let b = fit_template(&bvh, &fixture_opts()).unwrap();
        assert_eq!(a.0, b.0, "the skeleton must be reproducible");
        assert_eq!(a.1, b.1, "and so must the report");
        assert_eq!(
            inf_asset::encode(&a.0).unwrap(),
            inf_asset::encode(&b.0).unwrap(),
            "byte-identical through the asset codec"
        );
    }

    #[test]
    fn a_degenerate_mesh_refuses_as_a_value() {
        // A single flat quad: no depth, so no volume.
        let flat = Bvh::new(vec![
            Tri {
                a: DVec3::new(-1.0, 0.0, -1.0),
                b: DVec3::new(1.0, 0.0, -1.0),
                c: DVec3::new(1.0, 0.0, 1.0),
            },
            Tri {
                a: DVec3::new(-1.0, 0.0, -1.0),
                b: DVec3::new(1.0, 0.0, 1.0),
                c: DVec3::new(-1.0, 0.0, 1.0),
            },
        ]);
        assert!(matches!(
            fit_template(&flat, &fixture_opts()),
            Err(FitError::Degenerate {
                what: "the height",
                ..
            })
        ));
        assert!(matches!(
            fit_template(&Bvh::new(Vec::new()), &fixture_opts()),
            Err(FitError::Degenerate { .. })
        ));
    }

    /// A mesh with no bilateral symmetry is refused **by name**, and the control
    /// is the same shape made symmetric — so the refusal is about the asymmetry
    /// and not about the fixture.
    #[test]
    fn an_asymmetric_mesh_refuses_by_name() {
        let mut tris =
            crate::bvh::tests::box_tris(DVec3::new(-0.2, 0.0, -0.2), DVec3::new(0.2, 1.8, 0.2));
        // One arm, on one side only, and a long one — so no plane mirrors this
        // shape onto itself.
        tris.extend(crate::bvh::tests::box_tris(
            DVec3::new(0.2, 1.2, -0.1),
            DVec3::new(1.6, 1.5, 0.1),
        ));
        let lopsided = Bvh::new(tris);
        match fit_template(&lopsided, &fixture_opts()) {
            Err(FitError::NoSymmetryPlane { best, tolerance }) => {
                assert!(best > tolerance, "{best} vs {tolerance}");
            }
            other => panic!("expected NoSymmetryPlane, got {other:?}"),
        }

        // The control: give it the other arm and it fits.
        let mut tris =
            crate::bvh::tests::box_tris(DVec3::new(-0.2, 0.0, -0.2), DVec3::new(0.2, 1.8, 0.2));
        for sign in [-1.0f64, 1.0] {
            let (a, b) = if sign < 0.0 { (-1.6, -0.2) } else { (0.2, 1.6) };
            tris.extend(crate::bvh::tests::box_tris(
                DVec3::new(a, 1.2, -0.1),
                DVec3::new(b, 1.5, 0.1),
            ));
        }
        let symmetric = Bvh::new(tris);
        assert!(
            fit_template(&symmetric, &fixture_opts()).is_ok(),
            "the symmetric control must fit, or the refusal above is the fixture"
        );
    }

    /// A quadruped fits too, and takes its LENGTH from the mesh rather than from
    /// the defaults.
    #[test]
    fn a_quadruped_takes_its_length_from_the_mesh() {
        let mut tris = crate::bvh::tests::box_tris(
            DVec3::new(-0.30, 0.55, -0.90),
            DVec3::new(0.30, 1.00, 0.90),
        );
        for x in [-0.24f64, 0.12] {
            for z in [-0.80f64, 0.55] {
                tris.extend(crate::bvh::tests::box_tris(
                    DVec3::new(x, 0.0, z),
                    DVec3::new(x + 0.12, 0.70, z + 0.20),
                ));
            }
        }
        let beast = Bvh::new(tris);
        let opts = FitOptions {
            plan: BodyPlan::Quadruped,
            ..Default::default()
        };
        let (asset, report) = fit_template(&beast, &opts).expect("a quadruped fits");
        assert_eq!(report.joints, asset.skeleton.len());
        assert!(
            asset.skeleton.index_of("foot_l0").is_some()
                && asset.skeleton.index_of("foot_r1").is_some(),
            "a quadruped names its feet per girdle"
        );
        assert!(report.height_m > 0.9 && report.height_m < 1.1, "{report:?}");
    }

    #[test]
    fn the_mirror_name_rule_is_the_one_written_down() {
        assert_eq!(mirror_name("hand_l").as_deref(), Some("hand_r"));
        assert_eq!(mirror_name("upper_leg_r").as_deref(), Some("upper_leg_l"));
        assert_eq!(mirror_name("foot_l0").as_deref(), Some("foot_r0"));
        assert_eq!(
            mirror_name("lower_leg_r12").as_deref(),
            Some("lower_leg_l12")
        );
        // Midline joints, and anything unrecognised.
        assert_eq!(mirror_name("hips"), None);
        assert_eq!(mirror_name("spine_1"), None);
        assert_eq!(mirror_name("girdle_0"), None);
    }
}
