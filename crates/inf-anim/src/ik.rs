//! **Inverse kinematics** (P24.2): two-bone and FABRIK, as a post-pass over an
//! evaluated pose.
//!
//! A clip says where a hand is. IK says where a hand *must be* — on the sword
//! hilt, on the ladder rung, on the ground the foot is actually standing on — and
//! bends the chain behind it to get there. Both solvers here take a chain of
//! joint indices, a target in **model space**, and hand back a pose whose chain
//! reaches it (or, when it cannot, extends fully toward it).
//!
//! # THE PORTABILITY LAW, and what it cost
//!
//! An IK result reaches the pose, the pose reaches `state_bytes`, and
//! `state_bytes` is compared between the editor's Simulate and the shipped
//! player — so **every arithmetic operation here has to be bit-identical across
//! targets**. That is the P14 law (`f32` `sin`/`cos` from `std` are not), and it
//! rules out the textbook formulations of both solvers:
//!
//! * **Two-bone is the law of cosines in its POSITIONAL form.** The usual
//!   write-up computes the elbow angle with `acos` and rebuilds the bone with
//!   `sin`/`cos`. This one never names an angle: the projection of the mid joint
//!   along the root→target axis is `(d² + l₁² − l₂²) / 2d`, its offset from that
//!   axis is `sqrt(l₁² − x²)`, and the direction of the offset comes from a cross
//!   product. Four divisions, two square roots, no transcendentals.
//! * **FABRIK is already sqrt-only** — it is nothing but "walk back along this
//!   direction by this length" — which is exactly why it is the general-chain
//!   solver here.
//! * **`Quat::from_rotation_arc` is NOT usable**, and that is the trap worth
//!   writing down: its ordinary branch is a cross product and a normalize, but
//!   its *antiparallel* branch calls `Quat::from_axis_angle(axis, PI)`, which
//!   reaches `sin_cos` inside glam where no grep of this crate would ever see it
//!   — the same shape as the P23.5 finding about `DQuat::from_axis_angle`.
//!   [`rotation_between`] is the hand-rolled replacement, and its 180° case is
//!   exact by construction (`sin(π/2) = 1`, `cos(π/2) = 0`) rather than computed.
//!
//! # Iteration counts are fixed, never a tolerance
//!
//! [`FABRIK_ITERATIONS`] is a constant, and the solve does not stop early on a
//! residual. A `while error > tol` loop makes the iteration count depend on a
//! floating comparison and therefore makes the *answer* depend on rounding — the
//! P23.5 ruling about the LSCM solver, restated on a solver whose output is
//! compared byte-for-byte between two processes. The reached distance is
//! **reported** instead ([`IkReport::reach_error`]), so "did it get there" is a
//! number rather than a hidden branch.
//!
//! # Refusals are values, and an unreachable target is not a refusal
//!
//! A chain that is too short, a non-finite target, a bone of zero length: those
//! are [`IkError`]s. A target the chain simply cannot reach is **not** — the
//! answer is full extension toward it, which is what a real limb does, and it is
//! never a NaN. [`IkReport::reached`] says which happened.

use glam::{Mat4, Quat, Vec3};

use crate::pose::{global_transforms, Pose};
use crate::skeleton::Skeleton;
use crate::template::{ConeLimit, JointLimit};

/// How many forward/backward sweeps [`solve_fabrik`] runs.
///
/// Ten is far past convergence for the 2-to-6-joint chains a character rig has
/// (FABRIK's error falls roughly geometrically, and a 4-joint chain is under a
/// micrometre by six), and it is a **constant** for the reason the module docs
/// give: a count that depends on a residual makes the result depend on rounding.
pub const FABRIK_ITERATIONS: usize = 10;

/// Lengths below this are treated as zero.
///
/// In metres. A bone shorter than a tenth of a millimetre carries no direction
/// worth normalizing, and dividing by it is how a chain becomes NaN.
pub const MIN_BONE_LENGTH_M: f32 = 1.0e-4;

/// A target within this of the chain's reach counts as reached.
///
/// A *reading* threshold, not a control-flow one: it decides what
/// [`IkReport::reached`] says and nothing else. The solve runs the same number of
/// iterations either way.
pub const REACH_TOLERANCE_M: f32 = 1.0e-3;

/// What a solve did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IkReport {
    /// Distance from the chain's tip to the target after solving (metres).
    pub reach_error: f32,
    /// Whether the tip landed within [`REACH_TOLERANCE_M`] of the target.
    ///
    /// `false` is the honest answer for a target beyond the chain's reach; the
    /// pose is still valid and fully extended toward it.
    pub reached: bool,
    /// The chain's total bone length — its reach (metres).
    pub chain_length: f32,
    /// **How many of the chain's joints a [`JointLimit`] clamped** (P24.3).
    ///
    /// Observable on purpose, and it is what makes "the elbow does not bend
    /// backwards" a thing a gate can assert rather than infer from a pose: a
    /// clamp that silently did nothing and a chain that never needed one are the
    /// same picture, and they are different numbers here.
    ///
    /// Counts only joints the clamp actually **moved** — a hinge already inside
    /// its range is not reported, so this reads as "how often the limit was
    /// load-bearing" and not "how many limits were consulted".
    pub clamped: u32,
}

/// Why a solve refused.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum IkError {
    /// Fewer than two joints: there is no bone to bend.
    #[error("an IK chain needs at least 2 joints, got {0}")]
    ChainTooShort(usize),
    /// A chain naming a joint the skeleton does not have.
    #[error("the IK chain names joint {joint}, but the skeleton has {joints}")]
    NoSuchJoint { joint: u16, joints: usize },
    /// A chain whose joints are not a parent-to-child walk.
    ///
    /// Refused rather than solved anyway: a "chain" of unrelated joints produces
    /// a pose in which rotating one does not move the next, so the solve
    /// converges on nothing while looking like it ran.
    #[error("joint {child} is not a child of {parent}, so this is not a chain")]
    NotAChain { parent: u16, child: u16 },
    /// A bone of (effectively) zero length — see [`MIN_BONE_LENGTH_M`].
    #[error("the bone from joint {parent} to {child} is {length} m long, which has no direction")]
    DegenerateBone {
        parent: u16,
        child: u16,
        length: f32,
    },
    /// A non-finite target or pole.
    #[error("{what} is not finite: {value:?}")]
    NonFinite { what: &'static str, value: [f32; 3] },
}

/// Whether `len` is a **usable** bone length: finite, and at least
/// [`MIN_BONE_LENGTH_M`].
///
/// A *positive* predicate, deliberately. The P24.2 audit's B1 finding was that
/// `len < MIN_BONE_LENGTH_M` is **false** for a NaN, so the guard written the
/// obvious way waves every non-finite length through to arithmetic that then
/// panics or manufactures a NaN. Writing the good case and negating the *call*
/// puts non-finite values on the refusal side without a negated float
/// comparison — which clippy distrusts for precisely this reason.
fn usable_length(len: f32) -> bool {
    len.is_finite() && len >= MIN_BONE_LENGTH_M
}

/// **Two-bone IK in positional form** — root, mid, tip, and a pole that decides
/// which way the joint bends.
///
/// Returns the new `(mid, tip)` in the same space as the inputs. The tip lands on
/// the target when it is reachable and at full extension toward it when it is
/// not; nothing here can produce a NaN from finite inputs.
///
/// `pole` is a point the mid joint bends *toward* — a knee's forward, an elbow's
/// back. When it is degenerate (on the root→target line) the previous bend plane
/// is used, and when that is degenerate too a fixed orthonormal fallback is,
/// which keeps the answer defined rather than arbitrary.
pub fn two_bone_positions(
    root: Vec3,
    mid: Vec3,
    tip: Vec3,
    target: Vec3,
    pole: Vec3,
) -> (Vec3, Vec3) {
    // **Finiteness first, on the POINTS** — measured while writing the B1
    // regression: `root.x = +inf` gives `l1 = +inf`, which satisfies
    // `l1 >= MIN_BONE_LENGTH_M` perfectly well and walks the whole computation
    // into NaN. A polarity-flipped length guard closes the NaN half of the audit
    // finding and not the infinity half; this closes both, at the primitive, so a
    // direct caller is as safe as one coming through `solve_chain`.
    if !(root.is_finite() && mid.is_finite() && tip.is_finite() && target.is_finite()) {
        return (mid, tip);
    }
    let l1 = (mid - root).length();
    let l2 = (tip - mid).length();
    let to_target = target - root;
    let d_raw = to_target.length();
    // **Polarity, not taste** (P24.2 audit B1). `x < MIN` is FALSE for a NaN, so
    // the written-the-obvious-way guard waves a non-finite length straight
    // through — and `f32::clamp` below asserts `min <= max`, which panics inside
    // a fixed step. `!(x >= MIN)` is TRUE for a NaN, so every non-finite input
    // takes the refusal branch. The `+inf` sibling was worse than the panic: no
    // assert fires, `[NaN; 4]` lands in `pose.locals`, and it rides
    // `state_bytes` into the palette.
    if !usable_length(l1) || !usable_length(l2) || !usable_length(d_raw) {
        return (mid, tip);
    }
    let u = to_target / d_raw;
    // Clamp the reach into the triangle inequality's open interval. Both bounds
    // are nudged inward: at exactly `l1 + l2` the offset below is `sqrt(0)`, which
    // is fine, but at exactly `|l1 − l2|` the chain folds back on itself and the
    // bend direction becomes undefined.
    let lo = (l1 - l2).abs() + MIN_BONE_LENGTH_M;
    let hi = l1 + l2 - MIN_BONE_LENGTH_M;
    let d = d_raw.clamp(lo.min(hi), hi.max(lo));

    // The law of cosines, never naming an angle: `x` is the mid joint's
    // projection onto the root→target axis, `h` its distance from it.
    let x = (d * d + l1 * l1 - l2 * l2) / (2.0 * d);
    let h = (l1 * l1 - x * x).max(0.0).sqrt();

    // The bend plane: perpendicular to `u`, pointing at the pole.
    let bend = perpendicular_toward(u, pole - root, mid - root);
    let new_mid = root + u * x + bend * h;
    // The tip sits on the target when reachable, and at the chain's full reach
    // along the same direction when not — a real limb's answer, not a refusal.
    let new_tip = root + u * d_raw.min(l1 + l2);
    (new_mid, new_tip)
}

/// **FABRIK** over a chain of points with fixed bone lengths.
///
/// `points[0]` is pinned (the chain's root); `points.last()` is driven to
/// `target`. Runs exactly [`FABRIK_ITERATIONS`] forward/backward sweeps — see the
/// module docs for why a fixed count and not a tolerance.
///
/// An unreachable target is handled **before** the sweeps, exactly: every joint
/// is laid along the root→target direction at its own bone length. That is the
/// limit the sweeps converge to anyway, reached in closed form, so an unreachable
/// target costs nothing and cannot drift.
pub fn fabrik(points: &mut [Vec3], target: Vec3) {
    let n = points.len();
    if n < 2 {
        return;
    }
    // Same rule at the other entry point.
    if !target.is_finite() || !points.iter().all(|p| p.is_finite()) {
        return;
    }
    let lengths: Vec<f32> = (1..n)
        .map(|i| (points[i] - points[i - 1]).length())
        .collect();
    let total: f32 = lengths.iter().sum();
    let root = points[0];
    let to_target = target - root;
    let reach = to_target.length();
    if !usable_length(reach) {
        return;
    }
    if reach >= total {
        // Out of reach: full extension, in closed form.
        let u = to_target / reach;
        let mut at = root;
        for (i, len) in lengths.iter().enumerate() {
            at += u * *len;
            points[i + 1] = at;
        }
        return;
    }
    for _ in 0..FABRIK_ITERATIONS {
        // Backward: pin the tip to the target and walk to the root.
        points[n - 1] = target;
        for i in (0..n - 1).rev() {
            points[i] = step_toward(points[i + 1], points[i], lengths[i]);
        }
        // Forward: pin the root and walk back out.
        points[0] = root;
        for i in 1..n {
            points[i] = step_toward(points[i - 1], points[i], lengths[i - 1]);
        }
    }
}

/// The smallest twist a clamp counts as having **moved** a joint, radians.
///
/// ~6e-4°. The clamp rebuilds a quaternion through `psin64`/`pcos64` even when
/// the angle was already in range, and that round trip is not bit-exact, so a
/// zero threshold would report every limited joint as clamped and
/// [`IkReport::clamped`] would say nothing. Far below any authored range and far
/// above the rebuild's error.
const CLAMP_EPSILON_RAD: f64 = 1.0e-5;

/// **Clamp one local rotation into a hinge's authored range** (P24.3).
///
/// Returns the clamped rotation and whether it actually moved.
///
/// # What "0 degrees" means, and why the bind pose is an input
///
/// [`JointLimit`]'s range is measured **from the bind pose** — `hinge_x(knee,
/// -150, 0)` describes a leg that is straight at rest and flexes to 150°
/// backwards. So the clamp is applied to the *delta from bind*, not to the raw
/// local rotation, and the bind rotation is put back afterwards. Clamping the
/// raw local would make the range mean "from identity", which is a different
/// (and, for any rig whose bind pose is not the identity, wrong) statement.
///
/// # Hinges only, deliberately, and it is counted
///
/// A limit with **exactly one** free axis is a hinge and is applied exactly: the
/// delta is swing-twist decomposed about that axis, the swing (the two locked
/// axes) is discarded, and the twist angle is clamped. A limit free on two or
/// three axes has no canonical decomposition — three independent Euler clamps
/// depend on the order you pick and gimbal-lock at the poles — so it is **left alone**
/// rather than approximated. [`build_template`](crate::template::build_template)
/// emits hinges and nothing else, so this covers every limit the engine
/// currently produces; the gap is in ROADMAP §12's P24 block rather than hidden
/// behind a plausible-looking cone solve.
///
/// # Portable arithmetic
///
/// `patan2_64` / `psin64` / `pcos64`, in `f64`, converting once at the wire — the
/// P14 law (`std` trig is not bit-identical across targets) reaches here because
/// the pose this edits is folded into `state_bytes` and compared between the
/// editor's Simulate and the shipped player.
fn clamp_to_limit(local: Quat, bind: Quat, limit: &JointLimit) -> (Quat, bool) {
    // Exactly one free axis ⇒ a hinge. Anything else is not applied (see docs).
    let mut axis_idx = None;
    for a in 0..3 {
        if limit.is_free(a) {
            if axis_idx.is_some() {
                return (local, false);
            }
            axis_idx = Some(a);
        }
    }
    let mut axis = glam::DVec3::ZERO;
    // A fully locked limit (no free axis) pins the joint to its bind pose, which
    // is the coherent reading of "this joint may not rotate": the twist below is
    // then clamped to the empty range [0, 0] about an arbitrary axis and the
    // swing is discarded, so the delta collapses to the identity.
    let a = axis_idx.unwrap_or(0);
    axis[a] = 1.0;

    let delta = (bind.inverse() * local).normalize();
    let d = glam::DVec3::new(delta.x as f64, delta.y as f64, delta.z as f64);
    let along = d.dot(axis);
    // Signed twist about `axis`: q = (axis·sin(θ/2), cos(θ/2)), so
    // θ = 2·atan2(q.xyz·axis, q.w) — and `patan2_64` handles the (0, 0) case as
    // 0, which is the identity's answer and the right one.
    let mut angle = 2.0 * inf_math::patan2_64(along, delta.w as f64);
    // `atan2` puts the half-angle in (-π, π], so θ lands in (-2π, 2π]. Fold it
    // into (-π, π] so a 190° twist clamps as −170° rather than as +190°, which
    // an author's range would reject for the wrong reason.
    if angle > std::f64::consts::PI {
        angle -= std::f64::consts::TAU;
    } else if angle < -std::f64::consts::PI {
        angle += std::f64::consts::TAU;
    }
    let lo = (limit.min_deg[a] as f64).to_radians();
    let hi = (limit.max_deg[a] as f64).to_radians();
    // A limit authored backwards (min > max) would make `clamp` panic. Treat it
    // as the degenerate empty range at `min`, which is what "you may not leave
    // this angle" reads as, rather than aborting a fixed step.
    let clamped_angle = if lo <= hi { angle.clamp(lo, hi) } else { lo };

    let half = clamped_angle * 0.5;
    let s = inf_math::psin64(half);
    let v = axis * s;
    let twist = Quat::from_xyzw(
        v.x as f32,
        v.y as f32,
        v.z as f32,
        inf_math::pcos64(half) as f32,
    )
    .normalize();
    let out = (bind * twist).normalize();
    // "Moved" is measured on the ANGLE, not on the quaternion: the rebuild is not
    // bit-exact even for an in-range joint (see `CLAMP_EPSILON_RAD`), and the
    // swing this discards is zero for the hinge poses the solver produces.
    let moved = (clamped_angle - angle).abs() > CLAMP_EPSILON_RAD;
    (out, moved)
}

/// **Clamp one local rotation into a swing-twist [`ConeLimit`]** (SK1b).
///
/// Returns the clamped rotation and whether it actually moved.
///
/// # The gap this closes
///
/// [`ConeLimit`] shipped on SK1a's one `.inf_skel` bump and the SK1a audit
/// recorded what that left behind: *"`ConeLimit` is authored and enforced by
/// nothing"* — [`clamp_to_limit`] reads `min_deg`/`max_deg` only, the ragdoll
/// builder reads neither, and no generator produced one. This is the first
/// consumer, and [`crate::grip`]'s finger curl is what needed it: three
/// independent per-axis ranges either forbid a legal finger pose at the corners
/// or admit an illegal one at the diagonals, which is exactly why a hand cannot
/// be described by a box.
///
/// # The decomposition, and the two clamps
///
/// The delta from bind is split as `swing · twist` about the cone's own axis
/// ([`crate::drive::twist_about`] is the same projection, and this is the other
/// half of it). The **swing** — how far the bone leans off the axis — is clamped
/// to `swing_deg` by rebuilding it at the cone's half-angle about the same swing
/// axis, which is a *rescale* and not a discard: a finger asked to curl 150°
/// through a 90° cone comes back curled 90° in the direction it was asked, not
/// straight. The **twist** — the roll about the axis — is clamped into
/// `twist_deg`, folded into `(-π, π]` first for [`clamp_to_limit`]'s reason.
///
/// # Portable arithmetic
///
/// `patan2_64` / `psin64` / `pcos64` in `f64`, converting once at the wire —
/// the P14 law, for the same reason [`clamp_to_limit`] obeys it: this edits a
/// pose that is folded into `state_bytes` and compared between the editor's
/// Simulate and the shipped player.
pub fn clamp_to_cone(local: Quat, bind: Quat, cone: &ConeLimit) -> (Quat, bool) {
    use glam::{DQuat, DVec3};

    let axis = DVec3::new(
        cone.axis[0] as f64,
        cone.axis[1] as f64,
        cone.axis[2] as f64,
    );
    let len2 = axis.length_squared();
    // A degenerate or non-finite axis describes no cone. Left alone rather than
    // guessed at — the `drive_twists` discipline, at the other end of the same
    // decomposition.
    if !axis.is_finite() || len2 <= 1.0e-12 || !cone.swing_deg.is_finite() {
        return (local, false);
    }
    let axis = axis / len2.sqrt();

    let d = DQuat::from_xyzw(
        local.x as f64,
        local.y as f64,
        local.z as f64,
        local.w as f64,
    );
    let b = DQuat::from_xyzw(bind.x as f64, bind.y as f64, bind.z as f64, bind.w as f64);
    if !d.is_finite() || !b.is_finite() {
        return (local, false);
    }
    let delta = (b.inverse() * d).normalize();
    if !delta.is_finite() {
        return (local, false);
    }

    // ── the twist half: the component about `axis` ──
    let v = DVec3::new(delta.x, delta.y, delta.z);
    let proj = axis * v.dot(axis);
    let tw_len2 = proj.length_squared() + delta.w * delta.w;
    // A half turn about a perpendicular axis has no twist to speak of; the
    // identity is the only answer that is not a division by zero (the
    // `twist_about` degenerate case, restated in f64).
    let twist = if tw_len2 <= 1.0e-12 {
        DQuat::IDENTITY
    } else {
        let inv = 1.0 / tw_len2.sqrt();
        DQuat::from_xyzw(proj.x * inv, proj.y * inv, proj.z * inv, delta.w * inv)
    };
    let swing = (delta * twist.inverse()).normalize();

    // ── the swing clamp ──
    //
    // Canonicalized to the `w >= 0` hemisphere first, so the angle below lands
    // in `[0, π]` and "how far off the axis" is a magnitude rather than a
    // quantity that flips sign with the quaternion's double cover.
    let swing = if swing.w < 0.0 {
        DQuat::from_xyzw(-swing.x, -swing.y, -swing.z, -swing.w)
    } else {
        swing
    };
    let sv = DVec3::new(swing.x, swing.y, swing.z);
    let sv_len = sv.length();
    let swing_angle = 2.0 * inf_math::patan2_64(sv_len, swing.w);
    let swing_max = (cone.swing_deg.max(0.0) as f64).to_radians();
    let mut moved = false;
    let swing = if swing_angle > swing_max && sv_len > 1.0e-9 {
        moved = true;
        let half = swing_max * 0.5;
        let dir = sv / sv_len;
        let s = inf_math::psin64(half);
        DQuat::from_xyzw(dir.x * s, dir.y * s, dir.z * s, inf_math::pcos64(half)).normalize()
    } else {
        swing
    };

    // ── the twist clamp ──
    let along = DVec3::new(twist.x, twist.y, twist.z).dot(axis);
    let mut twist_angle = 2.0 * inf_math::patan2_64(along, twist.w);
    if twist_angle > std::f64::consts::PI {
        twist_angle -= std::f64::consts::TAU;
    } else if twist_angle < -std::f64::consts::PI {
        twist_angle += std::f64::consts::TAU;
    }
    let lo = (cone.twist_deg[0] as f64).to_radians();
    let hi = (cone.twist_deg[1] as f64).to_radians();
    // A range authored backwards is the degenerate empty one at `min`, not a
    // panic in a fixed step — `clamp_to_limit`'s rule, verbatim.
    let clamped_twist = if lo <= hi {
        twist_angle.clamp(lo, hi)
    } else {
        lo
    };
    if (clamped_twist - twist_angle).abs() > CLAMP_EPSILON_RAD {
        moved = true;
    }
    let half = clamped_twist * 0.5;
    let s = inf_math::psin64(half);
    let tv = axis * s;
    let twist = DQuat::from_xyzw(tv.x, tv.y, tv.z, inf_math::pcos64(half)).normalize();

    if !moved {
        // Nothing was out of range, so the rebuild is not spent: returning the
        // input unchanged keeps an in-range joint **bit**-identical, which a
        // round trip through `psin64`/`pcos64` would not (`CLAMP_EPSILON_RAD`'s
        // own reason, one level up).
        return (local, false);
    }
    let out = (b * (swing * twist)).normalize();
    if !out.is_finite() {
        return (local, false);
    }
    (
        Quat::from_xyzw(out.x as f32, out.y as f32, out.z as f32, out.w as f32).normalize(),
        true,
    )
}

/// **Apply one authored [`JointLimit`] to a local rotation** — the door both the
/// chain solver and [`crate::grip`]'s finger curl go through.
///
/// # A cone wins over the box
///
/// A limit that carries a [`ConeLimit`] is described **by the cone**, and the
/// per-axis `min_deg`/`max_deg` box is not read. Two descriptions of one joint's
/// freedom would have to agree, and nothing can make them: a 90° cone and a
/// three-axis box disagree at every diagonal by construction. The cone is the
/// more specific statement, so it is the one that applies — and it is what lets
/// [`JointLimit::cone_only`] author a finger without spelling a box that would
/// otherwise read as *fully locked* ([`clamp_to_limit`] pins a joint with no free
/// axis to its bind pose, which is the coherent reading of an all-zero box and
/// exactly the wrong answer for a finger).
pub fn apply_joint_limit(local: Quat, bind: Quat, limit: &JointLimit) -> (Quat, bool) {
    match &limit.cone {
        Some(cone) => clamp_to_cone(local, bind, cone),
        None => clamp_to_limit(local, bind, limit),
    }
}

/// **Apply an IK chain to a pose**, in model space.
///
/// `chain` is joint indices from the chain's root to its tip, each the parent of
/// the next. The pose's *rotations* are edited so the tip reaches `target`;
/// translations and scales are untouched, because a bone that changes length to
/// reach something is not IK.
///
/// A 3-joint chain uses [`two_bone_positions`] (exact, one shot, and the pole
/// decides the bend); anything longer uses [`fabrik`].
///
/// `limits` is the skeleton's [`JointLimit`] side table — `SkeletonAsset::limits`
/// — and is consulted **per joint as that joint's rotation is written**, not as a
/// pass afterwards. That ordering is load-bearing: joint `i+1`'s aim is measured
/// against where joint `i` has just been put (see the loop below), so clamping
/// after the fact would leave every downstream joint aimed at a position its
/// parent is no longer in. Pass `&[]` for an unlimited solve.
///
/// **The parameter is required rather than defaulted** (P24.3). It has exactly
/// one production caller — `inf_ecs::pose::step_pose_evaluation`, which has the
/// `SkeletonAsset` in hand — and a second, defaulting door is how "the solver
/// respects limits" becomes true of one call site and false of the other.
pub fn solve_chain(
    skeleton: &Skeleton,
    pose: &mut Pose,
    chain: &[u16],
    target: Vec3,
    pole: Option<Vec3>,
    limits: &[JointLimit],
) -> Result<IkReport, IkError> {
    if chain.len() < 2 {
        return Err(IkError::ChainTooShort(chain.len()));
    }
    finite("an IK target", target)?;
    if let Some(p) = pole {
        finite("an IK pole", p)?;
    }
    for &j in chain {
        if skeleton.joint(j as usize).is_none() {
            return Err(IkError::NoSuchJoint {
                joint: j,
                joints: skeleton.len(),
            });
        }
    }
    for w in chain.windows(2) {
        let child = skeleton.joint(w[1] as usize).expect("checked above");
        if child.parent != Some(w[0]) {
            return Err(IkError::NotAChain {
                parent: w[0],
                child: w[1],
            });
        }
    }

    let positions = |pose: &Pose| -> Vec<Vec3> {
        let globals = global_transforms(skeleton, pose);
        chain
            .iter()
            .map(|&j| globals[j as usize].transform_point3(Vec3::ZERO))
            .collect()
    };
    // **The pose is an input too** (audit B1). `solve_chain` validated its target
    // and its pole and never the thing it was solving *over*: a pose carrying a
    // NaN — from a clip whose keys are corrupt, a blend that divided by zero, or
    // an earlier goal on an overlapping chain — reached the arithmetic below and
    // came back as `[NaN; 4]` in `pose.locals`, which the trace then committed.
    // Checked on the CHAIN's joints only: an unrelated joint's NaN is not this
    // solve's business, and refusing the whole pose for it would make one bad
    // clip disable every chain on the character.
    for &j in chain {
        let l = &pose.locals[j as usize];
        for (what, v) in [
            (
                "an IK chain joint's translation",
                Vec3::from_array(l.translation),
            ),
            ("an IK chain joint's scale", Vec3::from_array(l.scale)),
        ] {
            finite(what, v)?;
        }
        if !l.rotation.iter().all(|c| c.is_finite()) {
            return Err(IkError::NonFinite {
                what: "an IK chain joint's rotation",
                value: [l.rotation[0], l.rotation[1], l.rotation[2]],
            });
        }
    }

    let start = positions(pose);
    // …and the model-space positions the pose produced, because a finite local
    // TRS can still compose into a non-finite global (the M3 lesson: the
    // finiteness contract holds on RESULTS).
    for (i, p) in start.iter().enumerate() {
        if !p.is_finite() {
            return Err(IkError::NonFinite {
                what: "an IK chain joint's model-space position",
                value: [chain[i] as f32, p.x, p.y],
            });
        }
    }
    let mut lengths = Vec::with_capacity(chain.len() - 1);
    for i in 1..start.len() {
        let len = (start[i] - start[i - 1]).length();
        if !usable_length(len) {
            return Err(IkError::DegenerateBone {
                parent: chain[i - 1],
                child: chain[i],
                length: len,
            });
        }
        lengths.push(len);
    }
    let chain_length: f32 = lengths.iter().sum();

    // Where the chain WANTS to be.
    let mut goal = start.clone();
    if chain.len() == 3 {
        // The pole defaults to the current mid joint, which keeps the existing
        // bend — the least surprising answer when a caller has no opinion.
        let pole = pole.unwrap_or(start[1]);
        let (mid, tip) = two_bone_positions(start[0], start[1], start[2], target, pole);
        goal[1] = mid;
        goal[2] = tip;
    } else {
        fabrik(&mut goal, target);
    }

    // Turn the goal positions into ROTATIONS, one joint at a time, recomputing
    // the pose's globals after each — so joint `i+1`'s aim is measured against
    // where joint `i` has just put it, rather than against a stale frame.
    let mut clamped = 0u32;
    for i in 0..chain.len() - 1 {
        let globals = global_transforms(skeleton, pose);
        let here = globals[chain[i] as usize].transform_point3(Vec3::ZERO);
        let child_now = globals[chain[i + 1] as usize].transform_point3(Vec3::ZERO);
        let cur = child_now - here;
        let want = goal[i + 1] - here;
        if !usable_length(cur.length()) || !usable_length(want.length()) {
            continue;
        }
        let delta = rotation_between(cur.normalize(), want.normalize());
        // Compose in the joint's own frame: the delta is a MODEL-space rotation,
        // so it is pushed through the parent's inverse to become a local one.
        let parent_rot = match skeleton.joint(chain[i] as usize).and_then(|j| j.parent) {
            Some(p) => rotation_of(&globals[p as usize]),
            None => Quat::IDENTITY,
        };
        let global_rot = rotation_of(&globals[chain[i] as usize]);
        let new_global = (delta * global_rot).normalize();
        let mut local = (parent_rot.inverse() * new_global).normalize();
        // ── P24.3: the authored range, applied HERE (SK1b: cones too) ──
        //
        // Inside the loop and not after it, for the reason `solve_chain`'s doc
        // gives: the next iteration recomputes `global_transforms` from this
        // write, so a clamp applied afterwards would leave every downstream joint
        // aimed at a position its parent had already been pulled out of. This is
        // what closes P24.2's ledger entry — an elbow can no longer bend
        // backwards because a target asked it to.
        if let Some(limit) = limits.iter().find(|l| l.joint == chain[i]) {
            let bind = skeleton
                .joint(chain[i] as usize)
                .map(|j| Quat::from_array(j.local_bind.rotation).normalize())
                .unwrap_or(Quat::IDENTITY);
            let (fixed, moved) = apply_joint_limit(local, bind, limit);
            local = fixed;
            if moved {
                clamped += 1;
            }
        }
        pose.locals[chain[i] as usize].rotation = local.to_array();
    }

    let end = positions(pose);
    let reach_error = (end[end.len() - 1] - target).length();
    Ok(IkReport {
        reach_error,
        reached: reach_error <= REACH_TOLERANCE_M,
        chain_length,
        clamped,
    })
}

/// The rotation taking unit `a` onto unit `b`, **without trigonometry**.
///
/// The replacement for `Quat::from_rotation_arc`, whose antiparallel branch
/// reaches `sin_cos` inside glam (see the module docs). Three cases:
///
/// * nearly parallel — identity;
/// * nearly antiparallel — a half turn about any axis perpendicular to `a`,
///   written *exactly* as `(axis, 0)`: a quaternion for angle θ is
///   `(axis·sin(θ/2), cos(θ/2))`, and at θ = π that is `(axis·1, 0)` with no
///   transcendental evaluated;
/// * otherwise — the standard `(a × b, 1 + a·b)`, normalized.
pub fn rotation_between(a: Vec3, b: Vec3) -> Quat {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    if dot > 1.0 - 1e-6 {
        return Quat::IDENTITY;
    }
    if dot < -1.0 + 1e-6 {
        let axis = any_perpendicular(a);
        return Quat::from_xyzw(axis.x, axis.y, axis.z, 0.0);
    }
    let c = a.cross(b);
    Quat::from_xyzw(c.x, c.y, c.z, 1.0 + dot).normalize()
}

/// A unit vector perpendicular to `v`, chosen deterministically.
///
/// Crossed with whichever cardinal axis `v` leans on least, so the cross product
/// is never near-degenerate and the answer is a pure function of `v`.
fn any_perpendicular(v: Vec3) -> Vec3 {
    let a = v.abs();
    let axis = if a.x <= a.y && a.x <= a.z {
        Vec3::X
    } else if a.y <= a.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let c = v.cross(axis);
    let len = c.length();
    if usable_length(len) {
        c / len
    } else {
        Vec3::X
    }
}

/// A unit vector perpendicular to `axis`, leaning toward `hint`, falling back to
/// `fallback` and then to a fixed choice.
///
/// The bend direction of a two-bone chain. Two fallbacks rather than one because
/// both hints are legitimately degenerate in practice: a pole exactly on the
/// root→target line (an author who put the pole at the target) and a chain
/// already perfectly straight (a leg at full extension, which is where a foot IK
/// most often starts).
fn perpendicular_toward(axis: Vec3, hint: Vec3, fallback: Vec3) -> Vec3 {
    for candidate in [hint, fallback] {
        let projected = candidate - axis * candidate.dot(axis);
        let len = projected.length();
        if len >= MIN_BONE_LENGTH_M {
            return projected / len;
        }
    }
    any_perpendicular(axis)
}

/// `to`, moved to sit exactly `length` from `from` along the same direction.
fn step_toward(from: Vec3, to: Vec3, length: f32) -> Vec3 {
    let d = to - from;
    let len = d.length();
    if !usable_length(len) {
        // No direction to preserve; extend along +Y so the chain stays the right
        // length rather than collapsing.
        return from + Vec3::Y * length;
    }
    from + d * (length / len)
}

/// The rotation part of a global transform, normalized.
fn rotation_of(m: &Mat4) -> Quat {
    let (_, r, _) = m.to_scale_rotation_translation();
    r.normalize()
}

fn finite(what: &'static str, v: Vec3) -> Result<(), IkError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(IkError::NonFinite {
            what,
            value: v.to_array(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Joint, JointTransform};

    /// A straight chain of `n` joints, each 1 m above the last, with identity
    /// inverse binds.
    fn chain_skeleton(n: usize) -> Skeleton {
        let mut joints = Vec::new();
        for i in 0..n {
            joints.push(Joint {
                name: format!("j{i}"),
                parent: (i > 0).then(|| i as u16 - 1),
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::from_trs(
                    if i == 0 { Vec3::ZERO } else { Vec3::Y },
                    Quat::IDENTITY,
                    Vec3::ONE,
                ),
            });
        }
        Skeleton::new(joints).unwrap()
    }

    fn tip_of(sk: &Skeleton, pose: &Pose, joint: u16) -> Vec3 {
        global_transforms(sk, pose)[joint as usize].transform_point3(Vec3::ZERO)
    }

    /// **Two bones reach a reachable target**, exactly.
    #[test]
    fn a_two_bone_chain_reaches_its_target() {
        let sk = chain_skeleton(3);
        let mut pose = Pose::rest(&sk);
        // Straight up, 2 m of reach. Ask for a point 1.4 m out at 45°-ish.
        let target = Vec3::new(1.2, 1.2, 0.0);
        let report = solve_chain(
            &sk,
            &mut pose,
            &[0, 1, 2],
            target,
            Some(Vec3::new(2.0, 1.0, 0.0)),
            &[],
        )
        .expect("solves");
        assert!(
            report.reached,
            "the tip is {} m from a reachable target",
            report.reach_error
        );
        let tip = tip_of(&sk, &pose, 2);
        assert!((tip - target).length() < REACH_TOLERANCE_M, "{tip:?}");
        assert!((report.chain_length - 2.0).abs() < 1e-5);
        // The bones did not stretch.
        let g = global_transforms(&sk, &pose);
        let p: Vec<Vec3> = (0..3).map(|i| g[i].transform_point3(Vec3::ZERO)).collect();
        assert!(((p[1] - p[0]).length() - 1.0).abs() < 1e-4, "{p:?}");
        assert!(((p[2] - p[1]).length() - 1.0).abs() < 1e-4, "{p:?}");
    }

    /// **An unreachable target is a full extension, never a NaN** — a refusal as
    /// a value, in the shape a limb actually has.
    #[test]
    fn an_unreachable_target_extends_fully_toward_it() {
        for chain in [vec![0u16, 1, 2], vec![0, 1, 2, 3, 4]] {
            let sk = chain_skeleton(chain.len());
            let mut pose = Pose::rest(&sk);
            let reach = (chain.len() - 1) as f32;
            let target = Vec3::new(100.0, 0.0, 0.0);
            let report = solve_chain(&sk, &mut pose, &chain, target, None, &[]).expect("solves");
            assert!(!report.reached, "{report:?}");
            let tip = tip_of(&sk, &pose, *chain.last().unwrap());
            assert!(tip.is_finite(), "an unreachable target produced {tip:?}");
            // Fully extended: the tip is at the chain's whole reach…
            assert!(
                (tip.length() - reach).abs() < 1e-3,
                "chain {chain:?}: tip at {tip:?}, reach {reach}"
            );
            // …and pointing AT the target.
            assert!(
                tip.normalize().dot(target.normalize()) > 0.999,
                "chain {chain:?}: tip {tip:?} does not point at the target"
            );
            assert!(
                (report.reach_error - (100.0 - reach)).abs() < 1e-2,
                "{report:?}"
            );
        }
    }

    /// A longer chain converges through FABRIK.
    #[test]
    fn a_long_chain_converges_within_epsilon() {
        let sk = chain_skeleton(5);
        let mut pose = Pose::rest(&sk);
        let target = Vec3::new(2.0, 2.0, 1.0);
        let report =
            solve_chain(&sk, &mut pose, &[0, 1, 2, 3, 4], target, None, &[]).expect("solves");
        assert!(report.reached, "{report:?}");
        assert!(tip_of(&sk, &pose, 4).is_finite());
        // Bone lengths survive.
        let g = global_transforms(&sk, &pose);
        for i in 1..5 {
            let len = (g[i].transform_point3(Vec3::ZERO) - g[i - 1].transform_point3(Vec3::ZERO))
                .length();
            assert!((len - 1.0).abs() < 1e-3, "bone {i} is {len} m");
        }
    }

    /// The pole decides the bend: two solves to the same target with opposite
    /// poles put the mid joint on opposite sides.
    #[test]
    fn the_pole_decides_which_way_the_joint_bends() {
        let sk = chain_skeleton(3);
        let target = Vec3::new(0.0, 1.2, 0.0);
        let mut front = Pose::rest(&sk);
        let mut back = Pose::rest(&sk);
        solve_chain(
            &sk,
            &mut front,
            &[0, 1, 2],
            target,
            Some(Vec3::new(0.0, 0.6, 5.0)),
            &[],
        )
        .unwrap();
        solve_chain(
            &sk,
            &mut back,
            &[0, 1, 2],
            target,
            Some(Vec3::new(0.0, 0.6, -5.0)),
            &[],
        )
        .unwrap();
        let (a, b) = (tip_of(&sk, &front, 1), tip_of(&sk, &back, 1));
        assert!(a.z > 0.2, "the front pole must bend forward: {a:?}");
        assert!(b.z < -0.2, "the back pole must bend backward: {b:?}");
    }

    #[test]
    fn every_refusal_is_a_value() {
        let sk = chain_skeleton(3);
        let mut pose = Pose::rest(&sk);
        assert_eq!(
            solve_chain(&sk, &mut pose, &[0], Vec3::ZERO, None, &[]),
            Err(IkError::ChainTooShort(1))
        );
        assert_eq!(
            solve_chain(&sk, &mut pose, &[0, 9], Vec3::ZERO, None, &[]),
            Err(IkError::NoSuchJoint {
                joint: 9,
                joints: 3
            })
        );
        // 0 → 2 skips a joint, so it is not a chain.
        assert_eq!(
            solve_chain(&sk, &mut pose, &[0, 2], Vec3::ZERO, None, &[]),
            Err(IkError::NotAChain {
                parent: 0,
                child: 2
            })
        );
        assert!(matches!(
            solve_chain(
                &sk,
                &mut pose,
                &[0, 1, 2],
                Vec3::new(f32::NAN, 0.0, 0.0),
                None,
                &[],
            ),
            Err(IkError::NonFinite { .. })
        ));
        // …and the pose is untouched by any of them.
        assert_eq!(pose, Pose::rest(&sk));
    }

    /// Bit-for-bit pose equality.
    ///
    /// `assert_eq!` cannot express "unchanged" for a pose carrying a NaN:
    /// `NaN != NaN`, so a pose compares unequal to *itself*. These tests are
    /// about a refusal being **inert**, which is a statement about bytes.
    fn same_bits(a: &Pose, b: &Pose) -> bool {
        a.locals.len() == b.locals.len()
            && a.locals.iter().zip(&b.locals).all(|(x, y)| {
                x.translation
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(y.translation.iter().map(|v| v.to_bits()))
                    && x.rotation
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(y.rotation.iter().map(|v| v.to_bits()))
                    && x.scale
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(y.scale.iter().map(|v| v.to_bits()))
            })
    }

    /// **A non-finite pose is a REFUSAL, never a panic and never a NaN**
    /// (P24.2 audit B1).
    ///
    /// Both halves of the finding, because they failed differently:
    ///
    /// * **NaN panicked.** `l1 < MIN` is false for a NaN, so the guard waved it
    ///   through to `f32::clamp`, whose `assert!(min <= max)` aborts — inside a
    ///   fixed step, taking the session with it.
    /// * **`+inf` did not panic, which was worse.** No assert fires, the
    ///   arithmetic produces `[NaN; 4]`, and it lands in `pose.locals` → the
    ///   evaluated pose → `state_bytes` → the GPU palette. A silent NaN in a
    ///   committed trace is the failure a panic at least announces.
    #[test]
    fn a_non_finite_pose_is_refused_by_name_and_leaves_the_pose_alone() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            // (a) a non-finite TRANSLATION on a chain joint.
            let sk = chain_skeleton(3);
            let mut pose = Pose::rest(&sk);
            pose.locals[1].translation[1] = bad;
            let before = pose.clone();
            let err = solve_chain(
                &sk,
                &mut pose,
                &[0, 1, 2],
                Vec3::new(1.0, 1.0, 0.0),
                None,
                &[],
            )
            .expect_err("a non-finite pose must refuse");
            assert!(
                matches!(err, IkError::NonFinite { .. }),
                "{bad}: got {err:?}"
            );
            assert!(
                same_bits(&pose, &before),
                "{bad}: a refusal must leave the pose alone"
            );

            // (b) a non-finite ROTATION, which the translation check cannot see.
            let mut pose = Pose::rest(&sk);
            pose.locals[1].rotation[0] = bad;
            let before = pose.clone();
            assert!(matches!(
                solve_chain(
                    &sk,
                    &mut pose,
                    &[0, 1, 2],
                    Vec3::new(1.0, 1.0, 0.0),
                    None,
                    &[]
                ),
                Err(IkError::NonFinite { .. })
            ));
            assert!(same_bits(&pose, &before));

            // (c) a non-finite SCALE, which composes into the globals.
            let mut pose = Pose::rest(&sk);
            pose.locals[0].scale[0] = bad;
            let before = pose.clone();
            assert!(matches!(
                solve_chain(
                    &sk,
                    &mut pose,
                    &[0, 1, 2],
                    Vec3::new(1.0, 1.0, 0.0),
                    None,
                    &[]
                ),
                Err(IkError::NonFinite { .. })
            ));
            assert!(same_bits(&pose, &before));
        }
    }

    /// The primitive underneath it: `two_bone_positions` on non-finite inputs
    /// returns its inputs rather than panicking or manufacturing a NaN.
    ///
    /// Called directly, so the guard is exercised where it lives — the refusal in
    /// `solve_chain` above would satisfy the test without this ever running.
    #[test]
    fn two_bone_positions_never_panics_and_never_produces_a_nan() {
        let ok = (Vec3::ZERO, Vec3::Y, Vec3::new(0.0, 2.0, 0.0));
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for case in 0..4 {
                let (mut root, mut mid, mut tip) = ok;
                let mut target = Vec3::new(1.0, 1.0, 0.0);
                match case {
                    0 => root.x = bad,
                    1 => mid.y = bad,
                    2 => tip.z = bad,
                    _ => target.x = bad,
                }
                let (m, t) = two_bone_positions(root, mid, tip, target, Vec3::X);
                let unchanged = m
                    .to_array()
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(mid.to_array().iter().map(|v| v.to_bits()))
                    && t.to_array()
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(tip.to_array().iter().map(|v| v.to_bits()));
                assert!(
                    (m.is_finite() && t.is_finite()) || unchanged,
                    "{bad} at case {case}: produced {m:?} / {t:?}, which is \
                     neither a finite answer nor the untouched input"
                );
            }
        }
        // …and FABRIK, the other entry point.
        for bad in [f32::NAN, f32::INFINITY] {
            let mut pts = vec![Vec3::ZERO, Vec3::Y, Vec3::new(0.0, 2.0, 0.0)];
            let before = pts.clone();
            fabrik(&mut pts, Vec3::new(bad, 1.0, 0.0));
            assert!(
                pts.iter().all(|p| p.is_finite()) || pts == before,
                "{bad}: fabrik produced {pts:?}"
            );
            assert!(
                pts.iter().all(|p| !p.is_nan()),
                "{bad}: fabrik manufactured a NaN: {pts:?}"
            );
        }
    }

    #[test]
    fn a_degenerate_bone_refuses_by_name() {
        let mut sk = chain_skeleton(3);
        // Collapse the second bone.
        let joints: Vec<Joint> = sk
            .joints()
            .iter()
            .enumerate()
            .map(|(i, j)| {
                let mut j = j.clone();
                if i == 2 {
                    j.local_bind.translation = [0.0; 3];
                }
                j
            })
            .collect();
        sk = Skeleton::new(joints).unwrap();
        let mut pose = Pose::rest(&sk);
        assert!(matches!(
            solve_chain(
                &sk,
                &mut pose,
                &[0, 1, 2],
                Vec3::new(1.0, 1.0, 0.0),
                None,
                &[]
            ),
            Err(IkError::DegenerateBone {
                parent: 1,
                child: 2,
                ..
            })
        ));
    }

    /// Deterministic: the same inputs give the same pose bits.
    #[test]
    fn two_solves_are_bit_identical() {
        let sk = chain_skeleton(4);
        let target = Vec3::new(1.1, 1.7, -0.4);
        let mut a = Pose::rest(&sk);
        let mut b = Pose::rest(&sk);
        solve_chain(&sk, &mut a, &[0, 1, 2, 3], target, None, &[]).unwrap();
        solve_chain(&sk, &mut b, &[0, 1, 2, 3], target, None, &[]).unwrap();
        assert_eq!(a, b);
    }

    /// [`rotation_between`] handles the antiparallel case **without** the trig
    /// branch `Quat::from_rotation_arc` takes — and gets the right answer.
    #[test]
    fn the_antiparallel_rotation_is_exact_and_trig_free() {
        for v in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(1.0, 2.0, 3.0).normalize(),
        ] {
            let q = rotation_between(v, -v);
            let out = q * v;
            assert!(
                (out + v).length() < 1e-5,
                "{v:?} → {out:?} is not a half turn"
            );
            // The scalar part is EXACTLY zero — cos(π/2), written rather than
            // computed. A `from_axis_angle(_, PI)` would land near it and not on
            // it, and "near" is what a byte comparison between two machines
            // cannot use.
            assert_eq!(q.w, 0.0, "the half turn must be exact");
        }
        // The ordinary cases still work.
        assert!(rotation_between(Vec3::X, Vec3::X).abs_diff_eq(Quat::IDENTITY, 1e-6));
        let q = rotation_between(Vec3::X, Vec3::Y);
        assert!((q * Vec3::X - Vec3::Y).length() < 1e-6);
    }

    /// A solved pose leaves translations and scales alone — IK bends bones, it
    /// does not stretch them.
    #[test]
    fn only_rotations_move() {
        let sk = chain_skeleton(4);
        let rest = Pose::rest(&sk);
        let mut pose = rest.clone();
        solve_chain(
            &sk,
            &mut pose,
            &[0, 1, 2, 3],
            Vec3::new(1.0, 2.0, 0.5),
            None,
            &[],
        )
        .unwrap();
        for (a, b) in rest.locals.iter().zip(&pose.locals) {
            assert_eq!(a.translation, b.translation);
            assert_eq!(a.scale, b.scale);
        }
        assert_ne!(rest, pose, "…and something DID move");
    }

    // ── P24.3: the authored joint limits, applied ─────────────────────────

    /// A hinge about local **Z**, which is the axis `chain_skeleton`'s bones bend
    /// about: its joints run along +Y, so a bend in the XY plane is a rotation
    /// about Z. `JointLimit::hinge_x` is the shape `build_template` emits for a
    /// rig whose bones run along its own X; the *kind* is what is under test
    /// here, not the axis, and picking the wrong one would have made every
    /// assertion below vacuous (measured: the first draft did, and the clamp
    /// silently discarded the whole bend as "swing").
    fn hinge_z(joint: u16, min_deg: f32, max_deg: f32) -> JointLimit {
        JointLimit {
            joint,
            min_deg: [0.0, 0.0, min_deg],
            max_deg: [0.0, 0.0, max_deg],
            cone: None,
        }
    }

    /// An elbow that may flex toward **+X only**.
    ///
    /// The permitted range is [0°, 150°] and anything negative is the backwards
    /// bend the limit exists to forbid. The sign is **measured, not derived**: the
    /// elbow's local rotation is the fold of the forearm relative to the upper
    /// arm, not the world-space swing of the limb, so reasoning "rotation about
    /// +Z takes +Y onto −X" gets it backwards — which the first draft of this
    /// test did, and the two assertions below are what caught it.
    fn elbow_limit() -> JointLimit {
        hinge_z(1, 0.0, 150.0)
    }

    /// **The headline gate for the P24.2 ledger entry** — an elbow asked to bend
    /// backwards does not.
    ///
    /// The unlimited solve reaches the target by flexing the elbow the wrong way
    /// (its range is 0°..150° and the pole puts the bend on the negative side);
    /// the limited solve refuses to leave the range, is *reported* as having
    /// clamped, and lands somewhere else. Three separate claims, because "the
    /// pose differs" alone would also pass if the clamp merely jittered it.
    #[test]
    fn a_hinge_elbow_will_not_bend_backwards() {
        let sk = chain_skeleton(3);
        // A target that pulls the mid joint to NEGATIVE X, which for this rig is
        // the direction the hinge forbids.
        let target = Vec3::new(-1.2, 1.2, 0.0);
        let pole = Some(Vec3::new(-2.0, 1.0, 0.0));

        let mut free = Pose::rest(&sk);
        let free_report = solve_chain(&sk, &mut free, &[0, 1, 2], target, pole, &[]).unwrap();
        let mut held = Pose::rest(&sk);
        let held_report =
            solve_chain(&sk, &mut held, &[0, 1, 2], target, pole, &[elbow_limit()]).unwrap();

        // 1. The unlimited solve really does go out of range — otherwise this
        //    test measures a clamp that had nothing to clamp.
        let free_z = free.locals[1].rotation[2];
        assert!(
            free_z < -1e-3,
            "the unlimited elbow must bend the FORBIDDEN way for this to be a test; got {free_z}"
        );
        assert!(free_report.reached, "the unlimited solve reaches");
        assert_eq!(free_report.clamped, 0, "nothing was limited");

        // 2. The limited solve stays inside [−150, 0]°, measured on the joint's
        //    own local rotation rather than on the report.
        let held_z = held.locals[1].rotation[2];
        assert!(
            held_z >= -1e-4,
            "the hinge left its 0..150 range: local rotation z = {held_z}"
        );

        // 3. …and it SAYS so, which is what makes the clamp observable.
        assert_eq!(held_report.clamped, 1, "exactly the elbow was clamped");
        assert_ne!(free.locals[1].rotation, held.locals[1].rotation);
    }

    /// A hinge already inside its range is left alone **and is not counted** —
    /// the anti-vacuity twin of the test above. Without this, a `clamped` that
    /// simply counted "limits consulted" would pass everything.
    #[test]
    fn an_in_range_hinge_is_neither_moved_nor_counted() {
        let sk = chain_skeleton(3);
        // A target on the +X side: the elbow flexes the way its range allows.
        let target = Vec3::new(1.2, 1.2, 0.0);
        let pole = Some(Vec3::new(2.0, 1.0, 0.0));
        let mut free = Pose::rest(&sk);
        let free_report = solve_chain(&sk, &mut free, &[0, 1, 2], target, pole, &[]).unwrap();
        let mut held = Pose::rest(&sk);
        let held_report =
            solve_chain(&sk, &mut held, &[0, 1, 2], target, pole, &[elbow_limit()]).unwrap();
        assert_eq!(held_report.clamped, 0, "an in-range hinge is not clamped");
        assert!(free_report.reached && held_report.reached);
        // The rebuild through psin/pcos is not bit-exact, so the poses are
        // compared within a tolerance far tighter than any authored range.
        for (a, b) in free.locals.iter().zip(&held.locals) {
            for k in 0..4 {
                assert!(
                    (a.rotation[k] - b.rotation[k]).abs() < 1e-5,
                    "an in-range hinge moved the pose: {a:?} vs {b:?}"
                );
            }
        }
    }

    /// A limit whose range is measured **from the bind pose**, not from the
    /// identity — the claim `clamp_to_limit`'s doc rests on.
    ///
    /// The rig's elbow is bent 45° at bind and the hinge permits ±0° (a fully
    /// pinned joint), so the clamped result must be the *bind* rotation, which is
    /// visibly not the identity. Clamping the raw local would have produced the
    /// identity and this would fail.
    #[test]
    fn the_range_is_measured_from_the_bind_pose() {
        let bend = Quat::from_rotation_x(std::f32::consts::FRAC_PI_4);
        let mut joints = Vec::new();
        for i in 0..3usize {
            joints.push(Joint {
                name: format!("j{i}"),
                parent: (i > 0).then(|| i as u16 - 1),
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::from_trs(
                    if i == 0 { Vec3::ZERO } else { Vec3::Y },
                    if i == 1 { bend } else { Quat::IDENTITY },
                    Vec3::ONE,
                ),
            });
        }
        let sk = Skeleton::new(joints).unwrap();
        let mut pose = Pose::rest(&sk);
        // A fully pinned hinge: min == max == 0, so the delta from bind collapses
        // to the identity and the local must come back as the bind rotation.
        let pinned = JointLimit {
            joint: 1,
            min_deg: [0.0; 3],
            max_deg: [0.0; 3],
            cone: None,
        };
        solve_chain(
            &sk,
            &mut pose,
            &[0, 1, 2],
            Vec3::new(1.5, 0.5, 0.0),
            None,
            &[pinned],
        )
        .unwrap();
        let got = Quat::from_array(pose.locals[1].rotation).normalize();
        assert!(
            got.abs_diff_eq(bend, 1e-5) || got.abs_diff_eq(-bend, 1e-5),
            "a pinned joint must return to its BIND rotation, got {got:?} want {bend:?}"
        );
        assert!(
            !got.abs_diff_eq(Quat::IDENTITY, 1e-3),
            "…and the bind rotation is not the identity, or this proves nothing"
        );
    }

    /// A limit free on **more than one axis** is left alone rather than
    /// approximated — stated as a test so the gap is a decision and not a
    /// surprise (ROADMAP §12's P24 block carries it).
    #[test]
    fn a_multi_axis_limit_is_not_applied() {
        let sk = chain_skeleton(3);
        let target = Vec3::new(-1.2, 1.2, 0.0);
        let pole = Some(Vec3::new(-2.0, 1.0, 0.0));
        let cone = JointLimit {
            joint: 1,
            min_deg: [0.0, -10.0, 0.0],
            max_deg: [150.0, 10.0, 0.0],
            cone: None,
        };
        let mut free = Pose::rest(&sk);
        solve_chain(&sk, &mut free, &[0, 1, 2], target, pole, &[]).unwrap();
        let mut held = Pose::rest(&sk);
        let report = solve_chain(&sk, &mut held, &[0, 1, 2], target, pole, &[cone]).unwrap();
        assert_eq!(report.clamped, 0);
        assert_eq!(
            free.locals, held.locals,
            "a two-axis limit must be byte-identical to no limit at all"
        );
    }

    /// A limit naming a joint that is not in the chain never fires, and a limit
    /// authored backwards (`min > max`) does not panic a fixed step.
    #[test]
    fn irrelevant_and_backwards_limits_are_survivable() {
        let sk = chain_skeleton(3);
        let target = Vec3::new(-1.2, 1.2, 0.0);
        let mut a = Pose::rest(&sk);
        let r = solve_chain(
            &sk,
            &mut a,
            &[0, 1, 2],
            target,
            None,
            &[JointLimit::hinge_x(9, 0.0, 10.0)],
        )
        .unwrap();
        assert_eq!(r.clamped, 0, "a limit on an absent joint cannot fire");

        let mut b = Pose::rest(&sk);
        let backwards = JointLimit {
            joint: 1,
            min_deg: [30.0, 0.0, 0.0],
            max_deg: [-30.0, 0.0, 0.0],
            cone: None,
        };
        let r = solve_chain(&sk, &mut b, &[0, 1, 2], target, None, &[backwards]).unwrap();
        assert!(b
            .locals
            .iter()
            .all(|l| l.rotation.iter().all(|c| c.is_finite())));
        // `is_free` is `min < max`, so a backwards range has NO free axis: the
        // joint is treated as pinned, which is a value and not a panic.
        let _ = r.clamped;
    }

    /// The clamp is **deterministic** and uses no `std` transcendental: two
    /// solves of the same inputs are bit-identical, which is the property the
    /// PIE-vs-shipping trace comparison rests on.
    #[test]
    fn the_clamp_is_bit_deterministic() {
        let sk = chain_skeleton(3);
        let mut a = Pose::rest(&sk);
        let mut b = Pose::rest(&sk);
        for p in [&mut a, &mut b] {
            solve_chain(
                &sk,
                p,
                &[0, 1, 2],
                Vec3::new(-1.1, 1.3, 0.2),
                Some(Vec3::new(-2.0, 1.0, 0.0)),
                &[elbow_limit()],
            )
            .unwrap();
        }
        assert_eq!(a, b);
    }

    /// **The cone is a cone**: a swing past its half-angle comes back AT the
    /// half-angle, in the direction it was asked for, and a swing inside it is
    /// returned bit-for-bit unchanged.
    ///
    /// The second half is the one that matters for the trace: a clamp that
    /// rebuilt every in-range rotation through `psin64`/`pcos64` would move a
    /// finger by an ulp on every step of every grip, and every determinism gate
    /// would still pass because both hosts would move it identically.
    #[test]
    fn a_cone_clamps_a_swing_and_leaves_an_in_range_one_alone() {
        use crate::template::ConeLimit;
        let cone = ConeLimit {
            axis: [1.0, 0.0, 0.0],
            swing_deg: 40.0,
            twist_deg: [-10.0, 10.0],
        };
        let bind = Quat::IDENTITY;
        // A swing about Z (perpendicular to the axis) of 20 deg: inside.
        let inside = about_z(20f32.to_radians());
        let (out, moved) = clamp_to_cone(inside, bind, &cone);
        assert!(!moved, "an in-range swing was reported as clamped");
        assert_eq!(
            out.to_array().map(f32::to_bits),
            inside.to_array().map(f32::to_bits),
            "an in-range rotation was rebuilt rather than returned"
        );
        // 90 deg: outside, and it comes back at 40 in the same plane.
        let outside = about_z(90f32.to_radians());
        let (out, moved) = clamp_to_cone(outside, bind, &cone);
        assert!(
            moved,
            "a 90 deg swing through a 40 deg cone was not clamped"
        );
        let angle = swing_deg_of(out, Vec3::X);
        assert!(
            (angle - 40.0).abs() < 0.05,
            "clamped to {angle} deg rather than 40"
        );
        // …in the direction it was asked for, not some canonical one.
        let v = Vec3::new(out.x, out.y, out.z);
        assert!(v.z > 0.0 && v.x.abs() < 1e-5 && v.y.abs() < 1e-5, "{out:?}");
        // The same swing the other way round comes back the other way round.
        let (back, _) = clamp_to_cone(about_z(-90f32.to_radians()), bind, &cone);
        assert!(Vec3::new(back.x, back.y, back.z).z < 0.0);

        // A ROLL about the axis is twist, and is clamped by `twist_deg`.
        let rolled = about_x(60f32.to_radians());
        let (out, moved) = clamp_to_cone(rolled, bind, &cone);
        assert!(moved);
        let twist = 2.0 * inf_math::patan2_64(out.x as f64, out.w as f64);
        assert!(
            (twist.to_degrees() - 10.0).abs() < 0.05,
            "a 60 deg roll clamped to {} deg rather than 10",
            twist.to_degrees()
        );
    }

    /// A cone that describes nothing leaves the joint alone, and a limit that
    /// carries one is described BY it — the `apply_joint_limit` precedence rule,
    /// which is what lets `JointLimit::cone_only` write an all-zero box.
    #[test]
    fn a_degenerate_cone_is_inert_and_a_cone_outranks_its_box() {
        use crate::template::{ConeLimit, JointLimit};
        let q = about_z(1.2);
        for bad in [
            ConeLimit {
                axis: [0.0; 3],
                swing_deg: 10.0,
                twist_deg: [-5.0, 5.0],
            },
            ConeLimit {
                axis: [f32::NAN, 0.0, 0.0],
                swing_deg: 10.0,
                twist_deg: [-5.0, 5.0],
            },
            ConeLimit {
                axis: [1.0, 0.0, 0.0],
                swing_deg: f32::NAN,
                twist_deg: [-5.0, 5.0],
            },
        ] {
            let (out, moved) = clamp_to_cone(q, Quat::IDENTITY, &bad);
            assert!(!moved && out == q, "{bad:?} moved a joint");
        }
        // A backwards twist range is the degenerate empty one at `min`, not a
        // panic in a fixed step.
        let backwards = ConeLimit {
            axis: [1.0, 0.0, 0.0],
            swing_deg: 180.0,
            twist_deg: [10.0, -10.0],
        };
        let (out, moved) = clamp_to_cone(about_x(1.0), Quat::IDENTITY, &backwards);
        assert!(moved && out.is_finite());

        // The precedence: an all-zero box would PIN this joint to bind if the box
        // were read, and the cone lets it swing 40 deg.
        let limit = JointLimit::cone_only(
            0,
            ConeLimit {
                axis: [1.0, 0.0, 0.0],
                swing_deg: 40.0,
                twist_deg: [-10.0, 10.0],
            },
        );
        let (out, _) = apply_joint_limit(about_z(20f32.to_radians()), Quat::IDENTITY, &limit);
        assert!(
            swing_deg_of(out, Vec3::X) > 19.0,
            "the box won over the cone and pinned the joint to bind"
        );
        // …and a limit with NO cone still goes to the hinge clamp.
        let hinge = JointLimit::hinge_x(0, 0.0, 30.0);
        let (out, moved) = apply_joint_limit(about_x(90f32.to_radians()), Quat::IDENTITY, &hinge);
        assert!(
            moved && (2.0 * inf_math::patan2_64(out.x as f64, out.w as f64)).to_degrees() < 30.5
        );
    }

    /// Portable rotation builders for the two arms above — the crate bans
    /// `Quat::from_rotation_x` on the pose path and a test fixture that used it
    /// would be checking the code under test with the thing the code under test
    /// is not allowed to use.
    fn about_x(angle: f32) -> Quat {
        let h = angle as f64 * 0.5;
        Quat::from_xyzw(
            inf_math::psin64(h) as f32,
            0.0,
            0.0,
            inf_math::pcos64(h) as f32,
        )
        .normalize()
    }

    fn about_z(angle: f32) -> Quat {
        let h = angle as f64 * 0.5;
        Quat::from_xyzw(
            0.0,
            0.0,
            inf_math::psin64(h) as f32,
            inf_math::pcos64(h) as f32,
        )
        .normalize()
    }

    /// How far `q` swings away from `axis`, degrees.
    fn swing_deg_of(q: Quat, axis: Vec3) -> f64 {
        let tw = crate::drive::twist_about(q, axis);
        let swing = (q * tw.inverse()).normalize();
        let swing = if swing.w < 0.0 { -swing } else { swing };
        let v = Vec3::new(swing.x, swing.y, swing.z).length() as f64;
        (2.0 * inf_math::patan2_64(v, swing.w as f64)).to_degrees()
    }
}
