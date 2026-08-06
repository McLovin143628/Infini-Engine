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

/// **Apply an IK chain to a pose**, in model space.
///
/// `chain` is joint indices from the chain's root to its tip, each the parent of
/// the next. The pose's *rotations* are edited so the tip reaches `target`;
/// translations and scales are untouched, because a bone that changes length to
/// reach something is not IK.
///
/// A 3-joint chain uses [`two_bone_positions`] (exact, one shot, and the pole
/// decides the bend); anything longer uses [`fabrik`].
pub fn solve_chain(
    skeleton: &Skeleton,
    pose: &mut Pose,
    chain: &[u16],
    target: Vec3,
    pole: Option<Vec3>,
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
        let local = (parent_rot.inverse() * new_global).normalize();
        pose.locals[chain[i] as usize].rotation = local.to_array();
    }

    let end = positions(pose);
    let reach_error = (end[end.len() - 1] - target).length();
    Ok(IkReport {
        reach_error,
        reached: reach_error <= REACH_TOLERANCE_M,
        chain_length,
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
            let report = solve_chain(&sk, &mut pose, &chain, target, None).expect("solves");
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
        let report = solve_chain(&sk, &mut pose, &[0, 1, 2, 3, 4], target, None).expect("solves");
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
        )
        .unwrap();
        solve_chain(
            &sk,
            &mut back,
            &[0, 1, 2],
            target,
            Some(Vec3::new(0.0, 0.6, -5.0)),
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
            solve_chain(&sk, &mut pose, &[0], Vec3::ZERO, None),
            Err(IkError::ChainTooShort(1))
        );
        assert_eq!(
            solve_chain(&sk, &mut pose, &[0, 9], Vec3::ZERO, None),
            Err(IkError::NoSuchJoint {
                joint: 9,
                joints: 3
            })
        );
        // 0 → 2 skips a joint, so it is not a chain.
        assert_eq!(
            solve_chain(&sk, &mut pose, &[0, 2], Vec3::ZERO, None),
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
                None
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
            let err = solve_chain(&sk, &mut pose, &[0, 1, 2], Vec3::new(1.0, 1.0, 0.0), None)
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
                solve_chain(&sk, &mut pose, &[0, 1, 2], Vec3::new(1.0, 1.0, 0.0), None),
                Err(IkError::NonFinite { .. })
            ));
            assert!(same_bits(&pose, &before));

            // (c) a non-finite SCALE, which composes into the globals.
            let mut pose = Pose::rest(&sk);
            pose.locals[0].scale[0] = bad;
            let before = pose.clone();
            assert!(matches!(
                solve_chain(&sk, &mut pose, &[0, 1, 2], Vec3::new(1.0, 1.0, 0.0), None),
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
                    "{bad} at case {case}: produced {m:?} / {t:?}, which is                      neither a finite answer nor the untouched input"
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
            solve_chain(&sk, &mut pose, &[0, 1, 2], Vec3::new(1.0, 1.0, 0.0), None),
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
        solve_chain(&sk, &mut a, &[0, 1, 2, 3], target, None).unwrap();
        solve_chain(&sk, &mut b, &[0, 1, 2, 3], target, None).unwrap();
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
        )
        .unwrap();
        for (a, b) in rest.locals.iter().zip(&pose.locals) {
            assert_eq!(a.translation, b.translation);
            assert_eq!(a.scale, b.scale);
        }
        assert_ne!(rest, pose, "…and something DID move");
    }
}
