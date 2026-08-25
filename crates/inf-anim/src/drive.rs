//! **The procedural drive pass** (SK1a): the bones a clip never authors.
//!
//! A production rig has two families of bone no animation track ever touches, and
//! before this module this engine had neither:
//!
//! * **twist bones**, which take a fraction of a neighbouring joint's roll so the
//!   skin between two joints rotates gradually instead of pinching at one of them
//!   ([`TwistDriver`] carries the rule, and its docs carry the law);
//! * **IK handles** (`ik_hand_l`, `ik_foot_r`, `ik_hand_gun`), which are markers a
//!   rig publishes so that a solver, an attachment or an exported clip can name
//!   "where the hand is" without walking the hierarchy. When nothing is driving
//!   them they must **follow their FK source**, or they sit at bind while the
//!   character walks away from them.
//!
//! [`drive_pose`] is the one Ring-0 rule for both, and it is called from exactly
//! one place — `inf_ecs::pose::step_pose_evaluation`, the fixed step's single pose
//! door — right after the layer stack has produced a pose and before any IK
//! corrects it.
//!
//! # Where it sits, and the bound that costs
//!
//! *After the layer stack* means a twist bone reflects the pose the **animation**
//! authored. It does **not** reflect the pose the IK passes below it go on to
//! correct: a foot IK solve that rolls an ankle 20° leaves `calf_twist_01_l`
//! showing the pre-solve roll for that frame. That is stated rather than hidden.
//! Fixing it means either running this pass twice (paying the whole thing to
//! correct four bones) or having each solver re-drive the twists in its own chain;
//! the wave that gives hand IK a real consumer is where that gets measured, and it
//! is routed there by name rather than guessed at here.
//!
//! # Determinism
//!
//! Everything is arithmetic, one `sqrt`, and [`inf_math::pslerp`] — **no `sin`, no
//! `cos`, no `atan2`, no `acos`** on the `f32` path. The P14 law (`f32` `std` trig
//! is not bit-portable) reaches this module because its output is folded into
//! `pose_state_bytes` and compared between the editor's PIE and the shipped
//! player. `the_drive_is_deterministic_and_portable` and the crate's
//! `portable_pose` source gate are what keep it true.
//!
//! # Absent costs nothing
//!
//! A rig with no drive tables — every `.inf_skel` written before this schema, every
//! imported glTF — takes the two early returns and its pose is byte-identical to
//! what it was before this module existed.

use glam::{Mat4, Quat, Vec3};

use crate::pose::Pose;
use crate::roles::{IkFollow, TwistDriver};
use crate::skeleton::Skeleton;

/// **Run the drive pass** over `pose`, in place.
///
/// Twists first (they read and write **local** rotations only, so they need no
/// global pass), then the IK handles in ascending joint order (they need globals,
/// and a handle whose parent is another handle must read its parent's *driven*
/// global — which is why the pass recomputes one column at a time instead of
/// taking a snapshot).
///
/// Returns how many bones it actually drove — an **engagement counter**, because
/// "the pass ran" and "the pass did anything" are different claims and a gate that
/// cannot tell them apart certifies a no-op.
pub fn drive_pose(
    skeleton: &Skeleton,
    pose: &mut Pose,
    twists: &[TwistDriver],
    follows: &[IkFollow],
) -> usize {
    let mut driven = drive_twists(skeleton, pose, twists);
    driven += drive_ik_follow(skeleton, pose, follows);
    driven
}

/// The twist half. Separated because it is the half that needs no globals, and
/// because a caller with no IK handles must not pay for a global pass.
pub fn drive_twists(skeleton: &Skeleton, pose: &mut Pose, twists: &[TwistDriver]) -> usize {
    if twists.is_empty() {
        return 0;
    }
    let n = skeleton.len().min(pose.locals.len());
    let mut driven = 0usize;
    for d in twists {
        let (joint, source) = (d.joint as usize, d.source as usize);
        if joint >= n || source >= n || joint == source {
            continue;
        }
        let axis = Vec3::from_array(d.axis);
        if !axis.is_finite() || !d.fraction.is_finite() {
            continue;
        }
        let len2 = axis.length_squared();
        if len2 <= 1.0e-12 {
            continue;
        }
        // Normalized here rather than trusted: an authored axis is content, and a
        // near-unit axis would bias the projection below.
        let axis = axis / len2.sqrt();
        let roll = twist_about(pose.locals[source].rotation_quat(), axis);
        let f = d.fraction.clamp(-1.0, 1.0);
        // A negative fraction COUNTERS the source's roll — see `TwistDriver`. The
        // conjugate is the inverse of a unit quaternion, exactly, with no divide.
        let target = if f < 0.0 { roll.conjugate() } else { roll };
        let driven_rot = inf_math::pslerp(Quat::IDENTITY, target, f.abs());
        pose.locals[joint].rotation = driven_rot.to_array();
        driven += 1;
    }
    driven
}

/// The IK-handle half: put each handle where its FK source is.
///
/// "Where its source is" means the **global** transform, so a handle parented to
/// the rig's root lands on a hand that is six joints away. The handle's own local
/// is solved for: `local = parent_global⁻¹ · source_global`.
pub fn drive_ik_follow(skeleton: &Skeleton, pose: &mut Pose, follows: &[IkFollow]) -> usize {
    if follows.is_empty() {
        return 0;
    }
    let joints = skeleton.joints();
    let n = joints.len().min(pose.locals.len());
    if n == 0 {
        return 0;
    }
    let mut globals = crate::pose::global_transforms(skeleton, pose);
    if globals.len() < n {
        return 0;
    }
    // **Ascending joint order matters**, and is an invariant of the table rather
    // than something this pass establishes: a handle whose parent is another
    // handle must see the parent AFTER it moved, and `ik_hand_l` under
    // `ik_hand_gun` is exactly that case on the mannequin. Sorting a copy here
    // would be an allocation per posed character per fixed step; `SkeletonAsset`'s
    // decode refuses an out-of-order table instead, so this walks it as it is.
    let mut driven = 0usize;
    for &f in follows {
        let (joint, source) = (f.joint as usize, f.source as usize);
        if joint >= n || source >= n || joint == source {
            continue;
        }
        let parent_global = match joints[joint].parent {
            Some(p) if (p as usize) < n => globals[p as usize],
            Some(_) => continue,
            None => Mat4::IDENTITY,
        };
        let local = parent_global.inverse() * globals[source];
        if !local.is_finite() {
            continue;
        }
        let (scale, rotation, translation) = local.to_scale_rotation_translation();
        pose.locals[joint].translation = translation.to_array();
        pose.locals[joint].rotation = rotation.to_array();
        pose.locals[joint].scale = scale.to_array();
        // The column this handle publishes, so a child handle reads the driven
        // value rather than the one it had before this loop started.
        globals[joint] = parent_global * local;
        driven += 1;
    }
    driven
}

/// The **twist component** of `q` about `axis` — the swing-twist decomposition's
/// second factor, and nothing else.
///
/// `q = swing · twist` where `twist` is a rotation about `axis`. Projecting the
/// quaternion's vector part onto the axis and renormalizing is the whole of it:
/// no angle is ever formed, so there is no `atan2` and no `sin` on this path.
///
/// The degenerate case is real and named: a rotation of exactly 180° about an axis
/// *perpendicular* to `axis` has a zero vector-projection **and** a zero `w`, so
/// there is no twist to speak of and the identity is the only answer that is not a
/// division by zero.
pub fn twist_about(q: Quat, axis: Vec3) -> Quat {
    let v = Vec3::new(q.x, q.y, q.z);
    let proj = axis * v.dot(axis);
    let len2 = proj.length_squared() + q.w * q.w;
    if len2 <= 1.0e-12 {
        return Quat::IDENTITY;
    }
    let inv = 1.0 / len2.sqrt();
    Quat::from_xyzw(proj.x * inv, proj.y * inv, proj.z * inv, q.w * inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Joint, JointTransform};

    /// Two rotations, compared through their dot product rather than
    /// `Quat::angle_between` — glam's is `acos_approx`, whose absolute error near
    /// a dot of 1 is worse than the tolerance a bit-exactness claim needs.
    fn close(a: Quat, b: Quat) -> bool {
        (1.0 - a.dot(b).abs()) < 1.0e-7
    }

    fn joint(name: &str, parent: Option<u16>, t: Vec3) -> Joint {
        Joint {
            name: name.into(),
            parent,
            inverse_bind: Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::from_trs(t, Quat::IDENTITY, Vec3::ONE),
        }
    }

    /// `root → arm → twist_a (1/3) → …`, plus a hand and an IK handle hanging off
    /// the root — the shape the Manny rig has, at four joints.
    fn rig() -> Skeleton {
        Skeleton::new(vec![
            joint("root", None, Vec3::ZERO),
            joint("arm", Some(0), Vec3::new(0.0, 1.0, 0.0)),
            joint("hand", Some(1), Vec3::new(1.0, 0.0, 0.0)),
            joint("twist_a", Some(1), Vec3::new(1.0 / 3.0, 0.0, 0.0)),
            joint("ik_hand", Some(0), Vec3::ZERO),
        ])
        .unwrap()
    }

    /// **The decomposition is a decomposition**: swing · twist reproduces the
    /// rotation, and the twist really is about the axis.
    #[test]
    fn the_twist_component_is_the_rotation_about_the_axis() {
        let axis = Vec3::X;
        for (i, q) in [
            Quat::from_rotation_x(0.9),
            Quat::from_rotation_y(0.4) * Quat::from_rotation_x(1.2),
            Quat::from_rotation_z(-0.7) * Quat::from_rotation_x(-0.3),
            Quat::IDENTITY,
        ]
        .into_iter()
        .enumerate()
        {
            let t = twist_about(q, axis);
            assert!(
                (t.length() - 1.0).abs() < 1.0e-5,
                "case {i}: not a unit quaternion"
            );
            // The twist's own axis is the given one (or it is the identity).
            let v = Vec3::new(t.x, t.y, t.z);
            assert!(
                v.length() < 1.0e-6 || v.normalize().cross(axis).length() < 1.0e-5,
                "case {i}: the twist is not about X"
            );
            // …and the residue is a pure swing: it has no component about X.
            let swing = q * t.inverse();
            assert!(
                Vec3::new(swing.x, swing.y, swing.z).dot(axis).abs() < 1.0e-5,
                "case {i}: the swing still carries roll"
            );
        }
        // A pure roll IS its own twist.
        let roll = Quat::from_rotation_x(0.8);
        assert!(close(twist_about(roll, Vec3::X), roll));
        // A 180° rotation about a PERPENDICULAR axis has no twist to extract, and
        // must not divide by zero.
        assert_eq!(
            twist_about(Quat::from_rotation_y(std::f32::consts::PI), Vec3::X),
            Quat::IDENTITY
        );
    }

    /// **The law, asserted**: the roll along a segment is linear in the position
    /// along it — and the sign convention delivers it from both ends.
    #[test]
    fn the_roll_along_a_segment_is_linear_in_the_position_along_it() {
        let sk = rig();
        let (arm, hand, twist) = (1u16, 2u16, 3u16);
        let axis = [1.0, 0.0, 0.0];

        // A LOWER segment: the roll comes from the distal child, so a bone one
        // third along adds one third of it.
        let mut pose = Pose::rest(&sk);
        pose.locals[hand as usize].rotation = Quat::from_rotation_x(1.2).to_array();
        let n = drive_twists(
            &sk,
            &mut pose,
            &[TwistDriver::new(twist, hand, axis, 1.0 / 3.0)],
        );
        assert_eq!(n, 1, "the pass drove nothing");
        let got = Quat::from_array(pose.locals[twist as usize].rotation);
        assert!(
            close(got, Quat::from_rotation_x(0.4)),
            "a third of 1.2 rad is 0.4, got {got:?}"
        );

        // An UPPER segment: the roll is the segment's own and is already inherited,
        // so a bone one third along GIVES BACK two thirds.
        let mut pose = Pose::rest(&sk);
        pose.locals[arm as usize].rotation = Quat::from_rotation_x(1.2).to_array();
        drive_twists(
            &sk,
            &mut pose,
            &[TwistDriver::new(twist, arm, axis, -2.0 / 3.0)],
        );
        let local = Quat::from_array(pose.locals[twist as usize].rotation);
        // Its own local counters two thirds…
        assert!(close(local, Quat::from_rotation_x(-0.8)), "{local:?}");
        // …so what the WORLD sees at that point is one third of the segment's roll,
        // which is the claim the law actually makes.
        let world = Quat::from_rotation_x(1.2) * local;
        assert!(
            close(world, Quat::from_rotation_x(0.4)),
            "the driven bone should show 1/3 of the roll, got {world:?}"
        );
    }

    /// A driver naming a joint that is not there, a zero axis, a NaN fraction —
    /// each costs its own row and nothing else.
    #[test]
    fn a_degenerate_driver_costs_its_own_row() {
        let sk = rig();
        let mut pose = Pose::rest(&sk);
        let before = pose.clone();
        let n = drive_twists(
            &sk,
            &mut pose,
            &[
                TwistDriver::new(99, 2, [1.0, 0.0, 0.0], 0.5),
                TwistDriver::new(3, 99, [1.0, 0.0, 0.0], 0.5),
                TwistDriver::new(3, 3, [1.0, 0.0, 0.0], 0.5),
                TwistDriver::new(3, 2, [0.0, 0.0, 0.0], 0.5),
                TwistDriver::new(3, 2, [1.0, 0.0, 0.0], f32::NAN),
                TwistDriver::new(3, 2, [f32::INFINITY, 0.0, 0.0], 0.5),
            ],
        );
        assert_eq!(n, 0, "every one of those is degenerate");
        assert_eq!(pose.locals, before.locals, "and none of them wrote a pose");
        // An empty table is free and touches nothing.
        assert_eq!(drive_twists(&sk, &mut pose, &[]), 0);
        assert_eq!(drive_ik_follow(&sk, &mut pose, &[]), 0);
        assert_eq!(pose.locals, before.locals);
    }

    /// **A handle follows its source into world space**, through a parent that is
    /// nowhere near it.
    #[test]
    fn an_ik_handle_lands_on_the_joint_it_follows() {
        let sk = rig();
        let (hand, ik) = (2u16, 4u16);
        let mut pose = Pose::rest(&sk);
        pose.locals[1].rotation = Quat::from_rotation_z(0.6).to_array();
        let n = drive_ik_follow(&sk, &mut pose, &[IkFollow::new(ik, hand)]);
        assert_eq!(n, 1);
        let g = crate::pose::global_transforms(&sk, &pose);
        let at = |i: usize| g[i].transform_point3(Vec3::ZERO);
        assert!(
            (at(ik as usize) - at(hand as usize)).length() < 1.0e-5,
            "the handle is at {:?}, the hand at {:?}",
            at(ik as usize),
            at(hand as usize)
        );
    }

    /// **A handle parented to another handle reads the DRIVEN parent.**
    ///
    /// The Manny hierarchy puts `ik_hand_l` and `ik_hand_r` under `ik_hand_gun`,
    /// which is itself a follow. A pass that snapshots the globals once places the
    /// child against the parent's *bind* frame — which is the whole arm away from
    /// where the parent just moved to, and looks perfectly plausible.
    #[test]
    fn a_handle_under_a_handle_reads_the_driven_parent() {
        // root → gun → child, and a hand off to the side.
        let sk = Skeleton::new(vec![
            joint("root", None, Vec3::ZERO),
            joint("hand", Some(0), Vec3::new(2.0, 1.0, 0.0)),
            joint("ik_gun", Some(0), Vec3::ZERO),
            joint("ik_child", Some(2), Vec3::ZERO),
        ])
        .unwrap();
        let mut pose = Pose::rest(&sk);
        let n = drive_ik_follow(
            &sk,
            &mut pose,
            // In ascending joint order, which is the table's own invariant: the
            // gun (2) before the child handle (3) that hangs off it.
            &[IkFollow::new(2, 1), IkFollow::new(3, 1)],
        );
        assert_eq!(n, 2);
        let g = crate::pose::global_transforms(&sk, &pose);
        let at = |i: usize| g[i].transform_point3(Vec3::ZERO);
        assert!((at(2) - at(1)).length() < 1.0e-5, "the gun missed the hand");
        assert!(
            (at(3) - at(1)).length() < 1.0e-5,
            "the child handle landed at {:?} rather than on the hand at {:?} — it \
             read a stale parent",
            at(3),
            at(1)
        );
        // And the child's LOCAL really is the identity, which is only true if it
        // was solved against the moved parent.
        assert!(Vec3::from_array(pose.locals[3].translation).length() < 1.0e-6);
    }

    /// Two runs, same bits. The output of this pass is folded into a determinism
    /// trace and compared across two processes.
    #[test]
    fn the_drive_is_deterministic_and_portable() {
        let sk = rig();
        let twists = [TwistDriver::new(3, 2, [1.0, 0.0, 0.0], 1.0 / 3.0)];
        let follows = [IkFollow::new(4, 2)];
        let run = || {
            let mut pose = Pose::rest(&sk);
            pose.locals[1].rotation = Quat::from_rotation_z(0.37).to_array();
            pose.locals[2].rotation = Quat::from_rotation_x(1.11).to_array();
            drive_pose(&sk, &mut pose, &twists, &follows);
            pose
        };
        let (a, b) = (run(), run());
        for (i, (x, y)) in a.locals.iter().zip(b.locals.iter()).enumerate() {
            for (u, v) in x
                .translation
                .iter()
                .chain(x.rotation.iter())
                .chain(x.scale.iter())
                .zip(
                    y.translation
                        .iter()
                        .chain(y.rotation.iter())
                        .chain(y.scale.iter()),
                )
            {
                assert_eq!(u.to_bits(), v.to_bits(), "joint {i} is not bit-stable");
            }
        }
        // …and the counter really counted both halves.
        let mut pose = Pose::rest(&sk);
        assert_eq!(drive_pose(&sk, &mut pose, &twists, &follows), 2);
    }
}
