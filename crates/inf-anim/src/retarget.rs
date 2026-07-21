//! Retargeting v1 (P11.3): copy a humanoid animation from one skeleton onto
//! another by **joint name**.
//!
//! A [`RetargetMap`] is a list of `(source_name → target_name)` joint pairs.
//! [`retarget_pose`] takes a pose sampled on the *source* skeleton and produces a
//! pose on the *target* skeleton, copying each mapped joint's **local rotation**
//! bind-pose-relatively so the two rigs' different rest poses line up:
//!
//! ```text
//! dst_local_rot = dst_bind_rot · (src_bind_rot⁻¹ · src_anim_rot)
//! ```
//!
//! Read right-to-left: `src_bind_rot⁻¹ · src_anim_rot` is the source joint's
//! rotation **relative to its own bind pose** (the "delta" the animation applies);
//! pre-composing the target's bind rotation re-expresses that delta in the target
//! rig's rest orientation. When the two skeletons are identical this collapses to
//! the identity `dst_local_rot = src_anim_rot`, so retargeting a rig onto itself
//! reproduces the pose (the `identity` gate).
//!
//! ## v1 scope
//!
//! * **Rotations** are copied for every mapped joint. **Translation** is copied
//!   **only for the root** (bind-relative: `dst_bind_t + (src_anim_t −
//!   src_bind_t)`) — limb bone lengths come from the *target* rig's bind pose, so
//!   copying child translations would stretch it. **Translation/height scaling**
//!   between differently-proportioned rigs is a documented follow-up.
//! * **Unmapped** target joints keep their bind (rest) transform.
//! * No IK pass — foot/hand fix-up after retargeting is a later deliverable.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::pose::Pose;
use crate::skeleton::Skeleton;

/// A humanoid name map: source joint name → target joint name.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetargetMap {
    /// `(source_name, target_name)` pairs. Order is irrelevant; duplicate targets
    /// take the last write.
    pub pairs: Vec<(String, String)>,
}

impl RetargetMap {
    /// An empty map.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from explicit `(source, target)` name pairs.
    pub fn from_pairs<I, A, B>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (A, B)>,
        A: Into<String>,
        B: Into<String>,
    {
        Self {
            pairs: pairs
                .into_iter()
                .map(|(a, b)| (a.into(), b.into()))
                .collect(),
        }
    }

    /// An identity name map over `names` (`name → name`) — the map to use when
    /// two rigs share the humanoid naming convention.
    pub fn identity<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            pairs: names
                .into_iter()
                .map(|n| {
                    let n = n.into();
                    (n.clone(), n)
                })
                .collect(),
        }
    }

    /// The standard humanoid map: identity over [`humanoid_joint_names`]. A rig
    /// that follows the convention retargets onto another such rig with no manual
    /// pairing.
    pub fn humanoid_identity() -> Self {
        Self::identity(humanoid_joint_names().iter().copied())
    }

    /// Add a `(source, target)` pair (builder style).
    pub fn with(mut self, source: impl Into<String>, target: impl Into<String>) -> Self {
        self.pairs.push((source.into(), target.into()));
        self
    }
}

/// The canonical humanoid joint names retarget v1 recognizes (hips → head, arms
/// and legs left/right). A rig using these names maps to another with
/// [`RetargetMap::humanoid_identity`].
pub fn humanoid_joint_names() -> &'static [&'static str] {
    &[
        "hips",
        "spine",
        "chest",
        "neck",
        "head",
        "shoulder_l",
        "upper_arm_l",
        "lower_arm_l",
        "hand_l",
        "shoulder_r",
        "upper_arm_r",
        "lower_arm_r",
        "hand_r",
        "upper_leg_l",
        "lower_leg_l",
        "foot_l",
        "upper_leg_r",
        "lower_leg_r",
        "foot_r",
    ]
}

/// Retarget `src_pose` (sampled on `src_skel`) onto `dst_skel` by `map`.
///
/// Every mapped joint's **local rotation** is copied bind-relatively (see the
/// module docs); the **root** joint also copies translation bind-relatively.
/// Unmapped target joints keep their bind transform. Pairs that name a joint
/// absent from either skeleton are skipped (a partial map is valid).
pub fn retarget_pose(
    src_skel: &Skeleton,
    src_pose: &Pose,
    dst_skel: &Skeleton,
    map: &RetargetMap,
) -> Pose {
    let mut out = Pose::rest(dst_skel);

    for (src_name, dst_name) in &map.pairs {
        let (Some(si), Some(di)) = (src_skel.index_of(src_name), dst_skel.index_of(dst_name))
        else {
            continue;
        };
        let (si, di) = (si as usize, di as usize);
        let Some(src_anim) = src_pose.locals.get(si) else {
            continue;
        };
        let src_joint = &src_skel.joints()[si];
        let dst_joint = &dst_skel.joints()[di];

        // dst_local_rot = dst_bind · (src_bind⁻¹ · src_anim).
        let delta = src_joint.local_bind.rotation_quat().inverse() * src_anim.rotation_quat();
        let dst_rot = (dst_joint.local_bind.rotation_quat() * delta).normalize();
        out.locals[di].rotation = dst_rot.to_array();

        // Translation is copied for the root only (bind-relative).
        if dst_joint.parent.is_none() {
            let src_t = src_anim.translation_vec();
            let src_bind_t = src_joint.local_bind.translation_vec();
            let dst_bind_t = dst_joint.local_bind.translation_vec();
            let t: Vec3 = dst_bind_t + (src_t - src_bind_t);
            out.locals[di].translation = t.to_array();
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{Joint, JointTransform};
    use glam::{Mat4, Quat, Vec3};

    /// A named 3-joint humanoid-ish chain: hips → spine → head, each +1 Y.
    fn rig(names: [&str; 3]) -> Skeleton {
        let mut joints = Vec::new();
        let mut global = Mat4::IDENTITY;
        for (i, name) in names.iter().enumerate() {
            let local = JointTransform::from_trs(
                if i == 0 { Vec3::ZERO } else { Vec3::Y },
                Quat::IDENTITY,
                Vec3::ONE,
            );
            global *= local.to_mat4();
            joints.push(Joint {
                name: (*name).into(),
                parent: if i == 0 { None } else { Some(i as u16 - 1) },
                inverse_bind: global.inverse().to_cols_array(),
                local_bind: local,
            });
        }
        Skeleton::new(joints).unwrap()
    }

    /// A pose rotating the spine and translating the root (real anim shape).
    fn animated(sk: &Skeleton) -> Pose {
        let mut p = Pose::rest(sk);
        p.locals[1].rotation = Quat::from_rotation_z(30f32.to_radians()).to_array();
        p.locals[0].translation = [0.5, 0.0, 0.0];
        p
    }

    fn quats_close(a: [f32; 4], b: [f32; 4]) -> bool {
        Quat::from_array(a)
            .normalize()
            .angle_between(Quat::from_array(b).normalize())
            < 1e-4
    }

    #[test]
    fn identity_retarget_reproduces_the_pose() {
        let sk = rig(["hips", "spine", "head"]);
        let pose = animated(&sk);
        let map = RetargetMap::identity(["hips", "spine", "head"]);
        let out = retarget_pose(&sk, &pose, &sk, &map);
        for i in 0..sk.len() {
            assert!(
                quats_close(out.locals[i].rotation, pose.locals[i].rotation),
                "joint {i} rotation drifted"
            );
        }
        // Root translation is reproduced.
        assert!(
            (Vec3::from_array(out.locals[0].translation) - Vec3::new(0.5, 0.0, 0.0)).length()
                < 1e-5
        );
    }

    #[test]
    fn renamed_skeleton_reproduces_rotations() {
        // Same rest geometry, different joint names → an explicit name map still
        // reproduces the source rotations on the target.
        let src = rig(["hips", "spine", "head"]);
        let dst = rig(["Hips", "Spine1", "Head"]);
        let pose = animated(&src);
        let map =
            RetargetMap::from_pairs([("hips", "Hips"), ("spine", "Spine1"), ("head", "Head")]);
        let out = retarget_pose(&src, &pose, &dst, &map);
        assert!(quats_close(out.locals[1].rotation, pose.locals[1].rotation));
        // Root translation carried across the rename.
        assert!(
            (Vec3::from_array(out.locals[0].translation) - Vec3::new(0.5, 0.0, 0.0)).length()
                < 1e-5
        );
    }

    #[test]
    fn unmapped_joints_stay_at_bind() {
        let src = rig(["hips", "spine", "head"]);
        let dst = rig(["hips", "spine", "head"]);
        let pose = animated(&src);
        // Map only the spine; hips + head must remain at the target bind pose.
        let map = RetargetMap::from_pairs([("spine", "spine")]);
        let out = retarget_pose(&src, &pose, &dst, &map);
        assert_eq!(out.locals[2], dst.joints()[2].local_bind, "head unmapped");
        // The mapped spine took the rotation.
        assert!(quats_close(out.locals[1].rotation, pose.locals[1].rotation));
    }

    #[test]
    fn bind_relative_rotation_composes() {
        // Target spine has a non-identity bind rotation; the retargeted rotation
        // must be dst_bind · (src_bind⁻¹ · src_anim), not a raw copy.
        let src = rig(["hips", "spine", "head"]);
        let mut dst = rig(["hips", "spine", "head"]);
        let dst_bind = Quat::from_rotation_x(20f32.to_radians());
        // Rebuild dst joint 1 with a rotated bind.
        let joints: Vec<Joint> = dst
            .joints()
            .iter()
            .enumerate()
            .map(|(i, j)| {
                let mut j = j.clone();
                if i == 1 {
                    j.local_bind.rotation = dst_bind.to_array();
                }
                j
            })
            .collect();
        dst = Skeleton::new(joints).unwrap();

        let pose = animated(&src); // spine anim = 30° about Z, src bind = identity
        let map = RetargetMap::identity(["hips", "spine", "head"]);
        let out = retarget_pose(&src, &pose, &dst, &map);
        let expected = dst_bind * Quat::from_rotation_z(30f32.to_radians());
        assert!(quats_close(out.locals[1].rotation, expected.to_array()));
    }
}
