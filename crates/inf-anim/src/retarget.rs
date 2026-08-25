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

    /// **The canonical vocabulary onto the mannequin's** — the pairing that makes
    /// a 20-joint clip drive a 161-bone rig (SK1a).
    ///
    /// Without it `humanoid_identity` is a *silent no-op* on a Manny rig: the two
    /// vocabularies overlap at exactly five names (`head`, `hand_l`, `hand_r`,
    /// `foot_l`, `foot_r`), so fourteen of nineteen pairs name a target that is
    /// not there,
    /// [`retarget_pose`] skips them without a word, and the result is a rig
    /// standing in its bind pose that looks like a retarget that "did nothing
    /// visible".
    ///
    /// The two structural choices, stated rather than buried: a canonical `spine`
    /// pairs with **`spine_01`** and a canonical `chest` with **`spine_05`** — the
    /// bottom and the top of the mannequin's five-segment chain, so a torso twist
    /// arrives at both ends of the back rather than being spent on one vertebra;
    /// the three spine segments between them keep their bind, which is exactly
    /// what "this source rig has no opinion about them" means.
    pub fn canonical_to_manny() -> Self {
        Self::from_pairs(CANONICAL_MANNY_PAIRS.iter().map(|(c, m)| (*c, *m)))
    }

    /// The reverse: a mannequin-authored clip onto a canonical-vocabulary rig.
    pub fn manny_to_canonical() -> Self {
        Self::from_pairs(CANONICAL_MANNY_PAIRS.iter().map(|(c, m)| (*m, *c)))
    }
}

/// `(canonical, mannequin)` — the pairing both mannequin maps are built from, so
/// the two directions cannot drift apart.
static CANONICAL_MANNY_PAIRS: [(&str, &str); 19] = [
    ("hips", "pelvis"),
    ("spine", "spine_01"),
    ("chest", "spine_05"),
    ("neck", "neck_01"),
    ("head", "head"),
    ("shoulder_l", "clavicle_l"),
    ("upper_arm_l", "upperarm_l"),
    ("lower_arm_l", "lowerarm_l"),
    ("hand_l", "hand_l"),
    ("shoulder_r", "clavicle_r"),
    ("upper_arm_r", "upperarm_r"),
    ("lower_arm_r", "lowerarm_r"),
    ("hand_r", "hand_r"),
    ("upper_leg_l", "thigh_l"),
    ("lower_leg_l", "calf_l"),
    ("foot_l", "foot_l"),
    ("upper_leg_r", "thigh_r"),
    ("lower_leg_r", "calf_r"),
    ("foot_r", "foot_r"),
];

/// **What a retarget actually moved** (SK1a).
///
/// Retarget v1 skips a pair naming a joint either skeleton lacks, and says
/// nothing. That silence is the defect this type exists to end: a map whose every
/// pair misses produces a perfectly valid bind pose, and the only way to tell it
/// from a correct retarget of a still character is to be told.
///
/// Every list is sorted and deduplicated, so the report is a property of the
/// `(source, target, map)` triple and not of the order the pairs were written in.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RetargetReport {
    /// Target joints that were written, by name.
    pub copied: Vec<String>,
    /// Target joints that kept their bind because no pair named them.
    pub unmapped_target: Vec<String>,
    /// Pairs whose **source** name is not in the source skeleton.
    pub missing_source: Vec<String>,
    /// Pairs whose **target** name is not in the target skeleton.
    pub missing_target: Vec<String>,
}

impl RetargetReport {
    /// How many target joints the retarget wrote.
    pub fn copied_count(&self) -> usize {
        self.copied.len()
    }

    /// Whether the retarget moved **nothing** — the silent failure, named.
    pub fn is_vacuous(&self) -> bool {
        self.copied.is_empty()
    }

    /// A one-line summary for a log or a wizard warning.
    pub fn summary(&self) -> String {
        format!(
            "retargeted {} joints; {} kept bind, {} pairs missed the source rig, {} missed the target",
            self.copied.len(),
            self.unmapped_target.len(),
            self.missing_source.len(),
            self.missing_target.len()
        )
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
    retarget_inner(src_skel, src_pose, dst_skel, map, false).0
}

/// [`retarget_pose`], **and what it moved** (SK1a) — see [`RetargetReport`] for
/// why the silence this replaces was worth a type.
///
/// The pose is identical to `retarget_pose`'s, bit for bit: the report is
/// accumulated alongside and costs a `Vec<String>` per category — plus one
/// `String` clone per target joint — that a caller who does not want it never
/// pays for, because `retarget_pose` goes through the same body with the
/// accumulation switched **off**. (SK1a audit: this sentence was true of the
/// intent and false of the code, which delegated here and paid 161 clones and
/// four sorts on a mannequin target.)
pub fn retarget_pose_reported(
    src_skel: &Skeleton,
    src_pose: &Pose,
    dst_skel: &Skeleton,
    map: &RetargetMap,
) -> (Pose, RetargetReport) {
    retarget_inner(src_skel, src_pose, dst_skel, map, true)
}

/// The one body both doors run, with the report accumulation as a flag.
///
/// One body rather than two, for the reason the mannequin pairing is one table:
/// a second copy of the copy rule is a second answer waiting to disagree.
fn retarget_inner(
    src_skel: &Skeleton,
    src_pose: &Pose,
    dst_skel: &Skeleton,
    map: &RetargetMap,
    want_report: bool,
) -> (Pose, RetargetReport) {
    let mut out = Pose::rest(dst_skel);
    let mut report = RetargetReport::default();
    // Only the report reads this, so a caller that does not want one does not
    // allocate it either.
    let mut written = if want_report {
        vec![false; dst_skel.len()]
    } else {
        Vec::new()
    };

    for (src_name, dst_name) in &map.pairs {
        let (src_at, dst_at) = (src_skel.index_of(src_name), dst_skel.index_of(dst_name));
        if want_report {
            if src_at.is_none() {
                report.missing_source.push(src_name.clone());
            }
            if dst_at.is_none() {
                report.missing_target.push(dst_name.clone());
            }
        }
        let (Some(si), Some(di)) = (src_at, dst_at) else {
            continue;
        };
        let (si, di) = (si as usize, di as usize);
        let Some(src_anim) = src_pose.locals.get(si) else {
            continue;
        };
        if let Some(w) = written.get_mut(di) {
            *w = true;
        }
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

    if want_report {
        for (i, j) in dst_skel.joints().iter().enumerate() {
            if written.get(i).copied().unwrap_or(false) {
                report.copied.push(j.name.clone());
            } else {
                report.unmapped_target.push(j.name.clone());
            }
        }
        // Sorted and deduplicated so the report is a property of the triple and
        // not of the order the pairs happen to be written in — the same reason
        // `socket_transforms` sorts.
        for list in [
            &mut report.copied,
            &mut report.unmapped_target,
            &mut report.missing_source,
            &mut report.missing_target,
        ] {
            list.sort();
            list.dedup();
        }
    }
    (out, report)
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

    /// **The silent no-op, measured and then closed** (SK1a).
    ///
    /// The canonical vocabulary and the mannequin's overlap at exactly five
    /// names, so `humanoid_identity` on a mannequin target writes five joints out
    /// of 161 and says nothing about the other fourteen pairs. The pairing writes
    /// nineteen. Both halves are asserted, because the second number only means
    /// something beside the first.
    #[test]
    fn the_canonical_map_is_nearly_a_no_op_on_a_mannequin_and_the_pairing_is_not() {
        use crate::template::{build_template, BodyParams, BodyPlan};
        let src = build_template(BodyPlan::BipedCanonical, &BodyParams::default()).unwrap();
        let dst = build_template(BodyPlan::Biped, &BodyParams::default()).unwrap();
        let mut pose = Pose::rest(&src.skeleton);
        for name in ["upper_arm_l", "lower_leg_r", "chest", "hips"] {
            let i = src.skeleton.index_of(name).unwrap() as usize;
            pose.locals[i].rotation = Quat::from_rotation_x(0.5).to_array();
        }

        let (_, blind) = retarget_pose_reported(
            &src.skeleton,
            &pose,
            &dst.skeleton,
            &RetargetMap::humanoid_identity(),
        );
        assert_eq!(
            blind.copied,
            ["foot_l", "foot_r", "hand_l", "hand_r", "head"],
            "the identity map's whole overlap with the mannequin"
        );
        assert_eq!(blind.missing_target.len(), 14, "{:?}", blind.missing_target);
        assert!(blind.missing_target.contains(&"upper_leg_l".to_string()));
        assert!(
            !blind.is_vacuous(),
            "five is not zero, and that is the trap"
        );

        let (out, report) = retarget_pose_reported(
            &src.skeleton,
            &pose,
            &dst.skeleton,
            &RetargetMap::canonical_to_manny(),
        );
        assert_eq!(report.copied_count(), 19, "{:?}", report.copied);
        assert!(
            report.missing_target.is_empty(),
            "{:?}",
            report.missing_target
        );
        assert!(
            report.missing_source.is_empty(),
            "{:?}",
            report.missing_source
        );
        assert_eq!(
            report.unmapped_target.len(),
            dst.skeleton.len() - 19,
            "everything else kept bind, and the report says so"
        );
        // …and the bones really moved: the source's bent elbow arrives on
        // `upperarm_l`, its bent knee on `calf_r`, its chest on `spine_05`.
        for name in ["upperarm_l", "calf_r", "spine_05", "pelvis"] {
            let i = dst.skeleton.index_of(name).unwrap() as usize;
            let q = Quat::from_array(out.locals[i].rotation);
            assert!(
                (1.0 - q.dot(Quat::from_rotation_x(0.5)).abs()) < 1.0e-6,
                "`{name}` did not receive the rotation: {q:?}"
            );
        }
        // A joint with nothing paired onto it is at bind, not at zero.
        let twist = dst.skeleton.index_of("upperarm_twist_01_l").unwrap() as usize;
        assert_eq!(
            out.locals[twist],
            dst.skeleton.joints()[twist].local_bind,
            "an unmapped joint must keep its bind"
        );
        assert!(
            report.summary().contains("retargeted 19 joints"),
            "{}",
            report.summary()
        );
    }

    /// **The reported door and the cheap door write the same pose, bit for
    /// bit** (SK1a audit).
    ///
    /// They used to be the same function, which made this true and made the
    /// cheap door pay for a report it threw away. Now they are one body and a
    /// flag, and this is what stops the flag becoming two behaviours.
    #[test]
    fn the_two_retarget_doors_write_the_same_pose() {
        let src = crate::template::build_template(
            crate::template::BodyPlan::BipedCanonical,
            &crate::template::BodyParams::default(),
        )
        .unwrap();
        let dst = crate::template::build_template(
            crate::template::BodyPlan::Biped,
            &crate::template::BodyParams::default(),
        )
        .unwrap();
        let mut pose = Pose::rest(&src.skeleton);
        for (i, local) in pose.locals.iter_mut().enumerate() {
            local.rotation = Quat::from_xyzw(0.0, 0.0, (i as f32 * 0.07).sin(), 1.0)
                .normalize()
                .to_array();
        }
        let map = RetargetMap::canonical_to_manny();
        let cheap = retarget_pose(&src.skeleton, &pose, &dst.skeleton, &map);
        let (reported, report) = retarget_pose_reported(&src.skeleton, &pose, &dst.skeleton, &map);
        assert_eq!(cheap.locals.len(), reported.locals.len());
        for (i, (a, b)) in cheap.locals.iter().zip(reported.locals.iter()).enumerate() {
            for (u, v) in a
                .translation
                .iter()
                .chain(a.rotation.iter())
                .chain(a.scale.iter())
                .zip(
                    b.translation
                        .iter()
                        .chain(b.rotation.iter())
                        .chain(b.scale.iter()),
                )
            {
                assert_eq!(u.to_bits(), v.to_bits(), "joint {i} differs between doors");
            }
        }
        // …and only the reported door built a report.
        assert_eq!(report.copied.len(), 19);
    }

    /// The two directions are one table, so they cannot drift.
    #[test]
    fn the_mannequin_pairing_reverses_exactly() {
        let fwd = RetargetMap::canonical_to_manny();
        let back = RetargetMap::manny_to_canonical();
        assert_eq!(fwd.pairs.len(), humanoid_joint_names().len());
        assert_eq!(back.pairs.len(), fwd.pairs.len());
        for ((a, b), (c, d)) in fwd.pairs.iter().zip(back.pairs.iter()) {
            assert_eq!((a, b), (d, c));
        }
        // Every canonical name is spoken for — that is what makes the map a map
        // of the vocabulary rather than of whichever joints somebody remembered.
        let mut left: Vec<&str> = fwd.pairs.iter().map(|(a, _)| a.as_str()).collect();
        left.sort_unstable();
        let mut want: Vec<&str> = humanoid_joint_names().to_vec();
        want.sort_unstable();
        assert_eq!(left, want);
    }

    /// A **vacuous** retarget is named as one — the report's whole reason to exist.
    #[test]
    fn a_map_that_hits_nothing_reports_itself_as_vacuous() {
        let src = rig(["hips", "spine", "head"]);
        let dst = rig(["a", "b", "c"]);
        let (_, r) = retarget_pose_reported(
            &src,
            &Pose::rest(&src),
            &dst,
            &RetargetMap::humanoid_identity(),
        );
        assert!(r.is_vacuous());
        assert_eq!(r.copied_count(), 0);
        assert_eq!(r.unmapped_target, ["a", "b", "c"]);
        assert_eq!(r.missing_target.len(), 19);
        assert_eq!(r.missing_source.len(), 16, "hips/spine/head are there");
    }
}
