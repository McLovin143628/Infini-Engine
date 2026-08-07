//! **Modular rigging** (P24.3): assembling one skeleton out of parts.
//!
//! The dcc-vision's headline flow — a torso, two arms and a head dropped
//! together in the Model Editor, each part skinned, the assembly **one**
//! skeleton — needs exactly two rules, and both of them are about *not moving
//! what is already there*.
//!
//! # Rule 1: joints are APPEND-ONLY
//!
//! [`merge_skeletons`] leaves every base joint at the index it had and puts the
//! incoming skeleton's joints after them, at `base.len() + i`. That is not a
//! convenience; it is what makes the merge safe for everything that already
//! *names* a joint by index:
//!
//! * a **weight table** on the torso mesh (`VertWeights::joints` is
//!   `[u16; 4]` — no names, so a shifted index silently re-weights a character to
//!   a different bone);
//! * an **IK chain** (`IkGoalRecord::chain` is `Vec<u16>`) — a leg chain
//!   authored on the torso survives an arm being attached **by construction**,
//!   not by fix-up;
//! * a **socket** (`Socket::joint`);
//! * a **limit** ([`JointLimit::joint`]).
//!
//! The incoming side's indices *do* move — by a constant `joint_offset`, which is
//! returned so the caller can remap the one weight table that needs it. Every
//! remap in the engine is that single addition.
//!
//! # Rule 2: a name collision is a REFUSAL, not a rename
//!
//! Two parts that each author a `hand_r` socket are two parts that disagree about
//! what `hand_r` is, and silently renaming one to `hand_r_001` produces an
//! assembly whose attach points no author can predict — the sword goes on
//! whichever arm happened to be dropped first. So the merge refuses, **as a
//! value** ([`SkeletonMergeError`]), naming the socket. The same is true of joint
//! names: they are not unique-by-construction in a skeleton, but a *duplicate*
//! after a merge makes `mirror_joint_map` and every by-name lookup ambiguous, so
//! it is reported too.
//!
//! # Attachment: merging AT a socket
//!
//! `attach` is the base joint the incoming skeleton's roots become children of.
//! Dropping an arm onto the torso's `hand_r` socket is
//! `merge_skeletons(torso, arm, torso.socket("hand_r").joint)` — the arm binds
//! *relative to that joint*, so it rides the hand for the rest of the character's
//! life. A part merged at the base root is a part that simply joins the rig.
//!
//! # Why this is in `inf-anim` and not in the kernel
//!
//! `inf_dcc` holds a `SkinBinding`, which is a `.inf_skel` GUID and a **joint
//! count** — it deliberately holds no joint list, so it cannot pair
//! `upper_arm_l` with `upper_arm_r` or union a socket table. The names live here,
//! and `inf-dcc` already depends on this crate (`autofit.rs`), so this is where
//! the rule goes and the kernel keeps its "a mesh knows how many joints, not
//! which" property.

use crate::asset::SkeletonAsset;
use crate::skeleton::{Joint, Skeleton, SkeletonError};
use crate::template::JointLimit;

/// Why a merge refused.
///
/// Every variant is a **value**: a merge that cannot be done leaves both inputs
/// untouched and says what stopped it, in the `inf_dcc::OpError` tradition.
/// `Clone` is deliberately absent: [`SkeletonError`] is not `Clone` — it is a
/// terminal refusal, not a value anyone copies — and deriving it here would
/// force a derive onto that type for no caller's benefit. `FitError` one crate
/// over declines `Clone` for the same reason rather than widening someone else's
/// type to satisfy its own.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum SkeletonMergeError {
    /// The attach joint is not in the base skeleton.
    #[error("attach joint {joint} is not in the base skeleton ({joints} joints)")]
    NoSuchAttachJoint { joint: u16, joints: usize },
    /// Both parts author a socket with the same name.
    ///
    /// Refused rather than renamed: see the module docs.
    #[error(
        "both parts author a socket named {name:?} — rename one before merging \
         (a silent rename would put the attachment on whichever part was dropped first)"
    )]
    SocketNameCollision { name: String },
    /// Both parts author a joint with the same name.
    ///
    /// Not fatal to the *math* — indices are what everything uses — but it makes
    /// every by-name lookup in the engine ambiguous, including the l/r pairing
    /// [`mirror_joint_map`] does, so it is refused at the door rather than
    /// discovered later as a mirror that swapped the wrong bone.
    #[error("both parts author a joint named {name:?} — rename one before merging")]
    JointNameCollision { name: String },
    /// The merged joint list is not a valid skeleton.
    ///
    /// Reachable in exactly one way — the combined count overflowing `u16` — but
    /// carried through rather than asserted, because [`Skeleton::new`] is the
    /// authority on what a skeleton is and this module is not.
    #[error("the merged skeleton is invalid: {0}")]
    Invalid(#[from] SkeletonError),
}

/// The result of [`merge_skeletons`].
#[derive(Debug, Clone, PartialEq)]
pub struct SkeletonMerge {
    /// The assembled rig: base joints at their original indices, incoming joints
    /// appended, sockets unioned, limits carried from both sides.
    pub asset: SkeletonAsset,
    /// What to add to an **incoming** joint index to get its index in
    /// [`SkeletonMerge::asset`]. Equal to the base skeleton's joint count.
    ///
    /// This is the whole of "remap the merged part's weight table": every
    /// non-zero influence's joint index gets this added, and nothing else moves.
    pub joint_offset: u16,
}

/// **Append `incoming` onto `base`**, parenting its roots to `attach`.
///
/// See the module docs for the two rules. In summary:
///
/// * base joints, sockets and limits keep their indices **exactly**;
/// * incoming joints land at `base.len() + i`, with their parents shifted by the
///   same amount and their *roots* reparented to `attach`;
/// * incoming sockets and limits are shifted by the same offset;
/// * a socket- or joint-name collision refuses.
///
/// The result is topologically ordered by construction: `attach` is a base index
/// and therefore below every appended index, and the incoming skeleton was
/// already ordered, so shifting it by a constant preserves "parents precede
/// children". [`Skeleton::new`] re-checks that rather than trusting it.
pub fn merge_skeletons(
    base: &SkeletonAsset,
    incoming: &SkeletonAsset,
    attach: u16,
) -> Result<SkeletonMerge, SkeletonMergeError> {
    let base_len = base.skeleton.len();
    if attach as usize >= base_len {
        return Err(SkeletonMergeError::NoSuchAttachJoint {
            joint: attach,
            joints: base_len,
        });
    }
    // Collisions are checked BEFORE anything is built, so a refusal costs no
    // allocation and — more to the point — cannot half-apply.
    for s in &incoming.sockets {
        if base.sockets.iter().any(|b| b.name == s.name) {
            return Err(SkeletonMergeError::SocketNameCollision {
                name: s.name.clone(),
            });
        }
    }
    for i in 0..incoming.skeleton.len() {
        let name = &incoming.skeleton.joint(i).expect("in range").name;
        if (0..base_len).any(|b| &base.skeleton.joint(b).expect("in range").name == name) {
            return Err(SkeletonMergeError::JointNameCollision { name: name.clone() });
        }
    }

    let offset = u16::try_from(base_len)
        .map_err(|_| SkeletonMergeError::Invalid(SkeletonError::TooManyJoints(base_len)))?;

    let mut joints: Vec<Joint> = (0..base_len)
        .map(|i| base.skeleton.joint(i).expect("in range").clone())
        .collect();
    for i in 0..incoming.skeleton.len() {
        let j = incoming.skeleton.joint(i).expect("in range");
        joints.push(Joint {
            name: j.name.clone(),
            // A ROOT of the incoming part becomes a child of `attach`; everything
            // else keeps its own parent, shifted. This is the whole of
            // "attachment": the part rides the joint it was dropped on.
            parent: Some(match j.parent {
                Some(p) => p + offset,
                None => attach,
            }),
            inverse_bind: j.inverse_bind,
            local_bind: j.local_bind,
        });
    }

    let mut sockets = base.sockets.clone();
    sockets.extend(incoming.sockets.iter().map(|s| {
        let mut s = s.clone();
        s.joint += offset;
        s
    }));
    let mut limits = base.limits.clone();
    limits.extend(incoming.limits.iter().map(|l| JointLimit {
        joint: l.joint + offset,
        ..*l
    }));

    let skeleton = Skeleton::new(joints)?;
    let mut asset = SkeletonAsset::with_sockets(skeleton, sockets);
    asset.limits = limits;
    Ok(SkeletonMerge {
        asset,
        joint_offset: offset,
    })
}

/// The canonical **left/right suffix** of a joint name, if it has one.
///
/// The convention `build_template` emits is a `_l` / `_r` **suffix** —
/// `upper_arm_l`, `hand_r`, and for a multi-girdle plan `upper_leg_l0` /
/// `upper_leg_r0` (see `leg_suffix`). So the pairing key is "the name with the
/// side letter flipped", which is what this returns: `Some(twin_name)` for a
/// sided joint, `None` for a spine, a head or a root.
///
/// Written against what the generator actually produces rather than against the
/// `l_`/`r_` *prefix* convention other engines use — a mirror keyed on the wrong
/// convention pairs nothing and silently does the old thing.
pub fn mirrored_joint_name(name: &str) -> Option<String> {
    // `upper_arm_l`, or `upper_leg_r0` — the side letter is the character right
    // after the last `_`, and anything after it is the girdle ordinal.
    let cut = name.rfind('_')?;
    let tail = &name[cut + 1..];
    let mut chars = tail.chars();
    let side = chars.next()?;
    let rest = chars.as_str();
    // The ordinal must be digits (or empty); `_left` and `_root` are not sides.
    if !rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let flipped = match side {
        'l' => 'r',
        'r' => 'l',
        _ => return None,
    };
    Some(format!("{}_{flipped}{rest}", &name[..cut]))
}

/// **Which joint each joint mirrors onto**, by canonical name.
///
/// `map[i]` is the index of joint `i`'s left/right twin, or `i` itself when the
/// joint has no twin (a spine, a head, a root, or a sided joint whose partner is
/// missing). Applying it to a weight table is what turns a mirrored *left* arm
/// into a right arm that is actually weighted to the right arm's bones.
///
/// A **sided joint with no partner** maps to itself rather than refusing here:
/// this function is a lookup table, and the caller is the one that knows whether
/// an unmatched joint matters. [`unmatched_sided_joints`] is the door for asking.
pub fn mirror_joint_map(skeleton: &Skeleton) -> Vec<u16> {
    let n = skeleton.len();
    let mut map: Vec<u16> = (0..n as u16).collect();
    for (i, slot) in map.iter_mut().enumerate() {
        let name = &skeleton.joints()[i].name;
        let Some(twin) = mirrored_joint_name(name) else {
            continue;
        };
        if let Some(j) = skeleton.index_of(&twin) {
            *slot = j;
        }
    }
    map
}

/// Every joint whose name says it has a side and whose twin is **absent** — the
/// refusal-as-value half of [`mirror_joint_map`].
///
/// A caller mirroring geometry across a rig with a `hand_l` and no `hand_r` is
/// about to produce a mesh weighted to the wrong arm and look correct doing it,
/// so the list is returned and the decision is the caller's. Names, not indices,
/// because the answer is going in front of an author.
pub fn unmatched_sided_joints(skeleton: &Skeleton) -> Vec<String> {
    let map = mirror_joint_map(skeleton);
    (0..skeleton.len())
        .filter(|&i| {
            let name = &skeleton.joint(i).expect("in range").name;
            mirrored_joint_name(name).is_some() && map[i] as usize == i
        })
        .map(|i| skeleton.joint(i).expect("in range").name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::JointTransform;
    use crate::template::{build_template, BodyParams, BodyPlan};
    use crate::Socket;
    use glam::{Mat4, Vec3};

    fn joint(name: &str, parent: Option<u16>, at: Vec3) -> Joint {
        Joint {
            name: name.into(),
            parent,
            inverse_bind: Mat4::from_translation(-at).to_cols_array(),
            local_bind: JointTransform::from_trs(at, glam::Quat::IDENTITY, Vec3::ONE),
        }
    }

    /// A three-joint "torso": an **off-origin** root and **non-uniform** bones.
    ///
    /// Deliberately not unit-length-at-the-origin (F4): a symmetric chain makes
    /// FABRIK's answer independent of which bones it used, so the old fixture's
    /// `reach_error` comparison was bit-identical across configurations and held
    /// whichever bones the chain resolved to.
    ///
    /// **Corrected by the P24.3 re-audit**: the report said that fixture "could
    /// not fail", and it did not reproduce — checked out at `af1a301` the old
    /// suite already failed two tests. The narrower, true claim is that the
    /// *equality assertion* proved nothing without these proportions and the
    /// two-chains-differ control below.
    fn torso() -> SkeletonAsset {
        let sk = Skeleton::new(vec![
            joint("root", None, Vec3::new(0.13, 0.87, -0.05)),
            joint("chest", Some(0), Vec3::new(0.0, 0.42, 0.0)),
            joint("neck", Some(1), Vec3::new(0.02, 0.17, 0.01)),
        ])
        .unwrap();
        SkeletonAsset::with_sockets(sk, vec![Socket::new("spine_top", 1)])
    }

    /// A two-joint "arm", asymmetric and a different length from the torso's
    /// bones, so a chain through it is distinguishable from one through them.
    fn arm() -> SkeletonAsset {
        let sk = Skeleton::new(vec![
            joint("upper_arm_r", None, Vec3::new(0.24, -0.03, 0.0)),
            joint("hand_r", Some(0), Vec3::new(0.31, -0.09, 0.02)),
        ])
        .unwrap();
        let mut a = SkeletonAsset::with_sockets(sk, vec![Socket::new("grip_r", 1)]);
        a.limits = vec![JointLimit::hinge_x(1, 0.0, 150.0)];
        a
    }

    /// **The headline gate**: merging appends, and every base index is where it
    /// was — which is what makes an IK chain authored on the torso survive.
    #[test]
    fn a_merge_appends_and_never_moves_a_base_joint() {
        let (t, a) = (torso(), arm());
        let m = merge_skeletons(&t, &a, 1).expect("merges");
        assert_eq!(m.joint_offset, 3);
        assert_eq!(m.asset.skeleton.len(), 5);

        // Base joints: same index, same name, same bind — byte-for-byte.
        for i in 0..t.skeleton.len() {
            assert_eq!(
                m.asset.skeleton.joint(i).unwrap(),
                t.skeleton.joint(i).unwrap(),
                "base joint {i} moved"
            );
        }
        // The arm's root became a child of the attach joint; its own child kept
        // its parent, shifted.
        assert_eq!(m.asset.skeleton.joint(3).unwrap().name, "upper_arm_r");
        assert_eq!(m.asset.skeleton.joint(3).unwrap().parent, Some(1));
        assert_eq!(m.asset.skeleton.joint(4).unwrap().parent, Some(3));
        // Sockets unioned, the incoming one shifted.
        assert_eq!(m.asset.sockets.len(), 2);
        assert_eq!(m.asset.sockets[0], t.sockets[0]);
        assert_eq!(m.asset.sockets[1].name, "grip_r");
        assert_eq!(m.asset.sockets[1].joint, 4);
        // …and the limits came with it.
        assert_eq!(m.asset.limits.len(), 1);
        assert_eq!(m.asset.limits[0].joint, 4);
    }

    /// **An IK chain on the torso survives an arm being attached** — the claim
    /// the append-only rule exists to make, asserted on a chain rather than on
    /// the joint list.
    #[test]
    fn an_ik_chain_on_the_base_survives_the_merge() {
        let (t, a) = (torso(), arm());
        let chain: Vec<u16> = vec![0, 1];
        let target = Vec3::new(0.31, 0.55, -0.12);
        let mut pose = crate::Pose::rest(&t.skeleton);
        let before = crate::solve_chain(&t.skeleton, &mut pose, &chain, target, None, &t.limits)
            .expect("solves before");

        let m = merge_skeletons(&t, &a, 1).expect("merges");
        let mut pose = crate::Pose::rest(&m.asset.skeleton);
        let after = crate::solve_chain(
            &m.asset.skeleton,
            &mut pose,
            &chain,
            target,
            None,
            &m.asset.limits,
        )
        .expect("the SAME chain still solves after the merge");
        assert_eq!(
            before.reach_error.to_bits(),
            after.reach_error.to_bits(),
            "the chain resolved to different bones after a merge"
        );

        // **The control that makes the equality mean something** (F4). The old
        // fixture was unit-length bones at the origin, so its reach error was
        // bit-identical across configurations — the assertion held even if the
        // chain had resolved to entirely different bones. (The report's stronger
        // "the fixture could not fail" did not reproduce; see `torso`.)
        // Solving the merged rig's OTHER chain on the same target must give a
        // measurably different answer; if it did not, the rig's bones would be
        // interchangeable and the equality above would prove nothing.
        let mut other = crate::Pose::rest(&m.asset.skeleton);
        let arm_chain: Vec<u16> = vec![m.joint_offset, m.joint_offset + 1];
        let arm_solve = crate::solve_chain(
            &m.asset.skeleton,
            &mut other,
            &arm_chain,
            target,
            None,
            &m.asset.limits,
        )
        .expect("the arm chain solves");
        assert_ne!(
            after.reach_error.to_bits(),
            arm_solve.reach_error.to_bits(),
            "two different chains give a bit-identical reach error — the fixture is degenerate and the equality above proves nothing"
        );
        assert_ne!(
            before.chain_length.to_bits(),
            arm_solve.chain_length.to_bits(),
            "the two chains are the same length, so they are not distinguishable"
        );
    }

    /// **F5: append-only is only testable at an INTERIOR attach joint.**
    ///
    /// Every earlier merge test attached at `base_len - 1`, where "append" and
    /// "insert directly after the attach joint" produce the *same* layout — so
    /// the distinction the whole rule is about was never exercised. Here the part
    /// attaches at joint 0 of a three-joint torso, so an insert-after-attach
    /// implementation would put it at indices 1..2 and shift `chest` and `neck`
    /// up by two. Every base index staying put is now a claim with content.
    #[test]
    fn attaching_at_an_interior_joint_still_appends() {
        let (t, a) = (torso(), arm());
        let base_len = t.skeleton.len();
        assert!(base_len >= 3, "the fixture must have an interior joint");
        let m = merge_skeletons(&t, &a, 0).expect("merges");
        assert_eq!(m.joint_offset as usize, base_len);

        for i in 0..base_len {
            assert_eq!(
                m.asset.skeleton.joint(i).unwrap(),
                t.skeleton.joint(i).unwrap(),
                "base joint {i} moved; the merge inserted rather than appended"
            );
        }
        assert_eq!(m.asset.skeleton.joint(1).unwrap().name, "chest");
        assert_eq!(m.asset.skeleton.joint(2).unwrap().name, "neck");
        assert_eq!(
            m.asset.skeleton.joint(base_len).unwrap().name,
            "upper_arm_r"
        );
        assert_eq!(m.asset.skeleton.joint(base_len).unwrap().parent, Some(0));
        let sock = m
            .asset
            .sockets
            .iter()
            .find(|s| s.name == "spine_top")
            .unwrap();
        assert_eq!(sock.joint, 1, "a base socket followed a shift");
    }

    /// A colliding socket name refuses, by value, naming the socket — and
    /// neither input is touched.
    #[test]
    fn a_socket_name_collision_is_a_refusal() {
        let t = torso();
        let mut a = arm();
        a.sockets[0].name = "spine_top".into();
        assert_eq!(
            merge_skeletons(&t, &a, 1),
            Err(SkeletonMergeError::SocketNameCollision {
                name: "spine_top".into()
            })
        );
        let msg = merge_skeletons(&t, &a, 1).unwrap_err().to_string();
        assert!(msg.contains("spine_top") && msg.contains("rename"), "{msg}");
    }

    /// …and so does a colliding joint name, for the reason the error records.
    #[test]
    fn a_joint_name_collision_is_a_refusal() {
        let t = torso();
        let mut a = arm();
        let sk = Skeleton::new(vec![
            joint("chest", None, Vec3::ZERO),
            joint("hand_r", Some(0), Vec3::X),
        ])
        .unwrap();
        a.skeleton = sk;
        assert_eq!(
            merge_skeletons(&t, &a, 1),
            Err(SkeletonMergeError::JointNameCollision {
                name: "chest".into()
            })
        );
    }

    /// An attach joint the base does not have refuses rather than parenting to
    /// nothing.
    #[test]
    fn an_out_of_range_attach_joint_refuses() {
        assert_eq!(
            merge_skeletons(&torso(), &arm(), 9),
            Err(SkeletonMergeError::NoSuchAttachJoint {
                joint: 9,
                joints: 3
            })
        );
    }

    /// **Merging AT a socket** is the modular flow: the part lands under the
    /// joint the socket rides, so it moves with it.
    #[test]
    fn merging_at_a_socket_parents_the_part_to_that_sockets_joint() {
        let t = torso();
        let socket_joint = t
            .sockets
            .iter()
            .find(|s| s.name == "spine_top")
            .expect("the torso authors it")
            .joint;
        let m = merge_skeletons(&t, &arm(), socket_joint).expect("merges");
        assert_eq!(
            m.asset
                .skeleton
                .joint(m.joint_offset as usize)
                .unwrap()
                .parent,
            Some(socket_joint),
            "the dropped part must ride the socket's joint"
        );
    }

    /// The l/r pairing is keyed on the suffix the generator actually emits.
    #[test]
    fn the_mirror_name_map_follows_the_generators_convention() {
        assert_eq!(
            mirrored_joint_name("upper_arm_l").as_deref(),
            Some("upper_arm_r")
        );
        assert_eq!(mirrored_joint_name("hand_r").as_deref(), Some("hand_l"));
        // Multi-girdle legs carry an ordinal after the side letter.
        assert_eq!(
            mirrored_joint_name("upper_leg_l0").as_deref(),
            Some("upper_leg_r0")
        );
        assert_eq!(
            mirrored_joint_name("upper_leg_r12").as_deref(),
            Some("upper_leg_l12")
        );
        // Non-sided joints, and words that merely start with the letter.
        for n in ["root", "spine_0", "head", "neck_1", "girdle_0"] {
            assert_eq!(mirrored_joint_name(n), None, "{n} is not a sided joint");
        }
        // `l_upper_arm` is the OTHER convention; this repo does not use it, and
        // treating it as a side would pair names nothing emits.
        assert_eq!(mirrored_joint_name("l_upper_arm"), None);
    }

    /// On a real template the map pairs every arm and leg joint and leaves the
    /// spine alone — the property a weight swap rests on.
    #[test]
    fn the_mirror_map_pairs_a_real_biped() {
        let sk = build_template(BodyPlan::Biped, &BodyParams::default()).unwrap();
        let map = mirror_joint_map(&sk.skeleton);
        assert!(unmatched_sided_joints(&sk.skeleton).is_empty());

        let index = |n: &str| {
            (0..sk.skeleton.len())
                .find(|&i| sk.skeleton.joint(i).unwrap().name == n)
                .unwrap_or_else(|| panic!("{n} missing")) as u16
        };
        for (a, b) in [
            ("upper_arm_l", "upper_arm_r"),
            ("hand_l", "hand_r"),
            ("upper_leg_l", "upper_leg_r"),
            ("foot_l", "foot_r"),
        ] {
            let (ia, ib) = (index(a), index(b));
            assert_eq!(map[ia as usize], ib, "{a} did not pair with {b}");
            assert_eq!(map[ib as usize], ia, "the pairing is not symmetric");
        }
        // A spine joint maps to itself — mirroring it must be a no-op. The
        // biped root is named "hips", not "root" (template.rs).
        let root = index("hips");
        assert_eq!(map[root as usize], root);
        // …and the map is a PERMUTATION, which is what makes applying it twice
        // the identity (a swap that lost a joint would fail here).
        let mut seen = map.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            sk.skeleton.len(),
            "the mirror map is not a bijection"
        );
        for i in 0..sk.skeleton.len() {
            assert_eq!(
                map[map[i] as usize] as usize, i,
                "mirroring twice is not the identity"
            );
        }
    }

    /// A rig with one side of a pair missing REPORTS it rather than silently
    /// mapping the joint to itself — the anti-vacuity arm for
    /// `unmatched_sided_joints`.
    #[test]
    fn a_half_sided_rig_reports_its_unmatched_joints() {
        let sk = Skeleton::new(vec![
            joint("root", None, Vec3::ZERO),
            joint("upper_arm_l", Some(0), Vec3::X),
        ])
        .unwrap();
        assert_eq!(unmatched_sided_joints(&sk), vec!["upper_arm_l".to_string()]);
        // …and it maps to itself, so a caller that ignores the report gets the
        // old behaviour rather than an out-of-range index.
        assert_eq!(mirror_joint_map(&sk), vec![0, 1]);
    }
}
