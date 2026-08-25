//! **Hands** (SK1b): the arm that reaches and the fingers that close on what it
//! reached for.
//!
//! SK1a shipped [`GripAffordance`] with an empty table on every rig and said so:
//! *"`grips` is empty. The type ships; SK1b's finger solver is its first
//! consumer, and `JointLimit::cone` is there for it."* This is that solver.
//!
//! # Three things, and they are separable on purpose
//!
//! * [`arm_chain`] — which three joints an arm IK solve runs over, asked of the
//!   role table first and of a name rule only for a rig that carries none. It is
//!   [`crate::ik::solve_chain`] that does the solving; nothing here re-implements
//!   two-bone IK.
//! * [`Hand`] — a hand's **derived geometry**: which descendant chain is the
//!   index finger, which way a fingertip travels as it curls, how far each digit
//!   reaches. All of it read off the rig's own bind pose rather than authored,
//!   because a hand that arrives from a retarget or a glTF import has no table.
//! * [`apply_grip`] — the curl itself, cone-limited.
//!
//! # Why the frame is derived and not authored
//!
//! [`GripAffordance`] carries a palm transform, an aperture and five curl
//! targets — and *no axes*, because axes are not a property of a grip: they are a
//! property of the hand, and the hand already states them, in the only way a rig
//! ever states anything, which is where its bones are. Authoring them as well
//! would be a second description of one fact, and the SK1a audit's fourth
//! decision is about what happens to those.
//!
//! The derivation, in the hand joint's own bind frame:
//!
//! * **`along`** — hand origin to the farthest fingertip. Which way the fingers
//!   point.
//! * **`spread`** — the knuckle line, taken between the two finger roots that
//!   are farthest apart, signed **away from the thumb**. Which is what makes the
//!   four fingers nameable: sorted along it they are index, middle, ring, pinky,
//!   on any rig, without a single string compare.
//! * **`palm_in`** — where the thumb root sits once `along` and `spread` are
//!   projected out of it. A thumb is on the palm side of a hand; that is what a
//!   thumb *is*, and it is the only bone in a hand that says which side the palm
//!   is on.
//! * **`curl_axis`** — `along × palm_in`, per digit rather than per hand, so the
//!   thumb (whose bone does not run along the fingers) opposes instead of
//!   flexing sideways.
//!
//! # Determinism
//!
//! `psin64` / `pcos64` in `f64` and otherwise arithmetic, cross products and
//! `sqrt` — **no `sin`, no `cos`, no `acos`, no `from_axis_angle`** on the `f32`
//! path. The P14 law reaches here because a curled finger is a pose and a pose is
//! folded into `pose_state_bytes` and compared between the editor's PIE and the
//! shipped player. The `portable_pose` source gate covers this file.

use glam::{Quat, Vec3};

use crate::ik::apply_joint_limit;
use crate::pose::Pose;
use crate::roles::{BoneRoleKind, BoneSide, GripAffordance, RoleIndex};
use crate::skeleton::Skeleton;
use crate::template::JointLimit;

/// Which digit a finger chain is.
///
/// The order is the order [`GripAffordance::curl`] is indexed in — thumb first,
/// then index outward to the pinky — and it is a **wire order** in that sense: a
/// grip authored in a `.inf_skel` names its curl targets by position in that
/// array, so this list may not be reordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Digit {
    /// The thumb.
    Thumb = 0,
    /// The index finger.
    Index = 1,
    /// The middle finger.
    Middle = 2,
    /// The ring finger.
    Ring = 3,
    /// The little finger.
    Pinky = 4,
}

impl Digit {
    /// All five, in [`GripAffordance::curl`] order.
    pub const ALL: [Digit; 5] = [
        Digit::Thumb,
        Digit::Index,
        Digit::Middle,
        Digit::Ring,
        Digit::Pinky,
    ];

    /// This digit's slot in [`GripAffordance::curl`].
    pub fn slot(self) -> usize {
        self as usize
    }
}

/// **How far each bone of a digit may flex**, degrees, from the knuckle outward.
///
/// Human ranges, rounded: a metacarpal barely moves (the palm cups a little, and
/// the ring and little fingers are the ones that do it), the joint at the knuckle
/// gives about a right angle, the middle joint gives the most, and the last one
/// gives less again. They are the **cone half-angles** [`crate::manny`] authors
/// on every finger bone, so the number below and the number in the rig are one
/// number.
///
/// A chain longer than this is clamped to the last entry rather than refused: a
/// rig with four phalanges is unusual, not wrong.
pub const FINGER_FLEX_DEG: [f32; 4] = [12.0, 90.0, 105.0, 80.0];

/// [`FINGER_FLEX_DEG`] for a thumb, which has no metacarpal in its chain on the
/// mannequin and opposes rather than flexing.
pub const THUMB_FLEX_DEG: [f32; 4] = [45.0, 55.0, 75.0, 75.0];

/// How much a finger bone may **roll** about its own axis, degrees.
///
/// Small and symmetric: a finger is a hinge with slack, not a ball joint. It is
/// the `twist_deg` of the cones [`crate::manny`] authors, and it exists so that a
/// curl composed with a retarget cannot spin a fingertip.
pub const FINGER_TWIST_DEG: f32 = 10.0;

/// One digit of one hand: its bones, root first.
#[derive(Clone, Debug, PartialEq)]
pub struct FingerChain {
    /// Which digit this is.
    pub digit: Digit,
    /// The chain's joints, from the bone attached to the hand out to the tip.
    /// A finger's first entry is its metacarpal where the rig has one.
    pub joints: Vec<u16>,
    /// Where the curl carries the fingertip, in the **hand's** bind frame, unit
    /// length. See the module docs.
    pub curl_axis: [f32; 3],
    /// The chain's total bind length, metres — how far this digit reaches, and
    /// therefore the thickest thing it can close around ([`apply_grip`]).
    pub reach_m: f32,
}

/// A hand, as its own bind pose describes it.
#[derive(Clone, Debug, PartialEq)]
pub struct Hand {
    /// The hand joint.
    pub joint: u16,
    /// Which side of the body it is on.
    pub side: BoneSide,
    /// Hand origin toward the fingertips, in the hand's bind frame, unit length.
    pub along: [f32; 3],
    /// The knuckle line, index toward pinky, in the hand's bind frame.
    pub spread: [f32; 3],
    /// The way a fingertip travels as it curls, in the hand's bind frame.
    pub palm_in: [f32; 3],
    /// The five digits, in [`Digit`] order. A rig missing one leaves a `None`
    /// rather than shifting the others along.
    pub fingers: [Option<FingerChain>; 5],
}

impl Hand {
    /// This hand's chain for `digit`, if the rig has one.
    pub fn finger(&self, digit: Digit) -> Option<&FingerChain> {
        self.fingers[digit.slot()].as_ref()
    }

    /// How many digits this hand actually has.
    pub fn digit_count(&self) -> usize {
        self.fingers.iter().filter(|f| f.is_some()).count()
    }
}

/// What one [`apply_grip`] did — an **engagement counter**, because "the solver
/// ran" and "the solver moved a finger" are different claims and a gate that
/// cannot tell them apart certifies a no-op (the SK1a audit's first decision).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GripReport {
    /// Bones whose rotation this call wrote.
    pub joints: u32,
    /// Of those, how many an authored [`crate::template::ConeLimit`] pulled back.
    pub clamped: u32,
    /// Per-digit closure after the aperture was accounted for, in [`Digit`]
    /// order — the number the aperture actually produced, so a gate asserts it
    /// rather than the input.
    pub closure: [f32; 5],
}

/// The three joints an arm IK solve runs over — shoulder, elbow, wrist — for
/// `side`.
///
/// **The role table first** (the SK1a rule), and a name rule only for a rig that
/// carries no table. The name half is not decoration: `BodyPlan::BipedCanonical`
/// is the twenty-joint rig every committed clip in this repository is
/// index-bound to and it carries no side tables at all, so without it the engine
/// could not reach for anything on its own canonical rig.
///
/// Returns `None` unless all three are present **and form a parent chain**, which
/// is [`crate::ik::solve_chain`]'s own precondition: answering with three joints
/// that are not a chain would move the refusal from here to there, where it costs
/// a whole solve to discover.
pub fn arm_chain(skeleton: &Skeleton, roles: RoleIndex<'_>, side: BoneSide) -> Option<[u16; 3]> {
    let by_role = (
        roles.first(BoneRoleKind::UpperArm, side),
        roles.first(BoneRoleKind::LowerArm, side),
        roles.first(BoneRoleKind::Hand, side),
    );
    let chain = match by_role {
        (Some(a), Some(b), Some(c)) => [a, b, c],
        _ => arm_chain_by_name(skeleton, side)?,
    };
    let joints = skeleton.joints();
    for w in chain.windows(2) {
        let child = joints.get(w[1] as usize)?;
        if child.parent != Some(w[0]) {
            return None;
        }
    }
    Some(chain)
}

/// The **legacy** arm rule, for a rig with no role table — see [`arm_chain`].
///
/// The vocabulary is deliberately generous for [`crate::pose`]'s reason: a rig
/// arrives from wherever it arrives. `upper_arm_l` is this engine's canonical
/// spelling, `upperarm_l` is the mannequin's and every ALS clip's, `LeftArm` is
/// Mixamo's.
fn arm_chain_by_name(skeleton: &Skeleton, side: BoneSide) -> Option<[u16; 3]> {
    let want = |n: &str, part: &[&str]| -> bool {
        let n = n.to_ascii_lowercase();
        let sided = match side {
            BoneSide::Left => n.starts_with("left") || n.ends_with("_l") || n.ends_with(".l"),
            BoneSide::Right => n.starts_with("right") || n.ends_with("_r") || n.ends_with(".r"),
            BoneSide::Center => false,
        };
        sided && part.iter().any(|p| n.contains(p))
    };
    let find = |part: &[&str]| -> Option<u16> {
        skeleton
            .joints()
            .iter()
            .position(|j| want(&j.name, part))
            .map(|i| i as u16)
    };
    // "hand" last and matched exactly enough not to take `ik_hand_l`: the rig
    // that carries both is the one this rule exists for the absence of, and a
    // solve run over `[root, ik_hand_root, ik_hand_l]` is not an arm — the
    // `foot_l` versus `ik_foot_l` trap, one limb up.
    let upper = find(&["upperarm", "upper_arm", "leftarm", "rightarm"])?;
    let lower = find(&["lowerarm", "lower_arm", "forearm"])?;
    let hand = skeleton
        .joints()
        .iter()
        .position(|j| j.parent == Some(lower) && want(&j.name, &["hand", "wrist"]))
        .map(|i| i as u16)?;
    Some([upper, lower, hand])
}

/// **Where an elbow should point**, model space.
///
/// A pole is a *point the mid joint bends toward*, and an arm has no opinion of
/// its own about which way to fold: the mannequin's bind pose is a T-pose of pure
/// translations (SK1a's law — a bind pose here carries no rotation), so the
/// shoulder, elbow and wrist are exactly collinear and every bend plane is as
/// good as every other. Left to [`crate::ik::solve_chain`]'s default (the current
/// mid joint) an arm reaching forward would fold sideways, or not fold at all.
///
/// Backward is `-Z`: model space in this engine faces `+Z` (which is what
/// [`crate::manny`]'s `Place::Ball` means by *"forward along +Z to the ball of
/// the foot"*), and an elbow points behind the arm. The pole is put a whole
/// chain-length back from the midpoint of shoulder and target so it is never on
/// the line between them, which is the one case `two_bone_positions` has to fall
/// back out of.
pub fn elbow_pole(shoulder: Vec3, target: Vec3, chain_length: f32) -> Vec3 {
    let mid = (shoulder + target) * 0.5;
    mid + Vec3::new(
        0.0,
        0.0,
        -chain_length.abs().max(crate::ik::MIN_BONE_LENGTH_M),
    )
}

/// A knee's pole — [`elbow_pole`] mirrored, because a leg folds the other way.
///
/// Not used by this module; it is here so the sign convention lives in one place
/// rather than being rediscovered by the next caller that needs it.
pub fn knee_pole(hip: Vec3, target: Vec3, chain_length: f32) -> Vec3 {
    let mid = (hip + target) * 0.5;
    mid + Vec3::new(
        0.0,
        0.0,
        chain_length.abs().max(crate::ik::MIN_BONE_LENGTH_M),
    )
}

/// The bind transform of every joint **relative to `root`'s frame**, for the
/// subtree under `root` — position and rotation.
///
/// Entries outside the subtree are the identity and are never read; the walk is
/// one forward pass because the skeleton is topologically ordered.
fn bind_relative(skeleton: &Skeleton, root: u16) -> (Vec<Vec3>, Vec<Quat>) {
    let joints = skeleton.joints();
    let n = joints.len();
    let mut pos = vec![Vec3::ZERO; n];
    let mut rot = vec![Quat::IDENTITY; n];
    let mut inside = vec![false; n];
    if let Some(slot) = inside.get_mut(root as usize) {
        *slot = true;
    }
    for (i, j) in joints.iter().enumerate() {
        if i == root as usize {
            continue;
        }
        let Some(p) = j.parent else { continue };
        let p = p as usize;
        if !inside.get(p).copied().unwrap_or(false) {
            continue;
        }
        inside[i] = true;
        let pr = rot[p];
        pos[i] = pos[p] + pr * j.local_bind.translation_vec();
        rot[i] = (pr * j.local_bind.rotation_quat()).normalize();
    }
    (pos, rot)
}

/// The component of `v` with `axis` (assumed unit) projected out.
fn reject(v: Vec3, axis: Vec3) -> Vec3 {
    v - axis * v.dot(axis)
}

/// `v` normalized, or `None` if it is too short or not finite to carry a
/// direction — `ik::usable_length`'s discipline, spelled locally because this
/// module's degenerate cases are geometric rather than skeletal.
fn unit(v: Vec3) -> Option<Vec3> {
    if !v.is_finite() {
        return None;
    }
    let len2 = v.length_squared();
    if len2 <= 1.0e-12 {
        return None;
    }
    Some(v / len2.sqrt())
}

/// **Read a hand off its own bind pose** — the derivation the module docs
/// describe.
///
/// `hand` is the wrist joint. Returns `None` for a rig whose hand has no thumb
/// and no fingers, because every axis here is derived from where the digits are
/// and there is nothing to derive them from.
pub fn hand_of(skeleton: &Skeleton, roles: RoleIndex<'_>, hand: u16) -> Option<Hand> {
    let joints = skeleton.joints();
    if hand as usize >= joints.len() {
        return None;
    }
    let side = roles
        .role_of(hand)
        .map(|r| r.side)
        .unwrap_or(BoneSide::Center);
    let (pos, _) = bind_relative(skeleton, hand);

    // Every chain hanging off the hand that is made of digit bones. The role
    // table answers "is this a finger" where it exists; a table-less rig falls
    // back to the name rule, which is the crate's standing arrangement.
    let is_digit = |i: u16| -> Option<bool> {
        match roles.kind_of(i) {
            Some(BoneRoleKind::Finger) => Some(false),
            Some(BoneRoleKind::Thumb) => Some(true),
            Some(_) => None,
            None => {
                if roles.is_empty() {
                    let n = joints[i as usize].name.to_ascii_lowercase();
                    if n.contains("thumb") {
                        Some(true)
                    } else if ["index", "middle", "ring", "pinky", "little", "finger"]
                        .iter()
                        .any(|f| n.contains(f))
                    {
                        Some(false)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    };

    // A chain is grown from each digit bone that is a direct child of the hand,
    // taking at each step the first child that is also a digit bone. "First
    // child that is a digit" and not "first child" is `RoleIndex::deform_child`'s
    // lesson: the mannequin hangs `wrist_inner_l` and `weapon_l` off the hand
    // too, and a chain built from the first child would be a marker.
    let mut roots: Vec<(u16, bool)> = Vec::new();
    for (i, j) in joints.iter().enumerate() {
        if j.parent != Some(hand) {
            continue;
        }
        let i = i as u16;
        if let Some(thumb) = is_digit(i) {
            roots.push((i, thumb));
        }
    }
    if roots.is_empty() {
        return None;
    }
    let grow = |start: u16| -> Vec<u16> {
        let mut chain = vec![start];
        loop {
            let last = *chain.last().expect("non-empty");
            let next = joints.iter().enumerate().find_map(|(i, j)| {
                (j.parent == Some(last) && is_digit(i as u16).is_some()).then_some(i as u16)
            });
            match next {
                // A cycle is impossible (parents precede children) and a chain
                // longer than a hand is a rig bug, not a hang: bounded anyway.
                Some(n) if chain.len() < 8 => chain.push(n),
                _ => break,
            }
        }
        chain
    };
    let chains: Vec<(Vec<u16>, bool)> = roots.iter().map(|&(r, thumb)| (grow(r), thumb)).collect();

    let tip_of = |c: &[u16]| pos[*c.last().expect("non-empty") as usize];
    let (thumbs, fingers): (Vec<_>, Vec<_>) = chains.iter().partition(|(_, t)| *t);

    // ── `along`: the farthest fingertip ──
    let far = chains.iter().map(|(c, _)| c).max_by(|a, b| {
        tip_of(a)
            .length_squared()
            .total_cmp(&tip_of(b).length_squared())
    })?;
    let along = unit(tip_of(far))?;

    // ── `spread`: the knuckle line, between the two finger roots farthest apart ──
    //
    // Signed away from the thumb, which is what makes the four nameable. With no
    // thumb (or fewer than two fingers) the axis has no defensible sign, so the
    // digits keep the order the rig indexed them in — stated rather than guessed.
    let roots_xyz: Vec<Vec3> = fingers
        .iter()
        .map(|(c, _)| reject(pos[c[0] as usize], along))
        .collect();
    let thumb_root = thumbs
        .first()
        .map(|(c, _)| reject(pos[c[0] as usize], along));
    let mut spread = None;
    if roots_xyz.len() >= 2 {
        let mut best = (0usize, 1usize, -1.0f32);
        for a in 0..roots_xyz.len() {
            for b in (a + 1)..roots_xyz.len() {
                let d = (roots_xyz[b] - roots_xyz[a]).length_squared();
                if d > best.2 {
                    best = (a, b, d);
                }
            }
        }
        let (a, b, _) = best;
        // From the root nearer the thumb toward the one farther from it.
        let (a, b) = match thumb_root {
            Some(t) => {
                if (roots_xyz[a] - t).length_squared() <= (roots_xyz[b] - t).length_squared() {
                    (a, b)
                } else {
                    (b, a)
                }
            }
            None => (a, b),
        };
        spread = unit(roots_xyz[b] - roots_xyz[a]);
    }
    let spread = spread.unwrap_or_else(|| {
        // No two fingers to draw a line between. Any unit vector perpendicular to
        // `along` keeps the frame orthonormal and the digit order is the rig's.
        crate::ik::rotation_between(Vec3::X, along) * Vec3::Z
    });

    // ── `palm_in`: where the thumb is, once `along` and `spread` are removed ──
    let mean: Vec3 = if roots_xyz.is_empty() {
        Vec3::ZERO
    } else {
        roots_xyz.iter().fold(Vec3::ZERO, |a, b| a + *b) / roots_xyz.len() as f32
    };
    let palm_in = thumb_root
        .and_then(|t| unit(reject(reject(t - mean, along), spread)))
        // No thumb: the palm faces whichever way completes a right-handed frame.
        // A hand with no thumb cannot be asked which side its palm is on, and
        // this at least keeps the curl perpendicular to the fingers.
        .unwrap_or_else(|| along.cross(spread));

    // ── the digits, named by where they sit along `spread` ──
    let mut fingers_sorted: Vec<&(Vec<u16>, bool)> = fingers.to_vec();
    fingers_sorted.sort_by(|a, b| {
        let (x, y) = (
            pos[a.0[0] as usize].dot(spread),
            pos[b.0[0] as usize].dot(spread),
        );
        // Tie-broken by joint index, so a rig whose knuckles are exactly in line
        // still produces one answer rather than an allocator's.
        x.total_cmp(&y).then(a.0[0].cmp(&b.0[0]))
    });

    let mut out: [Option<FingerChain>; 5] = [None, None, None, None, None];
    let named = [Digit::Index, Digit::Middle, Digit::Ring, Digit::Pinky];
    let mut place = |digit: Digit, chain: &[u16]| {
        let root = pos[chain[0] as usize];
        let tip = pos[*chain.last().expect("non-empty") as usize];
        // Each digit curls about ITS OWN direction crossed with the palm, so a
        // thumb — whose bone does not run along the fingers — opposes rather than
        // flexing sideways.
        let dir = unit(tip - root).unwrap_or(along);
        let axis = unit(dir.cross(palm_in)).unwrap_or_else(|| along.cross(palm_in));
        let mut reach = 0.0f32;
        for w in chain.windows(2) {
            reach += (pos[w[1] as usize] - pos[w[0] as usize]).length();
        }
        // The tip bone has no child to measure to, so its own length is the
        // segment that carried it — a chain of three phalanges otherwise reads as
        // two bones long and every aperture answer would be short.
        if chain.len() >= 2 {
            let last = chain.len() - 1;
            reach += (pos[chain[last] as usize] - pos[chain[last - 1] as usize]).length();
        }
        out[digit.slot()] = Some(FingerChain {
            digit,
            joints: chain.to_vec(),
            curl_axis: axis.to_array(),
            reach_m: reach,
        });
    };
    for (d, (chain, _)) in named.iter().zip(fingers_sorted.iter()) {
        place(*d, chain);
    }
    if let Some((chain, _)) = thumbs.first() {
        place(Digit::Thumb, chain);
    }

    Some(Hand {
        joint: hand,
        side,
        along: along.to_array(),
        spread: spread.to_array(),
        palm_in: palm_in.to_array(),
        fingers: out,
    })
}

/// Both hands of a rig, `[left, right]`.
pub fn hands_of(skeleton: &Skeleton, roles: RoleIndex<'_>) -> [Option<Hand>; 2] {
    [BoneSide::Left, BoneSide::Right].map(|side| {
        let hand = roles
            .first(BoneRoleKind::Hand, side)
            .or_else(|| arm_chain(skeleton, roles, side).map(|c| c[2]))?;
        hand_of(skeleton, roles, hand)
    })
}

/// **How far a digit closes around a handle of `aperture_m`**, in `[0, 1]`.
///
/// A first-order model and named as one: a digit closing around a cylinder has to
/// get *past* it, so the fraction of its flexion it spends is the fraction of its
/// own reach the handle's radius does not already occupy. A handle as thick as
/// the finger is long leaves the finger straight; a handle of no thickness is a
/// fist.
///
/// It is per-digit rather than per-hand because that is what makes it mean
/// anything: a pinky closes further around a given rifle grip than an index does,
/// and it does so *because it is shorter*, which is the whole content of the
/// number. A wrap solve against the prop's actual surface is what this is not,
/// and the prop's surface is not something a skeleton can see.
pub fn digit_closure(reach_m: f32, aperture_m: f32) -> f32 {
    if !reach_m.is_finite() || reach_m <= 0.0 {
        return 0.0;
    }
    let radius = if aperture_m.is_finite() {
        aperture_m.max(0.0) * 0.5
    } else {
        return 0.0;
    };
    (1.0 - radius / reach_m).clamp(0.0, 1.0)
}

/// A rotation of `angle` radians about unit `axis`, built **portably**.
///
/// `Quat::from_axis_angle` is on the `portable_pose` gate's banned list because
/// it reaches `sin_cos` inside glam where no grep of this crate would see it —
/// the P23.5 finding, which is also why `ik::rotation_between` exists.
fn about(axis: Vec3, angle: f32) -> Quat {
    let half = angle as f64 * 0.5;
    let s = inf_math::psin64(half);
    Quat::from_xyzw(
        (axis.x as f64 * s) as f32,
        (axis.y as f64 * s) as f32,
        (axis.z as f64 * s) as f32,
        inf_math::pcos64(half) as f32,
    )
    .normalize()
}

/// **Close a hand onto a grip**, in place.
///
/// `amount` in `[0, 1]` is how far into the grip the hand is — a grip eases in
/// and a release eases out, and a value of zero writes nothing at all, so a
/// released hand poses exactly the bytes an ungripped one does.
///
/// Each bone is set to `bind · curl`, not `current · curl`: a curl is a *pose*,
/// not a delta, so re-running it with the same arguments is idempotent and a
/// grip held for a thousand steps does not wind a finger into a spiral. That is
/// the same reading [`crate::drive`] gives a twist bone, for the same reason.
///
/// Every bone is then put through [`apply_joint_limit`], so an authored
/// [`crate::template::ConeLimit`] pulls back a curl the knuckle cannot give. On
/// [`crate::manny`] every finger bone carries one.
pub fn apply_grip(
    skeleton: &Skeleton,
    pose: &mut Pose,
    hand: &Hand,
    limits: &[JointLimit],
    grip: &GripAffordance,
    amount: f32,
) -> GripReport {
    let mut report = GripReport::default();
    if !amount.is_finite() || amount <= 0.0 {
        return report;
    }
    let amount = amount.min(1.0);
    let joints = skeleton.joints();
    let n = joints.len().min(pose.locals.len());
    // Once, not once per digit: the walk is O(joints) and there are five of them.
    let (_, rel) = bind_relative(skeleton, hand.joint);
    for digit in Digit::ALL {
        let Some(chain) = hand.finger(digit) else {
            continue;
        };
        let want = grip.curl[digit.slot()];
        if !want.is_finite() {
            continue;
        }
        let closure = digit_closure(chain.reach_m, grip.aperture_m);
        report.closure[digit.slot()] = closure;
        let curl = want.clamp(0.0, 1.0) * closure * amount;
        if curl <= 0.0 {
            continue;
        }
        let axis_hand = Vec3::from_array(chain.curl_axis);
        // The axis has to be expressed in the frame the local rotation acts in,
        // which is the joint's PARENT's — a local rotation composes on the left
        // of the bind rotation and therefore rotates within the parent's frame.
        // On the mannequin every hand bind rotation is the identity and this is
        // the axis itself; on a rig whose hand is rotated at bind it is not, and
        // the difference is a fingertip that curls sideways.
        let flex = if digit == Digit::Thumb {
            THUMB_FLEX_DEG
        } else {
            FINGER_FLEX_DEG
        };
        for (k, &j) in chain.joints.iter().enumerate() {
            let ji = j as usize;
            if ji >= n {
                continue;
            }
            let max_deg = flex[k.min(flex.len() - 1)];
            let angle = (curl * max_deg).to_radians();
            let parent_rel = joints[ji]
                .parent
                .and_then(|p| rel.get(p as usize).copied())
                .unwrap_or(Quat::IDENTITY);
            let axis_local = (parent_rel.inverse() * axis_hand).normalize_or_zero();
            if axis_local.length_squared() <= 0.0 {
                continue;
            }
            let bind = joints[ji].local_bind.rotation_quat();
            let mut local = (about(axis_local, angle) * bind).normalize();
            if let Some(limit) = limits.iter().find(|l| l.joint == j) {
                let (fixed, moved) = apply_joint_limit(local, bind, limit);
                local = fixed;
                if moved {
                    report.clamped += 1;
                }
            }
            if !local.is_finite() {
                continue;
            }
            pose.locals[ji].rotation = local.to_array();
            report.joints += 1;
        }
    }
    report
}

/// **Reach a three-joint limb whose middle joint is a HINGE** — exactly (SK1b).
///
/// # Why this exists beside [`crate::ik::solve_chain`]
///
/// `solve_chain` places the joints first (law of cosines, positional form) and
/// turns the positions into rotations second, clamping each as it writes it. That
/// is the right shape for a chain of *unconstrained* joints, and it is measurably
/// the wrong shape for an arm: the elbow position it chooses comes from a **pole**,
/// the pole picks a bend plane freely, and the clamp then discards whatever
/// component of the bend does not lie in the hinge's own plane. Measured on the
/// mannequin, reaching a point 55 cm in front of and 29 cm below the shoulder:
///
/// | | reach error |
/// |---|---|
/// | no limits at all | **3e-8 m** |
/// | P24.1's `hinge_x` elbow | **0.484 m** — the elbow pinned straight |
/// | a correct `hinge_y` elbow through `solve_chain` | **0.083 m**, and iterating the pole is a fixed point |
/// | a correct `hinge_y` elbow through **this** | **exact** |
///
/// # How, and why there is no transcendental in it
///
/// A hinge takes the freedom away, so the answer is closed-form and there is
/// nothing to search for:
///
/// 1. **The elbow angle is determined by the distance alone.** The interior angle
///    `θ` at the elbow has `cos θ = (l₁² + l₂² − d²) / 2·l₁·l₂`, and the bend away
///    from straight is `π − θ`. A quaternion needs the *half* angle, and the
///    half-angle identities give it from the cosine with two square roots —
///    `sin(φ/2) = √((1 + cos θ)/2)`, `cos(φ/2) = √((1 − cos θ)/2)` — so **no
///    `acos`, no `sin`, no `cos`** appears. (`psin64`/`pcos64` are used only on the
///    path where the authored range actually clamps the bend, which is a rebuild
///    from a known angle rather than a measurement.)
/// 2. **The elbow is set about its own hinge axis**, in the sign the authored
///    range permits. With the elbow set, the shoulder-to-wrist distance is now
///    exactly `d` by construction.
/// 3. **The shoulder aims the assembly**, one `rotation_between`. Aiming a rigid
///    two-bone assembly whose end is already at the right distance puts the end
///    exactly on the target — which is why this is exact rather than iterated.
///
/// The bend plane is left where the pose already had it (the aim is the minimal
/// rotation, which adds no roll), so an arm keeps whatever orientation the
/// animation gave it and does not snap into a canonical plane.
///
/// # Refusals, and what is not one
///
/// Returns [`crate::ik::IkError`] for a chain that is not a chain, a joint that does not
/// exist, a degenerate bone or a non-finite target — [`crate::ik::solve_chain`]'s
/// list, because it is the same list. A target **out of reach** is not a refusal:
/// the arm extends toward it, exactly as a real one does, and
/// [`crate::ik::IkReport::reached`] says which happened. A middle joint that is **not** a
/// single-axis hinge is not a refusal either: the solve delegates to
/// `solve_chain` with a derived pole, so a caller writes one call rather than a
/// branch.
pub fn reach(
    skeleton: &Skeleton,
    pose: &mut Pose,
    chain: [u16; 3],
    target: Vec3,
    limits: &[JointLimit],
) -> Result<crate::ik::IkReport, crate::ik::IkError> {
    use crate::ik::{rotation_between, IkError, IkReport, MIN_BONE_LENGTH_M, REACH_TOLERANCE_M};
    use crate::pose::global_transforms;

    let joints = skeleton.joints();
    for &j in &chain {
        if skeleton.joint(j as usize).is_none() {
            return Err(IkError::NoSuchJoint {
                joint: j,
                joints: skeleton.len(),
            });
        }
    }
    for w in chain.windows(2) {
        if joints[w[1] as usize].parent != Some(w[0]) {
            return Err(IkError::NotAChain {
                parent: w[0],
                child: w[1],
            });
        }
    }
    if !target.is_finite() {
        return Err(IkError::NonFinite {
            what: "an arm IK target",
            value: target.to_array(),
        });
    }

    // The hinge, read off the authored table. Exactly one free axis is a hinge;
    // anything else has a bend plane this cannot know, so it goes the other way.
    let elbow_limit = limits.iter().find(|l| l.joint == chain[1]);
    let hinge = elbow_limit.and_then(|l| {
        if l.cone.is_some() {
            return None;
        }
        let mut found = None;
        for a in 0..3 {
            if l.is_free(a) {
                if found.is_some() {
                    return None;
                }
                found = Some(a);
            }
        }
        found.map(|a| (a, l.min_deg[a], l.max_deg[a]))
    });
    let Some((axis_idx, lo_deg, hi_deg)) = hinge else {
        let g = global_transforms(skeleton, pose);
        let at = |j: u16| g[j as usize].transform_point3(Vec3::ZERO);
        let span = (at(chain[1]) - at(chain[0])).length() + (at(chain[2]) - at(chain[1])).length();
        let pole = elbow_pole(at(chain[0]), target, span);
        return crate::ik::solve_chain(skeleton, pose, &chain, target, Some(pole), limits);
    };

    let globals = global_transforms(skeleton, pose);
    let at = |j: u16| globals[j as usize].transform_point3(Vec3::ZERO);
    let (shoulder, elbow, wrist) = (at(chain[0]), at(chain[1]), at(chain[2]));
    if !shoulder.is_finite() || !elbow.is_finite() || !wrist.is_finite() {
        return Err(IkError::NonFinite {
            what: "an arm chain joint's model-space position",
            value: shoulder.to_array(),
        });
    }
    let l1 = (elbow - shoulder).length();
    let l2 = (wrist - elbow).length();
    for (parent, child, len) in [(chain[0], chain[1], l1), (chain[1], chain[2], l2)] {
        if !(len.is_finite() && len >= MIN_BONE_LENGTH_M) {
            return Err(IkError::DegenerateBone {
                parent,
                child,
                length: len,
            });
        }
    }
    let chain_length = l1 + l2;

    // ── 1. the elbow angle the distance implies ──
    //
    // Clamped into the range the two bones can actually span, so an unreachable
    // target becomes full extension (or a full fold) rather than a NaN — the
    // `two_bone_positions` contract, restated.
    let to_target = target - shoulder;
    let d = to_target.length().clamp((l1 - l2).abs(), chain_length);
    let cos_theta = (((l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2)) as f64).clamp(-1.0, 1.0);
    // The bend AWAY from straight is `π − θ`, so its half-angle sine is θ's
    // half-angle cosine and vice versa. Two square roots and no angle is ever
    // formed.
    let mut half_sin = ((1.0 + cos_theta) * 0.5).sqrt();
    let mut half_cos = ((1.0 - cos_theta) * 0.5).sqrt();

    // ── the authored range ──
    //
    // Which way the hinge folds is the SIGN of the range: a knee's is negative and
    // a right elbow's is negative, a left elbow's is positive. A range that
    // straddles zero takes the wider side, because an elbow that may fold either
    // way is one this rig did not mean to constrain.
    let sign = if hi_deg.abs() >= lo_deg.abs() {
        1.0
    } else {
        -1.0
    };
    let max_bend_rad = (hi_deg.abs().max(lo_deg.abs()) as f64).to_radians();
    let mut clamped = 0u32;
    // `2·atan2(sin, cos)` is the bend this is about to write; comparing it against
    // the range is the only place an angle is formed at all, and it is formed to
    // be *compared*, not to be built from.
    let bend = 2.0 * inf_math::patan2_64(half_sin, half_cos);
    if bend > max_bend_rad {
        clamped = 1;
        let half = max_bend_rad * 0.5;
        half_sin = inf_math::psin64(half);
        half_cos = inf_math::pcos64(half);
    }

    let mut axis = Vec3::ZERO;
    axis[axis_idx] = sign;
    let bind = joints[chain[1] as usize].local_bind.rotation_quat();
    let flex = Quat::from_xyzw(
        (axis.x as f64 * half_sin) as f32,
        (axis.y as f64 * half_sin) as f32,
        (axis.z as f64 * half_sin) as f32,
        half_cos as f32,
    )
    .normalize();
    let elbow_local = (flex * bind).normalize();
    if !elbow_local.is_finite() {
        return Err(IkError::NonFinite {
            what: "an arm IK elbow rotation",
            value: [half_sin as f32, half_cos as f32, sign],
        });
    }
    pose.locals[chain[1] as usize].rotation = elbow_local.to_array();

    // ── 3. aim the assembly ──
    let globals = global_transforms(skeleton, pose);
    let at = |j: u16| globals[j as usize].transform_point3(Vec3::ZERO);
    let (shoulder, wrist) = (at(chain[0]), at(chain[2]));
    let (Some(from), Some(to)) = (unit(wrist - shoulder), unit(to_target)) else {
        // The wrist is ON the shoulder, or the target is. Nothing to aim; the
        // elbow above is still the honest answer for the distance.
        let end = crate::pose::global_transforms(skeleton, pose)[chain[2] as usize]
            .transform_point3(Vec3::ZERO);
        let reach_error = (end - target).length();
        return Ok(IkReport {
            reach_error,
            reached: reach_error <= REACH_TOLERANCE_M,
            chain_length,
            clamped,
        });
    };
    let delta = rotation_between(from, to);
    let parent_rot = match joints[chain[0] as usize].parent {
        Some(p) => rotation_of(&globals[p as usize]),
        None => Quat::IDENTITY,
    };
    let global_rot = rotation_of(&globals[chain[0] as usize]);
    let mut local = (parent_rot.inverse() * (delta * global_rot)).normalize();
    if let Some(limit) = limits.iter().find(|l| l.joint == chain[0]) {
        let bind = joints[chain[0] as usize].local_bind.rotation_quat();
        let (fixed, moved) = apply_joint_limit(local, bind, limit);
        local = fixed;
        if moved {
            clamped += 1;
        }
    }
    if local.is_finite() {
        pose.locals[chain[0] as usize].rotation = local.to_array();
    }

    let end = global_transforms(skeleton, pose)[chain[2] as usize].transform_point3(Vec3::ZERO);
    let reach_error = (end - target).length();
    Ok(IkReport {
        reach_error,
        reached: reach_error <= REACH_TOLERANCE_M,
        chain_length,
        clamped,
    })
}

/// The rotation part of an affine matrix, normalized — `ik`'s helper, which is
/// private there.
fn rotation_of(m: &glam::Mat4) -> Quat {
    Quat::from_mat4(m).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manny::build_manny;
    use crate::template::{build_template, BodyParams, BodyPlan};

    fn manny() -> crate::SkeletonAsset {
        build_manny(&BodyParams::default()).expect("the mannequin builds")
    }

    /// **The four fingers are named by where they are, and the names are right.**
    ///
    /// Nothing in [`hand_of`] compares a string on a rig that carries a role
    /// table; the digit order comes out of the knuckle line's sign, which comes
    /// out of where the thumb is. This arm is the only thing that says the
    /// derivation agrees with the rig's own vocabulary — and it asserts BOTH
    /// hands, because the sign of `spread` is the half that a mirror can flip.
    #[test]
    fn the_digits_are_derived_and_they_are_the_ones_the_rig_names() {
        let asset = manny();
        let sk = &asset.skeleton;
        let hands = hands_of(sk, asset.role_index());
        for (side, hand) in [(BoneSide::Left, &hands[0]), (BoneSide::Right, &hands[1])] {
            let hand = hand.as_ref().unwrap_or_else(|| panic!("{side:?} hand"));
            assert_eq!(hand.digit_count(), 5, "{side:?}: five digits");
            let suffix = if side == BoneSide::Left { "_l" } else { "_r" };
            for (digit, want) in [
                (Digit::Thumb, "thumb"),
                (Digit::Index, "index"),
                (Digit::Middle, "middle"),
                (Digit::Ring, "ring"),
                (Digit::Pinky, "pinky"),
            ] {
                let chain = hand.finger(digit).unwrap_or_else(|| panic!("{digit:?}"));
                for &j in &chain.joints {
                    let name = &sk.joints()[j as usize].name;
                    assert!(
                        name.starts_with(want) && name.ends_with(suffix),
                        "{side:?} {digit:?} took `{name}`, which is not a {want}{suffix}"
                    );
                }
            }
            // A finger is metacarpal + three phalanges; a thumb has no
            // metacarpal in this rig's chain.
            assert_eq!(hand.finger(Digit::Index).unwrap().joints.len(), 4);
            assert_eq!(hand.finger(Digit::Thumb).unwrap().joints.len(), 3);
            // …and the wrist markers and the weapon socket bone are NOT digits.
            let taken: Vec<u16> = Digit::ALL
                .iter()
                .filter_map(|d| hand.finger(*d))
                .flat_map(|c| c.joints.clone())
                .collect();
            for marker in ["wrist_inner", "wrist_outer", "weapon"] {
                let j = sk
                    .index_of(&format!("{marker}{suffix}"))
                    .unwrap_or_else(|| panic!("{marker}{suffix} exists"));
                assert!(
                    !taken.contains(&j),
                    "{marker}{suffix} was taken for a digit"
                );
            }
        }
    }

    /// **The two hands are mirrors, and the curl axes are opposite.**
    ///
    /// The mirror is what makes a single curl rule work on both hands: `along`
    /// flips with the side and `palm_in` does not, so `curl_axis` flips — which
    /// is the difference between two hands closing and two hands closing in
    /// opposite directions, one of them through the palm.
    #[test]
    fn the_hands_mirror_and_both_curl_toward_the_palm() {
        let asset = manny();
        let hands = hands_of(&asset.skeleton, asset.role_index());
        let (l, r) = (hands[0].as_ref().unwrap(), hands[1].as_ref().unwrap());
        let (la, ra) = (Vec3::from_array(l.along), Vec3::from_array(r.along));
        assert!((la.x + ra.x).abs() < 1.0e-5, "along did not mirror in x");
        assert!((la.y - ra.y).abs() < 1.0e-5 && (la.z - ra.z).abs() < 1.0e-5);
        // The palm faces the same way on both hands — the thumbs are both under
        // the palm plane in this rig's frame — and the small component along the
        // arm axis mirrors with the side, exactly as `along` does.
        let (lp, rp) = (Vec3::from_array(l.palm_in), Vec3::from_array(r.palm_in));
        assert!((lp.x + rp.x).abs() < 1.0e-5, "palm_in did not mirror in x");
        assert!((lp.y - rp.y).abs() < 1.0e-5 && (lp.z - rp.z).abs() < 1.0e-5);
        assert!(
            lp.y < -0.9,
            "the palm should face -Y on this rig, got {lp:?}"
        );
        for digit in Digit::ALL {
            let (a, b) = (
                Vec3::from_array(l.finger(digit).unwrap().curl_axis),
                Vec3::from_array(r.finger(digit).unwrap().curl_axis),
            );
            assert!(
                a.dot(b) < -0.5,
                "{digit:?}: the two hands curl about {a:?} and {b:?}, which is not a mirror"
            );
        }
    }

    /// **A curl closes the hand, and a bigger curl closes it further.**
    ///
    /// Two claims, and they need two different measurements — which is the
    /// finding this arm was rewritten around. *Direction* is the fingertip's
    /// displacement projected onto `palm_in`: the mutation it kills is a sign
    /// error, an axis the wrong way round curling the fingers out through the
    /// back of the hand, which every count-based assertion in this file is
    /// perfectly happy with. *Amount* is *not* that projection, because the
    /// projection is *not monotone*: a real fist takes the fingertip past the
    /// palm and back up toward the knuckles, and measured here it peaks at a half
    /// curl (0.073 m) and falls to 0.051 at a full one. The monotone quantity is
    /// how far the tip is **from the wrist**, and on this rig it runs 18.4 cm
    /// straight → 8.1 cm closed, which is a fist.
    #[test]
    fn a_curl_closes_the_hand_and_a_bigger_curl_closes_it_further() {
        let asset = manny();
        let sk = &asset.skeleton;
        let hand = hands_of(sk, asset.role_index())[1].clone().expect("right");
        let tip = *hand
            .finger(Digit::Middle)
            .unwrap()
            .joints
            .last()
            .expect("a tip") as usize;
        let palm_in = Vec3::from_array(hand.palm_in);
        let wrist = hand.joint as usize;
        let rest = crate::pose::global_transforms(sk, &Pose::rest(sk));

        let measure = |curl: f32| -> (f32, f32, GripReport) {
            let mut pose = Pose::rest(sk);
            let grip = GripAffordance {
                name: "fist".into(),
                hand: hand.joint,
                palm: crate::JointTransform::IDENTITY,
                aperture_m: 0.0,
                curl: [curl; 5],
            };
            let report = apply_grip(sk, &mut pose, &hand, &asset.limits, &grip, 1.0);
            let g = crate::pose::global_transforms(sk, &pose);
            // Toward the palm, in the hand's own frame.
            let d = g[tip].w_axis.truncate() - rest[tip].w_axis.truncate();
            let toward = (glam::Quat::from_mat4(&g[wrist]).normalize().inverse() * d).dot(palm_in);
            // …and how far the tip still is from the wrist, which is the half
            // that stays monotone all the way to a closed fist.
            let span = (g[tip].w_axis.truncate() - g[wrist].w_axis.truncate()).length();
            (toward, span, report)
        };

        let (none, straight, r0) = measure(0.0);
        assert_eq!(r0.joints, 0, "a zero curl must write nothing");
        assert_eq!(none, 0.0);
        let (toward, _, r1) = measure(0.5);
        assert!(r1.joints >= 15, "a curl wrote {} bones", r1.joints);
        assert!(
            toward > 0.01,
            "a half curl moved the fingertip {toward} m toward the palm — the sign is wrong"
        );
        // Monotone in the quantity that IS monotone, over the whole range.
        let mut last = straight;
        for f in [0.25f32, 0.5, 0.75, 1.0] {
            let (_, span, r) = measure(f);
            assert_eq!(
                r.joints, r1.joints,
                "a curl of {f} wrote a different bone set"
            );
            assert!(
                span < last,
                "a curl of {f} left the fingertip {span} m from the wrist, no closer than {last}"
            );
            last = span;
        }
        assert!(
            last < straight * 0.5,
            "a closed fist left the fingertip {last} m from the wrist against {straight} m straight"
        );
    }

    /// **The aperture is load-bearing**, and it is per digit.
    #[test]
    fn a_thicker_handle_straightens_the_fingers_and_the_pinky_gives_up_first() {
        let asset = manny();
        let sk = &asset.skeleton;
        let hand = hands_of(sk, asset.role_index())[1].clone().expect("right");
        let index = hand.finger(Digit::Index).unwrap().reach_m;
        let pinky = hand.finger(Digit::Pinky).unwrap().reach_m;
        assert!(pinky < index, "a pinky is shorter than an index finger");
        assert_eq!(digit_closure(index, 0.0), 1.0, "no handle is a fist");
        assert_eq!(digit_closure(index, 10.0), 0.0, "a tree is not gripped");
        // At the same aperture the shorter digit spends LESS of its flexion,
        // because the handle takes up more of its reach.
        let a = 0.04;
        assert!(
            digit_closure(pinky, a) < digit_closure(index, a),
            "the pinky closed at least as far as the index around a {a} m handle"
        );
        // Monotone, and the report says what actually happened rather than what
        // was asked for.
        let grip = |aperture: f32| {
            let mut pose = Pose::rest(sk);
            let g = GripAffordance {
                name: "handle".into(),
                hand: hand.joint,
                palm: crate::JointTransform::IDENTITY,
                aperture_m: aperture,
                curl: [1.0; 5],
            };
            apply_grip(sk, &mut pose, &hand, &asset.limits, &g, 1.0)
        };
        let thin = grip(0.02);
        let thick = grip(0.06);
        for d in Digit::ALL {
            assert!(
                thin.closure[d.slot()] > thick.closure[d.slot()],
                "{d:?}: a 2 cm handle closed no further than a 6 cm one"
            );
        }
        // A handle wider than the hand leaves every finger straight and writes
        // nothing at all.
        let open = grip(1.0);
        assert_eq!(open.joints, 0);
        assert_eq!(open.closure, [0.0; 5]);
    }

    /// **The cones are enforced** — the SK1a audit's *"authored and enforced by
    /// nothing"* closed, measured through the real door.
    ///
    /// A curl asked for far past what the rig allows comes back at the rig's
    /// number, and the report says how many bones the cone had to pull back.
    #[test]
    fn a_cone_pulls_back_a_curl_the_knuckle_cannot_give() {
        let asset = manny();
        let sk = &asset.skeleton;
        assert!(
            asset.limits.iter().filter(|l| l.cone.is_some()).count() >= 30,
            "the mannequin should author a cone on every finger bone, found {}",
            asset.limits.iter().filter(|l| l.cone.is_some()).count()
        );
        let hand = hands_of(sk, asset.role_index())[1].clone().expect("right");
        let grip = GripAffordance {
            name: "over".into(),
            hand: hand.joint,
            palm: crate::JointTransform::IDENTITY,
            aperture_m: 0.0,
            curl: [1.0; 5],
        };
        // With the rig's own limits: nothing is clamped, because the flex table
        // and the authored cones are one number (`FINGER_FLEX_DEG`).
        let mut pose = Pose::rest(sk);
        let honest = apply_grip(sk, &mut pose, &hand, &asset.limits, &grip, 1.0);
        assert_eq!(
            honest.clamped, 0,
            "the solver's own maxima should already sit inside the authored cones"
        );
        // Halve every cone and the same grip is pulled back on every bone.
        let tight: Vec<JointLimit> = asset
            .limits
            .iter()
            .map(|l| {
                let mut l = *l;
                if let Some(c) = l.cone.as_mut() {
                    c.swing_deg *= 0.5;
                }
                l
            })
            .collect();
        let mut pose_t = Pose::rest(sk);
        let clamped = apply_grip(sk, &mut pose_t, &hand, &tight, &grip, 1.0);
        assert_eq!(clamped.joints, honest.joints);
        assert!(
            clamped.clamped >= 15,
            "halving every cone pulled back only {} bones of {}",
            clamped.clamped,
            clamped.joints
        );
        // …and the pose really is less closed, not merely differently reported.
        let tip = *hand.finger(Digit::Middle).unwrap().joints.last().unwrap() as usize;
        let g = crate::pose::global_transforms(sk, &pose);
        let gt = crate::pose::global_transforms(sk, &pose_t);
        let rest = crate::pose::global_transforms(sk, &Pose::rest(sk));
        let moved =
            |gs: &[glam::Mat4]| (gs[tip].w_axis.truncate() - rest[tip].w_axis.truncate()).length();
        assert!(
            moved(&gt) < moved(&g),
            "the clamped hand closed as far as the free one ({} vs {})",
            moved(&gt),
            moved(&g)
        );
    }

    /// **A grip is a pose, not a delta**: applying it twice is applying it once.
    #[test]
    fn a_grip_is_idempotent_and_bit_stable() {
        let asset = manny();
        let sk = &asset.skeleton;
        let hand = hands_of(sk, asset.role_index())[0].clone().expect("left");
        let grip = GripAffordance {
            name: "rifle".into(),
            hand: hand.joint,
            palm: crate::JointTransform::IDENTITY,
            aperture_m: 0.035,
            curl: [0.8, 0.9, 1.0, 1.0, 1.0],
        };
        let run = |times: usize| {
            let mut pose = Pose::rest(sk);
            for _ in 0..times {
                apply_grip(sk, &mut pose, &hand, &asset.limits, &grip, 1.0);
            }
            pose
        };
        let (once, thrice) = (run(1), run(3));
        for (i, (a, b)) in once.locals.iter().zip(thrice.locals.iter()).enumerate() {
            for (u, v) in a.rotation.iter().zip(b.rotation.iter()) {
                assert_eq!(
                    u.to_bits(),
                    v.to_bits(),
                    "joint {i} wound up over three calls"
                );
            }
        }
        // …and two runs of the same call are bit-identical, which is the claim
        // the determinism trace rests on.
        let (a, b) = (run(1), run(1));
        assert_eq!(a.locals, b.locals);
    }

    /// **The arm chain is three joints that really are a chain**, on both the
    /// mannequin (role table) and the canonical biped (name rule).
    #[test]
    fn an_arm_chain_comes_from_the_table_first_and_the_names_second() {
        let asset = manny();
        let sk = &asset.skeleton;
        for (side, want) in [
            (BoneSide::Left, ["upperarm_l", "lowerarm_l", "hand_l"]),
            (BoneSide::Right, ["upperarm_r", "lowerarm_r", "hand_r"]),
        ] {
            let chain = arm_chain(sk, asset.role_index(), side).expect("an arm");
            for (j, name) in chain.iter().zip(want.iter()) {
                assert_eq!(&sk.joints()[*j as usize].name, name);
            }
        }
        // The name rule, on a rig that carries no table at all.
        let canonical = build_template(BodyPlan::BipedCanonical, &BodyParams::default())
            .expect("the canonical biped builds");
        assert!(canonical.roles.is_empty(), "the fixture must be table-less");
        let chain = arm_chain(&canonical.skeleton, canonical.role_index(), BoneSide::Left)
            .expect("the name rule finds an arm");
        assert_eq!(
            canonical.skeleton.joints()[chain[2] as usize].name,
            "hand_l"
        );
        // …and the trap it was written against: with the table present, the rule
        // must not take an `ik_hand_*` marker.
        let chain = arm_chain(sk, asset.role_index(), BoneSide::Right).expect("an arm");
        assert!(!sk.joints()[chain[2] as usize].name.starts_with("ik_"));
    }

    /// **A degenerate hand costs its own row.** A rig with no digits, a grip
    /// naming a NaN curl, an amount of zero — none of them writes a pose.
    #[test]
    fn a_hand_with_nothing_to_grip_with_writes_nothing() {
        let asset = manny();
        let sk = &asset.skeleton;
        let hand = hands_of(sk, asset.role_index())[1].clone().expect("right");
        let base = Pose::rest(sk);
        for (why, grip, amount) in [
            (
                "a NaN curl",
                GripAffordance {
                    name: "n".into(),
                    hand: hand.joint,
                    palm: crate::JointTransform::IDENTITY,
                    aperture_m: 0.0,
                    curl: [f32::NAN; 5],
                },
                1.0f32,
            ),
            (
                "a zero amount",
                GripAffordance::new("z", hand.joint, 0.0),
                0.0,
            ),
            (
                "a NaN amount",
                GripAffordance::new("z", hand.joint, 0.0),
                f32::NAN,
            ),
            (
                "a NaN aperture",
                GripAffordance {
                    name: "a".into(),
                    hand: hand.joint,
                    palm: crate::JointTransform::IDENTITY,
                    aperture_m: f32::NAN,
                    curl: [1.0; 5],
                },
                1.0,
            ),
        ] {
            let mut pose = base.clone();
            let r = apply_grip(sk, &mut pose, &hand, &asset.limits, &grip, amount);
            assert_eq!(r.joints, 0, "{why} wrote a pose");
            assert_eq!(pose.locals, base.locals, "{why} moved a bone");
        }
        // A hand joint that is not a hand has no digits and is refused at the
        // door rather than producing an empty `Hand`.
        let root = 0u16;
        assert!(hand_of(sk, asset.role_index(), root).is_none());
        assert!(hand_of(sk, asset.role_index(), 9999).is_none());
    }

    /// **The hinged solve is exact where the pole solve is not** — the table in
    /// [`reach`]'s docs, asserted, so the numbers in it are numbers a test prints.
    ///
    /// This is the arm that justifies `reach` existing beside `solve_chain` at
    /// all. Mutation: route `reach` straight to `solve_chain` and the first
    /// assertion goes red at 0.083 m.
    #[test]
    fn a_hinged_arm_reaches_exactly_where_a_pole_solve_misses() {
        use crate::pose::global_transforms;
        let asset = manny();
        let sk = &asset.skeleton;
        for side in [BoneSide::Left, BoneSide::Right] {
            let chain = arm_chain(sk, asset.role_index(), side).expect("an arm");
            let sx: f32 = if side == BoneSide::Left { -1.0 } else { 1.0 };
            // In front of the chest and below the shoulder — a point no bend
            // plane through a straight `-Z` pole contains.
            let target = Vec3::new(sx * 0.25, 1.15, 0.45);

            let mut hinged = Pose::rest(sk);
            let r = reach(sk, &mut hinged, chain, target, &asset.limits).expect("a solve");
            println!("{side:?} hinged reach error {:.7} m", r.reach_error);
            assert!(
                r.reached && r.reach_error < 1.0e-4,
                "{side:?}: the hinged solve missed by {} m",
                r.reach_error
            );

            // The same target, the same limits, through the pole solver.
            let mut poled = Pose::rest(sk);
            let g = global_transforms(sk, &poled);
            let at = |j: u16| g[j as usize].transform_point3(Vec3::ZERO);
            let span =
                (at(chain[1]) - at(chain[0])).length() + (at(chain[2]) - at(chain[1])).length();
            let pole = elbow_pole(at(chain[0]), target, span);
            let p =
                crate::ik::solve_chain(sk, &mut poled, &chain, target, Some(pole), &asset.limits)
                    .expect("a solve");
            println!("{side:?} pole reach error {:.5} m", p.reach_error);
            assert!(
                p.reach_error > 0.05,
                "{side:?}: the pole solve managed {} m, so this comparison proves nothing",
                p.reach_error
            );

            // …and the elbow really did bend about its authored hinge axis and
            // nothing else, which is what makes it exact.
            let elbow = Quat::from_array(hinged.locals[chain[1] as usize].rotation);
            let v = Vec3::new(elbow.x, elbow.y, elbow.z);
            assert!(
                v.length() > 1.0e-3 && v.normalize().y.abs() > 0.9999,
                "{side:?}: the elbow turned about {v:?}, which is not its hinge"
            );
            // The sign is the side's: a left forearm flexes about +Y, a right
            // one about -Y.
            assert!(
                v.y * sx < 0.0,
                "{side:?}: the elbow folded the wrong way ({v:?})"
            );
        }
    }

    /// An **unreachable** target is not a refusal — the arm extends toward it and
    /// says it did not get there. And a target the arm cannot fold tightly enough
    /// to touch is the same shape at the other end.
    #[test]
    fn an_out_of_reach_target_extends_the_arm_and_reports_the_miss() {
        let asset = manny();
        let sk = &asset.skeleton;
        let chain = arm_chain(sk, asset.role_index(), BoneSide::Right).expect("an arm");
        let mut pose = Pose::rest(sk);
        let far = Vec3::new(0.0, 1.4, 40.0);
        let r = reach(sk, &mut pose, chain, far, &asset.limits).expect("a solve");
        assert!(!r.reached, "40 m away is not reached");
        println!(
            "out of reach: error {:.3} m over a {:.3} m arm",
            r.reach_error, r.chain_length
        );
        // The arm is STRAIGHT and pointing at it — which is what a real one does.
        let g = crate::pose::global_transforms(sk, &pose);
        let at = |j: u16| g[j as usize].transform_point3(Vec3::ZERO);
        let (s, e, w) = (at(chain[0]), at(chain[1]), at(chain[2]));
        assert!(
            ((w - s).length() - r.chain_length).abs() < 1.0e-3,
            "the arm is not fully extended"
        );
        assert!(
            (e - s).normalize().dot((far - s).normalize()) > 0.999,
            "the extended arm is not pointing at the target"
        );
        assert!(pose
            .locals
            .iter()
            .all(|l| l.rotation.iter().all(|c| c.is_finite())));
    }

    /// Every refusal is a **value**, and a middle joint that is not a hinge is
    /// not one of them — it delegates.
    #[test]
    fn a_reach_refuses_by_name_and_falls_back_when_there_is_no_hinge() {
        use crate::ik::IkError;
        let asset = manny();
        let sk = &asset.skeleton;
        let chain = arm_chain(sk, asset.role_index(), BoneSide::Right).expect("an arm");
        let mut pose = Pose::rest(sk);
        let before = pose.clone();

        assert!(matches!(
            reach(sk, &mut pose, [chain[0], chain[1], 9999], Vec3::ZERO, &[]),
            Err(IkError::NoSuchJoint { joint: 9999, .. })
        ));
        // Not a parent walk: the hand's grandparent is not its parent.
        assert!(matches!(
            reach(
                sk,
                &mut pose,
                [chain[0], chain[0], chain[2]],
                Vec3::ZERO,
                &[]
            ),
            Err(IkError::NotAChain { .. })
        ));
        assert!(matches!(
            reach(
                sk,
                &mut pose,
                chain,
                Vec3::new(f32::NAN, 0.0, 0.0),
                &asset.limits
            ),
            Err(IkError::NonFinite { .. })
        ));
        assert_eq!(pose.locals, before.locals, "a refusal wrote a pose");

        // No limit at all on the elbow: it delegates to the pole solver and
        // still lands somewhere sensible, rather than refusing.
        let r = reach(sk, &mut pose, chain, Vec3::new(0.25, 1.15, 0.45), &[]).expect("a solve");
        assert!(
            r.reached,
            "the unlimited fallback missed by {}",
            r.reach_error
        );
        // A CONE on the elbow is also not a hinge, and also delegates.
        let coned = [crate::template::JointLimit::cone_only(
            chain[1],
            crate::template::ConeLimit {
                axis: [1.0, 0.0, 0.0],
                swing_deg: 150.0,
                twist_deg: [-10.0, 10.0],
            },
        )];
        let mut pose = Pose::rest(sk);
        assert!(reach(sk, &mut pose, chain, Vec3::new(0.25, 1.15, 0.45), &coned).is_ok());
    }

    /// Two solves, same bits — the claim the determinism trace rests on, for the
    /// solver this wave added.
    #[test]
    fn a_reach_is_bit_identical_across_runs() {
        let asset = manny();
        let sk = &asset.skeleton;
        let chain = arm_chain(sk, asset.role_index(), BoneSide::Left).expect("an arm");
        let run = || {
            let mut pose = Pose::rest(sk);
            reach(
                sk,
                &mut pose,
                chain,
                Vec3::new(-0.2, 1.2, 0.4),
                &asset.limits,
            )
            .expect("solved");
            pose
        };
        let (a, b) = (run(), run());
        for (i, (x, y)) in a.locals.iter().zip(b.locals.iter()).enumerate() {
            for (u, v) in x.rotation.iter().zip(y.rotation.iter()) {
                assert_eq!(u.to_bits(), v.to_bits(), "joint {i} is not bit-stable");
            }
        }
    }

    /// **What a gripping hand costs on the fixed step**, measured rather than
    /// argued — the SK1a audit's own finding about the drive pass, which was
    /// "argued, not measured", not repeated here.
    ///
    /// `hand_of` is the part that could be a problem: it walks the skeleton once
    /// per digit chain and this engine now hands it 161-joint rigs, and it runs
    /// **per gripping hand per posed character per fixed step** because nothing
    /// caches it. So the number is taken and printed, and the assertion is a
    /// ceiling against one 60 Hz frame with its slack named.
    #[test]
    fn a_gripping_hand_is_priced_against_a_frame() {
        use std::time::Instant;
        let asset = manny();
        let sk = &asset.skeleton;
        let roles = asset.role_index();
        let wrist = roles
            .first(BoneRoleKind::Hand, BoneSide::Right)
            .expect("a hand");

        let mut derive_us = f64::MAX;
        let mut curl_us = f64::MAX;
        let mut reach_us = f64::MAX;
        let hand = hand_of(sk, roles, wrist).expect("a hand");
        let grip = GripAffordance {
            name: "g".into(),
            hand: wrist,
            palm: crate::JointTransform::IDENTITY,
            aperture_m: 0.035,
            curl: [1.0; 5],
        };
        let chain = arm_chain(sk, roles, BoneSide::Right).expect("an arm");
        for _ in 0..5 {
            let t = Instant::now();
            let h = hand_of(sk, roles, wrist).expect("a hand");
            derive_us = derive_us.min(t.elapsed().as_secs_f64() * 1.0e6);
            assert_eq!(h.digit_count(), 5);

            let mut pose = Pose::rest(sk);
            let t = Instant::now();
            apply_grip(sk, &mut pose, &hand, &asset.limits, &grip, 1.0);
            curl_us = curl_us.min(t.elapsed().as_secs_f64() * 1.0e6);

            let mut pose = Pose::rest(sk);
            let t = Instant::now();
            reach(
                sk,
                &mut pose,
                chain,
                Vec3::new(0.2, 1.2, 0.4),
                &asset.limits,
            )
            .expect("solved");
            reach_us = reach_us.min(t.elapsed().as_secs_f64() * 1.0e6);
        }
        println!(
            "one gripping hand on a 161-bone rig: derive {derive_us:.1} us, curl {curl_us:.1} us, \
             arm reach {reach_us:.1} us — against a 16 667 us frame"
        );
        // A ceiling, not a performance claim: a debug CI runner is an order of
        // magnitude slower than this, and 1 ms is still a sixteenth of a frame
        // for ONE hand. It is a tripwire against an accidental O(J^2).
        for (what, us) in [
            ("the hand derivation", derive_us),
            ("the curl", curl_us),
            ("the arm reach", reach_us),
        ] {
            assert!(
                us < 1_000.0,
                "{what} took {us} us per hand per step, which is not a fixed-step cost"
            );
        }
    }
}
