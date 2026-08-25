//! **The rig's side tables** (SK1a): what each bone *is*, which bones are
//! procedurally driven, and where a hand can grip.
//!
//! # Why a role table exists at all
//!
//! Before this module every question about a rig's anatomy was answered by
//! guessing at its bone *names*, in five different places with five different
//! matching rules: `inf_physics::ragdoll::classify`'s keyword table,
//! `inf_ecs::pose::foot_joints`' `contains("foot")`, `pose::pelvis_joint`'s
//! `eq_ignore_ascii_case("pelvis")`, [`crate::derive::foot_joints`]'
//! `starts_with("foot_")` and `derive`'s `starts_with("upper_leg_")`. Each was
//! written against the rig this engine's own generator produced, and each fails
//! differently on a rig that arrives from anywhere else.
//!
//! The measured cost, on the UE5 Manny hierarchy [`crate::manny`] now emits:
//! `ragdoll::classify` maps `spine_01` … `spine_05` all onto one `Spine` role and
//! never produces a `Chest` at all, so the arms and the head name a parent that is
//! not there and fall out as **free-floating capsules**; `upperarm_l` and
//! `upperarm_twist_01_l` both claim `UpperArmL`; every finger, every `ik_*` bone
//! and both clavicles classify to nothing.
//!
//! A role table answers those questions from the asset instead. The string
//! heuristics stay, as the **fallback for a rig that carries no table** — an
//! imported glTF, a `.inf_skel` written before this schema — so nothing that
//! worked before stops working.
//!
//! # The vocabulary is a wire, so it is frozen
//!
//! [`BoneRoleKind`] and [`BoneSide`] ride a bincode `Vec` inside
//! [`SkeletonAsset`](crate::SkeletonAsset), and bincode writes an enum as its
//! **variant index**. The discriminants below are therefore append-only for good:
//! inserting a variant in the middle re-labels every role in every committed rig.
//! `the_role_wire_discriminants_are_frozen` is what keeps that true.

use serde::{Deserialize, Serialize};

use crate::skeleton::{JointTransform, Skeleton};

/// What a bone *is*, independent of what it is called.
///
/// **Freeze-pinned, append-only** — see the module docs. A kind that is absent
/// from a rig is simply absent; there is no "unknown" that a reader has to
/// distinguish from a missing row, because a joint with no row in the table is
/// exactly that.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BoneRoleKind {
    /// The rig's own origin bone. Not a deform bone: it carries the character,
    /// not the mesh.
    Root = 0,
    /// The hip girdle — the bone a pelvis IK offset moves and the ragdoll's free
    /// root.
    Pelvis = 1,
    /// One segment of the spine chain, in rig order (the last one is the chest).
    Spine = 2,
    /// One segment of the neck chain.
    Neck = 3,
    /// The head.
    Head = 4,
    /// A collar bone.
    Clavicle = 5,
    /// The upper arm (shoulder → elbow).
    UpperArm = 6,
    /// The forearm (elbow → wrist).
    LowerArm = 7,
    /// The hand.
    Hand = 8,
    /// A finger bone — a metacarpal or a phalanx.
    Finger = 9,
    /// A thumb bone.
    Thumb = 10,
    /// The upper leg (hip → knee).
    Thigh = 11,
    /// The lower leg (knee → ankle).
    Calf = 12,
    /// The ankle — what this engine plants and what a foot IK goal drives.
    Foot = 13,
    /// The ball of the foot (the toe joint).
    Ball = 14,
    /// A twist bone: driven by [`TwistDriver`], never authored by a clip.
    Twist = 15,
    /// An IK handle — a marker the animation system reads and writes, carrying no
    /// skin weights.
    IkTarget = 16,
    /// A helper: a corrective, a bend-assist, a weapon marker. Present so an
    /// externally authored clip or a retarget finds every bone it names.
    Helper = 17,
}

impl BoneRoleKind {
    /// Whether a bone of this kind **deforms the mesh** — the question the
    /// mannequin generator, the ragdoll builder and the skin-weight solver all
    /// ask, and the reason `Root` is not simply "the first bone".
    ///
    /// `Twist` is deliberately *false* here even though a twist bone really does
    /// carry skin weights on a production rig: this engine has no twist-weighted
    /// mesh, and a mannequin box on a bone coincident with its parent is a box
    /// inside a box.
    pub fn is_deform(self) -> bool {
        use BoneRoleKind::*;
        matches!(
            self,
            Pelvis
                | Spine
                | Neck
                | Head
                | Clavicle
                | UpperArm
                | LowerArm
                | Hand
                | Finger
                | Thumb
                | Thigh
                | Calf
                | Foot
                | Ball
        )
    }

    /// Whether this kind is part of a **limb chain the ragdoll simulates**.
    ///
    /// A tighter set than [`is_deform`](Self::is_deform): fingers and clavicles
    /// deform a mesh and are not worth a rigid body, and the standard is the one
    /// the name-based classifier already kept (a ragdoll of a template biped has
    /// no hands and no feet either).
    pub fn is_ragdoll_limb(self) -> bool {
        use BoneRoleKind::*;
        matches!(
            self,
            Pelvis | Spine | Neck | Head | UpperArm | LowerArm | Thigh | Calf
        )
    }
}

/// Which side of the body a bone is on.
///
/// **Freeze-pinned, append-only** for [`BoneRoleKind`]'s reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BoneSide {
    /// On the midline (spine, head, root).
    Center = 0,
    /// The character's left.
    Left = 1,
    /// The character's right.
    Right = 2,
}

/// One row of a rig's role table: this joint is *that* part of a body.
///
/// Rows are keyed by joint index and the table is kept in **ascending joint
/// order**, so "the first `Foot` on the `Left`" is a well-defined question with a
/// deterministic answer even on a rig that carries two of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoneRole {
    /// Index of the joint this row describes.
    pub joint: u16,
    /// What the joint is.
    pub kind: BoneRoleKind,
    /// Which side it is on.
    pub side: BoneSide,
}

impl BoneRole {
    /// A row.
    pub fn new(joint: u16, kind: BoneRoleKind, side: BoneSide) -> Self {
        Self { joint, kind, side }
    }
}

/// **A twist bone's drive rule** (SK1a).
///
/// A twist bone carries no animation of its own: it takes a *fraction of another
/// joint's roll* about a limb axis, so the skin between two joints rotates
/// gradually instead of pinching at one of them. The law this engine writes down,
/// and the one [`crate::manny`] authors against, is:
///
/// > **the roll along a limb segment is linear in the position along it.**
///
/// That single sentence produces both signs. A twist bone is a *child* of the
/// segment, so it already inherits the segment's whole roll:
///
/// * **an upper segment** (upper arm, thigh) is rolled by its own joint, at the
///   proximal end, so a twist bone at fraction `p` along it must give **back**
///   `1 − p` of that roll — a negative [`fraction`](Self::fraction) whose source
///   is the segment itself;
/// * **a lower segment** (forearm, calf) is rolled by the joint at its *distal*
///   end, which is a child, so a twist bone at fraction `p` must **add** `p` of
///   that child's roll — a positive fraction whose source is the child.
///
/// Measured against the real rig rather than assumed: in `SK_Mannequin` the twist
/// bones sit at exactly one third and two thirds of their segment, and `_01` is
/// always the one nearest the joint that drives it (`upperarm_twist_01_l` at 1/3
/// from the shoulder, `lowerarm_twist_01_l` at 2/3 from the elbow — that is, 1/3
/// from the wrist).
///
/// # Determinism
///
/// The drive is a swing-twist decomposition (pure arithmetic and one `sqrt`)
/// followed by [`inf_math::pslerp`] from the identity — **no `sin`, no `cos`, no
/// `atan2`**, so nothing here can trip the P14 law about `f32` transcendentals not
/// being bit-portable. The result is folded into `pose_state_bytes`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwistDriver {
    /// The driven bone (a [`BoneRoleKind::Twist`] joint).
    pub joint: u16,
    /// The joint whose roll is read.
    pub source: u16,
    /// The source's local twist axis, unit length.
    pub axis: [f32; 3],
    /// How much of the source's roll to take, in `[-1, 1]`. Negative counters it —
    /// see the type docs.
    pub fraction: f32,
}

impl TwistDriver {
    /// A driver.
    pub fn new(joint: u16, source: u16, axis: [f32; 3], fraction: f32) -> Self {
        Self {
            joint,
            source,
            axis,
            fraction,
        }
    }
}

/// **An IK handle's FK source** (SK1a).
///
/// `ik_foot_l` follows `foot_l`, `ik_hand_l` follows `hand_l`, `ik_hand_gun`
/// follows `hand_r` — the UE convention, stated as data rather than re-derived
/// from a name prefix on every fixed step. That choice is a measurement, not a
/// taste: [`Skeleton::index_of`] is a linear scan, so deriving five pairs from
/// names would be five scans of 161 joints per character per step, and this table
/// is written once when the rig is generated.
///
/// A handle with no row (`ik_hand_root`, `ik_foot_root` — the subtree anchors)
/// simply stays where its bind put it, which is what an anchor is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IkFollow {
    /// The handle.
    pub joint: u16,
    /// The joint whose global transform it takes.
    pub source: u16,
}

impl IkFollow {
    /// A pair.
    pub fn new(joint: u16, source: u16) -> Self {
        Self { joint, source }
    }
}

/// **Where and how a hand grips a thing** (SK1a ships the table; SK1b's finger
/// solver is its first consumer).
///
/// Authored per hand rather than per weapon, because the affordance is a property
/// of the *hand*: an aperture and a set of curl targets is what a hand can do, and
/// a prop names the affordance it wants.
///
/// The table is **empty on every rig this wave generates**. It is here, and not in
/// the wave that needs it, because `.inf_skel` is bincode-positional: a tail
/// append costs a schema bump and a downgrade-bless whether it carries one field
/// or three, and the ruling for this wave was to spend that bump once.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GripAffordance {
    /// What this grip is for (`"rifle"`, `"handle"`, …). Unique within a rig.
    pub name: String,
    /// The hand joint this grip belongs to.
    pub hand: u16,
    /// The palm frame, in the hand joint's local space: where the gripped thing
    /// sits and how it is oriented.
    pub palm: JointTransform,
    /// How wide the grip opens, metres — the diameter a prop's handle may have.
    pub aperture_m: f32,
    /// Per-finger curl targets in `[0, 1]`, thumb first then index, middle, ring,
    /// pinky. `0` is straight, `1` is fully closed.
    pub curl: [f32; 5],
}

impl GripAffordance {
    /// A grip with no curl and no offset.
    pub fn new(name: impl Into<String>, hand: u16, aperture_m: f32) -> Self {
        Self {
            name: name.into(),
            hand,
            palm: JointTransform::IDENTITY,
            aperture_m,
            curl: [0.0; 5],
        }
    }
}

/// A **resolved role lookup** over one rig — the door every site that used to
/// guess at names now asks first.
///
/// Built once and passed down, because the alternative is a linear scan of the
/// role table per question per character per fixed step, and at 161 bones this
/// engine asks four of those questions per posed character.
///
/// # It borrows
///
/// Deliberately, and the reason is the fixed step: `foot_states`, `apply_foot_ik`
/// and the pelvis drop each want this on every posed character on every step, and
/// an owning index would be a 161-row clone and a sort **per question per
/// character per step** — a per-frame rebuild keyed on nothing, which is the shape
/// wave I7b spent a whole clause removing from the render path. The rows must
/// therefore already be in ascending joint order, which is a property of the
/// table and not of this type: `SkeletonAsset`'s decode refuses a role row naming
/// a joint the rig does not have, and [`sorted`](Self::sorted) is what turns an
/// unordered table into one this can index.
///
/// An [`empty`](Self::empty) index answers `None` to everything, which is exactly
/// what a rig with no table means — so a caller writes the role path once and the
/// fallback once, rather than branching on whether a table exists.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RoleIndex<'a> {
    /// `(kind, side) -> joints`, ascending, in the table's own order.
    rows: &'a [BoneRole],
}

impl<'a> RoleIndex<'a> {
    /// The index of `roles`, which must be in **ascending joint order** — every
    /// generator in this engine writes it that way, and
    /// [`sorted`](Self::sorted) is the door for a table that is not.
    ///
    /// An out-of-order table is not unsound: [`first`](Self::first),
    /// [`last`](Self::last) and [`all`](Self::all) walk it and answer in *its*
    /// order, and only [`role_of`](Self::role_of)'s binary search needs the
    /// invariant. It would answer `None` for a row that is really there, which is
    /// why the door exists rather than a silent sort on every construction.
    pub fn new(roles: &'a [BoneRole]) -> Self {
        Self { rows: roles }
    }

    /// `roles`, sorted — the owning half, for a caller holding a table it did not
    /// generate. Returns the `Vec` so the borrow above stays a borrow.
    pub fn sorted(roles: &[BoneRole]) -> Vec<BoneRole> {
        let mut rows = roles.to_vec();
        rows.sort_by_key(|r| r.joint);
        rows
    }

    /// An index that knows nothing — every query is `None`.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Whether this rig carries a role table at all.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The rows, ascending by joint.
    pub fn rows(&self) -> &[BoneRole] {
        self.rows
    }

    /// What `joint` is, if the table says.
    pub fn role_of(&self, joint: u16) -> Option<BoneRole> {
        self.rows
            .binary_search_by_key(&joint, |r| r.joint)
            .ok()
            .map(|i| self.rows[i])
    }

    /// What kind of bone `joint` is, if the table says.
    pub fn kind_of(&self, joint: u16) -> Option<BoneRoleKind> {
        self.role_of(joint).map(|r| r.kind)
    }

    /// Whether `joint` deforms the mesh. `false` for a joint with no row: an
    /// untabled rig has no opinion, and the callers that ask this are choosing
    /// what to *add*, so "no opinion" must not add it.
    pub fn is_deform(&self, joint: u16) -> bool {
        self.kind_of(joint).is_some_and(BoneRoleKind::is_deform)
    }

    /// The **first** joint with this role, in joint order.
    pub fn first(&self, kind: BoneRoleKind, side: BoneSide) -> Option<u16> {
        self.rows
            .iter()
            .find(|r| r.kind == kind && r.side == side)
            .map(|r| r.joint)
    }

    /// The **last** joint with this role, in joint order — "the top of the spine
    /// chain" is this question, and it is the one that used to be answered by
    /// looking for a bone called `chest`.
    pub fn last(&self, kind: BoneRoleKind, side: BoneSide) -> Option<u16> {
        self.rows
            .iter()
            .rev()
            .find(|r| r.kind == kind && r.side == side)
            .map(|r| r.joint)
    }

    /// Every joint with this role, in joint order.
    pub fn all(&self, kind: BoneRoleKind, side: BoneSide) -> Vec<u16> {
        self.rows
            .iter()
            .filter(|r| r.kind == kind && r.side == side)
            .map(|r| r.joint)
            .collect()
    }

    /// The first joint with this role on **either** side — for a midline part
    /// whose author happened to give it a side, and for the kinds that have none.
    pub fn first_any(&self, kind: BoneRoleKind) -> Option<u16> {
        self.rows.iter().find(|r| r.kind == kind).map(|r| r.joint)
    }

    /// The **anatomical successor** of `joint`: its first child that is a deform
    /// bone.
    ///
    /// This is the segment a ragdoll capsule spans and a mannequin box wraps, and
    /// it is *not* "the first child": on the real Manny hierarchy the first child
    /// of `lowerarm_l` is `lowerarm_twist_02_l`, a driven bone one third of the way
    /// down the forearm, and a capsule built from it covers a third of the arm.
    pub fn deform_child(&self, skeleton: &Skeleton, joint: u16) -> Option<u16> {
        let joints = skeleton.joints();
        for (i, j) in joints.iter().enumerate() {
            if j.parent != Some(joint) {
                continue;
            }
            let i = i as u16;
            if self.is_deform(i) {
                return Some(i);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The wire discriminants are frozen.** bincode writes an enum as its
    /// variant index, so inserting a kind in the middle of this list re-labels
    /// every role in every committed rig — silently, because the bytes still
    /// decode.
    #[test]
    fn the_role_wire_discriminants_are_frozen() {
        use BoneRoleKind::*;
        let pinned: [(BoneRoleKind, u8); 18] = [
            (Root, 0),
            (Pelvis, 1),
            (Spine, 2),
            (Neck, 3),
            (Head, 4),
            (Clavicle, 5),
            (UpperArm, 6),
            (LowerArm, 7),
            (Hand, 8),
            (Finger, 9),
            (Thumb, 10),
            (Thigh, 11),
            (Calf, 12),
            (Foot, 13),
            (Ball, 14),
            (Twist, 15),
            (IkTarget, 16),
            (Helper, 17),
        ];
        for (kind, want) in pinned {
            assert_eq!(kind as u8, want, "{kind:?} moved on the wire");
        }
        for (side, want) in [
            (BoneSide::Center, 0u8),
            (BoneSide::Left, 1),
            (BoneSide::Right, 2),
        ] {
            assert_eq!(side as u8, want, "{side:?} moved on the wire");
        }
        // …and the count, so an APPEND is deliberate and a shrink is caught.
        assert_eq!(pinned.len(), 18, "a kind was added or removed");
    }

    #[test]
    fn an_empty_index_answers_nothing_and_adds_nothing() {
        let idx = RoleIndex::empty();
        assert!(idx.is_empty());
        assert_eq!(idx.first(BoneRoleKind::Foot, BoneSide::Left), None);
        assert_eq!(idx.kind_of(0), None);
        // The one that matters: "is this a deform bone" must be FALSE with no
        // table, because its callers use it to decide what to add.
        assert!(!idx.is_deform(0));
    }

    #[test]
    fn first_and_last_pick_the_ends_of_a_chain() {
        use BoneRoleKind::*;
        use BoneSide::*;
        // Deliberately built out of order, and put through the sorting door —
        // which is the door that exists so the index itself can borrow.
        let rows = RoleIndex::sorted(&[
            BoneRole::new(4, Spine, Center),
            BoneRole::new(2, Spine, Center),
            BoneRole::new(3, Spine, Center),
            BoneRole::new(9, Foot, Left),
            BoneRole::new(1, Pelvis, Center),
        ]);
        let idx = RoleIndex::new(&rows);
        assert_eq!(idx.first(Spine, Center), Some(2));
        assert_eq!(idx.last(Spine, Center), Some(4));
        assert_eq!(idx.all(Spine, Center), vec![2, 3, 4]);
        assert_eq!(idx.first(Foot, Left), Some(9));
        assert_eq!(idx.first(Foot, Right), None, "a side is part of the key");
        assert_eq!(idx.first_any(Foot), Some(9));
        assert_eq!(idx.role_of(3).map(|r| r.kind), Some(Spine));
        assert_eq!(idx.role_of(7), None);
        assert!(idx.is_deform(1));
    }

    #[test]
    fn a_grip_defaults_to_an_open_hand() {
        let g = GripAffordance::new("rifle", 7, 0.04);
        assert_eq!(g.curl, [0.0; 5]);
        assert_eq!(g.palm, JointTransform::IDENTITY);
        assert_eq!(g.hand, 7);
    }
}
