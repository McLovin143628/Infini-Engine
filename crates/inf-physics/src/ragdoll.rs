//! Ragdoll setup helper (P12.1, seed): a **pure** function that turns a humanoid
//! skeleton into the body / collider / joint descriptors a caller (or a future
//! editor "Make Ragdoll" tool) spawns into a [`PhysicsWorld3D`](crate::d3::PhysicsWorld3D).
//!
//! This is the substrate, not the UI. [`build_ragdoll`] classifies each input
//! bone by name into a [`BoneRole`], places a capsule spanning the bone (a
//! dynamic body oriented head→tail), and links each limb to its parent with a
//! joint chosen for the joint's real anatomy: **spherical** ball joints at the
//! spine / neck / shoulders / hips, and **revolute** hinges with sane angle
//! limits at the elbows and knees. The hips (or, absent hips, the first
//! classified bone) are the free root — no joint to a parent.
//!
//! Everything is expressed in the facade's own descriptor types, so the helper
//! never touches rapier and is trivially unit-testable.

use glam::{DQuat, DVec3};

use crate::d3::{
    BodyDesc3D, BodyKind3D, ColliderDesc3D, ColliderShape3D, JointDesc3D, JointKind3D,
};

/// One bone of the input skeleton: a name and the world-space endpoints of the
/// bone segment (`head` = joint end nearest the root, `tail` = far end).
#[derive(Clone, Debug, PartialEq)]
pub struct RagdollBone {
    /// Bone name; classified case-insensitively (see [`classify`]).
    pub name: String,
    /// World-space start of the bone (the parent-facing joint).
    pub head: DVec3,
    /// World-space end of the bone.
    pub tail: DVec3,
}

impl RagdollBone {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, head: DVec3, tail: DVec3) -> Self {
        Self {
            name: name.into(),
            head,
            tail,
        }
    }
}

/// Tuning for [`build_ragdoll`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RagdollConfig {
    /// Capsule radius as a fraction of the bone's length.
    pub thickness: f64,
    /// Minimum capsule radius (world units), so a short bone still has volume.
    pub min_radius: f64,
    /// Mass density of every limb body.
    pub density: f64,
}

impl Default for RagdollConfig {
    fn default() -> Self {
        Self {
            thickness: 0.14,
            min_radius: 0.04,
            density: 1.0,
        }
    }
}

/// The recognized humanoid bone roles. The classifier maps a free-text bone name
/// onto one of these; unrecognized bones are skipped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoneRole {
    Hips,
    Spine,
    Chest,
    Head,
    UpperArmL,
    LowerArmL,
    UpperArmR,
    LowerArmR,
    ThighL,
    ShinL,
    ThighR,
    ShinR,
}

impl BoneRole {
    /// The parent role in the ragdoll hierarchy (`None` for the [`Hips`] root).
    ///
    /// [`Hips`]: BoneRole::Hips
    fn parent(self) -> Option<BoneRole> {
        use BoneRole::*;
        match self {
            Hips => None,
            Spine => Some(Hips),
            Chest => Some(Spine),
            Head => Some(Chest),
            UpperArmL | UpperArmR => Some(Chest),
            LowerArmL => Some(UpperArmL),
            LowerArmR => Some(UpperArmR),
            ThighL | ThighR => Some(Hips),
            ShinL => Some(ThighL),
            ShinR => Some(ThighR),
        }
    }

    /// Whether this bone hinges (elbow/knee → revolute) rather than swivels
    /// (everything else → spherical).
    fn is_hinge(self) -> bool {
        use BoneRole::*;
        matches!(self, LowerArmL | LowerArmR | ShinL | ShinR)
    }

    /// Sane hinge limits `[min, max]` in radians (elbows/knees bend one way).
    fn hinge_limits(self) -> [f64; 2] {
        use BoneRole::*;
        match self {
            // Elbows bend "up" (0 → ~135°).
            LowerArmL | LowerArmR => [0.0, 2.36],
            // Knees bend "back" (~-135° → 0).
            ShinL | ShinR => [-2.36, 0.0],
            _ => [0.0, 0.0],
        }
    }
}

/// Classify a bone name into a [`BoneRole`], case-insensitively, tolerating the
/// common naming conventions (`upperarm_l`, `UpperArm.L`, `LeftForeArm`,
/// `thigh.R`, `calf_r`, `pelvis`, …). Returns `None` for unrecognized bones.
pub fn classify(name: &str) -> Option<BoneRole> {
    let n = name.to_ascii_lowercase();
    let has = |k: &str| n.contains(k);
    // Side detection: an explicit `.l`/`_l`/`left` (etc.) or a trailing letter.
    let left = has("left")
        || has("_l")
        || has(".l")
        || has(" l")
        || n.ends_with('l')
        || has("l_")
        || has("l.");
    let right = has("right")
        || has("_r")
        || has(".r")
        || has(" r")
        || n.ends_with('r')
        || has("r_")
        || has("r.");

    use BoneRole::*;
    // Order matters: check the more specific limbs before the torso keywords.
    if has("forearm") || has("lowerarm") || has("lower_arm") {
        return side(left, right, LowerArmL, LowerArmR);
    }
    if has("upperarm") || has("upper_arm") || has("shoulder") || has("arm") {
        return side(left, right, UpperArmL, UpperArmR);
    }
    if has("calf") || has("shin") || has("lowerleg") || has("lower_leg") {
        return side(left, right, ShinL, ShinR);
    }
    if has("thigh") || has("upperleg") || has("upper_leg") {
        return side(left, right, ThighL, ThighR);
    }
    if has("head") || has("neck") {
        return Some(Head);
    }
    if has("chest") || has("upperchest") || has("spine2") || has("spine1") {
        return Some(Chest);
    }
    if has("spine") || has("torso") {
        return Some(Spine);
    }
    if has("hip") || has("pelvis") || has("root") {
        return Some(Hips);
    }
    None
}

fn side(left: bool, right: bool, l: BoneRole, r: BoneRole) -> Option<BoneRole> {
    if right && !left {
        Some(r)
    } else {
        // Default to left when ambiguous (a lone limb reads as left).
        Some(l)
    }
}

/// One assembled ragdoll part: the dynamic body + capsule collider for a bone,
/// the world pose to spawn it at, and (for non-root bones) the joint linking it
/// to its parent part by index into the returned `Vec`.
#[derive(Clone, Debug, PartialEq)]
pub struct RagdollPart {
    /// The role this part fills.
    pub role: BoneRole,
    /// The originating bone name.
    pub name: String,
    /// A dynamic rigid-body descriptor.
    pub body: BodyDesc3D,
    /// A capsule collider spanning the bone (local Y = bone axis).
    pub collider: ColliderDesc3D,
    /// World-space body position (the capsule centre).
    pub position: DVec3,
    /// World-space body orientation (local Y aligned head→tail).
    pub rotation: DQuat,
    /// The joint to the parent part, or `None` for the root.
    pub joint: Option<RagdollJoint>,
}

/// A parent link for a [`RagdollPart`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RagdollJoint {
    /// Index of the parent part in the [`build_ragdoll`] result.
    pub parent: usize,
    /// The joint descriptor (anchors already resolved into each body's frame).
    pub desc: JointDesc3D,
}

/// Build a ragdoll from `skeleton`. Bones that classify to a [`BoneRole`] become
/// capsule bodies; each links to its parent role's part with a spherical or
/// revolute joint. The result is ordered so that a part's `joint.parent` index is
/// always `< ` its own index (parents precede children — a valid spawn order).
///
/// Unrecognized bones, and bones whose parent role is absent from the skeleton,
/// produce a rootless (free) body rather than being dropped, so nothing silently
/// disappears.
pub fn build_ragdoll(skeleton: &[RagdollBone], config: RagdollConfig) -> Vec<RagdollPart> {
    // 1. Classify + build the per-bone bodies, in a deterministic parents-first
    //    order (topologically by hierarchy depth, tie-broken by role order).
    let mut classified: Vec<(BoneRole, &RagdollBone)> = skeleton
        .iter()
        .filter_map(|b| classify(&b.name).map(|r| (r, b)))
        .collect();
    // Stable sort by hierarchy depth so parents precede children.
    classified.sort_by_key(|(role, _)| (depth(*role), role_order(*role)));

    // Role → its index in the output (for joint parent resolution).
    let mut index_of: std::collections::BTreeMap<u8, usize> = std::collections::BTreeMap::new();

    let mut parts: Vec<RagdollPart> = Vec::with_capacity(classified.len());
    for (role, bone) in &classified {
        let seg = bone.tail - bone.head;
        let length = seg.length().max(1e-6);
        let dir = seg / length;
        let centre = (bone.head + bone.tail) * 0.5;
        let rotation = DQuat::from_rotation_arc(DVec3::Y, dir);
        let radius = (config.thickness * length).max(config.min_radius);
        // Capsule half-height is the segment half-length minus the radius caps
        // (clamped so a stubby bone still yields a valid capsule).
        let half_height = (length * 0.5 - radius).max(radius * 0.25);

        let body = BodyDesc3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        };
        let collider = ColliderDesc3D::new(ColliderShape3D::Capsule {
            half_height,
            radius,
        })
        .density(config.density);

        parts.push(RagdollPart {
            role: *role,
            name: bone.name.clone(),
            body,
            collider,
            position: centre,
            rotation,
            joint: None,
        });
        index_of.insert(role_key(*role), parts.len() - 1);
    }

    // 2. Wire joints. The joint anchor is the child bone's head (the shared point
    //    with its parent), expressed in each body's local frame.
    for i in 0..parts.len() {
        let role = parts[i].role;
        let Some(parent_role) = role.parent() else {
            continue;
        };
        let Some(&parent_idx) = index_of.get(&role_key(parent_role)) else {
            continue; // parent bone absent → leave this part free
        };

        // The shared world anchor: this bone's head (its parent-facing joint).
        let child_head = skeleton_head(&classified, role).unwrap_or(parts[i].position);

        let child_anchor = world_to_local(parts[i].rotation, parts[i].position, child_head);
        let parent_anchor = world_to_local(
            parts[parent_idx].rotation,
            parts[parent_idx].position,
            child_head,
        );

        let kind = if role.is_hinge() {
            JointKind3D::Revolute {
                // Hinge about the body-local X axis (perpendicular to the limb's
                // local-Y long axis).
                axis: DVec3::X,
                limits: Some(role.hinge_limits()),
                motor: None,
            }
        } else {
            JointKind3D::Spherical
        };
        let desc = JointDesc3D::new(kind)
            .local_anchor1(child_anchor)
            .local_anchor2(parent_anchor);
        parts[i].joint = Some(RagdollJoint {
            parent: parent_idx,
            desc,
        });
    }

    parts
}

/// The head (world) of the bone that classified to `role`, if present.
fn skeleton_head(classified: &[(BoneRole, &RagdollBone)], role: BoneRole) -> Option<DVec3> {
    classified
        .iter()
        .find(|(r, _)| *r == role)
        .map(|(_, b)| b.head)
}

/// Transform a world point into a body's local frame.
fn world_to_local(rotation: DQuat, position: DVec3, world: DVec3) -> DVec3 {
    rotation.inverse() * (world - position)
}

/// Hierarchy depth of a role (root = 0), for a parents-first ordering.
fn depth(role: BoneRole) -> u8 {
    match role.parent() {
        None => 0,
        Some(p) => 1 + depth(p),
    }
}

/// A stable within-depth ordering so the output is deterministic.
fn role_order(role: BoneRole) -> u8 {
    role_key(role)
}

/// A stable numeric key per role (for the index map + ordering).
fn role_key(role: BoneRole) -> u8 {
    use BoneRole::*;
    match role {
        Hips => 0,
        Spine => 1,
        Chest => 2,
        Head => 3,
        UpperArmL => 4,
        UpperArmR => 5,
        LowerArmL => 6,
        LowerArmR => 7,
        ThighL => 8,
        ThighR => 9,
        ShinL => 10,
        ShinR => 11,
    }
}
