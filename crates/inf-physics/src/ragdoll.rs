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
    /// Bone name; classified case-insensitively (see [`classify`]) when this bone
    /// carries no [`role`](Self::role).
    pub name: String,
    /// World-space start of the bone (the parent-facing joint).
    pub head: DVec3,
    /// World-space end of the bone.
    pub tail: DVec3,
    /// Index of this bone's parent **in the same slice** (SK1a), or `None` for the
    /// rig's root. The slice is one entry per joint in joint order.
    pub parent: Option<u16>,
    /// What the rig says this bone **is** (SK1a). `None` on every bone of a rig
    /// that carries no role table, which is what routes [`build_ragdoll`] to its
    /// name classifier.
    pub role: Option<inf_anim::BoneRole>,
}

impl RagdollBone {
    /// A bone with no hierarchy and no role — the shape every caller built before
    /// SK1a, and the one the name classifier reads.
    pub fn new(name: impl Into<String>, head: DVec3, tail: DVec3) -> Self {
        Self {
            name: name.into(),
            head,
            tail,
            parent: None,
            role: None,
        }
    }

    /// This bone with its place in the rig and what the rig says it is.
    pub fn with_role(mut self, parent: Option<u16>, role: Option<inf_anim::BoneRole>) -> Self {
        self.parent = parent;
        self.role = role;
        self
    }
}

/// Tuning for [`build_ragdoll`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RagdollConfig {
    /// Capsule radius as a fraction of the bone's length.
    pub thickness: f64,
    /// Minimum capsule radius (world units), so a short bone still has volume.
    pub min_radius: f64,
    /// Mass density of every limb body, kg/m³.
    ///
    /// # THE PLACEHOLDER, FOR THE FOURTH TIME (P29.6)
    ///
    /// This was **1.0** — rapier's "no opinion" value, one kilogram per cubic
    /// metre, lighter than air — from P12.1 until P29.6, which is seventeen
    /// sub-phases in which the builder had no runtime consumer to notice. The
    /// same defect P20.2 found on `Buoyancy`, P22.3 found on `Destructible` and
    /// P22.4 found again on a 0.4 m wheel that weighed 268 grams.
    ///
    /// What it cost here: a limb capsule is a few litres, so at 1 kg/m³ a thigh
    /// weighed about **six grams**. The P29.6 course lands a character at
    /// 10.9 m/s — hard enough for the classifier to call a ragdoll — and the
    /// gram-weight limbs, spawned touching the floor and carrying the whole
    /// impact velocity, were fired **44 metres into the air** and half a
    /// kilometre downrange. Nothing had seen it because the only fixtures that
    /// spawned a ragdoll before this wave spawned it from rest, and because the
    /// rig published its bones half a capsule ABOVE the floor until P29.6's
    /// character-space ruling put them where the character actually is.
    ///
    /// **985** is human tissue — very slightly denser than water, which is why
    /// people float with their lungs full and sink with them empty. A 6-litre
    /// thigh is about 6 kg at it.
    pub density: f64,
}

impl Default for RagdollConfig {
    fn default() -> Self {
        Self {
            thickness: 0.14,
            min_radius: 0.04,
            density: 985.0,
        }
    }
}

/// The collision-layer bit a ragdoll's limbs belong to, and the only one they do
/// **not** collide with (P29.6).
///
/// A ragdoll's limb capsules overlap **by construction** — adjacent bones share
/// an endpoint, and a collapsed pose folds a forearm into a chest — so limbs that
/// push each other apart are a permanent depenetration force inside the body.
/// Turning contacts off between *jointed* pairs (`JointDesc3D::without_contacts`)
/// deals with the adjacent ones and not with the rest: measured on the P29.6
/// course, a settled ragdoll's pelvis climbed **14 cm per fixed step with a
/// velocity of four centimetres per second** — a position correction, not a
/// motion — and rose ten metres.
///
/// The mask is the standard first answer: limbs are members of one bit and filter
/// everything except that bit, so they collide with the world and with nothing of
/// their own kind. Every other collider in the engine is a member of **all**
/// layers ([`crate::CollisionLayers::default`]), so a ragdoll still lands on a floor,
/// hits a crate and is swept by a camera.
///
/// **The bound**, stated rather than discovered: two *different* ragdolls pass
/// through each other. Separating them needs a bit per ragdoll and there are
/// thirty-two, so the honest version is a per-body group id rather than a wider
/// mask — the follow-up, and not this wave's.
pub const RAGDOLL_LAYER_BIT: u32 = 1 << 31;

/// The layers every ragdoll limb carries — see [`RAGDOLL_LAYER_BIT`].
pub fn ragdoll_layers() -> crate::CollisionLayers {
    crate::CollisionLayers::new(RAGDOLL_LAYER_BIT, !RAGDOLL_LAYER_BIT)
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
    // **The role table first** (SK1a). A rig that says what its bones ARE is
    // assembled from what it says, and the hierarchy comes with it — which is the
    // whole difference, because the classifier below cannot express a body whose
    // parts do not map one-to-one onto twelve fixed roles. Measured on the
    // mannequin: five spine segments all classify to one `Spine`, no `Chest` is
    // ever produced, and the arms and the head therefore name a parent that is not
    // there and spawn as free bodies.
    if skeleton.iter().any(|b| b.role.is_some()) {
        return build_from_roles(skeleton, config);
    }
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
        parts.push(capsule_part(*role, bone, config));
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
        // **`local_anchor1` is BODY 1's, and body 1 is the PARENT.**
        // `spawn` calls `add_joint(parent, child, desc)`, so the parent's own
        // local anchor has to go in slot 1. These two were the other way round
        // from P12.1 until P29.6, which nothing could see: `inf_physics::ragdoll`
        // was a pure builder with no runtime consumer for seven phases, and
        // P29.4's fixtures all spawn a ragdoll AT REST on flat ground, where a
        // mis-anchored joint still settles into a heap — just the wrong heap.
        //
        // What it cost: a ragdoll spawned with real velocity had every joint
        // yanking two bodies toward two different points, and the solver put
        // 7 m/s of sideways velocity into a pelvis in ONE step. The P29.6 course
        // lands a character at 10.7 m/s, which is the first time in this
        // repository's life that a ragdoll has been given one.
        let desc = JointDesc3D::new(kind)
            .without_contacts()
            .local_anchor1(parent_anchor)
            .local_anchor2(child_anchor);
        parts[i].joint = Some(RagdollJoint {
            parent: parent_idx,
            desc,
        });
    }

    parts
}

/// **Build a ragdoll from the rig's own role table** (SK1a).
///
/// The parts are the bones the table calls limb bones
/// ([`inf_anim::BoneRoleKind::is_ragdoll_limb`]); each one's ragdoll parent is
/// the nearest such ancestor **by index**, not by role, which is the difference
/// that makes a five-segment spine or a second neck bone a body rather than a
/// collision of labels. Everything else — twist bones, fingers, clavicles, IK
/// handles, correctives — is simply not a part, and folds into whichever capsule
/// spans the segment it lives on.
///
/// The result is **connected by construction**: the input is in joint order, so a
/// part's parent always precedes it, and the only part with no joint is the one
/// whose ancestor chain reaches the rig's root without meeting another part.
fn build_from_roles(skeleton: &[RagdollBone], config: RagdollConfig) -> Vec<RagdollPart> {
    let is_part = |b: &RagdollBone| -> bool { b.role.is_some_and(|r| r.kind.is_ragdoll_limb()) };
    // Input index -> output index, for the bones that became parts.
    let mut part_of: Vec<Option<usize>> = vec![None; skeleton.len()];
    let mut parts: Vec<RagdollPart> = Vec::new();
    let mut heads: Vec<DVec3> = Vec::new();
    let mut parents: Vec<Option<usize>> = Vec::new();
    // The topmost spine segment is the chest, so a chest-parented arm reads as one
    // in the report even though the LINK is an index. Labels are documentation
    // here; the hierarchy below is not.
    let last_spine = skeleton
        .iter()
        .enumerate()
        .filter(|(_, b)| b.role.map(|r| r.kind) == Some(inf_anim::BoneRoleKind::Spine))
        .map(|(i, _)| i)
        .next_back();

    for (i, bone) in skeleton.iter().enumerate() {
        if !is_part(bone) {
            continue;
        }
        let Some(role) = bone.role else { continue };
        // The nearest ANCESTOR that is a part — bounded by the chain's own length
        // so a malformed parent list cannot spin.
        let mut cur = bone.parent;
        let mut hops = 0usize;
        let mut parent_part = None;
        while let Some(p) = cur {
            let p = p as usize;
            if p >= skeleton.len() || hops > skeleton.len() {
                break;
            }
            if let Some(x) = part_of[p] {
                parent_part = Some(x);
                break;
            }
            cur = skeleton[p].parent;
            hops += 1;
        }
        let label = label_of(role, Some(i) == last_spine);
        parts.push(capsule_part(label, bone, config));
        heads.push(bone.head);
        parents.push(parent_part);
        part_of[i] = Some(parts.len() - 1);
    }

    for i in 0..parts.len() {
        let Some(parent_idx) = parents[i] else {
            continue;
        };
        let child_head = heads[i];
        let child_anchor = world_to_local(parts[i].rotation, parts[i].position, child_head);
        let parent_anchor = world_to_local(
            parts[parent_idx].rotation,
            parts[parent_idx].position,
            child_head,
        );
        let kind = if parts[i].role.is_hinge() {
            JointKind3D::Revolute {
                axis: DVec3::X,
                limits: Some(parts[i].role.hinge_limits()),
                motor: None,
            }
        } else {
            JointKind3D::Spherical
        };
        let desc = JointDesc3D::new(kind)
            .without_contacts()
            .local_anchor1(parent_anchor)
            .local_anchor2(child_anchor);
        parts[i].joint = Some(RagdollJoint {
            parent: parent_idx,
            desc,
        });
    }
    parts
}

/// The twelve-role label a rig role wears in the report. Documentation, not
/// linkage: the role-driven builder chains by index, so two parts sharing a label
/// is a legal and common state (five spine segments, two neck bones).
fn label_of(role: inf_anim::BoneRole, is_top_of_spine: bool) -> BoneRole {
    use inf_anim::BoneRoleKind as K;
    use inf_anim::BoneSide as S;
    let left = role.side != S::Right;
    match role.kind {
        K::Pelvis => BoneRole::Hips,
        K::Spine if is_top_of_spine => BoneRole::Chest,
        K::Spine => BoneRole::Spine,
        K::Neck | K::Head => BoneRole::Head,
        K::UpperArm => {
            if left {
                BoneRole::UpperArmL
            } else {
                BoneRole::UpperArmR
            }
        }
        K::LowerArm => {
            if left {
                BoneRole::LowerArmL
            } else {
                BoneRole::LowerArmR
            }
        }
        K::Thigh => {
            if left {
                BoneRole::ThighL
            } else {
                BoneRole::ThighR
            }
        }
        K::Calf => {
            if left {
                BoneRole::ShinL
            } else {
                BoneRole::ShinR
            }
        }
        _ => BoneRole::Spine,
    }
}

/// The body + capsule spanning one bone. Shared by both builders, so the two
/// paths cannot disagree about what a limb weighs or how thick it is.
fn capsule_part(role: BoneRole, bone: &RagdollBone, config: RagdollConfig) -> RagdollPart {
    let seg = bone.tail - bone.head;
    let raw = seg.length();
    let length = raw.max(1.0e-6);
    // **A zero-length bone has no direction**, and `from_rotation_arc` on a
    // near-zero vector is a NaN quaternion that a solver propagates into every
    // body it is jointed to. A rig with helper bones at their parent's origin has
    // plenty of these; none of them is a part, but a caller may hand us one.
    let rotation = if raw > 1.0e-6 && seg.is_finite() {
        DQuat::from_rotation_arc(DVec3::Y, seg / length)
    } else {
        DQuat::IDENTITY
    };
    let centre = (bone.head + bone.tail) * 0.5;
    let radius = (config.thickness * length).max(config.min_radius);
    let half_height = (length * 0.5 - radius).max(radius * 0.25);
    RagdollPart {
        role,
        name: bone.name.clone(),
        body: BodyDesc3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        },
        collider: ColliderDesc3D::new(ColliderShape3D::Capsule {
            half_height,
            radius,
        })
        .density(config.density)
        .layers(ragdoll_layers()),
        position: centre,
        rotation,
        joint: None,
    }
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
