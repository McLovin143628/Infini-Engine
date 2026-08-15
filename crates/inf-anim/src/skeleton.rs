//! Skeleton data model (P11.1): a flat, **topologically ordered** list of joints.
//!
//! # Why f32 (not f64)
//!
//! Every quantity here is `f32`, deliberately — the opposite of the engine's f64
//! world-coordinate doctrine (architecture rule 3). That rule exists to defeat
//! floating-origin precision loss at *planetary* distances; a skeleton lives in
//! **model space near the origin** (a character rig spans metres, not a solar
//! system), so f32 has ample precision, and pose evaluation's product is the GPU
//! **skinning palette** — an `f32` storage buffer (§2.5, P11.1 deliverable 4).
//! Keeping the whole pose pipeline f32 avoids a per-frame f64→f32 narrowing on
//! the hot path. Determinism is preserved: IEEE-754 f32 ops are deterministic
//! per platform, and pose math feeds the *render* projection, never the
//! authoritative sim state hash (which stays f64 in the ECS/physics layer).

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A joint's local transform as translation / rotation / scale (TRS). Stored
/// decomposed (not a baked matrix) so clip tracks can drive each channel
/// independently and blends can slerp rotations.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointTransform {
    /// Local translation relative to the parent joint (metres).
    pub translation: [f32; 3],
    /// Local rotation quaternion, `[x, y, z, w]`.
    pub rotation: [f32; 4],
    /// Local scale.
    pub scale: [f32; 3],
}

impl Default for JointTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl JointTransform {
    /// The identity transform (no translation, identity rotation, unit scale).
    pub const IDENTITY: Self = Self {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
    };

    /// Build from glam components.
    pub fn from_trs(translation: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            translation: translation.to_array(),
            rotation: rotation.to_array(),
            scale: scale.to_array(),
        }
    }

    /// The translation as a glam vector.
    pub fn translation_vec(&self) -> Vec3 {
        Vec3::from_array(self.translation)
    }

    /// The rotation as a glam quaternion (normalized on read — clip lerp can
    /// leave it slightly off-unit).
    pub fn rotation_quat(&self) -> Quat {
        Quat::from_array(self.rotation).normalize()
    }

    /// The scale as a glam vector.
    pub fn scale_vec(&self) -> Vec3 {
        Vec3::from_array(self.scale)
    }

    /// Compose the local TRS into an affine matrix (`T · R · S`).
    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            self.scale_vec(),
            self.rotation_quat(),
            self.translation_vec(),
        )
    }
}

/// One joint (bone) in a [`Skeleton`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Joint {
    /// Human-readable joint name (from the glTF node, for sockets/retarget later).
    pub name: String,
    /// Index of the parent joint in the skeleton's `joints` list, or `None` for a
    /// root. **Invariant:** `parent < self_index` (topological order).
    pub parent: Option<u16>,
    /// The inverse bind matrix (world-of-mesh → joint-local at bind time), stored
    /// column-major (`Mat4::to_cols_array`). Multiplying a joint's animated global
    /// transform by this yields the skinning matrix.
    pub inverse_bind: [f32; 16],
    /// The joint's transform in the bind (rest) pose — the value a [`Pose`] uses
    /// when no clip track animates this joint.
    pub local_bind: JointTransform,
}

impl Joint {
    /// The inverse bind matrix as a glam matrix.
    pub fn inverse_bind_mat(&self) -> Mat4 {
        Mat4::from_cols_array(&self.inverse_bind)
    }
}

/// Errors from constructing/validating a [`Skeleton`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SkeletonError {
    /// A skeleton must have at least one joint.
    #[error("skeleton has no joints")]
    Empty,
    /// More joints than a `u16` parent index can address.
    #[error("skeleton has {0} joints, exceeding the u16 joint-index limit")]
    TooManyJoints(usize),
    /// A joint's parent index is out of range.
    #[error("joint {joint} names parent {parent} which is out of range (len {len})")]
    ParentOutOfRange {
        joint: usize,
        parent: u16,
        len: usize,
    },
    /// A joint's parent does not precede it (breaks topological order).
    #[error("joint {joint} names parent {parent}: parents must precede their children")]
    NotTopological { joint: usize, parent: u16 },
}

/// A skinning skeleton: a flat, topologically ordered joint hierarchy.
///
/// "Topologically ordered" means every joint's parent appears **before** it in
/// `joints`, so a single forward pass computes global transforms (see
/// [`crate::pose::global_transforms`]). glTF skins already satisfy this in
/// practice; [`Skeleton::new`] validates it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Skeleton {
    joints: Vec<Joint>,
}

impl Skeleton {
    /// Build and validate a skeleton. Errors if empty, oversized, or if any
    /// parent index is out of range or does not precede its child.
    pub fn new(joints: Vec<Joint>) -> Result<Self, SkeletonError> {
        let me = Self { joints };
        me.validate()?;
        Ok(me)
    }

    /// The questions [`Skeleton::new`] asks, asked of a skeleton that already
    /// exists (round-2 finding B2's sweep).
    ///
    /// # Why this is separable at all
    ///
    /// `joints` is private and `Skeleton` derives `Deserialize`, and serde's
    /// derive is expanded in this very module — so it reconstructs `Skeleton {
    /// joints }` **directly from the wire and never calls `new`**. A `.inf_skel`
    /// whose joint 1 names parent 9999 therefore decoded as a perfectly valid
    /// skeleton, and [`crate::pose::global_transforms`] indexes
    /// `globals[p as usize]` with no bound — a panic in the editor, in PIE and
    /// in the shipped player reading a cooked pack, reachable from every posed
    /// character, every socket resolve, every cloth and hair seed and the IK
    /// solve (which walks the whole skeleton, not just its chain).
    ///
    /// An in-range but non-topological parent does not panic: it reads the
    /// still-identity slot ahead of it and produces a pose that is wrong rather
    /// than absent, which is worse.
    pub fn validate(&self) -> Result<(), SkeletonError> {
        if self.joints.is_empty() {
            return Err(SkeletonError::Empty);
        }
        if self.joints.len() > u16::MAX as usize + 1 {
            return Err(SkeletonError::TooManyJoints(self.joints.len()));
        }
        let len = self.joints.len();
        for (i, joint) in self.joints.iter().enumerate() {
            if let Some(parent) = joint.parent {
                if parent as usize >= len {
                    return Err(SkeletonError::ParentOutOfRange {
                        joint: i,
                        parent,
                        len,
                    });
                }
                if parent as usize >= i {
                    return Err(SkeletonError::NotTopological { joint: i, parent });
                }
            }
        }
        Ok(())
    }

    /// The joints, in topological order.
    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    /// Number of joints.
    pub fn len(&self) -> usize {
        self.joints.len()
    }

    /// Always `false` (a valid skeleton has ≥ 1 joint), provided for lint parity.
    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
    }

    /// The joint at `index`, if in range.
    pub fn joint(&self, index: usize) -> Option<&Joint> {
        self.joints.get(index)
    }

    /// Find a joint index by name (linear scan; skeletons are small).
    pub fn index_of(&self, name: &str) -> Option<u16> {
        self.joints
            .iter()
            .position(|j| j.name == name)
            .map(|i| i as u16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn joint(name: &str, parent: Option<u16>) -> Joint {
        Joint {
            name: name.into(),
            parent,
            inverse_bind: Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::IDENTITY,
        }
    }

    #[test]
    fn valid_chain_constructs() {
        let sk = Skeleton::new(vec![
            joint("root", None),
            joint("mid", Some(0)),
            joint("tip", Some(1)),
        ])
        .unwrap();
        assert_eq!(sk.len(), 3);
        assert_eq!(sk.index_of("tip"), Some(2));
    }

    #[test]
    fn empty_is_rejected() {
        assert_eq!(Skeleton::new(vec![]), Err(SkeletonError::Empty));
    }

    #[test]
    fn out_of_range_parent_is_rejected() {
        let err = Skeleton::new(vec![joint("root", Some(5))]).unwrap_err();
        assert!(matches!(err, SkeletonError::ParentOutOfRange { .. }));
    }

    #[test]
    fn non_topological_parent_is_rejected() {
        // Child (index 0) names parent 1, which comes after it.
        let err = Skeleton::new(vec![joint("a", Some(1)), joint("b", None)]).unwrap_err();
        assert_eq!(
            err,
            SkeletonError::NotTopological {
                joint: 0,
                parent: 1
            }
        );
    }

    #[test]
    fn joint_transform_round_trips_through_matrix() {
        let t = JointTransform::from_trs(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(0.5),
            Vec3::splat(2.0),
        );
        let m = t.to_mat4();
        let (s, r, tr) = m.to_scale_rotation_translation();
        assert!((tr - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
        assert!((s - Vec3::splat(2.0)).length() < 1e-5);
        assert!(r.dot(Quat::from_rotation_y(0.5)).abs() > 0.999);
    }
}
