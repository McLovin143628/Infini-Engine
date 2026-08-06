//! Sockets & attachments (P11.3): named attach points that ride a skeleton joint.
//!
//! A [`Socket`] is authored **per skeleton** (stored additively on
//! [`SkeletonAsset`](crate::asset::SkeletonAsset)) as a joint index plus a fixed
//! **local offset** in that joint's space — the classic "weapon in the hand",
//! "muzzle on the barrel" attach point. Given a live [`Pose`], the socket's
//! model-space transform is the joint's animated global transform composed with
//! the offset:
//!
//! ```text
//! socket_world = global_transforms(skeleton, pose)[socket.joint] · offset
//! ```
//!
//! This module stays **pure** (like the rest of `inf-anim`): it turns a
//! `(skeleton, pose, socket)` triple into a matrix. Making one *entity* follow
//! another entity's socket is an ECS concern (`inf_ecs::AttachedTo` plus the
//! post-anim-tick attachment system) — the pure math here is what it evaluates.
//!
//! It had **no runtime caller at all** until P24.1: the attachment system
//! composed the target's origin and the offset, so a sword on `hand_r` rode the
//! pelvis. [`socket_transforms`] is the batch form the fixed step calls, once per
//! posed character, and `inf_ecs::pose::EvaluatedPose::sockets` is where its
//! result lives.

use glam::Mat4;
use serde::{Deserialize, Serialize};

use crate::pose::{global_transforms, Pose};
use crate::skeleton::{JointTransform, Skeleton};

/// A named attach point riding a skeleton joint at a fixed local offset.
///
/// `joint` indexes the skeleton's `joints`; `local_offset` is the socket's TRS in
/// that joint's local space (identity = coincident with the joint). Serde-clean
/// (no `skip_serializing_if`) so it round-trips through the `.inf_skel` payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Socket {
    /// Human-readable socket name (unique within a skeleton), e.g. `"hand_r"`.
    pub name: String,
    /// Index of the joint this socket rides (into the skeleton's `joints`).
    pub joint: u16,
    /// The socket's transform in the joint's local space.
    pub local_offset: JointTransform,
}

impl Socket {
    /// A socket named `name` on `joint` with the identity local offset.
    pub fn new(name: impl Into<String>, joint: u16) -> Self {
        Self {
            name: name.into(),
            joint,
            local_offset: JointTransform::IDENTITY,
        }
    }

    /// Builder: set the local offset.
    pub fn with_offset(mut self, offset: JointTransform) -> Self {
        self.local_offset = offset;
        self
    }
}

/// The socket's **model-space** transform for a given pose: the joint's animated
/// global transform composed with the socket's local offset. A socket whose
/// `joint` is out of range for the skeleton/pose falls back to the offset alone
/// (defensive — a malformed socket never panics the pose pipeline).
pub fn socket_transform(skeleton: &Skeleton, pose: &Pose, socket: &Socket) -> Mat4 {
    let globals = global_transforms(skeleton, pose);
    let base = globals
        .get(socket.joint as usize)
        .copied()
        .unwrap_or(Mat4::IDENTITY);
    base * socket.local_offset.to_mat4()
}

/// Find a socket by name in a slice (sockets are few; linear scan).
pub fn find_socket<'a>(sockets: &'a [Socket], name: &str) -> Option<&'a Socket> {
    sockets.iter().find(|s| s.name == name)
}

/// Every socket's model-space transform for one pose, as `(name, matrix)` pairs
/// **sorted by name**.
///
/// The batch form of [`socket_transform`], and the one a fixed step calls: it
/// runs [`global_transforms`] **once** for the whole skeleton instead of once per
/// socket, which is the difference between `O(joints)` and `O(sockets · joints)`
/// per character per step. The per-socket rule is unchanged — `globals[joint] ·
/// local_offset`, with the same defensive identity fallback for an out-of-range
/// joint — and `socket_transforms_agree_with_the_single_socket_rule` below pins
/// the two against each other so the fast path can never quietly mean something
/// different from the definition.
///
/// Sorted by name because this is sim state: the attachment system looks a socket
/// up by name, and a deterministic order makes the result of a *pose* a function
/// of the skeleton alone rather than of the order sockets happened to be authored
/// in. Duplicate names (which the editor does not produce) keep every entry; a
/// lookup takes the first.
pub fn socket_transforms(
    skeleton: &Skeleton,
    pose: &Pose,
    sockets: &[Socket],
) -> Vec<(String, Mat4)> {
    if sockets.is_empty() {
        return Vec::new();
    }
    let globals = global_transforms(skeleton, pose);
    let mut out: Vec<(String, Mat4)> = sockets
        .iter()
        .map(|s| {
            let base = globals
                .get(s.joint as usize)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            (s.name.clone(), base * s.local_offset.to_mat4())
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::Joint;
    use glam::{Quat, Vec3};

    /// A straight 3-joint chain along +Y (root at origin, each child +1 Y).
    fn chain() -> Skeleton {
        let mut joints = Vec::new();
        let mut global = Mat4::IDENTITY;
        for i in 0..3 {
            let local = JointTransform::from_trs(
                if i == 0 { Vec3::ZERO } else { Vec3::Y },
                Quat::IDENTITY,
                Vec3::ONE,
            );
            global *= local.to_mat4();
            joints.push(Joint {
                name: format!("j{i}"),
                parent: if i == 0 { None } else { Some(i as u16 - 1) },
                inverse_bind: global.inverse().to_cols_array(),
                local_bind: local,
            });
        }
        Skeleton::new(joints).unwrap()
    }

    #[test]
    fn socket_rides_its_joint_in_the_rest_pose() {
        let sk = chain();
        // A socket on the tip joint (index 2, at world +2Y) offset +1 along local X.
        let socket = Socket::new("tip", 2).with_offset(JointTransform::from_trs(
            Vec3::X,
            Quat::IDENTITY,
            Vec3::ONE,
        ));
        let m = socket_transform(&sk, &Pose::rest(&sk), &socket);
        let p = m.transform_point3(Vec3::ZERO);
        // Joint 2 sits at (0,2,0); the +X offset lands the socket at (1,2,0).
        assert!((p - Vec3::new(1.0, 2.0, 0.0)).length() < 1e-5, "{p:?}");
    }

    #[test]
    fn socket_follows_the_posed_joint() {
        let sk = chain();
        // Rotate the middle joint 90° about +Z: its child (joint 2) swings toward
        // −X, so a socket on joint 2 rides along with it.
        let mut pose = Pose::rest(&sk);
        pose.locals[1].rotation = Quat::from_rotation_z(90f32.to_radians()).to_array();
        let socket = Socket::new("tip", 2);
        let m = socket_transform(&sk, &pose, &socket);
        let p = m.transform_point3(Vec3::ZERO);
        // Joint 2's origin swings from (0,2,0) to about (-1,1,0).
        assert!((p - Vec3::new(-1.0, 1.0, 0.0)).length() < 1e-4, "{p:?}");
    }

    #[test]
    fn find_socket_by_name() {
        let sockets = vec![Socket::new("a", 0), Socket::new("b", 1)];
        assert_eq!(find_socket(&sockets, "b").unwrap().joint, 1);
        assert!(find_socket(&sockets, "missing").is_none());
    }

    /// The batch form is the single-socket rule, run once per skeleton rather
    /// than once per socket. If they ever disagreed, a character's weapon would
    /// sit somewhere the socket editor's own preview says it does not.
    #[test]
    fn socket_transforms_agree_with_the_single_socket_rule() {
        let sk = chain();
        let mut pose = Pose::rest(&sk);
        pose.locals[1].rotation = Quat::from_rotation_z(37f32.to_radians()).to_array();
        let sockets = vec![
            Socket::new("tip", 2).with_offset(JointTransform::from_trs(
                Vec3::X,
                Quat::from_rotation_y(0.3),
                Vec3::ONE,
            )),
            Socket::new("base", 0),
            // Out of range on purpose: the defensive fallback must be the same
            // one on both paths.
            Socket::new("ghost", 99),
        ];
        let batch = socket_transforms(&sk, &pose, &sockets);
        assert_eq!(batch.len(), 3);
        // Sorted by name — sim state, so the order is a property of the skeleton.
        assert_eq!(
            batch.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            vec!["base", "ghost", "tip"]
        );
        for s in &sockets {
            let one = socket_transform(&sk, &pose, s);
            let (_, many) = batch.iter().find(|(n, _)| n == &s.name).unwrap();
            assert_eq!(one.to_cols_array(), many.to_cols_array(), "{}", s.name);
        }
        assert!(socket_transforms(&sk, &pose, &[]).is_empty());
    }
}
