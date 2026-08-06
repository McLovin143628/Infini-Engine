//! Pose evaluation (P11.1): clip → local pose → global transforms → skinning
//! matrices, plus pose blending and clip-time advance.
//!
//! Everything here is **pure and deterministic** (a function of its inputs
//! alone) so it can move onto the §2.5 parallel pose hot loop later without a
//! rewrite — no interior mutability, no globals, no allocation-order surprises.

use glam::Mat4;

use crate::clip::{resolve_time, AnimClip};
use crate::skeleton::{JointTransform, Skeleton};

/// A skeletal pose: one **local** TRS per joint, index-aligned to a skeleton's
/// `joints`. Seed it from the skeleton's bind pose ([`Pose::rest`]) and overlay
/// animated joints with [`sample_clip`].
#[derive(Clone, Debug, PartialEq)]
pub struct Pose {
    /// Local transforms, one per skeleton joint (same order/length).
    pub locals: Vec<JointTransform>,
}

impl Pose {
    /// The skeleton's bind (rest) pose — every joint at its `local_bind`.
    pub fn rest(skeleton: &Skeleton) -> Self {
        Self {
            locals: skeleton.joints().iter().map(|j| j.local_bind).collect(),
        }
    }

    /// Number of joints this pose covers.
    pub fn len(&self) -> usize {
        self.locals.len()
    }

    /// True if the pose has no joints.
    pub fn is_empty(&self) -> bool {
        self.locals.is_empty()
    }
}

/// Sample `clip` at time `t` (seconds) into a full [`Pose`] over `skeleton`.
///
/// The skeleton is required to seed joints the clip does **not** animate from
/// their bind transform (a clip that only rotates one joint must still yield a
/// complete pose). `looping` wraps `t` into `[0, duration]` (else it clamps).
/// Each present channel overrides the corresponding component of the joint's
/// bind transform; absent channels keep the bind value.
pub fn sample_clip(skeleton: &Skeleton, clip: &AnimClip, t: f32, looping: bool) -> Pose {
    let mut pose = Pose::rest(skeleton);
    let time = resolve_time(t, clip.duration, looping);
    for track in &clip.tracks {
        let Some(local) = pose.locals.get_mut(track.joint as usize) else {
            continue; // track references a joint outside this skeleton — skip
        };
        if let Some(ch) = &track.translation {
            if let Some(v) = ch.sample(time) {
                local.translation = v.to_array();
            }
        }
        if let Some(ch) = &track.rotation {
            if let Some(q) = ch.sample(time) {
                local.rotation = q.to_array();
            }
        }
        if let Some(ch) = &track.scale {
            if let Some(v) = ch.sample(time) {
                local.scale = v.to_array();
            }
        }
    }
    pose
}

/// Blend two poses component-wise: lerp translation/scale, **slerp** rotation
/// (shortest path), at `alpha ∈ [0,1]` (`0` = `a`, `1` = `b`).
///
/// The poses must share a joint count (both sampled from the same skeleton). A
/// mismatch takes the shorter length (defensive; callers should match them).
pub fn blend_poses(a: &Pose, b: &Pose, alpha: f32) -> Pose {
    let alpha = alpha.clamp(0.0, 1.0);
    let n = a.locals.len().min(b.locals.len());
    let mut locals = Vec::with_capacity(n);
    for i in 0..n {
        let la = &a.locals[i];
        let lb = &b.locals[i];
        let t = la.translation_vec().lerp(lb.translation_vec(), alpha);
        let s = la.scale_vec().lerp(lb.scale_vec(), alpha);
        let qa = la.rotation_quat();
        let qb = lb.rotation_quat();
        // **`inf_math::pslerp`, not `Quat::slerp`** (P24.2 audit M-SLERP).
        // glam's calls `acos_approx` plus three `sin`s -- `std` libm, which the
        // P14 law says is not bit-identical across targets -- and this blend runs
        // on every state-machine transition with a non-zero duration and in every
        // blend space. Its result reaches the evaluated pose, which is folded into
        // `state_bytes`, which two hosts are compared on. `pslerp` takes the
        // shortest arc itself, so the sign fix that used to be here is inside it.
        let r = inf_math::pslerp(qa, qb, alpha);
        locals.push(JointTransform::from_trs(t, r, s));
    }
    Pose { locals }
}

/// Propagate a local [`Pose`] to **global** (model-space) matrices — one per
/// joint. Because the skeleton is topologically ordered, a single forward pass
/// suffices: `global[i] = global[parent] · local[i]` (roots use `local[i]`).
///
/// If the pose is shorter than the skeleton, missing joints fall back to their
/// bind transform (defensive).
pub fn global_transforms(skeleton: &Skeleton, pose: &Pose) -> Vec<Mat4> {
    let joints = skeleton.joints();
    let mut globals = vec![Mat4::IDENTITY; joints.len()];
    for (i, joint) in joints.iter().enumerate() {
        let local = pose
            .locals
            .get(i)
            .copied()
            .unwrap_or(joint.local_bind)
            .to_mat4();
        globals[i] = match joint.parent {
            Some(p) => globals[p as usize] * local,
            None => local,
        };
    }
    globals
}

/// The GPU **skinning palette**: `skin[i] = global[i] · inverse_bind[i]`, one
/// matrix per joint. This is what the vertex shader multiplies a bind-space
/// vertex by (weighted across its joints) to deform the mesh.
pub fn skinning_matrices(skeleton: &Skeleton, pose: &Pose) -> Vec<Mat4> {
    let globals = global_transforms(skeleton, pose);
    skeleton
        .joints()
        .iter()
        .zip(globals)
        .map(|(joint, global)| global * joint.inverse_bind_mat())
        .collect()
}

/// Advance a clip play-head by one step: `t + speed·dt`, then wrap (looping) or
/// clamp (non-looping) against `duration`. Pure and deterministic — the shared
/// time-integration used by both the editor Simulate tick and the runtime sim
/// (P11.1 deliverable 5). A non-positive `duration` (clip length unknown) leaves
/// `t` free-running so pose sampling can wrap later.
pub fn advance_clip_time(t: f64, speed: f64, dt: f64, duration: f64, looping: bool) -> f64 {
    let next = t + speed * dt;
    if duration <= 0.0 {
        return next;
    }
    if looping {
        next.rem_euclid(duration)
    } else {
        next.clamp(0.0, duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Interpolation, JointTrack, QuatTrack};
    use crate::skeleton::Joint;
    use glam::{Quat, Vec3};

    /// A straight 3-joint chain along +Y: root at origin, each child 1 unit above
    /// its parent (local translation `+Y`). Inverse binds are the inverse of each
    /// joint's global bind (so the rest pose yields identity skinning matrices).
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
    fn rest_pose_globals_stack_along_y() {
        let sk = chain();
        let globals = global_transforms(&sk, &Pose::rest(&sk));
        // Hand-computed: joint k sits k units up the +Y axis.
        for (k, g) in globals.iter().enumerate() {
            let p = g.transform_point3(Vec3::ZERO);
            assert!(
                (p - Vec3::new(0.0, k as f32, 0.0)).length() < 1e-5,
                "joint {k} at {p:?}"
            );
        }
    }

    #[test]
    fn rest_pose_skinning_matrices_are_identity() {
        let sk = chain();
        for m in skinning_matrices(&sk, &Pose::rest(&sk)) {
            assert!(m.abs_diff_eq(Mat4::IDENTITY, 1e-5));
        }
    }

    #[test]
    fn rotating_a_joint_swings_its_descendants() {
        let sk = chain();
        // Rotate the middle joint (index 1) 90° about +Z: its child (index 2),
        // which was at +Y relative to it, swings toward −X.
        let mut pose = Pose::rest(&sk);
        pose.locals[1].rotation = Quat::from_rotation_z(90f32.to_radians()).to_array();
        let globals = global_transforms(&sk, &pose);
        // Joint 1 stays at (0,1,0); joint 2's tip swings to about (-1, 1, 0).
        let p1 = globals[1].transform_point3(Vec3::ZERO);
        let p2 = globals[2].transform_point3(Vec3::ZERO);
        assert!((p1 - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5, "{p1:?}");
        assert!((p2 - Vec3::new(-1.0, 1.0, 0.0)).length() < 1e-4, "{p2:?}");
    }

    #[test]
    fn sample_clip_at_and_between_keys() {
        let sk = chain();
        // A clip rotating joint 1 from 0° to 90° over 1 s.
        let mut jt = JointTrack::new(1);
        jt.rotation = Some(QuatTrack::new(
            vec![0.0, 1.0],
            vec![
                Quat::IDENTITY.to_array(),
                Quat::from_rotation_z(90f32.to_radians()).to_array(),
            ],
            Interpolation::Linear,
        ));
        let clip = AnimClip::new("bend", vec![jt]);

        // At t=0 the pose equals rest.
        let p0 = sample_clip(&sk, &clip, 0.0, false);
        assert_eq!(p0, Pose::rest(&sk));

        // At t=1 joint 1 is fully rotated 90°.
        let p1 = sample_clip(&sk, &clip, 1.0, false);
        let q1 = p1.locals[1].rotation_quat();
        assert!(q1.angle_between(Quat::from_rotation_z(90f32.to_radians())) < 1e-4);

        // At the midpoint it is ~45°.
        let pm = sample_clip(&sk, &clip, 0.5, false);
        let qm = pm.locals[1].rotation_quat();
        assert!((qm.angle_between(Quat::IDENTITY).to_degrees() - 45.0).abs() < 1.0);

        // Looping wraps: t=1.5 == t=0.5 for a 1 s clip.
        let pw = sample_clip(&sk, &clip, 1.5, true);
        assert_eq!(pw, pm);
    }

    #[test]
    fn blend_midpoint_is_halfway() {
        let sk = chain();
        let a = Pose::rest(&sk);
        let mut b = Pose::rest(&sk);
        b.locals[1].rotation = Quat::from_rotation_z(90f32.to_radians()).to_array();
        let mid = blend_poses(&a, &b, 0.5);
        let ang = mid.locals[1].rotation_quat().angle_between(Quat::IDENTITY);
        assert!(
            (ang.to_degrees() - 45.0).abs() < 1.0,
            "{}",
            ang.to_degrees()
        );
    }

    #[test]
    fn advance_clip_time_wraps_and_clamps() {
        // Free-running when duration unknown.
        assert_eq!(advance_clip_time(0.0, 1.0, 0.5, 0.0, true), 0.5);
        // Looping wraps.
        assert!((advance_clip_time(1.8, 1.0, 0.5, 2.0, true) - 0.3).abs() < 1e-9);
        // Non-looping clamps at the end.
        assert_eq!(advance_clip_time(1.9, 1.0, 0.5, 2.0, false), 2.0);
    }
}
