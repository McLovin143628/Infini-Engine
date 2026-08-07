//! Root motion (P11.3): extract the **root joint's** ground-plane displacement
//! (and turn) from a clip, so a locomotion clip drives the *entity* instead of
//! sliding the character's feet on a stationary capsule.
//!
//! [`root_delta`] samples the root joint's local translation/rotation at two
//! play-head times and returns the **XZ translation delta** (metres) plus the
//! **yaw delta** (radians about the up axis) between them. The sim tick calls it
//! each fixed step with `(prev_t, cur_t)` and applies the delta to the entity's
//! `Transform` — through the 3D character mover when the entity is a character
//! controller, else as a raw transform add.
//!
//! ## Conventions & v1 scope
//!
//! * **Root joint** = the first joint with no parent (index 0 in a well-formed
//!   skeleton).
//! * **Ground plane only.** The vertical (Y) component of root translation is
//!   dropped in v1 — jumps/steps are the character controller's job (gravity +
//!   `move_and_slide`), not baked root height. Full 3-axis root motion and
//!   **blend-space root motion** (averaging the delta across a blended pair) are
//!   documented follow-ups.
//! * **Loop wrap.** When `looping` and `cur_t < prev_t` (the play-head wrapped
//!   past the clip end this step), the delta is summed across the seam:
//!   `(end − prev) + (cur − start)`, i.e. one clip's worth of locomotion carries
//!   the entity forward across the loop point instead of snapping it back. A
//!   single step is assumed shorter than the clip (true at any sane fixed rate).

use glam::Vec3;

use crate::clip::AnimClip;
use crate::pose::sample_clip;
use crate::skeleton::Skeleton;

/// The root-joint motion extracted over one time interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootMotionDelta {
    /// Ground-plane translation delta (X, 0, Z) in metres. Y is always `0` (see
    /// the module docs — vertical root motion is dropped in v1).
    pub translation: Vec3,
    /// Turn delta about the up (Y) axis, in radians.
    pub yaw: f32,
}

impl RootMotionDelta {
    /// The zero delta (no translation, no turn).
    pub const ZERO: Self = Self {
        translation: Vec3::ZERO,
        yaw: 0.0,
    };
}

/// Index of the skeleton's root joint (first parentless joint), if any.
pub fn root_joint_index(skeleton: &Skeleton) -> Option<usize> {
    skeleton.joints().iter().position(|j| j.parent.is_none())
}

/// The root joint's local translation at clip time `t` (non-looping sample; the
/// wrap is handled by [`root_delta`]).
fn root_translation_at(skeleton: &Skeleton, clip: &AnimClip, root: usize, t: f32) -> Vec3 {
    sample_clip(skeleton, clip, t, false).locals[root].translation_vec()
}

/// The root joint's local yaw (radians about +Y) at clip time `t`.
fn root_yaw_at(skeleton: &Skeleton, clip: &AnimClip, root: usize, t: f32) -> f32 {
    let q = sample_clip(skeleton, clip, t, false).locals[root].rotation_quat();
    // **`inf_math::pyaw`, not `Quat::to_euler`** (P24.2 re-audit F1).
    //
    // `to_euler` is `atan2` and `asin` inside glam — `std` libm, which the P14
    // law says is not bit-identical across targets — and this function is
    // reached from BOTH fixed steps (`simulate.rs` and `runtime_sim.rs` both
    // call `root_delta`), writing the result into an entity's `Transform` and
    // therefore into `state_bytes`. It slipped the P24.2 portability gate because
    // that gate read four files and this was not one of them: its title was true
    // of its file list and false of the pipeline.
    //
    // `pyaw` takes the same angle in closed form — `atan2(m02, m22)` of the YXZ
    // decomposition — so it is one angle instead of three, and portable.
    inf_math::pyaw(q)
}

/// The root-joint motion between play-head times `t0` and `t1` over `clip`.
///
/// `t0`/`t1` are play-head values already resolved into `[0, duration]` (the
/// [`AnimPlayer`](../../inf_ecs/index.html)-style wrapped `t`). When `looping`
/// and `t1 < t0`, the interval crossed the loop seam and the delta is summed
/// across it. The returned translation is XZ-only (Y dropped); the yaw is the
/// shortest signed turn about +Y.
pub fn root_delta(
    clip: &AnimClip,
    skeleton: &Skeleton,
    t0: f32,
    t1: f32,
    looping: bool,
) -> RootMotionDelta {
    let Some(root) = root_joint_index(skeleton) else {
        return RootMotionDelta::ZERO;
    };
    let dur = clip.duration;

    let (dt_trans, dt_yaw) = if looping && dur > 0.0 && t1 < t0 {
        // Crossed the loop seam: (end − prev) + (cur − start).
        let a = root_translation_at(skeleton, clip, root, dur)
            - root_translation_at(skeleton, clip, root, t0);
        let b = root_translation_at(skeleton, clip, root, t1)
            - root_translation_at(skeleton, clip, root, 0.0);
        let ya = wrap_angle(
            root_yaw_at(skeleton, clip, root, dur) - root_yaw_at(skeleton, clip, root, t0),
        );
        let yb = wrap_angle(
            root_yaw_at(skeleton, clip, root, t1) - root_yaw_at(skeleton, clip, root, 0.0),
        );
        (a + b, ya + yb)
    } else {
        let d = root_translation_at(skeleton, clip, root, t1)
            - root_translation_at(skeleton, clip, root, t0);
        let y = wrap_angle(
            root_yaw_at(skeleton, clip, root, t1) - root_yaw_at(skeleton, clip, root, t0),
        );
        (d, y)
    };

    RootMotionDelta {
        translation: Vec3::new(dt_trans.x, 0.0, dt_trans.z),
        yaw: dt_yaw,
    }
}

/// Wrap an angle into `(−π, π]` so a small turn near the ±π boundary stays small.
fn wrap_angle(a: f32) -> f32 {
    let two_pi = std::f32::consts::TAU;
    let mut a = a % two_pi;
    if a > std::f32::consts::PI {
        a -= two_pi;
    } else if a <= -std::f32::consts::PI {
        a += two_pi;
    }
    a
}

/// Build a single-joint clip whose root translates along `axis` at `speed`
/// m/s over `dur` seconds — a test/helper generator for locomotion root motion.
pub fn straight_line_clip(name: &str, axis: Vec3, speed: f32, dur: f32) -> AnimClip {
    use crate::clip::{Interpolation, JointTrack, Vec3Track};
    let mut jt = JointTrack::new(0);
    jt.translation = Some(Vec3Track::new(
        vec![0.0, dur],
        vec![Vec3::ZERO.to_array(), (axis * speed * dur).to_array()],
        Interpolation::Linear,
    ));
    AnimClip::new(name, vec![jt])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Interpolation, JointTrack, QuatTrack};
    use crate::skeleton::{Joint, JointTransform};
    use glam::{Mat4, Quat};

    fn one_joint() -> Skeleton {
        Skeleton::new(vec![Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::IDENTITY,
        }])
        .unwrap()
    }

    #[test]
    fn extracts_linear_ground_translation() {
        let sk = one_joint();
        // Root moves +1 m/s along +X over a 1 s clip.
        let clip = straight_line_clip("walk", Vec3::X, 1.0, 1.0);
        let d = root_delta(&clip, &sk, 0.0, 0.5, false);
        assert!((d.translation.x - 0.5).abs() < 1e-5, "{d:?}");
        assert_eq!(d.translation.y, 0.0);
        assert!(d.translation.z.abs() < 1e-6);
    }

    #[test]
    fn vertical_component_is_dropped() {
        let sk = one_joint();
        let clip = straight_line_clip("jump", Vec3::Y, 2.0, 1.0);
        let d = root_delta(&clip, &sk, 0.0, 1.0, false);
        assert_eq!(d.translation, Vec3::ZERO, "Y root motion must be dropped");
    }

    #[test]
    fn wraps_across_the_loop_seam() {
        let sk = one_joint();
        // +1 m/s along +X, 1 s loop. Step 0.9 → 0.1 crosses the seam: it should
        // report (1.0−0.9) + (0.1−0) = 0.2, not the −0.8 a naive subtract gives.
        let clip = straight_line_clip("walk", Vec3::X, 1.0, 1.0);
        let d = root_delta(&clip, &sk, 0.9, 0.1, true);
        assert!((d.translation.x - 0.2).abs() < 1e-5, "{d:?}");
    }

    #[test]
    fn extracts_yaw_turn() {
        let sk = one_joint();
        // Root yaws 0 → 90° about +Y over 1 s.
        let mut jt = JointTrack::new(0);
        jt.rotation = Some(QuatTrack::new(
            vec![0.0, 1.0],
            vec![
                Quat::IDENTITY.to_array(),
                Quat::from_rotation_y(90f32.to_radians()).to_array(),
            ],
            Interpolation::Linear,
        ));
        let clip = AnimClip::new("turn", vec![jt]);
        let d = root_delta(&clip, &sk, 0.0, 1.0, false);
        assert!(
            (d.yaw.to_degrees() - 90.0).abs() < 0.5,
            "{}",
            d.yaw.to_degrees()
        );
    }

    #[test]
    fn no_root_track_is_zero() {
        let sk = one_joint();
        let clip = AnimClip::new("static", vec![]);
        assert_eq!(
            root_delta(&clip, &sk, 0.0, 1.0, false),
            RootMotionDelta::ZERO
        );
    }
}
