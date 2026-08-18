//! Pose snapshots and pose matching (P29.2 clause 6).
//!
//! Two primitives the movement catalogue's amendment assigned to this wave, whose
//! consumers land in P29.4:
//!
//! * a **snapshot** — the pose that is on screen, captured as a value, so a blend
//!   can start from *what the character was actually doing* rather than from a
//!   state the machine happens to name. ALS does this literally
//!   (`SavePoseSnapshot("RagdollPose")` on ragdoll exit, so the AnimBP can blend
//!   out of a physics pose); ours is a plain [`Pose`] plus the play-head and an
//!   optional per-joint velocity, because a snapshot with no velocity can only be
//!   blended *from* and not *matched into*.
//! * **pose matching** — score candidate frames against a query pose and pick the
//!   nearest. This is what §13 chose over ALS's dynamic-transition corrective
//!   (which fires a canned single-foot animation whenever a foot drifts more than
//!   8 cm from its virtual target): rather than patching the mismatch, enter the
//!   next clip at the frame that does not have one.
//!
//! # The cost function
//!
//! Three terms, each a squared error so nothing needs a square root and nothing
//! needs an angle:
//!
//! * **position** — model-space joint positions, metres². Model space and not
//!   local, because that is where a mismatch is visible: a shoulder rotation error
//!   moves the hand, and the local-space term would price both the same.
//! * **rotation** — `1 − |dot(q_a, q_b)|` per joint, which is `0` for the same
//!   orientation and grows monotonically with the angle between them. Monotone is
//!   all a *comparison* needs, and it costs no `acos` — which the pose path bans.
//! * **velocity** — model-space joint velocity, (m/s)². Only when both sides have
//!   one: the query from [`PoseSnapshot::capture_moving`], the candidate from a
//!   central difference on its own clip. Without it a match is a *shape* match and
//!   a foot swinging forward scores the same as one swinging back, which is how a
//!   pose-matched landing picks a frame that immediately reverses.
//!
//! A [`crate::JointMask`] weights the terms per joint, so "match the legs, ignore
//! what the arms are carrying" is a mask rather than a second cost function.
//!
//! # Portability
//!
//! Squared distances, dot products and multiplies. No transcendental, no `sqrt`.

use glam::{Mat4, Vec3};

use crate::clip::AnimClip;
use crate::layers::JointMask;
use crate::pose::{global_transforms, sample_clip, Pose};
use crate::skeleton::Skeleton;

/// A captured pose: what was on screen, and (optionally) how fast it was moving.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseSnapshot {
    /// A name, for diagnostics — ALS's `"RagdollPose"` in our spelling.
    pub label: String,
    /// The captured pose.
    pub pose: Pose,
    /// The play-head the pose was captured at, seconds. Carried so a consumer can
    /// say *where* it was as well as *what* it was.
    pub time_s: f64,
    /// Per-joint **model-space** velocity, m/s, index-aligned with `pose.locals`.
    /// `None` when the snapshot was taken from a single frame.
    pub velocity: Option<Vec<[f32; 3]>>,
}

impl PoseSnapshot {
    /// Capture a pose with no velocity — enough to blend *from*.
    pub fn capture(label: impl Into<String>, pose: &Pose, time_s: f64) -> Self {
        Self {
            label: label.into(),
            pose: pose.clone(),
            time_s,
            velocity: None,
        }
    }

    /// Capture a pose **and** its velocity, from this frame and the last.
    ///
    /// `dt` is the interval between them, seconds; a non-positive or non-finite
    /// `dt` yields a velocity-free snapshot rather than an infinity — the first
    /// frame of a session has no previous frame and must not poison the cost.
    pub fn capture_moving(
        label: impl Into<String>,
        skeleton: &Skeleton,
        pose: &Pose,
        previous: &Pose,
        dt: f32,
        time_s: f64,
    ) -> Self {
        let velocity = crate::positive(dt)
            .then(|| model_velocity(skeleton, previous, pose, dt))
            .filter(|v| v.iter().all(|x| x.iter().all(|c| c.is_finite())));
        Self {
            label: label.into(),
            pose: pose.clone(),
            time_s,
            velocity,
        }
    }
}

/// Per-joint model-space velocity between two poses, m/s.
fn model_velocity(skeleton: &Skeleton, from: &Pose, to: &Pose, dt: f32) -> Vec<[f32; 3]> {
    let a = global_transforms(skeleton, from);
    let b = global_transforms(skeleton, to);
    a.iter()
        .zip(&b)
        .map(|(x, y)| ((origin(*y) - origin(*x)) / dt).to_array())
        .collect()
}

/// The translation column of a model matrix.
fn origin(m: Mat4) -> Vec3 {
    m.w_axis.truncate()
}

/// How the three error terms are priced, and where.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseMatchWeights {
    /// Weight on squared model-space position error (per m²).
    pub position: f32,
    /// Weight on rotation error (`1 − |dot|`, dimensionless).
    pub rotation: f32,
    /// Weight on squared model-space velocity error (per (m/s)²).
    pub velocity: f32,
    /// Which joints count, and how much. `None` = every joint at 1.
    pub mask: Option<JointMask>,
}

impl Default for PoseMatchWeights {
    /// Position-dominant, with rotation as a tie-break and velocity priced to
    /// matter at running speed.
    ///
    /// The numbers are a *starting* balance and are meant to be overridden per
    /// consumer (a get-up cares about the torso's orientation; a landing cares
    /// about where the feet are going). What they encode: a 10 cm limb error
    /// (0.01 m² × 1.0 = 0.01) prices about the same as a 15° rotation error
    /// (1 − cos(7.5°) ≈ 0.0086 × 1.0) and about the same as a 0.3 m/s velocity
    /// error (0.09 × 0.1 = 0.009).
    fn default() -> Self {
        Self {
            position: 1.0,
            rotation: 1.0,
            velocity: 0.1,
            mask: None,
        }
    }
}

impl PoseMatchWeights {
    /// Restrict the match to a joint mask.
    pub fn with_mask(mut self, mask: JointMask) -> Self {
        self.mask = Some(mask);
        self
    }

    /// The dense per-joint weights over `joints` joints.
    fn joint_weights(&self, joints: usize) -> Vec<f32> {
        match &self.mask {
            Some(m) => m.resolve(joints, 1.0),
            None => vec![1.0; joints],
        }
    }
}

/// The cost of `candidate` as a match for `query` — **lower is better**, `0` is
/// identical.
///
/// `query_velocity` / `candidate_velocity` are model-space, m/s, index-aligned
/// with the poses; the velocity term is priced only where both are present.
pub fn pose_cost(
    skeleton: &Skeleton,
    query: &Pose,
    query_velocity: Option<&[[f32; 3]]>,
    candidate: &Pose,
    candidate_velocity: Option<&[[f32; 3]]>,
    w: &PoseMatchWeights,
) -> f32 {
    let n = query.locals.len().min(candidate.locals.len());
    let weights = w.joint_weights(n);
    let qg = global_transforms(skeleton, query);
    let cg = global_transforms(skeleton, candidate);
    let mut cost = 0.0f32;
    for i in 0..n {
        let jw = weights.get(i).copied().unwrap_or(0.0);
        if !(jw > 0.0) {
            continue;
        }
        if let (Some(a), Some(b)) = (qg.get(i), cg.get(i)) {
            cost += jw * w.position * (origin(*b) - origin(*a)).length_squared();
        }
        let qa = query.locals[i].rotation_quat();
        let qb = candidate.locals[i].rotation_quat();
        // `1 − |dot|`: zero for the same orientation, monotone in the angle, and
        // free of the `acos` the pose path bans. The absolute value is the
        // double-cover — `q` and `−q` are the same rotation.
        cost += jw * w.rotation * (1.0 - qa.dot(qb).abs());
        if let (Some(qv), Some(cv)) = (query_velocity, candidate_velocity) {
            if let (Some(a), Some(b)) = (qv.get(i), cv.get(i)) {
                cost += jw
                    * w.velocity
                    * (Vec3::from_array(*b) - Vec3::from_array(*a)).length_squared();
            }
        }
    }
    cost
}

/// The best frame of one clip.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseMatch {
    /// Index into the candidate list handed to [`match_clips`]; always `0` from
    /// [`match_clip`].
    pub candidate: usize,
    /// The clip time, seconds.
    pub time_s: f32,
    /// The cost at that time.
    pub cost: f32,
}

/// Scan `clip` at `samples` evenly-spaced times and return the cheapest.
///
/// `samples` is clamped to at least 2, and the scan **includes both endpoints**,
/// so a one-shot clip's first and last frames are always candidates. Ties go to
/// the earliest time: a match is a place to *start playing from*, and starting
/// earlier is the answer that has more clip left.
///
/// `None` when the clip has no usable duration or the skeleton is empty.
pub fn match_clip(
    skeleton: &Skeleton,
    query: &PoseSnapshot,
    clip: &AnimClip,
    samples: usize,
    w: &PoseMatchWeights,
) -> Option<PoseMatch> {
    if skeleton.is_empty() || !crate::positive(clip.duration) {
        return None;
    }
    let samples = samples.max(2);
    let dur = clip.duration;
    // The central-difference interval for the candidate's velocity: one 30 Hz
    // frame, which is the rate the reference content is authored at and is small
    // enough not to smear a foot-plant.
    const VEL_DT: f32 = 1.0 / 30.0;
    let want_velocity = query.velocity.is_some() && w.velocity != 0.0;
    let mut best: Option<PoseMatch> = None;
    for k in 0..samples {
        let t = dur * (k as f32) / ((samples - 1) as f32);
        let pose = sample_clip(skeleton, clip, t, true);
        let vel = want_velocity.then(|| {
            let before = sample_clip(skeleton, clip, t - VEL_DT, true);
            let after = sample_clip(skeleton, clip, t + VEL_DT, true);
            model_velocity(skeleton, &before, &after, 2.0 * VEL_DT)
        });
        let cost = pose_cost(
            skeleton,
            &query.pose,
            query.velocity.as_deref(),
            &pose,
            vel.as_deref(),
            w,
        );
        if !cost.is_finite() {
            continue;
        }
        // `<` and not `<=`: ties keep the earlier time.
        if best.as_ref().is_none_or(|b| cost < b.cost) {
            best = Some(PoseMatch {
                candidate: 0,
                time_s: t,
                cost,
            });
        }
    }
    best
}

/// The best `(clip, time)` across several candidates.
///
/// Ties go to the earlier candidate index, so the caller's order is the
/// preference order and the answer does not depend on anything else.
pub fn match_clips(
    skeleton: &Skeleton,
    query: &PoseSnapshot,
    clips: &[&AnimClip],
    samples: usize,
    w: &PoseMatchWeights,
) -> Option<PoseMatch> {
    let mut best: Option<PoseMatch> = None;
    for (i, c) in clips.iter().enumerate() {
        let Some(mut m) = match_clip(skeleton, query, c, samples, w) else {
            continue;
        };
        m.candidate = i;
        if best.as_ref().is_none_or(|b| m.cost < b.cost) {
            best = Some(m);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Interpolation, JointTrack, QuatTrack, Vec3Track};
    use crate::skeleton::{Joint, JointTransform};
    use glam::Quat;

    /// A 3-joint chain along +Y, so a rotation at joint 1 moves joint 2 in model
    /// space and the position term has something to price.
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

    /// A clip swinging joint 1 through a full `0 → deg → 0` cycle over `dur`.
    fn swing(name: &str, deg: f32, dur: f32) -> AnimClip {
        let far = inf_math::pslerp(
            Quat::IDENTITY,
            Quat::from_xyzw(0.0, 0.0, 1.0, 0.0),
            deg / 180.0,
        );
        let mut jt = JointTrack::new(1);
        jt.rotation = Some(QuatTrack::new(
            vec![0.0, dur * 0.5, dur],
            vec![
                Quat::IDENTITY.to_array(),
                far.to_array(),
                Quat::IDENTITY.to_array(),
            ],
            Interpolation::Linear,
        ));
        AnimClip::new(name, vec![jt])
    }

    fn pose_at(sk: &Skeleton, clip: &AnimClip, t: f32) -> Pose {
        sample_clip(sk, clip, t, true)
    }

    #[test]
    fn a_snapshot_captures_the_pose_and_optionally_its_velocity() {
        let sk = chain();
        let clip = swing("s", 90.0, 1.0);
        let a = pose_at(&sk, &clip, 0.2);
        let b = pose_at(&sk, &clip, 0.216);
        let still = PoseSnapshot::capture("still", &b, 0.216);
        assert!(still.velocity.is_none());
        assert_eq!(still.pose, b);
        assert_eq!(still.label, "still");

        let moving = PoseSnapshot::capture_moving("moving", &sk, &b, &a, 0.016, 0.216);
        let v = moving.velocity.expect("a moving snapshot carries velocity");
        assert_eq!(v.len(), 3);
        assert!(
            Vec3::from_array(v[2]).length() > 0.1,
            "the tip should be moving: {:?}",
            v[2]
        );
        // Joint 0 is the root and never moves.
        assert!(Vec3::from_array(v[0]).length() < 1e-6);
        // A non-positive dt yields no velocity rather than an infinity.
        for bad in [0.0f32, -0.016, f32::NAN] {
            assert!(PoseSnapshot::capture_moving("x", &sk, &b, &a, bad, 0.0)
                .velocity
                .is_none());
        }
    }

    #[test]
    fn the_cost_of_a_pose_against_itself_is_zero_and_grows_with_the_error() {
        let sk = chain();
        let clip = swing("s", 90.0, 1.0);
        let w = PoseMatchWeights::default();
        let a = pose_at(&sk, &clip, 0.25);
        assert_eq!(pose_cost(&sk, &a, None, &a, None, &w), 0.0);
        let mut last = 0.0;
        for t in [0.26f32, 0.30, 0.40, 0.50] {
            let c = pose_cost(&sk, &a, None, &pose_at(&sk, &clip, t), None, &w);
            assert!(c > last, "cost did not grow at t={t}: {c} after {last}");
            last = c;
        }
    }

    /// A mask really confines the cost: with the arm chain masked out, two poses
    /// that differ **only** there cost nothing.
    #[test]
    fn a_mask_confines_what_the_cost_can_see() {
        let sk = chain();
        let clip = swing("s", 90.0, 1.0);
        let a = pose_at(&sk, &clip, 0.0);
        let b = pose_at(&sk, &clip, 0.5);
        let plain = PoseMatchWeights::default();
        assert!(pose_cost(&sk, &a, None, &b, None, &plain) > 0.1);
        // Joint 1 and its descendant are the only difference; mask them out.
        let masked = PoseMatchWeights::default()
            .with_mask(JointMask::new("root-only", [(0u16, 1.0)], 0.0));
        assert_eq!(pose_cost(&sk, &a, None, &b, None, &masked), 0.0);
    }

    /// **Matching finds the frame that was captured**, which is the property the
    /// P29.4 consumers rest on.
    #[test]
    fn matching_recovers_the_frame_a_snapshot_was_taken_at() {
        let sk = chain();
        let clip = swing("s", 120.0, 1.0);
        let w = PoseMatchWeights::default();
        for want in [0.1f32, 0.35, 0.6, 0.85] {
            let q = PoseSnapshot::capture("q", &pose_at(&sk, &clip, want), want as f64);
            let m = match_clip(&sk, &q, &clip, 101, &w).expect("a match");
            // The swing is symmetric about its midpoint, so a *shape* match has
            // two answers and the earlier one wins. That is the documented
            // tie-break, and it is why the velocity term exists.
            let mirror = clip.duration - want;
            assert!(
                (m.time_s - want).abs() < 0.02 || (m.time_s - mirror).abs() < 0.02,
                "wanted {want} (or its mirror {mirror}), got {}",
                m.time_s
            );
            assert!(m.cost < 1e-3, "cost {} at {}", m.cost, m.time_s);
        }
    }

    /// **Velocity breaks the mirror.** A symmetric swing has two frames with the
    /// same shape and opposite motion; a shape-only match takes the earlier, and a
    /// match that prices velocity takes the right one.
    ///
    /// This is the arm that makes the velocity term non-decorative: dropping it
    /// (weight 0) puts the answer on the wrong side of the swing.
    #[test]
    fn the_velocity_term_tells_the_two_halves_of_a_swing_apart() {
        let sk = chain();
        let clip = swing("s", 120.0, 1.0);
        // A query taken on the RETURN half (t = 0.7, the joint swinging back).
        let want = 0.7f32;
        let dt = 1.0 / 60.0;
        let q = PoseSnapshot::capture_moving(
            "q",
            &sk,
            &pose_at(&sk, &clip, want),
            &pose_at(&sk, &clip, want - dt),
            dt,
            want as f64,
        );
        let with_v = match_clip(&sk, &q, &clip, 201, &PoseMatchWeights::default()).unwrap();
        assert!(
            (with_v.time_s - want).abs() < 0.03,
            "velocity-aware match landed at {}",
            with_v.time_s
        );
        // Shape only: the mirror frame on the outward half wins, because ties go
        // to the earlier time.
        let shape_only = PoseMatchWeights {
            velocity: 0.0,
            ..Default::default()
        };
        let without_v = match_clip(&sk, &q, &clip, 201, &shape_only).unwrap();
        assert!(
            (without_v.time_s - (clip.duration - want)).abs() < 0.03,
            "shape-only match landed at {} rather than the mirror {}",
            without_v.time_s,
            clip.duration - want
        );
    }

    #[test]
    fn matching_across_candidates_picks_the_right_clip() {
        let sk = chain();
        let small = swing("small", 20.0, 1.0);
        let big = swing("big", 120.0, 1.0);
        let w = PoseMatchWeights::default();
        // A query taken from the BIG swing at its extreme cannot be matched by the
        // small one at any time.
        let q = PoseSnapshot::capture("q", &pose_at(&sk, &big, 0.5), 0.5);
        let m = match_clips(&sk, &q, &[&small, &big], 61, &w).unwrap();
        assert_eq!(m.candidate, 1, "picked the wrong clip: {m:?}");
        assert!((m.time_s - 0.5).abs() < 0.03, "{m:?}");
        // The other way round is **not** symmetric, and it is worth saying why
        // rather than asserting the pleasing answer: the small swing's every pose
        // also occurs inside the big one, so both candidates reach cost ~0 and the
        // documented tie-break — the earlier candidate — decides. The caller's
        // order is therefore the preference order, which is the contract.
        let q = PoseSnapshot::capture("q", &pose_at(&sk, &small, 0.5), 0.5);
        let m = match_clips(&sk, &q, &[&big, &small], 61, &w).unwrap();
        assert_eq!(m.candidate, 0, "the tie did not go to the earlier candidate");
        assert!(m.cost < 1e-3, "{m:?}");
        assert_eq!(
            match_clips(&sk, &q, &[&small, &big], 61, &w)
                .unwrap()
                .candidate,
            0,
            "reversing the preference order should reverse the winner"
        );
    }

    #[test]
    fn a_clip_with_no_duration_or_an_empty_rig_has_no_match() {
        let sk = chain();
        let q = PoseSnapshot::capture("q", &Pose::rest(&sk), 0.0);
        let empty = AnimClip::new("empty", vec![JointTrack::new(0)]);
        assert_eq!(empty.duration, 0.0);
        assert!(match_clip(&sk, &q, &empty, 8, &PoseMatchWeights::default()).is_none());
        assert!(match_clips(&sk, &q, &[], 8, &PoseMatchWeights::default()).is_none());
        // A translation-only clip still matches — the position term is what it is
        // for.
        let mut jt = JointTrack::new(0);
        jt.translation = Some(Vec3Track::new(
            vec![0.0, 1.0],
            vec![[0.0; 3], [2.0, 0.0, 0.0]],
            Interpolation::Linear,
        ));
        let slide = AnimClip::new("slide", vec![jt]);
        let target = pose_at(&sk, &slide, 0.75);
        let q = PoseSnapshot::capture("q", &target, 0.75);
        let m = match_clip(&sk, &q, &slide, 101, &PoseMatchWeights::default()).unwrap();
        assert!((m.time_s - 0.75).abs() < 0.02, "{m:?}");
    }
}
