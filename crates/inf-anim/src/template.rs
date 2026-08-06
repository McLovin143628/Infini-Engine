//! **Template body plans** (P24.1): a parametric N-pedal skeleton generator.
//!
//! A new character stops being painful at the point where you can *get a rig*
//! without modelling one. [`build_template`] turns a [`BodyPlan`] plus a small
//! set of named proportions into a validated [`SkeletonAsset`] — joints in
//! topological order, inverse binds, standard sockets, and (P24.2's input) joint
//! limits for the hinges.
//!
//! # Humanoid names are the retarget contract
//!
//! A [`BodyPlan::Biped`] emits **exactly** the nineteen names
//! [`crate::retarget::humanoid_joint_names`] lists, in that vocabulary, so
//! retarget v1 works between two generated bipeds — and between a generated biped
//! and any imported rig that follows the convention — with
//! [`RetargetMap::humanoid_identity`](crate::retarget::RetargetMap::humanoid_identity)
//! and no manual pairing. `biped_retargets_onto_itself` is what keeps that true;
//! it is a *contract* test, not a smoke test, because renaming one joint here
//! silently breaks every retarget the convention was for.
//!
//! Extra spine or neck segments are *inserted* into the chain
//! (`spine → spine_1 → … → chest`), never renamed over the canonical five, so a
//! long-backed creature still retargets its hips, chest, neck and head.
//!
//! # Determinism
//!
//! Every proportion is `f64`, every emitted quantity is `f32`, and the conversion
//! happens once at the wire. **No trigonometry**: a bind pose is built from axis
//! offsets alone, so nothing here can trip the P14 law about `sin`/`cos` not being
//! bit-portable — and the inverse binds are computed as the exact negation of an
//! accumulated translation rather than by inverting a matrix, because a bind pose
//! with identity rotation and unit scale has a closed-form inverse and a general
//! `Mat4::inverse` would put a divide on a path that does not need one.
//!
//! Two generations from the same parameters produce byte-identical payloads
//! (`the_same_params_generate_the_same_bytes`).
//!
//! # Refusals are values
//!
//! Degenerate parameters — a zero height, a NaN ratio, a hip girdle above the
//! shoulders, an odd leg count — return a named [`TemplateError`] rather than
//! panicking or clamping. Clamping would be worse than either: it produces a rig
//! that is *almost* what was asked for and gives no signal that it is not.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::asset::SkeletonAsset;
use crate::skeleton::{Joint, JointTransform, Skeleton, SkeletonError};
use crate::sockets::Socket;

/// Which body plan a template generates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyPlan {
    /// Two legs + two arms + a head: the humanoid, named with the canonical
    /// nineteen (see the module docs).
    Biped,
    /// Four legs on two girdles, no arms.
    Quadruped,
    /// Six legs on three girdles, no arms.
    Hexapod,
    /// `legs` legs on `legs / 2` girdles, no arms. `Quadruped` is
    /// `Npedal { legs: 4 }` by construction; the named variants exist because
    /// they are what a template library offers.
    Npedal {
        /// Total leg count. **Even**, at least 2, at most [`MAX_LEGS`].
        legs: u16,
    },
}

impl BodyPlan {
    /// Total leg count for this plan.
    pub fn legs(self) -> u16 {
        match self {
            BodyPlan::Biped => 2,
            BodyPlan::Quadruped => 4,
            BodyPlan::Hexapod => 6,
            BodyPlan::Npedal { legs } => legs,
        }
    }

    /// Whether this plan grows arms (only the biped does).
    pub fn has_arms(self) -> bool {
        matches!(self, BodyPlan::Biped)
    }
}

/// The most legs a template will generate.
///
/// Not a technical limit (a skeleton addresses 65 536 joints) but a *sanity*
/// one: past this a "body plan" is a different kind of object — a centipede is
/// authored procedurally, not from a proportions dialog — and an unbounded count
/// makes a typo in a UI field into a multi-megabyte asset.
pub const MAX_LEGS: u16 = 64;

/// Named proportions a template is shaped by. Every length is **metres** and
/// every ratio is a **fraction of `height_m`** (SI, per the units doctrine).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyParams {
    /// Total standing height, ground to the top of the head (m).
    pub height_m: f64,
    /// Height of the hip girdle, as a fraction of `height_m`.
    pub hip_height_ratio: f64,
    /// Height of the shoulder girdle (the top of the spine chain), as a fraction
    /// of `height_m`.
    pub shoulder_height_ratio: f64,
    /// Height of the head joint (the top of the neck chain), as a fraction of
    /// `height_m`.
    pub head_height_ratio: f64,
    /// Number of joints between the hips and the shoulders, inclusive of `chest`.
    /// `2` is the canonical `spine → chest`; more inserts `spine_1 …` between.
    pub spine_segments: u16,
    /// Number of joints between the shoulders and the head, inclusive of `neck`.
    pub neck_segments: u16,
    /// Distance between the two shoulder joints (m). Also the arm girdle's width.
    pub shoulder_width_m: f64,
    /// Distance between a girdle's two hip joints (m).
    pub hip_width_m: f64,
    /// Fraction of a limb's length taken by its **upper** bone (the rest is the
    /// lower bone; the foot/hand sits at the end).
    pub upper_limb_ratio: f64,
    /// Arm length shoulder-to-wrist, as a fraction of `height_m`.
    pub arm_length_ratio: f64,
    /// Body length from the rearmost girdle to the shoulder girdle (m).
    /// Ignored by [`BodyPlan::Biped`], whose torso is vertical.
    pub body_length_m: f64,
    /// How far the head sits **forward** of the shoulders (m). Ignored by the
    /// biped, whose neck is vertical.
    pub head_forward_m: f64,
}

impl Default for BodyParams {
    /// An average adult human: 1.75 m, hips at 53 %, shoulders at 82 %, head at
    /// 93 %, a 40 cm shoulder span and a 22 cm hip span.
    ///
    /// The quadruped/hexapod-only fields carry a mid-sized four-legged animal's
    /// values so `BodyParams::default()` is a valid input for *every* plan — a
    /// default that is only valid for one of four plans is a trap.
    fn default() -> Self {
        Self {
            height_m: 1.75,
            hip_height_ratio: 0.53,
            shoulder_height_ratio: 0.82,
            head_height_ratio: 0.93,
            spine_segments: 2,
            neck_segments: 1,
            shoulder_width_m: 0.40,
            hip_width_m: 0.22,
            upper_limb_ratio: 0.52,
            arm_length_ratio: 0.42,
            body_length_m: 0.90,
            head_forward_m: 0.25,
        }
    }
}

/// Why a template refused to generate.
///
/// `PartialEq` but not `Eq`: [`InvertedBody`](Self::InvertedBody) reports the
/// f64 ratios it refused, so a caller can name them back to the user.
#[derive(Debug, Error, PartialEq)]
pub enum TemplateError {
    /// A parameter was NaN, infinite, or outside its documented range.
    #[error("parameter `{param}` is {value}, which is not a usable value ({why})")]
    BadParam {
        param: &'static str,
        value: String,
        why: &'static str,
    },
    /// The girdles are not in ascending height order.
    #[error(
        "the body's heights must ascend hips < shoulders <= head (got \
         {hip:.3} / {shoulder:.3} / {head:.3} as fractions of the total)"
    )]
    InvertedBody { hip: f64, shoulder: f64, head: f64 },
    /// An N-pedal body plan is bilaterally symmetric, so it needs an even count.
    #[error("a body plan needs an even leg count (got {0}); an odd leg has no symmetric girdle")]
    OddLegCount(u16),
    /// The leg count is outside `2 ..= MAX_LEGS`.
    #[error("leg count {0} is outside 2..={MAX_LEGS}")]
    LegCountOutOfRange(u16),
    /// The generated joint list failed [`Skeleton::new`] — a generator bug, kept
    /// as a value so it can never be a panic in a user's editor.
    #[error("the generated skeleton is invalid: {0}")]
    Invalid(#[from] SkeletonError),
}

/// A rotation limit on one joint, for IK (P24.2 consumes these).
///
/// Per-axis minimum/maximum rotation about the joint's **local** X / Y / Z, in
/// **degrees** — the engine's authoring convention for rotations (euler degrees
/// at the authoring boundary; a solver converts once, at its own edge).
///
/// A joint **absent** from the table is unlimited. That is the meaningful default:
/// a table that listed every joint with ±180° would be indistinguishable from an
/// unlimited rig while costing a row per joint, and a solver reading it could not
/// tell "no limit was authored" from "a full-range limit was authored".
///
/// **Frozen once shipped**: `SkeletonAsset` is bincode-positional, so these three
/// fields are the shape for good. A later limit *kind* (a swing-twist cone, a
/// preferred axis) is an append at the tail of this struct behind a schema bump,
/// not an edit of what is here.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointLimit {
    /// Index of the limited joint (into the skeleton's `joints`).
    pub joint: u16,
    /// Minimum rotation about local X / Y / Z, degrees.
    pub min_deg: [f32; 3],
    /// Maximum rotation about local X / Y / Z, degrees.
    pub max_deg: [f32; 3],
}

impl JointLimit {
    /// A **hinge**: free about local X between `min_deg`/`max_deg`, locked on the
    /// other two axes. Knees and elbows.
    pub fn hinge_x(joint: u16, min_deg: f32, max_deg: f32) -> Self {
        Self {
            joint,
            min_deg: [min_deg, 0.0, 0.0],
            max_deg: [max_deg, 0.0, 0.0],
        }
    }

    /// Whether this limit permits any rotation at all about `axis` (0 = X).
    pub fn is_free(&self, axis: usize) -> bool {
        self.min_deg.get(axis).copied().unwrap_or(0.0)
            < self.max_deg.get(axis).copied().unwrap_or(0.0)
    }
}

/// A knee's flexion range, degrees — the hinge a leg IK chain bends on.
const KNEE_RANGE_DEG: (f32, f32) = (-150.0, 0.0);
/// An elbow's flexion range, degrees.
const ELBOW_RANGE_DEG: (f32, f32) = (0.0, 150.0);

/// Build a validated [`SkeletonAsset`] for `plan` with `params`.
///
/// The joint list is topological **by construction** — every joint is pushed
/// after its parent — so [`Skeleton::new`] is a check rather than a sort, and a
/// failure there is a generator bug (surfaced as [`TemplateError::Invalid`], not a
/// panic).
///
/// Emits the standard sockets (`hand_l`/`hand_r` on a biped, `foot_*` on every
/// plan, `head`, `back`) and hinge limits on the knees and elbows.
pub fn build_template(plan: BodyPlan, params: &BodyParams) -> Result<SkeletonAsset, TemplateError> {
    validate(plan, params)?;
    let mut b = Builder::default();

    let h = params.height_m;
    let hip_y = h * params.hip_height_ratio;
    let shoulder_y = h * params.shoulder_height_ratio;
    let head_y = h * params.head_height_ratio;
    let legs = plan.legs();
    let girdles = (legs / 2) as usize;

    // ── the spine chain: hips → spine → (spine_i …) → chest ────────────────
    //
    // A biped's torso is vertical; a multi-girdle body's runs FORWARD (+Z) as
    // well, from the rearmost girdle to the shoulders. Both are pure axis
    // offsets — the run and the rise are separate components of one translation,
    // so nothing here needs an angle.
    let torso_run = if girdles > 1 {
        params.body_length_m
    } else {
        0.0
    };
    let root = b.push("hips", None, Vec3::new(0.0, hip_y as f32, 0.0));

    let spine_n = params.spine_segments as usize;
    let rise = (shoulder_y - hip_y) / spine_n as f64;
    let run = torso_run / spine_n as f64;
    let mut parent = root;
    for i in 0..spine_n {
        let name = if i + 1 == spine_n {
            "chest".to_string()
        } else if i == 0 {
            "spine".to_string()
        } else {
            format!("spine_{i}")
        };
        parent = b.push(&name, Some(parent), Vec3::new(0.0, rise as f32, run as f32));
    }
    let chest = parent;

    // ── the neck chain: neck → (neck_i …) → head ───────────────────────────
    let neck_n = params.neck_segments as usize;
    let neck_rise = (head_y - shoulder_y) / neck_n as f64;
    let neck_run = if girdles > 1 {
        params.head_forward_m / neck_n as f64
    } else {
        0.0
    };
    let mut parent = chest;
    for i in 0..neck_n {
        let name = if i == 0 {
            "neck".to_string()
        } else {
            format!("neck_{i}")
        };
        parent = b.push(
            &name,
            Some(parent),
            Vec3::new(0.0, neck_rise as f32, neck_run as f32),
        );
    }
    // The head joint sits at the top of the neck chain; the remaining height to
    // `height_m` is the skull, which a skeleton does not articulate.
    let head = b.push("head", Some(parent), Vec3::ZERO);

    // ── arms (biped only) ──────────────────────────────────────────────────
    let arm_len = h * params.arm_length_ratio;
    let upper_arm = arm_len * params.upper_limb_ratio;
    let lower_arm = arm_len - upper_arm;
    let mut hands: Vec<(String, u16)> = Vec::new();
    let mut limits: Vec<JointLimit> = Vec::new();
    if plan.has_arms() {
        for (side, sx) in [("l", -1.0f64), ("r", 1.0f64)] {
            let sh = b.push(
                &format!("shoulder_{side}"),
                Some(chest),
                Vec3::new((sx * params.shoulder_width_m * 0.5) as f32, 0.0, 0.0),
            );
            let ua = b.push(&format!("upper_arm_{side}"), Some(sh), Vec3::ZERO);
            let la = b.push(
                &format!("lower_arm_{side}"),
                Some(ua),
                Vec3::new(0.0, -upper_arm as f32, 0.0),
            );
            let hand = b.push(
                &format!("hand_{side}"),
                Some(la),
                Vec3::new(0.0, -lower_arm as f32, 0.0),
            );
            limits.push(JointLimit::hinge_x(
                la,
                ELBOW_RANGE_DEG.0,
                ELBOW_RANGE_DEG.1,
            ));
            hands.push((format!("hand_{side}"), hand));
        }
    }

    // ── legs, one girdle at a time ─────────────────────────────────────────
    //
    // Girdle 0 is the REARMOST (it is the hips), and the last girdle is at the
    // shoulders. A biped has one girdle and it is the hips, so its leg joints
    // take the canonical unindexed names.
    //
    // **A leg is as long as its own girdle is high** (P24.1 audit M3). The first
    // cut computed one `leg_len = hip_y` outside this loop, which is right only
    // for girdle 0: every other girdle sits at `gy > hip_y`, so its feet landed at
    // `gy − hip_y` — about half a metre in the air on a default quadruped, and
    // those are the `foot_*` sockets P24.2's IK will plant. Standing on the ground
    // is a property of each leg, not of the hips.
    let mut feet: Vec<(String, u16)> = Vec::new();
    for g in 0..girdles {
        // Girdles are spaced evenly from the hips to the shoulders along the run.
        let (gz, gy) = if girdles > 1 {
            let t = g as f64 / (girdles - 1) as f64;
            (torso_run * t, hip_y + (shoulder_y - hip_y) * t)
        } else {
            (0.0, hip_y)
        };
        let upper_leg = gy * params.upper_limb_ratio;
        let lower_leg = gy - upper_leg;
        // Attached to the root so a girdle's placement is independent of the
        // spine's segment count: the offset is measured from the hips.
        let anchor = b.push(
            &girdle_name(g, girdles),
            Some(root),
            Vec3::new(0.0, (gy - hip_y) as f32, gz as f32),
        );
        for (side, sx) in [("l", -1.0f64), ("r", 1.0f64)] {
            let suffix = leg_suffix(side, g, girdles);
            let ul = b.push(
                &format!("upper_leg_{suffix}"),
                Some(anchor),
                Vec3::new((sx * params.hip_width_m * 0.5) as f32, 0.0, 0.0),
            );
            let ll = b.push(
                &format!("lower_leg_{suffix}"),
                Some(ul),
                Vec3::new(0.0, -upper_leg as f32, 0.0),
            );
            let foot = b.push(
                &format!("foot_{suffix}"),
                Some(ll),
                Vec3::new(0.0, -lower_leg as f32, 0.0),
            );
            limits.push(JointLimit::hinge_x(ll, KNEE_RANGE_DEG.0, KNEE_RANGE_DEG.1));
            feet.push((format!("foot_{suffix}"), foot));
        }
    }

    // ── the standard sockets ───────────────────────────────────────────────
    let mut sockets: Vec<Socket> = Vec::new();
    for (name, joint) in hands.into_iter().chain(feet) {
        sockets.push(Socket::new(name, joint));
    }
    sockets.push(Socket::new("head", head));
    sockets.push(Socket::new("back", chest));

    let skeleton = Skeleton::new(b.joints)?;
    let mut asset = SkeletonAsset::with_sockets(skeleton, sockets);
    asset.limits = limits;
    Ok(asset)
}

/// The name of girdle `g` of `girdles`. A single-girdle body (the biped) has no
/// girdle joint name to spare from the canonical vocabulary, so its legs hang off
/// a `pelvis` node that is *not* one of the nineteen — the nineteen name the
/// joints an animation drives, and a girdle anchor is structure.
fn girdle_name(g: usize, girdles: usize) -> String {
    if girdles == 1 {
        "pelvis".to_string()
    } else {
        format!("girdle_{g}")
    }
}

/// The `{side}` / `{side}{girdle}` suffix a leg's joints carry. A single-girdle
/// body uses the canonical unindexed `l` / `r` so a biped's legs are exactly
/// `upper_leg_l`, `lower_leg_l`, `foot_l`, … — the retarget vocabulary.
fn leg_suffix(side: &str, g: usize, girdles: usize) -> String {
    if girdles == 1 {
        side.to_string()
    } else {
        format!("{side}{g}")
    }
}

/// Accumulates joints in topological order, tracking each one's **global bind
/// position** so the inverse bind is an exact negated translation rather than a
/// matrix inversion.
#[derive(Default)]
struct Builder {
    joints: Vec<Joint>,
    /// Global bind position of each pushed joint, index-aligned to `joints`.
    globals: Vec<Vec3>,
}

impl Builder {
    /// Push a joint whose local bind is a pure translation of `local` from
    /// `parent`. Returns its index.
    fn push(&mut self, name: &str, parent: Option<u16>, local: Vec3) -> u16 {
        let base = parent
            .and_then(|p| self.globals.get(p as usize).copied())
            .unwrap_or(Vec3::ZERO);
        let global = base + local;
        let index = self.joints.len() as u16;
        self.joints.push(Joint {
            name: name.to_string(),
            parent,
            // Identity rotation + unit scale everywhere, so the bind's inverse is
            // the inverse translation exactly — no divide, no matrix inversion,
            // bit-identical on every platform.
            inverse_bind: glam::Mat4::from_translation(-global).to_cols_array(),
            local_bind: JointTransform::from_trs(local, Quat::IDENTITY, Vec3::ONE),
        });
        self.globals.push(global);
        index
    }
}

/// Reject degenerate parameters **as values**.
fn validate(plan: BodyPlan, p: &BodyParams) -> Result<(), TemplateError> {
    let legs = plan.legs();
    if !legs.is_multiple_of(2) {
        return Err(TemplateError::OddLegCount(legs));
    }
    if !(2..=MAX_LEGS).contains(&legs) {
        return Err(TemplateError::LegCountOutOfRange(legs));
    }
    let positive = |param: &'static str, v: f64| -> Result<(), TemplateError> {
        if !v.is_finite() || v <= 0.0 {
            return Err(TemplateError::BadParam {
                param,
                value: format!("{v}"),
                why: "must be a finite length greater than zero",
            });
        }
        Ok(())
    };
    let fraction = |param: &'static str, v: f64| -> Result<(), TemplateError> {
        if !v.is_finite() || v <= 0.0 || v >= 1.0 {
            return Err(TemplateError::BadParam {
                param,
                value: format!("{v}"),
                why: "must be a finite fraction strictly between 0 and 1",
            });
        }
        Ok(())
    };
    positive("height_m", p.height_m)?;
    positive("shoulder_width_m", p.shoulder_width_m)?;
    positive("hip_width_m", p.hip_width_m)?;
    fraction("hip_height_ratio", p.hip_height_ratio)?;
    fraction("shoulder_height_ratio", p.shoulder_height_ratio)?;
    fraction("head_height_ratio", p.head_height_ratio)?;
    fraction("upper_limb_ratio", p.upper_limb_ratio)?;
    fraction("arm_length_ratio", p.arm_length_ratio)?;
    // Multi-girdle bodies measure along their own length; a biped ignores both,
    // so a zero is only a refusal where it is read.
    if plan.legs() > 2 {
        positive("body_length_m", p.body_length_m)?;
        if !p.head_forward_m.is_finite() || p.head_forward_m < 0.0 {
            return Err(TemplateError::BadParam {
                param: "head_forward_m",
                value: format!("{}", p.head_forward_m),
                why: "must be a finite non-negative length",
            });
        }
    }
    if p.spine_segments < 2 {
        return Err(TemplateError::BadParam {
            param: "spine_segments",
            value: format!("{}", p.spine_segments),
            why: "a spine needs at least two joints (`spine` and `chest`)",
        });
    }
    if p.neck_segments < 1 {
        return Err(TemplateError::BadParam {
            param: "neck_segments",
            value: format!("{}", p.neck_segments),
            why: "a neck needs at least one joint",
        });
    }
    if !(p.hip_height_ratio < p.shoulder_height_ratio
        && p.shoulder_height_ratio <= p.head_height_ratio)
    {
        return Err(TemplateError::InvertedBody {
            hip: p.hip_height_ratio,
            shoulder: p.shoulder_height_ratio,
            head: p.head_height_ratio,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::{global_transforms, Pose};
    use crate::retarget::{humanoid_joint_names, retarget_pose, RetargetMap};

    fn biped() -> SkeletonAsset {
        build_template(BodyPlan::Biped, &BodyParams::default()).expect("a biped generates")
    }

    /// **Every template validates.** `Skeleton::new` is the contract: joints in
    /// topological order, parents in range.
    #[test]
    fn every_plan_generates_a_valid_skeleton() {
        for plan in [
            BodyPlan::Biped,
            BodyPlan::Quadruped,
            BodyPlan::Hexapod,
            BodyPlan::Npedal { legs: 8 },
            BodyPlan::Npedal { legs: MAX_LEGS },
        ] {
            let asset = build_template(plan, &BodyParams::default())
                .unwrap_or_else(|e| panic!("{plan:?} refused: {e}"));
            let sk = &asset.skeleton;
            // Re-validating through the public door proves the invariant rather
            // than trusting the builder that produced it.
            Skeleton::new(sk.joints().to_vec()).expect("re-validates");
            assert_eq!(
                sk.joints().iter().filter(|j| j.parent.is_none()).count(),
                1,
                "{plan:?} has more than one root"
            );
            // One upper/lower/foot per leg.
            let feet = sk
                .joints()
                .iter()
                .filter(|j| j.name.starts_with("foot_"))
                .count();
            assert_eq!(feet, plan.legs() as usize, "{plan:?} foot count");
        }
    }

    /// **The retarget contract.** A generated biped carries exactly the canonical
    /// nineteen, so `RetargetMap::humanoid_identity` maps it with no manual
    /// pairing — and retargeting it onto itself reproduces the pose.
    #[test]
    fn biped_retargets_onto_itself() {
        let asset = biped();
        let sk = &asset.skeleton;
        for name in humanoid_joint_names() {
            assert!(
                sk.index_of(name).is_some(),
                "the canonical humanoid joint `{name}` is missing — retarget v1 \
                 would silently skip it"
            );
        }
        // A pose with a bent elbow and a displaced root: the two things retarget
        // v1 actually copies.
        let mut pose = Pose::rest(sk);
        let elbow = sk.index_of("lower_arm_r").unwrap() as usize;
        pose.locals[elbow].rotation = Quat::from_rotation_x(0.7).to_array();
        pose.locals[0].translation = [0.3, 0.0, 0.1];

        let out = retarget_pose(sk, &pose, sk, &RetargetMap::humanoid_identity());
        for name in humanoid_joint_names() {
            let i = sk.index_of(name).unwrap() as usize;
            let a = Quat::from_array(out.locals[i].rotation).normalize();
            let b = Quat::from_array(pose.locals[i].rotation).normalize();
            assert!(a.angle_between(b) < 1e-4, "joint `{name}` drifted");
        }
        assert!(
            (Vec3::from_array(out.locals[0].translation) - Vec3::new(0.3, 0.0, 0.1)).length()
                < 1e-5
        );
    }

    /// The standard sockets are present and ride the joints they name.
    #[test]
    fn standard_sockets_are_present_and_ride_their_joints() {
        let asset = biped();
        let names: Vec<&str> = asset.sockets.iter().map(|s| s.name.as_str()).collect();
        for want in ["hand_l", "hand_r", "foot_l", "foot_r", "head", "back"] {
            assert!(
                names.contains(&want),
                "socket `{want}` missing from {names:?}"
            );
        }
        // A socket's joint is the joint of the same name where one exists (the
        // limb sockets), so a weapon attached to `hand_r` really rides the hand.
        for s in &asset.sockets {
            if let Some(j) = asset.skeleton.index_of(&s.name) {
                assert_eq!(s.joint, j, "socket `{}` rides the wrong joint", s.name);
            }
        }
        // Every socket indexes a real joint.
        for s in &asset.sockets {
            assert!(
                (s.joint as usize) < asset.skeleton.len(),
                "socket `{}` is out of range",
                s.name
            );
        }

        // A quadruped's feet are indexed per girdle, and each one exists.
        let quad = build_template(BodyPlan::Quadruped, &BodyParams::default()).unwrap();
        for want in ["foot_l0", "foot_r0", "foot_l1", "foot_r1", "head", "back"] {
            assert!(
                quad.sockets.iter().any(|s| s.name == want),
                "quadruped socket `{want}` missing"
            );
        }
    }

    /// The rest pose is the shape the parameters asked for: feet on the ground,
    /// head at the head height, hands hanging at the wrist line.
    #[test]
    fn the_rest_pose_has_the_proportions_it_was_given() {
        let p = BodyParams::default();
        let asset = biped();
        let sk = &asset.skeleton;
        let globals = global_transforms(sk, &Pose::rest(sk));
        let at =
            |name: &str| globals[sk.index_of(name).unwrap() as usize].transform_point3(Vec3::ZERO);
        assert!(
            at("foot_l").y.abs() < 1e-4,
            "feet on the ground: {}",
            at("foot_l").y
        );
        assert!(at("foot_r").y.abs() < 1e-4);
        assert!(
            (at("head").y as f64 - p.height_m * p.head_height_ratio).abs() < 1e-4,
            "head at {}",
            at("head").y
        );
        assert!((at("hips").y as f64 - p.height_m * p.hip_height_ratio).abs() < 1e-4);
        // Left is −X and right is +X, and they are a shoulder-width apart.
        assert!(at("shoulder_l").x < 0.0 && at("shoulder_r").x > 0.0);
        assert!(
            ((at("shoulder_r").x - at("shoulder_l").x) as f64 - p.shoulder_width_m).abs() < 1e-4
        );
        // The rest pose's skinning palette is the identity, which is what makes a
        // freshly generated rig draw its mesh unchanged.
        for m in crate::pose::skinning_matrices(sk, &Pose::rest(sk)) {
            assert!(m.abs_diff_eq(glam::Mat4::IDENTITY, 1e-5), "{m:?}");
        }
    }

    /// **EVERY plan stands on the ground** (P24.1 audit M3).
    ///
    /// The proportions test above builds only `biped()` — the one plan whose
    /// single girdle sits at the hips, and therefore the one plan that could not
    /// fail. A multi-girdle body puts each girdle at its own height, and a leg
    /// sized from the hips instead of from its own girdle leaves that girdle's
    /// feet in the air: about half a metre on a default quadruped, on the very
    /// sockets P24.2's IK will plant.
    ///
    /// Computed independently, out of `global_transforms`, over every foot joint
    /// of every plan the drawer offers plus an arbitrary N-pedal.
    #[test]
    fn every_plan_stands_its_feet_on_the_ground() {
        for plan in [
            BodyPlan::Biped,
            BodyPlan::Quadruped,
            BodyPlan::Hexapod,
            BodyPlan::Npedal { legs: 10 },
        ] {
            let asset = build_template(plan, &BodyParams::default()).unwrap();
            let sk = &asset.skeleton;
            let globals = global_transforms(sk, &Pose::rest(sk));
            let feet: Vec<(usize, f32)> = sk
                .joints()
                .iter()
                .enumerate()
                .filter(|(_, j)| j.name.starts_with("foot_"))
                .map(|(i, _)| (i, globals[i].transform_point3(Vec3::ZERO).y))
                .collect();
            assert_eq!(
                feet.len(),
                plan.legs() as usize,
                "{plan:?} did not grow one foot per leg"
            );
            for (i, y) in &feet {
                assert!(
                    y.abs() < 1e-4,
                    "{plan:?} foot `{}` floats at y = {y}",
                    sk.joints()[*i].name
                );
            }
            // …and the girdles really are at DIFFERENT heights, or this test is
            // measuring one girdle repeated and would pass on the defect.
            if plan.legs() > 2 {
                let hips = globals[sk.index_of("hips").unwrap() as usize]
                    .transform_point3(Vec3::ZERO)
                    .y;
                let last = globals[sk
                    .index_of(&format!("girdle_{}", plan.legs() / 2 - 1))
                    .unwrap() as usize]
                    .transform_point3(Vec3::ZERO)
                    .y;
                assert!(
                    (last - hips) > 0.1,
                    "{plan:?}'s girdles are all at one height, so a hips-sized leg                      would have reached the ground and this gate proves nothing"
                );
            }
        }
    }

    /// Deterministic: two generations from the same parameters are byte-identical
    /// through the asset codec.
    #[test]
    fn the_same_params_generate_the_same_bytes() {
        for plan in [BodyPlan::Biped, BodyPlan::Hexapod] {
            let a = build_template(plan, &BodyParams::default()).unwrap();
            let b = build_template(plan, &BodyParams::default()).unwrap();
            assert_eq!(a, b);
            let (ea, eb) = (
                inf_asset::encode(&a).unwrap(),
                inf_asset::encode(&b).unwrap(),
            );
            assert_eq!(ea, eb, "{plan:?} is not byte-reproducible");
            // …and it round-trips, limits included.
            let back: SkeletonAsset = inf_asset::decode(&ea).unwrap();
            assert_eq!(back, a);
            assert!(!back.limits.is_empty());
        }
        // A *different* parameter set produces different bytes — without this the
        // assertion above would pass for a generator that ignored its input.
        let tall = BodyParams {
            height_m: 2.4,
            ..BodyParams::default()
        };
        assert_ne!(
            inf_asset::encode(&build_template(BodyPlan::Biped, &tall).unwrap()).unwrap(),
            inf_asset::encode(&biped()).unwrap()
        );
    }

    /// Knees and elbows carry hinge limits; nothing else is listed, because an
    /// absent joint means "unlimited" and a full-range row would be noise.
    #[test]
    fn hinges_carry_limits_and_nothing_else_does() {
        let asset = biped();
        let sk = &asset.skeleton;
        let limited: Vec<&str> = asset
            .limits
            .iter()
            .map(|l| sk.joints()[l.joint as usize].name.as_str())
            .collect();
        assert_eq!(limited.len(), 4, "{limited:?}");
        for want in ["lower_arm_l", "lower_arm_r", "lower_leg_l", "lower_leg_r"] {
            assert!(limited.contains(&want), "{want} unlimited: {limited:?}");
        }
        // A hinge is free about X and locked on Y/Z.
        let elbow = sk.index_of("lower_arm_r").unwrap();
        let l = asset.limits.iter().find(|l| l.joint == elbow).unwrap();
        assert!(l.is_free(0) && !l.is_free(1) && !l.is_free(2));

        // A quadruped limits every knee it has, and no arms (it has none).
        let quad = build_template(BodyPlan::Quadruped, &BodyParams::default()).unwrap();
        assert_eq!(quad.limits.len(), 4, "one hinge per leg");
        assert!(quad.skeleton.index_of("hand_l").is_none());
    }

    /// **Degenerate parameters are refused as VALUES.** Every arm is named, so a
    /// UI can say which field is wrong rather than "invalid".
    #[test]
    fn degenerate_params_are_refused() {
        let bad = |p: BodyParams| build_template(BodyPlan::Biped, &p).unwrap_err();

        assert!(matches!(
            bad(BodyParams {
                height_m: 0.0,
                ..Default::default()
            }),
            TemplateError::BadParam {
                param: "height_m",
                ..
            }
        ));
        assert!(matches!(
            bad(BodyParams {
                height_m: f64::NAN,
                ..Default::default()
            }),
            TemplateError::BadParam { .. }
        ));
        assert!(matches!(
            bad(BodyParams {
                upper_limb_ratio: 1.0,
                ..Default::default()
            }),
            TemplateError::BadParam {
                param: "upper_limb_ratio",
                ..
            }
        ));
        assert!(matches!(
            bad(BodyParams {
                spine_segments: 1,
                ..Default::default()
            }),
            TemplateError::BadParam {
                param: "spine_segments",
                ..
            }
        ));
        // Hips above the shoulders is a body, not a parameter, so it has its own
        // arm.
        assert!(matches!(
            bad(BodyParams {
                hip_height_ratio: 0.9,
                ..Default::default()
            }),
            TemplateError::InvertedBody { .. }
        ));
        // Leg counts.
        assert_eq!(
            build_template(BodyPlan::Npedal { legs: 5 }, &BodyParams::default()).unwrap_err(),
            TemplateError::OddLegCount(5)
        );
        assert_eq!(
            build_template(
                BodyPlan::Npedal { legs: MAX_LEGS + 2 },
                &BodyParams::default()
            )
            .unwrap_err(),
            TemplateError::LegCountOutOfRange(MAX_LEGS + 2)
        );
        assert_eq!(
            build_template(BodyPlan::Npedal { legs: 0 }, &BodyParams::default()).unwrap_err(),
            TemplateError::LegCountOutOfRange(0)
        );
        // A multi-girdle body reads `body_length_m`, so a zero there is refused —
        // and a biped ignores it, so the same value is fine.
        assert!(build_template(
            BodyPlan::Biped,
            &BodyParams {
                body_length_m: 0.0,
                ..Default::default()
            }
        )
        .is_ok());
        assert!(matches!(
            build_template(
                BodyPlan::Quadruped,
                &BodyParams {
                    body_length_m: 0.0,
                    ..Default::default()
                }
            )
            .unwrap_err(),
            TemplateError::BadParam {
                param: "body_length_m",
                ..
            }
        ));
    }

    /// Extra spine/neck segments are INSERTED, never renamed over the canonical
    /// five — so a long-backed creature still retargets.
    #[test]
    fn extra_segments_keep_the_canonical_names() {
        let asset = build_template(
            BodyPlan::Biped,
            &BodyParams {
                spine_segments: 5,
                neck_segments: 3,
                ..Default::default()
            },
        )
        .unwrap();
        let sk = &asset.skeleton;
        for name in ["hips", "spine", "spine_1", "spine_2", "spine_3", "chest"] {
            assert!(sk.index_of(name).is_some(), "{name} missing");
        }
        for name in ["neck", "neck_1", "neck_2", "head"] {
            assert!(sk.index_of(name).is_some(), "{name} missing");
        }
        // …and it is still one chain, in order.
        let chest = sk.index_of("chest").unwrap();
        let spine3 = sk.index_of("spine_3").unwrap();
        assert_eq!(sk.joint(chest as usize).unwrap().parent, Some(spine3));
        for name in humanoid_joint_names() {
            assert!(sk.index_of(name).is_some(), "{name} lost to a longer spine");
        }
    }
}
