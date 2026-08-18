//! Additive poses, per-bone masks and the layer stack (P29.2 clause 2).
//!
//! §13's P29.1 gave a cross-fade a per-joint [`BlendProfile`], which answers "the
//! upper body finishes this transition sooner than the legs". It does not answer
//! the other question, which is the one every overlay system in the catalogue
//! asks: *play a second animation on part of the body, on top of whatever the
//! machine is doing, without the machine knowing.* That is a **layer**, and it
//! needs three things this crate did not have:
//!
//! 1. **Additive poses** — the difference between two poses, applicable as a
//!    delta. A lean, an aim offset, a breath, a limp: authored once against a
//!    reference pose and added to *any* base, so it does not have to be
//!    re-authored per locomotion state. [`AdditiveRef`] is how
//!    a clip says what it is additive over; [`additive_delta`] and
//!    [`apply_additive`] are the arithmetic.
//! 2. **Per-bone masks** — [`JointMask`], a per-joint weight with an explicit
//!    `default_weight` for joints it does not mention. That explicitness is the
//!    whole difference from [`BlendProfile`]: a profile's absent joint means
//!    "blend at the transition's own alpha" (a *scale* on something else), while
//!    a mask's absent joint means whatever the mask says it means, and the two
//!    readings are not interchangeable. [`JointMask::from_profile`] converts, and
//!    is the seam an authored `.inf_sm` profile reaches a layer through.
//! 3. **A stack** — [`apply_layers`], folding layers over a base pose in order.
//!
//! # Not serialized, deliberately
//!
//! None of these types is a wire type. `.inf_anim` bumps **exactly once** (§13's
//! 2026-08-17 amendment, Ruling 2) and `.inf_sm` bumped in P29.1, so a persisted
//! layer stack would need a third schema this phase has no budget for — and the
//! layer set a character runs is a *runtime* composition (overlay state ×
//! equipment × damage), not a property of a clip. The persistence question
//! belongs with P29.4's routing, where there is something authored to persist.
//! Until then the stack is built in code, which is what the wizard and the
//! `anim.*` kit will do anyway.
//!
//! # Portability
//!
//! Every operation here is `lerp`, quaternion multiply, quaternion conjugate and
//! [`inf_math::pslerp`]. No transcendental, for the standing reason: a layered
//! pose is the pose, and the pose is folded into `state_bytes`.

use glam::{Quat, Vec3};

use crate::channels::AdditiveRef;
use crate::clip::AnimClip;
use crate::pose::{sample_clip, Pose};
use crate::skeleton::{JointTransform, Skeleton};
use crate::state_machine::BlendProfile;

/// A per-joint weight set, with an explicit weight for the joints it does not
/// mention.
///
/// `weights` is kept sorted by joint index (the constructors do it), so
/// [`resolve`](Self::resolve) is a merge rather than a search per joint and two
/// masks built from the same joints in different orders are the same mask.
#[derive(Clone, Debug, PartialEq)]
pub struct JointMask {
    /// A human name, for diagnostics and for the authoring UI.
    pub name: String,
    /// `(joint index, weight)`, ascending by joint, deduplicated (last wins).
    weights: Vec<(u16, f32)>,
    /// The weight of a joint this mask does not mention.
    pub default_weight: f32,
}

impl JointMask {
    /// A mask over the listed joints; every other joint gets `default_weight`.
    pub fn new(
        name: impl Into<String>,
        weights: impl IntoIterator<Item = (u16, f32)>,
        default_weight: f32,
    ) -> Self {
        let mut w: Vec<(u16, f32)> = weights.into_iter().collect();
        // Sort, then keep the LAST entry for a repeated joint: a later authored
        // row overriding an earlier one is the intuitive reading, and a mask that
        // silently kept the first would depend on iteration order.
        w.sort_by_key(|(j, _)| *j);
        w.dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 = a.1;
                true
            } else {
                false
            }
        });
        Self {
            name: name.into(),
            weights: w,
            default_weight,
        }
    }

    /// A mask that weights every joint the same — the "no mask" case, spelled so
    /// a caller never has to special-case `Option<JointMask>` twice.
    pub fn uniform(weight: f32) -> Self {
        Self::new("uniform", [], weight)
    }

    /// The rows, ascending by joint.
    pub fn rows(&self) -> &[(u16, f32)] {
        &self.weights
    }

    /// A mask covering `root` and **everything below it** in the skeleton, at
    /// `inside`; everything else at `outside`.
    ///
    /// This is the authoring primitive the overlay vocabulary actually wants —
    /// `Layering_Spine`, `Layering_Arm_L`, `Layering_Head` are all "this joint and
    /// its descendants". Written against the skeleton's topological order (a
    /// parent always precedes its children, which [`Skeleton::new`] guarantees and
    /// `SkeletonAsset::migrate` re-checks at the decode door), so
    /// one forward pass marks the whole subtree.
    pub fn from_subtree(
        name: impl Into<String>,
        skeleton: &Skeleton,
        root: u16,
        inside: f32,
        outside: f32,
    ) -> Self {
        let joints = skeleton.joints();
        let mut inside_set = vec![false; joints.len()];
        if let Some(slot) = inside_set.get_mut(root as usize) {
            *slot = true;
        }
        for (i, j) in joints.iter().enumerate() {
            if let Some(p) = j.parent {
                if inside_set.get(p as usize).copied().unwrap_or(false) {
                    inside_set[i] = true;
                }
            }
        }
        let rows = inside_set
            .iter()
            .enumerate()
            .filter(|(_, on)| **on)
            .map(|(i, _)| (i as u16, inside));
        Self::new(name, rows, outside)
    }

    /// Read an authored [`BlendProfile`] as a mask.
    ///
    /// The conversion is where the two meanings meet, so it takes `default` from
    /// the caller rather than guessing: a profile's silence means "at the
    /// transition's alpha", and what that becomes as a *layer* weight is a
    /// decision the layer owns. Passing `0.0` gives "only the joints the profile
    /// names"; passing `1.0` gives "everything except the ones it damps".
    pub fn from_profile(profile: &BlendProfile, default: f32) -> Self {
        Self::new(
            profile.name.clone(),
            profile.weights.iter().map(|w| (w.joint, w.weight)),
            default,
        )
    }

    /// The weight of one joint.
    pub fn weight(&self, joint: u16) -> f32 {
        match self.weights.binary_search_by_key(&joint, |(j, _)| *j) {
            Ok(i) => self.weights[i].1,
            Err(_) => self.default_weight,
        }
    }

    /// A dense `joints`-long weight vector, each entry scaled by `scale` and
    /// clamped into `[0,1]` — what [`crate::pose::blend_poses_weighted`] and
    /// [`apply_additive`] take.
    ///
    /// A non-finite authored weight resolves to `0` (leave the base alone), which
    /// is the same conservative reading `blend_poses_weighted` already applies to
    /// a joint past the end of its slice.
    pub fn resolve(&self, joints: usize, scale: f32) -> Vec<f32> {
        let scale = if scale.is_finite() {
            scale.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut out = vec![
            if self.default_weight.is_finite() {
                (self.default_weight * scale).clamp(0.0, 1.0)
            } else {
                0.0
            };
            joints
        ];
        for &(j, w) in &self.weights {
            if let Some(slot) = out.get_mut(j as usize) {
                *slot = if w.is_finite() {
                    (w * scale).clamp(0.0, 1.0)
                } else {
                    0.0
                };
            }
        }
        out
    }
}

/// How a layer combines with the pose beneath it.
///
/// Not a wire enum (see the module docs), so it carries no freeze pin and no
/// reserved slots — the day it is persisted is the day it needs both.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LayerMode {
    /// Overwrite the base toward the layer's pose (`lerp`/`slerp`) — an override.
    #[default]
    Blend,
    /// Add the layer's pose to the base as a **delta** (see [`apply_additive`]).
    /// The layer's pose must already be a delta — [`additive_delta`] makes one.
    Additive,
}

/// One layer of the stack: what it does, how strongly, and where.
#[derive(Clone, Debug, PartialEq)]
pub struct AnimLayer {
    pub name: String,
    pub mode: LayerMode,
    /// Overall weight in `[0,1]`, multiplied into the mask.
    pub weight: f32,
    /// Which joints it reaches. `None` = every joint at `weight`.
    pub mask: Option<JointMask>,
}

impl AnimLayer {
    /// A full-body blend (override) layer.
    pub fn blend(name: impl Into<String>, weight: f32) -> Self {
        Self {
            name: name.into(),
            mode: LayerMode::Blend,
            weight,
            mask: None,
        }
    }

    /// A full-body additive layer.
    pub fn additive(name: impl Into<String>, weight: f32) -> Self {
        Self {
            name: name.into(),
            mode: LayerMode::Additive,
            weight,
            mask: None,
        }
    }

    /// Restrict the layer to a mask.
    pub fn with_mask(mut self, mask: JointMask) -> Self {
        self.mask = Some(mask);
        self
    }

    /// The dense per-joint weight vector this layer applies over `joints` joints.
    pub fn weights(&self, joints: usize) -> Vec<f32> {
        match &self.mask {
            Some(m) => m.resolve(joints, self.weight),
            None => {
                let w = if self.weight.is_finite() {
                    self.weight.clamp(0.0, 1.0)
                } else {
                    0.0
                };
                vec![w; joints]
            }
        }
    }

    /// Whether this layer can change anything (a zero weight everywhere cannot).
    pub fn is_active(&self) -> bool {
        if !crate::positive(self.weight) {
            return false;
        }
        match &self.mask {
            None => true,
            Some(m) => m.default_weight > 0.0 || m.weights.iter().any(|(_, x)| *x > 0.0),
        }
    }
}

/// The **delta** of `pose` relative to `base`, joint by joint, in each joint's
/// own local space.
///
/// * translation — `pose − base` (metres);
/// * rotation — `base⁻¹ · pose`, so applying it is a post-multiply in the joint's
///   own frame, which is what makes an aim offset authored on one skeleton
///   meaningful on another pose of it;
/// * scale — `pose / base`, componentwise, because scale composes by product. A
///   zero base component yields `1` for that axis rather than an infinity: a
///   collapsed joint has no scale to be relative to, and a `1` leaves the base
///   untouched when the delta is applied.
///
/// The result is a [`Pose`] because that is what every consumer already takes; it
/// is **not** a pose in the sense of "a set of joint placements", and applying it
/// with [`crate::pose::blend_poses`] instead of [`apply_additive`] is meaningless.
pub fn additive_delta(base: &Pose, pose: &Pose) -> Pose {
    let n = base.locals.len().min(pose.locals.len());
    let mut locals = Vec::with_capacity(n);
    for i in 0..n {
        let (b, p) = (&base.locals[i], &pose.locals[i]);
        let t = p.translation_vec() - b.translation_vec();
        let r = b.rotation_quat().conjugate() * p.rotation_quat();
        let bs = b.scale_vec();
        let ps = p.scale_vec();
        let s = Vec3::new(
            safe_ratio(ps.x, bs.x),
            safe_ratio(ps.y, bs.y),
            safe_ratio(ps.z, bs.z),
        );
        locals.push(JointTransform::from_trs(t, r, s));
    }
    Pose { locals }
}

/// `a / b`, with `1.0` for a non-usable denominator (see [`additive_delta`]).
fn safe_ratio(a: f32, b: f32) -> f32 {
    // `!(b.abs() > MIN)` rather than `<=`: a NaN scale fails every ordering
    // comparison it takes part in, and the positive spelling would divide by it.
    const MIN: f32 = 1e-9;
    if !crate::greater(b.abs(), MIN) {
        return 1.0;
    }
    let r = a / b;
    if r.is_finite() {
        r
    } else {
        1.0
    }
}

/// Apply an additive `delta` (from [`additive_delta`]) on top of `base` with a
/// per-joint weight.
///
/// `weights[i]` is joint `i`'s share; a joint past the end of `weights` gets `0`
/// — the same conservative reading [`crate::pose::blend_poses_weighted`] uses, and
/// for the same reason (a mask that does not mention a joint must not move it).
///
/// The rotation is `base · slerp(identity, delta, w)`: `pslerp` from identity
/// scales a rotation's **angle** exactly (slerp travels at constant angular
/// velocity), so a half-weight additive is half the turn and not half the
/// quaternion.
pub fn apply_additive(base: &Pose, delta: &Pose, weights: &[f32]) -> Pose {
    let n = base.locals.len().min(delta.locals.len());
    let mut locals = Vec::with_capacity(n);
    for i in 0..n {
        let w = weights.get(i).copied().unwrap_or(0.0);
        let w = if w.is_finite() {
            w.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let b = &base.locals[i];
        let d = &delta.locals[i];
        let t = b.translation_vec() + d.translation_vec() * w;
        let r = b.rotation_quat() * inf_math::pslerp(Quat::IDENTITY, d.rotation_quat(), w);
        let s = b.scale_vec() * Vec3::ONE.lerp(d.scale_vec(), w);
        locals.push(JointTransform::from_trs(t, r, s));
    }
    // Joints the delta does not cover keep the base exactly — a shorter delta
    // must not truncate the character.
    for l in base.locals.iter().skip(n) {
        locals.push(*l);
    }
    Pose { locals }
}

/// The pose an **additive clip** is additive *over* — the door that reads
/// [`AdditiveRef`] (`.inf_anim` v2).
///
/// * [`AdditiveRef::None`] — the clip is not additive, so there is no base and
///   this is `None`. A caller that fell back to the rest pose here would turn
///   every ordinary clip into a delta against bind, which is a plausible-looking
///   pose and a wrong one.
/// * [`AdditiveRef::FirstFrame`] — the clip's own frame 0, sampled **non-looping**
///   so a negative or wrapped time cannot pick a different frame.
/// * [`AdditiveRef::BaseClip`] — the named clip at `time_s`. `None` when it does
///   not resolve: a delta measured against a base that is not there is the clip's
///   *absolute* pose wearing a delta's name, and [`apply_additive`] would then add
///   a whole character on top of a whole character.
/// * a **reserved** kind — `None`. It was written by a newer build; the decode
///   door already refuses it by name, and this is the second lock for a clip built
///   in memory that never passed one.
pub fn additive_base_pose<'c>(
    skeleton: &Skeleton,
    clip: &AnimClip,
    clips: &dyn Fn([u8; 16]) -> Option<&'c AnimClip>,
) -> Option<Pose> {
    match clip.additive {
        AdditiveRef::None => None,
        AdditiveRef::FirstFrame => Some(sample_clip(skeleton, clip, 0.0, false)),
        AdditiveRef::BaseClip { clip: id, time_s } => {
            clips(id).map(|c| sample_clip(skeleton, c, time_s, false))
        }
        AdditiveRef::Reserved3 { .. } | AdditiveRef::Reserved4 { .. } => None,
    }
}

/// Sample an additive clip into the **delta** [`apply_additive`] takes, resolving
/// its base through [`additive_base_pose`].
///
/// `None` for a clip that is not additive — which is the answer a caller needs,
/// because the alternative is silently treating a full pose as a delta and adding
/// it to whatever is underneath.
pub fn sample_additive_clip<'c>(
    skeleton: &Skeleton,
    clip: &AnimClip,
    t: f32,
    looping: bool,
    clips: &dyn Fn([u8; 16]) -> Option<&'c AnimClip>,
) -> Option<Pose> {
    let base = additive_base_pose(skeleton, clip, clips)?;
    Some(additive_delta(
        &base,
        &sample_clip(skeleton, clip, t, looping),
    ))
}

/// Apply one layer's pose over `base`.
pub fn apply_layer(base: &Pose, layer_pose: &Pose, layer: &AnimLayer) -> Pose {
    if !layer.is_active() {
        return base.clone();
    }
    let joints = base.locals.len();
    let weights = layer.weights(joints);
    match layer.mode {
        LayerMode::Blend => crate::pose::blend_poses_weighted(base, layer_pose, &weights),
        LayerMode::Additive => apply_additive(base, layer_pose, &weights),
    }
}

/// Fold a whole stack over `base`, **in order** — layer 0 first, so a later layer
/// sees what the earlier ones produced.
///
/// Order is the stack's semantics and not an implementation detail: an additive
/// lean under an override aim is a different character from an override aim under
/// an additive lean, and the caller owns which one it wants.
pub fn apply_layers<'a>(
    base: &Pose,
    layers: impl IntoIterator<Item = (&'a AnimLayer, &'a Pose)>,
) -> Pose {
    let mut out = base.clone();
    for (layer, pose) in layers {
        out = apply_layer(&out, pose, layer);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::Joint;
    use glam::Mat4;

    /// A 4-joint chain: 0 root, 1 spine, 2 head, 3 arm (child of spine) — enough
    /// to have a subtree that is not "everything".
    fn rig() -> Skeleton {
        let names = ["root", "spine", "head", "arm"];
        let parents = [None, Some(0u16), Some(1), Some(1)];
        let joints = names
            .iter()
            .zip(parents)
            .map(|(n, p)| Joint {
                name: (*n).into(),
                parent: p,
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::IDENTITY,
            })
            .collect();
        Skeleton::new(joints).unwrap()
    }

    fn pose_with(rot: &[(usize, f32)]) -> Pose {
        let mut p = Pose {
            locals: vec![JointTransform::IDENTITY; 4],
        };
        for &(i, deg) in rot {
            p.locals[i].rotation = inf_math::pslerp(
                Quat::IDENTITY,
                Quat::from_xyzw(0.0, 0.0, 1.0, 0.0),
                deg / 180.0,
            )
            .to_array();
        }
        p
    }

    fn deg_of(p: &Pose, i: usize) -> f32 {
        p.locals[i]
            .rotation_quat()
            .angle_between(Quat::IDENTITY)
            .to_degrees()
    }

    #[test]
    fn an_additive_delta_round_trips_at_full_weight() {
        let base = pose_with(&[(1, 20.0)]);
        let target = pose_with(&[(1, 50.0), (2, 30.0)]);
        let delta = additive_delta(&base, &target);
        let back = apply_additive(&base, &delta, &[1.0; 4]);
        for i in 0..4 {
            assert!(
                back.locals[i]
                    .rotation_quat()
                    .angle_between(target.locals[i].rotation_quat())
                    .to_degrees()
                    < 1e-2,
                "joint {i}: {} vs {}",
                deg_of(&back, i),
                deg_of(&target, i)
            );
        }
        // …and at zero weight it is the base, exactly.
        let none = apply_additive(&base, &delta, &[0.0; 4]);
        for i in 0..4 {
            assert!(
                none.locals[i]
                    .rotation_quat()
                    .angle_between(base.locals[i].rotation_quat())
                    .to_degrees()
                    < 1e-3
            );
        }
    }

    /// **A half-weight additive is half the ANGLE**, which is the property that
    /// makes `pslerp`-from-identity the right scaling and a componentwise
    /// quaternion lerp the wrong one.
    #[test]
    fn additive_weight_scales_the_angle_not_the_quaternion() {
        let base = Pose {
            locals: vec![JointTransform::IDENTITY; 4],
        };
        let target = pose_with(&[(1, 90.0)]);
        let delta = additive_delta(&base, &target);
        let half = apply_additive(&base, &delta, &[0.5; 4]);
        assert!(
            (deg_of(&half, 1) - 45.0).abs() < 0.5,
            "expected ~45°, got {}",
            deg_of(&half, 1)
        );
    }

    #[test]
    fn additive_translation_and_scale_compose_the_right_way() {
        let mut base = Pose {
            locals: vec![JointTransform::IDENTITY; 2],
        };
        base.locals[0].translation = [1.0, 2.0, 3.0];
        base.locals[0].scale = [2.0, 2.0, 2.0];
        let mut target = base.clone();
        target.locals[0].translation = [1.5, 2.0, 3.0];
        target.locals[0].scale = [4.0, 2.0, 2.0];
        let d = additive_delta(&base, &target);
        // Translation is a difference, scale is a ratio.
        assert!((d.locals[0].translation[0] - 0.5).abs() < 1e-6);
        assert!((d.locals[0].scale[0] - 2.0).abs() < 1e-6);
        let applied = apply_additive(&base, &d, &[1.0, 1.0]);
        assert!((applied.locals[0].translation[0] - 1.5).abs() < 1e-5);
        assert!((applied.locals[0].scale[0] - 4.0).abs() < 1e-5);
        // A collapsed base scale yields a ratio of 1 rather than an infinity, so
        // the applied pose is the base rather than NaN.
        let mut zero = base.clone();
        zero.locals[0].scale = [0.0, 0.0, 0.0];
        let d0 = additive_delta(&zero, &target);
        assert_eq!(d0.locals[0].scale, [1.0, 1.0, 1.0]);
        assert!(apply_additive(&zero, &d0, &[1.0, 1.0]).locals[0]
            .scale_vec()
            .is_finite());
    }

    #[test]
    fn a_subtree_mask_covers_the_joint_and_its_descendants_only() {
        let sk = rig();
        // "spine and below" is spine(1), head(2), arm(3) — not root(0).
        let m = JointMask::from_subtree("upper", &sk, 1, 1.0, 0.0);
        assert_eq!(m.weight(0), 0.0);
        assert_eq!(m.weight(1), 1.0);
        assert_eq!(m.weight(2), 1.0);
        assert_eq!(m.weight(3), 1.0);
        assert_eq!(m.resolve(4, 1.0), vec![0.0, 1.0, 1.0, 1.0]);
        // A leaf covers only itself.
        let head = JointMask::from_subtree("head", &sk, 2, 1.0, 0.0);
        assert_eq!(head.resolve(4, 1.0), vec![0.0, 0.0, 1.0, 0.0]);
        // The overall layer weight scales the mask.
        assert_eq!(m.resolve(4, 0.5), vec![0.0, 0.5, 0.5, 0.5]);
    }

    /// **A mask's silence is explicit and a profile's is not** — the two readings
    /// that must not be confused, asserted as two different answers to the same
    /// profile.
    #[test]
    fn a_profile_reads_as_a_mask_only_with_a_stated_default() {
        use crate::state_machine::JointBlendWeight;
        let p = BlendProfile::new(
            "upper-body",
            vec![JointBlendWeight {
                joint: 1,
                weight: 0.25,
            }],
        );
        let only_named = JointMask::from_profile(&p, 0.0);
        let all_but_damped = JointMask::from_profile(&p, 1.0);
        assert_eq!(only_named.resolve(4, 1.0), vec![0.0, 0.25, 0.0, 0.0]);
        assert_eq!(all_but_damped.resolve(4, 1.0), vec![1.0, 0.25, 1.0, 1.0]);
    }

    #[test]
    fn a_mask_deduplicates_and_sorts_its_rows() {
        let m = JointMask::new("m", [(3u16, 0.1), (1, 0.2), (3, 0.9)], 0.0);
        assert_eq!(m.rows(), &[(1, 0.2), (3, 0.9)], "the later row wins");
        // A joint past the end of the pose is dropped rather than panicking.
        let wide = JointMask::new("m", [(99u16, 1.0)], 0.0);
        assert_eq!(wide.resolve(2, 1.0), vec![0.0, 0.0]);
        // A non-finite authored weight leaves the base alone.
        let nan = JointMask::new("m", [(0u16, f32::NAN)], f32::NAN);
        assert_eq!(nan.resolve(2, 1.0), vec![0.0, 0.0]);
    }

    /// **The layer stack is ordered, and a mask really does confine a layer.**
    ///
    /// The mutation this arm exists to kill: dropping the mask (or resolving it as
    /// uniform) moves joint 0, and the first assertion goes red.
    #[test]
    fn an_upper_body_layer_leaves_the_legs_alone_and_the_stack_is_ordered() {
        let sk = rig();
        let base = pose_with(&[(0, 40.0), (1, 10.0)]);
        let overlay = pose_with(&[(0, 0.0), (1, 80.0)]);
        let layer = AnimLayer::blend("aim", 1.0)
            .with_mask(JointMask::from_subtree("upper", &sk, 1, 1.0, 0.0));
        let out = apply_layer(&base, &overlay, &layer);
        assert!(
            (deg_of(&out, 0) - 40.0).abs() < 0.5,
            "the masked-out root moved: {}",
            deg_of(&out, 0)
        );
        assert!(
            (deg_of(&out, 1) - 80.0).abs() < 0.5,
            "the masked-in spine did not take the overlay: {}",
            deg_of(&out, 1)
        );

        // Order matters: an additive lean under a *partial* override is not the
        // same as a partial override under an additive lean. (At full override
        // weight the two agree, because an override at 1.0 discards whatever it
        // was handed — which is why this arm uses 0.5 and a lean that reaches the
        // same joint.)
        let lean = additive_delta(&base, &pose_with(&[(0, 60.0), (1, 40.0)]));
        let add = AnimLayer::additive("lean", 1.0);
        let half = AnimLayer::blend("aim", 0.5)
            .with_mask(JointMask::from_subtree("upper", &sk, 1, 1.0, 0.0));
        let a = apply_layers(&base, [(&add, &lean), (&half, &overlay)]);
        let b = apply_layers(&base, [(&half, &overlay), (&add, &lean)]);
        assert!(
            (deg_of(&a, 1) - deg_of(&b, 1)).abs() > 1.0,
            "the two orders produced the same spine angle ({} vs {}), so the fold \
             is not ordered",
            deg_of(&a, 1),
            deg_of(&b, 1)
        );
    }

    #[test]
    fn an_inactive_layer_is_the_base_exactly() {
        let base = pose_with(&[(1, 30.0)]);
        let overlay = pose_with(&[(1, 90.0)]);
        for layer in [
            AnimLayer::blend("off", 0.0),
            AnimLayer::blend("nan", f32::NAN),
            AnimLayer::blend("empty-mask", 1.0).with_mask(JointMask::new("m", [], 0.0)),
        ] {
            assert!(!layer.is_active(), "{} claimed to be active", layer.name);
            assert_eq!(apply_layer(&base, &overlay, &layer), base);
        }
        // …and a uniform mask at full weight is the same as no mask at all.
        let masked = AnimLayer::blend("u", 1.0).with_mask(JointMask::uniform(1.0));
        assert_eq!(
            apply_layer(&base, &overlay, &masked),
            apply_layer(&base, &overlay, &AnimLayer::blend("u", 1.0))
        );
    }

    /// A delta shorter than the base leaves the uncovered joints **exactly** as
    /// they were, rather than truncating the character to the delta's length.
    #[test]
    fn a_short_delta_does_not_truncate_the_pose() {
        let base = pose_with(&[(0, 10.0), (1, 20.0), (2, 30.0), (3, 40.0)]);
        let short = Pose {
            locals: vec![JointTransform::IDENTITY; 2],
        };
        let out = apply_additive(&base, &short, &[1.0; 4]);
        assert_eq!(out.locals.len(), 4);
        assert_eq!(out.locals[3], base.locals[3]);
    }
    /// **The `AdditiveRef` door**, which is what stops the `.inf_anim` v2 field
    /// being data nothing reads.
    ///
    /// Three claims: a `FirstFrame` clip's delta at its own frame 0 is the
    /// identity (nothing has happened yet) and grows from there; a `BaseClip`
    /// reference is measured against the *named* clip and not against itself; and
    /// a clip that is **not** additive is refused rather than silently treated as
    /// a delta — which is the failure that looks most like success, because
    /// `apply_additive` would then add a whole character on top of a whole
    /// character.
    #[test]
    fn an_additive_clip_resolves_the_base_its_reference_names() {
        use crate::channels::AdditiveRef;
        use crate::clip::{Interpolation, JointTrack, QuatTrack};

        let sk = rig();
        // A clip taking joint 1 from 20° to 60° over a second.
        let swing = |from: f32, to: f32| {
            let mut jt = JointTrack::new(1);
            jt.rotation = Some(QuatTrack::new(
                vec![0.0, 1.0],
                vec![
                    inf_math::pslerp(
                        Quat::IDENTITY,
                        Quat::from_xyzw(0.0, 0.0, 1.0, 0.0),
                        from / 180.0,
                    )
                    .to_array(),
                    inf_math::pslerp(
                        Quat::IDENTITY,
                        Quat::from_xyzw(0.0, 0.0, 1.0, 0.0),
                        to / 180.0,
                    )
                    .to_array(),
                ],
                Interpolation::Linear,
            ));
            AnimClip::new("swing", vec![jt])
        };
        let none = |_: [u8; 16]| -> Option<&AnimClip> { None };

        // Not additive: refused.
        let plain = swing(20.0, 60.0);
        assert!(additive_base_pose(&sk, &plain, &none).is_none());
        assert!(sample_additive_clip(&sk, &plain, 0.5, false, &none).is_none());

        // FirstFrame: the delta is identity at t = 0 and 40° at t = 1.
        let ff = swing(20.0, 60.0).with_additive(AdditiveRef::FirstFrame);
        let d0 = sample_additive_clip(&sk, &ff, 0.0, false, &none).unwrap();
        assert!(deg_of(&d0, 1) < 1e-3, "{}", deg_of(&d0, 1));
        let d1 = sample_additive_clip(&sk, &ff, 1.0, false, &none).unwrap();
        assert!((deg_of(&d1, 1) - 40.0).abs() < 0.5, "{}", deg_of(&d1, 1));
        // …and applying it to an arbitrary base really adds those 40°.
        let base = pose_with(&[(1, 15.0)]);
        let out = apply_additive(&base, &d1, &[1.0; 4]);
        assert!((deg_of(&out, 1) - 55.0).abs() < 0.5, "{}", deg_of(&out, 1));

        // BaseClip: measured against the NAMED clip, at the named time.
        let reference = swing(0.0, 0.0);
        let over = swing(20.0, 60.0).with_additive(AdditiveRef::BaseClip {
            clip: [9; 16],
            time_s: 0.0,
        });
        let resolver =
            |id: [u8; 16]| -> Option<&AnimClip> { (id == [9; 16]).then_some(&reference) };
        let d = sample_additive_clip(&sk, &over, 0.0, false, &resolver).unwrap();
        assert!(
            (deg_of(&d, 1) - 20.0).abs() < 0.5,
            "the delta was measured against the wrong base: {}",
            deg_of(&d, 1)
        );
        // An unresolved base is a refusal, not a full pose wearing a delta's name.
        assert!(sample_additive_clip(&sk, &over, 0.0, false, &none).is_none());

        // A reserved kind — written by a newer build — is refused here too.
        let future = swing(20.0, 60.0).with_additive(AdditiveRef::Reserved3 {
            clip: [9; 16],
            time_s: 0.0,
        });
        assert!(sample_additive_clip(&sk, &future, 0.0, false, &resolver).is_none());
    }
}
