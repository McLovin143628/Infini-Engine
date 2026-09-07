//! **The lean** (wave CHAR1b.2, clause 6) — the pose consumer for the two
//! numbers `lean_x` / `lean_y` have been published as since this wave's first
//! commit.
//!
//! # What was here before
//!
//! P29.4 computed `MovementRuntime::relative_accel` every step with **zero
//! readers**. This wave's first commit interpolated it into
//! `MovementRuntime::lean` at ALS's own `GroundedLeanInterpSpeed`
//! (`ALSCharacterAnimInstance.cpp:633-638`) and published it as `lean_x` /
//! `lean_y` — and left it there, a parameter with no pose pass, which is
//! precisely the shape the CHAR1b.1 audit spent its day on (*"a row with no
//! reader is the defect this audit spent its day on"*). This is the reader.
//!
//! # ALS does it with clips, and this engine has none
//!
//! ALS feeds `LeanAmount` into an **authored additive** — a `Lean` blend space
//! whose poses an artist made — so the donor carries no *number* to port. Its
//! import into this engine carries no lean sequence either (the manifest's
//! `movement_sets` census names mantle 13 and cover / vault / slide / throwing /
//! swimming / prone 0; there is no lean row at all). So the magnitude is derived
//! rather than ported, and the derivation is stated:
//!
//! * A body accelerating at `a` leans by `atan(a / g)`, because that is the
//!   angle at which the resultant of gravity and inertia runs down the line the
//!   feet push along. At this engine's own committed maximum ground
//!   acceleration — `default_accel_curve`'s walk anchor, **8.0 m/s²** — against
//!   9.81 that is **39.20°**, and `|lean| = 1` is by construction "as hard as
//!   this character can accelerate" ([`inf_ecs::movement::relative_acceleration`]
//!   normalizes against the same maximum).
//! * The **legs take most of it.** A sprinter's shins and thighs carry the tilt;
//!   the spine's own contribution is the part a viewer reads as *leaning in*.
//!   [`LEAN_SPINE_SHARE`] is that fraction, and it is a stylisation with a
//!   number rather than a number with no argument: 0.30 of 39.20° is **11.76°**
//!   of spine at a full-effort start, which is a lean you can see and not a
//!   character folded double.
//!
//! # The frame is the RIG's, not a bone's
//!
//! The CHAR1b.1 audit's priority zero was a look-at that turned the head about a
//! *bone's local* axis, so a sideways mouse made the head nod. This pass takes
//! the same route out of that trap and shares its machinery: every rotation is
//! built in **model space** and conjugated into each joint's own frame
//! ([`crate::look_at::local_delta_for`]), with an accumulator so the links
//! compose exactly.
//!
//! # It costs nothing on a rig it cannot read
//!
//! A rig with no role table — every `.inf_skel` older than v3, every imported
//! glTF, every quadruped — has no `Spine` rows and this writes nothing. A lean
//! of zero returns before a quaternion is built, so a character standing still
//! poses the bytes it posed before this pass existed.

use glam::Quat;

use crate::pose::Pose;
use crate::roles::{BoneRoleKind, BoneSide, RoleIndex};
use crate::skeleton::Skeleton;

/// **The whole body's tilt at `|lean| = 1`**, degrees.
///
/// `atan(8.0 / 9.81)` — this engine's committed maximum ground acceleration
/// (`default_accel_curve`'s walk anchor) over standard gravity. See the module
/// docs for why a body leans by exactly that angle.
pub const LEAN_BODY_DEG: f64 = 39.20;

/// **The share of [`LEAN_BODY_DEG`] the SPINE takes**; the legs take the rest.
///
/// A stylisation, stated as one. ALS's own lean is an authored additive and
/// carries no number to port.
pub const LEAN_SPINE_SHARE: f64 = 0.30;

/// Below this the pass returns having written nothing, so a character that is
/// not accelerating poses byte-identical bytes. `rotation_quat()` normalizes on
/// read, so composing with an identity quaternion can still rewrite a joint
/// whose stored rotation is slightly off-unit — the reason [`crate::look_at`]
/// has the same guard.
const DEAD: f64 = 1.0e-6;

/// What the lean pass did — the report a gate reads beside the joints.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LeanReport {
    /// How many spine joints were written.
    pub spine: usize,
    /// The total roll applied, degrees — positive tips the character's top
    /// toward its own **right** (`+X`).
    pub roll_deg: f64,
    /// The total pitch applied, degrees — positive tips it **forward** (`+Z`).
    pub pitch_deg: f64,
}

/// A roll **in the rig's own frame**: positive tips the character's top toward
/// its own right, which is model `+X` (this engine's rigs put `hand_l` at
/// negative `x`).
///
/// A rotation about model `-Z`, because `R_z(-θ)` carries `+Y` toward `+X`.
/// Portable sine and cosine, because this result reaches the pose and therefore
/// `pose_state_bytes`.
fn roll_model(deg: f64) -> Quat {
    let h = (deg as f32).to_radians() * 0.5;
    Quat::from_xyzw(0.0, 0.0, -inf_math::psin(h), inf_math::pcos(h))
}

/// A pitch **in the rig's own frame**: positive tips the character forward,
/// toward model `+Z`.
///
/// The opposite sense to [`crate::look_at`]'s pitch, and deliberately: a
/// positive *look* pitch raises the eyes, and a positive *lean* tips the chest
/// down the direction of travel. Both are stated where they are built rather
/// than inferred from a shared helper that would have to mean two things.
fn pitch_model(deg: f64) -> Quat {
    let h = (deg as f32).to_radians() * 0.5;
    Quat::from_xyzw(inf_math::psin(h), 0.0, 0.0, inf_math::pcos(h))
}

/// **Lean the spine into the acceleration.**
///
/// `lean_x` is the character's own right (`+1` = accelerating to its right) and
/// `lean_y` its forward, both `[-1, 1]` — exactly what
/// [`inf_ecs::movement::relative_acceleration`] produces and what `lean_x` /
/// `lean_y` publish. `weight` scales the whole pass.
///
/// The total angle is [`LEAN_BODY_DEG`] × [`LEAN_SPINE_SHARE`] × the lean, split
/// **equally across the spine joints**, so a five-segment spine curves and a
/// one-segment one hinges — the same rule [`crate::look_at`] applies to its own
/// spine share, and the reason neither pass needs to know how many segments a
/// rig has.
///
/// Applied as a post-multiply on each joint's local rotation, so it composes
/// with whatever the animation authored instead of replacing it.
pub fn apply_lean(
    skeleton: &Skeleton,
    pose: &mut Pose,
    roles: RoleIndex<'_>,
    lean_x: f64,
    lean_y: f64,
    weight: f64,
) -> LeanReport {
    let mut report = LeanReport::default();
    if !weight.is_finite() || weight <= 0.0 || !lean_x.is_finite() || !lean_y.is_finite() {
        return report;
    }
    let w = weight.clamp(0.0, 1.0);
    let full = LEAN_BODY_DEG * LEAN_SPINE_SHARE * w;
    let roll = lean_x.clamp(-1.0, 1.0) * full;
    let pitch = lean_y.clamp(-1.0, 1.0) * full;
    if roll.abs() < DEAD && pitch.abs() < DEAD {
        return report;
    }
    let spines = roles.all(BoneRoleKind::Spine, BoneSide::Center);
    if spines.is_empty() {
        return report;
    }
    let n = spines.len() as f64;
    let per = roll_model(roll / n) * pitch_model(pitch / n);
    // The accumulator: a joint written here rotates every joint below it, so
    // what this pass has already applied above the joint being written has to be
    // folded into the model rotation the next conjugation is taken against.
    let mut ancestor = Quat::IDENTITY;
    for j in &spines {
        let model = ancestor * crate::look_at::model_rot(skeleton, pose, *j);
        if let Some(l) = pose.locals.get_mut(*j as usize) {
            l.rotation =
                (l.rotation_quat() * crate::look_at::local_delta_for(model, per)).to_array();
            report.spine += 1;
            ancestor = per * ancestor;
        }
    }
    report.roll_deg = roll / n * report.spine as f64;
    report.pitch_deg = pitch / n * report.spine as f64;
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::global_transforms;

    /// The rig every arm here uses: the mannequin, which carries a role table
    /// and five spine segments.
    fn body() -> crate::SkeletonAsset {
        crate::manny::build_manny(&crate::BodyParams::default()).expect("the mannequin builds")
    }

    /// The chest's model position, so a lean is measured off the JOINT rather
    /// than off the report that says one happened.
    fn chest(asset: &crate::SkeletonAsset, pose: &Pose) -> glam::Vec3 {
        let roles = asset.role_index();
        let j = roles
            .last(BoneRoleKind::Spine, BoneSide::Center)
            .expect("a spine");
        global_transforms(&asset.skeleton, pose)[j as usize]
            .w_axis
            .truncate()
    }

    /// **A lean of zero writes nothing at all** — the bytes, not "close to".
    #[test]
    fn a_character_that_is_not_accelerating_poses_the_bytes_it_always_did() {
        let asset = body();
        let base = Pose::rest(&asset.skeleton);
        let mut p = base.clone();
        let r = apply_lean(&asset.skeleton, &mut p, asset.role_index(), 0.0, 0.0, 1.0);
        assert_eq!(r, LeanReport::default());
        assert_eq!(p.locals, base.locals, "a zero lean rewrote a joint");
        // …and so does a zero weight, and a non-finite input.
        let mut p2 = base.clone();
        apply_lean(&asset.skeleton, &mut p2, asset.role_index(), 1.0, 1.0, 0.0);
        assert_eq!(p2.locals, base.locals);
        let mut p3 = base.clone();
        apply_lean(
            &asset.skeleton,
            &mut p3,
            asset.role_index(),
            f64::NAN,
            0.0,
            1.0,
        );
        assert_eq!(p3.locals, base.locals);
    }

    /// **The chest goes the way the character is accelerating**, and the answer
    /// is read off the joint.
    #[test]
    fn the_chest_leads_the_acceleration_in_all_four_directions() {
        let asset = body();
        let rest = Pose::rest(&asset.skeleton);
        let at = chest(&asset, &rest);
        let leaned = |x: f64, y: f64| -> glam::Vec3 {
            let mut p = rest.clone();
            apply_lean(&asset.skeleton, &mut p, asset.role_index(), x, y, 1.0);
            chest(&asset, &p)
        };
        let fwd = leaned(0.0, 1.0);
        let back = leaned(0.0, -1.0);
        let right = leaned(1.0, 0.0);
        let left = leaned(-1.0, 0.0);
        println!(
            "the lean, off the chest joint (rest {at:?}):\n  forward {:+.4} m z, back {:+.4}, \
             right {:+.4} m x, left {:+.4}",
            fwd.z - at.z,
            back.z - at.z,
            right.x - at.x,
            left.x - at.x
        );
        assert!(
            fwd.z - at.z > 0.02,
            "a forward lean moved the chest {fwd:?}"
        );
        assert!(back.z - at.z < -0.02, "a back lean moved it {back:?}");
        assert!(
            right.x - at.x > 0.02,
            "a right lean moved the chest {right:?}"
        );
        assert!(left.x - at.x < -0.02, "a left lean moved it {left:?}");
        // Symmetric: the same magnitude either way, because the same angle is.
        assert!((fwd.z - at.z + (back.z - at.z)).abs() < 1.0e-4);
        assert!((right.x - at.x + (left.x - at.x)).abs() < 1.0e-4);
    }

    /// **The legs do not lean.** The pass is masked to the spine by construction
    /// — it writes only `Spine` rows — and this asserts the consequence on the
    /// joints, because a mask that is right in the table and wrong in the tree is
    /// the defect the breath's neck counter-rotation exists for.
    #[test]
    fn a_lean_moves_the_chest_and_leaves_the_pelvis_and_feet_alone() {
        let asset = body();
        let roles = asset.role_index();
        let rest = Pose::rest(&asset.skeleton);
        let mut p = rest.clone();
        let r = apply_lean(&asset.skeleton, &mut p, roles, 0.8, 0.8, 1.0);
        assert!(r.spine > 0, "nothing was written");
        let a = global_transforms(&asset.skeleton, &rest);
        let b = global_transforms(&asset.skeleton, &p);
        let moved = |j: u16| {
            (b[j as usize].w_axis - a[j as usize].w_axis)
                .truncate()
                .length()
        };
        let pelvis = roles
            .first(BoneRoleKind::Pelvis, BoneSide::Center)
            .expect("a pelvis");
        let feet = crate::derive::foot_joints(&asset);
        let ch = roles
            .last(BoneRoleKind::Spine, BoneSide::Center)
            .expect("a spine");
        println!(
            "the lean's reach: chest {:.2} mm, pelvis {:.4} mm, feet {:?} mm ({} spine joints, \
             roll {:.2}° pitch {:.2}°)",
            moved(ch) * 1000.0,
            moved(pelvis) * 1000.0,
            feet.iter().map(|j| moved(*j) * 1000.0).collect::<Vec<_>>(),
            r.spine,
            r.roll_deg,
            r.pitch_deg
        );
        assert!(moved(ch) > 0.01, "the chest barely moved");
        assert!(moved(pelvis) < 1.0e-6, "the lean moved the pelvis");
        for j in feet {
            assert!(moved(j) < 1.0e-6, "the lean moved a foot");
        }
    }

    /// The magnitude is the one the module docs derive, and it is **linear in
    /// the lean** — so a half-effort start leans half as far.
    #[test]
    fn the_angle_is_the_derived_one_and_it_scales_with_the_effort() {
        let asset = body();
        let full = {
            let mut p = Pose::rest(&asset.skeleton);
            apply_lean(&asset.skeleton, &mut p, asset.role_index(), 0.0, 1.0, 1.0)
        };
        let half = {
            let mut p = Pose::rest(&asset.skeleton);
            apply_lean(&asset.skeleton, &mut p, asset.role_index(), 0.0, 0.5, 1.0)
        };
        let want = LEAN_BODY_DEG * LEAN_SPINE_SHARE;
        assert!(
            (full.pitch_deg - want).abs() < 1.0e-9,
            "a full lean drew {:.4}° against the derived {want:.4}°",
            full.pitch_deg
        );
        assert!((half.pitch_deg - want * 0.5).abs() < 1.0e-9);
        // 8.0 m/s² over 9.81 really is this angle, to two places.
        assert!((LEAN_BODY_DEG - (8.0f64 / 9.81).atan().to_degrees()).abs() < 0.01);
    }
}
