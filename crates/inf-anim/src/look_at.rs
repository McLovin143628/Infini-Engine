//! **Look-at: the head, the neck and the spine follow the camera** (wave
//! CHAR1b.1, clause 4).
//!
//! The user's own sentence, and the reason this exists: *"when the user moves
//! their mouse … the character turns their head and sometimes body."* Before
//! this the answer was no: `CharacterMovement`'s runtime has carried
//! `aim_sweep`, `spine_yaw_deg` and `aim_offset_weight` since P29.4 and
//! **nothing read any of them** — the numbers were computed every fixed step and
//! dropped, which is the same shape as this wave's foot-IK finding one pass over.
//!
//! # What is ported, and where it is written down
//!
//! `UALSCharacterAnimInstance::UpdateAimingValues` (`ALSCharacterAnimInstance.cpp`
//! `:235–290`) is the donor:
//!
//! * the aiming angle is the **delta between the aim rotation and the actor's
//!   own**, normalized — so a character that turns to face where it is looking
//!   stops leaning, without the leaning code knowing anything about turning;
//! * `SpineRotation.Yaw = AimingAngle.X / 4`, applied to **four** bones so the
//!   chain sums to the whole angle. `SPINE_SHARE` is that 4, kept as a name;
//! * the sweep time is `GetMappedRangeValueClamped({-90, 90} → {1, 0}, Pitch)`,
//!   which is [`crate::movement`]'s `aim_sweep` on the runtime side and is what
//!   the additive layer's vertical axis reads;
//! * `SmoothedAimingRotation` is an `RInterpTo` **before** the angle is taken, so
//!   a slow aim under a fast body turn stays slow. That smoothing lives on the
//!   runtime (the movement step owns the aim); this module is the geometry.
//!
//! What ALS does **not** do and this does: clamp. ALS's aim offset is an
//! authored asset whose extremes are the animator's, so the pose cannot exceed
//! them; a procedural chain has no such author, and a head that can yaw 180°
//! is a horror film. [`LookAtLimits`] is that author.
//!
//! # Portability
//!
//! Quaternions built from [`inf_math::psin`]/[`inf_math::pcos`] on the
//! half-angle, never `Quat::from_rotation_*` — this rotation reaches
//! `pose_state_bytes` and both hosts compare those bytes (the P14 law, which
//! `portable_pose` enforces over this file).

use glam::Quat;

use crate::pose::Pose;
use crate::roles::{BoneRoleKind, BoneSide, RoleIndex};
use crate::skeleton::Skeleton;

/// **How many bones ALS divides the aim yaw across** (`AimingAngle.X / 4.0`).
///
/// Four in the donor, and four here, because it is not a tuning constant: it is
/// the number of joints the donor's own rig gives the chain, and the quotient is
/// what makes the chain sum to the whole angle rather than to four times it.
/// A rig with a different spine length gets the same TOTAL — the share is
/// recomputed from the chain's real length in [`LookAt::spine_share`].
pub const SPINE_SHARE: f64 = 4.0;

/// How far each link of the look-at chain may turn.
///
/// The head's numbers are the brief's and are a human's: a neck rotates about
/// 70–80° to each side and extends about 60° / flexes 45°, and a character that
/// reaches the limit should **turn** rather than keep craning, which is what the
/// turn-in-place rules already do with the same angle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LookAtLimits {
    /// Head yaw, degrees each way.
    pub head_yaw_deg: f64,
    /// Head pitch, degrees each way.
    pub head_pitch_deg: f64,
    /// How much of the residual the **neck** takes, `[0, 1]` — applied before
    /// the head, so the head's clamp is against what is left.
    pub neck_share: f64,
    /// How much of the total yaw the **spine** leans, `[0, 1]`. ALS's own share
    /// is `1/4` per bone over four bones, i.e. all of it; a character who is not
    /// aiming leans a fraction of that, which is the `sometimes body` half of
    /// the user's sentence.
    pub spine_share: f64,
    /// The spine's own ceiling, degrees each way, summed over the chain.
    pub spine_yaw_deg: f64,
}

impl Default for LookAtLimits {
    fn default() -> Self {
        Self {
            head_yaw_deg: 70.0,
            head_pitch_deg: 35.0,
            neck_share: 0.45,
            spine_share: 0.35,
            spine_yaw_deg: 40.0,
        }
    }
}

impl LookAtLimits {
    /// **The aiming posture**: the body follows the camera, as ALS's
    /// `LookingDirection` and `Aiming` rotation modes do.
    ///
    /// The spine takes ALS's whole share (`AimingAngle.X / 4` over four bones)
    /// and the ceiling rises with it, so an aiming character squares up to what
    /// it is looking at instead of peering over its own shoulder.
    pub fn aiming() -> Self {
        Self {
            spine_share: 1.0,
            spine_yaw_deg: 60.0,
            ..Self::default()
        }
    }
}

/// Where a character is looking, relative to its own facing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LookAt {
    /// Yaw from the body's forward, degrees, positive to its right.
    pub yaw_deg: f64,
    /// Pitch, degrees, positive **up**.
    pub pitch_deg: f64,
    /// How much of the whole chain to apply, `[0, 1]` — ALS's
    /// `EnableAimOffset`, which is `1 - Mask_AimOffset`, so a state that wants
    /// none authors a `1` on the clip and gets none.
    pub weight: f32,
}

impl LookAt {
    /// The per-joint yaw a spine chain of `n` bones takes for this look —
    /// ALS's `AimingAngle.X / 4` generalized to the chain the rig really has.
    ///
    /// `0` for an empty chain, which is the honest answer rather than a
    /// division by zero.
    pub fn spine_share(&self, n: usize, limits: &LookAtLimits) -> f64 {
        if n == 0 {
            return 0.0;
        }
        let total =
            (self.yaw_deg * limits.spine_share).clamp(-limits.spine_yaw_deg, limits.spine_yaw_deg);
        // The donor divides by a hard-coded four; dividing by the chain's own
        // length is the same statement about the TOTAL on a rig that has four.
        let _ = SPINE_SHARE;
        total / n as f64
    }
}

/// What [`apply_look_at`] moved.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LookAtReport {
    /// Spine joints turned.
    pub spine: usize,
    /// Neck joints turned.
    pub neck: usize,
    /// Whether the head was turned.
    pub head: bool,
    /// **The yaw the head ended up carrying**, degrees — the number the gate
    /// measures against the look direction, and the one that shows the clamp.
    pub head_yaw_deg: f64,
    /// …and its pitch.
    pub head_pitch_deg: f64,
    /// The yaw the whole chain accounts for, degrees: spine + neck + head.
    ///
    /// Reported rather than assumed, because "the chain follows the camera"
    /// means the SUM tracks the look direction and no single joint does.
    pub total_yaw_deg: f64,
}

impl LookAtReport {
    /// Whether anything was written.
    pub fn wrote(&self) -> bool {
        self.head || self.spine > 0 || self.neck > 0
    }
}

/// **A yaw, in the RIG'S OWN frame** — about model `+Y`, the axis and the sense
/// `Transform::rotation.y` turns the body in (`d3::movement`'s step 12 writes
/// `body_yaw_deg` there). A character that yaws its head by d and one that yaws
/// its body by d therefore face the same way, which is the whole contract.
fn yaw_model(deg: f64) -> Quat {
    let h = (deg as f32).to_radians() * 0.5;
    Quat::from_xyzw(0.0, inf_math::psin(h), 0.0, inf_math::pcos(h))
}

/// **A pitch, in the rig's own frame** — about model `-X`, so a POSITIVE pitch
/// tips the head's forward toward `+Y` and the character looks UP. That is the
/// convention `aim_pitch_deg` and `inf_ecs::movement::aim_sweep` already use.
fn pitch_model(deg: f64) -> Quat {
    let h = (deg as f32).to_radians() * 0.5;
    Quat::from_xyzw(-inf_math::psin(h), 0.0, 0.0, inf_math::pcos(h))
}

/// The joint's rotation in **model space** under `pose`, by walking its parents.
///
/// The look is asked for in model space and has to be applied there; a joint's
/// LOCAL axes are the rig author's and are not it. On the MetaHuman rig the head
/// bone's local `X` points along model `+Y` and its local `Y` along model `-X`,
/// which is why the first version of this pass drew a yaw as a nod.
fn model_rot(skeleton: &Skeleton, pose: &Pose, joint: u16) -> Quat {
    let mut q = Quat::IDENTITY;
    let mut cur = Some(joint);
    // Bounded by the chain's depth; a `Skeleton` is validated with
    // `parent < self_index`, so this terminates.
    while let Some(j) = cur {
        let Some(local) = pose.locals.get(j as usize) else {
            break;
        };
        q = local.rotation_quat() * q;
        cur = skeleton.joints().get(j as usize).and_then(|jj| jj.parent);
    }
    q
}

/// The LOCAL post-multiply that rotates `joint` by `wanted` **in model space**.
///
/// `global = parent * local`, and we want `global' = wanted * global`; the
/// identity `local * (global⁻¹ · wanted · global)` gives exactly that, so the
/// call sites keep the post-multiply shape they had.
fn local_delta_for(model: Quat, wanted: Quat) -> Quat {
    model.inverse() * wanted * model
}

/// **Turn the head, the neck and the spine toward `look`.**
///
/// Applied in the rig's own local frames as a post-multiply, so it composes with
/// whatever the animation authored rather than replacing it — the same rule
/// [`crate::layers::apply_additive`] follows, and for the same reason: a look is
/// a *delta*, and a character that is already glancing left should end up
/// looking further left.
///
/// The order is hips-outward: spine, then neck, then head. Each link's clamp is
/// against what the links before it did **not** take, so a character whose spine
/// has already turned 30° only asks its head for the remaining 40 and the total
/// tracks the camera exactly until the whole chain saturates.
///
/// # It costs nothing on a rig it cannot read
///
/// A rig with no role table — every `.inf_skel` older than v3, every imported
/// glTF, every quadruped — has no `Head` row, and this returns a zero report
/// having written nothing. A zero `weight` does the same. That is what keeps
/// every committed sample byte-identical to its pre-CHAR1b self.
pub fn apply_look_at(
    skeleton: &Skeleton,
    pose: &mut Pose,
    roles: RoleIndex<'_>,
    look: LookAt,
    limits: &LookAtLimits,
) -> LookAtReport {
    let mut report = LookAtReport::default();
    let w = if look.weight.is_finite() {
        look.weight.clamp(0.0, 1.0) as f64
    } else {
        0.0
    };
    if w <= 0.0 || !look.yaw_deg.is_finite() || !look.pitch_deg.is_finite() {
        return report;
    }
    // **A character looking straight ahead is not looking at anything**, and
    // must pose the bytes it posed before this pass existed. Not an
    // optimisation: `rotation_quat()` normalizes on read, so composing with an
    // identity quaternion can still rewrite a joint whose stored rotation was
    // slightly off-unit — which is exactly what `sample_clip`'s lerp leaves
    // behind — and every determinism trace in the tree compares those bytes.
    const DEAD_DEG: f64 = 1.0e-6;
    if look.yaw_deg.abs() < DEAD_DEG && look.pitch_deg.abs() < DEAD_DEG {
        return report;
    }
    let head = roles.first(BoneRoleKind::Head, BoneSide::Center);
    let necks = roles.all(BoneRoleKind::Neck, BoneSide::Center);
    let spines = roles.all(BoneRoleKind::Spine, BoneSide::Center);
    if head.is_none() && necks.is_empty() && spines.is_empty() {
        return report;
    }

    // **THE MODEL-SPACE ROTATIONS THIS PASS TURNS ABOUT**, taken once before
    // anything is written. A joint written here rotates every joint below it, so
    // `ancestor` accumulates what this pass has already applied above the joint
    // being written and the later links compose exactly rather than approximately.
    let mut ancestor = Quat::IDENTITY;

    // ── the spine leans ──────────────────────────────────────────────────────
    let per_spine = look.spine_share(spines.len(), limits) * w;
    if per_spine.abs() > 1.0e-9 {
        let q = yaw_model(per_spine);
        for j in &spines {
            let model = ancestor * model_rot(skeleton, pose, *j);
            if let Some(l) = pose.locals.get_mut(*j as usize) {
                l.rotation = (l.rotation_quat() * local_delta_for(model, q)).to_array();
                report.spine += 1;
                ancestor = q * ancestor;
            }
        }
    }
    let spine_took = per_spine * report.spine as f64;

    // ── the neck takes its share of what is left ─────────────────────────────
    let left_yaw = look.yaw_deg * w - spine_took;
    let left_pitch = look.pitch_deg * w;
    let neck_yaw_total = (left_yaw * limits.neck_share).clamp(
        -limits.head_yaw_deg * limits.neck_share,
        limits.head_yaw_deg * limits.neck_share,
    );
    let neck_pitch_total = (left_pitch * limits.neck_share).clamp(
        -limits.head_pitch_deg * limits.neck_share,
        limits.head_pitch_deg * limits.neck_share,
    );
    let mut neck_took = 0.0;
    let mut neck_pitch_took = 0.0;
    if !necks.is_empty() {
        let per_y = neck_yaw_total / necks.len() as f64;
        let per_p = neck_pitch_total / necks.len() as f64;
        if per_y.abs() > 1.0e-9 || per_p.abs() > 1.0e-9 {
            let q = yaw_model(per_y) * pitch_model(per_p);
            for j in &necks {
                let model = ancestor * model_rot(skeleton, pose, *j);
                if let Some(l) = pose.locals.get_mut(*j as usize) {
                    l.rotation = (l.rotation_quat() * local_delta_for(model, q)).to_array();
                    report.neck += 1;
                    ancestor = q * ancestor;
                }
            }
            neck_took = per_y * report.neck as f64;
            neck_pitch_took = per_p * report.neck as f64;
        }
    }

    // ── and the head takes the rest, up to its own limit ─────────────────────
    let head_yaw = (left_yaw - neck_took).clamp(-limits.head_yaw_deg, limits.head_yaw_deg);
    let head_pitch =
        (left_pitch - neck_pitch_took).clamp(-limits.head_pitch_deg, limits.head_pitch_deg);
    if let Some(j) = head {
        if head_yaw.abs() > DEAD_DEG || head_pitch.abs() > DEAD_DEG {
            let model = ancestor * model_rot(skeleton, pose, j);
            if let Some(l) = pose.locals.get_mut(j as usize) {
                let q = yaw_model(head_yaw) * pitch_model(head_pitch);
                l.rotation = (l.rotation_quat() * local_delta_for(model, q)).to_array();
                report.head = true;
            }
        }
        // Reported whether or not it was written: "the head is carrying 0°" is
        // an answer, and a reader that had to infer it from `head == false`
        // could not tell it from "this rig has no head".
        report.head_yaw_deg = head_yaw;
        report.head_pitch_deg = head_pitch;
    }
    report.total_yaw_deg = spine_took + neck_took + report.head_yaw_deg;
    report
}

/// **How far a look-at chain can turn in total**, degrees — the angle past which
/// a character has to turn its body instead.
///
/// The number the turn-in-place rules are checked against
/// (`inf_ecs::movement::turn_in_place`'s `TURN_CHECK_MIN_DEG`), so "past the yaw
/// limit a turn-in-place plays" is a statement about one number and not two.
pub fn chain_yaw_limit_deg(spines: usize, necks: usize, limits: &LookAtLimits) -> f64 {
    let spine = if spines == 0 {
        0.0
    } else {
        limits.spine_yaw_deg
    };
    let neck = if necks == 0 {
        0.0
    } else {
        limits.head_yaw_deg * limits.neck_share
    };
    spine + neck + limits.head_yaw_deg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roles::BoneRole;
    use crate::skeleton::{Joint, JointTransform};
    use glam::Vec3;

    /// A spine → neck → head chain with a role table.
    fn torso() -> crate::asset::SkeletonAsset {
        fn joint(name: &str, parent: Option<u16>, y: f32) -> Joint {
            Joint {
                name: name.into(),
                parent,
                inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::from_trs(
                    Vec3::new(0.0, y, 0.0),
                    Quat::IDENTITY,
                    Vec3::ONE,
                ),
            }
        }
        let sk = Skeleton::new(vec![
            joint("pelvis", None, 1.0),
            joint("spine_01", Some(0), 0.12),
            joint("spine_02", Some(1), 0.12),
            joint("spine_03", Some(2), 0.12),
            joint("neck_01", Some(3), 0.12),
            joint("head", Some(4), 0.10),
        ])
        .expect("a torso");
        let mut a = crate::asset::SkeletonAsset::new(sk);
        a.roles = vec![
            BoneRole::new(0, BoneRoleKind::Pelvis, BoneSide::Center),
            BoneRole::new(1, BoneRoleKind::Spine, BoneSide::Center),
            BoneRole::new(2, BoneRoleKind::Spine, BoneSide::Center),
            BoneRole::new(3, BoneRoleKind::Spine, BoneSide::Center),
            BoneRole::new(4, BoneRoleKind::Neck, BoneSide::Center),
            BoneRole::new(5, BoneRoleKind::Head, BoneSide::Center),
        ];
        a
    }

    /// **A torso whose bones are oriented like a real rig's** — every joint's
    /// bind rotation carries the UE/MetaHuman convention, where the bone's local
    /// `X` runs UP the chain and its local `Y` across the body. On such a rig a
    /// rotation about local `Y` is a NOD and one about local `X` is a TURN, which
    /// is why the first version of this pass drew a yaw as a pitch on the island's
    /// hero and the user saw the character look up when the mouse went sideways.
    ///
    /// The identity-oriented `torso()` above cannot see that defect at all: on it
    /// local `Y` IS model `Y`. This is the rig that can.
    fn bone_oriented_torso() -> crate::asset::SkeletonAsset {
        let mut a = torso();
        // local X -> model +Y, local Y -> model -X, local Z -> model +Z: the
        // exact frame measured on the island hero's `head`.
        let bone = Quat::from_xyzw(
            0.0,
            0.0,
            inf_math::psin(std::f32::consts::FRAC_PI_4),
            inf_math::pcos(std::f32::consts::FRAC_PI_4),
        );
        let joints: Vec<Joint> = a
            .skeleton
            .joints()
            .iter()
            .enumerate()
            .map(|(i, j)| {
                let mut j = j.clone();
                if i > 0 {
                    // Only the first bone off the root carries the change; the
                    // rest inherit it, which is what a real chain does.
                    if i == 1 {
                        j.local_bind = JointTransform::from_trs(
                            j.local_bind.translation_vec(),
                            bone,
                            Vec3::ONE,
                        );
                    }
                }
                j
            })
            .collect();
        a.skeleton = Skeleton::new(joints).expect("the oriented torso");
        a
    }

    /// **A YAW TURNS THE HEAD AND A PITCH TIPS IT, ON A RIG WHOSE BONES ARE NOT
    /// AXIS-ALIGNED** (audit CHAR1b.1, the user's own report).
    ///
    /// *"When the user moves their mouse left/right, the character in the game
    /// looks up/down rather than left/right."* Measured on the island hero's rig
    /// before the fix: an asked yaw of 45 deg rotated the head 44.35 deg about
    /// model `-X` — a nod — and an asked pitch of 45 deg rotated it about model
    /// `+Y` — a turn. The two were exactly swapped, because the pass built its
    /// rotations about the JOINT'S LOCAL axes and a rig author's local axes are
    /// not the world's.
    ///
    /// **The mutation that reds it**: swap `yaw_model` and `pitch_model`, or drop
    /// the `local_delta_for` conjugation and post-multiply the raw quaternion.
    #[test]
    fn a_yaw_moves_the_azimuth_and_a_pitch_the_elevation_whatever_the_bones_do() {
        let rig = bone_oriented_torso();
        // Where the head POINTS: its forward (local +Z at rest is model +Z) in
        // model space, as an azimuth and an elevation.
        let dir = |pose: &Pose| -> (f64, f64) {
            let g = crate::pose::global_transforms(&rig.skeleton, pose);
            let f = g[5].to_scale_rotation_translation().1 * Vec3::Z;
            (
                inf_math::patan2_64(f64::from(f.x), f64::from(f.z)).to_degrees(),
                inf_math::patan2_64(f64::from(f.y), f64::from((f.x * f.x + f.z * f.z).sqrt()))
                    .to_degrees(),
            )
        };
        let base = dir(&Pose::rest(&rig.skeleton));
        for (yaw, pitch) in [(35.0_f64, 0.0_f64), (-35.0, 0.0), (0.0, 25.0), (0.0, -25.0)] {
            let mut p = Pose::rest(&rig.skeleton);
            apply_look_at(
                &rig.skeleton,
                &mut p,
                rig.role_index(),
                LookAt {
                    yaw_deg: yaw,
                    pitch_deg: pitch,
                    weight: 1.0,
                },
                &LookAtLimits::default(),
            );
            let (az, el) = dir(&p);
            let (daz, del) = (az - base.0, el - base.1);
            println!("  asked yaw {yaw:6.1} pitch {pitch:6.1} -> head az {daz:+7.2} el {del:+7.2}");
            if yaw.abs() > 0.0 {
                assert!(
                    daz * yaw > 0.0 && daz.abs() > 10.0,
                    "a yaw of {yaw} moved the head's azimuth by {daz:.2} deg"
                );
                assert!(
                    del.abs() < 1.0,
                    "a yaw of {yaw} tipped the head by {del:.2} deg — the pass is turning about a bone axis rather than the rig's own up"
                );
            } else {
                assert!(
                    del * pitch > 0.0 && del.abs() > 5.0,
                    "a pitch of {pitch} moved the head's elevation by {del:.2} deg"
                );
                assert!(
                    daz.abs() < 1.0,
                    "a pitch of {pitch} turned the head {daz:.2} deg — the axes are swapped, which is what the player sees as looking up when the mouse goes sideways"
                );
            }
        }
    }

    /// The head's world yaw under a pose, degrees.
    fn head_yaw_of(rig: &crate::asset::SkeletonAsset, pose: &Pose) -> f64 {
        let g = crate::pose::global_transforms(&rig.skeleton, pose);
        let q = g[5].to_scale_rotation_translation().1;
        f64::from(inf_math::pyaw(q).to_degrees())
    }

    #[test]
    fn a_zero_look_writes_nothing_and_a_rig_with_no_roles_cannot_be_looked_at() {
        let rig = torso();
        let base = Pose::rest(&rig.skeleton);
        let mut p = base.clone();
        let r = apply_look_at(
            &rig.skeleton,
            &mut p,
            rig.role_index(),
            LookAt {
                yaw_deg: 40.0,
                pitch_deg: 0.0,
                weight: 0.0,
            },
            &LookAtLimits::default(),
        );
        assert!(!r.wrote());
        assert_eq!(p, base, "a zero weight moved the pose");

        // A rig with no role table cannot be looked at, and says so by writing
        // nothing rather than by guessing which joint is a head.
        let bare = crate::asset::SkeletonAsset::new(rig.skeleton.clone());
        let mut p = base.clone();
        let r = apply_look_at(
            &bare.skeleton,
            &mut p,
            bare.role_index(),
            LookAt {
                yaw_deg: 40.0,
                pitch_deg: 0.0,
                weight: 1.0,
            },
            &LookAtLimits::default(),
        );
        assert!(!r.wrote());
        assert_eq!(p, base);
    }

    /// **THE CLAIM**: inside the limits the chain's total tracks the look
    /// direction, and past them it clamps.
    #[test]
    fn the_chain_tracks_the_look_direction_and_then_clamps() {
        let rig = torso();
        let limits = LookAtLimits::default();
        for want in [-60.0, -25.0, 0.0, 25.0, 60.0] {
            let mut p = Pose::rest(&rig.skeleton);
            let r = apply_look_at(
                &rig.skeleton,
                &mut p,
                rig.role_index(),
                LookAt {
                    yaw_deg: want,
                    pitch_deg: 0.0,
                    weight: 1.0,
                },
                &limits,
            );
            assert!((r.total_yaw_deg - want).abs() < 1.0e-9, "{want}: {r:?}");
            // …and the head really is where the report says: the world yaw of
            // the head joint is the chain's sum, which is the falsification the
            // report on its own cannot give.
            let got = head_yaw_of(&rig, &p);
            assert!(
                (got - want).abs() < 2.0,
                "{want}° asked, {got:.3}° drawn: {r:?}"
            );
        }
        // The head alone never exceeds its own limit, whatever the chain does.
        let mut p = Pose::rest(&rig.skeleton);
        let r = apply_look_at(
            &rig.skeleton,
            &mut p,
            rig.role_index(),
            LookAt {
                yaw_deg: 300.0,
                pitch_deg: 200.0,
                weight: 1.0,
            },
            &limits,
        );
        assert!(
            r.head_yaw_deg.abs() <= limits.head_yaw_deg + 1.0e-9,
            "the head yawed {:.3}°",
            r.head_yaw_deg
        );
        assert!(
            r.head_pitch_deg.abs() <= limits.head_pitch_deg + 1.0e-9,
            "the head pitched {:.3}°",
            r.head_pitch_deg
        );
        assert!(
            r.total_yaw_deg.abs() <= chain_yaw_limit_deg(3, 1, &limits) + 1.0e-9,
            "the whole chain took {:.3}°",
            r.total_yaw_deg
        );
        // …and the ceiling is not vacuous: it is well short of the 300 asked.
        assert!(r.total_yaw_deg < 200.0, "{r:?}");
    }

    /// The aiming posture leans the body further than the looking one — the
    /// "sometimes body" half of the user's sentence, as a measurement.
    #[test]
    fn aiming_leans_the_body_and_looking_mostly_does_not() {
        let rig = torso();
        let look = LookAt {
            yaw_deg: 50.0,
            pitch_deg: 0.0,
            weight: 1.0,
        };
        let mut a = Pose::rest(&rig.skeleton);
        let ra = apply_look_at(
            &rig.skeleton,
            &mut a,
            rig.role_index(),
            look,
            &LookAtLimits::default(),
        );
        let mut b = Pose::rest(&rig.skeleton);
        let rb = apply_look_at(
            &rig.skeleton,
            &mut b,
            rig.role_index(),
            look,
            &LookAtLimits::aiming(),
        );
        let spine_a = 50.0 * LookAtLimits::default().spine_share;
        let spine_b = 50.0 * LookAtLimits::aiming().spine_share;
        assert!(spine_b > spine_a * 2.0, "{spine_a} vs {spine_b}");
        // Both still track the same total: the difference is WHERE it is taken.
        assert!((ra.total_yaw_deg - rb.total_yaw_deg).abs() < 1.0e-9);
        assert!(
            ra.head_yaw_deg > rb.head_yaw_deg,
            "aiming should ask less of the head: {:.2} vs {:.2}",
            ra.head_yaw_deg,
            rb.head_yaw_deg
        );
    }

    /// Applied as a **delta**: a character already glancing left ends up looking
    /// further left, rather than being overwritten.
    #[test]
    fn the_look_composes_with_the_animation_rather_than_replacing_it() {
        let rig = torso();
        let mut posed = Pose::rest(&rig.skeleton);
        posed.locals[5].rotation = yaw_model(20.0).to_array();
        let before = head_yaw_of(&rig, &posed);
        apply_look_at(
            &rig.skeleton,
            &mut posed,
            rig.role_index(),
            LookAt {
                yaw_deg: 30.0,
                pitch_deg: 0.0,
                weight: 1.0,
            },
            &LookAtLimits::default(),
        );
        let after = head_yaw_of(&rig, &posed);
        assert!(
            (after - before - 30.0).abs() < 2.0,
            "{before:.2}° + 30° should be {after:.2}°"
        );
    }
}
