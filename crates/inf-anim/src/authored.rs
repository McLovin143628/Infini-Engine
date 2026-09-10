//! **The sets the donor does not ship** (wave CHAR1b.2) — slide, throwing,
//! swimming, prone and the standing get-up, derived from the rig that will play
//! them.
//!
//! # Why these nine clips are generated and not imported
//!
//! ALS ships 164 sequences and a census by name found **none** of these: no
//! slide, no throw, no swim, no prone, no standing get-up, no cover, no vault
//! (`movement_sets` in the island manifest: `mantle 13`; the rest `0`). The
//! engine's own catalogue has modes for five of them —
//! `MovementMode::{Slide, Prone, SwimSurface, SwimUnder, Cover}` — and a mode
//! with no clip is a character sliding in its idle pose, or standing to
//! attention with its back to a wall.
//!
//! They are **derived from the rig**, for [`crate::locomotion`]'s reason: a clip
//! is bound to a skeleton by joint *index*, so a shipped one would fit exactly
//! one rig, and this engine's characters are a 161-bone mannequin, a 342-bone
//! MetaHuman and whatever a wizard generates. Everything here is addressed
//! through [`RoleIndex`], which is the same door the look-at chain, the aim mask
//! and the hand pass use.
//!
//! # What each clip depicts, stated
//!
//! Nothing here is claimed to be ALS's. Each is a kinematic statement with its
//! numbers written down, and a reviewer who thinks a number is wrong can change
//! it in one place and re-derive:
//!
//! | clip | kinematics | cycle |
//! |---|---|---|
//! | `INF_Slide` | pelvis drops 45 % of hip height, lead knee 95°, trail leg trails 30°, torso pitches 25° forward, arms back 40° | 1.10 s, one-shot |
//! | `INF_Throw_Over` | shoulder sweeps −80° → +110° about its own X, elbow 100° → 10°, chest rotates 25° into the throw | 0.85 s, one-shot |
//! | `INF_Throw_Under` | shoulder sweeps +50° → −70°, elbow 60° → 5°, chest 10° | 0.75 s, one-shot |
//! | `INF_Swim_Surface` | alternating 140° shoulder crawl, legs flutter ±14° at twice the arm rate, torso pitches 60° to horizontal | 1.60 s, looping |
//! | `INF_Swim_Tread` | shoulders sculling ±35° at 0.8 Hz, legs cycling ±25°, torso 20° | 2.40 s, looping |
//! | `INF_Swim_Under` | the surface stroke with the torso at 75° and a 6° roll, arms sweeping 160° | 2.00 s, looping |
//! | `INF_Prone_Idle` | torso 88° to horizontal, pelvis at 12 % of hip height, elbows 90°, legs straight | 3.00 s, looping (a 2° breath) |
//! | `INF_Prone_Crawl` | the prone pose with alternating 35° hip and 55° elbow drive | 1.80 s, looping |
//! | `INF_GetUp_Standing` | pelvis rises from 20 % to 100 % of hip height while the torso unfolds 70° → 0° and the knees 100° → 0° | 1.20 s, one-shot |
//! | `INF_Cover_Low_Idle` | pelvis at 62 % of hip height, chest 12° forward and 18° twisted toward the surface, the near forearm AIMED across the chest (86° up, 66° across) and the far arm braced, hips folded 55° and knees 75° | 3.20 s, looping (a 2° breath) |
//! | `INF_Cover_Low_Move` | the same stance with a 22° abduction / 30° knee shuffle and a 10° torso counter-sway | 1.20 s, looping |
//! | `INF_Cover_High_Idle` | upright, chest 8° back and 22° twisted toward the surface, the near forearm AIMED across the chest (84° up, 72° across), the far arm braced | 3.60 s, looping (a 2° breath) |
//! | `INF_Cover_High_Move` | the same stance side-stepping: 22° hip abduction alternating, 30° knee, 10° torso counter-sway | 1.10 s, looping |
//! | `INF_Cover_BlindFire` | the weapon arm from the cover stance (34°/26°, 86°/66°) to 150°/8° and 168°/4° — straight up over the parapet — the far arm tucked, and the chest pitching 12° → 34° so the head goes DOWN; holds to 75 %, recovers a third | 0.70 s, one-shot additive |
//!
//! # Determinism
//!
//! Every angle goes through [`inf_math::psin64`]/[`inf_math::pcos64`] and is cast
//! to `f32` once, at the wire — [`crate::template`]'s doctrine, and the reason
//! is the same: these are **committed content**, a `.inf_anim` whose bytes a
//! cook, a pack and a determinism gate all compare. No `glam` rotation
//! constructor is called, because every one of them is `f32::sin_cos`.
//!
//! Two derivations from the same rig are byte-identical
//! (`the_same_rig_authors_the_same_bytes`).

use inf_math::{pcos64, psin64};

use crate::asset::SkeletonAsset;
use crate::clip::{AnimClip, Interpolation, JointTrack, QuatTrack, Vec3Track};
use crate::roles::{BoneRoleKind, BoneSide};
use crate::skeleton::Skeleton;

/// Every clip this module authors, in the order it authors them.
///
/// The names carry an `INF_` prefix, deliberately: they sit in the same content
/// root as 164 `ALS_*` sequences and **nothing here is ALS's**. A reader who
/// meets `INF_Swim_Tread` in a machine, a pack or an advisory should not have to
/// look it up to know who wrote it.
/// **How long the overhand throw plays**, seconds — the clip's own duration
/// (wave CHAR1b.2 authored it; wave WPN2d names it).
///
/// Named beside the generator rather than read off a loaded asset because the
/// GAMEPLAY step has to start the additive's clock with it and the gameplay step
/// has no clip: `inf_ecs::anim_bridge::start_throw` takes a duration precisely
/// so the two cannot disagree, and this is the number it takes. `throw()` below
/// is where it is spent.
pub const THROW_OVER_S: f64 = 0.85;

/// **How long the underhand throw plays**, seconds. See [`THROW_OVER_S`].
pub const THROW_UNDER_S: f64 = 0.75;

/// **How long a blind shot from cover plays**, seconds (wave WPN2e).
///
/// **0.70.** Named beside the generator on [`THROW_OVER_S`]'s argument verbatim:
/// the GAMEPLAY step has to start the additive's clock and the gameplay step has
/// no clip, so `inf_ecs::anim_bridge::start_blind_fire` takes a duration
/// precisely so the two cannot disagree.
///
/// Seven tenths of a second is long enough that the arm visibly goes up, holds,
/// and comes down, and short enough that a unit on a 1.6 s engagement cadence is
/// back behind its wall before the next burst.
pub const BLIND_FIRE_S: f64 = 0.70;

/// **How far into a throw the hand lets go**, as a fraction of the clip (wave
/// WPN2d).
///
/// Three fifths. It is read off the clip's OWN shape rather than chosen: the
/// overhand throw sweeps its shoulder from -80 deg to +110 deg and extends its
/// elbow from 100 deg to 10 deg over the whole clip, so the forearm is pointing
/// down-range and nearly straight at about 60 % of it — before that the hand is
/// still behind the head, and after it the arm is following through on an empty
/// palm. The underhand clip's sweep (+50 deg to -70 deg) crosses its own release
/// at the same fraction, which is why there is one number and not two.
///
/// It is a FRACTION rather than a time so the two clips share it, and it is here
/// beside the generator that draws the arms for the same reason the durations
/// are: the notify has to fire where the animation actually releases, and the
/// only thing that knows where that is is the thing that authored the sweep.
pub const THROW_RELEASE_FRAC: f64 = 0.60;

pub const AUTHORED_CLIPS: &[&str] = &[
    "INF_Slide",
    "INF_Throw_Over",
    "INF_Throw_Under",
    "INF_Swim_Surface",
    "INF_Swim_Tread",
    "INF_Swim_Under",
    "INF_Prone_Idle",
    "INF_Prone_Crawl",
    "INF_GetUp_Standing",
    // ── wave COV1: taking cover ──
    //
    // ALS ships **no cover clip of any kind** — the census by name that found
    // no slide and no swim found no cover and no vault either, and the module
    // doc above has said so since CHAR1b.2. These four are the set: a stance
    // per class, and a move per class, because a character side-stepping along
    // a wall in its walk cycle is a character that is not in cover.
    "INF_Cover_Low_Idle",
    "INF_Cover_Low_Move",
    "INF_Cover_High_Idle",
    "INF_Cover_High_Move",
    // -- wave WPN2e: firing without looking --
    //
    // The upper-body one-shot COV1 priced and did not build. An OVERLAY and not
    // a state, on the throws' argument verbatim: a character blind-firing is
    // still in cover, still crouched or still standing against its wall, and a
    // state would replace the stance the whole shot is taken from.
    "INF_Cover_BlindFire",
];

/// Why an authored set refused to generate.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AuthorError {
    /// The rig carries no role of a kind every one of these clips drives.
    #[error(
        "this rig has no `{role}` role, which every authored clip drives -- derive a role \
         table for it (`ue_import::sweep_roles`) or generate it from a template"
    )]
    MissingRole { role: &'static str },
}

/// The joints an authored clip drives, resolved once from the role table.
///
/// Everything is `Option` but the pelvis and the chest: a rig with one arm gets
/// a one-armed throw rather than a refusal, because a refusal here would take
/// the whole set down over a bone one clip wanted.
#[derive(Debug, Clone)]
struct Rig {
    /// **Every joint's bind local ROTATION**, indexed by joint.
    ///
    /// Used by [`cover`] and by nothing else, and the split is the point: the
    /// clips in this module that depict a pose far from the bind — prone, the
    /// swims, the get-up — write a local rotation OUTRIGHT, because what the
    /// arm was doing at bind is irrelevant to a body lying on its face. A cover
    /// stance is a STANDING pose that differs from bind by a small delta, and
    /// writing the local outright puts the shoulder wherever the clavicle's own
    /// frame happens to point: measured in the demo loop's first cover frame,
    /// the hero stood behind a wall with both arms out sideways. So `cover`
    /// composes `bind * delta` and says so.
    bind_rot: Vec<[f32; 4]>,
    /// **Every joint's bind GLOBAL rotation**, composed down the parent chain.
    ///
    /// The other half of what [`Rig::aim_under`] needs, and the audit finding that
    /// asked for it: a delta written as [`qx`] is a rotation about the JOINT's
    /// own X, and on a UE-derived rig -- every rig this engine imports -- local
    /// X points **down the bone**. So `qx` on an upper arm is a twist about the
    /// humerus rather than a lift of it, `qx` on a thigh is a twist about the
    /// femur rather than a hip fold, and a stance authored out of them moves
    /// nothing a viewer can see. Measured on the island's own hero
    /// (`upperarm_l`: local X reaches world (0.61, -0.79, 0.03), which is the
    /// shoulder-to-elbow direction to two decimals).
    ///
    /// With the globals in hand a clip says where a bone should POINT and the
    /// rig answers with the rotation, which is the shape `crate::retarget`
    /// already uses.
    bind_gq: Vec<[f32; 4]>,
    /// Every joint's bind global TRANSLATION -- the other input to a bone
    /// direction.
    bind_pos: Vec<glam::Vec3>,
    /// Every joint's parent, so [`Rig::is_ancestor`] can answer whether a
    /// rotation this clip writes is one another bone hangs under.
    parents: Vec<Option<u16>>,
    /// **Which way "across the chest" is for each arm**, `+1` toward `+X` and
    /// `-1` toward `-X` — derived from where the shoulder actually sits.
    ///
    /// Measured rather than assumed, because the two rigs this engine holds
    /// disagree: the shipped MetaHuman's `upperarm_l` is at **x = +0.198** and
    /// the biped template's left upper arm is at **x = -0.200**. A clip that
    /// picked one of them folded the near arm on one rig and threw it wide on
    /// the other, which is a mirrored pose that no test comparing joint COUNTS
    /// could see.
    across_sign: [f64; 2],
    pelvis: u16,
    pelvis_bind: [f32; 3],
    hip_height_m: f64,
    spine: Vec<u16>,
    upper_arm: [Option<u16>; 2],
    lower_arm: [Option<u16>; 2],
    hand: [Option<u16>; 2],
    thigh: [Option<u16>; 2],
    calf: [Option<u16>; 2],
    foot: [Option<u16>; 2],
}

impl Rig {
    fn of(rig: &SkeletonAsset) -> Result<Self, AuthorError> {
        let roles = rig.role_index();
        let sk = &rig.skeleton;
        let bind_rot: Vec<[f32; 4]> = sk.joints().iter().map(|j| j.local_bind.rotation).collect();
        let pelvis = roles
            .first(BoneRoleKind::Pelvis, BoneSide::Center)
            .ok_or(AuthorError::MissingRole { role: "Pelvis" })?;
        let spine: Vec<u16> = roles
            .rows()
            .iter()
            .filter(|r| r.kind == BoneRoleKind::Spine)
            .map(|r| r.joint)
            .collect();
        if spine.is_empty() {
            return Err(AuthorError::MissingRole { role: "Spine" });
        }
        let side = |k: BoneRoleKind| -> [Option<u16>; 2] {
            [
                roles.first(k, BoneSide::Left),
                roles.first(k, BoneSide::Right),
            ]
        };
        let globals = bind_globals(sk);
        let h = f64::from(globals[pelvis as usize].y).abs();
        // The global bind ROTATIONS, composed the way the translations above
        // are: multiplies only, so this stays inside the module's determinism
        // rule for `bind_globals`' own reason.
        let mut bind_gq: Vec<[f32; 4]> = Vec::with_capacity(sk.len());
        for j in sk.joints() {
            let q = match j.parent {
                Some(p) => quat_mul(bind_gq[p as usize], j.local_bind.rotation),
                None => j.local_bind.rotation,
            };
            bind_gq.push(q);
        }
        let upper_arm = side(BoneRoleKind::UpperArm);
        let across_sign = [0usize, 1].map(|i| {
            let default = if i == 0 { -1.0 } else { 1.0 };
            match upper_arm[i] {
                Some(j) => {
                    let d = globals[j as usize].x - globals[pelvis as usize].x;
                    if d.abs() < 1.0e-4 {
                        default
                    } else {
                        -f64::from(d.signum())
                    }
                }
                None => default,
            }
        });
        Ok(Self {
            bind_rot,
            bind_gq,
            bind_pos: globals,
            parents: sk.joints().iter().map(|j| j.parent).collect(),
            across_sign,
            pelvis,
            pelvis_bind: sk.joints()[pelvis as usize].local_bind.translation,
            hip_height_m: if h > 1.0e-4 { h } else { 0.95 },
            spine,
            upper_arm,
            lower_arm: side(BoneRoleKind::LowerArm),
            hand: side(BoneRoleKind::Hand),
            thigh: side(BoneRoleKind::Thigh),
            calf: side(BoneRoleKind::Calf),
            foot: side(BoneRoleKind::Foot),
        })
    }

    /// The chest — the last segment of the spine chain, which is what a torso
    /// pitch is written onto so the whole upper body follows.
    fn chest(&self) -> u16 {
        *self.spine.last().expect("checked non-empty")
    }

    /// **Is `a` an ancestor of `j`?**
    ///
    /// A chain compounds only through the bones that are actually in it. The
    /// spine's last segment is the arms' ancestor on a MetaHuman and need not
    /// be on every rig, and a clip that assumed it would apply the chest's
    /// twist to an arm that never inherited it.
    fn is_ancestor(&self, a: u16, j: Option<u16>) -> bool {
        let mut cur = j;
        while let Some(c) = cur {
            if c == a {
                return true;
            }
            cur = self.parents.get(c as usize).copied().flatten();
        }
        false
    }

    /// **Which way a bone POINTS at bind**, in the rig's own frame: from
    /// `joint` to `tip`, normalised. `None` when the rig has neither, or when
    /// the two sit on top of each other.
    fn bone_dir(&self, joint: Option<u16>, tip: Option<u16>) -> Option<[f32; 3]> {
        let (j, t) = (joint?, tip?);
        let d = self.bind_pos[t as usize] - self.bind_pos[j as usize];
        let len = d.length();
        (len > 1.0e-6).then(|| (d / len).to_array())
    }

    /// **Aim a bone whose ANCESTORS have already been rotated by `parent`** —
    /// and answer both the rotation in the rig's frame and the local delta.
    ///
    /// A chain compounds: rotating a thigh moves the knee, and the shin's own
    /// delta is then applied *under* that. Aiming the shin at a rig-frame
    /// direction without allowing for its parent leaves the leg straight, which
    /// is what the first spelling of this repair did — measured on the biped
    /// template: hip z 0.000, knee z 0.395, ankle z 0.650, an ankle FORWARD of
    /// the knee on a crouch.
    ///
    /// So the target is seen from under the parent's own rotation first. The
    /// world rotation comes back with it, because the next bone down needs it.
    fn aim_under(
        &self,
        parent: [f32; 4],
        joint: Option<u16>,
        tip: Option<u16>,
        target: [f64; 3],
    ) -> ([f32; 4], [f32; 4]) {
        let Some(j) = joint else {
            return (QUAT_ID, QUAT_ID);
        };
        let (Some(from), Some(t)) = (
            self.bone_dir(joint, tip),
            norm3([target[0] as f32, target[1] as f32, target[2] as f32]),
        ) else {
            return (QUAT_ID, QUAT_ID);
        };
        let world = min_arc(from, rotate_vec(quat_conj(parent), t));
        (world, local_of_world(self.bind_gq[j as usize], world))
    }

    /// **A rotation stated in the rig's frame, written in a joint's own** --
    /// so a caller can say "pitch the chest forward" without knowing which way
    /// the chest's local axes happen to point.
    fn in_frame_of(&self, joint: u16, world: [f32; 4]) -> [f32; 4] {
        local_of_world(self.bind_gq[joint as usize], world)
    }
}

/// The identity quaternion, `[x, y, z, w]`.
const QUAT_ID: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// Normalise, or `None` for a vector too short to have a direction.
fn norm3(v: [f32; 3]) -> Option<[f32; 3]> {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (len > 1.0e-6).then(|| [v[0] / len, v[1] / len, v[2] / len])
}

/// **The shortest rotation taking `a` to `b`**, both unit vectors.
///
/// Square roots and products only -- no trigonometry -- so it obeys the
/// module's determinism rule for [`quat_mul`]'s reason.
fn min_arc(a: [f32; 3], b: [f32; 3]) -> [f32; 4] {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    if dot < -1.0 + 1.0e-6 {
        // Opposed: every axis perpendicular to `a` is a half turn, and one of
        // them is picked rather than a NaN being produced.
        let axis = if a[0].abs() < 0.9 {
            [0.0, -a[2], a[1]]
        } else {
            [-a[1], a[0], 0.0]
        };
        let axis = norm3(axis).unwrap_or([0.0, 0.0, 1.0]);
        return [axis[0], axis[1], axis[2], 0.0];
    }
    let s = ((1.0 + dot) * 2.0).sqrt();
    let inv = 1.0 / s;
    let q = [cross[0] * inv, cross[1] * inv, cross[2] * inv, s * 0.5];
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
}

/// `G^-1 q G` -- a rotation stated in the rig's frame, written in the frame of
/// a joint whose bind global rotation is `g`.
fn local_of_world(g: [f32; 4], q: [f32; 4]) -> [f32; 4] {
    quat_mul(quat_mul(quat_conj(g), q), g)
}

/// The conjugate of a unit quaternion -- its inverse.
fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Rotate a vector by a unit quaternion. Products and sums only.
fn rotate_vec(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let t = [
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    ];
    [
        v[0] + w * t[0] + y * t[2] - z * t[1],
        v[1] + w * t[1] + z * t[0] - x * t[2],
        v[2] + w * t[2] + x * t[1] - y * t[0],
    ]
}

/// Bind-pose global translations, composed down the parent chain. Multiplies and
/// adds only — [`crate::locomotion`]'s own helper, spelled here because it is
/// private there and this module must not reach into it.
fn bind_globals(skeleton: &Skeleton) -> Vec<glam::Vec3> {
    let mut mats: Vec<glam::Mat4> = Vec::with_capacity(skeleton.len());
    for j in skeleton.joints() {
        let local = j.local_bind.to_mat4();
        let m = match j.parent {
            Some(p) => mats[p as usize] * local,
            None => local,
        };
        mats.push(m);
    }
    mats.iter()
        .map(|m| m.to_scale_rotation_translation().2)
        .collect()
}

/// A rotation of `angle_rad` about the joint's local **X**, written out rather
/// than built — [`crate::locomotion::quat_x`]'s rule, and its reason: every
/// `glam` rotation constructor is `f32::sin_cos`, which is libm, which is not
/// bit-portable, and these values are written into a `.inf_anim` on disk.
fn qx(angle_rad: f64) -> [f32; 4] {
    let half = angle_rad * 0.5;
    [psin64(half) as f32, 0.0, 0.0, pcos64(half) as f32]
}

/// The same about local **Y** — a twist.
fn qy(angle_rad: f64) -> [f32; 4] {
    let half = angle_rad * 0.5;
    [0.0, psin64(half) as f32, 0.0, pcos64(half) as f32]
}

/// The same about local **Z** — a roll or an abduction.
fn qz(angle_rad: f64) -> [f32; 4] {
    let half = angle_rad * 0.5;
    [0.0, 0.0, psin64(half) as f32, pcos64(half) as f32]
}

/// `n + 1` key times over `period`, closing exactly on it so a loop's seam is
/// exact rather than nearly exact.
fn times_of(period: f64, n: usize) -> Vec<f32> {
    (0..=n)
        .map(|i| (period * i as f64 / n as f64) as f32)
        .collect()
}

/// Push a rotation track onto `joint`, if the rig has it.
fn rot(tracks: &mut Vec<JointTrack>, joint: Option<u16>, times: &[f32], values: Vec<[f32; 4]>) {
    let Some(j) = joint else { return };
    let mut t = JointTrack::new(j);
    t.rotation = Some(QuatTrack::new(
        times.to_vec(),
        values,
        Interpolation::Linear,
    ));
    tracks.push(t);
}

/// The same for a joint the rig always has.
fn rot1(tracks: &mut Vec<JointTrack>, joint: u16, times: &[f32], values: Vec<[f32; 4]>) {
    rot(tracks, Some(joint), times, values);
}

/// **Author every clip this module writes**, in [`AUTHORED_CLIPS`] order.
///
/// A clip is grounded ([`crate::retarget::settle_to_ground`]) exactly when it
/// depicts a character touching the floor — the slide, the prone pair, the
/// get-up and the four cover clips do; the three swims and the two throws do
/// not, because a swimmer's soles are not on anything and a throw is an
/// upper-body overlay whose legs are the locomotion's.
pub fn author_clips(rig: &SkeletonAsset) -> Result<Vec<(String, AnimClip)>, AuthorError> {
    let r = Rig::of(rig)?;
    let mut out: Vec<(String, AnimClip)> = vec![
        ("INF_Slide".into(), slide(&r)),
        ("INF_Throw_Over".into(), throw(&r, true)),
        ("INF_Throw_Under".into(), throw(&r, false)),
        (
            "INF_Swim_Surface".into(),
            swim_stroke(&r, 60.0, 1.6, 140.0, 0.0),
        ),
        ("INF_Swim_Tread".into(), swim_tread(&r)),
        (
            "INF_Swim_Under".into(),
            swim_stroke(&r, 75.0, 2.0, 160.0, 6.0),
        ),
        ("INF_Prone_Idle".into(), prone(&r, false)),
        ("INF_Prone_Crawl".into(), prone(&r, true)),
        ("INF_GetUp_Standing".into(), get_up(&r)),
        ("INF_Cover_Low_Idle".into(), cover(&r, true, false)),
        ("INF_Cover_Low_Move".into(), cover(&r, true, true)),
        ("INF_Cover_High_Idle".into(), cover(&r, false, false)),
        ("INF_Cover_High_Move".into(), cover(&r, false, true)),
        ("INF_Cover_BlindFire".into(), blind_fire(&r)),
    ];
    for (name, clip) in out.iter_mut() {
        if matches!(
            name.as_str(),
            "INF_Slide"
                | "INF_Prone_Idle"
                | "INF_Prone_Crawl"
                | "INF_GetUp_Standing"
                | "INF_Cover_Low_Idle"
                | "INF_Cover_Low_Move"
                | "INF_Cover_High_Idle"
                | "INF_Cover_High_Move"
        ) {
            crate::retarget::settle_to_ground(clip, &rig.skeleton);
        }
    }
    Ok(out)
}

/// **THE SLIDE** — a crouch-sprint slide, 1.10 s, one-shot.
///
/// The kinematics, stated: the pelvis drops to 55 % of its bind height over the
/// first 30 % of the clip and holds; the lead (left) knee folds to 95° and the
/// trail leg extends 30° behind; the torso pitches 25° forward; the arms sweep
/// 40° back. The recovery is the last 20 %, where everything returns a third of
/// the way toward the bind so the blend out of `slide` into a stance has
/// somewhere to go.
///
/// It is a **one-shot**: `MovementMode::Slide` ends when the speed decays, and a
/// looping slide would restart the drop every 1.1 s.
fn slide(r: &Rig) -> AnimClip {
    let n = 12;
    let times = times_of(1.10, n);
    let mut tracks: Vec<JointTrack> = Vec::new();
    let drop = 0.45 * r.hip_height_m;

    let shape = |i: usize| -> f64 {
        // 0 → 1 over the first 30 %, hold, then back to 0.66 over the last 20 %.
        let a = i as f64 / n as f64;
        if a < 0.30 {
            a / 0.30
        } else if a > 0.80 {
            1.0 - (a - 0.80) / 0.20 * 0.34
        } else {
            1.0
        }
    };

    let mut hip = JointTrack::new(r.pelvis);
    hip.translation = Some(Vec3Track::new(
        times.clone(),
        (0..=n)
            .map(|i| {
                [
                    r.pelvis_bind[0],
                    r.pelvis_bind[1] - (drop * shape(i)) as f32,
                    r.pelvis_bind[2],
                ]
            })
            .collect(),
        Interpolation::Linear,
    ));
    tracks.push(hip);

    rot1(
        &mut tracks,
        r.chest(),
        &times,
        (0..=n).map(|i| qx(25f64.to_radians() * shape(i))).collect(),
    );
    // The lead leg folds under, the trail leg extends behind.
    rot(
        &mut tracks,
        r.thigh[0],
        &times,
        (0..=n)
            .map(|i| qx(-60f64.to_radians() * shape(i)))
            .collect(),
    );
    rot(
        &mut tracks,
        r.calf[0],
        &times,
        (0..=n)
            .map(|i| qx(-95f64.to_radians() * shape(i)))
            .collect(),
    );
    rot(
        &mut tracks,
        r.thigh[1],
        &times,
        (0..=n).map(|i| qx(30f64.to_radians() * shape(i))).collect(),
    );
    for side in 0..2 {
        rot(
            &mut tracks,
            r.upper_arm[side],
            &times,
            (0..=n).map(|i| qx(40f64.to_radians() * shape(i))).collect(),
        );
    }
    AnimClip::new("INF_Slide", tracks)
}

/// **TAKING COVER** (wave COV1) — a stance per class, moving or not.
///
/// `low` crouches (the character is behind a car's flank, a counter, a low
/// wall); otherwise it stands (a façade, a container). `moving` adds the
/// side-step the cover slide plays.
///
/// # What it depicts, and the two things that make it read as COVER
///
/// A cover pose is not a crouch and not an idle, and the difference is two
/// rotations rather than a different skeleton:
///
/// * **the torso is TWISTED toward the surface** (18° low, 22° high) while the
///   pelvis stays square to it, which is what "pressed against it" looks like
///   from behind — a body flat to a wall is a body facing the wall, and a
///   character in cover is facing ALONG it;
/// * **the head is turned further still** (35° / 40°), because the thing the
///   player is looking at is down the wall, not into it.
///
/// The near arm folds across the chest — which is where a weapon is held when
/// it is not being aimed, and what keeps the elbow out of the surface the
/// character is leaning on.
///
/// # The arms and the legs are AIMED, and that is a repair
///
/// Every rotation this clip writes is a delta in the JOINT's own bind frame,
/// and the first cut chose the axis by hand: `qx` for a shoulder lift, `qx` for
/// a hip fold, `qx` for a knee. On a UE-derived rig — which is every rig this
/// engine imports — a bone's local **X runs down its own length**, measured on
/// the island's shipped hero: `upperarm_l`'s local X reaches world
/// (0.61, -0.79, 0.03), which is its shoulder-to-elbow direction, and
/// `thigh_l`'s reaches (-0.05, 1.00, 0.02), which is straight up the femur. So
/// every one of those angles was a **twist about the bone**, not a fold of it:
/// the arms hung at the sides in the demo loop's frames (carried 184) and the
/// crouch was carried entirely by the pelvis translation.
///
/// [`Rig::aim`] replaces the guess with the rig's own answer: the clip states
/// where a bone should POINT, in the rig's frame, and the minimal-arc rotation
/// that gets it there is conjugated into the joint's own. Nothing here now
/// assumes which local axis is which.
///
/// The breath is the same 2° chest oscillation the prone idle carries, so a
/// character holding cover is not a photograph. `apply_breath`'s own additive
/// rides on top of this; the 2° here is what the clip does on its own, which is
/// what a viewer sees when no additive is layered.
///
/// # Which side is "near"
///
/// The **left**, always, and it is a statement rather than a measurement: the
/// engine's cover state knows which way the character is leaning
/// (`inf_ecs::cover::CoverSide`) and a mirrored pair of clips per class would
/// be eight files to say what one additive lean already says. The pose is
/// authored for a character with the wall on its left and the layer mirrors it;
/// stated here so a reader meeting a right-hand corner knows why there is no
/// `_RH` file beside this one, exactly as the mantle's own left/right split is
/// stated at `MantleState::left_hand`.
fn cover(r: &Rig, low: bool, moving: bool) -> AnimClip {
    let n = 12;
    let period = match (low, moving) {
        (true, false) => 3.20,
        (true, true) => 1.20,
        (false, false) => 3.60,
        (false, true) => 1.10,
    };
    let times = times_of(period, n);
    let tau = std::f64::consts::TAU;
    let ph = |i: usize| tau * i as f64 / n as f64;
    let mut tracks: Vec<JointTrack> = Vec::new();

    // The pelvis drops for a crouch and stays put for a stand. 62 % of hip
    // height is a little taller than the slide's 55 %: a character behind a car
    // is ready to rise over it, not sitting down.
    let drop = if low { 0.38 * r.hip_height_m } else { 0.0 };
    let mut hip = JointTrack::new(r.pelvis);
    hip.translation = Some(Vec3Track::new(
        times.clone(),
        (0..=n)
            .map(|i| {
                // A 1.5 cm bob on the move, so the side-step has weight.
                let bob = if moving {
                    0.015 * psin64(2.0 * ph(i))
                } else {
                    0.0
                };
                [
                    r.pelvis_bind[0],
                    r.pelvis_bind[1] - (drop + bob) as f32,
                    r.pelvis_bind[2],
                ]
            })
            .collect(),
        Interpolation::Linear,
    ));
    tracks.push(hip);

    // **Every rotation below is `bind * delta`.** See `Rig::bind_rot`.
    let of = |j: u16, q: [f32; 4]| -> [f32; 4] { quat_mul(r.bind_rot[j as usize], q) };
    let of_opt = |j: Option<u16>, q: [f32; 4]| -> [f32; 4] { j.map(|j| of(j, q)).unwrap_or(q) };

    // The chest: a pitch, a twist toward the surface, and the breath — stated
    // in the RIG's frame and written in the chest's own (`Rig::in_frame_of`).
    //
    // A pitch about `+X` tips the sternum toward `+Z`, and a twist about `+Y`
    // turns it toward `+X`, which is the character's left and therefore the
    // wall. Written as `qx`/`qy` on the joint they were a twist and a bend
    // respectively: the last spine joint's own local X runs UP the spine on
    // this rig (measured: `spine_05` local X reaches world (0.00, 0.98, -0.19)),
    // so the two axes were each other's.
    let pitch: f64 = if low { 12.0 } else { -8.0 };
    let twist: f64 = if low { 18.0 } else { 22.0 };
    let sway: f64 = if moving { 10.0 } else { 0.0 };
    // The chest's rotation IN THE RIG'S FRAME, per key — the arms hang under
    // it, so they need it as their parent.
    let chest_world = |i: usize| -> [f32; 4] {
        let breath = 2.0 * psin64(ph(i));
        quat_mul(
            qx((pitch + breath).to_radians()),
            qy((twist + sway * psin64(ph(i))).to_radians()),
        )
    };
    rot1(
        &mut tracks,
        r.chest(),
        &times,
        (0..=n)
            .map(|i| of(r.chest(), r.in_frame_of(r.chest(), chest_world(i))))
            .collect(),
    );

    // ── THE ARMS, POINTED RATHER THAN ROTATED ───────────────────────────────
    //
    // The near (left) arm folds across the chest and the far arm braces, and
    // both are said as DIRECTIONS in the rig's frame — down, forward, and how
    // far across the body's centre line — because an angle about a bone's own
    // axis is a twist and not a fold (`Rig::bind_gq`, and carried 184).
    //
    // Which way "across the chest" is comes off the RIG (`Rig::across_sign`):
    // the two rigs in this tree put the left shoulder on opposite sides of the
    // origin, and a hard-coded sign folds one arm and throws the other wide.
    let arm_dir = |side: usize, lift_deg: f64, across_deg: f64| -> [f64; 3] {
        let side_sign = r.across_sign[side];
        let (l, a) = (lift_deg.to_radians(), across_deg.to_radians());
        [
            side_sign * psin64(l) * psin64(a),
            -pcos64(l),
            psin64(l) * pcos64(a),
        ]
    };
    // The near arm's own two angles. A stand keeps the elbow a little higher
    // and a little further across than a crouch does: behind a wall the weapon
    // rides at the chest, behind a bonnet it rides lower.
    let (near_lift, near_across) = if low { (34.0, 26.0) } else { (30.0, 30.0) };
    let (near_fore_lift, near_fore_across) = if low { (86.0, 66.0) } else { (84.0, 72.0) };
    // Each arm is one chain: the shoulder aims under the CHEST, and the elbow
    // aims under the shoulder's own answer.
    let mut arm_local: [[Vec<[f32; 4]>; 2]; 2] = Default::default();
    for i in 0..=n {
        for (side, slot) in arm_local.iter_mut().enumerate() {
            // The chest is the arms' ancestor on a MetaHuman; whether it is on
            // any given rig is a question about that rig.
            let parent = if r.is_ancestor(r.chest(), r.upper_arm[side]) {
                chest_world(i)
            } else {
                QUAT_ID
            };
            let (lift, across, fore_lift, fore_across) = if side == 0 {
                (near_lift, near_across, near_fore_lift, near_fore_across)
            } else {
                (22.0, 12.0, 58.0, 28.0)
            };
            let (upper_world, upper_local) = r.aim_under(
                parent,
                r.upper_arm[side],
                r.lower_arm[side],
                arm_dir(side, lift, across),
            );
            let (_, lower_local) = r.aim_under(
                quat_mul(parent, upper_world),
                r.lower_arm[side],
                r.hand[side],
                arm_dir(side, fore_lift, fore_across),
            );
            slot[0].push(of_opt(r.upper_arm[side], upper_local));
            slot[1].push(of_opt(r.lower_arm[side], lower_local));
        }
    }
    for (side, slot) in arm_local.iter_mut().enumerate() {
        rot(
            &mut tracks,
            r.upper_arm[side],
            &times,
            std::mem::take(&mut slot[0]),
        );
        rot(
            &mut tracks,
            r.lower_arm[side],
            &times,
            std::mem::take(&mut slot[1]),
        );
    }

    // ── THE LEGS, POINTED THE SAME WAY ──────────────────────────────────────
    //
    // Standing still they take the stance; moving, they side-step — alternating
    // hip abduction with a knee fold, which is a shuffle rather than a walk,
    // because a cover slide never crosses its feet.
    //
    // The thigh is aimed forward-and-down by the hip fold and the shin BACK
    // from it by the knee fold, so the two together are a crouch a viewer can
    // see. Written as `qx` on the joints they were twists about the femur and
    // the tibia (`thigh_l` local X reaches world (-0.05, 1.00, 0.02) — straight
    // up, which is the leg's own length), and a crouch made of two twists is a
    // character standing to attention with its knees rolled.
    let hip_fold: f64 = if low { 55.0 } else { 4.0 };
    let knee_fold: f64 = if low { 75.0 } else { 4.0 };
    // A leg direction: `fold` forward from straight down, then `abduct` out to
    // the character's own left (`+X`) — one spelling for both bones.
    let leg_dir = |fold_deg: f64, abduct_deg: f64| -> [f64; 3] {
        let (f, a) = (fold_deg.to_radians(), abduct_deg.to_radians());
        [pcos64(f) * psin64(a), -pcos64(f) * pcos64(a), psin64(f)]
    };
    let mut leg_local: [[Vec<[f32; 4]>; 2]; 2] = Default::default();
    for i in 0..=n {
        for (side, slot) in leg_local.iter_mut().enumerate() {
            let phase = if side == 0 { 0.0 } else { std::f64::consts::PI };
            let step_amp = if moving { 22.0 } else { 0.0 };
            let knee_amp = if moving { 30.0 } else { 0.0 };
            let abduct = step_amp * psin64(ph(i) + phase);
            let knee = knee_fold + knee_amp * (0.5 - 0.5 * pcos64(ph(i) + phase));
            // The pelvis is TRANSLATED and never rotated, so a thigh's parent
            // is the identity; the shin's is the thigh's own answer.
            let (thigh_world, thigh_local) = r.aim_under(
                QUAT_ID,
                r.thigh[side],
                r.calf[side],
                leg_dir(hip_fold, abduct),
            );
            let (_, calf_local) = r.aim_under(
                thigh_world,
                r.calf[side],
                r.foot[side],
                leg_dir(hip_fold - knee, abduct),
            );
            slot[0].push(of_opt(r.thigh[side], thigh_local));
            slot[1].push(of_opt(r.calf[side], calf_local));
        }
    }
    for (side, slot) in leg_local.iter_mut().enumerate() {
        rot(
            &mut tracks,
            r.thigh[side],
            &times,
            std::mem::take(&mut slot[0]),
        );
        rot(
            &mut tracks,
            r.calf[side],
            &times,
            std::mem::take(&mut slot[1]),
        );
    }
    let name = match (low, moving) {
        (true, false) => "INF_Cover_Low_Idle",
        (true, true) => "INF_Cover_Low_Move",
        (false, false) => "INF_Cover_High_Idle",
        (false, true) => "INF_Cover_High_Move",
    };
    AnimClip::new(name, tracks)
}

/// **FIRING WITHOUT LOOKING** (wave WPN2e) — the upper-body one-shot COV1 priced
/// and did not build. [`BLIND_FIRE_S`], one-shot, an **additive overlay**.
///
/// # What it depicts, stated
///
/// The weapon goes over the cover and the head does not. Over the first 45 % of
/// the clip:
///
/// * the **weapon (near) arm** goes from the cover stance's own numbers
///   (34° lift, 26° across at the shoulder; 86° / 66° at the forearm — the same
///   figures [`cover`] holds it at) to **150° lift / 8° across** at the shoulder
///   and **168° / 4°** at the forearm: the arm straight up and slightly across
///   the body's centre line, which is where a hand holding a weapon over a
///   parapet is;
/// * the **far arm** tucks in — 20° lift, 22° across; 70° / 44° — because the
///   shooter is not bracing anything, it is hiding;
/// * the **chest** pitches from the stance's 12° forward to **34°**, which is
///   the head going down. The whole point of the pose: a viewer must be able to
///   see that this character cannot see what it is shooting at.
///
/// It **holds** to 75 % and then returns a third of the way back over the last
/// quarter, so the blend out of the additive has somewhere to go — the slide's
/// own recovery rule.
///
/// # Why no legs and no pelvis
///
/// Because it is an ADDITIVE over whatever cover stance the machine is in, and
/// that stance owns the crouch. A blind-fire clip that also wrote a pelvis drop
/// would fight `INF_Cover_Low_Idle` for the same channel and win by being later
/// in the layer stack, so a crouched shooter would stand up to hide.
///
/// Five joints — chest, two upper arms, two lower arms — which is over
/// `AUTHORED_MIN_JOINTS` and is the same shape the throws are.
fn blind_fire(r: &Rig) -> AnimClip {
    let n = 8;
    let times = times_of(BLIND_FIRE_S, n);
    // The clip's own shape: rise over the first 45 %, hold to 75 %, recover a
    // third of the way over the last quarter.
    let shape = |i: usize| -> f64 {
        let a = i as f64 / n as f64;
        if a <= 0.45 {
            a / 0.45
        } else if a <= 0.75 {
            1.0
        } else {
            1.0 - (a - 0.75) / 0.25 / 3.0
        }
    };
    let mut tracks: Vec<JointTrack> = Vec::new();
    // **Every rotation below is `bind * delta`** — `cover`'s rule, and its
    // reason: this is a STANDING (or crouching) pose a small delta away from
    // bind, so writing the local outright would put the shoulder wherever the
    // clavicle's frame happens to point.
    let of = |j: u16, q: [f32; 4]| -> [f32; 4] { quat_mul(r.bind_rot[j as usize], q) };
    let of_opt = |j: Option<u16>, q: [f32; 4]| -> [f32; 4] { j.map(|j| of(j, q)).unwrap_or(q) };
    // The chest, in the RIG's frame, written in the chest's own.
    let chest_world = |i: usize| -> [f32; 4] {
        let pitch = 12.0 + (34.0 - 12.0) * shape(i);
        quat_mul(qx(pitch.to_radians()), qy(18.0_f64.to_radians()))
    };
    rot1(
        &mut tracks,
        r.chest(),
        &times,
        (0..=n)
            .map(|i| of(r.chest(), r.in_frame_of(r.chest(), chest_world(i))))
            .collect(),
    );
    // The arms, POINTED rather than rotated — `cover`'s `arm_dir`, verbatim,
    // including the measured `across_sign`: the two rigs in this tree put the
    // left shoulder on opposite sides of the origin.
    let arm_dir = |side: usize, lift_deg: f64, across_deg: f64| -> [f64; 3] {
        let side_sign = r.across_sign[side];
        let (l, a) = (lift_deg.to_radians(), across_deg.to_radians());
        [
            side_sign * psin64(l) * psin64(a),
            -pcos64(l),
            psin64(l) * pcos64(a),
        ]
    };
    let mut arm_local: [[Vec<[f32; 4]>; 2]; 2] = Default::default();
    for i in 0..=n {
        let t = shape(i);
        for (side, slot) in arm_local.iter_mut().enumerate() {
            let parent = if r.is_ancestor(r.chest(), r.upper_arm[side]) {
                chest_world(i)
            } else {
                QUAT_ID
            };
            // Near arm: the cover stance's own start, and up over the top.
            // Far arm: tucked, and it barely moves.
            let (lift, across, fore_lift, fore_across) = if side == 0 {
                (
                    34.0 + (150.0 - 34.0) * t,
                    26.0 + (8.0 - 26.0) * t,
                    86.0 + (168.0 - 86.0) * t,
                    66.0 + (4.0 - 66.0) * t,
                )
            } else {
                (
                    22.0 + (20.0 - 22.0) * t,
                    12.0 + (22.0 - 12.0) * t,
                    58.0 + (70.0 - 58.0) * t,
                    28.0 + (44.0 - 28.0) * t,
                )
            };
            let (upper_world, upper_local) = r.aim_under(
                parent,
                r.upper_arm[side],
                r.lower_arm[side],
                arm_dir(side, lift, across),
            );
            let (_, lower_local) = r.aim_under(
                quat_mul(parent, upper_world),
                r.lower_arm[side],
                r.hand[side],
                arm_dir(side, fore_lift, fore_across),
            );
            slot[0].push(of_opt(r.upper_arm[side], upper_local));
            slot[1].push(of_opt(r.lower_arm[side], lower_local));
        }
    }
    for (side, slot) in arm_local.iter_mut().enumerate() {
        rot(
            &mut tracks,
            r.upper_arm[side],
            &times,
            std::mem::take(&mut slot[0]),
        );
        rot(
            &mut tracks,
            r.lower_arm[side],
            &times,
            std::mem::take(&mut slot[1]),
        );
    }
    AnimClip::new("INF_Cover_BlindFire", tracks)
}

/// Hamilton product of two `[x, y, z, w]` quaternions.
///
/// Spelled here rather than through `glam::Quat`: every rotation this module
/// writes is committed content, and `glam`'s constructors are `f32::sin_cos`.
/// This one only multiplies and adds, so composing two `psin64`-built rotations
/// stays inside the module's own determinism rule.
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// **THE THROW** — overhand (`over`) or underhand, 0.85 / 0.75 s, one-shot.
///
/// The right shoulder sweeps through its own X while the elbow unfolds and the
/// chest rotates into the throw; the release is at 65 % of the clip, which is
/// where the arm passes the shoulder line. Written on the RIGHT arm and the
/// chest only: it is consumed as an **upper-body additive** over whatever the
/// locomotion graph is doing (WPN1's throwable items), so a track on a leg would
/// fight the walk it is layered over.
fn throw(r: &Rig, over: bool) -> AnimClip {
    let n = 10;
    // The two durations are named constants, so the fixed step's own clock
    // (`anim_bridge::start_throw`) and the clip it drives cannot disagree.
    let dur = if over { THROW_OVER_S } else { THROW_UNDER_S };
    let times = times_of(dur, n);
    let (from, to) = if over { (-80.0, 110.0) } else { (50.0, -70.0) };
    let (elbow_from, elbow_to) = if over { (100.0, 10.0) } else { (60.0, 5.0) };
    let twist = if over { 25.0 } else { 10.0 };
    let mut tracks: Vec<JointTrack> = Vec::new();
    let a = |i: usize| i as f64 / n as f64;
    rot(
        &mut tracks,
        r.upper_arm[1],
        &times,
        (0..=n)
            .map(|i| qx((from + (to - from) * a(i)).to_radians()))
            .collect(),
    );
    rot(
        &mut tracks,
        r.lower_arm[1],
        &times,
        (0..=n)
            .map(|i| qx((elbow_from + (elbow_to - elbow_from) * a(i)).to_radians()))
            .collect(),
    );
    rot1(
        &mut tracks,
        r.chest(),
        &times,
        (0..=n)
            .map(|i| qy((-twist + 2.0 * twist * a(i)).to_radians()))
            .collect(),
    );
    AnimClip::new(
        if over {
            "INF_Throw_Over"
        } else {
            "INF_Throw_Under"
        },
        tracks,
    )
}

/// **A SWIMMING STROKE** — the surface crawl and the underwater one, which are
/// the same generator at a different torso pitch, period, reach and roll.
///
/// `pitch_deg` brings the chest toward horizontal (the whole body follows it,
/// because the chest is the top of the spine chain); the shoulders alternate a
/// full `reach_deg` sweep half a period apart; the legs flutter at **twice** the
/// arm rate, which is what a front crawl does; `roll_deg` rolls the chest with
/// the stroke and is zero at the surface, where a swimmer's face has to stay up.
fn swim_stroke(r: &Rig, pitch_deg: f64, period: f64, reach_deg: f64, roll_deg: f64) -> AnimClip {
    let n = 16;
    let times = times_of(period, n);
    let mut tracks: Vec<JointTrack> = Vec::new();
    let tau = std::f64::consts::TAU;
    let ph = |i: usize| tau * i as f64 / n as f64;

    rot1(
        &mut tracks,
        r.chest(),
        &times,
        (0..=n)
            .map(|i| {
                // pitch about X, roll about Z; composed by hand because a `Quat`
                // product of two written-out halves is multiplies and adds only.
                let p = qx(pitch_deg.to_radians());
                let ro = qz(roll_deg.to_radians() * psin64(ph(i)));
                (glam::Quat::from_array(p) * glam::Quat::from_array(ro)).to_array()
            })
            .collect(),
    );
    for side in 0..2 {
        let offset = if side == 0 { 0.0 } else { tau * 0.5 };
        rot(
            &mut tracks,
            r.upper_arm[side],
            &times,
            (0..=n)
                .map(|i| qx(reach_deg.to_radians() * 0.5 * psin64(ph(i) + offset)))
                .collect(),
        );
        rot(
            &mut tracks,
            r.lower_arm[side],
            &times,
            (0..=n)
                .map(|i| qx(35f64.to_radians() * (0.5 + 0.5 * pcos64(ph(i) + offset))))
                .collect(),
        );
        rot(
            &mut tracks,
            r.thigh[side],
            &times,
            (0..=n)
                .map(|i| qx(14f64.to_radians() * psin64(2.0 * ph(i) + offset)))
                .collect(),
        );
        rot(
            &mut tracks,
            r.calf[side],
            &times,
            (0..=n)
                .map(|i| qx(-10f64.to_radians() * (0.5 + 0.5 * psin64(2.0 * ph(i) + offset))))
                .collect(),
        );
    }
    AnimClip::new(
        if roll_deg > 0.0 {
            "INF_Swim_Under"
        } else {
            "INF_Swim_Surface"
        },
        tracks,
    )
}

/// **TREADING WATER** — 2.40 s, looping. Upright (20° of pitch), the shoulders
/// sculling ±35° out of phase with the legs' ±25° cycle.
fn swim_tread(r: &Rig) -> AnimClip {
    let n = 16;
    let period = 2.40;
    let times = times_of(period, n);
    let tau = std::f64::consts::TAU;
    let ph = |i: usize| tau * i as f64 / n as f64;
    let mut tracks: Vec<JointTrack> = Vec::new();
    rot1(
        &mut tracks,
        r.chest(),
        &times,
        (0..=n).map(|_| qx(20f64.to_radians())).collect(),
    );
    for side in 0..2 {
        let offset = if side == 0 { 0.0 } else { tau * 0.5 };
        rot(
            &mut tracks,
            r.upper_arm[side],
            &times,
            (0..=n)
                .map(|i| qz(35f64.to_radians() * psin64(ph(i) + offset)))
                .collect(),
        );
        rot(
            &mut tracks,
            r.thigh[side],
            &times,
            (0..=n)
                .map(|i| qx(25f64.to_radians() * psin64(ph(i) + offset + tau * 0.25)))
                .collect(),
        );
    }
    AnimClip::new("INF_Swim_Tread", tracks)
}

/// **PRONE** — the idle (3.00 s, a 2° breath) and the crawl (1.80 s, alternating
/// 35° hip and 55° elbow drive), both looping.
///
/// The pelvis drops to 12 % of its bind height and the chest pitches 88°, which
/// is what puts a body flat; the elbows fold to 90° so the forearms take the
/// weight. `MovementMode::Prone` has existed since P29.3 with no clip at all.
fn prone(r: &Rig, crawl: bool) -> AnimClip {
    let n = 12;
    let period = if crawl { 1.80 } else { 3.00 };
    let times = times_of(period, n);
    let tau = std::f64::consts::TAU;
    let ph = |i: usize| tau * i as f64 / n as f64;
    let mut tracks: Vec<JointTrack> = Vec::new();

    let mut hip = JointTrack::new(r.pelvis);
    hip.translation = Some(Vec3Track::new(
        times.clone(),
        (0..=n)
            .map(|_| {
                [
                    r.pelvis_bind[0],
                    (0.12 * r.hip_height_m) as f32,
                    r.pelvis_bind[2],
                ]
            })
            .collect(),
        Interpolation::Linear,
    ));
    tracks.push(hip);

    rot1(
        &mut tracks,
        r.chest(),
        &times,
        (0..=n)
            .map(|i| {
                let breath = if crawl {
                    0.0
                } else {
                    2f64.to_radians() * psin64(ph(i))
                };
                qx(88f64.to_radians() + breath)
            })
            .collect(),
    );
    for side in 0..2 {
        let offset = if side == 0 { 0.0 } else { tau * 0.5 };
        let drive = |i: usize, amp: f64| {
            if crawl {
                amp.to_radians() * psin64(ph(i) + offset)
            } else {
                0.0
            }
        };
        rot(
            &mut tracks,
            r.lower_arm[side],
            &times,
            (0..=n)
                .map(|i| qx(90f64.to_radians() + drive(i, 55.0)))
                .collect(),
        );
        rot(
            &mut tracks,
            r.thigh[side],
            &times,
            (0..=n)
                .map(|i| qx(-88f64.to_radians() + drive(i, 35.0)))
                .collect(),
        );
    }
    AnimClip::new(
        if crawl {
            "INF_Prone_Crawl"
        } else {
            "INF_Prone_Idle"
        },
        tracks,
    )
}

/// **THE STANDING GET-UP** — 1.20 s, one-shot.
///
/// ALS ships only the CROUCHED get-ups (`ALS_CLF_GetUp_Front/Back`), which is
/// what a character plays when a ragdoll leaves it on the floor. A character
/// whose ragdoll settled on its feet needs the other one, and the ragdoll bridge
/// has measured that since P29.4 (`RagdollRuntime::pelvis`) with nothing reading
/// it.
///
/// The pelvis rises from 20 % to 100 % of its bind height while the torso
/// unfolds 70° → 0° and the knees straighten 100° → 0°, on a smoothstep so the
/// character does not shoot up on the first frame.
fn get_up(r: &Rig) -> AnimClip {
    let n = 12;
    let period = 1.20;
    let times = times_of(period, n);
    let mut tracks: Vec<JointTrack> = Vec::new();
    // `warp_ease` is this crate's own smoothstep, and it is a polynomial.
    let a = |i: usize| crate::warp::warp_ease(i as f64 / n as f64);

    let mut hip = JointTrack::new(r.pelvis);
    hip.translation = Some(Vec3Track::new(
        times.clone(),
        (0..=n)
            .map(|i| {
                let f = 0.20 + 0.80 * a(i);
                [
                    r.pelvis_bind[0],
                    (f * f64::from(r.pelvis_bind[1])) as f32,
                    r.pelvis_bind[2],
                ]
            })
            .collect(),
        Interpolation::Linear,
    ));
    tracks.push(hip);

    rot1(
        &mut tracks,
        r.chest(),
        &times,
        (0..=n)
            .map(|i| qx(70f64.to_radians() * (1.0 - a(i))))
            .collect(),
    );
    for side in 0..2 {
        rot(
            &mut tracks,
            r.thigh[side],
            &times,
            (0..=n)
                .map(|i| qx(-70f64.to_radians() * (1.0 - a(i))))
                .collect(),
        );
        rot(
            &mut tracks,
            r.calf[side],
            &times,
            (0..=n)
                .map(|i| qx(-100f64.to_radians() * (1.0 - a(i))))
                .collect(),
        );
        rot(
            &mut tracks,
            r.upper_arm[side],
            &times,
            (0..=n)
                .map(|i| qx(35f64.to_radians() * (1.0 - a(i))))
                .collect(),
        );
    }
    AnimClip::new("INF_GetUp_Standing", tracks)
}

/// The role table an authored set needs, for a caller that wants to know before
/// it asks.
pub fn can_author(rig: &SkeletonAsset) -> bool {
    Rig::of(rig).is_ok()
}

/// How many of a rig's joints an authored set moves, by clip — the number the
/// gate's shell test compares against.
pub fn authored_joint_counts(rig: &SkeletonAsset) -> Result<Vec<(String, usize)>, AuthorError> {
    Ok(author_clips(rig)?
        .into_iter()
        .map(|(n, c)| (n, c.tracks.len()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::{build_template, BodyParams, BodyPlan};

    fn biped() -> SkeletonAsset {
        build_template(BodyPlan::Biped, &BodyParams::default()).unwrap()
    }

    /// **Every clip this module names is authored, and none of them is a shell.**
    #[test]
    fn every_authored_clip_moves_real_joints() {
        let rig = biped();
        let set = author_clips(&rig).expect("the mannequin has a role table");
        let names: Vec<&str> = set.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, AUTHORED_CLIPS, "{names:?}");
        for (name, clip) in &set {
            assert!(
                clip.tracks.len() >= 3,
                "{name} drives {} joints, which is a shell",
                clip.tracks.len()
            );
            assert!(clip.duration > 0.0, "{name} has no duration");
            for t in &clip.tracks {
                assert!(
                    (t.joint as usize) < rig.skeleton.len(),
                    "{name} drives joint {} on a {}-joint rig",
                    t.joint,
                    rig.skeleton.len()
                );
            }
        }
    }

    /// **Two derivations from the same rig are byte-identical** — the rule every
    /// generator in this crate is held to, because these bytes are committed.
    #[test]
    fn the_same_rig_authors_the_same_bytes() {
        let rig = biped();
        let a = author_clips(&rig).unwrap();
        let b = author_clips(&rig).unwrap();
        assert_eq!(a, b);
    }

    /// **THE COVER STANCE FOLDS AN ARM AND BENDS A KNEE** — the arm carried 184
    /// asked for, and the one the wave did not have.
    ///
    /// The wave's own anti-vacuity arm compares the cover pose against the
    /// crouch idle and passes when 55 joints differ. A pose made entirely of
    /// TWISTS about the bones' own long axes differs on 55 joints too: that is
    /// what `qx` on an `upperarm_l` whose local X runs shoulder-to-elbow
    /// produces, it is what the shipped clips were, and it is why the demo
    /// loop's frames show a hero standing to attention behind a wall.
    ///
    /// So this measures the POSE, in world space, on two things a twist cannot
    /// fake:
    ///
    /// * the near HAND crosses toward the body's centre line — the fold;
    /// * the KNEE of a low cover is FORWARD of both the hip and the ankle — the
    ///   bend. A femur rolled about its own length leaves the knee exactly
    ///   where the bind put it.
    #[test]
    fn the_cover_stance_folds_the_near_arm_and_bends_the_knee() {
        let rig = biped();
        let set = author_clips(&rig).expect("the mannequin has a role table");
        let roles = rig.role_index();
        let bind = crate::pose::Pose::rest(&rig.skeleton);
        let g_bind = crate::pose::global_transforms(&rig.skeleton, &bind);
        let at = |g: &[glam::Mat4], j: u16| -> glam::Vec3 {
            g[j as usize].to_scale_rotation_translation().2
        };
        let joint = |k: BoneRoleKind, side: BoneSide| roles.first(k, side).expect("a biped bone");
        let pelvis = joint(BoneRoleKind::Pelvis, BoneSide::Center);
        let shoulder = joint(BoneRoleKind::UpperArm, BoneSide::Left);
        let hand = joint(BoneRoleKind::Hand, BoneSide::Left);
        let hip = joint(BoneRoleKind::Thigh, BoneSide::Left);
        let knee = joint(BoneRoleKind::Calf, BoneSide::Left);
        let ankle = joint(BoneRoleKind::Foot, BoneSide::Left);

        for name in ["INF_Cover_High_Idle", "INF_Cover_Low_Idle"] {
            let clip = &set
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("{name} is not in the authored set"))
                .1;
            let pose = crate::pose::sample_clip(&rig.skeleton, clip, 0.0, true);
            let g = crate::pose::global_transforms(&rig.skeleton, &pose);
            // ── the fold. `across` is how far the hand is from the body's own
            //    centre line, measured on the axis the shoulders are spread
            //    along, so it needs no assumption about which way that is.
            let axis = (at(&g_bind, shoulder) - at(&g_bind, pelvis)).x.signum();
            let before = (at(&g_bind, hand).x - at(&g_bind, pelvis).x) * axis;
            let after = (at(&g, hand).x - at(&g, pelvis).x) * axis;
            println!(
                "{name}: the near hand is {before:.4} m out at bind and {after:.4} m in the stance"
            );
            assert!(
                after < 0.0 && after < before - 0.5,
                "{name}: the near hand is {after:.4} m from the centre line and bind has it\
                 at {before:.4} m -- it is twisted, not folded"
            );
            // ── the bend, on the low stance only: a stand has straight legs.
            if name == "INF_Cover_Low_Idle" {
                let (h, k, a) = (at(&g, hip), at(&g, knee), at(&g, ankle));
                let mid = (h.z + a.z) * 0.5;
                println!(
                    "{name}: hip z {:.4}, knee z {:.4}, ankle z {:.4}",
                    h.z, k.z, a.z
                );
                assert!(
                    (k.z - mid).abs() > 0.05,
                    "{name}'s knee sits {:.4} m from the hip-ankle midpoint — the leg is \
                     straight and the crouch is the pelvis translation alone",
                    k.z - mid
                );
                assert!(
                    at(&g, pelvis).y < at(&g_bind, pelvis).y - 0.10,
                    "{name} did not drop the pelvis"
                );
            }
        }
    }

    /// **A rig with no role table refuses by name** rather than authoring a set
    /// of empty clips, which is the failure that looks most like success.
    #[test]
    fn a_rig_with_no_roles_refuses_and_says_which_one() {
        let mut rig = biped();
        rig.roles.clear();
        let err = author_clips(&rig).unwrap_err();
        assert_eq!(err, AuthorError::MissingRole { role: "Pelvis" });
        assert!(!can_author(&rig));
    }

    /// **The throws are UPPER BODY only.** They are consumed as an additive over
    /// the locomotion pose, so a track on a leg would fight the walk it is
    /// layered over.
    #[test]
    fn a_throw_drives_no_leg() {
        let rig = biped();
        let roles = rig.role_index();
        let legs: std::collections::BTreeSet<u16> = roles
            .rows()
            .iter()
            .filter(|r| {
                matches!(
                    r.kind,
                    BoneRoleKind::Thigh | BoneRoleKind::Calf | BoneRoleKind::Foot
                )
            })
            .map(|r| r.joint)
            .collect();
        for (name, clip) in author_clips(&rig).unwrap() {
            if !name.starts_with("INF_Throw") {
                continue;
            }
            for t in &clip.tracks {
                assert!(
                    !legs.contains(&t.joint),
                    "{name} drives leg joint {}",
                    t.joint
                );
            }
        }
    }
}
