//! **The sets the donor does not ship** (wave CHAR1b.2) — slide, throwing,
//! swimming, prone and the standing get-up, derived from the rig that will play
//! them.
//!
//! # Why these nine clips are generated and not imported
//!
//! ALS ships 164 sequences and a census by name found **none** of these: no
//! slide, no throw, no swim, no prone, no standing get-up, no cover, no vault
//! (`movement_sets` in the island manifest: `mantle 13`; the rest `0`). The
//! engine's own catalogue has modes for four of them —
//! `MovementMode::{Slide, Prone, SwimSurface, SwimUnder}` — and a mode with no
//! clip is a character sliding in its idle pose.
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
    pelvis: u16,
    pelvis_bind: [f32; 3],
    hip_height_m: f64,
    spine: Vec<u16>,
    upper_arm: [Option<u16>; 2],
    lower_arm: [Option<u16>; 2],
    thigh: [Option<u16>; 2],
    calf: [Option<u16>; 2],
}

impl Rig {
    fn of(rig: &SkeletonAsset) -> Result<Self, AuthorError> {
        let roles = rig.role_index();
        let sk = &rig.skeleton;
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
        Ok(Self {
            pelvis,
            pelvis_bind: sk.joints()[pelvis as usize].local_bind.translation,
            hip_height_m: if h > 1.0e-4 { h } else { 0.95 },
            spine,
            upper_arm: side(BoneRoleKind::UpperArm),
            lower_arm: side(BoneRoleKind::LowerArm),
            thigh: side(BoneRoleKind::Thigh),
            calf: side(BoneRoleKind::Calf),
        })
    }

    /// The chest — the last segment of the spine chain, which is what a torso
    /// pitch is written onto so the whole upper body follows.
    fn chest(&self) -> u16 {
        *self.spine.last().expect("checked non-empty")
    }
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
/// depicts a character touching the floor — the slide, the prone pair and the
/// get-up do; the three swims and the two throws do not, because a swimmer's
/// soles are not on anything and a throw is an upper-body overlay whose legs are
/// the locomotion's.
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
    ];
    for (name, clip) in out.iter_mut() {
        if matches!(
            name.as_str(),
            "INF_Slide" | "INF_Prone_Idle" | "INF_Prone_Crawl" | "INF_GetUp_Standing"
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
    let dur = if over { 0.85 } else { 0.75 };
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
