//! **A body that is not walking** (wave VEN1b) — the seated and dancing poses
//! a venue's crowd wears, as authored clip data addressed by BONE ROLE.
//!
//! # Why a role-addressed clip and not an `.inf_anim`
//!
//! The obvious form is the one the mandate names first: two clips in the
//! character's own asset folder and two states on its `.inf_sm`. It was priced
//! and declined, and the reason is not effort:
//!
//! * a `.inf_anim` cannot be authored as data. `inf_anim::text` is a
//!   `StateMachine` door and nothing else; the only ways a clip enters this
//!   engine are glTF import and Rust generation (`locomotion::build_locomotion`);
//! * the crowd wears the level's own rig, which on the island is
//!   `samples/starter-character` — a folder whose committed bytes are
//!   **literally `build_character`'s output**, byte-locked by
//!   `the_starter_character_builds_clean_and_reproducibly`. Two more clips and
//!   two more `CharacterIds` rows would re-bless that lock, and the lock is a
//!   lock on the **New Character wizard**. A wave about nightlife has no
//!   business moving the wizard's output;
//! * and a clip authored against joint INDICES is a clip for one rig.
//!   `build_locomotion`'s own doc says so — its cycles are generated per rig
//!   "which is the whole reason they are generated rather than shipped".
//!
//! So the clip is authored here, in Rust, against the **role table** every
//! `.inf_skel` has carried since v3 — which is what a retargeted clip is — and
//! applied inside [`crate::pose`]'s own evaluation, at the one door both hosts
//! run. No asset, no wizard, no schema, and the result is folded into the pose
//! trace joint by joint like every other pose writer.
//!
//! # The delta is in the PARENT's frame
//!
//! [`apply_posture`] composes `delta * local`, which rotates a joint in its
//! parent's frame. Every rig this engine generates binds with an **identity
//! rotation** (`manny.rs` asserts it over all 161 joints), so a parent's frame
//! is the model's: `+Y` up, `+Z` the way the body faces, `+X` its left. That is
//! what makes an authored table of euler degrees readable — `thigh: −85° about
//! X` swings a leg from straight down to straight forward, which is what
//! sitting is.
//!
//! A rig with rotated binds still poses; the numbers simply mean what they mean
//! in ITS parents' frames. `a_generated_rig_binds_without_rotation` is the arm
//! that says the assumption holds for everything this engine makes.
//!
//! # The drop is DERIVED, the rotations are AUTHORED
//!
//! A sitting body's pelvis is one thigh-length lower than a standing one's —
//! the thigh goes horizontal and the shin takes over as the vertical member —
//! so [`sit_drop_m`] measures the rig's own femur rather than carrying a metre
//! constant that is wrong for every character but one. That is
//! `locomotion.rs`'s pattern (its gait thresholds come out of the same bind
//! pose) applied to a posture.
//!
//! # Portable math
//!
//! Sampling is a linear interpolation between two keys and a `rem_euclid`;
//! composition is a quaternion product. The euler → quaternion conversion goes
//! through [`inf_math::psin`]/[`inf_math::pcos`] and never `glam`'s
//! `Quat::from_euler`, which is `sin_cos` inside — this pose reaches
//! `pose_state_bytes` and therefore the replay trace two hosts compare (P14).

use glam::{Quat, Vec3};

use crate::asset::SkeletonAsset;
use crate::roles::{BoneRoleKind, BoneSide, RoleIndex};
use crate::Pose;

/// **What a body is doing instead of standing.**
///
/// Three, and no `Stand`: standing is the absence of a posture, and an enum with
/// an identity member would put a branch that does nothing on every posed
/// character in the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Posture {
    /// Seated — hips dropped a femur, thighs forward, shins down.
    Sit,
    /// Dancing on the spot — a weight shift, a hip sway and the arms up.
    Dance,
    /// **Down on one knee** (wave EMS2) — a paramedic working on somebody who
    /// is on the ground.
    Kneel,
}

impl Posture {
    /// A stable short name for diagnostics and gate traces.
    pub fn name(self) -> &'static str {
        match self {
            Posture::Sit => "sit",
            Posture::Dance => "dance",
            Posture::Kneel => "kneel",
        }
    }

    /// The authored clip this posture plays.
    pub fn clip(self) -> &'static PostureClip {
        match self {
            Posture::Sit => &SIT,
            Posture::Dance => &DANCE,
            Posture::Kneel => &KNEEL,
        }
    }
}

/// **Which joints of a role chain a track drives.**
///
/// A spine is several bones and a thigh is one, and a table that could only say
/// "the first" would put a sitting body's whole recline on its hips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pick {
    /// The chain's first joint in rig order — the hips of a spine, the only
    /// thigh.
    First,
    /// The chain's last — the chest of a spine.
    Last,
    /// Every joint of the chain, each taking the whole authored delta. A
    /// three-segment spine given `+4°` therefore leans `12°`, which is what an
    /// authored spine curve is.
    All,
}

/// One authored channel of a [`PostureClip`].
#[derive(Debug, Clone, Copy)]
pub struct PostureTrack {
    /// What the joint is.
    pub kind: BoneRoleKind,
    /// Which side, or [`BoneSide::Center`] for a spine.
    pub side: BoneSide,
    /// Which joints of that chain.
    pub pick: Pick,
    /// `(time in seconds, euler XYZ in DEGREES)`, ascending, the last equal to
    /// the first for a loop.
    pub keys: &'static [(f32, [f32; 3])],
}

/// A translation channel — the only one any posture authors, and only on the
/// hips.
#[derive(Debug, Clone, Copy)]
pub struct PostureShift {
    /// `(time in seconds, metres XYZ in the parent's frame)`.
    pub keys: &'static [(f32, [f32; 3])],
}

/// **An authored posture, as clip data.**
#[derive(Debug, Clone, Copy)]
pub struct PostureClip {
    /// What it is called, for a trace.
    pub name: &'static str,
    /// The loop length, seconds. A static pose still has one — see [`SIT`].
    pub duration_s: f32,
    /// The rotation channels.
    pub tracks: &'static [PostureTrack],
    /// The hips' own translation, or `None`.
    pub shift: Option<PostureShift>,
}

/// **SEATED** — `venues/0012`'s patrons in their armchairs and `/0028`'s row
/// along the catwalk edge.
///
/// The geometry is one sentence: **the thigh goes forward and the shin goes
/// down**. `−85°` about X swings a bone pointing straight down to point
/// straight forward (`R_x(−90)·(0,−1,0) = (0,0,1)`), and `+85°` on the calf
/// puts the shin back under the knee — which is why the foot needs no track of
/// its own, its frame having been turned and turned back.
///
/// The rest is what stops it reading as a mannequin folded at the hips: a small
/// recline on the pelvis, a counter-curve up the spine so the chest stays over
/// the hips, and the arms brought forward onto the knees.
///
/// **It breathes.** Four seconds, ±1.5° on the spine: a seated crowd authored
/// as one static frame is a row of statues, and this is the cheapest thing that
/// is not. Every other channel is one key and is therefore constant.
pub static SIT: PostureClip = PostureClip {
    name: "sit",
    duration_s: 4.0,
    tracks: &[
        PostureTrack {
            kind: BoneRoleKind::Pelvis,
            side: BoneSide::Center,
            pick: Pick::First,
            keys: &[(0.0, [-10.0, 0.0, 0.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::Spine,
            side: BoneSide::Center,
            pick: Pick::All,
            keys: &[
                (0.0, [4.0, 0.0, 0.0]),
                (2.0, [5.5, 0.0, 0.0]),
                (4.0, [4.0, 0.0, 0.0]),
            ],
        },
        PostureTrack {
            kind: BoneRoleKind::Thigh,
            side: BoneSide::Left,
            pick: Pick::All,
            keys: &[(0.0, [-85.0, 0.0, 3.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::Thigh,
            side: BoneSide::Right,
            pick: Pick::All,
            keys: &[(0.0, [-85.0, 0.0, -3.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::Calf,
            side: BoneSide::Left,
            pick: Pick::All,
            keys: &[(0.0, [85.0, 0.0, 0.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::Calf,
            side: BoneSide::Right,
            pick: Pick::All,
            keys: &[(0.0, [85.0, 0.0, 0.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::UpperArm,
            side: BoneSide::Left,
            pick: Pick::First,
            keys: &[(0.0, [-18.0, 0.0, 6.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::UpperArm,
            side: BoneSide::Right,
            pick: Pick::First,
            keys: &[(0.0, [-18.0, 0.0, -6.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::LowerArm,
            side: BoneSide::Left,
            pick: Pick::First,
            keys: &[(0.0, [-35.0, 0.0, 0.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::LowerArm,
            side: BoneSide::Right,
            pick: Pick::First,
            keys: &[(0.0, [-35.0, 0.0, 0.0])],
        },
    ],
    // The hips' drop is DERIVED from the rig — see `sit_drop_m` — so the
    // authored shift carries only the small slide back into the seat.
    shift: Some(PostureShift {
        keys: &[(0.0, [0.0, 0.0, -0.06])],
    }),
};

/// **DANCING** — `venues/0028` and `/0044`, on the floor and on the deck.
///
/// A 1.2-second loop, which is fifty beats to the bar at a hundred bpm: two
/// beats, one weight shift each way. Authored as the three things a body
/// actually does on a dance floor, in the order a viewer reads them:
///
/// * **the weight shift** — the hips slide 5 cm off-axis and drop 3 cm at each
///   extreme, because a dance that only rotates reads as a mannequin on a
///   turntable;
/// * **the hip sway and the counter-twist** — ±6° of roll on the pelvis
///   against ∓3° a segment up the spine, so the shoulders stay level while the
///   hips move, which is the whole silhouette;
/// * **the arms** — up and alternating, ±70°/±20° of shoulder roll a beat
///   apart, with the elbows bent so they are arms rather than semaphore.
///
/// The last key equals the first at every channel, so the loop closes without a
/// seam.
pub static DANCE: PostureClip = PostureClip {
    name: "dance",
    duration_s: 1.2,
    tracks: &[
        PostureTrack {
            kind: BoneRoleKind::Pelvis,
            side: BoneSide::Center,
            pick: Pick::First,
            keys: &[
                (0.0, [0.0, 0.0, 6.0]),
                (0.3, [0.0, 5.0, 0.0]),
                (0.6, [0.0, 0.0, -6.0]),
                (0.9, [0.0, -5.0, 0.0]),
                (1.2, [0.0, 0.0, 6.0]),
            ],
        },
        PostureTrack {
            kind: BoneRoleKind::Spine,
            side: BoneSide::Center,
            pick: Pick::All,
            keys: &[
                (0.0, [0.0, 0.0, -3.0]),
                (0.3, [0.0, -3.0, 0.0]),
                (0.6, [0.0, 0.0, 3.0]),
                (0.9, [0.0, 3.0, 0.0]),
                (1.2, [0.0, 0.0, -3.0]),
            ],
        },
        PostureTrack {
            kind: BoneRoleKind::Head,
            side: BoneSide::Center,
            pick: Pick::First,
            keys: &[
                (0.0, [5.0, 0.0, 0.0]),
                (0.6, [-5.0, 0.0, 0.0]),
                (1.2, [5.0, 0.0, 0.0]),
            ],
        },
        PostureTrack {
            kind: BoneRoleKind::UpperArm,
            side: BoneSide::Left,
            pick: Pick::First,
            keys: &[
                (0.0, [0.0, 0.0, -70.0]),
                (0.3, [0.0, 0.0, -40.0]),
                (0.6, [0.0, 0.0, -20.0]),
                (0.9, [0.0, 0.0, -40.0]),
                (1.2, [0.0, 0.0, -70.0]),
            ],
        },
        PostureTrack {
            kind: BoneRoleKind::UpperArm,
            side: BoneSide::Right,
            pick: Pick::First,
            keys: &[
                (0.0, [0.0, 0.0, 20.0]),
                (0.3, [0.0, 0.0, 40.0]),
                (0.6, [0.0, 0.0, 70.0]),
                (0.9, [0.0, 0.0, 40.0]),
                (1.2, [0.0, 0.0, 20.0]),
            ],
        },
        PostureTrack {
            kind: BoneRoleKind::LowerArm,
            side: BoneSide::Left,
            pick: Pick::First,
            keys: &[
                (0.0, [-60.0, 0.0, 0.0]),
                (0.6, [-80.0, 0.0, 0.0]),
                (1.2, [-60.0, 0.0, 0.0]),
            ],
        },
        PostureTrack {
            kind: BoneRoleKind::LowerArm,
            side: BoneSide::Right,
            pick: Pick::First,
            keys: &[
                (0.0, [-80.0, 0.0, 0.0]),
                (0.6, [-60.0, 0.0, 0.0]),
                (1.2, [-80.0, 0.0, 0.0]),
            ],
        },
        PostureTrack {
            kind: BoneRoleKind::Thigh,
            side: BoneSide::Left,
            pick: Pick::First,
            keys: &[
                (0.0, [0.0, 0.0, 5.0]),
                (0.6, [0.0, 0.0, -3.0]),
                (1.2, [0.0, 0.0, 5.0]),
            ],
        },
        PostureTrack {
            kind: BoneRoleKind::Thigh,
            side: BoneSide::Right,
            pick: Pick::First,
            keys: &[
                (0.0, [0.0, 0.0, 3.0]),
                (0.6, [0.0, 0.0, -5.0]),
                (1.2, [0.0, 0.0, 3.0]),
            ],
        },
    ],
    shift: Some(PostureShift {
        keys: &[
            (0.0, [0.05, -0.03, 0.0]),
            (0.3, [0.0, 0.0, 0.0]),
            (0.6, [-0.05, -0.03, 0.0]),
            (0.9, [0.0, 0.0, 0.0]),
            (1.2, [0.05, -0.03, 0.0]),
        ],
    }),
};

/// **DOWN ON ONE KNEE** (wave EMS2) — a paramedic at a patient, a firefighter at
/// a hydrant, an officer over evidence.
///
/// The geometry is one sentence and it is the SIT's sentence applied to one leg
/// each way: the **left** leg folds under (a small thigh rotation and a big
/// calf one, so the shin lies along the ground behind the knee), the **right**
/// leg takes the sit's own pose (thigh forward, shin down, foot flat), and the
/// hips drop between the two — see [`KNEEL_DROP_FRAC`] for why that is a
/// fraction of the femur rather than the whole of it.
///
/// The rest is what stops it reading as a mannequin folded in half: a small
/// forward lean on the spine, because somebody working on the ground is looking
/// at it, and the arms brought forward and down onto whatever they are doing.
///
/// **It breathes**, on [`SIT`]'s terms and for its reason: four seconds, ±1.5°
/// on the spine, so a crew standing at a scene for ten seconds is not a
/// photograph. Every other channel is one key and is therefore constant.
pub static KNEEL: PostureClip = PostureClip {
    name: "kneel",
    duration_s: 4.0,
    tracks: &[
        PostureTrack {
            kind: BoneRoleKind::Pelvis,
            side: BoneSide::Center,
            pick: Pick::First,
            keys: &[(0.0, [-4.0, 0.0, 0.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::Spine,
            side: BoneSide::Center,
            pick: Pick::All,
            keys: &[
                (0.0, [6.0, 0.0, 0.0]),
                (2.0, [7.5, 0.0, 0.0]),
                (4.0, [6.0, 0.0, 0.0]),
            ],
        },
        // The kneeling leg: the thigh stays nearly vertical and the shin folds
        // back along the ground.
        PostureTrack {
            kind: BoneRoleKind::Thigh,
            side: BoneSide::Left,
            pick: Pick::All,
            keys: &[(0.0, [-12.0, 0.0, 6.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::Calf,
            side: BoneSide::Left,
            pick: Pick::All,
            keys: &[(0.0, [125.0, 0.0, 0.0])],
        },
        // The standing leg: the sit's own thigh and shin, so the foot is flat
        // and the knee is up.
        PostureTrack {
            kind: BoneRoleKind::Thigh,
            side: BoneSide::Right,
            pick: Pick::All,
            keys: &[(0.0, [-85.0, 0.0, -6.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::Calf,
            side: BoneSide::Right,
            pick: Pick::All,
            keys: &[(0.0, [85.0, 0.0, 0.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::UpperArm,
            side: BoneSide::Left,
            pick: Pick::First,
            keys: &[(0.0, [-38.0, 0.0, 8.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::UpperArm,
            side: BoneSide::Right,
            pick: Pick::First,
            keys: &[(0.0, [-38.0, 0.0, -8.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::LowerArm,
            side: BoneSide::Left,
            pick: Pick::First,
            keys: &[(0.0, [-50.0, 0.0, 0.0])],
        },
        PostureTrack {
            kind: BoneRoleKind::LowerArm,
            side: BoneSide::Right,
            pick: Pick::First,
            keys: &[(0.0, [-50.0, 0.0, 0.0])],
        },
    ],
    // The drop is DERIVED like the sit's — see `KNEEL_DROP_FRAC` — so the
    // authored shift carries only the small slide back over the trailing heel.
    shift: Some(PostureShift {
        keys: &[(0.0, [0.0, 0.0, -0.04])],
    }),
};

/// **How much of a femur a kneeling body's hips drop**, as a fraction.
///
/// A seated body's hips fall a whole femur: the thigh goes horizontal and the
/// shin takes over as the vertical member. A body on one knee has **one** shin
/// on the ground and one thigh still carrying it, so its hips sit between the
/// two — a little over half a femur, measured from the same bone rather than
/// from a metre constant that would be wrong for every character but one.
///
/// 0.55, and the arm that says it is right is
/// `a_kneeling_body_has_one_shin_down_and_one_knee_up`, which measures the pose
/// in model space instead of asserting the table back.
pub const KNEEL_DROP_FRAC: f32 = 0.55;

/// What one [`apply_posture`] did — the falsifier for a pass that ran and wrote
/// nothing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PostureReport {
    /// Joints whose rotation this pass changed.
    pub rotated: usize,
    /// Whether the hips were moved.
    pub shifted: bool,
    /// **Tracks that named a role the rig does not have.** Non-zero is not a
    /// failure — a four-legged creature has no `UpperArm` — but it is the
    /// number that tells "the rig has no role table" from "the pose ran".
    pub unmatched: usize,
}

impl PostureReport {
    /// Whether anything moved at all.
    pub fn wrote(&self) -> bool {
        self.rotated > 0 || self.shifted
    }
}

/// **How far a sitting body's hips drop, metres** — the rig's own femur.
///
/// A standing pelvis is `thigh + shin` above the floor and a seated one is
/// `shin` above the seat, so the difference is the thigh. Measured off the bind
/// pose (the calf joint's offset from the thigh joint IS the femur), so a
/// child, a giant and the 1.8 m mannequin each sit at their own height without
/// a table.
///
/// `0.0` for a rig with no leg roles, which drops nothing rather than guessing.
pub fn sit_drop_m(rig: &SkeletonAsset) -> f32 {
    let roles = RoleIndex::new(&rig.roles);
    for side in [BoneSide::Left, BoneSide::Right] {
        let Some(calf) = roles.first(BoneRoleKind::Calf, side) else {
            continue;
        };
        let Some(j) = rig.skeleton.joints().get(calf as usize) else {
            continue;
        };
        let d = Vec3::from(j.local_bind.translation).length();
        if d.is_finite() && d > 0.0 {
            return d;
        }
    }
    0.0
}

/// Sample a keyed channel at `t`, linearly, holding the ends.
///
/// One key answers itself at every time, which is what makes a static posture a
/// clip with no special case.
fn sample(keys: &[(f32, [f32; 3])], t: f32) -> Vec3 {
    if keys.is_empty() {
        return Vec3::ZERO;
    }
    if t <= keys[0].0 {
        return Vec3::from(keys[0].1);
    }
    for w in keys.windows(2) {
        let (t0, a) = w[0];
        let (t1, b) = w[1];
        if t <= t1 {
            let span = t1 - t0;
            let u = if span > 0.0 { (t - t0) / span } else { 0.0 };
            return Vec3::from(a).lerp(Vec3::from(b), u.clamp(0.0, 1.0));
        }
    }
    Vec3::from(keys[keys.len() - 1].1)
}

/// A quaternion from euler XYZ **degrees**, through the portable trig.
///
/// `glam::Quat::from_euler` is `sin_cos` inside, and this rotation lands on a
/// joint that lands in `pose_state_bytes` — the P14 law's own subject. The
/// order is X then Y then Z applied in the parent's frame, which is
/// `q = qz * qy * qx`.
///
/// **`psin64` and not `psin`**, and the difference is visible rather than
/// pedantic: the f32 pair is documented "demo-grade, ~5e-3 accuracy", which at
/// the sit's own 42.5-degree half-angle puts the thigh a quarter of a degree
/// off and failed `the_portable_euler_is_the_rotation_it_claims` at any
/// tolerance worth writing. The f64 pair is ~1e-7 and is what that module's own
/// doc says to use "for accurate committed geometry"; the cast to f32 happens
/// after the arithmetic, so what reaches the joint is a correctly-rounded f32.
fn euler_deg(e: Vec3) -> Quat {
    let axis = |a: Vec3, deg: f32| -> Quat {
        let h = f64::from(deg).to_radians() * 0.5;
        let (s, c) = (inf_math::psin64(h) as f32, inf_math::pcos64(h) as f32);
        Quat::from_xyzw(a.x * s, a.y * s, a.z * s, c)
    };
    axis(Vec3::Z, e.z) * axis(Vec3::Y, e.y) * axis(Vec3::X, e.x)
}

/// **Apply a posture to a pose** (wave VEN1b) — the one door.
///
/// `t_s` is the clip's own clock, already phased per agent by the caller;
/// it is wrapped into `[0, duration)` so a posture always loops.
///
/// Every track is **composed onto** what the state machine put there rather
/// than replacing it: `delta * local`, a rotation in the joint's parent frame.
/// A machine that is playing `idle` therefore keeps its idle's breathing under
/// the sit, and a rig whose role table is empty is left exactly as it was.
pub fn apply_posture(
    rig: &SkeletonAsset,
    pose: &mut Pose,
    posture: Posture,
    t_s: f32,
) -> PostureReport {
    let clip = posture.clip();
    let mut report = PostureReport::default();
    let roles = RoleIndex::new(&rig.roles);
    if roles.is_empty() || pose.is_empty() {
        // **The early return says WHY it returned.** "Absent costs nothing" is
        // right — a rig with no role table is left exactly as it was — but a
        // report of all zeroes cannot be told from one where every track
        // matched and moved nothing, and that is the difference a caller asking
        // "did this rig understand the clip" needs. Every track went unmatched,
        // because there was nothing to match them against.
        report.unmatched = clip.tracks.len();
        return report;
    }
    let t = if clip.duration_s > 0.0 && t_s.is_finite() {
        t_s.rem_euclid(clip.duration_s)
    } else {
        0.0
    };
    for track in clip.tracks {
        let all = roles.all(track.kind, track.side);
        if all.is_empty() {
            report.unmatched += 1;
            continue;
        }
        let picked: &[u16] = match track.pick {
            Pick::First => &all[..1],
            Pick::Last => &all[all.len() - 1..],
            Pick::All => &all[..],
        };
        let delta = euler_deg(sample(track.keys, t));
        for j in picked {
            let Some(local) = pose.locals.get_mut(*j as usize) else {
                continue;
            };
            let base = Quat::from_array(local.rotation);
            local.rotation = (delta * base).normalize().to_array();
            report.rotated += 1;
        }
    }
    if let Some(shift) = clip.shift {
        // The hips, and only the hips: a posture that translated a knee would
        // pull the leg out of its own socket, which is the reason `pelvis_drop`
        // is spelled the same way one pass down.
        if let Some(p) = roles
            .first(BoneRoleKind::Pelvis, BoneSide::Center)
            .or_else(|| roles.first_any(BoneRoleKind::Pelvis))
        {
            if let Some(local) = pose.locals.get_mut(p as usize) {
                let mut d = sample(shift.keys, t);
                match posture {
                    Posture::Sit => d.y -= sit_drop_m(rig),
                    // One shin down and one thigh still carrying — see
                    // `KNEEL_DROP_FRAC`.
                    Posture::Kneel => d.y -= sit_drop_m(rig) * KNEEL_DROP_FRAC,
                    Posture::Dance => {}
                }
                local.translation[0] += d.x;
                local.translation[1] += d.y;
                local.translation[2] += d.z;
                report.shifted = true;
            }
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manny() -> SkeletonAsset {
        crate::manny::build_manny(&crate::template::BodyParams::default())
            .expect("the mannequin builds")
    }

    /// Where a joint's HEAD is in model space, under `pose`.
    fn model_of(rig: &SkeletonAsset, pose: &Pose, joint: u16) -> Vec3 {
        let g = crate::pose::global_transforms(&rig.skeleton, pose);
        g[joint as usize].w_axis.truncate()
    }

    fn joint(rig: &SkeletonAsset, kind: BoneRoleKind, side: BoneSide) -> u16 {
        RoleIndex::new(&rig.roles)
            .first(kind, side)
            .unwrap_or_else(|| panic!("the rig has no {kind:?} {side:?}"))
    }

    /// **Every rig this engine generates binds with an IDENTITY rotation**, so
    /// an authored delta in a parent's frame is a delta in the model's — the
    /// assumption the whole authored table rests on, stated as an arm rather
    /// than as a paragraph.
    #[test]
    fn a_generated_rig_binds_without_rotation() {
        let rig = manny();
        for j in rig.skeleton.joints() {
            assert_eq!(
                j.local_bind.rotation,
                [0.0, 0.0, 0.0, 1.0],
                "{} binds rotated, so an authored euler means something else \
                 in its parent's frame",
                j.name
            );
        }
    }

    /// **THE SIT READS AS SITTING**, measured in model space rather than
    /// asserted from the table:
    ///
    /// * the knee ends up **in front of** the hip (it was under it);
    /// * the ankle stays under the knee, so the shin is vertical;
    /// * and the hips come **down** by about a femur.
    ///
    /// Every one of the three is a claim a wrong-signed euler fails.
    #[test]
    fn a_seated_body_has_its_knees_in_front_and_its_hips_down() {
        let rig = manny();
        let stand = Pose::rest(&rig.skeleton);
        let mut sit = stand.clone();
        let report = apply_posture(&rig, &mut sit, Posture::Sit, 0.0);
        assert!(report.wrote(), "the sit pass wrote nothing: {report:?}");
        assert_eq!(
            report.unmatched, 0,
            "the mannequin is missing a role the sit names: {report:?}"
        );

        let (hip, knee, ankle) = (
            joint(&rig, BoneRoleKind::Thigh, BoneSide::Left),
            joint(&rig, BoneRoleKind::Calf, BoneSide::Left),
            joint(&rig, BoneRoleKind::Foot, BoneSide::Left),
        );
        let (h0, k0, a0) = (
            model_of(&rig, &stand, hip),
            model_of(&rig, &stand, knee),
            model_of(&rig, &stand, ankle),
        );
        let (h1, k1, a1) = (
            model_of(&rig, &sit, hip),
            model_of(&rig, &sit, knee),
            model_of(&rig, &sit, ankle),
        );
        let femur = (k0 - h0).length();
        println!(
            "VEN1b sit: femur {femur:.3} m; hip {:.3} -> {:.3}; knee dz {:.3} \
             -> {:.3}; shin |dz| {:.3} -> {:.3}",
            h0.y,
            h1.y,
            k0.z - h0.z,
            k1.z - h1.z,
            (a0.z - k0.z).abs(),
            (a1.z - k1.z).abs()
        );
        assert!(femur > 0.1, "the fixture rig has no femur ({femur})");
        // Standing: the knee is under the hip.
        assert!((k0.z - h0.z).abs() < femur * 0.3);
        // Seated: it is in FRONT of it, by most of a femur.
        assert!(
            k1.z - h1.z > femur * 0.7,
            "the knee is {:.3} m in front of the hip against a {femur:.3} m \
             femur — the thigh did not go forward",
            k1.z - h1.z
        );
        // …and the shin is vertical: the ankle is under the knee.
        assert!(
            (a1.z - k1.z).abs() < femur * 0.3,
            "the ankle is {:.3} m off the knee in plan — the shin is not down",
            (a1.z - k1.z).abs()
        );
        // …and the hips came down by about a femur.
        let drop = h0.y - h1.y;
        assert!(
            drop > femur * 0.7 && drop < femur * 1.4,
            "the hips dropped {drop:.3} m against a {femur:.3} m femur"
        );
        assert!(
            (sit_drop_m(&rig) - femur).abs() < 1e-3,
            "the derived drop {} is not the rig's own femur {femur}",
            sit_drop_m(&rig)
        );
    }

    /// **THE DANCE MOVES, AND LOOPS.** Three claims a static overlay fails:
    /// the hips are in different places at different times, the arms swap which
    /// one is up, and the loop closes seamlessly at its own duration.
    #[test]
    fn a_dancing_body_shifts_its_weight_and_loops() {
        let rig = manny();
        let at = |t: f32| -> Pose {
            let mut p = Pose::rest(&rig.skeleton);
            apply_posture(&rig, &mut p, Posture::Dance, t);
            p
        };
        let (pelvis, la, ra) = (
            joint(&rig, BoneRoleKind::Pelvis, BoneSide::Center),
            joint(&rig, BoneRoleKind::UpperArm, BoneSide::Left),
            joint(&rig, BoneRoleKind::UpperArm, BoneSide::Right),
        );
        let hand = |p: &Pose, upper: u16| -> Vec3 {
            // The upper arm's own tip: its child's model position.
            let child = RoleIndex::new(&rig.roles)
                .deform_child(&rig.skeleton, upper)
                .expect("an upper arm has a forearm");
            model_of(&rig, p, child)
        };
        let (a, b) = (at(0.0), at(0.6));
        let ha = model_of(&rig, &a, pelvis);
        let hb = model_of(&rig, &b, pelvis);
        println!(
            "VEN1b dance: hips {:.3} -> {:.3} ({:.3} m apart); left elbow y {:.3} \
             -> {:.3}; right elbow y {:.3} -> {:.3}",
            ha.x,
            hb.x,
            (ha - hb).length(),
            hand(&a, la).y,
            hand(&b, la).y,
            hand(&a, ra).y,
            hand(&b, ra).y
        );
        assert!(
            (ha - hb).length() > 0.05,
            "the hips moved {:.4} m over half a loop — that is not a weight \
             shift",
            (ha - hb).length()
        );
        // The arms alternate: whichever is higher at t=0 is lower at t=0.6.
        let (l0, r0) = (hand(&a, la).y, hand(&a, ra).y);
        let (l1, r1) = (hand(&b, la).y, hand(&b, ra).y);
        assert!(
            (l0 > r0) != (l1 > r1),
            "the same arm is up at both ends of the loop: {l0:.3}/{r0:.3} then \
             {l1:.3}/{r1:.3}"
        );
        // The loop closes: the pose at the duration is the pose at zero.
        //
        // Compared with a tolerance and not with `==`: `rem_euclid(1.2)` of 1.2
        // is a float zero reached by a different route from the one a caller
        // passes in, so the sample lands on the same key with a different last
        // bit (-0.04999998 against -0.05, measured). A seam a viewer could see
        // is millimetres, not ulps.
        let close = |x: &crate::JointTransform, y: &crate::JointTransform| -> bool {
            Vec3::from(x.translation).distance(Vec3::from(y.translation)) < 1e-5
                && Quat::from_array(x.rotation)
                    .dot(Quat::from_array(y.rotation))
                    .abs()
                    > 1.0 - 1e-5
        };
        let end = at(DANCE.duration_s);
        for (i, (x, y)) in a.locals.iter().zip(end.locals.iter()).enumerate() {
            assert!(
                close(x, y),
                "joint {i} does not come back to where the loop started"
            );
        }
        // …and it wraps rather than clamping.
        let wrapped = at(DANCE.duration_s * 3.0 + 0.6);
        assert!(close(
            &wrapped.locals[pelvis as usize],
            &b.locals[pelvis as usize]
        ));
    }

    /// **THE KNEEL READS AS KNEELING** (wave EMS2), measured in model space
    /// rather than asserted from the table:
    ///
    /// * the **left** shin lies along the ground — its ankle is well BEHIND its
    ///   knee, which is what folding a leg under you means;
    /// * the **right** shin is vertical — its ankle is under its knee, and its
    ///   knee is in front of its hip, exactly as the sit's is;
    /// * the hips come down by about [`KNEEL_DROP_FRAC`] of a femur, which is
    ///   between standing and sitting rather than either;
    /// * and the two legs are doing **different** things, which is the whole
    ///   difference between kneeling and squatting.
    ///
    /// Every one of the four is a claim a wrong-signed euler fails.
    #[test]
    fn a_kneeling_body_has_one_shin_down_and_one_knee_up() {
        let rig = manny();
        let stand = Pose::rest(&rig.skeleton);
        let mut kneel = stand.clone();
        let report = apply_posture(&rig, &mut kneel, Posture::Kneel, 0.0);
        assert!(report.wrote(), "the kneel pass wrote nothing: {report:?}");
        assert_eq!(
            report.unmatched, 0,
            "the mannequin is missing a role the kneel names: {report:?}"
        );

        let j = |kind, side| joint(&rig, kind, side);
        let (lh, lk, la) = (
            j(BoneRoleKind::Thigh, BoneSide::Left),
            j(BoneRoleKind::Calf, BoneSide::Left),
            j(BoneRoleKind::Foot, BoneSide::Left),
        );
        let (rh, rk, ra) = (
            j(BoneRoleKind::Thigh, BoneSide::Right),
            j(BoneRoleKind::Calf, BoneSide::Right),
            j(BoneRoleKind::Foot, BoneSide::Right),
        );
        let femur = (model_of(&rig, &stand, lk) - model_of(&rig, &stand, lh)).length();
        assert!(femur > 0.1, "the fixture rig has no femur ({femur})");

        let (h0, h1) = (model_of(&rig, &stand, lh).y, model_of(&rig, &kneel, lh).y);
        let lk1 = model_of(&rig, &kneel, lk);
        let la1 = model_of(&rig, &kneel, la);
        let rk1 = model_of(&rig, &kneel, rk);
        let ra1 = model_of(&rig, &kneel, ra);
        let rh1 = model_of(&rig, &kneel, rh);
        println!(
            "EMS2 kneel: femur {femur:.3} m; hips {h0:.3} -> {h1:.3}; left \
             ankle-behind-knee {:.3}; right ankle-under-knee {:.3}; right \
             knee-ahead-of-hip {:.3}",
            lk1.z - la1.z,
            (ra1.z - rk1.z).abs(),
            rk1.z - rh1.z
        );
        // The kneeling shin lies BACK along the ground.
        assert!(
            lk1.z - la1.z > femur * 0.5,
            "the left ankle is {:.3} m behind its knee against a {femur:.3} m \
             femur — that leg is not folded under",
            lk1.z - la1.z
        );
        // The standing shin is vertical, and its knee is forward.
        assert!(
            (ra1.z - rk1.z).abs() < femur * 0.35,
            "the right ankle is {:.3} m off its knee in plan — that shin is not \
             down",
            (ra1.z - rk1.z).abs()
        );
        assert!(
            rk1.z - rh1.z > femur * 0.6,
            "the right knee is {:.3} m in front of the hip — the standing leg \
             is not forward",
            rk1.z - rh1.z
        );
        // The hips are between standing and sitting.
        let drop = h0 - h1;
        assert!(
            drop > femur * (KNEEL_DROP_FRAC - 0.2) && drop < femur * (KNEEL_DROP_FRAC + 0.25),
            "the hips dropped {drop:.3} m against {KNEEL_DROP_FRAC} of a \
             {femur:.3} m femur"
        );
        // …and the two legs really differ, which a squat would fail.
        let asymmetry = ((lk1.z - la1.z) - (rk1.z - ra1.z)).abs();
        assert!(
            asymmetry > femur * 0.5,
            "the two shins are doing the same thing ({asymmetry:.3} m apart) — \
             that is a squat, not a kneel"
        );
    }

    /// A rig with no role table poses exactly what it posed before, which is
    /// what makes "absent costs nothing" structural.
    #[test]
    fn a_rig_with_no_roles_is_left_alone() {
        let mut rig = manny();
        rig.roles.clear();
        let before = Pose::rest(&rig.skeleton);
        let mut after = before.clone();
        let r = apply_posture(&rig, &mut after, Posture::Sit, 0.0);
        assert_eq!(before, after);
        assert!(!r.wrote());
        assert_eq!(r.rotated, 0);
        // …and the counter says WHY it wrote nothing, which is the difference
        // between "this rig has no arms" and "the pass never ran". A counter
        // nothing reads is the shape this tree has a law about, so the arm that
        // proves the refusal is the arm that reads it.
        assert_eq!(
            r.unmatched,
            SIT.tracks.len(),
            "the sit named {} roles and {} went unmatched on a rig with none",
            SIT.tracks.len(),
            r.unmatched
        );
    }

    /// The euler helper is the portable trig and agrees with glam's own
    /// conversion to within a rounding — the arm that says P14's spelling did
    /// not silently change the pose.
    #[test]
    fn the_portable_euler_is_the_rotation_it_claims() {
        for e in [
            Vec3::new(-85.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 70.0),
            Vec3::new(4.0, -3.0, 6.0),
        ] {
            let ours = euler_deg(e);
            let glams = Quat::from_rotation_z(e.z.to_radians())
                * Quat::from_rotation_y(e.y.to_radians())
                * Quat::from_rotation_x(e.x.to_radians());
            assert!(
                ours.dot(glams).abs() > 1.0 - 1e-6,
                "{e:?}: {ours:?} against {glams:?}"
            );
        }
    }
}
