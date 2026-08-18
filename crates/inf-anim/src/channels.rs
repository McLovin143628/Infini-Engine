//! Clip channels — the `.inf_anim` **v2** payload beyond the joint tracks
//! (P29.2, §13's 2026-08-17 amendment, Ruling 2).
//!
//! v1 of `.inf_anim` was `{name, duration, tracks}`: keyframed TRS per joint and
//! nothing else. Everything the later sub-phases have to read off a clip —
//! whether the foot is planted, how far the root has travelled, which frame the
//! footstep sound fires on, what an additive pose is additive *over* — was
//! therefore either derived on the fly or not expressible at all.
//!
//! The amendment's ruling is that the format bumps **exactly once**, so v2 has to
//! be wide enough for P29.3–P29.6 on the day it lands. Four shapes, and the last
//! two are reserved capacity rather than a feature this wave uses:
//!
//! 1. [`CurveChannel`] — named `f32` curves on the clip timeline. The names are
//!    free-form `String`s and **not** an enum, deliberately: the 25 verbatim ALS
//!    curve names (`Enable_FootIK_L`, `FootLock_R`, `RotationAmount`, `W_Gait`,
//!    `YawOffset`, `Mask_AimOffset`, the ten `Layering_*`, …) are the *reference
//!    vocabulary* that import derivation (P29.5) and the bridge (P29.4) target,
//!    and a studio that wants one of its own must not need an engine schema bump
//!    to have it. [`als`] spells the reference vocabulary out as constants so the
//!    names are checkable rather than retyped.
//! 2. [`AnimMarker`] — `{time_s, name, group}`. One shape serves both halves of
//!    what §13 asked for: a *sync* group (foot phase, so a walk↔run blend keeps
//!    its feet — see [`crate::sync`]) and an *event* marker (a footstep notify,
//!    P29.4's consumer).
//! 3. [`AdditiveRef`] — what a clip is additive **over**.
//! 4. [`RootMotionTrack`] and [`DistanceTrack`] — present and optional day one,
//!    produced and consumed in P29.4/P29.5. The root-motion track carries
//!    translation **including Y**, which is the port map's finding about our
//!    current extraction: [`crate::root_motion::root_delta`] drops the vertical
//!    component by design (v1 left jumps to the character controller), and that
//!    is simply wrong for a mantle, where the clip's own rise *is* the movement.
//!
//! # Portability
//!
//! Nothing in this file calls a transcendental. Sampling is `locate` + a lerp,
//! and the file is on `tests/portable_pose.rs`'s list for the usual reason: a
//! curve value read here reaches a blend weight, and a blend weight reaches the
//! pose that is folded into `state_bytes`.

use serde::{Deserialize, Serialize};

use crate::clip::{locate, Interpolation};

/// A named scalar curve sampled on the clip's own timeline.
///
/// `times` is the key list (the same convention as [`crate::Vec3Track`]:
/// increasing, clamped at both ends, and total over an unsorted list because a
/// committed clip is entitled to whatever a pre-import-door build wrote);
/// `values` is index-aligned with it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurveChannel {
    /// The curve's name. Free-form — see the module docs on why this is not an
    /// enum.
    pub name: String,
    pub times: Vec<f32>,
    pub values: Vec<f32>,
    pub interp: Interpolation,
}

impl CurveChannel {
    /// Build a channel; `times` must match `values` in length (checked at the
    /// decode door by [`crate::AnimClip::validate_timing`], `debug_assert`ed
    /// here, exactly like the joint tracks).
    pub fn new(
        name: impl Into<String>,
        times: Vec<f32>,
        values: Vec<f32>,
        interp: Interpolation,
    ) -> Self {
        debug_assert_eq!(times.len(), values.len(), "times/values length mismatch");
        Self {
            name: name.into(),
            times,
            values,
            interp,
        }
    }

    /// A curve that holds one value for the whole clip — the shape most of the
    /// `Layering_*` and `Enable_*` vocabulary actually has on an overlay pose.
    pub fn constant(name: impl Into<String>, value: f32) -> Self {
        Self::new(name, vec![0.0], vec![value], Interpolation::Step)
    }

    /// Sample the curve at `t` seconds (clamped to the key range). `None` only
    /// when the channel has no keys.
    pub fn sample(&self, t: f32) -> Option<f32> {
        let (i0, i1, frac) = locate(&self.times, t)?;
        let a = *self.values.get(i0)?;
        if i0 == i1 || matches!(self.interp, Interpolation::Step) {
            return Some(a);
        }
        let b = *self.values.get(i1)?;
        Some(a + (b - a) * frac)
    }
}

/// A timed marker on a clip: a sync point, an event notify, or both.
///
/// `group` is what makes one shape serve both jobs. A marker whose group is
/// non-empty participates in that **sync group** ([`crate::sync`]); a marker with
/// an empty group is a plain event notify, fired by name when the play-head
/// crosses it (P29.4's consumer). A marker may be both: a `foot` group marker
/// named `footstep_l` both aligns the blend and rings the sound.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimMarker {
    /// Position on the clip timeline, seconds.
    pub time_s: f32,
    /// The marker's name — the thing a sync group matches on and a notify fires.
    pub name: String,
    /// The sync group this marker belongs to, or empty for "event only".
    pub group: String,
}

impl AnimMarker {
    /// A sync-group marker.
    pub fn sync(time_s: f32, name: impl Into<String>, group: impl Into<String>) -> Self {
        Self {
            time_s,
            name: name.into(),
            group: group.into(),
        }
    }

    /// An event-only marker (no sync group).
    pub fn event(time_s: f32, name: impl Into<String>) -> Self {
        Self {
            time_s,
            name: name.into(),
            group: String::new(),
        }
    }

    /// Whether this marker takes part in sync-group alignment.
    pub fn is_sync(&self) -> bool {
        !self.group.is_empty()
    }
}

/// What a clip is **additive over**.
///
/// # Wire discriminants are frozen, and two slots are reserved
///
/// bincode writes a variant as its index, so this list is append-only like every
/// other wire enum in the engine (`freeze_pins` in this module's tests). The two
/// `Reserved*` slots are not decoration: `.inf_anim` bumps **once**, in P29.2, so
/// a future additive kind cannot arrive behind a schema bump. Declaring the slots
/// now means a newer build's file **decodes** on this one and is refused *by
/// name* with a remedy, instead of dying as `Decode("unexpected variant")` — and
/// each reserved slot carries `BaseClip`'s payload shape so a future kind has
/// room for a clip reference and a time without desyncing the positional decode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum AdditiveRef {
    /// Not additive: the clip is a full pose.
    #[default]
    None,
    /// Additive over the clip's **own first frame** — the common authoring case
    /// (an aim offset, a lean, a breath), and the one that needs no second asset.
    FirstFrame,
    /// Additive over another clip, sampled at `time_s` seconds.
    BaseClip {
        /// Raw 16-byte GUID of the base clip (this crate stays `uuid`-free; the
        /// editor records the dependency edge).
        clip: [u8; 16],
        time_s: f32,
    },
    /// Reserved. See the type's docs — a build that meets one refuses by name.
    Reserved3 { clip: [u8; 16], time_s: f32 },
    /// Reserved. See the type's docs — a build that meets one refuses by name.
    Reserved4 { clip: [u8; 16], time_s: f32 },
}

impl AdditiveRef {
    /// Whether the clip is additive at all.
    pub fn is_additive(&self) -> bool {
        !matches!(self, AdditiveRef::None)
    }

    /// The base clip this reference names, if it names one.
    pub fn base_clip(&self) -> Option<([u8; 16], f32)> {
        match self {
            AdditiveRef::BaseClip { clip, time_s } => Some((*clip, *time_s)),
            _ => None,
        }
    }

    /// `Some(slot)` when this is a reserved variant a future build wrote — the
    /// decode door's named refusal.
    pub fn reserved_slot(&self) -> Option<u32> {
        match self {
            AdditiveRef::Reserved3 { .. } => Some(3),
            AdditiveRef::Reserved4 { .. } => Some(4),
            _ => None,
        }
    }
}

/// The clip's **root motion**, baked: where the root joint is, over time,
/// including the vertical.
///
/// Present-but-optional in v2 (Ruling 2.4). Nothing in P29.2 writes one —
/// P29.4/P29.5 are the producers — and the reason it is here now is that the
/// format does not get a second bump.
///
/// # Why Y is in it
///
/// [`crate::root_motion::root_delta`] extracts XZ and drops Y, which was a sound
/// v1 simplification (a jump is the character controller's arc, not the clip's)
/// and is wrong for the whole traversal half of the catalogue: a mantle's rise
/// *is* the clip's, and a controller that re-derives it from gravity puts the
/// hands somewhere other than the ledge.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RootMotionTrack {
    pub times: Vec<f32>,
    /// Root translation in the clip's own space, metres, **Y included**.
    pub translation: Vec<[f32; 3]>,
    /// Root yaw about +Y, **radians** — the same unit and axis
    /// [`crate::RootMotionDelta::yaw`] already uses, so a consumer never converts.
    pub yaw_rad: Vec<f32>,
}

impl RootMotionTrack {
    /// The baked root transform at `t` seconds: `(translation, yaw_rad)`, both
    /// linearly interpolated, clamped at the ends. `None` when the track is empty.
    pub fn sample(&self, t: f32) -> Option<([f32; 3], f32)> {
        let (i0, i1, frac) = locate(&self.times, t)?;
        let a = *self.translation.get(i0)?;
        let b = *self.translation.get(i1)?;
        let ya = *self.yaw_rad.get(i0)?;
        let yb = *self.yaw_rad.get(i1)?;
        let lerp = |x: f32, y: f32| x + (y - x) * frac;
        Some((
            [lerp(a[0], b[0]), lerp(a[1], b[1]), lerp(a[2], b[2])],
            lerp(ya, yb),
        ))
    }
}

/// The clip's **distance track**: metres travelled by the root, cumulative.
///
/// The input to distance matching (P29.4) — "how far into this stop animation is
/// the frame where the foot lands exactly 1.4 m from here". Derived at import
/// from [`RootMotionTrack`]; present-but-optional in v2 for the same reason it is.
///
/// `distance_m` is non-decreasing, and the decode door checks that rather than
/// assuming it: the consumer binary-searches it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DistanceTrack {
    pub times: Vec<f32>,
    /// Cumulative ground distance travelled by the root at each key, metres.
    pub distance_m: Vec<f32>,
}

impl DistanceTrack {
    /// Distance travelled at `t` seconds (clamped, linear). `None` when empty.
    pub fn sample(&self, t: f32) -> Option<f32> {
        let (i0, i1, frac) = locate(&self.times, t)?;
        let a = *self.distance_m.get(i0)?;
        let b = *self.distance_m.get(i1)?;
        Some(a + (b - a) * frac)
    }

    /// The **first** clip time at which the root has travelled `d` metres —
    /// distance matching's primitive, run backwards through the same table.
    ///
    /// Clamps: below the first sample returns the first time, past the last
    /// returns the last. `None` when the track is empty.
    pub fn time_at_distance(&self, d: f32) -> Option<f32> {
        let n = self.times.len().min(self.distance_m.len());
        if n == 0 {
            return None;
        }
        if n == 1 || !(d > self.distance_m[0]) {
            return Some(self.times[0]);
        }
        if d >= self.distance_m[n - 1] {
            return Some(self.times[n - 1]);
        }
        // `distance_m` is non-decreasing (checked at the decode door), so the
        // first key strictly past `d` brackets it.
        let hi = self.distance_m[..n]
            .partition_point(|&x| x <= d)
            .clamp(1, n - 1);
        let (lo, span) = (hi - 1, self.distance_m[hi] - self.distance_m[hi - 1]);
        let frac = if span > 0.0 {
            ((d - self.distance_m[lo]) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Some(self.times[lo] + (self.times[hi] - self.times[lo]) * frac)
    }
}

/// The **reference vocabulary**: the 25 curve names ALS reads by name, spelled
/// byte-exactly (port map §2.14).
///
/// These are not an enum and not a restriction — see the module docs. They are
/// here so that the derivation P29.5 writes and the bridge P29.4 reads agree on
/// a spelling that was copied out of C++ once, instead of being retyped at three
/// call sites. `ALL` is the list a test asserts is deduplicated and sorted-stable,
/// which is the only property a vocabulary can actually have.
pub mod als {
    /// Normal base-pose weight.
    pub const BASE_POSE_N: &str = "BasePose_N";
    /// Crouching base-pose weight (also biases stride blend).
    pub const BASE_POSE_CLF: &str = "BasePose_CLF";
    pub const ENABLE_FOOT_IK_L: &str = "Enable_FootIK_L";
    pub const ENABLE_FOOT_IK_R: &str = "Enable_FootIK_R";
    pub const ENABLE_HAND_IK_L: &str = "Enable_HandIK_L";
    pub const ENABLE_HAND_IK_R: &str = "Enable_HandIK_R";
    /// Gates turn-in-place and the dynamic-transition corrective.
    pub const ENABLE_TRANSITION: &str = "Enable_Transition";
    pub const FOOT_LOCK_L: &str = "FootLock_L";
    pub const FOOT_LOCK_R: &str = "FootLock_R";
    pub const LAYERING_ARM_L: &str = "Layering_Arm_L";
    pub const LAYERING_ARM_L_ADD: &str = "Layering_Arm_L_Add";
    /// Local-space vs mesh-space switch for the left arm layer.
    pub const LAYERING_ARM_L_LS: &str = "Layering_Arm_L_LS";
    pub const LAYERING_ARM_R: &str = "Layering_Arm_R";
    pub const LAYERING_ARM_R_ADD: &str = "Layering_Arm_R_Add";
    pub const LAYERING_ARM_R_LS: &str = "Layering_Arm_R_LS";
    pub const LAYERING_HAND_L: &str = "Layering_Hand_L";
    pub const LAYERING_HAND_R: &str = "Layering_Hand_R";
    pub const LAYERING_HEAD_ADD: &str = "Layering_Head_Add";
    pub const LAYERING_SPINE_ADD: &str = "Layering_Spine_Add";
    /// Suppresses the aim offset (read inverted).
    pub const MASK_AIM_OFFSET: &str = "Mask_AimOffset";
    pub const MASK_LAND_PREDICTION: &str = "Mask_LandPrediction";
    pub const MASK_FOOTSTEP_SOUND: &str = "Mask_FootstepSound";
    /// Per-frame turn **degrees**, authored at 30 fps — the curve that drives
    /// actor yaw during a turn-in-place. Ours is superseded by a real warp window
    /// (Ruling 1); the name is kept because imported content carries it.
    pub const ROTATION_AMOUNT: &str = "RotationAmount";
    /// 1 = walk, 2 = run, 3 = sprint; biased and clamped twice to yield two
    /// independent 0..1 factors from one curve.
    pub const W_GAIT: &str = "W_Gait";
    /// Body-vs-aim yaw offset in looking-direction mode.
    pub const YAW_OFFSET: &str = "YawOffset";

    /// All 25, in the order the port map tabulates them.
    pub const ALL: [&str; 25] = [
        BASE_POSE_N,
        BASE_POSE_CLF,
        ENABLE_FOOT_IK_L,
        ENABLE_FOOT_IK_R,
        ENABLE_HAND_IK_L,
        ENABLE_HAND_IK_R,
        ENABLE_TRANSITION,
        FOOT_LOCK_L,
        FOOT_LOCK_R,
        LAYERING_ARM_L,
        LAYERING_ARM_L_ADD,
        LAYERING_ARM_L_LS,
        LAYERING_ARM_R,
        LAYERING_ARM_R_ADD,
        LAYERING_ARM_R_LS,
        LAYERING_HAND_L,
        LAYERING_HAND_R,
        LAYERING_HEAD_ADD,
        LAYERING_SPINE_ADD,
        MASK_AIM_OFFSET,
        MASK_LAND_PREDICTION,
        MASK_FOOTSTEP_SOUND,
        ROTATION_AMOUNT,
        W_GAIT,
        YAW_OFFSET,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[test]
    fn a_curve_channel_samples_linearly_and_clamps() {
        let c = CurveChannel::new(
            als::W_GAIT,
            vec![0.0, 1.0, 2.0],
            vec![1.0, 2.0, 3.0],
            Interpolation::Linear,
        );
        assert_eq!(c.sample(0.0), Some(1.0));
        assert_eq!(c.sample(0.5), Some(1.5));
        assert_eq!(c.sample(2.0), Some(3.0));
        // Clamps at both ends.
        assert_eq!(c.sample(-9.0), Some(1.0));
        assert_eq!(c.sample(99.0), Some(3.0));
        // A NaN probe locates rather than panics (`clip::locate`'s C4-7 rule).
        assert_eq!(c.sample(f32::NAN), Some(1.0));
        // An empty channel has nothing to say.
        assert_eq!(
            CurveChannel::new("x", Vec::new(), Vec::new(), Interpolation::Linear).sample(0.0),
            None
        );
    }

    #[test]
    fn a_step_curve_holds_the_earlier_key() {
        let c = CurveChannel::new(
            als::ENABLE_FOOT_IK_L,
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            Interpolation::Step,
        );
        assert_eq!(c.sample(0.99), Some(1.0));
        assert_eq!(c.sample(1.0), Some(0.0));
        // The constant helper is a one-key step curve.
        assert_eq!(CurveChannel::constant("k", 0.25).sample(7.0), Some(0.25));
    }

    #[test]
    fn a_marker_tells_a_sync_group_from_an_event() {
        let s = AnimMarker::sync(0.25, "plant_l", "foot");
        let e = AnimMarker::event(0.25, "footstep_l");
        assert!(s.is_sync());
        assert!(!e.is_sync());
    }

    #[test]
    fn a_distance_track_inverts() {
        let d = DistanceTrack {
            times: vec![0.0, 1.0, 2.0],
            distance_m: vec![0.0, 1.0, 4.0],
        };
        assert_eq!(d.sample(0.5), Some(0.5));
        assert_eq!(d.sample(1.5), Some(2.5));
        // …and back the other way.
        assert_eq!(d.time_at_distance(0.0), Some(0.0));
        assert_eq!(d.time_at_distance(0.5), Some(0.5));
        assert_eq!(d.time_at_distance(2.5), Some(1.5));
        // Clamps past both ends.
        assert_eq!(d.time_at_distance(-1.0), Some(0.0));
        assert_eq!(d.time_at_distance(99.0), Some(2.0));
        // A flat run (the character stood still) picks the first time of the run
        // rather than dividing by zero.
        let flat = DistanceTrack {
            times: vec![0.0, 1.0, 2.0],
            distance_m: vec![0.0, 0.0, 1.0],
        };
        assert_eq!(flat.time_at_distance(0.0), Some(0.0));
        assert_eq!(DistanceTrack::default().time_at_distance(1.0), None);
    }

    #[test]
    fn a_root_motion_track_carries_y() {
        let r = RootMotionTrack {
            times: vec![0.0, 1.0],
            translation: vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            yaw_rad: vec![0.0, 0.5],
        };
        let (t, yaw) = r.sample(0.5).unwrap();
        assert_eq!(t, [0.5, 1.0, 1.5]);
        assert_eq!(yaw, 0.25);
        // The vertical is the whole point: it is NOT zero (see the type's docs
        // on why `root_delta` drops it and this must not).
        assert!(t[1] > 0.0, "the root-motion track dropped Y");
        assert_eq!(RootMotionTrack::default().sample(0.0), None);
    }

    /// **The wire discriminants of [`AdditiveRef`] are frozen, and the reserved
    /// slots really are reachable.**
    ///
    /// The freeze is the P19 wire-enum law. The second half is the one that is
    /// easy to write and never check: a reserved slot that no test ever encodes
    /// is a comment, not a slot — so both are round-tripped and named.
    #[test]
    fn freeze_pins() {
        fn idx<T: Serialize>(v: &T) -> u32 {
            let bytes = bincode::serde::encode_to_vec(v, inf_asset::bincode_config()).unwrap();
            bytes[0] as u32
        }
        let base = AdditiveRef::BaseClip {
            clip: [7; 16],
            time_s: 0.0,
        };
        assert_eq!(
            [
                idx(&AdditiveRef::None),
                idx(&AdditiveRef::FirstFrame),
                idx(&base),
                idx(&AdditiveRef::Reserved3 {
                    clip: [0; 16],
                    time_s: 0.0
                }),
                idx(&AdditiveRef::Reserved4 {
                    clip: [0; 16],
                    time_s: 0.0
                }),
            ],
            [0, 1, 2, 3, 4]
        );
        assert_eq!(AdditiveRef::default(), AdditiveRef::None);
        assert!(!AdditiveRef::None.is_additive());
        assert!(AdditiveRef::FirstFrame.is_additive());
        assert_eq!(base.base_clip(), Some(([7; 16], 0.0)));
        assert_eq!(base.reserved_slot(), None);
        assert_eq!(
            AdditiveRef::Reserved4 {
                clip: [0; 16],
                time_s: 0.0
            }
            .reserved_slot(),
            Some(4)
        );
        // A reserved slot carries `BaseClip`'s payload, so a future kind has room
        // without desyncing the positional decode: same encoded length.
        let cfg = inf_asset::bincode_config();
        assert_eq!(
            bincode::serde::encode_to_vec(&base, cfg).unwrap().len(),
            bincode::serde::encode_to_vec(
                &AdditiveRef::Reserved3 {
                    clip: [7; 16],
                    time_s: 0.0
                },
                cfg
            )
            .unwrap()
            .len()
        );
    }

    /// The reference vocabulary is a **list**, and the only property a list of
    /// names can have is that it has no duplicates and every entry is non-empty.
    /// (A retyped name would show up here as a duplicate or as a changed count.)
    #[test]
    fn the_als_curve_vocabulary_is_twenty_five_distinct_names() {
        let set: std::collections::BTreeSet<&str> = als::ALL.iter().copied().collect();
        assert_eq!(set.len(), 25, "a name is duplicated: {:?}", als::ALL);
        assert!(als::ALL.iter().all(|n| !n.is_empty()));
        // Byte-exactness matters — these are compared against imported content.
        assert!(set.contains("Layering_Arm_L_LS"));
        assert!(set.contains("Mask_AimOffset"));
        assert!(set.contains("W_Gait"));
    }
}
