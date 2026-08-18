//! **Motion warping** (P29.4, pillar S8): per-clip warp windows that scale a
//! clip's root motion onto a runtime target, plus the two runtime primitives
//! that make starts, stops and pivots hit their marks — **distance matching**
//! and **orientation warping**.
//!
//! # What it replaces, and why the donor's version is not ported
//!
//! ALS's mantle drives the actor transform *directly* through three independent
//! alphas read out of one vector curve, plus an "animated start offset" that
//! records where the animation thinks it began (port map §3.3–3.4). All of that
//! bookkeeping exists because the montage's own displacement is not available as
//! data: the animation is played, the actor is placed, and the two are reconciled
//! by hand-authored correction curves.
//!
//! With a **baked root-motion track** (`.inf_anim` v2, P29.2) the displacement
//! *is* data, so the whole reconciliation collapses into one statement:
//!
//! > scale the root motion delivered inside the window so that consuming the
//! > whole window lands exactly on the target.
//!
//! That is [`warp_offset`]. The animated-start bookkeeping, the horizontal and
//! vertical correction lerps and the outer blend-in all disappear — the clip's
//! arc is preserved (it is what is being scaled) and the endpoint is exact by
//! construction rather than by an ease that hopefully converges.
//!
//! ALS's **height → (start position, play rate)** remap is kept, because it is a
//! genuinely good idea and it is orthogonal: it chooses *where in the clip to
//! start and how fast to play it* so that a 0.9 m ledge and a 1.3 m ledge share
//! one animation. That is [`height_remap`], with the constants named as the
//! reference behaviour and every centimetre converted to metres once, here.
//!
//! # Portability
//!
//! Adds, multiplies, comparisons and `sqrt`. The one rotation goes through
//! [`crate::root_motion::root_delta_world_3d`], which is the `psin64`/`pcos64`
//! door both fixed steps already share. A warped position is written onto an
//! entity's `Transform` and folded into `state_bytes`, so this file is on
//! `tests/portable_pose.rs`'s list.

use glam::{DVec3, Vec3};

use crate::channels::DistanceTrack;

/// The **hard high/low mantle split**, metres.
///
/// ALS's `MantleCheck` classifies on a literal `125.0` centimetres (port map
/// §3.2, `.cpp:282`) — not a config field, a literal in the check. Converted once,
/// here, per the cm→m rule at the port boundary.
pub const MANTLE_HIGH_SPLIT_M: f64 = 1.25;

/// A **warp window**: the span of a clip whose root motion is scaled onto a
/// runtime target.
///
/// Times are clip seconds. A window with `end <= start` is degenerate and every
/// consumer treats it as "warp nothing", which is a value rather than a division.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WarpWindow {
    /// Clip time the window opens, seconds.
    pub start_s: f32,
    /// Clip time the window closes, seconds.
    pub end_s: f32,
}

impl WarpWindow {
    /// A window over `[start_s, end_s]`.
    pub fn new(start_s: f32, end_s: f32) -> Self {
        Self { start_s, end_s }
    }

    /// The whole clip.
    pub fn whole(duration_s: f32) -> Self {
        Self {
            start_s: 0.0,
            end_s: duration_s,
        }
    }

    /// Whether the window spans a positive amount of clip.
    pub fn is_usable(&self) -> bool {
        crate::greater(self.end_s, self.start_s)
    }

    /// `t` clamped into the window.
    pub fn clamp(&self, t: f32) -> f32 {
        if !self.is_usable() {
            return self.start_s;
        }
        t.clamp(self.start_s, self.end_s)
    }

    /// How far through the window `t` is, `[0, 1]`; `1` for a degenerate window
    /// (nothing left to warp).
    pub fn alpha(&self, t: f32) -> f32 {
        if !self.is_usable() {
            return 1.0;
        }
        ((t - self.start_s) / (self.end_s - self.start_s)).clamp(0.0, 1.0)
    }
}

/// The **warp offset**: where a warped character is, relative to where the warp
/// started, given how much root motion the clip has delivered so far.
///
/// * `start_yaw_deg` is the character's facing when the window opened, so the
///   clip's own frame can be rotated into the world's.
/// * `delivered` is the root motion the clip has produced from the window's start
///   up to now, in the clip's own facing frame (Y included).
/// * `total` is what the clip produces over the **whole** window, same frame.
/// * `target_offset` is where the character must end up, world space, relative to
///   where it started.
///
/// # The rule, stated once
///
/// Per world axis, the answer is `delivered_world × (target / total_world)` — the
/// clip's shape, scaled so that `delivered == total` lands exactly on the target.
/// An axis the clip barely moves along cannot be scaled (the factor would be
/// unbounded and one keyframe of noise would become metres), so an axis whose
/// total is under [`MIN_WARP_AXIS_M`] is **corrected additively** instead: the
/// clip contributes what it has, and the remainder is closed linearly in the
/// window's own progress. Both halves end on the target; they differ only in how
/// the middle looks, and the additive half is the honest answer when there is no
/// shape to preserve.
///
/// `alpha` is the window progress used by that additive half — pass
/// [`WarpWindow::alpha`] of the current clip time.
pub fn warp_offset(
    start_yaw_deg: f64,
    delivered: Vec3,
    total: Vec3,
    target_offset: DVec3,
    alpha: f64,
) -> DVec3 {
    let delivered_w = crate::root_motion::root_delta_world_3d(start_yaw_deg, delivered);
    let total_w = crate::root_motion::root_delta_world_3d(start_yaw_deg, total);
    let a = if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let axis = |d: f64, t: f64, target: f64| -> f64 {
        if !d.is_finite() || !t.is_finite() || !target.is_finite() {
            return 0.0;
        }
        if t.abs() >= MIN_WARP_AXIS_M {
            d * (target / t)
        } else {
            // Nothing to scale: carry the clip's own contribution and close the
            // rest linearly. At `a == 1` this is exactly `target`.
            d + (target - d) * a
        }
    };
    DVec3::new(
        axis(delivered_w.x, total_w.x, target_offset.x),
        axis(delivered_w.y, total_w.y, target_offset.y),
        axis(delivered_w.z, total_w.z, target_offset.z),
    )
}

/// The smallest per-axis root-motion total, metres, that [`warp_offset`] will
/// scale rather than correct additively.
///
/// One centimetre over a whole warp window. Below it the scale factor is both
/// unbounded and meaningless — the clip has no shape along that axis to preserve.
pub const MIN_WARP_AXIS_M: f64 = 0.01;

/// **Orientation warping**: the yaw a warped character should be at.
///
/// The same rule as [`warp_offset`] on one angle — scale the clip's own turn so
/// that consuming the window arrives exactly at `target_delta_deg`, and fall back
/// to a linear close when the clip barely turns.
///
/// This is what supersedes ALS's `RotationAmount ÷ 30 fps` scheme (§13's
/// [SUPERSEDE], Ruling 1): there, a 90°-authored clip is rescaled by a scalar
/// written into a curve at the authoring frame rate. Here the clip's turn is
/// read off its own baked track and rescaled against the runtime target, so the
/// authoring frame rate never enters the arithmetic.
pub fn warp_yaw_deg(
    delivered_deg: f64,
    total_deg: f64,
    target_delta_deg: f64,
    alpha: f64,
) -> f64 {
    let a = if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        1.0
    };
    if !delivered_deg.is_finite() || !total_deg.is_finite() || !target_delta_deg.is_finite() {
        return 0.0;
    }
    if total_deg.abs() >= MIN_WARP_YAW_DEG {
        delivered_deg * (target_delta_deg / total_deg)
    } else {
        delivered_deg + (target_delta_deg - delivered_deg) * a
    }
}

/// The smallest total turn, degrees, [`warp_yaw_deg`] will scale.
pub const MIN_WARP_YAW_DEG: f64 = 1.0;

/// **Distance matching**: the clip time at which the root has travelled
/// `distance_m`, and the play rate that would consume the rest of the clip in
/// `time_left_s`.
///
/// `None` when the clip carries no [`DistanceTrack`] — a refusal as a value, so a
/// caller with an unbaked clip plays it at unit rate rather than dividing by a
/// distance nobody measured.
///
/// The play rate is clamped to `[MIN_PLAY_RATE, MAX_PLAY_RATE]`, which is ALS's
/// own `clamp(…, 0, 3)` on the standing play rate widened at the bottom: a rate
/// of zero is a frozen character, and this primitive is used to *stop* on a mark.
pub fn distance_match(track: &DistanceTrack, distance_m: f32) -> Option<f32> {
    track.time_at_distance(distance_m)
}

/// The lower bound [`play_rate_for`] clamps to.
pub const MIN_PLAY_RATE: f64 = 0.1;
/// The upper bound [`play_rate_for`] clamps to (ALS's `CalculateStandingPlayRate`
/// ceiling).
pub const MAX_PLAY_RATE: f64 = 3.0;

/// The play rate that consumes `clip_left_s` of clip in `time_left_s` of world
/// time.
///
/// Both non-positive inputs answer `1.0` — a rate of "as authored" — because a
/// caller with nothing left to fit has no rate to derive.
pub fn play_rate_for(clip_left_s: f64, time_left_s: f64) -> f64 {
    if !clip_left_s.is_finite() || !time_left_s.is_finite() || clip_left_s <= 0.0 || time_left_s <= 0.0
    {
        return 1.0;
    }
    (clip_left_s / time_left_s).clamp(MIN_PLAY_RATE, MAX_PLAY_RATE)
}

/// The reference constants of one traversal clip: what ledge heights it was
/// authored for and how it is entered at each end.
///
/// This is ALS's `FALSMantleAsset` (port map §3.3, `.cpp:103–111`) with every
/// centimetre converted to metres **once, here**. It is a *reference behaviour*
/// rather than a ported mechanism: the placement is [`warp_offset`]'s, and this
/// only decides where in the clip to start and how fast to play it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightRemap {
    /// The lowest ledge this clip covers, metres.
    pub low_height_m: f64,
    /// The highest, metres.
    pub high_height_m: f64,
    /// Clip start time for a ledge at `low_height_m`, seconds.
    pub low_start_s: f64,
    /// Clip start time for a ledge at `high_height_m`, seconds.
    pub high_start_s: f64,
    /// Play rate at `low_height_m`.
    pub low_play_rate: f64,
    /// Play rate at `high_height_m`.
    pub high_play_rate: f64,
}

impl Default for HeightRemap {
    /// ALS's shipped `Mantle_1m` numbers, in metres: a clip authored for a 1 m
    /// ledge, entered later and played faster for a low one.
    fn default() -> Self {
        Self {
            low_height_m: 0.5,
            high_height_m: 1.25,
            low_start_s: 0.6,
            high_start_s: 0.0,
            low_play_rate: 1.2,
            high_play_rate: 1.0,
        }
    }
}

impl HeightRemap {
    /// The `(start_s, play_rate)` for a ledge `height_m` above the feet — ALS's
    /// two `MappedRangeClamped`s, which is a 1-D distance-matching scheme.
    pub fn resolve(&self, height_m: f64) -> (f64, f64) {
        (
            map_range(
                height_m,
                (self.low_height_m, self.high_height_m),
                (self.low_start_s, self.high_start_s),
            ),
            map_range(
                height_m,
                (self.low_height_m, self.high_height_m),
                (self.low_play_rate, self.high_play_rate),
            )
            .clamp(MIN_PLAY_RATE, MAX_PLAY_RATE),
        )
    }
}

/// [`HeightRemap::resolve`] as a free function, for a caller that has the six
/// numbers and no struct.
pub fn height_remap(height_m: f64, remap: &HeightRemap) -> (f64, f64) {
    remap.resolve(height_m)
}

/// Linear map with a clamp and a degenerate-range guard — the same arithmetic
/// `inf_ecs::movement::mapped_range` performs, spelled here because this crate
/// does not depend on that one.
fn map_range(value: f64, from: (f64, f64), to: (f64, f64)) -> f64 {
    let span = from.1 - from.0;
    if !span.is_finite() || span.abs() < f64::EPSILON || !value.is_finite() {
        return to.0;
    }
    let t = ((value - from.0) / span).clamp(0.0, 1.0);
    to.0 + (to.1 - to.0) * t
}

/// A **smooth window ease** `[0, 1] → [0, 1]` for the additive half of a warp.
///
/// The smoothstep polynomial `3a² − 2a³`: zero derivative at both ends, so a
/// correction that is closing linearly does not start or stop with a velocity
/// step. A polynomial rather than a curve asset, because this file is on the
/// portable-math ban list.
pub fn warp_ease(alpha: f64) -> f64 {
    let a = if alpha.is_finite() {
        alpha.clamp(0.0, 1.0)
    } else {
        1.0
    };
    a * a * (3.0 - 2.0 * a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_normalizes_and_refuses_the_degenerate_case() {
        let w = WarpWindow::new(0.25, 0.75);
        assert!(w.is_usable());
        assert_eq!(w.alpha(0.25), 0.0);
        assert_eq!(w.alpha(0.5), 0.5);
        assert_eq!(w.alpha(0.75), 1.0);
        assert_eq!(w.alpha(9.0), 1.0, "clamps past the end");
        assert_eq!(w.clamp(0.0), 0.25);

        let dead = WarpWindow::new(0.5, 0.5);
        assert!(!dead.is_usable());
        assert_eq!(dead.alpha(0.5), 1.0, "nothing left to warp");
        // A NaN end is not usable, and does not panic on the way to saying so.
        assert!(!WarpWindow::new(0.0, f32::NAN).is_usable());
    }

    /// **The endpoint is exact by construction**, which is the whole claim: a
    /// clip that moves 1 m forward, warped onto a 1.6 m target, lands on 1.6 m —
    /// and the middle of the warp is the CLIP's shape, not a lerp.
    #[test]
    fn a_warp_lands_exactly_on_its_target_and_keeps_the_clips_shape() {
        // The clip rises in a curve: half the forward travel happens in the first
        // quarter of the window (an ease-out), so a lerp and a warp differ.
        let total = Vec3::new(0.0, 1.0, 1.0);
        let target = DVec3::new(0.0, 1.5, 1.6);

        // End of the window: exact.
        let end = warp_offset(0.0, total, total, target, 1.0);
        assert!((end - target).length() < 1e-12, "{end:?}");

        // A quarter of the way through the window but HALF the clip's travel:
        // the warp must be at half the target, not at a quarter.
        let delivered = Vec3::new(0.0, 0.5, 0.5);
        let mid = warp_offset(0.0, delivered, total, target, 0.25);
        assert!((mid.z - 0.8).abs() < 1e-12, "{mid:?}");
        assert!((mid.y - 0.75).abs() < 1e-12, "{mid:?}");

        // Yawed: the clip's local frame is rotated into the world's. Facing +X
        // (yaw 90), the clip's forward becomes world +X.
        let yawed = warp_offset(90.0, total, total, DVec3::new(1.6, 1.5, 0.0), 1.0);
        assert!((yawed - DVec3::new(1.6, 1.5, 0.0)).length() < 1e-9, "{yawed:?}");
    }

    /// The axis the clip does not move along is **corrected additively** rather
    /// than divided by — and still ends exactly on the target.
    #[test]
    fn an_axis_with_no_root_motion_is_corrected_not_divided() {
        // The clip goes straight forward; the target is 0.4 m to the side.
        let total = Vec3::new(0.0, 0.0, 1.0);
        let target = DVec3::new(0.4, 0.0, 1.0);
        let half = warp_offset(0.0, Vec3::new(0.0, 0.0, 0.5), total, target, 0.5);
        assert!((half.x - 0.2).abs() < 1e-12, "half the strafe at half alpha: {half:?}");
        assert!(half.x.is_finite(), "an unscalable axis must not divide: {half:?}");
        let end = warp_offset(0.0, total, total, target, 1.0);
        assert!((end - target).length() < 1e-12, "{end:?}");
        // Non-finite inputs are answers, not NaNs downstream.
        let bad = warp_offset(0.0, Vec3::new(f32::NAN, 0.0, 0.0), total, target, 0.5);
        assert!(bad.is_finite(), "{bad:?}");
    }

    #[test]
    fn orientation_warping_rescales_the_clips_own_turn() {
        // A clip authored to turn 90 degrees, warped onto a 63-degree target —
        // ALS's turn-in-place case, without its 30 fps constant.
        assert!((warp_yaw_deg(90.0, 90.0, 63.0, 1.0) - 63.0).abs() < 1e-12);
        assert!((warp_yaw_deg(45.0, 90.0, 63.0, 0.5) - 31.5).abs() < 1e-12);
        // A clip that does not turn closes the target linearly instead.
        assert!((warp_yaw_deg(0.0, 0.0, 40.0, 0.5) - 20.0).abs() < 1e-12);
        assert!((warp_yaw_deg(0.0, 0.0, 40.0, 1.0) - 40.0).abs() < 1e-12);
        assert_eq!(warp_yaw_deg(f64::NAN, 90.0, 63.0, 1.0), 0.0);
    }

    #[test]
    fn the_height_remap_is_als_arithmetic_in_metres() {
        let r = HeightRemap::default();
        // At the low end: start late, play fast.
        let (s, p) = r.resolve(0.5);
        assert!((s - 0.6).abs() < 1e-12 && (p - 1.2).abs() < 1e-12, "{s} {p}");
        // At the high end: start at the beginning, play as authored.
        let (s, p) = r.resolve(1.25);
        assert!((s - 0.0).abs() < 1e-12 && (p - 1.0).abs() < 1e-12, "{s} {p}");
        // Halfway is halfway, and outside the band clamps rather than extrapolates.
        let (s, _) = r.resolve(0.875);
        assert!((s - 0.3).abs() < 1e-12, "{s}");
        assert_eq!(r.resolve(-5.0).0, 0.6);
        assert_eq!(r.resolve(99.0).0, 0.0);
        // The split ALS hard-codes at 125 cm is 1.25 m here, converted once.
        assert_eq!(MANTLE_HIGH_SPLIT_M, 1.25);
    }

    #[test]
    fn distance_matching_and_the_play_rate_it_implies() {
        let track = DistanceTrack {
            times: vec![0.0, 0.5, 1.0],
            distance_m: vec![0.0, 1.0, 1.5],
        };
        assert_eq!(distance_match(&track, 1.0), Some(0.5));
        assert_eq!(distance_match(&track, 1.25), Some(0.75));
        // 0.5 s of clip left, 0.25 s of world time: play at 2x.
        assert!((play_rate_for(0.5, 0.25) - 2.0).abs() < 1e-12);
        // Clamped at both ends, and inert for a caller with nothing to fit.
        assert_eq!(play_rate_for(10.0, 0.001), MAX_PLAY_RATE);
        assert_eq!(play_rate_for(0.0, 1.0), 1.0);
        assert_eq!(play_rate_for(1.0, 0.0), 1.0);
    }

    #[test]
    fn the_ease_is_flat_at_both_ends() {
        assert_eq!(warp_ease(0.0), 0.0);
        assert_eq!(warp_ease(1.0), 1.0);
        assert!((warp_ease(0.5) - 0.5).abs() < 1e-12);
        // Flat at both ends: the first and last increments are far smaller than
        // the middle one, which is the property a velocity step would break.
        let d0 = warp_ease(0.01) - warp_ease(0.0);
        let dm = warp_ease(0.51) - warp_ease(0.5);
        assert!(d0 * 10.0 < dm, "{d0} vs {dm}");
        assert_eq!(warp_ease(f64::NAN), 1.0);
    }
}
