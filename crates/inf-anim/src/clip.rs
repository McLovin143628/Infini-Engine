//! Animation clips (P11.1): per-joint keyframe curves sampled to a [`Pose`].
//!
//! A clip is a bag of [`JointTrack`]s, each holding up to three channels
//! (translation / rotation / scale) as time-stamped keyframe curves. Sampling
//! binary-searches the keys and interpolates.
//!
//! # Interpolation modes (v1)
//!
//! glTF supports `STEP`, `LINEAR`, and `CUBICSPLINE`. v1 keeps **step** and
//! **linear**; cubic curves are **resampled to linear at their input keyframe
//! times on import** (a documented v1 simplification — the sample points are
//! preserved, only the in-between curvature is dropped). Higher-order tangents
//! return with the animation-graph work (P11.2).

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// How a track interpolates between adjacent keyframes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Interpolation {
    /// Hold the earlier keyframe's value until the next key (no blending).
    Step,
    /// Linearly interpolate (lerp for vectors, slerp for rotations).
    #[default]
    Linear,
}

/// A keyframed `vec3` channel (translation or scale). `times` is strictly
/// increasing; `times.len() == values.len()`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vec3Track {
    pub times: Vec<f32>,
    pub values: Vec<[f32; 3]>,
    pub interp: Interpolation,
}

impl Vec3Track {
    /// Build a track; `times` must match `values` in length.
    pub fn new(times: Vec<f32>, values: Vec<[f32; 3]>, interp: Interpolation) -> Self {
        debug_assert_eq!(times.len(), values.len(), "times/values length mismatch");
        Self {
            times,
            values,
            interp,
        }
    }

    /// Sample the curve at time `t` (clamped to the key range). Empty tracks
    /// return `None`.
    pub fn sample(&self, t: f32) -> Option<Vec3> {
        let (i0, i1, frac) = locate(&self.times, t)?;
        let a = Vec3::from_array(self.values[i0]);
        if i0 == i1 || matches!(self.interp, Interpolation::Step) {
            return Some(a);
        }
        let b = Vec3::from_array(self.values[i1]);
        Some(a.lerp(b, frac))
    }
}

/// A keyframed rotation channel (quaternions, `[x, y, z, w]`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuatTrack {
    pub times: Vec<f32>,
    pub values: Vec<[f32; 4]>,
    pub interp: Interpolation,
}

impl QuatTrack {
    /// Build a track; `times` must match `values` in length.
    pub fn new(times: Vec<f32>, values: Vec<[f32; 4]>, interp: Interpolation) -> Self {
        debug_assert_eq!(times.len(), values.len(), "times/values length mismatch");
        Self {
            times,
            values,
            interp,
        }
    }

    /// Sample the curve at time `t` (clamped). Uses shortest-path slerp for
    /// linear tracks. Empty tracks return `None`.
    pub fn sample(&self, t: f32) -> Option<Quat> {
        let (i0, i1, frac) = locate(&self.times, t)?;
        let a = normalized_or_identity(self.values[i0]);
        if i0 == i1 || matches!(self.interp, Interpolation::Step) {
            return Some(a);
        }
        let mut b = normalized_or_identity(self.values[i1]);
        // Shortest-path: negate the far quaternion so slerp takes the short arc.
        if a.dot(b) < 0.0 {
            b = -b;
        }
        // **Portable, not `Quat::slerp`** (P24.2 audit M-SLERP): this is the
        // `#[default]` interpolation and what the glTF importer emits, so it is
        // the single most-travelled quaternion blend in the engine -- and its
        // result rides `state_bytes`. `pslerp` normalizes internally.
        Some(inf_math::pslerp(a, b, frac))
    }
}

/// Normalize a stored quaternion, falling back to identity when the result is
/// not a rotation.
///
/// # `Quat::normalize()` on a zero quaternion is NaN, silently (C4-31)
///
/// The workspace pins bare `glam = "0.33"` with **no `glam-assert` /
/// `debug-glam-assert` feature**, so `normalize()` is `self * length_recip()`
/// with the check compiled out — in debug as well as release. `[0, 0, 0, 0]` is
/// legal bytes in a glTF `ROTATION` accessor and in a `.inf_anim`, and it comes
/// back as four NaNs.
///
/// Where that goes is the reason this is not cosmetic: into
/// `Pose.locals[].rotation`, and from there into
/// `inf_ecs::pose::pose_state_bytes`, which folds `f32::to_le_bytes()` of every
/// joint TRS into **the committed trace the PIE-==-shipping parity gate
/// compares**. A NaN never equals itself, so the gate's own comparison becomes
/// meaningless rather than failing usefully.
///
/// Identity is the right fallback for the same reason it is the right fallback
/// in `JointTransform`: a joint with no usable rotation should sit at its bind
/// orientation, not vanish.
fn normalized_or_identity(v: [f32; 4]) -> Quat {
    let q = Quat::from_array(v).normalize();
    if q.is_finite() {
        q
    } else {
        Quat::IDENTITY
    }
}

/// The animated channels of a single joint. Any subset of the three channels may
/// be present; a missing channel falls back to the joint's bind transform.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointTrack {
    /// Index of the joint (into the skeleton's `joints`) this track drives.
    pub joint: u16,
    pub translation: Option<Vec3Track>,
    pub rotation: Option<QuatTrack>,
    pub scale: Option<Vec3Track>,
}

impl JointTrack {
    /// An empty track targeting `joint` (no channels).
    pub fn new(joint: u16) -> Self {
        Self {
            joint,
            translation: None,
            rotation: None,
            scale: None,
        }
    }
}

/// A named animation clip: a set of per-joint keyframe tracks over `duration`
/// seconds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimClip {
    pub name: String,
    /// Total clip length in seconds (the max keyframe time across all tracks).
    pub duration: f32,
    pub tracks: Vec<JointTrack>,
}

impl AnimClip {
    /// Build a clip, computing `duration` as the maximum keyframe time across
    /// every channel of every track (0 for an empty/static clip).
    pub fn new(name: impl Into<String>, tracks: Vec<JointTrack>) -> Self {
        let mut duration = 0.0f32;
        for tr in &tracks {
            for times in [
                tr.translation.as_ref().map(|c| &c.times),
                tr.rotation.as_ref().map(|c| &c.times),
                tr.scale.as_ref().map(|c| &c.times),
            ]
            .into_iter()
            .flatten()
            {
                if let Some(&last) = times.last() {
                    duration = duration.max(last);
                }
            }
        }
        Self {
            name: name.into(),
            duration,
            tracks,
        }
    }

    /// Whether this clip's timing is something the samplers can be handed.
    ///
    /// Called by `AnimClipAsset::migrate`, so no decoded `.inf_anim` reaches the
    /// frame loop without it. Three facts, each one an assumption a consumer
    /// already makes:
    ///
    /// * `duration` is finite and not negative — `resolve_time` clamps against
    ///   it and `AnimPlayer::advance` takes a `rem_euclid` of it;
    /// * every keyframe time is finite — [`locate`] binary-searches them;
    /// * `times.len() == values.len()` on every channel — `sample()` indexes
    ///   `values` with an index derived from `times`, and the constructors only
    ///   `debug_assert` the agreement, which is compiled out in release.
    ///
    /// Monotonicity is checked at the **import** door
    /// (`inf_mesh::validate::reject_non_increasing`) rather than here: a
    /// committed clip that a P11-era build wrote is entitled to whatever key
    /// spacing it has, and `locate` is now total over an unsorted list. A NaN
    /// is not spacing.
    pub fn validate_timing(&self) -> Result<(), String> {
        if !self.duration.is_finite() || self.duration < 0.0 {
            return Err(format!(
                "clip {:?} has duration {}",
                self.name, self.duration
            ));
        }
        for tr in &self.tracks {
            let check = |times: &[f32], values: usize, chan: &str| -> Result<(), String> {
                if times.len() != values {
                    return Err(format!(
                        "clip {:?} joint {} {chan}: {} keyframe times against {values} values",
                        self.name,
                        tr.joint,
                        times.len()
                    ));
                }
                match times.iter().position(|t| !t.is_finite()) {
                    Some(i) => Err(format!(
                        "clip {:?} joint {} {chan}: keyframe time {i} is {}",
                        self.name, tr.joint, times[i]
                    )),
                    None => Ok(()),
                }
            };
            if let Some(c) = &tr.translation {
                check(&c.times, c.values.len(), "translation")?;
            }
            if let Some(c) = &tr.rotation {
                check(&c.times, c.values.len(), "rotation")?;
            }
            if let Some(c) = &tr.scale {
                check(&c.times, c.values.len(), "scale")?;
            }
        }
        Ok(())
    }
}

/// Map a raw sample time to a looped/clamped time within `[0, duration]`.
///
/// Looping wraps with a Euclidean remainder (so negative times wrap too);
/// non-looping clamps. A zero/negative **or non-finite** duration collapses to
/// `0`.
///
/// # `!(duration > 0.0)`, not `duration <= 0.0` (C4-4)
///
/// This is the house's standing NaN-blind-guard class, and it was live in the
/// frame loop. `NaN <= 0.0` is **false**, so a NaN duration passed the guard and
/// reached `t.clamp(0.0, NaN)` — which panics, because `f32::clamp` asserts
/// `min <= max`. `pose::sample_clip` calls this every frame, so the panic fired
/// in the editor, in PIE and in the shipped player, from nothing worse than a
/// `.inf_anim` with one poisoned float. In the looping branch there is no panic
/// and `rem_euclid(NaN)` poisons `self.t` instead, which is persisted back into
/// the `.inf_lvl`.
///
/// Written as the negation of the predicate the code actually needs — "is this a
/// usable positive duration" — which is the one phrasing a NaN answers
/// correctly.
pub fn resolve_time(t: f32, duration: f32, looping: bool) -> f32 {
    if !(duration > 0.0) {
        return 0.0;
    }
    if looping {
        t.rem_euclid(duration)
    } else {
        t.clamp(0.0, duration)
    }
}

/// Locate `t` within a strictly-increasing `times` list, returning the bracketing
/// key indices and the `[0,1]` fraction between them. Clamps outside the range.
/// `None` only when `times` is empty.
///
/// # Total over any `times` and any `t` (C4-7)
///
/// The range guards used to be `t <= times[0]` and `t >= times[last]`. A NaN `t`
/// fails **both** — every ordering comparison a NaN takes part in is false — so
/// it fell through to `partition_point`, whose predicate is also false for every
/// key, which returns `0`, and `let i0 = hi - 1` underflowed: a panic in the
/// shipped player. Asking `!(t > times[0])` instead puts NaN on the side that
/// returns a value.
///
/// The `clamp` on `hi` is the second half: a committed clip whose keys are not
/// sorted (nothing before the import door refused one) makes `partition_point`'s
/// result meaningless rather than merely wrong, and index arithmetic on a
/// meaningless number is how this class panics.
fn locate(times: &[f32], t: f32) -> Option<(usize, usize, f32)> {
    match times.len() {
        0 => None,
        1 => Some((0, 0, 0.0)),
        _ => {
            let last = times.len() - 1;
            if !(t > times[0]) {
                return Some((0, 0, 0.0));
            }
            if !(t < times[last]) {
                return Some((last, last, 0.0));
            }
            // Binary search for the first key strictly greater than t.
            let hi = times.partition_point(|&k| k <= t).clamp(1, last);
            let i0 = hi - 1;
            let i1 = hi;
            let span = times[i1] - times[i0];
            let frac = if span > 0.0 {
                ((t - times[i0]) / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            Some((i0, i1, frac))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_linear_samples_endpoints_and_midpoint() {
        let tr = Vec3Track::new(
            vec![0.0, 2.0],
            vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            Interpolation::Linear,
        );
        assert_eq!(tr.sample(0.0).unwrap(), Vec3::ZERO);
        assert_eq!(tr.sample(2.0).unwrap(), Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(tr.sample(1.0).unwrap(), Vec3::new(5.0, 0.0, 0.0));
        // Clamps outside the range.
        assert_eq!(tr.sample(-1.0).unwrap(), Vec3::ZERO);
        assert_eq!(tr.sample(99.0).unwrap(), Vec3::new(10.0, 0.0, 0.0));
    }

    #[test]
    fn step_holds_the_earlier_key() {
        let tr = Vec3Track::new(
            vec![0.0, 1.0],
            vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            Interpolation::Step,
        );
        assert_eq!(tr.sample(0.99).unwrap(), Vec3::ZERO);
        assert_eq!(tr.sample(1.0).unwrap(), Vec3::ONE);
    }

    #[test]
    fn quat_slerp_takes_shortest_path() {
        // 0° → 170°, but stored as the "long way" quaternion (negated); slerp
        // must still take the short 170° arc through ~85°, not the 190° arc.
        let a = Quat::IDENTITY;
        let far = Quat::from_rotation_z(170f32.to_radians());
        let neg = -far; // same orientation, opposite hemisphere
        let tr = QuatTrack::new(
            vec![0.0, 1.0],
            vec![a.to_array(), neg.to_array()],
            Interpolation::Linear,
        );
        let mid = tr.sample(0.5).unwrap();
        let ang = a.angle_between(mid).to_degrees();
        assert!((ang - 85.0).abs() < 1.0, "expected ~85°, got {ang}");
    }

    #[test]
    fn duration_is_the_max_key_time() {
        let mut jt = JointTrack::new(0);
        jt.translation = Some(Vec3Track::new(
            vec![0.0, 1.5],
            vec![[0.0; 3], [1.0; 3]],
            Interpolation::Linear,
        ));
        jt.rotation = Some(QuatTrack::new(
            vec![0.0, 3.0],
            vec![Quat::IDENTITY.to_array(), Quat::IDENTITY.to_array()],
            Interpolation::Linear,
        ));
        let clip = AnimClip::new("c", vec![jt]);
        assert_eq!(clip.duration, 3.0);
    }

    #[test]
    fn resolve_time_wraps_and_clamps() {
        // Looping wraps.
        assert_eq!(resolve_time(2.5, 2.0, true), 0.5);
        assert_eq!(resolve_time(-0.5, 2.0, true), 1.5);
        // Non-looping clamps.
        assert_eq!(resolve_time(2.5, 2.0, false), 2.0);
        assert_eq!(resolve_time(-1.0, 2.0, false), 0.0);
        // Degenerate duration.
        assert_eq!(resolve_time(5.0, 0.0, true), 0.0);
    }

    /// **C4-4 — the NaN-blind-guard class, in the frame loop.**
    ///
    /// `duration <= 0.0` is false for NaN, so a NaN duration used to reach
    /// `t.clamp(0.0, NaN)` — which panics, every frame, for as long as the clip
    /// is playing. Un-fix mutation: put `duration <= 0.0` back and the
    /// non-looping half of this test panics rather than failing.
    ///
    /// `+∞` is deliberately **not** in the refused set here: it is a positive
    /// duration, `rem_euclid(∞)` and `clamp(0, ∞)` both answer sensibly, and a
    /// decoded `.inf_anim` cannot carry one anyway (`AnimClipAsset::migrate`
    /// refuses non-finite). The guard's job is NaN and non-positive.
    #[test]
    fn a_non_finite_duration_collapses_instead_of_panicking() {
        for bad in [f32::NAN, f32::NEG_INFINITY, -1.0, 0.0] {
            for looping in [true, false] {
                assert_eq!(
                    resolve_time(0.25, bad, looping),
                    0.0,
                    "duration {bad} looping={looping} did not collapse"
                );
            }
        }
        for looping in [true, false] {
            assert!(resolve_time(0.25, f32::INFINITY, looping).is_finite());
        }
    }

    /// **C4-7 — `locate` is total.**
    ///
    /// A NaN `t` failed both range guards, `partition_point` returned 0, and
    /// `hi - 1` underflowed. An unsorted `times` (which a clip committed before
    /// the import door existed may carry) made the search result meaningless in
    /// the same arithmetic.
    #[test]
    fn locate_survives_a_nan_time_and_an_unsorted_key_list() {
        let times = [0.0, 1.0, 2.0];
        let (i0, i1, f) = locate(&times, f32::NAN).expect("NaN must locate, not panic");
        assert_eq!((i0, i1), (0, 0));
        assert_eq!(f, 0.0);
        // Unsorted keys: whatever bracket comes back, it must be in range.
        for probe in [-5.0f32, 0.5, 1.5, 9.0, f32::NAN] {
            let (i0, i1, f) = locate(&[3.0, 1.0, 2.0, 0.5], probe).unwrap();
            assert!(i0 < 4 && i1 < 4, "bracket ({i0}, {i1}) is out of range");
            assert!((0.0..=1.0).contains(&f), "fraction {f} is not in [0,1]");
        }
        // The healthy path is unchanged.
        let (i0, i1, f) = locate(&times, 1.5).unwrap();
        assert_eq!((i0, i1), (1, 2));
        assert!((f - 0.5).abs() < 1e-6);
    }

    /// **C4-31 — a zero quaternion is not a rotation.**
    ///
    /// `glam` is pinned without `glam-assert`, so `Quat::normalize()` on
    /// `[0,0,0,0]` is four NaNs with no complaint — and they ride
    /// `pose_state_bytes`, the trace the PIE-==-shipping parity gate compares.
    /// Un-fix mutation: restore `Quat::from_array(..).normalize()` at both call
    /// sites and this fails on `is_finite`.
    #[test]
    fn a_zero_quaternion_samples_as_identity_not_as_nan() {
        let tr = QuatTrack::new(
            vec![0.0, 1.0],
            vec![[0.0, 0.0, 0.0, 0.0], Quat::IDENTITY.to_array()],
            Interpolation::Linear,
        );
        for t in [0.0f32, 0.5, 1.0] {
            let q = tr.sample(t).unwrap();
            assert!(q.is_finite(), "sample({t}) produced {q:?}");
        }
        assert_eq!(tr.sample(0.0).unwrap(), Quat::IDENTITY);
        // Step interpolation reads the same door.
        let step = QuatTrack::new(vec![0.0], vec![[0.0, 0.0, 0.0, 0.0]], Interpolation::Step);
        assert_eq!(step.sample(0.0).unwrap(), Quat::IDENTITY);
    }

    /// The decode-door check (C4-4's other half): a clip whose timing the
    /// samplers cannot be handed is refused rather than guarded around.
    #[test]
    fn validate_timing_refuses_what_the_samplers_assume() {
        let good = AnimClip::new(
            "ok",
            vec![{
                let mut jt = JointTrack::new(0);
                jt.translation = Some(Vec3Track::new(
                    vec![0.0, 1.0],
                    vec![[0.0; 3], [1.0; 3]],
                    Interpolation::Linear,
                ));
                jt
            }],
        );
        good.validate_timing().unwrap();

        let mut nan_duration = good.clone();
        nan_duration.duration = f32::NAN;
        assert!(nan_duration.validate_timing().is_err());

        let mut negative = good.clone();
        negative.duration = -1.0;
        assert!(negative.validate_timing().is_err());

        // A stream-length disagreement: `sample()` indexes `values` with an
        // index derived from `times`, and the constructor only debug_asserts it.
        let mut short = good.clone();
        short.tracks[0].translation.as_mut().unwrap().values.pop();
        let e = short.validate_timing().unwrap_err();
        assert!(e.contains("2 keyframe times against 1"), "{e}");

        let mut nan_key = good.clone();
        nan_key.tracks[0].translation.as_mut().unwrap().times[1] = f32::NAN;
        assert!(nan_key.validate_timing().is_err());
    }
}
