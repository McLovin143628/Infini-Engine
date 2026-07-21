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
        let a = Quat::from_array(self.values[i0]).normalize();
        if i0 == i1 || matches!(self.interp, Interpolation::Step) {
            return Some(a);
        }
        let mut b = Quat::from_array(self.values[i1]).normalize();
        // Shortest-path: negate the far quaternion so slerp takes the short arc.
        if a.dot(b) < 0.0 {
            b = -b;
        }
        Some(a.slerp(b, frac).normalize())
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
}

/// Map a raw sample time to a looped/clamped time within `[0, duration]`.
///
/// Looping wraps with a Euclidean remainder (so negative times wrap too);
/// non-looping clamps. A zero/negative duration collapses to `0`.
pub fn resolve_time(t: f32, duration: f32, looping: bool) -> f32 {
    if duration <= 0.0 {
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
fn locate(times: &[f32], t: f32) -> Option<(usize, usize, f32)> {
    match times.len() {
        0 => None,
        1 => Some((0, 0, 0.0)),
        _ => {
            if t <= times[0] {
                return Some((0, 0, 0.0));
            }
            let last = times.len() - 1;
            if t >= times[last] {
                return Some((last, last, 0.0));
            }
            // Binary search for the first key strictly greater than t.
            let hi = times.partition_point(|&k| k <= t);
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
}
