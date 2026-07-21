//! Spatial-audio **basics**: a listener pose, a per-emitter position, and the
//! pure distance-attenuation + panning math that feeds kira's volume/panning.
//!
//! This is deliberately simple — an amplitude curve and a stereo pan derived from
//! geometry, all pure functions with no kira and no IO, so it is fully unit-
//! tested. Real HRTF, occlusion, and reverb are **P12.3** (`docs/ROADMAP.md`);
//! the API here is shaped so that work can slot in behind the same listener/emitter
//! vocabulary without a breaking change.

use glam::DVec3;

/// How an emitter's loudness falls off with distance from the listener.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttenuationModel {
    /// Linear ramp from full volume at `min_distance` to silence at
    /// `max_distance` — predictable, the usual pick for gameplay cues.
    Linear,
    /// Inverse-distance falloff (`min_distance / distance`), clamped to `[0, 1]`
    /// and to silence past `max_distance` — closer to physical 1/r rolloff.
    Inverse,
}

/// The distance-attenuation curve for a spatial emitter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attenuation {
    pub model: AttenuationModel,
    /// At or within this distance the emitter is at full volume (gain 1.0).
    pub min_distance: f64,
    /// At or beyond this distance the emitter is silent (gain 0.0).
    pub max_distance: f64,
}

impl Default for Attenuation {
    fn default() -> Self {
        Self {
            model: AttenuationModel::Inverse,
            min_distance: 1.0,
            max_distance: 100.0,
        }
    }
}

impl Attenuation {
    /// A linear-falloff curve between `min` and `max` world units.
    pub fn linear(min_distance: f64, max_distance: f64) -> Self {
        Self {
            model: AttenuationModel::Linear,
            min_distance,
            max_distance,
        }
    }

    /// An inverse-distance curve between `min` and `max` world units.
    pub fn inverse(min_distance: f64, max_distance: f64) -> Self {
        Self {
            model: AttenuationModel::Inverse,
            min_distance,
            max_distance,
        }
    }

    /// The amplitude gain in `[0, 1]` for an emitter `distance` world units from
    /// the listener. Guards degenerate configs (`max <= min`) by treating them as
    /// a hard cutoff at `min_distance`.
    pub fn gain(&self, distance: f64) -> f64 {
        let d = distance.max(0.0);
        if d <= self.min_distance {
            return 1.0;
        }
        if d >= self.max_distance || self.max_distance <= self.min_distance {
            return 0.0;
        }
        match self.model {
            AttenuationModel::Linear => {
                ((self.max_distance - d) / (self.max_distance - self.min_distance)).clamp(0.0, 1.0)
            }
            AttenuationModel::Inverse => (self.min_distance / d).clamp(0.0, 1.0),
        }
    }
}

/// The listener's pose in world space — a position and an orientation basis. The
/// orientation drives stereo panning; `forward` and `up` need not be exactly
/// orthonormal (they are re-orthonormalized when computing `right`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Listener {
    pub position: DVec3,
    pub forward: DVec3,
    pub up: DVec3,
}

impl Default for Listener {
    fn default() -> Self {
        // Facing -Z with +Y up: the engine's default camera convention.
        Self {
            position: DVec3::ZERO,
            forward: DVec3::NEG_Z,
            up: DVec3::Y,
        }
    }
}

impl Listener {
    /// The listener's rightward axis (`forward × up`, normalized). Falls back to
    /// `+X` if the basis is degenerate.
    pub fn right(&self) -> DVec3 {
        let r = self.forward.cross(self.up);
        r.normalize_or(DVec3::X)
    }

    /// The `(gain, panning)` an emitter at `emitter` produces for this listener
    /// under `attenuation`. `gain` is `[0, 1]`; `panning` is `[-1, 1]`
    /// (−1 = hard left, 0 = centre, +1 = hard right), the sign of the emitter's
    /// projection onto the listener's right axis.
    pub fn resolve(&self, emitter: DVec3, attenuation: &Attenuation) -> (f64, f64) {
        let rel = emitter - self.position;
        let distance = rel.length();
        let gain = attenuation.gain(distance);
        let panning = if distance > 1e-9 {
            (rel / distance).dot(self.right()).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        (gain, panning)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_gain_endpoints_and_midpoint() {
        let a = Attenuation::linear(1.0, 11.0);
        assert_eq!(a.gain(0.0), 1.0);
        assert_eq!(a.gain(1.0), 1.0); // at min → full
        assert!((a.gain(6.0) - 0.5).abs() < 1e-9); // halfway → 0.5
        assert_eq!(a.gain(11.0), 0.0); // at max → silent
        assert_eq!(a.gain(100.0), 0.0); // beyond max → silent
    }

    #[test]
    fn inverse_gain_falls_as_reciprocal() {
        let a = Attenuation::inverse(1.0, 100.0);
        assert_eq!(a.gain(1.0), 1.0);
        assert!((a.gain(2.0) - 0.5).abs() < 1e-9);
        assert!((a.gain(4.0) - 0.25).abs() < 1e-9);
        assert_eq!(a.gain(100.0), 0.0);
    }

    #[test]
    fn degenerate_attenuation_is_hard_cutoff() {
        let a = Attenuation {
            model: AttenuationModel::Linear,
            min_distance: 5.0,
            max_distance: 5.0,
        };
        assert_eq!(a.gain(5.0), 1.0);
        assert_eq!(a.gain(5.0001), 0.0);
    }

    #[test]
    fn panning_follows_listener_right_axis() {
        // Default listener faces -Z, up +Y → right = (-Z)×(Y) = +X. So an emitter
        // at +X is on the listener's RIGHT (positive panning).
        let l = Listener::default();
        let att = Attenuation::linear(1.0, 100.0);
        let (_, pan_px) = l.resolve(DVec3::new(10.0, 0.0, 0.0), &att);
        assert!(pan_px > 0.9, "emitter at +X should pan right, got {pan_px}");
        let (_, pan_nx) = l.resolve(DVec3::new(-10.0, 0.0, 0.0), &att);
        assert!(pan_nx < -0.9, "emitter at -X should pan left, got {pan_nx}");
        // Directly ahead → centred.
        let (_, pan_ahead) = l.resolve(DVec3::new(0.0, 0.0, -10.0), &att);
        assert!(pan_ahead.abs() < 1e-9, "ahead should be centred");
    }

    #[test]
    fn co_located_emitter_is_centred_full_volume() {
        let l = Listener::default();
        let att = Attenuation::inverse(1.0, 100.0);
        let (gain, pan) = l.resolve(DVec3::ZERO, &att);
        assert_eq!(gain, 1.0);
        assert_eq!(pan, 0.0);
    }
}
