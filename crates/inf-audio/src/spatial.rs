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
///
/// All three models share the `[min_distance, max_distance]` clamp envelope
/// (full volume at/within `min`, silent at/beyond `max`); they differ only in
/// the curve between. Mirrors [`inf_ecs`](../../inf_ecs)'s `DistanceModel` so the
/// sim can translate a component's model 1:1 (P12.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttenuationModel {
    /// Linear ramp from full volume at `min_distance` to silence at
    /// `max_distance` — predictable, the usual pick for gameplay cues.
    Linear,
    /// Inverse-distance falloff (`min_distance / distance`), clamped to `[0, 1]`
    /// and to silence past `max_distance` — closer to physical 1/r rolloff.
    Inverse,
    /// Exponential falloff: `(min_distance / distance)^rolloff` (P12.3), clamped
    /// to `[0, 1]` and to silence past `max_distance`. `rolloff = 1` matches
    /// [`Inverse`](Self::Inverse); larger values fall off faster. The rolloff
    /// exponent lives on [`Attenuation::rolloff`].
    Exponential,
}

/// The distance-attenuation curve for a spatial emitter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Attenuation {
    pub model: AttenuationModel,
    /// At or within this distance the emitter is at full volume (gain 1.0).
    pub min_distance: f64,
    /// At or beyond this distance the emitter is silent (gain 0.0).
    pub max_distance: f64,
    /// Falloff exponent for [`AttenuationModel::Exponential`]; ignored by the
    /// other models. `1.0` = plain inverse; higher = steeper. Clamped `>= 0`.
    pub rolloff: f64,
}

impl Default for Attenuation {
    fn default() -> Self {
        Self {
            model: AttenuationModel::Inverse,
            min_distance: 1.0,
            max_distance: 100.0,
            rolloff: 1.0,
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
            rolloff: 1.0,
        }
    }

    /// An inverse-distance curve between `min` and `max` world units.
    pub fn inverse(min_distance: f64, max_distance: f64) -> Self {
        Self {
            model: AttenuationModel::Inverse,
            min_distance,
            max_distance,
            rolloff: 1.0,
        }
    }

    /// An exponential-falloff curve (`(min/d)^rolloff`) between `min` and `max`
    /// world units. `rolloff` is clamped non-negative.
    pub fn exponential(min_distance: f64, max_distance: f64, rolloff: f64) -> Self {
        Self {
            model: AttenuationModel::Exponential,
            min_distance,
            max_distance,
            rolloff: rolloff.max(0.0),
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
            AttenuationModel::Exponential => (self.min_distance / d)
                .powf(self.rolloff.max(0.0))
                .clamp(0.0, 1.0),
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
    fn exponential_gain_matches_inverse_at_rolloff_one_and_steepens() {
        // rolloff = 1 is identical to Inverse.
        let e1 = Attenuation::exponential(1.0, 100.0, 1.0);
        let inv = Attenuation::inverse(1.0, 100.0);
        assert!((e1.gain(2.0) - inv.gain(2.0)).abs() < 1e-12);
        assert!((e1.gain(4.0) - inv.gain(4.0)).abs() < 1e-12);
        // rolloff = 2 falls off faster: at d=2, (1/2)^2 = 0.25 < 0.5.
        let e2 = Attenuation::exponential(1.0, 100.0, 2.0);
        assert!((e2.gain(2.0) - 0.25).abs() < 1e-12);
        assert!(e2.gain(2.0) < e1.gain(2.0));
        // Envelope still applies: full within min, silent past max.
        assert_eq!(e2.gain(0.5), 1.0);
        assert_eq!(e2.gain(100.0), 0.0);
    }

    #[test]
    fn degenerate_attenuation_is_hard_cutoff() {
        let a = Attenuation {
            model: AttenuationModel::Linear,
            min_distance: 5.0,
            max_distance: 5.0,
            rolloff: 1.0,
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
