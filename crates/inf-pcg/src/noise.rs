//! Hand-rolled, seedable **value noise + fBm** — the procedural density source.
//!
//! Deterministic and dependency-free: lattice corner values come from the
//! counter-based [`Hash64`](crate::hash::Hash64) so the field is a pure function
//! of `(seed, position)`. Value noise (hash-per-lattice-point, quintic-faded
//! bilinear interpolation) is cheaper and lower-quality than gradient/simplex
//! noise; swapping in a simplex kernel later is contained to [`value_noise_2d`]
//! because [`ValueNoise::sample`] only depends on it through that one call.

use serde::{Deserialize, Serialize};

use crate::hash::Hash64;

/// Quintic fade `6t^5 − 15t^4 + 10t^3` — C² continuous, so fBm has no lattice
/// creases.
#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// The `[0, 1)` value hashed at integer lattice point `(xi, zi)`.
#[inline]
fn lattice(seed: u64, xi: i64, zi: i64) -> f64 {
    Hash64::new(seed).mix_i64(xi).mix_i64(zi).unit()
}

/// A single octave of value noise at `(x, z)`, returned in `[0, 1)`.
pub fn value_noise_2d(seed: u64, x: f64, z: f64) -> f64 {
    let xf0 = x.floor();
    let zf0 = z.floor();
    let tx = x - xf0;
    let tz = z - zf0;
    let xi = xf0 as i64;
    let zi = zf0 as i64;

    let v00 = lattice(seed, xi, zi);
    let v10 = lattice(seed, xi + 1, zi);
    let v01 = lattice(seed, xi, zi + 1);
    let v11 = lattice(seed, xi + 1, zi + 1);

    let u = fade(tx);
    let w = fade(tz);
    lerp(lerp(v00, v10, u), lerp(v01, v11, u), w)
}

/// A fractal (fBm) value-noise field. `sample` returns values normalized to
/// `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ValueNoise {
    /// Seed for the lattice hash (per-octave sub-seeds are derived from it).
    pub seed: u64,
    /// Spatial frequency of the base octave (cycles per world unit).
    pub frequency: f64,
    /// Number of summed octaves (clamped to ≥ 1).
    pub octaves: u32,
    /// Frequency multiplier between successive octaves (typically 2.0).
    pub lacunarity: f64,
    /// Amplitude multiplier between successive octaves (typically 0.5).
    pub gain: f64,
}

impl Default for ValueNoise {
    fn default() -> Self {
        Self {
            seed: 0,
            frequency: 0.01,
            octaves: 4,
            lacunarity: 2.0,
            gain: 0.5,
        }
    }
}

impl ValueNoise {
    /// Sample the fractal field at `(x, z)`, normalized to `[0, 1]`.
    pub fn sample(&self, x: f64, z: f64) -> f64 {
        let octaves = self.octaves.max(1);
        let mut freq = self.frequency;
        let mut amp = 1.0;
        let mut sum = 0.0;
        let mut norm = 0.0;
        for o in 0..octaves {
            // Per-octave sub-seed so octaves are decorrelated, not scaled copies.
            let sub_seed = Hash64::new(self.seed).mix_u64(o as u64).finish();
            sum += amp * value_noise_2d(sub_seed, x * freq, z * freq);
            norm += amp;
            amp *= self.gain;
            freq *= self.lacunarity;
        }
        if norm > 0.0 {
            (sum / norm).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic() {
        let n = ValueNoise::default();
        assert_eq!(n.sample(12.3, -4.5), n.sample(12.3, -4.5));
    }

    #[test]
    fn stays_in_unit_range() {
        let n = ValueNoise {
            seed: 7,
            frequency: 0.05,
            octaves: 6,
            lacunarity: 2.0,
            gain: 0.5,
        };
        for i in 0..2000 {
            let x = i as f64 * 0.37;
            let z = i as f64 * -0.19;
            let v = n.sample(x, z);
            assert!((0.0..=1.0).contains(&v), "v={v}");
        }
    }

    #[test]
    fn continuous_across_lattice_boundary() {
        let n = ValueNoise {
            octaves: 1,
            frequency: 1.0,
            ..Default::default()
        };
        // Straddling an integer lattice line, the field should not jump.
        let a = n.sample(2.0 - 1e-6, 0.5);
        let b = n.sample(2.0 + 1e-6, 0.5);
        assert!((a - b).abs() < 1e-4, "discontinuity: {a} vs {b}");
    }

    #[test]
    fn seed_changes_field() {
        let a = ValueNoise {
            seed: 1,
            ..Default::default()
        };
        let b = ValueNoise {
            seed: 2,
            ..Default::default()
        };
        // Somewhere in a modest window the two fields must differ.
        let differs = (0..50).any(|i| {
            let x = i as f64 * 3.0;
            (a.sample(x, x) - b.sample(x, x)).abs() > 1e-6
        });
        assert!(differs);
    }
}
