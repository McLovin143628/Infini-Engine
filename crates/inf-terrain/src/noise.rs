//! A small, self-contained **value-noise + fBm** field for the Noise sculpt brush.
//!
//! ## Why this is duplicated (and not a dep on `inf-pcg`)
//!
//! `inf-pcg` already owns the engine's value-noise + counter-based hash
//! ([`inf_pcg::noise`], [`inf_pcg::hash::Hash64`]). We deliberately do **not**
//! depend on it here: the ring order is `inf-pcg → inf-terrain` (PCG samples the
//! terrain's [`crate::HeightSource`]), so a reverse edge would tangle the rings
//! into a cycle. Instead this module **mirrors** the ~60-line noise core (same
//! SplitMix64 finalizer, same quintic fade, same lattice-value construction) so
//! both stay seeded-deterministic and produce the *same shape* of field. The
//! duplication is a conscious ring-hygiene trade: the alternative (hoisting the
//! noise into a shared Ring-0 leaf crate) is deferred until a third consumer
//! needs it — see the P10.2a report. Keep the two implementations in sync if the
//! kernel ever changes.
//!
//! The brush uses a **signed** fBm ([`fbm_signed`], roughly `[-1, 1]`) so a Noise
//! dab adds bumps *and* pits, and it is sampled in **world space** — the value at
//! a world position is a pure function of `(seed, frequency, x, z)`, independent
//! of the stroke path — which is what makes the Noise brush order-independent and
//! reproducible.

/// The golden-ratio odd constant used to decorrelate successive mixed words
/// (mirrors `inf_pcg::hash`).
const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64 finalizer: a full-avalanche 64→64 bit mix (mirrors `inf_pcg::hash`).
#[inline]
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// A minimal fluent counter-based hasher (mirrors `inf_pcg::hash::Hash64`).
#[derive(Clone, Copy, Debug)]
struct Hash64(u64);

impl Hash64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Hash64(mix64(seed ^ GOLDEN))
    }

    #[inline]
    fn mix_u64(self, v: u64) -> Self {
        Hash64(mix64(self.0 ^ v.wrapping_mul(GOLDEN)))
    }

    #[inline]
    fn mix_i64(self, v: i64) -> Self {
        self.mix_u64(v as u64)
    }

    #[inline]
    fn finish(self) -> u64 {
        self.0
    }

    /// A uniform `f64` in `[0, 1)` (top 53 bits).
    #[inline]
    fn unit(self) -> f64 {
        (self.0 >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

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
fn value_noise_2d(seed: u64, x: f64, z: f64) -> f64 {
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

/// **Signed** fractal (fBm) value noise at world `(x, z)`, in roughly `[-1, 1]`.
///
/// `frequency` is cycles per world unit of the base octave; each successive
/// octave doubles the frequency (lacunarity 2) and halves the amplitude (gain
/// 0.5). Per-octave sub-seeds decorrelate the octaves (they are not scaled copies
/// of one field). The result is normalized by the summed amplitude, then remapped
/// `[0, 1] → [-1, 1]`. Pure function of the inputs — deterministic in world space.
pub fn fbm_signed(seed: u64, frequency: f64, octaves: u32, x: f64, z: f64) -> f64 {
    let octaves = octaves.max(1);
    let mut freq = frequency;
    let mut amp = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for o in 0..octaves {
        let sub_seed = Hash64::new(seed).mix_u64(o as u64).finish();
        sum += amp * value_noise_2d(sub_seed, x * freq, z * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    let unit = if norm > 0.0 { sum / norm } else { 0.0 };
    unit.clamp(0.0, 1.0) * 2.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_in_world_space() {
        let a = fbm_signed(7, 0.05, 4, 12.3, -4.5);
        let b = fbm_signed(7, 0.05, 4, 12.3, -4.5);
        assert_eq!(a, b);
    }

    #[test]
    fn stays_in_signed_unit_range() {
        for i in 0..2000 {
            let x = i as f64 * 0.37;
            let z = i as f64 * -0.19;
            let v = fbm_signed(3, 0.05, 6, x, z);
            assert!((-1.0..=1.0).contains(&v), "v={v}");
        }
    }

    #[test]
    fn continuous_across_lattice_boundary() {
        // Straddling an integer lattice line, a single octave must not jump.
        let a = fbm_signed(0, 1.0, 1, 2.0 - 1e-6, 0.5);
        let b = fbm_signed(0, 1.0, 1, 2.0 + 1e-6, 0.5);
        assert!((a - b).abs() < 1e-4, "discontinuity: {a} vs {b}");
    }

    #[test]
    fn seed_changes_field() {
        let differs = (0..50).any(|i| {
            let x = i as f64 * 3.0;
            (fbm_signed(1, 0.01, 4, x, x) - fbm_signed(2, 0.01, 4, x, x)).abs() > 1e-6
        });
        assert!(differs);
    }
}
