//! Deterministic **counter-based** hashing — the determinism backbone of the PCG
//! runtime.
//!
//! Sampling and scattering NEVER use a stateful RNG shared across threads.
//! Instead every random draw is a *pure function* of an integer coordinate tuple
//! `(seed, cell_x, cell_z, slot_i, slot_j, salt)`: the same inputs always hash to
//! the same value, independent of evaluation order or thread count. That is what
//! makes `scatter_region` bit-identical across pool sizes.
//!
//! The mixer is a small hand-rolled avalanche hash (the SplitMix64 finalizer
//! constants). It is not a cryptographic hash; it only needs good bit-diffusion
//! and reproducibility. Swapping in xxh3 later is contained to this file.

/// The golden-ratio odd constant used to decorrelate successive mixed words.
const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64 finalizer: a full-avalanche 64→64 bit mix.
#[inline]
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// A fluent counter-based hasher. Build it from a seed, fold in the integer
/// coordinates of the thing you are sampling, then read a `u64` or a uniform
/// `f64`. Cloning it (it is `Copy`) lets one base state spawn many decorrelated
/// draws via distinct salts — the pattern the scatter kernel uses per slot.
#[derive(Clone, Copy, Debug)]
pub struct Hash64(u64);

impl Hash64 {
    /// Start a hash stream from `seed`.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Hash64(mix64(seed ^ GOLDEN))
    }

    /// Fold an unsigned word into the state.
    #[inline]
    pub fn mix_u64(self, v: u64) -> Self {
        Hash64(mix64(self.0 ^ v.wrapping_mul(GOLDEN)))
    }

    /// Fold a signed integer coordinate into the state.
    #[inline]
    pub fn mix_i64(self, v: i64) -> Self {
        self.mix_u64(v as u64)
    }

    /// Fold the bit pattern of an `f64` into the state (used to derive a stable
    /// draw from a world position).
    #[inline]
    pub fn mix_f64(self, v: f64) -> Self {
        self.mix_u64(v.to_bits())
    }

    /// The raw 64-bit hash.
    #[inline]
    pub fn finish(self) -> u64 {
        self.0
    }

    /// A uniform `f64` in `[0, 1)` (53-bit mantissa precision).
    #[inline]
    pub fn unit(self) -> f64 {
        // Top 53 bits → [0, 2^53) → scale into [0, 1).
        (self.0 >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_pure_and_reproducible() {
        let a = Hash64::new(42).mix_i64(-7).mix_i64(3).mix_u64(9).finish();
        let b = Hash64::new(42).mix_i64(-7).mix_i64(3).mix_u64(9).finish();
        assert_eq!(a, b);
    }

    #[test]
    fn distinct_inputs_decorrelate() {
        let base = Hash64::new(1).mix_i64(10).mix_i64(20);
        // Different salts off the same base give different draws.
        assert_ne!(base.mix_u64(0xA1).finish(), base.mix_u64(0xB2).finish());
        // Different coordinates give different draws.
        assert_ne!(
            Hash64::new(1).mix_i64(10).mix_i64(20).finish(),
            Hash64::new(1).mix_i64(10).mix_i64(21).finish()
        );
    }

    #[test]
    fn unit_is_in_range() {
        for i in 0..10_000u64 {
            let u = Hash64::new(i).mix_u64(i.wrapping_mul(7)).unit();
            assert!((0.0..1.0).contains(&u), "u={u} out of range");
        }
    }

    #[test]
    fn unit_is_roughly_uniform() {
        // A crude mean check: 100k draws should average near 0.5.
        let n = 100_000u64;
        let mut sum = 0.0;
        for i in 0..n {
            sum += Hash64::new(0xdead).mix_u64(i).unit();
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean={mean}");
    }
}
