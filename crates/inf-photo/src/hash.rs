//! Deterministic **counter-based** hashing — the crate's only source of
//! pseudo-randomness.
//!
//! RANSAC needs random minimal samples. A stateful RNG would make the sample a
//! function of *how many draws happened before it*, which is a function of
//! evaluation order, which on a job pool is a function of the worker count — the
//! exact drift every gate in this repo exists to prevent. So there is no
//! stateful RNG here. Every draw is a **pure function** of the integers folded
//! into it: `(seed, stage, iteration, slot)`. Iteration 900 of the essential
//! RANSAC draws the same eight indices whether it ran first, last, or on a
//! different machine.
//!
//! The mixer is the SplitMix64 finalizer. That is a *specification* (three
//! shift-xor-multiply rounds with published constants), not an implementation
//! that could drift, which is why this is a thirty-line copy of
//! `inf_pcg::hash::Hash64` rather than a dependency — a photogrammetry crate has
//! no business depending on the procedural-generation runtime. The constants are
//! asserted equal to the specification (and to the published SplitMix64 test
//! vector) by `splitmix_constants_are_the_published_ones`, and `inf-mesh`
//! carries the same copy under the same rule.

/// The golden-ratio odd constant used to decorrelate successive mixed words.
const GOLDEN: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64 finalizer: a full-avalanche 64 -> 64 bit mix.
///
/// `const` because [`crate::features`] builds its descriptor sampling pattern
/// with it **at compile time** — the pattern is a constant table, so there is no
/// runtime randomness in the descriptor at all.
#[inline]
pub const fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// A fluent counter-based hasher. Build it from a seed, fold in the integer
/// coordinates of the thing you are drawing for, then read a `u64` or a uniform
/// `f64`. It is `Copy`, so one base state spawns many decorrelated draws via
/// distinct salts — the pattern the RANSAC samplers use per slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hash64(u64);

impl Hash64 {
    /// Start a hash stream from `seed`.
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Hash64(mix64(seed ^ GOLDEN))
    }

    /// Fold an unsigned word into the state.
    #[inline]
    pub const fn mix_u64(self, v: u64) -> Self {
        Hash64(mix64(self.0 ^ v.wrapping_mul(GOLDEN)))
    }

    /// Fold a `usize` into the state (widened, so a 32-bit host agrees with a
    /// 64-bit one).
    #[inline]
    pub const fn mix_usize(self, v: usize) -> Self {
        self.mix_u64(v as u64)
    }

    /// Fold a signed integer coordinate into the state (two's-complement
    /// reinterpretation, so `-1` and `u64::MAX` fold identically — which is
    /// fine, because a hasher only needs to be a function).
    #[inline]
    pub const fn mix_i64(self, v: i64) -> Self {
        self.mix_u64(v as u64)
    }

    /// The raw 64-bit hash.
    #[inline]
    pub const fn finish(self) -> u64 {
        self.0
    }

    /// A uniform `f64` in `[0, 1)` (53-bit mantissa precision).
    #[inline]
    pub fn unit(self) -> f64 {
        (self.0 >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// A uniform index in `0..n`. Panics if `n == 0`.
    ///
    /// This is a modulo reduction and therefore carries the usual modulo bias.
    /// For `n` in the hundreds against a 64-bit hash the bias is around
    /// `n / 2^64` and is irrelevant to a RANSAC sampler; it is written down
    /// rather than papered over.
    #[inline]
    pub fn index(self, n: usize) -> usize {
        assert!(n > 0, "Hash64::index needs a non-empty range");
        (self.0 % n as u64) as usize
    }
}

/// Draw `k` **distinct** indices from `0..n`, deterministically from `base`.
///
/// A partial Fisher–Yates over a scratch permutation: at step `i` it swaps slot
/// `i` with a hashed slot in `i..n`, so the result is a uniform `k`-subset with
/// no rejection loop (a rejection loop's *length* would depend on the draws,
/// which is fine for determinism but makes the cost data-dependent for no gain).
///
/// Returns `None` when `n < k`.
pub fn distinct_indices(
    base: Hash64,
    n: usize,
    k: usize,
    scratch: &mut Vec<u32>,
) -> Option<Vec<u32>> {
    if n < k {
        return None;
    }
    scratch.clear();
    scratch.extend(0..n as u32);
    let mut out = Vec::with_capacity(k);
    for i in 0..k {
        let span = n - i;
        let j = i + base.mix_usize(i).index(span);
        scratch.swap(i, j);
        out.push(scratch[i]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_constants_are_the_published_ones() {
        // The SplitMix64 finalizer, spelled out independently of `mix64`.
        fn reference(mut z: u64) -> u64 {
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        for x in [0u64, 1, 42, u64::MAX, 0x9e37_79b9_7f4a_7c15] {
            assert_eq!(mix64(x), reference(x), "mixer drifted from the spec at {x}");
        }
        // The published SplitMix64 test vector: from a zero state the generator
        // advances by the golden constant and emits this. `Hash64::new(0)` is
        // exactly `mix64(0 ^ GOLDEN)`, so it must reproduce it — a hard pin
        // against an external number, so a "harmless" refactor of the mixer is a
        // red test rather than a silently different reconstruction.
        assert_eq!(Hash64::new(0).finish(), 0xe220_a839_7b1d_cdaf);
    }

    #[test]
    fn is_pure_and_order_independent() {
        let a = Hash64::new(7).mix_usize(3).mix_u64(11).finish();
        let b = Hash64::new(7).mix_usize(3).mix_u64(11).finish();
        assert_eq!(a, b);
        // Folding is order-sensitive by design (it is a chain, not a set hash).
        let c = Hash64::new(7).mix_u64(11).mix_usize(3).finish();
        assert_ne!(a, c);
    }

    #[test]
    fn unit_stays_in_range() {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for i in 0..4096u64 {
            let u = Hash64::new(0xdead_beef).mix_u64(i).unit();
            assert!((0.0..1.0).contains(&u), "unit() escaped [0,1): {u}");
            lo = lo.min(u);
            hi = hi.max(u);
        }
        // Not a statistical test — just proof it is not a constant.
        assert!(lo < 0.01, "lower tail never sampled: {lo}");
        assert!(hi > 0.99, "upper tail never sampled: {hi}");
    }

    #[test]
    fn distinct_indices_are_distinct_and_reproducible() {
        let mut scratch = Vec::new();
        for iter in 0..256usize {
            let base = Hash64::new(1234).mix_usize(iter);
            let a = distinct_indices(base, 40, 8, &mut scratch).expect("40 >= 8");
            let b = distinct_indices(base, 40, 8, &mut scratch).expect("40 >= 8");
            assert_eq!(a, b, "the same base drew a different sample");
            let mut sorted = a.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 8, "sample {a:?} repeated an index");
            assert!(a.iter().all(|&i| (i as usize) < 40));
        }
        assert!(distinct_indices(Hash64::new(1), 7, 8, &mut scratch).is_none());
    }

    #[test]
    fn distinct_indices_cover_the_range() {
        // A sampler that always returns the first k indices is "distinct" too.
        let mut scratch = Vec::new();
        let mut seen = vec![false; 40];
        for iter in 0..512usize {
            let s = distinct_indices(Hash64::new(99).mix_usize(iter), 40, 8, &mut scratch).unwrap();
            for i in s {
                seen[i as usize] = true;
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "some indices are unreachable: {seen:?}"
        );
    }
}
