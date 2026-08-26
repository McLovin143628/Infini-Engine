//! A **void-and-cluster** blue-noise tile (wave SKY2), generated here rather
//! than shipped as an asset or pulled from a crate.
//!
//! # What it is for
//!
//! A ray-march that starts every pixel's first sample at the same fraction of a
//! step produces *banding*: the sampling lattice is coherent across the screen, so
//! the integration error is coherent too and reads as concentric shells. The fix
//! is to offset each pixel's first sample by a per-pixel fraction — and *which*
//! fraction matters. White noise trades banding for the same total error scattered
//! at every spatial frequency, including the low ones the eye is most sensitive
//! to. Blue noise has almost no energy at low frequencies: the same error, moved
//! where it reads as fine grain instead of as blotches.
//!
//! # Why it is generated and not baked
//!
//! Ulichney's void-and-cluster is about a hundred lines and runs once. A committed
//! 16 KB binary in `tests/` would be one more thing whose provenance nobody can
//! check, and a dependency would be a licence review for a hundred lines of
//! arithmetic. The generator is also the *documentation*: what makes the tile blue
//! is legible in [`generate`], and [`tests`] measures the property rather than
//! trusting the label.
//!
//! # Determinism (the house law)
//!
//! Nothing here calls a transcendental. The seed pattern comes from
//! [`crate::clouds::cloud_hash`] — the same pure-integer avalanche the cloud field
//! uses — and the energy kernel is a **rational** falloff evaluated with multiplies
//! and one divide, both of which IEEE-754 specifies exactly. Rust does not contract
//! `a * b + c` into an FMA, so the energy sums are evaluated in written order. The
//! tile is therefore bit-identical on every platform, which it has to be: it
//! reaches a golden PNG.
//!
//! # Progressive, not merely blue
//!
//! The output is a **rank** per texel, normalized to `[0, 1)`. That is stronger
//! than a blue-noise *image*: every prefix of the ranking — the lowest 64 texels,
//! the lowest 256, the lowest 1024 — is itself well-distributed, which is the
//! property a thresholded dither needs and the one
//! `every_prefix_of_the_ranking_is_well_spread` measures.

use std::sync::OnceLock;

/// Edge of the (square, toroidal) tile, in texels.
///
/// 64 is the usual size and the reason is the tile's own spectrum: the pattern
/// repeats every `BLUE_NOISE_RES` pixels, which puts a spike at that frequency,
/// and 64 px is fine enough that the spike is well above the band the eye
/// integrates over at a normal viewing distance.
pub const BLUE_NOISE_RES: u32 = 64;

/// Texel count of one tile.
pub const BLUE_NOISE_TEXELS: usize = (BLUE_NOISE_RES * BLUE_NOISE_RES) as usize;

/// Radius, in texels, past which the energy kernel is treated as zero.
///
/// Not an approximation worth worrying about: at `r = 6` the weight below is
/// 1/730 of its peak, and the whole point of the kernel is a *ranking*, which a
/// uniform 0.1 % tail cannot reorder.
const KERNEL_RADIUS: i32 = 6;

/// The kernel's width term, `2σ²` for the conventional `σ = 1.5`.
const KERNEL_SIGMA_TERM: f32 = 4.5;

/// Fraction of the tile set in the initial prototype pattern. Ulichney's
/// recommendation; the relaxation below removes any dependence on the exact
/// number, and the ranking that comes out is what is measured.
const PROTOTYPE_FRACTION: usize = 10;

/// Seed of the prototype pattern. Any value gives a blue tile; this one is the
/// tile the goldens were blessed against, so it is a **committed constant**.
const PROTOTYPE_SEED: u32 = 0x5b17_2e01;

/// The process-wide tile: `BLUE_NOISE_TEXELS` values in `[0, 1)`, row-major.
///
/// Built on first use (~10 ms) and never again. Callers upload it to an
/// `R32Float` texture; the value at a texel is its **rank** divided by the texel
/// count, so thresholding at `t` selects exactly the best-spread `t` fraction of
/// the tile.
pub fn blue_noise_tile() -> &'static [f32] {
    static TILE: OnceLock<Vec<f32>> = OnceLock::new();
    TILE.get_or_init(|| {
        let ranks = generate(PROTOTYPE_SEED);
        ranks
            .iter()
            .map(|&r| r as f32 / BLUE_NOISE_TEXELS as f32)
            .collect()
    })
}

/// Toroidal energy kernel: `(dx, dy)` offset pairs inside [`KERNEL_RADIUS`] and
/// the weight each contributes.
///
/// `1 / (1 + d²/2σ²)³` rather than `exp(-d²/2σ²)`. It is monotone decreasing in
/// `d²` — which is the only property void-and-cluster needs of it — and it is
/// four multiplies and a divide, so it is bit-identical everywhere, which
/// `exp` is not.
fn kernel() -> Vec<(i32, i32, f32)> {
    let mut out = Vec::new();
    for dy in -KERNEL_RADIUS..=KERNEL_RADIUS {
        for dx in -KERNEL_RADIUS..=KERNEL_RADIUS {
            if dx == 0 && dy == 0 {
                continue;
            }
            let d2 = (dx * dx + dy * dy) as f32;
            if d2 > (KERNEL_RADIUS * KERNEL_RADIUS) as f32 {
                continue;
            }
            let q = 1.0 / (1.0 + d2 / KERNEL_SIGMA_TERM);
            out.push((dx, dy, q * q * q));
        }
    }
    out
}

/// The energy field over a binary pattern: `energy[i]` is the sum of the kernel
/// over every *set* texel other than `i` itself.
struct Field {
    energy: Vec<f32>,
    kernel: Vec<(i32, i32, f32)>,
}

impl Field {
    fn new(kernel: Vec<(i32, i32, f32)>) -> Self {
        Self {
            energy: vec![0.0; BLUE_NOISE_TEXELS],
            kernel,
        }
    }

    /// Add (`sign = 1.0`) or remove (`sign = -1.0`) one texel's contribution.
    fn stamp(&mut self, index: usize, sign: f32) {
        let res = BLUE_NOISE_RES as i32;
        let x = (index % BLUE_NOISE_RES as usize) as i32;
        let y = (index / BLUE_NOISE_RES as usize) as i32;
        for &(dx, dy, w) in &self.kernel {
            let nx = (x + dx).rem_euclid(res);
            let ny = (y + dy).rem_euclid(res);
            let n = (ny * res + nx) as usize;
            self.energy[n] += sign * w;
        }
    }

    /// The **tightest cluster**: the set texel with the most set neighbours.
    /// Ties go to the lowest index, so the walk is a pure function of the input.
    fn tightest_cluster(&self, pattern: &[bool]) -> usize {
        let mut best = usize::MAX;
        let mut best_e = f32::NEG_INFINITY;
        for (i, &on) in pattern.iter().enumerate() {
            if on && self.energy[i] > best_e {
                best_e = self.energy[i];
                best = i;
            }
        }
        best
    }

    /// The **largest void**: the clear texel with the fewest set neighbours.
    fn largest_void(&self, pattern: &[bool]) -> usize {
        let mut best = usize::MAX;
        let mut best_e = f32::INFINITY;
        for (i, &on) in pattern.iter().enumerate() {
            if !on && self.energy[i] < best_e {
                best_e = self.energy[i];
                best = i;
            }
        }
        best
    }

    /// The clear texel with the **most clear neighbours** — the tightest cluster
    /// of the inverse pattern, which is where Ulichney's third phase puts the next
    /// sample once the majority of the tile is already set.
    fn tightest_void_cluster(&self, pattern: &[bool]) -> usize {
        let mut best = usize::MAX;
        let mut best_e = f32::NEG_INFINITY;
        for (i, &on) in pattern.iter().enumerate() {
            if !on && self.energy[i] > best_e {
                best_e = self.energy[i];
                best = i;
            }
        }
        best
    }
}

/// The ranking: `out[i]` is texel `i`'s position in a well-spread ordering of the
/// whole tile, `0 ..= BLUE_NOISE_TEXELS - 1`.
///
/// Ulichney's three phases, in order:
///
/// 1. **Relax** an arbitrary prototype until moving the tightest cluster into the
///    largest void is a no-op. What comes out is a well-spread minority pattern
///    whose *content* no longer depends on the seed's clumps, only on the seed.
/// 2. **Rank downward** from the prototype: repeatedly remove the tightest
///    cluster, giving it the highest remaining rank below the prototype's size.
///    (A texel removed early was in the densest company, so it is the one a
///    sparse prefix should omit.)
/// 3. **Rank upward** to half, filling the largest void each time — and then past
///    half, where "void" stops meaning anything and the next set texel goes into
///    the tightest cluster of the *clear* texels instead.
pub fn generate(seed: u32) -> Vec<u32> {
    let ones = BLUE_NOISE_TEXELS / PROTOTYPE_FRACTION;
    let mut field = Field::new(kernel());
    let mut pattern = vec![false; BLUE_NOISE_TEXELS];

    // ── the prototype: `ones` distinct texels from the integer hash ──
    let mut placed = 0usize;
    let mut draw = 0u32;
    while placed < ones {
        let h = crate::clouds::cloud_hash(draw, 0, 0, seed);
        draw = draw.wrapping_add(1);
        let i = (h as usize) % BLUE_NOISE_TEXELS;
        if !pattern[i] {
            pattern[i] = true;
            field.stamp(i, 1.0);
            placed += 1;
        }
    }

    // ── phase 0: relax ──
    // Bounded rather than "until stable": the fixed point is reached in a few
    // hundred swaps at this size, and a cap means a pathological seed cannot hang
    // a renderer's first frame. The rank phases below do not require the fixed
    // point, only a reasonable starting pattern.
    for _ in 0..BLUE_NOISE_TEXELS {
        let tight = field.tightest_cluster(&pattern);
        pattern[tight] = false;
        field.stamp(tight, -1.0);
        let void = field.largest_void(&pattern);
        pattern[void] = true;
        field.stamp(void, 1.0);
        if void == tight {
            break;
        }
    }

    let prototype = pattern.clone();
    let mut rank = vec![0u32; BLUE_NOISE_TEXELS];

    // ── phase 1: rank the prototype downward ──
    for r in (0..ones).rev() {
        let tight = field.tightest_cluster(&pattern);
        pattern[tight] = false;
        field.stamp(tight, -1.0);
        rank[tight] = r as u32;
    }

    // ── phase 2: back to the prototype, then fill voids up to half ──
    pattern.clone_from(&prototype);
    field.energy.iter_mut().for_each(|e| *e = 0.0);
    for (i, &on) in prototype.iter().enumerate() {
        if on {
            field.stamp(i, 1.0);
        }
    }
    let half = BLUE_NOISE_TEXELS / 2;
    for r in ones..half {
        let void = field.largest_void(&pattern);
        pattern[void] = true;
        field.stamp(void, 1.0);
        rank[void] = r as u32;
    }

    // ── phase 3: past half, rank against the energy of the CLEAR texels ──
    // The field is rebuilt over the zeros, because a "largest void" in a pattern
    // that is more than half set is meaningless — what is scarce now is empty
    // space, and the next sample belongs where empty space is most clustered.
    field.energy.iter_mut().for_each(|e| *e = 0.0);
    for (i, &on) in pattern.iter().enumerate() {
        if !on {
            field.stamp(i, 1.0);
        }
    }
    for r in half..BLUE_NOISE_TEXELS {
        let i = field.tightest_void_cluster(&pattern);
        pattern[i] = true;
        // `i` is no longer a clear texel, so it stops contributing to the
        // clear-texel energy.
        field.stamp(i, -1.0);
        rank[i] = r as u32;
    }

    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Toroidal squared distance between two texel indices.
    fn dist2(a: usize, b: usize) -> i32 {
        let res = BLUE_NOISE_RES as i32;
        let (ax, ay) = ((a % 64) as i32, (a / 64) as i32);
        let (bx, by) = ((b % 64) as i32, (b / 64) as i32);
        let dx = (ax - bx).abs().min(res - (ax - bx).abs());
        let dy = (ay - by).abs().min(res - (ay - by).abs());
        dx * dx + dy * dy
    }

    /// Mean squared distance from each point of `set` to its nearest other
    /// point — the direct measure of "well spread", with no transform in the way.
    fn mean_nearest(set: &[usize]) -> f64 {
        let mut sum = 0.0f64;
        for (k, &a) in set.iter().enumerate() {
            let mut best = i32::MAX;
            for (j, &b) in set.iter().enumerate() {
                if j != k {
                    best = best.min(dist2(a, b));
                }
            }
            sum += f64::from(best);
        }
        sum / set.len() as f64
    }

    /// The ranking is a **permutation**: every rank appears exactly once, so
    /// thresholding it at any level selects exactly that fraction of the tile.
    /// A generator that silently overwrote a rank would still look like noise.
    #[test]
    fn the_ranking_is_a_permutation() {
        let rank = generate(PROTOTYPE_SEED);
        assert_eq!(rank.len(), BLUE_NOISE_TEXELS);
        let mut seen = vec![false; BLUE_NOISE_TEXELS];
        for &r in &rank {
            let r = r as usize;
            assert!(r < BLUE_NOISE_TEXELS, "rank {r} out of range");
            assert!(!seen[r], "rank {r} appears twice");
            seen[r] = true;
        }
        // ...and the published tile is that permutation, scaled into [0, 1).
        let tile = blue_noise_tile();
        assert_eq!(tile.len(), BLUE_NOISE_TEXELS);
        for (i, &v) in tile.iter().enumerate() {
            assert!((0.0..1.0).contains(&v), "tile[{i}] = {v}");
            assert_eq!(v, rank[i] as f32 / BLUE_NOISE_TEXELS as f32);
        }
    }

    /// The tile is a pure function of its seed — the determinism law, asserted on
    /// the generator rather than on the `OnceLock`, which would pass trivially.
    #[test]
    fn the_tile_is_a_pure_function_of_the_seed() {
        assert_eq!(generate(PROTOTYPE_SEED), generate(PROTOTYPE_SEED));
        assert_ne!(generate(PROTOTYPE_SEED), generate(PROTOTYPE_SEED ^ 1));
    }

    /// **The property the name claims.** Every prefix of the ranking is spread
    /// out: the lowest-ranked N texels sit further from each other than N texels
    /// picked by a white-noise hash do, at every N a jittered march would use.
    ///
    /// Measured as the mean nearest-neighbour distance, which is what "no
    /// low-frequency energy" means in the spatial domain — and unlike a DFT it
    /// needs no trigonometry, so the test is as portable as the tile.
    ///
    /// The white-noise control is drawn from the *same* hash the prototype uses,
    /// so what is compared is the void-and-cluster ordering and nothing else.
    #[test]
    fn every_prefix_of_the_ranking_is_well_spread() {
        let rank = generate(PROTOTYPE_SEED);
        for &n in &[64usize, 256, 1024] {
            let blue: Vec<usize> = (0..BLUE_NOISE_TEXELS)
                .filter(|&i| (rank[i] as usize) < n)
                .collect();
            assert_eq!(blue.len(), n);

            // White-noise control: the first `n` distinct texels the hash names.
            let mut white = Vec::with_capacity(n);
            let mut seen = vec![false; BLUE_NOISE_TEXELS];
            let mut draw = 0u32;
            while white.len() < n {
                let h = crate::clouds::cloud_hash(draw, 7, 0, PROTOTYPE_SEED);
                draw = draw.wrapping_add(1);
                let i = (h as usize) % BLUE_NOISE_TEXELS;
                if !seen[i] {
                    seen[i] = true;
                    white.push(i);
                }
            }

            let b = mean_nearest(&blue);
            let w = mean_nearest(&white);
            eprintln!("prefix {n}: blue mean nearest d^2 {b:.2} vs white {w:.2}");
            assert!(
                b > w * 1.5,
                "prefix {n} is not blue: mean nearest d^2 {b:.2} against white noise's {w:.2}"
            );
        }
    }

    /// A blue tile has no *clumps* either: no texel in a sparse prefix should sit
    /// on top of another. The bound is the strong half of the claim above — a
    /// distribution can have a good mean and still stack two samples on adjacent
    /// texels, which is exactly what produces a visible speckle in a jittered
    /// march.
    #[test]
    fn a_sparse_prefix_has_no_adjacent_pair() {
        let rank = generate(PROTOTYPE_SEED);
        let n = 256usize;
        let set: Vec<usize> = (0..BLUE_NOISE_TEXELS)
            .filter(|&i| (rank[i] as usize) < n)
            .collect();
        let mut worst = i32::MAX;
        for (k, &a) in set.iter().enumerate() {
            for &b in set.iter().skip(k + 1) {
                worst = worst.min(dist2(a, b));
            }
        }
        // 256 points on a 64x64 torus average one per 16 texels, i.e. a spacing
        // of 4; anything closer than 2 texels apart is a clump.
        assert!(
            worst >= 4,
            "two of the 256 lowest ranks are {worst} squared texels apart"
        );
    }
}
