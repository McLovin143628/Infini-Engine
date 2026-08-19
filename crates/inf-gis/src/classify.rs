//! Classifying continuous values into a small number of classes (ported from
//! GeoCanvas).
//!
//! # Provenance
//!
//! Adapted from `geocanvas-core/src/project/classify.rs` (equal-interval,
//! quantile, Fisher-Jenks natural breaks). The DP is the classic geostatistics
//! formulation and crossed essentially intact; the sampling cap, the degenerate
//! collapses and the monotonicity guard are the parts worth having and are
//! re-documented here for what they do in *this* engine.
//!
//! # What it is for here
//!
//! **Land cover → biome ids.** A published land-cover or zoning raster carries
//! anywhere from a handful to thousands of distinct values; this engine's biomes
//! are a `u8` per terrain sample with a per-biome PCG graph behind each id. Going
//! from the first to the second needs a rule for "which of my ≤255 biomes does
//! this source value mean", and equal-interval is usually the wrong one: real
//! land-cover distributions are lumpy, so equal-width classes put nine tenths of
//! the map in one biome. Natural breaks finds the gaps that are actually in the
//! data.
//!
//! # Determinism
//!
//! Every function here is a pure function of its input slice, sorted with
//! `f64::total_cmp` (a total order, so NaN cannot make the sort
//! implementation-defined) and sampled on a fixed stride. Two runs on the same
//! data produce the same breaks on any machine — which matters, because a biome
//! id raster derived from this is cooked and shipped.

/// How to place class boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ClassifyMethod {
    /// `k` equal-width classes across `[min, max]`. Fast, and usually wrong for
    /// real geographic data — see the module docs.
    EqualInterval,
    /// Equal-count classes: every class holds the same number of samples.
    Quantile,
    /// **Fisher-Jenks natural breaks**: minimise total within-class variance.
    /// The default, because it is the one that finds the gaps in the data.
    #[default]
    NaturalBreaks,
}

/// Above this many values, Jenks runs on a uniform-stride sample.
///
/// The DP is `O(n²·k)`; at 6 000 values that is already ~10⁸ operations for
/// `k = 8`. A land-cover raster has millions of samples, so an unsampled Jenks
/// would take hours. The stride preserves the shape of the distribution, which
/// is all the breaks depend on.
pub const JENKS_SAMPLE_CAP: usize = 6_000;

/// The largest number of classes worth asking for. Biome ids are a `u8` and the
/// engine's own ceiling is 255 biomes; 64 is well past anything an author will
/// hand-tune.
pub const MAX_CLASSES: usize = 64;

/// Class boundaries for `values`: `k + 1` fenceposts, ascending, ends pinned to
/// the data's own min and max.
///
/// Non-finite values are dropped (a raster's no-data is not a class). Returns
/// `[0.0, 0.0]` for empty input — a single degenerate class, which
/// [`class_of`] then maps everything into.
pub fn classify_breaks(values: &[f64], method: ClassifyMethod, classes: usize) -> Vec<f64> {
    let mut nums: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if nums.is_empty() {
        return vec![0.0, 0.0];
    }
    nums.sort_by(f64::total_cmp);
    // Indexed rather than `.last().expect(..)`: the emptiness was already ruled
    // out four lines up, and the only `expect` in this crate's production code
    // sitting on a fact a reader has to go and re-check is a worse trade than
    // one more index.
    let min = nums[0];
    let max = nums[nums.len() - 1];

    // Never ask for more classes than the data can distinguish: `k` classes need
    // `k` distinct values, and asking for more produces duplicate fenceposts that
    // read as empty classes.
    let distinct = {
        let mut d = 1usize;
        for w in nums.windows(2) {
            if w[0] != w[1] {
                d += 1;
            }
        }
        d
    };
    let k = classes.clamp(1, MAX_CLASSES).min(distinct);
    if k <= 1 || min == max {
        return vec![min, max];
    }

    match method {
        ClassifyMethod::EqualInterval => {
            let step = (max - min) / k as f64;
            let mut b: Vec<f64> = (0..=k).map(|i| min + step * i as f64).collect();
            // Pin the ends exactly — float accumulation nudges the last fencepost,
            // and a fencepost below the data's own max drops the top sample out of
            // every class.
            b[0] = min;
            b[k] = max;
            b
        }
        ClassifyMethod::Quantile => {
            let n = nums.len();
            let mut b = Vec::with_capacity(k + 1);
            b.push(min);
            for i in 1..k {
                b.push(nums[(i * n / k).min(n - 1)]);
            }
            b.push(max);
            b
        }
        ClassifyMethod::NaturalBreaks => jenks_breaks(&jenks_sample(&nums), k),
    }
}

/// Uniform-stride sample of already-sorted values down to [`JENKS_SAMPLE_CAP`],
/// keeping the first and last so the range stays exact.
fn jenks_sample(sorted: &[f64]) -> Vec<f64> {
    let n = sorted.len();
    if n <= JENKS_SAMPLE_CAP {
        return sorted.to_vec();
    }
    let step = (n - 1) as f64 / (JENKS_SAMPLE_CAP - 1) as f64;
    let mut out: Vec<f64> = (0..JENKS_SAMPLE_CAP)
        .map(|i| sorted[((i as f64 * step).round() as usize).min(n - 1)])
        .collect();
    // Rounding can repeat an index (harmless) but the DP below requires sorted
    // input, so this is restated rather than assumed.
    out.sort_by(f64::total_cmp);
    out
}

/// Fisher-Jenks 1-D dynamic program: minimise total within-class squared
/// deviation. `data` must be sorted ascending, non-empty, with `k <= data.len()`.
fn jenks_breaks(data: &[f64], k: usize) -> Vec<f64> {
    let n = data.len();
    let mut lower = vec![vec![0usize; k + 1]; n + 1];
    let mut variance = vec![vec![f64::INFINITY; k + 1]; n + 1];

    for j in 1..=k {
        lower[1][j] = 1;
        variance[1][j] = 0.0;
    }

    for l in 2..=n {
        let (mut sum, mut sum_sq, mut w) = (0.0f64, 0.0f64, 0.0f64);
        for m in 1..=l {
            let lower_limit = l - m + 1;
            let val = data[lower_limit - 1];
            w += 1.0;
            sum += val;
            sum_sq += val * val;
            let var = sum_sq - (sum * sum) / w;
            let prev = lower_limit - 1;
            if prev != 0 {
                for j in 2..=k {
                    let candidate = var + variance[prev][j - 1];
                    if variance[l][j] >= candidate {
                        lower[l][j] = lower_limit;
                        variance[l][j] = candidate;
                    }
                }
            }
        }
        lower[l][1] = 1;
        let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
        for &v in data.iter().take(l) {
            sum += v;
            sum_sq += v * v;
        }
        variance[l][1] = sum_sq - (sum * sum) / l as f64;
    }

    let mut breaks = vec![0.0f64; k + 1];
    breaks[0] = data[0];
    breaks[k] = data[n - 1];
    let mut kk = n;
    let mut count = k;
    while count > 1 {
        let idx = lower[kk][count].saturating_sub(1);
        breaks[count - 1] = data[idx.min(n - 1)];
        kk = lower[kk][count].saturating_sub(1).max(1);
        count -= 1;
    }
    // Numerical safety: fenceposts must never decrease, or `class_of`'s binary
    // reasoning stops meaning anything.
    for i in 1..=k {
        if breaks[i] < breaks[i - 1] {
            breaks[i] = breaks[i - 1];
        }
    }
    breaks
}

/// Which class a value falls in, given `breaks` from [`classify_breaks`].
///
/// Classes are **lower-inclusive**: `[breaks[i], breaks[i+1])`, with the top
/// class closed so the maximum value is in a class rather than past the last
/// one. Returns `0` for a degenerate break list.
///
/// Non-finite values return `None` — a no-data sample has no class, and the
/// caller decides what to write there (which is a nodata policy decision, and
/// belongs at the door that has one).
pub fn class_of(value: f64, breaks: &[f64]) -> Option<usize> {
    if !value.is_finite() || breaks.len() < 2 {
        return if value.is_finite() { Some(0) } else { None };
    }
    let k = breaks.len() - 1;
    if value <= breaks[0] {
        return Some(0);
    }
    if value >= breaks[k] {
        return Some(k - 1);
    }
    // `k` is at most MAX_CLASSES, so a linear scan is the right shape and avoids
    // the off-by-one traps a hand-rolled binary search on fenceposts invites.
    for i in 0..k {
        if value < breaks[i + 1] {
            return Some(i);
        }
    }
    Some(k - 1)
}

/// Map every value onto a class id in `0..k`, as a `u8` — the biome-id shape.
///
/// `nodata_id` is written where a value is non-finite. Capped at 255 because a
/// biome id is a `u8`.
pub fn classify_to_ids(
    values: &[f64],
    method: ClassifyMethod,
    classes: usize,
    nodata_id: u8,
) -> (Vec<u8>, Vec<f64>) {
    let breaks = classify_breaks(values, method, classes.min(255));
    let ids = values
        .iter()
        .map(|&v| class_of(v, &breaks).map_or(nodata_id, |c| c.min(255) as u8))
        .collect();
    (ids, breaks)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property that makes natural breaks the right default: a break lands
    /// **in the gap** of a bimodal distribution, where equal-interval would put
    /// it in the middle of a cluster.
    ///
    /// Un-fix mutation: route NaturalBreaks to equal_interval and the "inside the
    /// gap" assertion fails.
    #[test]
    fn natural_breaks_lands_in_the_gap_of_a_bimodal_set() {
        // Two tight clusters around 10 and 90, with an empty 20..80 gap.
        let mut v: Vec<f64> = (0..40).map(|i| 10.0 + (i % 7) as f64 * 0.5).collect();
        v.extend((0..40).map(|i| 90.0 + (i % 5) as f64 * 0.5));

        let b = classify_breaks(&v, ClassifyMethod::NaturalBreaks, 2);
        assert_eq!(b.len(), 3, "2 classes = 3 fenceposts");
        // Jenks fenceposts are the LOWER value of each class, so the interior
        // one lands at the bottom of the upper cluster (90) rather than strictly
        // inside the void. What matters is that it is past the top of the lower
        // cluster (13) — i.e. that the gap, not a cluster, was cut.
        //
        // **The band has to exclude the answer the mutation gives.** The first
        // version wrote `b[1] > 13.0 && b[1] <= 90.0`, and equal-interval on
        // this data breaks at (10 + 92) / 2 = 51 — comfortably inside that band.
        // So the named un-fix mutation (route `NaturalBreaks` to
        // `equal_interval`) passed every assertion in this test, including the
        // two classification loops, because 51 also separates the clusters. The
        // band that means something is the one that admits only the Jenks
        // answer: the bottom of the upper cluster, not anywhere in the void.
        assert!(
            (b[1] - 90.0).abs() < 1e-9,
            "the interior break is {}; Jenks on two tight clusters puts it at the \
             BOTTOM OF THE UPPER ONE (90). A break anywhere else in the void — 51, \
             say — is what equal-interval gives, which is the method this one is \
             being chosen over",
            b[1]
        );
        // Every low value classes low, every high value classes high.
        for &x in v.iter().filter(|&&x| x < 50.0) {
            assert_eq!(class_of(x, &b), Some(0), "{x}");
        }
        for &x in v.iter().filter(|&&x| x > 50.0) {
            assert_eq!(class_of(x, &b), Some(1), "{x}");
        }

        // Equal-interval, on the same data, splits at the arithmetic midpoint —
        // which is what natural breaks is being chosen over.
        let ei = classify_breaks(&v, ClassifyMethod::EqualInterval, 2);
        assert!(
            (ei[1] - (ei[0] + ei[2]) / 2.0).abs() < 1e-9,
            "equal-interval's break must be the midpoint, got {ei:?}"
        );
    }

    #[test]
    fn quantile_classes_hold_equal_counts() {
        let v: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let b = classify_breaks(&v, ClassifyMethod::Quantile, 4);
        let mut counts = [0usize; 4];
        for &x in &v {
            counts[class_of(x, &b).unwrap()] += 1;
        }
        for (i, c) in counts.iter().enumerate() {
            assert!(
                (*c as i64 - 25).abs() <= 1,
                "quantile class {i} holds {c}, expected ~25 (all: {counts:?})"
            );
        }
    }

    /// Breaks are always ascending and always span the data — the two invariants
    /// `class_of` depends on, checked across every method and a range of `k`.
    #[test]
    fn breaks_are_ascending_and_span_the_data_for_every_method() {
        let v: Vec<f64> = (0..500)
            .map(|i| ((i * 7919) % 997) as f64 / 3.0 - 50.0)
            .collect();
        let (lo, hi) = v
            .iter()
            .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
        for method in [
            ClassifyMethod::EqualInterval,
            ClassifyMethod::Quantile,
            ClassifyMethod::NaturalBreaks,
        ] {
            for k in [2usize, 3, 5, 8] {
                let b = classify_breaks(&v, method, k);
                assert_eq!(b.len(), k + 1, "{method:?} k={k}");
                assert_eq!(b[0], lo, "{method:?} k={k}: first fencepost is the min");
                assert_eq!(b[k], hi, "{method:?} k={k}: last fencepost is the max");
                for w in b.windows(2) {
                    assert!(w[0] <= w[1], "{method:?} k={k}: breaks descend: {b:?}");
                }
                // Every value gets a class, and it is in range.
                for &x in &v {
                    let c = class_of(x, &b).expect("finite values always class");
                    assert!(c < k, "{method:?} k={k}: {x} classed {c}");
                }
            }
        }
    }

    /// The degenerate collapses: fewer distinct values than classes, all-equal
    /// data, empty data, and non-finite data.
    #[test]
    fn degenerate_inputs_collapse_rather_than_producing_empty_classes() {
        // Three distinct values, eight classes asked for → collapses to three.
        let b = classify_breaks(&[1.0, 1.0, 2.0, 2.0, 3.0], ClassifyMethod::NaturalBreaks, 8);
        assert_eq!(
            b.len(),
            4,
            "3 distinct values cannot support 8 classes: {b:?}"
        );

        // All equal → one class.
        let b = classify_breaks(&[5.0; 20], ClassifyMethod::NaturalBreaks, 5);
        assert_eq!(b, vec![5.0, 5.0]);
        assert_eq!(class_of(5.0, &b), Some(0));

        // Empty.
        assert_eq!(
            classify_breaks(&[], ClassifyMethod::Quantile, 4),
            vec![0.0, 0.0]
        );

        // Non-finite values are dropped from the fit and have no class.
        let b = classify_breaks(
            &[1.0, f64::NAN, 2.0, f64::INFINITY, 3.0],
            ClassifyMethod::EqualInterval,
            2,
        );
        assert_eq!(b[0], 1.0);
        assert_eq!(b[2], 3.0);
        assert_eq!(class_of(f64::NAN, &b), None, "no-data has no class");
        assert_eq!(class_of(f64::INFINITY, &b), None);
    }

    /// The land-cover → biome-id path, including where no-data lands.
    #[test]
    fn classifying_to_biome_ids_writes_the_nodata_id_for_no_data() {
        let vals = vec![0.0, 10.0, 20.0, f64::NAN, 30.0, 40.0];
        let (ids, breaks) = classify_to_ids(&vals, ClassifyMethod::Quantile, 3, 255);
        assert_eq!(ids.len(), vals.len());
        assert_eq!(ids[3], 255, "the NaN sample takes the nodata id");
        for (i, &v) in vals.iter().enumerate() {
            if v.is_finite() {
                assert!(ids[i] < 3, "sample {i} ({v}) got id {}", ids[i]);
            }
        }
        assert_eq!(breaks.len(), 4);
        // Monotone source → monotone ids.
        assert!(ids[0] <= ids[1] && ids[4] <= ids[5]);
    }

    /// Determinism, and that the sampling cap actually engages without changing
    /// the answer's shape.
    #[test]
    fn large_inputs_are_sampled_deterministically() {
        let v: Vec<f64> = (0..50_000)
            .map(|i| ((i * 2654435761u64 as usize) % 100_000) as f64)
            .collect();
        assert!(v.len() > JENKS_SAMPLE_CAP);
        let a = classify_breaks(&v, ClassifyMethod::NaturalBreaks, 5);
        let b = classify_breaks(&v, ClassifyMethod::NaturalBreaks, 5);
        assert_eq!(a, b, "two runs on the same data must agree exactly");
        assert_eq!(a.len(), 6);
        for w in a.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }
}
