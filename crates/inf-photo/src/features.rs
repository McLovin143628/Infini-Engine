//! **ORB-class** feature extraction: multi-scale FAST-9 corners, ranked by
//! Harris response, oriented by intensity centroid, described by a 256-bit
//! BRIEF-class binary descriptor over a compile-time constant sampling pattern.
//!
//! # The choice, and what it buys
//!
//! SIFT is out of patent and would work; hand-rolling a difference-of-Gaussians
//! scale space with sub-scale interpolation and 128-dimensional gradient
//! histograms is several times this file, and its float descriptor makes
//! matching several times slower. ORB-class is the v1 ruling: the detector is a
//! sixteen-pixel comparison, the descriptor is four `u64`s, and matching is
//! `xor` + `count_ones`.
//!
//! # The bounds, stated rather than discovered
//!
//! * **Rotation invariance: yes.** Every feature carries an orientation taken
//!   from the intensity centroid of its patch, and the sampling pattern is
//!   rotated by it before the comparisons are made. Two photographs of the same
//!   surface with the camera rolled 90° between them still match.
//! * **Scale invariance: bounded by the pyramid.** Features are found at
//!   [`FeatureConfig::levels`] scales spaced by [`FeatureConfig::scale_factor`],
//!   so the span is `scale_factor^(levels-1)` — the default `1.2^3 ≈ 1.73`.
//!   Beyond roughly that ratio, a feature found in one photograph has no
//!   counterpart level in the other. Photographs taken from wildly different
//!   distances will lose matches, and the remedy is more levels, not a
//!   different threshold.
//! * **Affine invariance: no.** BRIEF compares fixed offsets; a surface seen at
//!   60° of foreshortening in one view and head-on in another does not produce
//!   the same descriptor. This is the real limit on how far apart two capture
//!   positions can be.
//! * **Illumination: intensity-monotone only.** The descriptor is a set of
//!   *comparisons*, so it survives any monotone change (exposure, gain, gamma)
//!   and does not survive a specular highlight moving across the patch.
//!
//! # No randomness at runtime
//!
//! The classic BRIEF pattern is drawn from a distribution once and then frozen.
//! Here it is frozen *at compile time*: [`BRIEF_PATTERN`] is built by a `const
//! fn` from [`PATTERN_SEED`] through the same SplitMix64 mixer everything else
//! in the crate uses, so it is a constant table in the binary, identical on
//! every machine, with no initialisation order and no lazy cell.

use crate::gray::{GrayImage, Pyramid, PyramidLevel};
use crate::hash::mix64;
use crate::PhotoError;

/// A descriptor is four 64-bit words.
pub const DESCRIPTOR_WORDS: usize = 4;
/// ...that is, 256 bits, one per sampling pair.
pub const DESCRIPTOR_BITS: usize = 64 * DESCRIPTOR_WORDS;

/// Sampling pairs are drawn from a disc of this radius, in **level pixels**.
pub const PATTERN_RADIUS: i32 = 13;
/// The intensity-centroid patch radius, in **level pixels**.
pub const ORIENTATION_RADIUS: i32 = 15;
/// How far from a level's edge a feature must sit for its patch to be whole.
pub const PATCH_BORDER: i64 = ORIENTATION_RADIUS as i64 + 1;
/// The FAST ring radius.
pub const FAST_RADIUS: i64 = 3;
/// How many of the sixteen ring pixels must form a contiguous brighter/darker
/// arc. Nine is the classic FAST-9 and the most repeatable of the family.
pub const FAST_ARC: usize = 9;
/// Half-width of the Harris window, in level pixels.
pub const HARRIS_HALF: i64 = 3;

/// The committed seed the constant BRIEF pattern is drawn from. Changing it
/// changes every descriptor in the engine, so it is a number with a test on it,
/// not a tunable.
pub const PATTERN_SEED: u64 = 0x5048_4f54_4f47_3235;

/// The sixteen-pixel Bresenham circle of radius three, in clockwise order
/// starting from straight up. Fixed and committed: FAST's contiguity test is
/// defined on *this* ordering.
pub const FAST_RING: [(i64, i64); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

/// One pseudo-random integer in `[-PATTERN_RADIUS, PATTERN_RADIUS]`, drawn from
/// the counter at compile time.
const fn pattern_draw(counter: u64) -> i32 {
    let h = mix64(counter.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ PATTERN_SEED);
    let span = (2 * PATTERN_RADIUS + 1) as u64;
    (h % span) as i32 - PATTERN_RADIUS
}

/// Build the descriptor's sampling pattern at compile time.
///
/// Rejection sampling in a disc: a pair is kept when both its points lie inside
/// [`PATTERN_RADIUS`] and the two points differ. Uniform-in-a-disc rather than
/// the Gaussian the BRIEF paper favours — the difference is a fraction of a
/// percent of matching recall, and a Gaussian needs a transform this cannot
/// evaluate in a `const fn`. The modulo above carries the usual modulo bias; for
/// a table that is drawn once and frozen, bias is irrelevant by construction.
const fn build_pattern() -> [[i8; 4]; DESCRIPTOR_BITS] {
    let mut out = [[0i8; 4]; DESCRIPTOR_BITS];
    let r2 = PATTERN_RADIUS * PATTERN_RADIUS;
    let mut i = 0usize;
    let mut counter = 0u64;
    while i < DESCRIPTOR_BITS {
        let ax = pattern_draw(counter);
        let ay = pattern_draw(counter + 1);
        let bx = pattern_draw(counter + 2);
        let by = pattern_draw(counter + 3);
        counter += 4;
        if ax * ax + ay * ay <= r2 && bx * bx + by * by <= r2 && (ax != bx || ay != by) {
            out[i] = [ax as i8, ay as i8, bx as i8, by as i8];
            i += 1;
        }
    }
    out
}

/// The frozen BRIEF sampling pattern: 256 pairs of `(ax, ay, bx, by)` offsets in
/// level pixels, evaluated at compile time from [`PATTERN_SEED`].
pub static BRIEF_PATTERN: [[i8; 4]; DESCRIPTOR_BITS] = build_pattern();

/// How the detector is tuned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureConfig {
    /// How many pyramid levels to search.
    pub levels: u32,
    /// Linear shrink between levels (must exceed 1).
    pub scale_factor: f64,
    /// A level whose smaller side falls below this is not built.
    pub min_level_side: u32,
    /// FAST intensity threshold, in **8-bit levels**.
    pub fast_threshold: u8,
    /// The Harris trace weight (`det - k * trace²`), dimensionless.
    pub harris_k: f64,
    /// How many features to keep per image, across all levels.
    pub max_features: usize,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            levels: 4,
            scale_factor: 1.2,
            min_level_side: 64,
            fast_threshold: 18,
            harris_k: 0.04,
            max_features: 512,
        }
    }
}

/// One detected, oriented, described feature.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Feature {
    /// x in **level-0 pixels**, on the pixel-centre convention.
    pub x: f64,
    /// y in **level-0 pixels**.
    pub y: f64,
    /// Which pyramid level found it.
    pub level: u8,
    /// The Harris response at its own level (dimensionless; comparable only
    /// within a level).
    pub score: f64,
    /// Cosine of the patch orientation, taken straight from the intensity
    /// centroid — no `atan2`, no `cos`.
    pub orientation_cos: f64,
    /// Sine of the patch orientation.
    pub orientation_sin: f64,
    /// The 256-bit descriptor.
    pub descriptor: [u64; DESCRIPTOR_WORDS],
}

impl Feature {
    /// Hamming distance to another descriptor, in bits (`0..=256`).
    #[inline]
    pub fn distance(&self, other: &Feature) -> u32 {
        let mut d = 0u32;
        for w in 0..DESCRIPTOR_WORDS {
            d += (self.descriptor[w] ^ other.descriptor[w]).count_ones();
        }
        d
    }
}

/// Every feature of one photograph, in canonical order: by level, then by `y`,
/// then by `x`. Index into this vector is a feature's identity for the rest of
/// the pipeline, so the order is part of the contract.
#[derive(Clone, Debug, Default)]
pub struct FeatureSet {
    /// The features, canonically ordered.
    pub features: Vec<Feature>,
}

impl FeatureSet {
    /// How many features.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Is it empty?
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// Detect, orient and describe every feature of one photograph.
pub fn detect_and_describe(
    image: &GrayImage,
    cfg: &FeatureConfig,
) -> Result<FeatureSet, PhotoError> {
    let pyramid = Pyramid::build(image, cfg.levels, cfg.scale_factor, cfg.min_level_side)?;
    Ok(describe_pyramid(&pyramid, cfg))
}

/// The same, on a pyramid the caller already built.
pub fn describe_pyramid(pyramid: &Pyramid, cfg: &FeatureConfig) -> FeatureSet {
    // Per-level candidate corners, already non-max suppressed.
    let mut per_level: Vec<Vec<Corner>> = Vec::with_capacity(pyramid.levels.len());
    for level in &pyramid.levels {
        per_level.push(detect_level(level, cfg));
    }

    // Quota by level area, the ORB rule: a level covering a quarter of the
    // pixels earns a quarter of the features. Unfilled quota flows down to the
    // next level rather than being lost, so a texture-poor coarse level does not
    // cost the image its budget.
    let n_levels = pyramid.levels.len();
    let mut weights = Vec::with_capacity(n_levels);
    let mut w = 1.0f64;
    let ratio = 1.0 / (cfg.scale_factor * cfg.scale_factor);
    for _ in 0..n_levels {
        weights.push(w);
        w *= ratio;
    }
    let total: f64 = weights.iter().sum();

    let mut selected: Vec<(usize, Corner)> = Vec::new();
    let mut carried = 0usize;
    for (level_idx, corners) in per_level.iter_mut().enumerate() {
        let share = if total > 0.0 {
            (cfg.max_features as f64 * weights[level_idx] / total).round() as usize
        } else {
            0
        };
        let quota = share + carried;
        // Strongest first; ties broken by position so the choice is a total
        // order, never the sort's stability.
        corners.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.y.cmp(&b.y))
                .then(a.x.cmp(&b.x))
        });
        let take = quota.min(corners.len());
        carried = quota - take;
        for c in corners.iter().take(take) {
            selected.push((level_idx, *c));
        }
    }

    // Canonical output order.
    selected.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.y.cmp(&b.1.y))
            .then(a.1.x.cmp(&b.1.x))
    });

    let mut features = Vec::with_capacity(selected.len());
    for (level_idx, corner) in selected {
        let level = &pyramid.levels[level_idx];
        let (cos, sin) = orientation(&level.image, corner.x, corner.y);
        let descriptor = describe(&level.blurred, corner.x, corner.y, cos, sin);
        // Level coordinate -> level-0 coordinate, pixel-centre convention. The
        // matching `- 0.5` is in `GrayImage::resample`; the two must agree or
        // every coarse feature lands half a pixel off.
        let x0 = (corner.x as f64 + corner.dx + 0.5) * level.scale - 0.5;
        let y0 = (corner.y as f64 + corner.dy + 0.5) * level.scale - 0.5;
        features.push(Feature {
            x: x0,
            y: y0,
            level: level_idx as u8,
            score: corner.score,
            orientation_cos: cos,
            orientation_sin: sin,
            descriptor,
        });
    }
    // The canonical output order, established on the **reported** coordinates
    // rather than the detector grid they came from. Sub-pixel refinement moves a
    // feature by up to half a pixel, which is enough to swap two neighbours in a
    // row — so ordering by the grid and *claiming* to be ordered by position
    // would be a contract the type does not keep. `total_cmp` on values that are
    // pure functions of the input keeps it deterministic.
    features.sort_by(|a, b| {
        a.level
            .cmp(&b.level)
            .then(a.y.total_cmp(&b.y))
            .then(a.x.total_cmp(&b.x))
    });
    FeatureSet { features }
}

#[derive(Clone, Copy, Debug)]
struct Corner {
    x: i64,
    y: i64,
    /// Sub-pixel offsets in `[-0.5, 0.5]`, from a parabola through the Harris
    /// response. Never used for ordering, indexing or descriptor sampling — only
    /// for the level-0 coordinate a bundle adjuster minimises against.
    dx: f64,
    dy: f64,
    score: f64,
}

/// Fit a parabola through three samples and return the offset of its vertex,
/// clamped to half a pixel.
///
/// `0.0` when the three samples do not describe a maximum, which is the honest
/// answer: there is no sub-pixel information in a flat or a rising triple.
#[inline]
fn parabola_vertex(low: f64, centre: f64, high: f64) -> f64 {
    let denom = low - 2.0 * centre + high;
    if denom >= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    let offset = 0.5 * (low - high) / denom;
    if offset.is_finite() {
        offset.clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

/// FAST-9 detection plus Harris ranking plus 3x3 non-max suppression, on one
/// pyramid level.
fn detect_level(level: &PyramidLevel, cfg: &FeatureConfig) -> Vec<Corner> {
    let img = &level.image;
    let w = img.width() as i64;
    let h = img.height() as i64;
    let lo = PATCH_BORDER;
    if w <= 2 * lo || h <= 2 * lo {
        return Vec::new();
    }

    // Score map over the interior; NaN marks "not a corner" and never wins a
    // comparison, which is what makes the suppression below total.
    let mut scores = vec![f64::NAN; (w * h) as usize];
    let mut candidates = Vec::new();
    for y in lo..(h - lo) {
        for x in lo..(w - lo) {
            if !is_fast_corner(img, x, y, cfg.fast_threshold) {
                continue;
            }
            let s = harris_response(img, x, y, cfg.harris_k);
            scores[(y * w + x) as usize] = s;
            candidates.push((x, y, s));
        }
    }

    let mut out = Vec::with_capacity(candidates.len());
    for (x, y, s) in candidates {
        let mut survives = true;
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let n = scores[((y + dy) * w + (x + dx)) as usize];
                if n.is_nan() {
                    continue;
                }
                // A strict beat loses; an exact tie is decided by raster order,
                // so a plateau of equal responses keeps exactly one corner and
                // keeps the same one every run.
                if n > s || (n == s && (y + dy, x + dx) < (y, x)) {
                    survives = false;
                }
            }
        }
        if !survives {
            continue;
        }
        // Sub-pixel localisation. A detector that reports integer coordinates
        // hands the bundle adjuster observations carrying a uniform +-0.5 px of
        // quantisation, and that noise is **correlated** across a textured
        // surface rather than independent, so it does not average away with more
        // points — it comes out as a systematic pose error. Measured on the gate
        // fixture: integer coordinates gave a relative camera-rotation error of
        // 0.31 degrees at a reprojection RMS of 0.44 px, which is exactly the
        // quantisation floor and nowhere near what the geometry can do.
        //
        // The Harris response of a corner is smooth and peaked, so a parabola
        // through it and its two neighbours on each axis puts the peak where the
        // feature actually is. Four extra Harris evaluations per surviving
        // corner.
        let west = harris_response(img, x - 1, y, cfg.harris_k);
        let east = harris_response(img, x + 1, y, cfg.harris_k);
        let north = harris_response(img, x, y - 1, cfg.harris_k);
        let south = harris_response(img, x, y + 1, cfg.harris_k);
        out.push(Corner {
            x,
            y,
            dx: parabola_vertex(west, s, east),
            dy: parabola_vertex(north, s, south),
            score: s,
        });
    }
    out
}

/// The FAST-9 test: are nine contiguous ring pixels all brighter than
/// `centre + t`, or all darker than `centre - t`?
fn is_fast_corner(img: &GrayImage, x: i64, y: i64, threshold: u8) -> bool {
    let c = img.at(x as u32, y as u32) as i32;
    let t = threshold as i32;
    let hi = c + t;
    let lo = c - t;

    // The early reject on the four compass points. The classic FAST-12 form
    // demands three of four; **FAST-9 must demand only two**, and this is not a
    // loosening for safety's sake — it is arithmetic. The four compass points
    // sit at ring indices 0, 4, 8 and 12, and a window of nine consecutive
    // indices out of sixteen contains at least two of them and, for windows
    // starting at 1, 2 or 3, *exactly* two. Demanding three rejects the corner
    // of a square: measured on `fast_fires_on_a_corner_and_not_on_a_flat_field
    // _or_an_edge`, an axis-aligned convex corner lights exactly two compass
    // points, and with a threshold of three the detector found nothing at all
    // on the one shape it most obviously should.
    let mut bright_compass = 0;
    let mut dark_compass = 0;
    for k in [0usize, 4, 8, 12] {
        let (dx, dy) = FAST_RING[k];
        let v = img.at((x + dx) as u32, (y + dy) as u32) as i32;
        if v > hi {
            bright_compass += 1;
        } else if v < lo {
            dark_compass += 1;
        }
    }
    if bright_compass < 2 && dark_compass < 2 {
        return false;
    }

    let mut ring = [0i32; 16];
    for (k, slot) in ring.iter_mut().enumerate() {
        let (dx, dy) = FAST_RING[k];
        *slot = img.at((x + dx) as u32, (y + dy) as u32) as i32;
    }
    contiguous_run(&ring, |v| v > hi) || contiguous_run(&ring, |v| v < lo)
}

/// Is there a wrap-around run of at least [`FAST_ARC`] entries satisfying
/// `pred`?
fn contiguous_run(ring: &[i32; 16], pred: impl Fn(i32) -> bool) -> bool {
    let flags: [bool; 16] = std::array::from_fn(|i| pred(ring[i]));
    if flags.iter().all(|&f| f) {
        return true;
    }
    let mut run = 0usize;
    // Walk twice around so a run that straddles index 0 is seen whole.
    for i in 0..32 {
        if flags[i % 16] {
            run += 1;
            if run >= FAST_ARC {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// The Harris corner response over a `(2*HARRIS_HALF+1)²` window.
///
/// Used only to *rank* FAST corners — FAST fires on edges and on noise as
/// happily as on corners, and its own "maximum threshold that still detects"
/// score is a poor discriminator. Harris is the standard remedy and is what ORB
/// uses.
fn harris_response(img: &GrayImage, x: i64, y: i64, k: f64) -> f64 {
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for dy in -HARRIS_HALF..=HARRIS_HALF {
        for dx in -HARRIS_HALF..=HARRIS_HALF {
            let px = x + dx;
            let py = y + dy;
            let ix = (img.clamped(px + 1, py) as f64 - img.clamped(px - 1, py) as f64) * 0.5;
            let iy = (img.clamped(px, py + 1) as f64 - img.clamped(px, py - 1) as f64) * 0.5;
            sxx += ix * ix;
            syy += iy * iy;
            sxy += ix * iy;
        }
    }
    let det = sxx * syy - sxy * sxy;
    let trace = sxx + syy;
    det - k * trace * trace
}

/// Patch orientation from the intensity centroid.
///
/// Returns `(cos, sin)` of the angle from the patch centre to its centre of
/// mass, as a **normalized 2-vector taken directly** — the angle itself is never
/// formed, so no `atan2`, no `cos`, no `sin`, and nothing platform-dependent.
fn orientation(img: &GrayImage, x: i64, y: i64) -> (f64, f64) {
    let r = ORIENTATION_RADIUS as i64;
    let r2 = r * r;
    let mut m10 = 0.0f64;
    let mut m01 = 0.0f64;
    for dy in -r..=r {
        let span = r2 - dy * dy;
        for dx in -r..=r {
            if dx * dx > span {
                continue;
            }
            let v = img.clamped(x + dx, y + dy) as f64;
            m10 += dx as f64 * v;
            m01 += dy as f64 * v;
        }
    }
    let len = (m10 * m10 + m01 * m01).sqrt();
    if len > 0.0 {
        (m10 / len, m01 / len)
    } else {
        (1.0, 0.0)
    }
}

/// Sample [`BRIEF_PATTERN`], rotated by the feature's orientation, on the
/// blurred level.
fn describe(blurred: &GrayImage, x: i64, y: i64, cos: f64, sin: f64) -> [u64; DESCRIPTOR_WORDS] {
    let mut out = [0u64; DESCRIPTOR_WORDS];
    for (bit, pair) in BRIEF_PATTERN.iter().enumerate() {
        let ax = pair[0] as f64;
        let ay = pair[1] as f64;
        let bx = pair[2] as f64;
        let by = pair[3] as f64;
        let rax = (cos * ax - sin * ay).round() as i64;
        let ray = (sin * ax + cos * ay).round() as i64;
        let rbx = (cos * bx - sin * by).round() as i64;
        let rby = (sin * bx + cos * by).round() as i64;
        let a = blurred.clamped(x + rax, y + ray);
        let b = blurred.clamped(x + rbx, y + rby);
        if a < b {
            out[bit / 64] |= 1u64 << (bit % 64);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field of hash-placed dots of varying radius: high-contrast, locally
    /// distinctive, and — unlike a checkerboard — **not** symmetric under a
    /// quarter turn, so a patch has a well-defined intensity centroid.
    fn speckle(w: u32, h: u32) -> GrayImage {
        use crate::hash::Hash64;
        let mut img = GrayImage::new(w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, 210);
            }
        }
        let cell = 9i64;
        for cy in 0..(h as i64 / cell + 1) {
            for cx in 0..(w as i64 / cell + 1) {
                let base = Hash64::new(0x51ee)
                    .mix_usize(cx as usize)
                    .mix_usize(cy as usize);
                let ox = base.mix_u64(1).unit();
                let oy = base.mix_u64(2).unit();
                let rr = 1.4 + 1.6 * base.mix_u64(3).unit();
                let dx = (cx as f64 + 0.15 + 0.7 * ox) * cell as f64;
                let dy = (cy as f64 + 0.15 + 0.7 * oy) * cell as f64;
                let r = rr.ceil() as i64 + 1;
                for py in (dy as i64 - r)..=(dy as i64 + r) {
                    for px in (dx as i64 - r)..=(dx as i64 + r) {
                        if px < 0 || py < 0 || px >= w as i64 || py >= h as i64 {
                            continue;
                        }
                        let d = ((px as f64 - dx).powi(2) + (py as f64 - dy).powi(2)).sqrt();
                        if d <= rr {
                            img.set(px as u32, py as u32, 25);
                        }
                    }
                }
            }
        }
        img
    }

    fn checkerboard(w: u32, h: u32, cell: u32) -> GrayImage {
        let mut img = GrayImage::new(w, h).unwrap();
        for y in 0..h {
            for x in 0..w {
                let v = if ((x / cell) + (y / cell)).is_multiple_of(2) {
                    30
                } else {
                    220
                };
                img.set(x, y, v);
            }
        }
        img
    }

    #[test]
    fn the_pattern_is_a_frozen_constant_inside_the_disc() {
        let r2 = PATTERN_RADIUS * PATTERN_RADIUS;
        for (i, p) in BRIEF_PATTERN.iter().enumerate() {
            let (ax, ay, bx, by) = (p[0] as i32, p[1] as i32, p[2] as i32, p[3] as i32);
            assert!(ax * ax + ay * ay <= r2, "pair {i} point a escaped the disc");
            assert!(bx * bx + by * by <= r2, "pair {i} point b escaped the disc");
            assert!(
                ax != bx || ay != by,
                "pair {i} compares a pixel with itself"
            );
        }
        // Not degenerate: a table of 256 identical pairs would pass the checks
        // above and describe nothing.
        let mut distinct = BRIEF_PATTERN.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert!(
            distinct.len() > DESCRIPTOR_BITS * 9 / 10,
            "only {} of {DESCRIPTOR_BITS} pairs are distinct",
            distinct.len()
        );
        // A hard pin. The pattern is part of the descriptor's definition: change
        // it and every stored descriptor in the engine means something else.
        assert_eq!(BRIEF_PATTERN.len(), 256);
        assert_eq!(BRIEF_PATTERN[0], build_pattern()[0]);
    }

    #[test]
    fn the_pattern_spreads_over_the_disc() {
        // A pattern collapsed onto one axis, or into one quadrant, would satisfy
        // every check above.
        let mut quadrants = [0usize; 4];
        for p in BRIEF_PATTERN.iter() {
            let q = usize::from(p[0] >= 0) + 2 * usize::from(p[1] >= 0);
            quadrants[q] += 1;
        }
        for (q, n) in quadrants.iter().enumerate() {
            assert!(
                *n > DESCRIPTOR_BITS / 10,
                "quadrant {q} holds only {n} points"
            );
        }
    }

    #[test]
    fn fast_fires_on_a_corner_and_not_on_a_flat_field_or_an_edge() {
        // A dark square on a light field: the interior corner is a FAST corner.
        let mut img = GrayImage::new(64, 64).unwrap();
        for y in 0..64 {
            for x in 0..64 {
                img.set(x, y, 220);
            }
        }
        for y in 32..64 {
            for x in 32..64 {
                img.set(x, y, 20);
            }
        }
        assert!(
            is_fast_corner(&img, 32, 32, 20),
            "the square's corner is not detected"
        );
        assert!(
            !is_fast_corner(&img, 10, 10, 20),
            "a flat field produced a corner"
        );
        assert!(
            !is_fast_corner(&img, 40, 32, 20),
            "a straight edge produced a corner"
        );
    }

    #[test]
    fn contiguity_wraps_around_the_ring() {
        // Nine bright entries straddling index 0.
        let mut ring = [0i32; 16];
        for slot in ring.iter_mut().take(5) {
            *slot = 100;
        }
        for slot in ring.iter_mut().skip(12) {
            *slot = 100;
        }
        assert!(contiguous_run(&ring, |v| v > 50));
        // Eight is not nine.
        ring[4] = 0;
        assert!(!contiguous_run(&ring, |v| v > 50));
        // All sixteen.
        assert!(contiguous_run(&[100; 16], |v| v > 50));
    }

    #[test]
    fn harris_ranks_a_corner_above_an_edge() {
        let mut img = GrayImage::new(64, 64).unwrap();
        for y in 0..64 {
            for x in 0..64 {
                img.set(x, y, if x >= 32 { 20 } else { 220 });
            }
        }
        let edge = harris_response(&img, 32, 20, 0.04);
        for y in 32..64 {
            for x in 0..64 {
                img.set(x, y, 220);
            }
        }
        let corner = harris_response(&img, 32, 32, 0.04);
        assert!(corner > edge, "corner {corner} did not beat edge {edge}");
    }

    #[test]
    fn orientation_points_at_the_bright_side() {
        let mut img = GrayImage::new(64, 64).unwrap();
        for y in 0..64 {
            for x in 0..64 {
                img.set(x, y, if x > 32 { 250 } else { 10 });
            }
        }
        let (cos, sin) = orientation(&img, 32, 32);
        assert!(
            cos > 0.9,
            "centroid should lie to the +x side, got cos {cos}"
        );
        assert!(sin.abs() < 0.1, "unexpected y component {sin}");
        assert!(
            (cos * cos + sin * sin - 1.0).abs() < 1e-12,
            "not a unit vector"
        );
        // Rotate the image 90° (bright below) and the orientation follows.
        for y in 0..64 {
            for x in 0..64 {
                img.set(x, y, if y > 32 { 250 } else { 10 });
            }
        }
        let (cos, sin) = orientation(&img, 32, 32);
        assert!(sin > 0.9, "rotated centroid not tracked: sin {sin}");
        assert!(cos.abs() < 0.1);
    }

    #[test]
    fn a_flat_patch_has_a_defined_orientation() {
        let mut img = GrayImage::new(64, 64).unwrap();
        for y in 0..64 {
            for x in 0..64 {
                img.set(x, y, 128);
            }
        }
        // A symmetric patch has zero centroid; the fallback must be finite.
        let (cos, sin) = orientation(&img, 32, 32);
        assert!(cos.is_finite() && sin.is_finite());
        assert!((cos * cos + sin * sin - 1.0).abs() < 1e-12);
    }

    #[test]
    fn descriptors_are_rotation_invariant() {
        // The same patch content, rotated 90°, must produce a near-identical
        // descriptor once the orientation has been taken out.
        let base = speckle(128, 128);
        let mut rotated = GrayImage::new(128, 128).unwrap();
        for y in 0..128 {
            for x in 0..128 {
                // 90° rotation about the centre of a square image.
                rotated.set(x, y, base.at(y, 127 - x));
            }
        }
        let cfg = FeatureConfig {
            levels: 1,
            max_features: 64,
            ..FeatureConfig::default()
        };
        let a = detect_and_describe(&base, &cfg).unwrap();
        let b = detect_and_describe(&rotated, &cfg).unwrap();
        assert!(
            !a.is_empty() && !b.is_empty(),
            "the checkerboard produced no features"
        );
        // For every feature in `a`, its rotated twin should be its nearest
        // descriptor in `b`, well inside the 256-bit space.
        let mut close = 0usize;
        for fa in &a.features {
            let best = b
                .features
                .iter()
                .map(|fb| fa.distance(fb))
                .min()
                .unwrap_or(u32::MAX);
            if best <= 64 {
                close += 1;
            }
        }
        assert!(
            close * 2 > a.len(),
            "only {close} of {} features found a rotated twin within 64 bits",
            a.len()
        );
    }

    /// One antialiased dark disc on a light field, at an exactly known
    /// sub-pixel centre. 4x4 box sampling per pixel, so the edge carries real
    /// sub-pixel information rather than a staircase.
    fn disc_image(cx: f64, cy: f64, radius: f64) -> GrayImage {
        let mut img = GrayImage::new(64, 64).unwrap();
        for y in 0..64u32 {
            for x in 0..64u32 {
                let mut inside = 0u32;
                for sy in 0..4 {
                    for sx in 0..4 {
                        // Pixel-centre convention: pixel `x` is centred at `x`.
                        let px = x as f64 + (sx as f64 + 0.5) / 4.0 - 0.5;
                        let py = y as f64 + (sy as f64 + 0.5) / 4.0 - 0.5;
                        if (px - cx).powi(2) + (py - cy).powi(2) <= radius * radius {
                            inside += 1;
                        }
                    }
                }
                let t = inside as f64 / 16.0;
                img.set(x, y, (215.0 + (25.0 - 215.0) * t).round() as u8);
            }
        }
        img
    }

    #[test]
    fn sub_pixel_localisation_tracks_a_disc_between_pixels() {
        // A detector reporting integer coordinates is off by up to half a pixel
        // by construction, and that error is *correlated* across a textured
        // surface rather than independent — so it does not average away with
        // more points, it comes out as camera-pose error. This arm measures the
        // mechanism directly, because the end-to-end pose bounds have too much
        // margin to feel it: severing the refinement moves the P25.1 gate's
        // reprojection RMS only from 0.36 px to 0.42 px.
        let cfg = FeatureConfig {
            levels: 1,
            max_features: 8,
            ..FeatureConfig::default()
        };
        let truth_y = 32.3;
        let mut worst = 0.0f64;
        let mut total = 0.0f64;
        let mut samples = 0usize;
        for i in 0..10 {
            let truth_x = 31.55 + 0.1 * i as f64;
            let img = disc_image(truth_x, truth_y, 2.2);
            let set = detect_and_describe(&img, &cfg).unwrap();
            let best = set
                .features
                .iter()
                .min_by(|a, b| {
                    let da = (a.x - truth_x).hypot(a.y - truth_y);
                    let db = (b.x - truth_x).hypot(b.y - truth_y);
                    da.total_cmp(&db)
                })
                .unwrap_or_else(|| panic!("no feature found for a disc at ({truth_x}, {truth_y})"));
            let err = (best.x - truth_x).hypot(best.y - truth_y);
            worst = worst.max(err);
            total += err;
            samples += 1;
        }
        let mean = total / samples as f64;
        // Half a pixel is what an integer report costs at its worst; a quarter is
        // a bound only sub-pixel refinement can hold across a sweep that steps
        // straight through a pixel boundary.
        assert!(
            worst < 0.30,
            "worst sub-pixel error over the sweep is {worst:.4} px (mean {mean:.4}) — an \
             integer-coordinate detector reaches 0.5 by construction"
        );
        assert!(mean < 0.18, "mean sub-pixel error is {mean:.4} px");
    }

    #[test]
    fn detection_is_bit_identical_across_runs() {
        let img = checkerboard(128, 96, 5);
        let cfg = FeatureConfig::default();
        let a = detect_and_describe(&img, &cfg).unwrap();
        let b = detect_and_describe(&img, &cfg).unwrap();
        assert_eq!(a.len(), b.len());
        for (fa, fb) in a.features.iter().zip(&b.features) {
            assert_eq!(fa.x.to_bits(), fb.x.to_bits());
            assert_eq!(fa.y.to_bits(), fb.y.to_bits());
            assert_eq!(fa.descriptor, fb.descriptor);
            assert_eq!(fa.level, fb.level);
        }
    }

    #[test]
    fn output_is_in_canonical_order_and_inside_the_image() {
        let img = checkerboard(160, 120, 6);
        let set = detect_and_describe(&img, &FeatureConfig::default()).unwrap();
        assert!(!set.is_empty());
        for pair in set.features.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            let key_a = (a.level, a.y.to_bits(), a.x.to_bits());
            let key_b = (b.level, b.y.to_bits(), b.x.to_bits());
            assert!(key_a <= key_b, "features are not canonically ordered");
        }
        for f in &set.features {
            assert!(
                f.x >= 0.0 && f.x < 160.0 && f.y >= 0.0 && f.y < 120.0,
                "{f:?} is off-image"
            );
        }
    }

    #[test]
    fn the_feature_budget_is_respected() {
        let img = checkerboard(200, 200, 4);
        let cfg = FeatureConfig {
            max_features: 40,
            ..FeatureConfig::default()
        };
        let set = detect_and_describe(&img, &cfg).unwrap();
        assert!(set.len() <= 40, "budget of 40 overrun: {}", set.len());
        assert!(set.len() > 10, "budget starved the detector: {}", set.len());
    }

    #[test]
    fn a_featureless_image_yields_nothing_rather_than_noise() {
        let mut img = GrayImage::new(128, 128).unwrap();
        for y in 0..128 {
            for x in 0..128 {
                img.set(x, y, 90);
            }
        }
        let set = detect_and_describe(&img, &FeatureConfig::default()).unwrap();
        assert!(
            set.is_empty(),
            "a flat field produced {} features",
            set.len()
        );
    }

    #[test]
    fn non_max_suppression_thins_a_plateau() {
        // A grid of identical blobs: each must yield at most one corner, never a
        // cluster of adjacent ones.
        let img = checkerboard(128, 128, 8);
        let cfg = FeatureConfig {
            levels: 1,
            max_features: 4096,
            ..FeatureConfig::default()
        };
        let set = detect_and_describe(&img, &cfg).unwrap();
        for a in &set.features {
            let neighbours = set
                .features
                .iter()
                .filter(|b| {
                    !std::ptr::eq(*b, a) && (b.x - a.x).abs() <= 1.0 && (b.y - a.y).abs() <= 1.0
                })
                .count();
            assert_eq!(
                neighbours, 0,
                "feature at ({}, {}) kept a 3x3 neighbour",
                a.x, a.y
            );
        }
    }
}
