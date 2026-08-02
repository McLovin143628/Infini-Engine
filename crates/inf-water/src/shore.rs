//! Shore queries: where the water meets the ground.
//!
//! # Two shore computations, deliberately
//!
//! **In the shader**, shore blending is a *screen-space depth difference*: the
//! water fragment knows its own depth and reads the opaque scene depth behind it,
//! and the difference is the water column between them. That handles terrain, a
//! jetty, a boat hull and a rock in the same expression, needs no CPU state, and
//! is what actually fades the surface out where it meets the ground.
//!
//! **Here on the CPU**, shore is a *world-space* question about terrain heights,
//! and it exists for the things a screen-space term cannot serve: a cook advisory,
//! a gameplay query ("am I in the shallows?"), P20.2's buoyancy, and P20.4's
//! authoring tools. It reads a caller-supplied height function rather than
//! `inf-terrain` directly — this crate is pure math and must not pull an image
//! decoder in to ask how tall a hill is (the same reason `inf_math::spline` takes
//! plain points rather than the ECS component).
//!
//! Both are named "shore" because they answer the same question; neither is
//! derived from the other, and the split is stated here so a later reader does not
//! go looking for the one that "should" have been shared.

use glam::DVec2;
use inf_math::portable::{pcos64, psin64};

/// Where a point sits relative to a waterline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShoreClass {
    /// The ground is below the water surface by more than the shore band.
    Submerged,
    /// The ground is within the shore band of the water surface — the waterline
    /// itself, where foam and wetness belong.
    Shoreline,
    /// The ground is above the water surface by more than the shore band.
    Dry,
}

/// Classify a ground height against a water level, with a band of `band_m`
/// either side.
///
/// A **band** rather than a strict comparison because a waterline is not a line:
/// it is where wet sand, foam and the surface fade all live, and every consumer
/// (foam, wetness, a "you are paddling" gameplay check) wants the same width for
/// it. `band_m ≤ 0` degenerates to the strict comparison, with `Shoreline` for an
/// exact tie.
pub fn shore_class(water_level_m: f64, ground_m: f64, band_m: f64) -> ShoreClass {
    let band = if band_m.is_finite() {
        band_m.max(0.0)
    } else {
        0.0
    };
    let depth = water_level_m - ground_m;
    if depth > band {
        ShoreClass::Submerged
    } else if depth < -band {
        ShoreClass::Dry
    } else {
        ShoreClass::Shoreline
    }
}

/// How "fully water" a point is, `[0, 1]` — `0` at and above the waterline,
/// rising smoothly to `1` once the ground is `band_m` below it.
///
/// A smoothstep, so the surface fades in with a zero derivative at both ends and
/// there is no visible crease where the blend starts. This is the CPU twin of
/// what the shader does with its depth difference, and both use the same curve so
/// a gameplay check and a pixel agree about where the shallows are.
pub fn shore_blend(water_level_m: f64, ground_m: f64, band_m: f64) -> f64 {
    let band = if band_m.is_finite() { band_m } else { 0.0 };
    let depth = water_level_m - ground_m;
    if band <= 0.0 {
        return if depth > 0.0 { 1.0 } else { 0.0 };
    }
    let t = (depth / band).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Number of radial directions [`shore_distance`] probes. Sixteen puts the
/// worst-case angular error at 11.25°, which on a 40 m search bounds the
/// over-estimate at ~2 % of the true distance for a straight shoreline.
pub const SHORE_PROBE_DIRECTIONS: u32 = 16;

/// An **SDF-class** distance from `p` to the nearest waterline, metres.
///
/// "SDF-class", not an SDF: it is a bounded radial probe, not an exact distance
/// field, and the difference is stated rather than papered over.
///
/// * `Some(d)` — a class change was found within `max_m`; `d` is the radius at
///   which it happened, refined by bisection to `max_m / (steps · 2^BISECTIONS)`.
///   Positive whichever side of the waterline `p` is on; use [`shore_class`] at
///   `p` for the sign.
/// * `None` — no waterline within `max_m` (or the terrain answered for nothing).
///
/// **Its guarantees**, which are what make it usable: it never *under*-reports
/// (a returned `d` really does have a class change at that radius), it is a pure
/// deterministic function of its inputs, and it is exact for a straight shoreline
/// probed head-on. What it does not promise is exactness on a concave shore,
/// where the nearest waterline may lie between two probe directions — the error
/// is bounded by the angular step, and a caller that needs better should raise
/// `steps` or ask the terrain directly.
///
/// Directions come from [`psin64`]/[`pcos64`], never `std` trig: a cook advisory
/// derived from this reaches a shipped report, and a report that differs between
/// a developer's machine and CI is worse than no report.
pub fn shore_distance(
    p: DVec2,
    water_level_m: f64,
    band_m: f64,
    max_m: f64,
    steps: u32,
    height_at: impl Fn(DVec2) -> Option<f64>,
) -> Option<f64> {
    const BISECTIONS: u32 = 6;
    if !max_m.is_finite() || max_m <= 0.0 {
        return None;
    }
    let steps = steps.max(1);
    let here = classify(p, water_level_m, band_m, &height_at)?;
    let mut best: Option<f64> = None;
    for d in 0..SHORE_PROBE_DIRECTIONS {
        let angle = std::f64::consts::TAU * d as f64 / SHORE_PROBE_DIRECTIONS as f64;
        let dir = DVec2::new(pcos64(angle), psin64(angle));
        let mut prev_r = 0.0;
        for i in 1..=steps {
            let r = max_m * i as f64 / steps as f64;
            // Never probe past a radius already beaten — the answer could not
            // improve on it, and a shorter loop is a cheaper advisory.
            if best.is_some_and(|b| prev_r >= b) {
                break;
            }
            let Some(there) = classify(p + dir * r, water_level_m, band_m, &height_at) else {
                prev_r = r;
                continue;
            };
            if there != here {
                // Bisect the bracket for a tighter radius. A fixed count, so the
                // work — and the answer — is identical on every machine.
                let (mut lo, mut hi) = (prev_r, r);
                for _ in 0..BISECTIONS {
                    let mid = 0.5 * (lo + hi);
                    match classify(p + dir * mid, water_level_m, band_m, &height_at) {
                        Some(c) if c == here => lo = mid,
                        _ => hi = mid,
                    }
                }
                best = Some(match best {
                    Some(b) => b.min(hi),
                    None => hi,
                });
                break;
            }
            prev_r = r;
        }
    }
    best
}

fn classify(
    p: DVec2,
    water_level_m: f64,
    band_m: f64,
    height_at: &impl Fn(DVec2) -> Option<f64>,
) -> Option<ShoreClass> {
    height_at(p).map(|h| shore_class(water_level_m, h, band_m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_is_banded_around_the_waterline() {
        assert_eq!(shore_class(10.0, 5.0, 0.5), ShoreClass::Submerged);
        assert_eq!(shore_class(10.0, 15.0, 0.5), ShoreClass::Dry);
        assert_eq!(shore_class(10.0, 10.2, 0.5), ShoreClass::Shoreline);
        assert_eq!(shore_class(10.0, 9.8, 0.5), ShoreClass::Shoreline);
        // A zero band degenerates to the strict comparison, with a tie on the line.
        assert_eq!(shore_class(10.0, 10.0, 0.0), ShoreClass::Shoreline);
        assert_eq!(shore_class(10.0, 9.999, 0.0), ShoreClass::Submerged);
        // A non-finite band is treated as zero rather than poisoning the answer.
        assert_eq!(shore_class(10.0, 12.0, f64::NAN), ShoreClass::Dry);
    }

    /// Known values: the blend is exactly 0 at and above the waterline, exactly 1
    /// a full band below it, and exactly ½ half-way — the smoothstep's fixed
    /// points, which is what a shader and a gameplay check must agree on.
    #[test]
    fn the_blend_has_known_values() {
        let band = 2.0;
        assert_eq!(shore_blend(10.0, 10.0, band), 0.0);
        assert_eq!(shore_blend(10.0, 11.0, band), 0.0);
        assert_eq!(shore_blend(10.0, 8.0, band), 1.0);
        assert_eq!(shore_blend(10.0, 5.0, band), 1.0);
        assert!((shore_blend(10.0, 9.0, band) - 0.5).abs() < 1e-12);
        // Quarter and three-quarter depth: smoothstep(0.25) and smoothstep(0.75).
        assert!((shore_blend(10.0, 9.5, band) - 0.15625).abs() < 1e-12);
        assert!((shore_blend(10.0, 8.5, band) - 0.84375).abs() < 1e-12);
        // Monotone non-decreasing as the ground drops.
        let mut prev = 0.0;
        for i in 0..=40 {
            let ground = 11.0 - i as f64 * 0.1;
            let b = shore_blend(10.0, ground, band);
            assert!(b >= prev - 1e-15, "blend fell as the ground dropped");
            prev = b;
        }
        // A zero band is a hard edge, not a NaN.
        assert_eq!(shore_blend(10.0, 9.0, 0.0), 1.0);
        assert_eq!(shore_blend(10.0, 11.0, 0.0), 0.0);
    }

    /// The exactness claim: a straight shoreline probed head-on. Ground rises
    /// linearly with +X and crosses the water level at x = 20; from the origin the
    /// nearest *class change* is where the band begins, at x = 18 for a 1 m band
    /// on a 0.5 m/m slope.
    #[test]
    fn distance_is_exact_for_a_straight_shore() {
        let ground = |p: DVec2| Some(-10.0 + p.x * 0.5); // = 0 at x = 20
        let d = shore_distance(DVec2::ZERO, 0.0, 1.0, 40.0, 64, ground).unwrap();
        // Submerged → Shoreline happens where depth == band, i.e. x = 18.
        assert!((d - 18.0).abs() < 0.15, "distance {d}");
        // …and the probe really did find a class change there.
        assert_eq!(
            shore_class(0.0, ground(DVec2::new(d + 0.2, 0.0)).unwrap(), 1.0),
            ShoreClass::Shoreline
        );
        assert_eq!(
            shore_class(0.0, ground(DVec2::new(d - 0.5, 0.0)).unwrap(), 1.0),
            ShoreClass::Submerged
        );
    }

    #[test]
    fn no_shore_within_range_answers_none() {
        // A flat bed 10 m under water, everywhere.
        let ground = |_: DVec2| Some(-10.0);
        assert!(shore_distance(DVec2::ZERO, 0.0, 0.5, 50.0, 32, ground).is_none());
        // …and a terrain that answers for nothing at all.
        assert!(shore_distance(DVec2::ZERO, 0.0, 0.5, 50.0, 32, |_| None).is_none());
        // A non-positive range is refused rather than looped over.
        assert!(shore_distance(DVec2::ZERO, 0.0, 0.5, 0.0, 32, ground).is_none());
    }

    #[test]
    fn distance_is_deterministic_and_direction_independent() {
        // A circular island of radius 30 centred at the origin: from 50 m out in
        // ANY direction the shore is the same distance away.
        let island = |p: DVec2| {
            let r = p.length();
            Some(if r < 30.0 { 5.0 } else { 5.0 - (r - 30.0) })
        };
        let mut seen: Vec<f64> = Vec::new();
        for k in 0..8 {
            let a = std::f64::consts::TAU * k as f64 / 8.0;
            let p = DVec2::new(pcos64(a), psin64(a)) * 50.0;
            let d = shore_distance(p, 0.0, 0.5, 40.0, 64, island).unwrap();
            // Repeated calls agree exactly.
            assert_eq!(
                d,
                shore_distance(p, 0.0, 0.5, 40.0, 64, island).unwrap(),
                "not deterministic"
            );
            seen.push(d);
        }
        let min = seen.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = seen.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            max - min < 1.0,
            "the same geometry gave {min}..{max} depending on where it was probed"
        );
    }
}
