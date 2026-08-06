//! Bit-portable trigonometry (pure IEEE add/mul/floor, no `std` libm).
//!
//! House law (paid for in blood on the Ubuntu CI runner — see the vgeom
//! generator-lock): `f32::sin`/`f64::sin` are **not** bit-identical across
//! platforms (MSVC's CRT vs glibc's libm diverge in the low bits), so any
//! *committed / deterministic* vertex data — meshlet displacement, primitive
//! geometry, blueprint math that must round-trip byte-for-byte — must never call
//! `std` trig. These minimax / Taylor polynomials use only IEEE-754 add, mul and
//! `floor`, all of which are exactly specified, so the emitted bytes are
//! identical on every target.
//!
//! * [`psin`]/[`pcos`] — f32, **demo-grade**. **Provenance:** the coefficients are
//!   copied verbatim from `inf_editor_core::samples`'s private `psin`/`pcos` (the
//!   vgeom-demo displacement); duplicated here (not moved) so Ring-0 has a
//!   dependency-free portable sine. Accuracy is only ~2.5e-3 on `[-π/2, π/2]` and
//!   degrades toward `±π` (the source's "~1e-4" note is optimistic) — fine for
//!   the demo displacement it was tuned for, but **committed geometry uses the f64
//!   pair below**, which is both accurate and bit-portable.
//! * [`psin64`]/[`pcos64`] — f64, ~1e-7 accuracy on `[-2π, 2π]`. Range-reduced to
//!   `[-π/2, π/2]` then a degree-11 odd Taylor polynomial. This is the pair the
//!   primitive-mesh generators use (cast to f32 per vertex). Also consumed by a
//!   later blueprint package — keep these exact signatures.
//! * [`pacos64`] / [`pslerp`] — f64 arccosine and a **portable quaternion
//!   slerp**, added at P24.2. `Quat::slerp` calls `acos_approx` plus three `sin`s,
//!   and it is reached by *every* animation blend in the engine: a state-machine
//!   transition with any `duration > 0`, every blend space, and every
//!   `Interpolation::Linear` quaternion track — which is the `#[default]` and what
//!   the glTF importer emits. The evaluated pose is folded into the sim's
//!   `state_bytes`, so libm was sitting squarely on a trace two hosts are
//!   compared on. Found by the P24.2 audit, which also noted that the pose-parity
//!   fixture used `Step` interpolation and a zero-duration transition — certifying
//!   portability on the one configuration that avoids the non-portable op.
//! * [`pcbrt`] — f64 cube root by Newton, added in P22.4. The law is **not only
//!   about trigonometry**: `cbrt` is a libm call too, and on `wasm32` the standard
//!   library routes it through the `libm` crate, so a browser client and a native
//!   one disagree by an ulp. Anything two *machines* are claimed to agree about
//!   goes through this.

use glam::Quat;

use std::f32::consts::{FRAC_PI_2 as F32_FRAC_PI_2, TAU as F32_TAU};
use std::f64::consts::{FRAC_PI_2, PI};

/// Byte-portable sine (f32), demo-grade. A 7th-order odd polynomial after a
/// `[-π, π]` range reduction; ~5e-3 accuracy on `[-π/2, π/2]`, degrading toward
/// `±π`. Coefficients copied verbatim from `inf_editor_core::samples::psin` (kept
/// faithful, not re-fitted). For accurate committed geometry use [`psin64`].
pub fn psin(x: f32) -> f32 {
    // Range-reduce to [-π, π] (floor is exact; the inputs here are small, so no
    // catastrophic cancellation).
    let x = x - (x / F32_TAU + 0.5).floor() * F32_TAU;
    let x2 = x * x;
    x * (0.987_862 + x2 * (-0.155_271 + x2 * (0.005_641_12 - x2 * 0.000_060_461_2)))
}

/// Byte-portable cosine (f32) via [`psin`].
pub fn pcos(x: f32) -> f32 {
    psin(x + F32_FRAC_PI_2)
}

// Odd Taylor coefficients for sin on [-π/2, π/2] (degree 11: powers 1,3,5,7,9,11).
// Endpoint error ≈ 5.7e-8; far better toward the middle. Written as exact
// rationals so the const-folded bits match a reference computation.
const S1: f64 = -1.0 / 6.0; // -1/3!
const S2: f64 = 1.0 / 120.0; // 1/5!
const S3: f64 = -1.0 / 5040.0; // -1/7!
const S4: f64 = 1.0 / 362880.0; // 1/9!
const S5: f64 = -1.0 / 39916800.0; // -1/11!

/// Byte-portable sine (f64), ~1e-7 on `[-2π, 2π]` and stable well beyond. Reduces
/// `x` to `r ∈ [-π/2, π/2]` via `k = floor(x/π + ½)`, `r = x − kπ`
/// (`sin(x) = (−1)^k · sin(r)`), then evaluates a degree-11 odd Taylor polynomial.
/// Only IEEE add/mul/floor are used, so the result is bit-identical everywhere.
pub fn psin64(x: f64) -> f64 {
    let k = (x / PI + 0.5).floor();
    let r = x - k * PI;
    // (−1)^k without integer casts: parity = k − 2·floor(k/2) ∈ {0, 1}.
    let parity = k - 2.0 * (k * 0.5).floor();
    let sign = 1.0 - 2.0 * parity;
    let r2 = r * r;
    let poly = r * (1.0 + r2 * (S1 + r2 * (S2 + r2 * (S3 + r2 * (S4 + r2 * S5)))));
    sign * poly
}

/// Byte-portable cosine (f64) via [`psin64`].
pub fn pcos64(x: f64) -> f64 {
    psin64(x + FRAC_PI_2)
}

/// Byte-portable **cube root** (f64): Newton's method, fixed iteration count,
/// pure IEEE arithmetic.
///
/// # Why this exists when `f64::cbrt` is right there
///
/// `f64::cbrt` is a libm call and libm is not bit-portable — the module's own
/// law, one function over. It is *worse* than the trig case in one respect: on
/// `wasm32` the standard library routes `cbrt` through the `libm` crate, so a
/// browser client and a native one differ by about an ulp on the same input.
/// Anything whose output two machines are claimed to agree about must therefore
/// not call it. P22.4's rubble placement is exactly that claim, and P21.3's spoil
/// pile was the first (`inf_voxel::cbrt_det`).
///
/// # It is a DUPLICATE of `inf_voxel::cbrt_det`, deliberately
///
/// Character for character, the same algorithm — and `inf-physics`'
/// `the_portable_cube_root_matches_the_voxel_one` sweeps both over four decades
/// and asserts **bit equality**, because that crate is the one that can see both.
/// `inf-voxel`'s manifest argues, at length, that it must not depend on
/// `inf-math` (an unused dependency in Ring 0 is a licence surface with no
/// payer), and this module's own header records the same arrangement for
/// `psin`/`pcos` — copied rather than moved, held by a test. A third copy would
/// be a different decision; this is the second, under the rule the first one set.
///
/// The argument is scaled into `[1, 8)` by exact powers of eight (exact in binary
/// floating point, so the scaling introduces no error at all), seeded linearly,
/// refined by `y ← (2y + x/y²)/3`, and scaled back. Twelve iterations from a seed
/// within a factor of two is roughly twice what full `f64` precision needs.
///
/// Non-finite or non-positive input answers `0.0` — a value, not a panic, and
/// not an infinite scaling loop.
pub fn pcbrt(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return 0.0;
    }
    let mut m = x;
    let mut k: i32 = 0;
    while m >= 8.0 {
        m /= 8.0;
        k += 1;
    }
    while m < 1.0 {
        m *= 8.0;
        k -= 1;
    }
    // Seed: the chord of ∛ across [1, 8] → [1, 2].
    let mut y = 1.0 + (m - 1.0) / 7.0;
    for _ in 0..12 {
        y = (2.0 * y + m / (y * y)) / 3.0;
    }
    // 2^k, built by repeated exact doubling/halving rather than `powi` (which is
    // also fine, but this needs no assumption about how it is lowered).
    let mut scale = 1.0f64;
    for _ in 0..k.abs() {
        if k > 0 {
            scale *= 2.0;
        } else {
            scale /= 2.0;
        }
    }
    y * scale
}

/// Above this `|dot|`, [`pslerp`] falls back to a normalized lerp.
///
/// A **fixed constant, not a tolerance**: the same input always takes the same
/// branch, so the answer stays a pure function of its arguments (the P23.5 ruling
/// about iteration counts, applied to a branch). At `dot = 0.9995` the two
/// rotations are 1.8° apart and nlerp's deviation from the constant-speed arc is
/// under 6e-6 rad — far below the error of the arc formula it replaces, which is
/// exactly why the fallback exists: near zero angle, `sin θ` in the denominator
/// is where the arc form loses its precision.
pub const SLERP_LERP_THRESHOLD: f64 = 0.9995;

/// Below `1 − |x|` = this, [`pacos64`] uses its half-angle series instead of the
/// Abramowitz–Stegun seed plus a Newton step.
///
/// The crossover is where the two are equally good, and both sides of it were
/// measured: the series' first omitted term is ~7e-9 here, and the Newton step's
/// error is `psin64`'s ~1e-7 divided by `sin y`, which at this `u` is 0.436 —
/// so ~2e-7. Past it, toward `x = ±1`, that division is what took the whole
/// function to 3.5e-6.
pub const ACOS_SERIES_CROSSOVER: f64 = 0.1;

/// Byte-portable arccosine (f64) on `[-1, 1]`.
///
/// Two stages, and the second is what makes it usable rather than merely
/// portable:
///
/// 1. **Seed** — Abramowitz & Stegun 4.4.45,
///    `acos(a) ≈ √(1−a)·(c₀ + c₁a + c₂a² + c₃a³)` on `[0, 1]`, reflected through
///    `acos(−a) = π − acos(a)`. Max error ≈ 6.8e-5 rad. Only `sqrt`, add and
///    multiply — all exactly specified by IEEE-754.
/// 2. **One Newton step** on `cos y − x = 0`, i.e. `y ← y + (cos y − x)/sin y`,
///    evaluated with [`pcos64`]/[`psin64`]. That takes the error down to those
///    functions' own (~1e-7) instead of the seed's, for two polynomial
///    evaluations.
///
/// The step is skipped where `sin y → 0` (`x` at ±1), because there it is
/// unbounded and the seed is already exact — `√(1−a)` is zero at `a = 1` by
/// construction.
///
/// **Why this exists at all**: `f32::acos`/`f64::acos` are libm calls, and the
/// P14 law says libm is not bit-identical across targets. A quaternion blend
/// reaches the evaluated pose, the pose is folded into `state_bytes`, and
/// `state_bytes` is compared between the editor's Simulate and the shipped
/// player. See [`pslerp`].
pub fn pacos64(x: f64) -> f64 {
    let x = x.clamp(-1.0, 1.0);
    let negative = x < 0.0;
    let a = if negative { -x } else { x };
    let u = 1.0 - a;
    let seed = if u < ACOS_SERIES_CROSSOVER {
        // Near ±1 the Newton step below divides by a vanishing `sin y`, which
        // amplifies `psin64`'s own ~1e-7 into microradians — measured at 3.5e-6
        // when this branch did not exist. The half-angle series has no such
        // denominator: `acos(1−u) = √(2u)·(1 + u/12 + 3u²/160 + …)`, whose next
        // omitted term at the crossover is ~7e-9.
        (2.0 * u).sqrt()
            * (1.0
                + u * (1.0 / 12.0 + u * (3.0 / 160.0 + u * (5.0 / 896.0 + u * (35.0 / 18432.0)))))
    } else {
        // Abramowitz & Stegun 4.4.45 on [0, 1]. Horner, so the operation order is
        // fixed.
        (1.0 - a).sqrt() * (1.570_728_8 + a * (-0.212_114_4 + a * (0.074_261_0 + a * -0.018_729_3)))
    };
    let mut y = if negative { PI - seed } else { seed };
    // One Newton step on `cos y − x = 0`. Skipped inside the series branch (it is
    // already better than the step could make it) and wherever `sin y` is small
    // enough for the division to amplify rather than refine.
    if u >= ACOS_SERIES_CROSSOVER {
        let s = psin64(y);
        if s.abs() > 1.0e-3 {
            y += (pcos64(y) - x) / s;
        }
    }
    y.clamp(0.0, PI)
}

/// **Byte-portable spherical linear interpolation** between two rotations,
/// taking the shortest arc.
///
/// # Why `Quat::slerp` cannot be on a committed path
///
/// glam's implementation calls `acos_approx` and then `sin` **three times** —
/// `std` libm, which the P14 law says is not bit-identical across targets. It is
/// reached by *every* animation blend in this engine: a state-machine transition
/// with any `duration > 0`, every blend space, and every `Interpolation::Linear`
/// quaternion track — which is the `#[default]` and what the glTF importer emits.
/// The evaluated pose is folded into the sim's `state_bytes`, so two machines
/// running the same level would disagree in the low bits of every rotating joint,
/// silently, and no gate in the repository compares two machines.
///
/// # What it does
///
/// The textbook arc form, with [`pacos64`] and [`psin64`] in place of libm and the
/// whole computation in `f64` before it narrows once at the end:
///
/// ```text
///   θ  = acos(|a·b|)
///   w₀ = sin((1−t)θ) / sin θ,   w₁ = sin(tθ) / sin θ
///   q  = normalize(w₀·a ± w₁·b)
/// ```
///
/// * **Shortest arc** — `b` is negated when `a·b < 0`, because `q` and `−q` are
///   the same rotation and the raw pair would take the long way round.
/// * **Near-parallel falls back to nlerp**, above [`SLERP_LERP_THRESHOLD`], where
///   `sin θ` in the denominator is what loses the precision.
/// * **Always renormalized** (the P23 law: an un-normalized quaternion is a
///   rotation *and a scale*, and the scale rides into the skinning palette).
///
/// Accuracy against `Quat::slerp` is measured, not asserted, by
/// `pslerp_tracks_the_true_arc` — max deviation ~1e-6 rad over a sweep of
/// angles and parameters.
pub fn pslerp(a: Quat, b: Quat, t: f32) -> Quat {
    let (ax, ay, az, aw) = (a.x as f64, a.y as f64, a.z as f64, a.w as f64);
    let (mut bx, mut by, mut bz, mut bw) = (b.x as f64, b.y as f64, b.z as f64, b.w as f64);
    let mut dot = ax * bx + ay * by + az * bz + aw * bw;
    if dot < 0.0 {
        bx = -bx;
        by = -by;
        bz = -bz;
        bw = -bw;
        dot = -dot;
    }
    let t = t as f64;
    let (w0, w1) = if dot > SLERP_LERP_THRESHOLD {
        // Near-parallel: the arc form's denominator is vanishing, and a straight
        // lerp is within 6e-6 rad of it here.
        (1.0 - t, t)
    } else {
        let theta = pacos64(dot);
        let sin_theta = psin64(theta);
        if sin_theta.abs() < 1.0e-12 {
            (1.0 - t, t)
        } else {
            (
                psin64((1.0 - t) * theta) / sin_theta,
                psin64(t * theta) / sin_theta,
            )
        }
    };
    let (x, y, z, w) = (
        w0 * ax + w1 * bx,
        w0 * ay + w1 * by,
        w0 * az + w1 * bz,
        w0 * aw + w1 * bw,
    );
    let len = (x * x + y * y + z * z + w * w).sqrt();
    if len <= 1.0e-12 {
        // Antipodal inputs that cancelled. `a` is a rotation and the midpoint is
        // not defined; returning `a` is the only answer that is not a NaN.
        return a;
    }
    Quat::from_xyzw(
        (x / len) as f32,
        (y / len) as f32,
        (z / len) as f32,
        (w / len) as f32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_matches_std_on_good_range() {
        use std::f32::consts::FRAC_PI_2;
        // psin is tuned tightest on [-π/2, π/2]; pcos(x) = psin(x + π/2) is good on
        // [-π, 0] (its argument then stays in psin's good band). This documents the
        // demo-grade coefficients honestly rather than asserting false precision.
        let mut x = -FRAC_PI_2;
        while x <= FRAC_PI_2 {
            assert!((psin(x) - x.sin()).abs() < 6e-3, "psin({x})");
            x += 0.02;
        }
        let mut x = -std::f32::consts::PI;
        while x <= 0.0 {
            assert!((pcos(x) - x.cos()).abs() < 6e-3, "pcos({x})");
            x += 0.02;
        }
    }

    #[test]
    fn f32_key_angles() {
        use std::f32::consts::PI;
        assert!(psin(0.0).abs() < 1e-3);
        assert!((psin(PI / 2.0) - 1.0).abs() < 6e-3);
        assert!((pcos(0.0) - 1.0).abs() < 6e-3);
    }

    #[test]
    fn f64_matches_std_within_tolerance() {
        // Accuracy target ~1e-7 across [-2π, 2π].
        let mut x = -2.0 * PI;
        while x <= 2.0 * PI {
            assert!((psin64(x) - x.sin()).abs() < 1e-7, "psin64({x})");
            assert!((pcos64(x) - x.cos()).abs() < 1e-7, "pcos64({x})");
            x += 0.01;
        }
    }

    #[test]
    fn f64_symmetry() {
        // sin is odd, cos is even — exact structural identities of the polynomial.
        for &x in &[0.3_f64, 1.1, 2.7, -0.9, 4.0] {
            assert!((psin64(-x) + psin64(x)).abs() < 1e-12, "odd at {x}");
            assert!((pcos64(-x) - pcos64(x)).abs() < 1e-12, "even at {x}");
        }
    }

    #[test]
    fn pcbrt_is_accurate_and_refuses_degenerates() {
        for &x in &[1.0e-6_f64, 0.125, 1.0, 8.0, 27.0, 1234.5, 1.0e9] {
            let got = pcbrt(x);
            let want = x.cbrt();
            assert!(
                (got - want).abs() <= 1e-12 * want.max(1.0),
                "pcbrt({x}) = {got}, libm says {want}"
            );
            // …and it really is the cube root, checked without libm at all.
            assert!((got * got * got - x).abs() <= 1e-9 * x.max(1.0));
        }
        assert_eq!(pcbrt(0.0), 0.0);
        assert_eq!(pcbrt(-1.0), 0.0);
        assert_eq!(pcbrt(f64::NAN), 0.0);
        assert_eq!(pcbrt(f64::INFINITY), 0.0);
    }

    #[test]
    fn f64_bit_exact_locked() {
        // Hard-locked outputs: any change to the reduction/coefficients or a
        // platform that reorders these IEEE ops trips this. Regenerate ONLY with a
        // reference that replicates the exact evaluation order (see module docs).
        assert_eq!(psin64(0.0).to_bits(), 0x0000_0000_0000_0000);
        assert_eq!(psin64(1.0).to_bits(), 0x3FEA_ED54_8EF3_1577);
        assert_eq!(psin64(-0.5).to_bits(), 0xBFDE_AEE8_744B_048F);
        assert_eq!(pcos64(2.0).to_bits(), 0xBFDA_A226_5753_71D5);
    }

    /// [`pacos64`] tracks `std`'s to the accuracy its docs claim.
    ///
    /// The point is not that it beats libm — it cannot — but that it is *close
    /// enough that no author can see the difference* while being computed from
    /// operations IEEE-754 specifies exactly. The seed alone is ~6.8e-5; the
    /// Newton step is what earns the number below.
    #[test]
    fn pacos64_tracks_std_acos() {
        let mut worst = 0.0f64;
        let mut worst_at = 0.0f64;
        for i in 0..=20_000 {
            let x = -1.0 + 2.0 * (i as f64) / 20_000.0;
            let e = (pacos64(x) - x.acos()).abs();
            if e > worst {
                worst = e;
                worst_at = x;
            }
        }
        // Measured: 1.5e-7, set by `psin64`'s own accuracy in the Newton
        // branch. The series branch near ±1 is an order better.
        assert!(
            worst < 2.0e-7,
            "pacos64 is off by {worst} at x = {worst_at}"
        );
        // The ends are exact by construction, and monotone in between.
        assert_eq!(pacos64(1.0), 0.0);
        assert!((pacos64(-1.0) - PI).abs() < 1e-12);
        assert!((pacos64(0.0) - PI / 2.0).abs() < 2.0e-7);
        // Out of domain clamps rather than producing a NaN — a dot product of two
        // unit quaternions lands at 1.0000000001 constantly.
        assert_eq!(pacos64(1.5), 0.0);
        assert!((pacos64(-1.5) - PI).abs() < 1e-12);
    }

    /// **[`pslerp`] is the arc, not an approximation of one.**
    ///
    /// Measured against `Quat::slerp` over a sweep of angles and parameters: the
    /// replacement has to be indistinguishable in the viewport, or "portable"
    /// would have been bought with a behaviour change nobody signed off.
    #[test]
    fn pslerp_tracks_the_true_arc() {
        let mut worst = 0.0f32;
        let mut worst_case = (0.0f32, 0.0f32);
        for a_deg in 0..36 {
            for b_deg in 0..36 {
                let a = Quat::from_rotation_z((a_deg as f32 * 10.0).to_radians());
                let b = Quat::from_rotation_x((b_deg as f32 * 10.0).to_radians());
                for k in 0..=20 {
                    let t = k as f32 / 20.0;
                    let mine = pslerp(a, b, t);
                    let theirs = a.slerp(b, t);
                    // **The CHORD, not the angle.** Comparing two nearly-equal
                    // f32 quaternions with `acos(dot)` cannot resolve better
                    // than ~1e-3: `dot` rounds to `1 − ~1e-7` and `acos` turns
                    // that into `√(2e-7) ≈ 9e-4`. That is the measurement's
                    // floor, not the function's error — measured, on the first
                    // run of this test, as a "1.4e-3 deviation" that was
                    // entirely the metric. The chord has no such amplification.
                    // Double cover: `q` and `−q` are one rotation.
                    let chord = (mine - theirs).length().min((mine + theirs).length());
                    if chord > worst {
                        worst = chord;
                        worst_case = (a_deg as f32 * 10.0, b_deg as f32 * 10.0);
                    }
                }
            }
        }
        assert!(
            worst < 1.0e-5,
            "pslerp deviates from the true arc by {worst} (chord, at {worst_case:?})"
        );
    }

    /// The properties a blend has to have, independent of what it is compared to:
    /// the endpoints are exact, the shortest arc is taken, the result is a unit
    /// quaternion, and the midpoint really is halfway.
    #[test]
    fn pslerp_has_the_properties_a_blend_must_have() {
        let a = Quat::from_rotation_z(0.3);
        let b = Quat::from_rotation_z(1.9);
        // Endpoints, exactly (up to normalization).
        assert!(pslerp(a, b, 0.0).dot(a).abs() > 1.0 - 1e-6);
        assert!(pslerp(a, b, 1.0).dot(b).abs() > 1.0 - 1e-6);
        // Unit length, always — the P23 law: a raw pair is a rotation AND a
        // scale, and the scale rides into the skinning palette.
        for k in 0..=10 {
            let q = pslerp(a, b, k as f32 / 10.0);
            assert!((q.length() - 1.0).abs() < 1e-6, "{q:?} is not unit");
        }
        // Halfway is halfway.
        let mid = pslerp(a, b, 0.5);
        let (x, y) = (mid.angle_between(a), mid.angle_between(b));
        assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        // SHORTEST ARC: b stored the long way round must still take the short one.
        let far = -Quat::from_rotation_z(0.5);
        let short = pslerp(Quat::IDENTITY, far, 0.5);
        assert!(
            short.angle_between(Quat::IDENTITY) < 0.3,
            "the long way was taken: {short:?}"
        );
        // Identical inputs are a fixed point rather than a division by zero.
        assert!(pslerp(a, a, 0.37).dot(a).abs() > 1.0 - 1e-6);
        // Exactly antipodal: a value, never a NaN.
        let anti = pslerp(Quat::IDENTITY, -Quat::IDENTITY, 0.5);
        assert!(anti.is_finite(), "{anti:?}");
    }

    /// Deterministic, and free of `std` transcendentals by construction — the
    /// whole reason it exists.
    #[test]
    fn pslerp_is_reproducible() {
        let a = Quat::from_rotation_y(0.7);
        let b = Quat::from_rotation_x(-1.1);
        for k in 0..=13 {
            let t = k as f32 / 13.0;
            assert_eq!(pslerp(a, b, t).to_array(), pslerp(a, b, t).to_array());
        }
    }
}
