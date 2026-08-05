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
//! * [`pcbrt`] — f64 cube root by Newton, added in P22.4. The law is **not only
//!   about trigonometry**: `cbrt` is a libm call too, and on `wasm32` the standard
//!   library routes it through the `libm` crate, so a browser client and a native
//!   one disagree by an ulp. Anything two *machines* are claimed to agree about
//!   goes through this.

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
}
