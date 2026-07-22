//! The pure math builtins the `math.*` blueprint palette lowers to — the
//! **single source of truth** shared by the interpreter and the transpiled Rust
//! (ROADMAP B-P1). The interpreter's `math` dispatch calls these directly; the
//! generated Rust emits `math::<name>(..)` free calls that resolve to the very
//! same functions (`pub use inf_blueprint::math_builtins::*`), so
//! interpreter-vs-compiled parity holds for the whole math palette *by
//! construction* rather than by matching two hand-written implementations.
//!
//! Numeric contract:
//! - Float ops are `f64`. `floor`/`ceil`/`round`/`sqrt` are the correctly-rounded
//!   IEEE-754 std operations, so they are **bit-identical across platforms**.
//! - `sin`/`cos` route to [`inf_math::portable`]'s `psin64`/`pcos64` because
//!   `std` trig is **not** bit-portable across libms (the house law paid for on
//!   CI — see that module). Committed blueprint math therefore stays
//!   deterministic.
//! - `pow` uses [`f64::powf`], which is **not** guaranteed bit-identical across
//!   platforms' libm. That is fine for editor preview and ordinary gameplay
//!   math, but if committed / deterministic content ever depends on `pow` it
//!   needs a portable polynomial replacement (documented follow-up — the same
//!   caveat as the trig house law, only `powf` has no in-tree portable pair
//!   yet).
//! - `abs`/`min`/`max` come in an `f64` and an `i64` flavour so the interpreter
//!   can keep an all-`Int` computation in `Int` (mirroring the `arith` promotion
//!   rule) while the compiled path — whose math pins are all `Float` — uses the
//!   `f64` flavour.
//! - `min`/`max` are **NaN-absorbing** (they delegate to `f64::min`/`f64::max`,
//!   which return the non-NaN operand when exactly one is NaN).
//! - `clamp` is **non-panicking**: `x.max(lo).min(hi)`. This deliberately
//!   diverges from [`f64::clamp`], which panics when `lo > hi`; here an inverted
//!   range silently yields `hi`.
//! - `to_int` truncates toward zero and casts with Rust's **saturating** `as`
//!   semantics (NaN → 0, out-of-range / ±∞ → `i64::MIN` / `i64::MAX`).

use inf_math::portable::{pcos64, psin64};

/// `|x|` (f64).
pub fn abs(x: f64) -> f64 {
    x.abs()
}

/// `|x|` (i64), wrapping at `i64::MIN` (`abs(i64::MIN)` has no positive form).
pub fn abs_i64(x: i64) -> i64 {
    x.wrapping_abs()
}

/// Round toward −∞ (IEEE-exact).
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// Round toward +∞ (IEEE-exact).
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

/// Round half away from zero (IEEE-exact).
pub fn round(x: f64) -> f64 {
    x.round()
}

/// Non-negative square root (IEEE-exact; `sqrt(neg)` is NaN).
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Portable sine — routes to [`inf_math::portable::psin64`], NOT `std` (house law).
pub fn sin(x: f64) -> f64 {
    psin64(x)
}

/// Portable cosine — routes to [`inf_math::portable::pcos64`], NOT `std`.
pub fn cos(x: f64) -> f64 {
    pcos64(x)
}

/// Lesser of two f64s, **NaN-absorbing** (delegates to [`f64::min`]).
pub fn min(a: f64, b: f64) -> f64 {
    a.min(b)
}

/// Greater of two f64s, **NaN-absorbing** (delegates to [`f64::max`]).
pub fn max(a: f64, b: f64) -> f64 {
    a.max(b)
}

/// Lesser of two i64s.
pub fn min_i64(a: i64, b: i64) -> i64 {
    a.min(b)
}

/// Greater of two i64s.
pub fn max_i64(a: i64, b: i64) -> i64 {
    a.max(b)
}

/// `a` raised to the `b` power. **Not bit-portable** across libms — see module docs.
pub fn pow(a: f64, b: f64) -> f64 {
    a.powf(b)
}

/// Constrain `x` to `[lo, hi]`, non-panicking. An inverted range (`lo > hi`)
/// yields `hi` rather than panicking like [`f64::clamp`].
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Linear interpolation `a + (b − a) · t` (unclamped `t`).
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Truncate toward zero and convert to i64 with saturating `as` semantics
/// (NaN → 0, out-of-range → `i64::MIN` / `i64::MAX`).
pub fn to_int(x: f64) -> i64 {
    x.trunc() as i64
}

/// Widen an i64 to f64 (exact for magnitudes ≤ 2^53).
pub fn to_float(x: i64) -> f64 {
    x as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_int_and_float() {
        assert_eq!(abs(-2.5), 2.5);
        assert_eq!(abs_i64(-7), 7);
        // wrapping at the negative extreme (documented divergence from a panic).
        assert_eq!(abs_i64(i64::MIN), i64::MIN);
    }

    #[test]
    fn min_max_nan_absorbing() {
        // Exactly one NaN operand → the other value wins (f64::min/max contract).
        assert_eq!(min(f64::NAN, 1.0), 1.0);
        assert_eq!(min(1.0, f64::NAN), 1.0);
        assert_eq!(max(f64::NAN, 1.0), 1.0);
        assert_eq!(max(1.0, f64::NAN), 1.0);
        assert_eq!(min_i64(3, -3), -3);
        assert_eq!(max_i64(3, -3), 3);
    }

    #[test]
    fn clamp_inverted_range_yields_hi() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-1.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(99.0, 0.0, 10.0), 10.0);
        // Inverted range (lo > hi): std::clamp would panic; we yield hi.
        assert_eq!(clamp(5.0, 10.0, 0.0), 0.0);
    }

    #[test]
    fn lerp_endpoints_and_midpoint() {
        assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        // Unclamped t extrapolates.
        assert_eq!(lerp(0.0, 10.0, 2.0), 20.0);
    }

    #[test]
    fn to_int_saturates() {
        assert_eq!(to_int(3.9), 3);
        assert_eq!(to_int(-3.9), -3);
        assert_eq!(to_int(f64::NAN), 0);
        assert_eq!(to_int(f64::INFINITY), i64::MAX);
        assert_eq!(to_int(f64::NEG_INFINITY), i64::MIN);
        // Far beyond i64 range saturates rather than wrapping/UB.
        assert_eq!(to_int(1e30), i64::MAX);
    }

    #[test]
    fn to_float_widens() {
        assert_eq!(to_float(5), 5.0);
        assert_eq!(to_float(-5), -5.0);
    }

    #[test]
    fn floor_ceil_round_sqrt_are_std() {
        assert_eq!(floor(2.7), 2.0);
        assert_eq!(ceil(2.1), 3.0);
        assert_eq!(round(2.5), 3.0);
        assert_eq!(sqrt(9.0), 3.0);
    }

    #[test]
    fn trig_matches_std_within_tolerance() {
        // Portable trig tracks std to ~1e-7 across the tested range (the
        // inf_math::portable accuracy guarantee).
        let mut x = -6.0_f64;
        while x <= 6.0 {
            assert!((sin(x) - x.sin()).abs() < 1e-7, "sin({x})");
            assert!((cos(x) - x.cos()).abs() < 1e-7, "cos({x})");
            x += 0.05;
        }
    }

    #[test]
    fn trig_bit_exact_locked() {
        // A hard-locked output pins the portable evaluation order end-to-end
        // (any coefficient/reduction drift trips this). Mirrors the inf_math
        // lock; regenerate only against a reference reproducing the exact order.
        assert_eq!(sin(1.0).to_bits(), 0x3FEA_ED54_8EF3_1577);
        assert_eq!(cos(2.0).to_bits(), 0xBFDA_A226_5753_71D5);
    }

    #[test]
    fn pow_basic() {
        assert_eq!(pow(2.0, 10.0), 1024.0);
        assert_eq!(pow(9.0, 0.5), 3.0);
    }
}
