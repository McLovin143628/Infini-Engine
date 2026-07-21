//! A pure, dimension-agnostic fixed-timestep accumulator.
//!
//! The engine simulates on a **fixed** `dt` (determinism depends on it) but is
//! driven by a **variable** frame time. [`FixedStepper`] bridges the two: feed it
//! the wall-clock delta of a frame and it tells you how many fixed steps to run
//! this frame, carrying the remainder forward. [`FixedStepper::alpha`] then gives
//! the interpolation factor render code uses to blend the previous and current
//! physics poses so motion looks smooth between fixed steps.
//!
//! It is deliberately pure — it holds no clock and does no I/O — so the editor's
//! Simulate loop and `inf-runtime`'s game loop (P9) share one audited
//! implementation instead of each reinventing the accumulate/interpolate dance
//! (and each getting the spiral-of-death guard subtly wrong).

/// Fixed-timestep accumulator. Construct with the fixed `dt`, call
/// [`accumulate`](Self::accumulate) once per frame with the frame's elapsed time,
/// run that many fixed steps, then read [`alpha`](Self::alpha) for render
/// interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedStepper {
    fixed_dt: f64,
    accumulator: f64,
    max_steps: u32,
}

impl FixedStepper {
    /// Default cap on the number of fixed steps a single frame may trigger. This
    /// is the spiral-of-death guard: if a frame stalls badly, we run at most this
    /// many steps and drop the rest of the backlog rather than fall permanently
    /// behind (which would make each frame ever longer).
    pub const DEFAULT_MAX_STEPS: u32 = 8;

    /// A stepper running at `hz` fixed updates per second (e.g. `60.0`).
    ///
    /// # Panics
    /// Panics if `hz` is not finite and positive.
    pub fn from_hz(hz: f64) -> Self {
        assert!(hz.is_finite() && hz > 0.0, "hz must be finite and positive");
        Self::new(1.0 / hz)
    }

    /// A stepper with the given fixed `dt` in seconds.
    ///
    /// # Panics
    /// Panics if `fixed_dt` is not finite and positive.
    pub fn new(fixed_dt: f64) -> Self {
        Self::with_max_steps(fixed_dt, Self::DEFAULT_MAX_STEPS)
    }

    /// A stepper with an explicit spiral-of-death cap.
    ///
    /// # Panics
    /// Panics if `fixed_dt` is not finite and positive, or `max_steps` is 0.
    pub fn with_max_steps(fixed_dt: f64, max_steps: u32) -> Self {
        assert!(
            fixed_dt.is_finite() && fixed_dt > 0.0,
            "fixed_dt must be finite and positive"
        );
        assert!(max_steps > 0, "max_steps must be at least 1");
        Self {
            fixed_dt,
            accumulator: 0.0,
            max_steps,
        }
    }

    /// The fixed timestep in seconds — the `dt` to hand to
    /// [`step`](crate::d2::PhysicsWorld2D::step).
    pub fn fixed_dt(&self) -> f64 {
        self.fixed_dt
    }

    /// Add a frame's elapsed time and return how many fixed steps to run now.
    ///
    /// Non-finite or negative `frame_dt` contributes zero (a paused or glitched
    /// frame must never rewind or explode the accumulator). The count is clamped
    /// to `max_steps`; when clamped, the surplus backlog is discarded so the sim
    /// cannot enter a spiral of death.
    #[must_use = "run this many fixed steps, then read alpha() for interpolation"]
    pub fn accumulate(&mut self, frame_dt: f64) -> u32 {
        if frame_dt.is_finite() && frame_dt > 0.0 {
            self.accumulator += frame_dt;
        }
        // `accumulator` only ever grows by finite positive deltas and `fixed_dt`
        // is validated finite-positive at construction, so `full` is a finite
        // non-negative whole number here.
        let mut full = (self.accumulator / self.fixed_dt).floor();
        if full <= 0.0 {
            return 0; // nothing to run this frame
        }
        // Remove *every* whole step from the accumulator, keeping only the
        // sub-step remainder (which drives `alpha`). When `full` exceeds
        // `max_steps` the surplus is thereby dropped from the accumulator, not
        // carried forward — this is the spiral-of-death guard: a long stall runs
        // at most `max_steps` steps and abandons the rest of the backlog.
        self.accumulator -= full * self.fixed_dt;
        if self.accumulator < 0.0 {
            self.accumulator = 0.0;
        }
        if full > self.max_steps as f64 {
            full = self.max_steps as f64;
        }
        full as u32
    }

    /// The interpolation factor in `[0, 1)`: how far the accumulator has advanced
    /// toward the next fixed step. Render code blends `prev` and `curr` poses by
    /// this fraction so on-screen motion is smooth between fixed updates.
    pub fn alpha(&self) -> f64 {
        (self.accumulator / self.fixed_dt).clamp(0.0, 1.0)
    }

    /// Unconsumed time still in the accumulator (less than `fixed_dt` after a
    /// normal `accumulate`).
    pub fn remainder(&self) -> f64 {
        self.accumulator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_one_step_per_fixed_dt() {
        let mut s = FixedStepper::from_hz(60.0);
        // Exactly one tick's worth of time → exactly one step.
        assert_eq!(s.accumulate(1.0 / 60.0), 1);
        assert!(s.alpha() < 1e-9, "alpha should reset after a whole step");
    }

    #[test]
    fn accumulates_partial_frames() {
        let mut s = FixedStepper::new(0.1);
        assert_eq!(s.accumulate(0.06), 0); // 0.06 < 0.1 → no step yet
        assert!((s.alpha() - 0.6).abs() < 1e-9);
        assert_eq!(s.accumulate(0.06), 1); // 0.12 total → one step, 0.02 left
        assert!((s.alpha() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn multiple_steps_in_one_frame() {
        let mut s = FixedStepper::new(0.1);
        assert_eq!(s.accumulate(0.35), 3); // 0.05 remainder
        assert!((s.remainder() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn clamps_to_max_steps_no_spiral() {
        let mut s = FixedStepper::with_max_steps(0.1, 4);
        // A 10s stall would be 100 steps; clamp to 4 and drop the rest.
        assert_eq!(s.accumulate(10.0), 4);
        // Backlog discarded: the next normal frame is back to a single step.
        assert_eq!(s.accumulate(0.1), 1);
    }

    #[test]
    fn ignores_bad_frame_dt() {
        let mut s = FixedStepper::new(0.1);
        assert_eq!(s.accumulate(-5.0), 0);
        assert_eq!(s.accumulate(f64::NAN), 0);
        assert_eq!(s.accumulate(f64::INFINITY), 0);
        assert_eq!(s.remainder(), 0.0);
    }

    #[test]
    fn alpha_is_bounded() {
        let mut s = FixedStepper::new(0.1);
        let _ = s.accumulate(0.099);
        assert!(s.alpha() >= 0.0 && s.alpha() < 1.0);
    }
}
