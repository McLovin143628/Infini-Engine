//! Fixed-timestep accumulator for the runtime game loop (P9.1a).
//!
//! This mirrors the semantics of `inf_physics::stepper::FixedStepper` (variable
//! frame time → whole fixed steps + a render interpolation `alpha`, with a
//! spiral-of-death cap). It is re-implemented here rather than imported because a
//! runtime → inf-physics dependency would pull rapier (and its build weight) into
//! every player/PIE build, and `inf-physics` is owned by a concurrent task.
//!
//! **Follow-up (dedup):** hoist a single `FixedStepper` into a shared low crate
//! (e.g. `inf-core`) and have both `inf-physics` and `inf-runtime` reuse it, so
//! the accumulate/interpolate/spiral-guard logic is audited in exactly one place.

/// Fixed-timestep accumulator. Construct with a fixed `dt`, call
/// [`accumulate`](Self::accumulate) once per frame with the frame's elapsed time,
/// run that many fixed steps, then read [`alpha`](Self::alpha) for render
/// interpolation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedStep {
    fixed_dt: f64,
    accumulator: f64,
    max_steps: u32,
}

impl FixedStep {
    /// Default spiral-of-death cap: a stalled frame runs at most this many steps
    /// and drops the rest of the backlog rather than falling permanently behind.
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

    /// The fixed timestep in seconds.
    pub fn fixed_dt(&self) -> f64 {
        self.fixed_dt
    }

    /// Add a frame's elapsed time and return how many fixed steps to run now.
    ///
    /// Non-finite or negative `frame_dt` contributes zero. The count is clamped to
    /// `max_steps`; when clamped, the surplus backlog is discarded (spiral guard).
    #[must_use = "run this many fixed steps, then read alpha() for interpolation"]
    pub fn accumulate(&mut self, frame_dt: f64) -> u32 {
        if frame_dt.is_finite() && frame_dt > 0.0 {
            self.accumulator += frame_dt;
        }
        let mut full = (self.accumulator / self.fixed_dt).floor();
        if full <= 0.0 {
            return 0;
        }
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

    /// Unconsumed time still in the accumulator (< `fixed_dt` after a normal
    /// `accumulate`).
    pub fn remainder(&self) -> f64 {
        self.accumulator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_one_step_per_fixed_dt() {
        let mut s = FixedStep::from_hz(60.0);
        assert_eq!(s.accumulate(1.0 / 60.0), 1);
        assert!(s.alpha() < 1e-9);
    }

    #[test]
    fn accumulates_partial_frames() {
        let mut s = FixedStep::new(0.1);
        assert_eq!(s.accumulate(0.06), 0);
        assert!((s.alpha() - 0.6).abs() < 1e-9);
        assert_eq!(s.accumulate(0.06), 1);
        assert!((s.alpha() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn clamps_to_max_steps_no_spiral() {
        let mut s = FixedStep::with_max_steps(0.1, 4);
        assert_eq!(s.accumulate(10.0), 4);
        assert_eq!(s.accumulate(0.1), 1);
    }

    #[test]
    fn ignores_bad_frame_dt() {
        let mut s = FixedStep::new(0.1);
        assert_eq!(s.accumulate(-5.0), 0);
        assert_eq!(s.accumulate(f64::NAN), 0);
        assert_eq!(s.accumulate(f64::INFINITY), 0);
        assert_eq!(s.remainder(), 0.0);
    }
}
