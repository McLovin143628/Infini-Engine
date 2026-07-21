//! The runtime game loop (P9.1a, ROADMAP P9.1 item 1).
//!
//! [`GameLoop`] owns the pieces the roadmap assigns it: a fixed-step accumulator
//! ([`FixedStep`]), an ECS [`EcsWorld`], and a parallel sim [`SimSchedule`]. It
//! advances the world on a **fixed** `dt` while being driven by a **variable**
//! frame time, and exposes the render interpolation `alpha`.
//!
//! It is deliberately **headless**: no wall clock (the caller passes frame deltas)
//! and no rendering dependency, so it runs identically in a test, a bench, a
//! headless CI player, and (in P9.3) behind a real window. The determinism of a
//! step is a property of the schedule's access discipline, so
//! [`ScheduleMode::Serial`] and [`ScheduleMode::Parallel`] advance the world
//! identically — proven by the [`crate::replay`] harness (the §8 CI gate).

use inf_ecs::{sim, EcsWorld, ScheduleMode, SimConfig, SimSchedule};

use crate::step::FixedStep;

/// A headless, fixed-step game loop over an ECS world driven by a parallel sim
/// schedule.
pub struct GameLoop {
    world: EcsWorld,
    schedule: SimSchedule,
    stepper: FixedStep,
    /// Total fixed steps advanced since construction (the sim frame count).
    steps: u64,
}

impl GameLoop {
    /// A game loop over `world` running the standard engine sim schedule at
    /// `config.fixed_dt`, in the given execution mode. Installs the sim resources.
    pub fn new(mut world: EcsWorld, config: SimConfig, mode: ScheduleMode) -> Self {
        sim::install_resources(&mut world, config);
        world.propagate();
        Self {
            world,
            schedule: SimSchedule::engine_default(mode),
            stepper: FixedStep::new(config.fixed_dt),
            steps: 0,
        }
    }

    /// A game loop with a caller-supplied schedule (e.g. propagation-only, or a
    /// specific [`ScheduleMode`]). The world must already be seeded; sim resources
    /// are still installed here.
    pub fn with_schedule(mut world: EcsWorld, config: SimConfig, schedule: SimSchedule) -> Self {
        sim::install_resources(&mut world, config);
        world.propagate();
        Self {
            world,
            schedule,
            stepper: FixedStep::new(config.fixed_dt),
            steps: 0,
        }
    }

    /// Advance exactly one fixed step, regardless of the accumulator (for tests
    /// and the replay harness). Returns the new total step count.
    pub fn step_once(&mut self) -> u64 {
        // Sim frame-budget span (P15.1); the inf-ecs per-phase `trace` spans nest
        // inside it under a Tracy capture. Free at the default `info` filter when
        // no subscriber records it.
        let _span = tracing::info_span!("sim_step", step = self.steps).entered();
        self.schedule.run(&mut self.world);
        self.steps += 1;
        self.steps
    }

    /// Drive the loop with one frame's elapsed time: accumulate, run the whole
    /// fixed-step backlog this frame permits (spiral-capped), and return how many
    /// steps ran. Read [`alpha`](Self::alpha) afterward for render interpolation.
    ///
    /// No wall clock is consulted — `frame_dt` is the caller's measured delta.
    pub fn run_frame(&mut self, frame_dt: f64) -> u32 {
        let n = self.stepper.accumulate(frame_dt);
        for _ in 0..n {
            self.step_once();
        }
        n
    }

    /// Render interpolation factor in `[0, 1)`: how far the accumulator sits toward
    /// the next fixed step. The renderer (P9.3) blends previous/current poses by
    /// this to smooth motion between fixed updates.
    pub fn alpha(&self) -> f64 {
        self.stepper.alpha()
    }

    /// Total fixed steps advanced since construction.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// The fixed timestep in seconds.
    pub fn fixed_dt(&self) -> f64 {
        self.stepper.fixed_dt()
    }

    /// The current execution mode of the sim schedule.
    pub fn mode(&self) -> ScheduleMode {
        self.schedule.mode()
    }

    /// Switch serial/parallel execution (advances remain bit-identical).
    pub fn set_mode(&mut self, mode: ScheduleMode) {
        self.schedule.set_mode(mode);
    }

    /// Borrow the world (read-only projection for the renderer / snapshots).
    pub fn world(&self) -> &EcsWorld {
        &self.world
    }

    /// Borrow the world mutably (e.g. to spawn/seed before stepping).
    pub fn world_mut(&mut self) -> &mut EcsWorld {
        &mut self.world
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_ecs::Vec3d;

    fn seeded() -> EcsWorld {
        let mut w = EcsWorld::new();
        for i in 0..8u128 {
            let e = w.spawn_with_guid(uuid::Uuid::from_u128(i + 1), "m", None);
            sim::set_velocity(&mut w, e, Vec3d::new(2.0, 1.0, -1.0));
        }
        w
    }

    #[test]
    fn step_once_advances_frame_count() {
        let mut gl = GameLoop::new(seeded(), SimConfig::default(), ScheduleMode::Serial);
        assert_eq!(gl.steps(), 0);
        gl.step_once();
        gl.step_once();
        assert_eq!(gl.steps(), 2);
    }

    #[test]
    fn run_frame_runs_backlog_and_exposes_alpha() {
        let cfg = SimConfig {
            fixed_dt: 0.1,
            ..SimConfig::default()
        };
        let mut gl = GameLoop::new(seeded(), cfg, ScheduleMode::Serial);
        // 0.25s of frame time → two 0.1s steps, 0.05 remainder → alpha 0.5.
        let n = gl.run_frame(0.25);
        assert_eq!(n, 2);
        assert_eq!(gl.steps(), 2);
        assert!((gl.alpha() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn moving_world_actually_moves() {
        let mut gl = GameLoop::new(seeded(), SimConfig::default(), ScheduleMode::Serial);
        let before = sim::sim_snapshot(gl.world_mut());
        for _ in 0..30 {
            gl.step_once();
        }
        let after = sim::sim_snapshot(gl.world_mut());
        assert_ne!(before, after, "the sim did not advance any state");
    }
}
