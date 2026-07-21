//! The replay-determinism harness — the §8 concurrency-determinism CI gate.
//!
//! §2.5 requires that the parallel ECS schedule produce a **byte-identical trace**
//! to the serial baseline, and that the result be invariant to the ECS task-pool
//! size. This module builds a representative world (a few hundred entities with a
//! transform hierarchy, moving systems with deliberate cross-entity coupling, and
//! command-buffer spawns/despawns at sync points), runs it under a chosen
//! [`ScheduleMode`], and folds every step's guid-sorted state into a single rolling
//! hash — the trace fingerprint the gate compares.
//!
//! ## Why guid-sorted hashing makes this robust
//!
//! Entity ids recycle and appear in archetype order, which the parallel executor
//! must never be allowed to leak into results. [`inf_ecs::sim_snapshot`] therefore
//! keys and sorts the per-step state by the stable [`Guid`](inf_ecs::Guid) before
//! we hash it. Two runs — serial or parallel, 1 thread or 8 — that agree on every
//! entity's state agree on the trace hash, and a divergence at any step changes it.
//!
//! ## Pool-size coverage
//!
//! `bevy_ecs`'s `ComputeTaskPool` is a process-global `OnceLock`, so a single
//! process cannot re-run the schedule under several pool sizes. In-process tests
//! here compare **serial vs parallel** and re-run the parallel executor repeatedly
//! (scheduling nondeterminism would surface across repeats). The explicit
//! **1/2/8-thread** matrix is covered by the `replay_probe` binary, which each test
//! subprocess launches with a fixed [`init_ecs_task_pool`](inf_ecs::init_ecs_task_pool)
//! size — see `tests/replay_determinism.rs`.

use inf_ecs::{sim, EcsWorld, ScheduleMode, SimConfig};
use uuid::Uuid;
use xxhash_rust::xxh3::Xxh3;

use crate::game_loop::GameLoop;

/// Fixed-step count for the determinism gate.
pub const HARNESS_STEPS: u64 = 300;

/// The harness sim tuning (bounded box + centroid attraction so movers interact).
pub fn harness_config() -> SimConfig {
    SimConfig {
        fixed_dt: 1.0 / 60.0,
        bounds_min: inf_ecs::Vec3d::new(-40.0, -40.0, -40.0),
        bounds_max: inf_ecs::Vec3d::new(40.0, 40.0, 40.0),
        attract: 0.75,
    }
}

/// Build the representative harness world: ~275 entities across a 3-deep
/// transform hierarchy, seeded deterministically with velocities, spins, and a
/// spread of initial positions. Structural churn (transient spawns + their
/// despawns) is driven at runtime by the schedule's director/reap systems.
pub fn build_world() -> EcsWorld {
    let mut w = EcsWorld::new();
    let mut next: u128 = 1;
    let guid = |n: &mut u128| {
        let g = Uuid::from_u128(*n);
        *n += 1;
        g
    };

    for r in 0..50u128 {
        let root = w.spawn_with_guid(guid(&mut next), &format!("root-{r}"), None);
        // Position roots on a lattice; give them a velocity and (odd roots) a spin.
        let rx = (r % 9) as f64 * 3.0 - 12.0;
        let rz = (r / 9) as f64 * 3.0 - 6.0;
        sim::set_translation(&mut w, root, inf_ecs::Vec3d::new(rx, 0.0, rz));
        sim::set_velocity(
            &mut w,
            root,
            inf_ecs::Vec3d::new(
                1.0 + (r % 4) as f64,
                0.5 - (r % 3) as f64,
                -1.0 + (r % 5) as f64,
            ),
        );
        if r % 2 == 1 {
            sim::set_angular_velocity(&mut w, root, 30.0 + (r % 7) as f64 * 5.0);
        }

        for c in 0..3u128 {
            let child = w.spawn_with_guid(guid(&mut next), &format!("child-{r}-{c}"), Some(root));
            sim::set_translation(
                &mut w,
                child,
                inf_ecs::Vec3d::new(1.0 + c as f64, 1.5, -1.0 - c as f64),
            );
            sim::set_velocity(
                &mut w,
                child,
                inf_ecs::Vec3d::new(0.5 - c as f64 * 0.25, 0.75, 0.25 + c as f64 * 0.1),
            );

            // Even children carry a grandchild — deepens the hierarchy walk.
            if c.is_multiple_of(2) {
                let gc = w.spawn_with_guid(guid(&mut next), &format!("gc-{r}-{c}"), Some(child));
                sim::set_translation(&mut w, gc, inf_ecs::Vec3d::new(0.5, -0.5, 0.5));
                sim::set_velocity(&mut w, gc, inf_ecs::Vec3d::new(-0.3, 0.4, 0.6));
                if r.is_multiple_of(3) {
                    sim::set_angular_velocity(&mut w, gc, 12.0);
                }
            }
        }
    }

    w.propagate();
    w
}

/// Run `steps` fixed steps under `mode` and return the rolling 128-bit trace hash
/// (a fold of every step's guid-sorted world state). This is the value the §8 gate
/// compares across executor kinds and pool sizes.
pub fn run_trace(mode: ScheduleMode, steps: u64) -> u128 {
    let mut gl = GameLoop::new(build_world(), harness_config(), mode);
    let mut hasher = Xxh3::new();
    let cfg = bincode::config::standard();
    for _ in 0..steps {
        gl.step_once();
        let snap = sim::sim_snapshot(gl.world_mut());
        let bytes =
            bincode::serde::encode_to_vec(&snap, cfg).expect("sim snapshot is always encodable");
        hasher.update(&bytes);
    }
    hasher.digest128()
}

/// Convenience for the probe binary: initialize the ECS task pool to `threads`
/// workers (process-global; first call wins) and return the parallel trace hash.
pub fn run_trace_parallel_with_pool(threads: usize, steps: u64) -> u128 {
    inf_ecs::init_ecs_task_pool(threads);
    run_trace(ScheduleMode::Parallel, steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_world_has_a_few_hundred_entities() {
        let mut w = build_world();
        let n = w.entities().len();
        assert!(
            (250..=320).contains(&n),
            "expected a few hundred entities, got {n}"
        );
    }

    #[test]
    fn trace_is_stable_run_to_run_serial() {
        let a = run_trace(ScheduleMode::Serial, 120);
        let b = run_trace(ScheduleMode::Serial, 120);
        assert_eq!(a, b, "serial trace is not reproducible");
    }

    #[test]
    fn trace_actually_evolves() {
        // Different step counts must give different fingerprints (the world moves
        // and churns structurally — a constant hash would mean nothing happened).
        let short = run_trace(ScheduleMode::Serial, 10);
        let long = run_trace(ScheduleMode::Serial, 300);
        assert_ne!(short, long);
    }
}
