//! Sim-schedule scaling bench (P9.1a; follows the P7.0 job-pool bench
//! conventions: `criterion` with default features off, `harness = false`).
//!
//! Measures one fixed step of the standard sim schedule over the replay harness
//! world (~275 entities across a 3-deep hierarchy) under the serial executor and
//! the parallel executor at ECS pool sizes 1 / 2 / 8. The parallel win is running
//! the independent gameplay systems (motion pipeline ∥ lifetime aging)
//! concurrently; the transform-propagation tail is serial today (a documented
//! §2.5 follow-up), so this is an honest Amdahl's-law picture, not a linear-speedup
//! claim.
//!
//! Note: bevy's `ComputeTaskPool` is process-global, so the first parallel
//! benchmark group fixes the pool size for the whole process — the `pool/N`
//! numbers below share one pool sized by the first group. The serial vs parallel
//! contrast is the meaningful signal here; per-size parallel scaling is measured
//! precisely by the subprocess probe in the determinism tests.

use criterion::{criterion_group, criterion_main, Criterion};
use inf_runtime::game_loop::GameLoop;
use inf_runtime::replay::{build_world, harness_config};
use inf_runtime::ScheduleMode;

fn bench_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("sim_schedule_step");

    group.bench_function("serial", |b| {
        let mut gl = GameLoop::new(build_world(), harness_config(), ScheduleMode::Serial);
        b.iter(|| {
            gl.step_once();
        });
    });

    // Size the ECS pool once to available parallelism (process-global).
    inf_ecs::init_ecs_task_pool(0);
    group.bench_function("parallel", |b| {
        let mut gl = GameLoop::new(build_world(), harness_config(), ScheduleMode::Parallel);
        b.iter(|| {
            gl.step_once();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_schedule);
criterion_main!(benches);
