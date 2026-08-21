//! Subprocess probe for the pool-size-invariance leg of the determinism gate.
//!
//! `bevy_ecs`'s `ComputeTaskPool` is a process-global `OnceLock`, so a single
//! process can only ever run the parallel schedule under one pool size. This tiny
//! binary initializes the pool to a requested thread count, runs the replay
//! harness, and prints the trace hashes — the `replay_determinism` integration
//! test launches it once per pool size (1/2/8) and asserts every hash matches the
//! serial baseline.
//!
//! Usage: `replay_probe [threads] [steps]`  (defaults: threads=0→auto, steps=300)
//! Output (stdout, one `key=value` per line):
//!
//! ```text
//! threads=<actual pool worker count>
//! steps=<n>
//! serial=<hex 128-bit trace hash>
//! parallel=<hex 128-bit trace hash>
//! ```

use inf_runtime::replay;
use inf_runtime::ScheduleMode;

fn main() {
    let mut args = std::env::args().skip(1);
    let threads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let steps: u64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(replay::HARNESS_STEPS);

    // Serial baseline (no pool involved).
    let serial = replay::run_trace(ScheduleMode::Serial, steps);

    // Parallel under a pinned pool size (first init wins; this process only does one).
    let actual = inf_ecs::init_ecs_task_pool(threads);
    let parallel = replay::run_trace(ScheduleMode::Parallel, steps);

    println!("threads={actual}");
    println!("steps={steps}");
    println!("serial={serial:032x}");
    println!("parallel={parallel:032x}");
}
