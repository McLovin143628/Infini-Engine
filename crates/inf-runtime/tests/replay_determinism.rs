//! The §8 concurrency-determinism CI gate.
//!
//! Proves the parallel ECS schedule (P9.1a, §2.5) advances the fixed-step sim to a
//! **byte-identical trace** compared with the serial baseline, and that the result
//! is invariant to the ECS task-pool size.
//!
//! Two legs:
//!   1. **In-process** — serial vs parallel on the process pool, with repeated
//!      parallel runs to shake out scheduling nondeterminism. (`serial_eq_parallel`,
//!      `parallel_is_repeatable`)
//!   2. **Subprocess matrix** — the `replay_probe` binary re-run under fixed pool
//!      sizes 1 / 2 / 8, each asserted equal to the serial baseline. This is the
//!      only way to vary bevy's process-global `ComputeTaskPool` size.
//!      (`pool_size_invariance`)

use std::process::Command;

use inf_runtime::replay;
use inf_runtime::ScheduleMode;

const STEPS: u64 = replay::HARNESS_STEPS;

#[test]
fn serial_baseline_is_reproducible() {
    let a = replay::run_trace(ScheduleMode::Serial, STEPS);
    let b = replay::run_trace(ScheduleMode::Serial, STEPS);
    assert_eq!(a, b, "serial trace is not reproducible run-to-run");
}

#[test]
fn serial_eq_parallel_in_process() {
    let serial = replay::run_trace(ScheduleMode::Serial, STEPS);
    // The process ComputeTaskPool lazily initializes here (or was set by another
    // test); either way the parallel trace must equal the serial one.
    let parallel = replay::run_trace(ScheduleMode::Parallel, STEPS);
    assert_eq!(
        serial, parallel,
        "parallel schedule diverged from the serial baseline"
    );
}

#[test]
fn parallel_is_repeatable() {
    // Scheduling nondeterminism (which non-conflicting system runs on which worker
    // first) must never change the result. Re-run several times.
    let serial = replay::run_trace(ScheduleMode::Serial, STEPS);
    for i in 0..6 {
        let parallel = replay::run_trace(ScheduleMode::Parallel, STEPS);
        assert_eq!(serial, parallel, "parallel run {i} diverged");
    }
}

/// Launch the `replay_probe` binary at a fixed pool size and return `(serial,
/// parallel)` trace hashes it printed.
fn probe(threads: usize) -> (String, String) {
    let exe = env!("CARGO_BIN_EXE_replay_probe");
    let out = Command::new(exe)
        .arg(threads.to_string())
        .arg(STEPS.to_string())
        .output()
        .expect("spawn replay_probe");
    assert!(
        out.status.success(),
        "replay_probe (threads={threads}) failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("probe stdout utf8");
    let mut serial = None;
    let mut parallel = None;
    for line in stdout.lines() {
        if let Some(v) = line.strip_prefix("serial=") {
            serial = Some(v.to_string());
        } else if let Some(v) = line.strip_prefix("parallel=") {
            parallel = Some(v.to_string());
        }
    }
    (
        serial.expect("probe printed serial="),
        parallel.expect("probe printed parallel="),
    )
}

#[test]
fn pool_size_invariance() {
    // In-process serial reference, formatted like the probe's hex output.
    let reference = format!("{:032x}", replay::run_trace(ScheduleMode::Serial, STEPS));

    for &threads in &[1usize, 2, 8] {
        let (serial, parallel) = probe(threads);
        assert_eq!(
            serial, reference,
            "probe serial baseline (threads={threads}) differs from in-process serial"
        );
        assert_eq!(
            parallel, reference,
            "parallel trace at pool size {threads} differs from the serial baseline"
        );
    }
}
