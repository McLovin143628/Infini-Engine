//! Sim-step frame-budget smoke (P15.1 / §8 ratchet).
//!
//! Runs the representative replay-harness world (~275 entities across a 3-deep
//! hierarchy) N fixed steps under the serial executor and asserts the **mean**
//! step time is under a hard budget. This is a regression tripwire, not a perf
//! claim: the budget is generous and, per §8, only ever ratchets **down** — never
//! up. It needs no GPU, so it runs in the normal test job on every OS.

use inf_runtime::game_loop::GameLoop;
use inf_runtime::replay::{build_world, harness_config};
use inf_runtime::ScheduleMode;

/// Hard mean-per-step budget, in milliseconds. The harness world steps in tens of
/// microseconds on real hardware; this leaves a wide margin for loaded CI runners
/// and debug builds. **RATCHET RULE (§8): this constant may only ever DECREASE.**
/// Lower it when the measured floor drops; never raise it to paper over a
/// regression.
const SIM_STEP_BUDGET_MS: f64 = 2.0;

#[test]
fn sim_step_stays_under_budget() {
    const WARMUP: u64 = 60;
    const MEASURED: u64 = 1000;

    let mut gl = GameLoop::new(build_world(), harness_config(), ScheduleMode::Serial);

    // Warm up (allocations, archetype layout, branch predictors) before timing.
    for _ in 0..WARMUP {
        gl.step_once();
    }

    let start = std::time::Instant::now();
    for _ in 0..MEASURED {
        gl.step_once();
    }
    let elapsed = start.elapsed();
    let mean_ms = elapsed.as_secs_f64() * 1000.0 / MEASURED as f64;

    eprintln!(
        "sim_step: mean {mean_ms:.4} ms/step over {MEASURED} steps (budget {SIM_STEP_BUDGET_MS} ms)"
    );
    assert!(
        mean_ms < SIM_STEP_BUDGET_MS,
        "sim step mean {mean_ms:.4} ms exceeded the {SIM_STEP_BUDGET_MS} ms budget \
         (the §8 budget only ratchets DOWN — investigate the regression, do not raise it)"
    );
}
