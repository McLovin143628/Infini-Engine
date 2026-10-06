//! **WAVE PERF1b — THE ISLAND AT FRAME RATE, PART TWO.**
//!
//! PERF1 left the shipped window at ~55-65 ms a frame in every host, at two
//! fixed steps a frame with the backlog dropped: the world ran at about half
//! real time. This gate holds what PERF1b bought:
//!
//! * the windowed loop runs its fixed steps on a worker BESIDE the record and
//!   submit of the frame projected from the previous steps, through one door
//!   ([`the_windowed_loop_steps_beside_the_record_through_one_door`]);
//! * the step as the window runs it is attributed phase by phase in the
//!   window itself (`INF_STEP_LOG`,
//!   [`the_step_log_reads_the_window_step_by_phase`]) -- the instrument the
//!   clause-0 table was taken with;
//! * the window counter's own logs, when a session is named, hold the floor:
//!   p95 at or under [`PERF1B_FLOOR_MS`] with the simulation at real time and
//!   no backlog dropped ([`the_window_counter_holds_the_floor_at_real_time`]).
//!   A clock arm: printed everywhere, asserted only in release, off CI.
//!
//! The sweep the physics bridge now skips when it can prove it empty is held
//! in `inf-physics` (`a_skipped_despawn_sweep_leaves_exactly_what_the_full_sweep_leaves`).

use std::path::Path;

/// The floor of "done" for wave PERF1b, milliseconds: the window counter's p95
/// in every host at both hours, WITH the simulation at real time.
pub const PERF1B_FLOOR_MS: f64 = 33.0;

/// **THE STEPS RUN BESIDE THE RECORD, THROUGH ONE DOOR** (wave PERF1b).
///
/// The windowed frame used to be steps + projection + record end to end; on
/// the island that was ~2 x 14 ms + ~28 ms and a frame slower than the step
/// period owes two steps. The frame now decides what the steps need (the held
/// set, the owed time), projects the state the previous steps left, and runs
/// this frame's steps on a scoped worker while it records and submits.
///
/// The arms read the source, because the property is an ordering in a winit
/// callback no headless test can drive: the steps are spawned on a scoped
/// thread BEFORE the render call and joined AFTER it, the frame reaches the
/// sim through exactly one `run_frame`, and the projection happens before the
/// scope opens. Behaviourally: the sim is `Send` (the mod session's raw world
/// pointer is cleared between calls -- `inf-wasm-host`'s `ModState`), and the
/// catch-up cap's own arm (`perf1_budget_gate`) still holds the accumulator.
#[test]
fn the_windowed_loop_steps_beside_the_record_through_one_door() {
    fn is_send<T: Send>() {}
    is_send::<inf_player::runtime_sim::RuntimeSim>();

    let window = include_str!("../src/window.rs");
    assert_eq!(
        window.matches(".run_frame(").count(),
        1,
        "the windowed loop must reach the sim through ONE run_frame call"
    );
    let frame_at = window
        .find("    fn frame(&mut self, event_loop: &ActiveEventLoop)")
        .expect("the windowed frame");
    let frame = &window[frame_at..];
    let project = frame
        .find("live.host.project(&self.sim, alpha, eye);")
        .expect("the frame projects the sim");
    let scope = frame
        .find("std::thread::scope(")
        .expect("the frame opens a thread scope for its steps");
    let spawn = frame
        .find("scope.spawn(move || Self::run_owed_steps(")
        .expect("the steps are spawned on the scope");
    let render = frame
        .find("live.host.render(&view);")
        .expect("the frame renders");
    let join = frame.find(".join()").expect("the steps are joined");
    assert!(
        project < scope && scope < spawn && spawn < render && render < join,
        "the order must be project -> scope -> spawn(steps) -> render -> join \
         (project {project}, scope {scope}, spawn {spawn}, render {render}, join {join})"
    );
    println!(
        "PERF1b: project @{project}, scope @{scope}, spawn @{spawn}, render @{render}, join @{join}"
    );
}

/// **THE STEP AS THE WINDOW RUNS IT, PHASE BY PHASE** (wave PERF1b clause 0).
///
/// `INF_STEP_LOG` arms the fixed step's per-phase clock in the window and
/// writes, per window of frames, the steps run, the projection and render
/// milliseconds per frame and every step phase per STEP. Ten frames of two
/// steps each, with `solver` charged 3 ms per step, must read 20 steps, a
/// 3.000 ms solver and a 3.00 ms step -- a log that averaged per frame would
/// read 6.
#[test]
fn the_step_log_reads_the_window_step_by_phase() {
    let mut log = inf_player::ui::StepLog::default();
    let solver = inf_player::step_profile::STEP_PHASE_NAMES
        .iter()
        .position(|n| *n == "solver")
        .expect("a solver phase");
    let mut two_steps = inf_player::step_profile::StepProfile::default();
    two_steps.ms[solver] = 6.0;
    let mut record = inf_render::RecordProfile::default();
    record.ms[inf_render::record::SUBMIT] = 4.0;
    for _ in 0..10 {
        log.push((two_steps, 2), 1.5, 20.0, &record);
    }
    let line = log.line();
    println!("{line}");
    assert!(line.contains(" frames=10 steps=20 "), "{line}");
    assert!(
        line.contains(" proj=1.50 render=20.00 step=3.00 |"),
        "{line}"
    );
    assert!(line.contains(" solver=3.000"), "{line}");
    assert!(line.contains(" submit=4.000"), "{line}");
}

/// One leg of a windowed session as the frame log saw it.
struct Leg {
    name: String,
    frames: usize,
    p50: f64,
    p95: f64,
    sim_rate: f64,
    dropped_ms: f64,
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    let k = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1;
    sorted[k]
}

/// Parse a session directory's `fps.log` (`INF_FPS_LOG`) against its
/// `perf-legs.txt` (`demo.ps1 -PerfOnly`): per leg, the frame interval's p50 /
/// p95, the simulation rate (fixed steps x period over wall time) and the
/// backlog the catch-up cap dropped.
fn session_legs(dir: &Path, step_s: f64) -> Vec<Leg> {
    let Ok(log) = std::fs::read_to_string(dir.join("fps.log")) else {
        return Vec::new();
    };
    // (end time s, dt ms, steps, dropped ms)
    let mut frames: Vec<(f64, f64, u32, f64)> = Vec::new();
    for line in log.lines() {
        let mut t = 0.0;
        let (mut dts, mut steps, mut drops) = (Vec::new(), Vec::new(), Vec::new());
        for part in line.split_whitespace() {
            let Some((k, v)) = part.split_once('=') else {
                continue;
            };
            let nums = || v.split(',').filter_map(|x| x.parse::<f64>().ok());
            match k {
                "t" => t = v.parse().unwrap_or(0.0),
                "dt" => dts = nums().map(|x| x / 10.0).collect(),
                "steps" => steps = nums().map(|x| x as u32).collect(),
                "drop" => drops = nums().map(|x| x / 10.0).collect(),
                _ => {}
            }
        }
        let mut end = t;
        let mut rows = Vec::new();
        for i in (0..dts.len()).rev() {
            rows.push((
                end,
                dts[i],
                steps.get(i).copied().unwrap_or(0),
                drops.get(i).copied().unwrap_or(0.0),
            ));
            end -= dts[i] / 1000.0;
        }
        rows.reverse();
        frames.extend(rows);
    }
    let Ok(legs) = std::fs::read_to_string(dir.join("perf-legs.txt")) else {
        return Vec::new();
    };
    legs.lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let name = it.next()?.to_string();
            let a: f64 = it.next()?.parse().ok()?;
            let b: f64 = it.next()?.parse().ok()?;
            let sel: Vec<&(f64, f64, u32, f64)> =
                frames.iter().filter(|f| f.0 >= a && f.0 <= b).collect();
            if sel.is_empty() {
                return None;
            }
            let mut dts: Vec<f64> = sel.iter().map(|f| f.1).collect();
            dts.sort_by(f64::total_cmp);
            let wall: f64 = dts.iter().sum::<f64>() / 1000.0;
            let steps: u32 = sel.iter().map(|f| f.2).sum();
            Some(Leg {
                name,
                frames: sel.len(),
                p50: pct(&dts, 0.5),
                p95: pct(&dts, 0.95),
                sim_rate: f64::from(steps) * step_s / wall.max(1e-9),
                dropped_ms: sel.iter().map(|f| f.3).sum(),
            })
        })
        .collect()
}

/// **THE WINDOW COUNTER HOLDS THE FLOOR AT REAL TIME** (wave PERF1b).
///
/// `INF_PERF1B_SESSIONS` names session directories (`;`-separated) the demo
/// loop wrote with `-PerfOnly` -- standalone, Play in New Window, embedded, at
/// noon and 21:00. Every leg of every session is printed with its p50 / p95,
/// its simulation rate and the backlog the cap dropped; in release, off CI,
/// every leg must be at or under [`PERF1B_FLOOR_MS`] at p95 with the sim at
/// real time (rate >= 0.99) and nothing dropped. Unset, it says so and
/// returns: the sessions need a GPU, a display and the cooked island.
#[test]
fn the_window_counter_holds_the_floor_at_real_time() {
    let Some(list) = std::env::var_os("INF_PERF1B_SESSIONS") else {
        println!("SKIP: INF_PERF1B_SESSIONS names no session (a GPU, a display and the cooked island are needed)");
        return;
    };
    let step_s = 1.0 / f64::from(inf_runtime::TICK_HZ);
    let mut worst = 0.0f64;
    let mut failures = Vec::new();
    let mut legs_seen = 0usize;
    for dir in std::env::split_paths(&list) {
        for leg in session_legs(&dir, step_s) {
            legs_seen += 1;
            println!(
                "PERF1b {} {:6}: {:5} frames  p50 {:6.1}  p95 {:6.1} ms  sim {:.2}x  dropped {:7.1} ms",
                dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                leg.name,
                leg.frames,
                leg.p50,
                leg.p95,
                leg.sim_rate,
                leg.dropped_ms
            );
            worst = worst.max(leg.p95);
            if leg.p95 > PERF1B_FLOOR_MS || leg.sim_rate < 0.99 || leg.dropped_ms > 0.0 {
                failures.push(format!(
                    "{} {}: p95 {:.1} sim {:.2}x dropped {:.1} ms",
                    dir.display(),
                    leg.name,
                    leg.p95,
                    leg.sim_rate,
                    leg.dropped_ms
                ));
            }
        }
    }
    println!("PERF1b: worst p95 {worst:.1} ms over {legs_seen} legs against the {PERF1B_FLOOR_MS} ms floor");
    if cfg!(debug_assertions) || std::env::var_os("CI").is_some() {
        println!("dev build or CI: reported, not asserted");
        return;
    }
    assert!(legs_seen > 0, "the sessions named hold no leg");
    assert!(
        failures.is_empty(),
        "legs over the floor or under real time: {failures:#?}"
    );
}
