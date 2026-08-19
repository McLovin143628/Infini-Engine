//! P9.3 headless coverage: the demo world's determinism trace, the Coyote
//! blueprint actually running under `runtime_sim` with a real input host, the CI
//! smoke exit code, and the crash-capture / nonzero-exit path (subprocess).

use std::path::PathBuf;
use std::process::Command;

use glam::DVec3;
use uuid::Uuid;

use inf_player::demo;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
}

fn new_demo_sim() -> RuntimeSim {
    let b = demo::build();
    RuntimeSim::with_gravity(b.world, b.actors, b.gravity, b.hz)
}

fn player_guid() -> Uuid {
    Uuid::from_u128(demo::PLAYER_GUID)
}

fn player_pos(sim: &RuntimeSim) -> DVec3 {
    let e = sim
        .world()
        .entity_of(player_guid())
        .expect("player entity present");
    sim.world()
        .world_translation(e)
        .expect("player has a world transform")
}

// ── determinism ──────────────────────────────────────────────────────────────

#[test]
fn demo_trace_is_reproducible_and_evolves() {
    // Two folds of the same world are byte-identical (the §8 determinism gate,
    // same xxh3 pattern as `inf_runtime::replay`).
    assert_eq!(inf_player::demo_trace(300), inf_player::demo_trace(300));
    // Different step counts differ — the world actually moves.
    assert_ne!(inf_player::demo_trace(10), inf_player::demo_trace(300));
}

// ── the Coyote blueprint runs under runtime_sim ──────────────────────────────

#[test]
fn coyote_moves_right_when_the_right_action_is_held() {
    let mut sim = new_demo_sim();
    for _ in 0..30 {
        sim.step_once(RuntimeInput::default()); // settle onto the ground
    }
    let before = player_pos(&sim).x;
    for _ in 0..40 {
        sim.step_once(RuntimeInput::with_down(["right"]));
    }
    let after = player_pos(&sim).x;
    assert!(
        after > before + 0.5,
        "holding 'right' should move the coyote right: before={before} after={after}"
    );
}

#[test]
fn coyote_jump_raises_the_player() {
    let mut sim = new_demo_sim();
    for _ in 0..30 {
        sim.step_once(RuntimeInput::default()); // settle + build coyote grounding
    }
    let rest = player_pos(&sim).y;
    // Rising edge on 'jump' (just_pressed) while grounded fires the jump.
    sim.step_once(RuntimeInput::with_down(["jump"]));
    let mut peak = player_pos(&sim).y;
    for _ in 0..20 {
        sim.step_once(RuntimeInput::default());
        peak = peak.max(player_pos(&sim).y);
    }
    assert!(
        peak > rest + 0.1,
        "a jump should raise the player above rest: rest={rest} peak={peak}"
    );
}

// ── CI smoke exit code ───────────────────────────────────────────────────────

#[test]
fn headless_ci_smoke_exits_zero_and_prints_hash() {
    let output = Command::new(player_bin())
        .args(["--headless", "--run-frames", "300", "--assert-exit"])
        .output()
        .expect("player runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("final-state-hash:"));
}

// ── crash capture + nonzero exit (subprocess) ────────────────────────────────

#[test]
fn headless_panic_is_captured_to_crash_file_and_exits_nonzero() {
    let dir = tempfile::tempdir().expect("tempdir");
    let crash = dir.path().join("crash.txt");
    let output = Command::new(player_bin())
        .args([
            "--headless",
            "--demo",
            "--run-frames",
            "300",
            "--panic-after",
            "10",
            "--crash-file",
            crash.to_str().unwrap(),
        ])
        .output()
        .expect("player runs");

    assert!(!output.status.success(), "a panic must not exit 0");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("deliberate headless panic"),
        "panic text must reach stderr; got:\n{stderr}"
    );

    let report = std::fs::read_to_string(&crash).expect("crash report written");
    assert!(report.contains("inf-player CRASH"), "report:\n{report}");
    assert!(
        report.contains("deliberate headless panic"),
        "report must carry the panic message:\n{report}"
    );
    assert!(
        report.contains("location:"),
        "report must carry the panic location:\n{report}"
    );
}
