//! End-to-end Simulate proof (P8.4): load the committed platformer sample, enter
//! Simulate headlessly, script keyboard input, run fixed steps, and assert the
//! player's trajectory — including that the **coyote-time** window lets a jump
//! fire a few frames after walking off a ledge, but not after the window closes.
//!
//! This exercises the real path: `SimSession` → `PhysicsBridge2D` (rapier2d) +
//! the blueprint interpreter over the coyote `.inf_act`, with a real
//! `Physics2dHost` adapter. No GPU, no Tauri — pure CI.

use glam::DVec2;
use inf_ecs::components::Transform;
use inf_editor_core::samples::{platformer_actors, platformer_scene, PLAYER_GUID};
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};

/// The player's world-space Y (its feet ride the ground at y ≈ 0.8).
fn player_y(doc: &SceneDoc) -> f64 {
    let e = doc.entity_of(PLAYER_GUID).expect("player entity");
    doc.world()
        .world()
        .get::<Transform>(e)
        .expect("player transform")
        .translation
        .y
}

fn player_x(doc: &SceneDoc) -> f64 {
    let e = doc.entity_of(PLAYER_GUID).expect("player entity");
    doc.world()
        .world()
        .get::<Transform>(e)
        .expect("player transform")
        .translation
        .x
}

fn enter() -> (SceneDoc, SimSession) {
    let mut doc = platformer_scene();
    // World gravity is ZERO: the character is kinematic and applies gravity
    // itself in the blueprint (so it isn't double-counted).
    let session = SimSession::enter(&mut doc, platformer_actors(), DVec2::ZERO, SIM_HZ);
    (doc, session)
}

/// Run right until the player leaves the ground; return the step index at which
/// it first became airborne, plus the airborne Y at that moment.
fn run_off_the_ledge(doc: &mut SceneDoc, session: &mut SimSession) -> usize {
    let run_right = || SimInput::with_down(["right"]);
    // On the ground the player is grounded and its Y is stable; past the ledge
    // (world x > 3) it goes airborne.
    for step in 0..200 {
        session.step_once(doc, run_right());
        if !session.is_grounded(PLAYER_GUID) && player_x(doc) > 3.0 {
            return step;
        }
    }
    panic!("player never left the ledge (x={})", player_x(doc));
}

#[test]
fn player_runs_right_and_walks_off_the_ledge() {
    let (mut doc, mut session) = enter();
    let y0 = player_y(&doc);
    assert!(session.steps() == 0);

    // While on the ground, running right keeps the player grounded and level.
    for _ in 0..5 {
        session.step_once(&mut doc, SimInput::with_down(["right"]));
    }
    assert!(
        session.is_grounded(PLAYER_GUID),
        "should be grounded on the ground"
    );
    assert!(
        (player_y(&doc) - y0).abs() < 0.2,
        "Y stays level on the ground"
    );
    assert!(player_x(&doc) > 1.5, "moved right");

    // Keep going: eventually it walks off the ledge and starts falling.
    run_off_the_ledge(&mut doc, &mut session);
    let y_air = player_y(&doc);
    for _ in 0..10 {
        session.step_once(&mut doc, SimInput::with_down(["right"]));
    }
    assert!(!session.is_grounded(PLAYER_GUID), "airborne past the ledge");
    assert!(player_y(&doc) < y_air, "falling after the ledge");
}

#[test]
fn coyote_jump_fires_within_the_window() {
    let (mut doc, mut session) = enter();
    run_off_the_ledge(&mut doc, &mut session);
    let y_leave = player_y(&doc);

    // 3 fixed steps after leaving the ground (well inside the ~7-step coyote
    // window), press jump. It must fire despite not being grounded.
    for _ in 0..3 {
        session.step_once(&mut doc, SimInput::with_down(["right"]));
    }
    let y_before_jump = player_y(&doc);
    // Rising edge: jump goes down this step.
    session.step_once(&mut doc, SimInput::with_down(["right", "jump"]));
    // Give the jump a few steps to carry the player upward.
    let mut peak = player_y(&doc);
    for _ in 0..6 {
        session.step_once(&mut doc, SimInput::with_down(["right"]));
        peak = peak.max(player_y(&doc));
    }
    assert!(
        peak > y_before_jump + 0.1,
        "coyote jump should lift the player: peak {peak} vs {y_before_jump} (left ground at {y_leave})"
    );
}

#[test]
fn jump_after_the_coyote_window_does_not_fire() {
    let (mut doc, mut session) = enter();
    run_off_the_ledge(&mut doc, &mut session);

    // Wait out the coyote window (~7 steps at 0.12 s): 12 steps of falling.
    for _ in 0..12 {
        session.step_once(&mut doc, SimInput::with_down(["right"]));
    }
    let y_before_jump = player_y(&doc);
    // Now press jump — the window has closed, so it must be ignored.
    session.step_once(&mut doc, SimInput::with_down(["right", "jump"]));
    let mut peak_after = player_y(&doc);
    for _ in 0..6 {
        session.step_once(&mut doc, SimInput::with_down(["right"]));
        peak_after = peak_after.max(player_y(&doc));
    }
    assert!(
        peak_after < y_before_jump,
        "a late jump must NOT fire: peak_after {peak_after} should stay below {y_before_jump}"
    );
}

#[test]
fn enter_then_exit_restores_the_world_exactly() {
    let before = {
        let doc = platformer_scene();
        inf_editor_core::scene::serialize::encode(
            &inf_editor_core::scene::serialize::to_scene_file(&doc),
        )
        .unwrap()
    };

    let mut doc = platformer_scene();
    let mut session = SimSession::enter(&mut doc, platformer_actors(), DVec2::ZERO, SIM_HZ);
    // Perturb the world by simulating.
    for _ in 0..30 {
        session.step_once(&mut doc, SimInput::with_down(["right"]));
    }
    assert!(player_x(&doc) > 1.5, "the sim moved the player");

    session.exit(&mut doc);
    let after = inf_editor_core::scene::serialize::encode(
        &inf_editor_core::scene::serialize::to_scene_file(&doc),
    )
    .unwrap();
    assert_eq!(
        before, after,
        "exit must restore the pre-Simulate world byte-for-byte"
    );
}

/// Determinism: two independent runs of the same scripted input produce the same
/// trajectory (the §2.5 fixed-step, Guid-ordered guarantee).
#[test]
fn simulation_is_deterministic() {
    fn run() -> Vec<(f64, f64)> {
        let (mut doc, mut session) = enter();
        let mut trace = Vec::new();
        for i in 0..40 {
            let input = if i == 25 {
                SimInput::with_down(["right", "jump"])
            } else {
                SimInput::with_down(["right"])
            };
            session.step_once(&mut doc, input);
            trace.push((player_x(&doc), player_y(&doc)));
        }
        trace
    }
    assert_eq!(run(), run(), "Simulate must be deterministic");
}
