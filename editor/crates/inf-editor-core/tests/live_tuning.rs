//! **Live tuning during Simulate** — pillar S4's gate (P29.5).
//!
//! Four claims, and each one asserts the **world** rather than the queue:
//!
//! (a) a queued tune does nothing until the next fixed step, and then does;
//! (b) a movement tunable really changes how the character moves, measured in
//!     metres against a control that did not tune;
//! (c) a `Session`-scoped tune is gone after Stop and a `Keep`-scoped one is
//!     there — and is undoable, because it is an ordinary edit;
//! (d) a state machine's parameter and its trigger are reachable from the
//!     editor, which they were not before this wave at all.
//!
//! The fifth claim — that the **shipped player has no tuning door** — is
//! `player_has_no_tuning_door.rs`, because it is a statement about source and
//! not about a world.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{
    AnimStateMachine, BodyKind3D, CharacterMovement, Collider3D, ColliderShape3DKind, Gait,
    RigidBody3D, Transform,
};
use inf_ecs::math::Vec3d;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};
use inf_editor_core::tuning::{Tune, TuneScope, MOVEMENT_TYPE_PATH};

const FLOOR: Uuid = Uuid::from_u128(0x2905_1001);
const HERO: Uuid = Uuid::from_u128(0x2905_1002);
const SM: Uuid = Uuid::from_u128(0x2905_1003);

macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

/// A floor and one player-controlled character standing on it.
fn world() -> SceneDoc {
    let mut doc = SceneDoc::new();
    doc.create_with_guid(FLOOR, SpawnKind::Empty, "Floor", None);
    insert!(
        doc,
        FLOOR,
        Transform::from_translation(DVec3::new(0.0, -0.5, 0.0))
    );
    insert!(
        doc,
        FLOOR,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        }
    );
    insert!(
        doc,
        FLOOR,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(40.0, 0.5, 40.0),
            ..Default::default()
        }
    );

    let cm = CharacterMovement {
        player_controlled: true,
        gait: Gait::Run,
        ..Default::default()
    };
    let half = cm.stand_half_height_m;
    doc.create_with_guid(HERO, SpawnKind::Empty, "Hero", None);
    insert!(
        doc,
        HERO,
        Transform::from_translation(DVec3::new(0.0, half + 0.3, 0.0))
    );
    insert!(
        doc,
        HERO,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        }
    );
    insert!(
        doc,
        HERO,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(0.3, half, 0.3),
            radius: 0.3,
            ..Default::default()
        }
    );
    insert!(
        doc,
        HERO,
        inf_ecs::components::CharacterController3D::default()
    );
    insert!(doc, HERO, cm);
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    doc
}

fn session(doc: &mut SceneDoc) -> SimSession {
    SimSession::enter(doc, Vec::new(), glam::DVec2::new(0.0, -9.81), SIM_HZ)
}

/// Full forward stick for the whole run, on the analog axis the movement step
/// reads (`inf_ecs::movement::actions::MOVE_Y`).
fn forward() -> SimInput {
    let mut axes = std::collections::BTreeMap::new();
    axes.insert(inf_ecs::movement::actions::MOVE_Y.to_string(), 1.0f32);
    SimInput::default().with_axes(axes)
}

fn speed_field(doc: &SceneDoc) -> f64 {
    let e = doc.entity_of(HERO).expect("hero");
    doc.world()
        .world()
        .get::<CharacterMovement>(e)
        .expect("movement")
        .run_speed_mps
}

fn pos(doc: &SceneDoc) -> DVec3 {
    let e = doc.entity_of(HERO).expect("hero");
    doc.world()
        .world()
        .get::<Transform>(e)
        .expect("transform")
        .translation
        .to_dvec3()
}

/// The character's own horizontal speed this step, m/s — what the movement step
/// arrived at, rather than what the document says it may reach.
fn speed_now(doc: &SceneDoc) -> f64 {
    let e = doc.entity_of(HERO).expect("hero");
    let v = doc
        .world()
        .world()
        .get::<CharacterMovement>(e)
        .expect("movement")
        .runtime
        .velocity
        .to_dvec3();
    DVec3::new(v.x, 0.0, v.z).length()
}

/// **(a) The next fixed step, and not this one.**
///
/// The contract `crate::tuning` exists to make explicit: a queued tune has not
/// happened yet, and the step that happens next is where it lands.
#[test]
fn a_queued_tune_lands_on_the_next_fixed_step_and_not_before() {
    let mut doc = world();
    let mut sim = session(&mut doc);
    let before = speed_field(&doc);
    assert!((before - 3.75).abs() < 1e-9, "the ALS default: {before}");

    sim.tune(
        Tune::movement(HERO, "run_speed_mps", 9.0),
        TuneScope::Session,
    );
    assert_eq!(sim.pending_tunes(), 1);
    assert_eq!(
        speed_field(&doc),
        before,
        "queueing a tune must not touch the world"
    );

    sim.step_once(&mut doc, SimInput::default());
    assert_eq!(sim.pending_tunes(), 0, "the queue must drain");
    assert!(
        (speed_field(&doc) - 9.0).abs() < 1e-9,
        "the tune did not apply: {}",
        speed_field(&doc)
    );

    // …and it is not re-applied for ever: a second step over an empty queue
    // leaves whatever the world says now, so a Blueprint writing the same field
    // is not fighting a ghost.
    sim.step_once(&mut doc, SimInput::default());
    assert!((speed_field(&doc) - 9.0).abs() < 1e-9);
}

/// **(a2) The tune lands at the TOP of the step, so the step that drains it is
/// the step that reads it** (P29.5 audit, A5).
///
/// Arm (a) proves the value moved by the time the step returned, which is true
/// wherever in the step the queue is drained — an `apply_pending_tunes` moved to
/// the very *last* line of `fixed_step` passes it, and passes the whole of the
/// rest of this file. That is the S4 contract unfalsified: "queued and drained
/// at the very top of the fixed step, before anything reads anything" is exactly
/// the sentence no arm was asking about.
///
/// So this one asks the movement step. Let the character settle at its
/// cruising speed, queue a tune that raises the cap, and run **one** step: with
/// the drain at the top, that step accelerates toward the new cap; with it
/// anywhere after the movement solve, the character is still capped at the old
/// one and does not move until the step after.
///
/// The control is the same settled character stepped once with no tune at all,
/// which must not speed up — otherwise this measures the settling and not the
/// tune.
#[test]
fn a_tune_is_read_by_the_step_that_drained_it_and_not_the_one_after() {
    let settle = |sim: &mut SimSession, doc: &mut SceneDoc| {
        for _ in 0..180 {
            sim.step_once(doc, forward());
        }
    };

    // THE CONTROL first, so a failure below cannot be the fixture drifting.
    let mut doc = world();
    let mut sim = session(&mut doc);
    settle(&mut sim, &mut doc);
    let cruise = speed_now(&doc);
    assert!(
        (cruise - 3.75).abs() < 0.05,
        "the character must be at its run cap before the tune, not still \
         accelerating into it: {cruise}"
    );
    sim.step_once(&mut doc, forward());
    let control = speed_now(&doc);
    assert!(
        (control - cruise).abs() < 1e-9,
        "an untuned step changed the speed on its own ({cruise} -> {control})"
    );

    // THE CLAIM: the very next step after the queue is drained is already faster.
    let mut doc = world();
    let mut sim = session(&mut doc);
    settle(&mut sim, &mut doc);
    let before = speed_now(&doc);
    sim.tune(
        Tune::movement(HERO, "run_speed_mps", 9.0),
        TuneScope::Session,
    );
    sim.step_once(&mut doc, forward());
    let after = speed_now(&doc);
    assert!(
        after > before + 1e-6,
        "the step that drained the tune ran at the OLD cap ({before} -> {after}) \
         — the queue is not being drained before the movement solve"
    );
}

/// **(b) A movement tunable changes the movement**, in metres, against a control
/// that ran the same input with no tune at all.
///
/// The number and not the field: reading `run_speed_mps` back proves a write,
/// which is arm (a)'s job. This one proves the character.
#[test]
fn a_tuned_speed_moves_the_character_further_than_an_untuned_one() {
    let run = |tuned: bool| -> f64 {
        let mut doc = world();
        let mut sim = session(&mut doc);
        if tuned {
            sim.tune(
                Tune::movement(HERO, "run_speed_mps", 9.0),
                TuneScope::Session,
            );
        }
        for _ in 0..120 {
            sim.step_once(&mut doc, forward());
        }
        pos(&doc).length()
    };
    let control = run(false);
    let tuned = run(true);
    assert!(
        control > 1.0,
        "the control must actually move, or this compares two zeros: {control}"
    );
    assert!(
        tuned > control * 1.5,
        "a 2.4x speed tune moved {tuned:.3} m against a control's {control:.3} m"
    );
}

/// **(c) Session forgets, Keep remembers — and Keep is undoable.**
#[test]
fn a_session_tune_is_gone_after_stop_and_a_kept_one_is_an_undoable_edit() {
    // Session scope.
    let mut doc = world();
    let mut sim = session(&mut doc);
    sim.tune(
        Tune::movement(HERO, "run_speed_mps", 9.0),
        TuneScope::Session,
    );
    sim.step_once(&mut doc, SimInput::default());
    assert!((speed_field(&doc) - 9.0).abs() < 1e-9);
    assert!(sim.kept_tunes().is_empty());
    sim.exit(&mut doc);
    assert!(
        (speed_field(&doc) - 3.75).abs() < 1e-9,
        "a session tune survived Stop: {}",
        speed_field(&doc)
    );

    // Keep scope, dragged: forty values, one kept edit.
    let mut doc = world();
    let mut sim = session(&mut doc);
    for k in 0..40 {
        sim.tune(
            Tune::movement(HERO, "run_speed_mps", 4.0 + 0.1 * k as f64),
            TuneScope::Keep,
        );
        sim.step_once(&mut doc, SimInput::default());
    }
    let settled = speed_field(&doc);
    assert!((settled - 7.9).abs() < 1e-9, "{settled}");
    assert_eq!(sim.kept_tunes().len(), 40, "every applied Keep is recorded");
    sim.exit(&mut doc);
    assert!(
        (speed_field(&doc) - settled).abs() < 1e-9,
        "a kept tune did not survive Stop: {}",
        speed_field(&doc)
    );
    // ONE undo step, not forty — the drag was one decision.
    doc.undo();
    assert!(
        (speed_field(&doc) - 3.75).abs() < 1e-9,
        "one undo did not take the whole drag back: {}",
        speed_field(&doc)
    );
}

/// **(d) A machine's parameters and triggers are reachable from the editor.**
///
/// They were not before this wave: the `anim.*` kit is a Blueprint's door, so an
/// author watching a machine had no way to say "what does it do at speed 4?"
/// without writing a program to say it.
#[test]
fn a_machine_parameter_and_a_trigger_are_editor_reachable() {
    let mut doc = world();
    insert!(
        doc,
        HERO,
        AnimStateMachine {
            sm: Some(SM),
            ..Default::default()
        }
    );
    doc.world_mut().mark_dirty();
    let mut sim = session(&mut doc);

    sim.tune(
        Tune::Param {
            guid: HERO,
            name: "speed".into(),
            value: 4.0,
        },
        TuneScope::Session,
    );
    sim.tune(
        Tune::Trigger {
            guid: HERO,
            name: "jump".into(),
        },
        TuneScope::Session,
    );
    // Nothing before the step.
    assert_eq!(inf_ecs::anim_param(doc.world(), HERO, "speed"), None);
    sim.step_once(&mut doc, SimInput::default());
    assert_eq!(
        inf_ecs::anim_param(doc.world(), HERO, "speed"),
        Some(4.0),
        "the parameter overlay did not reach the bridge"
    );
    let bridge = inf_ecs::anim_bridge::bridge(doc.world()).expect("a bridge");
    assert!(
        bridge
            .triggers
            .get(&HERO)
            .is_some_and(|t| t.contains("jump")),
        "the trigger was not armed"
    );

    // The same tune on an entity with **no machine** is a refusal, not a write:
    // the bridge's own `has_machine` gate, reached through the tuning door.
    sim.tune(
        Tune::Param {
            guid: FLOOR,
            name: "speed".into(),
            value: 1.0,
        },
        TuneScope::Session,
    );
    sim.step_once(&mut doc, SimInput::default());
    assert_eq!(inf_ecs::anim_param(doc.world(), FLOOR, "speed"), None);
}

/// A tune naming a field the component does not have is a **value**, and the
/// step it was queued on runs normally.
#[test]
fn a_tune_with_a_bad_field_is_a_value_and_the_step_still_runs() {
    let mut doc = world();
    let mut sim = session(&mut doc);
    sim.tune(
        Tune::Field {
            guid: HERO,
            type_path: MOVEMENT_TYPE_PATH.into(),
            path: "sprint_speed_mph".into(),
            value: inf_ecs::PropValue::Number(99.0),
        },
        TuneScope::Keep,
    );
    sim.step_once(&mut doc, forward());
    assert_eq!(sim.pending_tunes(), 0);
    assert!(
        sim.kept_tunes().is_empty(),
        "a tune that did nothing must not be kept"
    );
    // The world moved, so the step ran.
    for _ in 0..30 {
        sim.step_once(&mut doc, forward());
    }
    assert!(pos(&doc).length() > 0.1, "{:?}", pos(&doc));
}
