//! Simulate integration for animation state machines (P11.2).
//!
//! Proves the tick wiring: an entity carrying an `AnimStateMachine` whose `sm`
//! GUID resolves in the session's registered machines is stepped each fixed step
//! (after `advance_anim_players`), its runtime advancing deterministically. The
//! machine's condition/param lookups read the actor's Blueprint variables through
//! the `SmContext` seam; an entity with no actor gets an empty variable set, so
//! params default to `0` — exercised here (the entity is not a blueprint actor).
//!
//! Pose evaluation → skinning palette **used to be** render-time (the same
//! placeholder gap as `SkeletalMesh`), and P24.1 closed that: the same fixed step
//! now evaluates the machine's pose and publishes it for both projectors
//! (`inf_ecs::pose`). This file still asserts only the *runtime state*, which is
//! the narrower claim it was written for; the pose half is
//! `runtime/inf-player/tests/pose_parity.rs`.

use std::collections::BTreeMap;

use glam::DVec2;
use uuid::Uuid;

use inf_anim::state_machine::{CmpOp, SmCondition, SmState, SmTransition, StateMachine};
use inf_ecs::components::AnimStateMachine;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};

fn spawn_sm_entity(doc: &mut SceneDoc, guid: Uuid, sm: Uuid) -> inf_ecs::Entity {
    let e = doc.world_mut().spawn_with_guid(guid, "hero", None);
    doc.world_mut()
        .world_mut()
        .entity_mut(e)
        .insert(AnimStateMachine {
            sm: Some(sm),
            params_from_vars: true,
            ..Default::default()
        });
    e
}

#[test]
fn unconditional_machine_advances_in_the_tick() {
    let mut doc = SceneDoc::new();
    let entity_guid = Uuid::from_u128(0xA0);
    let sm_guid = Uuid::from_u128(0x5A);
    let e = spawn_sm_entity(&mut doc, entity_guid, sm_guid);

    // idle → walk, unconditional (no conditions, no exit-time) → fires at once.
    let machine = StateMachine {
        states: vec![
            SmState::clip("idle", [1; 16]),
            SmState::clip("walk", [2; 16]),
        ],
        transitions: vec![SmTransition {
            from: 0,
            to: 1,
            duration: 0.0,
            conditions: vec![],
            exit_time: None,
        }],
        entry: 0,
    };
    let mut machines = BTreeMap::new();
    machines.insert(sm_guid, machine);

    // No actors: the SM entity is not a blueprint → params default 0, but the
    // machine (with no conditions) still steps.
    let mut session = SimSession::enter(&mut doc, vec![], DVec2::ZERO, SIM_HZ);
    session.set_state_machines(machines);
    session.step_once(&mut doc, SimInput::default());

    let asm = doc.world().world().get::<AnimStateMachine>(e).unwrap();
    assert!(asm.runtime.started, "runtime must be entered");
    assert_eq!(
        asm.runtime.current, 1,
        "unconditional transition should fire"
    );
}

#[test]
fn unresolved_machine_is_left_untouched() {
    // An `AnimStateMachine` whose `sm` GUID is NOT registered must not advance.
    let mut doc = SceneDoc::new();
    let e = spawn_sm_entity(&mut doc, Uuid::from_u128(0xB0), Uuid::from_u128(0x99));

    let mut session = SimSession::enter(&mut doc, vec![], DVec2::ZERO, SIM_HZ);
    session.set_state_machines(BTreeMap::new()); // nothing resolvable
    session.step_once(&mut doc, SimInput::default());

    let asm = doc.world().world().get::<AnimStateMachine>(e).unwrap();
    assert!(!asm.runtime.started, "an unresolved machine must not step");
}

#[test]
fn condition_machine_holds_until_its_variable_flips() {
    // A machine gated on `moving > 0.5`. With no actor the variable is absent
    // (reads 0), so it must stay in idle across steps.
    let mut doc = SceneDoc::new();
    let entity_guid = Uuid::from_u128(0xC0);
    let sm_guid = Uuid::from_u128(0x77);
    let e = spawn_sm_entity(&mut doc, entity_guid, sm_guid);

    let machine = StateMachine {
        states: vec![
            SmState::clip("idle", [1; 16]),
            SmState::clip("walk", [2; 16]),
        ],
        transitions: vec![SmTransition {
            from: 0,
            to: 1,
            duration: 0.2,
            conditions: vec![SmCondition {
                var: "moving".into(),
                op: CmpOp::Gt,
                value: 0.5,
            }],
            exit_time: None,
        }],
        entry: 0,
    };
    let mut machines = BTreeMap::new();
    machines.insert(sm_guid, machine);

    let mut session = SimSession::enter(&mut doc, vec![], DVec2::ZERO, SIM_HZ);
    session.set_state_machines(machines);
    for _ in 0..10 {
        session.step_once(&mut doc, SimInput::default());
    }

    let asm = doc.world().world().get::<AnimStateMachine>(e).unwrap();
    assert_eq!(asm.runtime.current, 0, "condition never met → stays idle");
}
