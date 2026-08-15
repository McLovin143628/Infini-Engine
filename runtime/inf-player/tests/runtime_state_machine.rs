//! P11.2 runtime parity: the shipped player's `runtime_sim` steps animation state
//! machines exactly like the editor Simulate (preview == shipped). An entity with
//! an `AnimStateMachine` whose `sm` GUID is registered advances each fixed step;
//! conditions read the actor's Blueprint variables (here: none → params default
//! 0). Pure Ring-0, headless CI.

use std::collections::BTreeMap;

use glam::DVec2;
use uuid::Uuid;

use inf_anim::state_machine::{CmpOp, SmState, SmTransition, StateMachine};
use inf_ecs::components::AnimStateMachine;
use inf_ecs::EcsWorld;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

fn world_with_sm(entity: Uuid, sm: Uuid) -> EcsWorld {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(entity, "Hero", None);
    world.world_mut().entity_mut(e).insert(AnimStateMachine {
        sm: Some(sm),
        params_from_vars: true,
        ..Default::default()
    });
    world.mark_dirty();
    world
}

fn sm_current(sim: &RuntimeSim, entity: Uuid) -> usize {
    let e = sim.world().entity_of(entity).expect("entity");
    sim.world()
        .world()
        .get::<AnimStateMachine>(e)
        .expect("AnimStateMachine")
        .runtime
        .current
}

#[test]
fn runtime_state_machine_advances() {
    let ent = Uuid::from_u128(0x1120_0001);
    let sm_guid = Uuid::from_u128(0x1120_005A);
    let world = world_with_sm(ent, sm_guid);

    let machine = StateMachine {
        states: vec![
            SmState::clip("idle", [1; 16]),
            SmState::clip("walk", [2; 16]),
        ],
        transitions: vec![SmTransition::new(0, 1, 0.0)],
        entry: 0,
        ..Default::default()
    };
    let mut machines = BTreeMap::new();
    machines.insert(sm_guid, machine);

    let mut sim = RuntimeSim::new(world, vec![], DVec2::ZERO, 60.0);
    sim.set_state_machines(machines);
    sim.step_once(RuntimeInput::default());

    assert_eq!(sm_current(&sim, ent), 1, "unconditional transition fires");
}

#[test]
fn runtime_condition_holds_without_the_variable() {
    let ent = Uuid::from_u128(0x1120_0002);
    let sm_guid = Uuid::from_u128(0x1120_0077);
    let world = world_with_sm(ent, sm_guid);

    let machine = StateMachine {
        states: vec![
            SmState::clip("idle", [1; 16]),
            SmState::clip("walk", [2; 16]),
        ],
        transitions: vec![SmTransition::on(0, 1, 0.2, "moving", CmpOp::Gt, 0.5)],
        entry: 0,
        ..Default::default()
    };
    let mut machines = BTreeMap::new();
    machines.insert(sm_guid, machine);

    let mut sim = RuntimeSim::new(world, vec![], DVec2::ZERO, 60.0);
    sim.set_state_machines(machines);
    for _ in 0..10 {
        sim.step_once(RuntimeInput::default());
    }
    assert_eq!(sm_current(&sim, ent), 0, "condition never met → stays idle");
}
