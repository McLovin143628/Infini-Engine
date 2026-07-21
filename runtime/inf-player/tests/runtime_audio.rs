//! P12.3 runtime parity: the shipped player's `runtime_sim` runs the same audio
//! step as the editor Simulate — preview == shipped for audio too. The
//! deterministic-queue payoff: a headless test asserts the drained **command
//! stream** (play params over scripted steps), not device output (the engine runs
//! its no-device fallback here). Pure Ring-0, headless CI.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_audio::AudioCommand;
use inf_ecs::components::{AudioSource, DistanceModel, Transform};
use inf_ecs::EcsWorld;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

#[test]
fn runtime_autoplay_source_emits_one_deterministic_play_command() {
    let emitter = Uuid::from_u128(0xD0A0_0001);
    let clip = Uuid::from_u128(0xD0A0_C11B);

    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(emitter, "Emitter", None);
    world
        .world_mut()
        .entity_mut(e)
        .insert(Transform::from_translation(DVec3::new(-4.0, 1.0, 0.0)));
    world.world_mut().entity_mut(e).insert(AudioSource {
        clip: Some(clip),
        bus: "sfx".into(),
        volume: 0.75,
        pitch: 1.25,
        looping: false,
        spatial: true,
        distance_model: DistanceModel::Exponential,
        rolloff: 2.0,
        autoplay: true,
        ..Default::default()
    });
    world.mark_dirty();

    let mut sim = RuntimeSim::new(world, vec![], DVec2::ZERO, 60.0);
    for _ in 0..5 {
        sim.step_once(RuntimeInput::default());
    }

    let plays: Vec<&AudioCommand> = sim
        .audio_command_log()
        .iter()
        .filter(|c| matches!(c, AudioCommand::Play(_)))
        .collect();
    assert_eq!(plays.len(), 1, "autoplay enqueues exactly one Play");

    let AudioCommand::Play(p) = plays[0] else {
        unreachable!()
    };
    assert_eq!(p.clip, clip);
    assert_eq!(p.bus, "sfx");
    assert_eq!(p.volume, 0.75);
    assert_eq!(p.pitch, 1.25);
    assert_eq!(p.position, Some(DVec3::new(-4.0, 1.0, 0.0)));
}
