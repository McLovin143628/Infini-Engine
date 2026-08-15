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

/// **The audio log is bounded, and it says what it dropped** (Hardening D).
///
/// `audio_log` accumulated the whole session's drained command stream **in the
/// shipped player** — at least one `SetListener` per fixed step whenever a
/// listener exists, i.e. ~216 000 commands an hour — for a value whose only
/// consumer is `audio_command_log()`, a test accessor. It is an
/// `inf_core::BoundedLog` now: a hard ceiling plus a count of what fell off the
/// front, so a test that reasons about the *first* command can tell that it is
/// reading a tail.
#[test]
fn the_audio_command_log_cannot_grow_without_bound() {
    let listener = Uuid::from_u128(0xD0A0_11E5);
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(listener, "Listener", None);
    world
        .world_mut()
        .entity_mut(e)
        .insert(Transform::from_translation(DVec3::ZERO));
    world
        .world_mut()
        .entity_mut(e)
        .insert(inf_ecs::components::AudioListener { active: true });
    world.mark_dirty();

    let mut sim = RuntimeSim::new(world, vec![], DVec2::ZERO, 60.0);
    let steps = inf_core::DEFAULT_LOG_CAPACITY + 512;
    for _ in 0..steps {
        sim.step_once(RuntimeInput::default());
    }

    // The premise: this really is a per-step producer. Without it the bound below
    // would be satisfied by a stream that never grew at all.
    assert!(
        sim.dropped_audio_commands() > 0,
        "the fixture must actually overflow the ring — {} steps produced no eviction",
        steps
    );
    assert!(
        sim.audio_command_log().len() <= inf_core::DEFAULT_LOG_CAPACITY,
        "the retained window is {} commands, past the {} ceiling",
        sim.audio_command_log().len(),
        inf_core::DEFAULT_LOG_CAPACITY
    );
    // And the newest command survives: a ring that kept the head would be worse
    // than no ring, because the interesting end of a command stream is the end.
    assert!(
        matches!(
            sim.audio_command_log().last(),
            Some(AudioCommand::SetListener(_))
        ),
        "the most recent step's command must be retained"
    );
}
