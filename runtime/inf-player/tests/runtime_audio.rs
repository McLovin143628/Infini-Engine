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

/// **A despawned emitter is Stopped exactly once, and a live one never is**
/// (Hardening Wave G).
///
/// The despawn sweep is the only consumer of the "which entities exist" set the
/// audio step built, and that set used to be *every guid in the world*, minted
/// into a fresh `BTreeSet` per fixed step — thirteen thousand inserts on a
/// furnished town so that a handful of emitters could each be asked one
/// question (lens 3 P34). It is now the started emitters that the same walk saw
/// alive.
///
/// Nothing covered this transition. The narrowing broke autoplay on its first
/// run — an emitter that started *this* step was not yet in the narrowed set,
/// so the sweep Stopped it on the step it began and autoplay re-fired it for
/// ever — and the arm that caught it was counting `Play`s, not `Stop`s. This is
/// the missing half: both directions, with the negative asserted, because
/// "the Stop arrives" is satisfied by a sweep that Stops everything.
#[test]
fn a_despawned_emitter_is_stopped_and_a_live_one_is_left_alone() {
    let doomed = Uuid::from_u128(0xD0A0_0011);
    let survivor = Uuid::from_u128(0xD0A0_0012);

    let mut world = EcsWorld::new();
    let place = |w: &mut EcsWorld, guid: Uuid, x: f64| {
        let e = w.spawn_with_guid(guid, "Emitter", None);
        w.world_mut()
            .entity_mut(e)
            .insert(Transform::from_translation(DVec3::new(x, 0.0, 0.0)))
            .insert(AudioSource {
                clip: Some(Uuid::from_u128(0xD0A0_C11B)),
                autoplay: true,
                spatial: true,
                ..Default::default()
            });
        e
    };
    let e_doomed = place(&mut world, doomed, -2.0);
    place(&mut world, survivor, 2.0);
    world.mark_dirty();

    let mut sim = RuntimeSim::new(world, vec![], DVec2::ZERO, 60.0);
    for _ in 0..3 {
        sim.step_once(RuntimeInput::default());
    }
    let stops = |sim: &RuntimeSim| -> Vec<u64> {
        sim.audio_command_log()
            .iter()
            .filter_map(|c| match c {
                AudioCommand::Stop { source } => Some(*source),
                _ => None,
            })
            .collect()
    };
    assert!(
        stops(&sim).is_empty(),
        "a live emitter was Stopped while it still exists"
    );
    let plays_before = sim
        .audio_command_log()
        .iter()
        .filter(|c| matches!(c, AudioCommand::Play(_)))
        .count();
    assert_eq!(
        plays_before, 2,
        "both emitters must have autoplayed once, or the fixture proves nothing"
    );

    sim.world_mut().despawn(e_doomed);
    sim.world_mut().mark_dirty();
    for _ in 0..3 {
        sim.step_once(RuntimeInput::default());
    }

    let after = stops(&sim);
    assert_eq!(
        after.len(),
        1,
        "expected exactly one Stop for the despawned emitter, got {after:?} — a \
         second one means the sweep forgot it and found it 'gone' again"
    );
    assert_eq!(
        sim.audio_command_log()
            .iter()
            .filter(|c| matches!(c, AudioCommand::Play(_)))
            .count(),
        plays_before,
        "an emitter was re-played after the sweep — it was Stopped and forgotten \
         while still alive, so autoplay started it again"
    );
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
