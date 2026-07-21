//! End-to-end Simulate proof for the **P12.3 audio step**: an autoplay
//! `AudioSource` drives a deterministic command stream through the real
//! `SimSession` fixed-step — no device, no Tauri, pure CI.
//!
//! The determinism payoff: we assert the drained **command stream** (the play/
//! stop/set sequence with its params), not any audible output — the audio engine
//! runs its no-device fallback here. A scene-placed emitter with `autoplay` enqueues
//! exactly one `Play` (not one per tick), carrying the component's clip/bus/volume/
//! spatial settings verbatim.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_audio::AudioCommand;
use inf_ecs::components::{AudioListener, AudioSource, DistanceModel, Transform};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};

macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

#[test]
fn autoplay_source_emits_a_deterministic_play_command_exactly_once() {
    let emitter = Uuid::from_u128(0xA0D1_0001);
    let clip = Uuid::from_u128(0xA0D1_C11B);

    let mut doc = SceneDoc::new();
    doc.create_with_guid(emitter, SpawnKind::Empty, "Emitter", None);
    insert!(
        doc,
        emitter,
        Transform::from_translation(DVec3::new(3.0, 0.0, 0.0))
    );
    insert!(
        doc,
        emitter,
        AudioSource {
            clip: Some(clip),
            bus: "music".into(),
            volume: 0.5,
            pitch: 1.0,
            looping: true,
            spatial: true,
            distance_model: DistanceModel::Linear,
            autoplay: true,
            ..Default::default()
        }
    );
    doc.world_mut().propagate();

    // No actors — the audio step iterates the world for autoplay sources directly.
    let mut sim = SimSession::enter(&mut doc, vec![], DVec2::new(0.0, -9.81), SIM_HZ);

    // Three scripted steps: autoplay must fire on the first and never again.
    for _ in 0..3 {
        sim.step_once(&mut doc, SimInput::default());
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
    assert_eq!(p.bus, "music");
    assert_eq!(p.volume, 0.5);
    assert!(p.looping);
    assert!(
        p.position.is_some(),
        "spatial source carries a world position"
    );
    assert_eq!(p.position.unwrap(), DVec3::new(3.0, 0.0, 0.0));
}

#[test]
fn active_listener_sets_the_pose_each_step() {
    let listener = Uuid::from_u128(0xA0D2_0001);

    let mut doc = SceneDoc::new();
    doc.create_with_guid(listener, SpawnKind::Empty, "Ears", None);
    insert!(
        doc,
        listener,
        Transform::from_translation(DVec3::new(0.0, 2.0, 0.0))
    );
    insert!(doc, listener, AudioListener { active: true });
    doc.world_mut().propagate();

    let mut sim = SimSession::enter(&mut doc, vec![], DVec2::ZERO, SIM_HZ);
    sim.step_once(&mut doc, SimInput::default());

    let set_listener = sim
        .audio_command_log()
        .iter()
        .find_map(|c| match c {
            AudioCommand::SetListener(l) => Some(*l),
            _ => None,
        })
        .expect("an active AudioListener sets the pose");
    assert_eq!(set_listener.position, DVec3::new(0.0, 2.0, 0.0));
}
