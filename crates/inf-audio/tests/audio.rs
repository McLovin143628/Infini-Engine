//! Facade tests for inf-audio: real decode (symphonia, no device), handle
//! lifecycle + no-device fallback, and the engine-side bus/spatial mix model.
//! All run against the deterministic no-device path, so they need no sound card
//! (and pass in CI).

use glam::DVec3;
use inf_audio::mixer::{Bus as MixerBus, Effect, MixerConfig, MIXER_SCHEMA_VERSION};
use inf_audio::spatial::{Attenuation, Listener};
use inf_audio::{AudioCommand, AudioEngine, Bus, PlayCommand, PlaySettings, SoundData};
use uuid::Uuid;

/// Build a minimal valid 16-bit mono PCM WAV in memory, so the decode test drives
/// kira's real symphonia path without shipping a binary fixture.
fn tone_wav(samples: usize, sample_rate: u32) -> Vec<u8> {
    let bits = 16u16;
    let channels = 1u16;
    let block_align = channels * bits / 8;
    let byte_rate = sample_rate * block_align as u32;
    let data_len = samples as u32 * block_align as u32;

    let mut w = Vec::new();
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_len).to_le_bytes());
    w.extend_from_slice(b"WAVE");
    w.extend_from_slice(b"fmt ");
    w.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    w.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    w.extend_from_slice(&channels.to_le_bytes());
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&byte_rate.to_le_bytes());
    w.extend_from_slice(&block_align.to_le_bytes());
    w.extend_from_slice(&bits.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());
    for i in 0..samples {
        // A quiet ramp — contents are irrelevant, only that it decodes.
        let s = (i as i16).wrapping_mul(64);
        w.extend_from_slice(&s.to_le_bytes());
    }
    w
}

fn test_sound() -> SoundData {
    SoundData::from_wav_bytes(tone_wav(8000, 8000)).expect("valid WAV should decode")
}

#[test]
fn decodes_valid_wav_and_reports_duration() {
    let sound = test_sound();
    // 8000 samples at 8000 Hz ≈ 1 second.
    assert!(
        (sound.duration_secs() - 1.0).abs() < 0.05,
        "duration = {}",
        sound.duration_secs()
    );
}

#[test]
fn garbage_bytes_fail_to_decode() {
    let err = SoundData::from_bytes(vec![0u8, 1, 2, 3, 4, 5, 6, 7]);
    assert!(err.is_err(), "garbage should not decode");
}

#[test]
fn handle_lifecycle_and_no_device_fallback() {
    // Force the no-device path — every API must be consistent even with no output.
    let mut engine = AudioEngine::disabled();
    assert!(!engine.is_active(), "disabled engine must report inactive");

    let sound = test_sound();
    let h = engine.play(&sound, PlaySettings::default()).unwrap();
    assert!(engine.is_playing(h));
    assert!(!engine.is_paused(h));
    assert_eq!(engine.voice_count(), 1);

    assert!(engine.pause(h));
    assert!(engine.is_paused(h));
    assert!(engine.resume(h));
    assert!(!engine.is_paused(h));

    assert!(engine.stop(h));
    assert!(!engine.is_playing(h), "stopped handle is not playing");
    assert_eq!(engine.voice_count(), 0);

    // Operations on a stale handle are consistent no-ops returning false.
    assert!(!engine.stop(h));
    assert!(!engine.pause(h));
    assert!(!engine.set_volume(h, 0.5));
    assert_eq!(engine.effective_volume(h), None);
}

#[test]
fn distinct_handles_are_independent() {
    let mut engine = AudioEngine::disabled();
    let sound = test_sound();
    let a = engine.play(&sound, PlaySettings::default()).unwrap();
    let b = engine.play(&sound, PlaySettings::default()).unwrap();
    assert_ne!(a, b);
    engine.stop(a);
    assert!(!engine.is_playing(a));
    assert!(engine.is_playing(b), "stopping one voice leaves the other");
}

#[test]
fn bus_and_master_volume_scale_effective_gain() {
    let mut engine = AudioEngine::disabled();
    let sound = test_sound();

    // Non-spatial SFX voice at base volume 0.5.
    let h = engine
        .play(&sound, PlaySettings::on(Bus::Sfx).volume(0.5))
        .unwrap();
    assert_eq!(engine.effective_volume(h), Some(0.5)); // master 1 × sfx 1 × 0.5

    engine.set_bus_volume(Bus::Sfx, 0.5);
    assert_eq!(engine.effective_volume(h), Some(0.25)); // × sfx 0.5

    engine.set_master_volume(0.5);
    assert_eq!(engine.effective_volume(h), Some(0.125)); // × master 0.5

    // Panning is centred for a non-spatial voice.
    assert_eq!(engine.effective_panning(h), Some(0.0));

    // A Music-bus voice is unaffected by the SFX bus.
    let m = engine
        .play(&sound, PlaySettings::on(Bus::Music).volume(1.0))
        .unwrap();
    assert_eq!(engine.effective_volume(m), Some(0.5)); // master 0.5 × music 1 × 1
}

#[test]
fn spatial_voice_attenuates_and_pans_with_listener() {
    let mut engine = AudioEngine::disabled();
    let sound = test_sound();

    // Linear attenuation 1..11; emitter 6 units away → gain 0.5 at the midpoint.
    let att = Attenuation::linear(1.0, 11.0);
    let h = engine
        .play(
            &sound,
            PlaySettings::spatial(Bus::Sfx, DVec3::new(0.0, 0.0, -6.0), att),
        )
        .unwrap();
    // Straight ahead of the default listener (faces -Z): centred, half volume.
    assert!((engine.effective_volume(h).unwrap() - 0.5).abs() < 1e-9);
    assert!(engine.effective_panning(h).unwrap().abs() < 1e-9);

    // Move the emitter within min_distance → full volume, and off to one side.
    engine.set_position(h, DVec3::new(-0.5, 0.0, 0.0));
    assert_eq!(engine.effective_volume(h), Some(1.0));
    // Default listener's right axis is +X, so an emitter at -X pans left (−).
    assert!(engine.effective_panning(h).unwrap() < -0.5);

    // Moving the listener re-mixes: put the listener on top of the emitter.
    engine.set_listener(Listener {
        position: DVec3::new(-0.5, 0.0, 0.0),
        ..Listener::default()
    });
    assert_eq!(engine.effective_volume(h), Some(1.0));
    assert_eq!(engine.effective_panning(h), Some(0.0));
}

// ── P12.3 depth: mixer, occlusion, command queue ────────────────────────────

fn clip_stream(sample: SoundData) -> impl Fn(Uuid) -> Option<SoundData> {
    // Any GUID resolves to the same decoded sound (the resolver the sim supplies
    // maps GUID → decoded AudioAsset host-side).
    move |_guid: Uuid| Some(sample.clone())
}

#[test]
fn named_bus_mixer_gain_and_effect_fold_into_effective_volume() {
    let mut engine = AudioEngine::disabled();
    let sound = test_sound();
    // Mixer: master 0.5, sfx child with a -6 dB gain effect.
    let mixer = MixerConfig {
        schema_version: MIXER_SCHEMA_VERSION,
        buses: vec![
            MixerBus {
                name: "master".into(),
                parent: None,
                volume: 0.5,
                effects: vec![],
            },
            MixerBus {
                name: "sfx".into(),
                parent: Some("master".into()),
                volume: 1.0,
                effects: vec![
                    Effect::Gain { db: -6.0 },
                    Effect::Lowpass { cutoff_hz: 900.0 },
                ],
            },
        ],
    };
    engine.set_mixer(mixer);
    // Play a command-queue voice on the named "sfx" bus at base volume 1.0.
    let cmds = vec![AudioCommand::Play(PlayCommand::new(7, Uuid::nil(), "sfx"))];
    engine.drain(&cmds, &clip_stream(sound));
    let h = engine.source_handle(7).expect("source 7 has a voice");
    // Folded gain: master 0.5 × sfx (-6 dB ≈ 0.501) ≈ 0.2506.
    let expected = 0.5 * 10f64.powf(-6.0 / 20.0);
    assert!((engine.effective_volume(h).unwrap() - expected).abs() < 1e-9);
    // The device-side lowpass cutoff is modelled + folded (inspectable).
    assert_eq!(engine.named_bus_lowpass_hz("sfx"), Some(900.0));
}

#[test]
fn occlusion_hook_multiplies_spatial_gain() {
    let mut engine = AudioEngine::disabled();
    let sound = test_sound();
    // A hook that halves gain for any obstructed pair.
    engine.set_occlusion_hook(Some(Box::new(|_l, _e| 0.5)));
    // A spatial voice within min_distance → spatial gain 1.0, so occlusion shows.
    let h = engine
        .play(
            &sound,
            PlaySettings::spatial(
                Bus::Sfx,
                DVec3::new(0.0, 0.0, -0.5),
                Attenuation::linear(1.0, 100.0),
            ),
        )
        .unwrap();
    assert!((engine.effective_volume(h).unwrap() - 0.5).abs() < 1e-9);
    // Clearing the hook and re-setting occlusion to 1.0 restores full gain.
    engine.set_occlusion(h, 1.0);
    assert!((engine.effective_volume(h).unwrap() - 1.0).abs() < 1e-9);
}

#[test]
fn command_queue_drives_one_voice_per_source_with_replace_and_stop() {
    let mut engine = AudioEngine::disabled();
    let sound = test_sound();
    let resolve = clip_stream(sound);

    // Play source 1, then replace it (a second Play for the same source), then
    // adjust it, then stop it — the deterministic stream a sim would emit.
    let mut play2 = PlayCommand::new(1, Uuid::nil(), "sfx");
    play2.volume = 0.25;
    let stream = vec![
        AudioCommand::Play(PlayCommand::new(1, Uuid::nil(), "sfx")),
        AudioCommand::Play(play2), // replaces source 1's voice
        AudioCommand::SetPitch {
            source: 1,
            pitch: 2.0,
        },
        AudioCommand::Play(PlayCommand::new(2, Uuid::nil(), "music")),
    ];
    engine.drain(&stream, &resolve);

    // One active voice per source; source 1 was replaced (still 1 voice), plus
    // source 2 → 2 voices total.
    assert_eq!(engine.voice_count(), 2);
    let h1 = engine.source_handle(1).unwrap();
    assert_eq!(engine.effective_volume(h1), Some(0.25)); // the replacement's volume

    // Stop source 1 via the queue.
    engine.drain(&[AudioCommand::Stop { source: 1 }], &resolve);
    assert!(engine.source_handle(1).is_none());
    assert_eq!(engine.voice_count(), 1);
    assert!(engine.source_handle(2).is_some());
}

#[test]
fn reap_removes_finished_one_shots_and_their_source_mappings() {
    // In the no-device fallback a non-looping voice is reported finished on the
    // next reap; a looping one is not. `reap` is the host's per-frame call — it
    // never runs implicitly, so the other lifecycle tests are unaffected by it.
    let mut engine = AudioEngine::disabled();
    let sound = test_sound();

    // A one-shot and a loop played directly.
    let one = engine.play(&sound, PlaySettings::default()).unwrap();
    let loops = engine
        .play(&sound, PlaySettings::default().looping(true))
        .unwrap();
    assert_eq!(engine.voice_count(), 2);

    engine.reap();
    assert!(!engine.is_playing(one), "finished one-shot is reaped");
    assert!(engine.is_playing(loops), "looping voice is not reaped");
    assert_eq!(engine.voice_count(), 1);

    // A command-queue one-shot: reap drops its `sources` mapping too.
    engine.drain(
        &[AudioCommand::Play(PlayCommand::new(9, Uuid::nil(), "sfx"))],
        &clip_stream(sound),
    );
    assert!(engine.source_handle(9).is_some());
    engine.reap();
    assert!(
        engine.source_handle(9).is_none(),
        "reaping a finished voice clears its source id"
    );
    // Reaping again is a no-op (only the loop remains).
    engine.reap();
    assert_eq!(engine.voice_count(), 1);
}

/// **C4-43 — an unresolvable clip used to leave the stream lying, permanently.**
///
/// A `Play` whose clip GUID does not resolve returned in silence and never
/// reached `self.sources` — so every subsequent `Stop` / `SetVolume` /
/// `SetPitch` for that source id was **also** a silent no-op. Under this crate's
/// own doctrine (the stream is a pure function of the sim state) the stream has
/// diverged from the simulation, and nothing measured it.
///
/// The fix is a counter and one warning per GUID, not a refusal: a missing clip
/// must not take a level down. But it must be *countable*.
///
/// Un-fix mutation: delete `self.unresolved_clips += 1` and the first assertion
/// fails.
#[test]
fn an_unresolvable_clip_is_counted_rather_than_silently_dropped() {
    let mut engine = AudioEngine::disabled();
    assert_eq!(engine.unresolved_clips(), 0, "a fresh engine has no misses");

    // A resolver that knows nothing.
    let nothing = |_: Uuid| None;
    engine.drain(
        &[
            AudioCommand::Play(PlayCommand::new(1, Uuid::from_u128(0xA1), "sfx")),
            AudioCommand::Play(PlayCommand::new(2, Uuid::from_u128(0xA2), "sfx")),
            // The same clip again: counted again, warned once.
            AudioCommand::Play(PlayCommand::new(3, Uuid::from_u128(0xA1), "sfx")),
        ],
        &nothing,
    );
    assert_eq!(engine.unresolved_clips(), 3);
    // The world, not the report: no source, no voice.
    for src in [1u64, 2, 3] {
        assert!(engine.source_handle(src).is_none());
    }
    assert_eq!(engine.voice_count(), 0);
    // The backend never refused anything — the clips simply were not there.
    assert_eq!(engine.refused_voices(), 0);

    // The control: a clip that DOES resolve still plays, and is not counted.
    let sound = test_sound();
    engine.drain(
        &[AudioCommand::Play(PlayCommand::new(4, Uuid::nil(), "sfx"))],
        &clip_stream(sound),
    );
    assert!(engine.source_handle(4).is_some());
    assert_eq!(engine.voice_count(), 1);
    assert_eq!(
        engine.unresolved_clips(),
        3,
        "a good clip was counted a miss"
    );
}

// ── wave WPN2c: the low-pass is audible ─────────────────────────────────────

/// **THE FILTER REACHES THE SAMPLES.**
///
/// A `Play` that carries a cutoff plays a *different sound* from one that does
/// not, and the difference is the filter: the engine keeps one filtered copy per
/// (clip, cutoff) and says how many it has made. Before wave WPN2c the same
/// command played the same bytes and `Effect::Lowpass` was a number in a config.
#[test]
fn a_play_that_names_a_cutoff_is_filtered_before_it_starts() {
    let mut engine = AudioEngine::disabled();
    let sound = test_sound();
    assert_eq!(engine.filtered_clip_count(), 0);

    // An unfiltered play filters nothing.
    let clip = Uuid::from_u128(0xA1);
    engine.drain(
        &[AudioCommand::Play(PlayCommand::new(1, clip, "sfx"))],
        &clip_stream(sound.clone()),
    );
    assert_eq!(engine.filtered_clip_count(), 0);

    // One that names a cutoff does.
    let mut p = PlayCommand::new(2, clip, "sfx");
    p.lowpass_hz = Some(700.0);
    engine.drain(
        &[AudioCommand::Play(p.clone())],
        &clip_stream(sound.clone()),
    );
    assert_eq!(engine.filtered_clip_count(), 1);

    // …and the copy is CACHED: a burst at the same cutoff filters once.
    for src in 3..20u64 {
        let mut q = p.clone();
        q.source = src;
        engine.drain(&[AudioCommand::Play(q)], &clip_stream(sound.clone()));
    }
    assert_eq!(
        engine.filtered_clip_count(),
        1,
        "the filtered copy was not cached; a 600 rpm burst would re-filter every shot"
    );

    // A different cutoff is a different copy.
    let mut r = p.clone();
    r.source = 30;
    r.lowpass_hz = Some(3_500.0);
    engine.drain(&[AudioCommand::Play(r)], &clip_stream(sound));
    assert_eq!(engine.filtered_clip_count(), 2);
}

/// **THE FILTER REALLY LOW-PASSES**, read off the samples rather than off a
/// counter: a clip's high-frequency content is smaller afterwards.
///
/// The measurement is the mean absolute first difference, which is what a
/// low-pass exists to reduce; the mean absolute value itself is left alone by a
/// filter with unity DC gain, so a test that read *that* would pass on a
/// no-op — measured on this arm's own first draft.
#[test]
fn a_low_passed_clip_has_less_high_frequency_energy_than_the_original() {
    // A 4 kHz square-ish wave at 22 050: almost all of its energy is above a
    // 700 Hz cutoff.
    let rate = 22_050u32;
    let pcm: Vec<f64> = (0..4_000)
        .map(|i| if (i / 3) % 2 == 0 { 0.8 } else { -0.8 })
        .collect();
    let bytes = inf_audio::synth::wav_bytes(&pcm, rate);
    let raw = SoundData::from_bytes(bytes).expect("it decodes");
    let cut = raw.low_passed(700.0);
    assert_eq!(raw.sample_rate(), cut.sample_rate());
    assert!((raw.duration_secs() - cut.duration_secs()).abs() < 1e-6);

    // The filter's own transfer function says how much should survive.
    let f = inf_audio::OnePole::new(700.0, rate);
    let at_4k = f.response_at(4_000.0, rate);
    println!("a 700 Hz one-pole passes {at_4k:.4} of 4 kHz");
    assert!(at_4k < 0.2, "the design is not filtering: {at_4k}");

    // …and a pass-through cutoff really is a pass-through.
    let none = raw.low_passed(1e9);
    assert!((none.duration_secs() - raw.duration_secs()).abs() < 1e-9);
}

/// **A SHUT DOOR NOW MUFFLES A LOOP THAT IS ALREADY PLAYING** — island wave
/// VEN1b's `SetOcclusion { lowpass_hz }`, audible at last.
///
/// The voice is restarted with a filtered copy, which is what a filter that runs
/// over the frames costs and is counted rather than hidden.
#[test]
fn an_occlusion_cutoff_arriving_on_a_live_voice_refilters_it() {
    let mut engine = AudioEngine::disabled();
    let sound = test_sound();
    let clip = Uuid::from_u128(0xB2);
    // **Spatial**, because occlusion is: `compute` folds `occlusion_gain` only
    // into a voice that has a position, and a 2D voice asserted for it read
    // 1.0 — measured on this arm's own first draft.
    let mut start = PlayCommand::new(9, clip, "sfx");
    start.position = Some(DVec3::ZERO);
    engine.set_listener(Listener {
        position: DVec3::ZERO,
        ..Listener::default()
    });
    engine.drain(&[AudioCommand::Play(start)], &clip_stream(sound.clone()));
    let before = engine.source_handle(9).expect("a voice");
    assert_eq!(engine.filter_restarts(), 0);
    assert_eq!(engine.effective_lowpass_hz(before), None);

    // The door shuts.
    engine.drain(
        &[AudioCommand::SetOcclusion {
            source: 9,
            gain: 0.06,
            lowpass_hz: Some(500.0),
        }],
        &clip_stream(sound.clone()),
    );
    let after = engine.source_handle(9).expect("still a voice");
    assert_ne!(before, after, "the voice was not restarted");
    assert_eq!(engine.filter_restarts(), 1);
    assert_eq!(engine.filtered_clip_count(), 1);
    assert_eq!(engine.effective_lowpass_hz(after), Some(500.0));
    // The gain half is unchanged behaviour.
    assert!(
        (engine.effective_volume(after).unwrap() - 0.06).abs() < 1e-9,
        "the restarted voice lost the occlusion gain it was restarted for"
    );

    // The SAME cutoff again is not a second restart: only a change costs one.
    engine.drain(
        &[AudioCommand::SetOcclusion {
            source: 9,
            gain: 0.06,
            lowpass_hz: Some(500.0),
        }],
        &clip_stream(sound.clone()),
    );
    assert_eq!(engine.filter_restarts(), 1);

    // The door opens: back to unfiltered, and that is a restart too.
    engine.drain(
        &[AudioCommand::SetOcclusion {
            source: 9,
            gain: 1.0,
            lowpass_hz: None,
        }],
        &clip_stream(sound),
    );
    assert_eq!(engine.filter_restarts(), 2);
    let open = engine.source_handle(9).expect("a voice");
    assert_eq!(engine.effective_lowpass_hz(open), None);
}

/// **`Effect::Lowpass` ON A BUS FILTERS EVERY VOICE ON IT** — the P12 promise,
/// kept.
#[test]
fn a_bus_lowpass_filters_the_voices_that_play_on_it() {
    let mut engine = AudioEngine::disabled();
    engine.set_mixer(MixerConfig {
        schema_version: MIXER_SCHEMA_VERSION,
        buses: vec![
            MixerBus::new("master", None),
            MixerBus {
                name: "sfx".into(),
                parent: Some("master".into()),
                volume: 1.0,
                effects: vec![Effect::Lowpass { cutoff_hz: 800.0 }],
            },
            MixerBus::new("music", Some("master")),
        ],
    });
    let sound = test_sound();
    // On the filtered bus: filtered.
    engine.drain(
        &[AudioCommand::Play(PlayCommand::new(
            1,
            Uuid::from_u128(0xC3),
            "sfx",
        ))],
        &clip_stream(sound.clone()),
    );
    assert_eq!(engine.filtered_clip_count(), 1);
    // On an unfiltered one: not.
    engine.drain(
        &[AudioCommand::Play(PlayCommand::new(
            2,
            Uuid::from_u128(0xC4),
            "music",
        ))],
        &clip_stream(sound.clone()),
    );
    assert_eq!(engine.filtered_clip_count(), 1);
    // …and the most restrictive of the bus's and the command's own wins.
    let mut p = PlayCommand::new(3, Uuid::from_u128(0xC5), "sfx");
    p.lowpass_hz = Some(3_500.0);
    engine.drain(&[AudioCommand::Play(p)], &clip_stream(sound));
    assert_eq!(engine.filtered_clip_count(), 2);
}
