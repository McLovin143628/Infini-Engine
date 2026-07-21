//! Facade tests for inf-audio: real decode (symphonia, no device), handle
//! lifecycle + no-device fallback, and the engine-side bus/spatial mix model.
//! All run against the deterministic no-device path, so they need no sound card
//! (and pass in CI).

use glam::DVec3;
use inf_audio::spatial::{Attenuation, Listener};
use inf_audio::{AudioEngine, Bus, PlaySettings, SoundData};

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
    let h = engine.play(&sound, PlaySettings::default());
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
    let a = engine.play(&sound, PlaySettings::default());
    let b = engine.play(&sound, PlaySettings::default());
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
    let h = engine.play(&sound, PlaySettings::on(Bus::Sfx).volume(0.5));
    assert_eq!(engine.effective_volume(h), Some(0.5)); // master 1 × sfx 1 × 0.5

    engine.set_bus_volume(Bus::Sfx, 0.5);
    assert_eq!(engine.effective_volume(h), Some(0.25)); // × sfx 0.5

    engine.set_master_volume(0.5);
    assert_eq!(engine.effective_volume(h), Some(0.125)); // × master 0.5

    // Panning is centred for a non-spatial voice.
    assert_eq!(engine.effective_panning(h), Some(0.0));

    // A Music-bus voice is unaffected by the SFX bus.
    let m = engine.play(&sound, PlaySettings::on(Bus::Music).volume(1.0));
    assert_eq!(engine.effective_volume(m), Some(0.5)); // master 0.5 × music 1 × 1
}

#[test]
fn spatial_voice_attenuates_and_pans_with_listener() {
    let mut engine = AudioEngine::disabled();
    let sound = test_sound();

    // Linear attenuation 1..11; emitter 6 units away → gain 0.5 at the midpoint.
    let att = Attenuation::linear(1.0, 11.0);
    let h = engine.play(
        &sound,
        PlaySettings::spatial(Bus::Sfx, DVec3::new(0.0, 0.0, -6.0), att),
    );
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
