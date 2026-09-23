//! **The render-to-file door** (wave VEH3e): the offline engine plays through
//! kira's own mixer, clocked by the caller, and what it renders is measured —
//! the device path, Ring-0, with no device.

use uuid::Uuid;

use inf_audio::{AudioCommand, AudioEngine, PlayCommand, SoundData};

const RATE: u32 = 48_000;

/// Goertzel power at `hz` over `x`.
fn power_at(x: &[f64], hz: f64, rate: u32) -> f64 {
    let w = 2.0 * std::f64::consts::PI * hz / f64::from(rate);
    let c = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for v in x {
        let s0 = v + c * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    s1 * s1 + s2 * s2 - c * s1 * s2
}

/// The strongest frequency in `x` on a 5 Hz grid between `lo` and `hi`.
fn peak_hz(x: &[f64], lo: f64, hi: f64, rate: u32) -> f64 {
    let mut best = (0.0f64, lo);
    let mut f = lo;
    while f <= hi {
        let p = power_at(x, f, rate);
        if p > best.0 {
            best = (p, f);
        }
        f += 5.0;
    }
    best.1
}

fn clip(index: u8) -> SoundData {
    SoundData::from_bytes(inf_audio::vehicle_synth::vehicle_clip_wav(index)).expect("it decodes")
}

/// Render `steps` fixed steps of the engine after `cmds`, and keep the LEFT
/// channel (the voices are non-spatial, so both channels are the same).
fn render(
    engine: &mut AudioEngine,
    cmds: &[AudioCommand],
    data: &SoundData,
    steps: usize,
) -> Vec<f64> {
    let d = data.clone();
    engine.drain(cmds, &move |_| Some(d.clone()));
    let mut out = Vec::new();
    for _ in 0..steps {
        let block = engine.render(800);
        assert_eq!(block.len(), 1600, "800 stereo frames");
        out.extend(block.chunks_exact(2).map(|f| f64::from(f[0])));
    }
    out
}

/// **THE WHINE IS HEARD AT THE PITCH THE COMMAND SENT.** The committed whine
/// clip (1 225 Hz at a pitch of one) played through the offline mixer at 48 kHz:
/// the rendered samples peak at 1 225 Hz, and a `SetPitch` of 1.5 moves the
/// peak to 1 837.5 Hz — kira's playback rate, measured in its own output.
#[test]
fn the_offline_mixer_plays_the_pitch_the_command_sent() {
    let mut engine = AudioEngine::offline(RATE);
    assert!(engine.is_active(), "the offline engine is a device");
    let data = clip(15);
    let mut play = PlayCommand::new(7, Uuid::from_u128(1), "sfx");
    play.looping = true;
    let a = render(&mut engine, &[AudioCommand::Play(play)], &data, 30);
    let pa = peak_hz(&a[a.len() / 2..], 800.0, 3_000.0, RATE);
    let b = render(
        &mut engine,
        &[AudioCommand::SetPitch {
            source: 7,
            pitch: 1.5,
        }],
        &data,
        30,
    );
    let pb = peak_hz(&b[b.len() / 2..], 800.0, 3_000.0, RATE);
    let rms = |x: &[f64]| (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
    println!(
        "the whine renders at {pa} Hz, then at {pb} Hz after SetPitch 1.5; rms {:.4}",
        rms(&a)
    );
    assert!(rms(&a) > 0.05, "the mixer rendered silence");
    assert!((pa - 1_225.0).abs() <= 5.0, "pitch 1 peaked at {pa}");
    assert!((pb - 1_837.5).abs() <= 7.5, "pitch 1.5 peaked at {pb}");
    // A Stop silences it within the next block.
    let c = render(&mut engine, &[AudioCommand::Stop { source: 7 }], &data, 4);
    assert!(rms(&c[1600..]) < 1e-4, "the voice outlived its Stop");
}

/// **A V8 GRAIN AT A PITCH IS HEARD FIRING AT (rpm/60)(cyl/2).** The committed
/// cross-plane V8's full-load grain played at `rpm / 2400` for 4 800 rpm: the
/// rendered output's envelope repeats at 320 Hz (4 800 / 60 x 8 / 2), measured
/// off the rendered samples by the strongest line of the rectified signal.
#[test]
fn a_grain_at_a_pitch_fires_at_its_revs() {
    let mut engine = AudioEngine::offline(RATE);
    let data = clip(8); // Grain_P8X_Full
    let mut play = PlayCommand::new(9, Uuid::from_u128(2), "sfx");
    play.looping = true;
    play.pitch = 4_800.0 / 2_400.0;
    let x = render(&mut engine, &[AudioCommand::Play(play)], &data, 60);
    let rect: Vec<f64> = x[x.len() / 2..].iter().map(|v| v.abs()).collect();
    let mean = rect.iter().sum::<f64>() / rect.len() as f64;
    let env: Vec<f64> = rect.iter().map(|v| v - mean).collect();
    let f = peak_hz(&env, 100.0, 600.0, RATE);
    println!(
        "the V8 grain at 4800 rpm: the envelope's strongest line is {f} Hz (firing rate 320 Hz)"
    );
    assert!((f - 320.0).abs() <= 5.0, "the grain fires at {f} Hz");
}
