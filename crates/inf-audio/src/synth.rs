//! **The gunshot generator** (wave WPN2c) — thirty-six `.inf_audio` clips, made
//! of arithmetic.
//!
//! # Why this exists at all
//!
//! [`SoundData::from_bytes`](crate::SoundData::from_bytes) is the **only** PCM
//! door this facade has: WAV/OGG/FLAC/MP3 in, decoded samples out. There is no
//! streaming source and no per-sample synthesizer callback, and adding one means
//! implementing kira's `Sound` trait behind a facade whose first rule is that it
//! never re-exports a kira type. So synthesis happens at **asset-build time**:
//! this module writes RIFF bytes, [`crate::AudioAsset::from_encoded`] decodes
//! them once to record duration, and the result is committed under a fixed GUID.
//!
//! That is not a workaround. Every committed sound in this repository is already
//! generated — `samples/phase30-gameplay/Report.inf_audio` has been eight
//! hundred samples of a linear ramp since wave WPN1 — and the difference here is
//! only that the arithmetic is trying to sound like something.
//!
//! # Licence
//!
//! Moot. Nothing here is recorded, sampled or derived from a recording; every
//! byte is the output of the functions below. A **real** recorded pack is a
//! decision for the user rather than for this wave — the door takes any WAV, so
//! swapping one in is `AudioAsset::from_encoded(std::fs::read(…)?, Wav)` with
//! the same GUIDs.
//!
//! # Everything here is bit-exact on every platform
//!
//! These bytes are **committed**, so a generator whose output depended on the
//! host's libm would produce a different set of files on Windows and on the CI
//! runner, and the byte gate would red on one of them. There is therefore
//! **no `sin`, no `exp`, no `powf` and no `sqrt`** below:
//!
//! * the oscillator is the parabolic sine `4x(1 - |x|)`, four arithmetic ops;
//! * the envelope is a one-pole decay, `env *= 1 - 1/(rate·tau)`, which is one
//!   division computed once and one multiply per sample;
//! * the noise is counter-based splitmix64 into a 53-bit double, which is the
//!   same construction `inf_ecs::weapon::shot_uniforms` uses and is exact;
//! * the filters are one-poles, add and multiply.
//!
//! The prescriptions are therefore **measured rather than asserted**: a
//! one-pole's own decay is not its stated time constant, so
//! [`clip_pcm`]'s tests read the −60 dB point off the samples and print it.

use crate::filter::OnePole;

/// **The sample rate every generated clip is written at**, hertz.
///
/// Twenty-two thousand and fifty — half of CD rate, so the Nyquist limit is
/// 11 025 Hz. That is above every band these clips carry (the brightest thing
/// here is a 5 kHz-shaped noise tail) and it halves what thirty-six committed
/// files cost. The engine's own pre-WPN2c clips are 8 kHz, which is below the
/// transient's own band; this is the first rate in the tree chosen against what
/// is in the file.
pub const SYNTH_RATE: u32 = 22_050;

/// **How loud a generated clip peaks**, as a fraction of full scale.
///
/// Nine tenths. The headroom is left for the mixer, which is
/// `inf_ecs::weapon::REPORT_VOLUME`'s own argument one layer down: this is the
/// loudest transient the engine makes and it is heard at the muzzle.
pub const SYNTH_PEAK: f64 = 0.9;

/// **The seven classes, by index** — the order
/// `inf_ecs::weapon::WeaponClass::ALL` publishes.
///
/// This crate cannot name `WeaponClass` (it is Ring 0 beneath `inf-ecs`), so the
/// contract between them is the INDEX, and this list is what makes the contract
/// legible. `wpn2c_gate` asserts the two agree name for name, so they cannot
/// drift silently.
pub const CLASS_NAMES: [&str; 7] = [
    "pistol", "smg", "ar", "dmr", "sniper", "shotgun", "launcher",
];

/// **The five layer clips, by index** — the order
/// `inf_ecs::weapon::ReportClip::ALL` publishes.
pub const CLIP_NAMES: [&str; 5] = ["Transient", "Body", "IndoorTail", "OutdoorTail", "Crack"];

/// **What one class's gunshot is made of** — seven rows, one per class.
///
/// Only two numbers vary with the class and both are what a person actually
/// hears: the body's **fundamental**, which is the pitch of the bang, and its
/// **length**, which is how long the bang rings. A 9 mm pistol cracks at 78 Hz
/// for a fifth of a second and an RPG thumps at 40 Hz for half a second, and
/// every one of the seven sits inside the research doc section 4's own
/// 40-80 Hz band for the body layer.
///
/// The transient, the two tails and the crack are the same shape for every
/// class, with the transient's own length the third column: a bolt is a bolt.
/// A per-class action noise is content, not engineering, and a real recorded
/// pack is where it would come from.
const CLASS_ROWS: [ClassRow; 7] = [
    // fundamental, body seconds, transient seconds
    ClassRow::new(78.0, 0.22, 0.0040),
    ClassRow::new(72.0, 0.20, 0.0040),
    ClassRow::new(58.0, 0.30, 0.0050),
    ClassRow::new(52.0, 0.34, 0.0050),
    ClassRow::new(44.0, 0.42, 0.0050),
    ClassRow::new(48.0, 0.36, 0.0050),
    ClassRow::new(40.0, 0.55, 0.0050),
];

/// One class's two numbers. See [`CLASS_ROWS`].
#[derive(Clone, Copy, Debug)]
struct ClassRow {
    fundamental_hz: f64,
    body_s: f64,
    transient_s: f64,
}

impl ClassRow {
    const fn new(fundamental_hz: f64, body_s: f64, transient_s: f64) -> Self {
        Self {
            fundamental_hz,
            body_s,
            transient_s,
        }
    }
}

/// **How long an indoor tail is**, seconds — the brief's own 0.3 s.
pub const INDOOR_TAIL_S: f64 = 0.30;

/// **How long an outdoor tail is**, seconds — the brief's own 1.5 s.
pub const OUTDOOR_TAIL_S: f64 = 1.50;

/// **How long the N-wave is**, seconds — the brief's own 2 ms.
///
/// A sonic boom's N-wave really is about this: a step up, a linear fall through
/// zero, a step back. Two milliseconds is forty-four samples at [`SYNTH_RATE`],
/// which is why it is written as a shape rather than as a filtered noise.
pub const CRACK_S: f64 = 0.002;

/// **How long the casing's one-shot is**, seconds.
pub const CASING_S: f64 = 0.12;

/// **The cutoff shaping an INDOOR tail's noise**, hertz — bright, because the
/// doc calls a concrete room's reverb *"tight, metallic"*.
pub const INDOOR_TAIL_CUTOFF_HZ: f64 = 5_000.0;

/// **The cutoff shaping an OUTDOOR tail's noise**, hertz — dark, because the
/// doc calls the outdoor answer a *"long mountain-echo"* and distance is a
/// low-pass.
pub const OUTDOOR_TAIL_CUTOFF_HZ: f64 = 1_800.0;

/// **The cutoff a transient is high-passed at**, hertz. A bolt is all top end;
/// without this it is a click with a thump in it and the body layer's job is
/// taken twice.
pub const TRANSIENT_HIGHPASS_HZ: f64 = 1_500.0;

/// **The two partials a brass case rings at**, hertz.
///
/// A 9 mm case is about 19 mm long and rings in the low kilohertz; two partials
/// a fifth apart is what makes it read as metal rather than as a tap.
pub const CASING_PARTIALS_HZ: [f64; 2] = [2_800.0, 4_300.0];

// ── the arithmetic ──────────────────────────────────────────────────────────

/// **The parabolic sine**, `4x(1 − |x|)` for `x` in `[-1, 1)` — the oscillator
/// every tonal thing here is built from.
///
/// Four arithmetic operations, no libm, bit-exact everywhere, and about 3 % of
/// total harmonic distortion against a real sine — which for a gunshot's body
/// is not a defect, since a real one is not a sine either. Its peak is exactly
/// `1.0` at `x = ±0.5`, so the envelope below is the whole of the amplitude.
///
/// `phase` is turns: `0.0` and `1.0` are the same point.
pub fn psine(phase: f64) -> f64 {
    let p = phase - phase.floor();
    // One minus two p, not two p minus one: the second is the same curve upside
    // down, which is a phase inversion and would have made this function a
    // **negative** sine. Measured on this module's own first draft, where
    // `psine(0.25)` read -1 and the arm that asks for a sine-shaped thing
    // caught it.
    let x = 1.0 - 2.0 * p;
    4.0 * x * (1.0 - x.abs())
}

/// **Deterministic noise** in `[-1, 1)` from a counter — splitmix64 into a
/// 53-bit double, the construction `inf_ecs::weapon::shot_uniforms` uses.
///
/// Counter-based rather than stateful, so a clip regenerates byte for byte on
/// any machine and a reviewer can diff it.
pub fn pnoise(seed: u64, i: u64) -> f64 {
    let mut z = seed
        .wrapping_add(i.wrapping_mul(0x2545_f491_4f6c_dd1d))
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    let unit = ((z >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0);
    2.0 * unit - 1.0
}

/// **The per-sample factor of a one-pole decay** with time constant `tau_s`.
///
/// `1 − 1/(rate·tau)`, which is one division and no `exp`. The envelope it
/// produces is not exactly `e^{-t/tau}` — it is the discrete one-pole whose
/// continuous limit that is — so the decay a clip actually has is **measured**
/// rather than claimed; see this module's own tests, which print the −60 dB
/// point of every generated clip.
///
/// Clamped into `[0, 1)`: a factor at or above one would never decay and a
/// negative one would ring at Nyquist.
pub fn decay_per_sample(tau_s: f64, rate: u32) -> f64 {
    if tau_s <= 0.0 || rate == 0 {
        return 0.0;
    }
    (1.0 - 1.0 / (f64::from(rate) * tau_s)).clamp(0.0, 0.999_999_999)
}

// ── the clips ───────────────────────────────────────────────────────────────

/// **One generated clip's samples**, `[-1, 1]`, at [`SYNTH_RATE`].
///
/// `class` is `inf_ecs::weapon::WeaponClass::index()` and `clip` is
/// `inf_ecs::weapon::ReportClip::index()`; both out of range answer an empty
/// clip rather than panicking, because a generator that panicked on a bad index
/// would take an editor down for a number a test can print.
pub fn clip_pcm(class: u8, clip: u8) -> Vec<f64> {
    let Some(row) = CLASS_ROWS.get(class as usize).copied() else {
        return Vec::new();
    };
    // The seed makes two classes' noise different without making it random.
    let seed = 0x5750_4e32_0000_0000u64 ^ ((u64::from(class) << 8) | u64::from(clip));
    match clip {
        0 => transient_pcm(row.transient_s, seed),
        1 => body_pcm(row.fundamental_hz, row.body_s, seed),
        2 => tail_pcm(INDOOR_TAIL_S, INDOOR_TAIL_CUTOFF_HZ, seed),
        3 => tail_pcm(OUTDOOR_TAIL_S, OUTDOOR_TAIL_CUTOFF_HZ, seed),
        4 => crack_pcm(),
        _ => Vec::new(),
    }
}

/// **Layer 1 — the mechanical snap.** A high-passed noise burst with a decay
/// measured in tenths of a millisecond: bolt, hammer, extractor.
fn transient_pcm(seconds: f64, seed: u64) -> Vec<f64> {
    let n = samples_for(seconds);
    let k = decay_per_sample(seconds / 6.0, SYNTH_RATE);
    let mut hp = OnePole::new(TRANSIENT_HIGHPASS_HZ, SYNTH_RATE);
    let mut env = 1.0;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let raw = pnoise(seed, i as u64);
        // High-pass = the signal minus its own low-passed self.
        let hi = raw - hp.step(raw);
        out.push(hi * env);
        env *= k;
    }
    normalize(out)
}

/// **Layer 2 — the explosive body.** The class's fundamental on the parabolic
/// oscillator, with a short noise transient folded in so the attack is not a
/// pure tone, under a decay that gives the clip its stated length.
fn body_pcm(fundamental_hz: f64, seconds: f64, seed: u64) -> Vec<f64> {
    let n = samples_for(seconds);
    let k = decay_per_sample(seconds / 6.9, SYNTH_RATE);
    let kn = decay_per_sample(seconds / 60.0, SYNTH_RATE);
    let step = fundamental_hz / f64::from(SYNTH_RATE);
    let mut phase = 0.0f64;
    let mut env = 1.0;
    let mut noise_env = 1.0;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let tone = psine(phase);
        let n_i = pnoise(seed, i as u64) * 0.35 * noise_env;
        out.push(tone * env + n_i);
        phase += step;
        env *= k;
        noise_env *= kn;
    }
    normalize(out)
}

/// **Layer 3 — the room.** Low-passed noise under a decay. The two rows differ
/// only in how long they ring and how bright they are, which is the whole of
/// what "inside" and "outside" mean here.
fn tail_pcm(seconds: f64, cutoff_hz: f64, seed: u64) -> Vec<f64> {
    let n = samples_for(seconds);
    let k = decay_per_sample(seconds / 6.9, SYNTH_RATE);
    let mut lp = OnePole::new(cutoff_hz, SYNTH_RATE);
    let mut env = 1.0;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(lp.step(pnoise(seed, i as u64)) * env);
        env *= k;
    }
    normalize(out)
}

/// **Layer 4 — the N-wave.** A step up, a straight line down through zero, a
/// step back: the pressure signature of a body going past faster than the air
/// can get out of the way, which is exactly what a supersonic crack is.
fn crack_pcm() -> Vec<f64> {
    let n = samples_for(CRACK_S).max(2);
    normalize(
        (0..n)
            .map(|i| 1.0 - 2.0 * (i as f64) / ((n - 1) as f64))
            .collect(),
    )
}

/// **The casing's one-shot** — two decaying partials a fifth apart plus a short
/// contact click, which is what a brass case landing on concrete is.
///
/// One clip for every calibre: the pitch hash on the `Play` is what makes two
/// of them different (`inf_ecs::casing`).
pub fn casing_pcm() -> Vec<f64> {
    let n = samples_for(CASING_S);
    let k = decay_per_sample(CASING_S / 6.9, SYNTH_RATE);
    let kc = decay_per_sample(CASING_S / 90.0, SYNTH_RATE);
    let mut phases = [0.0f64; 2];
    let steps = [
        CASING_PARTIALS_HZ[0] / f64::from(SYNTH_RATE),
        CASING_PARTIALS_HZ[1] / f64::from(SYNTH_RATE),
    ];
    let mut env = 1.0;
    let mut click = 1.0;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let tone = psine(phases[0]) + 0.6 * psine(phases[1]);
        let c = pnoise(0x5750_4e32_0000_00f0, i as u64) * click;
        out.push(tone * env * 0.55 + c * 0.45);
        phases[0] += steps[0];
        phases[1] += steps[1];
        env *= k;
        click *= kc;
    }
    normalize(out)
}

/// How many samples `seconds` is at [`SYNTH_RATE`], at least one.
fn samples_for(seconds: f64) -> usize {
    ((seconds * f64::from(SYNTH_RATE)).round() as usize).max(1)
}

/// Scale a clip so its loudest sample is exactly [`SYNTH_PEAK`].
///
/// The layers are mixed by `inf_ecs::weapon::report_layers`' own volumes, so a
/// clip's job is to have a shape and not a level — and a normalized clip is one
/// whose quantisation to 16 bits wastes no bits.
fn normalize(mut pcm: Vec<f64>) -> Vec<f64> {
    let peak = pcm.iter().fold(0.0f64, |a, s| a.max(s.abs()));
    if peak > 0.0 {
        let g = SYNTH_PEAK / peak;
        for s in &mut pcm {
            *s *= g;
        }
    }
    pcm
}

// ── the container ───────────────────────────────────────────────────────────

/// **PCM to RIFF** — 16-bit mono WAV bytes, the one container writer on this
/// crate's public surface.
///
/// The tree has carried this header three times (an `asset.rs` test helper, an
/// editor sample generator, and now this), which was defensible while every
/// committed sound was one ramp. Thirty-six clips is where it stops being
/// defensible, so this is the door and the two older copies are test fixtures
/// that produce their own bytes and are left alone deliberately: pointing them
/// here would re-bless two committed clips for a refactor.
pub fn wav_bytes(pcm: &[f64], rate: u32) -> Vec<u8> {
    let channels = 1u16;
    let bits = 16u16;
    let block_align = channels * bits / 8;
    let byte_rate = rate * u32::from(block_align);
    let data_len = (pcm.len() as u32) * u32::from(block_align);
    let mut w = Vec::with_capacity(44 + data_len as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_len).to_le_bytes());
    w.extend_from_slice(b"WAVE");
    w.extend_from_slice(b"fmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&channels.to_le_bytes());
    w.extend_from_slice(&rate.to_le_bytes());
    w.extend_from_slice(&byte_rate.to_le_bytes());
    w.extend_from_slice(&block_align.to_le_bytes());
    w.extend_from_slice(&bits.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        // Round-half-away-from-zero into i16, clamped: the same number on every
        // target, which `f64 as i16`'s saturating cast alone would not promise
        // for a value out of range.
        let v = (s * 32_767.0).round().clamp(-32_768.0, 32_767.0) as i16;
        w.extend_from_slice(&v.to_le_bytes());
    }
    w
}

/// **One report layer's WAV bytes**, ready for
/// [`AudioAsset::from_encoded`](crate::AudioAsset::from_encoded).
pub fn report_wav(class: u8, clip: u8) -> Vec<u8> {
    wav_bytes(&clip_pcm(class, clip), SYNTH_RATE)
}

/// **The casing one-shot's WAV bytes.**
pub fn casing_wav() -> Vec<u8> {
    wav_bytes(&casing_pcm(), SYNTH_RATE)
}

/// **How long a blast rings**, seconds (wave WPN2d).
///
/// One and eight tenths -- longer than the outdoor gunshot tail (1.5 s),
/// because an explosion is the one thing in this engine that is louder than a
/// rifle and the ear reads "bigger" as "rings longer" before it reads it as
/// "starts louder".
pub const BLAST_S: f64 = 1.80;

/// **The blast's fundamental**, hertz. Thirty-five: an octave below the
/// launcher class's own body row, which is as low as a 22 050 Hz clip can carry
/// four cycles of before its own decay eats it.
pub const BLAST_FUNDAMENTAL_HZ: f64 = 35.0;

/// **How long a melee impact lasts**, seconds (wave WPN2d). Ninety
/// milliseconds, which is a contact rather than a ring.
pub const MELEE_IMPACT_S: f64 = 0.09;

/// **The two melee surfaces' cutoffs**, hertz -- flesh, then everything else.
///
/// A body is a dull thud (a 900 Hz low-pass over a short noise burst) and a
/// wall is a sharp clack (a 6 kHz one over the same burst), which is the whole
/// of the two-way surface stub: one generator, one number, two clips.
pub const MELEE_CUTOFF_HZ: [f64; 2] = [900.0, 6_000.0];

/// **The blast's PCM** -- a very low body under a long decay with a broadband
/// noise front, which is what an explosion is: a pressure step and then a room
/// full of it.
pub fn blast_pcm() -> Vec<f64> {
    let n = samples_for(BLAST_S);
    let k = decay_per_sample(BLAST_S / 5.5, SYNTH_RATE);
    // The noise front dies in a tenth of the body's time: a bang, then a boom.
    let kn = decay_per_sample(BLAST_S / 55.0, SYNTH_RATE);
    let step = BLAST_FUNDAMENTAL_HZ / f64::from(SYNTH_RATE);
    let mut lp = OnePole::new(2_400.0, SYNTH_RATE);
    let mut phase = 0.0f64;
    let mut env = 1.0;
    let mut front = 1.0;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let tone = psine(phase);
        let noise = lp.step(pnoise(0x5750_4e32_0000_00f1, i as u64)) * front;
        out.push(tone * env * 0.7 + noise * 0.6);
        phase += step;
        env *= k;
        front *= kn;
    }
    normalize(out)
}

/// **A melee impact's PCM**, by surface index (`0` flesh, `1` everything else).
///
/// One generator and one number, deliberately: the difference between hitting a
/// person and hitting a wall is how much high end survives the contact, and a
/// second hand-written clip would be a second thing to keep in step. VEH3a's
/// per-surface table extends this by adding rows, not by adding generators.
pub fn melee_impact_pcm(surface: u8) -> Vec<f64> {
    let cutoff = MELEE_CUTOFF_HZ[(surface as usize).min(MELEE_CUTOFF_HZ.len() - 1)];
    let n = samples_for(MELEE_IMPACT_S);
    let k = decay_per_sample(MELEE_IMPACT_S / 7.0, SYNTH_RATE);
    let mut lp = OnePole::new(cutoff, SYNTH_RATE);
    let mut env = 1.0;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(lp.step(pnoise(0x5750_4e32_0000_00f2 + u64::from(surface), i as u64)) * env);
        env *= k;
    }
    normalize(out)
}

/// **The blast's WAV bytes.**
pub fn blast_wav() -> Vec<u8> {
    wav_bytes(&blast_pcm(), SYNTH_RATE)
}

/// **A melee impact's WAV bytes**, by surface index.
pub fn melee_impact_wav(surface: u8) -> Vec<u8> {
    wav_bytes(&melee_impact_pcm(surface), SYNTH_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The oscillator peaks at one, crosses zero where a sine does, and is
    /// periodic — which is the whole of what the body layer asks of it.
    #[test]
    fn the_parabolic_sine_is_a_sine_shaped_thing_made_of_four_operations() {
        assert!((psine(0.0)).abs() < 1e-12);
        assert!((psine(0.25) - 1.0).abs() < 1e-12);
        assert!((psine(0.5)).abs() < 1e-12);
        assert!((psine(0.75) + 1.0).abs() < 1e-12);
        // Periodic, including for a phase accumulator that has run for a while.
        assert!((psine(0.3) - psine(7.3)).abs() < 1e-9);
        // Bounded, so `normalize` is a scale and never a clip.
        for i in 0..1000 {
            assert!(psine(i as f64 * 0.0017).abs() <= 1.0 + 1e-12);
        }
    }

    /// The noise is uniform-ish, zero-mean-ish and, above all, the same every
    /// time — which is what makes thirty-six committed files diffable.
    #[test]
    fn the_noise_is_deterministic_and_centred() {
        assert_eq!(pnoise(7, 11), pnoise(7, 11));
        assert_ne!(pnoise(7, 11), pnoise(7, 12));
        let n = 20_000;
        let mean: f64 = (0..n).map(|i| pnoise(3, i)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.02, "mean {mean}");
        for i in 0..n {
            assert!(pnoise(3, i).abs() <= 1.0);
        }
    }

    /// **THE ENVELOPE TABLE** — every prescription in the brief, measured off
    /// the samples rather than asserted about the parameters.
    #[test]
    fn every_generated_clip_has_the_length_and_the_decay_the_brief_asked_for() {
        // The −60 dB point of a clip, in seconds, or its whole length if it
        // never gets there.
        let minus_60 = |pcm: &[f64]| -> f64 {
            let peak = pcm.iter().fold(0.0f64, |a, s| a.max(s.abs()));
            let floor = peak * 0.001;
            // Walk back from the end: the last sample above the floor.
            let last = pcm.iter().rposition(|s| s.abs() > floor).unwrap_or(0);
            (last as f64) / f64::from(SYNTH_RATE)
        };
        println!(
            "{:7} {:12} {:>8} {:>7} {:>8} {:>6}",
            "class", "clip", "samples", "secs", "-60 dB", "peak"
        );
        for c in 0..7u8 {
            for k in 0..5u8 {
                let pcm = clip_pcm(c, k);
                assert!(!pcm.is_empty(), "class {c} clip {k} is empty");
                let secs = pcm.len() as f64 / f64::from(SYNTH_RATE);
                let peak = pcm.iter().fold(0.0f64, |a, s| a.max(s.abs()));
                println!(
                    "{:7} {:12} {:8} {:7.4} {:8.4} {:6.3}",
                    CLASS_NAMES[c as usize],
                    CLIP_NAMES[k as usize],
                    pcm.len(),
                    secs,
                    minus_60(&pcm),
                    peak
                );
                assert!(
                    (peak - SYNTH_PEAK).abs() < 1e-9,
                    "class {c} clip {k} peaks at {peak}"
                );
                match k {
                    // The transient is at most five milliseconds — the brief's
                    // own bound, and the thing that keeps it a snap.
                    0 => assert!(secs <= 0.005 + 1e-9, "transient is {secs} s"),
                    1 => assert!((0.20..=0.56).contains(&secs), "body is {secs} s"),
                    2 => assert!((secs - INDOOR_TAIL_S).abs() < 1e-4),
                    3 => assert!((secs - OUTDOOR_TAIL_S).abs() < 1e-4),
                    4 => assert!((secs - CRACK_S).abs() < 1e-4),
                    _ => unreachable!(),
                }
            }
        }
        // …and the outdoor tail really does ring five times as long as the
        // indoor one, which is the pair the enclosure probe chooses between.
        let inside = clip_pcm(2, 2);
        let outside = clip_pcm(2, 3);
        assert!(outside.len() > inside.len() * 4);
    }

    /// **THE BODY'S BAND** — a zero-crossing count is a period measurement, and
    /// the doc asks for 40-80 Hz.
    #[test]
    fn every_bodys_fundamental_is_inside_the_docs_forty_to_eighty_hertz() {
        println!("class     fundamental   measured");
        for c in 0..7u8 {
            let pcm = clip_pcm(c, 1);
            // Count zero crossings over the MIDDLE third. The attack's own
            // noise burst adds crossings of its own, and counting from sample
            // zero read a pistol at 95.5 Hz against an authored 78 — measured
            // on this test's first draft. The noise envelope is a sixtieth of
            // the clip, so by a third of the way in there is nothing left of it.
            let (lo, hi) = (pcm.len() / 3, 2 * pcm.len() / 3);
            let mut crossings = 0usize;
            for i in lo + 1..hi {
                if (pcm[i - 1] < 0.0) != (pcm[i] < 0.0) {
                    crossings += 1;
                }
            }
            let hz = (crossings as f64 / 2.0) / ((hi - lo) as f64 / f64::from(SYNTH_RATE));
            println!(
                "{:9} {:11.1} {:10.1}",
                CLASS_NAMES[c as usize], CLASS_ROWS[c as usize].fundamental_hz, hz
            );
            assert!(
                (40.0..=80.0).contains(&hz),
                "{} body reads {hz:.1} Hz",
                CLASS_NAMES[c as usize]
            );
            assert!(
                (hz - CLASS_ROWS[c as usize].fundamental_hz).abs() < 6.0,
                "{} is authored at {} Hz and measures {hz:.1}",
                CLASS_NAMES[c as usize],
                CLASS_ROWS[c as usize].fundamental_hz
            );
        }
    }

    /// The crack is an N-wave: it starts positive, ends negative, and crosses
    /// zero exactly once.
    #[test]
    fn the_crack_is_an_n_wave() {
        let pcm = clip_pcm(2, 4);
        assert!(pcm[0] > 0.8 * SYNTH_PEAK);
        assert!(pcm[pcm.len() - 1] < -0.8 * SYNTH_PEAK);
        let crossings = (1..pcm.len())
            .filter(|&i| (pcm[i - 1] < 0.0) != (pcm[i] < 0.0))
            .count();
        assert_eq!(crossings, 1, "an N-wave crosses zero once");
    }

    /// The container really is a WAV, and kira really decodes it — the whole
    /// point of the module.
    #[test]
    fn a_generated_clip_round_trips_through_the_only_pcm_door() {
        let bytes = report_wav(2, 1);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        let sound = crate::SoundData::from_bytes(bytes.clone()).expect("it decodes");
        assert_eq!(sound.sample_rate(), SYNTH_RATE);
        let want = clip_pcm(2, 1).len() as f64 / f64::from(SYNTH_RATE);
        assert!(
            (sound.duration_secs() - want).abs() < 0.005,
            "{} vs {want}",
            sound.duration_secs()
        );
        // Deterministic: the same bytes twice.
        assert_eq!(bytes, report_wav(2, 1));
        // …and the casing's clip too.
        assert!(crate::SoundData::from_bytes(casing_wav()).is_ok());
    }

    /// An out-of-range index is a value, not a panic.
    #[test]
    fn a_class_this_generator_does_not_have_is_an_empty_clip() {
        assert!(clip_pcm(9, 0).is_empty());
        assert!(clip_pcm(0, 9).is_empty());
        assert_eq!(CLASS_NAMES.len(), CLASS_ROWS.len());
    }
}
