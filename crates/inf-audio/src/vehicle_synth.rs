//! **The vehicle generator** (wave VEH3e) — thirty-one `.inf_audio` clips a
//! car is heard through, made of arithmetic at asset-build time (the last
//! three, the rolling road, from the VEH3e audit).
//!
//! [`crate::synth`]'s header is the argument for synthesising at build time and
//! committing the bytes, and every rule of it holds here: the only PCM door is
//! [`SoundData::from_bytes`](crate::SoundData::from_bytes), there is no
//! per-sample callback behind the facade, and the committed bytes must be the
//! same on every machine — so there is **no `sin`, no `exp`, no `powf` and no
//! `sqrt`** below. The oscillator is [`crate::synth::psine`], the noise is
//! [`crate::synth::pnoise`], the decays are per-sample one-pole factors and the
//! filters are [`OnePole`]s.
//!
//! # The combustion grains
//!
//! The research doc's *"sample individual cylinder firing events at different
//! engine loads (0 %, 50 %, 100 % throttle)"*, baked. One clip is
//! [`GRAIN_CYCLES`] whole engine cycles at [`GRAIN_REF_RPM`]: a four-stroke
//! fires every cylinder once per two revolutions, so at the reference the clip
//! holds `cylinders × GRAIN_CYCLES` firing events spaced
//! `1 / ((rpm / 60) · (cylinders / 2))` seconds apart — **the firing period is
//! in the file**, which is what `veh3e_gate` measures off the bytes. The engine
//! plays it pitched by `rpm / GRAIN_REF_RPM`, so a four-cylinder and a V8 at the
//! same revs fire at 1 : 2 by construction and not by a number in a table.
//!
//! Five families: four-, six- and eight-cylinder petrol (the eight in two
//! firing orders — a cross-plane's uneven bank pulses and a flat-plane's even
//! ones) and a six-cylinder diesel with injector clatter on every event. Three
//! loads each: the doc's 0 / 50 / 100 %, which differ in how hard each event
//! is, how bright it is and how much noise rides it.
//!
//! # Licence
//!
//! Moot, as for the gunshots: nothing is recorded, sampled or derived from a
//! recording. A recorded pack swaps in behind the same GUIDs.

use crate::filter::OnePole;
use crate::synth::{decay_per_sample, pnoise, psine, wav_bytes, SYNTH_PEAK, SYNTH_RATE};

/// **The rpm every combustion grain is baked at.** Two thousand four hundred:
/// near the geometric middle of an 800-rpm idle and a 7 000-rpm redline, so the
/// playback rate spans roughly 0.33x to 2.9x and neither end is stretched as far
/// as a clip baked at either extreme would be. It also makes one engine cycle
/// exactly 0.05 s, which is 1 102.5 samples: [`GRAIN_CYCLES`] is even so the
/// loop is a whole number of samples.
pub const GRAIN_REF_RPM: f64 = 2_400.0;

/// **How many whole engine cycles one grain loop holds.** Eight: 0.4 s, long
/// enough that the loop's own repetition is not a buzz in its own right.
pub const GRAIN_CYCLES: usize = 8;

/// **The five grain families, by index** — the order
/// `inf_ecs::vehicle_audio::GrainFamily::ALL` publishes.
pub const FAMILY_NAMES: [&str; 5] = ["P4", "P6", "P8X", "P8F", "D6"];

/// How many cylinders each family's clip fires, in [`FAMILY_NAMES`] order.
pub const FAMILY_CYLINDERS: [u32; 5] = [4, 6, 8, 8, 6];

/// **The three loads, by index** — the doc's 0 %, 50 % and 100 %.
pub const LOAD_NAMES: [&str; 3] = ["Idle", "Mid", "Full"];

/// **Every clip this module generates, by index** — the order
/// `inf_ecs::vehicle_audio::VehicleClip::ALL` publishes. The first fifteen are
/// the grains, family-major.
pub const VEHICLE_CLIP_NAMES: [&str; 31] = [
    "Grain_P4_Idle",
    "Grain_P4_Mid",
    "Grain_P4_Full",
    "Grain_P6_Idle",
    "Grain_P6_Mid",
    "Grain_P6_Full",
    "Grain_P8X_Idle",
    "Grain_P8X_Mid",
    "Grain_P8X_Full",
    "Grain_P8F_Idle",
    "Grain_P8F_Mid",
    "Grain_P8F_Full",
    "Grain_D6_Idle",
    "Grain_D6_Mid",
    "Grain_D6_Full",
    "Whine",
    "Turbo",
    "BlowOff",
    "Squeal_Asphalt",
    "Squeal_Gravel",
    "Squeal_Soft",
    "Impulse_Kerb",
    "Impulse_Gravel",
    "Impulse_Soft",
    "Door_Latch",
    "Door_Creak",
    "Door_Slam",
    "Body_Thud",
    "Roll_Asphalt",
    "Roll_Gravel",
    "Roll_Soft",
];

/// **The gear-whine tone at a pitch of one**, hertz: 22 050 / 18, so one cycle
/// is exactly eighteen samples and the loop closes on itself.
pub const WHINE_HZ: f64 = 1_225.0;

/// **The turbo's whistle at a pitch of one**, hertz: 22 050 / 9.
pub const TURBO_HZ: f64 = 2_450.0;

/// **The asphalt squeal's fundamental**, hertz — a tyre's stick-slip band.
pub const SQUEAL_HZ: f64 = 1_100.0;

/// How long each looping clip is, samples, for the ones that are not grains.
const TONE_LOOP_SAMPLES: usize = 5_400;

/// The per-event amplitude pattern of each family, one engine cycle long.
///
/// A cross-plane V8 fires evenly — every 90 degrees of crank — and burbles
/// because each BANK's exhaust pulses are uneven, which is an amplitude pattern
/// and not a timing one. The flat-plane alternates banks and is even. The small
/// imbalance on the others is what keeps a four from sounding like a buzzer.
const FAMILY_AMPS: [&[f64]; 5] = [
    &[1.00, 0.93, 0.97, 0.90],
    &[1.00, 0.95, 0.98, 0.94, 0.99, 0.93],
    &[1.00, 0.62, 0.88, 0.70, 0.95, 0.60, 0.84, 0.68],
    &[1.00, 0.90, 1.00, 0.90, 1.00, 0.90, 1.00, 0.90],
    &[1.00, 0.96, 0.99, 0.95, 0.98, 0.94],
];

/// One load's pulse shape: the exhaust resonance, its decay as a fraction of
/// the firing interval, and how much filtered noise rides each event.
#[derive(Clone, Copy)]
struct LoadShape {
    resonance_hz: f64,
    decay_frac: f64,
    noise: f64,
    noise_cutoff_hz: f64,
    overtone: f64,
}

const LOAD_SHAPES: [LoadShape; 3] = [
    LoadShape {
        resonance_hz: 170.0,
        decay_frac: 0.30,
        noise: 0.12,
        noise_cutoff_hz: 900.0,
        overtone: 0.10,
    },
    LoadShape {
        resonance_hz: 230.0,
        decay_frac: 0.26,
        noise: 0.30,
        noise_cutoff_hz: 2_200.0,
        overtone: 0.25,
    },
    LoadShape {
        resonance_hz: 300.0,
        decay_frac: 0.22,
        noise: 0.50,
        noise_cutoff_hz: 4_500.0,
        overtone: 0.40,
    },
];

/// **How many samples one grain loop is**: [`GRAIN_CYCLES`] engine cycles at
/// [`GRAIN_REF_RPM`], exactly.
pub fn grain_loop_samples() -> usize {
    // One cycle is two revolutions: 120 / rpm seconds.
    let cycle_s = 120.0 / GRAIN_REF_RPM;
    (cycle_s * GRAIN_CYCLES as f64 * f64::from(SYNTH_RATE)).round() as usize
}

/// **The firing interval a family's grain is baked at**, seconds —
/// `60 / ((rpm / 60) · (cylinders / 2))` at [`GRAIN_REF_RPM`].
pub fn grain_firing_interval_s(family: u8) -> f64 {
    let cyl = f64::from(FAMILY_CYLINDERS[(family as usize).min(4)]);
    1.0 / ((GRAIN_REF_RPM / 60.0) * (cyl / 2.0))
}

/// **One combustion grain loop**, `[-1, 1]`, at [`SYNTH_RATE`].
///
/// Every event is a decaying exhaust resonance with a 0.3 ms attack, a second
/// partial, and low-passed noise under the same envelope; the diesel adds a
/// bright injector click. Events are placed at FRACTIONAL sample positions and
/// wrap around the loop's end, so the loop is seamless and the spacing is
/// exact rather than rounded to a sample.
pub fn grain_pcm(family: u8, load: u8) -> Vec<f64> {
    let (Some(amps), Some(shape)) = (
        FAMILY_AMPS.get(family as usize).copied(),
        LOAD_SHAPES.get(load as usize).copied(),
    ) else {
        return Vec::new();
    };
    let diesel = family == 4;
    let rate = f64::from(SYNTH_RATE);
    let len = grain_loop_samples();
    let interval_s = grain_firing_interval_s(family);
    let events = amps.len() * GRAIN_CYCLES;
    let tau = interval_s * shape.decay_frac;
    let k = decay_per_sample(tau, SYNTH_RATE);
    // Each event rings for five time constants, which is well under one
    // interval at every family's decay fraction.
    let ring = ((tau * 5.0) * rate).ceil() as usize + 2;
    let attack_s = 0.0003;
    let seed = 0x5645_4833_0000_0000u64 ^ ((u64::from(family) << 8) | u64::from(load));
    let mut out = vec![0.0f64; len];
    for e in 0..events {
        let amp = amps[e % amps.len()];
        let at = e as f64 * interval_s * rate;
        let first = at.ceil() as usize;
        // The envelope at the first whole sample: one step of decay, pro rata
        // for the fraction of a sample the event started before it.
        let mut env = 1.0 - (first as f64 - at) * (1.0 - k);
        let mut lp = OnePole::new(shape.noise_cutoff_hz, SYNTH_RATE);
        let mut click = OnePole::new(3_000.0, SYNTH_RATE);
        let knock = decay_per_sample(0.0005, SYNTH_RATE);
        let mut kenv = 1.0;
        for j in 0..ring {
            let n = first + j;
            let t = (n as f64 - at) / rate;
            let att = (t / attack_s).min(1.0);
            let tone = psine(shape.resonance_hz * t)
                + shape.overtone * psine(2.0 * shape.resonance_hz * t + 0.25);
            let noise = lp.step(pnoise(seed ^ e as u64, j as u64));
            let mut v = (tone + shape.noise * 2.0 * noise) * env * att;
            if diesel {
                // The injector knock: high-passed noise under a fast decay,
                // the diesel's clatter on every event.
                let raw = pnoise(seed ^ 0xd1e5 ^ e as u64, j as u64);
                let hi = raw - click.step(raw);
                v += 0.9 * hi * kenv;
                kenv *= knock;
            }
            out[n % len] += amp * v;
            env *= k;
        }
    }
    normalize(out)
}

/// A looping tone of `partials` (frequency, gain) over [`TONE_LOOP_SAMPLES`],
/// plus `hiss` of seamless low-passed noise. Every partial is chosen to close
/// the loop in whole cycles.
fn tone_loop(partials: &[(f64, f64)], hiss: f64, hiss_cutoff: f64, seed: u64) -> Vec<f64> {
    let len = TONE_LOOP_SAMPLES;
    let rate = f64::from(SYNTH_RATE);
    let noise = seamless_noise(len, hiss_cutoff, seed);
    let mut out = Vec::with_capacity(len);
    for (i, nz) in noise.iter().enumerate() {
        let t = i as f64 / rate;
        let mut v = 0.0;
        for (f, g) in partials {
            v += g * psine(f * t);
        }
        out.push(v + hiss * nz);
    }
    normalize(out)
}

/// **Low-passed noise that loops without a seam**: `len + fade` samples of it,
/// with the last `fade` cross-faded over the first, so sample `len - 1` runs
/// into sample `0` exactly as it ran into sample `len` in the filtered stream.
fn seamless_noise(len: usize, cutoff_hz: f64, seed: u64) -> Vec<f64> {
    let fade = len / 8;
    let mut lp = OnePole::new(cutoff_hz, SYNTH_RATE);
    let raw: Vec<f64> = (0..len + fade)
        .map(|i| lp.step(pnoise(seed, i as u64)))
        .collect();
    let mut out = raw[..len].to_vec();
    for (i, o) in out.iter_mut().enumerate().take(fade) {
        let w = i as f64 / fade as f64;
        *o = raw[i] * w + raw[len + i] * (1.0 - w);
    }
    out
}

/// **A tyre on a surface, sliding** — by surface index: `0` sealed (the squeal:
/// three tonal partials under slow amplitude modulation), `1` loose (the hiss:
/// bright noise with a crunch of grains), `2` soft (the scrub: dark noise).
///
/// Half a second, looping; every modulation and partial closes in whole cycles
/// over it (frequencies are multiples of 2 Hz).
pub fn squeal_pcm(surface: u8) -> Vec<f64> {
    let len = (0.5 * f64::from(SYNTH_RATE)).round() as usize;
    let rate = f64::from(SYNTH_RATE);
    let seed = 0x5645_4833_0000_0100u64 + u64::from(surface);
    match surface {
        0 => {
            let noise = seamless_noise(len, 3_000.0, seed);
            let out = (0..len)
                .map(|i| {
                    let t = i as f64 / rate;
                    let am = 0.75 + 0.15 * psine(8.0 * t) + 0.10 * psine(12.0 * t + 0.3);
                    let tone = psine(SQUEAL_HZ * t)
                        + 0.45 * psine(1_650.0 * t + 0.1)
                        + 0.25 * psine(2_200.0 * t + 0.2);
                    tone * am + 0.35 * noise[i]
                })
                .collect();
            normalize(out)
        }
        1 => {
            let hiss = seamless_noise(len, 7_000.0, seed);
            let mut hp = OnePole::new(1_800.0, SYNTH_RATE);
            let mut out: Vec<f64> = hiss
                .iter()
                .map(|x| {
                    let lo = hp.step(*x);
                    x - lo
                })
                .collect();
            // The crunch: a grain every ~11 ms at a deterministic strength,
            // wrapped around the loop so the seam carries grains too.
            let step = (0.011 * rate) as usize;
            let k = decay_per_sample(0.002, SYNTH_RATE);
            let mut g = 0usize;
            while g * step < len {
                let amp = 0.5 + 0.5 * pnoise(seed ^ 0x9, g as u64).abs();
                let start = g * step;
                let mut env = amp;
                for j in 0..(0.012 * rate) as usize {
                    out[(start + j) % len] += env * pnoise(seed ^ 0xa, (g * 1000 + j) as u64);
                    env *= k;
                }
                g += 1;
            }
            normalize(out)
        }
        2 => {
            let noise = seamless_noise(len, 500.0, seed);
            let out = (0..len)
                .map(|i| {
                    let t = i as f64 / rate;
                    noise[i] * (0.7 + 0.3 * psine(6.0 * t))
                })
                .collect();
            normalize(out)
        }
        _ => Vec::new(),
    }
}

/// **A tyre ROLLING on a surface** (VEH3e audit) — by surface index: `0`
/// sealed (the road roar: a dark broadband rush with a tread-block pulse), `1`
/// loose (the crunch: bright noise under a patter of grains), `2` soft (the
/// rumble: very dark noise, slowly breathing).
///
/// Six tenths of a second, looping without a seam: every modulation closes in
/// whole cycles over it (multiples of 5/3 Hz), and the noise is
/// [`seamless_noise`]. The engine voices it by SPEED and picks it by SURFACE —
/// the research doc's "continuous surface roll noise".
pub fn roll_pcm(surface: u8) -> Vec<f64> {
    let rate = f64::from(SYNTH_RATE);
    let len = (0.6 * rate).round() as usize;
    let seed = 0x5645_4833_0000_0400u64 + u64::from(surface);
    match surface {
        0 => {
            let body = seamless_noise(len, 650.0, seed);
            let air = seamless_noise(len, 2_400.0, seed ^ 0x5);
            let out = (0..len)
                .map(|i| {
                    let t = i as f64 / rate;
                    let tread = 0.85 + 0.15 * psine(25.0 * t);
                    (body[i] * 1.6 + air[i] * 0.35) * tread
                })
                .collect();
            normalize(out)
        }
        1 => {
            let rush = seamless_noise(len, 4_500.0, seed);
            let mut out: Vec<f64> = rush.iter().map(|x| 0.45 * x).collect();
            let step = (0.007 * rate) as usize;
            let k = decay_per_sample(0.0015, SYNTH_RATE);
            let mut g = 0usize;
            while g * step < len {
                let amp = 0.3 + 0.7 * pnoise(seed ^ 0x9, g as u64).abs();
                let start =
                    g * step + (pnoise(seed ^ 0xb, g as u64).abs() * step as f64 * 0.5) as usize;
                let mut env = amp;
                for j in 0..(0.006 * rate) as usize {
                    out[(start + j) % len] += env * pnoise(seed ^ 0xa, (g * 1000 + j) as u64);
                    env *= k;
                }
                g += 1;
            }
            normalize(out)
        }
        2 => {
            let noise = seamless_noise(len, 220.0, seed);
            let out = (0..len)
                .map(|i| {
                    let t = i as f64 / rate;
                    noise[i] * (0.75 + 0.25 * psine(5.0 * t))
                })
                .collect();
            normalize(out)
        }
        _ => Vec::new(),
    }
}

/// **A one-shot of `seconds`**: a low decaying tone (fundamental, decay), an
/// attack click, and a noise body under its own cutoff and decay.
#[allow(clippy::too_many_arguments)]
fn thump(
    seconds: f64,
    tone_hz: f64,
    tone_tau: f64,
    second_hz: f64,
    noise_cutoff: f64,
    noise_tau: f64,
    noise_gain: f64,
    seed: u64,
) -> Vec<f64> {
    let rate = f64::from(SYNTH_RATE);
    let n = ((seconds * rate).round() as usize).max(1);
    let kt = decay_per_sample(tone_tau, SYNTH_RATE);
    let kn = decay_per_sample(noise_tau, SYNTH_RATE);
    let kc = decay_per_sample(0.0015, SYNTH_RATE);
    let mut lp = OnePole::new(noise_cutoff, SYNTH_RATE);
    let mut hp = OnePole::new(2_500.0, SYNTH_RATE);
    let (mut et, mut en, mut ec) = (1.0, 1.0, 1.0);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / rate;
        let att = (t / 0.0008).min(1.0);
        let tone = psine(tone_hz * t) + 0.4 * psine(second_hz * t + 0.1);
        let body = lp.step(pnoise(seed, i as u64));
        let raw = pnoise(seed ^ 0xc1, i as u64);
        let click = raw - hp.step(raw);
        out.push((tone * et + noise_gain * 2.5 * body * en) * att + 0.5 * click * ec);
        et *= kt;
        en *= kn;
        ec *= kc;
    }
    normalize(out)
}

/// **A surface impulse** — by surface index: `0` the kerb/seam thump, `1` a
/// gravel hit (a spray of bright grains), `2` a soft thud.
pub fn impulse_pcm(surface: u8) -> Vec<f64> {
    let seed = 0x5645_4833_0000_0200u64 + u64::from(surface);
    match surface {
        0 => thump(0.22, 75.0, 0.045, 160.0, 900.0, 0.02, 0.35, seed),
        1 => {
            let rate = f64::from(SYNTH_RATE);
            let n = (0.14 * rate).round() as usize;
            let mut out = vec![0.0f64; n];
            let k = decay_per_sample(0.004, SYNTH_RATE);
            let mut hp = OnePole::new(2_000.0, SYNTH_RATE);
            let hi: Vec<f64> = (0..n)
                .map(|i| {
                    let x = pnoise(seed, i as u64);
                    x - hp.step(x)
                })
                .collect();
            for g in 0..6usize {
                let start = (g as f64 * 0.017 * rate) as usize;
                let amp = 1.0 - g as f64 * 0.12;
                let mut env = amp;
                for j in start..n {
                    out[j] += hi[j] * env;
                    env *= k;
                }
            }
            normalize(out)
        }
        2 => thump(0.18, 60.0, 0.03, 110.0, 400.0, 0.04, 0.6, seed),
        _ => Vec::new(),
    }
}

/// **The door clips** — `0` the latch (two metallic clicks), `1` the creak (a
/// looping hinge friction), `2` the slam, `3` the bail-out body thud.
pub fn door_pcm(which: u8) -> Vec<f64> {
    let rate = f64::from(SYNTH_RATE);
    let seed = 0x5645_4833_0000_0300u64 + u64::from(which);
    match which {
        0 => {
            let n = (0.09 * rate).round() as usize;
            let k = decay_per_sample(0.006, SYNTH_RATE);
            let mut out = vec![0.0f64; n];
            for (c, start_s) in [0.0f64, 0.028].iter().enumerate() {
                let start = (start_s * rate) as usize;
                let mut env = 1.0 - 0.3 * c as f64;
                for (j, o) in out.iter_mut().enumerate().skip(start) {
                    let t = (j - start) as f64 / rate;
                    let ring = psine(3_200.0 * t) + 0.6 * psine(5_000.0 * t + 0.2);
                    let tick = pnoise(seed, j as u64) * (1.0 - (t / 0.001).min(1.0));
                    *o += (ring * 0.7 + tick) * env;
                    env *= k;
                }
            }
            normalize(out)
        }
        1 => {
            // 0.6 s: every partial and the modulation close in whole cycles
            // (multiples of 5/3 Hz over 0.6 s are whole numbers of cycles).
            let len = (0.6 * rate).round() as usize;
            let noise = seamless_noise(len, 1_500.0, seed);
            let out = (0..len)
                .map(|i| {
                    let t = i as f64 / rate;
                    let am = 0.6 + 0.4 * psine(5.0 * t);
                    let tone = psine(220.0 * t)
                        + 0.5 * psine(440.0 * t + 0.2)
                        + 0.3 * psine(660.0 * t + 0.4)
                        + 0.2 * psine(1_330.0 * t);
                    tone * am + 0.25 * noise[i]
                })
                .collect();
            normalize(out)
        }
        2 => thump(0.40, 65.0, 0.07, 180.0, 3_000.0, 0.025, 0.45, seed),
        3 => thump(0.30, 55.0, 0.06, 95.0, 300.0, 0.06, 0.8, seed),
        _ => Vec::new(),
    }
}

/// **The blow-off valve** — a 0.45 s burst of bright noise with a fast attack
/// and a falling brightness: the "pssh" of boost dumped to atmosphere when the
/// throttle shuts.
pub fn blow_off_pcm() -> Vec<f64> {
    let rate = f64::from(SYNTH_RATE);
    let n = (0.45 * rate).round() as usize;
    let seed = 0x5645_4833_0000_0011u64;
    let k = decay_per_sample(0.11, SYNTH_RATE);
    let mut bright = OnePole::new(7_000.0, SYNTH_RATE);
    let mut dark = OnePole::new(2_000.0, SYNTH_RATE);
    let mut hp = OnePole::new(1_200.0, SYNTH_RATE);
    let mut env = 1.0;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / rate;
        let att = (t / 0.005).min(1.0);
        let x = pnoise(seed, i as u64);
        let hi = x - hp.step(x);
        let w = (t / 0.45).min(1.0);
        let v = bright.step(hi) * (1.0 - w) + dark.step(hi) * w;
        out.push(v * env * att);
        env *= k;
    }
    normalize(out)
}

/// **One vehicle clip's samples, by index** into [`VEHICLE_CLIP_NAMES`]. Out of
/// range answers an empty clip.
pub fn vehicle_clip_pcm(index: u8) -> Vec<f64> {
    match index {
        0..=14 => grain_pcm(index / 3, index % 3),
        15 => tone_loop(
            &[
                (WHINE_HZ, 1.0),
                (2.0 * WHINE_HZ, 0.3),
                (3.0 * WHINE_HZ, 0.1),
            ],
            0.05,
            4_000.0,
            0x5645_4833_0000_000f,
        ),
        16 => tone_loop(
            &[(TURBO_HZ, 1.0), (2.0 * TURBO_HZ, 0.25)],
            0.35,
            6_000.0,
            0x5645_4833_0000_0010,
        ),
        17 => blow_off_pcm(),
        18..=20 => squeal_pcm(index - 18),
        21..=23 => impulse_pcm(index - 21),
        24..=27 => door_pcm(index - 24),
        28..=30 => roll_pcm(index - 28),
        _ => Vec::new(),
    }
}

/// **One vehicle clip's WAV bytes**, ready for
/// [`AudioAsset::from_encoded`](crate::AudioAsset::from_encoded).
pub fn vehicle_clip_wav(index: u8) -> Vec<u8> {
    wav_bytes(&vehicle_clip_pcm(index), SYNTH_RATE)
}

/// Scale a clip so its loudest sample is exactly [`SYNTH_PEAK`].
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

/// **The firing period a grain actually has**, seconds, measured off its
/// samples: the circular autocorrelation of its 400 Hz-smoothed, mean-removed
/// envelope, and the SHORTEST lag between 1 ms and 40 ms that is a local maximum
/// reaching 60 % of the strongest lag in that window, refined by a parabola
/// through its neighbours. Circular because the clip is a LOOP, so the lag that
/// crosses the end is as real as any other.
///
/// Not "count the pulses": a firing event is a ringing exhaust resonance and its
/// second lobe crosses any fixed threshold as readily as its first (measured on
/// this function's first draft, which counted 64 events in a 32-event clip). An
/// autocorrelation asks the question the ear asks — at what spacing does this
/// sound repeat — and the SHORTEST strong repeat is the firing period, where
/// the strongest may be a whole engine cycle (a cross-plane V8's burble repeats
/// exactly every eight events and only nearly every one).
///
/// Public because `veh3e_gate` runs it over the COMMITTED bytes: a period
/// measured by the generator's own test against its own constant would be a
/// number checking itself.
pub fn grain_period_s(pcm: &[f64], rate: u32) -> Option<f64> {
    let n = pcm.len();
    if n < 64 || rate == 0 {
        return None;
    }
    let mut lp = OnePole::new(400.0, rate);
    // Two passes over the loop so the smoother has settled when the kept pass
    // begins: the envelope of a loop is itself a loop.
    for s in pcm {
        lp.step(s.abs());
    }
    let env: Vec<f64> = pcm.iter().map(|s| lp.step(s.abs())).collect();
    let mean = env.iter().sum::<f64>() / n as f64;
    let e: Vec<f64> = env.iter().map(|v| v - mean).collect();
    let fr = f64::from(rate);
    let lo = ((0.001 * fr) as usize).max(2);
    let hi = ((0.040 * fr) as usize).min(n / 2);
    if hi <= lo + 2 {
        return None;
    }
    let r = |lag: usize| -> f64 { (0..n).map(|i| e[i] * e[(i + lag) % n]).sum() };
    let rs: Vec<f64> = (lo - 1..=hi + 1).map(r).collect();
    let at = |lag: usize| rs[lag + 1 - lo];
    let best = (lo..=hi).map(at).fold(f64::MIN, f64::max);
    if best <= 0.0 {
        return None;
    }
    let lag = (lo..=hi).find(|&l| {
        let v = at(l);
        v >= 0.6 * best && v >= at(l - 1) && v >= at(l + 1)
    })?;
    let (a, b, c) = (at(lag - 1), at(lag), at(lag + 1));
    let denom = a - 2.0 * b + c;
    let shift = if denom.abs() > 1e-18 {
        (0.5 * (a - c) / denom).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    Some((lag as f64 + shift) / fr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every clip exists, peaks at exactly [`SYNTH_PEAK`] and decodes through
    /// the only PCM door — the whole contract with the engine.
    #[test]
    fn every_vehicle_clip_is_a_normalised_wav_that_decodes() {
        println!("{:16} {:>7} {:>7}", "clip", "samples", "secs");
        for (i, name) in VEHICLE_CLIP_NAMES.iter().enumerate() {
            let pcm = vehicle_clip_pcm(i as u8);
            assert!(!pcm.is_empty(), "{name} is empty");
            let peak = pcm.iter().fold(0.0f64, |a, s| a.max(s.abs()));
            assert!((peak - SYNTH_PEAK).abs() < 1e-9, "{name} peaks at {peak}");
            println!(
                "{:16} {:7} {:7.4}",
                name,
                pcm.len(),
                pcm.len() as f64 / f64::from(SYNTH_RATE)
            );
            let bytes = vehicle_clip_wav(i as u8);
            assert!(
                crate::SoundData::from_bytes(bytes.clone()).is_ok(),
                "{name}"
            );
            assert_eq!(
                bytes,
                vehicle_clip_wav(i as u8),
                "{name} is not deterministic"
            );
        }
        assert!(vehicle_clip_pcm(31).is_empty());
    }

    /// **THE FIRING PERIOD IS IN THE SAMPLES** — counted, per family and per
    /// load, against `1 / ((rpm / 60) · (cylinders / 2))` s at the reference.
    #[test]
    fn every_grain_fires_at_its_cylinder_counts_rate() {
        assert_eq!(grain_loop_samples(), 8_820);
        for family in 0..5u8 {
            for load in 0..3u8 {
                let pcm = grain_pcm(family, load);
                let period_s = grain_period_s(&pcm, SYNTH_RATE).expect("a period");
                let expect = grain_firing_interval_s(family);
                let events = pcm.len() as f64 / f64::from(SYNTH_RATE) / period_s;
                println!(
                    "{:4} {:5} period {:.5} s (baked {:.5}), {:.2} events in the loop",
                    FAMILY_NAMES[family as usize],
                    LOAD_NAMES[load as usize],
                    period_s,
                    expect,
                    events
                );
                assert!(
                    (period_s - expect).abs() < 0.01 * expect,
                    "{} {}: period {period_s} s against {expect}",
                    FAMILY_NAMES[family as usize],
                    LOAD_NAMES[load as usize]
                );
                let want = (FAMILY_CYLINDERS[family as usize] as usize * GRAIN_CYCLES) as f64;
                assert!((events - want).abs() < 0.5, "{events} events, {want} baked");
            }
        }
    }

    /// The whine and the turbo loops close in whole cycles: the sample after
    /// the last is the first.
    #[test]
    fn the_tone_loops_close_on_themselves() {
        for (f, name) in [(WHINE_HZ, "whine"), (TURBO_HZ, "turbo")] {
            let cycles = f * TONE_LOOP_SAMPLES as f64 / f64::from(SYNTH_RATE);
            assert!((cycles - cycles.round()).abs() < 1e-9, "{name}: {cycles}");
        }
    }
}
