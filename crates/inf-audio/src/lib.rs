//! Infini Engine audio facade (Ring 0).
//!
//! This crate wraps [kira](https://docs.rs/kira) behind a small, engine-flavoured
//! API so the rest of the engine never names kira types directly. The public
//! vocabulary is [`AudioEngine`], [`SoundData`], [`SoundHandle`], the [`Bus`]
//! model, and the [`spatial`] listener/attenuation types — all plain data and
//! `glam::DVec3`.
//!
//! # Why kira
//!
//! kira is game-oriented (sample playback with clocks, tweens, and a mixer),
//! where `rodio` is playback-only (§7 tech matrix). We use its decoders and its
//! audio-thread mixer; the engine computes the **mix policy** (buses + spatial
//! attenuation) itself and feeds kira per-voice volume/panning, so the kira
//! surface we depend on stays tiny.
//!
//! # No-device fallback (a first-class path, not an error case)
//!
//! kira's real device backend (cpal) is pulled in **only** behind this crate's
//! non-default `cpal` feature — the default build has no device dependency (which
//! keeps CI free of ALSA/dbus system libraries; see the crate manifest). So by
//! default, and on any headless machine even with `cpal` on, [`AudioEngine`] runs
//! a **graceful no-device fallback**: every method is a consistent no-op for
//! audible output while all engine-side state — buses, listener, per-voice volume/
//! pitch/position, handle lifecycle — stays fully live and inspectable. The
//! desktop runtime enables `cpal` for real output; enabling it never breaks a
//! headless run because a missing device lands on the very same fallback path.
//! [`AudioEngine::disabled`] forces that path for tests/tools.
//!
//! # Ring-0 threading note
//!
//! When a device is active, kira owns its own real-time audio thread. That is
//! acceptable Ring-0 IO — the same category as a GPU queue — and it is the *only*
//! thread this crate is party to: the facade spawns none of its own, reads no wall
//! clock, and its mix math is pure. Determinism-sensitive callers that must not
//! depend on the audio thread simply run the no-device fallback.
//!
//! # Spatial scope (P12.3 depth)
//!
//! [`spatial`] provides a listener pose, per-emitter position, and distance
//! attenuation (linear / inverse / **exponential**, with min/max clamps) feeding
//! kira volume + panning, plus an **occlusion** multiplier the caller supplies
//! ([`AudioEngine::set_occlusion_hook`] / [`AudioEngine::set_occlusion`]). The
//! named-bus [`mixer`] adds a hierarchical [`MixerConfig`]
//! with per-bus volume + a v1 effect chain (compute-side `Gain`, device-side
//! `Lowpass`), persisted at `.infinity/mixer.toml`. The sim drives playback
//! through the deterministic [`command`] queue, drained host-side by
//! [`AudioEngine::drain`]. Real HRTF, reverb/sends, doppler, and audible per-bus
//! DSP wiring are the documented follow-ups.
//!
//! # Asset
//!
//! [`asset`] is the `.inf_audio` [`AudioAsset`] payload:
//! original compressed bytes + format tag + duration metadata, decoded on load
//! (see its module docs for why not PCM).

mod backend;
pub mod command;
mod engine;
pub mod mixer;
mod sound;
pub mod spatial;

pub mod asset;

pub use asset::{AudioAsset, AudioFormat, AudioImportSettings, BusChoice};
pub use command::{AudioCommand, AudioCommandQueue, PlayCommand};
pub use engine::{AudioEngine, Bus, BusRef, OcclusionHook, PlaySettings, SoundHandle};
pub use mixer::{Effect, MixerConfig, ResolvedBus};
pub use sound::{DecodeError, SoundData};
pub use spatial::{Attenuation, AttenuationModel, Listener};
