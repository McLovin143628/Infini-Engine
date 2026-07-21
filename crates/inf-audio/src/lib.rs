//! Infinity Engine audio facade (Ring 0).
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
//! # Spatial scope
//!
//! [`spatial`] provides the **basics**: a listener pose, per-emitter position, and
//! distance attenuation (linear / inverse) feeding kira volume + panning. Real
//! HRTF, occlusion, reverb, and a full effect graph are **P12.3**.

mod backend;
mod engine;
mod sound;
pub mod spatial;

pub use engine::{AudioEngine, Bus, PlaySettings, SoundHandle};
pub use sound::{DecodeError, SoundData};
pub use spatial::{Attenuation, AttenuationModel, Listener};
