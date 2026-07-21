//! The kira boundary. Everything that names a kira manager or handle lives here;
//! the rest of the crate deals only in plain gains, pannings, and rates.
//!
//! # No-device fallback (the Ring-0 / CI contract)
//!
//! The real audio device is opened by kira's cpal backend, which this crate pulls
//! in **only** behind its non-default `cpal` feature (see the crate manifest for
//! the Linux-ALSA/CI rationale). Two things can therefore leave the engine with no
//! backend, and both are handled identically — every method below becomes a
//! consistent no-op:
//!
//! * the crate is built **without** the `cpal` feature (the default, and what CI +
//!   the unit tests exercise); or
//! * the `cpal` feature is on but [`Backend::new`] fails to open a device (a
//!   headless machine / CI runner / no sound card).
//!
//! In both cases [`AudioEngine`](crate::AudioEngine) keeps its full engine-side
//! state (buses, listener, per-sound bookkeeping) so its API stays behaviourally
//! consistent; only the actual sample playback is skipped. kira owns its own audio
//! thread when active — that is acceptable Ring-0 IO, the same category as a GPU
//! queue, and it is documented as such at the crate root.

use crate::sound::SoundData;

/// The engine's opaque per-sound id, shared with [`crate::AudioEngine`].
pub(crate) type VoiceId = u64;

/// Owns the kira manager (when active) and the live per-sound kira handles.
pub(crate) struct Backend {
    #[cfg(feature = "cpal")]
    inner: Option<cpal_impl::Inner>,
}

impl Backend {
    /// Try to bring up a real audio device. Falls back to the disabled (no-op)
    /// state when the `cpal` feature is off or no device can be opened.
    pub(crate) fn new() -> Self {
        #[cfg(feature = "cpal")]
        {
            Self {
                inner: cpal_impl::Inner::try_new(),
            }
        }
        #[cfg(not(feature = "cpal"))]
        {
            Self {}
        }
    }

    /// A backend forced into the disabled state, regardless of build features —
    /// the deterministic target the no-device fallback tests run against.
    pub(crate) fn disabled() -> Self {
        #[cfg(feature = "cpal")]
        {
            Self { inner: None }
        }
        #[cfg(not(feature = "cpal"))]
        {
            Self {}
        }
    }

    /// Whether a real audio device is currently driving playback.
    pub(crate) fn is_active(&self) -> bool {
        #[cfg(feature = "cpal")]
        {
            self.inner.is_some()
        }
        #[cfg(not(feature = "cpal"))]
        {
            false
        }
    }

    /// Start playing `data` as voice `id` with the given effective linear `gain`
    /// (`0..`), stereo `panning` (`-1..1`), and playback `rate` (pitch factor).
    /// No-op when disabled.
    #[allow(unused_variables)]
    pub(crate) fn play(
        &mut self,
        id: VoiceId,
        data: &SoundData,
        gain: f64,
        panning: f64,
        rate: f64,
    ) {
        #[cfg(feature = "cpal")]
        if let Some(inner) = self.inner.as_mut() {
            inner.play(id, data, gain, panning, rate);
        }
    }

    /// Push updated mix parameters onto a live voice. No-op when disabled or the
    /// voice is unknown.
    #[allow(unused_variables)]
    pub(crate) fn set_params(&mut self, id: VoiceId, gain: f64, panning: f64, rate: f64) {
        #[cfg(feature = "cpal")]
        if let Some(inner) = self.inner.as_mut() {
            inner.set_params(id, gain, panning, rate);
        }
    }

    /// Pause a voice. No-op when disabled.
    #[allow(unused_variables)]
    pub(crate) fn pause(&mut self, id: VoiceId) {
        #[cfg(feature = "cpal")]
        if let Some(inner) = self.inner.as_mut() {
            inner.pause(id);
        }
    }

    /// Resume a paused voice. No-op when disabled.
    #[allow(unused_variables)]
    pub(crate) fn resume(&mut self, id: VoiceId) {
        #[cfg(feature = "cpal")]
        if let Some(inner) = self.inner.as_mut() {
            inner.resume(id);
        }
    }

    /// Stop and forget a voice. No-op when disabled.
    #[allow(unused_variables)]
    pub(crate) fn stop(&mut self, id: VoiceId) {
        #[cfg(feature = "cpal")]
        if let Some(inner) = self.inner.as_mut() {
            inner.stop(id);
        }
    }
}

#[cfg(feature = "cpal")]
mod cpal_impl {
    use std::collections::HashMap;

    use kira::backend::cpal::CpalBackend;
    use kira::sound::static_sound::StaticSoundHandle;
    use kira::{AudioManager, AudioManagerSettings, Decibels, Panning, PlaybackRate, Tween};

    use super::VoiceId;
    use crate::sound::SoundData;

    /// Convert a linear amplitude gain to kira decibels, flooring near-silence to
    /// −80 dB (instead of −∞).
    fn to_decibels(gain: f64) -> Decibels {
        let db = if gain <= 1e-4 {
            -80.0
        } else {
            20.0 * gain.log10()
        };
        Decibels(db as f32)
    }

    fn instant() -> Tween {
        // Zero-duration tween: mixing changes apply on the next audio frame. The
        // engine already does its own smoothing decisions; the facade stays literal.
        Tween {
            duration: std::time::Duration::ZERO,
            ..Default::default()
        }
    }

    pub(super) struct Inner {
        manager: AudioManager<CpalBackend>,
        voices: HashMap<VoiceId, StaticSoundHandle>,
    }

    impl Inner {
        pub(super) fn try_new() -> Option<Self> {
            match AudioManager::<CpalBackend>::new(AudioManagerSettings::default()) {
                Ok(manager) => Some(Self {
                    manager,
                    voices: HashMap::new(),
                }),
                // No device / no backend: the caller falls back to no-op mode.
                Err(_) => None,
            }
        }

        pub(super) fn play(
            &mut self,
            id: VoiceId,
            data: &SoundData,
            gain: f64,
            panning: f64,
            rate: f64,
        ) {
            let sound = data
                .inner
                .clone()
                .volume(to_decibels(gain))
                .panning(Panning(panning as f32))
                .playback_rate(PlaybackRate(rate));
            // `play` can fail if the command queue is momentarily full; a dropped
            // one-shot is a non-fatal, no-op outcome for the facade.
            if let Ok(handle) = self.manager.play(sound) {
                self.voices.insert(id, handle);
            }
        }

        pub(super) fn set_params(&mut self, id: VoiceId, gain: f64, panning: f64, rate: f64) {
            if let Some(h) = self.voices.get_mut(&id) {
                h.set_volume(to_decibels(gain), instant());
                h.set_panning(Panning(panning as f32), instant());
                h.set_playback_rate(PlaybackRate(rate), instant());
            }
        }

        pub(super) fn pause(&mut self, id: VoiceId) {
            if let Some(h) = self.voices.get_mut(&id) {
                h.pause(instant());
            }
        }

        pub(super) fn resume(&mut self, id: VoiceId) {
            if let Some(h) = self.voices.get_mut(&id) {
                h.resume(instant());
            }
        }

        pub(super) fn stop(&mut self, id: VoiceId) {
            if let Some(mut h) = self.voices.remove(&id) {
                h.stop(instant());
            }
        }
    }
}
