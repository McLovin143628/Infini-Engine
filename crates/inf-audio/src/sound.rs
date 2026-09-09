//! [`SoundData`]: decoded audio ready to play, and its decode error.
//!
//! Decoding uses **kira's own symphonia decoders** (enabled by this crate's
//! default features), so no separate codec dependency is introduced. Supported
//! input formats: **WAV/PCM, OGG (Vorbis), FLAC, and MP3**. Decoding needs no
//! audio device, so it works headlessly (and in CI) even when [`AudioEngine`]
//! is running its no-device fallback.
//!
//! [`AudioEngine`]: crate::AudioEngine

use std::io::Cursor;

use kira::sound::static_sound::StaticSoundData;

/// A failure to decode audio bytes into a [`SoundData`].
///
/// Wraps kira's underlying error as a message (kira's `FromFileError` is not on
/// our public surface — the facade never re-exports a kira type). Implements
/// `std::error::Error` so callers can `?` it up.
#[derive(Debug, Clone)]
pub struct DecodeError(String);

impl DecodeError {
    /// The underlying decoder message.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl core::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "audio decode failed: {}", self.0)
    }
}

impl std::error::Error for DecodeError {}

/// Decoded, ready-to-play audio. Cheap to `clone` (kira stores the samples behind
/// an `Arc`), so one decode can seed many concurrent plays.
#[derive(Clone)]
pub struct SoundData {
    pub(crate) inner: StaticSoundData,
}

/// **What it is, not what is in it** — a `Debug` that names the frame count and
/// the rate rather than thirty-three thousand samples.
///
/// It exists because [`crate::AudioEngine`]'s per-voice state holds the
/// UNFILTERED sound a voice started from (wave WPN2c's audit, closing carried
/// 235: a mixer edit mid-session has to be able to re-filter the voices that
/// are already playing, and the engine has no clip resolver outside `apply`),
/// and that state derives `Debug`.
impl core::fmt::Debug for SoundData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SoundData")
            .field("frames", &self.inner.num_frames())
            .field("sample_rate", &self.inner.sample_rate)
            .finish()
    }
}

impl SoundData {
    /// Decode audio from an in-memory byte buffer (WAV/OGG/FLAC/MP3 — the format
    /// is detected from the contents). Takes ownership of the bytes so the decoded
    /// sound is fully self-contained.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, DecodeError> {
        let inner = StaticSoundData::from_cursor(Cursor::new(bytes))
            .map_err(|e| DecodeError(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Decode WAV bytes. Convenience alias for [`from_bytes`](Self::from_bytes)
    /// (kira auto-detects the container); named for call-site clarity.
    pub fn from_wav_bytes(bytes: Vec<u8>) -> Result<Self, DecodeError> {
        Self::from_bytes(bytes)
    }

    /// Decode OGG (Vorbis) bytes. Convenience alias for
    /// [`from_bytes`](Self::from_bytes).
    pub fn from_ogg_bytes(bytes: Vec<u8>) -> Result<Self, DecodeError> {
        Self::from_bytes(bytes)
    }

    /// The sound's duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.inner.duration().as_secs_f64()
    }

    /// **A copy of this sound with a one-pole low-pass over its frames**
    /// (wave WPN2c) — the whole of how a cutoff becomes audible in this engine.
    ///
    /// # Why the samples and not a track effect
    ///
    /// See [`crate::filter`]: the cutoffs this engine decides are per VOICE (a
    /// shut door muffles one loop; a distant gunshot layer muffles one shot),
    /// and kira expresses a filter as a per-TRACK effect, so the alternative was
    /// a sub-track created and destroyed per one-shot at hundreds a second.
    /// Filtering the decoded frames once, before the voice starts, costs one
    /// pass over the clip and then nothing, works identically with and without a
    /// device, and is the only shape a headless test can measure.
    ///
    /// The filter is run **per channel**, so a stereo clip keeps its image.
    /// A cutoff at or above Nyquist is a pass-through and this is then a clone.
    ///
    /// It is not free: a 1.5 s tail at 22 050 Hz is 33 000 multiply-adds, which
    /// is why [`crate::AudioEngine`] caches the result by (clip, cutoff) rather
    /// than filtering the same tail on every shot.
    pub fn low_passed(&self, cutoff_hz: f64) -> Self {
        let rate = self.inner.sample_rate;
        let a = crate::filter::OnePole::alpha(cutoff_hz, rate);
        if a >= 1.0 {
            return self.clone();
        }
        let mut l = crate::filter::OnePole::new(cutoff_hz, rate);
        let mut r = crate::filter::OnePole::new(cutoff_hz, rate);
        let frames: Vec<kira::Frame> = self
            .inner
            .frames
            .iter()
            .map(|f| kira::Frame {
                left: l.step(f64::from(f.left)) as f32,
                right: r.step(f64::from(f.right)) as f32,
            })
            .collect();
        let mut inner = self.inner.clone();
        inner.frames = frames.into();
        Self { inner }
    }

    /// The sound's sample rate in hertz, derived from its frame count and
    /// duration (`0` if the duration is degenerate). Used to record `.inf_audio`
    /// metadata at import.
    pub fn sample_rate(&self) -> u32 {
        let secs = self.inner.duration().as_secs_f64();
        if secs <= 0.0 {
            return 0;
        }
        (self.inner.num_frames() as f64 / secs).round() as u32
    }
}
