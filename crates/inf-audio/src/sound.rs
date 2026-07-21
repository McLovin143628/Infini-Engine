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
}
