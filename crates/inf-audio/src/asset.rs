//! The `.inf_audio` asset payload (P12.3) + its import settings sidecar.
//!
//! # Why the payload stores the *original compressed bytes*, not PCM
//!
//! An [`AudioAsset`] carries the **original encoded file bytes** (WAV/OGG/FLAC/
//! MP3) plus a [`format`](AudioAsset::format) tag and cheap duration metadata,
//! and is decoded on load via kira ([`SoundData::from_bytes`]). We deliberately do
//! **not** store decoded PCM: a few minutes of 48 kHz stereo PCM is tens of
//! megabytes, whereas the compressed source is a fraction of that — bloating every
//! `.inf_audio` and the cooked pack for no runtime gain (kira decodes once at load,
//! then plays from an `Arc`). Duration is decoded *once at import* so the editor
//! can show it without touching the bytes again.
//!
//! Mirrors `inf_mesh::MeshAsset`: a `schema_version` field, a `CURRENT_VERSION`
//! const, and an [`AssetPayload`] impl keyed to [`AssetKind::Audio`].

use inf_asset::{AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

use crate::sound::{DecodeError, SoundData};

/// The container format of the stored bytes. A tag only — kira auto-detects the
/// container on decode; this records the import source for UI/tooling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    /// WAV / PCM.
    #[default]
    Wav,
    /// OGG (Vorbis).
    Ogg,
    /// FLAC (lossless).
    Flac,
    /// MP3.
    Mp3,
    /// Unknown / other (still decodes via kira's content sniffing).
    Unknown,
}

impl AudioFormat {
    /// Classify from a lowercase file extension (`"wav"`, `"ogg"`, …).
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            "wav" => Self::Wav,
            "ogg" => Self::Ogg,
            "flac" => Self::Flac,
            "mp3" => Self::Mp3,
            _ => Self::Unknown,
        }
    }
}

/// The `.inf_audio` payload: original compressed bytes + format tag + duration
/// metadata (decode-on-load; see the module docs for why not PCM).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioAsset {
    pub schema_version: u32,
    /// Container format of [`bytes`](Self::bytes).
    pub format: AudioFormat,
    /// The original encoded file bytes (WAV/OGG/FLAC/MP3).
    pub bytes: Vec<u8>,
    /// Clip duration in seconds, decoded once at import.
    pub duration_secs: f64,
    /// Sample rate in hertz, decoded once at import (`0` = unknown).
    pub sample_rate: u32,
}

impl AudioAsset {
    /// Schema v1.
    pub const CURRENT_VERSION: u32 = 1;

    /// Build a payload from encoded bytes, decoding once to record duration +
    /// sample rate metadata. Fails only if the bytes don't decode.
    pub fn from_encoded(bytes: Vec<u8>, format: AudioFormat) -> Result<Self, DecodeError> {
        let sound = SoundData::from_bytes(bytes.clone())?;
        let duration_secs = sound.duration_secs();
        let sample_rate = sound.sample_rate();
        Ok(Self {
            schema_version: Self::CURRENT_VERSION,
            format,
            bytes,
            duration_secs,
            sample_rate,
        })
    }

    /// Decode the stored bytes into a ready-to-play [`SoundData`].
    pub fn decode(&self) -> Result<SoundData, DecodeError> {
        SoundData::from_bytes(self.bytes.clone())
    }
}

impl AssetPayload for AudioAsset {
    const KIND: AssetKind = AssetKind::Audio;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

/// Per-import settings written to the asset's TOML sidecar `import` table
/// (mirrors `inf_material::TextureImportSettings`). Drives default playback when a
/// scene-placed [`AudioSource`](../../inf_ecs) references the clip.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AudioImportSettings {
    /// Default `looping` flag stamped onto a new [`AudioSource`].
    pub looping: bool,
    /// Default mixer bus name for the clip.
    #[serde(default = "default_bus")]
    pub default_bus: BusChoice,
    /// A volume trim in `[0, 1]` applied as the source's base volume.
    pub volume_trim: f64,
}

fn default_bus() -> BusChoice {
    BusChoice::Sfx
}

/// A small, serde-stable default-bus choice for import settings (the mixer's
/// named buses are open-ended, but new clips default to one of the built-ins).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BusChoice {
    Master,
    #[default]
    Sfx,
    Music,
}

impl BusChoice {
    /// The mixer bus name this choice maps to.
    pub fn name(self) -> &'static str {
        match self {
            BusChoice::Master => "master",
            BusChoice::Sfx => "sfx",
            BusChoice::Music => "music",
        }
    }
}

impl Default for AudioImportSettings {
    fn default() -> Self {
        Self {
            looping: false,
            default_bus: BusChoice::Sfx,
            volume_trim: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::{decode, encode};

    /// A minimal valid 16-bit mono PCM WAV, so the payload test drives kira's real
    /// decode path without a binary fixture (mirrors the facade tests).
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
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes());
        w.extend_from_slice(&channels.to_le_bytes());
        w.extend_from_slice(&sample_rate.to_le_bytes());
        w.extend_from_slice(&byte_rate.to_le_bytes());
        w.extend_from_slice(&block_align.to_le_bytes());
        w.extend_from_slice(&bits.to_le_bytes());
        w.extend_from_slice(b"data");
        w.extend_from_slice(&data_len.to_le_bytes());
        for i in 0..samples {
            let s = (i as i16).wrapping_mul(64);
            w.extend_from_slice(&s.to_le_bytes());
        }
        w
    }

    #[test]
    fn payload_records_duration_and_round_trips_via_bincode() {
        let wav = tone_wav(8000, 8000);
        let asset = AudioAsset::from_encoded(wav.clone(), AudioFormat::Wav).unwrap();
        assert!((asset.duration_secs - 1.0).abs() < 0.05);
        assert_eq!(asset.sample_rate, 8000);
        assert_eq!(asset.bytes, wav);

        // The dual-format bincode round-trip (encode/decode go through migrate()).
        let bytes = encode(&asset).unwrap();
        let back: AudioAsset = decode(&bytes).unwrap();
        assert_eq!(asset, back);
        // And it still decodes to a playable sound.
        assert!(back.decode().is_ok());
    }

    #[test]
    fn import_settings_serialize_to_a_toml_table() {
        let s = AudioImportSettings::default();
        let v = toml::Value::try_from(s).unwrap();
        assert!(v.as_table().is_some());
        assert_eq!(BusChoice::Sfx.name(), "sfx");
    }
}
