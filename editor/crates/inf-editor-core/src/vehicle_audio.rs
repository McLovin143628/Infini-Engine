//! **The vehicle sound library** (wave VEH3e) — the thirty-one generated
//! `.inf_audio` clips a car is heard through, committed under
//! `samples/vehicle-audio/`.
//!
//! `crate::weapon_audio`'s shape and its reasons: engine content on
//! [`crate::ground`]'s terms (a car belongs to whatever drives one, not to a
//! sample level), every GUID computed by a Ring-0 constant
//! (`inf_ecs::vehicle_audio::vehicle_clip`), every byte the output of
//! `inf_audio::vehicle_synth` — arithmetic, so the files are licence-free and
//! regenerate byte for byte on any machine. Both island recipes name every
//! file, and the cook closes over them through
//! `inf_ecs::audio::engine_spawned_clips`.

use uuid::Uuid;

use inf_audio::{AudioAsset, AudioFormat};

/// The folder under `samples/` this library lives in.
pub const VEHICLE_AUDIO_FOLDER: &str = "vehicle-audio";

/// **What one committed clip is** — its file, its GUID, its payload.
pub struct VehicleClip {
    /// The file name under [`VEHICLE_AUDIO_FOLDER`].
    pub file: String,
    /// The GUID `inf_ecs::vehicle_audio` names it by.
    pub guid: Uuid,
    /// The generated payload.
    pub asset: AudioAsset,
}

/// **The whole library**, in clip-index order.
pub fn vehicle_audio_clips() -> Vec<VehicleClip> {
    inf_ecs::vehicle_audio::VEHICLE_CLIP_NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| VehicleClip {
            file: format!("Vehicle_{name}.inf_audio"),
            guid: inf_ecs::vehicle_audio::vehicle_clip(i as u8),
            asset: AudioAsset::from_encoded(
                inf_audio::vehicle_synth::vehicle_clip_wav(i as u8),
                AudioFormat::Wav,
            )
            .expect("a generated clip decodes"),
        })
        .collect()
}

/// **Every file this library commits**, payload and sidecar, sorted.
pub fn vehicle_audio_files() -> Vec<String> {
    let mut out: Vec<String> = vehicle_audio_clips()
        .into_iter()
        .flat_map(|c| [format!("{}.toml", c.file), c.file])
        .collect();
    out.sort();
    out
}

/// Write the library into `dir`, payload plus `inf_asset` sidecar.
pub fn write_vehicle_audio_library(dir: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    for clip in vehicle_audio_clips() {
        let bytes =
            inf_asset::encode(&clip.asset).map_err(|e| format!("encode {}: {e}", clip.file))?;
        let path = dir.join(&clip.file);
        std::fs::write(&path, &bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(clip.guid),
            inf_asset::AssetKind::Audio,
            inf_asset::ContentHash::of(&bytes),
        )
        .save(&path)
        .map_err(|e| format!("write the sidecar for {}: {e}", clip.file))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Thirty-six clips (the five craft voices of wave VEH3g appended), as many
    /// GUIDs, as many names, each one
    /// decoding — and the names are the generator's, index for index.
    #[test]
    fn the_library_is_twenty_eight_distinct_resolvable_clips() {
        let clips = vehicle_audio_clips();
        assert_eq!(clips.len(), 36);
        assert_eq!(
            inf_ecs::vehicle_audio::VEHICLE_CLIP_NAMES,
            inf_audio::vehicle_synth::VEHICLE_CLIP_NAMES,
            "the engine and the generator number the clips differently"
        );
        let mut guids = std::collections::BTreeSet::new();
        let mut bytes = 0usize;
        for c in &clips {
            assert!(guids.insert(c.guid), "{} shares a GUID", c.file);
            assert!(c.asset.decode().is_ok(), "{} does not decode", c.file);
            assert_eq!(c.asset.sample_rate, inf_audio::synth::SYNTH_RATE);
            bytes += c.asset.bytes.len();
        }
        println!(
            "the vehicle-audio library is {} clips, {:.1} KiB of WAV",
            clips.len(),
            bytes as f64 / 1024.0
        );
        assert_eq!(vehicle_audio_files().len(), 72);
    }
}
