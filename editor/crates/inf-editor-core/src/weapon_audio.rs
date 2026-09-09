//! **The gunshot library** (wave WPN2c) — the thirty-five generated
//! `.inf_audio` clips a weapon's report and its brass are built from, committed
//! under `samples/weapon-audio/`.
//!
//! # Why a library and not a fixture
//!
//! `crate::samples::GAMEPLAY_REPORT_FILE`'s own doc says it: *"it sits [beside
//! the gameplay fixture] rather than in the settlement library because a
//! gunshot is not settlement content and there is no weapon library to put it
//! in."* This is that weapon library. It is engine content on
//! [`crate::ground`]'s terms — the island binds it, a template can, and any
//! project can drag it in — because a gunshot belongs to whatever fires one and
//! not to a sample level.
//!
//! # Thirty-five, and where the thirty-sixth is
//!
//! Seven classes times five layer clips is thirty-five, plus the casing's own
//! metal one-shot is thirty-six — and the ASSAULT RIFLE'S BODY is **not here**.
//! That one is `inf_ecs::weapon::WEAPON_REPORT_CLIP`, which has been committed
//! as `samples/phase30-gameplay/Report.inf_audio` since wave WPN1, is named by
//! the gameplay gate's own file list, and is counted by that gate's report
//! arm. A GUID has to name exactly one file, so it stays where it is and this
//! library holds the other thirty-five. Both island recipes name it beside
//! them.
//!
//! The asymmetry is carried deliberately rather than tidied: moving it means
//! moving a file two gates name, and what is bought is a directory listing.

use uuid::Uuid;

use inf_audio::{AudioAsset, AudioFormat};
use inf_ecs::weapon::{
    melee_impact_clip, report_clip, ImpactSurface, ReportClip, WeaponClass, BLAST_CLIP,
    CASING_CLIP, WEAPON_REPORT_CLIP,
};

/// The folder under `samples/` this library lives in.
pub const WEAPON_AUDIO_FOLDER: &str = "weapon-audio";

/// **The file the casing's metal one-shot lives in.**
pub const CASING_FILE: &str = "Casing_Drop.inf_audio";

/// **The file the blast's boom lives in** (wave WPN2d).
pub const BLAST_FILE: &str = "Blast.inf_audio";

/// **What one committed clip is** — the file it lives in, the GUID that names
/// it, and the payload.
pub struct WeaponClip {
    /// The file name under [`WEAPON_AUDIO_FOLDER`].
    pub file: String,
    /// The GUID `inf_ecs::weapon` names it by.
    pub guid: Uuid,
    /// The generated payload.
    pub asset: AudioAsset,
}

/// **The whole library**, in a fixed order (class, then layer, then the
/// casing) — so the committed set is reproducible and a diff is readable.
///
/// The assault rifle's body is skipped; see this module's own header.
pub fn weapon_audio_clips() -> Vec<WeaponClip> {
    let mut out = Vec::with_capacity(38);
    for class in WeaponClass::ALL {
        for clip in ReportClip::ALL {
            let guid = report_clip(class, clip);
            if guid == WEAPON_REPORT_CLIP {
                continue;
            }
            let bytes = inf_audio::synth::report_wav(class.index(), clip.index());
            out.push(WeaponClip {
                file: format!(
                    "Report_{}_{}.inf_audio",
                    capitalised(class.name()),
                    clip.file_stem()
                ),
                guid,
                asset: AudioAsset::from_encoded(bytes, AudioFormat::Wav)
                    .expect("a generated clip decodes"),
            });
        }
    }
    out.push(WeaponClip {
        file: CASING_FILE.to_string(),
        guid: CASING_CLIP,
        asset: AudioAsset::from_encoded(inf_audio::synth::casing_wav(), AudioFormat::Wav)
            .expect("a generated clip decodes"),
    });
    // **Wave WPN2d's three**: the blast, and one melee impact per surface. They
    // are here rather than in a library of their own for the reason this module
    // exists -- a gunshot belongs to whatever fires one, and so does the bang
    // its rocket makes and the sound its knife makes landing.
    out.push(WeaponClip {
        file: BLAST_FILE.to_string(),
        guid: BLAST_CLIP,
        asset: AudioAsset::from_encoded(inf_audio::synth::blast_wav(), AudioFormat::Wav)
            .expect("a generated clip decodes"),
    });
    for surface in ImpactSurface::ALL {
        out.push(WeaponClip {
            file: format!("Melee_{}.inf_audio", capitalised(surface.name())),
            guid: melee_impact_clip(surface),
            asset: AudioAsset::from_encoded(
                inf_audio::synth::melee_impact_wav(surface.index()),
                AudioFormat::Wav,
            )
            .expect("a generated clip decodes"),
        });
    }
    out
}

/// `ar` becomes `Ar`, `smg` becomes `Smg` — the file names are read by people.
fn capitalised(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + c.as_str(),
        None => String::new(),
    }
}

/// **Every file this library commits**, payload and sidecar, sorted — the set a
/// recipe has to name and a gate compares against the folder.
///
/// `crate::settlement::settlement_files`' shape exactly, and for its reason: a
/// library whose file list is only in a `write_*` function is a library a
/// recipe silently stops naming half of.
pub fn weapon_audio_files() -> Vec<String> {
    let mut out: Vec<String> = weapon_audio_clips()
        .into_iter()
        .flat_map(|c| [format!("{}.toml", c.file), c.file])
        .collect();
    out.sort();
    out
}

/// Write the library into `dir`, payload plus `inf_asset` sidecar, on
/// `crate::settlement::write_settlement_library`'s own pattern.
pub fn write_weapon_audio_library(dir: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    for clip in weapon_audio_clips() {
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

    /// Thirty-eight clips, thirty-eight distinct GUIDs, thirty-eight distinct
    /// names, and every one of them decodes — which is the whole contract with
    /// the engine, since a `Play` that names a clip nothing resolves is silence.
    ///
    /// Thirty-five until wave WPN2d, which added the blast and the two melee
    /// surfaces (and the casing has always been the thirty-sixth).
    #[test]
    fn the_library_is_thirty_eight_distinct_resolvable_clips() {
        let clips = weapon_audio_clips();
        assert_eq!(
            clips.len(),
            38,
            "seven classes times five, less the rifle's body, plus the casing,              the blast and the two melee surfaces"
        );
        let mut guids = std::collections::BTreeSet::new();
        let mut files = std::collections::BTreeSet::new();
        let mut bytes = 0usize;
        for c in &clips {
            assert!(guids.insert(c.guid), "{} shares a GUID", c.file);
            assert!(files.insert(c.file.clone()), "{} is named twice", c.file);
            assert!(c.asset.decode().is_ok(), "{} does not decode", c.file);
            assert_eq!(c.asset.sample_rate, inf_audio::synth::SYNTH_RATE);
            bytes += c.asset.bytes.len();
        }
        // The rifle's body is deliberately absent — it is WPN1's committed file.
        assert!(!guids.contains(&WEAPON_REPORT_CLIP));
        assert!(guids.contains(&CASING_CLIP));
        println!(
            "the weapon-audio library is {} clips, {:.1} KiB of WAV",
            clips.len(),
            bytes as f64 / 1024.0
        );
        // Seventy-six files: a payload and a sidecar each, and nothing else.
        let files = weapon_audio_files();
        assert_eq!(files.len(), 76);
        assert_eq!(
            files.iter().filter(|f| f.ends_with(".toml")).count(),
            38,
            "every payload needs a sidecar or its GUID is invisible"
        );
    }

    /// The names are the class and the layer, spelled the way a person reads
    /// them — so a directory listing is a table.
    #[test]
    fn a_clips_file_name_says_which_gun_and_which_layer() {
        let files: Vec<String> = weapon_audio_clips().into_iter().map(|c| c.file).collect();
        for want in [
            "Report_Pistol_Transient.inf_audio",
            "Report_Ar_OutdoorTail.inf_audio",
            "Report_Launcher_Crack.inf_audio",
            "Casing_Drop.inf_audio",
        ] {
            assert!(files.iter().any(|f| f == want), "{want} is missing");
        }
        // …and the one that is NOT here.
        assert!(!files.iter().any(|f| f == "Report_Ar_Body.inf_audio"));
    }
}
