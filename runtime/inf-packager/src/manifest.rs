//! The cooked-build `manifest.toml` (P9.2).
//!
//! A small, deterministic, human-readable descriptor written beside the pack(s).
//! The player reads it to find the root level and the pack list; humans read it
//! to see what a cook produced. Deterministic emission (fixed field order, `toml`
//! sorts map keys, no timestamps) keeps it byte-stable across cooks.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The manifest file name written into the cook output directory.
pub const MANIFEST_FILE: &str = "manifest.toml";

/// Current manifest schema version.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

/// The cooked-build manifest. Scalar/array fields precede the single `[kinds]`
/// table so `toml` emits a valid document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CookManifest {
    pub schema_version: u32,
    /// The engine version that produced this build.
    pub engine_version: String,
    /// The project display name.
    pub project_name: String,
    /// Pack files in the output directory (relative names).
    pub packs: Vec<String>,
    /// The primary level to boot into (lowest-GUID level root), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub root_level: Option<Uuid>,
    /// Every level GUID in the pack, sorted.
    pub levels: Vec<Uuid>,
    /// Total asset count in the pack.
    pub asset_count: u32,
    /// **The project's session-default animation blend mode**, copied from
    /// `inf.toml` (`inf_project::ProjectManifest::anim_blend`).
    ///
    /// The shipped player links no editor crate and `inf-project` is only a
    /// **dev**-dependency of it, so `inf.toml` is unreadable at boot. Carrying
    /// the name here is what closes the gap the I1 audit measured: the PIE path
    /// applies `ScenePayload::blend_mode` and the cooked path applied nothing at
    /// all, so a project that set the mode would preview one blend and ship
    /// another.
    ///
    /// A defaulted string in a name-keyed TOML, so `MANIFEST_SCHEMA_VERSION`
    /// does not move: an older player ignores the key, and an older manifest
    /// reads as the engine's own default.
    #[serde(default = "default_anim_blend")]
    pub anim_blend: String,
    /// Per-kind asset counts, keyed by the kind slug (sorted by `BTreeMap`).
    pub kinds: BTreeMap<String, u32>,
}

fn default_anim_blend() -> String {
    inf_project::ANIM_BLEND_INERTIALIZE.to_string()
}

impl CookManifest {
    /// Serialize to deterministic pretty TOML.
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// Parse from TOML text.
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_is_deterministic() {
        let mut kinds = BTreeMap::new();
        kinds.insert("level".to_string(), 1u32);
        kinds.insert("blueprint".to_string(), 2u32);
        let m = CookManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            engine_version: "0.1.0".into(),
            project_name: "Game".into(),
            packs: vec!["content.inf_pack".into()],
            root_level: Some(Uuid::from_u128(0x84_0100)),
            levels: vec![Uuid::from_u128(0x84_0100)],
            asset_count: 3,
            anim_blend: "crossfade".into(),
            kinds,
        };
        let t1 = m.to_toml().unwrap();
        assert_eq!(t1, m.to_toml().unwrap(), "re-emit is byte-identical");
        assert_eq!(CookManifest::from_toml(&t1).unwrap(), m);

        // **A manifest written before the key existed still reads.** The player
        // is the reader that matters and it defaults the field too, so an old
        // build boots on the engine's own blend rather than refusing.
        let older = t1
            .lines()
            .filter(|l| !l.starts_with("anim_blend"))
            .collect::<Vec<_>>()
            .join("\n");
        let back = CookManifest::from_toml(&older).expect("an older manifest reads");
        assert_eq!(back.anim_blend, "inertialize");
        assert_eq!(back.asset_count, 3, "and the rest of it is unchanged");
    }
}
