//! The `.inf_mat` material payload.
//!
//! Phase 4 ships the material *model* (a PBR metallic-roughness parameter block
//! with texture references by GUID). The node-graph editor and WGSL codegen are
//! Phase 7 — this is the data those build on, and what an imported glTF material
//! becomes.

use inf_asset::{AssetId, AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

/// How a [`MaterialAsset`] blends against the framebuffer (R-P5). Ring-0 mirror of
/// the ECS `inf_ecs::BlendMode` (inf-material must NOT depend on inf-ecs — the
/// editor glue maps between the two): `Opaque` is the pre-R-P5 behaviour;
/// `Masked` alpha-tests against [`MaterialAsset::alpha_cutoff`]; `Translucent`
/// alpha-blends. Serialized as an externally-tagged enum (bincode-safe — no
/// internal tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MatBlend {
    #[default]
    Opaque,
    Masked,
    Translucent,
}

/// A PBR metallic-roughness material. Texture slots hold asset GUIDs (the
/// material's dependency edges); `None` means "use the factor alone".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialAsset {
    pub schema_version: u32,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    #[serde(default)]
    pub base_color_texture: Option<AssetId>,
    #[serde(default)]
    pub normal_texture: Option<AssetId>,
    #[serde(default)]
    pub metallic_roughness_texture: Option<AssetId>,
    /// Blend / transparency mode (schema v2). Additive: `#[serde(default)]` →
    /// [`MatBlend::Opaque`], the pre-v2 behaviour.
    #[serde(default)]
    pub blend: MatBlend,
    /// Alpha-test threshold used when `blend == MatBlend::Masked` (schema v2).
    #[serde(default = "default_alpha_cutoff")]
    pub alpha_cutoff: f32,
}

fn default_alpha_cutoff() -> f32 {
    0.5
}

impl Default for MaterialAsset {
    fn default() -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            base_color: [0.8, 0.8, 0.8, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            base_color_texture: None,
            normal_texture: None,
            metallic_roughness_texture: None,
            blend: MatBlend::Opaque,
            alpha_cutoff: default_alpha_cutoff(),
        }
    }
}

impl MaterialAsset {
    /// v2 (R-P5): added `blend` + `alpha_cutoff`. Old v1 payloads decode with the
    /// additive defaults (Opaque / 0.5) — the "old → Opaque" migration.
    pub const CURRENT_VERSION: u32 = 2;

    /// Every texture GUID this material references, for building the asset
    /// dependency edges.
    pub fn texture_dependencies(&self) -> Vec<AssetId> {
        [
            self.base_color_texture,
            self.normal_texture,
            self.metallic_roughness_texture,
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

impl AssetPayload for MaterialAsset {
    const KIND: AssetKind = AssetKind::Material;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::{decode, encode};

    #[test]
    fn dependencies_list_present_textures() {
        let tex = AssetId::new();
        let m = MaterialAsset {
            base_color_texture: Some(tex),
            ..Default::default()
        };
        assert_eq!(m.texture_dependencies(), vec![tex]);
        assert!(MaterialAsset::default().texture_dependencies().is_empty());
    }

    #[test]
    fn round_trips() {
        let m = MaterialAsset::default();
        assert_eq!(decode::<MaterialAsset>(&encode(&m).unwrap()).unwrap(), m);
    }

    #[test]
    fn default_blend_is_opaque_v2() {
        let m = MaterialAsset::default();
        assert_eq!(m.blend, MatBlend::Opaque);
        assert_eq!(m.alpha_cutoff, 0.5);
        assert_eq!(m.schema_version, 2);
        assert_eq!(MaterialAsset::CURRENT_VERSION, 2);
    }

    /// A translucent/masked material round-trips its blend + cutoff through
    /// bincode (schema v2). The default `migrate` accepts an equal-or-older
    /// version and leaves the additive defaults (old → Opaque) untouched.
    #[test]
    fn blend_and_cutoff_round_trip_and_migrate() {
        let m = MaterialAsset {
            blend: MatBlend::Translucent,
            alpha_cutoff: 0.3,
            ..Default::default()
        };
        let back = decode::<MaterialAsset>(&encode(&m).unwrap()).unwrap();
        assert_eq!(back.blend, MatBlend::Translucent);
        assert_eq!(back.alpha_cutoff, 0.3);
        // A value stamped with the older schema still migrates cleanly (version
        // guard only — additive fields keep their Opaque defaults).
        let old = MaterialAsset {
            schema_version: 1,
            ..Default::default()
        };
        let migrated = old.migrate().unwrap();
        assert_eq!(migrated.blend, MatBlend::Opaque);
    }
}
