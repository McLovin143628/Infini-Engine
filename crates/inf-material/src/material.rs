//! The `.inf_mat` material payload.
//!
//! Phase 4 ships the material *model* (a PBR metallic-roughness parameter block
//! with texture references by GUID). The node-graph editor and WGSL codegen are
//! Phase 7 — this is the data those build on, and what an imported glTF material
//! becomes.

use inf_asset::{AssetId, AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

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
        }
    }
}

impl MaterialAsset {
    pub const CURRENT_VERSION: u32 = 1;

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
}
