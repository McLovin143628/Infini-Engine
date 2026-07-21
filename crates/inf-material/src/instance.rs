//! Material instances (`.inf_mati`, P7.4): a lightweight asset that inherits a
//! parent [`MaterialAsset`] and overrides a *subset* of its parameters. Instances
//! keep a scene consistent (edit the parent, every instance follows) while
//! letting individual uses tweak a colour or roughness.

use inf_asset::{AssetId, AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

use crate::MaterialAsset;

/// The sparse set of parameters an instance may override. `None` = inherit.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MatOverrides {
    #[serde(default)]
    pub base_color: Option<[f32; 4]>,
    #[serde(default)]
    pub metallic: Option<f32>,
    #[serde(default)]
    pub roughness: Option<f32>,
    #[serde(default)]
    pub emissive: Option<[f32; 3]>,
}

impl MatOverrides {
    /// True when nothing is overridden (the instance equals its parent).
    pub fn is_empty(&self) -> bool {
        self.base_color.is_none()
            && self.metallic.is_none()
            && self.roughness.is_none()
            && self.emissive.is_none()
    }
}

/// A material instance: parent + sparse overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialInstance {
    pub schema_version: u32,
    pub parent: AssetId,
    #[serde(default)]
    pub overrides: MatOverrides,
}

impl MaterialInstance {
    pub const CURRENT_VERSION: u32 = 1;

    /// A fresh instance of `parent` with no overrides (equals the parent).
    pub fn new(parent: AssetId) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            parent,
            overrides: MatOverrides::default(),
        }
    }

    /// Resolve to a concrete material by overlaying the overrides on an
    /// already-resolved `parent`.
    pub fn resolve(&self, parent: &MaterialAsset) -> MaterialAsset {
        let mut m = parent.clone();
        if let Some(v) = self.overrides.base_color {
            m.base_color = v;
        }
        if let Some(v) = self.overrides.metallic {
            m.metallic = v;
        }
        if let Some(v) = self.overrides.roughness {
            m.roughness = v;
        }
        if let Some(v) = self.overrides.emissive {
            m.emissive = v;
        }
        m
    }

    /// Dependency edges (the parent material).
    pub fn dependencies(&self) -> Vec<AssetId> {
        vec![self.parent]
    }
}

impl AssetPayload for MaterialInstance {
    const KIND: AssetKind = AssetKind::MaterialInstance;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::{decode, encode};

    fn parent() -> MaterialAsset {
        MaterialAsset {
            base_color: [0.8, 0.2, 0.1, 1.0],
            metallic: 1.0,
            roughness: 0.3,
            emissive: [0.0; 3],
            ..Default::default()
        }
    }

    #[test]
    fn overrides_apply_and_unset_inherit() {
        let p = parent();
        let mut inst = MaterialInstance::new(inf_asset::AssetId::new());
        inst.overrides.roughness = Some(0.9);
        inst.overrides.base_color = Some([0.1, 0.1, 0.9, 1.0]);
        let r = inst.resolve(&p);
        // Overridden.
        assert_eq!(r.roughness, 0.9);
        assert_eq!(r.base_color, [0.1, 0.1, 0.9, 1.0]);
        // Inherited.
        assert_eq!(r.metallic, 1.0);
        assert_eq!(r.emissive, p.emissive);
    }

    #[test]
    fn empty_instance_equals_parent() {
        let p = parent();
        let inst = MaterialInstance::new(inf_asset::AssetId::new());
        assert!(inst.overrides.is_empty());
        assert_eq!(inst.resolve(&p), p);
    }

    #[test]
    fn depends_on_parent_and_round_trips() {
        let pid = inf_asset::AssetId::new();
        let inst = MaterialInstance::new(pid);
        assert_eq!(inst.dependencies(), vec![pid]);
        assert_eq!(
            decode::<MaterialInstance>(&encode(&inst).unwrap()).unwrap(),
            inst
        );
    }
}
