//! The `.inf_skel` / `.inf_anim` payload schemas (P11.1, ROADMAP §3).
//!
//! Thin versioned wrappers around the pure [`Skeleton`] / [`AnimClip`] data
//! model, implementing [`AssetPayload`] so the asset database stores them with
//! the dual-format rule (bincode payload + TOML sidecar). Deterministic serde;
//! **no `skip_serializing_if`** on any field — the recurring engine law for
//! bincode-bound types (a skipped field desyncs the sequential decoder).

use inf_asset::{AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

use crate::clip::AnimClip;
use crate::skeleton::Skeleton;

/// The `.inf_skel` payload: a skinning [`Skeleton`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkeletonAsset {
    pub schema_version: u32,
    pub skeleton: Skeleton,
}

impl SkeletonAsset {
    pub const CURRENT_VERSION: u32 = 1;

    /// Wrap a skeleton as a current-schema asset.
    pub fn new(skeleton: Skeleton) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            skeleton,
        }
    }
}

impl AssetPayload for SkeletonAsset {
    const KIND: AssetKind = AssetKind::Skeleton;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

/// The `.inf_anim` payload: one [`AnimClip`] plus the GUID of the skeleton it
/// was authored against (stored as raw bytes so this crate needs no `uuid` dep;
/// the editor sets the dependency edge). `None` = skeleton-agnostic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimClipAsset {
    pub schema_version: u32,
    pub clip: AnimClip,
    /// Raw 16-byte GUID of the skeleton this clip targets (a dependency edge the
    /// importer/editor also records in the sidecar). `None` = unbound.
    pub skeleton: Option<[u8; 16]>,
}

impl AnimClipAsset {
    pub const CURRENT_VERSION: u32 = 1;

    /// Wrap a clip (optionally bound to a skeleton GUID) as a current-schema asset.
    pub fn new(clip: AnimClip, skeleton: Option<[u8; 16]>) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            clip,
            skeleton,
        }
    }
}

impl AssetPayload for AnimClipAsset {
    const KIND: AssetKind = AssetKind::AnimClip;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{AnimClip, Interpolation, JointTrack, QuatTrack};
    use crate::skeleton::{Joint, JointTransform};
    use glam::{Mat4, Quat};
    use inf_asset::{decode, encode};

    fn skel() -> Skeleton {
        Skeleton::new(vec![
            Joint {
                name: "root".into(),
                parent: None,
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::IDENTITY,
            },
            Joint {
                name: "child".into(),
                parent: Some(0),
                inverse_bind: Mat4::IDENTITY.to_cols_array(),
                local_bind: JointTransform::IDENTITY,
            },
        ])
        .unwrap()
    }

    #[test]
    fn skeleton_asset_round_trips_deterministically() {
        let a = SkeletonAsset::new(skel());
        let e1 = encode(&a).unwrap();
        let e2 = encode(&a).unwrap();
        assert_eq!(e1, e2, "re-encoding is byte-identical");
        let back: SkeletonAsset = decode(&e1).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn clip_asset_round_trips_deterministically() {
        let mut jt = JointTrack::new(1);
        jt.rotation = Some(QuatTrack::new(
            vec![0.0, 1.0],
            vec![
                Quat::IDENTITY.to_array(),
                Quat::from_rotation_z(1.0).to_array(),
            ],
            Interpolation::Linear,
        ));
        let clip = AnimClip::new("spin", vec![jt]);
        let a = AnimClipAsset::new(clip, Some([7u8; 16]));
        let e1 = encode(&a).unwrap();
        let e2 = encode(&a).unwrap();
        assert_eq!(e1, e2);
        let back: AnimClipAsset = decode(&e1).unwrap();
        assert_eq!(back, a);
        assert_eq!(back.skeleton, Some([7u8; 16]));
    }
}
