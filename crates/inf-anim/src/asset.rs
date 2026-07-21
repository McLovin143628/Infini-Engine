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
use crate::sockets::Socket;
use crate::state_machine::StateMachine;

/// The `.inf_skel` payload: a skinning [`Skeleton`] plus its authored
/// [`Socket`]s (P11.3 — sockets are per-skeleton attach points).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkeletonAsset {
    pub schema_version: u32,
    pub skeleton: Skeleton,
    /// Named attach points riding the skeleton's joints. **Additive** (P11.3):
    /// `#[serde(default)]` so a pre-socket `.inf_skel` payload still decodes to an
    /// empty socket list. Serde-clean (no `skip_serializing_if`) per the crate's
    /// bincode law.
    #[serde(default)]
    pub sockets: Vec<Socket>,
}

impl SkeletonAsset {
    pub const CURRENT_VERSION: u32 = 1;

    /// Wrap a skeleton as a current-schema asset (no sockets).
    pub fn new(skeleton: Skeleton) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            skeleton,
            sockets: Vec::new(),
        }
    }

    /// Wrap a skeleton and its authored sockets.
    pub fn with_sockets(skeleton: Skeleton, sockets: Vec<Socket>) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            skeleton,
            sockets,
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

/// The `.inf_sm` payload: an authored animation [`StateMachine`] (P11.2).
///
/// The machine is a plain typed model (states + transitions + layout), not an
/// `inf-graph` document, so it stores directly here — no JSON-string escape hatch
/// (the PCG-style workaround that model needs for its `skip_serializing_if`
/// fields does not apply: [`StateMachine`] and everything it contains is
/// serde-clean). The optional `skeleton` GUID records the dependency edge (raw
/// bytes, keeping the crate `uuid`-free), like [`AnimClipAsset`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateMachineAsset {
    pub schema_version: u32,
    pub machine: StateMachine,
    /// Raw 16-byte GUID of the skeleton this machine animates. `None` = unbound.
    pub skeleton: Option<[u8; 16]>,
}

impl StateMachineAsset {
    pub const CURRENT_VERSION: u32 = 1;

    /// Wrap a machine (optionally bound to a skeleton GUID) as a current-schema
    /// asset.
    pub fn new(machine: StateMachine, skeleton: Option<[u8; 16]>) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            machine,
            skeleton,
        }
    }
}

impl AssetPayload for StateMachineAsset {
    const KIND: AssetKind = AssetKind::StateMachine;
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
        assert!(back.sockets.is_empty());
    }

    #[test]
    fn skeleton_asset_with_sockets_round_trips() {
        use crate::skeleton::JointTransform;
        use crate::sockets::Socket;
        let sockets = vec![
            Socket::new("hand_r", 1),
            Socket::new("muzzle", 1).with_offset(JointTransform::from_trs(
                glam::Vec3::new(0.0, 0.2, 0.0),
                Quat::from_rotation_y(0.5),
                glam::Vec3::ONE,
            )),
        ];
        let a = SkeletonAsset::with_sockets(skel(), sockets.clone());
        let e1 = encode(&a).unwrap();
        let back: SkeletonAsset = decode(&e1).unwrap();
        assert_eq!(back, a);
        assert_eq!(back.sockets, sockets);
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

    #[test]
    fn state_machine_asset_round_trips_deterministically() {
        use crate::state_machine::{SmState, SmTransition, StateMachine};
        let machine = StateMachine {
            states: vec![
                SmState::clip("idle", [1; 16]),
                SmState::clip("walk", [2; 16]),
            ],
            transitions: vec![SmTransition {
                from: 0,
                to: 1,
                duration: 0.2,
                conditions: vec![crate::state_machine::SmCondition {
                    var: "moving".into(),
                    op: crate::state_machine::CmpOp::Gt,
                    value: 0.5,
                }],
                exit_time: Some(0.8),
            }],
            entry: 0,
        };
        let a = StateMachineAsset::new(machine, Some([9u8; 16]));
        let e1 = encode(&a).unwrap();
        let e2 = encode(&a).unwrap();
        assert_eq!(e1, e2, "re-encoding is byte-identical");
        let back: StateMachineAsset = decode(&e1).unwrap();
        assert_eq!(back, a);
        assert_eq!(back.skeleton, Some([9u8; 16]));
    }
}
