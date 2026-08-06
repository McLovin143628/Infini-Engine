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
use crate::template::JointLimit;

/// The `.inf_skel` payload: a skinning [`Skeleton`], its authored [`Socket`]s
/// (P11.3 — per-skeleton attach points) and its per-joint rotation
/// [`JointLimit`]s (P24.1 — the IK input P24.2 consumes).
///
/// # The v2 ladder
///
/// `limits` is a **side table appended at the tail**, not fields on [`Joint`],
/// and the reason is the engine's recurring bincode law: the codec is
/// **positional**, and `Joint` lives inside a `Vec` inside this payload — growing
/// it would re-interpret every byte after the first joint, in a container whose
/// length prefix is already committed. A tail append is the only additive shape
/// this format has, and even that is not free: `#[serde(default)]` cannot rescue
/// a bincode struct from a short read, so a v1 payload does **not** decode as a
/// v2 one. That is what `schema_version` is for, and it fails loudly (a decode
/// error) rather than quietly.
///
/// The one committed v1 `.inf_skel` in the tree (`samples/character-demo`) is
/// regenerated from its generator under `INF_BLESS_SAMPLES=1`.
///
/// **[`JointLimit`]'s fields are frozen** now that this has shipped — a later
/// limit *kind* is an append behind another bump, not an edit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkeletonAsset {
    pub schema_version: u32,
    pub skeleton: Skeleton,
    /// Named attach points riding the skeleton's joints (P11.3). Serde-clean (no
    /// `skip_serializing_if`) per the crate's bincode law.
    #[serde(default)]
    pub sockets: Vec<Socket>,
    /// Per-joint rotation limits for IK (P24.1, schema v2). A joint **absent**
    /// from this table is unlimited — see [`JointLimit`] for why that, and not a
    /// full-range row, is the meaningful default. Serde-clean, at the tail.
    #[serde(default)]
    pub limits: Vec<JointLimit>,
}

impl SkeletonAsset {
    /// v2 (P24.1) — `limits`. See the type's docs for the ladder.
    pub const CURRENT_VERSION: u32 = 2;

    /// Wrap a skeleton as a current-schema asset (no sockets, no limits).
    pub fn new(skeleton: Skeleton) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            skeleton,
            sockets: Vec::new(),
            limits: Vec::new(),
        }
    }

    /// Wrap a skeleton and its authored sockets.
    pub fn with_sockets(skeleton: Skeleton, sockets: Vec<Socket>) -> Self {
        Self {
            schema_version: Self::CURRENT_VERSION,
            skeleton,
            sockets,
            limits: Vec::new(),
        }
    }

    /// The rotation limit on `joint`, if one is authored.
    pub fn limit(&self, joint: u16) -> Option<&JointLimit> {
        self.limits.iter().find(|l| l.joint == joint)
    }
}

impl AssetPayload for SkeletonAsset {
    const KIND: AssetKind = AssetKind::Skeleton;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    // A rig has TWO doors, unlike most imported kinds, and a user reading this
    // message is usually looking at a project imported before P24.1.
    const UPGRADE_REMEDY: &'static str =
        "re-import the source model (Content Drawer ▸ Import), or generate a fresh          rig from a template (Content Drawer ▸ Add ▸ Skeleton)";
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
    use serde::Deserialize;

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

    /// v2: the limits side table rides the payload and round-trips.
    #[test]
    fn skeleton_asset_limits_round_trip() {
        use crate::template::JointLimit;
        let mut a = SkeletonAsset::new(skel());
        assert_eq!(a.schema_version, 2);
        assert!(a.limits.is_empty(), "an unlimited rig lists nothing");
        a.limits = vec![JointLimit::hinge_x(1, -150.0, 0.0)];
        let e1 = encode(&a).unwrap();
        assert_eq!(e1, encode(&a).unwrap(), "re-encoding is byte-identical");
        let back: SkeletonAsset = decode(&e1).unwrap();
        assert_eq!(back, a);
        assert_eq!(back.limit(1).unwrap().min_deg[0], -150.0);
        assert!(back.limit(0).is_none(), "an absent joint is unlimited");
    }

    /// The **v1 wire shape** of a `.inf_skel`, spelled out.
    ///
    /// A real shadow struct, not a byte trimmed off a v2 encoding. The trimming
    /// trick reproduced the right bytes and pinned nothing: it was *derived from
    /// the live encoder*, so appending another tail field without bumping the
    /// version would have kept it passing. This says what v1 was.
    #[derive(Serialize)]
    struct SkeletonAssetV1 {
        schema_version: u32,
        skeleton: Skeleton,
        sockets: Vec<Socket>,
    }

    /// The **v2 wire shape**, positionally — the twin
    /// `the_wire_shape_is_pinned_field_for_field` decodes a real encoding
    /// through.
    #[derive(Deserialize)]
    struct SkeletonAssetV2Wire {
        schema_version: u32,
        skeleton: Skeleton,
        sockets: Vec<Socket>,
        limits: Vec<JointLimit>,
    }

    fn v1_bytes() -> Vec<u8> {
        bincode::serde::encode_to_vec(
            &SkeletonAssetV1 {
                schema_version: 1,
                skeleton: skel(),
                sockets: vec![Socket::new("hand_r", 1)],
            },
            inf_asset::bincode_config(),
        )
        .unwrap()
    }

    /// **The v1 → v2 break is a NAMED refusal that says what to do.**
    ///
    /// bincode is positional and `#[serde(default)]` cannot rescue a short read,
    /// so a payload written before the `limits` tail cannot be read back. What
    /// matters is that it fails as [`AssetError::SchemaTooOld`] carrying this
    /// type's remedy, and not as `Decode("UnexpectedEnd")` — a user with a
    /// project imported before P24.1 sees this message and nothing else, and the
    /// P24.1 audit's blocker was precisely that the sim path threw it away.
    #[test]
    fn a_v1_payload_is_refused_by_name_and_names_the_remedy() {
        let err = decode::<SkeletonAsset>(&v1_bytes()).expect_err("v1 must not decode as v2");
        match err {
            inf_asset::AssetError::SchemaTooOld {
                kind,
                found,
                current,
                remedy,
            } => {
                assert_eq!(kind, "skeleton");
                assert_eq!((found, current), (1, SkeletonAsset::CURRENT_VERSION));
                // The remedy has to be an INSTRUCTION, not a restatement.
                assert!(remedy.contains("Import"), "{remedy}");
                assert!(remedy.contains("template"), "{remedy}");
            }
            other => panic!("expected SchemaTooOld, got {other:?}"),
        }
        // …and the rendered message carries it, because that string is what a
        // user actually reads in the Output Log.
        let msg = decode::<SkeletonAsset>(&v1_bytes())
            .unwrap_err()
            .to_string();
        assert!(msg.contains("re-import the source model"), "{msg}");
    }

    /// **The other direction.** A payload from a newer build decodes structurally
    /// and is rejected by `migrate`. Its three sibling `.inf_*` formats all pin
    /// this; the skeleton did not until the P24.1 audit.
    #[test]
    fn a_future_payload_is_refused_as_too_new() {
        let mut a = SkeletonAsset::new(skel());
        a.schema_version = SkeletonAsset::CURRENT_VERSION + 1;
        let bytes = encode(&a).unwrap();
        assert!(matches!(
            decode::<SkeletonAsset>(&bytes),
            Err(inf_asset::AssetError::SchemaTooNew { .. })
        ));
    }

    /// **The wire SHAPE is pinned, so a tail field cannot be appended without a
    /// bump.**
    ///
    /// Two claims, and the second is the load-bearing one:
    ///
    ///  * the four fields decode positionally, in this order, with these types;
    ///  * the decode consumes **every byte** of the encoding. A fifth field
    ///    appended to `SkeletonAsset` would leave bytes unconsumed here and fail
    ///    — which is exactly what the previous, encoder-derived fixture could not
    ///    see, because it was built by asking the encoder what it emitted.
    #[test]
    fn the_wire_shape_is_pinned_field_for_field() {
        let mut want = SkeletonAsset::with_sockets(skel(), vec![Socket::new("hand_r", 1)]);
        want.limits = vec![JointLimit::hinge_x(1, -150.0, 0.0)];
        let bytes = encode(&want).unwrap();

        let (wire, consumed): (SkeletonAssetV2Wire, usize) =
            bincode::serde::decode_from_slice(&bytes, inf_asset::bincode_config())
                .expect("the v2 shape decodes the v2 wire");
        assert_eq!(
            consumed,
            bytes.len(),
            "the encoding carries bytes the pinned four-field shape does not \
             account for — a field was appended to `SkeletonAsset` without \
             bumping `CURRENT_VERSION`"
        );
        assert_eq!(wire.schema_version, SkeletonAsset::CURRENT_VERSION);
        assert_eq!(wire.skeleton, want.skeleton);
        assert_eq!(wire.sockets, want.sockets);
        assert_eq!(wire.limits, want.limits);
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
