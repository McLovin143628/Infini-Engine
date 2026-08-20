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
        "re-import the source model (Content Drawer ▸ Import), or generate a fresh rig from a template (Content Drawer ▸ Add ▸ Skeleton)";
    fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Reject newer-than-current, then re-ask [`Skeleton::new`]'s own questions
    /// of the decoded rig (round-2 finding **B2**'s default-`migrate` sweep).
    ///
    /// `Skeleton::joints` is private and `Skeleton` derives `Deserialize`, and
    /// serde's derive is expanded in `skeleton.rs` itself — so it builds the
    /// struct straight from the wire and `Skeleton::new`'s validation never
    /// runs on the decode path. A `.inf_skel` naming a parent past the end of
    /// its own joint list decoded cleanly, and
    /// [`crate::pose::global_transforms`] then indexes `globals[p as usize]`
    /// with no bound: a panic on **every** posed character, in the editor, in
    /// PIE and in the shipped player reading a cooked pack. It is also the one
    /// asset whose in-range-but-wrong case is silent — a non-topological
    /// parent reads the identity slot ahead of it and produces a pose that is
    /// wrong rather than missing.
    fn migrate(self) -> inf_asset::Result<Self> {
        let found = self.schema_version;
        if found > Self::SCHEMA_VERSION {
            return Err(inf_asset::AssetError::SchemaTooNew {
                kind: Self::KIND.slug(),
                found,
                current: Self::SCHEMA_VERSION,
            });
        }
        self.skeleton
            .validate()
            .map_err(|e| inf_asset::AssetError::Decode(format!("invalid skeleton: {e}")))?;
        Ok(self)
    }
}

/// The `.inf_anim` payload: one [`AnimClip`] plus the GUID of the skeleton it
/// was authored against (stored as raw bytes so this crate needs no `uuid` dep;
/// the editor sets the dependency edge). `None` = skeleton-agnostic.
///
/// # The v2 ladder — the bump that only happens once
///
/// v2 (P29.2) grows the *clip*, not this wrapper: [`AnimClip`] gained the channel
/// model — named scalar curves, timed markers, an additive reference, and a
/// present-but-optional root-motion and distance track. §13's 2026-08-17
/// amendment (Ruling 2) is explicit that `.inf_anim` bumps **exactly once**, which
/// is why all five land together rather than as each sub-phase needs one: P29.4's
/// notify consumer, P29.5's import derivation and P29.6's showcase all read this
/// format, and a second bump would break every project imported in between.
///
/// bincode is positional and the five fields are a **tail append inside `clip`**,
/// so v1 bytes stop short and no `#[serde(default)]` rescues the read. That is
/// what `schema_version` is for, and because it is this type's **first** field
/// [`inf_asset::peek_schema_version`] reads it without decoding: the failure
/// arrives as [`AssetError::SchemaTooOld`](inf_asset::AssetError::SchemaTooOld)
/// carrying [`UPGRADE_REMEDY`](AssetPayload::UPGRADE_REMEDY) rather than as
/// `Decode("UnexpectedEnd")`.
///
/// The three committed v1 clips (`samples/character-demo/{Idle,Run,Jump}.inf_anim`)
/// are regenerated from their generator under `INF_BLESS_SAMPLES=1` — the
/// downgrade-bless — and the v1 wire shape they *used* to have is pinned below
/// from ladder-local literals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimClipAsset {
    pub schema_version: u32,
    pub clip: AnimClip,
    /// Raw 16-byte GUID of the skeleton this clip targets (a dependency edge the
    /// importer/editor also records in the sidecar). `None` = unbound.
    pub skeleton: Option<[u8; 16]>,
}

impl AnimClipAsset {
    /// v2 (P29.2) — the clip's channel model: curves, markers, additive
    /// reference, root motion, distance. See the type's docs for the ladder.
    pub const CURRENT_VERSION: u32 = 2;

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
    // A clip has TWO doors, like its sibling rig: it arrives by import, or it is
    // generated for a character. A user reading this message is looking at a
    // project imported before P29.2.
    const UPGRADE_REMEDY: &'static str =
        "re-import the source animation (Content Drawer ▸ Import), or regenerate the character through the New Character wizard (which derives its clips from the rig)";
    fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Reject newer-than-current (the default rule), then **validate the clip's
    /// timing** — the `BiomeSet::migrate` pattern, applied to the payload whose
    /// numbers the frame loop divides by.
    ///
    /// # Why a decode-time check and not only a guard (C4-4)
    ///
    /// `duration`, and every keyframe time inside `tracks`, are plain serde
    /// fields of a bincode payload with no structural check anywhere on the
    /// path. A NaN `duration` used to reach `t.clamp(0.0, duration)`, which
    /// **panics** (`assert!(min <= max)`) — every frame, in the editor, in PIE
    /// and in the shipped player — because the guard in front of it was
    /// `duration <= 0.0` and every ordering comparison a NaN takes part in is
    /// false. The guards are fixed too ([`crate::clip::resolve_time`],
    /// and `clip::locate`), but a guard that survives a poisoned value is
    /// not the same as refusing to hold one: the looping branch has no panic and
    /// instead writes `rem_euclid(NaN)` back into `AnimPlayer.t`, which is
    /// persisted into the `.inf_lvl`. That is NaN in committed bytes, and the
    /// place to stop it is the door.
    fn migrate(self) -> inf_asset::Result<Self> {
        let found = self.schema_version;
        if found > Self::SCHEMA_VERSION {
            return Err(inf_asset::AssetError::SchemaTooNew {
                kind: Self::KIND.slug(),
                found,
                current: Self::SCHEMA_VERSION,
            });
        }
        self.clip
            .validate_timing()
            .map_err(|e| inf_asset::AssetError::Decode(format!("invalid anim clip: {e}")))?;
        Ok(self)
    }
}

/// The `.inf_sm` payload: an authored animation [`StateMachine`] (P11.2 v1,
/// **P29.1 v2**).
///
/// The machine is a plain typed model (states + transitions + layout), not an
/// `inf-graph` document, so it stores directly here — no JSON-string escape hatch
/// (the PCG-style workaround that model needs for its `skip_serializing_if`
/// fields does not apply: [`StateMachine`] and everything it contains is
/// serde-clean). The optional `skeleton` GUID records the dependency edge (raw
/// bytes, keeping the crate `uuid`-free), like [`AnimClipAsset`].
///
/// # The v2 ladder, and why v1 cannot be upgraded in place
///
/// v2 (P29.1) did not *append* to this schema — it changed shapes **inside** it:
/// a transition's `conditions: Vec<SmCondition>` became a `condition: SmCond`
/// tree, its `from: usize` became an [`SmSource`](crate::state_machine::SmSource)
/// enum, and typed parameters and blend profiles joined the machine. bincode is
/// **positional**, so a v1 payload's bytes are not a prefix of a v2 one and no
/// `#[serde(default)]` can rescue the read. That is what `schema_version` is for,
/// and — because it is this type's **first** field —
/// [`inf_asset::peek_schema_version`] reads it without decoding and the failure
/// arrives as [`AssetError::SchemaTooOld`](inf_asset::AssetError::SchemaTooOld)
/// carrying [`UPGRADE_REMEDY`](AssetPayload::UPGRADE_REMEDY), rather than as
/// `Decode("UnexpectedEnd")`.
///
/// The remedy is a real instruction because a `.inf_sm` has two doors that both
/// write a current-schema machine: the State Machine editor, and the character
/// wizard (which derives one from the rig). The one committed v1 `.inf_sm` in the
/// tree (`samples/character-demo/Locomotion.inf_sm`) is regenerated from its
/// generator under `INF_BLESS_SAMPLES=1` — the downgrade-bless — and the v1 wire
/// shape it *used* to have is pinned below from ladder-local literals, so a
/// future field cannot be added without this ladder noticing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateMachineAsset {
    pub schema_version: u32,
    pub machine: StateMachine,
    /// Raw 16-byte GUID of the skeleton this machine animates. `None` = unbound.
    pub skeleton: Option<[u8; 16]>,
}

impl StateMachineAsset {
    /// v3 (the island phase) — the **per-transition blend mode**.
    ///
    /// * v1 → refused. See the type's docs: v2 changed shapes *inside* the
    ///   machine, so no honest in-place upgrade exists and the remedy is a door.
    /// * v2 → **migrated**, because this rung is the opposite kind: one
    ///   `Option<SmBlendMode>` appended to the tail of `SmTransition`, whose
    ///   default (`None`, "inherit the session default") is exactly what every v2
    ///   machine already meant. A v2 file therefore loses nothing and behaves
    ///   identically, which is what makes a migration honest here and dishonest
    ///   there.
    pub const CURRENT_VERSION: u32 = 3;

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
    // Two doors, and a user reading this message is looking at a project authored
    // before P29.1.
    const UPGRADE_REMEDY: &'static str =
        "re-create the machine in the State Machine editor, or regenerate the character through the New Character wizard (which derives one from the rig)";
    fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Reject newer-than-current, then ask the decoded machine
    /// [`StateMachine::validate`]'s questions — the campaign's U6 standard, and
    /// for this payload it is not a formality.
    ///
    /// `StateMachine`'s fields are public and it derives `Deserialize`, so serde
    /// builds it straight off the wire and no constructor's validation runs on
    /// the decode path. Everything the evaluator would then have to trust is a
    /// plain decoded number: a transition's `to` indexes `sm.states`, a
    /// `profile` indexes `sm.profiles`, an `entry` seeds the runtime, a
    /// `duration` divides the cross-fade alpha, and a condition tree is walked
    /// **recursively** in the fixed step — in the editor, in PIE and in the
    /// shipped player reading a cooked pack. A NaN duration or exit time is the
    /// quiet one: every ordering comparison a NaN takes part in is false, so the
    /// machine simply never transitions and looks like content that was authored
    /// wrong.
    fn migrate(mut self) -> inf_asset::Result<Self> {
        let found = self.schema_version;
        if found > Self::SCHEMA_VERSION {
            return Err(inf_asset::AssetError::SchemaTooNew {
                kind: Self::KIND.slug(),
                found,
                current: Self::SCHEMA_VERSION,
            });
        }
        // A v2 payload arrives here already lifted by `decode_wire` (the shape
        // door); the stamp is what says so. Restamping here rather than there
        // keeps `decode_wire` about BYTES and this about the value.
        self.schema_version = Self::SCHEMA_VERSION;
        self.machine
            .validate()
            .map_err(|e| inf_asset::AssetError::Decode(format!("invalid state machine: {e}")))?;
        Ok(self)
    }

    /// **The v2 → v3 rung.** v3 appended `blend` to the tail of `SmTransition`,
    /// which sits inside a `Vec` in the middle of the stream — so a v2 payload is
    /// **not a prefix** of a v3 one and the current shape cannot read it. The
    /// frozen [`v2::StateMachineAsset`] can, and lifting it is a pure default-fill.
    ///
    /// v1 is deliberately absent: it falls through to the current shape, fails,
    /// and `decode` turns that into `SchemaTooOld` with this type's own remedy —
    /// the behaviour P29.1 chose and this rung does not change.
    /// v2 has a rung; v1 does not. So a v1 payload keeps its `SchemaTooOld`
    /// remedy, and a v2 one that fails for a structural reason reports **that**
    /// reason rather than being told it is old.
    fn migrates_from(v: u32) -> bool {
        v == 2
    }

    fn decode_wire(bytes: &[u8], found: Option<u32>) -> inf_asset::Result<Self> {
        if found == Some(2) {
            let old: v2::StateMachineAsset = inf_asset::decode_shape(bytes)
                .map_err(|e| inf_asset::AssetError::Decode(format!("v2 state machine: {e}")))?;
            return Ok(old.into_current());
        }
        inf_asset::decode_shape(bytes)
    }
}

/// **The frozen schema-v2 `.inf_sm` record** — the shape before the island phase
/// appended `SmTransition::blend`.
///
/// Ladder-local and declared field-for-field rather than derived from the live
/// types, which is the whole point: if `SmTransition` grows again without a bump,
/// this shape stops matching the bytes the old encoder wrote and
/// `the_frozen_v2_record_is_the_shape_v2_actually_wrote` goes red. A frozen record
/// assembled *from* the live one asserts nothing.
///
/// # Four types are re-declared, and the fourth is the one that was nearly missed
///
/// `SmTransition` obviously. But a state's `Motion::SubMachine` nests a whole
/// `StateMachine`, whose transitions are `SmTransition`s **too** — so a frozen
/// record that reached `SmState` through the *live* declaration would write the
/// v3 shape for every nested transition while claiming to be v2. It did, and
/// `v3_costs_one_discriminant_per_transition` caught it: the fixture's three
/// transitions cost **two** bytes, because the sub-machine's one was already
/// paying on both sides of the comparison. Hence `v2::SmState`, `v2::Motion` and
/// the recursion back into `v2::StateMachine`.
///
/// Everything else inside the machine is byte-unchanged by v3 and is reached
/// through the live declarations — the same economy `inf_scene`'s
/// `EntityRecordV20Gen` uses, and the same limit: the day one of them changes, it
/// gets frozen here too.
pub(crate) mod v2 {
    use serde::{Deserialize, Serialize};

    use crate::blend_space::{BlendSpace1D, BlendSpace2D, ClipRef};
    use crate::state_machine::{BlendCurve, BlendProfile, SmCond, SmInterrupt, SmParam, SmSource};

    /// A v2 state. Identical to the live one except that its `motion` reaches
    /// [`Motion`] — the v2 sub-machine — rather than the live one.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SmState {
        pub name: String,
        pub motion: Motion,
        pub looping: bool,
        pub speed: f64,
        pub position: (f32, f32),
        #[serde(default)]
        pub on_enter: Vec<String>,
        #[serde(default)]
        pub on_exit: Vec<String>,
    }

    /// A v2 motion. The `SubMachine` arm is the whole reason this exists.
    ///
    /// # It carries the SAME depth guard the live `Motion` does
    ///
    /// The first draft of this record did not, on the reasoning that these bytes
    /// "have already survived the live guard". They have not: `decode_wire`
    /// dispatches a v2 payload **straight here**, so the live `de_sub_machine`
    /// never runs on this path and a crafted `.inf_sm` with a deep sub-machine
    /// chain would have reached the bottom of the stack before `validate` saw the
    /// tree. A version ladder decodes old bytes through a *different*
    /// declaration, which means every hostile-input guard on the live shape has to
    /// be re-entered on the frozen one — the guard is shared
    /// (`state_machine::sub_machine_depth_guard`), not restated.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Motion {
        Clip(ClipRef),
        Blend1D(BlendSpace1D),
        Blend2D(BlendSpace2D),
        SubMachine(#[serde(deserialize_with = "de_sub_machine_v2")] Box<StateMachine>),
    }

    fn de_sub_machine_v2<'de, D>(d: D) -> Result<Box<StateMachine>, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let _guard = crate::state_machine::sub_machine_depth_guard::<D::Error>()?;
        Ok(Box::new(StateMachine::deserialize(d)?))
    }

    /// A v2 transition: everything the live one has except the v3 tail.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SmTransition {
        pub from: SmSource,
        pub to: usize,
        pub duration: f64,
        pub condition: SmCond,
        pub exit_time: Option<f64>,
        #[serde(default)]
        pub priority: i32,
        #[serde(default)]
        pub interrupt: SmInterrupt,
        #[serde(default)]
        pub curve: BlendCurve,
        #[serde(default)]
        pub profile: Option<usize>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct StateMachine {
        pub states: Vec<SmState>,
        pub transitions: Vec<SmTransition>,
        pub entry: usize,
        #[serde(default)]
        pub params: Vec<SmParam>,
        #[serde(default)]
        pub profiles: Vec<BlendProfile>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct StateMachineAsset {
        pub schema_version: u32,
        pub machine: StateMachine,
        pub skeleton: Option<[u8; 16]>,
    }

    impl SmTransition {
        /// Lift one transition: the v3 tail arrives **absent**, which is what a v2
        /// machine meant — "blend the way this session blends".
        fn into_current(self) -> crate::state_machine::SmTransition {
            crate::state_machine::SmTransition {
                from: self.from,
                to: self.to,
                duration: self.duration,
                condition: self.condition,
                exit_time: self.exit_time,
                priority: self.priority,
                interrupt: self.interrupt,
                curve: self.curve,
                profile: self.profile,
                blend: None,
            }
        }

        /// Project a live transition back onto the v2 shape — the
        /// **downgrade-bless** direction, which is what regenerates the committed
        /// v2 fixture from ladder-local literals rather than from a stale file.
        #[cfg_attr(not(test), allow(dead_code))]
        pub fn from_current(t: crate::state_machine::SmTransition) -> Self {
            Self {
                from: t.from,
                to: t.to,
                duration: t.duration,
                condition: t.condition,
                exit_time: t.exit_time,
                priority: t.priority,
                interrupt: t.interrupt,
                curve: t.curve,
                profile: t.profile,
            }
        }
    }

    impl SmState {
        fn into_current(self) -> crate::state_machine::SmState {
            crate::state_machine::SmState {
                name: self.name,
                motion: match self.motion {
                    Motion::Clip(c) => crate::state_machine::Motion::Clip(c),
                    Motion::Blend1D(b) => crate::state_machine::Motion::Blend1D(b),
                    Motion::Blend2D(b) => crate::state_machine::Motion::Blend2D(b),
                    Motion::SubMachine(m) => {
                        crate::state_machine::Motion::SubMachine(Box::new(m.into_current()))
                    }
                },
                looping: self.looping,
                speed: self.speed,
                position: self.position,
                on_enter: self.on_enter,
                on_exit: self.on_exit,
            }
        }

        #[cfg_attr(not(test), allow(dead_code))]
        fn from_current(s: crate::state_machine::SmState) -> Self {
            Self {
                name: s.name,
                motion: match s.motion {
                    crate::state_machine::Motion::Clip(c) => Motion::Clip(c),
                    crate::state_machine::Motion::Blend1D(b) => Motion::Blend1D(b),
                    crate::state_machine::Motion::Blend2D(b) => Motion::Blend2D(b),
                    crate::state_machine::Motion::SubMachine(m) => {
                        Motion::SubMachine(Box::new(StateMachine::from_current(*m)))
                    }
                },
                looping: s.looping,
                speed: s.speed,
                position: s.position,
                on_enter: s.on_enter,
                on_exit: s.on_exit,
            }
        }
    }

    impl StateMachine {
        fn into_current(self) -> crate::state_machine::StateMachine {
            crate::state_machine::StateMachine {
                states: self.states.into_iter().map(SmState::into_current).collect(),
                transitions: self
                    .transitions
                    .into_iter()
                    .map(SmTransition::into_current)
                    .collect(),
                entry: self.entry,
                params: self.params,
                profiles: self.profiles,
            }
        }

        #[cfg_attr(not(test), allow(dead_code))]
        fn from_current(m: crate::state_machine::StateMachine) -> Self {
            Self {
                states: m.states.into_iter().map(SmState::from_current).collect(),
                transitions: m
                    .transitions
                    .into_iter()
                    .map(SmTransition::from_current)
                    .collect(),
                entry: m.entry,
                params: m.params,
                profiles: m.profiles,
            }
        }
    }

    impl StateMachineAsset {
        pub fn into_current(self) -> super::StateMachineAsset {
            super::StateMachineAsset {
                schema_version: self.schema_version,
                machine: self.machine.into_current(),
                skeleton: self.skeleton,
            }
        }

        /// The downgrade: a current asset as the bytes v2 would have written.
        #[cfg_attr(not(test), allow(dead_code))]
        pub fn from_current(a: &super::StateMachineAsset) -> Self {
            Self {
                schema_version: 2,
                machine: StateMachine::from_current(a.machine.clone()),
                skeleton: a.skeleton,
            }
        }
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
                // …and it has to be READABLE. A run of spaces inside a
                // user-facing literal is the scripted-edit law's signature: a
                // `\` line continuation eaten by a non-raw Python string, which
                // has now cost this repo ten mangled messages across two phases.
                // The check is one line and this text is the whole point of B1.
                assert!(
                    !remedy.contains("  "),
                    "the remedy carries a run of spaces — a line continuation was \
                     eaten: {remedy:?}"
                );
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
        let a = AnimClipAsset::new(v2_clip(), Some([7u8; 16]));
        assert_eq!(a.schema_version, 2);
        let e1 = encode(&a).unwrap();
        let e2 = encode(&a).unwrap();
        assert_eq!(e1, e2);
        let back: AnimClipAsset = decode(&e1).unwrap();
        assert_eq!(back, a, "every v2 shape survived the wire");
        assert_eq!(back.skeleton, Some([7u8; 16]));
    }

    /// A **v2** clip that exercises every shape the schema grew, so the round-trip
    /// above is a statement about v2 and not about v1 re-encoded.
    fn v2_clip() -> AnimClip {
        use crate::channels::{
            als, AdditiveRef, AnimMarker, CurveChannel, DistanceTrack, RootMotionTrack,
        };
        let mut jt = JointTrack::new(1);
        jt.rotation = Some(QuatTrack::new(
            vec![0.0, 1.0],
            vec![
                Quat::IDENTITY.to_array(),
                Quat::from_rotation_z(1.0).to_array(),
            ],
            Interpolation::Linear,
        ));
        AnimClip::new("spin", vec![jt])
            .with_curves(vec![
                CurveChannel::new(
                    als::W_GAIT,
                    vec![0.0, 1.0],
                    vec![1.0, 2.0],
                    Interpolation::Linear,
                ),
                CurveChannel::constant(als::ENABLE_FOOT_IK_L, 1.0),
            ])
            .with_markers(vec![
                AnimMarker::sync(0.0, "plant_l", "foot"),
                AnimMarker::sync(0.5, "plant_r", "foot"),
                AnimMarker::event(0.5, "footstep_r"),
            ])
            .with_additive(AdditiveRef::BaseClip {
                clip: [3; 16],
                time_s: 0.25,
            })
            .with_root_motion(RootMotionTrack {
                times: vec![0.0, 1.0],
                translation: vec![[0.0, 0.0, 0.0], [0.0, 0.4, 1.5]],
                yaw_rad: vec![0.0, 0.25],
            })
            .with_distance(DistanceTrack {
                times: vec![0.0, 1.0],
                distance_m: vec![0.0, 1.5],
            })
    }

    /// The **v1 wire shape** of an `.inf_anim`, spelled out from ladder-local
    /// literals.
    ///
    /// Real shadow structs, not a v2 encoding with its tail trimmed off: trimming
    /// is *derived from the live encoder*, so it reproduces whatever the encoder
    /// currently does and pins nothing (the `.inf_skel` ladder's own finding). This
    /// says what v1 **was** — a clip of `{name, duration, tracks}` and nothing else.
    mod anim_v1 {
        use super::{JointTrack, Serialize};

        #[derive(Serialize)]
        pub struct AnimClip {
            pub name: String,
            pub duration: f32,
            pub tracks: Vec<JointTrack>,
        }
        #[derive(Serialize)]
        pub struct AnimClipAsset {
            pub schema_version: u32,
            pub clip: AnimClip,
            pub skeleton: Option<[u8; 16]>,
        }
    }

    fn v1_anim_bytes() -> Vec<u8> {
        let mut jt = JointTrack::new(1);
        jt.rotation = Some(QuatTrack::new(
            vec![0.0, 1.0],
            vec![
                Quat::IDENTITY.to_array(),
                Quat::from_rotation_z(1.0).to_array(),
            ],
            Interpolation::Linear,
        ));
        bincode::serde::encode_to_vec(
            &anim_v1::AnimClipAsset {
                schema_version: 1,
                clip: anim_v1::AnimClip {
                    name: "spin".into(),
                    duration: 1.0,
                    tracks: vec![jt],
                },
                skeleton: Some([7u8; 16]),
            },
            inf_asset::bincode_config(),
        )
        .unwrap()
    }

    /// **The v1 → v2 break is a NAMED refusal that says what to do.**
    ///
    /// The five v2 fields are a tail append *inside* `clip`, so a v1 payload stops
    /// short and `#[serde(default)]` cannot rescue it. What matters is that a user
    /// with a project imported before P29.2 sees `SchemaTooOld` carrying this
    /// type's remedy — the `.inf_skel` / `.inf_sm` ladder ruling, applied to the
    /// third format that has two authoring doors.
    #[test]
    fn a_v1_anim_clip_is_refused_by_name_and_names_the_remedy() {
        let err = decode::<AnimClipAsset>(&v1_anim_bytes()).expect_err("v1 must not decode as v2");
        match err {
            inf_asset::AssetError::SchemaTooOld {
                kind,
                found,
                current,
                remedy,
            } => {
                assert_eq!(kind, "anim_clip");
                assert_eq!((found, current), (1, AnimClipAsset::CURRENT_VERSION));
                // An INSTRUCTION, not a restatement — and it names both doors.
                assert!(remedy.contains("Import"), "{remedy}");
                assert!(remedy.contains("wizard"), "{remedy}");
                // …and it is READABLE: a run of spaces inside a user-facing literal
                // is the scripted-edit law's signature (a `\` line continuation
                // eaten by a non-raw Python string).
                assert!(
                    !remedy.contains("  "),
                    "the remedy carries a run of spaces — a line continuation was \
                     eaten: {remedy:?}"
                );
            }
            other => panic!("expected SchemaTooOld, got {other:?}"),
        }
        let msg = decode::<AnimClipAsset>(&v1_anim_bytes())
            .unwrap_err()
            .to_string();
        assert!(msg.contains("re-import the source animation"), "{msg}");
    }

    /// The other direction: a payload from a newer build decodes structurally and
    /// is rejected by `migrate`.
    #[test]
    fn a_future_anim_clip_is_refused_as_too_new() {
        let mut a = AnimClipAsset::new(v2_clip(), None);
        a.schema_version = AnimClipAsset::CURRENT_VERSION + 1;
        let bytes = encode(&a).unwrap();
        assert!(matches!(
            decode::<AnimClipAsset>(&bytes),
            Err(inf_asset::AssetError::SchemaTooNew { .. })
        ));
    }

    /// **The v2 wire SHAPE is pinned, so a tail field cannot be appended without a
    /// bump.**
    ///
    /// Two claims, and the second is the load-bearing one: the fields decode
    /// positionally, in this order, with these types — **and** the decode consumes
    /// every byte of the encoding. A sixth v2 field appended to `AnimClip` (or a
    /// fourth to the asset) leaves bytes unaccounted for here and fails, which is
    /// exactly the check an encoder-derived fixture cannot make.
    ///
    /// The clip is spelled out field-for-field rather than reusing `AnimClip`,
    /// because a shape that *is* the type under test agrees with it by
    /// construction and pins nothing at all.
    #[test]
    fn the_anim_clip_wire_shape_is_pinned_field_for_field() {
        use crate::channels::{
            AdditiveRef, AnimMarker, CurveChannel, DistanceTrack, RootMotionTrack,
        };

        #[derive(Deserialize)]
        struct ClipWire {
            name: String,
            duration: f32,
            tracks: Vec<JointTrack>,
            curves: Vec<CurveChannel>,
            markers: Vec<AnimMarker>,
            additive: AdditiveRef,
            root_motion: Option<RootMotionTrack>,
            distance: Option<DistanceTrack>,
        }
        #[derive(Deserialize)]
        struct Wire {
            schema_version: u32,
            clip: ClipWire,
            skeleton: Option<[u8; 16]>,
        }

        let want = AnimClipAsset::new(v2_clip(), Some([7u8; 16]));
        let bytes = encode(&want).unwrap();
        let (wire, consumed): (Wire, usize) =
            bincode::serde::decode_from_slice(&bytes, inf_asset::bincode_config())
                .expect("the v2 shape decodes the v2 wire");
        assert_eq!(
            consumed,
            bytes.len(),
            "the encoding carries bytes the pinned shape does not account for — a \
             field was appended to `AnimClip` or `AnimClipAsset` without bumping \
             `CURRENT_VERSION`"
        );
        assert_eq!(wire.schema_version, AnimClipAsset::CURRENT_VERSION);
        assert_eq!(wire.skeleton, want.skeleton);
        assert_eq!(wire.clip.name, want.clip.name);
        assert_eq!(wire.clip.duration, want.clip.duration);
        assert_eq!(wire.clip.tracks, want.clip.tracks);
        assert_eq!(wire.clip.curves, want.clip.curves);
        assert_eq!(wire.clip.markers, want.clip.markers);
        assert_eq!(wire.clip.additive, want.clip.additive);
        assert_eq!(wire.clip.root_motion, want.clip.root_motion);
        assert_eq!(wire.clip.distance, want.clip.distance);
    }

    /// **`migrate` asks the STRUCTURAL questions** (the campaign's U6 standard),
    /// and for this payload every one of them is a value the frame loop would
    /// otherwise index with or divide by.
    ///
    /// The hostile clips are built through the public fields and encoded — which
    /// is how a corrupt or hand-edited `.inf_anim` arrives.
    #[test]
    fn a_structurally_broken_v2_clip_is_refused_at_decode() {
        use crate::channels::{AdditiveRef, AnimMarker, CurveChannel, DistanceTrack};

        // Control: the full v2 clip decodes.
        let bytes = encode(&AnimClipAsset::new(v2_clip(), None)).unwrap();
        decode::<AnimClipAsset>(&bytes).expect("a healthy v2 clip must decode");

        let refused = |name: &str, f: &dyn Fn(&mut AnimClip)| -> String {
            let mut c = v2_clip();
            f(&mut c);
            let bytes = encode(&AnimClipAsset::new(c, None)).unwrap();
            let e = decode::<AnimClipAsset>(&bytes)
                .err()
                .unwrap_or_else(|| panic!("{name} decoded as a valid clip"));
            let s = e.to_string();
            assert!(s.contains("invalid anim clip"), "{name}: {s}");
            s
        };

        // A curve whose parallel vectors disagree — `sample` indexes `values` with
        // an index derived from `times`.
        let e = refused("a short curve", &|c| {
            c.curves[0].values.pop();
        });
        assert!(e.contains("2 key times against 1"), "{e}");
        // A NaN key time, in each of the three new keyframe lists.
        refused("a NaN curve key", &|c| c.curves[0].times[1] = f32::NAN);
        refused("a NaN root-motion key", &|c| {
            c.root_motion.as_mut().unwrap().times[1] = f32::NAN
        });
        refused("a NaN distance key", &|c| {
            c.distance.as_mut().unwrap().times[1] = f32::NAN
        });
        // A NaN marker time — `markers_in` sorts on it and `sync` then trusts the
        // order.
        refused("a NaN marker time", &|c| {
            c.markers.push(AnimMarker::sync(f32::NAN, "x", "foot"))
        });
        // Root motion whose three parallel vectors disagree.
        refused("a short root-motion yaw list", &|c| {
            c.root_motion.as_mut().unwrap().yaw_rad.pop();
        });
        // Distance that goes backwards — `time_at_distance` `partition_point`s it.
        let e = refused("a non-monotone distance track", &|c| {
            c.distance = Some(DistanceTrack {
                times: vec![0.0, 1.0, 2.0],
                distance_m: vec![0.0, 2.0, 1.0],
            })
        });
        assert!(e.contains("goes backwards"), "{e}");
        // …and its NaN twin, which the positive spelling of that comparison would
        // have let straight through.
        refused("a NaN distance", &|c| {
            c.distance = Some(DistanceTrack {
                times: vec![0.0, 1.0],
                distance_m: vec![0.0, f32::NAN],
            })
        });
        // A RESERVED additive slot: a newer build wrote it, this one decodes it
        // (that is what the slot is for) and refuses it by name.
        let e = refused("a reserved additive reference", &|c| {
            c.additive = AdditiveRef::Reserved3 {
                clip: [1; 16],
                time_s: 0.0,
            }
        });
        assert!(e.contains("newer engine"), "{e}");
        // The control's mirror: an empty curve list is not a defect.
        let mut plain = v2_clip();
        plain.curves = vec![CurveChannel::constant("k", 1.0)];
        decode::<AnimClipAsset>(&encode(&AnimClipAsset::new(plain, None)).unwrap())
            .expect("a one-key constant curve is ordinary content");
    }

    /// A v2 machine that exercises **every** shape the schema grew, so the
    /// round-trip below is a statement about v2 and not about v1 re-encoded.
    fn v2_machine() -> StateMachine {
        use crate::state_machine::{
            BlendCurve, BlendProfile, CmpOp, InterruptBlend, InterruptSource, JointBlendWeight,
            SmCond, SmInterrupt, SmParam, SmParamKind, SmState, SmTransition,
        };
        let sub = StateMachine {
            states: vec![
                SmState::clip("stand", [3; 16]),
                SmState::clip("crouch", [4; 16]),
            ],
            transitions: vec![SmTransition::on(0, 1, 0.1, "crouching", CmpOp::Gt, 0.5)],
            entry: 0,
            ..Default::default()
        };
        StateMachine {
            states: vec![
                SmState::clip("idle", [1; 16]),
                SmState::clip("walk", [2; 16]),
                SmState::sub_machine("ground", sub),
            ],
            transitions: vec![
                SmTransition::on(0, 1, 0.2, "moving", CmpOp::Gt, 0.5)
                    .with_exit_time(0.8)
                    .with_priority(3)
                    .with_curve(BlendCurve::EaseInOut)
                    .with_profile(0)
                    .with_interrupt(SmInterrupt {
                        source: InterruptSource::SourceOrDestination,
                        blend: InterruptBlend::Carry,
                    }),
                SmTransition::any(2, 0.05).when(SmCond::Or(vec![
                    SmCond::Trigger("land".into()),
                    SmCond::Not(Box::new(SmCond::bool("airborne", true))),
                ])),
            ],
            entry: 0,
            params: vec![
                SmParam::float("moving"),
                SmParam::trigger("land"),
                SmParam::new("airborne", SmParamKind::Bool),
                SmParam::new("stance", SmParamKind::Int),
            ],
            profiles: vec![BlendProfile::new(
                "upper-body",
                vec![JointBlendWeight {
                    joint: 1,
                    weight: 0.25,
                }],
            )],
        }
    }

    #[test]
    fn state_machine_asset_round_trips_deterministically() {
        let a = StateMachineAsset::new(v2_machine(), Some([9u8; 16]));
        assert_eq!(a.schema_version, 3);
        let e1 = encode(&a).unwrap();
        let e2 = encode(&a).unwrap();
        assert_eq!(e1, e2, "re-encoding is byte-identical");
        let back: StateMachineAsset = decode(&e1).unwrap();
        assert_eq!(back, a, "every v2 shape survived the wire");
        assert_eq!(back.skeleton, Some([9u8; 16]));
    }

    /// The **v1 wire shape** of a `.inf_sm`, spelled out from ladder-local
    /// literals.
    ///
    /// Real shadow structs, not a v2 encoding with bytes trimmed off: the
    /// trimming trick is *derived from the live encoder*, so it reproduces
    /// whatever the encoder currently does and pins nothing. This says what v1
    /// **was** — a flat `Vec<SmCondition>` of `var`/`op`/`value`, a `usize`
    /// source, and a machine with no parameter table and no profiles.
    mod v1 {
        use serde::Serialize;

        #[derive(Serialize)]
        pub struct SmCondition {
            pub var: String,
            pub op: u32,
            pub value: f64,
        }
        #[derive(Serialize)]
        pub enum Motion {
            Clip([u8; 16]),
        }
        #[derive(Serialize)]
        pub struct SmState {
            pub name: String,
            pub motion: Motion,
            pub looping: bool,
            pub speed: f64,
            pub position: (f32, f32),
        }
        #[derive(Serialize)]
        pub struct SmTransition {
            pub from: usize,
            pub to: usize,
            pub duration: f64,
            pub conditions: Vec<SmCondition>,
            pub exit_time: Option<f64>,
        }
        #[derive(Serialize)]
        pub struct StateMachine {
            pub states: Vec<SmState>,
            pub transitions: Vec<SmTransition>,
            pub entry: usize,
        }
        #[derive(Serialize)]
        pub struct StateMachineAsset {
            pub schema_version: u32,
            pub machine: StateMachine,
            pub skeleton: Option<[u8; 16]>,
        }
    }

    fn v1_sm_bytes() -> Vec<u8> {
        bincode::serde::encode_to_vec(
            &v1::StateMachineAsset {
                schema_version: 1,
                machine: v1::StateMachine {
                    states: vec![v1::SmState {
                        name: "idle".into(),
                        motion: v1::Motion::Clip([1; 16]),
                        looping: true,
                        speed: 1.0,
                        position: (0.0, 0.0),
                    }],
                    transitions: vec![v1::SmTransition {
                        from: 0,
                        to: 0,
                        duration: 0.2,
                        conditions: vec![v1::SmCondition {
                            var: "moving".into(),
                            op: 0,
                            value: 0.5,
                        }],
                        exit_time: None,
                    }],
                    entry: 0,
                },
                skeleton: Some([9u8; 16]),
            },
            inf_asset::bincode_config(),
        )
        .unwrap()
    }

    // ── the v2 → v3 rung (the island phase: per-transition blend) ───────────

    /// The exact bytes a **v2** encoder wrote for [`v2_machine`], built from the
    /// frozen record rather than from the live one — the downgrade-bless.
    fn v2_sm_bytes() -> Vec<u8> {
        let live = StateMachineAsset::new(v2_machine(), Some([9u8; 16]));
        inf_asset::encode_shape(&v2::StateMachineAsset::from_current(&live)).unwrap()
    }

    /// **A v2 `.inf_sm` still loads, and every transition inherits.**
    ///
    /// The rung's whole claim: a machine authored before the island phase behaves
    /// exactly as it did, because `None` means "blend the way this session
    /// blends" and that is what a v2 machine always meant.
    #[test]
    fn a_v2_state_machine_migrates_and_every_transition_inherits() {
        let bytes = v2_sm_bytes();
        assert_eq!(
            inf_asset::peek_schema_version(&bytes, Some(StateMachineAsset::CURRENT_VERSION)),
            Some(2),
            "the fixture is not a genuine v2 payload"
        );
        let a: StateMachineAsset = decode(&bytes).expect("a v2 machine must migrate");
        assert_eq!(a.schema_version, StateMachineAsset::CURRENT_VERSION);
        assert_eq!(a.skeleton, Some([9u8; 16]));

        // Everything v2 could express survived…
        let want = v2_machine();
        assert_eq!(a.machine.states, want.states);
        assert_eq!(a.machine.params, want.params);
        assert_eq!(a.machine.profiles, want.profiles);
        assert_eq!(a.machine.entry, want.entry);
        assert_eq!(a.machine.transitions.len(), want.transitions.len());
        for (got, w) in a.machine.transitions.iter().zip(want.transitions.iter()) {
            assert!(got.blend.is_none(), "a v2 transition named a blend mode");
            // …field for field, compared through a patched clone rather than a
            // field list, so a slot added later cannot quietly stop being checked.
            let mut patched = got.clone();
            patched.blend = w.blend;
            assert_eq!(&patched, w, "the v2 lift lost something other than `blend`");
        }
        // ANTI-VACUITY: the fixture really exercises the sub-machine arm.
        assert!(a
            .machine
            .states
            .iter()
            .any(|s| matches!(s.motion, crate::state_machine::Motion::SubMachine(_))));
    }

    /// **The frozen v2 record is the shape v2 ACTUALLY wrote** — the ladder's
    /// forcing function.
    ///
    /// If `SmTransition` grows again without a bump, the live encoder writes a
    /// longer transition than this frozen shape describes and the byte-consumption
    /// check below fails. A frozen record assembled *from* the live one could
    /// never notice, which is why `v2::SmTransition` is declared field-for-field.
    #[test]
    fn the_frozen_v2_record_is_the_shape_v2_actually_wrote() {
        let bytes = v2_sm_bytes();
        // It decodes THROUGH the frozen shape, consuming every byte.
        let (_, consumed) = bincode::serde::decode_from_slice::<v2::StateMachineAsset, _>(
            &bytes,
            inf_asset::bincode_config(),
        )
        .expect("the frozen v2 shape reads its own bytes");
        assert_eq!(
            consumed,
            bytes.len(),
            "the frozen v2 record leaves {} byte(s) unread — the shape and the \
             encoder disagree",
            bytes.len() - consumed
        );
        // …and the CURRENT shape cannot, because v3 appended inside a Vec in the
        // middle of the stream. That is the whole reason this rung needs a
        // frozen record rather than a `#[serde(default)]`.
        assert!(
            bincode::serde::decode_from_slice::<StateMachineAsset, _>(
                &bytes,
                inf_asset::bincode_config()
            )
            .is_err(),
            "a v2 payload decoded at the v3 shape — then the ladder is measuring \
             nothing and the migration is dead code"
        );
    }

    /// v3 costs a v2 machine **one byte per transition** — the `None`
    /// discriminant, the price every additive tail has paid since v8 of the scene.
    #[test]
    fn v3_costs_one_discriminant_per_transition() {
        let live = StateMachineAsset::new(v2_machine(), Some([9u8; 16]));
        let v3 = inf_asset::encode(&live).unwrap();
        let v2b = v2_sm_bytes();
        // The sub-machine's transitions pay it too — they are `SmTransition`s.
        let total = live.machine.transitions.len()
            + live
                .machine
                .states
                .iter()
                .filter_map(|s| match &s.motion {
                    crate::state_machine::Motion::SubMachine(inner) => {
                        Some(inner.transitions.len())
                    }
                    _ => None,
                })
                .sum::<usize>();
        assert!(total > 1, "the fixture has {total} transitions");
        assert_eq!(
            v3.len(),
            v2b.len() + total,
            "v3 costs {} bytes over {total} transitions, not {total}",
            v3.len() - v2b.len()
        );
    }

    /// An **authored** mode really costs more, and round-trips — so the arm above
    /// is about an absent one rather than about a field that never encodes.
    #[test]
    fn an_authored_blend_mode_round_trips_and_is_not_free() {
        let mut m = v2_machine();
        m.transitions[0].blend = Some(crate::SmBlendMode::CrossFade);
        let a = StateMachineAsset::new(m, None);
        let bytes = inf_asset::encode(&a).unwrap();
        let back: StateMachineAsset = decode(&bytes).unwrap();
        assert_eq!(
            back.machine.transitions[0].blend,
            Some(crate::SmBlendMode::CrossFade)
        );
        assert!(
            back.machine.transitions[1..]
                .iter()
                .all(|t| t.blend.is_none()),
            "a codec that wrote one value into every slot would pass the line above"
        );
        let bare = inf_asset::encode(&StateMachineAsset::new(v2_machine(), None)).unwrap();
        assert!(
            bytes.len() > bare.len(),
            "an authored mode encoded to nothing"
        );
    }

    /// **The wire discriminants are frozen** — `SmBlendMode` is content now.
    ///
    /// bincode writes an externally-tagged enum as its variant index, so
    /// inserting a variant anywhere but the end re-reads every committed
    /// machine's transitions as a different mode. Pinned against literals, not
    /// against the enum, because a pin that asks the enum what it is asserts
    /// nothing.
    #[test]
    fn the_blend_mode_wire_discriminants_are_frozen() {
        for (mode, index) in [
            (crate::SmBlendMode::Inertialize, 0u32),
            (crate::SmBlendMode::CrossFade, 1u32),
        ] {
            let got = inf_asset::encode_shape(&mode).unwrap();
            let want = inf_asset::encode_shape(&index).unwrap();
            assert_eq!(
                got, want,
                "{mode:?} no longer encodes as variant index {index} — every \
                 committed .inf_sm just changed meaning"
            );
        }
        // …and `Option<SmBlendMode>`'s absent case is the single `0` byte the
        // cost arms above count.
        assert_eq!(
            inf_asset::encode_shape(&Option::<crate::SmBlendMode>::None)
                .unwrap()
                .len(),
            1
        );
    }

    /// **A pathologically nested V2 payload is refused too** — the ladder's own
    /// hostile-input hole, found while writing its docs.
    ///
    /// `decode_wire` dispatches a v2 payload straight into the frozen record, so
    /// the LIVE `de_sub_machine` guard never runs on that path. The first draft of
    /// `v2::Motion` had no guard and a note explaining why it did not need one;
    /// the note was wrong. A version ladder decodes old bytes through a
    /// *different* declaration, so every hostile-input guard on the live shape has
    /// to be re-entered on the frozen one.
    ///
    /// # The payload is SPLICED, not built
    ///
    /// The same technique as the v3 sibling below, and for a reason this arm
    /// learned the hard way: constructing four thousand nested `StateMachine`s in
    /// Rust overflows the stack in the **fixture** — in `Serialize`, before the
    /// decoder is ever reached — so a test that built its own hostile value would
    /// abort while proving nothing about the decoder. One level costs a fixed
    /// prefix/suffix block; a chain of N is that block N times.
    #[test]
    fn a_pathologically_nested_v2_payload_is_refused_rather_than_aborting() {
        use crate::state_machine::SmState;

        let leaf = || StateMachine {
            states: vec![SmState::clip("x", [1; 16])],
            ..Default::default()
        };
        let nest = |inner: StateMachine| StateMachine {
            states: vec![SmState::sub_machine("n", inner)],
            ..Default::default()
        };
        // Encoded through the FROZEN v2 writer, so these are genuine v2 bytes.
        let v2b = |m: StateMachine| {
            let live = StateMachineAsset::new(m, None);
            inf_asset::encode_shape(&v2::StateMachineAsset::from_current(&live)).unwrap()
        };
        let a = v2b(leaf());
        let b = v2b(nest(leaf()));
        let head = (0..a.len()).find(|&i| a[i] != b[i]).unwrap();
        let tail = (0..a.len() - head)
            .find(|&i| a[a.len() - 1 - i] != b[b.len() - 1 - i])
            .unwrap();
        let unit = b.len() - a.len();
        let (mut pre, mut suf) = (Vec::new(), Vec::new());
        for plen in 0..=unit {
            let p = &b[head..head + plen];
            let s = &b[b.len() - tail - (unit - plen)..b.len() - tail];
            let mut cand = b[..head].to_vec();
            cand.extend_from_slice(p);
            cand.extend_from_slice(&a[head..a.len() - tail]);
            cand.extend_from_slice(s);
            cand.extend_from_slice(&b[b.len() - tail..]);
            if cand == b {
                pre = p.to_vec();
                suf = s.to_vec();
                break;
            }
        }
        assert!(
            !pre.is_empty(),
            "the per-level v2 byte block was not recovered"
        );
        let sub_payload = |n: usize| -> Vec<u8> {
            let mut out = b[..head].to_vec();
            for _ in 0..n {
                out.extend_from_slice(&pre);
            }
            out.extend_from_slice(&a[head..a.len() - tail]);
            for _ in 0..n {
                out.extend_from_slice(&suf);
            }
            out.extend_from_slice(&b[b.len() - tail..]);
            out
        };

        // Control: one level is the feature, from a v2 file, and it migrates.
        let one = sub_payload(1);
        assert_eq!(
            inf_asset::peek_schema_version(&one, Some(StateMachineAsset::CURRENT_VERSION)),
            Some(2),
            "the spliced fixture is not a genuine v2 payload"
        );
        decode::<StateMachineAsset>(&one).expect("one sub-machine is legal v2 content");

        // …and a deep chain is a REFUSAL, not a stack overflow. 4 096 levels is
        // the depth the v3 sibling uses, and is enough to abort an unguarded
        // decoder.
        for depth in [3usize, 8, 64, 4096] {
            let bytes = sub_payload(depth);
            let err = decode::<StateMachineAsset>(&bytes)
                .err()
                .unwrap_or_else(|| {
                    panic!("a v2 sub-machine chain {depth} deep decoded as a machine")
                });
            // NAMED, not incidental — and specifically NOT `SchemaTooOld`, which
            // would send the author to re-create content whose problem is not its
            // age. That mis-diagnosis is what `AssetPayload::migrates_from`
            // exists to prevent, and this is the arm that would notice it coming
            // back.
            let msg = err.to_string();
            assert!(
                !matches!(err, inf_asset::AssetError::SchemaTooOld { .. }),
                "a structurally hostile v2 payload was diagnosed as merely old: {msg}"
            );
            assert!(
                msg.contains("nested state machine"),
                "the refusal does not name the nesting: {msg}"
            );
        }
    }

    /// **Hostile decode, the P29.1 A1 precedent**: a truncated v2 payload and a
    /// bad enum discriminant are refusals, not panics and not silent successes.
    #[test]
    fn a_damaged_state_machine_is_refused_rather_than_trusted() {
        let good = v2_sm_bytes();
        // 1. TRUNCATION at every length. None may panic; none may succeed.
        for cut in 1..good.len() {
            let r = std::panic::catch_unwind(|| decode::<StateMachineAsset>(&good[..cut]));
            let r = r.unwrap_or_else(|_| panic!("truncating to {cut} bytes PANICKED the decoder"));
            assert!(
                r.is_err(),
                "a v2 payload truncated to {cut} of {} bytes decoded as a machine",
                good.len()
            );
        }
        // 2. A BAD DISCRIMINANT for the v3 blend mode. The two modes differ at
        //    exactly one byte, found by diffing rather than counted, so this arm
        //    cannot drift with the layout.
        let mut m = v2_machine();
        m.transitions[0].blend = Some(crate::SmBlendMode::CrossFade);
        let mut bytes = inf_asset::encode(&StateMachineAsset::new(m, None)).unwrap();
        let mut alt = v2_machine();
        alt.transitions[0].blend = Some(crate::SmBlendMode::Inertialize);
        let other = inf_asset::encode(&StateMachineAsset::new(alt, None)).unwrap();
        let at = bytes
            .iter()
            .zip(other.iter())
            .position(|(a, b)| a != b)
            .expect("the two modes differ somewhere");
        bytes[at] = 200;
        assert!(
            decode::<StateMachineAsset>(&bytes).is_err(),
            "variant index 200 decoded as a blend mode"
        );
    }
    /// **The v1 → v2 break is a NAMED refusal that says what to do.**
    ///
    /// v2 changed shapes *inside* the payload (a condition list became a tree, a
    /// `usize` source became an enum), so v1 bytes are not a prefix of v2 bytes
    /// and no `#[serde(default)]` rescues the read. What matters is that a user
    /// with a pre-P29.1 project sees `SchemaTooOld` carrying this type's remedy —
    /// the `.inf_skel` ladder's ruling, applied to the format that has *two*
    /// authoring doors.
    #[test]
    fn a_v1_state_machine_is_refused_by_name_and_names_the_remedy() {
        let err =
            decode::<StateMachineAsset>(&v1_sm_bytes()).expect_err("v1 must not decode as v2");
        match err {
            inf_asset::AssetError::SchemaTooOld {
                kind,
                found,
                current,
                remedy,
            } => {
                assert_eq!(kind, "state_machine");
                assert_eq!((found, current), (1, StateMachineAsset::CURRENT_VERSION));
                // An INSTRUCTION, not a restatement — and it names both doors.
                assert!(remedy.contains("State Machine editor"), "{remedy}");
                assert!(remedy.contains("wizard"), "{remedy}");
                // …and it is READABLE: a run of spaces inside a user-facing
                // literal is the scripted-edit law's signature (a `\` line
                // continuation eaten by a non-raw Python string).
                assert!(
                    !remedy.contains("  "),
                    "the remedy carries a run of spaces — a line continuation was \
                     eaten: {remedy:?}"
                );
            }
            other => panic!("expected SchemaTooOld, got {other:?}"),
        }
        let msg = decode::<StateMachineAsset>(&v1_sm_bytes())
            .unwrap_err()
            .to_string();
        assert!(msg.contains("State Machine editor"), "{msg}");
    }

    /// The other direction: a payload from a newer build decodes structurally and
    /// is rejected by `migrate`.
    #[test]
    fn a_future_state_machine_is_refused_as_too_new() {
        let mut a = StateMachineAsset::new(v2_machine(), None);
        a.schema_version = StateMachineAsset::CURRENT_VERSION + 1;
        let bytes = encode(&a).unwrap();
        assert!(matches!(
            decode::<StateMachineAsset>(&bytes),
            Err(inf_asset::AssetError::SchemaTooNew { .. })
        ));
    }

    /// **`migrate` asks the structural questions, not a version question** (U6).
    ///
    /// The hostile machines are built through the public fields and encoded —
    /// which is exactly how a corrupt or hand-edited `.inf_sm` arrives — and each
    /// one is a value the fixed step would otherwise index, divide by, or recurse
    /// through.
    #[test]
    fn a_structurally_broken_state_machine_is_refused_at_decode() {
        use crate::state_machine::{CmpOp, SmCond, SmState, SmTransition};
        let healthy = || StateMachine {
            states: vec![
                SmState::clip("idle", [1; 16]),
                SmState::clip("walk", [2; 16]),
            ],
            transitions: vec![SmTransition::on(0, 1, 0.2, "moving", CmpOp::Gt, 0.5)],
            entry: 0,
            ..Default::default()
        };
        // Control: it decodes.
        let bytes = encode(&StateMachineAsset::new(healthy(), None)).unwrap();
        decode::<StateMachineAsset>(&bytes).expect("a healthy machine must decode");

        let refused = |name: &str, f: &dyn Fn(&mut StateMachine)| {
            let mut m = healthy();
            f(&mut m);
            let bytes = encode(&StateMachineAsset::new(m, None)).unwrap();
            let e = decode::<StateMachineAsset>(&bytes)
                .err()
                .unwrap_or_else(|| panic!("{name} decoded as a valid machine"));
            assert!(
                e.to_string().contains("invalid state machine"),
                "{name}: {e}"
            );
        };
        // An index into `states` that the evaluator would follow.
        refused("a transition past the end", &|m| m.transitions[0].to = 9);
        refused("an entry past the end", &|m| m.entry = 9);
        // A NaN the fixed step divides the fade alpha by — the quiet one.
        refused("a NaN duration", &|m| m.transitions[0].duration = f64::NAN);
        refused("a NaN exit time", &|m| {
            m.transitions[0].exit_time = Some(f64::NAN)
        });
        // A recursion the fixed step walks.
        refused("an over-deep condition", &|m| {
            let mut c = SmCond::Always;
            for _ in 0..crate::state_machine::MAX_COND_DEPTH + 1 {
                c = SmCond::Not(Box::new(c));
            }
            m.transitions[0].condition = c;
        });
        // A profile index the blend would follow.
        refused("a profile past the end", &|m| {
            m.transitions[0].profile = Some(0)
        });
        // Two levels of sub-machine — no slot to keep the play state in.
        refused("a doubly nested sub-machine", &|m| {
            let inner = StateMachine {
                states: vec![SmState::clip("x", [1; 16])],
                ..Default::default()
            };
            let mid = StateMachine {
                states: vec![SmState::sub_machine("mid", inner)],
                ..Default::default()
            };
            m.states.push(SmState::sub_machine("outer", mid));
        });
    }

    /// **A hostile file cannot take the process down on the way in** (P29.1
    /// audit, finding A1).
    ///
    /// v2 gave `.inf_sm` its first two *recursive* shapes — `SmCond`'s
    /// `Not`/`And`/`Or`, and `Motion::SubMachine` — where v1 had a flat `Vec` of
    /// compares and three flat motions. `StateMachine::validate` bounds both, and
    /// that is the P19 parser-depth law correctly applied — but `validate` runs
    /// inside `migrate`, which runs **after** serde has already built the tree,
    /// one stack frame per level. Measured before the fix: a **7.9 KB** payload
    /// of nothing but `Not` tags, and a **54 KB** one of nested sub-machines,
    /// each ended the process with `STATUS_STACK_OVERFLOW` — in the editor's
    /// Content-Drawer scan of every loose file under a project root, and in the
    /// shipped player reading a cooked pack. An abort is not a refusal.
    ///
    /// The payloads are built as **bytes**, not by encoding a deep value: a test
    /// that has to hold the hostile tree in order to write it is a test that can
    /// overflow on its own construction (and on the `Drop` of it), which would
    /// make this arm's failure mode indistinguishable from the defect's.
    #[test]
    fn a_pathologically_nested_payload_is_refused_by_name_rather_than_aborting() {
        use crate::state_machine::{SmCond, SmState, SmTransition};
        let healthy = || StateMachine {
            states: vec![SmState::clip("idle", [1; 16])],
            transitions: vec![SmTransition::new(0, 0, 0.2)],
            entry: 0,
            ..Default::default()
        };

        // ── the condition tree ──
        // One `Not` costs exactly one byte (its variant tag), so a chain of N is
        // the healthy encoding with N copies of that tag spliced in.
        let base = encode(&StateMachineAsset::new(healthy(), None)).unwrap();
        let mut one = healthy();
        one.transitions[0].condition = SmCond::Not(Box::new(SmCond::Always));
        let one = encode(&StateMachineAsset::new(one, None)).unwrap();
        assert_eq!(one.len(), base.len() + 1, "one `Not` is one byte");
        let at = (0..base.len())
            .find(|&i| base[i] != one[i])
            .expect("the extra tag lands somewhere");
        let cond_payload = |n: usize| -> Vec<u8> {
            let mut out = base[..at].to_vec();
            out.extend(std::iter::repeat_n(one[at], n));
            out.extend_from_slice(&base[at..]);
            out
        };

        // The control: a legal tree still decodes through the same door.
        let mut legal = healthy();
        legal.transitions[0].condition = SmCond::Not(Box::new(SmCond::Not(Box::new(
            SmCond::float("speed", crate::state_machine::CmpOp::Gt, 0.1),
        ))));
        let legal = encode(&StateMachineAsset::new(legal, None)).unwrap();
        decode::<StateMachineAsset>(&legal).expect("a two-deep tree is ordinary content");

        // Just past the model's bound: `validate` owns this, and its message
        // names the transition and the depth, which is what an author can act on.
        let e =
            decode::<StateMachineAsset>(&cond_payload(crate::state_machine::MAX_COND_DEPTH + 1))
                .expect_err("an over-deep tree must be refused");
        assert!(e.to_string().contains("invalid state machine"), "{e}");

        // And far past it — where the DECODER has to be the one that stops,
        // because nothing downstream would ever run. Reaching the assertion at
        // all is half the claim: the pre-fix threshold was under eight thousand
        // tags, and past it this line was never executed.
        for n in [8_000usize, 200_000] {
            let e = decode::<StateMachineAsset>(&cond_payload(n))
                .expect_err("a 200 KB `Not` chain must be refused, not run off the stack");
            assert!(
                e.to_string().contains("nests more than"),
                "at {n}: the refusal must name the bound it hit: {e}"
            );
        }

        // ── the sub-machine chain, the same way ──
        let leaf = || StateMachine {
            states: vec![SmState::clip("x", [1; 16])],
            ..Default::default()
        };
        let nest = |inner: StateMachine| StateMachine {
            states: vec![SmState::sub_machine("n", inner)],
            ..Default::default()
        };
        let a = encode(&StateMachineAsset::new(leaf(), None)).unwrap();
        let b = encode(&StateMachineAsset::new(nest(leaf()), None)).unwrap();
        let head = (0..a.len()).find(|&i| a[i] != b[i]).unwrap();
        let tail = (0..a.len() - head)
            .find(|&i| a[a.len() - 1 - i] != b[b.len() - 1 - i])
            .unwrap();
        let unit = b.len() - a.len();
        let (mut pre, mut suf) = (Vec::new(), Vec::new());
        for plen in 0..=unit {
            let p = &b[head..head + plen];
            let s = &b[b.len() - tail - (unit - plen)..b.len() - tail];
            let mut cand = b[..head].to_vec();
            cand.extend_from_slice(p);
            cand.extend_from_slice(&a[head..a.len() - tail]);
            cand.extend_from_slice(s);
            cand.extend_from_slice(&b[b.len() - tail..]);
            if cand == b {
                pre = p.to_vec();
                suf = s.to_vec();
                break;
            }
        }
        assert!(
            !pre.is_empty(),
            "the per-level byte block was not recovered"
        );
        let sub_payload = |n: usize| -> Vec<u8> {
            let mut out = b[..head].to_vec();
            for _ in 0..n {
                out.extend_from_slice(&pre);
            }
            out.extend_from_slice(&a[head..a.len() - tail]);
            for _ in 0..n {
                out.extend_from_slice(&suf);
            }
            out.extend_from_slice(&b[b.len() - tail..]);
            out
        };
        // Control: one level of nesting is the feature, and it decodes.
        decode::<StateMachineAsset>(&sub_payload(1)).expect("one sub-machine is legal content");
        // Two: `validate` owns it and says what v2 does not support.
        let e = decode::<StateMachineAsset>(&sub_payload(2)).expect_err("two levels are refused");
        assert!(
            e.to_string().contains("sub-machine inside a sub-machine"),
            "{e}"
        );
        // Deep: the decoder stops it. 100 000 levels is a 2.7 MB file; the
        // pre-fix threshold was two thousand, or about 54 KB.
        let e = decode::<StateMachineAsset>(&sub_payload(100_000))
            .expect_err("a 100 000-deep nest must be refused, not run off the stack");
        assert!(e.to_string().contains("nests more than"), "{e}");
    }

    /// **The v2 wire SHAPE is pinned, so a tail field cannot be appended without a
    /// bump.**
    ///
    /// Two claims, and the second is the load-bearing one: the three fields decode
    /// positionally in this order, **and** the decode consumes every byte of the
    /// encoding. A fourth field appended to `StateMachineAsset` leaves bytes
    /// unaccounted for here and fails — the exact check an encoder-derived
    /// fixture cannot make, because it asks the encoder what it emitted.
    #[test]
    fn the_state_machine_wire_shape_is_pinned_field_for_field() {
        #[derive(Deserialize)]
        struct Wire {
            schema_version: u32,
            machine: StateMachine,
            skeleton: Option<[u8; 16]>,
        }
        let want = StateMachineAsset::new(v2_machine(), Some([9u8; 16]));
        let bytes = encode(&want).unwrap();
        let (wire, consumed): (Wire, usize) =
            bincode::serde::decode_from_slice(&bytes, inf_asset::bincode_config())
                .expect("the v2 shape decodes the v2 wire");
        assert_eq!(
            consumed,
            bytes.len(),
            "the encoding carries bytes the pinned three-field shape does not \
             account for — a field was appended to `StateMachineAsset` without \
             bumping `CURRENT_VERSION`"
        );
        assert_eq!(wire.schema_version, StateMachineAsset::CURRENT_VERSION);
        assert_eq!(wire.machine, want.machine);
        assert_eq!(wire.skeleton, want.skeleton);
    }

    /// **Round-2 finding B2's sweep, the one that was a live crash.**
    ///
    /// `Skeleton::joints` is private and `Skeleton` derives `Deserialize`, and
    /// serde's derive is expanded in `skeleton.rs` itself — so it builds the
    /// struct straight from the wire and `Skeleton::new`'s validation never
    /// ran on the decode path. `pose::global_transforms` then indexes
    /// `globals[p as usize]` with no bound: every posed character, every
    /// socket resolve, every cloth and hair seed, and the IK solve (which
    /// walks the whole skeleton, not just its chain), in the editor, in PIE
    /// and in the shipped player reading a cooked pack.
    ///
    /// The hostile rigs are built through **serde** rather than
    /// `Skeleton::new`, which is exactly how a corrupt file arrives.
    #[test]
    fn a_skeleton_naming_a_parent_it_does_not_have_is_refused_at_decode() {
        use crate::skeleton::{Joint, JointTransform};

        let joint = |name: &str, parent: Option<u16>| Joint {
            name: name.into(),
            parent,
            inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::IDENTITY,
        };
        // The wire shape, straight through serde — the door `Skeleton::new`
        // does not stand in.
        let wire = |joints: Vec<Joint>| -> Skeleton {
            serde_json::from_value(serde_json::json!({ "joints": joints }))
                .expect("serde builds the struct without asking Skeleton::new")
        };

        // Control: a healthy two-joint rig round-trips.
        let ok = SkeletonAsset::new(wire(vec![joint("root", None), joint("child", Some(0))]));
        let bytes = encode(&ok).unwrap();
        decode::<SkeletonAsset>(&bytes).expect("a healthy rig must still decode");

        // A parent past the end.
        let hostile = wire(vec![joint("root", None), joint("child", Some(9999))]);
        let bytes = encode(&SkeletonAsset::new(hostile)).unwrap();
        let e = decode::<SkeletonAsset>(&bytes)
            .expect_err("a parent past the joint list decoded as a valid rig");
        assert!(
            e.to_string().contains("9999"),
            "the refusal must name the offending parent: {e}"
        );

        // The silent case: an in-range parent that does not precede its child
        // reads the still-identity slot ahead of it, so the pose is wrong
        // rather than absent.
        let backwards = wire(vec![joint("a", Some(1)), joint("b", None)]);
        let bytes = encode(&SkeletonAsset::new(backwards)).unwrap();
        assert!(
            decode::<SkeletonAsset>(&bytes).is_err(),
            "a non-topological rig decoded as valid"
        );

        // An empty rig is not a rig.
        let bytes = encode(&SkeletonAsset::new(wire(Vec::new()))).unwrap();
        assert!(decode::<SkeletonAsset>(&bytes).is_err());
    }
}
