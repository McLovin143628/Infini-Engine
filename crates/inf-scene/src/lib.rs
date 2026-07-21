//! Scene model: the **runtime** `.inf_lvl` reader (P9.2 · deliverable 3).
//!
//! The editor authors levels through `inf_editor_core::scene::serialize`
//! (Ring 1). The shipped **player** (Ring 2, engine-side) must load the very
//! same `.inf_lvl` bytes without depending on any editor crate — so this Ring-0
//! crate owns a **decode-only** reader that produces the runtime level
//! representation ([`RuntimeLevel`]).
//!
//! # Why this is byte-compatible with the editor
//!
//! `.inf_lvl` is a deterministic `bincode` payload of concrete, `serde`-derived
//! [`inf_ecs`] component types (`Transform`, `MeshRef`, `Sprite`, `Tilemap`, …).
//! `bincode` is not self-describing, so byte-compatibility is purely a matter of
//! mirroring the field layout. This reader reuses the *same* `inf_ecs` component
//! types and reproduces the editor's record layout exactly (both schema
//! versions), so editor-written bytes decode field-for-field. The cross-tests
//! prove it against the committed sample/template levels and the frozen v1
//! fixture (the editor crate cannot be a dependency here — a ring inversion — so
//! we assert against committed bytes, which is the stronger guarantee anyway).
//!
//! The editor keeps its authoring codec (and its byte-determinism/undo tests)
//! untouched; this is the read side of the same wire format.
//!
//! # Schema versions (kept in lockstep with the editor)
//!
//! * **v1** — 3D only: transform + mesh/material/light/camera.
//! * **v2** — appends the five 2D component slots (sprite / tilemap / nine-slice
//!   / text / 2D light). A v1 payload is decoded through its frozen record and
//!   lifted with those slots defaulted — never by reinterpreting the shorter
//!   byte stream.
//! * **v3** — appends the six physics slots + the `actor` blueprint binding and a
//!   file-level settings record (gravity + rate).
//! * **v4** — appends the two P10 world components: `terrain` and `pcg_volume`.
//! * **v5** — appends the five P11 animation / character components:
//!   `skeletal_mesh` / `anim_player` / `anim_state_machine` / `root_motion` /
//!   `attached_to`. `AnimStateMachine`'s transient `runtime` is `#[serde(skip)]`,
//!   so the machine persists without its play state (rebuilt on load), like a
//!   `PcgVolume`'s `evaluated` cache. [`encode`] always writes the current
//!   schema, so cooking an older level **rewrites it to v5** (the "rewrite the
//!   level payload for runtime" step).

use inf_ecs::components::{
    AnimPlayer, AnimStateMachine, AttachedTo, Camera, CharacterController2D, CharacterController3D,
    Collider2D, Collider3D, Light, Light2D, Material, MeshRef, NineSlice, PcgVolume, RigidBody2D,
    RigidBody3D, RootMotion, SkeletalMesh, Sprite, Terrain, Text2D, Tilemap, Transform,
};
use inf_ecs::math::{Vec2d, Vec3d};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The current on-disk `.inf_lvl` schema (matches the editor's `SCHEMA_VERSION`).
pub const SCHEMA_VERSION: u32 = 5;

/// File-level simulation settings (schema v3), mirroring the editor's
/// `LevelSettings` byte-for-byte. The serde defaults preserve pre-v3 behaviour:
/// 2D gravity **zero** (the character-self-gravity convention), 3D gravity
/// `(0, -9.81, 0)`, and a 60 Hz fixed rate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RuntimeSettings {
    #[serde(default)]
    pub gravity_2d: Vec2d,
    #[serde(default = "default_gravity_3d")]
    pub gravity_3d: Vec3d,
    #[serde(default = "default_sim_hz")]
    pub sim_hz: f64,
}

fn default_gravity_3d() -> Vec3d {
    Vec3d::new(0.0, -9.81, 0.0)
}
fn default_sim_hz() -> f64 {
    60.0
}

impl Default for RuntimeSettings {
    fn default() -> Self {
        Self {
            gravity_2d: Vec2d::ZERO,
            gravity_3d: default_gravity_3d(),
            sim_hz: default_sim_hz(),
        }
    }
}

/// A failure decoding or encoding a `.inf_lvl`.
#[derive(Debug, thiserror::Error)]
pub enum SceneError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("decode: {0}")]
    Decode(String),
    #[error("encode: {0}")]
    Encode(String),
    /// The payload's leading schema version is newer than this build understands.
    #[error("scene schema v{found} is newer than this build (v{current})")]
    SchemaTooNew { found: u32, current: u32 },
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, SceneError>;

/// One entity's persisted state — the runtime record.
///
/// This is the **current (schema-v4)** wire layout: field order and the
/// `#[serde(default)]` markers mirror the editor's `EntityRecord` byte-for-byte so
/// the same bincode payload decodes here. Component slots are `Option`s (a slot is
/// `Some` when the entity carries that component).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeEntity {
    /// Stable entity GUID (what selection/hierarchy reference across a reload).
    pub guid: Uuid,
    /// Display / Outliner name.
    pub name: String,
    /// Hierarchy parent GUID, if any.
    pub parent: Option<Uuid>,
    /// Local transform (translation m, rotation euler-deg, scale).
    pub transform: Transform,
    /// Self-visibility toggle.
    pub visible: bool,
    /// Renderable mesh (primitive; asset-mesh variant is a later phase).
    pub mesh: Option<MeshRef>,
    /// PBR material parameter block.
    pub material: Option<Material>,
    /// Light source.
    pub light: Option<Light>,
    /// Scene camera.
    pub camera: Option<Camera>,
    // ── v2 (P8.2b) 2D component slots ─────────────────────────────────────
    #[serde(default)]
    pub sprite: Option<Sprite>,
    #[serde(default)]
    pub tilemap: Option<Tilemap>,
    #[serde(default)]
    pub nine_slice: Option<NineSlice>,
    #[serde(default)]
    pub text2d: Option<Text2D>,
    #[serde(default)]
    pub light_2d: Option<Light2D>,
    // ── v3 (P9.5) physics components + actor binding ──────────────────────
    #[serde(default)]
    pub rigid_body_2d: Option<RigidBody2D>,
    #[serde(default)]
    pub collider_2d: Option<Collider2D>,
    #[serde(default)]
    pub character_controller_2d: Option<CharacterController2D>,
    #[serde(default)]
    pub rigid_body_3d: Option<RigidBody3D>,
    #[serde(default)]
    pub collider_3d: Option<Collider3D>,
    #[serde(default)]
    pub character_controller_3d: Option<CharacterController3D>,
    /// GUID of the `.inf_act` blueprint-class asset bound to this entity.
    #[serde(default)]
    pub actor: Option<Uuid>,
    // ── v4 (P10.6) world components ───────────────────────────────────────
    /// A heightfield terrain (paged heights + splat weights + material layers).
    #[serde(default)]
    pub terrain: Option<Terrain>,
    /// A procedural scatter volume; its `evaluated` cache is `#[serde(skip)]`, so
    /// only the `graph` ref + region + seed persist and the player re-evaluates
    /// the scatter on load.
    #[serde(default)]
    pub pcg_volume: Option<PcgVolume>,
    // ── v5 (P11.4) animation / character components ───────────────────────
    /// A skinned-mesh binding (skeletal mesh + skeleton GUID refs).
    #[serde(default)]
    pub skeletal_mesh: Option<SkeletalMesh>,
    /// A single-clip play-head.
    #[serde(default)]
    pub anim_player: Option<AnimPlayer>,
    /// An animation state machine; its `runtime` play state is `#[serde(skip)]`
    /// (rebuilt on load, like `PcgVolume.evaluated`).
    #[serde(default)]
    pub anim_state_machine: Option<AnimStateMachine>,
    /// How the entity consumes its clip's root motion.
    #[serde(default)]
    pub root_motion: Option<RootMotion>,
    /// A socket-follow attachment (rides another entity's socket).
    #[serde(default)]
    pub attached_to: Option<AttachedTo>,
}

/// A decoded level ready for the runtime to instantiate.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeLevel {
    /// The level title.
    pub title: String,
    /// Entities in creation order (parents precede children).
    pub entities: Vec<RuntimeEntity>,
    /// File-level simulation settings (gravity + rate).
    pub settings: RuntimeSettings,
}

impl RuntimeLevel {
    /// Decode a `.inf_lvl` payload (any supported schema version).
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        decode(bytes)
    }

    /// Encode to the **current** schema (v4) — a deterministic bincode payload.
    pub fn encode(&self) -> Result<Vec<u8>> {
        encode(self)
    }

    /// Load and decode a `.inf_lvl` file.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        Self::decode(&std::fs::read(path)?)
    }

    /// Number of entities.
    pub fn len(&self) -> usize {
        self.entities.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// The entity with the given GUID, if present.
    pub fn entity(&self, guid: Uuid) -> Option<&RuntimeEntity> {
        self.entities.iter().find(|e| e.guid == guid)
    }
}

// ── wire records (serde layouts mirroring the editor codec) ─────────────────

fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
}

/// Just the leading `schema_version` — decoded first (bincode reads fields in
/// order and stops) to select the right versioned record.
#[derive(Deserialize)]
struct Header {
    schema_version: u32,
}

/// The schema-v5 file layout (current). `entities` reuses [`RuntimeEntity`].
#[derive(Serialize, Deserialize)]
struct SceneFileV5 {
    schema_version: u32,
    title: String,
    entities: Vec<RuntimeEntity>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v4 entity record (pre-P11.4: 3D + 2D + physics + actor +
/// terrain + pcg, no anim/character components). The FIELD SET never changes;
/// note the pre-1.0 caveat: these records embed the LIVE component types, so a
/// component-layout change re-blesses the committed fixtures (the sanctioned
/// `INF_BLESS_FIXTURES=1` path below) — true loads-forever begins when 1.0
/// freezes component snapshots inside the versioned records.
/// `Serialize` exists solely for that bless path (downgrade-writing fixtures).
#[derive(Serialize, Deserialize)]
struct EntityRecordV4 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<Material>,
    light: Option<Light>,
    camera: Option<Camera>,
    #[serde(default)]
    sprite: Option<Sprite>,
    #[serde(default)]
    tilemap: Option<Tilemap>,
    #[serde(default)]
    nine_slice: Option<NineSlice>,
    #[serde(default)]
    text2d: Option<Text2D>,
    #[serde(default)]
    light_2d: Option<Light2D>,
    #[serde(default)]
    rigid_body_2d: Option<RigidBody2D>,
    #[serde(default)]
    collider_2d: Option<Collider2D>,
    #[serde(default)]
    character_controller_2d: Option<CharacterController2D>,
    #[serde(default)]
    rigid_body_3d: Option<RigidBody3D>,
    #[serde(default)]
    collider_3d: Option<Collider3D>,
    #[serde(default)]
    character_controller_3d: Option<CharacterController3D>,
    #[serde(default)]
    actor: Option<Uuid>,
    #[serde(default)]
    terrain: Option<Terrain>,
    #[serde(default)]
    pcg_volume: Option<PcgVolume>,
}

/// A frozen schema-v4 file layout (carries the v3 file-level settings record).
#[derive(Serialize, Deserialize)]
struct SceneFileV4 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV4>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v3 entity record (pre-P10.6: 3D + 2D + physics + actor, no
/// terrain/pcg). Field set frozen; same pre-1.0 embed-live-components caveat
/// (and bless-path `Serialize`) as [`EntityRecordV4`].
#[derive(Serialize, Deserialize)]
struct EntityRecordV3 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<Material>,
    light: Option<Light>,
    camera: Option<Camera>,
    #[serde(default)]
    sprite: Option<Sprite>,
    #[serde(default)]
    tilemap: Option<Tilemap>,
    #[serde(default)]
    nine_slice: Option<NineSlice>,
    #[serde(default)]
    text2d: Option<Text2D>,
    #[serde(default)]
    light_2d: Option<Light2D>,
    #[serde(default)]
    rigid_body_2d: Option<RigidBody2D>,
    #[serde(default)]
    collider_2d: Option<Collider2D>,
    #[serde(default)]
    character_controller_2d: Option<CharacterController2D>,
    #[serde(default)]
    rigid_body_3d: Option<RigidBody3D>,
    #[serde(default)]
    collider_3d: Option<Collider3D>,
    #[serde(default)]
    character_controller_3d: Option<CharacterController3D>,
    #[serde(default)]
    actor: Option<Uuid>,
}

/// A frozen schema-v3 file layout (carries the v3 file-level settings record).
#[derive(Serialize, Deserialize)]
struct SceneFileV3 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV3>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v1 entity record (pre-P8.2b, 3D only). Never changes.
#[derive(Deserialize)]
struct EntityRecordV1 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<Material>,
    light: Option<Light>,
    camera: Option<Camera>,
}

/// A frozen schema-v1 file layout.
#[derive(Deserialize)]
struct SceneFileV1 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV1>,
}

/// A frozen schema-v2 entity record (pre-P9.5: 3D + the five 2D slots, no
/// physics/actor). Never changes.
#[derive(Deserialize)]
struct EntityRecordV2 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<Material>,
    light: Option<Light>,
    camera: Option<Camera>,
    #[serde(default)]
    sprite: Option<Sprite>,
    #[serde(default)]
    tilemap: Option<Tilemap>,
    #[serde(default)]
    nine_slice: Option<NineSlice>,
    #[serde(default)]
    text2d: Option<Text2D>,
    #[serde(default)]
    light_2d: Option<Light2D>,
}

/// A frozen schema-v2 file layout.
#[derive(Deserialize)]
struct SceneFileV2 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV2>,
}

impl EntityRecordV1 {
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material,
            light: self.light,
            camera: self.camera,
            sprite: None,
            tilemap: None,
            nine_slice: None,
            text2d: None,
            light_2d: None,
            rigid_body_2d: None,
            collider_2d: None,
            character_controller_2d: None,
            rigid_body_3d: None,
            collider_3d: None,
            character_controller_3d: None,
            actor: None,
            terrain: None,
            pcg_volume: None,
            skeletal_mesh: None,
            anim_player: None,
            anim_state_machine: None,
            root_motion: None,
            attached_to: None,
        }
    }
}

impl EntityRecordV2 {
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material,
            light: self.light,
            camera: self.camera,
            sprite: self.sprite,
            tilemap: self.tilemap,
            nine_slice: self.nine_slice,
            text2d: self.text2d,
            light_2d: self.light_2d,
            rigid_body_2d: None,
            collider_2d: None,
            character_controller_2d: None,
            rigid_body_3d: None,
            collider_3d: None,
            character_controller_3d: None,
            actor: None,
            terrain: None,
            pcg_volume: None,
            skeletal_mesh: None,
            anim_player: None,
            anim_state_machine: None,
            root_motion: None,
            attached_to: None,
        }
    }
}

impl EntityRecordV3 {
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material,
            light: self.light,
            camera: self.camera,
            sprite: self.sprite,
            tilemap: self.tilemap,
            nine_slice: self.nine_slice,
            text2d: self.text2d,
            light_2d: self.light_2d,
            rigid_body_2d: self.rigid_body_2d,
            collider_2d: self.collider_2d,
            character_controller_2d: self.character_controller_2d,
            rigid_body_3d: self.rigid_body_3d,
            collider_3d: self.collider_3d,
            character_controller_3d: self.character_controller_3d,
            actor: self.actor,
            terrain: None,
            pcg_volume: None,
            skeletal_mesh: None,
            anim_player: None,
            anim_state_machine: None,
            root_motion: None,
            attached_to: None,
        }
    }
}

impl EntityRecordV4 {
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material,
            light: self.light,
            camera: self.camera,
            sprite: self.sprite,
            tilemap: self.tilemap,
            nine_slice: self.nine_slice,
            text2d: self.text2d,
            light_2d: self.light_2d,
            rigid_body_2d: self.rigid_body_2d,
            collider_2d: self.collider_2d,
            character_controller_2d: self.character_controller_2d,
            rigid_body_3d: self.rigid_body_3d,
            collider_3d: self.collider_3d,
            character_controller_3d: self.character_controller_3d,
            actor: self.actor,
            terrain: self.terrain,
            pcg_volume: self.pcg_volume,
            skeletal_mesh: None,
            anim_player: None,
            anim_state_machine: None,
            root_motion: None,
            attached_to: None,
        }
    }
}

/// Decode a `.inf_lvl` payload, lifting older schemas to [`RuntimeLevel`].
pub fn decode(bytes: &[u8]) -> Result<RuntimeLevel> {
    let (header, _): (Header, usize) = bincode::serde::decode_from_slice(bytes, bincode_config())
        .map_err(|e| SceneError::Decode(format!("header: {e}")))?;
    match header.schema_version {
        0 | 1 => {
            let (v1, _): (SceneFileV1, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v1: {e}")))?;
            Ok(RuntimeLevel {
                title: v1.title,
                entities: v1
                    .entities
                    .into_iter()
                    .map(EntityRecordV1::into_runtime)
                    .collect(),
                settings: RuntimeSettings::default(),
            })
        }
        2 => {
            let (v2, _): (SceneFileV2, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v2: {e}")))?;
            Ok(RuntimeLevel {
                title: v2.title,
                entities: v2
                    .entities
                    .into_iter()
                    .map(EntityRecordV2::into_runtime)
                    .collect(),
                settings: RuntimeSettings::default(),
            })
        }
        3 => {
            let (v3, _): (SceneFileV3, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v3: {e}")))?;
            Ok(RuntimeLevel {
                title: v3.title,
                entities: v3
                    .entities
                    .into_iter()
                    .map(EntityRecordV3::into_runtime)
                    .collect(),
                settings: v3.settings,
            })
        }
        4 => {
            let (v4, _): (SceneFileV4, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v4: {e}")))?;
            Ok(RuntimeLevel {
                title: v4.title,
                entities: v4
                    .entities
                    .into_iter()
                    .map(EntityRecordV4::into_runtime)
                    .collect(),
                settings: v4.settings,
            })
        }
        5 => {
            let (v5, _): (SceneFileV5, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v5: {e}")))?;
            Ok(RuntimeLevel {
                title: v5.title,
                entities: v5.entities,
                settings: v5.settings,
            })
        }
        found => Err(SceneError::SchemaTooNew {
            found,
            current: SCHEMA_VERSION,
        }),
    }
}

/// Encode a level to the current schema (v5) as a deterministic bincode payload.
pub fn encode(level: &RuntimeLevel) -> Result<Vec<u8>> {
    let file = SceneFileV5 {
        schema_version: SCHEMA_VERSION,
        title: level.title.clone(),
        entities: level.entities.clone(),
        settings: level.settings,
    };
    bincode::serde::encode_to_vec(&file, bincode_config())
        .map_err(|e| SceneError::Encode(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    /// The workspace root, reachable from this crate at `crates/inf-scene`.
    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
    }

    fn read_committed(rel: &str) -> Vec<u8> {
        let p = workspace_root().join(rel);
        std::fs::read(&p).unwrap_or_else(|e| panic!("read committed {}: {e}", p.display()))
    }

    #[test]
    fn decodes_the_committed_platformer_level() {
        let level =
            RuntimeLevel::decode(&read_committed("samples/platformer-2d/Platformer.inf_lvl"))
                .expect("platformer decodes");
        assert_eq!(level.title, "Platformer 2D");
        // 5 entities (see the committed sidecar's entity_count).
        assert_eq!(level.len(), 5);
        // The platformer is a 2D scene: at least one entity carries a tilemap.
        assert!(
            level.entities.iter().any(|e| e.tilemap.is_some()),
            "platformer has a tilemap ground"
        );
        // Every entity has a name + a finite transform we can read.
        for e in &level.entities {
            assert!(!e.name.is_empty());
            let _ = e.transform.translation;
        }
    }

    #[test]
    fn decodes_the_committed_hybrid_template() {
        let level = RuntimeLevel::decode(&read_committed("templates/hybrid-2.5d/Hybrid.inf_lvl"))
            .expect("hybrid decodes");
        assert_eq!(level.title, "Hybrid 2.5D");
        assert_eq!(level.len(), 5);
    }

    #[test]
    fn decodes_the_frozen_v1_fixture_and_defaults_2d_slots() {
        // The editor's forever-load v1 fixture, copied into this crate.
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v1.inf_lvl"),
        )
        .expect("committed v1 fixture present");
        assert_eq!(bytes[0], 1, "fixture is a genuine schema-v1 payload");

        let level = RuntimeLevel::decode(&bytes).expect("v1 fixture decodes");
        assert_eq!(level.title, "Fixture Level");
        assert_eq!(level.len(), 4);

        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();
        // Legacy 3D data preserved through the lift.
        assert!(by_name("Ground").mesh.is_some());
        assert!(by_name("Ground").material.is_some());
        assert!(by_name("Sun").light.is_some());
        assert!(by_name("Cam").camera.is_some());
        assert!(!by_name("Cam").visible);
        // Every 2D slot defaulted to None on the old payload.
        for e in &level.entities {
            assert!(e.sprite.is_none());
            assert!(e.tilemap.is_none());
            assert!(e.nine_slice.is_none());
            assert!(e.text2d.is_none());
            assert!(e.light_2d.is_none());
        }
    }

    #[test]
    fn current_decode_encode_is_byte_identical_for_committed_bytes() {
        // The committed platformer is now a schema-v5 level; decode/encode is a
        // lossless identity on current-schema content (so the cook's runtime
        // rewrite of an already-current level is a no-op, and deterministic).
        let original = read_committed("samples/platformer-2d/Platformer.inf_lvl");
        assert_eq!(original[0], 5, "committed platformer is schema v5");
        let level = RuntimeLevel::decode(&original).unwrap();
        let reencoded = level.encode().unwrap();
        assert_eq!(original, reencoded, "v5 round trip must be byte-identical");
    }

    /// Re-bless the v3/v4 platformer fixtures after a (pre-1.0-sanctioned)
    /// component-layout change: downgrade the committed v5 sample through the
    /// frozen record types. Run with `INF_BLESS_FIXTURES=1`; inert otherwise.
    #[test]
    fn bless_downgraded_platformer_fixtures() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let v5 = read_committed("samples/platformer-2d/Platformer.inf_lvl");
        let level = RuntimeLevel::decode(&v5).unwrap();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

        let v4 = SceneFileV4 {
            schema_version: 4,
            title: level.title.clone(),
            entities: level
                .entities
                .iter()
                .map(|e| EntityRecordV4 {
                    guid: e.guid,
                    name: e.name.clone(),
                    parent: e.parent,
                    transform: e.transform,
                    visible: e.visible,
                    mesh: e.mesh,
                    material: e.material,
                    light: e.light,
                    camera: e.camera,
                    sprite: e.sprite.clone(),
                    tilemap: e.tilemap.clone(),
                    nine_slice: e.nine_slice.clone(),
                    text2d: e.text2d.clone(),
                    light_2d: e.light_2d,
                    rigid_body_2d: e.rigid_body_2d,
                    collider_2d: e.collider_2d,
                    character_controller_2d: e.character_controller_2d,
                    rigid_body_3d: e.rigid_body_3d,
                    collider_3d: e.collider_3d,
                    character_controller_3d: e.character_controller_3d,
                    actor: e.actor,
                    terrain: e.terrain.clone(),
                    pcg_volume: e.pcg_volume.clone(),
                })
                .collect(),
            settings: level.settings,
        };
        let bytes = bincode::serde::encode_to_vec(&v4, bincode_config()).unwrap();
        assert_eq!(bytes[0], 4);
        std::fs::write(fixtures.join("platformer_v4.inf_lvl"), &bytes).unwrap();

        let v3 = SceneFileV3 {
            schema_version: 3,
            title: level.title.clone(),
            entities: v4
                .entities
                .iter()
                .map(|e| EntityRecordV3 {
                    guid: e.guid,
                    name: e.name.clone(),
                    parent: e.parent,
                    transform: e.transform,
                    visible: e.visible,
                    mesh: e.mesh,
                    material: e.material,
                    light: e.light,
                    camera: e.camera,
                    sprite: e.sprite.clone(),
                    tilemap: e.tilemap.clone(),
                    nine_slice: e.nine_slice.clone(),
                    text2d: e.text2d.clone(),
                    light_2d: e.light_2d,
                    rigid_body_2d: e.rigid_body_2d,
                    collider_2d: e.collider_2d,
                    character_controller_2d: e.character_controller_2d,
                    rigid_body_3d: e.rigid_body_3d,
                    collider_3d: e.collider_3d,
                    character_controller_3d: e.character_controller_3d,
                    actor: e.actor,
                })
                .collect(),
            settings: level.settings,
        };
        let bytes = bincode::serde::encode_to_vec(&v3, bincode_config()).unwrap();
        assert_eq!(bytes[0], 3);
        std::fs::write(fixtures.join("platformer_v3.inf_lvl"), &bytes).unwrap();
    }

    /// The frozen pre-P11.4 (schema v4) platformer, load-tested forever so v4
    /// decode stays covered even though the committed sample is now v5.
    #[test]
    fn v4_platformer_fixture_loads_forever_and_lifts() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/platformer_v4.inf_lvl"),
        )
        .expect("committed v4 platformer fixture present");
        assert_eq!(bytes[0], 4, "fixture is a genuine schema-v4 payload");
        let level = RuntimeLevel::decode(&bytes).expect("v4 fixture decodes");
        assert_eq!(level.title, "Platformer 2D");
        assert_eq!(level.len(), 5);
        // v4 carried physics + actor, but no anim/character components → defaulted.
        assert!(level.entities.iter().any(|e| e.rigid_body_2d.is_some()));
        assert!(level.entities.iter().any(|e| e.actor.is_some()));
        for e in &level.entities {
            assert!(e.skeletal_mesh.is_none());
            assert!(e.anim_player.is_none());
            assert!(e.anim_state_machine.is_none());
            assert!(e.root_motion.is_none());
            assert!(e.attached_to.is_none());
        }
        // Rewriting a v4 level upgrades it to v5 (the cook's runtime rewrite).
        let out = level.encode().unwrap();
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
    }

    /// The frozen pre-P10.6 (schema v3) platformer, load-tested forever so v3
    /// decode stays covered even though the committed sample is now v4.
    #[test]
    fn v3_platformer_fixture_loads_forever_and_lifts() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/platformer_v3.inf_lvl"),
        )
        .expect("committed v3 platformer fixture present");
        assert_eq!(bytes[0], 3, "fixture is a genuine schema-v3 payload");
        let level = RuntimeLevel::decode(&bytes).expect("v3 fixture decodes");
        assert_eq!(level.title, "Platformer 2D");
        assert_eq!(level.len(), 5);
        // v3 carried physics + actor, but no terrain/pcg → all defaulted.
        assert!(level.entities.iter().any(|e| e.rigid_body_2d.is_some()));
        assert!(level.entities.iter().any(|e| e.actor.is_some()));
        for e in &level.entities {
            assert!(e.terrain.is_none());
            assert!(e.pcg_volume.is_none());
        }
        // Rewriting a v3 level upgrades it to v4 (the cook's runtime rewrite).
        let out = level.encode().unwrap();
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
    }

    /// The frozen pre-P9.5 (schema v2) platformer, load-tested forever so v2
    /// decode stays covered even though the committed sample is now v3.
    #[test]
    fn v2_platformer_fixture_loads_forever_and_lifts() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/platformer_v2.inf_lvl"),
        )
        .expect("committed v2 platformer fixture present");
        assert_eq!(bytes[0], 2, "fixture is a genuine schema-v2 payload");
        let level = RuntimeLevel::decode(&bytes).expect("v2 fixture decodes");
        assert_eq!(level.title, "Platformer 2D");
        assert_eq!(level.len(), 5);
        assert!(level.entities.iter().any(|e| e.tilemap.is_some()));
        // v2 had no physics/actor slots → all defaulted, settings default.
        for e in &level.entities {
            assert!(e.rigid_body_2d.is_none());
            assert!(e.collider_2d.is_none());
            assert!(e.actor.is_none());
        }
        assert_eq!(level.settings, RuntimeSettings::default());
        // Rewriting a v2 level upgrades it to v3 (the cook's runtime rewrite).
        let out = level.encode().unwrap();
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
    }

    #[test]
    fn v1_reencodes_to_current() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v1.inf_lvl"),
        )
        .unwrap();
        let level = RuntimeLevel::decode(&bytes).unwrap();
        let out = level.encode().unwrap();
        // The rewritten payload is genuine current schema (v3), re-decodes equal.
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
    }

    #[test]
    fn committed_platformer_persists_physics_and_actor() {
        // The regenerated v3 sample carries the player's physics + actor binding.
        let level =
            RuntimeLevel::decode(&read_committed("samples/platformer-2d/Platformer.inf_lvl"))
                .unwrap();
        assert!(
            level.entities.iter().any(|e| e.rigid_body_2d.is_some()),
            "v3 sample persists a rigid body"
        );
        assert!(
            level
                .entities
                .iter()
                .any(|e| e.character_controller_2d.is_some()),
            "v3 sample persists a character controller"
        );
        assert!(
            level.entities.iter().any(|e| e.actor.is_some()),
            "v3 sample persists an actor binding (the Coyote class)"
        );
    }

    #[test]
    fn rejects_a_future_schema() {
        // Hand-forge a bincode payload whose leading varint is a huge version.
        let level = RuntimeLevel {
            title: "x".into(),
            entities: vec![],
            settings: RuntimeSettings::default(),
        };
        let mut bytes = level.encode().unwrap();
        // schema_version is the first byte for small values; bump it well past us.
        bytes[0] = 99;
        assert!(matches!(
            RuntimeLevel::decode(&bytes),
            Err(SceneError::SchemaTooNew { .. })
        ));
    }
}
