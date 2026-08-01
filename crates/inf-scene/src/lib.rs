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
//!   `PcgVolume`'s `evaluated` cache.
//! * **v6** — appends the four P12 joints / spatial-audio components: `joint_2d` /
//!   `joint_3d` (the physics constraint linking two bodies; the `#[reflect(ignore)]`
//!   `other` entity ref is serde-persisted), `audio_source` (a spatialized emitter,
//!   its `clip` ref persisted) and `audio_listener` (the active-listener flag).
//!   [`encode`] always writes the current schema, so cooking an older level
//!   **rewrites it to the current version** (the "rewrite the level payload for
//!   runtime" step).
//! * **v7** — P13.4: [`MeshRef`] gained a mesh-**asset** GUID field
//!   (`asset: Option<Uuid>`). This changed `MeshRef`'s byte layout, so the pre-v7
//!   layout is frozen as [`MeshRefV6`] and every v1..v6 record decodes its `mesh`
//!   slot through it, lifting to the live [`MeshRef`] with `asset: None`. No new
//!   entity slot was added — v7 differs from v6 only inside `MeshRef`.

use inf_ecs::components::{
    AnimPlayer, AnimStateMachine, AttachedTo, AudioListener, AudioSource, BlendMode, Camera,
    CharacterController2D, CharacterController3D, Collider2D, Collider3D, Decal, Foliage, Joint2D,
    Joint3D, Light, Light2D, LightKind, Material, MeshRef, NineSlice, PcgVolume, RigidBody2D,
    RigidBody3D, RootMotion, SkeletalMesh, Spline, Sprite, Terrain, TerrainLayer, Text2D, Tilemap,
    Transform, Volume,
};
use inf_ecs::math::{Color, Vec2d, Vec3d};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// The current on-disk `.inf_lvl` schema (matches the editor's `SCHEMA_VERSION`).
///
/// * **v8** — R-P0: `Light` gained `range` / cone / `cast_shadows` fields;
///   `Material` gained `blend` / `alpha_cutoff`; the entity record appended four
///   world-decoration slots (`decal` / `volume` / `spline` / `foliage`); and the
///   file settings gained a `render` ([`RenderSettingsRecord`]) block. The pre-v8
///   `Light`/`Material`/settings shapes are frozen as [`LightV7`] / [`MaterialV7`]
///   / [`RuntimeSettingsV7`]; every v1..v7 record carries `light`/`material`
///   through those, and the v7→v8 lift fills the new fields at their defaults.
/// * **v9** — P16.3: `Terrain` gained an `asset: Option<Uuid>` reference to a
///   `.inf_terrain` streaming asset. No new entity slot — v9 differs from v8 only
///   inside `Terrain`, so the pre-v9 shape is frozen as [`TerrainV8`] and every
///   v4..v8 record carries its `terrain` slot through it, lifted with
///   `asset: None` (the inline `data` stays authoritative, which is exactly what
///   an older level meant).
pub const SCHEMA_VERSION: u32 = 9;

/// File-level simulation settings (schema v3+), mirroring the editor's
/// `LevelSettings` byte-for-byte. The serde defaults preserve pre-v3 behaviour:
/// 2D gravity **zero** (the character-self-gravity convention), 3D gravity
/// `(0, -9.81, 0)`, and a 60 Hz fixed rate. Schema **v8** appends a `render`
/// block (mirrors the editor's `LevelSettings::render`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RuntimeSettings {
    #[serde(default)]
    pub gravity_2d: Vec2d,
    #[serde(default = "default_gravity_3d")]
    pub gravity_3d: Vec3d,
    #[serde(default = "default_sim_hz")]
    pub sim_hz: f64,
    /// Renderer HDR / post / lighting configuration (schema v8). Additive:
    /// `#[serde(default)]` → [`RenderSettingsRecord::default`].
    #[serde(default)]
    pub render: RenderSettingsRecord,
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
            render: RenderSettingsRecord::default(),
        }
    }
}

/// Persisted renderer HDR / post / lighting settings (schema v8) — a byte-for-byte
/// mirror of the editor's `RenderSettingsRecord`. A flat, fully-explicit mirror of
/// the fields of `inf_render::RenderSettings` (kept here so the Ring-0 runtime
/// reader stays wgpu-free); the host applies it to the live `RenderSettings` at
/// load. Every default equals `inf_render::RenderSettings::default()`
/// field-for-field (see `crates/inf-render/src/settings.rs`):
/// exposure 1.0, dither true; bloom off / threshold 1.0 / knee 0.5 / intensity
/// 0.06; ssao off / radius 0.6 / intensity 1.0 / bias 0.025; taa off; shadows off
/// / max_distance 60.0; gi off / intensity 1.0.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RenderSettingsRecord {
    pub exposure: f32,
    pub dither: bool,
    pub bloom_enabled: bool,
    pub bloom_threshold: f32,
    pub bloom_knee: f32,
    pub bloom_intensity: f32,
    pub ssao_enabled: bool,
    pub ssao_radius: f32,
    pub ssao_intensity: f32,
    pub ssao_bias: f32,
    pub taa: bool,
    pub shadows_enabled: bool,
    pub shadows_max_distance: f32,
    pub gi_enabled: bool,
    pub gi_intensity: f32,
}

impl Default for RenderSettingsRecord {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            dither: true,
            bloom_enabled: false,
            bloom_threshold: 1.0,
            bloom_knee: 0.5,
            bloom_intensity: 0.06,
            ssao_enabled: false,
            ssao_radius: 0.6,
            ssao_intensity: 1.0,
            ssao_bias: 0.025,
            taa: false,
            shadows_enabled: false,
            shadows_max_distance: 60.0,
            gi_enabled: false,
            gi_intensity: 1.0,
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
/// This is the **current (schema-v7)** wire layout: field order and the
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
    // ── v6 (P12.4) joints / spatial-audio components ──────────────────────
    /// A 2D physics joint (links this body to `other`'s; the physics bridge
    /// reconciles it from the ECS components each step).
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    /// A 3D physics joint.
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    /// A spatialized sound emitter (its `clip` ref persists; the sim emits audio
    /// commands from it — the cook ships the referenced `.inf_audio`).
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    /// The active spatial-audio listener flag.
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
    // ── v8 (R-P0) world-decoration components ─────────────────────────────
    /// A projected decal.
    #[serde(default)]
    pub decal: Option<Decal>,
    /// A trigger / blocking gameplay volume.
    #[serde(default)]
    pub volume: Option<Volume>,
    /// A control-point spline (path / rail).
    #[serde(default)]
    pub spline: Option<Spline>,
    /// A foliage scatter (palette + bulk instances).
    #[serde(default)]
    pub foliage: Option<Foliage>,
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

    /// Encode to the **current** schema (v7) — a deterministic bincode payload.
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

/// The schema-v9 file layout (current). `entities` reuses [`RuntimeEntity`].
#[derive(Serialize, Deserialize)]
struct SceneFileV9 {
    schema_version: u32,
    title: String,
    entities: Vec<RuntimeEntity>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// The **pre-v8** `Light` byte layout (schema v8 froze this when `Light` gained
/// its `range` / cone / `cast_shadows` fields). Frozen entity records (v1..v7)
/// carry `light` as `Option<LightV7>`; [`LightV7::into_current`] lifts it.
#[derive(Clone, Copy, Serialize, Deserialize)]
struct LightV7 {
    kind: LightKind,
    color: Color,
    intensity: f32,
}

impl LightV7 {
    fn into_current(self) -> Light {
        Light {
            kind: self.kind,
            color: self.color,
            intensity: self.intensity,
            range: 0.0,
            inner_cone_deg: 30.0,
            outer_cone_deg: 40.0,
            cast_shadows: true,
        }
    }

    /// Downgrade a live [`Light`] to the pre-v8 layout (fixture bless path).
    #[cfg(test)]
    fn from_current(l: Light) -> Self {
        Self {
            kind: l.kind,
            color: l.color,
            intensity: l.intensity,
        }
    }
}

/// The **pre-v8** `Material` byte layout (schema v8 froze this when `Material`
/// gained its `blend` / `alpha_cutoff` fields). Frozen entity records (v1..v7)
/// carry `material` as `Option<MaterialV7>`; [`MaterialV7::into_current`] lifts it.
#[derive(Clone, Copy, Serialize, Deserialize)]
struct MaterialV7 {
    base_color: Color,
    #[serde(default)]
    metallic: f32,
    #[serde(default)]
    roughness: f32,
    #[serde(default)]
    emissive: Color,
}

impl MaterialV7 {
    fn into_current(self) -> Material {
        Material {
            base_color: self.base_color,
            metallic: self.metallic,
            roughness: self.roughness,
            emissive: self.emissive,
            blend: BlendMode::Opaque,
            alpha_cutoff: 0.5,
        }
    }

    /// Downgrade a live [`Material`] to the pre-v8 layout (fixture bless path).
    #[cfg(test)]
    fn from_current(m: Material) -> Self {
        Self {
            base_color: m.base_color,
            metallic: m.metallic,
            roughness: m.roughness,
            emissive: m.emissive,
        }
    }
}

/// The **pre-v8** file-settings byte layout (schema v8 froze this when the file
/// settings gained a `render` block). Frozen file records (v3..v7) carry
/// `settings` as [`RuntimeSettingsV7`]; [`RuntimeSettingsV7::into_current`] lifts it.
#[derive(Clone, Copy, Serialize, Deserialize)]
struct RuntimeSettingsV7 {
    #[serde(default)]
    gravity_2d: Vec2d,
    #[serde(default = "default_gravity_3d")]
    gravity_3d: Vec3d,
    #[serde(default = "default_sim_hz")]
    sim_hz: f64,
}

impl RuntimeSettingsV7 {
    fn into_current(self) -> RuntimeSettings {
        RuntimeSettings {
            gravity_2d: self.gravity_2d,
            gravity_3d: self.gravity_3d,
            sim_hz: self.sim_hz,
            render: RenderSettingsRecord::default(),
        }
    }

    /// Downgrade live [`RuntimeSettings`] to the pre-v8 layout (fixture bless path).
    #[cfg(test)]
    fn from_current(s: RuntimeSettings) -> Self {
        Self {
            gravity_2d: s.gravity_2d,
            gravity_3d: s.gravity_3d,
            sim_hz: s.sim_hz,
        }
    }
}

impl Default for RuntimeSettingsV7 {
    fn default() -> Self {
        Self {
            gravity_2d: Vec2d::ZERO,
            gravity_3d: default_gravity_3d(),
            sim_hz: default_sim_hz(),
        }
    }
}

/// The **pre-v7** `MeshRef` byte layout (P13.4 froze this when `MeshRef` gained
/// its `asset` field). Every frozen entity record (v1..v6) decodes its `mesh`
/// slot as `Option<MeshRefV6>`; [`MeshRefV6::into_current`] lifts it to the live
/// [`MeshRef`] with `asset: None` (pre-v7 levels never referenced a mesh asset).
#[derive(Clone, Copy, Serialize, Deserialize)]
struct MeshRefV6 {
    primitive: inf_ecs::components::Primitive,
}

impl MeshRefV6 {
    fn into_current(self) -> MeshRef {
        MeshRef {
            primitive: self.primitive,
            asset: None,
        }
    }

    /// Downgrade a live [`MeshRef`] to the pre-v7 layout (drops the asset ref) —
    /// used by the fixture bless path that writes older-schema records.
    #[cfg(test)]
    fn from_current(m: MeshRef) -> Self {
        Self {
            primitive: m.primitive,
        }
    }
}

/// A frozen schema-v6 entity record (pre-P13.4): all component slots through the
/// P12 joints/audio, but with the pre-v7 [`MeshRefV6`] mesh slot. v7 changed only
/// `MeshRef`, so this differs from the live [`RuntimeEntity`] only in `mesh`.
#[derive(Serialize, Deserialize)]
struct EntityRecordV6 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRefV6>,
    material: Option<MaterialV7>,
    light: Option<LightV7>,
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
    terrain: Option<TerrainV8>,
    #[serde(default)]
    pcg_volume: Option<PcgVolume>,
    #[serde(default)]
    skeletal_mesh: Option<SkeletalMesh>,
    #[serde(default)]
    anim_player: Option<AnimPlayer>,
    #[serde(default)]
    anim_state_machine: Option<AnimStateMachine>,
    #[serde(default)]
    root_motion: Option<RootMotion>,
    #[serde(default)]
    attached_to: Option<AttachedTo>,
    #[serde(default)]
    joint_2d: Option<Joint2D>,
    #[serde(default)]
    joint_3d: Option<Joint3D>,
    #[serde(default)]
    audio_source: Option<AudioSource>,
    #[serde(default)]
    audio_listener: Option<AudioListener>,
}

/// A frozen schema-v6 file layout.
#[derive(Serialize, Deserialize)]
struct SceneFileV6 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV6>,
    #[serde(default)]
    settings: RuntimeSettingsV7,
}

/// A frozen schema-v5 entity record (pre-P12.4: 3D + 2D + physics + actor +
/// terrain + pcg + the five anim/character slots, no joints/audio). The FIELD SET
/// never changes; same pre-1.0 embed-live-components caveat (and bless-path
/// `Serialize`) as [`EntityRecordV4`].
#[derive(Serialize, Deserialize)]
struct EntityRecordV5 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRefV6>,
    material: Option<MaterialV7>,
    light: Option<LightV7>,
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
    terrain: Option<TerrainV8>,
    #[serde(default)]
    pcg_volume: Option<PcgVolume>,
    #[serde(default)]
    skeletal_mesh: Option<SkeletalMesh>,
    #[serde(default)]
    anim_player: Option<AnimPlayer>,
    #[serde(default)]
    anim_state_machine: Option<AnimStateMachine>,
    #[serde(default)]
    root_motion: Option<RootMotion>,
    #[serde(default)]
    attached_to: Option<AttachedTo>,
}

/// A frozen schema-v5 file layout (carries the v3 file-level settings record).
#[derive(Serialize, Deserialize)]
struct SceneFileV5 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV5>,
    #[serde(default)]
    settings: RuntimeSettingsV7,
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
    mesh: Option<MeshRefV6>,
    material: Option<MaterialV7>,
    light: Option<LightV7>,
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
    terrain: Option<TerrainV8>,
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
    settings: RuntimeSettingsV7,
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
    mesh: Option<MeshRefV6>,
    material: Option<MaterialV7>,
    light: Option<LightV7>,
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
    settings: RuntimeSettingsV7,
}

/// A frozen schema-v1 entity record (pre-P8.2b, 3D only). Never changes.
#[derive(Deserialize)]
struct EntityRecordV1 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRefV6>,
    material: Option<MaterialV7>,
    light: Option<LightV7>,
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
    mesh: Option<MeshRefV6>,
    material: Option<MaterialV7>,
    light: Option<LightV7>,
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
            mesh: self.mesh.map(MeshRefV6::into_current),
            material: self.material.map(MaterialV7::into_current),
            light: self.light.map(LightV7::into_current),
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
            joint_2d: None,
            joint_3d: None,
            audio_source: None,
            audio_listener: None,
            decal: None,
            volume: None,
            spline: None,
            foliage: None,
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
            mesh: self.mesh.map(MeshRefV6::into_current),
            material: self.material.map(MaterialV7::into_current),
            light: self.light.map(LightV7::into_current),
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
            joint_2d: None,
            joint_3d: None,
            audio_source: None,
            audio_listener: None,
            decal: None,
            volume: None,
            spline: None,
            foliage: None,
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
            mesh: self.mesh.map(MeshRefV6::into_current),
            material: self.material.map(MaterialV7::into_current),
            light: self.light.map(LightV7::into_current),
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
            joint_2d: None,
            joint_3d: None,
            audio_source: None,
            audio_listener: None,
            decal: None,
            volume: None,
            spline: None,
            foliage: None,
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
            mesh: self.mesh.map(MeshRefV6::into_current),
            material: self.material.map(MaterialV7::into_current),
            light: self.light.map(LightV7::into_current),
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
            terrain: self.terrain.map(TerrainV8::into_current),
            pcg_volume: self.pcg_volume,
            skeletal_mesh: None,
            anim_player: None,
            anim_state_machine: None,
            root_motion: None,
            attached_to: None,
            joint_2d: None,
            joint_3d: None,
            audio_source: None,
            audio_listener: None,
            decal: None,
            volume: None,
            spline: None,
            foliage: None,
        }
    }
}

impl EntityRecordV5 {
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh.map(MeshRefV6::into_current),
            material: self.material.map(MaterialV7::into_current),
            light: self.light.map(LightV7::into_current),
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
            terrain: self.terrain.map(TerrainV8::into_current),
            pcg_volume: self.pcg_volume,
            skeletal_mesh: self.skeletal_mesh,
            anim_player: self.anim_player,
            anim_state_machine: self.anim_state_machine,
            root_motion: self.root_motion,
            attached_to: self.attached_to,
            joint_2d: None,
            joint_3d: None,
            audio_source: None,
            audio_listener: None,
            decal: None,
            volume: None,
            spline: None,
            foliage: None,
        }
    }
}

impl EntityRecordV6 {
    /// Lift a frozen v6 record to the live [`RuntimeEntity`] (v7): the pre-v7
    /// [`MeshRefV6`] mesh slot gains a `None` asset ref; every other slot carries
    /// through unchanged (v7 added no new entity slot).
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh.map(MeshRefV6::into_current),
            material: self.material.map(MaterialV7::into_current),
            light: self.light.map(LightV7::into_current),
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
            terrain: self.terrain.map(TerrainV8::into_current),
            pcg_volume: self.pcg_volume,
            skeletal_mesh: self.skeletal_mesh,
            anim_player: self.anim_player,
            anim_state_machine: self.anim_state_machine,
            root_motion: self.root_motion,
            attached_to: self.attached_to,
            joint_2d: self.joint_2d,
            joint_3d: self.joint_3d,
            audio_source: self.audio_source,
            audio_listener: self.audio_listener,
            decal: None,
            volume: None,
            spline: None,
            foliage: None,
        }
    }
}

/// The **pre-v9** `Terrain` byte layout (schema v9 froze this when `Terrain` gained
/// its `asset` reference to a `.inf_terrain` streaming asset). Frozen entity
/// records (v4..v8) carry `terrain` as `Option<TerrainV8>`;
/// [`TerrainV8::into_current`] lifts it.
///
/// The fields mirror the live component one-for-one **including their
/// `#[serde(default)]` markers** — bincode ignores defaults on the write side, but
/// keeping them identical means this record decodes every partial payload the live
/// one did, in the human-readable codecs too.
#[derive(Clone, Serialize, Deserialize)]
struct TerrainV8 {
    #[serde(default = "default_terrain_mps")]
    meters_per_sample: f64,
    #[serde(default = "default_terrain_resolution")]
    tile_resolution: u32,
    #[serde(default)]
    data: inf_terrain::TerrainData,
    #[serde(default = "inf_ecs::components::default_terrain_layers")]
    layers: [TerrainLayer; inf_ecs::components::TERRAIN_LAYERS],
    #[serde(default = "default_macro_variation")]
    macro_variation: f64,
}

fn default_terrain_mps() -> f64 {
    inf_terrain::DEFAULT_METERS_PER_SAMPLE
}
fn default_terrain_resolution() -> u32 {
    inf_terrain::DEFAULT_TILE_RESOLUTION
}
fn default_macro_variation() -> f64 {
    0.15
}

impl TerrainV8 {
    /// Lift to the live [`Terrain`]: `asset` defaults to `None`, i.e. the inline
    /// `data` remains the terrain's only authority — what a pre-v9 level meant.
    fn into_current(self) -> Terrain {
        Terrain {
            meters_per_sample: self.meters_per_sample,
            tile_resolution: self.tile_resolution,
            data: self.data,
            layers: self.layers,
            macro_variation: self.macro_variation,
            asset: None,
        }
    }

    /// Project a live [`Terrain`] back onto the frozen shape (the downgrade-bless
    /// path). The `asset` reference has no v8 home and is dropped — a lossy
    /// direction, used only to regenerate old fixtures from a current sample.
    #[cfg(test)]
    fn from_current(t: Terrain) -> Self {
        Self {
            meters_per_sample: t.meters_per_sample,
            tile_resolution: t.tile_resolution,
            data: t.data,
            layers: t.layers,
            macro_variation: t.macro_variation,
        }
    }
}

/// A frozen schema-v7 entity record (pre-R-P0): the live [`MeshRef`] mesh slot,
/// but the pre-v8 [`MaterialV7`] / [`LightV7`] slots and none of the four v8
/// world-decoration slots. v8 changed only `Light`/`Material` and appended those
/// four slots, so this differs from the live [`RuntimeEntity`] only there.
#[derive(Serialize, Deserialize)]
struct EntityRecordV7 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<MaterialV7>,
    light: Option<LightV7>,
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
    terrain: Option<TerrainV8>,
    #[serde(default)]
    pcg_volume: Option<PcgVolume>,
    #[serde(default)]
    skeletal_mesh: Option<SkeletalMesh>,
    #[serde(default)]
    anim_player: Option<AnimPlayer>,
    #[serde(default)]
    anim_state_machine: Option<AnimStateMachine>,
    #[serde(default)]
    root_motion: Option<RootMotion>,
    #[serde(default)]
    attached_to: Option<AttachedTo>,
    #[serde(default)]
    joint_2d: Option<Joint2D>,
    #[serde(default)]
    joint_3d: Option<Joint3D>,
    #[serde(default)]
    audio_source: Option<AudioSource>,
    #[serde(default)]
    audio_listener: Option<AudioListener>,
}

/// A frozen schema-v7 file layout.
#[derive(Serialize, Deserialize)]
struct SceneFileV7 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV7>,
    #[serde(default)]
    settings: RuntimeSettingsV7,
}

impl EntityRecordV7 {
    /// Lift a frozen v7 record to the live (v8) [`RuntimeEntity`]: `material`/
    /// `light` gain their v8 fields at the documented defaults; the four
    /// world-decoration slots default to `None`.
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material.map(MaterialV7::into_current),
            light: self.light.map(LightV7::into_current),
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
            terrain: self.terrain.map(TerrainV8::into_current),
            pcg_volume: self.pcg_volume,
            skeletal_mesh: self.skeletal_mesh,
            anim_player: self.anim_player,
            anim_state_machine: self.anim_state_machine,
            root_motion: self.root_motion,
            attached_to: self.attached_to,
            joint_2d: self.joint_2d,
            joint_3d: self.joint_3d,
            audio_source: self.audio_source,
            audio_listener: self.audio_listener,
            decal: None,
            volume: None,
            spline: None,
            foliage: None,
        }
    }
}

/// A frozen schema-v8 entity record (pre-P16.3): the full v8 slot set with the
/// live `Light`/`Material`/decoration components, but the **pre-v9
/// [`TerrainV8`]** terrain slot. v9 changed only `Terrain`, so this differs from
/// the live [`RuntimeEntity`] only there.
#[derive(Serialize, Deserialize)]
struct EntityRecordV8 {
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
    terrain: Option<TerrainV8>,
    #[serde(default)]
    pcg_volume: Option<PcgVolume>,
    #[serde(default)]
    skeletal_mesh: Option<SkeletalMesh>,
    #[serde(default)]
    anim_player: Option<AnimPlayer>,
    #[serde(default)]
    anim_state_machine: Option<AnimStateMachine>,
    #[serde(default)]
    root_motion: Option<RootMotion>,
    #[serde(default)]
    attached_to: Option<AttachedTo>,
    #[serde(default)]
    joint_2d: Option<Joint2D>,
    #[serde(default)]
    joint_3d: Option<Joint3D>,
    #[serde(default)]
    audio_source: Option<AudioSource>,
    #[serde(default)]
    audio_listener: Option<AudioListener>,
    #[serde(default)]
    decal: Option<Decal>,
    #[serde(default)]
    volume: Option<Volume>,
    #[serde(default)]
    spline: Option<Spline>,
    #[serde(default)]
    foliage: Option<Foliage>,
}

/// A frozen schema-v8 file layout.
#[derive(Serialize, Deserialize)]
struct SceneFileV8 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV8>,
    #[serde(default)]
    settings: RuntimeSettings,
}

impl EntityRecordV8 {
    /// Lift a frozen v8 record to the live (v9) [`RuntimeEntity`]: the terrain
    /// slot gains `asset: None` (inline data stays authoritative); everything else
    /// is carried through unchanged.
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
            terrain: self.terrain.map(TerrainV8::into_current),
            pcg_volume: self.pcg_volume,
            skeletal_mesh: self.skeletal_mesh,
            anim_player: self.anim_player,
            anim_state_machine: self.anim_state_machine,
            root_motion: self.root_motion,
            attached_to: self.attached_to,
            joint_2d: self.joint_2d,
            joint_3d: self.joint_3d,
            audio_source: self.audio_source,
            audio_listener: self.audio_listener,
            decal: self.decal,
            volume: self.volume,
            spline: self.spline,
            foliage: self.foliage,
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
                settings: v3.settings.into_current(),
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
                settings: v4.settings.into_current(),
            })
        }
        5 => {
            let (v5, _): (SceneFileV5, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v5: {e}")))?;
            Ok(RuntimeLevel {
                title: v5.title,
                entities: v5
                    .entities
                    .into_iter()
                    .map(EntityRecordV5::into_runtime)
                    .collect(),
                settings: v5.settings.into_current(),
            })
        }
        6 => {
            let (v6, _): (SceneFileV6, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v6: {e}")))?;
            Ok(RuntimeLevel {
                title: v6.title,
                entities: v6
                    .entities
                    .into_iter()
                    .map(EntityRecordV6::into_runtime)
                    .collect(),
                settings: v6.settings.into_current(),
            })
        }
        7 => {
            let (v7, _): (SceneFileV7, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v7: {e}")))?;
            Ok(RuntimeLevel {
                title: v7.title,
                entities: v7
                    .entities
                    .into_iter()
                    .map(EntityRecordV7::into_runtime)
                    .collect(),
                settings: v7.settings.into_current(),
            })
        }
        8 => {
            let (v8, _): (SceneFileV8, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v8: {e}")))?;
            Ok(RuntimeLevel {
                title: v8.title,
                entities: v8
                    .entities
                    .into_iter()
                    .map(EntityRecordV8::into_runtime)
                    .collect(),
                settings: v8.settings,
            })
        }
        9 => {
            let (v9, _): (SceneFileV9, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v9: {e}")))?;
            Ok(RuntimeLevel {
                title: v9.title,
                entities: v9.entities,
                settings: v9.settings,
            })
        }
        found => Err(SceneError::SchemaTooNew {
            found,
            current: SCHEMA_VERSION,
        }),
    }
}

/// Encode a level to the current schema (v9) as a deterministic bincode payload.
pub fn encode(level: &RuntimeLevel) -> Result<Vec<u8>> {
    let file = SceneFileV9 {
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
    fn editor_encoded_current_sample_decodes_and_round_trips_byte_identical() {
        // The committed platformer is an **editor-encoded** current-schema (v9)
        // level. This is the editor→runtime cross-decode: the Ring-0 reader parses
        // the editor's bytes field-for-field, and re-encoding is byte-identical
        // (the cook's runtime rewrite of an already-current level is a no-op).
        let original = read_committed("samples/platformer-2d/Platformer.inf_lvl");
        assert_eq!(
            original[0], SCHEMA_VERSION as u8,
            "committed platformer is the current schema (v9)"
        );
        let level = RuntimeLevel::decode(&original).unwrap();
        let reencoded = level.encode().unwrap();
        assert_eq!(
            original, reencoded,
            "current-schema round trip must be byte-identical"
        );
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
                    mesh: e.mesh.map(MeshRefV6::from_current),
                    material: e.material.map(MaterialV7::from_current),
                    light: e.light.map(LightV7::from_current),
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
                    terrain: e.terrain.clone().map(TerrainV8::from_current),
                    pcg_volume: e.pcg_volume.clone(),
                })
                .collect(),
            settings: RuntimeSettingsV7::from_current(level.settings),
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
                    // `e` is an EntityRecordV4 here — its mesh is already MeshRefV6.
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
            settings: RuntimeSettingsV7::from_current(level.settings),
        };
        let bytes = bincode::serde::encode_to_vec(&v3, bincode_config()).unwrap();
        assert_eq!(bytes[0], 3);
        std::fs::write(fixtures.join("platformer_v3.inf_lvl"), &bytes).unwrap();

        // v5: strip only the v6 joints/audio slots (the platformer carries none,
        // so this is a lossless downgrade of the committed sample).
        let v5 = SceneFileV5 {
            schema_version: 5,
            title: level.title.clone(),
            entities: level
                .entities
                .iter()
                .map(|e| EntityRecordV5 {
                    guid: e.guid,
                    name: e.name.clone(),
                    parent: e.parent,
                    transform: e.transform,
                    visible: e.visible,
                    mesh: e.mesh.map(MeshRefV6::from_current),
                    material: e.material.map(MaterialV7::from_current),
                    light: e.light.map(LightV7::from_current),
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
                    terrain: e.terrain.clone().map(TerrainV8::from_current),
                    pcg_volume: e.pcg_volume.clone(),
                    skeletal_mesh: e.skeletal_mesh,
                    anim_player: e.anim_player,
                    anim_state_machine: e.anim_state_machine,
                    root_motion: e.root_motion,
                    attached_to: e.attached_to.clone(),
                })
                .collect(),
            settings: RuntimeSettingsV7::from_current(level.settings),
        };
        let bytes = bincode::serde::encode_to_vec(&v5, bincode_config()).unwrap();
        assert_eq!(bytes[0], 5);
        std::fs::write(fixtures.join("platformer_v5.inf_lvl"), &bytes).unwrap();
    }

    /// The frozen pre-P12.4 (schema v5) platformer, load-tested forever so v5
    /// decode stays covered even though the committed sample is now v6.
    #[test]
    fn v5_platformer_fixture_loads_forever_and_lifts() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/platformer_v5.inf_lvl");
        if !path.exists() {
            eprintln!(
                "SKIP: v5 platformer fixture not blessed yet ({})",
                path.display()
            );
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes[0], 5, "fixture is a genuine schema-v5 payload");
        let level = RuntimeLevel::decode(&bytes).expect("v5 fixture decodes");
        assert_eq!(level.title, "Platformer 2D");
        assert_eq!(level.len(), 5);
        assert!(level.entities.iter().any(|e| e.rigid_body_2d.is_some()));
        assert!(level.entities.iter().any(|e| e.actor.is_some()));
        // v5 had no joints/audio slots → all defaulted.
        for e in &level.entities {
            assert!(e.joint_2d.is_none());
            assert!(e.joint_3d.is_none());
            assert!(e.audio_source.is_none());
            assert!(e.audio_listener.is_none());
        }
        // Rewriting a v5 level upgrades it to v6 (the cook's runtime rewrite).
        let out = level.encode().unwrap();
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
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

    // ── v7 forever-load fixture + v8 (R-P0) world-decoration decode ─────────

    /// A minimal all-`None` frozen v7 entity record, filled via struct-update
    /// syntax by [`v7_scene_reference`].
    fn v7_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV7 {
        EntityRecordV7 {
            guid,
            name: name.into(),
            parent,
            transform: Transform::IDENTITY,
            visible: true,
            mesh: None,
            material: None,
            light: None,
            camera: None,
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
            joint_2d: None,
            joint_3d: None,
            audio_source: None,
            audio_listener: None,
        }
    }

    /// A representative frozen schema-v7 scene (mesh, materials, each light kind,
    /// a joint, non-default settings) — the provenance source for the committed
    /// `scene_v7.inf_lvl`. Byte-identical to an editor-encoded v7 file (same
    /// inf_ecs wire types).
    fn v7_scene_reference() -> SceneFileV7 {
        use inf_ecs::components::Primitive;
        let g = uuid::Uuid::from_u128;
        let cube = g(0x7002);
        SceneFileV7 {
            schema_version: 7,
            title: "V7 Fixture Level".into(),
            entities: vec![
                EntityRecordV7 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Plane,
                        asset: None,
                    }),
                    material: Some(MaterialV7 {
                        base_color: Color::new(0.3, 0.32, 0.35, 1.0),
                        metallic: 0.0,
                        roughness: 0.5,
                        emissive: Color::new(0.0, 0.0, 0.0, 1.0),
                    }),
                    ..v7_rec(g(0x7001), "Ground", None)
                },
                EntityRecordV7 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: None,
                    }),
                    joint_3d: Some(Joint3D {
                        other: inf_ecs::EntityRef::new(g(0x7001)),
                        ..Default::default()
                    }),
                    ..v7_rec(cube, "Cube", None)
                },
                EntityRecordV7 {
                    light: Some(LightV7 {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 1.0,
                    }),
                    ..v7_rec(g(0x7006), "Sun", None)
                },
                EntityRecordV7 {
                    light: Some(LightV7 {
                        kind: LightKind::Spot,
                        color: Color::WHITE,
                        intensity: 3.0,
                    }),
                    ..v7_rec(g(0x7008), "Spot", Some(cube))
                },
            ],
            settings: RuntimeSettingsV7 {
                gravity_2d: Vec2d::new(0.0, -20.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 120.0,
            },
        }
    }

    /// Bless the committed `scene_v7.inf_lvl` from [`v7_scene_reference`] under
    /// `INF_BLESS_FIXTURES=1` (inert otherwise), mirroring the platformer-fixture
    /// bless discipline this crate already uses.
    #[test]
    fn bless_scene_v7_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v7_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 7);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v7.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed scene_v7 fixture: {}", path.display());
    }

    /// The committed schema-v7 fixture decodes here with every new v8 light /
    /// material field lifted to its default and the four world-decoration slots
    /// defaulted to `None` (the frozen-record + `into_runtime` lift).
    #[test]
    fn scene_v7_fixture_decodes_with_v8_defaults() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v7.inf_lvl");
        if !path.exists() {
            eprintln!(
                "SKIP: scene_v7 fixture not blessed yet ({})",
                path.display()
            );
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes[0], 7, "fixture is a genuine schema-v7 payload");
        // Reproducibility lock (matches the frozen writer byte-for-byte).
        let rebuilt =
            bincode::serde::encode_to_vec(v7_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "committed v7 fixture matches the frozen writer"
        );

        let level = RuntimeLevel::decode(&bytes).expect("v7 fixture decodes");
        assert_eq!(level.title, "V7 Fixture Level");
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();
        // Material/light lift to the v8 defaults.
        let m = by_name("Ground").material.unwrap();
        assert_eq!(m.blend, inf_ecs::components::BlendMode::Opaque);
        assert_eq!(m.alpha_cutoff, 0.5);
        let l = by_name("Spot").light.unwrap();
        assert_eq!(l.range, 0.0);
        assert_eq!(l.inner_cone_deg, 30.0);
        assert_eq!(l.outer_cone_deg, 40.0);
        assert!(l.cast_shadows);
        // The four world-decoration slots default to None; render block defaults.
        for e in &level.entities {
            assert!(e.decal.is_none() && e.volume.is_none());
            assert!(e.spline.is_none() && e.foliage.is_none());
        }
        assert_eq!(level.settings.sim_hz, 120.0);
        assert_eq!(level.settings.render, RenderSettingsRecord::default());
        // Rewriting lifts to the current schema (v8) and re-decodes equal.
        let out = level.encode().unwrap();
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
    }

    // ── v8 forever-load fixture (frozen pre-v9) ─────────────────────────────

    /// A deterministic authored terrain for the v8 fixture: two shared-edge tiles
    /// written from one **polynomial** height field (never `sin`/`cos` — f32/f64
    /// `std` trig is not bit-portable across targets, the P14 law), plus a painted
    /// splat weight so the fixture exercises the tile's materialized weight buffer.
    fn fixture_terrain() -> Terrain {
        let mut t = Terrain::configured(4, 2.0);
        let f = |x: f64, z: f64| x * 0.5 - z * 0.25 + 3.0;
        t.data.author_tile((0, 0), f);
        t.data.author_tile((1, 0), f);
        t.data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_weight_sample(4, 1, 2, [40, 100, 80, 35]);
        t.macro_variation = 0.25;
        t
    }

    /// A minimal all-`None` frozen v8 entity record, filled via struct-update
    /// syntax by [`v8_scene_reference`].
    fn v8_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV8 {
        EntityRecordV8 {
            guid,
            name: name.into(),
            parent,
            transform: Transform::IDENTITY,
            visible: true,
            mesh: None,
            material: None,
            light: None,
            camera: None,
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
            joint_2d: None,
            joint_3d: None,
            audio_source: None,
            audio_listener: None,
            decal: None,
            volume: None,
            spline: None,
            foliage: None,
        }
    }

    /// A representative frozen schema-v8 scene — the provenance source for the
    /// committed `scene_v8.inf_lvl`. Carries a v8 `Material` (blend + cutoff), a
    /// v8 `Light` (range + cones + shadows), the four v8 world-decoration slots
    /// and — the point of this fixture for P16.3 — a **populated `Terrain`**, so
    /// the pre-v9 `Terrain` byte layout is pinned by committed bytes.
    fn v8_scene_reference() -> SceneFileV8 {
        use inf_ecs::components::{
            BlendMode, Decal, Foliage, FoliageInstance, FoliagePaletteEntry, Primitive, Spline,
            SplineInterp, Volume, VolumeKind,
        };
        let g = uuid::Uuid::from_u128;
        let cube = g(0x8002);
        SceneFileV8 {
            schema_version: 8,
            title: "V8 Fixture Level".into(),
            entities: vec![
                EntityRecordV8 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Plane,
                        asset: None,
                    }),
                    material: Some(Material {
                        base_color: Color::new(0.3, 0.32, 0.35, 1.0),
                        metallic: 0.0,
                        roughness: 0.5,
                        emissive: Color::new(0.0, 0.0, 0.0, 1.0),
                        blend: BlendMode::Masked,
                        alpha_cutoff: 0.25,
                    }),
                    ..v8_rec(g(0x8001), "Ground", None)
                },
                EntityRecordV8 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0x80A1)),
                    }),
                    decal: Some(Decal {
                        size: Vec3d::new(3.0, 1.0, 3.0),
                        color: Color::new(0.1, 0.1, 0.1, 1.0),
                        opacity: 0.8,
                        fade_angle_deg: 50.0,
                    }),
                    volume: Some(Volume {
                        kind: VolumeKind::Blocking,
                        tint: Color::new(0.9, 0.2, 0.2, 0.5),
                    }),
                    spline: Some(Spline {
                        points: vec![
                            Vec3d::ZERO,
                            Vec3d::new(2.0, 0.0, 1.0),
                            Vec3d::new(4.0, 1.0, 0.0),
                        ],
                        closed: true,
                        interp: SplineInterp::Linear,
                    }),
                    foliage: Some(Foliage {
                        palette: vec![
                            FoliagePaletteEntry {
                                primitive: Primitive::Cone,
                                tint: Color::new(0.1, 0.6, 0.1, 1.0),
                            },
                            FoliagePaletteEntry::default(),
                        ],
                        instances: vec![
                            FoliageInstance {
                                position: Vec3d::new(1.0, 0.0, 2.0),
                                rotation: Vec3d::new(0.0, 45.0, 0.0),
                                scale: 1.2,
                                kind: 0,
                            },
                            FoliageInstance::default(),
                        ],
                    }),
                    ..v8_rec(cube, "Cube", None)
                },
                EntityRecordV8 {
                    light: Some(Light {
                        kind: LightKind::Spot,
                        color: Color::WHITE,
                        intensity: 3.0,
                        range: 25.0,
                        inner_cone_deg: 18.0,
                        outer_cone_deg: 32.0,
                        cast_shadows: false,
                    }),
                    ..v8_rec(g(0x8008), "Spot", Some(cube))
                },
                EntityRecordV8 {
                    terrain: Some(TerrainV8::from_current(fixture_terrain())),
                    ..v8_rec(g(0x8009), "Terrain", None)
                },
            ],
            settings: RuntimeSettings {
                gravity_2d: Vec2d::new(0.0, -20.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 120.0,
                render: RenderSettingsRecord {
                    exposure: 1.4,
                    dither: false,
                    bloom_enabled: true,
                    bloom_threshold: 0.8,
                    bloom_knee: 0.3,
                    bloom_intensity: 0.12,
                    ssao_enabled: true,
                    ssao_radius: 0.9,
                    ssao_intensity: 0.75,
                    ssao_bias: 0.03,
                    taa: true,
                    shadows_enabled: true,
                    shadows_max_distance: 80.0,
                    gi_enabled: true,
                    gi_intensity: 1.25,
                },
            },
        }
    }

    /// Bless the committed `scene_v8.inf_lvl` from [`v8_scene_reference`] under
    /// `INF_BLESS_FIXTURES=1` (inert otherwise) — the same discipline the v7
    /// fixture uses. Never hand-edit the committed bytes.
    #[test]
    fn bless_scene_v8_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v8_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 8);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v8.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed scene_v8 fixture: {}", path.display());
    }

    /// The committed schema-v8 fixture — written by the **pre-v9 codec**, before
    /// `Terrain` grew its asset reference — still decodes here, with the terrain
    /// lifted through the frozen [`TerrainV8`] record and `asset` defaulted to
    /// `None` (the inline data stays authoritative, which is what a v8 level
    /// meant). This is the "old bytes load forever" gate for the v9 bump.
    #[test]
    fn scene_v8_fixture_decodes_with_v9_defaults() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v8.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v8 fixture present");
        assert_eq!(bytes[0], 8, "fixture is a genuine schema-v8 payload");
        // Reproducibility lock: the frozen v8 writer still emits those exact bytes.
        let rebuilt =
            bincode::serde::encode_to_vec(v8_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "committed v8 fixture matches the frozen writer"
        );

        let level = RuntimeLevel::decode(&bytes).expect("v8 fixture decodes");
        assert_eq!(level.title, "V8 Fixture Level");
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // The terrain survives the frozen-record hop intact — heights, the painted
        // weight sample, the layers and the macro variation …
        let terrain = by_name("Terrain").terrain.as_ref().expect("terrain slot");
        assert_eq!(terrain, &fixture_terrain(), "v8 terrain decodes unchanged");
        assert_eq!(terrain.data.tile_count(), 2);
        assert_eq!(terrain.macro_variation, 0.25);
        assert_eq!(
            terrain
                .data
                .get_tile((0, 0))
                .unwrap()
                .weight_sample(4, 1, 2),
            [40, 100, 80, 35]
        );
        // … and the v9 field lifts to its documented default.
        assert_eq!(terrain.asset, None, "a v8 terrain is inline-authoritative");

        // The rest of the v8 slot set is carried through unchanged.
        assert_eq!(
            by_name("Ground").material.unwrap().blend,
            inf_ecs::components::BlendMode::Masked
        );
        assert_eq!(by_name("Spot").light.unwrap().range, 25.0);
        assert_eq!(by_name("Cube").foliage.as_ref().unwrap().instances.len(), 2);
        assert_eq!(level.settings.render.exposure, 1.4);

        // Rewriting lifts to the current schema (v9) and re-decodes equal.
        let out = level.encode().unwrap();
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
    }

    /// A v9 terrain **asset reference** persists and round-trips byte-identically,
    /// and the inline data rides alongside it (both paths legal — P16.3).
    #[test]
    fn v9_terrain_asset_reference_round_trips() {
        let asset_guid = uuid::Uuid::from_u128(0x1603_00AA);
        let mut level = RuntimeLevel {
            title: "V9 Terrain".into(),
            entities: vec![RuntimeEntity {
                terrain: Some(Terrain {
                    asset: Some(asset_guid),
                    ..fixture_terrain()
                }),
                ..v8_rec(uuid::Uuid::from_u128(0x9001), "Streamed", None).into_runtime()
            }],
            settings: RuntimeSettings::default(),
        };

        let bytes = level.encode().unwrap();
        assert_eq!(bytes[0], 9, "a v9 terrain writes a genuine v9 payload");
        let back = RuntimeLevel::decode(&bytes).expect("v9 decodes");
        assert_eq!(back, level);
        assert_eq!(back.encode().unwrap(), bytes, "re-encode is byte-identical");
        assert_eq!(
            back.entities[0].terrain.as_ref().unwrap().asset,
            Some(asset_guid)
        );
        // The inline tiles are untouched by the reference — both paths coexist.
        assert_eq!(
            back.entities[0].terrain.as_ref().unwrap().data.tile_count(),
            2
        );

        // Clearing the reference goes back to a pure inline terrain.
        level.entities[0].terrain.as_mut().unwrap().asset = None;
        let inline = level.encode().unwrap();
        assert_ne!(inline, bytes, "the asset ref is really in the bytes");
        assert_eq!(RuntimeLevel::decode(&inline).unwrap(), level);
    }

    /// A level carrying every v8 component (spot light with cones+range,
    /// translucent material, decal, volume, spline, 3-instance foliage) and
    /// non-default render settings encodes (at the current schema) and decodes with identical
    /// values. The inf-scene encode path is byte-identical to the editor's (same
    /// inf_ecs wire types), so this doubles as the editor↔runtime cross-decode.
    #[test]
    fn v8_world_decoration_components_round_trip() {
        use inf_ecs::components::{
            BlendMode, Decal, Foliage, FoliageInstance, FoliagePaletteEntry, Primitive, Spline,
            SplineInterp, Volume, VolumeKind,
        };
        // Start from the v7 reference (decoded → lifted to a v8 RuntimeLevel), then
        // author the new v8 components onto it.
        let v7_bytes =
            bincode::serde::encode_to_vec(v7_scene_reference(), bincode_config()).unwrap();
        let mut base = RuntimeLevel::decode(&v7_bytes).unwrap();
        base.settings.render = RenderSettingsRecord {
            exposure: 1.4,
            dither: false,
            bloom_enabled: true,
            bloom_threshold: 0.8,
            bloom_knee: 0.3,
            bloom_intensity: 0.12,
            ssao_enabled: true,
            ssao_radius: 0.9,
            ssao_intensity: 0.75,
            ssao_bias: 0.03,
            taa: true,
            shadows_enabled: true,
            shadows_max_distance: 80.0,
            gi_enabled: true,
            gi_intensity: 1.25,
        };
        // Give the spot light its v8 cone/range/shadow fields.
        if let Some(spot) = base.entities.iter_mut().find(|e| e.name == "Spot") {
            let l = spot.light.as_mut().unwrap();
            l.range = 25.0;
            l.inner_cone_deg = 18.0;
            l.outer_cone_deg = 32.0;
            l.cast_shadows = false;
        }
        // A translucent material on the Cube.
        if let Some(cube) = base.entities.iter_mut().find(|e| e.name == "Cube") {
            cube.material = Some(Material {
                base_color: Color::new(0.2, 0.5, 0.9, 0.4),
                metallic: 0.0,
                roughness: 0.1,
                emissive: Color::new(0.0, 0.0, 0.0, 1.0),
                blend: BlendMode::Translucent,
                alpha_cutoff: 0.3,
            });
            cube.decal = Some(Decal {
                size: Vec3d::new(3.0, 1.0, 3.0),
                color: Color::new(0.1, 0.1, 0.1, 1.0),
                opacity: 0.8,
                fade_angle_deg: 50.0,
            });
            cube.volume = Some(Volume {
                kind: VolumeKind::Blocking,
                tint: Color::new(0.9, 0.2, 0.2, 0.5),
            });
            cube.spline = Some(Spline {
                points: vec![
                    Vec3d::ZERO,
                    Vec3d::new(2.0, 0.0, 1.0),
                    Vec3d::new(4.0, 1.0, 0.0),
                ],
                closed: true,
                interp: SplineInterp::Linear,
            });
            cube.foliage = Some(Foliage {
                palette: vec![
                    FoliagePaletteEntry {
                        primitive: Primitive::Cone,
                        tint: Color::new(0.1, 0.6, 0.1, 1.0),
                    },
                    FoliagePaletteEntry::default(),
                ],
                instances: vec![
                    FoliageInstance {
                        position: Vec3d::new(1.0, 0.0, 2.0),
                        rotation: Vec3d::new(0.0, 45.0, 0.0),
                        scale: 1.2,
                        kind: 0,
                    },
                    FoliageInstance {
                        position: Vec3d::new(-2.0, 0.0, 3.0),
                        rotation: Vec3d::ZERO,
                        scale: 0.8,
                        kind: 1,
                    },
                    FoliageInstance::default(),
                ],
            });
        }

        let bytes = base.encode().unwrap();
        assert_eq!(
            bytes[0], SCHEMA_VERSION as u8,
            "encode always writes the current schema"
        );
        let back = RuntimeLevel::decode(&bytes).expect("current schema decodes");
        assert_eq!(back, base, "v8 round trip preserves every new component");
        // Re-encode is deterministic (byte-identical).
        assert_eq!(back.encode().unwrap(), bytes);

        // Spot-check the decoded values.
        let by_name = |n: &str| back.entities.iter().find(|e| e.name == n).unwrap();
        let l = by_name("Spot").light.unwrap();
        assert_eq!(l.range, 25.0);
        assert!(!l.cast_shadows);
        let cube = by_name("Cube");
        assert_eq!(cube.material.unwrap().blend, BlendMode::Translucent);
        assert_eq!(cube.decal.unwrap().opacity, 0.8);
        assert_eq!(cube.volume.unwrap().kind, VolumeKind::Blocking);
        assert_eq!(cube.spline.as_ref().unwrap().points.len(), 3);
        assert_eq!(cube.foliage.as_ref().unwrap().instances.len(), 3);
        assert_eq!(back.settings.render.exposure, 1.4);
        assert!(back.settings.render.gi_enabled);
    }
}
