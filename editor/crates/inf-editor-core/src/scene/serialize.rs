//! `.inf_lvl` serialization (P3.5).
//!
//! A level is written as two files, per the asset-system rule (ROADMAP §3):
//!   * `<name>.inf_lvl`      — bincode payload (fast, compact scene data);
//!   * `<name>.inf_lvl.toml` — human-readable, git-diffable sidecar metadata
//!     (schema version, GUID, title, entity count, content hash).
//!
//! Determinism is load-bearing: entities serialize in creation order with a
//! fixed component layout, so save → load → save is **byte-identical** (the
//! phase gate). Every record carries concrete, `serde`-derived components — not
//! reflection — so the format is stable and diffable. `schema_version` +
//! [`migrate`] keep old files loadable forever.

use std::path::{Path, PathBuf};

use inf_ecs::components::{
    ActorClass, AnimPlayer, AnimStateMachine, AttachedTo, AudioListener, AudioSource, Camera,
    CharacterController2D, CharacterController3D, Collider2D, Collider3D, Joint2D, Joint3D, Light,
    Light2D, Material, MeshRef, NineSlice, PcgVolume, RigidBody2D, RigidBody3D, RootMotion,
    SkeletalMesh, Sprite, Terrain, Text2D, Tilemap, Transform, Visibility,
};
use inf_ecs::math::{Vec2d, Vec3d};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::scene::SceneDoc;

/// Current on-disk schema. Bump on any breaking layout change and add a step to
/// [`migrate`].
///
/// * v1 — P3.5: transform + mesh/material/light/camera.
/// * v2 — P8.2b: appended the five 2D components (sprite / tilemap / nine-slice
///   / text / 2D light). Older v1 payloads load with those slots defaulted to
///   `None` (see [`decode`] + [`SceneFileV1`]).
/// * v3 — P9.5: appended the six physics components (`rigid_body_2d` /
///   `collider_2d` / `character_controller_2d` + the 3D trio) and the per-entity
///   blueprint-class binding (`actor`), plus a file-level [`LevelSettings`]
///   record (gravity + sim rate). Older v1/v2 payloads load with the new slots
///   defaulted and default settings (see [`decode`] + [`SceneFileV2`]).
/// * v4 — P10.6: appended the two P10 world components — `terrain`
///   ([`inf_ecs::components::Terrain`], heightfield + splat weights + layers) and
///   `pcg_volume` ([`inf_ecs::components::PcgVolume`], a scatter volume; its
///   `evaluated` cache is `#[serde(skip)]` so it persists **empty** and is
///   re-evaluated on demand). Older v1/v2/v3 payloads load with both slots
///   defaulted (see [`decode`] + [`SceneFileV3`]).
/// * v5 — P11.4: appended the five P11 **animation / character** components —
///   `skeletal_mesh` ([`SkeletalMesh`], skinned-mesh + skeleton GUID refs),
///   `anim_player` ([`AnimPlayer`], a clip play-head), `anim_state_machine`
///   ([`AnimStateMachine`]; its `runtime` state is `#[serde(skip)]` so the
///   machine persists **without** transient play state, exactly like a
///   `PcgVolume`'s `evaluated` cache), `root_motion` ([`RootMotion`], the
///   root-motion consume mode) and `attached_to` ([`AttachedTo`], a socket
///   follow). All five were live-session-only through v4 (the
///   `skeletal_components_serde_round_trip` guard pinned the gap); v5 is where
///   they first persist. Older v1..v4 payloads load with all five slots
///   defaulted (see [`decode`] + [`SceneFileV4`]).
///
/// * v6 — P12.4: appended the four P12 **joints / spatial-audio** components —
///   `joint_2d` / `joint_3d` ([`Joint2D`] / [`Joint3D`], the physics constraint
///   linking two bodies; the `#[reflect(ignore)]` `other` entity ref is
///   serde-persisted), `audio_source` ([`AudioSource`], a spatialized emitter) and
///   `audio_listener` ([`AudioListener`], the active listener flag). All four were
///   live-session-only through v5 (the `joint_3d_serde_round_trip_including_entity_ref`
///   and `audio_components_serde_round_trip` guards pinned the gap); v6 is where
///   they first persist. The collision-layer / combine-rule / CCD fields added in
///   the same P12.1 batch are `#[serde(default)]` extensions of the **existing**
///   `Collider*` / `RigidBody*` slots, so they persisted from v3 with no version
///   bump. Older v1..v5 payloads load with all four slots defaulted (see [`decode`]
///   and [`SceneFileV5`]).
pub const SCHEMA_VERSION: u32 = 6;

/// File-level simulation settings (P9.5 · schema v3). Replaces the player's
/// hard-coded `DEFAULT_GRAVITY`/`DEFAULT_HZ`. The serde defaults **preserve the
/// pre-v3 behaviour exactly**:
///
/// * `gravity_2d` = [`Vec2d::ZERO`] — the 2D **character-self-gravity**
///   convention: the platformer character applies its own gravity in the
///   blueprint (`vy -= GRAVITY*dt`), so a nonzero world gravity would double it.
/// * `gravity_3d` = `(0, -9.81, 0)` — real-world down for the 3D dynamic solver.
/// * `sim_hz` = `60` — the fixed update rate.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LevelSettings {
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

impl Default for LevelSettings {
    fn default() -> Self {
        Self {
            gravity_2d: Vec2d::ZERO,
            gravity_3d: default_gravity_3d(),
            sim_hz: default_sim_hz(),
        }
    }
}

/// One entity's persisted state. All component slots are always present in the
/// binary stream (bincode is not self-describing — `Option` encodes its own
/// tag, but a field may never be conditionally skipped).
///
/// **Layout is append-only across schema versions.** New component slots are
/// added at the end; a payload from schema `v(N-1)` is decoded via its
/// version-specific record ([`EntityRecordV1`]) and lifted with the new slots
/// defaulted — never by reinterpreting the shorter byte stream.
///
/// # Terrain + PcgVolume persistence (P10.6 · schema v4)
///
/// As of schema **v4** this record carries a `terrain` slot
/// ([`inf_ecs::components::Terrain`]) and a `pcg_volume` slot
/// ([`inf_ecs::components::PcgVolume`]). A spawned terrain (Add ▸ Terrain,
/// [`SpawnKind::Terrain`]) — including in-viewport **sculpted** height
/// (`EditCommand::SculptTerrain`) and **painted** splat weights
/// (`EditCommand::PaintSplat`) — now survives save/load and undo/redo of a
/// Create/Delete. `TerrainData`'s manual serde keeps unpainted tiles byte-stable.
///
/// A [`PcgVolume`]'s `evaluated` instance cache is `#[serde(skip)]`, so the
/// persisted volume carries only its `graph` ref + region + seed; the instances
/// are re-evaluated on demand (in-editor by `pcg_evaluate`, in the shipped player
/// on load — see `inf_player::level`). Save → load → save is byte-identical: the
/// skipped cache never reaches the stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecord {
    pub guid: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    pub transform: Transform,
    pub visible: bool,
    pub mesh: Option<MeshRef>,
    pub material: Option<Material>,
    pub light: Option<Light>,
    pub camera: Option<Camera>,
    // ── v2 (P8.2b) 2D components ──────────────────────────────────────────
    /// A 2D sprite quad.
    #[serde(default)]
    pub sprite: Option<Sprite>,
    /// A chunked 2D tilemap (sparse, multi-chunk content persists in full).
    #[serde(default)]
    pub tilemap: Option<Tilemap>,
    /// A 9-slice bordered panel.
    #[serde(default)]
    pub nine_slice: Option<NineSlice>,
    /// A bitmap-text label.
    #[serde(default)]
    pub text2d: Option<Text2D>,
    /// A 2D radial light.
    #[serde(default)]
    pub light_2d: Option<Light2D>,
    // ── v3 (P9.5) physics components + actor binding ──────────────────────
    /// A 2D rigid body.
    #[serde(default)]
    pub rigid_body_2d: Option<RigidBody2D>,
    /// A 2D collider.
    #[serde(default)]
    pub collider_2d: Option<Collider2D>,
    /// A 2D kinematic character mover tuning block.
    #[serde(default)]
    pub character_controller_2d: Option<CharacterController2D>,
    /// A 3D rigid body.
    #[serde(default)]
    pub rigid_body_3d: Option<RigidBody3D>,
    /// A 3D collider.
    #[serde(default)]
    pub collider_3d: Option<Collider3D>,
    /// A 3D kinematic character mover tuning block.
    #[serde(default)]
    pub character_controller_3d: Option<CharacterController3D>,
    /// The GUID of the `.inf_act` blueprint-class asset bound to this entity
    /// (the [`ActorClass`] link); `None` when the entity runs no blueprint.
    #[serde(default)]
    pub actor: Option<Uuid>,
    // ── v4 (P10.6) world components ───────────────────────────────────────
    /// A heightfield terrain (paged heights + splat weights + material layers).
    /// `TerrainData`'s manual serde keeps unpainted tiles byte-stable.
    #[serde(default)]
    pub terrain: Option<Terrain>,
    /// A procedural scatter volume. Its `evaluated` instance cache is
    /// `#[serde(skip)]`, so only the `graph` ref + region + seed persist.
    #[serde(default)]
    pub pcg_volume: Option<PcgVolume>,
    // ── v5 (P11.4) animation / character components ───────────────────────
    /// A skinned-mesh binding (skeletal mesh + skeleton GUID refs).
    #[serde(default)]
    pub skeletal_mesh: Option<SkeletalMesh>,
    /// A single-clip play-head.
    #[serde(default)]
    pub anim_player: Option<AnimPlayer>,
    /// An animation state machine. Its `runtime` play state is `#[serde(skip)]`
    /// — persisted **without** transient state (rebuilt each play session), like
    /// [`PcgVolume`]'s `evaluated` cache.
    #[serde(default)]
    pub anim_state_machine: Option<AnimStateMachine>,
    /// How the entity consumes its clip's root motion.
    #[serde(default)]
    pub root_motion: Option<RootMotion>,
    /// A socket-follow attachment (rides another entity's socket).
    #[serde(default)]
    pub attached_to: Option<AttachedTo>,
    // ── v6 (P12.4) joints / spatial-audio components ──────────────────────
    /// A 2D physics joint (links this body to `other`'s). Its `#[reflect(ignore)]`
    /// `other` entity ref is serde-persisted.
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    /// A 3D physics joint (links this body to `other`'s).
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    /// A spatialized sound emitter (its `clip` ref persists; playback is output-only).
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    /// The active spatial-audio listener flag.
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
}

/// A schema-v1 [`EntityRecord`] (pre-P8.2b) — exactly the byte layout written by
/// older editors, used only to decode legacy payloads. Kept frozen forever so
/// the committed v1 fixture (and any level saved before P8.2b) loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV1 {
    pub guid: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    pub transform: Transform,
    pub visible: bool,
    pub mesh: Option<MeshRef>,
    pub material: Option<Material>,
    pub light: Option<Light>,
    pub camera: Option<Camera>,
}

impl EntityRecordV1 {
    /// Lift a v1 record to the **v2** shape (2D component slots default to
    /// `None`). First hop of the v1→v2→v3 chain.
    fn into_v2(self) -> EntityRecordV2 {
        EntityRecordV2 {
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
        }
    }
}

/// A schema-v1 [`SceneFile`] (frozen layout for legacy decode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV1 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV1>,
}

impl SceneFileV1 {
    /// Lift a v1 file to the **v2** shape (first hop of the v1→v2→v3 chain).
    fn into_v2(self) -> SceneFileV2 {
        SceneFileV2 {
            schema_version: 2,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV1::into_v2)
                .collect(),
        }
    }
}

/// A schema-**v2** [`EntityRecord`] (pre-P9.5) — the exact byte layout written
/// by P8.2b..P9.4 editors, used only to decode legacy payloads. Frozen forever
/// so the committed v2 fixture (and any level saved before P9.5) loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV2 {
    pub guid: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    pub transform: Transform,
    pub visible: bool,
    pub mesh: Option<MeshRef>,
    pub material: Option<Material>,
    pub light: Option<Light>,
    pub camera: Option<Camera>,
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
}

impl EntityRecordV2 {
    /// Lift a v2 record to the **v3** shape: physics slots + actor default to
    /// `None`. Second hop of the v1→v2→v3→v4 chain.
    fn into_v3(self) -> EntityRecordV3 {
        EntityRecordV3 {
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
        }
    }
}

/// A schema-v2 [`SceneFile`] (frozen layout for legacy decode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV2 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV2>,
}

impl SceneFileV2 {
    /// Lift a v2 file to the **v3** shape (default [`LevelSettings`]).
    fn into_v3(self) -> SceneFileV3 {
        SceneFileV3 {
            schema_version: 3,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV2::into_v3)
                .collect(),
            settings: LevelSettings::default(),
        }
    }
}

/// A schema-**v3** [`EntityRecord`] (pre-P10.6) — the exact byte layout written by
/// P9.5..P10.5 editors (3D + 2D + physics + actor), used only to decode legacy
/// payloads. Frozen forever so the committed v3 fixture (and any level saved
/// before P10.6) loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV3 {
    pub guid: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    pub transform: Transform,
    pub visible: bool,
    pub mesh: Option<MeshRef>,
    pub material: Option<Material>,
    pub light: Option<Light>,
    pub camera: Option<Camera>,
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
    #[serde(default)]
    pub actor: Option<Uuid>,
}

impl EntityRecordV3 {
    /// Lift a v3 record to the **v4** shape: terrain + pcg_volume default to
    /// `None`. Second-to-last hop of the v1→…→v5 chain.
    fn into_v4(self) -> EntityRecordV4 {
        EntityRecordV4 {
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
        }
    }
}

/// A schema-v3 [`SceneFile`] (frozen layout for legacy decode). Carries the
/// file-level [`LevelSettings`] added in v3.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV3 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV3>,
    #[serde(default)]
    pub settings: LevelSettings,
}

impl SceneFileV3 {
    /// Lift a v3 file to the **v4** shape (settings carry through).
    fn into_v4(self) -> SceneFileV4 {
        SceneFileV4 {
            schema_version: 4,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV3::into_v4)
                .collect(),
            settings: self.settings,
        }
    }
}

/// A schema-**v4** [`EntityRecord`] (pre-P11.4) — the exact byte layout written by
/// P10.6..P11.3 editors (3D + 2D + physics + actor + terrain + pcg), used only to
/// decode legacy payloads. Frozen forever so the committed v4 fixture (and any
/// level saved before P11.4) loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV4 {
    pub guid: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    pub transform: Transform,
    pub visible: bool,
    pub mesh: Option<MeshRef>,
    pub material: Option<Material>,
    pub light: Option<Light>,
    pub camera: Option<Camera>,
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
    #[serde(default)]
    pub actor: Option<Uuid>,
    #[serde(default)]
    pub terrain: Option<Terrain>,
    #[serde(default)]
    pub pcg_volume: Option<PcgVolume>,
}

impl EntityRecordV4 {
    /// Lift a v4 record to the **v5** shape: the five P11 animation / character
    /// slots default to `None`.
    fn into_v5(self) -> EntityRecordV5 {
        EntityRecordV5 {
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

/// A schema-v4 [`SceneFile`] (frozen layout for legacy decode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV4 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV4>,
    #[serde(default)]
    pub settings: LevelSettings,
}

impl SceneFileV4 {
    /// Lift a v4 file to the **v5** shape (settings carry through).
    fn into_v5(self) -> SceneFileV5 {
        SceneFileV5 {
            schema_version: 5,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV4::into_v5)
                .collect(),
            settings: self.settings,
        }
    }
}

/// A schema-**v5** [`EntityRecord`] (pre-P12.4) — the exact byte layout written by
/// P11.4..P12.3 editors (3D + 2D + physics + actor + terrain + pcg + the five
/// anim/character slots), used only to decode legacy payloads. Frozen forever so
/// the committed v5 fixture (and any level saved before P12.4) loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV5 {
    pub guid: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    pub transform: Transform,
    pub visible: bool,
    pub mesh: Option<MeshRef>,
    pub material: Option<Material>,
    pub light: Option<Light>,
    pub camera: Option<Camera>,
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
    #[serde(default)]
    pub actor: Option<Uuid>,
    #[serde(default)]
    pub terrain: Option<Terrain>,
    #[serde(default)]
    pub pcg_volume: Option<PcgVolume>,
    #[serde(default)]
    pub skeletal_mesh: Option<SkeletalMesh>,
    #[serde(default)]
    pub anim_player: Option<AnimPlayer>,
    #[serde(default)]
    pub anim_state_machine: Option<AnimStateMachine>,
    #[serde(default)]
    pub root_motion: Option<RootMotion>,
    #[serde(default)]
    pub attached_to: Option<AttachedTo>,
}

impl EntityRecordV5 {
    /// Lift a v5 record to the current (v6) shape: the four P12 joints/audio slots
    /// default to `None`.
    fn into_current(self) -> EntityRecord {
        EntityRecord {
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
            skeletal_mesh: self.skeletal_mesh,
            anim_player: self.anim_player,
            anim_state_machine: self.anim_state_machine,
            root_motion: self.root_motion,
            attached_to: self.attached_to,
            joint_2d: None,
            joint_3d: None,
            audio_source: None,
            audio_listener: None,
        }
    }
}

/// A schema-v5 [`SceneFile`] (frozen layout for legacy decode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV5 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV5>,
    #[serde(default)]
    pub settings: LevelSettings,
}

impl SceneFileV5 {
    /// Lift a v5 file to the current (v6) shape (settings carry through).
    fn into_current(self) -> SceneFile {
        SceneFile {
            schema_version: SCHEMA_VERSION,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV5::into_current)
                .collect(),
            settings: self.settings,
        }
    }
}

/// Just the leading `schema_version` field — decoded first (bincode reads fields
/// in order and stops) to pick the right versioned record before decoding the
/// whole payload.
#[derive(Deserialize)]
struct SceneFileHeader {
    schema_version: u32,
}

/// The full level payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFile {
    pub schema_version: u32,
    pub title: String,
    /// Entities in creation order (parents precede children).
    pub entities: Vec<EntityRecord>,
    /// File-level simulation settings (schema v3). `#[serde(default)]` keeps the
    /// dual-format (TOML/JSON) round trip working for older, settings-less docs.
    #[serde(default)]
    pub settings: LevelSettings,
}

/// Sidecar metadata (TOML). Deterministic field order → stable git diffs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sidecar {
    pub schema_version: u32,
    pub guid: Uuid,
    pub title: String,
    pub entity_count: u32,
    /// xxh3 of the bincode payload — a cheap integrity + change signal.
    pub content_hash: String,
}

fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
}

/// Serialize a single entity's state (used by [`to_scene_file`] and by undo,
/// which snapshots entities before a destructive edit).
pub fn record_of(doc: &SceneDoc, guid: Uuid) -> Option<EntityRecord> {
    let world = doc.world();
    let w = world.world();
    let e = world.entity_of(guid)?;
    let parent = world.parent_of(e).and_then(|p| world.guid_of(p));
    Some(EntityRecord {
        guid,
        name: world.name_of(e).unwrap_or("").to_string(),
        parent,
        transform: w
            .get::<Transform>(e)
            .copied()
            .unwrap_or(Transform::IDENTITY),
        visible: w.get::<Visibility>(e).map(|v| v.visible).unwrap_or(true),
        mesh: w.get::<MeshRef>(e).copied(),
        material: w.get::<Material>(e).copied(),
        light: w.get::<Light>(e).copied(),
        camera: w.get::<Camera>(e).copied(),
        sprite: w.get::<Sprite>(e).cloned(),
        tilemap: w.get::<Tilemap>(e).cloned(),
        nine_slice: w.get::<NineSlice>(e).cloned(),
        text2d: w.get::<Text2D>(e).cloned(),
        light_2d: w.get::<Light2D>(e).copied(),
        rigid_body_2d: w.get::<RigidBody2D>(e).copied(),
        collider_2d: w.get::<Collider2D>(e).copied(),
        character_controller_2d: w.get::<CharacterController2D>(e).copied(),
        rigid_body_3d: w.get::<RigidBody3D>(e).copied(),
        collider_3d: w.get::<Collider3D>(e).copied(),
        character_controller_3d: w.get::<CharacterController3D>(e).copied(),
        actor: w.get::<ActorClass>(e).map(|a| a.0),
        terrain: w.get::<Terrain>(e).cloned(),
        pcg_volume: w.get::<PcgVolume>(e).cloned(),
        skeletal_mesh: w.get::<SkeletalMesh>(e).copied(),
        anim_player: w.get::<AnimPlayer>(e).copied(),
        anim_state_machine: w.get::<AnimStateMachine>(e).copied(),
        root_motion: w.get::<RootMotion>(e).copied(),
        attached_to: w.get::<AttachedTo>(e).cloned(),
        joint_2d: w.get::<Joint2D>(e).copied(),
        joint_3d: w.get::<Joint3D>(e).copied(),
        audio_source: w.get::<AudioSource>(e).cloned(),
        audio_listener: w.get::<AudioListener>(e).copied(),
    })
}

/// Project the document into a serializable [`SceneFile`].
pub fn to_scene_file(doc: &SceneDoc) -> SceneFile {
    let entities = doc
        .order()
        .iter()
        .filter_map(|&guid| record_of(doc, guid))
        .collect();
    SceneFile {
        schema_version: SCHEMA_VERSION,
        title: doc.title().to_string(),
        entities,
        settings: doc.settings(),
    }
}

/// Encode a [`SceneFile`] to the deterministic bincode payload.
pub fn encode(file: &SceneFile) -> Result<Vec<u8>, String> {
    bincode::serde::encode_to_vec(file, bincode_config()).map_err(|e| format!("encode: {e}"))
}

/// Decode a bincode payload, running migrations to the current schema. The
/// leading `schema_version` is decoded first to select the versioned record —
/// an older, shorter payload is never reinterpreted as the current (longer)
/// layout.
pub fn decode(bytes: &[u8]) -> Result<SceneFile, String> {
    let (header, _): (SceneFileHeader, usize) =
        bincode::serde::decode_from_slice(bytes, bincode_config())
            .map_err(|e| format!("decode header: {e}"))?;
    match header.schema_version {
        0 | 1 => {
            let (v1, _): (SceneFileV1, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v1: {e}"))?;
            migrate(v1.into_v2().into_v3().into_v4().into_v5().into_current())
        }
        2 => {
            let (v2, _): (SceneFileV2, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v2: {e}"))?;
            migrate(v2.into_v3().into_v4().into_v5().into_current())
        }
        3 => {
            let (v3, _): (SceneFileV3, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v3: {e}"))?;
            migrate(v3.into_v4().into_v5().into_current())
        }
        4 => {
            let (v4, _): (SceneFileV4, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v4: {e}"))?;
            migrate(v4.into_v5().into_current())
        }
        5 => {
            let (v5, _): (SceneFileV5, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v5: {e}"))?;
            migrate(v5.into_current())
        }
        6 => {
            let (file, _): (SceneFile, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode: {e}"))?;
            migrate(file)
        }
        n => Err(format!(
            "scene schema v{n} is newer than this editor (v{SCHEMA_VERSION})"
        )),
    }
}

/// Upgrade an older [`SceneFile`] to [`SCHEMA_VERSION`]. Newer-than-current is a
/// hard error (the editor is older than the file).
pub fn migrate(file: SceneFile) -> Result<SceneFile, String> {
    if file.schema_version > SCHEMA_VERSION {
        return Err(format!(
            "scene schema v{} is newer than this editor (v{SCHEMA_VERSION})",
            file.schema_version
        ));
    }
    // Records are already lifted to the current shape by the versioned decode
    // (v1→v2→v3→v4→v5→v6); nothing more to do here. Future upgrades chain in `decode`.
    Ok(file)
}

/// Rebuild a document from a decoded [`SceneFile`]. Entities are recreated in
/// file order, so parents always exist before their children.
pub fn apply_to_doc(doc: &mut SceneDoc, file: &SceneFile) {
    doc.reset();
    for rec in &file.entities {
        let e = doc.spawn_bare(rec.guid, &rec.name, rec.parent);
        let w = doc.world_mut().world_mut();
        w.entity_mut(e).insert((
            rec.transform,
            Visibility {
                visible: rec.visible,
            },
        ));
        if let Some(m) = rec.mesh {
            w.entity_mut(e).insert(m);
        }
        if let Some(m) = rec.material {
            w.entity_mut(e).insert(m);
        }
        if let Some(l) = rec.light {
            w.entity_mut(e).insert(l);
        }
        if let Some(c) = rec.camera {
            w.entity_mut(e).insert(c);
        }
        if let Some(s) = &rec.sprite {
            w.entity_mut(e).insert(s.clone());
        }
        if let Some(t) = &rec.tilemap {
            w.entity_mut(e).insert(t.clone());
        }
        if let Some(n) = &rec.nine_slice {
            w.entity_mut(e).insert(n.clone());
        }
        if let Some(t) = &rec.text2d {
            w.entity_mut(e).insert(t.clone());
        }
        if let Some(l) = &rec.light_2d {
            w.entity_mut(e).insert(*l);
        }
        if let Some(c) = &rec.rigid_body_2d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.collider_2d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.character_controller_2d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.rigid_body_3d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.collider_3d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.character_controller_3d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(actor) = rec.actor {
            w.entity_mut(e).insert(ActorClass(actor));
        }
        if let Some(t) = &rec.terrain {
            w.entity_mut(e).insert(t.clone());
        }
        if let Some(v) = &rec.pcg_volume {
            w.entity_mut(e).insert(v.clone());
        }
        if let Some(c) = &rec.skeletal_mesh {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.anim_player {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.anim_state_machine {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.root_motion {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.attached_to {
            w.entity_mut(e).insert(c.clone());
        }
        if let Some(c) = &rec.joint_2d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.joint_3d {
            w.entity_mut(e).insert(*c);
        }
        if let Some(c) = &rec.audio_source {
            w.entity_mut(e).insert(c.clone());
        }
        if let Some(c) = &rec.audio_listener {
            w.entity_mut(e).insert(*c);
        }
    }
    doc.set_title(&file.title);
    doc.set_settings(file.settings);
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
}

fn hash_hex(bytes: &[u8]) -> String {
    format!("{:016x}", xxh3(bytes))
}

/// Minimal xxh3-64. (inf-editor-core doesn't pull xxhash-rust; the asset DB
/// crate will. A small local implementation keeps the sidecar hash cheap here.)
fn xxh3(bytes: &[u8]) -> u64 {
    // FNV-1a 64 — not xxh3, but a stable content signal for the sidecar. The
    // real content-addressed hashing lands with the asset DB (P4.1); this only
    // needs to change when the payload changes.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Build the sidecar for a document + its encoded payload.
pub fn sidecar(doc: &SceneDoc, guid: Uuid, payload: &[u8]) -> Sidecar {
    Sidecar {
        schema_version: SCHEMA_VERSION,
        guid,
        title: doc.title().to_string(),
        entity_count: doc.order().len() as u32,
        content_hash: hash_hex(payload),
    }
}

/// The sidecar path for a `.inf_lvl` payload path (`foo.inf_lvl` →
/// `foo.inf_lvl.toml`).
pub fn sidecar_path(payload_path: &Path) -> PathBuf {
    let mut s = payload_path.as_os_str().to_os_string();
    s.push(".toml");
    PathBuf::from(s)
}

/// Save `doc` to `path` (payload) + its `.toml` sidecar. Returns the level GUID
/// written (fresh if `guid` is `None`).
pub fn save(doc: &SceneDoc, path: &Path, guid: Option<Uuid>) -> Result<Uuid, String> {
    let file = to_scene_file(doc);
    let payload = encode(&file)?;
    let guid = guid.unwrap_or_else(Uuid::new_v4);
    let side = sidecar(doc, guid, &payload);
    let toml = toml::to_string_pretty(&side).map_err(|e| format!("sidecar toml: {e}"))?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }
    std::fs::write(path, &payload).map_err(|e| format!("write payload: {e}"))?;
    std::fs::write(sidecar_path(path), toml).map_err(|e| format!("write sidecar: {e}"))?;
    Ok(guid)
}

/// Load a `.inf_lvl` payload into a fresh document.
pub fn load(path: &Path) -> Result<SceneDoc, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
    let file = decode(&bytes)?;
    let mut doc = SceneDoc::new();
    apply_to_doc(&mut doc, &file);
    doc.mark_saved();
    Ok(doc)
}

// ── autosave / crash recovery (P3.5.4) ───────────────────────────────────

/// The crash-recovery payload path inside `dir` (the app data dir).
pub fn recovery_path(dir: &Path) -> PathBuf {
    dir.join("crash-recovery.inf_lvl")
}

/// Write the document to the recovery file (called on a debounced autosave).
pub fn write_recovery(doc: &SceneDoc, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let payload = encode(&to_scene_file(doc))?;
    std::fs::write(recovery_path(dir), payload).map_err(|e| format!("write recovery: {e}"))
}

/// If a recovery file exists, load it and delete it (consumed on startup so a
/// clean exit removes it). Returns `None` when there's nothing to recover.
pub fn take_recovery(dir: &Path) -> Option<SceneDoc> {
    let path = recovery_path(dir);
    if !path.exists() {
        return None;
    }
    let doc = load(&path).ok();
    let _ = std::fs::remove_file(&path);
    doc
}

/// Remove the recovery file (called on a clean save / exit).
pub fn clear_recovery(dir: &Path) {
    let _ = std::fs::remove_file(recovery_path(dir));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::SpawnKind;

    fn transform_path(doc: &SceneDoc) -> &'static str {
        doc.world()
            .registry()
            .editable()
            .iter()
            .find(|c| c.display == "Transform")
            .unwrap()
            .type_path
    }

    #[test]
    fn round_trip_is_byte_identical() {
        let mut doc = SceneDoc::with_demo();
        // Author an extra edit so it's not just the demo.
        let c = doc.create(SpawnKind::Cone, "Cone", None);
        let tp = transform_path(&doc);
        doc.write_prop(
            c,
            tp,
            "translation",
            &inf_ecs::PropValue::Vec3([1.0, 2.0, 3.0]),
        );

        let file1 = to_scene_file(&doc);
        let bytes1 = encode(&file1).unwrap();

        // Load into a new doc and re-encode.
        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&doc2)).unwrap();

        assert_eq!(bytes1, bytes2, "save→load→save must be byte-identical");
    }

    #[test]
    fn save_load_through_disk_preserves_scene_and_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("Level.inf_lvl");

        let mut doc = SceneDoc::with_demo();
        let n_before = doc.snapshot().nodes.len();
        let guid = save(&doc, &path, None).unwrap();

        // Sidecar exists, is TOML, and names the level GUID + entity count.
        let toml = std::fs::read_to_string(sidecar_path(&path)).unwrap();
        assert!(toml.contains(&guid.to_string()));
        assert!(toml.contains("entity_count"));

        let mut loaded = load(&path).unwrap();
        assert_eq!(loaded.snapshot().nodes.len(), n_before);
        assert!(!loaded.is_dirty(), "a freshly loaded doc is clean");
    }

    /// FLIPPED (P10.6 · schema v4): a spawned `Terrain` — including sculpted
    /// height and painted splat weights — now **persists** across save/load and is
    /// byte-identical on re-encode. (This is the guard the v3 batch left as
    /// `terrain_is_not_persisted_yet_v4_todo`; the v4 slot closed the gap.)
    #[test]
    fn terrain_persists_across_save_load_v4() {
        use inf_ecs::components::Terrain;

        let mut doc = SceneDoc::new();
        let g = doc.create(SpawnKind::Terrain, "Terrain", None);

        // Author a multi-tile sculpted terrain (heights) + paint a splat band
        // (materialized weights on some tiles) so both the sparse-default and
        // materialized-weight paths round-trip.
        {
            let e = doc.entity_of(g).unwrap();
            let mut t = Terrain::configured(8, 1.0);
            let span = t.data.tile_span();
            // A sine hill across a 2×2 tile block (heights authored, tiles created).
            t.data.write_region(
                inf_ecs::math::Vec2d::ZERO.to_dvec2(),
                inf_ecs::math::Vec2d::splat(span * 2.0).to_dvec2(),
                |x, z| 3.0 * (x * 0.1).sin() * (z * 0.1).cos(),
            );
            // Paint layer 1 into one tile → materialized weights there, defaults
            // elsewhere (the sparse/materialized mix).
            let _ = inf_terrain::apply_paint(
                &mut t.data,
                1,
                inf_terrain::BrushParams::new(
                    glam::DVec2::new(span * 0.5, span * 0.5),
                    span * 0.4,
                    1.0,
                ),
            );
            t.macro_variation = 0.4;
            doc.world_mut().world_mut().entity_mut(e).insert(t);
            doc.world_mut().mark_dirty();
        }

        // Snapshot the authored terrain for a value comparison after reload.
        let (want_tiles, want_probe, want_painted) = {
            let e = doc.entity_of(g).unwrap();
            let t = doc.world().world().get::<Terrain>(e).unwrap();
            let probe = t.data.height_at(glam::DVec2::new(4.0, 4.0));
            let painted = t.data.tiles().any(|(_, tile)| !tile.weights_are_default());
            (t.data.tile_count(), probe, painted)
        };
        assert!(want_tiles >= 4, "multi-tile terrain authored");
        assert!(
            want_painted,
            "at least one tile carries materialized weights"
        );

        // save → load → save is byte-identical.
        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(bytes1[0], 6, "authored payload is a genuine schema-v6 file");
        let mut loaded = SceneDoc::new();
        apply_to_doc(&mut loaded, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&loaded)).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "terrain save→load→save must be byte-identical"
        );

        // The reloaded terrain is present and preserves heights + weights.
        let e2 = loaded.entity_of(g).expect("entity persists");
        let t2 = loaded
            .world()
            .world()
            .get::<Terrain>(e2)
            .expect("terrain persists across save/load (v4)");
        assert_eq!(t2.data.tile_count(), want_tiles);
        assert_eq!(t2.data.height_at(glam::DVec2::new(4.0, 4.0)), want_probe);
        assert!(
            t2.data.tiles().any(|(_, tile)| !tile.weights_are_default()),
            "painted (materialized) splat weights survive the round trip"
        );
        assert_eq!(t2.macro_variation, 0.4);
    }

    /// A [`PcgVolume`] with a graph ref round-trips (its `evaluated` cache is
    /// `#[serde(skip)]`, so a save must never carry it, and the persisted volume
    /// keeps its graph ref + region + seed).
    #[test]
    fn pcg_volume_persists_across_save_load_v4() {
        use inf_ecs::components::PcgVolume;

        let graph_guid = uuid::Uuid::from_u128(0x00C0_FFEE_1234);
        let mut doc = SceneDoc::new();
        let g = doc.create(SpawnKind::Empty, "Scatter", None);
        {
            let e = doc.entity_of(g).unwrap();
            let mut vol = PcgVolume {
                graph: Some(graph_guid),
                extent: Vec2d::new(80.0, 40.0),
                seed: 7,
                ..Default::default()
            };
            // A non-empty evaluated cache must NOT reach disk (serde-skipped).
            vol.evaluated.push(inf_ecs::components::ScatteredInstance {
                position: glam::DVec3::new(1.0, 2.0, 3.0),
                rotation: glam::DQuat::IDENTITY,
                scale: 1.0,
                kind: 0,
            });
            doc.world_mut().world_mut().entity_mut(e).insert(vol);
            doc.world_mut().mark_dirty();
        }

        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        let mut loaded = SceneDoc::new();
        apply_to_doc(&mut loaded, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&loaded)).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "PCG volume save→load→save must be byte-identical (evaluated skipped)"
        );

        let e2 = loaded.entity_of(g).unwrap();
        let v2 = loaded
            .world()
            .world()
            .get::<PcgVolume>(e2)
            .expect("pcg volume persists (v4)");
        assert_eq!(v2.graph, Some(graph_guid));
        assert_eq!(v2.extent, Vec2d::new(80.0, 40.0));
        assert_eq!(v2.seed, 7);
        assert!(
            v2.evaluated.is_empty(),
            "the evaluated cache is re-computed on demand, never persisted"
        );
    }

    /// Delete → undo restores both a `Terrain` and a `PcgVolume` (the v3 batch
    /// noted delete→undo lost `Terrain`; the v4 record slots fix it, since
    /// Create/Delete snapshot through [`record_of`] / `raw_spawn_record`).
    #[test]
    fn delete_undo_restores_terrain_and_pcg_v4() {
        use inf_ecs::components::{PcgVolume, Terrain};

        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Empty, "World", None);
        {
            let e = doc.entity_of(g).unwrap();
            let mut t = Terrain::configured(8, 1.0);
            t.data.author_tile((0, 0), |x, z| 0.25 * (x + z));
            let w = doc.world_mut().world_mut();
            w.entity_mut(e).insert(t);
            w.entity_mut(e).insert(PcgVolume {
                graph: Some(uuid::Uuid::from_u128(0xABCD)),
                ..Default::default()
            });
            doc.world_mut().mark_dirty();
        }

        // Delete (records the subtree through EntityRecord) then undo.
        doc.edit_delete(&[g]);
        assert!(doc.entity_of(g).is_none(), "deleted");
        doc.undo();

        let e = doc.entity_of(g).expect("entity restored by undo");
        let w = doc.world().world();
        let t = w
            .get::<Terrain>(e)
            .expect("terrain restored by delete→undo (v4)");
        assert_eq!(t.data.tile_count(), 1);
        let v = w
            .get::<PcgVolume>(e)
            .expect("pcg volume restored by delete→undo (v4)");
        assert_eq!(v.graph, Some(uuid::Uuid::from_u128(0xABCD)));
    }

    #[test]
    fn recovery_round_trips_then_clears() {
        let dir = tempfile::tempdir().unwrap();
        let doc = SceneDoc::with_demo();
        write_recovery(&doc, dir.path()).unwrap();
        assert!(recovery_path(dir.path()).exists());
        let recovered = take_recovery(dir.path());
        assert!(recovered.is_some());
        assert!(!recovery_path(dir.path()).exists(), "consumed on recovery");
    }

    #[test]
    fn migrate_rejects_newer_schema() {
        let mut file = SceneFile {
            schema_version: SCHEMA_VERSION + 1,
            title: "x".into(),
            entities: vec![],
            settings: LevelSettings::default(),
        };
        assert!(migrate(file.clone()).is_err());
        file.schema_version = SCHEMA_VERSION;
        assert!(migrate(file).is_ok());
    }

    #[test]
    fn decode_rejects_future_schema() {
        // A payload whose leading version is newer than us must fail cleanly,
        // not decode as v2 garbage.
        let file = SceneFile {
            schema_version: SCHEMA_VERSION + 3,
            title: "future".into(),
            entities: vec![],
            settings: LevelSettings::default(),
        };
        let bytes = encode(&file).unwrap();
        assert!(decode(&bytes).is_err());
    }

    // ── v2 (P8.2b) 2D-component persistence ───────────────────────────────

    use inf_ecs::components::{
        Light2D, NineSlice, Sprite, Text2D, Tilemap, Transform as EcsTransform,
    };
    use inf_ecs::math::{Color, Vec2d};

    /// Insert a component onto `guid` (test-only; bypasses undo). A macro sits
    /// in for a generic fn so no `bevy_ecs::Bundle` bound has to be named (this
    /// crate deliberately doesn't depend on bevy directly).
    macro_rules! insert {
        ($doc:expr, $guid:expr, $comp:expr) => {{
            if let Some(e) = $doc.entity_of($guid) {
                $doc.world_mut().world_mut().entity_mut(e).insert($comp);
                $doc.world_mut().mark_dirty();
            }
        }};
    }

    /// Author one entity per 2D component (tilemap carries multi-chunk content)
    /// plus a 3D actor, so a round trip exercises every persisted slot.
    fn authored_2d_scene() -> SceneDoc {
        let mut doc = SceneDoc::new();
        doc.set_title("2D Level");

        // A plain 3D cube (mixed 2D/3D scene).
        doc.create(SpawnKind::Cube, "Cube", None);

        let spr = doc.create(SpawnKind::Empty, "Sprite", None);
        insert!(
            doc,
            spr,
            Sprite {
                texture: Some(uuid::Uuid::from_u128(0xABCD)),
                size: Vec2d::new(2.0, 3.0),
                pivot: Vec2d::new(0.25, 0.75),
                color: Color::new(0.2, 0.4, 0.6, 0.8),
                sorting_layer: -2,
                order: 4,
                flip_x: true,
                ..Default::default()
            }
        );

        let map = doc.create(SpawnKind::Empty, "Tilemap", None);
        let mut tm = Tilemap {
            atlas_cols: 4,
            atlas_rows: 4,
            ..Default::default()
        };
        // Two occupied chunks → the multi-chunk requirement.
        tm.set_tile(1, 1, 5);
        tm.set_tile(2, 3, 9);
        tm.set_tile(100, -50, 8);
        insert!(doc, map, tm);

        let panel = doc.create(SpawnKind::Empty, "Panel", None);
        insert!(
            doc,
            panel,
            NineSlice {
                size: Vec2d::new(6.0, 4.0),
                border_uv: [0.2, 0.3, 0.25, 0.15],
                ..Default::default()
            }
        );

        let label = doc.create(SpawnKind::Empty, "Label", None);
        insert!(
            doc,
            label,
            Text2D {
                text: "Hello\nInfinity".to_string(),
                tracking: 0.1,
                ..Default::default()
            }
        );

        let lamp = doc.create(SpawnKind::Empty, "Light2D", None);
        insert!(
            doc,
            lamp,
            Light2D {
                color: Color::new(1.0, 0.5, 0.2, 1.0),
                intensity: 2.5,
                radius: 8.0,
            }
        );

        doc.world_mut().propagate();
        doc
    }

    #[test]
    fn round_trip_with_2d_components_is_byte_identical() {
        let doc = authored_2d_scene();
        let bytes1 = encode(&to_scene_file(&doc)).unwrap();

        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&doc2)).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "save→load→save with all five 2D components must be byte-identical"
        );

        // The reloaded doc keeps every component's data, incl. multi-chunk tiles.
        let file = to_scene_file(&doc2);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        let s = by_name("Sprite").sprite.as_ref().unwrap();
        assert_eq!(s.texture, Some(uuid::Uuid::from_u128(0xABCD)));
        assert_eq!(s.sorting_layer, -2);
        assert!(s.flip_x);

        let tm = by_name("Tilemap").tilemap.as_ref().unwrap();
        assert_eq!(tm.get_tile(1, 1), 5);
        assert_eq!(tm.get_tile(2, 3), 9);
        assert_eq!(tm.get_tile(100, -50), 8);
        assert_eq!(
            tm.chunks.len(),
            2,
            "multi-chunk content survives the round trip"
        );

        assert!(by_name("Panel").nine_slice.is_some());
        assert_eq!(
            by_name("Label").text2d.as_ref().unwrap().text,
            "Hello\nInfinity"
        );
        assert_eq!(by_name("Light2D").light_2d.unwrap().radius, 8.0);
        // The 3D cube carries none of the 2D slots.
        assert!(by_name("Cube").sprite.is_none());
        assert!(by_name("Cube").tilemap.is_none());
    }

    #[test]
    fn two_d_scene_survives_disk_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("TwoD.inf_lvl");
        let doc = authored_2d_scene();
        save(&doc, &path, None).unwrap();
        let mut loaded = load(&path).unwrap();
        let file = to_scene_file(&loaded);
        assert!(file.entities.iter().any(|r| r.tilemap.is_some()));
        assert!(file.entities.iter().any(|r| r.sprite.is_some()));
        assert_eq!(loaded.snapshot().nodes.len(), 6);
    }

    #[test]
    fn scene_file_is_dual_format_serde_safe() {
        // The asset rule (ROADMAP §4) requires records serialize in the
        // human-readable format too — the chunked tilemap, atlas rects and UUIDs
        // all have to survive TOML/JSON, not just bincode.
        let file = to_scene_file(&authored_2d_scene());

        let toml_s = toml::to_string(&file).expect("scene serializes to TOML");
        let back: SceneFile = toml::from_str(&toml_s).expect("scene deserializes from TOML");
        assert_eq!(back, file, "TOML round trip preserves every 2D component");

        let json = serde_json::to_string(&file).unwrap();
        let back_json: SceneFile = serde_json::from_str(&json).unwrap();
        assert_eq!(back_json, file);
    }

    // ── schema-migration fixture discipline (ROADMAP §3) ──────────────────

    /// The committed pre-P8.2b (schema v1) payload, load-tested forever.
    fn v1_fixture_bytes() -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v1.inf_lvl");
        std::fs::read(path).expect("committed v1 fixture is present")
    }

    /// Rebuild the exact schema-v1 `SceneFile` the fixture was generated from,
    /// so its provenance is reproducible from frozen legacy types. Any change to
    /// [`EntityRecordV1`]/[`SceneFileV1`] that alters the v1 layout breaks this.
    fn v1_reference() -> SceneFileV1 {
        let g = uuid::Uuid::from_u128;
        SceneFileV1 {
            schema_version: 1,
            title: "Fixture Level".into(),
            entities: vec![
                EntityRecordV1 {
                    guid: g(0x1001),
                    name: "Ground".into(),
                    parent: None,
                    transform: EcsTransform {
                        translation: inf_ecs::math::Vec3d::ZERO,
                        rotation: inf_ecs::math::Vec3d::ZERO,
                        scale: inf_ecs::math::Vec3d::new(20.0, 1.0, 20.0),
                    },
                    visible: true,
                    mesh: Some(inf_ecs::components::MeshRef {
                        primitive: inf_ecs::components::Primitive::Plane,
                    }),
                    material: Some(inf_ecs::components::Material {
                        base_color: Color::new(0.3, 0.32, 0.35, 1.0),
                        ..Default::default()
                    }),
                    light: None,
                    camera: None,
                },
                EntityRecordV1 {
                    guid: g(0x1002),
                    name: "Hero".into(),
                    parent: None,
                    transform: EcsTransform::from_translation(glam::DVec3::new(-2.0, 0.5, 0.0)),
                    visible: true,
                    mesh: Some(inf_ecs::components::MeshRef {
                        primitive: inf_ecs::components::Primitive::Cube,
                    }),
                    material: Some(inf_ecs::components::Material::default()),
                    light: None,
                    camera: None,
                },
                EntityRecordV1 {
                    guid: g(0x1003),
                    name: "Sun".into(),
                    parent: None,
                    transform: EcsTransform::IDENTITY,
                    visible: true,
                    mesh: None,
                    material: None,
                    light: Some(inf_ecs::components::Light {
                        kind: inf_ecs::components::LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 1.0,
                    }),
                    camera: None,
                },
                EntityRecordV1 {
                    guid: g(0x1004),
                    name: "Cam".into(),
                    parent: None,
                    transform: EcsTransform::IDENTITY,
                    visible: false,
                    mesh: None,
                    material: None,
                    light: None,
                    camera: Some(inf_ecs::components::Camera::default()),
                },
            ],
        }
    }

    #[test]
    fn v1_fixture_is_reproducible_and_genuinely_v1() {
        let bytes = v1_fixture_bytes();
        // The very first byte is the schema version varint (1).
        assert_eq!(bytes[0], 1, "fixture must be a genuine schema-v1 payload");
        let rebuilt = bincode::serde::encode_to_vec(v1_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed fixture must match our frozen v1 writer"
        );
    }

    #[test]
    fn v1_fixture_loads_forever() {
        let file = decode(&v1_fixture_bytes()).expect("v1 fixture decodes");
        // Migrated up to the current schema.
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "Fixture Level");
        assert_eq!(file.entities.len(), 4);

        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();
        // Legacy 3D data preserved.
        assert!(by_name("Ground").mesh.is_some());
        assert!(by_name("Ground").material.is_some());
        assert!(by_name("Sun").light.is_some());
        assert!(by_name("Cam").camera.is_some());
        assert!(!by_name("Cam").visible);
        // Every 2D + v3 slot defaulted on the old payload.
        for r in &file.entities {
            assert!(r.sprite.is_none());
            assert!(r.tilemap.is_none());
            assert!(r.nine_slice.is_none());
            assert!(r.text2d.is_none());
            assert!(r.light_2d.is_none());
            assert!(r.rigid_body_2d.is_none());
            assert!(r.collider_2d.is_none());
            assert!(r.character_controller_2d.is_none());
            assert!(r.actor.is_none());
            // v4 world slots defaulted on the old payload.
            assert!(r.terrain.is_none());
            assert!(r.pcg_volume.is_none());
        }
        // Legacy files carry no settings → the defaults (2D gravity zero).
        assert_eq!(file.settings, LevelSettings::default());
    }

    #[test]
    fn v1_fixture_loads_into_a_document() {
        let mut doc = SceneDoc::new();
        apply_to_doc(&mut doc, &decode(&v1_fixture_bytes()).unwrap());
        assert_eq!(doc.snapshot().nodes.len(), 4);
        assert_eq!(doc.title(), "Fixture Level");
    }

    // ── schema-v3 (P9.5) physics + actor + settings persistence ────────────

    use inf_ecs::components::{
        ActorClass, BodyKind2D, CharacterController2D as CC2D, Collider2D, ColliderShape2DKind,
        RigidBody2D,
    };
    use inf_ecs::math::Vec3d;

    /// Author a scene exercising every v3 slot: a static ground (rb2d+collider),
    /// a player (rb2d + collider + character controller + an actor binding), and
    /// non-default level settings — so a round trip covers physics + actor +
    /// settings.
    fn authored_v3_scene() -> (SceneDoc, uuid::Uuid) {
        let actor_guid = uuid::Uuid::from_u128(0xAC70_0001);
        let mut doc = SceneDoc::new();
        doc.set_title("Physics Level");
        doc.set_settings(LevelSettings {
            gravity_2d: Vec2d::new(0.0, -20.0),
            gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
            sim_hz: 120.0,
        });

        let ground = doc.create(SpawnKind::Empty, "Ground", None);
        insert!(
            doc,
            ground,
            RigidBody2D {
                kind: BodyKind2D::Static,
                ..Default::default()
            }
        );
        insert!(
            doc,
            ground,
            Collider2D {
                shape_kind: ColliderShape2DKind::Box,
                half_extents: Vec2d::new(3.0, 0.5),
                ..Default::default()
            }
        );

        let player = doc.create(SpawnKind::Empty, "Player", None);
        insert!(
            doc,
            player,
            RigidBody2D {
                kind: BodyKind2D::Kinematic,
                fixed_rotation: true,
                ..Default::default()
            }
        );
        insert!(
            doc,
            player,
            Collider2D {
                shape_kind: ColliderShape2DKind::Capsule,
                half_extents: Vec2d::new(0.3, 0.35),
                radius: 0.3,
                ..Default::default()
            }
        );
        insert!(doc, player, CC2D::default());
        insert!(doc, player, ActorClass(actor_guid));

        doc.world_mut().propagate();
        (doc, actor_guid)
    }

    #[test]
    fn round_trip_with_v3_physics_and_actor_is_byte_identical() {
        let (doc, actor_guid) = authored_v3_scene();
        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0], 6,
            "the physics/actor content now writes as a schema-v6 file"
        );

        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&doc2)).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "save→load→save with physics + actor + settings must be byte-identical"
        );

        // The reloaded doc keeps the physics components, the actor binding, and
        // the non-default settings.
        let file = to_scene_file(&doc2);
        let player = file.entities.iter().find(|r| r.name == "Player").unwrap();
        assert_eq!(player.rigid_body_2d.unwrap().kind, BodyKind2D::Kinematic);
        assert_eq!(
            player.collider_2d.unwrap().shape_kind,
            ColliderShape2DKind::Capsule
        );
        assert!(player.character_controller_2d.is_some());
        assert_eq!(player.actor, Some(actor_guid));
        assert_eq!(doc2.settings().gravity_2d, Vec2d::new(0.0, -20.0));
        assert_eq!(doc2.settings().sim_hz, 120.0);
    }

    #[test]
    fn v3_scene_is_dual_format_serde_safe() {
        // The dual-format rule: physics + actor + settings must survive TOML/JSON.
        let (doc, _) = authored_v3_scene();
        let file = to_scene_file(&doc);
        let toml_s = toml::to_string(&file).expect("v3 scene serializes to TOML");
        let back: SceneFile = toml::from_str(&toml_s).expect("v3 scene deserializes from TOML");
        assert_eq!(
            back, file,
            "TOML round trip preserves physics + actor + settings"
        );
        let json = serde_json::to_string(&file).unwrap();
        assert_eq!(serde_json::from_str::<SceneFile>(&json).unwrap(), file);
    }

    // ── v2 forever-load fixture discipline (mirrors the v1 fixture) ─────────

    /// The committed pre-P9.5 (schema v2) payload, load-tested forever.
    fn v2_fixture_bytes() -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v2.inf_lvl");
        std::fs::read(path).expect("committed v2 fixture is present")
    }

    /// Rebuild the exact schema-v2 `SceneFile` the v2 fixture was generated from,
    /// from the frozen [`EntityRecordV2`]/[`SceneFileV2`] types. Any change to the
    /// v2 layout breaks this (the provenance lock).
    fn v2_reference() -> SceneFileV2 {
        let g = uuid::Uuid::from_u128;
        let mut tm = Tilemap {
            atlas_cols: 2,
            atlas_rows: 2,
            ..Default::default()
        };
        tm.set_tile(0, 0, 1);
        tm.set_tile(1, 0, 2);
        SceneFileV2 {
            schema_version: 2,
            title: "V2 Fixture Level".into(),
            entities: vec![
                EntityRecordV2 {
                    guid: g(0x2001),
                    name: "Ground".into(),
                    parent: None,
                    transform: EcsTransform {
                        translation: inf_ecs::math::Vec3d::ZERO,
                        rotation: inf_ecs::math::Vec3d::ZERO,
                        scale: inf_ecs::math::Vec3d::new(10.0, 1.0, 1.0),
                    },
                    visible: true,
                    mesh: Some(inf_ecs::components::MeshRef {
                        primitive: inf_ecs::components::Primitive::Plane,
                    }),
                    material: Some(inf_ecs::components::Material::default()),
                    light: None,
                    camera: None,
                    sprite: None,
                    tilemap: Some(tm),
                    nine_slice: None,
                    text2d: None,
                    light_2d: None,
                },
                EntityRecordV2 {
                    guid: g(0x2002),
                    name: "Sprite".into(),
                    parent: None,
                    transform: EcsTransform::from_translation(glam::DVec3::new(1.0, 2.0, 0.0)),
                    visible: true,
                    mesh: None,
                    material: None,
                    light: None,
                    camera: None,
                    sprite: Some(Sprite {
                        size: Vec2d::new(0.8, 1.2),
                        color: Color::new(0.9, 0.4, 0.3, 1.0),
                        ..Default::default()
                    }),
                    tilemap: None,
                    nine_slice: None,
                    text2d: None,
                    light_2d: Some(Light2D {
                        color: Color::WHITE,
                        intensity: 1.0,
                        radius: 5.0,
                    }),
                },
            ],
        }
    }

    #[test]
    fn v2_fixture_is_reproducible_and_genuinely_v2() {
        let bytes = v2_fixture_bytes();
        assert_eq!(bytes[0], 2, "fixture must be a genuine schema-v2 payload");
        let rebuilt = bincode::serde::encode_to_vec(v2_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v2 fixture must match our frozen v2 writer"
        );
    }

    #[test]
    fn v2_fixture_loads_forever_and_lifts_to_v4() {
        let file = decode(&v2_fixture_bytes()).expect("v2 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V2 Fixture Level");
        assert_eq!(file.entities.len(), 2);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();
        // v2 data preserved through the v2→v3→v4 lift.
        assert!(by_name("Ground").tilemap.is_some());
        assert!(by_name("Sprite").sprite.is_some());
        assert!(by_name("Sprite").light_2d.is_some());
        // Every v3 + v4 slot defaulted, and settings default (2D gravity zero).
        for r in &file.entities {
            assert!(r.rigid_body_2d.is_none());
            assert!(r.collider_2d.is_none());
            assert!(r.rigid_body_3d.is_none());
            assert!(r.actor.is_none());
            assert!(r.terrain.is_none());
            assert!(r.pcg_volume.is_none());
        }
        assert_eq!(file.settings, LevelSettings::default());
    }

    // ── v3 forever-load fixture discipline (mirrors the v1/v2 fixtures) ─────

    /// The committed pre-P10.6 (schema v3) payload, load-tested forever.
    fn v3_fixture_bytes() -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v3.inf_lvl");
        std::fs::read(path).expect("committed v3 fixture is present")
    }

    /// Rebuild the exact schema-v3 `SceneFile` the v3 fixture was generated from,
    /// from the frozen [`EntityRecordV3`]/[`SceneFileV3`] types. Any change to the
    /// v3 layout breaks this (the provenance lock). Exercises the physics + actor
    /// slots v3 introduced so the frozen layout is genuinely covered.
    fn v3_reference() -> SceneFileV3 {
        let g = uuid::Uuid::from_u128;
        SceneFileV3 {
            schema_version: 3,
            title: "V3 Fixture Level".into(),
            entities: vec![
                EntityRecordV3 {
                    guid: g(0x3001),
                    name: "Ground".into(),
                    parent: None,
                    transform: EcsTransform {
                        translation: inf_ecs::math::Vec3d::ZERO,
                        rotation: inf_ecs::math::Vec3d::ZERO,
                        scale: inf_ecs::math::Vec3d::new(6.0, 1.0, 1.0),
                    },
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
                    rigid_body_2d: Some(RigidBody2D {
                        kind: BodyKind2D::Static,
                        ..Default::default()
                    }),
                    collider_2d: Some(Collider2D {
                        shape_kind: ColliderShape2DKind::Box,
                        half_extents: Vec2d::new(3.0, 0.5),
                        ..Default::default()
                    }),
                    character_controller_2d: None,
                    rigid_body_3d: None,
                    collider_3d: None,
                    character_controller_3d: None,
                    actor: None,
                },
                EntityRecordV3 {
                    guid: g(0x3002),
                    name: "Player".into(),
                    parent: None,
                    transform: EcsTransform::from_translation(glam::DVec3::new(1.5, 0.8, 0.0)),
                    visible: true,
                    mesh: None,
                    material: None,
                    light: None,
                    camera: None,
                    sprite: Some(Sprite {
                        size: Vec2d::new(0.8, 1.2),
                        ..Default::default()
                    }),
                    tilemap: None,
                    nine_slice: None,
                    text2d: None,
                    light_2d: None,
                    rigid_body_2d: Some(RigidBody2D {
                        kind: BodyKind2D::Kinematic,
                        fixed_rotation: true,
                        ..Default::default()
                    }),
                    collider_2d: None,
                    character_controller_2d: Some(CC2D::default()),
                    rigid_body_3d: None,
                    collider_3d: None,
                    character_controller_3d: None,
                    actor: Some(g(0x3ACC)),
                },
            ],
            settings: LevelSettings {
                gravity_2d: Vec2d::new(0.0, -20.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 120.0,
            },
        }
    }

    /// Write the committed v3 fixture from [`v3_reference`] under
    /// `INF_BLESS_FIXTURES=1`. This is the "temporary writer" the fixture
    /// provenance discipline calls for — it regenerates the frozen-layout bytes
    /// from the frozen types, then the reproducibility test locks them forever.
    #[test]
    fn bless_v3_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v3_reference(), bincode_config()).unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v3.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v3 fixture");
        eprintln!("blessed v3 fixture: {}", path.display());
    }

    #[test]
    fn v3_fixture_is_reproducible_and_genuinely_v3() {
        let bytes = v3_fixture_bytes();
        assert_eq!(bytes[0], 3, "fixture must be a genuine schema-v3 payload");
        let rebuilt = bincode::serde::encode_to_vec(v3_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v3 fixture must match our frozen v3 writer"
        );
    }

    #[test]
    fn v3_fixture_loads_forever_and_lifts_to_v4() {
        let file = decode(&v3_fixture_bytes()).expect("v3 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V3 Fixture Level");
        assert_eq!(file.entities.len(), 2);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();
        // v3 physics/actor data preserved through the v3→v4 lift.
        assert!(by_name("Ground").rigid_body_2d.is_some());
        assert!(by_name("Ground").collider_2d.is_some());
        assert!(by_name("Player").character_controller_2d.is_some());
        assert!(by_name("Player").actor.is_some());
        // v3 settings carry through.
        assert_eq!(file.settings.gravity_2d, Vec2d::new(0.0, -20.0));
        assert_eq!(file.settings.sim_hz, 120.0);
        // Every v4 slot defaulted on the old payload.
        for r in &file.entities {
            assert!(r.terrain.is_none());
            assert!(r.pcg_volume.is_none());
            // v5 anim/character slots also defaulted on the old payload.
            assert!(r.skeletal_mesh.is_none());
            assert!(r.anim_player.is_none());
            assert!(r.anim_state_machine.is_none());
            assert!(r.root_motion.is_none());
            assert!(r.attached_to.is_none());
        }
    }

    // ── v5 (P11.4) animation / character component persistence ─────────────

    /// FLIPPED (P11.4 · schema v5): a spawned `SkeletalMesh` + `AnimPlayer` +
    /// `AnimStateMachine` + `RootMotion` + `AttachedTo` now **persist** across
    /// save/load and are byte-identical on re-encode — the guard the P11.1..P11.3
    /// batches left as `skeletal_components_serde_round_trip` (component-only,
    /// no `.inf_lvl` slot). The v5 slots close the gap. The `AnimStateMachine`'s
    /// transient `runtime` state is `#[serde(skip)]`, so it must persist reset.
    #[test]
    fn anim_components_persist_across_save_load_v5() {
        use inf_ecs::components::{
            AnimPlayer, AnimStateMachine, AttachedTo, RootMotion, SkeletalMesh,
        };

        let skel_guid = uuid::Uuid::from_u128(0x11_5EE1);
        let mesh_guid = uuid::Uuid::from_u128(0x11_3E54);
        let clip_guid = uuid::Uuid::from_u128(0x11_C11B);
        let sm_guid = uuid::Uuid::from_u128(0x11_5A11);
        let target_guid = uuid::Uuid::from_u128(0x11_7A67);

        let mut doc = SceneDoc::new();
        let g = doc.create(SpawnKind::Empty, "Character", None);
        {
            let e = doc.entity_of(g).unwrap();
            let mut asm = AnimStateMachine {
                sm: Some(sm_guid),
                params_from_vars: true,
                ..Default::default()
            };
            // A non-default runtime must NOT reach disk (serde-skipped).
            asm.runtime.current = 2;
            asm.runtime.started = true;
            let w = doc.world_mut().world_mut();
            w.entity_mut(e).insert(SkeletalMesh {
                mesh: Some(mesh_guid),
                skeleton: Some(skel_guid),
            });
            w.entity_mut(e).insert(AnimPlayer {
                clip: Some(clip_guid),
                speed: 1.5,
                looping: false,
                ..Default::default()
            });
            w.entity_mut(e).insert(asm);
            w.entity_mut(e).insert(RootMotion::apply());
            w.entity_mut(e).insert(AttachedTo::new(
                target_guid,
                "hand_r",
                Vec3d::new(0.0, 1.0, 0.0),
            ));
            doc.world_mut().mark_dirty();
        }

        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0], 6,
            "anim content writes as a genuine schema-v6 file"
        );
        let mut loaded = SceneDoc::new();
        apply_to_doc(&mut loaded, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&loaded)).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "anim components save→load→save must be byte-identical (runtime skipped)"
        );

        // Every component survives with its values.
        let e2 = loaded.entity_of(g).expect("entity persists");
        let w = loaded.world().world();
        let sk = w.get::<SkeletalMesh>(e2).expect("skeletal_mesh persists");
        assert_eq!(sk.skeleton, Some(skel_guid));
        assert_eq!(sk.mesh, Some(mesh_guid));
        let ap = w.get::<AnimPlayer>(e2).expect("anim_player persists");
        assert_eq!(ap.clip, Some(clip_guid));
        assert_eq!(ap.speed, 1.5);
        assert!(!ap.looping);
        let asm = w
            .get::<AnimStateMachine>(e2)
            .expect("anim_state_machine persists");
        assert_eq!(asm.sm, Some(sm_guid));
        assert!(asm.params_from_vars);
        assert_eq!(
            asm.runtime,
            inf_ecs::components::SmRuntimeState::default(),
            "the transient runtime state is never persisted"
        );
        let rm = w.get::<RootMotion>(e2).expect("root_motion persists");
        assert_eq!(rm.mode, inf_ecs::components::RootMotionMode::ApplyToEntity);
        let at = w.get::<AttachedTo>(e2).expect("attached_to persists");
        assert_eq!(at.target, target_guid);
        assert_eq!(at.socket, "hand_r");
    }

    /// Delete → undo restores the full P11 animation/character component set (the
    /// v5 record slots feed Create/Delete snapshotting through [`record_of`]).
    #[test]
    fn delete_undo_restores_anim_components_v5() {
        use inf_ecs::components::{AnimStateMachine, RootMotion, SkeletalMesh};

        let sm_guid = uuid::Uuid::from_u128(0x11_DEAD);
        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Empty, "Character", None);
        {
            let e = doc.entity_of(g).unwrap();
            let w = doc.world_mut().world_mut();
            w.entity_mut(e).insert(SkeletalMesh {
                skeleton: Some(uuid::Uuid::from_u128(0x11_B0AE)),
                ..Default::default()
            });
            w.entity_mut(e).insert(AnimStateMachine {
                sm: Some(sm_guid),
                ..Default::default()
            });
            w.entity_mut(e).insert(RootMotion::apply());
            doc.world_mut().mark_dirty();
        }

        doc.edit_delete(&[g]);
        assert!(doc.entity_of(g).is_none(), "deleted");
        doc.undo();

        let e = doc.entity_of(g).expect("entity restored by undo");
        let w = doc.world().world();
        assert!(w.get::<SkeletalMesh>(e).is_some(), "skeletal_mesh restored");
        assert_eq!(
            w.get::<AnimStateMachine>(e)
                .expect("state machine restored")
                .sm,
            Some(sm_guid)
        );
        assert!(w.get::<RootMotion>(e).is_some(), "root_motion restored");
    }

    // ── v4 forever-load fixture discipline (mirrors the v1/v2/v3 fixtures) ───

    /// The committed pre-P11.4 (schema v4) payload, load-tested forever.
    fn v4_fixture_bytes() -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v4.inf_lvl");
        std::fs::read(path).expect("committed v4 fixture is present")
    }

    /// Rebuild the exact schema-v4 `SceneFile` the v4 fixture was generated from,
    /// from the frozen [`EntityRecordV4`]/[`SceneFileV4`] types (the provenance
    /// lock). Exercises the terrain + pcg slots v4 introduced so the frozen layout
    /// is genuinely covered.
    fn v4_reference() -> SceneFileV4 {
        use inf_ecs::components::{
            BodyKind3D, Collider3D, ColliderShape3DKind, PcgVolume, RigidBody3D,
        };
        let g = uuid::Uuid::from_u128;
        SceneFileV4 {
            schema_version: 4,
            title: "V4 Fixture Level".into(),
            entities: vec![
                EntityRecordV4 {
                    guid: g(0x4001),
                    name: "Ground".into(),
                    parent: None,
                    transform: EcsTransform::from_translation(glam::DVec3::new(0.0, -0.5, 0.0)),
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
                    rigid_body_3d: Some(RigidBody3D {
                        kind: BodyKind3D::Static,
                        ..Default::default()
                    }),
                    collider_3d: Some(Collider3D {
                        shape_kind: ColliderShape3DKind::Box,
                        half_extents: Vec3d::new(5.0, 0.5, 5.0),
                        ..Default::default()
                    }),
                    character_controller_3d: None,
                    actor: None,
                    terrain: None,
                    pcg_volume: None,
                },
                EntityRecordV4 {
                    guid: g(0x4002),
                    name: "Scatter".into(),
                    parent: None,
                    transform: EcsTransform::IDENTITY,
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
                    pcg_volume: Some(PcgVolume {
                        graph: Some(g(0x4ACC)),
                        extent: Vec2d::new(40.0, 40.0),
                        seed: 3,
                        ..Default::default()
                    }),
                },
            ],
            settings: LevelSettings::default(),
        }
    }

    /// Write the committed v4 fixture from [`v4_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline the fixture
    /// provenance rule calls for).
    #[test]
    fn bless_v4_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v4_reference(), bincode_config()).unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v4.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v4 fixture");
        eprintln!("blessed v4 fixture: {}", path.display());
    }

    #[test]
    fn v4_fixture_is_reproducible_and_genuinely_v4() {
        let bytes = v4_fixture_bytes();
        assert_eq!(bytes[0], 4, "fixture must be a genuine schema-v4 payload");
        let rebuilt = bincode::serde::encode_to_vec(v4_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v4 fixture must match our frozen v4 writer"
        );
    }

    #[test]
    fn v4_fixture_loads_forever_and_lifts_to_v5() {
        let file = decode(&v4_fixture_bytes()).expect("v4 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V4 Fixture Level");
        assert_eq!(file.entities.len(), 2);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();
        // v4 physics + pcg data preserved through the v4→v5 lift.
        assert!(by_name("Ground").rigid_body_3d.is_some());
        assert!(by_name("Ground").collider_3d.is_some());
        assert!(by_name("Scatter").pcg_volume.is_some());
        // Every v5 anim/character slot defaulted on the old payload.
        for r in &file.entities {
            assert!(r.skeletal_mesh.is_none());
            assert!(r.anim_player.is_none());
            assert!(r.anim_state_machine.is_none());
            assert!(r.root_motion.is_none());
            assert!(r.attached_to.is_none());
            // v6 joints/audio slots also defaulted on the old payload.
            assert!(r.joint_2d.is_none());
            assert!(r.joint_3d.is_none());
            assert!(r.audio_source.is_none());
            assert!(r.audio_listener.is_none());
        }
    }

    // ── v6 (P12.4) joints / spatial-audio component persistence ────────────

    use inf_ecs::components::{
        AudioListener, AudioSource, DistanceModel, Joint2D, Joint3D, JointKind2D, JointKind3D,
    };

    /// FLIPPED (P12.4 · schema v6): a spawned `Joint2D` / `Joint3D` (including the
    /// `#[reflect(ignore)]` `other` entity ref) + `AudioSource` (with its `clip`
    /// ref) + `AudioListener` now **persist** across save/load and are
    /// byte-identical on re-encode — the guards the P12.1..P12.3 batches left as
    /// `joint_3d_serde_round_trip_including_entity_ref` + `audio_components_serde_round_trip`
    /// (component-only, no `.inf_lvl` slot). The v6 slots close the gap.
    #[test]
    fn joints_and_audio_persist_across_save_load_v6() {
        let other_guid = uuid::Uuid::from_u128(0x12_0B01);
        let clip_guid = uuid::Uuid::from_u128(0x12_C11B);

        let mut doc = SceneDoc::new();
        // The "other" body a joint links to.
        let g_anchor = doc.create(SpawnKind::Empty, "Anchor", None);
        // A body with a 3D joint + a 2D joint + an audio emitter.
        let g = doc.create(SpawnKind::Empty, "Body", None);
        {
            let e = doc.entity_of(g).unwrap();
            let w = doc.world_mut().world_mut();
            w.entity_mut(e).insert(Joint3D {
                other: Some(other_guid),
                kind: JointKind3D::Revolute,
                axis: Vec3d::new(0.0, 0.0, 1.0),
                limits_enabled: true,
                limit_min: -1.5,
                limit_max: 1.5,
                motor_enabled: true,
                motor_target_vel: 8.0,
                ..Default::default()
            });
            w.entity_mut(e).insert(Joint2D {
                other: Some(other_guid),
                kind: JointKind2D::Distance,
                max_distance: 1.5,
                ..Default::default()
            });
            w.entity_mut(e).insert(AudioSource {
                clip: Some(clip_guid),
                bus: "sfx".into(),
                volume: 0.75,
                looping: true,
                spatial: true,
                distance_model: DistanceModel::Exponential,
                rolloff: 2.0,
                occlusion: true,
                autoplay: true,
                ..Default::default()
            });
            doc.world_mut().mark_dirty();
        }
        // A listener on a second entity.
        let g_listener = doc.create(SpawnKind::Empty, "Listener", None);
        {
            let e = doc.entity_of(g_listener).unwrap();
            doc.world_mut()
                .world_mut()
                .entity_mut(e)
                .insert(AudioListener { active: true });
            doc.world_mut().mark_dirty();
        }
        let _ = g_anchor;

        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(bytes1[0], 6, "joints/audio content writes a schema-v6 file");
        let mut loaded = SceneDoc::new();
        apply_to_doc(&mut loaded, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&loaded)).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "joints/audio save→load→save must be byte-identical"
        );

        // Every component survives with its values (incl. the joint entity ref).
        let e2 = loaded.entity_of(g).expect("body persists");
        let w = loaded.world().world();
        let j3 = w.get::<Joint3D>(e2).expect("joint_3d persists");
        assert_eq!(j3.other, Some(other_guid));
        assert_eq!(j3.kind, JointKind3D::Revolute);
        assert!(j3.motor_enabled);
        assert_eq!(j3.motor_target_vel, 8.0);
        let j2 = w.get::<Joint2D>(e2).expect("joint_2d persists");
        assert_eq!(j2.other, Some(other_guid));
        assert_eq!(j2.kind, JointKind2D::Distance);
        let src = w.get::<AudioSource>(e2).expect("audio_source persists");
        assert_eq!(src.clip, Some(clip_guid));
        assert!(src.autoplay && src.looping && src.occlusion);
        let le = loaded.entity_of(g_listener).unwrap();
        assert!(
            loaded
                .world()
                .world()
                .get::<AudioListener>(le)
                .expect("audio_listener persists")
                .active
        );
    }

    /// Delete → undo restores the P12 joints/audio component set (the v6 record
    /// slots feed Create/Delete snapshotting through [`record_of`]).
    #[test]
    fn delete_undo_restores_joints_and_audio_v6() {
        let clip = uuid::Uuid::from_u128(0x12_DEAD);
        let mut doc = SceneDoc::new();
        let g = doc.edit_create(SpawnKind::Empty, "Body", None);
        {
            let e = doc.entity_of(g).unwrap();
            let w = doc.world_mut().world_mut();
            w.entity_mut(e).insert(Joint3D {
                other: Some(uuid::Uuid::from_u128(0x12_A0)),
                kind: JointKind3D::Spherical,
                ..Default::default()
            });
            w.entity_mut(e).insert(AudioSource {
                clip: Some(clip),
                autoplay: true,
                ..Default::default()
            });
            doc.world_mut().mark_dirty();
        }
        doc.edit_delete(&[g]);
        assert!(doc.entity_of(g).is_none(), "deleted");
        doc.undo();
        let e = doc.entity_of(g).expect("entity restored by undo");
        let w = doc.world().world();
        assert_eq!(
            w.get::<Joint3D>(e).expect("joint_3d restored").kind,
            JointKind3D::Spherical
        );
        assert_eq!(
            w.get::<AudioSource>(e).expect("audio_source restored").clip,
            Some(clip)
        );
    }

    // ── v5 forever-load fixture discipline (mirrors the v1..v4 fixtures) ────

    /// Rebuild the exact schema-v5 `SceneFile` the v5 fixture was generated from,
    /// from the frozen [`EntityRecordV5`]/[`SceneFileV5`] types (the provenance
    /// lock). Exercises the anim/character slots v5 introduced so the frozen layout
    /// is genuinely covered.
    fn v5_reference() -> SceneFileV5 {
        use inf_ecs::components::{RootMotion, SkeletalMesh};
        let g = uuid::Uuid::from_u128;
        SceneFileV5 {
            schema_version: 5,
            title: "V5 Fixture Level".into(),
            entities: vec![
                EntityRecordV5 {
                    guid: g(0x5001),
                    name: "Ground".into(),
                    parent: None,
                    transform: EcsTransform::from_translation(glam::DVec3::new(0.0, -0.5, 0.0)),
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
                },
                EntityRecordV5 {
                    guid: g(0x5002),
                    name: "Character".into(),
                    parent: None,
                    transform: EcsTransform::from_translation(glam::DVec3::new(0.0, 0.9, 0.0)),
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
                    actor: Some(g(0x5ACC)),
                    terrain: None,
                    pcg_volume: None,
                    skeletal_mesh: Some(SkeletalMesh {
                        mesh: None,
                        skeleton: Some(g(0x5_5EE1)),
                    }),
                    anim_player: None,
                    anim_state_machine: None,
                    root_motion: Some(RootMotion::apply()),
                    attached_to: None,
                },
            ],
            settings: LevelSettings::default(),
        }
    }

    /// Write the committed v5 fixture from [`v5_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline).
    #[test]
    fn bless_v5_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v5_reference(), bincode_config()).unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v5.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v5 fixture");
        eprintln!("blessed v5 fixture: {}", path.display());
    }

    #[test]
    fn v5_fixture_is_reproducible_and_genuinely_v5() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v5.inf_lvl");
        if !path.exists() {
            eprintln!("SKIP: v5 fixture not blessed yet ({})", path.display());
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes[0], 5, "fixture must be a genuine schema-v5 payload");
        let rebuilt = bincode::serde::encode_to_vec(v5_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v5 fixture must match our frozen v5 writer"
        );
    }

    #[test]
    fn v5_fixture_loads_forever_and_lifts_to_v6() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v5.inf_lvl");
        if !path.exists() {
            eprintln!("SKIP: v5 fixture not blessed yet ({})", path.display());
            return;
        }
        let file = decode(&std::fs::read(&path).unwrap()).expect("v5 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V5 Fixture Level");
        assert_eq!(file.entities.len(), 2);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();
        // v5 anim/character data preserved through the v5→v6 lift.
        assert!(by_name("Character").skeletal_mesh.is_some());
        assert!(by_name("Character").root_motion.is_some());
        assert!(by_name("Character").actor.is_some());
        // Every v6 joints/audio slot defaulted on the old payload.
        for r in &file.entities {
            assert!(r.joint_2d.is_none());
            assert!(r.joint_3d.is_none());
            assert!(r.audio_source.is_none());
            assert!(r.audio_listener.is_none());
        }
    }
}
