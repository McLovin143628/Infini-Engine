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
    ActorClass, AlwaysLoaded, AnimPlayer, AnimStateMachine, AttachedTo, AudioListener, AudioSource,
    BlendMode, Buoyancy, Camera, CharacterController2D, CharacterController3D, Collider2D,
    Collider3D, Decal, Foliage, Joint2D, Joint3D, Light, Light2D, LightKind, Material, MeshRef,
    NineSlice, PcgVolume, RigidBody2D, RigidBody3D, RootMotion, SkeletalMesh, SkyAtmosphere,
    Spline, Sprite, StreamingSource, Terrain, Text2D, Tilemap, TimeOfDay, Transform, Visibility,
    Volume, VoxelVolume, WaterBody,
};
use inf_ecs::math::{Color, Vec2d, Vec3d};
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
///
/// * v7 — P13.4: [`MeshRef`] gained a mesh-**asset** GUID field
///   (`asset: Option<Uuid>`) so an entity can reference a `.inf_mesh` asset (the
///   virtualized-geometry gate scene). This changed `MeshRef`'s byte layout, so
///   the pre-v7 layout is frozen as [`MeshRefV6`] and every v1..v6 record carries
///   its `mesh` slot as `Option<MeshRefV6>`; the [`EntityRecordV6::into_current`]
///   hop lifts it with `asset: None`. No new entity slot was added — v7 differs
///   from v6 only inside `MeshRef`. Older v1..v6 payloads load with `asset`
///   defaulted to `None` (see [`decode`] + [`SceneFileV6`]).
///
/// * v8 — R-P0: `Light` gained `range` / `inner_cone_deg` / `outer_cone_deg` /
///   `cast_shadows`; `Material` gained `blend` / `alpha_cutoff`; the
///   [`EntityRecord`] appended four world-decoration slots — `decal`
///   ([`Decal`]), `volume` ([`Volume`]), `spline` ([`Spline`]) and `foliage`
///   ([`Foliage`]); and [`LevelSettings`] gained a `render`
///   ([`RenderSettingsRecord`]) block. `Light`/`Material` changed byte layout, so
///   the pre-v8 shapes are frozen as [`LightV7`] / [`MaterialV7`] (and the file
///   settings as [`LevelSettingsV7`]); every v1..v7 record carries its
///   `light`/`material` slots through those frozen types, and the
///   [`EntityRecordV7::into_current`] hop lifts them (new light/material fields at
///   their documented defaults) and appends the four v8 slots defaulted to
///   `None`. Older v1..v7 payloads load with all the new fields/slots defaulted
///   (see [`decode`] + [`SceneFileV7`]).
///
/// * v9 — P16.3: [`Terrain`] gained an `asset: Option<Uuid>` reference to a
///   `.inf_terrain` **streaming asset** (a header + tile directory + 16-byte-
///   aligned per-tile blobs across an LOD pyramid, cooked uncompressed so a
///   runtime pages tiles out of an mmap'd pack). `None` — what the editor still
///   writes — means the inline `data` is the terrain's only authority, so an
///   older level's meaning is preserved exactly. This changed `Terrain`'s byte
///   layout, so the pre-v9 shape is frozen as [`TerrainV8`] and every v4..v8
///   record carries its `terrain` slot as `Option<TerrainV8>`; the
///   [`EntityRecordV8::into_current`] hop lifts it with `asset: None`. No new
///   entity slot was added — v9 differs from v8 only inside `Terrain` (see
///   [`decode`] + [`SceneFileV8`]).
///
/// * v10 — P16.5: the [`EntityRecord`] appended two **world-partition** slots —
///   `streaming_source` ([`StreamingSource`], the sim-side driver of cell
///   residency) and `always_loaded` ([`AlwaysLoaded`], the never-streamed
///   marker) — and [`LevelSettings`] gained a `partition`
///   ([`PartitionSettings`]) block. Neither `Light`/`Material`/`Terrain` nor any
///   existing slot moved, so the pre-v10 shapes are frozen as
///   [`LevelSettingsV9`] / [`EntityRecordV9`] and the
///   [`EntityRecordV9::into_current`] hop lifts them with the two slots `None`
///   and partitioning **off** — exactly what a pre-v10 level meant. Older
///   v1..v9 payloads load unchanged (see [`decode`] + [`SceneFileV9`]).
///
///   Second bump in one phase, deliberately: P16.3 shipped v9 before partition
///   metadata was designed, and retro-fitting it into v9 would mean re-blessing
///   bytes that are already committed and already load.
/// * v11 — P17.1: the [`EntityRecord`] appends the two **sky-authority** slots —
///   `time_of_day` ([`TimeOfDay`], the world clock the sun and moon are a pure
///   function of) and `sky_atmosphere` ([`SkyAtmosphere`], how that sun lights the
///   world and tints the gradient) — retiring the renderer's compile-time
///   `SUN_DIR`. [`LevelSettings`] is **untouched**, so only the entity record
///   freezes: the pre-v11 shape is [`EntityRecordV10`], and the
///   [`EntityRecordV10::into_v11`] hop lifts it with both slots `None` — a
///   level with no clock, which is exactly what every pre-v11 level was and which
///   the scene projectors render under the retired constant's direction. Older
///   v1..v10 payloads load unchanged (see [`decode`] + [`SceneFileV10`]).
///
/// * v12 — P17.2: [`SkyAtmosphere`] grew the **physical-atmosphere block** — the
///   physical-sky switch and its LUT knobs (`physical`, `sky_intensity`,
///   `turbidity`, `mie_anisotropy`), the sun/moon disc angular diameters
///   (`sun_disc_deg`, `moon_disc_deg`), `star_intensity`, the gradient
///   `tint_strength`, `aerial_perspective`, and height fog in **SI metres**
///   (`fog_density`, `fog_falloff`, `fog_height`, `fog_color`). That changed the
///   component's byte layout, so the pre-v12 shape is frozen as
///   [`SkyAtmosphereV11`] and the pre-v12 entity record as [`EntityRecordV11`],
///   which carries `sky_atmosphere` as `Option<SkyAtmosphereV11>` exactly as
///   v4..v8 carry `terrain` as `Option<TerrainV8>`; the
///   [`EntityRecordV11::into_current`] hop lifts it. **No entity slot was added
///   or moved** and [`LevelSettings`] is untouched — this is the `TerrainV8`
///   shape of bump, not the `EntityRecordV10` shape. Older v1..v11 payloads load
///   unchanged, with the 13 new fields at their live `SkyAtmosphere::default()`
///   values: a gradient sky and no fog, which is what a v11 level meant (see
///   [`decode`] + [`SceneFileV11`]).
///
///   **Why bump at all, when every new field is `#[serde(default)]`?** Because
///   bincode is **not self-describing**: the decoder reads a fixed field count
///   positionally, with no names or lengths on the wire, so a v11 payload fed to
///   the grown struct would keep reading past the end of its `SkyAtmosphere` and
///   into the next record's bytes. `#[serde(default)]` only rescues the
///   self-describing codecs (the JSON/TOML sidecars, the Details grid). Same root
///   cause as the house law that `skip_serializing_if` desyncs bincode; the
///   frozen-record ladder exists for exactly this case.
///
/// * v13 — P17.3: [`SkyAtmosphere`] grew the **volumetric-cloud block** — the
///   cloud switch (`clouds_enabled`), the weather-field shape (`cloud_coverage`,
///   `cloud_type`, `cloud_detail`, `cloud_seed`), the layer slab in **SI metres**
///   (`cloud_bottom`, `cloud_top`), the optics (`cloud_density`, `cloud_phase_g`,
///   `cloud_shadow`, `cloud_ambient`, `cloud_color`) and the wind in **m/s**
///   (`cloud_wind_x`, `cloud_wind_z`). Same shape of bump as v12, for the same
///   reason: bincode is **positional**, so *growing a component is a wire-format
///   change* even though every new field is `#[serde(default)]` — a v12 payload
///   fed to the grown struct would read past the end of its `SkyAtmosphere` and
///   into the next record. So the pre-v13 shape is frozen as [`SkyAtmosphereV12`]
///   and the pre-v13 entity record as [`EntityRecordV12`], which carries
///   `sky_atmosphere` as `Option<SkyAtmosphereV12>` exactly as [`EntityRecordV11`]
///   carries it as `Option<SkyAtmosphereV11>`; the
///   [`EntityRecordV12::into_v13`] hop lifts it. **Only the component's shape
///   changed** — no entity slot was added or moved and [`LevelSettings`] is
///   untouched — so older v1..v12 payloads load unchanged, with the 14 new fields
///   at their live `SkyAtmosphere::default()` values. That means
///   `clouds_enabled: false`: a v12 level had **no clouds**, which is exactly what
///   a v12 level meant (see [`decode`] + [`SceneFileV12`]).
///
/// * v15 — P19.1: every terrain **tile** gained its sparse erosion data-map layer
///   (flow / deposition / wear). No component field was added and none moved —
///   the change is one level deeper, inside [`inf_terrain::TerrainTile`]'s wire
///   form — but it is the **same law** as v12/v13 for the same reason: bincode is
///   positional, so an extra length-prefixed layer inside a tile is a wire-format
///   change, and a v14 payload fed to the grown tile would read past the end of
///   its heights and into the next tile. So the pre-v15 heightfield is frozen as
///   [`inf_terrain::TerrainDataFrozenV1`], the pre-v15 component as [`TerrainV14`],
///   and the pre-v15 entity record as [`EntityRecordV14`] (which carries
///   `terrain` as `Option<TerrainV14>`, exactly as v4..v8 carry it as
///   `Option<TerrainV8>`). v1..v14 payloads load unchanged, with every tile's
///   maps lifted to **empty** — never eroded, which is exactly what a v14 level
///   meant. An un-eroded terrain pays one zero-length count per tile.
///
/// * v16 — P19.2: every terrain **tile** gained its sparse per-sample **biome id**
///   layer (`Vec<u8>`; empty means every sample is
///   [`inf_terrain::UNASSIGNED_BIOME`]), and [`Terrain`] itself gained a
///   `biome_set: Option<Uuid>` reference to the `.inf_biomes` vocabulary those ids
///   name.
///
///   **The tile layer is what forces the bump**, not the component field: bincode
///   is positional, so an extra length-prefixed layer *inside a tile* is a
///   wire-format change even though the field is `#[serde(default)]` — a v15
///   payload fed to the grown tile reads past the end of its data maps and into
///   the next tile. Fourth instance of the same law (v12, v13 and v15 were the
///   others). `biome_set` alone would have been an ordinary append at the tail of
///   one component; it rides along once the bump is unavoidable.
///
///   So the pre-v16 heightfield is frozen as
///   [`inf_terrain::TerrainDataFrozenV2`], the pre-v16 component as [`TerrainV15`]
///   (which has **no** `biome_set` field — that is precisely what v16 added), and
///   the pre-v16 entity record as [`EntityRecordV15`] (which carries `terrain` as
///   `Option<TerrainV15>`, exactly as v9..v14 carry it as `Option<TerrainV14>`);
///   the [`TerrainV14::into_v15`] hop feeds the chained ladder. v1..v15 payloads
///   load unchanged, with every tile's biome ids lifted to **empty** and
///   `biome_set: None` — an unpainted terrain with no biome vocabulary, which is
///   exactly what a v15 level meant. An unpainted terrain pays one zero-length
///   count per tile plus one discriminant byte for the `None` biome set.
///
/// **P19.3 bumped nothing.** The biome→PCG binding gave [`Terrain`] a
/// `biome_population: Vec<ScatteredInstance>`, but it is `#[serde(skip)]` — a
/// derived cache rebuilt by the editor's evaluate command and by the player on
/// level load — so it is **wire-neutral** and every ladder rung below stays
/// byte-identical. Same precedent as `PcgVolume::evaluated`, and the reason the
/// schema stays at 16: only what reaches the bytes can force a bump.
/// * **v17** — P20.1: the entity record appends the **water** slot — `water_body`
///   ([`WaterBody`]: an ocean, a lake or a spline river, carrying its Gerstner
///   wave state, its river cross-section and its shading). No component changed
///   shape and [`LevelSettings`] is untouched, so this is the
///   [`EntityRecordV10`] *shape* of bump (a new slot at the tail), not the
///   [`EntityRecordV14`] one (a component that grew). The pre-v17 entity record
///   is frozen as [`EntityRecordV16`] and lifts with `water_body: None` — a level
///   with no water, which is exactly what every pre-v17 level was.
///
///   **A river's centreline is the `Spline` on the same entity**, not a
///   reference: composition rather than a GUID, so v17 adds no asset edge, the
///   cook's dependency closure is unchanged, and there is no dangling-reference
///   advisory to write. The wire price to a water-free level is **one
///   discriminant byte per entity** — the same price every additive slot since v8
///   has paid.
///
/// * **v18** — P20.2: the entity record appends the **buoyancy** slot —
///   `buoyancy` ([`Buoyancy`]: opt-in flotation and hydrodynamic drag for a
///   dynamic 3D body). No component changed shape and [`LevelSettings`] is
///   untouched, so this is again the [`EntityRecordV10`] *shape* of bump (a new
///   slot at the tail), not the [`EntityRecordV14`] one (a component that grew).
///   The pre-v18 entity record is frozen as [`EntityRecordV17`] and lifts with
///   `buoyancy: None` — a level in which nothing floats, which is exactly what
///   every pre-v18 level was.
///
///   **Why the component exists at all, rather than flotation being a rule:**
///   it is opt-in because a default-on rule (every dynamic body floats, its
///   density read from its collider) would have silently rewritten the physics of
///   every dynamic body in any level that gained water — and
///   `Collider3D::density` defaults to `1.0`, which is rapier's mass placeholder
///   and not a material density, so under that rule essentially every existing
///   body would bob like a cork on a millimetre of draught. The wire price to a
///   level where nothing floats is **one discriminant byte per non-buoyant
///   entity** — the same price every additive slot since v8 has paid.
///
/// * **v19** — P21.1: the entity record appends the **volumetric-terrain** slot —
///   `voxel_volume` ([`VoxelVolume`]: a sparse SDF voxel volume — the caves,
///   tunnels and excavations that *locally extend* the heightfield terrain). No
///   component changed shape and [`LevelSettings`] is untouched, so this is once
///   again the [`EntityRecordV10`] *shape* of bump (a new slot at the tail), not
///   the [`EntityRecordV14`] one (a component that grew). The pre-v19 entity
///   record is frozen as [`EntityRecordV18`] and lifts with `voxel_volume: None`
///   — a level whose ground is a heightfield and nothing else, which is exactly
///   what every pre-v19 level was.
///
///   **The planet-scale base stays a heightfield**, deliberately: the P16 clipmap
///   economics are unbeatable at that scale, and v19 does not voxelize the world.
///   Volumetric capability arrives as chunk volumes that override and extend the
///   heightfield *locally*, which is the hybrid every serious open-world engine
///   settles on. In particular **no tile layout moved**, so unlike v15 and v16
///   this bump costs the frozen-terrain ladder nothing.
///
///   A [`VoxelVolume`] is a *reference plus its two authored knobs* — the chunks
///   live in the `.inf_voxel` asset, which versions itself, and there is no
///   inline `data` field for them (the asymmetry with [`Terrain`] is deliberate:
///   `Terrain` carries one because it predates streaming, and a voxel volume has
///   never had a pre-streaming form to keep loading). So the slot adds exactly one
///   new edge to the cook's dependency closure — that GUID — and the component
///   itself can never be the reason a *future* schema has to move: growing it
///   would cost another bump in both codec mirrors, which is why its three fields
///   are frozen as shipped.
///
///   The wire price to a level with no volumes is **one discriminant byte per
///   entity** — the same price every additive slot since v8 has paid.
pub const SCHEMA_VERSION: u32 = 19;

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
    /// Renderer HDR / post / lighting configuration (schema v8). Additive field:
    /// `#[serde(default)]` → [`RenderSettingsRecord::default`], which mirrors
    /// `inf_render::RenderSettings::default()` field-for-field, so a pre-v8 level
    /// (and every existing fixture) loads with the stable default look.
    #[serde(default)]
    pub render: RenderSettingsRecord,
    /// World-partition / level-streaming configuration (schema v10). Additive
    /// field: `#[serde(default)]` → [`PartitionSettings::default`], whose
    /// `enabled` is `false`, so every pre-v10 level (and every existing fixture)
    /// keeps cooking and loading as one document.
    #[serde(default)]
    pub partition: PartitionSettings,
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
            render: RenderSettingsRecord::default(),
            partition: PartitionSettings::default(),
        }
    }
}

/// World-partition configuration (schema v10) — a **flat, fully-explicit**
/// mirror of `inf_scene::PartitionSettings`, field-for-field and default-for-
/// default, kept here for the same reason [`RenderSettingsRecord`] is: this Ring-1
/// codec must not depend on the Ring-0 runtime reader, and bincode is not
/// self-describing, so the two shapes staying identical *is* the wire contract.
///
/// The cross-check is the `.inf_lvl` cross-decode test in `inf-scene` (which
/// parses editor-written bytes field-for-field), plus
/// [`partition_settings_mirror_matches_the_runtime_defaults`] below.
///
/// ## What each field means
///
/// * `enabled` — off by default; a level only partitions when an author says so.
/// * `cell_size_m` — the square grid cell edge, metres.
/// * `activation_radius_m` — how close a streaming source must come to a cell
///   before its entities **spawn**. This is sim-visible: it decides what exists.
/// * `prefetch_margin_m` — extra metres within which a cell may be *loaded*
///   ahead of need. This is **not** sim-visible: a cell that reaches its
///   activation step unloaded blocks the step, so the margin buys latency and
///   can never move a simulation result.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PartitionSettings {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_cell_size_m")]
    pub cell_size_m: f64,
    #[serde(default = "default_activation_radius_m")]
    pub activation_radius_m: f64,
    #[serde(default = "default_prefetch_margin_m")]
    pub prefetch_margin_m: f64,
}

fn default_cell_size_m() -> f64 {
    256.0
}
fn default_activation_radius_m() -> f64 {
    256.0
}
fn default_prefetch_margin_m() -> f64 {
    256.0
}

impl Default for PartitionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            cell_size_m: default_cell_size_m(),
            activation_radius_m: default_activation_radius_m(),
            prefetch_margin_m: default_prefetch_margin_m(),
        }
    }
}

/// Persisted renderer HDR / post / lighting settings (schema v8). A **flat,
/// fully-explicit** mirror of the fields of `inf_render::RenderSettings` (and its
/// nested `BloomSettings` / `SsaoSettings` / `ShadowSettings` / `GiSettings`) that
/// a level authors; the host applies it to the live `RenderSettings` at load.
/// Kept here (not a dependency on `inf-render`) so this Ring-1 codec, its Ring-0
/// runtime mirror (`inf_scene::RenderSettingsRecord`), and the CLI stay
/// wgpu-free — the field defaults below are asserted against `inf-render`.
///
/// Every default equals `inf_render::RenderSettings::default()` field-for-field
/// (sourced from `crates/inf-render/src/settings.rs`):
/// * `exposure = 1.0` (settings.rs `RenderSettings::default`, l.221)
/// * `dither = true` (l.222)
/// * `bloom_enabled = false`, `bloom_threshold = 1.0`, `bloom_knee = 0.5`,
///   `bloom_intensity = 0.06` (`BloomSettings::default`, l.24-27)
/// * `ssao_enabled = false`, `ssao_radius = 0.6`, `ssao_intensity = 1.0`,
///   `ssao_bias = 0.025` (`SsaoSettings::default`, l.48-51)
/// * `taa = false` (l.225)
/// * `shadows_enabled = false`, `shadows_max_distance = 60.0`
///   (`ShadowSettings::default`, l.133/136)
/// * `gi_enabled = false`, `gi_intensity = 1.0` (`GiSettings::default`, l.171/174)
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
    /// A heightfield terrain (paged heights + splat weights + erosion data maps +
    /// material layers). `TerrainData`'s manual serde keeps unpainted, un-eroded
    /// tiles byte-stable.
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
    // ── v10 (P16.5) world-partition components ────────────────────────────
    /// Marks this entity as a **streaming source**: world-partition cell
    /// residency is computed from its position at the fixed-step boundary. The
    /// editor stays single-document, so this only takes effect in PIE / a
    /// cooked run.
    #[serde(default)]
    pub streaming_source: Option<StreamingSource>,
    /// Marks this entity as never-streamed: it cooks into the partition's
    /// persistent cell and exists for the whole run.
    #[serde(default)]
    pub always_loaded: Option<AlwaysLoaded>,
    // ── v11 (P17.1) sky-authority components ──────────────────────────────
    /// The level's world clock. At most one entity should carry it; the
    /// resolution rule (lowest `Guid` wins) lives in `inf_ecs::sky`, shared by
    /// both scene projectors so they can never disagree about which one it is.
    #[serde(default)]
    pub time_of_day: Option<TimeOfDay>,
    /// How the clock's sun and moon light the world and tint the sky gradient.
    /// Sits on the same entity as `time_of_day`.
    #[serde(default)]
    pub sky_atmosphere: Option<SkyAtmosphere>,
    // ── v17 (P20.1) water ─────────────────────────────────────
    /// An ocean, a lake or a spline river. A `River` reads the `spline` slot on
    /// **this same entity** for its centreline — no reference to resolve, so no
    /// cook edge and no dangling-reference advisory.
    #[serde(default)]
    pub water_body: Option<WaterBody>,
    // ── v18 (P20.2) buoyancy ──────────────────────────────────────────────
    /// Opt-in flotation + hydrodynamic drag for a dynamic 3D body. Absent means
    /// the body ignores water, which is what every pre-v18 level meant — see the
    /// [`SCHEMA_VERSION`] ladder for why this is a component rather than a rule
    /// applied to every `RigidBody3D`.
    #[serde(default)]
    pub buoyancy: Option<Buoyancy>,
    // ── v19 (P21.1) volumetric terrain ────────────────────────────────────
    /// A sparse SDF voxel volume (caves / tunnels / excavations) that locally
    /// extends the heightfield terrain. The chunks live in the `.inf_voxel` this
    /// points at — the component is that reference plus its two authored knobs,
    /// so a level never carries volumetric data inline and the slot's whole cost
    /// to a volume-free level is its `None` discriminant.
    #[serde(default)]
    pub voxel_volume: Option<VoxelVolume>,
}

/// The pre-v8 `Light` byte layout (schema v8 froze this when `Light` gained its
/// `range` / cone / `cast_shadows` fields). Every frozen entity record (v1..v7)
/// carries its `light` slot as `Option<LightV7>`; [`LightV7::into_current`] lifts
/// it to the live [`Light`] with the new fields at their documented defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LightV7 {
    pub kind: LightKind,
    pub color: Color,
    pub intensity: f32,
}

impl LightV7 {
    /// Lift to the current [`Light`] (range unbounded, default cones, casts).
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
}

/// The pre-v8 `Material` byte layout (schema v8 froze this when `Material` gained
/// its `blend` / `alpha_cutoff` fields). Every frozen entity record (v1..v7)
/// carries its `material` slot as `Option<MaterialV7>`; [`MaterialV7::into_current`]
/// lifts it to the live [`Material`] with the new fields at their defaults.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MaterialV7 {
    pub base_color: Color,
    #[serde(default)]
    pub metallic: f32,
    #[serde(default)]
    pub roughness: f32,
    #[serde(default)]
    pub emissive: Color,
}

impl MaterialV7 {
    /// Lift to the current [`Material`] (opaque blend, 0.5 alpha cutoff).
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
}

impl Default for MaterialV7 {
    fn default() -> Self {
        // Mirrors the pre-v8 `Material::default()` exactly (byte-stable fixtures).
        Self {
            base_color: Color::new(0.8, 0.8, 0.8, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            emissive: Color::new(0.0, 0.0, 0.0, 1.0),
        }
    }
}

/// The pre-v8 file-level settings byte layout (schema v8 froze this when
/// [`LevelSettings`] gained its `render` block). Frozen entity/file records
/// (v3..v7) carry `settings` as [`LevelSettingsV7`]; [`LevelSettingsV7::into_v9`]
/// lifts it into the next frozen shape with a default [`RenderSettingsRecord`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LevelSettingsV7 {
    #[serde(default)]
    pub gravity_2d: Vec2d,
    #[serde(default = "default_gravity_3d")]
    pub gravity_3d: Vec3d,
    #[serde(default = "default_sim_hz")]
    pub sim_hz: f64,
}

impl LevelSettingsV7 {
    /// Lift to the frozen **v8/v9** shape ([`LevelSettingsV9`]) with a default
    /// render block. The ladder stops here rather than jumping to the live type,
    /// so the v7 → v10 path is the composition of two documented one-version
    /// hops instead of one hop that has to be rewritten on every future bump.
    fn into_v9(self) -> LevelSettingsV9 {
        LevelSettingsV9 {
            gravity_2d: self.gravity_2d,
            gravity_3d: self.gravity_3d,
            sim_hz: self.sim_hz,
            render: RenderSettingsRecord::default(),
        }
    }
}

impl Default for LevelSettingsV7 {
    fn default() -> Self {
        Self {
            gravity_2d: Vec2d::ZERO,
            gravity_3d: default_gravity_3d(),
            sim_hz: default_sim_hz(),
        }
    }
}

/// The **pre-v7** `MeshRef` byte layout (P13.4 froze this when `MeshRef` gained
/// its `asset: Option<Uuid>` field). Every frozen entity record (v1..v6) carries
/// its `mesh` slot as `Option<MeshRefV6>` so the committed v1..v6 fixtures — and
/// any level saved before P13.4 — decode with their original bytes; the
/// `into_current` hop lifts it to the live [`MeshRef`] with `asset: None`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeshRefV6 {
    pub primitive: inf_ecs::components::Primitive,
}

impl MeshRefV6 {
    /// Lift to the current [`MeshRef`] (no asset reference — pre-v7 levels never
    /// carried one).
    fn into_current(self) -> MeshRef {
        MeshRef {
            primitive: self.primitive,
            asset: None,
        }
    }
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
    pub mesh: Option<MeshRefV6>,
    pub material: Option<MaterialV7>,
    pub light: Option<LightV7>,
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
    pub mesh: Option<MeshRefV6>,
    pub material: Option<MaterialV7>,
    pub light: Option<LightV7>,
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
            settings: LevelSettingsV7::default(),
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
    pub mesh: Option<MeshRefV6>,
    pub material: Option<MaterialV7>,
    pub light: Option<LightV7>,
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
    pub settings: LevelSettingsV7,
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
    pub mesh: Option<MeshRefV6>,
    pub material: Option<MaterialV7>,
    pub light: Option<LightV7>,
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
    pub terrain: Option<TerrainV8>,
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
    pub settings: LevelSettingsV7,
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
    pub mesh: Option<MeshRefV6>,
    pub material: Option<MaterialV7>,
    pub light: Option<LightV7>,
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
    pub terrain: Option<TerrainV8>,
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
    /// Lift a v5 record to the **v6** shape: the four P12 joints/audio slots
    /// default to `None`.
    fn into_v6(self) -> EntityRecordV6 {
        EntityRecordV6 {
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
    pub settings: LevelSettingsV7,
}

impl SceneFileV5 {
    /// Lift a v5 file to the **v6** shape (settings carry through).
    fn into_v6(self) -> SceneFileV6 {
        SceneFileV6 {
            schema_version: 6,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV5::into_v6)
                .collect(),
            settings: self.settings,
        }
    }
}

/// A schema-**v6** [`EntityRecord`] (pre-P13.4) — the exact byte layout written by
/// P12.4..P13.3 editors (all component slots through the P12 joints/audio, with the
/// pre-v7 [`MeshRefV6`] mesh slot). Frozen forever so the committed v6 fixture (and
/// any level saved before P13.4) loads. v7 changed only `MeshRef` (added `asset`),
/// so this record differs from the live [`EntityRecord`] **only** in its `mesh` type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV6 {
    pub guid: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    pub transform: Transform,
    pub visible: bool,
    pub mesh: Option<MeshRefV6>,
    pub material: Option<MaterialV7>,
    pub light: Option<LightV7>,
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
    pub terrain: Option<TerrainV8>,
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
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
}

impl EntityRecordV6 {
    /// Lift a v6 record to the **v7** shape: the `mesh` slot's pre-v7
    /// [`MeshRefV6`] gains a `None` asset reference; every other slot (including
    /// the frozen `material`/`light`) carries through unchanged (v7 added no new
    /// entity slots — only the `MeshRef` layout changed).
    fn into_v7(self) -> EntityRecordV7 {
        EntityRecordV7 {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh.map(MeshRefV6::into_current),
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
            joint_2d: self.joint_2d,
            joint_3d: self.joint_3d,
            audio_source: self.audio_source,
            audio_listener: self.audio_listener,
        }
    }
}

/// A schema-v6 [`SceneFile`] (frozen layout for legacy decode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV6 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV6>,
    #[serde(default)]
    pub settings: LevelSettingsV7,
}

impl SceneFileV6 {
    /// Lift a v6 file to the **v7** shape (settings carry through).
    fn into_v7(self) -> SceneFileV7 {
        SceneFileV7 {
            schema_version: 7,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV6::into_v7)
                .collect(),
            settings: self.settings,
        }
    }
}

/// A schema-**v7** [`EntityRecord`] (pre-R-P0) — the exact byte layout written by
/// P13.4..P14 editors (all component slots through the P12 joints/audio, with the
/// live [`MeshRef`] mesh slot, but the pre-v8 [`MaterialV7`] / [`LightV7`] slots).
/// Frozen forever so the committed v7 fixture (and any level saved before R-P0)
/// loads. v8 changed only `Light`/`Material` (added fields) and appended the four
/// world-decoration slots, so this record differs from the live [`EntityRecord`]
/// only in its `material`/`light` types and the absence of those four slots.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV7 {
    pub guid: Uuid,
    pub name: String,
    pub parent: Option<Uuid>,
    pub transform: Transform,
    pub visible: bool,
    pub mesh: Option<MeshRef>,
    pub material: Option<MaterialV7>,
    pub light: Option<LightV7>,
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
    pub terrain: Option<TerrainV8>,
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
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
}

impl EntityRecordV7 {
    /// Lift a v7 record to the **v8** shape: `material`/`light` gain their new v8
    /// fields at the documented defaults ([`MaterialV7::into_current`] /
    /// [`LightV7::into_current`]); the four world-decoration slots default to
    /// `None`; every other slot (terrain included — still the frozen
    /// [`TerrainV8`] at this hop) carries through unchanged.
    fn into_v8(self) -> EntityRecordV8 {
        EntityRecordV8 {
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
            terrain: self.terrain,
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

/// A schema-v7 [`SceneFile`] (frozen layout for legacy decode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV7 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV7>,
    #[serde(default)]
    pub settings: LevelSettingsV7,
}

impl SceneFileV7 {
    /// Lift a v7 file to the **v8** shape (the frozen settings lift to the live
    /// [`LevelSettings`] with a default render block).
    fn into_v8(self) -> SceneFileV8 {
        SceneFileV8 {
            schema_version: 8,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV7::into_v8)
                .collect(),
            settings: self.settings.into_v9(),
        }
    }
}

/// The **pre-v9** `Terrain` byte layout (schema v9 froze this when [`Terrain`]
/// gained its `asset` reference to a `.inf_terrain` streaming asset). Every
/// frozen v4..v8 record carries `terrain` as `Option<TerrainV8>`;
/// [`TerrainV8::into_current`] lifts it.
///
/// The fields mirror the live component one-for-one **including their
/// `#[serde(default)]` markers**, so this record decodes every partial payload the
/// live one did (bincode ignores defaults on the write side, but the TOML/JSON
/// dual-format path does not).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerrainV8 {
    #[serde(default = "default_terrain_mps")]
    pub meters_per_sample: f64,
    #[serde(default = "default_terrain_resolution")]
    pub tile_resolution: u32,
    /// The **pre-v15** heightfield shape — see [`TerrainV14`]. A v4..v8 payload's
    /// tiles have no data maps, so the frozen wire type is what reads them.
    #[serde(default)]
    pub data: inf_terrain::TerrainDataFrozenV1,
    #[serde(default = "inf_ecs::components::default_terrain_layers")]
    pub layers: [inf_ecs::components::TerrainLayer; inf_ecs::components::TERRAIN_LAYERS],
    #[serde(default = "default_macro_variation")]
    pub macro_variation: f64,
}

/// The **pre-v15** `Terrain` byte layout (schema v15 froze this when P19.1 gave
/// every terrain tile its sparse erosion **data-map** layer). Frozen entity
/// records v9..v14 carry `terrain` as `Option<TerrainV14>`;
/// [`TerrainV14::into_current`] lifts it.
///
/// Only the heightfield's *tile* layout changed — no field was added to the
/// component and none moved — but bincode is positional, so an extra
/// length-prefixed layer inside each tile is a wire-format change all the same:
/// a v14 payload fed to the grown tile would read past the end of its heights and
/// into the next tile. Same law as v12/v13, one level deeper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerrainV14 {
    #[serde(default = "default_terrain_mps")]
    pub meters_per_sample: f64,
    #[serde(default = "default_terrain_resolution")]
    pub tile_resolution: u32,
    #[serde(default)]
    pub data: inf_terrain::TerrainDataFrozenV1,
    #[serde(default = "inf_ecs::components::default_terrain_layers")]
    pub layers: [inf_ecs::components::TerrainLayer; inf_ecs::components::TERRAIN_LAYERS],
    #[serde(default = "default_macro_variation")]
    pub macro_variation: f64,
    #[serde(default)]
    pub asset: Option<Uuid>,
}

impl TerrainV14 {
    /// Lift to the live [`Terrain`]: every tile's data maps come up **empty**
    /// (never eroded) and its biome ids likewise, with no biome vocabulary
    /// (`biome_set: None`) — exactly what a pre-P19.1 level meant.
    pub fn into_current(self) -> Terrain {
        Terrain {
            meters_per_sample: self.meters_per_sample,
            tile_resolution: self.tile_resolution,
            data: self.data.into_current(),
            layers: self.layers,
            macro_variation: self.macro_variation,
            asset: self.asset,
            biome_set: None,
            biome_population: Vec::new(),
        }
    }

    /// Lift a v14 terrain to the **v15** shape — the one hop the v14 → v15 record
    /// upgrade needs, since v15's frozen record now carries a [`TerrainV15`].
    ///
    /// Lossless: v14's tiles are a strict subset of v15's (heights + weights, no
    /// data maps), [`inf_terrain::TerrainDataFrozenV1::into_v2`] lifts them with
    /// their maps empty — which is what a v14 payload meant — and every scalar
    /// carries through untouched. The `biome_set` field does not exist on either
    /// side, so nothing is invented here either.
    pub fn into_v15(self) -> TerrainV15 {
        TerrainV15 {
            meters_per_sample: self.meters_per_sample,
            tile_resolution: self.tile_resolution,
            data: self.data.into_v2(),
            layers: self.layers,
            macro_variation: self.macro_variation,
            asset: self.asset,
        }
    }

    /// Project a live [`Terrain`] back onto the frozen shape (the downgrade-bless
    /// path that regenerates old fixtures). The data maps have no v14 home and
    /// are dropped — as are P19.2's biome ids and `biome_set` — a deliberately
    /// lossy direction.
    pub fn from_current(t: Terrain) -> Self {
        Self {
            meters_per_sample: t.meters_per_sample,
            tile_resolution: t.tile_resolution,
            data: inf_terrain::TerrainDataFrozenV1::from_current(&t.data),
            layers: t.layers,
            macro_variation: t.macro_variation,
            asset: t.asset,
        }
    }
}

/// The **pre-v16** `Terrain` byte layout (schema v16 froze this when P19.2 gave
/// every terrain tile its sparse per-sample **biome id** layer). Frozen entity
/// record [`EntityRecordV15`] carries `terrain` as `Option<TerrainV15>`;
/// [`TerrainV15::into_current`] lifts it, and [`TerrainV14::into_v15`] is the rung
/// below it in the chained ladder.
///
/// Two things changed at v16 and this record is the negative image of both. The
/// *tile* grew a length-prefixed `Vec<u8>` of biome ids — and bincode is
/// positional, so that is a wire-format change however defaulted the field is: a
/// v15 payload fed to the grown tile would read past the end of its data maps and
/// into the next tile. That is the fourth instance of the v12/v13/v15 law, and it
/// is the reason for the bump. The *component* separately gained
/// `biome_set: Option<Uuid>`; note this record deliberately has **no such field**,
/// because a v15 payload's bytes end at `asset`. Reading one back therefore always
/// yields a terrain with no biome vocabulary and nothing painted, which is exactly
/// what a v15 level meant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerrainV15 {
    #[serde(default = "default_terrain_mps")]
    pub meters_per_sample: f64,
    #[serde(default = "default_terrain_resolution")]
    pub tile_resolution: u32,
    /// The **pre-v16** heightfield shape: tiles with erosion data maps but no
    /// biome ids (generation 2 of the frozen-tile ladder).
    #[serde(default)]
    pub data: inf_terrain::TerrainDataFrozenV2,
    #[serde(default = "inf_ecs::components::default_terrain_layers")]
    pub layers: [inf_ecs::components::TerrainLayer; inf_ecs::components::TERRAIN_LAYERS],
    #[serde(default = "default_macro_variation")]
    pub macro_variation: f64,
    #[serde(default)]
    pub asset: Option<Uuid>,
}

impl TerrainV15 {
    /// Lift to the live [`Terrain`]: every tile's biome ids come up **empty**
    /// (nothing painted, every sample [`inf_terrain::UNASSIGNED_BIOME`]) and
    /// `biome_set` at `None` (no vocabulary to paint from) — exactly what a
    /// pre-P19.2 level meant.
    pub fn into_current(self) -> Terrain {
        Terrain {
            meters_per_sample: self.meters_per_sample,
            tile_resolution: self.tile_resolution,
            data: self.data.into_current(),
            layers: self.layers,
            macro_variation: self.macro_variation,
            asset: self.asset,
            biome_set: None,
            biome_population: Vec::new(),
        }
    }

    /// Project a live [`Terrain`] back onto the frozen shape (the downgrade-bless
    /// path that regenerates old fixtures). Both P19.2 additions are dropped — the
    /// per-tile biome ids and the `biome_set` reference — a deliberately lossy
    /// direction.
    pub fn from_current(t: Terrain) -> Self {
        Self {
            meters_per_sample: t.meters_per_sample,
            tile_resolution: t.tile_resolution,
            data: inf_terrain::TerrainDataFrozenV2::from_current(&t.data),
            layers: t.layers,
            macro_variation: t.macro_variation,
            asset: t.asset,
        }
    }
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
    /// `data` remains the terrain's only authority — exactly what a pre-v9 level
    /// meant. `biome_set` likewise (P19.2 did not exist).
    pub fn into_current(self) -> Terrain {
        Terrain {
            meters_per_sample: self.meters_per_sample,
            tile_resolution: self.tile_resolution,
            data: self.data.into_current(),
            layers: self.layers,
            macro_variation: self.macro_variation,
            asset: None,
            biome_set: None,
            biome_population: Vec::new(),
        }
    }

    /// Lift a v8 terrain to the **v14** shape — the one hop the v8 → v9 record
    /// upgrade needs, since v9's frozen record now carries a [`TerrainV14`].
    /// Lossless: v8's fields are a subset, and `asset` starts `None` exactly as
    /// [`into_current`](Self::into_current) sets it.
    pub fn into_v14(self) -> TerrainV14 {
        TerrainV14 {
            meters_per_sample: self.meters_per_sample,
            tile_resolution: self.tile_resolution,
            data: self.data,
            layers: self.layers,
            macro_variation: self.macro_variation,
            asset: None,
        }
    }

    /// Project a live [`Terrain`] back onto the frozen shape (the downgrade-bless
    /// path that regenerates old fixtures). The `asset` reference has no v8 home
    /// and is dropped — as are the P19.1 data maps — a deliberately lossy
    /// direction.
    pub fn from_current(t: Terrain) -> Self {
        Self {
            meters_per_sample: t.meters_per_sample,
            tile_resolution: t.tile_resolution,
            data: inf_terrain::TerrainDataFrozenV1::from_current(&t.data),
            layers: t.layers,
            macro_variation: t.macro_variation,
        }
    }
}

/// A frozen schema-v8 entity record (pre-P16.3): the full v8 slot set with the
/// live `Light`/`Material`/decoration components, but the **pre-v9**
/// [`TerrainV8`] terrain slot. v9 changed only `Terrain`, so this differs from the
/// live [`EntityRecord`] only there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV8 {
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
    pub terrain: Option<TerrainV8>,
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
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
    #[serde(default)]
    pub decal: Option<Decal>,
    #[serde(default)]
    pub volume: Option<Volume>,
    #[serde(default)]
    pub spline: Option<Spline>,
    #[serde(default)]
    pub foliage: Option<Foliage>,
}

impl EntityRecordV8 {
    /// Lift a v8 record to the **v9** shape: the terrain slot gains
    /// `asset: None`; every other slot carries through unchanged.
    fn into_v9(self) -> EntityRecordV9 {
        EntityRecordV9 {
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
            terrain: self.terrain.map(TerrainV8::into_v14),
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

/// The pre-v10 file-level settings byte layout (schema v10 froze this when
/// [`LevelSettings`] gained its `partition` block). Frozen file records (v8..v9)
/// carry `settings` as [`LevelSettingsV9`] — v9 did not touch the settings, so
/// one frozen record serves both. [`LevelSettingsV9::into_current`] lifts it with
/// a default (disabled) [`PartitionSettings`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LevelSettingsV9 {
    #[serde(default)]
    pub gravity_2d: Vec2d,
    #[serde(default = "default_gravity_3d")]
    pub gravity_3d: Vec3d,
    #[serde(default = "default_sim_hz")]
    pub sim_hz: f64,
    #[serde(default)]
    pub render: RenderSettingsRecord,
}

impl LevelSettingsV9 {
    /// Lift to the live [`LevelSettings`]: partitioning defaults to **off**,
    /// which is exactly what a pre-v10 level meant (one document, no streaming).
    pub fn into_current(self) -> LevelSettings {
        LevelSettings {
            gravity_2d: self.gravity_2d,
            gravity_3d: self.gravity_3d,
            sim_hz: self.sim_hz,
            render: self.render,
            partition: PartitionSettings::default(),
        }
    }

    /// Project live [`LevelSettings`] back onto the frozen shape (the
    /// downgrade-bless path that regenerates old fixtures). The partition block
    /// has no v9 home and is dropped — a deliberately lossy direction.
    pub fn from_current(s: LevelSettings) -> Self {
        Self {
            gravity_2d: s.gravity_2d,
            gravity_3d: s.gravity_3d,
            sim_hz: s.sim_hz,
            render: s.render,
        }
    }
}

impl Default for LevelSettingsV9 {
    fn default() -> Self {
        Self::from_current(LevelSettings::default())
    }
}

/// A schema-v8 [`SceneFile`] (frozen layout for legacy decode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV8 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV8>,
    /// The **pre-v10** settings shape (v9 did not touch them).
    #[serde(default)]
    pub settings: LevelSettingsV9,
}

impl SceneFileV8 {
    /// Lift a v8 file to the **v9** shape (only `Terrain` changed).
    fn into_v9(self) -> SceneFileV9 {
        SceneFileV9 {
            schema_version: 9,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV8::into_v9)
                .collect(),
            settings: self.settings,
        }
    }
}

/// A schema-**v9** [`EntityRecord`] (pre-P16.5) — the exact byte layout written
/// by P16.3..P16.4b editors: the full v9 slot set with the live `Terrain`
/// (asset reference included), but **neither** v10 world-partition slot. Frozen
/// forever so the committed v9 fixture (and any level saved before P16.5) loads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV9 {
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
    pub terrain: Option<TerrainV14>,
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
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
    #[serde(default)]
    pub decal: Option<Decal>,
    #[serde(default)]
    pub volume: Option<Volume>,
    #[serde(default)]
    pub spline: Option<Spline>,
    #[serde(default)]
    pub foliage: Option<Foliage>,
}

impl EntityRecordV9 {
    /// Lift a v9 record to the current (v10) shape: both world-partition slots
    /// default to `None` — a pre-v10 level named no streaming source and marked
    /// nothing always-loaded.
    fn into_v10(self) -> EntityRecordV10 {
        EntityRecordV10 {
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
            joint_2d: self.joint_2d,
            joint_3d: self.joint_3d,
            audio_source: self.audio_source,
            audio_listener: self.audio_listener,
            decal: self.decal,
            volume: self.volume,
            spline: self.spline,
            foliage: self.foliage,
            streaming_source: None,
            always_loaded: None,
        }
    }
}

/// A schema-v9 [`SceneFile`] (frozen layout for legacy decode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV9 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV9>,
    #[serde(default)]
    pub settings: LevelSettingsV9,
}

impl SceneFileV9 {
    /// Lift a v9 file to the frozen v10 shape (the next hop in the ladder).
    fn into_v10(self) -> SceneFileV10 {
        SceneFileV10 {
            schema_version: 10,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV9::into_v10)
                .collect(),
            settings: self.settings.into_current(),
        }
    }
}

/// The **pre-v11** entity byte layout (schema v11 froze this when the record
/// gained its two sky-authority slots). A v10 payload decodes through this and
/// [`EntityRecordV10::into_v11`] lifts it with both slots `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV10 {
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
    pub terrain: Option<TerrainV14>,
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
    // ── v10 (P16.5) world-partition components ────────────────────────────
    /// Marks this entity as a **streaming source**: world-partition cell
    /// residency is computed from its position at the fixed-step boundary. The
    /// editor stays single-document, so this only takes effect in PIE / a
    /// cooked run.
    #[serde(default)]
    pub streaming_source: Option<StreamingSource>,
    /// Marks this entity as never-streamed: it cooks into the partition's
    /// persistent cell and exists for the whole run.
    #[serde(default)]
    pub always_loaded: Option<AlwaysLoaded>,
}

/// A frozen schema-v10 file layout. It carried the **live** [`LevelSettings`]
/// shape (v11 did not touch the settings), so only `entities` is repointed at the
/// frozen [`EntityRecordV10`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV10 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV10>,
    #[serde(default)]
    pub settings: LevelSettings,
}

impl SceneFileV10 {
    /// Lift a v10 file to the frozen v11 shape (the next hop in the ladder).
    fn into_v11(self) -> SceneFileV11 {
        SceneFileV11 {
            schema_version: 11,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV10::into_v11)
                .collect(),
            settings: self.settings,
        }
    }
}

impl EntityRecordV10 {
    /// Lift a frozen v10 record to the frozen v11 shape: both sky-authority slots
    /// default to `None` — a pre-v11 level had no clock, so the projectors render
    /// it with the retired `SUN_DIR` direction, which is exactly the sun it was
    /// authored under.
    pub fn into_v11(self) -> EntityRecordV11 {
        EntityRecordV11 {
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
            joint_2d: self.joint_2d,
            joint_3d: self.joint_3d,
            audio_source: self.audio_source,
            audio_listener: self.audio_listener,
            decal: self.decal,
            volume: self.volume,
            spline: self.spline,
            foliage: self.foliage,
            streaming_source: self.streaming_source,
            always_loaded: self.always_loaded,
            time_of_day: None,
            sky_atmosphere: None,
        }
    }

    /// Project a live [`EntityRecord`] back onto the frozen v10 shape (the
    /// downgrade-bless path that regenerates the committed v10 fixture). The two
    /// sky slots have no v10 home and are dropped — the one deliberately lossy
    /// direction, asserted as a property by
    /// `v10_entity_downgrade_is_lossless_except_for_the_sky_slots`. Takes the
    /// **live** record (not the frozen v11 one) so the bless path always starts
    /// from the current shape.
    pub fn from_current(r: EntityRecord) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material,
            light: r.light,
            camera: r.camera,
            sprite: r.sprite,
            tilemap: r.tilemap,
            nine_slice: r.nine_slice,
            text2d: r.text2d,
            light_2d: r.light_2d,
            rigid_body_2d: r.rigid_body_2d,
            collider_2d: r.collider_2d,
            character_controller_2d: r.character_controller_2d,
            rigid_body_3d: r.rigid_body_3d,
            collider_3d: r.collider_3d,
            character_controller_3d: r.character_controller_3d,
            actor: r.actor,
            terrain: r.terrain.map(TerrainV14::from_current),
            pcg_volume: r.pcg_volume,
            skeletal_mesh: r.skeletal_mesh,
            anim_player: r.anim_player,
            anim_state_machine: r.anim_state_machine,
            root_motion: r.root_motion,
            attached_to: r.attached_to,
            joint_2d: r.joint_2d,
            joint_3d: r.joint_3d,
            audio_source: r.audio_source,
            audio_listener: r.audio_listener,
            decal: r.decal,
            volume: r.volume,
            spline: r.spline,
            foliage: r.foliage,
            streaming_source: r.streaming_source,
            always_loaded: r.always_loaded,
        }
    }
}

/// The **pre-v12** [`SkyAtmosphere`] byte layout (schema v12 froze this when the
/// component grew its physical-atmosphere block). The frozen [`EntityRecordV11`]
/// carries `sky_atmosphere` as `Option<SkyAtmosphereV11>`, exactly as v4..v8 carry
/// `terrain` as `Option<TerrainV8>`; [`SkyAtmosphereV11::into_current`] lifts it.
///
/// The nine fields mirror the v11 component one-for-one **including their
/// `#[serde(default = "…")]` markers**, and the default fns are reproduced here
/// rather than reached for in `inf_ecs` — a frozen record must not be able to move
/// when the live component's defaults are re-tuned. bincode ignores defaults on
/// the write side, but keeping them identical means this record decodes every
/// partial payload the live one did, in the human-readable codecs too.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SkyAtmosphereV11 {
    #[serde(default = "v11_sky_true")]
    pub enabled: bool,
    #[serde(default = "v11_sun_intensity")]
    pub sun_intensity: f32,
    #[serde(default = "v11_sun_color")]
    pub sun_color: Color,
    #[serde(default = "v11_moon_intensity")]
    pub moon_intensity: f32,
    #[serde(default = "v11_moon_color")]
    pub moon_color: Color,
    #[serde(default = "v11_sky_zenith")]
    pub zenith: Color,
    #[serde(default = "v11_sky_horizon")]
    pub horizon: Color,
    #[serde(default = "v11_sky_ground")]
    pub ground: Color,
    #[serde(default = "v11_night_darkening")]
    pub night_darkening: f32,
}

fn v11_sky_true() -> bool {
    true
}
fn v11_sun_intensity() -> f32 {
    3.0
}
fn v11_sun_color() -> Color {
    Color::new(1.0, 0.98, 0.95, 1.0)
}
fn v11_moon_intensity() -> f32 {
    0.15
}
fn v11_moon_color() -> Color {
    Color::new(0.62, 0.72, 1.0, 1.0)
}
fn v11_sky_zenith() -> Color {
    Color::new(0.012, 0.021, 0.038, 1.0)
}
fn v11_sky_horizon() -> Color {
    Color::new(0.055, 0.081, 0.120, 1.0)
}
fn v11_sky_ground() -> Color {
    Color::new(0.009, 0.011, 0.015, 1.0)
}
fn v11_night_darkening() -> f32 {
    0.85
}

impl SkyAtmosphereV11 {
    /// Lift to the frozen v12 shape (the next hop in the ladder): the v11 half
    /// carries through verbatim and the 13 P17.2 fields take the values a v11
    /// level meant — `tint_strength: 0` and `fog_density: 0` reproduce the
    /// gradient sky with no height fog, and the disc / star / aerial knobs are the
    /// physical constants the v11 renderer already used implicitly. Those come
    /// from **this ladder's own** v12 literals, not from the live component, for
    /// the reason spelled out on [`SkyAtmosphereV12`].
    pub fn into_v12(self) -> SkyAtmosphereV12 {
        SkyAtmosphereV12 {
            enabled: self.enabled,
            sun_intensity: self.sun_intensity,
            sun_color: self.sun_color,
            moon_intensity: self.moon_intensity,
            moon_color: self.moon_color,
            zenith: self.zenith,
            horizon: self.horizon,
            ground: self.ground,
            night_darkening: self.night_darkening,
            physical: v12_sky_true(),
            sky_intensity: v12_one(),
            turbidity: v12_one(),
            mie_anisotropy: v12_mie_anisotropy(),
            sun_disc_deg: v12_sun_disc_deg(),
            moon_disc_deg: v12_moon_disc_deg(),
            star_intensity: v12_one(),
            tint_strength: 0.0,
            aerial_perspective: v12_one(),
            fog_density: 0.0,
            fog_falloff: v12_fog_falloff(),
            fog_height: 0.0,
            fog_color: v12_fog_color(),
        }
    }

    /// Project a live [`SkyAtmosphere`] back onto the frozen v11 shape (the
    /// downgrade-bless path). The whole physical-atmosphere block has no v11 home
    /// and is dropped — the deliberately lossy direction, used only to regenerate
    /// an old fixture from a current record.
    pub fn from_current(a: SkyAtmosphere) -> Self {
        Self {
            enabled: a.enabled,
            sun_intensity: a.sun_intensity,
            sun_color: a.sun_color,
            moon_intensity: a.moon_intensity,
            moon_color: a.moon_color,
            zenith: a.zenith,
            horizon: a.horizon,
            ground: a.ground,
            night_darkening: a.night_darkening,
        }
    }
}

/// The **pre-v12** entity byte layout (schema v12 froze this when
/// [`SkyAtmosphere`] grew its physical-atmosphere block). Identical to the live
/// [`EntityRecord`] except that `sky_atmosphere` is typed as the frozen
/// [`SkyAtmosphereV11`] — v12 added **no** entity slot and moved none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV11 {
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
    #[serde(default)]
    pub actor: Option<Uuid>,
    // ── v4 (P10.6) world components ───────────────────────────────────────
    #[serde(default)]
    pub terrain: Option<TerrainV14>,
    #[serde(default)]
    pub pcg_volume: Option<PcgVolume>,
    // ── v5 (P11.4) animation / character components ───────────────────────
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
    // ── v6 (P12.4) joints / spatial-audio components ──────────────────────
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
    // ── v8 (R-P0) world-decoration components ─────────────────────────────
    #[serde(default)]
    pub decal: Option<Decal>,
    #[serde(default)]
    pub volume: Option<Volume>,
    #[serde(default)]
    pub spline: Option<Spline>,
    #[serde(default)]
    pub foliage: Option<Foliage>,
    // ── v10 (P16.5) world-partition components ────────────────────────────
    #[serde(default)]
    pub streaming_source: Option<StreamingSource>,
    #[serde(default)]
    pub always_loaded: Option<AlwaysLoaded>,
    // ── v11 (P17.1) sky-authority components ──────────────────────────────
    #[serde(default)]
    pub time_of_day: Option<TimeOfDay>,
    /// The **pre-v12** atmosphere shape — the one field that makes this record
    /// differ from the live [`EntityRecord`].
    #[serde(default)]
    pub sky_atmosphere: Option<SkyAtmosphereV11>,
}

impl EntityRecordV11 {
    /// Lift a frozen v11 record to the frozen v12 shape (the next hop in the
    /// ladder). Every slot carries through unchanged; only the atmosphere is
    /// lifted, through [`SkyAtmosphereV11::into_v12`].
    pub fn into_v12(self) -> EntityRecordV12 {
        EntityRecordV12 {
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
            joint_2d: self.joint_2d,
            joint_3d: self.joint_3d,
            audio_source: self.audio_source,
            audio_listener: self.audio_listener,
            decal: self.decal,
            volume: self.volume,
            spline: self.spline,
            foliage: self.foliage,
            streaming_source: self.streaming_source,
            always_loaded: self.always_loaded,
            time_of_day: self.time_of_day,
            sky_atmosphere: self.sky_atmosphere.map(SkyAtmosphereV11::into_v12),
        }
    }

    /// Project a live [`EntityRecord`] back onto the frozen v11 shape (the
    /// downgrade-bless path that regenerates the committed v11 fixture). Only the
    /// physical-atmosphere block is lost — asserted as a property by
    /// `v11_entity_downgrade_is_lossless_except_for_the_physical_atmosphere_block`.
    pub fn from_current(r: EntityRecord) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material,
            light: r.light,
            camera: r.camera,
            sprite: r.sprite,
            tilemap: r.tilemap,
            nine_slice: r.nine_slice,
            text2d: r.text2d,
            light_2d: r.light_2d,
            rigid_body_2d: r.rigid_body_2d,
            collider_2d: r.collider_2d,
            character_controller_2d: r.character_controller_2d,
            rigid_body_3d: r.rigid_body_3d,
            collider_3d: r.collider_3d,
            character_controller_3d: r.character_controller_3d,
            actor: r.actor,
            terrain: r.terrain.map(TerrainV14::from_current),
            pcg_volume: r.pcg_volume,
            skeletal_mesh: r.skeletal_mesh,
            anim_player: r.anim_player,
            anim_state_machine: r.anim_state_machine,
            root_motion: r.root_motion,
            attached_to: r.attached_to,
            joint_2d: r.joint_2d,
            joint_3d: r.joint_3d,
            audio_source: r.audio_source,
            audio_listener: r.audio_listener,
            decal: r.decal,
            volume: r.volume,
            spline: r.spline,
            foliage: r.foliage,
            streaming_source: r.streaming_source,
            always_loaded: r.always_loaded,
            time_of_day: r.time_of_day,
            sky_atmosphere: r.sky_atmosphere.map(SkyAtmosphereV11::from_current),
        }
    }
}

/// A frozen schema-v11 file layout. It carried the **live** [`LevelSettings`]
/// shape (v12 did not touch the settings), so only `entities` is repointed at the
/// frozen [`EntityRecordV11`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV11 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV11>,
    #[serde(default)]
    pub settings: LevelSettings,
}

impl SceneFileV11 {
    /// Lift a v11 file to the frozen v12 shape (the next hop in the ladder).
    fn into_v12(self) -> SceneFileV12 {
        SceneFileV12 {
            schema_version: 12,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV11::into_v12)
                .collect(),
            settings: self.settings,
        }
    }
}

/// The **pre-v13** [`SkyAtmosphere`] byte layout (schema v13 froze this when the
/// component grew its volumetric-cloud block). The frozen [`EntityRecordV12`]
/// carries `sky_atmosphere` as `Option<SkyAtmosphereV12>`, exactly as
/// [`EntityRecordV11`] carries it as `Option<SkyAtmosphereV11>`;
/// [`SkyAtmosphereV12::into_current`] lifts it.
///
/// The 22 fields mirror the v12 component one-for-one **including their
/// `#[serde(default = "…")]` markers**, and the default fns are reproduced here
/// rather than reached for in `inf_ecs` — a frozen record must not be able to move
/// when the live component's defaults are re-tuned. bincode ignores defaults on
/// the write side, but keeping them identical means this record decodes every
/// partial payload the live one did, in the human-readable codecs too.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SkyAtmosphereV12 {
    #[serde(default = "v12_sky_true")]
    pub enabled: bool,
    #[serde(default = "v12_sun_intensity")]
    pub sun_intensity: f32,
    #[serde(default = "v12_sun_color")]
    pub sun_color: Color,
    #[serde(default = "v12_moon_intensity")]
    pub moon_intensity: f32,
    #[serde(default = "v12_moon_color")]
    pub moon_color: Color,
    #[serde(default = "v12_sky_zenith")]
    pub zenith: Color,
    #[serde(default = "v12_sky_horizon")]
    pub horizon: Color,
    #[serde(default = "v12_sky_ground")]
    pub ground: Color,
    #[serde(default = "v12_night_darkening")]
    pub night_darkening: f32,
    #[serde(default = "v12_sky_true")]
    pub physical: bool,
    #[serde(default = "v12_one")]
    pub sky_intensity: f32,
    #[serde(default = "v12_one")]
    pub turbidity: f32,
    #[serde(default = "v12_mie_anisotropy")]
    pub mie_anisotropy: f32,
    #[serde(default = "v12_sun_disc_deg")]
    pub sun_disc_deg: f32,
    #[serde(default = "v12_moon_disc_deg")]
    pub moon_disc_deg: f32,
    #[serde(default = "v12_one")]
    pub star_intensity: f32,
    #[serde(default)]
    pub tint_strength: f32,
    #[serde(default = "v12_one")]
    pub aerial_perspective: f32,
    #[serde(default)]
    pub fog_density: f32,
    #[serde(default = "v12_fog_falloff")]
    pub fog_falloff: f32,
    #[serde(default)]
    pub fog_height: f32,
    #[serde(default = "v12_fog_color")]
    pub fog_color: Color,
}

fn v12_sky_true() -> bool {
    true
}
fn v12_sun_intensity() -> f32 {
    3.0
}
fn v12_sun_color() -> Color {
    Color::new(1.0, 0.98, 0.95, 1.0)
}
fn v12_moon_intensity() -> f32 {
    0.15
}
fn v12_moon_color() -> Color {
    Color::new(0.62, 0.72, 1.0, 1.0)
}
fn v12_sky_zenith() -> Color {
    Color::new(0.012, 0.021, 0.038, 1.0)
}
fn v12_sky_horizon() -> Color {
    Color::new(0.055, 0.081, 0.120, 1.0)
}
fn v12_sky_ground() -> Color {
    Color::new(0.009, 0.011, 0.015, 1.0)
}
fn v12_night_darkening() -> f32 {
    0.85
}
fn v12_one() -> f32 {
    1.0
}
fn v12_mie_anisotropy() -> f32 {
    0.8
}
fn v12_sun_disc_deg() -> f32 {
    0.545
}
fn v12_moon_disc_deg() -> f32 {
    0.52
}
fn v12_fog_falloff() -> f32 {
    0.002 // 500 m e-folding height
}
fn v12_fog_color() -> Color {
    Color::new(1.0, 1.0, 1.0, 1.0)
}

impl SkyAtmosphereV12 {
    /// Lift to the frozen **v13** shape (not to the live component): the 22 v12
    /// fields carry through verbatim and the 14 P17.3 cloud fields take *this
    /// ladder's own* `v13_*` literals.
    ///
    /// It targets `SkyAtmosphereV13` rather than `SkyAtmosphere` for the reason
    /// P17.3 already had to learn one version down: `EntityRecordV12` carries the
    /// **frozen** atmosphere, so this hop cannot reach the live component without
    /// the frozen record stopping being frozen. Filling from the `v13_*` fns keeps
    /// the whole ladder independent of how `SkyAtmosphere::default()` is re-tuned,
    /// and it is byte-identical to `inf-scene`'s one-hop lift exactly while the two
    /// crates agree about the v13 defaults — which is what
    /// `cloud_defaults_are_the_documented_ones` asserts, field by field.
    pub fn into_v13(self) -> SkyAtmosphereV13 {
        SkyAtmosphereV13 {
            enabled: self.enabled,
            sun_intensity: self.sun_intensity,
            sun_color: self.sun_color,
            moon_intensity: self.moon_intensity,
            moon_color: self.moon_color,
            zenith: self.zenith,
            horizon: self.horizon,
            ground: self.ground,
            night_darkening: self.night_darkening,
            physical: self.physical,
            sky_intensity: self.sky_intensity,
            turbidity: self.turbidity,
            mie_anisotropy: self.mie_anisotropy,
            sun_disc_deg: self.sun_disc_deg,
            moon_disc_deg: self.moon_disc_deg,
            star_intensity: self.star_intensity,
            tint_strength: self.tint_strength,
            aerial_perspective: self.aerial_perspective,
            fog_density: self.fog_density,
            fog_falloff: self.fog_falloff,
            fog_height: self.fog_height,
            fog_color: self.fog_color,
            clouds_enabled: false,
            cloud_coverage: v13_cloud_coverage(),
            cloud_type: v13_cloud_type(),
            cloud_bottom: v13_cloud_bottom(),
            cloud_top: v13_cloud_top(),
            cloud_density: v13_cloud_density(),
            cloud_detail: v13_cloud_detail(),
            cloud_seed: 0,
            cloud_wind_x: v13_cloud_wind_x(),
            cloud_wind_z: v13_cloud_wind_z(),
            cloud_phase_g: v13_cloud_phase_g(),
            cloud_shadow: v13_one(),
            cloud_ambient: v13_one(),
            cloud_color: v13_cloud_color(),
        }
    }

    pub fn from_current(a: SkyAtmosphere) -> Self {
        Self {
            enabled: a.enabled,
            sun_intensity: a.sun_intensity,
            sun_color: a.sun_color,
            moon_intensity: a.moon_intensity,
            moon_color: a.moon_color,
            zenith: a.zenith,
            horizon: a.horizon,
            ground: a.ground,
            night_darkening: a.night_darkening,
            physical: a.physical,
            sky_intensity: a.sky_intensity,
            turbidity: a.turbidity,
            mie_anisotropy: a.mie_anisotropy,
            sun_disc_deg: a.sun_disc_deg,
            moon_disc_deg: a.moon_disc_deg,
            star_intensity: a.star_intensity,
            tint_strength: a.tint_strength,
            aerial_perspective: a.aerial_perspective,
            fog_density: a.fog_density,
            fog_falloff: a.fog_falloff,
            fog_height: a.fog_height,
            fog_color: a.fog_color,
        }
    }
}

/// The **pre-v13** entity byte layout (schema v13 froze this when
/// [`SkyAtmosphere`] grew its volumetric-cloud block). Identical to the live
/// [`EntityRecord`] except that `sky_atmosphere` is typed as the frozen
/// [`SkyAtmosphereV12`] — v13 added **no** entity slot and moved none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV12 {
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
    #[serde(default)]
    pub actor: Option<Uuid>,
    // ── v4 (P10.6) world components ───────────────────────────────────────
    #[serde(default)]
    pub terrain: Option<TerrainV14>,
    #[serde(default)]
    pub pcg_volume: Option<PcgVolume>,
    // ── v5 (P11.4) animation / character components ───────────────────────
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
    // ── v6 (P12.4) joints / spatial-audio components ──────────────────────
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
    // ── v8 (R-P0) world-decoration components ─────────────────────────────
    #[serde(default)]
    pub decal: Option<Decal>,
    #[serde(default)]
    pub volume: Option<Volume>,
    #[serde(default)]
    pub spline: Option<Spline>,
    #[serde(default)]
    pub foliage: Option<Foliage>,
    // ── v10 (P16.5) world-partition components ────────────────────────────
    #[serde(default)]
    pub streaming_source: Option<StreamingSource>,
    #[serde(default)]
    pub always_loaded: Option<AlwaysLoaded>,
    // ── v11 (P17.1) sky-authority components ──────────────────────────────
    #[serde(default)]
    pub time_of_day: Option<TimeOfDay>,
    /// The **pre-v13** atmosphere shape — the one field that makes this record
    /// differ from the live [`EntityRecord`].
    #[serde(default)]
    pub sky_atmosphere: Option<SkyAtmosphereV12>,
}

impl EntityRecordV12 {
    /// Lift a frozen v12 record to the live (v13) [`EntityRecord`]. Every slot
    /// carries through unchanged; only the atmosphere is lifted, through
    /// [`SkyAtmosphereV12::into_v13`].
    pub fn into_v13(self) -> EntityRecordV13 {
        EntityRecordV13 {
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
            joint_2d: self.joint_2d,
            joint_3d: self.joint_3d,
            audio_source: self.audio_source,
            audio_listener: self.audio_listener,
            decal: self.decal,
            volume: self.volume,
            spline: self.spline,
            foliage: self.foliage,
            streaming_source: self.streaming_source,
            always_loaded: self.always_loaded,
            time_of_day: self.time_of_day,
            sky_atmosphere: self.sky_atmosphere.map(SkyAtmosphereV12::into_v13),
        }
    }

    /// Project a live [`EntityRecord`] back onto the frozen v12 shape (the
    /// downgrade-bless path that regenerates the committed v12 fixture). Only the
    /// volumetric-cloud block is lost — asserted as a property by
    /// `v12_entity_downgrade_is_lossless_except_for_the_cloud_block`.
    pub fn from_current(r: EntityRecord) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material,
            light: r.light,
            camera: r.camera,
            sprite: r.sprite,
            tilemap: r.tilemap,
            nine_slice: r.nine_slice,
            text2d: r.text2d,
            light_2d: r.light_2d,
            rigid_body_2d: r.rigid_body_2d,
            collider_2d: r.collider_2d,
            character_controller_2d: r.character_controller_2d,
            rigid_body_3d: r.rigid_body_3d,
            collider_3d: r.collider_3d,
            character_controller_3d: r.character_controller_3d,
            actor: r.actor,
            terrain: r.terrain.map(TerrainV14::from_current),
            pcg_volume: r.pcg_volume,
            skeletal_mesh: r.skeletal_mesh,
            anim_player: r.anim_player,
            anim_state_machine: r.anim_state_machine,
            root_motion: r.root_motion,
            attached_to: r.attached_to,
            joint_2d: r.joint_2d,
            joint_3d: r.joint_3d,
            audio_source: r.audio_source,
            audio_listener: r.audio_listener,
            decal: r.decal,
            volume: r.volume,
            spline: r.spline,
            foliage: r.foliage,
            streaming_source: r.streaming_source,
            always_loaded: r.always_loaded,
            time_of_day: r.time_of_day,
            sky_atmosphere: r.sky_atmosphere.map(SkyAtmosphereV12::from_current),
        }
    }
}

/// The **pre-v14** [`SkyAtmosphere`] byte layout (schema v14 froze this when the
/// component grew its **weather block**). The frozen [`EntityRecordV13`] carries
/// `sky_atmosphere` as `Option<SkyAtmosphereV13>`, exactly as
/// [`EntityRecordV12`] carries it as `Option<SkyAtmosphereV12>`;
/// [`SkyAtmosphereV13::into_current`] lifts it.
///
/// The 36 fields mirror the v13 component one-for-one **including their
/// `#[serde(default = "...")]` markers**, and the default fns are reproduced here
/// rather than reached for in `inf_ecs` -- a frozen record must not be able to
/// move when the live component's defaults are re-tuned.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SkyAtmosphereV13 {
    #[serde(default = "v13_sky_true")]
    pub enabled: bool,
    #[serde(default = "v13_sun_intensity")]
    pub sun_intensity: f32,
    #[serde(default = "v13_sun_color")]
    pub sun_color: Color,
    #[serde(default = "v13_moon_intensity")]
    pub moon_intensity: f32,
    #[serde(default = "v13_moon_color")]
    pub moon_color: Color,
    #[serde(default = "v13_sky_zenith")]
    pub zenith: Color,
    #[serde(default = "v13_sky_horizon")]
    pub horizon: Color,
    #[serde(default = "v13_sky_ground")]
    pub ground: Color,
    #[serde(default = "v13_night_darkening")]
    pub night_darkening: f32,
    #[serde(default = "v13_sky_true")]
    pub physical: bool,
    #[serde(default = "v13_one")]
    pub sky_intensity: f32,
    #[serde(default = "v13_one")]
    pub turbidity: f32,
    #[serde(default = "v13_mie_anisotropy")]
    pub mie_anisotropy: f32,
    #[serde(default = "v13_sun_disc_deg")]
    pub sun_disc_deg: f32,
    #[serde(default = "v13_moon_disc_deg")]
    pub moon_disc_deg: f32,
    #[serde(default = "v13_one")]
    pub star_intensity: f32,
    #[serde(default)]
    pub tint_strength: f32,
    #[serde(default = "v13_one")]
    pub aerial_perspective: f32,
    #[serde(default)]
    pub fog_density: f32,
    #[serde(default = "v13_fog_falloff")]
    pub fog_falloff: f32,
    #[serde(default)]
    pub fog_height: f32,
    #[serde(default = "v13_fog_color")]
    pub fog_color: Color,
    #[serde(default)]
    pub clouds_enabled: bool,
    #[serde(default = "v13_cloud_coverage")]
    pub cloud_coverage: f32,
    #[serde(default = "v13_cloud_type")]
    pub cloud_type: f32,
    #[serde(default = "v13_cloud_bottom")]
    pub cloud_bottom: f32,
    #[serde(default = "v13_cloud_top")]
    pub cloud_top: f32,
    #[serde(default = "v13_cloud_density")]
    pub cloud_density: f32,
    #[serde(default = "v13_cloud_detail")]
    pub cloud_detail: f32,
    #[serde(default)]
    pub cloud_seed: u32,
    #[serde(default = "v13_cloud_wind_x")]
    pub cloud_wind_x: f32,
    #[serde(default = "v13_cloud_wind_z")]
    pub cloud_wind_z: f32,
    #[serde(default = "v13_cloud_phase_g")]
    pub cloud_phase_g: f32,
    #[serde(default = "v13_one")]
    pub cloud_shadow: f32,
    #[serde(default = "v13_one")]
    pub cloud_ambient: f32,
    #[serde(default = "v13_cloud_color")]
    pub cloud_color: Color,
}

fn v13_sky_true() -> bool {
    true
}
fn v13_sun_intensity() -> f32 {
    3.0
}
fn v13_sun_color() -> Color {
    Color::new(1.0, 0.98, 0.95, 1.0)
}
fn v13_moon_intensity() -> f32 {
    0.15
}
fn v13_moon_color() -> Color {
    Color::new(0.62, 0.72, 1.0, 1.0)
}
fn v13_sky_zenith() -> Color {
    Color::new(0.012, 0.021, 0.038, 1.0)
}
fn v13_sky_horizon() -> Color {
    Color::new(0.055, 0.081, 0.120, 1.0)
}
fn v13_sky_ground() -> Color {
    Color::new(0.009, 0.011, 0.015, 1.0)
}
fn v13_night_darkening() -> f32 {
    0.85
}
fn v13_one() -> f32 {
    1.0
}
fn v13_mie_anisotropy() -> f32 {
    0.8
}
fn v13_sun_disc_deg() -> f32 {
    0.545
}
fn v13_moon_disc_deg() -> f32 {
    0.52
}
fn v13_fog_falloff() -> f32 {
    0.002 // 500 m e-folding height
}
fn v13_fog_color() -> Color {
    Color::new(1.0, 1.0, 1.0, 1.0)
}
fn v13_cloud_coverage() -> f32 {
    0.35
}
fn v13_cloud_type() -> f32 {
    0.7
}
fn v13_cloud_bottom() -> f32 {
    1500.0
}
fn v13_cloud_top() -> f32 {
    4000.0
}
fn v13_cloud_density() -> f32 {
    0.04
}
fn v13_cloud_detail() -> f32 {
    0.6
}
fn v13_cloud_wind_x() -> f32 {
    6.0
}
fn v13_cloud_wind_z() -> f32 {
    2.0
}
fn v13_cloud_phase_g() -> f32 {
    0.8
}
fn v13_cloud_color() -> Color {
    Color::new(1.0, 1.0, 1.0, 1.0)
}

impl SkyAtmosphereV13 {
    /// Lift to the live [`SkyAtmosphere`]: the 36 v13 fields carry through
    /// verbatim and the 11 P17.4 weather fields take their live
    /// `SkyAtmosphere::default()` values. That default *is* what a v13 level
    /// meant -- `weather_enabled: false` leaves the authored cloud and fog fields
    /// driving the sky exactly as they did.
    ///
    /// This is the ladder's **last** hop, which is why it may reach
    /// `SkyAtmosphere::default()` at all: the fields it fills are the ones the
    /// live component just gained, and Ring 0's default is their definition.
    pub fn into_current(self) -> SkyAtmosphere {
        SkyAtmosphere {
            enabled: self.enabled,
            sun_intensity: self.sun_intensity,
            sun_color: self.sun_color,
            moon_intensity: self.moon_intensity,
            moon_color: self.moon_color,
            zenith: self.zenith,
            horizon: self.horizon,
            ground: self.ground,
            night_darkening: self.night_darkening,
            physical: self.physical,
            sky_intensity: self.sky_intensity,
            turbidity: self.turbidity,
            mie_anisotropy: self.mie_anisotropy,
            sun_disc_deg: self.sun_disc_deg,
            moon_disc_deg: self.moon_disc_deg,
            star_intensity: self.star_intensity,
            tint_strength: self.tint_strength,
            aerial_perspective: self.aerial_perspective,
            fog_density: self.fog_density,
            fog_falloff: self.fog_falloff,
            fog_height: self.fog_height,
            fog_color: self.fog_color,
            clouds_enabled: self.clouds_enabled,
            cloud_coverage: self.cloud_coverage,
            cloud_type: self.cloud_type,
            cloud_bottom: self.cloud_bottom,
            cloud_top: self.cloud_top,
            cloud_density: self.cloud_density,
            cloud_detail: self.cloud_detail,
            cloud_seed: self.cloud_seed,
            cloud_wind_x: self.cloud_wind_x,
            cloud_wind_z: self.cloud_wind_z,
            cloud_phase_g: self.cloud_phase_g,
            cloud_shadow: self.cloud_shadow,
            cloud_ambient: self.cloud_ambient,
            cloud_color: self.cloud_color,
            ..SkyAtmosphere::default()
        }
    }

    /// Project a live [`SkyAtmosphere`] back onto the frozen v13 shape (the
    /// downgrade-bless path). The whole weather block has no v13 home and is
    /// dropped -- the deliberately lossy direction, used only to regenerate an
    /// old fixture from a current record.
    pub fn from_current(a: SkyAtmosphere) -> Self {
        Self {
            enabled: a.enabled,
            sun_intensity: a.sun_intensity,
            sun_color: a.sun_color,
            moon_intensity: a.moon_intensity,
            moon_color: a.moon_color,
            zenith: a.zenith,
            horizon: a.horizon,
            ground: a.ground,
            night_darkening: a.night_darkening,
            physical: a.physical,
            sky_intensity: a.sky_intensity,
            turbidity: a.turbidity,
            mie_anisotropy: a.mie_anisotropy,
            sun_disc_deg: a.sun_disc_deg,
            moon_disc_deg: a.moon_disc_deg,
            star_intensity: a.star_intensity,
            tint_strength: a.tint_strength,
            aerial_perspective: a.aerial_perspective,
            fog_density: a.fog_density,
            fog_falloff: a.fog_falloff,
            fog_height: a.fog_height,
            fog_color: a.fog_color,
            clouds_enabled: a.clouds_enabled,
            cloud_coverage: a.cloud_coverage,
            cloud_type: a.cloud_type,
            cloud_bottom: a.cloud_bottom,
            cloud_top: a.cloud_top,
            cloud_density: a.cloud_density,
            cloud_detail: a.cloud_detail,
            cloud_seed: a.cloud_seed,
            cloud_wind_x: a.cloud_wind_x,
            cloud_wind_z: a.cloud_wind_z,
            cloud_phase_g: a.cloud_phase_g,
            cloud_shadow: a.cloud_shadow,
            cloud_ambient: a.cloud_ambient,
            cloud_color: a.cloud_color,
        }
    }
}

/// The **pre-v14** entity byte layout (schema v14 froze this when
/// [`SkyAtmosphere`] grew its weather block). Identical to the live
/// [`EntityRecord`] except that `sky_atmosphere` is typed as the frozen
/// [`SkyAtmosphereV13`] — v14 added **no** entity slot and moved none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV13 {
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
    #[serde(default)]
    pub actor: Option<Uuid>,
    // ── v4 (P10.6) world components ───────────────────────────────────────
    #[serde(default)]
    pub terrain: Option<TerrainV14>,
    #[serde(default)]
    pub pcg_volume: Option<PcgVolume>,
    // ── v5 (P11.4) animation / character components ───────────────────────
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
    // ── v6 (P12.4) joints / spatial-audio components ──────────────────────
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
    // ── v8 (R-P0) world-decoration components ─────────────────────────────
    #[serde(default)]
    pub decal: Option<Decal>,
    #[serde(default)]
    pub volume: Option<Volume>,
    #[serde(default)]
    pub spline: Option<Spline>,
    #[serde(default)]
    pub foliage: Option<Foliage>,
    // ── v10 (P16.5) world-partition components ────────────────────────────
    #[serde(default)]
    pub streaming_source: Option<StreamingSource>,
    #[serde(default)]
    pub always_loaded: Option<AlwaysLoaded>,
    // ── v11 (P17.1) sky-authority components ──────────────────────────────
    #[serde(default)]
    pub time_of_day: Option<TimeOfDay>,
    /// The **pre-v13** atmosphere shape — the one field that makes this record
    /// differ from the live [`EntityRecord`].
    #[serde(default)]
    pub sky_atmosphere: Option<SkyAtmosphereV13>,
}

impl EntityRecordV13 {
    /// Lift a frozen v12 record to the live (v13) [`EntityRecord`]. Every slot
    /// carries through unchanged; only the atmosphere is lifted, through
    /// [`SkyAtmosphereV13::into_current`].
    pub fn into_current(self) -> EntityRecord {
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
            terrain: self.terrain.map(TerrainV14::into_current),
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
            streaming_source: self.streaming_source,
            always_loaded: self.always_loaded,
            time_of_day: self.time_of_day,
            sky_atmosphere: self.sky_atmosphere.map(SkyAtmosphereV13::into_current),
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
        }
    }

    /// Project a live [`EntityRecord`] back onto the frozen v12 shape (the
    /// downgrade-bless path that regenerates the committed v12 fixture). Only the
    /// volumetric-cloud block is lost — asserted as a property by
    /// `v12_entity_downgrade_is_lossless_except_for_the_cloud_block`.
    pub fn from_current(r: EntityRecord) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material,
            light: r.light,
            camera: r.camera,
            sprite: r.sprite,
            tilemap: r.tilemap,
            nine_slice: r.nine_slice,
            text2d: r.text2d,
            light_2d: r.light_2d,
            rigid_body_2d: r.rigid_body_2d,
            collider_2d: r.collider_2d,
            character_controller_2d: r.character_controller_2d,
            rigid_body_3d: r.rigid_body_3d,
            collider_3d: r.collider_3d,
            character_controller_3d: r.character_controller_3d,
            actor: r.actor,
            terrain: r.terrain.map(TerrainV14::from_current),
            pcg_volume: r.pcg_volume,
            skeletal_mesh: r.skeletal_mesh,
            anim_player: r.anim_player,
            anim_state_machine: r.anim_state_machine,
            root_motion: r.root_motion,
            attached_to: r.attached_to,
            joint_2d: r.joint_2d,
            joint_3d: r.joint_3d,
            audio_source: r.audio_source,
            audio_listener: r.audio_listener,
            decal: r.decal,
            volume: r.volume,
            spline: r.spline,
            foliage: r.foliage,
            streaming_source: r.streaming_source,
            always_loaded: r.always_loaded,
            time_of_day: r.time_of_day,
            sky_atmosphere: r.sky_atmosphere.map(SkyAtmosphereV13::from_current),
        }
    }
}

/// The **pre-v15** entity byte layout (schema v15 froze this when P19.1 gave
/// every terrain tile its sparse erosion data-map layer). Identical to the live
/// [`EntityRecord`] except that `terrain` is typed as the frozen [`TerrainV14`]
/// — v15 added **no** entity slot and moved none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV14 {
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
    #[serde(default)]
    pub actor: Option<Uuid>,
    // ── v4 (P10.6) world components ───────────────────────────────────────
    #[serde(default)]
    pub terrain: Option<TerrainV14>,
    #[serde(default)]
    pub pcg_volume: Option<PcgVolume>,
    // ── v5 (P11.4) animation / character components ───────────────────────
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
    // ── v6 (P12.4) joints / spatial-audio components ──────────────────────
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
    // ── v8 (R-P0) world-decoration components ─────────────────────────────
    #[serde(default)]
    pub decal: Option<Decal>,
    #[serde(default)]
    pub volume: Option<Volume>,
    #[serde(default)]
    pub spline: Option<Spline>,
    #[serde(default)]
    pub foliage: Option<Foliage>,
    // ── v10 (P16.5) world-partition components ────────────────────────────
    #[serde(default)]
    pub streaming_source: Option<StreamingSource>,
    #[serde(default)]
    pub always_loaded: Option<AlwaysLoaded>,
    // ── v11 (P17.1) sky-authority components ──────────────────────────────
    #[serde(default)]
    pub time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    pub sky_atmosphere: Option<SkyAtmosphere>,
}

impl EntityRecordV14 {
    /// Lift a frozen v14 record one rung, to [`EntityRecordV15`]. Every slot
    /// carries through unchanged; only the terrain hops, through
    /// [`TerrainV14::into_v15`].
    ///
    /// A *single-step* hop, not a jump to the live shape: v16 froze
    /// [`EntityRecordV15`] between this record and [`EntityRecord`], and the
    /// chained ladder's whole value is that each rung is one small, separately
    /// reviewable, separately tested transformation.
    pub fn into_v15(self) -> EntityRecordV15 {
        EntityRecordV15 {
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
            terrain: self.terrain.map(TerrainV14::into_v15),
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
            streaming_source: self.streaming_source,
            always_loaded: self.always_loaded,
            time_of_day: self.time_of_day,
            sky_atmosphere: self.sky_atmosphere,
        }
    }

    /// Project a live [`EntityRecord`] back onto the frozen v14 shape (the
    /// downgrade-bless path that regenerates the committed v14 fixture). Only the
    /// volumetric-erosion data maps are lost — asserted as a property by
    /// `v14_entity_downgrade_is_lossless_except_for_the_data_maps`.
    pub fn from_current(r: EntityRecord) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material,
            light: r.light,
            camera: r.camera,
            sprite: r.sprite,
            tilemap: r.tilemap,
            nine_slice: r.nine_slice,
            text2d: r.text2d,
            light_2d: r.light_2d,
            rigid_body_2d: r.rigid_body_2d,
            collider_2d: r.collider_2d,
            character_controller_2d: r.character_controller_2d,
            rigid_body_3d: r.rigid_body_3d,
            collider_3d: r.collider_3d,
            character_controller_3d: r.character_controller_3d,
            actor: r.actor,
            terrain: r.terrain.map(TerrainV14::from_current),
            pcg_volume: r.pcg_volume,
            skeletal_mesh: r.skeletal_mesh,
            anim_player: r.anim_player,
            anim_state_machine: r.anim_state_machine,
            root_motion: r.root_motion,
            attached_to: r.attached_to,
            joint_2d: r.joint_2d,
            joint_3d: r.joint_3d,
            audio_source: r.audio_source,
            audio_listener: r.audio_listener,
            decal: r.decal,
            volume: r.volume,
            spline: r.spline,
            foliage: r.foliage,
            streaming_source: r.streaming_source,
            always_loaded: r.always_loaded,
            time_of_day: r.time_of_day,
            sky_atmosphere: r.sky_atmosphere,
        }
    }
}

/// The **pre-v16** entity byte layout (schema v16 froze this when P19.2 gave every
/// terrain tile its sparse per-sample biome id layer, and [`Terrain`] its
/// `biome_set` reference). Identical to the live [`EntityRecord`] except that
/// `terrain` is typed as the frozen [`TerrainV15`] — v16 added **no** entity slot
/// and moved none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV15 {
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
    #[serde(default)]
    pub actor: Option<Uuid>,
    // ── v4 (P10.6) world components ───────────────────────────────────────
    #[serde(default)]
    pub terrain: Option<TerrainV15>,
    #[serde(default)]
    pub pcg_volume: Option<PcgVolume>,
    // ── v5 (P11.4) animation / character components ───────────────────────
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
    // ── v6 (P12.4) joints / spatial-audio components ──────────────────────
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
    // ── v8 (R-P0) world-decoration components ─────────────────────────────
    #[serde(default)]
    pub decal: Option<Decal>,
    #[serde(default)]
    pub volume: Option<Volume>,
    #[serde(default)]
    pub spline: Option<Spline>,
    #[serde(default)]
    pub foliage: Option<Foliage>,
    // ── v10 (P16.5) world-partition components ────────────────────────────
    #[serde(default)]
    pub streaming_source: Option<StreamingSource>,
    #[serde(default)]
    pub always_loaded: Option<AlwaysLoaded>,
    // ── v11 (P17.1) sky-authority components ──────────────────────────────
    #[serde(default)]
    pub time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    pub sky_atmosphere: Option<SkyAtmosphere>,
}

impl EntityRecordV15 {
    /// Lift a frozen v15 record one rung, to the frozen [`EntityRecordV16`]. Every
    /// slot carries through unchanged; only the terrain is lifted, through
    /// [`TerrainV15::into_current`].
    ///
    /// v17 inserted [`EntityRecordV16`] between this record and [`EntityRecord`],
    /// and the rung is what keeps a v15 payload loading forever — the same shape
    /// v15 gave the v14 rung.
    pub fn into_v16(self) -> EntityRecordV16 {
        EntityRecordV16 {
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
            terrain: self.terrain.map(TerrainV15::into_current),
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
            streaming_source: self.streaming_source,
            always_loaded: self.always_loaded,
            time_of_day: self.time_of_day,
            sky_atmosphere: self.sky_atmosphere,
        }
    }

    /// Project a live [`EntityRecord`] back onto the frozen v15 shape (the
    /// downgrade-bless path that regenerates the committed v15 fixture). Only the
    /// per-tile biome ids and the `biome_set` reference are lost — asserted as a
    /// property by `v15_entity_downgrade_is_lossless_except_for_the_biome_ids`.
    pub fn from_current(r: EntityRecord) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material,
            light: r.light,
            camera: r.camera,
            sprite: r.sprite,
            tilemap: r.tilemap,
            nine_slice: r.nine_slice,
            text2d: r.text2d,
            light_2d: r.light_2d,
            rigid_body_2d: r.rigid_body_2d,
            collider_2d: r.collider_2d,
            character_controller_2d: r.character_controller_2d,
            rigid_body_3d: r.rigid_body_3d,
            collider_3d: r.collider_3d,
            character_controller_3d: r.character_controller_3d,
            actor: r.actor,
            terrain: r.terrain.map(TerrainV15::from_current),
            pcg_volume: r.pcg_volume,
            skeletal_mesh: r.skeletal_mesh,
            anim_player: r.anim_player,
            anim_state_machine: r.anim_state_machine,
            root_motion: r.root_motion,
            attached_to: r.attached_to,
            joint_2d: r.joint_2d,
            joint_3d: r.joint_3d,
            audio_source: r.audio_source,
            audio_listener: r.audio_listener,
            decal: r.decal,
            volume: r.volume,
            spline: r.spline,
            foliage: r.foliage,
            streaming_source: r.streaming_source,
            always_loaded: r.always_loaded,
            time_of_day: r.time_of_day,
            sky_atmosphere: r.sky_atmosphere,
        }
    }
}

/// The **pre-v17** entity byte layout (schema v17 froze this when P20.1 appended
/// the `water_body` slot). Identical to the live [`EntityRecord`] except that it
/// has **no** `water_body` field \u2014 that is precisely what v17 added \u2014 so this is
/// the [`EntityRecordV10`] shape of bump (a new slot at the tail), not the
/// [`EntityRecordV14`] one (a component that grew).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV16 {
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
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
    #[serde(default)]
    pub decal: Option<Decal>,
    #[serde(default)]
    pub volume: Option<Volume>,
    #[serde(default)]
    pub spline: Option<Spline>,
    #[serde(default)]
    pub foliage: Option<Foliage>,
    #[serde(default)]
    pub streaming_source: Option<StreamingSource>,
    #[serde(default)]
    pub always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    pub time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    pub sky_atmosphere: Option<SkyAtmosphere>,
}

impl EntityRecordV16 {
    /// Lift a frozen v16 record one rung, to the frozen [`EntityRecordV17`].
    /// Every slot carries through unchanged; the one new slot lifts to `None` \u2014
    /// a level with no water, which is exactly what a v16 level was.
    ///
    /// v18 inserted [`EntityRecordV17`] between this record and [`EntityRecord`],
    /// and the rung is what keeps a v16 payload loading forever \u2014 the same shape
    /// v17 gave the v15 rung.
    pub fn into_v17(self) -> EntityRecordV17 {
        EntityRecordV17 {
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
            joint_2d: self.joint_2d,
            joint_3d: self.joint_3d,
            audio_source: self.audio_source,
            audio_listener: self.audio_listener,
            decal: self.decal,
            volume: self.volume,
            spline: self.spline,
            foliage: self.foliage,
            streaming_source: self.streaming_source,
            always_loaded: self.always_loaded,
            time_of_day: self.time_of_day,
            sky_atmosphere: self.sky_atmosphere,
            water_body: None,
        }
    }

    /// Project a live [`EntityRecord`] back onto the frozen v16 shape (the
    /// **downgrade-bless** path that regenerates the committed v16 fixture). Only
    /// the water body is lost \u2014 asserted as a property, not as a field list, by
    /// `v16_entity_downgrade_is_lossless_except_for_the_water_body`.
    ///
    /// Takes the **live** record rather than the frozen one a rung up, so the
    /// bless path always starts from today's truth.
    pub fn from_current(r: EntityRecord) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material,
            light: r.light,
            camera: r.camera,
            sprite: r.sprite,
            tilemap: r.tilemap,
            nine_slice: r.nine_slice,
            text2d: r.text2d,
            light_2d: r.light_2d,
            rigid_body_2d: r.rigid_body_2d,
            collider_2d: r.collider_2d,
            character_controller_2d: r.character_controller_2d,
            rigid_body_3d: r.rigid_body_3d,
            collider_3d: r.collider_3d,
            character_controller_3d: r.character_controller_3d,
            actor: r.actor,
            terrain: r.terrain,
            pcg_volume: r.pcg_volume,
            skeletal_mesh: r.skeletal_mesh,
            anim_player: r.anim_player,
            anim_state_machine: r.anim_state_machine,
            root_motion: r.root_motion,
            attached_to: r.attached_to,
            joint_2d: r.joint_2d,
            joint_3d: r.joint_3d,
            audio_source: r.audio_source,
            audio_listener: r.audio_listener,
            decal: r.decal,
            volume: r.volume,
            spline: r.spline,
            foliage: r.foliage,
            streaming_source: r.streaming_source,
            always_loaded: r.always_loaded,
            time_of_day: r.time_of_day,
            sky_atmosphere: r.sky_atmosphere,
        }
    }
}

/// The **pre-v18** entity byte layout (schema v18 froze this when P20.2 appended
/// the `buoyancy` slot). Identical to the live [`EntityRecord`] except that it
/// has **no** `buoyancy` field — that is precisely what v18 added — so this is
/// the [`EntityRecordV10`] shape of bump (a new slot at the tail), not the
/// [`EntityRecordV14`] one (a component that grew).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV17 {
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
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
    #[serde(default)]
    pub decal: Option<Decal>,
    #[serde(default)]
    pub volume: Option<Volume>,
    #[serde(default)]
    pub spline: Option<Spline>,
    #[serde(default)]
    pub foliage: Option<Foliage>,
    #[serde(default)]
    pub streaming_source: Option<StreamingSource>,
    #[serde(default)]
    pub always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    pub time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    pub sky_atmosphere: Option<SkyAtmosphere>,
    /// The v17 slot this record exists to keep carrying — a v17 level's water
    /// must survive the v18 hop, not merely decode.
    #[serde(default)]
    pub water_body: Option<WaterBody>,
}

impl EntityRecordV17 {
    /// Lift a frozen v17 record one rung, to the frozen [`EntityRecordV18`].
    /// Every slot carries through unchanged; the one new slot lifts to `None` — a
    /// level in which nothing floats, which is exactly what a v17 level was.
    ///
    /// v19 inserted [`EntityRecordV18`] between this record and [`EntityRecord`],
    /// and the rung is what keeps a v17 payload loading forever — the same shape
    /// v18 gave the v16 rung.
    pub fn into_v18(self) -> EntityRecordV18 {
        EntityRecordV18 {
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
            joint_2d: self.joint_2d,
            joint_3d: self.joint_3d,
            audio_source: self.audio_source,
            audio_listener: self.audio_listener,
            decal: self.decal,
            volume: self.volume,
            spline: self.spline,
            foliage: self.foliage,
            streaming_source: self.streaming_source,
            always_loaded: self.always_loaded,
            time_of_day: self.time_of_day,
            sky_atmosphere: self.sky_atmosphere,
            water_body: self.water_body,
            buoyancy: None,
        }
    }

    /// Project a live [`EntityRecord`] back onto the frozen v17 shape (the
    /// **downgrade-bless** path that regenerates the committed v17 fixture). Only
    /// the buoyancy is lost — asserted as a property, not as a field list, by
    /// `v17_entity_downgrade_is_lossless_except_for_the_buoyancy`.
    ///
    /// Takes the **live** record rather than the frozen one a rung up, so the
    /// bless path always starts from today's truth.
    pub fn from_current(r: EntityRecord) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material,
            light: r.light,
            camera: r.camera,
            sprite: r.sprite,
            tilemap: r.tilemap,
            nine_slice: r.nine_slice,
            text2d: r.text2d,
            light_2d: r.light_2d,
            rigid_body_2d: r.rigid_body_2d,
            collider_2d: r.collider_2d,
            character_controller_2d: r.character_controller_2d,
            rigid_body_3d: r.rigid_body_3d,
            collider_3d: r.collider_3d,
            character_controller_3d: r.character_controller_3d,
            actor: r.actor,
            terrain: r.terrain,
            pcg_volume: r.pcg_volume,
            skeletal_mesh: r.skeletal_mesh,
            anim_player: r.anim_player,
            anim_state_machine: r.anim_state_machine,
            root_motion: r.root_motion,
            attached_to: r.attached_to,
            joint_2d: r.joint_2d,
            joint_3d: r.joint_3d,
            audio_source: r.audio_source,
            audio_listener: r.audio_listener,
            decal: r.decal,
            volume: r.volume,
            spline: r.spline,
            foliage: r.foliage,
            streaming_source: r.streaming_source,
            always_loaded: r.always_loaded,
            time_of_day: r.time_of_day,
            sky_atmosphere: r.sky_atmosphere,
            water_body: r.water_body,
        }
    }
}

/// The **pre-v19** entity byte layout (schema v19 froze this when P21.1 appended
/// the `voxel_volume` slot). Identical to the live [`EntityRecord`] except that it
/// has **no** `voxel_volume` field — that is precisely what v19 added — so this is
/// the [`EntityRecordV10`] shape of bump (a new slot at the tail), not the
/// [`EntityRecordV14`] one (a component that grew).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecordV18 {
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
    #[serde(default)]
    pub joint_2d: Option<Joint2D>,
    #[serde(default)]
    pub joint_3d: Option<Joint3D>,
    #[serde(default)]
    pub audio_source: Option<AudioSource>,
    #[serde(default)]
    pub audio_listener: Option<AudioListener>,
    #[serde(default)]
    pub decal: Option<Decal>,
    #[serde(default)]
    pub volume: Option<Volume>,
    #[serde(default)]
    pub spline: Option<Spline>,
    #[serde(default)]
    pub foliage: Option<Foliage>,
    #[serde(default)]
    pub streaming_source: Option<StreamingSource>,
    #[serde(default)]
    pub always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    pub time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    pub sky_atmosphere: Option<SkyAtmosphere>,
    /// The v17 slot this record still carries — a v18 level's water must survive
    /// the v19 hop, not merely decode.
    #[serde(default)]
    pub water_body: Option<WaterBody>,
    /// The v18 slot this record exists to keep carrying — a v18 level's buoyancy
    /// must survive the v19 hop, not merely decode.
    #[serde(default)]
    pub buoyancy: Option<Buoyancy>,
}

impl EntityRecordV18 {
    /// Lift a frozen v18 record to the live (v19) [`EntityRecord`]. Every slot
    /// carries through unchanged; the one new slot lifts to `None` — a level whose
    /// ground is a heightfield and nothing else, which is exactly what a v18 level
    /// was.
    pub fn into_current(self) -> EntityRecord {
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
            joint_2d: self.joint_2d,
            joint_3d: self.joint_3d,
            audio_source: self.audio_source,
            audio_listener: self.audio_listener,
            decal: self.decal,
            volume: self.volume,
            spline: self.spline,
            foliage: self.foliage,
            streaming_source: self.streaming_source,
            always_loaded: self.always_loaded,
            time_of_day: self.time_of_day,
            sky_atmosphere: self.sky_atmosphere,
            water_body: self.water_body,
            buoyancy: self.buoyancy,
            voxel_volume: None,
        }
    }

    /// Project a live [`EntityRecord`] back onto the frozen v18 shape (the
    /// **downgrade-bless** path that regenerates the committed v18 fixture). Only
    /// the voxel volume is lost — asserted as a property, not as a field list, by
    /// `v18_entity_downgrade_is_lossless_except_for_the_voxel_volume`.
    ///
    /// Takes the **live** record rather than the frozen one a rung up, so the
    /// bless path always starts from today's truth.
    pub fn from_current(r: EntityRecord) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material,
            light: r.light,
            camera: r.camera,
            sprite: r.sprite,
            tilemap: r.tilemap,
            nine_slice: r.nine_slice,
            text2d: r.text2d,
            light_2d: r.light_2d,
            rigid_body_2d: r.rigid_body_2d,
            collider_2d: r.collider_2d,
            character_controller_2d: r.character_controller_2d,
            rigid_body_3d: r.rigid_body_3d,
            collider_3d: r.collider_3d,
            character_controller_3d: r.character_controller_3d,
            actor: r.actor,
            terrain: r.terrain,
            pcg_volume: r.pcg_volume,
            skeletal_mesh: r.skeletal_mesh,
            anim_player: r.anim_player,
            anim_state_machine: r.anim_state_machine,
            root_motion: r.root_motion,
            attached_to: r.attached_to,
            joint_2d: r.joint_2d,
            joint_3d: r.joint_3d,
            audio_source: r.audio_source,
            audio_listener: r.audio_listener,
            decal: r.decal,
            volume: r.volume,
            spline: r.spline,
            foliage: r.foliage,
            streaming_source: r.streaming_source,
            always_loaded: r.always_loaded,
            time_of_day: r.time_of_day,
            sky_atmosphere: r.sky_atmosphere,
            water_body: r.water_body,
            buoyancy: r.buoyancy,
        }
    }
}

/// A frozen schema-v14 file layout, holding [`EntityRecordV14`]s. v15 did not
/// touch [`LevelSettings`], so only `entities` is repointed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFileV14 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV14>,
    #[serde(default)]
    pub settings: LevelSettings,
}

impl SceneFileV14 {
    /// Lift every record one rung, to the frozen [`SceneFileV15`] shape. Stamped
    /// v15, not [`SCHEMA_VERSION`] — this is a *rung*, and the version it claims
    /// must match the records it actually holds; `into_current` on the next rung
    /// is what stamps the current schema.
    fn into_v15(self) -> SceneFileV15 {
        SceneFileV15 {
            schema_version: 15,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV14::into_v15)
                .collect(),
            settings: self.settings,
        }
    }
}

/// A frozen schema-v15 file layout, holding [`EntityRecordV15`]s. v16 did not
/// touch [`LevelSettings`], so only `entities` is repointed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFileV15 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV15>,
    #[serde(default)]
    pub settings: LevelSettings,
}

impl SceneFileV15 {
    /// Lift every record one rung, to the frozen [`SceneFileV16`] shape. Stamped
    /// v16, not [`SCHEMA_VERSION`] — this is a *rung*, and the version it claims
    /// must match the records it actually holds; `into_current` on the next rung
    /// is what stamps the current schema.
    fn into_v16(self) -> SceneFileV16 {
        SceneFileV16 {
            schema_version: 16,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV15::into_v16)
                .collect(),
            settings: self.settings,
        }
    }
}

/// A frozen schema-v16 file layout, holding [`EntityRecordV16`]s. v17 did not
/// touch [`LevelSettings`], so only `entities` is repointed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFileV16 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV16>,
    #[serde(default)]
    pub settings: LevelSettings,
}

impl SceneFileV16 {
    /// Lift every record one rung, to the frozen [`SceneFileV17`] shape. Stamped
    /// v17, not [`SCHEMA_VERSION`] — this is a *rung*, and the version it claims
    /// must match the records it actually holds; `into_current` on the next rung
    /// is what stamps the current schema.
    fn into_v17(self) -> SceneFileV17 {
        SceneFileV17 {
            schema_version: 17,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV16::into_v17)
                .collect(),
            settings: self.settings,
        }
    }
}

/// A frozen schema-v17 file layout, holding [`EntityRecordV17`]s. v18 did not
/// touch [`LevelSettings`], so only `entities` is repointed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFileV17 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV17>,
    #[serde(default)]
    pub settings: LevelSettings,
}

impl SceneFileV17 {
    /// Lift every record one rung, to the frozen [`SceneFileV18`] shape. Stamped
    /// v18, not [`SCHEMA_VERSION`] — this is a *rung*, and the version it claims
    /// must match the records it actually holds; `into_current` on the next rung
    /// is what stamps the current schema.
    fn into_v18(self) -> SceneFileV18 {
        SceneFileV18 {
            schema_version: 18,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV17::into_v18)
                .collect(),
            settings: self.settings,
        }
    }
}

/// A frozen schema-v18 file layout, holding [`EntityRecordV18`]s. v19 did not
/// touch [`LevelSettings`], so only `entities` is repointed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFileV18 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV18>,
    #[serde(default)]
    pub settings: LevelSettings,
}

impl SceneFileV18 {
    /// Lift every record to the current shape and stamp the current version.
    fn into_current(self) -> SceneFile {
        SceneFile {
            schema_version: SCHEMA_VERSION,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV18::into_current)
                .collect(),
            settings: self.settings,
        }
    }
}

/// A frozen schema-v13 file layout, holding [`EntityRecordV13`]s. v14 did not
/// touch [`LevelSettings`], so only `entities` is repointed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFileV13 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV13>,
    #[serde(default)]
    pub settings: LevelSettings,
}

impl SceneFileV13 {
    /// Lift every record to the current shape and stamp the current version.
    fn into_current(self) -> SceneFile {
        SceneFile {
            schema_version: SCHEMA_VERSION,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV13::into_current)
                .collect(),
            settings: self.settings,
        }
    }
}

/// A frozen schema-v12 file layout. It carried the **live** [`LevelSettings`]
/// shape (v13 did not touch the settings), so only `entities` is repointed at the
/// frozen [`EntityRecordV12`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFileV12 {
    pub schema_version: u32,
    pub title: String,
    pub entities: Vec<EntityRecordV12>,
    #[serde(default)]
    pub settings: LevelSettings,
}

impl SceneFileV12 {
    /// Lift a v12 file to the current (v13) shape.
    fn into_v13(self) -> SceneFileV13 {
        SceneFileV13 {
            schema_version: SCHEMA_VERSION,
            title: self.title,
            entities: self
                .entities
                .into_iter()
                .map(EntityRecordV12::into_v13)
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
    /// Entities in creation order. Parent links are resolved by GUID on load
    /// (see [`apply_to_doc`]), so a child may appear before its parent here (a
    /// node reparented under a later-created one) without losing the hierarchy.
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
        decal: w.get::<Decal>(e).copied(),
        volume: w.get::<Volume>(e).copied(),
        spline: w.get::<Spline>(e).cloned(),
        foliage: w.get::<Foliage>(e).cloned(),
        streaming_source: w.get::<StreamingSource>(e).copied(),
        always_loaded: w.get::<AlwaysLoaded>(e).copied(),
        time_of_day: w.get::<TimeOfDay>(e).copied(),
        sky_atmosphere: w.get::<SkyAtmosphere>(e).copied(),
        water_body: w.get::<WaterBody>(e).copied(),
        buoyancy: w.get::<Buoyancy>(e).copied(),
        voxel_volume: w.get::<VoxelVolume>(e).copied(),
    })
}

/// Drop a **streamed** terrain's working set before it is written to a file.
///
/// The `.inf_terrain` is the authority for an asset-backed terrain (P16.4b), and
/// the level is a *reference* to it — so the level must never persist the tiles
/// the editor paged in to sculpt. Without this, a session that brushed a few
/// square kilometres would silently grow the `.inf_lvl` by every page it touched
/// and, worse, that copy would shadow the asset on the next load: two authorities
/// for the same ground, disagreeing the moment either is edited.
///
/// A no-op for an inline terrain (`asset: None`), whose `data` *is* the
/// authority, and a no-op for a streamed terrain that was never edited (its
/// working set is already empty) — so no existing level's bytes move.
///
/// Deliberately applied in [`to_scene_file`] only, **not** in [`record_of`]:
/// undo's delete/restore snapshots go through the same record type and must keep
/// the working set, or undoing a terrain delete would resurrect the entity with
/// its unsaved edits thrown away.
///
/// # It applies to the PIE handoff too, and that is the point
///
/// `crate::pie` ships the live scene to the player through this same function, so
/// **PIE sees the last *saved* `.inf_terrain`, not the unsaved working set** — the
/// player streams the file, exactly as a shipped build does. That is the PIE ==
/// shipping invariant working as intended, not a gap: handing the player tiles
/// that are not in any asset would make PIE show something no shipped run could
/// ever reproduce. Save before playing to see terrain edits in PIE.
fn strip_streamed_terrain(mut rec: EntityRecord) -> EntityRecord {
    if let Some(t) = rec.terrain.as_mut() {
        if t.asset.is_some() && !t.data.is_empty() {
            t.data = inf_terrain::TerrainData::new(t.tile_resolution, t.meters_per_sample);
        }
    }
    rec
}

/// Where a [`SceneFile`] projection is going, and therefore whether a streamed
/// terrain's working set survives it (P16.4b).
///
/// The distinction is **load-bearing**, not cosmetic: the same projection feeds
/// two very different consumers, and getting it wrong silently destroys work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenePersist {
    /// The projection becomes **bytes** — an `.inf_lvl`, an autosave payload, a
    /// diagnostics dump, or the PIE handoff. A streamed terrain's working set is
    /// stripped: the `.inf_terrain` is the authority, and the level must not
    /// carry (or shadow) a second copy. See [`strip_streamed_terrain`].
    File,
    /// The projection stays **in this process** and will be applied back onto the
    /// same document — Simulate's enter/exit snapshot. Nothing is stripped,
    /// because "restore the document exactly as it was" has to include the
    /// streamed terrain's unsaved working set and its write-back marks.
    ///
    /// Using [`File`](Self::File) here would destroy every unsaved terrain edit on
    /// Play → Stop, *and* leave the undo stack replaying height deltas into tiles
    /// `revert_delta` would recreate flat — the exact corruption the residency
    /// design note warns about.
    Memory,
}

/// Project the document into a serializable [`SceneFile`] **for a file**.
///
/// Shorthand for `to_scene_file_for(doc, ScenePersist::File)`. An in-process
/// snapshot must use [`to_scene_file_for`] with
/// [`ScenePersist::Memory`] — see there.
pub fn to_scene_file(doc: &SceneDoc) -> SceneFile {
    to_scene_file_for(doc, ScenePersist::File)
}

/// Project the document into a [`SceneFile`], choosing whether streamed-terrain
/// working sets survive ([`ScenePersist`]).
pub fn to_scene_file_for(doc: &SceneDoc, persist: ScenePersist) -> SceneFile {
    let entities = doc
        .order()
        .iter()
        .filter_map(|&guid| {
            record_of(doc, guid).map(|rec| match persist {
                ScenePersist::File => strip_streamed_terrain(rec),
                ScenePersist::Memory => rec,
            })
        })
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
            migrate(
                v1.into_v2()
                    .into_v3()
                    .into_v4()
                    .into_v5()
                    .into_v6()
                    .into_v7()
                    .into_v8()
                    .into_v9()
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current(),
            )
        }
        2 => {
            let (v2, _): (SceneFileV2, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v2: {e}"))?;
            migrate(
                v2.into_v3()
                    .into_v4()
                    .into_v5()
                    .into_v6()
                    .into_v7()
                    .into_v8()
                    .into_v9()
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current(),
            )
        }
        3 => {
            let (v3, _): (SceneFileV3, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v3: {e}"))?;
            migrate(
                v3.into_v4()
                    .into_v5()
                    .into_v6()
                    .into_v7()
                    .into_v8()
                    .into_v9()
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current(),
            )
        }
        4 => {
            let (v4, _): (SceneFileV4, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v4: {e}"))?;
            migrate(
                v4.into_v5()
                    .into_v6()
                    .into_v7()
                    .into_v8()
                    .into_v9()
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current(),
            )
        }
        5 => {
            let (v5, _): (SceneFileV5, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v5: {e}"))?;
            migrate(
                v5.into_v6()
                    .into_v7()
                    .into_v8()
                    .into_v9()
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current(),
            )
        }
        6 => {
            let (v6, _): (SceneFileV6, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v6: {e}"))?;
            migrate(
                v6.into_v7()
                    .into_v8()
                    .into_v9()
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current(),
            )
        }
        7 => {
            let (v7, _): (SceneFileV7, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v7: {e}"))?;
            migrate(
                v7.into_v8()
                    .into_v9()
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current(),
            )
        }
        8 => {
            let (v8, _): (SceneFileV8, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v8: {e}"))?;
            migrate(
                v8.into_v9()
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current(),
            )
        }
        9 => {
            let (v9, _): (SceneFileV9, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v9: {e}"))?;
            migrate(
                v9.into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current(),
            )
        }
        10 => {
            let (v10, _): (SceneFileV10, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v10: {e}"))?;
            migrate(v10.into_v11().into_v12().into_v13().into_current())
        }
        11 => {
            let (v11, _): (SceneFileV11, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v11: {e}"))?;
            migrate(v11.into_v12().into_v13().into_current())
        }
        12 => {
            let (v12, _): (SceneFileV12, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v12: {e}"))?;
            migrate(v12.into_v13().into_current())
        }
        13 => {
            let (v13, _): (SceneFileV13, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v13: {e}"))?;
            migrate(v13.into_current())
        }
        14 => {
            let (v14, _): (SceneFileV14, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v14: {e}"))?;
            migrate(
                v14.into_v15()
                    .into_v16()
                    .into_v17()
                    .into_v18()
                    .into_current(),
            )
        }
        15 => {
            let (v15, _): (SceneFileV15, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v15: {e}"))?;
            migrate(v15.into_v16().into_v17().into_v18().into_current())
        }
        16 => {
            let (v16, _): (SceneFileV16, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v16: {e}"))?;
            migrate(v16.into_v17().into_v18().into_current())
        }
        17 => {
            let (v17, _): (SceneFileV17, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v17: {e}"))?;
            migrate(v17.into_v18().into_current())
        }
        18 => {
            let (v18, _): (SceneFileV18, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| format!("decode v18: {e}"))?;
            migrate(v18.into_current())
        }
        19 => {
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
    // (v1→…→v18→v19); nothing more to do here. Future upgrades chain in `decode`.
    Ok(file)
}

/// Rebuild a document from a decoded [`SceneFile`].
///
/// Two passes, so the hierarchy survives regardless of the file's entity order:
/// the first spawns every entity with its components (resolving each parent by
/// GUID where it already exists), and the second re-attaches any child whose
/// parent appeared LATER in the file — a node reparented under a later-created
/// one, which a single in-order pass would silently drop to the root
/// (`spawn_bare` resolves an unspawned parent GUID to `None`). The file still
/// writes entities in `doc.order` (creation) sequence and `doc.order` mirrors
/// the file sequence after load, so `save→load→save` stays byte-identical.
/// Write every component an [`EntityRecord`] carries onto entity `e`, treating
/// the record as the full truth for the component set.
///
/// * `remove_absent = false` — insert only (fresh entities in [`apply_to_doc`]).
/// * `remove_absent = true` — also *remove* every optional component the record
///   leaves `None` (E-P1 add/remove-component undo via `SwapComponents`: reverting
///   to a snapshot must delete components gained since). `Transform` /
///   `Visibility` are structural (always present) and never removed; computed
///   components, `Guid`, `Name`, and hierarchy links are outside the record and
///   left untouched.
pub(crate) fn write_record_components(
    ecs: &mut inf_ecs::EcsWorld,
    e: inf_ecs::Entity,
    rec: &EntityRecord,
    remove_absent: bool,
) {
    let w = ecs.world_mut();
    // Structural components — always present.
    w.entity_mut(e).insert((
        rec.transform,
        Visibility {
            visible: rec.visible,
        },
    ));

    // Each optional slot: insert when present, else (when reverting) remove.
    // `copy_slot`/`clone_slot` keep the binding inside the macro (hygiene).
    macro_rules! copy_slot {
        ($opt:expr, $ty:ty) => {
            if let Some(c) = $opt {
                w.entity_mut(e).insert(*c);
            } else if remove_absent {
                w.entity_mut(e).remove::<$ty>();
            }
        };
    }
    macro_rules! clone_slot {
        ($opt:expr, $ty:ty) => {
            if let Some(c) = $opt {
                w.entity_mut(e).insert(c.clone());
            } else if remove_absent {
                w.entity_mut(e).remove::<$ty>();
            }
        };
    }
    copy_slot!(&rec.mesh, MeshRef);
    copy_slot!(&rec.material, Material);
    copy_slot!(&rec.light, Light);
    copy_slot!(&rec.camera, Camera);
    clone_slot!(&rec.sprite, Sprite);
    clone_slot!(&rec.tilemap, Tilemap);
    clone_slot!(&rec.nine_slice, NineSlice);
    clone_slot!(&rec.text2d, Text2D);
    copy_slot!(&rec.light_2d, Light2D);
    copy_slot!(&rec.rigid_body_2d, RigidBody2D);
    copy_slot!(&rec.collider_2d, Collider2D);
    copy_slot!(&rec.character_controller_2d, CharacterController2D);
    copy_slot!(&rec.rigid_body_3d, RigidBody3D);
    copy_slot!(&rec.collider_3d, Collider3D);
    copy_slot!(&rec.character_controller_3d, CharacterController3D);
    // `actor` is stored bare (`Option<Uuid>`) and wraps into `ActorClass`.
    if let Some(actor) = rec.actor {
        w.entity_mut(e).insert(ActorClass(actor));
    } else if remove_absent {
        w.entity_mut(e).remove::<ActorClass>();
    }
    clone_slot!(&rec.terrain, Terrain);
    clone_slot!(&rec.pcg_volume, PcgVolume);
    copy_slot!(&rec.skeletal_mesh, SkeletalMesh);
    copy_slot!(&rec.anim_player, AnimPlayer);
    copy_slot!(&rec.anim_state_machine, AnimStateMachine);
    copy_slot!(&rec.root_motion, RootMotion);
    clone_slot!(&rec.attached_to, AttachedTo);
    copy_slot!(&rec.joint_2d, Joint2D);
    copy_slot!(&rec.joint_3d, Joint3D);
    clone_slot!(&rec.audio_source, AudioSource);
    copy_slot!(&rec.audio_listener, AudioListener);
    copy_slot!(&rec.decal, Decal);
    copy_slot!(&rec.volume, Volume);
    clone_slot!(&rec.spline, Spline);
    clone_slot!(&rec.foliage, Foliage);
    copy_slot!(&rec.streaming_source, StreamingSource);
    copy_slot!(&rec.always_loaded, AlwaysLoaded);
    copy_slot!(&rec.time_of_day, TimeOfDay);
    copy_slot!(&rec.sky_atmosphere, SkyAtmosphere);
    copy_slot!(&rec.water_body, WaterBody);
    copy_slot!(&rec.buoyancy, Buoyancy);
    copy_slot!(&rec.voxel_volume, VoxelVolume);
}

pub fn apply_to_doc(doc: &mut SceneDoc, file: &SceneFile) {
    doc.reset();
    for rec in &file.entities {
        let e = doc.spawn_bare(rec.guid, &rec.name, rec.parent);
        // Fresh entity → only the insert half (nothing to remove).
        write_record_components(doc.world_mut(), e, rec, false);
    }
    // Second pass: now that every GUID exists, re-attach any child whose parent
    // didn't resolve in the first pass (a reparent under a later-created node).
    // A no-op for the common parents-precede-children ordering — `doc.order` is
    // untouched, so an already-valid file re-saves byte-identically.
    for rec in &file.entities {
        doc.raw_fixup_parent(rec.guid, rec.parent);
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

/// A scene encoded to memory: the bincode payload + its TOML sidecar text +
/// the level GUID written. Splitting the encode (which needs the doc) from the
/// file writes lets a caller hold the doc lock only for the encode and do disk
/// IO after releasing it (the viewport locks the same doc every frame) — see
/// [`encode_scene`] / [`write_encoded`].
pub struct EncodedScene {
    pub guid: Uuid,
    pub payload: Vec<u8>,
    pub sidecar_toml: String,
}

/// Encode `doc` to its payload + sidecar TOML in memory (no file IO). Pair with
/// [`write_encoded`] to persist outside the doc lock. `guid` reuses an existing
/// level GUID or mints a fresh one when `None`.
pub fn encode_scene(doc: &SceneDoc, guid: Option<Uuid>) -> Result<EncodedScene, String> {
    let file = to_scene_file(doc);
    let payload = encode(&file)?;
    let guid = guid.unwrap_or_else(Uuid::new_v4);
    let side = sidecar(doc, guid, &payload);
    let sidecar_toml = toml::to_string_pretty(&side).map_err(|e| format!("sidecar toml: {e}"))?;
    Ok(EncodedScene {
        guid,
        payload,
        sidecar_toml,
    })
}

/// Write a pre-[`encode_scene`]d scene to `path` (payload) + its `.toml` sidecar
/// — the file-IO half of [`save`], callable after the doc lock is released.
pub fn write_encoded(enc: &EncodedScene, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
    }
    std::fs::write(path, &enc.payload).map_err(|e| format!("write payload: {e}"))?;
    std::fs::write(sidecar_path(path), &enc.sidecar_toml)
        .map_err(|e| format!("write sidecar: {e}"))?;
    Ok(())
}

/// Save `doc` to `path` (payload) + its `.toml` sidecar. Returns the level GUID
/// written (fresh if `guid` is `None`). Encode + write are the same bytes in the
/// same order as before — the `encode_scene` / `write_encoded` split is a seam,
/// not a behaviour change.
pub fn save(doc: &SceneDoc, path: &Path, guid: Option<Uuid>) -> Result<Uuid, String> {
    let enc = encode_scene(doc, guid)?;
    write_encoded(&enc, path)?;
    Ok(enc.guid)
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
    let payload = encode(&to_scene_file(doc))?;
    write_recovery_bytes(&payload, dir)
}

/// Write a pre-encoded recovery payload to `dir` — the file-IO half of
/// [`write_recovery`], callable after the doc lock is released (the autosave
/// command encodes under the lock, then writes outside it).
pub fn write_recovery_bytes(payload: &[u8], dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    std::fs::write(recovery_path(dir), payload).map_err(|e| format!("write recovery: {e}"))
}

// ── the streamed-terrain recovery note (P16.4b) ──────────────────────────

/// The sidecar note recording that unsaved **terrain** edits existed when the
/// recovery file was written.
///
/// A plain text file beside the recovery payload rather than a field inside it:
/// the note is a *diagnostic about what was lost*, not document content, and the
/// `.inf_lvl` schema has no business growing a slot for something no load can
/// ever act on.
pub fn recovery_terrain_note_path(dir: &Path) -> PathBuf {
    dir.join("crash-recovery.terrain-edits.txt")
}

/// Record (or, with `None`, clear) the unsaved-terrain-edits note beside the
/// recovery file.
///
/// Autosave calls this every time it writes recovery, because terrain edits are
/// **not** autosaved: asset writes are explicit (see [`crate::terrain_edit`]), so
/// a crash really does lose them and the recovered level's terrain really is the
/// last saved asset. Saying so is the honest thing; silently restoring a level
/// whose terrain is older than the rest of it is not.
pub fn write_recovery_terrain_note(dir: &Path, note: Option<&str>) -> Result<(), String> {
    let path = recovery_terrain_note_path(dir);
    match note {
        Some(text) => {
            std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
            std::fs::write(&path, text).map_err(|e| format!("write terrain note: {e}"))
        }
        None => {
            let _ = std::fs::remove_file(&path);
            Ok(())
        }
    }
}

/// Take the unsaved-terrain-edits note, consuming it (like the recovery file).
///
/// `None` when the last session had no unsaved terrain edits — or none at all,
/// which is the same thing from a recovery's point of view.
pub fn take_recovery_terrain_note(dir: &Path) -> Option<String> {
    let path = recovery_terrain_note_path(dir);
    let note = std::fs::read_to_string(&path).ok()?;
    let _ = std::fs::remove_file(&path);
    if note.trim().is_empty() {
        None
    } else {
        Some(note)
    }
}

/// If a recovery file exists, load it and delete it (consumed on startup so a
/// clean exit removes it). Returns `None` when there's nothing to recover.
///
/// Hardened (P15.2): a **corrupt / truncated** recovery file never panics and is
/// never silently dropped — it is moved aside to `crash-recovery.inf_lvl.corrupt`
/// and a warning is logged, so startup falls back cleanly to the last good save
/// while the bad file is preserved for diagnosis.
pub fn take_recovery(dir: &Path) -> Option<SceneDoc> {
    let path = recovery_path(dir);
    if !path.exists() {
        return None;
    }
    // Whatever happens to the payload, the terrain note is consumed and reported
    // exactly once — a recovery that silently kept it would warn again next boot.
    if let Some(note) = take_recovery_terrain_note(dir) {
        tracing::warn!("crash recovery: {note}");
    }
    match load(&path) {
        Ok(doc) => {
            let _ = std::fs::remove_file(&path);
            Some(doc)
        }
        Err(e) => {
            tracing::warn!(
                "crash-recovery file is corrupt ({e}); preserving it as .corrupt and \
                 falling back to the last saved level"
            );
            let aside = path.with_extension("inf_lvl.corrupt");
            if std::fs::rename(&path, &aside).is_err() {
                // If we cannot move it aside, remove it so it does not block the
                // next boot — but never panic.
                let _ = std::fs::remove_file(&path);
            }
            None
        }
    }
}

/// Remove the recovery file (called on a clean save / exit), and its
/// unsaved-terrain-edits note with it — a clean save wrote the terrain too.
pub fn clear_recovery(dir: &Path) {
    let _ = std::fs::remove_file(recovery_path(dir));
    let _ = std::fs::remove_file(recovery_terrain_note_path(dir));
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

    /// Order-independent parent lookup by GUID.
    fn parent_guid(doc: &SceneDoc, g: Uuid) -> Option<Uuid> {
        let e = doc.entity_of(g)?;
        doc.world()
            .parent_of(e)
            .and_then(|p| doc.world().guid_of(p))
    }

    /// A node reparented under a LATER-created node survives save→load with its
    /// hierarchy intact, and the reload re-saves byte-identically. Repro of the
    /// "drag Cube under a later Empty, reopen → Cube silently became a root" bug:
    /// `doc.order` is creation order (A before B), but A's parent (B) is created
    /// after A, so a single in-order spawn pass can't resolve it — the two-pass
    /// parent fix-up in [`apply_to_doc`] closes the gap.
    #[test]
    fn reparent_under_later_created_node_survives_round_trip() {
        let mut doc = SceneDoc::new();
        let a = doc.create(SpawnKind::Cube, "A", None);
        let b = doc.create(SpawnKind::Empty, "B", None); // created AFTER A
        assert!(doc.reparent(a, Some(b)), "A reparented under B");

        // The child A precedes its parent B in the file's entity sequence.
        let file1 = to_scene_file(&doc);
        let a_at = file1.entities.iter().position(|r| r.guid == a).unwrap();
        let b_at = file1.entities.iter().position(|r| r.guid == b).unwrap();
        assert!(a_at < b_at, "child A is written before its parent B");

        let bytes1 = encode(&file1).unwrap();
        let mut loaded = SceneDoc::new();
        apply_to_doc(&mut loaded, &decode(&bytes1).unwrap());

        // Hierarchy intact: A is still parented under B (not silently a root).
        assert_eq!(
            parent_guid(&loaded, a),
            Some(b),
            "A stays under B across save→load"
        );
        // And the reload re-saves byte-identically.
        let bytes2 = encode(&to_scene_file(&loaded)).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "reparented hierarchy save→load→save is byte-identical"
        );
    }

    /// A deeper case: a grandchild written before both its parent and grandparent
    /// (creation order C, B, A with A←B←C parenting) still rebuilds the full
    /// chain across the round trip.
    #[test]
    fn deep_reparent_chain_survives_round_trip() {
        let mut doc = SceneDoc::new();
        // Create so every child precedes its parent in creation order.
        let c = doc.create(SpawnKind::Cube, "C", None);
        let b = doc.create(SpawnKind::Empty, "B", None);
        let a = doc.create(SpawnKind::Empty, "A", None);
        assert!(doc.reparent(b, Some(a)));
        assert!(doc.reparent(c, Some(b)));

        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        let mut loaded = SceneDoc::new();
        apply_to_doc(&mut loaded, &decode(&bytes1).unwrap());

        assert_eq!(parent_guid(&loaded, c), Some(b), "C under B");
        assert_eq!(parent_guid(&loaded, b), Some(a), "B under A");
        assert_eq!(parent_guid(&loaded, a), None, "A is a root");

        let bytes2 = encode(&to_scene_file(&loaded)).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "deep reparented chain is byte-identical on re-save"
        );
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
        assert_eq!(
            bytes1[0], SCHEMA_VERSION as u8,
            "authored payload is written at the current schema"
        );
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

    /// P15.2: autosave fires (writes recovery) only when the doc is dirty — the
    /// `scene_autosave` command's `is_dirty()` gate, exercised at the core level.
    #[test]
    fn autosave_only_persists_a_dirty_doc() {
        let dir = tempfile::tempdir().unwrap();

        // A freshly-loaded/saved doc is clean → the command would skip; simulate
        // that gate here (a clean doc leaves no recovery file behind).
        let mut clean = SceneDoc::with_demo();
        clean.mark_saved();
        assert!(!clean.is_dirty(), "a saved doc is clean");
        if clean.is_dirty() {
            write_recovery(&clean, dir.path()).unwrap();
        }
        assert!(
            !recovery_path(dir.path()).exists(),
            "a clean doc must not autosave"
        );

        // A mutation dirties the doc → the command would write recovery.
        let mut doc = SceneDoc::with_demo();
        doc.mark_saved();
        doc.edit_create(SpawnKind::Cube, "Cube", None);
        assert!(doc.is_dirty(), "an edit dirties the doc");
        if doc.is_dirty() {
            write_recovery(&doc, dir.path()).unwrap();
        }
        assert!(
            recovery_path(dir.path()).exists(),
            "a dirty doc autosaves a recovery file"
        );
    }

    /// P15.2: a simulated crash → reopen recovers the exact pre-crash document.
    #[test]
    fn recovery_restores_the_pre_crash_document() {
        let dir = tempfile::tempdir().unwrap();

        // Author some work, then "autosave" (the state the process holds when it
        // dies without an explicit save).
        let mut doc = SceneDoc::with_demo();
        let c = doc.edit_create(SpawnKind::Cone, "Recovered Cone", None);
        let tp = transform_path(&doc);
        doc.write_prop(
            c,
            tp,
            "translation",
            &inf_ecs::PropValue::Vec3([7.0, 8.0, 9.0]),
        );
        write_recovery(&doc, dir.path()).unwrap();
        let pre_crash = encode(&to_scene_file(&doc)).unwrap();

        // "Crash" (drop the doc) then reboot: take_recovery yields an equal doc.
        drop(doc);
        let recovered = take_recovery(dir.path()).expect("recovered a doc");
        let post_crash = encode(&to_scene_file(&recovered)).unwrap();
        assert_eq!(
            pre_crash, post_crash,
            "recovered document must equal the pre-crash document"
        );
        assert!(
            !recovery_path(dir.path()).exists(),
            "the recovery file is consumed on successful recovery"
        );
    }

    /// P16.4b: the recovery **terrain note** is written beside the payload,
    /// consumed exactly once by a recovery, and cleared by a clean save — so the
    /// user is warned about lost terrain edits once and never again.
    #[test]
    fn the_terrain_note_rides_beside_recovery_and_is_consumed_once() {
        let dir = tempfile::tempdir().unwrap();
        let doc = SceneDoc::with_demo();
        assert!(take_recovery_terrain_note(dir.path()).is_none(), "no note");

        write_recovery(&doc, dir.path()).unwrap();
        write_recovery_terrain_note(dir.path(), Some("42 unsaved terrain tile(s)")).unwrap();
        assert!(recovery_terrain_note_path(dir.path()).exists());

        // Recovering consumes the note (and logs it).
        assert!(take_recovery(dir.path()).is_some());
        assert!(
            !recovery_terrain_note_path(dir.path()).exists(),
            "the note must be consumed with the recovery"
        );
        assert!(take_recovery_terrain_note(dir.path()).is_none());

        // A clean save clears both, so the next boot warns about nothing.
        write_recovery(&doc, dir.path()).unwrap();
        write_recovery_terrain_note(dir.path(), Some("still unsaved")).unwrap();
        clear_recovery(dir.path());
        assert!(!recovery_path(dir.path()).exists());
        assert!(!recovery_terrain_note_path(dir.path()).exists());

        // `None` clears rather than writing an empty note.
        write_recovery_terrain_note(dir.path(), Some("x")).unwrap();
        write_recovery_terrain_note(dir.path(), None).unwrap();
        assert!(take_recovery_terrain_note(dir.path()).is_none());
    }

    /// P15.2: a corrupt / truncated recovery file is handled gracefully — no
    /// panic, no crash — and the bad file is preserved as `.corrupt` while
    /// startup falls back to the last good save (`take_recovery` returns `None`).
    #[test]
    fn corrupt_recovery_file_is_handled_gracefully() {
        let dir = tempfile::tempdir().unwrap();

        // Write a valid recovery file, then truncate it to garbage.
        let doc = SceneDoc::with_demo();
        write_recovery(&doc, dir.path()).unwrap();
        let path = recovery_path(dir.path());
        std::fs::write(&path, b"\x00\x01\x02 not a valid inf_lvl payload").unwrap();

        // No panic; returns None (caller falls back to last good save).
        let recovered = take_recovery(dir.path());
        assert!(recovered.is_none(), "a corrupt file recovers nothing");
        assert!(
            !path.exists(),
            "the corrupt file is moved aside (not left in place)"
        );
        assert!(
            path.with_extension("inf_lvl.corrupt").exists(),
            "the corrupt file is preserved as .corrupt for diagnosis"
        );
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
                    mesh: Some(super::MeshRefV6 {
                        primitive: inf_ecs::components::Primitive::Plane,
                    }),
                    material: Some(MaterialV7 {
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
                    mesh: Some(super::MeshRefV6 {
                        primitive: inf_ecs::components::Primitive::Cube,
                    }),
                    material: Some(MaterialV7::default()),
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
                    light: Some(LightV7 {
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
            render: RenderSettingsRecord::default(),
            partition: PartitionSettings::default(),
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
            bytes1[0], SCHEMA_VERSION as u8,
            "the physics/actor content is written at the current schema"
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
                    mesh: Some(super::MeshRefV6 {
                        primitive: inf_ecs::components::Primitive::Plane,
                    }),
                    material: Some(MaterialV7::default()),
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
            settings: LevelSettingsV7 {
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
            bytes1[0], SCHEMA_VERSION as u8,
            "anim content is written at the current schema"
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
            settings: LevelSettingsV7::default(),
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
                other: inf_ecs::EntityRef::new(other_guid),
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
                other: inf_ecs::EntityRef::new(other_guid),
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
        assert_eq!(
            bytes1[0], SCHEMA_VERSION as u8,
            "joints/audio content is written at the current schema"
        );
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
        assert_eq!(j3.other, inf_ecs::EntityRef::new(other_guid));
        assert_eq!(j3.kind, JointKind3D::Revolute);
        assert!(j3.motor_enabled);
        assert_eq!(j3.motor_target_vel, 8.0);
        let j2 = w.get::<Joint2D>(e2).expect("joint_2d persists");
        assert_eq!(j2.other, inf_ecs::EntityRef::new(other_guid));
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
                other: inf_ecs::EntityRef::new(uuid::Uuid::from_u128(0x12_A0)),
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
            settings: LevelSettingsV7::default(),
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

    // ── v7 (pre-R-P0) forever-load fixture discipline ──────────────────────

    use inf_ecs::components::{LightKind, MeshRef, Primitive};

    /// A minimal all-`None` frozen v7 record, filled in via struct-update syntax
    /// by [`v7_reference`] (`EntityRecordV7` intentionally has no `Default`, like
    /// the other frozen records — this local helper stands in).
    fn v7_base(guid: uuid::Uuid, name: &str, parent: Option<uuid::Uuid>) -> EntityRecordV7 {
        EntityRecordV7 {
            guid,
            name: name.into(),
            parent,
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

    /// Rebuild the exact schema-v7 `SceneFile` the v7 fixture was generated from,
    /// from the frozen [`EntityRecordV7`]/[`SceneFileV7`] types (the provenance
    /// lock). Covers every current requirement: a `Material` (frozen [`MaterialV7`]),
    /// a `Light` of each kind (frozen [`LightV7`]), a `MeshRef` of each primitive,
    /// a parent link (Cone under Cube), a `Joint3D`, and non-default settings.
    fn v7_reference() -> SceneFileV7 {
        let g = uuid::Uuid::from_u128;
        let cube = g(0x7002);
        let mesh = |p| {
            Some(MeshRef {
                primitive: p,
                asset: None,
            })
        };
        SceneFileV7 {
            schema_version: 7,
            title: "V7 Fixture Level".into(),
            entities: vec![
                EntityRecordV7 {
                    transform: EcsTransform {
                        translation: inf_ecs::math::Vec3d::ZERO,
                        rotation: inf_ecs::math::Vec3d::ZERO,
                        scale: inf_ecs::math::Vec3d::new(20.0, 1.0, 20.0),
                    },
                    mesh: mesh(Primitive::Plane),
                    material: Some(MaterialV7 {
                        base_color: Color::new(0.3, 0.32, 0.35, 1.0),
                        ..Default::default()
                    }),
                    ..v7_base(g(0x7001), "Ground", None)
                },
                EntityRecordV7 {
                    mesh: mesh(Primitive::Cube),
                    material: Some(MaterialV7::default()),
                    ..v7_base(cube, "Cube", None)
                },
                EntityRecordV7 {
                    mesh: mesh(Primitive::Sphere),
                    ..v7_base(g(0x7003), "Sphere", None)
                },
                EntityRecordV7 {
                    mesh: mesh(Primitive::Cylinder),
                    ..v7_base(g(0x7004), "Cylinder", None)
                },
                // Cone parented under Cube (parent link) + a Joint3D to the cube.
                EntityRecordV7 {
                    mesh: mesh(Primitive::Cone),
                    joint_3d: Some(Joint3D {
                        other: inf_ecs::EntityRef::new(cube),
                        kind: JointKind3D::Revolute,
                        ..Default::default()
                    }),
                    ..v7_base(g(0x7005), "Cone", Some(cube))
                },
                EntityRecordV7 {
                    light: Some(LightV7 {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 1.0,
                    }),
                    ..v7_base(g(0x7006), "Sun", None)
                },
                EntityRecordV7 {
                    light: Some(LightV7 {
                        kind: LightKind::Point,
                        color: Color::new(1.0, 0.9, 0.8, 1.0),
                        intensity: 2.0,
                    }),
                    ..v7_base(g(0x7007), "Lamp", None)
                },
                EntityRecordV7 {
                    light: Some(LightV7 {
                        kind: LightKind::Spot,
                        color: Color::WHITE,
                        intensity: 3.0,
                    }),
                    ..v7_base(g(0x7008), "Spot", None)
                },
            ],
            settings: LevelSettingsV7 {
                gravity_2d: Vec2d::new(0.0, -20.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 120.0,
            },
        }
    }

    /// Write the committed v7 fixture from [`v7_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline).
    #[test]
    fn bless_v7_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v7_reference(), bincode_config()).unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v7.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v7 fixture");
        eprintln!("blessed v7 fixture: {}", path.display());
    }

    #[test]
    fn v7_fixture_is_reproducible_and_genuinely_v7() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v7.inf_lvl");
        if !path.exists() {
            eprintln!("SKIP: v7 fixture not blessed yet ({})", path.display());
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes[0], 7, "fixture must be a genuine schema-v7 payload");
        let rebuilt = bincode::serde::encode_to_vec(v7_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v7 fixture must match our frozen v7 writer"
        );
    }

    #[test]
    fn v7_fixture_loads_forever_and_lifts_to_v8() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v7.inf_lvl");
        if !path.exists() {
            eprintln!("SKIP: v7 fixture not blessed yet ({})", path.display());
            return;
        }
        let file = decode(&std::fs::read(&path).unwrap()).expect("v7 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V7 Fixture Level");
        assert_eq!(file.entities.len(), 8);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();
        // v7 data preserved through the v7→v8 lift.
        assert!(by_name("Ground").material.is_some());
        assert!(by_name("Cone").joint_3d.is_some());
        assert_eq!(by_name("Cone").parent, Some(uuid::Uuid::from_u128(0x7002)));
        // Every new v8 material/light field lifts to its documented default …
        let m = by_name("Ground").material.unwrap();
        assert_eq!(m.blend, BlendMode::Opaque);
        assert_eq!(m.alpha_cutoff, 0.5);
        for name in ["Sun", "Lamp", "Spot"] {
            let l = by_name(name).light.unwrap();
            assert_eq!(l.range, 0.0);
            assert_eq!(l.inner_cone_deg, 30.0);
            assert_eq!(l.outer_cone_deg, 40.0);
            assert!(l.cast_shadows);
        }
        // … the four new entity slots default to None …
        for r in &file.entities {
            assert!(r.decal.is_none());
            assert!(r.volume.is_none());
            assert!(r.spline.is_none());
            assert!(r.foliage.is_none());
        }
        // … the non-default file settings carry through, and the render block
        // lifts to its default.
        assert_eq!(file.settings.gravity_2d, Vec2d::new(0.0, -20.0));
        assert_eq!(file.settings.sim_hz, 120.0);
        assert_eq!(file.settings.render, RenderSettingsRecord::default());
    }

    // ── v8 forever-load fixture (frozen pre-v9) ─────────────────────────────

    /// A deterministic authored terrain for the v8 fixture: two shared-edge tiles
    /// written from one **polynomial** height field (never `sin`/`cos` — `std`
    /// trig is not bit-portable across targets, the P14 law), plus a painted splat
    /// weight so the fixture exercises a materialized weight buffer.
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

    /// A minimal all-`None` frozen v8 entity record ([`EntityRecordV8`] has no
    /// `Default`, like the other frozen records — this local helper stands in).
    fn v8_base(guid: uuid::Uuid, name: &str, parent: Option<uuid::Uuid>) -> EntityRecordV8 {
        EntityRecordV8 {
            guid,
            name: name.into(),
            parent,
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

    /// Rebuild the exact schema-v8 file the committed v8 fixture was generated
    /// from, out of the frozen v8 record types (the provenance lock). Carries a v8
    /// `Material` (blend + cutoff), a v8 `Light` (range + cones + shadows), the
    /// four v8 world-decoration slots and — the point of this fixture for P16.3 —
    /// a **populated `Terrain`**, pinning the pre-v9 `Terrain` byte layout.
    fn v8_reference() -> SceneFileV8 {
        use inf_ecs::components::{
            BlendMode, Decal, Foliage, FoliageInstance, FoliagePaletteEntry, Light, LightKind,
            Material, Spline, SplineInterp, Volume, VolumeKind,
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
                    ..v8_base(g(0x8001), "Ground", None)
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
                    ..v8_base(cube, "Cube", None)
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
                    ..v8_base(g(0x8008), "Spot", Some(cube))
                },
                EntityRecordV8 {
                    terrain: Some(TerrainV8::from_current(fixture_terrain())),
                    ..v8_base(g(0x8009), "Terrain", None)
                },
            ],
            settings: LevelSettingsV9 {
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

    /// Write the committed v8 fixture from [`v8_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline). Never hand-edit
    /// the committed bytes.
    #[test]
    fn bless_v8_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v8_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 8);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v8.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v8 fixture");
        eprintln!("blessed v8 fixture: {}", path.display());
    }

    #[test]
    fn v8_fixture_is_reproducible_and_genuinely_v8() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v8.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v8 fixture present");
        assert_eq!(bytes[0], 8, "fixture must be a genuine schema-v8 payload");
        let rebuilt = bincode::serde::encode_to_vec(v8_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v8 fixture must match our frozen v8 writer"
        );
    }

    /// The committed v8 fixture — written by the **pre-v9 codec**, before
    /// `Terrain` grew its asset reference — still loads, with the terrain lifted
    /// through the frozen [`TerrainV8`] record and `asset` defaulted to `None`.
    /// The "old bytes load forever" gate for the v9 bump.
    #[test]
    fn v8_fixture_loads_forever_and_lifts_to_v9() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v8.inf_lvl");
        let file = decode(&std::fs::read(&path).unwrap()).expect("v8 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V8 Fixture Level");
        assert_eq!(file.entities.len(), 4);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        // The terrain survives the frozen-record hop intact …
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
        // … and the new v9 field lifts to its documented default: a pre-v9 level's
        // terrain is inline-authoritative, which is exactly what it always meant.
        assert_eq!(terrain.asset, None);

        // The rest of the v8 slot set carries through untouched.
        assert_eq!(by_name("Ground").material.unwrap().blend, BlendMode::Masked);
        assert_eq!(by_name("Spot").light.unwrap().range, 25.0);
        assert_eq!(by_name("Cube").foliage.as_ref().unwrap().instances.len(), 2);
        assert_eq!(file.settings.render.exposure, 1.4);

        // Rebuilding a doc from it and re-saving yields a current-schema file that
        // round-trips byte-identically (the load → save → load identity).
        let mut doc = SceneDoc::new();
        apply_to_doc(&mut doc, &file);
        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(bytes1[0], SCHEMA_VERSION as u8);
        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &decode(&bytes1).unwrap());
        assert_eq!(encode(&to_scene_file(&doc2)).unwrap(), bytes1);
    }

    // ── v9 (P16.3) terrain asset reference ─────────────────────────────────

    /// A `.inf_terrain` asset reference on a `Terrain` persists across
    /// save → load and re-encodes byte-identically — and (P16.4b) the level
    /// **does not** carry the working set beside it: the asset is the authority,
    /// so the tiles the editor paged in to sculpt are stripped on write.
    #[test]
    fn terrain_asset_reference_persists_across_save_load_v9() {
        let asset_guid = uuid::Uuid::from_u128(0x1603_00AA);
        let mut doc = SceneDoc::new();
        let e = doc.create(SpawnKind::Empty, "Streamed Terrain", None);
        insert!(
            doc,
            e,
            Terrain {
                asset: Some(asset_guid),
                ..fixture_terrain()
            }
        );
        doc.world_mut().propagate();

        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0], SCHEMA_VERSION as u8,
            "encode always writes the current schema"
        );
        let mut loaded = SceneDoc::new();
        apply_to_doc(&mut loaded, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&loaded)).unwrap();
        assert_eq!(bytes1, bytes2, "save→load→save must be byte-identical");

        let file = to_scene_file(&loaded);
        let t = file.entities[0]
            .terrain
            .as_ref()
            .expect("terrain persisted");
        assert_eq!(t.asset, Some(asset_guid), "the asset ref survives the trip");
        assert_eq!(
            t.data.tile_count(),
            0,
            "a streamed terrain's working set must never reach the .inf_lvl"
        );
        // The grid configuration still does — a streamed terrain still needs to
        // know its own resolution/spacing before the first page arrives.
        assert_eq!(t.tile_resolution, fixture_terrain().tile_resolution);
        assert_eq!(t.meters_per_sample, fixture_terrain().meters_per_sample);

        // The IN-MEMORY component is untouched by the write: stripping is a
        // serialization rule, not a mutation (the working set is still there to
        // sculpt and still there to write back).
        let live = doc.terrain_data_and_origin(e).expect("terrain in the doc");
        assert_eq!(live.0.tile_count(), 2, "the write must not mutate the doc");
        assert_eq!(
            live.0.get_tile((0, 0)).unwrap().weight_sample(4, 1, 2),
            [40, 100, 80, 35]
        );

        // Clearing the reference produces different bytes (so the ref really is
        // persisted, not silently dropped) and still round-trips.
        let mut inline_doc = SceneDoc::new();
        let e = inline_doc.create(SpawnKind::Empty, "Streamed Terrain", None);
        insert!(inline_doc, e, fixture_terrain());
        inline_doc.world_mut().propagate();
        let inline = encode(&to_scene_file(&inline_doc)).unwrap();
        assert_ne!(inline, bytes1, "the asset ref is really in the bytes");
    }

    /// The terrain slot survives the human-readable codec too (the dual-format
    /// rule) — including the new `asset` reference.
    #[test]
    fn terrain_asset_reference_is_dual_format_serde_safe() {
        let t = Terrain {
            asset: Some(uuid::Uuid::from_u128(0x1603_00BB)),
            ..fixture_terrain()
        };
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(serde_json::from_str::<Terrain>(&json).unwrap(), t);
        // A pre-v9 JSON object (no `asset` key) decodes with the field defaulted.
        let stripped = json.replace(&format!(",\"asset\":\"{}\"", t.asset.unwrap()), "");
        assert_ne!(stripped, json, "the asset key is present in the JSON");
        let back: Terrain = serde_json::from_str(&stripped).unwrap();
        assert_eq!(back.asset, None);
        assert_eq!(back.data, t.data);
    }

    // ── v8 (R-P0) world-decoration + render-settings persistence ───────────

    /// A spot light with cones + range, a translucent material, a Decal, a Volume,
    /// a Spline, a 3-instance Foliage, and non-default render settings all persist
    /// across save → load and re-encode byte-identically.
    #[test]
    fn round_trip_with_v8_components_is_byte_identical() {
        use inf_ecs::components::{
            BlendMode, Decal, Foliage, FoliageInstance, FoliagePaletteEntry, Light, LightKind,
            Material, Spline, SplineInterp, Volume, VolumeKind,
        };

        let mut doc = SceneDoc::new();
        doc.set_title("V8 Level");
        // Non-default render settings on the file.
        doc.set_settings(LevelSettings {
            partition: PartitionSettings::default(),
            gravity_2d: Vec2d::new(0.0, -18.0),
            gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
            sim_hz: 90.0,
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
        });

        let spot = doc.create(SpawnKind::Empty, "Spot", None);
        insert!(
            doc,
            spot,
            Light {
                kind: LightKind::Spot,
                color: Color::new(1.0, 0.8, 0.6, 1.0),
                intensity: 4.0,
                range: 25.0,
                inner_cone_deg: 18.0,
                outer_cone_deg: 32.0,
                cast_shadows: false,
            }
        );

        let surf = doc.create(SpawnKind::Cube, "Glass", None);
        insert!(
            doc,
            surf,
            Material {
                base_color: Color::new(0.2, 0.5, 0.9, 0.4),
                metallic: 0.0,
                roughness: 0.1,
                emissive: Color::new(0.0, 0.0, 0.0, 1.0),
                blend: BlendMode::Translucent,
                alpha_cutoff: 0.3,
            }
        );

        let deco = doc.create(SpawnKind::Empty, "Decals", None);
        insert!(
            doc,
            deco,
            Decal {
                size: Vec3d::new(3.0, 1.0, 3.0),
                color: Color::new(0.1, 0.1, 0.1, 1.0),
                opacity: 0.8,
                fade_angle_deg: 50.0,
            }
        );
        insert!(
            doc,
            deco,
            Volume {
                kind: VolumeKind::Blocking,
                tint: Color::new(0.9, 0.2, 0.2, 0.5),
            }
        );

        let path = doc.create(SpawnKind::Empty, "Rail", None);
        insert!(
            doc,
            path,
            Spline {
                points: vec![
                    Vec3d::ZERO,
                    Vec3d::new(2.0, 0.0, 1.0),
                    Vec3d::new(4.0, 1.0, 0.0),
                ],
                closed: true,
                interp: SplineInterp::Linear,
            }
        );

        let scatter = doc.create(SpawnKind::Empty, "Grass", None);
        insert!(
            doc,
            scatter,
            Foliage {
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
            }
        );

        doc.world_mut().propagate();

        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0], SCHEMA_VERSION as u8,
            "v8 content is written at the current schema"
        );
        let mut loaded = SceneDoc::new();
        apply_to_doc(&mut loaded, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&loaded)).unwrap();
        assert_eq!(bytes1, bytes2, "v8 save→load→save must be byte-identical");

        // Spot-check the reloaded values.
        let file = to_scene_file(&loaded);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();
        let l = by_name("Spot").light.unwrap();
        assert_eq!(l.range, 25.0);
        assert_eq!(l.inner_cone_deg, 18.0);
        assert!(!l.cast_shadows);
        let m = by_name("Glass").material.unwrap();
        assert_eq!(m.blend, BlendMode::Translucent);
        assert_eq!(m.alpha_cutoff, 0.3);
        assert_eq!(by_name("Decals").decal.unwrap().opacity, 0.8);
        assert_eq!(by_name("Decals").volume.unwrap().kind, VolumeKind::Blocking);
        let sp = by_name("Rail").spline.as_ref().unwrap();
        assert_eq!(sp.points.len(), 3);
        assert!(sp.closed);
        let fo = by_name("Grass").foliage.as_ref().unwrap();
        assert_eq!(fo.instances.len(), 3);
        assert_eq!(fo.palette.len(), 2);
        assert_eq!(loaded.settings().render.exposure, 1.4);
        assert!(loaded.settings().render.gi_enabled);
    }

    /// The v8 scene also survives TOML/JSON (the dual-format rule) — the new
    /// components + render block must round-trip in the human-readable codec too.
    #[test]
    fn v8_render_settings_are_dual_format_serde_safe() {
        let mut s = LevelSettings::default();
        s.render.exposure = 2.0;
        s.render.shadows_enabled = true;
        let toml_s = toml::to_string(&s).unwrap();
        let back: LevelSettings = toml::from_str(&toml_s).unwrap();
        assert_eq!(back, s);
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<LevelSettings>(&json).unwrap(), s);
        // The default record equals inf-render's RenderSettings::default() mapping.
        let d = RenderSettingsRecord::default();
        assert_eq!(d.exposure, 1.0);
        assert!(d.dither && !d.bloom_enabled && !d.ssao_enabled && !d.taa);
        assert_eq!(d.bloom_threshold, 1.0);
        assert_eq!(d.bloom_knee, 0.5);
        assert_eq!(d.bloom_intensity, 0.06);
        assert_eq!(d.ssao_radius, 0.6);
        assert_eq!(d.ssao_intensity, 1.0);
        assert_eq!(d.ssao_bias, 0.025);
        assert!(!d.shadows_enabled && !d.gi_enabled);
        assert_eq!(d.shadows_max_distance, 60.0);
        assert_eq!(d.gi_intensity, 1.0);
    }

    // ── v9 forever-load fixture + v10 (P16.5) world partition ───────────────

    /// A minimal all-`None` frozen v9 entity record.
    fn v9_base(guid: uuid::Uuid, name: &str, parent: Option<uuid::Uuid>) -> EntityRecordV9 {
        EntityRecordV9 {
            guid,
            name: name.into(),
            parent,
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

    /// Rebuild the exact schema-v9 file the committed v9 fixture was generated
    /// from, out of the frozen v9 record types (the provenance lock). Carries a
    /// **v9 `Terrain` with an asset reference** — the thing v9 added — plus a mesh
    /// with an asset ref and a light, so the pre-v10 entity + settings byte
    /// layouts are pinned by committed bytes.
    fn v9_reference() -> SceneFileV9 {
        use inf_ecs::components::{Light, LightKind, Material, Primitive};
        let g = uuid::Uuid::from_u128;
        SceneFileV9 {
            schema_version: 9,
            title: "V9 Fixture Level".into(),
            entities: vec![
                EntityRecordV9 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0x90A1)),
                    }),
                    material: Some(Material::default()),
                    ..v9_base(g(0x9001), "Cube", None)
                },
                EntityRecordV9 {
                    // A frozen record carries the frozen component: v9's terrain
                    // predates P19.1's data maps, so it is a `TerrainV14`.
                    terrain: Some(TerrainV14::from_current(Terrain {
                        asset: Some(g(0x9_00AA)),
                        ..fixture_terrain()
                    })),
                    ..v9_base(g(0x9002), "Terrain", None)
                },
                EntityRecordV9 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v9_base(g(0x9003), "Sun", None)
                },
            ],
            settings: LevelSettingsV9 {
                gravity_2d: Vec2d::new(0.0, -18.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 90.0,
                render: RenderSettingsRecord {
                    exposure: 1.1,
                    ..RenderSettingsRecord::default()
                },
            },
        }
    }

    /// Write the committed v9 fixture from [`v9_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline). Never hand-edit
    /// the committed bytes.
    #[test]
    fn bless_v9_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v9_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 9);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v9.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v9 fixture");
        eprintln!("blessed v9 fixture: {}", path.display());
    }

    #[test]
    fn v9_fixture_is_reproducible_and_genuinely_v9() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v9.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v9 fixture present");
        assert_eq!(bytes[0], 9, "fixture must be a genuine schema-v9 payload");
        let rebuilt = bincode::serde::encode_to_vec(v9_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v9 fixture must match our frozen v9 writer"
        );
    }

    /// The committed v9 fixture — written by the **pre-v10 codec**, before the
    /// entity record grew its two world-partition slots and the settings grew
    /// their partition block — still loads, with every v10 field at its
    /// documented default. The "old bytes load forever" gate for the v10 bump.
    #[test]
    fn v9_fixture_loads_forever_and_lifts_to_v10() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v9.inf_lvl");
        let file = decode(&std::fs::read(&path).unwrap()).expect("v9 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V9 Fixture Level");
        assert_eq!(file.entities.len(), 3);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        // The v9 content survives the frozen-record hop intact …
        let terrain = by_name("Terrain").terrain.as_ref().expect("terrain slot");
        assert_eq!(terrain.asset, Some(uuid::Uuid::from_u128(0x9_00AA)));
        assert_eq!(terrain.data.tile_count(), 2);
        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(uuid::Uuid::from_u128(0x90A1))
        );
        assert_eq!(file.settings.sim_hz, 90.0);
        assert_eq!(file.settings.render.exposure, 1.1);

        // … and every v10 field lifts to its documented default: no streaming
        // sources, nothing pinned always-loaded, partitioning OFF.
        for e in &file.entities {
            assert!(e.streaming_source.is_none());
            assert!(e.always_loaded.is_none());
        }
        assert_eq!(file.settings.partition, PartitionSettings::default());
        assert!(!file.settings.partition.enabled);

        // Rebuilding a doc from it and re-saving yields a current-schema file that
        // round-trips byte-identically (the load → save → load identity).
        let mut doc = SceneDoc::new();
        apply_to_doc(&mut doc, &file);
        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(bytes1[0], SCHEMA_VERSION as u8);
        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &decode(&bytes1).unwrap());
        assert_eq!(encode(&to_scene_file(&doc2)).unwrap(), bytes1);
    }

    /// The two v10 component slots and the file-level partition block persist
    /// across save → load and re-encode byte-identically — including through the
    /// live ECS (`record_of` reads them, `write_record_components` writes them).
    #[test]
    fn v10_partition_components_and_settings_round_trip() {
        use inf_ecs::components::{AlwaysLoaded, StreamingSource};

        let mut doc = SceneDoc::new();
        doc.set_title("Partitioned Level");
        doc.set_settings(LevelSettings {
            partition: PartitionSettings {
                enabled: true,
                cell_size_m: 128.0,
                activation_radius_m: 200.0,
                prefetch_margin_m: 300.0,
            },
            ..LevelSettings::default()
        });
        let player = doc.create(SpawnKind::Empty, "Player", None);
        insert!(doc, player, StreamingSource { radius_m: 384.0 });
        let manager = doc.create(SpawnKind::Empty, "GameMode", None);
        insert!(doc, manager, AlwaysLoaded);
        doc.create(SpawnKind::Empty, "Prop", None);
        doc.world_mut().propagate();

        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(
            bytes1[0], SCHEMA_VERSION as u8,
            "a partitioned level writes a current-schema payload"
        );
        let mut loaded = SceneDoc::new();
        apply_to_doc(&mut loaded, &decode(&bytes1).unwrap());
        let bytes2 = encode(&to_scene_file(&loaded)).unwrap();
        assert_eq!(bytes1, bytes2, "v10 save→load→save must be byte-identical");

        let file = to_scene_file(&loaded);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();
        assert_eq!(
            by_name("Player").streaming_source.unwrap().radius_m,
            384.0,
            "the streaming source survived the ECS round trip"
        );
        assert_eq!(by_name("GameMode").always_loaded, Some(AlwaysLoaded));
        assert!(by_name("Prop").streaming_source.is_none());
        assert!(by_name("Prop").always_loaded.is_none());
        assert_eq!(file.settings.partition.cell_size_m, 128.0);
        assert_eq!(file.settings.partition.activation_radius_m, 200.0);
        assert_eq!(file.settings.partition.prefetch_margin_m, 300.0);
        assert!(file.settings.partition.enabled);

        // The settings block is really persisted, not inferred: turning it off
        // moves the bytes.
        let mut off = loaded;
        off.set_settings(LevelSettings::default());
        assert_ne!(encode(&to_scene_file(&off)).unwrap(), bytes1);
    }

    /// This Ring-1 codec's [`PartitionSettings`] mirror must stay field-for-field
    /// and default-for-default identical to the Ring-0 runtime one, or a level
    /// written here decodes to different settings there — silently, since bincode
    /// is not self-describing. The editor cannot depend on `inf-scene` (ring
    /// inversion), so the mirror is asserted against the documented constants the
    /// runtime publishes, and the byte-level cross-check lives in `inf-scene`'s
    /// cross-decode test over the committed samples.
    #[test]
    fn partition_settings_mirror_matches_the_runtime_defaults() {
        let d = PartitionSettings::default();
        assert!(
            !d.enabled,
            "a level is unpartitioned until an author says so"
        );
        assert_eq!(d.cell_size_m, 256.0);
        assert_eq!(d.activation_radius_m, 256.0);
        assert_eq!(d.prefetch_margin_m, 256.0);
        // Every field carries `#[serde(default)]`, so a partial (human-readable)
        // payload fills the same values.
        let partial: PartitionSettings = serde_json::from_str("{}").unwrap();
        assert_eq!(partial, d);
        let one: PartitionSettings = serde_json::from_str(r#"{"enabled":true}"#).unwrap();
        assert!(one.enabled);
        assert_eq!(one.cell_size_m, 256.0);
    }

    // ── schema v11 (P17.1 sky authority) ──────────────────────────────────

    /// An all-`None` frozen v10 entity — the struct-update base for
    /// [`v10_reference`]. Built through the downgrade hop so the field list can
    /// never drift from the live record.
    fn v10_base(guid: uuid::Uuid, name: &str, parent: Option<uuid::Uuid>) -> EntityRecordV10 {
        EntityRecordV10::from_current(
            v9_base(guid, name, parent)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current(),
        )
    }

    /// Rebuild the exact schema-v10 file the committed v10 fixture was generated
    /// from, out of the frozen v10 record types (the provenance lock). Carries the
    /// **v10** additions (a streaming source, an always-loaded marker, a
    /// partitioned settings block) plus a mesh and a light, so the pre-v11 entity
    /// byte layout is pinned by committed bytes.
    fn v10_reference() -> SceneFileV10 {
        use inf_ecs::components::{
            AlwaysLoaded, Light, LightKind, Material, Primitive, StreamingSource,
        };
        let g = uuid::Uuid::from_u128;
        SceneFileV10 {
            schema_version: 10,
            title: "V10 Fixture Level".into(),
            entities: vec![
                EntityRecordV10 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xA0A1)),
                    }),
                    material: Some(Material::default()),
                    streaming_source: Some(StreamingSource { radius_m: 300.0 }),
                    ..v10_base(g(0xA001), "Player", None)
                },
                EntityRecordV10 {
                    always_loaded: Some(AlwaysLoaded),
                    ..v10_base(g(0xA002), "GameMode", None)
                },
                EntityRecordV10 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v10_base(g(0xA003), "Sun", None)
                },
            ],
            settings: LevelSettings {
                gravity_2d: Vec2d::new(0.0, -18.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 90.0,
                render: RenderSettingsRecord {
                    exposure: 1.1,
                    ..RenderSettingsRecord::default()
                },
                partition: PartitionSettings {
                    enabled: true,
                    cell_size_m: 128.0,
                    activation_radius_m: 200.0,
                    prefetch_margin_m: 300.0,
                },
            },
        }
    }

    /// Write the committed v10 fixture from [`v10_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline). Never hand-edit
    /// the committed bytes.
    #[test]
    fn bless_v10_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v10_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 10);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v10.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v10 fixture");
        eprintln!("blessed v10 fixture: {}", path.display());
    }

    #[test]
    fn v10_fixture_is_reproducible_and_genuinely_v10() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v10.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v10 fixture present");
        assert_eq!(bytes[0], 10, "fixture must be a genuine schema-v10 payload");
        let rebuilt = bincode::serde::encode_to_vec(v10_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v10 fixture must match our frozen v10 writer"
        );
    }

    /// This crate's committed v10 fixture must be **byte-identical** to the Ring-0
    /// runtime reader's, because the two codecs are one wire contract written
    /// twice. A divergence here is the exact bug the mirror doctrine exists to
    /// catch, and it would otherwise only surface as a player that cannot open a
    /// level the editor just saved.
    #[test]
    fn v10_fixture_matches_the_runtime_codecs_copy() {
        let mine = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v10.inf_lvl");
        let theirs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../crates/inf-scene/tests/fixtures/scene_v10.inf_lvl");
        assert_eq!(
            std::fs::read(&mine).expect("editor v10 fixture"),
            std::fs::read(&theirs).expect("runtime v10 fixture"),
            "the two v11-bump fixtures diverged — the codecs are no longer mirrors"
        );
    }

    /// The committed v10 fixture — written by the **pre-v11 codec**, before the
    /// entity record grew its two sky-authority slots — still loads, with both new
    /// slots at their documented default. The "old bytes load forever" gate for
    /// the v11 bump.
    #[test]
    fn v10_fixture_loads_forever_and_lifts_to_v11() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v10.inf_lvl");
        let file = decode(&std::fs::read(&path).unwrap()).expect("v10 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V10 Fixture Level");
        assert_eq!(file.entities.len(), 3);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        // The v10 content survives the frozen-record hop intact …
        assert_eq!(by_name("Player").streaming_source.unwrap().radius_m, 300.0);
        assert!(by_name("GameMode").always_loaded.is_some());
        assert_eq!(
            by_name("Player").mesh.unwrap().asset,
            Some(uuid::Uuid::from_u128(0xA0A1))
        );
        assert_eq!(file.settings.sim_hz, 90.0);
        assert!(file.settings.partition.enabled);
        assert_eq!(file.settings.partition.cell_size_m, 128.0);

        // … and every v11 slot lifts to its documented default: no clock at all,
        // which is what makes a pre-v11 level render under the retired sun.
        for e in &file.entities {
            assert!(e.time_of_day.is_none());
            assert!(e.sky_atmosphere.is_none());
        }

        // Load → save → load is byte-identical at the current schema.
        let mut doc = SceneDoc::new();
        apply_to_doc(&mut doc, &file);
        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(bytes1[0], SCHEMA_VERSION as u8);
        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &decode(&bytes1).unwrap());
        assert_eq!(encode(&to_scene_file(&doc2)).unwrap(), bytes1);
    }

    /// The **downgrade-bless** direction for the v10 entity record, as a checked
    /// property rather than a path only `INF_BLESS_FIXTURES=1` walks.
    #[test]
    fn v10_entity_downgrade_is_lossless_except_for_the_sky_slots() {
        use inf_ecs::components::{AlwaysLoaded, StreamingSource};
        let g = uuid::Uuid::from_u128;
        let live = EntityRecord {
            streaming_source: Some(StreamingSource { radius_m: 42.0 }),
            always_loaded: Some(AlwaysLoaded),
            time_of_day: Some(TimeOfDay {
                seconds: 1234.0,
                rate: 60.0,
                ..TimeOfDay::default()
            }),
            sky_atmosphere: Some(SkyAtmosphere::default()),
            ..v9_base(g(0xB001), "Sky", None)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current()
        };
        let back = EntityRecordV10::from_current(live.clone())
            .into_v11()
            .into_v12()
            .into_v13()
            .into_current();
        assert_eq!(back.streaming_source, live.streaming_source);
        assert_eq!(back.always_loaded, live.always_loaded);
        assert_eq!(back.name, live.name);
        assert!(
            back.time_of_day.is_none() && back.sky_atmosphere.is_none(),
            "the sky slots have no v10 home and must come back empty"
        );
        // A record with no clock survives the hop exactly.
        let plain = v9_base(g(0xB002), "Prop", None)
            .into_v10()
            .into_v11()
            .into_v12()
            .into_v13()
            .into_current();
        assert_eq!(
            EntityRecordV10::from_current(plain.clone())
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current(),
            plain
        );
    }

    /// The two v11 component slots persist across save → load and re-encode
    /// byte-identically — including through the live ECS (`record_of` reads them,
    /// `write_record_components` writes them).
    #[test]
    fn v11_sky_components_round_trip() {
        let mut doc = SceneDoc::new();
        doc.set_title("Sky Level");
        let sky = doc.create(SpawnKind::Empty, "Sky", None);
        let tod = TimeOfDay {
            seconds: 3_600.0,
            day_of_year: 355,
            latitude_deg: -33.9,
            longitude_deg: 151.2,
            rate: 120.0,
        };
        let atmos = SkyAtmosphere {
            sun_intensity: 5.5,
            night_darkening: 0.4,
            ..SkyAtmosphere::default()
        };
        {
            let e = doc.world().entity_of(sky).unwrap();
            let w = doc.world_mut().world_mut();
            w.entity_mut(e).insert(tod);
            w.entity_mut(e).insert(atmos);
        }
        doc.create(SpawnKind::Cube, "Prop", None);

        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(bytes1[0], SCHEMA_VERSION as u8);
        let file = decode(&bytes1).unwrap();
        let rec = file.entities.iter().find(|r| r.name == "Sky").unwrap();
        assert_eq!(rec.time_of_day, Some(tod));
        assert_eq!(rec.sky_atmosphere, Some(atmos));
        assert!(file
            .entities
            .iter()
            .find(|r| r.name == "Prop")
            .unwrap()
            .time_of_day
            .is_none());

        // Save → load → save is byte-identical (the single-byte standard).
        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &file);
        assert_eq!(encode(&to_scene_file(&doc2)).unwrap(), bytes1);

        // The components really reached the ECS, not just the record.
        let e = doc2.world().entity_of(sky).unwrap();
        assert_eq!(doc2.world().world().get::<TimeOfDay>(e).copied(), Some(tod));
        assert_eq!(
            doc2.world().world().get::<SkyAtmosphere>(e).copied(),
            Some(atmos)
        );
    }

    /// The `Ring-0 ↔ Ring-1` default lock for the new components: `inf-scene` and
    /// this codec must agree field-for-field, because the two shapes staying
    /// identical *is* the wire contract (the same reason
    /// `partition_settings_mirror_matches_the_runtime_defaults` exists).
    #[test]
    fn sky_component_defaults_are_the_documented_ones() {
        let t = TimeOfDay::default();
        assert_eq!(t.seconds, 36_000.0);
        assert_eq!(t.day_of_year, 172);
        assert_eq!(t.latitude_deg, 48.9);
        assert_eq!(t.longitude_deg, 0.0);
        assert_eq!(t.rate, 0.0, "a level opts into a moving sun explicitly");
        let a = SkyAtmosphere::default();
        assert!(a.enabled);
        assert_eq!(a.sun_intensity, 3.0);
        // The gradient defaults must equal the renderer's `SkyParams::default()`
        // exactly — that identity is what keeps the sky byte-identical.
        assert_eq!([a.zenith.r, a.zenith.g, a.zenith.b], [0.012, 0.021, 0.038]);
        assert_eq!(
            [a.horizon.r, a.horizon.g, a.horizon.b],
            [0.055, 0.081, 0.120]
        );
        assert_eq!([a.ground.r, a.ground.g, a.ground.b], [0.009, 0.011, 0.015]);
        // Partial payloads fill the same values (the additive contract).
        let partial: SkyAtmosphere = serde_json::from_str("{}").unwrap();
        assert_eq!(partial, a);
        let partial: TimeOfDay = serde_json::from_str(r#"{"rate":60.0}"#).unwrap();
        assert_eq!(partial.rate, 60.0);
        assert_eq!(partial.seconds, 36_000.0);
    }

    // ── schema v12 (P17.2 physical atmosphere) ────────────────────────────

    /// An all-`None` frozen v11 entity — the struct-update base for
    /// [`v11_reference`]. Built through the downgrade hop so the field list can
    /// never drift from the live record.
    fn v11_base(guid: uuid::Uuid, name: &str, parent: Option<uuid::Uuid>) -> EntityRecordV11 {
        EntityRecordV11::from_current(
            v9_base(guid, name, parent)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current(),
        )
    }

    /// The **v11** atmosphere the fixture's sky entity carries: deliberately
    /// **non-default** in two of the nine frozen fields, so the v12 hop is proven
    /// to preserve the v11 half rather than merely to produce defaults.
    fn v11_fixture_atmosphere() -> SkyAtmosphereV11 {
        SkyAtmosphereV11 {
            sun_intensity: 4.25,
            night_darkening: 0.35,
            ..SkyAtmosphereV11::from_current(SkyAtmosphere::default())
        }
    }

    /// Rebuild the exact schema-v11 file the committed v11 fixture was generated
    /// from, out of the frozen v11 record types (the provenance lock). Carries the
    /// **v11** additions (a clock plus a non-default pre-v12 `SkyAtmosphere`) on
    /// top of the v10 world-partition content, so the pre-v12 entity byte layout
    /// is pinned by committed bytes.
    fn v11_reference() -> SceneFileV11 {
        use inf_ecs::components::{
            AlwaysLoaded, Light, LightKind, Material, Primitive, StreamingSource,
        };
        let g = uuid::Uuid::from_u128;
        SceneFileV11 {
            schema_version: 11,
            title: "V11 Fixture Level".into(),
            entities: vec![
                EntityRecordV11 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xB0A1)),
                    }),
                    material: Some(Material::default()),
                    streaming_source: Some(StreamingSource { radius_m: 300.0 }),
                    ..v11_base(g(0xB001), "Player", None)
                },
                EntityRecordV11 {
                    always_loaded: Some(AlwaysLoaded),
                    ..v11_base(g(0xB002), "GameMode", None)
                },
                EntityRecordV11 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    time_of_day: Some(TimeOfDay {
                        seconds: 3_600.0,
                        day_of_year: 355,
                        latitude_deg: -33.9,
                        longitude_deg: 151.2,
                        rate: 120.0,
                    }),
                    sky_atmosphere: Some(v11_fixture_atmosphere()),
                    ..v11_base(g(0xB003), "Sky", None)
                },
            ],
            settings: LevelSettings {
                gravity_2d: Vec2d::new(0.0, -18.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 90.0,
                render: RenderSettingsRecord {
                    exposure: 1.1,
                    ..RenderSettingsRecord::default()
                },
                partition: PartitionSettings {
                    enabled: true,
                    cell_size_m: 128.0,
                    activation_radius_m: 200.0,
                    prefetch_margin_m: 300.0,
                },
            },
        }
    }

    /// Write the committed v11 fixture from [`v11_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline). Never hand-edit
    /// the committed bytes.
    #[test]
    fn bless_v11_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v11_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 11);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v11.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v11 fixture");
        eprintln!("blessed v11 fixture: {}", path.display());
    }

    #[test]
    fn v11_fixture_is_reproducible_and_genuinely_v11() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v11.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v11 fixture present");
        assert_eq!(bytes[0], 11, "fixture must be a genuine schema-v11 payload");
        let rebuilt = bincode::serde::encode_to_vec(v11_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v11 fixture must match our frozen v11 writer"
        );
    }

    /// This crate's committed v11 fixture must be **byte-identical** to the Ring-0
    /// runtime reader's, because the two codecs are one wire contract written
    /// twice. A divergence here is the exact bug the mirror doctrine exists to
    /// catch, and it would otherwise only surface as a player that cannot open a
    /// level the editor just saved.
    #[test]
    fn v11_fixture_matches_the_runtime_codecs_copy() {
        let mine = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v11.inf_lvl");
        let theirs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../crates/inf-scene/tests/fixtures/scene_v11.inf_lvl");
        assert_eq!(
            std::fs::read(&mine).expect("editor v11 fixture"),
            std::fs::read(&theirs).expect("runtime v11 fixture"),
            "the two v12-bump fixtures diverged — the codecs are no longer mirrors"
        );
    }

    /// The committed v11 fixture — written by the **pre-v12 codec**, before
    /// `SkyAtmosphere` grew its physical-atmosphere block — still loads, with the
    /// v11 half preserved verbatim and the 13 new fields at their documented
    /// defaults. The "old bytes load forever" gate for the v12 bump.
    #[test]
    fn v11_loads_and_lifts_the_atmosphere() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v11.inf_lvl");
        let file = decode(&std::fs::read(&path).unwrap()).expect("v11 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V11 Fixture Level");
        assert_eq!(file.entities.len(), 3);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        // The v11 content survives the frozen-record hop intact …
        assert_eq!(by_name("Player").streaming_source.unwrap().radius_m, 300.0);
        assert!(by_name("GameMode").always_loaded.is_some());
        assert_eq!(by_name("Sky").time_of_day.unwrap().rate, 120.0);
        assert_eq!(file.settings.sim_hz, 90.0);
        assert!(file.settings.partition.enabled);

        // … including the **non-default** half of the frozen atmosphere: the hop
        // preserves what v11 authored, it does not merely produce defaults.
        let a = by_name("Sky")
            .sky_atmosphere
            .expect("sky carries an atmosphere");
        assert_eq!(a.sun_intensity, 4.25);
        assert_eq!(a.night_darkening, 0.35);
        assert!(a.enabled);
        assert_eq!(a.zenith, SkyAtmosphere::default().zenith);

        // … and every v12 field lifts to its documented default: a gradient sky
        // with no fog, which is exactly what a v11 level meant.
        let d = SkyAtmosphere::default();
        assert_eq!(a.physical, d.physical);
        assert_eq!(a.sky_intensity, d.sky_intensity);
        assert_eq!(a.turbidity, d.turbidity);
        assert_eq!(a.mie_anisotropy, d.mie_anisotropy);
        assert_eq!(a.sun_disc_deg, d.sun_disc_deg);
        assert_eq!(a.moon_disc_deg, d.moon_disc_deg);
        assert_eq!(a.star_intensity, d.star_intensity);
        assert_eq!(a.aerial_perspective, d.aerial_perspective);
        assert_eq!(a.fog_falloff, d.fog_falloff);
        assert_eq!(a.fog_height, d.fog_height);
        assert_eq!(a.fog_color, d.fog_color);
        assert_eq!(
            a.tint_strength, 0.0,
            "a v11 level is not tinted back toward itself"
        );
        assert_eq!(a.fog_density, 0.0, "a v11 level had no height fog");

        // Load → save → load is byte-identical at the current schema.
        let mut doc = SceneDoc::new();
        apply_to_doc(&mut doc, &file);
        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(bytes1[0], SCHEMA_VERSION as u8);
        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &decode(&bytes1).unwrap());
        assert_eq!(encode(&to_scene_file(&doc2)).unwrap(), bytes1);
    }

    /// The **downgrade-bless** direction for the v11 entity record, as a checked
    /// property rather than a path only `INF_BLESS_FIXTURES=1` walks.
    #[test]
    fn v11_entity_downgrade_is_lossless_except_for_the_physical_atmosphere_block() {
        use inf_ecs::components::{AlwaysLoaded, StreamingSource};
        let g = uuid::Uuid::from_u128;
        let tod = TimeOfDay {
            seconds: 1234.0,
            rate: 60.0,
            ..TimeOfDay::default()
        };
        let authored = SkyAtmosphere {
            // the v11 half — must survive …
            sun_intensity: 4.25,
            night_darkening: 0.35,
            // … and the v12 block — must not.
            physical: false,
            turbidity: 3.5,
            fog_density: 6e-4,
            fog_height: 120.0,
            ..SkyAtmosphere::default()
        };
        let live = EntityRecord {
            streaming_source: Some(StreamingSource { radius_m: 42.0 }),
            always_loaded: Some(AlwaysLoaded),
            time_of_day: Some(tod),
            sky_atmosphere: Some(authored),
            ..v9_base(g(0xC001), "Sky", None)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current()
        };
        let back = EntityRecordV11::from_current(live.clone())
            .into_v12()
            .into_v13()
            .into_current();
        assert_eq!(back.streaming_source, live.streaming_source);
        assert_eq!(back.always_loaded, live.always_loaded);
        assert_eq!(back.time_of_day, Some(tod), "v11 already had the clock");
        assert_eq!(back.name, live.name);

        let a = back.sky_atmosphere.unwrap();
        // The v11 nine survive verbatim …
        assert_eq!(a.sun_intensity, 4.25);
        assert_eq!(a.night_darkening, 0.35);
        // … and the physical-atmosphere block has no v11 home, so it comes back at
        // the live defaults — the one deliberately lossy direction.
        let d = SkyAtmosphere::default();
        assert_eq!(
            a.physical, d.physical,
            "`physical: false` cannot be stored in v11"
        );
        assert_eq!(a.turbidity, d.turbidity);
        assert_eq!(a.fog_density, d.fog_density);
        assert_eq!(a.fog_height, d.fog_height);

        // A record whose atmosphere is entirely default survives the hop exactly,
        // as does one with no atmosphere at all.
        let defaulted = EntityRecord {
            sky_atmosphere: Some(SkyAtmosphere::default()),
            ..v9_base(g(0xC002), "PlainSky", None)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current()
        };
        assert_eq!(
            EntityRecordV11::from_current(defaulted.clone())
                .into_v12()
                .into_v13()
                .into_current(),
            defaulted
        );
        let plain = v9_base(g(0xC003), "Prop", None)
            .into_v10()
            .into_v11()
            .into_v12()
            .into_v13()
            .into_current();
        assert_eq!(
            EntityRecordV11::from_current(plain.clone())
                .into_v12()
                .into_v13()
                .into_current(),
            plain
        );
    }

    /// The v12 physical-atmosphere block persists across save → load and
    /// re-encodes byte-identically — including through the live ECS (`record_of`
    /// reads it, `write_record_components` writes it) — and changing one v12-only
    /// field really moves the bytes. That last assertion is the guard that would
    /// have caught the bump being skipped.
    #[test]
    fn v12_physical_atmosphere_round_trips() {
        let mut doc = SceneDoc::new();
        doc.set_title("Foggy Level");
        let sky = doc.create(SpawnKind::Empty, "Sky", None);
        let atmos = SkyAtmosphere {
            physical: true,
            sky_intensity: 1.4,
            turbidity: 2.5,
            mie_anisotropy: 0.72,
            sun_disc_deg: 1.2,
            moon_disc_deg: 0.6,
            star_intensity: 0.25,
            tint_strength: 0.3,
            aerial_perspective: 1.8,
            fog_density: 6e-4,
            fog_falloff: 0.004,
            fog_height: 120.0,
            fog_color: Color::new(0.8, 0.9, 1.0, 1.0),
            ..SkyAtmosphere::default()
        };
        {
            let e = doc.world().entity_of(sky).unwrap();
            let w = doc.world_mut().world_mut();
            w.entity_mut(e).insert(TimeOfDay::default());
            w.entity_mut(e).insert(atmos);
        }
        doc.create(SpawnKind::Cube, "Prop", None);

        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(bytes1[0], SCHEMA_VERSION as u8);
        let file = decode(&bytes1).unwrap();
        let rec = file.entities.iter().find(|r| r.name == "Sky").unwrap();
        assert_eq!(rec.sky_atmosphere, Some(atmos));

        // Save → load → save is byte-identical (the single-byte standard).
        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &file);
        assert_eq!(encode(&to_scene_file(&doc2)).unwrap(), bytes1);

        // The component really reached the ECS, not just the record.
        let e = doc2.world().entity_of(sky).unwrap();
        assert_eq!(
            doc2.world().world().get::<SkyAtmosphere>(e).copied(),
            Some(atmos)
        );

        // Clearing one v12-only field moves the bytes: the block is persisted, not
        // inferred. If v12 had been skipped, this payload would be v11-shaped and
        // this assertion would fail.
        {
            let e = doc2.world().entity_of(sky).unwrap();
            let w = doc2.world_mut().world_mut();
            w.entity_mut(e).insert(SkyAtmosphere {
                fog_density: 0.0,
                ..atmos
            });
        }
        assert_ne!(encode(&to_scene_file(&doc2)).unwrap(), bytes1);
    }

    /// The `Ring-0 ↔ Ring-1` default lock for the physical-atmosphere block, the
    /// v12 sibling of `sky_component_defaults_are_the_documented_ones`: `inf-scene`
    /// and this codec must agree field-for-field, and the frozen
    /// [`SkyAtmosphereV11`] must reproduce the v11 nine exactly — a frozen record
    /// that drifts is a silently mis-decoded level.
    #[test]
    fn physical_atmosphere_defaults_are_the_documented_ones() {
        let a = SkyAtmosphere::default();
        assert!(a.physical, "a level with a clock wants a real sky");
        assert_eq!(a.sky_intensity, 1.0);
        assert_eq!(a.turbidity, 1.0);
        assert_eq!(a.mie_anisotropy, 0.8);
        assert_eq!(a.sun_disc_deg, 0.545);
        assert_eq!(a.moon_disc_deg, 0.52);
        assert_eq!(a.star_intensity, 1.0);
        assert_eq!(a.tint_strength, 0.0);
        assert_eq!(a.aerial_perspective, 1.0);
        assert_eq!(a.fog_density, 0.0, "no height fog until an author says so");
        assert_eq!(a.fog_falloff, 0.002, "a 500 m e-folding height");
        assert_eq!(a.fog_height, 0.0);
        assert_eq!(a.fog_color, Color::new(1.0, 1.0, 1.0, 1.0));

        // Lifting a default-frozen record yields a default live one.
        let frozen = SkyAtmosphereV11::from_current(a);
        assert_eq!(frozen.into_v12().into_v13().into_current(), a);

        // The frozen record's OWN defaults are pinned to LITERALS, never to the
        // live component's. That is the whole point of freezing: if a future phase
        // re-tunes `SkyAtmosphere::default()`'s v11 half, a v11 payload that
        // omitted a field must still decode to what v11 meant, and comparing the
        // two shapes to each other would let them drift together in silence.
        // (Doctrine: `SkyAtmosphereV11`'s own doc comment — "a frozen record must
        // not be able to move when the live component's defaults are re-tuned".)
        let partial: SkyAtmosphereV11 = serde_json::from_str("{}").unwrap();
        assert!(partial.enabled);
        assert_eq!(partial.sun_intensity, 3.0);
        assert_eq!(partial.sun_color, Color::new(1.0, 0.98, 0.95, 1.0));
        assert_eq!(partial.moon_intensity, 0.15);
        assert_eq!(partial.moon_color, Color::new(0.62, 0.72, 1.0, 1.0));
        assert_eq!(partial.zenith, Color::new(0.012, 0.021, 0.038, 1.0));
        assert_eq!(partial.horizon, Color::new(0.055, 0.081, 0.120, 1.0));
        assert_eq!(partial.ground, Color::new(0.009, 0.011, 0.015, 1.0));
        assert_eq!(partial.night_darkening, 0.85);

        // Today the two happen to agree, which is what makes v12 a pure append —
        // asserted here as a *fact about today*, downstream of the literals above,
        // rather than as the definition of either side.
        assert_eq!(partial, frozen, "v12 appended fields; it moved none");
    }

    // ── schema v13 (P17.3 volumetric clouds) ──────────────────────────────

    /// An all-`None` frozen v12 entity — the struct-update base for
    /// [`v12_reference`]. Built through the downgrade hop so the field list can
    /// never drift from the live record.
    fn v12_base(guid: uuid::Uuid, name: &str, parent: Option<uuid::Uuid>) -> EntityRecordV12 {
        EntityRecordV12::from_current(
            v9_base(guid, name, parent)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current(),
        )
    }

    /// The **v12** atmosphere the fixture's sky entity carries: deliberately
    /// **non-default** in four of the 22 frozen fields (two from the v11 half, two
    /// from the physical block), so the v13 hop is proven to preserve what v12
    /// authored rather than merely to produce defaults.
    fn v12_fixture_atmosphere() -> SkyAtmosphereV12 {
        SkyAtmosphereV12 {
            sun_intensity: 4.25,
            night_darkening: 0.35,
            turbidity: 2.5,
            fog_density: 6e-4,
            ..SkyAtmosphereV12::from_current(SkyAtmosphere::default())
        }
    }

    /// Rebuild the exact schema-v12 file the committed v12 fixture was generated
    /// from, out of the frozen v12 record types (the provenance lock). Carries the
    /// **v12** additions (a clock plus a non-default pre-v13 `SkyAtmosphere`) on
    /// top of the v10 world-partition content, so the pre-v13 entity byte layout
    /// is pinned by committed bytes.
    fn v12_reference() -> SceneFileV12 {
        use inf_ecs::components::{
            AlwaysLoaded, Light, LightKind, Material, Primitive, StreamingSource,
        };
        let g = uuid::Uuid::from_u128;
        SceneFileV12 {
            schema_version: 12,
            title: "V12 Fixture Level".into(),
            entities: vec![
                EntityRecordV12 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xB0A1)),
                    }),
                    material: Some(Material::default()),
                    streaming_source: Some(StreamingSource { radius_m: 300.0 }),
                    ..v12_base(g(0xB001), "Player", None)
                },
                EntityRecordV12 {
                    always_loaded: Some(AlwaysLoaded),
                    ..v12_base(g(0xB002), "GameMode", None)
                },
                EntityRecordV12 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    time_of_day: Some(TimeOfDay {
                        seconds: 3_600.0,
                        day_of_year: 355,
                        latitude_deg: -33.9,
                        longitude_deg: 151.2,
                        rate: 120.0,
                    }),
                    sky_atmosphere: Some(v12_fixture_atmosphere()),
                    ..v12_base(g(0xB003), "Sky", None)
                },
            ],
            settings: LevelSettings {
                gravity_2d: Vec2d::new(0.0, -18.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 90.0,
                render: RenderSettingsRecord {
                    exposure: 1.1,
                    ..RenderSettingsRecord::default()
                },
                partition: PartitionSettings {
                    enabled: true,
                    cell_size_m: 128.0,
                    activation_radius_m: 200.0,
                    prefetch_margin_m: 300.0,
                },
            },
        }
    }

    /// Write the committed v12 fixture from [`v12_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline). Never hand-edit
    /// the committed bytes.
    #[test]
    fn bless_v12_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v12_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 12);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v12.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v12 fixture");
        eprintln!("blessed v12 fixture: {}", path.display());
    }

    #[test]
    fn v12_fixture_is_reproducible_and_genuinely_v12() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v12.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v12 fixture present");
        assert_eq!(bytes[0], 12, "fixture must be a genuine schema-v12 payload");
        let rebuilt = bincode::serde::encode_to_vec(v12_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v12 fixture must match our frozen v12 writer"
        );
    }

    /// This crate's committed v12 fixture must be **byte-identical** to the Ring-0
    /// runtime reader's, because the two codecs are one wire contract written
    /// twice. A divergence here is the exact bug the mirror doctrine exists to
    /// catch, and it would otherwise only surface as a player that cannot open a
    /// level the editor just saved.
    #[test]
    fn v12_fixture_matches_the_runtime_codecs_copy() {
        let mine = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v12.inf_lvl");
        let theirs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../crates/inf-scene/tests/fixtures/scene_v12.inf_lvl");
        assert_eq!(
            std::fs::read(&mine).expect("editor v12 fixture"),
            std::fs::read(&theirs).expect("runtime v12 fixture"),
            "the two v13-bump fixtures diverged — the codecs are no longer mirrors"
        );
    }

    /// The committed v12 fixture — written by the **pre-v13 codec**, before
    /// `SkyAtmosphere` grew its volumetric-cloud block — still loads, with the v12
    /// shape preserved verbatim and the 14 new fields at their documented
    /// defaults. The "old bytes load forever" gate for the v13 bump.
    #[test]
    fn v12_loads_and_lifts_the_clouds() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v12.inf_lvl");
        let file = decode(&std::fs::read(&path).unwrap()).expect("v12 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V12 Fixture Level");
        assert_eq!(file.entities.len(), 3);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        // The v12 content survives the frozen-record hop intact …
        assert_eq!(by_name("Player").streaming_source.unwrap().radius_m, 300.0);
        assert!(by_name("GameMode").always_loaded.is_some());
        assert_eq!(by_name("Sky").time_of_day.unwrap().rate, 120.0);
        assert_eq!(file.settings.sim_hz, 90.0);
        assert!(file.settings.partition.enabled);

        // … including the **non-default** fields of the frozen atmosphere, from
        // both the v11 half and the physical block: the hop preserves what v12
        // authored, it does not merely produce defaults.
        let a = by_name("Sky")
            .sky_atmosphere
            .expect("sky carries an atmosphere");
        assert_eq!(a.sun_intensity, 4.25);
        assert_eq!(a.night_darkening, 0.35);
        assert_eq!(a.turbidity, 2.5);
        assert_eq!(a.fog_density, 6e-4);
        assert!(a.enabled);
        assert!(a.physical);
        assert_eq!(a.zenith, SkyAtmosphere::default().zenith);

        // … and every v13 field lifts to its documented default: no clouds, which
        // is exactly what a v12 level meant.
        let d = SkyAtmosphere::default();
        assert_eq!(a.clouds_enabled, d.clouds_enabled);
        assert_eq!(a.cloud_coverage, d.cloud_coverage);
        assert_eq!(a.cloud_type, d.cloud_type);
        assert_eq!(a.cloud_bottom, d.cloud_bottom);
        assert_eq!(a.cloud_top, d.cloud_top);
        assert_eq!(a.cloud_density, d.cloud_density);
        assert_eq!(a.cloud_detail, d.cloud_detail);
        assert_eq!(a.cloud_seed, d.cloud_seed);
        assert_eq!(a.cloud_wind_x, d.cloud_wind_x);
        assert_eq!(a.cloud_wind_z, d.cloud_wind_z);
        assert_eq!(a.cloud_phase_g, d.cloud_phase_g);
        assert_eq!(a.cloud_shadow, d.cloud_shadow);
        assert_eq!(a.cloud_ambient, d.cloud_ambient);
        assert_eq!(a.cloud_color, d.cloud_color);
        assert!(!a.clouds_enabled, "a v12 level had no clouds");

        // Load → save → load is byte-identical at the current schema.
        let mut doc = SceneDoc::new();
        apply_to_doc(&mut doc, &file);
        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(bytes1[0], SCHEMA_VERSION as u8);
        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &decode(&bytes1).unwrap());
        assert_eq!(encode(&to_scene_file(&doc2)).unwrap(), bytes1);
    }

    /// The **downgrade-bless** direction for the v12 entity record, as a checked
    /// property rather than a path only `INF_BLESS_FIXTURES=1` walks.
    #[test]
    fn v12_entity_downgrade_is_lossless_except_for_the_cloud_block() {
        use inf_ecs::components::{AlwaysLoaded, StreamingSource};
        let g = uuid::Uuid::from_u128;
        let tod = TimeOfDay {
            seconds: 1234.0,
            rate: 60.0,
            ..TimeOfDay::default()
        };
        let authored = SkyAtmosphere {
            // the v12 shape — must survive …
            sun_intensity: 4.25,
            night_darkening: 0.35,
            physical: false,
            turbidity: 3.5,
            fog_density: 6e-4,
            fog_height: 120.0,
            // … and the v13 block — must not.
            clouds_enabled: true,
            cloud_coverage: 0.9,
            cloud_seed: 7,
            cloud_wind_x: 20.0,
            ..SkyAtmosphere::default()
        };
        let live = EntityRecord {
            streaming_source: Some(StreamingSource { radius_m: 42.0 }),
            always_loaded: Some(AlwaysLoaded),
            time_of_day: Some(tod),
            sky_atmosphere: Some(authored),
            ..v9_base(g(0xE001), "Sky", None)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current()
        };
        let back = EntityRecordV12::from_current(live.clone())
            .into_v13()
            .into_current();
        assert_eq!(back.streaming_source, live.streaming_source);
        assert_eq!(back.always_loaded, live.always_loaded);
        assert_eq!(back.time_of_day, Some(tod), "v12 already had the clock");
        assert_eq!(back.name, live.name);

        let a = back.sky_atmosphere.unwrap();
        // The v12 twenty-two survive verbatim …
        assert_eq!(a.sun_intensity, 4.25);
        assert_eq!(a.night_darkening, 0.35);
        assert!(!a.physical, "v12 could store `physical: false`");
        assert_eq!(a.turbidity, 3.5);
        assert_eq!(a.fog_density, 6e-4);
        assert_eq!(a.fog_height, 120.0);
        // … and the volumetric-cloud block has no v12 home, so it comes back at the
        // live defaults — the one deliberately lossy direction.
        let d = SkyAtmosphere::default();
        assert_eq!(
            a.clouds_enabled, d.clouds_enabled,
            "`clouds_enabled: true` cannot be stored in v12"
        );
        assert_eq!(a.cloud_coverage, d.cloud_coverage);
        assert_eq!(a.cloud_seed, d.cloud_seed);
        assert_eq!(a.cloud_wind_x, d.cloud_wind_x);

        // A record whose atmosphere is entirely default survives the hop exactly,
        // as does one with no atmosphere at all.
        let defaulted = EntityRecord {
            sky_atmosphere: Some(SkyAtmosphere::default()),
            ..v9_base(g(0xE002), "PlainSky", None)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current()
        };
        assert_eq!(
            EntityRecordV12::from_current(defaulted.clone())
                .into_v13()
                .into_current(),
            defaulted
        );
        let plain = v9_base(g(0xE003), "Prop", None)
            .into_v10()
            .into_v11()
            .into_v12()
            .into_v13()
            .into_current();
        assert_eq!(
            EntityRecordV12::from_current(plain.clone())
                .into_v13()
                .into_current(),
            plain
        );
    }

    /// The v13 volumetric-cloud block persists across save → load and re-encodes
    /// byte-identically — including through the live ECS (`record_of` reads it,
    /// `write_record_components` writes it) — and changing one v13-only field
    /// really moves the bytes. That last assertion is the guard that would have
    /// caught the bump being skipped.
    #[test]
    fn v13_clouds_round_trip() {
        let mut doc = SceneDoc::new();
        doc.set_title("Cloudy Level");
        let sky = doc.create(SpawnKind::Empty, "Sky", None);
        let atmos = SkyAtmosphere {
            clouds_enabled: true,
            cloud_coverage: 0.62,
            cloud_type: 0.4,
            cloud_bottom: 900.0,
            cloud_top: 5200.0,
            cloud_density: 0.07,
            cloud_detail: 0.85,
            cloud_seed: 90_210,
            cloud_wind_x: -11.5,
            cloud_wind_z: 3.25,
            cloud_phase_g: 0.55,
            cloud_shadow: 0.4,
            cloud_ambient: 1.75,
            cloud_color: Color::new(0.94, 0.96, 1.0, 1.0),
            ..SkyAtmosphere::default()
        };
        {
            let e = doc.world().entity_of(sky).unwrap();
            let w = doc.world_mut().world_mut();
            w.entity_mut(e).insert(TimeOfDay::default());
            w.entity_mut(e).insert(atmos);
        }
        doc.create(SpawnKind::Cube, "Prop", None);

        let bytes1 = encode(&to_scene_file(&doc)).unwrap();
        assert_eq!(bytes1[0], SCHEMA_VERSION as u8);
        let file = decode(&bytes1).unwrap();
        let rec = file.entities.iter().find(|r| r.name == "Sky").unwrap();
        assert_eq!(rec.sky_atmosphere, Some(atmos));

        // Save → load → save is byte-identical (the single-byte standard).
        let mut doc2 = SceneDoc::new();
        apply_to_doc(&mut doc2, &file);
        assert_eq!(encode(&to_scene_file(&doc2)).unwrap(), bytes1);

        // The component really reached the ECS, not just the record.
        let e = doc2.world().entity_of(sky).unwrap();
        assert_eq!(
            doc2.world().world().get::<SkyAtmosphere>(e).copied(),
            Some(atmos)
        );

        // Clearing one v13-only field moves the bytes: the block is persisted, not
        // inferred. If v13 had been skipped, this payload would be v12-shaped and
        // this assertion would fail.
        {
            let e = doc2.world().entity_of(sky).unwrap();
            let w = doc2.world_mut().world_mut();
            w.entity_mut(e).insert(SkyAtmosphere {
                clouds_enabled: false,
                ..atmos
            });
        }
        assert_ne!(encode(&to_scene_file(&doc2)).unwrap(), bytes1);
    }

    /// The `Ring-0 ↔ Ring-1` default lock for the volumetric-cloud block, the v13
    /// sibling of `physical_atmosphere_defaults_are_the_documented_ones`:
    /// `inf-scene` and this codec must agree field-for-field, and the frozen
    /// [`SkyAtmosphereV12`] must reproduce the v12 twenty-two exactly — a frozen
    /// record that drifts is a silently mis-decoded level.
    #[test]
    fn cloud_defaults_are_the_documented_ones() {
        let a = SkyAtmosphere::default();
        assert!(
            !a.clouds_enabled,
            "clouds cost frames; existing content keeps the sky it was authored against"
        );
        assert_eq!(a.cloud_coverage, 0.35, "broken cumulus with real gaps");
        assert_eq!(a.cloud_type, 0.7);
        assert_eq!(a.cloud_bottom, 1500.0, "metres (SI)");
        assert_eq!(a.cloud_top, 4000.0, "metres (SI)");
        assert_eq!(a.cloud_density, 0.04, "m^-1");
        assert_eq!(a.cloud_detail, 0.6);
        assert_eq!(a.cloud_seed, 0);
        assert_eq!(a.cloud_wind_x, 6.0, "m/s");
        assert_eq!(a.cloud_wind_z, 2.0, "m/s");
        assert_eq!(a.cloud_phase_g, 0.8);
        assert_eq!(a.cloud_shadow, 1.0, "the physical amount");
        assert_eq!(a.cloud_ambient, 1.0, "the physical amount");
        assert_eq!(a.cloud_color, Color::new(1.0, 1.0, 1.0, 1.0));

        // Lifting a default-frozen record yields a default live one.
        let frozen = SkyAtmosphereV12::from_current(a);
        assert_eq!(frozen.into_v13().into_current(), a);

        // The frozen record's OWN defaults are pinned to LITERALS, never to the
        // live component's — the same doctrine as `SkyAtmosphereV11`: if a future
        // phase re-tunes `SkyAtmosphere::default()`'s v12 half, a v12 payload that
        // omitted a field must still decode to what v12 meant.
        let partial: SkyAtmosphereV12 = serde_json::from_str("{}").unwrap();
        assert!(partial.enabled);
        assert_eq!(partial.sun_intensity, 3.0);
        assert_eq!(partial.sun_color, Color::new(1.0, 0.98, 0.95, 1.0));
        assert_eq!(partial.moon_intensity, 0.15);
        assert_eq!(partial.moon_color, Color::new(0.62, 0.72, 1.0, 1.0));
        assert_eq!(partial.zenith, Color::new(0.012, 0.021, 0.038, 1.0));
        assert_eq!(partial.horizon, Color::new(0.055, 0.081, 0.120, 1.0));
        assert_eq!(partial.ground, Color::new(0.009, 0.011, 0.015, 1.0));
        assert_eq!(partial.night_darkening, 0.85);
        assert!(partial.physical);
        assert_eq!(partial.sky_intensity, 1.0);
        assert_eq!(partial.turbidity, 1.0);
        assert_eq!(partial.mie_anisotropy, 0.8);
        assert_eq!(partial.sun_disc_deg, 0.545);
        assert_eq!(partial.moon_disc_deg, 0.52);
        assert_eq!(partial.star_intensity, 1.0);
        assert_eq!(partial.tint_strength, 0.0);
        assert_eq!(partial.aerial_perspective, 1.0);
        assert_eq!(partial.fog_density, 0.0);
        assert_eq!(partial.fog_falloff, 0.002);
        assert_eq!(partial.fog_height, 0.0);
        assert_eq!(partial.fog_color, Color::new(1.0, 1.0, 1.0, 1.0));

        // Today the two happen to agree, which is what makes v13 a pure append —
        // asserted here as a *fact about today*, downstream of the literals above,
        // rather than as the definition of either side. It is also what keeps
        // `SkyAtmosphereV11::into_v12` (which fills the physical block from this
        // ladder's own literals) byte-identical to the runtime codec's v11 lift
        // (which fills it from `SkyAtmosphere::default()`).
        assert_eq!(partial, frozen, "v13 appended fields; it moved none");
    }

    // ── schema v14 (P17.4 weather states) ─────────────────────────────────

    /// An all-`None` frozen v13 entity — the struct-update base for
    /// [`v13_reference`]. Built through the downgrade hop so the field list can
    /// never drift from the live record.
    fn v13_base(guid: uuid::Uuid, name: &str, parent: Option<uuid::Uuid>) -> EntityRecordV13 {
        EntityRecordV13::from_current(
            v9_base(guid, name, parent)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current(),
        )
    }

    /// The **v13** atmosphere the fixture's sky entity carries: deliberately
    /// **non-default** in fields drawn from all three earlier blocks (the v11
    /// half, the physical block and the cloud block), so the v14 hop is proven to
    /// preserve what v13 authored rather than merely to produce defaults.
    ///
    /// The literals must match `inf-scene`'s `v13_fixture_atmosphere` exactly —
    /// the two committed fixtures are byte-compared by
    /// [`v13_fixture_matches_the_runtime_codecs_copy`], which is the whole point
    /// of writing them twice.
    fn v13_fixture_atmosphere() -> SkyAtmosphereV13 {
        SkyAtmosphereV13 {
            sun_intensity: 4.25,
            night_darkening: 0.35,
            turbidity: 2.5,
            fog_density: 6e-4,
            clouds_enabled: true,
            cloud_coverage: 0.62,
            cloud_type: 0.4,
            cloud_bottom: 900.0,
            cloud_top: 5200.0,
            cloud_density: 0.07,
            cloud_detail: 0.85,
            cloud_seed: 90_210,
            cloud_wind_x: -11.5,
            cloud_wind_z: 3.25,
            cloud_phase_g: 0.55,
            cloud_shadow: 0.4,
            cloud_ambient: 1.75,
            cloud_color: Color::new(0.94, 0.96, 1.0, 1.0),
            ..SkyAtmosphereV13::from_current(SkyAtmosphere::default())
        }
    }

    /// Rebuild the exact schema-v13 file the committed v13 fixture was generated
    /// from, out of the frozen v13 record types (the provenance lock).
    fn v13_reference() -> SceneFileV13 {
        use inf_ecs::components::{
            AlwaysLoaded, Light, LightKind, Material, Primitive, StreamingSource,
        };
        let g = uuid::Uuid::from_u128;
        SceneFileV13 {
            schema_version: 13,
            title: "V13 Fixture Level".into(),
            entities: vec![
                EntityRecordV13 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xC0A1)),
                    }),
                    material: Some(Material::default()),
                    streaming_source: Some(StreamingSource { radius_m: 300.0 }),
                    ..v13_base(g(0xC001), "Player", None)
                },
                EntityRecordV13 {
                    always_loaded: Some(AlwaysLoaded),
                    ..v13_base(g(0xC002), "GameMode", None)
                },
                EntityRecordV13 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    time_of_day: Some(TimeOfDay {
                        seconds: 3_600.0,
                        day_of_year: 355,
                        latitude_deg: -33.9,
                        longitude_deg: 151.2,
                        rate: 120.0,
                    }),
                    sky_atmosphere: Some(v13_fixture_atmosphere()),
                    ..v13_base(g(0xC003), "Sky", None)
                },
            ],
            settings: LevelSettings {
                gravity_2d: Vec2d::new(0.0, -18.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 90.0,
                render: RenderSettingsRecord {
                    exposure: 1.1,
                    ..RenderSettingsRecord::default()
                },
                partition: PartitionSettings {
                    enabled: true,
                    cell_size_m: 128.0,
                    activation_radius_m: 200.0,
                    prefetch_margin_m: 300.0,
                },
            },
        }
    }

    /// Write the committed v13 fixture from [`v13_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline). Never hand-edit
    /// the committed bytes.
    #[test]
    fn bless_v13_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v13_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 13);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v13.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v13 fixture");
        eprintln!("blessed v13 fixture: {}", path.display());
    }

    #[test]
    fn v13_fixture_is_reproducible_and_genuinely_v13() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v13.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v13 fixture present");
        assert_eq!(bytes[0], 13, "fixture must be a genuine schema-v13 payload");
        let rebuilt = bincode::serde::encode_to_vec(v13_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v13 fixture must match our frozen v13 writer"
        );
    }

    /// This crate's committed v13 fixture must be **byte-identical** to the Ring-0
    /// runtime reader's, because the two codecs are one wire contract written
    /// twice. A divergence here is the exact bug the mirror doctrine exists to
    /// catch, and it would otherwise only surface as a player that cannot open a
    /// level the editor just saved.
    #[test]
    fn v13_fixture_matches_the_runtime_codecs_copy() {
        let mine = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v13.inf_lvl");
        let theirs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../crates/inf-scene/tests/fixtures/scene_v13.inf_lvl");
        assert_eq!(
            std::fs::read(&mine).expect("editor v13 fixture"),
            std::fs::read(&theirs).expect("runtime v13 fixture"),
            "the two v14-bump fixtures diverged — the codecs are no longer mirrors"
        );
    }

    /// The committed v13 fixture — written by the **pre-v14 codec**, before
    /// `SkyAtmosphere` grew its weather block — still loads, with the v13 shape
    /// preserved verbatim and the 11 new fields at their documented defaults. The
    /// "old bytes load forever" gate for the v14 bump.
    #[test]
    fn v13_loads_and_lifts_the_weather() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v13.inf_lvl");
        let file = decode(&std::fs::read(&path).unwrap()).expect("v13 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V13 Fixture Level");
        assert_eq!(file.entities.len(), 3);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        // The v13 content survives the frozen-record hop intact …
        assert_eq!(by_name("Player").streaming_source.unwrap().radius_m, 300.0);
        assert!(by_name("GameMode").always_loaded.is_some());
        assert_eq!(by_name("Sky").time_of_day.unwrap().rate, 120.0);
        assert_eq!(file.settings.sim_hz, 90.0);

        let a = by_name("Sky")
            .sky_atmosphere
            .expect("sky has an atmosphere");
        assert_eq!(a.sun_intensity, 4.25);
        assert_eq!(a.turbidity, 2.5);
        assert!(a.clouds_enabled, "v13 could author clouds");
        assert_eq!(a.cloud_seed, 90_210);

        // … and every v14 field lifts to its documented default: weather OFF, so
        // the authored cloud and fog fields above are still what drives the sky.
        let d = SkyAtmosphere::default();
        assert!(!a.weather_enabled, "a v13 level had no weather block");
        assert_eq!(a.weather_target, d.weather_target);
        assert_eq!(a.weather_blend_seconds, d.weather_blend_seconds);
        assert_eq!(a.weather_blend_remaining, 0.0);
        assert_eq!(a.weather_coverage, d.weather_coverage);
        assert_eq!(a.weather_precipitation, 0.0);
        assert_eq!(a.weather_snowiness, 0.0);
    }

    /// v14 sibling of `cloud_defaults_are_the_documented_ones`: `inf-scene` and
    /// this codec must agree field-for-field about what a v13 payload means, and
    /// the frozen record's own defaults must be **literals**.
    #[test]
    fn weather_defaults_are_the_documented_ones() {
        use inf_ecs::components::WeatherPreset;
        let a = SkyAtmosphere::default();
        assert!(!a.weather_enabled);
        assert_eq!(a.weather_target, WeatherPreset::Clear);
        assert_eq!(a.weather_blend_seconds, 8.0);
        assert_eq!(a.weather_blend_remaining, 0.0);
        assert_eq!(a.weather_params(), WeatherPreset::Clear.params());

        // A default-frozen record lifts to a default live one.
        let frozen = SkyAtmosphereV13::from_current(a);
        assert_eq!(frozen.into_current(), a);

        // The frozen record's OWN defaults are pinned to LITERALS: a v13 payload
        // that omitted a field must still decode to what v13 meant, however the
        // live component is re-tuned later. Reached the way this crate always
        // reaches it — through a self-describing codec, which the Ring-0 twin
        // cannot use and therefore asserts against its `v13_*` fns instead.
        let partial: SkyAtmosphereV13 = serde_json::from_str("{}").unwrap();
        assert!(partial.enabled);
        assert!(partial.physical);
        assert_eq!(partial.sun_intensity, 3.0);
        assert_eq!(partial.night_darkening, 0.85);
        assert_eq!(partial.fog_falloff, 0.002);
        assert!(!partial.clouds_enabled);
        assert_eq!(partial.cloud_coverage, 0.35);
        assert_eq!(partial.cloud_type, 0.7);
        assert_eq!(partial.cloud_bottom, 1500.0);
        assert_eq!(partial.cloud_top, 4000.0);
        assert_eq!(partial.cloud_density, 0.04);
        assert_eq!(partial.cloud_detail, 0.6);
        assert_eq!(partial.cloud_seed, 0);
        assert_eq!(partial.cloud_wind_x, 6.0);
        assert_eq!(partial.cloud_wind_z, 2.0);
        assert_eq!(partial.cloud_phase_g, 0.8);
        assert_eq!(partial.cloud_shadow, 1.0);
        assert_eq!(partial.cloud_ambient, 1.0);
        assert_eq!(partial.cloud_color, Color::new(1.0, 1.0, 1.0, 1.0));

        // Today the two happen to agree, which is what makes v14 a pure append —
        // a fact about today, downstream of the literals above, and what keeps
        // `SkyAtmosphereV12::into_v13` (which fills the cloud block from this
        // ladder's own literals) byte-identical to the runtime codec's v12 lift
        // (which fills it from `SkyAtmosphere::default()`).
        assert_eq!(partial, frozen, "v14 appended fields; it moved none");
    }

    /// The v14 additions round-trip through the whole editor codec, and the
    /// weather block really moves the bytes.
    #[test]
    fn v14_weather_round_trips_through_the_codec() {
        use inf_ecs::components::WeatherPreset;
        let g = uuid::Uuid::from_u128;
        let atmos = SkyAtmosphere {
            weather_enabled: true,
            weather_target: WeatherPreset::Storm,
            weather_blend_seconds: 12.5,
            weather_blend_remaining: 3.25,
            weather_coverage: 0.77,
            weather_cloud_type: 0.31,
            weather_wind_x: -14.5,
            weather_wind_z: 6.25,
            weather_fog_density: 9e-4,
            weather_precipitation: 0.65,
            weather_snowiness: 0.5,
            ..SkyAtmosphere::default()
        };
        let mut file = SceneFile {
            schema_version: SCHEMA_VERSION,
            title: "V14 Weather".into(),
            entities: vec![EntityRecord {
                time_of_day: Some(TimeOfDay::default()),
                sky_atmosphere: Some(atmos),
                ..v9_base(g(0xA201), "Sky", None)
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current()
            }],
            settings: LevelSettings::default(),
        };
        let bytes = bincode::serde::encode_to_vec(&file, bincode_config()).unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = decode(&bytes).expect("v14 decodes");
        assert_eq!(back.entities[0].sky_atmosphere, Some(atmos));

        // The enum really crosses the wire: swapping the preset moves the bytes
        // without moving the length (both variant indices are one varint byte).
        file.entities[0].sky_atmosphere = Some(SkyAtmosphere {
            weather_target: WeatherPreset::Fog,
            ..atmos
        });
        let other = bincode::serde::encode_to_vec(&file, bincode_config()).unwrap();
        assert_ne!(other, bytes);
        assert_eq!(other.len(), bytes.len());
    }

    // ── schema v15 (P19.1 erosion data maps) ──────────────────────────────

    /// An all-`None` frozen v14 entity — the struct-update base for
    /// [`v14_reference`]. Built through the downgrade hop so the field list can
    /// never drift from the live record.
    fn v14_base(guid: uuid::Uuid, name: &str, parent: Option<uuid::Uuid>) -> EntityRecordV14 {
        EntityRecordV14::from_current(
            v9_base(guid, name, parent)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current(),
        )
    }

    /// The **v14** terrain the fixture carries: two authored tiles, one painted
    /// splat sample, a non-default macro variation and an asset reference — so
    /// the v15 hop is proven to preserve what v14 authored, not merely to
    /// produce defaults. Its tiles have **no** data maps, because v14 could not
    /// express them; that is exactly what the lift has to reproduce.
    ///
    /// The literals must match `inf-scene`'s `v14_fixture_terrain` exactly — the
    /// two committed fixtures are byte-compared by
    /// [`v14_fixture_matches_the_runtime_codecs_copy`], which is the whole point
    /// of writing them twice.
    fn v14_fixture_terrain() -> TerrainV14 {
        let mut t = Terrain::configured(4, 2.0);
        let f = |x: f64, z: f64| x * 0.5 - z * 0.25 + 3.0;
        t.data.author_tile((0, 0), f);
        t.data.author_tile((1, 0), f);
        t.data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_weight_sample(4, 1, 2, [40, 100, 80, 35]);
        t.macro_variation = 0.25;
        t.asset = Some(uuid::Uuid::from_u128(0xD_00AA));
        TerrainV14::from_current(t)
    }

    /// Rebuild the exact schema-v14 file the committed v14 fixture was generated
    /// from, out of the frozen v14 record types (the provenance lock).
    fn v14_reference() -> SceneFileV14 {
        use inf_ecs::components::{Light, LightKind, Material, MeshRef, Primitive};
        let g = uuid::Uuid::from_u128;
        SceneFileV14 {
            schema_version: 14,
            title: "V14 Fixture Level".into(),
            entities: vec![
                EntityRecordV14 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xD0A1)),
                    }),
                    material: Some(Material::default()),
                    ..v14_base(g(0xD001), "Cube", None)
                },
                EntityRecordV14 {
                    terrain: Some(v14_fixture_terrain()),
                    ..v14_base(g(0xD002), "Terrain", None)
                },
                EntityRecordV14 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v14_base(g(0xD003), "Sun", None)
                },
            ],
            settings: LevelSettings {
                gravity_2d: Vec2d::new(0.0, -18.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 90.0,
                render: RenderSettingsRecord {
                    exposure: 1.1,
                    ..RenderSettingsRecord::default()
                },
                partition: PartitionSettings::default(),
            },
        }
    }

    /// Write the committed v14 fixture from [`v14_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline). Never hand-edit
    /// the committed bytes.
    #[test]
    fn bless_v14_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v14_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 14);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v14.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v14 fixture");
        eprintln!("blessed v14 fixture: {}", path.display());
    }

    #[test]
    fn v14_fixture_is_reproducible_and_genuinely_v14() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v14.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v14 fixture present");
        assert_eq!(bytes[0], 14, "fixture must be a genuine schema-v14 payload");
        let rebuilt = bincode::serde::encode_to_vec(v14_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v14 fixture must match our frozen v14 writer"
        );
    }

    /// This crate's committed v14 fixture must be **byte-identical** to the Ring-0
    /// runtime reader's — the two codecs are one wire contract written twice.
    #[test]
    fn v14_fixture_matches_the_runtime_codecs_copy() {
        let mine = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v14.inf_lvl");
        let theirs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../crates/inf-scene/tests/fixtures/scene_v14.inf_lvl");
        assert_eq!(
            std::fs::read(&mine).expect("editor v14 fixture"),
            std::fs::read(&theirs).expect("runtime v14 fixture"),
            "the two v15-bump fixtures diverged — the codecs are no longer mirrors"
        );
    }

    /// The committed v14 fixture — written by the **pre-v15 codec**, before every
    /// terrain tile grew its erosion data-map layer — still loads, with the v14
    /// content preserved verbatim and every tile's maps at the never-eroded
    /// default. The "old bytes load forever" gate for the v15 bump.
    #[test]
    fn v14_loads_and_lifts_the_data_maps() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v14.inf_lvl");
        let file = decode(&std::fs::read(&path).unwrap()).expect("v14 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V14 Fixture Level");
        assert_eq!(file.entities.len(), 3);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        // The v14 content survives the frozen-record hop intact …
        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(uuid::Uuid::from_u128(0xD0A1))
        );
        assert_eq!(by_name("Sun").light.unwrap().intensity, 2.0);
        assert_eq!(file.settings.sim_hz, 90.0);

        let t = by_name("Terrain").terrain.clone().expect("terrain slot");
        assert_eq!(t.tile_resolution, 4);
        assert_eq!(t.meters_per_sample, 2.0);
        assert_eq!(t.macro_variation, 0.25);
        assert_eq!(t.asset, Some(uuid::Uuid::from_u128(0xD_00AA)));
        assert_eq!(t.data.tile_count(), 2);
        assert_eq!(
            t.data.get_tile((0, 0)).unwrap().weight_sample(4, 1, 2),
            [40, 100, 80, 35],
            "v14 could author splat weights, and they must survive"
        );

        // … and every tile's data maps lift to the documented default: **empty**,
        // i.e. never eroded, which is exactly what a v14 level meant.
        assert!(
            t.data.data_maps_are_default(),
            "a v14 level had no data maps; the lift must not conjure any"
        );
        for (coord, tile) in t.data.tiles() {
            assert_eq!(tile.maps_len(), 0, "tile {coord:?} conjured a maps buffer");
            for kind in inf_terrain::DataMapKind::ALL {
                assert_eq!(tile.map_sample(4, kind, 1, 2), 0.0);
            }
        }
    }

    /// The v14 downgrade is lossless **except** for the erosion data maps — the
    /// one thing v14 cannot express. Proven as a property (round-trip a live
    /// record through the frozen shape) rather than by listing fields, so a slot
    /// added later cannot silently fall out of the ladder.
    #[test]
    fn v14_entity_downgrade_is_lossless_except_for_the_data_maps() {
        let mut terrain = Terrain::configured(4, 2.0);
        terrain.data.author_tile((0, 0), |x, z| x + z);
        terrain
            .data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_map_texel(4, 1, 1, [5.0, 2.0, 1.0]);
        assert!(!terrain.data.data_maps_are_default());

        let live = EntityRecord {
            terrain: Some(terrain),
            ..v9_base(uuid::Uuid::from_u128(0xD100), "T", None)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current()
        };
        let back = EntityRecordV14::from_current(live.clone())
            .into_v15()
            .into_v16()
            .into_v17()
            .into_v18()
            .into_current();

        // Everything but the maps survives …
        let t = back.terrain.clone().unwrap();
        assert_eq!(t.tile_resolution, 4);
        assert_eq!(t.data.tile_count(), 1);
        assert_eq!(
            t.data.get_tile((0, 0)).unwrap().heights(),
            live.terrain
                .as_ref()
                .unwrap()
                .data
                .get_tile((0, 0))
                .unwrap()
                .heights()
        );
        // … and the maps are exactly what is lost.
        assert!(t.data.data_maps_are_default());
        assert_eq!(
            EntityRecord {
                terrain: Some(t),
                ..back.clone()
            },
            back,
            "nothing outside the terrain moved"
        );
    }

    /// The v15 addition round-trips through the whole editor codec: a terrain
    /// carrying data maps saves and reloads byte-identically, at whatever the
    /// current schema is (the maps are a *live* feature and must keep working
    /// across later bumps — that is the point of re-running this after v16).
    #[test]
    fn v15_data_maps_round_trip_through_the_codec() {
        let mut terrain = Terrain::configured(4, 2.0);
        terrain.data.author_tile((0, 0), |x, z| x - z);
        terrain
            .data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_map_texel(4, 2, 1, [12.5, 0.25, 3.75]);

        let file = SceneFile {
            schema_version: SCHEMA_VERSION,
            title: "Eroded".into(),
            entities: vec![EntityRecord {
                terrain: Some(terrain.clone()),
                ..v9_base(uuid::Uuid::from_u128(0xD200), "T", None)
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current()
            }],
            settings: LevelSettings::default(),
        };
        let bytes = bincode::serde::encode_to_vec(&file, bincode_config()).unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = decode(&bytes).expect("the current schema decodes");
        let t = back.entities[0].terrain.clone().unwrap();
        assert_eq!(
            t.data.get_tile((0, 0)).unwrap().map_texel(4, 2, 1),
            [12.5, 0.25, 3.75]
        );
        // Re-encoding is byte-identical — the maps are as byte-stable as heights.
        assert_eq!(
            bincode::serde::encode_to_vec(&back, bincode_config()).unwrap(),
            bytes
        );

        // An un-eroded terrain costs exactly one length byte per tile more than
        // the same terrain would have at v14 — the sparse claim, in bytes.
        //
        // Priced **frozen against frozen** (v14 vs v15) rather than against the
        // live record. The live shape is a moving target — v16 grew it again —
        // and a v15-vs-live measurement would silently start pricing every later
        // addition too. The frozen pair is what v15 actually was.
        let mut plain = file.clone();
        plain.entities[0].terrain = Some({
            let mut t = terrain.clone();
            t.data.get_tile_mut((0, 0)).unwrap().clear_maps();
            t
        });
        let plain_bytes = bincode::serde::encode_to_vec(&plain, bincode_config()).unwrap();
        let v14_bytes = bincode::serde::encode_to_vec(
            EntityRecordV14::from_current(plain.entities[0].clone()),
            bincode_config(),
        )
        .unwrap();
        let v15_entity = bincode::serde::encode_to_vec(
            EntityRecordV15::from_current(plain.entities[0].clone()),
            bincode_config(),
        )
        .unwrap();
        assert_eq!(
            v15_entity.len(),
            v14_bytes.len() + 1,
            "an un-eroded 1-tile terrain must cost exactly one extra byte at v15"
        );
        assert!(
            plain_bytes.len() < bytes.len(),
            "the dense buffer really costs"
        );
    }

    // ── schema v16 (P19.2 biome ids) ──────────────────────────────────────

    /// An all-`None` frozen v15 entity — the struct-update base for
    /// [`v15_reference`]. Built through the downgrade hop so the field list can
    /// never drift from the live record.
    fn v15_base(guid: uuid::Uuid, name: &str, parent: Option<uuid::Uuid>) -> EntityRecordV15 {
        EntityRecordV15::from_current(
            v9_base(guid, name, parent)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current(),
        )
    }

    /// The **v15** terrain the fixture carries: two authored tiles, one painted
    /// splat sample, a **materialized erosion data map** (the one thing v15 could
    /// express that v14 could not), a non-default macro variation and an asset
    /// reference — so the v16 hop is proven to preserve what v15 authored, not
    /// merely to produce defaults. Its tiles have **no** biome ids and it has no
    /// `biome_set`, because v15 could express neither; that is exactly what the
    /// lift has to reproduce.
    ///
    /// The literals must match `inf-scene`'s `v15_fixture_terrain` exactly — the
    /// two committed fixtures are byte-compared by
    /// [`v15_fixture_matches_the_runtime_codecs_copy`], which is the whole point
    /// of writing them twice.
    fn v15_fixture_terrain() -> TerrainV15 {
        let mut t = Terrain::configured(4, 2.0);
        let f = |x: f64, z: f64| x * 0.5 - z * 0.25 + 3.0;
        t.data.author_tile((0, 0), f);
        t.data.author_tile((1, 0), f);
        t.data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_weight_sample(4, 1, 2, [40, 100, 80, 35]);
        t.data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_map_texel(4, 1, 1, [7.5, 0.5, 2.25]);
        t.macro_variation = 0.25;
        t.asset = Some(uuid::Uuid::from_u128(0xE_00AA));
        TerrainV15::from_current(t)
    }

    /// Rebuild the exact schema-v15 file the committed v15 fixture was generated
    /// from, out of the frozen v15 record types (the provenance lock).
    fn v15_reference() -> SceneFileV15 {
        use inf_ecs::components::{Light, LightKind, Material, MeshRef, Primitive};
        let g = uuid::Uuid::from_u128;
        SceneFileV15 {
            schema_version: 15,
            title: "V15 Fixture Level".into(),
            entities: vec![
                EntityRecordV15 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xE0A1)),
                    }),
                    material: Some(Material::default()),
                    ..v15_base(g(0xE001), "Cube", None)
                },
                EntityRecordV15 {
                    terrain: Some(v15_fixture_terrain()),
                    ..v15_base(g(0xE002), "Terrain", None)
                },
                EntityRecordV15 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v15_base(g(0xE003), "Sun", None)
                },
            ],
            settings: LevelSettings {
                gravity_2d: Vec2d::new(0.0, -18.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 90.0,
                render: RenderSettingsRecord {
                    exposure: 1.1,
                    ..RenderSettingsRecord::default()
                },
                partition: PartitionSettings::default(),
            },
        }
    }

    /// Write the committed v15 fixture from [`v15_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline). Never hand-edit
    /// the committed bytes.
    #[test]
    fn bless_v15_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v15_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 15);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v15.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v15 fixture");
        eprintln!("blessed v15 fixture: {}", path.display());
    }

    #[test]
    fn v15_fixture_is_reproducible_and_genuinely_v15() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v15.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v15 fixture present");
        assert_eq!(bytes[0], 15, "fixture must be a genuine schema-v15 payload");
        let rebuilt = bincode::serde::encode_to_vec(v15_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v15 fixture must match our frozen v15 writer"
        );
    }

    /// This crate's committed v15 fixture must be **byte-identical** to the Ring-0
    /// runtime reader's — the two codecs are one wire contract written twice.
    #[test]
    fn v15_fixture_matches_the_runtime_codecs_copy() {
        let mine = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v15.inf_lvl");
        let theirs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../crates/inf-scene/tests/fixtures/scene_v15.inf_lvl");
        assert_eq!(
            std::fs::read(&mine).expect("editor v15 fixture"),
            std::fs::read(&theirs).expect("runtime v15 fixture"),
            "the two v16-bump fixtures diverged — the codecs are no longer mirrors"
        );
    }

    /// The committed v15 fixture — written by the **pre-v16 codec**, before every
    /// terrain tile grew its per-sample biome-id layer — still loads, with the v15
    /// content (erosion data maps included) preserved verbatim, every tile's biome
    /// ids at the unpainted default, and no biome vocabulary. The "old bytes load
    /// forever" gate for the v16 bump.
    #[test]
    fn v15_loads_and_lifts_the_biomes() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v15.inf_lvl");
        let file = decode(&std::fs::read(&path).unwrap()).expect("v15 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V15 Fixture Level");
        assert_eq!(file.entities.len(), 3);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        // The v15 content survives the frozen-record hop intact …
        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(uuid::Uuid::from_u128(0xE0A1))
        );
        assert_eq!(by_name("Sun").light.unwrap().intensity, 2.0);
        assert_eq!(file.settings.sim_hz, 90.0);

        let t = by_name("Terrain").terrain.clone().expect("terrain slot");
        assert_eq!(t.tile_resolution, 4);
        assert_eq!(t.meters_per_sample, 2.0);
        assert_eq!(t.macro_variation, 0.25);
        assert_eq!(t.asset, Some(uuid::Uuid::from_u128(0xE_00AA)));
        assert_eq!(t.data.tile_count(), 2);
        assert_eq!(
            t.data.get_tile((0, 0)).unwrap().weight_sample(4, 1, 2),
            [40, 100, 80, 35],
            "v15 could author splat weights, and they must survive"
        );
        // … the erosion maps above all: v15 is the generation that could express
        // them, so their survival is what proves this is a v15→v16 hop and not a
        // v14 one wearing a new number.
        assert!(!t.data.data_maps_are_default());
        assert_eq!(
            t.data.get_tile((0, 0)).unwrap().map_texel(4, 1, 1),
            [7.5, 0.5, 2.25],
            "v15 could author erosion data maps, and they must survive"
        );

        // … and the two P19.2 additions lift to their documented defaults:
        // nothing painted, and no vocabulary to have painted from.
        assert!(
            t.data.biomes_are_default(),
            "a v15 level had no biome ids; the lift must not conjure any"
        );
        for (coord, tile) in t.data.tiles() {
            assert_eq!(
                tile.biomes_len(),
                0,
                "tile {coord:?} conjured a biome buffer"
            );
            assert_eq!(
                tile.biome_sample(4, 1, 2),
                inf_terrain::UNASSIGNED_BIOME,
                "tile {coord:?} must read as unassigned everywhere"
            );
        }
        assert_eq!(
            t.biome_set, None,
            "a v15 payload's bytes end at `asset`; there is no biome set to find"
        );
    }

    /// The v15 downgrade is lossless **except** for the per-sample biome ids —
    /// and, less obviously, for `biome_set`: [`TerrainV15`] has no such field, so
    /// the reference to the biome vocabulary is lost with the ids that named it.
    /// (That is the honest reading of "v15 could not express biomes": neither
    /// half of the feature has a v15 home.)
    ///
    /// Proven as a property — round-trip a live record through the frozen shape
    /// and assert the *whole record* is unchanged once the biome layer is put
    /// back — rather than by listing fields, so a slot added later cannot silently
    /// fall out of the ladder.
    #[test]
    fn v15_entity_downgrade_is_lossless_except_for_the_biome_ids() {
        let mut terrain = Terrain::configured(4, 2.0);
        terrain.data.author_tile((0, 0), |x, z| x + z);
        // Both post-v14 layers, so the test can tell "dropped the biomes" apart
        // from "dropped everything the frozen generation could not carry".
        terrain
            .data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_map_texel(4, 1, 1, [5.0, 2.0, 1.0]);
        terrain
            .data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_biome_sample(4, 1, 1, 7);
        terrain.biome_set = Some(uuid::Uuid::from_u128(0xB10E));
        assert!(!terrain.data.biomes_are_default());

        let live = EntityRecord {
            terrain: Some(terrain),
            ..v9_base(uuid::Uuid::from_u128(0xE100), "T", None)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current()
        };
        let back = EntityRecordV15::from_current(live.clone())
            .into_v16()
            .into_v17()
            .into_v18()
            .into_current();

        // Everything but the biome layer survives — the erosion maps included,
        // which is what makes this a v15 record and not a v14 one …
        let t = back.terrain.clone().unwrap();
        assert_eq!(t.tile_resolution, 4);
        assert_eq!(t.data.tile_count(), 1);
        assert_eq!(
            t.data.get_tile((0, 0)).unwrap().map_texel(4, 1, 1),
            [5.0, 2.0, 1.0]
        );
        assert_eq!(
            t.data.get_tile((0, 0)).unwrap().heights(),
            live.terrain
                .as_ref()
                .unwrap()
                .data
                .get_tile((0, 0))
                .unwrap()
                .heights()
        );

        // … and the biome ids plus their vocabulary are exactly what is lost.
        assert!(t.data.biomes_are_default());
        assert_eq!(
            t.biome_set, None,
            "TerrainV15 has no biome_set field to carry"
        );

        // The property: put the biome layer back and the records are equal, so
        // nothing outside it moved.
        let restored = Terrain {
            data: live.terrain.as_ref().unwrap().data.clone(),
            biome_set: live.terrain.as_ref().unwrap().biome_set,
            ..t
        };
        assert_eq!(
            EntityRecord {
                terrain: Some(restored),
                ..back
            },
            live,
            "nothing outside the biome layer moved"
        );
    }

    /// The v16 addition round-trips through the whole editor codec: a terrain
    /// carrying biome ids saves and reloads byte-identically, the payload is
    /// stamped v16, and the wire cost is priced rather than guessed.
    #[test]
    fn v16_biome_ids_round_trip_through_the_codec() {
        let mut terrain = Terrain::configured(4, 2.0);
        terrain.data.author_tile((0, 0), |x, z| x - z);
        terrain
            .data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_biome_sample(4, 2, 1, 9);
        terrain.biome_set = Some(uuid::Uuid::from_u128(0xB10E));

        let file = SceneFile {
            schema_version: SCHEMA_VERSION,
            title: "Painted".into(),
            entities: vec![EntityRecord {
                terrain: Some(terrain.clone()),
                ..v9_base(uuid::Uuid::from_u128(0xE200), "T", None)
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current()
            }],
            settings: LevelSettings::default(),
        };
        let bytes = bincode::serde::encode_to_vec(&file, bincode_config()).unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = decode(&bytes).expect("v16 decodes");
        let t = back.entities[0].terrain.clone().unwrap();
        assert_eq!(t.data.get_tile((0, 0)).unwrap().biome_sample(4, 2, 1), 9);
        assert_eq!(
            t.data.get_tile((0, 0)).unwrap().biome_sample(4, 0, 0),
            inf_terrain::UNASSIGNED_BIOME
        );
        assert_eq!(t.biome_set, Some(uuid::Uuid::from_u128(0xB10E)));
        // Re-encoding is byte-identical — biome ids are as byte-stable as heights.
        assert_eq!(
            bincode::serde::encode_to_vec(&back, bincode_config()).unwrap(),
            bytes
        );

        // ── the pricing half ──────────────────────────────────────────────
        // Two independent contributions, both measured. First `biome_set` alone,
        // on a tile-less terrain so the per-tile counts cannot mask it: at `None`
        // it is a bare bincode `Option` discriminant.
        let bare = Terrain::configured(4, 2.0);
        let biome_set_cost = bincode::serde::encode_to_vec(&bare, bincode_config())
            .unwrap()
            .len()
            - bincode::serde::encode_to_vec(TerrainV15::from_current(bare), bincode_config())
                .unwrap()
                .len();

        // Then the tile layer: an unpainted 1-tile terrain costs exactly one
        // extra length byte per tile over what v15 would have written, plus the
        // `biome_set` discriminant above.
        let mut plain = terrain.clone();
        plain.data.get_tile_mut((0, 0)).unwrap().clear_biomes();
        plain.biome_set = None;
        let v15_terrain = bincode::serde::encode_to_vec(
            TerrainV15::from_current(plain.clone()),
            bincode_config(),
        )
        .unwrap();
        let v16_terrain = bincode::serde::encode_to_vec(&plain, bincode_config()).unwrap();
        assert_eq!(
            v16_terrain.len(),
            v15_terrain.len() + 1 + biome_set_cost,
            "an unpainted 1-tile terrain must cost exactly one extra byte for the empty \
             biome sequence, plus {biome_set_cost} for `biome_set: None`"
        );

        // A painted tile then pays its dense buffer — one `u8` per sample, NOT
        // ×4 like the splat weights and not ×4×channels like the erosion maps.
        let mut painted = plain.clone();
        painted
            .data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_biome_sample(4, 2, 1, 9);
        assert_eq!(
            bincode::serde::encode_to_vec(&painted, bincode_config())
                .unwrap()
                .len(),
            v16_terrain.len() + 4 * 4,
            "a painted tile costs exactly its dense res² buffer of u8 biome ids"
        );
    }

    /// **The two-ladder tripwire, scene half.** Each frozen tile layout stands in
    /// for payloads from *two* independently-versioned containers, and neither
    /// container knows about the other's numbering — so nothing but a pin stops
    /// one of them bumping past its row and quietly decoding its own old payloads
    /// through the wrong wire type, positionally, into the next record's bytes:
    ///
    /// | frozen tile shape | carries | `.inf_lvl` | `.inf_terrain` |
    /// |---|---|---|---|
    /// | `TerrainTileFrozenV1` | origin + heights + weights | v1..=v14 | v1..=v2 |
    /// | `TerrainTileFrozenV2` | + erosion data maps | v15 | v3 |
    /// | live `TerrainTile` | + per-sample biome ids | v16+ | v4+ |
    ///
    /// `inf-terrain` carries the asset half of this assertion
    /// (`frozen_tile_generation_is_pinned_to_both_ladders`); this is the scene
    /// half, and the runtime codec mirrors it.
    #[test]
    fn the_frozen_tile_generation_covers_this_schema() {
        assert_eq!(
            SCHEMA_VERSION, 19,
            "the scene schema moved. Generation-1 frozen tiles (TerrainTileFrozenV1, via \
             TerrainV14) cover .inf_lvl v1..=v14, generation-2 (TerrainTileFrozenV2, via \
             TerrainV15) covers v15, and the live TerrainTile covers v16+. If the TILE \
             layout changed again, add inf_terrain::TerrainTileFrozenV3 and a new frozen \
             Terrain record; if only the scene changed, update this pin and \
             TerrainTileFrozenV1's generation table. (v17, v18 and v19 are all the latter \
             case: each appended an entity slot and left every tile layout alone — v19's \
             voxel volume in particular extends the ground LOCALLY, out in its own \
             .inf_voxel, and does not touch a single heightfield tile.)"
        );
        // The mapping the pin is about, one rung at a time. Start from a terrain
        // that exercises BOTH post-v14 layers.
        let mut live = fixture_terrain();
        {
            let tile = live.data.get_tile_mut((0, 0)).unwrap();
            tile.set_map_texel(4, 1, 1, [1.0, 2.0, 3.0]);
            tile.set_biome_sample(4, 1, 1, 5);
        }
        assert!(!live.data.data_maps_are_default());
        assert!(!live.data.biomes_are_default());

        // Rung 1 — generation 1 can carry neither addition.
        let gen1 = TerrainV14::from_current(live.clone()).into_current();
        assert!(
            gen1.data.data_maps_are_default() && gen1.data.biomes_are_default(),
            "generation 1 cannot carry data maps or biome ids — that is what makes it frozen"
        );

        // Rung 2 — generation 2 carries the maps but still not the biome ids,
        // and has no `biome_set` field at all.
        let gen2 = TerrainV15::from_current(live.clone()).into_current();
        assert!(
            !gen2.data.data_maps_are_default(),
            "generation 2 exists precisely to carry the erosion data maps"
        );
        assert_eq!(
            gen2.data.get_tile((0, 0)).unwrap().map_texel(4, 1, 1),
            [1.0, 2.0, 3.0]
        );
        assert!(
            gen2.data.biomes_are_default(),
            "generation 2 cannot carry biome ids — that is what makes it frozen"
        );
        assert_eq!(
            gen2.biome_set, None,
            "TerrainV15 has no biome_set field to round-trip"
        );
    }

    // \u2500\u2500 schema v17 (P20.1 water) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    /// An all-`None` frozen v16 entity \u2014 the struct-update base for
    /// [`v16_reference`]. Built through the downgrade hop so the field list can
    /// never drift from the live record.
    fn v16_base(guid: uuid::Uuid, name: &str, parent: Option<uuid::Uuid>) -> EntityRecordV16 {
        EntityRecordV16::from_current(
            v9_base(guid, name, parent)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current(),
        )
    }

    /// The **v16** terrain the fixture carries: two authored tiles, a painted
    /// splat sample, a materialized erosion data map, **a painted biome id and a
    /// `biome_set` reference** (the two things v16 could express that v15 could
    /// not), a non-default macro variation and an asset reference \u2014 so the v17 hop
    /// is proven to preserve what v16 authored, not merely to produce defaults.
    ///
    /// The literals must match `inf-scene`'s `v16_fixture_terrain` exactly \u2014 the
    /// two committed fixtures are byte-compared by
    /// [`v16_fixture_matches_the_runtime_codecs_copy`], which is the whole point
    /// of writing them twice.
    fn v16_fixture_terrain() -> Terrain {
        let mut t = Terrain::configured(4, 2.0);
        let f = |x: f64, z: f64| x * 0.5 - z * 0.25 + 3.0;
        t.data.author_tile((0, 0), f);
        t.data.author_tile((1, 0), f);
        t.data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_weight_sample(4, 1, 2, [40, 100, 80, 35]);
        t.data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_map_texel(4, 1, 1, [7.5, 0.5, 2.25]);
        t.data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_biome_sample(4, 1, 1, 3);
        t.macro_variation = 0.25;
        t.asset = Some(uuid::Uuid::from_u128(0xF_00AA));
        t.biome_set = Some(uuid::Uuid::from_u128(0xF_00BB));
        t
    }

    /// Rebuild the exact schema-v16 file the committed v16 fixture was generated
    /// from, out of the frozen v16 record types (the provenance lock).
    fn v16_reference() -> SceneFileV16 {
        use inf_ecs::components::{Light, LightKind, Material, MeshRef, Primitive};
        let g = uuid::Uuid::from_u128;
        SceneFileV16 {
            schema_version: 16,
            title: "V16 Fixture Level".into(),
            entities: vec![
                EntityRecordV16 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xF0A1)),
                    }),
                    material: Some(Material::default()),
                    ..v16_base(g(0xF001), "Cube", None)
                },
                EntityRecordV16 {
                    terrain: Some(v16_fixture_terrain()),
                    ..v16_base(g(0xF002), "Terrain", None)
                },
                EntityRecordV16 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v16_base(g(0xF003), "Sun", None)
                },
            ],
            settings: LevelSettings {
                gravity_2d: Vec2d::new(0.0, -18.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 90.0,
                render: RenderSettingsRecord {
                    exposure: 1.1,
                    ..RenderSettingsRecord::default()
                },
                partition: PartitionSettings::default(),
            },
        }
    }

    /// Write the committed v16 fixture from [`v16_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline). Never hand-edit
    /// the committed bytes.
    #[test]
    fn bless_v16_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v16_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 16);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v16.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v16 fixture");
        eprintln!("blessed v16 fixture: {}", path.display());
    }

    #[test]
    fn v16_fixture_is_reproducible_and_genuinely_v16() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v16.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v16 fixture present");
        assert_eq!(bytes[0], 16, "fixture must be a genuine schema-v16 payload");
        let rebuilt = bincode::serde::encode_to_vec(v16_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v16 fixture must match our frozen v16 writer"
        );
    }

    /// This crate's committed v16 fixture must be **byte-identical** to the Ring-0
    /// runtime reader's \u2014 the two codecs are one wire contract written twice.
    #[test]
    fn v16_fixture_matches_the_runtime_codecs_copy() {
        let mine = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v16.inf_lvl");
        let theirs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../crates/inf-scene/tests/fixtures/scene_v16.inf_lvl");
        assert_eq!(
            std::fs::read(&mine).expect("editor v16 fixture"),
            std::fs::read(&theirs).expect("runtime v16 fixture"),
            "the two v17-bump fixtures diverged — the codecs are no longer mirrors"
        );
    }

    /// The committed v16 fixture \u2014 written by the **pre-v17 codec**, before the
    /// entity record grew its `water_body` slot \u2014 still loads, with the v16
    /// content preserved verbatim and no water conjured. The "old bytes load
    /// forever" gate for the v17 bump.
    #[test]
    fn v16_loads_and_lifts_without_water() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v16.inf_lvl");
        let file = decode(&std::fs::read(&path).unwrap()).expect("v16 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V16 Fixture Level");
        assert_eq!(file.entities.len(), 3);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        // The v16 content survives the frozen-record hop intact \u2026
        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(uuid::Uuid::from_u128(0xF0A1))
        );
        assert_eq!(by_name("Sun").light.unwrap().intensity, 2.0);
        assert_eq!(file.settings.sim_hz, 90.0);
        let t = by_name("Terrain").terrain.clone().expect("terrain slot");
        assert_eq!(t.biome_set, Some(uuid::Uuid::from_u128(0xF_00BB)));
        assert_eq!(t.data.get_tile((0, 0)).unwrap().biome_sample(4, 1, 1), 3);
        assert_eq!(
            t.data.get_tile((0, 0)).unwrap().map_texel(4, 1, 1),
            [7.5, 0.5, 2.25]
        );

        // \u2026 and the one new slot lifts to `None` \u2014 a level with no water, which is
        // exactly what a v16 level was.
        for r in &file.entities {
            assert!(
                r.water_body.is_none(),
                "a v16 level has no water; the lift must not conjure any"
            );
        }
    }

    /// The v16 downgrade is lossless **except** for the water body \u2014 the one thing
    /// v16 cannot express. Proven as a property (round-trip a live record through
    /// the frozen shape) rather than by listing fields, so a slot added later
    /// cannot silently fall out of the ladder.
    #[test]
    fn v16_entity_downgrade_is_lossless_except_for_the_water_body() {
        use inf_ecs::components::{WaterBody, WaterKind};

        let live = EntityRecord {
            terrain: Some(v16_fixture_terrain()),
            spline: Some(Spline::default()),
            water_body: Some(WaterBody {
                kind: WaterKind::River,
                river_flow_m_s: 2.5,
                ..WaterBody::default()
            }),
            ..v9_base(uuid::Uuid::from_u128(0xF100), "Rill", None)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current()
        };
        let back = EntityRecordV16::from_current(live.clone())
            .into_v17()
            .into_v18()
            .into_current();

        // The water is exactly what is lost \u2026
        assert!(back.water_body.is_none());
        // \u2026 and nothing else moved: put it back and the records are equal, which is
        // the property form of "only this field".
        assert_eq!(
            EntityRecord {
                water_body: live.water_body,
                ..back
            },
            live,
            "the v16 downgrade lost something other than the water body"
        );
    }

    /// The v17 addition round-trips through the whole editor codec, and a water
    /// body really does reach an ECS world through `write_record_components`
    /// (the read half of the codec, which a bytes-only test would never touch).
    #[test]
    fn v17_water_round_trips_through_the_codec_and_the_world() {
        use inf_ecs::components::{WaterBody, WaterKind};

        let water = WaterBody {
            kind: WaterKind::Lake,
            level_m: 12.5,
            extent: Vec2d::new(40.0, 25.0),
            wave_seed: 0xABCD,
            wind_from_weather: false,
            ..WaterBody::default()
        };
        let file = SceneFile {
            schema_version: SCHEMA_VERSION,
            title: "Wet".into(),
            entities: vec![EntityRecord {
                water_body: Some(water),
                ..v9_base(uuid::Uuid::from_u128(0xF200), "Lake", None)
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current()
            }],
            settings: LevelSettings::default(),
        };
        let bytes = bincode::serde::encode_to_vec(&file, bincode_config()).unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = decode(&bytes).expect("the current schema decodes");
        assert_eq!(back.entities[0].water_body, Some(water));
        // Re-encoding is byte-identical.
        assert_eq!(
            bincode::serde::encode_to_vec(&back, bincode_config()).unwrap(),
            bytes
        );

        // \u2026 and it lands on a real entity, which is what `record_of` then reads back.
        let mut doc = SceneDoc::new();
        apply_to_doc(&mut doc, &back);
        let guid = uuid::Uuid::from_u128(0xF200);
        let e = doc.world().entity_of(guid).unwrap();
        assert_eq!(
            doc.world().world().get::<WaterBody>(e).copied(),
            Some(water)
        );
        assert_eq!(record_of(&doc, guid).unwrap().water_body, Some(water));
    }

    /// **The v17 price, isolated**: an entity with no water pays exactly one
    /// discriminant byte. Measured as a delta between the frozen v16 and frozen
    /// v17 shapes of the very same record, so it is a *price* rather than an
    /// absolute that could silently absorb any other growth — v18's own slot
    /// included, which is why this prices the frozen v17 shape and not the live
    /// record it used to.
    #[test]
    fn v17_costs_one_byte_per_water_free_entity() {
        let live = v9_base(uuid::Uuid::from_u128(0xF300), "Dry", None)
            .into_v10()
            .into_v11()
            .into_v12()
            .into_v13()
            .into_current();
        let v17 = bincode::serde::encode_to_vec(
            EntityRecordV17::from_current(live.clone()),
            bincode_config(),
        )
        .unwrap();
        let v16 = bincode::serde::encode_to_vec(
            EntityRecordV16::from_current(live.clone()),
            bincode_config(),
        )
        .unwrap();
        assert_eq!(
            v17.len(),
            v16.len() + 1,
            "the v17 slot must cost exactly one discriminant byte on a dry entity"
        );

        // A record that *carries* water costs more than its own discriminant.
        let wet = EntityRecordV17::from_current(EntityRecord {
            water_body: Some(inf_ecs::components::WaterBody::default()),
            ..live
        });
        assert!(
            bincode::serde::encode_to_vec(&wet, bincode_config())
                .unwrap()
                .len()
                > v17.len() + 1
        );
    }

    // ── schema v18 (P20.2 buoyancy) ───────────────────────────────────────

    /// An all-`None` frozen v17 entity — the struct-update base for
    /// [`v17_reference`]. Built through the downgrade hop so the field list can
    /// never drift from the live record.
    fn v17_base(guid: uuid::Uuid, name: &str, parent: Option<uuid::Uuid>) -> EntityRecordV17 {
        EntityRecordV17::from_current(
            v9_base(guid, name, parent)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current(),
        )
    }

    /// The **v17** water body the fixture's river carries: a spline river with a
    /// non-default flow, cross-section, seed and wind — the thing v17 could
    /// express and v16 could not — so the v18 hop is proven to preserve what v17
    /// authored rather than merely to produce defaults.
    ///
    /// The literals must match `inf-scene`'s `v17_fixture_water` exactly — the two
    /// committed fixtures are byte-compared by
    /// [`v17_fixture_matches_the_runtime_codecs_copy`], which is the whole point
    /// of writing them twice.
    fn v17_fixture_water() -> WaterBody {
        use inf_ecs::components::WaterKind;
        WaterBody {
            kind: WaterKind::River,
            level_m: 4.5,
            river_width_start_m: 6.0,
            river_width_end_m: 9.5,
            river_depth_start_m: 1.25,
            river_depth_end_m: 2.5,
            river_flow_m_s: 2.75,
            wave_seed: 0xC0FFEE,
            wind_from_weather: false,
            wind_x: 3.5,
            wind_z: -1.5,
            ..WaterBody::default()
        }
    }

    /// Rebuild the exact schema-v17 file the committed v17 fixture was generated
    /// from, out of the frozen v17 record types (the provenance lock). Carries the
    /// v16 fixture's terrain unchanged — v17 touched no tile layout — plus the
    /// river entity that only v17 could write.
    fn v17_reference() -> SceneFileV17 {
        use inf_ecs::components::{Light, LightKind, Material, MeshRef, Primitive};
        let g = uuid::Uuid::from_u128;
        SceneFileV17 {
            schema_version: 17,
            title: "V17 Fixture Level".into(),
            entities: vec![
                EntityRecordV17 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xF1A1)),
                    }),
                    material: Some(Material::default()),
                    ..v17_base(g(0xF101), "Cube", None)
                },
                EntityRecordV17 {
                    terrain: Some(v16_fixture_terrain()),
                    ..v17_base(g(0xF102), "Terrain", None)
                },
                EntityRecordV17 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v17_base(g(0xF103), "Sun", None)
                },
                // The river: a `Spline` centreline and a `WaterBody` on **one**
                // entity, which is the composition rule v17 established.
                EntityRecordV17 {
                    spline: Some(Spline {
                        points: vec![
                            Vec3d::new(0.0, 0.0, 0.0),
                            Vec3d::new(10.0, 0.0, 4.0),
                            Vec3d::new(18.0, 0.0, 14.0),
                        ],
                        ..Spline::default()
                    }),
                    water_body: Some(v17_fixture_water()),
                    ..v17_base(g(0xF104), "River", None)
                },
            ],
            settings: LevelSettings {
                gravity_2d: Vec2d::new(0.0, -18.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 90.0,
                render: RenderSettingsRecord {
                    exposure: 1.1,
                    ..RenderSettingsRecord::default()
                },
                partition: PartitionSettings::default(),
            },
        }
    }

    /// Write the committed v17 fixture from [`v17_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline). Never hand-edit
    /// the committed bytes.
    #[test]
    fn bless_v17_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v17_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 17);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v17.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v17 fixture");
        eprintln!("blessed v17 fixture: {}", path.display());
    }

    #[test]
    fn v17_fixture_is_reproducible_and_genuinely_v17() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v17.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v17 fixture present");
        assert_eq!(bytes[0], 17, "fixture must be a genuine schema-v17 payload");
        let rebuilt = bincode::serde::encode_to_vec(v17_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v17 fixture must match our frozen v17 writer"
        );
    }

    /// This crate's committed v17 fixture must be **byte-identical** to the Ring-0
    /// runtime reader's — the two codecs are one wire contract written twice.
    #[test]
    fn v17_fixture_matches_the_runtime_codecs_copy() {
        let mine = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v17.inf_lvl");
        let theirs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../crates/inf-scene/tests/fixtures/scene_v17.inf_lvl");
        assert_eq!(
            std::fs::read(&mine).expect("editor v17 fixture"),
            std::fs::read(&theirs).expect("runtime v17 fixture"),
            "the two v18-bump fixtures diverged — the codecs are no longer mirrors"
        );
    }

    /// The committed v17 fixture — written by the **pre-v18 codec**, before the
    /// entity record grew its `buoyancy` slot — still loads, with the v17 content
    /// (the river's water body included) preserved verbatim and nothing made to
    /// float. The "old bytes load forever" gate for the v18 bump.
    #[test]
    fn v17_loads_and_lifts_without_buoyancy() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v17.inf_lvl");
        let file = decode(&std::fs::read(&path).unwrap()).expect("v17 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V17 Fixture Level");
        assert_eq!(file.entities.len(), 4);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        // The v17 content survives the frozen-record hop intact — including the
        // slot v17 itself added, which is the half a defaults-only fixture would
        // never have proven …
        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(uuid::Uuid::from_u128(0xF1A1))
        );
        assert_eq!(by_name("Sun").light.unwrap().intensity, 2.0);
        assert_eq!(file.settings.sim_hz, 90.0);
        let t = by_name("Terrain").terrain.clone().expect("terrain slot");
        assert_eq!(t.biome_set, Some(uuid::Uuid::from_u128(0xF_00BB)));
        assert_eq!(t.data.get_tile((0, 0)).unwrap().biome_sample(4, 1, 1), 3);
        let river = by_name("River");
        assert_eq!(river.water_body, Some(v17_fixture_water()));
        assert_eq!(river.spline.as_ref().unwrap().points.len(), 3);

        // … and the one new slot lifts to `None` — a level in which nothing
        // floats, which is exactly what a v17 level was.
        for r in &file.entities {
            assert!(
                r.buoyancy.is_none(),
                "a v17 level floats nothing; the lift must not conjure buoyancy"
            );
        }
    }

    /// The v17 downgrade is lossless **except** for the buoyancy — the one thing
    /// v17 cannot express. Proven as a property (round-trip a live record through
    /// the frozen shape) rather than by listing fields, so a slot added later
    /// cannot silently fall out of the ladder.
    #[test]
    fn v17_entity_downgrade_is_lossless_except_for_the_buoyancy() {
        let live = EntityRecord {
            terrain: Some(v16_fixture_terrain()),
            spline: Some(Spline::default()),
            water_body: Some(v17_fixture_water()),
            buoyancy: Some(Buoyancy {
                density_kg_m3: 450.0,
                linear_drag: 3.5,
                ..Buoyancy::default()
            }),
            ..v9_base(uuid::Uuid::from_u128(0xF400), "Raft", None)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current()
        };
        let back = EntityRecordV17::from_current(live.clone())
            .into_v18()
            .into_current();

        // The buoyancy is exactly what is lost — the water, which v17 *can*
        // express, is not …
        assert!(back.buoyancy.is_none());
        assert_eq!(back.water_body, live.water_body);
        // … and nothing else moved: put it back and the records are equal, which
        // is the property form of "only this field".
        assert_eq!(
            EntityRecord {
                buoyancy: live.buoyancy,
                ..back
            },
            live,
            "the v17 downgrade lost something other than the buoyancy"
        );
    }

    /// The v18 addition round-trips through the whole editor codec — including
    /// the **new decode arm**, which only a payload stamped v18 exercises — and a
    /// buoyancy really does reach an ECS world through `write_record_components`
    /// (the read half of the codec, which a bytes-only test would never touch).
    #[test]
    fn v18_buoyancy_round_trips_through_the_codec_and_the_world() {
        let float = Buoyancy {
            enabled: true,
            density_kg_m3: 500.0,
            fluid_density_kg_m3: 1025.0,
            linear_drag: 2.5,
            angular_drag: 1.25,
        };
        let file = SceneFile {
            schema_version: SCHEMA_VERSION,
            title: "Floating".into(),
            entities: vec![EntityRecord {
                buoyancy: Some(float),
                ..v9_base(uuid::Uuid::from_u128(0xF500), "Crate", None)
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current()
            }],
            settings: LevelSettings::default(),
        };
        let bytes = bincode::serde::encode_to_vec(&file, bincode_config()).unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = decode(&bytes).expect("the current schema decodes");
        assert_eq!(back.entities[0].buoyancy, Some(float));
        // Re-encoding is byte-identical.
        assert_eq!(
            bincode::serde::encode_to_vec(&back, bincode_config()).unwrap(),
            bytes
        );

        // … and it lands on a real entity, which is what `record_of` then reads back.
        let mut doc = SceneDoc::new();
        apply_to_doc(&mut doc, &back);
        let guid = uuid::Uuid::from_u128(0xF500);
        let e = doc.world().entity_of(guid).unwrap();
        assert_eq!(doc.world().world().get::<Buoyancy>(e).copied(), Some(float));
        assert_eq!(record_of(&doc, guid).unwrap().buoyancy, Some(float));
    }

    /// **The v18 price, isolated**: an entity that does not float pays exactly one
    /// discriminant byte. Measured as a delta between the frozen v17 and frozen
    /// v18 shapes of the very same record, so it is a *price* rather than an
    /// absolute that could silently absorb any other growth — v19's own slot
    /// included, which is why this prices the frozen v18 shape and not the live
    /// record it used to.
    #[test]
    fn v18_costs_one_byte_per_buoyancy_free_entity() {
        let live = v9_base(uuid::Uuid::from_u128(0xF600), "Rock", None)
            .into_v10()
            .into_v11()
            .into_v12()
            .into_v13()
            .into_current();
        let v18 = bincode::serde::encode_to_vec(
            EntityRecordV18::from_current(live.clone()),
            bincode_config(),
        )
        .unwrap();
        let v17 = bincode::serde::encode_to_vec(
            EntityRecordV17::from_current(live.clone()),
            bincode_config(),
        )
        .unwrap();
        assert_eq!(
            v18.len(),
            v17.len() + 1,
            "the v18 slot must cost exactly one discriminant byte on a sinking entity"
        );

        // A record that *carries* buoyancy costs more than its own discriminant.
        let afloat = EntityRecordV18::from_current(EntityRecord {
            buoyancy: Some(Buoyancy::default()),
            ..live
        });
        assert!(
            bincode::serde::encode_to_vec(&afloat, bincode_config())
                .unwrap()
                .len()
                > v18.len() + 1
        );
    }

    // ── schema v19 (P21.1 volumetric terrain) ─────────────────────────────

    /// An all-`None` frozen v18 entity — the struct-update base for
    /// [`v18_reference`]. Built through the downgrade hop so the field list can
    /// never drift from the live record.
    fn v18_base(guid: uuid::Uuid, name: &str, parent: Option<uuid::Uuid>) -> EntityRecordV18 {
        EntityRecordV18::from_current(
            v9_base(guid, name, parent)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current(),
        )
    }

    /// The **v18** buoyancy the fixture's raft carries: non-default in four of the
    /// five fields — the thing v18 could express and v17 could not — so the v19
    /// hop is proven to preserve what v18 authored rather than merely to produce
    /// defaults.
    ///
    /// The literals must match `inf-scene`'s `v18_fixture_buoyancy` exactly — the
    /// two committed fixtures are byte-compared by
    /// [`v18_fixture_matches_the_runtime_codecs_copy`], which is the whole point
    /// of writing them twice.
    fn v18_fixture_buoyancy() -> Buoyancy {
        Buoyancy {
            enabled: true,
            density_kg_m3: 420.0,
            fluid_density_kg_m3: 1035.0,
            linear_drag: 3.25,
            angular_drag: 1.75,
        }
    }

    /// Rebuild the exact schema-v18 file the committed v18 fixture was generated
    /// from, out of the frozen v18 record types (the provenance lock). Carries the
    /// v16 fixture's terrain and the v17 fixture's river unchanged — v19 touched
    /// neither a tile layout nor a component's shape — plus the raft entity that
    /// only v18 could write.
    fn v18_reference() -> SceneFileV18 {
        use inf_ecs::components::{Light, LightKind, Material, MeshRef, Primitive};
        let g = uuid::Uuid::from_u128;
        SceneFileV18 {
            schema_version: 18,
            title: "V18 Fixture Level".into(),
            entities: vec![
                EntityRecordV18 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xF2A1)),
                    }),
                    material: Some(Material::default()),
                    ..v18_base(g(0xF201), "Cube", None)
                },
                EntityRecordV18 {
                    terrain: Some(v16_fixture_terrain()),
                    ..v18_base(g(0xF202), "Terrain", None)
                },
                EntityRecordV18 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v18_base(g(0xF203), "Sun", None)
                },
                // The river: a `Spline` centreline and a `WaterBody` on **one**
                // entity, which is the composition rule v17 established.
                EntityRecordV18 {
                    spline: Some(Spline {
                        points: vec![
                            Vec3d::new(0.0, 0.0, 0.0),
                            Vec3d::new(10.0, 0.0, 4.0),
                            Vec3d::new(18.0, 0.0, 14.0),
                        ],
                        ..Spline::default()
                    }),
                    water_body: Some(v17_fixture_water()),
                    ..v18_base(g(0xF204), "River", None)
                },
                // The raft: the dynamic body and the opt-in flotation that v18
                // added, which is the content this fixture exists to carry.
                EntityRecordV18 {
                    rigid_body_3d: Some(RigidBody3D::default()),
                    buoyancy: Some(v18_fixture_buoyancy()),
                    ..v18_base(g(0xF205), "Raft", None)
                },
            ],
            settings: LevelSettings {
                gravity_2d: Vec2d::new(0.0, -18.0),
                gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
                sim_hz: 90.0,
                render: RenderSettingsRecord {
                    exposure: 1.1,
                    ..RenderSettingsRecord::default()
                },
                partition: PartitionSettings::default(),
            },
        }
    }

    /// Write the committed v18 fixture from [`v18_reference`] under
    /// `INF_BLESS_FIXTURES=1` (the temporary-writer discipline). Never hand-edit
    /// the committed bytes.
    #[test]
    fn bless_v18_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").is_err() {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v18_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 18);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v18.inf_lvl");
        std::fs::write(&path, &bytes).expect("write v18 fixture");
        eprintln!(
            "blessed v18 fixture: {} ({} bytes)",
            path.display(),
            bytes.len()
        );
    }

    #[test]
    fn v18_fixture_is_reproducible_and_genuinely_v18() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v18.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v18 fixture present");
        assert_eq!(bytes[0], 18, "fixture must be a genuine schema-v18 payload");
        let rebuilt = bincode::serde::encode_to_vec(v18_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v18 fixture must match our frozen v18 writer"
        );
    }

    /// This crate's committed v18 fixture must be **byte-identical** to the Ring-0
    /// runtime reader's — the two codecs are one wire contract written twice.
    #[test]
    fn v18_fixture_matches_the_runtime_codecs_copy() {
        let mine = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v18.inf_lvl");
        let theirs = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../crates/inf-scene/tests/fixtures/scene_v18.inf_lvl");
        assert_eq!(
            std::fs::read(&mine).expect("editor v18 fixture"),
            std::fs::read(&theirs).expect("runtime v18 fixture"),
            "the two v19-bump fixtures diverged — the codecs are no longer mirrors"
        );
    }

    /// The committed v18 fixture — written by the **pre-v19 codec**, before the
    /// entity record grew its `voxel_volume` slot — still loads, with the v18
    /// content (the raft's buoyancy included) preserved verbatim and no volumetric
    /// ground conjured. The "old bytes load forever" gate for the v19 bump.
    #[test]
    fn v18_loads_and_lifts_without_a_voxel_volume() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/scene_v18.inf_lvl");
        let file = decode(&std::fs::read(&path).unwrap()).expect("v18 fixture decodes");
        assert_eq!(file.schema_version, SCHEMA_VERSION);
        assert_eq!(file.title, "V18 Fixture Level");
        assert_eq!(file.entities.len(), 5);
        let by_name = |n: &str| file.entities.iter().find(|r| r.name == n).unwrap();

        // The v18 content survives the frozen-record hop intact — including the
        // slot v18 itself added, which is the half a defaults-only fixture would
        // never have proven …
        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(uuid::Uuid::from_u128(0xF2A1))
        );
        assert_eq!(by_name("Sun").light.unwrap().intensity, 2.0);
        assert_eq!(file.settings.sim_hz, 90.0);
        let t = by_name("Terrain").terrain.clone().expect("terrain slot");
        assert_eq!(t.biome_set, Some(uuid::Uuid::from_u128(0xF_00BB)));
        assert_eq!(t.data.get_tile((0, 0)).unwrap().biome_sample(4, 1, 1), 3);
        let river = by_name("River");
        assert_eq!(river.water_body, Some(v17_fixture_water()));
        assert_eq!(river.spline.as_ref().unwrap().points.len(), 3);
        let raft = by_name("Raft");
        assert_eq!(raft.buoyancy, Some(v18_fixture_buoyancy()));
        assert!(raft.rigid_body_3d.is_some());

        // … and the one new slot lifts to `None` — a level whose ground is a
        // heightfield and nothing else, which is exactly what a v18 level was.
        for r in &file.entities {
            assert!(
                r.voxel_volume.is_none(),
                "a v18 level has no volumetric ground; the lift must not conjure any"
            );
        }
    }

    /// The v18 downgrade is lossless **except** for the voxel volume — the one
    /// thing v18 cannot express. Proven as a property (round-trip a live record
    /// through the frozen shape) rather than by listing fields, so a slot added
    /// later cannot silently fall out of the ladder.
    #[test]
    fn v18_entity_downgrade_is_lossless_except_for_the_voxel_volume() {
        let live = EntityRecord {
            terrain: Some(v16_fixture_terrain()),
            spline: Some(Spline::default()),
            water_body: Some(v17_fixture_water()),
            buoyancy: Some(v18_fixture_buoyancy()),
            voxel_volume: Some(VoxelVolume {
                asset: Some(uuid::Uuid::from_u128(0xF_0CA5)),
                voxel_size_m: 0.25,
                runtime_carve: false,
            }),
            ..v9_base(uuid::Uuid::from_u128(0xF700), "Cavern", None)
                .into_v10()
                .into_v11()
                .into_v12()
                .into_v13()
                .into_current()
        };
        let back = EntityRecordV18::from_current(live.clone()).into_current();

        // The voxel volume is exactly what is lost — the water and the buoyancy,
        // which v18 *can* express, are not …
        assert!(back.voxel_volume.is_none());
        assert_eq!(back.water_body, live.water_body);
        assert_eq!(back.buoyancy, live.buoyancy);
        // … and nothing else moved: put it back and the records are equal, which
        // is the property form of "only this field".
        assert_eq!(
            EntityRecord {
                voxel_volume: live.voxel_volume,
                ..back
            },
            live,
            "the v18 downgrade lost something other than the voxel volume"
        );
    }

    /// The v19 addition round-trips through the whole editor codec — including
    /// the **new decode arm**, which only a payload stamped v19 exercises — and a
    /// voxel volume really does reach an ECS world through
    /// `write_record_components` (the read half of the codec, which a bytes-only
    /// test would never touch).
    #[test]
    fn v19_voxel_volume_round_trips_through_the_codec_and_the_world() {
        let volume = VoxelVolume {
            asset: Some(uuid::Uuid::from_u128(0xF_0CA5)),
            voxel_size_m: 0.25,
            runtime_carve: false,
        };
        let file = SceneFile {
            schema_version: SCHEMA_VERSION,
            title: "Caves".into(),
            entities: vec![EntityRecord {
                voxel_volume: Some(volume),
                ..v9_base(uuid::Uuid::from_u128(0xF800), "Cave", None)
                    .into_v10()
                    .into_v11()
                    .into_v12()
                    .into_v13()
                    .into_current()
            }],
            settings: LevelSettings::default(),
        };
        let bytes = bincode::serde::encode_to_vec(&file, bincode_config()).unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = decode(&bytes).expect("the current schema decodes");
        assert_eq!(back.entities[0].voxel_volume, Some(volume));
        // Re-encoding is byte-identical.
        assert_eq!(
            bincode::serde::encode_to_vec(&back, bincode_config()).unwrap(),
            bytes
        );

        // … and it lands on a real entity, which is what `record_of` then reads back.
        let mut doc = SceneDoc::new();
        apply_to_doc(&mut doc, &back);
        let guid = uuid::Uuid::from_u128(0xF800);
        let e = doc.world().entity_of(guid).unwrap();
        assert_eq!(
            doc.world().world().get::<VoxelVolume>(e).copied(),
            Some(volume)
        );
        assert_eq!(record_of(&doc, guid).unwrap().voxel_volume, Some(volume));
    }

    /// **The v19 price, isolated**: an entity with no volumetric ground pays
    /// exactly one discriminant byte. Measured as a delta between the frozen v18
    /// and live v19 encodings of the very same record, so it is a *price* rather
    /// than an absolute that could silently absorb any other growth.
    #[test]
    fn v19_costs_one_byte_per_voxel_free_entity() {
        let live = v9_base(uuid::Uuid::from_u128(0xF900), "Bedrock", None)
            .into_v10()
            .into_v11()
            .into_v12()
            .into_v13()
            .into_current();
        let v19 = bincode::serde::encode_to_vec(&live, bincode_config()).unwrap();
        let v18 = bincode::serde::encode_to_vec(
            EntityRecordV18::from_current(live.clone()),
            bincode_config(),
        )
        .unwrap();
        assert_eq!(
            v19.len(),
            v18.len() + 1,
            "the v19 slot must cost exactly one discriminant byte on a solid entity"
        );

        // A record that *carries* a volume costs more than its own discriminant —
        // and never more than its three fields, because the chunks live in the
        // `.inf_voxel` the GUID points at, not in the level.
        let carved = EntityRecord {
            voxel_volume: Some(VoxelVolume::default()),
            ..live
        };
        assert!(
            bincode::serde::encode_to_vec(&carved, bincode_config())
                .unwrap()
                .len()
                > v19.len() + 1
        );
    }
}
