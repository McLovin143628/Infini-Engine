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

pub mod partition;

pub use partition::{PartitionSettings, DEFAULT_CELL_SIZE_M};

use inf_ecs::components::{
    AlwaysLoaded, AnimPlayer, AnimStateMachine, AttachedTo, AudioListener, AudioSource, BlendMode,
    Buoyancy, Camera, CharacterController2D, CharacterController3D, CharacterMovement, ClothSim,
    Collider2D, Collider3D, Decal, Destructible, Foliage, HairGuides, IkTarget, Joint2D, Joint3D,
    Light, Light2D, LightKind, Material, MeshRef, NineSlice, PcgVolume, RigidBody2D, RigidBody3D,
    RootMotion, SkeletalMesh, SkyAtmosphere, Spline, Sprite, StreamingSource, Terrain,
    TerrainLayer, Text2D, Tilemap, TimeOfDay, Transform, Volume, VoxelVolume, WaterBody,
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
/// * **v10** — P16.5: the entity record appends two **world-partition** slots —
///   `streaming_source` ([`StreamingSource`], the sim-side driver of cell
///   residency) and `always_loaded` ([`AlwaysLoaded`], the never-streamed
///   marker) — and the file settings gain a `partition`
///   ([`PartitionSettings`]) block. The pre-v10 settings shape is frozen as
///   [`RuntimeSettingsV9`] and the pre-v10 entity record as [`EntityRecordV9`];
///   both lift with the new fields defaulted, i.e. `partition.enabled = false`,
///   which is exactly what an older level meant (no partitioning).
///
///   **Second bump in one phase, deliberately.** P16.3 shipped v9 before the
///   partition metadata existed as a design; retro-fitting `partition` into v9
///   would mean re-blessing bytes that are already committed and already load.
///   An append-only v10 is the honest alternative, and the frozen-record ladder
///   is exactly the machinery that makes a second bump cheap.
/// * **v11** — P17.1: the entity record appends the two **sky-authority** slots —
///   `time_of_day` ([`TimeOfDay`], the world clock the sun is a pure function of)
///   and `sky_atmosphere` ([`SkyAtmosphere`], how that sun lights the world and
///   tints the gradient). Retires the renderer's compile-time `SUN_DIR`. The file
///   settings are untouched, so the pre-v11 entity record is frozen as
///   [`EntityRecordV10`] and lifts with both slots `None` — a level with no clock,
///   which is exactly what every pre-v11 level was, and which the projectors
///   render with the retired constant's direction.
/// * **v12** — P17.2: [`SkyAtmosphere`] grew the **physical-atmosphere block** —
///   the physical-sky switch and its LUT knobs (`physical`, `sky_intensity`,
///   `turbidity`, `mie_anisotropy`), the sun/moon disc angular diameters
///   (`sun_disc_deg`, `moon_disc_deg`), `star_intensity`, the gradient
///   `tint_strength`, `aerial_perspective`, and height fog in **SI metres**
///   (`fog_density`, `fog_falloff`, `fog_height`, `fog_color`). That changed the
///   component's byte layout, so the pre-v12 shape is frozen as
///   [`SkyAtmosphereV11`] and the pre-v12 entity record as [`EntityRecordV11`]
///   (which carries `sky_atmosphere` as `Option<SkyAtmosphereV11>`, exactly as
///   v4..v8 carry `terrain` as `Option<TerrainV8>`). **No entity slot was added
///   or moved** and the file settings are untouched; v1..v11 payloads load
///   unchanged, with the 13 new fields lifted to their live
///   `SkyAtmosphere::default()` values — a gradient sky and no fog, which is what
///   a v11 level meant.
///
///   **Why a bump at all, when every new field is `#[serde(default)]`?** Because
///   bincode is **not self-describing**: the decoder reads a fixed field count
///   positionally, with no names or lengths on the wire, so a v11 payload fed to
///   the grown struct would keep reading past the end of its `SkyAtmosphere` and
///   into the next record's bytes. `#[serde(default)]` rescues only the
///   self-describing codecs (JSON/TOML). This is the same root cause as the house
///   law that `skip_serializing_if` desyncs bincode, and the frozen-record ladder
///   exists for exactly this case.
/// * **v13** — P17.3: [`SkyAtmosphere`] grew the **volumetric-cloud block** — the
///   cloud switch (`clouds_enabled`), the weather-field shape (`cloud_coverage`,
///   `cloud_type`, `cloud_detail`, `cloud_seed`), the layer slab in **SI metres**
///   (`cloud_bottom`, `cloud_top`), the optics (`cloud_density`, `cloud_phase_g`,
///   `cloud_shadow`, `cloud_ambient`, `cloud_color`) and the wind in **m/s**
///   (`cloud_wind_x`, `cloud_wind_z`). Same shape of bump as v12, for the same
///   reason: bincode is **positional**, so *growing a component is a wire-format
///   change* even though every new field is `#[serde(default)]` — a v12 payload
///   fed to the grown struct would read past the end of its `SkyAtmosphere` and
///   into the next record. So the pre-v13 shape is frozen as [`SkyAtmosphereV12`]
///   and the pre-v13 entity record as [`EntityRecordV12`] (which carries
///   `sky_atmosphere` as `Option<SkyAtmosphereV12>`). **Only the component's shape
///   changed** — no entity slot was added or moved and the file settings are
///   untouched — so v1..v12 payloads load unchanged, with the 14 new fields lifted
///   to their live `SkyAtmosphere::default()` values. That means
///   `clouds_enabled: false`: a v12 level had **no clouds**, which is exactly what
///   a v12 level meant.
/// * **v15** — P19.1: every terrain **tile** gained its sparse erosion data-map
///   layer (flow / deposition / wear). No component field was added and none
///   moved — the change is one level deeper, inside
///   [`inf_terrain::TerrainTile`]'s wire form — but it is the **same law** as
///   v12/v13 for the same reason: bincode is positional, so an extra
///   length-prefixed layer inside a tile is a wire-format change, and a v14
///   payload fed to the grown tile would read past the end of its heights and
///   into the next tile. The pre-v15 heightfield is therefore frozen as
///   [`inf_terrain::TerrainDataFrozenV1`], the pre-v15 component as [`TerrainV14`],
///   and the pre-v15 entity record as [`EntityRecordV14`] (which carries
///   `terrain` as `Option<TerrainV14>`, exactly as v4..v8 carry it as
///   `Option<TerrainV8>`). v1..v14 payloads load unchanged, with every tile's
///   maps lifted to **empty** — never eroded, which is exactly what a v14 level
///   meant. The cost to an un-eroded terrain is one zero-length count per tile.
/// * **v16** — P19.2: every terrain **tile** gained its sparse per-sample
///   **biome id** layer (`Vec<u8>`; empty means every sample is
///   [`inf_terrain::UNASSIGNED_BIOME`]), and [`Terrain`] itself gained a
///   `biome_set: Option<Uuid>` reference to the `.inf_biomes` vocabulary those
///   ids name.
///
///   **The tile layer is what forces the bump**, not the component field: bincode
///   is positional, so an extra length-prefixed layer *inside a tile* is a
///   wire-format change even though the field is `#[serde(default)]` — a v15
///   payload fed to the grown tile reads past the end of its data maps and into
///   the next tile. Fourth instance of the same law (v12, v13 and v15 were the
///   others). `biome_set` alone would have been an ordinary append at the tail of
///   one component, but it rides along for free once the bump is unavoidable.
///
///   So the pre-v16 heightfield is frozen as
///   [`inf_terrain::TerrainDataFrozenV2`], the pre-v16 component as [`TerrainV15`]
///   (which has **no** `biome_set` field — that is precisely what v16 added), and
///   the pre-v16 entity record as [`EntityRecordV15`] (which carries `terrain` as
///   `Option<TerrainV15>`, exactly as v9..v14 carry it as `Option<TerrainV14>`).
///   v1..v15 payloads load unchanged, with every tile's biome ids lifted to
///   **empty** and `biome_set: None` — an unpainted terrain with no biome
///   vocabulary, which is exactly what a v15 level meant. The cost to an unpainted
///   terrain is one zero-length count per tile plus one discriminant byte for the
///   `None` biome set.
///
/// **P19.3 bumped nothing.** The biome→PCG binding gave [`Terrain`] a
/// `biome_population: Vec<ScatteredInstance>`, but it is `#[serde(skip)]` — a
/// derived cache rebuilt by the editor's evaluate command and by the player on
/// level load — so it is **wire-neutral** and every ladder rung below stays
/// byte-identical. Same precedent as `PcgVolume::evaluated`, and the reason the
/// schema stays at 16: only what reaches the bytes can force a bump.
/// * **v17** — P20.1: the entity record appends the **water** slot — `water_body`
///   ([`WaterBody`]: an ocean, a lake or a spline river, with its Gerstner wave
///   state, its river cross-section and its shading). No component changed shape
///   and the file settings are untouched, so this is the [`EntityRecordV10`]
///   *shape* of bump (a new slot at the tail), not the [`EntityRecordV14`] one.
///   The pre-v17 entity record is frozen as [`EntityRecordV16`] and lifts with
///   `water_body: None` — a level with no water, which is exactly what every
///   pre-v17 level was.
///
///   **A river's centreline is the [`Spline`] on the same entity**, not a
///   reference, so v17 introduces no new asset edge and the cook's dependency
///   closure is unchanged. The wire cost to a water-free level is **one
///   discriminant byte per entity** — the `None` tag — which is the same price
///   every additive slot since v8 has paid.
/// * **v18** — P20.2: the entity record appends the **buoyancy** slot —
///   `buoyancy` ([`Buoyancy`]: opt-in flotation and hydrodynamic drag for a
///   dynamic 3D body). No component changed shape and the file settings are
///   untouched, so this is again the [`EntityRecordV10`] *shape* of bump (a new
///   slot at the tail), not the [`EntityRecordV14`] one. The pre-v18 entity
///   record is frozen as [`EntityRecordV17`] and lifts with `buoyancy: None` — a
///   level in which nothing floats, which is exactly what every pre-v18 level
///   was.
///
///   **Why the component exists at all, rather than flotation being a rule:** it
///   is opt-in because a default-on rule (every dynamic body floats, its density
///   read from its collider) would have silently rewritten the physics of every
///   dynamic body in any level that gained water — and `Collider3D::density`
///   defaults to `1.0`, which is rapier's mass placeholder and not a material
///   density, so under that rule essentially every existing body would bob like a
///   cork. The wire cost to a level where nothing floats is **one discriminant
///   byte per non-buoyant entity**, the same price every additive slot since v8
///   has paid.
/// * **v19** — P21.1: the entity record appends the **volumetric-terrain** slot —
///   `voxel_volume` ([`VoxelVolume`]: a sparse SDF voxel volume — the caves,
///   tunnels and excavations that *locally extend* the heightfield terrain). No
///   component changed shape and the file settings are untouched, so this is once
///   again the [`EntityRecordV10`] *shape* of bump (a new slot at the tail), not
///   the [`EntityRecordV14`] one (a component that grew). The pre-v19 entity
///   record is frozen as [`EntityRecordV18`] and lifts with `voxel_volume: None`
///   — a level whose ground is a heightfield and nothing else, which is exactly
///   what every pre-v19 level was.
///
///   **The planet-scale base stays a heightfield**, deliberately: the P16 clipmap
///   economics are unbeatable at that scale, and v19 does not voxelize the world.
///   A [`VoxelVolume`] is a *reference plus its two authored knobs* — the chunks
///   live in the `.inf_voxel` asset, which versions itself — so the only new edge
///   the cook follows is that one GUID, and the component can never be the reason
///   a future schema has to move: growing it would cost another bump in **both**
///   codec mirrors, which is why its three fields are frozen as shipped.
///
///   The wire cost to a level with no volumes is **one discriminant byte per
///   entity**, the same price every additive slot since v8 has paid.
/// * **v20** — P22.2: the entity record appends the **destruction** slot —
///   `destructible` ([`Destructible`]: the marker that says this entity's mesh
///   can break, plus the five numbers that decide how). No component changed
///   shape and the file settings are untouched, so this is once again the
///   [`EntityRecordV10`] *shape* of bump (a new slot at the tail), not the
///   [`EntityRecordV14`] one (a component that grew). The pre-v20 entity record
///   is frozen as [`EntityRecordV19`] and lifts with `destructible: None` — a
///   level in which nothing breaks, which is exactly what every pre-v20 level
///   was.
///
///   **The component references no asset**, which is why it is this cheap. What
///   breaks is the mesh already on the entity ([`MeshRef`]), and the chunk set is
///   *derived from that mesh at cook time* — a `.inf_fracture` whose GUID is a
///   pure function of the mesh's, exactly as a `.inf_vmesh`'s is. So v20 adds no
///   new edge to the cook's dependency closure, there is no fracture reference
///   to leave dangling, and no dangling-reference advisory to write.
///
///   **This is Phase 22's ONLY bump** and the five fields are frozen as shipped:
///   P22.3 and P22.4 must both fit inside them. `docs/memos/p22-strength.md`
///   argues why they suffice.
///
///   The wire cost to a level where nothing breaks is **one discriminant byte
///   per entity**, the same price every additive slot since v8 has paid.
/// * **v21** — P24.3: the entity record appends **three** character slots at
///   once — `ik_target` ([`IkTarget`]: the authored IK goals, which
///   `inf_ecs::pose::step_pose_evaluation` re-reads every fixed step),
///   `cloth_sim` ([`ClothSim`]) and `hair_guides` ([`HairGuides`]). No component
///   changed shape and the file settings are untouched, so this is once again the
///   [`EntityRecordV10`] *shape* of bump (new slots at the tail), not the
///   [`EntityRecordV14`] one (a component that grew). The pre-v21 entity record
///   is frozen as [`EntityRecordV20`] and lifts with all three `None` — a level
///   with no IK, no garments and no hair, which is exactly what every pre-v21
///   level was.
///
///   **Why three slots and not one.** A phase gets one scene bump. P24.3 needs
///   `ik_target` (it is what retires P24.2's "an IK target cannot be authored or
///   saved"); P24.4 needs cloth and hair, and would otherwise need a **v22**
///   inside the same phase. So the choice was one bump carrying three slots or
///   two bumps carrying one and two, and the empty slots cost a `None`
///   discriminant byte each — the price every additive slot since v8 has paid.
///   Stated plainly because a reserved slot is exactly the kind of thing that
///   rots — and this is the update that keeps it from rotting: **P24.4 gave
///   `cloth_sim` its reader** (`inf_ecs::cloth::step_cloth_simulation`, the ONE
///   Ring-0 rule both fixed steps call, whose result is folded into
///   `cloth_state_bytes` and compared between the two hosts). `hair_guides` got its reader in
///   the same batch (`inf_ecs::hair::step_hair_simulation`, folded into
///   `hair_state_bytes`), so all three v21 slots are now read. `ik_target` was read on the day it landed.
///
///   **Two of the three reference an asset, and one does not.** `ClothSim` and
///   `HairGuides` each carry an `Option<Uuid>` naming a `.inf_cloth` / `.inf_hair`
///   — asset kinds P24.4 defined (`AssetKind::Cloth` / `AssetKind::Hair`), which
///   was an append-only `AssetKind` addition and touched no scene wire. A slot
///   left at `None` still costs the cook's dependency closure nothing. `IkTarget` references no asset at all: a
///   chain is joint indices into the skeleton the entity's `SkeletalMesh` already
///   names, so v21 adds no edge for it and there is nothing to dangle.
///
///   All three follow the [`VoxelVolume`] law — *the component is a reference
///   plus its authored knobs, so it can never be the reason a future schema has
///   to move* — and are therefore **frozen as shipped**.
/// * **v22** — P26.3b: [`Material`] gained `asset: Option<Uuid>`, the persisted
///   `.inf_mat` binding. **No entity slot was added or moved** and the file
///   settings are untouched, so this is the [`EntityRecordV14`] *shape* of bump
///   (a component that grew), not the [`EntityRecordV10`] one — the pre-v22
///   component is frozen as [`MaterialV21`] and the pre-v22 entity record as
///   [`EntityRecordV21`] (which carries `material` as `Option<MaterialV21>`,
///   exactly as v1..v7 carry it as `Option<MaterialV7>`). v1..v21 payloads load
///   unchanged, with every material lifted to `asset: None` — a surface whose
///   scalars are the whole story, which is what every pre-v22 level was.
///
///   **What it buys.** P26.1–P26.3 built the whole streaming-virtual-texturing
///   stack — the tiled container, the pool and residency, the WGSL sample — and
///   left it with nothing to sample, because *nothing on disk said which
///   material a surface uses*. Apply-material has flattened a `.inf_mat`'s
///   scalars onto this component since P7.1 and thrown the reference away, so a
///   level referenced textures only in the author's memory. That is the
///   spec-clause-4 gap the P26.3 ledger named, and this field is it.
///
///   **The scalars stay.** The binding *adds* the texture edge; it never becomes
///   the only copy of the numbers. `None` is exactly the pre-v22 behaviour and
///   is the permanent no-texture path, so an unresolvable binding renders as it
///   always did rather than as an error — the fallback is structural, not a
///   runtime branch.
///
///   **The second freeze of one component**, and the reason bincode forces it is
///   the v12/v13/v15/v16 law a fifth time: growing a component is a wire-format
///   change even when the new field is `#[serde(default)]`, because the decoder
///   reads a fixed field count positionally. Here the read would not even run
///   past the end of the record — it would run past the end of the *material*
///   and into the entity's `light` slot.
///
///   The wire cost to a level with no material bindings is **one discriminant
///   byte per materialed entity**, the same price every additive field since v8
///   has paid.
/// * **v23** — P29.3: the movement component. [`CharacterMovement`] is a new
///   entity slot at the record's **tail**, which is the cheap rung of this
///   ladder — the [`EntityRecordV10`] shape of bump rather than the
///   [`EntityRecordV14`] one — so every frozen historical record above is
///   byte-unchanged and only the live record grew. It carries a character's
///   whole tunable set (per-gait speeds, the four curves keyed on normalized
///   speed, air control, capsule heights, step height, the sprint gate, the
///   slide and landing constants), its `MovementMode`, its `RotationMode` and
///   its overlay id.
///
///   The slot is a **new component** rather than fields appended to
///   `CharacterController3D` precisely because of the paragraph above: that
///   struct appears inside eighteen frozen records across two mirrors, and
///   growing it would mean freezing a copy of it into every one of them.
pub const SCHEMA_VERSION: u32 = 23;

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
    /// World-partition / level-streaming configuration (schema v10). Additive:
    /// `#[serde(default)]` → [`PartitionSettings::default`], whose `enabled` is
    /// `false`, so every pre-v10 level keeps cooking and loading as one document.
    #[serde(default)]
    pub partition: PartitionSettings,
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
            partition: PartitionSettings::default(),
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
    // ── v10 (P16.5) world-partition components ────────────────────────────
    /// Marks this entity as a **streaming source**: cell residency is computed
    /// from its position at the fixed-step boundary (never from a camera).
    #[serde(default)]
    pub streaming_source: Option<StreamingSource>,
    /// Marks this entity as never-streamed: it cooks into the partition's
    /// persistent cell and exists for the whole run.
    #[serde(default)]
    pub always_loaded: Option<AlwaysLoaded>,
    // ── v11 (P17.1) sky-authority components ──────────────────────────────
    /// The level's world clock. At most one entity should carry it; the
    /// resolution rule (lowest `Guid` wins) lives in `inf_ecs::sky`.
    #[serde(default)]
    pub time_of_day: Option<TimeOfDay>,
    /// How the clock's sun and moon light the world and tint the sky gradient.
    /// Sits on the same entity as `time_of_day`.
    #[serde(default)]
    pub sky_atmosphere: Option<SkyAtmosphere>,
    // ── v17 (P20.1) water ───────────────────────────────────────────
    /// An ocean, a lake or a spline river. A `River` reads the `spline` slot on
    /// **this same entity** for its centreline — there is no reference to
    /// resolve, and therefore no cook edge and no dangling-reference advisory.
    #[serde(default)]
    pub water_body: Option<WaterBody>,
    // ── v18 (P20.2) buoyancy ────────────────────────────────────────
    /// Opt-in flotation + hydrodynamic drag for a dynamic 3D body. Absent means
    /// the body ignores water, which is what every pre-v18 level meant — see the
    /// [`SCHEMA_VERSION`] ladder for why this is a component rather than a rule
    /// applied to every `RigidBody3D`.
    #[serde(default)]
    pub buoyancy: Option<Buoyancy>,
    // ── v19 (P21.1) volumetric terrain ──────────────────────────────
    /// A sparse SDF voxel volume (caves / tunnels / excavations) that locally
    /// extends the heightfield terrain. The chunks themselves live in the
    /// `.inf_voxel` this points at — the component is the reference plus its two
    /// authored knobs, so a level carries no volumetric data inline.
    #[serde(default)]
    pub voxel_volume: Option<VoxelVolume>,
    // ── v20 (P22.2) destruction ─────────────────────────────────────
    /// This entity's mesh can break: the fracture seed + chunk count the cook
    /// pre-fractures it with, the material strength and density the structural
    /// solve and the chunk bodies read, and the runtime gate. References no
    /// asset — the chunk set is derived from this entity's own `MeshRef`.
    #[serde(default)]
    pub destructible: Option<Destructible>,
    // ── v21 (P24.3) character components ───────────────────────────
    /// The authored IK goals on this character. Re-read from the document every
    /// fixed step by `inf_ecs::pose::step_pose_evaluation`, converted from world
    /// space into the character's own frame there — so a saved foot plant works
    /// identically in the editor's Simulate and in the shipped player, through the
    /// door both already used.
    #[serde(default)]
    pub ik_target: Option<IkTarget>,
    /// A simulated garment (reference + per-wearer knobs). Authored in v21, read
    /// by P24.4 — see the [`SCHEMA_VERSION`] ladder for why the slot is spent now.
    #[serde(default)]
    pub cloth_sim: Option<ClothSim>,
    /// Strand hair (reference + per-wearer knobs). Same story as `cloth_sim`.
    #[serde(default)]
    pub hair_guides: Option<HairGuides>,
    /// **The movement component** (schema v23, P29.3): the character's tunable
    /// set, its rotation mode, its overlay id and the mode it is in. The slot
    /// this version exists for.
    #[serde(default)]
    pub character_movement: Option<CharacterMovement>,
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

    /// Encode to the **current** schema (v23) — a deterministic bincode payload.
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

/// The schema-v23 file layout (current). `entities` reuses [`RuntimeEntity`].
#[derive(Serialize, Deserialize)]
struct SceneFileV23 {
    schema_version: u32,
    title: String,
    entities: Vec<RuntimeEntity>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v22 file layout. v23 did not touch [`RuntimeSettings`], so
/// only `entities` is repointed at the frozen [`EntityRecordV22`] — the shape
/// this file held as [`RuntimeEntity`] until v23 appended the movement slot.
#[derive(Serialize, Deserialize)]
struct SceneFileV22 {
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV22>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v21 file layout. It carried the **live** settings shape (v22
/// did not touch [`RuntimeSettings`]), so only `entities` is repointed at the
/// frozen [`EntityRecordV21`] — the shape this file held as [`RuntimeEntity`]
/// until v22 grew [`Material`] its `asset` binding.
#[derive(Serialize, Deserialize)]
struct SceneFileV21 {
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV21>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v20 file layout. It carried the **live** settings shape (v21
/// did not touch [`RuntimeSettings`]), so only `entities` is repointed at the
/// frozen [`EntityRecordV20`] — the shape this file held as [`RuntimeEntity`]
/// until v21 appended the three character slots to the live record.
#[derive(Serialize, Deserialize)]
struct SceneFileV20 {
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV20>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v19 file layout. It carried the **live** settings shape (v20
/// did not touch [`RuntimeSettings`]), so only `entities` is repointed at the
/// frozen [`EntityRecordV19`] — the shape this file held as [`RuntimeEntity`]
/// until v20 appended the `destructible` slot to the live record.
#[derive(Serialize, Deserialize)]
struct SceneFileV19 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV19>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v18 file layout. It carried the **live** settings shape (v19
/// did not touch [`RuntimeSettings`]), so only `entities` is repointed at the
/// frozen [`EntityRecordV18`] — the shape this file held as [`RuntimeEntity`]
/// until v19 appended the `voxel_volume` slot to the live record.
#[derive(Serialize, Deserialize)]
struct SceneFileV18 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV18>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v17 file layout. It carried the **live** settings shape (v18
/// did not touch [`RuntimeSettings`]), so only `entities` is repointed at the
/// frozen [`EntityRecordV17`].
#[derive(Serialize, Deserialize)]
struct SceneFileV17 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV17>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v16 file layout. It carried the **live** settings shape (v17
/// did not touch [`RuntimeSettings`]), so only `entities` is repointed at the
/// frozen [`EntityRecordV16`].
#[derive(Serialize, Deserialize)]
struct SceneFileV16 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV16>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v15 file layout. It carried the **live** settings shape (v16
/// did not touch [`RuntimeSettings`]), so only `entities` is repointed at the
/// frozen [`EntityRecordV15`].
#[derive(Serialize, Deserialize)]
struct SceneFileV15 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV15>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v14 file layout. It carried the **live** settings shape (v15
/// did not touch [`RuntimeSettings`]), so only `entities` is repointed at the
/// frozen [`EntityRecordV14`].
#[derive(Serialize, Deserialize)]
struct SceneFileV14 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV14>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v13 file layout. It carried the **live** settings shape (v14
/// did not touch [`RuntimeSettings`]), so only `entities` is repointed at the
/// frozen [`EntityRecordV13`].
#[derive(Serialize, Deserialize)]
struct SceneFileV13 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV13>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v12 file layout. It carried the **live** settings shape (v13
/// did not touch [`RuntimeSettings`]), so only `entities` is repointed at the
/// frozen [`EntityRecordV12`].
#[derive(Serialize, Deserialize)]
struct SceneFileV12 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV12>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v11 file layout. It carried the **live** settings shape (v12
/// did not touch [`RuntimeSettings`]), so only `entities` is repointed at the
/// frozen [`EntityRecordV11`].
#[derive(Serialize, Deserialize)]
struct SceneFileV11 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV11>,
    #[serde(default)]
    settings: RuntimeSettings,
}

/// A frozen schema-v10 file layout. It carried the **live** settings shape (v11
/// did not touch [`RuntimeSettings`]), so only `entities` is repointed at the
/// frozen [`EntityRecordV10`].
#[derive(Serialize, Deserialize)]
struct SceneFileV10 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV10>,
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
            asset: None,
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

/// The **pre-v22** `Material` byte layout (schema v22 froze this when P26.3b
/// gave `Material` its `asset` binding). Frozen entity records **v8..v21** carry
/// `material` as `Option<MaterialV21>`; [`MaterialV21::into_current`] lifts it
/// with `asset: None`.
///
/// The second freeze of one component, and the reason is the same one that froze
/// [`MeshRefV6`] and [`TerrainV8`]: bincode is positional, so *growing* a
/// component is a wire-format change even though the new field is
/// `#[serde(default)]` — a v21 payload fed to the grown struct reads past the end
/// of its `Material` and into the next slot of the same record. `MaterialV7`
/// above stays exactly as it was; v1..v7 records still carry it, and they now lift
/// straight to the live shape rather than through this one, because the two
/// freezes describe two different byte layouts and neither is a step toward the
/// other.
#[derive(Clone, Copy, Serialize, Deserialize)]
struct MaterialV21 {
    base_color: Color,
    #[serde(default)]
    metallic: f32,
    #[serde(default)]
    roughness: f32,
    #[serde(default)]
    emissive: Color,
    #[serde(default)]
    blend: BlendMode,
    #[serde(default)]
    alpha_cutoff: f32,
}

impl MaterialV21 {
    /// Lift to the live [`Material`] with **no material binding** — which is
    /// exactly what a v21 level meant: its scalars are the whole surface.
    fn into_current(self) -> Material {
        Material {
            base_color: self.base_color,
            metallic: self.metallic,
            roughness: self.roughness,
            emissive: self.emissive,
            blend: self.blend,
            alpha_cutoff: self.alpha_cutoff,
            asset: None,
        }
    }

    /// Downgrade a live [`Material`] to the pre-v22 layout (the
    /// **downgrade-bless** path that regenerates the committed v21 fixture). Only
    /// `asset` is lost, which is precisely what v22 added.
    #[cfg(test)]
    fn from_current(m: Material) -> Self {
        Self {
            base_color: m.base_color,
            metallic: m.metallic,
            roughness: m.roughness,
            emissive: m.emissive,
            blend: m.blend,
            alpha_cutoff: m.alpha_cutoff,
        }
    }
}

/// The pre-v22 shape of `Material::default()`, **derived from the live default**
/// rather than restated — so a change to the live defaults can never silently
/// leave the frozen fixtures describing a material nobody would author.
#[cfg(test)]
impl Default for MaterialV21 {
    fn default() -> Self {
        Self::from_current(Material::default())
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
            partition: PartitionSettings::default(),
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

/// The **pre-v10** file-settings byte layout (schema v10 froze this when the file
/// settings gained a `partition` block). Frozen file records (v8..v9) carry
/// `settings` as [`RuntimeSettingsV9`]; [`RuntimeSettingsV9::into_current`] lifts
/// it with a default (disabled) [`PartitionSettings`].
#[derive(Clone, Copy, Serialize, Deserialize)]
struct RuntimeSettingsV9 {
    #[serde(default)]
    gravity_2d: Vec2d,
    #[serde(default = "default_gravity_3d")]
    gravity_3d: Vec3d,
    #[serde(default = "default_sim_hz")]
    sim_hz: f64,
    #[serde(default)]
    render: RenderSettingsRecord,
}

impl RuntimeSettingsV9 {
    fn into_current(self) -> RuntimeSettings {
        RuntimeSettings {
            gravity_2d: self.gravity_2d,
            gravity_3d: self.gravity_3d,
            sim_hz: self.sim_hz,
            render: self.render,
            partition: PartitionSettings::default(),
        }
    }

    /// Downgrade live [`RuntimeSettings`] to the pre-v10 layout (fixture bless
    /// path). The partition block has no v9 home and is dropped — a lossy
    /// direction, used only to regenerate old fixtures from a current sample.
    #[cfg(test)]
    fn from_current(s: RuntimeSettings) -> Self {
        Self {
            gravity_2d: s.gravity_2d,
            gravity_3d: s.gravity_3d,
            sim_hz: s.sim_hz,
            render: s.render,
        }
    }
}

impl Default for RuntimeSettingsV9 {
    fn default() -> Self {
        Self {
            gravity_2d: Vec2d::ZERO,
            gravity_3d: default_gravity_3d(),
            sim_hz: default_sim_hz(),
            render: RenderSettingsRecord::default(),
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
            streaming_source: None,
            always_loaded: None,
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
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
            streaming_source: None,
            always_loaded: None,
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
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
            streaming_source: None,
            always_loaded: None,
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
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
            streaming_source: None,
            always_loaded: None,
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
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
            streaming_source: None,
            always_loaded: None,
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
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
            streaming_source: None,
            always_loaded: None,
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
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
    /// The **pre-v15** heightfield shape — see [`TerrainV14`]. A v4..v8 payload's
    /// tiles have no data maps, so the frozen wire type is what reads them.
    #[serde(default)]
    data: inf_terrain::TerrainDataFrozenV1,
    #[serde(default = "inf_ecs::components::default_terrain_layers")]
    layers: [TerrainLayer; inf_ecs::components::TERRAIN_LAYERS],
    #[serde(default = "default_macro_variation")]
    macro_variation: f64,
}

/// The **pre-v15** `Terrain` byte layout (schema v15 froze this when P19.1 gave
/// every terrain tile its sparse erosion **data-map** layer). Frozen entity
/// records v9..v14 carry `terrain` as `Option<TerrainV14>`;
/// [`TerrainV14::into_current`] lifts it.
///
/// Only the heightfield's *tile* layout changed — no field was added to the
/// component and none moved — but bincode is positional, so an extra
/// length-prefixed layer inside each tile is a wire-format change all the same:
/// a v14 payload fed to the grown tile would read past the end of its heights
/// and into the next tile. This is the third instance of that same law (v12 and
/// v13 were the others), and the frozen-record ladder is exactly the machinery
/// for it.
#[derive(Clone, Serialize, Deserialize)]
struct TerrainV14 {
    #[serde(default = "default_terrain_mps")]
    meters_per_sample: f64,
    #[serde(default = "default_terrain_resolution")]
    tile_resolution: u32,
    #[serde(default)]
    data: inf_terrain::TerrainDataFrozenV1,
    #[serde(default = "inf_ecs::components::default_terrain_layers")]
    layers: [TerrainLayer; inf_ecs::components::TERRAIN_LAYERS],
    #[serde(default = "default_macro_variation")]
    macro_variation: f64,
    #[serde(default)]
    asset: Option<Uuid>,
}

impl TerrainV14 {
    /// Lift to the live [`Terrain`]: every tile's data maps come up **empty**
    /// (never eroded) and its biome ids likewise, with no biome vocabulary
    /// (`biome_set: None`) — exactly what a pre-P19.1 level meant.
    fn into_current(self) -> Terrain {
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
    /// path). The data maps have no v14 home and are dropped — as are P19.2's
    /// biome ids and `biome_set` — a lossy direction, used only to regenerate old
    /// fixtures from a current sample.
    #[cfg(test)]
    fn from_current(t: Terrain) -> Self {
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
/// [`TerrainV15::into_current`] lifts it.
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
#[derive(Clone, Serialize, Deserialize)]
struct TerrainV15 {
    #[serde(default = "default_terrain_mps")]
    meters_per_sample: f64,
    #[serde(default = "default_terrain_resolution")]
    tile_resolution: u32,
    /// The **pre-v16** heightfield shape: tiles with erosion data maps but no
    /// biome ids (generation 2 of the frozen-tile ladder).
    #[serde(default)]
    data: inf_terrain::TerrainDataFrozenV2,
    #[serde(default = "inf_ecs::components::default_terrain_layers")]
    layers: [TerrainLayer; inf_ecs::components::TERRAIN_LAYERS],
    #[serde(default = "default_macro_variation")]
    macro_variation: f64,
    #[serde(default)]
    asset: Option<Uuid>,
}

impl TerrainV15 {
    /// Lift to the live [`Terrain`]: every tile's biome ids come up **empty**
    /// (nothing painted, every sample [`inf_terrain::UNASSIGNED_BIOME`]) and
    /// `biome_set` at `None` (no vocabulary to paint from) — exactly what a
    /// pre-P19.2 level meant.
    fn into_current(self) -> Terrain {
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
    /// path). Both P19.2 additions are dropped — the per-tile biome ids and the
    /// `biome_set` reference — a lossy direction, used only to regenerate old
    /// fixtures from a current sample.
    #[cfg(test)]
    fn from_current(t: Terrain) -> Self {
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
    /// `data` remains the terrain's only authority — what a pre-v9 level meant.
    /// `biome_set` likewise (P19.2 did not exist).
    fn into_current(self) -> Terrain {
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

    /// Project a live [`Terrain`] back onto the frozen shape (the downgrade-bless
    /// path). The `asset` reference has no v8 home and is dropped — as are the
    /// P19.1 data maps — a lossy direction, used only to regenerate old fixtures
    /// from a current sample.
    #[cfg(test)]
    fn from_current(t: Terrain) -> Self {
        Self {
            meters_per_sample: t.meters_per_sample,
            tile_resolution: t.tile_resolution,
            data: inf_terrain::TerrainDataFrozenV1::from_current(&t.data),
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
            streaming_source: None,
            always_loaded: None,
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
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
    material: Option<MaterialV21>,
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

/// A frozen schema-v8 file layout. Its `settings` are the **pre-v10** shape
/// ([`RuntimeSettingsV9`]) — v9 did not touch the file settings, so one frozen
/// record serves both v8 and v9 payloads.
#[derive(Serialize, Deserialize)]
struct SceneFileV8 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV8>,
    #[serde(default)]
    settings: RuntimeSettingsV9,
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
            material: self.material.map(MaterialV21::into_current),
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
            streaming_source: None,
            always_loaded: None,
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
        }
    }
}

/// A frozen schema-v9 entity record (pre-P16.5): the full v9 slot set with the
/// live `Terrain` (asset reference included), but **neither** v10
/// world-partition slot. v10 appended only `streaming_source` / `always_loaded`,
/// so this differs from the live [`RuntimeEntity`] only there.
#[derive(Serialize, Deserialize)]
struct EntityRecordV9 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<MaterialV21>,
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
    terrain: Option<TerrainV14>,
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

/// A frozen schema-v9 file layout (carries the pre-v10 [`RuntimeSettingsV9`]).
#[derive(Serialize, Deserialize)]
struct SceneFileV9 {
    #[allow(dead_code)]
    schema_version: u32,
    title: String,
    entities: Vec<EntityRecordV9>,
    #[serde(default)]
    settings: RuntimeSettingsV9,
}

impl EntityRecordV9 {
    /// Lift a frozen v9 record to the live (v10) [`RuntimeEntity`]: both
    /// world-partition slots default to `None` — a pre-v10 level named no
    /// streaming source and marked nothing always-loaded, which is exactly what
    /// an unpartitioned level means.
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material.map(MaterialV21::into_current),
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
            streaming_source: None,
            always_loaded: None,
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
        }
    }
}

/// The **pre-v11** entity byte layout (schema v11 froze this when the record
/// gained its two sky-authority slots). A v10 payload decodes through this and
/// [`EntityRecordV10::into_runtime`] lifts it with both slots `None`.
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV10 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<MaterialV21>,
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
    terrain: Option<TerrainV14>,
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
}

impl EntityRecordV10 {
    /// Lift a frozen v10 record to the live (v11) [`RuntimeEntity`]: both
    /// sky-authority slots default to `None` — a pre-v11 level had no clock, so
    /// the projectors render it with the retired `SUN_DIR` direction, which is
    /// exactly the sun it was authored under.
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material.map(MaterialV21::into_current),
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
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
        }
    }

    /// Project a live [`RuntimeEntity`] back onto the frozen v10 shape (the
    /// downgrade-bless path that regenerates the committed v10 fixture). The two
    /// sky slots have no v10 home and are dropped — the one deliberately lossy
    /// direction, asserted as a property by
    /// `v10_entity_downgrade_is_lossless_except_for_the_sky_slots`.
    #[cfg(test)]
    fn from_current(r: RuntimeEntity) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material.map(MaterialV21::from_current),
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
#[derive(Clone, Copy, Serialize, Deserialize)]
struct SkyAtmosphereV11 {
    #[serde(default = "v11_sky_true")]
    enabled: bool,
    #[serde(default = "v11_sun_intensity")]
    sun_intensity: f32,
    #[serde(default = "v11_sun_color")]
    sun_color: Color,
    #[serde(default = "v11_moon_intensity")]
    moon_intensity: f32,
    #[serde(default = "v11_moon_color")]
    moon_color: Color,
    #[serde(default = "v11_sky_zenith")]
    zenith: Color,
    #[serde(default = "v11_sky_horizon")]
    horizon: Color,
    #[serde(default = "v11_sky_ground")]
    ground: Color,
    #[serde(default = "v11_night_darkening")]
    night_darkening: f32,
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
    /// Lift to the live [`SkyAtmosphere`]: the v11 half carries through verbatim
    /// and the 13 P17.2 fields take their live `SkyAtmosphere::default()` values.
    /// That default *is* what a v11 level meant — `tint_strength: 0` and
    /// `fog_density: 0` reproduce the gradient sky with no height fog, and the
    /// disc/star/aerial knobs are the physical constants the v11 renderer already
    /// used implicitly.
    fn into_current(self) -> SkyAtmosphere {
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
            ..SkyAtmosphere::default()
        }
    }

    /// Project a live [`SkyAtmosphere`] back onto the frozen v11 shape (the
    /// downgrade-bless path). The whole physical-atmosphere block has no v11 home
    /// and is dropped — the deliberately lossy direction, used only to regenerate
    /// an old fixture from a current record.
    #[cfg(test)]
    fn from_current(a: SkyAtmosphere) -> Self {
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
/// [`RuntimeEntity`] except that `sky_atmosphere` is typed as the frozen
/// [`SkyAtmosphereV11`] — v12 added **no** entity slot and moved none, so this is
/// the `TerrainV8` shape of bump, not the `EntityRecordV10` shape.
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV11 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<MaterialV21>,
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
    terrain: Option<TerrainV14>,
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    sky_atmosphere: Option<SkyAtmosphereV11>,
}

impl EntityRecordV11 {
    /// Lift a frozen v11 record to the live (v12) [`RuntimeEntity`]. Every slot
    /// carries through unchanged; only the atmosphere is lifted, through
    /// [`SkyAtmosphereV11::into_current`].
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material.map(MaterialV21::into_current),
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
            sky_atmosphere: self.sky_atmosphere.map(SkyAtmosphereV11::into_current),
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
        }
    }

    /// Project a live [`RuntimeEntity`] back onto the frozen v11 shape (the
    /// downgrade-bless path that regenerates the committed v11 fixture). Only the
    /// physical-atmosphere block is lost — asserted as a property by
    /// `v11_entity_downgrade_is_lossless_except_for_the_physical_atmosphere_block`.
    #[cfg(test)]
    fn from_current(r: RuntimeEntity) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material.map(MaterialV21::from_current),
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
#[derive(Clone, Copy, Serialize, Deserialize)]
struct SkyAtmosphereV12 {
    #[serde(default = "v12_sky_true")]
    enabled: bool,
    #[serde(default = "v12_sun_intensity")]
    sun_intensity: f32,
    #[serde(default = "v12_sun_color")]
    sun_color: Color,
    #[serde(default = "v12_moon_intensity")]
    moon_intensity: f32,
    #[serde(default = "v12_moon_color")]
    moon_color: Color,
    #[serde(default = "v12_sky_zenith")]
    zenith: Color,
    #[serde(default = "v12_sky_horizon")]
    horizon: Color,
    #[serde(default = "v12_sky_ground")]
    ground: Color,
    #[serde(default = "v12_night_darkening")]
    night_darkening: f32,
    #[serde(default = "v12_sky_true")]
    physical: bool,
    #[serde(default = "v12_one")]
    sky_intensity: f32,
    #[serde(default = "v12_one")]
    turbidity: f32,
    #[serde(default = "v12_mie_anisotropy")]
    mie_anisotropy: f32,
    #[serde(default = "v12_sun_disc_deg")]
    sun_disc_deg: f32,
    #[serde(default = "v12_moon_disc_deg")]
    moon_disc_deg: f32,
    #[serde(default = "v12_one")]
    star_intensity: f32,
    #[serde(default)]
    tint_strength: f32,
    #[serde(default = "v12_one")]
    aerial_perspective: f32,
    #[serde(default)]
    fog_density: f32,
    #[serde(default = "v12_fog_falloff")]
    fog_falloff: f32,
    #[serde(default)]
    fog_height: f32,
    #[serde(default = "v12_fog_color")]
    fog_color: Color,
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
    /// Lift to the live [`SkyAtmosphere`]: the 22 v12 fields carry through verbatim
    /// and the 14 P17.3 fields take their live `SkyAtmosphere::default()` values.
    /// That default *is* what a v12 level meant — `clouds_enabled: false` is a
    /// cloudless sky, and the rest of the block is inert while it is off.
    fn into_current(self) -> SkyAtmosphere {
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
            ..SkyAtmosphere::default()
        }
    }

    /// Project a live [`SkyAtmosphere`] back onto the frozen v12 shape (the
    /// downgrade-bless path). The whole volumetric-cloud block has no v12 home and
    /// is dropped — the deliberately lossy direction, used only to regenerate an
    /// old fixture from a current record.
    #[cfg(test)]
    fn from_current(a: SkyAtmosphere) -> Self {
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
/// [`RuntimeEntity`] except that `sky_atmosphere` is typed as the frozen
/// [`SkyAtmosphereV12`] — v13 added **no** entity slot and moved none, so this is
/// the `TerrainV8` shape of bump, not the `EntityRecordV10` shape.
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV12 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<MaterialV21>,
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
    terrain: Option<TerrainV14>,
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    time_of_day: Option<TimeOfDay>,
    /// The **pre-v13** atmosphere shape — the one field that makes this record
    /// differ from the live [`RuntimeEntity`].
    #[serde(default)]
    sky_atmosphere: Option<SkyAtmosphereV12>,
}

impl EntityRecordV12 {
    /// Lift a frozen v12 record to the live (v13) [`RuntimeEntity`]. Every slot
    /// carries through unchanged; only the atmosphere is lifted, through
    /// [`SkyAtmosphereV12::into_current`].
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material.map(MaterialV21::into_current),
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
            sky_atmosphere: self.sky_atmosphere.map(SkyAtmosphereV12::into_current),
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
        }
    }

    /// Project a live [`RuntimeEntity`] back onto the frozen v12 shape (the
    /// downgrade-bless path that regenerates the committed v12 fixture). Only the
    /// volumetric-cloud block is lost — asserted as a property by
    /// `v12_entity_downgrade_is_lossless_except_for_the_cloud_block`.
    #[cfg(test)]
    fn from_current(r: RuntimeEntity) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material.map(MaterialV21::from_current),
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
/// move when the live component's defaults are re-tuned. bincode ignores defaults
/// on the write side, but keeping them identical means this record decodes every
/// partial payload the live one did, in the human-readable codecs too.
#[derive(Clone, Copy, Serialize, Deserialize)]
struct SkyAtmosphereV13 {
    #[serde(default = "v13_sky_true")]
    enabled: bool,
    #[serde(default = "v13_sun_intensity")]
    sun_intensity: f32,
    #[serde(default = "v13_sun_color")]
    sun_color: Color,
    #[serde(default = "v13_moon_intensity")]
    moon_intensity: f32,
    #[serde(default = "v13_moon_color")]
    moon_color: Color,
    #[serde(default = "v13_sky_zenith")]
    zenith: Color,
    #[serde(default = "v13_sky_horizon")]
    horizon: Color,
    #[serde(default = "v13_sky_ground")]
    ground: Color,
    #[serde(default = "v13_night_darkening")]
    night_darkening: f32,
    #[serde(default = "v13_sky_true")]
    physical: bool,
    #[serde(default = "v13_one")]
    sky_intensity: f32,
    #[serde(default = "v13_one")]
    turbidity: f32,
    #[serde(default = "v13_mie_anisotropy")]
    mie_anisotropy: f32,
    #[serde(default = "v13_sun_disc_deg")]
    sun_disc_deg: f32,
    #[serde(default = "v13_moon_disc_deg")]
    moon_disc_deg: f32,
    #[serde(default = "v13_one")]
    star_intensity: f32,
    #[serde(default)]
    tint_strength: f32,
    #[serde(default = "v13_one")]
    aerial_perspective: f32,
    #[serde(default)]
    fog_density: f32,
    #[serde(default = "v13_fog_falloff")]
    fog_falloff: f32,
    #[serde(default)]
    fog_height: f32,
    #[serde(default = "v13_fog_color")]
    fog_color: Color,
    #[serde(default)]
    clouds_enabled: bool,
    #[serde(default = "v13_cloud_coverage")]
    cloud_coverage: f32,
    #[serde(default = "v13_cloud_type")]
    cloud_type: f32,
    #[serde(default = "v13_cloud_bottom")]
    cloud_bottom: f32,
    #[serde(default = "v13_cloud_top")]
    cloud_top: f32,
    #[serde(default = "v13_cloud_density")]
    cloud_density: f32,
    #[serde(default = "v13_cloud_detail")]
    cloud_detail: f32,
    #[serde(default)]
    cloud_seed: u32,
    #[serde(default = "v13_cloud_wind_x")]
    cloud_wind_x: f32,
    #[serde(default = "v13_cloud_wind_z")]
    cloud_wind_z: f32,
    #[serde(default = "v13_cloud_phase_g")]
    cloud_phase_g: f32,
    #[serde(default = "v13_one")]
    cloud_shadow: f32,
    #[serde(default = "v13_one")]
    cloud_ambient: f32,
    #[serde(default = "v13_cloud_color")]
    cloud_color: Color,
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
    /// driving the sky exactly as they did, which is the byte-stability promise.
    fn into_current(self) -> SkyAtmosphere {
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
    #[cfg(test)]
    fn from_current(a: SkyAtmosphere) -> Self {
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
/// [`RuntimeEntity`] except that `sky_atmosphere` is typed as the frozen
/// [`SkyAtmosphereV13`] — v14 added **no** entity slot and moved none, so this is
/// the `TerrainV8` shape of bump, not the `EntityRecordV10` shape.
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV13 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<MaterialV21>,
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
    terrain: Option<TerrainV14>,
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    time_of_day: Option<TimeOfDay>,
    /// The **pre-v14** atmosphere shape — the one field that makes this record
    /// differ from the live [`RuntimeEntity`].
    #[serde(default)]
    sky_atmosphere: Option<SkyAtmosphereV13>,
}

impl EntityRecordV13 {
    /// Lift a frozen v13 record to the live (v14) [`RuntimeEntity`]. Every slot
    /// carries through unchanged; only the atmosphere is lifted, through
    /// [`SkyAtmosphereV13::into_current`].
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material.map(MaterialV21::into_current),
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
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
        }
    }

    /// Project a live [`RuntimeEntity`] back onto the frozen v13 shape (the
    /// downgrade-bless path that regenerates the committed v13 fixture). Only the
    /// weather block is lost — asserted as a property by
    /// `v13_entity_downgrade_is_lossless_except_for_the_weather_block`.
    #[cfg(test)]
    fn from_current(r: RuntimeEntity) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material.map(MaterialV21::from_current),
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
/// [`RuntimeEntity`] except that `terrain` is typed as the frozen
/// [`TerrainV14`] — v15 added **no** entity slot and moved none, so this is the
/// `EntityRecordV13` shape of bump, not the `EntityRecordV10` shape.
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV14 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<MaterialV21>,
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
    terrain: Option<TerrainV14>,
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    sky_atmosphere: Option<SkyAtmosphere>,
}

impl EntityRecordV14 {
    /// Lift a frozen v14 record to the live (v15) [`RuntimeEntity`]. Every slot
    /// carries through unchanged; only the terrain is lifted, through
    /// [`TerrainV14::into_current`].
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material.map(MaterialV21::into_current),
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
            sky_atmosphere: self.sky_atmosphere,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
        }
    }

    /// Project a live [`RuntimeEntity`] back onto the frozen v14 shape (the
    /// downgrade-bless path that regenerates the committed v14 fixture). Only the
    /// erosion data maps are lost — asserted as a property by
    /// `v14_entity_downgrade_is_lossless_except_for_the_data_maps`.
    #[cfg(test)]
    fn from_current(r: RuntimeEntity) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material.map(MaterialV21::from_current),
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

/// The **pre-v16** entity byte layout (schema v16 froze this when P19.2 gave
/// every terrain tile its sparse per-sample biome id layer, and [`Terrain`] its
/// `biome_set` reference). Identical to the live [`RuntimeEntity`] except that
/// `terrain` is typed as the frozen [`TerrainV15`] — v16 added **no** entity slot
/// and moved none, so this is the `EntityRecordV14` shape of bump, not the
/// `EntityRecordV10` shape.
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV15 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<MaterialV21>,
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
    terrain: Option<TerrainV15>,
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    sky_atmosphere: Option<SkyAtmosphere>,
}

impl EntityRecordV15 {
    /// Lift a frozen v15 record to the live (v16) [`RuntimeEntity`]. Every slot
    /// carries through unchanged; only the terrain is lifted, through
    /// [`TerrainV15::into_current`].
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material.map(MaterialV21::into_current),
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
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
        }
    }

    /// Project a live [`RuntimeEntity`] back onto the frozen v15 shape (the
    /// downgrade-bless path that regenerates the committed v15 fixture). Only the
    /// per-tile biome ids and the `biome_set` reference are lost — asserted as a
    /// property by the editor codec's
    /// `v15_entity_downgrade_is_lossless_except_for_the_biome_ids`.
    #[cfg(test)]
    fn from_current(r: RuntimeEntity) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material.map(MaterialV21::from_current),
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
/// the `water_body` slot). Identical to the live [`RuntimeEntity`] except that it
/// has **no** `water_body` field \u2014 that is precisely what v17 added \u2014 so this is
/// the [`EntityRecordV10`] shape of bump (a new slot at the tail), not the
/// [`EntityRecordV14`] one (a component that grew).
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV16 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<MaterialV21>,
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    sky_atmosphere: Option<SkyAtmosphere>,
}

impl EntityRecordV16 {
    /// Lift a frozen v16 record to the live (v17) [`RuntimeEntity`]. Every slot
    /// carries through unchanged; the one new slot lifts to `None` \u2014 a level with
    /// no water, which is what a v16 level was.
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material.map(MaterialV21::into_current),
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
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
        }
    }

    /// Project a live [`RuntimeEntity`] back onto the frozen v16 shape (the
    /// downgrade-bless path that regenerates the committed v16 fixture). Only the
    /// water body is lost \u2014 asserted as a property by the editor codec's
    /// `v16_entity_downgrade_is_lossless_except_for_the_water_body`.
    #[cfg(test)]
    fn from_current(r: RuntimeEntity) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material.map(MaterialV21::from_current),
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
/// the `buoyancy` slot). Identical to the live [`RuntimeEntity`] except that it
/// has **no** `buoyancy` field — that is precisely what v18 added — so this is
/// the [`EntityRecordV10`] shape of bump (a new slot at the tail), not the
/// [`EntityRecordV14`] one (a component that grew).
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV17 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<MaterialV21>,
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    sky_atmosphere: Option<SkyAtmosphere>,
    /// The v17 slot this record exists to keep carrying — a v17 level's water
    /// must survive the v18 hop, not merely decode.
    #[serde(default)]
    water_body: Option<WaterBody>,
}

impl EntityRecordV17 {
    /// Lift a frozen v17 record straight to the live [`RuntimeEntity`] (this
    /// codec lifts each frozen record directly rather than rung by rung). Every
    /// slot carries through unchanged; every slot appended *since* v17 lifts to
    /// `None` — v18's buoyancy (a level in which nothing floats) and v19's voxel
    /// volume (a level whose ground is a heightfield and nothing else), which is
    /// exactly what a v17 level was.
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material.map(MaterialV21::into_current),
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
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
        }
    }

    /// Project a live [`RuntimeEntity`] back onto the frozen v17 shape (the
    /// downgrade-bless path that regenerates the committed v17 fixture). Only the
    /// buoyancy is lost — asserted as a property by the editor codec's
    /// `v17_entity_downgrade_is_lossless_except_for_the_buoyancy`.
    #[cfg(test)]
    fn from_current(r: RuntimeEntity) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material.map(MaterialV21::from_current),
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
/// the `voxel_volume` slot). Identical to the live [`RuntimeEntity`] except that
/// it has **no** `voxel_volume` field — that is precisely what v19 added — so this
/// is the [`EntityRecordV10`] shape of bump (a new slot at the tail), not the
/// [`EntityRecordV14`] one (a component that grew).
///
/// This is the shape [`SceneFileV18`] used to hold as the live record, which is
/// why the v18 decode arm now reads *this* type and lifts it: the bytes a v18
/// player wrote have one fewer discriminant per entity than today's, and nothing
/// but a frozen record can keep them decoding positionally forever.
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV18 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<MaterialV21>,
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    sky_atmosphere: Option<SkyAtmosphere>,
    /// The v17 slot this record still carries — a v18 level's water must survive
    /// the v19 hop, not merely decode.
    #[serde(default)]
    water_body: Option<WaterBody>,
    /// The v18 slot this record exists to keep carrying — a v18 level's buoyancy
    /// must survive the v19 hop, not merely decode.
    #[serde(default)]
    buoyancy: Option<Buoyancy>,
}

impl EntityRecordV18 {
    /// Lift a frozen v18 record one rung, to the frozen [`EntityRecordV19`].
    /// Every slot carries through unchanged; the one new slot lifts to `None` —
    /// a level whose ground is a heightfield and nothing else, which is what a
    /// v18 level was.
    fn into_v19(self) -> EntityRecordV19 {
        EntityRecordV19 {
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

    /// Project a live [`RuntimeEntity`] back onto the frozen v18 shape (the
    /// downgrade-bless path that regenerates the committed v18 fixture). Only the
    /// voxel volume is lost — asserted as a property by the editor codec's
    /// `v18_entity_downgrade_is_lossless_except_for_the_voxel_volume`.
    #[cfg(test)]
    fn from_current(r: RuntimeEntity) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material.map(MaterialV21::from_current),
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

/// The **pre-v20** entity byte layout (schema v20 froze this when P22.2 appended
/// the `destructible` slot). Identical to the live [`RuntimeEntity`] except that
/// it has **no** `destructible` field — that is precisely what v20 added — so
/// this is the [`EntityRecordV10`] shape of bump (a new slot at the tail), not
/// the [`EntityRecordV14`] one (a component that grew).
///
/// This is the shape [`SceneFileV19`] used to hold as the live record, which is
/// why the v19 decode arm now reads *this* type and lifts it: the bytes a v19
/// player wrote have one fewer discriminant per entity than today's, and nothing
/// but a frozen record can keep them decoding positionally forever.
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV19 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<MaterialV21>,
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    sky_atmosphere: Option<SkyAtmosphere>,
    /// The v17 slot this record still carries — a v18 level's water must survive
    /// the v19 hop, not merely decode.
    #[serde(default)]
    water_body: Option<WaterBody>,
    /// The v18 slot this record exists to keep carrying — a v18 level's buoyancy
    /// must survive the v19 hop, not merely decode.
    #[serde(default)]
    buoyancy: Option<Buoyancy>,
    /// The v19 slot this record exists to keep carrying — a v19 level's caves
    /// must survive the v20 hop, not merely decode.
    #[serde(default)]
    voxel_volume: Option<VoxelVolume>,
}

impl EntityRecordV19 {
    /// Lift a frozen v19 record one rung, to the frozen [`EntityRecordV20`].
    ///
    /// It used to lift straight to the live record; v21 inserted
    /// [`EntityRecordV20`] between this record and [`RuntimeEntity`], and a lift
    /// that skipped the new rung would have to be re-audited on every future
    /// bump. One hop per rung, exactly like [`EntityRecordV18::into_v19`].
    fn into_v20(self) -> EntityRecordV20 {
        EntityRecordV20 {
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
            voxel_volume: self.voxel_volume,
            destructible: None,
        }
    }

    /// Project a live [`RuntimeEntity`] back onto the frozen v19 shape (the
    /// downgrade-bless path that regenerates the committed v19 fixture). Only
    /// the destructible is lost — asserted as a property by the editor codec's
    /// `v19_entity_downgrade_is_lossless_except_for_the_destructible`.
    #[cfg(test)]
    fn from_current(r: RuntimeEntity) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material.map(MaterialV21::from_current),
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
            voxel_volume: r.voxel_volume,
        }
    }
}

/// The **pre-v21** entity byte layout (schema v21 froze this when P24.3 appended
/// the three character slots). Identical to the live [`RuntimeEntity`] except that it
/// has **no** `ik_target` / `cloth_sim` / `hair_guides` fields — that is
/// precisely what v21 added — so this is the [`EntityRecordV10`] shape of bump (
/// new slots at the tail), not the [`EntityRecordV14`] one (a component that
/// grew).
///
/// This is the shape [`SceneFileV20`] used to hold as the live record, which is
/// why the v20 decode arm now reads *this* type and lifts it: the bytes a v20
/// player wrote have three fewer discriminants per entity than today's, and
/// nothing but a frozen record can keep them decoding positionally forever.
///
/// **Generic in its material slot only** (P26.3b), and the alias below is the
/// frozen record — `EntityRecordV20Gen<MaterialV21>` is byte-for-byte what this
/// declaration always was. The parameter exists so the v22 wire pin can
/// re-declare the ONE component v22 grew, independently of `inf_ecs::Material`,
/// and still compose the other 43 fields from a frozen list rather than
/// restating them. Nothing outside `#[cfg(test)]` ever instantiates it at
/// anything but [`MaterialV21`].
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV20Gen<M> {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    material: Option<M>,
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    sky_atmosphere: Option<SkyAtmosphere>,
    /// The v17 slot this record still carries — a v20 level's water must
    /// survive the v21 hop, not merely decode.
    #[serde(default)]
    water_body: Option<WaterBody>,
    #[serde(default)]
    buoyancy: Option<Buoyancy>,
    /// The v19 slot this record still carries — a v20 level's caves must
    /// survive the v21 hop, not merely decode.
    #[serde(default)]
    voxel_volume: Option<VoxelVolume>,
    /// The v20 slot this record exists to keep carrying — a v20 level's
    /// breakable walls must survive the v21 hop, not merely decode.
    #[serde(default)]
    destructible: Option<Destructible>,
}

/// **The frozen v20 record.** The generic above exists for the wire pin alone;
/// this is the type the ladder decodes, and its bytes are unchanged.
type EntityRecordV20 = EntityRecordV20Gen<MaterialV21>;

impl EntityRecordV20 {
    /// Lift a frozen v20 record to the live (v21) [`RuntimeEntity`]. Every slot carries
    /// through unchanged; the three new slots lift to `None` — a level with no
    /// IK, no garments and no hair, which is what a v20 level was.
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material.map(MaterialV21::into_current),
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
            voxel_volume: self.voxel_volume,
            destructible: self.destructible,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
            character_movement: None,
        }
    }

    /// Project a live [`RuntimeEntity`] back onto the frozen v20 shape (the
    /// **downgrade-bless** path that regenerates the committed v20 fixture).
    /// Only the three character slots are lost — asserted as a property, not as a
    /// field list, by `v20_entity_downgrade_is_lossless_except_for_the_character_slots`.
    #[cfg(test)]
    fn from_current(r: RuntimeEntity) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material.map(MaterialV21::from_current),
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
            voxel_volume: r.voxel_volume,
            destructible: r.destructible,
        }
    }
}

/// A frozen schema-v22 entity record: every slot the live record had
/// **before** P29.3 appended [`CharacterMovement`]. v23 is a pure tail
/// append, so this differs from the live record by exactly one field and no
/// component inside it changed shape — which is what makes the whole
/// historical ladder byte-unchanged by this bump.
///
/// This is the shape [`RuntimeEntity`] used to hold as the live record.
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV22 {
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    sky_atmosphere: Option<SkyAtmosphere>,
    #[serde(default)]
    water_body: Option<WaterBody>,
    #[serde(default)]
    buoyancy: Option<Buoyancy>,
    #[serde(default)]
    voxel_volume: Option<VoxelVolume>,
    #[serde(default)]
    destructible: Option<Destructible>,
    #[serde(default)]
    ik_target: Option<IkTarget>,
    #[serde(default)]
    cloth_sim: Option<ClothSim>,
    #[serde(default)]
    hair_guides: Option<HairGuides>,
}

impl EntityRecordV22 {
    /// Lift a frozen v22 record to the live (v23) [`RuntimeEntity`]. Every slot
    /// carries through unchanged and the new one arrives empty: a pre-v23
    /// level has no authored movement component, which is exactly what
    /// `None` means here.
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
            voxel_volume: self.voxel_volume,
            destructible: self.destructible,
            ik_target: self.ik_target,
            cloth_sim: self.cloth_sim,
            hair_guides: self.hair_guides,
            character_movement: None,
        }
    }

    /// Project a live [`RuntimeEntity`] back onto the frozen v22 shape (the
    /// **downgrade-bless** path that regenerates the committed v22 fixture).
    /// Only the movement component is lost, and it is asserted as a property
    /// rather than as a field list by the downgrade arm.
    #[cfg_attr(not(test), allow(dead_code))]
    fn from_current(r: RuntimeEntity) -> Self {
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
            voxel_volume: r.voxel_volume,
            destructible: r.destructible,
            ik_target: r.ik_target,
            cloth_sim: r.cloth_sim,
            hair_guides: r.hair_guides,
        }
    }
}

/// The **pre-v22** entity byte layout (schema v22 froze this when P26.3b grew
/// [`Material`] its `asset` binding). Field-for-field the live [`RuntimeEntity`]
/// except that `material` is an [`Option<MaterialV21>`](MaterialV21) — this is
/// the [`EntityRecordV14`] *shape* of bump (a component that grew), not the
/// [`EntityRecordV10`] one (a new slot at the tail): **no entity slot was added
/// or moved**, and the file settings are untouched.
///
/// This is the shape [`SceneFileV21`] used to hold as the live record.
#[derive(Clone, Serialize, Deserialize)]
struct EntityRecordV21 {
    guid: Uuid,
    name: String,
    parent: Option<Uuid>,
    transform: Transform,
    visible: bool,
    mesh: Option<MeshRef>,
    /// The one field that differs from the live record — and the whole of v22.
    material: Option<MaterialV21>,
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
    #[serde(default)]
    streaming_source: Option<StreamingSource>,
    #[serde(default)]
    always_loaded: Option<AlwaysLoaded>,
    #[serde(default)]
    time_of_day: Option<TimeOfDay>,
    #[serde(default)]
    sky_atmosphere: Option<SkyAtmosphere>,
    #[serde(default)]
    water_body: Option<WaterBody>,
    #[serde(default)]
    buoyancy: Option<Buoyancy>,
    #[serde(default)]
    voxel_volume: Option<VoxelVolume>,
    #[serde(default)]
    destructible: Option<Destructible>,
    /// The three v21 slots this record exists to keep carrying — a v21 level's
    /// IK goals, garments and hair must survive the v22 hop, not merely decode.
    #[serde(default)]
    ik_target: Option<IkTarget>,
    #[serde(default)]
    cloth_sim: Option<ClothSim>,
    #[serde(default)]
    hair_guides: Option<HairGuides>,
}

impl EntityRecordV21 {
    /// Lift a frozen v21 record to the live (v22) [`RuntimeEntity`]. Every slot
    /// carries through unchanged; the material lifts with `asset: None` — a
    /// surface whose scalars are the whole story, which is what every pre-v22
    /// level was.
    fn into_runtime(self) -> RuntimeEntity {
        RuntimeEntity {
            guid: self.guid,
            name: self.name,
            parent: self.parent,
            transform: self.transform,
            visible: self.visible,
            mesh: self.mesh,
            material: self.material.map(MaterialV21::into_current),
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
            voxel_volume: self.voxel_volume,
            destructible: self.destructible,
            ik_target: self.ik_target,
            cloth_sim: self.cloth_sim,
            hair_guides: self.hair_guides,
            character_movement: None,
        }
    }

    /// Project a live [`RuntimeEntity`] back onto the frozen v21 shape (the
    /// **downgrade-bless** path that regenerates the committed v21 fixture).
    /// Only `Material::asset` is lost — asserted as a property, not as a field
    /// list, by `v21_entity_downgrade_is_lossless_except_for_the_material_binding`.
    #[cfg(test)]
    fn from_current(r: RuntimeEntity) -> Self {
        Self {
            guid: r.guid,
            name: r.name,
            parent: r.parent,
            transform: r.transform,
            visible: r.visible,
            mesh: r.mesh,
            material: r.material.map(MaterialV21::from_current),
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
            voxel_volume: r.voxel_volume,
            destructible: r.destructible,
            ik_target: r.ik_target,
            cloth_sim: r.cloth_sim,
            hair_guides: r.hair_guides,
        }
    }
}

/// Decode a `.inf_lvl` payload, lifting older schemas to [`RuntimeLevel`].
pub fn decode(bytes: &[u8]) -> Result<RuntimeLevel> {
    let (header, _): (Header, usize) = bincode::serde::decode_from_slice(bytes, bincode_config())
        .map_err(|e| SceneError::Decode(format!("header: {e}")))?;
    match header.schema_version {
        // **`1`, not `0 | 1`** (L5.F4) — the Ring-0 half of the same fix; see
        // the editor mirror's `decode` for the argument. Three zero bytes are a
        // structurally valid `SceneFileV1` under bincode's varints, so an
        // all-zero buffer used to load as a valid empty level rather than being
        // refused. No writer has ever emitted version 0, so no committed file
        // loses anything. Wire bytes unchanged.
        1 => {
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
                settings: v8.settings.into_current(),
            })
        }
        9 => {
            let (v9, _): (SceneFileV9, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v9: {e}")))?;
            Ok(RuntimeLevel {
                title: v9.title,
                entities: v9
                    .entities
                    .into_iter()
                    .map(EntityRecordV9::into_runtime)
                    .collect(),
                settings: v9.settings.into_current(),
            })
        }
        10 => {
            let (v10, _): (SceneFileV10, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v10: {e}")))?;
            Ok(RuntimeLevel {
                title: v10.title,
                entities: v10
                    .entities
                    .into_iter()
                    .map(EntityRecordV10::into_runtime)
                    .collect(),
                settings: v10.settings,
            })
        }
        11 => {
            let (v11, _): (SceneFileV11, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v11: {e}")))?;
            Ok(RuntimeLevel {
                title: v11.title,
                entities: v11
                    .entities
                    .into_iter()
                    .map(EntityRecordV11::into_runtime)
                    .collect(),
                settings: v11.settings,
            })
        }
        12 => {
            let (v12, _): (SceneFileV12, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v12: {e}")))?;
            Ok(RuntimeLevel {
                title: v12.title,
                entities: v12
                    .entities
                    .into_iter()
                    .map(EntityRecordV12::into_runtime)
                    .collect(),
                settings: v12.settings,
            })
        }
        13 => {
            let (v13, _): (SceneFileV13, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v13: {e}")))?;
            Ok(RuntimeLevel {
                title: v13.title,
                entities: v13
                    .entities
                    .into_iter()
                    .map(EntityRecordV13::into_runtime)
                    .collect(),
                settings: v13.settings,
            })
        }
        14 => {
            let (v14, _): (SceneFileV14, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v14: {e}")))?;
            Ok(RuntimeLevel {
                title: v14.title,
                entities: v14
                    .entities
                    .into_iter()
                    .map(EntityRecordV14::into_runtime)
                    .collect(),
                settings: v14.settings,
            })
        }
        15 => {
            let (v15, _): (SceneFileV15, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v15: {e}")))?;
            Ok(RuntimeLevel {
                title: v15.title,
                entities: v15
                    .entities
                    .into_iter()
                    .map(EntityRecordV15::into_runtime)
                    .collect(),
                settings: v15.settings,
            })
        }
        16 => {
            let (v16, _): (SceneFileV16, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v16: {e}")))?;
            Ok(RuntimeLevel {
                title: v16.title,
                entities: v16
                    .entities
                    .into_iter()
                    .map(EntityRecordV16::into_runtime)
                    .collect(),
                settings: v16.settings,
            })
        }
        17 => {
            let (v17, _): (SceneFileV17, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v17: {e}")))?;
            Ok(RuntimeLevel {
                title: v17.title,
                entities: v17
                    .entities
                    .into_iter()
                    .map(EntityRecordV17::into_runtime)
                    .collect(),
                settings: v17.settings,
            })
        }
        18 => {
            let (v18, _): (SceneFileV18, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v18: {e}")))?;
            Ok(RuntimeLevel {
                title: v18.title,
                entities: v18
                    .entities
                    .into_iter()
                    .map(EntityRecordV18::into_v19)
                    .map(EntityRecordV19::into_v20)
                    .map(EntityRecordV20::into_runtime)
                    .collect(),
                settings: v18.settings,
            })
        }
        19 => {
            let (v19, _): (SceneFileV19, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v19: {e}")))?;
            Ok(RuntimeLevel {
                title: v19.title,
                entities: v19
                    .entities
                    .into_iter()
                    .map(EntityRecordV19::into_v20)
                    .map(EntityRecordV20::into_runtime)
                    .collect(),
                settings: v19.settings,
            })
        }
        20 => {
            let (v20, _): (SceneFileV20, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v20: {e}")))?;
            Ok(RuntimeLevel {
                title: v20.title,
                entities: v20
                    .entities
                    .into_iter()
                    .map(EntityRecordV20::into_runtime)
                    .collect(),
                settings: v20.settings,
            })
        }
        21 => {
            let (v21, _): (SceneFileV21, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v21: {e}")))?;
            Ok(RuntimeLevel {
                title: v21.title,
                entities: v21
                    .entities
                    .into_iter()
                    .map(EntityRecordV21::into_runtime)
                    .collect(),
                settings: v21.settings,
            })
        }
        22 => {
            let (v22, _): (SceneFileV22, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v22: {e}")))?;
            Ok(RuntimeLevel {
                title: v22.title,
                entities: v22
                    .entities
                    .into_iter()
                    .map(EntityRecordV22::into_runtime)
                    .collect(),
                settings: v22.settings,
            })
        }
        23 => {
            let (v23, _): (SceneFileV23, usize) =
                bincode::serde::decode_from_slice(bytes, bincode_config())
                    .map_err(|e| SceneError::Decode(format!("v23: {e}")))?;
            Ok(RuntimeLevel {
                title: v23.title,
                entities: v23.entities,
                settings: v23.settings,
            })
        }
        // A refusal that names the wrong cause sends the user to the wrong fix
        // (L5.F4). Version 0 is not a level from a newer build; it is not a
        // level.
        0 => Err(SceneError::Decode(
            "schema version 0, which no build has ever written (a zero-filled or truncated \
             file reads this way)"
                .into(),
        )),
        found => Err(SceneError::SchemaTooNew {
            found,
            current: SCHEMA_VERSION,
        }),
    }
}

/// Encode a level to the current schema (v23) as a deterministic bincode payload.
pub fn encode(level: &RuntimeLevel) -> Result<Vec<u8>> {
    let file = SceneFileV23 {
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

    // P29.3: the movement component's own types. Reached through `inf_ecs`
    // directly rather than re-exported from this crate's prelude, because the
    // wire pin below re-declares the component and needs its field types by
    // name, not the component itself.
    use inf_ecs::components::{Gait, MovementMode, MovementRuntime, RotationMode, SpeedCurve};

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
        // The committed platformer is an **editor-encoded** current-schema level.
        // This is the editor→runtime cross-decode: the Ring-0 reader parses the
        // editor's bytes field-for-field, and re-encoding is byte-identical (the
        // cook's runtime rewrite of an already-current level is a no-op).
        let original = read_committed("samples/platformer-2d/Platformer.inf_lvl");
        assert_eq!(
            original[0], SCHEMA_VERSION as u8,
            "committed platformer is the current schema"
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

    /// The **v6 rung** of the reference scene (L5.F6) — the same entities as
    /// [`v7_scene_reference`], written through the frozen v6 record.
    ///
    /// v6 → v7 changed exactly one thing: `MeshRef` gained its `asset` field, so
    /// the v6 record carries [`MeshRefV6`] (primitive only). That is a
    /// *lossless* downgrade of this scene, whose meshes are all primitives —
    /// which is why the fixture can be derived rather than invented, and why the
    /// derivation is stated here rather than left for a reader to reconstruct.
    fn v6_scene_reference() -> SceneFileV6 {
        let v7 = v7_scene_reference();
        SceneFileV6 {
            schema_version: 6,
            // Its own title, so a fixture that ends up in the wrong slot says so
            // rather than passing as its neighbour.
            title: "V6 Fixture Level".into(),
            entities: v7
                .entities
                .into_iter()
                .map(|e| EntityRecordV6 {
                    guid: e.guid,
                    name: e.name,
                    parent: e.parent,
                    transform: e.transform,
                    visible: e.visible,
                    mesh: e.mesh.map(MeshRefV6::from_current),
                    material: e.material,
                    light: e.light,
                    camera: e.camera,
                    sprite: e.sprite,
                    tilemap: e.tilemap,
                    nine_slice: e.nine_slice,
                    text2d: e.text2d,
                    light_2d: e.light_2d,
                    rigid_body_2d: e.rigid_body_2d,
                    collider_2d: e.collider_2d,
                    character_controller_2d: e.character_controller_2d,
                    rigid_body_3d: e.rigid_body_3d,
                    collider_3d: e.collider_3d,
                    character_controller_3d: e.character_controller_3d,
                    actor: e.actor,
                    terrain: e.terrain,
                    pcg_volume: e.pcg_volume,
                    skeletal_mesh: e.skeletal_mesh,
                    anim_player: e.anim_player,
                    anim_state_machine: e.anim_state_machine,
                    root_motion: e.root_motion,
                    attached_to: e.attached_to,
                    joint_2d: e.joint_2d,
                    joint_3d: e.joint_3d,
                    audio_source: e.audio_source,
                    audio_listener: e.audio_listener,
                })
                .collect(),
            settings: v7.settings,
        }
    }

    /// Bless the committed `scene_v6.inf_lvl` (L5.F6). See
    /// [`bless_scene_v7_fixture`] for the discipline.
    #[test]
    fn bless_scene_v6_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v6_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 6);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v6.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed scene_v6 fixture: {}", path.display());
    }

    /// **v6 was the one rung of the ladder with no committed fixture on either
    /// mirror** (L5.F6).
    ///
    /// v6 is the rung that appended `joint_2d` / `joint_3d` / `audio_source` /
    /// `audio_listener`. Its frozen records were exercised only by round-tripping
    /// the *current* definitions through themselves, so a silent edit to either
    /// mirror's v6 rung — or a drift between the two mirrors at that rung —
    /// failed no test. v5 below it and v7 above it are both byte-pinned; this
    /// closes the gap between them.
    #[test]
    fn scene_v6_fixture_loads_forever_and_lifts() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v6.inf_lvl");
        if !path.exists() {
            eprintln!(
                "SKIP: scene_v6 fixture not blessed yet ({})",
                path.display()
            );
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes[0], 6, "fixture is a genuine schema-v6 payload");
        // Reproducibility lock: the committed bytes match the frozen writer.
        let rebuilt =
            bincode::serde::encode_to_vec(v6_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "committed v6 fixture matches the frozen writer"
        );

        let level = RuntimeLevel::decode(&bytes).expect("v6 fixture decodes");
        assert_eq!(level.title, "V6 Fixture Level");
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();
        // The v6 slots this rung EXISTS for are carried, not defaulted away.
        assert!(
            by_name("Cube").joint_3d.is_some(),
            "v6 is the rung that appended the joint slots; the fixture must exercise one"
        );
        // And the v7 field v6 does not have lifts to its default.
        assert_eq!(by_name("Ground").mesh.unwrap().asset, None);
        assert_eq!(
            by_name("Ground").mesh.unwrap().primitive,
            inf_ecs::components::Primitive::Plane
        );
        assert_eq!(level.settings.sim_hz, 120.0);
        // Rewriting lifts to the current schema and re-decodes equal.
        let out = level.encode().unwrap();
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
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
                    material: Some(MaterialV21 {
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
            settings: RuntimeSettingsV9 {
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
        assert_eq!(
            bytes[0], SCHEMA_VERSION as u8,
            "encode always writes the current schema"
        );
        let back = RuntimeLevel::decode(&bytes).expect("current schema decodes");
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
                asset: None,
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

    // ── v9 forever-load fixture (frozen pre-v10) ────────────────────────────

    /// A minimal all-`None` frozen v9 entity record, filled via struct-update
    /// syntax by [`v9_scene_reference`].
    fn v9_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV9 {
        EntityRecordV9 {
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

    /// A representative frozen schema-v9 scene — the provenance source for the
    /// committed `scene_v9.inf_lvl`. Carries a **v9 `Terrain` with an asset
    /// reference** (the thing v9 added) plus a mesh and a light, so the pre-v10
    /// entity + settings byte layouts are pinned by committed bytes.
    fn v9_scene_reference() -> SceneFileV9 {
        use inf_ecs::components::Primitive;
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
                    material: Some(MaterialV21::default()),
                    ..v9_rec(g(0x9001), "Cube", None)
                },
                EntityRecordV9 {
                    // A frozen record carries the frozen component: v9's terrain
                    // predates P19.1's data maps, so it is a `TerrainV14`.
                    terrain: Some(TerrainV14::from_current(Terrain {
                        asset: Some(g(0x9_00AA)),
                        ..fixture_terrain()
                    })),
                    ..v9_rec(g(0x9002), "Terrain", None)
                },
                EntityRecordV9 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v9_rec(g(0x9003), "Sun", None)
                },
            ],
            settings: RuntimeSettingsV9 {
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

    /// Bless the committed `scene_v9.inf_lvl` from [`v9_scene_reference`] under
    /// `INF_BLESS_FIXTURES=1` (inert otherwise) — the same discipline the v7/v8
    /// fixtures use. Never hand-edit the committed bytes.
    #[test]
    fn bless_scene_v9_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v9_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 9);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v9.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed scene_v9 fixture: {}", path.display());
    }

    /// The committed schema-v9 fixture — written by the **pre-v10 codec**, before
    /// the entity record grew its two world-partition slots and the settings grew
    /// their partition block — still decodes here, with everything lifted to its
    /// documented default. This is the "old bytes load forever" gate for the v10
    /// bump.
    #[test]
    fn scene_v9_fixture_decodes_with_v10_defaults() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v9.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v9 fixture present");
        assert_eq!(bytes[0], 9, "fixture is a genuine schema-v9 payload");
        // Reproducibility lock: the frozen v9 writer still emits those exact bytes.
        let rebuilt =
            bincode::serde::encode_to_vec(v9_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "committed v9 fixture matches the frozen writer"
        );

        let level = RuntimeLevel::decode(&bytes).expect("v9 fixture decodes");
        assert_eq!(level.title, "V9 Fixture Level");
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // The v9 content survives the frozen-record hop intact …
        let terrain = by_name("Terrain").terrain.as_ref().expect("terrain slot");
        assert_eq!(terrain.asset, Some(uuid::Uuid::from_u128(0x9_00AA)));
        assert_eq!(terrain.data.tile_count(), 2);
        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(uuid::Uuid::from_u128(0x90A1))
        );
        assert_eq!(level.settings.sim_hz, 90.0);
        assert_eq!(level.settings.render.exposure, 1.1);

        // … and every v10 field lifts to its documented default: no streaming
        // sources, nothing pinned always-loaded, partitioning OFF.
        for e in &level.entities {
            assert!(e.streaming_source.is_none());
            assert!(e.always_loaded.is_none());
        }
        assert_eq!(level.settings.partition, PartitionSettings::default());
        assert!(
            !level.settings.partition.enabled,
            "a pre-v10 level is a single document"
        );

        // Rewriting lifts to the current schema (v10) and re-decodes equal.
        let out = level.encode().unwrap();
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
    }

    /// The **downgrade-bless** direction, as a checked property rather than a
    /// path only a `INF_BLESS_FIXTURES=1` run ever walks.
    ///
    /// `from_current` → `into_current` must be the identity on everything the v9
    /// shape can hold, and must drop exactly one thing: the partition block,
    /// which has no v9 home. That is what "lossy in one documented direction"
    /// means, and it is the property a future fixture re-bless depends on.
    #[test]
    fn v9_settings_downgrade_is_lossless_except_for_the_partition_block() {
        let live = RuntimeSettings {
            gravity_2d: Vec2d::new(0.0, -18.0),
            gravity_3d: Vec3d::new(0.0, -9.81, 0.0),
            sim_hz: 90.0,
            render: RenderSettingsRecord {
                exposure: 1.4,
                taa: true,
                ..RenderSettingsRecord::default()
            },
            partition: PartitionSettings {
                enabled: true,
                cell_size_m: 64.0,
                activation_radius_m: 80.0,
                prefetch_margin_m: 96.0,
            },
        };
        let back = RuntimeSettingsV9::from_current(live).into_current();
        assert_eq!(back.gravity_2d, live.gravity_2d);
        assert_eq!(back.gravity_3d, live.gravity_3d);
        assert_eq!(back.sim_hz, live.sim_hz);
        assert_eq!(back.render, live.render);
        assert_eq!(
            back.partition,
            PartitionSettings::default(),
            "the partition block has no v9 home and must come back defaulted"
        );
        // An already-unpartitioned settings block survives the hop exactly.
        let plain = RuntimeSettings::default();
        assert_eq!(RuntimeSettingsV9::from_current(plain).into_current(), plain);
    }

    /// The v10 additions round-trip byte-identically: both new entity slots and
    /// the file-level partition block.
    #[test]
    fn v10_partition_slots_and_settings_round_trip() {
        use inf_ecs::components::{AlwaysLoaded, StreamingSource};
        let g = uuid::Uuid::from_u128;
        let mut level = RuntimeLevel {
            title: "V10 Partition".into(),
            entities: vec![
                RuntimeEntity {
                    streaming_source: Some(StreamingSource { radius_m: 300.0 }),
                    ..v9_rec(g(0xA001), "Player", None).into_runtime()
                },
                RuntimeEntity {
                    always_loaded: Some(AlwaysLoaded),
                    ..v9_rec(g(0xA002), "GameMode", None).into_runtime()
                },
                v9_rec(g(0xA003), "Prop", None).into_runtime(),
            ],
            settings: RuntimeSettings {
                partition: PartitionSettings {
                    enabled: true,
                    cell_size_m: 128.0,
                    activation_radius_m: 200.0,
                    prefetch_margin_m: 300.0,
                },
                ..RuntimeSettings::default()
            },
        };

        let bytes = level.encode().unwrap();
        assert_eq!(
            bytes[0], SCHEMA_VERSION as u8,
            "a partitioned level writes a current-schema payload"
        );
        let back = RuntimeLevel::decode(&bytes).expect("v10 decodes");
        assert_eq!(back, level);
        assert_eq!(back.encode().unwrap(), bytes, "re-encode is byte-identical");
        assert_eq!(back.entities[0].streaming_source.unwrap().radius_m, 300.0);
        assert_eq!(back.entities[1].always_loaded, Some(AlwaysLoaded));
        assert!(back.entities[2].streaming_source.is_none());
        assert_eq!(back.settings.partition.cell_size_m, 128.0);

        // Turning partitioning off really moves the bytes (the block is persisted,
        // not inferred).
        level.settings.partition = PartitionSettings::default();
        let off = level.encode().unwrap();
        assert_ne!(off, bytes);
        assert_eq!(RuntimeLevel::decode(&off).unwrap(), level);
    }

    // ── schema v11 (P17.1 sky authority) ──────────────────────────────────

    /// An all-`None` frozen v10 entity — the struct-update base for
    /// [`v10_scene_reference`]. Built through the downgrade hop so the field list
    /// can never drift from the live record.
    fn v10_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV10 {
        EntityRecordV10::from_current(v9_rec(guid, name, parent).into_runtime())
    }

    /// A representative frozen schema-v10 scene — the provenance source for the
    /// committed `scene_v10.inf_lvl`. Carries the **v10** additions (a streaming
    /// source, an always-loaded marker, a partitioned settings block) plus a mesh
    /// and a light, so the pre-v11 entity byte layout is pinned by committed bytes.
    fn v10_scene_reference() -> SceneFileV10 {
        use inf_ecs::components::Primitive;
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
                    material: Some(MaterialV21::default()),
                    streaming_source: Some(StreamingSource { radius_m: 300.0 }),
                    ..v10_rec(g(0xA001), "Player", None)
                },
                EntityRecordV10 {
                    always_loaded: Some(AlwaysLoaded),
                    ..v10_rec(g(0xA002), "GameMode", None)
                },
                EntityRecordV10 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v10_rec(g(0xA003), "Sun", None)
                },
            ],
            settings: RuntimeSettings {
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

    /// Bless the committed `scene_v10.inf_lvl` from [`v10_scene_reference`] under
    /// `INF_BLESS_FIXTURES=1` (inert otherwise). Never hand-edit the committed
    /// bytes.
    #[test]
    fn bless_scene_v10_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v10_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 10);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v10.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed scene_v10 fixture: {}", path.display());
    }

    /// The committed schema-v10 fixture — written by the **pre-v11 codec**, before
    /// the entity record grew its two sky-authority slots — still decodes here,
    /// with everything lifted to its documented default. The "old bytes load
    /// forever" gate for the v11 bump.
    #[test]
    fn scene_v10_fixture_decodes_with_v11_defaults() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v10.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v10 fixture present");
        assert_eq!(bytes[0], 10, "fixture is a genuine schema-v10 payload");
        // Reproducibility lock: the frozen v10 writer still emits those exact bytes.
        let rebuilt =
            bincode::serde::encode_to_vec(v10_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "committed v10 fixture matches the frozen writer"
        );

        let level = RuntimeLevel::decode(&bytes).expect("v10 fixture decodes");
        assert_eq!(level.title, "V10 Fixture Level");
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // The v10 content survives the frozen-record hop intact …
        assert_eq!(by_name("Player").streaming_source.unwrap().radius_m, 300.0);
        assert_eq!(by_name("GameMode").always_loaded, Some(AlwaysLoaded));
        assert_eq!(
            by_name("Player").mesh.unwrap().asset,
            Some(uuid::Uuid::from_u128(0xA0A1))
        );
        assert_eq!(level.settings.sim_hz, 90.0);
        assert!(level.settings.partition.enabled);
        assert_eq!(level.settings.partition.cell_size_m, 128.0);

        // … and every v11 slot lifts to its documented default: no clock at all,
        // which is what makes a pre-v11 level render under the retired sun.
        for e in &level.entities {
            assert!(e.time_of_day.is_none());
            assert!(e.sky_atmosphere.is_none());
        }

        // Rewriting lifts to the current schema (v11) and re-decodes equal.
        let out = level.encode().unwrap();
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
    }

    /// The **downgrade-bless** direction for the v10 entity record, as a checked
    /// property rather than a path only `INF_BLESS_FIXTURES=1` walks.
    #[test]
    fn v10_entity_downgrade_is_lossless_except_for_the_sky_slots() {
        let g = uuid::Uuid::from_u128;
        let live = RuntimeEntity {
            streaming_source: Some(StreamingSource { radius_m: 42.0 }),
            always_loaded: Some(AlwaysLoaded),
            time_of_day: Some(TimeOfDay {
                seconds: 1234.0,
                rate: 60.0,
                ..TimeOfDay::default()
            }),
            sky_atmosphere: Some(SkyAtmosphere::default()),
            ..v9_rec(g(0xB001), "Sky", None).into_runtime()
        };
        let back = EntityRecordV10::from_current(live.clone()).into_runtime();
        assert_eq!(back.streaming_source, live.streaming_source);
        assert_eq!(back.always_loaded, live.always_loaded);
        assert_eq!(back.name, live.name);
        assert!(
            back.time_of_day.is_none() && back.sky_atmosphere.is_none(),
            "the sky slots have no v10 home and must come back empty"
        );
        // An entity with no clock survives the hop exactly.
        let plain = v9_rec(g(0xB002), "Prop", None).into_runtime();
        assert_eq!(
            EntityRecordV10::from_current(plain.clone()).into_runtime(),
            plain
        );
    }

    /// The v11 additions round-trip byte-identically, and a level that carries a
    /// clock really moves the bytes (the slots are persisted, not inferred).
    #[test]
    fn v11_sky_slots_round_trip() {
        let g = uuid::Uuid::from_u128;
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
        let mut level = RuntimeLevel {
            title: "V11 Sky".into(),
            entities: vec![
                RuntimeEntity {
                    time_of_day: Some(tod),
                    sky_atmosphere: Some(atmos),
                    ..v9_rec(g(0xC001), "Sky", None).into_runtime()
                },
                v9_rec(g(0xC002), "Prop", None).into_runtime(),
            ],
            settings: RuntimeSettings::default(),
        };

        let bytes = level.encode().unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = RuntimeLevel::decode(&bytes).expect("v11 decodes");
        assert_eq!(back, level);
        assert_eq!(back.encode().unwrap(), bytes, "re-encode is byte-identical");
        assert_eq!(back.entities[0].time_of_day, Some(tod));
        assert_eq!(back.entities[0].sky_atmosphere, Some(atmos));
        assert!(back.entities[1].time_of_day.is_none());

        // Dropping the clock really moves the bytes.
        level.entities[0].time_of_day = None;
        level.entities[0].sky_atmosphere = None;
        let without = level.encode().unwrap();
        assert_ne!(without, bytes);
        assert_eq!(RuntimeLevel::decode(&without).unwrap(), level);
    }

    // ── schema v12 (P17.2 physical atmosphere) ────────────────────────────

    /// An all-`None` frozen v11 entity — the struct-update base for
    /// [`v11_scene_reference`]. Built through the downgrade hop so the field list
    /// can never drift from the live record.
    fn v11_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV11 {
        EntityRecordV11::from_current(v9_rec(guid, name, parent).into_runtime())
    }

    /// The **v11** atmosphere the fixture's sky entity carries: deliberately
    /// **non-default** in two of the nine frozen fields, so the v12 hop is proven
    /// to preserve the v11 half rather than merely to produce defaults.
    ///
    /// Spelled out in **literals** rather than built from
    /// `SkyAtmosphereV11::from_current(SkyAtmosphere::default())`: the committed
    /// bytes this produces must be traceable to written values, and a frozen
    /// record must not be able to move when the live component's defaults are
    /// re-tuned (the doctrine on `SkyAtmosphereV11` itself).
    fn v11_fixture_atmosphere() -> SkyAtmosphereV11 {
        SkyAtmosphereV11 {
            enabled: true,
            sun_intensity: 4.25,
            sun_color: Color::new(1.0, 0.98, 0.95, 1.0),
            moon_intensity: 0.15,
            moon_color: Color::new(0.62, 0.72, 1.0, 1.0),
            zenith: Color::new(0.012, 0.021, 0.038, 1.0),
            horizon: Color::new(0.055, 0.081, 0.120, 1.0),
            ground: Color::new(0.009, 0.011, 0.015, 1.0),
            night_darkening: 0.35,
        }
    }

    /// A representative frozen schema-v11 scene — the provenance source for the
    /// committed `scene_v11.inf_lvl`. Carries the **v11** additions (a clock plus a
    /// non-default pre-v12 `SkyAtmosphere`) on top of the v10 world-partition
    /// content, so the pre-v12 entity byte layout is pinned by committed bytes.
    fn v11_scene_reference() -> SceneFileV11 {
        use inf_ecs::components::Primitive;
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
                    material: Some(MaterialV21::default()),
                    streaming_source: Some(StreamingSource { radius_m: 300.0 }),
                    ..v11_rec(g(0xB001), "Player", None)
                },
                EntityRecordV11 {
                    always_loaded: Some(AlwaysLoaded),
                    ..v11_rec(g(0xB002), "GameMode", None)
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
                    ..v11_rec(g(0xB003), "Sky", None)
                },
            ],
            settings: RuntimeSettings {
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

    /// Bless the committed `scene_v11.inf_lvl` from [`v11_scene_reference`] under
    /// `INF_BLESS_FIXTURES=1` (inert otherwise). Never hand-edit the committed
    /// bytes.
    #[test]
    fn bless_scene_v11_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v11_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 11);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v11.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed scene_v11 fixture: {}", path.display());
    }

    /// The committed schema-v11 fixture — written by the **pre-v12 codec**, before
    /// `SkyAtmosphere` grew its physical-atmosphere block — still decodes here,
    /// with the v11 half preserved verbatim and the 13 new fields lifted to their
    /// live defaults. The "old bytes load forever" gate for the v12 bump.
    #[test]
    fn v11_loads_and_lifts_the_atmosphere() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v11.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v11 fixture present");
        assert_eq!(bytes[0], 11, "fixture is a genuine schema-v11 payload");
        // Reproducibility lock: the frozen v11 writer still emits those exact bytes.
        let rebuilt =
            bincode::serde::encode_to_vec(v11_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "committed v11 fixture matches the frozen writer"
        );

        let level = RuntimeLevel::decode(&bytes).expect("v11 fixture decodes");
        assert_eq!(level.title, "V11 Fixture Level");
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // The v11 content survives the frozen-record hop intact …
        assert_eq!(by_name("Player").streaming_source.unwrap().radius_m, 300.0);
        assert_eq!(by_name("GameMode").always_loaded, Some(AlwaysLoaded));
        assert_eq!(by_name("Sky").time_of_day.unwrap().rate, 120.0);
        assert_eq!(level.settings.sim_hz, 90.0);
        assert!(level.settings.partition.enabled);

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
        assert_eq!(a.tint_strength, d.tint_strength);
        assert_eq!(a.aerial_perspective, d.aerial_perspective);
        assert_eq!(a.fog_density, d.fog_density);
        assert_eq!(a.fog_falloff, d.fog_falloff);
        assert_eq!(a.fog_height, d.fog_height);
        assert_eq!(a.fog_color, d.fog_color);
        assert_eq!(
            a.tint_strength, 0.0,
            "a v11 level is not tinted toward itself"
        );
        assert_eq!(a.fog_density, 0.0, "a v11 level had no height fog");

        // Rewriting lifts to the current schema (v13) and re-decodes equal.
        let out = level.encode().unwrap();
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
    }

    /// The **downgrade-bless** direction for the v11 entity record, as a checked
    /// property rather than a path only `INF_BLESS_FIXTURES=1` walks.
    #[test]
    fn v11_entity_downgrade_is_lossless_except_for_the_physical_atmosphere_block() {
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
        let live = RuntimeEntity {
            streaming_source: Some(StreamingSource { radius_m: 42.0 }),
            always_loaded: Some(AlwaysLoaded),
            time_of_day: Some(tod),
            sky_atmosphere: Some(authored),
            ..v9_rec(g(0xC001), "Sky", None).into_runtime()
        };
        let back = EntityRecordV11::from_current(live.clone()).into_runtime();
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

        // An entity whose atmosphere is entirely default survives the hop exactly,
        // as does one with no atmosphere at all.
        let defaulted = RuntimeEntity {
            sky_atmosphere: Some(SkyAtmosphere::default()),
            ..v9_rec(g(0xC002), "PlainSky", None).into_runtime()
        };
        assert_eq!(
            EntityRecordV11::from_current(defaulted.clone()).into_runtime(),
            defaulted
        );
        let plain = v9_rec(g(0xC003), "Prop", None).into_runtime();
        assert_eq!(
            EntityRecordV11::from_current(plain.clone()).into_runtime(),
            plain
        );
    }

    /// The v12 additions round-trip byte-identically, and a level that authors the
    /// physical-atmosphere block really moves the bytes (the fields are persisted,
    /// not inferred) — the guard that would have caught the bump being skipped.
    #[test]
    fn v12_physical_atmosphere_round_trips() {
        let g = uuid::Uuid::from_u128;
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
        let mut level = RuntimeLevel {
            title: "V12 Sky".into(),
            entities: vec![
                RuntimeEntity {
                    time_of_day: Some(TimeOfDay::default()),
                    sky_atmosphere: Some(atmos),
                    ..v9_rec(g(0xD001), "Sky", None).into_runtime()
                },
                v9_rec(g(0xD002), "Prop", None).into_runtime(),
            ],
            settings: RuntimeSettings::default(),
        };

        let bytes = level.encode().unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = RuntimeLevel::decode(&bytes).expect("v12 decodes");
        assert_eq!(back, level);
        assert_eq!(back.encode().unwrap(), bytes, "re-encode is byte-identical");
        assert_eq!(back.entities[0].sky_atmosphere, Some(atmos));
        assert!(back.entities[1].sky_atmosphere.is_none());

        // Changing one v12-only field really moves the bytes. If the block were
        // not persisted (i.e. if the bump had been skipped) this would not hold.
        level.entities[0].sky_atmosphere = Some(SkyAtmosphere {
            fog_density: 0.0,
            ..atmos
        });
        let without_fog = level.encode().unwrap();
        assert_ne!(without_fog, bytes);
        assert_eq!(RuntimeLevel::decode(&without_fog).unwrap(), level);
    }

    /// A **v12 payload is genuinely longer** than the v11 one for the same level —
    /// the concrete reason `#[serde(default)]` could not have rescued this bump.
    /// bincode is not self-describing: it reads a fixed field count positionally,
    /// so feeding these v11 bytes to the grown struct would run off the end of the
    /// atmosphere and into the next record.
    #[test]
    fn v12_atmosphere_is_wider_on_the_wire_than_v11() {
        let v11 = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v11.inf_lvl"),
        )
        .expect("committed v11 fixture present");
        let lifted = RuntimeLevel::decode(&v11).unwrap().encode().unwrap();
        // The v12 block: 13 appended fields = 1 bool byte + 11 f32 + a 4-f32
        // Color = 1 + 60 bytes. `encode` writes the **current** schema, so the
        // lift also appends the v13 cloud block's 62 bytes (broken down in
        // `v13_clouds_are_wider_on_the_wire_than_v12`) and the v14 weather
        // block's 38 (broken down in `v14_weather_is_wider_on_the_wire_than_v13`)
        // — one growth per bump, all on the one entity that carries an
        // atmosphere.
        // …and **every** entity pays the v17 `water_body: None`, v18
        // `buoyancy: None`, v19 `voxel_volume: None`, v20 `destructible: None` and the THREE v21 character
        // discriminants, which are the whole price of P20.1, P20.2, P21.1 and
        // P22.2 for a level with no water, nothing that floats, no volumetric
        // ground and nothing breakable.
        let entities = RuntimeLevel::decode(&v11).unwrap().entities.len();
        let materialed = materialed_entities(&v11);
        assert_eq!(
            lifted.len(),
            v11.len()
                + 61
                + 62
                + 38
                + entities
                    * (WATER_SLOT_BYTES
                        + BUOYANCY_SLOT_BYTES
                        + VOXEL_SLOT_BYTES
                        + DESTRUCTIBLE_SLOT_BYTES
                        + CHARACTER_SLOT_BYTES
                        + MOVEMENT_SLOT_BYTES)
                + materialed * MATERIAL_BINDING_BYTES,
            "the one entity carrying an atmosphere grew by the physical block"
        );
    }

    // ── schema v13 (P17.3 volumetric clouds) ──────────────────────────────

    /// An all-`None` frozen v12 entity — the struct-update base for
    /// [`v12_scene_reference`]. Built through the downgrade hop so the field list
    /// can never drift from the live record.
    fn v12_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV12 {
        EntityRecordV12::from_current(v9_rec(guid, name, parent).into_runtime())
    }

    /// The **v12** atmosphere the fixture's sky entity carries: deliberately
    /// **non-default** in four of the 22 frozen fields (two from the v11 half, two
    /// from the physical block), so the v13 hop is proven to preserve what v12
    /// authored rather than merely to produce defaults.
    ///
    /// Spelled out in **literals** rather than built from
    /// `SkyAtmosphereV12::from_current(SkyAtmosphere::default())`: the committed
    /// bytes this produces must be traceable to written values, and a frozen
    /// record must not be able to move when the live component's defaults are
    /// re-tuned (the doctrine on `SkyAtmosphereV12` itself).
    fn v12_fixture_atmosphere() -> SkyAtmosphereV12 {
        SkyAtmosphereV12 {
            enabled: true,
            sun_intensity: 4.25,
            sun_color: Color::new(1.0, 0.98, 0.95, 1.0),
            moon_intensity: 0.15,
            moon_color: Color::new(0.62, 0.72, 1.0, 1.0),
            zenith: Color::new(0.012, 0.021, 0.038, 1.0),
            horizon: Color::new(0.055, 0.081, 0.120, 1.0),
            ground: Color::new(0.009, 0.011, 0.015, 1.0),
            night_darkening: 0.35,
            physical: true,
            sky_intensity: 1.0,
            turbidity: 2.5,
            mie_anisotropy: 0.8,
            sun_disc_deg: 0.545,
            moon_disc_deg: 0.52,
            star_intensity: 1.0,
            tint_strength: 0.0,
            aerial_perspective: 1.0,
            fog_density: 6e-4,
            fog_falloff: 0.002,
            fog_height: 0.0,
            fog_color: Color::new(1.0, 1.0, 1.0, 1.0),
        }
    }

    /// A representative frozen schema-v12 scene — the provenance source for the
    /// committed `scene_v12.inf_lvl`. Carries the **v12** additions (a clock plus a
    /// non-default pre-v13 `SkyAtmosphere`) on top of the v10 world-partition
    /// content, so the pre-v13 entity byte layout is pinned by committed bytes.
    fn v12_scene_reference() -> SceneFileV12 {
        use inf_ecs::components::Primitive;
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
                    material: Some(MaterialV21::default()),
                    streaming_source: Some(StreamingSource { radius_m: 300.0 }),
                    ..v12_rec(g(0xB001), "Player", None)
                },
                EntityRecordV12 {
                    always_loaded: Some(AlwaysLoaded),
                    ..v12_rec(g(0xB002), "GameMode", None)
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
                    ..v12_rec(g(0xB003), "Sky", None)
                },
            ],
            settings: RuntimeSettings {
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

    /// Bless the committed `scene_v12.inf_lvl` from [`v12_scene_reference`] under
    /// `INF_BLESS_FIXTURES=1` (inert otherwise). Never hand-edit the committed
    /// bytes.
    #[test]
    fn bless_scene_v12_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v12_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 12);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v12.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed scene_v12 fixture: {}", path.display());
    }

    /// The committed schema-v12 fixture — written by the **pre-v13 codec**, before
    /// `SkyAtmosphere` grew its volumetric-cloud block — still decodes here, with
    /// the v12 shape preserved verbatim and the 14 new fields lifted to their live
    /// defaults. The "old bytes load forever" gate for the v13 bump.
    #[test]
    fn v12_loads_and_lifts_the_clouds() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v12.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v12 fixture present");
        assert_eq!(bytes[0], 12, "fixture is a genuine schema-v12 payload");
        // Reproducibility lock: the frozen v12 writer still emits those exact bytes.
        let rebuilt =
            bincode::serde::encode_to_vec(v12_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "committed v12 fixture matches the frozen writer"
        );

        let level = RuntimeLevel::decode(&bytes).expect("v12 fixture decodes");
        assert_eq!(level.title, "V12 Fixture Level");
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // The v12 content survives the frozen-record hop intact …
        assert_eq!(by_name("Player").streaming_source.unwrap().radius_m, 300.0);
        assert_eq!(by_name("GameMode").always_loaded, Some(AlwaysLoaded));
        assert_eq!(by_name("Sky").time_of_day.unwrap().rate, 120.0);
        assert_eq!(level.settings.sim_hz, 90.0);
        assert!(level.settings.partition.enabled);

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

        // Rewriting lifts to the current schema (v13) and re-decodes equal.
        let out = level.encode().unwrap();
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
    }

    /// The **downgrade-bless** direction for the v12 entity record, as a checked
    /// property rather than a path only `INF_BLESS_FIXTURES=1` walks.
    #[test]
    fn v12_entity_downgrade_is_lossless_except_for_the_cloud_block() {
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
        let live = RuntimeEntity {
            streaming_source: Some(StreamingSource { radius_m: 42.0 }),
            always_loaded: Some(AlwaysLoaded),
            time_of_day: Some(tod),
            sky_atmosphere: Some(authored),
            ..v9_rec(g(0xE001), "Sky", None).into_runtime()
        };
        let back = EntityRecordV12::from_current(live.clone()).into_runtime();
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

        // An entity whose atmosphere is entirely default survives the hop exactly,
        // as does one with no atmosphere at all.
        let defaulted = RuntimeEntity {
            sky_atmosphere: Some(SkyAtmosphere::default()),
            ..v9_rec(g(0xE002), "PlainSky", None).into_runtime()
        };
        assert_eq!(
            EntityRecordV12::from_current(defaulted.clone()).into_runtime(),
            defaulted
        );
        let plain = v9_rec(g(0xE003), "Prop", None).into_runtime();
        assert_eq!(
            EntityRecordV12::from_current(plain.clone()).into_runtime(),
            plain
        );
    }

    /// The v13 additions round-trip byte-identically, and a level that authors the
    /// volumetric-cloud block really moves the bytes (the fields are persisted, not
    /// inferred) — the guard that would have caught the bump being skipped.
    #[test]
    fn v13_clouds_round_trip() {
        let g = uuid::Uuid::from_u128;
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
        let mut level = RuntimeLevel {
            title: "V13 Sky".into(),
            entities: vec![
                RuntimeEntity {
                    time_of_day: Some(TimeOfDay::default()),
                    sky_atmosphere: Some(atmos),
                    ..v9_rec(g(0xF001), "Sky", None).into_runtime()
                },
                v9_rec(g(0xF002), "Prop", None).into_runtime(),
            ],
            settings: RuntimeSettings::default(),
        };

        let bytes = level.encode().unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = RuntimeLevel::decode(&bytes).expect("v13 decodes");
        assert_eq!(back, level);
        assert_eq!(back.encode().unwrap(), bytes, "re-encode is byte-identical");
        assert_eq!(back.entities[0].sky_atmosphere, Some(atmos));
        assert!(back.entities[1].sky_atmosphere.is_none());

        // Changing one v13-only field really moves the bytes. If the block were
        // not persisted (i.e. if the bump had been skipped) this would not hold.
        level.entities[0].sky_atmosphere = Some(SkyAtmosphere {
            clouds_enabled: false,
            ..atmos
        });
        let without_clouds = level.encode().unwrap();
        assert_ne!(without_clouds, bytes);
        assert_eq!(RuntimeLevel::decode(&without_clouds).unwrap(), level);
    }

    /// The Ring-0 twin of the editor codec's `cloud_defaults_are_the_documented_ones`.
    ///
    /// Both codecs carry their own `SkyAtmosphereV12`, and the whole frozen-record
    /// doctrine rests on those being pinned to **literals** rather than to
    /// `SkyAtmosphere::default()` — a v12 payload that omitted a field must decode
    /// to what v12 meant, however the live component is re-tuned later. Having the
    /// gate in only one crate would leave the other free to drift, and the two
    /// ladders lift v11 by different routes (this crate in one hop from
    /// `SkyAtmosphere::default()`, the editor's through its own `into_v12`
    /// literals), so their agreeing is exactly what makes those routes equivalent.
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
        assert_eq!(frozen.into_current(), a);

        // This crate's frozen record has its OWN literal defaults, so a v12 payload
        // that omitted a field still decodes to what v12 meant however the live
        // component is re-tuned later. Asserted against the `v12_*` fns the
        // `#[serde(default = "…")]` markers actually name — which is the same
        // claim the editor codec makes by round-tripping an empty JSON object,
        // reached without adding a self-describing codec to a Ring-0 crate that
        // has no use for one.
        assert!(v12_sky_true());
        assert_eq!(v12_sun_intensity(), 3.0);
        assert_eq!(v12_sun_color(), Color::new(1.0, 0.98, 0.95, 1.0));
        assert_eq!(v12_moon_intensity(), 0.15);
        assert_eq!(v12_moon_color(), Color::new(0.62, 0.72, 1.0, 1.0));
        assert_eq!(v12_sky_zenith(), Color::new(0.012, 0.021, 0.038, 1.0));
        assert_eq!(v12_sky_horizon(), Color::new(0.055, 0.081, 0.120, 1.0));
        assert_eq!(v12_sky_ground(), Color::new(0.009, 0.011, 0.015, 1.0));
        assert_eq!(v12_night_darkening(), 0.85);
        assert_eq!(v12_one(), 1.0);
        assert_eq!(v12_mie_anisotropy(), 0.8);
        assert_eq!(v12_sun_disc_deg(), 0.545);
        assert_eq!(v12_moon_disc_deg(), 0.52);
        assert_eq!(v12_fog_falloff(), 0.002);
        assert_eq!(v12_fog_color(), Color::new(1.0, 1.0, 1.0, 1.0));

        // ...and today those literals agree with the live component's v12 half,
        // which is what makes v13 a pure append and what keeps this crate's
        // one-hop v11 lift equivalent to the editor ladder's two-hop one. A fact
        // about today, downstream of the literals above, not a definition of
        // either side.
        assert_eq!(frozen.sun_intensity, v12_sun_intensity());
        assert_eq!(frozen.night_darkening, v12_night_darkening());
        assert_eq!(frozen.mie_anisotropy, v12_mie_anisotropy());
        assert_eq!(frozen.sun_disc_deg, v12_sun_disc_deg());
        assert_eq!(frozen.moon_disc_deg, v12_moon_disc_deg());
        assert_eq!(frozen.fog_falloff, v12_fog_falloff());
        assert_eq!(frozen.fog_color, v12_fog_color());
        assert_eq!(frozen.enabled, v12_sky_true());
        assert_eq!(frozen.physical, v12_sky_true());
    }

    /// A **v13 payload is genuinely longer** than the v12 one for the same level —
    /// the concrete reason `#[serde(default)]` could not have rescued this bump.
    /// bincode is not self-describing: it reads a fixed field count positionally,
    /// so feeding these v12 bytes to the grown struct would run off the end of the
    /// atmosphere and into the next record.
    #[test]
    fn v13_clouds_are_wider_on_the_wire_than_v12() {
        let v12 = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v12.inf_lvl"),
        )
        .expect("committed v12 fixture present");
        let lifted = RuntimeLevel::decode(&v12).unwrap().encode().unwrap();

        // The 14 appended fields, priced individually rather than as one number.
        //
        // The `cloud_seed` term is the reason this is spelled out: the workspace
        // uses `bincode::config::standard()`, whose integer encoding is
        // **variable length**, so the seed's cost is a function of the *default
        // seed's value* and not of its type. A future change to that default
        // would silently move the total, and a bare `+ 62` would then fail with
        // "the cloud block grew" — which would be a lie. Priced this way it fails
        // saying which field moved and why.
        const BOOL: usize = 1; // bincode encodes bool as one byte
        const F32: usize = 4; // floats are fixed width even under varint configs
        const COLOR: usize = 4 * F32;
        let seed = SkyAtmosphere::default().cloud_seed;
        assert_eq!(
            seed, 0,
            "the default cloud seed moved; the wire-width breakdown below prices it as a varint, so re-derive the total rather than editing the number"
        );
        let expected = BOOL                 // clouds_enabled
            + 11 * F32                      // coverage/type/bottom/top/density/detail/
                                            // wind_x/wind_z/phase_g/shadow/ambient
            + varint_len(u64::from(seed))   // cloud_seed
            + COLOR; // cloud_color
        assert_eq!(
            expected, 62,
            "the field-by-field price no longer sums to 62"
        );

        // `encode` writes the **current** schema, so the lift also appends the
        // v14 weather block — priced in `v14_weather_is_wider_on_the_wire_than_v13`
        // and named here rather than folded into the number.
        let entities = RuntimeLevel::decode(&v12).unwrap().entities.len();
        let materialed = materialed_entities(&v12);
        assert_eq!(
            lifted.len(),
            v12.len()
                + expected
                + WEATHER_WIRE_BYTES
                + entities
                    * (WATER_SLOT_BYTES
                        + BUOYANCY_SLOT_BYTES
                        + VOXEL_SLOT_BYTES
                        + DESTRUCTIBLE_SLOT_BYTES
                        + CHARACTER_SLOT_BYTES
                        + MOVEMENT_SLOT_BYTES)
                + materialed * MATERIAL_BINDING_BYTES,
            "the one entity carrying an atmosphere grew by the cloud block"
        );
    }

    /// Bytes `bincode::config::standard()` spends on an unsigned integer.
    ///
    /// Its varint form is the "one byte below 251, then a tag plus a fixed-width
    /// payload" scheme; only the small case can occur for a masked 24-bit cloud
    /// seed, but the rest is here so the breakdown above stays honest if a
    /// default ever moves out of it.
    fn varint_len(v: u64) -> usize {
        match v {
            0..=250 => 1,
            251..=0xffff => 3,
            0x1_0000..=0xffff_ffff => 5,
            _ => 9,
        }
    }

    // ── schema v14 (P17.4 weather states) ─────────────────────────────────

    /// Wire width of the appended v14 weather block, priced field by field.
    ///
    /// 1 bool + a fieldless-enum variant index (bincode writes it as a varint, so
    /// the default `Clear` at index 0 costs one byte) + 9 f32. Named rather than
    /// spelled `38` at each use, so a future field lands in one place.
    const WEATHER_WIRE_BYTES: usize = 1 + 1 + 9 * 4;

    /// An all-`None` frozen v13 entity — the struct-update base for
    /// [`v13_scene_reference`]. Built through the downgrade hop so the field list
    /// can never drift from the live record.
    fn v13_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV13 {
        EntityRecordV13::from_current(v9_rec(guid, name, parent).into_runtime())
    }

    /// The **v13** atmosphere the fixture's sky entity carries: deliberately
    /// **non-default** in fields drawn from all three earlier blocks (the v11
    /// half, the physical block and the cloud block), so the v14 hop is proven to
    /// preserve what v13 authored rather than merely to produce defaults.
    ///
    /// Spelled out in **literals** for the reason the whole frozen-record scheme
    /// exists: the committed bytes must be traceable to written values, and a
    /// frozen record must not be able to move when the live component is
    /// re-tuned.
    fn v13_fixture_atmosphere() -> SkyAtmosphereV13 {
        SkyAtmosphereV13 {
            enabled: true,
            sun_intensity: 4.25,
            sun_color: Color::new(1.0, 0.98, 0.95, 1.0),
            moon_intensity: 0.15,
            moon_color: Color::new(0.62, 0.72, 1.0, 1.0),
            zenith: Color::new(0.012, 0.021, 0.038, 1.0),
            horizon: Color::new(0.055, 0.081, 0.120, 1.0),
            ground: Color::new(0.009, 0.011, 0.015, 1.0),
            night_darkening: 0.35,
            physical: true,
            sky_intensity: 1.0,
            turbidity: 2.5,
            mie_anisotropy: 0.8,
            sun_disc_deg: 0.545,
            moon_disc_deg: 0.52,
            star_intensity: 1.0,
            tint_strength: 0.0,
            aerial_perspective: 1.0,
            fog_density: 6e-4,
            fog_falloff: 0.002,
            fog_height: 0.0,
            fog_color: Color::new(1.0, 1.0, 1.0, 1.0),
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
        }
    }

    /// A representative frozen schema-v13 scene — the provenance source for the
    /// committed `scene_v13.inf_lvl`. Carries the **v13** additions (a clock plus
    /// a non-default pre-v14 `SkyAtmosphere` with clouds on) over the v10
    /// world-partition content, so the pre-v14 entity byte layout is pinned by
    /// committed bytes.
    fn v13_scene_reference() -> SceneFileV13 {
        use inf_ecs::components::Primitive;
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
                    material: Some(MaterialV21::default()),
                    streaming_source: Some(StreamingSource { radius_m: 300.0 }),
                    ..v13_rec(g(0xC001), "Player", None)
                },
                EntityRecordV13 {
                    always_loaded: Some(AlwaysLoaded),
                    ..v13_rec(g(0xC002), "GameMode", None)
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
                    ..v13_rec(g(0xC003), "Sky", None)
                },
            ],
            settings: RuntimeSettings {
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

    /// Bless the committed `scene_v13.inf_lvl` from [`v13_scene_reference`] under
    /// `INF_BLESS_FIXTURES=1` (inert otherwise). Never hand-edit the committed
    /// bytes.
    #[test]
    fn bless_scene_v13_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v13_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 13);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v13.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed scene_v13 fixture: {}", path.display());
    }

    /// The committed schema-v13 fixture — written by the **pre-v14 codec**, before
    /// `SkyAtmosphere` grew its weather block — still decodes here, with the v13
    /// shape preserved verbatim and the 11 new fields lifted to their live
    /// defaults. The "old bytes load forever" gate for the v14 bump.
    #[test]
    fn v13_loads_and_lifts_the_weather() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v13.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v13 fixture present");
        assert_eq!(bytes[0], 13, "fixture is a genuine schema-v13 payload");
        // Reproducibility lock: the frozen v13 writer still emits those exact bytes.
        let rebuilt =
            bincode::serde::encode_to_vec(v13_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "committed v13 fixture matches the frozen writer"
        );

        let level = RuntimeLevel::decode(&bytes).expect("v13 fixture decodes");
        assert_eq!(level.title, "V13 Fixture Level");
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // The v13 content survives the frozen-record hop intact …
        assert_eq!(by_name("Player").streaming_source.unwrap().radius_m, 300.0);
        assert_eq!(by_name("GameMode").always_loaded, Some(AlwaysLoaded));
        assert_eq!(by_name("Sky").time_of_day.unwrap().rate, 120.0);
        assert_eq!(level.settings.sim_hz, 90.0);

        // … including the **non-default** fields from all three earlier blocks.
        let a = by_name("Sky")
            .sky_atmosphere
            .expect("sky carries an atmosphere");
        assert_eq!(a.sun_intensity, 4.25);
        assert_eq!(a.night_darkening, 0.35);
        assert_eq!(a.turbidity, 2.5);
        assert_eq!(a.fog_density, 6e-4);
        assert!(a.clouds_enabled, "v13 could author clouds");
        assert_eq!(a.cloud_seed, 90_210);
        assert_eq!(a.cloud_wind_x, -11.5);

        // … and every v14 field lifts to its documented default: weather OFF, so
        // the authored cloud and fog fields above are still what drives the sky.
        // That is exactly what a v13 level meant.
        let d = SkyAtmosphere::default();
        assert!(!a.weather_enabled, "a v13 level had no weather block");
        assert_eq!(a.weather_target, d.weather_target);
        assert_eq!(a.weather_blend_seconds, d.weather_blend_seconds);
        assert_eq!(a.weather_blend_remaining, 0.0, "nothing in flight");
        assert_eq!(a.weather_coverage, d.weather_coverage);
        assert_eq!(a.weather_cloud_type, d.weather_cloud_type);
        assert_eq!(a.weather_wind_x, d.weather_wind_x);
        assert_eq!(a.weather_wind_z, d.weather_wind_z);
        assert_eq!(a.weather_fog_density, d.weather_fog_density);
        assert_eq!(a.weather_precipitation, d.weather_precipitation);
        assert_eq!(a.weather_snowiness, d.weather_snowiness);

        // Rewriting lifts to the current schema (v14) and re-decodes equal.
        let out = level.encode().unwrap();
        assert_eq!(out[0], SCHEMA_VERSION as u8);
        assert_eq!(RuntimeLevel::decode(&out).unwrap(), level);
    }

    /// The **downgrade-bless** direction for the v13 entity record, as a checked
    /// property rather than a path only `INF_BLESS_FIXTURES=1` walks.
    #[test]
    fn v13_entity_downgrade_is_lossless_except_for_the_weather_block() {
        let g = uuid::Uuid::from_u128;
        let authored = SkyAtmosphere {
            // the v13 shape — must survive …
            sun_intensity: 4.25,
            turbidity: 3.5,
            fog_density: 6e-4,
            clouds_enabled: true,
            cloud_coverage: 0.9,
            cloud_seed: 7,
            cloud_wind_x: 20.0,
            // … and the v14 block — must not.
            weather_enabled: true,
            weather_target: inf_ecs::components::WeatherPreset::Storm,
            weather_blend_remaining: 4.0,
            weather_precipitation: 1.0,
            weather_snowiness: 1.0,
            ..SkyAtmosphere::default()
        };
        let live = RuntimeEntity {
            time_of_day: Some(TimeOfDay::default()),
            sky_atmosphere: Some(authored),
            ..v9_rec(g(0xD001), "Sky", None).into_runtime()
        };
        let back = EntityRecordV13::from_current(live.clone()).into_runtime();
        assert_eq!(back.name, live.name);
        assert_eq!(back.time_of_day, live.time_of_day);

        let a = back.sky_atmosphere.unwrap();
        // The v13 thirty-six survive verbatim …
        assert_eq!(a.sun_intensity, 4.25);
        assert_eq!(a.turbidity, 3.5);
        assert_eq!(a.fog_density, 6e-4);
        assert!(a.clouds_enabled);
        assert_eq!(a.cloud_coverage, 0.9);
        assert_eq!(a.cloud_seed, 7);
        assert_eq!(a.cloud_wind_x, 20.0);
        // … and the weather block has no v13 home, so it comes back at the live
        // defaults — the one deliberately lossy direction.
        let d = SkyAtmosphere::default();
        assert_eq!(
            a.weather_enabled, d.weather_enabled,
            "`weather_enabled: true` cannot be stored in v13"
        );
        assert_eq!(a.weather_target, d.weather_target);
        assert_eq!(a.weather_blend_remaining, d.weather_blend_remaining);
        assert_eq!(a.weather_precipitation, d.weather_precipitation);
        assert_eq!(a.weather_snowiness, d.weather_snowiness);

        // An entity whose atmosphere is entirely default survives the hop exactly,
        // as does one with no atmosphere at all.
        let defaulted = RuntimeEntity {
            sky_atmosphere: Some(SkyAtmosphere::default()),
            ..v9_rec(g(0xD002), "PlainSky", None).into_runtime()
        };
        assert_eq!(
            EntityRecordV13::from_current(defaulted.clone()).into_runtime(),
            defaulted
        );
        let plain = v9_rec(g(0xD003), "Prop", None).into_runtime();
        assert_eq!(
            EntityRecordV13::from_current(plain.clone()).into_runtime(),
            plain
        );
    }

    /// The v14 additions round-trip byte-identically, and a level that authors the
    /// weather block really moves the bytes (the fields are persisted, not
    /// inferred) — the guard that would have caught the bump being skipped.
    #[test]
    fn v14_weather_round_trip() {
        use inf_ecs::components::WeatherPreset;
        let g = uuid::Uuid::from_u128;
        let atmos = SkyAtmosphere {
            weather_enabled: true,
            weather_target: WeatherPreset::Snow,
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
        let mut level = RuntimeLevel {
            title: "V14 Weather".into(),
            entities: vec![
                RuntimeEntity {
                    time_of_day: Some(TimeOfDay::default()),
                    sky_atmosphere: Some(atmos),
                    ..v9_rec(g(0xA101), "Sky", None).into_runtime()
                },
                v9_rec(g(0xA102), "Prop", None).into_runtime(),
            ],
            settings: RuntimeSettings::default(),
        };

        let bytes = level.encode().unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = RuntimeLevel::decode(&bytes).expect("v14 decodes");
        assert_eq!(back, level);
        assert_eq!(back.encode().unwrap(), bytes, "re-encode is byte-identical");
        assert_eq!(back.entities[0].sky_atmosphere, Some(atmos));
        assert!(back.entities[1].sky_atmosphere.is_none());

        // Changing one v14-only field really moves the bytes. If the block were
        // not persisted (i.e. if the bump had been skipped) this would not hold —
        // and the field chosen is the *enum*, which is the one whose wire form is
        // a varint rather than a fixed-width float.
        level.entities[0].sky_atmosphere = Some(SkyAtmosphere {
            weather_target: WeatherPreset::Fog,
            ..atmos
        });
        let other = level.encode().unwrap();
        assert_ne!(other, bytes);
        assert_eq!(
            other.len(),
            bytes.len(),
            "both variant indices are one byte"
        );
        assert_eq!(RuntimeLevel::decode(&other).unwrap(), level);
    }

    /// A **v13 payload is genuinely narrower** than the current one for the same
    /// level — the concrete reason `#[serde(default)]` could not have rescued this
    /// bump either. bincode is not self-describing: it reads a fixed field count
    /// positionally, so feeding these v13 bytes to the grown struct would run off
    /// the end of the atmosphere and into the next record.
    #[test]
    fn v14_weather_is_wider_on_the_wire_than_v13() {
        use inf_ecs::components::WeatherPreset;
        let v13 = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v13.inf_lvl"),
        )
        .expect("committed v13 fixture present");
        let lifted = RuntimeLevel::decode(&v13).unwrap().encode().unwrap();

        // The 11 appended fields, priced individually rather than as one number.
        //
        // The `weather_target` term is the reason this is spelled out: a fieldless
        // serde enum is written as its **variant index**, and the workspace's
        // `bincode::config::standard()` encodes integers as varints — so the
        // preset's cost is a function of the *default variant's index*, not of its
        // type. A future preset inserted before `Clear` would silently move the
        // total, and a bare `+ 38` would then fail saying "the weather block
        // grew", which would be a lie.
        const BOOL: usize = 1;
        const F32: usize = 4;
        let target = SkyAtmosphere::default().weather_target;
        assert_eq!(
            target,
            WeatherPreset::Clear,
            "the default weather preset moved; the breakdown below prices it as a \
             varint over its variant index, so re-derive the total"
        );
        // Derived from the **encoding**, not from the hand-maintained `ALL`
        // array: what costs bytes is serde's variant index, and reading it off a
        // list someone has to remember to reorder would just move the fragility.
        // A fieldless enum encodes to exactly its varint index, so the encoded
        // length IS `varint_len(index)` and the bytes ARE the index.
        let encoded = bincode::serde::encode_to_vec(target, bincode_config()).unwrap();
        assert_eq!(
            encoded,
            vec![0u8],
            "the default preset no longer encodes as variant index 0; re-derive the width below rather than editing the number"
        );
        let index = u64::from(encoded[0]);
        let expected = BOOL                 // weather_enabled
            + varint_len(index)             // weather_target
            + 9 * F32; // blend_seconds/blend_remaining/coverage/cloud_type/
                       // wind_x/wind_z/fog_density/precipitation/snowiness
        assert_eq!(
            expected, WEATHER_WIRE_BYTES,
            "the field-by-field price no longer sums to the named constant"
        );
        assert_eq!(expected, 38, "the weather block is 38 bytes on the wire");

        let entities = RuntimeLevel::decode(&v13).unwrap().entities.len();
        let materialed = materialed_entities(&v13);
        assert_eq!(
            lifted.len(),
            v13.len()
                + expected
                + entities
                    * (WATER_SLOT_BYTES
                        + BUOYANCY_SLOT_BYTES
                        + VOXEL_SLOT_BYTES
                        + DESTRUCTIBLE_SLOT_BYTES
                        + CHARACTER_SLOT_BYTES
                        + MOVEMENT_SLOT_BYTES)
                + materialed * MATERIAL_BINDING_BYTES,
            "the one entity carrying an atmosphere grew by the weather block"
        );
    }

    /// The Ring-0 twin of the editor codec's `weather_defaults_are_the_documented_ones`.
    ///
    /// Both codecs carry their own `SkyAtmosphereV13`, and the whole frozen-record
    /// doctrine rests on those being pinned to **literals** rather than to
    /// `SkyAtmosphere::default()`. Having the gate in only one crate would leave
    /// the other free to drift, and the two ladders lift v12 by different routes
    /// (this crate in one hop from `SkyAtmosphere::default()`, the editor's through
    /// its own `into_v13` literals), so their agreeing is exactly what makes those
    /// routes equivalent.
    #[test]
    fn weather_defaults_are_the_documented_ones() {
        use inf_ecs::components::WeatherPreset;
        let a = SkyAtmosphere::default();
        assert!(
            !a.weather_enabled,
            "existing content keeps the sky it was authored against"
        );
        assert_eq!(a.weather_target, WeatherPreset::Clear);
        assert_eq!(a.weather_blend_seconds, 8.0, "seconds");
        assert_eq!(a.weather_blend_remaining, 0.0, "settled");
        // The live defaults ARE a settled Clear state — the property that makes
        // enabling weather one boolean rather than a parameter hunt.
        assert_eq!(a.weather_params(), WeatherPreset::Clear.params());

        // Lifting a default-frozen record yields a default live one.
        let frozen = SkyAtmosphereV13::from_current(a);
        assert_eq!(frozen.into_current(), a);

        // This crate's frozen record has its OWN literal defaults, asserted
        // against the `v13_*` fns the `#[serde(default = "…")]` markers name.
        assert!(v13_sky_true());
        assert_eq!(v13_sun_intensity(), 3.0);
        assert_eq!(v13_night_darkening(), 0.85);
        assert_eq!(v13_one(), 1.0);
        assert_eq!(v13_mie_anisotropy(), 0.8);
        assert_eq!(v13_sun_disc_deg(), 0.545);
        assert_eq!(v13_moon_disc_deg(), 0.52);
        assert_eq!(v13_fog_falloff(), 0.002);
        assert_eq!(v13_fog_color(), Color::new(1.0, 1.0, 1.0, 1.0));
        assert_eq!(v13_cloud_coverage(), 0.35);
        assert_eq!(v13_cloud_type(), 0.7);
        assert_eq!(v13_cloud_bottom(), 1500.0);
        assert_eq!(v13_cloud_top(), 4000.0);
        assert_eq!(v13_cloud_density(), 0.04);
        assert_eq!(v13_cloud_detail(), 0.6);
        assert_eq!(v13_cloud_wind_x(), 6.0);
        assert_eq!(v13_cloud_wind_z(), 2.0);
        assert_eq!(v13_cloud_phase_g(), 0.8);
        assert_eq!(v13_cloud_color(), Color::new(1.0, 1.0, 1.0, 1.0));

        // …and today those literals agree with the live component's v13 half,
        // which is what makes v14 a pure append and what keeps this crate's
        // one-hop v12 lift equivalent to the editor ladder's two-hop one.
        assert_eq!(frozen.cloud_coverage, v13_cloud_coverage());
        assert_eq!(frozen.cloud_wind_x, v13_cloud_wind_x());
        assert_eq!(frozen.cloud_shadow, v13_one());
        assert_eq!(frozen.fog_falloff, v13_fog_falloff());
        assert_eq!(frozen.enabled, v13_sky_true());
        assert_eq!(frozen.physical, v13_sky_true());
    }

    // ── schema v15 (P19.1 erosion data maps) ──────────────────────────────

    /// An all-`None` frozen v14 entity — the struct-update base for
    /// [`v14_scene_reference`]. Built through the downgrade hop so the field list
    /// can never drift from the live record.
    fn v14_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV14 {
        EntityRecordV14::from_current(v9_rec(guid, name, parent).into_runtime())
    }

    /// The **v14** terrain the fixture carries: two shared-edge tiles from one
    /// **polynomial** height field (never `sin`/`cos` — the P14 bit-portability
    /// law), a painted splat sample, a non-default macro variation and an asset
    /// reference. Its tiles have **no** data maps, because v14 could not express
    /// them; that is exactly what the v15 lift has to reproduce.
    ///
    /// Spelled out in **literals**, and identical to the editor codec's
    /// `v14_fixture_terrain` — the two committed fixtures are byte-compared by
    /// the editor's `v14_fixture_matches_the_runtime_codecs_copy`.
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

    /// A representative frozen schema-v14 scene — the provenance source for the
    /// committed `scene_v14.inf_lvl`. Carries a terrain (the thing v15 changed)
    /// plus a mesh and a light, so the pre-v15 entity byte layout is pinned by
    /// committed bytes.
    fn v14_scene_reference() -> SceneFileV14 {
        use inf_ecs::components::Primitive;
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
                    material: Some(MaterialV21::default()),
                    ..v14_rec(g(0xD001), "Cube", None)
                },
                EntityRecordV14 {
                    terrain: Some(v14_fixture_terrain()),
                    ..v14_rec(g(0xD002), "Terrain", None)
                },
                EntityRecordV14 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v14_rec(g(0xD003), "Sun", None)
                },
            ],
            settings: RuntimeSettings {
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

    /// Bless the committed `scene_v14.inf_lvl` from [`v14_scene_reference`] under
    /// `INF_BLESS_FIXTURES=1` (inert otherwise). Never hand-edit the committed
    /// bytes.
    #[test]
    fn bless_scene_v14_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v14_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 14);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v14.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed scene_v14 fixture: {}", path.display());
    }

    /// The committed schema-v14 fixture — written by the **pre-v15 codec**, before
    /// every terrain tile grew its erosion data-map layer — still decodes here,
    /// with the v14 content preserved verbatim and every tile's maps at the
    /// never-eroded default. The "old bytes load forever" gate for the v15 bump.
    #[test]
    fn v14_loads_and_lifts_the_data_maps() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v14.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v14 fixture present");
        assert_eq!(bytes[0], 14, "fixture is a genuine schema-v14 payload");
        // Reproducibility lock: the frozen v14 writer still emits those exact bytes.
        let rebuilt =
            bincode::serde::encode_to_vec(v14_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "committed v14 fixture matches the frozen writer"
        );

        let level = RuntimeLevel::decode(&bytes).expect("v14 fixture decodes");
        assert_eq!(level.title, "V14 Fixture Level");
        assert_eq!(level.entities.len(), 3);
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // The v14 content survives the frozen-record hop intact …
        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(uuid::Uuid::from_u128(0xD0A1))
        );
        assert_eq!(by_name("Sun").light.unwrap().intensity, 2.0);
        assert_eq!(level.settings.sim_hz, 90.0);

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
        }
    }

    /// The **wire cost** of the v15 addition, priced rather than asserted by
    /// diffing files: an un-eroded tile grows by exactly one byte (the
    /// zero-length count of the sparse `maps` sequence), and an eroded one by its
    /// dense buffer.
    ///
    /// Priced **frozen against frozen** ([`TerrainV14`] vs [`TerrainV15`]) rather
    /// than against the live component. The live type is a moving target — v16
    /// grew it again — and a v15-vs-live measurement would silently start pricing
    /// every later addition too. The frozen pair is what v15 actually was.
    #[test]
    fn v15_costs_one_byte_per_un_eroded_tile() {
        let frozen = v14_fixture_terrain();
        let tiles = frozen.data.tiles.len();
        assert_eq!(tiles, 2);
        let v14_bytes = bincode::serde::encode_to_vec(&frozen, bincode_config()).unwrap();
        let live = frozen.clone().into_current();
        let v15_bytes =
            bincode::serde::encode_to_vec(TerrainV15::from_current(live.clone()), bincode_config())
                .unwrap();
        assert_eq!(
            v15_bytes.len(),
            v14_bytes.len() + tiles,
            "an un-eroded terrain must cost exactly one extra byte per tile at v15"
        );

        // An eroded tile pays its dense buffer, and the round trip is exact.
        let mut eroded = live;
        eroded
            .data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_map_texel(4, 1, 1, [7.5, 0.5, 2.25]);
        let dense =
            bincode::serde::encode_to_vec(TerrainV15::from_current(eroded), bincode_config())
                .unwrap();
        assert_eq!(
            dense.len(),
            v15_bytes.len() + 4 * 4 * inf_terrain::DATA_MAP_CHANNELS * 4,
            "a materialized tile costs exactly its dense buffer"
        );
        let (back, _): (TerrainV15, usize) =
            bincode::serde::decode_from_slice(&dense, bincode_config()).unwrap();
        assert_eq!(
            back.into_current()
                .data
                .get_tile((0, 0))
                .unwrap()
                .map_texel(4, 1, 1),
            [7.5, 0.5, 2.25]
        );
    }

    // ── schema v16 (P19.2 biome ids) ──────────────────────────────────────

    /// An all-`None` frozen v15 entity — the struct-update base for
    /// [`v15_scene_reference`]. Built through the downgrade hop so the field list
    /// can never drift from the live record.
    fn v15_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV15 {
        EntityRecordV15::from_current(v9_rec(guid, name, parent).into_runtime())
    }

    /// The **v15** terrain the fixture carries: two shared-edge tiles from one
    /// **polynomial** height field (never `sin`/`cos` — the P14 bit-portability
    /// law), a painted splat sample, a **materialized erosion data map** (the one
    /// thing v15 could express that v14 could not), a non-default macro variation
    /// and an asset reference. Its tiles have **no** biome ids and it has no
    /// `biome_set`, because v15 could express neither; that is exactly what the
    /// v16 lift has to reproduce — while the data maps must survive untouched, or
    /// the fixture would only be re-proving the v15 bump.
    ///
    /// Spelled out in **literals**, and identical to the editor codec's
    /// `v15_fixture_terrain` — the two committed fixtures are byte-compared by
    /// the editor's `v15_fixture_matches_the_runtime_codecs_copy`.
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

    /// A representative frozen schema-v15 scene — the provenance source for the
    /// committed `scene_v15.inf_lvl`. Carries a terrain (the thing v16 changed)
    /// plus a mesh and a light, so the pre-v16 entity byte layout is pinned by
    /// committed bytes.
    fn v15_scene_reference() -> SceneFileV15 {
        use inf_ecs::components::Primitive;
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
                    material: Some(MaterialV21::default()),
                    ..v15_rec(g(0xE001), "Cube", None)
                },
                EntityRecordV15 {
                    terrain: Some(v15_fixture_terrain()),
                    ..v15_rec(g(0xE002), "Terrain", None)
                },
                EntityRecordV15 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v15_rec(g(0xE003), "Sun", None)
                },
            ],
            settings: RuntimeSettings {
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

    /// Bless the committed `scene_v15.inf_lvl` from [`v15_scene_reference`] under
    /// `INF_BLESS_FIXTURES=1` (inert otherwise). Never hand-edit the committed
    /// bytes.
    #[test]
    fn bless_scene_v15_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v15_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 15);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v15.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed scene_v15 fixture: {}", path.display());
    }

    /// The committed schema-v15 fixture — written by the **pre-v16 codec**, before
    /// every terrain tile grew its per-sample biome-id layer — still decodes here,
    /// with the v15 content (erosion data maps included) preserved verbatim, every
    /// tile's biome ids at the unpainted default, and no biome vocabulary. The
    /// "old bytes load forever" gate for the v16 bump.
    #[test]
    fn v15_loads_and_lifts_the_biomes() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v15.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v15 fixture present");
        assert_eq!(bytes[0], 15, "fixture is a genuine schema-v15 payload");
        // Reproducibility lock: the frozen v15 writer still emits those exact bytes.
        let rebuilt =
            bincode::serde::encode_to_vec(v15_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "committed v15 fixture matches the frozen writer"
        );

        let level = RuntimeLevel::decode(&bytes).expect("v15 fixture decodes");
        assert_eq!(level.title, "V15 Fixture Level");
        assert_eq!(level.entities.len(), 3);
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // The v15 content survives the frozen-record hop intact …
        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(uuid::Uuid::from_u128(0xE0A1))
        );
        assert_eq!(by_name("Sun").light.unwrap().intensity, 2.0);
        assert_eq!(level.settings.sim_hz, 90.0);

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

    /// The **wire cost** of the v16 addition, priced rather than asserted by
    /// diffing files. Two independent contributions, both measured here:
    ///
    /// * one byte *per tile* — the zero-length count of the sparse `biomes`
    ///   sequence, exactly as v15 charged for `maps`; and
    /// * the component's own new `biome_set: Option<Uuid>`, which at `None` is a
    ///   bare bincode `Option` discriminant. Priced separately below, on a
    ///   **tile-less** terrain, so the two contributions cannot mask each other.
    ///
    /// A painted tile then pays `res²` bytes — one `u8` per sample, *not* `×4`
    /// like the splat weights and not `×4×channels` like the erosion maps.
    #[test]
    fn v16_costs_one_byte_per_unpainted_tile() {
        // Price `biome_set` alone: with no tiles in play, the whole delta between
        // the frozen and live shapes is the component's new tail field.
        let bare_v15 = TerrainV15::from_current(Terrain::configured(4, 2.0));
        assert!(bare_v15.data.tiles.is_empty());
        let biome_set_cost =
            bincode::serde::encode_to_vec(bare_v15.clone().into_current(), bincode_config())
                .unwrap()
                .len()
                - bincode::serde::encode_to_vec(&bare_v15, bincode_config())
                    .unwrap()
                    .len();

        let frozen = v15_fixture_terrain();
        let tiles = frozen.data.tiles.len();
        assert_eq!(tiles, 2);
        let v15_bytes = bincode::serde::encode_to_vec(&frozen, bincode_config()).unwrap();
        let live = frozen.clone().into_current();
        let v16_bytes = bincode::serde::encode_to_vec(&live, bincode_config()).unwrap();
        assert_eq!(
            v16_bytes.len(),
            v15_bytes.len() + tiles + biome_set_cost,
            "an unpainted terrain must cost exactly one extra byte per tile ({tiles}) \
             for the empty biome sequence, plus {biome_set_cost} for `biome_set: None`"
        );

        // A painted tile pays its dense buffer — one `u8` per sample — and the
        // round trip is exact.
        let mut painted = live;
        painted
            .data
            .get_tile_mut((0, 0))
            .unwrap()
            .set_biome_sample(4, 1, 1, 3);
        let dense = bincode::serde::encode_to_vec(&painted, bincode_config()).unwrap();
        assert_eq!(
            dense.len(),
            v16_bytes.len() + 4 * 4,
            "a painted tile costs exactly its dense res² buffer of u8 biome ids"
        );
        let (back, _): (Terrain, usize) =
            bincode::serde::decode_from_slice(&dense, bincode_config()).unwrap();
        assert_eq!(back.data.get_tile((0, 0)).unwrap().biome_sample(4, 1, 1), 3);
        assert_eq!(
            back.data.get_tile((0, 0)).unwrap().biome_sample(4, 0, 0),
            inf_terrain::UNASSIGNED_BIOME
        );
    }

    /// **The pin's other half: the wire really is generation 3.** The assertion
    /// above says the scene schema has not moved; this says what that costs, in
    /// the only currency that matters — a level's terrain bytes.
    ///
    /// A carved tile and an un-carved one must serialize **identically** through
    /// this codec, because generation 3 has no hole field. That is a deliberate,
    /// stated loss (see the table above), and it is pinned here so that whoever
    /// eventually bumps to v20 finds a failing test naming exactly which promise
    /// they are then allowed to keep — rather than discovering the silence by
    /// shipping it twice.
    #[test]
    fn terrain_data_wire_is_pinned_at_generation_three() {
        let plain = fixture_terrain();
        let mut carved = plain.clone();
        {
            let tile = carved.data.get_tile_mut((0, 0)).unwrap();
            tile.set_hole(4, 1, 1, true);
            assert!(tile.has_holes(), "the fixture must actually be carved");
        }
        assert_ne!(
            plain.data.get_tile((0, 0)).unwrap().holes_len(),
            carved.data.get_tile((0, 0)).unwrap().holes_len(),
            "the two tiles must differ in memory, or this proves nothing"
        );

        let a = bincode::serde::encode_to_vec(&plain, bincode_config()).unwrap();
        let b = bincode::serde::encode_to_vec(&carved, bincode_config()).unwrap();
        assert_eq!(
            a, b,
            "the scene wire grew a hole layer — that is a v20, not a v19"
        );

        // … and the round trip comes back un-carved, which is the honest shape of
        // the loss: not corrupted, not half-written. Just absent.
        let (back, _): (Terrain, usize) =
            bincode::serde::decode_from_slice(&b, bincode_config()).unwrap();
        assert!(
            back.data.get_tile((0, 0)).unwrap().holes_are_default(),
            "a generation-3 wire cannot bring holes back, and must not pretend to"
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
    /// | `TerrainTileFrozenV3` | + per-sample biome ids | v16..=v20 | v4 |
    /// | live `TerrainTile` | + the P21.2 hole mask | *(none)* | v5+ |
    ///
    /// **The last row's `.inf_lvl` cell is empty on purpose.** P21.2 gave tiles a
    /// hole mask while the scene schema was already frozen at v19 (Phase 21 spent
    /// its one bump on the P21.1 voxel volume), and bincode is positional — a
    /// sixth tile field in this stream would be a v20. So `TerrainData`'s wire
    /// form is pinned at generation 3 and an `.inf_lvl` does not carry holes; the
    /// `.inf_terrain` asset does, and that is the container every carve tool
    /// targets. `inf_terrain::TerrainTileFrozenV1`'s generation table states the
    /// consequence in full under *THE EMPTY CELL*.
    ///
    /// `inf-terrain` carries the asset half of this assertion
    /// (`frozen_tile_generations_are_pinned_to_both_ladders`); this is the scene
    /// half, and the editor codec mirrors it.
    #[test]
    fn the_frozen_tile_generation_covers_this_schema() {
        assert_eq!(
            SCHEMA_VERSION, 23,
            "the scene schema moved. Generation-1 frozen tiles (TerrainTileFrozenV1, via \
             TerrainV14) cover .inf_lvl v1..=v14, generation-2 (TerrainTileFrozenV2, via \
             TerrainV15) covers v15, and generation-3 (TerrainTileFrozenV3, which \
             TerrainData's own wire form is pinned at) covers v16+. If the TILE layout \
             changed again, add inf_terrain::TerrainTileFrozenV4 and a new frozen Terrain \
             record; if only the scene changed, update this pin and TerrainTileFrozenV1's \
             generation table. (v17, v18 and v19 are all the latter case: each appended an \
             entity slot and left every tile layout alone — v19's voxel volume in \
             particular extends the ground LOCALLY, out in its own .inf_voxel, and does \
             not touch a single heightfield tile. P21.2's hole mask DID grow the tile, and \
             the scene answered by pinning rather than bumping — see the table above. v22 \
             is the latter case too: it grew the MATERIAL component, which no tile has \
             ever contained.)"
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

    // \u2500\u2500 v17 water (P20.1) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

    /// What the v17 `water_body` slot costs an entity that has no water: the
    /// `Option` discriminant, and nothing else. Named once so the three older
    /// wire-price tests price it by reference rather than by re-deriving `1`.
    const WATER_SLOT_BYTES: usize = 1;

    /// **The v17 price, isolated.** An entity with no water pays exactly one
    /// discriminant byte; an entity *with* water pays the component.
    ///
    /// Priced as a delta between two otherwise-identical levels rather than as an
    /// absolute number, because the absolute number would silently absorb any
    /// other growth and stop being a price at all.
    #[test]
    fn v17_costs_one_byte_per_water_free_entity() {
        use inf_ecs::components::{WaterBody, WaterKind};

        let dry = RuntimeLevel {
            title: "dry".into(),
            entities: vec![v9_rec(uuid::Uuid::from_u128(1), "A", None).into_runtime()],
            settings: RuntimeSettings::default(),
        };
        let mut wet = dry.clone();
        wet.entities[0].water_body = Some(WaterBody::default());

        let dry_bytes = encode(&dry).unwrap();
        let wet_bytes = encode(&wet).unwrap();
        assert_eq!(dry_bytes[0], SCHEMA_VERSION as u8);
        assert!(
            wet_bytes.len() > dry_bytes.len() + WATER_SLOT_BYTES,
            "a water body must cost more than its own discriminant"
        );

        // The `None` half of the price, measured between the frozen v16 and
        // frozen v17 shapes of the very same record: exactly one byte per entity,
        // which is what the three older wire-price tests reference. Priced
        // between the two *frozen* shapes rather than against the live one, so
        // v18's own slot cannot be absorbed into v17's price.
        let v16 = SceneFileV16 {
            schema_version: 16,
            title: dry.title.clone(),
            entities: dry
                .entities
                .iter()
                .cloned()
                .map(EntityRecordV16::from_current)
                .collect(),
            settings: dry.settings,
        };
        let v17 = SceneFileV17 {
            schema_version: 17,
            title: dry.title.clone(),
            entities: dry
                .entities
                .iter()
                .cloned()
                .map(EntityRecordV17::from_current)
                .collect(),
            settings: dry.settings,
        };
        let v16_bytes = bincode::serde::encode_to_vec(&v16, bincode_config()).unwrap();
        let v17_bytes = bincode::serde::encode_to_vec(&v17, bincode_config()).unwrap();
        assert_eq!(
            v17_bytes.len(),
            v16_bytes.len() + WATER_SLOT_BYTES,
            "the v17 slot must cost exactly one discriminant byte on a dry entity"
        );

        // …and the whole component round-trips through the live codec.
        let back = decode(&wet_bytes).unwrap();
        assert_eq!(back.entities[0].water_body, Some(WaterBody::default()));
        assert_eq!(back.entities[0].water_body.unwrap().kind, WaterKind::Ocean);
    }

    /// An all-`None` frozen v16 entity \u2014 the struct-update base for
    /// [`v16_scene_reference`]. Built through the downgrade hop so the field list
    /// can never drift from the live record.
    fn v16_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV16 {
        EntityRecordV16::from_current(v9_rec(guid, name, parent).into_runtime())
    }

    /// The **v16** terrain the fixture carries: two authored tiles, a painted
    /// splat sample, a materialized erosion data map, **a painted biome id and a
    /// `biome_set` reference** (the two things v16 could express that v15 could
    /// not), a non-default macro variation and an asset reference \u2014 so the v17 hop
    /// is proven to preserve what v16 authored, not merely to produce defaults.
    ///
    /// The literals must match the editor codec's `v16_fixture_terrain` exactly:
    /// the two committed fixtures are byte-compared by the editor's
    /// `v16_fixture_matches_the_runtime_codecs_copy`, which is the whole point of
    /// writing them twice.
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
        t.asset = Some(Uuid::from_u128(0xF_00AA));
        t.biome_set = Some(Uuid::from_u128(0xF_00BB));
        t
    }

    /// Rebuild the exact schema-v16 file the committed v16 fixture was generated
    /// from, out of the frozen v16 record types (the provenance lock).
    fn v16_scene_reference() -> SceneFileV16 {
        use inf_ecs::components::Primitive;
        let g = Uuid::from_u128;
        SceneFileV16 {
            schema_version: 16,
            title: "V16 Fixture Level".into(),
            entities: vec![
                EntityRecordV16 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xF0A1)),
                    }),
                    material: Some(MaterialV21::default()),
                    ..v16_rec(g(0xF001), "Cube", None)
                },
                EntityRecordV16 {
                    terrain: Some(v16_fixture_terrain()),
                    ..v16_rec(g(0xF002), "Terrain", None)
                },
                EntityRecordV16 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v16_rec(g(0xF003), "Sun", None)
                },
            ],
            settings: RuntimeSettings {
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

    /// Regenerate `tests/fixtures/scene_v16.inf_lvl` \u2014 the **downgrade-bless**
    /// path, walked only under `INF_BLESS_FIXTURES=1` (exactly `1`, matching this
    /// crate's other bless guards).
    #[test]
    fn bless_scene_v16_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v16_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 16);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v16.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed {} ({} bytes)", path.display(), bytes.len());
    }

    /// A committed **v16** payload still loads, keeps everything v16 could
    /// express, and lifts with **no water** \u2014 which is exactly what a v16 level
    /// was. The "old bytes load forever" gate for the v17 bump.
    #[test]
    fn v16_loads_and_lifts_without_water() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v16.inf_lvl"),
        )
        .expect("committed v16 fixture present");
        assert_eq!(bytes[0], 16, "the fixture must really be v16");
        let level = decode(&bytes).unwrap();
        assert_eq!(level.title, "V16 Fixture Level");
        assert_eq!(level.entities.len(), 3);
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // The v16 content survives the frozen-record hop intact \u2026
        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(Uuid::from_u128(0xF0A1))
        );
        assert_eq!(by_name("Sun").light.unwrap().intensity, 2.0);
        assert_eq!(level.settings.sim_hz, 90.0);
        let t = by_name("Terrain").terrain.clone().expect("terrain slot");
        assert_eq!(t.biome_set, Some(Uuid::from_u128(0xF_00BB)));
        assert_eq!(t.data.get_tile((0, 0)).unwrap().biome_sample(4, 1, 1), 3);
        assert_eq!(
            t.data.get_tile((0, 0)).unwrap().map_texel(4, 1, 1),
            [7.5, 0.5, 2.25]
        );

        // \u2026 and the one new slot lifts to `None` on every entity.
        for e in &level.entities {
            assert!(e.water_body.is_none(), "a v16 level has no water to lift");
        }

        // Re-encoding writes the current schema and round-trips.
        let re = encode(&level).unwrap();
        assert_eq!(re[0], SCHEMA_VERSION as u8);
        assert_eq!(decode(&re).unwrap(), level);
    }

    #[test]
    fn v16_fixture_is_reproducible_and_genuinely_v16() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v16.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v16 fixture present");
        assert_eq!(bytes[0], 16, "fixture must be a genuine schema-v16 payload");
        let rebuilt =
            bincode::serde::encode_to_vec(v16_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v16 fixture must match our frozen v16 writer"
        );
    }

    // ── v18 buoyancy (P20.2) ──────────────────────────────────────────────

    /// What the v18 `buoyancy` slot costs an entity that does not float: the
    /// `Option` discriminant, and nothing else. Named once so the three older
    /// wire-price tests price it by reference rather than by re-deriving `1`.
    const BUOYANCY_SLOT_BYTES: usize = 1;

    /// An all-`None` frozen v17 entity — the struct-update base for
    /// [`v17_scene_reference`]. Built through the downgrade hop so the field list
    /// can never drift from the live record.
    fn v17_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV17 {
        EntityRecordV17::from_current(v9_rec(guid, name, parent).into_runtime())
    }

    /// The **v17** water body the fixture's river carries: a spline river with a
    /// non-default flow, cross-section, seed and wind — the thing v17 could
    /// express and v16 could not — so the v18 hop is proven to preserve what v17
    /// authored rather than merely to produce defaults.
    ///
    /// The literals must match the editor codec's `v17_fixture_water` exactly:
    /// the two committed fixtures are byte-compared by the editor's
    /// `v17_fixture_matches_the_runtime_codecs_copy`, which is the whole point of
    /// writing them twice.
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
    fn v17_scene_reference() -> SceneFileV17 {
        use inf_ecs::components::Primitive;
        let g = Uuid::from_u128;
        SceneFileV17 {
            schema_version: 17,
            title: "V17 Fixture Level".into(),
            entities: vec![
                EntityRecordV17 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xF1A1)),
                    }),
                    material: Some(MaterialV21::default()),
                    ..v17_rec(g(0xF101), "Cube", None)
                },
                EntityRecordV17 {
                    terrain: Some(v16_fixture_terrain()),
                    ..v17_rec(g(0xF102), "Terrain", None)
                },
                EntityRecordV17 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v17_rec(g(0xF103), "Sun", None)
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
                    ..v17_rec(g(0xF104), "River", None)
                },
            ],
            settings: RuntimeSettings {
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

    /// Regenerate `tests/fixtures/scene_v17.inf_lvl` — the **downgrade-bless**
    /// path, walked only under `INF_BLESS_FIXTURES=1` (exactly `1`, matching this
    /// crate's other bless guards).
    #[test]
    fn bless_scene_v17_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v17_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 17);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v17.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed {} ({} bytes)", path.display(), bytes.len());
    }

    /// A committed **v17** payload still loads, keeps everything v17 could
    /// express — its river's water body included, which is the half a
    /// defaults-only fixture would never have proven — and lifts with **nothing
    /// floating**, which is exactly what a v17 level was. The "old bytes load
    /// forever" gate for the v18 bump.
    #[test]
    fn v17_loads_and_lifts_without_buoyancy() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v17.inf_lvl"),
        )
        .expect("committed v17 fixture present");
        assert_eq!(bytes[0], 17, "the fixture must really be v17");
        let level = decode(&bytes).unwrap();
        assert_eq!(level.title, "V17 Fixture Level");
        assert_eq!(level.entities.len(), 4);
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // The v17 content survives the frozen-record hop intact …
        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(Uuid::from_u128(0xF1A1))
        );
        assert_eq!(by_name("Sun").light.unwrap().intensity, 2.0);
        assert_eq!(level.settings.sim_hz, 90.0);
        let t = by_name("Terrain").terrain.clone().expect("terrain slot");
        assert_eq!(t.biome_set, Some(Uuid::from_u128(0xF_00BB)));
        assert_eq!(t.data.get_tile((0, 0)).unwrap().biome_sample(4, 1, 1), 3);
        let river = by_name("River");
        assert_eq!(river.water_body, Some(v17_fixture_water()));
        assert_eq!(river.spline.as_ref().unwrap().points.len(), 3);

        // … and the one new slot lifts to `None` on every entity.
        for e in &level.entities {
            assert!(
                e.buoyancy.is_none(),
                "a v17 level floats nothing; the lift must not conjure buoyancy"
            );
        }

        // Re-encoding writes the current schema and round-trips — which is also
        // the **v19 decode arm's** only exercise from this fixture.
        let re = encode(&level).unwrap();
        assert_eq!(re[0], SCHEMA_VERSION as u8);
        assert_eq!(decode(&re).unwrap(), level);
    }

    #[test]
    fn v17_fixture_is_reproducible_and_genuinely_v17() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v17.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v17 fixture present");
        assert_eq!(bytes[0], 17, "fixture must be a genuine schema-v17 payload");
        let rebuilt =
            bincode::serde::encode_to_vec(v17_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v17 fixture must match our frozen v17 writer"
        );
    }

    /// **The v18 price, isolated.** An entity that does not float pays exactly one
    /// discriminant byte; an entity *with* buoyancy pays the component.
    ///
    /// Priced as a delta between the frozen v17 and frozen v18 shapes of the very
    /// same record rather than as an absolute number, because the absolute number
    /// would silently absorb any other growth and stop being a price at all —
    /// v19's own slot included, which is why this prices the frozen v18 shape and
    /// not the live record it used to.
    #[test]
    fn v18_costs_one_byte_per_buoyancy_free_entity() {
        let sinks = RuntimeLevel {
            title: "sinks".into(),
            entities: vec![v9_rec(Uuid::from_u128(1), "A", None).into_runtime()],
            settings: RuntimeSettings::default(),
        };
        let mut floats = sinks.clone();
        floats.entities[0].buoyancy = Some(Buoyancy::default());

        let sinks_bytes = encode(&sinks).unwrap();
        let floats_bytes = encode(&floats).unwrap();
        assert_eq!(sinks_bytes[0], SCHEMA_VERSION as u8);
        assert!(
            floats_bytes.len() > sinks_bytes.len() + BUOYANCY_SLOT_BYTES,
            "a buoyancy component must cost more than its own discriminant"
        );

        // The `None` half of the price, measured between the frozen v17 and
        // frozen v18 shapes of the very same record: exactly one byte per entity.
        // Priced between the two *frozen* shapes rather than against the live one,
        // so v19's own slot cannot be absorbed into v18's price.
        let v17 = SceneFileV17 {
            schema_version: 17,
            title: sinks.title.clone(),
            entities: sinks
                .entities
                .iter()
                .cloned()
                .map(EntityRecordV17::from_current)
                .collect(),
            settings: sinks.settings,
        };
        let v18 = SceneFileV18 {
            schema_version: 18,
            title: sinks.title.clone(),
            entities: sinks
                .entities
                .iter()
                .cloned()
                .map(EntityRecordV18::from_current)
                .collect(),
            settings: sinks.settings,
        };
        let v17_bytes = bincode::serde::encode_to_vec(&v17, bincode_config()).unwrap();
        let v18_bytes = bincode::serde::encode_to_vec(&v18, bincode_config()).unwrap();
        assert_eq!(
            v18_bytes.len(),
            v17_bytes.len() + BUOYANCY_SLOT_BYTES,
            "the v18 slot must cost exactly one discriminant byte on a sinking entity"
        );

        // …and the whole component round-trips through the live codec, which is
        // now the v19 decode arm.
        let back = decode(&floats_bytes).unwrap();
        assert_eq!(back.entities[0].buoyancy, Some(Buoyancy::default()));
        assert_eq!(back.entities[0].buoyancy.unwrap().density_kg_m3, 600.0);
    }

    // ── v19 volumetric terrain (P21.1) ────────────────────────────────────

    /// What the v19 `voxel_volume` slot costs an entity with no volume: the
    /// `Option` discriminant, and nothing else. Named once so the older
    /// wire-price tests price it by reference rather than by re-deriving `1`.
    const VOXEL_SLOT_BYTES: usize = 1;

    /// An all-`None` frozen v18 entity — the struct-update base for
    /// [`v18_scene_reference`]. Built through the downgrade hop so the field list
    /// can never drift from the live record.
    fn v18_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV18 {
        EntityRecordV18::from_current(v9_rec(guid, name, parent).into_runtime())
    }

    /// The **v18** buoyancy the fixture's raft carries: non-default in four of the
    /// five fields — the thing v18 could express and v17 could not — so the v19
    /// hop is proven to preserve what v18 authored rather than merely to produce
    /// defaults.
    ///
    /// The literals must match the editor codec's `v18_fixture_buoyancy` exactly:
    /// the two committed fixtures are byte-compared by the editor's
    /// `v18_fixture_matches_the_runtime_codecs_copy`, which is the whole point of
    /// writing them twice.
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
    fn v18_scene_reference() -> SceneFileV18 {
        use inf_ecs::components::Primitive;
        let g = Uuid::from_u128;
        SceneFileV18 {
            schema_version: 18,
            title: "V18 Fixture Level".into(),
            entities: vec![
                EntityRecordV18 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xF2A1)),
                    }),
                    material: Some(MaterialV21::default()),
                    ..v18_rec(g(0xF201), "Cube", None)
                },
                EntityRecordV18 {
                    terrain: Some(v16_fixture_terrain()),
                    ..v18_rec(g(0xF202), "Terrain", None)
                },
                EntityRecordV18 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v18_rec(g(0xF203), "Sun", None)
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
                    ..v18_rec(g(0xF204), "River", None)
                },
                // The raft: the dynamic body and the opt-in flotation that v18
                // added, which is the content this fixture exists to carry.
                EntityRecordV18 {
                    rigid_body_3d: Some(RigidBody3D::default()),
                    buoyancy: Some(v18_fixture_buoyancy()),
                    ..v18_rec(g(0xF205), "Raft", None)
                },
            ],
            settings: RuntimeSettings {
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

    /// Regenerate `tests/fixtures/scene_v18.inf_lvl` — the **downgrade-bless**
    /// path, walked only under `INF_BLESS_FIXTURES=1` (exactly `1`, matching this
    /// crate's other bless guards).
    #[test]
    fn bless_scene_v18_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v18_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 18);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v18.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed {} ({} bytes)", path.display(), bytes.len());
    }

    /// A committed **v18** payload still loads, keeps everything v18 could
    /// express — its raft's buoyancy included, which is the half a defaults-only
    /// fixture would never have proven — and lifts with **no voxel volume**, which
    /// is exactly what a v18 level was. The "old bytes load forever" gate for the
    /// v19 bump.
    #[test]
    fn v18_loads_and_lifts_without_a_voxel_volume() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v18.inf_lvl"),
        )
        .expect("committed v18 fixture present");
        assert_eq!(bytes[0], 18, "the fixture must really be v18");
        let level = decode(&bytes).unwrap();
        assert_eq!(level.title, "V18 Fixture Level");
        assert_eq!(level.entities.len(), 5);
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // The v18 content survives the frozen-record hop intact …
        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(Uuid::from_u128(0xF2A1))
        );
        assert_eq!(by_name("Sun").light.unwrap().intensity, 2.0);
        assert_eq!(level.settings.sim_hz, 90.0);
        let t = by_name("Terrain").terrain.clone().expect("terrain slot");
        assert_eq!(t.biome_set, Some(Uuid::from_u128(0xF_00BB)));
        assert_eq!(t.data.get_tile((0, 0)).unwrap().biome_sample(4, 1, 1), 3);
        let river = by_name("River");
        assert_eq!(river.water_body, Some(v17_fixture_water()));
        assert_eq!(river.spline.as_ref().unwrap().points.len(), 3);
        let raft = by_name("Raft");
        assert_eq!(raft.buoyancy, Some(v18_fixture_buoyancy()));
        assert!(raft.rigid_body_3d.is_some());

        // … and the one new slot lifts to `None` on every entity.
        for e in &level.entities {
            assert!(
                e.voxel_volume.is_none(),
                "a v18 level has no volumetric ground; the lift must not conjure any"
            );
        }

        // Re-encoding writes the current schema and round-trips.
        let re = encode(&level).unwrap();
        assert_eq!(re[0], SCHEMA_VERSION as u8);
        assert_eq!(decode(&re).unwrap(), level);
    }

    #[test]
    fn v18_fixture_is_reproducible_and_genuinely_v18() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v18.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v18 fixture present");
        assert_eq!(bytes[0], 18, "fixture must be a genuine schema-v18 payload");
        let rebuilt =
            bincode::serde::encode_to_vec(v18_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v18 fixture must match our frozen v18 writer"
        );
    }

    /// **The v19 price, isolated.** An entity with no volumetric ground pays
    /// exactly one discriminant byte; an entity *with* a volume pays the
    /// component — three fields, and never the chunks, which live in the
    /// `.inf_voxel` the GUID points at.
    ///
    /// Priced as a delta against the frozen v18 shape of the very same record
    /// rather than as an absolute number, because the absolute number would
    /// silently absorb any other growth and stop being a price at all.
    #[test]
    fn v19_costs_one_byte_per_voxel_free_entity() {
        let solid = RuntimeLevel {
            title: "solid".into(),
            entities: vec![v9_rec(Uuid::from_u128(1), "A", None).into_runtime()],
            settings: RuntimeSettings::default(),
        };
        let mut carved = solid.clone();
        carved.entities[0].voxel_volume = Some(VoxelVolume {
            asset: Some(Uuid::from_u128(0xF_0CA5)),
            voxel_size_m: 0.25,
            runtime_carve: false,
        });

        let solid_bytes = encode(&solid).unwrap();
        let carved_bytes = encode(&carved).unwrap();
        assert_eq!(solid_bytes[0], SCHEMA_VERSION as u8);
        assert!(
            carved_bytes.len() > solid_bytes.len() + VOXEL_SLOT_BYTES,
            "a voxel volume must cost more than its own discriminant"
        );

        // The `None` half of the price, measured against the frozen v18 shape of
        // the very same record: exactly one byte per entity.
        let v18 = SceneFileV18 {
            schema_version: 18,
            title: solid.title.clone(),
            entities: solid
                .entities
                .iter()
                .cloned()
                .map(EntityRecordV18::from_current)
                .collect(),
            settings: solid.settings,
        };
        let v18_bytes = bincode::serde::encode_to_vec(&v18, bincode_config()).unwrap();
        // Measured between the two FROZEN rungs, not against the live encoding:
        // v20 appended a slot of its own, so `solid_bytes` now carries BOTH
        // bumps' discriminants and would misreport v19's price as two bytes.
        let v19 = SceneFileV19 {
            schema_version: 19,
            title: solid.title.clone(),
            entities: solid
                .entities
                .iter()
                .cloned()
                .map(EntityRecordV19::from_current)
                .collect(),
            settings: solid.settings,
        };
        let v19_bytes = bincode::serde::encode_to_vec(&v19, bincode_config()).unwrap();
        assert_eq!(
            v19_bytes.len(),
            v18_bytes.len() + VOXEL_SLOT_BYTES,
            "the v19 slot must cost exactly one discriminant byte on a solid entity"
        );

        // …and the whole component round-trips through the live codec.
        let back = decode(&carved_bytes).unwrap();
        let v = back.entities[0]
            .voxel_volume
            .expect("voxel volume survives");
        assert_eq!(v.asset, Some(Uuid::from_u128(0xF_0CA5)));
        assert_eq!(v.voxel_size_m, 0.25);
        assert!(!v.runtime_carve);
    }

    // ── v20 destruction (P22.2) ───────────────────────────────────────────

    /// What the v20 `destructible` slot costs an entity that cannot break: the
    /// `Option` discriminant, and nothing else.
    const DESTRUCTIBLE_SLOT_BYTES: usize = 1;

    /// An all-`None` frozen v19 entity — the struct-update base for
    /// [`v19_scene_reference`]. Built through the downgrade hop so the field list
    /// can never drift from the live record.
    fn v19_rec(guid: Uuid, name: &str, parent: Option<Uuid>) -> EntityRecordV19 {
        EntityRecordV19::from_current(v9_rec(guid, name, parent).into_runtime())
    }

    /// The **v19** voxel volume the fixture's cavern carries: non-default in all
    /// three fields — the thing v19 could express and v18 could not — so the v20
    /// hop is proven to preserve what v19 authored rather than merely to produce
    /// defaults.
    ///
    /// The literals must match the editor codec's `v19_fixture_volume` exactly:
    /// the two committed fixtures are byte-compared by the editor's
    /// `v19_fixture_matches_the_runtime_codecs_copy`, which is the whole point of
    /// writing them twice.
    fn v19_fixture_volume() -> VoxelVolume {
        VoxelVolume {
            asset: Some(Uuid::from_u128(0xF_0CA5)),
            voxel_size_m: 0.25,
            runtime_carve: false,
        }
    }

    /// Rebuild the exact schema-v19 file the committed v19 fixture was generated
    /// from, out of the frozen v19 record types (the provenance lock). Carries
    /// the v16 terrain, the v17 river and the v18 raft unchanged — v20 touched
    /// neither a tile layout nor a component's shape — plus the cavern entity
    /// that only v19 could write.
    fn v19_scene_reference() -> SceneFileV19 {
        use inf_ecs::components::Primitive;
        let g = Uuid::from_u128;
        SceneFileV19 {
            schema_version: 19,
            title: "V19 Fixture Level".into(),
            entities: vec![
                EntityRecordV19 {
                    mesh: Some(MeshRef {
                        primitive: Primitive::Cube,
                        asset: Some(g(0xF2A1)),
                    }),
                    material: Some(MaterialV21::default()),
                    ..v19_rec(g(0xF201), "Cube", None)
                },
                EntityRecordV19 {
                    terrain: Some(v16_fixture_terrain()),
                    ..v19_rec(g(0xF202), "Terrain", None)
                },
                EntityRecordV19 {
                    light: Some(Light {
                        kind: LightKind::Directional,
                        color: Color::WHITE,
                        intensity: 2.0,
                        ..Default::default()
                    }),
                    ..v19_rec(g(0xF203), "Sun", None)
                },
                EntityRecordV19 {
                    spline: Some(Spline {
                        points: vec![
                            Vec3d::new(0.0, 0.0, 0.0),
                            Vec3d::new(10.0, 0.0, 4.0),
                            Vec3d::new(18.0, 0.0, 14.0),
                        ],
                        ..Spline::default()
                    }),
                    water_body: Some(v17_fixture_water()),
                    ..v19_rec(g(0xF204), "River", None)
                },
                EntityRecordV19 {
                    rigid_body_3d: Some(RigidBody3D::default()),
                    buoyancy: Some(v18_fixture_buoyancy()),
                    ..v19_rec(g(0xF205), "Raft", None)
                },
                // The cavern: the volumetric ground that only v19 could write,
                // which is the content this fixture exists to carry.
                EntityRecordV19 {
                    voxel_volume: Some(v19_fixture_volume()),
                    ..v19_rec(g(0xF206), "Cavern", None)
                },
            ],
            settings: RuntimeSettings {
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

    /// Regenerate `tests/fixtures/scene_v19.inf_lvl` — the **downgrade-bless**
    /// path, walked only under `INF_BLESS_FIXTURES=1`.
    #[test]
    fn bless_scene_v19_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v19_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 19);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v19.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed {} ({} bytes)", path.display(), bytes.len());
    }

    /// A committed **v19** payload still loads, keeps everything v19 could
    /// express — its cavern's volume included — and lifts with **nothing
    /// destructible**, which is exactly what a v19 level was. The "old bytes load
    /// forever" gate for the v20 bump.
    #[test]
    fn v19_loads_and_lifts_without_a_destructible() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v19.inf_lvl"),
        )
        .expect("committed v19 fixture present");
        assert_eq!(bytes[0], 19, "the fixture must really be v19");
        let level = decode(&bytes).unwrap();
        assert_eq!(level.title, "V19 Fixture Level");
        assert_eq!(level.entities.len(), 6);
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        assert_eq!(
            by_name("Cube").mesh.unwrap().asset,
            Some(Uuid::from_u128(0xF2A1))
        );
        assert_eq!(by_name("Sun").light.unwrap().intensity, 2.0);
        assert_eq!(level.settings.sim_hz, 90.0);
        let t = by_name("Terrain").terrain.clone().expect("terrain slot");
        assert_eq!(t.biome_set, Some(Uuid::from_u128(0xF_00BB)));
        assert_eq!(t.data.get_tile((0, 0)).unwrap().biome_sample(4, 1, 1), 3);
        assert_eq!(by_name("River").water_body, Some(v17_fixture_water()));
        assert_eq!(by_name("Raft").buoyancy, Some(v18_fixture_buoyancy()));
        assert_eq!(by_name("Cavern").voxel_volume, Some(v19_fixture_volume()));

        // … and the one new slot lifts to `None` on every entity.
        for e in &level.entities {
            assert!(
                e.destructible.is_none(),
                "a v19 level has nothing destructible; the lift must not conjure any"
            );
        }

        // Re-encoding writes the current schema and round-trips.
        let re = encode(&level).unwrap();
        assert_eq!(re[0], SCHEMA_VERSION as u8);
        assert_eq!(decode(&re).unwrap(), level);
    }

    #[test]
    fn v19_fixture_is_reproducible_and_genuinely_v19() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v19.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v19 fixture present");
        assert_eq!(bytes[0], 19, "fixture must be a genuine schema-v19 payload");
        let rebuilt =
            bincode::serde::encode_to_vec(v19_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v19 fixture must match our frozen v19 writer"
        );
    }

    /// **The v20 price, isolated.** An entity that cannot break pays exactly one
    /// discriminant byte; an entity that can pays the component — five fields,
    /// and never the chunks, which live in the derived `.inf_fracture`.
    ///
    /// Priced as a delta against the frozen v19 shape of the very same record
    /// rather than as an absolute number, because the absolute would silently
    /// absorb any other growth and stop being a price at all.
    #[test]
    fn v20_costs_one_byte_per_indestructible_entity() {
        let solid = RuntimeLevel {
            title: "solid".into(),
            entities: vec![v9_rec(Uuid::from_u128(1), "A", None).into_runtime()],
            settings: RuntimeSettings::default(),
        };
        let mut breakable = solid.clone();
        breakable.entities[0].destructible = Some(Destructible {
            fracture_seed: 9,
            chunk_count: 24,
            strength: 1.2e7,
            density_kg_m3: 1900.0,
            runtime_destruct: false,
        });

        let solid_bytes = encode(&solid).unwrap();
        let breakable_bytes = encode(&breakable).unwrap();
        assert_eq!(solid_bytes[0], SCHEMA_VERSION as u8);
        assert!(
            breakable_bytes.len() > solid_bytes.len() + DESTRUCTIBLE_SLOT_BYTES,
            "a destructible must cost more than its own discriminant"
        );

        let v19 = SceneFileV19 {
            schema_version: 19,
            title: solid.title.clone(),
            entities: solid
                .entities
                .iter()
                .cloned()
                .map(EntityRecordV19::from_current)
                .collect(),
            settings: solid.settings,
        };
        let v19_bytes = bincode::serde::encode_to_vec(&v19, bincode_config()).unwrap();
        // **Measured between the two FROZEN rungs, not against the live
        // encoding** — the P22.2 lesson, met again the moment v21 landed. Against
        // `solid_bytes` this asserted 258 == 255: v21's three appended slots were
        // being reported as part of v20's price, which is exactly the way a
        // wire-price test stops measuring a price and starts measuring a total.
        let v20 = SceneFileV20 {
            schema_version: 20,
            title: solid.title.clone(),
            entities: solid
                .entities
                .iter()
                .cloned()
                .map(EntityRecordV20::from_current)
                .collect(),
            settings: solid.settings,
        };
        let v20_bytes = bincode::serde::encode_to_vec(&v20, bincode_config()).unwrap();
        assert_eq!(
            v20_bytes.len(),
            v19_bytes.len() + DESTRUCTIBLE_SLOT_BYTES,
            "the v20 slot must cost exactly one discriminant byte on a solid entity"
        );
        // …and the live encoding is exactly the v20 rung plus v21's three, which
        // is what makes the line above a price rather than a coincidence.
        assert_eq!(
            solid_bytes.len(),
            v20_bytes.len() + CHARACTER_SLOT_BYTES + MOVEMENT_SLOT_BYTES,
            "the v21 slots must cost exactly three discriminant bytes, and v23's one"
        );

        // …and the whole component round-trips through the live codec, which is
        // now the v20 decode arm.
        let back = decode(&breakable_bytes).unwrap();
        let d = back.entities[0]
            .destructible
            .expect("destructible survives");
        assert_eq!(d.fracture_seed, 9);
        assert_eq!(d.chunk_count, 24);
        assert_eq!(d.strength, 1.2e7);
        assert_eq!(d.density_kg_m3, 1900.0);
        assert!(!d.runtime_destruct);
    }

    // ── v21 character components (P24.3) ──────────────────────────────────

    /// What the three v21 slots cost an entity that has no IK, no garment and no
    /// hair: three `Option` discriminants, and nothing else.
    const CHARACTER_SLOT_BYTES: usize = 3;

    /// How many entities of an encoded level carry a `Material` — the multiplier
    /// for [`MATERIAL_BINDING_BYTES`] in the three lift-price arms.
    ///
    /// Counted rather than assumed, because v22's field lives **inside** a
    /// component: an entity with no material pays nothing for it, so `entities`
    /// is the wrong multiplier and would have made those arms wrong the moment a
    /// fixture gained a light-only entity.
    fn materialed_entities(bytes: &[u8]) -> usize {
        RuntimeLevel::decode(bytes)
            .unwrap()
            .entities
            .iter()
            .filter(|e| e.material.is_some())
            .count()
    }

    /// The **v20** destructible the fixture's wall carries: every field away from
    /// its default, so a hop that produced defaults would be caught.
    ///
    /// The literals must match the editor codec's `v20_fixture_destructible`
    /// exactly — the two committed fixtures are byte-compared by the editor's
    /// `v20_fixture_matches_the_runtime_codecs_copy`, which is the whole point of
    /// writing them twice.
    fn v20_fixture_destructible() -> Destructible {
        Destructible {
            fracture_seed: 9,
            chunk_count: 24,
            strength: 1.2e7,
            density_kg_m3: 1900.0,
            runtime_destruct: false,
        }
    }

    /// Rebuild the exact schema-v20 file the committed v20 fixture was generated
    /// from, out of the frozen v20 record type (the provenance lock).
    ///
    /// Built by lifting [`v19_scene_reference`] one rung and then authoring the
    /// slot v20 added, so the field list can never drift from the ladder: a slot
    /// appended later shows up here as a compile error rather than as a fixture
    /// that quietly stopped covering it.
    fn v20_scene_reference() -> SceneFileV20 {
        let v19 = v19_scene_reference();
        let mut entities: Vec<EntityRecordV20> = v19
            .entities
            .into_iter()
            .map(EntityRecordV19::into_v20)
            .collect();
        // The wall is what only v20 could write.
        let wall = entities
            .iter_mut()
            .find(|e| e.name == "Cube")
            .expect("the v19 fixture has a Cube");
        wall.destructible = Some(v20_fixture_destructible());
        SceneFileV20 {
            schema_version: 20,
            title: "V20 Fixture Level".into(),
            entities,
            settings: v19.settings,
        }
    }

    /// Regenerate `tests/fixtures/scene_v20.inf_lvl` — the **downgrade-bless**
    /// path, walked only under `INF_BLESS_FIXTURES=1`.
    #[test]
    fn bless_scene_v20_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v20_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 20);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v20.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed {} ({} bytes)", path.display(), bytes.len());
    }

    #[test]
    fn v20_fixture_is_reproducible_and_genuinely_v20() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v20.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v20 fixture present");
        assert_eq!(bytes[0], 20, "fixture must be a genuine schema-v20 payload");
        let rebuilt =
            bincode::serde::encode_to_vec(v20_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v20 fixture must match our frozen v20 writer"
        );
    }

    /// A committed **v20** payload still loads, keeps everything v20 could
    /// express — its wall's destructible and its cavern's volume included — and
    /// lifts with **no IK, no garment and no hair**, which is exactly what a v20
    /// level was. The "old bytes load forever" gate for the v21 bump.
    #[test]
    fn v20_loads_and_lifts_without_the_character_slots() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v20.inf_lvl"),
        )
        .expect("committed v20 fixture present");
        assert_eq!(bytes[0], 20, "the fixture must really be v20");
        let level = decode(&bytes).unwrap();
        assert_eq!(level.title, "V20 Fixture Level");
        assert_eq!(level.entities.len(), 6);
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // Everything v19 authored is still here after TWO hops …
        assert_eq!(by_name("River").water_body, Some(v17_fixture_water()));
        assert_eq!(by_name("Raft").buoyancy, Some(v18_fixture_buoyancy()));
        assert_eq!(by_name("Cavern").voxel_volume, Some(v19_fixture_volume()));
        let t = by_name("Terrain").terrain.clone().expect("terrain slot");
        assert_eq!(t.data.get_tile((0, 0)).unwrap().biome_sample(4, 1, 1), 3);
        assert_eq!(level.settings.sim_hz, 90.0);
        // … including the slot v20 itself added, which is the half a
        // defaults-only fixture would never have proven.
        assert_eq!(
            by_name("Cube").destructible,
            Some(v20_fixture_destructible())
        );

        // … and the three new slots lift to `None` on every entity.
        for e in &level.entities {
            assert!(
                e.ik_target.is_none() && e.cloth_sim.is_none() && e.hair_guides.is_none(),
                "a v20 level has no IK, no cloth and no hair; the lift must not conjure any"
            );
        }

        // Re-encoding writes the current schema and round-trips.
        let re = encode(&level).unwrap();
        assert_eq!(re[0], SCHEMA_VERSION as u8);
        assert_eq!(decode(&re).unwrap(), level);
    }

    /// The v20 downgrade is lossless **except** for the three character slots —
    /// the only things v20 cannot express. Proven as a property (round-trip a
    /// live record through the frozen shape) rather than by listing fields, so a
    /// slot added later cannot silently fall out of the ladder.
    #[test]
    fn v20_entity_downgrade_is_lossless_except_for_the_character_slots() {
        let live = RuntimeEntity {
            water_body: Some(v17_fixture_water()),
            buoyancy: Some(v18_fixture_buoyancy()),
            voxel_volume: Some(v19_fixture_volume()),
            destructible: Some(v20_fixture_destructible()),
            ik_target: Some(v21_fixture_ik_target()),
            cloth_sim: Some(v21_fixture_cloth()),
            hair_guides: Some(v21_fixture_hair()),
            ..v9_rec(Uuid::from_u128(0xFA00), "Wall", None).into_runtime()
        };
        let back = EntityRecordV20::from_current(live.clone()).into_runtime();

        // The three are exactly what is lost — the destructible and the voxel
        // volume, which v20 *can* express, are not …
        assert!(back.ik_target.is_none() && back.cloth_sim.is_none() && back.hair_guides.is_none());
        assert_eq!(back.destructible, live.destructible);
        assert_eq!(back.voxel_volume, live.voxel_volume);
        // … and nothing else moved: put them back and the records are equal,
        // which is the property form of "only these fields".
        assert_eq!(
            RuntimeEntity {
                ik_target: live.ik_target.clone(),
                cloth_sim: live.cloth_sim,
                hair_guides: live.hair_guides,
                ..back
            },
            live,
            "the v20 downgrade lost something other than the character slots"
        );
    }

    /// The non-default `IkTarget` the v21 tests author: two chains, an entity
    /// reference, a pole, a lowered weight and a disabled goal — every field away
    /// from its default, so a hop that produced defaults would be caught.
    fn v21_fixture_ik_target() -> IkTarget {
        use inf_ecs::components::IkGoalRecord;
        IkTarget {
            goals: vec![
                IkGoalRecord {
                    chain: vec![3, 4, 5],
                    target_entity: inf_ecs::refs::EntityRef::new(Uuid::from_u128(0xF_1CE0)),
                    target: Vec3d::new(0.25, -0.5, 1.5),
                    pole: Some(Vec3d::new(0.0, 0.0, 2.0)),
                    weight: 0.75,
                    enabled: true,
                },
                IkGoalRecord {
                    chain: vec![7, 8],
                    target: Vec3d::new(-1.0, 0.0, 0.0),
                    enabled: false,
                    ..Default::default()
                },
            ],
        }
    }

    fn v21_fixture_cloth() -> ClothSim {
        ClothSim {
            asset: Some(Uuid::from_u128(0xF_C107)),
            enabled: false,
            quality: 3,
        }
    }

    fn v21_fixture_hair() -> HairGuides {
        HairGuides {
            asset: Some(Uuid::from_u128(0xF_4A12)),
            enabled: false,
            quality: 2,
        }
    }

    /// **The v21 price, isolated.** An entity with no character components pays
    /// exactly three discriminant bytes; one that carries them pays the
    /// components.
    ///
    /// Measured as a delta between the frozen v20 and live v21 encodings of the
    /// very same record, so it is a *price* rather than an absolute that could
    /// silently absorb a later bump's growth (the P22.2 lesson, which the v20
    /// price test next door had to re-learn the moment this landed).
    #[test]
    fn v21_costs_three_bytes_per_ordinary_entity() {
        let plain = v9_rec(Uuid::from_u128(0xFC01), "Bedrock", None).into_runtime();
        let live = encode(&RuntimeLevel {
            title: "t".into(),
            entities: vec![plain.clone()],
            settings: RuntimeSettings::default(),
        })
        .unwrap();
        let frozen = bincode::serde::encode_to_vec(
            &SceneFileV20 {
                schema_version: 20,
                title: "t".into(),
                entities: vec![EntityRecordV20::from_current(plain.clone())],
                settings: RuntimeSettings::default(),
            },
            bincode_config(),
        )
        .unwrap();
        assert_eq!(
            live.len(),
            frozen.len() + CHARACTER_SLOT_BYTES + MOVEMENT_SLOT_BYTES,
            "the three v21 slots must cost exactly one discriminant byte each, \
             and v23's movement slot one more"
        );

        // A record that CARRIES them costs more than its own discriminants —
        // and the IK goals, which are the only unbounded one, cost their chains.
        let rigged = RuntimeEntity {
            ik_target: Some(v21_fixture_ik_target()),
            cloth_sim: Some(v21_fixture_cloth()),
            hair_guides: Some(v21_fixture_hair()),
            ..plain
        };
        let rigged_bytes = encode(&RuntimeLevel {
            title: "t".into(),
            entities: vec![rigged],
            settings: RuntimeSettings::default(),
        })
        .unwrap();
        assert!(rigged_bytes.len() > live.len() + CHARACTER_SLOT_BYTES);
    }

    /// The v21 additions round-trip through the whole codec — including the
    /// **new decode arm**, which only a payload stamped v21 exercises.
    #[test]
    fn v21_character_slots_round_trip_through_the_codec() {
        let level = RuntimeLevel {
            title: "Rigged".into(),
            entities: vec![RuntimeEntity {
                ik_target: Some(v21_fixture_ik_target()),
                cloth_sim: Some(v21_fixture_cloth()),
                hair_guides: Some(v21_fixture_hair()),
                ..v9_rec(Uuid::from_u128(0xFB01), "Hero", None).into_runtime()
            }],
            settings: RuntimeSettings::default(),
        };
        let bytes = encode(&level).unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = decode(&bytes).unwrap();
        assert_eq!(back, level);
        // Re-encoding is byte-identical.
        assert_eq!(encode(&back).unwrap(), bytes);

        // The goals survive field-for-field, including the disabled one — a hop
        // that dropped `enabled: false` would still round-trip the vector's
        // length.
        let t = back.entities[0].ik_target.clone().unwrap();
        assert_eq!(t.goals.len(), 2);
        assert_eq!(t.goals[0].chain, vec![3, 4, 5]);
        assert_eq!(t.goals[0].weight, 0.75);
        assert_eq!(
            t.goals[0].target_entity.get(),
            Some(Uuid::from_u128(0xF_1CE0))
        );
        assert!(!t.goals[1].enabled);
    }

    /// The **v22 `Material` wire**, declared here and not imported: the six
    /// fields v8 froze plus the `asset` binding v22 appended.
    ///
    /// This is the independent half of the pin now, because v22 is a component
    /// that GREW rather than a slot appended to the entity. `MaterialV21` up in
    /// the ladder cannot serve — it is precisely the shape without `asset` — and
    /// `inf_ecs::Material` cannot serve either, because a pin that imports the
    /// type it is pinning asserts nothing.
    #[derive(serde::Deserialize)]
    struct MaterialV22Wire {
        #[allow(dead_code)]
        base_color: Color,
        #[allow(dead_code)]
        metallic: f32,
        #[allow(dead_code)]
        roughness: f32,
        #[allow(dead_code)]
        emissive: Color,
        #[allow(dead_code)]
        blend: BlendMode,
        #[allow(dead_code)]
        alpha_cutoff: f32,
        /// What v22 added. Read by the pin, so a binding that fails to land in
        /// the material's tail position is a decode failure rather than a
        /// silently ignored field.
        asset: Option<Uuid>,
    }

    /// **`CharacterMovement`, re-declared independently** for the v23 wire pin.
    ///
    /// The `MaterialV22Wire` idiom: a shape that is NOT the live struct, so a
    /// field appended to the component without a schema bump leaves bytes this
    /// declaration cannot account for. The live `runtime` field is absent
    /// because it is `#[serde(skip)]` -- and that absence is itself pinned: if
    /// the skip were ever removed, the byte-consumption check below fails.
    #[derive(serde::Deserialize)]
    #[allow(dead_code)]
    struct CharacterMovementWire {
        mode: MovementMode,
        gait: Gait,
        rotation_mode: RotationMode,
        overlay: String,
        player_controlled: bool,
        walk_speed_mps: f64,
        run_speed_mps: f64,
        sprint_speed_mps: f64,
        crouch_speed_mps: f64,
        prone_speed_mps: f64,
        swim_surface_speed_mps: f64,
        swim_under_speed_mps: f64,
        looking_speed_scale: f64,
        aiming_speed_scale: f64,
        acceleration: SpeedCurve,
        braking: SpeedCurve,
        ground_friction: SpeedCurve,
        rotation_rate: SpeedCurve,
        air_control: f64,
        air_control_reduced: f64,
        air_accel_max_mps2: f64,
        terminal_velocity_mps: f64,
        gravity_mps2: f64,
        jump_speed_mps: f64,
        stand_half_height_m: f64,
        crouch_half_height_m: f64,
        prone_half_height_m: f64,
        step_height_m: f64,
        step_min_width_m: f64,
        slope_limit_deg: f64,
        slide_slope_deg: f64,
        sprint_input_min: f64,
        sprint_angle_deg: f64,
        slide_entry_speed_mps: f64,
        slide_exit_speed_mps: f64,
        slide_friction_flat: f64,
        slide_friction_slope: f64,
        roll_speed_mps: f64,
        roll_time_s: f64,
        dive_speed_mps: f64,
        dive_up_speed_mps: f64,
        land_hard_mps: f64,
        land_ragdoll_mps: f64,
        brake_friction_input: f64,
        brake_friction_idle: f64,
        land_friction_time_s: f64,
    }

    /// One entity as the v23 wire lays it out: the frozen 44-field record **with
    /// its material slot re-declared above**, the three tails P24.3 appended,
    /// and P29.3's movement slot **re-declared field-for-field**. A `type`
    /// because clippy counts the tuple's nesting, and because naming it says
    /// what it is.
    type V23EntityWire = (
        EntityRecordV20Gen<MaterialV22Wire>,
        Option<IkTarget>,
        Option<ClothSim>,
        Option<HairGuides>,
        Option<CharacterMovementWire>,
    );

    /// The live entity plus one appended slot — the shadow v24's entity.
    ///
    /// Serialized from the **live** record rather than reassembled from frozen
    /// parts, because the shadow's only job is to produce bytes that carry one
    /// slot more than the current wire; reassembling it would make the fixture
    /// depend on the very downgrade the pin is meant to be independent of.
    type V24EntityShadow<'a> = (&'a RuntimeEntity, Option<u8>);

    /// **The v23 wire shape, pinned against an INDEPENDENT declaration.**
    ///
    /// The `SkeletonAssetV2Wire` idiom (`inf-anim`), applied to the scene. The
    /// pair this replaces proved nothing: one encoded *and* decoded with
    /// `SceneFileV21`, which is true of any struct whatsoever, and the other
    /// appended a `None` byte to a **live** encoding and asserted the **live**
    /// decoder ignored it — true of every bincode struct at every version. Both
    /// were tautologies wearing the vocabulary of a wire pin.
    ///
    /// This shape is independent where it matters. `EntityRecordV20Gen` is a
    /// FROZEN declaration whose whole purpose is not to drift with the live
    /// record, [`MaterialV22Wire`] re-declares the one component v22 grew, and
    /// bincode encodes a tuple as its elements concatenated with no framing — so
    /// the tuple below is byte-for-byte the live record's layout, assembled from
    /// 44 frozen fields plus the three tails **named here by type**. A 48th slot
    /// appended to the live record leaves bytes this shape cannot account for; a
    /// tail whose type changes fails to decode; and a field appended to
    /// `Material` without a bump shifts every byte after it, which is the case
    /// v22 itself is.
    #[derive(serde::Deserialize)]
    struct SceneFileV23Wire {
        schema_version: u32,
        title: String,
        entities: Vec<V23EntityWire>,
        settings: RuntimeSettings,
    }

    /// A **shadow v24** — the live wire plus one appended tail slot, exactly
    /// what an author adding a component would write.
    #[derive(serde::Serialize)]
    struct SceneFileV24Shadow<'a> {
        schema_version: u32,
        title: &'a str,
        entities: Vec<V24EntityShadow<'a>>,
        settings: RuntimeSettings,
    }

    #[test]
    fn the_v23_wire_shape_is_pinned_against_an_independent_declaration() {
        let bound = Uuid::from_u128(0x00FA_7E12);
        let level = RuntimeLevel {
            title: "Pinned".into(),
            entities: vec![
                RuntimeEntity {
                    material: Some(Material {
                        asset: Some(bound),
                        ..Material::default()
                    }),
                    ik_target: Some(v21_fixture_ik_target()),
                    cloth_sim: Some(v21_fixture_cloth()),
                    hair_guides: Some(v21_fixture_hair()),
                    character_movement: Some(v23_fixture_movement()),
                    ..v9_rec(Uuid::from_u128(0xFD10), "Hero", None).into_runtime()
                },
                v9_rec(Uuid::from_u128(0xFD11), "Prop", None).into_runtime(),
            ],
            settings: RuntimeSettings::default(),
        };
        let bytes = encode(&level).unwrap();

        let (wire, consumed): (SceneFileV23Wire, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config())
                .expect("the pinned v23 shape decodes the v23 wire");
        assert_eq!(
            consumed,
            bytes.len(),
            "the encoding carries {} bytes the pinned shape does not account for \
             — a slot was appended to the entity record without bumping \
             SCHEMA_VERSION",
            bytes.len() - consumed
        );
        assert_eq!(wire.schema_version, SCHEMA_VERSION);
        assert_eq!(wire.entities.len(), 2);
        // The v22 field really landed at the END of the material and nowhere
        // else — a shape that decoded but mis-assigned it would still consume
        // every byte, which is exactly what a positional format lets happen.
        assert_eq!(
            wire.entities[0].0.material.as_ref().and_then(|m| m.asset),
            Some(bound),
            "the material binding is not in the material's tail position"
        );
        // The three tails really landed in the three tail positions — a shape
        // that decoded but mis-assigned them would still consume every byte.
        assert!(
            wire.entities[0].1.is_some(),
            "the IkTarget is not in slot 45"
        );
        assert!(
            wire.entities[0].2.is_some(),
            "the ClothSim is not in slot 46"
        );
        assert!(
            wire.entities[0].3.is_some(),
            "the HairGuides is not in slot 47"
        );
        assert!(
            wire.entities[0].4.is_some(),
            "the CharacterMovement is not in slot 48"
        );
        assert!(wire.entities[1].1.is_none() && wire.entities[1].3.is_none());
        assert!(
            wire.entities[1].4.is_none(),
            "and an entity without one really has none"
        );
        let _ = &wire.title;
        let _ = &wire.settings;
    }

    /// **An appended tail slot without a bump is CAUGHT** — measured against the
    /// SHADOW, which is what the previous arm got wrong.
    ///
    /// The old version appended a byte to a live encoding and asked the live
    /// decoder about it; `decode_from_slice` ignores trailing bytes by contract,
    /// so it asserted a property of bincode rather than of this ladder. Here the
    /// v23 bytes come from an independently declared struct that really has an
    /// extra tail, and the claim is that the **pinned v22 shape** cannot account
    /// for them — which is the signal the arm above refuses.
    #[test]
    fn a_tail_slot_appended_without_a_bump_leaves_bytes_the_pin_refuses() {
        let level = RuntimeLevel {
            title: "Pinned".into(),
            entities: vec![
                RuntimeEntity {
                    ik_target: Some(v21_fixture_ik_target()),
                    cloth_sim: Some(v21_fixture_cloth()),
                    hair_guides: Some(v21_fixture_hair()),
                    ..v9_rec(Uuid::from_u128(0xFD10), "Hero", None).into_runtime()
                },
                v9_rec(Uuid::from_u128(0xFD11), "Prop", None).into_runtime(),
            ],
            settings: RuntimeSettings::default(),
        };
        let v23 = encode(&level).unwrap();
        let v24 = bincode::serde::encode_to_vec(
            &SceneFileV24Shadow {
                schema_version: SCHEMA_VERSION,
                title: &level.title,
                entities: vec![(&level.entities[0], Some(7u8))],
                settings: RuntimeSettings::default(),
            },
            bincode_config(),
        )
        .unwrap();

        // 1. A slot really was added.
        let one_entity = encode(&RuntimeLevel {
            entities: vec![level.entities[0].clone()],
            ..level.clone()
        })
        .unwrap();
        assert!(
            v24.len() > one_entity.len(),
            "the shadow v24 is not longer than v23, so nothing was appended and \
             this test is measuring nothing"
        );
        // 2. …and the pinned v22 shape CANNOT READ THEM AT ALL.
        //
        // Stronger than the trailing-byte signal the first draft looked for, and
        // the difference is instructive: an entity slot is appended in the
        // MIDDLE of the file (the entity vector precedes `settings`), so it does
        // not leave a tail — it shifts every byte after it and the decoder walks
        // off into the settings block. That is exactly why a positional format
        // needs a version bump rather than a `#[serde(default)]`, and it is the
        // failure this pin exists to produce.
        let err =
            bincode::serde::decode_from_slice::<SceneFileV23Wire, _>(&v24, bincode_config()).err();
        assert!(
            err.is_some(),
            "the pinned v23 shape read a payload with an extra entity slot as if nothing had changed — the shape pin has no forcing function at all"
        );
        // …and the SAME bytes minus the appended slot decode cleanly, so the
        // refusal above is the slot's doing and not a broken fixture.
        assert!(
            bincode::serde::decode_from_slice::<SceneFileV23Wire, _>(&one_entity, bincode_config())
                .is_ok(),
            "the control payload does not decode either — the fixture is wrong, not the pin"
        );
        let _ = v23;
    }

    // ── v22 the persisted material binding (P26.3b) ───────────────────────

    /// What the v22 field costs an entity that carries a `Material` and no
    /// binding: one `Option` discriminant, and nothing else. An entity with no
    /// `Material` at all pays **zero** — the field is inside the component.
    const MATERIAL_BINDING_BYTES: usize = 1;

    /// The `.inf_mat` the v22 tests bind. A fixed GUID so the two codec mirrors
    /// can be byte-compared.
    fn v22_fixture_material_binding() -> Uuid {
        Uuid::from_u128(0xFA7E_0026)
    }

    /// The **v21** material the fixture's wall carries: every scalar away from
    /// its default, so a v22 hop that produced defaults would be caught. It is
    /// the shape v21 could express and v22 grew — the binding is exactly what it
    /// cannot say.
    ///
    /// The literals must match the editor codec's `v21_fixture_material` exactly:
    /// the two committed fixtures are byte-compared by the editor's
    /// `v21_fixture_matches_the_runtime_codecs_copy`, which is the whole point of
    /// writing them twice.
    fn v21_fixture_material() -> MaterialV21 {
        MaterialV21 {
            base_color: Color::new(0.42, 0.18, 0.66, 0.9),
            metallic: 0.875,
            roughness: 0.1875,
            emissive: Color::new(0.05, 0.0, 0.125, 1.0),
            blend: BlendMode::Masked,
            alpha_cutoff: 0.3125,
        }
    }

    /// Rebuild the exact schema-v21 file the committed v21 fixture was generated
    /// from, out of the frozen v21 record type (the provenance lock).
    ///
    /// Built by downgrading the **v20** reference one rung and then authoring the
    /// three slots v21 added, so the field list can never drift from the ladder.
    fn v21_scene_reference() -> SceneFileV21 {
        let v20 = v20_scene_reference();
        let mut entities: Vec<EntityRecordV21> = v20
            .entities
            .into_iter()
            .map(|e| EntityRecordV21::from_current(e.into_runtime()))
            .collect();
        // The hero is what only v21 could write.
        let hero = entities
            .iter_mut()
            .find(|e| e.name == "Cube")
            .expect("the v20 fixture has a Cube");
        hero.ik_target = Some(v21_fixture_ik_target());
        hero.cloth_sim = Some(v21_fixture_cloth());
        hero.hair_guides = Some(v21_fixture_hair());
        // …and a fully non-default material, which is what makes the v22 hop's
        // "the scalars survive" claim a measurement rather than a hope.
        hero.material = Some(v21_fixture_material());
        SceneFileV21 {
            schema_version: 21,
            title: "V21 Fixture Level".into(),
            entities,
            settings: v20.settings,
        }
    }

    /// Regenerate `tests/fixtures/scene_v21.inf_lvl` — the **downgrade-bless**
    /// path, walked only under `INF_BLESS_FIXTURES=1`.
    #[test]
    fn bless_scene_v21_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v21_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 21);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v21.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed {} ({} bytes)", path.display(), bytes.len());
    }

    #[test]
    fn v21_fixture_is_reproducible_and_genuinely_v21() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v21.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v21 fixture present");
        assert_eq!(bytes[0], 21, "fixture must be a genuine schema-v21 payload");
        let rebuilt =
            bincode::serde::encode_to_vec(v21_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v21 fixture must match our frozen v21 writer"
        );
    }

    /// A committed **v21** payload still loads, keeps everything v21 could
    /// express — its wall's destructible, its cavern's volume, its hero's IK,
    /// garment and hair — and lifts with **no material binding**, which is
    /// exactly what a v21 level was. The "old bytes load forever" gate for the
    /// v22 bump.
    #[test]
    fn v21_loads_and_lifts_without_a_material_binding() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v21.inf_lvl"),
        )
        .expect("committed v21 fixture present");
        assert_eq!(bytes[0], 21, "the fixture must really be v21");
        let level = decode(&bytes).unwrap();
        assert_eq!(level.title, "V21 Fixture Level");
        assert_eq!(level.entities.len(), 6);
        let by_name = |n: &str| level.entities.iter().find(|e| e.name == n).unwrap();

        // Everything the older rungs authored is still here after the hop …
        assert_eq!(by_name("River").water_body, Some(v17_fixture_water()));
        assert_eq!(by_name("Raft").buoyancy, Some(v18_fixture_buoyancy()));
        assert_eq!(by_name("Cavern").voxel_volume, Some(v19_fixture_volume()));
        assert_eq!(
            by_name("Cube").destructible,
            Some(v20_fixture_destructible())
        );
        // … including the three slots v21 itself added, which is the half a
        // defaults-only fixture would never have proven …
        assert_eq!(by_name("Cube").ik_target, Some(v21_fixture_ik_target()));
        assert_eq!(by_name("Cube").cloth_sim, Some(v21_fixture_cloth()));
        assert_eq!(by_name("Cube").hair_guides, Some(v21_fixture_hair()));
        // … and the material's SCALARS survive the frozen-component hop, which
        // is the part a bump inside a component gets wrong: every one of the six
        // fields, compared against the frozen v21 literal rather than against a
        // default.
        let m = by_name("Cube").material.expect("the Cube is materialed");
        let want = v21_fixture_material();
        assert_eq!(m.base_color, want.base_color);
        assert_eq!(m.metallic, want.metallic);
        assert_eq!(m.roughness, want.roughness);
        assert_eq!(m.emissive, want.emissive);
        assert_eq!(m.blend, want.blend);
        assert_eq!(m.alpha_cutoff, want.alpha_cutoff);

        // … and the new field lifts to `None` on every material.
        for e in &level.entities {
            assert!(
                e.material.is_none_or(|m| m.asset.is_none()),
                "a v21 level has no material binding; the lift must not conjure one"
            );
        }

        // Re-encoding writes the current schema and round-trips.
        let re = encode(&level).unwrap();
        assert_eq!(re[0], SCHEMA_VERSION as u8);
        assert_eq!(decode(&re).unwrap(), level);
    }

    /// The v21 downgrade is lossless **except** for the material binding — the
    /// only thing v21 cannot express. Proven as a property (round-trip a live
    /// record through the frozen shape) rather than by listing fields, so a slot
    /// added later cannot silently fall out of the ladder.
    #[test]
    fn v21_entity_downgrade_is_lossless_except_for_the_material_binding() {
        let live = RuntimeEntity {
            material: Some(Material {
                blend: BlendMode::Translucent,
                alpha_cutoff: 0.125,
                asset: Some(v22_fixture_material_binding()),
                ..Material::default()
            }),
            water_body: Some(v17_fixture_water()),
            voxel_volume: Some(v19_fixture_volume()),
            destructible: Some(v20_fixture_destructible()),
            ik_target: Some(v21_fixture_ik_target()),
            cloth_sim: Some(v21_fixture_cloth()),
            hair_guides: Some(v21_fixture_hair()),
            ..v9_rec(Uuid::from_u128(0xFA00), "Wall", None).into_runtime()
        };
        let back = EntityRecordV21::from_current(live.clone()).into_runtime();

        // The binding is exactly what is lost — the material's own scalars, and
        // the three v21 slots, which v21 *can* express, are not …
        let m = back.material.expect("the material survives");
        assert!(m.asset.is_none());
        assert_eq!(m.blend, BlendMode::Translucent);
        assert_eq!(m.alpha_cutoff, 0.125);
        assert_eq!(back.cloth_sim, live.cloth_sim);
        assert_eq!(back.destructible, live.destructible);
        // … and nothing else moved: put it back and the records are equal, which
        // is the property form of "only this field".
        assert_eq!(
            RuntimeEntity {
                material: back.material.map(|m| Material {
                    asset: live.material.and_then(|l| l.asset),
                    ..m
                }),
                ..back
            },
            live,
            "the v21 downgrade lost something other than the material binding"
        );
    }

    /// **The v22 price, isolated.** A materialed entity with no binding pays
    /// exactly one discriminant byte; an entity with no `Material` pays nothing
    /// at all, because the field lives inside the component.
    ///
    /// Measured as a delta between the frozen v21 and live v22 encodings of the
    /// very same record, so it is a *price* rather than an absolute that could
    /// silently absorb a later bump's growth (the P22.2 lesson).
    #[test]
    fn v22_costs_one_byte_per_materialed_entity_and_nothing_otherwise() {
        let bare = RuntimeEntity {
            material: None,
            ..v9_rec(Uuid::from_u128(0xFC02), "Marker", None).into_runtime()
        };
        let materialed = RuntimeEntity {
            material: Some(Material::default()),
            ..bare.clone()
        };
        let frozen = |e: &RuntimeEntity| {
            bincode::serde::encode_to_vec(
                &SceneFileV21 {
                    schema_version: 21,
                    title: "t".into(),
                    entities: vec![EntityRecordV21::from_current(e.clone())],
                    settings: RuntimeSettings::default(),
                },
                bincode_config(),
            )
            .unwrap()
        };
        let live = |e: &RuntimeEntity| {
            encode(&RuntimeLevel {
                title: "t".into(),
                entities: vec![e.clone()],
                settings: RuntimeSettings::default(),
            })
            .unwrap()
        };
        assert_eq!(
            live(&materialed).len(),
            frozen(&materialed).len() + MATERIAL_BINDING_BYTES + MOVEMENT_SLOT_BYTES,
            "the v22 binding must cost exactly one discriminant byte on a \
             materialed entity (and v23's slot one more, since `live` is v23)"
        );
        assert_eq!(
            live(&bare).len(),
            frozen(&bare).len() + MOVEMENT_SLOT_BYTES,
            "an entity with no Material must pay nothing for v22 — the field is \
             inside the component, not on the entity record — and exactly one \
             byte for v23, whose slot is on the record"
        );

        // A record that CARRIES a binding costs more than its discriminant.
        let bound = RuntimeEntity {
            material: Some(Material {
                asset: Some(v22_fixture_material_binding()),
                ..Material::default()
            }),
            ..bare.clone()
        };
        assert!(live(&bound).len() > live(&materialed).len());
    }

    /// The v22 addition round-trips through the whole codec — including the
    /// **new decode arm**, which only a payload stamped v22 exercises.
    #[test]
    fn v22_material_binding_round_trips_through_the_codec() {
        let level = RuntimeLevel {
            title: "Textured".into(),
            entities: vec![RuntimeEntity {
                material: Some(Material {
                    base_color: Color::new(0.9, 0.2, 0.1, 1.0),
                    metallic: 0.75,
                    roughness: 0.25,
                    emissive: Color::new(0.1, 0.0, 0.0, 1.0),
                    blend: BlendMode::Masked,
                    alpha_cutoff: 0.375,
                    asset: Some(v22_fixture_material_binding()),
                }),
                ..v9_rec(Uuid::from_u128(0xFB02), "Wall", None).into_runtime()
            }],
            settings: RuntimeSettings::default(),
        };
        let bytes = encode(&level).unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = decode(&bytes).unwrap();
        assert_eq!(back, level);
        // Re-encoding is byte-identical.
        assert_eq!(encode(&back).unwrap(), bytes);

        // The binding survives beside every scalar — a hop that kept the GUID
        // and defaulted the blend would still round-trip the entity count.
        let m = back.entities[0].material.expect("material");
        assert_eq!(m.asset, Some(v22_fixture_material_binding()));
        assert_eq!(m.blend, BlendMode::Masked);
        assert_eq!(m.alpha_cutoff, 0.375);
        assert_eq!(m.metallic, 0.75);

        // Clearing the binding really changes the bytes — so the field is on
        // the wire and not merely in the struct.
        let mut cleared = level.clone();
        cleared.entities[0].material.as_mut().unwrap().asset = None;
        let cleared_bytes = encode(&cleared).unwrap();
        assert_ne!(cleared_bytes, bytes);
        assert_eq!(decode(&cleared_bytes).unwrap(), cleared);
    }

    // ── v23 the movement component (P29.3) ────────────────────────────────

    /// What the v23 slot costs an entity with no movement component: one
    /// `Option` discriminant, and nothing else. Unlike v22's binding it is on
    /// the **record**, not inside a component, so every entity pays it.
    const MOVEMENT_SLOT_BYTES: usize = 1;

    /// The movement component the v23 arms author — deliberately non-default in
    /// every family of field (a mode, a gait, a rotation mode, an overlay name,
    /// a speed, a curve, a capsule height, a threshold), so "it round-trips" is
    /// a measurement rather than a statement about `Default`.
    fn v23_fixture_movement() -> CharacterMovement {
        CharacterMovement {
            mode: MovementMode::Crouch,
            gait: Gait::Sprint,
            rotation_mode: RotationMode::Aiming,
            overlay: "rifle".into(),
            player_controlled: true,
            walk_speed_mps: 1.25,
            sprint_speed_mps: 7.5,
            acceleration: SpeedCurve::new(1.0, 2.0, 3.0, 4.0),
            crouch_half_height_m: 0.28,
            step_height_m: 0.35,
            land_ragdoll_mps: 11.5,
            ..CharacterMovement::default()
        }
    }

    /// Rebuild the exact schema-v22 file the committed v22 fixture was generated
    /// from, out of the frozen v22 record type (the provenance lock).
    ///
    /// Built by lifting the **v21** reference one rung and then authoring the
    /// binding v22 added, so the field list can never drift from the ladder.
    fn v22_scene_reference() -> SceneFileV22 {
        let v21 = v21_scene_reference();
        let mut entities: Vec<EntityRecordV22> = v21
            .entities
            .into_iter()
            .map(|e| EntityRecordV22::from_current(e.into_runtime()))
            .collect();
        // The material binding is what only v22 could write.
        let hero = entities
            .iter_mut()
            .find(|e| e.name == "Cube")
            .expect("the v21 fixture has a Cube");
        if let Some(m) = hero.material.as_mut() {
            m.asset = Some(v22_fixture_material_binding());
        }
        SceneFileV22 {
            schema_version: 22,
            title: "V22 Fixture Level".into(),
            entities,
            settings: v21.settings,
        }
    }

    /// Regenerate `tests/fixtures/scene_v22.inf_lvl` — the **downgrade-bless**
    /// path, walked only under `INF_BLESS_FIXTURES=1`.
    #[test]
    fn bless_scene_v22_fixture() {
        if std::env::var("INF_BLESS_FIXTURES").as_deref() != Ok("1") {
            return;
        }
        let bytes = bincode::serde::encode_to_vec(v22_scene_reference(), bincode_config()).unwrap();
        assert_eq!(bytes[0], 22);
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v22.inf_lvl");
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("blessed {} ({} bytes)", path.display(), bytes.len());
    }

    #[test]
    fn v22_fixture_is_reproducible_and_genuinely_v22() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v22.inf_lvl");
        let bytes = std::fs::read(&path).expect("committed v22 fixture present");
        assert_eq!(bytes[0], 22, "fixture must be a genuine schema-v22 payload");
        let rebuilt =
            bincode::serde::encode_to_vec(v22_scene_reference(), bincode_config()).unwrap();
        assert_eq!(
            rebuilt, bytes,
            "the committed v22 fixture must match our frozen v22 writer"
        );
    }

    /// A committed **v22** payload still loads, keeps everything v22 could
    /// express — its hero's IK, garment, hair and material binding — and lifts
    /// with **no movement component**, which is exactly what a v22 level was.
    /// The "old bytes load forever" gate for the v23 bump.
    #[test]
    fn v22_loads_and_lifts_without_a_movement_component() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v22.inf_lvl"),
        )
        .expect("committed v22 fixture present");
        assert_eq!(bytes[0], 22);
        let level = decode(&bytes).expect("a v22 level decodes");
        assert!(!level.entities.is_empty());
        // Nothing in a v22 file could have authored one.
        for e in &level.entities {
            assert!(
                e.character_movement.is_none(),
                "{} arrived with a movement component out of a v22 file",
                e.name
            );
        }
        // …and everything v22 COULD express survived the hop, or "it loads"
        // would be satisfied by a decoder that dropped the payload.
        let hero = level
            .entities
            .iter()
            .find(|e| e.name == "Cube")
            .expect("the fixture's hero");
        assert!(hero.ik_target.is_some(), "the v21 IK survived");
        assert!(hero.cloth_sim.is_some(), "the v21 garment survived");
        assert!(hero.hair_guides.is_some(), "the v21 hair survived");
        assert_eq!(
            hero.material.and_then(|m| m.asset),
            Some(v22_fixture_material_binding()),
            "the v22 material binding survived"
        );
        // Re-encoding stamps the CURRENT version — the lift is a real upgrade.
        let re = encode(&level).unwrap();
        assert_eq!(re[0], SCHEMA_VERSION as u8);
        assert_eq!(decode(&re).unwrap(), level);
    }

    /// The **downgrade** direction: projecting a live record onto the frozen v22
    /// shape loses the movement component and **nothing else**.
    ///
    /// Asserted as a property — everything else compared field-for-field through
    /// the struct's own `PartialEq` — rather than as a hand-written field list,
    /// which is the only version of this claim that cannot rot.
    #[test]
    fn v22_entity_downgrade_is_lossless_except_for_the_movement_component() {
        let live = RuntimeEntity {
            character_movement: Some(v23_fixture_movement()),
            ik_target: Some(v21_fixture_ik_target()),
            cloth_sim: Some(v21_fixture_cloth()),
            hair_guides: Some(v21_fixture_hair()),
            material: Some(Material {
                asset: Some(v22_fixture_material_binding()),
                ..Material::default()
            }),
            ..v9_rec(Uuid::from_u128(0xFD22), "Hero", None).into_runtime()
        };
        let back = EntityRecordV22::from_current(live.clone()).into_runtime();
        assert_eq!(
            back.character_movement, None,
            "a v22 record cannot carry a movement component"
        );
        assert_eq!(
            RuntimeEntity {
                character_movement: live.character_movement.clone(),
                ..back
            },
            live,
            "the v22 downgrade lost something other than the movement component"
        );
    }

    /// **The v23 price, isolated.** Every entity pays exactly one discriminant
    /// byte, and an entity that carries a component pays more than that.
    ///
    /// Measured as a delta between the frozen v22 and live v23 encodings of the
    /// very same record, so it is a *price* rather than an absolute that could
    /// silently absorb a later bump's growth (the P22.2 lesson).
    #[test]
    fn v23_costs_one_byte_per_entity() {
        let plain = v9_rec(Uuid::from_u128(0xFC03), "Marker", None).into_runtime();
        let live = |e: &RuntimeEntity| {
            encode(&RuntimeLevel {
                title: "t".into(),
                entities: vec![e.clone()],
                settings: RuntimeSettings::default(),
            })
            .unwrap()
        };
        let frozen = |e: &RuntimeEntity| {
            bincode::serde::encode_to_vec(
                &SceneFileV22 {
                    schema_version: 22,
                    title: "t".into(),
                    entities: vec![EntityRecordV22::from_current(e.clone())],
                    settings: RuntimeSettings::default(),
                },
                bincode_config(),
            )
            .unwrap()
        };
        assert_eq!(
            live(&plain).len(),
            frozen(&plain).len() + MOVEMENT_SLOT_BYTES,
            "the v23 slot must cost exactly one discriminant byte on an entity \
             that has no movement component"
        );

        // A record that CARRIES one costs a great deal more than its
        // discriminant — a ~46-field tunable block — so the byte above really is
        // the empty slot and not the component.
        let moving = RuntimeEntity {
            character_movement: Some(v23_fixture_movement()),
            ..plain.clone()
        };
        assert!(
            live(&moving).len() > live(&plain).len() + 100,
            "a carried movement component is {} bytes, which is not a tunable set",
            live(&moving).len() - live(&plain).len()
        );
    }

    /// The v23 addition round-trips through the whole codec — including the
    /// **new decode arm**, which only a payload stamped v23 exercises.
    #[test]
    fn v23_movement_component_round_trips_through_the_codec() {
        let level = RuntimeLevel {
            title: "Locomotion".into(),
            entities: vec![RuntimeEntity {
                character_movement: Some(v23_fixture_movement()),
                ..v9_rec(Uuid::from_u128(0xFB03), "Hero", None).into_runtime()
            }],
            settings: RuntimeSettings::default(),
        };
        let bytes = encode(&level).unwrap();
        assert_eq!(bytes[0], SCHEMA_VERSION as u8);
        let back = decode(&bytes).unwrap();
        assert_eq!(back, level);
        assert_eq!(
            encode(&back).unwrap(),
            bytes,
            "re-encoding is byte-identical"
        );

        // Every family of field survived — a hop that kept the mode and
        // defaulted the curves would still round-trip the entity count.
        let m = back.entities[0]
            .character_movement
            .clone()
            .expect("the movement component");
        assert_eq!(m.mode, MovementMode::Crouch);
        assert_eq!(m.gait, Gait::Sprint);
        assert_eq!(m.rotation_mode, RotationMode::Aiming);
        assert_eq!(m.overlay, "rifle");
        assert!(m.player_controlled);
        assert_eq!(m.walk_speed_mps, 1.25);
        assert_eq!(m.acceleration, SpeedCurve::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(m.crouch_half_height_m, 0.28);
        assert_eq!(m.step_height_m, 0.35);
        assert_eq!(m.land_ragdoll_mps, 11.5);
        // The live runtime is `#[serde(skip)]`, so it comes back at rest however
        // the writer left it — the property the wire pin's absent field asserts
        // from the other side.
        assert_eq!(m.runtime, MovementRuntime::default());

        // Clearing the component really changes the bytes, so the slot is on the
        // wire and not merely in the struct.
        let mut cleared = level.clone();
        cleared.entities[0].character_movement = None;
        let cleared_bytes = encode(&cleared).unwrap();
        assert_ne!(cleared_bytes, bytes);
        assert_eq!(decode(&cleared_bytes).unwrap(), cleared);
    }

    /// A payload from a **newer** build is refused by name, with both versions in
    /// the message — the [`SceneError::SchemaTooNew`] doctrine, re-checked at the
    /// new ceiling.
    #[test]
    fn a_v24_payload_is_refused_by_name() {
        let level = RuntimeLevel {
            title: "Future".into(),
            entities: vec![v9_rec(Uuid::from_u128(0xFF01), "A", None).into_runtime()],
            settings: RuntimeSettings::default(),
        };
        let mut bytes = encode(&level).unwrap();
        bytes[0] = SCHEMA_VERSION as u8 + 1;
        match decode(&bytes) {
            Err(SceneError::SchemaTooNew { found, current }) => {
                assert_eq!(found, SCHEMA_VERSION + 1);
                assert_eq!(current, SCHEMA_VERSION);
            }
            other => panic!("expected SchemaTooNew, got {other:?}"),
        }
        // …and there is no "too old" direction to refuse: every schema from v1 up
        // is migrated by the ladder above, which is what makes this codec's
        // contract different from `SessionSave`'s two-sided one.
        assert!(decode(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scene_v20.inf_lvl")
            )
            .unwrap()
        )
        .is_ok());
    }

    /// **L5.F4 — an all-zero buffer used to decode as a valid empty level.**
    ///
    /// `SceneFileV1` is `{ schema_version: u32, title: String, entities: Vec<_> }`
    /// and `bincode::config::standard()` varint-encodes all three, so three zero
    /// bytes are a *structurally valid* v1 record and `decode_from_slice`
    /// ignores whatever follows them. Any zero-filled or sparse file of at least
    /// three bytes therefore loaded as an untitled, empty level.
    ///
    /// This is the reasoning lens 5 could only derive from struct shapes and
    /// varint semantics (it was read-only), executed. Un-fix mutation: restore
    /// the `0 |` arm and the first assertion fails by decoding successfully.
    #[test]
    fn a_zero_filled_buffer_is_not_a_level() {
        for len in [3usize, 8, 64, 4096] {
            let zeros = vec![0u8; len];
            let err = decode(&zeros)
                .err()
                .unwrap_or_else(|| panic!("{len} zero bytes decoded as a level"));
            let msg = err.to_string();
            assert!(
                msg.contains("version 0"),
                "the refusal must name the cause, not blame a newer build: {msg}"
            );
        }
        // The control: a real v1-through-current payload still decodes. Version 0
        // is a version no writer has ever emitted, which is why dropping it costs
        // no committed file anything.
        let level = RuntimeLevel {
            title: "Real".into(),
            entities: Vec::new(),
            settings: RuntimeSettings::default(),
        };
        assert!(decode(&encode(&level).unwrap()).is_ok());
    }
}
