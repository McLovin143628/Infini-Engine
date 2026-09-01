//! Core scene components (P3.1.3).
//!
//! Every editable component derives `Component + Reflect + serde`, is registered
//! in [`crate::registry`], and reflects `Component + Default` so the Details
//! panel can read/write/reset it generically. Computed, non-editable components
//! (`GlobalTransform`, `ComputedVisibility`) are plain components refreshed by
//! transform propagation.

use std::collections::BTreeMap;

use bevy_ecs::prelude::*;
use bevy_ecs::reflect::ReflectComponent;
use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::Reflect;
use glam::{DAffine3, DQuat, DVec3};
use inf_terrain::TerrainData;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::math::{Color, Vec2d, Vec3d};
use crate::refs::EntityRef;

/// Stable, save-surviving identity. Bevy `Entity` ids are reused across
/// spawn/despawn and never persisted; the `Guid` is what `.inf_lvl` stores and
/// what selection/undo reference across a reload. Not reflected — identity is
/// not an editable property.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Guid(pub Uuid);

impl Guid {
    pub fn new(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

/// A per-entity **blueprint-class binding** (P9.5 · `.inf_lvl` schema v3): the
/// GUID of the `.inf_act` blueprint-class asset this entity runs at play time.
/// Persisted in `.inf_lvl` (the `EntityRecord::actor` slot) and resolved by the
/// player / in-editor Simulate to a `BlueprintClass`.
///
/// Not reflected — like [`Guid`], it is an identity/link, not an editable
/// numeric property, so it stays out of the generic Details grid (it is authored
/// by dragging a `.inf_act` onto the entity, and shown read-only in Details).
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActorClass(pub Uuid);

impl ActorClass {
    pub fn new(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

/// The Outliner label (UE's "actor label"). Shown in the Outliner + Details
/// header and renamed through a dedicated command, so it is registered for
/// reflection but excluded from the generic property grid.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[reflect(Component, Default)]
pub struct Name(pub String);

impl Name {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Name {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Local transform: translation (metres), rotation (**euler degrees**), scale.
///
/// Rotation is euler degrees (X pitch, Y yaw, Z roll) exactly as UE presents it
/// — a quaternion in a numeric grid is hostile UX. Math goes through
/// [`Transform::affine`], which composes a `DQuat` via `from_euler(YXZ, …)`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Transform {
    pub translation: Vec3d,
    pub rotation: Vec3d,
    pub scale: Vec3d,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3d::ZERO,
        rotation: Vec3d::ZERO,
        scale: Vec3d::ONE,
    };

    pub fn from_translation(t: DVec3) -> Self {
        Self {
            translation: t.into(),
            ..Self::IDENTITY
        }
    }

    /// Rotation as a quaternion (YXZ euler order: yaw, pitch, roll).
    pub fn quat(self) -> DQuat {
        let r = self.rotation.to_dvec3();
        DQuat::from_euler(
            glam::EulerRot::YXZ,
            r.y.to_radians(),
            r.x.to_radians(),
            r.z.to_radians(),
        )
    }

    /// Replace the rotation from a quaternion (extracts YXZ euler degrees).
    pub fn set_quat(&mut self, q: DQuat) {
        let (y, x, z) = q.to_euler(glam::EulerRot::YXZ);
        self.rotation = Vec3d::new(x.to_degrees(), y.to_degrees(), z.to_degrees());
    }

    /// Local TRS as an affine (correct under non-uniform scale + rotation).
    pub fn affine(self) -> DAffine3 {
        DAffine3::from_scale_rotation_translation(
            self.scale.to_dvec3(),
            self.quat(),
            self.translation.to_dvec3(),
        )
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// World-space transform, recomputed by [`crate::transform::propagate`].
/// Not editable / not reflected.
#[derive(Component, Clone, Copy, Debug)]
pub struct GlobalTransform(pub DAffine3);

impl Default for GlobalTransform {
    fn default() -> Self {
        Self(DAffine3::IDENTITY)
    }
}

impl GlobalTransform {
    pub fn translation(&self) -> DVec3 {
        self.0.translation
    }
}

/// Self visibility toggle (the Outliner eye).
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Visibility {
    pub visible: bool,
}

impl Default for Visibility {
    fn default() -> Self {
        Self { visible: true }
    }
}

/// Effective visibility (self AND every ancestor), set by propagation.
/// Not editable / not reflected.
#[derive(Component, Clone, Copy, Debug)]
pub struct ComputedVisibility(pub bool);

impl Default for ComputedVisibility {
    fn default() -> Self {
        Self(true)
    }
}

/// Built-in primitive mesh kinds (a placeholder for `MeshRef`-to-asset in
/// Phase 4). Enough to author the primitive+lights scene the phase gate needs.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Primitive {
    #[default]
    Cube,
    Sphere,
    Plane,
    Cylinder,
    Cone,
}

/// How a [`Material`] blends against the framebuffer (`.inf_lvl` schema v8). A
/// flat reflected enum (Details dropdown, like [`LightKind`]): `Opaque` is the
/// pre-v8 behaviour; `Masked` alpha-tests against [`Material::alpha_cutoff`];
/// `Translucent` alpha-blends.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Opaque,
    Masked,
    Translucent,
}

/// A renderable mesh reference: a built-in [`Primitive`] and/or a mesh-**asset**
/// GUID (P13.4).
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[reflect(Component, Default)]
pub struct MeshRef {
    pub primitive: Primitive,
    /// Mesh-asset GUID (P13.4 · `.inf_lvl` schema v7). `None` → render the
    /// built-in `primitive` (the pre-P13.4 behaviour). `Some(guid)` → the entity
    /// references a `.inf_mesh` asset; the **player** renders its real geometry —
    /// through the cook-derived `.inf_vmesh` meshlet path when virtualized
    /// geometry is enabled (the auto-tier's High tier), or the classic
    /// discrete-LOD fallback otherwise — while the interactive editor viewport
    /// keeps drawing the `primitive` placeholder (a documented gap: the
    /// asset-DB-in-viewport binding is a follow-up).
    ///
    /// Additive field: `#[serde(default)]` so pre-v7 `.inf_lvl` files load with
    /// `None`; `#[reflect(ignore)]` (assigned by drag-drop, not the Details
    /// numeric grid, like [`Sprite::texture`]) — still serde-persisted, and the
    /// cook walks it into the pack dependency closure so a referenced mesh (and
    /// its derived vmesh) ship with the level.
    #[serde(default)]
    #[reflect(ignore)]
    pub asset: Option<Uuid>,
}

/// Surface appearance: the metallic-roughness PBR parameter block (Phase 7).
/// New fields carry `#[serde(default)]` so pre-P7 `.inf_lvl` files that only
/// stored `base_color` still deserialize.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Material {
    pub base_color: Color,
    /// 0 = dielectric, 1 = metal.
    #[serde(default = "default_metallic")]
    pub metallic: f32,
    /// Perceptual roughness, 0 = mirror-smooth, 1 = fully rough.
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    /// Self-emitted color (rgb; alpha ignored). Black = non-emissive.
    #[serde(default = "default_emissive")]
    pub emissive: Color,
    /// **How bright that colour actually is** (`.inf_lvl` schema v26, wave
    /// VIS1a). A multiplier on [`emissive`](Self::emissive), which is an 8-bit
    /// sRGB colour and therefore cannot exceed 1.0 in any channel.
    ///
    /// **This is what made authored emissive unreachable.** The renderer's HDR
    /// path thresholds bloom at a linear luminance of 1.0 by default, and the
    /// brightest colour the Details panel can produce is exactly 1.0 — so an
    /// emissive material authored in the editor could touch the threshold and
    /// never cross it, while `hdr_bloom.png` constructs a value of 9.0 by hand
    /// because a golden may reach past the UI. A colour picker answers "what
    /// colour", and nothing answered "how bright".
    ///
    /// Additive field: `#[serde(default = "default_emissive_intensity")]` → 1.0,
    /// which reproduces the pre-v26 behaviour exactly (the colour, unscaled).
    #[serde(default = "default_emissive_intensity")]
    pub emissive_intensity: f32,
    /// Blend / transparency mode (schema v8). Additive field: `#[serde(default)]`
    /// → [`BlendMode::Opaque`], the pre-v8 behaviour.
    #[serde(default)]
    pub blend: BlendMode,
    /// Alpha-test threshold used when `blend == BlendMode::Masked`: fragments with
    /// alpha below this are discarded (schema v8). Additive field.
    #[serde(default = "default_alpha_cutoff")]
    pub alpha_cutoff: f32,
    /// The `.inf_mat` this surface is bound to (P26.3b · `.inf_lvl` schema v22).
    ///
    /// **`None` is exactly today's behaviour and stays the no-texture path**: the
    /// scalars above are the whole material, the renderer's per-instance
    /// attributes carry them, and the fallback is structural rather than a
    /// runtime branch. `Some(guid)` says *these scalars came from that material,
    /// and that material may also name textures* — which is the fact nothing on
    /// disk recorded before this bump, and the reason `.inf_mat` texture
    /// references resolved in neither host (the P26.3 spec-clause-4 gap).
    ///
    /// Apply-material still flattens the parameters onto the fields above
    /// (P7.1), so a level whose binding cannot be resolved renders exactly as it
    /// did: the binding *adds* the texture edge, it never becomes the only copy
    /// of the numbers.
    ///
    /// Additive field: `#[serde(default)]` so a pre-v22 `.inf_lvl` loads with
    /// `None`; `#[reflect(ignore)]` because it is assigned by drag-drop and the
    /// apply-material command, not by the Details numeric grid — the same
    /// arrangement [`MeshRef::asset`] and `Sprite::texture` have. Still
    /// serde-persisted, and the cook walks it into the pack dependency closure
    /// so a referenced material (and the `.inf_tex` containers it names) ship
    /// with the level.
    #[serde(default)]
    #[reflect(ignore)]
    pub asset: Option<Uuid>,
}

fn default_metallic() -> f32 {
    0.0
}
fn default_roughness() -> f32 {
    0.5
}
fn default_emissive() -> Color {
    Color::new(0.0, 0.0, 0.0, 1.0)
}
fn default_emissive_intensity() -> f32 {
    1.0
}
fn default_alpha_cutoff() -> f32 {
    0.5
}

impl Material {
    /// **The emitted radiance this surface actually contributes** — the authored
    /// colour times [`emissive_intensity`](Self::emissive_intensity), rgb (schema
    /// v26, wave VIS1a).
    ///
    /// A method rather than a multiplication at each packing site, because there
    /// are five of them across two hosts and "the editor viewport and the shipped
    /// player agree about what a material emits" is exactly the sort of claim
    /// that five copies of one expression quietly stop supporting. The intensity
    /// is clamped at zero: a negative multiplier is a light that removes light.
    pub fn emissive_linear(&self) -> [f32; 3] {
        let k = self.emissive_intensity.max(0.0);
        let e = self.emissive.to_array();
        [e[0] * k, e[1] * k, e[2] * k]
    }
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Color::new(0.8, 0.8, 0.8, 1.0),
            metallic: default_metallic(),
            roughness: default_roughness(),
            emissive: default_emissive(),
            emissive_intensity: default_emissive_intensity(),
            blend: BlendMode::Opaque,
            alpha_cutoff: default_alpha_cutoff(),
            asset: None,
        }
    }
}

/// Normalized UV sub-rect into a sprite's texture/atlas. Default = the full
/// texture `(0,0)-(1,1)`. As a nested struct it isn't surfaced in the generic
/// Details grid (the reflection walker only descends value types); atlas rects
/// are authored by the sprite-sheet slicer (P8.2) and always serde-persisted.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Default)]
pub struct AtlasRect {
    pub min: Vec2d,
    pub max: Vec2d,
}

impl Default for AtlasRect {
    fn default() -> Self {
        Self {
            min: Vec2d::ZERO,
            max: Vec2d::ONE,
        }
    }
}

/// How a [`Sprite`] orients its quad relative to the camera (P8.4a, 2.5D).
///
/// A **flat** reflected enum (like [`ColliderShape2DKind`]): the Details grid
/// surfaces it on the unit-enum dropdown. `None` is the classic world-XY-plane
/// sprite (facing +Z); the billboard modes rotate the quad in the vertex shader
/// so a 2D sprite reads as a card standing up in a 3D scene.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BillboardMode {
    /// Fixed in the world XY plane, facing +Z (the pre-2.5D behaviour).
    #[default]
    None,
    /// Always faces the camera fully (rotates about both axes) — the classic
    /// "particle" billboard.
    Spherical,
    /// Faces the camera but stays upright about world **+Y** (trees, characters
    /// in a 2.5D scene): rotates only about the vertical axis.
    Cylindrical,
}

/// A 2D sprite: a textured, tinted, sortable quad. Rendered by the 2D pass
/// (`inf-render-2d` batcher + `inf-render` sprite pass) over the 3D scene.
///
/// Visibility follows the shared [`Visibility`]/`ComputedVisibility` components
/// (like meshes) — there is no per-sprite `visible` field.
///
/// Additive component (P8.1a): every new field carries `#[serde(default)]` so
/// levels saved before this component existed still load.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Sprite {
    /// Texture/atlas asset GUID; `None` → the renderer's 1×1 white fallback.
    ///
    /// `#[reflect(ignore)]`: the reflection Details grid has no asset-ref widget
    /// yet (a follow-up shared with material texture refs), so the texture is
    /// assigned via drag-drop rather than typed. It is still serde-persisted.
    #[serde(default)]
    #[reflect(ignore)]
    pub texture: Option<Uuid>,
    /// Quad extent in world units (width, height).
    pub size: Vec2d,
    /// Normalized anchor in `[0,1]²`; `(0.5, 0.5)` centers the quad.
    pub pivot: Vec2d,
    /// Linear tint multiplied with the sampled texel (straight alpha).
    pub color: Color,
    /// Atlas UV sub-rect (defaults to the full texture).
    #[serde(default)]
    pub atlas_rect: AtlasRect,
    /// Coarse draw bucket (lower draws further back).
    #[serde(default)]
    pub sorting_layer: i32,
    /// Fine ordering within a layer (lower draws further back).
    #[serde(default)]
    pub order: i32,
    #[serde(default)]
    pub flip_x: bool,
    #[serde(default)]
    pub flip_y: bool,
    /// Camera-facing mode (P8.4a). Additive field: `#[serde(default)]` →
    /// [`BillboardMode::None`], so a sprite saved before 2.5D (and every existing
    /// self-describing payload) round-trips unchanged.
    #[serde(default)]
    pub billboard: BillboardMode,
}

impl Default for Sprite {
    fn default() -> Self {
        Self {
            texture: None,
            size: Vec2d::ONE,
            pivot: Vec2d::splat(0.5),
            color: Color::WHITE,
            atlas_rect: AtlasRect::default(),
            sorting_layer: 0,
            order: 0,
            flip_x: false,
            flip_y: false,
            billboard: BillboardMode::None,
        }
    }
}

/// Side length (in tiles) of one square [`TileChunk`]. A tilemap stores tiles
/// in fixed `CHUNK_DIM × CHUNK_DIM` blocks keyed by chunk coordinate, so a huge
/// (or negative-addressed) map only allocates memory for the regions that are
/// actually painted, and serialization/iteration stay chunk-granular. Must match
/// `inf_render_2d::TILE_CHUNK_DIM` (the renderer expands chunks with the same
/// stride).
pub const CHUNK_DIM: i32 = 32;

/// Number of tiles in one [`TileChunk`] (`CHUNK_DIM²` = 1024).
pub const CHUNK_TILES: usize = (CHUNK_DIM * CHUNK_DIM) as usize;

/// Row-major local index of tile `(lx, ly)` within a chunk (`0 ≤ lx,ly < CHUNK_DIM`).
#[inline]
fn chunk_index(lx: i32, ly: i32) -> usize {
    (ly * CHUNK_DIM + lx) as usize
}

/// Split a global tile coordinate into `(chunk, local)`, correct for negatives
/// (floored chunk, Euclidean remainder → local always in `0..CHUNK_DIM`).
#[inline]
fn split_coord(v: i32) -> (i32, i32) {
    (v.div_euclid(CHUNK_DIM), v.rem_euclid(CHUNK_DIM))
}

/// A fixed `CHUNK_DIM × CHUNK_DIM` block of tile indices. `0` = empty; any other
/// value is a **1-based** index into the tilemap's atlas grid (so cell `0` is
/// still addressable as index `1`).
///
/// Stored as a heap-boxed fixed array (dense within a painted region, cheap to
/// index). Serde (manual — `serde` only derives arrays up to length 32) writes
/// it as a flat sequence of `CHUNK_TILES` `u32`s; the chunk map itself is
/// `#[reflect(ignore)]` on [`Tilemap`], so this type is not reflected.
#[derive(Clone, PartialEq, Eq)]
pub struct TileChunk {
    tiles: Box<[u32; CHUNK_TILES]>,
}

impl TileChunk {
    /// An all-empty chunk.
    pub fn empty() -> Self {
        Self {
            tiles: Box::new([0; CHUNK_TILES]),
        }
    }

    /// The tile index at local `(lx, ly)` (`0` = empty).
    pub fn get(&self, lx: i32, ly: i32) -> u32 {
        self.tiles[chunk_index(lx, ly)]
    }

    /// Write the tile index at local `(lx, ly)`.
    pub fn set(&mut self, lx: i32, ly: i32, idx: u32) {
        self.tiles[chunk_index(lx, ly)] = idx;
    }

    /// True when every tile is empty (used to drop chunks that were fully erased).
    pub fn is_empty(&self) -> bool {
        self.tiles.iter().all(|&t| t == 0)
    }

    /// The raw row-major tile array (length [`CHUNK_TILES`]); consumed by the
    /// renderer's chunk expansion.
    pub fn tiles(&self) -> &[u32] {
        &self.tiles[..]
    }
}

impl Default for TileChunk {
    fn default() -> Self {
        Self::empty()
    }
}

impl std::fmt::Debug for TileChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let occupied = self.tiles.iter().filter(|&&t| t != 0).count();
        f.debug_struct("TileChunk")
            .field("occupied", &occupied)
            .finish()
    }
}

impl Serialize for TileChunk {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        // Slices serialize as a length-prefixed sequence (bincode) / array (TOML/JSON).
        self.tiles[..].serialize(s)
    }
}

impl<'de> Deserialize<'de> for TileChunk {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v: Vec<u32> = Vec::deserialize(d)?;
        if v.len() != CHUNK_TILES {
            return Err(serde::de::Error::invalid_length(
                v.len(),
                &"a chunk of CHUNK_TILES (1024) tile indices",
            ));
        }
        let mut tiles = Box::new([0u32; CHUNK_TILES]);
        tiles.copy_from_slice(&v);
        Ok(Self { tiles })
    }
}

/// Inclusive tile-coordinate bounding box of a tilemap's occupied tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileBounds {
    pub min_x: i32,
    pub min_y: i32,
    pub max_x: i32,
    pub max_y: i32,
}

/// A 2D tilemap: a sparse, chunked grid of atlas-indexed tiles rendered as one
/// batch per occupied chunk by the 2D pass (`inf-render-2d` expansion +
/// `inf-render` sprite pass).
///
/// Tiles are addressed by signed integer grid coordinates and stored in fixed
/// [`TileChunk`]s (see [`CHUNK_DIM`]) keyed by chunk coordinate in a
/// `BTreeMap` — deterministic serialization/iteration and memory proportional to
/// the painted area, not the addressable range. Visibility follows the shared
/// [`Visibility`]/`ComputedVisibility` components (like sprites/meshes).
///
/// Additive component: every field carries `#[serde(default)]` so older levels
/// still load. The Details grid surfaces the scalar fields only; the chunk map
/// is `#[reflect(ignore)]` (painting is a dedicated tool, P8.x).
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Tilemap {
    /// Atlas texture asset GUID; `None` → the renderer's 1×1 white fallback.
    /// `#[reflect(ignore)]` + serde-persisted, exactly like [`Sprite::texture`].
    #[serde(default)]
    #[reflect(ignore)]
    pub texture: Option<Uuid>,
    /// World units per tile (width, height) — one tile cell's extent.
    pub tile_size: Vec2d,
    /// Atlas grid width in cells: a 1-based tile index maps to atlas cell
    /// `index - 1`, at `(col, row) = ((index-1) % atlas_cols, (index-1) / atlas_cols)`.
    #[serde(default = "default_atlas_dim")]
    pub atlas_cols: u32,
    /// Atlas grid height in cells.
    #[serde(default = "default_atlas_dim")]
    pub atlas_rows: u32,
    /// Coarse draw bucket (lower draws further back).
    #[serde(default)]
    pub sorting_layer: i32,
    /// Fine ordering within a layer (lower draws further back).
    #[serde(default)]
    pub order: i32,
    /// Linear tint multiplied with every sampled tile texel (straight alpha).
    pub tint: Color,
    /// Sparse tile storage: chunk coordinate → its `CHUNK_DIM²` block. Empty
    /// chunks are never stored (erasing the last tile drops the chunk).
    ///
    /// Serialized as a flat, deterministically-ordered sequence of
    /// `(x, y, chunk)` entries (not a native map) so it encodes in formats that
    /// forbid non-string map keys — JSON/TOML as well as the `.inf_lvl` bincode
    /// payload. `BTreeMap` iteration keeps the order stable.
    #[serde(default, with = "chunk_map_serde")]
    #[reflect(ignore)]
    pub chunks: BTreeMap<(i32, i32), TileChunk>,
}

/// serde adapter: `BTreeMap<(i32,i32), TileChunk>` ⇄ a flat `(x, y, chunk)`
/// sequence (portable across bincode/TOML/JSON; deterministic via `BTreeMap`).
mod chunk_map_serde {
    use super::{BTreeMap, TileChunk};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        map: &BTreeMap<(i32, i32), TileChunk>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let entries: Vec<(i32, i32, &TileChunk)> =
            map.iter().map(|(&(x, y), c)| (x, y, c)).collect();
        entries.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<(i32, i32), TileChunk>, D::Error> {
        let entries: Vec<(i32, i32, TileChunk)> = Vec::deserialize(d)?;
        Ok(entries.into_iter().map(|(x, y, c)| ((x, y), c)).collect())
    }
}

fn default_atlas_dim() -> u32 {
    1
}

impl Default for Tilemap {
    fn default() -> Self {
        Self {
            texture: None,
            tile_size: Vec2d::ONE,
            atlas_cols: 1,
            atlas_rows: 1,
            sorting_layer: 0,
            order: 0,
            tint: Color::WHITE,
            chunks: BTreeMap::new(),
        }
    }
}

impl Tilemap {
    /// The tile index at grid coordinate `(x, y)` (`0` = empty). Reads of
    /// unpainted regions return `0` without allocating.
    pub fn get_tile(&self, x: i32, y: i32) -> u32 {
        let (cx, lx) = split_coord(x);
        let (cy, ly) = split_coord(y);
        self.chunks
            .get(&(cx, cy))
            .map(|c| c.get(lx, ly))
            .unwrap_or(0)
    }

    /// Set the tile index at grid coordinate `(x, y)`. `idx == 0` clears the
    /// tile (and drops the chunk if it becomes empty); a non-zero `idx` is a
    /// 1-based atlas cell and allocates the chunk on demand.
    pub fn set_tile(&mut self, x: i32, y: i32, idx: u32) {
        let (cx, lx) = split_coord(x);
        let (cy, ly) = split_coord(y);
        if idx == 0 {
            if let Some(chunk) = self.chunks.get_mut(&(cx, cy)) {
                chunk.set(lx, ly, 0);
                if chunk.is_empty() {
                    self.chunks.remove(&(cx, cy));
                }
            }
            return;
        }
        self.chunks.entry((cx, cy)).or_default().set(lx, ly, idx);
    }

    /// Clear the tile at grid coordinate `(x, y)` (equivalent to `set_tile(x, y, 0)`).
    pub fn clear_tile(&mut self, x: i32, y: i32) {
        self.set_tile(x, y, 0);
    }

    /// True when no tile is painted anywhere.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Iterate occupied chunks in deterministic (chunk-coordinate) order.
    pub fn occupied_chunks(&self) -> impl Iterator<Item = (&(i32, i32), &TileChunk)> {
        self.chunks.iter()
    }

    /// Inclusive tile-coordinate bounds of the painted tiles, or `None` when the
    /// map is empty. Scans occupied tiles (chunk-sparse), so cost is proportional
    /// to the painted area.
    pub fn bounds(&self) -> Option<TileBounds> {
        let mut acc: Option<TileBounds> = None;
        for (&(cx, cy), chunk) in &self.chunks {
            for ly in 0..CHUNK_DIM {
                for lx in 0..CHUNK_DIM {
                    if chunk.get(lx, ly) == 0 {
                        continue;
                    }
                    let gx = cx * CHUNK_DIM + lx;
                    let gy = cy * CHUNK_DIM + ly;
                    acc = Some(match acc {
                        None => TileBounds {
                            min_x: gx,
                            min_y: gy,
                            max_x: gx,
                            max_y: gy,
                        },
                        Some(b) => TileBounds {
                            min_x: b.min_x.min(gx),
                            min_y: b.min_y.min(gy),
                            max_x: b.max_x.max(gx),
                            max_y: b.max_y.max(gy),
                        },
                    });
                }
            }
        }
        acc
    }
}

/// A 9-slice sprite (P8.1c): a bordered panel whose four corners keep a fixed
/// world thickness while its edges stretch along one axis and its center
/// stretches along both. Expanded into nine quads by the 2D pass
/// (`inf_render_2d::expand_nine_slice`). Visibility follows the shared
/// [`Visibility`]/`ComputedVisibility` components.
///
/// Additive component: every field carries `#[serde(default)]` so older levels
/// still load.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct NineSlice {
    /// Texture/atlas asset GUID; `None` → the renderer's 1×1 white fallback.
    /// `#[reflect(ignore)]` + serde-persisted, exactly like [`Sprite::texture`].
    #[serde(default)]
    #[reflect(ignore)]
    pub texture: Option<Uuid>,
    /// Total panel extent in world units (width, height).
    pub size: Vec2d,
    /// Normalized border fractions of the **texture**: `[left, right, top,
    /// bottom]`. An array has no Details widget yet, so it is not surfaced in the
    /// generic grid (authored by the 9-slice tool, a follow-up); still
    /// serde-persisted.
    #[serde(default = "default_border_uv")]
    pub border_uv: [f64; 4],
    /// World thickness of the borders (`x` = left/right column, `y` = top/bottom
    /// row). Clamped to half the size at expansion so borders never overlap.
    #[serde(default = "default_border_world")]
    pub border_world: Vec2d,
    /// Linear tint multiplied with every cell texel (straight alpha).
    pub tint: Color,
    #[serde(default)]
    pub sorting_layer: i32,
    #[serde(default)]
    pub order: i32,
}

fn default_border_uv() -> [f64; 4] {
    [1.0 / 3.0; 4]
}
fn default_border_world() -> Vec2d {
    Vec2d::splat(0.25)
}

impl Default for NineSlice {
    fn default() -> Self {
        Self {
            texture: None,
            size: Vec2d::splat(2.0),
            border_uv: default_border_uv(),
            border_world: default_border_world(),
            tint: Color::WHITE,
            sorting_layer: 0,
            order: 0,
        }
    }
}

/// Horizontal alignment of a [`Text2D`] block about its anchor.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

/// A bitmap-text label (P8.1c): a string laid out as one monospace quad per
/// glyph, sampling a fixed-grid ASCII bitmap-font atlas. Expanded by the 2D pass
/// (`inf_render_2d::expand_text`). Visibility follows the shared
/// [`Visibility`]/`ComputedVisibility` components.
///
/// Additive component: every optional field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Text2D {
    /// The string to render (`'\n'` breaks lines).
    pub text: String,
    /// Bitmap-font atlas asset GUID; `None` → the renderer's built-in 8×8 font.
    /// `#[reflect(ignore)]` + serde-persisted, exactly like [`Sprite::texture`].
    #[serde(default)]
    #[reflect(ignore)]
    pub font_texture: Option<Uuid>,
    /// Font-atlas grid dimensions in glyph cells (built-in font = 16×6).
    #[serde(default = "default_glyph_cols")]
    pub glyph_cols: u32,
    #[serde(default = "default_glyph_rows")]
    pub glyph_rows: u32,
    /// Codepoint of atlas cell 0 (usually 32 = space).
    #[serde(default = "default_first_codepoint")]
    pub first_codepoint: u32,
    /// World size of one glyph cell (width, height).
    pub glyph_size: Vec2d,
    /// Extra advance as a fraction of glyph width (0 = tight monospace).
    #[serde(default)]
    pub tracking: f64,
    /// Linear tint multiplied with every glyph texel (straight alpha).
    pub tint: Color,
    #[serde(default)]
    pub sorting_layer: i32,
    #[serde(default)]
    pub order: i32,
    /// Per-line horizontal alignment. (Editable in Details via the enum-dropdown
    /// reflection support used by [`LightKind`].)
    #[serde(default)]
    pub halign: TextAlign,
}

fn default_glyph_cols() -> u32 {
    16
}
fn default_glyph_rows() -> u32 {
    6
}
fn default_first_codepoint() -> u32 {
    32
}

impl Default for Text2D {
    fn default() -> Self {
        Self {
            text: String::new(),
            font_texture: None,
            glyph_cols: default_glyph_cols(),
            glyph_rows: default_glyph_rows(),
            first_codepoint: default_first_codepoint(),
            glyph_size: Vec2d::ONE,
            tracking: 0.0,
            tint: Color::WHITE,
            sorting_layer: 0,
            order: 0,
            halign: TextAlign::Left,
        }
    }
}

/// A minimal 2D light (P8.1c): a soft radial falloff in the sprite plane, added
/// to the scene ambient by the sprite fragment shader. A per-light **layer mask**
/// (restricting which sorting layers it affects) is a documented follow-up — for
/// now every 2D light affects every sprite/tile/text/9-slice fragment.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Light2D {
    /// Linear light color.
    pub color: Color,
    /// Brightness multiplier.
    pub intensity: f32,
    /// World-space falloff radius (contribution is `smoothstep(radius, 0, dist)`).
    pub radius: f32,
}

impl Default for Light2D {
    fn default() -> Self {
        Self {
            color: Color::WHITE,
            intensity: 1.0,
            radius: 5.0,
        }
    }
}

#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LightKind {
    #[default]
    Directional,
    Point,
    Spot,
}

/// A light source.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Light {
    pub kind: LightKind,
    pub color: Color,
    pub intensity: f32,
    /// Falloff / influence radius in world units (schema v8). `0.0` = unbounded
    /// (the pre-v8 behaviour: matches the current windowed inverse-square shader).
    /// Additive field: `#[serde(default)]` → `0.0`.
    #[serde(default)]
    pub range: f32,
    /// Spot inner cone half-angle in degrees (full brightness inside; schema v8).
    #[serde(default = "default_inner_cone_deg")]
    pub inner_cone_deg: f32,
    /// Spot outer cone half-angle in degrees (falloff edge; schema v8).
    #[serde(default = "default_outer_cone_deg")]
    pub outer_cone_deg: f32,
    /// Whether this light casts shadows (schema v8). Additive field:
    /// `#[serde(default)]` → `true`.
    #[serde(default = "default_cast_shadows")]
    pub cast_shadows: bool,
}

fn default_inner_cone_deg() -> f32 {
    30.0
}
fn default_outer_cone_deg() -> f32 {
    40.0
}
fn default_cast_shadows() -> bool {
    true
}

impl Default for Light {
    fn default() -> Self {
        Self {
            kind: LightKind::Point,
            color: Color::WHITE,
            intensity: 1.0,
            range: 0.0,
            inner_cone_deg: default_inner_cone_deg(),
            outer_cone_deg: default_outer_cone_deg(),
            cast_shadows: default_cast_shadows(),
        }
    }
}

/// How a [`RigidBody2D`] is driven by the 2D solver (mirrors
/// [`inf_physics::BodyKind`], kept as a plain reflected enum so the Details
/// grid surfaces it on the enum-dropdown widget — like [`LightKind`]).
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BodyKind2D {
    /// Never moved by the solver; infinite mass (floors, walls).
    #[default]
    Static,
    /// Moved only by its `Transform` (a moving platform / mover); pushes dynamic
    /// bodies but is not pushed back.
    Kinematic,
    /// Fully simulated: gravity, forces, impulses, and contacts move it.
    Dynamic,
}

/// A 2D rigid body (P8.3b). Pairs with a [`Collider2D`] on the same entity; the
/// `PhysicsBridge2D` in `inf-physics` mirrors it into the rapier world.
///
/// Additive component: every field carries `#[serde(default)]` so a minimal
/// payload (and any future field) round-trips.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct RigidBody2D {
    /// Static / Kinematic / Dynamic.
    #[serde(default)]
    pub kind: BodyKind2D,
    /// Per-body multiplier on world gravity (`1` = full, `0` = float, `<0` =
    /// anti-gravity). Dynamic bodies only.
    #[serde(default = "default_gravity_scale")]
    pub gravity_scale: f64,
    /// Lock rotation so the body never spins (typical for characters).
    #[serde(default)]
    pub fixed_rotation: bool,
    /// Linear velocity decay per second (drag).
    #[serde(default)]
    pub linear_damping: f64,
    /// Angular velocity decay per second.
    #[serde(default)]
    pub angular_damping: f64,
    /// Continuous Collision Detection (P12.1): stops a fast body from tunnelling
    /// through thin geometry in one step, at extra solver cost. Enable for
    /// bullets/projectiles.
    #[serde(default)]
    pub ccd_enabled: bool,
}

fn default_gravity_scale() -> f64 {
    1.0
}

impl Default for RigidBody2D {
    fn default() -> Self {
        Self {
            kind: BodyKind2D::Static,
            gravity_scale: default_gravity_scale(),
            fixed_rotation: false,
            linear_damping: 0.0,
            angular_damping: 0.0,
            ccd_enabled: false,
        }
    }
}

/// The shape family of a [`Collider2D`].
///
/// A **flat** enum (no per-variant data) rather than a data-carrying enum: the
/// Details reflection walker (`crate::props`) only descends value types and
/// surfaces enums as a *unit* dropdown, so the shape's numeric parameters live
/// in sibling fields ([`Collider2D::half_extents`] / [`Collider2D::radius`])
/// that the grid can edit. Which field applies is documented per variant.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColliderShape2DKind {
    /// Axis-aligned box; uses [`Collider2D::half_extents`] (radius ignored).
    #[default]
    Box,
    /// Circle; uses [`Collider2D::radius`] (half_extents ignored).
    Circle,
    /// Vertical capsule; segment half-length = `half_extents.y`, swept by
    /// [`Collider2D::radius`] (half_extents.x ignored).
    Capsule,
}

/// How two colliders' friction (or restitution) coefficients combine into the
/// effective value for their contact (P12.1). A flat reflected enum (Details
/// surfaces it as a dropdown), mirroring [`inf_physics::CombineRule`].
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CombineRule {
    /// `(a + b) / 2` — the default, balanced and intuitive.
    #[default]
    Average,
    /// `min(a, b)` — "slippery wins".
    Min,
    /// `a * b` — both must be high.
    Multiply,
    /// `max(a, b)` — "sticky/bouncy wins".
    Max,
}

fn default_collision_mask() -> u32 {
    u32::MAX
}

/// A 2D collider (P8.3b). Attaches to the entity's [`RigidBody2D`] (or, if the
/// entity has none, an implicit static body) in the rapier world.
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Collider2D {
    /// Box / Circle / Capsule (selects which of the fields below apply).
    #[serde(default)]
    pub shape_kind: ColliderShape2DKind,
    /// Box half-width/half-height; capsule uses `.y` as the segment half-length.
    /// (Ignored for Circle.)
    #[serde(default = "default_half_extents")]
    pub half_extents: Vec2d,
    /// Circle / capsule radius. (Ignored for Box.)
    #[serde(default = "default_radius")]
    pub radius: f64,
    /// Offset from the body origin, in the body frame (world units).
    #[serde(default)]
    pub offset: Vec2d,
    /// Coulomb friction coefficient.
    #[serde(default = "default_friction")]
    pub friction: f64,
    /// Bounciness in `[0, 1]`.
    #[serde(default)]
    pub restitution: f64,
    /// Mass density (drives a dynamic body's mass/inertia).
    ///
    /// # THE DEFAULT IS A PLACEHOLDER, NOT A MATERIAL
    ///
    /// [`default_density`] is rapier's `1.0` — **one kilogram per cubic metre**,
    /// which is lighter than air and is not any substance a level contains. It is
    /// the solver's "no opinion" value, not a sensible starting point, and an
    /// authored collider that leaves it alone gets a body whose mass is a
    /// thousandth of what the geometry looks like.
    ///
    /// This has now been paid for twice. P20.2's buoyancy work found it first and
    /// put the honest number on `Buoyancy::density_kg_m3` rather than reading this
    /// one. P22.4 found it again from the other end: a 0.4 m wheel at the default
    /// weighs **268 grams**, so a car-bomb impulse sized against 5-tonne fracture
    /// chunks threw it out of the level — and `Destructible::density_kg_m3` exists
    /// precisely so a chunk's mass never comes from here.
    ///
    /// So: **author it.** Classes, kg/m³ — pine 500, oak 750, brick 1900, concrete
    /// 2400, granite 2700, steel 7850, rubber ~1100, and a hollow shell (a car
    /// body, a crate) is far lower than its material — 150 is right for a 4 × 1 × 2 m
    /// car at ~1200 kg. The default is left at `1.0` deliberately rather than
    /// "fixed" to something plausible: changing it would silently re-mass every
    /// committed level in the repository, and a wrong number that *looks* right is
    /// worse than one that is obviously a placeholder.
    #[serde(default = "default_density")]
    pub density: f64,
    /// A trigger volume: detects overlaps but generates no contact force.
    #[serde(default)]
    pub sensor: bool,
    /// Collision-layer membership bitmask (P12.1): which of the 32 named layers
    /// this collider belongs to. Default = all (`u32::MAX`). Raw `u32` in Details
    /// (a named-bitmask widget is a follow-up); layers are named per-project in
    /// `.infinity/collision_layers.toml`.
    #[serde(default = "default_collision_mask")]
    pub collision_memberships: u32,
    /// Collision-layer filter bitmask (P12.1): which layers this collider will
    /// interact with. Default = all. Two colliders touch iff each is in the
    /// other's filter.
    #[serde(default = "default_collision_mask")]
    pub collision_filter: u32,
    /// How this collider's friction combines with a contacting collider's (P12.1).
    #[serde(default)]
    pub friction_combine: CombineRule,
    /// How this collider's restitution combines with a contacting collider's.
    #[serde(default)]
    pub restitution_combine: CombineRule,
}

fn default_half_extents() -> Vec2d {
    Vec2d::splat(0.5)
}
fn default_radius() -> f64 {
    0.5
}
fn default_friction() -> f64 {
    0.5
}
fn default_density() -> f64 {
    1.0
}

impl Default for Collider2D {
    fn default() -> Self {
        Self {
            shape_kind: ColliderShape2DKind::Box,
            half_extents: default_half_extents(),
            radius: default_radius(),
            offset: Vec2d::ZERO,
            friction: default_friction(),
            restitution: 0.0,
            density: default_density(),
            sensor: false,
            collision_memberships: default_collision_mask(),
            collision_filter: default_collision_mask(),
            friction_combine: CombineRule::Average,
            restitution_combine: CombineRule::Average,
        }
    }
}

/// Tuning for a kinematic character mover (P8.3b) — an entity driven by the
/// Blueprint `physics2d.move_and_slide` node rather than the dynamic solver.
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct CharacterController2D {
    /// Steepest slope (degrees from "up") the character can walk up.
    #[serde(default = "default_max_slope_deg")]
    pub max_slope_deg: f64,
    /// Snap to ground when within this distance (world units; `0` disables — the
    /// character flies off ledges/ramps).
    #[serde(default = "default_snap_to_ground")]
    pub snap_to_ground: f64,
    /// Skin width kept between the character and the world (must be > 0 for
    /// numerical stability).
    #[serde(default = "default_mover_offset")]
    pub offset: f64,
}

fn default_max_slope_deg() -> f64 {
    45.0
}
fn default_snap_to_ground() -> f64 {
    0.2
}
fn default_mover_offset() -> f64 {
    0.02
}

impl Default for CharacterController2D {
    fn default() -> Self {
        Self {
            max_slope_deg: default_max_slope_deg(),
            snap_to_ground: default_snap_to_ground(),
            offset: default_mover_offset(),
        }
    }
}

/// How a [`RigidBody3D`] is driven by the 3D solver (the `d3` mirror of
/// [`BodyKind2D`]; a plain reflected enum so the Details grid surfaces it on the
/// enum-dropdown widget — like [`LightKind`]).
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BodyKind3D {
    /// Never moved by the solver; infinite mass (floors, walls, level geometry).
    #[default]
    Static,
    /// Moved only by its `Transform` (a moving platform / mover); pushes dynamic
    /// bodies but is not pushed back.
    Kinematic,
    /// Fully simulated: gravity, forces, impulses, and contacts move it.
    Dynamic,
}

/// A 3D rigid body (P9.1). Pairs with a [`Collider3D`] on the same entity; the
/// `PhysicsBridge3D` in `inf-physics` mirrors it into the rapier world. The `d3`
/// mirror of [`RigidBody2D`].
///
/// Additive component: every field carries `#[serde(default)]` so a minimal
/// payload (and any future field) round-trips.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct RigidBody3D {
    /// Static / Kinematic / Dynamic.
    #[serde(default)]
    pub kind: BodyKind3D,
    /// Per-body multiplier on world gravity (`1` = full, `0` = float, `<0` =
    /// anti-gravity). Dynamic bodies only.
    #[serde(default = "default_gravity_scale")]
    pub gravity_scale: f64,
    /// Lock all rotation so the body never spins (typical for characters).
    #[serde(default)]
    pub fixed_rotation: bool,
    /// Linear velocity decay per second (drag).
    #[serde(default)]
    pub linear_damping: f64,
    /// Angular velocity decay per second.
    #[serde(default)]
    pub angular_damping: f64,
    /// Continuous Collision Detection (P12.1): stops a fast body from tunnelling
    /// through thin geometry in one step, at extra solver cost. Enable for
    /// bullets/projectiles.
    #[serde(default)]
    pub ccd_enabled: bool,
}

impl Default for RigidBody3D {
    fn default() -> Self {
        Self {
            kind: BodyKind3D::Static,
            gravity_scale: default_gravity_scale(),
            fixed_rotation: false,
            linear_damping: 0.0,
            angular_damping: 0.0,
            ccd_enabled: false,
        }
    }
}

/// The shape family of a [`Collider3D`].
///
/// A **flat** enum (no per-variant data), like [`ColliderShape2DKind`]: the
/// Details reflection walker surfaces it as a *unit* dropdown, so the shape's
/// numeric parameters live in sibling fields ([`Collider3D::half_extents`] /
/// [`Collider3D::radius`]). Trimesh (static-mesh) colliders are intentionally
/// omitted — they are not authored as a primitive shape and land with P12.
///
/// # Why P22.2's convex hull is not here either
///
/// `inf_physics::d3::ColliderShape3D` gained a `ConvexHull { points }` variant so
/// a fracture chunk can be a *dynamic* body with a real mass. It deliberately has
/// **no** counterpart in this enum, for exactly the reason `Trimesh` has none: a
/// hull is a point cloud, not three numbers in sibling fields. Adding it would
/// mean either a unit variant whose points come from nowhere (a dropdown an
/// author can pick that silently produces no collider) or a `Vec<Vec3d>` in a
/// `Copy` component — which would grow the wire record and cost a scene-schema
/// bump to hold data nobody types in by hand.
///
/// A chunk's hull points are *cook output* (`inf_mesh::fracture`), read straight
/// from the `.inf_fracture` by the runtime bridge that builds the chunk bodies —
/// the same route `voxel_chunk_guid`'s trimesh colliders take, and they are not
/// in this enum either. So this remains the **authored primitive** vocabulary,
/// and nothing about P22.2 moves its bytes.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColliderShape3DKind {
    /// Axis-aligned box; uses [`Collider3D::half_extents`] (radius ignored).
    #[default]
    Box,
    /// Sphere; uses [`Collider3D::radius`] (half_extents ignored).
    Sphere,
    /// Vertical capsule; segment half-length = `half_extents.y`, swept by
    /// [`Collider3D::radius`] (half_extents.x/z ignored).
    Capsule,
}

/// A 3D collider (P9.1). Attaches to the entity's [`RigidBody3D`] (or, if the
/// entity has none, an implicit static body) in the rapier world. The `d3`
/// mirror of [`Collider2D`].
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Collider3D {
    /// Box / Sphere / Capsule (selects which of the fields below apply).
    #[serde(default)]
    pub shape_kind: ColliderShape3DKind,
    /// Box half-extents; capsule uses `.y` as the segment half-length.
    /// (Ignored for Sphere.)
    #[serde(default = "default_half_extents_3d")]
    pub half_extents: Vec3d,
    /// Sphere / capsule radius. (Ignored for Box.)
    #[serde(default = "default_radius")]
    pub radius: f64,
    /// Offset from the body origin, in the body frame (world units).
    #[serde(default)]
    pub offset: Vec3d,
    /// Coulomb friction coefficient.
    #[serde(default = "default_friction")]
    pub friction: f64,
    /// Bounciness in `[0, 1]`.
    #[serde(default)]
    pub restitution: f64,
    /// Mass density (drives a dynamic body's mass/inertia).
    ///
    /// # THE DEFAULT IS A PLACEHOLDER, NOT A MATERIAL
    ///
    /// [`default_density`] is rapier's `1.0` — **one kilogram per cubic metre**,
    /// which is lighter than air and is not any substance a level contains. It is
    /// the solver's "no opinion" value, not a sensible starting point, and an
    /// authored collider that leaves it alone gets a body whose mass is a
    /// thousandth of what the geometry looks like.
    ///
    /// This has now been paid for twice. P20.2's buoyancy work found it first and
    /// put the honest number on `Buoyancy::density_kg_m3` rather than reading this
    /// one. P22.4 found it again from the other end: a 0.4 m wheel at the default
    /// weighs **268 grams**, so a car-bomb impulse sized against 5-tonne fracture
    /// chunks threw it out of the level — and `Destructible::density_kg_m3` exists
    /// precisely so a chunk's mass never comes from here.
    ///
    /// So: **author it.** Classes, kg/m³ — pine 500, oak 750, brick 1900, concrete
    /// 2400, granite 2700, steel 7850, rubber ~1100, and a hollow shell (a car
    /// body, a crate) is far lower than its material — 150 is right for a 4 × 1 × 2 m
    /// car at ~1200 kg. The default is left at `1.0` deliberately rather than
    /// "fixed" to something plausible: changing it would silently re-mass every
    /// committed level in the repository, and a wrong number that *looks* right is
    /// worse than one that is obviously a placeholder.
    #[serde(default = "default_density")]
    pub density: f64,
    /// A trigger volume: detects overlaps but generates no contact force.
    #[serde(default)]
    pub sensor: bool,
    /// Collision-layer membership bitmask (P12.1): which of the 32 named layers
    /// this collider belongs to. Default = all (`u32::MAX`). Raw `u32` in Details
    /// (a named-bitmask widget is a follow-up); layers are named per-project in
    /// `.infinity/collision_layers.toml`.
    #[serde(default = "default_collision_mask")]
    pub collision_memberships: u32,
    /// Collision-layer filter bitmask (P12.1): which layers this collider will
    /// interact with. Default = all. Two colliders touch iff each is in the
    /// other's filter.
    #[serde(default = "default_collision_mask")]
    pub collision_filter: u32,
    /// How this collider's friction combines with a contacting collider's (P12.1).
    #[serde(default)]
    pub friction_combine: CombineRule,
    /// How this collider's restitution combines with a contacting collider's.
    #[serde(default)]
    pub restitution_combine: CombineRule,
}

fn default_half_extents_3d() -> Vec3d {
    Vec3d::splat(0.5)
}

impl Default for Collider3D {
    fn default() -> Self {
        Self {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: default_half_extents_3d(),
            radius: default_radius(),
            offset: Vec3d::ZERO,
            friction: default_friction(),
            restitution: 0.0,
            density: default_density(),
            sensor: false,
            collision_memberships: default_collision_mask(),
            collision_filter: default_collision_mask(),
            friction_combine: CombineRule::Average,
            restitution_combine: CombineRule::Average,
        }
    }
}

/// Tuning for a kinematic 3D character mover (P9.1) — an entity driven by a
/// Blueprint `physics3d.move_and_slide` node rather than the dynamic solver. The
/// `d3` mirror of [`CharacterController2D`].
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct CharacterController3D {
    /// Steepest slope (degrees from "up") the character can walk up.
    #[serde(default = "default_max_slope_deg")]
    pub max_slope_deg: f64,
    /// Snap to ground when within this distance (world units; `0` disables — the
    /// character flies off ledges/ramps).
    #[serde(default = "default_snap_to_ground")]
    pub snap_to_ground: f64,
    /// Skin width kept between the character and the world (must be > 0 for
    /// numerical stability).
    #[serde(default = "default_mover_offset")]
    pub offset: f64,
}

impl Default for CharacterController3D {
    fn default() -> Self {
        Self {
            max_slope_deg: default_max_slope_deg(),
            snap_to_ground: default_snap_to_ground(),
            offset: default_mover_offset(),
        }
    }
}

// ── P29.3 movement component v2 ─────────────────────────────────────────────
//
// The catalogue amendment's twelve movements, ALS's curve-driven settings model,
// and the mode enum that absorbs the P20 swim latch. The whole design note is on
// [`CharacterMovement`]; the enums below carry the parts of it that are wire
// contracts and therefore frozen on the day they are born.

/// **What a character is doing.** The frozen wire enum of §13's movement
/// catalogue (2026-08-15 amendment) plus its reserved growth room.
///
/// # Frozen on day one, and why that is not premature
///
/// This reaches `.inf_lvl` bytes, so bincode writes it as its **declaration
/// index** and the append-only law applies from the first commit. The catalogue
/// amendment is explicit that the whole discriminant set is frozen now rather
/// than grown per sub-phase: P29.3 is the scene's one allowed bump this phase,
/// and a mode that arrives in P29.4 or P29.7 without a slot would need a second
/// one. Four modes here — `Mantle`, `Ragdoll`, `Driving`, `Flying` — therefore
/// **exist without their mechanics**, and asking to enter one is a typed refusal
/// ([`MovementRefusal::ModeNotYetImplemented`]) rather than a stub that pretends.
/// A refusal is a value.
///
/// # The axis question (Ruling 4), and where this deviates
///
/// ALS's "what the character is doing" is a *product* of seven fields — 11 700
/// nominal combinations — and Ruling 4's answer is: stance **folds into** the
/// mode (crouch and prone change the capsule, the speed set and the integration
/// regime together, so they are modes), [`RotationMode`] stays its own frozen
/// enum, overlay stays an open interned id, and gait rides *inside* `Grounded`.
///
/// The deviation is that last clause: gait is the sibling field
/// [`CharacterMovement::gait`] rather than a payload on `Grounded`. The reason is
/// mechanical and lives two crates away — the Details grid's write-back door
/// (`crate::props`) applies an edited enum as `DynamicVariant::Unit`, so a
/// struct variant is a value the editor that must *tune* this component cannot
/// round-trip; and the ECS census pins wire enums by asserting a unit variant
/// encodes to one byte. `Fall` and `Swim` take the same medicine and split into
/// two discriminants each, which costs nothing: the sub-state is a closed
/// two-valued thing in both cases, and [`is_falling`](Self::is_falling) /
/// [`is_swimming`](Self::is_swimming) express the grouping in code.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum MovementMode {
    /// Standing locomotion. The gait tier is [`CharacterMovement::gait`].
    #[default]
    Grounded,
    /// Crouched: the crouch capsule, the crouch speed, standing again gated by
    /// an overhead-clearance sweep.
    Crouch,
    /// Prone: the minimum capsule and the crawl speed.
    Prone,
    /// Sliding, entered from sprint + crouch; friction is a function of slope.
    Slide,
    /// A roll — the double-tap-crouch action and the soft outcome of the landing
    /// classifier. The root-motion half is P29.4's.
    Roll,
    /// A ballistic dive that lands into prone or a roll, chosen by the landing
    /// classifier.
    Dive,
    /// Airborne with **full** air-control authority (a deliberate jump).
    FallFree,
    /// Airborne with **reduced** authority — walked off a ledge, knocked back,
    /// or dived. The distinction §13's catalogue calls "free vs controlled".
    FallControlled,
    /// Swimming at the surface. Absorbs the P20 latch.
    SwimSurface,
    /// Fully submerged.
    SwimUnder,
    /// Mantling/vaulting. **P29.4 owns the mechanics**; entering refuses here.
    Mantle,
    /// Physics-driven ragdoll. **P29.4 owns the mechanics**; entering refuses.
    Ragdoll,
    /// Driving a vehicle. **P29.7 owns the mechanics**; entering refuses.
    Driving,
    /// 6-DOF flight. **P29.7 owns the mechanics**; entering refuses.
    Flying,
    /// Reserved. A build that meets one refuses by name — see the type's docs.
    Reserved14,
    /// Reserved. A build that meets one refuses by name — see the type's docs.
    Reserved15,
    /// Reserved. A build that meets one refuses by name — see the type's docs.
    Reserved16,
    /// Reserved. A build that meets one refuses by name — see the type's docs.
    Reserved17,
}

impl MovementMode {
    /// `Some(index)` if this is a reserved slot a newer build wrote, else `None`.
    ///
    /// A reserved slot no reader ever asks about is a comment rather than a slot.
    pub fn reserved_slot(self) -> Option<u8> {
        match self {
            MovementMode::Reserved14 => Some(14),
            MovementMode::Reserved15 => Some(15),
            MovementMode::Reserved16 => Some(16),
            MovementMode::Reserved17 => Some(17),
            _ => None,
        }
    }

    /// Whether this mode's mechanics belong to a later sub-phase, so that asking
    /// to enter it is a typed refusal rather than a stub.
    ///
    /// **P29.4 took two of the four** — `Mantle` and `Ragdoll` (the ledge probe
    /// and the warp; the articulated handoff and the pose-matched get-up) — and
    /// **P29.7 took the last two**: `Driving` is a raycast vehicle
    /// (`inf_ecs::vehicle`) reached through a seat warp, and `Flying` is 6-DOF
    /// with banking. Every catalogue mode has its mechanics now, so what is left
    /// here is exactly the reserved slots: a mode a NEWER build wrote into a file
    /// this one is reading, which must refuse by name rather than be entered.
    pub fn is_deferred(self) -> bool {
        self.reserved_slot().is_some()
    }

    /// Whether the character's feet are on something — the modes the ground
    /// integrator runs for.
    pub fn is_grounded_family(self) -> bool {
        matches!(
            self,
            MovementMode::Grounded
                | MovementMode::Crouch
                | MovementMode::Prone
                | MovementMode::Slide
                | MovementMode::Roll
        )
    }

    /// Whether the character is airborne: the two fall modes **and** `Dive`,
    /// which is a ballistic leap and integrates under the same air branch.
    pub fn is_falling(self) -> bool {
        matches!(
            self,
            MovementMode::FallFree | MovementMode::FallControlled | MovementMode::Dive
        )
    }

    /// Whether this is one of the two swim modes.
    pub fn is_swimming(self) -> bool {
        matches!(self, MovementMode::SwimSurface | MovementMode::SwimUnder)
    }
}

/// The discrete gait tier inside [`MovementMode::Grounded`] (Ruling 4: "an
/// analog speed plus a discrete tier" — the analog half is
/// [`MovementRuntime::mapped_speed`]).
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum Gait {
    /// The slowest tier — ALS's `WalkAction` toggle.
    Walk,
    /// The default.
    #[default]
    Run,
    /// Gated: see [`CharacterMovement::sprint_input_min`].
    Sprint,
    /// Reserved. A build that meets one refuses by name.
    Reserved3,
    /// Reserved. A build that meets one refuses by name.
    Reserved4,
}

impl Gait {
    /// `Some(index)` if this is a reserved slot a newer build wrote.
    pub fn reserved_slot(self) -> Option<u8> {
        match self {
            Gait::Reserved3 => Some(3),
            Gait::Reserved4 => Some(4),
            _ => None,
        }
    }
}

/// **How the body's yaw relates to where it is looking** (Ruling 4: its own
/// frozen wire enum). Selects the movement-settings block *and* the rotation
/// target, exactly as ALS's two-level dispatch does.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum RotationMode {
    /// The body faces where it is going. The third-person default.
    #[default]
    VelocityDirection,
    /// The body faces where the camera looks, and strafes.
    LookingDirection,
    /// As `LookingDirection`, but slower and with sprint refused.
    Aiming,
    /// Reserved. A build that meets one refuses by name.
    Reserved3,
    /// Reserved. A build that meets one refuses by name.
    Reserved4,
}

impl RotationMode {
    /// `Some(index)` if this is a reserved slot a newer build wrote.
    pub fn reserved_slot(self) -> Option<u8> {
        match self {
            RotationMode::Reserved3 => Some(3),
            RotationMode::Reserved4 => Some(4),
            _ => None,
        }
    }
}

/// Which quadrant of the aim frame the character is actually moving in — the
/// input a four-way locomotion blend space reads (P29.4).
///
/// Derived with hysteresis so the boundary cannot chatter; see
/// [`crate::movement::quadrant`].
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum MovementDirection {
    /// Within the forward band.
    #[default]
    Forward,
    /// To the character's right.
    Right,
    /// To the character's left.
    Left,
    /// Behind.
    Backward,
}

/// What the landing classifier decided, keyed to **impact speed** rather than to
/// whichever animation happened to be playing (§13's catalogue, verbatim).
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum LandingKind {
    /// Nothing has landed yet this session.
    #[default]
    None,
    /// Under [`CharacterMovement::land_hard_mps`]: keep running.
    Soft,
    /// Between the two thresholds with no movement input: plant, with the
    /// heavier braking friction for
    /// [`land_friction_time_s`](CharacterMovement::land_friction_time_s).
    Hard,
    /// Between the two thresholds *with* movement input: break-fall into a roll.
    Roll,
    /// Past [`CharacterMovement::land_ragdoll_mps`]. ALS ragdolls here; P29.4
    /// owns the ragdoll, so this lands as `Hard` and records the verdict.
    Ragdoll,
}

/// Why a requested mode change did not happen. **A refusal is a value** (the P21
/// law): the movement step never fails, it answers.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub enum MovementRefusal {
    /// Nothing was refused.
    #[default]
    None,
    /// The taller capsule does not fit here — the overhead sweep hit something.
    NoOverheadClearance,
    /// The mode exists on the frozen wire enum and its mechanics belong to a
    /// later sub-phase (P29.4's mantle and ragdoll, P29.7's driving and flight),
    /// or it is a reserved slot a newer build wrote.
    ModeNotYetImplemented,
    /// The transition is not in the mode table (e.g. sliding while swimming).
    IllegalTransition,
    /// The entry condition was not met — too slow to slide, not grounded to
    /// jump, not in water to swim.
    ConditionNotMet,
}

/// A piecewise-linear curve sampled on ALS's **normalized speed**: `0` stopped,
/// `1` walk, `2` run, `3` sprint.
///
/// # Why four numbers and not a keyframe list
///
/// This is the single idea most worth taking from ALS (`GetMappedSpeed`,
/// `ALSCharacterMovementComponent.cpp:156`). Every movement curve is keyed on
/// the *normalized* speed rather than on metres per second, so retuning a
/// character's walk from 1.5 to 1.8 m/s does not invalidate its acceleration,
/// braking, friction or turn-rate curves. The anchors of that axis are exactly
/// the three gait speeds plus zero, so a curve with a value at each anchor and a
/// straight line between them is not an approximation of ALS's curve asset — it
/// is the shape those assets are authored in.
///
/// Four `f64`s also make this `Copy`, reflectable as four scalars in the Details
/// grid, and free of any interpolation that could reach a transcendental (the
/// portable-math law). Sampling clamps outside `[0, 3]`.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct SpeedCurve {
    /// Value at normalized speed 0 (stopped).
    pub at_stop: f64,
    /// Value at normalized speed 1 (walk speed).
    pub at_walk: f64,
    /// Value at normalized speed 2 (run speed).
    pub at_run: f64,
    /// Value at normalized speed 3 (sprint speed).
    pub at_sprint: f64,
}

impl SpeedCurve {
    /// A flat curve — the same value at every speed.
    pub const fn flat(v: f64) -> Self {
        Self {
            at_stop: v,
            at_walk: v,
            at_run: v,
            at_sprint: v,
        }
    }

    /// A curve from its four anchors.
    pub const fn new(at_stop: f64, at_walk: f64, at_run: f64, at_sprint: f64) -> Self {
        Self {
            at_stop,
            at_walk,
            at_run,
            at_sprint,
        }
    }

    /// Sample at normalized speed `s`, clamped to `[0, 3]`. Pure adds and
    /// multiplies — no transcendental, so it is portable by construction.
    ///
    /// A non-finite `s` reads as `0`: every ordering comparison a NaN takes part
    /// in is false, so an unguarded clamp would let it through and put a NaN
    /// into the velocity integrator, where it becomes a character at no position
    /// at all (the P29.2 blend-space finding, one crate over).
    pub fn sample(&self, s: f64) -> f64 {
        if !s.is_finite() {
            return self.at_stop;
        }
        let s = s.clamp(0.0, 3.0);
        let (a, b, t) = if s <= 1.0 {
            (self.at_stop, self.at_walk, s)
        } else if s <= 2.0 {
            (self.at_walk, self.at_run, s - 1.0)
        } else {
            (self.at_run, self.at_sprint, s - 2.0)
        };
        a + (b - a) * t
    }
}

impl Default for SpeedCurve {
    fn default() -> Self {
        Self::flat(0.0)
    }
}

/// The live movement state — never serialized, never reflected.
///
/// The same shape as [`SmRuntimeState`]'s role on `AnimStateMachine`: the
/// authored tunables persist, the per-step derivations do not. Every field here
/// is either recomputed each fixed step or is a latch the step owns, and all of
/// them are read by P29.4's animation bridge, which is why they are `pub` rather
/// than private to the integrator.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct MovementRuntime {
    /// Whether the fixed step has taken this character's **authored facing**
    /// off its `Transform` yet (P29.3 audit, A1).
    ///
    /// The step writes `body_yaw_deg` back onto the entity's rotation every
    /// step. Every other field here starts at zero and is recomputed before it
    /// is read; the body yaw is not, because there is nothing to recompute it
    /// *from* — a character standing still has no velocity to face. So without
    /// this latch the first step wrote a zero over whatever the level author
    /// had placed, and a squad of NPCs faced north on their first frame. The
    /// same shape as the gait and rotation-mode defect this wave found: an
    /// authored value is not the controller's to take.
    pub seeded: bool,

    // ── intent, written by `crate::movement::apply_intent` ──
    /// Desired planar motion in the **aim frame**: `x` right, `y` forward, each
    /// in `[-1, 1]`.
    pub intent_move: Vec2d,
    /// Requested yaw rate, **degrees per second** (a mouse delta already divided
    /// by the frame's own dt — see `inf_input::InputState::axis_snapshot`).
    pub intent_look_yaw_dps: f64,
    /// Requested pitch rate, degrees per second. Consumed by P29.6's camera; the
    /// movement step only stores it.
    pub intent_look_pitch_dps: f64,
    /// Requested vertical motion while swimming or flying, `[-1, 1]`.
    pub intent_vertical: f64,
    /// Held intent flags: sprint, walk, aim.
    pub want_sprint: bool,
    /// See [`want_sprint`](Self::want_sprint).
    pub want_walk: bool,
    /// See [`want_sprint`](Self::want_sprint).
    pub want_aim: bool,
    /// Edge intents, consumed by the mode table on the step they arrive.
    pub press_jump: bool,
    /// See [`press_jump`](Self::press_jump).
    pub press_crouch: bool,
    /// See [`press_jump`](Self::press_jump).
    pub press_prone: bool,
    /// See [`press_jump`](Self::press_jump).
    pub press_roll: bool,
    /// See [`press_jump`](Self::press_jump).
    pub press_dive: bool,

    // ── integrated state ──
    /// World velocity, m/s. **Owned by the movement step** — this is the whole
    /// of impedance mismatch IM-2: rapier's kinematic controller has no velocity
    /// model at all, so the engine keeps one and uses the controller purely as
    /// sweep-and-slide.
    pub velocity: Vec3d,
    /// Smoothed aim yaw, degrees. ALS's `AimingRotation.Yaw`.
    pub aim_yaw_deg: f64,
    /// Aim pitch, degrees, clamped to `[-89, 89]`. Stored for P29.6.
    pub aim_pitch_deg: f64,
    /// `|Δ aim yaw| / dt`, degrees per second — ALS's `AimYawRate`, read by the
    /// grounded rotation-rate multiplier here and by three P29.4 systems.
    pub aim_yaw_rate_dps: f64,
    /// The body's own facing, degrees. Written back onto the entity's
    /// `Transform` every step; kept here as well because the smoother below
    /// needs last step's value and a `Transform` may be moved by anything.
    pub body_yaw_deg: f64,
    /// The **intermediate** goal of ALS's two-stage rotation smoother: a
    /// constant-rate chase of the real goal, which the body then chases
    /// exponentially. Bounding peak angular velocity in the first stage is what
    /// stops a 179-degree input flip from snapping the character round.
    pub target_yaw_deg: f64,
    /// Whether the last sweep ended on the ground.
    pub grounded: bool,
    /// The surface normal under the character, from the ground probe; `+Y` when
    /// there is nothing under it.
    pub ground_normal: Vec3d,
    /// Seconds spent in the current [`MovementMode`].
    pub time_in_mode_s: f64,
    /// Seconds since the last landing. Meaningless until
    /// [`landing`](Self::landing) leaves [`LandingKind::None`], which is the
    /// latch [`landing_friction_scale`](crate::movement::landing_friction_scale)
    /// reads rather than trusting this to start large (P29.3 audit, A5).
    pub time_since_land_s: f64,
    /// Downward speed at the last landing, m/s — the classifier's input.
    pub land_impact_mps: f64,
    /// What the classifier decided about the last landing.
    pub landing: LandingKind,
    /// The most recent refusal, cleared when a transition succeeds.
    pub refusal: MovementRefusal,
    /// How many transitions have been refused since the entity was created.
    /// A counter rather than a log: it is the thing a gate can assert on.
    pub refusals: u32,

    // ── derived outputs, for P29.4's animation bridge ──
    /// Horizontal speed mapped onto `[0, 3]` — the X axis of every
    /// [`SpeedCurve`].
    pub mapped_speed: f64,
    /// The gait the *body* is actually in, which lags the requested one during
    /// deceleration (ALS's `GetActualGait` and its hysteresis band).
    pub actual_gait: Gait,
    /// Which quadrant of the aim frame the motion is in, with hysteresis.
    pub direction: MovementDirection,
    /// The `W_Gait`-style scalar: `0` at a walk, `1` at a run, `2` at a sprint,
    /// continuous between.
    pub gait_scalar: f64,
    /// Stride blend `[0, 1]` — how far the locomotion clip's stride is scaled.
    pub stride_blend: f64,
    /// Walk↔run blend `[0, 1]`.
    pub walk_run_blend: f64,
    /// Acceleration relative to the character's own facing, normalized against
    /// the *current* curve-derived accel/braking maxima, `[-1, 1]` per axis.
    pub relative_accel: Vec2d,
    /// Lean inputs `[-1, 1]`: `x` left/right, `y` forward/back.
    pub lean: Vec2d,

    // ── traversal (P29.4) ──
    /// The mantle in progress, if one is.
    pub mantle: MantleState,
    /// How close a predicted landing is, `[0, 1]`; `0` when none is predicted.
    /// The value an in-air animation blends a landing pose on.
    pub land_alpha: f64,
    /// The speed the land-prediction sweep says the character will arrive at,
    /// m/s — which is what lets the classifier run **before** the touch.
    pub land_predicted_mps: f64,
    /// What the classifier would say about that predicted landing.
    pub predicted_landing: LandingKind,

    // ── turn / rotate in place (P29.4) ──
    /// How long the turn-in-place conditions have held, seconds. Reset the
    /// moment either of them stops.
    pub turn_delay_s: f64,
    /// The yaw a turn-in-place is turning toward, degrees.
    pub turn_target_yaw_deg: f64,
    /// Whether a turn-in-place is running.
    pub turning_in_place: bool,
    /// Rotate-in-place gates (ALS's `bRotateL`/`bRotateR`): the aim has left the
    /// ±50° band while the character is standing still and aiming.
    pub rotate_left: bool,
    /// See [`rotate_left`](Self::rotate_left).
    pub rotate_right: bool,
    /// The play-rate scale a rotate-in-place animation runs at, from the aim yaw
    /// rate (ALS 1.15…3.0).
    pub rotate_rate: f64,

    // ── aim offsets (P29.4) ──
    /// The vertical aim-offset sweep parameter, `[0, 1]`, `0` looking up.
    pub aim_sweep: f64,
    /// Per-spine-joint yaw for the aim offset, degrees — the aim/body yaw delta
    /// divided across the spine chain.
    pub spine_yaw_deg: f64,
    /// How much aim offset to apply, `[0, 1]` — `1 - Mask_AimOffset`.
    pub aim_offset_weight: f64,

    // ── foot IK / foot lock (P29.4) ──
    /// The left foot's world-space lock.
    pub foot_lock_l: inf_anim::FootLock,
    /// The right foot's.
    pub foot_lock_r: inf_anim::FootLock,
    /// **The gate's number**: how far the locked left foot slid last step,
    /// metres, on the ground plane. `0` when nothing is locked.
    pub foot_slide_l_m: f64,
    /// The right foot's, same units.
    pub foot_slide_r_m: f64,
    /// Where the left foot is **drawn** this step — the pose's position with the
    /// lock applied. Published so a gate can measure a planted foot against the
    /// world rather than against the lock's own opinion of itself: an unlocked
    /// foot reports a slide of zero (it makes no claim), so the number that
    /// distinguishes a plant from a skate has to be the position.
    pub foot_world_l: Vec3d,
    /// The right foot's.
    pub foot_world_r: Vec3d,
    /// **The pelvis drop the two foot offsets imply**, metres — computed and
    /// deliberately **not applied** (P29.4 audit, A9).
    ///
    /// [`inf_anim::pelvis_offset`] answers how far the hips must come down for
    /// the lower foot to reach its ground without the leg straightening past its
    /// limit. Routing it into the rig is a *pose* edit that P29.5's authoring
    /// pass owns; applying it to the capsule would move the character, which is
    /// not what it means. It is recorded here rather than dropped on the floor
    /// because a number a step computes and nobody can read is a number no test
    /// can see — and the whole point of this block is that the animation bridge
    /// reads what the movement step derives.
    pub pelvis_offset: Vec3d,

    // ── ragdoll (P29.4) ──
    /// The ragdoll bridge's state for this character.
    pub ragdoll: RagdollRuntime,

    // ── vehicles and flight (P29.7) ──
    /// Edge: the enter/exit control. Enters the nearest vehicle from the ground
    /// and leaves the one being driven.
    pub press_interact: bool,
    /// **Edge: the lock control** (island wave I8b) — throw or draw the bolt on
    /// the door in reach.
    ///
    /// A runtime field like every other edge here, so it moves no schema.
    pub press_lock: bool,
    /// Edge: the flight toggle.
    pub press_fly: bool,
    /// Held: the handbrake, routed to the vehicle being driven.
    pub want_handbrake: bool,

    // ── weapons and the verbs that share their buttons (I6) ──
    /// **Held**: the attack control. An automatic weapon reads the level, which
    /// is why this is not only an edge.
    pub want_attack: bool,
    /// **Edge**: the attack control went down. What arms a door kick — a kick is
    /// a press, never a hold — and what a semi-automatic weapon's own
    /// `trigger_held` rule is checked against.
    ///
    /// Two fields for one button on purpose: an automatic weapon needs the level
    /// and a kick needs the edge, and deriving one from the other at the consumer
    /// is how a press made in one mode fires in another (the P29.7 A1 class).
    pub press_attack: bool,
    /// **Edge**: reload.
    pub press_reload: bool,
    /// **The wheel's SIGN this step**, `-1`, `0` or `+1`.
    ///
    /// A sign and not a count, because `weapon_switch` reaches the engine as a
    /// **rate** — the wheel is a delta source and `axis_snapshot` divides it by
    /// the frame time (the I5 remainder) — so a notch count is a number this
    /// engine does not have. One slot per step in whichever direction the wheel
    /// turned is what a player reads as one notch.
    pub weapon_switch: i32,
    /// The seat this character is in (or climbing into).
    pub seat: SeatState,
    /// **Bank angle while flying**, degrees — roll into the turn, derived from
    /// the yaw rate and bounded. Written onto the entity's `Transform` roll by
    /// the flight step, and zero in every other mode.
    pub bank_deg: f64,
}

/// A **seat**: which vehicle a character is in, and how far through the
/// enter/exit choreography it is (P29.7).
///
/// Plain `Copy` scalars for the reason [`MantleState`] gives — this is inlined
/// into a component — and on the **runtime** rather than on the wire, which is
/// what lets `MovementMode::Driving` mean something without a schema move: the
/// mode is serialized (it always was), the seat is derived, and a level saved
/// mid-drive loads with a character standing where the car was rather than with
/// a dangling reference to it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SeatState {
    /// The chassis entity being driven, or `Uuid::nil()` for none.
    pub vehicle: Uuid,
    /// Whether the enter warp is still running.
    pub entering: bool,
    /// Seconds into the choreography.
    pub time_s: f64,
    /// Where the character's transform was when the warp began, world metres —
    /// the warp's `start`, exactly as [`MantleState::start`] is the mantle's.
    pub start: Vec3d,
    /// Its facing then, degrees.
    pub start_yaw_deg: f64,
}

impl SeatState {
    /// Whether this character is in (or entering) a vehicle.
    pub fn is_seated(&self) -> bool {
        !self.vehicle.is_nil()
    }
}

/// A **mantle in progress** (P29.4) — where it started, where it must end, and
/// how far through it is.
///
/// Plain `Copy` scalars, because [`MovementRuntime`] is inlined into an ECS
/// component and must not allocate. The *placement* is
/// [`inf_anim::warp::warp_offset`]; this is only the endpoints and the clock.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MantleState {
    /// Whether a mantle is running.
    pub active: bool,
    /// The character's feet when it started, world metres.
    pub start: Vec3d,
    /// Its facing then, degrees.
    pub start_yaw_deg: f64,
    /// Where the feet must end up, world metres.
    pub target: Vec3d,
    /// The facing to arrive at, degrees.
    pub target_yaw_deg: f64,
    /// Seconds elapsed.
    pub elapsed_s: f64,
    /// How long the whole mantle takes, seconds.
    pub duration_s: f64,
    /// How far up the ledge was, metres.
    pub height_m: f64,
    /// Whether it is a high mantle (past
    /// [`inf_anim::MANTLE_HIGH_SPLIT_M`]).
    pub high: bool,
    /// The clip time the traversal animation should be entered at, seconds —
    /// ALS's `StartingPosition` from the height remap.
    pub clip_start_s: f64,
    /// The rate that animation should play at — ALS's `PlayRate`.
    pub play_rate: f64,
}

impl MantleState {
    /// How far through the mantle it is, `[0, 1]`.
    pub fn alpha(&self) -> f64 {
        if self.duration_s.is_nan() || self.duration_s <= 0.0 {
            return 1.0;
        }
        (self.elapsed_s / self.duration_s).clamp(0.0, 1.0)
    }
}

/// The **ragdoll bridge's** per-character state (P29.4).
///
/// The phase and its clock are what [`inf_anim::ragdoll::blend_weight`] reads, so
/// the blend really is a pure function of sim state; the rest is the handoff in
/// each direction.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RagdollRuntime {
    /// What the bridge is doing with this character.
    pub phase: inf_anim::RagdollPhase,
    /// Seconds in that phase — the other half of the blend weight's input.
    pub time_in_phase_s: f64,
    /// The velocity the articulated bodies were last seen at, m/s. Handed back
    /// to the movement integrator when a ragdoll ends in the air.
    pub last_velocity: Vec3d,
    /// Whether the pelvis says the character is on its back.
    pub face_up: bool,
    /// Whether the ragdoll's own ground probe found a floor under the pelvis.
    pub on_ground: bool,
    /// Whether the articulated bodies exist in the physics world right now.
    pub spawned: bool,
    /// The pelvis's world position last step, m — what the capsule follows.
    pub pelvis: Vec3d,
    /// The pelvis's yaw, degrees.
    pub pelvis_yaw_deg: f64,
    /// The angular-drive stiffness the motor is being driven at, from the
    /// ragdoll's own speed.
    pub motor_stiffness: f64,
    /// Whether the bodies have come to rest — the condition the get-up waits
    /// for. On the runtime rather than only inside the physics bridge so a gate
    /// can assert on it without reaching across the seam.
    pub settled_hint: bool,
}

/// **The movement component** (P29.3) — the full tunable set §13's catalogue
/// amendment asks for, the ALS curve-driven settings model, and the mode the
/// fixed step integrates.
///
/// # What this replaces
///
/// [`CharacterController3D`] is three fields (`max_slope_deg`, `snap_to_ground`,
/// `offset`) and stays exactly that: it describes the *mover*, and it is what
/// `physics3d.move_and_slide` has always read. This describes the *character* —
/// how fast it walks, how hard it accelerates, how much authority it has in the
/// air, how tall it is when it crouches — and it is what the P29.3 fixed step
/// integrates. Before it, an entity's movement tunables were those three
/// numbers and every scrap of motion had to be computed by a Blueprint.
///
/// It is a **new component** rather than fields appended to
/// `CharacterController3D`, and the reason is the bincode-positional law rather
/// than taste: `CharacterController3D` appears inside eighteen frozen historical
/// entity records, in two codec mirrors. Growing it means freezing a
/// `CharacterController3DV22` copy into every one of those thirty-six places,
/// because bincode reads a struct's fields positionally and `#[serde(default)]`
/// buys nothing. Appending a whole component to the live record's tail is the
/// cheap rung of the same ladder and leaves every historical record byte-for-byte
/// untouched.
///
/// # Units, converted exactly once
///
/// SI throughout: metres, seconds, m/s, m/s², degrees only for angles. ALS's
/// constants arrive in centimetres and are converted **here, at the port
/// boundary**, in the defaults below — never at runtime. Each converted default
/// names its ALS source so the conversion can be checked rather than trusted.
///
/// # The settings model
///
/// ALS selects one of six `FALSMovementSettings` blocks on
/// `RotationMode × Stance` and then picks a *speed* inside it by gait. The shape
/// is kept ([`crate::movement::settings_for`]) and the storage is flattened: the
/// stance axis is the per-mode speeds (`crouch_speed_mps`, `prone_speed_mps`),
/// the rotation-mode axis is a pair of scales, and the four curves are shared.
/// That is a real reduction from ALS's 6 × (3 speeds + 2 curve assets) and it is
/// stated rather than hidden: the shipped ALS data table points all six blocks
/// at the same curve pair, so what is lost is the ability to give aiming a
/// *different acceleration profile* — recoverable later as two more curve
/// fields at this struct's tail, which is the cheap direction.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct CharacterMovement {
    // ── state (authored initial value; the fixed step writes it back) ────────
    /// What the character is doing. Authored as the spawn mode; the fixed step
    /// owns it afterwards.
    #[serde(default)]
    pub mode: MovementMode,
    /// The **requested** gait tier. The gait the body is in is
    /// [`MovementRuntime::actual_gait`], which lags this during deceleration.
    #[serde(default)]
    pub gait: Gait,
    /// How the body's yaw relates to the aim direction.
    #[serde(default)]
    pub rotation_mode: RotationMode,
    /// The **overlay id** — an open, string-keyed name (`""` = default,
    /// `"rifle"`, `"torch"`, `"injured"`, …), interned at runtime by
    /// [`crate::movement::OverlayRegistry`].
    ///
    /// Ruling 4 is explicit that this is **not** a wire enum: ALS ships thirteen
    /// overlay states and a studio must be able to add a fourteenth without an
    /// engine schema bump. A `String` on the wire and a `u32` in the hot loop is
    /// what "open interned id" means.
    #[serde(default)]
    pub overlay: String,
    /// Whether this character reads the local player's intent
    /// ([`crate::movement::apply_intent`]). An AI or a Blueprint writes the
    /// runtime intent directly instead.
    #[serde(default)]
    pub player_controlled: bool,

    // ── speeds, m/s ─────────────────────────────────────────────────────────
    /// ALS `WalkSpeed` 165 cm/s.
    #[serde(default = "default_walk_speed")]
    pub walk_speed_mps: f64,
    /// ALS `RunSpeed` 375 cm/s.
    #[serde(default = "default_run_speed")]
    pub run_speed_mps: f64,
    /// ALS `SprintSpeed` 650 cm/s.
    #[serde(default = "default_sprint_speed")]
    pub sprint_speed_mps: f64,
    /// Crouched locomotion speed.
    #[serde(default = "default_crouch_speed")]
    pub crouch_speed_mps: f64,
    /// Prone crawl speed.
    #[serde(default = "default_prone_speed")]
    pub prone_speed_mps: f64,
    /// Surface swim speed. Capped again by the P20 swim transform, which owns
    /// the water's opinion.
    #[serde(default = "default_swim_surface_speed")]
    pub swim_surface_speed_mps: f64,
    /// Submerged swim speed.
    #[serde(default = "default_swim_under_speed")]
    pub swim_under_speed_mps: f64,
    /// Speed scale applied in [`RotationMode::LookingDirection`].
    #[serde(default = "default_move_speed_scale")]
    pub looking_speed_scale: f64,
    /// Speed scale applied in [`RotationMode::Aiming`] — ALS aims slower.
    #[serde(default = "default_aiming_speed_scale")]
    pub aiming_speed_scale: f64,

    // ── the four curves, keyed on normalized speed ───────────────────────────
    /// Max acceleration, m/s² (ALS `MovementCurve.X`).
    #[serde(default = "default_accel_curve")]
    pub acceleration: SpeedCurve,
    /// Max braking deceleration, m/s² (ALS `MovementCurve.Y`).
    #[serde(default = "default_braking_curve")]
    pub braking: SpeedCurve,
    /// Ground friction, 1/s (ALS `MovementCurve.Z`).
    #[serde(default = "default_friction_curve")]
    pub ground_friction: SpeedCurve,
    /// Body turn rate, degrees per second (ALS `RotationRateCurve`).
    #[serde(default = "default_rotation_rate_curve")]
    pub rotation_rate: SpeedCurve,

    // ── air ─────────────────────────────────────────────────────────────────
    /// Fraction of ground acceleration available in [`MovementMode::FallFree`].
    #[serde(default = "default_air_control")]
    pub air_control: f64,
    /// Fraction available in [`MovementMode::FallControlled`] — the "controlled
    /// fall" half of the catalogue row.
    #[serde(default = "default_air_control_reduced")]
    pub air_control_reduced: f64,
    /// Hard ceiling on air acceleration, m/s², whatever the curve says.
    #[serde(default = "default_air_accel_max")]
    pub air_accel_max_mps2: f64,
    /// Terminal velocity, m/s (positive magnitude).
    #[serde(default = "default_terminal_velocity")]
    pub terminal_velocity_mps: f64,
    /// Gravity acting on this character, m/s² (positive magnitude, down). Kept
    /// per-character rather than read from the physics world because the mover
    /// is kinematic: the solver never applies gravity to it, so nothing else
    /// would.
    #[serde(default = "default_char_gravity")]
    pub gravity_mps2: f64,
    /// Take-off speed of a jump, m/s.
    #[serde(default = "default_jump_speed")]
    pub jump_speed_mps: f64,

    // ── capsule ─────────────────────────────────────────────────────────────
    /// Capsule half-height (the segment half-length, **excluding** the radius)
    /// while standing.
    #[serde(default = "default_stand_half_height")]
    pub stand_half_height_m: f64,
    /// Capsule half-height while crouched, sliding or rolling.
    #[serde(default = "default_crouch_half_height")]
    pub crouch_half_height_m: f64,
    /// Capsule half-height while prone.
    #[serde(default = "default_prone_half_height")]
    pub prone_half_height_m: f64,

    // ── mover ───────────────────────────────────────────────────────────────
    /// **Autostep**: the tallest obstacle the character steps over, in metres.
    /// Zero disables it.
    ///
    /// This field is the reason stairs work. rapier's autostep setter existed
    /// from P9.1 and was **never called from production code**, so the default
    /// (`None`) applied and a character walked into a step instead of up it.
    #[serde(default = "default_step_height")]
    pub step_height_m: f64,
    /// Free space that must exist beyond a step for it to be taken, in metres.
    #[serde(default = "default_step_min_width")]
    pub step_min_width_m: f64,
    /// The steepest slope the character can walk up, degrees.
    #[serde(default = "default_slope_limit")]
    pub slope_limit_deg: f64,
    /// The slope at which the character starts sliding back down, degrees.
    /// rapier's `min_slope_slide_angle`, likewise never called before P29.3.
    #[serde(default = "default_slide_slope")]
    pub slide_slope_deg: f64,

    // ── sprint gate (ALS `CanSprint`) ───────────────────────────────────────
    /// Minimum input magnitude for a sprint, `[0, 1]` (ALS 0.9).
    #[serde(default = "default_sprint_input_min")]
    pub sprint_input_min: f64,
    /// In [`RotationMode::LookingDirection`], how far off the aim direction the
    /// input may point and still sprint, degrees (ALS 50).
    #[serde(default = "default_sprint_angle")]
    pub sprint_angle_deg: f64,

    // ── slide ───────────────────────────────────────────────────────────────
    /// Minimum speed to enter a slide, m/s.
    #[serde(default = "default_slide_entry_speed")]
    pub slide_entry_speed_mps: f64,
    /// Speed at which a slide ends, m/s.
    #[serde(default = "default_slide_exit_speed")]
    pub slide_exit_speed_mps: f64,
    /// Slide friction on level ground, 1/s.
    #[serde(default = "default_slide_friction_flat")]
    pub slide_friction_flat: f64,
    /// Slide friction at [`slope_limit_deg`](Self::slope_limit_deg), 1/s.
    /// Interpolated against the measured slope, which is the "friction-versus-
    /// slope curve" the catalogue's slide row names: downhill keeps you sliding,
    /// flat ground stops you.
    #[serde(default = "default_slide_friction_slope")]
    pub slide_friction_slope: f64,

    // ── roll / dive ─────────────────────────────────────────────────────────
    /// Forward speed a roll carries, m/s.
    #[serde(default = "default_roll_speed")]
    pub roll_speed_mps: f64,
    /// Seconds a roll lasts before returning to crouch.
    #[serde(default = "default_roll_time")]
    pub roll_time_s: f64,
    /// Forward speed a dive launches with, m/s.
    #[serde(default = "default_dive_speed")]
    pub dive_speed_mps: f64,
    /// Upward speed a dive launches with, m/s.
    #[serde(default = "default_dive_up_speed")]
    pub dive_up_speed_mps: f64,

    // ── landing (ALS `EventOnLanded`) ───────────────────────────────────────
    /// Impact speed past which a landing is hard, m/s (ALS 700 cm/s).
    #[serde(default = "default_land_hard")]
    pub land_hard_mps: f64,
    /// Impact speed past which a landing would ragdoll, m/s (ALS 1000 cm/s).
    #[serde(default = "default_land_ragdoll")]
    pub land_ragdoll_mps: f64,
    /// Braking-friction multiplier for [`land_friction_time_s`](Self::land_friction_time_s)
    /// after a hard landing **with** movement input — ALS 0.5, i.e. a slide.
    #[serde(default = "default_brake_friction_input")]
    pub brake_friction_input: f64,
    /// The same **without** input — ALS 3.0, i.e. a plant.
    #[serde(default = "default_brake_friction_idle")]
    pub brake_friction_idle: f64,
    /// How long the landing friction override lasts, seconds (ALS 0.5).
    #[serde(default = "default_land_friction_time")]
    pub land_friction_time_s: f64,

    /// Live state — transient, never serialized or reflected. See
    /// [`MovementRuntime`].
    #[serde(skip)]
    #[reflect(ignore)]
    pub runtime: MovementRuntime,
}

// ── defaults, with the ALS constant each was converted from ─────────────────
//
// IM-1, discharged in one place: every ALS distance is centimetres and every
// speed centimetres per second. The conversion happens HERE and nowhere else, so
// there is no cm value anywhere in the runtime to be divided twice.

fn default_walk_speed() -> f64 {
    1.65 // ALS WalkSpeed 165 cm/s
}
fn default_run_speed() -> f64 {
    3.75 // ALS RunSpeed 375 cm/s
}
fn default_sprint_speed() -> f64 {
    6.5 // ALS SprintSpeed 650 cm/s
}
fn default_crouch_speed() -> f64 {
    1.5 // ALS AnimatedCrouchSpeed 150 cm/s
}
fn default_prone_speed() -> f64 {
    0.7 // no ALS precedent: prone is one of the seven catalogue rows ALS lacks
}
fn default_swim_surface_speed() -> f64 {
    2.0
}
fn default_swim_under_speed() -> f64 {
    1.6
}
fn default_move_speed_scale() -> f64 {
    1.0
}
fn default_aiming_speed_scale() -> f64 {
    0.65
}
fn default_accel_curve() -> SpeedCurve {
    // ALS NormalMovement.X: brisk from a stop, softer at speed.
    SpeedCurve::new(8.0, 8.0, 6.0, 4.0)
}
fn default_braking_curve() -> SpeedCurve {
    SpeedCurve::new(20.0, 20.0, 8.0, 5.0)
}
fn default_friction_curve() -> SpeedCurve {
    SpeedCurve::new(6.0, 6.0, 3.0, 1.0)
}
fn default_rotation_rate_curve() -> SpeedCurve {
    SpeedCurve::new(360.0, 360.0, 500.0, 700.0)
}
fn default_air_control() -> f64 {
    0.35
}
fn default_air_control_reduced() -> f64 {
    0.12
}
fn default_air_accel_max() -> f64 {
    12.0
}
fn default_terminal_velocity() -> f64 {
    53.0 // the human terminal velocity, belly-to-earth
}
fn default_char_gravity() -> f64 {
    9.81
}
fn default_jump_speed() -> f64 {
    4.5 // ALS JumpZVelocity 450 cm/s
}
fn default_stand_half_height() -> f64 {
    0.6 // a 1.8 m capsule with a 0.3 m radius
}
fn default_crouch_half_height() -> f64 {
    0.3
}
fn default_prone_half_height() -> f64 {
    0.05
}
fn default_step_height() -> f64 {
    0.45 // UE's MaxStepHeight 45 cm, which is what ALS gets stairs from
}
fn default_step_min_width() -> f64 {
    0.15
}
fn default_slope_limit() -> f64 {
    45.0
}
fn default_slide_slope() -> f64 {
    50.0
}
fn default_sprint_input_min() -> f64 {
    0.9 // ALS CanSprint
}
fn default_sprint_angle() -> f64 {
    50.0 // ALS CanSprint, LookingDirection branch
}
fn default_slide_entry_speed() -> f64 {
    4.0
}
fn default_slide_exit_speed() -> f64 {
    1.5
}
fn default_slide_friction_flat() -> f64 {
    3.5
}
fn default_slide_friction_slope() -> f64 {
    0.3
}
fn default_roll_speed() -> f64 {
    4.0
}
fn default_roll_time() -> f64 {
    0.75
}
fn default_dive_speed() -> f64 {
    5.5
}
fn default_dive_up_speed() -> f64 {
    2.5
}
fn default_land_hard() -> f64 {
    7.0 // ALS BreakfallOnLandVelocity 700 cm/s
}
fn default_land_ragdoll() -> f64 {
    10.0 // ALS RagdollOnLandVelocity 1000 cm/s
}
fn default_brake_friction_input() -> f64 {
    0.5 // ALS EventOnLanded, with input
}
fn default_brake_friction_idle() -> f64 {
    3.0 // ALS EventOnLanded, without input
}
fn default_land_friction_time() -> f64 {
    0.5 // ALS OnLandFrictionReset timer
}

impl Default for CharacterMovement {
    fn default() -> Self {
        Self {
            mode: MovementMode::default(),
            gait: Gait::default(),
            rotation_mode: RotationMode::default(),
            overlay: String::new(),
            player_controlled: false,
            walk_speed_mps: default_walk_speed(),
            run_speed_mps: default_run_speed(),
            sprint_speed_mps: default_sprint_speed(),
            crouch_speed_mps: default_crouch_speed(),
            prone_speed_mps: default_prone_speed(),
            swim_surface_speed_mps: default_swim_surface_speed(),
            swim_under_speed_mps: default_swim_under_speed(),
            looking_speed_scale: default_move_speed_scale(),
            aiming_speed_scale: default_aiming_speed_scale(),
            acceleration: default_accel_curve(),
            braking: default_braking_curve(),
            ground_friction: default_friction_curve(),
            rotation_rate: default_rotation_rate_curve(),
            air_control: default_air_control(),
            air_control_reduced: default_air_control_reduced(),
            air_accel_max_mps2: default_air_accel_max(),
            terminal_velocity_mps: default_terminal_velocity(),
            gravity_mps2: default_char_gravity(),
            jump_speed_mps: default_jump_speed(),
            stand_half_height_m: default_stand_half_height(),
            crouch_half_height_m: default_crouch_half_height(),
            prone_half_height_m: default_prone_half_height(),
            step_height_m: default_step_height(),
            step_min_width_m: default_step_min_width(),
            slope_limit_deg: default_slope_limit(),
            slide_slope_deg: default_slide_slope(),
            sprint_input_min: default_sprint_input_min(),
            sprint_angle_deg: default_sprint_angle(),
            slide_entry_speed_mps: default_slide_entry_speed(),
            slide_exit_speed_mps: default_slide_exit_speed(),
            slide_friction_flat: default_slide_friction_flat(),
            slide_friction_slope: default_slide_friction_slope(),
            roll_speed_mps: default_roll_speed(),
            roll_time_s: default_roll_time(),
            dive_speed_mps: default_dive_speed(),
            dive_up_speed_mps: default_dive_up_speed(),
            land_hard_mps: default_land_hard(),
            land_ragdoll_mps: default_land_ragdoll(),
            brake_friction_input: default_brake_friction_input(),
            brake_friction_idle: default_brake_friction_idle(),
            land_friction_time_s: default_land_friction_time(),
            runtime: MovementRuntime::default(),
        }
    }
}

impl CharacterMovement {
    /// The capsule half-height this mode wants, metres.
    pub fn half_height_for(&self, mode: MovementMode) -> f64 {
        match mode {
            MovementMode::Crouch | MovementMode::Slide | MovementMode::Roll => {
                self.crouch_half_height_m
            }
            MovementMode::Prone | MovementMode::Dive => self.prone_half_height_m,
            _ => self.stand_half_height_m,
        }
    }

    /// The target speed for a mode and gait, m/s, **before** the rotation-mode
    /// scale. ALS's `GetSpeedForGait` with the stance axis folded into the mode.
    pub fn speed_for(&self, mode: MovementMode, gait: Gait) -> f64 {
        match mode {
            MovementMode::Crouch | MovementMode::Slide | MovementMode::Roll => {
                self.crouch_speed_mps
            }
            MovementMode::Prone => self.prone_speed_mps,
            MovementMode::SwimSurface => self.swim_surface_speed_mps,
            MovementMode::SwimUnder => self.swim_under_speed_mps,
            _ => match gait {
                Gait::Walk => self.walk_speed_mps,
                Gait::Sprint => self.sprint_speed_mps,
                // A reserved gait a newer build wrote reads as the safe middle
                // tier rather than as zero, which would freeze the character.
                _ => self.run_speed_mps,
            },
        }
    }

    /// The rotation-mode speed scale (the second half of ALS's two-level
    /// settings dispatch).
    pub fn rotation_speed_scale(&self) -> f64 {
        match self.rotation_mode {
            RotationMode::VelocityDirection => 1.0,
            RotationMode::LookingDirection => self.looking_speed_scale,
            RotationMode::Aiming => self.aiming_speed_scale,
            _ => 1.0,
        }
    }
}

// ── The vehicle class (schema v25, island phase IB-10) ──────────────────────

/// **A vehicle's authored tunables** — the per-vehicle numbers P29.7 could not
/// spend a schema move on.
///
/// P29.7's own remainder, verbatim: *"The vehicle has no per-vehicle authored
/// tuning. A committed rig uses the Ring-0 defaults in both hosts, because a tune
/// is an editor-only door by law and a scene field is a schema move. The course
/// therefore adapts to the car (six tenths of throttle) rather than the car to
/// the course. A vehicle **class** with its own numbers is the island's, and it
/// is what the `Vehicle` trait is shaped for."* This is that field set.
///
/// # It is exactly [`VehicleTuning::names`](crate::vehicle::VehicleTuning::names)
///
/// **Sixty-two** `f64`s since wave VEH2a (fifteen at v25), reaching a running
/// vehicle through [`Vehicle::tune`](crate::vehicle::Vehicle::tune) — the door
/// the live tuner already uses. Two consequences, both deliberate:
///
/// * **An island class gets this for free.** The bridge applies a class by name
///   through the trait, so a motorbike or a tank implementing `Vehicle` reads the
///   authored numbers it recognizes and refuses the ones it does not (a refusal
///   is a value — `VehicleTuning::set` returns `bool`), rather than needing a
///   parallel component per class.
/// * **The component and the door cannot drift.** `names()` is the enumeration;
///   a field added here without a name there fails
///   `the_vehicle_class_is_exactly_the_tuning_door`.
///
/// # The bound v25 stated, and how VEH2a closed it
///
/// v25's own note read: *"`VehicleTuning::enter_window` is **not** here, because
/// it is not nameable through `set`… A class that wants its own seat-warp window
/// needs a trait setter first."* It needed no such thing — it needed the window
/// to stop being a type and start being its two numbers. Wave VEH2a's schema
/// window carries `enter_warp_start` and `enter_warp_end` on the one door, and
/// `VehicleTuning::enter_window()` rebuilds the `WarpWindow` from them. **There
/// is no longer any tunable this component cannot author.**
///
/// # The wire order is APPEND-ONLY, and that is the whole of the bump discipline
///
/// bincode is positional. The fifteen v25 fields keep their v25 order and their
/// v25 offsets; the forty-seven VEH2a fields are a run at the **tail**, sorted
/// among themselves. A field inserted in the middle would decode a torque curve
/// out of a spring rate, which is the third time this repository has paid for
/// that lesson (the P16 `skip_serializing_if` law and the v5 mid-struct insert).
///
/// # Absent means the Ring-0 defaults
///
/// Which is exactly what every pre-v25 level meant, and what the committed P29.7
/// rig still means: the component is opt-in, and a level that never adds one is
/// byte-for-byte the level it was.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
#[serde(default)]
pub struct VehicleClass {
    /// Peak braking force, newtons, summed over all wheels.
    pub brake_force_n: f64,
    /// Damping, newton-seconds per metre, per wheel.
    pub damping_ns_per_m: f64,
    /// Aerodynamic drag, newtons per (m/s)².
    pub drag_n_per_mps2: f64,
    /// How long the enter/exit choreography takes, seconds.
    pub enter_time_s: f64,
    /// Peak handbrake force, newtons, applied to the **rear** wheels only.
    pub handbrake_force_n: f64,
    /// Lateral friction coefficient (µ) at the contact.
    pub lateral_grip: f64,
    /// Longitudinal friction coefficient (µ) for drive and brake force.
    pub longitudinal_grip: f64,
    /// Peak drive force, newtons, summed over the driven wheels.
    pub max_engine_force_n: f64,
    /// The speed at which the drive force reaches zero, m/s.
    pub max_speed_mps: f64,
    /// Steering angle at a standstill, degrees.
    pub max_steer_deg: f64,
    /// Steering angle at `max_speed_mps` and above, degrees.
    pub min_steer_deg: f64,
    /// Suspension length at full extension, metres.
    pub rest_length_m: f64,
    /// Rolling resistance coefficient — a force of `c × load` opposing motion.
    pub rolling_resistance: f64,
    /// Spring rate, newtons per metre, per wheel.
    pub stiffness_n_per_m: f64,
    /// How far the suspension may compress from rest, metres.
    pub travel_m: f64,

    // ── the VEH2a tail (schema v27) — forty-seven, appended, never reordered ──
    /// ABS: the slip ratio above which brake torque is bled off; `0` is off.
    pub abs_slip: f64,
    /// Ackermann, `[0, 1]` — how much more the inside front wheel turns.
    pub ackermann: f64,
    /// Front anti-roll bar rate, newtons per metre of compression difference.
    pub anti_roll_front_n_per_m: f64,
    /// Rear anti-roll bar rate, same units.
    pub anti_roll_rear_n_per_m: f64,
    /// The share of the brake budget the front axle takes, `[0, 1]`.
    pub brake_bias: f64,
    /// Centre-of-gravity height above the chassis origin, metres (negative for a
    /// car whose mass is in its floor).
    pub cog_height_m: f64,
    /// Front differential lock, `[0, 1]` — `0` open, `1` a spool.
    pub diff_lock_front: f64,
    /// Rear differential lock, `[0, 1]`.
    pub diff_lock_rear: f64,
    /// Where downforce acts along the wheelbase, in fractions of half of it.
    pub downforce_centre_z: f64,
    /// Downforce, newtons per (m/s)².
    pub downforce_n_per_mps2: f64,
    /// Sideways aerodynamic drag, newtons per (m/s)².
    pub drag_lateral_n_per_mps2: f64,
    /// Engine braking at the crank with the throttle shut, N·m at the redline.
    pub engine_brake_nm: f64,
    /// Clip time the seat warp closes, seconds.
    pub enter_warp_end: f64,
    /// Clip time the seat warp opens, seconds.
    pub enter_warp_start: f64,
    /// The final drive, multiplying every gear.
    pub final_drive: f64,
    /// The share of drive torque the front axle takes, `[0, 1]` — the drivetrain.
    pub front_torque_split: f64,
    /// First gear's ratio.
    pub gear_1_ratio: f64,
    /// Second gear's ratio.
    pub gear_2_ratio: f64,
    /// Third gear's ratio.
    pub gear_3_ratio: f64,
    /// Fourth gear's ratio.
    pub gear_4_ratio: f64,
    /// Fifth gear's ratio.
    pub gear_5_ratio: f64,
    /// Sixth gear's ratio.
    pub gear_6_ratio: f64,
    /// Seventh gear's ratio.
    pub gear_7_ratio: f64,
    /// Eighth gear's ratio.
    pub gear_8_ratio: f64,
    /// How many forward gears are in use.
    pub gear_count: f64,
    /// Idle speed, rpm.
    pub idle_rpm: f64,
    /// Torque at idle, as a fraction of the peak.
    pub idle_torque_frac: f64,
    /// Peak crankshaft torque, newton-metres.
    pub peak_torque_nm: f64,
    /// Where the torque curve peaks, rpm.
    pub peak_torque_rpm: f64,
    /// Where the limiter cuts, rpm.
    pub redline_rpm: f64,
    /// Torque at the redline, as a fraction of the peak.
    pub redline_torque_frac: f64,
    /// Reverse's ratio — reverse is a gear.
    pub reverse_ratio: f64,
    /// Downshift below this, rpm.
    pub shift_down_rpm: f64,
    /// How long a shift takes, seconds — no drive torque crosses it.
    pub shift_time_s: f64,
    /// Upshift above this, rpm.
    pub shift_up_rpm: f64,
    /// Stability-control strength, `[0, 1]`; `0` is off.
    pub stability_control: f64,
    /// How fast the road wheels turn toward the demand, degrees per second.
    pub steer_rate_deg_per_s: f64,
    /// How fast they return to centre, degrees per second.
    pub steer_return_deg_per_s: f64,
    /// The torque curve's one shape knob, `(0, 1)`; `0.5` is straight lines.
    pub torque_curve_bias: f64,
    /// Traction control: the drive slip ratio above which torque is bled off;
    /// `0` is off.
    pub traction_control_slip: f64,
    /// Tangent of the slip angle at which lateral grip peaks.
    pub tyre_lat_peak_slip: f64,
    /// How stiff the lateral rise is, `(0, 1)`.
    pub tyre_lat_rise_bias: f64,
    /// How fast µ falls as vertical load rises over the static share.
    pub tyre_load_sensitivity: f64,
    /// Slip ratio at which longitudinal grip peaks.
    pub tyre_long_peak_slip: f64,
    /// How stiff the longitudinal rise is, `(0, 1)`.
    pub tyre_long_rise_bias: f64,
    /// Grip once fully sliding, as a fraction of the peak.
    pub tyre_slide_frac: f64,
    /// A wheel's rotational inertia, kg·m².
    pub wheel_inertia_kgm2: f64,
}

impl Default for VehicleClass {
    /// The Ring-0 defaults, read from the ONE definition of them rather than
    /// restated — a second copy of fifteen numbers is a second thing to update.
    fn default() -> Self {
        Self::from_tuning(&crate::vehicle::VehicleTuning::default())
    }
}

impl VehicleClass {
    /// Project a [`VehicleTuning`](crate::vehicle::VehicleTuning) onto the
    /// authored subset.
    pub fn from_tuning(t: &crate::vehicle::VehicleTuning) -> Self {
        Self {
            brake_force_n: t.brake_force_n,
            damping_ns_per_m: t.damping_ns_per_m,
            drag_n_per_mps2: t.drag_n_per_mps2,
            enter_time_s: t.enter_time_s,
            handbrake_force_n: t.handbrake_force_n,
            lateral_grip: t.lateral_grip,
            longitudinal_grip: t.longitudinal_grip,
            max_engine_force_n: t.max_engine_force_n,
            max_speed_mps: t.max_speed_mps,
            max_steer_deg: t.max_steer_deg,
            min_steer_deg: t.min_steer_deg,
            rest_length_m: t.rest_length_m,
            rolling_resistance: t.rolling_resistance,
            stiffness_n_per_m: t.stiffness_n_per_m,
            travel_m: t.travel_m,
            abs_slip: t.abs_slip,
            ackermann: t.ackermann,
            anti_roll_front_n_per_m: t.anti_roll_front_n_per_m,
            anti_roll_rear_n_per_m: t.anti_roll_rear_n_per_m,
            brake_bias: t.brake_bias,
            cog_height_m: t.cog_height_m,
            diff_lock_front: t.diff_lock_front,
            diff_lock_rear: t.diff_lock_rear,
            downforce_centre_z: t.downforce_centre_z,
            downforce_n_per_mps2: t.downforce_n_per_mps2,
            drag_lateral_n_per_mps2: t.drag_lateral_n_per_mps2,
            engine_brake_nm: t.engine_brake_nm,
            enter_warp_end: t.enter_warp_end,
            enter_warp_start: t.enter_warp_start,
            final_drive: t.final_drive,
            front_torque_split: t.front_torque_split,
            gear_1_ratio: t.gear_1_ratio,
            gear_2_ratio: t.gear_2_ratio,
            gear_3_ratio: t.gear_3_ratio,
            gear_4_ratio: t.gear_4_ratio,
            gear_5_ratio: t.gear_5_ratio,
            gear_6_ratio: t.gear_6_ratio,
            gear_7_ratio: t.gear_7_ratio,
            gear_8_ratio: t.gear_8_ratio,
            gear_count: t.gear_count,
            idle_rpm: t.idle_rpm,
            idle_torque_frac: t.idle_torque_frac,
            peak_torque_nm: t.peak_torque_nm,
            peak_torque_rpm: t.peak_torque_rpm,
            redline_rpm: t.redline_rpm,
            redline_torque_frac: t.redline_torque_frac,
            reverse_ratio: t.reverse_ratio,
            shift_down_rpm: t.shift_down_rpm,
            shift_time_s: t.shift_time_s,
            shift_up_rpm: t.shift_up_rpm,
            stability_control: t.stability_control,
            steer_rate_deg_per_s: t.steer_rate_deg_per_s,
            steer_return_deg_per_s: t.steer_return_deg_per_s,
            torque_curve_bias: t.torque_curve_bias,
            traction_control_slip: t.traction_control_slip,
            tyre_lat_peak_slip: t.tyre_lat_peak_slip,
            tyre_lat_rise_bias: t.tyre_lat_rise_bias,
            tyre_load_sensitivity: t.tyre_load_sensitivity,
            tyre_long_peak_slip: t.tyre_long_peak_slip,
            tyre_long_rise_bias: t.tyre_long_rise_bias,
            tyre_slide_frac: t.tyre_slide_frac,
            wheel_inertia_kgm2: t.wheel_inertia_kgm2,
        }
    }

    /// Lift this class back into a full
    /// [`VehicleTuning`](crate::vehicle::VehicleTuning).
    ///
    /// The exact inverse of [`from_tuning`](Self::from_tuning) since wave VEH2a —
    /// every tunable the door knows is now authored, so nothing is taken from the
    /// default any more (v25 took `enter_window`, which the window closed). That
    /// is what makes [`set`](Self::set) able to reuse the tuning door's own name
    /// list instead of restating it, and it is asserted both ways by
    /// `the_class_and_the_tuning_are_the_same_sixty_two_numbers`.
    pub fn to_tuning(&self) -> crate::vehicle::VehicleTuning {
        let mut t = crate::vehicle::VehicleTuning::default();
        for (name, value) in self.settings() {
            t.set(name, value);
        }
        t
    }

    /// Set one authored tunable **by name** (island wave VEH1a), answering
    /// whether it took.
    ///
    /// Routed through [`to_tuning`](Self::to_tuning) and
    /// `VehicleTuning::set` rather than matching on names here, because a second
    /// name list is the P29.6 audit's A14 defect and this type already exists to
    /// be the *serializable projection* of that one. The consequence worth
    /// stating: whatever `VehicleTuning::names()` accepts, an authored catalogue
    /// row accepts, on the day it is added and not a wave later.
    pub fn set(&mut self, name: &str, value: f64) -> bool {
        let mut t = self.to_tuning();
        if !t.set(name, value) {
            return false;
        }
        *self = Self::from_tuning(&t);
        true
    }

    /// This class's `(name, value)` pairs in [`VehicleTuning::names`]'s sorted
    /// order — what the bridge feeds
    /// [`Vehicle::tune`](crate::vehicle::Vehicle::tune).
    ///
    /// Ordered, because two tunables can interact (a `max_speed_mps` below the
    /// current speed changes what `max_engine_force_n` does) and an unordered
    /// application would make the installed class depend on a map's iteration.
    pub fn settings(&self) -> [(&'static str, f64); 62] {
        [
            ("abs_slip", self.abs_slip),
            ("ackermann", self.ackermann),
            ("anti_roll_front_n_per_m", self.anti_roll_front_n_per_m),
            ("anti_roll_rear_n_per_m", self.anti_roll_rear_n_per_m),
            ("brake_bias", self.brake_bias),
            ("brake_force_n", self.brake_force_n),
            ("cog_height_m", self.cog_height_m),
            ("damping_ns_per_m", self.damping_ns_per_m),
            ("diff_lock_front", self.diff_lock_front),
            ("diff_lock_rear", self.diff_lock_rear),
            ("downforce_centre_z", self.downforce_centre_z),
            ("downforce_n_per_mps2", self.downforce_n_per_mps2),
            ("drag_lateral_n_per_mps2", self.drag_lateral_n_per_mps2),
            ("drag_n_per_mps2", self.drag_n_per_mps2),
            ("engine_brake_nm", self.engine_brake_nm),
            ("enter_time_s", self.enter_time_s),
            ("enter_warp_end", self.enter_warp_end),
            ("enter_warp_start", self.enter_warp_start),
            ("final_drive", self.final_drive),
            ("front_torque_split", self.front_torque_split),
            ("gear_1_ratio", self.gear_1_ratio),
            ("gear_2_ratio", self.gear_2_ratio),
            ("gear_3_ratio", self.gear_3_ratio),
            ("gear_4_ratio", self.gear_4_ratio),
            ("gear_5_ratio", self.gear_5_ratio),
            ("gear_6_ratio", self.gear_6_ratio),
            ("gear_7_ratio", self.gear_7_ratio),
            ("gear_8_ratio", self.gear_8_ratio),
            ("gear_count", self.gear_count),
            ("handbrake_force_n", self.handbrake_force_n),
            ("idle_rpm", self.idle_rpm),
            ("idle_torque_frac", self.idle_torque_frac),
            ("lateral_grip", self.lateral_grip),
            ("longitudinal_grip", self.longitudinal_grip),
            ("max_engine_force_n", self.max_engine_force_n),
            ("max_speed_mps", self.max_speed_mps),
            ("max_steer_deg", self.max_steer_deg),
            ("min_steer_deg", self.min_steer_deg),
            ("peak_torque_nm", self.peak_torque_nm),
            ("peak_torque_rpm", self.peak_torque_rpm),
            ("redline_rpm", self.redline_rpm),
            ("redline_torque_frac", self.redline_torque_frac),
            ("rest_length_m", self.rest_length_m),
            ("reverse_ratio", self.reverse_ratio),
            ("rolling_resistance", self.rolling_resistance),
            ("shift_down_rpm", self.shift_down_rpm),
            ("shift_time_s", self.shift_time_s),
            ("shift_up_rpm", self.shift_up_rpm),
            ("stability_control", self.stability_control),
            ("steer_rate_deg_per_s", self.steer_rate_deg_per_s),
            ("steer_return_deg_per_s", self.steer_return_deg_per_s),
            ("stiffness_n_per_m", self.stiffness_n_per_m),
            ("torque_curve_bias", self.torque_curve_bias),
            ("traction_control_slip", self.traction_control_slip),
            ("travel_m", self.travel_m),
            ("tyre_lat_peak_slip", self.tyre_lat_peak_slip),
            ("tyre_lat_rise_bias", self.tyre_lat_rise_bias),
            ("tyre_load_sensitivity", self.tyre_load_sensitivity),
            ("tyre_long_peak_slip", self.tyre_long_peak_slip),
            ("tyre_long_rise_bias", self.tyre_long_rise_bias),
            ("tyre_slide_frac", self.tyre_slide_frac),
            ("wheel_inertia_kgm2", self.wheel_inertia_kgm2),
        ]
    }

    /// **Install this class on a running vehicle**, through the trait's own
    /// tuning door. Answers how many settings the implementation took.
    ///
    /// A class need not be understood in full: an island `Vehicle` that has no
    /// handbrake refuses `handbrake_force_n` and keeps everything else, which is
    /// the standing "a refusal is a value" rule and is why this returns a count
    /// rather than a `Result`. A **non-finite** value is refused by
    /// `VehicleTuning::set` for the same reason and is not this layer's business.
    pub fn install(&self, vehicle: &mut dyn crate::vehicle::Vehicle) -> usize {
        self.settings()
            .into_iter()
            .filter(|(name, value)| vehicle.tune(name, *value))
            .count()
    }
}

// ── Joints (P12.1) ──────────────────────────────────────────────────────────
//
// A `Joint2D`/`Joint3D` links its entity's body to ANOTHER entity's body,
// mirroring the flat-struct precedent of the collider components (a `kind` enum
// selects the family; sibling fields carry the per-family numeric params, so the
// reflection Details grid can edit them). The other body is referenced by its
// stable `Guid` in `other`, now an `EntityRef` (E-P1) — reflected opaquely so
// the Details panel surfaces an entity-picker, serde-transparent so the wire is
// byte-identical to the old `Option<Uuid>`.

fn default_joint_axis() -> Vec3d {
    Vec3d::new(0.0, 1.0, 0.0)
}
fn default_joint_axis_2d() -> Vec2d {
    Vec2d::new(1.0, 0.0)
}
fn default_motor_damping() -> f64 {
    1.0
}
fn default_motor_max_force() -> f64 {
    f64::MAX
}
fn default_rope_length() -> f64 {
    1.0
}

/// The joint family for a [`Joint3D`]. A flat reflected enum (Details dropdown),
/// mirroring [`inf_physics::JointKind3D`].
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum JointKind3D {
    /// Weld the two bodies rigidly.
    #[default]
    Fixed,
    /// A hinge about `axis` (angle limits + motor optional).
    Revolute,
    /// A slider along `axis` (distance limits + motor optional).
    Prismatic,
    /// A ball-and-socket (3-DOF rotation).
    Spherical,
    /// A rope: anchors kept within `max_distance`.
    Distance,
}

/// A 3D joint (P12.1) linking this entity's [`RigidBody3D`] to `other`'s. The
/// `PhysicsBridge3D` spawns/despawns it alongside the bodies.
///
/// Additive component: every field carries `#[serde(default)]`. **NOTE (v6
/// persistence gap):** `Joint3D` is NOT yet in the `.inf_lvl` `EntityRecord`
/// (that needs a schema bump to v6) — it round-trips through the live ECS and the
/// physics bridge, but is not persisted to disk this batch. See
/// `inf-editor-core::scene::serialize` (SCHEMA_VERSION doc + guard-test comment).
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Joint3D {
    /// The OTHER body's entity `Guid`. `None` → the joint is unbound and the
    /// bridge skips it. An [`EntityRef`] (E-P1): reflected
    /// opaquely so the Details panel shows an entity-picker widget;
    /// serde-transparent, so the on-disk stream is byte-identical to the old
    /// `Option<Uuid>`.
    #[serde(default)]
    pub other: EntityRef,
    /// Fixed / Revolute / Prismatic / Spherical / Distance.
    #[serde(default)]
    pub kind: JointKind3D,
    /// Anchor on this body, in its local frame.
    #[serde(default)]
    pub local_anchor: Vec3d,
    /// Anchor on the other body, in its local frame.
    #[serde(default)]
    pub other_anchor: Vec3d,
    /// Hinge/slider axis (Revolute/Prismatic), body-local. Default `+Y`.
    #[serde(default = "default_joint_axis")]
    pub axis: Vec3d,
    /// Enable the `[limit_min, limit_max]` limits.
    #[serde(default)]
    pub limits_enabled: bool,
    /// Lower limit (radians for Revolute, world units for Prismatic).
    #[serde(default)]
    pub limit_min: f64,
    /// Upper limit.
    #[serde(default)]
    pub limit_max: f64,
    /// Enable the motor (Revolute/Prismatic).
    #[serde(default)]
    pub motor_enabled: bool,
    /// Motor target position (angle/distance).
    #[serde(default)]
    pub motor_target_pos: f64,
    /// Motor target velocity.
    #[serde(default)]
    pub motor_target_vel: f64,
    /// Motor stiffness (spring constant; `0` → a pure velocity motor).
    #[serde(default)]
    pub motor_stiffness: f64,
    /// Motor damping.
    #[serde(default = "default_motor_damping")]
    pub motor_damping: f64,
    /// Maximum motor force/torque.
    #[serde(default = "default_motor_max_force")]
    pub motor_max_force: f64,
    /// Rope max length (Distance kind).
    #[serde(default = "default_rope_length")]
    pub max_distance: f64,
}

impl Default for Joint3D {
    fn default() -> Self {
        Self {
            other: EntityRef::NONE,
            kind: JointKind3D::Fixed,
            local_anchor: Vec3d::ZERO,
            other_anchor: Vec3d::ZERO,
            axis: default_joint_axis(),
            limits_enabled: false,
            limit_min: 0.0,
            limit_max: 0.0,
            motor_enabled: false,
            motor_target_pos: 0.0,
            motor_target_vel: 0.0,
            motor_stiffness: 0.0,
            motor_damping: default_motor_damping(),
            motor_max_force: default_motor_max_force(),
            max_distance: default_rope_length(),
        }
    }
}

/// The joint family for a [`Joint2D`]. In 2D the only hinge is Revolute (about
/// the implicit Z axis); Spherical does not exist. Mirrors
/// [`inf_physics::JointKind2D`].
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum JointKind2D {
    /// Weld the two bodies rigidly.
    #[default]
    Fixed,
    /// A hinge about the Z axis (angle limits + motor optional).
    Revolute,
    /// A slider along `axis` (distance limits + motor optional).
    Prismatic,
    /// A rope: anchors kept within `max_distance`.
    Distance,
}

/// A 2D joint (P12.1) linking this entity's [`RigidBody2D`] to `other`'s. The
/// `d2` mirror of [`Joint3D`] (same v6 persistence-gap note applies).
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Joint2D {
    /// The OTHER body's entity `Guid`. `None` → unbound (skipped). An
    /// [`EntityRef`] (E-P1) — see [`Joint3D::other`].
    #[serde(default)]
    pub other: EntityRef,
    /// Fixed / Revolute / Prismatic / Distance.
    #[serde(default)]
    pub kind: JointKind2D,
    /// Anchor on this body (local, XY).
    #[serde(default)]
    pub local_anchor: Vec2d,
    /// Anchor on the other body (local, XY).
    #[serde(default)]
    pub other_anchor: Vec2d,
    /// Slider axis (Prismatic), body-local. Default `+X`.
    #[serde(default = "default_joint_axis_2d")]
    pub axis: Vec2d,
    /// Enable the `[limit_min, limit_max]` limits.
    #[serde(default)]
    pub limits_enabled: bool,
    /// Lower limit (radians for Revolute, world units for Prismatic).
    #[serde(default)]
    pub limit_min: f64,
    /// Upper limit.
    #[serde(default)]
    pub limit_max: f64,
    /// Enable the motor.
    #[serde(default)]
    pub motor_enabled: bool,
    /// Motor target position.
    #[serde(default)]
    pub motor_target_pos: f64,
    /// Motor target velocity.
    #[serde(default)]
    pub motor_target_vel: f64,
    /// Motor stiffness (`0` → velocity motor).
    #[serde(default)]
    pub motor_stiffness: f64,
    /// Motor damping.
    #[serde(default = "default_motor_damping")]
    pub motor_damping: f64,
    /// Maximum motor force/torque.
    #[serde(default = "default_motor_max_force")]
    pub motor_max_force: f64,
    /// Rope max length (Distance kind).
    #[serde(default = "default_rope_length")]
    pub max_distance: f64,
}

impl Default for Joint2D {
    fn default() -> Self {
        Self {
            other: EntityRef::NONE,
            kind: JointKind2D::Fixed,
            local_anchor: Vec2d::ZERO,
            other_anchor: Vec2d::ZERO,
            axis: default_joint_axis_2d(),
            limits_enabled: false,
            limit_min: 0.0,
            limit_max: 0.0,
            motor_enabled: false,
            motor_target_pos: 0.0,
            motor_target_vel: 0.0,
            motor_stiffness: 0.0,
            motor_damping: default_motor_damping(),
            motor_max_force: default_motor_max_force(),
            max_distance: default_rope_length(),
        }
    }
}

/// A scene camera (distinct from the editor's fly camera).
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Camera {
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov_y_deg: 60.0,
            near: 0.05,
            far: 10_000.0,
        }
    }
}

/// The number of splat material layers a [`Terrain`] blends (packed as RGBA8
/// per-sample weights in [`TerrainData`]). Matches `inf_terrain::SPLAT_LAYERS`
/// and the four channels of a weight sample.
pub const TERRAIN_LAYERS: usize = 4;

/// One splat material layer definition (P10.4): the surface a terrain sample
/// blends toward where its weight channel dominates.
///
/// As a nested `#[reflect(ignore)]` array element it isn't surfaced in the
/// generic Details grid (authored via the paint panel / defaults); it derives
/// `Reflect` + `Default` so the array serdes and reflect-constructs.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Default)]
pub struct TerrainLayer {
    /// Linear base colour blended for this layer.
    pub albedo: Color,
    /// Perceptual roughness `[0, 1]` (feeds the shared terrain lighting spec).
    #[serde(default = "default_layer_roughness")]
    pub roughness: f64,
    /// World metres per procedural detail-grain tile (triplanar detail scale).
    #[serde(default = "default_layer_tex_scale")]
    pub tex_scale: f64,
    /// The `.inf_mat` whose textures this layer samples (schema **v24**,
    /// Wave G) — `None` keeps the solid-albedo-plus-procedural-grain surface
    /// every terrain had before.
    ///
    /// # The remainder this closes
    ///
    /// The comment that used to stand here said a texture GUID was *deliberately
    /// absent* because "the interactive viewport can't yet upload asset
    /// textures". That stopped being true at Wave T: the terrain shader grew a
    /// four-layer virtual-texture path (`RenderTerrainLayer::vt`,
    /// `terrain_layers` in `terrain.wgsl`) which resolves real albedo/normal/ORM
    /// pages per splat layer. What it did not grow was any way for an author to
    /// *say which material* — every projector wrote `VtTextureSet::NONE`, so the
    /// branch could never execute. Wave T's own disposition memo names this field
    /// as the thing that turns that path from a capability into a feature.
    ///
    /// # Wire note
    ///
    /// This is stored as a bare `Uuid` rather than an `AssetId` because
    /// `inf-ecs` deliberately does not depend on `inf-asset` (the same reason
    /// [`Terrain::asset`] and [`Terrain::biome_set`] are `Option<Uuid>`); the
    /// editor glue resolves it. Appending it grows `TerrainLayer`, which is a
    /// wire-format change under positional bincode — hence the frozen
    /// `TerrainLayerV23` in both scene codec mirrors.
    ///
    /// `#[reflect(ignore)]` for the same reason every other asset reference in
    /// this file carries it: `Uuid` is not `Reflect`, and an asset binding is
    /// picked from the Content Drawer rather than typed into the Details grid.
    #[serde(default)]
    #[reflect(ignore)]
    pub material: Option<uuid::Uuid>,
}

fn default_layer_roughness() -> f64 {
    0.9
}
fn default_layer_tex_scale() -> f64 {
    8.0
}

impl Default for TerrainLayer {
    fn default() -> Self {
        Self {
            albedo: Color::new(0.35, 0.35, 0.35, 1.0),
            roughness: default_layer_roughness(),
            tex_scale: default_layer_tex_scale(),
            material: None,
        }
    }
}

/// The default four-layer palette: grass → rock → dirt → snow. The paint UI's
/// layer swatches mirror these (per-terrain layer-colour editing in Details is
/// the follow-up), and the terrain golden authors gradients across them.
pub fn default_terrain_layers() -> [TerrainLayer; TERRAIN_LAYERS] {
    [
        TerrainLayer {
            albedo: Color::new(0.20, 0.34, 0.14, 1.0), // grass
            roughness: 0.92,
            tex_scale: 6.0,
            material: None,
        },
        TerrainLayer {
            albedo: Color::new(0.33, 0.30, 0.27, 1.0), // rock
            roughness: 0.85,
            tex_scale: 4.0,
            material: None,
        },
        TerrainLayer {
            albedo: Color::new(0.42, 0.30, 0.18, 1.0), // dirt
            roughness: 0.95,
            tex_scale: 5.0,
            material: None,
        },
        TerrainLayer {
            albedo: Color::new(0.86, 0.89, 0.94, 1.0), // snow
            roughness: 0.65,
            tex_scale: 10.0,
            material: None,
        },
    ]
}

fn default_macro_variation() -> f64 {
    0.15
}

/// A heightfield terrain (P10.1) — the engine's massive-terrain component.
///
/// The scalar config (`meters_per_sample`, `tile_resolution`) is reflected so the
/// Details grid surfaces it; the paged height data ([`TerrainData`]) is
/// `#[reflect(ignore)]` + serde-persisted, exactly like the [`Tilemap`] chunk
/// store — sculpt/paint/import tools (P10.2/P10.4) edit it, not the property
/// grid. The [`TerrainData`] is the authority for rendering and queries; the
/// reflected scalars mirror its config (changing them in Details is a reconfigure
/// hint the P10.2 terrain tooling applies — the stored `data` keeps its own
/// config until then).
///
/// ## Splat materials (P10.4)
///
/// The paged [`TerrainData`] now stores a per-sample RGBA8 splat weight beside
/// each height; `layers` defines the four [`TerrainLayer`]s those weights blend,
/// and `macro_variation` is the amplitude of a large-scale fBm albedo modulation
/// applied in the terrain shader. `layers` is `#[reflect(ignore)]` (an
/// array-of-struct is not a Details-grid scalar; edited via the paint UI /
/// defaults), while `macro_variation` is a plain reflected `f64` (Details-editable).
///
/// ## Streaming from a `.inf_terrain` asset (P16.3 · schema v9)
///
/// [`asset`](Self::asset) optionally points at a `.inf_terrain` asset — the
/// out-of-level container that holds the tiles plus their LOD pyramid, cooked
/// uncompressed so a runtime pages tiles straight out of an mmap'd `.ipack`
/// ([`inf_terrain::TerrainAsset`]). **Both paths are legal and the rule is
/// simple: the inline [`data`](Self::data) is authoritative while `asset` is
/// `None`.** When `asset` is set, the tile data streams from it and the inline
/// `data` is the resident working set. The editor keeps terrain inline for now;
/// the authoring flow that promotes a terrain to an asset lands in a later batch.
/// The cook follows `Terrain.asset` as a real level → asset dependency edge.
///
/// ## FROZEN SHAPE (`.inf_lvl` schema v4)
///
/// This component's field set is **finalized for the upcoming `.inf_lvl` v4
/// schema batch** (the migration that first persists `Terrain`/[`PcgVolume`] in
/// the `EntityRecord`). Every field is additive and `#[serde(default)]`, so a
/// minimal `{}` payload and any pre-v4 partial payload decode; new fields append.
/// Do not reorder or repurpose existing fields — extend additively. (`asset`
/// appended at v9 is exactly such an additive extension; because bincode is not
/// self-describing, the pre-v9 layout is frozen as `TerrainV8` in both scene
/// codecs and every v4..v8 record decodes its terrain slot through it.)
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Terrain {
    /// World units between adjacent height samples.
    #[serde(default = "default_terrain_mps")]
    pub meters_per_sample: f64,
    /// Samples per tile side (mirrors the paged data's resolution).
    #[serde(default = "default_terrain_resolution")]
    pub tile_resolution: u32,
    /// The paged `f64` world-space heightfield (heights + splat weights).
    /// `#[reflect(ignore)]` (not a property-grid scalar) but serde-persisted so
    /// authored terrain survives a save/load. `TerrainData`'s own manual serde
    /// keeps it deterministic and byte-stable for unpainted tiles.
    #[serde(default)]
    #[reflect(ignore)]
    pub data: TerrainData,
    /// The four splat material layers the per-sample weights blend (P10.4).
    /// `#[reflect(ignore)]` + serde-persisted (authored via the paint UI /
    /// defaults), like an array field with no Details widget.
    #[serde(default = "default_terrain_layers")]
    #[reflect(ignore)]
    pub layers: [TerrainLayer; TERRAIN_LAYERS],
    /// Amplitude of the large-scale fBm albedo modulation applied in the terrain
    /// shader (`0` = off). A plain reflected `f64` → Details-editable.
    #[serde(default = "default_macro_variation")]
    pub macro_variation: f64,
    /// GUID of the `.inf_terrain` asset this terrain streams its tiles from
    /// (schema v9, P16.3).
    ///
    /// `None` — the default and what the editor still writes — means the inline
    /// [`data`](Self::data) is the whole terrain and its only authority. When set,
    /// the tiles (and their LOD pyramid) live in the asset and `data` holds the
    /// resident working set; the cook follows this edge to pack the
    /// `.inf_terrain`. `#[reflect(ignore)]` (an asset reference, not a
    /// Details-grid scalar — picked through the asset UI, like
    /// [`MeshRef::asset`]) and `#[serde(default)]` so every pre-v9 payload
    /// decodes with it absent.
    #[serde(default)]
    #[reflect(ignore)]
    pub asset: Option<Uuid>,
    /// GUID of the `.inf_biomes` [`BiomeSet`](inf_terrain::BiomeSet) this
    /// terrain's per-sample biome ids name (schema v16, P19.2).
    ///
    /// `None` means the terrain has no biome vocabulary: the paint tool has
    /// nothing to offer, and every sample reads
    /// [`UNASSIGNED_BIOME`](inf_terrain::UNASSIGNED_BIOME). Setting it is what
    /// arms biome painting and the Biomes view mode; the cook follows the edge to
    /// pack the set beside the level, exactly like [`asset`](Self::asset).
    ///
    /// `#[reflect(ignore)]` (an asset reference, not a Details-grid scalar) and
    /// `#[serde(default)]` so every pre-v16 payload decodes with it absent —
    /// though the v16 bump is forced by the *tile* layout below it, not by this
    /// field.
    #[serde(default)]
    #[reflect(ignore)]
    pub biome_set: Option<Uuid>,
    /// The instances P19.3's **biome binding** scattered over this terrain: each
    /// painted biome's `.inf_pcg` graph evaluated over the region its id owns,
    /// merged in ascending biome-id order.
    ///
    /// **Derived, exactly like [`PcgVolume::evaluated`]** — `#[serde(skip)]`, so it
    /// costs the wire nothing and the schema stays where P19.2 left it (bincode is
    /// positional; a *persisted* field here would have forced v17 in both codec
    /// mirrors). It is rebuilt by the editor's evaluate command and by the player
    /// on level load, which is what makes those two paths comparable.
    #[serde(skip)]
    #[reflect(ignore)]
    pub biome_population: Vec<ScatteredInstance>,
}

fn default_terrain_mps() -> f64 {
    inf_terrain::DEFAULT_METERS_PER_SAMPLE
}
fn default_terrain_resolution() -> u32 {
    inf_terrain::DEFAULT_TILE_RESOLUTION
}

impl Default for Terrain {
    fn default() -> Self {
        let data = TerrainData::default();
        Self {
            meters_per_sample: data.meters_per_sample(),
            tile_resolution: data.tile_resolution(),
            data,
            layers: default_terrain_layers(),
            macro_variation: default_macro_variation(),
            asset: None,
            biome_set: None,
            biome_population: Vec::new(),
        }
    }
}

impl Terrain {
    /// An empty terrain configured with `tile_resolution` samples per tile side
    /// and `meters_per_sample` world spacing. The reflected scalars mirror the
    /// paged data's config; layers/macro use the defaults.
    pub fn configured(tile_resolution: u32, meters_per_sample: f64) -> Self {
        let data = TerrainData::new(tile_resolution, meters_per_sample);
        Self {
            meters_per_sample: data.meters_per_sample(),
            tile_resolution: data.tile_resolution(),
            data,
            layers: default_terrain_layers(),
            macro_variation: default_macro_variation(),
            asset: None,
            biome_set: None,
            biome_population: Vec::new(),
        }
    }

    /// **Every `.inf_mat` GUID this terrain's layers bind** (wave TER2a), in
    /// layer order, deduplicated only by the caller.
    ///
    /// # Why it is a function and not four field reads
    ///
    /// Because *four* separate walks have to agree about what a level binds, and
    /// the P22.2 law is about exactly this shape: the editor's virtual-texture
    /// cache key, the editor's material resolver, the PIE payload builder and
    /// the shipped player's pack reader each enumerate a level's material
    /// bindings, and `VtTextures::want_floor` is a pure function of the
    /// registration *sequence*, so a walk that missed a binding — or found it in
    /// a different order — would page a different set of texture tiles from the
    /// other three. "Which of a terrain's fields is a material binding" is a
    /// rule; it lives here, once.
    ///
    /// It answers **nothing** for a terrain whose layers name no material, which
    /// is every terrain authored before Wave G's `TerrainLayer::material` and
    /// the permanent flat-albedo path.
    pub fn layer_materials(&self) -> impl Iterator<Item = uuid::Uuid> + '_ {
        self.layers.iter().filter_map(|l| l.material)
    }
}

/// One baked PCG instance — the world-space result of evaluating a
/// [`PcgVolume`]'s graph over the scene terrain. The viewport projects each into
/// the existing mesh-instance path.
///
/// A deliberately dependency-light mirror of `inf_pcg::PcgInstance` (kept local
/// so the foundational ECS crate does not pull the whole scatter runtime just to
/// hold a result cache): the `pcg_evaluate` command converts one to the other.
/// Not reflected / not serialized — it is a derived cache (see [`PcgVolume`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScatteredInstance {
    /// World-space position.
    pub position: DVec3,
    /// World-space orientation.
    pub rotation: DQuat,
    /// Uniform scale.
    pub scale: f64,
    /// Palette index resolved by the rule (which mesh/primitive kind).
    ///
    /// **Rule-local, and therefore not an identity.** See [`mesh`](Self::mesh):
    /// populations are concatenated without offsetting this, so it survives only
    /// as the placeholder *tint* index it has always been.
    pub kind: u32,
    /// **The `.inf_mesh` this instance draws** (wave TER2b) — `inf_pcg::PcgKind::mesh`,
    /// carried through from the rule that owns the palette.
    ///
    /// Mirrors `inf_pcg::PcgInstance::mesh` field for field, and exists for the
    /// same reason: [`kind`](Self::kind) is a *rule-local* index into a palette
    /// that no longer exists by the time a projector reads it, so two instances
    /// carrying `0` may be a grass tuft and a wall module. A GUID is global.
    ///
    /// `None` is "draw the placeholder primitive" — a kind with no mesh, or a
    /// mesh the host could not resolve. Since island wave I8b a **grammar
    /// module names one too**.
    pub mesh: Option<uuid::Uuid>,
    /// **The half-extents of the box this instance occupies**, metres, in its
    /// own rotated frame — `None` for "the unit primitive at
    /// [`scale`](Self::scale)" (island wave I8b).
    ///
    /// Mirrors `inf_pcg::PcgInstance::extent` field for field. [`scale`] is one
    /// uniform `f64` and every building module carries `1.0`, so before this
    /// field a 10 m slab and a 0.3 m mullion drew as the same one-metre cube
    /// while their colliders were right. The projector turns it into
    /// `ScatterInstance::scale`, which has been a `Vec3` since IB-2b gave the
    /// structure shell three half-extents.
    ///
    /// [`scale`]: Self::scale
    pub extent: Option<[f32; 3]>,
    /// **How brightly this instance emits at night**, as a multiplier on its own
    /// colour — `0.0` for everything that does not (island wave I8b).
    ///
    /// Mirrors `inf_pcg::PcgInstance::glow`. Authored, not resolved: the hour is
    /// applied once, by the projector, which is the only place that knows it.
    pub glow: f32,
}

/// One placed **collision box** a `.inf_pcg` graph produced (P19.5) — the solid
/// half of [`PcgVolume`]'s evaluation, beside [`ScatteredInstance`]'s visible
/// half.
///
/// A dependency-light mirror of `inf_pcg::PcgCollider`, on exactly the terms
/// [`ScatteredInstance`] mirrors `inf_pcg::PcgInstance`. Not reflected, not
/// serialized: it is derived state, recomputed wherever `evaluated` is.
///
/// **This is what makes a grammar-built building enterable.** Scattered content
/// has always been geometry and nothing else; a wall that a character walks
/// through is a picture of a wall. The physics bridge reads these and builds one
/// static box collider each, so a doorway — a stretch of wall where no module,
/// and therefore no box, was placed — is a hole you can walk through.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScatteredSolid {
    /// World-space centre of the box.
    pub center: DVec3,
    /// Half-extents in metres, in the box's own (rotated) frame.
    pub half_extents: DVec3,
    /// Yaw-only orientation.
    pub rotation: DQuat,
}

/// **One doorway a `.inf_pcg` graph's building planned** (I6) — the hinge half
/// of [`PcgVolume`]'s evaluation, beside [`ScatteredSolid`]'s solid half.
///
/// A dependency-light mirror of `inf_pcg::building::PcgDoorway`, on exactly the
/// terms [`ScatteredSolid`] mirrors `inf_pcg::PcgCollider`. Not reflected, not
/// serialized: derived state, recomputed wherever `evaluated` is, so it moves no
/// schema.
///
/// **This is what makes a grammar-built building's doorway a DOOR.** P19.5 made
/// a scattered wall solid so a doorway became a hole a character could walk
/// through; the hole has never had anything in it. The physics bridge reads
/// these and hangs one kinematic leaf on each.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoorwaySlot {
    /// The hinge, world metres, at the leaf's mid-height.
    pub hinge: DVec3,
    /// The compass yaw from the hinge toward the free edge when shut, degrees.
    pub closed_yaw_deg: f64,
    /// The leaf's width, metres.
    pub width_m: f64,
    /// The leaf's height, metres.
    pub height_m: f64,
    /// The leaf's thickness, metres.
    pub thickness_m: f64,
    /// The compass yaw of the inside face's outward normal, degrees.
    pub inside_yaw_deg: f64,
    /// Whether this is the building's one exterior door.
    pub exterior: bool,
    /// Which storey it is on.
    pub floor: u32,
}

/// **Which run of a volume's derived lists is one building**, and what that
/// building's shell box is (IB-2b).
///
/// A dependency-light mirror of `inf_pcg::building::StructureGroup`, on exactly
/// the terms [`ScatteredSolid`] mirrors `inf_pcg::PcgCollider`. Derived, never
/// serialized, never reflected.
///
/// # What it buys
///
/// A grammar building is ~800–2 000 boxes; a city of a thousand is ~10⁶, and at
/// the certification's measured **0.363 µs per collider per fixed step** that is
/// a third of a second of physics for a scene nobody can walk across. A group
/// lets both consumers of a volume's derived state work in units of *buildings*
/// rather than boxes: the physics bridge attaches a distant building's **shell**
/// (one collider) instead of its parts, and a host's projection draws one
/// instance instead of a thousand. Ranges rather than a per-box tag because a
/// building's boxes are already contiguous and a `u32` on every
/// [`ScatteredSolid`] would be megabytes of tag per city.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StructureGroup {
    /// The smallest oriented box containing the group's solids.
    pub shell: ScatteredSolid,
    /// First index into [`PcgVolume::structures`].
    pub start: u32,
    /// Solid count.
    pub len: u32,
    /// First index into [`PcgVolume::evaluated`].
    pub inst_start: u32,
    /// Instance count.
    pub inst_len: u32,
}

impl StructureGroup {
    /// The half-open range this group covers in [`PcgVolume::structures`].
    #[inline]
    pub fn range(&self) -> std::ops::Range<usize> {
        self.start as usize..(self.start as usize + self.len as usize)
    }

    /// The half-open range this group covers in [`PcgVolume::evaluated`].
    #[inline]
    pub fn instance_range(&self) -> std::ops::Range<usize> {
        self.inst_start as usize..(self.inst_start as usize + self.inst_len as usize)
    }
}

/// A procedural scatter volume (P10.5b): a rectangular XZ region, centered on the
/// entity's [`Transform`], populated by evaluating a `.inf_pcg` graph over the
/// scene terrain. The editor evaluates on demand (`pcg_evaluate`) and stores the
/// result in [`evaluated`](Self::evaluated); the viewport projects that cache
/// into scattered mesh instances.
///
/// Follows the shared [`Visibility`]/[`ComputedVisibility`] components (like
/// sprites / tilemaps / terrain) — no per-volume `visible` field.
///
/// ## Persistence — the same v4 gap as [`Terrain`]
///
/// This is an **additive** component (registered + reflected + serde), but — like
/// [`Terrain`] — it is **not yet persisted in `.inf_lvl`**: the schema-v3
/// `EntityRecord` has no `pcg_volume` slot, so a spawned volume is live-session
/// only. Closing the gap is the same schema-v3→v4 migration the `Terrain` guard
/// test (`terrain_is_not_persisted_yet_v4_todo`) pins; this component's guard test
/// (`pcg_volume_serde_round_trips`) documents the same. No schema bump is made
/// here.
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct PcgVolume {
    /// The `.inf_pcg` graph asset that drives scatter. `#[reflect(ignore)]` +
    /// serde-persisted (assigned by drag / the PCG panel), exactly like
    /// [`Sprite::texture`]. `None` → nothing to evaluate.
    #[serde(default)]
    #[reflect(ignore)]
    pub graph: Option<Uuid>,
    /// Half-extent of the scatter region in world XZ, centered on the entity's
    /// transform: the evaluated region is `[center − extent, center + extent]`.
    #[serde(default = "default_pcg_extent")]
    pub extent: Vec2d,
    /// Seed offset mixed into every rule's scatter seed, so two volumes sharing a
    /// graph scatter differently.
    #[serde(default)]
    pub seed: u32,
    /// Instances farther than this from the camera are skipped at projection
    /// (`0` = unlimited). A simple, documented per-instance draw-distance cull.
    #[serde(default = "default_pcg_draw_distance")]
    pub draw_distance: f64,
    /// The last evaluation's instances — a derived cache refreshed by the
    /// `pcg_evaluate` command. `#[serde(skip)]` (NOT persisted — recomputed on
    /// demand, never written to `.inf_lvl`) + `#[reflect(ignore)]`.
    #[serde(skip)]
    #[reflect(ignore)]
    pub evaluated: Vec<ScatteredInstance>,
    /// The same evaluation's **solid boxes** (P19.5) — grammar modules that
    /// declared a `collider`, and every structural part of a building. Read by
    /// the physics bridge, which turns each into one static box collider.
    ///
    /// `#[serde(skip)]` + `#[reflect(ignore)]`, exactly like
    /// [`evaluated`](Self::evaluated), and for the same reason: it is derived
    /// from the graph and the terrain, both of which the loading host already
    /// has. **That is also why P19.5 bumps no schema** — only what reaches the
    /// bytes can force a bump, and this reaches none. Pinned by the serializer's
    /// own round-trip guard.
    #[serde(skip)]
    #[reflect(ignore)]
    pub structures: Vec<ScatteredSolid>,
    /// The same evaluation's **doorways** (I6) — every place the building
    /// grammar planned a door, as a hinge a leaf can be hung on.
    ///
    /// `#[serde(skip)]` + `#[reflect(ignore)]` for exactly the reason
    /// [`structures`](Self::structures) is: it is derived from the graph and the
    /// terrain, both of which the loading host already has, so it reaches no
    /// bytes and forces no schema move. Read by the physics bridge, which turns
    /// each into one kinematic leaf under `door_leaf_guid`.
    ///
    /// Located in the world and naming no index, so nothing here has to be
    /// re-based when populations are concatenated.
    #[serde(skip)]
    #[reflect(ignore)]
    pub doorways: Vec<DoorwaySlot>,
    /// Bumped every time the derived population is replaced through
    /// [`set_structures`](Self::set_structures) or
    /// [`set_population`](Self::set_population) — the **change stamp** the
    /// physics bridge and the sim→render projection reconcile against.
    ///
    /// Without it the bridge would rebuild a descriptor for every solid on every
    /// fixed step: a furnished town is ~13 000 immovable boxes, and re-describing
    /// and re-sorting them 60 times a second to discover that a wall has not
    /// moved is the kind of cost that never shows up in a load-time budget
    /// measurement. This is the same version-stamp shape `SceneDoc::version` and
    /// `Terrain`'s tile stamps already use.
    ///
    /// # It is drawn from a PROCESS-GLOBAL counter, and that is load-bearing
    ///
    /// (Island wave I8a audit.) It used to be a per-volume `wrapping_add(1)`, so
    /// the first write of *every* volume produced `1`. Both consumers key their
    /// memo by the volume's `Guid`, and a cell that deactivates and reactivates
    /// destroys the component and builds a fresh one — whose first write is `1`
    /// again, over ground that may have been re-paged in between. Same guid, same
    /// stamp, different content: a stale hit, and the common case rather than an
    /// exotic one.
    ///
    /// Drawn from a process-global `NEXT_STRUCTURES_GEN` a stamp is unique
    /// across every volume,
    /// every level and every incarnation in the process, which is exactly the
    /// argument `inf_terrain`'s `NEXT_TILE_VERSION` and `inf_voxel`'s
    /// `NEXT_MESH_VERSION` already make for the terrain and voxel carry-forwards
    /// (see `inf_render::take_unchanged_terrain`'s memo). **Read it as an
    /// identity, never as a count**: nothing may assume the second write of a
    /// volume is `2`, or that two stamps taken from one volume differ by one.
    ///
    /// `#[serde(skip)]` like the cache it stamps. `0` means "never written" on a
    /// fresh or freshly-decoded component and is a forced miss for every
    /// consumer, which is what makes an unseen volume always resync.
    #[serde(skip)]
    #[reflect(ignore)]
    pub structures_gen: u64,
    /// Which runs of [`structures`](Self::structures) and
    /// [`evaluated`](Self::evaluated) are one building, and each building's shell
    /// (IB-2b). Empty for a volume that grows no buildings — a scatter or a
    /// `grammar.expand` fence is banded box by box, which is correct for content
    /// that has no "one structure" to speak of.
    ///
    /// Derived and stamped with [`structures`](Self::structures): the two are
    /// written together through [`set_population`](Self::set_population),
    /// because a range that outlived the list it indexes would name somebody
    /// else's walls and nothing would fail.
    #[serde(skip)]
    #[reflect(ignore)]
    pub structure_groups: Vec<StructureGroup>,
    /// **Every place a person can be in this volume's buildings** (NPC1d) — a
    /// dependency-light mirror of `inf_pcg::PcgSlot`, on exactly the terms
    /// [`DoorwaySlot`] mirrors `inf_pcg::building::PcgDoorway`.
    ///
    /// `#[serde(skip)]` + `#[reflect(ignore)]` for the reason
    /// [`doorways`](Self::doorways) is, and **that is why the wave that puts a
    /// population on an island bumps no schema**: a slot is derived from the
    /// graph and the terrain, both of which the loading host already has, so it
    /// reaches no bytes.
    ///
    /// Located in the world and naming only its own building's ordinal, so
    /// nothing here has to be re-based when populations are concatenated.
    ///
    /// Read by `inf_ecs::society`, which is the one thing that knows the whole
    /// level and can therefore pair a home with a work.
    #[serde(skip)]
    #[reflect(ignore)]
    pub residents: Vec<ResidentSlot>,
    /// **The walkable interior of this volume's slot-bearing buildings**
    /// (NPC1d), in the level's own nav namespace.
    ///
    /// Derived with [`residents`](Self::residents) and written through the same
    /// door, because a slot names a node *in this graph* and a graph that
    /// outlived the slots that index it would answer for somebody else's rooms.
    /// `#[serde(skip)]` + `#[reflect(ignore)]` on the same terms as everything
    /// else here — `inf_nav::NavGraph` is not even `Serialize`, which is the
    /// point rather than an obstacle.
    #[serde(skip)]
    #[reflect(ignore)]
    pub interior_nav: inf_nav::NavGraph,
}

/// **One place one person can be** (NPC1d): a dependency-light mirror of
/// `inf_pcg::building::society::PcgSlot`, on the P19.5 mirror terms
/// ([`ScatteredSolid`], [`DoorwaySlot`]). Derived, never serialized, never
/// reflected.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidentSlot {
    /// What this place is for.
    pub role: SlotRole,
    /// The room's centre on its own walking surface, world metres.
    pub at: DVec3,
    /// Index into the plan's rooms — what a caller turns into an `inf_nav` node
    /// id once it has minted the building's salt.
    pub room: u32,
    /// Which of the volume's own buildings this belongs to.
    pub building: u32,
    /// The storey, 0-based.
    pub floor: u32,
    /// Which of the room's own slots this is.
    pub index: u32,
    /// The node of [`PcgVolume::interior_nav`] this slot stands on.
    pub node: inf_nav::NavNodeId,
}

/// **What a room offers a person** — the mirror of `inf_pcg::SlotRole`, and the
/// vocabulary `inf_ecs::society`'s schedules are written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SlotRole {
    /// Somewhere to sleep.
    Home,
    /// Somewhere to work.
    Work,
    /// Somewhere to go that is neither — a shop.
    Errand,
}

impl SlotRole {
    /// A stable short name for diagnostics and gate traces.
    pub fn name(self) -> &'static str {
        match self {
            SlotRole::Home => "home",
            SlotRole::Work => "work",
            SlotRole::Errand => "errand",
        }
    }

    /// The byte this role folds into a trace. Frozen: `Home` 0, `Work` 1,
    /// `Errand` 2.
    pub fn as_u8(self) -> u8 {
        match self {
            SlotRole::Home => 0,
            SlotRole::Work => 1,
            SlotRole::Errand => 2,
        }
    }
}

fn default_pcg_extent() -> Vec2d {
    Vec2d::splat(50.0)
}
fn default_pcg_draw_distance() -> f64 {
    1000.0
}

impl Default for PcgVolume {
    fn default() -> Self {
        Self {
            graph: None,
            extent: default_pcg_extent(),
            seed: 0,
            draw_distance: default_pcg_draw_distance(),
            evaluated: Vec::new(),
            structures: Vec::new(),
            doorways: Vec::new(),
            structures_gen: 0,
            structure_groups: Vec::new(),
            residents: Vec::new(),
            interior_nav: inf_nav::NavGraph::new(),
        }
    }
}

/// **The process-global source of [`PcgVolume::structures_gen`]** (island wave
/// I8a audit).
///
/// Starts at `1` because `0` is reserved for "never written" on both sides of
/// every consumer. Monotone and never reset: a stamp names one population of one
/// volume in this process for the life of the process, so a memo keyed on
/// `(guid, stamp)` cannot serve a previous incarnation's payload for a volume a
/// cell destroyed and rebuilt. The twin of `inf_terrain`'s `NEXT_TILE_VERSION`
/// and `inf_voxel`'s `NEXT_MESH_VERSION`, for the same reason and with the same
/// rule.
static NEXT_STRUCTURES_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The next population change stamp.
///
/// `Relaxed` is enough: the value is compared for equality only and never orders
/// anything, and the atomic's own read-modify-write is what makes it unique.
fn next_structures_gen() -> u64 {
    NEXT_STRUCTURES_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl PcgVolume {
    /// Replace the derived solid cache and **bump its change stamp**, with no
    /// grouping — every solid stands on its own.
    ///
    /// The one supported way to write [`structures`](Self::structures): the
    /// physics bridge skips rebuilding a volume's colliders while
    /// [`structures_gen`](Self::structures_gen) is unchanged, so an assignment
    /// that bypasses this would leave stale colliders in the world. The field
    /// stays public for *reads* (and for struct literals, which start at
    /// generation `0` and therefore always resync once).
    pub fn set_structures(&mut self, solids: Vec<ScatteredSolid>) {
        self.structures = solids;
        self.structure_groups = Vec::new();
        self.structures_gen = next_structures_gen();
    }

    /// Replace a volume's **whole derived population** — instances, solids and
    /// grouping — bumping the change stamp once.
    ///
    /// # Why all three at once
    ///
    /// A [`StructureGroup`] is a pair of index ranges, and an index range is only
    /// meaningful against the exact lists it was derived from. Writing the three
    /// through three setters would make the *order* of those writes load-bearing
    /// — set the groups before the instances and every range validates against a
    /// list that is not there yet — and an ordering nobody can see is exactly the
    /// hazard `set_structures` was introduced to close for the change stamp.
    /// One call, one stamp, one validation.
    ///
    /// Groups whose ranges fall outside either list, or which are not in
    /// **strictly ascending, non-overlapping** order in *both* lists, are
    /// **dropped here**: a bad range is a composition bug, and the honest
    /// response at the door is to refuse it rather than to panic inside a fixed
    /// step sixty times a second.
    ///
    /// The ordering guarantee is not decoration — it is the invariant the
    /// physics bridge's single-cursor walk relies on to find the solids that
    /// belong to **no** group (a fence beside a building), and an overlapping
    /// pair would make one solid both a part and part of another's shell.
    /// Nothing downstream may assume the groups *cover* the lists; everything
    /// downstream may assume they are ordered.
    pub fn set_population(
        &mut self,
        instances: Vec<ScatteredInstance>,
        solids: Vec<ScatteredSolid>,
        groups: Vec<StructureGroup>,
        doorways: Vec<DoorwaySlot>,
        residents: Vec<ResidentSlot>,
        interior_nav: inf_nav::NavGraph,
    ) {
        self.doorways = doorways;
        self.residents = residents;
        self.interior_nav = interior_nav;
        let (ns, ni) = (solids.len(), instances.len());
        self.evaluated = instances;
        self.structures = solids;
        let (mut sc, mut ic) = (0usize, 0usize);
        self.structure_groups = groups
            .into_iter()
            .filter(|g| {
                let (s, i) = (g.range(), g.instance_range());
                let ok = s.end <= ns && i.end <= ni && s.start >= sc && i.start >= ic;
                if ok {
                    sc = s.end;
                    ic = i.end;
                }
                ok
            })
            .collect();
        self.structures_gen = next_structures_gen();
    }
}

/// A skinned mesh binding (P11.1): the GUIDs of the skeletal mesh asset and the
/// skeleton (`.inf_skel`) that deforms it. The `d3` skeletal analogue of
/// [`MeshRef`] — an entity carrying this renders the skinned mesh driven by its
/// [`AnimPlayer`]'s pose.
///
/// Both GUIDs are `#[reflect(ignore)]` (no asset-ref Details widget yet — the
/// same documented gap as [`Sprite::texture`]) + `#[serde(default)]`, so they are
/// assigned by drag-drop and still serde-persisted.
///
/// ## Persistence
///
/// **Persisted since scene v5.** `EntityRecord` carries `skeletal_mesh` and
/// `anim_player` slots (`inf_scene`), the cook closes
/// `level → SkeletalMesh.{mesh, skeleton}` and `level → AnimPlayer.clip` as real
/// dependency edges, and the PIE payload ships the bytes. This block used to say
/// the opposite — "not yet a slot in the `.inf_lvl` `EntityRecord` (frozen at
/// v4) … live-session only" — which stopped being true at the v5 migration it
/// itself pointed at, and was still here at v20.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[reflect(Component, Default)]
pub struct SkeletalMesh {
    /// Skeletal mesh asset GUID (`.inf_mesh` with per-vertex skin); `None` → the
    /// renderer's placeholder.
    #[serde(default)]
    #[reflect(ignore)]
    pub mesh: Option<Uuid>,
    /// Skeleton asset GUID (`.inf_skel`) that binds the mesh's joint indices.
    #[serde(default)]
    #[reflect(ignore)]
    pub skeleton: Option<Uuid>,
}

/// A clip play-head (P11.1): drives an entity's [`SkeletalMesh`] by advancing a
/// `.inf_anim` clip's time each fixed step (see [`crate::anim`]). Deterministic —
/// `t` integrates at the fixed `dt`.
///
/// See [`SkeletalMesh`] for the shared v5 persistence gap.
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct AnimPlayer {
    /// The `.inf_anim` clip GUID to play; `None` → the bind pose. `#[reflect(ignore)]`
    /// + serde-persisted (assigned by drag), like [`SkeletalMesh::mesh`].
    #[serde(default)]
    #[reflect(ignore)]
    pub clip: Option<Uuid>,
    /// Current play-head in seconds. Advanced by `speed·dt` each fixed step.
    #[serde(default)]
    pub t: f64,
    /// Playback rate multiplier (`1` = real time, `0.5` = half speed, `<0` =
    /// reverse).
    #[serde(default = "default_anim_speed")]
    pub speed: f64,
    /// Wrap `t` at the clip end (`true`) or clamp/hold the last pose (`false`).
    #[serde(default = "default_anim_looping")]
    pub looping: bool,
    /// Whether the play-head advances at all.
    #[serde(default = "default_anim_playing")]
    pub playing: bool,
    /// Clip length in seconds, resolved from the `.inf_anim` asset by the anim
    /// system so the tick can wrap/clamp without loading the asset. `0` = unknown
    /// → `t` free-runs (pose sampling still wraps). A derived cache, but
    /// `#[serde(default)]` (not `skip`) so it survives a round-trip once resolved.
    #[serde(default)]
    pub duration: f64,
}

/// `v > 0.0`, written once so the **NaN-rejecting** form reads as intent rather
/// than as a negated comparison — the `inf_pcg::grammar::span::positive`
/// discipline, which exists so clippy's `neg_cmp_op_on_partial_ord` is not
/// suppressed one `allow` at a time.
#[inline]
fn positive_duration(v: f64) -> bool {
    v > 0.0
}

fn default_anim_speed() -> f64 {
    1.0
}
fn default_anim_looping() -> bool {
    true
}
fn default_anim_playing() -> bool {
    true
}

impl Default for AnimPlayer {
    fn default() -> Self {
        Self {
            clip: None,
            t: 0.0,
            speed: default_anim_speed(),
            looping: default_anim_looping(),
            playing: default_anim_playing(),
            duration: 0.0,
        }
    }
}

impl AnimPlayer {
    /// Advance the play-head by one step of `dt` seconds (no-op when paused):
    /// `t + speed·dt`, then wrap ([`looping`](Self::looping)) or clamp against
    /// [`duration`](Self::duration). A non-positive duration leaves `t`
    /// free-running (pose sampling wraps later). Pure + deterministic — the same
    /// integration the runtime and editor Simulate ticks share (kept inline so
    /// the foundational ECS crate needs no `inf-anim` dependency; mirrors
    /// `inf_anim::advance_clip_time`).
    ///
    /// **`!positive_duration(..)`, not `<= 0.0`** (C4-4). `duration` is a plain
    /// `#[serde(default)]` field of the `.inf_lvl` `EntityRecord` — decoded by
    /// bincode with no structural check on the way in — and every ordering
    /// comparison a NaN takes part in is false, so a NaN used to pass this guard
    /// and reach `next.clamp(0.0, NaN)`, which panics (`f64::clamp` asserts
    /// `min <= max`). The looping branch is worse than a panic: `rem_euclid(NaN)`
    /// writes a NaN into `self.t`, and `t` **is** persisted, so the poison
    /// survives into the level file and out of it again.
    pub fn advance(&mut self, dt: f64) {
        if !self.playing {
            return;
        }
        let next = self.t + self.speed * dt;
        self.t = if !positive_duration(self.duration) {
            next
        } else if self.looping {
            next.rem_euclid(self.duration)
        } else {
            next.clamp(0.0, self.duration)
        };
    }
}

// ── P11.2 animation state machine (`.inf_sm`) ───────────────────────────────

/// The live play state of an animation state machine on one entity (P11.2).
///
/// # It used to be a hand-copied mirror, and P29.1 retired it
///
/// This was a field-for-field POD **copy** of `inf_anim::SmRuntime`, kept so the
/// foundational ECS crate needed no `inf-anim` dependency — with
/// `to_anim_runtime` / `from_anim_runtime` converting between the two around
/// every step. That reason expired at P24.1, when [`crate::pose`] took the
/// `inf-anim` dependency in order to own the one fixed-step pose rule; what
/// survived was a hand-maintained pair of structs that had to be edited in
/// lockstep, for no boundary that still existed.
///
/// P29.1 would have grown it from seven fields to fifteen. It is a **type alias**
/// instead, so there is one struct and the conversion functions are gone.
///
/// The derives still work out, and that is worth stating rather than discovering:
/// [`AnimStateMachine`] carries this field as `#[serde(skip)]` +
/// `#[reflect(ignore)]`, so neither `Serialize` nor `Reflect` is ever asked of
/// it — only `Clone + Copy + Debug + PartialEq + Default`, which `SmRuntime`
/// derives. Never serialized (see [`AnimStateMachine`]), which is also why v2
/// could grow the runtime without touching the scene schema.
pub type SmRuntimeState = inf_anim::SmRuntime;

fn default_params_from_vars() -> bool {
    true
}

/// Drives an entity's [`SkeletalMesh`] from an animation state machine
/// (`.inf_sm`, P11.2) instead of a single clip: each fixed step the machine
/// evaluates its transition conditions against the actor's Blueprint variables
/// and cross-fades between states (see `inf_anim::state_machine`). An entity may
/// carry either an [`AnimPlayer`] or an `AnimStateMachine`; when both are present
/// the **state machine wins**.
///
/// "Wins" was true of the sim and false of the renderer until P24.1: both fixed
/// steps advanced this component correctly and both render stores read only
/// `AnimPlayer.clip + t`, so a character in a non-entry state drew its rest pose.
/// [`crate::pose::step_pose_evaluation`] is what makes the sentence true — the
/// same fixed step evaluates the machine's pose and publishes it, and both
/// projectors prefer it over the `AnimPlayer`.
///
/// ## Persistence
///
/// **Persisted since scene v5**, like [`SkeletalMesh`] / [`AnimPlayer`]:
/// `EntityRecord` carries an `anim_state_machine` slot and the cook closes
/// `level → AnimStateMachine.sm`. (This block used to claim the opposite; it was
/// written at v4 and never revisited.) The [`runtime`](Self::runtime) field is
/// `#[serde(skip)]` + `#[reflect(ignore)]` — rebuilt each play session, never
/// persisted, like a physics solver's transient state — and so is the pose it
/// drives, which lives in [`crate::pose`]'s resource and reaches no file at all.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct AnimStateMachine {
    /// The `.inf_sm` state-machine asset GUID; `None` → the bind pose.
    /// `#[reflect(ignore)]` + serde-persisted (assigned by drag), like
    /// [`SkeletalMesh::mesh`].
    #[serde(default)]
    #[reflect(ignore)]
    pub sm: Option<Uuid>,
    /// Whether transition conditions + blend-space params read the actor's
    /// Blueprint variables (v1 is always `true`; the field is present so a future
    /// explicit param-binding table is an additive change).
    #[serde(default = "default_params_from_vars")]
    pub params_from_vars: bool,
    /// Live runtime state — transient, never serialized or reflected.
    #[serde(skip)]
    #[reflect(ignore)]
    pub runtime: SmRuntimeState,
}

impl Default for AnimStateMachine {
    fn default() -> Self {
        Self {
            sm: None,
            params_from_vars: default_params_from_vars(),
            runtime: SmRuntimeState::default(),
        }
    }
}

// ── P11.3 character tools: root motion + attachments ────────────────────────
//
// These are **live-session-only** components (like [`SkeletalMesh`]/[`AnimPlayer`],
// the same v5 `.inf_lvl` persistence gap): they derive serde for completeness but
// are not yet slots in the `EntityRecord`. They are deliberately **not reflected /
// not registered** as editable Details components — they are authored by dedicated
// P11.3 tooling (a root-motion toggle, a socket-attach action), a documented
// follow-up — so this batch touches neither `registry.rs` nor its editable-count.

/// How an entity consumes its [`AnimPlayer`] clip's **root motion** (P11.3).
///
/// A plain (non-reflected) enum: the root-motion mode is a gameplay toggle, not a
/// Details-grid scalar. `ApplyToEntity` moves the *entity* by the clip's root-joint
/// ground displacement each fixed step (through the 3D character mover when the
/// entity is a [`CharacterController3D`], else a raw transform add).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RootMotionMode {
    /// The clip's root motion stays in the pose (in-place animation); the entity
    /// does not move. The default.
    #[default]
    None,
    /// Extract the root-joint XZ translation + yaw each step and drive the entity's
    /// `Transform` with it.
    ApplyToEntity,
}

/// Marks an entity as **root-motion driven** (P11.3). Kept a small, orthogonal
/// component (rather than a field on [`AnimPlayer`]) so animation-graph work can
/// evolve pose driving independently. The sim tick reads this + the entity's
/// [`AnimPlayer`] and applies [`inf_anim::root_delta`](../../inf_anim/index.html)
/// once per fixed step.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct RootMotion {
    /// How the entity consumes its clip's root motion.
    #[serde(default)]
    pub mode: RootMotionMode,
}

impl RootMotion {
    /// A component that drives the entity from its clip's root motion.
    pub fn apply() -> Self {
        Self {
            mode: RootMotionMode::ApplyToEntity,
        }
    }
}

/// Makes this entity **follow another entity's socket** (P11.3 attachments).
///
/// The attachment system ([`crate::attach::update_attachments`]) writes this
/// entity's world `Transform` = `target.GlobalTransform · offset` each fixed step
/// (post-anim-tick), so a weapon rides the hand, a hat rides the head, etc.
///
/// `target` is the followed entity's stable [`Guid`]; **socket** names the
/// authored skeleton socket to ride, and an EMPTY name means the target's origin.
///
/// **Pose-driven since P24.1**: the follow composes `target.GlobalTransform ·
/// socket_model · offset`, where `socket_model` is the socket's transform under
/// the pose the sim evaluated for the target this fixed step
/// ([`crate::pose::EvaluatedPose`]). Before that the socket name was recorded and
/// never read, so a sword attached to `hand_r` rode the pelvis. A target with no
/// evaluated pose — or a socket its skeleton does not author — keeps the origin
/// follow, which is the right answer for an unbound rig and not an error.
///
/// Not reflected (it carries a `Guid` link, like [`ActorClass`]), and **there is
/// no attach tool yet** — this doc claimed one from P11.3 on. An attachment is
/// authored by code today (`AttachedTo::new`) and persisted through
/// `EntityRecord::attached_to`; a viewport action that bakes one from a picked
/// socket is a P24.3 deliverable ("sockets integration").
#[derive(Component, Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AttachedTo {
    /// The followed entity's stable identity.
    pub target: Uuid,
    /// The socket name on the target's skeleton this offset was baked from
    /// (informational in v1; empty = attach to the target's origin).
    #[serde(default)]
    pub socket: String,
    /// Local offset from the socket/target frame: translation (metres).
    #[serde(default)]
    pub offset_translation: Vec3d,
    /// Local offset rotation (euler degrees, YXZ — the [`Transform`] convention).
    #[serde(default)]
    pub offset_rotation: Vec3d,
}

impl AttachedTo {
    /// Attach to `target`'s `socket` with a pure translation offset.
    pub fn new(target: Uuid, socket: impl Into<String>, offset_translation: Vec3d) -> Self {
        Self {
            target,
            socket: socket.into(),
            offset_translation,
            offset_rotation: Vec3d::ZERO,
        }
    }

    /// The offset as a local affine (matches [`Transform::affine`] conventions).
    pub fn offset_affine(&self) -> DAffine3 {
        let r = self.offset_rotation.to_dvec3();
        let q = DQuat::from_euler(
            glam::EulerRot::YXZ,
            r.y.to_radians(),
            r.x.to_radians(),
            r.z.to_radians(),
        );
        DAffine3::from_rotation_translation(q, self.offset_translation.to_dvec3())
    }
}

// ── Audio (P12.3) ───────────────────────────────────────────────────────────

/// How an [`AudioSource`]'s loudness falls off with distance from the listener.
/// Mirrors `inf_audio::AttenuationModel` 1:1 (the sim translates this to the
/// engine model); kept local so the ECS crate does not depend on the audio crate.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DistanceModel {
    /// Linear ramp from full volume at `min_distance` to silence at `max_distance`.
    #[default]
    Linear,
    /// Inverse-distance falloff (`min_distance / distance`).
    Inverse,
    /// Exponential falloff (`(min_distance / distance)^rolloff`), steeper than
    /// inverse for `rolloff > 1`.
    Exponential,
}

/// A spatialized sound emitter (P12.3). Playback is **output-only** and lives
/// outside the deterministic world: the sim reads this component to emit audio
/// commands (autoplay on the first tick, or Blueprint `audio.*` nodes), which are
/// drained host-side into the long-lived `inf_audio::AudioEngine`. Nothing here is
/// a device handle — only the authoring intent.
///
/// Additive component: every field carries `#[serde(default)]`.
///
/// ## Persistence
///
/// **Persisted since scene v6.** `EntityRecord` carries `audio_source` and
/// `audio_listener` slots (`inf_scene`), and the v5 downgrade path strips
/// exactly those two. This block used to say the opposite — "not yet an
/// `EntityRecord` slot … no schema bump is made here" — describing the state of
/// the tree at the moment it was written, one batch before the v6 bump that
/// closed it, and it was still here at v20. The same sentence, in the same
/// words, was stale on [`SkeletalMesh`] for fifteen schema versions (lens 5
/// F15, Hardening Wave G): *a "not yet" in a doc comment has no expiry and
/// nothing checks it*, which is why the claim now names the version it became
/// true at rather than the one it was false at.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct AudioSource {
    /// The `.inf_audio` clip GUID to play; `None` → silent. `#[reflect(ignore)]`
    /// + serde-persisted (assigned by drag), like [`AnimPlayer::clip`].
    #[serde(default)]
    #[reflect(ignore)]
    pub clip: Option<Uuid>,
    /// Named mixer bus (see `inf_audio::MixerConfig`). Defaults to `"sfx"`.
    #[serde(default = "default_audio_bus")]
    pub bus: String,
    /// Base linear volume before bus/master/spatial/occlusion scaling.
    #[serde(default = "default_audio_volume")]
    pub volume: f64,
    /// Playback-rate factor (`1.0` = normal pitch).
    #[serde(default = "default_audio_pitch")]
    pub pitch: f64,
    /// Loop the whole clip (`true`) or play once (`false`).
    #[serde(default)]
    pub looping: bool,
    /// Spatialize (`true`) using the emitter's world transform, or play 2D
    /// (`false`).
    #[serde(default = "default_audio_spatial")]
    pub spatial: bool,
    /// At/within this distance the emitter is at full volume.
    #[serde(default = "default_audio_min_distance")]
    pub min_distance: f64,
    /// At/beyond this distance the emitter is silent.
    #[serde(default = "default_audio_max_distance")]
    pub max_distance: f64,
    /// Distance-attenuation curve.
    #[serde(default = "default_audio_distance_model")]
    pub distance_model: DistanceModel,
    /// Falloff exponent for [`DistanceModel::Exponential`] (ignored otherwise).
    #[serde(default = "default_audio_rolloff")]
    pub rolloff: f64,
    /// Apply a physics-raycast occlusion cut when the listener's line of sight to
    /// this emitter is obstructed (the sim's audio step does one 3D ray).
    #[serde(default)]
    pub occlusion: bool,
    /// Start playing automatically on the first tick after BeginPlay.
    #[serde(default)]
    pub autoplay: bool,
}

fn default_audio_bus() -> String {
    "sfx".to_string()
}
fn default_audio_volume() -> f64 {
    1.0
}
fn default_audio_pitch() -> f64 {
    1.0
}
fn default_audio_spatial() -> bool {
    true
}
fn default_audio_min_distance() -> f64 {
    1.0
}
fn default_audio_max_distance() -> f64 {
    100.0
}
fn default_audio_rolloff() -> f64 {
    1.0
}
fn default_audio_distance_model() -> DistanceModel {
    DistanceModel::Inverse
}

impl Default for AudioSource {
    fn default() -> Self {
        Self {
            clip: None,
            bus: default_audio_bus(),
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            spatial: true,
            min_distance: 1.0,
            max_distance: 100.0,
            distance_model: DistanceModel::Inverse,
            rolloff: 1.0,
            occlusion: false,
            autoplay: false,
        }
    }
}

/// The active spatial-audio listener (P12.3). The sim picks the **first active**
/// listener each tick to place the audio listener pose; with none active it falls
/// back to the editor/play camera pose (documented in the sim audio step). Only
/// one should be active at a time.
///
/// Additive component; shares the pinned v6 persistence gap with [`AudioSource`].
#[derive(
    Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default,
)]
#[reflect(Component, Default)]
pub struct AudioListener {
    /// Whether this entity is the active listener.
    #[serde(default)]
    pub active: bool,
}

// ── P-R0 world-decoration components (`.inf_lvl` schema v8) ──────────────────

/// A projected decal (schema v8): a box-projected overlay (bullet holes, blood,
/// signage) that stamps `color` onto surfaces within its oriented `size` box.
/// Registered + reflected so the Details grid surfaces it; spawnable later (no
/// `SpawnKind` yet).
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Decal {
    /// Oriented projection-box extent in world units (centered on the entity's
    /// [`Transform`]).
    #[serde(default = "default_decal_size")]
    pub size: Vec3d,
    /// Linear tint multiplied into the projected surface.
    #[serde(default)]
    pub color: Color,
    /// Overall opacity `[0, 1]`.
    #[serde(default = "default_decal_opacity")]
    pub opacity: f32,
    /// Surfaces whose normal deviates from the projection axis by more than this
    /// (degrees) fade out — kills stretching on grazing faces.
    #[serde(default = "default_decal_fade_angle")]
    pub fade_angle_deg: f32,
}

fn default_decal_size() -> Vec3d {
    Vec3d::ONE
}
fn default_decal_opacity() -> f32 {
    1.0
}
fn default_decal_fade_angle() -> f32 {
    60.0
}

impl Default for Decal {
    fn default() -> Self {
        Self {
            size: default_decal_size(),
            color: Color::WHITE,
            opacity: default_decal_opacity(),
            fade_angle_deg: default_decal_fade_angle(),
        }
    }
}

/// The behaviour of a [`Volume`]: a flat reflected enum (Details dropdown, like
/// [`LightKind`]).
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VolumeKind {
    /// Detects overlaps (fires Blueprint events) but never blocks movement.
    #[default]
    Trigger,
    /// A solid, movement-blocking region.
    Blocking,
}

/// A rectangular gameplay volume (schema v8): a trigger or blocking region sized
/// by the entity's [`Transform`] scale. Registered + reflected.
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Volume {
    /// Trigger vs Blocking.
    #[serde(default)]
    pub kind: VolumeKind,
    /// Editor gizmo tint (linear RGBA).
    #[serde(default)]
    pub tint: Color,
}

impl Default for Volume {
    fn default() -> Self {
        Self {
            kind: VolumeKind::Trigger,
            tint: Color::WHITE,
        }
    }
}

/// How a [`Spline`]'s control points are interpolated: a flat reflected enum
/// (Details dropdown).
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SplineInterp {
    /// Straight segments between points.
    Linear,
    /// Catmull-Rom smooth interpolation through the points.
    #[default]
    CatmullRom,
}

/// A control-point spline (schema v8): a path for camera rails, patrol routes,
/// or procedural placement. `points` are entity-local (metres). Registered +
/// reflected.
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Spline {
    /// Control points in the entity's local frame (metres).
    #[serde(default = "default_spline_points")]
    pub points: Vec<Vec3d>,
    /// Whether the last point connects back to the first (a loop).
    #[serde(default)]
    pub closed: bool,
    /// Linear vs Catmull-Rom interpolation.
    #[serde(default)]
    pub interp: SplineInterp,
}

fn default_spline_points() -> Vec<Vec3d> {
    vec![Vec3d::ZERO, Vec3d::new(0.0, 0.0, 5.0)]
}

impl Default for Spline {
    fn default() -> Self {
        Self {
            points: default_spline_points(),
            closed: false,
            interp: SplineInterp::CatmullRom,
        }
    }
}

/// One palette entry of a [`Foliage`] component (schema v8): the primitive kind
/// and tint an instance of that `kind` draws with. A nested reflected struct
/// (like [`AtlasRect`]).
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Default)]
pub struct FoliagePaletteEntry {
    /// The primitive drawn for instances referencing this palette slot.
    pub primitive: Primitive,
    /// Linear tint for this palette slot.
    pub tint: Color,
}

impl Default for FoliagePaletteEntry {
    fn default() -> Self {
        Self {
            primitive: Primitive::Cube,
            tint: Color::WHITE,
        }
    }
}

/// One scattered [`Foliage`] instance (schema v8). Rotation is stored as **euler
/// degrees** (YXZ), matching the [`Transform`] house convention — no serde quat
/// type is introduced. Serde-persisted but never reflected (the instance list is
/// too large for the Details grid; see [`Foliage::instances`]).
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct FoliageInstance {
    /// Local position (metres).
    pub position: Vec3d,
    /// Local rotation as euler degrees (YXZ — the [`Transform`] convention).
    pub rotation: Vec3d,
    /// Uniform scale.
    pub scale: f64,
    /// Palette index (into [`Foliage::palette`]) selecting the mesh/tint.
    pub kind: u32,
}

impl Default for FoliageInstance {
    fn default() -> Self {
        Self {
            position: Vec3d::ZERO,
            rotation: Vec3d::ZERO,
            scale: 1.0,
            kind: 0,
        }
    }
}

/// A foliage scatter (schema v8): a small palette of primitive kinds and a bulk
/// list of placed instances (grass, rocks, trees). Registered + reflected — but
/// only the `palette` is surfaced in Details; `instances` is `#[reflect(ignore)]`
/// (too large for the grid, authored by a scatter tool) yet still serde-persisted.
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[reflect(Component, Default)]
pub struct Foliage {
    /// The palette of primitive kinds instances draw from.
    #[serde(default)]
    pub palette: Vec<FoliagePaletteEntry>,
    /// The placed instances. `#[reflect(ignore)]` (bulk data, not a Details
    /// scalar) + serde-persisted.
    #[serde(default)]
    #[reflect(ignore)]
    pub instances: Vec<FoliageInstance>,
}

// ── P16.5 world-partition components (`.inf_lvl` schema v10) ─────────────────

/// Default [`StreamingSource::radius_m`] — a 512 m personal radius, two default
/// 256 m cells in every direction.
pub fn default_streaming_source_radius() -> f64 {
    512.0
}

/// A **streaming source** (schema v10): an entity whose position drives world
/// -partition cell residency.
///
/// # This is a SIM component, deliberately
///
/// Cell streaming spawns and despawns entities, so it changes what the fixed step
/// can see — which means residency has to be a function of *sim state alone*, or
/// the replay / PIE-==-shipping gates die. That is why the want set is derived
/// from entities carrying this component (the possessed character, a scripted
/// convoy, a spectator pawn) and **never** from the free camera: a camera is a
/// render concern, and letting it decide which entities exist would be exactly
/// the coupling the terrain-streaming doctrine forbids.
///
/// The editor viewport's free camera *is* a legitimate source while authoring —
/// there is no simulation there to corrupt — but the editor stays single-document
/// in v1, so nothing streams in it at all.
///
/// A world with no streaming source wants no cells: only the persistent cell is
/// resident, which is the correct (and deterministic) answer.
///
/// `radius_m` **overrides** the level's
/// `PartitionSettings::activation_radius_m` for this source when it is larger, so
/// a long-range observer (a flying vehicle, a sniper camera pawn) can pull more
/// world in without changing the level default for everyone.
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct StreamingSource {
    /// Activation radius around this entity, in **metres**. Cells whose footprint
    /// comes within this distance are activated at the next deterministic sync
    /// point. Clamped against the level's `activation_radius_m` (the larger wins).
    #[serde(default = "default_streaming_source_radius")]
    pub radius_m: f64,
}

impl Default for StreamingSource {
    fn default() -> Self {
        Self {
            radius_m: default_streaming_source_radius(),
        }
    }
}

/// **Always loaded** (schema v10): a marker excluding this entity from grid
/// partitioning — it is cooked into the partition's *persistent* cell and exists
/// for the whole run, wherever the streaming sources are.
///
/// This is what a level's managers, global volumes, the sun, the boot camera and
/// anything gameplay assumes is always spawned carry. It is a marker (no fields):
/// the residency answer for such an entity is not a parameter, it is "yes".
///
/// Note the partitioner *also* routes entities with **no meaningful world
/// position** to the persistent cell without this marker — see
/// `inf_scene::partition::is_persistent`. The marker is the explicit, authored
/// override for an entity that *does* have a position but must never stream.
#[derive(
    Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default,
)]
#[reflect(Component, Default)]
pub struct AlwaysLoaded;

// ── Time of day & sky (P17.1, schema v11) ───────────────────────────────────

/// **Time of day** (schema v11): the world clock the sun and moon are a pure
/// function of. Retires the compile-time `SUN_DIR` constant the renderer shipped
/// from Phase 2 to Phase 16.
///
/// One entity in a level carries this — the *sky authority*, resolved
/// deterministically by [`crate::sky::sky_authority`] (lowest `Guid` wins) so the
/// editor viewport and the shipped player agree even though they walk the world
/// in different orders. It pairs with [`SkyAtmosphere`] on the same entity.
///
/// Units follow architecture rule 6 (SI, documented): **seconds** for the clock,
/// **degrees** for the angles (the Details/UI boundary convention), and a
/// dimensionless multiplier for [`rate`](Self::rate).
///
/// The clock advances only inside the fixed simulation step (`rate × dt`,
/// wrapping the day) — never while merely authoring, so an idle editor never
/// dirties the document. `rate` defaults to **0** (frozen): a level opts into a
/// moving sun explicitly.
///
/// Every field is `#[serde(default)]`, and the defaults place the sun within
/// **1.6°** of the retired `SUN_DIR` — a scene that adds this component keeps
/// essentially the look it had.
///
/// Blueprint-drivable through the `sky.*` host namespace, and
/// `TimeOfDay.seconds` is a valid sequencer property-track path.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct TimeOfDay {
    /// UTC seconds since midnight, `[0, 86400)`. Local solar time is
    /// `seconds + longitude_deg / 15 h`, so 12:00 at longitude 0 is solar noon on
    /// the prime meridian. Values outside the range wrap.
    #[serde(default = "default_tod_seconds")]
    pub seconds: f64,
    /// Day of the year, `1..=365`. The engine's year is fixed-length: no leap
    /// day, no year field (a game clock, not an ephemeris). Day 172 ≈ the June
    /// solstice.
    #[serde(default = "default_tod_day_of_year")]
    pub day_of_year: u32,
    /// Geodetic latitude in **degrees**, positive north (`[-90, 90]`).
    #[serde(default = "default_tod_latitude_deg")]
    pub latitude_deg: f64,
    /// Longitude in **degrees**, positive east (`[-180, 180)`).
    #[serde(default)]
    pub longitude_deg: f64,
    /// Simulated seconds of clock per real second of simulation — dimensionless.
    /// `0` freezes the clock (the default); `60` makes a day pass in 24 minutes;
    /// negative runs time backwards.
    #[serde(default)]
    pub rate: f64,
}

fn default_tod_seconds() -> f64 {
    36_000.0 // 10:00 UTC
}
fn default_tod_day_of_year() -> u32 {
    172 // ≈ the June solstice
}
fn default_tod_latitude_deg() -> f64 {
    48.9
}

impl Default for TimeOfDay {
    fn default() -> Self {
        Self {
            seconds: default_tod_seconds(),
            day_of_year: default_tod_day_of_year(),
            latitude_deg: default_tod_latitude_deg(),
            longitude_deg: 0.0,
            rate: 0.0,
        }
    }
}

impl TimeOfDay {
    /// The solar inputs this clock represents — the bridge to
    /// [`inf_math::solar`], which owns every bit of the actual astronomy.
    pub fn solar_input(&self) -> inf_math::solar::SolarInput {
        inf_math::solar::SolarInput {
            seconds: self.seconds,
            day_of_year: self.day_of_year,
            latitude_deg: self.latitude_deg,
            longitude_deg: self.longitude_deg,
        }
    }

    /// Advance the clock by `dt` real seconds at this component's `rate`,
    /// wrapping the day. Pure IEEE add/mul/floor — bit-identical everywhere, so
    /// the replay and PIE-vs-shipping traces are portable.
    pub fn advance(&mut self, dt: f64) {
        let (seconds, day) =
            inf_math::solar::advance(self.seconds, self.day_of_year, self.rate, dt);
        self.seconds = seconds;
        self.day_of_year = day;
    }
}

/// A named **weather state** (schema v14 · P17.4).
///
/// Five presets, each a coherent set of values for the whole weather block —
/// cloud coverage and type, wind, fog density and precipitation — rather than
/// five independent sliders an author has to keep consistent by hand. Blending
/// between two of them is what "the weather changed" means.
///
/// The variants are the *targets* a level or a Blueprint names; the live values
/// are [`SkyAtmosphere`]'s `weather_*` fields, which the fixed step walks toward
/// the target (see [`crate::sky::advance_weather`]). Once a transition has
/// settled the blender never writes them again, which is precisely what lets the
/// sequencer and the Details grid own them.
///
/// The wire form is a **fieldless serde enum**: bincode writes the variant index
/// as one varint byte, so a preset costs a byte rather than the seven floats its
/// [`params`](Self::params) expand to.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WeatherPreset {
    /// A clean, dry day: a few fair-weather cumulus, a light breeze, no fog, no
    /// precipitation. The default, and what `weather_*`'s own defaults spell.
    #[default]
    Clear,
    /// A solid grey deck, a stronger wind and a touch of haze — still dry.
    Overcast,
    /// Full coverage, a hard wind and heavy rain.
    Storm,
    /// A thick ground-level fog layer under a half-covered sky, almost no wind.
    /// Visibility ≈ 500 m (Koschmieder `3/σ`).
    Fog,
    /// Heavy snowfall under a near-solid deck. The one preset whose
    /// [`WeatherParams::snowiness`] is 1, so it is the one that drives
    /// [`crate::sky::ResolvedSky::snow_accumulation_rate`] — the P22 hook.
    Snow,
}

/// The seven blendable numbers a [`WeatherPreset`] expands to — the *whole*
/// weather state, and exactly the fields [`SkyAtmosphere`]'s `weather_*` block
/// stores live.
///
/// Units per architecture rule 6: wind in **m/s**, fog extinction in **m⁻¹**
/// (SI), the rest dimensionless `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeatherParams {
    /// Fractional cloud coverage `[0, 1]`, driving [`SkyAtmosphere::cloud_coverage`].
    pub coverage: f32,
    /// Cloud type `[0, 1]` (0 = stratus sheet, 1 = towering cumulus).
    pub cloud_type: f32,
    /// Wind velocity in world **X**, m/s. Drifts the clouds *and* slants the
    /// precipitation.
    pub wind_x: f32,
    /// Wind velocity in world **Z**, m/s.
    pub wind_z: f32,
    /// Height-fog extinction, **m⁻¹**. Visibility ≈ `3 / density` metres.
    pub fog_density: f32,
    /// Precipitation intensity `[0, 1]`. `0` ⇒ the precipitation pass draws
    /// nothing at all.
    pub precipitation: f32,
    /// How frozen the precipitation is, `[0, 1]`: `0` = rain (fast, streaked),
    /// `1` = snow (slow, round). Blendable, so a Storm → Snow transition is a
    /// continuous change of phase rather than a swap.
    pub snowiness: f32,
}

impl WeatherPreset {
    /// Every preset, in menu order — the one place the list is enumerated, so a
    /// new state cannot be added to the enum and forgotten by the editor.
    pub const ALL: [WeatherPreset; 5] = [
        WeatherPreset::Clear,
        WeatherPreset::Overcast,
        WeatherPreset::Storm,
        WeatherPreset::Fog,
        WeatherPreset::Snow,
    ];

    /// The preset's values. Pure, `const`-shaped and total — the single table
    /// both the blend target and the editor's preset buttons read.
    pub fn params(self) -> WeatherParams {
        match self {
            // A few fair-weather cumulus. Coverage well under the cloud block's
            // own 0.35 default, because "Clear" has to look clear.
            WeatherPreset::Clear => WeatherParams {
                coverage: 0.08,
                cloud_type: 0.75,
                wind_x: 4.0,
                wind_z: 1.5,
                fog_density: 0.0,
                precipitation: 0.0,
                snowiness: 0.0,
            },
            // A flat grey deck with a little haze under it (~40 km visibility).
            WeatherPreset::Overcast => WeatherParams {
                coverage: 0.85,
                cloud_type: 0.2,
                wind_x: 9.0,
                wind_z: 3.5,
                fog_density: 7.5e-5,
                precipitation: 0.0,
                snowiness: 0.0,
            },
            // Solid, hard wind (≈ 24 m/s ≈ Beaufort 9), heavy rain, ~5 km
            // visibility through the downpour.
            WeatherPreset::Storm => WeatherParams {
                coverage: 1.0,
                cloud_type: 0.35,
                wind_x: 22.0,
                wind_z: 9.0,
                fog_density: 6.0e-4,
                precipitation: 1.0,
                snowiness: 0.0,
            },
            // Thick ground fog: 6e-3 m⁻¹ is a Koschmieder visibility of ~500 m.
            // Half a sky above it, and almost no wind — fog and wind do not
            // coexist, which is why this preset's wind is near zero.
            WeatherPreset::Fog => WeatherParams {
                coverage: 0.5,
                cloud_type: 0.1,
                wind_x: 1.5,
                wind_z: 0.5,
                fog_density: 6.0e-3,
                precipitation: 0.0,
                snowiness: 0.0,
            },
            // Heavy snow under a near-solid deck; the flakes themselves cut
            // visibility to a couple of kilometres.
            WeatherPreset::Snow => WeatherParams {
                coverage: 0.9,
                cloud_type: 0.3,
                wind_x: 5.0,
                wind_z: 2.0,
                fog_density: 1.2e-3,
                precipitation: 0.7,
                snowiness: 1.0,
            },
        }
    }

    /// The lowercase identifier a Blueprint / the sequencer / a saved preset
    /// names this state by (`"clear"`, `"overcast"`, `"storm"`, `"fog"`,
    /// `"snow"`).
    pub fn as_str(self) -> &'static str {
        match self {
            WeatherPreset::Clear => "clear",
            WeatherPreset::Overcast => "overcast",
            WeatherPreset::Storm => "storm",
            WeatherPreset::Fog => "fog",
            WeatherPreset::Snow => "snow",
        }
    }

    /// The **reflect variant name** (`"Clear"`, `"Storm"`, …) — the Rust
    /// spelling `bevy_reflect`'s `DynamicEnum` matches on, which is what the
    /// editor's `edit_set_prop` writes. Deliberately distinct from
    /// [`as_str`](Self::as_str), which is the lowercase *wire* spelling a
    /// Blueprint and the DTO use; conflating the two writes a variant that
    /// silently fails to apply.
    pub fn variant_name(self) -> &'static str {
        match self {
            WeatherPreset::Clear => "Clear",
            WeatherPreset::Overcast => "Overcast",
            WeatherPreset::Storm => "Storm",
            WeatherPreset::Fog => "Fog",
            WeatherPreset::Snow => "Snow",
        }
    }

    /// Parse the identifier [`as_str`](Self::as_str) writes, case- and
    /// whitespace-insensitively. `None` for anything else — a Blueprint that
    /// typos a preset name must be a **no-op**, not a silently different sky.
    pub fn parse(name: &str) -> Option<Self> {
        let n = name.trim().to_ascii_lowercase();
        WeatherPreset::ALL.into_iter().find(|p| p.as_str() == n)
    }
}

/// **Sky atmosphere** (schema v11; grown in v12, v13 and v14): how the
/// [`TimeOfDay`] sun and moon light the world, the physically-based sky that is
/// drawn behind it, and the artistic gradient tint layered over that.
///
/// The v11 half — [`enabled`](Self::enabled), the sun/moon colour + intensity
/// pairs, the three gradient colours and [`night_darkening`](Self::night_darkening)
/// — is unchanged. P17.2 appends the **physical-atmosphere block**: a
/// Hillaire-2020-class transmittance / sky-view LUT pair drives the sky, the sun
/// and moon discs, the starfield, aerial perspective on lit geometry, and
/// exponential height fog. P17.3 appends the **volumetric-cloud block**: a
/// raymarched layer between two authored altitudes, shaped by deterministic 3D
/// noise, drifting with the level's clock, and casting a soft large-scale shadow
/// on the world. P17.4 appends the **weather block**: one coherent, blendable
/// state — coverage, type, wind, fog and precipitation together — that *drives*
/// the cloud and fog fields above it whenever
/// [`weather_enabled`](Self::weather_enabled) is set.
///
/// # Units (architecture rule 6)
///
/// The Rayleigh / Mie / ozone coefficients themselves are **not** authored here:
/// they are physical constants of Earth's atmosphere and live once, in
/// `inf_render::atmosphere::AtmosphereParams` (kilometre altitudes, per-kilometre
/// coefficients — the atmospheric-optics convention, documented there). What a
/// level authors is a handful of dimensionless multipliers over them, plus the
/// **height fog in SI metres**: [`fog_density`](Self::fog_density) and
/// [`fog_falloff`](Self::fog_falloff) are per-metre (m⁻¹) and
/// [`fog_height`](Self::fog_height) is a world altitude in metres, because fog is
/// authored against world geometry rather than against the planet.
///
/// Lives on the same entity as [`TimeOfDay`] (the sky authority). Every field is
/// `#[serde(default)]`, so the reflection Details grid, the World Settings panel
/// and the JSON/TOML sidecar all tolerate a partial record. **That is not what
/// makes the bincode payload safe** — bincode is not self-describing, so growing
/// this struct is a wire-format change: P17.2 bumped the level schema to v12 with
/// the v11 shape frozen (`SkyAtmosphereV11` in both codecs), and P17.3 bumped it
/// to v13 the same way for the volumetric-cloud block (`SkyAtmosphereV12`).
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct SkyAtmosphere {
    /// Whether the sun/moon light the scene. When `false` the clock still runs
    /// and the sky still tints, but no directional light is projected — a level
    /// that wants to author its own suns turns this off.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Sun radiant-intensity multiplier. `3.0` matches the intensity the mesh
    /// shaders' hard-coded fallback sun used through Phase 16.
    #[serde(default = "default_sun_intensity")]
    pub sun_intensity: f32,
    /// Linear sun colour (alpha unused).
    #[serde(default = "default_sun_color")]
    pub sun_color: Color,
    /// Moon radiant-intensity multiplier, applied while the sun is below the
    /// horizon.
    #[serde(default = "default_moon_intensity")]
    pub moon_intensity: f32,
    /// Linear moon colour (alpha unused).
    #[serde(default = "default_moon_color")]
    pub moon_color: Color,
    /// Linear sky gradient colour straight overhead.
    #[serde(default = "default_sky_zenith")]
    pub zenith: Color,
    /// Linear sky gradient colour at the horizon.
    #[serde(default = "default_sky_horizon")]
    pub horizon: Color,
    /// Linear sky gradient colour below the horizon (ground haze).
    #[serde(default = "default_sky_ground")]
    pub ground: Color,
    /// How far the gradient darkens once the sun is fully below the horizon,
    /// `[0, 1]`: `0` keeps the authored colours all night, `1` fades them to
    /// black. The blend is a smoothstep over the sun's elevation, and it is
    /// **exactly 1.0 (no change) whenever the sun is more than ~9° up**, so a
    /// daytime scene renders the authored gradient unmodified.
    #[serde(default = "default_night_darkening")]
    pub night_darkening: f32,

    // ── the physical-atmosphere block (schema v12 · P17.2) ────────────────
    /// Draw the **physically-based sky** (transmittance + sky-view LUTs, sun and
    /// moon discs, stars) instead of the three-colour gradient. `true` by
    /// default: a level that has opted into a clock wants a real sky, and no
    /// pre-P17.2 content carries this component at all, so nothing that exists
    /// today changes. Turning it off restores the exact v11 gradient pass.
    #[serde(default = "default_true")]
    pub physical: bool,
    /// Linear exposure multiplier on the physical sky's radiance (dimensionless).
    /// Scales the sky *only* — the sun and moon discs and the lighting are driven
    /// by [`sun_intensity`](Self::sun_intensity) / [`moon_intensity`](Self::moon_intensity).
    #[serde(default = "default_one")]
    pub sky_intensity: f32,
    /// Aerosol (Mie) density multiplier over the clear-air reference —
    /// dimensionless "turbidity". `1` is a clean day; larger values thicken the
    /// haze band at the horizon and warm the sun's halo. Clamped to `[0, 16]`.
    #[serde(default = "default_one")]
    pub turbidity: f32,
    /// Mie phase asymmetry `g`, `[-0.95, 0.95]`. Positive = forward-scattering
    /// (the real atmosphere, ≈ 0.8) which is what produces the bright glow around
    /// the sun.
    #[serde(default = "default_mie_anisotropy")]
    pub mie_anisotropy: f32,
    /// Sun **angular diameter** in degrees as seen from the ground. The true
    /// value is ≈ 0.53°; the default rounds to 0.545° (the mean apparent
    /// diameter). Larger values give a softer, bigger sun.
    #[serde(default = "default_sun_disc_deg")]
    pub sun_disc_deg: f32,
    /// Moon angular diameter in degrees (true mean ≈ 0.52°).
    #[serde(default = "default_moon_disc_deg")]
    pub moon_disc_deg: f32,
    /// Starfield brightness multiplier (dimensionless). `0` removes the stars.
    /// The field fades in as the sun sets and is a pure function of view
    /// direction, so it is screen-stable under camera rotation.
    #[serde(default = "default_one")]
    pub star_intensity: f32,
    /// How far the physical sky is pulled back toward the authored
    /// [`zenith`](Self::zenith)/[`horizon`](Self::horizon)/[`ground`](Self::ground)
    /// gradient, `[0, 1]`. **`0` by default** — the physical result stands on its
    /// own and the gradient colours are the artistic override, not a tax on it.
    /// `1` reproduces the v11 gradient exactly.
    #[serde(default)]
    pub tint_strength: f32,
    /// Strength of **aerial perspective** on lit geometry, `[0, 4]`: how much of
    /// the atmosphere's in-scattered light is mixed into distant surfaces (and
    /// how much of their own radiance is extinguished). `1` is the physical
    /// amount; `0` disables it.
    #[serde(default = "default_one")]
    pub aerial_perspective: f32,
    /// Exponential **height-fog** extinction at [`fog_height`](Self::fog_height),
    /// in **m⁻¹** (SI). `0` — the default — means no height fog at all, and the
    /// receivers take the byte-identical no-fog path. A visibility of `d` metres
    /// is roughly `3 / d`, so `6e-4` ≈ 5 km visibility and `1.5e-4` ≈ 20 km.
    #[serde(default)]
    pub fog_density: f32,
    /// Height-fog vertical falloff in **m⁻¹**: density decays as
    /// `exp(-falloff · (y − fog_height))`. The default `0.002` is a 500 m
    /// e-folding height (a valley-floor haze layer). `0` makes the fog uniform
    /// with altitude.
    #[serde(default = "default_fog_falloff")]
    pub fog_falloff: f32,
    /// World altitude in **metres** at which the fog reaches
    /// [`fog_density`](Self::fog_density).
    #[serde(default)]
    pub fog_height: f32,
    /// Linear tint multiplied into the fog's in-scattered light. The in-scatter
    /// itself is sampled from the sky in the view direction, so white (the
    /// default) already gives fog that matches the time of day; the tint is for
    /// stylised (green, sepia, alien) air.
    #[serde(default = "default_fog_color")]
    pub fog_color: Color,

    // ── the volumetric-cloud block (schema v13 · P17.3) ───────────────────
    /// Draw **volumetric clouds**. `false` by default — unlike
    /// [`physical`](Self::physical), which every level that opted into a clock
    /// wanted — because clouds are a per-level artistic choice with a real frame
    /// cost, and because every v12 level must keep the sky it was authored
    /// against. The editor's default scene opts *in*; existing content does not.
    ///
    /// Clouds require [`physical`](Self::physical): they are lit by the sun's
    /// transmittance through the atmosphere and ambient-lit by the sky-view LUT,
    /// so a cloud over the v11 gradient would be a grey blob with nothing to light
    /// it. The renderer enforces that (`AtmosphereParams::clouds_active`).
    #[serde(default)]
    pub clouds_enabled: bool,
    /// Fractional sky **coverage**, `[0, 1]`. `0` is cloudless, `0.35` (the
    /// default) is broken cumulus with real gaps, `1` is solid overcast. It biases
    /// a procedural weather field rather than naming a literal area fraction, so
    /// the realised cover tracks it monotonically without matching it exactly.
    #[serde(default = "default_cloud_coverage")]
    pub cloud_coverage: f32,
    /// Cloud **type**, `[0, 1]`: `0` = stratus (a flat sheet along the bottom of
    /// the layer), `1` = cumulus (towering, rounded, filling the slab). Drives the
    /// vertical density profile.
    #[serde(default = "default_cloud_type")]
    pub cloud_type: f32,
    /// Bottom of the cloud layer, **metres** of world altitude (SI).
    #[serde(default = "default_cloud_bottom")]
    pub cloud_bottom: f32,
    /// Top of the cloud layer, **metres**. Held above
    /// [`cloud_bottom`](Self::cloud_bottom) by the renderer; a degenerate slab
    /// simply draws nothing.
    #[serde(default = "default_cloud_top")]
    pub cloud_top: f32,
    /// Cloud extinction at full density, **m⁻¹** (SI). Real cloud is 0.01–0.1;
    /// over a two-kilometre column even the low end is optically opaque, which is
    /// why the default sits near the bottom of the range.
    #[serde(default = "default_cloud_density")]
    pub cloud_density: f32,
    /// Strength of the high-frequency **erosion** detail, `[0, 1]`. `0` leaves
    /// smooth blobs; `1` carves the wispy edges that make a cloud read as vapour.
    /// The Low render tier ignores this (it skips the detail volume entirely).
    #[serde(default = "default_cloud_detail")]
    pub cloud_detail: f32,
    /// Field **seed**. Changing it re-rolls the whole sky; only the low 24 bits
    /// are used (the renderer carries it through an f32 uniform).
    #[serde(default)]
    pub cloud_seed: u32,
    /// Wind velocity in world **X**, **m/s** (SI). The drift is a deterministic
    /// function of the [`TimeOfDay`] clock, *not* of a wall clock — so two runs at
    /// the same time of day see the same sky.
    #[serde(default = "default_cloud_wind_x")]
    pub cloud_wind_x: f32,
    /// Wind velocity in world **Z**, **m/s**.
    #[serde(default = "default_cloud_wind_z")]
    pub cloud_wind_z: f32,
    /// Forward-scattering asymmetry of the dominant cloud phase lobe,
    /// `[0, 0.95]`. The back lobe is derived from it, so one number still buys the
    /// two-lobe phase that produces silver linings.
    #[serde(default = "default_cloud_phase_g")]
    pub cloud_phase_g: f32,
    /// How much the cloud layer darkens the **sun on the ground**, `[0, 1]`. `0`
    /// disables the cloud-shadow map entirely and the lit passes take the
    /// byte-identical no-cloud-shadow path; `1` is the physical amount.
    #[serde(default = "default_one")]
    pub cloud_shadow: f32,
    /// Multiplier on the ambient (sky) light inside a cloud, `[0, 4]`. `1` is the
    /// physical amount; raising it lifts the shaded undersides of an overcast
    /// deck, which is the usual artistic complaint about correct clouds.
    #[serde(default = "default_one")]
    pub cloud_ambient: f32,
    /// Linear albedo tint of the cloud droplets (alpha unused). White is physical
    /// — water is grey — and the tint is for stylised skies.
    #[serde(default = "default_cloud_color")]
    pub cloud_color: Color,

    // ── the weather block (schema v14 · P17.4) ────────────────────────────
    //
    // DESIGN NOTE — why the live parameters are the state, and the preset is
    // only the target.
    //
    // The alternative shape is a pure state machine: store `from`, `to` and a
    // blend fraction, and derive everything. It is smaller, and it is wrong for
    // this engine for two reasons. (1) The sequencer keys **reflected component
    // fields**; a `from/to/t` triple is not something a curve can be drawn
    // through, so weather would have needed bespoke sequencer code — and P17.1
    // bought "zero sequencer code" by making the clock a plain reflected field.
    // (2) It leaves the Details grid with nothing authorable between presets,
    // i.e. dead controls, which the batch brief rules out.
    //
    // So the live values ARE the component, and the transition is two extra
    // fields beside them. The blend is advanced in the sim fixed step, exactly
    // like `TimeOfDay::rate`, by closing `dt / remaining` of the gap each step —
    // which is *exactly* linear (after n steps the gap is scaled by
    // `(T − n·dt)/T`), reaches the target precisely as `remaining` hits 0, and
    // needs no `from` snapshot to do it. Once settled
    // (`weather_blend_remaining == 0`) the blender never writes these fields
    // again, so a sequencer track or a Details edit owns them outright.
    /// Whether the weather block **drives** the sky. When `true`,
    /// [`cloud_coverage`](Self::cloud_coverage), [`cloud_type`](Self::cloud_type),
    /// the cloud wind pair and [`fog_density`](Self::fog_density) are taken from
    /// the `weather_*` fields below and the authored ones are ignored; when
    /// `false` the authored fields stand exactly as they did in v13 and the
    /// weather block is inert (no precipitation, no blending — the fixed step
    /// skips it). **`false` by default**: every v13 level must keep the sky it
    /// was authored against, and the projection must stay byte-identical for it.
    #[serde(default)]
    pub weather_enabled: bool,
    /// The state the live `weather_*` values are blending **toward**. Once
    /// [`weather_blend_remaining`](Self::weather_blend_remaining) reaches `0`
    /// they equal it — unless something (the sequencer, a Details edit, a
    /// Blueprint writing a field directly) has since moved them, which is
    /// allowed and is why this names the *target* rather than the current state.
    #[serde(default)]
    pub weather_target: WeatherPreset,
    /// How long a `sky.set_weather` transition takes when the caller does not
    /// say, **seconds**. Also what the editor's preset buttons use.
    #[serde(default = "default_weather_blend_seconds")]
    pub weather_blend_seconds: f32,
    /// Seconds left in the transition in flight; `0` = settled, and the fixed
    /// step then touches nothing. Persisted (rather than reset on load) so a
    /// saved mid-transition level resumes exactly where it was — the same reason
    /// [`TimeOfDay::seconds`] is persisted.
    #[serde(default)]
    pub weather_blend_remaining: f32,
    /// Live cloud coverage `[0, 1]` — drives [`cloud_coverage`](Self::cloud_coverage).
    #[serde(default = "default_weather_coverage")]
    pub weather_coverage: f32,
    /// Live cloud type `[0, 1]` — drives [`cloud_type`](Self::cloud_type).
    #[serde(default = "default_weather_cloud_type")]
    pub weather_cloud_type: f32,
    /// Live wind in world **X**, m/s — drives the cloud drift *and* the angle
    /// the precipitation falls at.
    #[serde(default = "default_weather_wind_x")]
    pub weather_wind_x: f32,
    /// Live wind in world **Z**, m/s.
    #[serde(default = "default_weather_wind_z")]
    pub weather_wind_z: f32,
    /// Live height-fog extinction, **m⁻¹** (SI) — drives
    /// [`fog_density`](Self::fog_density). Visibility ≈ `3 / density` metres.
    #[serde(default)]
    pub weather_fog_density: f32,
    /// Live precipitation intensity `[0, 1]`. `0` ⇒ the precipitation pass is
    /// instruction-neutral (it dispatches no draw at all).
    #[serde(default)]
    pub weather_precipitation: f32,
    /// Live precipitation phase, `[0, 1]`: `0` = rain, `1` = snow. Blendable, so
    /// rain turning to snow is a continuous transition. Feeds
    /// [`crate::sky::ResolvedSky::snow_accumulation_rate`] — the P22 hook.
    #[serde(default)]
    pub weather_snowiness: f32,
}

fn default_true() -> bool {
    true
}
fn default_sun_intensity() -> f32 {
    3.0
}
fn default_sun_color() -> Color {
    Color::new(1.0, 0.98, 0.95, 1.0)
}
fn default_moon_intensity() -> f32 {
    0.15
}
fn default_moon_color() -> Color {
    Color::new(0.62, 0.72, 1.0, 1.0)
}
// The renderer's `SkyParams::default()` gradient, mirrored here so a default
// `SkyAtmosphere` writes back exactly the pixels the engine drew before P17.1.
fn default_sky_zenith() -> Color {
    Color::new(0.012, 0.021, 0.038, 1.0)
}
fn default_sky_horizon() -> Color {
    Color::new(0.055, 0.081, 0.120, 1.0)
}
fn default_sky_ground() -> Color {
    Color::new(0.009, 0.011, 0.015, 1.0)
}
fn default_night_darkening() -> f32 {
    0.85
}
fn default_one() -> f32 {
    1.0
}
fn default_mie_anisotropy() -> f32 {
    0.8
}
fn default_sun_disc_deg() -> f32 {
    0.545
}
fn default_moon_disc_deg() -> f32 {
    0.52
}
fn default_fog_falloff() -> f32 {
    0.002 // 500 m e-folding height
}
fn default_fog_color() -> Color {
    Color::new(1.0, 1.0, 1.0, 1.0)
}
// ── volumetric clouds (v13 · P17.3). Mirrors `inf_render::clouds::CloudParams::default`
// field for field, minus the two projected-not-authored ones (`enabled` is this
// component's `clouds_enabled`; `time_s` comes from the `TimeOfDay` clock).
fn default_cloud_coverage() -> f32 {
    0.35 // broken cumulus with real gaps — see the golden `clouds_scattered`
}
fn default_cloud_type() -> f32 {
    0.7 // cumulus-leaning stratocumulus
}
fn default_cloud_bottom() -> f32 {
    1500.0 // m — a temperate fair-weather cumulus base
}
fn default_cloud_top() -> f32 {
    4000.0 // m
}
fn default_cloud_density() -> f32 {
    0.04 // m^-1
}
fn default_cloud_detail() -> f32 {
    0.6
}
fn default_cloud_wind_x() -> f32 {
    6.0 // m/s — a light breeze; the layer crosses 8 km in ~23 minutes
}
fn default_cloud_wind_z() -> f32 {
    2.0 // m/s
}
fn default_cloud_phase_g() -> f32 {
    0.8
}
fn default_cloud_color() -> Color {
    Color::new(1.0, 1.0, 1.0, 1.0)
}
// ── weather (v14 · P17.4). The live values default to the **Clear** preset, so
// a default component is a *settled* Clear state rather than an arbitrary one —
// enabling weather is then a single boolean, exactly like enabling clouds.
fn default_weather_blend_seconds() -> f32 {
    8.0 // s — long enough to read as weather changing, short enough to test
}
fn default_weather_coverage() -> f32 {
    WeatherPreset::Clear.params().coverage
}
fn default_weather_cloud_type() -> f32 {
    WeatherPreset::Clear.params().cloud_type
}
fn default_weather_wind_x() -> f32 {
    WeatherPreset::Clear.params().wind_x
}
fn default_weather_wind_z() -> f32 {
    WeatherPreset::Clear.params().wind_z
}

impl SkyAtmosphere {
    /// The live weather values as a [`WeatherParams`] — the shape the blend and
    /// the preset table both speak.
    #[inline]
    pub fn weather_params(&self) -> WeatherParams {
        WeatherParams {
            coverage: self.weather_coverage,
            cloud_type: self.weather_cloud_type,
            wind_x: self.weather_wind_x,
            wind_z: self.weather_wind_z,
            fog_density: self.weather_fog_density,
            precipitation: self.weather_precipitation,
            snowiness: self.weather_snowiness,
        }
    }

    /// Write the live weather values back. Used by the blender, by the editor's
    /// preset buttons and by `sky.set_weather` when it is told to snap.
    #[inline]
    pub fn set_weather_params(&mut self, p: WeatherParams) {
        self.weather_coverage = p.coverage;
        self.weather_cloud_type = p.cloud_type;
        self.weather_wind_x = p.wind_x;
        self.weather_wind_z = p.wind_z;
        self.weather_fog_density = p.fog_density;
        self.weather_precipitation = p.precipitation;
        self.weather_snowiness = p.snowiness;
    }
}

impl Default for SkyAtmosphere {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            sun_intensity: default_sun_intensity(),
            sun_color: default_sun_color(),
            moon_intensity: default_moon_intensity(),
            moon_color: default_moon_color(),
            zenith: default_sky_zenith(),
            horizon: default_sky_horizon(),
            ground: default_sky_ground(),
            night_darkening: default_night_darkening(),
            physical: default_true(),
            sky_intensity: default_one(),
            turbidity: default_one(),
            mie_anisotropy: default_mie_anisotropy(),
            sun_disc_deg: default_sun_disc_deg(),
            moon_disc_deg: default_moon_disc_deg(),
            star_intensity: default_one(),
            tint_strength: 0.0,
            aerial_perspective: default_one(),
            fog_density: 0.0,
            fog_falloff: default_fog_falloff(),
            fog_height: 0.0,
            fog_color: default_fog_color(),
            clouds_enabled: false,
            cloud_coverage: default_cloud_coverage(),
            cloud_type: default_cloud_type(),
            cloud_bottom: default_cloud_bottom(),
            cloud_top: default_cloud_top(),
            cloud_density: default_cloud_density(),
            cloud_detail: default_cloud_detail(),
            cloud_seed: 0,
            cloud_wind_x: default_cloud_wind_x(),
            cloud_wind_z: default_cloud_wind_z(),
            cloud_phase_g: default_cloud_phase_g(),
            cloud_shadow: default_one(),
            cloud_ambient: default_one(),
            cloud_color: default_cloud_color(),
            weather_enabled: false,
            weather_target: WeatherPreset::Clear,
            weather_blend_seconds: default_weather_blend_seconds(),
            weather_blend_remaining: 0.0,
            weather_coverage: default_weather_coverage(),
            weather_cloud_type: default_weather_cloud_type(),
            weather_wind_x: default_weather_wind_x(),
            weather_wind_z: default_weather_wind_z(),
            weather_fog_density: WeatherPreset::Clear.params().fog_density,
            weather_precipitation: WeatherPreset::Clear.params().precipitation,
            weather_snowiness: WeatherPreset::Clear.params().snowiness,
        }
    }
}

// ── water (schema v17 · P20.1) ──────────────────────────────────────────────

/// Which kind of water body a [`WaterBody`] describes.
///
/// A **flat reflected enum** with per-kind fields carried flat on the component,
/// exactly like [`LightKind`] (whose Directional / Point / Spot share one
/// `Light`) and [`VolumeKind`]. A data-carrying enum would read more tidily in
/// Rust and would be the wrong shape here for two concrete reasons: the Details
/// reflection walker surfaces flat scalars and enum *dropdowns*, so struct
/// variants would have no widget at all; and a variant-dependent field set makes
/// the bincode record's length depend on the variant, which is a worse wire
/// contract than one record with `serde(default)` fields.
///
/// **THE ORDERING LAW (P19.2): new variants go at the end, always.** bincode
/// encodes this as its declaration index, so inserting a kind in the middle
/// silently renumbers the ones after it and every committed level's oceans become
/// lakes. Pinned by `water_kind_discriminants_are_frozen`.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WaterKind {
    /// An unbounded sea at [`WaterBody::level_m`], carrying wind-driven Gerstner
    /// waves. The default: it is the body that needs the fewest other components
    /// to mean something.
    #[default]
    Ocean,
    /// A bounded rectangle in world XZ — [`WaterBody::extent`] either side of the
    /// entity's transform — at [`WaterBody::level_m`], carrying a gentle ripple.
    Lake,
    /// A ribbon following the [`Spline`] **on the same entity**, with width and
    /// depth profiles and a flow speed. An entity with `WaterKind::River` and no
    /// `Spline` has no centreline and draws nothing.
    River,
}

impl WaterKind {
    /// Stable identifier for logs, advisories and tooling. Never localized and
    /// never reordered — the string half of the discriminant freeze.
    pub fn as_str(self) -> &'static str {
        match self {
            WaterKind::Ocean => "ocean",
            WaterKind::Lake => "lake",
            WaterKind::River => "river",
        }
    }
}

/// **A water body** (schema v17, P20.1): an ocean, a lake or a spline river,
/// with its wave state, its shading, and — for a river — its cross-section.
///
/// ## One component, three bodies
///
/// The three kinds differ in their *footprint* and in the frame their waves are
/// evaluated in, not in the wave model: an ocean and a lake evaluate a Gerstner
/// sum in world XZ, a river evaluates one in its own `(arc length, offset)`
/// frame so the ripple travels downstream. That is why there is one component
/// here, one [`inf_water::WaveField`] behind it and one water pass in the
/// renderer — and why a "lake" is just a bounded ocean with a small amplitude.
///
/// ## The river's spline is a **same-entity component**, not a reference
///
/// A river reads the [`Spline`] on its own entity. That is deliberately not an
/// asset reference and not an entity reference: composition on one entity is how
/// [`Terrain`] and [`Transform`] already relate, it cannot dangle, it needs no
/// cook edge and no dangling-reference advisory, and it makes "select the river,
/// drag its points" the obvious authoring gesture. The one cost is that a river
/// cannot share a centreline with a road; if that is ever wanted, an
/// `EntityRef` field is the additive change, and it would be the thing that
/// introduced a reference to keep alive.
///
/// ## Units
///
/// SI, per the units doctrine: metres, seconds, m/s, m⁻¹. Angles are **degrees**
/// at this boundary (the Details grid's convention, like `Light::inner_cone_deg`)
/// and radians everywhere below it.
///
/// Additive component: every field carries `#[serde(default)]`, so a hand-written
/// or partial record still decodes — though the schema bump is forced regardless,
/// because bincode is positional (the v12/v13/v15/v16 law).
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct WaterBody {
    /// Ocean, lake or river.
    #[serde(default)]
    pub kind: WaterKind,
    /// Still-water surface elevation in **metres of world Y** — an absolute
    /// altitude, not an offset from the entity (a sea level that moved when you
    /// nudged the entity would be a trap). Ignored by [`WaterKind::River`], whose
    /// surface follows its spline.
    #[serde(default)]
    pub level_m: f64,
    /// Half-extent of a [`WaterKind::Lake`]'s region in world XZ, **metres**,
    /// centred on the entity's transform — the same convention
    /// [`PcgVolume::extent`] uses.
    #[serde(default = "default_water_extent")]
    pub extent: Vec2d,

    // ── waves ─────────────────────────────────────────────────────────────
    /// Peak surface displacement from the still level, **metres**. A *bound*:
    /// the derived component amplitudes sum to exactly this (times the wind
    /// gain), so `|height − level| ≤ amplitude_m` always. `0` gives a mirror.
    #[serde(default = "default_wave_amplitude")]
    pub wave_amplitude_m: f64,
    /// Wavelength of the longest component, **metres**. Shorter components follow
    /// on a geometric ladder.
    #[serde(default = "default_wave_length")]
    pub wave_length_m: f64,
    /// Gerstner steepness, `[0, 1]`. `0` is a sine surface; `1` is the physical
    /// limit at which a crest cusps. Above ~0.8 the horizontal crowding is very
    /// visible, which is the look "stormy" wants.
    #[serde(default = "default_wave_steepness")]
    pub wave_steepness: f64,
    /// Number of Gerstner components, clamped to `1..=8`.
    #[serde(default = "default_wave_count")]
    pub wave_count: u32,
    /// Half-angle of the directional spread about the wind, **degrees**. `0` is a
    /// perfectly unidirectional (and obviously artificial) sea.
    #[serde(default = "default_wave_spread_deg")]
    pub wave_spread_deg: f64,
    /// Seed for the per-component hash. Two bodies with identical settings and
    /// different seeds carry different — but each internally deterministic — seas.
    #[serde(default)]
    pub wave_seed: u32,
    /// Whether the waves follow the **level's weather wind**
    /// (`ResolvedSky::weather()`), which is what makes a storm raise the sea.
    /// When `false` the body uses [`wind_x`](Self::wind_x) /
    /// [`wind_z`](Self::wind_z) instead, so a sheltered inlet can stay calm
    /// through a gale.
    #[serde(default = "default_wind_from_weather")]
    pub wind_from_weather: bool,
    /// Body-local wind in world **+X**, m/s — used when
    /// [`wind_from_weather`](Self::wind_from_weather) is `false`.
    #[serde(default = "default_water_wind_x")]
    pub wind_x: f64,
    /// Body-local wind in world **+Z**, m/s.
    #[serde(default)]
    pub wind_z: f64,

    // ── river profile ─────────────────────────────────────────────────────
    /// Full width at the **start** of a river's spline, metres.
    #[serde(default = "default_river_width")]
    pub river_width_start_m: f64,
    /// Full width at the **end**, metres. Linear in arc length between the two.
    #[serde(default = "default_river_width")]
    pub river_width_end_m: f64,
    /// Depth from surface to bed at the start, metres. Drives the absorption tint
    /// and the shallow-water foam band.
    #[serde(default = "default_river_depth")]
    pub river_depth_start_m: f64,
    /// Depth at the end, metres.
    #[serde(default = "default_river_depth")]
    pub river_depth_end_m: f64,
    /// Surface flow speed along the spline, **m/s**. Negative reverses the river
    /// without re-authoring its points.
    #[serde(default = "default_river_flow")]
    pub river_flow_m_s: f64,

    // ── shading ───────────────────────────────────────────────────────────
    /// Linear colour of **shallow** water — what the surface tends toward where
    /// the column under it is thin.
    #[serde(default = "default_shallow_color")]
    pub shallow_color: Color,
    /// Linear colour of **deep** water — the asymptote the absorption drives
    /// toward as the column thickens.
    #[serde(default = "default_deep_color")]
    pub deep_color: Color,
    /// Per-channel **Beer-Lambert extinction** of the water column, in **m⁻¹**
    /// (SI) — the same shape as [`SkyAtmosphere::fog_density`]. Red is absorbed
    /// roughly an order of magnitude faster than blue in real water, which is why
    /// the default is so lopsided and why deep water is blue without anyone
    /// painting it blue.
    #[serde(default = "default_absorption")]
    pub absorption: Vec3d,
    /// Surface roughness for the specular lobe, `[0, 1]`. Water is very smooth;
    /// the default is a calm-sea sheen rather than a mirror, because a true
    /// mirror shows every flaw in the reflection source.
    #[serde(default = "default_water_roughness")]
    pub roughness: f64,
    /// Screen-space **refraction** offset at grazing incidence, metres of
    /// apparent displacement at the water plane. `0` disables refraction (the
    /// background is sampled straight through).
    #[serde(default = "default_refraction")]
    pub refraction_m: f64,
    /// Depth of the **shore fade** band, metres: the surface's opacity ramps from
    /// zero where the ground meets the water to full at this depth. The CPU twin
    /// is `inf_water::shore_blend`, and both use the same smoothstep.
    #[serde(default = "default_shore_fade")]
    pub shore_fade_m: f64,
    /// Maximum surface opacity, `[0, 1]` — what the water reaches once it is
    /// deeper than the shore band. Below `1` the geometry behind shows through
    /// even in deep water, which stylised water sometimes wants.
    #[serde(default = "default_one_f64")]
    pub opacity: f64,

    // ── foam ──────────────────────────────────────────────────────────────
    /// Linear colour of foam.
    #[serde(default = "default_foam_color")]
    pub foam_color: Color,
    /// Crest factor above which wave foam appears, `[0, 1]`. The crest factor is
    /// the surface-folding measure the Gerstner model already contains, so this
    /// is "how close to breaking before it goes white", not a taste dial with no
    /// referent. `1` disables crest foam.
    #[serde(default = "default_foam_crest")]
    pub foam_crest_threshold: f64,
    /// Width of the **shoreline** foam band, metres of water depth. Foam fills the
    /// band from the waterline down to this depth. `0` disables shore foam.
    #[serde(default = "default_foam_shore")]
    pub foam_shore_m: f64,
    /// Flow speed, m/s, at which a river is fully foamed. Rapids go white;
    /// a slow river does not. `0` disables flow foam.
    #[serde(default = "default_foam_flow")]
    pub foam_flow_m_s: f64,
}

fn default_water_extent() -> Vec2d {
    Vec2d::splat(50.0)
}
fn default_wave_amplitude() -> f64 {
    0.6
}
fn default_wave_length() -> f64 {
    40.0
}
fn default_wave_steepness() -> f64 {
    0.5
}
fn default_wave_count() -> u32 {
    4
}
fn default_wave_spread_deg() -> f64 {
    45.0
}
fn default_wind_from_weather() -> bool {
    true
}
fn default_water_wind_x() -> f64 {
    6.0
}
fn default_river_width() -> f64 {
    8.0
}
fn default_river_depth() -> f64 {
    1.5
}
fn default_river_flow() -> f64 {
    1.5
}
fn default_shallow_color() -> Color {
    Color::new(0.20, 0.48, 0.50, 1.0)
}
fn default_deep_color() -> Color {
    Color::new(0.015, 0.075, 0.13, 1.0)
}
/// Extinction of clear natural water, m⁻¹ — red goes first. Rounded from the
/// standard clear-ocean absorption spectrum sampled at 620/540/460 nm; the point
/// is the *ratio*, which is what makes a 10 m column blue-green and a 40 m one
/// nearly black.
fn default_absorption() -> Vec3d {
    Vec3d::new(0.45, 0.09, 0.035)
}
fn default_water_roughness() -> f64 {
    0.04
}
fn default_refraction() -> f64 {
    0.35
}
fn default_shore_fade() -> f64 {
    1.2
}
fn default_one_f64() -> f64 {
    1.0
}
fn default_foam_color() -> Color {
    Color::new(0.92, 0.95, 0.97, 1.0)
}
fn default_foam_crest() -> f64 {
    0.65
}
fn default_foam_shore() -> f64 {
    0.5
}
fn default_foam_flow() -> f64 {
    4.0
}

impl Default for WaterBody {
    fn default() -> Self {
        Self {
            kind: WaterKind::Ocean,
            level_m: 0.0,
            extent: default_water_extent(),
            wave_amplitude_m: default_wave_amplitude(),
            wave_length_m: default_wave_length(),
            wave_steepness: default_wave_steepness(),
            wave_count: default_wave_count(),
            wave_spread_deg: default_wave_spread_deg(),
            wave_seed: 0,
            wind_from_weather: default_wind_from_weather(),
            wind_x: default_water_wind_x(),
            wind_z: 0.0,
            river_width_start_m: default_river_width(),
            river_width_end_m: default_river_width(),
            river_depth_start_m: default_river_depth(),
            river_depth_end_m: default_river_depth(),
            river_flow_m_s: default_river_flow(),
            shallow_color: default_shallow_color(),
            deep_color: default_deep_color(),
            absorption: default_absorption(),
            roughness: default_water_roughness(),
            refraction_m: default_refraction(),
            shore_fade_m: default_shore_fade(),
            opacity: default_one_f64(),
            foam_color: default_foam_color(),
            foam_crest_threshold: default_foam_crest(),
            foam_shore_m: default_foam_shore(),
            foam_flow_m_s: default_foam_flow(),
        }
    }
}

impl WaterBody {
    /// A lake preset: bounded, still, with a gentle ripple.
    pub fn lake(level_m: f64, extent: Vec2d) -> Self {
        Self {
            kind: WaterKind::Lake,
            level_m,
            extent,
            // A lake is a bounded ocean with small numbers — see the type docs.
            wave_amplitude_m: 0.05,
            wave_length_m: 7.0,
            wave_steepness: 0.12,
            wave_count: 3,
            // A lake has fetch measured in hundreds of metres, not hundreds of
            // kilometres, so a storm does not raise a swell on it. Decoupling it
            // from the weather wind is what keeps a lake a lake in a gale.
            wind_from_weather: false,
            wind_x: 1.0,
            ..Self::default()
        }
    }

    /// A river preset: reads the [`Spline`] on the same entity.
    pub fn river(width_m: f64, depth_m: f64, flow_m_s: f64) -> Self {
        Self {
            kind: WaterKind::River,
            river_width_start_m: width_m,
            river_width_end_m: width_m,
            river_depth_start_m: depth_m,
            river_depth_end_m: depth_m,
            river_flow_m_s: flow_m_s,
            wave_amplitude_m: 0.06,
            wave_length_m: 4.0,
            wave_steepness: 0.12,
            wave_count: 3,
            wind_from_weather: false,
            // A river's ripple travels downstream, so its "wind" is `+arc length`
            // in the body's own frame — never a world direction.
            wind_x: 1.0,
            wind_z: 0.0,
            ..Self::default()
        }
    }

    /// The wind this body's waves respond to, m/s in world XZ, given the level's
    /// resolved weather wind.
    ///
    /// Defined **here, once**, because both scene projectors need it and a
    /// per-host copy of "does this body follow the weather" is exactly the
    /// divergence the MIRROR gate exists to stop — the same reasoning that put
    /// `ResolvedSky::cloud_time_s` in Ring 0.
    pub fn effective_wind(&self, weather_wind: (f64, f64)) -> (f64, f64) {
        if self.wind_from_weather {
            weather_wind
        } else {
            (self.wind_x, self.wind_z)
        }
    }
}

// ── buoyancy (schema v18 · P20.2) ───────────────────────────────────────────

/// **Buoyancy + hydrodynamic drag** (schema v18, P20.2): the opt-in marker that
/// makes a dynamic 3D body *float* when a [`WaterBody`]'s surface is over it.
///
/// ## Why this is opt-in rather than on for every `RigidBody3D`
///
/// The alternative — buoyancy on by default, derived from the collider's
/// existing `density` — was considered and rejected for two concrete reasons.
///
/// 1. **It changes what committed levels mean.** Nothing in a pre-v18 `.inf_lvl`
///    says "this crate floats"; under a default-on rule, adding a lake to an
///    existing level silently rewrites the physics of every dynamic body in it,
///    and a replay recorded before the lake would diverge. Opt-in makes the
///    change visible in the file that caused it.
/// 2. **[`Collider3D::density`] defaults to `1.0`, which is not a material
///    density.** It is rapier's placeholder, and it has never mattered, because a
///    rigid body's fall is mass-independent. Buoyancy is the first system that
///    reads it as physics: at 1 kg/m³ against water's 1000, *every* default body
///    would float like a cork on a millimetre of draught. A default-on rule would
///    therefore be wrong for essentially all existing content and the fix would be
///    "go author a density on every collider in your level".
///
/// The same trap is why flotation reads [`density_kg_m3`](Self::density_kg_m3)
/// here rather than the collider's: this field is the body's density *for
/// flotation* and defaults to seasoned wood, so a body that opts in floats
/// sensibly the moment the component is added. The collider's `density` keeps
/// doing its own job — it is what rapier turns into the body's **mass and
/// inertia**, i.e. how *hard* the body is to move, while this is how *high* it
/// rides. When the two agree the model is exactly Archimedes; the equilibrium
/// submerged fraction is always `density_kg_m3 / fluid_density_kg_m3`.
///
/// ## The model, stated
///
/// Buoyant force is `submerged_fraction × displaced_volume × ρ_fluid × g`, taken
/// against gravity — with the **displaced volume read from rapier's own exact
/// per-shape mass properties** (`mass / density_kg_m3`) rather than from a second,
/// hand-written volume table that could drift from the one the solver uses. Drag
/// is **linear** in the velocity relative to the water's flow (still water for an
/// ocean or lake, the river's tangent flow for a river) and scaled by the
/// submerged fraction: an honest v1 that cannot express a hull's shape-dependent
/// terminal speed, which quadratic drag would. See `inf_physics::d3::water`.
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Buoyancy {
    /// Whether this body floats at all. `false` keeps the component (and its
    /// tuning) on the entity while the water ignores it — the same "authored but
    /// off" affordance `SkyAtmosphere::enabled` gives the sky.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// The body's density **for flotation**, kg/m³. The equilibrium submerged
    /// fraction is exactly this over [`fluid_density_kg_m3`](Self::fluid_density_kg_m3),
    /// so `500` against fresh water floats half-submerged and `2000` sinks.
    /// Defaults to seasoned wood (600 kg/m³) so an opted-in body floats with its
    /// deck clear rather than awash.
    #[serde(default = "default_body_density")]
    pub density_kg_m3: f64,
    /// Density of the water, kg/m³. Fresh water is 1000; sea water is ~1025, and
    /// authoring that is the difference between a hull that rides a little higher
    /// at sea and one that does not.
    #[serde(default = "default_fluid_density")]
    pub fluid_density_kg_m3: f64,
    /// Linear hydrodynamic drag, **s⁻¹** — the same units and the same meaning as
    /// [`RigidBody3D::linear_damping`], so "2" means the body's speed *relative to
    /// the water* decays with a ~0.35 s half-life while fully submerged. Scaled by
    /// the submerged fraction, so a body lifted clear of the water stops paying it.
    #[serde(default = "default_water_linear_drag")]
    pub linear_drag: f64,
    /// Angular hydrodynamic drag, **s⁻¹** — what stops a floating crate spinning
    /// forever after a wave rolls it. Also scaled by the submerged fraction.
    #[serde(default = "default_water_angular_drag")]
    pub angular_drag: f64,
}

/// Seasoned wood — the density of something that obviously floats, chosen so the
/// component works the moment it is added. See the type docs on why this is not
/// read from the collider.
fn default_body_density() -> f64 {
    600.0
}
/// Fresh water at 4 °C, the SI reference. Sea water is ~1025.
fn default_fluid_density() -> f64 {
    1000.0
}
fn default_water_linear_drag() -> f64 {
    2.0
}
fn default_water_angular_drag() -> f64 {
    1.5
}

impl Default for Buoyancy {
    fn default() -> Self {
        Self {
            enabled: true,
            density_kg_m3: default_body_density(),
            fluid_density_kg_m3: default_fluid_density(),
            linear_drag: default_water_linear_drag(),
            angular_drag: default_water_angular_drag(),
        }
    }
}

impl Buoyancy {
    /// A body of the given flotation density in fresh water, everything else at
    /// its default. `Buoyancy::of_density(500.0)` floats half-submerged.
    pub fn of_density(density_kg_m3: f64) -> Self {
        Self {
            density_kg_m3,
            ..Self::default()
        }
    }

    /// The submerged fraction this body settles at in still water, `[0, 1]` —
    /// `density / fluid_density`, saturating at 1 for anything denser than the
    /// fluid (which sinks rather than settling at all).
    ///
    /// Defined here rather than in the physics pass because it is the *contract*:
    /// the statics tests assert against this number, and a pass that computed a
    /// different equilibrium would be failing this function, not its own.
    pub fn equilibrium_fraction(&self) -> f64 {
        if self.fluid_density_kg_m3 <= 0.0 {
            return 0.0;
        }
        (self.density_kg_m3 / self.fluid_density_kg_m3).clamp(0.0, 1.0)
    }
}

// ── volumetric terrain (schema v19 · P21.1) ─────────────────────────────────

/// **A sparse SDF voxel volume** (schema v19, P21.1): the component that binds an
/// entity to a `.inf_voxel` asset — the caves, tunnels and excavations that
/// *locally extend* the heightfield terrain.
///
/// ## What it is, and what it deliberately is not
///
/// The planet-scale base stays a **heightfield** (the P16 clipmap economics are
/// unbeatable at that scale). Volumetric capability arrives as chunk volumes that
/// override and extend it *locally*, which is the hybrid every serious open-world
/// engine uses. Nothing here voxelizes the world.
///
/// The component is a **reference plus its two authored knobs**, exactly like
/// [`Terrain`]'s asset half: the chunks themselves live in the `.inf_voxel`
/// ([`inf_voxel::VoxelAsset`]), which is a streaming-class container paged out of
/// an mmap. There is no inline `data` field, and that asymmetry with [`Terrain`]
/// is deliberate — `Terrain` carries one because it predates streaming and an
/// inline heightfield is still a legitimate authoring mode; a voxel volume has
/// never had a pre-streaming form to keep loading.
///
/// ## The fields are FROZEN
///
/// bincode is positional, so growing this component is a wire-format change that
/// costs a scene-schema bump in **both** codec mirrors (the law paid for at v12,
/// v13, v15 and v16). v19 is Phase 21's only bump, so these three fields are what
/// the whole phase gets: the asset, the world scale of one voxel, and the
/// runtime-carve gate P21.4 reads. Anything later either fits in the `.inf_voxel`
/// (which versions itself) or buys its own bump.
///
/// Additive component: every field carries `#[serde(default)]`.
///
/// [`inf_voxel::VoxelAsset`]: the `.inf_voxel` container in the `inf-voxel` crate.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct VoxelVolume {
    /// GUID of the `.inf_voxel` asset holding this volume's chunks, or `None` for
    /// a volume that has been placed but not yet given content (it draws and
    /// collides as nothing, which is what "no chunks" means).
    ///
    /// `#[reflect(ignore)]` — an asset reference, picked through the asset UI
    /// rather than typed into the Details grid, exactly like [`MeshRef::asset`]
    /// and [`Terrain::asset`]. Still serde-persisted, and it is the edge the cook
    /// follows to pack the `.inf_voxel` with its level.
    ///
    /// **A `Uuid`, not an `inf_asset::AssetId`**, because `inf-ecs` does not
    /// depend on `inf-asset` — every asset reference in this file is a bare
    /// `Uuid` for that reason, and a fourth spelling of "asset id" in the ECS
    /// would be the drift, not the fix.
    #[serde(default)]
    #[reflect(ignore)]
    pub asset: Option<Uuid>,
    /// World size of one voxel cell edge, **metres** (SI, architecture rule 6 —
    /// 1 world unit = 1 metre, no scale factors).
    ///
    /// Defaults to `0.5` m: fine enough that a carved tunnel mouth reads as a
    /// tunnel rather than as a staircase, coarse enough that a chunk
    /// (`inf_voxel::CHUNK_DIM`³ = 16³ samples) spans 8 m and a cave system costs
    /// tens of chunks rather than thousands.
    ///
    /// This is the *authored* scale and must match the `voxel_size_m` recorded in
    /// the `.inf_voxel` header; the asset's value is the authority for anything
    /// already on disk, and a mismatch is what a P21.2 cook advisory will report.
    #[serde(default = "default_voxel_size_m")]
    pub voxel_size_m: f64,
    /// Whether **gameplay** may carve this volume at runtime (P21.4's Blueprint
    /// carve nodes read exactly this flag before touching a chunk).
    ///
    /// Defaults to `true` — a volume you placed to be dug is the common case, and
    /// the flag exists so an author can *withhold* permission from geometry that
    /// must stay as built (a level's load-bearing tunnel, a rooftop a player must
    /// not dig through). It is a **gate**, not a hint: with it `false` a runtime
    /// carve is refused and reported, never silently applied, so a replay cannot
    /// diverge on whether some node happened to run.
    ///
    /// It says nothing about *editor* carving, which is P21.3 and always allowed —
    /// an author changing the world is not the same act as a game changing it.
    #[serde(default = "default_true")]
    pub runtime_carve: bool,
}

/// Half a metre per voxel — see [`VoxelVolume::voxel_size_m`].
fn default_voxel_size_m() -> f64 {
    0.5
}

impl Default for VoxelVolume {
    fn default() -> Self {
        Self {
            asset: None,
            voxel_size_m: default_voxel_size_m(),
            runtime_carve: true,
        }
    }
}

impl VoxelVolume {
    /// A volume bound to `asset` at the default half-metre voxel scale.
    pub fn from_asset(asset: Uuid) -> Self {
        Self {
            asset: Some(asset),
            ..Self::default()
        }
    }

    /// The voxel scale to actually use, **clamped to a positive finite length**.
    ///
    /// A zero or negative `voxel_size_m` would collapse every chunk to a point
    /// (and divide by zero in world↔grid conversion); a non-finite one would put
    /// NaN into world positions. Both are reachable from the Details grid, so the
    /// door is closed here rather than in each of the several consumers — the same
    /// discipline `TerrainData::new` applies to its own spacing.
    pub fn effective_voxel_size_m(&self) -> f64 {
        if self.voxel_size_m.is_finite() && self.voxel_size_m > 0.0 {
            self.voxel_size_m
        } else {
            default_voxel_size_m()
        }
    }
}

// ── destruction (schema v20 · P22.2) ────────────────────────────────────────

/// **Destructible** (schema v20, P22.2): the marker that says an entity's mesh
/// can break, and the five numbers that decide *how*.
///
/// ## What it references, and what it deliberately does not
///
/// Nothing. It names no asset. The thing that breaks is the mesh already on this
/// entity ([`MeshRef`]), and the chunk set is **derived from that mesh at cook
/// time** — a `.inf_fracture` whose GUID is a pure function of the mesh's
/// (`inf_mesh::fracture::derived_fracture_id`), exactly as a `.inf_vmesh`'s is.
/// So there is no fracture reference to author, no fracture reference to leave
/// dangling, and no new edge in the cook's dependency closure: the
/// `MeshRef.asset` edge that already pulls the mesh in is the only one needed.
/// An entity with a `Destructible` and no `MeshRef` has nothing to break, which
/// is a cook advisory rather than a field.
///
/// ## The fields are FROZEN as shipped
///
/// bincode is positional, so growing this component is a wire-format change
/// costing a scene-schema bump in **both** codec mirrors (the law paid for at
/// v12, v13, v15 and v16). v20 is Phase 22's only bump — P22.3 (runtime
/// destruction) and P22.4 (debris at scale) must both fit inside these five
/// fields, which is why all five are decided now rather than as each batch
/// discovers it wants one. The argument that they suffice — and the list of
/// things that look like missing fields but are not — is
/// `docs/memos/p22-strength.md`. In short: seed and count are everything the
/// *cook* needs, strength is everything the *structural solve* needs, density is
/// everything the *chunk bodies* need, and the gate is the gate.
///
/// Additive component: every field carries `#[serde(default)]`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Destructible {
    /// Authored seed **offset** for the fracture, folded in beside the mesh's own
    /// GUID. Changing it re-shatters the same mesh a different way; leaving it
    /// alone means two cooks of the same content produce the same pieces, which
    /// is what makes a destruction replay reproducible.
    ///
    /// Unitless by nature — it is an identifier, not a measurement.
    #[serde(default)]
    pub fracture_seed: u32,
    /// Target number of chunks the cook fractures the mesh into.
    ///
    /// Clamped by the cook to `inf_mesh::fracture::{MIN,MAX}_CHUNK_COUNT` with an
    /// advisory naming the clamp, and the produced count can still be lower when
    /// a site's Voronoi cell misses the solid. Defaults to
    /// [`DEFAULT_DESTRUCTIBLE_CHUNKS`] — pinned equal to the mesh crate's own
    /// default by a cross-crate test, because `inf-ecs` must not depend on
    /// `inf-mesh` to say the number.
    #[serde(default = "default_destructible_chunks")]
    pub chunk_count: u32,
    /// The stress this material bears before a bond between two chunks fails,
    /// **pascals (N/m², SI — architecture rule 6)**.
    ///
    /// It is a *failure stress*, not hit points and not a hardness rating: P22.3
    /// compares a force spread over a shared chunk face against
    /// `strength × face_area`, so the units cancel into newtons and the number
    /// means something a materials table can be checked against. The unit
    /// doctrine forbids a unitless "durability" precisely because it makes that
    /// comparison unwritable.
    ///
    /// Rough classes (full derivation and the caveats in
    /// `docs/memos/p22-strength.md`): plaster ~1e6, masonry and unreinforced
    /// concrete ~2–4e6, glass ~5e6, wood across the grain ~5e6, reinforced
    /// concrete ~1e7, steel ~4e8. Defaults to [`DEFAULT_STRENGTH_PA`], the
    /// masonry/concrete class — a wall an explosion breaks and a footstep does
    /// not.
    #[serde(default = "default_strength_pa")]
    pub strength: f64,
    /// Material density, **kg/m³** — with each chunk's cook-computed volume this
    /// is its mass.
    ///
    /// **This is the honest one.** [`Collider3D::density`] also exists and also
    /// defaults to a number, but that number is `1.0`, which is rapier's mass
    /// placeholder and not a material density (the finding P20.2's buoyancy
    /// component was built around). A chunk of concrete weighing one kilogram per
    /// cubic metre would drift like ash. So chunk mass comes from here.
    ///
    /// Defaults to [`DEFAULT_DESTRUCTIBLE_DENSITY`] (concrete). Classes: pine
    /// ~500, oak ~750, brick ~1900, concrete ~2400, granite ~2700, steel ~7850.
    #[serde(default = "default_destructible_density")]
    pub density_kg_m3: f64,
    /// Whether **gameplay** may destroy this at runtime. P22.3's damage nodes
    /// read exactly this flag before swapping the intact mesh for chunk bodies.
    ///
    /// Defaults to `true` — an asset you marked destructible is one you meant to
    /// break — and exists so an author can *withhold* permission from geometry
    /// that must stay as built (a level's load-bearing wall, the bridge the
    /// critical path crosses). It is a **gate**, not a hint: with it `false` a
    /// runtime destruction is refused and reported, never silently applied, so a
    /// replay cannot diverge on whether some node happened to run. The
    /// `VoxelVolume::runtime_carve` precedent, one phase on.
    ///
    /// It says nothing about the cook, which fractures the mesh either way: a
    /// scripted or cinematic break of a `false` asset is still P22.3's to allow,
    /// and re-deriving the chunks at that point would not be possible.
    #[serde(default = "default_true")]
    pub runtime_destruct: bool,
}

/// Twelve pieces — see `inf_mesh::fracture::DEFAULT_CHUNK_COUNT`, which this is
/// pinned equal to by `inf-packager`'s `destructible_defaults_match_the_mesh_crate`
/// (the crate that can see both).
pub const DEFAULT_DESTRUCTIBLE_CHUNKS: u32 = 12;
/// 5 MPa — the masonry / unreinforced-concrete class. See
/// `docs/memos/p22-strength.md`.
pub const DEFAULT_STRENGTH_PA: f64 = 5.0e6;
/// 2400 kg/m³ — concrete. Heavy enough that debris reads as rubble rather than
/// as polystyrene the first time an author adds the component.
pub const DEFAULT_DESTRUCTIBLE_DENSITY: f64 = 2400.0;

/// **How far a bond opens before it is broken, metres** (P22.3).
///
/// A bond holds `strength × area` newtons ([`Destructible::bond_force_n`]);
/// work is force times distance; so the **energy** to break it is
/// `strength × area × CRACK_OPENING_M` joules, and `Pa · m² · m = N · m = J`
/// with no invented conversion anywhere. The number itself is the brittle
/// strain range (`1e-4 … 1e-3`) over a chunk about a metre across.
///
/// **Here, beside `Destructible`, and no longer in `inf-physics`** (island wave
/// I6). It is the *contract*, exactly as `bond_force_n`'s own doc says of the
/// force half — and I6 needed a second caller: a door's **lock is one bond**,
/// and a lock priced by a second expression would be a second answer to "what
/// does this material cost to break". `inf_physics::d3::CRACK_OPENING_M` is now
/// a re-export of this, so every path that already named it still resolves.
pub const CRACK_OPENING_M: f64 = 1.0e-3;

fn default_destructible_chunks() -> u32 {
    DEFAULT_DESTRUCTIBLE_CHUNKS
}
fn default_strength_pa() -> f64 {
    DEFAULT_STRENGTH_PA
}
fn default_destructible_density() -> f64 {
    DEFAULT_DESTRUCTIBLE_DENSITY
}

impl Default for Destructible {
    fn default() -> Self {
        Self {
            fracture_seed: 0,
            chunk_count: DEFAULT_DESTRUCTIBLE_CHUNKS,
            strength: DEFAULT_STRENGTH_PA,
            density_kg_m3: DEFAULT_DESTRUCTIBLE_DENSITY,
            runtime_destruct: true,
        }
    }
}

impl Destructible {
    /// The force, **newtons**, needed to break a bond across a shared chunk face
    /// of `area_m2` — `strength × area`, the one place the pascal is turned into
    /// something a solver compares against.
    ///
    /// Defined here rather than in P22.3 because it is the *contract*: a solve
    /// that computed a different threshold would be failing this function, not
    /// its own. (The `Buoyancy::equilibrium_fraction` precedent.)
    ///
    /// Non-positive or non-finite inputs give `0.0` — a bond that cannot hold,
    /// which is the safe reading of "this face has no area".
    pub fn bond_force_n(&self, area_m2: f64) -> f64 {
        if !self.strength.is_finite()
            || !area_m2.is_finite()
            || self.strength <= 0.0
            || area_m2 <= 0.0
        {
            return 0.0;
        }
        self.strength * area_m2
    }

    /// **The energy, joules, to break one bond of `area_m2`** —
    /// `strength × area × `[`CRACK_OPENING_M`], the P22 rule as one expression
    /// (island wave I6).
    ///
    /// It was two: `bond_energies` and `ground_bond_energies` each spelled the
    /// multiplication out, which was survivable while destruction was the only
    /// consumer and stopped being so the moment a **door lock** became a bond.
    /// A lock is one bond of a small steel area — see `inf_ecs::door` — and the
    /// kick, the crash and the fracture solve all price a break here.
    ///
    /// Inherits `bond_force_n`'s refusal: a bond with no area cannot hold, so
    /// it costs `0.0` to break.
    pub fn bond_energy_j(&self, area_m2: f64) -> f64 {
        self.bond_force_n(area_m2) * CRACK_OPENING_M
    }

    /// The mass, **kg**, of a chunk of the given volume.
    pub fn chunk_mass_kg(&self, volume_m3: f64) -> f64 {
        if !self.density_kg_m3.is_finite() || !volume_m3.is_finite() {
            return 0.0;
        }
        (self.density_kg_m3 * volume_m3).max(0.0)
    }
}

// ── v21 (P24.3) character components ────────────────────────────────────────

/// **One authored IK goal**: which joints, where their tip must go, and how much
/// of the solve to apply.
///
/// The persisted twin of [`crate::pose::IkGoal`], and deliberately *not* the same
/// type. Three differences, each load-bearing:
///
/// * **World space, not model space.** `IkGoal` is model space, because that is
///   the frame a pose is evaluated in. An author places a foot target with a
///   gizmo, in the world, and a character that walks forward must not drag its
///   goal along with it. [`crate::pose::step_pose_evaluation`] converts once, per
///   step, through the entity's own global transform.
/// * **`f64` positions.** Architecture rule 3: an authored world position is a
///   world position, and `IkGoal`'s `f32` is a render-adjacent value derived from
///   it after the rebase into the character's own frame.
/// * **A target *entity*.** A goal that follows a moving hand-hold is the common
///   case and cannot be expressed by a constant.
///
/// **Frozen once shipped** — `EntityRecord` is positional bincode, so growing
/// this struct costs another bump in *both* codec mirrors.
#[derive(Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[reflect(Default)]
pub struct IkGoalRecord {
    /// Joint indices into the bound skeleton, from the chain's **root** to its
    /// **tip**, each the parent of the next. Fewer than two joints, or a list
    /// that is not a parent walk, is refused by `inf_anim::solve_chain` — as a
    /// value, reported through [`crate::pose::ik_outcomes`], never as a panic.
    #[serde(default)]
    pub chain: Vec<u16>,
    /// Follow this entity: the goal is `target` **offset from** its world
    /// position. Unbound ⇒ `target` is an absolute world position.
    ///
    /// An [`EntityRef`] (E-P1) rather than a bare
    /// `Option<Uuid>`, so the Details panel surfaces the entity-picker widget it
    /// already has instead of `#[reflect(ignore)]`-ing the one field an author is
    /// most likely to want to set by clicking. `#[serde(transparent)]` on that
    /// wrapper makes the bytes identical to `Option<Uuid>`.
    ///
    /// A GUID that resolves to nothing is treated as unbound for the step rather
    /// than disabling the goal, so a target deleted mid-session leaves the chain
    /// reaching for the last thing it was told rather than snapping to rest.
    #[serde(default)]
    pub target_entity: crate::refs::EntityRef,
    /// Where the tip must go — **world metres**, or the offset from
    /// `target_entity` when that is set.
    #[serde(default)]
    pub target: Vec3d,
    /// A point the middle of the chain bends toward — a knee's forward, an
    /// elbow's back — in **world metres**. `None` keeps the pose's existing bend,
    /// which is what `inf_anim::two_bone_positions` does with no opinion.
    #[serde(default)]
    pub pole: Option<Vec3d>,
    /// How much of the solve to apply, `0..=1`.
    ///
    /// Not decoration: `1.0` applies the solved rotations verbatim (and takes a
    /// path that is byte-identical to having no blend at all, so a level that
    /// never lowers the weight has exactly its pre-P24.3 trace); below 1 the
    /// chain's joints are `pslerp`ed from the pre-solve pose toward the solved
    /// one, which is how a foot plant is faded in over a few steps instead of
    /// snapping. `0.0` leaves the pose untouched and still **reports** — a
    /// disabled-by-weight goal is distinguishable from an absent one.
    #[serde(default = "default_ik_weight")]
    pub weight: f32,
    /// Off ⇒ the goal is authored, saved and not solved. The
    /// `VoxelVolume::runtime_carve` shape of gate: an author disabling a chain
    /// must not have to delete it and retype four joint indices.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_ik_weight() -> f32 {
    1.0
}

impl Default for IkGoalRecord {
    fn default() -> Self {
        Self {
            chain: Vec::new(),
            target_entity: crate::refs::EntityRef::NONE,
            target: Vec3d::ZERO,
            pole: None,
            weight: 1.0,
            enabled: true,
        }
    }
}

/// **The authored IK goals on one character** (scene v21, P24.3).
///
/// # Why a list, and why one component
///
/// A bevy component is one-per-entity and a character has four chains (two feet,
/// two hands) before anything exotic. So the component is the *list*, exactly as
/// [`crate::pose::IkTargetsRes`] keys a `Vec<IkGoal>` per entity — the two shapes
/// line up on purpose, because the fixed step's job is to turn one into the
/// other and a mismatch there would be a conversion nobody could read.
///
/// # Why this closes P24.2's ledger entry rather than extending it
///
/// P24.2 shipped IK as a **resource** with the reasoning written on
/// [`crate::pose::IkTargetsRes`]: an authored component needs a scene bump, and
/// v20 was frozen. The honest consequence it recorded — *"an IK target cannot be
/// authored or saved today"* — is what this component retires. The resource
/// stays, and stays the runtime write door
/// ([`crate::pose::set_ik_goals`], which the `ik.*` Blueprint kit calls); the
/// component is the **authored** half, re-derived from the document every fixed
/// step. Both are solved, authored first, and
/// [`crate::pose::step_pose_evaluation`] is still the one door.
///
/// That layering is what makes the PIE gate possible: the component rides
/// `EntityRecord` into the `.inf_lvl` bytes a `ScenePayload` already carries, so
/// a real `--pie` subprocess engages IK through the door it always used, with no
/// test-only injection hook anywhere (the P21.4 law).
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[reflect(Component, Default)]
pub struct IkTarget {
    /// The goals, applied **in order**. An empty list is the same as no
    /// component: nothing is solved and no verdict is published.
    #[serde(default)]
    pub goals: Vec<IkGoalRecord>,
}

/// **A simulated garment on this character** (scene v21, P24.3 — **read since
/// P24.4**).
///
/// # The reference-plus-knobs shape, and why the parameters are NOT here
///
/// This is `VoxelVolume`'s law applied a phase later, and that type's own doc
/// states it: *"the component can never be the reason a future schema has to
/// move: growing it would cost another bump in **both** codec mirrors"*. So the
/// garment mesh, the XPBD constraint set, stiffness, damping, thickness, areal
/// mass, iteration count and the collision set all live inside the `.inf_cloth`
/// this points at — an asset, which versions itself, which P24.4 defines and
/// which an `AssetKind` addition reaches without touching any scene wire at all.
/// What is left on the entity is what genuinely differs **per wearer**.
///
/// # Why the slot is spent now rather than at P24.4
///
/// A phase gets one scene bump. P24.3 spends v21 on [`IkTarget`], and cloth
/// authoring one batch later would need v22 — so the choice was one bump with
/// three slots or two bumps with one and two. Both spare slots cost a
/// `None` discriminant byte per entity, the price every additive slot since v8
/// has paid, and the alternative is a forbidden second bump inside one phase.
///
/// # Its reader (P24.4)
///
/// [`crate::cloth::step_cloth_simulation`], the ONE Ring-0 rule both hosts' fixed
/// steps call. It seeds a particle set from the `.inf_cloth` this names, places
/// that garment's collision capsules on the pose the sim evaluated this step, and
/// advances an XPBD solve whose result is folded into
/// [`crate::cloth::cloth_state_bytes`] — so a coat is sim state, compared between
/// the editor's Simulate and the shipped player like everything else.
///
/// P24.3 recorded here, plainly, that *"as of P24.3 nothing simulates cloth"*.
/// That sentence is retired: the component is read.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct ClothSim {
    /// The `.inf_cloth` GUID this garment is described by, or `None` when the
    /// component is authored but not yet bound.
    ///
    /// `None` is the honest state for "I added the component and have not picked
    /// a garment", and it is what keeps the cook's dependency closure unchanged
    /// for every level written before P24.4.
    ///
    /// `#[reflect(ignore)]` + serde-persisted, exactly like [`Sprite::texture`]:
    /// an ASSET reference, which is the case `refs.rs` deliberately leaves on the
    /// bare-`Uuid` convention until an asset-picker widget lands.
    #[serde(default)]
    #[reflect(ignore)]
    pub asset: Option<Uuid>,
    /// Whether the garment simulates at runtime.
    ///
    /// Defaults to `true` — a garment you attached is one you meant to wear —
    /// and exists so a distant NPC's cloak can be pinned to the body without
    /// deleting the authoring.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Per-wearer quality lever: `0` takes the **garment's own** authored substep
    /// budget ([`inf_anim::ClothMaterial::substeps`]); `1..=255` pins one, so a
    /// hero's coat can out-simulate a crowd's.
    ///
    /// Per-**entity** and not in the asset precisely because two characters
    /// wearing the same garment can afford different budgets, which is the test
    /// for whether something belongs on the component at all.
    ///
    /// # `0` is NOT "follow the machine's capability tier" — a P24.4 correction
    ///
    /// This field was written at P24.3 saying it was. Reading it that way would
    /// put the *machine's* tier into the substep count, into the particle
    /// positions, into `cloth_state_bytes` and therefore into the trace two hosts
    /// are compared on — which is precisely the P22.4 finding (*a preview must
    /// run what it previews*: the debris budget was tier-clamped in embedded PIE
    /// and not in Simulate, and the two ran different simulations on any Medium
    /// machine). A **render** budget may follow the tier; a **sim** budget may
    /// not. Both readings of `0` are now properties of the content, so two
    /// machines fold the same coat.
    #[serde(default)]
    pub quality: u8,
}

impl Default for ClothSim {
    fn default() -> Self {
        Self {
            asset: None,
            enabled: true,
            quality: 0,
        }
    }
}

/// **Strand hair on this character** (scene v21, P24.3 — read by P24.4).
///
/// The [`ClothSim`] shape, for the [`ClothSim`] reasons: the guide curves, the
/// clump and curl parameters, the interpolation counts and the card-generation
/// recipe for lower tiers all live in the `.inf_hair` this points at, because
/// they describe the *hairstyle* and not the *head wearing it*.
///
/// # Its reader (P24.4)
///
/// [`crate::hair::step_hair_simulation`], the ONE Ring-0 rule both hosts' fixed
/// steps call: it seeds guide strands from the `.inf_hair` this names, anchors
/// each strand's root on the joint it rides in the pose the sim evaluated this
/// step, advances a per-strand XPBD chain against the same capsules a garment
/// collides with, and rebuilds the ribbons both projectors draw. The result is
/// folded into [`crate::hair::hair_state_bytes`].
///
/// P24.3 recorded here that *"as of P24.3 nothing renders or simulates hair"*.
/// That sentence is retired: the component is read, and drawn.
///
/// `quality` is read exactly as [`ClothSim::quality`] is, and **not** as the
/// machine's capability tier — see that field for the P22.4 reason.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct HairGuides {
    /// The `.inf_hair` GUID, or `None` when authored but unbound.
    /// `#[reflect(ignore)]` + serde-persisted, exactly like [`ClothSim::asset`].
    #[serde(default)]
    #[reflect(ignore)]
    pub asset: Option<Uuid>,
    /// Whether the strands simulate at runtime. `false` still draws the hair —
    /// it just does not move — which is the lower tiers' behaviour anyway.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Per-wearer quality lever, exactly [`ClothSim::quality`]: `0` takes the
    /// **hairstyle's own** authored substep budget
    /// ([`inf_anim::hair::HairMaterial::substeps`]), `1..=255` pins one.
    ///
    /// **Not the machine's capability tier**, for the reason written out at
    /// length on [`ClothSim::quality`] — a sim budget that followed the tier would
    /// put the machine into `state_bytes`. The tier reaches hair in exactly one
    /// place, and it is a *render* budget: [`inf_anim::hair::HairDetail`], which
    /// decides how many ribbons are drawn and is never folded into the trace.
    /// (This field's own doc said "follows the capability tier" when the slot was
    /// reserved at P24.3; that sentence was corrected on `ClothSim` when cloth
    /// landed and is corrected here for the same reason.)
    #[serde(default)]
    pub quality: u8,
}

impl Default for HairGuides {
    fn default() -> Self {
        Self {
            asset: None,
            enabled: true,
            quality: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The emissive intensity is the multiplier the 8-bit colour cannot be**
    /// (schema v26, wave VIS1a) — the arm the whole field exists for.
    ///
    /// `Color` is sRGB in eight bits, so `emissive` cannot exceed 1.0 in any
    /// channel and an authored emissive could touch the renderer's default bloom
    /// threshold of 1.0 without ever crossing it. What `emissive_linear` has to
    /// do is get *past* that ceiling.
    #[test]
    fn the_emissive_intensity_reaches_past_the_eight_bit_ceiling() {
        let m = Material {
            emissive: Color::new(1.0, 0.5, 0.25, 1.0),
            emissive_intensity: 12.0,
            ..Material::default()
        };
        assert_eq!(m.emissive_linear(), [12.0, 6.0, 3.0]);
        assert!(
            m.emissive_linear()[0] > 1.0,
            "the whole point: the authored colour alone cannot exceed 1.0"
        );

        // The default is an identity, so every pre-v26 material emits exactly
        // what it always did.
        let d = Material {
            emissive: Color::new(0.6, 0.4, 0.2, 1.0),
            ..Material::default()
        };
        assert_eq!(d.emissive_intensity, 1.0);
        assert_eq!(d.emissive_linear(), [0.6, 0.4, 0.2]);

        // A negative multiplier is a light that removes light; it is clamped,
        // not trusted.
        let neg = Material {
            emissive: Color::new(0.5, 0.5, 0.5, 1.0),
            emissive_intensity: -4.0,
            ..Material::default()
        };
        assert_eq!(neg.emissive_linear(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn terrain_serde_round_trips_and_defaults() {
        let mut t = Terrain::configured(5, 2.0);
        // Author a couple of tiles so the paged data is non-trivial.
        t.data.author_tile((0, 0), |x, z| (x + z) * 0.1);
        t.data.author_tile((1, 0), |x, z| (x + z) * 0.1);
        // Non-default material params.
        t.layers[1].albedo = Color::new(0.9, 0.1, 0.1, 1.0);
        t.macro_variation = 0.4;
        let json = serde_json::to_string(&t).unwrap();
        let back: Terrain = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);
        assert_eq!(back.data.tile_count(), 2);
        assert_eq!(back.tile_resolution, 5);
        assert_eq!(back.layers[1].albedo, Color::new(0.9, 0.1, 0.1, 1.0));
        assert_eq!(back.macro_variation, 0.4);

        // A minimal payload fills the scalar defaults + an empty terrain + the
        // default layer palette + default macro variation (the frozen-shape
        // additive guarantee for the v4 schema).
        let d: Terrain = serde_json::from_str("{}").unwrap();
        assert_eq!(d.tile_resolution, inf_terrain::DEFAULT_TILE_RESOLUTION);
        assert!(d.data.is_empty());
        assert_eq!(d.layers, default_terrain_layers());
        assert_eq!(d.macro_variation, default_macro_variation());

        // A pre-P10.4 payload (config + data, no layers/macro) still decodes,
        // filling the material defaults — the additive-field guarantee.
        let pre = serde_json::to_string(&serde_json::json!({
            "meters_per_sample": 1.0,
            "tile_resolution": 5,
            "data": { "tile_resolution": 5, "meters_per_sample": 1.0, "tiles": [] }
        }))
        .unwrap();
        let old: Terrain = serde_json::from_str(&pre).unwrap();
        assert_eq!(old.layers, default_terrain_layers());
        assert_eq!(old.macro_variation, default_macro_variation());
    }

    #[test]
    fn pcg_volume_serde_round_trips() {
        // The evaluated cache is `#[serde(skip)]` — a save must never carry it,
        // and a decode fills it empty (recomputed on demand). Everything else
        // round-trips. NOTE: like `Terrain`, `PcgVolume` is NOT yet a slot in the
        // v3 `.inf_lvl` `EntityRecord` — this component-level round-trip is all
        // the persistence it has until the schema-v4 migration (the same gap the
        // `terrain_is_not_persisted_yet_v4_todo` guard pins).
        let v = PcgVolume {
            graph: Some(Uuid::from_u128(0xC0FFEE)),
            extent: Vec2d::new(80.0, 40.0),
            seed: 7,
            draw_distance: 600.0,
            doorways: Vec::new(),
            evaluated: vec![ScatteredInstance {
                position: DVec3::new(1.0, 2.0, 3.0),
                rotation: DQuat::IDENTITY,
                scale: 1.5,
                kind: 2,
                mesh: None,
                extent: None,
                glow: 0.0,
            }],
            // P19.5's solid half is derived state on exactly the same terms —
            // and **this is the whole schema answer for the batch**: a field
            // that never reaches the bytes cannot force a bump in either codec
            // mirror.
            structures: vec![ScatteredSolid {
                center: DVec3::new(4.0, 5.0, 6.0),
                half_extents: DVec3::new(0.1, 1.5, 1.0),
                rotation: DQuat::IDENTITY,
            }],
            structures_gen: 9,
            // IB-2b's grouping rides on exactly the same terms — derived,
            // skipped, and therefore free of the schema window this wave has
            // already spent.
            structure_groups: vec![StructureGroup {
                shell: ScatteredSolid {
                    center: DVec3::new(4.0, 5.0, 6.0),
                    half_extents: DVec3::new(0.1, 1.5, 1.0),
                    rotation: DQuat::IDENTITY,
                },
                start: 0,
                len: 1,
                inst_start: 0,
                inst_len: 1,
            }],
            // NPC1d's population rides the same way, and **that is the whole
            // schema answer for the wave that puts a society on an island**: a
            // level's residents are derived from its own buildings, so they
            // reach no bytes and force no bump.
            residents: vec![ResidentSlot {
                role: SlotRole::Home,
                at: DVec3::new(4.0, 5.0, 6.0),
                room: 3,
                building: 1,
                floor: 2,
                index: 0,
                node: 0x3000_0000_0000_0003,
            }],
            interior_nav: {
                let mut g = inf_nav::NavGraph::new();
                g.add_node(
                    0x3000_0000_0000_0003,
                    DVec3::new(4.0, 5.0, 6.0),
                    inf_nav::NavKind::Room,
                );
                g
            },
        };
        let json = serde_json::to_string(&v).unwrap();
        // The skipped caches are absent from the serialized form …
        assert!(!json.contains("evaluated"));
        assert!(!json.contains("structures"));
        assert!(!json.contains("structures_gen"));
        assert!(!json.contains("structure_groups"));
        assert!(!json.contains("residents"));
        assert!(!json.contains("interior_nav"));
        let back: PcgVolume = serde_json::from_str(&json).unwrap();
        // … and decode empty, while the persisted fields round-trip.
        assert!(back.evaluated.is_empty());
        assert!(back.structures.is_empty());
        assert!(back.structure_groups.is_empty());
        assert!(back.residents.is_empty());
        assert!(back.interior_nav.is_empty());
        assert_eq!(back.structures_gen, 0, "the change stamp is derived too");
        assert_eq!(back.graph, v.graph);
        assert_eq!(back.extent, v.extent);
        assert_eq!(back.seed, 7);
        assert_eq!(back.draw_distance, 600.0);

        // A minimal payload fills every field from the defaults.
        let d: PcgVolume = serde_json::from_str("{}").unwrap();
        assert_eq!(d, PcgVolume::default());
        assert_eq!(d.extent, Vec2d::splat(50.0));
        assert_eq!(d.draw_distance, 1000.0);
    }

    /// **The whole population is written through one door, and the door checks
    /// the ranges** (IB-2b).
    ///
    /// A `StructureGroup` naming solids that are not there is not a crash — it is
    /// a distant building drawn with somebody else's walls, or a shell collider
    /// with nothing inside it. So the setter refuses out-of-range groups, and
    /// this arm is what says so; without it, the filter could be deleted and
    /// every other test in the tree would stay green.
    #[test]
    fn the_population_door_refuses_a_group_that_names_solids_that_are_not_there() {
        let solid = ScatteredSolid {
            center: DVec3::ZERO,
            half_extents: DVec3::splat(1.0),
            rotation: DQuat::IDENTITY,
        };
        let inst = ScatteredInstance {
            position: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            scale: 1.0,
            kind: 0,
            mesh: None,
            extent: None,
            glow: 0.0,
        };
        let group = |start, len, inst_start, inst_len| StructureGroup {
            shell: solid,
            start,
            len,
            inst_start,
            inst_len,
        };

        let mut v = PcgVolume::default();
        v.set_population(
            vec![inst; 4],
            vec![solid; 3],
            vec![
                group(0, 3, 0, 4), // fits both lists
                group(0, 4, 0, 4), // one solid past the end
                group(2, 1, 3, 2), // one instance past the end
            ],
            Vec::new(),
            Vec::new(),
            Default::default(),
        );
        assert_eq!(v.structure_groups.len(), 1, "{:?}", v.structure_groups);
        assert_eq!(v.structure_groups[0].range(), 0..3);
        assert_eq!(v.structure_groups[0].instance_range(), 0..4);
        // **The stamp is an IDENTITY, not a count** (island wave I8a audit): it
        // is drawn from a process-global counter, so what a write guarantees is
        // that the value moved and that no other volume in this process — or
        // this volume's own next incarnation — ever holds it. Asserting `1` here
        // would pin the exact property that made a memo keyed on
        // `(guid, stamp)` able to serve a destroyed volume's payload.
        let first = v.structures_gen;
        assert_ne!(first, 0, "one write is one stamp");

        // The three-list write is one stamp, and re-writing takes a fresh one.
        v.set_population(
            vec![inst; 1],
            vec![solid; 1],
            vec![group(0, 1, 0, 1)],
            Vec::new(),
            Vec::new(),
            Default::default(),
        );
        let second = v.structures_gen;
        assert!(second > first, "{first} then {second}");
        assert_eq!(v.evaluated.len(), 1);

        // `set_structures` still means "no grouping", and clears a stale one.
        v.set_structures(vec![solid; 2]);
        assert!(
            v.structure_groups.is_empty(),
            "a stale range must not survive"
        );
        assert!(v.structures_gen > second);

        // **And a fresh component never repeats a stamp**, which is the whole
        // point: a cell that deactivates and reactivates builds a new
        // `PcgVolume` under the SAME guid, and a per-volume counter gave both
        // incarnations `1` on their first write.
        let mut reborn = PcgVolume::default();
        assert_eq!(reborn.structures_gen, 0, "a fresh component is unwritten");
        reborn.set_structures(vec![solid; 1]);
        assert!(
            reborn.structures_gen > v.structures_gen,
            "a reincarnated volume repeated a stamp ({} against {})",
            reborn.structures_gen,
            v.structures_gen
        );

        // **And the ORDER is part of the contract**, because the bridge finds
        // the ungrouped solids with a single forward cursor: an overlapping or
        // out-of-order pair would make one box both a building's part and part
        // of another building's shell.
        let mut w = PcgVolume::default();
        w.set_population(
            vec![inst; 6],
            vec![solid; 6],
            vec![
                group(0, 2, 0, 2), // kept
                group(1, 2, 2, 2), // overlaps the first in SOLIDS
                group(2, 2, 1, 2), // overlaps the first in INSTANCES
                group(2, 2, 2, 2), // kept — the first legal successor
                group(0, 1, 0, 1), // backwards
            ],
            Vec::new(),
            Vec::new(),
            Default::default(),
        );
        assert_eq!(w.structure_groups.len(), 2, "{:?}", w.structure_groups);
        assert_eq!(w.structure_groups[0].range(), 0..2);
        assert_eq!(w.structure_groups[1].range(), 2..4);
    }

    #[test]
    fn terrain_biome_population_costs_zero_bytes() {
        // P19.3's biome population is `#[serde(skip)]` — the terrain-level sibling
        // of `PcgVolume::evaluated`. THAT IS WHY NO SCHEMA BUMP WAS NEEDED: bincode
        // is positional, so a *persisted* field here would have grown the wire and
        // forced v17 in both codec mirrors. A skipped one is byte-neutral, and this
        // is the proof — the same terrain encodes to the SAME bytes whether its
        // population is empty or a thousand instances long.
        let empty = Terrain::configured(5, 1.0);
        let populated = Terrain {
            biome_population: (0..1000)
                .map(|i| ScatteredInstance {
                    position: DVec3::new(i as f64, 2.0, 3.0),
                    rotation: DQuat::IDENTITY,
                    scale: 1.5,
                    kind: i % 4,
                    mesh: None,
                    extent: None,
                    glow: 0.0,
                })
                .collect(),
            ..Terrain::configured(5, 1.0)
        };
        assert_ne!(empty, populated, "the two terrains really do differ");

        let cfg = bincode::config::standard();
        let a = bincode::serde::encode_to_vec(&empty, cfg).unwrap();
        let b = bincode::serde::encode_to_vec(&populated, cfg).unwrap();
        assert_eq!(
            a, b,
            "`Terrain.biome_population` reached the wire — it is `serde(skip)` \
             precisely so it costs zero bytes and the `.inf_lvl` schema stays where \
             P19.2 left it"
        );

        // …and a decode leaves it empty (rebuilt on demand by the evaluate command
        // in the editor and by the level load in the player).
        let (back, _): (Terrain, _) = bincode::serde::decode_from_slice(&b, cfg).unwrap();
        assert!(back.biome_population.is_empty());
        assert_eq!(back, empty);

        // The dual-format (TOML/JSON) side skips it too — no key, and a decode
        // fills it empty exactly like `PcgVolume::evaluated`.
        let json = serde_json::to_string(&populated).unwrap();
        assert!(!json.contains("biome_population"));
        let from_json: Terrain = serde_json::from_str(&json).unwrap();
        assert!(from_json.biome_population.is_empty());
    }

    #[test]
    fn sprite_serde_round_trips() {
        let s = Sprite {
            texture: Some(Uuid::from_u128(0x1234_5678_9abc_def0_1122_3344_5566_7788)),
            size: Vec2d::new(2.0, 3.0),
            pivot: Vec2d::new(0.25, 0.75),
            color: Color::new(0.2, 0.4, 0.6, 0.8),
            atlas_rect: AtlasRect {
                min: Vec2d::new(0.1, 0.2),
                max: Vec2d::new(0.9, 0.8),
            },
            sorting_layer: -3,
            order: 5,
            flip_x: true,
            flip_y: false,
            billboard: BillboardMode::Spherical,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Sprite = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn tilemap_set_get_clear_roundtrips() {
        let mut tm = Tilemap::default();
        assert_eq!(tm.get_tile(0, 0), 0);
        assert!(tm.is_empty());

        tm.set_tile(3, 5, 7);
        assert_eq!(tm.get_tile(3, 5), 7);
        assert!(!tm.is_empty());
        // Setting one tile allocates exactly one chunk.
        assert_eq!(tm.chunks.len(), 1);

        // Clearing the only tile drops the chunk (memory-sane).
        tm.clear_tile(3, 5);
        assert_eq!(tm.get_tile(3, 5), 0);
        assert!(tm.is_empty());
        assert_eq!(tm.chunks.len(), 0);
    }

    #[test]
    fn tilemap_negative_coords_split_into_distinct_chunks() {
        let mut tm = Tilemap::default();
        // (-1, -1) floors to chunk (-1,-1) local (31,31); (0,0) is chunk (0,0).
        tm.set_tile(-1, -1, 2);
        tm.set_tile(0, 0, 3);
        assert_eq!(tm.get_tile(-1, -1), 2);
        assert_eq!(tm.get_tile(0, 0), 3);
        assert_eq!(tm.chunks.len(), 2);
        assert!(tm.chunks.contains_key(&(-1, -1)));
        assert!(tm.chunks.contains_key(&(0, 0)));
        // Tiles across the chunk boundary stay independent.
        assert_eq!(tm.get_tile(-1, 0), 0);
    }

    #[test]
    fn tilemap_bounds_span_occupied_tiles_across_chunks() {
        let mut tm = Tilemap::default();
        assert_eq!(tm.bounds(), None);
        tm.set_tile(-2, 4, 1);
        tm.set_tile(40, 3, 1); // chunk (1,0)
        tm.set_tile(5, -7, 1); // chunk (0,-1)
        let b = tm.bounds().unwrap();
        assert_eq!(
            b,
            TileBounds {
                min_x: -2,
                min_y: -7,
                max_x: 40,
                max_y: 4,
            }
        );
    }

    #[test]
    fn tile_chunk_serde_roundtrips_full_array() {
        let mut chunk = TileChunk::empty();
        chunk.set(0, 0, 11);
        chunk.set(31, 31, 22);
        let json = serde_json::to_string(&chunk).unwrap();
        let back: TileChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk, back);
        // A wrong-length payload is rejected.
        assert!(serde_json::from_str::<TileChunk>("[1,2,3]").is_err());
    }

    #[test]
    fn tilemap_serde_roundtrips_with_chunks() {
        let mut tm = Tilemap {
            texture: Some(Uuid::from_u128(0xABCD)),
            tile_size: Vec2d::new(0.5, 0.25),
            atlas_cols: 4,
            atlas_rows: 2,
            sorting_layer: -1,
            order: 3,
            tint: Color::new(0.5, 0.6, 0.7, 1.0),
            chunks: BTreeMap::new(),
        };
        tm.set_tile(1, 1, 5);
        tm.set_tile(100, -50, 8);
        let json = serde_json::to_string(&tm).unwrap();
        let back: Tilemap = serde_json::from_str(&json).unwrap();
        assert_eq!(tm, back);
    }

    #[test]
    fn tilemap_defaults_fill_missing_fields() {
        // A minimal payload (pre-additive-field) reconstructs via serde defaults.
        let minimal = r#"{
            "texture": null,
            "tile_size": { "x": 1.0, "y": 1.0 },
            "tint": { "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0 }
        }"#;
        let tm: Tilemap = serde_json::from_str(minimal).unwrap();
        assert_eq!(tm, Tilemap::default());
        assert_eq!(tm.atlas_cols, 1);
        assert_eq!(tm.atlas_rows, 1);
    }

    #[test]
    fn nine_slice_serde_round_trips_and_defaults() {
        let ns = NineSlice {
            texture: Some(Uuid::from_u128(0xBEEF)),
            size: Vec2d::new(6.0, 4.0),
            border_uv: [0.2, 0.3, 0.25, 0.15],
            border_world: Vec2d::new(0.5, 0.75),
            tint: Color::new(0.1, 0.2, 0.3, 1.0),
            sorting_layer: 2,
            order: 1,
        };
        let json = serde_json::to_string(&ns).unwrap();
        let back: NineSlice = serde_json::from_str(&json).unwrap();
        assert_eq!(ns, back);

        // A minimal payload reconstructs the additive fields via serde defaults.
        let minimal = r#"{
            "texture": null,
            "size": { "x": 2.0, "y": 2.0 },
            "tint": { "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0 }
        }"#;
        let ns: NineSlice = serde_json::from_str(minimal).unwrap();
        assert_eq!(ns, NineSlice::default());
    }

    #[test]
    fn text2d_serde_round_trips_and_defaults() {
        let t = Text2D {
            text: "Hi\nthere".to_string(),
            font_texture: Some(Uuid::from_u128(0xF0)),
            glyph_cols: 16,
            glyph_rows: 6,
            first_codepoint: 32,
            glyph_size: Vec2d::new(0.5, 0.5),
            tracking: 0.1,
            tint: Color::new(0.9, 0.8, 0.7, 1.0),
            sorting_layer: 3,
            order: 4,
            halign: TextAlign::Center,
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: Text2D = serde_json::from_str(&json).unwrap();
        assert_eq!(t, back);

        // A minimal payload (only always-present fields) fills the rest.
        let minimal = r#"{
            "text": "x",
            "glyph_size": { "x": 1.0, "y": 1.0 },
            "tint": { "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0 }
        }"#;
        let t: Text2D = serde_json::from_str(minimal).unwrap();
        assert_eq!(t.glyph_cols, 16);
        assert_eq!(t.glyph_rows, 6);
        assert_eq!(t.first_codepoint, 32);
        assert_eq!(t.halign, TextAlign::Left);
    }

    #[test]
    fn light2d_serde_round_trips() {
        let l = Light2D {
            color: Color::new(1.0, 0.5, 0.2, 1.0),
            intensity: 2.5,
            radius: 8.0,
        };
        let json = serde_json::to_string(&l).unwrap();
        let back: Light2D = serde_json::from_str(&json).unwrap();
        assert_eq!(l, back);
        assert_eq!(Light2D::default().radius, 5.0);
    }

    #[test]
    fn rigid_body_2d_serde_round_trips_and_defaults() {
        let rb = RigidBody2D {
            kind: BodyKind2D::Dynamic,
            gravity_scale: 2.0,
            fixed_rotation: true,
            linear_damping: 0.1,
            angular_damping: 0.2,
            ccd_enabled: true,
        };
        let json = serde_json::to_string(&rb).unwrap();
        let back: RigidBody2D = serde_json::from_str(&json).unwrap();
        assert_eq!(rb, back);
        // An empty payload fills every field from the defaults.
        let d: RigidBody2D = serde_json::from_str("{}").unwrap();
        assert_eq!(d, RigidBody2D::default());
        assert_eq!(d.kind, BodyKind2D::Static);
        assert_eq!(d.gravity_scale, 1.0);
    }

    #[test]
    fn collider_2d_serde_round_trips_and_defaults() {
        let c = Collider2D {
            shape_kind: ColliderShape2DKind::Capsule,
            half_extents: Vec2d::new(0.3, 0.8),
            radius: 0.25,
            offset: Vec2d::new(0.0, 0.1),
            friction: 0.9,
            restitution: 0.2,
            density: 2.0,
            sensor: true,
            collision_memberships: 0b1010,
            collision_filter: 0b0110,
            friction_combine: CombineRule::Min,
            restitution_combine: CombineRule::Max,
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: Collider2D = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
        // Defaults: box, unit-ish material.
        let d: Collider2D = serde_json::from_str("{}").unwrap();
        assert_eq!(d, Collider2D::default());
        assert_eq!(d.shape_kind, ColliderShape2DKind::Box);
        assert_eq!(d.half_extents, Vec2d::splat(0.5));
        assert_eq!(d.friction, 0.5);
        assert_eq!(d.density, 1.0);
    }

    #[test]
    fn character_controller_2d_serde_round_trips_and_defaults() {
        let cc = CharacterController2D {
            max_slope_deg: 60.0,
            snap_to_ground: 0.5,
            offset: 0.05,
        };
        let json = serde_json::to_string(&cc).unwrap();
        let back: CharacterController2D = serde_json::from_str(&json).unwrap();
        assert_eq!(cc, back);
        let d: CharacterController2D = serde_json::from_str("{}").unwrap();
        assert_eq!(d, CharacterController2D::default());
        assert_eq!(d.max_slope_deg, 45.0);
    }

    #[test]
    fn rigid_body_3d_serde_round_trips_and_defaults() {
        let rb = RigidBody3D {
            kind: BodyKind3D::Dynamic,
            gravity_scale: 2.0,
            fixed_rotation: true,
            linear_damping: 0.1,
            angular_damping: 0.2,
            ccd_enabled: true,
        };
        let json = serde_json::to_string(&rb).unwrap();
        let back: RigidBody3D = serde_json::from_str(&json).unwrap();
        assert_eq!(rb, back);
        // An empty payload fills every field from the defaults.
        let d: RigidBody3D = serde_json::from_str("{}").unwrap();
        assert_eq!(d, RigidBody3D::default());
        assert_eq!(d.kind, BodyKind3D::Static);
        assert_eq!(d.gravity_scale, 1.0);
    }

    #[test]
    fn collider_3d_serde_round_trips_and_defaults() {
        let c = Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(0.3, 0.8, 0.3),
            radius: 0.25,
            offset: Vec3d::new(0.0, 0.1, 0.0),
            friction: 0.9,
            restitution: 0.2,
            density: 2.0,
            sensor: true,
            collision_memberships: 0b1010,
            collision_filter: 0b0110,
            friction_combine: CombineRule::Min,
            restitution_combine: CombineRule::Max,
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: Collider3D = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
        // Defaults: box, unit-ish material.
        let d: Collider3D = serde_json::from_str("{}").unwrap();
        assert_eq!(d, Collider3D::default());
        assert_eq!(d.shape_kind, ColliderShape3DKind::Box);
        assert_eq!(d.half_extents, Vec3d::splat(0.5));
        assert_eq!(d.friction, 0.5);
        assert_eq!(d.density, 1.0);
    }

    #[test]
    fn character_controller_3d_serde_round_trips_and_defaults() {
        let cc = CharacterController3D {
            max_slope_deg: 60.0,
            snap_to_ground: 0.5,
            offset: 0.05,
        };
        let json = serde_json::to_string(&cc).unwrap();
        let back: CharacterController3D = serde_json::from_str(&json).unwrap();
        assert_eq!(cc, back);
        let d: CharacterController3D = serde_json::from_str("{}").unwrap();
        assert_eq!(d, CharacterController3D::default());
        assert_eq!(d.max_slope_deg, 45.0);
    }

    #[test]
    fn character_movement_serde_round_trips_and_leaves_the_runtime_behind() {
        let mut cm = CharacterMovement {
            mode: MovementMode::Crouch,
            gait: Gait::Sprint,
            rotation_mode: RotationMode::Aiming,
            overlay: "rifle".into(),
            player_controlled: true,
            walk_speed_mps: 1.1,
            acceleration: SpeedCurve::new(1.0, 2.0, 3.0, 4.0),
            ..Default::default()
        };
        // A dirtied runtime must not reach the wire: it is derived state, and a
        // saved level that carried one step's velocity would replay differently
        // on load. (`AnimStateMachine::runtime` is the precedent.)
        cm.runtime.velocity = Vec3d::new(1.0, 2.0, 3.0);
        cm.runtime.refusals = 9;

        let json = serde_json::to_string(&cm).unwrap();
        assert!(
            !json.contains("\"runtime\"") && !json.contains("aim_yaw_rate_dps"),
            "the live runtime is `#[serde(skip)]`: {json}"
        );
        let back: CharacterMovement = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mode, MovementMode::Crouch);
        assert_eq!(back.gait, Gait::Sprint);
        assert_eq!(back.rotation_mode, RotationMode::Aiming);
        assert_eq!(back.overlay, "rifle");
        assert!(back.player_controlled);
        assert_eq!(back.walk_speed_mps, 1.1);
        assert_eq!(back.acceleration, SpeedCurve::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(back.runtime, MovementRuntime::default());

        let d: CharacterMovement = serde_json::from_str("{}").unwrap();
        assert_eq!(d, CharacterMovement::default());
        // The ALS constants, converted ONCE at the port boundary (IM-1). These
        // four are the ones a later reader is most likely to want to check
        // against the source: 165/375/650 cm/s and the two landing thresholds.
        assert_eq!(d.walk_speed_mps, 1.65);
        assert_eq!(d.run_speed_mps, 3.75);
        assert_eq!(d.sprint_speed_mps, 6.5);
        assert_eq!(d.land_hard_mps, 7.0);
        assert_eq!(d.land_ragdoll_mps, 10.0);
        // And the field that makes stairs work is non-zero by default, because a
        // step height of zero is exactly the state this wave found the engine in.
        assert!(d.step_height_m > 0.0, "autostep is on by default");
    }

    /// A reserved slot no reader ever asks about is a comment rather than a
    /// slot. Both directions on all three enums that carry them, so the
    /// accessors cannot answer `Some` for everything.
    #[test]
    fn the_movement_reserved_slots_are_reachable_and_named() {
        assert_eq!(MovementMode::Reserved14.reserved_slot(), Some(14));
        assert_eq!(MovementMode::Reserved17.reserved_slot(), Some(17));
        assert_eq!(MovementMode::Grounded.reserved_slot(), None);
        assert_eq!(MovementMode::Flying.reserved_slot(), None);
        assert_eq!(Gait::Reserved3.reserved_slot(), Some(3));
        assert_eq!(Gait::Run.reserved_slot(), None);
        assert_eq!(RotationMode::Reserved4.reserved_slot(), Some(4));
        assert_eq!(RotationMode::Aiming.reserved_slot(), None);

        // Entering a mode this build has no mechanics for is a typed refusal.
        // **P29.4 took two of the four and P29.7 took the last two**, so the
        // deferred set is now exactly the reserved slots — a mode a NEWER build
        // wrote into a file this one is reading, which is the case the freeze-pin
        // exists for and the one that stays live for ever. Moving a row is the
        // only way this arm can change, which is what makes it a ledger of what
        // is implemented rather than a restatement of the enum.
        for m in [
            MovementMode::Reserved14,
            MovementMode::Reserved15,
            MovementMode::Reserved16,
            MovementMode::Reserved17,
        ] {
            assert!(m.is_deferred(), "{m:?}");
        }
        for m in [
            MovementMode::Grounded,
            MovementMode::Crouch,
            MovementMode::Prone,
            MovementMode::Slide,
            MovementMode::Roll,
            MovementMode::Dive,
            MovementMode::FallFree,
            MovementMode::FallControlled,
            MovementMode::SwimSurface,
            MovementMode::SwimUnder,
            MovementMode::Mantle,
            MovementMode::Ragdoll,
        ] {
            assert!(!m.is_deferred(), "{m:?} has its mechanics");
        }
        // The three families, asserted rather than described.
        assert!(MovementMode::Slide.is_grounded_family());
        assert!(!MovementMode::FallFree.is_grounded_family());
        assert!(MovementMode::Dive.is_falling());
        assert!(MovementMode::SwimUnder.is_swimming());
        assert!(!MovementMode::SwimUnder.is_falling());
    }

    #[test]
    fn joint_3d_serde_round_trips_including_entity_ref() {
        // GUARD (v6 persistence gap): `Joint3D` round-trips through serde — the
        // `other` entity ref (now an `EntityRef`, serde-transparent) IS
        // serde-persisted, so the component itself is disk-ready. It is NOT yet
        // wired into the `.inf_lvl` `EntityRecord` (that is the deferred v6 schema
        // bump); this test pins that the component's own serialization is stable so
        // the eventual record slot is a pure append. See
        // `inf-editor-core::scene::serialize`.
        let j = Joint3D {
            other: EntityRef::new(Uuid::from_u128(42)),
            kind: JointKind3D::Revolute,
            axis: Vec3d::new(0.0, 0.0, 1.0),
            limits_enabled: true,
            limit_min: -1.0,
            limit_max: 1.0,
            motor_enabled: true,
            motor_target_vel: 3.0,
            ..Default::default()
        };
        let json = serde_json::to_string(&j).unwrap();
        let back: Joint3D = serde_json::from_str(&json).unwrap();
        assert_eq!(j, back);
        assert_eq!(back.other, EntityRef::new(Uuid::from_u128(42)));
        // Defaults: a Fixed, unbound joint.
        let d: Joint3D = serde_json::from_str("{}").unwrap();
        assert_eq!(d, Joint3D::default());
        assert_eq!(d.kind, JointKind3D::Fixed);
        assert_eq!(d.other, EntityRef::NONE);
        assert_eq!(d.motor_max_force, f64::MAX);
    }

    #[test]
    fn joint_2d_serde_round_trips() {
        let j = Joint2D {
            other: EntityRef::new(Uuid::from_u128(7)),
            kind: JointKind2D::Distance,
            max_distance: 3.5,
            ..Default::default()
        };
        let back: Joint2D = serde_json::from_str(&serde_json::to_string(&j).unwrap()).unwrap();
        assert_eq!(j, back);
        let d: Joint2D = serde_json::from_str("{}").unwrap();
        assert_eq!(d, Joint2D::default());
        assert_eq!(d.kind, JointKind2D::Fixed);
    }

    #[test]
    fn sprite_defaults_fill_missing_fields() {
        // A pre-P8 payload predates every additive field: only the always-present
        // fields are stored. `#[serde(default)]` must reconstruct the rest.
        let minimal = r#"{
            "texture": null,
            "size": { "x": 1.0, "y": 1.0 },
            "pivot": { "x": 0.5, "y": 0.5 },
            "color": { "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0 }
        }"#;
        let s: Sprite = serde_json::from_str(minimal).unwrap();
        assert_eq!(s, Sprite::default());
        assert_eq!(s.atlas_rect, AtlasRect::default());
        assert_eq!(s.sorting_layer, 0);
        // The appended 2.5D field defaults to None on a pre-billboard payload.
        assert_eq!(s.billboard, BillboardMode::None);
    }

    #[test]
    fn sprite_billboard_field_round_trips_and_defaults() {
        // A full sprite carrying a billboard mode round-trips.
        let s = Sprite {
            billboard: BillboardMode::Cylindrical,
            ..Default::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Sprite = serde_json::from_str(&json).unwrap();
        assert_eq!(back.billboard, BillboardMode::Cylindrical);

        // A payload written before the field existed (no `billboard` key) decodes
        // with `billboard` defaulted to None — the additive-field discipline.
        let pre_billboard = r#"{
            "texture": null,
            "size": { "x": 1.0, "y": 1.0 },
            "pivot": { "x": 0.5, "y": 0.5 },
            "color": { "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0 },
            "atlas_rect": { "min": {"x":0.0,"y":0.0}, "max": {"x":1.0,"y":1.0} },
            "sorting_layer": 0, "order": 0, "flip_x": false, "flip_y": false
        }"#;
        let s: Sprite = serde_json::from_str(pre_billboard).unwrap();
        assert_eq!(s.billboard, BillboardMode::None);
    }

    #[test]
    fn skeletal_components_serde_round_trip() {
        // NOTE (corrected P24.1): this used to say `SkeletalMesh`/`AnimPlayer` were
        // "NOT yet slots in the v4 `.inf_lvl` `EntityRecord`" and that a
        // component-level round trip was "all the persistence they have". Both
        // became false at the schema-v5 migration the note itself pointed at —
        // `inf_scene::EntityRecord` has carried `skeletal_mesh` and `anim_player`
        // ever since, and the scene is at v20. What this test still covers is the
        // component's OWN serde shape, which the record's slot delegates to.
        let sm = SkeletalMesh {
            mesh: Some(Uuid::from_u128(0xA11CE)),
            skeleton: Some(Uuid::from_u128(0xB0B)),
        };
        let back: SkeletalMesh =
            serde_json::from_str(&serde_json::to_string(&sm).unwrap()).unwrap();
        assert_eq!(sm, back);

        let ap = AnimPlayer {
            clip: Some(Uuid::from_u128(0xC11E)),
            t: 1.25,
            speed: 0.5,
            looping: false,
            playing: true,
            duration: 2.0,
        };
        let back: AnimPlayer = serde_json::from_str(&serde_json::to_string(&ap).unwrap()).unwrap();
        assert_eq!(ap, back);

        // A minimal payload fills the additive defaults (speed 1, looping, playing).
        let d: AnimPlayer = serde_json::from_str("{}").unwrap();
        assert_eq!(d, AnimPlayer::default());
        assert_eq!(d.speed, 1.0);
        assert!(d.looping && d.playing);
        let d: SkeletalMesh = serde_json::from_str("{}").unwrap();
        assert_eq!(d, SkeletalMesh::default());
    }

    #[test]
    fn audio_components_serde_round_trip() {
        // NOTE: like Terrain/PcgVolume and the anim components, `AudioSource`/
        // `AudioListener` are NOT yet slots in the `.inf_lvl` `EntityRecord` — this
        // component-level round-trip is all the persistence they have until the
        // schema-v6 migration. This test pins that gap; no schema bump is made.
        let src = AudioSource {
            clip: Some(Uuid::from_u128(0x5000D)),
            bus: "music".into(),
            volume: 0.5,
            pitch: 1.5,
            looping: true,
            spatial: true,
            min_distance: 2.0,
            max_distance: 40.0,
            distance_model: DistanceModel::Exponential,
            rolloff: 2.0,
            occlusion: true,
            autoplay: true,
        };
        let back: AudioSource =
            serde_json::from_str(&serde_json::to_string(&src).unwrap()).unwrap();
        assert_eq!(src, back);

        let lis = AudioListener { active: true };
        let back: AudioListener =
            serde_json::from_str(&serde_json::to_string(&lis).unwrap()).unwrap();
        assert_eq!(lis, back);

        // A minimal payload fills the additive defaults (bus "sfx", unity volume/
        // pitch, spatial, inverse falloff).
        let d: AudioSource = serde_json::from_str("{}").unwrap();
        assert_eq!(d, AudioSource::default());
        assert_eq!(d.bus, "sfx");
        assert_eq!(d.volume, 1.0);
        assert!(d.spatial);
        assert_eq!(d.distance_model, DistanceModel::Inverse);
        let d: AudioListener = serde_json::from_str("{}").unwrap();
        assert_eq!(d, AudioListener::default());
        assert!(!d.active);
    }

    #[test]
    fn anim_state_machine_serde_skips_runtime() {
        // The `sm` GUID + `params_from_vars` persist; the `runtime` is transient
        // (serde-skip) and always comes back at its default (the same v5 gap as
        // AnimPlayer — component-level round-trip only, until the schema migration).
        let asm = AnimStateMachine {
            sm: Some(Uuid::from_u128(0x5A11)),
            params_from_vars: true,
            runtime: SmRuntimeState {
                current: 3,
                state_time: 9.0,
                started: true,
                ..Default::default()
            },
        };
        let json = serde_json::to_string(&asm).unwrap();
        assert!(!json.contains("current"), "runtime must not serialize");
        let back: AnimStateMachine = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sm, asm.sm);
        assert!(back.params_from_vars);
        assert_eq!(back.runtime, SmRuntimeState::default());

        // A minimal payload fills the additive default (`params_from_vars = true`).
        let d: AnimStateMachine = serde_json::from_str("{}").unwrap();
        assert_eq!(d, AnimStateMachine::default());
        assert!(d.params_from_vars);
    }

    #[test]
    fn anim_player_advance_wraps_clamps_and_pauses() {
        // Looping wraps against duration.
        let mut p = AnimPlayer {
            t: 1.8,
            speed: 1.0,
            duration: 2.0,
            looping: true,
            ..Default::default()
        };
        p.advance(0.5);
        assert!((p.t - 0.3).abs() < 1e-9);
        // Non-looping clamps at the end.
        let mut p = AnimPlayer {
            t: 1.9,
            speed: 1.0,
            duration: 2.0,
            looping: false,
            ..Default::default()
        };
        p.advance(0.5);
        assert_eq!(p.t, 2.0);
        // Unknown duration free-runs.
        let mut p = AnimPlayer {
            t: 0.0,
            speed: 2.0,
            duration: 0.0,
            ..Default::default()
        };
        p.advance(0.5);
        assert_eq!(p.t, 1.0);
        // Paused never advances.
        let mut p = AnimPlayer {
            t: 0.4,
            playing: false,
            ..Default::default()
        };
        p.advance(1.0);
        assert_eq!(p.t, 0.4);
    }

    #[test]
    fn material_v8_fields_round_trip_and_default() {
        let m = Material {
            base_color: Color::new(0.1, 0.2, 0.3, 0.5),
            metallic: 0.2,
            roughness: 0.7,
            emissive: Color::new(0.0, 0.0, 0.0, 1.0),
            // v26 (wave VIS1a): the multiplier the 8-bit colour above cannot
            // express, round-tripped beside it.
            emissive_intensity: 12.5,
            blend: BlendMode::Translucent,
            alpha_cutoff: 0.25,
            // v22 (P26.3b): the persisted `.inf_mat` binding, round-tripped here
            // beside the scalars it does not replace.
            asset: Some(Uuid::from_u128(0xFA7E_0026)),
        };
        let back: Material = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(m, back);
        assert_eq!(back.asset, Some(Uuid::from_u128(0xFA7E_0026)));
        // Defaults for the v8 fields — and for v22's, which is the no-texture
        // path and must stay `None` so a fresh surface renders off its scalars.
        let d = Material::default();
        assert_eq!(d.blend, BlendMode::Opaque);
        assert_eq!(d.alpha_cutoff, 0.5);
        assert_eq!(d.asset, None);
        // A pre-v8 payload (no blend / alpha_cutoff) fills the additive defaults.
        let pre = r#"{ "base_color": {"r":0.8,"g":0.8,"b":0.8,"a":1.0},
            "metallic": 0.0, "roughness": 0.5,
            "emissive": {"r":0.0,"g":0.0,"b":0.0,"a":1.0} }"#;
        let old: Material = serde_json::from_str(pre).unwrap();
        assert_eq!(old, Material::default());
    }

    #[test]
    fn light_v8_fields_round_trip_and_default() {
        let l = Light {
            kind: LightKind::Spot,
            color: Color::WHITE,
            intensity: 2.0,
            range: 12.0,
            inner_cone_deg: 20.0,
            outer_cone_deg: 35.0,
            cast_shadows: false,
        };
        let back: Light = serde_json::from_str(&serde_json::to_string(&l).unwrap()).unwrap();
        assert_eq!(l, back);
        let d = Light::default();
        assert_eq!(d.range, 0.0);
        assert_eq!(d.inner_cone_deg, 30.0);
        assert_eq!(d.outer_cone_deg, 40.0);
        assert!(d.cast_shadows);
        // A pre-v8 payload (kind/color/intensity only) fills the additive defaults.
        let pre = r#"{ "kind": "Directional",
            "color": {"r":1.0,"g":1.0,"b":1.0,"a":1.0}, "intensity": 1.0 }"#;
        let old: Light = serde_json::from_str(pre).unwrap();
        assert_eq!(old.range, 0.0);
        assert_eq!(old.inner_cone_deg, 30.0);
        assert_eq!(old.outer_cone_deg, 40.0);
        assert!(old.cast_shadows);
    }

    #[test]
    fn decal_serde_round_trips_and_defaults() {
        let dc = Decal {
            size: Vec3d::new(2.0, 3.0, 4.0),
            color: Color::new(1.0, 0.0, 0.0, 1.0),
            opacity: 0.5,
            fade_angle_deg: 45.0,
        };
        let back: Decal = serde_json::from_str(&serde_json::to_string(&dc).unwrap()).unwrap();
        assert_eq!(dc, back);
        let d: Decal = serde_json::from_str("{}").unwrap();
        assert_eq!(d, Decal::default());
        assert_eq!(d.size, Vec3d::ONE);
        assert_eq!(d.opacity, 1.0);
        assert_eq!(d.fade_angle_deg, 60.0);
    }

    #[test]
    fn volume_serde_round_trips_and_defaults() {
        let v = Volume {
            kind: VolumeKind::Blocking,
            tint: Color::new(0.2, 0.4, 0.6, 0.8),
        };
        let back: Volume = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
        assert_eq!(v, back);
        let d: Volume = serde_json::from_str("{}").unwrap();
        assert_eq!(d, Volume::default());
        assert_eq!(d.kind, VolumeKind::Trigger);
    }

    #[test]
    fn spline_serde_round_trips_and_defaults() {
        let s = Spline {
            points: vec![
                Vec3d::ZERO,
                Vec3d::new(1.0, 2.0, 3.0),
                Vec3d::new(4.0, 5.0, 6.0),
            ],
            closed: true,
            interp: SplineInterp::Linear,
        };
        let back: Spline = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(s, back);
        let d: Spline = serde_json::from_str("{}").unwrap();
        assert_eq!(d, Spline::default());
        assert_eq!(d.points, vec![Vec3d::ZERO, Vec3d::new(0.0, 0.0, 5.0)]);
        assert_eq!(d.interp, SplineInterp::CatmullRom);
        assert!(!d.closed);
    }

    #[test]
    fn foliage_serde_round_trips_and_defaults() {
        let f = Foliage {
            palette: vec![
                FoliagePaletteEntry {
                    primitive: Primitive::Sphere,
                    tint: Color::new(0.1, 0.5, 0.1, 1.0),
                },
                FoliagePaletteEntry::default(),
            ],
            instances: vec![
                FoliageInstance {
                    position: Vec3d::new(1.0, 0.0, 2.0),
                    rotation: Vec3d::new(0.0, 90.0, 0.0),
                    scale: 1.5,
                    kind: 1,
                },
                FoliageInstance {
                    position: Vec3d::new(-3.0, 0.0, 4.0),
                    rotation: Vec3d::ZERO,
                    scale: 0.8,
                    kind: 0,
                },
                FoliageInstance::default(),
            ],
        };
        let back: Foliage = serde_json::from_str(&serde_json::to_string(&f).unwrap()).unwrap();
        assert_eq!(f, back);
        assert_eq!(back.instances.len(), 3);
        let d: Foliage = serde_json::from_str("{}").unwrap();
        assert_eq!(d, Foliage::default());
        assert!(d.palette.is_empty() && d.instances.is_empty());
    }

    #[test]
    fn streaming_source_serde_round_trips_and_defaults() {
        let s = StreamingSource { radius_m: 384.0 };
        let back: StreamingSource =
            serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(s, back);
        let d: StreamingSource = serde_json::from_str("{}").unwrap();
        assert_eq!(d, StreamingSource::default());
        assert_eq!(d.radius_m, 512.0);
    }

    #[test]
    fn time_of_day_serde_round_trips_and_defaults() {
        let t = TimeOfDay {
            seconds: 3_600.0,
            day_of_year: 355,
            latitude_deg: -33.9,
            longitude_deg: 151.2,
            rate: 60.0,
        };
        let back: TimeOfDay = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
        assert_eq!(t, back);
        // Every field is `#[serde(default)]`, so an empty payload is the default.
        let d: TimeOfDay = serde_json::from_str("{}").unwrap();
        assert_eq!(d, TimeOfDay::default());
        assert_eq!(d.seconds, 36_000.0);
        assert_eq!(d.day_of_year, 172);
        assert_eq!(d.latitude_deg, 48.9);
        assert_eq!(d.longitude_deg, 0.0);
        assert_eq!(d.rate, 0.0, "a level opts into a moving sun explicitly");
        // A partial payload fills the rest with defaults (the additive contract).
        let p: TimeOfDay = serde_json::from_str(r#"{"rate":120.0}"#).unwrap();
        assert_eq!(p.rate, 120.0);
        assert_eq!(p.seconds, 36_000.0);
    }

    #[test]
    fn time_of_day_default_reproduces_the_retired_sun_constant() {
        // The compile-time `inf_render::camera::SUN_DIR` this component retires.
        let legacy = DVec3::new(0.45, 0.75, 0.3).normalize();
        let d = inf_math::solar::sun_direction(&TimeOfDay::default().solar_input());
        let angle = d.dot(legacy).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(
            angle < 1.6,
            "default sun is {angle}° off the retired constant"
        );
    }

    #[test]
    fn time_of_day_advance_wraps_and_freezes() {
        let mut t = TimeOfDay {
            seconds: 86_399.0,
            day_of_year: 365,
            rate: 1.0,
            ..TimeOfDay::default()
        };
        t.advance(2.0);
        assert_eq!(t.seconds, 1.0);
        assert_eq!(t.day_of_year, 1, "the year rolls with the day");
        // rate 0 freezes.
        let mut frozen = TimeOfDay::default();
        let before = frozen;
        frozen.advance(1_000.0);
        assert_eq!(frozen, before);
    }

    #[test]
    fn sky_atmosphere_serde_round_trips_and_defaults() {
        let s = SkyAtmosphere {
            enabled: false,
            sun_intensity: 5.0,
            night_darkening: 0.25,
            ..SkyAtmosphere::default()
        };
        let back: SkyAtmosphere =
            serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(s, back);
        let d: SkyAtmosphere = serde_json::from_str("{}").unwrap();
        assert_eq!(d, SkyAtmosphere::default());
        assert!(d.enabled);
        assert_eq!(d.sun_intensity, 3.0);
        // The gradient defaults must be the renderer's existing `SkyParams`
        // defaults verbatim — that identity is what keeps the sky byte-identical.
        assert_eq!(d.zenith, Color::new(0.012, 0.021, 0.038, 1.0));
        assert_eq!(d.horizon, Color::new(0.055, 0.081, 0.120, 1.0));
        assert_eq!(d.ground, Color::new(0.009, 0.011, 0.015, 1.0));
    }

    #[test]
    fn always_loaded_is_a_zero_field_marker() {
        let m = AlwaysLoaded;
        let back: AlwaysLoaded = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(m, back);
        // A marker carries no state, so a bincode `Option<AlwaysLoaded>` is one
        // tag byte — the cheapest possible per-entity slot.
        let bytes =
            bincode::serde::encode_to_vec(Some(AlwaysLoaded), bincode::config::standard()).unwrap();
        assert_eq!(bytes.len(), 1);
    }

    // ── water (P20.1) ────────────────────────────────────────────────────

    #[test]
    fn water_body_serde_round_trips_and_defaults() {
        let w = WaterBody {
            kind: WaterKind::River,
            level_m: -3.5,
            river_flow_m_s: -2.0,
            wind_from_weather: false,
            wave_seed: 0xDEAD_BEEF,
            ..WaterBody::default()
        };
        let back: WaterBody = serde_json::from_str(&serde_json::to_string(&w).unwrap()).unwrap();
        assert_eq!(w, back);
        // Every field is `serde(default)`, so an empty record is the default
        // component — the additive-component contract.
        let d: WaterBody = serde_json::from_str("{}").unwrap();
        assert_eq!(d, WaterBody::default());
        assert_eq!(d.kind, WaterKind::Ocean);
        assert_eq!(d.level_m, 0.0);
        assert!(d.wind_from_weather);
        // …and the same through bincode, which is the codec that actually ships.
        let cfg = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(w, cfg).unwrap();
        let (rt, _): (WaterBody, usize) = bincode::serde::decode_from_slice(&bytes, cfg).unwrap();
        assert_eq!(rt, w);
    }

    /// **THE WIRE-ENUM LAW (P19.2), applied to the first new scene enum since it
    /// was written down.** bincode encodes an externally-tagged enum as its
    /// *declaration index*, so inserting a kind in the middle silently renumbers
    /// every kind after it — and every committed level's oceans decode as lakes.
    /// New variants go at the end, always.
    ///
    /// The string identifiers are pinned in the same breath: they reach cook
    /// advisories and logs, and a silently-renamed one makes an old report
    /// unreadable.
    #[test]
    fn water_kind_discriminants_are_frozen() {
        let cfg = bincode::config::standard();
        let tag = |k: WaterKind| bincode::serde::encode_to_vec(k, cfg).unwrap()[0];
        assert_eq!(tag(WaterKind::Ocean), 0);
        assert_eq!(tag(WaterKind::Lake), 1);
        assert_eq!(tag(WaterKind::River), 2);
        assert_eq!(WaterKind::Ocean.as_str(), "ocean");
        assert_eq!(WaterKind::Lake.as_str(), "lake");
        assert_eq!(WaterKind::River.as_str(), "river");
        // The default is variant 0, which is what makes a defaulted record's
        // first byte a zero and keeps a hand-written JSON record honest.
        assert_eq!(WaterKind::default(), WaterKind::Ocean);
        // Spelled out once as raw bytes, so the encoding itself is visible rather
        // than inferred from the helper above.
        assert_eq!(
            &bincode::serde::encode_to_vec(WaterKind::River, cfg).unwrap()[..],
            &[2]
        );
    }

    /// **The wire-enum law, applied to all eighteen** (L5.F7).
    ///
    /// `water_kind_discriminants_are_frozen` above states the law and its own
    /// docstring is honest about its scope — *"applied to the **first new** scene
    /// enum since it was written down"*. The other seventeen enums that reach
    /// `.inf_lvl` bytes predate P19.2 and carried the identical exposure with no
    /// test that would notice: bincode encodes an externally-tagged enum as its
    /// **declaration index**, so inserting a variant mid-list renumbers
    /// everything after it and every committed level's `Spot` lights decode as
    /// `Point`s, its `Dynamic` bodies as `Kinematic`, its `Storm` as `Fog`.
    ///
    /// A law with a one-instance implementation is an aspiration. This is the
    /// tripwire that makes it enforceable — one table, every enum, the encoded
    /// first byte per variant in declaration order.
    ///
    /// **To add a variant: append it to the end of the enum AND to the end of
    /// its row here.** Anything else is a wire break, and this test is what says
    /// so.
    #[test]
    fn every_scene_wire_enum_has_frozen_discriminants() {
        let cfg = bincode::config::standard();
        /// One row: the enum's name, and its variants in declaration order.
        /// `assert_row!` encodes each and pins it to its index.
        macro_rules! frozen {
            ($ty:ty => [$($variant:expr),+ $(,)?]) => {{
                let variants: &[$ty] = &[$($variant),+];
                for (i, v) in variants.iter().enumerate() {
                    let bytes = bincode::serde::encode_to_vec(v, cfg).unwrap();
                    assert_eq!(
                        bytes.len(),
                        1,
                        "{}::{:?} encodes to {} bytes; a unit variant is one",
                        stringify!($ty),
                        v,
                        bytes.len()
                    );
                    assert_eq!(
                        bytes[0] as usize,
                        i,
                        "{}::{:?} moved from discriminant {i} to {} — every committed \
                         level that stored it now decodes as a different variant",
                        stringify!($ty),
                        v,
                        bytes[0]
                    );
                }
                variants.len()
            }};
        }

        let mut pinned = 0usize;
        let mut enums = 0usize;
        macro_rules! pin {
            ($ty:ty => [$($variant:expr),+ $(,)?]) => {
                pinned += frozen!($ty => [$($variant),+]);
                enums += 1;
            };
        }

        pin!(Primitive => [
            Primitive::Cube, Primitive::Sphere, Primitive::Plane,
            Primitive::Cylinder, Primitive::Cone,
        ]);
        pin!(BlendMode => [BlendMode::Opaque, BlendMode::Masked, BlendMode::Translucent]);
        pin!(BillboardMode => [
            BillboardMode::None, BillboardMode::Spherical, BillboardMode::Cylindrical,
        ]);
        pin!(TextAlign => [TextAlign::Left, TextAlign::Center, TextAlign::Right]);
        pin!(LightKind => [LightKind::Directional, LightKind::Point, LightKind::Spot]);
        pin!(BodyKind2D => [BodyKind2D::Static, BodyKind2D::Kinematic, BodyKind2D::Dynamic]);
        pin!(ColliderShape2DKind => [
            ColliderShape2DKind::Box, ColliderShape2DKind::Circle, ColliderShape2DKind::Capsule,
        ]);
        pin!(CombineRule => [
            CombineRule::Average, CombineRule::Min, CombineRule::Multiply, CombineRule::Max,
        ]);
        pin!(BodyKind3D => [BodyKind3D::Static, BodyKind3D::Kinematic, BodyKind3D::Dynamic]);
        pin!(ColliderShape3DKind => [
            ColliderShape3DKind::Box, ColliderShape3DKind::Sphere, ColliderShape3DKind::Capsule,
        ]);
        pin!(JointKind3D => [
            JointKind3D::Fixed, JointKind3D::Revolute, JointKind3D::Prismatic,
            JointKind3D::Spherical, JointKind3D::Distance,
        ]);
        pin!(JointKind2D => [
            JointKind2D::Fixed, JointKind2D::Revolute, JointKind2D::Prismatic,
            JointKind2D::Distance,
        ]);
        pin!(RootMotionMode => [RootMotionMode::None, RootMotionMode::ApplyToEntity]);
        pin!(DistanceModel => [
            DistanceModel::Linear, DistanceModel::Inverse, DistanceModel::Exponential,
        ]);
        pin!(VolumeKind => [VolumeKind::Trigger, VolumeKind::Blocking]);
        pin!(SplineInterp => [SplineInterp::Linear, SplineInterp::CatmullRom]);
        pin!(WeatherPreset => [
            WeatherPreset::Clear, WeatherPreset::Overcast, WeatherPreset::Storm,
            WeatherPreset::Fog, WeatherPreset::Snow,
        ]);
        pin!(WaterKind => [WaterKind::Ocean, WaterKind::Lake, WaterKind::River]);
        // ── P29.3, the movement catalogue ──
        //
        // `MovementMode` is frozen on the day it is born (the 2026-08-15
        // catalogue amendment says so in as many words) with four reserved
        // slots, because the scene bumps once this phase and a mode arriving in
        // P29.4 or P29.7 without a slot would need a second bump.
        pin!(MovementMode => [
            MovementMode::Grounded, MovementMode::Crouch, MovementMode::Prone,
            MovementMode::Slide, MovementMode::Roll, MovementMode::Dive,
            MovementMode::FallFree, MovementMode::FallControlled,
            MovementMode::SwimSurface, MovementMode::SwimUnder,
            MovementMode::Mantle, MovementMode::Ragdoll,
            MovementMode::Driving, MovementMode::Flying,
            MovementMode::Reserved14, MovementMode::Reserved15,
            MovementMode::Reserved16, MovementMode::Reserved17,
        ]);
        pin!(Gait => [
            Gait::Walk, Gait::Run, Gait::Sprint, Gait::Reserved3, Gait::Reserved4,
        ]);
        pin!(RotationMode => [
            RotationMode::VelocityDirection, RotationMode::LookingDirection,
            RotationMode::Aiming, RotationMode::Reserved3, RotationMode::Reserved4,
        ]);
        pin!(MovementDirection => [
            MovementDirection::Forward, MovementDirection::Right,
            MovementDirection::Left, MovementDirection::Backward,
        ]);
        pin!(LandingKind => [
            LandingKind::None, LandingKind::Soft, LandingKind::Hard,
            LandingKind::Roll, LandingKind::Ragdoll,
        ]);
        pin!(MovementRefusal => [
            MovementRefusal::None, MovementRefusal::NoOverheadClearance,
            MovementRefusal::ModeNotYetImplemented, MovementRefusal::IllegalTransition,
            MovementRefusal::ConditionNotMet,
        ]);

        // The counts are part of the pin. **They count TABLE ROWS, not enums in
        // the module** (round-2 finding R2.D) — so a nineteenth enum added to
        // the scene wire and never added here leaves both numbers at 18/59 and
        // passes green, which is exactly the state this test was written to end.
        // `the_freeze_table_covers_every_wire_enum_in_this_module` below is the
        // census that closes it; these two stay because they are what makes an
        // edit to an EXISTING row deliberate.
        assert_eq!(enums, 24, "a scene wire enum joined or left the table");
        assert_eq!(pinned, 101, "a variant joined or left without a decision");
        assert_eq!(
            enums,
            FROZEN_ENUMS.len(),
            "the freeze table and its name list disagree about how many enums it covers; the census below reads the names"
        );
    }

    /// The names in the freeze table above, as data the census can read.
    ///
    /// A second copy of the list, deliberately: the `pin!` macro's rows are
    /// *types*, and Rust has no way to reflect a module's enums back out of the
    /// type system. So the correspondence is asserted from two directions — the
    /// row count against this list's length, and this list against the module's
    /// own source.
    const FROZEN_ENUMS: &[&str] = &[
        "Primitive",
        "BlendMode",
        "BillboardMode",
        "TextAlign",
        "LightKind",
        "BodyKind2D",
        "ColliderShape2DKind",
        "CombineRule",
        "BodyKind3D",
        "ColliderShape3DKind",
        "JointKind3D",
        "JointKind2D",
        "RootMotionMode",
        "DistanceModel",
        "VolumeKind",
        "SplineInterp",
        "WeatherPreset",
        "WaterKind",
        // ── P29.3 ──
        "MovementMode",
        "Gait",
        "RotationMode",
        "MovementDirection",
        "LandingKind",
        "MovementRefusal",
    ];

    /// **Round-2 finding R2.D: the count guard counted the wrong thing.**
    ///
    /// `assert_eq!(enums, 18)` counts the rows somebody wrote, so it fires when
    /// a row is *removed* and never when an enum is *added* — the direction
    /// that matters. A nineteenth wire enum shipped with no pin at all and the
    /// whole file stayed green, which is the eighth vacuous-gate shape of this
    /// campaign.
    ///
    /// The in-kind fix is L7.M7's drift-gate shape: a **census** of the module's
    /// own source. Every `pub enum` in `components.rs` that derives `Serialize`
    /// reaches `.inf_lvl` bytes by construction — this file *is* the scene wire
    /// — so the census needs no judgement call and carries no exclusion list to
    /// rot.
    #[test]
    fn the_freeze_table_covers_every_wire_enum_in_this_module() {
        // The P22 CRLF law: a `.rs` read by a test is normalized first.
        //
        // **Round 3: this call did nothing.** A scripted edit ate the escapes
        // and left `replace("\n", "\n")` — spelled as two literals each holding
        // a real newline, which is why clippy's `no_effect_replace` never fired
        // (the tree was clippy-clean with it in). The census survived only
        // because `str::lines()` strips a trailing `\r` of its own accord, so
        // the normalization it names was decoration on a path that happened not
        // to need it. Restored, because the next reader of this file will
        // believe the comment.
        let src = include_str!("components.rs").replace("\r\n", "\n");
        let lines: Vec<&str> = src.lines().collect();

        let mut on_the_wire: Vec<String> = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let Some(rest) = line.strip_prefix("pub enum ") else {
                continue;
            };
            let name = rest.trim_end_matches(" {").trim().to_string();
            // Its derives sit in the few lines above, past the doc comment.
            let derives = lines[i.saturating_sub(8)..i].join("\n");
            if derives.contains("Serialize") {
                on_the_wire.push(name);
            }
        }

        assert!(
            on_the_wire.len() >= 24,
            "the census found only {} serializable enums in components.rs — it is not reading what it thinks it is, and a census that finds nothing covers everything",
            on_the_wire.len()
        );

        let pinned: std::collections::BTreeSet<&str> = FROZEN_ENUMS.iter().copied().collect();
        let missing: Vec<&String> = on_the_wire
            .iter()
            .filter(|n| !pinned.contains(n.as_str()))
            .collect();
        assert!(
            missing.is_empty(),
            "these enums reach `.inf_lvl` bytes and no row of the freeze table pins their discriminants: {missing:?}. bincode encodes an externally-tagged enum as its DECLARATION INDEX, so inserting a variant mid-list renumbers everything after it in every committed level. Add a `pin!` row and a name to FROZEN_ENUMS."
        );

        let found: std::collections::BTreeSet<&str> =
            on_the_wire.iter().map(String::as_str).collect();
        let stale: Vec<&&str> = FROZEN_ENUMS
            .iter()
            .filter(|n| !found.contains(*n))
            .collect();
        assert!(
            stale.is_empty(),
            "FROZEN_ENUMS names {stale:?}, which no longer exist in this module — a pin about nothing, which the next enum to take that name inherits"
        );
    }

    #[test]
    fn the_water_presets_are_what_they_claim() {
        let lake = WaterBody::lake(12.0, Vec2d::new(40.0, 25.0));
        assert_eq!(lake.kind, WaterKind::Lake);
        assert_eq!(lake.level_m, 12.0);
        assert_eq!(lake.extent, Vec2d::new(40.0, 25.0));
        assert!(lake.wave_amplitude_m < WaterBody::default().wave_amplitude_m);
        assert!(
            !lake.wind_from_weather,
            "a lake has no fetch — a gale must not raise a swell on it"
        );

        let river = WaterBody::river(9.0, 2.0, 3.0);
        assert_eq!(river.kind, WaterKind::River);
        assert_eq!(river.river_width_start_m, 9.0);
        assert_eq!(river.river_width_end_m, 9.0);
        assert_eq!(river.river_depth_start_m, 2.0);
        assert_eq!(river.river_flow_m_s, 3.0);
        assert!(!river.wind_from_weather);
    }

    /// The one derivation both projectors share. A host that inlined its own
    /// `if wind_from_weather` is exactly the drift the MIRROR gate exists to stop,
    /// so the rule lives on the component and is pinned here.
    #[test]
    fn effective_wind_picks_weather_or_the_body() {
        let follows = WaterBody::default();
        assert!(follows.wind_from_weather);
        assert_eq!(follows.effective_wind((13.0, -4.0)), (13.0, -4.0));

        let sheltered = WaterBody {
            wind_from_weather: false,
            wind_x: 1.5,
            wind_z: 0.25,
            ..WaterBody::default()
        };
        assert_eq!(
            sheltered.effective_wind((13.0, -4.0)),
            (1.5, 0.25),
            "a sheltered body must ignore the level's gale"
        );
    }

    // ── buoyancy (P20.2) ────────────────────────────────────────────────────

    #[test]
    fn buoyancy_serde_round_trips_and_defaults() {
        let b = Buoyancy {
            enabled: false,
            density_kg_m3: 2500.0,
            fluid_density_kg_m3: 1025.0,
            linear_drag: 0.5,
            angular_drag: 0.25,
        };
        let back: Buoyancy = serde_json::from_str(&serde_json::to_string(&b).unwrap()).unwrap();
        assert_eq!(b, back);

        // A partial record still decodes — the additive-field guarantee.
        let d: Buoyancy = serde_json::from_str("{}").unwrap();
        assert_eq!(d, Buoyancy::default());
        assert!(d.enabled, "a component you added should do something");
        assert_eq!(d.density_kg_m3, 600.0);
        assert_eq!(d.fluid_density_kg_m3, 1000.0);

        let partial: Buoyancy = serde_json::from_str(r#"{"density_kg_m3": 500.0}"#).unwrap();
        assert_eq!(partial.density_kg_m3, 500.0);
        assert_eq!(partial.fluid_density_kg_m3, 1000.0);

        // bincode round-trip (the wire the scene records ride).
        let cfg = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(b, cfg).unwrap();
        let (rt, _): (Buoyancy, usize) = bincode::serde::decode_from_slice(&bytes, cfg).unwrap();
        assert_eq!(rt, b);
    }

    /// The **contract** the physics statics tests assert against: where a body
    /// settles is `density / fluid_density`, and nothing else.
    #[test]
    fn the_equilibrium_fraction_is_the_density_ratio() {
        assert_eq!(Buoyancy::of_density(500.0).equilibrium_fraction(), 0.5);
        assert_eq!(Buoyancy::of_density(1000.0).equilibrium_fraction(), 1.0);
        assert_eq!(
            Buoyancy::of_density(2000.0).equilibrium_fraction(),
            1.0,
            "denser than water saturates at 'fully submerged' — it sinks rather \
             than settling"
        );
        assert_eq!(Buoyancy::default().equilibrium_fraction(), 0.6);
        // Sea water floats a body a touch higher than fresh.
        let sea = Buoyancy {
            fluid_density_kg_m3: 1025.0,
            ..Buoyancy::of_density(500.0)
        };
        assert!(sea.equilibrium_fraction() < 0.5);
        // A degenerate fluid is not a divide by zero.
        let vacuum = Buoyancy {
            fluid_density_kg_m3: 0.0,
            ..Buoyancy::default()
        };
        assert_eq!(vacuum.equilibrium_fraction(), 0.0);
    }

    // ── voxel volume (P21.1) ────────────────────────────────────────────────

    #[test]
    fn voxel_volume_serde_round_trips_and_defaults() {
        let v = VoxelVolume {
            asset: Some(Uuid::from_u128(0x1CE)),
            voxel_size_m: 0.25,
            runtime_carve: false,
        };
        let back: VoxelVolume = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
        assert_eq!(v, back);

        // A partial record still decodes — the additive-field guarantee.
        let d: VoxelVolume = serde_json::from_str("{}").unwrap();
        assert_eq!(d, VoxelVolume::default());
        assert_eq!(d.asset, None);
        assert_eq!(d.voxel_size_m, 0.5);
        assert!(
            d.runtime_carve,
            "P21.4's carve gate defaults to permitted — see the field docs"
        );

        let partial: VoxelVolume = serde_json::from_str(r#"{"voxel_size_m": 1.0}"#).unwrap();
        assert_eq!(partial.voxel_size_m, 1.0);
        assert!(partial.runtime_carve);

        // bincode round-trip (the wire the scene records ride).
        let cfg = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(v, cfg).unwrap();
        let (rt, _): (VoxelVolume, usize) = bincode::serde::decode_from_slice(&bytes, cfg).unwrap();
        assert_eq!(rt, v);
    }

    /// A voxel scale reachable from the Details grid but meaningless as geometry
    /// falls back to the default rather than reaching world↔grid conversion,
    /// where it would divide by zero or conjure NaN world positions.
    #[test]
    fn a_degenerate_voxel_size_falls_back_to_the_default() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let v = VoxelVolume {
                voxel_size_m: bad,
                ..VoxelVolume::default()
            };
            assert_eq!(v.effective_voxel_size_m(), 0.5, "{bad} slipped through");
        }
        let good = VoxelVolume {
            voxel_size_m: 0.125,
            ..VoxelVolume::default()
        };
        assert_eq!(good.effective_voxel_size_m(), 0.125);
        assert_eq!(
            VoxelVolume::from_asset(Uuid::from_u128(7)).asset,
            Some(Uuid::from_u128(7))
        );
    }

    /// The v20 component: every field survives both wires, a partial record
    /// still decodes (the additive guarantee), and each default is asserted
    /// against its literal rather than against `Default::default()` — which
    /// would be a tautology.
    #[test]
    fn destructible_serde_round_trips_and_defaults() {
        let d = Destructible {
            fracture_seed: 17,
            chunk_count: 24,
            strength: 3.5e7,
            density_kg_m3: 780.0,
            runtime_destruct: false,
        };
        let back: Destructible = serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap();
        assert_eq!(d, back);

        let empty: Destructible = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, Destructible::default());
        assert_eq!(empty.fracture_seed, 0);
        assert_eq!(empty.chunk_count, 12);
        assert_eq!(empty.strength, 5.0e6);
        assert_eq!(empty.density_kg_m3, 2400.0);
        assert!(
            empty.runtime_destruct,
            "the destruction gate defaults to permitted — see the field docs"
        );

        // Each field is independently addressable: a partial record keeps the
        // rest at their defaults rather than resetting the record.
        let partial: Destructible = serde_json::from_str(r#"{"strength": 4.0e8}"#).unwrap();
        assert_eq!(partial.strength, 4.0e8);
        assert_eq!(partial.chunk_count, 12);
        assert!(partial.runtime_destruct);

        let cfg = bincode::config::standard();
        let bytes = bincode::serde::encode_to_vec(d, cfg).unwrap();
        let (rt, _): (Destructible, usize) =
            bincode::serde::decode_from_slice(&bytes, cfg).unwrap();
        assert_eq!(rt, d);
        // The wire really carries all five: mutating any one moves the bytes.
        for mutate in [
            Destructible {
                fracture_seed: 18,
                ..d
            },
            Destructible {
                chunk_count: 25,
                ..d
            },
            Destructible {
                strength: 3.5e7 + 1.0,
                ..d
            },
            Destructible {
                density_kg_m3: 781.0,
                ..d
            },
            Destructible {
                runtime_destruct: true,
                ..d
            },
        ] {
            assert_ne!(
                bincode::serde::encode_to_vec(mutate, cfg).unwrap(),
                bytes,
                "a field that does not move the wire is not really persisted"
            );
        }
    }

    /// The two SI conversions the component owns, because P22.3 will assert
    /// against them rather than re-deriving its own.
    #[test]
    fn strength_and_density_convert_to_newtons_and_kilograms() {
        let d = Destructible::default();
        // 5 MPa over a 0.2 m x 0.2 m face = 5e6 * 0.04 = 200 kN.
        assert!((d.bond_force_n(0.04) - 200_000.0).abs() < 1e-6);
        // A 0.05 m3 lump of concrete weighs 120 kg.
        assert!((d.chunk_mass_kg(0.05) - 120.0).abs() < 1e-9);

        // Degenerate inputs are values, not panics: a face with no area holds
        // nothing, and a negative volume weighs nothing.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(d.bond_force_n(bad), 0.0, "area {bad}");
        }
        assert_eq!(d.chunk_mass_kg(-1.0), 0.0);
        assert_eq!(d.chunk_mass_kg(f64::NAN), 0.0);
        let broken = Destructible {
            strength: f64::NAN,
            ..d
        };
        assert_eq!(broken.bond_force_n(1.0), 0.0);
    }
}
