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
    /// Blend / transparency mode (schema v8). Additive field: `#[serde(default)]`
    /// → [`BlendMode::Opaque`], the pre-v8 behaviour.
    #[serde(default)]
    pub blend: BlendMode,
    /// Alpha-test threshold used when `blend == BlendMode::Masked`: fragments with
    /// alpha below this are discarded (schema v8). Additive field.
    #[serde(default = "default_alpha_cutoff")]
    pub alpha_cutoff: f32,
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
fn default_alpha_cutoff() -> f32 {
    0.5
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Color::new(0.8, 0.8, 0.8, 1.0),
            metallic: default_metallic(),
            roughness: default_roughness(),
            emissive: default_emissive(),
            blend: BlendMode::Opaque,
            alpha_cutoff: default_alpha_cutoff(),
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
    /// bridge skips it. An [`EntityRef`](crate::refs::EntityRef) (E-P1): reflected
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
/// A **texture GUID is deliberately absent**: the interactive viewport can't yet
/// upload asset textures (the same documented gap as [`Sprite::texture`] /
/// material previews), so a layer is proven by its solid `albedo` + a procedural
/// triplanar detail grain scaled by `tex_scale`. Per-layer albedo/normal/ORM
/// texture refs are the documented follow-up. As a nested `#[reflect(ignore)]`
/// array element it isn't surfaced in the generic Details grid (authored via the
/// paint panel / defaults); it derives `Reflect` + `Default` so the array
/// serdes and reflect-constructs.
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
        },
        TerrainLayer {
            albedo: Color::new(0.33, 0.30, 0.27, 1.0), // rock
            roughness: 0.85,
            tex_scale: 4.0,
        },
        TerrainLayer {
            albedo: Color::new(0.42, 0.30, 0.18, 1.0), // dirt
            roughness: 0.95,
            tex_scale: 5.0,
        },
        TerrainLayer {
            albedo: Color::new(0.86, 0.89, 0.94, 1.0), // snow
            roughness: 0.65,
            tex_scale: 10.0,
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
/// uncompressed so a runtime pages tiles straight out of an mmap'd `.inf_pack`
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
        }
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
    pub kind: u32,
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
        }
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
/// ## Persistence — the same v-slot gap as [`Terrain`] / [`PcgVolume`]
///
/// This is an **additive** component (registered + reflected + serde) but it is
/// **not yet a slot in the `.inf_lvl` `EntityRecord`** (frozen at v4). A spawned
/// [`SkeletalMesh`]/[`AnimPlayer`] is therefore live-session only; the v5 schema
/// migration is where the pair first persists (the same pattern the
/// `terrain_is_not_persisted_yet_v4_todo` guard pins). The guard test
/// `skeletal_components_serde_round_trip` documents the gap; **no schema bump is
/// made here.**
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
    pub fn advance(&mut self, dt: f64) {
        if !self.playing {
            return;
        }
        let next = self.t + self.speed * dt;
        self.t = if self.duration <= 0.0 {
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
/// A plain POD **mirror** of `inf_anim::SmRuntime` — kept here so the foundational
/// ECS crate needs no `inf-anim` dependency (the same boundary [`AnimPlayer`]
/// preserves by re-deriving `advance` inline). The editor Simulate loop and the
/// runtime sim — which *do* depend on `inf-anim` — convert this to/from
/// `SmRuntime` around each step. Never serialized (see [`AnimStateMachine`]).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SmRuntimeState {
    /// Active state index.
    pub current: usize,
    /// The state being cross-faded out of, if a fade is in progress.
    pub prev: Option<usize>,
    /// The outgoing state's frozen play-head.
    pub prev_time: f64,
    /// Elapsed cross-fade time (seconds).
    pub fade_t: f64,
    /// Total cross-fade duration (seconds).
    pub fade_dur: f64,
    /// Seconds spent in the current state.
    pub state_time: f64,
    /// Whether the runtime has been entered onto the machine's `entry` state.
    pub started: bool,
}

fn default_params_from_vars() -> bool {
    true
}

/// Drives an entity's [`SkeletalMesh`] from an animation state machine
/// (`.inf_sm`, P11.2) instead of a single clip: each fixed step the machine
/// evaluates its transition conditions against the actor's Blueprint variables
/// and cross-fades between states (see `inf_anim::state_machine`). An entity may
/// carry either an [`AnimPlayer`] or an `AnimStateMachine`; when both are present
/// the **state machine wins** (documented in the Simulate/runtime tick).
///
/// ## Persistence — the same v-slot gap as [`SkeletalMesh`] / [`AnimPlayer`]
///
/// Additive component (registered + reflected + serde), but **not yet a slot in
/// the `.inf_lvl` `EntityRecord`** (frozen at v4): a spawned `AnimStateMachine` is
/// live-session only until the v5 schema migration. The [`runtime`](Self::runtime)
/// field is `#[serde(skip)]` + `#[reflect(ignore)]` — rebuilt each play session,
/// never persisted (like a physics solver's transient state).
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
/// `target` is the followed entity's stable [`Guid`]. The **socket** name records
/// which authored skeleton socket the offset was baked from; in v1 the follow uses
/// the target's `GlobalTransform` composed with `offset` (the socket's bind
/// transform folded into the offset by the attach tool). Live pose-driven socket
/// tracking — evaluating the skeleton's animated joint each step — needs the
/// skeleton/clip assets in the sim world and is a documented follow-up.
///
/// Not reflected (it carries a `Guid` link, like [`ActorClass`]); authored by the
/// attach tool, shown read-only in Details.
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
/// Additive component: every field carries `#[serde(default)]`. Like the anim
/// components it is **not yet an `EntityRecord` slot** — the v6 `.inf_lvl`
/// migration is pinned (`audio_components_serde_round_trip` documents the gap); no
/// schema bump is made here.
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

#[cfg(test)]
mod tests {
    use super::*;

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
            evaluated: vec![ScatteredInstance {
                position: DVec3::new(1.0, 2.0, 3.0),
                rotation: DQuat::IDENTITY,
                scale: 1.5,
                kind: 2,
            }],
        };
        let json = serde_json::to_string(&v).unwrap();
        // The skipped cache is absent from the serialized form …
        assert!(!json.contains("evaluated"));
        let back: PcgVolume = serde_json::from_str(&json).unwrap();
        // … and decodes empty, while the persisted fields round-trip.
        assert!(back.evaluated.is_empty());
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
        // NOTE: like `Terrain`/`PcgVolume`, `SkeletalMesh`/`AnimPlayer` are NOT yet
        // slots in the v4 `.inf_lvl` `EntityRecord` — this component-level
        // round-trip is all the persistence they have until the schema-v5
        // migration (the same gap `terrain_is_not_persisted_yet_v4_todo` pins).
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
            blend: BlendMode::Translucent,
            alpha_cutoff: 0.25,
        };
        let back: Material = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(m, back);
        // Defaults for the v8 fields.
        let d = Material::default();
        assert_eq!(d.blend, BlendMode::Opaque);
        assert_eq!(d.alpha_cutoff, 0.5);
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
}
