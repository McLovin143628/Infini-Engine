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
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::math::{Color, Vec2d, Vec3d};

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

/// A renderable mesh reference. Phase 4 adds an asset-GUID variant.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[reflect(Component, Default)]
pub struct MeshRef {
    pub primitive: Primitive,
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

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Color::new(0.8, 0.8, 0.8, 1.0),
            metallic: default_metallic(),
            roughness: default_roughness(),
            emissive: default_emissive(),
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
}

impl Default for Light {
    fn default() -> Self {
        Self {
            kind: LightKind::Point,
            color: Color::WHITE,
            intensity: 1.0,
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
}

impl Default for RigidBody3D {
    fn default() -> Self {
        Self {
            kind: BodyKind3D::Static,
            gravity_scale: default_gravity_scale(),
            fixed_rotation: false,
            linear_damping: 0.0,
            angular_damping: 0.0,
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
