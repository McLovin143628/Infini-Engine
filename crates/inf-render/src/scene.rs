//! The renderer's input: a flat, engine-agnostic scene description.
//!
//! Phase 2 scope: unit-cube instances with f64 world transforms (ECS binding
//! arrives in Phase 3 — the host converts whatever it has into this). The
//! `version` counter gates GPU re-uploads: bump it on any instance change.

use std::sync::Arc;

use glam::{DVec3, Mat4, Quat, Vec3};

use crate::debug_draw::DebugDraw;
use crate::primitives::PrimMesh;

pub use inf_vgeom::{VgeomMesh, VgeomSource};

pub use inf_render_2d::{
    PrebatchedRun, RenderChunk, RenderTilemap, SpriteInstance, TextureHandle, TilemapParams,
};

/// A one-shot request to upload an RGBA8 texture into the sprite pass's GPU
/// cache, keyed by [`TextureHandle`]. The pass dedups by handle, so re-listing
/// an already-uploaded texture is a cheap no-op. Straight RGBA8 rows,
/// `width*height*4` bytes, sRGB-encoded base color.
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteTextureUpload {
    pub handle: TextureHandle,
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// Reserved instance id meaning "nothing" (ID buffer clear value).
pub const ID_NONE: u32 = 0;

/// Instance ids at or above this are gizmo parts, not scene objects
/// (see `gizmo.rs`).
pub const ID_GIZMO_BASE: u32 = 0xffff_ff00;

#[derive(Debug, Clone, Copy)]
pub struct MeshInstance {
    /// World-space translation (f64 — architecture rule 3).
    pub translation: DVec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// Linear-space base color (rgba).
    pub color: [f32; 4],
    /// Metallic-roughness PBR parameters.
    pub metallic: f32,
    pub roughness: f32,
    /// Linear self-emitted color (rgb).
    pub emissive: [f32; 3],
    /// Stable pick id; `ID_NONE` is reserved, ids ≥ `ID_GIZMO_BASE` are
    /// reserved for gizmo parts.
    pub id: u32,
    /// Which built-in primitive geometry to draw (R-P1). Defaults to
    /// [`PrimMesh::Cube`], so a caller that doesn't set it — and every pre-R-P1
    /// scene — renders exactly as before.
    pub mesh: PrimMesh,
    /// Blend mode (R-P5): `0` opaque, `1` masked (alpha-test), `2` translucent
    /// (alpha-blend). Defaults to `0` so every pre-R-P5 scene renders exactly as
    /// before. Projected from the ECS `Material::blend` at the seams; drives both
    /// the bucketing partition ([`crate::passes::mesh::pack_bucketed`]) and the
    /// packed `pbr.w` the shader reads for the masked discard.
    pub blend: u8,
    /// Alpha-test threshold used when `blend == 1` (masked): fragments with base
    /// color alpha below this are discarded. Defaults to `0.5`. Packed into
    /// `pbr.z`.
    pub cutoff: f32,
}

impl MeshInstance {
    /// A plain lit **cube** instance (metallic 0, roughness 0.5, no emission,
    /// opaque) — the common case for tests and simple callers.
    pub fn lit(translation: DVec3, rotation: Quat, scale: Vec3, color: [f32; 4], id: u32) -> Self {
        Self {
            translation,
            rotation,
            scale,
            color,
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            id,
            mesh: PrimMesh::Cube,
            blend: 0,
            cutoff: 0.5,
        }
    }
}

/// A virtualized-geometry (meshlet DAG) asset referenced by one or more
/// [`VgeomInstance`]s (P13.1b, **streamed since P18.2**).
///
/// The scene carries a [`VgeomSource`] — a *lazily indexed* `.inf_vmesh`, header
/// and page directory only — rather than a decoded [`VgeomMesh`]. The renderer's
/// [`VgeomStreamer`](inf_vgeom::VgeomStreamer) pages levels of it in and out of
/// shared GPU pools against a byte budget, so a scene that references a hundred
/// meshes costs a hundred *directory parses* up front, not a hundred full
/// decodes, and VRAM tracks what the camera can actually see.
///
/// `id` is the cook-derived `.inf_vmesh` asset GUID (as a `u128`), so the host
/// keys stable content and the streamer's residency survives across frames.
///
/// [`id`]: VgeomAsset::id
#[derive(Clone)]
pub struct VgeomAsset {
    /// Stable asset id (the derived `.inf_vmesh` GUID as a `u128`).
    pub id: u128,
    /// The paged `.inf_vmesh` this asset streams from (shared; one per asset).
    pub source: Arc<VgeomSource>,
}

impl VgeomAsset {
    /// Reference an already-indexed source.
    pub fn new(id: u128, source: Arc<VgeomSource>) -> Self {
        Self { id, source }
    }

    /// Lay an in-memory [`VgeomMesh`] out as a `.inf_vmesh` image and index it —
    /// the door for tests, the editor's in-memory builds, and any host that has a
    /// decoded DAG rather than a packed asset. Identical downstream to a cooked
    /// pack: there is only one paged path.
    pub fn from_mesh(id: u128, mesh: &VgeomMesh) -> Result<Self, String> {
        Ok(Self {
            id,
            source: Arc::new(VgeomSource::from_mesh(mesh)?),
        })
    }

    /// Whole-mesh bounding sphere (local space) — read from the header, so the
    /// per-instance LOD projection never pages anything in.
    pub fn bounds(&self) -> ([f32; 3], f32) {
        self.source.bounds()
    }
}

impl std::fmt::Debug for VgeomAsset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VgeomAsset")
            .field("id", &format_args!("{:#034x}", self.id))
            .field("source", &self.source)
            .finish()
    }
}

/// One placed instance of a [`VgeomAsset`] (P13.1b) — the meshlet-path twin of a
/// [`MeshInstance`]. Multiple instances of the same `asset` share its GPU buffers;
/// the cull compute emits one visible-list entry per surviving (instance, meshlet)
/// pair. World transform is f64 (architecture rule 3); the renderer projects it to
/// an origin-relative model matrix at upload, exactly like [`MeshInstance`].
#[derive(Debug, Clone, Copy)]
pub struct VgeomInstance {
    /// Which [`VgeomAsset`] (by [`VgeomAsset::id`]) this instance draws.
    pub asset: u128,
    pub translation: DVec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// Linear-space base color (rgba).
    pub color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    /// Linear self-emitted color (rgb).
    pub emissive: [f32; 3],
    /// Stable pick id (`ID_NONE` reserved).
    pub id: u32,
}

impl VgeomInstance {
    /// A plain lit instance of `asset` (metallic 0, roughness 0.5, no emission).
    pub fn lit(
        asset: u128,
        translation: DVec3,
        rotation: Quat,
        scale: Vec3,
        color: [f32; 4],
        id: u32,
    ) -> Self {
        Self {
            asset,
            translation,
            rotation,
            scale,
            color,
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            id,
        }
    }
}

/// One vertex of a [`SkinnedMeshData`] — position + normal in **bind (rest)
/// space**, plus the four joint influences that deform it. `#[repr(C)]` + `Pod`
/// so it uploads straight to a GPU vertex buffer (56 bytes, no padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    /// Joint indices into the instance's skinning palette.
    pub joints: [u32; 4],
    /// Normalized influence weights (`Σ = 1`).
    pub weights: [f32; 4],
}

/// Bind-space geometry for a skinned mesh: an interleaved [`SkinnedVertex`]
/// buffer + a 32-bit index buffer. Referenced by [`SkinnedInstance::mesh`].
#[derive(Debug, Clone, PartialEq)]
pub struct SkinnedMeshData {
    pub vertices: Vec<SkinnedVertex>,
    pub indices: Vec<u32>,
}

/// One skinned draw: a [`SkinnedMeshData`] (by index into
/// [`RenderScene::skinned_meshes`]) placed by a world transform and deformed by a
/// per-instance **skinning palette** (`global · inverse_bind` per joint, computed
/// CPU-side by the host — v1; a GPU palette compute pass is a P15 optimization).
///
/// The palette is applied in the vertex shader **before** the model matrix, so it
/// stays in bind/model space (no floating-origin adjustment needed — only the
/// model translation is origin-relative, exactly like [`MeshInstance`]).
#[derive(Debug, Clone, PartialEq)]
pub struct SkinnedInstance {
    /// World-space translation (f64 — architecture rule 3).
    pub translation: DVec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// Linear-space base color (rgba).
    pub color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    /// Stable pick id (`ID_NONE` reserved).
    pub id: u32,
    /// Index into [`RenderScene::skinned_meshes`].
    pub mesh: usize,
    /// The skinning palette: one matrix per skeleton joint, indexed by the
    /// vertex `joints`. Bound as a `@group(3)` storage buffer.
    pub palette: Vec<Mat4>,
}

/// Directional, point, or spot light (R-P3). Spot is a point light with a cone
/// mask; its emission axis is `-direction` (see [`RenderLight`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightKind {
    Directional,
    Point,
    Spot,
}

/// A scene light in world space. The mesh pass converts point/spot positions to
/// render-local (floating-origin-relative) space at upload, exactly like
/// instance transforms.
///
/// ## Direction conventions
///
/// * [`direction`](Self::direction) is the unit vector *toward* the light for
///   [`Directional`](LightKind::Directional) — the existing convention.
/// * For [`Spot`](LightKind::Spot) the same "toward-the-light" vector is stored
///   in `direction`; the beam **emission** axis is therefore `-direction`. The
///   seams project an entity whose forward is `-Z` as `direction = rot * +Z`, so
///   emission = `-direction = rot * -Z`.
///
/// ## Shadows (R-P3 scope)
///
/// [`cast_shadows`](Self::cast_shadows) is honoured only for
/// [`Directional`](LightKind::Directional) lights, where it gates CSM caster
/// selection (see [`crate::passes::shadow`]). Point/spot shadow maps are
/// **deferred**, so the flag is inert (stored, never sampled) for those kinds.
#[derive(Debug, Clone, Copy)]
pub struct RenderLight {
    pub kind: LightKind,
    /// Linear light color.
    pub color: [f32; 3],
    /// Radiant intensity multiplier.
    pub intensity: f32,
    /// Unit direction *toward* the light (directional + spot; see the type docs
    /// — spot emission is `-direction`).
    pub direction: Vec3,
    /// World-space position (point + spot).
    pub position: DVec3,
    /// Influence radius in metres (point + spot); 0 ⇒ unbounded.
    pub range: f32,
    /// Spot inner-cone cosine (full brightness where `cos(angle) ≥ inner_cos`).
    /// Unused for directional/point. Default = `cos(30°)`.
    pub inner_cos: f32,
    /// Spot outer-cone cosine (zero where `cos(angle) ≤ outer_cos`; `outer_cos <
    /// inner_cos`). Unused for directional/point. Default = `cos(40°)`.
    pub outer_cos: f32,
    /// Whether this light casts shadows. Honoured for directional (CSM caster
    /// selection); inert for point/spot (shadow maps deferred).
    pub cast_shadows: bool,
}

impl Default for RenderLight {
    fn default() -> Self {
        Self {
            kind: LightKind::Directional,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            direction: Vec3::Y,
            position: DVec3::ZERO,
            range: 0.0,
            // 30° / 40° half-angles, mirroring the ECS `Light` cone defaults.
            inner_cos: 0.866_025_4,  // cos(30°)
            outer_cos: 0.766_044_44, // cos(40°)
            cast_shadows: true,
        }
    }
}

/// A minimal 2D light (P8.1c): a soft radial falloff in the sprite plane. The
/// sprite pass converts `position` to render-local (floating-origin-relative) at
/// upload, exactly like 3D point lights, and lights every sprite/tile/text/
/// 9-slice fragment by world-XY distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderLight2D {
    /// Linear light color.
    pub color: [f32; 3],
    /// Brightness multiplier.
    pub intensity: f32,
    /// World-space falloff radius; the contribution is `smoothstep(radius, 0,
    /// dist)`, so it is full at the light and zero at/after `radius`.
    pub radius: f32,
    /// World-space position (the sprite plane's XY is what matters).
    pub position: DVec3,
}

/// Scene-level 2D ambient term. **Defaults to white (`1,1,1`)** so that with no
/// [`RenderLight2D`] present every sprite renders exactly as before
/// (`texel·tint·1`) — the byte-stability guarantee for pre-P8.1c goldens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ambient2D(pub [f32; 3]);

impl Default for Ambient2D {
    fn default() -> Self {
        Self([1.0, 1.0, 1.0])
    }
}

/// A projected terrain tile's identity: its **asset LOD level** plus its grid
/// coordinate *at that level* (P16.3b1).
///
/// This is the renderer-local mirror of `inf_terrain::TileKey` — `inf-render` is
/// Ring 0 and deliberately does **not** depend on `inf-terrain`, so the two
/// documented projectors (`inf_viewport::host::project_terrain` and
/// `inf_player::render::project_terrain`) map one onto the other, exactly like
/// every other scene DTO.
///
/// Level `0` is the authored, full-resolution heightfield. A level-`n` tile
/// covers `2ⁿ ×` the world span at the same sample count (metres-per-sample
/// doubles per level), so level-`n` tile `(TX, TZ)` is the 2 × 2 block of
/// level-`(n−1)` tiles `(2TX+a, 2TZ+b)` decimated 2:1.
///
/// `Ord` sorts by **`lod` first, then `coord`** (matching `inf_terrain::TileKey`),
/// so a projection that emits level 0 and then each coarse level in ascending key
/// order is globally key-ascending — the order the tile list is documented to
/// arrive in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TerrainTileKey {
    /// Asset LOD level; `0` is the authored full-resolution level.
    pub lod: u32,
    /// Tile grid coordinate `(tx, tz)` **within that level**.
    pub coord: (i32, i32),
}

impl TerrainTileKey {
    /// A level-0 (authored, full-resolution) key.
    #[inline]
    pub const fn lod0(coord: (i32, i32)) -> Self {
        Self { lod: 0, coord }
    }

    /// A key at an explicit level.
    #[inline]
    pub const fn new(lod: u32, coord: (i32, i32)) -> Self {
        Self { lod, coord }
    }

    /// The coarser key one level up that contains this tile (`lod + 1`, coordinate
    /// halved with **floor** semantics so negative coordinates group correctly —
    /// tile `−1` belongs to block `−1`, not block `0`).
    #[inline]
    pub const fn parent(self) -> Self {
        Self {
            lod: self.lod + 1,
            coord: (self.coord.0.div_euclid(2), self.coord.1.div_euclid(2)),
        }
    }

    /// The four finer keys this tile decimates (`lod − 1`, coordinate doubled),
    /// in a fixed scan order. Empty at level 0 (nothing is finer).
    ///
    /// Saturating doubling: a coordinate that would overflow `i32` cannot name a
    /// real tile anyway, so the clamped key simply never matches a resident one.
    #[inline]
    pub fn children(self) -> [Self; 4] {
        let lod = self.lod.saturating_sub(1);
        let (x, z) = (
            self.coord.0.saturating_mul(2),
            self.coord.1.saturating_mul(2),
        );
        [
            Self::new(lod, (x, z)),
            Self::new(lod, (x.saturating_add(1), z)),
            Self::new(lod, (x, z.saturating_add(1))),
            Self::new(lod, (x.saturating_add(1), z.saturating_add(1))),
        ]
    }
}

/// One terrain tile handed to the [`TerrainNode`](crate::passes::terrain): its
/// [`TerrainTileKey`] (asset LOD + grid coordinate), the `f64` world origin of
/// sample `(0,0)`, the row-major `f32` height offsets (from `origin.y`), and the
/// **change stamp** the GPU cache gates its upload on. Mirrors
/// `inf_terrain::TerrainTile` but stays renderer-agnostic (the host projects it,
/// like `RenderTilemap`).
#[derive(Debug, Clone, PartialEq)]
pub struct RenderTerrainTile {
    /// Asset LOD level + grid coordinate at that level.
    pub key: TerrainTileKey,
    /// World position of sample `(0,0)` (`f64` anchor).
    pub origin: DVec3,
    /// `resolution²` row-major height offsets (metres) from `origin.y`.
    pub heights: Vec<f32>,
    /// `resolution²` row-major RGBA8 splat weights (P10.4), resolved from the
    /// tile's sparse store (an unpainted tile projects the uniform default). The
    /// terrain pass uploads these into a per-tile `Rgba8Unorm` weight texture
    /// beside the height texture. Coarse (LOD ≥ 1) tiles carry no painted weights
    /// — the pyramid is heights-only — so they project the uniform default.
    pub weights: Vec<[u8; 4]>,
    /// Inclusive `(min, max)` of `heights` (for the tile's AABB cull bound).
    pub height_bounds: (f32, f32),
    /// The tile's **monotone change stamp** (P16.3b1), projected from
    /// `inf_terrain::TerrainData::tile_version`. The GPU texture cache re-uploads
    /// this tile if — and only if — the stamp differs from the one its cached
    /// copy was built at. `0` means "no stamp" (a tile the source could not
    /// version) and is treated conservatively as *always re-upload*, never as a
    /// cache hit.
    pub version: u64,
}

/// One terrain splat material layer (P10.4), projected from the ECS
/// `TerrainLayer`. `tex_scale` is world metres per procedural detail-grain tile.
/// (Layer texture GUIDs are deferred — the viewport can't upload asset textures
/// yet; the shader proves the blend with albedo + procedural triplanar grain.)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderTerrainLayer {
    /// Linear albedo (rgba).
    pub albedo: [f32; 4],
    /// Perceptual roughness `[0, 1]`.
    pub roughness: f32,
    /// World metres per detail-grain tile.
    pub tex_scale: f32,
}

impl Default for RenderTerrainLayer {
    fn default() -> Self {
        Self {
            albedo: [0.35, 0.35, 0.35, 1.0],
            roughness: 0.9,
            tex_scale: 8.0,
        }
    }
}

/// The renderer's terrain input: the **resident** page set of a paged heightfield
/// projected from the ECS `Terrain` component. The
/// [`TerrainNode`](crate::passes::terrain) uploads a per-tile R32Float height
/// texture + a per-tile Rgba8Unorm splat-weight texture (both cached, gated by
/// each tile's own [`version`](RenderTerrainTile::version)) and assembles
/// concentric clipmap LOD rings around the camera each frame, blending the four
/// `layers` by the splat weights.
///
/// ## Residency (P16.3b1)
///
/// `tiles` is whatever the projector handed over — it is **not** assumed to be a
/// complete terrain. A missing tile simply produces no patch (a hole, exactly as
/// an unauthored tile always did), and coarse (LOD ≥ 1) pyramid tiles may ride
/// beside the level-0 ones to cover the outer rings. The renderer never invents a
/// want: it faithfully draws the set it is given (camera-driven residency
/// selection lives above this DTO).
///
/// There is deliberately **no whole-terrain version counter** (P16.3b1 removed
/// it): the per-tile stamps are strictly more precise, and a single global
/// counter is exactly the field a projector forgets to bump — the shipped player
/// pinned it to a constant, which would have frozen the GPU cache the moment
/// residency started changing. The one terrain-wide GPU upload left (the splat
/// material uniform) is gated by comparing the packed value, which cannot
/// desync.
///
/// ## Multi-terrain (P16.6)
///
/// A scene carries **N** of these ([`RenderScene::terrains`]); each is an
/// independent heightfield with its own grid, its own residency and its own splat
/// material. [`id`](Self::id) is what keeps their GPU caches apart — see there.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RenderTerrain {
    /// Stable identity of the terrain this projection describes (P16.6).
    ///
    /// The per-tile GPU texture cache and the splat-material uniform are keyed by
    /// `(id, tile key)` / `id`, so two terrains whose grids share a coordinate
    /// cannot overwrite each other's pages, and a terrain that leaves the scene
    /// releases exactly its own resources. Both projectors derive it from the
    /// terrain entity's `Guid` ([`terrain_id_from_guid`]); `0` is the "unkeyed"
    /// value a single-terrain caller (and every pre-P16.6 test) leaves at its
    /// default, which is why single-terrain scenes stay byte-identical.
    ///
    /// **Distinct terrains in one scene must carry distinct ids.** The renderer
    /// cannot check that — two projections claiming one id simply share a cache
    /// slot and fight over it.
    pub id: u64,
    /// Samples per tile side (the height/weight-texture dimension). Terrain-wide:
    /// a coarse level keeps the resolution and doubles the spacing.
    pub tile_resolution: u32,
    /// World units between samples **at level 0**.
    pub meters_per_sample: f64,
    /// The resident tiles, ascending by [`TerrainTileKey`] (level 0 first, then
    /// each coarse level) — a deterministic upload/draw order.
    pub tiles: Vec<RenderTerrainTile>,
    /// The four splat material layers the per-sample weights blend (P10.4).
    pub layers: [RenderTerrainLayer; 4],
    /// Amplitude of the large-scale fBm albedo modulation (`0` = off).
    pub macro_variation: f32,
}

impl RenderTerrain {
    /// World edge length of one **level-0** tile: `(resolution − 1) · mps`. Also
    /// the unit the clipmap ring thresholds are scaled by.
    pub fn tile_span(&self) -> f64 {
        (self.tile_resolution.max(2) as f64 - 1.0) * self.meters_per_sample
    }

    /// World edge length of a tile at asset LOD `lod`: `tile_span · 2^lod` (the
    /// metres-per-sample doubling, at a constant sample count).
    pub fn tile_span_at(&self, lod: u32) -> f64 {
        self.tile_span() * (1u64 << lod.min(62)) as f64
    }

    /// The coarsest asset LOD present in the projection (`0` for a level-0-only
    /// terrain — every inline, non-streamed terrain).
    pub fn max_lod(&self) -> u32 {
        self.tiles.iter().map(|t| t.key.lod).max().unwrap_or(0)
    }

    /// The projected `(key → change stamp)` ledger, in tile order — the input the
    /// GPU texture cache gates its uploads on.
    pub fn tile_versions(&self) -> impl Iterator<Item = (TerrainTileKey, u64)> + '_ {
        self.tiles.iter().map(|t| (t.key, t.version))
    }
}

/// Fold a terrain entity's 128-bit `Guid` into the 64-bit
/// [`RenderTerrain::id`] both projectors use (P16.6).
///
/// XOR-folding the halves, then forcing a non-zero result: `0` is reserved for
/// "unkeyed" (the default a single-terrain caller leaves in place), so a GUID that
/// happens to fold to zero must not silently become it. Pure, so the editor
/// viewport and the shipped player derive the same id for the same entity — which
/// is what makes a PIE-vs-shipping comparison of the projected scene meaningful.
#[inline]
pub fn terrain_id_from_guid(guid: u128) -> u64 {
    let folded = (guid as u64) ^ ((guid >> 64) as u64);
    if folded == 0 {
        1
    } else {
        folded
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyParams {
    /// Linear colors.
    pub zenith: [f32; 3],
    pub horizon: [f32; 3],
    pub ground: [f32; 3],
}

impl Default for SkyParams {
    fn default() -> Self {
        // Editor-dark sky tuned to the infinity-dark theme.
        Self {
            zenith: [0.012, 0.021, 0.038],
            horizon: [0.055, 0.081, 0.120],
            ground: [0.009, 0.011, 0.015],
        }
    }
}

/// The sun this renderer shipped with from Phase 2 to Phase 16, as a
/// compile-time `camera::SUN_DIR` constant. **P17.1 retired the constant**: the
/// direction is now projected from the scene's `TimeOfDay` + `SkyAtmosphere`
/// components, and this value survives only as [`SunParams::default`] — the
/// fallback a scene with no time-of-day authority (every unit test, every
/// pre-P17.1 golden, a bare `RenderScene::default()`) still renders with, so
/// those pixels are byte-identical to what they always were.
///
/// Kept **un-normalized**, exactly as the deleted constant was: every one of its
/// three call sites wrote `SUN_DIR.normalize()`, and
/// [`SunParams::unit_direction`] does the same multiplication on the same bits.
/// Hand-transcribing the normalized triple would risk a 1-ULP drift that moves
/// every pre-P17.1 golden, so the arithmetic is reproduced rather than the
/// result — pinned by `default_sun_is_the_retired_constant_normalized`, which
/// compares raw `to_bits()`.
pub const DEFAULT_SUN_DIR: Vec3 = Vec3::new(0.45, 0.75, 0.3);

/// The scene's **sun and moon** (P17.1) — direction, colour and intensity for
/// each, projected from the `TimeOfDay` + `SkyAtmosphere` component pair by both
/// scene builders (`inf_viewport::host` and `inf_player::render`).
///
/// This is the renderer's single source of a sun direction. It feeds:
///
/// * `ViewUniforms::sun_dir` — read by the sky gradient's glow, the terrain
///   shader, and the mesh/skinned/vgeom shaders' no-light fallback;
/// * the CSM caster fallback ([`crate::passes::shadow`]) when a scene has no
///   directional light at all;
/// * the GI sun fallback ([`crate::passes::gi`]), so probe radiance tracks the
///   time of day.
///
/// A scene that authors its own directional light still wins over the fallbacks —
/// exactly the precedence the renderer had before P17.1, with the constant
/// swapped for a projected value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunParams {
    /// Unit direction **toward** the sun (same convention as
    /// [`RenderLight::direction`]).
    pub direction: Vec3,
    /// Linear sun colour.
    pub color: [f32; 3],
    /// Sun radiant-intensity multiplier.
    pub intensity: f32,
    /// Unit direction **toward** the moon.
    pub moon_direction: Vec3,
    /// Linear moon colour.
    pub moon_color: [f32; 3],
    /// Moon radiant-intensity multiplier (used while the sun is below the
    /// horizon).
    pub moon_intensity: f32,
    /// Lunar phase, `[0, 1)` — `0` new, `0.5` full (P17.2). Only the sky pass
    /// reads it, to place the moon disc's terminator; it lights nothing.
    pub moon_phase: f32,
}

impl Default for SunParams {
    /// The retired [`DEFAULT_SUN_DIR`] at the intensity the shaders' hard-coded
    /// fallback used (`3.0`), so a scene that never opts into time of day renders
    /// exactly the pixels it rendered before P17.1. `direction` is the raw
    /// constant; consumers read [`unit_direction`](SunParams::unit_direction).
    fn default() -> Self {
        Self {
            direction: DEFAULT_SUN_DIR,
            color: [1.0, 1.0, 1.0],
            intensity: 3.0,
            // Straight down — a moon nobody projected is a moon nobody sees. The
            // renderer never reads it unless the projection filled it in.
            moon_direction: Vec3::NEG_Y,
            moon_color: [0.62, 0.72, 1.0],
            moon_intensity: 0.0,
            // Full — the phase a projector that never filled it in would want if
            // anything ever drew this moon, which nothing does at intensity 0.
            moon_phase: 0.5,
        }
    }
}

impl SunParams {
    /// The sun direction as a unit vector — what every consumer actually reads.
    ///
    /// A projector that hands over a degenerate (zero / non-finite) vector gets
    /// the retired default rather than a `NaN` uniform, which would otherwise
    /// black out the sky glow and the CSM cascade fit.
    pub fn unit_direction(&self) -> Vec3 {
        let d = self.direction.normalize_or_zero();
        if d.length_squared() > 0.5 {
            d
        } else {
            DEFAULT_SUN_DIR.normalize()
        }
    }

    /// The moon direction as a unit vector; degenerate input reads as "straight
    /// down", i.e. a moon below the world.
    pub fn unit_moon_direction(&self) -> Vec3 {
        let d = self.moon_direction.normalize_or_zero();
        if d.length_squared() > 0.5 {
            d
        } else {
            Vec3::NEG_Y
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RenderScene {
    pub instances: Vec<MeshInstance>,
    /// Bind-space geometry for skinned meshes (P11.1), referenced by
    /// [`SkinnedInstance::mesh`]. Empty ⇒ the skinned pass is a no-op (every
    /// pre-P11 scene stays byte-identical).
    ///
    /// **`Arc`, since P18.3, and for two reasons that are really one.** A host
    /// rebuilds this list on every projection; a character's bind-space stream is
    /// megabytes, so an owned `SkinnedMeshData` meant a full CPU copy per document
    /// change *and* — because the pass keyed its uploads on `scene.version` — a
    /// full GPU re-upload of geometry that had not moved. Sharing the buffer lets
    /// the pass cache by **pointer identity** instead: same `Arc`, same GPU
    /// buffers, no copy and no upload. Palettes are still rebuilt every projection,
    /// which is correct — they are the part that actually changes.
    pub skinned_meshes: Vec<std::sync::Arc<SkinnedMeshData>>,
    /// GPU-skinned instances (P11.1). Each carries its own joint palette; drawn
    /// by the skinned mesh pass after the rigid mesh pass, into the same targets.
    pub skinned: Vec<SkinnedInstance>,
    /// Virtualized-geometry (meshlet DAG) assets referenced by
    /// [`vgeom_instances`](Self::vgeom_instances) (P13.1b). Each is uploaded to GPU
    /// storage buffers once (cached by [`VgeomAsset::id`]). Empty ⇒ the meshlet
    /// pass is a no-op (every scene without vmesh content stays byte-identical).
    pub vgeom_assets: Vec<VgeomAsset>,
    /// Placed meshlet-path instances (P13.1b). Drawn by the GPU-driven
    /// [`crate::passes::vgeom`] pass (cull+LOD compute → vertex-pulled indirect
    /// draw) when [`RenderSettings::vgeom`](crate::RenderSettings) is enabled,
    /// after the rigid mesh pass, into the same MSAA targets.
    pub vgeom_instances: Vec<VgeomInstance>,
    /// Scene lights (directional + point). Empty ⇒ the shader falls back to a
    /// default editor sun so unlit demo scenes still render.
    pub lights: Vec<RenderLight>,
    /// 2D sprites (batched + drawn by the sprite pass over the 3D scene).
    pub sprites: Vec<SpriteInstance>,
    /// Heightfield terrains (P10.1; **N of them** since P16.6). The terrain pass
    /// draws each one's clipmap LOD rings around the camera, in list order; an
    /// empty list ⇒ the pass is a no-op (so scenes without terrain — every
    /// pre-P10.1 golden — stay byte-identical). Each tile's own
    /// [`version`](RenderTerrainTile::version) stamp gates its height/weight
    /// texture upload (P16.3b1), keyed per terrain by
    /// [`RenderTerrain::id`] (P16.6).
    ///
    /// Ordering is the projector's, and it is what the draw order follows — so it
    /// must be deterministic. It is **not** the same order in both projectors: the
    /// player walks its world in `Guid` order, the editor viewport walks the
    /// document's own entity order. Each is deterministic for its own side, which
    /// is what a per-side determinism gate needs; what makes a *cross-side*
    /// comparison meaningful is [`id`](RenderTerrain::id), which both derive from
    /// the terrain entity's `Guid`, so a PIE-vs-shipping diff matches terrains up
    /// by identity rather than by position in a list.
    ///
    /// The terrains are independent: one's residency, grid and splat material say
    /// nothing about another's, and they may legitimately overlap in world space
    /// (the depth test resolves it, exactly as it does for two meshes).
    pub terrains: Vec<RenderTerrain>,
    /// 2D tilemaps (P8.1b). The sprite pass culls each tilemap's chunks against
    /// the camera and expands the visible ones into prebatched sprite runs, then
    /// batches them together with the loose `sprites`. Because culling depends on
    /// the live camera, the pass re-expands tilemaps every frame (not gated by
    /// `version`) while any tilemap is present — a documented v1 cost (a
    /// camera-delta / dirty-region optimization is a follow-up).
    pub tilemaps: Vec<RenderTilemap>,
    /// Host-expanded 2D primitives that are already in draw order and share one
    /// `(layer, order, texture)` per run — 9-slices (nine quads) and text blocks
    /// (one quad per glyph), expanded by `inf-render-2d`. The sprite pass merges
    /// these with the loose `sprites` and the tilemap runs in one painter sort.
    /// Version-gated (the host rebuilds them on document change), unlike
    /// tilemaps which additionally re-expand per frame for culling.
    pub prebatched: Vec<PrebatchedRun>,
    /// Minimal 2D lights (P8.1c). Empty ⇒ only `ambient_2d` lights the sprites.
    pub lights_2d: Vec<RenderLight2D>,
    /// Scene-level 2D ambient (default white → unlit sprites unchanged).
    pub ambient_2d: Ambient2D,
    /// Textures to hand to the sprite pass's GPU cache (drained/deduped by
    /// handle). The host populates this once per newly-referenced texture.
    pub pending_texture_uploads: Vec<SpriteTextureUpload>,
    /// Bump on every change to `instances`/`lights`/`sprites`/`tilemaps` — gates
    /// buffer re-upload (tilemaps additionally re-expand per frame for culling).
    pub version: u64,
    pub sky: SkyParams,
    /// The sun and moon (P17.1). Defaults to the retired `SUN_DIR` constant, so a
    /// scene whose projector found no time-of-day authority renders exactly as it
    /// did before. See [`SunParams`].
    pub sun: SunParams,
    /// The physically-based atmosphere (P17.2): the LUT-driven sky, the sun/moon
    /// discs, the starfield, aerial perspective and height fog. **Disabled** by
    /// default, so a scene with no time-of-day authority draws the P17.1 gradient
    /// and its lit passes take the byte-identical no-atmosphere arithmetic. See
    /// [`crate::atmosphere::AtmosphereParams`].
    pub atmosphere: crate::atmosphere::AtmosphereParams,
    pub grid_enabled: bool,
    /// Ids drawn with the selection outline.
    pub selected: Vec<u32>,
    /// Id drawn with the hover outline (weaker), if any.
    pub hovered: Option<u32>,
    /// Immediate-mode debug lines, rebuilt by the host each frame
    /// (render-local space — not gated by `version`).
    pub debug: DebugDraw,
}

impl RenderScene {
    pub fn mark_dirty(&mut self) {
        self.version = self.version.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_sun_is_the_retired_constant_normalized() {
        // `DEFAULT_SUN_DIR` must be `normalize(0.45, 0.75, 0.3)` — the value the
        // deleted `camera::SUN_DIR` produced at every one of its call sites. This
        // identity is the whole reason every pre-P17.1 golden stays byte-identical.
        let legacy = Vec3::new(0.45, 0.75, 0.3).normalize();
        let s = SunParams::default();
        assert_eq!(
            s.unit_direction().to_array().map(f32::to_bits),
            legacy.to_array().map(f32::to_bits),
            "the default sun moved — every pre-P17.1 golden would move with it"
        );
        assert!((s.unit_direction().length() - 1.0).abs() < 1e-6);
        assert_eq!(RenderScene::default().sun, s);
        // A scene that never opts in never lights anything with the moon.
        assert_eq!(s.moon_intensity, 0.0);
        assert_eq!(s.unit_moon_direction(), Vec3::NEG_Y);
    }

    #[test]
    fn degenerate_sun_direction_falls_back() {
        let s = SunParams {
            direction: Vec3::ZERO,
            moon_direction: Vec3::ZERO,
            ..SunParams::default()
        };
        assert_eq!(s.unit_direction(), DEFAULT_SUN_DIR.normalize());
        assert_eq!(s.unit_moon_direction(), Vec3::NEG_Y);
        let nan = SunParams {
            direction: Vec3::splat(f32::NAN),
            ..SunParams::default()
        };
        assert!(nan.unit_direction().is_finite());
    }

    #[test]
    fn dirty_bumps_version() {
        let mut s = RenderScene::default();
        let v0 = s.version;
        s.instances.push(MeshInstance::lit(
            DVec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
            [1.0; 4],
            1,
        ));
        s.mark_dirty();
        assert_ne!(s.version, v0);
    }
}
