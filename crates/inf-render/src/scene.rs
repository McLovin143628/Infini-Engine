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

// ── GPU-instanced scatter (P18.5) ────────────────────────────────────────────

/// One scattered instance as a **host** authors it: world position, orientation,
/// uniform scale, tint (P18.5).
///
/// This is the shape both projectors already had in hand — `PcgVolume::evaluated`
/// and `Foliage::instances` — and it is *not* what reaches the GPU. Packing into
/// [`ScatterInstanceRaw`] happens once, in [`ScatterData::build`], which is Ring 0
/// so the editor viewport and the shipped player cannot pack differently.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScatterInstance {
    /// World-space position (f64 — architecture rule 3).
    pub position: DVec3,
    pub rotation: Quat,
    /// Uniform scale.
    pub scale: f32,
    /// Linear-space tint (rgba).
    pub color: [f32; 4],
}

/// One scattered instance as the **GPU** stores it (P18.5) — 48 bytes, `Pod`, and
/// deliberately **origin-independent**.
///
/// `offset` is relative to the batch's [`ScatterBatch::anchor`], not to the
/// floating origin. That is the load-bearing part: a render-local pack would be
/// invalidated by every origin rebase, so a camera flying across the world would
/// re-upload every instance buffer it can see. Anchor-relative offsets are a pure
/// function of the *content*, so the buffer is uploaded once per content change
/// and the anchor rides in a per-frame uniform instead.
///
/// Precision: f32 relative to the batch anchor, so a batch spanning 1 km resolves
/// to ~6e-5 m and one spanning 100 km to ~6e-3 m. Scatter volumes are authored at
/// tens-to-hundreds of metres (`PcgVolume::extent` defaults to 50 m), so this is
/// several orders of margin; a single batch covering a continent would not be.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScatterInstanceRaw {
    /// Position relative to the batch anchor (metres).
    pub offset: [f32; 3],
    /// Uniform scale.
    pub scale: f32,
    /// Orientation quaternion, `xyzw`.
    pub rotation: [f32; 4],
    /// Linear-space tint (rgba).
    pub color: [f32; 4],
}

/// The immutable, content-addressed payload of a [`ScatterBatch`] (P18.5).
///
/// **Content-keyed, like everything else since P18.3.** [`key`](Self::key) is an
/// `xxh3` 128-bit hash over the packed instance bytes *and* the primitive kind, so
/// two batches with identical geometry share one GPU upload (two foliage entities
/// painted from the same stroke, or the editor and the player rendering the same
/// level) and a changed batch is a *different* asset rather than a stale one under
/// a reused id. The renderer's `GenCache` keys on it directly; nothing hashes per
/// frame.
///
/// The hash is folded **while packing**, in one pass over data the projector was
/// building anyway — and the path it replaces built one 96-byte `MeshInstance` per
/// scattered instance and pushed it into `RenderScene::instances`, so this is
/// strictly cheaper than what it supersedes.
#[derive(Debug, Clone, PartialEq)]
pub struct ScatterData {
    /// Which built-in primitive every instance of this batch draws.
    pub mesh: PrimMesh,
    /// Packed, anchor-relative instance records in authored order.
    pub instances: Vec<ScatterInstanceRaw>,
    key: u128,
    max_scale: f32,
}

impl ScatterData {
    /// Pack world-space instances into an anchor-relative GPU payload and derive
    /// the content key.
    ///
    /// Deterministic: the output is a pure function of `(mesh, anchor, instances)`
    /// in authored order, with no floating-origin, camera or frame input.
    pub fn build(
        mesh: PrimMesh,
        anchor: DVec3,
        instances: impl IntoIterator<Item = ScatterInstance>,
    ) -> Self {
        let mut packed = Vec::new();
        let mut max_scale: f32 = 0.0;
        for i in instances {
            let o = i.position - anchor;
            max_scale = max_scale.max(i.scale.abs());
            packed.push(ScatterInstanceRaw {
                offset: [o.x as f32, o.y as f32, o.z as f32],
                scale: i.scale,
                rotation: i.rotation.to_array(),
                color: i.color,
            });
        }
        // The primitive kind is part of the identity: the same placements drawn as
        // cubes and as spheres are two different batches, and an id that did not
        // say so would serve one from the other's cached buffers.
        let mut bytes = Vec::with_capacity(packed.len() * 48 + 4);
        bytes.extend_from_slice(&(mesh as u32).to_le_bytes());
        bytes.extend_from_slice(bytemuck::cast_slice(&packed));
        let key = xxhash_rust::xxh3::xxh3_128(&bytes);
        Self {
            mesh,
            instances: packed,
            key,
            max_scale,
        }
    }

    /// The content key — the renderer's GPU-cache identity for this payload.
    pub fn key(&self) -> u128 {
        self.key
    }

    /// The largest uniform scale in the batch. Multiplied by the primitive's own
    /// bounding radius it gives one conservative cull radius for every instance,
    /// which is what lets the cull compute carry a single scalar instead of a
    /// per-instance one.
    pub fn max_scale(&self) -> f32 {
        self.max_scale
    }

    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

/// A GPU-scattered instance batch (P18.5) — the render-side form of a
/// `PcgVolume`'s evaluated cache or a `Foliage` component's placed instances.
///
/// Before this batch both projectors expanded those lists into one
/// [`MeshInstance`] each and pushed them onto [`RenderScene::instances`], so a
/// 100k-instance scatter cost 100k CPU-side structs, 100k packed
/// `InstanceRaw`s and a per-frame vertex-buffer upload of ~17 MB. Now the payload
/// is uploaded **once per content change** into a storage buffer and culled
/// per-instance on the GPU (frustum + the P18.1 HZB), with distance bands
/// selecting full mesh / impostor / nothing.
///
/// The batch is one *object* as far as selection is concerned — a scatter is
/// authored, moved and deleted as a whole — so it carries one pick [`id`](Self::id)
/// rather than one per instance.
#[derive(Debug, Clone, PartialEq)]
pub struct ScatterBatch {
    /// The content-addressed payload. `Arc` so a projection that re-runs without
    /// the scatter changing costs a pointer copy.
    pub data: Arc<ScatterData>,
    /// World-space anchor the payload's offsets are relative to. **Not** part of
    /// the content key: a batch whose anchor moves (an interpolated actor) keeps
    /// its buffer and only its uniform changes.
    pub anchor: DVec3,
    pub metallic: f32,
    pub roughness: f32,
    /// Linear self-emitted color (rgb).
    pub emissive: [f32; 3],
    /// Stable pick id for the whole batch (`ID_NONE` reserved).
    pub id: u32,
    /// Authored draw distance in metres; `0` ⇒ unlimited. Clamps the renderer's
    /// own impostor/cull band **down**, never up — content may ask for less detail
    /// than the tier allows, never for more.
    ///
    /// This is the one *content* LOD knob (`PcgVolume::draw_distance`, which has
    /// existed since P10.5). Honouring it inside the cull compute is what finally
    /// makes both hosts agree about it: the editor used to cull against its own
    /// camera eye on the CPU while the player ignored the field entirely, so a
    /// shipped build drew strictly more scatter than its preview.
    pub draw_distance: f64,
}

impl ScatterBatch {
    /// A plain lit batch (metallic 0, no emission, unlimited draw distance).
    pub fn lit(data: Arc<ScatterData>, anchor: DVec3, roughness: f32, id: u32) -> Self {
        Self {
            data,
            anchor,
            metallic: 0.0,
            roughness,
            emissive: [0.0; 3],
            id,
            draw_distance: 0.0,
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
    /// `resolution²` row-major **biome ids** (P19.2), one `u8` per sample, resolved
    /// from the tile's sparse store exactly like [`weights`](Self::weights) — this
    /// vec is ALWAYS dense, the projector expands the sparse default. A tile that
    /// has never been painted projects all-`0` (`inf_terrain::UNASSIGNED_BIOME`),
    /// and coarse (LOD ≥ 1) pyramid pages carry no painted data at all — the
    /// pyramid is heights-only — so they project that same uniform default. The
    /// terrain pass uploads these into a per-tile `R8Uint` texture beside the
    /// height + weight ones; only the Biomes view mode reads it.
    ///
    /// A biome id is **categorical**: it indexes
    /// [`RenderTerrain::biome_palette`] and is never filtered or interpolated (the
    /// shader loads it nearest — the midpoint of ids 1 and 3 is not id 2).
    pub biomes: Vec<u8>,
    /// `ceil(resolution / 32) * resolution` **row-packed hole bits** (P21.2), or
    /// **empty** for a tile nothing has carved. Word `w + j * words_per_row` holds
    /// the bits for samples `[w*32, w*32+32)` of row `j`, LSB-first.
    ///
    /// Row-packed, not tile-packed like `inf_terrain::TerrainTile`'s own mask: the
    /// fragment's index becomes `(i >> 5, j)`, which needs no division by the
    /// resolution and maps one-to-one onto an R32Uint texture's texel grid. The
    /// projector does that repack, once per carved tile per edit, and only for
    /// tiles that have holes at all — a hole-free tile projects an empty `Vec`
    /// and the pass binds a 1x1 zero texture for it (four bytes, no permutation).
    ///
    /// Empty is therefore not "unknown", it is **"nothing is holed"** — the same
    /// sparse-default rule the source layer carries, projected intact.
    pub holes: Vec<u32>,
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

/// **Fill every voxel volume's P21.2 seam terms from the heightfields beside
/// them**, and arm the blend at `band_m` metres.
///
/// The one implementation both host projectors call, and that is the point
/// rather than a convenience: the editor viewport and the shipped player must
/// agree pixel for pixel about where a cave mouth stops being cave, and two
/// hand-synced copies of a per-vertex loop is exactly the shape that eventually
/// does not. (Same reasoning as [`RenderTerrain::seam_sample`] living on the
/// DTO.) A host calls this once, after it has projected both halves.
///
/// A vertex over **no** terrain — or over a holed sample — keeps
/// [`RenderVoxelVertex::NO_SEAM`] and blends nothing, which is what makes the
/// mouth of a cave shade into the hillside while its interior does not. Passing
/// `band_m <= 0` disarms every volume, so a host with no terrain at all produces
/// byte-identical frames to its pre-P21.2 self.
///
/// Cost is one `seam_sample` per voxel vertex per projection, and projections
/// happen on a change stamp, not per frame.
///
/// Terrains are consulted in projection order and the **first** one that answers
/// wins. Overlapping heightfields are already outside what the clipmap can draw
/// coherently, so there is no better rule available here — only a deterministic
/// one, which this is. A terrain that answers and is then vetoed (below) has
/// still answered: the search does not fall through to the next heightfield,
/// because "the ground here is carved away" is an answer.
///
/// # The mask-free veto, and where it applies
///
/// [`RenderTerrain::seam_sample`] reads the residency floor, so on a **streamed**
/// terrain it reads a coarse pyramid page — which carries no hole mask, so its
/// poison rule cannot fire (see
/// [`seam_holes_are_known`](RenderTerrain::seam_holes_are_known)). Blending
/// anyway would put grass on a cave ceiling: at a mouth the roof is *at* the
/// heightfield's surface, which is exactly where the band is widest.
///
/// So where the mask is missing, one rule that needs no mask stands in for it: a
/// voxel surface only *continues* a heightfield if it faces the same way it does,
/// `dot(vertex normal, heightfield normal) > 0`. A cave roof faces down and a
/// heightfield normal always faces up, so a roof is refused; a cave wall is
/// perpendicular and is refused too; the mouth's outward-turning lip — the one
/// surface the blend exists for — faces up and keeps its seam.
///
/// It is deliberately applied **only** where the mask is absent, and that is not
/// timidity: it is strictly weaker than the mask (it cannot see a hole in flat
/// ground at all), so using it in place of a mask that *is* present would trade a
/// correct answer for an approximate one — and would silently move every existing
/// golden, which has an inline terrain and therefore a mask.
pub fn apply_seam(volumes: &mut [RenderVoxelVolume], terrains: &[RenderTerrain], band_m: f32) {
    if band_m <= 0.0 || terrains.is_empty() {
        return;
    }
    for volume in volumes {
        volume.seam_band_m = band_m;
        for chunk in &mut volume.chunks {
            let base = chunk.origin;
            for v in &mut chunk.vertices {
                let wx = base.x + v.pos[0] as f64;
                let wz = base.z + v.pos[2] as f64;
                let Some((terrain, sample)) = terrains
                    .iter()
                    .find_map(|t| t.seam_sample(wx, wz).map(|s| (t, s)))
                else {
                    continue;
                };
                if !terrain.seam_holes_are_known() && !continues_surface(v.normal, sample.normal) {
                    continue;
                }
                // `origin.y` of the chunk is already the anchor the vertex
                // positions are relative to, so the packed height is measured
                // in the same space the shader compares it against.
                let (nh, albedo) = sample.pack(base.y);
                v.seam_nh = nh;
                v.seam_albedo = albedo;
            }
        }
    }
}

/// Does a voxel surface with normal `voxel_n` *continue* a heightfield whose
/// surface normal there is `terrain_n`? — the mask-free half of the poison rule
/// (see [`apply_seam`]).
///
/// Strictly positive, not `>= 0`: a perpendicular surface (a cave wall against
/// flat ground) is refused. Blending it would smear hillside down a vertical face
/// for the band's whole width, which is the loudest form of the artefact this
/// guards, and a wall is not a continuation of the ground it is cut into.
#[inline]
fn continues_surface(voxel_n: [f32; 3], terrain_n: [f32; 3]) -> bool {
    voxel_n[0] * terrain_n[0] + voxel_n[1] * terrain_n[1] + voxel_n[2] * terrain_n[2] > 0.0
}

/// What the heightfield looks like at one world `(x, z)`, for the P21.2 voxel
/// seam blend — the output of [`RenderTerrain::seam_sample`].
///
/// Deliberately the *resolved* surface (a normal, a height, one blended colour)
/// rather than the inputs it was resolved from. The consumer is a vertex
/// attribute, and handing four weights plus a palette across that boundary would
/// mean the voxel shader carrying the terrain's layers — a second place for the
/// blend to be implemented, and a second place for it to be wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeamSample {
    /// Unit heightfield normal (+Y strictly positive).
    pub normal: [f32; 3],
    /// World surface height (`f64`, the world-space precision doctrine).
    pub height: f64,
    /// Splat-blended linear albedo.
    pub albedo: [f32; 3],
    /// Splat-blended perceptual roughness.
    pub roughness: f32,
}

impl SeamSample {
    /// Pack into the two [`RenderVoxelVertex`] seam attributes, given the
    /// vertex's own world position (only its Y matters — the shader measures the
    /// band as `|world.y - height|`).
    ///
    /// The height rides as `f32` because it is only ever used as a *difference*
    /// against an already render-local vertex Y; the `f64` above is the world
    /// value, and the projector subtracts the floating origin before calling this.
    pub fn pack(&self, origin_y: f64) -> ([f32; 4], [f32; 4]) {
        (
            [
                self.normal[0],
                self.normal[1],
                self.normal[2],
                (self.height - origin_y) as f32,
            ],
            [
                self.albedo[0],
                self.albedo[1],
                self.albedo[2],
                self.roughness,
            ],
        )
    }
}

impl RenderTerrainTile {
    /// Words per packed hole-mask row: `ceil(resolution / 32)`. Zero-width for a
    /// degenerate resolution, which cannot happen through a projector but keeps
    /// the arithmetic total.
    #[inline]
    pub fn hole_words_per_row(resolution: u32) -> u32 {
        resolution.div_ceil(32)
    }

    /// `true` when **some** sample of this tile is holed — the cheap gate the GPU
    /// cache takes before sizing a hole texture at all.
    #[inline]
    pub fn has_holes(&self) -> bool {
        self.holes.iter().any(|&w| w != 0)
    }

    /// Is sample `(i, j)` holed? An empty mask reads `false` everywhere — the
    /// sparse default, projected intact. Out-of-range words read `false` too,
    /// matching the shader, whose `textureLoad` past the texture returns zero.
    #[inline]
    pub fn is_hole(&self, resolution: u32, i: u32, j: u32) -> bool {
        if self.holes.is_empty() {
            return false;
        }
        let stride = Self::hole_words_per_row(resolution) as usize;
        match self.holes.get(j as usize * stride + (i >> 5) as usize) {
            Some(word) => word & (1u32 << (i & 31)) != 0,
            None => false,
        }
    }

    /// The splat weight at sample `(i, j)`, clamping out-of-range indices to the
    /// edge. The projected `weights` vec is always dense (the projector expands
    /// the source's sparse default), so this never resolves a default itself —
    /// but an empty vec still answers, uniformly layer 0, rather than panicking on
    /// a projection that forgot.
    #[inline]
    pub fn weight_sample(&self, resolution: u32, i: u32, j: u32) -> [u8; 4] {
        if self.weights.is_empty() {
            return [255, 0, 0, 0];
        }
        let r = resolution.max(1);
        let idx = (j.min(r - 1) * r + i.min(r - 1)) as usize;
        self.weights.get(idx).copied().unwrap_or([255, 0, 0, 0])
    }
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
/// texture, a per-tile Rgba8Unorm splat-weight texture and a per-tile R8Uint
/// biome-id texture (all cached, gated by each tile's own
/// [`version`](RenderTerrainTile::version)) and assembles concentric clipmap LOD
/// rings around the camera each frame, blending the four `layers` by the splat
/// weights.
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
/// residency started changing. The terrain-wide GPU uploads left (the splat
/// material uniform and the P19.2 biome palette) are each gated by comparing the
/// packed value, which cannot desync.
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
    /// Biome id → debug colour for the **Biomes** view mode (P19.2), **indexed by
    /// id**: `biome_palette[id]` is the colour a sample carrying that id is tinted
    /// with. The array position IS the id — never an ordinal into the level's
    /// biome list, which is a sparse set of authored ids. Slot 0 is the reserved
    /// "unassigned" colour, and an id the set never defined reads it too;
    /// `inf_terrain::BiomeSet::palette` builds exactly that shape.
    ///
    /// **May be empty**: a terrain with no `BiomeSet` bound — or a projector that
    /// cannot resolve one — passes `Vec::new()` and the renderer pads every slot
    /// with the unassigned colour, so the mode still draws something honest
    /// (uniform neutral grey = "no biome vocabulary here") instead of failing.
    /// Ids past the end of a short palette pad the same way.
    pub biome_palette: Vec<[f32; 4]>,
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

    /// **The P21.2 seam sample**: the heightfield's surface normal, world height
    /// and splat-blended material at world `(x, z)`, or `None` where this
    /// projection has no level-0 surface there — no resident tile, or a tile whose
    /// sample is holed.
    ///
    /// This is what a projector calls per voxel vertex to fill
    /// [`RenderVoxelVertex::seam_nh`] / [`seam_albedo`](RenderVoxelVertex::seam_albedo),
    /// and it lives **here**, on the already-projected terrain, for a specific
    /// reason: the alternative is for each of the two host projectors to sample
    /// `inf_terrain` and re-implement the splat blend, which would make the seam
    /// colour a function of *which host is drawing* — exactly the class of
    /// divergence the mirrored-pair discipline exists to prevent. One
    /// implementation, over the DTO both hosts already built, cannot diverge.
    ///
    /// ## THE RESIDENCY FLOOR, and not the finest resident page (P21.2 audit)
    ///
    /// This reads **only** the projection's coarsest asset level
    /// ([`max_lod`](Self::max_lod)) — the same restriction, for the same reason,
    /// that [`gi::voxelization_tiles`](crate::gi::voxelization_tiles) puts on the
    /// GI voxelizer. [`tiles`](Self::tiles) is the streamer's **camera-driven**
    /// working set, so a seam resolved against "whichever level-0 page happens to
    /// be paged in" makes a voxel surface's albedo, roughness and shading normal —
    /// and therefore its **lighting** — a function of where the camera has *been*.
    /// That is the P18 law (`camera-driven residency never feeds lighting`), and
    /// the first cut of this function broke it: level 0 is exactly the part of the
    /// cut that pages, so walking away from a cave mouth silently turned its blend
    /// off.
    ///
    /// The coarsest level is the terrain's always-resident root:
    /// `inf_terrain::TerrainStreamer` pins it as its **residency floor** and
    /// reseeds the published cut from it, which is what makes `max_lod` a property
    /// of the *asset* rather than of the camera (see `TerrainStreamer::
    /// residency_floor`). An inline, non-streamed terrain has `max_lod() == 0`, so
    /// for every such terrain — every unit test, every golden, every level that has
    /// not been through the Terrain Import wizard — this is byte-identical to
    /// sampling level 0, because level 0 *is* the coarsest level.
    ///
    /// **Two consequences on a streamed terrain, stated rather than discovered.**
    /// A coarse pyramid page carries downsampled heights and **no painted
    /// weights**, so the seam colour there is the uniform layer-0 blend rather than
    /// the painted one — the identical fidelity trade the GI voxelizer already
    /// took, and the right way round: a slightly flat seam everywhere beats a
    /// differently-coloured one depending on where the player walked. And a coarse
    /// page carries **no hole mask** (`inf_terrain::pyramid::downsample_block`
    /// reduces heights, biome ids and data maps and nothing else — pinned by
    /// `a_coarse_page_carries_no_hole_mask`), which is why
    /// [`seam_holes_are_known`](Self::seam_holes_are_known) exists and why
    /// [`apply_seam`] carries a mask-free veto for the case where it answers
    /// `false`.
    ///
    /// The hole test below is the **same poison rule** the fragment shader and
    /// `inf_terrain::TerrainData::height_at` apply: one holed corner of the
    /// bilinear cell removes it. Which is what makes a cave mouth work — the
    /// vertices *inside* the hole get no seam, the ones just outside it do, and
    /// the band falls off across the rim.
    pub fn seam_sample(&self, x: f64, z: f64) -> Option<SeamSample> {
        let res = self.tile_resolution.max(2);
        let lod = self.max_lod();
        let mps = self.meters_per_sample * (1u64 << lod.min(62)) as f64;
        let span = self.tile_span_at(lod);
        let coord = ((x / span).floor() as i32, (z / span).floor() as i32);
        let tile = self
            .tiles
            .iter()
            .find(|t| t.key.lod == lod && t.key.coord == coord)?;

        let u = ((x - coord.0 as f64 * span) / mps).clamp(0.0, (res - 1) as f64);
        let v = ((z - coord.1 as f64 * span) / mps).clamp(0.0, (res - 1) as f64);
        let (i0, j0) = (u.floor() as u32, v.floor() as u32);
        let (i1, j1) = ((i0 + 1).min(res - 1), (j0 + 1).min(res - 1));
        if tile.is_hole(res, i0, j0)
            || tile.is_hole(res, i1, j0)
            || tile.is_hole(res, i0, j1)
            || tile.is_hole(res, i1, j1)
        {
            return None;
        }
        let (fx, fz) = (u - i0 as f64, v - j0 as f64);
        let h = |i: u32, j: u32| tile.heights[(j * res + i) as usize] as f64;
        let lerp2 = |a: f64, b: f64, c: f64, d: f64| {
            let x0 = a + (b - a) * fx;
            let x1 = c + (d - c) * fx;
            x0 + (x1 - x0) * fz
        };
        let height = tile.origin.y + lerp2(h(i0, j0), h(i1, j0), h(i0, j1), h(i1, j1));

        // Central differences on the sample lattice — the same gradient the
        // terrain fragment shader takes, so the two normals agree at the seam.
        let e = mps as f32;
        let im = i0.saturating_sub(1);
        let ip = (i0 + 1).min(res - 1);
        let jm = j0.saturating_sub(1);
        let jp = (j0 + 1).min(res - 1);
        let dx = (h(ip, j0) - h(im, j0)) as f32 / ((ip - im).max(1) as f32 * e);
        let dz = (h(i0, jp) - h(i0, jm)) as f32 / ((jp - jm).max(1) as f32 * e);
        let n = {
            let v = [-dx, 1.0, -dz];
            let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
            [v[0] / l, v[1] / l, v[2] / l]
        };

        // Splat blend, against THIS terrain's layers.
        let w = |i: u32, j: u32| tile.weight_sample(res, i, j);
        let wq = [w(i0, j0), w(i1, j0), w(i0, j1), w(i1, j1)];
        let mut weights = [0.0f32; 4];
        for k in 0..4 {
            let a = wq[0][k] as f32 / 255.0;
            let b = wq[1][k] as f32 / 255.0;
            let c = wq[2][k] as f32 / 255.0;
            let d = wq[3][k] as f32 / 255.0;
            weights[k] = lerp2(a as f64, b as f64, c as f64, d as f64) as f32;
        }
        let sum: f32 = weights.iter().sum();
        if sum > 1e-4 {
            for w in &mut weights {
                *w /= sum;
            }
        } else {
            weights = [1.0, 0.0, 0.0, 0.0];
        }
        let mut albedo = [0.0f32; 3];
        let mut roughness = 0.0f32;
        for (k, layer) in self.layers.iter().enumerate() {
            for (c, out) in albedo.iter_mut().enumerate() {
                *out += weights[k] * layer.albedo[c];
            }
            roughness += weights[k] * layer.roughness;
        }

        Some(SeamSample {
            normal: n,
            height,
            albedo,
            roughness: roughness.clamp(0.04, 1.0),
        })
    }

    /// The coarsest asset LOD present in the projection (`0` for a level-0-only
    /// terrain — every inline, non-streamed terrain).
    pub fn max_lod(&self) -> u32 {
        self.tiles.iter().map(|t| t.key.lod).max().unwrap_or(0)
    }

    /// Whether the level [`seam_sample`](Self::seam_sample) reads — the residency
    /// floor, [`max_lod`](Self::max_lod) — carries the per-sample hole mask.
    ///
    /// `true` exactly when that level is 0, because holes live **only** on
    /// authored level-0 tiles: `inf_terrain::pyramid::downsample_block` reduces
    /// heights, biome ids and erosion data maps into a coarse page and carries no
    /// hole mask upward at all (the P21.2 remainder, pinned by
    /// `a_coarse_page_carries_no_hole_mask`).
    ///
    /// The consequence is the one [`apply_seam`] acts on: where this is `false`
    /// the poison rule inside `seam_sample` **cannot fire**, so a cave ceiling
    /// under a mouth would answer "there is hillside here" and wear the hillside's
    /// material. Not knowing about a hole is not the same as there being none, and
    /// this is the predicate that says which of the two the caller is holding.
    pub fn seam_holes_are_known(&self) -> bool {
        self.max_lod() == 0
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

/// A projected voxel chunk's identity: its integer position in the volume's
/// chunk grid (P21.1).
///
/// This is the renderer-local mirror of `inf_voxel::ChunkKey` — `inf-render` is
/// Ring 0 and deliberately does **not** depend on `inf-voxel`, exactly as it does
/// not depend on `inf-terrain` (see [`TerrainTileKey`]). The two documented
/// projectors map one onto the other, like every other scene DTO, which is what
/// keeps the meshed surface the renderer draws *triangle soup* rather than a
/// second copy of the SDF model.
///
/// A chunk is a fixed-size cube of the volume's grid, so the key is a plain 3D
/// lattice coordinate: unlike a terrain tile there is no LOD component, because a
/// voxel volume is a **local** extension of the heightfield (a cave, an
/// excavation, an overhang) rather than a paged world-scale surface — P21.1 meshes
/// every resident chunk at full resolution.
///
/// `Ord` is the **derived, natural field order** (`x`, then `y`, then `z`) rather
/// than a hand-written `(z, y, x)`: the projector hands chunks over in
/// `inf_voxel`'s `BTreeMap` order, which is that same derived field order, so the
/// two orders agree *by construction* instead of by a comment nobody re-checks.
/// A projection walked in key order is therefore also key-ascending here — the
/// order [`RenderVoxelVolume::chunks`] is documented to arrive in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct VoxelChunkKey {
    /// Chunk grid coordinate along world X.
    pub x: i32,
    /// Chunk grid coordinate along world Y (voxel volumes are genuinely 3D —
    /// this is the axis a heightfield tile key does not have).
    pub y: i32,
    /// Chunk grid coordinate along world Z.
    pub z: i32,
}

impl VoxelChunkKey {
    /// A key at an explicit lattice coordinate.
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

/// One meshed voxel-surface vertex (P21.1) — the output of the isosurface
/// extraction, handed to the [`VoxelNode`](crate::passes::voxel) as plain
/// triangle soup.
///
/// Positions are **chunk-local `f32` metres**: the owning
/// [`RenderVoxelChunk::origin`] carries the `f64` world anchor and the pass builds
/// a per-chunk model matrix from it against the frame's floating origin. That is
/// architecture rule 3 applied at the natural seam — a chunk is metres across, so
/// its interior needs no `f64` precision at all, and the one place the world's
/// magnitude enters is the anchor the origin is subtracted from.
///
/// Normals arrive **already normalized** (the mesher takes them from the SDF
/// gradient, which is what makes a voxel surface smooth without a smoothing pass);
/// the shader re-normalizes after interpolation, which is a different thing.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderVoxelVertex {
    /// Chunk-local position (metres, relative to the chunk's `origin`).
    pub pos: [f32; 3],
    /// Unit surface normal (chunk-local == world, chunks are unrotated).
    pub normal: [f32; 3],
    /// Splat-layer index `0..=3`, selecting into [`RenderVoxelVolume::layers`].
    ///
    /// **Categorical**, exactly like [`RenderTerrainTile::biomes`]: it names a
    /// layer, it is not a quantity, so it is passed to the fragment stage
    /// `@interpolate(flat)` — the midpoint of layers 1 and 3 is not layer 2.
    /// Values past `3` clamp in the shader rather than reading out of bounds, so a
    /// projector that grows a fifth material before the renderer does degrades to
    /// the last layer instead of to undefined behaviour.
    pub material: u32,
    /// **Seam (P21.2)**: the heightfield's unit surface normal at this vertex's
    /// world XZ in `xyz`, and the heightfield's world surface height in `w`.
    ///
    /// [`NO_SEAM`](RenderVoxelVertex::NO_SEAM) — all zeros — means "no
    /// heightfield over this point", and it is a sentinel that cannot be confused
    /// with data: a heightfield normal is the gradient of a single-valued
    /// function of `(x, z)`, so `y` is strictly positive for every real sample.
    /// The shader tests `y <= 0` and skips the blend entirely, which is what keeps
    /// a volume with no terrain over it byte-identical to its pre-P21.2 render.
    pub seam_nh: [f32; 4],
    /// **Seam (P21.2)**: the terrain's splat-blended albedo in `rgb` and its
    /// blended perceptual roughness in `a` at this vertex's world XZ, resolved
    /// against the **terrain's** layer palette (not this volume's) by
    /// [`RenderTerrain::seam_sample`]. Ignored when
    /// [`seam_nh`](Self::seam_nh) is the sentinel.
    pub seam_albedo: [f32; 4],
}

impl RenderVoxelVertex {
    /// The "no heightfield here" seam sentinel — see [`seam_nh`](Self::seam_nh).
    /// A projector with no terrain to sample writes this and the blend is off.
    pub const NO_SEAM: [f32; 4] = [0.0; 4];
}

/// One meshed chunk handed to the [`VoxelNode`](crate::passes::voxel) (P21.1).
///
/// The renderer never sees the SDF: it sees the triangles a chunk currently
/// extracts to, plus the stamp that says whether they changed. Everything about
/// *why* the surface has this shape — the density field, the brush history, the
/// meshing algorithm — lives in `inf-voxel` and stops at this boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderVoxelChunk {
    /// Grid position of this chunk within its volume.
    pub key: VoxelChunkKey,
    /// `f64` world position of this chunk's local origin — the anchor
    /// [`vertices`](Self::vertices) are relative to.
    pub origin: DVec3,
    /// The chunk's meshed surface vertices (chunk-local, see
    /// [`RenderVoxelVertex`]).
    pub vertices: Vec<RenderVoxelVertex>,
    /// Triangle list into [`vertices`](Self::vertices). Length must be a multiple
    /// of 3 and every index must be in range; a chunk that fails either is
    /// **dropped** by the cache planner rather than uploaded (see
    /// [`plan_chunk_cache`](crate::passes::voxel::plan_chunk_cache)) — a bad page
    /// never becomes silent geometry.
    pub indices: Vec<u32>,
    /// Inclusive chunk-local `(min, max)` AABB of [`vertices`](Self::vertices) —
    /// the frustum-cull bound. Projected rather than recomputed per frame because
    /// the mesher already knows it.
    pub bounds: ([f32; 3], [f32; 3]),
    /// The chunk's **monotone change stamp**. The GPU buffer cache re-uploads this
    /// chunk if — and only if — the stamp differs from the one its cached copy was
    /// built at. `0` means "no stamp" (a chunk the source could not version) and is
    /// treated conservatively as *always re-upload*, never as a cache hit — exactly
    /// like [`RenderTerrainTile::version`], and for the same reason: a source that
    /// cannot version its chunks must degrade to re-uploading, not to a stale
    /// frame.
    pub version: u64,
}

/// A voxel volume's meshed, resident chunk set (P21.1) — the renderer's
/// volumetric-terrain input, projected from the ECS `VoxelVolume` component.
///
/// ## Residency
///
/// `chunks` is whatever the projector handed over — it is **not** assumed to be a
/// complete volume. A missing chunk simply produces no geometry (a hole, exactly
/// as an unmeshed region always was); the renderer never invents a want. This is
/// the same contract [`RenderTerrain::tiles`] carries, and for the same reason:
/// residency selection is a decision about the world, and it lives above this DTO.
///
/// ## Relationship to the heightfield
///
/// A volume *locally extends* the heightfield rather than replacing it — a cave
/// mouth, an excavated pit, an overhang the 2.5D surface cannot express. That is
/// why [`layers`](Self::layers) is deliberately the SAME
/// [`RenderTerrainLayer`] type the terrain splat uses, at deliberately the same
/// indices: a cave mouth must shade continuously into the hillside it opens out
/// of, and two independently-authored material vocabularies could not.
///
/// Seam *blending* across that boundary (and shadow/GI participation, and the
/// depth prepass) is **P21.2** — P21.1 draws the surfaces with their own simple
/// lit pass and says so in `shaders/voxel.wgsl`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RenderVoxelVolume {
    /// Stable identity of the volume this projection describes.
    ///
    /// The per-chunk GPU buffer cache is keyed by `(id, chunk key)`, so two
    /// volumes whose grids share a coordinate cannot overwrite each other's
    /// chunks, and a volume that leaves the scene releases exactly its own
    /// buffers. Both projectors derive it from the volume entity's `Guid`
    /// ([`terrain_id_from_guid`] — the same fold, so a voxel volume and a terrain
    /// are identified by the same rule); `0` is the "unkeyed" value a
    /// single-volume caller (and every unit test) leaves at its default.
    ///
    /// **Distinct volumes in one scene must carry distinct ids.** The renderer
    /// cannot check that — two projections claiming one id simply share a cache
    /// slot and fight over it.
    pub id: u64,
    /// The resident meshed chunks, ascending by [`VoxelChunkKey`] — a
    /// deterministic upload/draw order (see the key's `Ord` note).
    pub chunks: Vec<RenderVoxelChunk>,
    /// The four splat material layers a vertex's
    /// [`material`](RenderVoxelVertex::material) index selects.
    ///
    /// Deliberately the SAME [`RenderTerrainLayer`] the heightfield uses, and
    /// deliberately indices `0..=3` **aligned with the terrain splat layers** — a
    /// cave mouth must shade continuously into the hillside it opens out of, which
    /// it cannot do if layer 2 means "rock" on one side of the seam and "moss" on
    /// the other. (`tex_scale` is carried but unused: the voxel shader has no
    /// triplanar detail grain — P21.2 closed the seam through the per-vertex blend
    /// below instead, which is a different mechanism.)
    pub layers: [RenderTerrainLayer; 4],
    /// **Seam blend band (P21.2), in metres.** A voxel fragment within this
    /// distance of the heightfield surface mixes toward the terrain's own albedo,
    /// roughness and normal there; `0` disables the blend for this volume.
    ///
    /// A width, not a switch, because the right value is a property of the
    /// content: a cave mouth cut into 1 m-per-sample ground wants a band about a
    /// metre or two wide — wide enough that the transition is not a visible line,
    /// narrow enough that the cave interior does not turn into hillside. The
    /// projector defaults it to [`DEFAULT_SEAM_BAND_M`].
    ///
    /// What it buys and what it does not is stated in `voxel.wgsl`: material and
    /// normal continuity, **not** geometric welding.
    pub seam_band_m: f32,
}

/// Default width of the P21.2 voxel-to-heightfield seam blend band, in metres.
///
/// Two metres: about two samples of a default 1 m terrain, so the band spans a
/// few pixels of hillside at any reasonable viewing distance and reads as a
/// gradient rather than a boundary, while a player standing in a cave four metres
/// under the surface is entirely outside it.
pub const DEFAULT_SEAM_BAND_M: f32 = 2.0;

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
    /// GPU-scattered instance batches (P18.5) — PCG volumes and painted foliage.
    /// Empty ⇒ the [`crate::passes::scatter`] node emits no commands, so every
    /// scene without scatter content (including all 39 goldens) is untouched.
    ///
    /// Version-gated like everything else, but the *upload* is gated by each
    /// batch's own [`ScatterData::key`] instead: a projection that rebuilds the
    /// list without the scatter changing re-uses the GPU buffers.
    pub scatter: Vec<ScatterBatch>,
    /// Water bodies (P20.1) — oceans, lakes and spline rivers. Empty ⇒ the
    /// [`crate::passes::water`] node returns before touching the encoder, so every
    /// scene without water (including all 42 pre-P20.1 goldens) records the exact
    /// command stream it always did.
    ///
    /// Ordering is the projector's and is what the draw order follows, so it must
    /// be deterministic per side. It is **not** the same order in both projectors
    /// (the player walks `Guid` order, the viewport document order) — the same
    /// arrangement `terrains` has, and for the same reason: what makes a
    /// cross-side comparison meaningful is each body's `id`, which both derive
    /// from the entity.
    pub waters: Vec<crate::water::RenderWater>,
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
    /// Volumetric-terrain volumes (P21.1) — SDF voxel chunk sets, already meshed
    /// into triangle soup by the projector, that locally extend the heightfield
    /// above. Empty ⇒ the [`crate::passes::voxel`] node returns before touching the
    /// encoder, so every scene without volumetric terrain (including all 47
    /// pre-P21.1 goldens) records the exact command stream it always did.
    ///
    /// Ordering is the projector's, and it is what the draw order follows — so it
    /// must be deterministic. It is **not** the same order in both projectors: the
    /// player walks its world in `Guid` order, the editor viewport walks the
    /// document's own entity order. Each is deterministic for its own side, which
    /// is what a per-side determinism gate needs; what makes a *cross-side*
    /// comparison meaningful is [`id`](RenderVoxelVolume::id), which both derive
    /// from the volume entity's `Guid`, so a PIE-vs-shipping diff matches volumes
    /// up by identity rather than by position in a list. The same arrangement
    /// [`terrains`](Self::terrains) and `waters` have, for the same reason.
    ///
    /// Each chunk's own [`version`](RenderVoxelChunk::version) stamp gates its
    /// vertex/index buffer upload, keyed per volume by
    /// [`RenderVoxelVolume::id`].
    pub voxels: Vec<RenderVoxelVolume>,
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
    // ── P21.2 seam ──────────────────────────────────────────────

    /// A 5×5 tile at y = 10, sloping +1 m per sample along +X, optionally with
    /// sample `(2, 2)` holed.
    fn seam_terrain(holed: bool) -> RenderTerrain {
        const RES: u32 = 5;
        let heights: Vec<f32> = (0..RES).flat_map(|_| (0..RES).map(|i| i as f32)).collect();
        let mut holes = Vec::new();
        if holed {
            holes = vec![0u32; RES as usize];
            holes[2] |= 1 << 2;
        }
        RenderTerrain {
            id: 1,
            tile_resolution: RES,
            meters_per_sample: 1.0,
            tiles: vec![RenderTerrainTile {
                key: TerrainTileKey::lod0((0, 0)),
                origin: DVec3::new(0.0, 10.0, 0.0),
                heights,
                weights: vec![[255, 0, 0, 0]; (RES * RES) as usize],
                biomes: vec![0; (RES * RES) as usize],
                holes,
                height_bounds: (0.0, 4.0),
                version: 1,
            }],
            layers: [
                RenderTerrainLayer {
                    albedo: [0.2, 0.4, 0.1, 1.0],
                    roughness: 0.8,
                    tex_scale: 1.0,
                },
                RenderTerrainLayer::default(),
                RenderTerrainLayer::default(),
                RenderTerrainLayer::default(),
            ],
            macro_variation: 0.0,
            biome_palette: Vec::new(),
        }
    }

    /// The sampler answers with the surface it was given — the interpolated
    /// height, a normal that leans away from the slope, and the layer-0 albedo
    /// the weights select.
    #[test]
    fn seam_sample_resolves_height_normal_and_layer() {
        let t = seam_terrain(false);
        let s = t.seam_sample(1.5, 1.0).expect("inside the tile");
        assert!((s.height - 11.5).abs() < 1e-6, "height {}", s.height);
        // Heights rise with +X, so the normal tilts toward -X, and +Y stays
        // positive — which is what makes `y <= 0` a usable sentinel.
        assert!(s.normal[0] < 0.0 && s.normal[1] > 0.0, "{:?}", s.normal);
        assert!((s.normal[2]).abs() < 1e-6, "{:?}", s.normal);
        assert_eq!(s.albedo, [0.2, 0.4, 0.1]);
        assert!((s.roughness - 0.8).abs() < 1e-6);

        // Outside the projected tile there is no answer at all — not a guess.
        assert!(t.seam_sample(500.0, 500.0).is_none());
    }

    /// **The poison rule, projector half.** A holed sample removes the seam from
    /// every cell that interpolates it, and from no other — the same rule
    /// `terrain.wgsl` and `inf_terrain::TerrainData::height_at` apply.
    #[test]
    fn seam_sample_refuses_a_holed_cell_and_only_that_cell() {
        let t = seam_terrain(true);
        // The four cells around sample (2, 2).
        for (x, z) in [(1.5, 1.5), (2.5, 1.5), (1.5, 2.5), (2.5, 2.5)] {
            assert!(t.seam_sample(x, z).is_none(), "({x}, {z}) survived");
        }
        // One cell further out is untouched.
        for (x, z) in [(0.5, 0.5), (3.5, 3.5)] {
            assert!(t.seam_sample(x, z).is_some(), "({x}, {z}) was poisoned");
        }
        // The same query on the un-carved twin answers everywhere — so the test
        // above is about the hole and not about the fixture.
        let clean = seam_terrain(false);
        assert!(clean.seam_sample(2.0, 2.0).is_some());
    }

    /// `apply_seam` arms the band and fills the vertices that have terrain over
    /// them, leaves the rest at the sentinel, and is a **no-op** when disarmed —
    /// which is what keeps a terrain-free scene byte-identical to its pre-P21.2
    /// render.
    #[test]
    fn apply_seam_fills_only_where_there_is_ground() {
        let terrain = seam_terrain(true);
        let vertex = |x: f32, z: f32| RenderVoxelVertex {
            pos: [x, 0.0, z],
            normal: [0.0, 1.0, 0.0],
            material: 0,
            seam_nh: RenderVoxelVertex::NO_SEAM,
            seam_albedo: [0.0; 4],
        };
        let mut volumes = vec![RenderVoxelVolume {
            id: 1,
            chunks: vec![RenderVoxelChunk {
                key: VoxelChunkKey::default(),
                origin: DVec3::ZERO,
                // Under the hole, on clear ground, and off the terrain entirely.
                vertices: vec![vertex(2.0, 2.0), vertex(0.5, 0.5), vertex(400.0, 400.0)],
                indices: vec![0, 1, 2],
                bounds: ([0.0; 3], [1.0; 3]),
                version: 1,
            }],
            layers: [RenderTerrainLayer::default(); 4],
            seam_band_m: 0.0,
        }];

        // Disarmed: nothing moves, and the band stays off.
        let mut off = volumes.clone();
        apply_seam(&mut off, std::slice::from_ref(&terrain), 0.0);
        assert_eq!(off[0].seam_band_m, 0.0);
        assert!(off[0].chunks[0]
            .vertices
            .iter()
            .all(|v| v.seam_nh == RenderVoxelVertex::NO_SEAM));

        apply_seam(&mut volumes, std::slice::from_ref(&terrain), 2.0);
        assert_eq!(volumes[0].seam_band_m, 2.0);
        let vs = &volumes[0].chunks[0].vertices;
        assert_eq!(
            vs[0].seam_nh,
            RenderVoxelVertex::NO_SEAM,
            "a vertex under the hole must not pick up a seam"
        );
        assert!(
            vs[1].seam_nh[1] > 0.0,
            "clear ground must seam: {:?}",
            vs[1]
        );
        assert_eq!(vs[1].seam_albedo, [0.2, 0.4, 0.1, 0.8]);
        assert_eq!(
            vs[2].seam_nh,
            RenderVoxelVertex::NO_SEAM,
            "a vertex off the terrain must not pick up a seam"
        );
    }

    /// The row-packed mask reads back the bit that was set, and an empty mask
    /// reads `false` everywhere — the sparse default, which is what lets the
    /// pass bind a 1×1 sentinel texture.
    #[test]
    fn the_projected_hole_mask_reads_back() {
        let t = seam_terrain(true);
        let tile = &t.tiles[0];
        assert!(tile.has_holes());
        assert!(tile.is_hole(5, 2, 2));
        for (i, j) in [(1, 2), (3, 2), (2, 1), (2, 3)] {
            assert!(!tile.is_hole(5, i, j), "({i},{j})");
        }
        let clean = seam_terrain(false);
        assert!(!clean.tiles[0].has_holes());
        assert!(!clean.tiles[0].is_hole(5, 2, 2));
    }

    // ── P21.2 audit: the seam may not read camera-driven residency ──────

    /// A **streamed** seam terrain: the level-0 page of [`seam_terrain`] (holed)
    /// beside the coarse pyramid page the streamer pins as its residency floor.
    ///
    /// `level_0` is what the camera controls — near the terrain it is paged in,
    /// far away it is not — so the two states this builds are two genuinely
    /// different *residency histories* over one asset.
    ///
    /// The coarse page is deliberately **not** a decimation of the fine one (flat
    /// at +5 m, where level 0 slopes from +0 to +4): a fixture whose two levels
    /// agreed could not say which one answered, and that is the whole question
    /// here. It carries no weights and no hole mask, which is exactly what a real
    /// `downsample_block` page carries.
    fn streamed_seam_terrain(level_0: bool) -> RenderTerrain {
        const RES: u32 = 5;
        let coarse = RenderTerrainTile {
            key: TerrainTileKey::new(1, (0, 0)),
            origin: DVec3::new(0.0, 10.0, 0.0),
            heights: vec![5.0; (RES * RES) as usize],
            weights: Vec::new(),
            biomes: Vec::new(),
            holes: Vec::new(),
            height_bounds: (5.0, 5.0),
            version: 2,
        };
        let mut t = seam_terrain(true);
        if !level_0 {
            t.tiles.clear();
        }
        t.tiles.push(coarse);
        t
    }

    /// **THE B1 REGRESSION, projector half.** The seam is resolved against the
    /// terrain's *residency floor* — never against whichever fine page the camera
    /// dragged in — so two residency histories over one asset produce the identical
    /// sample.
    ///
    /// Before the fix `seam_sample` took `key.lod == 0`, which is precisely the
    /// part of the published cut that pages: the same point answered with a full
    /// blend near the camera and `None` far from it, and a voxel surface's albedo,
    /// roughness and shading normal — its **lighting** — moved with it.
    #[test]
    fn seam_sample_reads_the_residency_floor_not_the_finest_page() {
        let near = streamed_seam_terrain(true);
        let far = streamed_seam_terrain(false);
        assert_eq!(near.max_lod(), 1);
        assert_eq!(far.max_lod(), 1);
        assert!(
            far.tiles.len() < near.tiles.len(),
            "the far state must actually be a smaller residency set"
        );

        for (x, z) in [(0.5, 0.5), (2.0, 2.0), (3.75, 1.25)] {
            let a = near.seam_sample(x, z);
            let b = far.seam_sample(x, z);
            assert_eq!(a, b, "({x}, {z}) answered differently across residency");
            let s = a.expect("the floor covers the whole terrain");
            // …and it is the FLOOR's surface, not the fine page's: level 0 is at
            // 10 + x here, the coarse page is flat at 15.
            assert!((s.height - 15.0).abs() < 1e-6, "height {}", s.height);
            assert_eq!(s.normal, [0.0, 1.0, 0.0]);
        }

        // The level-0 page is genuinely a different surface, so the equality above
        // is a claim about which level was read and not about a flat fixture.
        let inline = seam_terrain(false);
        assert_eq!(inline.max_lod(), 0);
        let s = inline.seam_sample(0.5, 0.5).expect("inside");
        assert!((s.height - 10.5).abs() < 1e-6, "height {}", s.height);
    }

    /// Holes do **not** propagate into the pyramid, so a coarse-floor projection
    /// cannot answer the hole question — and says so rather than implying "no
    /// hole". The stated remainder is `downsample_block` carrying a hole mask; the
    /// day it does, this flips and the veto below becomes dead weight.
    #[test]
    fn a_streamed_projection_does_not_know_where_the_holes_are() {
        assert!(seam_terrain(true).seam_holes_are_known());
        let streamed = streamed_seam_terrain(true);
        assert!(!streamed.seam_holes_are_known());
        // The level-0 poison rule refuses (2, 2); the coarse floor has never heard
        // of it. That is the divergence, pinned rather than left to be found.
        assert!(seam_terrain(true).seam_sample(2.0, 2.0).is_none());
        assert!(streamed.seam_sample(2.0, 2.0).is_some());
    }

    /// **THE B1 GATE, CPU half.** Every seam attribute of every vertex is
    /// bit-identical across two residency histories — asserted on the raw `f32`
    /// arrays, because a lighting input that is "close" across camera history is
    /// still a lighting input that depends on camera history.
    #[test]
    fn the_seam_is_bit_identical_across_two_residency_histories() {
        let vertex = |x: f32, z: f32, n: [f32; 3]| RenderVoxelVertex {
            pos: [x, 0.0, z],
            normal: n,
            material: 0,
            seam_nh: RenderVoxelVertex::NO_SEAM,
            seam_albedo: [0.0; 4],
        };
        let volume = RenderVoxelVolume {
            id: 1,
            chunks: vec![RenderVoxelChunk {
                key: VoxelChunkKey::default(),
                origin: DVec3::ZERO,
                vertices: vec![
                    vertex(0.5, 0.5, [0.0, 1.0, 0.0]),
                    vertex(2.0, 2.0, [0.0, 1.0, 0.0]),
                    vertex(3.5, 1.5, [0.3, 0.9, 0.3]),
                    vertex(1.0, 3.0, [0.0, -1.0, 0.0]),
                    vertex(400.0, 400.0, [0.0, 1.0, 0.0]),
                ],
                indices: vec![0, 1, 2],
                bounds: ([0.0; 3], [1.0; 3]),
                version: 1,
            }],
            layers: [RenderTerrainLayer::default(); 4],
            seam_band_m: 0.0,
        };

        let seamed = |terrain: RenderTerrain| {
            let mut v = vec![volume.clone()];
            apply_seam(&mut v, std::slice::from_ref(&terrain), DEFAULT_SEAM_BAND_M);
            v
        };
        let near = seamed(streamed_seam_terrain(true));
        let far = seamed(streamed_seam_terrain(false));
        for (i, (a, b)) in near[0].chunks[0]
            .vertices
            .iter()
            .zip(&far[0].chunks[0].vertices)
            .enumerate()
        {
            assert_eq!(
                (a.seam_nh, a.seam_albedo),
                (b.seam_nh, b.seam_albedo),
                "vertex {i} was lit differently depending on where the camera had \
                 been — camera-driven residency is feeding lighting"
            );
        }
        // Not vacuous: the seam really did fire on the ground-facing vertices.
        assert!(
            near[0].chunks[0].vertices[..3]
                .iter()
                .all(|v| v.seam_nh[1] > 0.0),
            "no vertex picked up a seam at all"
        );
    }

    /// The mask-free veto, where the mask is missing: a surface that does not
    /// **continue** the heightfield gets no seam, so a coarse floor that cannot see
    /// a hole still does not paint hillside onto a cave ceiling.
    ///
    /// The second half is the part that keeps every existing golden byte-stable:
    /// over an *inline* terrain, whose mask IS present, the veto does not apply and
    /// a down-facing vertex seams exactly as it always did.
    #[test]
    fn a_coarse_seam_refuses_the_surfaces_that_do_not_continue_the_ground() {
        let vertex = |n: [f32; 3]| RenderVoxelVertex {
            pos: [0.5, 0.0, 0.5],
            normal: n,
            material: 0,
            seam_nh: RenderVoxelVertex::NO_SEAM,
            seam_albedo: [0.0; 4],
        };
        let volume = RenderVoxelVolume {
            id: 1,
            chunks: vec![RenderVoxelChunk {
                key: VoxelChunkKey::default(),
                origin: DVec3::ZERO,
                // A cave floor (up), a cave wall (perpendicular), a cave roof (down).
                vertices: vec![
                    vertex([0.0, 1.0, 0.0]),
                    vertex([1.0, 0.0, 0.0]),
                    vertex([0.0, -1.0, 0.0]),
                ],
                indices: vec![0, 1, 2],
                bounds: ([0.0; 3], [1.0; 3]),
                version: 1,
            }],
            layers: [RenderTerrainLayer::default(); 4],
            seam_band_m: 0.0,
        };

        let mut streamed = vec![volume.clone()];
        apply_seam(
            &mut streamed,
            std::slice::from_ref(&streamed_seam_terrain(true)),
            DEFAULT_SEAM_BAND_M,
        );
        let vs = &streamed[0].chunks[0].vertices;
        assert!(vs[0].seam_nh[1] > 0.0, "the floor must still seam");
        assert_eq!(
            vs[1].seam_nh,
            RenderVoxelVertex::NO_SEAM,
            "a wall is perpendicular to the ground, not a continuation of it"
        );
        assert_eq!(
            vs[2].seam_nh,
            RenderVoxelVertex::NO_SEAM,
            "hillside was blended onto a cave ceiling"
        );

        // Inline terrain: the mask answers, so the veto stays out of it.
        let mut inline = vec![volume];
        apply_seam(
            &mut inline,
            std::slice::from_ref(&seam_terrain(false)),
            DEFAULT_SEAM_BAND_M,
        );
        assert!(
            inline[0].chunks[0]
                .vertices
                .iter()
                .all(|v| v.seam_nh[1] > 0.0),
            "the veto fired over a terrain whose hole mask is present — every \
             pre-P21.2-audit golden depends on it not doing that"
        );
    }
}
