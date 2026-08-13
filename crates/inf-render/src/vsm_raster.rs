//! **Casters into pages** (P27.2): the per-page GPU cull, the one render pass
//! that owns the whole atlas, and the `set_viewport` / `set_scissor` pair that
//! pins each page to its own 128 × 128 slot.
//!
//! P27.1 made an admit an *allocation*: the slot was published in the indirection
//! table and the page's depth was whatever the atlas last held. This module is
//! what turns the allocation into depth.
//!
//! # One pass, many viewports — the inversion this batch makes
//!
//! `passes::shadow` opens **one render pass per cascade**, each against its own
//! single-layer view, and therefore never needed a scissor: the attachment *is*
//! the cascade. A page atlas cannot work that way — a slot is a rectangle of one
//! texture, not a subresource, and there is no such thing as a view of a
//! rectangle. So the shape inverts: **one** render pass over the whole atlas, and
//! a viewport + scissor per page. This is the first `set_scissor_rect` in the
//! tree (the ROADMAP's own grounding says so, and it was verified before it was
//! written: the only two mentions of either call in `crates/inf-render` were the
//! two comments in `vsm_atlas.rs` and `vsm.rs` promising this batch).
//!
//! **Which of the two actually pins the rect is measured rather than assumed.**
//! Clipping happens against the clip volume and the viewport transform is applied
//! after it, so a triangle cannot land outside its viewport rect: the viewport is
//! what pins the mapping, and the scissor is defence in depth. Both are set
//! because the pair is what the ROADMAP asks for and because the scissor is the
//! only one of the two that survives a future per-page *clear* quad — but the
//! honest statement is in `tests/vsm_raster.rs`, where deleting the viewport
//! fails an arm and the scissor's redundancy is recorded as a measurement.
//!
//! # The whole atlas is cleared every frame, and P27.3 is where that stops
//!
//! `LoadOp::Clear` clears an attachment, not a scissor rectangle, so one pass over
//! the atlas clears every slot. That is exactly right for P27.2, which has **no
//! caching**: every resident page is re-rasterized every frame, so every slot that
//! matters is rewritten in the same pass that cleared it. It is also precisely
//! what P27.3's "static pages survive frames untouched" has to replace — with
//! `LoadOp::Load` and a per-page clear, which is where the scissor stops being
//! defence in depth and becomes load-bearing.
//!
//! # What a page's caster set is
//!
//! The same set the cascaded shadow map casts, plus the paths it never had.
//! Rigid instances and GPU-scattered foliage are packed through
//! `passes::shadow`'s own doors (`pack_bucketed`, `shadow_caster_settings`,
//! `pack_fallback`, `merge_bucketed`) so a page and a cascade cannot disagree
//! about what casts; the caster *band* is VSM's own, because a clipmap's reach is
//! its coarsest level's extent rather than `ShadowSettings::max_distance`.

use glam::Mat4;
use inf_vsm::{VsmLightHandle, VsmPage, VsmResidency};

use crate::camera::{RenderView, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::primitives::{PrimGpu, PrimMesh};
use crate::scene::RenderScene;
use crate::vsm::{vsm_page_matrix, VsmProjection, VSM_DEPTH_CLEAR, VSM_DEPTH_COMPARE};

/// Page rectangles one frame may rasterize.
///
/// A **cost** ceiling, not a quality one, and it is the honest place for it: with
/// no caching (P27.3) every resident page is re-drawn every frame, so the work is
/// `pages × casters` and a 1 024-slot atlas full of pages would issue 1 024
/// viewport switches and 1 024 × groups indirect draws. Pages past the cap keep
/// the depth the clear left them — the [`VSM_ENTRY_NONE`] fail direction, *lit* —
/// and [`VsmRasterStats::deferred_pages`] counts them rather than the cap being
/// silent.
///
/// [`VSM_ENTRY_NONE`]: inf_vsm::VSM_ENTRY_NONE
pub const VSM_MAX_RASTER_PAGES: u32 = 256;

/// Caster records one frame may pack. The visible list is `pages × casters`
/// entries, so this and [`VSM_MAX_RASTER_PAGES`] are two of the three factors of
/// the allocations that grow as a product.
pub const VSM_MAX_CASTERS: u32 = 16_384;

/// Geometry groups one frame may draw with — the **third** factor, and the
/// binding one (P27.2 audit).
///
/// The first write-up named the visible list (`pages × casters × 4 B`) as "the one
/// allocation that grows as a product, and the reason both ceilings exist". It is
/// not the one that binds. The per-(page, group) draw uniform is the same product
/// at [`VSM_PAGE_DRAW_STRIDE`] — **64× the bytes per entry** — and its second
/// factor is *groups*, which nothing capped: a skinned instance is one group
/// because its palette is a bind group, and so is a resident terrain tile. So
/// `groups ≈ casters` on any level with characters or streamed ground, and the
/// worst case was `256 × 16 384 × 256 B` = **1 GiB** — four times `wgpu`'s default
/// `max_buffer_size`, which is a device error at `create_buffer` and not a
/// counted refusal.
///
/// At 1 024 the three worst cases are 64 MiB (draws), 5 MiB (args) and 16 MiB
/// (visible), pinned by
/// `the_page_rasters_worst_case_buffers_fit_the_default_device_limits`. Groups past
/// it are refused — and **counted**, in [`VsmRasterStats::dropped_groups`], with
/// their casters added to [`VsmRasterStats::dropped_casters`], because a cap that
/// stops a level's characters casting must not be silent.
pub const VSM_MAX_GROUPS: u32 = 1_024;

const _: () = assert!(VSM_MAX_GROUPS as usize > PrimMesh::ALL.len());

/// Words in one `DrawIndexedIndirectArgs`: `index_count, instance_count,
/// first_index, base_vertex, first_instance`. **Word 1 is the counter** — the
/// `inf_vgeom` construction, where the atomic the cull increments *is* the draw's
/// instance count, so there is no separate counter buffer and no publish pass.
pub const VSM_ARG_WORDS: u64 = 5;

/// Bytes between two per-(page, group) draw uniforms. `wgpu`'s default
/// `min_uniform_buffer_offset_alignment`, which is what a dynamic offset must be a
/// multiple of.
pub const VSM_PAGE_DRAW_STRIDE: u64 = 256;

/// Geometry groups the rigid path owns — one per [`PrimMesh`] kind, which is
/// exactly the per-kind index ranges `PrimGpu` already holds.
pub const VSM_RIGID_GROUPS: u32 = PrimMesh::ALL.len() as u32;

/// The draw-args buffer's usage. `COPY_SRC` is a **gate's** flag, exactly as the
/// atlas's is: the per-page cull's verdict lives only in these `instance_count`
/// words, and *mirrored ≠ measured* (P26.5) means a gate has to read the GPU's own
/// answer rather than a CPU re-derivation of it. Nothing in the shipping path
/// copies it.
const ARGS_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::INDIRECT)
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);

/// One caster, as the cull and the raster read it. **112 bytes.**
///
/// A storage-buffer record, so `wgpu` validates nothing about its layout and a
/// Rust struct that stops matching `struct VsmCaster` in the two WGSL files is
/// silent corruption rather than a pipeline error — the `VgeomInstanceGpu`
/// reasoning, met again. Hence the pins below.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VsmCasterRaw {
    /// Render-local model matrix.
    pub model: [f32; 16],
    /// xyz = render-local bounding-sphere centre, w = radius. **Conservative**:
    /// the primitive's unit bounding radius times the largest axis scale, so the
    /// cull can only ever keep too much.
    pub sphere: [f32; 4],
    /// x = alpha cutoff, y = blend code (R-P5: 0 opaque, 1 masked), z = base
    /// colour alpha, w = reserved.
    pub mat: [f32; 4],
    /// x = geometry group, y = the caster's index inside its group, z = the
    /// group's first caster — its base in a page's visible list — w = flags.
    pub ids: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<VsmCasterRaw>() == 112);
const _: () = assert!(std::mem::offset_of!(VsmCasterRaw, sphere) == 64);
const _: () = assert!(std::mem::offset_of!(VsmCasterRaw, mat) == 80);
const _: () = assert!(std::mem::offset_of!(VsmCasterRaw, ids) == 96);

/// One page, as the cull reads it. 80 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct VsmPageRaw {
    view_proj: [f32; 16],
    /// x = the page's base in the visible list, y = the atlas slot, z = the light
    /// handle, w = flags.
    info: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<VsmPageRaw>() == 80);

/// One (page, group) draw uniform, padded to the dynamic-offset alignment.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VsmPageDrawRaw {
    view_proj: [f32; 16],
    /// x = this (page, group)'s base in the visible list, y = the atlas slot,
    /// z = the group, w = flags.
    info: [u32; 4],
    /// Padding to [`VSM_PAGE_DRAW_STRIDE`]. Explicit, because `bytemuck::Pod`
    /// forbids a struct with implicit padding bytes.
    _pad: [u32; 44],
}

impl Default for VsmPageDrawRaw {
    fn default() -> Self {
        Self {
            view_proj: [0.0; 16],
            info: [0; 4],
            _pad: [0; 44],
        }
    }
}

const _: () = assert!(std::mem::size_of::<VsmPageDrawRaw>() as u64 == VSM_PAGE_DRAW_STRIDE);

/// The four `SkinnedVertex` fields a page raster reads: position, normal (bound
/// because the layout declares it, never read), joints and weights. **The uv at
/// offset 24 is deliberately skipped** — a depth-only pass has nothing to sample.
const SKINNED_CASTER_ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 12,
        shader_location: 1,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x4,
        offset: 32,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 48,
        shader_location: 3,
    },
];

/// Cells a side of a terrain tile's caster mesh.
///
/// A tile's own resolution can be 512 samples a side, which is a 8 MiB vertex
/// buffer per tile held for the life of the level. A shadow page is 128 texels
/// across and a clipmap level covers many tiles, so terrain shadow detail is
/// bounded by the page long before it is bounded by the heightfield: 64 cells is
/// 4 225 vertices and 135 KiB. The cost is real and stated — a ridge thinner
/// than the sample stride is *smoothed*, so it casts a slightly thinner shadow
/// than it should. The fail direction is a leak (lit), which is the one this
/// phase already chose for `VSM_ENTRY_NONE`.
pub const VSM_TERRAIN_CASTER_CELLS: u32 = 64;

/// One skinned mesh's bind-pose caster geometry.
struct SkinnedCasterGeom {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    /// Local-space bounding sphere of the BIND pose. Conservative for a posed
    /// instance only in the sense that a pose can leave it — which is why the
    /// radius is inflated by [`SKINNED_POSE_MARGIN`] rather than trusted.
    bounds: (glam::Vec3, f32),
}

/// How far a posed skinned mesh may leave its bind-pose bounding sphere, as a
/// fraction of that sphere's radius.
///
/// A skeleton moves vertices, so a bind-pose bound is **not** conservative for an
/// arbitrary pose — an arm raised over the head leaves it. The lit skinned pass
/// never had to care (it does not cull), and a shadow caster that culled on a
/// bound the pose escapes would delete a limb's shadow at exactly the moment the
/// limb moved, which is the worst shape a defect can have. 50 % is a large margin
/// for a rig whose joints stay inside their own mesh and it is a *cull* margin,
/// so its only cost is drawing a caster a page would have clipped anyway. The
/// exact bound is the posed skeleton's own AABB, which the renderer is not given
/// (`SkinnedInstance` carries a palette, not a bound) — P27.3 is where a mover's
/// bounds become load-bearing, and that is where this becomes exact.
pub const SKINNED_POSE_MARGIN: f32 = 0.5;

/// One terrain tile as a static caster mesh: the tile's own height samples, in
/// render-local space, at full tile resolution.
///
/// **Not the clipmap's LOD patch.** The terrain pass draws a camera-fitted
/// clipmap with morphing and skirts, all of which are functions of where the
/// camera is; a shadow drawn through them would move when the camera moved while
/// nothing in the world had. A tile's heights do not move, so this mesh is built
/// once per tile version and cached — which also makes it exactly the geometry
/// P27.3 needs to keep a static page from re-rasterizing.
struct TerrainCasterGeom {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    /// Render-local bounding sphere at the origin the mesh was built under.
    bounds: (glam::Vec3, f32),
    /// The tile version this mesh was built from.
    version: u64,
    /// The floating origin it was built under — a rebase moves render-local space
    /// under it, so the mesh has to be rebuilt (the same rule every packed
    /// instance buffer in the tree follows).
    origin: glam::DVec3,
}

/// The cull's uniform. 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct VsmCullParamsRaw {
    /// x = casters, y = pages, z = groups, w = reserved.
    counts: [u32; 4],
}

/// Which vertex + index buffers a geometry group draws out of.
///
/// Both sources hand the **same** pipeline the same vertex format
/// (`passes::mesh::MeshVertex`), which is why virtualized geometry needs no second
/// caster shader: the classic-LOD chain of a `.inf_vmesh` is a plain index buffer
/// over a plain vertex buffer, exactly as the built-in primitives are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupSource {
    /// The shared [`PrimGpu`] buffers, at this kind's index range.
    Prim,
    /// One virtualized-geometry asset at one classic LOD level.
    Vgeom { asset: u128, level: usize },
    /// One skinned instance: its mesh's bind-pose buffers plus its own joint
    /// palette. A group per *instance* rather than per mesh, because the palette
    /// is per instance and it is a bind group.
    Skinned { instance: usize },
    /// One terrain tile, as a static world-space caster mesh.
    ///
    /// Keyed by the tile's **own** `TerrainTileKey` and not by its index in the
    /// terrain's list: that list is a streaming residency, so index 3 is a
    /// different tile from one frame to the next and a cache keyed on it would
    /// hand a tile another tile's heights whenever the two happened to share a
    /// version stamp. (Found by this batch's own review, before the battery.)
    Terrain {
        terrain: u64,
        tile: crate::scene::TerrainTileKey,
    },
}

/// Where one geometry group's indices live — the three constant words of its
/// indirect args, which the CPU writes and the cull never touches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GroupGeom {
    source: GroupSource,
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    /// Casters this group holds. Zero ⇒ the draw is skipped on the CPU, which is
    /// the one thing the CPU legitimately knows about the drawn set (how many
    /// casters *could* be visible), as opposed to how many *are*.
    casters: u32,
}

/// One virtualized-geometry asset's caster geometry: the shared vertex buffer and
/// one index buffer per classic LOD level, finest first.
///
/// **This is how `vgeom`'s "casts no shadows" hole closes**, and the shape is a
/// ruling with a memo behind it (`docs/memos/p27-2-vgeom-casters.md`): a page
/// caster draws the asset's *classic LOD chain* — the same chain
/// `passes::classic_vgeom` draws, derived from the same meshlet DAG and picked by
/// the same `pick_classic_level` against the same `lod_threshold` — rather than a
/// per-page meshlet cut. So vgeom content casts, with per-page culling on the GPU
/// at *instance* granularity; meshlet granularity is P27.3's, where page caching
/// makes the cost of a second DAG cut per page a number rather than a guess.
struct VgeomCasterGeom {
    vertices: wgpu::Buffer,
    /// `(index buffer, index count)` per level, finest first.
    levels: Vec<(wgpu::Buffer, u32)>,
    /// Per-level object-space error, finest first — `pick_classic_level`'s input.
    errors: Vec<f32>,
    /// Local-space bounding sphere: `(centre, radius)`.
    bounds: ([f32; 3], f32),
}

/// What the page raster did — **the engagement counters**.
///
/// The anti-vacuity instruments for a pass whose entire output is depth in a
/// texture nothing samples yet: a raster that draws nothing and a scene with no
/// casters produce the same atlas, and only a counter tells them apart.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VsmRasterStats {
    /// Frames that opened the page pass.
    pub frames: u64,
    /// Page rectangles rasterized, summed over frames — each one a
    /// `set_viewport` + `set_scissor_rect` pair.
    pub pages: u64,
    /// Indirect draws issued, summed over frames.
    pub draws: u64,
    /// Caster records packed, summed over frames.
    pub casters: u64,
    /// Of those, the ones that came from GPU scatter batches rather than from
    /// `scene.instances` — so "scatter casts" is a measurement.
    pub scatter_casters: u64,
    /// Resident pages the frame's [`VSM_MAX_RASTER_PAGES`] cap did not reach.
    /// **Never a silent cap.**
    pub deferred_pages: u64,
    /// Frames whose caster set contained a masked material, so the alpha-testing
    /// pipeline was bound rather than the fragment-less one.
    pub masked_frames: u64,
    /// Of the packed casters, the ones that are **virtualized-geometry** instances
    /// — so "vgeom casts" is a measurement rather than a claim about a code path.
    pub vgeom_casters: u64,
    /// The classic LOD level each of those drew at, summed over frames — so
    /// *which* level the deviation memo's "the camera's, not the page's" ruling
    /// actually picks is a measurement too (P27.2 audit).
    ///
    /// Nothing else can see it. A caster drawn from the coarsest level of the DAG's
    /// chain lands in the same pages at almost the same depths as one drawn from
    /// the finest, so every image-side arm in the file is satisfied by a raster
    /// that ignores `pick_classic_level` entirely.
    pub vgeom_level_sum: u64,
    /// Skinned casters packed, summed over frames.
    pub skinned_casters: u64,
    /// Terrain-tile casters packed, summed over frames.
    pub terrain_casters: u64,
    /// Casters the [`VSM_MAX_CASTERS`] and [`VSM_MAX_GROUPS`] ceilings dropped,
    /// summed over frames. **Never a silent cap** — a caster set that quietly
    /// stopped at 16 384 reads as "the far half of the level stopped casting".
    pub dropped_casters: u64,
    /// Geometry groups the [`VSM_MAX_GROUPS`] ceiling refused, summed over frames.
    /// Their casters are in [`dropped_casters`](Self::dropped_casters) too; this
    /// says *which* ceiling refused them.
    pub dropped_groups: u64,
}

impl VsmRasterStats {
    /// A one-line human summary, in the shape the other streamers ship.
    pub fn summary(&self) -> String {
        format!(
            "vsm raster: {} frames, {} pages, {} draws, {} casters ({} scattered, \
             {} meshlet-asset, {} skinned, {} terrain), {} pages deferred, \
             {} casters dropped, {} groups refused",
            self.frames,
            self.pages,
            self.draws,
            self.casters,
            self.scatter_casters,
            self.vgeom_casters,
            self.skinned_casters,
            self.terrain_casters,
            self.deferred_pages,
            self.dropped_casters,
            self.dropped_groups,
        )
    }
}

/// The per-page cull and the page-atlas raster.
pub struct VsmRaster {
    cull_pipeline: wgpu::ComputePipeline,
    cull_bgl: wgpu::BindGroupLayout,
    page_bgl: wgpu::BindGroupLayout,
    caster_bgl: wgpu::BindGroupLayout,
    rigid: wgpu::RenderPipeline,
    rigid_masked: wgpu::RenderPipeline,
    skinned_pipeline: wgpu::RenderPipeline,
    palette_bgl: wgpu::BindGroupLayout,
    prim: PrimGpu,

    /// `pages × casters` entries. `STORAGE` only — nothing reads it back.
    visible: wgpu::Buffer,
    visible_entries: u64,
    /// `pages × groups × 5` words, `STORAGE | INDIRECT | COPY_DST`.
    args: wgpu::Buffer,
    args_words: u64,
    casters: wgpu::Buffer,
    caster_capacity: u64,
    pages: wgpu::Buffer,
    page_capacity: u64,
    /// The per-(page, group) draw uniforms, at [`VSM_PAGE_DRAW_STRIDE`].
    draws: wgpu::Buffer,
    draw_capacity: u64,
    params: wgpu::Buffer,

    /// Rebuilt when a buffer is reallocated — a bind group over a dead buffer is
    /// the hazard `crate::vt`'s generation counters exist for, and here the cheap
    /// answer is to drop them whenever anything grew.
    cull_bind: Option<wgpu::BindGroup>,
    caster_bind: Option<wgpu::BindGroup>,
    page_bind: Option<wgpu::BindGroup>,

    /// Caster geometry for the scene's virtualized-geometry assets, keyed by asset
    /// id and evicted when the asset leaves the scene. A **second** copy of a
    /// DAG's vertices when the meshlet path is running — `passes::classic_vgeom`
    /// drops its own copy in that case precisely to avoid the duplication, and
    /// this one is the price of casting: the meshlet pools are streamed, paged and
    /// owned by a render node, and a shadow caster that reached into them would
    /// couple the page raster to residency it does not control.
    vgeom: std::collections::BTreeMap<u128, VgeomCasterGeom>,
    /// Bind-pose GPU buffers per skinned mesh index, and one palette buffer +
    /// bind group per skinned instance. Keyed by the scene's own indices and
    /// rebuilt when the scene version moves, because a palette is *animation* and
    /// changes every frame by construction.
    skinned: Vec<SkinnedCasterGeom>,
    skinned_palettes: Vec<(wgpu::Buffer, wgpu::BindGroup, usize)>,
    /// Static caster meshes for terrain tiles, keyed `(terrain id, tile index)`
    /// and rebuilt when that tile's version moves.
    terrain: std::collections::BTreeMap<(u64, crate::scene::TerrainTileKey), TerrainCasterGeom>,
    /// The scene version the skinned caches were built for.
    skinned_version: Option<u64>,

    /// The pages the last [`record`](Self::record) rasterized, in draw order, and
    /// how many geometry groups each one was drawn with. Kept so a gate can map
    /// an args word back onto the page it decided — the args buffer is a flat
    /// `(page, group)` grid and nothing else in the tree knows its order.
    last_pages: Vec<(VsmLightHandle, VsmPage, (u32, u32, u32))>,
    last_groups: usize,
    /// The frame's packed caster records, moved out of `pack_casters` rather than
    /// copied. See [`last_casters`](Self::last_casters).
    last_casters: Vec<VsmCasterRaw>,

    stats: VsmRasterStats,
}

impl VsmRaster {
    /// Build the cull pipeline, the two rigid page pipelines and the (empty)
    /// buffers. One per [`crate::vsm_mark::VsmSystem`].
    pub fn new(gpu: &GpuContext) -> Self {
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        // Five bindings, four of them storage — under the six the P18.5 scatter
        // cull budgets for and well under the eight `inf_vgeom`'s cull spends
        // with no headroom left.
        let cull_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vsm-cull"),
                entries: &[
                    entry(
                        0,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    ),
                    entry(1, storage(true)),
                    entry(2, storage(true)),
                    entry(3, storage(false)),
                    entry(4, storage(false)),
                ],
            });
        let cull_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vsm-cull"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vsm_cull.wgsl").into()),
            });
        let cull_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vsm-cull"),
                bind_group_layouts: &[Some(&cull_bgl)],
                immediate_size: 0,
            });
        let cull_pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("vsm-cull"),
                layout: Some(&cull_layout),
                module: &cull_shader,
                entry_point: Some("cs_cull"),
                compilation_options: Default::default(),
                cache: None,
            });

        let page_bgl = page_bind_group_layout(&gpu.device);
        let caster_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vsm-casters"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vsm-caster"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vsm_caster.wgsl").into()),
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vsm-caster"),
                bind_group_layouts: &[Some(&page_bgl), Some(&caster_bgl)],
                immediate_size: 0,
            });
        // Vertex buffer 0 of the rigid mesh layout — the shared primitive
        // geometry — and **no instance buffer**: the instance record is pulled from
        // storage, because an indirect draw over a GPU-compacted list cannot use a
        // vertex-step buffer without `INDIRECT_FIRST_INSTANCE`.
        let vertex_buffers = [crate::passes::mesh::vertex_layouts()[0].clone()];
        let make = |label: &str, vs: &str, fs: Option<wgpu::FragmentState>| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(vs),
                        compilation_options: Default::default(),
                        buffers: &vertex_buffers,
                    },
                    fragment: fs,
                    primitive: page_primitive_state(),
                    depth_stencil: Some(page_depth_state()),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let rigid = make("vsm-caster", "vs", None);
        let rigid_masked = make(
            "vsm-caster-masked",
            "vs_masked",
            Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_masked"),
                compilation_options: Default::default(),
                // Depth-only discard; no colour target — `shadow_depth.wgsl`'s
                // construct, already proven in two passes.
                targets: &[],
            }),
        );

        // ── the skinned page pipeline. Its own vertex format and its own third
        // group (the instance's joint palette); everything else — the page
        // uniform, the pulled caster record, the depth state — is shared, so a
        // skinned caster and a rigid one cannot end up in different conventions.
        let palette_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vsm-palette"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let skinned_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vsm-skinned"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/vsm_skinned.wgsl").into()),
            });
        let skinned_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vsm-skinned"),
                bind_group_layouts: &[Some(&page_bgl), Some(&caster_bgl), Some(&palette_bgl)],
                immediate_size: 0,
            });
        let skinned_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("vsm-skinned"),
                layout: Some(&skinned_layout),
                vertex: wgpu::VertexState {
                    module: &skinned_shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<crate::scene::SkinnedVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &SKINNED_CASTER_ATTRIBUTES,
                    })],
                },
                fragment: None,
                primitive: page_primitive_state(),
                depth_stencil: Some(page_depth_state()),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let empty = |label: &str, usage: wgpu::BufferUsages| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 256,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            cull_pipeline,
            cull_bgl,
            page_bgl,
            caster_bgl,
            rigid,
            rigid_masked,
            skinned_pipeline,
            palette_bgl,
            prim: PrimGpu::new(gpu, "vsm"),
            visible: empty("vsm-visible", wgpu::BufferUsages::STORAGE),
            visible_entries: 64,
            args: empty("vsm-draw-args", ARGS_USAGE),
            args_words: 64,
            casters: empty(
                "vsm-casters",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            caster_capacity: 2,
            pages: empty(
                "vsm-pages",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            page_capacity: 3,
            draws: empty(
                "vsm-page-draws",
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            draw_capacity: 1,
            params: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vsm-cull-params"),
                size: std::mem::size_of::<VsmCullParamsRaw>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            cull_bind: None,
            caster_bind: None,
            page_bind: None,
            vgeom: std::collections::BTreeMap::new(),
            skinned: Vec::new(),
            skinned_palettes: Vec::new(),
            terrain: std::collections::BTreeMap::new(),
            skinned_version: None,
            last_pages: Vec::new(),
            last_groups: 0,
            last_casters: Vec::new(),
            stats: VsmRasterStats::default(),
        }
    }

    /// The engagement counters.
    #[inline]
    pub fn stats(&self) -> VsmRasterStats {
        self.stats
    }

    /// The pages the last [`record`](Self::record) rasterized, in the order the
    /// args buffer indexes them: `(light, page, atlas rect)`.
    #[inline]
    pub fn last_pages(&self) -> &[(VsmLightHandle, VsmPage, (u32, u32, u32))] {
        &self.last_pages
    }

    /// Geometry groups the last [`record`](Self::record) used — the args buffer's
    /// row stride.
    #[inline]
    pub fn last_groups(&self) -> usize {
        self.last_groups
    }

    /// **The caster records the last [`record`](Self::record) packed**, exactly as
    /// the cull and the raster read them.
    ///
    /// Kept by *moving* the frame's own `Vec` rather than cloning it, so the
    /// shipping path pays nothing for it, and kept at all for two readers: a gate
    /// that wants the **cull sphere** a caster was culled with — the skinned pose
    /// margin has no other observable, because a bound that is too tight deletes a
    /// limb's shadow only in the pages that limb reached (P27.2 audit) — and
    /// P27.3, whose invalidation has to ask "which pages did this mover touch"
    /// about exactly this set.
    #[inline]
    pub fn last_casters(&self) -> &[VsmCasterRaw] {
        &self.last_casters
    }

    /// **The cull's own verdict**, read back off the device: `instance_count` per
    /// `(page, group)`, in `last_pages` × `last_groups` order.
    ///
    /// A gate's door, not a renderer's. The counts the GPU wrote are the only
    /// record of what the per-page cull decided — the atlas cannot tell a culled
    /// caster from a clipped one, because a caster the page's frustum rejects
    /// writes nothing either way — so this is where "the cull is SUBTRACTIVE"
    /// stops being a claim about performance and becomes a measurement.
    pub fn read_draw_counts(&self, gpu: &GpuContext) -> Vec<u32> {
        let n = (self.last_pages.len() * self.last_groups) as u64;
        if n == 0 {
            return Vec::new();
        }
        let bytes = n * VSM_ARG_WORDS * 4;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vsm-draw-args-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vsm-draw-args-readback"),
            });
        encoder.copy_buffer_to_buffer(&self.args, 0, &staging, 0, bytes);
        gpu.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        if rx.recv().is_err() {
            return Vec::new();
        }
        let Ok(data) = slice.get_mapped_range() else {
            return Vec::new();
        };
        let words: &[u32] = bytemuck::cast_slice(&data);
        let out = words
            .chunks_exact(VSM_ARG_WORDS as usize)
            .map(|a| a[1])
            .collect();
        drop(data);
        staging.unmap();
        out
    }

    /// The shared page-uniform layout — the seam a caster pipeline built outside
    /// this module would bind through. Every P27.2 path lives here, so it has no
    /// reader yet; it is the door P27.4's receiver debug view and P28.3's merged
    /// streamer are expected to come through, named rather than implied.
    #[inline]
    pub fn page_layout(&self) -> &wgpu::BindGroupLayout {
        &self.page_bgl
    }

    /// **Record the page raster.**
    ///
    /// Everything is a pure function of `(residency, projections, scene, view)`;
    /// nothing here reads a clock, a frame index or a stamp.
    ///
    /// Returns the number of page rectangles rasterized — 0 means the encoder was
    /// not touched at all, which is the off-path guarantee every pass in this tree
    /// keeps.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        residency: &VsmResidency,
        projections: &[VsmProjection],
        proj_base: &[u32],
        atlas: &wgpu::TextureView,
        scene: &RenderScene,
        view: &RenderView,
        settings: &crate::settings::RenderSettings,
    ) -> u32 {
        let pages = self.collect_pages(residency, projections, proj_base);
        if pages.is_empty() {
            return 0;
        }
        self.sync_vgeom(gpu, scene);
        self.sync_skinned(gpu, scene);
        self.sync_terrain(gpu, scene, view);
        let CasterPack {
            casters,
            groups,
            masked,
            scattered,
            dropped,
            dropped_groups,
            level_sum,
        } = pack_casters(
            &self.prim,
            &self.vgeom,
            &self.skinned,
            &self.terrain,
            scene,
            view,
            settings,
        );
        // **A page with nothing to draw into it still has to be CLEARED.** This
        // used to return early, and the comment that stood here argued the return
        // was only taken when there was no pass to open — which is exactly wrong:
        // it is taken when there are *pages* and no *casters*, and then the atlas
        // keeps the previous frame's depth in every slot. That configuration is
        // reachable and it is not exotic: the editor's infinite grid writes camera
        // depth and is not a caster, so an empty level under a shadow-casting sun
        // marks pages and packs nothing — and every object the author just deleted
        // goes on casting out of the atlas. (P27.2 audit.)
        //
        // So the pass opens either way and the clear is the frame's whole output.
        // `draws` and `casters` stay zero, which is what tells a caster-less frame
        // apart from an empty one in the counters.
        if casters.is_empty() {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vsm-pages-clear"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: atlas,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(VSM_DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.last_pages = pages
                .iter()
                .map(|p| (VsmLightHandle(p.light), p.page, p.rect))
                .collect();
            self.last_groups = 0;
            self.last_casters.clear();
            self.stats.frames += 1;
            self.stats.pages += pages.len() as u64;
            return pages.len() as u32;
        }

        let page_count = pages.len() as u32;
        let caster_count = casters.len() as u32;
        let group_count = groups.len() as u32;
        self.ensure(gpu, page_count, caster_count, group_count);

        // ── uploads, all through the queue, so they are ordered BEFORE every
        // command this encoder holds (the `crate::vt` ordering contract).
        gpu.queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&VsmCullParamsRaw {
                counts: [caster_count, page_count, group_count, 0],
            }),
        );
        gpu.queue
            .write_buffer(&self.casters, 0, bytemuck::cast_slice(&casters));
        let page_raw: Vec<VsmPageRaw> = pages
            .iter()
            .enumerate()
            .map(|(i, p)| VsmPageRaw {
                view_proj: p.view_proj.to_cols_array(),
                info: [i as u32 * caster_count, p.slot, p.light, 0],
            })
            .collect();
        gpu.queue
            .write_buffer(&self.pages, 0, bytemuck::cast_slice(&page_raw));

        // The per-(page, group) draw uniforms and the indirect args, built
        // together so a draw and the args it reads can never name different
        // groups. `instance_count` is written as **zero** and only the GPU ever
        // raises it — the `inf_vgeom` reset, at one args block per (page, group).
        let mut draw_raw = Vec::with_capacity(pages.len() * groups.len());
        let mut arg_raw: Vec<u32> = Vec::with_capacity(pages.len() * groups.len() * 5);
        for (i, p) in pages.iter().enumerate() {
            for (g, geom) in groups.iter().enumerate() {
                draw_raw.push(VsmPageDrawRaw {
                    view_proj: p.view_proj.to_cols_array(),
                    info: [
                        i as u32 * caster_count + group_first(&groups, g),
                        p.slot,
                        g as u32,
                        0,
                    ],
                    _pad: [0; 44],
                });
                arg_raw.extend_from_slice(&[
                    geom.index_count,
                    0,
                    geom.first_index,
                    geom.base_vertex as u32,
                    0,
                ]);
            }
        }
        gpu.queue
            .write_buffer(&self.draws, 0, bytemuck::cast_slice(&draw_raw));
        gpu.queue
            .write_buffer(&self.args, 0, bytemuck::cast_slice(&arg_raw));

        // Bind groups are rebuilt only when a buffer moved (`ensure` clears them),
        // which is the cheap half of `crate::vt`'s generation discipline: a bind
        // group over a reallocated buffer keeps a dead resource alive and `wgpu`
        // validates nothing about it.
        if self.cull_bind.is_none() {
            self.cull_bind = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vsm-cull"),
                layout: &self.cull_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.pages.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.casters.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.visible.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.args.as_entire_binding(),
                    },
                ],
            }));
        }
        if self.caster_bind.is_none() {
            self.caster_bind = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vsm-casters"),
                layout: &self.caster_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.casters.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.visible.as_entire_binding(),
                    },
                ],
            }));
        }
        if self.page_bind.is_none() {
            self.page_bind = Some(page_bind_group(&gpu.device, &self.page_bgl, &self.draws));
        }

        // ── the cull: one thread per (caster, page) pair.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vsm-cull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull_pipeline);
            pass.set_bind_group(0, self.cull_bind.as_ref().expect("just built"), &[]);
            pass.dispatch_workgroups(caster_count.div_ceil(64).max(1), page_count, 1);
        }

        // ── the raster: ONE pass over the atlas, one viewport per page.
        let caster_bind = self.caster_bind.as_ref().expect("just built");
        let page_bind = self.page_bind.as_ref().expect("just built");
        let mut draws = 0u64;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vsm-pages"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: atlas,
                    depth_ops: Some(wgpu::Operations {
                        // Reverse-Z: 0 is "far / nothing", so a slot nothing draws
                        // into reads as no caster at all.
                        load: wgpu::LoadOp::Clear(VSM_DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let rigid = if masked {
                &self.rigid_masked
            } else {
                &self.rigid
            };
            let mut bound_skinned = false;
            pass.set_pipeline(rigid);
            pass.set_bind_group(1, caster_bind, &[]);
            for (i, p) in pages.iter().enumerate() {
                // **The page rect, pinned exactly.** `slot_origin` is the atlas
                // texel of the slot's corner and `stored_page_size` its side, so
                // this is the 128 × 128 rectangle `inf_vsm` planned and not a
                // rounding of it.
                let (x, y) = (p.rect.0, p.rect.1);
                let s = p.rect.2;
                pass.set_viewport(x as f32, y as f32, s as f32, s as f32, 0.0, 1.0);
                pass.set_scissor_rect(x, y, s, s);
                for (g, geom) in groups.iter().enumerate() {
                    if geom.casters == 0 || geom.index_count == 0 {
                        continue;
                    }
                    // Bound per group rather than once per pass: the built-in
                    // primitives share one pair of buffers, and every other caster
                    // path brings its own. The pipeline switches only for the
                    // skinned path, whose vertex format is a different one —
                    // groups 0 and 1 survive the switch because both layouts name
                    // the same two at the same indices.
                    let wants_skinned = matches!(geom.source, GroupSource::Skinned { .. });
                    if wants_skinned != bound_skinned {
                        pass.set_pipeline(if wants_skinned {
                            &self.skinned_pipeline
                        } else {
                            rigid
                        });
                        bound_skinned = wants_skinned;
                    }
                    match geom.source {
                        GroupSource::Prim => self.prim.bind_geometry(&mut pass),
                        GroupSource::Vgeom { asset, level } => {
                            let Some(a) = self.vgeom.get(&asset) else {
                                continue;
                            };
                            let Some((indices, _)) = a.levels.get(level) else {
                                continue;
                            };
                            pass.set_vertex_buffer(0, a.vertices.slice(..));
                            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                        }
                        GroupSource::Skinned { instance } => {
                            let Some(mesh_index) =
                                self.skinned_palettes.get(instance).map(|(_, _, m)| *m)
                            else {
                                continue;
                            };
                            let Some(m) = self.skinned.get(mesh_index) else {
                                continue;
                            };
                            pass.set_bind_group(2, &self.skinned_palettes[instance].1, &[]);
                            pass.set_vertex_buffer(0, m.vertices.slice(..));
                            pass.set_index_buffer(m.indices.slice(..), wgpu::IndexFormat::Uint32);
                        }
                        GroupSource::Terrain { terrain, tile } => {
                            let Some(t) = self.terrain.get(&(terrain, tile)) else {
                                continue;
                            };
                            pass.set_vertex_buffer(0, t.vertices.slice(..));
                            pass.set_index_buffer(t.indices.slice(..), wgpu::IndexFormat::Uint32);
                        }
                    }
                    let slot = (i * groups.len() + g) as u64;
                    pass.set_bind_group(
                        0,
                        page_bind,
                        &[(slot * VSM_PAGE_DRAW_STRIDE) as wgpu::DynamicOffset],
                    );
                    // Issued whether or not the cull found anything: the drawn set
                    // lives entirely on the GPU, and a CPU-side assumption about it
                    // is the shortcut that turns into missing geometry (the
                    // `inf_vgeom` rule, quoted because it is the same rule).
                    pass.draw_indexed_indirect(&self.args, slot * VSM_ARG_WORDS * 4);
                    draws += 1;
                }
            }
        }

        self.last_pages = pages
            .iter()
            .map(|p| (VsmLightHandle(p.light), p.page, p.rect))
            .collect();
        self.last_groups = groups.len();
        self.last_casters = casters;
        self.stats.frames += 1;
        self.stats.pages += u64::from(page_count);
        self.stats.draws += draws;
        self.stats.casters += u64::from(caster_count);
        self.stats.scatter_casters += u64::from(scattered);
        let of = |f: fn(&GroupSource) -> bool| -> u64 {
            groups
                .iter()
                .filter(|g| f(&g.source))
                .map(|g| u64::from(g.casters))
                .sum()
        };
        self.stats.vgeom_casters += of(|s| matches!(s, GroupSource::Vgeom { .. }));
        self.stats.skinned_casters += of(|s| matches!(s, GroupSource::Skinned { .. }));
        self.stats.terrain_casters += of(|s| matches!(s, GroupSource::Terrain { .. }));
        self.stats.masked_frames += u64::from(masked);
        self.stats.dropped_casters += u64::from(dropped);
        self.stats.dropped_groups += u64::from(dropped_groups);
        self.stats.vgeom_level_sum += level_sum;
        if dropped > 0 {
            tracing::warn!(
                "inf-render: {dropped} shadow casters past this frame's ceilings \
                 ({} casters, {} geometry groups) cast nothing, {dropped_groups} of \
                 them whole groups; raise a ceiling or reduce the caster set",
                VSM_MAX_CASTERS,
                VSM_MAX_GROUPS
            );
        }
        page_count
    }

    /// The resident pages this frame will rasterize, in **slot order** — a total
    /// order that is a function of the residency alone, so two runs of one want
    /// sequence rasterize the same pages in the same order.
    ///
    /// Walking slots rather than the address space is not just cheaper (an atlas
    /// has thousands of slots and a clipmap has millions of addresses): it is what
    /// makes the set exactly "the pages a receiver could read", because a page that
    /// occupies no slot is one the table resolves past.
    fn collect_pages(
        &mut self,
        residency: &VsmResidency,
        projections: &[VsmProjection],
        proj_base: &[u32],
    ) -> Vec<PageDraw> {
        let geometry = residency.geometry();
        let mut out = Vec::new();
        let mut deferred = 0u64;
        for slot in 0..geometry.slot_count() {
            let Some((light, page)) = residency.slot_occupant(slot) else {
                continue;
            };
            if out.len() as u32 >= VSM_MAX_RASTER_PAGES {
                deferred += 1;
                continue;
            }
            let Some(desc) = residency.desc(light) else {
                continue;
            };
            let Some(g) = desc.levels.get(page.level as usize) else {
                continue;
            };
            let Some(&base) = proj_base.get(light.index()) else {
                continue;
            };
            let Some(proj) = projections.get((base + page.face) as usize) else {
                continue;
            };
            let Some((x, y)) = geometry.slot_origin(slot) else {
                continue;
            };
            out.push(PageDraw {
                view_proj: vsm_page_matrix(
                    Mat4::from_cols_array(&proj.view_proj),
                    desc.kind,
                    page.level,
                    g.pages_x,
                    g.pages_y,
                    page.x,
                    page.y,
                ),
                rect: (x, y, geometry.stored_page_size),
                slot,
                light: light.0,
                page,
            });
        }
        if deferred > 0 {
            self.stats.deferred_pages += deferred;
            tracing::warn!(
                "inf-render: {deferred} resident shadow pages past the {} \
                 rasterized this frame keep the depth the clear left them (lit)",
                VSM_MAX_RASTER_PAGES
            );
        }
        out
    }

    /// Build (and evict) the caster geometry for the scene's vgeom assets.
    ///
    /// Content-addressed by asset id and built **once**: a `.inf_vmesh`'s classic
    /// chain is derived from a cooked DAG, so it cannot change without the id
    /// changing (the P18.3 derived-id discipline). Eviction is the same rule
    /// `passes::classic_vgeom` uses — keep exactly what the scene names.
    fn sync_vgeom(&mut self, gpu: &GpuContext, scene: &RenderScene) {
        let live: std::collections::BTreeSet<u128> =
            scene.vgeom_assets.iter().map(|a| a.id).collect();
        self.vgeom.retain(|id, _| live.contains(id));
        for asset in &scene.vgeom_assets {
            if self.vgeom.contains_key(&asset.id) {
                continue;
            }
            let Ok(mesh) = asset.source.to_mesh() else {
                // Reported, not swallowed: a caster set that silently lost an
                // asset reads as "that building stopped casting".
                tracing::warn!(
                    "inf-render: shadow casters for vmesh {:#x} could not be \
                     decoded; it casts nothing",
                    asset.id
                );
                continue;
            };
            let verts: Vec<crate::passes::mesh::MeshVertex> = mesh
                .vertices
                .iter()
                .map(|v| crate::passes::mesh::MeshVertex {
                    pos: v.position,
                    normal: v.normal,
                    uv: v.uv,
                })
                .collect();
            let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vsm-vgeom-verts"),
                size: (std::mem::size_of::<crate::passes::mesh::MeshVertex>() * verts.len().max(1))
                    as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !verts.is_empty() {
                gpu.queue
                    .write_buffer(&vertices, 0, bytemuck::cast_slice(&verts));
            }
            let mut errors = Vec::new();
            let mut levels = Vec::new();
            for l in mesh.classic_lods() {
                errors.push(l.error);
                let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("vsm-vgeom-indices"),
                    size: (std::mem::size_of::<u32>() * l.indices.len().max(1)) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                if !l.indices.is_empty() {
                    gpu.queue
                        .write_buffer(&buf, 0, bytemuck::cast_slice(&l.indices));
                }
                levels.push((buf, l.indices.len() as u32));
            }
            self.vgeom.insert(
                asset.id,
                VgeomCasterGeom {
                    vertices,
                    levels,
                    errors,
                    bounds: asset.bounds(),
                },
            );
        }
    }

    /// Upload the bind-pose geometry of every skinned mesh and one palette buffer
    /// per skinned instance.
    ///
    /// Keyed on `scene.version` and rebuilt whole when it moves, because a
    /// palette **is** the animation: it changes every frame by construction, and
    /// a cache keyed on anything finer would be a cache that always misses. The
    /// bind-pose vertices are re-uploaded with it, which is the one wasteful half
    /// — `passes::skinned` avoids it by caching on `Arc` pointer identity, and
    /// doing the same here is a P27.3 refinement rather than a correctness one.
    fn sync_skinned(&mut self, gpu: &GpuContext, scene: &RenderScene) {
        if self.skinned_version == Some(scene.version) {
            return;
        }
        self.skinned_version = Some(scene.version);
        self.skinned.clear();
        self.skinned_palettes.clear();
        for mesh in &scene.skinned_meshes {
            let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vsm-skinned-verts"),
                size: (std::mem::size_of::<crate::scene::SkinnedVertex>()
                    * mesh.vertices.len().max(1)) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let indices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vsm-skinned-indices"),
                size: (std::mem::size_of::<u32>() * mesh.indices.len().max(1)) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !mesh.vertices.is_empty() {
                gpu.queue
                    .write_buffer(&vertices, 0, bytemuck::cast_slice(&mesh.vertices));
            }
            if !mesh.indices.is_empty() {
                gpu.queue
                    .write_buffer(&indices, 0, bytemuck::cast_slice(&mesh.indices));
            }
            // The bind pose's bound, inflated: see `SKINNED_POSE_MARGIN`.
            let mut lo = glam::Vec3::splat(f32::INFINITY);
            let mut hi = glam::Vec3::splat(f32::NEG_INFINITY);
            for v in &mesh.vertices {
                let p = glam::Vec3::from(v.pos);
                lo = lo.min(p);
                hi = hi.max(p);
            }
            let bounds = if mesh.vertices.is_empty() {
                (glam::Vec3::ZERO, 0.0)
            } else {
                let c = 0.5 * (lo + hi);
                (c, (hi - c).length() * (1.0 + SKINNED_POSE_MARGIN))
            };
            self.skinned.push(SkinnedCasterGeom {
                vertices,
                indices,
                index_count: mesh.indices.len() as u32,
                bounds,
            });
        }
        for inst in &scene.skinned {
            let n = inst.palette.len().max(1);
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vsm-skinned-palette"),
                size: (n * std::mem::size_of::<glam::Mat4>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !inst.palette.is_empty() {
                let raw: Vec<[f32; 16]> = inst.palette.iter().map(|m| m.to_cols_array()).collect();
                gpu.queue
                    .write_buffer(&buffer, 0, bytemuck::cast_slice(&raw));
            }
            let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vsm-skinned-palette"),
                layout: &self.palette_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            self.skinned_palettes.push((buffer, bind, inst.mesh));
        }
    }

    /// Build (and evict) one static caster mesh per resident terrain tile.
    ///
    /// Rebuilt when the tile's version moves **or** the floating origin rebases,
    /// which is the same key every packed buffer in this tree carries — a mesh
    /// in render-local space is a mesh under one origin.
    fn sync_terrain(&mut self, gpu: &GpuContext, scene: &RenderScene, view: &RenderView) {
        let origin = view.origin.origin();
        let mut live = std::collections::BTreeSet::new();
        for terrain in &scene.terrains {
            let res = terrain.tile_resolution.max(2);
            for tile in &terrain.tiles {
                let key = (terrain.id, tile.key);
                live.insert(key);
                if self
                    .terrain
                    .get(&key)
                    .is_some_and(|g| g.version == tile.version && g.origin == origin)
                {
                    continue;
                }
                let expect = (res * res) as usize;
                if tile.heights.len() < expect {
                    continue;
                }
                let span = terrain.tile_span_at(tile.key.lod) as f32;
                let cells = VSM_TERRAIN_CASTER_CELLS.min(res - 1).max(1);
                let step = (res - 1) as f32 / cells as f32;
                let o = view.origin.to_render(tile.origin);
                let mut verts: Vec<crate::passes::mesh::MeshVertex> =
                    Vec::with_capacity(((cells + 1) * (cells + 1)) as usize);
                let mut lo = glam::Vec3::splat(f32::INFINITY);
                let mut hi = glam::Vec3::splat(f32::NEG_INFINITY);
                for j in 0..=cells {
                    for i in 0..=cells {
                        let (si, sj) = (
                            ((i as f32 * step).round() as u32).min(res - 1),
                            ((j as f32 * step).round() as u32).min(res - 1),
                        );
                        let h = tile.heights[(sj * res + si) as usize];
                        let u = si as f32 / (res - 1) as f32;
                        let v = sj as f32 / (res - 1) as f32;
                        let p = glam::Vec3::new(o.x + u * span, o.y + h, o.z + v * span);
                        lo = lo.min(p);
                        hi = hi.max(p);
                        verts.push(crate::passes::mesh::MeshVertex {
                            pos: [p.x, p.y, p.z],
                            normal: [0.0, 1.0, 0.0],
                            uv: [u, v],
                        });
                    }
                }
                let stride = cells + 1;
                let mut indices: Vec<u32> = Vec::with_capacity((cells * cells * 6) as usize);
                for j in 0..cells {
                    for i in 0..cells {
                        let a = j * stride + i;
                        let b = a + 1;
                        let c = a + stride;
                        let d = c + 1;
                        indices.extend_from_slice(&[a, c, b, b, c, d]);
                    }
                }
                let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("vsm-terrain-verts"),
                    size: (std::mem::size_of::<crate::passes::mesh::MeshVertex>() * verts.len())
                        as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue
                    .write_buffer(&vertices, 0, bytemuck::cast_slice(&verts));
                let index_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("vsm-terrain-indices"),
                    size: (std::mem::size_of::<u32>() * indices.len()) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue
                    .write_buffer(&index_buf, 0, bytemuck::cast_slice(&indices));
                let centre = 0.5 * (lo + hi);
                self.terrain.insert(
                    key,
                    TerrainCasterGeom {
                        vertices,
                        indices: index_buf,
                        index_count: indices.len() as u32,
                        bounds: (centre, (hi - centre).length()),
                        version: tile.version,
                        origin,
                    },
                );
            }
        }
        self.terrain.retain(|k, _| live.contains(k));
    }

    /// Grow the buffers to hold `(pages, casters, groups)`, dropping every bind
    /// group that named one of them.
    fn ensure(&mut self, gpu: &GpuContext, pages: u32, casters: u32, groups: u32) {
        let mut grew = false;
        let mut grow = |buf: &mut wgpu::Buffer,
                        cap: &mut u64,
                        want: u64,
                        unit: u64,
                        label: &str,
                        usage: wgpu::BufferUsages| {
            if *cap >= want {
                return;
            }
            let n = want.next_power_of_two().max(4);
            *buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n * unit).max(256),
                usage,
                mapped_at_creation: false,
            });
            *cap = n;
            grew = true;
        };
        grow(
            &mut self.visible,
            &mut self.visible_entries,
            u64::from(pages) * u64::from(casters),
            4,
            "vsm-visible",
            wgpu::BufferUsages::STORAGE,
        );
        grow(
            &mut self.args,
            &mut self.args_words,
            u64::from(pages) * u64::from(groups) * VSM_ARG_WORDS,
            4,
            "vsm-draw-args",
            ARGS_USAGE,
        );
        grow(
            &mut self.casters,
            &mut self.caster_capacity,
            u64::from(casters),
            std::mem::size_of::<VsmCasterRaw>() as u64,
            "vsm-casters",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        grow(
            &mut self.pages,
            &mut self.page_capacity,
            u64::from(pages),
            std::mem::size_of::<VsmPageRaw>() as u64,
            "vsm-pages",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        grow(
            &mut self.draws,
            &mut self.draw_capacity,
            u64::from(pages) * u64::from(groups),
            VSM_PAGE_DRAW_STRIDE,
            "vsm-page-draws",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        if grew {
            self.cull_bind = None;
            self.caster_bind = None;
            self.page_bind = None;
        }
    }
}

/// One page's raster state.
struct PageDraw {
    view_proj: Mat4,
    /// `(x, y, side)` in atlas texels.
    rect: (u32, u32, u32),
    slot: u32,
    light: u32,
    page: VsmPage,
}

/// **Whether one more geometry group fits this frame's ceiling** — one door for
/// the three paths that push a group (vgeom, skinned, terrain).
///
/// Three copies of a four-line refusal is how two of them end up not refusing
/// (the one-door law, P27.2 audit). `carries` is the caster count the group would
/// have held, so a refused group's casters land in `dropped` and are never silent.
fn admit_group(groups: usize, carries: u32, dropped: &mut u32, dropped_groups: &mut u32) -> bool {
    if groups as u32 >= VSM_MAX_GROUPS {
        *dropped_groups += 1;
        *dropped += carries;
        return false;
    }
    true
}

/// The first caster of group `g` — the base of that group's slice inside every
/// page's visible list.
fn group_first(groups: &[GroupGeom], g: usize) -> u32 {
    groups[..g].iter().map(|x| x.casters).sum()
}

/// The shared page-uniform bind-group layout: one dynamic-offset uniform, read in
/// the vertex stage.
pub(crate) fn page_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vsm-page"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn page_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    draws: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vsm-page"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: draws,
                offset: 0,
                size: std::num::NonZeroU64::new(std::mem::size_of::<VsmPageDrawRaw>() as u64),
            }),
        }],
    })
}

/// The primitive state every page pipeline shares.
///
/// **No face culling, deliberately.** `passes::shadow` culls front faces to fight
/// acne on its box casters; a page raster draws terrain patches and open skinned
/// shells too, and dropping a face of those deletes a caster rather than biasing
/// it. Reverse-Z + `Greater` already keeps the NEAREST surface, which is the
/// value a receiver wants; the bias story is re-derived at P27.4 with the
/// receiver in hand (`docs/memos/p27-1-depth-convention.md` says so explicitly),
/// which is also why there is no `DepthBiasState` here — the cascade's
/// `constant: 2, slope_scale: 2.0` is tuned for forward-Z and would push the
/// wrong way.
pub(crate) fn page_primitive_state() -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        cull_mode: None,
        ..Default::default()
    }
}

/// The depth state every page pipeline shares — the camera's convention, through
/// the camera's own constants.
pub(crate) fn page_depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(VSM_DEPTH_COMPARE),
        stencil: Default::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// What one frame's [`pack_casters`] produced.
///
/// A struct rather than a tuple because the tuple reached five members and its own
/// doc comment described four of them (P27.2 audit) — the shape that lets a
/// counter be added and never read.
struct CasterPack {
    casters: Vec<VsmCasterRaw>,
    /// Sorted by group, which is what makes a page's visible list overflow-proof:
    /// a group's slice is its own caster count long.
    groups: Vec<GroupGeom>,
    /// Any caster carries the R-P5 masked blend code, so the alpha-testing
    /// pipeline is bound rather than the fragment-less one.
    masked: bool,
    /// Of the casters, the ones that came from GPU-scatter batches.
    scattered: u32,
    /// Casters the two ceilings refused.
    dropped: u32,
    /// Geometry groups [`VSM_MAX_GROUPS`] refused.
    dropped_groups: u32,
    /// The classic LOD level each packed vgeom caster drew at, summed.
    level_sum: u64,
}

/// **Pack this frame's caster records** — the rigid instances and the GPU-scatter
/// batches, through `passes::shadow`'s own doors.
#[allow(clippy::too_many_arguments)]
fn pack_casters(
    prim: &PrimGpu,
    vgeom: &std::collections::BTreeMap<u128, VgeomCasterGeom>,
    skinned: &[SkinnedCasterGeom],
    terrain: &std::collections::BTreeMap<(u64, crate::scene::TerrainTileKey), TerrainCasterGeom>,
    scene: &RenderScene,
    view: &RenderView,
    settings: &crate::settings::RenderSettings,
) -> CasterPack {
    let origin = &view.origin;
    // Opaque + masked only. Translucent geometry does not cast — the cascade's
    // scope, kept, so the two shadow paths agree about what a caster is.
    let (rigid, ranges, _translucent) =
        crate::passes::mesh::pack_bucketed(origin, &scene.instances);
    // Scatter: the same CPU pack the cascade uses, at VSM's own reach. A clipmap's
    // range is its coarsest level's extent rather than `ShadowSettings::
    // max_distance`, and reading the cascade's number here would make foliage stop
    // casting at 60 m in an atlas that reaches kilometres.
    let vsm = &settings.vsm;
    let reach = vsm.first_level_extent_m.max(1.0)
        * (1u32 << vsm.clipmap_levels.clamp(1, 16).saturating_sub(1)) as f32;
    let scatter_eye = origin.to_world(view.eye_local());
    let caster_settings = crate::passes::scatter::shadow_caster_settings(&settings.scatter, reach);
    let pack = crate::passes::scatter::pack_fallback(
        origin,
        &scene.scatter,
        crate::passes::scatter::bucket_center(scatter_eye),
        &caster_settings,
        crate::passes::scatter::MAX_CPU_SCATTER_INSTANCES,
    );
    let scattered = pack.instances.len() as u32;
    let (merged, ranges) =
        crate::passes::scatter::merge_bucketed((rigid, ranges), (pack.instances, pack.ranges));

    let mut casters = Vec::with_capacity(merged.len());
    let mut groups = Vec::with_capacity(PrimMesh::ALL.len());
    let mut masked = false;
    let mut first = 0u32;
    // The ceilings count what they refuse. Every `break`/`continue` on
    // `VSM_MAX_CASTERS` or `VSM_MAX_GROUPS` below increments one of these, so a
    // truncated caster set is a number in the summary rather than a level whose
    // far half stopped casting.
    let mut dropped = 0u32;
    let mut dropped_groups = 0u32;
    // The clipmap level each vgeom caster drew at, summed — the counter that makes
    // "vgeom casts at the CAMERA's LOD" a measurement (P27.2 audit). Nothing else
    // can see it: a shadow drawn from the coarsest level of the chain is still a
    // shadow, in the same pages, at almost the same depths.
    let mut level_sum = 0u64;
    for (k, kind) in PrimMesh::ALL.iter().enumerate() {
        let range = ranges[k].clone();
        let unit = kind.bounding_radius();
        let mut count = 0u32;
        for (local, raw) in merged[range.start as usize..range.end as usize]
            .iter()
            .enumerate()
        {
            if casters.len() as u32 >= VSM_MAX_CASTERS {
                dropped += (range.end - range.start) - local as u32;
                break;
            }
            let model = Mat4::from_cols_array(&raw.model);
            // The sphere from the matrix rather than from the source instance, so
            // one derivation serves rigid and scattered casters alike (a scatter
            // batch reaches here as `InstanceRaw` and has no `MeshInstance`).
            let scale = model
                .x_axis
                .truncate()
                .length()
                .max(model.y_axis.truncate().length())
                .max(model.z_axis.truncate().length());
            let centre = model.w_axis.truncate();
            masked |= raw.pbr[3] > 0.5 && raw.pbr[3] < 1.5;
            casters.push(VsmCasterRaw {
                model: raw.model,
                sphere: [centre.x, centre.y, centre.z, unit * scale],
                mat: [raw.pbr[2], raw.pbr[3], raw.color[3], 0.0],
                ids: [k as u32, local as u32, first, 0],
            });
            count += 1;
        }
        let r = prim.range(*kind);
        groups.push(GroupGeom {
            source: GroupSource::Prim,
            index_count: r.index_count,
            first_index: r.index_start,
            base_vertex: r.base_vertex,
            casters: count,
        });
        first += count;
    }

    // ── virtualized geometry ─────────────────────────────────────────
    //
    // One group per (asset, level), and the level is picked by the SAME rule the
    // classic tier picks with — `lod_threshold` against the camera, then
    // `pick_classic_level`. Against the camera and not against the page,
    // deliberately: a VSM page exists because a visible pixel asked for it, so the
    // camera's tolerance is the one that bounds how much silhouette detail the
    // shadow can possibly show. Per-page LOD is a P27.3 question and the memo says
    // so.
    let mut by_group: std::collections::BTreeMap<(u128, usize), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, inst) in scene.vgeom_instances.iter().enumerate() {
        let Some(a) = vgeom.get(&inst.asset) else {
            continue;
        };
        if a.errors.is_empty() {
            continue;
        }
        // **The classic tier's own door**, not a second copy of its five lines
        // (P27.2 audit): `classic_vgeom::instance_threshold` is what
        // `passes::classic_vgeom` picks with, so "the same rule at the same
        // `pixel_error`" is one function rather than an agreement.
        let threshold = crate::passes::classic_vgeom::instance_threshold(
            origin,
            view,
            a.bounds,
            inst,
            settings.vgeom.pixel_error,
        );
        let level = inf_vgeom::pick_classic_level(&a.errors, threshold);
        by_group.entry((inst.asset, level)).or_default().push(i);
    }
    for ((asset, level), members) in by_group {
        let a = &vgeom[&asset];
        let Some(&(_, index_count)) = a.levels.get(level) else {
            continue;
        };
        if !admit_group(
            groups.len(),
            members.len() as u32,
            &mut dropped,
            &mut dropped_groups,
        ) {
            continue;
        }
        let mut count = 0u32;
        for (local, &i) in members.iter().enumerate() {
            if casters.len() as u32 >= VSM_MAX_CASTERS {
                dropped += (members.len() - local) as u32;
                break;
            }
            let inst = &scene.vgeom_instances[i];
            let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
            let max_scale = inst.scale.abs().max_element().max(1e-6);
            let centre = model.transform_point3(glam::Vec3::from(a.bounds.0));
            casters.push(VsmCasterRaw {
                model: model.to_cols_array(),
                sphere: [centre.x, centre.y, centre.z, a.bounds.1 * max_scale],
                // Opaque: virtualized geometry has no per-instance blend code, so
                // it takes the fragment-less path exactly as it does in the classic
                // tier.
                mat: [0.0, 0.0, 1.0, 0.0],
                ids: [groups.len() as u32, local as u32, first, 0],
            });
            level_sum += level as u64;
            count += 1;
        }
        groups.push(GroupGeom {
            source: GroupSource::Vgeom { asset, level },
            index_count,
            first_index: 0,
            base_vertex: 0,
            casters: count,
        });
        first += count;
    }

    // ── skinned ────────────────────────────────────────────────────────
    //
    // One group per INSTANCE, because the joint palette is per instance and it is
    // a bind group. Each group therefore holds exactly one caster, and the cull's
    // verdict for it is 0 or 1 — which is still the per-page cull doing its job,
    // just at the granularity a character has.
    for (i, inst) in scene.skinned.iter().enumerate() {
        if casters.len() as u32 >= VSM_MAX_CASTERS {
            dropped += (scene.skinned.len() - i) as u32;
            break;
        }
        let Some(mesh) = skinned.get(inst.mesh) else {
            continue;
        };
        if mesh.index_count == 0 {
            continue;
        }
        if !admit_group(groups.len(), 1, &mut dropped, &mut dropped_groups) {
            continue;
        }
        let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
        let max_scale = inst.scale.abs().max_element().max(1e-6);
        let centre = model.transform_point3(mesh.bounds.0);
        casters.push(VsmCasterRaw {
            model: model.to_cols_array(),
            sphere: [centre.x, centre.y, centre.z, mesh.bounds.1 * max_scale],
            mat: [0.0, 0.0, 1.0, 0.0],
            ids: [groups.len() as u32, 0, first, 0],
        });
        groups.push(GroupGeom {
            source: GroupSource::Skinned { instance: i },
            index_count: mesh.index_count,
            first_index: 0,
            base_vertex: 0,
            casters: 1,
        });
        first += 1;
    }

    // ── terrain ────────────────────────────────────────────────────────
    //
    // One group per resident tile, drawn from the tile's OWN heights rather than
    // from the camera-fitted clipmap patch the terrain pass draws — see
    // `TerrainCasterGeom`. The model matrix is the identity: the mesh is already
    // in render-local space, which is what lets it be cached across frames.
    for ((tid, tkey), geom) in terrain {
        if geom.index_count == 0 {
            continue;
        }
        if casters.len() as u32 >= VSM_MAX_CASTERS {
            dropped += 1;
            continue;
        }
        if !admit_group(groups.len(), 1, &mut dropped, &mut dropped_groups) {
            continue;
        }
        casters.push(VsmCasterRaw {
            model: glam::Mat4::IDENTITY.to_cols_array(),
            sphere: [
                geom.bounds.0.x,
                geom.bounds.0.y,
                geom.bounds.0.z,
                geom.bounds.1,
            ],
            mat: [0.0, 0.0, 1.0, 0.0],
            ids: [groups.len() as u32, 0, first, 0],
        });
        groups.push(GroupGeom {
            source: GroupSource::Terrain {
                terrain: *tid,
                tile: *tkey,
            },
            index_count: geom.index_count,
            first_index: 0,
            base_vertex: 0,
            casters: 1,
        });
        first += 1;
    }
    CasterPack {
        casters,
        groups,
        masked,
        scattered,
        dropped,
        dropped_groups,
        level_sum,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Stride is contract** (the standing law): the two WGSL files read these
    /// records out of storage buffers, where `wgpu` validates nothing, so the
    /// layout is pinned in Rust *and* the shader's own declaration is read.
    #[test]
    fn the_caster_record_matches_the_shader_that_reads_it() {
        assert_eq!(std::mem::size_of::<VsmCasterRaw>(), 112);
        assert_eq!(std::mem::size_of::<VsmPageRaw>(), 80);
        assert_eq!(
            std::mem::size_of::<VsmPageDrawRaw>() as u64,
            VSM_PAGE_DRAW_STRIDE
        );
        for src in [
            include_str!("shaders/vsm_cull.wgsl"),
            include_str!("shaders/vsm_caster.wgsl"),
        ] {
            // The field ORDER is what a `@repr(C)` mirror rests on, so it is read
            // rather than counted: four members, in this sequence.
            let decl = src
                .split("struct VsmCaster {")
                .nth(1)
                .expect("both files declare the caster record");
            let body = decl.split('}').next().expect("a closed struct");
            let fields: Vec<&str> = body
                .lines()
                .filter_map(|l| l.trim().split(':').next())
                .filter(|f| !f.is_empty() && !f.starts_with("//"))
                .collect();
            assert_eq!(fields, ["model", "sphere", "mat", "ids"], "{body}");
        }
        // …and the args stride the draw offsets are computed from is the one the
        // cull indexes with.
        assert!(include_str!("shaders/vsm_cull.wgsl")
            .contains(&format!("const VSM_ARG_WORDS: u32 = {VSM_ARG_WORDS}u;")));
    }

    /// **"The identical predicate" is one spelling, and now it is read** (P27.2
    /// audit).
    ///
    /// The ledger says the page raster alpha-tests with "the identical predicate
    /// `mesh.wgsl`, `depth_prepass.wgsl` and `shadow_depth.wgsl` all spell". That
    /// was a claim about four *copies* of a predicate with nothing comparing them —
    /// the one-door law's own failure shape, and the R-P5 blend code is a wire
    /// constant that only a copy can drift from. There is no way to share a
    /// fragment body across four standalone WGSL modules, so the door is this arm:
    /// the three depth-only casters spell it character for character, and the lit
    /// pass spells the same structure against its own field names.
    #[test]
    fn every_caster_pass_alpha_tests_with_one_predicate() {
        const DEPTH_ONLY: &str = "if (in.blend > 0.5 && in.blend < 1.5 && in.alpha < in.cutoff) {";
        for (label, src) in [
            ("vsm_caster", include_str!("shaders/vsm_caster.wgsl")),
            ("shadow_depth", include_str!("shaders/shadow_depth.wgsl")),
            ("depth_prepass", include_str!("shaders/depth_prepass.wgsl")),
        ] {
            assert_eq!(
                src.matches(DEPTH_ONLY).count(),
                1,
                "{label}.wgsl does not spell the masked predicate exactly once; a \
                 cut-out that shadows as a solid is the symptom"
            );
        }
        // The lit pass reads the same two words out of its own interpolants
        // (`pbr.w` is the blend code, `pbr.z` the cutoff, `color.a` the alpha), so
        // it is pinned by structure rather than by text — the numbers 0.5 and 1.5
        // are the freeze-pinned R-P5 window and they are what a copy drifts on.
        assert_eq!(
            include_str!("shaders/mesh.wgsl")
                .matches("if (in.pbr.w > 0.5 && in.pbr.w < 1.5 && in.color.a < in.pbr.z) {")
                .count(),
            1,
            "mesh.wgsl's masked predicate moved away from the caster passes'"
        );
        // ANTI-VACUITY: the Rust side that decides WHICH pipeline to bind reads the
        // same window, so a caster the CPU calls opaque cannot be discarded by a
        // shader that calls it masked.
        assert!(include_str!("vsm_raster.rs")
            .contains("masked |= raw.pbr[3] > 0.5 && raw.pbr[3] < 1.5;"));
    }

    /// **The three ceilings multiply out to buffers a default device can make**
    /// (P27.2 audit) — the arithmetic the missing group ceiling got wrong.
    ///
    /// `ensure` rounds every allocation up to a power of two, so the worst case is
    /// `next_power_of_two(pages × N) × unit`, and the *draw uniform* is the one
    /// that binds: 256 bytes an entry against the visible list's 4. The first
    /// write-up named the visible list "the one allocation that grows as a product,
    /// and the reason both ceilings exist" and never noticed that the draw uniform
    /// is the same product 64× larger, with **groups** — uncapped — as its second
    /// factor. A skinned instance is a group and a resident terrain tile is a
    /// group, so `groups ≈ casters` on a real level, and the old worst case is
    /// asserted below as the failure it was.
    #[test]
    fn the_page_rasters_worst_case_buffers_fit_the_default_device_limits() {
        let limit = wgpu::Limits::default().max_buffer_size;
        let pages = u64::from(VSM_MAX_RASTER_PAGES);
        let grown = |entries: u64, unit: u64| entries.next_power_of_two().max(4) * unit;
        let draws = grown(pages * u64::from(VSM_MAX_GROUPS), VSM_PAGE_DRAW_STRIDE);
        let args = grown(pages * u64::from(VSM_MAX_GROUPS) * VSM_ARG_WORDS, 4);
        let visible = grown(pages * u64::from(VSM_MAX_CASTERS), 4);
        for (label, bytes) in [("draws", draws), ("args", args), ("visible", visible)] {
            assert!(
                bytes <= limit,
                "the page raster's {label} buffer is {bytes} B at its ceilings and \
                 `wgpu::Limits::default().max_buffer_size` is {limit} B — that is a \
                 device error at `create_buffer`, not a counted refusal"
            );
        }
        // **The defect, as arithmetic.** Without a group ceiling the second factor
        // is the caster ceiling (one group per skinned instance, one per terrain
        // tile), and that buffer is four times what a default device will allocate.
        let unbounded = grown(pages * u64::from(VSM_MAX_CASTERS), VSM_PAGE_DRAW_STRIDE);
        assert!(
            unbounded > limit,
            "the arm no longer describes the defect it was written for: an uncapped \
             group count gives {unbounded} B against a {limit} B limit"
        );
        // …and the draw uniform really is the binding one, which is why capping
        // casters alone did not bound it.
        assert!(draws > visible && draws > args);
    }

    /// **The group ceiling's door refuses at the ceiling and counts what it
    /// refuses** (P27.2 audit) — the pure half of the arm
    /// `the_group_ceiling_counts_the_groups_it_refuses` makes on a device.
    ///
    /// Both halves are needed and neither is the other: the device arm proves the
    /// *skinned* path calls the door, and this proves the door itself, for the
    /// vgeom and terrain callers a fixture cannot reach a thousand of.
    #[test]
    fn the_group_door_refuses_at_the_ceiling_and_counts_it() {
        let (mut dropped, mut groups) = (0u32, 0u32);
        // One below the ceiling: admitted, nothing counted.
        assert!(admit_group(
            VSM_MAX_GROUPS as usize - 1,
            7,
            &mut dropped,
            &mut groups
        ));
        assert_eq!((dropped, groups), (0, 0));
        // ON it: refused, and the casters the group would have carried are counted
        // rather than lost — so the bound is `>=` and the refusal is never silent.
        assert!(!admit_group(
            VSM_MAX_GROUPS as usize,
            7,
            &mut dropped,
            &mut groups
        ));
        assert_eq!((dropped, groups), (7, 1));
        assert!(!admit_group(
            VSM_MAX_GROUPS as usize + 40,
            1,
            &mut dropped,
            &mut groups
        ));
        assert_eq!((dropped, groups), (8, 2));
        // The rigid groups are pushed unconditionally, so the ceiling must be
        // above them or the very first frame refuses a primitive kind.
        assert!(admit_group(
            PrimMesh::ALL.len(),
            1,
            &mut dropped,
            &mut groups
        ));
        assert_eq!((dropped, groups), (8, 2));
    }

    /// A group's slice of a page's visible list starts at that group's first
    /// caster, so the slices tile the list and cannot overlap — which is why the
    /// cull has no bounds test.
    #[test]
    fn the_group_slices_tile_a_pages_visible_list() {
        let groups = vec![
            GroupGeom {
                source: GroupSource::Prim,
                index_count: 36,
                first_index: 0,
                base_vertex: 0,
                casters: 3,
            },
            GroupGeom {
                source: GroupSource::Prim,
                index_count: 2_304,
                first_index: 36,
                base_vertex: 24,
                casters: 0,
            },
            GroupGeom {
                source: GroupSource::Vgeom { asset: 7, level: 1 },
                index_count: 6,
                first_index: 0,
                base_vertex: 0,
                casters: 5,
            },
        ];
        assert_eq!(group_first(&groups, 0), 0);
        assert_eq!(group_first(&groups, 1), 3);
        // An EMPTY group must not consume a slot, or every group after it is
        // offset by one and the last one runs off the end of the page's slice.
        assert_eq!(group_first(&groups, 2), 3);
        let total: u32 = groups.iter().map(|g| g.casters).sum();
        assert_eq!(group_first(&groups, groups.len()), total);
        assert_eq!(total, 8);
    }

    /// **The one line a host reads is one line** (P27.2 audit).
    ///
    /// Both of this module's user-facing literals shipped with a run of fourteen
    /// and eighteen spaces in the middle of a sentence — the P22 shape, where a
    /// scripted edit ate a `\` continuation and left its indentation behind. Nothing
    /// reads a `tracing::warn!`, but the summary is returned to a caller, and a
    /// counter nobody can read is a counter nobody checks.
    #[test]
    fn the_summary_is_one_line_with_no_mangled_whitespace() {
        let s = VsmRasterStats {
            frames: 3,
            pages: 17,
            draws: 51,
            casters: 9,
            scatter_casters: 2,
            deferred_pages: 1,
            masked_frames: 3,
            vgeom_casters: 4,
            vgeom_level_sum: 5,
            skinned_casters: 1,
            terrain_casters: 2,
            dropped_casters: 6,
            dropped_groups: 1,
        }
        .summary();
        assert!(!s.contains("  "), "a run of spaces in the summary: {s:?}");
        assert!(!s.contains('\n') && !s.contains('\t'), "{s:?}");
        // …and every counter that can be non-zero appears in it, so a field added
        // to the struct and forgotten in the line fails here.
        for n in ["3", "17", "51", "9", "2", "1", "4", "6"] {
            assert!(s.contains(n), "{n} is missing from {s:?}");
        }
    }
}
