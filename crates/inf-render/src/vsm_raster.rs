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
/// entries, so this and [`VSM_MAX_RASTER_PAGES`] are the two factors of the one
/// allocation that can grow without bound.
pub const VSM_MAX_CASTERS: u32 = 16_384;

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

/// The cull's uniform. 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct VsmCullParamsRaw {
    /// x = casters, y = pages, z = groups, w = reserved.
    counts: [u32; 4],
}

/// Where one geometry group's indices live — the three constant words of its
/// indirect args, which the CPU writes and the cull never touches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GroupGeom {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    /// Casters this group holds. Zero ⇒ the draw is skipped on the CPU, which is
    /// the one thing the CPU legitimately knows about the drawn set (how many
    /// casters *could* be visible), as opposed to how many *are*.
    casters: u32,
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
}

impl VsmRasterStats {
    /// A one-line human summary, in the shape the other streamers ship.
    pub fn summary(&self) -> String {
        format!(
            "vsm raster: {} frames, {} pages, {} draws, {} casters ({} scattered), \
             {} pages deferred",
            self.frames,
            self.pages,
            self.draws,
            self.casters,
            self.scatter_casters,
            self.deferred_pages,
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

    /// The pages the last [`record`](Self::record) rasterized, in draw order, and
    /// how many geometry groups each one was drawn with. Kept so a gate can map
    /// an args word back onto the page it decided — the args buffer is a flat
    /// `(page, group)` grid and nothing else in the tree knows its order.
    last_pages: Vec<(VsmLightHandle, VsmPage, (u32, u32, u32))>,
    last_groups: usize,

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
            last_pages: Vec::new(),
            last_groups: 0,
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

    /// The shared page-uniform layout, for the caster paths that live outside this
    /// module's own pipelines (P27.2's skinned / terrain / meshlet rasters).
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
        let (casters, groups, masked, scattered) = pack_casters(&self.prim, scene, view, settings);
        // A page with nothing to draw into it still has to be CLEARED, or it keeps
        // a previous occupant's depth — a slot is reused the moment its page is
        // evicted, and stale depth in a live page is a shadow from geometry that
        // is not there. The pass below clears the whole atlas, so this early
        // return is only taken when there is no pass to open at all.
        if casters.is_empty() {
            return 0;
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
            let pipeline = if masked {
                &self.rigid_masked
            } else {
                &self.rigid
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, caster_bind, &[]);
            self.prim.bind_geometry(&mut pass);
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
        self.stats.frames += 1;
        self.stats.pages += u64::from(page_count);
        self.stats.draws += draws;
        self.stats.casters += u64::from(caster_count);
        self.stats.scatter_casters += u64::from(scattered);
        self.stats.masked_frames += u64::from(masked);
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

/// **Pack this frame's caster records** — the rigid instances and the GPU-scatter
/// batches, through `passes::shadow`'s own doors.
///
/// Returns `(casters, groups, any masked, scatter casters)`. The records are
/// sorted by group, which is what makes a page's visible list overflow-proof: a
/// group's slice is its own caster count long.
fn pack_casters(
    prim: &PrimGpu,
    scene: &RenderScene,
    view: &RenderView,
    settings: &crate::settings::RenderSettings,
) -> (Vec<VsmCasterRaw>, Vec<GroupGeom>, bool, u32) {
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
    for (k, kind) in PrimMesh::ALL.iter().enumerate() {
        let range = ranges[k].clone();
        let unit = kind.bounding_radius();
        let mut count = 0u32;
        for (local, raw) in merged[range.start as usize..range.end as usize]
            .iter()
            .enumerate()
        {
            if casters.len() as u32 >= VSM_MAX_CASTERS {
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
            index_count: r.index_count,
            first_index: r.index_start,
            base_vertex: r.base_vertex,
            casters: count,
        });
        first += count;
    }
    (casters, groups, masked, scattered)
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

    /// A group's slice of a page's visible list starts at that group's first
    /// caster, so the slices tile the list and cannot overlap — which is why the
    /// cull has no bounds test.
    #[test]
    fn the_group_slices_tile_a_pages_visible_list() {
        let groups = vec![
            GroupGeom {
                index_count: 36,
                first_index: 0,
                base_vertex: 0,
                casters: 3,
            },
            GroupGeom {
                index_count: 2_304,
                first_index: 36,
                base_vertex: 24,
                casters: 0,
            },
            GroupGeom {
                index_count: 6,
                first_index: 2_340,
                base_vertex: 410,
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
}
