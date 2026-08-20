//! **The visibility-buffer path's GPU resources** (P28.1): the `Rg32Uint` target
//! and its single-sample depth, the three pipelines, and the per-frame flat
//! instance table the resolve addresses.
//!
//! The passes themselves are recorded by [`crate::passes::vgeom::VgeomNode`],
//! not by a node of their own, and that is a decision rather than an
//! accident. The visibility raster needs the cull compute's output, the shared
//! meshlet pools and the streamer's residency; the resolve needs the same pools
//! *and* the lights and environment binds; the feedback needs the pools again.
//! All four live on `VgeomNode`. A separate node would have to reach them
//! through an `Arc<Mutex<..>>` published across the graph — which is how
//! `SharedStreamReport` carries *statistics*, and is the wrong shape for
//! resources a pass dereferences per draw.
//!
//! # The three passes, in order
//!
//! 1. **`visbuffer`** — the same indirect, vertex-pulled draw the forward path
//!    issues, into `Rg32Uint` + its own single-sample depth. Depth-tested
//!    last-writer-wins.
//! 2. **`vis_resolve`** — one fullscreen triangle into the **MSAA** scene colour
//!    and the **MSAA** scene depth, writing `@builtin(frag_depth)` so the
//!    meshlet depth reaches every pass downstream (translucency, water, the
//!    shadow marking) exactly as the forward raster's did.
//! 3. **`vis_feedback`** — one compute thread per pixel, marking the virtual
//!    texture tiles those pixels actually sampled.
//!
//! # Why the resolve writes into a 4x target from a 1x buffer
//!
//! [`crate::renderer::SCENE_SAMPLES`] is a compile-time `4` and the ROADMAP's
//! clause 1 says *single-sample*. Those are not reconcilable by
//! configuration: a fullscreen fragment shading from a 1x id buffer writes one
//! colour to all four samples of its pixel, so a meshlet silhouette is a hard
//! edge where the forward path resolves four. That is the whole content of the
//! P28.1 MSAA ruling, it is measured by
//! `the_edge_population_is_where_the_two_paths_are_allowed_to_differ` (the first
//! draft of this line named an arm that has never existed — the P20 law, and the
//! P28.1 audit's correction), and it is why this mode is a setting rather than
//! the High-tier default. See `docs/memos/p28-1-visbuffer.md` §5.

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::visbuffer::{VisPackError, VisPacking};

use inf_vgeom::asset::MESHLET_REC_LEN;

/// **Texels a flat instance record occupies** in the instance table's texture.
///
/// `VgeomInstanceGpu` is 176 bytes — eleven `rgba32float` texels — and this is
/// sixteen, so a row is exactly `COPY_BYTES_PER_ROW_ALIGNMENT`. The five wasted
/// texels an instance buy the ability to `write_texture` the whole table in one
/// call: the alignment is a hard requirement above one row, and the alternative
/// is a per-row upload or a staging buffer, either of which is more moving parts
/// than 112 bytes an instance (1.8 MiB at 16 384 instances, and the packing's own
/// ceiling is 16 777 214 since IB-8 — the table is sized by the FRAME, not by the
/// ceiling).
pub const VIS_INSTANCE_TEXELS: u32 = 16;

/// **Why the instance table is a TEXTURE and not a storage buffer.**
///
/// `Limits::default()` grants **8** storage buffers per shader stage, and the
/// resolve is a FRAGMENT pass that binds the shared environment group — which
/// already spends four of them: the GI SH probes (binding 5), the virtual-texture
/// indirection table (16), and the shadow page table and its projections (18, 19).
/// The four meshlet pools are the other four. A fifth storage binding for the
/// instances is nine against eight, and the device says so:
///
/// > Too many bindings of type StorageBuffers in Stage ShaderStages(FRAGMENT),
/// > limit is 8, count was 9.
///
/// This is the P26.3 scarcity ruling one binding class over — there the scarce
/// resource was the bind GROUP and the answer was to fold; here it is the storage
/// binding and the answer is to change the resource kind. A texture costs nothing
/// from that budget, `textureLoad` is valid in the vertex, fragment and compute
/// stages alike (so all three passes read the table through one representation),
/// and `Rgba32Float` with `TextureSampleType::Float { filterable: false }` is core
/// wgpu on every backend.
///
/// **The remaining headroom is zero**, and that is stated rather than discovered:
/// `the_resolve_spends_every_fragment_storage_binding_the_default_limit_grants`
/// counts them, so the next binding the environment group grows fails a test
/// here instead of failing `create_pipeline_layout` on a user's machine.
pub const VIS_FRAGMENT_STORAGE_BINDINGS: u32 = 8;

/// The visibility buffer's own format. `Rg32Uint` because the packing is exactly
/// sixty-four bits (IB-8) and an integer target neither filters nor blends — a
/// visibility id that a driver was free to average would be a different
/// triangle.
pub const VIS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;

/// Bytes one visibility texel occupies — `VIS_ID_BITS / 8`, and the stride the
/// readback walks. Derived rather than written, because a literal here and a
/// format there is how a readback comes to decode half a frame.
pub const VIS_TEXEL_BYTES: u32 = crate::visbuffer::VIS_ID_BITS / 8;

/// Per-asset uniform for the visibility raster: `x` = this asset's base in the
/// flat instance table.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VisFlagsGpu {
    pub flags: [u32; 4],
}

/// The per-pixel feedback pass's uniform. `view_proj` is the **jittered** matrix
/// (the P27.1 law: a reconstruction must use the matrix its depth was
/// rasterized with); `counts` is `(width, height, mask words, texture count)`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VisFeedbackParamsGpu {
    pub view_proj: [f32; 16],
    pub counts: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<VisFeedbackParamsGpu>() == 80);

/// Why a frame did not take the visibility-buffer path, counted rather than
/// logged. Zero on every frame that took it, and the *reason* is carried so a
/// host can tell "the setting is off" from "this scene is too big for the
/// packing" without a debugger.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VisAudit {
    /// Frames that rasterized a visibility buffer.
    pub frames: u32,
    /// Frames refused by [`VisPacking::admit`], by ceiling.
    pub refused_instances: u32,
    pub refused_meshlet_slots: u32,
    pub refused_triangles: u32,
    /// Instances in the flat table on the last admitted frame.
    pub flat_instances: u32,
    /// Meshlet pool slots the last admission measured.
    pub meshlet_slots: u32,
}

impl VisAudit {
    /// Total refusals across the three ceilings — the number a host watches.
    pub fn refused(&self) -> u32 {
        self.refused_instances + self.refused_meshlet_slots + self.refused_triangles
    }

    /// Record one refusal against the ceiling it names. Split per ceiling on the
    /// P27.5 ruling that a single counter would wear both names: "the scene has
    /// too many instances" and "an asset cooks meshlets this format cannot
    /// address" are different problems with different fixes.
    pub fn refuse(&mut self, e: VisPackError) {
        match e {
            VisPackError::Instances { .. } => self.refused_instances += 1,
            VisPackError::MeshletSlots { .. } => self.refused_meshlet_slots += 1,
            VisPackError::TrianglesPerMeshlet { .. } => self.refused_triangles += 1,
        }
    }
}

/// The audit counters the node publishes and the renderer reads.
///
/// A `Mutex` the two share, exactly like `SharedStreamReport`: a node lives
/// inside the render graph and is not reachable from outside it, so a counter a
/// host has to see travels this way and not through an accessor.
pub type SharedVisReport = std::sync::Arc<std::sync::Mutex<VisAudit>>;

/// The viewport-sized targets. Recreated with the frame targets, and keyed on
/// their generation for the same reason every resizable resource in this
/// renderer is (`GenCache`'s invariant): a bind group cached across a resize
/// keeps the old texture alive and silently shades last size's ids.
pub struct VisTargets {
    /// The texture behind [`color`](Self::color) - kept because
    /// `copy_texture_to_buffer` takes a texture and a view cannot be reversed
    /// into one.
    pub texture: wgpu::Texture,
    pub color: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    pub size: (u32, u32),
    pub generation: u64,
}

impl VisTargets {
    fn new(gpu: &GpuContext, size: (u32, u32), generation: u64) -> Self {
        let tex = |label, format, usage| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0.max(1),
                    height: size.1.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let color = tex(
            "visbuffer",
            VIS_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let color_view = color.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = tex(
            "visbuffer-depth",
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture: color,
            color: color_view,
            depth,
            size,
            generation,
        }
    }
}

/// The **visibility-buffer readback** (P28.1) — off by default, and the only way
/// an id ever leaves the GPU.
///
/// It is the `VgeomAuditResources` shape one subsystem over, for the same
/// reason: the buffer lives inside a graph node and a graph node is not
/// reachable from outside the graph, so the resource the node copies into has to
/// be the renderer's. Nothing on the shipping path turns it on, so nothing on the
/// shipping path pays for the copy.
///
/// Two arms need it and neither could exist without it: the determinism pin
/// (two runs, byte-identical buffers, on a real device) and the parity oracle's
/// **interior test** — a pixel whose four neighbours carry its own id is fully
/// covered by one triangle, which is exactly the pixel where a 4x-MSAA forward
/// frame and a 1x resolve are obliged to agree to the byte.
pub struct VisReadback {
    pub(crate) enabled: bool,
    buffer: wgpu::Buffer,
    /// `(width, height, padded bytes per row)` of what the last copy holds.
    shape: (u32, u32, u32),
    capacity: u64,
}

/// `copy_texture_to_buffer`'s row alignment.
const COPY_ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

impl VisReadback {
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        Self {
            enabled: false,
            buffer: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("visbuffer-readback"),
                size: COPY_ALIGN as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            shape: (0, 0, 0),
            capacity: COPY_ALIGN as u64,
        }
    }

    pub(crate) fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }

    /// Record the copy, growing the staging buffer if the viewport did.
    pub(crate) fn record(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        targets: &VisTargets,
    ) {
        if !self.enabled {
            return;
        }
        let (w, h) = targets.size;
        let row = (w * VIS_TEXEL_BYTES).next_multiple_of(COPY_ALIGN);
        let bytes = row as u64 * h as u64;
        if bytes > self.capacity {
            self.buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("visbuffer-readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            self.capacity = bytes;
        }
        self.shape = (w, h, row);
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &targets.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Map the ids the **last submitted** frame wrote, row padding removed.
    /// Blocking - a test/tools path, never the hot path.
    pub(crate) fn read(&self, gpu: &GpuContext) -> Option<VisImage> {
        let (w, h, row) = self.shape;
        if !self.enabled || w == 0 || h == 0 {
            return None;
        }
        let slice = self.buffer.slice(..row as u64 * h as u64);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        let view = slice.get_mapped_range().expect("mapped above");
        let mut ids = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            let start = (y * row) as usize;
            let bytes = &view[start..start + (w * VIS_TEXEL_BYTES) as usize];
            // Decoded as PAIRS of `u32` and folded by hand rather than cast
            // straight to `u64`: `bytemuck::cast_slice` requires the slice to be
            // 8-aligned and a mapped sub-range's alignment is the driver's to
            // decide. The fold is the same arithmetic `VisPacking::from_words`
            // states, and it is called rather than restated.
            ids.extend(
                bytemuck::cast_slice::<u8, u32>(bytes)
                    .chunks_exact(2)
                    .map(|w| crate::visbuffer::VisPacking::from_words([w[0], w[1]])),
            );
        }
        drop(view);
        self.buffer.unmap();
        Some(VisImage {
            width: w,
            height: h,
            ids,
        })
    }
}

/// One frame's visibility ids, unpadded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisImage {
    pub width: u32,
    pub height: u32,
    /// Row-major, `width * height` entries. [`crate::visbuffer::VIS_EMPTY`] where
    /// no meshlet covered the pixel.
    pub ids: Vec<u64>,
}

impl VisImage {
    /// The id at `(x, y)`, or [`crate::visbuffer::VIS_EMPTY`] outside the image.
    pub fn at(&self, x: u32, y: u32) -> u64 {
        if x >= self.width || y >= self.height {
            return crate::visbuffer::VIS_EMPTY;
        }
        self.ids[(y * self.width + x) as usize]
    }

    /// Pixels covered by geometry.
    pub fn covered(&self) -> usize {
        self.ids
            .iter()
            .filter(|&&v| v != crate::visbuffer::VIS_EMPTY)
            .count()
    }

    /// **Interior pixels**: covered, and carrying the same id as all four
    /// four-neighbours.
    ///
    /// This is the parity oracle's whole load-bearing idea. A pixel whose
    /// neighbours all name the same triangle is a pixel the forward path's four
    /// MSAA samples are all inside that triangle for, so the forward frame
    /// shades it ONCE at the pixel centre and resolves four identical samples -
    /// which is precisely what the single-sample resolve computes. Anywhere else
    /// the two are *obliged* to differ, and comparing them there would be
    /// measuring MSAA rather than the resolve. The border is excluded because a
    /// neighbour off the image is not evidence either way.
    pub fn interior(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for y in 1..self.height.saturating_sub(1) {
            for x in 1..self.width.saturating_sub(1) {
                let id = self.at(x, y);
                if id == crate::visbuffer::VIS_EMPTY {
                    continue;
                }
                if self.at(x - 1, y) == id
                    && self.at(x + 1, y) == id
                    && self.at(x, y - 1) == id
                    && self.at(x, y + 1) == id
                {
                    out.push((x, y));
                }
            }
        }
        out
    }
}

/// Everything the three passes need that outlives a frame.
pub struct VisState {
    /// `visbuffer.wgsl`'s raster pipeline + its `@group(1)` layout.
    pub raster: wgpu::RenderPipeline,
    pub raster_bgl: wgpu::BindGroupLayout,
    /// `vis_resolve.wgsl`'s fullscreen pipeline + its `@group(3)` layout.
    pub resolve: wgpu::RenderPipeline,
    pub resolve_bgl: wgpu::BindGroupLayout,
    /// `vis_feedback.wgsl`'s compute pipeline + its `@group(0)` layout.
    pub feedback: wgpu::ComputePipeline,
    pub feedback_bgl: wgpu::BindGroupLayout,
    pub feedback_params: wgpu::Buffer,
    /// Per-texture-handle first mask bit, rebuilt whenever the layout moves.
    pub feedback_bases: wgpu::Buffer,
    feedback_bases_cap: u32,
    /// The frame's flat instance table — every asset's packed instances
    /// concatenated in the deterministic asset order the raster walks, so
    /// `base + local` is the global index the packing stores.
    ///
    /// An `Rgba32Float` texture, one row an instance: see
    /// [`VIS_FRAGMENT_STORAGE_BINDINGS`] for why it is not a buffer.
    pub instances: wgpu::Texture,
    pub instances_view: wgpu::TextureView,
    instances_cap: u32,
    /// Per-asset `VisFlagsGpu` (the instance base), one buffer per asset slot.
    pub flags: Vec<wgpu::Buffer>,
    pub targets: Option<VisTargets>,
    pub audit: VisAudit,
}

impl VisState {
    pub fn new(
        gpu: &GpuContext,
        view_bgl: &wgpu::BindGroupLayout,
        lights_bgl: &wgpu::BindGroupLayout,
        env_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let ro = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty,
            count: None,
        };
        let uint_tex = |multisampled| wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled,
        };
        // `filterable: false` because nothing samples it: the instance table is
        // read with `textureLoad`, which takes integer texels and no sampler at
        // all. Declaring it filterable would ask every adapter for
        // `Rgba32Float` linear filtering, which is an OPTIONAL feature
        // (`FLOAT32_FILTERABLE`) — a capability requirement bought for a
        // convenience nothing uses is the class the P25 audit named.
        let float_tex = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };

        // ── the raster ──────────────────────────────────────────────────────
        let vs = wgpu::ShaderStages::VERTEX;
        let raster_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("visbuffer-raster"),
                entries: &[
                    entry(0, vs, ro),
                    entry(1, vs, ro),
                    entry(2, vs, ro),
                    entry(3, vs, ro),
                    entry(4, vs, float_tex),
                    entry(5, vs, ro),
                    // The instance base rides the vertex stage here, where the
                    // forward path's twin is a FRAGMENT uniform — this pass has
                    // no fragment work but the id it writes is composed in `vs`.
                    entry(6, vs, uniform),
                    entry(7, vs, ro),
                ],
            });
        let raster_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("visbuffer"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("visbuffer").into()),
            });
        let raster_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visbuffer-raster"),
                bind_group_layouts: &[Some(view_bgl), Some(&raster_bgl)],
                immediate_size: 0,
            });
        let raster = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("visbuffer-raster"),
                layout: Some(&raster_layout),
                vertex: wgpu::VertexState {
                    module: &raster_shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &raster_shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: VIS_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(DEPTH_COMPARE),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // ── the resolve ─────────────────────────────────────────────────────
        let resolve_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("visbuffer-resolve"),
                entries: &resolve_bgl_entries(),
            });
        let resolve_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vis-resolve"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("vis_resolve").into()),
            });
        let resolve_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visbuffer-resolve"),
                bind_group_layouts: &[
                    Some(view_bgl),
                    Some(lights_bgl),
                    Some(env_bgl),
                    Some(&resolve_bgl),
                ],
                immediate_size: 0,
            });
        let resolve = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("visbuffer-resolve"),
                layout: Some(&resolve_layout),
                vertex: wgpu::VertexState {
                    module: &resolve_shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &resolve_shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(DEPTH_COMPARE),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: SCENE_SAMPLES,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            });

        // ── the per-pixel feedback ──────────────────────────────────────────
        let cs = wgpu::ShaderStages::COMPUTE;
        let feedback_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("visbuffer-feedback"),
                entries: &[
                    entry(0, cs, uniform),
                    entry(1, cs, uint_tex(false)),
                    entry(2, cs, ro),
                    entry(3, cs, ro),
                    entry(4, cs, ro),
                    entry(5, cs, ro),
                    entry(6, cs, float_tex),
                    entry(7, cs, ro),
                    entry(8, cs, ro),
                    entry(
                        9,
                        cs,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    ),
                ],
            });
        let feedback_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vis-feedback"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/vis_feedback.wgsl").into(),
                ),
            });
        let feedback_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("visbuffer-feedback"),
                bind_group_layouts: &[Some(&feedback_bgl)],
                immediate_size: 0,
            });
        let feedback = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("visbuffer-feedback"),
                layout: Some(&feedback_layout),
                module: &feedback_shader,
                entry_point: Some("cs_feedback"),
                compilation_options: Default::default(),
                cache: None,
            });

        let feedback_params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("visbuffer-feedback-params"),
            size: std::mem::size_of::<VisFeedbackParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let feedback_bases = storage(gpu, "visbuffer-feedback-bases", 16);
        let instances = instance_texture(gpu, 1);
        let instances_view = instances.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            raster,
            raster_bgl,
            resolve,
            resolve_bgl,
            feedback,
            feedback_bgl,
            feedback_params,
            feedback_bases,
            feedback_bases_cap: 0,
            instances,
            instances_view,
            instances_cap: 0,
            flags: Vec::new(),
            targets: None,
            audit: VisAudit::default(),
        }
    }

    /// Resize the visibility targets to the frame's. Returns whether anything was
    /// recreated (⇒ every bind group naming one is stale).
    pub fn ensure_targets(&mut self, gpu: &GpuContext, frame: &FrameData) -> bool {
        let size = frame.targets.size;
        let gen = frame.targets.generation;
        let stale = self
            .targets
            .as_ref()
            .is_none_or(|t| t.size != size || t.generation != gen);
        if stale {
            self.targets = Some(VisTargets::new(gpu, size, gen));
        }
        stale
    }

    /// Grow the flat instance table to `count` records. Returns whether it was
    /// recreated.
    pub fn ensure_instances(&mut self, gpu: &GpuContext, count: u32) -> bool {
        if count <= self.instances_cap {
            return false;
        }
        let cap = count.next_power_of_two().max(4);
        self.instances = instance_texture(gpu, cap);
        self.instances_view = self
            .instances
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.instances_cap = cap;
        true
    }

    /// Write the frame's flat table. `records` is the packed
    /// `VgeomInstanceGpu` bytes; each is padded out to a
    /// [`VIS_INSTANCE_TEXELS`]-texel row.
    pub fn write_instances(&self, gpu: &GpuContext, records: &[u8], stride: usize) {
        if records.is_empty() || stride == 0 {
            return;
        }
        let row = (VIS_INSTANCE_TEXELS * 16) as usize;
        debug_assert!(
            stride <= row,
            "a {stride}-byte instance does not fit a {row}-byte row"
        );
        let n = records.len() / stride;
        let mut padded = vec![0u8; n * row];
        for i in 0..n {
            padded[i * row..i * row + stride]
                .copy_from_slice(&records[i * stride..(i + 1) * stride]);
        }
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.instances,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row as u32),
                rows_per_image: Some(n as u32),
            },
            wgpu::Extent3d {
                width: VIS_INSTANCE_TEXELS,
                height: n as u32,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Grow the per-texture bit-base table to `count` handles.
    pub fn ensure_bases(&mut self, gpu: &GpuContext, count: u32) -> bool {
        if count <= self.feedback_bases_cap {
            return false;
        }
        let cap = count.next_power_of_two().max(4);
        self.feedback_bases = storage(gpu, "visbuffer-feedback-bases", cap as u64 * 4);
        self.feedback_bases_cap = cap;
        true
    }

    /// Ensure `n` per-asset flag uniforms exist. One buffer per asset rather than
    /// one written `n` times: both draws are recorded into the same encoder
    /// before submit, so a single buffer would apply every write before any draw
    /// ran — the exact defect `AssetDraw`'s two `params` buffers exist for.
    pub fn ensure_flags(&mut self, gpu: &GpuContext, n: usize) {
        while self.flags.len() < n {
            self.flags
                .push(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("visbuffer-flags"),
                    size: std::mem::size_of::<VisFlagsGpu>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
    }

    /// The admission door, applied to a frame's measured shape.
    ///
    /// `meshlet_slots` is the shared pool's **capacity** in records — not its
    /// occupancy — because a slot that exists can be named by a page that pages
    /// in later in the same frame, and a ceiling checked against occupancy would
    /// pass the check and fail the frame.
    pub fn admit(
        &mut self,
        instances: u32,
        pool_bytes: u64,
        max_tri: u32,
    ) -> Result<(), VisPackError> {
        let slots = meshlet_slots_for_bytes(pool_bytes);
        match VisPacking::admit(instances, slots, max_tri) {
            Ok(()) => {
                self.audit.flat_instances = instances;
                self.audit.meshlet_slots = slots;
                Ok(())
            }
            Err(e) => {
                self.audit.refuse(e);
                Err(e)
            }
        }
    }
}

/// **The resolve's own `@group(3)` layout entries**, hoisted out of
/// [`VisState::new`] so a test can COUNT the storage bindings it spends
/// (P28.1 audit — see [`VIS_FRAGMENT_STORAGE_BINDINGS`] and
/// [`crate::passes::env_bgl_entries`]).
fn resolve_bgl_entries() -> [wgpu::BindGroupLayoutEntry; 6] {
    let fs = wgpu::ShaderStages::FRAGMENT;
    let ro = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: fs,
        ty,
        count: None,
    };
    [
        entry(
            0,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Uint,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        ),
        entry(1, ro),
        entry(2, ro),
        entry(3, ro),
        entry(4, ro),
        // The flat instance table. A TEXTURE, and the whole reason the sum
        // above is four and not five.
        entry(
            5,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        ),
    ]
}

/// **The pool ceiling's byte→slot conversion**, in ONE place because it is
/// measured in bytes at the call site and enforced in slots in the packing
/// (P28.1 audit).
///
/// Its first draft lived only inside [`VisState::admit`] and the arm about it
/// re-implemented the division — so replacing `MESHLET_REC_LEN` with a wrong
/// constant left `the_pool_ceiling_converts_bytes_to_slots_at_the_record_length`
/// green. Mutation-measured. A mirror is not a measurement (P24).
pub fn meshlet_slots_for_bytes(pool_bytes: u64) -> u32 {
    (pool_bytes / MESHLET_REC_LEN as u64).min(u32::MAX as u64) as u32
}

fn instance_texture(gpu: &GpuContext, rows: u32) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("visbuffer-instances"),
        size: wgpu::Extent3d {
            width: VIS_INSTANCE_TEXELS,
            height: rows.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn storage(gpu: &GpuContext, label: &str, bytes: u64) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).next_multiple_of(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visbuffer::{VIS_MAX_INSTANCES, VIS_MAX_MESHLET_SLOTS};

    /// **Every fragment storage binding the default limit grants is spent**, and
    /// the arm exists so the ninth is a failing test here rather than a
    /// `create_pipeline_layout` refusal on a user's machine — which is how this
    /// ceiling was found in the first place.
    ///
    /// **Its first draft could not do that** (P28.1 audit). It asserted
    /// `4 + 4 == 8` with both fours written as literals and a comment claiming
    /// they were "read off `passes::mod`'s own constants"; they were not, and
    /// turning the environment group's binding 6 from a uniform into a storage
    /// buffer left the arm green — mutation-measured. It now COUNTS, off the two
    /// entry lists the two layouts are actually built from, so the growth it is
    /// named for is the thing it sees.
    #[test]
    fn the_resolve_spends_every_fragment_storage_binding_the_default_limit_grants() {
        assert_eq!(
            wgpu::Limits::default().max_storage_buffers_per_shader_stage,
            VIS_FRAGMENT_STORAGE_BINDINGS,
            "the pinned wgpu's default storage-binding limit moved; the ruling in \
             VIS_FRAGMENT_STORAGE_BINDINGS is arithmetic about the old one"
        );
        let is_fragment_storage = |e: &wgpu::BindGroupLayoutEntry| {
            e.visibility.contains(wgpu::ShaderStages::FRAGMENT)
                && matches!(
                    e.ty,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { .. },
                        ..
                    }
                )
        };
        // The environment group's: GI SH probes (5), the VT indirection table
        // (16), the VSM page table (18) and its projections (19) — counted, not
        // recited, and counted through `VtPools`/`vsm_receiver`'s own entry
        // constructors, which is where two of the four actually come from.
        let env_storage = super::super::env_bgl_entries()
            .iter()
            .filter(|e| is_fragment_storage(e))
            .count() as u32;
        // The resolve's own: the four meshlet pools. The instance table is a
        // texture precisely so this sum is not five.
        let resolve_storage = resolve_bgl_entries()
            .iter()
            .filter(|e| is_fragment_storage(e))
            .count() as u32;
        assert_eq!(
            (env_storage, resolve_storage),
            (4, 4),
            "the split moved: the ruling in VIS_FRAGMENT_STORAGE_BINDINGS is \
             arithmetic about four environment bindings and four pools"
        );
        assert_eq!(
            env_storage + resolve_storage,
            VIS_FRAGMENT_STORAGE_BINDINGS,
            "the resolve no longer sits exactly at the limit — if this is a \
             saving, the instance table can go back to being a buffer; if it is \
             a growth, the pipeline will not build"
        );
    }

    #[test]
    fn a_flat_instance_record_fits_its_padded_row() {
        let stride = std::mem::size_of::<crate::passes::vgeom::VgeomInstanceGpu>();
        let row = (VIS_INSTANCE_TEXELS * 16) as usize;
        assert!(
            stride <= row,
            "a {stride}-byte instance record does not fit a {row}-byte texture row"
        );
        // …and the row is exactly the copy alignment, which is the whole reason
        // it is sixteen texels and not eleven.
        assert_eq!(row, COPY_ALIGN as usize);
    }

    #[test]
    fn the_audit_partitions_its_refusals_by_ceiling() {
        let mut a = VisAudit::default();
        a.refuse(VisPackError::Instances {
            count: 1,
            ceiling: 0,
        });
        a.refuse(VisPackError::MeshletSlots {
            count: 1,
            ceiling: 0,
        });
        a.refuse(VisPackError::MeshletSlots {
            count: 1,
            ceiling: 0,
        });
        a.refuse(VisPackError::TrianglesPerMeshlet {
            max_tri: 1,
            ceiling: 0,
        });
        assert_eq!(a.refused_instances, 1);
        assert_eq!(a.refused_meshlet_slots, 2);
        assert_eq!(a.refused_triangles, 1);
        assert_eq!(a.refused(), 4);
        // The tail is partitioned exactly once — the P27.5 counter ruling.
        assert_eq!(
            a.refused(),
            a.refused_instances + a.refused_meshlet_slots + a.refused_triangles
        );
    }

    /// The pool ceiling is expressed in BYTES at the call site and in SLOTS in
    /// the packing, and the conversion is the meshlet record length. A frame
    /// whose pool is one record past the field is refused; one record short of
    /// it is admitted.
    ///
    /// **Calls the door's own conversion** (P28.1 audit). The first draft
    /// re-implemented the division locally — "mirrors `VisState::admit`'s
    /// arithmetic without a GpuContext" — so replacing `MESHLET_REC_LEN` with a
    /// wrong constant inside `admit` left this green. Mutation-measured.
    #[test]
    fn the_pool_ceiling_converts_bytes_to_slots_at_the_record_length() {
        let rec = MESHLET_REC_LEN as u64;
        let mut s = VisAudit::default();
        let at = (VIS_MAX_MESHLET_SLOTS as u64) * rec;
        let over = at + rec;
        let slots = super::meshlet_slots_for_bytes;
        assert_eq!(slots(at), VIS_MAX_MESHLET_SLOTS);
        assert_eq!(slots(over), VIS_MAX_MESHLET_SLOTS + 1);
        assert_eq!(VisPacking::admit(1, slots(at), 124), Ok(()));
        let e = VisPacking::admit(1, slots(over), 124).unwrap_err();
        s.refuse(e);
        assert_eq!(s.refused_meshlet_slots, 1);
        // …and the byte figure the ceiling corresponds to, stated so a reader of
        // the ledger can check it. It was 16 384 slots x 64 B = 1 MiB, which
        // P28.1 measured at ~19 MiB of resident pool — 7.4 % of the 256 MiB
        // default budget, and the binding ceiling. IB-8 made it 33 554 432 slots
        // x 64 B = **2 GiB of meshlet records**, which is eight times the whole
        // budget: the meshlet refusal is no longer reachable by a configuration
        // anyone would ship, and the binding ceiling moved to `budget_bytes`,
        // which is tunable and reported where a packed field was neither.
        assert_eq!(at, 2 * 1024 * 1024 * 1024);
        // The field can no longer bind before the BUDGET does, and that is the
        // whole of IB-8's meshlet half. The descriptors the field addresses are
        // themselves larger than the entire default streaming budget — and
        // descriptors are only 5.26 – 5.44 % of a cooked asset's pool bytes
        // (P28.1's measurement), so the pool this ceiling corresponds to is
        // ~37 – 39 GiB against a 344 MiB budget.
        let budget = crate::settings::DEFAULT_STREAM_BUDGET_BYTES;
        println!(
            "IB-8 meshlet ceiling: {} slots = {at} B of descriptors, {:.1}x the \
             {budget} B default budget; at 5.44 % descriptor share that is a \
             {:.1} GiB pool",
            VIS_MAX_MESHLET_SLOTS,
            at as f64 / budget as f64,
            at as f64 / 0.0544 / (1024.0 * 1024.0 * 1024.0),
        );
        assert!(
            at > budget,
            "the meshlet field addresses {at} B of descriptors against a \
             {budget} B budget — the field is becoming the binding ceiling again \
             and IB-8's ledger needs re-reading"
        );
    }

    #[test]
    fn the_flagship_scenes_shape_is_admitted_and_the_next_order_of_magnitude_is_not() {
        // `vgeom-demo`: an 18 x 18 grid, one asset, 124 triangles a meshlet.
        assert_eq!(VisPacking::admit(324, 512, 124), Ok(()));
        assert!(VisPacking::admit(VIS_MAX_INSTANCES + 1, 512, 124).is_err());
    }

    /// **IB-8: a city enters the visibility path.**
    ///
    /// The certification's complaint is one number — `VIS_MAX_INSTANCES = 2047`,
    /// refused beyond, so *"a city of thousands of Nanite-class building meshes
    /// cannot all enter the visbuffer path in one frame"*. This arm is that
    /// sentence, made false, at the shape the island's own city fixture has and
    /// at the brief's target.
    // The ceiling IS a constant, and comparing it to the number the
    // certification names is the whole content of this arm — `clippy` is right
    // that the comparison const-folds and wrong that it is therefore pointless.
    // The alternative it suggests (a `const _: () = assert!(…)`) would make a
    // future re-cut a compile error instead of a failing test, which is worse:
    // the failure this arm exists to force is *revisit the ledger*, not *stop
    // building*.
    #[allow(clippy::assertions_on_constants)]
    #[test]
    fn the_instance_ceiling_admits_a_city() {
        // The thousand-building city fixture, and the brief's 16k target.
        for instances in [1_000u32, 4_096, 16_384, 65_536] {
            assert_eq!(
                VisPacking::admit(instances, 512, 124),
                Ok(()),
                "{instances} instances must enter the visibility path"
            );
        }
        // The old ceiling, which is what this item retired.
        const WAS: u32 = 2_047;
        assert!(
            VIS_MAX_INSTANCES > 8_000 * WAS,
            "the instance ceiling is {VIS_MAX_INSTANCES}, only {:.0}x the {WAS} \
             IB-8 names",
            f64::from(VIS_MAX_INSTANCES) / f64::from(WAS)
        );
        println!(
            "IB-8 instance ceiling: {WAS} -> {VIS_MAX_INSTANCES} ({:.0}x); the \
             brief asked for 16 384",
            f64::from(VIS_MAX_INSTANCES) / f64::from(WAS)
        );
        // …and it is still a CEILING, not an absence of one: the refusal is
        // typed, named and reachable by construction.
        let e = VisPacking::admit(VIS_MAX_INSTANCES + 1, 512, 124).unwrap_err();
        assert!(
            matches!(e, VisPackError::Instances { .. }),
            "the instance ceiling stopped being an instance refusal: {e:?}"
        );
        assert!(e.to_string().contains(&VIS_MAX_INSTANCES.to_string()));
    }
}
