//! **The visibility-buffer path's GPU resources** (P28.1): the `R32Uint` target
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
//!    issues, into `R32Uint` + its own single-sample depth. Depth-tested
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
//! clause 1 says *single-sample* `R32Uint`. Those are not reconcilable by
//! configuration: a fullscreen fragment shading from a 1x id buffer writes one
//! colour to all four samples of its pixel, so a meshlet silhouette is a hard
//! edge where the forward path resolves four. That is the whole content of the
//! P28.1 MSAA ruling, it is measured by
//! `the_visbuffer_edge_cost_against_the_forward_path`, and it is why this mode
//! is a setting rather than the High-tier default. See
//! `docs/memos/p28-1-visbuffer.md` §5.

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::visbuffer::{VisPackError, VisPacking};

use inf_vgeom::asset::MESHLET_REC_LEN;

/// The visibility buffer's own format. `R32Uint` because the packing is exactly
/// thirty-two bits and an integer target neither filters nor blends — a
/// visibility id that a driver was free to average would be a different
/// triangle.
pub const VIS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

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

/// The viewport-sized targets. Recreated with the frame targets, and keyed on
/// their generation for the same reason every resizable resource in this
/// renderer is (`GenCache`'s invariant): a bind group cached across a resize
/// keeps the old texture alive and silently shades last size's ids.
pub struct VisTargets {
    pub color: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    pub size: (u32, u32),
    pub generation: u64,
}

impl VisTargets {
    fn new(gpu: &GpuContext, size: (u32, u32), generation: u64) -> Self {
        let tex = |label, format, usage| {
            gpu.device
                .create_texture(&wgpu::TextureDescriptor {
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
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        Self {
            color: tex(
                "visbuffer",
                VIS_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            ),
            depth: tex(
                "visbuffer-depth",
                DEPTH_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            ),
            size,
            generation,
        }
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
    pub instances: wgpu::Buffer,
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
                    entry(4, vs, ro),
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
        let fs = wgpu::ShaderStages::FRAGMENT;
        let resolve_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("visbuffer-resolve"),
                entries: &[
                    entry(0, fs, uint_tex(false)),
                    entry(1, fs, ro),
                    entry(2, fs, ro),
                    entry(3, fs, ro),
                    entry(4, fs, ro),
                    entry(5, fs, ro),
                ],
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
                    entry(6, cs, ro),
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
        let instances = storage(gpu, "visbuffer-instances", 256);

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
    pub fn ensure_instances(&mut self, gpu: &GpuContext, count: u32, stride: u64) -> bool {
        if count <= self.instances_cap {
            return false;
        }
        let cap = count.next_power_of_two().max(4);
        self.instances = storage(gpu, "visbuffer-instances", cap as u64 * stride);
        self.instances_cap = cap;
        true
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
        let slots = (pool_bytes / MESHLET_REC_LEN as u64).min(u32::MAX as u64) as u32;
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
    #[test]
    fn the_pool_ceiling_converts_bytes_to_slots_at_the_record_length() {
        let rec = MESHLET_REC_LEN as u64;
        let mut s = VisAudit::default();
        let at = (VIS_MAX_MESHLET_SLOTS as u64) * rec;
        let over = at + rec;
        // Mirrors `VisState::admit`'s arithmetic without a GpuContext.
        let slots = |bytes: u64| (bytes / rec).min(u32::MAX as u64) as u32;
        assert_eq!(slots(at), VIS_MAX_MESHLET_SLOTS);
        assert_eq!(slots(over), VIS_MAX_MESHLET_SLOTS + 1);
        assert_eq!(VisPacking::admit(1, slots(at), 124), Ok(()));
        let e = VisPacking::admit(1, slots(over), 124).unwrap_err();
        s.refuse(e);
        assert_eq!(s.refused_meshlet_slots, 1);
        // …and the byte figure the ceiling corresponds to, stated so a reader of
        // the ledger can check it: 16 384 slots x 64 B = 1 MiB of meshlet
        // records.
        assert_eq!(at, 1024 * 1024);
    }

    #[test]
    fn the_flagship_scenes_shape_is_admitted_and_the_next_order_of_magnitude_is_not() {
        // `vgeom-demo`: an 18 x 18 grid, one asset, 124 triangles a meshlet.
        assert_eq!(VisPacking::admit(324, 512, 124), Ok(()));
        assert!(VisPacking::admit(VIS_MAX_INSTANCES + 1, 512, 124).is_err());
    }
}
