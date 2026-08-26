//! Volumetric cloud raymarch (P17.3) — the pass that actually draws the sky's
//! clouds.
//!
//! ## Why it is a pass of its own, and why it sits *here*
//!
//! The obvious home is the sky pass, and it is the wrong one. [`super::sky`] is
//! the **first** scene pass: it *clears* colour and depth, so at the moment it
//! runs there is no geometry in the depth buffer to occlude anything, and a cloud
//! drawn there would hang in front of the terrain it is supposed to be behind.
//!
//! So the clouds are a dedicated pass placed after all opaque geometry (mesh,
//! vgeom, skinned, terrain) and before the translucent pass:
//!
//! * **after opaque** so the depth buffer is populated and the hardware can
//!   reject cloud fragments behind the world;
//! * **before translucent** so glass and water composite over clouds, as they
//!   would over any other distant background;
//! * **inside the MSAA scene target**, so the resolve → TAA → bloom → tonemap
//!   chain treats cloud radiance as ordinary scene radiance rather than as a
//!   post-hoc overlay. A cloud edge against the sun should bloom; it does,
//!   without a line of code here.
//!
//! ## Depth — two mechanisms, because one is not enough
//!
//! **1. The hardware test.** The fragment writes `@builtin(frag_depth)` at the
//! ray's **entry** into the cloud slab, with depth **writes off** and `Greater`
//! comparison (reverse-Z). That rejects, per MSAA sample and therefore with
//! antialiased silhouettes, every fragment whose geometry is entirely in front of
//! the slab — the common case, and the whole reason this pass is not in the sky
//! pass.
//!
//! **2. The march clamp.** The hardware test alone is *not* the occlusion, and
//! believing it was would be a real bug: a 2 km summit under a 1.5–4 km deck is
//! **inside** the slab, so it sits beyond the slab's entry plane, passes a
//! `Greater` test against the entry depth, and would have the entire marched span
//! — including the cloud physically *behind* the mountain — composited over it as
//! a veil. Ordinary content on an 8 km terrain hits this immediately. So the
//! shader also reads the scene depth for its pixel and clamps `t_far` to the
//! distance of the nearest geometry, which stops the march at the surface.
//!
//! To do both at once the depth attachment is bound **read-only**
//! (`depth_ops: None`) and the same view is additionally bound as a
//! `texture_depth_multisampled_2d` — the one arrangement WebGPU permits for
//! reading an attachment you are also testing against.
//!
//! **What is still approximate**, now that those two are in place: the clamp
//! samples MSAA sample 0 only, so on a pixel *partially* covered by intersecting
//! geometry the march length comes from one sample while the hardware test still
//! resolves coverage per sample — a sub-pixel discrepancy at a silhouette. And
//! the pass runs before the translucent one, so cloud behind glass composites in
//! the right order but cloud *inside* a translucent volume does not (there is no
//! ordering that would fix that without a per-fragment sort).
//!
//! ## Off path
//!
//! Clouds inactive ⇒ `run` returns before touching the encoder: no render pass,
//! no pipeline bind, no draw. Every pre-P17.3 golden therefore renders the exact
//! command stream it did before.

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};

pub struct CloudNode {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    /// Keyed on the full [`super::ResourceKey`] — **both** generations.
    ///
    /// The atmosphere generation covers the LUTs and the three cloud textures
    /// (recreated together when the quality tier changes). The *targets*
    /// generation is here because P17.3 added the scene depth to this bind group,
    /// and that view is recreated on every viewport resize — this is the one
    /// cloud bind group that is size-dependent, and leaving it out would sample a
    /// stale depth buffer at the old resolution after a resize.
    bind_group: super::GenCache<super::ResourceKey, wgpu::BindGroup>,
}

impl CloudNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cloud"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("cloud").into()),
            });

        let frag = wgpu::ShaderStages::FRAGMENT;
        let tex = |binding: u32, dim: wgpu::TextureViewDimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: dim,
                multisampled: false,
            },
            count: None,
        };
        let smp = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cloud"),
                entries: &[
                    // 0 = the shared atmosphere uniform (carries the cloud block).
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: frag,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 1..3 = the sky LUTs (sun transmittance at cloud altitude,
                    // ambient sky, aerial perspective on distant clouds).
                    tex(1, wgpu::TextureViewDimension::D2),
                    tex(2, wgpu::TextureViewDimension::D2),
                    smp(3),
                    // 4..6 = the cloud field.
                    tex(4, wgpu::TextureViewDimension::D3),
                    tex(5, wgpu::TextureViewDimension::D3),
                    smp(6),
                    // 7 = the scene depth, read back to clamp the march at
                    // geometry. Multisampled and unfilterable: it is `textureLoad`ed
                    // at an integer texel, never sampled.
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    },
                    // 8 = the blue-noise tile the first sample is offset by
                    // (SKY2). Unfilterable and `textureLoad`ed at an integer
                    // texel, for the reason on `CLOUD_BLUE_NOISE_FORMAT`.
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cloud"),
                bind_group_layouts: &[Some(view_bgl), Some(&bgl)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cloud"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        // PREMULTIPLIED alpha: the march accumulates radiance
                        // already weighted by the transmittance it was seen
                        // through, so the source is `(L·α, α)` and must not be
                        // multiplied by α a second time.
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    // Test but never write: a cloud is a participating medium, not
                    // a surface, and writing its entry depth would make later
                    // passes think there is geometry there.
                    depth_write_enabled: Some(false),
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

        Self {
            pipeline,
            bgl,
            bind_group: super::GenCache::default(),
        }
    }
}

impl RenderNode for CloudNode {
    fn name(&self) -> &'static str {
        "cloud"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        if !frame.scene.atmosphere.clouds_active() {
            return;
        }
        let atmos = frame.atmosphere;
        let bgl = &self.bgl;
        let bind_group = self
            .bind_group
            .get_or_build(super::resource_key(frame), || {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("cloud"),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: atmos.uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&atmos.transmittance),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&atmos.sky_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&atmos.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&atmos.cloud_shape),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&atmos.cloud_detail),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::Sampler(&atmos.cloud_sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(&frame.targets.depth),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(&atmos.cloud_blue_noise),
                        },
                    ],
                })
            })
            .clone();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cloud"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.color_msaa,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &frame.targets.depth,
                // `None` = READ-ONLY depth: contents are preserved for the passes
                // that follow, the hardware test still runs, and — the reason it
                // is here — the same view may simultaneously be bound as a sampled
                // texture so the shader can clamp its march at geometry. A
                // read/write attachment aliased into a binding is a validation
                // error; a read-only one is exactly what WebGPU allows.
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
