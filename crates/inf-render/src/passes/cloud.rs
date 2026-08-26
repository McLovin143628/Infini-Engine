//! Volumetric cloud raymarch (P17.3) — the pass that actually draws the sky's
//! clouds, at **half resolution** since wave SKY2.
//!
//! ## The three passes, and why there are three
//!
//! ```text
//! cloud            half-res march  -> (cloud, cloud_dist)
//! cloud-temporal   half-res blend  -> cloud_history[cur]      (optional)
//! cloud-composite  full-res upsample + blend into color_msaa
//! ```
//!
//! P17.3 did all of it in one pass straight into the MSAA scene target. Two
//! measurements moved it:
//!
//! * **cost.** At 1920x1080 with a ground camera pitched into open sky the
//!   full-res march reads **5.5 ms at High** on an RTX 4070 Ti. The 0.29 ms in
//!   the P17 cost table is not wrong, it is a different question — it measured a
//!   frame whose sky is almost entirely behind a cube field, where the hardware
//!   depth test rejects the march before it starts. Nothing else available is
//!   worth a factor of four.
//! * **temporal.** The march's jitter needs somewhere to converge, and it cannot
//!   be the scene's TAA: that pass reprojects through the depth prepass, and a
//!   cloud writes no depth, so every cloud pixel takes its "no depth" branch and
//!   reprojects to itself. A history of its own is not a luxury here.
//!
//! ## Why the composite still sits *here* in the graph
//!
//! Unchanged from P17.3, and still load-bearing. [`super::sky`] is the **first**
//! scene pass: it *clears* colour and depth, so a cloud drawn there would hang in
//! front of the terrain it belongs behind. The composite therefore runs after all
//! opaque geometry (mesh, vgeom, skinned, terrain) and before the translucent
//! pass:
//!
//! * **after opaque** so the depth buffer is populated and the hardware can
//!   reject cloud fragments behind the world;
//! * **before translucent** so glass and water composite over clouds, as they
//!   would over any other distant background;
//! * **into the MSAA scene target**, so the resolve → TAA → bloom → tonemap chain
//!   treats cloud radiance as ordinary scene radiance rather than as a post-hoc
//!   overlay. A cloud edge against the sun should bloom; it does, without a line
//!   of code here.
//!
//! ## Depth — two mechanisms, now in two passes
//!
//! **1. The march clamp**, here. A 2 km summit under a 1.5-4 km deck is *inside*
//! the slab, so it sits beyond the slab's entry plane and no hardware test can
//! reject the cloud behind it — it would be composited over the summit as a veil,
//! which ordinary content on an 8 km terrain hits immediately. So the march reads
//! the scene depth for its pixel and clamps `t_far` at the nearest geometry.
//!
//! **2. The hardware test**, in [`super::cloud_composite`]. `frag_depth` at the
//! ray's entry into the slab, `Greater` (reverse-Z), writes off — which rejects,
//! per MSAA sample and so with antialiased silhouettes, every fragment whose
//! geometry is entirely in front of the layer.
//!
//! Splitting them changed neither, and it retired the read-only-depth aliasing
//! the single pass needed (binding the depth attachment and the depth texture at
//! once): the march is not a depth attachment any more, and the composite does
//! not sample depth for the test.
//!
//! **What is still approximate.** The clamp reads MSAA sample 0 only, and now at
//! half resolution — one tap standing for a 2x2 block. That is what the
//! composite's *bilateral* upsample is for, and it is why that upsample is not
//! the 4x4 box blur SSAO uses: a cloud tap whose march stopped at a mountain and
//! the tap beside it looking past the summit must not be averaged. The pass still
//! runs before the translucent one, so cloud behind glass composites in the right
//! order but cloud *inside* a translucent volume does not (no ordering fixes that
//! without a per-fragment sort).
//!
//! ## Off path
//!
//! Clouds inactive ⇒ all three nodes return before touching the encoder: no
//! render pass, no pipeline bind, no draw. Every pre-P17.3 golden therefore
//! renders the exact command stream it did before.

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, CLOUD_DIST_FORMAT, CLOUD_FORMAT};

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
                    // TWO targets, both half-res and both written rather than
                    // blended: the march owns these textures outright. The
                    // premultiplied blend that used to be here moved to the
                    // composite, which is the pass that meets the scene.
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: CLOUD_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: CLOUD_DIST_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                }),
                primitive: Default::default(),
                // No depth attachment at all. The hardware test moved to the
                // composite with the `frag_depth` it needs; keeping a test here
                // would test a half-res fragment against a full-res buffer.
                depth_stencil: None,
                multisample: Default::default(),
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
            // Two half-res targets, CLEARED rather than loaded: the march writes
            // every pixel of both, including the early-out paths, so a load would
            // be a read of data nothing can see.
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &frame.targets.cloud,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &frame.targets.cloud_dist,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: None,
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
