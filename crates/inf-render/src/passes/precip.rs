//! Precipitation v1 (P17.4) — the pass that draws rain and snow.
//!
//! ## Where it sits, and why
//!
//! Immediately **after** [`super::cloud`] and **before** the translucent pass:
//!
//! * after the opaque passes, so the depth buffer is populated and the hardware
//!   can reject drops behind the world;
//! * after the clouds, so rain composites *over* the deck it is falling out of;
//! * before translucency, so glass composites over rain as it would over any
//!   other background;
//! * inside the MSAA scene target, so the resolve → TAA → bloom → tonemap chain
//!   treats a lit streak as ordinary scene radiance. A rain streak against a low
//!   sun blooms without a line of code for it.
//!
//! ## Depth — one mechanism, not two
//!
//! [`super::cloud`] needs both the hardware test *and* a manual depth read,
//! because a raymarch has to clamp `t_far` at geometry that lies **inside** the
//! marched slab. A precipitation particle is a flat quad at a single depth, so
//! there is no span to clamp and the hardware test is the entire answer — and it
//! resolves per MSAA sample, which the cloud pass's sample-0 load does not. The
//! depth attachment is still bound **read-only** (`depth_ops: None`), which is
//! both the P17.3 precedent and what preserves the buffer for the passes after
//! this one.
//!
//! ## No bind group of its own
//!
//! The whole parameter block rides in the atmosphere uniform (see
//! `AtmosphereGpu`'s precipitation fields), so this pass binds exactly what the
//! cloud pass already binds minus the two noise volumes and the scene depth:
//! the uniform, both LUTs and their sampler. Four entries, no new resource, no
//! new `EnvBinding` component, nothing resizable beyond what
//! [`super::ResourceKey`] already covers.
//!
//! ## Off ⇒ instruction-neutral
//!
//! Precipitation inactive ⇒ [`PrecipNode::run`] returns before touching the
//! encoder: no render pass, no pipeline bind, no draw. Every pre-P17.4 golden
//! therefore records the exact command stream it did before.

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};

/// Vertices per particle: two triangles over a quad.
const VERTS_PER_PARTICLE: u32 = 6;

pub struct PrecipNode {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    /// Keyed on the atmosphere generation alone — nothing bound here is
    /// viewport-size-dependent (the scene depth is an *attachment*, not a
    /// binding), so keying on the full [`super::ResourceKey`] would rebuild on
    /// every window resize and be a lie about what invalidates it. Same
    /// reasoning, and the same key type, as the cloud **bake** node.
    bind_group: super::GenCache<u64, wgpu::BindGroup>,
}

impl PrecipNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("precip"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("precip").into()),
            });

        // The atmosphere uniform is read in the VERTEX stage too (particle
        // placement is entirely vertex-side), which is the one way this bind
        // group differs from the cloud pass's fragment-only one.
        let both = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let frag = wgpu::ShaderStages::FRAGMENT;
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("precip"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: both,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: frag,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("precip"),
                bind_group_layouts: &[Some(view_bgl), Some(&bgl)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("precip"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    // No vertex buffers at all: a particle is a pure function of
                    // its instance index and the level's clock.
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        // PREMULTIPLIED, matching the cloud pass: the fragment
                        // already scales its radiance by its own coverage.
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    // A billboard's winding flips as it turns to face the eye,
                    // so culling would erase half the rain.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    // Test but never write: a drop is a translucent speck, and
                    // writing its depth would make the passes after this one
                    // think there is geometry hanging in the air.
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

impl RenderNode for PrecipNode {
    fn name(&self) -> &'static str {
        "precip"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let atmos_params = &frame.scene.atmosphere;
        if !atmos_params.precip_active() {
            return;
        }
        let quality = crate::precip::PrecipQuality::from_atmosphere(frame.atmosphere.quality);
        let count = atmos_params.precip.count(quality);
        if count == 0 {
            return;
        }

        let atmos = frame.atmosphere;
        let bgl = &self.bgl;
        let bind_group = self
            .bind_group
            .get_or_build(atmos.generation, || {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("precip"),
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
                    ],
                })
            })
            .clone();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("precip"),
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
                // READ-ONLY: the hardware test still runs, the contents survive
                // for the passes that follow, and nothing here needs to write.
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
        pass.draw(0..VERTS_PER_PARTICLE, 0..count);
    }
}
