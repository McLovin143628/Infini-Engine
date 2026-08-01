//! Sky node: first scene pass; clears color + depth for the frame, then fills
//! the background (depth writes off, everything else draws over).
//!
//! Since P17.2 it draws one of two things, selected by the scene's
//! [`AtmosphereParams::enabled`](crate::atmosphere::AtmosphereParams::enabled):
//!
//! * the schema-v11 three-colour gradient (every level without a `TimeOfDay`
//!   authority — the byte-stable path), or
//! * the physical sky: the sky-view LUT baked by [`super::sky_lut`], the sun and
//!   moon discs, and the procedural starfield.
//!
//! The pass binds the gradient uniform at `@group(1) @binding(0)` — exactly where
//! it always was — and the atmosphere block is *appended* at 1..4, so the
//! gradient path's shader arithmetic is unchanged.

use crate::camera::{DEPTH_CLEAR, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniforms {
    zenith: [f32; 4],
    horizon: [f32; 4],
    ground: [f32; 4],
}

pub struct SkyNode {
    pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    bgl: wgpu::BindGroupLayout,
    /// Cached against the atmosphere resources' generation — the LUT views are
    /// recreated when the quality tier changes. Nothing here is size-dependent,
    /// so the frame-target generation is deliberately NOT part of the key.
    bind_group: super::GenCache<u64, wgpu::BindGroup>,
    /// The sky params the uniform buffer currently holds; skip the re-upload while
    /// unchanged (the buffer is created empty, so the first frame always writes).
    uploaded: Option<crate::scene::SkyParams>,
}

impl SkyNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sky"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("sky").into()),
            });

        let uniforms = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky-uniforms"),
            size: std::mem::size_of::<SkyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let frag = wgpu::ShaderStages::FRAGMENT;
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let lut = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sky"),
                entries: &[
                    // 0 = the authored gradient (pre-P17.2 binding, unmoved).
                    uniform(0),
                    // 1..4 = the atmosphere block (P17.2).
                    uniform(1),
                    lut(2),
                    lut(3),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: frag,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sky"),
                bind_group_layouts: &[Some(view_bgl), Some(&bgl)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sky"),
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
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::Always),
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
            uniforms,
            bgl,
            bind_group: super::GenCache::default(),
            uploaded: None,
        }
    }
}

impl RenderNode for SkyNode {
    fn name(&self) -> &'static str {
        "sky"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let sky = &frame.scene.sky;
        // Re-upload only when the sky params changed (byte-identical bytes otherwise).
        if self.uploaded != Some(*sky) {
            let pad = |c: [f32; 3]| [c[0], c[1], c[2], 1.0];
            gpu.queue.write_buffer(
                &self.uniforms,
                0,
                bytemuck::bytes_of(&SkyUniforms {
                    zenith: pad(sky.zenith),
                    horizon: pad(sky.horizon),
                    ground: pad(sky.ground),
                }),
            );
            self.uploaded = Some(*sky);
        }

        // The atmosphere LUT views are recreated when the quality tier changes,
        // so the bind group is keyed on their generation (the same discipline
        // `EnvBinding` follows).
        let atmos = frame.atmosphere;
        let (bgl, uniforms) = (&self.bgl, &self.uniforms);
        let bind_group = self
            .bind_group
            .get_or_build(atmos.generation, || {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("sky"),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniforms.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: atmos.uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&atmos.transmittance),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&atmos.sky_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Sampler(&atmos.sampler),
                        },
                    ],
                })
            })
            .clone();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sky"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.color_msaa,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &frame.targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
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
