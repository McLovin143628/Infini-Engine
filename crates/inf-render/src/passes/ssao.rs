//! The ambient-occlusion node: reconstructs half-res AO from the single-sample
//! depth prepass, blurs it, and leaves the result in the `ao` target that the lit
//! passes multiply into their ambient term. When AO is **disabled** the node
//! simply clears `ao` to white (1.0), so the lit passes' AO bind is pixel-neutral
//! and every non-AO golden is stable.
//!
//! Two fullscreen passes: `fs_ssao` (the raw integral from depth) then `fs_blur`.
//!
//! **Wave VIS1a replaced the estimator.** P13.3a projected a rotated hemisphere
//! kernel through the depth buffer and counted occluded samples; this is GTAO —
//! a horizon search per screen-space slice and a closed-form cosine-weighted
//! visibility integral over the wedge between the two horizons. The blur went
//! with it, from a 4x4 box to a depth-aware bilateral one. See `ssao.wgsl` for
//! the argument. The node's *shape* is unchanged: same two passes, same targets,
//! same half resolution, same ambient-only consumer.

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, AO_FORMAT};

/// Screen-space slices per pixel, and horizon-search steps per slice direction.
///
/// **4 x 6 rather than the 24 kernel taps it replaces, and the equality is a
/// coincidence worth not reading anything into**: a kernel tap was one
/// projection and one depth fetch at a scattered coordinate, a horizon step is
/// one depth fetch along a straight line of texels. The budgets are fixed
/// constants for the same reason `SHADOW_CASCADES` is: they are a quality/cost
/// point the renderer owns, not something a level author should be choosing.
/// `MAX_SLICES` / `MAX_STEPS` in the shader bound the loops.
const SLICES: u32 = 4;
const STEPS: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SsaoParams {
    /// x = slice count, y = radius, z = intensity, w = bias.
    cfg: [f32; 4],
    /// xy = depth size (px), zw = ao size (px).
    dims: [f32; 4],
    /// x = steps per slice direction, yzw reserved.
    cfg2: [f32; 4],
}

pub struct SsaoNode {
    ssao_pipeline: wgpu::RenderPipeline,
    blur_pipeline: wgpu::RenderPipeline,
    ssao_bgl: wgpu::BindGroupLayout,
    blur_bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    sampler: wgpu::Sampler,
    ssao_bg: Option<(u64, wgpu::BindGroup)>,
    blur_bg: Option<(u64, wgpu::BindGroup)>,
}

impl SsaoNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ssao"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("ssao").into()),
            });

        // group(1) for fs_ssao: depth texture + params.
        let ssao_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ssao"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        // group(1) for fs_blur: params (binding 1) + ao_src texture (2) + sampler (3).
        let blur_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ssao-blur"),
                entries: &[
                    // Wave VIS1a: the blur is bilateral now, so it reads the same
                    // depth the integral did.
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let make_pipeline = |label, bgl: &wgpu::BindGroupLayout, entry: &str| {
            let layout = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(view_bgl), Some(bgl)],
                    immediate_size: 0,
                });
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: AO_FORMAT,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let ssao_pipeline = make_pipeline("ssao", &ssao_bgl, "fs_ssao");
        let blur_pipeline = make_pipeline("ssao-blur", &blur_bgl, "fs_blur");

        let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssao-params"),
            size: std::mem::size_of::<SsaoParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ssao"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            ssao_pipeline,
            blur_pipeline,
            ssao_bgl,
            blur_bgl,
            params_buf,
            sampler,
            ssao_bg: None,
            blur_bg: None,
        }
    }
}

impl RenderNode for SsaoNode {
    fn name(&self) -> &'static str {
        "ssao"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        if !frame.settings.ssao.enabled {
            // AO disabled → clear the AO target to white (ambient unchanged).
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao-clear-white"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame.targets.ao,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            return;
        }

        let s = &frame.settings.ssao;
        gpu.queue.write_buffer(
            &self.params_buf,
            0,
            bytemuck::bytes_of(&SsaoParams {
                cfg: [SLICES as f32, s.radius, s.intensity, s.bias],
                dims: [
                    frame.targets.size.0 as f32,
                    frame.targets.size.1 as f32,
                    frame.targets.ao_size.0 as f32,
                    frame.targets.ao_size.1 as f32,
                ],
                cfg2: [STEPS as f32, 0.0, 0.0, 0.0],
            }),
        );

        if self
            .ssao_bg
            .as_ref()
            .is_none_or(|(g, _)| *g != frame.targets.generation)
        {
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssao"),
                layout: &self.ssao_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&frame.targets.depth_prepass),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.params_buf.as_entire_binding(),
                    },
                ],
            });
            self.ssao_bg = Some((frame.targets.generation, bg));
        }
        if self
            .blur_bg
            .as_ref()
            .is_none_or(|(g, _)| *g != frame.targets.generation)
        {
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ssao-blur"),
                layout: &self.blur_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&frame.targets.depth_prepass),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&frame.targets.ao_raw),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.blur_bg = Some((frame.targets.generation, bg));
        }

        // Raw AO → ao_raw.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame.targets.ao_raw,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.ssao_pipeline);
            pass.set_bind_group(0, frame.view_bg, &[]);
            pass.set_bind_group(1, &self.ssao_bg.as_ref().unwrap().1, &[]);
            pass.draw(0..3, 0..1);
        }
        // Blur ao_raw → ao.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ssao-blur"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame.targets.ao,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, frame.view_bg, &[]);
            pass.set_bind_group(1, &self.blur_bg.as_ref().unwrap().1, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}
