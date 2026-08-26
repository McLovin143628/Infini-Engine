//! Tonemap node (P13.3a): the single post step that converts the linear HDR
//! scene (`post_hdr` = TAA output or the resolved `scene_hdr`) plus additive
//! bloom into the display-referred `scene_color` the composite blits. Exposure →
//! `+ bloom·intensity` → Narkowicz ACES → optional ordered dither. **Wave VIS1b
//! moved the exposure ahead of the bloom add and out of this pass's own uniform**
//! — it comes from [`crate::exposure::ExposureResources::state`], which the bloom
//! prefilter reads too, so the threshold is exposure-relative. Writes the
//! `Rgba8UnormSrgb` LDR target, so the hardware applies the sRGB OETF — matching
//! the old in-shader tonemap at defaults.

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, LDR_FORMAT};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TonemapParams {
    /// x = UNUSED since wave VIS1b (see the shader), y = bloom intensity,
    /// z = dither (>0.5), w = lens flare on (>0.5).
    knobs: [f32; 4],
    /// xy = resolution (px), zw unused.
    resolution: [f32; 4],
}

pub struct TonemapNode {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params_buf: wgpu::Buffer,
}

impl TonemapNode {
    pub fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("tonemap"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tonemap.wgsl").into()),
            });
        let float_tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
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
                label: Some("tonemap"),
                entries: &[
                    float_tex(0),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    float_tex(2),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // The frame's exposure (wave VIS1b) — the same sixteen bytes
                    // the bloom prefilter thresholds against, so the threshold
                    // and the multiply cannot describe different frames.
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // The half-res sun glare (wave VIS1b). Always bound — a bind
                    // group must be complete — and only sampled when the flare
                    // is on.
                    float_tex(5),
                ],
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tonemap"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("tonemap"),
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
                        format: LDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tonemap"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tonemap-params"),
            size: std::mem::size_of::<TonemapParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bgl,
            sampler,
            params_buf,
        }
    }
}

impl RenderNode for TonemapNode {
    fn name(&self) -> &'static str {
        "tonemap"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let bloom_intensity = if frame.settings.bloom.enabled {
            frame.settings.bloom.intensity
        } else {
            0.0
        };
        gpu.queue.write_buffer(
            &self.params_buf,
            0,
            bytemuck::bytes_of(&TonemapParams {
                knobs: [
                    0.0,
                    bloom_intensity,
                    if frame.settings.dither { 1.0 } else { 0.0 },
                    if frame.settings.flare.enabled {
                        1.0
                    } else {
                        0.0
                    },
                ],
                resolution: [
                    frame.targets.size.0 as f32,
                    frame.targets.size.1 as f32,
                    0.0,
                    0.0,
                ],
            }),
        );
        // post_hdr varies with TAA (ping-pong) → rebuild the bind group per frame.
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(frame.post_hdr),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&frame.targets.bloom[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: frame.exposure.state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&frame.targets.flare),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tonemap"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.scene_color,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
