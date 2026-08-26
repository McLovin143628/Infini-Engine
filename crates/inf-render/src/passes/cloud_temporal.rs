//! Cloud temporal resolve (wave SKY2): the cloud march's **own** history.
//!
//! Reads this frame's half-res march plus the previous frame's accumulation,
//! reprojects the history through the march's own mean cloud distance, clamps it
//! to the current 3×3 neighbourhood and blends. Writes the ping-pong half the
//! composite will read.
//!
//! ## Why the cloud does not ride [`super::taa`]
//!
//! That pass reprojects through the depth prepass. A cloud is a participating
//! medium and writes no depth, so for every cloud pixel `taa.wgsl` reads the
//! cleared reverse-Z 0, skips its reprojection branch and takes the history from
//! the *same* texel. Under a static camera that is exactly right and free; under
//! a turning one the whole sky smears. Copying its building blocks — the previous
//! view-projection, the world reconstruction, the neighbourhood clamp — and
//! giving them a depth the cloud actually has is the difference.
//!
//! ## Off path
//!
//! [`crate::RenderSettings::cloud_temporal`] is **off by default**, on exactly
//! the terms `taa` is: an accumulating buffer makes a frame a function of the
//! frames before it. Off ⇒ this node touches no encoder and the composite reads
//! the raw march instead.

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, CLOUD_FORMAT};

/// Fraction of the blend taken from the accumulated history.
///
/// The same 0.9 [`super::taa`] uses, and for the same arithmetic: with a
/// low-discrepancy jitter sequence a 0.9 blend has the pixel's error down by an
/// order of magnitude inside ten frames, which at any playable frame rate is
/// under a fifth of a second.
const BLEND: f32 = 0.9;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CloudTemporalParams {
    prev_view_proj: [f32; 16],
    /// x = blend, y = history valid (>0.5), zw = half-res size (px).
    cfg: [f32; 4],
}

pub struct CloudTemporalNode {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
}

impl CloudTemporalNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cloud-temporal"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("cloud_temporal").into()),
            });
        let frag = wgpu::ShaderStages::FRAGMENT;
        let tex = |binding: u32| wgpu::BindGroupLayoutEntry {
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
                label: Some("cloud-temporal"),
                entries: &[
                    tex(0),
                    tex(1),
                    tex(2),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: frag,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: frag,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cloud-temporal"),
                bind_group_layouts: &[Some(view_bgl), Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cloud-temporal"),
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
                        format: CLOUD_FORMAT,
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
        // Clamp-to-edge, and it is not a default: a reprojected history sample
        // that walked off the frame is rejected by the shader, so the addressing
        // mode only ever matters for the 3×3 clamp at the border, where wrapping
        // would fold the far side of the sky into the near one.
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cloud-temporal"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud-temporal-params"),
            size: std::mem::size_of::<CloudTemporalParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            bgl,
            sampler,
            params,
        }
    }
}

impl RenderNode for CloudTemporalNode {
    fn name(&self) -> &'static str {
        "cloud-temporal"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        if !frame.settings.cloud_temporal || !frame.scene.atmosphere.clouds_active() {
            return;
        }
        let (w, h) = frame.targets.cloud_size;
        gpu.queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&CloudTemporalParams {
                prev_view_proj: frame.taa_prev_view_proj,
                cfg: [
                    BLEND,
                    if frame.cloud_history_valid { 1.0 } else { 0.0 },
                    w as f32,
                    h as f32,
                ],
            }),
        );
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud-temporal"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&frame.targets.cloud),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(frame.cloud_history_prev),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&frame.targets.cloud_dist),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cloud-temporal"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame.cloud_history_cur,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
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
