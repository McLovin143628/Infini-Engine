//! Bloom node (P13.3a): soft-knee prefilter → downsample mip chain → additive
//! tent upsample, leaving the result in `bloom[0]` for the tonemap to add. When
//! bloom is **disabled** the node clears `bloom[0]` to black (the tonemap adds it
//! with a 0 intensity anyway, but a clean black keeps the target well-defined).
//!
//! The upsample pipeline uses an additive blend so each coarser level accumulates
//! into the finer one. Per-step param buffers (threshold/knee/texel) live in a
//! persistent pool; the bind groups are rebuilt each frame (bloom is opt-in, and
//! `post_hdr` — the prefilter source — alternates when TAA is on).

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, HDR_FORMAT};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BloomParams {
    /// x = threshold, y = knee, zw = source texel size.
    v: [f32; 4],
    /// x = Karis-average first downsample (>0.5), yzw unused.
    v2: [f32; 4],
}

pub struct BloomNode {
    bgl: wgpu::BindGroupLayout,
    prefilter: wgpu::RenderPipeline,
    downsample: wgpu::RenderPipeline,
    upsample: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    /// One uniform buffer per bloom step (grown as needed, reused across frames).
    pool: Vec<wgpu::Buffer>,
}

impl BloomNode {
    pub fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bloom"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bloom.wgsl").into()),
            });
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bloom"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // The frame's exposure (wave VIS1b). Bound on every step,
                    // because a bind group must be complete; read only by the
                    // prefilter, which is the only step that keys on brightness.
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
                ],
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bloom"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let make = |entry: &str, blend: Option<wgpu::BlendState>| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("bloom"),
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
                            format: HDR_FORMAT,
                            blend,
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
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Self {
            prefilter: make("fs_prefilter", None),
            downsample: make("fs_down", None),
            upsample: make("fs_up", Some(additive)),
            bgl,
            sampler,
            pool: Vec::new(),
        }
    }

    /// A pooled uniform buffer for step `i`, written with `params`.
    fn step_buffer(&mut self, gpu: &GpuContext, i: usize, params: BloomParams) -> &wgpu::Buffer {
        while self.pool.len() <= i {
            self.pool
                .push(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("bloom-params"),
                    size: std::mem::size_of::<BloomParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
        gpu.queue
            .write_buffer(&self.pool[i], 0, bytemuck::bytes_of(&params));
        &self.pool[i]
    }

    fn bind(
        &self,
        gpu: &GpuContext,
        src: &wgpu::TextureView,
        params: &wgpu::Buffer,
        exposure: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: exposure.as_entire_binding(),
                },
            ],
        })
    }

    fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        dst: &wgpu::TextureView,
        bind_group: &wgpu::BindGroup,
        clear: bool,
    ) {
        let load = if clear {
            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
        } else {
            wgpu::LoadOp::Load
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bloom"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: dst,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl RenderNode for BloomNode {
    fn name(&self) -> &'static str {
        "bloom"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let mips = &frame.targets.bloom;
        let sizes = &frame.targets.bloom_sizes;
        if !frame.settings.bloom.enabled || mips.is_empty() {
            // Clear the bloom result so the tonemap adds nothing.
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom-clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &mips[0],
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
            return;
        }

        let n = mips.len();
        let b = &frame.settings.bloom;
        let mut step = 0usize;

        // Prefilter: post_hdr → mip 0.
        let texel0 = [1.0 / sizes[0].0 as f32, 1.0 / sizes[0].1 as f32];
        let pf = self
            .step_buffer(
                gpu,
                step,
                BloomParams {
                    v: [b.threshold, b.knee, texel0[0], texel0[1]],
                    v2: [if b.karis { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
                },
            )
            .clone();
        let pf_bg = self.bind(gpu, frame.post_hdr, &pf, &frame.exposure.state);
        self.draw(encoder, &self.prefilter, &mips[0], &pf_bg, true);
        step += 1;

        // Downsample chain: mip i-1 → mip i.
        for i in 1..n {
            let (sw, sh) = sizes[i - 1];
            let params = BloomParams {
                v: [0.0, 0.0, 1.0 / sw as f32, 1.0 / sh as f32],
                v2: [0.0; 4],
            };
            let buf = self.step_buffer(gpu, step, params).clone();
            let bg = self.bind(gpu, &mips[i - 1], &buf, &frame.exposure.state);
            self.draw(encoder, &self.downsample, &mips[i], &bg, true);
            step += 1;
        }

        // Additive upsample: mip i+1 → mip i (blended over the downsampled mip i).
        for i in (0..n - 1).rev() {
            let (sw, sh) = sizes[i + 1];
            let params = BloomParams {
                v: [0.0, 0.0, 1.0 / sw as f32, 1.0 / sh as f32],
                v2: [0.0; 4],
            };
            let buf = self.step_buffer(gpu, step, params).clone();
            let bg = self.bind(gpu, &mips[i + 1], &buf, &frame.exposure.state);
            self.draw(encoder, &self.upsample, &mips[i], &bg, false);
            step += 1;
        }
    }
}
