//! Composite node: blits the resolved scene into the output target
//! (swapchain or headless) with the stretch/letterbox transform, and applies
//! the selection/hover outline from the mask target.

use std::collections::HashMap;

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::FrameData;

/// How the scene maps onto an output of a different size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlitMode {
    /// Fill the output (used while the swapchain lags a resize — the few
    /// debounced frames stretch imperceptibly).
    Stretch,
    /// Preserve this aspect ratio, bar the rest (PIE / play-in-viewport
    /// later).
    Letterbox { aspect: f32 },
}

/// uv' = uv * scale + offset; uv' outside [0,1] shows the bar color.
pub fn letterbox_uv_transform(aspect: f32, out_w: u32, out_h: u32) -> [f32; 4] {
    let out_aspect = out_w.max(1) as f32 / out_h.max(1) as f32;
    if out_aspect > aspect {
        // Output is wider than the content: pillarbox. Horizontal uv range
        // grows past [0,1] so the content occupies the center strip.
        let sx = out_aspect / aspect;
        [sx, 1.0, (1.0 - sx) * 0.5, 0.0]
    } else {
        let sy = aspect / out_aspect;
        [1.0, sy, 0.0, (1.0 - sy) * 0.5]
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CompositeParams {
    uv_transform: [f32; 4],
    bg: [f32; 4],
    outline: [f32; 4],
}

pub struct CompositeNode {
    shader: wgpu::ShaderModule,
    bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    sampler: wgpu::Sampler,
    uniforms: wgpu::Buffer,
    pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    /// Bind group depends on the target views; rebuilt when targets recreate.
    bind_group: Option<(u64, wgpu::BindGroup)>,
    pub mode: BlitMode,
}

impl CompositeNode {
    pub fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("composite"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/composite.wgsl").into()),
            });

        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("composite"),
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
                label: Some("composite"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("composite"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniforms = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite-params"),
            size: std::mem::size_of::<CompositeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            shader,
            bgl,
            layout,
            sampler,
            uniforms,
            pipelines: HashMap::new(),
            bind_group: None,
            mode: BlitMode::Stretch,
        }
    }

    fn pipeline(&mut self, gpu: &GpuContext, format: wgpu::TextureFormat) -> &wgpu::RenderPipeline {
        let shader = &self.shader;
        let layout = &self.layout;
        self.pipelines.entry(format).or_insert_with(|| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("composite"),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: shader,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: shader,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
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
        })
    }
}

impl RenderNode for CompositeNode {
    fn name(&self) -> &'static str {
        "composite"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let uv_transform = match self.mode {
            BlitMode::Stretch => [1.0, 1.0, 0.0, 0.0],
            BlitMode::Letterbox { aspect } => {
                letterbox_uv_transform(aspect, frame.out_size.0, frame.out_size.1)
            }
        };
        let outline_on = !frame.scene.selected.is_empty() || frame.scene.hovered.is_some();
        let (tw, th) = frame.targets.size;
        gpu.queue.write_buffer(
            &self.uniforms,
            0,
            bytemuck::bytes_of(&CompositeParams {
                uv_transform,
                bg: [0.0, 0.0, 0.0, 1.0],
                outline: [
                    if outline_on { 1.0 } else { 0.0 },
                    1.0 / tw.max(1) as f32,
                    1.0 / th.max(1) as f32,
                    0.0,
                ],
            }),
        );

        if self
            .bind_group
            .as_ref()
            .is_none_or(|(gen, _)| *gen != frame.targets.generation)
        {
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("composite"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&frame.targets.scene_color),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&frame.targets.mask),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.uniforms.as_entire_binding(),
                    },
                ],
            });
            self.bind_group = Some((frame.targets.generation, bg));
        }

        let pipeline = self.pipeline(gpu, frame.out_format).clone();
        let bind_group = &self.bind_group.as_ref().unwrap().1;

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame.out_view,
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
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretch_matches_same_aspect() {
        // Same aspect → identity transform either branch.
        let t = letterbox_uv_transform(16.0 / 9.0, 1600, 900);
        assert!((t[0] - 1.0).abs() < 1e-6 && (t[1] - 1.0).abs() < 1e-6);
        assert!(t[2].abs() < 1e-6 && t[3].abs() < 1e-6);
    }

    #[test]
    fn wider_output_pillarboxes() {
        // 16:9 content on a 32:9 output: uv.x spans 2× centered.
        let t = letterbox_uv_transform(16.0 / 9.0, 3200, 900);
        assert!((t[0] - 2.0).abs() < 1e-5);
        assert!((t[2] - (-0.5)).abs() < 1e-5);
        // Center of the output still maps to the center of the content.
        let center = 0.5 * t[0] + t[2];
        assert!((center - 0.5).abs() < 1e-5);
    }

    #[test]
    fn taller_output_letterboxes() {
        let t = letterbox_uv_transform(16.0 / 9.0, 1600, 1800);
        assert!((t[1] - 2.0).abs() < 1e-5);
        assert!((t[3] - (-0.5)).abs() < 1e-5);
    }
}
