//! Depth prepass (P13.3a): a depth-only, single-sample render of the rigid mesh
//! instances into a **sampleable** `Depth32Float` target. [`super::ssao`]
//! reconstructs position/normal from it and [`super::taa`] reprojects against it.
//! A no-op unless SSAO or TAA is enabled ([`RenderSettings::needs_depth_prepass`]).
//!
//! v1 covers the rigid [`MeshInstance`] geometry only (the SSAO golden is boxes);
//! folding terrain + skinned geometry into the prepass is a documented follow-up.
//! It re-packs its own instance buffer (mirroring [`super::mesh`]) so the pass is
//! self-contained — a small duplicate upload paid only on the opt-in SSAO/TAA path.

use std::ops::Range;

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::{pack_bucketed, vertex_layouts, InstanceRaw, EMPTY_RANGES};
use crate::primitives::PrimGpu;
use crate::renderer::FrameData;
use crate::settings::RenderSettings;

pub struct DepthPrepassNode {
    pipeline: wgpu::RenderPipeline,
    prim: PrimGpu,
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
    instance_count: u32,
    ranges: [Range<u32>; 5],
    uploaded_version: Option<(u64, glam::DVec3)>,
}

impl DepthPrepassNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("depth-prepass"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("depth_prepass").into()),
            });

        let prim = PrimGpu::new(gpu, "prepass");

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("depth-prepass"),
                bind_group_layouts: &[Some(view_bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("depth-prepass"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &vertex_layouts(),
                },
                fragment: None, // depth-only
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
                multisample: wgpu::MultisampleState::default(), // single-sample
                multiview_mask: None,
                cache: None,
            });

        Self {
            pipeline,
            prim,
            instances: None,
            instance_capacity: 0,
            instance_count: 0,
            ranges: EMPTY_RANGES,
            uploaded_version: None,
        }
    }

    fn sync(&mut self, gpu: &GpuContext, frame: &FrameData) {
        let key = (frame.scene.version, frame.view.origin.origin());
        if self.uploaded_version == Some(key) {
            return;
        }
        let (raw, ranges) = pack_bucketed(&frame.view.origin, &frame.scene.instances);
        self.ranges = ranges;
        self.instance_count = raw.len() as u32;
        if !raw.is_empty() {
            if self.instances.is_none() || self.instance_capacity < raw.len() {
                let capacity = raw.len().next_power_of_two().max(64);
                self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("prepass-instances"),
                    size: (capacity * std::mem::size_of::<InstanceRaw>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.instance_capacity = capacity;
            }
            gpu.queue.write_buffer(
                self.instances.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&raw),
            );
        }
        self.uploaded_version = Some(key);
    }
}

impl RenderNode for DepthPrepassNode {
    fn name(&self) -> &'static str {
        "depth-prepass"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        if !RenderSettings::needs_depth_prepass(frame.settings) {
            return;
        }
        self.sync(gpu, frame);

        // Always run (even with no instances) so the depth target is cleared to
        // DEPTH_CLEAR (0.0 = far) — SSAO then treats the whole frame as unoccluded.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("depth-prepass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &frame.targets.depth_prepass,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(crate::camera::DEPTH_CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if self.instance_count == 0 {
            return;
        }
        let Some(instances) = self.instances.as_ref() else {
            return;
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        self.prim.draw(&mut pass, instances, &self.ranges);
    }
}
