//! Depth prepass (P13.3a): a depth-only, single-sample render of the rigid mesh
//! instances into a **sampleable** `Depth32Float` target. [`super::ssao`]
//! reconstructs position/normal from it and [`super::taa`] reprojects against it.
//! A no-op unless SSAO or TAA is enabled ([`RenderSettings::needs_depth_prepass`]).
//!
//! This node covers the rigid [`MeshInstance`] geometry and **owns the clear**;
//! since wave VIS1a it is also the graph's prepass *anchor*
//! ([`RenderNode::opens_depth_prepass`]), so terrain, skinned meshes, voxel
//! volumes and fracture chunks record their own depth-only draws into the same
//! target immediately after this node runs — see [`crate::graph::RenderNode`] for
//! why that is a second entry point rather than four extra nodes.
//!
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
    /// R-P5 masked variant (`vs_masked` + `fs_masked`, alpha-test discard). Used
    /// only when the scene carries masked instances; otherwise the fragment-less
    /// `pipeline` above draws (byte-identical to the pre-R-P5 fast path).
    pipeline_masked: wgpu::RenderPipeline,
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
        let depth_stencil = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(DEPTH_COMPARE),
            stencil: Default::default(),
            bias: Default::default(),
        };
        // The fragment-less fast pipeline (opaque scenes) + the R-P5 masked variant
        // sharing everything but their vertex/fragment stages.
        let make = |label: &str, vs: &str, fs: Option<wgpu::FragmentState>| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some(vs),
                        compilation_options: Default::default(),
                        buffers: &vertex_layouts(),
                    },
                    fragment: fs,
                    primitive: wgpu::PrimitiveState {
                        cull_mode: Some(wgpu::Face::Back),
                        ..Default::default()
                    },
                    depth_stencil: Some(depth_stencil.clone()),
                    multisample: wgpu::MultisampleState::default(), // single-sample
                    multiview_mask: None,
                    cache: None,
                })
        };
        let pipeline = make("depth-prepass", "vs", None);
        let pipeline_masked = make(
            "depth-prepass-masked",
            "vs_masked",
            Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_masked"),
                compilation_options: Default::default(),
                targets: &[], // depth-only discard; no colour target
            }),
        );

        Self {
            pipeline,
            pipeline_masked,
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
        // Opaque+masked geometry only (translucent doesn't write depth here).
        let (raw, ranges, _translucent) = pack_bucketed(&frame.view.origin, &frame.scene.instances);
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

    /// This is the node that clears the prepass target, so it is the anchor the
    /// graph records every other node's contribution after (wave VIS1a).
    fn opens_depth_prepass(&self) -> bool {
        true
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
        // Only pay for the alpha-test fragment when the scene actually has masked
        // instances; otherwise the fragment-less fast pipeline draws (opaque
        // instances are unaffected by the discard, so this is purely a perf gate).
        let masked = frame.scene.instances.iter().any(|i| i.blend == 1);
        let pipeline = if masked {
            &self.pipeline_masked
        } else {
            &self.pipeline
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        self.prim.draw(&mut pass, instances, &self.ranges);
    }
}
