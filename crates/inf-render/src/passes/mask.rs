//! Selection/hover mask node: draws the selected and hovered instances as
//! flat silhouettes into the R8Unorm mask target (1.0 = selected, 0.5 =
//! hovered). The composite node dilates this into the colored outline.
//!
//! No depth test — we want the full screen silhouette so the outline wraps the
//! whole object even where nearer geometry would occlude it. Instance subsets
//! are small (the selection), so they're re-packed each frame.

use inf_math::FloatingOrigin;

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::{cube_geometry, vertex_layouts, InstanceRaw};
use crate::renderer::{FrameData, MASK_FORMAT};
use crate::scene::{MeshInstance, RenderScene};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaskParams {
    value: [f32; 4],
}

struct Batch {
    uniform_bg: wgpu::BindGroup,
    instances: Option<wgpu::Buffer>,
    capacity: usize,
    count: u32,
}

pub struct MaskNode {
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    selected: Batch,
    hovered: Batch,
}

impl MaskNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mask"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("mask").into()),
            });

        let mask_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mask-params"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let batch = |value: f32, label: &str| {
            let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<MaskParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(
                &buf,
                0,
                bytemuck::bytes_of(&MaskParams {
                    value: [value, 0.0, 0.0, 0.0],
                }),
            );
            let uniform_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &mask_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            Batch {
                uniform_bg,
                instances: None,
                capacity: 0,
                count: 0,
            }
        };

        let (verts, idx) = cube_geometry();
        let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mask-cube-vertices"),
            size: std::mem::size_of_val(verts.as_slice()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&vertices, 0, bytemuck::cast_slice(&verts));
        let indices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mask-cube-indices"),
            size: std::mem::size_of_val(idx.as_slice()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&indices, 0, bytemuck::cast_slice(&idx));

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mask"),
                bind_group_layouts: &[Some(view_bgl), Some(&mask_bgl)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mask"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &vertex_layouts(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_mask"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: MASK_FORMAT,
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

        Self {
            pipeline,
            vertices,
            indices,
            index_count: idx.len() as u32,
            selected: batch(1.0, "mask-selected"),
            hovered: batch(0.5, "mask-hovered"),
        }
    }

    fn sync_batch(
        gpu: &GpuContext,
        batch: &mut Batch,
        origin: &FloatingOrigin,
        picks: impl Iterator<Item = MeshInstance>,
    ) {
        let raw: Vec<InstanceRaw> = picks.map(|i| InstanceRaw::pack(origin, &i)).collect();
        batch.count = raw.len() as u32;
        if raw.is_empty() {
            return;
        }
        if batch.instances.is_none() || batch.capacity < raw.len() {
            let capacity = raw.len().next_power_of_two().max(8);
            batch.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mask-instances"),
                size: (capacity * std::mem::size_of::<InstanceRaw>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            batch.capacity = capacity;
        }
        gpu.queue.write_buffer(
            batch.instances.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&raw),
        );
    }

    /// Look up the full instance for each id (selection stores ids only).
    fn instances_for<'a>(
        scene: &'a RenderScene,
        ids: &'a [u32],
    ) -> impl Iterator<Item = MeshInstance> + 'a {
        ids.iter()
            .filter_map(move |id| scene.instances.iter().find(|i| i.id == *id).copied())
    }
}

impl RenderNode for MaskNode {
    fn name(&self) -> &'static str {
        "selection-mask"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let origin = &frame.view.origin;
        MaskNode::sync_batch(
            gpu,
            &mut self.selected,
            origin,
            MaskNode::instances_for(frame.scene, &frame.scene.selected),
        );
        // Don't double-draw an object that's both hovered and selected.
        let hovered: Vec<u32> = frame
            .scene
            .hovered
            .filter(|h| !frame.scene.selected.contains(h))
            .into_iter()
            .collect();
        MaskNode::sync_batch(
            gpu,
            &mut self.hovered,
            origin,
            MaskNode::instances_for(frame.scene, &hovered),
        );

        // Always run the pass to clear the mask (stale outline otherwise).
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("selection-mask"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.mask,
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
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        pass.set_vertex_buffer(0, self.vertices.slice(..));

        for batch in [&self.hovered, &self.selected] {
            if batch.count == 0 {
                continue;
            }
            let Some(instances) = batch.instances.as_ref() else {
                continue;
            };
            pass.set_bind_group(1, &batch.uniform_bg, &[]);
            pass.set_vertex_buffer(1, instances.slice(..));
            pass.draw_indexed(0..self.index_count, 0, 0..batch.count);
        }
    }
}
