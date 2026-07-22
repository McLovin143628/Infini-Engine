//! Selection/hover mask node: draws the selected and hovered instances as
//! flat silhouettes into the R8Unorm mask target (1.0 = selected, 0.5 =
//! hovered). The composite node dilates this into the colored outline.
//!
//! No depth test — we want the full screen silhouette so the outline wraps the
//! whole object even where nearer geometry would occlude it. Instance subsets
//! are small (the selection), so they're re-packed each frame.

use std::collections::HashMap;

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::{cube_geometry, vertex_layouts, InstanceRaw};
use crate::renderer::{FrameData, MASK_FORMAT};

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
    /// Instance id → index into `scene.instances`, rebuilt once per scene version
    /// (replaces a per-selected-id linear scan of the whole instance list). First
    /// match wins, matching the previous `find`.
    id_index: HashMap<u32, usize>,
    /// Scene version the `id_index` reflects.
    index_version: Option<u64>,
    /// (scene version, origin, selection, hover) the packed batches reflect. The
    /// selection and hover change **without** bumping `scene.version`, so they ride
    /// in the gate key alongside the version + origin.
    uploaded_key: Option<(u64, glam::DVec3, Vec<u32>, Option<u32>)>,
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
            id_index: HashMap::new(),
            index_version: None,
            uploaded_key: None,
        }
    }

    /// Upload the packed instances into `batch`, growing (never shrinking) its
    /// buffer. An empty `raw` leaves the buffer intact and sets `count = 0`, so the
    /// draw loop skips it.
    fn upload_batch(gpu: &GpuContext, batch: &mut Batch, raw: &[InstanceRaw]) {
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
            bytemuck::cast_slice(raw),
        );
    }

    /// Rebuild the id→index map (once per scene version) and re-pack the selected /
    /// hovered instance buffers — but only when the `(version, origin, selection,
    /// hover)` gate key changed. `run` still executes the pass every frame to clear
    /// the mask; only the CPU packing + buffer upload is gated here.
    fn sync(&mut self, gpu: &GpuContext, frame: &FrameData) {
        let scene = frame.scene;
        let origin = &frame.view.origin;

        // Rebuild the id→index map once per scene version (first match wins, matching
        // the previous linear `find`), so selection lookups are O(1) instead of a
        // full instance scan per selected id.
        if self.index_version != Some(scene.version) {
            self.id_index.clear();
            for (idx, inst) in scene.instances.iter().enumerate() {
                self.id_index.entry(inst.id).or_insert(idx);
            }
            self.index_version = Some(scene.version);
        }

        // Don't double-draw an object that's both hovered and selected.
        let hovered = scene.hovered.filter(|h| !scene.selected.contains(h));

        // Gate re-packing; only clone the selection into the key when it changed.
        let unchanged = self.uploaded_key.as_ref().is_some_and(|(v, o, sel, hov)| {
            *v == scene.version
                && *o == origin.origin()
                && sel.as_slice() == scene.selected.as_slice()
                && *hov == hovered
        });
        if unchanged {
            return;
        }
        self.uploaded_key = Some((
            scene.version,
            origin.origin(),
            scene.selected.clone(),
            hovered,
        ));

        let sel_raw: Vec<InstanceRaw> = scene
            .selected
            .iter()
            .filter_map(|id| {
                self.id_index
                    .get(id)
                    .map(|&i| InstanceRaw::pack(origin, &scene.instances[i]))
            })
            .collect();
        let hov_raw: Vec<InstanceRaw> = hovered
            .iter()
            .filter_map(|id| {
                self.id_index
                    .get(id)
                    .map(|&i| InstanceRaw::pack(origin, &scene.instances[i]))
            })
            .collect();
        Self::upload_batch(gpu, &mut self.selected, &sel_raw);
        Self::upload_batch(gpu, &mut self.hovered, &hov_raw);
    }
}

impl RenderNode for MaskNode {
    fn name(&self) -> &'static str {
        "selection-mask"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        self.sync(gpu, frame);

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
