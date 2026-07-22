//! ID-buffer picking: render scene instance ids into an R32Uint target and
//! read back a single pixel under the cursor. On-demand (only on click), so it
//! adds nothing to the per-frame cost; the target is sized to the viewport and
//! reused across picks.
//!
//! Gizmo handles are picked analytically on the CPU (see `gizmo.rs`); this
//! covers scene objects, whose silhouettes make a pixel-exact id buffer the
//! right tool.

use inf_math::FloatingOrigin;

use crate::camera::RenderView;
use crate::gpu::GpuContext;
use crate::passes::mesh::{cube_geometry, vertex_layouts, InstanceRaw};
use crate::scene::{RenderScene, ID_NONE};

const ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

pub struct Picker {
    pipeline: wgpu::RenderPipeline,
    view_buf: wgpu::Buffer,
    view_bg: wgpu::BindGroup,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
    target: Option<(u32, u32, wgpu::Texture, wgpu::TextureView)>,
    depth: Option<(u32, u32, wgpu::TextureView)>,
}

impl Picker {
    pub fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("pick"),
                source: wgpu::ShaderSource::Wgsl(crate::passes::shader_source("mesh").into()),
            });

        let view_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick-view"),
            size: std::mem::size_of::<crate::camera::ViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("pick-view"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let view_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pick-view"),
            layout: &view_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_buf.as_entire_binding(),
            }],
        });

        let (verts, idx) = cube_geometry();
        let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick-cube-vertices"),
            size: std::mem::size_of_val(verts.as_slice()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&vertices, 0, bytemuck::cast_slice(&verts));
        let indices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick-cube-indices"),
            size: std::mem::size_of_val(idx.as_slice()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&indices, 0, bytemuck::cast_slice(&idx));

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pick"),
                bind_group_layouts: &[Some(&view_bgl)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("pick"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &vertex_layouts(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_id"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ID_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: crate::camera::DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(crate::camera::DEPTH_COMPARE),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });

        Self {
            pipeline,
            view_buf,
            view_bg,
            vertices,
            indices,
            index_count: idx.len() as u32,
            instances: None,
            instance_capacity: 0,
            target: None,
            depth: None,
        }
    }

    fn ensure_target(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        if self
            .target
            .as_ref()
            .is_none_or(|(w, h, ..)| *w != width || *h != height)
        {
            let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("pick-id"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: ID_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.target = Some((width, height, tex, view));
        }
        if self
            .depth
            .as_ref()
            .is_none_or(|(w, h, _)| *w != width || *h != height)
        {
            let d = gpu
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("pick-depth"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: crate::camera::DEPTH_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.depth = Some((width, height, d));
        }
    }

    fn upload_instances(&mut self, gpu: &GpuContext, origin: &FloatingOrigin, scene: &RenderScene) {
        let raw: Vec<InstanceRaw> = scene
            .instances
            .iter()
            .map(|i| InstanceRaw::pack(origin, i))
            .collect();
        if raw.is_empty() {
            return;
        }
        if self.instances.is_none() || self.instance_capacity < raw.len() {
            let capacity = raw.len().next_power_of_two().max(64);
            self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pick-instances"),
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

    /// Render the id buffer and read the id at `(px, py)` (physical pixels,
    /// top-left origin). Returns `None` for the background (no object).
    pub fn pick(
        &mut self,
        gpu: &GpuContext,
        scene: &RenderScene,
        view: &RenderView,
        px: u32,
        py: u32,
    ) -> Option<u32> {
        let (width, height) = (view.width.max(1), view.height.max(1));
        if px >= width || py >= height || scene.instances.is_empty() {
            return None;
        }
        self.ensure_target(gpu, width, height);
        self.upload_instances(gpu, &view.origin, scene);
        let instances = self.instances.as_ref()?;

        gpu.queue.write_buffer(
            &self.view_buf,
            0,
            bytemuck::bytes_of(&crate::camera::ViewUniforms::from_view(view)),
        );

        let (_, _, _, id_view) = self.target.as_ref().unwrap();
        let (_, _, depth_view) = self.depth.as_ref().unwrap();

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pick"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pick"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: id_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: ID_NONE as f64,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(crate::camera::DEPTH_CLEAR),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.view_bg, &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.set_vertex_buffer(1, instances.slice(..));
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..self.index_count, 0, 0..scene.instances.len() as u32);
        }

        // Copy just the one texel. R32Uint = 4 bytes; bytes_per_row must be
        // 256-aligned, so we copy a 1-row 256-byte buffer.
        let read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick-readback"),
            size: wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let (_, _, id_tex, _) = self.target.as_ref().unwrap();
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: id_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &read,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit([encoder.finish()]);

        let slice = read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        if gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .is_err()
        {
            return None;
        }
        if rx.recv().ok()?.is_err() {
            return None;
        }
        let data = slice.get_mapped_range().ok()?;
        let id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        read.unmap();

        (id != ID_NONE).then_some(id)
    }
}
