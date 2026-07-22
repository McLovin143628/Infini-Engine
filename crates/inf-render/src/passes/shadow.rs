//! Cascaded shadow-map pass (P13.3b): renders the first directional light's three
//! view-frustum cascades into a `Depth32Float` array, and publishes the per-cascade
//! matrices + bias constants in the shared [`ShadowResources`] uniform the lit
//! passes sample (see [`crate::passes::EnvBinding`] + `shaders/mesh.wgsl`).
//!
//! The cascade split scheme, sphere fit, and texel snapping are pure functions in
//! [`crate::csm`]. This node owns the caster pipeline (a depth-only forward-Z
//! render, mirroring [`crate::passes::depth_prepass`]) and re-packs its own cube
//! instance buffer, so it is self-contained.
//!
//! ## Scope (v1, documented)
//!
//! * **Casters:** rigid [`MeshInstance`] geometry only (the golden is boxes on a
//!   plane). Terrain-patch + skinned casters are a follow-up.
//! * **Receivers:** mesh / skinned / terrain all *sample* the cascades (shadows
//!   land on all of them); only casting is scoped.
//! * **Off path:** when [`ShadowSettings::enabled`] is false the node still writes
//!   the shared uniform (with `enabled = 0`) so receivers read a valid flag, then
//!   renders nothing — receivers take the byte-stable un-shadowed instruction path.

use crate::camera::{DEPTH_FORMAT, SUN_DIR};
use crate::csm::{
    bounding_sphere, cascade_matrix, cascade_splits, frustum_slice_corners, SHADOW_CASCADES,
    SHADOW_RESOLUTION,
};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::{cube_geometry, vertex_layouts, InstanceRaw};
use crate::renderer::FrameData;
use crate::scene::LightKind;

/// Forward-Z shadow depth: nearest caster wins (clear to 1.0 = far, keep smaller).
const SHADOW_DEPTH_COMPARE: wgpu::CompareFunction = wgpu::CompareFunction::LessEqual;
const SHADOW_DEPTH_CLEAR: f32 = 1.0;

/// The shared shadow uniform block (`std140`), written by [`ShadowNode`] and read
/// by every lit pass through [`crate::passes::EnvBinding`]. Mirrors `struct
/// ShadowData` in the receiver shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowDataGpu {
    /// Per-cascade forward-Z light `view_proj`.
    pub cascade_vp: [[f32; 16]; SHADOW_CASCADES],
    /// Cascade far distances (x,y,z); w unused.
    pub splits: [f32; 4],
    /// Per-cascade world texel size (x,y,z); w unused (drives normal-offset bias).
    pub texel_world: [f32; 4],
    /// x = enabled, y = depth_bias (NDC), z = normal_bias (texels),
    /// w = cascade count.
    pub params: [f32; 4],
}

impl ShadowDataGpu {
    fn disabled() -> Self {
        Self {
            cascade_vp: [[0.0; 16]; SHADOW_CASCADES],
            splits: [0.0; 4],
            texel_world: [0.0; 4],
            params: [0.0, 0.0, 0.0, SHADOW_CASCADES as f32],
        }
    }
}

/// Renderer-owned shadow GPU resources, created once (independent of viewport size)
/// and shared with the lit passes via [`FrameData`]. The [`ShadowNode`] renders the
/// cascades into [`layer_views`](Self::layer_views) and writes [`uniform`](Self::uniform);
/// receivers sample [`array_view`](Self::array_view) + read the uniform.
pub struct ShadowResources {
    _map: wgpu::Texture,
    /// Full depth array view (receiver sampling).
    pub array_view: wgpu::TextureView,
    /// Per-cascade single-layer views (caster rendering).
    pub layer_views: Vec<wgpu::TextureView>,
    /// Shared `ShadowData` uniform (written by the node, read by receivers).
    pub uniform: wgpu::Buffer,
    /// Stable generation for receiver bind-group caching (resources never resize).
    pub generation: u64,
}

impl ShadowResources {
    pub fn new(gpu: &GpuContext) -> Self {
        let map = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-map"),
            size: wgpu::Extent3d {
                width: SHADOW_RESOLUTION,
                height: SHADOW_RESOLUTION,
                depth_or_array_layers: SHADOW_CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let array_view = map.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow-map-array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layer_views = (0..SHADOW_CASCADES as u32)
            .map(|layer| {
                map.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow-cascade"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow-data"),
            size: std::mem::size_of::<ShadowDataGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Seed with the disabled state so a receiver bound before the first
        // ShadowNode run still reads enabled = 0.
        gpu.queue
            .write_buffer(&uniform, 0, bytemuck::bytes_of(&ShadowDataGpu::disabled()));
        Self {
            _map: map,
            array_view,
            layer_views,
            uniform,
            generation: 0,
        }
    }
}

/// Per-cascade caster uniform (`struct Cascade` in `shadow_depth.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CascadeGpu {
    view_proj: [f32; 16],
}

pub struct ShadowNode {
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    /// One tiny uniform + bind group per cascade (distinct buffers so the
    /// per-cascade writes don't collide before the passes run).
    cascade_bufs: Vec<wgpu::Buffer>,
    cascade_bgs: Vec<wgpu::BindGroup>,
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
    instance_count: u32,
    uploaded_version: Option<(u64, glam::DVec3)>,
    /// Whether the disabled-shadows uniform is already published to the (stable,
    /// created-once) `frame.shadow.uniform`. Gates the constant re-write while
    /// shadows stay off; the enabled path clears it so a later disable re-publishes.
    published_disabled: bool,
}

impl ShadowNode {
    pub fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shadow-depth"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/shadow_depth.wgsl").into(),
                ),
            });

        let cascade_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow-cascade"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let (verts, idx) = cube_geometry();
        let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow-cube-vertices"),
            size: std::mem::size_of_val(verts.as_slice()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&vertices, 0, bytemuck::cast_slice(&verts));
        let indices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow-cube-indices"),
            size: std::mem::size_of_val(idx.as_slice()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&indices, 0, bytemuck::cast_slice(&idx));

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow-depth"),
                bind_group_layouts: &[Some(&cascade_bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow-depth"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &vertex_layouts(),
                },
                fragment: None, // depth-only
                primitive: wgpu::PrimitiveState {
                    // Front-face culling reduces peter-panning/acne on the classic
                    // box casters (shadow depth from back faces).
                    cull_mode: Some(wgpu::Face::Front),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(SHADOW_DEPTH_COMPARE),
                    stencil: Default::default(),
                    // A slope-scaled hardware depth bias further reduces acne.
                    bias: wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let cascade_bufs: Vec<wgpu::Buffer> = (0..SHADOW_CASCADES)
            .map(|c| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("shadow-cascade-{c}")),
                    size: std::mem::size_of::<CascadeGpu>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let cascade_bgs = cascade_bufs
            .iter()
            .map(|buf| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow-cascade"),
                    layout: &cascade_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                })
            })
            .collect();

        Self {
            pipeline,
            vertices,
            indices,
            index_count: idx.len() as u32,
            cascade_bufs,
            cascade_bgs,
            instances: None,
            instance_capacity: 0,
            instance_count: 0,
            uploaded_version: None,
            published_disabled: false,
        }
    }

    /// Re-pack the caster instance buffer when the scene changed or the origin
    /// rebased (mirrors [`crate::passes::depth_prepass`]).
    fn sync(&mut self, gpu: &GpuContext, frame: &FrameData) {
        let key = (frame.scene.version, frame.view.origin.origin());
        if self.uploaded_version == Some(key) {
            return;
        }
        let raw: Vec<InstanceRaw> = frame
            .scene
            .instances
            .iter()
            .map(|i| InstanceRaw::pack(&frame.view.origin, i))
            .collect();
        self.instance_count = raw.len() as u32;
        if !raw.is_empty() {
            if self.instances.is_none() || self.instance_capacity < raw.len() {
                let capacity = raw.len().next_power_of_two().max(64);
                self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("shadow-instances"),
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

impl RenderNode for ShadowNode {
    fn name(&self) -> &'static str {
        "shadow"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let s = &frame.settings.shadows;
        if !s.enabled {
            // Publish the disabled uniform (valid enabled=0 flag) once and render
            // nothing; the buffer is created once and only this node writes it, so
            // re-publishing the same constant every frame is redundant.
            if !self.published_disabled {
                gpu.queue.write_buffer(
                    &frame.shadow.uniform,
                    0,
                    bytemuck::bytes_of(&ShadowDataGpu::disabled()),
                );
                self.published_disabled = true;
            }
            return;
        }
        // The enabled path overwrites the uniform below, so a later disable must
        // re-publish the disabled block.
        self.published_disabled = false;
        self.sync(gpu, frame);

        // Shadow light: the first directional light's direction-to-light, else the
        // editor fallback sun (matches the receiver shaders' fallback).
        let light_dir_to = frame
            .scene
            .lights
            .iter()
            .find(|l| l.kind == LightKind::Directional)
            .map(|l| l.direction.normalize_or_zero())
            .filter(|d| d.length_squared() > 1e-6)
            .unwrap_or(SUN_DIR.normalize());

        // Cascade splits across the shadow range.
        let near = frame.view.near.max(0.05);
        let far = s.max_distance.max(near + 1.0);
        let splits = cascade_splits(near, far, s.lambda);

        let eye = frame.view.eye_local();
        let fwd = frame.view.forward;
        let up = frame.view.up;
        let fov = frame.view.fov_y;
        let aspect = frame.view.aspect();

        let mut data = ShadowDataGpu::disabled();
        data.params = [1.0, s.depth_bias, s.normal_bias, SHADOW_CASCADES as f32];
        for c in 0..SHADOW_CASCADES {
            let d0 = if c == 0 { near } else { splits[c - 1] };
            let d1 = splits[c];
            let corners = frustum_slice_corners(eye, fwd, up, fov, aspect, d0, d1);
            let (center, radius) = bounding_sphere(&corners);
            let (vp, texel) = cascade_matrix(light_dir_to, center, radius, SHADOW_RESOLUTION);
            data.cascade_vp[c] = vp.to_cols_array();
            data.splits[c] = d1;
            data.texel_world[c] = texel;
            gpu.queue.write_buffer(
                &self.cascade_bufs[c],
                0,
                bytemuck::bytes_of(&CascadeGpu {
                    view_proj: vp.to_cols_array(),
                }),
            );
        }
        gpu.queue
            .write_buffer(&frame.shadow.uniform, 0, bytemuck::bytes_of(&data));

        // One depth-only render per cascade (always run so the layer is cleared to
        // far even with no casters → everything reads as lit).
        for c in 0..SHADOW_CASCADES {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow-cascade"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &frame.shadow.layer_views[c],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(SHADOW_DEPTH_CLEAR),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if self.instance_count == 0 {
                continue;
            }
            let Some(instances) = self.instances.as_ref() else {
                continue;
            };
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.cascade_bgs[c], &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.set_vertex_buffer(1, instances.slice(..));
            pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..self.index_count, 0, 0..self.instance_count);
        }
    }
}
