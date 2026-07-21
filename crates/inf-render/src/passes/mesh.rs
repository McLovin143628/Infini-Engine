//! Instanced forward mesh pass. Phase 2 draws unit cubes only (real meshes
//! arrive with the asset system, P4); everything else — instance packing
//! against the floating origin, version-gated uploads, the 10k-at-vsync
//! target — is production-shaped.

use glam::Mat3;
use inf_math::FloatingOrigin;

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::{LightKind, MeshInstance, RenderScene};

/// Vertex: position + normal.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
}

/// Per-instance GPU data. Matches the `@location(3..=13)` attributes in
/// mesh.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    pub model: [f32; 16],
    /// Normal matrix (inverse-transpose of the model 3×3), 3 padded columns.
    pub normal_mat: [f32; 12],
    pub color: [f32; 4],
    /// x = pick id.
    pub misc: [u32; 4],
    /// PBR params: x = metallic, y = roughness (z, w reserved).
    pub pbr: [f32; 4],
    /// Emissive color: rgb (w reserved).
    pub emissive: [f32; 4],
}

impl InstanceRaw {
    pub fn pack(origin: &FloatingOrigin, inst: &MeshInstance) -> Self {
        let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
        // Inverse-transpose of R·S = R·S⁻¹ for the normal transform.
        let inv_scale = inst.scale.max(glam::Vec3::splat(1e-6)).recip();
        let nrm = Mat3::from_quat(inst.rotation) * Mat3::from_diagonal(inv_scale);
        let n = nrm.to_cols_array_2d();
        Self {
            model: model.to_cols_array(),
            normal_mat: [
                n[0][0], n[0][1], n[0][2], 0.0, //
                n[1][0], n[1][1], n[1][2], 0.0, //
                n[2][0], n[2][1], n[2][2], 0.0,
            ],
            color: inst.color,
            misc: [inst.id, 0, 0, 0],
            pbr: [inst.metallic, inst.roughness, 0.0, 0.0],
            emissive: [inst.emissive[0], inst.emissive[1], inst.emissive[2], 0.0],
        }
    }
}

pub const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 11] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 0,
        shader_location: 3,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 16,
        shader_location: 4,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 32,
        shader_location: 5,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 48,
        shader_location: 6,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 64,
        shader_location: 7,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 80,
        shader_location: 8,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 96,
        shader_location: 9,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 112,
        shader_location: 10,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x4,
        offset: 128,
        shader_location: 11,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 144,
        shader_location: 12,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 160,
        shader_location: 13,
    },
];

pub const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 12,
        shader_location: 1,
    },
];

pub fn vertex_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
    [
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRIBUTES,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRIBUTES,
        }),
    ]
}

/// Unit cube centered at the origin (extent ±0.5), 24 verts / 36 indices.
pub fn cube_geometry() -> (Vec<MeshVertex>, Vec<u16>) {
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        // (normal, tangent u, tangent v)
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    let mut verts = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (n, u, v) in faces {
        let n3 = glam::Vec3::from(n);
        let u3 = glam::Vec3::from(u);
        let v3 = glam::Vec3::from(v);
        let base = verts.len() as u16;
        for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let p = (n3 + u3 * su + v3 * sv) * 0.5;
            verts.push(MeshVertex {
                pos: p.to_array(),
                normal: n,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, indices)
}

/// Max scene lights uploaded per frame (must match `MAX_LIGHTS` in mesh.wgsl).
pub const MAX_LIGHTS: usize = 16;

/// One GPU light, std140-friendly (all vec4). `pos_dir.w` = kind (0 directional,
/// 1 point); for directional, `pos_dir.xyz` is the unit direction toward the
/// light; for point, the render-local position. `color.a` = intensity.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuLight {
    color: [f32; 4],
    pos_dir: [f32; 4],
    params: [f32; 4], // x = range
}

/// The lights uniform block bound at `@group(1)`. Shared by the rigid mesh pass
/// and the skinned mesh pass (identical lighting model).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LightsUniform {
    count: [u32; 4], // x = active count
    items: [GpuLight; MAX_LIGHTS],
}

impl LightsUniform {
    /// Project world-space scene lights into render-local GPU lights.
    pub(crate) fn from_scene(scene: &RenderScene, origin: &FloatingOrigin) -> Self {
        let mut items = [GpuLight {
            color: [0.0; 4],
            pos_dir: [0.0; 4],
            params: [0.0; 4],
        }; MAX_LIGHTS];
        let count = scene.lights.len().min(MAX_LIGHTS);
        for (slot, light) in items.iter_mut().zip(scene.lights.iter()).take(count) {
            slot.color = [
                light.color[0],
                light.color[1],
                light.color[2],
                light.intensity,
            ];
            match light.kind {
                LightKind::Directional => {
                    let d = light.direction.normalize_or_zero();
                    slot.pos_dir = [d.x, d.y, d.z, 0.0];
                }
                LightKind::Point => {
                    // Render-local position (origin-relative), like instances.
                    let p = (light.position - origin.origin()).as_vec3();
                    slot.pos_dir = [p.x, p.y, p.z, 1.0];
                    slot.params[0] = light.range;
                }
            }
        }
        Self {
            count: [count as u32, 0, 0, 0],
            items,
        }
    }
}

pub struct MeshNode {
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
    uploaded_version: Option<(u64, glam::DVec3)>,
    instance_count: u32,
    lights_buf: wgpu::Buffer,
    lights_bg: wgpu::BindGroup,
    env: super::EnvBinding,
}

impl MeshNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mesh"),
                source: wgpu::ShaderSource::Wgsl(
                    super::lit_scene_shader(include_str!("../shaders/mesh.wgsl"), 2).into(),
                ),
            });

        let (verts, idx) = cube_geometry();
        let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cube-vertices"),
            size: std::mem::size_of_val(verts.as_slice()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&vertices, 0, bytemuck::cast_slice(&verts));
        let indices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cube-indices"),
            size: std::mem::size_of_val(idx.as_slice()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&indices, 0, bytemuck::cast_slice(&idx));

        // Lights uniform block (@group(1)).
        let lights_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mesh-lights"),
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
        let lights_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh-lights"),
            size: std::mem::size_of::<LightsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mesh-lights"),
            layout: &lights_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights_buf.as_entire_binding(),
            }],
        });

        let env = super::EnvBinding::new(gpu);
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mesh"),
                bind_group_layouts: &[Some(view_bgl), Some(&lights_bgl), Some(&env.bgl)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mesh"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &vertex_layouts(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
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
                multisample: wgpu::MultisampleState {
                    count: SCENE_SAMPLES,
                    ..Default::default()
                },
                multiview_mask: None,
                cache: None,
            });

        Self {
            pipeline,
            vertices,
            indices,
            index_count: idx.len() as u32,
            instances: None,
            instance_capacity: 0,
            uploaded_version: None,
            instance_count: 0,
            lights_buf,
            lights_bg,
            env,
        }
    }

    /// Re-pack + upload instances when the scene changed or the floating
    /// origin rebased (model matrices are origin-relative).
    fn sync_instances(&mut self, gpu: &GpuContext, frame: &FrameData) {
        let key = (frame.scene.version, frame.view.origin.origin());
        if self.uploaded_version == Some(key) {
            return;
        }

        // Lights depend on the same key (scene version + origin, since point
        // positions are render-local).
        let lights = LightsUniform::from_scene(frame.scene, &frame.view.origin);
        gpu.queue
            .write_buffer(&self.lights_buf, 0, bytemuck::bytes_of(&lights));

        let raw: Vec<InstanceRaw> = frame
            .scene
            .instances
            .iter()
            .map(|i| InstanceRaw::pack(&frame.view.origin, i))
            .collect();
        self.instance_count = raw.len() as u32;

        if !raw.is_empty() {
            let needed = raw.len();
            if self.instances.is_none() || self.instance_capacity < needed {
                let capacity = needed.next_power_of_two().max(64);
                self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mesh-instances"),
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

    /// Shared by the ID pass: bind cube + instances and draw everything.
    pub fn draw_all<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>) {
        if self.instance_count == 0 {
            return;
        }
        let Some(instances) = self.instances.as_ref() else {
            return;
        };
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_vertex_buffer(1, instances.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..self.instance_count);
    }
}

impl RenderNode for MeshNode {
    fn name(&self) -> &'static str {
        "mesh"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        self.sync_instances(gpu, frame);
        if self.instance_count == 0 {
            return;
        }
        // Cheap Arc handle clone → no borrow of `self.env` held during the pass.
        let env_bg = self.env.bind_group(gpu, frame).clone();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.color_msaa,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &frame.targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_bind_group(1, &self.lights_bg, &[]);
        pass.set_bind_group(2, &env_bg, &[]);
        self.draw_all(&mut pass);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec3, Quat, Vec3};

    #[test]
    fn instance_raw_matches_attribute_offsets() {
        let v = InstanceRaw::pack(
            &FloatingOrigin::new(DVec3::ZERO),
            &MeshInstance::lit(DVec3::ZERO, Quat::IDENTITY, Vec3::ONE, [1.0; 4], 1),
        );
        let base = &v as *const InstanceRaw as usize;
        assert_eq!(std::mem::size_of::<InstanceRaw>(), 176);
        assert_eq!(&v.normal_mat as *const _ as usize - base, 64);
        assert_eq!(&v.color as *const _ as usize - base, 112);
        assert_eq!(&v.misc as *const _ as usize - base, 128);
        assert_eq!(&v.pbr as *const _ as usize - base, 144);
        assert_eq!(&v.emissive as *const _ as usize - base, 160);
    }

    #[test]
    fn cube_has_24_verts_36_indices() {
        let (v, i) = cube_geometry();
        assert_eq!(v.len(), 24);
        assert_eq!(i.len(), 36);
        // All verts on the ±0.5 cube surface.
        for vert in &v {
            let p = Vec3::from(vert.pos);
            assert!((p.abs().max_element() - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn pack_respects_floating_origin() {
        let origin = FloatingOrigin::new(DVec3::new(1000.0, 0.0, 0.0));
        let raw = InstanceRaw::pack(
            &origin,
            &MeshInstance::lit(
                DVec3::new(1001.0, 2.0, 3.0),
                Quat::IDENTITY,
                Vec3::ONE,
                [1.0; 4],
                7,
            ),
        );
        // Translation column is origin-relative.
        assert_eq!(&raw.model[12..15], &[1.0, 2.0, 3.0]);
        assert_eq!(raw.misc[0], 7);
    }
}
