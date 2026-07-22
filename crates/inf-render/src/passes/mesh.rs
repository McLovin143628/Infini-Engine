//! Instanced forward mesh pass. Phase 2 draws unit cubes only (real meshes
//! arrive with the asset system, P4); everything else — instance packing
//! against the floating origin, version-gated uploads, the 10k-at-vsync
//! target — is production-shaped.

use std::ops::Range;

use bytemuck::Zeroable;
use glam::Mat3;
use inf_math::FloatingOrigin;

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::primitives::PrimGpu;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::{LightKind, MeshInstance, RenderScene};

/// The five built-in primitive geometries live in one packed GPU buffer pair
/// (`passes::mesh` re-exports the cube generator for existing call sites).
pub use crate::primitives::cube_geometry;

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

/// Pack scene instances into GPU [`InstanceRaw`], **bucketed by primitive kind**
/// (stable within each kind), and return the per-kind sub-ranges of the packed
/// vector (indexed by [`PrimMesh::index`]). Because `Cube` is kind 0 and the
/// partition preserves original order within a kind, an all-cube scene produces a
/// `Vec<InstanceRaw>` byte-identical (and same-ordered) to a naïve linear pack —
/// the byte-stability guarantee for every pre-R-P1 golden.
pub(crate) fn pack_bucketed(
    origin: &FloatingOrigin,
    instances: &[MeshInstance],
) -> (Vec<InstanceRaw>, [Range<u32>; 5]) {
    let mut counts = [0u32; 5];
    for inst in instances {
        counts[inst.mesh.index()] += 1;
    }
    let mut starts = [0u32; 5];
    let mut acc = 0u32;
    for k in 0..5 {
        starts[k] = acc;
        acc += counts[k];
    }
    let ranges: [Range<u32>; 5] = std::array::from_fn(|k| starts[k]..starts[k] + counts[k]);

    let mut raw = vec![InstanceRaw::zeroed(); instances.len()];
    let mut cursor = starts;
    for inst in instances {
        let k = inst.mesh.index();
        raw[cursor[k] as usize] = InstanceRaw::pack(origin, inst);
        cursor[k] += 1;
    }
    (raw, ranges)
}

/// An empty per-kind range set (no draws) — the initial state before any upload.
pub(crate) const EMPTY_RANGES: [Range<u32>; 5] = [0..0, 0..0, 0..0, 0..0, 0..0];

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
    prim: PrimGpu,
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
    uploaded_version: Option<(u64, glam::DVec3)>,
    instance_count: u32,
    /// Per-primitive-kind sub-ranges of the packed instance buffer.
    ranges: [Range<u32>; 5],
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
                source: wgpu::ShaderSource::Wgsl(super::shader_source("mesh").into()),
            });

        let prim = PrimGpu::new(gpu, "mesh");

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
            prim,
            instances: None,
            instance_capacity: 0,
            uploaded_version: None,
            instance_count: 0,
            ranges: EMPTY_RANGES,
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

        let (raw, ranges) = pack_bucketed(&frame.view.origin, &frame.scene.instances);
        self.ranges = ranges;
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

    /// Bind the shared primitive geometry + instances and draw every kind's
    /// bucket (≤5 `draw_indexed` calls).
    pub fn draw_all<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>) {
        if self.instance_count == 0 {
            return;
        }
        let Some(instances) = self.instances.as_ref() else {
            return;
        };
        self.prim.draw(pass, instances, &self.ranges);
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
    use crate::primitives::PrimMesh;
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

    fn inst(mesh: PrimMesh, id: u32) -> MeshInstance {
        MeshInstance {
            mesh,
            id,
            ..MeshInstance::lit(
                DVec3::new(id as f64, 0.0, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
                [1.0; 4],
                id,
            )
        }
    }

    #[test]
    fn pack_bucketed_all_cube_matches_linear() {
        let origin = FloatingOrigin::new(DVec3::ZERO);
        let insts: Vec<MeshInstance> = (1..=5).map(|i| inst(PrimMesh::Cube, i)).collect();
        let naive: Vec<InstanceRaw> = insts
            .iter()
            .map(|i| InstanceRaw::pack(&origin, i))
            .collect();
        let (bucketed, ranges) = pack_bucketed(&origin, &insts);
        // Byte-identical to the old linear pack (the golden byte-stability gate).
        assert_eq!(
            bytemuck::cast_slice::<InstanceRaw, u8>(&naive),
            bytemuck::cast_slice::<InstanceRaw, u8>(&bucketed),
        );
        assert_eq!(ranges[0], 0..5);
        for r in &ranges[1..] {
            assert!(r.is_empty());
        }
    }

    #[test]
    fn pack_bucketed_stable_within_kind() {
        let origin = FloatingOrigin::new(DVec3::ZERO);
        // Interleaved kinds; ids encode original order.
        let insts = vec![
            inst(PrimMesh::Sphere, 10),
            inst(PrimMesh::Cube, 20),
            inst(PrimMesh::Sphere, 11),
            inst(PrimMesh::Cone, 30),
            inst(PrimMesh::Cube, 21),
        ];
        let (raw, ranges) = pack_bucketed(&origin, &insts);

        // Cube bucket first (ids 20, 21 in original order).
        assert_eq!(ranges[PrimMesh::Cube.index()], 0..2);
        assert_eq!(raw[0].misc[0], 20);
        assert_eq!(raw[1].misc[0], 21);
        // Sphere next (10, 11).
        assert_eq!(ranges[PrimMesh::Sphere.index()], 2..4);
        assert_eq!(raw[2].misc[0], 10);
        assert_eq!(raw[3].misc[0], 11);
        // Cone last (30); plane + cylinder empty.
        assert_eq!(ranges[PrimMesh::Cone.index()], 4..5);
        assert_eq!(raw[4].misc[0], 30);
        assert!(ranges[PrimMesh::Plane.index()].is_empty());
        assert!(ranges[PrimMesh::Cylinder.index()].is_empty());
    }
}
