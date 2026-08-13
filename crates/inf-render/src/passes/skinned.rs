//! GPU skinning pass (P11.1): the **additive** skinned variant of the rigid mesh
//! pass ([`super::mesh`]). It draws [`SkinnedInstance`]s — each a bind-space
//! [`SkinnedMeshData`] deformed in the vertex shader by a per-instance joint
//! **palette** — into the same MSAA scene targets, right after the rigid meshes.
//!
//! ## Byte-stability
//!
//! When `scene.skinned` is empty the pass emits **no commands** (an early return
//! in [`RenderNode::run`]), so every pre-P11 scene — and the unskinned pipeline —
//! renders identically. The rigid pipeline in [`super::mesh`] is untouched.
//!
//! ## Pipeline design
//!
//! * **Vertex layout** — buffer 0 is the mesh's [`SkinnedVertex`] stream
//!   (`pos @0`, `normal @1`, `joints @2` = `Uint32x4`, `weights @3` =
//!   `Float32x4`); buffer 1 is the per-instance [`InstanceRaw`] (model / normal
//!   matrix / color / pick id / pbr / emissive) at `@location(4..=14)`, selected
//!   per draw via `base_instance`.
//! * **Bind groups** — `@group(0)` view uniforms + `@group(1)` lights (shared
//!   with the rigid pass), `@group(2)` **reserved for the material seam** (an
//!   empty bind group today, mirroring the `inf-material` convention that binds
//!   material textures at group 2), and `@group(3)` the per-instance joint-matrix
//!   **storage buffer** (`array<mat4x4<f32>>`).
//! * **Batching (v1)** — one draw per skinned instance, each with its own palette
//!   bind group; palettes vary per instance so this is the simple correct
//!   baseline. A shared palette atlas + instanced draw (and a GPU palette compute
//!   pass) is the documented P15 optimization.

use glam::Mat3;
use inf_math::FloatingOrigin;

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::{InstanceRaw, LightsUniform};
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::{SkinnedInstance, SkinnedMeshData};

/// Vertex attributes for one [`SkinnedVertex`] (buffer 0).
///
/// **Five attributes, and the fifth is at 15** (P26.5). The instance block owns
/// `4..=14` here (one location further along than the rigid path, which starts
/// at 3), so the uv had exactly one address left: `@location(15)`, the last one
/// `Limits::default()`'s `max_vertex_attributes: 16` allows. That is the wall
/// `docs/memos/p26-5-vertex-streams.md` measures — this pipeline is now **full**,
/// and a tangent stream cannot join it without packing two channels into one
/// attribute or raising a limit the renderer has never raised.
const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 5] = [
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
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 24,
        shader_location: 15,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x4,
        offset: 32,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 48,
        shader_location: 3,
    },
];

/// Per-instance [`InstanceRaw`] attributes (buffer 1), at `@location(4..=14)` so
/// they clear the vertex attributes above.
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 11] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 0,
        shader_location: 4,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 16,
        shader_location: 5,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 32,
        shader_location: 6,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 48,
        shader_location: 7,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 64,
        shader_location: 8,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 80,
        shader_location: 9,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 96,
        shader_location: 10,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 112,
        shader_location: 11,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x4,
        offset: 128,
        shader_location: 12,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 144,
        shader_location: 13,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 160,
        shader_location: 14,
    },
];

/// `Limits::default()`'s ceiling on vertex attributes **across every buffer of
/// one pipeline** — the wall `docs/memos/p26-5-vertex-streams.md` measures.
///
/// The renderer has never raised a limit, so this is the number that decides
/// whether a tangent stream can join the skinned path. It is a `const` rather
/// than a sentence in a doc block because the memo's whole argument for
/// deferring the tangent to P28.2 rests on it, and the assertion below turns
/// "this pipeline is full" from a claim into a build failure (P26.5 audit).
const MAX_VERTEX_ATTRIBUTES: usize = 16;

/// **The skinned pipeline is at the wall, exactly.** Five vertex attributes plus
/// eleven instance ones is sixteen, and the uv took the last address there is
/// (`@location(15)`). Adding a seventeenth is a `wgpu` validation failure on
/// every adapter at `create_render_pipeline`; this fails the build instead, at
/// the file that would have to change.
const _: () = {
    assert!(VERTEX_ATTRIBUTES.len() + INSTANCE_ATTRIBUTES.len() == MAX_VERTEX_ATTRIBUTES);
    // …and no location is past the last one the limit allows, which is a
    // different fact from the count: `@location(15)` with a gap would satisfy
    // one and not the other.
    let mut i = 0;
    while i < VERTEX_ATTRIBUTES.len() {
        assert!((VERTEX_ATTRIBUTES[i].shader_location as usize) < MAX_VERTEX_ATTRIBUTES);
        i += 1;
    }
    let mut j = 0;
    while j < INSTANCE_ATTRIBUTES.len() {
        assert!((INSTANCE_ATTRIBUTES[j].shader_location as usize) < MAX_VERTEX_ATTRIBUTES);
        j += 1;
    }
};

fn vertex_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
    [
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<crate::scene::SkinnedVertex>() as u64,
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

/// Build an [`InstanceRaw`] for a skinned instance (origin-relative model matrix,
/// inverse-transpose normal matrix), mirroring `InstanceRaw::pack`.
fn instance_raw(origin: &FloatingOrigin, inst: &SkinnedInstance) -> InstanceRaw {
    let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
    let inv_scale = inst.scale.max(glam::Vec3::splat(1e-6)).recip();
    let nrm = Mat3::from_quat(inst.rotation) * Mat3::from_diagonal(inv_scale);
    let n = nrm.to_cols_array_2d();
    InstanceRaw {
        model: model.to_cols_array(),
        normal_mat: [
            n[0][0], n[0][1], n[0][2], 0.0, //
            n[1][0], n[1][1], n[1][2], 0.0, //
            n[2][0], n[2][1], n[2][2], 0.0,
        ],
        color: inst.color,
        // P26.3: the same three reserved words the rigid path uses, packed by
        // the same rule — so a skinned surface and a rigid one cannot disagree
        // about what "this instance samples nothing" looks like on the wire.
        misc: [inst.id, inst.vt.albedo, inst.vt.normal, inst.vt.orm],
        pbr: [inst.metallic, inst.roughness, 0.0, 0.0],
        emissive: [inst.emissive[0], inst.emissive[1], inst.emissive[2], 0.0],
    }
}

/// GPU buffers for one uploaded [`SkinnedMeshData`].
struct GpuSkinnedMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

/// One uploaded skinned instance: which mesh it draws + its palette bind group.
struct GpuSkinnedInstance {
    mesh: usize,
    palette_bg: wgpu::BindGroup,
}

pub struct SkinnedMeshNode {
    pipeline: wgpu::RenderPipeline,
    /// Wireframe (`PolygonMode::Line`) variant (R-P2), present only with the
    /// `POLYGON_MODE_LINE` feature; selected when the frame's view mode is
    /// [`ViewMode::Wireframe`](crate::renderer::ViewMode::Wireframe).
    pipeline_wire: Option<wgpu::RenderPipeline>,
    joints_bgl: wgpu::BindGroupLayout,
    /// AO + shadows + GI env bind at `@group(2)` (P13.3b; was the AO-only bind).
    env: super::EnvBinding,
    lights_buf: wgpu::Buffer,
    lights_bg: wgpu::BindGroup,
    /// Per `RenderScene::skinned_meshes` entry, its key into
    /// [`mesh_cache`](Self::mesh_cache) — so `SkinnedInstance::mesh` (an index into
    /// the scene's list) still resolves in one hop.
    meshes: Vec<usize>,
    /// Uploaded geometry by `Arc` identity (P18.3). Holds the `Arc` alongside the
    /// buffers so the pointer used as the key can never be recycled under a live
    /// entry; entries not referenced by a sync are dropped, which frees them.
    mesh_cache: std::collections::HashMap<usize, (std::sync::Arc<SkinnedMeshData>, GpuSkinnedMesh)>,
    instances: Vec<GpuSkinnedInstance>,
    /// One [`InstanceRaw`] per instance, selected per draw by `base_instance`.
    instance_buf: Option<wgpu::Buffer>,
    uploaded_version: Option<(u64, glam::DVec3)>,
    active: bool,
}

impl SkinnedMeshNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("skinned-mesh"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("skinned").into()),
            });

        // @group(1) lights (same layout/contents as the rigid mesh pass).
        let lights_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skinned-lights"),
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
            label: Some("skinned-lights"),
            size: std::mem::size_of::<LightsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinned-lights"),
            layout: &lights_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights_buf.as_entire_binding(),
            }],
        });

        // @group(2) AO + shadows + GI env bind (byte-stable when all are off).
        let env = super::EnvBinding::new(gpu);

        // @group(3) joint-matrix storage buffer.
        let joints_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skinned-joints"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("skinned-mesh"),
                bind_group_layouts: &[
                    Some(view_bgl),
                    Some(&lights_bgl),
                    Some(&env.bgl),
                    Some(&joints_bgl),
                ],
                immediate_size: 0,
            });

        // Fill + (feature-gated) R-P2 wireframe variants, identical but for the
        // primitive state — see the rigid `mesh` pass for the same shape.
        let make_pipeline = |label: &str, primitive: wgpu::PrimitiveState| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
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
                    primitive,
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
                })
        };

        let pipeline = make_pipeline(
            "skinned-mesh",
            wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
        );
        let pipeline_wire = gpu
            .device
            .features()
            .contains(wgpu::Features::POLYGON_MODE_LINE)
            .then(|| {
                make_pipeline(
                    "skinned-mesh-wire",
                    wgpu::PrimitiveState {
                        polygon_mode: wgpu::PolygonMode::Line,
                        ..Default::default()
                    },
                )
            });

        Self {
            pipeline,
            pipeline_wire,
            joints_bgl,
            env,
            lights_buf,
            lights_bg,
            meshes: Vec::new(),
            mesh_cache: std::collections::HashMap::new(),
            instances: Vec::new(),
            instance_buf: None,
            uploaded_version: None,
            active: false,
        }
    }

    /// Re-upload geometry, instance data, and palettes when the scene changed or
    /// the floating origin rebased.
    fn sync(&mut self, gpu: &GpuContext, frame: &FrameData) {
        let key = (frame.scene.version, frame.view.origin.origin());
        if self.uploaded_version == Some(key) {
            return;
        }
        self.uploaded_version = Some(key);
        self.active = !frame.scene.skinned.is_empty();

        // Lights (same projection as the rigid pass).
        let lights = LightsUniform::from_scene(frame.scene, &frame.view.origin);
        gpu.queue
            .write_buffer(&self.lights_buf, 0, bytemuck::bytes_of(&lights));

        // ── Mesh geometry, keyed on IDENTITY rather than on the frame (P18.3) ──
        //
        // Bind-space geometry does not change when a pose does, but this used to
        // re-upload every vertex of every skinned mesh whenever `scene.version`
        // moved — which for an editor is every gizmo tick of an unrelated entity.
        // The scene now shares each buffer as an `Arc`, so "is this the same
        // geometry I already uploaded?" is a pointer comparison. The cache holds
        // the `Arc` itself, which is what makes the pointer a sound key: the
        // allocation cannot be freed and reused under a live entry.
        //
        // Palettes below are deliberately NOT cached — they are the per-frame part.
        let mut cache = std::mem::take(&mut self.mesh_cache);
        self.meshes = frame
            .scene
            .skinned_meshes
            .iter()
            .map(|m| {
                let key = std::sync::Arc::as_ptr(m) as usize;
                let entry = cache
                    .remove(&key)
                    .unwrap_or_else(|| (m.clone(), upload_mesh(gpu, m)));
                self.mesh_cache.insert(key, entry);
                key
            })
            .collect();
        // Whatever `cache` still holds was not referenced this frame — dropped
        // here, which releases its GPU buffers.

        // Instance transforms (one InstanceRaw each, selected via base_instance).
        let raws: Vec<InstanceRaw> = frame
            .scene
            .skinned
            .iter()
            .map(|i| instance_raw(&frame.view.origin, i))
            .collect();
        self.instance_buf = (!raws.is_empty()).then(|| {
            let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("skinned-instances"),
                size: std::mem::size_of_val(raws.as_slice()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&raws));
            buf
        });

        // Per-instance palette storage buffers + bind groups.
        self.instances = frame
            .scene
            .skinned
            .iter()
            .map(|inst| {
                // Non-empty so the storage buffer is never zero-sized.
                let palette: Vec<[f32; 16]> = if inst.palette.is_empty() {
                    vec![glam::Mat4::IDENTITY.to_cols_array()]
                } else {
                    inst.palette.iter().map(|m| m.to_cols_array()).collect()
                };
                let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("skinned-palette"),
                    size: std::mem::size_of_val(palette.as_slice()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue
                    .write_buffer(&buf, 0, bytemuck::cast_slice(&palette));
                let palette_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("skinned-palette"),
                    layout: &self.joints_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    }],
                });
                // The buffer is kept alive by the bind group.
                GpuSkinnedInstance {
                    mesh: inst.mesh,
                    palette_bg,
                }
            })
            .collect();
    }
}

fn upload_mesh(gpu: &GpuContext, mesh: &SkinnedMeshData) -> GpuSkinnedMesh {
    let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("skinned-vertices"),
        size: std::mem::size_of_val(mesh.vertices.as_slice()).max(4) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&vertices, 0, bytemuck::cast_slice(&mesh.vertices));
    let indices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("skinned-indices"),
        size: std::mem::size_of_val(mesh.indices.as_slice()).max(4) as u64,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&indices, 0, bytemuck::cast_slice(&mesh.indices));
    GpuSkinnedMesh {
        vertices,
        indices,
        index_count: mesh.indices.len() as u32,
    }
}

impl RenderNode for SkinnedMeshNode {
    fn name(&self) -> &'static str {
        "skinned-mesh"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        self.sync(gpu, frame);
        if !self.active {
            return;
        }
        let Some(instance_buf) = self.instance_buf.as_ref() else {
            return;
        };
        let env_bg = self.env.bind_group(gpu, frame).clone();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("skinned-mesh"),
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
        // Wireframe view mode (R-P2) selects the line-raster variant when present.
        let pipeline = match &self.pipeline_wire {
            Some(wire) if frame.view_mode.wireframe() => wire,
            _ => &self.pipeline,
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_bind_group(1, &self.lights_bg, &[]);
        pass.set_bind_group(2, &env_bg, &[]);
        pass.set_vertex_buffer(1, instance_buf.slice(..));

        for (i, inst) in self.instances.iter().enumerate() {
            let Some(gpu_mesh) = self
                .meshes
                .get(inst.mesh)
                .and_then(|key| self.mesh_cache.get(key))
                .map(|(_, m)| m)
            else {
                continue;
            };
            if gpu_mesh.index_count == 0 {
                continue;
            }
            pass.set_bind_group(3, &inst.palette_bg, &[]);
            pass.set_vertex_buffer(0, gpu_mesh.vertices.slice(..));
            pass.set_index_buffer(gpu_mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            let base = i as u32;
            pass.draw_indexed(0..gpu_mesh.index_count, 0, base..base + 1);
        }
    }
}
