//! GPU-driven virtualized-geometry (meshlet) render path (P13.1b).
//!
//! The portable, mesh-shader-free Nanite-class runtime: a **compute cull+LOD**
//! pass selects the visible meshlets of every placed [`VgeomInstance`], then a
//! single **vertex-pulled `draw_indirect`** rasterizes them.
//!
//! # Pipeline (what happens each frame, per referenced vmesh asset)
//!
//! 1. **Upload / cache** — a [`VgeomGpu`] holds one asset's geometry in storage
//!    buffers (positions, meshlet descriptors, meshlet→vertex indices, packed
//!    meshlet→triangle bytes), built once and cached by asset id
//!    ([`RenderScene::vgeom_assets`]). Per-instance transforms are re-packed each
//!    frame (the LOD threshold depends on the live camera distance).
//! 2. **Cull compute** (`vgeom_cull.wgsl`) — one thread per (instance, meshlet):
//!    the LOD cut, frustum-sphere, backface-cone, and optional HZB tests; each
//!    survivor is appended to a `visible` list via an atomic that **is** the
//!    indirect draw's `instance_count`.
//! 3. **Raster** (`vgeom_mesh.wgsl`) — one non-indexed `draw_indirect` of
//!    `visible_count` instances, each drawing `max_tri*3` vertices pulled from the
//!    storage buffers (unused triangles collapse to a degenerate off-screen point).
//!    Shading is the same metallic-roughness PBR as the rigid mesh pass (shared
//!    lights `@group(1)` + AO `@group(2)`), into the same MSAA scene targets.
//!
//! # The vertex-pulling draw (exact buffers / indices)
//!
//! There are **no vertex buffers**. The draw is `draw_indirect` with args
//! `[vertex_count = max_tri*3, instance_count = visible, first_vertex = 0,
//! first_instance = 0]` (non-indexed — equivalent to the classic "shared
//! `0..max_tri*3` identity index buffer + `draw_indexed_indirect`" technique but
//! without the redundant identity index buffer). In the vertex shader:
//! `instance_index → visible[i] → (global_instance, meshlet_id)`;
//! `vertex_index → triangle = vertex_index/3, corner = vertex_index%3`; the
//! triangle's local index is read from the packed `meshlet_triangles` bytes,
//! resolved through `meshlet_vertices` to a global vertex, and transformed by the
//! instance model matrix. Cost of the fixed vertex count: up to
//! `(max_tri − triangle_count)*3` degenerate vertices per meshlet (a well-known
//! trade for a single indirect draw with no per-meshlet draw args).
//!
//! # Screen-space error projection (the LOD cut)
//!
//! We project the requested **pixel tolerance** to a *single per-instance
//! object-space scalar threshold* `t`, then apply the branchless cut
//! `error ≤ t < parent_error` per meshlet — which is **bit-identical** to
//! [`VgeomMesh::select(t)`](inf_vgeom::VgeomMesh::select). Using one `t` per mesh
//! instance (not per meshlet) is what preserves the DAG *cut invariant* (the
//! half-open `[error, parent_error)` intervals tile `[0,∞)` only against a single
//! threshold). For a perspective view of an instance whose bounding sphere is at
//! distance `d` (to the sphere surface) with instance scale `s`:
//!
//! ```text
//!   focal_px = screen_height / (2·tan(fov_y/2))
//!   screen_error_px(object_error) = object_error · s · focal_px / d
//!   ⇒  t = pixel_tol · d / (focal_px · s)          // solve screen_error == tol
//! ```
//!
//! `t` is monotonic (a positive scaling) in `object_error`, so the cut interval
//! test survives the projection. Farther instances ⇒ larger `d` ⇒ larger `t` ⇒
//! coarser meshlets selected ⇒ fewer visible meshlets (the LOD proof).
//!
//! # Determinism
//!
//! The atomic append order varies (so the *draw order* varies), but the visible
//! **set** is a pure function of (scene, view, settings). Meshlet raster is opaque
//! and depth-tested, so the resulting image is order-independent — byte-stable
//! across renders of the same frame ([`crate::golden`] asserts it).
//!
//! # Occlusion (v1 vs the full two-pass)
//!
//! [`VgeomSettings::occlusion`] adds an HZB depth test in the cull compute.
//! **v1 (this pass):** the HZB is built (`vgeom_hzb.wgsl`, reverse-Z min-depth
//! pyramid) from the **current frame's classic depth prepass**, so meshlets are
//! occluded by the classic geometry drawn this frame. It is **OFF by default** —
//! the tested/golden path is frustum + cone + LOD, which matches the CPU reference
//! exactly. The **full two-pass** technique (draw last-frame-visible meshlets →
//! build HZB from *their* depth → test the rest and draw newly-visible) is the
//! documented next step; the roadmap allows this sequencing.

use std::collections::BTreeMap;

use glam::{Mat3, Mat4, Vec3, Vec4};
use inf_math::FloatingOrigin;
use inf_vgeom::VgeomMesh;

use crate::camera::{RenderView, DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::LightsUniform;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::VgeomInstance;
use crate::settings::VgeomSettings;

// ── GPU-side layouts (must match the WGSL structs) ───────────────────────────

/// One meshlet descriptor as the cull/raster shaders read it (64 bytes, 16-byte
/// aligned lanes). Mirrors `struct Meshlet` in `vgeom_cull.wgsl`/`vgeom_mesh.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshletGpu {
    center: [f32; 3],
    radius: f32,
    cone_axis: [f32; 3],
    cone_cutoff: f32,
    vertex_offset: u32,
    triangle_offset: u32,
    vertex_count: u32,
    triangle_count: u32,
    error: f32,
    parent_error: f32,
    lod_level: u32,
    _pad: u32,
}

/// One placed instance as the shaders read it (176 bytes). Mirrors
/// `struct Instance`. `threshold` is the precomputed per-instance LOD scalar.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VgeomInstanceGpu {
    model: [f32; 16],
    n0: [f32; 4],
    n1: [f32; 4],
    n2: [f32; 4],
    color: [f32; 4],
    emissive: [f32; 4],
    threshold: f32,
    metallic: f32,
    roughness: f32,
    max_scale: f32,
    pick_id: u32,
    _p: [u32; 3],
}

/// Cull uniform block. Mirrors `struct CullParams`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullParamsGpu {
    view_proj: [f32; 16],
    frustum: [[f32; 4]; 6],
    eye: [f32; 4],
    counts: [u32; 4],
    hzb: [f32; 4],
}

/// Per-meshlet debug flag uniform (`struct VgeomFlags`, `@group(3) @binding(6)`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlagsGpu {
    flags: [u32; 4],
}

const FLAG_FRUSTUM: u32 = 1;
const FLAG_CONE: u32 = 2;
const FLAG_OCCLUSION: u32 = 4;

// ── Pure math (unit-tested without a GPU) ────────────────────────────────────

/// The per-instance object-space LOD threshold `t` for a bounding sphere at
/// render-local `center` with surface distance to `eye`, instance scale `max_scale`
/// (largest axis), under `view`, targeting `pixel_error` px. See the module docs.
pub fn lod_threshold(
    eye: Vec3,
    center: Vec3,
    radius_world: f32,
    max_scale: f32,
    view: &RenderView,
    pixel_error: f32,
) -> f32 {
    let s = max_scale.max(1e-6);
    match view.ortho {
        None => {
            let d = ((eye - center).length() - radius_world).max(1e-3);
            let focal = view.height.max(1) as f32 / (2.0 * (view.fov_y * 0.5).tan()).max(1e-6);
            pixel_error * d / (focal * s)
        }
        Some(o) => {
            // Parallel projection: world units per pixel is distance-independent.
            let wpp = 2.0 * o.half_height / view.height.max(1) as f32;
            pixel_error * wpp / s
        }
    }
}

/// Six normalized frustum planes (render-local clip space, wgpu NDC z ∈ [0,1])
/// extracted from a render-local `view_proj` (Gribb–Hartmann). Order: left,
/// right, bottom, top, near, far. A plane with a (near-)degenerate normal (the far
/// plane of a reverse-infinite projection) is returned as all-zeros → the cull
/// treats it as "always inside".
pub fn frustum_planes(view_proj: Mat4) -> [Vec4; 6] {
    let m = view_proj.to_cols_array(); // column-major: element (row r, col c) = m[c*4+r]
    let row = |r: usize| Vec4::new(m[r], m[4 + r], m[8 + r], m[12 + r]);
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    let raw = [
        r3 + r0, // left
        r3 - r0, // right
        r3 + r1, // bottom
        r3 - r1, // top
        r2,      // near  (wgpu/D3D: z ≥ 0)
        r3 - r2, // far
    ];
    raw.map(|p| {
        let n = Vec3::new(p.x, p.y, p.z).length();
        if n > 1e-6 {
            p / n
        } else {
            Vec4::ZERO
        }
    })
}

/// Whether a world sphere is entirely outside any frustum plane.
fn outside_frustum(center: Vec3, radius: f32, planes: &[Vec4; 6]) -> bool {
    planes.iter().any(|pl| {
        let n = Vec3::new(pl.x, pl.y, pl.z);
        n != Vec3::ZERO && n.dot(center) + pl.w < -radius
    })
}

/// Bit flags matching the cull compute (frustum | cone | occlusion).
pub fn cull_flags(settings: &VgeomSettings) -> u32 {
    let mut f = 0;
    if settings.frustum_cull {
        f |= FLAG_FRUSTUM;
    }
    if settings.cone_cull {
        f |= FLAG_CONE;
    }
    if settings.occlusion {
        f |= FLAG_OCCLUSION;
    }
    f
}

/// The **CPU reference** visible meshlet set for a single instance: the identical
/// LOD cut + frustum + cone filters the GPU cull compute applies, mirrored on the
/// CPU for the parity gate. Returns sorted meshlet indices. Occlusion is never
/// part of the reference (the parity scene is occlusion-free).
#[allow(clippy::too_many_arguments)]
pub fn cpu_visible_set(
    mesh: &VgeomMesh,
    model: Mat4,
    normal_mat: Mat3,
    eye: Vec3,
    threshold: f32,
    max_scale: f32,
    planes: &[Vec4; 6],
    flags: u32,
) -> Vec<u32> {
    let mut out = Vec::new();
    for (i, m) in mesh.meshlets.iter().enumerate() {
        // 1. LOD cut (== VgeomMesh::select semantics).
        if !(m.error <= threshold && threshold < m.parent_error) {
            continue;
        }
        let center = model.transform_point3(Vec3::from(m.center));
        let radius = m.radius * max_scale;
        // 2. Frustum.
        if (flags & FLAG_FRUSTUM) != 0 && outside_frustum(center, radius, planes) {
            continue;
        }
        // 3. Backface cone.
        if (flags & FLAG_CONE) != 0 && m.cone_cutoff < 1.0 {
            let axis = (normal_mat * Vec3::from(m.cone_axis)).normalize_or_zero();
            let vd = (center - eye).normalize_or_zero();
            if vd.dot(axis) >= m.cone_cutoff {
                continue;
            }
        }
        out.push(i as u32);
    }
    out
}

// ── Per-instance packing (shared by the node + the standalone readback) ──────

fn max_scale_of(scale: Vec3) -> f32 {
    scale.abs().max_element().max(1e-6)
}

/// Pack one instance's GPU record (transform, normal matrix, material, and the
/// precomputed LOD threshold from the live view).
fn pack_instance(
    origin: &FloatingOrigin,
    view: &RenderView,
    mesh: &VgeomMesh,
    inst: &VgeomInstance,
    pixel_error: f32,
) -> VgeomInstanceGpu {
    let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
    let max_scale = max_scale_of(inst.scale);
    let inv_scale = inst.scale.max(Vec3::splat(1e-6)).recip();
    let nrm = Mat3::from_quat(inst.rotation) * Mat3::from_diagonal(inv_scale);
    let c = nrm.to_cols_array_2d();

    let eye = view.eye_local();
    let center_world = model.transform_point3(Vec3::from(mesh.center));
    let radius_world = mesh.radius * max_scale;
    let threshold = lod_threshold(
        eye,
        center_world,
        radius_world,
        max_scale,
        view,
        pixel_error,
    );

    VgeomInstanceGpu {
        model: model.to_cols_array(),
        n0: [c[0][0], c[0][1], c[0][2], 0.0],
        n1: [c[1][0], c[1][1], c[1][2], 0.0],
        n2: [c[2][0], c[2][1], c[2][2], 0.0],
        color: inst.color,
        emissive: [inst.emissive[0], inst.emissive[1], inst.emissive[2], 0.0],
        threshold,
        metallic: inst.metallic,
        roughness: inst.roughness,
        max_scale,
        pick_id: inst.id,
        _p: [0; 3],
    }
}

fn cull_params(
    view: &RenderView,
    meshlet_count: u32,
    instance_count: u32,
    settings: &VgeomSettings,
    hzb: [f32; 4],
) -> CullParamsGpu {
    let vp = view.view_proj();
    let planes = frustum_planes(vp);
    CullParamsGpu {
        view_proj: vp.to_cols_array(),
        frustum: planes.map(|p| p.to_array()),
        eye: view.eye_local().extend(0.0).to_array(),
        counts: [
            instance_count * meshlet_count,
            meshlet_count,
            cull_flags(settings),
            0,
        ],
        hzb,
    }
}

// ── Per-asset static geometry cache ──────────────────────────────────────────

/// One vmesh asset's geometry, uploaded to GPU storage buffers once and cached by
/// asset id.
struct VgeomGpu {
    positions: wgpu::Buffer,
    meshlets: wgpu::Buffer,
    meshlet_verts: wgpu::Buffer,
    meshlet_tris: wgpu::Buffer,
    meshlet_count: u32,
    /// Largest `triangle_count` across the meshlets → the indirect draw's fixed
    /// `vertex_count = max_tri * 3`.
    max_tri: u32,
}

/// A storage buffer holding `data` (padded to ≥16 bytes so a small/empty payload
/// still creates a valid buffer).
fn storage_buffer(gpu: &GpuContext, label: &str, data: &[u8]) -> wgpu::Buffer {
    let size = (data.len().max(16)).next_multiple_of(4) as u64;
    let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        gpu.queue.write_buffer(&buf, 0, data);
    }
    buf
}

impl VgeomGpu {
    fn build(gpu: &GpuContext, mesh: &VgeomMesh) -> Self {
        // Positions: VgeomVertex is 8 f32 (pos.xyz, normal.xyz, uv.xy), read as a
        // flat `array<f32>` (8 per vertex) — sidesteps vec3 storage alignment.
        let positions = storage_buffer(
            gpu,
            "vgeom-positions",
            bytemuck::cast_slice::<_, u8>(&mesh.vertices),
        );

        let meshlets_gpu: Vec<MeshletGpu> = mesh
            .meshlets
            .iter()
            .map(|m| MeshletGpu {
                center: m.center,
                radius: m.radius,
                cone_axis: m.cone_axis,
                cone_cutoff: m.cone_cutoff,
                vertex_offset: m.vertex_offset,
                triangle_offset: m.triangle_offset,
                vertex_count: m.vertex_count,
                triangle_count: m.triangle_count,
                error: m.error,
                parent_error: m.parent_error,
                lod_level: m.lod_level as u32,
                _pad: 0,
            })
            .collect();
        let meshlets = storage_buffer(gpu, "vgeom-meshlets", bytemuck::cast_slice(&meshlets_gpu));
        let meshlet_verts = storage_buffer(
            gpu,
            "vgeom-meshlet-verts",
            bytemuck::cast_slice(&mesh.meshlet_vertices),
        );

        // Pack the meshlet triangle bytes (u8) into u32 words for GPU indexing.
        let tris = &mesh.meshlet_triangles;
        let mut packed = vec![0u32; tris.len().div_ceil(4).max(1)];
        for (i, &b) in tris.iter().enumerate() {
            packed[i / 4] |= (b as u32) << ((i % 4) * 8);
        }
        let meshlet_tris = storage_buffer(gpu, "vgeom-meshlet-tris", bytemuck::cast_slice(&packed));

        let max_tri = mesh
            .meshlets
            .iter()
            .map(|m| m.triangle_count)
            .max()
            .unwrap_or(0);

        Self {
            positions,
            meshlets,
            meshlet_verts,
            meshlet_tris,
            meshlet_count: mesh.meshlets.len() as u32,
            max_tri,
        }
    }
}

// ── The cull compute pipeline (shared by node + standalone readback) ─────────

struct CullPipeline {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl CullPipeline {
    fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vgeom-cull"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vgeom_cull.wgsl").into()),
            });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |ro| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vgeom-cull"),
                entries: &[
                    entry(
                        0,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    ),
                    entry(1, storage(true)),
                    entry(2, storage(true)),
                    entry(3, storage(false)),
                    entry(4, storage(false)),
                    entry(
                        5,
                        wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                    ),
                ],
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("vgeom-cull"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("vgeom-cull"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("cs_cull"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self { pipeline, bgl }
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_group(
        &self,
        gpu: &GpuContext,
        params: &wgpu::Buffer,
        geom: &VgeomGpu,
        instances: &wgpu::Buffer,
        visible: &wgpu::Buffer,
        draw_args: &wgpu::Buffer,
        hzb: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vgeom-cull"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: geom.meshlets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: visible.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: draw_args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(hzb),
                },
            ],
        })
    }
}

/// A 1×1 dummy HZB texture (bound when occlusion is off so the cull bind group is
/// always complete).
fn dummy_hzb(gpu: &GpuContext) -> wgpu::TextureView {
    gpu.device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("vgeom-hzb-dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

// ── Standalone cull readback (tests + player activation check) ───────────────

/// Run **only** the cull compute for `instances` of `mesh` under `view`+`settings`
/// and read back the visible `(instance, meshlet)` pairs, **sorted** (the atomic
/// append order is nondeterministic; the set is not). Occlusion is forced off
/// (no depth context). This is the exact machinery the render node uses, exposed
/// for the CPU-vs-GPU parity gate and the player's activation test.
pub fn cull_visible(
    gpu: &GpuContext,
    mesh: &VgeomMesh,
    instances: &[VgeomInstance],
    view: &RenderView,
    settings: &VgeomSettings,
) -> Vec<[u32; 2]> {
    if mesh.meshlets.is_empty() || instances.is_empty() {
        return Vec::new();
    }
    let mut settings = *settings;
    settings.occlusion = false;

    let geom = VgeomGpu::build(gpu, mesh);
    let origin = view.origin;
    let packed: Vec<VgeomInstanceGpu> = instances
        .iter()
        .map(|i| pack_instance(&origin, view, mesh, i, settings.pixel_error))
        .collect();
    let inst_buf = storage_buffer(gpu, "vgeom-inst", bytemuck::cast_slice(&packed));

    let total = packed.len() as u32 * geom.meshlet_count;
    let visible = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vgeom-visible"),
        size: (total as u64 * 8).max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let draw_args = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vgeom-args"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(
        &draw_args,
        0,
        bytemuck::cast_slice(&[geom.max_tri * 3, 0u32, 0u32, 0u32]),
    );

    let params = cull_params(
        view,
        geom.meshlet_count,
        packed.len() as u32,
        &settings,
        [1.0, 1.0, 1.0, 0.0],
    );
    let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vgeom-cull-params"),
        size: std::mem::size_of::<CullParamsGpu>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let hzb = dummy_hzb(gpu);
    let cull = CullPipeline::new(gpu);
    let bg = cull.bind_group(
        gpu,
        &params_buf,
        &geom,
        &inst_buf,
        &visible,
        &draw_args,
        &hzb,
    );

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vgeom-cull-readback"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("vgeom-cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&cull.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(total.div_ceil(64).max(1), 1, 1);
    }
    // Copy args + visible to a mappable buffer.
    let args_rb = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vgeom-args-rb"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let vis_rb = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vgeom-visible-rb"),
        size: (total as u64 * 8).max(16),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&draw_args, 0, &args_rb, 0, 16);
    encoder.copy_buffer_to_buffer(&visible, 0, &vis_rb, 0, (total as u64 * 8).max(16));
    gpu.queue.submit([encoder.finish()]);

    let count = map_u32(gpu, &args_rb)[1] as usize;
    let vis = map_u32(gpu, &vis_rb);
    let mut out: Vec<[u32; 2]> = (0..count.min(total as usize))
        .map(|i| [vis[i * 2], vis[i * 2 + 1]])
        .collect();
    out.sort_unstable();
    out
}

/// Blocking map of a `MAP_READ` buffer into a `Vec<u32>`.
fn map_u32(gpu: &GpuContext, buf: &wgpu::Buffer) -> Vec<u32> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    let _ = rx.recv();
    let data = slice.get_mapped_range().expect("map vgeom readback buffer");
    let out = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
    drop(data);
    buf.unmap();
    out
}

// ── Per-asset dynamic draw state (owned by the node) ─────────────────────────

struct AssetDraw {
    instances: wgpu::Buffer,
    instance_cap: u32,
    visible: wgpu::Buffer,
    visible_cap: u32,
    draw_args: wgpu::Buffer,
    params: wgpu::Buffer,
    flags: wgpu::Buffer,
}

impl AssetDraw {
    fn new(gpu: &GpuContext) -> Self {
        let mk = |label, size, usage| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            instances: mk(
                "vgeom-instances",
                256,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            instance_cap: 0,
            visible: mk("vgeom-visible", 256, wgpu::BufferUsages::STORAGE),
            visible_cap: 0,
            draw_args: mk(
                "vgeom-args",
                16,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
            ),
            params: mk(
                "vgeom-cull-params",
                std::mem::size_of::<CullParamsGpu>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            flags: mk(
                "vgeom-flags",
                std::mem::size_of::<FlagsGpu>() as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
        }
    }

    /// Ensure the instance + visible buffers hold `instance_count` instances of a
    /// `meshlet_count`-meshlet asset. Returns whether any buffer was recreated
    /// (⇒ bind groups must be rebuilt).
    fn ensure(&mut self, gpu: &GpuContext, instance_count: u32, meshlet_count: u32) -> bool {
        let mut rebuilt = false;
        if instance_count > self.instance_cap {
            let cap = instance_count.next_power_of_two().max(4);
            self.instances = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vgeom-instances"),
                size: cap as u64 * std::mem::size_of::<VgeomInstanceGpu>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_cap = cap;
            rebuilt = true;
        }
        let total = instance_count * meshlet_count;
        if total > self.visible_cap {
            let cap = total.next_power_of_two().max(4);
            self.visible = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vgeom-visible"),
                size: cap as u64 * 8,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.visible_cap = cap;
            rebuilt = true;
        }
        rebuilt
    }
}

// ── The render node ──────────────────────────────────────────────────────────

/// The virtualized-geometry render node (P13.1b): cull compute + vertex-pulled
/// indirect raster. A no-op unless [`VgeomSettings::enabled`] and the scene
/// carries vmesh assets + instances — so the classic path is byte-stable.
pub struct VgeomNode {
    cull: CullPipeline,
    raster: wgpu::RenderPipeline,
    raster_bgl: wgpu::BindGroupLayout,
    lights_buf: wgpu::Buffer,
    lights_bg: wgpu::BindGroup,
    ao: super::AoBinding,
    dummy_hzb: wgpu::TextureView,
    geom_cache: BTreeMap<u128, VgeomGpu>,
    draws: BTreeMap<u128, AssetDraw>,
    hzb: HzbChain,
}

impl VgeomNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let cull = CullPipeline::new(gpu);

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vgeom-mesh"),
                source: wgpu::ShaderSource::Wgsl(
                    super::scene_shader(include_str!("../shaders/vgeom_mesh.wgsl")).into(),
                ),
            });

        // Lights (@group(1)) — shared model with the rigid mesh pass.
        let lights_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vgeom-lights"),
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
            label: Some("vgeom-lights"),
            size: std::mem::size_of::<LightsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vgeom-lights"),
            layout: &lights_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights_buf.as_entire_binding(),
            }],
        });

        let ao = super::AoBinding::new(gpu);

        // Group 3: the vgeom storage buffers (vertex-visible) + flags uniform.
        let vs = wgpu::ShaderStages::VERTEX;
        let ro_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let raster_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vgeom-raster"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: vs,
                        ty: ro_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
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
                label: Some("vgeom-raster"),
                bind_group_layouts: &[
                    Some(view_bgl),
                    Some(&lights_bgl),
                    Some(&ao.bgl),
                    Some(&raster_bgl),
                ],
                immediate_size: 0,
            });
        let raster = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("vgeom-raster"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[], // pure vertex pulling
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
            cull,
            raster,
            raster_bgl,
            lights_buf,
            lights_bg,
            ao,
            dummy_hzb: dummy_hzb(gpu),
            geom_cache: BTreeMap::new(),
            draws: BTreeMap::new(),
            hzb: HzbChain::new(gpu),
        }
    }
}

impl RenderNode for VgeomNode {
    fn name(&self) -> &'static str {
        "vgeom"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let settings = &frame.settings.vgeom;
        if !settings.enabled
            || frame.scene.vgeom_instances.is_empty()
            || frame.scene.vgeom_assets.is_empty()
        {
            return;
        }

        // Lights (shared with the rigid pass).
        let lights = LightsUniform::from_scene(frame.scene, &frame.view.origin);
        gpu.queue
            .write_buffer(&self.lights_buf, 0, bytemuck::bytes_of(&lights));

        // Build/cache each referenced asset's geometry.
        for asset in &frame.scene.vgeom_assets {
            self.geom_cache
                .entry(asset.id)
                .or_insert_with(|| VgeomGpu::build(gpu, &asset.mesh));
        }
        // Look up the mesh (for the whole-mesh bound in threshold packing).
        let mesh_of: BTreeMap<u128, &VgeomMesh> = frame
            .scene
            .vgeom_assets
            .iter()
            .map(|a| (a.id, a.mesh.as_ref()))
            .collect();

        // Group instances by asset (deterministic asset order).
        let mut by_asset: BTreeMap<u128, Vec<&VgeomInstance>> = BTreeMap::new();
        for inst in &frame.scene.vgeom_instances {
            by_asset.entry(inst.asset).or_default().push(inst);
        }

        // Optional HZB from the current depth prepass (v1 occlusion).
        let occlusion = settings.occlusion;
        if occlusion {
            self.hzb.build(gpu, encoder, frame);
        }
        let hzb_dims = if occlusion {
            self.hzb
                .dims()
                .map(|(w, h, mips)| [mips as f32, w as f32, h as f32, 0.0])
                .unwrap_or([1.0, 1.0, 1.0, 0.0])
        } else {
            [1.0, 1.0, 1.0, 0.0]
        };

        let ao_bg = self.ao.bind_group(gpu, frame).clone();

        let flags = FlagsGpu {
            flags: [settings.debug_meshlets as u32, 0, 0, 0],
        };
        let origin = frame.view.origin;

        for (asset_id, insts) in &by_asset {
            let Some(geom) = self.geom_cache.get(asset_id) else {
                continue;
            };
            let Some(mesh) = mesh_of.get(asset_id) else {
                continue;
            };
            if geom.meshlet_count == 0 || geom.max_tri == 0 {
                continue;
            }
            let instance_count = insts.len() as u32;

            let draw = self
                .draws
                .entry(*asset_id)
                .or_insert_with(|| AssetDraw::new(gpu));
            draw.ensure(gpu, instance_count, geom.meshlet_count);

            // Pack instances (per-frame; the LOD threshold tracks the camera).
            let packed: Vec<VgeomInstanceGpu> = insts
                .iter()
                .map(|i| pack_instance(&origin, frame.view, mesh, i, settings.pixel_error))
                .collect();
            gpu.queue
                .write_buffer(&draw.instances, 0, bytemuck::cast_slice(&packed));

            // Reset draw args: vertex_count = max_tri*3, instance_count = 0.
            gpu.queue.write_buffer(
                &draw.draw_args,
                0,
                bytemuck::cast_slice(&[geom.max_tri * 3, 0u32, 0u32, 0u32]),
            );

            // Cull params + flags.
            let params = cull_params(
                frame.view,
                geom.meshlet_count,
                instance_count,
                settings,
                hzb_dims,
            );
            gpu.queue
                .write_buffer(&draw.params, 0, bytemuck::bytes_of(&params));
            gpu.queue
                .write_buffer(&draw.flags, 0, bytemuck::bytes_of(&flags));

            let hzb = if occlusion {
                self.hzb.full_view().unwrap_or(&self.dummy_hzb)
            } else {
                &self.dummy_hzb
            };
            let cull_bg = self.cull.bind_group(
                gpu,
                &draw.params,
                geom,
                &draw.instances,
                &draw.visible,
                &draw.draw_args,
                hzb,
            );

            // Cull compute.
            let total = instance_count * geom.meshlet_count;
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("vgeom-cull"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.cull.pipeline);
                pass.set_bind_group(0, &cull_bg, &[]);
                pass.dispatch_workgroups(total.div_ceil(64).max(1), 1, 1);
            }

            // Raster: vertex-pulled indirect draw into the MSAA scene targets.
            let raster_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vgeom-raster"),
                layout: &self.raster_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: geom.positions.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: geom.meshlets.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: geom.meshlet_verts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: geom.meshlet_tris.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: draw.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: draw.visible.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: draw.flags.as_entire_binding(),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vgeom-raster"),
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
            pass.set_pipeline(&self.raster);
            pass.set_bind_group(0, frame.view_bg, &[]);
            pass.set_bind_group(1, &self.lights_bg, &[]);
            pass.set_bind_group(2, &ao_bg, &[]);
            pass.set_bind_group(3, &raster_bg, &[]);
            pass.draw_indirect(&draw.draw_args, 0);
        }
    }
}

// ── HZB (v1): reverse-Z min-depth pyramid from the depth prepass ─────────────

/// A hierarchical depth pyramid built by compute from the single-sample depth
/// prepass. Only allocated/used when [`VgeomSettings::occlusion`] is on (off by
/// default), so the tested path never touches it.
struct HzbChain {
    copy_pipeline: wgpu::ComputePipeline,
    copy_bgl: wgpu::BindGroupLayout,
    down_pipeline: wgpu::ComputePipeline,
    down_bgl: wgpu::BindGroupLayout,
    texture: Option<wgpu::Texture>,
    mip_views: Vec<wgpu::TextureView>,
    full_view: Option<wgpu::TextureView>,
    size: (u32, u32),
    generation: u64,
}

impl HzbChain {
    fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vgeom-hzb"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vgeom_hzb.wgsl").into()),
            });
        let copy_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vgeom-hzb-copy"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let down_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vgeom-hzb-down"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        let mk = |label, bgl: &wgpu::BindGroupLayout, entry: &str| {
            let layout = gpu
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(label),
                    bind_group_layouts: &[Some(bgl)],
                    immediate_size: 0,
                });
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let copy_pipeline = mk("vgeom-hzb-copy", &copy_bgl, "cs_copy");
        let down_pipeline = mk("vgeom-hzb-down", &down_bgl, "cs_down");
        Self {
            copy_pipeline,
            copy_bgl,
            down_pipeline,
            down_bgl,
            texture: None,
            mip_views: Vec::new(),
            full_view: None,
            size: (0, 0),
            generation: u64::MAX,
        }
    }

    fn dims(&self) -> Option<(u32, u32, u32)> {
        self.texture
            .as_ref()
            .map(|_| (self.size.0, self.size.1, self.mip_views.len() as u32))
    }

    fn ensure(&mut self, gpu: &GpuContext, size: (u32, u32)) {
        if self.size == size && self.texture.is_some() {
            return;
        }
        let (w, h) = (size.0.max(1), size.1.max(1));
        let mips = (32 - (w.max(h)).leading_zeros()).max(1);
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vgeom-hzb"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        self.mip_views = (0..mips)
            .map(|m| {
                tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("vgeom-hzb-mip"),
                    base_mip_level: m,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        self.full_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        self.texture = Some(tex);
        self.size = (w, h);
    }

    /// The full-chain sampled view for the cull compute (valid after [`build`]).
    ///
    /// [`build`]: HzbChain::build
    fn full_view(&self) -> Option<&wgpu::TextureView> {
        self.full_view.as_ref()
    }

    /// Build the pyramid from the frame's depth prepass into [`full_view`].
    ///
    /// [`full_view`]: HzbChain::full_view
    fn build(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        self.ensure(gpu, frame.targets.size);
        if self.generation != frame.targets.generation {
            self.generation = frame.targets.generation;
        }
        let (w, h) = self.size;
        // Pass 0: copy prepass depth → mip 0.
        let bg0 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vgeom-hzb-copy"),
            layout: &self.copy_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&frame.targets.depth_prepass),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.mip_views[0]),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vgeom-hzb-copy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy_pipeline);
            pass.set_bind_group(0, &bg0, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        // Passes N: min-downsample.
        for m in 1..self.mip_views.len() {
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vgeom-hzb-down"),
                layout: &self.down_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.mip_views[m - 1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.mip_views[m]),
                    },
                ],
            });
            let mw = (w >> m).max(1);
            let mh = (h >> m).max(1);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vgeom-hzb-down"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.down_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(mw.div_ceil(8), mh.div_ceil(8), 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    fn view() -> RenderView {
        RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: DVec3::new(0.0, 0.0, 5.0),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width: 320,
            height: 180,
            ortho: None,
        }
    }

    #[test]
    fn threshold_grows_with_distance() {
        let v = view();
        let eye = v.eye_local();
        let near = lod_threshold(eye, Vec3::new(0.0, 0.0, 0.0), 1.0, 1.0, &v, 1.0);
        let far = lod_threshold(eye, Vec3::new(0.0, 0.0, -50.0), 1.0, 1.0, &v, 1.0);
        assert!(far > near, "far {far} should exceed near {near}");
        assert!(near > 0.0);
    }

    #[test]
    fn threshold_scales_with_pixel_error() {
        let v = view();
        let eye = v.eye_local();
        let t1 = lod_threshold(eye, Vec3::new(0.0, 0.0, -10.0), 1.0, 1.0, &v, 1.0);
        let t2 = lod_threshold(eye, Vec3::new(0.0, 0.0, -10.0), 1.0, 1.0, &v, 2.0);
        assert!((t2 - 2.0 * t1).abs() < 1e-4, "t2 {t2} ≈ 2·t1 {t1}");
    }

    #[test]
    fn frustum_planes_contain_center_reject_far_offscreen() {
        let v = view();
        let planes = frustum_planes(v.view_proj());
        // A point right in front of the camera is inside every plane.
        let inside = Vec3::new(0.0, 0.0, 0.0);
        assert!(!outside_frustum(inside, 0.1, &planes), "center culled");
        // A point far to the side (way outside the horizontal FOV) is rejected.
        let side = Vec3::new(1000.0, 0.0, 0.0);
        assert!(outside_frustum(side, 0.1, &planes), "side not culled");
        // Behind the camera.
        let behind = Vec3::new(0.0, 0.0, 100.0);
        assert!(outside_frustum(behind, 0.1, &planes), "behind not culled");
    }

    #[test]
    fn cull_flags_bits() {
        let mut s = VgeomSettings {
            frustum_cull: true,
            cone_cull: false,
            occlusion: false,
            ..VgeomSettings::default()
        };
        assert_eq!(cull_flags(&s), FLAG_FRUSTUM);
        s.cone_cull = true;
        assert_eq!(cull_flags(&s), FLAG_FRUSTUM | FLAG_CONE);
        s.occlusion = true;
        assert_eq!(cull_flags(&s), FLAG_FRUSTUM | FLAG_CONE | FLAG_OCCLUSION);
    }
}
