//! Classic discrete-LOD mesh fallback for virtualized-geometry content (P13.4).
//!
//! The **classic-LOD fallback** the roadmap requires for GPUs that cannot run the
//! GPU-driven meshlet path ([`crate::passes::vgeom`]). It renders the *same* scene
//! content — [`RenderScene::vgeom_assets`] + [`RenderScene::vgeom_instances`] — but
//! through the ordinary PBR mesh pipeline (`mesh.wgsl`, the same shader/bindings the
//! rigid [`MeshNode`](crate::passes::mesh) uses), so shadows/GI/SSAO come along for
//! free via the shared `@group(2)` environment bind.
//!
//! # Discrete LODs from the meshlet levels (true parity)
//!
//! Each vmesh asset's [`ClassicLod`](inf_vgeom::ClassicLod) chain is derived from
//! the meshlet DAG's LOD levels ([`VgeomMesh::classic_lods`]) — one index buffer per
//! level over the **shared** vmesh vertex buffer (no vertex duplication). Per frame,
//! each instance's screen-space error is projected to the *same* per-instance
//! object-space threshold the meshlet cut uses ([`lod_threshold`]), and the coarsest
//! level within tolerance is picked ([`pick_classic_level`]). So a far/small instance
//! draws a coarse level and a near one draws the finest — the classic twin of the
//! meshlet cut, keyed off the same errors (LOD parity, not a separate metric).
//!
//! # When it runs
//!
//! Only when **`RenderSettings.vgeom.enabled == false`** and the scene carries vgeom
//! content — the exact complement of [`VgeomNode`](crate::passes::vgeom), which runs
//! only when it is enabled. Exactly one of the two draws the content, so there is no
//! double-draw and scenes without vgeom content stay byte-identical (both are no-ops).

use std::collections::BTreeMap;

use glam::Vec3;
use inf_math::FloatingOrigin;
use inf_vgeom::{pick_classic_level, VgeomMesh};

use crate::camera::{RenderView, DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::passes::mesh::{vertex_layouts, InstanceRaw, LightsUniform, MeshVertex};
use crate::passes::vgeom::lod_threshold;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::{MeshInstance, VgeomAsset, VgeomInstance};

/// The per-instance object-space LOD threshold for `inst` of `mesh` under `view`,
/// targeting `pixel_error` px — identical to the meshlet path's `pack_instance`
/// projection, so the classic pick tracks the meshlet cut.
fn instance_threshold(
    origin: &FloatingOrigin,
    view: &RenderView,
    mesh: &VgeomMesh,
    inst: &VgeomInstance,
    pixel_error: f32,
) -> f32 {
    let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
    let max_scale = inst.scale.abs().max_element().max(1e-6);
    let eye = view.eye_local();
    let center_world = model.transform_point3(Vec3::from(mesh.center));
    let radius_world = mesh.radius * max_scale;
    lod_threshold(
        eye,
        center_world,
        radius_world,
        max_scale,
        view,
        pixel_error,
    )
}

/// Pack a [`VgeomInstance`] into the mesh pass's [`InstanceRaw`] (the classic path
/// reuses the rigid-mesh instance layout + shader).
fn pack_vgeom_instance(origin: &FloatingOrigin, inst: &VgeomInstance) -> InstanceRaw {
    InstanceRaw::pack(
        origin,
        &MeshInstance {
            translation: inst.translation,
            rotation: inst.rotation,
            scale: inst.scale,
            color: inst.color,
            metallic: inst.metallic,
            roughness: inst.roughness,
            emissive: inst.emissive,
            id: inst.id,
        },
    )
}

/// The classic-LOD **selection** for a scene's vgeom content: per-instance picked
/// level + the total triangles that would be drawn. Pure (no GPU) — the CI-provable
/// probe that the classic path activates and what it draws, and the same pick the
/// [`ClassicVgeomNode`] rasterizes. Instances referencing an absent/empty asset are
/// skipped (pick `None`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClassicSelection {
    /// Per input instance: `Some(level_index)` (finest-first) drawn, or `None`
    /// (asset missing / no LODs).
    pub picks: Vec<Option<usize>>,
    /// Total triangles drawn across all instances (sum of each picked level's
    /// triangle count).
    pub drawn_triangles: u64,
    /// Number of instances that resolved to a draw.
    pub drawn_instances: u32,
}

/// Compute the [`ClassicSelection`] for `instances` of `assets` under `view`,
/// targeting `pixel_error` px. Deterministic — a pure function of the inputs.
pub fn classic_lod_selection(
    assets: &[VgeomAsset],
    instances: &[VgeomInstance],
    view: &RenderView,
    pixel_error: f32,
) -> ClassicSelection {
    // Per-asset cheap pick data: (errors finest-first, per-level triangle counts).
    let per_asset: BTreeMap<u128, (Vec<f32>, Vec<u64>)> = assets
        .iter()
        .map(|a| {
            let errors = a.mesh.classic_lod_errors();
            let lods = a.mesh.classic_lods();
            let tris: Vec<u64> = lods.iter().map(|l| l.triangle_count() as u64).collect();
            (a.id, (errors, tris))
        })
        .collect();

    let origin = view.origin;
    let mut out = ClassicSelection {
        picks: Vec::with_capacity(instances.len()),
        drawn_triangles: 0,
        drawn_instances: 0,
    };
    for inst in instances {
        let Some((errors, tris)) = per_asset.get(&inst.asset) else {
            out.picks.push(None);
            continue;
        };
        if errors.is_empty() {
            out.picks.push(None);
            continue;
        }
        // Look up the mesh for the projection (bounds).
        let mesh = assets
            .iter()
            .find(|a| a.id == inst.asset)
            .map(|a| a.mesh.as_ref());
        let threshold = mesh
            .map(|m| instance_threshold(&origin, view, m, inst, pixel_error))
            .unwrap_or(0.0);
        let level = pick_classic_level(errors, threshold);
        out.drawn_triangles += tris.get(level).copied().unwrap_or(0);
        out.drawn_instances += 1;
        out.picks.push(Some(level));
    }
    out
}

// ── Per-asset GPU geometry cache ─────────────────────────────────────────────

/// One vmesh asset's classic-LOD GPU geometry: a shared vertex buffer + one index
/// buffer per LOD level (finest-first), with each level's representative error for
/// the pick.
struct ClassicGpu {
    vertices: wgpu::Buffer,
    /// Per LOD level (finest-first): `(error, index_buffer, index_count)`.
    levels: Vec<(f32, wgpu::Buffer, u32)>,
    /// Per-level errors (finest-first) — the pick input.
    errors: Vec<f32>,
}

impl ClassicGpu {
    fn build(gpu: &GpuContext, mesh: &VgeomMesh) -> Self {
        let verts: Vec<MeshVertex> = mesh
            .vertices
            .iter()
            .map(|v| MeshVertex {
                pos: v.position,
                normal: v.normal,
            })
            .collect();
        let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("classic-vgeom-verts"),
            size: (std::mem::size_of::<MeshVertex>() * verts.len().max(1)) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !verts.is_empty() {
            gpu.queue
                .write_buffer(&vertices, 0, bytemuck::cast_slice(&verts));
        }

        let lods = mesh.classic_lods();
        let mut errors = Vec::with_capacity(lods.len());
        let levels = lods
            .iter()
            .map(|l| {
                errors.push(l.error);
                let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("classic-vgeom-indices"),
                    size: (std::mem::size_of::<u32>() * l.indices.len().max(1)) as u64,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                if !l.indices.is_empty() {
                    gpu.queue
                        .write_buffer(&buf, 0, bytemuck::cast_slice(&l.indices));
                }
                (l.error, buf, l.indices.len() as u32)
            })
            .collect();

        Self {
            vertices,
            levels,
            errors,
        }
    }
}

/// The classic-LOD fallback render node (P13.4).
pub struct ClassicVgeomNode {
    pipeline: wgpu::RenderPipeline,
    lights_buf: wgpu::Buffer,
    lights_bg: wgpu::BindGroup,
    env: super::EnvBinding,
    geom: BTreeMap<u128, ClassicGpu>,
    /// Reused per-(asset,level) instance buffers, keyed `(asset_id, level)`.
    batches: BTreeMap<(u128, usize), wgpu::Buffer>,
    batch_caps: BTreeMap<(u128, usize), usize>,
}

impl ClassicVgeomNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("classic-vgeom"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("mesh").into()),
            });

        let lights_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("classic-vgeom-lights"),
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
            label: Some("classic-vgeom-lights"),
            size: std::mem::size_of::<LightsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("classic-vgeom-lights"),
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
                label: Some("classic-vgeom"),
                bind_group_layouts: &[Some(view_bgl), Some(&lights_bgl), Some(&env.bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("classic-vgeom"),
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
            lights_buf,
            lights_bg,
            env,
            geom: BTreeMap::new(),
            batches: BTreeMap::new(),
            batch_caps: BTreeMap::new(),
        }
    }

    /// Upload `raw` into the reusable batch buffer for `key`, growing it if needed.
    fn upload_batch(&mut self, gpu: &GpuContext, key: (u128, usize), raw: &[InstanceRaw]) {
        let needed = raw.len();
        let cap = self.batch_caps.get(&key).copied().unwrap_or(0);
        if cap < needed || !self.batches.contains_key(&key) {
            let new_cap = needed.next_power_of_two().max(4);
            let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("classic-vgeom-instances"),
                size: (new_cap * std::mem::size_of::<InstanceRaw>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.batches.insert(key, buf);
            self.batch_caps.insert(key, new_cap);
        }
        gpu.queue.write_buffer(
            self.batches.get(&key).unwrap(),
            0,
            bytemuck::cast_slice(raw),
        );
    }
}

impl RenderNode for ClassicVgeomNode {
    fn name(&self) -> &'static str {
        "classic-vgeom"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        // Runs only as the fallback: vgeom OFF + vgeom content present.
        if frame.settings.vgeom.enabled
            || frame.scene.vgeom_instances.is_empty()
            || frame.scene.vgeom_assets.is_empty()
        {
            return;
        }

        // Cache each referenced asset's classic geometry.
        for asset in &frame.scene.vgeom_assets {
            self.geom
                .entry(asset.id)
                .or_insert_with(|| ClassicGpu::build(gpu, &asset.mesh));
        }
        let mesh_of: BTreeMap<u128, &VgeomMesh> = frame
            .scene
            .vgeom_assets
            .iter()
            .map(|a| (a.id, a.mesh.as_ref()))
            .collect();

        // Lights (shared model with the rigid mesh pass).
        let lights = LightsUniform::from_scene(frame.scene, &frame.view.origin);
        gpu.queue
            .write_buffer(&self.lights_buf, 0, bytemuck::bytes_of(&lights));

        let origin = frame.view.origin;
        let pixel_error = frame.settings.vgeom.pixel_error;

        // Group instances by (asset, picked level).
        let mut grouped: BTreeMap<(u128, usize), Vec<InstanceRaw>> = BTreeMap::new();
        for inst in &frame.scene.vgeom_instances {
            let Some(geom) = self.geom.get(&inst.asset) else {
                continue;
            };
            if geom.errors.is_empty() {
                continue;
            }
            let Some(mesh) = mesh_of.get(&inst.asset) else {
                continue;
            };
            let threshold = instance_threshold(&origin, frame.view, mesh, inst, pixel_error);
            let level = pick_classic_level(&geom.errors, threshold);
            grouped
                .entry((inst.asset, level))
                .or_default()
                .push(pack_vgeom_instance(&origin, inst));
        }
        if grouped.is_empty() {
            return;
        }

        for (key, raw) in &grouped {
            self.upload_batch(gpu, *key, raw);
        }

        let env_bg = self.env.bind_group(gpu, frame).clone();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("classic-vgeom"),
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

        for (key, raw) in &grouped {
            let (asset_id, level) = *key;
            let Some(geom) = self.geom.get(&asset_id) else {
                continue;
            };
            let Some((_, index_buf, index_count)) = geom.levels.get(level) else {
                continue;
            };
            if *index_count == 0 {
                continue;
            }
            let Some(inst_buf) = self.batches.get(key) else {
                continue;
            };
            pass.set_vertex_buffer(0, geom.vertices.slice(..));
            pass.set_vertex_buffer(1, inst_buf.slice(..));
            pass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..*index_count, 0, 0..raw.len() as u32);
        }
    }
}
