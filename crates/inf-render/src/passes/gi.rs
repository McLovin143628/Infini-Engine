//! Dynamic global-illumination pass (P13.3b): the two-stage compute front end for
//! the real-time single-bounce diffuse GI. Each frame (when enabled) it
//!
//! 1. **voxelizes** the rigid mesh instances into a camera-centred
//!    [`crate::gi::GI_DIM`]³ occupancy+albedo storage volume (`gi_voxelize.wgsl`,
//!    analytic box test — see [`crate::gi`]), then
//! 2. **marches** the [`crate::gi::PROBE_DIMS`] probe grid through it, projecting
//!    single-bounce radiance to L1 SH per probe (`gi_probes.wgsl`).
//!
//! The lit passes then sample the SH buffer through [`crate::passes::EnvBinding`]
//! (their ambient term becomes the probe-interpolated irradiance). The SH buffer +
//! voxel buffer + shared [`GiData`] uniform live in the renderer-owned
//! [`GiResources`] (created once, like the shadow resources); this node owns the
//! two compute pipelines + the per-frame instance buffer.
//!
//! ## Scope (v1, documented)
//!
//! * Voxelizes **rigid mesh instances** (boxes) only — terrain + skinned
//!   voxelization is a follow-up.
//! * Revoxelizes **every frame** (correct; a temporal-amortization pass is a
//!   follow-up). The volume follows the camera, so it always covers the near field.
//! * **Off path:** disabled → the node writes the shared uniform with `enabled = 0`
//!   and dispatches nothing, so receivers keep the byte-stable hemispheric ambient.

use glam::Vec3;

use crate::gi::{probe_count, GI_DIM, PROBE_DIMS};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::FrameData;
use crate::scene::LightKind;

/// Max rigid instances voxelized per frame (each is a box). Excess is ignored.
const MAX_GI_INSTANCES: usize = 256;

/// The shared GI uniform (`std140`), written by [`GiNode`] and read by the two
/// compute shaders **and** every lit pass. Mirrors `struct GiData` across
/// `gi_voxelize.wgsl` / `gi_probes.wgsl` / the receiver shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GiDataGpu {
    /// xyz = render-local volume min corner, w = voxel size (m).
    pub vol_min: [f32; 4],
    /// xyz = render-local probe grid min corner, w = extent (m).
    pub probe_min: [f32; 4],
    /// x = voxel dim, yzw = probe dims.
    pub dims: [f32; 4],
    /// x = enabled, y = intensity, z = rays, w = instance count.
    pub params: [f32; 4],
    /// xyz = unit direction toward the sun.
    pub sun_dir: [f32; 4],
    /// rgb = sun radiance (colour × intensity).
    pub sun_color: [f32; 4],
    /// rgb = sky zenith radiance (ray miss, upward).
    pub sky_zenith: [f32; 4],
    /// rgb = sky horizon radiance (ray miss, sideways).
    pub sky_horizon: [f32; 4],
}

impl GiDataGpu {
    fn disabled() -> Self {
        Self {
            vol_min: [0.0, 0.0, 0.0, 1.0],
            probe_min: [0.0, 0.0, 0.0, 1.0],
            dims: [
                GI_DIM as f32,
                PROBE_DIMS[0] as f32,
                PROBE_DIMS[1] as f32,
                PROBE_DIMS[2] as f32,
            ],
            params: [0.0, 1.0, 48.0, 0.0],
            sun_dir: [0.0, 1.0, 0.0, 0.0],
            sun_color: [0.0; 4],
            sky_zenith: [0.0; 4],
            sky_horizon: [0.0; 4],
        }
    }
}

/// One voxelized instance (`struct GiInstance` in `gi_voxelize.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GiInstanceGpu {
    inv_model: [f32; 16],
    albedo: [f32; 4],
}

/// Renderer-owned GI GPU resources, created once and shared with the lit passes via
/// [`FrameData`]. The voxel + SH buffers are written by the compute passes; the lit
/// passes read [`sh`](Self::sh) + [`uniform`](Self::uniform).
pub struct GiResources {
    /// Occupancy+albedo volume (`GI_DIM³` packed RGBA8 `u32`s).
    pub voxels: wgpu::Buffer,
    /// L1 SH per probe (`probe_count × 4 vec4<f32>`).
    pub sh: wgpu::Buffer,
    /// Shared `GiData` uniform (written by the node, read by everyone).
    pub uniform: wgpu::Buffer,
    /// Stable generation for receiver bind-group caching.
    pub generation: u64,
}

impl GiResources {
    pub fn new(gpu: &GpuContext) -> Self {
        let voxel_count = (GI_DIM * GI_DIM * GI_DIM) as u64;
        let voxels = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi-voxels"),
            size: voxel_count * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let sh = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi-sh"),
            // 4 vec4<f32> per probe.
            size: probe_count() as u64 * 4 * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi-data"),
            size: std::mem::size_of::<GiDataGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&uniform, 0, bytemuck::bytes_of(&GiDataGpu::disabled()));
        Self {
            voxels,
            sh,
            uniform,
            generation: 0,
        }
    }
}

pub struct GiNode {
    voxelize_pipeline: wgpu::ComputePipeline,
    voxelize_bgl: wgpu::BindGroupLayout,
    probe_pipeline: wgpu::ComputePipeline,
    probe_bgl: wgpu::BindGroupLayout,
    instances: wgpu::Buffer,
    instance_cap: usize,
    /// Whether the disabled-GI uniform is already published to the (stable,
    /// created-once) `frame.gi.uniform`. Gates the constant re-write while GI stays
    /// off; the enabled path clears it so a later disable re-publishes.
    published_disabled: bool,
}

impl GiNode {
    pub fn new(gpu: &GpuContext) -> Self {
        let vox_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gi-voxelize"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../shaders/gi_voxelize.wgsl").into(),
                ),
            });
        let probe_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gi-probes"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gi_probes.wgsl").into()),
            });

        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_entry = |binding, ro| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let voxelize_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gi-voxelize"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, true),  // instances
                    storage_entry(2, false), // voxels (write)
                ],
            });
        let probe_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gi-probes"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, true),  // voxels (read)
                    storage_entry(2, false), // sh (write)
                ],
            });

        let mk = |label, bgl: &wgpu::BindGroupLayout, module, entry: &str| {
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
                    module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let voxelize_pipeline = mk("gi-voxelize", &voxelize_bgl, &vox_shader, "cs_voxelize");
        let probe_pipeline = mk("gi-probes", &probe_bgl, &probe_shader, "cs_probes");

        let instances = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi-instances"),
            size: (std::mem::size_of::<GiInstanceGpu>() * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            voxelize_pipeline,
            voxelize_bgl,
            probe_pipeline,
            probe_bgl,
            instances,
            instance_cap: 16,
            published_disabled: false,
        }
    }
}

impl RenderNode for GiNode {
    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let s = &frame.settings.gi;
        if !s.enabled {
            // Publish the disabled uniform once; the buffer is created once and only
            // this node writes it, so re-writing the constant every frame is redundant.
            if !self.published_disabled {
                gpu.queue.write_buffer(
                    &frame.gi.uniform,
                    0,
                    bytemuck::bytes_of(&GiDataGpu::disabled()),
                );
                self.published_disabled = true;
            }
            return;
        }
        // The enabled path writes the real uniform below, so a later disable must
        // re-publish the disabled block.
        self.published_disabled = false;

        let origin = &frame.view.origin;
        let eye = frame.view.eye_local();
        let extent = s.extent.max(1.0);
        let vsize = extent / GI_DIM as f32;
        let vol_min = eye - Vec3::splat(extent * 0.5);

        // Pack instances (as boxes: inverse model matrix + albedo).
        let packed: Vec<GiInstanceGpu> = frame
            .scene
            .instances
            .iter()
            .take(MAX_GI_INSTANCES)
            .map(|inst| {
                let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
                let inv = model.inverse();
                GiInstanceGpu {
                    inv_model: inv.to_cols_array(),
                    albedo: [inst.color[0], inst.color[1], inst.color[2], 0.0],
                }
            })
            .collect();
        let count = packed.len();
        if count > self.instance_cap {
            let cap = count.next_power_of_two().max(16);
            self.instances = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gi-instances"),
                size: (cap * std::mem::size_of::<GiInstanceGpu>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_cap = cap;
        }
        if !packed.is_empty() {
            gpu.queue
                .write_buffer(&self.instances, 0, bytemuck::cast_slice(&packed));
        }

        // Sun + sky radiance from the scene (first directional light, else the
        // projected time-of-day sun).
        let (sun_dir, sun_color) = frame
            .scene
            .lights
            .iter()
            .find(|l| l.kind == LightKind::Directional)
            .map(|l| {
                (
                    l.direction.normalize_or_zero(),
                    [
                        l.color[0] * l.intensity,
                        l.color[1] * l.intensity,
                        l.color[2] * l.intensity,
                    ],
                )
            })
            // No directional light: fall back to the scene's **projected** sun
            // (P17.1), so probe radiance tracks the time of day instead of a
            // compile-time constant. A scene with no time-of-day authority
            // projects the retired constant, so this is byte-identical for
            // content that has not opted in.
            .unwrap_or_else(|| {
                let sun = &frame.scene.sun;
                let i = sun.intensity;
                (
                    sun.unit_direction(),
                    [sun.color[0] * i, sun.color[1] * i, sun.color[2] * i],
                )
            });

        let sky = &frame.scene.sky;
        let data = GiDataGpu {
            vol_min: [vol_min.x, vol_min.y, vol_min.z, vsize],
            probe_min: [vol_min.x, vol_min.y, vol_min.z, extent],
            dims: [
                GI_DIM as f32,
                PROBE_DIMS[0] as f32,
                PROBE_DIMS[1] as f32,
                PROBE_DIMS[2] as f32,
            ],
            params: [1.0, s.intensity, s.rays.max(1) as f32, count as f32],
            sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z, 0.0],
            sun_color: [sun_color[0], sun_color[1], sun_color[2], 0.0],
            sky_zenith: [sky.zenith[0], sky.zenith[1], sky.zenith[2], 0.0],
            sky_horizon: [sky.horizon[0], sky.horizon[1], sky.horizon[2], 0.0],
        };
        gpu.queue
            .write_buffer(&frame.gi.uniform, 0, bytemuck::bytes_of(&data));

        // Voxelize → probe march.
        let vox_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi-voxelize"),
            layout: &self.voxelize_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.gi.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame.gi.voxels.as_entire_binding(),
                },
            ],
        });
        let probe_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gi-probes"),
            layout: &self.probe_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.gi.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: frame.gi.voxels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: frame.gi.sh.as_entire_binding(),
                },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi-voxelize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.voxelize_pipeline);
            pass.set_bind_group(0, &vox_bg, &[]);
            let total = GI_DIM * GI_DIM * GI_DIM;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi-probes"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.probe_pipeline);
            pass.set_bind_group(0, &probe_bg, &[]);
            pass.dispatch_workgroups(probe_count().div_ceil(64), 1, 1);
        }
    }

    fn name(&self) -> &'static str {
        "gi"
    }
}
