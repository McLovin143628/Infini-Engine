//! Cloud bake (P17.3): the three compute passes behind the raymarch — the two 3D
//! noise volumes, and the world-XZ cloud-shadow map that lets the layer darken
//! lit geometry.
//!
//! ```text
//! CloudBakeNode ──▶ shape volume   (seed only; baked once, ~never again)
//!                ├─▶ detail volume (seed only)
//!                └─▶ shadow map    (field + layer + weather + wind + sun + centre)
//! ```
//!
//! All three live in [`AtmosphereResources`] beside the sky LUTs, and share its
//! [`generation`](AtmosphereResources::generation). That is not laziness: they are
//! *recreated for the same reason* — [`crate::atmosphere::AtmosphereQuality`] is
//! clamped down by the render tier, and [`CloudQuality`] is derived from it — so a
//! second counter could only ever move in lockstep with the first, and a cache
//! keyed on one but not the other would be a bug waiting to be written. The
//! `EnvBinding` invariant is therefore already satisfied for the shadow map: it
//! rides the atmosphere generation the key already carries.
//!
//! ## Version gating
//!
//! * the **volumes** are a function of the seed alone ([`AtmosphereGpu::cloud_field_key`]).
//!   Coverage, type, wind, altitude and the sun all shape the field at *sample*
//!   time — which is the entire reason it is worth baking — so dragging a
//!   coverage slider re-bakes nothing and 8.4 MB of 3D noise is written once.
//! * the **shadow map** additionally keys on everything that can move a shadow:
//!   the layer geometry, the weather knobs, the wind's current displacement, the
//!   sun direction, the march budget and the map's texel-snapped centre. Not the
//!   camera's altitude — the map is a property of the world, not of the viewer.
//!
//! ## Off path
//!
//! Clouds disabled ⇒ this node dispatches **nothing**. Combined with
//! [`super::cloud::CloudNode`] drawing nothing and the lit shaders' guarded
//! branch, a scene without clouds issues exactly the P17.2 instruction stream and
//! its goldens stay byte-identical.

use crate::clouds::CloudQuality;
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::FrameData;

use super::sky_lut::{AtmosphereResources, CLOUD_NOISE_FORMAT, CLOUD_SHADOW_FORMAT};

/// Workgroup edge of the volume bakes; must match `@workgroup_size(4, 4, 4)` in
/// `cloud_bake.wgsl`.
const VOLUME_WG: u32 = 4;
/// Workgroup edge of the shadow bake; must match `@workgroup_size(8, 8, 1)`.
const SHADOW_WG: u32 = 8;

pub struct CloudBakeNode {
    noise_pipeline_shape: wgpu::ComputePipeline,
    noise_pipeline_detail: wgpu::ComputePipeline,
    noise_bgl: wgpu::BindGroupLayout,
    shadow_pipeline: wgpu::ComputePipeline,
    shadow_bgl: wgpu::BindGroupLayout,
    /// `(resource generation, seed)` the volumes hold.
    volumes_baked: Option<(u64, u32)>,
    /// `(resource generation, inputs)` the shadow map holds.
    shadow_baked: Option<(u64, super::sky_lut::CloudShadowKey)>,
    noise_bg: super::GenCache<u64, wgpu::BindGroup>,
    shadow_bg: super::GenCache<u64, wgpu::BindGroup>,
}

impl CloudBakeNode {
    pub fn new(gpu: &GpuContext) -> Self {
        let compute = wgpu::ShaderStages::COMPUTE;
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let volume_storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: CLOUD_NOISE_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        };
        let volume_sampled = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };

        let noise_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cloud-noise-bake"),
                entries: &[uniform(0), volume_storage(1), volume_storage(2)],
            });
        let shadow_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cloud-shadow-bake"),
                entries: &[
                    uniform(0),
                    volume_sampled(1),
                    volume_sampled(2),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: compute,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: compute,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: CLOUD_SHADOW_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let make = |label: &str, source: &str, bgl: &wgpu::BindGroupLayout, entry: &str| {
            let module = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(super::shader_source(source).into()),
                });
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
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        Self {
            noise_pipeline_shape: make("cloud-shape", "cloud_bake", &noise_bgl, "cs_cloud_shape"),
            noise_pipeline_detail: make(
                "cloud-detail",
                "cloud_bake",
                &noise_bgl,
                "cs_cloud_detail",
            ),
            noise_bgl,
            shadow_pipeline: make(
                "cloud-shadow",
                "cloud_shadow_bake",
                &shadow_bgl,
                "cs_cloud_shadow",
            ),
            shadow_bgl,
            volumes_baked: None,
            shadow_baked: None,
            noise_bg: super::GenCache::default(),
            shadow_bg: super::GenCache::default(),
        }
    }

    fn noise_bind_group<'a>(
        &'a mut self,
        gpu: &GpuContext,
        res: &AtmosphereResources,
    ) -> &'a wgpu::BindGroup {
        let layout = &self.noise_bgl;
        self.noise_bg.get_or_build(res.generation, || {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cloud-noise-bake"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: res.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&res.cloud_shape),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&res.cloud_detail),
                    },
                ],
            })
        })
    }

    fn shadow_bind_group<'a>(
        &'a mut self,
        gpu: &GpuContext,
        res: &AtmosphereResources,
    ) -> &'a wgpu::BindGroup {
        let layout = &self.shadow_bgl;
        self.shadow_bg.get_or_build(res.generation, || {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cloud-shadow-bake"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: res.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&res.cloud_shape),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&res.cloud_detail),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&res.cloud_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&res.cloud_shadow),
                    },
                ],
            })
        })
    }
}

/// Workgroup counts for a cubic volume of edge `res`.
fn volume_groups(res: u32) -> u32 {
    res.div_ceil(VOLUME_WG)
}

impl RenderNode for CloudBakeNode {
    fn name(&self) -> &'static str {
        "cloud-bake"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let params = &frame.scene.atmosphere;
        // The uniform is published by `AtmosphereNode`, which runs first; a
        // disabled scene never even reaches this node's dispatches.
        if !params.clouds_active() {
            return;
        }
        let res = frame.atmosphere;
        let quality: CloudQuality = res.cloud_quality;
        let data = super::sky_lut::AtmosphereGpu::from_frame(frame);

        // ── the two noise volumes: a function of the seed alone ──
        let seed = data.cloud_field_key();
        if self.volumes_baked != Some((res.generation, seed)) {
            let bg = self.noise_bind_group(gpu, res).clone();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cloud-noise-bake"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bg, &[]);
            let g = volume_groups(quality.shape_res());
            pass.set_pipeline(&self.noise_pipeline_shape);
            pass.dispatch_workgroups(g, g, g);
            let g = volume_groups(quality.detail_res());
            pass.set_pipeline(&self.noise_pipeline_detail);
            pass.dispatch_workgroups(g, g, g);
            drop(pass);
            self.volumes_baked = Some((res.generation, seed));
            // Fresh volumes invalidate the shadow map baked from the old ones.
            self.shadow_baked = None;
        }

        // ── the shadow map ──
        // Skipped entirely when the level does not want world shadowing, so the
        // cheap "clouds occlude the sky but do not darken the ground" setup costs
        // literally nothing extra.
        if !params.cloud_shadows_active() {
            return;
        }
        let key = data.cloud_shadow_key();
        if self.shadow_baked == Some((res.generation, key)) {
            return;
        }
        let bg = self.shadow_bind_group(gpu, res).clone();
        let g = quality.shadow_res().div_ceil(SHADOW_WG);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cloud-shadow-bake"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(g, g, 1);
        drop(pass);
        self.shadow_baked = Some((res.generation, key));
    }
}
