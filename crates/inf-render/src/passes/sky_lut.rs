//! Atmosphere LUT bake (P17.2): the two compute passes that turn
//! [`crate::atmosphere::AtmosphereParams`] + the projected sun into the pair of
//! lookup textures the sky pass and every lit pass sample.
//!
//! ```text
//! AtmosphereNode ──▶ transmittance LUT  (medium only; baked ~never)
//!                └─▶ sky-view LUT       (medium + sun + camera altitude; per-change)
//! ```
//!
//! Both live in the renderer-owned [`AtmosphereResources`], created once per
//! quality level and shared with the sky pass and [`super::EnvBinding`] through
//! [`crate::renderer::FrameData`] — the same arrangement
//! [`super::gi::GiResources`] uses, with one crucial difference: **these textures
//! are resizable**, because [`AtmosphereQuality`] can be clamped down by the
//! render tier at runtime. That is exactly the case the P13 `EnvBinding` cache
//! invariant warned about, so [`AtmosphereResources::generation`] exists and the
//! env bind-group cache keys on it. See [`super::EnvBinding::bind_group`].
//!
//! ## Version gating
//!
//! Neither LUT is re-baked unless its own inputs changed:
//!
//! * the **transmittance** LUT is a function of the medium and the planet shell
//!   alone ([`MediumKey`]) — not of the sun, not of the camera — so it is baked
//!   on the first enabled frame and then essentially never again;
//! * the **sky-view** LUT additionally depends on the sun direction, the sun's
//!   radiance, the exposure and the camera's altitude ([`SkyViewKey`]). The
//!   altitude is quantized to whole metres, so a hovering flycam does not
//!   re-bake 60 times a second over sub-millimetre jitter — a metre of altitude
//!   changes nothing visible at a 6360 km planet radius.
//!
//! ## Off path
//!
//! Disabled ⇒ the node dispatches **nothing** and only publishes the
//! `enabled = 0` uniform (once — a latch, like the GI node's). The sky pass and
//! the lit passes then take their pre-P17.2 arithmetic, so every golden without a
//! `TimeOfDay` authority is byte-identical.

use crate::atmosphere::{
    camera_radius_km, AtmosphereParams, AtmosphereQuality, SKY_EXPOSURE_CALIBRATION,
};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::FrameData;
use crate::scene::SunParams;

/// LUT storage format. `Rgba16Float` is storage-capable in core WebGPU and holds
/// the >1 radiances the sky-view LUT carries near the sun without clipping.
pub const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Bytes per LUT texel (four halves).
const LUT_TEXEL_BYTES: u32 = 8;

/// The shared atmosphere uniform (`std140`), written by [`AtmosphereNode`] and
/// read by both LUT compute shaders, the sky pass and every lit pass. Mirrors
/// `struct AtmosphereData` in `shaders/atmosphere.wgsl` field for field.
///
/// Units follow [`crate::atmosphere`]: kilometres and km⁻¹ everywhere except the
/// `fog` block, which is SI metres.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AtmosphereGpu {
    /// x = enabled, y = sky exposure, z = aerial-perspective strength,
    /// w = gradient tint strength.
    pub params: [f32; 4],
    /// rgb = Rayleigh scattering (km⁻¹), w = Rayleigh scale height (km).
    pub rayleigh: [f32; 4],
    /// x = Mie scattering (km⁻¹, turbidity-scaled), y = Mie absorption (km⁻¹),
    /// z = Mie scale height (km), w = Mie phase `g`.
    pub mie: [f32; 4],
    /// rgb = ozone absorption (km⁻¹), w = transmittance-bake step count.
    pub ozone: [f32; 4],
    /// x = ozone centre (km), y = ozone half-width (km), z = ground albedo,
    /// w = sky-view-bake step count.
    pub ozone_shape: [f32; 4],
    /// x = ground radius (km), y = top radius (km), z = camera radius (km),
    /// w = camera world altitude (m).
    pub planet: [f32; 4],
    /// xyz = unit direction toward the sun, w = cos(sun disc half-angle).
    pub sun_dir: [f32; 4],
    /// rgb = sun irradiance (colour × intensity), w reserved.
    pub sun_color: [f32; 4],
    /// xyz = unit direction toward the moon, w = cos(moon disc half-angle).
    pub moon_dir: [f32; 4],
    /// rgb = moon radiance (colour × intensity), w = lunar phase `[0, 1)`.
    pub moon_color: [f32; 4],
    /// x = star intensity, y = star cells per cube-face axis, z = night fade,
    /// w reserved.
    pub stars: [f32; 4],
    /// **SI metres**: x = fog density (m⁻¹), y = fog falloff (m⁻¹),
    /// z = fog reference height (m), w reserved.
    pub fog: [f32; 4],
    /// rgb = fog tint, w reserved.
    pub fog_color: [f32; 4],
}

impl AtmosphereGpu {
    /// The `enabled = 0` block every pre-P17.2 scene renders under. The medium
    /// values are still the physical ones so a debugger shows something sane, but
    /// nothing reads them.
    pub fn disabled() -> Self {
        Self::build(
            &AtmosphereParams::default(),
            &SunParams::default(),
            0.0,
            AtmosphereQuality::High,
        )
    }

    /// Pack a scene's atmosphere + sun into the shared uniform.
    ///
    /// `eye_altitude_m` is the camera's **world** altitude in metres (the only
    /// metre-valued input the planet block takes; [`camera_radius_km`] converts).
    pub fn build(
        p: &AtmosphereParams,
        sun: &SunParams,
        eye_altitude_m: f32,
        quality: AtmosphereQuality,
    ) -> Self {
        let sd = sun.unit_direction();
        let md = sun.unit_moon_direction();
        let i = sun.intensity.max(0.0);
        let mi = sun.moon_intensity.max(0.0);
        Self {
            params: [
                if p.enabled { 1.0 } else { 0.0 },
                p.sky_intensity.max(0.0) * SKY_EXPOSURE_CALIBRATION,
                p.aerial_perspective.clamp(0.0, 4.0),
                p.tint_strength.clamp(0.0, 1.0),
            ],
            rayleigh: [
                p.rayleigh_scattering[0],
                p.rayleigh_scattering[1],
                p.rayleigh_scattering[2],
                p.rayleigh_height.max(1e-3),
            ],
            mie: [
                p.mie_scattering_scaled(),
                p.mie_absorption_scaled(),
                p.mie_height.max(1e-3),
                p.mie_g.clamp(-0.95, 0.95),
            ],
            ozone: [
                p.ozone_absorption[0],
                p.ozone_absorption[1],
                p.ozone_absorption[2],
                quality.transmittance_steps() as f32,
            ],
            ozone_shape: [
                p.ozone_center,
                p.ozone_half_width.max(1e-3),
                p.ground_albedo.clamp(0.0, 1.0),
                quality.skyview_steps() as f32,
            ],
            planet: [
                p.ground_radius,
                p.top_radius,
                camera_radius_km(p, eye_altitude_m),
                eye_altitude_m,
            ],
            sun_dir: [sd.x, sd.y, sd.z, p.sun_disc_cos()],
            sun_color: [sun.color[0] * i, sun.color[1] * i, sun.color[2] * i, 0.0],
            moon_dir: [md.x, md.y, md.z, p.moon_disc_cos()],
            moon_color: [
                sun.moon_color[0] * mi,
                sun.moon_color[1] * mi,
                sun.moon_color[2] * mi,
                p.moon_phase.rem_euclid(1.0),
            ],
            stars: [
                p.star_intensity.max(0.0),
                quality.star_cells(),
                night_fade(sd.y),
                0.0,
            ],
            fog: [
                p.fog.density.max(0.0),
                p.fog.falloff.max(0.0),
                p.fog.height,
                0.0,
            ],
            fog_color: [p.fog.color[0], p.fog.color[1], p.fog.color[2], 0.0],
        }
    }

    fn medium_key(&self) -> MediumKey {
        MediumKey {
            rayleigh: self.rayleigh.map(f32::to_bits),
            mie: self.mie.map(f32::to_bits),
            ozone: self.ozone.map(f32::to_bits),
            ozone_shape: self.ozone_shape.map(f32::to_bits),
            shell: [self.planet[0].to_bits(), self.planet[1].to_bits()],
        }
    }

    fn skyview_key(&self) -> SkyViewKey {
        SkyViewKey {
            medium: self.medium_key(),
            sun_dir: [
                self.sun_dir[0].to_bits(),
                self.sun_dir[1].to_bits(),
                self.sun_dir[2].to_bits(),
            ],
            sun_color: [
                self.sun_color[0].to_bits(),
                self.sun_color[1].to_bits(),
                self.sun_color[2].to_bits(),
            ],
            exposure: self.params[1].to_bits(),
            // Camera radius, quantized to whole metres: a sub-metre flycam
            // wobble must not re-bake the LUT, and cannot change it visibly.
            radius_m: (self.planet[2] as f64 * 1000.0).round() as i64,
        }
    }
}

/// How far past sunset the stars have fully faded in, as a sine of elevation.
/// `-0.12` ≈ 6.9° below the horizon — the end of civil twilight, which is
/// exactly when a human first sees stars.
const NIGHT_FADE_ELEVATION_SIN: f32 = 0.12;

/// Star/night blend from the sun's elevation sine: `0` while the sun is up,
/// smoothly `1` by the end of civil twilight.
fn night_fade(sun_y: f32) -> f32 {
    let t = (-sun_y / NIGHT_FADE_ELEVATION_SIN).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// What the transmittance LUT is a function of. Bit patterns rather than floats
/// so the key is `Eq` and a `NaN` parameter cannot make the cache thrash.
#[derive(Clone, Copy, PartialEq, Eq)]
struct MediumKey {
    rayleigh: [u32; 4],
    mie: [u32; 4],
    ozone: [u32; 4],
    ozone_shape: [u32; 4],
    shell: [u32; 2],
}

/// What the sky-view LUT is a function of.
#[derive(Clone, Copy, PartialEq, Eq)]
struct SkyViewKey {
    medium: MediumKey,
    sun_dir: [u32; 3],
    sun_color: [u32; 3],
    exposure: u32,
    radius_m: i64,
}

/// Renderer-owned atmosphere GPU resources: the two LUTs, their sampler and the
/// shared uniform.
///
/// **Resizable** (unlike the shadow/GI resources): a change of
/// [`AtmosphereQuality`] recreates the textures and bumps
/// [`generation`](Self::generation), which every bind-group cache that embeds
/// them must key on.
pub struct AtmosphereResources {
    transmittance_tex: wgpu::Texture,
    sky_view_tex: wgpu::Texture,
    /// Transmittance LUT (sun-zenith-angle × altitude).
    pub transmittance: wgpu::TextureView,
    /// Sky-view LUT (view direction, sun-relative).
    pub sky_view: wgpu::TextureView,
    /// Clamped bilinear sampler both LUTs are read through.
    pub sampler: wgpu::Sampler,
    /// The shared `AtmosphereData` uniform.
    pub uniform: wgpu::Buffer,
    /// Bumped on every recreation. Bind-group caches MUST key on this.
    pub generation: u64,
    /// The quality these textures were sized for.
    pub quality: AtmosphereQuality,
}

impl AtmosphereResources {
    pub fn new(gpu: &GpuContext, quality: AtmosphereQuality, generation: u64) -> Self {
        let make = |label: &str, (w, h): (u32, u32)| {
            gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: LUT_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    // The LUT-determinism gate reads these back on the CPU.
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let transmittance_tex = make("atmos-transmittance-lut", quality.transmittance_size());
        let sky_view_tex = make("atmos-skyview-lut", quality.skyview_size());
        let transmittance = transmittance_tex.create_view(&Default::default());
        let sky_view = sky_view_tex.create_view(&Default::default());
        // Clamp-to-edge is load-bearing, not a default: the transmittance
        // parameterization puts the horizon at the very edge of the u axis, and a
        // repeating sampler would wrap a grazing ray onto an overhead one.
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atmos-lut"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmos-data"),
            size: std::mem::size_of::<AtmosphereGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&uniform, 0, bytemuck::bytes_of(&AtmosphereGpu::disabled()));
        Self {
            transmittance_tex,
            sky_view_tex,
            transmittance,
            sky_view,
            sampler,
            uniform,
            generation,
            quality,
        }
    }

    /// Blocking readback of the transmittance LUT as tightly-packed
    /// `Rgba16Float` texels. The LUT-determinism gate compares two of these.
    pub fn read_transmittance(&self, gpu: &GpuContext) -> Result<Vec<u8>, String> {
        read_lut(
            gpu,
            &self.transmittance_tex,
            self.quality.transmittance_size(),
        )
    }

    /// Blocking readback of the sky-view LUT. See [`Self::read_transmittance`].
    pub fn read_sky_view(&self, gpu: &GpuContext) -> Result<Vec<u8>, String> {
        read_lut(gpu, &self.sky_view_tex, self.quality.skyview_size())
    }
}

/// 256-byte-row-aligned texture → CPU copy, unpadded on the way out (the same
/// shape as [`crate::headless::HeadlessTarget::read_rgba`], for a 4×f16 format).
fn read_lut(gpu: &GpuContext, tex: &wgpu::Texture, (w, h): (u32, u32)) -> Result<Vec<u8>, String> {
    let unpadded = (w * LUT_TEXEL_BYTES) as usize;
    let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("atmos-lut-readback"),
        size: (padded * h as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("atmos-lut-readback"),
        });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("poll: {e}"))?;
    rx.recv()
        .map_err(|e| format!("map_async dropped: {e}"))?
        .map_err(|e| format!("map_async: {e}"))?;
    let data = slice
        .get_mapped_range()
        .map_err(|e| format!("get_mapped_range: {e}"))?;
    let mut out = Vec::with_capacity(unpadded * h as usize);
    for row in data.chunks(padded) {
        out.extend_from_slice(&row[..unpadded]);
    }
    drop(data);
    buffer.unmap();
    Ok(out)
}

/// The graph node that owns the two bake pipelines.
pub struct AtmosphereNode {
    transmittance_pipeline: wgpu::ComputePipeline,
    transmittance_bgl: wgpu::BindGroupLayout,
    sky_view_pipeline: wgpu::ComputePipeline,
    sky_view_bgl: wgpu::BindGroupLayout,
    /// The uniform value currently in `frame.atmosphere.uniform`.
    uploaded: Option<AtmosphereGpu>,
    /// `(resource generation, medium)` the transmittance LUT holds.
    transmittance_baked: Option<(u64, MediumKey)>,
    /// `(resource generation, inputs)` the sky-view LUT holds.
    sky_view_baked: Option<(u64, SkyViewKey)>,
    /// Bind groups, cached against the resource generation (see
    /// [`super::GenCache`]). Nothing here is size-dependent, so the frame-target
    /// generation is deliberately not part of the key.
    transmittance_bg: super::GenCache<u64, wgpu::BindGroup>,
    sky_view_bg: super::GenCache<u64, wgpu::BindGroup>,
}

impl AtmosphereNode {
    pub fn new(gpu: &GpuContext) -> Self {
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: LUT_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };

        let transmittance_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("atmos-transmittance"),
                    entries: &[uniform_entry(0), storage_entry(1)],
                });
        let sky_view_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("atmos-skyview"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline = |label: &str, source: String, bgl: &wgpu::BindGroupLayout, entry: &str| {
            let module = gpu
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
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
            transmittance_pipeline: pipeline(
                "atmos-transmittance",
                super::shader_source("atmos_transmittance"),
                &transmittance_bgl,
                "cs_transmittance",
            ),
            transmittance_bgl,
            sky_view_pipeline: pipeline(
                "atmos-skyview",
                super::shader_source("atmos_skyview"),
                &sky_view_bgl,
                "cs_skyview",
            ),
            sky_view_bgl,
            uploaded: None,
            transmittance_baked: None,
            sky_view_baked: None,
            transmittance_bg: super::GenCache::default(),
            sky_view_bg: super::GenCache::default(),
        }
    }
}

/// Compute workgroup edge; must match `@workgroup_size(8, 8, 1)` in both bakes.
const WG: u32 = 8;

impl RenderNode for AtmosphereNode {
    fn name(&self) -> &'static str {
        "atmosphere"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let res = frame.atmosphere;
        let params = &frame.scene.atmosphere;
        let data = AtmosphereGpu::build(
            params,
            &frame.scene.sun,
            frame.view.eye_world.y as f32,
            res.quality,
        );

        // Publish the uniform when it changed. `AtmosphereResources::new` already
        // wrote the disabled block, so a scene that never enables the atmosphere
        // writes this buffer exactly zero times.
        if self.uploaded != Some(data) {
            gpu.queue
                .write_buffer(&res.uniform, 0, bytemuck::bytes_of(&data));
            self.uploaded = Some(data);
        }
        if !params.enabled {
            return;
        }

        // ── transmittance LUT: medium only ──
        let medium = data.medium_key();
        if self.transmittance_baked != Some((res.generation, medium)) {
            let layout = &self.transmittance_bgl;
            let bg = self
                .transmittance_bg
                .get_or_build(res.generation, || {
                    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("atmos-transmittance"),
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: res.uniform.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(&res.transmittance),
                            },
                        ],
                    })
                })
                .clone();
            let (w, h) = res.quality.transmittance_size();
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("atmos-transmittance"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.transmittance_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(WG), h.div_ceil(WG), 1);
            drop(pass);
            self.transmittance_baked = Some((res.generation, medium));
            // A fresh transmittance LUT invalidates the sky view that was baked
            // from the old one.
            self.sky_view_baked = None;
        }

        // ── sky-view LUT: medium + sun + camera altitude ──
        let key = data.skyview_key();
        if self.sky_view_baked == Some((res.generation, key)) {
            return;
        }
        let layout = &self.sky_view_bgl;
        let bg = self
            .sky_view_bg
            .get_or_build(res.generation, || {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("atmos-skyview"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: res.uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&res.sky_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&res.transmittance),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&res.sampler),
                        },
                    ],
                })
            })
            .clone();
        let (w, h) = res.quality.skyview_size();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("atmos-skyview"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.sky_view_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(w.div_ceil(WG), h.div_ceil(WG), 1);
        drop(pass);
        self.sky_view_baked = Some((res.generation, key));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::HeightFog;

    fn enabled() -> AtmosphereParams {
        AtmosphereParams {
            enabled: true,
            ..AtmosphereParams::default()
        }
    }

    #[test]
    fn disabled_block_is_flagged_off() {
        let d = AtmosphereGpu::disabled();
        assert_eq!(d.params[0], 0.0);
        // The medium is still physical, so a capture shows something meaningful.
        assert!(d.rayleigh[2] > d.rayleigh[0]);
        assert_eq!(d.fog[0], 0.0);
    }

    /// The whole point of the version gate: the transmittance key must ignore the
    /// sun and the camera, and the sky-view key must not.
    #[test]
    fn lut_keys_track_the_right_inputs() {
        let p = enabled();
        let noon = SunParams {
            direction: glam::Vec3::new(0.0, 1.0, 0.0),
            ..SunParams::default()
        };
        let dusk = SunParams {
            direction: glam::Vec3::new(1.0, 0.05, 0.0),
            ..SunParams::default()
        };
        let q = AtmosphereQuality::High;
        let a = AtmosphereGpu::build(&p, &noon, 2.0, q);
        let b = AtmosphereGpu::build(&p, &dusk, 2.0, q);
        assert!(a.medium_key() == b.medium_key(), "sun moved the medium key");
        assert!(
            a.skyview_key() != b.skyview_key(),
            "sun missed the view key"
        );

        // Altitude: sub-metre jitter must NOT re-bake; a real climb must.
        let c = AtmosphereGpu::build(&p, &noon, 2.4, q);
        let d = AtmosphereGpu::build(&p, &noon, 250.0, q);
        assert!(
            a.skyview_key() == c.skyview_key(),
            "sub-metre jitter re-baked"
        );
        assert!(
            a.skyview_key() != d.skyview_key(),
            "a 248 m climb was ignored"
        );

        // Turbidity is a medium change, so both LUTs must fall out of date.
        let hazy = AtmosphereGpu::build(
            &AtmosphereParams {
                turbidity: 3.0,
                ..p
            },
            &noon,
            2.0,
            q,
        );
        assert!(a.medium_key() != hazy.medium_key());
        assert!(a.skyview_key() != hazy.skyview_key());

        // Fog is a *receiver* parameter: it changes neither LUT.
        let foggy = AtmosphereGpu::build(
            &AtmosphereParams {
                fog: HeightFog {
                    density: 1e-3,
                    ..HeightFog::default()
                },
                ..p
            },
            &noon,
            2.0,
            q,
        );
        assert!(a.medium_key() == foggy.medium_key());
        assert!(a.skyview_key() == foggy.skyview_key());
        assert_ne!(a, foggy, "the uniform itself must still change");
    }

    /// Stars appear only after the sun is down, and are fully out by the end of
    /// civil twilight.
    #[test]
    fn night_fade_matches_civil_twilight() {
        assert_eq!(night_fade(1.0), 0.0);
        assert_eq!(night_fade(0.0), 0.0);
        assert_eq!(night_fade(-1.0), 1.0);
        let dusk = night_fade(-0.06); // half way
        assert!(dusk > 0.4 && dusk < 0.6, "{dusk}");
        // Monotone.
        let mut prev = 0.0;
        for i in 0..=64 {
            let y = -(i as f32) / 64.0 * 0.2;
            let f = night_fade(y);
            assert!(f + 1e-6 >= prev, "not monotone at {y}");
            prev = f;
        }
    }

    /// The uniform is a pure function of its inputs — the CPU half of the LUT
    /// determinism gate (the GPU half reads the baked texels back).
    #[test]
    fn uniform_is_deterministic() {
        let p = enabled();
        let s = SunParams::default();
        let a = AtmosphereGpu::build(&p, &s, 12.5, AtmosphereQuality::Medium);
        let b = AtmosphereGpu::build(&p, &s, 12.5, AtmosphereQuality::Medium);
        assert_eq!(bytemuck::bytes_of(&a), bytemuck::bytes_of(&b));
    }

    /// Nonsense from a projector must not reach the GPU as `NaN`/`inf`.
    #[test]
    fn degenerate_input_stays_finite() {
        let p = AtmosphereParams {
            enabled: true,
            sky_intensity: f32::NAN,
            rayleigh_height: 0.0,
            mie_height: -1.0,
            mie_g: 12.0,
            ozone_half_width: 0.0,
            turbidity: 1e30,
            ..AtmosphereParams::default()
        };
        let s = SunParams {
            direction: glam::Vec3::ZERO,
            intensity: -5.0,
            ..SunParams::default()
        };
        let g = AtmosphereGpu::build(&p, &s, f32::INFINITY, AtmosphereQuality::Low);
        assert!(g.rayleigh[3] > 0.0 && g.mie[2] > 0.0 && g.ozone_shape[1] > 0.0);
        assert!((-0.95..=0.95).contains(&g.mie[3]));
        assert!(g.planet[2].is_finite() && g.planet[2] > g.planet[0]);
        assert!(g.sun_dir[0].is_finite() && g.sun_color[0] >= 0.0);
        // A NaN exposure is the one thing that would poison the whole HDR buffer —
        // so assert it cannot BE one. `f32::max` returns the non-NaN operand, so
        // `NaN.max(0.0)` is 0.0 and the NaN never reaches the uniform. That is a
        // real property of the clamp, not a hope about it.
        assert!(
            g.params[1].is_finite() && g.params[1] >= 0.0,
            "a NaN sky intensity reached the GPU: {}",
            g.params[1]
        );
    }
}
