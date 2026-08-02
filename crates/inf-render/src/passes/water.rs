//! Water surfaces (P20.1) — the pass that draws oceans, lakes and spline rivers.
//!
//! ## Where it sits, and why
//!
//! After every opaque pass (mesh, vgeom, skinned, terrain, scatter) and after the
//! clouds and precipitation; **before** the translucent pass. The reasoning is
//! the [`super::cloud`] placement argument applied to a surface rather than a
//! medium:
//!
//! * **after opaque**, because every interesting thing water does is a function
//!   of *what is behind it* — the Beer-Lambert column, the shore fade and the
//!   refraction all read the scene depth, and there is nothing to read before the
//!   opaque passes have written it;
//! * **after clouds and rain**, so a sea reflects and is veiled by the sky it is
//!   under rather than the other way round;
//! * **before translucency**, so glass composites over water exactly as it does
//!   over any other surface. Water is *not* in the translucent pass because that
//!   pass is a back-to-front sort of `MeshInstance`s through the `mesh` shader,
//!   and water is neither — it is one procedural surface per body with its own
//!   pipeline, its own vertex generation, and a fragment stage that reads the
//!   frame buffer. Putting it in the sort would mean either giving the sort a
//!   second shader path or pretending a river is a mesh.
//!
//! ## Two render passes, and why the first one exists
//!
//! Screen-space refraction needs the scene *colour* behind the water, and the
//! colour target being rendered into is the MSAA one — which cannot be sampled
//! while it is an attachment. So the node first records a **resolve-only pass**:
//! a render pass whose colour attachment is `color_msaa` with `scene_hdr` as its
//! resolve target and no draws at all. wgpu performs the MSAA resolve at pass
//! end regardless of what was drawn, so that one empty pass hands the water pass
//! a single-sample copy of everything behind the water. `scene_hdr` is
//! overwritten by the real [`super::resolve`] node later in the frame, so
//! nothing downstream sees the intermediate.
//!
//! It costs one resolve, and only on frames that actually carry water — and only
//! at a quality tier that pays for refraction at all
//! ([`WaterQuality::refraction`]).
//!
//! ## Depth
//!
//! The depth attachment is bound **read-only** (`depth_ops: None`) and the same
//! view is additionally bound as a `texture_depth_multisampled_2d` — the P17.3
//! arrangement, and the only one WebGPU permits for reading an attachment you are
//! also testing against. The hardware test rejects water behind geometry per MSAA
//! sample; the sampled copy is what the shader measures the water column with.
//!
//! ## Off path
//!
//! No water in the scene ⇒ `run` returns **before touching the encoder**: no
//! resolve, no render pass, no pipeline bind, no draw. Every pre-P20.1 golden
//! therefore renders the exact command stream it did before, which is what
//! `water_off_path_is_byte_identical` pins from the outside and
//! `a_scene_without_water_records_nothing` from the inside.

use bytemuck::{Pod, Zeroable};
use glam::DVec3;

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::water::{
    RenderWater, WaterKindGpu, WaterQuality, MAX_WAVES, OCEAN_EXTENT_M, OCEAN_SNAP_M,
};

/// Stride between per-body uniform slots, bytes.
///
/// 256 is the WebGPU guaranteed maximum for
/// `min_uniform_buffer_offset_alignment`, so any multiple of it is a legal
/// dynamic offset on every adapter without querying. The uniform itself is 448
/// bytes (12 header vec4s + 8 waves x 2 vec4s), so the stride is **512** — the
/// next multiple. `the_uniform_matches_the_shader_struct` pins the pair, because
/// a uniform that outgrew its stride would silently overlap the next body.
const UNIFORM_STRIDE: u64 = 512;

/// Bodies drawn per frame at most. A level with more water than this is not an
/// error — the extra bodies are skipped, deterministically, in projection order.
/// Sized so the uniform buffer stays a single 16 KiB allocation.
const MAX_BODIES: usize = 32;

/// Where the shader reads the water column when the depth buffer says there is
/// nothing behind the surface (the sea meeting the sky at the horizon), metres.
/// Deep enough that the absorption has saturated, so the far ocean is the
/// authored deep colour rather than a bright band.
const OPEN_WATER_DEPTH_M: f32 = 60.0;

/// The per-body uniform. Mirrors `struct Water` in `water.wgsl`; the pair is
/// pinned by `the_uniform_matches_the_shader_struct`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WaterUniform {
    dims: [u32; 4],
    flags: [u32; 4],
    center_extent: [f32; 4],
    level_flow: [f32; 4],
    shallow: [f32; 4],
    deep: [f32; 4],
    absorption: [f32; 4],
    foam: [f32; 4],
    foam2: [f32; 4],
    zenith: [f32; 4],
    horizon: [f32; 4],
    ground: [f32; 4],
    waves: [[f32; 4]; MAX_WAVES * 2],
}

/// One centreline frame as the storage buffer carries it. Mirrors
/// `struct WaterFrame` in `water.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct FrameGpu {
    center_width: [f32; 4],
    tangent_depth: [f32; 4],
    right_s: [f32; 4],
}

/// Vertex/index counts for a grid of `n` vertices per side.
fn grid_counts(n: u32) -> (u32, u32) {
    let quads = n.saturating_sub(1);
    (n * n, quads * quads * 6)
}

/// The triangle indices of an `n × n` vertex grid, row-major. Pure, so the same
/// tier always produces the same buffer.
fn grid_indices(n: u32) -> Vec<u32> {
    let quads = n.saturating_sub(1);
    let mut idx = Vec::with_capacity((quads * quads * 6) as usize);
    for z in 0..quads {
        for x in 0..quads {
            let a = z * n + x;
            let b = a + 1;
            let c = a + n;
            let d = c + 1;
            idx.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    idx
}

pub struct WaterNode {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// Per-body uniforms, one [`UNIFORM_STRIDE`] slot each.
    uniforms: wgpu::Buffer,
    /// All rivers' frames, concatenated. Grown on demand; never shrunk.
    frames: wgpu::Buffer,
    frames_capacity: usize,
    /// The shared grid index buffer, rebuilt when the quality tier changes.
    indices: Option<(WaterQuality, wgpu::Buffer, u32)>,
    /// Keyed on the full [`super::ResourceKey`] **and the frames buffer's
    /// generation**: the bind group embeds the scene depth and colour views
    /// (recreated on every resize) and the frames storage buffer (recreated when
    /// a level's rivers outgrow it).
    bind_group: super::GenCache<(super::ResourceKey, u64), wgpu::BindGroup>,
    /// Monotonic source of the frames-buffer half of the cache key.
    frames_generation: u64,
}

impl WaterNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("water"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("water").into()),
            });

        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let frag = wgpu::ShaderStages::FRAGMENT;
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("water"),
                entries: &[
                    // 0..3 = the shared atmosphere uniform + LUTs (the reflection
                    // source and the aerial-perspective term).
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: vf,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: frag,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // 4 = the per-body uniform, addressed by a dynamic offset so
                    // every body in the frame shares one buffer and one bind group.
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: vf,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 5 = every river's centreline frames, concatenated.
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: vf,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // 6 = the scene depth, read back to measure the water column.
                    // Multisampled and unfilterable: `textureLoad`ed at an integer
                    // texel, never sampled.
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    },
                    // 7/8 = the resolved scene colour + its sampler (refraction).
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: frag,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("water"),
                bind_group_layouts: &[Some(view_bgl), Some(&bgl)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("water"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    // No vertex buffers: the grid comes from `vertex_index`.
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        // Straight alpha: the shader returns a finished surface
                        // colour and a coverage (shore fade + foam), not a
                        // premultiplied radiance like the cloud march.
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    // Two-sided: a Gerstner trough seen from a low camera, and a
                    // river bank seen from the far side, both present back faces
                    // — and water has no inside to cull away.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    // Test but never write: the water is transparent, and writing
                    // its depth would make the translucent pass behind it think
                    // there is opaque geometry here.
                    depth_write_enabled: Some(false),
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

        let uniforms = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water-uniforms"),
            size: UNIFORM_STRIDE * MAX_BODIES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // A one-element placeholder so the bind group is valid on a frame with no
        // rivers (a zero-sized storage binding is a validation error).
        let frames = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water-frames"),
            size: std::mem::size_of::<FrameGpu>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("water-scene"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline,
            bgl,
            sampler,
            uniforms,
            frames,
            frames_capacity: 1,
            indices: None,
            bind_group: super::GenCache::default(),
            frames_generation: 1,
        }
    }

    /// The shared grid index buffer for `quality`, rebuilt when the tier moves.
    fn indices(&mut self, gpu: &GpuContext, quality: WaterQuality) -> (&wgpu::Buffer, u32) {
        let needs = self.indices.as_ref().is_none_or(|(q, _, _)| *q != quality);
        if needs {
            let n = quality.grid_vertices();
            let idx = grid_indices(n);
            let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("water-indices"),
                size: (idx.len() * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&idx));
            self.indices = Some((quality, buf, idx.len() as u32));
        }
        let (_, buf, len) = self.indices.as_ref().unwrap();
        (buf, *len)
    }
}

/// Pack one body's uniform. Split out of `run` so it is testable without a GPU.
fn pack_uniform(
    w: &RenderWater,
    origin: &inf_math::FloatingOrigin,
    eye: glam::Vec3,
    quality: WaterQuality,
    frame_offset: u32,
    sky: &crate::scene::SkyParams,
) -> WaterUniform {
    // The ocean's patch follows the camera, snapped so the tessellation does not
    // crawl; a lake's is the authored region. Both are converted to render-local.
    let (center_xz, half_extent) = match w.kind {
        WaterKindGpu::Ocean => {
            let world_eye = origin.to_world(eye);
            let cx = crate::water::snap(world_eye.x, OCEAN_SNAP_M);
            let cz = crate::water::snap(world_eye.z, OCEAN_SNAP_M);
            let local = origin.to_render(DVec3::new(cx, w.level_m, cz));
            (
                [local.x, local.z],
                [OCEAN_EXTENT_M as f32, OCEAN_EXTENT_M as f32],
            )
        }
        _ => {
            let local = origin.to_render(DVec3::new(w.center.x, w.level_m, w.center.y));
            (
                [local.x, local.z],
                [w.half_extent.x as f32, w.half_extent.y as f32],
            )
        }
    };
    let level_local = origin.to_render(DVec3::new(0.0, w.level_m, 0.0)).y;

    // The wave-space origin. An ocean/lake evaluates in world XZ, so the
    // floating origin is folded into every wave's reduced phase and the shader
    // can work in render-local coordinates. A river evaluates in its own
    // arc-length frame, which is origin-independent already.
    let wave_origin = match w.kind {
        WaterKindGpu::River => glam::DVec2::ZERO,
        _ => glam::DVec2::new(origin.origin().x, origin.origin().z),
    };

    let mut waves = [[0.0f32; 4]; MAX_WAVES * 2];
    for (i, wave) in w.waves.waves().iter().enumerate() {
        waves[2 * i] = [
            wave.dir.x as f32,
            wave.dir.y as f32,
            wave.amplitude_m as f32,
            wave.wavenumber as f32,
        ];
        waves[2 * i + 1] = [
            wave.steepness as f32,
            wave.phase_at(w.time_s, wave_origin) as f32,
            0.0,
            0.0,
        ];
    }

    WaterUniform {
        dims: [
            w.kind.code(),
            w.waves.waves().len() as u32,
            quality.grid_vertices(),
            w.frames.len() as u32,
        ],
        flags: [frame_offset, u32::from(quality.refraction()), 0, 0],
        center_extent: [center_xz[0], center_xz[1], half_extent[0], half_extent[1]],
        level_flow: [
            level_local,
            w.flow_speed_m_s as f32,
            w.shore_fade_m,
            w.opacity,
        ],
        shallow: [
            w.shallow_color[0],
            w.shallow_color[1],
            w.shallow_color[2],
            w.roughness,
        ],
        deep: [
            w.deep_color[0],
            w.deep_color[1],
            w.deep_color[2],
            w.refraction_m,
        ],
        absorption: [
            w.absorption[0],
            w.absorption[1],
            w.absorption[2],
            OPEN_WATER_DEPTH_M,
        ],
        foam: [
            w.foam_color[0],
            w.foam_color[1],
            w.foam_color[2],
            w.foam_crest_threshold,
        ],
        foam2: [w.foam_shore_m, w.foam_flow_m_s, 0.0, 0.0],
        zenith: [sky.zenith[0], sky.zenith[1], sky.zenith[2], 0.0],
        horizon: [sky.horizon[0], sky.horizon[1], sky.horizon[2], 0.0],
        ground: [sky.ground[0], sky.ground[1], sky.ground[2], 0.0],
        waves,
    }
}

/// Pack a river's frames, rebased against the floating origin.
fn pack_frames(w: &RenderWater, origin: &inf_math::FloatingOrigin, out: &mut Vec<FrameGpu>) {
    for f in &w.frames {
        let c = origin.to_render(f.center);
        out.push(FrameGpu {
            center_width: [c.x, c.y, c.z, f.width_m as f32],
            tangent_depth: [
                f.tangent.x as f32,
                f.tangent.y as f32,
                f.tangent.z as f32,
                f.depth_m as f32,
            ],
            right_s: [
                f.right.x as f32,
                f.right.y as f32,
                f.right.z as f32,
                f.s as f32,
            ],
        });
    }
}

impl RenderNode for WaterNode {
    fn name(&self) -> &'static str {
        "water"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        // OFF PATH: no water ⇒ the encoder is not touched at all.
        let bodies: Vec<&RenderWater> = frame
            .scene
            .waters
            .iter()
            .filter(|w| w.drawable())
            .take(MAX_BODIES)
            .collect();
        if bodies.is_empty() {
            return;
        }

        let quality = frame.settings.water.quality;
        let (verts, _) = grid_counts(quality.grid_vertices());
        debug_assert!(verts > 0);

        // ── frames storage ────────────────────────────────────────────────
        let mut frames: Vec<FrameGpu> = Vec::new();
        let mut offsets: Vec<u32> = Vec::with_capacity(bodies.len());
        for w in &bodies {
            offsets.push(frames.len() as u32);
            if w.kind == WaterKindGpu::River {
                pack_frames(w, &frame.view.origin, &mut frames);
            }
        }
        if frames.is_empty() {
            // A zero-sized storage binding is a validation error, and a frame
            // with only oceans and lakes legitimately has no centrelines.
            frames.push(FrameGpu::default());
        }
        if frames.len() > self.frames_capacity {
            let capacity = frames.len().next_power_of_two().max(64);
            self.frames = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("water-frames"),
                size: (capacity * std::mem::size_of::<FrameGpu>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.frames_capacity = capacity;
            // The bind group embeds this buffer; a new allocation must invalidate
            // it, or the pass would keep reading the old (smaller) one.
            self.frames_generation += 1;
        }
        gpu.queue
            .write_buffer(&self.frames, 0, bytemuck::cast_slice(&frames));

        // ── per-body uniforms ─────────────────────────────────────────────
        let eye = frame.view.eye_local();
        for (i, w) in bodies.iter().enumerate() {
            let u = pack_uniform(
                w,
                &frame.view.origin,
                eye,
                quality,
                offsets[i],
                &frame.scene.sky,
            );
            gpu.queue.write_buffer(
                &self.uniforms,
                i as u64 * UNIFORM_STRIDE,
                bytemuck::bytes_of(&u),
            );
        }

        // ── the refraction source ─────────────────────────────────────────
        //
        // One resolve-only pass: `color_msaa` in, `scene_hdr` out, no draws. See
        // the module docs. Skipped entirely at a tier that does not refract, so
        // Low pays nothing for a feature it does not use.
        if quality.refraction() {
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("water-refraction-resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame.targets.color_msaa,
                    resolve_target: Some(&frame.targets.scene_hdr),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        let atmos = frame.atmosphere;
        let bgl = &self.bgl;
        let sampler = &self.sampler;
        let uniforms = &self.uniforms;
        let frames_buf = &self.frames;
        let key = (super::resource_key(frame), self.frames_generation);
        let bind_group = self
            .bind_group
            .get_or_build(key, || {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("water"),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: atmos.uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&atmos.transmittance),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&atmos.sky_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Sampler(&atmos.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: uniforms,
                                offset: 0,
                                size: std::num::NonZeroU64::new(
                                    std::mem::size_of::<WaterUniform>() as u64,
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: frames_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&frame.targets.depth),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(&frame.targets.scene_hdr),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                    ],
                })
            })
            .clone();

        let (index_buf, index_count) = self.indices(gpu, quality);
        let index_buf = index_buf.clone();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("water"),
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
                // `None` = READ-ONLY depth: the hardware test still runs, the
                // contents survive for the passes that follow, and the same view
                // may simultaneously be bound as a sampled texture. The one
                // arrangement WebGPU permits (the P17.3 precedent).
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
        for i in 0..bodies.len() {
            pass.set_bind_group(1, &bind_group, &[i as u32 * UNIFORM_STRIDE as u32]);
            pass.draw_indexed(0..index_count, 0, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SkyParams;
    use crate::water::{WaterFrame, WaveField, WaveSpec};
    use glam::DVec2;
    use inf_math::FloatingOrigin;

    fn ocean() -> RenderWater {
        RenderWater {
            level_m: 3.0,
            waves: WaveField::from_spec(&WaveSpec::default()),
            time_s: 1_234.5,
            ..RenderWater::default()
        }
    }

    #[test]
    fn the_uniform_matches_the_shader_struct() {
        // `struct Water` in water.wgsl, field by field: 12 vec4 headers + 16 wave
        // vec4s. A mismatch here is silent on the GPU — the shader simply reads
        // the wrong 16 bytes for every field after the divergence.
        assert_eq!(std::mem::size_of::<WaterUniform>(), (12 + 16) * 16);
        assert_eq!(std::mem::align_of::<WaterUniform>(), 4);
        // …and it fits inside one dynamic-offset slot, which is what lets every
        // body in a frame share one buffer and one bind group.
        assert!(std::mem::size_of::<WaterUniform>() as u64 <= UNIFORM_STRIDE);
        // The frame record likewise: three vec4s.
        assert_eq!(std::mem::size_of::<FrameGpu>(), 3 * 16);
    }

    #[test]
    fn the_grid_index_buffer_is_well_formed() {
        for q in [WaterQuality::Low, WaterQuality::Medium, WaterQuality::High] {
            let n = q.grid_vertices();
            let (verts, tris) = grid_counts(n);
            let idx = grid_indices(n);
            assert_eq!(idx.len() as u32, tris, "{q:?}");
            assert!(idx.iter().all(|&i| i < verts), "{q:?}: index out of range");
            // Every quad contributes two triangles and every vertex is reachable.
            let mut seen = vec![false; verts as usize];
            for &i in &idx {
                seen[i as usize] = true;
            }
            assert!(seen.iter().all(|&s| s), "{q:?}: an unreferenced vertex");
        }
        // A degenerate grid produces nothing rather than panicking.
        assert!(grid_indices(1).is_empty());
        assert!(grid_indices(0).is_empty());
    }

    /// The ocean patch follows the camera but is **snapped**, so the vertex set is
    /// piecewise-constant in camera position. That is what stops the tessellation
    /// crawling under a moving camera.
    #[test]
    fn the_ocean_patch_snaps_to_the_camera() {
        let origin = FloatingOrigin::default();
        let sky = SkyParams::default();
        let pack = |eye_x: f32| {
            pack_uniform(
                &ocean(),
                &origin,
                glam::Vec3::new(eye_x, 20.0, 0.0),
                WaterQuality::High,
                0,
                &sky,
            )
            .center_extent[0]
        };
        // Two camera positions inside one snap cell give the SAME patch centre …
        let step = OCEAN_SNAP_M as f32;
        assert_eq!(pack(100.0), pack(100.0 + step * 0.5));
        // … and crossing the cell boundary moves it by exactly one cell.
        let a = pack(100.0);
        let b = pack(100.0 + step);
        assert!(((b - a) as f64 - OCEAN_SNAP_M).abs() < 1e-3, "{a} -> {b}");
    }

    /// The floating origin must move **no wave**: it is folded into each
    /// component's reduced phase, so the shader's render-local evaluation still
    /// lands on the world-space phase.
    #[test]
    fn a_rebase_moves_no_wave() {
        let sky = SkyParams::default();
        let w = ocean();
        let a = FloatingOrigin::default();
        let b = FloatingOrigin::new(DVec3::new(5_000.0, 0.0, -3_000.0));
        assert_ne!(a.origin(), b.origin(), "the rebase did nothing");

        // Pick a world point and its render-local image under each origin, then
        // check the phase the shader would evaluate is the same angle.
        let world = DVec3::new(4_990.0, 0.0, -2_970.0);
        let ua = pack_uniform(&w, &a, glam::Vec3::ZERO, WaterQuality::High, 0, &sky);
        let ub = pack_uniform(&w, &b, glam::Vec3::ZERO, WaterQuality::High, 0, &sky);
        let la = a.to_render(world);
        let lb = b.to_render(world);
        for i in 0..w.waves.waves().len() {
            let dir = [ua.waves[2 * i][0], ua.waves[2 * i][1]];
            let k = ua.waves[2 * i][3];
            let ta = k * (dir[0] * la.x + dir[1] * la.z) + ua.waves[2 * i + 1][1];
            let tb = k * (dir[0] * lb.x + dir[1] * lb.z) + ub.waves[2 * i + 1][1];
            // Same angle modulo 2π — the *displacement* is what must agree.
            assert!(
                (ta.sin() - tb.sin()).abs() < 2e-3 && (ta.cos() - tb.cos()).abs() < 2e-3,
                "wave {i}: θ {ta} vs {tb} after a rebase"
            );
        }
    }

    #[test]
    fn frames_are_rebased_and_packed_in_order() {
        let origin = FloatingOrigin::new(DVec3::new(1_000.0, 0.0, 0.0));
        let river = RenderWater {
            kind: WaterKindGpu::River,
            frames: (0..4)
                .map(|i| WaterFrame {
                    center: DVec3::new(1_000.0 + i as f64 * 10.0, 5.0, 0.0),
                    tangent: DVec3::X,
                    right: DVec3::Z,
                    s: i as f64 * 10.0,
                    width_m: 6.0,
                    depth_m: 1.5,
                })
                .collect(),
            ..RenderWater::default()
        };
        let mut out = Vec::new();
        pack_frames(&river, &origin, &mut out);
        assert_eq!(out.len(), 4);
        // Rebased: the first frame sits at the origin, not at x = 1000.
        assert!(
            out[0].center_width[0].abs() < 1.0,
            "{}",
            out[0].center_width[0]
        );
        // Arc length is a *distance* and is origin-independent.
        for (i, f) in out.iter().enumerate() {
            assert_eq!(f.right_s[3], i as f32 * 10.0);
            assert_eq!(f.center_width[3], 6.0);
            assert_eq!(f.tangent_depth[3], 1.5);
        }
    }

    #[test]
    fn a_lake_uses_its_authored_region_not_the_camera() {
        let sky = SkyParams::default();
        let lake = RenderWater {
            kind: WaterKindGpu::Lake,
            center: DVec2::new(100.0, -50.0),
            half_extent: DVec2::new(30.0, 20.0),
            level_m: 12.0,
            ..RenderWater::default()
        };
        let origin = FloatingOrigin::default();
        let a = pack_uniform(
            &lake,
            &origin,
            glam::Vec3::ZERO,
            WaterQuality::High,
            0,
            &sky,
        );
        let b = pack_uniform(
            &lake,
            &origin,
            glam::Vec3::new(9_000.0, 0.0, 9_000.0),
            WaterQuality::High,
            0,
            &sky,
        );
        assert_eq!(
            a.center_extent, b.center_extent,
            "a lake moved with the camera"
        );
        assert_eq!(a.center_extent[2], 30.0);
        assert_eq!(a.center_extent[3], 20.0);
        assert_eq!(a.dims[0], WaterKindGpu::Lake.code());
    }

    #[test]
    fn the_refraction_flag_follows_the_quality_tier() {
        let sky = SkyParams::default();
        let origin = FloatingOrigin::default();
        for (q, want) in [
            (WaterQuality::Low, 0u32),
            (WaterQuality::Medium, 1),
            (WaterQuality::High, 1),
        ] {
            let u = pack_uniform(&ocean(), &origin, glam::Vec3::ZERO, q, 0, &sky);
            assert_eq!(u.flags[1], want, "{q:?}");
            assert_eq!(u.dims[2], q.grid_vertices());
        }
    }

    #[test]
    fn packing_is_deterministic() {
        let sky = SkyParams::default();
        let origin = FloatingOrigin::default();
        let a = pack_uniform(
            &ocean(),
            &origin,
            glam::Vec3::ONE,
            WaterQuality::High,
            0,
            &sky,
        );
        let b = pack_uniform(
            &ocean(),
            &origin,
            glam::Vec3::ONE,
            WaterQuality::High,
            0,
            &sky,
        );
        assert_eq!(bytemuck::bytes_of(&a), bytemuck::bytes_of(&b));
    }
}
