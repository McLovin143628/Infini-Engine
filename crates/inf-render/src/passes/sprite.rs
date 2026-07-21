//! 2D sprite pass (P8.1a): the GPU half of the sprite pipeline. The CPU batcher
//! and data types live in `inf-render-2d` (see that crate's docs for the seam
//! rationale); this module owns the wgpu texture cache, the instance buffer, and
//! the render-graph node.
//!
//! Draws after the 3D meshes into the MSAA scene target, depth-tested against
//! the scene depth (reverse-Z) but **not** depth-writing, so sprite-vs-sprite
//! ordering is the painter order the batcher produced. Straight
//! (non-premultiplied) alpha.

use std::collections::BTreeMap;

use inf_math::FloatingOrigin;
use inf_render_2d::{
    aabb_visible, batch_scene, builtin_font_rgba8, chunk_world_aabb, expand_chunk, PrebatchedRun,
    SpriteBatch, SpriteInstance, TextureHandle, BUILTIN_FONT_TEXTURE,
};

use crate::camera::{RenderView, DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};
use crate::scene::{RenderScene, SpriteTextureUpload};

/// Per-instance GPU data; matches the `@location(0..=5)` attributes in
/// sprite.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpriteRaw {
    /// xyz = render-local pivot position (origin-relative), w unused.
    center: [f32; 4],
    /// x,y = size; z = rotation radians; w unused.
    size_rot: [f32; 4],
    /// x,y = pivot; zw unused.
    pivot: [f32; 4],
    /// xy = uv_min, zw = uv_max.
    uv: [f32; 4],
    color: [f32; 4],
    /// x = flip_x, y = flip_y (0/1).
    flags: [u32; 4],
}

impl SpriteRaw {
    fn pack(origin: &FloatingOrigin, s: &SpriteInstance) -> Self {
        let c = origin.to_render(s.position);
        Self {
            center: [c.x, c.y, c.z, 0.0],
            size_rot: [s.size.x, s.size.y, s.rotation, 0.0],
            pivot: [s.pivot.x, s.pivot.y, 0.0, 0.0],
            uv: [s.uv_min.x, s.uv_min.y, s.uv_max.x, s.uv_max.y],
            color: s.color,
            flags: [s.flip_x as u32, s.flip_y as u32, 0, 0],
        }
    }
}

const SPRITE_ATTRIBUTES: [wgpu::VertexAttribute; 6] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 16,
        shader_location: 1,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 32,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 48,
        shader_location: 3,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 64,
        shader_location: 4,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x4,
        offset: 80,
        shader_location: 5,
    },
];

/// Max 2D lights uploaded per frame (must match `MAX_2D_LIGHTS` in sprite.wgsl).
pub const MAX_2D_LIGHTS: usize = 16;

/// One GPU 2D light, std140-friendly (all vec4). `color.a` = intensity;
/// `pos.xy` = render-local position, `pos.z` = falloff radius.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLight2D {
    color: [f32; 4],
    pos: [f32; 4],
}

/// The 2D lights uniform block bound at `@group(2)` of the sprite pipeline.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Lights2DUniform {
    count: [u32; 4],   // x = active count
    ambient: [f32; 4], // rgb = scene ambient
    items: [GpuLight2D; MAX_2D_LIGHTS],
}

impl Lights2DUniform {
    /// Project world-space 2D lights into render-local GPU lights + ambient.
    fn from_scene(scene: &RenderScene, origin: &FloatingOrigin) -> Self {
        let mut items = [GpuLight2D {
            color: [0.0; 4],
            pos: [0.0; 4],
        }; MAX_2D_LIGHTS];
        let count = scene.lights_2d.len().min(MAX_2D_LIGHTS);
        for (slot, light) in items.iter_mut().zip(scene.lights_2d.iter()).take(count) {
            slot.color = [
                light.color[0],
                light.color[1],
                light.color[2],
                light.intensity,
            ];
            // Render-local position (origin-relative), like 3D point lights.
            let p = (light.position - origin.origin()).as_vec3();
            slot.pos = [p.x, p.y, light.radius, 0.0];
        }
        let a = scene.ambient_2d.0;
        Self {
            count: [count as u32, 0, 0, 0],
            ambient: [a[0], a[1], a[2], 0.0],
            items,
        }
    }

    /// The default (no lights, white ambient) block — what the buffer holds
    /// before the first sync so a sprite drawn on frame 0 is still unlit-correct.
    fn white_ambient() -> Self {
        Self {
            count: [0; 4],
            ambient: [1.0, 1.0, 1.0, 0.0],
            items: [GpuLight2D {
                color: [0.0; 4],
                pos: [0.0; 4],
            }; MAX_2D_LIGHTS],
        }
    }
}

/// A single cached GPU texture and its (texture+sampler) bind group.
struct TextureEntry {
    // Held to keep the texture (and its views) alive; the bind group borrows it.
    _texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

/// GPU texture cache keyed by [`TextureHandle`] (a `u64`). Eviction-free for now
/// (a documented follow-up): textures live for the renderer's lifetime. Uploads
/// RGBA8 with a full CPU-generated box mip chain; a 1×1 white fallback backs
/// missing handles.
pub struct SpriteTextures {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    fallback: TextureEntry,
    // BTreeMap → deterministic iteration (no HashMap order leaking anywhere).
    map: BTreeMap<TextureHandle, TextureEntry>,
}

impl SpriteTextures {
    fn new(gpu: &GpuContext, layout: wgpu::BindGroupLayout) -> Self {
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sprite-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // 1×1 opaque white fallback (sRGB target so sampling yields linear 1.0).
        let fallback = upload_rgba8(gpu, &layout, &sampler, 1, 1, &[255, 255, 255, 255]);
        let mut map = BTreeMap::new();
        // Built-in bitmap font, always available under its reserved handle so a
        // `Text2D` with no font asset renders with zero setup.
        let (font_rgba, fw, fh) = builtin_font_rgba8();
        map.insert(
            BUILTIN_FONT_TEXTURE,
            upload_rgba8(gpu, &layout, &sampler, fw, fh, &font_rgba),
        );
        Self {
            layout,
            sampler,
            fallback,
            map,
        }
    }

    /// Upload one texture if its handle isn't cached yet (dedup by handle).
    fn ingest(&mut self, gpu: &GpuContext, up: &SpriteTextureUpload) {
        if up.handle == inf_render_2d::WHITE_TEXTURE || self.map.contains_key(&up.handle) {
            return;
        }
        // Guard against a malformed upload rather than panicking in a frame.
        if up.width == 0 || up.height == 0 || up.rgba8.len() < (up.width * up.height * 4) as usize {
            tracing::warn!(
                handle = up.handle,
                "sprite: skipping malformed texture upload"
            );
            return;
        }
        let entry = upload_rgba8(
            gpu,
            &self.layout,
            &self.sampler,
            up.width,
            up.height,
            &up.rgba8,
        );
        self.map.insert(up.handle, entry);
    }

    /// The bind group for `handle`, falling back to the white texture when the
    /// handle is unknown (or is the [`WHITE_TEXTURE`](inf_render_2d::WHITE_TEXTURE)
    /// sentinel).
    fn bind_group(&self, handle: TextureHandle) -> &wgpu::BindGroup {
        self.map
            .get(&handle)
            .map(|e| &e.bind_group)
            .unwrap_or(&self.fallback.bind_group)
    }
}

/// Create + upload an RGBA8 (sRGB) texture with a full box mip chain and its
/// texture+sampler bind group.
fn upload_rgba8(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> TextureEntry {
    let mip_level_count = mip_levels(width, height);
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sprite-texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Base color is sRGB-encoded; sampling returns linear (matches mesh albedo).
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Upload level 0, then CPU-box-downsample each subsequent level.
    let mut level_data = rgba.to_vec();
    let (mut lw, mut lh) = (width, height);
    for level in 0..mip_level_count {
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &level_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(lw * 4),
                rows_per_image: Some(lh),
            },
            wgpu::Extent3d {
                width: lw,
                height: lh,
                depth_or_array_layers: 1,
            },
        );
        if level + 1 < mip_level_count {
            let (nw, nh) = ((lw / 2).max(1), (lh / 2).max(1));
            level_data = downsample_box(&level_data, lw, lh, nw, nh);
            lw = nw;
            lh = nh;
        }
    }

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sprite-texture"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    TextureEntry {
        _texture: texture,
        bind_group,
    }
}

fn mip_levels(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

/// 2×2 box-filter downsample of an RGBA8 image (clamps odd dimensions).
fn downsample_box(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut out = vec![0u8; (dw * dh * 4) as usize];
    for y in 0..dh {
        for x in 0..dw {
            let sx0 = (x * 2).min(sw - 1);
            let sy0 = (y * 2).min(sh - 1);
            let sx1 = (x * 2 + 1).min(sw - 1);
            let sy1 = (y * 2 + 1).min(sh - 1);
            let mut acc = [0u32; 4];
            for &(sx, sy) in &[(sx0, sy0), (sx1, sy0), (sx0, sy1), (sx1, sy1)] {
                let i = ((sy * sw + sx) * 4) as usize;
                for (c, a) in acc.iter_mut().enumerate() {
                    *a += src[i + c] as u32;
                }
            }
            let o = ((y * dw + x) * 4) as usize;
            for (c, a) in acc.iter().enumerate() {
                out[o + c] = (a / 4) as u8;
            }
        }
    }
    out
}

pub struct SpriteNode {
    pipeline: wgpu::RenderPipeline,
    textures: SpriteTextures,
    instances: Option<wgpu::Buffer>,
    instance_capacity: usize,
    batches: Vec<SpriteBatch>,
    /// (scene version, floating origin) the current instance buffer reflects.
    uploaded_key: Option<(u64, glam::DVec3)>,
    lights2d_buf: wgpu::Buffer,
    lights2d_bg: wgpu::BindGroup,
}

impl SpriteNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sprite"),
                source: wgpu::ShaderSource::Wgsl(
                    super::scene_shader(include_str!("../shaders/sprite.wgsl")).into(),
                ),
            });

        let tex_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sprite-texture"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // 2D lights uniform block (@group(2)).
        let lights2d_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sprite-lights2d"),
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
        let lights2d_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprite-lights2d"),
            size: std::mem::size_of::<Lights2DUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Seed with white ambient / no lights so a sprite is correct on frame 0.
        gpu.queue.write_buffer(
            &lights2d_buf,
            0,
            bytemuck::bytes_of(&Lights2DUniform::white_ambient()),
        );
        let lights2d_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sprite-lights2d"),
            layout: &lights2d_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: lights2d_buf.as_entire_binding(),
            }],
        });

        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sprite"),
                bind_group_layouts: &[Some(view_bgl), Some(&tex_bgl), Some(&lights2d_bgl)],
                immediate_size: 0,
            });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sprite"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<SpriteRaw>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &SPRITE_ATTRIBUTES,
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: SCENE_FORMAT,
                        // Straight (non-premultiplied) alpha over-blend.
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    // Depth-tested against the scene, but never writes depth
                    // (painter-sorted among sprites).
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

        Self {
            pipeline,
            textures: SpriteTextures::new(gpu, tex_bgl),
            instances: None,
            instance_capacity: 0,
            batches: Vec::new(),
            uploaded_key: None,
            lights2d_buf,
            lights2d_bg,
        }
    }

    /// Ingest any pending texture uploads (dedup by handle), then re-batch +
    /// re-pack the instance buffer when the scene changed or the origin rebased.
    ///
    /// Tilemaps are culled + expanded against the **current camera** every frame
    /// they are present (their visible-chunk set depends on the view), so the
    /// version/origin gate is bypassed whenever `scene.tilemaps` is non-empty.
    fn sync(&mut self, gpu: &GpuContext, frame: &FrameData) {
        for up in &frame.scene.pending_texture_uploads {
            self.textures.ingest(gpu, up);
        }

        let key = (frame.scene.version, frame.view.origin.origin());
        if self.uploaded_key == Some(key) && frame.scene.tilemaps.is_empty() {
            return;
        }
        self.uploaded_key = Some(key);

        // 2D lights are render-local, so they depend on the same (version,
        // origin) key; upload them whenever we re-batch.
        let lights2d = Lights2DUniform::from_scene(frame.scene, &frame.view.origin);
        gpu.queue
            .write_buffer(&self.lights2d_buf, 0, bytemuck::bytes_of(&lights2d));

        // Tilemap chunks (culled + expanded per frame) plus the host's
        // already-expanded 9-slice / text runs, all merged in one painter sort.
        let mut runs = build_tilemap_runs(frame.scene, frame.view);
        runs.extend(frame.scene.prebatched.iter().cloned());
        let batched = batch_scene(&frame.scene.sprites, &runs);
        self.batches = batched.batches;

        let raw: Vec<SpriteRaw> = batched
            .instances
            .iter()
            .map(|s| SpriteRaw::pack(&frame.view.origin, s))
            .collect();

        if !raw.is_empty() {
            if self.instances.is_none() || self.instance_capacity < raw.len() {
                let capacity = raw.len().next_power_of_two().max(64);
                self.instances = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sprite-instances"),
                    size: (capacity * std::mem::size_of::<SpriteRaw>()) as u64,
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
    }
}

impl RenderNode for SpriteNode {
    fn name(&self) -> &'static str {
        "sprite"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        self.sync(gpu, frame);
        if self.batches.is_empty() {
            return;
        }
        let Some(instances) = self.instances.as_ref() else {
            return;
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sprite"),
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
                    // Sprites do not write depth.
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
        pass.set_bind_group(2, &self.lights2d_bg, &[]);
        pass.set_vertex_buffer(0, instances.slice(..));
        for batch in &self.batches {
            pass.set_bind_group(1, self.textures.bind_group(batch.texture), &[]);
            let end = batch.start + batch.count;
            pass.draw(0..4, batch.start..end);
        }
    }
}

/// Cull each tilemap's chunks against `view` and expand the visible ones into
/// prebatched sprite runs (one run per occupied, on-screen chunk). Runs are
/// emitted in tilemap order (then chunk order), which `batch_scene` treats as
/// the deterministic tie-break for equal `(layer, order)`.
fn build_tilemap_runs(scene: &crate::scene::RenderScene, view: &RenderView) -> Vec<PrebatchedRun> {
    if scene.tilemaps.is_empty() {
        return Vec::new();
    }
    let view_proj = view.view_proj();
    let origin = &view.origin;
    let mut runs = Vec::new();
    for tm in &scene.tilemaps {
        // Margin so a chunk straddling a screen edge isn't popped early.
        let margin = glam::Vec3::splat(tm.params.tile_size.max_element());
        for chunk in &tm.chunks {
            let (wmin, wmax) = chunk_world_aabb(&tm.params, chunk.coord);
            // Floating-origin rebase to render-local (a pure translation, so it
            // preserves the min/max ordering).
            let lmin = origin.to_render(wmin) - margin;
            let lmax = origin.to_render(wmax) + margin;
            if !aabb_visible(lmin, lmax, &view_proj) {
                continue;
            }
            let mut instances = Vec::new();
            expand_chunk(&tm.params, chunk, &mut instances);
            if !instances.is_empty() {
                runs.push(PrebatchedRun {
                    texture: tm.params.texture,
                    sorting_layer: tm.params.sorting_layer,
                    order: tm.params.order,
                    instances,
                });
            }
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use inf_render_2d::SpriteInstance;

    #[test]
    fn sprite_raw_layout_offsets() {
        let v = SpriteRaw::pack(
            &FloatingOrigin::new(DVec3::ZERO),
            &SpriteInstance::default(),
        );
        let base = &v as *const SpriteRaw as usize;
        assert_eq!(std::mem::size_of::<SpriteRaw>(), 96);
        assert_eq!(&v.size_rot as *const _ as usize - base, 16);
        assert_eq!(&v.pivot as *const _ as usize - base, 32);
        assert_eq!(&v.uv as *const _ as usize - base, 48);
        assert_eq!(&v.color as *const _ as usize - base, 64);
        assert_eq!(&v.flags as *const _ as usize - base, 80);
    }

    #[test]
    fn pack_rebases_against_floating_origin() {
        let origin = FloatingOrigin::new(DVec3::new(1000.0, 0.0, 0.0));
        let raw = SpriteRaw::pack(
            &origin,
            &SpriteInstance {
                position: DVec3::new(1002.0, 3.0, -1.0),
                ..Default::default()
            },
        );
        assert_eq!(&raw.center[..3], &[2.0, 3.0, -1.0]);
    }

    #[test]
    fn lights2d_uniform_rebases_and_carries_ambient() {
        use crate::scene::{Ambient2D, RenderLight2D, RenderScene};
        let scene = RenderScene {
            ambient_2d: Ambient2D([0.2, 0.3, 0.4]),
            lights_2d: vec![RenderLight2D {
                color: [1.0, 0.5, 0.25],
                intensity: 2.0,
                radius: 3.5,
                position: DVec3::new(1002.0, 5.0, 0.0),
            }],
            ..Default::default()
        };
        let origin = FloatingOrigin::new(DVec3::new(1000.0, 0.0, 0.0));
        let u = Lights2DUniform::from_scene(&scene, &origin);
        assert_eq!(u.count[0], 1);
        assert_eq!(u.ambient, [0.2, 0.3, 0.4, 0.0]);
        // Position is origin-relative; radius packs into pos.z, intensity into color.a.
        assert_eq!(u.items[0].pos, [2.0, 5.0, 3.5, 0.0]);
        assert_eq!(u.items[0].color, [1.0, 0.5, 0.25, 2.0]);
    }

    #[test]
    fn lights2d_uniform_default_is_white_ambient() {
        use crate::scene::RenderScene;
        let scene = RenderScene::default();
        let u = Lights2DUniform::from_scene(&scene, &FloatingOrigin::new(DVec3::ZERO));
        // No lights, white ambient → sprites render exactly as pre-P8.1c.
        assert_eq!(u.count[0], 0);
        assert_eq!(u.ambient, [1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn mip_levels_counts_full_chain() {
        assert_eq!(mip_levels(1, 1), 1);
        assert_eq!(mip_levels(8, 8), 4); // 8,4,2,1
        assert_eq!(mip_levels(256, 64), 9); // driven by the larger side
    }

    #[test]
    fn build_runs_culls_offscreen_chunks_and_skips_empty() {
        use crate::scene::RenderScene;
        use inf_render_2d::{RenderChunk, RenderTilemap, TilemapParams, TILE_CHUNK_DIM};

        let view = RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: DVec3::new(0.0, 0.0, 6.0),
            forward: glam::Vec3::NEG_Z,
            up: glam::Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width: 320,
            height: 180,
        };
        let params = TilemapParams {
            origin: DVec3::new(-1.0, -1.0, 0.0),
            tile_size: glam::Vec2::new(0.1, 0.1),
            atlas_cols: 1,
            atlas_rows: 1,
            texture: 0xA1,
            color: [1.0; 4],
            sorting_layer: 0,
            order: 0,
        };
        let n = (TILE_CHUNK_DIM * TILE_CHUNK_DIM) as usize;
        // Chunk (0,0): one painted tile near the origin → on-screen.
        let mut near_tiles = vec![0u32; n];
        near_tiles[0] = 1;
        // Chunk (1000,0): painted but far off to the right → culled.
        let mut far_tiles = vec![0u32; n];
        far_tiles[0] = 1;
        // Chunk (0,1): entirely empty → produces no run even if on-screen.
        let empty_tiles = vec![0u32; n];

        let scene = RenderScene {
            tilemaps: vec![RenderTilemap {
                params,
                chunks: vec![
                    RenderChunk {
                        coord: (0, 0),
                        tiles: near_tiles,
                    },
                    RenderChunk {
                        coord: (1000, 0),
                        tiles: far_tiles,
                    },
                    RenderChunk {
                        coord: (0, 1),
                        tiles: empty_tiles,
                    },
                ],
            }],
            ..Default::default()
        };

        let runs = build_tilemap_runs(&scene, &view);
        // Only the near, non-empty chunk survives.
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].texture, 0xA1);
        assert_eq!(runs[0].instances.len(), 1);
    }
}
