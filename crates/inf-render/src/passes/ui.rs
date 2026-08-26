//! **The in-game UI pass** (island wave I5): the GPU half of `inf-ui`.
//!
//! # Where it sits, and why it is not the sprite pass
//!
//! After the **tonemap**, into the display-referred `scene_color` the composite
//! blits. The sprite pass draws into the HDR MSAA scene target before resolve,
//! TAA, bloom and tonemap — so a menu drawn through it would be bloomed,
//! temporally reprojected and colour-graded along with the world behind it, and
//! would be depth-tested against the wall the player is standing next to. A UI is
//! none of those things.
//!
//! It also draws in **pixels** rather than in metres: the whole point of
//! `inf_ui::draw` is a space that does not move when the camera does.
//!
//! # What it is inert on
//!
//! Everything. `RenderScene::ui` is empty on every frame nobody opened a menu
//! on, and an empty list returns before the encoder is touched — which is what
//! the frozen goldens rest on, and it is asserted rather than assumed (see
//! `the_ui_node_is_a_no_op_on_an_empty_list`).

use inf_render_2d::{builtin_font_rgba8, TextureHandle, BUILTIN_FONT_TEXTURE, WHITE_TEXTURE};

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, LDR_FORMAT};

/// The uniform the vertex shader maps pixels to clip space with.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UiParams {
    /// xy = the viewport in virtual pixels; zw unused.
    viewport: [f32; 4],
}

/// Per-instance data; matches `@location(0..=2)` in `ui.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiRaw {
    /// xy = the rect's top-left corner in pixels, zw = its extent.
    rect: [f32; 4],
    /// xy = uv_min, zw = uv_max.
    uv: [f32; 4],
    color: [f32; 4],
}

impl UiRaw {
    /// Pack one screen-space quad.
    ///
    /// **No floating origin and no pivot**, unlike `SpriteRaw`: the position is
    /// already the corner, in pixels, in a space that has no world in it. That
    /// is the whole difference between the two passes, and it is why they do not
    /// share an instance type — a UI quad rebased against a `FloatingOrigin`
    /// would move when the camera crossed a rebase boundary.
    pub fn pack(q: &inf_render_2d::SpriteInstance) -> Self {
        Self {
            rect: [q.position.x as f32, q.position.y as f32, q.size.x, q.size.y],
            uv: [q.uv_min.x, q.uv_min.y, q.uv_max.x, q.uv_max.y],
            color: q.color,
        }
    }
}

const UI_ATTRIBUTES: [wgpu::VertexAttribute; 3] = [
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
];

/// One texture the pass can bind. There are exactly two.
struct Bound {
    bind_group: wgpu::BindGroup,
}

/// The node.
pub struct UiNode {
    pipeline: wgpu::RenderPipeline,
    params_bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    params_bg: wgpu::BindGroup,
    /// The 1 × 1 white fallback — every filled rect samples it.
    white: Bound,
    /// The built-in 8 × 8 bitmap font atlas — every glyph samples it.
    font: Bound,
    instances: wgpu::Buffer,
    capacity: usize,
    scratch: Vec<UiRaw>,
}

impl UiNode {
    /// Build the pipeline and upload the two textures the UI can name.
    pub fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ui"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ui.wgsl").into()),
            });
        let params_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ui-params"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let tex_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ui-texture"),
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
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("ui"),
                bind_group_layouts: &[Some(&params_bgl), Some(&tex_bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("ui"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<UiRaw>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &UI_ATTRIBUTES,
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: LDR_FORMAT,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    cull_mode: None,
                    ..Default::default()
                },
                // No depth at all: a menu is not in the world, so there is
                // nothing for it to be behind.
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        let params_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui-params"),
            size: std::mem::size_of::<UiParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui-params"),
            layout: &params_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ui-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            // **Nearest**, not linear, and it is not a preference: the built-in
            // font is an 8 x 8 bitmap and a filtered sample between two texels of
            // a one-pixel stem is a smear rather than a letter.
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let white = bind_rgba8(gpu, &tex_bgl, &sampler, 1, 1, &[255, 255, 255, 255]);
        let (font_rgba, fw, fh) = builtin_font_rgba8();
        let font = bind_rgba8(gpu, &tex_bgl, &sampler, fw, fh, &font_rgba);
        Self {
            pipeline,
            params_bgl,
            params_buf,
            params_bg,
            white,
            font,
            instances: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ui-instances"),
                size: 64 * std::mem::size_of::<UiRaw>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            capacity: 64,
            scratch: Vec::new(),
        }
    }

    /// The bind group a handle names. There are two textures and no cache: a UI
    /// that referenced an asset would need the asset DB in the render thread,
    /// which the shipped player does not have (see `inf_player::render`'s own
    /// note), so a handle that is neither reads as the white fallback and tints.
    fn bound(&self, handle: TextureHandle) -> &wgpu::BindGroup {
        if handle == BUILTIN_FONT_TEXTURE {
            &self.font.bind_group
        } else {
            &self.white.bind_group
        }
    }

    fn grow(&mut self, gpu: &GpuContext, needed: usize) {
        if needed <= self.capacity {
            return;
        }
        let capacity = needed.next_power_of_two();
        self.instances = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui-instances"),
            size: (capacity * std::mem::size_of::<UiRaw>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.capacity = capacity;
    }
}

fn bind_rgba8(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Bound {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ui-texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // sRGB, so a colour written as linear straight alpha samples back as the
        // same number the sprite path would have got.
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Bound {
        bind_group: gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui-texture"),
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
        }),
    }
}

impl RenderNode for UiNode {
    fn name(&self) -> &'static str {
        "ui"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let list = &frame.scene.ui;
        // **The whole node, on every frame nobody opened a menu on.** No
        // encoder, no bind group, no draw — which is what the frozen goldens
        // rest on.
        if list.is_empty() {
            return;
        }
        // The list carries the viewport it was laid out for; the *target* is
        // what will be drawn. Using the target's size would stretch a list built
        // for a stale resolution, and using the list's would put it in the wrong
        // place. They are the same number on any frame a host built its UI on,
        // and the target is the honest one to map with.
        let (w, h) = frame.targets.size;
        gpu.queue.write_buffer(
            &self.params_buf,
            0,
            bytemuck::bytes_of(&UiParams {
                viewport: [w as f32, h as f32, 0.0, 0.0],
            }),
        );
        self.scratch.clear();
        self.scratch.extend(list.quads.iter().map(UiRaw::pack));
        self.grow(gpu, self.scratch.len());
        gpu.queue
            .write_buffer(&self.instances, 0, bytemuck::cast_slice(&self.scratch));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.scene_color,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    // **Load**, never clear: the tonemap just wrote the frame in
                    // here and the UI composites over it.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.params_bg, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        // Maximal runs of one texture, in painter order. A menu is two runs (the
        // panels, then the glyphs) interleaved per row, so this is a handful of
        // draws rather than one per quad and it needs no sort — the list is
        // already in the order it must be drawn.
        let mut i = 0usize;
        while i < list.quads.len() {
            let texture = list.quads[i].texture;
            let start = i;
            while i < list.quads.len() && list.quads[i].texture == texture {
                i += 1;
            }
            pass.set_bind_group(1, self.bound(texture), &[]);
            pass.draw(0..4, start as u32..i as u32);
        }
        let _ = &self.params_bgl;
        let _ = WHITE_TEXTURE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DVec3, Vec2};

    #[test]
    fn the_instance_layout_matches_the_shader() {
        assert_eq!(std::mem::size_of::<UiRaw>(), 48);
        let q = inf_render_2d::SpriteInstance {
            position: DVec3::new(10.0, 20.0, 0.0),
            size: Vec2::new(30.0, 40.0),
            ..Default::default()
        };
        let raw = UiRaw::pack(&q);
        assert_eq!(raw.rect, [10.0, 20.0, 30.0, 40.0]);
        let base = &raw as *const UiRaw as usize;
        assert_eq!(&raw.uv as *const _ as usize - base, 16);
        assert_eq!(&raw.color as *const _ as usize - base, 32);
    }

    /// **A UI quad is NOT rebased against the floating origin.**
    ///
    /// The sprite pass's `pack` calls `origin.to_render`, because a sprite is in
    /// the world. A UI quad is in pixels: rebasing one would move the menu every
    /// time the camera crossed a rebase boundary, which on a 50 km island is
    /// every kilometre of travel.
    #[test]
    fn a_ui_quad_is_in_pixels_and_the_camera_cannot_move_it() {
        let q = inf_render_2d::SpriteInstance {
            position: DVec3::new(100.0, 200.0, 0.0),
            size: Vec2::new(10.0, 10.0),
            ..Default::default()
        };
        // The same quad packs identically however far the world has travelled —
        // there is nothing in `pack` for an origin to reach.
        assert_eq!(UiRaw::pack(&q).rect, [100.0, 200.0, 10.0, 10.0]);
    }
}
