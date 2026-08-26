//! **Sun glare / lens flare** (wave VIS1b) — the veiling glare around the sun,
//! a modest ghost chain, a halo and an optional anamorphic streak.
//!
//! Half resolution, into [`FrameTargets::flare`](crate::renderer::FrameTargets),
//! which the tonemap adds beside the bloom. It sits **after bloom and before the
//! tonemap** for the same reason bloom does: the thing it gathers is the frame's
//! bright part, and the thing it feeds is the display transform.
//!
//! # It reads the MSAA scene depth, not the prepass
//!
//! The occlusion test asks whether the sun's own screen position is behind
//! geometry, and `targets.depth` is the depth **every** pass wrote — terrain,
//! meshlets, scattered foliage, water, all of it. `targets.depth_prepass` is
//! narrower (VIS-C1b: vgeom and scatter still do not write it) and would let the
//! sun glare straight through a tree. The MSAA depth has carried
//! `TEXTURE_BINDING` since P17.3, so this costs no new usage.
//!
//! # Off is a clear
//!
//! `FlareSettings::enabled == false` ⇒ the node clears the target to black and
//! records nothing else — the [`super::bloom`] shape exactly. The tonemap does
//! not even sample it then: its add is behind a uniform branch, so a level that
//! never asked for a lens does not pay a full-resolution fetch for one. Every
//! golden therefore runs the command stream it always did, and
//! `INF_GOLDEN_STRICT=1` is the check rather than the claim.

use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, HDR_FORMAT};

/// The bright-pass threshold, in **exposed** units.
///
/// Not authored, and that is a schema fact rather than a design one: the VIS
/// arc's one scene-schema window was spent in VIS1a and `FlareSettings` has five
/// fields, none of them a threshold. `1.0` is the same number bloom defaults to,
/// which is the right default anyway — a lens flares off what has blown past the
/// display's white, and after the exposure multiply that is exactly `> 1.0`.
const FLARE_THRESHOLD: f32 = 1.0;

/// The most ghosts the chain will draw. **MIRROR: `FLARE_MAX_GHOSTS` in
/// `flare.wgsl`**, and both halves are load-bearing — see the clamp's comment in
/// `run`.
const MAX_GHOSTS: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlareUniform {
    /// x = veiling intensity, y = ghost count, z = halo, w = streak.
    params: [f32; 4],
    /// x = bright-pass threshold in exposed units, yzw unused.
    tune: [f32; 4],
    /// xy = this pass's target size (px), zw = its texel size.
    dims: [f32; 4],
}

pub struct FlareNode {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl FlareNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("flare"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("flare").into()),
            });
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let frag = wgpu::ShaderStages::FRAGMENT;
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("flare"),
                entries: &[
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
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // The MSAA scene depth, `textureLoad`ed at sample 0 — the
                    // underwater pass's binding, for the underwater pass's reason.
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    },
                    // The frame's exposure — the third reader of the same sixteen
                    // bytes, bound rather than copied so there is no ordering
                    // between a queue write and an encoder copy to get right.
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: frag,
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
                label: Some("flare"),
                bind_group_layouts: &[Some(view_bgl), Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("flare"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flare-uniform"),
            size: std::mem::size_of::<FlareUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("flare"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Self {
            pipeline,
            bgl,
            uniform,
            sampler,
        }
    }
}

impl RenderNode for FlareNode {
    fn name(&self) -> &'static str {
        "flare"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        let f = frame.settings.flare;
        if !f.enabled {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("flare-clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame.targets.flare,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            return;
        }

        let (w, h) = frame.targets.flare_size;
        // The bright pass keys in EXPOSED units for the bloom prefilter's reason,
        // and the tonemap therefore adds this texture without re-exposing it. The
        // exposure itself is a *binding* rather than a number written from here,
        // because in auto mode the CPU does not know it — which is also why this
        // node runs after `passes::exposure`.
        //
        // The three finite-or-zero guards are the door: `FlareSettings` comes off
        // a level file, and a negative intensity would SUBTRACT light from the
        // frame while a NaN would poison every pixel the gather reaches.
        let sane = |v: f32| if v.is_finite() { v.max(0.0) } else { 0.0 };
        gpu.queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&FlareUniform {
                params: [
                    sane(f.intensity),
                    // Clamped HERE as well as in the shader. `ghost_count` is a
                    // `u32` off a level file, and `i32(4.29e9)` is an
                    // out-of-range float-to-int conversion — which WGSL leaves
                    // *indeterminate*, so the shader's own `min` would be
                    // guarding against a value the spec does not describe.
                    f.ghost_count.min(MAX_GHOSTS) as f32,
                    sane(f.halo),
                    sane(f.streak),
                ],
                tune: [FLARE_THRESHOLD, 0.0, 0.0, 0.0],
                dims: [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32],
            }),
        );

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flare"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(frame.post_hdr),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&frame.targets.depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: frame.exposure.state.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("flare"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.targets.flare,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.view_bg, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
