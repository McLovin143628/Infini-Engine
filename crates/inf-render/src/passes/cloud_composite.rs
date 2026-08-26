//! Cloud composite (wave SKY2): the half-res cloud, **bilaterally** upsampled
//! into the MSAA scene target.
//!
//! This is the pass that carries P17.3's hardware depth mechanism — `frag_depth`
//! at the ray's entry into the slab, `Greater` (reverse-Z), writes off — and the
//! premultiplied blend the march used to do itself. See [`super::cloud`] for why
//! it sits where it does in the graph and why the march moved out of it.
//!
//! ## The upsample is bilateral, not a box blur
//!
//! The tree already had a half-res pass with a full-res consumer: [`super::ssao`],
//! whose upsample is a 4×4 box **blur**. That is right for an occlusion term
//! that is already low-frequency and multiplies a surface, and it is wrong here.
//! A cloud tap whose march stopped at a mountain carries almost no cloud; the tap
//! beside it, looking past the summit, carries a sky's worth. Averaging them
//! paints a halo of half-cloud along every ridge — the tell of a half-res
//! volumetric, and the reason "render it at half res" is usually followed by "and
//! it looks like it".
//!
//! Each of the four taps is therefore weighted by how well the geometry **its**
//! march stopped at matches what this full-res pixel sees. The key is the
//! distance the march clamped to (published in `cloud_dist.g`), not a raw
//! reverse-Z depth: comparing reverse-Z is comparing a hyperbola, where a 2 km
//! summit and infinity sit a thousandth apart.
//!
//! ## Off path
//!
//! Clouds inactive ⇒ nothing is encoded.

use crate::camera::{DEPTH_COMPARE, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderNode;
use crate::renderer::{FrameData, SCENE_FORMAT, SCENE_SAMPLES};

pub struct CloudCompositeNode {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    /// Keyed on the full [`super::ResourceKey`]: the bind group holds the
    /// half-res cloud views and the scene depth, all of which are recreated on
    /// every viewport resize, plus the atmosphere uniform, which is recreated
    /// with the tier.
    ///
    /// **And on which history slot it points at**, which the resource key cannot
    /// see: the composite reads `cloud_history[cur]` when the temporal pass is on
    /// and `cloud` when it is not, and `cur` alternates every frame. Caching on
    /// the resource key alone would pin the group to whichever slot the first
    /// frame happened to use and show the previous frame's cloud for ever.
    bind_group: super::GenCache<(super::ResourceKey, usize), wgpu::BindGroup>,
}

impl CloudCompositeNode {
    pub fn new(gpu: &GpuContext, view_bgl: &wgpu::BindGroupLayout) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cloud-composite"),
                source: wgpu::ShaderSource::Wgsl(super::shader_source("cloud_composite").into()),
            });
        let frag = wgpu::ShaderStages::FRAGMENT;
        let tex = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: frag,
            ty: wgpu::BindingType::Texture {
                // `textureLoad`ed at integer texels — the taps are chosen and
                // weighted by hand, so nothing here is filtered and nothing needs
                // a sampler.
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cloud-composite"),
                entries: &[
                    // 0 = the shared atmosphere uniform, for the slab geometry the
                    // written depth comes from.
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: frag,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    tex(1),
                    tex(2),
                    // 3 = the scene depth. Read to recover THIS pixel's geometry
                    // distance, which is the bilateral key's other half. Bound
                    // read-only as an attachment at the same time, the one
                    // arrangement WebGPU permits.
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
                ],
            });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cloud-composite"),
                bind_group_layouts: &[Some(view_bgl), Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("cloud-composite"),
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
                        format: SCENE_FORMAT,
                        // PREMULTIPLIED alpha: the march accumulated radiance
                        // already weighted by the transmittance it was seen
                        // through, so the source is `(L·α, α)` and must not be
                        // multiplied by α a second time.
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    // Test but never write: a cloud is a participating medium, not
                    // a surface, and writing its entry depth would make later
                    // passes think there is geometry there.
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
            bgl,
            bind_group: super::GenCache::default(),
        }
    }
}

impl RenderNode for CloudCompositeNode {
    fn name(&self) -> &'static str {
        "cloud-composite"
    }

    fn run(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, frame: &FrameData) {
        if !frame.scene.atmosphere.clouds_active() {
            return;
        }
        let atmos = frame.atmosphere;
        let src = frame.cloud_src;
        let bgl = &self.bgl;
        let bind_group = self
            .bind_group
            .get_or_build((super::resource_key(frame), frame.cloud_slot), || {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("cloud-composite"),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: atmos.uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(src),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&frame.targets.cloud_dist),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&frame.targets.depth),
                        },
                    ],
                })
            })
            .clone();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cloud-composite"),
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
                // `None` = READ-ONLY depth: contents are preserved for the passes
                // that follow, the hardware test still runs, and — the reason it
                // is here — the same view may simultaneously be bound as a sampled
                // texture. A read/write attachment aliased into a binding is a
                // validation error; a read-only one is exactly what WebGPU allows.
                depth_ops: None,
                stencil_ops: None,
            }),
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
