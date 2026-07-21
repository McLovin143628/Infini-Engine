//! The engine renderer: owns frame targets, the shared view bind group, and
//! the render graph. One instance per output (editor viewport, headless test).

use crate::camera::{RenderView, ViewUniforms, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderGraph;
use crate::passes;
use crate::scene::RenderScene;

/// Offscreen scene color format — fixed regardless of the output/swapchain
/// format so shading and goldens behave identically everywhere; the composite
/// blit converts.
pub const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Selection/hover mask format.
pub const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
/// 4× MSAA — guaranteed support for all formats we use.
pub const SCENE_SAMPLES: u32 = 4;

/// Per-size GPU targets. Recreated when the scene size changes; `generation`
/// lets nodes cache bind groups against the current views.
pub struct FrameTargets {
    pub size: (u32, u32),
    pub generation: u64,
    pub color_msaa: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    pub scene_color: wgpu::TextureView,
    pub mask: wgpu::TextureView,
}

impl FrameTargets {
    fn create(gpu: &GpuContext, size: (u32, u32), generation: u64) -> Self {
        let (width, height) = (size.0.max(1), size.1.max(1));
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let tex = |label, format, samples, usage: wgpu::TextureUsages| {
            gpu.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        Self {
            size: (width, height),
            generation,
            color_msaa: tex(
                "scene-color-msaa",
                SCENE_FORMAT,
                SCENE_SAMPLES,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            ),
            depth: tex(
                "scene-depth",
                DEPTH_FORMAT,
                SCENE_SAMPLES,
                wgpu::TextureUsages::RENDER_ATTACHMENT,
            ),
            scene_color: tex(
                "scene-color",
                SCENE_FORMAT,
                1,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            mask: tex(
                "outline-mask",
                MASK_FORMAT,
                1,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
        }
    }
}

/// Everything a render node can see for the current frame.
pub struct FrameData<'a> {
    pub scene: &'a RenderScene,
    pub view: &'a RenderView,
    pub targets: &'a FrameTargets,
    pub view_bg: &'a wgpu::BindGroup,
    pub out_view: &'a wgpu::TextureView,
    pub out_size: (u32, u32),
    pub out_format: wgpu::TextureFormat,
}

pub struct EngineRenderer {
    view_buf: wgpu::Buffer,
    view_bg: wgpu::BindGroup,
    pub view_bgl: wgpu::BindGroupLayout,
    targets: Option<FrameTargets>,
    next_generation: u64,
    graph: RenderGraph,
    out_format: wgpu::TextureFormat,
}

impl EngineRenderer {
    pub fn new(gpu: &GpuContext, out_format: wgpu::TextureFormat) -> Self {
        let view_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view-uniforms"),
            size: std::mem::size_of::<ViewUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view_bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("view"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let view_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view"),
            layout: &view_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_buf.as_entire_binding(),
            }],
        });

        let mut graph = RenderGraph::default();
        graph.add(passes::sky::SkyNode::new(gpu, &view_bgl));
        graph.add(passes::mesh::MeshNode::new(gpu, &view_bgl));
        // Terrain draws opaque + depth-writing after meshes and before the grid,
        // so the infinite grid is occluded where terrain rises above the ground
        // plane. A no-op when the scene has no terrain (pre-P10.1 byte stability).
        graph.add(passes::terrain::TerrainNode::new(gpu, &view_bgl));
        graph.add(passes::grid::GridNode::new(gpu, &view_bgl));
        // Sprites draw over the 3D scene (depth-tested, not depth-writing) and
        // under the debug/gizmo overlay.
        graph.add(passes::sprite::SpriteNode::new(gpu, &view_bgl));
        graph.add(passes::debug::DebugNode::new(gpu, &view_bgl));
        graph.add(passes::resolve::ResolveNode);
        // The mask feeds the composite's outline dilate; it renders into the
        // single-sample mask target independently of the MSAA scene resolve.
        graph.add(passes::mask::MaskNode::new(gpu, &view_bgl));
        graph.add(passes::composite::CompositeNode::new(gpu));

        Self {
            view_buf,
            view_bg,
            view_bgl,
            targets: None,
            next_generation: 1,
            graph,
            out_format,
        }
    }

    /// Render one frame of `scene` into `out_view` (`out_size` = the output
    /// texture's size; may briefly differ from the view size while a resize
    /// debounce is pending — the composite stretch covers the gap).
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        scene: &RenderScene,
        view: &RenderView,
        out_view: &wgpu::TextureView,
        out_size: (u32, u32),
    ) {
        let scene_size = (view.width.max(1), view.height.max(1));
        if self.targets.as_ref().is_none_or(|t| t.size != scene_size) {
            self.targets = Some(FrameTargets::create(gpu, scene_size, self.next_generation));
            self.next_generation += 1;
        }

        gpu.queue.write_buffer(
            &self.view_buf,
            0,
            bytemuck::bytes_of(&ViewUniforms::from_view(view)),
        );

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        let frame = FrameData {
            scene,
            view,
            targets: self.targets.as_ref().unwrap(),
            view_bg: &self.view_bg,
            out_view,
            out_size,
            out_format: self.out_format,
        };
        self.graph.run(gpu, &mut encoder, &frame);
        gpu.queue.submit([encoder.finish()]);
    }
}
