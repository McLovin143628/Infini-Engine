//! Render-graph nodes: one module per pass. Scene passes share `group(0)`
//! (view uniforms) and render into the MSAA scene targets; the composite node
//! resolves to the output.

pub mod bloom;
pub mod composite;
pub mod debug;
pub mod depth_prepass;
pub mod grid;
pub mod mask;
pub mod mesh;
pub mod resolve;
pub mod skinned;
pub mod sky;
pub mod sprite;
pub mod ssao;
pub mod taa;
pub mod terrain;
pub mod tonemap;

use crate::gpu::GpuContext;
use crate::renderer::FrameData;

/// Scene shaders share the `View` uniform block + helpers.
pub(crate) fn scene_shader(source: &str) -> String {
    format!(
        "{}\n{}",
        include_str!("../shaders/common_view.wgsl"),
        source
    )
}

/// The SSAO/ambient-occlusion bind (texture + sampler) that the lit passes
/// (mesh/terrain/skinned) sample. When SSAO is disabled the [`ssao`] node clears
/// the AO target to **white** (1.0), so the ambient term is unchanged — the
/// binding is present in every lit pipeline but pixel-neutral when off. The
/// bind group is rebuilt whenever the frame targets are recreated.
pub(crate) struct AoBinding {
    pub bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    bg: Option<(u64, wgpu::BindGroup)>,
}

impl AoBinding {
    pub fn new(gpu: &GpuContext) -> Self {
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ao"),
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
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ao"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            bgl,
            sampler,
            bg: None,
        }
    }

    /// The AO bind group for this frame, rebuilt when the targets change.
    pub fn bind_group(&mut self, gpu: &GpuContext, frame: &FrameData) -> &wgpu::BindGroup {
        if self
            .bg
            .as_ref()
            .is_none_or(|(gen, _)| *gen != frame.targets.generation)
        {
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ao"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&frame.targets.ao),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.bg = Some((frame.targets.generation, bg));
        }
        &self.bg.as_ref().unwrap().1
    }
}
