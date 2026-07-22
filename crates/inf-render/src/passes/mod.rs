//! Render-graph nodes: one module per pass. Scene passes share `group(0)`
//! (view uniforms) and render into the MSAA scene targets; the composite node
//! resolves to the output.

pub mod bloom;
pub mod classic_vgeom;
pub mod composite;
pub mod debug;
pub mod depth_prepass;
pub mod gi;
pub mod grid;
pub mod mask;
pub mod mesh;
pub mod resolve;
pub mod shadow;
pub mod skinned;
pub mod sky;
pub mod sprite;
pub mod ssao;
pub mod taa;
pub mod terrain;
pub mod tonemap;
pub mod vgeom;

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

/// How a scene shader source is composed into the module a pass compiles.
pub(crate) enum ShaderKind {
    /// [`scene_shader`]: common_view prepended.
    Plain,
    /// [`lit_scene_shader`]: common_view + env_lighting at the given group.
    Lit(u32),
}

/// Every composed scene-shader module the renderer builds — label, WGSL
/// source, composition. Pass constructors fetch their source from this table
/// via [`shader_source`], and the `shader_compose` unit test naga-validates
/// each entry, so a composition that references undeclared identifiers (the
/// pick-shader `shadow` bug class) fails in CI instead of panicking the live
/// viewport thread. Add new composed call sites HERE, not inline.
pub(crate) const SHADER_TABLE: &[(&str, &str, ShaderKind)] = &[
    (
        "mesh",
        include_str!("../shaders/mesh.wgsl"),
        ShaderKind::Lit(2),
    ),
    (
        "skinned",
        include_str!("../shaders/skinned_mesh.wgsl"),
        ShaderKind::Lit(2),
    ),
    (
        "terrain",
        include_str!("../shaders/terrain.wgsl"),
        ShaderKind::Lit(3),
    ),
    (
        "depth_prepass",
        include_str!("../shaders/depth_prepass.wgsl"),
        ShaderKind::Plain,
    ),
    (
        "mask",
        include_str!("../shaders/mask.wgsl"),
        ShaderKind::Plain,
    ),
    (
        "debug",
        include_str!("../shaders/debug.wgsl"),
        ShaderKind::Plain,
    ),
    (
        "grid",
        include_str!("../shaders/grid.wgsl"),
        ShaderKind::Plain,
    ),
    (
        "sky",
        include_str!("../shaders/sky.wgsl"),
        ShaderKind::Plain,
    ),
    (
        "vgeom_mesh",
        include_str!("../shaders/vgeom_mesh.wgsl"),
        ShaderKind::Plain,
    ),
    (
        "taa",
        include_str!("../shaders/taa.wgsl"),
        ShaderKind::Plain,
    ),
    (
        "ssao",
        include_str!("../shaders/ssao.wgsl"),
        ShaderKind::Plain,
    ),
    (
        "sprite",
        include_str!("../shaders/sprite.wgsl"),
        ShaderKind::Plain,
    ),
];

/// Compose the named [`SHADER_TABLE`] entry. Panics on an unknown label —
/// that's a compile-time-adjacent programmer error, caught by the unit tests
/// and by any pass constructor running at all.
pub(crate) fn shader_source(label: &str) -> String {
    let (_, source, kind) = SHADER_TABLE
        .iter()
        .find(|(l, ..)| *l == label)
        .unwrap_or_else(|| panic!("unknown shader table entry: {label}"));
    match kind {
        ShaderKind::Plain => scene_shader(source),
        ShaderKind::Lit(group) => lit_scene_shader(source, *group),
    }
}

/// A **lit** scene shader (mesh/skinned/terrain): [`scene_shader`] plus the shared
/// environment-lighting snippet (AO + cascaded shadows + dynamic GI bindings and
/// sampling fns), with its `GROUP_ENV` token substituted for the pipeline's env
/// bind-group index (mesh/skinned = 2, terrain = 3). See [`EnvBinding`] +
/// `shaders/env_lighting.wgsl`.
pub(crate) fn lit_scene_shader(source: &str, env_group: u32) -> String {
    let env =
        include_str!("../shaders/env_lighting.wgsl").replace("GROUP_ENV", &env_group.to_string());
    format!(
        "{}\n{}\n{}",
        include_str!("../shaders/common_view.wgsl"),
        env,
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

/// The consolidated **environment** bind (P13.3b): AO + cascaded shadows + dynamic
/// GI in a *single* bind group so the lit passes stay within the 4-bind-group limit
/// (mesh needs view/lights/env; skinned adds joints; terrain adds tile/material).
///
/// It occupies the exact group index each shader previously used for [`AoBinding`]
/// (mesh/skinned = 2, terrain = 3), and keeps the AO texture+sampler at
/// bindings 0/1 — so the existing AO shader declarations are unchanged and only the
/// shadow (2,3,4) + GI (5,6) bindings are *appended*. With shadows/GI off the
/// receivers branch on the shared uniforms' `enabled` flags and take the byte-stable
/// AO-only path.
///
/// Bindings: `0` ao_tex, `1` ao_smp, `2` shadow_map (`texture_depth_2d_array`),
/// `3` shadow_smp (comparison), `4` shadow uniform, `5` gi SH storage, `6` gi
/// uniform. All fragment-stage. The bind group is rebuilt when the frame targets
/// change (the AO view is the only size-dependent resource; shadow/GI resources are
/// stable).
pub(crate) struct EnvBinding {
    pub bgl: wgpu::BindGroupLayout,
    ao_sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    bg: Option<(u64, wgpu::BindGroup)>,
}

impl EnvBinding {
    pub fn new(gpu: &GpuContext) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let bgl = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("env"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: frag,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: frag,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: frag,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
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
        let ao_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("env-ao"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        // Comparison sampler: a receiver is lit where its (biased) light-space depth
        // is ≤ the stored nearest-caster depth (forward-Z shadow map). Linear filter
        // gives hardware 2×2 PCF that the shader's 3×3 grid taps average further.
        let shadow_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("env-shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        Self {
            bgl,
            ao_sampler,
            shadow_sampler,
            bg: None,
        }
    }

    /// The env bind group for this frame, rebuilt when the frame targets change.
    pub fn bind_group(&mut self, gpu: &GpuContext, frame: &FrameData) -> &wgpu::BindGroup {
        if self
            .bg
            .as_ref()
            .is_none_or(|(gen, _)| *gen != frame.targets.generation)
        {
            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("env"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&frame.targets.ao),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.ao_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&frame.shadow.array_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: frame.shadow.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: frame.gi.sh.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: frame.gi.uniform.as_entire_binding(),
                    },
                ],
            });
            self.bg = Some((frame.targets.generation, bg));
        }
        &self.bg.as_ref().unwrap().1
    }
}

#[cfg(test)]
mod shader_compose_tests {
    use super::*;

    fn validate(label: &str, source: &str) {
        let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| {
            panic!(
                "shader '{label}' failed to parse:\n{}",
                e.emit_to_string(source)
            )
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("shader '{label}' failed validation: {e:?}"));
    }

    /// The pick-shader regression gate: every composed scene shader (the exact
    /// strings the pass constructors and `Picker` compile) must parse and
    /// validate — no GPU needed, so this runs on every CI platform.
    #[test]
    fn composed_scene_shaders_validate() {
        for (label, ..) in SHADER_TABLE {
            validate(label, &shader_source(label));
        }
    }

    /// Standalone (uncomposed) modules the passes compile as-is.
    #[test]
    fn standalone_shaders_validate() {
        for (label, source) in [
            ("bloom", include_str!("../shaders/bloom.wgsl")),
            ("composite", include_str!("../shaders/composite.wgsl")),
            ("gi_voxelize", include_str!("../shaders/gi_voxelize.wgsl")),
            ("gi_probes", include_str!("../shaders/gi_probes.wgsl")),
            ("vgeom_cull", include_str!("../shaders/vgeom_cull.wgsl")),
            ("vgeom_hzb", include_str!("../shaders/vgeom_hzb.wgsl")),
            ("shadow_depth", include_str!("../shaders/shadow_depth.wgsl")),
            ("tonemap", include_str!("../shaders/tonemap.wgsl")),
        ] {
            validate(label, source);
        }
    }
}
