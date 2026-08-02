//! Render-graph nodes: one module per pass. Scene passes share `group(0)`
//! (view uniforms) and render into the MSAA scene targets; the composite node
//! resolves to the output.

pub mod bloom;
pub mod classic_vgeom;
pub mod cloud;
pub mod cloud_bake;
pub mod composite;
pub mod debug;
pub mod depth_prepass;
pub mod gi;
pub mod grid;
pub mod mask;
pub mod mesh;
pub mod precip;
pub mod resolve;
pub mod scatter;
pub mod shadow;
pub mod skinned;
pub mod sky;
pub mod sky_lut;
pub mod sprite;
pub mod ssao;
pub mod taa;
pub mod terrain;
pub mod tonemap;
pub mod translucent;
pub mod vgeom;
pub mod water;

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
    /// [`lit_scene_shader`]: common_view + env_lighting + the atmosphere library
    /// at the given group.
    Lit(u32),
    /// [`sky_shader`]: common_view + the atmosphere library at `@group(1)` — the
    /// sky pass, which needs the LUTs but not the lighting env.
    Sky,
    /// [`atmosphere_compute_shader`]: the atmosphere library at `@group(0)` and
    /// nothing else. The two LUT bake compute passes have no view uniform and no
    /// lights (P17.2).
    AtmosphereCompute,
    /// [`cloud_shader`]: common_view + the atmosphere library + the cloud field at
    /// `@group(1)` — the raymarch pass (P17.3), which needs the sky LUTs for
    /// lighting and both noise volumes for density, but no lighting env.
    Cloud,
    /// [`cloud_bake_shader`]: the atmosphere uniform + the cloud **generators** at
    /// `@group(0)`, with no sampled textures at all — the two noise volumes are
    /// *storage* textures here, which is precisely why `cloud_field.wgsl` (which
    /// declares them as sampled) cannot be included (P17.3).
    CloudBake,
    /// [`cloud_shadow_bake_shader`]: the atmosphere uniform + generators + the
    /// sampled cloud field at `@group(0)`, plus the shadow map as storage — the
    /// only compute pass that both reads and writes cloud data (P17.3).
    CloudShadowBake,
    /// [`precip_shader`]: common_view + the atmosphere library at `@group(1)` +
    /// the cloud **generators** (for the hash alone) — the precipitation pass
    /// (P17.4), which places particles from an integer hash and lights them from
    /// the sky LUTs, and binds no volume and no depth texture.
    Precip,
    /// [`hzb_cull_shader`]: the shared HZB occlusion test prepended, and nothing
    /// else. The GPU culls (meshlet, P18.1; scatter, P18.5) have no view uniform
    /// and no lights — what they share is one proven-conservative depth test.
    HzbCull,
    /// [`water_shader`]: common_view + the atmosphere library at `@group(1)`
    /// bindings 0..3 — the water pass (P20.1), which reflects the sky-view LUT and
    /// applies aerial perspective, but binds no lighting env (water is shaded by
    /// its own Fresnel + absorption model, not by the PBR path).
    Water,
    /// [`gi_probe_shader`]: the atmosphere library at `@group(0)` bindings 3..6
    /// and nothing else — the P18.4 GI probe march, which has no view uniform and
    /// no lights but needs the sky-view LUT for its ray-miss term.
    GiProbes,
}

// ── atmosphere composition (P17.2) ───────────────────────────────────────────
//
// `atmosphere.wgsl` (the medium: densities, extinction, LUT parameterizations,
// phases, the star hash) and `atmosphere_lut.wgsl` (sampling the two baked LUTs
// + aerial perspective/height fog) are token-substituted into whichever group
// each consumer has spare — the same mechanism `GROUP_ENV` uses. There is no
// fourth bind group to spend, so the atmosphere rides along in the group the
// pass already binds.

/// Env-group binding indices the atmosphere occupies inside [`EnvBinding`],
/// appended after the AO (0,1) / shadow (2,3,4) / GI (5,6) block.
const ENV_ATMOS_TRANSMITTANCE: u32 = 7;
const ENV_ATMOS_SKYVIEW: u32 = 8;
const ENV_ATMOS_SAMPLER: u32 = 9;
const ENV_ATMOS_UNIFORM: u32 = 10;
/// The cloud-shadow map (P17.3) — the **one** binding volumetric clouds add to
/// the lit passes. It reuses [`ENV_ATMOS_SAMPLER`] (clamp-to-edge, linear), which
/// is exactly what it wants, and its parameters ride in the atmosphere uniform.
const ENV_CLOUD_SHADOW: u32 = 11;
/// The single-sample scene depth (P18.4) — the **one** binding SSR adds. Read with
/// `textureLoad`, so it needs no sampler of its own; it is the SSAO/TAA depth
/// prepass target, which [`crate::RenderSettings::needs_depth_prepass`] now forces
/// on whenever SSR is enabled.
const ENV_SCENE_DEPTH: u32 = 12;

/// The atmosphere *medium* library, bound at `group`/`binding`.
pub(crate) fn atmosphere_source(group: u32, binding: u32) -> String {
    include_str!("../shaders/atmosphere.wgsl")
        .replace("ATMOS_GROUP", &group.to_string())
        .replace("ATMOS_BIND", &binding.to_string())
}

/// The atmosphere *LUT-sampling* library. Only included by passes that bind both
/// baked LUTs (the sky pass and, through [`lit_scene_shader`], every lit pass);
/// the bake passes themselves take [`atmosphere_source`] alone.
pub(crate) fn atmosphere_lut_source(group: u32, t: u32, s: u32, smp: u32) -> String {
    include_str!("../shaders/atmosphere_lut.wgsl")
        .replace("ATMOS_GROUP", &group.to_string())
        .replace("ATMOS_TBIND", &t.to_string())
        .replace("ATMOS_SBIND", &s.to_string())
        .replace("ATMOS_SMPBIND", &smp.to_string())
}

/// Aerial perspective + height fog (`atmos_apply`). Split out of
/// [`atmosphere_lut_source`] in P18.4: it reads the `View` uniform, and the GI
/// probe march is a compute pass that has none — it wants the LUT samplers alone.
/// Binding-free, so it needs no token substitution.
pub(crate) fn atmosphere_apply_source() -> &'static str {
    include_str!("../shaders/atmosphere_apply.wgsl")
}

/// The sky pass module: common_view + the whole atmosphere library at
/// `@group(1)` (uniform at 1, transmittance 2, sky view 3, sampler 4 — binding 0
/// is the authored gradient the sky pass already had).
pub(crate) fn sky_shader(source: &str) -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}",
        include_str!("../shaders/common_view.wgsl"),
        atmosphere_source(1, 1),
        atmosphere_lut_source(1, 2, 3, 4),
        atmosphere_apply_source(),
        source
    )
}

/// A LUT bake compute module: the medium library at `@group(0) @binding(0)`, the
/// bake's own storage/input bindings, and nothing else.
pub(crate) fn atmosphere_compute_shader(source: &str) -> String {
    format!("{}\n{}", atmosphere_source(0, 0), source)
}

// ── cloud composition (P17.3) ────────────────────────────────────────────────

/// The cloud **generators** (hash, Perlin, Worley, weather, height profile,
/// phase). Binding-free by design — the noise bake declares the two volumes as
/// *storage* textures while the raymarch declares them as *sampled* ones, so a
/// single file that bound them could not serve both.
pub(crate) fn cloud_noise_source() -> &'static str {
    include_str!("../shaders/cloud_noise.wgsl")
}

/// The cloud **field**: the sampled volume bindings plus `cloud_density` and
/// `cloud_sun_transmittance`. Only included by passes that bind both volumes as
/// sampled textures (the raymarch and the shadow bake).
pub(crate) fn cloud_field_source(group: u32, shape: u32, detail: u32, smp: u32) -> String {
    include_str!("../shaders/cloud_field.wgsl")
        .replace("CLOUD_GROUP", &group.to_string())
        .replace("CLOUD_SHAPE_BIND", &shape.to_string())
        .replace("CLOUD_DETAIL_BIND", &detail.to_string())
        .replace("CLOUD_SMP_BIND", &smp.to_string())
}

/// The cloud raymarch module: common_view + the atmosphere at `@group(1)`
/// bindings 0..3 + the cloud field at 4/5/6. The scene depth at binding 7 is
/// declared by `cloud.wgsl` itself, since it is the only consumer.
pub(crate) fn cloud_shader(source: &str) -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}",
        include_str!("../shaders/common_view.wgsl"),
        atmosphere_source(1, 0),
        atmosphere_lut_source(1, 1, 2, 3),
        atmosphere_apply_source(),
        cloud_noise_source(),
        cloud_field_source(1, 4, 5, 6),
        source
    )
}

/// The precipitation module: common_view + the atmosphere at `@group(1)`
/// bindings 0..3 + the cloud **generators**, which is where `cloud_hash` /
/// `cloud_hash_unit` live. Reusing that hash rather than writing a second one is
/// the point: it is already pinned bit-for-bit by
/// `clouds::tests::committed_noise_values_are_bit_stable`, and a precipitation
/// hash of its own would be a second thing to keep portable.
pub(crate) fn precip_shader(source: &str) -> String {
    format!(
        "{}
{}
{}
{}
{}
{}",
        include_str!("../shaders/common_view.wgsl"),
        atmosphere_source(1, 0),
        atmosphere_lut_source(1, 1, 2, 3),
        atmosphere_apply_source(),
        cloud_noise_source(),
        source
    )
}

/// The water module (P20.1): common_view + the atmosphere at `@group(1)`
/// bindings 0..3. Everything else water binds — the per-body uniform, the river
/// frames, the scene depth and the resolved scene colour — is declared by
/// `water.wgsl` itself at bindings 4..8, since it is the only consumer.
pub(crate) fn water_shader(source: &str) -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}",
        include_str!("../shaders/common_view.wgsl"),
        atmosphere_source(1, 0),
        atmosphere_lut_source(1, 1, 2, 3),
        atmosphere_apply_source(),
        source
    )
}

/// The GI probe-march module (P18.4): the atmosphere medium at
/// `@group(0) @binding(3)` + the LUT sampling library at 4/5/6, and nothing else.
/// The probe pass has no view uniform and no lights; what it needs from the
/// atmosphere is exactly `atmos_sample_skyview` for its ray-**miss** term, which is
/// how the P17 sky finally reaches the bounce.
/// A GPU **cull** compute module (P18.5): the shared reverse-Z HZB occlusion test
/// prepended, and nothing else.
///
/// Both culls that read the P18.1 pyramid compose through here — the meshlet cull
/// (which had the test inline until P18.5) and the scatter cull. The test takes
/// its uniform, its dimensions and the pyramid texture as *parameters* rather than
/// reading globals, precisely so it can be one function serving two different
/// bind layouts; keeping it in one file is the P18-fixture lesson applied before
/// the second copy exists rather than after it drifts.
pub(crate) fn hzb_cull_shader(source: &str) -> String {
    format!(
        "{}\n{}",
        include_str!("../shaders/hzb_occlusion.wgsl"),
        source
    )
}

pub(crate) fn gi_probe_shader(source: &str) -> String {
    format!(
        "{}\n{}\n{}",
        atmosphere_source(0, 3),
        atmosphere_lut_source(0, 4, 5, 6),
        source
    )
}

/// The cloud noise bake: the atmosphere uniform (for the seed) + the generators,
/// and nothing sampled.
pub(crate) fn cloud_bake_shader(source: &str) -> String {
    format!(
        "{}\n{}\n{}",
        atmosphere_source(0, 0),
        cloud_noise_source(),
        source
    )
}

/// The cloud-shadow bake: the atmosphere uniform + generators + the sampled field
/// at `@group(0)` bindings 1/2/3 — the only compute pass that both reads and
/// writes cloud data.
pub(crate) fn cloud_shadow_bake_shader(source: &str) -> String {
    format!(
        "{}\n{}\n{}\n{}",
        atmosphere_source(0, 0),
        cloud_noise_source(),
        cloud_field_source(0, 1, 2, 3),
        source
    )
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
    ("sky", include_str!("../shaders/sky.wgsl"), ShaderKind::Sky),
    (
        "atmos_transmittance",
        include_str!("../shaders/sky_transmittance.wgsl"),
        ShaderKind::AtmosphereCompute,
    ),
    (
        "atmos_skyview",
        include_str!("../shaders/sky_view.wgsl"),
        ShaderKind::AtmosphereCompute,
    ),
    (
        "vgeom_mesh",
        include_str!("../shaders/vgeom_mesh.wgsl"),
        ShaderKind::Lit(2),
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
    (
        "cloud",
        include_str!("../shaders/cloud.wgsl"),
        ShaderKind::Cloud,
    ),
    (
        "cloud_bake",
        include_str!("../shaders/cloud_bake.wgsl"),
        ShaderKind::CloudBake,
    ),
    (
        "cloud_shadow_bake",
        include_str!("../shaders/cloud_shadow_bake.wgsl"),
        ShaderKind::CloudShadowBake,
    ),
    (
        "precip",
        include_str!("../shaders/precip.wgsl"),
        ShaderKind::Precip,
    ),
    (
        "gi_probes",
        include_str!("../shaders/gi_probes.wgsl"),
        ShaderKind::GiProbes,
    ),
    (
        "vgeom_cull",
        include_str!("../shaders/vgeom_cull.wgsl"),
        ShaderKind::HzbCull,
    ),
    (
        "scatter_cull",
        include_str!("../shaders/scatter_cull.wgsl"),
        ShaderKind::HzbCull,
    ),
    (
        "scatter_mesh",
        include_str!("../shaders/scatter_mesh.wgsl"),
        ShaderKind::Lit(2),
    ),
    (
        "water",
        include_str!("../shaders/water.wgsl"),
        ShaderKind::Water,
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
        ShaderKind::Sky => sky_shader(source),
        ShaderKind::AtmosphereCompute => atmosphere_compute_shader(source),
        ShaderKind::Cloud => cloud_shader(source),
        ShaderKind::CloudBake => cloud_bake_shader(source),
        ShaderKind::CloudShadowBake => cloud_shadow_bake_shader(source),
        ShaderKind::Precip => precip_shader(source),
        ShaderKind::HzbCull => hzb_cull_shader(source),
        ShaderKind::GiProbes => gi_probe_shader(source),
        ShaderKind::Water => water_shader(source),
    }
}

/// A **lit** scene shader (mesh/skinned/terrain/vgeom): [`scene_shader`] plus the
/// shared environment-lighting snippet (AO + cascaded shadows + dynamic GI
/// bindings and sampling fns) **and the atmosphere library** (P17.2: aerial
/// perspective + height fog), with their `GROUP_ENV` / `ATMOS_GROUP` tokens
/// substituted for the pipeline's env bind-group index (mesh/skinned/vgeom = 2,
/// terrain = 3). See [`EnvBinding`] + `shaders/env_lighting.wgsl` +
/// `shaders/atmosphere*.wgsl`.
pub(crate) fn lit_scene_shader(source: &str, env_group: u32) -> String {
    let env =
        include_str!("../shaders/env_lighting.wgsl").replace("GROUP_ENV", &env_group.to_string());
    // P17.3: the cloud-shadow receiver. It reads `atmos` (the cloud block) and
    // borrows `atmos_lut_smp`, so it must be composed AFTER both atmosphere
    // fragments — WGSL has no forward declarations.
    let cloud_shadow =
        include_str!("../shaders/cloud_shadow.wgsl").replace("GROUP_ENV", &env_group.to_string());
    format!(
        "{}\n{}\n{}\n{}\n{}\n{}\n{}",
        include_str!("../shaders/common_view.wgsl"),
        env,
        atmosphere_source(env_group, ENV_ATMOS_UNIFORM),
        atmosphere_lut_source(
            env_group,
            ENV_ATMOS_TRANSMITTANCE,
            ENV_ATMOS_SKYVIEW,
            ENV_ATMOS_SAMPLER,
        ),
        atmosphere_apply_source(),
        cloud_shadow,
        source
    )
}

/// The cache key every bind group that embeds a **resizable** resource is keyed
/// on: `(frame targets generation, atmosphere resources generation, GI resources
/// generation)`.
///
/// A tuple rather than a single counter because the three resources are recreated
/// for unrelated reasons — the targets on a viewport resize, the atmosphere LUTs on
/// a render-tier / atmosphere-quality clamp, the GI voxel/SH buffers on a
/// **GI**-quality clamp — and any one alone must invalidate.
///
/// The GI component is the P18.4 addition, and it is exactly what the P13 version
/// of the [`EnvBinding::bind_group`] comment predicted would be needed "if either
/// ever becomes resizable". It has.
pub(crate) type ResourceKey = (u64, u64, u64);

/// The current [`ResourceKey`] for a frame.
#[inline]
pub(crate) fn resource_key(frame: &FrameData) -> ResourceKey {
    (
        frame.targets.generation,
        frame.atmosphere.generation,
        frame.gi.generation,
    )
}

/// A generation-keyed single-slot cache: hands back the stored value while the
/// key is unchanged, and **rebuilds** it otherwise.
///
/// Extracted (P17.2) so the invalidation rule is one testable thing rather than
/// three hand-rolled `is_none_or(|(k, _)| *k != key)` sites that can each drift.
/// The failure it exists to prevent is quiet: with a stale key the pass keeps a
/// bind group built from the *previous* quality's LUT views and silently samples
/// last tier's textures — no validation error, no black frame, just wrong pixels
/// that a determinism gate would happily call stable.
///
/// `pointer_identity_changes_only_when_the_key_does` pins the behaviour on a
/// stand-in payload, so it runs on every CI leg with no adapter.
/// Generic over the key so a pass keys on exactly what it embeds: [`EnvBinding`]
/// on the full [`ResourceKey`], the sky pass and the LUT bake on the atmosphere
/// generation alone (they hold no size-dependent view, and rebuilding them on
/// every viewport resize would be a lie about what invalidates them).
pub(crate) struct GenCache<K, T> {
    slot: Option<(K, T)>,
}

impl<K, T> Default for GenCache<K, T> {
    fn default() -> Self {
        Self { slot: None }
    }
}

impl<K: PartialEq, T> GenCache<K, T> {
    /// The cached value for `key`, building it with `build` when the key moved
    /// (or on the first call).
    pub fn get_or_build(&mut self, key: K, build: impl FnOnce() -> T) -> &T {
        if self.slot.as_ref().is_none_or(|(k, _)| *k != key) {
            self.slot = Some((key, build()));
        }
        &self.slot.as_ref().unwrap().1
    }
}

/// The consolidated **environment** bind (P13.3b, grown in P17.2): AO + cascaded
/// shadows + dynamic GI + the atmosphere LUTs in a *single* bind group so the lit
/// passes stay within the 4-bind-group limit (mesh needs view/lights/env; skinned
/// adds joints; terrain adds tile/material; vgeom adds its meshlet storage group).
///
/// It occupies the group index each shader uses for its environment
/// (mesh/skinned/vgeom = 2, terrain = 3), and keeps the AO texture+sampler at
/// bindings 0/1 — so the existing AO shader declarations are unchanged and only the
/// shadow (2,3,4), GI (5,6) and atmosphere (7,8,9,10) bindings are *appended*. With
/// shadows/GI/atmosphere off the receivers branch on the shared uniforms' `enabled`
/// flags and take the byte-stable AO-only path.
///
/// Bindings: `0` ao_tex, `1` ao_smp, `2` shadow_map (`texture_depth_2d_array`),
/// `3` shadow_smp (comparison), `4` shadow uniform, `5` gi SH storage, `6` gi
/// uniform, `7` atmosphere transmittance LUT, `8` atmosphere sky-view LUT,
/// `9` atmosphere LUT sampler, `10` atmosphere uniform, `11` cloud-shadow map
/// (P17.3 — which borrows the sampler at `9` rather than adding a twelfth entry),
/// `12` single-sample scene depth (P18.4 SSR — read with `textureLoad`, so no
/// thirteenth sampler either). All fragment-stage.
pub(crate) struct EnvBinding {
    pub bgl: wgpu::BindGroupLayout,
    ao_sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    /// Keyed on [`ResourceKey`] — the two resizable inputs. See
    /// [`EnvBinding::bind_group`].
    bg: GenCache<ResourceKey, wgpu::BindGroup>,
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
                    // ── P17.2 atmosphere ──
                    wgpu::BindGroupLayoutEntry {
                        binding: ENV_ATMOS_TRANSMITTANCE,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: ENV_ATMOS_SKYVIEW,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: ENV_ATMOS_SAMPLER,
                        visibility: frag,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: ENV_ATMOS_UNIFORM,
                        visibility: frag,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // ── P17.3 clouds ──
                    wgpu::BindGroupLayoutEntry {
                        binding: ENV_CLOUD_SHADOW,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // ── P18.4 SSR ──
                    wgpu::BindGroupLayoutEntry {
                        binding: ENV_SCENE_DEPTH,
                        visibility: frag,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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
            bg: GenCache::default(),
        }
    }

    /// The env bind group for this frame, rebuilt when any resource it embeds is
    /// recreated.
    ///
    /// INVARIANT (extended in P17.2): the key is [`resource_key`] — **every
    /// resizable resource this bind group embeds must appear in it**. Miss one and
    /// the pass keeps a bind group built from the *previous* views and silently
    /// samples the previous quality's LUT: no validation error, no black frame,
    /// just wrong pixels that a determinism gate would happily call stable. (wgpu
    /// keeps the old texture alive as long as the bind group references it, so
    /// there is no use-after-free to trip over either — which is exactly what
    /// makes it quiet.)
    ///
    /// * `frame.targets.ao` is size-dependent → `targets.generation`.
    /// * `frame.atmosphere.transmittance` / `.sky_view` are **quality**-dependent:
    ///   [`crate::atmosphere::AtmosphereQuality`] can be clamped down at runtime by
    ///   the render tier, which recreates both LUTs at a new size →
    ///   `atmosphere.generation`. This is exactly the case the P13 version of this
    ///   comment warned about, arriving as predicted.
    /// * `frame.atmosphere.cloud_shadow` (P17.3) is quality-dependent for the same
    ///   reason — [`crate::clouds::CloudQuality`] is *derived* from
    ///   `AtmosphereQuality`, so the cloud textures are recreated in the same
    ///   breath as the LUTs and are covered by the same `atmosphere.generation`.
    ///   A separate counter would only ever move in lockstep with this one; what
    ///   would be a bug is a cloud resource recreated on some *other* trigger
    ///   without a key component of its own, so if one is ever added, add its
    ///   generation here too.
    /// * `frame.gi.sh` / `frame.gi.voxels` are **quality**-dependent since P18.4:
    ///   [`crate::gi::GiQuality`] can be clamped down at runtime by the render tier,
    ///   which recreates both buffers at a new size → `gi.generation`. This is the
    ///   other case the P13 version of this comment warned about, arriving as
    ///   predicted, and the exclusion it granted is now formally spent.
    /// * `frame.targets.depth_prepass` (P18.4 SSR) is size-dependent → already
    ///   covered by `targets.generation`; it needs no component of its own.
    /// * `frame.shadow.*` is still created **once** (in `EngineRenderer::new`) and
    ///   never recreated, so it needs no key component. If it ever becomes
    ///   resizable, add its generation here too.
    pub fn bind_group(&mut self, gpu: &GpuContext, frame: &FrameData) -> &wgpu::BindGroup {
        let (bgl, ao_sampler, shadow_sampler) = (&self.bgl, &self.ao_sampler, &self.shadow_sampler);
        self.bg.get_or_build(resource_key(frame), || {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("env"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&frame.targets.ao),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(ao_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&frame.shadow.array_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(shadow_sampler),
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
                    wgpu::BindGroupEntry {
                        binding: ENV_ATMOS_TRANSMITTANCE,
                        resource: wgpu::BindingResource::TextureView(
                            &frame.atmosphere.transmittance,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: ENV_ATMOS_SKYVIEW,
                        resource: wgpu::BindingResource::TextureView(&frame.atmosphere.sky_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: ENV_ATMOS_SAMPLER,
                        resource: wgpu::BindingResource::Sampler(&frame.atmosphere.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: ENV_ATMOS_UNIFORM,
                        resource: frame.atmosphere.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: ENV_CLOUD_SHADOW,
                        resource: wgpu::BindingResource::TextureView(
                            &frame.atmosphere.cloud_shadow,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: ENV_SCENE_DEPTH,
                        resource: wgpu::BindingResource::TextureView(&frame.targets.depth_prepass),
                    },
                ],
            })
        })
    }
}

#[cfg(test)]
mod gen_cache_tests {
    use super::*;

    /// The `EnvBinding` invariant, as a **pointer-identity** property on a
    /// stand-in payload — deterministic and adapter-free, so it runs on every CI
    /// leg rather than only where a GPU exists.
    ///
    /// This is the seam the golden `atmosphere_quality_switch_rebuilds_the_env_bind`
    /// cannot reach: from outside, a stale bind group is *silent* (wgpu keeps the
    /// old texture alive, so there is no validation error and no black frame — the
    /// pass simply samples the previous quality's LUT). Here the rebuild is
    /// observable directly: a new payload is a different allocation.
    #[test]
    fn pointer_identity_changes_only_when_the_key_does() {
        use std::rc::Rc;

        /// Fetch through the cache, counting how many times the payload was
        /// actually *built*. Returns an owned handle: every one is kept alive for
        /// the whole test, so the allocator cannot hand a freed address back and
        /// make a rebuild look like a cache hit.
        fn fetch(
            cache: &mut GenCache<ResourceKey, Rc<u32>>,
            key: ResourceKey,
            builds: &mut u32,
        ) -> Rc<u32> {
            let counter = &mut *builds;
            cache
                .get_or_build(key, || {
                    *counter += 1;
                    Rc::new(*counter)
                })
                .clone()
        }

        // `Rc` stands in for `wgpu::BindGroup`, which is itself a refcounted
        // handle — so this is the same identity question the real cache answers.
        let mut cache: GenCache<ResourceKey, Rc<u32>> = GenCache::default();
        let mut builds = 0u32;

        let first = fetch(&mut cache, (1, 1, 1), &mut builds);
        // Same key ⇒ the SAME allocation, and the builder never ran again.
        let again = fetch(&mut cache, (1, 1, 1), &mut builds);
        assert!(Rc::ptr_eq(&first, &again), "an unchanged key rebuilt");
        assert_eq!(builds, 1, "an unchanged key must not rebuild");

        // The atmosphere generation alone must invalidate — this is the key
        // component P17.2 added, and the one a regression would drop.
        let after_atmos = fetch(&mut cache, (1, 2, 1), &mut builds);
        assert!(
            !Rc::ptr_eq(&after_atmos, &first),
            "an atmosphere-generation bump did not rebuild the bind group"
        );
        assert_eq!(builds, 2);

        // ...and so must the frame-targets generation alone (the P13 component).
        let after_targets = fetch(&mut cache, (2, 2, 1), &mut builds);
        assert!(
            !Rc::ptr_eq(&after_targets, &after_atmos),
            "a targets-generation bump did not rebuild the bind group"
        );
        assert_eq!(builds, 3);

        // ...and the GI generation alone (the P18.4 component). GI resources became
        // resizable when `GiQuality` landed, which is precisely the case the P13
        // exclusion comment said would have to join this key.
        let after_gi = fetch(&mut cache, (2, 2, 2), &mut builds);
        assert!(
            !Rc::ptr_eq(&after_gi, &after_targets),
            "a GI-generation bump did not rebuild the bind group"
        );
        assert_eq!(builds, 4);

        // Returning to a previously-seen key still rebuilds: the cache is one
        // slot, not a map, so it can never hand back a handle to a resource that
        // has since been recreated under the same number.
        let back = fetch(&mut cache, (1, 1, 1), &mut builds);
        assert!(!Rc::ptr_eq(&back, &first));
        assert_eq!(builds, 5);
    }

    /// A key that ignores one of its inputs is exactly the regression this guards,
    /// so assert the tuple is injective in every component rather than trusting
    /// the call site.
    #[test]
    fn resource_key_is_injective_in_every_component() {
        let base: ResourceKey = (7, 11, 13);
        assert_ne!(base, (8, 11, 13), "targets generation dropped from the key");
        assert_ne!(
            base,
            (7, 12, 13),
            "atmosphere generation dropped from the key"
        );
        assert_ne!(base, (7, 11, 14), "GI generation dropped from the key");
        assert_eq!(base, (7, 11, 13));
    }

    /// The cloud bake's key type, given the same pointer-identity treatment as
    /// [`ResourceKey`] above — GPU-free and deterministic, so it runs on every CI
    /// leg rather than only where an adapter exists.
    ///
    /// [`super::cloud_bake::CloudBakeNode`] keys its two bind groups on the bare
    /// `AtmosphereResources::generation` (a `u64`) rather than on the full
    /// [`ResourceKey`], because nothing it binds is viewport-size-dependent —
    /// rebuilding a compute bind group on every window resize would be a lie
    /// about what invalidates it. What that buys is only worth having if a
    /// generation bump *does* rebuild, which is what this pins.
    ///
    /// The failure this guards is the one the golden
    /// `cloud_quality_switch_rebuilds_the_cloud_binds` catches from outside: a
    /// stale bake bind group keeps writing into the *previous* tier's texture
    /// views while the newly created ones stay zeroed — and wgpu keeps the old
    /// textures alive as long as the bind group references them, so there is no
    /// validation error and no black frame to notice.
    #[test]
    fn the_cloud_bake_key_rebuilds_on_a_generation_bump() {
        use std::rc::Rc;

        let mut cache: GenCache<u64, Rc<u32>> = GenCache::default();
        let mut builds = 0u32;
        let fetch = |cache: &mut GenCache<u64, Rc<u32>>, gen: u64, n: &mut u32| {
            let counter = &mut *n;
            cache
                .get_or_build(gen, || {
                    *counter += 1;
                    Rc::new(*counter)
                })
                .clone()
        };

        let first = fetch(&mut cache, 1, &mut builds);
        let again = fetch(&mut cache, 1, &mut builds);
        assert!(
            Rc::ptr_eq(&first, &again),
            "an unchanged generation rebuilt"
        );
        assert_eq!(builds, 1);

        // The whole point: a quality clamp recreates the cloud textures and bumps
        // the generation, and the bind group must follow.
        let after = fetch(&mut cache, 2, &mut builds);
        assert!(
            !Rc::ptr_eq(&after, &first),
            "a generation bump did not rebuild the cloud bake bind group"
        );
        assert_eq!(builds, 2);

        // Returning to a previously-seen generation still rebuilds: one slot, not
        // a map, so it can never hand back a handle to a resource that has since
        // been recreated under the same number.
        let back = fetch(&mut cache, 1, &mut builds);
        assert!(!Rc::ptr_eq(&back, &first));
        assert_eq!(builds, 3);
    }

    /// **Half** of why P17.3 could add a third resizable resource to `EnvBinding`
    /// — the cloud shadow map — without adding a third key component: the cloud
    /// tier is a *total, injective, deterministic function of* the atmosphere
    /// tier, so the atmosphere generation distinguishes every distinct set of
    /// cloud textures that can exist.
    ///
    /// That is all this proves, and the name now says so. The other half — that
    /// nothing assigns a cloud tier *outside* the constructor that bumps the
    /// generation — is a property of the code rather than of the mapping, and is
    /// pinned separately by
    /// `sky_lut::tests::cloud_quality_is_only_assigned_at_construction`. Both are
    /// needed: an injective mapping assigned in two places would still let the
    /// textures change under an unchanged generation.
    #[test]
    fn cloud_quality_is_a_total_function_of_atmosphere_quality() {
        use crate::atmosphere::AtmosphereQuality;
        use crate::clouds::CloudQuality;

        // Injective: two atmosphere tiers never share a cloud tier, so the
        // atmosphere generation distinguishes every distinct cloud resource set.
        let tiers = [
            AtmosphereQuality::Low,
            AtmosphereQuality::Medium,
            AtmosphereQuality::High,
        ];
        let mut seen: Vec<(u32, u32, u32)> = Vec::new();
        for q in tiers {
            let c = CloudQuality::from_atmosphere(q);
            let sizes = (c.shape_res(), c.detail_res(), c.shadow_res());
            assert!(
                !seen.contains(&sizes),
                "{q:?} shares its cloud texture sizes with another tier — the \
                 atmosphere generation would no longer distinguish them"
            );
            seen.push(sizes);
        }
        // ...and deterministic: the same atmosphere tier always yields the same
        // cloud tier, so a rebuild reproduces the same sizes.
        for q in tiers {
            assert_eq!(
                CloudQuality::from_atmosphere(q),
                CloudQuality::from_atmosphere(q)
            );
        }
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
            // `gi_probes` moved into SHADER_TABLE in P18.4 — it is a composed
            // module now (it includes the atmosphere library for its sky term).
            // `vgeom_cull` followed in P18.5, when the HZB occlusion test moved
            // into `hzb_occlusion.wgsl` so the scatter cull could share it.
            ("vgeom_hzb", include_str!("../shaders/vgeom_hzb.wgsl")),
            ("shadow_depth", include_str!("../shaders/shadow_depth.wgsl")),
            ("tonemap", include_str!("../shaders/tonemap.wgsl")),
        ] {
            validate(label, source);
        }
    }
}
