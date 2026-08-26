//! The engine renderer: owns frame targets, the shared view bind group, and
//! the render graph. One instance per output (editor viewport, headless test).
//!
//! ## HDR pipeline (P13.3a)
//!
//! Scene passes render into a linear **`Rgba16Float` HDR** MSAA target (no
//! per-pass tonemapping anymore — the ACES tonemap is a single post step). The
//! frame flows:
//!
//! ```text
//! sky → [depth prepass] → [SSAO] → mesh/skinned/terrain (sample AO) → grid →
//! sprite → debug → resolve(MSAA→scene_hdr) → [TAA] → bloom → tonemap(→LDR) →
//! ui → mask → composite(→swapchain)
//! ```
//!
//! Bracketed passes are gated by [`RenderSettings`]; at the defaults (bloom off,
//! SSAO off, TAA off, exposure 1.0) the post chain is just ACES tonemap + dither,
//! producing the same look the in-shader tonemap used to (the goldens were
//! regenerated once for the float-pipeline reorder).

use glam::{Mat4, Vec3};

use crate::camera::{RenderView, ViewUniforms, DEPTH_FORMAT};
use crate::gpu::GpuContext;
use crate::graph::RenderGraph;
use crate::passes;
use crate::passes::gi::GiResources;
use crate::passes::shadow::ShadowResources;
use crate::passes::sky_lut::AtmosphereResources;
use crate::passes::vgeom::{VgeomAudit, VgeomAuditResources};
use crate::scene::RenderScene;
use crate::settings::{halton_jitter, mip_chain_sizes, RenderSettings};
use crate::wetness::{pack_wetness, WetnessResources};

/// Viewport shading view mode (R-P2). Editor-transient renderer state — never
/// persisted, never touched by the player, so it lives on [`EngineRenderer`] and
/// NOT in [`RenderSettings`] (the parity law is not implicated).
///
/// * `Lit` — the full PBR/lit path (the default; every golden except `unlit`).
/// * `Unlit` — the lit scene passes short-circuit to albedo+emissive (no lighting);
///   driven by the `flags.x` uniform, so no pipeline swap is needed.
/// * `Wireframe` — Unlit shading rendered through a `PolygonMode::Line` pipeline
///   variant. Requires [`wgpu::Features::POLYGON_MODE_LINE`]; when the adapter
///   lacks it, [`EngineRenderer::set_view_mode`] degrades this to `Unlit`.
/// * `Biomes` (P19.2) — terrain tinted by its per-sample biome id through
///   `RenderTerrain::biome_palette`; driven by the `flags.y` uniform. It also sets
///   `flags.x`, so every **other** kind of geometry in the frame renders unlit —
///   the smallest honest treatment: a biome id is a terrain-only concept, and
///   flat-shading the rest keeps the painted map readable without pretending a
///   mesh has a biome. Needs no GPU feature, so it is never degraded.
/// * `VtResidency` (P26.5) — every virtual-textured surface painted by **how far
///   behind the streamer is** at that pixel (`vt_sample.wgsl`'s `vt_heat`, off
///   the same `vt_resolve` the sampling path runs); driven by the `flags.z`
///   uniform. It sets `flags.x` too, on the `Biomes` precedent and for the same
///   reason: residency is a property of a virtual texture, and flat-shading
///   everything else keeps the ramp readable instead of competing with it. Needs
///   no GPU feature, so it is never degraded — a scene with no virtual textures
///   paints entirely grey, which is the correct answer rather than a failure.
/// * `VsmPages` (P27.5) — the same idea one virtual system over: every surface
///   painted by **how far behind the shadow-page residency is** at that pixel
///   (`vsm_receive.wgsl`'s `vsm_heat`), with a distinct colour for the state
///   that has no colour in a lit frame at all — a page with no resident
///   ancestor, which the receiver reads as *lit*. Driven by `flags.w`, sets
///   `flags.x` on the same precedent, never degraded, and grey on a scene with
///   no shadow tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    #[default]
    Lit,
    Unlit,
    Wireframe,
    Biomes,
    VtResidency,
    VsmPages,
}

impl ViewMode {
    /// The `View.flags.x` uniform value: `1.0` when the lit passes must skip
    /// lighting (Unlit, Wireframe **and** Biomes all shade unlit), else `0.0`.
    ///
    /// Biomes rides this flag deliberately: the four mesh shaders
    /// (`mesh`/`skinned_mesh`/`vgeom_mesh`/`scatter_mesh`) then need no edit at
    /// all, and non-terrain geometry drops to albedo instead of competing with the
    /// tint for attention.
    pub fn unlit_flag(self) -> f32 {
        match self {
            ViewMode::Lit => 0.0,
            ViewMode::Unlit
            | ViewMode::Wireframe
            | ViewMode::Biomes
            | ViewMode::VtResidency
            | ViewMode::VsmPages => 1.0,
        }
    }

    /// The `View.flags.y` uniform value: `1.0` **only** for Biomes (P19.2), which
    /// is what makes the terrain shader's tint branch present-but-false — and its
    /// arithmetic instruction-for-instruction unchanged — in every other mode.
    pub fn biomes_flag(self) -> f32 {
        match self {
            ViewMode::Biomes => 1.0,
            ViewMode::Lit
            | ViewMode::Unlit
            | ViewMode::Wireframe
            | ViewMode::VtResidency
            | ViewMode::VsmPages => 0.0,
        }
    }

    /// The `View.flags.z` uniform value: `1.0` **only** for
    /// [`VtResidency`](Self::VtResidency) (P26.5).
    ///
    /// Its own flag rather than a second meaning for `flags.y`, because the two
    /// overlays answer different questions about different geometry — a terrain
    /// biome tint and a virtual-texture residency ramp can both be wanted and
    /// only one can be shown, and a shader made to disambiguate one float is the
    /// place that eventually gets it wrong.
    pub fn vt_residency_flag(self) -> f32 {
        match self {
            ViewMode::VtResidency => 1.0,
            ViewMode::Lit
            | ViewMode::Unlit
            | ViewMode::Wireframe
            | ViewMode::Biomes
            | ViewMode::VsmPages => 0.0,
        }
    }

    /// The `View.flags.w` uniform value: `1.0` **only** for
    /// [`VsmPages`](Self::VsmPages) (P27.5).
    ///
    /// Its own flag, on `vt_residency_flag`'s precedent and for its reason: a
    /// texture-residency ramp and a shadow-page-residency ramp can both be
    /// wanted and only one can be shown, and a shader made to disambiguate one
    /// float is the place that eventually gets it wrong.
    pub fn vsm_pages_flag(self) -> f32 {
        match self {
            ViewMode::VsmPages => 1.0,
            ViewMode::Lit
            | ViewMode::Unlit
            | ViewMode::Wireframe
            | ViewMode::Biomes
            | ViewMode::VtResidency => 0.0,
        }
    }

    /// Whether the mesh passes should select their `PolygonMode::Line` pipeline.
    pub fn wireframe(self) -> bool {
        matches!(self, ViewMode::Wireframe)
    }
}

/// Offscreen HDR scene-colour format: linear `Rgba16Float`. All scene passes
/// render into this (MSAA) target; the tonemap post pass converts to the LDR
/// swapchain format. Fixed regardless of the output format so shading + goldens
/// behave identically everywhere.
pub const SCENE_FORMAT: wgpu::TextureFormat = HDR_FORMAT;
/// The HDR intermediate format (scene colour, resolve, bloom, TAA history).
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Display-referred LDR the tonemap writes and the composite reads.
pub const LDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Selection/hover mask format.
pub const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
/// Single-channel SSAO target format (half-res).
pub const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

/// Format of the half-res cloud march target and its temporal history (SKY2).
///
/// The same HDR format the scene uses, because that is what it carries: the
/// march accumulates premultiplied radiance in the scene's units, and dropping
/// it to something narrower here would clip a sunlit cloud top before the
/// tonemap ever saw it.
pub const CLOUD_FORMAT: wgpu::TextureFormat = HDR_FORMAT;

/// Format of the half-res cloud **distance** target (SKY2): `r` = the
/// coverage-weighted mean distance to the cloud, `g` = the distance the march
/// clamped at, both in metres.
///
/// `Rg16Float` because both are distances a kilometre or two out, where an f16's
/// ~3 significant digits are metres of error on a value the temporal
/// reprojection and the bilateral upsample both tolerate an order of magnitude
/// more of. It also fixes the sentinel: f16 tops out at 65 504, so "no geometry"
/// is 60 000 m rather than an infinity.
pub const CLOUD_DIST_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
/// 4× MSAA — guaranteed support for all formats we use.
pub const SCENE_SAMPLES: u32 = 4;
/// Bloom downsample mip-chain depth (levels beyond half-res).
pub const BLOOM_MAX_MIPS: u32 = 6;

/// How many (surface × bound map) feedback requests one frame may dispatch
/// (P26.4).
///
/// The request buffer is allocated once at this size, so it is a VRAM number
/// (4 096 × 32 B = 128 KiB) rather than a quality one: past it, the tail of the
/// coverage list simply does not mark, and those surfaces stay on the analytic
/// floor — which is the correct degradation and the same one a dropped mask
/// produces. A scene with more than four thousand textured surfaces on screen at
/// once is a P28.3 arbitration problem, not a buffer-size one.
pub const VT_FEEDBACK_REQUEST_CAP: u32 = 4096;

/// Per-size GPU targets. Recreated when the scene size changes; `generation`
/// lets nodes cache bind groups against the current views.
pub struct FrameTargets {
    pub size: (u32, u32),
    pub generation: u64,
    /// MSAA HDR scene colour (scene passes render here).
    pub color_msaa: wgpu::TextureView,
    /// MSAA scene depth.
    pub depth: wgpu::TextureView,
    /// Resolved single-sample HDR scene colour (MSAA resolve target).
    pub scene_hdr: wgpu::TextureView,
    /// Display-referred LDR (tonemap output; composite input).
    pub scene_color: wgpu::TextureView,
    pub mask: wgpu::TextureView,
    /// Half-res SSAO output the lit passes sample (white when SSAO is off).
    pub ao: wgpu::TextureView,
    /// Half-res raw SSAO (before the blur).
    pub ao_raw: wgpu::TextureView,
    /// Single-sample sampleable scene depth (SSAO + TAA reprojection).
    pub depth_prepass: wgpu::TextureView,
    /// Bloom mip chain (index 0 = half-res; the tonemap adds mip 0).
    pub bloom: Vec<wgpu::TextureView>,
    pub bloom_sizes: Vec<(u32, u32)>,
    /// TAA history ping-pong (HDR).
    pub taa_history: [wgpu::TextureView; 2],
    /// Half-res AO target size in px (for the SSAO shader).
    pub ao_size: (u32, u32),
    /// **Half-res** cloud march output (SKY2): premultiplied radiance + coverage.
    pub cloud: wgpu::TextureView,
    /// Half-res cloud distances (SKY2). See [`CLOUD_DIST_FORMAT`].
    pub cloud_dist: wgpu::TextureView,
    /// Half-res cloud temporal history ping-pong (SKY2).
    pub cloud_history: [wgpu::TextureView; 2],
    /// Half-res cloud target size in px.
    pub cloud_size: (u32, u32),
}

impl FrameTargets {
    fn create(gpu: &GpuContext, size: (u32, u32), generation: u64) -> Self {
        let (width, height) = (size.0.max(1), size.1.max(1));
        let (aw, ah) = ((width / 2).max(1), (height / 2).max(1));
        let extent = |w: u32, h: u32| wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let tex = |label: &str, w: u32, h: u32, format, samples, usage: wgpu::TextureUsages| {
            gpu.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: extent(w, h),
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        const RT: wgpu::TextureUsages = wgpu::TextureUsages::RENDER_ATTACHMENT;
        const RT_TEX: wgpu::TextureUsages =
            wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING);

        let bloom_sizes = mip_chain_sizes(width, height, BLOOM_MAX_MIPS);
        let bloom: Vec<wgpu::TextureView> = bloom_sizes
            .iter()
            .map(|&(w, h)| tex("bloom-mip", w, h, HDR_FORMAT, 1, RT_TEX))
            .collect();

        Self {
            size: (width, height),
            generation,
            color_msaa: tex(
                "scene-color-msaa",
                width,
                height,
                HDR_FORMAT,
                SCENE_SAMPLES,
                RT,
            ),
            depth: tex(
                "scene-depth",
                width,
                height,
                DEPTH_FORMAT,
                SCENE_SAMPLES,
                // TEXTURE_BINDING as well as RENDER_ATTACHMENT (P17.3): the cloud
                // raymarch binds this as a `texture_depth_multisampled_2d` to
                // clamp its march at the nearest geometry, while the same view
                // stays bound as a READ-ONLY depth attachment so the hardware
                // test still rejects fully-occluded cloud fragments per sample.
                // Adding the usage changes no pixel — it only widens what the
                // texture may be bound as.
                RT_TEX,
            ),
            scene_hdr: tex("scene-hdr", width, height, HDR_FORMAT, 1, RT_TEX),
            scene_color: tex("scene-color", width, height, LDR_FORMAT, 1, RT_TEX),
            mask: tex("outline-mask", width, height, MASK_FORMAT, 1, RT_TEX),
            ao: tex("ssao", aw, ah, AO_FORMAT, 1, RT_TEX),
            ao_raw: tex("ssao-raw", aw, ah, AO_FORMAT, 1, RT_TEX),
            depth_prepass: tex("depth-prepass", width, height, DEPTH_FORMAT, 1, RT_TEX),
            taa_history: [
                tex("taa-history-0", width, height, HDR_FORMAT, 1, RT_TEX),
                tex("taa-history-1", width, height, HDR_FORMAT, 1, RT_TEX),
            ],
            bloom,
            bloom_sizes,
            ao_size: (aw, ah),
            // The cloud stack (SKY2), at the same half resolution SSAO uses.
            // Allocated unconditionally, like the TAA history beside it: at
            // 1080p the four targets are 960×540 and cost 4.15 + 2.07 + 2×4.15 =
            // **14.52 MB** against the ~33 MB that history already costs, and a
            // lazily-grown set would still need placeholders bound at every
            // binding to keep the bind groups valid. (The wave's ledger first
            // priced this at 13.5 MB by counting `Rg16Float` as two bytes a texel
            // rather than four; corrected at the SKY2 audit.)
            cloud: tex("cloud", aw, ah, CLOUD_FORMAT, 1, RT_TEX),
            cloud_dist: tex("cloud-dist", aw, ah, CLOUD_DIST_FORMAT, 1, RT_TEX),
            cloud_history: [
                tex("cloud-history-0", aw, ah, CLOUD_FORMAT, 1, RT_TEX),
                tex("cloud-history-1", aw, ah, CLOUD_FORMAT, 1, RT_TEX),
            ],
            cloud_size: (aw, ah),
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
    /// Active HDR/post settings for this frame.
    pub settings: &'a RenderSettings,
    /// Monotonic frame counter (drives the TAA jitter sequence + ping-pong).
    pub frame_index: u64,
    /// The HDR view bloom + tonemap read: the TAA output when TAA is on, else
    /// the resolved `scene_hdr`.
    pub post_hdr: &'a wgpu::TextureView,
    /// TAA history ping-pong: `prev` = last frame's accumulation, `cur` = the one
    /// the TAA node writes this frame (== `post_hdr` when TAA is on).
    pub taa_history_prev: &'a wgpu::TextureView,
    pub taa_history_cur: &'a wgpu::TextureView,
    /// Shared cascaded-shadow resources (P13.3b): the shadow node renders/writes
    /// them, the lit passes sample them (byte-neutral when shadows are off).
    pub shadow: &'a ShadowResources,
    /// Shared dynamic-GI resources (P13.3b): the GI node writes the SH probes, the
    /// lit passes sample them (byte-neutral when GI is off).
    pub gi: &'a GiResources,
    /// Occlusion-audit counters (P18.1): the vgeom cull compute increments them
    /// only when `enabled`, and the node records the readback copy. Off by
    /// default, so the shipping path pays nothing.
    pub vgeom_audit: &'a VgeomAuditResources,
    /// P28.1 visibility-buffer readback: off by default, and the only way an id
    /// ever leaves the GPU. See [`EngineRenderer::set_visbuffer_readback`].
    pub vis_readback: &'a std::cell::RefCell<passes::visbuffer::VisReadback>,
    /// Instance-cull audit counters (P18.5): the scatter cull compute increments
    /// them only when `enabled`, and the node records the readback copy. Off by
    /// default, so the shipping path pays nothing.
    pub scatter_audit: &'a passes::scatter::ScatterAuditResources,
    /// Shared atmosphere resources (P17.2): the two LUTs + the shared uniform the
    /// bake node writes and the sky/lit passes sample. **Resizable** — its
    /// `generation` is part of the `EnvBinding` cache key.
    pub atmosphere: &'a AtmosphereResources,
    /// **The matrix this frame's depth was rasterized with** — jittered whenever
    /// TAA is on, and identical to what `view.view_proj` holds in the uniform.
    ///
    /// Exposed (P28.1) because a COMPUTE pass cannot read the view uniform: the
    /// view bind group is vertex/fragment-visible, and a reconstruction that
    /// rebuilt a projection from [`RenderView`] instead would land on a different
    /// sub-pixel grid than its own buffer — the P27.1 jitter law, which the
    /// shadow marking pass obeys by being handed `jvp.inverse()` for exactly the
    /// same reason.
    pub view_proj: [f32; 16],
    /// Jittered view-projection of the **previous** frame (TAA reprojection).
    pub taa_prev_view_proj: [f32; 16],
    /// The half-res cloud view the composite upsamples: `cloud_history[cur]`
    /// when [`crate::RenderSettings::cloud_temporal`] is on, else the raw march
    /// in `targets.cloud` (SKY2).
    pub cloud_src: &'a wgpu::TextureView,
    /// Which slot `cloud_src` points at — `cur` with the temporal pass on, and a
    /// fixed sentinel with it off. The composite's bind-group cache keys on it,
    /// because a ping-pong slot is a change the passes' `ResourceKey` cannot
    /// see.
    pub cloud_slot: usize,
    pub cloud_history_prev: &'a wgpu::TextureView,
    pub cloud_history_cur: &'a wgpu::TextureView,
    /// Whether the previous frame's cloud history is usable this frame.
    pub cloud_history_valid: bool,
    /// False on the first frame / after a resize (history has nothing usable).
    pub taa_history_valid: bool,
    /// Shoreline wetness (P20.3): the fixed-size uniform `EngineRenderer::render`
    /// packs from `scene.waters` and the lit passes read through `EnvBinding`.
    /// **Not resizable** — see `passes::EnvBinding::bind_group`'s invariant.
    pub wetness: &'a WetnessResources,
    /// The P22.1 deformation window (texture + uniform), created once. See
    /// [`crate::deform`].
    pub deform: &'a crate::deform::DeformResources,
    /// Active shading view mode (R-P2). The lit scene passes select their
    /// wireframe pipeline variant on [`ViewMode::wireframe`]; the unlit branch is
    /// driven by the `View.flags.x` uniform instead (so no swap for Unlit).
    pub view_mode: ViewMode,
    /// The live virtual-texture pool (P26.3), when the host has given the
    /// renderer one. `None` on every textureless scene — including all 50
    /// committed goldens — and then the env bind group names
    /// [`vt_absent`](Self::vt_absent) instead.
    pub vt: Option<&'a crate::vt::VtPools>,
    /// The permanent EMPTY virtual-texture surface: what the env bind group
    /// names when [`vt`](Self::vt) is `None`. See [`crate::vt::VtEmptyPool`].
    pub vt_absent: &'a crate::vt::VtEmptyPool,
    /// The live virtual-shadow-map system (P27.4), when this frame has one.
    /// `None` on every scene with virtual shadows off or with no shadow-casting
    /// light — including all fifty committed goldens — and then the env bind
    /// group names [`vsm_absent`](Self::vsm_absent) instead.
    pub vsm: Option<&'a crate::vsm_mark::VsmSystem>,
    /// The permanent EMPTY shadow surface. See
    /// [`crate::vsm_receiver::VsmEmptyPool`].
    pub vsm_absent: &'a crate::vsm_receiver::VsmEmptyPool,
    /// The receiver's uniform (P27.4), packed at the frame's sync point.
    pub vsm_receiver: &'a crate::vsm_receiver::VsmReceiverResources,
    /// Per **scene light index**, the projection slot that light's tree starts
    /// at **+ 1** — 0 for a light with no tree. Rides into `GpuLight::params.w`,
    /// which every lit pass's point/spot branch reads. Empty when the frame has
    /// no system, which is what keeps `params.w` at the `0.0` it has held since
    /// P7.1 and every pre-P27.4 golden byte-identical.
    pub vsm_light_slots: &'a [u32],
    /// **The per-fragment virtual-texture feedback handover** (P28.1). `Some`
    /// only when the frame has a live VT pool AND the visibility path is on, and
    /// then it names the very mask `VtFeedback::record` cleared moments earlier
    /// and `VtFeedback::finish` will hand to the ring after the graph. The
    /// visibility pass ORs into it from inside the graph; nothing else does.
    pub vt_feedback: Option<VtFeedbackFrame<'a>>,
}

/// What a graph node needs to mark virtual-texture tiles (P28.1).
///
/// A struct rather than four `FrameData` fields because the four are meaningless
/// apart: a mask without its word count is unbounded, and a bit base table for a
/// different layout addresses the wrong texture.
#[derive(Clone, Copy)]
pub struct VtFeedbackFrame<'a> {
    /// The order-independent coverage bitmask (`inf_vt::feedback`'s layout).
    pub mask: &'a wgpu::Buffer,
    /// Per texture handle, the first mask bit — `VtFeedbackLayout::texture_base`.
    pub bases: &'a [u32],
    /// `VtFeedbackLayout::words`, the bound a marker must respect.
    pub words: u32,
    /// The indirection table the marks address through. The **same buffer** the
    /// env group binds, so producer and sampler cannot disagree about a mip.
    pub table: &'a wgpu::Buffer,
}

impl FrameData<'_> {
    /// The atlas the env bind group names this frame.
    #[inline]
    pub(crate) fn vt_atlas(&self) -> &wgpu::TextureView {
        self.vt
            .map_or_else(|| self.vt_absent.view(), |p| p.atlas_view())
    }
    /// The page sampler the env bind group names this frame.
    #[inline]
    pub(crate) fn vt_sampler(&self) -> &wgpu::Sampler {
        self.vt
            .map_or_else(|| self.vt_absent.sampler(), |p| p.sampler())
    }
    /// The indirection buffer the env bind group names this frame.
    #[inline]
    pub(crate) fn vt_table(&self) -> &wgpu::Buffer {
        self.vt
            .map_or_else(|| self.vt_absent.table(), |p| p.table())
    }
    /// The VT component of [`crate::passes::ResourceKey`].
    ///
    /// **0 when there is no pool, and never 0 when there is**: `inf-vt`'s stamp
    /// counter starts at 1 and only ever rises, so "absent" cannot collide with
    /// any live pool generation and a pool appearing or vanishing invalidates
    /// the cached bind group on its own.
    #[inline]
    pub(crate) fn vt_generation(&self) -> u64 {
        self.vt.map_or(0, |p| p.table_generation())
    }

    /// The shadow atlas the env bind group names this frame.
    #[inline]
    pub(crate) fn vsm_atlas(&self) -> &wgpu::TextureView {
        self.vsm
            .map_or_else(|| self.vsm_absent.view(), |v| v.pools().atlas_view())
    }
    /// The shadow indirection buffer the env bind group names this frame.
    #[inline]
    pub(crate) fn vsm_table(&self) -> &wgpu::Buffer {
        self.vsm
            .map_or_else(|| self.vsm_absent.table(), |v| v.pools().table())
    }
    /// This frame's per-(light × face) projection list, on the GPU.
    #[inline]
    pub(crate) fn vsm_projections(&self) -> &wgpu::Buffer {
        self.vsm.map_or_else(
            || self.vsm_absent.projections(),
            |v| v.marker().projection_buffer(),
        )
    }
    /// The receiver's uniform — the renderer's own when a system is live, the
    /// permanently-zero one otherwise, so a frame that switches the feature off
    /// cannot keep last frame's blend band or sun slot.
    #[inline]
    pub(crate) fn vsm_params(&self) -> &wgpu::Buffer {
        if self.vsm.is_some() {
            &self.vsm_receiver.uniform
        } else {
            self.vsm_absent.params()
        }
    }
    /// The VSM component of [`crate::passes::ResourceKey`].
    ///
    /// **0 when there is no system, and never 0 when there is**: `inf-vsm`'s
    /// stamp counter starts at 1 and only ever rises, so "absent" cannot collide
    /// with any live layout generation and a system appearing or vanishing
    /// invalidates the cached bind group on its own — the `vt_generation`
    /// argument, met again for three resources instead of one.
    #[inline]
    pub(crate) fn vsm_generation(&self) -> u64 {
        self.vsm.map_or(0, |v| v.pools().table_generation())
    }
}

pub struct EngineRenderer {
    view_buf: wgpu::Buffer,
    view_bg: wgpu::BindGroup,
    pub view_bgl: wgpu::BindGroupLayout,
    targets: Option<FrameTargets>,
    next_generation: u64,
    graph: RenderGraph,
    out_format: wgpu::TextureFormat,
    settings: RenderSettings,
    /// Active shading view mode (R-P2); editor-transient, never persisted.
    view_mode: ViewMode,
    /// Whether the device has `POLYGON_MODE_LINE` (probed once at construction).
    /// Gates the wireframe pipeline variants + clamps a Wireframe request down to
    /// Unlit when absent.
    polygon_mode_line: bool,
    /// Latched so the "wireframe unsupported → Unlit" degrade logs exactly once.
    wireframe_warned: bool,
    frame_index: u64,
    /// Jittered view-proj we rendered last frame (TAA reprojection source).
    prev_view_proj: Option<[f32; 16]>,
    /// Shared shadow GPU resources (created once, independent of viewport size;
    /// the shadow graph node writes them, the lit passes sample them).
    shadow: ShadowResources,
    /// Shared GI voxel/SH buffers + uniform. **Recreated** when
    /// [`crate::gi::GiQuality`] changes (P18.4), which is why they carry a
    /// generation the env bind-group cache keys on — exactly like the atmosphere.
    gi: GiResources,
    /// Monotonic source of `gi.generation`.
    next_gi_generation: u64,
    /// What the GI voxelizer consumed on the last rendered frame (P18.4).
    gi_audit: passes::gi::SharedGiAudit,
    /// P18.1 occlusion-audit counters (see [`EngineRenderer::set_vgeom_audit`]).
    vgeom_audit: VgeomAuditResources,
    /// P28.1 visibility-buffer readback. A `RefCell` because the node records the
    /// copy through a `&FrameData` and may have to grow the staging buffer while
    /// doing it — the same interior mutability `GenCache` needs and for the same
    /// reason, one indirection lower.
    vis_readback: std::cell::RefCell<passes::visbuffer::VisReadback>,
    /// P18.2 meshlet-streaming state, published by the vgeom node each frame.
    vgeom_stream: passes::vgeom::SharedStreamReport,
    /// P28.1: the visibility path's frame/refusal counters, published by the node.
    vis_report: passes::visbuffer::SharedVisReport,
    /// P18.5 instance-cull counters (see [`EngineRenderer::set_scatter_audit`]).
    scatter_audit: passes::scatter::ScatterAuditResources,
    /// The island-wave-I4 per-pass GPU stopwatch, or `None` — which it is on
    /// every frame nobody asked to measure. See
    /// [`EngineRenderer::set_gpu_timing`] and [`crate::timing`].
    frame_timer: Option<crate::timing::FrameTimer>,
    /// The last frame's **record-path** phase breakdown (island wave I7b),
    /// zeroed unless `frame_timer` is armed. See [`crate::timing::RecordProfile`]
    /// — the per-pass record column tiles the marked span, and this tiles the
    /// whole of [`EngineRenderer::render`], including everything before the
    /// frame's first command and everything after its last.
    record_profile: crate::timing::RecordProfile,
    /// Shared atmosphere LUTs + uniform (P17.2). Unlike `shadow`/`gi` these are
    /// **recreated** when [`crate::atmosphere::AtmosphereQuality`] changes, which
    /// is why they carry a generation the env bind-group cache keys on.
    atmosphere: AtmosphereResources,
    /// Monotonic source of `atmosphere.generation`.
    next_atmosphere_generation: u64,
    /// Shoreline-wetness uniform (P20.3). Created once, never resized — unlike
    /// `atmosphere`/`gi` it therefore carries no generation, which is exactly the
    /// exclusion `passes::EnvBinding::bind_group`'s invariant grants and explains.
    wetness: WetnessResources,
    /// The P22.1 surface-deformation window: one R32Float texture + one uniform.
    /// Created once at a **constant** size and never resized, so like `wetness`
    /// it carries no generation and adds no `passes::ResourceKey` component —
    /// see `crate::deform` for the rule quoted verbatim.
    deform: crate::deform::DeformResources,
    /// P20.3 underwater engagement counter (see
    /// [`EngineRenderer::underwater_engaged_frames`]).
    underwater: passes::underwater::UnderwaterReport,
    /// P21.1 voxel engagement counter (see
    /// [`EngineRenderer::voxel_engaged_frames`]).
    voxel: passes::voxel::VoxelReport,
    /// The live virtual-texture GPU mirror (P26.3), when a host has handed one
    /// over through [`EngineRenderer::set_vt_pools`]. The renderer OWNS it —
    /// rather than borrowing it per frame — because that is what lets the env
    /// bind group's cache key read the pool's generation without threading a
    /// fourth argument through `render`, and because `apply` must run before the
    /// frame's submit, which is an ordering this type is already responsible for.
    vt: Option<crate::vt::VtPools>,
    /// The level's virtual-texture **registry** (P26.4), set beside the mirror
    /// through [`EngineRenderer::set_vt_level`]. `None` on a textureless scene
    /// and whenever a host set only the mirror (the P26.3 door), in which case
    /// every projector's material lookup is `VtTextureSet::NONE`.
    vt_textures: Option<crate::VtTextures>,
    /// The P26.4 coverage-feedback pass + its readback ring. Built lazily
    /// against the registry's bitmask layout and rebuilt when a registration
    /// grows it (a mask sized for four textures cannot address a fifth).
    vt_feedback: Option<crate::vt_stream::VtFeedback>,
    /// Per texture handle, the first mask bit — recomputed with the layout, so a
    /// registration that grows the mask cannot leave a stale base behind.
    vt_feedback_bases: Vec<u32>,
    /// Requests the per-surface pass dispatched this frame; `VtFeedback::finish`
    /// hands the mask to the ring only when something produced marks.
    vt_feedback_dispatched: u32,
    /// P26.4 pop-in instruments — per tile class. See [`crate::VtPopIn`].
    vt_pop_in: crate::VtPopIn,
    /// **The committed camera history the predictor reads** (P28.4).
    ///
    /// Empty until a host calls [`EngineRenderer::commit_camera`], and that is
    /// the design rather than a lazy initialization: the history is the one
    /// place the "committed input" premise can be enforced, so a host that
    /// cannot honour it simply never fills this and gets the pre-P28.4
    /// streaming loop exactly. The editor viewport is that host.
    camera_history: inf_math::CameraHistory,
    /// **The one readback ledger** (P28.3): the two feedback consumers' hit and
    /// miss counts, under one pinned latency, so a gate can assert the property
    /// two independent rings cannot state — that both read the SAME source
    /// frame. Its `Geometry` line reads zero for ever, and that is a statement:
    /// the meshlet streamer's wants are CPU-derived by ruling
    /// (`inf_vgeom::stream`'s "why not cull feedback"), so it has no ring.
    stream_ring: inf_stream::RingLedger,
    /// The empty VT surface, created once and never recreated (so it adds no
    /// `passes::ResourceKey` component, on the `wetness` precedent).
    vt_absent: crate::vt::VtEmptyPool,
    /// How many frames were drawn with a VT pool bound — the P26.3 **engagement
    /// counter**. Zero on every textureless scene, which is what makes "the
    /// goldens did not change" a measurement rather than an expectation.
    vt_engaged_frames: u64,
    /// The P27.1 virtual-shadow-map system: the page residency, its atlas mirror
    /// and the marking pass. `None` unless `settings.vsm.enabled` **and** the
    /// scene carries a shadow-casting light — which is no scene by default, so
    /// the atlas is not even allocated on a frame that does not want it.
    vsm: Option<crate::vsm_mark::VsmSystem>,
    /// How many frames recorded the shadow-page marking pass — the P27.1
    /// **engagement counter**. Zero is the assertion on every scene that does
    /// not run VSM, and it is a claim about the command stream that no pixel
    /// comparison can make.
    vsm_engaged_frames: u64,
    /// Frames that rasterized at least one shadow PAGE (P27.2). A separate
    /// counter from the marking one on purpose: a frame can mark pages it has no
    /// caster for, and a frame can rasterize pages while its mask misses.
    vsm_raster_frames: u64,
    /// The empty shadow surface, created once and never recreated (so it adds no
    /// `passes::ResourceKey` component, on the `vt_absent` precedent).
    vsm_absent: crate::vsm_receiver::VsmEmptyPool,
    /// The receiver's uniform, created once and only written (P27.4).
    vsm_receiver: crate::vsm_receiver::VsmReceiverResources,
    /// The params packed into it this frame — kept on the CPU side too, because
    /// a gate that asserts which tree the sun reads cannot read a GPU uniform
    /// back cheaply and must not re-derive the answer it is checking.
    vsm_receiver_params: crate::vsm_receiver::VsmReceiverParams,
    /// This frame's per-scene-light projection slots (+1), rebuilt at the sync
    /// point. Held rather than passed so the borrow in `FrameData` is of a field
    /// and not of a temporary.
    vsm_light_slots: Vec<u32>,
    /// Frames drawn with the shadow atlas BOUND to the lit passes — the P27.4
    /// **engagement counter**, and the third of the phase's three.
    ///
    /// It is a different claim from the other two and it is the one the ROADMAP's
    /// "the setting stops being inert" rests on: marking says a pixel asked for a
    /// page, rastering says a caster filled one, and this says a lit fragment was
    /// in a position to read it. Zero on every scene with virtual shadows off,
    /// which is what makes "no golden moved" a measurement.
    vsm_receiver_frames: u64,
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

        let settings = RenderSettings::default();
        let atmosphere = AtmosphereResources::new(gpu, settings.atmosphere.quality, 1);

        let mut graph = RenderGraph::default();
        // Atmosphere LUT bake (P17.2): compute-only, so it touches no colour
        // target and safely precedes the sky pass that samples its output. A
        // no-op (not even a uniform write, after the first frame) unless the
        // scene carries an enabled `AtmosphereParams`.
        graph.add(passes::sky_lut::AtmosphereNode::new(gpu));
        // Cloud noise volumes + cloud-shadow map (P17.3): compute-only, so it
        // touches no colour target. It must precede BOTH the lit passes (which
        // sample the shadow map through `EnvBinding`) and the cloud raymarch
        // (which samples the volumes). A no-op unless the scene enables clouds.
        graph.add(passes::cloud_bake::CloudBakeNode::new(gpu));
        graph.add(passes::sky::SkyNode::new(gpu, &view_bgl));
        // Cascaded shadow maps (P13.3b): renders the first directional light's
        // cascades + publishes the shared shadow uniform. A no-op (uniform only)
        // unless RenderSettings.shadows is enabled.
        graph.add(passes::shadow::ShadowNode::new(gpu));
        // Dynamic GI (P13.3b, rebuilt P18.4): voxelize the scene (rigid + skinned
        // + vgeom + terrain) and march the probe grid to SH. It must follow the
        // atmosphere bake, whose sky-view LUT it now samples for the ray-miss term.
        // A no-op (uniform only) unless RenderSettings.gi is enabled.
        let gi_audit = passes::gi::SharedGiAudit::default();
        graph.add(passes::gi::GiNode::new(gpu, gi_audit.clone()));
        // SSAO/TAA scene-depth prepass (rigid meshes only); a no-op unless SSAO
        // or TAA is enabled. Runs before SSAO so the AO can sample it, and before
        // the lit passes so they can multiply AO into their ambient term.
        graph.add(passes::depth_prepass::DepthPrepassNode::new(gpu, &view_bgl));
        graph.add(passes::ssao::SsaoNode::new(gpu, &view_bgl));
        graph.add(passes::mesh::MeshNode::new(gpu, &view_bgl));
        // GPU-driven virtualized-geometry (meshlet) path (P13.1b). Runs right
        // after the rigid mesh pass, into the same MSAA scene targets; a no-op
        // unless RenderSettings.vgeom is enabled and the scene carries vmesh
        // instances (so the classic path stays byte-identical).
        let vgeom_stream = passes::vgeom::SharedStreamReport::default();
        let vis_report = passes::visbuffer::SharedVisReport::default();
        graph.add(passes::vgeom::VgeomNode::new(
            gpu,
            &view_bgl,
            vgeom_stream.clone(),
            vis_report.clone(),
        ));
        // Classic discrete-LOD fallback (P13.4): renders the SAME vgeom content as
        // the meshlet path but through the ordinary PBR mesh pipeline, only when
        // RenderSettings.vgeom is DISABLED (the auto-tier picks the path). The exact
        // complement of VgeomNode, so scenes without vgeom content stay byte-stable.
        graph.add(passes::classic_vgeom::ClassicVgeomNode::new(gpu, &view_bgl));
        graph.add(passes::skinned::SkinnedMeshNode::new(gpu, &view_bgl));
        graph.add(passes::terrain::TerrainNode::new(gpu, &view_bgl));
        // Volumetric terrain (P21.1): the meshed isosurface of the scene's SDF
        // voxel volumes. Beside the terrain in the opaque block on purpose — a
        // voxel surface is opaque geometry that LOCALLY REPLACES the heightfield
        // (a cave mouth, an excavation, an overhang), so it belongs with the
        // ground it extends rather than after the passes that composite over the
        // ground. A no-op on a scene with no volumes — the node returns before
        // touching the encoder — so every existing golden is untouched.
        let voxel = passes::voxel::VoxelReport::default();
        graph.add(passes::voxel::VoxelNode::new(gpu, &view_bgl, voxel.clone()));
        // Fracture debris (P22.3): the chunks a destructible broke into. Beside
        // the mesh pass in the opaque block — a chunk IS the wall it came off, so
        // it is lit by the same shader and written to the same depth — and after
        // the terrain/voxel ground it lands on. A no-op on a scene with nothing
        // broken (the node returns before touching the encoder), so every
        // existing golden is untouched.
        graph.add(passes::fracture::FractureNode::new(gpu, &view_bgl));
        // GPU-instanced scatter (P18.5): PCG volumes + painted foliage, culled
        // per-instance on the GPU with LOD/impostor banding. LAST of the opaque
        // passes on purpose — its HZB is built from the depth every other opaque
        // pass has already written, which is the richest occluder set available —
        // and still before clouds/precipitation/translucency, since scatter IS
        // opaque. A no-op on a scene with no `scatter` batches, so every existing
        // golden is untouched.
        graph.add(passes::scatter::ScatterNode::new(gpu, &view_bgl));
        // Volumetric clouds (P17.3, split three ways at wave SKY2): the HALF-RES
        // raymarch, an optional temporal accumulation against the cloud's own
        // history, and the full-res bilateral composite that meets the scene.
        // All three AFTER the opaque passes so the depth buffer can occlude the
        // clouds and clamp the march, and BEFORE translucency so glass
        // composites over them. Each is a no-op unless the scene enables clouds,
        // so opaque scenes stay byte-identical.
        graph.add(passes::cloud::CloudNode::new(gpu, &view_bgl));
        graph.add(passes::cloud_temporal::CloudTemporalNode::new(
            gpu, &view_bgl,
        ));
        graph.add(passes::cloud_composite::CloudCompositeNode::new(
            gpu, &view_bgl,
        ));
        // Precipitation (P17.4): a depth-tested, premultiplied-alpha billboard
        // layer around the camera. AFTER the clouds so rain composites over the
        // deck it falls out of, BEFORE translucency so glass composites over it.
        // A no-op unless the weather block is precipitating, so dry scenes stay
        // byte-identical.
        graph.add(passes::precip::PrecipNode::new(gpu, &view_bgl));
        // Water surfaces (P20.1): oceans, lakes and spline rivers, depth-tested
        // with READ-ONLY depth so the same buffer can be sampled for the water
        // column. AFTER every opaque pass (the column, the shore fade and the
        // refraction all read the scene behind the surface) and AFTER clouds and
        // rain (so a sea reflects the sky it is under), BEFORE translucency (so
        // glass composites over water like any other surface). A no-op unless the
        // scene carries water, so every existing golden stays byte-identical.
        graph.add(passes::water::WaterNode::new(gpu, &view_bgl));
        // Translucent forward pass (R-P5): alpha-blended, depth-tested but not
        // depth-writing, back-to-front sorted. Draws after all opaque geometry +
        // terrain, into the same MSAA scene target, before the grid. A no-op unless
        // the scene carries translucent instances (so opaque scenes stay
        // byte-identical).
        graph.add(passes::translucent::TranslucentNode::new(gpu, &view_bgl));
        // Underwater post (P20.3): a full-screen depth-graded fog + sun shafts,
        // applied when the camera is inside a water body. AFTER the water surface
        // (which is what the shafts are gathered from) and after translucency
        // (glass in the water is in the water), BEFORE the grid, the sprite layer
        // and the debug lines — editor furniture is not in the medium. A no-op
        // unless the camera is actually submerged, so every existing golden
        // (including the three P20.1 water ones, whose cameras are above their
        // water) records the command stream it always did.
        let underwater = passes::underwater::UnderwaterReport::default();
        graph.add(passes::underwater::UnderwaterNode::new(
            gpu,
            &view_bgl,
            underwater.clone(),
        ));
        graph.add(passes::grid::GridNode::new(gpu, &view_bgl));
        graph.add(passes::sprite::SpriteNode::new(gpu, &view_bgl));
        graph.add(passes::debug::DebugNode::new(gpu, &view_bgl));
        graph.add(passes::resolve::ResolveNode);
        // Post chain (all read/write single-sample HDR/LDR targets).
        graph.add(passes::taa::TaaNode::new(gpu, &view_bgl));
        graph.add(passes::bloom::BloomNode::new(gpu));
        graph.add(passes::tonemap::TonemapNode::new(gpu));
        // **The in-game UI, AFTER the tonemap** (island wave I5): a menu drawn
        // into the HDR scene target would be bloomed, temporally reprojected and
        // colour-graded along with the world behind it, and depth-tested against
        // the wall the player is standing next to. It composites over the
        // display-referred frame instead, in screen pixels, with `LoadOp::Load`.
        //
        // A **no-op** on every frame nobody opened a menu on: `RenderScene::ui`
        // is empty and the node returns before it touches the encoder. That is
        // what the frozen goldens rest on and it is asserted, not assumed.
        graph.add(passes::ui::UiNode::new(gpu));
        // The mask feeds the composite's outline dilate; it renders into the
        // single-sample mask target independently of the scene resolve.
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
            settings,
            view_mode: ViewMode::default(),
            polygon_mode_line: gpu
                .device
                .features()
                .contains(wgpu::Features::POLYGON_MODE_LINE),
            wireframe_warned: false,
            frame_index: 0,
            prev_view_proj: None,
            shadow: ShadowResources::new(gpu),
            gi: GiResources::new(gpu, settings.gi.quality, 1),
            next_gi_generation: 2,
            gi_audit,
            vgeom_audit: VgeomAuditResources::new(gpu),
            vis_readback: std::cell::RefCell::new(passes::visbuffer::VisReadback::new(gpu)),
            vgeom_stream,
            vis_report,
            scatter_audit: passes::scatter::ScatterAuditResources::new(gpu),
            frame_timer: None,
            record_profile: crate::timing::RecordProfile::default(),
            atmosphere,
            next_atmosphere_generation: 2,
            wetness: WetnessResources::new(gpu),
            deform: crate::deform::DeformResources::new(gpu),
            underwater,
            voxel,
            vt: None,
            vt_textures: None,
            vt_feedback: None,
            vt_feedback_bases: Vec::new(),
            vt_feedback_dispatched: 0,
            vt_pop_in: crate::VtPopIn::default(),
            camera_history: inf_math::CameraHistory::new(),
            stream_ring: inf_stream::RingLedger::default(),
            vt_absent: crate::vt::VtEmptyPool::new(&gpu.device),
            vt_engaged_frames: 0,
            vsm: None,
            vsm_engaged_frames: 0,
            vsm_raster_frames: 0,
            vsm_absent: crate::vsm_receiver::VsmEmptyPool::new(&gpu.device, &gpu.queue),
            vsm_receiver: crate::vsm_receiver::VsmReceiverResources::new(&gpu.device),
            vsm_receiver_params: crate::vsm_receiver::VsmReceiverParams::absent(),
            vsm_light_slots: Vec::new(),
            vsm_receiver_frames: 0,
        }
    }

    /// Hand the renderer the virtual-texture GPU mirror for the level it is
    /// drawing, or take it away.
    ///
    /// **The one door**, called identically by the editor viewport host and by
    /// the shipped player — the P16.6 mirror law applied to VT. Passing `None`
    /// restores the textureless path exactly: the env bind group names the empty
    /// surface, `vt_active()` is false in every lit shader, and the command
    /// stream is the one every pre-P26.3 golden recorded.
    pub fn set_vt_pools(&mut self, pools: Option<crate::vt::VtPools>) {
        self.vt = pools;
        self.vt_textures = None;
        self.vt_feedback = None;
    }

    /// **Hand the renderer a whole level's virtual-texture surface** (P26.4) —
    /// the registry and the mirror it was planned against, together.
    ///
    /// The pair is set through one door because they are one fact: a
    /// [`VtPools`](crate::vt::VtPools) built from a *different*
    /// [`VtTextures`](crate::VtTextures) has the wrong atlas geometry and the
    /// wrong table layout, and nothing downstream would notice. Passing `None`
    /// restores the textureless path exactly (see
    /// [`set_vt_pools`](Self::set_vt_pools)).
    ///
    /// The renderer owns the registry — rather than the host holding it and
    /// passing it per frame — for the same reason it already owns the mirror:
    /// the streaming step runs at the frame's sync point, inside `render`, and a
    /// projector that held the registry would have to be handed the camera to do
    /// it. Projectors ask [`crate::vt_set_for`] instead — the free function, over
    /// [`vt_textures`](Self::vt_textures) — which is a lookup and needs nothing,
    /// and which `projector_mirror` compares between the two hosts token for
    /// token.
    pub fn set_vt_level(&mut self, level: Option<(crate::VtTextures, crate::vt::VtPools)>) {
        match level {
            Some((textures, pools)) => {
                self.vt = Some(pools);
                self.vt_textures = Some(textures);
            }
            None => {
                self.vt = None;
                self.vt_textures = None;
            }
        }
        // A new level is a new address space: the mask layout, the ring's slot
        // size and any mask still in flight all describe the OLD one.
        self.vt_feedback = None;
    }

    // `vt_set_for_material` lived here and was **removed by the P26.4 audit**: it
    // was a second spelling of `crate::vt_set_for` — reaching past the door into
    // `VtTextures::set_for_material` — with ZERO callers, and its doc block (and
    // `set_vt_level`'s, above) asserted a projector usage that did not exist.
    // Clause 0's whole claim is that there is one expression, so a public one
    // beside it that nobody calls is a rule waiting to be written twice. A host
    // that wants the set asks `vt_set_for(renderer.vt_textures(), material)`.

    /// The live registry, for a host or a gate that wants to assert the WORLD
    /// (the resident set) rather than a report.
    #[inline]
    pub fn vt_textures(&self) -> Option<&crate::VtTextures> {
        self.vt_textures.as_ref()
    }

    /// The P26.4 pop-in instruments — per tile class, plus the ring's hit/miss
    /// counts. **Zero on a textureless scene**, which is what makes "the goldens
    /// did not change" a measurement of the command stream.
    #[inline]
    /// **Commit one camera pose at a fixed step** (P28.4) — the predictor's only
    /// input, and the seam the whole determinism argument rests on.
    ///
    /// `tick` is the host's **fixed-step** index. `false` — and nothing is
    /// stored — when it does not strictly advance, which is what a host wired to
    /// the render loop instead of the sim loop gets. There is no default and no
    /// fallback to a frame counter, deliberately: `self.frame_index` is right
    /// there and using it would make the speculation a function of how fast the
    /// machine draws, which is exactly the thing a learned predictor was
    /// rejected for being.
    ///
    /// A host that never calls this streams as it did before P28.4 — see
    /// [`PredictSettings`](crate::settings::PredictSettings).
    pub fn commit_camera(&mut self, tick: u64, view: &RenderView) -> bool {
        self.camera_history.commit(inf_math::CameraSample {
            tick,
            eye: view.eye_world,
            forward: view.forward.as_dvec3(),
            up: view.up.as_dvec3(),
        })
    }

    /// Forget the committed camera history — a level switch, a camera cut, an
    /// ejected PIE session. A secant across a cut is a teleport per tick.
    pub fn reset_camera_history(&mut self) {
        self.camera_history.clear();
    }

    /// The committed camera history, for a gate that wants to assert the
    /// predictor's own input rather than its output.
    #[inline]
    pub fn camera_history(&self) -> &inf_math::CameraHistory {
        &self.camera_history
    }

    /// **This frame's prediction**, or `None` when the history cannot support
    /// one (fewer than two committed poses) or the predictor is clamped off.
    ///
    /// A pure function of `(committed history, settings)` and of nothing the
    /// renderer holds — which is why it is safe for both consumers to ask for it
    /// separately and get one answer.
    pub fn prediction(&self) -> Option<inf_math::Prediction> {
        let p = self.settings.stream.predict;
        if !p.enabled {
            return None;
        }
        inf_math::dead_reckon(&self.camera_history, p.horizon_ticks)
    }

    pub fn vt_pop_in(&self) -> crate::VtPopIn {
        self.vt_pop_in
    }

    /// **The one line a host logs about the predictor** (P28.5) — and the
    /// reader `Prediction::turn`, `Prediction::clamped` and
    /// `CameraHistory::refused` were declared for and never got.
    ///
    /// The P28.4 ledger carried all three as "reported diagnostics on a public
    /// struct rather than dead counters — a host that wants to see the clamp
    /// bind can", and the P28.4 audit inherited them to this batch under the
    /// name they deserve: *a line nothing logs*. This is the line.
    ///
    /// `None` when no host has committed a camera, which is not the same as a
    /// line of zeros: an empty history is the predictor's real enable flag, so
    /// a host that prints nothing is saying "nothing here commits a pose" —
    /// which is the true and permanent state of the editor viewport — and one
    /// that prints zeros is saying "something does, and it is predicting
    /// nothing".
    ///
    /// The four numbers answer four different questions and no one of them
    /// implies another:
    ///
    /// * **`refused`** — pushes rejected for not advancing the tick. Non-zero
    ///   means this host is wired to the render loop rather than to the fixed
    ///   step, and it is the number that says so; without it such a host reads
    ///   as "the predictor does nothing".
    /// * **`turn`** — how far the prediction rotated the view this frame, in
    ///   degrees. Zero at the shipped horizon of 0, and that is the point: it
    ///   is what makes the lead-time ruling visible in a log.
    /// * **`clamped`** — whether [`inf_math::PREDICT_MAX_TURN`] bound it.
    /// * **`predict_wants` / `predict_fallback`** — the lane's own throughput,
    ///   from `VtPopIn`, so one line answers "is the lane running" as well as
    ///   "is the reckoner turning".
    pub fn predict_summary(&self) -> Option<String> {
        let h = &self.camera_history;
        if h.is_empty() && h.refused() == 0 {
            return None;
        }
        let p = self.prediction();
        Some(format!(
            "predict: {} committed poses ({} refused), horizon {} ticks — {}; \
             lane {} wants, {} at fallback",
            h.len(),
            h.refused(),
            self.settings.stream.predict.horizon_ticks,
            match p {
                None => "no prediction".to_string(),
                Some(p) => format!(
                    "turn {:.2} deg{}",
                    p.turn.to_degrees(),
                    if p.clamped { " (CLAMPED)" } else { "" }
                ),
            },
            self.vt_pop_in.predict_wants,
            self.vt_pop_in.predict_fallback,
        ))
    }

    /// **The one line a host logs about virtual texturing** (P26.5) — and the
    /// non-gate reader `VtStats::summary` and [`crate::VtPopIn::summary`] were
    /// both declared for.
    ///
    /// Two halves because they answer two questions and one number hides which:
    /// the residency's is *what is in the pool right now* (textures, slots,
    /// pinned roots, whether the budget clamped), the streamer's is *what this
    /// session has been through* (admits, deferrals, how many frames sat at
    /// fallback, how often the feedback ring landed). A pool that is full and a
    /// streamer that is thrashing look identical in either one alone.
    ///
    /// `None` on a textureless scene, which is not the same as a line of zeros:
    /// a host that prints nothing is saying "this level has no virtual
    /// textures", and one that prints zeros is saying "it has some and they are
    /// doing nothing".
    pub fn vt_summary(&self) -> Option<String> {
        let lib = self.vt_textures.as_ref()?;
        Some(format!(
            "{}; {}",
            lib.residency().stats().summary(),
            self.vt_pop_in.summary()
        ))
    }

    /// **The per-frame upload budget's sustained-demand advisory** (island wave
    /// I4, IB-16), or `None` while the throttle is only absorbing a burst.
    ///
    /// A host logs this the way it logs the registration advisories beside it: a
    /// burst that drains is the budget working, and a run of frames that never
    /// drains is content asking for more bandwidth than the budget grants, which
    /// is an author's decision rather than the renderer's.
    pub fn vt_upload_advisory(&self) -> Option<inf_vt::VtAdvisory> {
        self.vt_textures.as_ref()?.residency().upload_advisory()
    }

    /// **The streaming loop**, run once per frame at the sync point (P26.4).
    ///
    /// The five steps in `crate::vt_stream`'s module docs, in that order, and
    /// the order is load-bearing: the floor and the *previous* frame's feedback
    /// are applied to the residency BEFORE this frame's feedback pass is
    /// recorded, so the pass marks coverage against a table the frame is
    /// actually going to sample through.
    ///
    /// Returns immediately when there is no VT level, which is every textureless
    /// scene and all 50 committed goldens: no buffer is written, no pass is
    /// recorded, no counter moves.
    fn vt_stream(
        &mut self,
        gpu: &GpuContext,
        scene: &RenderScene,
        view: &RenderView,
        encoder: &mut wgpu::CommandEncoder,
        cluster: &[(u128, inf_vt::TileCoord)],
    ) {
        // **Before the residency is borrowed**, because the prediction is a pure
        // function of the committed history and the settings and must not be
        // able to read anything the transaction is about to change.
        let predicted = self.prediction();
        let (Some(lib), Some(pools)) = (self.vt_textures.as_mut(), self.vt.as_mut()) else {
            return;
        };
        let frame = self.frame_index;
        // The feedback pass is created lazily against the registry's layout, and
        // re-created when a registration grows it — a mask sized for four
        // textures cannot address a fifth, and the ring's slots are sized from
        // the same number.
        let layout = inf_vt::VtFeedbackLayout::for_residency(lib.residency());
        if self
            .vt_feedback
            .as_ref()
            .is_none_or(|f| *f.layout() != layout)
        {
            self.vt_feedback = Some(crate::vt_stream::VtFeedback::new(
                &gpu.device,
                layout,
                VT_FEEDBACK_REQUEST_CAP,
            ));
            // Recomputed WITH the layout, never beside it: the bases are a pure
            // function of the layout, and a stale one addresses another
            // texture's tiles — a mark that lands on the wrong bit is a page
            // streamed for a surface that never asked.
            let l = self.vt_feedback.as_ref().expect("just built").layout();
            self.vt_feedback_bases = (0..l.texture_count())
                .map(|t| {
                    l.texture_base(inf_vt::VtTextureHandle(t as u32))
                        .unwrap_or(0)
                })
                .collect();
        }
        let feedback = self.vt_feedback.as_mut().expect("just built");

        // 1–2. coverage and the analytic floor, plus (P28.2) the tiles every
        // resident cluster page's materials sample.
        //
        // The cluster wants ride at `VT_PRIORITY_FLOOR`, in the SAME want set as
        // the analytic floor, and that is the whole mechanism by which the
        // invariant survives between transactions rather than only at one:
        // `apply_wants` protects every want that is already resident before any
        // miss is offered a slot, so a tile belonging to a resident cluster page
        // cannot be evicted while the page is still resident. Put them in a
        // second transaction and that stops being true on the very next frame.
        let coverage = crate::vt_stream::scene_coverage(scene, &view.origin);
        let mut wants = crate::vt_stream::analytic_floor(lib, view, &coverage);
        for (guid, tile) in cluster {
            if let Some(h) = lib.handle(*guid) {
                wants.push(inf_vt::VtWant::new(h, *tile));
            }
        }
        let floor_len = wants.len();

        // 2b. **The speculative lane** (P28.4): the same footprint rule, asked
        // at the camera the committed history says is coming, at
        // `VT_PRIORITY_PREDICT`.
        //
        // Appended to the SAME want set, and that is load-bearing twice over.
        // It is what makes the strictly-lower rank mean anything — one
        // `apply_wants` walks the lanes in order, so a speculative miss is only
        // ever offered slots no floor or feedback want has claimed, and a
        // resident speculative tile is the first thing a floor miss takes. And
        // it is what keeps the invariant a *world* property rather than a
        // policy: there is no second transaction in which residency could dip
        // below `floor ∪ feedback`, because there is no second transaction.
        //
        // `prediction()` is `None` on every host that does not commit a camera,
        // which is every host that existed before this batch.
        if let Some(p) = predicted {
            wants.extend(crate::vt_stream::speculative_wants(
                lib.residency(),
                view,
                &coverage,
                &p,
            ));
        }

        // 3. frame F − 2's mask, or nothing. Never "whatever arrived".
        let landed = match feedback.take_wants(&gpu.device, lib, frame) {
            Some(refine) => {
                self.vt_pop_in.feedback_frames += 1;
                wants.extend(refine);
                true
            }
            None => {
                self.vt_pop_in.feedback_misses += 1;
                false
            }
        };
        debug_assert!(
            wants.len() >= floor_len,
            "the feedback removed a floor want"
        );

        // 4. apply, and upload the pages.
        let (txn, report) = lib.sync(&gpu.device, &gpu.queue, pools, &wants);
        self.vt_pop_in.deferred += u64::from(txn.deferred);
        self.vt_pop_in.admits += txn.admits.len() as u64;
        // P26.5's routed remainder: the apply report reached a caller per call
        // and was summed nowhere, so a re-upload regression had no counter.
        // `admits` is what residency decided, this is what the queue paid for,
        // and the pair diverges exactly when the mirror writes a page residency
        // did not newly admit.
        self.vt_pop_in.page_uploads += report.pages as u64;
        self.vt_pop_in.page_upload_bytes += report.page_bytes;
        self.vt_pop_in.frames += 1;
        crate::vt_stream::count_fallbacks(lib, &wants, &mut self.vt_pop_in);

        // 5. record the next feedback pass. The table it reads is the one the
        // transaction above just wrote, which is why this is last.
        //
        // With the visibility path on, the meshlet surfaces are handed to the
        // per-FRAGMENT producer inside the graph instead (P28.1) — see
        // `feedback_requests`' `skip_vgeom`.
        let skip_vgeom = self.settings.vgeom.enabled && self.settings.vgeom.visbuffer;
        let requests =
            crate::vt_stream::feedback_requests(lib, feedback.layout(), &coverage, skip_vgeom);
        self.vt_feedback_dispatched = feedback.record(
            &gpu.device,
            &gpu.queue,
            encoder,
            pools.table(),
            pools.table_generation(),
            view,
            &requests,
        );
        // …and the read is recorded in the ONE ledger (P28.3). The ring told us
        // *whether* the bytes arrived; the ledger pins *which frame* they were
        // allowed to be, so two consumers reading at this frame are comparable
        // — which is the property the two independent rings cannot state about
        // each other and `RingLedger::readers_agree` asserts.
        self.stream_ring
            .read(inf_stream::Consumer::Texture, frame, |_| landed);
    }

    /// The live VT mirror, for a host that needs to apply a transaction to it.
    #[inline]
    pub fn vt_pools_mut(&mut self) -> Option<&mut crate::vt::VtPools> {
        self.vt.as_mut()
    }

    /// The live VT mirror.
    #[inline]
    pub fn vt_pools(&self) -> Option<&crate::vt::VtPools> {
        self.vt.as_ref()
    }

    /// Frames drawn with a virtual-texture pool bound (P26.3 engagement
    /// counter). **Zero is the assertion on a textureless scene**, not an
    /// absence of evidence.
    #[inline]
    pub fn vt_engaged_frames(&self) -> u64 {
        self.vt_engaged_frames
    }

    /// How many deformation-window **texture uploads** this renderer has issued
    /// since it was created (P22.1) — the engagement counter.
    ///
    /// The claim it exists to make is one a pixel comparison cannot: *a frame in
    /// which nothing walked and the camera did not move costs nothing at all.*
    /// Two frames of a still scene are identical whether the window re-uploaded a
    /// megabyte or not, so the only honest evidence is the command stream. Zero
    /// forever on a scene with no deformation field.
    pub fn deform_uploads(&self) -> u64 {
        self.deform.uploads()
    }

    /// How many frames the P20.3 underwater pass has **engaged** on since this
    /// renderer was built — the off-path instrument.
    ///
    /// Unchanged across a frame ⇒ the node returned before touching the encoder,
    /// which is a claim about the command stream that no pixel comparison can
    /// make (a pass that engaged and wrote the scene back unchanged looks
    /// identical from outside). Read by
    /// `underwater_off_path_never_engages`.
    pub fn underwater_engaged_frames(&self) -> u64 {
        self.underwater.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// How many frames the P21.1 voxel pass has **engaged** on since this renderer
    /// was built — the off-path instrument for volumetric terrain.
    ///
    /// Unchanged across a frame ⇒ the node returned before touching the encoder.
    /// That is the byte-stability guarantee for every golden that predates
    /// volumetric terrain, and it is a claim about the *command stream* that a
    /// pixel comparison cannot make: rendering a volume-free scene twice is
    /// identical whether the node early-returned or opened a pass and drew
    /// nothing. Read by `voxel_off_path_never_engages`.
    pub fn voxel_engaged_frames(&self) -> u64 {
        self.voxel.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `(cached tile textures, cached splat-material slots)` held by the terrain
    /// pass right now (Hardening D).
    ///
    /// The instrument for a claim a pixel comparison structurally cannot make:
    /// "the cache was RELEASED". A stranded cache draws exactly the same frames a
    /// released one does — that is what let the terrain node hold four textures
    /// per tile (≈840 KiB at 256²) for the renderer's whole life after the last
    /// tile left the scene. `(0, 0)` when the graph carries no terrain node.
    pub fn terrain_cache_counts(&self) -> (usize, usize) {
        self.graph
            .node::<passes::terrain::TerrainNode>()
            .map(|n| n.cached_counts())
            .unwrap_or((0, 0))
    }

    /// `(joint-box meshes, meshlet-sphere assets)` the GI node is caching right
    /// now (Hardening D) — the same kind of instrument as
    /// [`terrain_cache_counts`](Self::terrain_cache_counts), for a cache whose
    /// entries hold a whole skinned mesh's vertices.
    pub fn gi_cache_counts(&self) -> (usize, usize) {
        self.graph
            .node::<passes::gi::GiNode>()
            .map(|n| n.cached_counts())
            .unwrap_or((0, 0))
    }

    /// How many sprite textures the 2D pass is caching (Hardening D). Includes
    /// the built-in bitmap font, which is always resident.
    pub fn sprite_cache_len(&self) -> usize {
        self.graph
            .node::<passes::sprite::SpriteNode>()
            .map(|n| n.cached_textures())
            .unwrap_or(0)
    }

    /// The active HDR/post settings.
    pub fn settings(&self) -> &RenderSettings {
        &self.settings
    }

    /// Replace the HDR/post settings (bloom, SSAO, TAA, exposure, dither). The
    /// viewport + player expose this to their UI; goldens set it per scene.
    ///
    /// **An illegal virtual-shadow configuration is refused, not clamped** (P27.2):
    /// the settings are left exactly as they were and the refusal is logged. See
    /// [`try_set_settings`](Self::set_settings) — this is the same call with the
    /// error dropped, kept infallible because every existing caller passes settings
    /// derived from [`RenderSettings::default`] and a `Result` at those call sites
    /// would be noise. A host that wants the reason takes the typed one.
    pub fn set_settings(&mut self, settings: RenderSettings) {
        if let Err(e) = self.try_set_settings(settings) {
            tracing::warn!(
                "inf-render: virtual shadow settings refused ({e}); the previous \
                 settings are unchanged"
            );
        }
    }

    /// [`set_settings`](Self::set_settings) with the refusal returned.
    ///
    /// **The settings boundary the P27.1 audit named.** A per-light truncation is a
    /// recovery, and a recovery that is the *only* door is how a wrapped `65 536²`
    /// page grid reached the allocator: it validated, it registered, and it walked
    /// four billion addresses into a table with no entries. The saturating counters
    /// in `inf-vsm` closed that from inside; this closes it from outside, and the
    /// two are deliberately redundant.
    ///
    /// **Nothing is applied on `Err`** — not the legal half, not the illegal half.
    /// A partially-applied settings block is a configuration no caller asked for,
    /// and it is exactly the silent cap this refusal exists to replace.
    pub fn try_set_settings(
        &mut self,
        settings: RenderSettings,
    ) -> Result<(), crate::settings::VsmSettingsError> {
        settings.vsm.validate()?;
        // Toggling TAA invalidates the accumulated history.
        if settings.taa != self.settings.taa {
            self.prev_view_proj = None;
        }
        self.settings = settings;
        Ok(())
    }

    /// Turn the P28.1 visibility-buffer readback on. Off by default: it records a
    /// full-viewport `copy_texture_to_buffer` every frame the visibility path
    /// runs, which no shipping frame should pay for.
    pub fn set_visbuffer_readback(&mut self, enabled: bool) {
        self.vis_readback.borrow_mut().set_enabled(enabled);
    }

    /// The ids the **last submitted** frame wrote, row padding removed. `None`
    /// unless [`set_visbuffer_readback`](Self::set_visbuffer_readback) is on and
    /// a frame has rasterized one. Blocking.
    pub fn read_visbuffer(&self, gpu: &GpuContext) -> Option<passes::visbuffer::VisImage> {
        self.vis_readback.borrow().read(gpu)
    }

    /// **Requests the PER-SURFACE feedback pass dispatched on the last frame**
    /// (P28.1).
    ///
    /// Zero is the observable form of the handover: with the visibility path on,
    /// meshlet surfaces are marked per FRAGMENT and `feedback_requests` skips
    /// them, so a scene whose only textured surfaces are meshlets dispatches
    /// nothing here. It is exposed because that number is the only place the
    /// handover is visible from outside — residency is `floor ∪ feedback` and the
    /// floor dominates, so two producers and one producer settle on sets that
    /// differ by ten tiles and by no structural property a test can name.
    pub fn vt_feedback_requests(&self) -> u32 {
        self.vt_feedback_dispatched
    }

    /// The P28.1 visibility path's audit counters — how many frames took it, and
    /// which ceiling refused the ones that did not.
    pub fn vis_audit(&self) -> passes::visbuffer::VisAudit {
        self.vis_report.lock().map(|a| *a).unwrap_or_default()
    }

    /// Enable/disable the P18.1 vgeom occlusion-audit counters. **Off by
    /// default**: the cull compute skips the atomics entirely and no readback copy
    /// is recorded, so the shipping frame is untouched. Turning it on costs four
    /// atomics per surviving (instance, meshlet) pair plus a 16-byte copy — a
    /// test/tools instrument, not a shipping counter.
    pub fn set_vgeom_audit(&mut self, enabled: bool) {
        self.vgeom_audit.enabled = enabled;
    }

    /// Read the audit counters recorded by the **last submitted** frame. Blocks on
    /// a buffer map; returns zeros if the audit was never enabled or the vgeom node
    /// did not run (vgeom off, or a scene with no meshlet content).
    pub fn vgeom_audit(&self, gpu: &GpuContext) -> VgeomAudit {
        if !self.vgeom_audit.enabled {
            return VgeomAudit::default();
        }
        self.vgeom_audit.read(gpu)
    }

    /// Enable/disable the **per-pass GPU clock** (island wave I4). **Off by
    /// default**, on the same terms as the two audits either side of it: with it
    /// off the frame records byte-identical commands, which is what lets an
    /// instrument share a renderer with 54 byte-frozen goldens.
    ///
    /// Returns whether timing is live afterwards — `false` when asked for on a
    /// device without `TIMESTAMP_QUERY`(`_INSIDE_ENCODERS`), which is every
    /// software rasterizer and most paravirtual runners. That is a **report, not
    /// an error**: the caller falls back to the CPU frame clock, and a harness
    /// that silently believed it had per-pass numbers would print zeros as
    /// measurements.
    pub fn set_gpu_timing(&mut self, gpu: &GpuContext, enabled: bool) -> bool {
        self.frame_timer = match enabled {
            true => crate::timing::FrameTimer::new(gpu),
            false => None,
        };
        self.frame_timer.is_some()
    }

    /// The **last submitted** frame's per-pass GPU timings, blocking on a buffer
    /// map — the sibling of [`vgeom_audit`](EngineRenderer::vgeom_audit) and read
    /// the same way.
    ///
    /// `None` when timing was never enabled, when this device cannot time an
    /// encoder segment, or when the frame has already been taken. A caller reads
    /// it once per frame, after the poll that made the frame finish.
    pub fn gpu_timings(&mut self, gpu: &GpuContext) -> Option<crate::timing::FrameTimings> {
        self.frame_timer.as_mut()?.take(gpu)
    }

    /// The **last recorded** frame's record-path phase breakdown (island wave
    /// I7b) — CPU only, so unlike [`gpu_timings`](Self::gpu_timings) it needs no
    /// poll and blocks on nothing.
    ///
    /// All zeros until [`set_gpu_timing`](Self::set_gpu_timing) is on, which is
    /// every shipped frame.
    pub fn record_profile(&self) -> crate::timing::RecordProfile {
        self.record_profile
    }

    /// Every graph node's name in run order (I4) — the list a per-pass report is
    /// checked against, so a pass that is renamed or dropped is a red arm rather
    /// than a line quietly missing from a diagnostic.
    pub fn pass_names(&self) -> Vec<&'static str> {
        self.graph.names()
    }

    /// Enable/disable the P18.5 scatter instance-cull audit counters. **Off by
    /// default**, on the same terms as the vgeom audit: the cull compute skips the
    /// atomics and no readback copy is recorded, so the shipping frame is
    /// untouched. It exists so a gate can prove the per-instance culling is real
    /// rather than a no-op that trivially satisfies a pixel comparison.
    pub fn set_scatter_audit(&mut self, enabled: bool) {
        self.scatter_audit.enabled = enabled;
    }

    /// Read the scatter cull counters recorded by the **last submitted** frame.
    /// Blocks on a buffer map; returns zeros if the audit was never enabled or the
    /// scatter node did not run (no scatter content, or the CPU fallback path).
    pub fn scatter_audit(&self, gpu: &GpuContext) -> passes::scatter::ScatterAudit {
        if !self.scatter_audit.enabled {
            // `shadow_casters` is a CPU counter the shadow node stores
            // unconditionally (one relaxed store), so it is reported even with the
            // GPU audit off — otherwise the one figure that says whether the tier
            // clamps reached the caster pack would only exist in the mode nobody
            // ships in.
            return passes::scatter::ScatterAudit {
                shadow_casters: self.scatter_audit.shadow_casters(),
                uploads: self.scatter_audit.uploads(),
                ..Default::default()
            };
        }
        self.scatter_audit.read(gpu)
    }

    /// What the P18.2 meshlet streamer did on the **last rendered frame**:
    /// residency, backlog, budget clamping, and the per-asset residency floor.
    ///
    /// Free and always on — these are CPU counters the streamer already maintains,
    /// unlike the GPU occlusion audit, which has to be enabled. Zeroed until the
    /// vgeom node has run (vgeom off, or a scene with no meshlet content).
    pub fn vgeom_stream_report(&self) -> passes::vgeom::VgeomStreamReport {
        self.vgeom_stream
            .lock()
            .map(|r| r.clone())
            .unwrap_or_default()
    }

    /// **The unified streamer's audit** (P28.3) — what the three page systems
    /// hold between them, what the arbiter would grant them from the live
    /// floors, how much of it is coupled, and whether the readback ring's two
    /// consumers agree.
    ///
    /// Free and always on: every number is a CPU counter the three consumers
    /// already maintain, folded into one struct because the whole point of the
    /// phase is that "how much streaming residency does this machine hold" was
    /// a question with three answers and no sum.
    ///
    /// **This is where `StreamError::FloorExceedsBudget` has its producer.**
    /// `RenderSettings::arbitrate_budgets` runs with zero floors, because a
    /// settings struct knows no content; here the floors are the live ones —
    /// the virtual texture's pinned roots, in bytes, and every resident asset's
    /// page 0 — so a unified budget that cannot hold them is a refusal with two
    /// real numbers in it rather than a constant nobody can reach.
    pub fn stream_report(&self) -> crate::stream::StreamReport {
        let vgeom = self.vgeom_stream_report();
        let vt = self.vt_textures.as_ref().map(|l| l.residency());
        let vsm = self.vsm.as_ref();

        let texture_floor = vt.map_or(0, |r| u64::from(r.stats().roots) * r.page_bytes());
        let resident = [
            vgeom.stats.resident_bytes,
            vt.map_or(0, |r| r.resident_bytes()),
            vsm.map_or(0, |v| v.residency().resident_bytes()),
        ];
        let requests = [
            inf_stream::BudgetRequest {
                floor_bytes: vgeom.floor_bytes,
                want_bytes: self.settings.vgeom.stream.budget_bytes,
            },
            inf_stream::BudgetRequest {
                floor_bytes: texture_floor,
                want_bytes: self.settings.vt.budget_bytes,
            },
            inf_stream::BudgetRequest::want(if self.settings.vsm.enabled {
                self.settings.vsm.budget_bytes
            } else {
                0
            }),
        ];
        crate::stream::StreamReport {
            budget_bytes: self.settings.stream.budget_bytes,
            grant: inf_stream::arbitrate(self.settings.stream.budget_bytes, &requests),
            resident,
            floors: [requests[0].floor_bytes, texture_floor, 0],
            coupled_groups: vgeom.coupled_groups,
            coupled_tiles: vgeom.coupled_tiles,
            dropped_groups: vgeom.dropped_groups,
            mismatched_textures: vgeom.mismatched_textures,
            retracted: vgeom.retracted,
            stale_tiles: vgeom.stale_tiles,
            ring: self.stream_ring,
        }
    }

    /// **The one line a host logs about streaming** (P28.3) — the three page
    /// systems in one place, which is what a unified streamer is for.
    ///
    /// `None` when nothing streams at all, which is not the same as a line of
    /// zeros: a host that prints nothing is saying "this level pages nothing",
    /// one that prints zeros is saying "it pages something and that something is
    /// doing nothing".
    pub fn stream_summary(&self) -> Option<String> {
        let r = self.stream_report();
        if r.resident.iter().all(|b| *b == 0) && r.coupled_groups == 0 {
            return None;
        }
        Some(r.summary())
    }

    /// What the P18.4 GI voxelizer consumed on the **last rendered frame**:
    /// candidate primitives, how many fitted the per-frame budget, how many were
    /// dropped, the macro-cell bin size, terrain columns and probes updated.
    ///
    /// Free and always on — CPU counters the pass already computes, like the P18.2
    /// streaming report and unlike the GPU occlusion audit. Zeroed while GI is off.
    /// `dropped > 0` is the signal that replaced `MAX_GI_INSTANCES`' silence.
    pub fn gi_audit(&self) -> crate::gi::GiAudit {
        self.gi_audit.lock().map(|a| *a).unwrap_or_default()
    }

    /// The shared GI resources (P18.4). Exposed for the gates that compare the GI
    /// **inputs** rather than the pixels — the residency-independence gate reads
    /// the voxel volume and the probe buffer back and byte-compares two residency
    /// states that legitimately draw different terrain detail.
    pub fn gi_resources(&self) -> &GiResources {
        &self.gi
    }

    /// The shared atmosphere resources (P17.2). Exposed for the LUT-determinism
    /// gate, which reads the baked textures back and byte-compares two bakes.
    pub fn atmosphere(&self) -> &crate::passes::sky_lut::AtmosphereResources {
        &self.atmosphere
    }

    /// The active shading view mode (R-P2).
    pub fn view_mode(&self) -> ViewMode {
        self.view_mode
    }

    /// Set the shading view mode (Lit / Unlit / Wireframe / Biomes /
    /// VtResidency). A `Wireframe` request on an adapter without
    /// `POLYGON_MODE_LINE` is clamped to `Unlit` (logged once via `tracing`) —
    /// wireframe is a hard GPU-feature requirement, so we degrade gracefully
    /// rather than fail. `Biomes` is **never** clamped: it is a uniform flag plus
    /// one texture, so every adapter that can draw terrain at all can draw it.
    /// Neither is `VtResidency` (P26.5) — it is a uniform flag over bindings the
    /// lit passes already carry, so an adapter that can sample a virtual texture
    /// can paint the heat-map, and one that has no virtual textures to paint
    /// gets the grey that says so. Nor is `VsmPages` (P27.5), for the identical
    /// reason one virtual system over. The editor viewport + goldens set this;
    /// it is never persisted and the player never touches it.
    pub fn set_view_mode(&mut self, mode: ViewMode) {
        let effective = if mode == ViewMode::Wireframe && !self.polygon_mode_line {
            if !self.wireframe_warned {
                tracing::warn!(
                    "inf-render: wireframe view mode unavailable (adapter lacks \
                     POLYGON_MODE_LINE) — falling back to Unlit"
                );
                self.wireframe_warned = true;
            }
            ViewMode::Unlit
        } else {
            mode
        };
        self.view_mode = effective;
    }

    /// **The virtual-shadow-map sync point** (P27.1), run at the frame's sync
    /// point beside the virtual-texture one.
    ///
    /// Returns immediately — building nothing, allocating nothing, counting
    /// nothing — unless the setting is on AND the scene carries a shadow-casting
    /// light. That is the off-path discipline every pass in this tree keeps, and
    /// it is what makes "no golden moved" a measurement
    /// ([`vsm_engaged_frames`](Self::vsm_engaged_frames)) rather than a hope.
    fn vsm_sync(&mut self, gpu: &GpuContext, scene: &RenderScene, view: &RenderView) {
        if !self.settings.vsm.enabled {
            self.vsm = None;
            self.vsm_light_slots.clear();
            self.vsm_receiver_params = crate::vsm_receiver::VsmReceiverParams::absent();
            return;
        }
        if self
            .vsm
            .as_ref()
            .is_none_or(|v| !v.matches(scene, &self.settings.vsm))
        {
            // A new light set is a new address space: the mask layout, the ring's
            // slot size and any mask still in flight all describe the old one.
            self.vsm = crate::vsm_mark::VsmSystem::for_scene(gpu, scene, &self.settings.vsm);
        }
        let frame = self.frame_index;
        // Read before the system is borrowed, and it is the SAME prediction the
        // texture lane got this frame: one committed history, one dead
        // reckoning, two consumers — so the two speculative sets cannot describe
        // two different cameras.
        let predicted = self.prediction();
        let Some(v) = self.vsm.as_mut() else {
            self.vsm_light_slots.clear();
            self.vsm_receiver_params = crate::vsm_receiver::VsmReceiverParams::absent();
            return;
        };
        let marks_before = v.stats().mark_frames;
        v.sync(gpu, scene, view, &self.settings.vsm, frame, predicted);
        let landed = v.stats().mark_frames > marks_before;
        self.stream_ring
            .read(inf_stream::Consumer::Shadow, frame, |_| landed);
        let Some(v) = self.vsm.as_mut() else {
            return;
        };
        // **THE RECEIVER'S SIDE OF THE SYNC POINT** (P27.4), here rather than
        // beside the marking pass, and the placement is the whole of why the lit
        // passes see this frame: `write_buffer` is staged on the queue and runs
        // before the commands of any command buffer submitted afterwards, and the
        // graph — which is where a lit fragment samples — is recorded after this.
        // `VsmSystem::sync` uploads the projection list for the same reason.
        self.vsm_light_slots = v.receiver_slots(scene);
        self.vsm_receiver_params = crate::vsm_receiver::VsmReceiverParams::new(
            &self.settings.vsm,
            crate::vt_stream::projection_scale(view),
            crate::vsm_receiver::sun_slot(
                &self.vsm_light_slots,
                scene
                    .lights
                    .iter()
                    .map(|l| l.kind == crate::scene::LightKind::Directional),
            ),
            v.projections().len() as u32,
        );
        self.vsm_receiver
            .write(&gpu.queue, &self.vsm_receiver_params);
    }

    /// The live virtual-shadow-map system, for a host or a gate that wants to
    /// assert the WORLD (the resident page set) rather than a report.
    #[inline]
    pub fn vsm(&self) -> Option<&crate::vsm_mark::VsmSystem> {
        self.vsm.as_ref()
    }

    /// The same system, mutably — **a gate's door** (P27.3), on
    /// `VsmRaster::read_draw_counts`'s precedent.
    ///
    /// The only `&mut` operation it exposes is
    /// [`VsmSystem::flush_page_cache`](crate::vsm_mark::VsmSystem::flush_page_cache),
    /// and nothing in the shipping path calls it: a page invalidation is meant to
    /// be a consequence of a content stamp moving, and throwing the cache away is
    /// the blunt instrument that lets an arm *force* the fresh raster it compares
    /// the cached texels against.
    #[inline]
    pub fn vsm_mut(&mut self) -> Option<&mut crate::vsm_mark::VsmSystem> {
        self.vsm.as_mut()
    }

    /// Frames that recorded the shadow-page marking pass — the P27.1 engagement
    /// counter. **Zero is the assertion** on a scene that does not run VSM, not
    /// an absence of evidence.
    #[inline]
    pub fn vsm_engaged_frames(&self) -> u64 {
        self.vsm_engaged_frames
    }

    /// Frames that rasterized at least one shadow page — the P27.2 engagement
    /// counter. **Zero is the assertion** on a scene that does not run virtual
    /// shadows, and it is a different claim from `vsm_engaged_frames`: marking a
    /// page and filling it are two passes with two failure modes.
    #[inline]
    pub fn vsm_raster_frames(&self) -> u64 {
        self.vsm_raster_frames
    }

    /// Frames drawn with the shadow atlas **bound to the lit passes** — the
    /// P27.4 engagement counter, and the one that says the setting stopped being
    /// inert. **Zero is the assertion** on a scene that does not run virtual
    /// shadows; a non-zero value says a lit fragment was in a position to read a
    /// page, which neither of the other two counters claims.
    #[inline]
    pub fn vsm_receiver_frames(&self) -> u64 {
        self.vsm_receiver_frames
    }

    /// The receiver params this frame's lit passes were handed (P27.4) — a
    /// gate's door, so an arm can assert the sun slot and the blend band the
    /// shader actually read rather than the ones it expected the renderer to
    /// pack. `None` when no system is live.
    pub fn vsm_receiver_params(&self) -> Option<crate::vsm_receiver::VsmReceiverParams> {
        self.vsm.as_ref().map(|_| self.vsm_receiver_params)
    }

    /// Per **scene light index**, the projection slot + 1 the lights uniform
    /// carried this frame (P27.4). Empty when no system is live.
    pub fn vsm_light_slots(&self) -> &[u32] {
        &self.vsm_light_slots
    }

    /// The caster pass's counters (P27.2). `None` when no system is built.
    pub fn vsm_raster_stats(&self) -> Option<crate::vsm_raster::VsmRasterStats> {
        self.vsm
            .as_ref()
            .map(crate::vsm_mark::VsmSystem::raster_stats)
    }

    /// The one line a host logs about virtual shadow maps. `None` when the
    /// system is not built, which is not the same as a line of zeros.
    pub fn vsm_summary(&self) -> Option<String> {
        self.vsm.as_ref().map(crate::vsm_mark::VsmSystem::summary)
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
        // Frame-budget profiling span (P15.1). Exists unconditionally via
        // `tracing`; the Tracy layer (behind the apps' `tracy` feature) records
        // it, and the per-pass spans in `RenderGraph::run` nest inside it.
        let _frame_span = tracing::info_span!("render_frame", frame = self.frame_index).entered();
        // **THE RECORD PATH'S OWN CLOCK** (island wave I7b). Armed with the GPU
        // timer and off in every shipped frame; see `crate::timing::RecordClock`
        // for why the record stage needed phases of its own when the per-pass
        // record column already existed.
        let mut rec = crate::timing::RecordClock::start(self.frame_timer.is_some());
        let scene_size = (view.width.max(1), view.height.max(1));
        let resized = self.targets.as_ref().is_none_or(|t| t.size != scene_size);
        if resized {
            self.targets = Some(FrameTargets::create(gpu, scene_size, self.next_generation));
            self.next_generation += 1;
            self.prev_view_proj = None; // history textures were reallocated
        }

        // The atmosphere LUTs are sized by quality, which a host may clamp down
        // at any time (tier detection, a mobile preset, a settings change).
        // Recreating them bumps `generation`, which is what makes the env
        // bind-group cache drop its now-dangling views — see
        // `passes::EnvBinding::bind_group`.
        if self.atmosphere.quality != self.settings.atmosphere.quality {
            self.atmosphere = AtmosphereResources::new(
                gpu,
                self.settings.atmosphere.quality,
                self.next_atmosphere_generation,
            );
            self.next_atmosphere_generation += 1;
        }

        // The GI voxel/SH buffers are sized by GI quality, which a host may clamp
        // down at any time (P18.4). Recreating them bumps `generation`, which is
        // the third component of `passes::ResourceKey` — without it, the lit passes
        // would keep a bind group pointing at the previous tier's buffers and
        // sample a probe grid whose dimensions no longer match the uniform. That
        // reads as *wrong pixels*, not a validation error, which is why the key
        // exists.
        if self.gi.quality != self.settings.gi.quality {
            self.gi = GiResources::new(gpu, self.settings.gi.quality, self.next_gi_generation);
            self.next_gi_generation += 1;
        }

        rec.mark(crate::timing::record::TARGETS);

        // Camera sub-pixel jitter (TAA only), applied to the projection.
        let base_vp = view.view_proj();
        let jitter = if self.settings.taa {
            halton_jitter(self.frame_index)
        } else {
            [0.0, 0.0]
        };
        let jvp = if self.settings.taa {
            let ox = 2.0 * jitter[0] / scene_size.0 as f32;
            let oy = 2.0 * jitter[1] / scene_size.1 as f32;
            Mat4::from_translation(Vec3::new(ox, oy, 0.0)) * base_vp
        } else {
            base_vp
        };

        let mut uniforms = ViewUniforms::from_view(view, &scene.sun);
        // R-P2 view mode: the unlit flag drives the lit passes' albedo+emissive
        // short-circuit (Unlit AND Wireframe both shade unlit). Lit writes 0, so
        // every pre-R-P2 golden stays byte-identical.
        uniforms.flags[0] = self.view_mode.unlit_flag();
        // P19.2 Biomes: the terrain pass tints by biome id. Every other mode
        // writes 0 here, so the branch is dead in the frames the goldens capture.
        uniforms.flags[1] = self.view_mode.biomes_flag();
        // P26.5 VtResidency: the lit mesh shaders paint `vt_heat` instead of
        // shading. Every other mode writes 0 here, so the branch is dead in the
        // frames the goldens capture.
        uniforms.flags[2] = self.view_mode.vt_residency_flag();
        // P27.5 VsmPages: the shadow-page residency ramp. `flags[3]` was
        // reserved, so no uniform grew and every pass that declares the
        // shorter `View` struct is unaffected.
        uniforms.flags[3] = self.view_mode.vsm_pages_flag();
        if self.settings.taa {
            uniforms.view_proj = jvp.to_cols_array();
            uniforms.inv_view_proj = jvp.inverse().to_cols_array();
        }
        gpu.queue
            .write_buffer(&self.view_buf, 0, bytemuck::bytes_of(&uniforms));

        // P20.3 shoreline wetness. Derived HERE, from the water bodies both
        // projectors already publish, rather than added to `RenderScene` — which
        // is the strongest form the mirror rule takes: two hosts cannot disagree
        // about a derivation neither of them performs. Pure in the scene + the
        // render origin; no camera reaches it (`crate::wetness`).
        gpu.queue.write_buffer(
            &self.wetness.uniform,
            0,
            bytemuck::bytes_of(&pack_wetness(&scene.waters, &view.origin)),
        );

        // P22.1 surface deformation. Same doctrine one phase on: the WINDOW is
        // derived here, from the sparse field both projectors publish, so no host
        // ever computes it and the two cannot disagree about which sample a texel
        // carries. This is also the one place a camera legally touches the
        // deformation path — it selects what is DRAWN out of sim-authoritative
        // state and never what is simulated (the P21 camera-paged-store law).
        //
        // The wind phase is `ResolvedSky::cloud_time_s` off the cloud block both
        // projectors already publish: the level's own clock, never a wall clock
        // and never a frame index, so a golden renders the same frame twice and a
        // replay reproduces the same sway.
        let deform_uniform = self.deform.update(
            gpu,
            scene.deform.as_deref(),
            &view.origin,
            (view.eye_world.x, view.eye_world.z),
            (
                scene.atmosphere.clouds.time_s,
                scene.atmosphere.clouds.wind_x,
                scene.atmosphere.clouds.wind_z,
            ),
            self.settings.scatter.foliage_wind,
        );
        gpu.queue
            .write_buffer(&self.deform.uniform, 0, bytemuck::bytes_of(&deform_uniform));

        rec.mark(crate::timing::record::VIEW_UNIFORMS);

        let history_valid = self.settings.taa && !resized && self.prev_view_proj.is_some();
        // The cloud's own history validity. Same three conditions, its own knob:
        // a resize reallocated the ping-pong, and the first frame has no previous
        // view-projection to reproject through.
        let cloud_history_valid =
            self.settings.cloud_temporal && !resized && self.prev_view_proj.is_some();
        let prev_vp = self.prev_view_proj.unwrap_or_else(|| jvp.to_cols_array());
        let cur = (self.frame_index & 1) as usize;
        let prev = 1 - cur;

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // The island-wave-I4 GPU stopwatch opens here, at the frame's first
        // command, so every segment below is measured against one origin. `None`
        // on every shipped frame — see `crate::timing`.
        if let Some(t) = self.frame_timer.as_mut() {
            t.begin(&mut encoder);
        }

        rec.mark(crate::timing::record::ENCODER);

        // **THE CLUSTER PAGE-IN** (P28.2), in three steps around one virtual-
        // texture transaction, because a page-in that feeds two consumers cannot
        // be seated at two different depths of the frame.
        //
        //   1. stage the geometry half (the meshlet streamer's plan);
        //   2. run ONE virtual-texture transaction whose want set contains the
        //      analytic floor, the previous frame's feedback AND the tiles every
        //      resident cluster page samples;
        //   3. pair — or retract the pages whose tiles the transaction could not
        //      seat, releasing their pool blocks before anything can write them.
        //
        // Disjoint field borrows: the node lives in `self.graph` and the texture
        // registry in `self.vt_textures`, so `&mut self.graph` and
        // `&self.vt_textures` coexist — which is why steps 1 and 3 can ask the
        // registry a question while holding the node.
        let vsettings = self.settings.vgeom;
        let cluster_wants = {
            let node = self.graph.node_mut::<passes::vgeom::VgeomNode>();
            match node {
                Some(n) => {
                    n.plan_cluster_pages(scene, view, &vsettings);
                    rec.mark(crate::timing::record::CLUSTER_PLAN);
                    let lib = self.vt_textures.as_ref();
                    // Registered AND addressable. `handle` alone was the P28.2
                    // filter and it let a **stale** address through — a pairing
                    // baked against another image of the same GUID, which
                    // `is_resident` reports exactly as an unseated tile and which
                    // no transaction can ever seat, so the page was retracted on
                    // every frame for ever (measured: residency reached zero and
                    // the mesh vanished). `can_address` separates the two.
                    n.cluster_tile_wants(
                        scene,
                        |g, tile| {
                            lib.is_some_and(|l| {
                                l.handle(g)
                                    .is_some_and(|h| l.residency().can_address(h, tile))
                            })
                        },
                        // P28.3's grid claim: the cooked digest against the
                        // REGISTERED image's own. A texture this level does not
                        // bind is not a mismatch — it is simply not part of the
                        // pairing, which the `addressable` filter already says.
                        |g, t| {
                            lib.is_none_or(|l| {
                                l.handle(g).is_none_or(|h| {
                                    l.residency().desc(h).is_none_or(|d| t.grid_matches(d))
                                })
                            })
                        },
                    )
                }
                None => Vec::new(),
            }
        };

        rec.mark(crate::timing::record::CLUSTER_WANTS);

        // **THE VIRTUAL-TEXTURE SYNC POINT** (P26.4). Before any pass samples
        // the atlas, and inside the frame's own encoder, so the ordering
        // contract in `crate::vt`'s module docs holds unchanged: a transaction
        // is applied whole, staged on the queue, and executes before the
        // commands of this encoder. A textureless frame does none of it.
        self.vt_stream(gpu, scene, view, &mut encoder, &cluster_wants);
        if let Some(t) = self.frame_timer.as_mut() {
            t.mark(&mut encoder, "vt-stream");
        }
        rec.mark(crate::timing::record::VT_STREAM);

        {
            let lib = self.vt_textures.as_ref();
            if let Some(n) = self.graph.node_mut::<passes::vgeom::VgeomNode>() {
                n.commit_cluster_pages(gpu, |guid, tile| {
                    lib.is_some_and(|l| {
                        l.handle(guid)
                            .is_some_and(|h| l.residency().is_resident(h, tile))
                    })
                });
            }
        }

        rec.mark(crate::timing::record::CLUSTER_COMMIT);

        // **THE SHADOW-PAGE SYNC POINT** (P27.1), for the same reason and at the
        // same place: frame F − 2's needed-page mask becomes this frame's
        // allocations and this frame's indirection table, staged on the queue
        // before any command in this encoder runs. The marking pass itself is
        // recorded AFTER the graph, where the depth buffer it reads exists.
        self.vsm_sync(gpu, scene, view);
        // **Its own segment** (island wave I4b). `vsm_sync` and the cluster-page
        // commit above it are CPU work between two GPU marks, so without a mark
        // here their recording cost is charged to `vsm-raster` — which is how a
        // caster pass that draws for 0.9 ms came to read as 8.8 ms of recording.
        // A segment must be named for what is inside it.
        if let Some(t) = self.frame_timer.as_mut() {
            t.mark(&mut encoder, "vsm-sync");
        }
        rec.mark(crate::timing::record::VSM_SYNC);

        // **THE CASTER PASS** (P27.2), recorded here and deliberately BEFORE the
        // graph: it produces the depth P27.4's receivers sample from inside the
        // lit passes, so it has to precede them. (The marking pass is the mirror
        // case and is recorded after — it consumes the depth the graph writes.)
        // A frame with no shadow-casting light, or with virtual shadows off, does
        // not touch the encoder at all.
        let rastered = match self.vsm.as_mut() {
            Some(v) => v.raster(gpu, &mut encoder, scene, view, &self.settings),
            None => 0,
        };
        self.vsm_raster_frames += u64::from(rastered > 0);
        if let Some(t) = self.frame_timer.as_mut() {
            t.mark(&mut encoder, "vsm-raster");
        }
        rec.mark(crate::timing::record::VSM_RASTER);

        let targets = self.targets.as_ref().unwrap();
        let post_hdr = if self.settings.taa {
            &targets.taa_history[cur]
        } else {
            &targets.scene_hdr
        };
        // The cloud composite reads the accumulation when there is one and the
        // raw march when there is not — the same shape `post_hdr` takes, and for
        // the same reason: one selection here rather than a branch inside the
        // pass that consumes it. `CLOUD_SLOT_RAW` is past both ping-pong indices
        // so the composite's bind-group cache tells the two cases apart.
        const CLOUD_SLOT_RAW: usize = 2;
        let (cloud_src, cloud_slot) = if self.settings.cloud_temporal {
            (&targets.cloud_history[cur], cur)
        } else {
            (&targets.cloud, CLOUD_SLOT_RAW)
        };
        let frame = FrameData {
            scene,
            view,
            targets,
            view_bg: &self.view_bg,
            out_view,
            out_size,
            out_format: self.out_format,
            settings: &self.settings,
            frame_index: self.frame_index,
            post_hdr,
            taa_history_prev: &targets.taa_history[prev],
            taa_history_cur: &targets.taa_history[cur],
            view_proj: jvp.to_cols_array(),
            taa_prev_view_proj: prev_vp,
            taa_history_valid: history_valid,
            cloud_src,
            cloud_slot,
            cloud_history_prev: &targets.cloud_history[prev],
            cloud_history_cur: &targets.cloud_history[cur],
            cloud_history_valid,
            shadow: &self.shadow,
            gi: &self.gi,
            vgeom_audit: &self.vgeom_audit,
            vis_readback: &self.vis_readback,
            scatter_audit: &self.scatter_audit,
            atmosphere: &self.atmosphere,
            wetness: &self.wetness,
            deform: &self.deform,
            view_mode: self.view_mode,
            vt: self.vt.as_ref(),
            vt_absent: &self.vt_absent,
            vsm: self.vsm.as_ref(),
            vsm_absent: &self.vsm_absent,
            vsm_receiver: &self.vsm_receiver,
            vsm_light_slots: &self.vsm_light_slots,
            vt_feedback: match (
                self.settings.vgeom.enabled && self.settings.vgeom.visbuffer,
                self.vt_feedback.as_ref(),
                self.vt.as_ref(),
            ) {
                (true, Some(f), Some(p)) => Some(VtFeedbackFrame {
                    mask: f.mask(),
                    bases: &self.vt_feedback_bases,
                    words: f.layout().words() as u32,
                    table: p.table(),
                }),
                _ => None,
            },
        };
        let vis_frames_before = self.vis_audit().frames;
        rec.mark(crate::timing::record::FRAME_DATA);
        self.graph
            .run(gpu, &mut encoder, &frame, self.frame_timer.as_mut());
        rec.mark(crate::timing::record::GRAPH);
        // No `drop(frame)`: `FrameData` implements no `Drop`, so NLL ends its
        // borrow of `self` at the last use above — the line above — and clippy's
        // `drop_non_drop` is right that spelling it out buys nothing.

        // **The virtual-texture feedback's ring copy** (P28.1), recorded here
        // rather than inside `VtFeedback::record`: the per-FRAGMENT producer
        // marks from inside the graph above, off a visibility buffer that did not
        // exist when the per-surface pass was recorded. Clear → per-surface
        // dispatch → graph → copy, one encoder and one submit, so the mask the
        // ring receives carries both producers of THIS frame and the pinned
        // `frame − 2` read is unchanged.
        //
        // "Did anything mark?" is the union of the two producers, MEASURED rather
        // than assumed: the visibility node publishes a frame counter, so a frame
        // it refused (a scene past a packing ceiling) correctly reports that its
        // per-pixel pass did not run.
        let vis_marked = self.vis_audit().frames > vis_frames_before;
        let produced = self.vt_feedback_dispatched > 0 || vis_marked;
        if let Some(f) = self.vt_feedback.as_mut() {
            f.finish(&mut encoder, self.frame_index, produced);
        }
        if let Some(t) = self.frame_timer.as_mut() {
            t.mark(&mut encoder, "vt-feedback");
        }
        rec.mark(crate::timing::record::VT_FEEDBACK);

        // **The shadow-page marking pass** (P27.1), recorded here and nowhere
        // else: it consumes the scene depth, so it has to follow every pass that
        // writes it, and it hands its buffer to a ring whose read is pinned two
        // frames later. It is NOT a graph node because it needs `&mut` on
        // renderer-owned state (the ring's slot bookkeeping), which a
        // `RenderNode` is deliberately not given — the same reason `vt_stream`
        // is a method rather than a node.
        let depth_generation = self.targets.as_ref().map_or(0, |t| t.generation);
        let marked = match self.vsm.as_mut() {
            Some(v) => {
                let depth = &self.targets.as_ref().expect("created above").depth;
                v.mark(
                    gpu,
                    &mut encoder,
                    depth,
                    depth_generation,
                    view,
                    // `jvp`, not `view.view_proj()`: this is the matrix the depth
                    // above was rasterized with, jitter included. Reconstructing
                    // a world position with the unjittered inverse would put
                    // every marked pixel up to half a pixel out whenever TAA is
                    // on, and would make the page trace a function of the Halton
                    // cursor.
                    jvp.inverse(),
                    self.frame_index,
                    // P27.5: the tier's marking stride, read off the live
                    // settings rather than baked into the pass, so a host that
                    // re-clamps mid-session marks at the new density on the very
                    // next frame.
                    self.settings.vsm.mark_stride,
                )
            }
            None => 0,
        };
        self.vsm_engaged_frames += u64::from(marked > 0);
        if let Some(t) = self.frame_timer.as_mut() {
            t.mark(&mut encoder, "vsm-mark");
            t.end(&mut encoder);
        }
        rec.mark(crate::timing::record::VSM_MARK);

        gpu.queue.submit([encoder.finish()]);
        rec.mark(crate::timing::record::SUBMIT);

        // Next frame reprojects against the matrix we actually rendered with.
        self.prev_view_proj = Some(jvp.to_cols_array());
        self.frame_index = self.frame_index.wrapping_add(1);
        // P26.3 engagement: counted where the frame actually went out, not where
        // a pool was handed over, so it says "a frame was drawn able to sample
        // virtual textures" and nothing weaker.
        if self.vt.is_some() {
            self.vt_engaged_frames += 1;
        }
        // P27.4 engagement, on exactly that precedent and counted in the same
        // place: the atlas was bound to the lit passes of a frame that went out.
        if self.vsm.is_some() {
            self.vsm_receiver_frames += 1;
        }
        rec.mark(crate::timing::record::EPILOGUE);
        if let Some(p) = rec.finish() {
            self.record_profile = p;
        }
    }
}
