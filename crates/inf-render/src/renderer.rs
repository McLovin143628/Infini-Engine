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
//! mask → composite(→swapchain)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    #[default]
    Lit,
    Unlit,
    Wireframe,
    Biomes,
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
            ViewMode::Unlit | ViewMode::Wireframe | ViewMode::Biomes => 1.0,
        }
    }

    /// The `View.flags.y` uniform value: `1.0` **only** for Biomes (P19.2), which
    /// is what makes the terrain shader's tint branch present-but-false — and its
    /// arithmetic instruction-for-instruction unchanged — in every other mode.
    pub fn biomes_flag(self) -> f32 {
        match self {
            ViewMode::Biomes => 1.0,
            ViewMode::Lit | ViewMode::Unlit | ViewMode::Wireframe => 0.0,
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
/// 4× MSAA — guaranteed support for all formats we use.
pub const SCENE_SAMPLES: u32 = 4;
/// Bloom downsample mip-chain depth (levels beyond half-res).
pub const BLOOM_MAX_MIPS: u32 = 6;

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
    /// Instance-cull audit counters (P18.5): the scatter cull compute increments
    /// them only when `enabled`, and the node records the readback copy. Off by
    /// default, so the shipping path pays nothing.
    pub scatter_audit: &'a passes::scatter::ScatterAuditResources,
    /// Shared atmosphere resources (P17.2): the two LUTs + the shared uniform the
    /// bake node writes and the sky/lit passes sample. **Resizable** — its
    /// `generation` is part of the `EnvBinding` cache key.
    pub atmosphere: &'a AtmosphereResources,
    /// Jittered view-projection of the **previous** frame (TAA reprojection).
    pub taa_prev_view_proj: [f32; 16],
    /// False on the first frame / after a resize (history has nothing usable).
    pub taa_history_valid: bool,
    /// Shoreline wetness (P20.3): the fixed-size uniform `EngineRenderer::render`
    /// packs from `scene.waters` and the lit passes read through `EnvBinding`.
    /// **Not resizable** — see `passes::EnvBinding::bind_group`'s invariant.
    pub wetness: &'a WetnessResources,
    /// Active shading view mode (R-P2). The lit scene passes select their
    /// wireframe pipeline variant on [`ViewMode::wireframe`]; the unlit branch is
    /// driven by the `View.flags.x` uniform instead (so no swap for Unlit).
    pub view_mode: ViewMode,
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
    /// P18.2 meshlet-streaming state, published by the vgeom node each frame.
    vgeom_stream: passes::vgeom::SharedStreamReport,
    /// P18.5 instance-cull counters (see [`EngineRenderer::set_scatter_audit`]).
    scatter_audit: passes::scatter::ScatterAuditResources,
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
    /// P20.3 underwater engagement counter (see
    /// [`EngineRenderer::underwater_engaged_frames`]).
    underwater: passes::underwater::UnderwaterReport,
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
        graph.add(passes::vgeom::VgeomNode::new(
            gpu,
            &view_bgl,
            vgeom_stream.clone(),
        ));
        // Classic discrete-LOD fallback (P13.4): renders the SAME vgeom content as
        // the meshlet path but through the ordinary PBR mesh pipeline, only when
        // RenderSettings.vgeom is DISABLED (the auto-tier picks the path). The exact
        // complement of VgeomNode, so scenes without vgeom content stay byte-stable.
        graph.add(passes::classic_vgeom::ClassicVgeomNode::new(gpu, &view_bgl));
        graph.add(passes::skinned::SkinnedMeshNode::new(gpu, &view_bgl));
        graph.add(passes::terrain::TerrainNode::new(gpu, &view_bgl));
        // GPU-instanced scatter (P18.5): PCG volumes + painted foliage, culled
        // per-instance on the GPU with LOD/impostor banding. LAST of the opaque
        // passes on purpose — its HZB is built from the depth every other opaque
        // pass has already written, which is the richest occluder set available —
        // and still before clouds/precipitation/translucency, since scatter IS
        // opaque. A no-op on a scene with no `scatter` batches, so every existing
        // golden is untouched.
        graph.add(passes::scatter::ScatterNode::new(gpu, &view_bgl));
        // Volumetric clouds (P17.3): a depth-tested, premultiplied-alpha raymarch
        // over everything opaque. AFTER the opaque passes so the depth buffer can
        // occlude it, BEFORE translucency so glass composites over it. A no-op
        // unless the scene enables clouds, so opaque scenes stay byte-identical.
        graph.add(passes::cloud::CloudNode::new(gpu, &view_bgl));
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
            vgeom_stream,
            scatter_audit: passes::scatter::ScatterAuditResources::new(gpu),
            atmosphere,
            next_atmosphere_generation: 2,
            wetness: WetnessResources::new(gpu),
            underwater,
        }
    }

    /// How many frames the P20.3 underwater pass has **engaged** on since this
    /// renderer was built — the off-path instrument.
    ///
    /// Unchanged across a frame ⇒ the node returned before touching the encoder,
    /// which is a claim about the command stream that no pixel comparison can
    /// make (a pass that engaged and wrote the scene back unchanged looks
    /// identical from outside). Read by
    /// `underwater_off_path_is_byte_identical`.
    pub fn underwater_engaged_frames(&self) -> u64 {
        self.underwater.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The active HDR/post settings.
    pub fn settings(&self) -> &RenderSettings {
        &self.settings
    }

    /// Replace the HDR/post settings (bloom, SSAO, TAA, exposure, dither). The
    /// viewport + player expose this to their UI; goldens set it per scene.
    pub fn set_settings(&mut self, settings: RenderSettings) {
        // Toggling TAA invalidates the accumulated history.
        if settings.taa != self.settings.taa {
            self.prev_view_proj = None;
        }
        self.settings = settings;
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

    /// Set the shading view mode (Lit / Unlit / Wireframe / Biomes). A `Wireframe`
    /// request on an adapter without `POLYGON_MODE_LINE` is clamped to `Unlit`
    /// (logged once via `tracing`) — wireframe is a hard GPU-feature requirement,
    /// so we degrade gracefully rather than fail. `Biomes` is **never** clamped: it
    /// is a uniform flag plus one texture, so every adapter that can draw terrain
    /// at all can draw it. The editor viewport + goldens set this; it is never
    /// persisted and the player never touches it.
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

        let history_valid = self.settings.taa && !resized && self.prev_view_proj.is_some();
        let prev_vp = self.prev_view_proj.unwrap_or_else(|| jvp.to_cols_array());
        let cur = (self.frame_index & 1) as usize;
        let prev = 1 - cur;

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        let targets = self.targets.as_ref().unwrap();
        let post_hdr = if self.settings.taa {
            &targets.taa_history[cur]
        } else {
            &targets.scene_hdr
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
            taa_prev_view_proj: prev_vp,
            taa_history_valid: history_valid,
            shadow: &self.shadow,
            gi: &self.gi,
            vgeom_audit: &self.vgeom_audit,
            scatter_audit: &self.scatter_audit,
            atmosphere: &self.atmosphere,
            wetness: &self.wetness,
            view_mode: self.view_mode,
        };
        self.graph.run(gpu, &mut encoder, &frame);
        gpu.queue.submit([encoder.finish()]);

        // Next frame reprojects against the matrix we actually rendered with.
        self.prev_view_proj = Some(jvp.to_cols_array());
        self.frame_index = self.frame_index.wrapping_add(1);
    }
}
