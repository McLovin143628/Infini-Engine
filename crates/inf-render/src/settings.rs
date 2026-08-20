//! Post-processing / HDR pipeline settings (P13.3a) + the pure math that drives
//! them (bloom prefilter knee, mip-chain sizing, Halton TAA jitter, SSAO
//! hemisphere kernel). The pure fns live here so they are unit-tested without a
//! GPU; the passes ([`crate::passes::bloom`], [`crate::passes::taa`],
//! [`crate::passes::ssao`]) consume them.

/// Bloom: threshold + soft-knee prefilter, separable-blur mip chain, additive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BloomSettings {
    pub enabled: bool,
    /// Luma above which a pixel starts contributing to bloom (linear HDR).
    pub threshold: f32,
    /// Soft-knee width around the threshold (0 = hard knee).
    pub knee: f32,
    /// Additive strength of the blurred bloom over the scene (0..~1).
    pub intensity: f32,
}

impl Default for BloomSettings {
    fn default() -> Self {
        // OFF by default so existing goldens keep their (regenerated) look and the
        // determinism/structural gates are unaffected by bloom.
        Self {
            enabled: false,
            threshold: 1.0,
            knee: 0.5,
            intensity: 0.06,
        }
    }
}

/// SSAO: half-res hemisphere AO from depth, blurred, multiplied into the
/// ambient/hemispheric term only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsaoSettings {
    pub enabled: bool,
    /// World-space sampling radius (metres).
    pub radius: f32,
    /// Occlusion strength multiplier (0 = none, 1 = full).
    pub intensity: f32,
    /// Range/depth bias to avoid self-occlusion acne (metres).
    pub bias: f32,
}

impl Default for SsaoSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            radius: 0.6,
            intensity: 1.0,
            bias: 0.025,
        }
    }
}

/// The renderer's HDR/post configuration. Lives on [`crate::EngineRenderer`];
/// the viewport and player both start from [`RenderSettings::default`] (post
/// off, ACES tonemap, exposure 1.0) so the default look is stable and the
/// headless goldens are deterministic. Per-project persistence is a follow-up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderSettings {
    /// Always true now (the scene always renders into an `Rgba16Float` HDR
    /// target and tonemaps as a post step); kept as an explicit field so a
    /// future LDR/2D display-transform path has a seam.
    pub hdr: bool,
    /// Linear exposure multiplier applied before the ACES tonemap.
    pub exposure: f32,
    /// Ordered dither before quantising to the 8-bit swapchain (kills banding).
    /// Deterministic (a function of pixel position), so goldens stay stable.
    pub dither: bool,
    pub bloom: BloomSettings,
    pub ssao: SsaoSettings,
    /// Temporal anti-aliasing. **OFF by default** (headless determinism); when
    /// on, the camera jitters (Halton 2,3) and a history buffer accumulates.
    pub taa: bool,
    /// Virtualized-geometry meshlet render path (P13.1b). When **on**, a scene's
    /// `vgeom_instances` are drawn by the GPU-driven meshlet path (cull+LOD
    /// compute → vertex-pulled indirect draw) instead of the classic mesh
    /// instance path. **OFF by default** so every existing golden — none of which
    /// carries `vgeom_instances` — stays byte-identical, and the field is inert
    /// on scenes with no vmesh content.
    pub vgeom: VgeomSettings,
    /// GPU-instanced scatter (P18.5): how far PCG/foliage instances draw as full
    /// meshes, how far as impostors, and how wide the cross-fade between them is.
    /// Inert on a scene with no `scatter` batches, so every existing golden is
    /// untouched.
    pub scatter: ScatterSettings,
    /// Cascaded shadow maps (P13.3b). **OFF by default** → every existing golden
    /// stays byte-identical (receivers take the un-shadowed instruction path).
    pub shadows: ShadowSettings,
    /// Dynamic global illumination (P13.3b). **OFF by default** → the hemispheric
    /// ambient path is byte-identical.
    pub gi: GiSettings,
    /// Physical-atmosphere quality (P17.2): LUT resolution, ray-march step
    /// counts and star density. Unlike its neighbours this has **no enable
    /// flag** — whether an atmosphere is drawn at all is a property of the
    /// *scene* (does the level have a `TimeOfDay` authority?), not of the
    /// renderer, so a setting could only ever disagree with the content. What
    /// the renderer owns is how expensively to draw it. Defaults to
    /// [`AtmosphereQuality::High`]; [`RenderTier::apply`](crate::caps::RenderTier::apply)
    /// clamps it **down** like every other capability knob.
    pub atmosphere: AtmosphereSettings,
    /// Water rendering quality (P20.1): grid density and whether screen-space
    /// refraction is paid for. Like [`atmosphere`](Self::atmosphere) and unlike
    /// bloom or SSAO this has **no enable flag** — whether there is water is a
    /// property of the *scene*, not of the renderer. Defaults to
    /// [`crate::water::WaterQuality::High`]; the tier clamps it **down**.
    pub water: crate::water::WaterSettings,
    /// Streaming virtual texturing (P26). The knobs are inert until P26.3 binds
    /// a VT sampler; what exists today is the **page format** decision, which is
    /// a capability question and therefore lives with the other capability
    /// knobs. Inert on a scene with no virtual textures, so every existing
    /// golden is untouched.
    pub vt: VirtualTextureSettings,
    /// Virtual shadow maps (P27). **OFF by default**, and it must stay off until
    /// P27.4 gives the pages a receiver — with it off the marking pass is never
    /// recorded, no atlas is allocated, and the CSM path is byte-identical.
    /// [`RenderTier::apply`](crate::caps::RenderTier::apply) clamps it **off** on
    /// Low, which is the P27.5 clause wired at the start of the phase rather than
    /// at the end of it.
    pub vsm: VsmSettings,
    /// **The unified streaming budget** (P28.3): one VRAM ceiling over the
    /// meshlet pools, the virtual-texture page pool and the shadow-page atlas
    /// together. Inert at the default, by construction — see [`StreamSettings`].
    pub stream: StreamSettings,
    /// **The ray-query shadow experiment** (P28.5) — off, and off on every
    /// shipped path. See [`RaytraceSettings`].
    pub raytrace: RaytraceSettings,
    /// GPU-capability auto-tier override (P13.4.2). `None` → the host probes the
    /// adapter and picks a [`RenderTier`](crate::caps::RenderTier)
    /// ([`detect_tier`](crate::caps::detect_tier)); `Some(tier)` forces it
    /// (bypasses detection — the gate forces `Low` to prove the vgeom
    /// auto-disable). Inert to rendering on its own: a host applies the tier via
    /// [`RenderTier::apply`](crate::caps::RenderTier::apply), which only clamps
    /// features **down**, so the byte-stable defaults are unaffected.
    pub tier_override: Option<crate::caps::RenderTier>,
}

/// Cascaded shadow map settings (P13.3b). The first directional light casts three
/// view-frustum-fit cascades into a `Depth32Float` array; the lit passes sample a
/// 3×3 PCF shadow factor that multiplies that light's direct term. **OFF by
/// default** so every existing golden — which renders with shadows off — stays
/// byte-identical (the receiver shaders take the un-shadowed instruction path
/// unchanged; see `shaders/mesh.wgsl`).
///
/// Cascade count + shadow-map resolution are compile-time constants
/// ([`crate::csm::SHADOW_CASCADES`] / [`crate::csm::SHADOW_RESOLUTION`]); the
/// tunable knobs are the split-scheme blend, the shadow view distance, and the two
/// bias constants (documented defaults below).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowSettings {
    /// Master enable. Off → the [`crate::passes::shadow`] node emits nothing and
    /// receivers take the byte-stable un-shadowed path.
    pub enabled: bool,
    /// Shadow view distance (metres): the far edge of the last cascade. Beyond it
    /// surfaces are fully lit. Default 60 m.
    pub max_distance: f32,
    /// Practical-split blend λ between the logarithmic and uniform split schemes
    /// (0 = uniform, 1 = logarithmic). Default 0.7 (the standard CSM value).
    pub lambda: f32,
    /// Constant depth bias in light-clip NDC units, subtracted from the receiver's
    /// compared depth to kill self-shadow acne. Default 0.0015.
    pub depth_bias: f32,
    /// Normal-offset bias in **shadow texels**: the receiver position is pushed
    /// along its normal by this many cascade texels before projection (slope acne).
    /// Default 2.0.
    pub normal_bias: f32,
    /// **Cascade blend band** (P18.4), as a fraction of each cascade's own view
    /// range. Across the last `blend × range` metres of a cascade the receiver
    /// additionally samples the *next* cascade and lerps between the two, so the
    /// resolution change stops showing up as a hard seam across the ground — the
    /// P13 deferral.
    ///
    /// `0.0` restores the pre-P18.4 hard switch **exactly**: the blend branch in
    /// `shadow_factor` is not taken and the second PCF is never issued, so a
    /// project that wants the old look pays nothing for the option. Default 0.1.
    pub cascade_blend: f32,
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            max_distance: 60.0,
            lambda: 0.7,
            depth_bias: 0.0015,
            normal_bias: 2.0,
            cascade_blend: 0.1,
        }
    }
}

/// Dynamic global-illumination settings (P13.3b) — the real-time single-bounce
/// diffuse GI. A camera-centred voxel volume is revoxelized each frame, a probe
/// grid marches rays through it to L1 spherical harmonics, and the lit passes
/// replace their hemispheric ambient constant with the probe-interpolated SH
/// irradiance. **OFF by default** so every existing golden keeps the hemispheric
/// ambient path byte-identical.
///
/// The voxel grid (64³) + probe grid (16×8×16) dimensions are compile-time
/// constants ([`crate::gi`]); the tunables are the world extent, ray count, and
/// output intensity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GiSettings {
    /// Master enable. Off → the [`crate::passes::gi`] nodes emit nothing and the
    /// receivers keep the byte-stable hemispheric ambient term.
    pub enabled: bool,
    /// World side length (metres) of the camera-centred voxel/probe volume.
    /// Default 40 m.
    pub extent: f32,
    /// Rays marched per probe (fixed golden-spiral directions, deterministic).
    /// Default 48.
    pub rays: u32,
    /// Multiplier on the reconstructed SH irradiance before it feeds the ambient
    /// term. Default 1.0.
    pub intensity: f32,
    /// Voxel/probe resolution + per-frame primitive budget (P18.4). Defaults to
    /// [`GiQuality::High`], which is **exactly** the pre-P18.4 geometry (64³ /
    /// 16×8×16), so the tiering itself moves no pixels;
    /// [`RenderTier::apply`](crate::RenderTier::apply) clamps it **down** like every
    /// other capability knob.
    pub quality: crate::gi::GiQuality,
    /// Ceiling on primitives voxelized per frame, before the
    /// [`quality`](GiSettings::quality) cap is applied on top (the effective budget
    /// is the smaller of the two). Overflow is *reported*
    /// ([`EngineRenderer::gi_audit`](crate::EngineRenderer::gi_audit)), never
    /// silently dropped: the nearest primitives are kept, so what is lost is
    /// distant. Default 4096 — the P18.4 replacement for `MAX_GI_INSTANCES = 256`.
    pub instance_budget: u32,
    /// Probes re-integrated per frame (**temporal amortization**, P18.4).
    ///
    /// `0` = **full update**, every probe every frame — the default, and what the
    /// goldens and determinism gates render with, because a full update makes a
    /// frame a pure function of the scene with no convergence transient to reason
    /// about. A non-zero budget sweeps the probe grid round-robin on a
    /// renderer-side cursor ([`crate::gi::ProbeSchedule`]): two cold renders still
    /// match, and a static scene's converged steady state is byte-identical to the
    /// full update — but a *moving* camera trades probe latency for the saving,
    /// which is why it is opt-in rather than default.
    pub probe_budget: u32,
    /// SH-derived **specular** (P18.4): the ambient specular term becomes radiance
    /// reconstructed along the reflection vector instead of a flat
    /// `ambient × f0 × 0.5`. Cheap (it reuses the probe fetch the diffuse term
    /// already does) and therefore **on by default** — but only reachable when
    /// [`enabled`](GiSettings::enabled) is set, so no non-GI golden is affected.
    pub specular: bool,
    /// **SSR v1** (P18.4): a screen-space raymarch against the scene depth that
    /// re-anchors the specular probe fetch at the ray's hit point. **Off by
    /// default** — it forces the depth prepass on
    /// ([`needs_depth_prepass`](RenderSettings::needs_depth_prepass)) and the march
    /// is 24 taps, so it is a deliberate opt-in; with it off the lit shaders take
    /// the identical instruction stream.
    pub ssr: bool,
    /// SSR march length in metres. Default 8 m — contact reflections, not mirrors.
    pub ssr_distance: f32,
    /// SSR **relative** depth-thickness tolerance: a hit is accepted when the ray
    /// sample sits behind the depth buffer by less than this fraction of its own
    /// view distance. Relative rather than absolute so one number works at every
    /// scale under the reverse-infinite-Z projection (which has no far plane to
    /// linearize against). Default 0.15.
    pub ssr_thickness: f32,
}

impl Default for GiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            extent: 40.0,
            rays: 48,
            intensity: 1.0,
            quality: crate::gi::GiQuality::High,
            instance_budget: 4096,
            probe_budget: 0,
            specular: true,
            ssr: false,
            ssr_distance: 8.0,
            ssr_thickness: 0.15,
        }
    }
}

/// Virtualized-geometry (meshlet) render-path settings (P13.1b).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VgeomSettings {
    /// Master enable for the meshlet path. Off → the [`crate::passes::vgeom`]
    /// node emits no commands (byte-stable classic path).
    pub enabled: bool,
    /// Target screen-space error, in **pixels**. A meshlet is drawn when its
    /// projected object-space error stays under this tolerance while its
    /// coarser-replacement's exceeds it (the LOD cut). Larger ⇒ coarser LODs
    /// selected sooner (fewer meshlets). Default ≈ 1 px.
    pub pixel_error: f32,
    /// Per-meshlet flat debug colouring (hash of meshlet id) instead of PBR
    /// shading — a visual proof of the cluster/LOD structure for the golden.
    pub debug_meshlets: bool,
    /// HZB occlusion test in the cull compute. **ON by default** since P18.1:
    /// the test is provably *subtractive* — it only removes meshlets the
    /// hierarchical depth proves contribute zero fragments — so a frame with it on
    /// is pixel-identical to a frame with it off (`tests/vgeom_occlusion.rs`), and
    /// the CPU-reference parity gate is unaffected because occlusion filters the
    /// LOD+frustum+cone cut rather than being part of it. See
    /// [`crate::passes::vgeom`] for the proof and
    /// [`crate::caps::AdapterCaps::supports_vgeom_occlusion`] for the capability
    /// floor; [`RenderTier::apply`](crate::caps::RenderTier::apply) clamps it off
    /// below the High tier.
    pub occlusion: bool,
    /// Real **two-pass** occlusion (P18.1): draw last frame's visible meshlets,
    /// rebuild the HZB from the depth they wrote, then cull + draw the remainder
    /// against it — so vgeom geometry occludes vgeom geometry. ON by default;
    /// requires [`occlusion`](VgeomSettings::occlusion).
    ///
    /// `false` falls back to the single-pass v1 shape (one cull, one draw, HZB
    /// from whatever the scene depth holds when the node starts). That path is
    /// kept deliberately: it is the temporal-state-free reference the CPU-parity
    /// machinery mirrors, and the safety valve if a driver mishandles the
    /// multisampled depth load.
    pub two_pass: bool,
    /// Backface normal-cone culling in the cull compute.
    pub cone_cull: bool,
    /// Frustum-sphere culling in the cull compute.
    pub frustum_cull: bool,
    /// **Visibility-buffer shading** (P28.1). With it on the meshlet path
    /// rasterizes `instance ⊕ meshlet ⊕ triangle` ids into a single-sample
    /// `R32Uint` target and a fullscreen material-resolve pass shades from them;
    /// with it off the forward meshlet raster shades in its own fragment stage,
    /// exactly as it has since P13.1b.
    ///
    /// **OFF by default on every tier, and that is the P28.1 ruling rather than
    /// a placeholder.** `docs/memos/p28-1-visbuffer.md` §5 carries the
    /// measurement: the visibility buffer is single-sample by the ROADMAP's own
    /// clause and the scene targets are a compile-time `SCENE_SAMPLES = 4`, so
    /// meshlet silhouettes lose the 4x coverage the forward path resolves — a
    /// loss TAA recovers and nothing else does, and
    /// [`RenderSettings::taa`](crate::RenderSettings::taa) is itself off by
    /// default for headless determinism. A default that turned one on without
    /// the other would ship a visibly worse frame; a default that turned both on
    /// would re-bless fifty-four byte-frozen goldens. So it is a setting, the
    /// forward path stays the shipped default, and P28.3 revisits it when the
    /// unified streamer makes the cost side of the ledger measurable.
    ///
    /// The forward path is also what a refused frame falls back to: see
    /// [`crate::visbuffer::VisPacking::admit`], whose ceilings this mode is
    /// admitted under every frame.
    ///
    /// Requires [`enabled`](VgeomSettings::enabled); clamped off below High by
    /// [`RenderTier::apply`](crate::caps::RenderTier::apply) — which it inherits
    /// for free, because Medium and Low clear `enabled` itself.
    pub visbuffer: bool,
    /// Meshlet **streaming** budget (P18.2): the VRAM ceiling for the shared
    /// meshlet pools, the per-frame load cap, and the eviction hysteresis.
    ///
    /// There is no enable flag, deliberately — streaming is the only path a
    /// `.inf_vmesh` reaches the GPU by, exactly as `.inf_terrain` has only the
    /// streamed path. What a host owns is *how much* it may hold resident. The
    /// default is generous enough that every shipping sample is fully resident on
    /// its first frame, which is what makes a streamed frame byte-identical to
    /// the pre-P18.2 whole-upload frame (the goldens' equivalence gate); a smaller
    /// budget degrades to coarser meshlets, never to a hole.
    pub stream: inf_vgeom::VgeomStreamBudget,
}

impl Default for VgeomSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            pixel_error: 1.0,
            debug_meshlets: false,
            occlusion: true,
            two_pass: true,
            cone_cull: true,
            frustum_cull: true,
            visbuffer: false,
            stream: inf_vgeom::VgeomStreamBudget::default(),
        }
    }
}

/// GPU-instanced scatter settings (P18.5).
///
/// **Deliberately renderer-side, and there is no enable flag for the *feature*.**
/// Whether a level has scatter is a property of the content (does it carry a
/// `PcgVolume` or a `Foliage`?), exactly as `AtmosphereSettings` argues about the
/// sky; what a host owns is how far it is willing to pay to draw it. The one
/// *content* knob — `PcgVolume::draw_distance`, authored since P10.5 — rides on
/// each [`crate::scene::ScatterBatch`] and clamps these bands **down**, so no
/// schema change was needed to land LOD banding.
///
/// [`gpu`](Self::gpu) selects the *mechanism*, not the content: with it off the
/// same batches draw through a CPU-culled classic instanced path, so a downlevel
/// adapter still sees its foliage. `RenderTier::apply` clears it below High
/// (the `ClassicVgeomNode` precedent).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScatterSettings {
    /// Use the GPU cull + indirect-draw path. **ON by default**; requires compute
    /// shaders and indirect execution ([`crate::caps::AdapterCaps::clamp_scatter`]).
    /// Off ⇒ the CPU fallback draws the same instances through the rigid mesh
    /// pipeline, distance-culled on the CPU with no impostors.
    pub gpu: bool,
    /// Distance out to which an instance draws as its **full mesh**, in metres.
    /// Beyond it the impostor takes over (or, with [`impostors`](Self::impostors)
    /// off, the mesh simply fades out).
    pub mesh_distance_m: f32,
    /// Distance out to which an instance draws **at all**, in metres. Past it the
    /// instance is culled by the compute pass and costs one thread.
    pub cull_distance_m: f32,
    /// Width of the dithered cross-fade band, in metres, applied at *both*
    /// transitions (mesh↔impostor and impostor↔nothing).
    pub fade_band_m: f32,
    /// Draw impostors in the far band. Off ⇒ the mesh band runs all the way to
    /// [`cull_distance_m`](Self::cull_distance_m) and fades out there instead.
    pub impostors: bool,
    /// Per-instance frustum culling in the cull compute.
    pub frustum_cull: bool,
    /// Per-instance HZB occlusion culling (P18.1's pyramid, rebuilt after the
    /// opaque geometry so scatter is occluded by meshes, meshlets and terrain
    /// alike). **ON by default**: like the meshlet test it is provably
    /// *subtractive*, so a frame with it on is pixel-identical to one with it off.
    pub occlusion: bool,
    /// Sway scattered foliage with the level's wind (P22.1). **OFF by default.**
    ///
    /// Off rather than on for one reason and it is not taste: sway is a visible
    /// change to every frame that contains grass, and the 49 goldens committed
    /// before P22.1 are the engine's record of what those frames look like.
    /// Turning an ambient animation on by default would have re-blessed all of
    /// them at once.
    ///
    /// It is a **setting** rather than a consequence of the deformation field
    /// because the alternative — sway on iff some cell is live somewhere — makes
    /// an ambient effect depend on whether anybody happens to have walked
    /// nearby, which is neither deterministic-looking nor explicable.
    pub foliage_wind: bool,
}

impl Default for ScatterSettings {
    fn default() -> Self {
        Self {
            gpu: true,
            mesh_distance_m: 120.0,
            cull_distance_m: 400.0,
            fade_band_m: 20.0,
            impostors: true,
            frustum_cull: true,
            occlusion: true,
            foliage_wind: false,
        }
    }
}

impl ScatterSettings {
    /// Ceilings a tier imposes on the three distance bands, as a fraction of the
    /// defaults. Absolute metres rather than a scale factor, deliberately: the tier
    /// clamps must be **idempotent and order-independent** (`caps::tests::
    /// apply_clamps_down_never_up` applies them twice and demands the same
    /// settings), and repeated multiplication is neither.
    pub const MEDIUM_BANDS_M: (f32, f32, f32) = (72.0, 240.0, 12.0);
    pub const LOW_BANDS_M: (f32, f32, f32) = (42.0, 140.0, 7.0);

    /// A stamp over exactly the fields the **shadow caster pack** reads, so the
    /// shadow node's re-pack key moves when a tier clamp changes the caster band
    /// and stays put when an unrelated knob (impostors, occlusion, the GPU path)
    /// does. Spelled here rather than in the pass so the two cannot drift.
    pub fn caster_stamp(&self) -> u64 {
        (self.mesh_distance_m.to_bits() as u64) << 32 | self.cull_distance_m.to_bits() as u64
    }

    /// Pull the three distance bands in to at most `(mesh, cull, fade)` metres.
    /// A pure `min` on each, so it only ever lowers, is idempotent, and composes
    /// in any order with the other clamps.
    pub fn clamp_bands(mut self, bands: (f32, f32, f32)) -> Self {
        self.mesh_distance_m = self.mesh_distance_m.min(bands.0);
        self.cull_distance_m = self.cull_distance_m.min(bands.1);
        self.fade_band_m = self.fade_band_m.min(bands.2);
        self
    }
}

/// Physical-atmosphere render settings (P17.2).
///
/// Deliberately thin: the atmosphere's *parameters* live on the scene (projected
/// from the level's `SkyAtmosphere` component), because they are content. What
/// lives here is the **cost** — how big the LUTs are and how many march steps
/// they take — which is a property of the machine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereSettings {
    /// LUT resolution / ray-march budget / star density.
    pub quality: crate::atmosphere::AtmosphereQuality,
}

impl Default for AtmosphereSettings {
    /// High. Unlike bloom/SSAO/TAA/shadows/GI this is **not** "off by default",
    /// and it does not need to be: the atmosphere node dispatches nothing at all
    /// unless the scene enables it, so a default-settings render of a
    /// pre-P17.2 scene issues exactly the commands it always did.
    fn default() -> Self {
        Self {
            quality: crate::atmosphere::AtmosphereQuality::High,
        }
    }
}

/// **The page pool's budget on the tier below High**, in bytes: 12 MiB.
///
/// Half the default. Not a taste: the analytic floor is bounded at
/// `VT_FLOOR_LEVELS`'s ≤ 21 pages per texture plus
/// [`crate::VT_FLOOR_MAX_TILES`]'s 16 per visible surface, so what a smaller
/// pool costs is *refinement*, never a hole — the shader still resolves to a
/// resident ancestor and the surface is blurrier rather than absent. 12 MiB is
/// 1 357 BC1 pages, which holds the camera-free floor of **64** textures before
/// anything is deferred at all.
pub const VT_BUDGET_MEDIUM_BYTES: u64 = 12 * 1024 * 1024;

/// **The page pool's budget on Low**, in bytes: 6 MiB — 678 BC1 pages, the
/// camera-free floor of 32 textures.
///
/// The Low tier is where the clamp law earns its keep: this turns nothing off.
/// A Low-tier machine samples the same virtual textures through the same door
/// and sees the same content at a coarser level, which is precisely the
/// degradation virtual texturing exists to make possible — and it is why the
/// budget is a *number* here and not a `bool`.
pub const VT_BUDGET_LOW_BYTES: u64 = 6 * 1024 * 1024;

/// Streaming virtual texturing (P26).
///
/// Two knobs, and neither is an on/off switch — whether a scene has virtual
/// textures is a property of its content, not of the renderer, the same
/// reasoning [`AtmosphereSettings`] and [`crate::water::WaterSettings`] are
/// shaped by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VirtualTextureSettings {
    /// Upload tiles as **block-compressed** pages (`TEXTURE_COMPRESSION_BC`).
    ///
    /// `true` by default: a `.inf_tex` cooks its tiles as BC1/BC3 blocks, so a
    /// BC-capable adapter uploads the stored bytes with no decode at all — the
    /// whole point of the container's format design.
    ///
    /// An adapter that does not expose the feature has this **clamped off** by
    /// [`AdapterCaps::clamp_bc_tiles`](crate::caps::AdapterCaps::clamp_bc_tiles),
    /// and the residency door transcodes each tile to RGBA8 on the CPU
    /// (`TiledTextureReader::tile_rgba8`) before uploading it — the same door,
    /// one format decision earlier, at **8×** the page bytes for BC1 and 4× for
    /// BC3 (4 and 8 bits per texel against RGBA8's 32).
    ///
    /// This is not a tier knob and [`RenderTier::apply`](crate::caps::RenderTier::apply)
    /// deliberately does not touch it: BC support is orthogonal to how much GPU
    /// there is, exactly as `POLYGON_MODE_LINE` is.
    pub bc_tiles: bool,
    /// **The physical page pool's VRAM ceiling**, in bytes (P26.5).
    ///
    /// The one number that decides how much unique texture detail is resident at
    /// once, and therefore the natural tier knob — which is what
    /// [`RenderTier::apply`](crate::caps::RenderTier::apply) makes it, clamping
    /// to [`VT_BUDGET_MEDIUM_BYTES`] / [`VT_BUDGET_LOW_BYTES`] with a `min`, so a
    /// caller that already asked for less keeps less and no tier ever hands a
    /// machine a bigger atlas than it asked for.
    ///
    /// Both hosts read it through
    /// [`build_vt_level`](crate::build_vt_level)'s `budget_bytes` argument, which
    /// is why it lives in settings rather than being passed at each call site:
    /// the editor viewport and the shipped player would otherwise be two places
    /// that decide how much VRAM a level gets.
    pub budget_bytes: u64,
    /// **Blend between two pyramid levels instead of snapping to one** (Wave T,
    /// the texture document's trilinear item).
    ///
    /// `vt_mip` truncates the gradient level of detail and samples one page, so
    /// a virtual texture changes in a visible step as the camera dollies. With
    /// this on, `vt_sample` takes a second tap at `m + 1` and mixes on the
    /// fractional part — the document's own prescription, early-outs included
    /// (`blend < 0.01`, and the pyramid's floor), so the second address walk is
    /// paid only where it shows.
    ///
    /// **Off by default, and that is a law rather than a taste.** It moves
    /// pixels in every textured scene, and the 54 committed goldens are frozen
    /// (`phase26_gate` / `phase27_gate` / `phase28_gate` each assert the set's
    /// digest independently). Turning it on is a blessed decision with a
    /// measurement behind it, not a default flip.
    ///
    /// Not a tier knob: it is one extra tap on the fragments that are actually
    /// between levels, which is a quality/cost trade an author makes rather than
    /// a capability a machine has.
    pub trilinear: bool,
    /// **Per-frame page-upload ceiling, in bytes** (island wave I4, IB-16). `0`
    /// = unlimited.
    ///
    /// The third knob, and the first that is about *time* rather than about
    /// space: `budget_bytes` is how much the atlas holds and is never exceeded;
    /// this is how much one frame may write, and a want past it is **deferred,
    /// not dropped** — re-offered on the next frame's want set, so a burst is
    /// smoothed and a tile arrives late rather than never.
    ///
    /// Not an on/off switch either, for [`VirtualTextureSettings`]' own reason:
    /// virtual texturing is the only way a `.inf_tex` reaches the GPU.
    pub upload_budget_bytes: u64,
}

impl Default for VirtualTextureSettings {
    fn default() -> Self {
        Self {
            bc_tiles: true,
            budget_bytes: crate::DEFAULT_VT_BUDGET_BYTES,
            trilinear: false,
            upload_budget_bytes: inf_vt::DEFAULT_VT_UPLOAD_BUDGET_BYTES,
        }
    }
}

/// **The meshlet pools' budget on the tier below High**, in bytes: 128 MiB.
///
/// Half the default, and the meshlet streamer's **first** tier knob (P28.3):
/// `VgeomStreamBudget::budget_bytes` shipped in P18.2 and no tier ever touched
/// it, so a Medium machine was handed the High ceiling by a settings struct that
/// never said so. It changes nothing today — Medium and Low both clear
/// `VgeomSettings::enabled`, so the pools are not allocated at all — and that is
/// exactly why it can land: the clamp exists before anything can ship that
/// forgets it, which is the shape `VsmSettings::enabled`'s own clamp took at the
/// start of P27 rather than at the end.
///
/// What a smaller pool costs is the same thing it has always cost: a coarser
/// meshlet cut. Softer detail, never a hole — residency is a prefix and page 0
/// is the always-resident floor.
pub const VGEOM_BUDGET_MEDIUM_BYTES: u64 = 128 * 1024 * 1024;

/// **The meshlet pools' budget on Low**, in bytes: 64 MiB.
pub const VGEOM_BUDGET_LOW_BYTES: u64 = 64 * 1024 * 1024;

/// **The unified streaming budget on High**, in bytes: the **sum of the three
/// consumers' own defaults** — 256 MiB of meshlet pools + 24 MiB of virtual
/// texture pages + 64 MiB of shadow-page atlas = 344 MiB.
///
/// It is the sum on purpose, and that is the whole of why this batch moves no
/// golden and no gate: `inf_stream::arbitrate` is an **identity** when the
/// requests fit the total, so at the shipped defaults every consumer is handed
/// exactly the number it had before there was an arbiter
/// (`inf_stream::budget::tests::the_shipped_default_is_an_identity`).
///
/// What the number buys is that the sum is now **bounded by something**. Before
/// P28.3 the three ceilings were independent and nothing anywhere said what a
/// machine's total streaming residency could reach; a project that raised one of
/// the three raised the total, silently. Lowering this one number now divides
/// deterministically between the three instead — floors first, then an even
/// water-fill clamped at each want.
pub const DEFAULT_STREAM_BUDGET_BYTES: u64 = inf_vgeom::DEFAULT_VGEOM_BUDGET_BYTES
    + crate::DEFAULT_VT_BUDGET_BYTES
    + inf_vsm::DEFAULT_VSM_BUDGET_BYTES;

/// **The unified streaming budget on Medium**, in bytes: the sum of that tier's
/// three ceilings — 128 + 12 + 32 = 172 MiB.
///
/// Consistent with the per-consumer clamps by construction rather than by
/// coincidence, so the tier's own arbitration is an identity too and the
/// arbiter binds exactly when a *caller* asks for more than its tier's total.
pub const STREAM_BUDGET_MEDIUM_BYTES: u64 =
    VGEOM_BUDGET_MEDIUM_BYTES + VT_BUDGET_MEDIUM_BYTES + VSM_BUDGET_MEDIUM_BYTES;

/// **The unified streaming budget on Low**, in bytes: 64 + 6 + 16 = 86 MiB.
pub const STREAM_BUDGET_LOW_BYTES: u64 =
    VGEOM_BUDGET_LOW_BYTES + VT_BUDGET_LOW_BYTES + VSM_BUDGET_LOW_BYTES;

/// **The unified streaming budget** (P28.3, clause 2): one VRAM ceiling over
/// the three page systems.
///
/// One knob and no enable flag, for `AtmosphereSettings`' reason: whether a
/// level streams is a property of its content. What a host owns is how much
/// VRAM the three of them may hold between them — which is a question nothing
/// in this tree could ask before, because there were three ceilings and no
/// fourth number over them.
///
/// The per-consumer numbers (`VgeomStreamBudget::budget_bytes`,
/// `VirtualTextureSettings::budget_bytes`, `VsmSettings::budget_bytes`) stay,
/// and they become **requests**: each consumer still says what it would use,
/// this says what the three may have, and `inf_stream::arbitrate` divides.
/// Keeping them is not a courtesy — a request is content-shaped (a level with
/// one 8 k texture and no meshlets should not be handed a third of the budget
/// for geometry), and an arbiter with no requests would have to guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamSettings {
    /// The ceiling on the **sum** of the three consumers' residency, in bytes.
    /// Clamped down by [`RenderTier::apply`](crate::caps::RenderTier::apply)
    /// like every other capability knob.
    pub budget_bytes: u64,
    /// **Predictive prefetch** (P28.4) — see [`PredictSettings`].
    pub predict: PredictSettings,
}

impl Default for StreamSettings {
    fn default() -> Self {
        Self {
            budget_bytes: DEFAULT_STREAM_BUDGET_BYTES,
            predict: PredictSettings::default(),
        }
    }
}

/// **The predictor's two knobs** (P28.4): whether it speculates, and how far
/// ahead.
///
/// # Why `enabled` defaults to `true` and changes nothing
///
/// Every other streaming feature in this file ships off and is turned on by a
/// host. This one ships on, and the reason is that its real enable flag is
/// somewhere a settings struct cannot reach: the predictor consumes
/// `inf_math::CameraHistory`, and a history is empty until a host **commits** a
/// camera pose at its fixed step (`EngineRenderer::commit_camera`). A host that
/// does not — the editor viewport, whose flycam is driven by OS input at render
/// rate and is not committed input at all — gets `dead_reckon() == None`, emits
/// no speculative want, and streams byte-for-byte as it did before P28.4.
///
/// So the flag is what a host with a committed camera turns *off*, and the
/// default is the honest one: a host that can prove its camera is committed
/// gets the prefetch without opting in twice.
///
/// # Why the horizon is in ticks
///
/// The ROADMAP's clause says 200–500 **ms**, and a millisecond is wall clock —
/// which the predictor may not read, on pain of the whole determinism argument.
/// A tick is committed, so the horizon is stored in ticks and a host converts
/// once at its own door with `inf_math::horizon_ticks(ms, hz)` against its own
/// fixed step. At the shipped 60 Hz step the default **18** is 300 ms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictSettings {
    /// Whether speculative wants are emitted at all.
    ///
    /// [`RenderTier::apply`](crate::caps::RenderTier::apply) clears it on
    /// **Low**, and [`RenderTier::clamp_mobile`](crate::caps::RenderTier::clamp_mobile)
    /// clears it too. Speculation
    /// spends slots a smaller pool does not have spare: on Low the unified
    /// budget is 86 MiB against High's 344, and a lane that only ever takes
    /// *idle* capacity has none to take. The clamp is the `vsm.enabled` shape —
    /// wired with the feature rather than after it.
    pub enabled: bool,
    /// How many committed **ticks** ahead to dead-reckon.
    ///
    /// **0** since P28.5 — the committed pose, no lead — and the reason is a
    /// measurement rather than a taste. See
    /// [`DEFAULT_PREDICT_HORIZON_TICKS`] and
    /// `docs/memos/p28-5-lead-time-ruling.md`.
    ///
    /// The knob is real and the whole dead-reckoner is still behind it: a
    /// non-zero value extrapolates exactly as P28.4 built it, and the day this
    /// loop grows a latency between *admitted* and *sampleable* a lead becomes
    /// worth paying for again.
    pub horizon_ticks: u32,
}

impl Default for PredictSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            horizon_ticks: DEFAULT_PREDICT_HORIZON_TICKS,
        }
    }
}

/// **The shipped horizon**, in committed ticks — **0**, the committed pose.
///
/// # The deviation, and the measurement that forced it
///
/// The ROADMAP's P28.4 clause 1 prescribes a *"200–500 ms horizon"*. P28.4
/// shipped 18 ticks (300 ms) out of that band and its own gate refuted it: the
/// lead is a **cost** at every horizon on the only fixture that exists, and the
/// speculative *lane* — a CPU-side want set at the refinement's cap, ranked
/// below every proved class — is the entire win.
///
/// Measured on `whip_pan.rs`'s 360° path, blur frames against **131** with the
/// predictor off:
///
/// | lead (ticks) | **0** | 3 | 6 | 12 | 18 | 24 | 36 |
/// |---|---|---|---|---|---|---|---|
/// | blur frames | **105** | 108 | 113 | 115 | 115 | 112 | 124 |
/// | arrival-window blur / 1 728 | **64** | 96 | 144 | 176 | 176 | 128 | 224 |
///
/// The mechanism is structural, not a tuning accident: `apply_wants` seats a
/// miss the frame it is offered, with no per-frame admission throttle and **no
/// latency between admitted and sampleable**, so having asked earlier buys
/// nothing anywhere in this loop and every want spent on where the camera will
/// be is a slot not spent on where it is. It is
/// `a_saturated_floor_cannot_be_prefetched_and_the_arm_says_so`, one want class
/// up.
///
/// So P28.5 ships the lane at zero lead and records the deviation rather than
/// shipping a knob its own gate measures backwards
/// (`docs/memos/p28-5-lead-time-ruling.md`). **An unmeasured prescription can
/// be backwards** — the P20/P25 law — and this one was.
///
/// The dead-reckoner, the sweep and the tripwire all stay: see
/// [`ROADMAP_PREDICT_HORIZON_TICKS`].
pub const DEFAULT_PREDICT_HORIZON_TICKS: u32 = 0;

/// **The horizon the ROADMAP's clause prescribed** — 18 ticks, 300 ms at the
/// 60 Hz fixed step, the middle of its 200–500 ms band.
///
/// Kept as a named constant rather than deleted, because it is the *lead* half
/// of the A/B that produced the ruling above. `whip_pan.rs`'s
/// `a_lead_time_costs_this_fixture_what_the_lane_earns_it` runs the shipped
/// zero against it and fails the day the inequality inverts — which is how the
/// ruling re-opens: by a red test rather than by memory. The day this tree
/// grows an admission throttle or a loader with real latency, this is the
/// number the default goes back to.
pub const ROADMAP_PREDICT_HORIZON_TICKS: u32 = 18;

/// **The page atlas's budget on the tier below High**, in bytes: 32 MiB — half
/// the default, 512 pages.
///
/// Not a taste. What a smaller atlas costs is *shadow resolution*, never a hole:
/// a page that cannot be seated resolves to its coarsest resident ancestor and
/// the shadow there is blurrier, exactly the degradation virtual shadow mapping
/// exists to make possible — which is why the tier knob is a **number** and not
/// a `bool`.
pub const VSM_BUDGET_MEDIUM_BYTES: u64 = 32 * 1024 * 1024;

/// **The page atlas's budget on Low**, in bytes: 16 MiB — 256 pages.
///
/// Inert in practice, because Low also clamps [`VsmSettings::enabled`] off and
/// keeps CSM (the P27.5 clause). It exists so the clamp is total: a host that
/// forced VSM on at Low would still get the Low atlas rather than the High one.
pub const VSM_BUDGET_LOW_BYTES: u64 = 16 * 1024 * 1024;

/// **The clipmap's page grid on Medium**: 32 pages a side — 4 k virtual texels
/// per level against High's 8 k.
pub const VSM_CLIPMAP_PAGES_MEDIUM: u32 = 32;

/// **The marking dispatch's stride below High** (P27.5): every second pixel on
/// both axes, so the pass costs a **quarter** of the threads and a quarter of
/// the depth loads.
///
/// The P27.1 ledger's own remainder — *"the marking dispatch is one thread per
/// pixel with no stride, so there is no tier knob for its cost"* — closed as a
/// knob rather than as a default: `VsmSettings::mark_stride` is **1** on High
/// and everywhere else the caller does not lower quality, and the clamp is a
/// `max` because a *larger* stride is *less* work.
///
/// What it costs is measured rather than adjectival, and it is the phase's own
/// chosen direction: a page only a skipped pixel would have asked for is not
/// marked, is not resident, and resolves to [`inf_vsm::VSM_ENTRY_NONE`] —
/// which the receiver reads as **lit**. So a coarser marking grid leaks light
/// at a silhouette rather than punching a hole in one.
pub const VSM_MARK_STRIDE_MEDIUM: u32 = 2;

/// **The PCF kernel's radius below High** (P27.5): `0`, a single tap — a hard
/// shadow edge instead of the 3 × 3 filter.
///
/// The clamp is a `min` because a *larger* radius is *more* work: nine
/// `textureLoad`s against one, over the same single table resolution
/// (`crate::vsm_receiver::pcf_resolution_cost` — the clamped kernel resolves
/// once whatever the radius). The derived slope bias follows the radius rather
/// than staying at High's, because it *is* `(R + ½)·√2` and a bias sized for a
/// kernel that is not running is peter-panning nobody asked for.
pub const VSM_PCF_RADIUS_MEDIUM: u32 = 0;

/// The largest [`VsmSettings::mark_stride`] the settings boundary accepts.
///
/// Eight is a 64× cost reduction and a 8 × 8-pixel marking grid; past it the
/// signal stops being *screen-driven* in any useful sense — a 1 % of pixels
/// sample is a sample of the depth buffer, not a coverage of it.
pub const VSM_MAX_MARK_STRIDE: u32 = 8;

/// The largest [`VsmSettings::pcf_radius`] the settings boundary accepts.
///
/// Three is a 7 × 7 kernel, 49 taps, and `pcf_crossing_fraction` at 128 texels
/// says **9.1 %** of a page's texels would have a tap outside it. Past that the
/// clamped kernel's error (the dropped weight at a page edge) stops being a
/// bounded correction and becomes the filter.
pub const VSM_MAX_PCF_RADIUS: u32 = 3;

/// Virtual shadow maps (P27).
///
/// Unlike [`VirtualTextureSettings`] this **does** carry an enable flag, and the
/// difference is not an oversight: virtual texturing is the only way a `.inf_tex`
/// reaches the GPU, so "off" would mean "no textures", while VSM is an
/// *alternative* to a cascaded shadow map that still ships. The flag is what
/// P27.5's "High/Medium run VSM; Low keeps CSM" clamps, and
/// [`RenderTier::apply`](crate::caps::RenderTier::apply) may only ever clear it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VsmSettings {
    /// Master enable. **False** until P27.4 gives the pages a receiver: with it
    /// off the marking pass is never recorded, no atlas is allocated, and every
    /// golden records the command stream it always did.
    pub enabled: bool,
    /// The physical page atlas's VRAM ceiling, in bytes. The tier knob, clamped
    /// with a `min` to [`VSM_BUDGET_MEDIUM_BYTES`] / [`VSM_BUDGET_LOW_BYTES`].
    pub budget_bytes: u64,
    /// Pages across one directional clipmap level. 64 at 128-texel pages is
    /// **8 192 virtual texels per level**, which is the phase's "≥8 k effective
    /// on High". Clamped down to [`VSM_CLIPMAP_PAGES_MEDIUM`] below High.
    pub clipmap_pages_per_side: u32,
    /// Clipmap levels. Eight doublings from
    /// [`first_level_extent_m`](Self::first_level_extent_m) reaches 128× it.
    pub clipmap_levels: u32,
    /// Half the world extent of clipmap level 0, in metres. 32 m at 64 pages of
    /// 128 texels is a **7.8 mm** shadow texel under the camera.
    pub first_level_extent_m: f32,
    /// Quadtree levels for a spot light: 7 is 64 pages a side at level 0, i.e.
    /// 8 192 virtual texels across the cone.
    pub spot_levels: u32,
    /// Quadtree levels per cube face for a point light: 6 is 32 pages a side,
    /// 4 096 virtual texels per face, six faces.
    pub point_levels: u32,
    /// The near plane of a spot's or a cube face's projection, in metres.
    pub perspective_near_m: f32,
    /// **The clipmap level blend band** (P27.4) — the virtual analogue of
    /// [`ShadowSettings::cascade_blend`], as a fraction of a level's own band.
    ///
    /// A receiver near the edge of the level it was served additionally resolves
    /// the **next coarser** level and lerps, so the resolution change stops
    /// showing as a line across the ground. A clipmap level has two edges a
    /// receiver can approach — its resolution band and its footprint ring — and
    /// the weight is the larger of the two proximities
    /// ([`crate::vsm_receiver::vsm_blend_weight`]).
    ///
    /// `0.0` restores the hard switch **exactly**: the branch is not taken and
    /// the second resolve is never issued, which is the escape hatch
    /// `cascade_blend` already ships. Default 0.1, the cascade's own default, so
    /// the two paths look alike at the seam rather than only being spelled
    /// alike.
    ///
    /// Deliberately a **second** knob rather than a reader of `cascade_blend`:
    /// the ROADMAP's "clipmap level blend replaces cascade blend" is about which
    /// blend *runs*, and a project that turned the cascade's seam fix off must
    /// not silently turn the clipmap's off with it — nor the reverse. The CSM's
    /// own knob is untouched by this batch.
    pub level_blend: f32,
    /// **The marking dispatch's stride, in screen pixels** (P27.5) — one thread
    /// per `stride × stride` block instead of one per pixel.
    ///
    /// `1` is the shipped default and is **bit-identical** to the P27.1–P27.4
    /// pass: the shader indexes `gid.xy * stride`, which at 1 is `gid.xy`.
    /// [`RenderTier::apply`](crate::caps::RenderTier::apply) raises it to
    /// [`VSM_MARK_STRIDE_MEDIUM`] below High — a `max`, because a larger stride
    /// is less work, so the clamp still only ever lowers quality.
    pub mark_stride: u32,
    /// **The receiver's PCF kernel radius, in page texels** (P27.5) — a
    /// `(2R+1)²` grid of `textureLoad`s over one table resolution.
    ///
    /// `1` is the shipped default and is
    /// [`VSM_PCF_RADIUS`](crate::vsm_receiver::VSM_PCF_RADIUS) — the cascade's
    /// own 3 × 3 — so the default path is bit-identical to P27.4's. Clamped
    /// **down** to [`VSM_PCF_RADIUS_MEDIUM`] below High.
    ///
    /// The slope bias is a function of it
    /// ([`vsm_slope_bias_texels`](crate::vsm_receiver::vsm_slope_bias_texels))
    /// and travels to the shader in the same uniform, so the two cannot
    /// disagree about which kernel is running.
    pub pcf_radius: u32,
}

/// A [`VsmSettings`] value outside the legal set — **the settings boundary's own
/// refusal** (P27.2).
///
/// # Why this type exists, and what it is defence in depth *for*
///
/// Until P27.2 the only door was per light: `VsmLightDesc::clipmap` clamped a
/// level count, `VsmLevelDesc::page_count` saturated a page grid, and
/// `VsmSystem::for_scene` truncated the light list with a `tracing::warn`. Every
/// one of those is a **recovery**, and the P27.1 audit found what a recovery costs
/// when it is the only door: `clipmap_pages_per_side = 65 536` is `2³²` pages, the
/// multiply wrapped to **zero** in release, the descriptor validated (a square
/// multiple of four is a legal clipmap grid), `register_light` compared 0 against
/// its ceiling and *accepted*, and the relayout then walked four billion addresses
/// into a table with no entry words at all.
///
/// The saturating counters closed that hole from inside. This closes it from
/// outside, so a host learns its configuration is illegal **when it sets it**
/// rather than one frame later in a log line — and the two are deliberately
/// redundant, because the settings struct is `Copy` and a caller may still
/// construct one by hand and hand it to a lower layer.
/// `PartialEq` but **not `Eq`**: [`Distance`](VsmSettingsError::Distance) carries
/// the offending `f32`, and one of the values it exists to reject is `NaN`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VsmSettingsError {
    /// A level count outside `1..=inf_vsm::MAX_VSM_LEVELS`.
    Levels {
        /// Which setting: `"clipmap_levels"`, `"spot_levels"`, `"point_levels"`.
        field: &'static str,
        levels: u32,
        max: usize,
    },
    /// A clipmap page grid that is zero, or not the multiple of four the
    /// concentric `N/4 + x/2` parent rule needs.
    ClipmapGrid { pages_per_side: u32 },
    /// A light's whole page space past [`inf_vsm::MAX_VSM_PAGES_PER_LIGHT`] — the
    /// ceiling `register_light` enforces, restated here so it is refused before an
    /// allocation is attempted rather than after.
    PageSpace {
        field: &'static str,
        pages: u64,
        max: u32,
    },
    /// A page budget that does not hold one page, so every want would be deferred
    /// and the atlas would be empty for ever.
    Budget { budget_bytes: u64, page_bytes: u64 },
    /// A distance that is not a positive finite number of metres.
    Distance { field: &'static str, value: f32 },
    /// A [`VsmSettings::level_blend`] that is not a fraction — zero is legal
    /// (the hard switch), a NaN is not.
    Blend { value: f32 },
    /// A [`VsmSettings::mark_stride`] of zero (a dispatch that covers nothing)
    /// or past [`VSM_MAX_MARK_STRIDE`] (P27.5).
    MarkStride { stride: u32, max: u32 },
    /// A [`VsmSettings::pcf_radius`] past [`VSM_MAX_PCF_RADIUS`] (P27.5). Zero
    /// is legal — it is the single-tap kernel the tier below High runs.
    PcfRadius { radius: u32, max: u32 },
}

impl std::fmt::Display for VsmSettingsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Levels { field, levels, max } => write!(
                f,
                "`{field}` is {levels}; a light's page tree needs 1..={max} levels \
                 (a packed indirection entry carries the level in 8 bits)"
            ),
            Self::ClipmapGrid { pages_per_side } => write!(
                f,
                "`clipmap_pages_per_side` is {pages_per_side}; a clipmap level must \
                 be a non-zero multiple of 4, or the concentric parent rule \
                 `N/4 + x/2` lands between pages"
            ),
            Self::PageSpace { field, pages, max } => write!(
                f,
                "`{field}` gives a light {pages} virtual pages and the ceiling is \
                 {max}; the indirection block and the residency vector are both \
                 linear in it, so an unbounded tree is an unbounded allocation at \
                 registration"
            ),
            Self::Budget {
                budget_bytes,
                page_bytes,
            } => write!(
                f,
                "`budget_bytes` is {budget_bytes} and one shadow page costs \
                 {page_bytes}; the atlas would have no slots and every page request \
                 would be deferred"
            ),
            Self::Distance { field, value } => {
                write!(
                    f,
                    "`{field}` is {value}; it must be a positive finite metre"
                )
            }
            Self::Blend { value } => write!(
                f,
                "`level_blend` is {value}; it must be a fraction in 0..=1 (0 is \
                 the hard level switch)"
            ),
            Self::MarkStride { stride, max } => write!(
                f,
                "`mark_stride` is {stride}; it must be 1..={max} (1 = one \
                 marking thread per screen pixel, and a stride of 0 would \
                 dispatch a grid that covers no pixel at all)"
            ),
            Self::PcfRadius { radius, max } => write!(
                f,
                "`pcf_radius` is {radius}; it must be 0..={max} (0 = a single \
                 tap, and past the ceiling the clamped kernel's dropped-weight \
                 error at a page edge stops being a correction)"
            ),
        }
    }
}

impl std::error::Error for VsmSettingsError {}

impl VsmSettings {
    /// **The settings boundary.** `Ok` iff every light tree these settings would
    /// build is one `inf_vsm` can hold, and every distance is a real metre.
    ///
    /// Checked **through the arithmetic the descriptors actually use** rather than
    /// beside it: the page counts below are the products
    /// `VsmLightDesc::page_count` computes, taken in `u64` so an illegal grid is
    /// *large* here instead of wrapped, which is precisely the failure mode the
    /// P27.1 audit found. Called by
    /// [`EngineRenderer::set_settings`](crate::EngineRenderer::set_settings), which
    /// refuses rather than clamps.
    ///
    /// `enabled` is deliberately **not** consulted: settings that are illegal while
    /// the feature is off become illegal the moment somebody turns it on, and a
    /// door that only guards when the light is already on is not a door.
    pub fn validate(&self) -> Result<(), VsmSettingsError> {
        let max_levels = inf_vsm::MAX_VSM_LEVELS;
        let ceiling = inf_vsm::MAX_VSM_PAGES_PER_LIGHT;
        for (field, levels) in [
            ("clipmap_levels", self.clipmap_levels),
            ("spot_levels", self.spot_levels),
            ("point_levels", self.point_levels),
        ] {
            if levels == 0 || levels as usize > max_levels {
                return Err(VsmSettingsError::Levels {
                    field,
                    levels,
                    max: max_levels,
                });
            }
        }
        let n = self.clipmap_pages_per_side;
        if n == 0 || !n.is_multiple_of(4) {
            return Err(VsmSettingsError::ClipmapGrid { pages_per_side: n });
        }
        // The clipmap: every level is the same `n × n` grid, so the whole tree is
        // `levels · n²` — in `u64`, which is the point.
        let clipmap = u64::from(n) * u64::from(n) * u64::from(self.clipmap_levels);
        // A quadtree: `4^(levels-1) + … + 1`, i.e. `(4^levels − 1) / 3`. A cube is
        // six of them. Both computed rather than approximated, so the refusal and
        // `register_light`'s agree about the same number.
        let quadtree = |levels: u32| -> u64 { (4u64.pow(levels) - 1) / 3 };
        for (field, pages) in [
            ("clipmap_pages_per_side", clipmap),
            ("spot_levels", quadtree(self.spot_levels)),
            ("point_levels", 6 * quadtree(self.point_levels)),
        ] {
            if pages > u64::from(ceiling) {
                return Err(VsmSettingsError::PageSpace {
                    field,
                    pages,
                    max: ceiling,
                });
            }
        }
        let page_bytes = inf_vsm::VsmAtlasConfig::default().page_bytes();
        if self.budget_bytes < page_bytes {
            return Err(VsmSettingsError::Budget {
                budget_bytes: self.budget_bytes,
                page_bytes,
            });
        }
        for (field, value) in [
            ("first_level_extent_m", self.first_level_extent_m),
            ("perspective_near_m", self.perspective_near_m),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(VsmSettingsError::Distance { field, value });
            }
        }
        // The blend band is a FRACTION, so zero is legal (the hard switch) and
        // the refusal is only for a value that is not one — a NaN band would
        // reach `vsm_blend_weight`'s comparisons, where every one of them is
        // false and the seam quietly comes back.
        if !(self.level_blend.is_finite() && (0.0..=1.0).contains(&self.level_blend)) {
            return Err(VsmSettingsError::Blend {
                value: self.level_blend,
            });
        }
        // P27.5's two tier knobs, at the same door as everything else. Both are
        // refused **whatever `enabled` says**, for the reason above: a
        // configuration that is illegal while the feature is off becomes illegal
        // the moment somebody turns it on.
        if self.mark_stride == 0 || self.mark_stride > VSM_MAX_MARK_STRIDE {
            return Err(VsmSettingsError::MarkStride {
                stride: self.mark_stride,
                max: VSM_MAX_MARK_STRIDE,
            });
        }
        if self.pcf_radius > VSM_MAX_PCF_RADIUS {
            return Err(VsmSettingsError::PcfRadius {
                radius: self.pcf_radius,
                max: VSM_MAX_PCF_RADIUS,
            });
        }
        Ok(())
    }
}

impl Default for VsmSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            budget_bytes: inf_vsm::DEFAULT_VSM_BUDGET_BYTES,
            clipmap_pages_per_side: 64,
            clipmap_levels: 8,
            first_level_extent_m: 32.0,
            spot_levels: 7,
            point_levels: 6,
            perspective_near_m: 0.05,
            level_blend: 0.1,
            mark_stride: 1,
            pcf_radius: crate::vsm_receiver::VSM_PCF_RADIUS as u32,
        }
    }
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            hdr: true,
            exposure: 1.0,
            dither: true,
            bloom: BloomSettings::default(),
            ssao: SsaoSettings::default(),
            taa: false,
            vgeom: VgeomSettings::default(),
            scatter: ScatterSettings::default(),
            shadows: ShadowSettings::default(),
            gi: GiSettings::default(),
            atmosphere: AtmosphereSettings::default(),
            water: crate::water::WaterSettings::default(),
            vt: VirtualTextureSettings::default(),
            vsm: VsmSettings::default(),
            stream: StreamSettings::default(),
            raytrace: RaytraceSettings::default(),
            tier_override: None,
        }
    }
}

/// **The ray-query shadow experiment's one knob** (P28.5) — default **off**,
/// and there is no path in this tree that turns it on.
///
/// The ROADMAP's clause is explicit that the experiment *"lands behind a
/// default-off setting with a `caps.rs` clamp"* and that *"VSM remains the
/// shipped path on every tier"*. Both halves are structural here rather than
/// documented:
///
/// * nothing inside [`EngineRenderer::render`](crate::EngineRenderer::render)
///   reads this field — `crate::raytrace` has no render-graph node and the
///   experiment's only caller in the tree is its own gate;
/// * [`AdapterCaps::clamp_ray_query`](crate::caps::AdapterCaps::clamp_ray_query)
///   only ever **clears** it, and no [`RenderTier`](crate::caps::RenderTier)
///   or preset assigns it at all — asserted by
///   `no_tier_or_preset_ever_turns_the_experiment_on`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RaytraceSettings {
    /// Trace sun shadows against a per-frame TLAS instead of reading them from
    /// the shadow path.
    ///
    /// **Never true on a shipped frame.** A gate sets it to run the comparison
    /// `docs/memos/p28-5-ray-query.md` records, and turning it on changes what
    /// the *experiment* does, not what the renderer draws.
    pub sun_shadows: bool,
}

impl RenderSettings {
    /// **Divide the unified streaming budget between the three page systems**
    /// (P28.3, clause 2) — the settings-level half of the arbiter.
    ///
    /// Each consumer's own `budget_bytes` is its **request**; this replaces the
    /// three with `inf_stream::arbitrate`'s grants. It is the last step of
    /// [`RenderTier::apply`](crate::caps::RenderTier::apply) and
    /// [`RenderTier::clamp_mobile`](crate::caps::RenderTier::clamp_mobile), so
    /// every host goes through one door, and it obeys the same law they do: a
    /// grant is never larger than its request, so this **only ever lowers**.
    ///
    /// **Floors are zero here, deliberately.** A consumer's mandatory floor is a
    /// fact about *content* — how many textures a level registers, how big an
    /// asset's page 0 is — and a settings struct has none in scope. Refusing a
    /// budget that cannot hold a floor therefore stays where it already lives
    /// and where the numbers are known: `VtError::MandatoryFloorExceedsBudget`
    /// at registration. What the live floors *are* is reported per frame by
    /// [`EngineRenderer::stream_report`](crate::EngineRenderer::stream_report),
    /// which is the arbiter's audit and where `StreamError::FloorExceedsBudget`
    /// has its producer.
    ///
    /// **Idempotent**, and that is what lets it sit at the end of a clamp that
    /// is itself idempotent: arbitrating an already-arbitrated set has requests
    /// equal to the previous grants, whose sum is at most the total, so the
    /// arbiter is an identity on it.
    pub fn arbitrate_budgets(mut self) -> Self {
        // A consumer that is not live asks for nothing, so its share is
        // available to the two that are. `vgeom.enabled` and `vsm.enabled` are
        // real switches; virtual texturing has none, because whether a level
        // has virtual textures is a property of its content (the
        // `VirtualTextureSettings` ruling) — so its request is always its
        // budget, and a textureless level simply never registers a texture.
        let requests = [
            inf_stream::BudgetRequest::want(if self.vgeom.enabled {
                self.vgeom.stream.budget_bytes
            } else {
                0
            }),
            inf_stream::BudgetRequest::want(self.vt.budget_bytes),
            inf_stream::BudgetRequest::want(if self.vsm.enabled {
                self.vsm.budget_bytes
            } else {
                0
            }),
        ];
        // Infallible: every floor is zero, so the only refusal `arbitrate` can
        // make is unreachable from here. `expect` rather than a silent `unwrap`
        // so the day a floor arrives the message names why it could not.
        let grant = inf_stream::arbitrate(self.stream.budget_bytes, &requests)
            .expect("a zero floor fits every budget");
        if self.vgeom.enabled {
            self.vgeom.stream.budget_bytes = grant.get(inf_stream::Consumer::Geometry);
        }
        self.vt.budget_bytes = grant.get(inf_stream::Consumer::Texture);
        if self.vsm.enabled {
            self.vsm.budget_bytes = grant.get(inf_stream::Consumer::Shadow);
        }
        self
    }

    /// Whether a single-sample scene-depth prepass is needed this frame (SSAO
    /// reads it for AO; TAA reprojects against it).
    ///
    /// **P18.1:** vgeom occlusion no longer appears here. The HZB used to seed
    /// from this prepass, which forced a full-res depth-only pass purely to enable
    /// occlusion; it now min-reduces the live MSAA scene depth instead — the same
    /// rasterization the meshlets depth-test against, which is what makes the
    /// occlusion test provably subtractive (see [`crate::passes::vgeom`]).
    /// **P18.4:** SSR appears here — the screen-space reflection march in the lit
    /// passes reads this exact texture, so turning SSR on without the prepass would
    /// march against whatever the last frame left behind.
    pub fn needs_depth_prepass(&self) -> bool {
        self.ssao.enabled || self.taa || (self.gi.enabled && self.gi.ssr)
    }
}

// ── Bloom math ───────────────────────────────────────────────────────────────

/// COD-style soft-knee bloom prefilter: returns the fraction of `brightness`
/// that contributes to bloom. Below `threshold - knee` it is 0; above
/// `threshold + knee` it is ~1; in between a quadratic knee ramps smoothly.
/// Mirrors `soft_knee` in `shaders/bloom.wgsl` exactly.
pub fn soft_knee_factor(brightness: f32, threshold: f32, knee: f32) -> f32 {
    let knee = knee.max(1e-5);
    // Quadratic knee contribution.
    let rq = (brightness - threshold + knee).clamp(0.0, 2.0 * knee);
    let soft = rq * rq / (4.0 * knee + 1e-5);
    let contrib = soft.max(brightness - threshold);
    (contrib / brightness.max(1e-5)).clamp(0.0, 1.0)
}

/// Downsample mip-chain dimensions: each level halves (rounding down, min 1)
/// until `max_levels` is reached or a 1×1 level is produced. Level 0 is
/// **half** the input size (bloom starts at half res).
pub fn mip_chain_sizes(width: u32, height: u32, max_levels: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut w = (width / 2).max(1);
    let mut h = (height / 2).max(1);
    for _ in 0..max_levels {
        out.push((w, h));
        if w == 1 && h == 1 {
            break;
        }
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    out
}

// ── TAA jitter ───────────────────────────────────────────────────────────────

/// The `index`-th value of the Halton low-discrepancy sequence in `base`
/// (`index` is 1-based per the classic definition; index 0 returns 0).
pub fn halton(mut index: u32, base: u32) -> f32 {
    debug_assert!(base >= 2);
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// Sub-pixel camera jitter for TAA frame `frame_index`, in **pixels**, centred
/// on the pixel (each component in `[-0.5, 0.5)`). Uses Halton(2), Halton(3)
/// over a 16-frame cycle (the standard TAA sequence).
pub fn halton_jitter(frame_index: u64) -> [f32; 2] {
    // 1-based index within a 16-sample cycle.
    let i = (frame_index % 16) as u32 + 1;
    [halton(i, 2) - 0.5, halton(i, 3) - 0.5]
}

// ── SSAO kernel ──────────────────────────────────────────────────────────────

/// A tiny deterministic LCG (numerical-recipes constants) → `[0,1)` floats, so
/// the SSAO kernel is identical on every platform (no `rand` dep, testable).
struct Lcg(u32);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        // Top 24 bits → [0,1).
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// A deterministic tangent-space hemisphere kernel of `count` samples for SSAO
/// (each `z >= 0`), with samples accelerating toward the origin (so near
/// occluders dominate). Seeded so the same `count`/`seed` always yields the same
/// kernel — unit-tested for determinism + hemisphere-ness.
pub fn ssao_hemisphere_kernel(count: usize, seed: u32) -> Vec<[f32; 3]> {
    let mut rng = Lcg(seed.max(1));
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        // Uniform-ish direction in the +Z hemisphere.
        let mut v = [
            rng.next_f32() * 2.0 - 1.0,
            rng.next_f32() * 2.0 - 1.0,
            rng.next_f32(), // z in [0,1) → hemisphere
        ];
        // Normalize.
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
        v[0] /= len;
        v[1] /= len;
        v[2] /= len;
        // Random magnitude, accelerated (t² lerp) so samples cluster near origin.
        let t = i as f32 / count.max(1) as f32;
        let scale = 0.1 + 0.9 * t * t;
        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
        out.push(v);
    }
    out
}

#[cfg(test)]
mod vsm_settings_boundary_tests {
    use super::*;
    use inf_vsm::{VsmLightDesc, VsmResidency, MAX_VSM_LEVELS, MAX_VSM_PAGES_PER_LIGHT};

    /// **The one the P27.1 audit asked for**: the grid that wrapped is refused at
    /// the door, and the refusal is a *type* rather than a log line.
    ///
    /// `65 536²` is exactly `2³²`. Before the saturating counters landed it wrapped
    /// to zero, validated as a legal square multiple of four, registered against a
    /// million-page ceiling as *zero* pages, and then walked four billion addresses
    /// into a table image with no entry words. The counters closed it from inside;
    /// this asserts the outside door refuses it *without needing them* — the
    /// arithmetic here is `u64`, so an illegal grid is large rather than absent.
    #[test]
    fn the_page_grid_that_wrapped_is_refused_at_the_settings_boundary() {
        let bad = VsmSettings {
            clipmap_pages_per_side: 65_536,
            ..Default::default()
        };
        let err = bad.validate().expect_err("2^32 pages was accepted");
        assert!(
            matches!(err, VsmSettingsError::PageSpace { field: "clipmap_pages_per_side", pages, .. }
                if pages == 65_536 * 65_536 * 8),
            "{err:?}"
        );
        // …and the second door the audit named: 40 000 fits per level and does not
        // fit summed over eight, which no per-level check could catch.
        let sum = VsmSettings {
            clipmap_pages_per_side: 40_000,
            ..Default::default()
        };
        assert!(matches!(
            sum.validate(),
            Err(VsmSettingsError::PageSpace { .. })
        ));
        // ANTI-VACUITY, and it is the whole point of a boundary: a grid one step
        // below the ceiling is ACCEPTED, so this is a bound and not a ban.
        let ok = VsmSettings {
            clipmap_pages_per_side: 256,
            clipmap_levels: 16,
            ..Default::default()
        };
        assert_eq!(
            u64::from(ok.clipmap_pages_per_side).pow(2) * u64::from(ok.clipmap_levels),
            u64::from(MAX_VSM_PAGES_PER_LIGHT),
            "the fixture is meant to sit exactly ON the ceiling"
        );
        assert_eq!(ok.validate(), Ok(()));
    }

    /// The refusal and `VsmResidency::register_light`'s agree about **which**
    /// configurations are legal — measured by building the descriptor the settings
    /// name and registering it, rather than by two lists of constants.
    ///
    /// This is what makes the boundary *defence in depth* rather than a second
    /// opinion: anything it lets through must register, and anything it refuses
    /// must be one registration would have had to recover from.
    #[test]
    fn the_boundary_and_the_registration_door_agree() {
        let cases = [
            (VsmSettings::default(), true),
            (
                VsmSettings {
                    clipmap_pages_per_side: 4,
                    clipmap_levels: 1,
                    spot_levels: 1,
                    point_levels: 1,
                    ..Default::default()
                },
                true,
            ),
            (
                VsmSettings {
                    clipmap_pages_per_side: 32,
                    clipmap_levels: 16,
                    spot_levels: 10,
                    point_levels: 9,
                    ..Default::default()
                },
                true,
            ),
            // Illegal: not a multiple of four, so `N/4 + x/2` lands between pages.
            (
                VsmSettings {
                    clipmap_pages_per_side: 6,
                    ..Default::default()
                },
                false,
            ),
            // Illegal: zero levels is not a tree.
            (
                VsmSettings {
                    spot_levels: 0,
                    ..Default::default()
                },
                false,
            ),
            // Illegal: past the 8-bit level field.
            (
                VsmSettings {
                    point_levels: MAX_VSM_LEVELS as u32 + 1,
                    ..Default::default()
                },
                false,
            ),
        ];
        let mut accepted = 0;
        let mut refused = 0;
        for (s, legal) in cases {
            assert_eq!(s.validate().is_ok(), legal, "{s:?}");
            if !legal {
                refused += 1;
                continue;
            }
            accepted += 1;
            // Everything the boundary accepts registers, with no clamp and no
            // truncation: the descriptor the settings name has exactly the levels
            // and the grid they asked for.
            let (mut res, _) = VsmResidency::new(inf_vsm::VsmAtlasConfig {
                budget_bytes: s.budget_bytes,
                ..Default::default()
            });
            for desc in [
                VsmLightDesc::clipmap(s.clipmap_levels, s.clipmap_pages_per_side),
                VsmLightDesc::quadtree(s.spot_levels),
                VsmLightDesc::cube(s.point_levels),
            ] {
                let handle = res
                    .register_light(desc.clone())
                    .expect("the boundary accepted a descriptor registration refuses");
                assert_eq!(res.desc(handle), Some(&desc));
            }
            assert_eq!(
                res.desc(inf_vsm::VsmLightHandle(0))
                    .expect("registered")
                    .level_count(),
                s.clipmap_levels,
                "a level count was clamped on the way in"
            );
        }
        assert!(
            accepted >= 3 && refused >= 3,
            "the sweep collapsed: {accepted} accepted, {refused} refused"
        );
    }

    /// A budget that cannot hold one page, and a distance that is not a distance.
    #[test]
    fn a_budget_below_one_page_and_a_nonsense_metre_are_refused() {
        let page = inf_vsm::VsmAtlasConfig::default().page_bytes();
        assert_eq!(page, 64 * 1024, "a 128² Depth32Float page is 64 KiB");
        assert!(matches!(
            VsmSettings {
                budget_bytes: page - 1,
                ..Default::default()
            }
            .validate(),
            Err(VsmSettingsError::Budget { .. })
        ));
        // …and exactly one page is legal, so the bound is `<` and not `<=`.
        assert_eq!(
            VsmSettings {
                budget_bytes: page,
                ..Default::default()
            }
            .validate(),
            Ok(())
        );
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                matches!(
                    VsmSettings {
                        first_level_extent_m: bad,
                        ..Default::default()
                    }
                    .validate(),
                    Err(VsmSettingsError::Distance {
                        field: "first_level_extent_m",
                        ..
                    })
                ),
                "{bad} was accepted as a clipmap extent"
            );
            assert!(matches!(
                VsmSettings {
                    perspective_near_m: bad,
                    ..Default::default()
                }
                .validate(),
                Err(VsmSettingsError::Distance {
                    field: "perspective_near_m",
                    ..
                })
            ));
        }
    }

    /// **The tier clamps never produce a configuration the boundary refuses.**
    ///
    /// `RenderTier::apply` may only clamp down, and every clamp it applies has to
    /// land on a legal value — a Medium grid of 32 is a multiple of four and a Low
    /// budget of 16 MiB holds 256 pages. Without this arm the two could drift and
    /// the symptom would be a host whose settings are refused *because the tier
    /// clamped them*, which is the worst possible place to discover it.
    #[test]
    fn every_tier_clamp_lands_on_a_legal_configuration() {
        let mut on = RenderSettings::default();
        on.vsm.enabled = true;
        for tier in [
            crate::caps::RenderTier::High,
            crate::caps::RenderTier::Medium,
            crate::caps::RenderTier::Low,
        ] {
            let applied = tier.apply(on);
            assert_eq!(
                applied.vsm.validate(),
                Ok(()),
                "{tier:?} clamped into an illegal configuration"
            );
        }
        assert_eq!(
            crate::caps::RenderTier::mobile_default().vsm.validate(),
            Ok(())
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_post_off_hdr_on() {
        let s = RenderSettings::default();
        assert!(s.hdr && !s.bloom.enabled && !s.ssao.enabled && !s.taa);
        assert!(!s.shadows.enabled && !s.gi.enabled);
        assert_eq!(s.exposure, 1.0);
        assert!(!s.needs_depth_prepass());
        let mut s2 = s;
        s2.ssao.enabled = true;
        assert!(s2.needs_depth_prepass());
        s2.ssao.enabled = false;
        s2.taa = true;
        assert!(s2.needs_depth_prepass());
    }

    /// P18.4 off-path discipline: the new GI knobs must not reach any pass while
    /// GI itself is off — SSR in particular must not conjure a depth prepass for a
    /// scene that never asked for GI.
    #[test]
    fn gi_v2_defaults_are_inert_until_gi_is_enabled() {
        let mut s = RenderSettings::default();
        assert!(!s.gi.enabled);
        assert_eq!(s.gi.quality, crate::gi::GiQuality::High);
        assert_eq!(s.gi.probe_budget, 0, "goldens render at full probe update");
        assert!(s.gi.specular, "the cheap SH specular is the default");
        assert!(!s.gi.ssr, "SSR is opt-in");

        // SSR requested but GI off → no prepass, nothing changes.
        s.gi.ssr = true;
        assert!(!s.needs_depth_prepass());
        // With GI on it forces the prepass it marches against.
        s.gi.enabled = true;
        assert!(s.needs_depth_prepass());

        // The cascade blend defaults on but is inert while shadows are off.
        assert_eq!(RenderSettings::default().shadows.cascade_blend, 0.1);
        assert!(!RenderSettings::default().shadows.enabled);
    }

    #[test]
    fn soft_knee_is_monotonic_and_bounded() {
        let (t, k) = (1.0, 0.5);
        // Below the knee: no contribution.
        assert_eq!(soft_knee_factor(0.2, t, k), 0.0);
        // Well above: nearly full contribution.
        let hi = soft_knee_factor(4.0, t, k);
        assert!(hi > 0.7 && hi <= 1.0, "hi {hi}");
        // Monotonic non-decreasing across the knee.
        let mut prev = -1.0;
        let mut b = 0.0;
        while b <= 3.0 {
            let f = soft_knee_factor(b, t, k);
            assert!((0.0..=1.0).contains(&f), "factor {f} at {b}");
            assert!(f + 1e-4 >= prev, "not monotonic at {b}: {f} < {prev}");
            prev = f;
            b += 0.05;
        }
    }

    #[test]
    fn mip_chain_halves_and_terminates() {
        let sizes = mip_chain_sizes(320, 180, 8);
        assert_eq!(sizes[0], (160, 90)); // half res
                                         // Strictly shrinking, never below 1.
        for w in sizes.windows(2) {
            assert!(w[1].0 <= w[0].0 && w[1].1 <= w[0].1);
            assert!(w[1].0 >= 1 && w[1].1 >= 1);
        }
        // A tiny input yields at least one level.
        assert_eq!(mip_chain_sizes(1, 1, 8), vec![(1, 1)]);
        // max_levels is respected.
        assert_eq!(mip_chain_sizes(4096, 4096, 3).len(), 3);
    }

    #[test]
    fn halton_matches_known_values() {
        // Classic Halton(2): 1/2, 1/4, 3/4, 1/8, 5/8 ...
        assert!((halton(1, 2) - 0.5).abs() < 1e-6);
        assert!((halton(2, 2) - 0.25).abs() < 1e-6);
        assert!((halton(3, 2) - 0.75).abs() < 1e-6);
        // Halton(3): 1/3, 2/3, 1/9 ...
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-6);
        assert!((halton(2, 3) - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn jitter_is_centered_and_cycles() {
        for f in 0..64u64 {
            let [x, y] = halton_jitter(f);
            assert!((-0.5..0.5).contains(&x) && (-0.5..0.5).contains(&y));
        }
        // 16-frame cycle.
        assert_eq!(halton_jitter(0), halton_jitter(16));
        assert_eq!(halton_jitter(5), halton_jitter(21));
        // Not all the same (it actually moves).
        assert_ne!(halton_jitter(0), halton_jitter(1));
    }

    #[test]
    fn ssao_kernel_is_deterministic_and_hemispherical() {
        let a = ssao_hemisphere_kernel(32, 1234);
        let b = ssao_hemisphere_kernel(32, 1234);
        assert_eq!(a, b, "kernel must be deterministic for a fixed seed");
        assert_eq!(a.len(), 32);
        let c = ssao_hemisphere_kernel(32, 9999);
        assert_ne!(a, c, "different seeds → different kernels");
        for s in &a {
            assert!(s[2] >= 0.0, "sample not in +Z hemisphere: {s:?}");
            let len = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
            assert!(len <= 1.0 + 1e-4, "sample outside unit ball: {len}");
        }
    }

    /// **The settings boundary refuses P27.5's two tier knobs out of range**,
    /// and it refuses them **whatever `enabled` says** — the rule the rest of
    /// `validate` already follows, restated here because these two are the first
    /// knobs whose illegal values are *cheap* rather than catastrophic and would
    /// therefore be easy to let through.
    ///
    /// A stride of 0 is the interesting one: `dispatch_workgroups(w / 0)` is a
    /// division by zero on the CPU, and the shader's own `max(stride, 1u)` would
    /// hide it — so the door has to be here, before the arithmetic.
    #[test]
    fn the_settings_boundary_refuses_an_illegal_stride_or_kernel() {
        let ok = VsmSettings::default();
        assert!(ok.validate().is_ok());

        let bad = |f: fn(&mut VsmSettings)| {
            let mut v = ok;
            f(&mut v);
            v.validate().expect_err("an illegal knob was accepted")
        };
        assert!(matches!(
            bad(|v| v.mark_stride = 0),
            VsmSettingsError::MarkStride { stride: 0, .. }
        ));
        assert!(matches!(
            bad(|v| v.mark_stride = VSM_MAX_MARK_STRIDE + 1),
            VsmSettingsError::MarkStride { .. }
        ));
        assert!(matches!(
            bad(|v| v.pcf_radius = VSM_MAX_PCF_RADIUS + 1),
            VsmSettingsError::PcfRadius { .. }
        ));
        // …and the refusal survives the feature being OFF, which is where a door
        // that only guards a live system would let it through.
        let mut off = ok;
        off.enabled = false;
        off.mark_stride = 0;
        assert!(off.validate().is_err());

        // BOUNDS, not bans: exactly ON the ceiling is legal, and a radius of
        // zero — the single tap the tier below High runs — is legal too.
        let mut edge = ok;
        edge.mark_stride = VSM_MAX_MARK_STRIDE;
        edge.pcf_radius = VSM_MAX_PCF_RADIUS;
        assert!(edge.validate().is_ok());
        let mut one_tap = ok;
        one_tap.pcf_radius = 0;
        assert!(one_tap.validate().is_ok());

        // The messages name the field, because a host reads them (P27.2's rule
        // for this enum).
        assert!(bad(|v| v.mark_stride = 0)
            .to_string()
            .contains("mark_stride"));
        assert!(bad(|v| v.pcf_radius = 9).to_string().contains("pcf_radius"));
    }
}
