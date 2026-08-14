//! GPU-capability detection + render-tier auto-selection (P13.4.2).
//!
//! At renderer/host init the adapter is probed ([`AdapterCaps::probe`]) and a
//! [`RenderTier`] is chosen ([`choose_tier`]) from a handful of portable signals
//! (compute + indirect support, storage-buffer count/size, workgroup limits).
//! The tier maps to a [`RenderSettings`](crate::RenderSettings) profile
//! ([`RenderTier::apply`]) that turns the virtualized-geometry meshlet path (and,
//! on the lowest tier, the expensive post effects) on or off — so a machine that
//! cannot run the GPU-driven meshlet cull automatically falls back to the classic
//! LOD path, and a very constrained GPU also drops bloom/SSAO/TAA/shadows/GI.
//!
//! The **decision is pure** ([`choose_tier`] is a function of [`AdapterCaps`]) so
//! it is unit-tested on synthetic capability sets with no GPU. A caller may force
//! a tier via [`RenderSettings::tier_override`], which bypasses detection (the
//! gate uses it to prove the Low-tier auto-disable of vgeom).

use crate::gpu::GpuContext;
use crate::settings::RenderSettings;

/// The virtualized-geometry meshlet path binds up to 6 storage buffers in one
/// shader stage (the vertex-pulling raster group) + **7 in the cull compute**
/// since P18.1 (meshlets, instances, visible, draw args, the two ping-ponged
/// per-pair visibility buffers, and the audit counters), so a High-tier GPU must
/// expose at least this many storage buffers per stage. (The wgpu default limit
/// is 8.)
pub const VGEOM_MIN_STORAGE_BUFFERS_PER_STAGE: u32 = 8;

/// Storage textures a stage must expose for the P18.1 HZB build: the pyramid's
/// destination mip is a write-only `r32float` storage texture, one at a time.
pub const VGEOM_OCCLUSION_MIN_STORAGE_TEXTURES_PER_STAGE: u32 = 1;

/// Minimum `max_storage_buffer_binding_size` (bytes) a High-tier GPU must expose
/// so a large meshlet/vertex payload fits in one binding. 128 MiB is the wgpu
/// default.
pub const VGEOM_MIN_STORAGE_BINDING_SIZE: u64 = 128 << 20;

/// Minimum `max_compute_workgroups_per_dimension` for the meshlet cull dispatch
/// (one workgroup per 64 (instance × meshlet) pairs). 65535 is the portable floor.
pub const VGEOM_MIN_WORKGROUPS_PER_DIM: u32 = 65535;

/// Storage buffers the P18.5 scatter cull compute binds in one stage: the packed
/// instances, the per-instance slot words, the per-workgroup partial sums, the
/// compacted visible list, the indirect draw args and the audit counters. Six —
/// two under the portable floor of 8, deliberately, so the pass has headroom the
/// meshlet cull no longer does.
pub const SCATTER_MIN_STORAGE_BUFFERS_PER_STAGE: u32 = 6;

/// Minimum `max_compute_workgroups_per_dimension` for the scatter cull dispatch
/// (one workgroup per 256 instances → 65535 workgroups is ~16.7M instances, well
/// past the 1M ROADMAP target).
pub const SCATTER_MIN_WORKGROUPS_PER_DIM: u32 = 65535;

/// The chosen render capability tier. Higher tiers are strict supersets of the
/// features lower tiers enable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderTier {
    /// Full pipeline: the GPU-driven virtualized-geometry meshlet path + all post
    /// effects the caller requested.
    High,
    /// Classic discrete-LOD mesh fallback (no meshlet compute cull) + full
    /// lighting/post. The GPU can raster + light, but lacks the headroom (storage
    /// buffers / compute) the meshlet path needs.
    Medium,
    /// Classic LOD + expensive screen-space/global effects **off** (bloom, SSAO,
    /// TAA, shadows, GI) — a downlevel / very constrained GPU.
    Low,
}

impl RenderTier {
    /// Whether the virtualized-geometry meshlet path is available on this tier.
    pub fn vgeom(self) -> bool {
        matches!(self, RenderTier::High)
    }

    /// Apply the tier to a [`RenderSettings`], turning features off that the tier
    /// cannot afford. **Never turns a feature on** — it only clamps down a
    /// caller's requested settings, so a High tier is a no-op and the byte-stable
    /// defaults are preserved. Returns the clamped settings.
    pub fn apply(self, mut settings: RenderSettings) -> RenderSettings {
        match self {
            RenderTier::High => {}
            RenderTier::Medium => {
                // No meshlet path → the classic LOD fallback renders vgeom content.
                settings.vgeom.enabled = false;
                settings.vgeom.occlusion = false;
                settings.vgeom.two_pass = false;
                // P28.1: and with the meshlet path gone the visibility buffer
                // has nothing to rasterize. Cleared explicitly rather than left
                // to ride on `enabled`, because a clamp that depends on another
                // clamp's ordering is one edit away from being wrong.
                settings.vgeom.visbuffer = false;
                // P18.5: the same shape for scatter — the *mechanism* drops to the
                // CPU-culled classic instanced path, the content still draws, and
                // the bands pull in so a mid GPU pays for less of it.
                settings.scatter.gpu = false;
                settings.scatter.occlusion = false;
                settings.scatter = settings
                    .scatter
                    .clamp_bands(crate::settings::ScatterSettings::MEDIUM_BANDS_M);
            }
            RenderTier::Low => {
                settings.vgeom.enabled = false;
                settings.vgeom.occlusion = false;
                settings.vgeom.two_pass = false;
                settings.vgeom.visbuffer = false;
                settings.scatter.gpu = false;
                settings.scatter.occlusion = false;
                settings.scatter.impostors = false;
                settings.scatter = settings
                    .scatter
                    .clamp_bands(crate::settings::ScatterSettings::LOW_BANDS_M);
                settings.bloom.enabled = false;
                settings.ssao.enabled = false;
                settings.taa = false;
                settings.shadows.enabled = false;
                settings.gi.enabled = false;
            }
        }
        // Water, like the atmosphere, is never turned *off* by a tier — a sea a
        // level authored must still be a sea on a weak GPU — but its tessellation
        // drops and, at Low, screen-space refraction goes with it. `clamp_to` only
        // ever lowers, so a caller that already asked for Low keeps Low.
        settings.water.quality = settings.water.quality.clamp_to(self);
        // The atmosphere is never turned *off* by a tier — a sky the level
        // authored must still be a sky on a weak GPU — but its LUT sizes and
        // march counts scale down (P17.2). `clamp_to` only ever lowers.
        //
        // The same one knob carries the P17.3 clouds: `CloudQuality` is derived
        // from `AtmosphereQuality` rather than authored separately, so a tier that
        // shrinks the LUTs also shrinks the noise volumes, the shadow map and the
        // march budget — and, at Low, drops the erosion volume from the march
        // entirely. Deliberately one knob and not two: a machine that can afford a
        // 256x64 transmittance LUT can afford a 128^3 cloud volume, and letting
        // them disagree would only ever produce combinations nobody tests.
        settings.atmosphere.quality = settings.atmosphere.quality.clamp_to(self);
        // The same shape for GI (P18.4): a tier never turns the *feature* on, and
        // on Low it has already turned it off above — what this does is scale the
        // voxel/probe geometry and the per-frame primitive budget for the tiers
        // that keep it. `clamp_to` only ever lowers, so a High tier is a no-op and
        // the byte-stable default (`GiQuality::High` == the pre-P18.4 geometry) is
        // preserved.
        settings.gi.quality = settings.gi.quality.clamp_to(self);
        // P26.5 — the virtual-texture page pool, and it is the shape of clamp
        // this law was written for. A tier never turns virtual texturing OFF:
        // whether a level has virtual textures is a property of its content, and
        // the analytic floor is bounded per texture and per visible surface, so a
        // smaller pool costs *refinement* and never a hole. What it does is take
        // the `min` — a caller that already asked for less keeps less, High is a
        // no-op, and the operation is idempotent.
        //
        // `bc_tiles` beside it is deliberately NOT touched: BC support is an
        // adapter capability, clamped by `clamp_bc_tiles`, and how much GPU there
        // is has nothing to say about it.
        settings.vt.budget_bytes = settings.vt.budget_bytes.min(match self {
            RenderTier::High => u64::MAX,
            RenderTier::Medium => crate::settings::VT_BUDGET_MEDIUM_BYTES,
            RenderTier::Low => crate::settings::VT_BUDGET_LOW_BYTES,
        });
        // P27.1 — virtual shadow maps, and this is the P27.5 clause ("High/Medium
        // run VSM; Low keeps CSM") wired at the START of the phase rather than at
        // the end of it, so the clamp exists before anything can be shipped that
        // forgets it. Three `min`s and an `&&`: every one of them only ever
        // lowers, so `apply` stays idempotent and order-independent, High is a
        // no-op, and a caller who already asked for less keeps less.
        //
        // The `enabled` clamp is the one the law is about. It **clears** a flag a
        // caller set; it can never set one, which is why it is `&=` against a
        // constant and not an assignment. A Low-tier machine therefore keeps the
        // cascaded shadow map — the path P27.5 demotes rather than deletes —
        // whatever the project's settings say.
        settings.vsm.enabled &= !matches!(self, RenderTier::Low);
        settings.vsm.budget_bytes = settings.vsm.budget_bytes.min(match self {
            RenderTier::High => u64::MAX,
            RenderTier::Medium => crate::settings::VSM_BUDGET_MEDIUM_BYTES,
            RenderTier::Low => crate::settings::VSM_BUDGET_LOW_BYTES,
        });
        settings.vsm.clipmap_pages_per_side = settings.vsm.clipmap_pages_per_side.min(match self {
            RenderTier::High => u32::MAX,
            RenderTier::Medium | RenderTier::Low => crate::settings::VSM_CLIPMAP_PAGES_MEDIUM,
        });
        // P27.5 — the two knobs the phase's tier clause owed beyond the atlas
        // budget, and the pair is worth reading together because they clamp in
        // **opposite arithmetic directions for the same reason**.
        //
        // `mark_stride` is a `max`: a larger stride dispatches fewer threads, so
        // *raising* it lowers the cost, and `max` is therefore the operation that
        // "never enables" is spelled with here. `pcf_radius` is a `min`: a larger
        // radius loads more texels. Both are idempotent, both are no-ops on High,
        // and both keep a caller who already asked for less — which is the half
        // of the law a table of expected values cannot see.
        settings.vsm.mark_stride = settings.vsm.mark_stride.max(match self {
            // 1 is the legal floor `validate` enforces, so this is a no-op by
            // construction rather than by coincidence.
            RenderTier::High => 1,
            RenderTier::Medium | RenderTier::Low => crate::settings::VSM_MARK_STRIDE_MEDIUM,
        });
        settings.vsm.pcf_radius = settings.vsm.pcf_radius.min(match self {
            RenderTier::High => u32::MAX,
            RenderTier::Medium | RenderTier::Low => crate::settings::VSM_PCF_RADIUS_MEDIUM,
        });
        settings
    }

    /// The mobile baseline render profile (P14.1) — the [`RenderSettings`] a
    /// phone / tablet / web player *starts from*, before the live adapter tier is
    /// applied on top. Everything a mobile GPU budget cannot afford is off: the
    /// GPU-driven virtualized-geometry meshlet path, SSAO, GI, TAA, and bloom;
    /// shadows are off too.
    ///
    /// **Honest note on "shadow res 1024":** the cascaded-shadow-map resolution is
    /// a *compile-time* constant ([`crate::csm::SHADOW_RESOLUTION`]), not a
    /// runtime knob, so this preset cannot shrink the shadow map — it turns
    /// shadows **off** on mobile instead (the safe, tile-GPU-friendly choice). A
    /// runtime shadow-resolution setting (so mobile could keep low-res shadows) is
    /// a documented follow-up. Likewise MSAA is a fixed 4× in the renderer, not a
    /// settings field; a mobile 1×/2× MSAA knob is the same follow-up.
    ///
    /// The player applies this when built for `target_os = "android"` or
    /// `target_arch = "wasm32"`, unless a [`RenderSettings::tier_override`] forces
    /// a specific tier.
    pub fn mobile_default() -> RenderSettings {
        Self::clamp_mobile(RenderSettings::default())
    }

    /// Clamp a caller's requested `settings` down to the mobile ceiling. Like
    /// [`apply`](RenderTier::apply) it **only turns features off, never on**, so a
    /// project that ships custom settings still gets a mobile-safe profile.
    pub fn clamp_mobile(mut settings: RenderSettings) -> RenderSettings {
        settings.vgeom.enabled = false;
        settings.vgeom.occlusion = false;
        settings.vgeom.two_pass = false;
        settings.vgeom.visbuffer = false;
        // A phone still draws its foliage — through the CPU-culled fallback, at a
        // third of the distance, with no impostor pass.
        settings.scatter.gpu = false;
        settings.scatter.occlusion = false;
        settings.scatter.impostors = false;
        settings.scatter = settings
            .scatter
            .clamp_bands(crate::settings::ScatterSettings::LOW_BANDS_M);
        settings.ssao.enabled = false;
        settings.gi.enabled = false;
        settings.taa = false;
        settings.bloom.enabled = false;
        settings.shadows.enabled = false;
        // A phone still gets a sky — at the smallest LUTs and the fewest steps.
        settings.atmosphere.quality = settings.atmosphere.quality.clamp_to(RenderTier::Low);
        // A phone still samples its virtual textures — out of the smallest pool
        // (P26.5), through the same door, at a coarser level. This is the tier
        // where the transcode fallback also lands (no BC ⇒ 8× the page bytes),
        // so the two decisions compound and the budget is the one that bounds it.
        settings.vt.budget_bytes = settings
            .vt
            .budget_bytes
            .min(crate::settings::VT_BUDGET_LOW_BYTES);
        settings
    }
}

/// **How densely hair draws on this tier** (P24.4) — the second place a
/// [`RenderTier`] is mapped onto a budget, and the twin of
/// [`crate::debris_budget_for`] down to the reason it lives here.
///
/// The renderer owns the *mapping* because it is the crate that knows what a GPU
/// is; the *value* is applied by the host, which calls `set_hair_detail` on its
/// sim session. `inf-anim` (which owns hair) must stay tier-blind for the same
/// reason `inf-physics` does: a Ring-0 simulation crate that could read a tier is
/// one edit away from putting the machine into `state_bytes`.
///
/// Two numbers rather than a named quality enum, because a projector needs the
/// geometry and a gate needs to measure it:
///
/// * **High** — three interpolated render strands around every guide.
/// * **Medium** — the guides themselves (what P24.4 v1 always drew).
/// * **Low** — **cards**: one strip per three guides, three guides wide, which is
///   a third of the vertices for the same silhouette.
pub fn hair_detail_for(tier: RenderTier) -> HairDetailSpec {
    match tier {
        RenderTier::High => HairDetailSpec {
            guide_stride: 1,
            strands_per_guide: 3,
        },
        RenderTier::Medium => HairDetailSpec {
            guide_stride: 1,
            strands_per_guide: 1,
        },
        RenderTier::Low => HairDetailSpec {
            guide_stride: 3,
            strands_per_guide: 1,
        },
    }
}

/// The hair render budget a tier maps to — the transport for
/// [`hair_detail_for`], carried as plain numbers so this crate never names
/// `inf_anim::HairDetail` (which it cannot: `inf-anim` is a **dev**-dependency
/// here, deliberately, so the shipping renderer links no animation code).
///
/// The host copies the two fields across. That is the same two-type arrangement
/// [`crate::DebrisBudgetSpec`] and `inf_physics::d3::DebrisBudget` already have,
/// and the reason is the same one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HairDetailSpec {
    /// Draw every `n`-th guide, widening it by `n` (`0` is read as `1`).
    pub guide_stride: u8,
    /// Interpolated render strands per drawn guide (`0` is read as `1`).
    pub strands_per_guide: u8,
}

/// The portable subset of adapter capabilities the tier decision reads. Captured
/// as a plain struct so [`choose_tier`] is a pure, GPU-free, unit-testable
/// function (synthetic caps in the tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdapterCaps {
    /// Compute shaders are supported (downlevel `COMPUTE_SHADERS`).
    pub compute_shaders: bool,
    /// Indirect draw/dispatch is supported (downlevel `INDIRECT_EXECUTION`).
    pub indirect_execution: bool,
    /// `max_storage_buffers_per_shader_stage`.
    pub max_storage_buffers_per_stage: u32,
    /// `max_storage_buffer_binding_size` (bytes).
    pub max_storage_buffer_binding_size: u64,
    /// `max_compute_workgroups_per_dimension`.
    pub max_compute_workgroups_per_dim: u32,
    /// `max_storage_textures_per_shader_stage` — the P18.1 HZB build writes the
    /// pyramid mips through a write-only storage texture.
    pub max_storage_textures_per_stage: u32,
    /// Whether the adapter reports itself as a CPU/software rasterizer
    /// (WARP/lavapipe) — never High even if the limits nominally qualify, since
    /// the meshlet path would be unusably slow.
    pub is_cpu: bool,
    /// Whether the adapter exposes `TEXTURE_COMPRESSION_BC` (P26.1) — i.e.
    /// whether a `.inf_tex` tile's BC1/BC3 blocks can be uploaded as-is into a
    /// physical page, with no decode anywhere.
    ///
    /// Independent of the render tier, exactly like
    /// [`polygon_mode_line`](AdapterCaps::polygon_mode_line): it is a *format*
    /// capability, not a measure of how much GPU there is, and a machine can
    /// easily have one without the other (BC is universal on desktop and absent
    /// on most mobile). So [`choose_tier`] ignores it, and the clamp that reads
    /// it is [`clamp_bc_tiles`](AdapterCaps::clamp_bc_tiles).
    pub texture_compression_bc: bool,
    /// Whether the adapter exposes `POLYGON_MODE_LINE` (R-P2 wireframe view mode).
    /// Independent of the render tier — a low-tier GPU may still raster lines, and
    /// a high-tier one may lack the feature — so [`choose_tier`] ignores it; it is
    /// surfaced only so a host can decide whether the wireframe view mode is
    /// offerable before the renderer would otherwise clamp it.
    pub polygon_mode_line: bool,
}

impl AdapterCaps {
    /// Probe a live adapter's capabilities from a [`GpuContext`].
    pub fn probe(gpu: &GpuContext) -> Self {
        let limits = gpu.adapter.limits();
        let downlevel = gpu.adapter.get_downlevel_capabilities();
        let info = gpu.adapter.get_info();
        Self {
            compute_shaders: downlevel
                .flags
                .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS),
            indirect_execution: downlevel
                .flags
                .contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION),
            max_storage_buffers_per_stage: limits.max_storage_buffers_per_shader_stage,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_compute_workgroups_per_dim: limits.max_compute_workgroups_per_dimension,
            max_storage_textures_per_stage: limits.max_storage_textures_per_shader_stage,
            is_cpu: info.device_type == wgpu::DeviceType::Cpu,
            texture_compression_bc: gpu
                .adapter
                .features()
                .contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
            polygon_mode_line: gpu
                .adapter
                .features()
                .contains(wgpu::Features::POLYGON_MODE_LINE),
        }
    }

    /// Whether this adapter can run the virtualized-geometry meshlet path.
    pub fn supports_vgeom(&self) -> bool {
        !self.is_cpu
            && self.compute_shaders
            && self.indirect_execution
            && self.max_storage_buffers_per_stage >= VGEOM_MIN_STORAGE_BUFFERS_PER_STAGE
            && self.max_storage_buffer_binding_size >= VGEOM_MIN_STORAGE_BINDING_SIZE
            && self.max_compute_workgroups_per_dim >= VGEOM_MIN_WORKGROUPS_PER_DIM
    }

    /// Whether this adapter can run the P18.1 **two-pass HZB occlusion** path: the
    /// meshlet path itself, plus a storage texture to write the depth pyramid
    /// into. A strict superset of [`supports_vgeom`](AdapterCaps::supports_vgeom),
    /// so it can only ever be a further restriction — an adapter that fails it
    /// still renders meshlets, just without occlusion culling.
    ///
    /// Callers that turn [`VgeomSettings::occlusion`](crate::VgeomSettings) on
    /// should gate it on this; [`clamp_occlusion`](AdapterCaps::clamp_occlusion)
    /// does it for them.
    pub fn supports_vgeom_occlusion(&self) -> bool {
        self.supports_vgeom()
            && self.max_storage_textures_per_stage >= VGEOM_OCCLUSION_MIN_STORAGE_TEXTURES_PER_STAGE
    }

    /// Clamp the occlusion knobs down to what this adapter supports. Like
    /// [`RenderTier::apply`] it **never turns a feature on**, so it is safe to
    /// compose in any order with the tier clamp.
    pub fn clamp_occlusion(&self, mut settings: RenderSettings) -> RenderSettings {
        if !self.supports_vgeom_occlusion() {
            settings.vgeom.occlusion = false;
            settings.vgeom.two_pass = false;
        }
        settings = self.clamp_scatter(settings);
        settings = self.clamp_bc_tiles(settings);
        settings
    }

    /// Clamp the virtual-texture **page format** to what this adapter supports
    /// (P26.1). Like every other clamp on this type it **never turns a feature
    /// on** — an adapter with BC does not enable `bc_tiles` for a caller who
    /// asked for RGBA8 pages — and it is idempotent, so it composes with the tier
    /// clamp in any order.
    ///
    /// Losing BC costs page *bytes*, never content: a tile is CPU-transcoded to
    /// RGBA8 (`inf_material::TiledTextureReader::tile_rgba8`, the decoder the
    /// thumbnailer has used since P4) and uploaded through the same residency
    /// door, at **8×** the size for BC1 and 4× for BC3 — BC1 is 4 bits per texel
    /// against RGBA8's 32, BC3 is 8. (9 248 B → 73 984 B, and 18 496 B → 73 984 B;
    /// the 8:1 is asserted in `tests/vt_bc_upload.rs`.)
    pub fn clamp_bc_tiles(&self, mut settings: RenderSettings) -> RenderSettings {
        if !self.texture_compression_bc {
            settings.vt.bc_tiles = false;
        }
        settings
    }

    /// Whether this adapter can run the P18.5 **GPU scatter** path: a cull compute
    /// writing indirect draw args, over six storage buffers.
    ///
    /// A far lower bar than [`supports_vgeom`](AdapterCaps::supports_vgeom) — the
    /// meshlet path needs eight storage buffers per stage and 128 MiB bindings, a
    /// scatter cull needs six and a few megabytes — because the content it draws is
    /// ground cover a mid-range GPU is expected to carry. That asymmetry is the
    /// point: `RenderTier::Medium` keeps its foliage on the CPU fallback, but an
    /// adapter that *can* do the compute is not held back by a meshlet floor it has
    /// nothing to do with.
    pub fn supports_scatter_gpu(&self) -> bool {
        self.compute_shaders
            && self.indirect_execution
            && self.max_storage_buffers_per_stage >= SCATTER_MIN_STORAGE_BUFFERS_PER_STAGE
            && self.max_compute_workgroups_per_dim >= SCATTER_MIN_WORKGROUPS_PER_DIM
    }

    /// Clamp the scatter knobs down to what this adapter supports. Like every
    /// other clamp on this type it **never turns a feature on**. Losing the GPU
    /// path costs the impostor band and per-instance occlusion, never the content:
    /// the CPU fallback draws the same batches through the rigid mesh pipeline.
    pub fn clamp_scatter(&self, mut settings: RenderSettings) -> RenderSettings {
        if !self.supports_scatter_gpu() {
            settings.scatter.gpu = false;
        }
        if !settings.scatter.gpu
            || self.max_storage_textures_per_stage < VGEOM_OCCLUSION_MIN_STORAGE_TEXTURES_PER_STAGE
        {
            // The HZB is written through a write-only storage texture, and there is
            // no pyramid to test against without the GPU path in the first place.
            settings.scatter.occlusion = false;
        }
        settings
    }
}

/// Choose the [`RenderTier`] for a set of adapter capabilities. Pure — the
/// unit-tested decision boundary:
///
/// * **High** iff the adapter [`supports_vgeom`](AdapterCaps::supports_vgeom).
/// * **Medium** iff it can still raster with storage buffers (the classic
///   fallback + full lighting) — i.e. at least one storage buffer per stage and
///   not a software rasterizer.
/// * **Low** otherwise (downlevel / software / no storage buffers): classic LOD
///   with the expensive effects off.
pub fn choose_tier(caps: &AdapterCaps) -> RenderTier {
    if caps.supports_vgeom() {
        RenderTier::High
    } else if !caps.is_cpu && caps.max_storage_buffers_per_stage >= 1 {
        RenderTier::Medium
    } else {
        RenderTier::Low
    }
}

/// Probe the adapter and return `settings` clamped down by **both** gates a host
/// has to pass: the render tier ([`RenderTier::apply`]) and the P18.1 occlusion
/// capability floor ([`AdapterCaps::clamp_occlusion`]). Equivalent to
/// `detect_tier(gpu, &s).apply(s)` followed by the occlusion clamp, in one call,
/// so a host cannot apply one and forget the other.
///
/// Only ever turns features **off**, and honours
/// [`RenderSettings::tier_override`] exactly as [`detect_tier`] does — with an
/// override the *tier* is forced but the occlusion clamp still reflects the real
/// adapter (a forced tier is a statement about the tier, not a claim that storage
/// textures exist).
pub fn detect_and_clamp(gpu: &GpuContext, settings: RenderSettings) -> RenderSettings {
    let tier = detect_tier(gpu, &settings);
    AdapterCaps::probe(gpu).clamp_occlusion(tier.apply(settings))
}

/// Detect the tier for a live GPU, honouring an explicit
/// [`RenderSettings::tier_override`]. Logs the decision. This is the seam a host
/// calls once at init; the result feeds [`RenderTier::apply`]. Prefer
/// [`detect_and_clamp`], which also applies the occlusion capability floor.
pub fn detect_tier(gpu: &GpuContext, settings: &RenderSettings) -> RenderTier {
    if let Some(forced) = settings.tier_override {
        tracing::info!("inf-render: render tier forced to {forced:?} (override)");
        return forced;
    }
    let caps = AdapterCaps::probe(gpu);
    let tier = choose_tier(&caps);
    let info = gpu.adapter.get_info();
    tracing::info!(
        "inf-render: render tier {tier:?} for '{}' ({:?}, {:?}) — compute={} indirect={} storage_bufs={} storage_texs={} storage_bind={}MiB occlusion={}",
        info.name,
        info.backend,
        info.device_type,
        caps.compute_shaders,
        caps.indirect_execution,
        caps.max_storage_buffers_per_stage,
        caps.max_storage_textures_per_stage,
        caps.max_storage_buffer_binding_size >> 20,
        caps.supports_vgeom_occlusion(),
    );
    tracing::info!(
        "inf-render: texture_compression_bc={} (absent -> virtual-texture tiles transcode to RGBA8 pages)",
        caps.texture_compression_bc,
    );
    tier
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A capable desktop GPU (the High baseline).
    fn high_caps() -> AdapterCaps {
        AdapterCaps {
            compute_shaders: true,
            indirect_execution: true,
            max_storage_buffers_per_stage: VGEOM_MIN_STORAGE_BUFFERS_PER_STAGE,
            max_storage_buffer_binding_size: VGEOM_MIN_STORAGE_BINDING_SIZE,
            max_compute_workgroups_per_dim: VGEOM_MIN_WORKGROUPS_PER_DIM,
            max_storage_textures_per_stage: VGEOM_OCCLUSION_MIN_STORAGE_TEXTURES_PER_STAGE,
            is_cpu: false,
            texture_compression_bc: true,
            polygon_mode_line: true,
        }
    }

    #[test]
    fn capable_gpu_is_high() {
        assert_eq!(choose_tier(&high_caps()), RenderTier::High);
    }

    #[test]
    fn no_compute_falls_to_medium() {
        let mut c = high_caps();
        c.compute_shaders = false;
        assert_eq!(choose_tier(&c), RenderTier::Medium);
        // No indirect → also Medium (can still raster classically).
        let mut c = high_caps();
        c.indirect_execution = false;
        assert_eq!(choose_tier(&c), RenderTier::Medium);
        // Too few storage buffers for the meshlet raster group → Medium.
        let mut c = high_caps();
        c.max_storage_buffers_per_stage = VGEOM_MIN_STORAGE_BUFFERS_PER_STAGE - 1;
        assert_eq!(choose_tier(&c), RenderTier::Medium);
    }

    #[test]
    fn software_and_downlevel_are_low_or_medium() {
        // A software rasterizer never runs vgeom (even with the limits): Low
        // (is_cpu also disqualifies Medium).
        let mut c = high_caps();
        c.is_cpu = true;
        assert_eq!(choose_tier(&c), RenderTier::Low);
        // A downlevel GPU with no storage buffers at all → Low.
        let c = AdapterCaps {
            compute_shaders: false,
            indirect_execution: false,
            max_storage_buffers_per_stage: 0,
            max_storage_buffer_binding_size: 0,
            max_compute_workgroups_per_dim: 0,
            max_storage_textures_per_stage: 0,
            is_cpu: false,
            texture_compression_bc: false,
            polygon_mode_line: false,
        };
        assert_eq!(choose_tier(&c), RenderTier::Low);
    }

    #[test]
    fn polygon_mode_line_is_orthogonal_to_tier() {
        // Wireframe support (R-P2) is a separate GPU feature from the meshlet-path
        // tier decision: toggling it must never change the chosen tier, in either
        // direction. A High GPU without line raster is still High; a Low GPU with
        // it is still Low.
        let mut high = high_caps();
        high.polygon_mode_line = false;
        assert_eq!(choose_tier(&high), RenderTier::High);

        let low = AdapterCaps {
            compute_shaders: false,
            indirect_execution: false,
            max_storage_buffers_per_stage: 0,
            max_storage_buffer_binding_size: 0,
            max_compute_workgroups_per_dim: 0,
            max_storage_textures_per_stage: 0,
            is_cpu: false,
            texture_compression_bc: false,
            polygon_mode_line: true,
        };
        assert_eq!(choose_tier(&low), RenderTier::Low);
    }

    #[test]
    fn apply_clamps_down_never_up() {
        // High is a no-op: a caller with vgeom on keeps it.
        let mut s = RenderSettings::default();
        s.vgeom.enabled = true;
        s.bloom.enabled = true;
        assert_eq!(RenderTier::High.apply(s), s);

        // Medium disables vgeom, keeps lighting/post.
        let m = RenderTier::Medium.apply(s);
        assert!(!m.vgeom.enabled);
        assert!(m.bloom.enabled, "Medium keeps post effects");

        // Low disables vgeom AND the expensive effects.
        let mut s2 = RenderSettings::default();
        s2.vgeom.enabled = true;
        s2.bloom.enabled = true;
        s2.ssao.enabled = true;
        s2.taa = true;
        s2.shadows.enabled = true;
        s2.gi.enabled = true;
        let l = RenderTier::Low.apply(s2);
        assert!(!l.vgeom.enabled && !l.bloom.enabled && !l.ssao.enabled);
        assert!(!l.taa && !l.shadows.enabled && !l.gi.enabled);
    }

    /// **The P27.5 clamp, wired in P27.1.** Low keeps CSM; nothing ever turns
    /// virtual shadow maps on.
    ///
    /// The arm the clamp law is actually about is the last one: `apply` is fed a
    /// tier for a machine that could run VSM and a settings block with it OFF —
    /// the default — and it must still be off afterwards. A clamp that "enables
    /// on High" would pass every other assertion here.
    #[test]
    fn a_tier_clamps_virtual_shadow_maps_down_and_never_up() {
        use crate::settings::{
            VSM_BUDGET_LOW_BYTES, VSM_BUDGET_MEDIUM_BYTES, VSM_CLIPMAP_PAGES_MEDIUM,
        };
        let mut on = RenderSettings::default();
        on.vsm.enabled = true;

        // High is a no-op on every field.
        assert_eq!(RenderTier::High.apply(on), on);

        // Medium keeps the feature and pulls the numbers in.
        let m = RenderTier::Medium.apply(on);
        assert!(m.vsm.enabled, "Medium runs VSM");
        assert_eq!(m.vsm.budget_bytes, VSM_BUDGET_MEDIUM_BYTES);
        assert_eq!(m.vsm.clipmap_pages_per_side, VSM_CLIPMAP_PAGES_MEDIUM);

        // Low turns it OFF — the clause: CSM is what a Low machine keeps.
        let l = RenderTier::Low.apply(on);
        assert!(!l.vsm.enabled, "Low did not fall back to CSM");
        assert_eq!(l.vsm.budget_bytes, VSM_BUDGET_LOW_BYTES);

        // A caller who already asked for less keeps less, on every tier.
        let mut frugal = RenderSettings::default();
        frugal.vsm.budget_bytes = 1024;
        frugal.vsm.clipmap_pages_per_side = 4;
        for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
            let a = tier.apply(frugal);
            assert_eq!(a.vsm.budget_bytes, 1024, "{tier:?} raised the budget");
            assert_eq!(a.vsm.clipmap_pages_per_side, 4, "{tier:?} raised the grid");
            // …and idempotent: applying twice is applying once.
            assert_eq!(tier.apply(a), a, "{tier:?} is not idempotent");
        }

        // THE LAW: a tier never turns it on. The default is off, and High is the
        // tier that would be tempted.
        let off = RenderSettings::default();
        assert!(!off.vsm.enabled);
        for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
            assert!(
                !tier.apply(off).vsm.enabled,
                "{tier:?} ENABLED virtual shadow maps — `apply` clamps, never enables"
            );
        }
        // …and the mobile preset, which starts from a profile rather than from
        // the defaults, is off too.
        assert!(!RenderTier::mobile_default().vsm.enabled);
    }

    /// **P27.5's two tier knobs, and the pair is only interesting together**:
    /// they clamp in **opposite arithmetic directions** and must both mean "never
    /// raise quality".
    ///
    /// A `max` on the stride and a `min` on the radius look like a bug in a diff
    /// and are the same law: a bigger stride dispatches fewer threads, a bigger
    /// radius loads more texels. So the arm asserts the *cost* falls with the
    /// tier rather than that two numbers move in some direction — which is what
    /// a table of expected values cannot say, and what a future re-tune that
    /// inverts one of them has to fail on.
    #[test]
    fn the_marking_stride_and_the_kernel_radius_clamp_cost_down_on_every_tier() {
        use crate::settings::{VSM_MARK_STRIDE_MEDIUM, VSM_PCF_RADIUS_MEDIUM};
        let s = RenderSettings::default();
        // The shipped default is the P27.1–P27.4 configuration, byte for byte:
        // one thread per pixel and the cascade's own 3 × 3.
        assert_eq!(s.vsm.mark_stride, 1);
        assert_eq!(s.vsm.pcf_radius, crate::vsm_receiver::VSM_PCF_RADIUS as u32);
        // High is a no-op on BOTH, which is the sentence "the default path is
        // unchanged" made checkable.
        assert_eq!(RenderTier::High.apply(s).vsm, s.vsm);

        // The cost model, in the two units that matter: marking threads per
        // pixel and kernel taps per shaded fragment. Both must FALL, and the
        // ladder must really descend or the three tiers are one tier.
        let cost = |t: RenderTier| {
            let v = t.apply(s).vsm;
            let threads = 1.0 / f64::from(v.mark_stride * v.mark_stride);
            let taps = f64::from((2 * v.pcf_radius + 1) * (2 * v.pcf_radius + 1));
            (threads, taps)
        };
        let (ht, hk) = cost(RenderTier::High);
        let (mt, mk) = cost(RenderTier::Medium);
        let (lt, lk) = cost(RenderTier::Low);
        assert!(mt < ht, "Medium marks as densely as High");
        assert!(mk < hk, "Medium filters as widely as High");
        assert!(lt <= mt && lk <= mk, "Low is above Medium");
        // The measured numbers, so the ledger's "a quarter of the threads, one
        // tap instead of nine" is produced by an arm rather than by prose.
        assert_eq!((ht, hk), (1.0, 9.0));
        assert_eq!((mt, mk), (0.25, 1.0));

        // A caller who already asked for LESS keeps it, on every tier — the half
        // of "clamps, never enables" a table of expected values cannot see. A
        // stride of 4 is *less* than Medium's 2, in quality, so it survives.
        let mut frugal = s;
        frugal.vsm.mark_stride = 4;
        frugal.vsm.pcf_radius = 0;
        for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
            let a = tier.apply(frugal);
            assert_eq!(a.vsm.mark_stride, 4, "{tier:?} made the marking DENSER");
            assert_eq!(a.vsm.pcf_radius, 0, "{tier:?} made the kernel WIDER");
            assert_eq!(tier.apply(a).vsm, a.vsm, "{tier:?} is not idempotent");
        }
        // …and the ladder is a real ladder, at the BUILD (the P26.3b precedent):
        // the day someone flattens these the compile fails rather than a test.
        const {
            assert!(VSM_MARK_STRIDE_MEDIUM > 1);
            assert!(VSM_PCF_RADIUS_MEDIUM < crate::vsm_receiver::VSM_PCF_RADIUS as u32);
        }

        // **Everything a tier produces is a configuration the settings boundary
        // accepts** — the P27.2 arm's rule, extended to the two new knobs, so a
        // clamp can never hand `try_set_settings` something it refuses.
        let mut asked = s;
        asked.vsm.enabled = true;
        for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
            tier.apply(asked)
                .vsm
                .validate()
                .unwrap_or_else(|e| panic!("{tier:?} produced an illegal config: {e}"));
        }
    }

    #[test]
    fn mobile_default_disables_the_expensive_features() {
        // The P14.1 preset: no vgeom / SSAO / GI / TAA / bloom / shadows.
        let m = RenderTier::mobile_default();
        assert!(!m.vgeom.enabled);
        assert!(!m.ssao.enabled);
        assert!(!m.gi.enabled);
        assert!(!m.taa);
        assert!(!m.bloom.enabled);
        assert!(!m.shadows.enabled);
        // Still a valid HDR profile (the base render path is unchanged).
        assert!(m.hdr);
    }

    /// P18.4: the GI quality knob follows the atmosphere's rule — clamped down by
    /// the tier, never up, and High is a no-op so the default GI geometry is the
    /// pre-P18.4 one on any capable machine.
    #[test]
    fn tier_clamps_gi_quality_down_only() {
        use crate::gi::GiQuality;
        let s = RenderSettings::default();
        assert_eq!(s.gi.quality, GiQuality::High);
        assert_eq!(RenderTier::High.apply(s).gi.quality, GiQuality::High);
        assert_eq!(RenderTier::Medium.apply(s).gi.quality, GiQuality::Medium);
        // Low turns GI off entirely; the quality is still clamped (a total map).
        let low = RenderTier::Low.apply(s);
        assert!(!low.gi.enabled);
        assert_eq!(low.gi.quality, GiQuality::Low);
        // A caller that already asked for Low keeps Low on a High tier.
        let mut want_low = s;
        want_low.gi.quality = GiQuality::Low;
        assert_eq!(RenderTier::High.apply(want_low).gi.quality, GiQuality::Low);
        // Idempotent.
        let m = RenderTier::Medium.apply(s);
        assert_eq!(RenderTier::Medium.apply(m), m);
    }

    /// **The VT pool budget is a tier knob, and the clamp only ever lowers it**
    /// (P26.5).
    ///
    /// The P26.4 ledger's remainder: *"the pool budget is still
    /// `DEFAULT_VT_BUDGET_BYTES` on both hosts; per-tier knobs are P26.5 (1)."*
    /// The law it lands under is the one the phase's own "Done when" names —
    /// `RenderTier::apply` clamps, never enables — so this arm is a `min` test in
    /// four directions, not a table of expected values.
    #[test]
    fn the_vt_pool_budget_clamps_down_and_never_up() {
        use crate::settings::{VT_BUDGET_LOW_BYTES, VT_BUDGET_MEDIUM_BYTES};
        let s = RenderSettings::default();
        assert_eq!(s.vt.budget_bytes, crate::DEFAULT_VT_BUDGET_BYTES);
        // High is a no-op: the default configuration on a capable machine is the
        // pre-P26.5 pool, byte for byte.
        assert_eq!(
            RenderTier::High.apply(s).vt.budget_bytes,
            crate::DEFAULT_VT_BUDGET_BYTES
        );
        assert_eq!(
            RenderTier::Medium.apply(s).vt.budget_bytes,
            VT_BUDGET_MEDIUM_BYTES
        );
        assert_eq!(
            RenderTier::Low.apply(s).vt.budget_bytes,
            VT_BUDGET_LOW_BYTES
        );
        // …and the ladder really descends, or the three assertions above are one
        // assertion written three times. A `const` block on the P26.3b
        // precedent: clippy is right that this is constant-valued, and its
        // remedy is the stronger one — the day someone flattens two tiers this
        // must fail the BUILD, not a test.
        const {
            assert!(VT_BUDGET_LOW_BYTES < VT_BUDGET_MEDIUM_BYTES);
            assert!(VT_BUDGET_MEDIUM_BYTES < crate::DEFAULT_VT_BUDGET_BYTES);
        }

        // A caller that already asked for LESS than the tier allows keeps its
        // own number — the half of "clamps, never enables" that a table of
        // expected values cannot see.
        let mut frugal = s;
        frugal.vt.budget_bytes = 1024 * 1024;
        for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
            assert_eq!(
                tier.apply(frugal).vt.budget_bytes,
                1024 * 1024,
                "{tier:?} RAISED a caller's pool budget"
            );
        }
        // Idempotent, like every other clamp here.
        let m = RenderTier::Medium.apply(s);
        assert_eq!(RenderTier::Medium.apply(m), m);
        // Mobile takes the Low pool, and `clamp_mobile` never raises either.
        assert_eq!(
            RenderTier::clamp_mobile(s).vt.budget_bytes,
            VT_BUDGET_LOW_BYTES
        );
        assert_eq!(
            RenderTier::clamp_mobile(frugal).vt.budget_bytes,
            1024 * 1024
        );
        // The tier never turns virtual texturing OFF — there is no switch, which
        // is the structural form of the law, and `bc_tiles` (an adapter
        // capability, not a budget) is untouched by every tier.
        for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
            assert!(
                tier.apply(s).vt.bc_tiles,
                "{tier:?} clamped an adapter capability"
            );
            assert!(
                tier.apply(s).vt.budget_bytes > 0,
                "{tier:?} disabled the pool"
            );
        }
    }

    #[test]
    fn clamp_mobile_only_turns_features_off() {
        // A caller that requested everything on gets a mobile-safe profile.
        let mut maxed = RenderSettings::default();
        maxed.vgeom.enabled = true;
        maxed.ssao.enabled = true;
        maxed.gi.enabled = true;
        maxed.taa = true;
        maxed.bloom.enabled = true;
        maxed.shadows.enabled = true;
        let clamped = RenderTier::clamp_mobile(maxed);
        assert!(!clamped.vgeom.enabled && !clamped.ssao.enabled && !clamped.gi.enabled);
        assert!(!clamped.taa && !clamped.bloom.enabled && !clamped.shadows.enabled);
        // Idempotent + equals the preset when applied to defaults.
        assert_eq!(RenderTier::clamp_mobile(clamped), clamped);
        assert_eq!(
            RenderTier::mobile_default(),
            RenderTier::clamp_mobile(RenderSettings::default())
        );
    }

    /// P18.1: the occlusion floor is a strict *further* restriction on the vgeom
    /// floor, and the clamp only ever turns features off.
    #[test]
    fn occlusion_floor_is_a_superset_of_the_vgeom_floor() {
        let high = high_caps();
        assert!(high.supports_vgeom() && high.supports_vgeom_occlusion());

        // Storage buffers fine, no storage textures → meshlets yes, occlusion no.
        let mut no_tex = high_caps();
        no_tex.max_storage_textures_per_stage = 0;
        assert!(no_tex.supports_vgeom(), "meshlets still run");
        assert!(!no_tex.supports_vgeom_occlusion());

        // Anything that fails the vgeom floor fails the occlusion floor too.
        let mut no_compute = high_caps();
        no_compute.compute_shaders = false;
        assert!(!no_compute.supports_vgeom_occlusion());

        let wanted = RenderSettings {
            vgeom: crate::settings::VgeomSettings {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(wanted.vgeom.occlusion && wanted.vgeom.two_pass);
        // Clamping never enables, is idempotent, and leaves the meshlet path alone.
        let c = no_tex.clamp_occlusion(wanted);
        assert!(!c.vgeom.occlusion && !c.vgeom.two_pass);
        assert!(c.vgeom.enabled, "occlusion is not the meshlet path");
        assert_eq!(no_tex.clamp_occlusion(c), c);
        assert_eq!(
            high.clamp_occlusion(wanted),
            wanted,
            "a capable GPU keeps it"
        );
    }

    /// **The hair mapping is monotone and every tier is distinct** (P24.4).
    ///
    /// Distinct because a mapping that answered the same numbers for two tiers
    /// would make "cards for lower tiers" a sentence with no consequence;
    /// monotone because a *lower* tier must never ask for *more* geometry, which
    /// is the whole point of a budget. Both are asserted rather than read off the
    /// literals, so a future re-tune that inverts a pair fails here.
    #[test]
    fn the_hair_detail_falls_with_the_tier() {
        let (high, medium, low) = (
            hair_detail_for(RenderTier::High),
            hair_detail_for(RenderTier::Medium),
            hair_detail_for(RenderTier::Low),
        );
        assert_ne!(high, medium);
        assert_ne!(medium, low);
        // Cost per hairstyle is copies / stride: strictly falling.
        let cost = |d: HairDetailSpec| {
            f32::from(d.strands_per_guide.max(1)) / f32::from(d.guide_stride.max(1))
        };
        assert!(cost(high) > cost(medium), "High must draw more than Medium");
        assert!(cost(medium) > cost(low), "Medium must draw more than Low");
        // Medium is exactly "the guides" — the byte-stable v1 geometry, so a
        // machine in the middle draws what every committed hair test compares.
        assert_eq!(medium.guide_stride, 1);
        assert_eq!(medium.strands_per_guide, 1);
        // Low is a CARD: fewer strips, not fewer particles.
        assert!(low.guide_stride > 1 && low.strands_per_guide == 1);
    }

    /// **P26.1: BC page format is orthogonal to the tier, and the clamp obeys
    /// the two standing laws.**
    ///
    /// Orthogonal because `TEXTURE_COMPRESSION_BC` says what *formats* exist,
    /// not how much GPU there is — a High-tier desktop card without it (a
    /// hypothetical, but the tier decision must not care) is still High, and a
    /// Low-tier one with it is still Low. Asserted in both directions, like
    /// `polygon_mode_line`, because "ignored" is a claim about a function and a
    /// claim is testable.
    #[test]
    fn bc_texture_support_is_orthogonal_to_tier() {
        let mut high = high_caps();
        high.texture_compression_bc = false;
        assert_eq!(choose_tier(&high), RenderTier::High);

        let low = AdapterCaps {
            compute_shaders: false,
            indirect_execution: false,
            max_storage_buffers_per_stage: 0,
            max_storage_buffer_binding_size: 0,
            max_compute_workgroups_per_dim: 0,
            max_storage_textures_per_stage: 0,
            is_cpu: false,
            texture_compression_bc: true,
            polygon_mode_line: false,
        };
        assert_eq!(choose_tier(&low), RenderTier::Low);

        // And no tier touches the knob — a Low machine WITH BC still uploads BC
        // pages, because the reason to drop them is the absent feature, not a
        // small GPU.
        let s = RenderSettings::default();
        assert!(s.vt.bc_tiles, "BC pages are the default");
        for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
            assert!(tier.apply(s).vt.bc_tiles, "{tier:?} clamped a format knob");
        }
        // The mobile preset does not touch it either: a phone without BC is
        // handled by the adapter clamp, and a phone WITH it should use it.
        assert!(RenderTier::clamp_mobile(s).vt.bc_tiles);
    }

    /// The clamp never enables, is idempotent, and composes with the tier clamp
    /// in either order — the three properties every capability clamp here owes.
    #[test]
    fn the_bc_clamp_only_turns_pages_off() {
        let with_bc = high_caps();
        let mut without = high_caps();
        without.texture_compression_bc = false;

        let wanted = RenderSettings::default();
        assert!(wanted.vt.bc_tiles);

        // Absent → off, and it stays off.
        let c = without.clamp_bc_tiles(wanted);
        assert!(!c.vt.bc_tiles);
        assert_eq!(without.clamp_bc_tiles(c), c, "not idempotent");
        // Present → untouched.
        assert_eq!(with_bc.clamp_bc_tiles(wanted), wanted);
        // NEVER turns it on: a caller that asked for RGBA8 pages on a BC-capable
        // adapter keeps RGBA8 pages.
        let mut asked_off = RenderSettings::default();
        asked_off.vt.bc_tiles = false;
        assert!(!with_bc.clamp_bc_tiles(asked_off).vt.bc_tiles);

        // Composes with the tier clamp in either order.
        for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
            assert_eq!(
                without.clamp_bc_tiles(tier.apply(wanted)),
                tier.apply(without.clamp_bc_tiles(wanted)),
                "{tier:?}"
            );
        }
        // And `clamp_occlusion` — the door a host actually calls — carries it,
        // so a host cannot apply one clamp and forget this one.
        assert!(!without.clamp_occlusion(wanted).vt.bc_tiles);
        assert!(with_bc.clamp_occlusion(wanted).vt.bc_tiles);
    }

    #[test]
    fn override_forces_tier_and_disables_vgeom_on_low() {
        // The gate's (d): forcing Low must auto-disable vgeom via apply.
        let mut s = RenderSettings::default();
        s.vgeom.enabled = true;
        s.tier_override = Some(RenderTier::Low);
        let clamped = s.tier_override.unwrap().apply(s);
        assert!(!clamped.vgeom.enabled);
    }
}
