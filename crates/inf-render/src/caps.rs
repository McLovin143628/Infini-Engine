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
/// shader stage (the vertex-pulling raster group) + 4 in the cull compute, so a
/// High-tier GPU must expose at least this many storage buffers per stage. (The
/// wgpu default limit is 8.)
pub const VGEOM_MIN_STORAGE_BUFFERS_PER_STAGE: u32 = 8;

/// Minimum `max_storage_buffer_binding_size` (bytes) a High-tier GPU must expose
/// so a large meshlet/vertex payload fits in one binding. 128 MiB is the wgpu
/// default.
pub const VGEOM_MIN_STORAGE_BINDING_SIZE: u64 = 128 << 20;

/// Minimum `max_compute_workgroups_per_dimension` for the meshlet cull dispatch
/// (one workgroup per 64 (instance × meshlet) pairs). 65535 is the portable floor.
pub const VGEOM_MIN_WORKGROUPS_PER_DIM: u32 = 65535;

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
            }
            RenderTier::Low => {
                settings.vgeom.enabled = false;
                settings.bloom.enabled = false;
                settings.ssao.enabled = false;
                settings.taa = false;
                settings.shadows.enabled = false;
                settings.gi.enabled = false;
            }
        }
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
        settings.ssao.enabled = false;
        settings.gi.enabled = false;
        settings.taa = false;
        settings.bloom.enabled = false;
        settings.shadows.enabled = false;
        settings
    }
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
    /// Whether the adapter reports itself as a CPU/software rasterizer
    /// (WARP/lavapipe) — never High even if the limits nominally qualify, since
    /// the meshlet path would be unusably slow.
    pub is_cpu: bool,
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
            is_cpu: info.device_type == wgpu::DeviceType::Cpu,
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

/// Detect the tier for a live GPU, honouring an explicit
/// [`RenderSettings::tier_override`]. Logs the decision. This is the seam a host
/// calls once at init; the result feeds [`RenderTier::apply`].
pub fn detect_tier(gpu: &GpuContext, settings: &RenderSettings) -> RenderTier {
    if let Some(forced) = settings.tier_override {
        tracing::info!("inf-render: render tier forced to {forced:?} (override)");
        return forced;
    }
    let caps = AdapterCaps::probe(gpu);
    let tier = choose_tier(&caps);
    let info = gpu.adapter.get_info();
    tracing::info!(
        "inf-render: render tier {tier:?} for '{}' ({:?}, {:?}) — compute={} indirect={} storage_bufs={} storage_bind={}MiB",
        info.name,
        info.backend,
        info.device_type,
        caps.compute_shaders,
        caps.indirect_execution,
        caps.max_storage_buffers_per_stage,
        caps.max_storage_buffer_binding_size >> 20,
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
            is_cpu: false,
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
            is_cpu: false,
        };
        assert_eq!(choose_tier(&c), RenderTier::Low);
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
