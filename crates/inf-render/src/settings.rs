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
        }
    }
}

impl RenderSettings {
    /// Whether a single-sample scene-depth prepass is needed this frame (SSAO
    /// reads it for AO; TAA reads it for camera reprojection).
    pub fn needs_depth_prepass(&self) -> bool {
        self.ssao.enabled || self.taa
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
mod tests {
    use super::*;

    #[test]
    fn defaults_are_post_off_hdr_on() {
        let s = RenderSettings::default();
        assert!(s.hdr && !s.bloom.enabled && !s.ssao.enabled && !s.taa);
        assert_eq!(s.exposure, 1.0);
        assert!(!s.needs_depth_prepass());
        let mut s2 = s;
        s2.ssao.enabled = true;
        assert!(s2.needs_depth_prepass());
        s2.ssao.enabled = false;
        s2.taa = true;
        assert!(s2.needs_depth_prepass());
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
}
