//! Water render parameters (P20.1) — what a projector hands the renderer for one
//! ocean, lake or river, and the GPU records the pass uploads.
//!
//! # The CPU derives, the GPU evaluates
//!
//! The Gerstner *parameters* (direction, wavenumber, amplitude, angular
//! frequency, steepness, phase) are derived once in Ring-0 `inf-water`, in
//! bit-portable `f64`, and shipped here already solved. The shader only sums
//! them. That split is the whole reason there is no second, `f32`, WGSL copy of
//! the wave model to keep in step with the one P20.2's buoyancy will sample — the
//! class of drift the terrain WGSL parity gate exists to catch, avoided by not
//! creating it.
//!
//! It is also why **time never reaches the GPU**. A wave's phase is
//! `φ − ωt`, and `t` is a level clock that can run into the millions of seconds;
//! at `f32` that quantises the phase into visible steps. So the *reduced* phase
//! is computed on the CPU in `f64` and wrapped into `[0, 2π)`
//! ([`inf_water::Wave::phase_at`]) — the same trick `CloudParams::wind_offset`
//! uses for cloud drift, for the same reason.
//!
//! The **floating origin** rides along in that same reduction: the phase a
//! projector uploads already includes `k·(d·origin_xz)`, so the shader evaluates
//! the sum at *render-local* coordinates and still gets the world-space phase,
//! with no large `f32` world coordinate anywhere. A rebase therefore moves no
//! wave.

use glam::{DVec2, DVec3};

pub use inf_water::{RiverPath, RiverProfile, Wave, WaveField, WaveSpec, MAX_WAVES};

/// Which footprint a [`RenderWater`] tessellates. Mirrors
/// `inf_ecs::components::WaterKind`, kept local because `inf-render` must not
/// depend on `inf-ecs` (the same arrangement `RenderLight` has).
///
/// The discriminants are the values written into the water uniform and switched
/// on in `water.wgsl`; the shader's constants mirror them, and
/// `water_kind_codes_match_the_shader` pins the pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WaterKindGpu {
    /// An unbounded plane: tessellated as a camera-following graded grid.
    #[default]
    Ocean,
    /// A bounded rectangle in world XZ: tessellated as a uniform grid.
    Lake,
    /// A spline ribbon: tessellated across [`RenderWater::frames`].
    River,
}

impl WaterKindGpu {
    /// The `u32` the uniform carries and `water.wgsl` switches on.
    pub fn code(self) -> u32 {
        match self {
            WaterKindGpu::Ocean => 0,
            WaterKindGpu::Lake => 1,
            WaterKindGpu::River => 2,
        }
    }
}

/// One sample of a river's centreline, in **world** space (the pass converts to
/// render-local at upload). A dependency-light mirror of
/// [`inf_water::RiverFrame`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterFrame {
    /// World-space centre of the water surface.
    pub center: DVec3,
    /// Unit flow direction.
    pub tangent: DVec3,
    /// Unit horizontal across-vector.
    pub right: DVec3,
    /// Arc length from the start, metres — the wave parameter along the river.
    pub s: f64,
    /// Full width, metres.
    pub width_m: f64,
    /// Depth to the bed, metres.
    pub depth_m: f64,
}

impl From<&inf_water::RiverFrame> for WaterFrame {
    fn from(f: &inf_water::RiverFrame) -> Self {
        Self {
            center: f.center,
            tangent: f.tangent,
            right: f.right,
            s: f.s,
            width_m: f.width_m,
            depth_m: f.depth_m,
        }
    }
}

/// **One water body, ready to draw.** Built identically by both scene projectors
/// (the P20.1 MIRROR gate) and consumed by [`crate::passes::water::WaterNode`].
///
/// World-space `f64` throughout; the pass rebases against the frame's floating
/// origin, exactly as `RenderTerrain` does.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderWater {
    /// Pick id (host-local, like every other renderable's).
    pub id: u32,
    /// Ocean / lake / river.
    pub kind: WaterKindGpu,
    /// Still-water elevation, metres of world Y. For a river this is the
    /// reference level; the surface itself follows [`frames`](Self::frames).
    pub level_m: f64,
    /// Centre of a lake's region in world XZ (unused by the other kinds).
    pub center: DVec2,
    /// Half-extent of a lake's region in world XZ, metres.
    pub half_extent: DVec2,
    /// A river's centreline frames, in flow order. Empty for the other kinds.
    pub frames: Vec<WaterFrame>,
    /// The derived Gerstner components. Evaluated in world XZ for an ocean or a
    /// lake, and in `(arc length, lateral offset)` for a river.
    pub waves: WaveField,
    /// Level clock in seconds — the *document's* clock
    /// (`ResolvedSky::cloud_time_s`), never a wall clock and never a frame
    /// counter. Folded into the uploaded phase, so it never reaches the GPU.
    pub time_s: f64,
    /// Surface flow speed along a river's tangent, m/s. `0` for still bodies.
    pub flow_speed_m_s: f64,
    /// Linear shallow-water colour.
    pub shallow_color: [f32; 3],
    /// Linear deep-water colour.
    pub deep_color: [f32; 3],
    /// Per-channel Beer-Lambert extinction of the water column, m⁻¹.
    pub absorption: [f32; 3],
    /// Specular roughness, `[0, 1]`.
    pub roughness: f32,
    /// Screen-space refraction offset at the water plane, metres.
    pub refraction_m: f32,
    /// Depth over which the surface fades in at the shore, metres.
    pub shore_fade_m: f32,
    /// Maximum surface opacity, `[0, 1]`.
    pub opacity: f32,
    /// Linear foam colour.
    pub foam_color: [f32; 3],
    /// Crest factor above which wave foam appears, `[0, 1]`.
    pub foam_crest_threshold: f32,
    /// Water depth over which shoreline foam fades out, metres.
    pub foam_shore_m: f32,
    /// Flow speed at which a river is fully foamed, m/s.
    pub foam_flow_m_s: f32,
}

impl Default for RenderWater {
    fn default() -> Self {
        Self {
            id: 0,
            kind: WaterKindGpu::Ocean,
            level_m: 0.0,
            center: DVec2::ZERO,
            half_extent: DVec2::splat(50.0),
            frames: Vec::new(),
            waves: WaveField::default(),
            time_s: 0.0,
            flow_speed_m_s: 0.0,
            shallow_color: [0.20, 0.48, 0.50],
            deep_color: [0.015, 0.075, 0.13],
            absorption: [0.45, 0.09, 0.035],
            roughness: 0.04,
            refraction_m: 0.35,
            shore_fade_m: 1.2,
            opacity: 1.0,
            foam_color: [0.92, 0.95, 0.97],
            foam_crest_threshold: 0.65,
            foam_shore_m: 0.5,
            foam_flow_m_s: 4.0,
        }
    }
}

impl RenderWater {
    /// Whether this body has any geometry to draw.
    ///
    /// A river with fewer than two frames has no ribbon — it is an authoring
    /// state (a `WaterKind::River` on an entity with no `Spline`), not an error,
    /// and it must draw nothing rather than a degenerate triangle strip.
    pub fn drawable(&self) -> bool {
        match self.kind {
            WaterKindGpu::Ocean => true,
            WaterKindGpu::Lake => self.half_extent.x > 0.0 && self.half_extent.y > 0.0,
            WaterKindGpu::River => self.frames.len() >= 2,
        }
    }
}

/// Water rendering quality (P20.1) — the one knob a render tier clamps.
///
/// Like [`crate::atmosphere::AtmosphereQuality`] and unlike bloom or SSAO, water
/// has **no enable flag**: whether there is water is a property of the *scene*
/// (does a level carry a `WaterBody`?), not of the renderer, so a setting could
/// only ever disagree with the content. What the renderer owns is how finely to
/// tessellate it and whether to pay for screen-space refraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, PartialOrd, Ord)]
pub enum WaterQuality {
    /// 16×16 quads per body, no refraction — the surface still absorbs, reflects
    /// and foams, it just does not bend what is behind it.
    Low,
    /// 32×32 quads, refraction on.
    Medium,
    /// 64×64 quads, refraction on.
    #[default]
    High,
}

impl WaterQuality {
    /// Grid **vertices** per side (`quads + 1`).
    pub fn grid_vertices(self) -> u32 {
        match self {
            WaterQuality::Low => 17,
            WaterQuality::Medium => 33,
            WaterQuality::High => 65,
        }
    }

    /// Whether this tier pays for the screen-space refraction resolve + sample.
    pub fn refraction(self) -> bool {
        !matches!(self, WaterQuality::Low)
    }

    /// Derive from a render tier. Never raises quality — the same "clamp down
    /// only" contract [`crate::caps::RenderTier::apply`] has.
    pub fn from_tier(tier: crate::caps::RenderTier) -> Self {
        match tier {
            crate::caps::RenderTier::High => WaterQuality::High,
            crate::caps::RenderTier::Medium => WaterQuality::Medium,
            crate::caps::RenderTier::Low => WaterQuality::Low,
        }
    }

    /// Clamp **down** to what `tier` affords, never up.
    pub fn clamp_to(self, tier: crate::caps::RenderTier) -> Self {
        self.min(Self::from_tier(tier))
    }
}

/// Water settings (P20.1). Quality only — see [`WaterQuality`] for why there is
/// no enable flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct WaterSettings {
    pub quality: WaterQuality,
}

/// How far an ocean's camera-following grid reaches, metres.
///
/// A **finite** patch, and the v1 limitation this file is most honest about: past
/// this radius there is no water, so a camera looking at a flat horizon sees the
/// patch end. It is 8 km because that is beyond where the P17 aerial-perspective
/// term has washed the surface into the sky anyway, and because a projected-grid
/// (screen-space, truly infinite) ocean is a different piece of work — named as
/// the follow-up rather than half-built here.
pub const OCEAN_EXTENT_M: f64 = 8_000.0;

/// The ocean grid's centre is snapped to a multiple of this, metres.
///
/// Without it the tessellation slides continuously under the camera and the
/// surface shimmers, because the *surface* is a function of world position (it
/// does not move) while the *samples* of it would be a function of the camera.
/// Snapping makes the vertex set piecewise-constant in camera position, so the
/// shimmer becomes an occasional one-cell jump at the far, sub-pixel end of the
/// grid instead of continuous crawl at the near end.
pub const OCEAN_SNAP_M: f64 = 4.0;

/// Snap `v` down to a multiple of `step` (a pure, branch-free, IEEE-exact
/// operation for the values involved).
#[inline]
pub fn snap(v: f64, step: f64) -> f64 {
    if step <= 0.0 {
        return v;
    }
    (v / step).floor() * step
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::RenderTier;

    #[test]
    fn water_kind_codes_are_frozen() {
        // These are the values `water.wgsl` switches on. Renumbering them would
        // draw every ocean as a lake, silently.
        assert_eq!(WaterKindGpu::Ocean.code(), 0);
        assert_eq!(WaterKindGpu::Lake.code(), 1);
        assert_eq!(WaterKindGpu::River.code(), 2);
        assert_eq!(WaterKindGpu::default(), WaterKindGpu::Ocean);
    }

    #[test]
    fn quality_clamps_down_never_up() {
        for q in [WaterQuality::Low, WaterQuality::Medium, WaterQuality::High] {
            for tier in [RenderTier::High, RenderTier::Medium, RenderTier::Low] {
                let c = q.clamp_to(tier);
                assert!(c <= q, "{q:?} was raised by {tier:?}");
                assert!(c <= WaterQuality::from_tier(tier));
            }
        }
        assert_eq!(
            WaterQuality::Low.clamp_to(RenderTier::High),
            WaterQuality::Low,
            "a High tier must not promote a Low request"
        );
        assert_eq!(
            WaterQuality::High.clamp_to(RenderTier::Low),
            WaterQuality::Low
        );
    }

    #[test]
    fn quality_tiers_are_distinct_and_ordered() {
        let mut prev = 0;
        for q in [WaterQuality::Low, WaterQuality::Medium, WaterQuality::High] {
            let n = q.grid_vertices();
            assert!(n > prev, "{q:?} did not increase the grid");
            assert_eq!(
                (n - 1) % 2,
                0,
                "quad count must stay a power-of-two-ish grid"
            );
            prev = n;
        }
        assert!(!WaterQuality::Low.refraction());
        assert!(WaterQuality::Medium.refraction());
        assert!(WaterQuality::High.refraction());
    }

    #[test]
    fn drawable_rejects_degenerate_bodies() {
        assert!(
            RenderWater::default().drawable(),
            "an ocean is always drawable"
        );

        let lake = RenderWater {
            kind: WaterKindGpu::Lake,
            half_extent: DVec2::ZERO,
            ..Default::default()
        };
        assert!(!lake.drawable());

        // A `River` on an entity with no `Spline` has no centreline — an
        // authoring state, not an error, and it must draw nothing.
        let river = RenderWater {
            kind: WaterKindGpu::River,
            ..Default::default()
        };
        assert!(!river.drawable());
        let one = RenderWater {
            kind: WaterKindGpu::River,
            frames: vec![WaterFrame {
                center: DVec3::ZERO,
                tangent: DVec3::X,
                right: DVec3::Z,
                s: 0.0,
                width_m: 4.0,
                depth_m: 1.0,
            }],
            ..Default::default()
        };
        assert!(!one.drawable(), "one frame is not a ribbon");
    }

    #[test]
    fn snapping_is_monotone_and_lands_on_the_grid() {
        for v in [-13.7_f64, -4.0, -0.1, 0.0, 3.9, 4.0, 4.1, 1234.5] {
            let s = snap(v, OCEAN_SNAP_M);
            assert!(s <= v + 1e-9, "snap must not move forward: {v} -> {s}");
            assert!(v - s < OCEAN_SNAP_M + 1e-9);
            assert!((s / OCEAN_SNAP_M - (s / OCEAN_SNAP_M).round()).abs() < 1e-9);
        }
        // A degenerate step is a no-op rather than a division by zero.
        assert_eq!(snap(7.5, 0.0), 7.5);
    }
}
