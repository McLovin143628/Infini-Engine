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

pub use inf_water::{
    RiverFrame, RiverPath, RiverProfile, WaterSurface, Wave, WaveField, WaveSpec, MAX_WAVES,
};

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

/// The inverse (P20.3): back to the Ring-0 frame, so a render record can be
/// turned into the [`WaterSurface`] the *evaluator* takes. See
/// [`RenderWater::surface`] for why that direction is needed at all.
impl From<&WaterFrame> for inf_water::RiverFrame {
    fn from(f: &WaterFrame) -> Self {
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
    /// Whether a river's authored spline **loops** (P20.3).
    ///
    /// Carried so [`RenderWater::surface`] can hand the Ring-0 evaluator the flag
    /// its [`RiverPath`] was built with instead of guessing `false`. Nothing reads
    /// it today — `RiverPath::sample`, the only thing a height query calls, does
    /// not consult it — which is exactly why dropping it would have been silent
    /// the day something did.
    pub spline_closed: bool,
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
            spline_closed: false,
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

    /// **The Ring-0 surface this record describes** (P20.3) — the bridge back to
    /// [`inf_water::WaterSurface`], and therefore to the *one* height evaluator
    /// the renderer, the cook and the fixed step all share.
    ///
    /// P20.3 needs to answer "is the camera under the water?", and the only
    /// honest way to answer it is to ask the same function P20.2's buoyancy asks
    /// — the one `the_sim_and_the_renderer_derive_the_same_waves` pins. A second,
    /// render-side surface implementation would be a second thing to keep in step
    /// with the sim, and the first frame the *wave model* drifted, the camera
    /// would fog while a swimmer's head was dry.
    ///
    /// # KNOWN DIVERGENCE: visibility
    ///
    /// Sharing the evaluator makes the two sides agree about **what a water
    /// surface is**. It does not make them agree about **which bodies exist**,
    /// and today they do not:
    ///
    /// * the render projectors skip a `WaterBody` on a hidden entity
    ///   (`host.rs`'s `if visible` / the player's mirror of it), so it never
    ///   reaches `RenderScene::waters`;
    /// * `PhysicsBridge3D`'s gather walks **every** `WaterBody` in the world with
    ///   no visibility test at all (`inf-physics/src/d3/ecs.rs`).
    ///
    /// So hiding a lake in the outliner leaves a swimmer swimming in it while the
    /// camera stays dry, and the buoyancy that lifts a boat keeps lifting it.
    /// Which side is wrong is a real design question — visibility is an *editor*
    /// concept and arguably has no business reaching the sim — and P20.3 is a
    /// render-only batch, so it is **named here rather than papered over**. See
    /// the ROADMAP §12 P20.3 ledger.
    ///
    /// The reconstruction is total for oceans and lakes (they carry their whole
    /// geometry). A river's [`RiverPath`] is rebuilt from the frames this record
    /// already carries: `length_m` is the last frame's arc length by
    /// construction, and `closed` is `false` because
    /// [`RiverPath::sample`](inf_water::RiverPath::sample) — the only thing a
    /// height query calls — does not read it. Both facts are pinned by
    /// `a_reconstructed_river_answers_like_the_path_it_came_from`.
    pub fn surface(&self) -> WaterSurface {
        match self.kind {
            WaterKindGpu::Ocean => WaterSurface::Ocean {
                level_m: self.level_m,
                waves: self.waves,
            },
            WaterKindGpu::Lake => WaterSurface::Lake {
                level_m: self.level_m,
                center: self.center,
                half_extent: self.half_extent,
                waves: self.waves,
            },
            WaterKindGpu::River => WaterSurface::River {
                path: RiverPath {
                    frames: self.frames.iter().map(RiverFrame::from).collect(),
                    length_m: self.frames.last().map(|f| f.s).unwrap_or(0.0),
                    closed: self.spline_closed,
                    flow_speed_m_s: self.flow_speed_m_s,
                },
                waves: self.waves,
            },
        }
    }
}

// ── P20.3: the camera under the water ────────────────────────────────────────

/// How deep the camera has to sink before the underwater treatment reaches full
/// strength, metres.
///
/// The v1 submersion model is a **whole-screen** one: the camera is either in the
/// medium or out of it, with no waterline split across the near plane. A bare
/// `eye.y < surface` test would therefore pop the entire frame at the instant a
/// wave crest passed the lens. Ramping over the first 25 cm — about a near-plane
/// height at the default 60° FOV — makes the switch *continuous in camera depth*
/// instead: at the waterline the effect is zero, so the hard switch has nothing
/// to show. It does not make the treatment correct for a half-submerged camera
/// (see the ROADMAP P20.3 ledger); it makes the error small where it is visible.
pub const UNDERWATER_RAMP_M: f64 = 0.25;

/// The column length the underwater fog assumes where the depth buffer holds
/// nothing at all, metres.
///
/// Sky seen straight down a ray that never crosses the surface is not a thing
/// that happens in water; what does happen is a ray that leaves the far plane
/// still in the medium, and 200 m of any authored absorption has saturated long
/// since. Chosen larger than [`OCEAN_EXTENT_M`]/40 so the far field is the deep
/// colour rather than a bright band — the same argument as the surface shader's
/// `OPEN_WATER_DEPTH_M`, one order up because this ray is horizontal.
pub const UNDERWATER_FAR_M: f32 = 200.0;

/// How much of the screen a light shaft may sweep toward the sun, as a fraction
/// of the distance from the shaded pixel to the sun's screen position.
///
/// 1.0 would smear every bright pixel all the way to the sun and read as a
/// radial wipe; 0.55 keeps the shafts short enough to look like beams entering
/// the water rather than a lens flare.
pub const SHAFT_REACH: f32 = 0.55;

/// Per-tap exponential decay along a shaft. `0.965^24 ≈ 0.42`, so the far end of
/// a shaft carries a little under half the weight of its root — a visible taper
/// without the near end blowing out.
pub const SHAFT_DECAY: f32 = 0.965;

/// The lobe exponent of the sun seen from **under** the surface.
///
/// A shaft's *source* is not a point: it is the patch of surface the sun's light
/// enters through, roughened by the waves. `cos^24` is a ≈12° half-width lobe —
/// two orders wider than the sun's own 0.53° disc, which is what makes a beam a
/// beam rather than a specular pinprick. It is also why the shafts are gathered
/// from an **analytic** lobe rather than from the frame's own luminance: from
/// below, the v1 surface shader renders the deep colour, so there would be
/// nothing bright in the frame to gather.
pub const SHAFT_GLOW_POWER: f32 = 24.0;

/// Overall shaft intensity, as a multiplier on the decayed mean of the lobe.
pub const SHAFT_INTENSITY: f32 = 0.35;

/// Sun elevation (as `sin`, i.e. the direction's `y`) at which shafts reach
/// **zero**: 2° below the horizon.
///
/// Not 0°, because a sun at geometric elevation 0 is not gone: atmospheric
/// refraction lifts the disc by ~0.57° and the disc is another ~0.27° in radius,
/// so a *geometrically* set sun is still above the visible horizon for a few more
/// minutes. Two degrees below is where the last of it has gone.
pub const SHAFT_SUN_SET_Y: f32 = -0.0349; // sin(-2°)

/// Sun elevation (as `sin`) at which shafts reach **full** strength: 5° up.
///
/// Between [`SHAFT_SUN_SET_Y`] and here the fade is a smoothstep. The band is
/// deliberately narrow and low: what it exists to prevent is 59 %-strength
/// god-rays at civil twilight, which is what a bare `pow(dot(ray, sun), 24)` lobe
/// produces with the sun 10° *below* the horizon and a ray rising 2° at the same
/// azimuth. Above 5° the sun is properly up and the shafts are the picture.
pub const SHAFT_SUN_FULL_Y: f32 = 0.0872; // sin(5°)

/// How strongly the sun drives light shafts at elevation `sun_y` (the sun
/// direction's `y`, i.e. `sin(elevation)`), `[0, 1]`.
///
/// **The shafts' one time-of-day coupling.** `uw_source` in the shader asks only
/// whether a ray rises to an unoccluded surface and how close it points to the
/// sun; nothing in that question knows whether the sun is *up*. `view.sun_dir`
/// genuinely goes below the horizon (P17.1's clock swings it, and a projector may
/// hand over straight-down), so without this factor a night dive is lit by god
/// rays. Smoothstepped so dusk is a fade rather than a switch.
pub fn shaft_sun_fade(sun_y: f32) -> f32 {
    let span = SHAFT_SUN_FULL_Y - SHAFT_SUN_SET_Y;
    let x = ((sun_y - SHAFT_SUN_SET_Y) / span).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// The column length a shaft's tint is evaluated at, metres.
///
/// A shaft is sunlight that has already crossed some water, so it is coloured by
/// the *same* extinction the fog uses — evaluated at one shaft-length rather
/// than at the pixel's distance, because the light did not travel the path the
/// view ray did. Four metres is a beam's own scale.
pub const SHAFT_TINT_DEPTH_M: f32 = 4.0;

/// **The camera is under the water** — which body, how deep, and how strongly the
/// treatment applies (P20.3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Underwater {
    /// Index into the `waters` slice of the body the camera is inside.
    pub body: usize,
    /// How far the eye is below that body's displaced surface, metres (> 0).
    pub depth_m: f64,
    /// The absolute world elevation of the surface over the eye, metres.
    pub surface_y: f64,
    /// `[0, 1]` — the [`UNDERWATER_RAMP_M`] ramp, smoothstepped.
    pub strength: f32,
}

/// A **conservative, allocation-free** "could this body possibly submerge the
/// eye?" test — the early-out in front of [`camera_underwater`]'s real query.
///
/// Conservative in one direction only: it may say `true` for a body that turns
/// out not to submerge the eye (the exact query then says so), and it must
/// **never** say `false` for one that does. That asymmetry is what makes it safe
/// to drop a rejected body entirely rather than feeding it to
/// [`inf_water::highest_surface`]: the answer is the topmost surface *that
/// exceeds the eye*, and a body whose surface provably cannot exceed the eye can
/// never be it — so dropping it changes neither the winner nor the
/// earliest-wins tie rule (candidates keep projection order).
///
/// The bounds are the still-water level plus the wave field's maximum amplitude
/// — the same bound `the_height_query_stays_near_the_level` pins in Ring 0 — and,
/// for bounded bodies, the footprint.
fn could_submerge(w: &RenderWater, eye: DVec3) -> bool {
    let amp = w.waves.max_amplitude_m();
    match w.kind {
        // Unbounded in XZ: only the height can reject.
        WaterKindGpu::Ocean => eye.y < w.level_m + amp,
        WaterKindGpu::Lake => {
            if eye.y >= w.level_m + amp {
                return false;
            }
            (eye.x - w.center.x).abs() <= w.half_extent.x.max(0.0)
                && (eye.z - w.center.y).abs() <= w.half_extent.y.max(0.0)
        }
        // A river's surface follows its spline, so the bound is the highest frame.
        // O(frames) and allocation-free, where `surface()` is O(frames) *and* a
        // heap allocation — which is the whole point of testing first.
        //
        // **No XZ reject for a river, deliberately.** The obvious one — the
        // bounding box of the frames widened by half a width — is NOT
        // conservative, and `the_cheap_reject_never_drops_a_submerging_body`
        // caught it: [`inf_water::RiverSample::inside`] tests only the *lateral*
        // offset from the centreline, not the longitudinal one, so
        // `RiverPath::sample` clamps to the end segment and reports a point
        // thirty metres beyond the river's mouth as inside its banks. The Ring-0
        // evaluator therefore treats a river as a ribbon extended along its end
        // tangents, and a box around the frames would drop cameras it answers
        // for. The height bound alone is sound, and it is the one that matters
        // for the case this early-out exists for (a camera far above the water).
        WaterKindGpu::River => {
            let top = w
                .frames
                .iter()
                .fold(f64::NEG_INFINITY, |a, f| a.max(f.center.y));
            eye.y < top + amp
        }
    }
}

/// Whether `eye_world` is under any of these bodies, and by how much (P20.3).
///
/// **Reuses the Ring-0 evaluator, deliberately.** The surface height comes from
/// [`RenderWater::surface`] → [`inf_water::WaterSurface::height_at`], and the
/// choice between overlapping bodies from [`inf_water::highest_surface`] — the
/// rule that crate defines once precisely so a renderer, a cook and a physics
/// step cannot each invent their own. Nothing here re-derives a wave.
///
/// Only **drawable** bodies are considered, which is the same filter
/// [`crate::passes::water::WaterNode`] applies. A body that draws nothing (a
/// zero-extent lake, a river with no spline) must not fog a camera either, or
/// `water_off_path_is_byte_identical` would be lying about the off path.
///
/// The clock is the **first body's** `time_s`. Both projectors resolve
/// `inf_ecs::sky::water_environment` once per projection and hand every body the
/// same number, so there is one "now" in a frame; taking it from the list rather
/// than from a wall clock is what keeps this a pure function of the scene.
pub fn camera_underwater(waters: &[RenderWater], eye_world: DVec3) -> Option<Underwater> {
    let mut indices: Vec<usize> = Vec::new();
    let mut surfaces: Vec<WaterSurface> = Vec::new();
    for (i, w) in waters.iter().enumerate() {
        // The cheap reject runs BEFORE `surface()`, which allocates a river's
        // whole centreline. A camera 500 m above a level is not in it, and a
        // level with three rivers should not pay three heap allocations per frame
        // to be told so.
        if w.drawable() && could_submerge(w, eye_world) {
            indices.push(i);
            surfaces.push(w.surface());
        }
    }
    if surfaces.is_empty() {
        return None;
    }
    let t = waters[indices[0]].time_s;
    let p = DVec2::new(eye_world.x, eye_world.z);
    let (k, surface_y) = inf_water::highest_surface(&surfaces, p, t)?;
    let depth_m = surface_y - eye_world.y;
    // NaN-safe on purpose: a degenerate body must read as "not submerged" rather
    // than as an infinitely deep one.
    if depth_m.is_nan() || depth_m <= 0.0 {
        return None;
    }
    // Smoothstep the ramp so the onset has a zero derivative at the waterline.
    let x = (depth_m / UNDERWATER_RAMP_M).clamp(0.0, 1.0);
    Some(Underwater {
        body: indices[k],
        depth_m,
        surface_y,
        strength: (x * x * (3.0 - 2.0 * x)) as f32,
    })
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

    /// Whether this tier pays for the P20.3 underwater **light shafts** — the
    /// 24-tap radial gather toward the sun's screen position.
    ///
    /// Tied to the same tier as refraction rather than to a flag of its own, for
    /// the reason [`WaterQuality`] has no enable flag at all: whether the camera
    /// is underwater is a property of the scene, not of the renderer. What the
    /// renderer owns is whether to pay for the gather. Low still fogs — the
    /// absorption is the *content*; the shafts are the garnish.
    pub fn light_shafts(self) -> bool {
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

    /// The reconstruction claim in [`RenderWater::surface`]'s doc comment, as a
    /// test: a river rebuilt from the frames a `RenderWater` carries answers the
    /// **same height** as the `RiverPath` those frames came from.
    ///
    /// This is what makes it legitimate for P20.3 to ask `inf-water` rather than
    /// to re-derive a surface: if the round trip lost anything, the camera would
    /// be testing against a different river from the one the sim floats boats on.
    #[test]
    fn a_reconstructed_river_answers_like_the_path_it_came_from() {
        let points = [
            DVec3::new(0.0, 20.0, 0.0),
            DVec3::new(60.0, 16.0, 10.0),
            DVec3::new(120.0, 12.0, -10.0),
            DVec3::new(180.0, 8.0, 0.0),
        ];
        let profile = RiverProfile {
            width_start_m: 8.0,
            width_end_m: 14.0,
            depth_start_m: 1.0,
            depth_end_m: 2.5,
            flow_speed_m_s: 1.8,
        };
        let path = RiverPath::from_points(
            &points,
            false,
            inf_math::spline::SplineInterp::CatmullRom,
            &profile,
        );
        let waves = WaveField::from_spec(&WaveSpec::ripple(0.06, 4.0, 5));
        let original = WaterSurface::River {
            path: path.clone(),
            waves,
        };
        let rendered = RenderWater {
            kind: WaterKindGpu::River,
            frames: path.frames.iter().map(WaterFrame::from).collect(),
            flow_speed_m_s: path.flow_speed_m_s,
            waves,
            ..RenderWater::default()
        };
        let rebuilt = rendered.surface();
        // Bit-identical heights, inside the banks and outside them, over a range
        // of clocks — not "close enough".
        let mut inside = 0;
        for i in 0..120 {
            let p = DVec2::new(i as f64 * 1.7 - 20.0, (i % 17) as f64 - 8.0);
            let t = i as f64 * 0.31;
            let a = original.height_at(p, t);
            let b = rebuilt.height_at(p, t);
            assert_eq!(
                a.map(f64::to_bits),
                b.map(f64::to_bits),
                "the reconstructed river disagrees at {p:?}, t = {t}"
            );
            inside += usize::from(a.is_some());
        }
        assert!(
            inside > 10,
            "the probe never landed in the river ({inside} hits) — the comparison \
             is vacuous"
        );
        // …and the arc length really did survive, which is what `length_m` is.
        let WaterSurface::River { path: rp, .. } = &rebuilt else {
            panic!("not a river")
        };
        assert_eq!(rp.length_m, path.length_m);
        assert_eq!(rp.frames.len(), path.frames.len());
        assert!(!rp.closed);

        // ── the SAME river, authored as a LOOP ──
        //
        // A closed spline produces a different frame set (it comes back round), so
        // the height comparison below is a real test of the round trip rather than
        // a re-run of the open case. `closed` itself is forwarded by both
        // projectors and must survive: nothing reads it today —
        // `RiverPath::sample` does not — which is exactly what would make dropping
        // it silent until something did.
        let loop_path = RiverPath::from_points(
            &points,
            true,
            inf_math::spline::SplineInterp::CatmullRom,
            &profile,
        );
        assert_ne!(
            loop_path.frames.len(),
            path.frames.len(),
            "a closed spline must build a different centreline, or this adds nothing"
        );
        let loop_original = WaterSurface::River {
            path: loop_path.clone(),
            waves,
        };
        let loop_rendered = RenderWater {
            kind: WaterKindGpu::River,
            frames: loop_path.frames.iter().map(WaterFrame::from).collect(),
            flow_speed_m_s: loop_path.flow_speed_m_s,
            spline_closed: true,
            waves,
            ..RenderWater::default()
        };
        let loop_rebuilt = loop_rendered.surface();
        let mut loop_inside = 0;
        for i in 0..120 {
            let p = DVec2::new(i as f64 * 1.7 - 20.0, (i % 17) as f64 - 8.0);
            let t = i as f64 * 0.31;
            let a = loop_original.height_at(p, t);
            assert_eq!(
                a.map(f64::to_bits),
                loop_rebuilt.height_at(p, t).map(f64::to_bits),
                "the reconstructed CLOSED river disagrees at {p:?}, t = {t}"
            );
            loop_inside += usize::from(a.is_some());
        }
        assert!(
            loop_inside > 10,
            "the closed probe never landed in the river"
        );
        let WaterSurface::River { path: lp, .. } = &loop_rebuilt else {
            panic!("not a river")
        };
        assert!(lp.closed, "`closed` was dropped by the reconstruction");
        assert_eq!(lp.length_m, loop_path.length_m);
    }

    /// **The sun-elevation fade** — the shafts' one time-of-day coupling.
    ///
    /// `uw_source` asks only whether a ray rises to an unoccluded surface and how
    /// close it points to `view.sun_dir`; nothing in that question knows whether
    /// the sun is *up*. With the sun 10° below the horizon and a ray rising 2° at
    /// the same azimuth, `dot ≈ 0.978` and `pow(0.978, 24) ≈ 0.59` — 59 %-strength
    /// god rays at civil twilight. This is the factor that stops it.
    #[test]
    fn light_shafts_fade_out_as_the_sun_sets() {
        // Full strength with the sun properly up, and the ramp is monotone.
        assert_eq!(shaft_sun_fade(1.0), 1.0);
        assert_eq!(shaft_sun_fade(SHAFT_SUN_FULL_Y), 1.0);
        assert!(shaft_sun_fade(0.5) == 1.0);

        // Zero at and below the set point — this is the assertion the bug had to
        // pass through.
        assert_eq!(shaft_sun_fade(SHAFT_SUN_SET_Y), 0.0);
        assert_eq!(
            shaft_sun_fade((-10f32).to_radians().sin()),
            0.0,
            "the sun 10° below the horizon still drove shafts"
        );
        assert_eq!(
            shaft_sun_fade(-1.0),
            0.0,
            "a straight-down sun drove shafts"
        );

        // The horizon itself is inside the band: a sun at geometric elevation 0 is
        // refracted above the visible horizon and still lights the water.
        let at_horizon = shaft_sun_fade(0.0);
        assert!(
            at_horizon > 0.0 && at_horizon < 1.0,
            "the horizon must be mid-fade, not a switch: {at_horizon}"
        );

        // Monotone non-decreasing across the whole range.
        let mut prev = 0.0;
        for i in 0..=200 {
            let y = -1.0 + i as f32 / 100.0;
            let f = shaft_sun_fade(y);
            assert!((0.0..=1.0).contains(&f), "out of range at {y}: {f}");
            assert!(f >= prev - 1e-6, "not monotone at {y}: {prev} -> {f}");
            prev = f;
        }
    }

    /// The early-out in front of the real query is **conservative**: it may keep a
    /// body that turns out not to submerge the eye, but it must never drop one
    /// that does. A `false` where the exact answer is `Some` is a camera that
    /// stops fogging.
    #[test]
    fn the_cheap_reject_never_drops_a_submerging_body() {
        let ocean = RenderWater {
            level_m: 4.0,
            waves: WaveField::from_spec(&WaveSpec::default()),
            ..RenderWater::default()
        };
        let lake = RenderWater {
            kind: WaterKindGpu::Lake,
            level_m: 10.0,
            center: DVec2::new(100.0, 0.0),
            half_extent: DVec2::new(20.0, 20.0),
            waves: WaveField::from_spec(&WaveSpec::ripple(0.04, 6.0, 11)),
            ..RenderWater::default()
        };
        let river_path = RiverPath::from_points(
            &[
                DVec3::new(0.0, 20.0, 0.0),
                DVec3::new(60.0, 16.0, 10.0),
                DVec3::new(120.0, 12.0, -10.0),
            ],
            false,
            inf_math::spline::SplineInterp::CatmullRom,
            &RiverProfile {
                width_start_m: 8.0,
                width_end_m: 14.0,
                depth_start_m: 1.0,
                depth_end_m: 2.5,
                flow_speed_m_s: 1.8,
            },
        );
        let river = RenderWater {
            kind: WaterKindGpu::River,
            frames: river_path.frames.iter().map(WaterFrame::from).collect(),
            flow_speed_m_s: river_path.flow_speed_m_s,
            waves: WaveField::from_spec(&WaveSpec::ripple(0.06, 4.0, 5)),
            ..RenderWater::default()
        };

        let mut kept_and_submerged = 0;
        let mut rejected = 0;
        for body in [&ocean, &lake, &river] {
            let surface = body.surface();
            for i in 0..400 {
                // A lattice that straddles every footprint and every level.
                let eye = DVec3::new(
                    (i % 20) as f64 * 12.0 - 30.0,
                    (i / 20) as f64 * 1.6 - 4.0,
                    ((i * 7) % 23) as f64 * 8.0 - 40.0,
                );
                let exact = surface
                    .height_at(DVec2::new(eye.x, eye.z), 0.0)
                    .is_some_and(|h| h > eye.y);
                let cheap = could_submerge(body, eye);
                assert!(
                    cheap || !exact,
                    "the cheap reject dropped a {:?} body that submerges the eye at {eye:?}",
                    body.kind
                );
                kept_and_submerged += usize::from(exact);
                rejected += usize::from(!cheap);
            }
        }
        // Both halves must be exercised, or the implication above is vacuous.
        assert!(
            kept_and_submerged > 20,
            "{kept_and_submerged} submerged samples"
        );
        assert!(
            rejected > 20,
            "{rejected} rejected samples — the early-out never fires"
        );

        // …and the query itself still answers the same thing it did before the
        // early-out existed: a camera far above every body is dry.
        assert!(camera_underwater(&[ocean, lake, river], DVec3::new(60.0, 500.0, 0.0)).is_none());
    }

    /// The camera-underwater query: it fires below the surface, not above it;
    /// it respects a lake's footprint; and it ignores bodies that do not draw.
    #[test]
    fn the_camera_knows_when_it_is_under_the_water() {
        let ocean = RenderWater {
            level_m: 4.0,
            waves: WaveField::from_spec(&WaveSpec::default()),
            time_s: 12.5,
            ..RenderWater::default()
        };
        // Well below the deepest trough ⇒ under; well above the highest crest ⇒ not.
        let bound = ocean.waves.max_amplitude_m();
        let deep = camera_underwater(std::slice::from_ref(&ocean), DVec3::new(3.0, -6.0, -2.0))
            .expect("a camera 10 m down is underwater");
        assert_eq!(deep.body, 0);
        assert!(deep.depth_m > 9.0, "{}", deep.depth_m);
        assert_eq!(deep.strength, 1.0, "past the ramp the treatment is full");
        assert!(
            camera_underwater(
                std::slice::from_ref(&ocean),
                DVec3::new(3.0, 4.0 + bound + 1.0, -2.0)
            )
            .is_none(),
            "a camera above every crest is not underwater"
        );

        // The ramp: just under the surface the treatment is present but faint.
        let just_under = camera_underwater(
            std::slice::from_ref(&ocean),
            DVec3::new(3.0, deep.surface_y - UNDERWATER_RAMP_M * 0.25, -2.0),
        )
        .expect("just under the surface");
        assert!(
            just_under.strength > 0.0 && just_under.strength < 0.3,
            "the ramp did not soften the onset: {}",
            just_under.strength
        );

        // A lake is bounded: the same depth outside its rectangle is dry air.
        let lake = RenderWater {
            kind: WaterKindGpu::Lake,
            level_m: 10.0,
            center: DVec2::new(100.0, 0.0),
            half_extent: DVec2::new(20.0, 20.0),
            ..RenderWater::default()
        };
        assert!(
            camera_underwater(std::slice::from_ref(&lake), DVec3::new(100.0, 5.0, 0.0)).is_some()
        );
        assert!(
            camera_underwater(std::slice::from_ref(&lake), DVec3::new(400.0, 5.0, 0.0)).is_none()
        );

        // An UNDRAWABLE body fogs nothing — the same filter the pass applies.
        let flat = RenderWater {
            half_extent: DVec2::ZERO,
            ..lake.clone()
        };
        assert!(!flat.drawable());
        assert!(
            camera_underwater(std::slice::from_ref(&flat), DVec3::new(100.0, 5.0, 0.0)).is_none(),
            "a zero-extent lake submerged the camera"
        );
        assert!(camera_underwater(&[], DVec3::ZERO).is_none());
    }

    /// Overlapping bodies resolve through `inf_water::highest_surface`, so the
    /// **topmost** surface is the one the camera is under — whichever order the
    /// projector listed them in.
    #[test]
    fn the_topmost_body_owns_the_camera() {
        let low = RenderWater {
            kind: WaterKindGpu::Lake,
            level_m: 2.0,
            half_extent: DVec2::splat(50.0),
            waves: WaveField::default(),
            ..RenderWater::default()
        };
        let high = RenderWater {
            level_m: 20.0,
            ..low.clone()
        };
        let eye = DVec3::new(1.0, 1.0, 1.0);
        let a = camera_underwater(&[low.clone(), high.clone()], eye).unwrap();
        let b = camera_underwater(&[high, low], eye).unwrap();
        assert_eq!(a.surface_y, 20.0);
        assert_eq!(b.surface_y, 20.0);
        assert_eq!((a.body, b.body), (1, 0), "the index must follow the list");
        assert_eq!(a.depth_m, b.depth_m);
    }

    #[test]
    fn light_shafts_follow_the_same_tier_as_refraction() {
        for q in [WaterQuality::Low, WaterQuality::Medium, WaterQuality::High] {
            assert_eq!(q.light_shafts(), q.refraction(), "{q:?}");
        }
        assert!(!WaterQuality::Low.light_shafts());
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
