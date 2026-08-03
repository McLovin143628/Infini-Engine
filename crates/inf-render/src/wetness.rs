//! Shoreline wetness (P20.3) — the band of darkened, smoothed ground a water
//! body leaves on whatever surface crosses its level.
//!
//! # It is content, not a camera effect
//!
//! A wet shoreline is a property of *where the water is*, so every number here is
//! a function of a water body and a fragment's world position and of nothing
//! else. There is no camera in this file, no residency, no frame index and no
//! screen space — the P18.2 law ("camera-driven residency never feeds lighting")
//! applied one phase early to a shading term that would have been very easy to
//! derive from the wrong thing. Move the camera and not one pixel of the wet band
//! moves with it; `wetness_is_a_pure_function_of_the_water` pins exactly that.
//!
//! # No schema, no maps
//!
//! Phase 20 has already spent scene schema v17 and v18, and this needed neither.
//! The band's *shape* comes from the bodies the projectors already publish (an
//! ocean's level, a lake's rectangle, a river's centreline frames) and its
//! *strength* from the engine constants below, whose values are argued in their
//! doc comments rather than authored. Nothing is baked and nothing is persisted:
//! it is evaluated per fragment, in the lit shaders, from a uniform this module
//! packs. P22's material response is where an authored per-material wetness
//! curve belongs, and it will have somewhere to hang it.
//!
//! # The mirror
//!
//! There is no new projector state, which is the strongest form the mirror rule
//! takes: both hosts already publish `RenderScene::waters` through the gated
//! `project_water`, and [`pack_wetness`] is the *only* consumer. Two hosts cannot
//! disagree about a derivation neither of them performs.

use bytemuck::{Pod, Zeroable};
use inf_math::FloatingOrigin;

use crate::water::{RenderWater, WaterKindGpu};

/// Water bodies that may wet a shoreline in one frame.
///
/// Eight rather than the pass's 32: past a handful of bodies the per-fragment
/// loop is the cost, and a level with nine distinct water bodies within one
/// camera's shading footprint is not the shape of thing this v1 is for. Bodies
/// past the eighth are dropped **deterministically**, in projection order, the
/// same rule `MAX_BODIES` applies to drawing.
pub const MAX_WET_BODIES: usize = 8;

/// Centreline segments shared by every river in the frame.
///
/// A river's wet band is evaluated against its polyline, and the polyline is a
/// **decimation** of the built path (a 3-knot Catmull-Rom river carries ~49
/// frames at `DEFAULT_SAMPLES_PER_SEGMENT`). 32 segments over a river is metres
/// of centreline per segment at any authored length, which is finer than the band
/// is wide — and it bounds the inner loop, which is what a per-fragment cost
/// needs above all.
pub const MAX_WET_SEGMENTS: usize = 32;

/// How far **above** the still-water level the wet band reaches, metres.
///
/// The swash zone: capillary rise in a porous substrate is centimetres, but what
/// actually keeps a shoreline dark is the last wave that ran up it, and that is
/// set by the wave height. 0.75 m sits just above the 0.6 m default ocean
/// amplitude, so at the engine's own default sea the band covers the run-up and
/// stops. Everything at or below the level is fully wet — it is submerged.
pub const WET_BAND_M: f32 = 0.75;

/// Albedo multiplier at full wetness.
///
/// Light entering the water film is refracted into the substrate, scattered, and
/// largely re-absorbed before it can escape back through the film — which is why
/// wet sand is dark sand rather than shiny sand. Measured wet/dry albedo ratios
/// for sand, soil and rock land between 0.4 and 0.7; 0.55 is the middle of that
/// and is one number rather than a per-material curve because a per-material
/// curve is P22's job.
pub const WET_ALBEDO_SCALE: f32 = 0.55;

/// Roughness multiplier at full wetness.
///
/// The film fills the microfacet valleys, so the surface the specular lobe sees
/// is the *film's*, not the substrate's. 0.35 takes a typical 0.8-rough ground to
/// ~0.28 — glossy, not mirror — which is where a wet rock reads without the
/// terrain turning into chrome at grazing angles.
pub const WET_ROUGHNESS_SCALE: f32 = 0.35;

/// How far **outside** a bounded body's footprint the band still applies, metres.
///
/// A lake's rectangle and a river's banks describe where the *water* is; the wet
/// ground is by definition just outside them. Without a margin the band would
/// stop dead on the authored polygon and draw a straight edge across a curved
/// shore. Two metres is about one terrain texel at the engine's usual grids —
/// enough to cross the boundary, small enough that it cannot wet a hillside.
pub const WET_SHORE_MARGIN_M: f32 = 2.0;

/// Body kinds as the shader switches on them. Deliberately the same numbering as
/// [`WaterKindGpu::code`] — pinned by `wet_kind_codes_match_the_draw_codes`,
/// because two tables of the same three numbers is exactly how they drift.
const WET_OCEAN: u32 = 0;
const WET_LAKE: u32 = 1;
const WET_RIVER: u32 = 2;

/// One wetting body. Mirrors `struct WetBody` in `wetness.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, PartialEq)]
pub struct WetBodyGpu {
    /// x = kind, y = first segment, z = segment count, w reserved.
    dims: [u32; 4],
    /// xy = render-local centre XZ, zw = half extent XZ (lakes).
    center_extent: [f32; 4],
    /// x = render-local still-water Y, yzw reserved.
    level: [f32; 4],
}

/// One river centreline segment. Mirrors `struct WetSegment` in `wetness.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug, PartialEq)]
pub struct WetSegmentGpu {
    /// xy = start XZ, zw = end XZ (render-local).
    ab: [f32; 4],
    /// x/y = start/end surface Y, z/w = start/end HALF width (m).
    ys: [f32; 4],
}

/// The wetness uniform. Mirrors `struct Wetness` in `wetness.wgsl`; the pair is
/// pinned by `the_uniform_matches_the_shader_struct`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub struct WetnessUniform {
    /// x = body count, yzw reserved.
    dims: [u32; 4],
    /// x = band height (m), y = albedo scale, z = roughness scale,
    /// w = footprint margin (m).
    params: [f32; 4],
    bodies: [WetBodyGpu; MAX_WET_BODIES],
    segments: [WetSegmentGpu; MAX_WET_SEGMENTS],
}

impl Default for WetnessUniform {
    fn default() -> Self {
        Self {
            dims: [0; 4],
            params: [
                WET_BAND_M,
                WET_ALBEDO_SCALE,
                WET_ROUGHNESS_SCALE,
                WET_SHORE_MARGIN_M,
            ],
            bodies: [WetBodyGpu::default(); MAX_WET_BODIES],
            segments: [WetSegmentGpu::default(); MAX_WET_SEGMENTS],
        }
    }
}

impl WetnessUniform {
    /// How many bodies this frame will wet with. `0` ⇒ the shader's `wet_apply`
    /// branch is never taken and every lit pass runs the arithmetic it always did.
    pub fn body_count(&self) -> u32 {
        self.dims[0]
    }
}

/// The GPU-side home of [`WetnessUniform`]: one buffer, created once and never
/// resized.
///
/// **That is load-bearing**, and it is why growing [`crate::passes::EnvBinding`]
/// by this binding did *not* add a fourth component to
/// [`crate::passes::ResourceKey`]. The key exists for resources that get
/// **recreated** — a resize, a quality clamp — because a bind group holding the
/// old view keeps sampling it, silently. A buffer allocated in
/// `EngineRenderer::new` and only ever `write_buffer`-ed cannot go stale, exactly
/// like `frame.shadow.*`. If it ever becomes size- or quality-dependent, it needs
/// a generation and a key component, and the `EnvBinding::bind_group` invariant
/// comment says so.
pub struct WetnessResources {
    pub uniform: wgpu::Buffer,
}

impl WetnessResources {
    pub fn new(gpu: &crate::gpu::GpuContext) -> Self {
        let uniform = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wetness"),
            size: std::mem::size_of::<WetnessUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { uniform }
    }
}

/// Pack the frame's water bodies into the wetness uniform.
///
/// Pure: the same bodies under the same origin always produce the same bytes, and
/// **no camera is involved** — see the module docs. Only `drawable()` bodies
/// participate, which is the same filter [`crate::passes::water::WaterNode`]
/// applies: a body that draws nothing must not darken a shore either.
///
/// Everything is converted to **render-local** through `origin`, because that is
/// the space the lit shaders work in. The band is a *difference* of two Y values
/// in that space, and the rebase is a translation, so the origin cancels: a
/// rebase moves the band no more than it moves a wave.
pub fn pack_wetness(waters: &[RenderWater], origin: &FloatingOrigin) -> WetnessUniform {
    let mut out = WetnessUniform::default();
    let mut segments = 0usize;
    let mut bodies = 0usize;

    for w in waters.iter().filter(|w| w.drawable()) {
        if bodies == MAX_WET_BODIES {
            break;
        }
        let level_local = origin.to_render(glam::DVec3::new(0.0, w.level_m, 0.0)).y;
        let slot = &mut out.bodies[bodies];
        match w.kind {
            WaterKindGpu::Ocean => {
                slot.dims = [WET_OCEAN, 0, 0, 0];
                // An ocean is unbounded, so it has no footprint to carry — and,
                // critically, no camera-following patch centre either. The
                // renderer's ocean *grid* follows the camera; its water level does
                // not, and this is the level.
                slot.center_extent = [0.0; 4];
                slot.level = [level_local, 0.0, 0.0, 0.0];
            }
            WaterKindGpu::Lake => {
                let c = origin.to_render(glam::DVec3::new(w.center.x, w.level_m, w.center.y));
                slot.dims = [WET_LAKE, 0, 0, 0];
                slot.center_extent = [c.x, c.z, w.half_extent.x as f32, w.half_extent.y as f32];
                slot.level = [level_local, 0.0, 0.0, 0.0];
            }
            WaterKindGpu::River => {
                let picks = decimate(w.frames.len(), MAX_WET_SEGMENTS.saturating_sub(segments));
                if picks.len() < 2 {
                    continue;
                }
                slot.dims = [WET_RIVER, segments as u32, (picks.len() - 1) as u32, 0];
                slot.center_extent = [0.0; 4];
                // A river's surface follows its spline, so the body-level `level`
                // is only a fallback the shader never reads for this kind.
                slot.level = [level_local, 0.0, 0.0, 0.0];
                for pair in picks.windows(2) {
                    let a = &w.frames[pair[0]];
                    let b = &w.frames[pair[1]];
                    let (ra, rb) = (origin.to_render(a.center), origin.to_render(b.center));
                    out.segments[segments] = WetSegmentGpu {
                        ab: [ra.x, ra.z, rb.x, rb.z],
                        ys: [
                            ra.y,
                            rb.y,
                            (a.width_m * 0.5) as f32,
                            (b.width_m * 0.5) as f32,
                        ],
                    };
                    segments += 1;
                }
            }
        }
        bodies += 1;
    }

    out.dims[0] = bodies as u32;
    out
}

/// Evenly-spaced indices into a `len`-frame centreline, at most `budget`
/// segments' worth (`budget + 1` picks).
///
/// Integer arithmetic throughout: the decimation of a given river is the same on
/// every machine, so the band it wets is too.
fn decimate(len: usize, budget: usize) -> Vec<usize> {
    if len < 2 || budget == 0 {
        return Vec::new();
    }
    let segs = (len - 1).min(budget);
    (0..=segs).map(|k| k * (len - 1) / segs).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::water::WaterFrame;
    use glam::{DVec2, DVec3};

    fn ocean() -> RenderWater {
        RenderWater::default()
    }

    fn lake() -> RenderWater {
        RenderWater {
            kind: WaterKindGpu::Lake,
            level_m: 5.0,
            center: DVec2::new(60.0, 60.0),
            half_extent: DVec2::new(55.0, 55.0),
            ..RenderWater::default()
        }
    }

    fn river(frames: usize) -> RenderWater {
        RenderWater {
            kind: WaterKindGpu::River,
            frames: (0..frames)
                .map(|i| WaterFrame {
                    center: DVec3::new(i as f64 * 5.0, 9.0 - i as f64 * 0.1, 0.0),
                    tangent: DVec3::X,
                    right: DVec3::Z,
                    s: i as f64 * 5.0,
                    width_m: 6.0 + i as f64 * 0.1,
                    depth_m: 1.5,
                })
                .collect(),
            ..RenderWater::default()
        }
    }

    #[test]
    fn the_uniform_matches_the_shader_struct() {
        // `struct Wetness` in wetness.wgsl, field by field: 2 vec4 header +
        // 8 × (3 vec4) bodies + 32 × (2 vec4) segments. A mismatch is silent on
        // the GPU — the shader reads the wrong 16 bytes for everything after the
        // divergence, and a shoreline drawn from garbage still looks like *a*
        // shoreline.
        assert_eq!(std::mem::size_of::<WetBodyGpu>(), 3 * 16);
        assert_eq!(std::mem::size_of::<WetSegmentGpu>(), 2 * 16);
        assert_eq!(
            std::mem::size_of::<WetnessUniform>(),
            2 * 16 + MAX_WET_BODIES * 48 + MAX_WET_SEGMENTS * 32
        );
        assert_eq!(std::mem::align_of::<WetnessUniform>(), 4);
    }

    #[test]
    fn wet_kind_codes_match_the_draw_codes() {
        // One numbering for "which kind of water is this", not two.
        assert_eq!(WET_OCEAN, WaterKindGpu::Ocean.code());
        assert_eq!(WET_LAKE, WaterKindGpu::Lake.code());
        assert_eq!(WET_RIVER, WaterKindGpu::River.code());
    }

    #[test]
    fn the_default_uniform_wets_nothing_but_still_carries_the_constants() {
        let d = WetnessUniform::default();
        assert_eq!(d.body_count(), 0);
        assert_eq!(
            d.params,
            [
                WET_BAND_M,
                WET_ALBEDO_SCALE,
                WET_ROUGHNESS_SCALE,
                WET_SHORE_MARGIN_M
            ]
        );
        // A water-free scene packs to exactly the default — the off path.
        assert_eq!(pack_wetness(&[], &FloatingOrigin::default()), d);
    }

    /// **The camera-residency law, as a test.** The packed uniform must be a
    /// function of the water and the render origin alone. The ocean is the
    /// dangerous one: its *drawn* patch is snapped to the camera, and reaching for
    /// `center_extent` there is exactly the mistake that would make a shoreline
    /// slide under a moving player.
    #[test]
    fn wetness_is_a_pure_function_of_the_water() {
        let origin = FloatingOrigin::default();
        let bodies = [ocean(), lake(), river(9)];
        let a = pack_wetness(&bodies, &origin);
        let b = pack_wetness(&bodies, &origin);
        assert_eq!(a, b, "packing is not deterministic");
        // The ocean slot carries no footprint at all, so there is nothing for a
        // camera to have been folded into.
        assert_eq!(a.bodies[0].dims[0], WET_OCEAN);
        assert_eq!(a.bodies[0].center_extent, [0.0; 4]);
        assert_eq!(a.bodies[0].level[0], 0.0);
        // The lake's footprint is its AUTHORED region.
        assert_eq!(a.bodies[1].center_extent, [60.0, 60.0, 55.0, 55.0]);
        assert_eq!(a.bodies[1].level[0], 5.0);
    }

    /// A rebase moves the band by exactly the rebase and no more: every packed
    /// coordinate is render-local, so the *difference* the shader evaluates
    /// (fragment Y − level Y) is unchanged.
    #[test]
    fn a_rebase_moves_the_band_with_the_world_and_not_against_it() {
        let a = FloatingOrigin::default();
        let b = FloatingOrigin::new(DVec3::new(5_000.0, 30.0, -3_000.0));
        assert_ne!(a.origin(), b.origin(), "the rebase did nothing");
        let bodies = [lake()];
        let ua = pack_wetness(&bodies, &a);
        let ub = pack_wetness(&bodies, &b);

        // A world point on the lake's shore, in each origin's local space.
        let world = DVec3::new(60.0, 5.4, 60.0);
        let la = a.to_render(world);
        let lb = b.to_render(world);
        let ha = la.y - ua.bodies[0].level[0];
        let hb = lb.y - ub.bodies[0].level[0];
        assert!(
            (ha - hb).abs() < 1e-3,
            "the band height above the level moved with the origin: {ha} vs {hb}"
        );
        // …and so does the footprint test.
        let da = (la.x - ua.bodies[0].center_extent[0]).abs();
        let db = (lb.x - ub.bodies[0].center_extent[0]).abs();
        assert!((da - db).abs() < 1e-2, "{da} vs {db}");
    }

    #[test]
    fn only_drawable_bodies_wet_anything() {
        let origin = FloatingOrigin::default();
        let flat = RenderWater {
            half_extent: DVec2::ZERO,
            ..lake()
        };
        assert!(!flat.drawable());
        assert_eq!(pack_wetness(&[flat], &origin).body_count(), 0);
        // A river with a single frame is the other undrawable shape.
        assert_eq!(pack_wetness(&[river(1)], &origin).body_count(), 0);
        // …and a real one does wet.
        assert_eq!(pack_wetness(&[lake()], &origin).body_count(), 1);
    }

    #[test]
    fn a_river_becomes_a_bounded_polyline() {
        let origin = FloatingOrigin::default();
        // 60 frames ⇒ 59 candidate segments, decimated to the cap.
        let u = pack_wetness(&[river(60)], &origin);
        assert_eq!(u.body_count(), 1);
        let b = u.bodies[0];
        assert_eq!(b.dims[0], WET_RIVER);
        assert_eq!(b.dims[1], 0, "the first river starts at segment 0");
        assert_eq!(b.dims[2] as usize, MAX_WET_SEGMENTS);
        // The polyline spans the whole river, not just its head.
        let first = u.segments[0];
        let last = u.segments[MAX_WET_SEGMENTS - 1];
        assert_eq!(first.ab[0], 0.0);
        assert!(
            (last.ab[2] - 59.0 * 5.0).abs() < 1e-3,
            "the decimation dropped the tail: {}",
            last.ab[2]
        );
        // Half widths, not full widths — the shader compares them to a distance.
        assert!((first.ys[2] - 3.0).abs() < 1e-6, "{}", first.ys[2]);
        // Surface Y follows the spline downhill.
        assert!(last.ys[1] < first.ys[0]);
    }

    #[test]
    fn the_segment_budget_is_shared_and_never_overrun() {
        let origin = FloatingOrigin::default();
        // Three greedy rivers: the first takes the whole budget, the rest get
        // what is left, and nothing writes past the array.
        let u = pack_wetness(&[river(80), river(80), river(80)], &origin);
        let total: usize = (0..u.body_count() as usize)
            .map(|i| u.bodies[i].dims[2] as usize)
            .sum();
        assert!(
            total <= MAX_WET_SEGMENTS,
            "{total} segments packed into {MAX_WET_SEGMENTS} slots"
        );
        for i in 0..u.body_count() as usize {
            let b = u.bodies[i];
            assert!((b.dims[1] + b.dims[2]) as usize <= MAX_WET_SEGMENTS);
        }
    }

    #[test]
    fn the_body_list_is_clamped_deterministically_in_order() {
        let origin = FloatingOrigin::default();
        let many: Vec<RenderWater> = (0..MAX_WET_BODIES + 5)
            .map(|i| RenderWater {
                level_m: i as f64,
                ..lake()
            })
            .collect();
        let u = pack_wetness(&many, &origin);
        assert_eq!(u.body_count() as usize, MAX_WET_BODIES);
        // The FIRST eight, in projection order — not an arbitrary eight.
        for i in 0..MAX_WET_BODIES {
            assert_eq!(u.bodies[i].level[0], i as f32, "slot {i}");
        }
    }

    #[test]
    fn decimation_is_even_and_endpoint_preserving() {
        assert!(decimate(0, 10).is_empty());
        assert!(decimate(1, 10).is_empty());
        assert!(decimate(10, 0).is_empty());
        for len in [2usize, 3, 17, 49, 200] {
            for budget in [1usize, 4, MAX_WET_SEGMENTS] {
                let picks = decimate(len, budget);
                assert_eq!(picks.len(), (len - 1).min(budget) + 1, "{len}/{budget}");
                assert_eq!(picks[0], 0);
                assert_eq!(*picks.last().unwrap(), len - 1);
                assert!(
                    picks.windows(2).all(|w| w[1] > w[0]),
                    "picks must be strictly increasing: {picks:?}"
                );
            }
        }
    }
}
