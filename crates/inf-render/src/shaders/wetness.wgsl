// Shoreline wetness (P20.3) — the shading-time half of `crate::wetness`,
// composed into every lit scene shader by `passes::lit_scene_shader` (after
// env_lighting.wgsl, so it lands in the same `GROUP_ENV` bind group).
//
// ## What it is
//
// A surface near a water body's level is WET: its albedo drops (light that
// enters the film is scattered into the substrate and re-absorbed rather than
// leaving) and its roughness drops (the film fills the microfacet valleys). That
// is the whole model. It is not a decal, not a map and not a texture — it is a
// function of the fragment's world position and of the water bodies in the
// frame, which is what makes it survive a moving camera, a rebase and a reload.
//
// ## Camera-free by construction
//
// Nothing in this file reads `view.eye`, a screen coordinate, a residency set or
// a frame index. It reads a fragment's render-local position and a uniform packed
// from the water alone (`crate::wetness::pack_wetness`). The P18.2 law —
// camera-driven residency never feeds lighting — applies here in advance: an
// ocean's *drawn* patch follows the camera, and taking the band's footprint from
// that patch instead of from the body's level is precisely the bug this
// arrangement makes unavailable.
//
// ## The off path
//
// `wet.dims.x == 0u` on every scene without water, and every call site is behind
// `if (wet.dims.x > 0u)`. So a water-free frame runs the arithmetic it always
// did, instruction for instruction, and every pre-P20.3 golden stands.

// Body kinds — must match `crate::water::WaterKindGpu::code()` (and
// `crate::wetness::WET_*`, pinned by `wet_kind_codes_match_the_draw_codes`).
const WET_OCEAN: u32 = 0u;
const WET_LAKE: u32 = 1u;
const WET_RIVER: u32 = 2u;

struct WetBody {
    // x = kind, y = first segment, z = segment count, w reserved.
    dims: vec4<u32>,
    // xy = render-local centre XZ, zw = half extent XZ (lakes).
    center_extent: vec4<f32>,
    // x = render-local still-water Y, yzw reserved.
    level: vec4<f32>,
};

struct WetSegment {
    // xy = start XZ, zw = end XZ (render-local).
    ab: vec4<f32>,
    // x/y = start/end surface Y, z/w = start/end HALF width (m).
    ys: vec4<f32>,
};

struct Wetness {
    // x = body count, yzw reserved.
    dims: vec4<u32>,
    // x = band height (m), y = albedo scale, z = roughness scale,
    // w = footprint margin (m).
    params: vec4<f32>,
    bodies: array<WetBody, 8>,
    segments: array<WetSegment, 32>,
};

@group(GROUP_ENV) @binding(13) var<uniform> wet: Wetness;

// The band profile: 1 at or below the water level (that ground is submerged),
// falling to 0 over `params.x` metres above it.
fn wet_band(height_above_level: f32) -> f32 {
    return 1.0 - smoothstep(0.0, max(wet.params.x, 1e-4), height_above_level);
}

// The wet fraction at a render-local world position, `[0, 1]`.
//
// Bodies combine by MAX rather than by sum: wetness is a coverage fraction, and
// a shore wetted by two overlapping lakes is not wetter than fully wet — the same
// argument the water shader's foam makes.
fn wet_factor(world_local: vec3<f32>) -> f32 {
    let count = wet.dims.x;
    if (count == 0u) {
        return 0.0;
    }
    let margin = wet.params.w;
    var best = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let b = wet.bodies[i];
        let kind = b.dims.x;
        if (kind == WET_OCEAN) {
            // Unbounded: an ocean has no authored edge, so every fragment is over
            // it and only the height decides.
            best = max(best, wet_band(world_local.y - b.level.x));
        } else if (kind == WET_LAKE) {
            // The authored rectangle, DILATED by the margin — the wet ground is
            // by definition just outside where the water is.
            let d = abs(world_local.xz - b.center_extent.xy)
                - (b.center_extent.zw + vec2<f32>(margin, margin));
            if (d.x <= 0.0 && d.y <= 0.0) {
                best = max(best, wet_band(world_local.y - b.level.x));
            }
        } else {
            // A river's level FOLLOWS ITS SPLINE, so the band has to as well:
            // find the nearest centreline segment this fragment is within reach
            // of, and take the surface elevation interpolated along it. Anything
            // simpler (the body's reference level) would wet a whole hillside at
            // the height of the river's source.
            let first = b.dims.y;
            let n = b.dims.z;
            var nearest = 1.0e30;
            var level = 0.0;
            for (var s = 0u; s < n; s = s + 1u) {
                let seg = wet.segments[first + s];
                let a = seg.ab.xy;
                let c = seg.ab.zw;
                let ac = c - a;
                let len2 = max(dot(ac, ac), 1e-12);
                let u = clamp(dot(world_local.xz - a, ac) / len2, 0.0, 1.0);
                let dist = length(world_local.xz - (a + ac * u));
                let half_w = mix(seg.ys.z, seg.ys.w, u);
                if (dist <= half_w + margin && dist < nearest) {
                    nearest = dist;
                    level = mix(seg.ys.x, seg.ys.y, u);
                }
            }
            if (nearest < 1.0e29) {
                best = max(best, wet_band(world_local.y - level));
            }
        }
    }
    return clamp(best, 0.0, 1.0);
}

// The wetness response: `rgb` = wetted albedo, `a` = wetted roughness. One
// function so the two lit shaders that call it cannot apply half of the effect
// each. Callers guard on `wet.dims.x > 0u` and re-clamp the roughness into their
// own floor.
fn wet_apply(world_local: vec3<f32>, albedo: vec3<f32>, rough: f32) -> vec4<f32> {
    let f = wet_factor(world_local);
    return vec4<f32>(
        albedo * mix(1.0, wet.params.y, f),
        rough * mix(1.0, wet.params.z, f),
    );
}
