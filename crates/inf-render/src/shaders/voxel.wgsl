// Voxel-surface forward pass (P21.1): the meshed isosurface of an SDF voxel
// volume — caves, excavations, overhangs — drawn as opaque geometry beside the
// heightfield it locally extends.
//
// DELIBERATELY SIMPLE, and this comment is the honest ledger of it. The whole
// shading model is Cook-Torrance GGX + a hemispheric ambient, lit by the
// `@group(1)` lights uniform. It is composed `ShaderKind::Plain` — common_view and
// nothing else — so there is NO env group and therefore:
//
//   * no screen-space AO,
//   * no cascaded shadows (these surfaces neither cast nor receive),
//   * no dynamic GI,
//   * no atmosphere / aerial perspective / height fog,
//   * no cloud shadow,
//   * no shoreline wetness.
//
// P21.2 owns all of that, plus the three things that make a voxel surface part of
// the world rather than merely in front of it: **seam blending against the
// heightfield** (triplanar detail grain and splat continuity across a cave mouth),
// **shadow/GI participation**, and the **depth prepass** (so SSAO and TAA see
// these surfaces at all).
//
// Keeping this pass off the `Lit` composition is what keeps this batch's surface
// area honest rather than half-wired. A voxel surface bound into the env group
// would *look* integrated — AO'd, fogged, shadow-receiving — while still casting
// no shadow, feeding no GI and being invisible to the prepass. One simple pass
// that states what it does not do is the truthful shape for P21.1; the seam is
// then a visible thing to close, not a quiet inconsistency to discover.
//
// The GGX / Smith / Schlick / attenuation math below is copied VERBATIM from
// mesh.wgsl (minus every env-group call) precisely so the two agree: a rock face
// and the voxel cave cut into it must respond to the same light identically, and
// two independently-tuned BRDFs would not.

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Splat-layer index (categorical — see the flat interpolation below).
    @location(2) material: u32,
    // Per-chunk instance data (matches ChunkInstanceRaw in passes/voxel.rs).
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(7) albedo_0: vec4<f32>,
    @location(8) albedo_1: vec4<f32>,
    @location(9) albedo_2: vec4<f32>,
    @location(10) albedo_3: vec4<f32>,
    // The four layers' perceptual roughness in x,y,z,w.
    @location(11) rough: vec4<f32>,
    @location(12) misc: vec4<u32>, // x = pick id (0 in P21.1 — see passes/voxel.rs)
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    // The palette select happens in the VERTEX stage and both results cross the
    // stage boundary @interpolate(flat): a material index is CATEGORICAL (the same
    // rule RenderTerrainTile::biomes carries — the midpoint of layers 1 and 3 is
    // not layer 2), so what must not be interpolated is the *index*. Resolving it
    // here and flat-passing the resolved albedo/roughness is the same statement,
    // and it keeps the fragment stage free of the array entirely.
    @location(2) @interpolate(flat) albedo: vec4<f32>,
    @location(3) @interpolate(flat) rough: f32,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    var out: VsOut;
    let wp = model * vec4<f32>(in.pos, 1.0);
    out.pos = view.view_proj * wp;
    out.world_pos = wp.xyz;
    // Chunks are axis-aligned and unrotated — the model matrix is a pure
    // translation — so there is no normal matrix to apply: the normal passes
    // through unchanged.
    out.normal = in.normal;

    // Clamped palette select. `min(mat, 3u)` is what makes an out-of-range index
    // degrade to the last layer instead of reading out of bounds; the DTO
    // documents the same clamp on RenderVoxelVertex::material. `var` (not `let`)
    // because a dynamically-indexed array must be addressable — the same pattern
    // fullscreen_ndc uses in common_view.wgsl.
    let mat = min(in.material, 3u);
    var albedo = array<vec4<f32>, 4>(in.albedo_0, in.albedo_1, in.albedo_2, in.albedo_3);
    var rough = array<f32, 4>(in.rough.x, in.rough.y, in.rough.z, in.rough.w);
    out.albedo = albedo[mat];
    out.rough = rough[mat];
    return out;
}

// ── Lights (must match LightsUniform / MAX_LIGHTS in passes/mesh.rs) ──
//
// The SAME uniform block the mesh pass declares, bound at the same group by the
// same `LightsUniform` type — the voxel node reuses that packing rather than
// minting a second lights format.
const MAX_LIGHTS: u32 = 16u;

struct GpuLight {
    color: vec4<f32>,   // rgb = color, a = intensity
    pos_dir: vec4<f32>, // xyz = dir-to-light (dir) or render-local pos (point/spot); w = kind (0 dir, 1 point, 2 spot)
    params: vec4<f32>,  // x = range, y = spot inner_cos, z = spot outer_cos
    spot_dir: vec4<f32>, // xyz = normalized spot emission direction (spot only)
};
struct Lights {
    count: vec4<u32>,   // x = active count
    items: array<GpuLight, MAX_LIGHTS>,
};
@group(1) @binding(0) var<uniform> lights: Lights;

const PI: f32 = 3.14159265359;

fn distribution_ggx(n_dot_h: f32, rough: f32) -> f32 {
    let a = rough * rough;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, rough: f32) -> f32 {
    let r = rough + 1.0;
    let k = (r * r) / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Single BRDF term for a light with unit direction `l` and incoming `radiance`.
// Dielectric only — a terrain splat layer carries no metallic channel — so the
// mesh version's `metallic` parameter collapses to 0 and is dropped.
fn shade_light(
    n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>,
    albedo: vec3<f32>, rough: f32, f0: vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    let d = distribution_ggx(n_dot_h, rough);
    let g = geometry_smith(n_dot_v, n_dot_l, rough);
    let f = fresnel_schlick(v_dot_h, f0);

    let spec = (d * g) * f / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    let kd = vec3<f32>(1.0) - f;
    let diffuse = kd * albedo / PI;
    return (diffuse + spec) * radiance * n_dot_l;
}

// UE-style windowed inverse-square point attenuation.
fn point_attenuation(dist: f32, range: f32) -> f32 {
    let inv_sq = 1.0 / max(dist * dist, 1e-4);
    if (range <= 0.0) {
        return inv_sq;
    }
    let t = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
    return inv_sq * t * t;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let albedo = in.albedo.rgb;
    // Unlit view mode (R-P2): return albedo directly, skipping the light loop —
    // the same early-out every other lit pass takes, so volumetric terrain is not
    // the one surface that ignores the view mode. `flags.x` is 0 in the default
    // Lit mode, so this branch is never taken there.
    if (view.flags.x > 0.5) {
        return vec4<f32>(albedo, 1.0);
    }
    // The SDF gradient hands normalized normals in, but interpolation across a
    // triangle does not preserve unit length — that is a different fact, and it
    // is why this normalize is not redundant with the projector's.
    let n = normalize(in.normal);
    let v = normalize(view.eye.xyz - in.world_pos);
    let rough = clamp(in.rough, 0.04, 1.0);
    // Dielectric reflectance: a terrain splat layer has no metallic channel, so
    // f0 is the 4% every non-metal shares.
    let f0 = vec3<f32>(0.04);

    var lo = vec3<f32>(0.0);
    let count = lights.count.x;
    if (count == 0u) {
        // Fallback editor sun, exactly as the mesh pass's no-light path — a demo
        // scene with no authored lights still renders. Unshadowed here: cascaded
        // shadows are P21.2.
        lo += shade_light(n, v, normalize(view.sun_dir.xyz), vec3<f32>(3.0),
                          albedo, rough, f0);
    } else {
        for (var i = 0u; i < count && i < MAX_LIGHTS; i = i + 1u) {
            let light = lights.items[i];
            let radiance_base = light.color.rgb * light.color.a;
            if (light.pos_dir.w < 0.5) {
                // Directional.
                lo += shade_light(n, v, normalize(light.pos_dir.xyz), radiance_base,
                                  albedo, rough, f0);
            } else {
                // Point (w == 1) / spot (w == 2): shared windowed inverse-square
                // attenuation; a spot additionally masks by its cone. `cone` stays
                // 1.0 for a point light.
                let to_light = light.pos_dir.xyz - in.world_pos;
                let dist = length(to_light);
                let l = to_light / max(dist, 1e-4);
                let att = point_attenuation(dist, light.params.x);
                var cone = 1.0;
                if (light.pos_dir.w > 1.5) {
                    // Cosine of the angle between frag→light and the beam axis
                    // (-spot_dir = toward-the-light), faded outer_cos→inner_cos.
                    let cos_dir = dot(l, -light.spot_dir.xyz);
                    cone = smoothstep(light.params.z, light.params.y, cos_dir);
                }
                lo += shade_light(n, v, l, radiance_base * att * cone,
                                  albedo, rough, f0);
            }
        }
    }

    // Hemispheric ambient — the same sky/ground constants the mesh pass falls back
    // to when GI is off, unmodulated: there is no AO texture to multiply here (no
    // env group), and inventing one would be a lie about what this pass samples.
    // A cave interior is therefore brighter than it should be; occlusion is
    // exactly what P21.2's shadow/GI participation fixes.
    let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    let amb = mix(vec3<f32>(0.03, 0.03, 0.035), vec3<f32>(0.10, 0.13, 0.18), up);
    lo += amb * albedo;

    // No distance haze and no aerial perspective: both belong with the seam work
    // in P21.2, where a voxel surface and the heightfield it opens out of must
    // fade together or the seam becomes visible at range.
    return vec4<f32>(lo, 1.0);
}
