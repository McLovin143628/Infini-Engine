// Shared environment lighting (P13.3b): AO + cascaded shadows + dynamic GI bindings
// and sampling functions, prepended (after common_view.wgsl) to the lit scene
// shaders by `passes::lit_scene_shader`. The `GROUP_ENV` token is substituted with
// each pipeline's env bind-group index (mesh/skinned = 2, terrain = 3) so the same
// source serves every lit pass while the bindings land in the right group. Mirrors
// `EnvBinding` (passes/mod.rs) and the CPU math in `crate::csm` / `crate::gi`.
//
// AO stays at bindings 0/1 (declarations moved here from the individual shaders, so
// the existing inline `textureSampleLevel(ao_tex, ao_smp, …)` fragment lines are
// unchanged → byte-stable). Shadows (2,3,4) + GI (5,6) are appended and only touched
// when their `enabled` flag is set, so the off-path instruction stream is identical.

const SHADOW_RES: f32 = 2048.0; // must equal crate::csm::SHADOW_RESOLUTION
const GI_PI: f32 = 3.14159265359;

struct ShadowData {
    cascade_vp: array<mat4x4<f32>, 3>,
    splits: vec4<f32>,       // cascade far distances (xyz)
    texel_world: vec4<f32>,  // per-cascade world texel size (xyz)
    params: vec4<f32>,       // x=enabled, y=depth_bias, z=normal_bias, w=cascade_count
};

struct GiData {
    vol_min: vec4<f32>,    // xyz render-local min, w = voxel_size
    probe_min: vec4<f32>,  // xyz render-local probe grid min, w = extent
    dims: vec4<f32>,       // x = gi_dim, yzw = probe dims
    params: vec4<f32>,     // x=enabled, y=intensity, z=rays, w=instance_count
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
};

@group(GROUP_ENV) @binding(0) var ao_tex: texture_2d<f32>;
@group(GROUP_ENV) @binding(1) var ao_smp: sampler;
@group(GROUP_ENV) @binding(2) var shadow_map: texture_depth_2d_array;
@group(GROUP_ENV) @binding(3) var shadow_smp: sampler_comparison;
@group(GROUP_ENV) @binding(4) var<uniform> shadow: ShadowData;
@group(GROUP_ENV) @binding(5) var<storage, read> gi_sh: array<vec4<f32>>;
@group(GROUP_ENV) @binding(6) var<uniform> gi: GiData;

// 3×3-PCF shadow factor for the first directional light. Returns 1.0 (fully lit)
// when shadows are off or the receiver is beyond the last cascade. Selects the
// first cascade whose projected UV/depth are in range, applies a normal-offset
// (world) + constant depth (NDC) bias, and averages a 3×3 comparison-sample grid.
fn shadow_factor(world_pos: vec3<f32>, n: vec3<f32>) -> f32 {
    if (shadow.params.x < 0.5) {
        return 1.0;
    }
    let cascades = i32(shadow.params.w);
    var ci = -1;
    var uv = vec2<f32>(0.0);
    var depth = 0.0;
    for (var c = 0; c < cascades; c = c + 1) {
        let offset_pos = world_pos + n * shadow.texel_world[c] * shadow.params.z;
        let clip = shadow.cascade_vp[c] * vec4<f32>(offset_pos, 1.0);
        let ndc = clip.xyz / clip.w;
        let t = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5; // clip → uv (flip y)
        if (all(t >= vec2<f32>(0.0)) && all(t <= vec2<f32>(1.0)) && ndc.z >= 0.0 && ndc.z <= 1.0) {
            ci = c;
            uv = t;
            depth = ndc.z;
            break;
        }
    }
    if (ci < 0) {
        return 1.0;
    }
    let compare = depth - shadow.params.y;
    let texel = 1.0 / SHADOW_RES;
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let o = vec2<f32>(f32(dx), f32(dy)) * texel;
            sum = sum + textureSampleCompareLevel(shadow_map, shadow_smp, uv + o, ci, compare);
        }
    }
    return sum / 9.0;
}

fn sh_basis(d: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(0.282095, 0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x);
}

// Trilinearly probe-interpolated L1-SH irradiance at `world_pos` for normal `n`.
// The caller only invokes this when GI is enabled, so it needn't re-check the flag.
fn gi_irradiance(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let pmin = gi.probe_min.xyz;
    let extent = gi.probe_min.w;
    let pd = vec3<f32>(gi.dims.y, gi.dims.z, gi.dims.w);
    let coord = clamp((world_pos - pmin) / extent, vec3<f32>(0.0), vec3<f32>(1.0)) * (pd - 1.0);
    let base = floor(coord);
    let f = coord - base;

    var c0 = vec3<f32>(0.0);
    var c1 = vec3<f32>(0.0);
    var c2 = vec3<f32>(0.0);
    var c3 = vec3<f32>(0.0);
    let maxc = vec3<i32>(pd) - vec3<i32>(1);
    for (var i = 0; i < 8; i = i + 1) {
        let off = vec3<f32>(f32(i & 1), f32((i >> 1) & 1), f32((i >> 2) & 1));
        let w = mix(1.0 - f, f, off);
        let weight = w.x * w.y * w.z;
        let gc = clamp(vec3<i32>(base + off), vec3<i32>(0), maxc);
        let flat = u32((gc.z * i32(pd.y) + gc.y) * i32(pd.x) + gc.x) * 4u;
        c0 = c0 + weight * gi_sh[flat + 0u].rgb;
        c1 = c1 + weight * gi_sh[flat + 1u].rgb;
        c2 = c2 + weight * gi_sh[flat + 2u].rgb;
        c3 = c3 + weight * gi_sh[flat + 3u].rgb;
    }

    let b = sh_basis(n);
    // Cosine-lobe convolution (Ramamoorthi): A0 = π, A1 = 2π/3.
    let a0 = GI_PI;
    let a1 = 2.0943951;
    let e = a0 * c0 * b.x + a1 * (c1 * b.y + c2 * b.z + c3 * b.w);
    return max(e, vec3<f32>(0.0)) * gi.params.y;
}
