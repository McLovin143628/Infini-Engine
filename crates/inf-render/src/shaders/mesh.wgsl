// Instanced forward mesh pass: metallic-roughness PBR (Cook-Torrance GGX) lit by
// the scene lights uniform. `fs` shades; `fs_id` writes the pick id for the
// ID-buffer pass (same vertex path, R32Uint target). The selection-mask
// fragment lives in mask.wgsl so this module can own the lights bind group.

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Instance data
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(7) nrm_0: vec4<f32>,
    @location(8) nrm_1: vec4<f32>,
    @location(9) nrm_2: vec4<f32>,
    @location(10) color: vec4<f32>,
    @location(11) misc: vec4<u32>, // x = pick id
    @location(12) pbr: vec4<f32>,      // x = metallic, y = roughness
    @location(13) emissive: vec4<f32>, // rgb = emissive
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) @interpolate(flat) id: u32,
    @location(4) @interpolate(flat) pbr: vec4<f32>,
    @location(5) @interpolate(flat) emissive: vec3<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let nrm = mat3x3<f32>(in.nrm_0.xyz, in.nrm_1.xyz, in.nrm_2.xyz);
    var out: VsOut;
    let wp = model * vec4<f32>(in.pos, 1.0);
    out.pos = view.view_proj * wp;
    out.world_pos = wp.xyz;
    out.normal = nrm * in.normal;
    out.color = in.color;
    out.id = in.misc.x;
    out.pbr = in.pbr;
    out.emissive = in.emissive.rgb;
    return out;
}

// ── Lights (must match LightsUniform / MAX_LIGHTS in passes/mesh.rs) ──
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

// AO + cascaded shadows + dynamic GI ride the shared env bind group at @group(2)
// (declared in env_lighting.wgsl, prepended by `lit_scene_shader`): `ao_tex`/`ao_smp`
// (SSAO, white when off), `shadow_factor()`, and `gi_irradiance()`.

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
fn shade_light(
    n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>,
    albedo: vec3<f32>, metallic: f32, rough: f32, f0: vec3<f32>,
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
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
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
    // R-P5 masked alpha-test: blend code 1 (pbr.w) discards fragments whose base
    // color alpha is below the cutoff (pbr.z). Opaque (code 0) and translucent
    // (code 2) never take this branch, so every pre-R-P5 golden stays
    // byte-identical (the branch is present but always false for them). Runs
    // before the unlit short-circuit so masked cutouts show in every view mode.
    if (in.pbr.w > 0.5 && in.pbr.w < 1.5 && in.color.a < in.pbr.z) {
        discard;
    }
    // Unlit view mode (R-P2): return albedo + emissive directly, skipping the
    // light loop entirely. Drives both Unlit and Wireframe; `flags.x` is 0 in the
    // default Lit mode so this branch is never taken there (goldens byte-stable).
    if (view.flags.x > 0.5) {
        return vec4<f32>(in.color.rgb + in.emissive, in.color.a);
    }
    let n = normalize(in.normal);
    let v = normalize(view.eye.xyz - in.world_pos);
    let albedo = in.color.rgb;
    let metallic = clamp(in.pbr.x, 0.0, 1.0);
    let rough = clamp(in.pbr.y, 0.04, 1.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);
    let count = lights.count.x;
    if (count == 0u) {
        // Fallback editor sun so unlit demo scenes still render (shadowed like the
        // first directional light when CSM is on).
        var d = shade_light(n, v, normalize(view.sun_dir.xyz), vec3<f32>(3.0),
                          albedo, metallic, rough, f0);
        if (shadow.params.x > 0.5) {
            d = d * shadow_factor(in.world_pos, n);
        }
        lo += d;
    } else {
        // The first directional light receives the cascaded shadow factor.
        var shadowed = false;
        for (var i = 0u; i < count && i < MAX_LIGHTS; i = i + 1u) {
            let light = lights.items[i];
            let radiance_base = light.color.rgb * light.color.a;
            if (light.pos_dir.w < 0.5) {
                // Directional.
                var d = shade_light(n, v, normalize(light.pos_dir.xyz), radiance_base,
                                 albedo, metallic, rough, f0);
                if (shadow.params.x > 0.5 && !shadowed) {
                    d = d * shadow_factor(in.world_pos, n);
                    shadowed = true;
                }
                lo += d;
            } else {
                // Point (w == 1) / spot (w == 2): shared windowed inverse-square
                // attenuation; a spot additionally masks by its cone. `cone` stays
                // 1.0 for a point light, so `* 1.0` leaves the point path exactly
                // as before (byte-stable goldens).
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
                                 albedo, metallic, rough, f0);
            }
        }
    }

    // Image-based ambient: hemispheric sky/ground irradiance by default, or the
    // dynamic-GI probe irradiance when GI is on — modulated by the screen-space AO
    // (ambient term only — never the direct light above).
    let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    var amb = mix(vec3<f32>(0.03, 0.03, 0.035), vec3<f32>(0.10, 0.13, 0.18), up);
    if (gi.params.x > 0.5) {
        amb = gi_irradiance(in.world_pos, n);
    }
    let ao = textureSampleLevel(ao_tex, ao_smp, in.pos.xy / view.grid_axis_viewport.zw, 0.0).r;
    lo += amb * albedo * (1.0 - metallic) * ao;
    lo += amb * f0 * 0.5 * ao;

    lo += in.emissive;

    // Distance haze toward the (linear) horizon colour — applied in HDR-linear;
    // the post tonemap pass (ACES + exposure) runs afterward on the whole buffer.
    let dist = length(in.world_pos - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.004);
    var col = mix(lo, vec3<f32>(0.055, 0.081, 0.120), haze * 0.4);
    // P17.2: with an atmosphere, the fixed haze is replaced wholesale by physical
    // aerial perspective + height fog. The branch is never taken for a scene
    // without a time-of-day authority, so the arithmetic above is exactly what
    // every pre-P17.2 golden ran.
    if (atmos.params.x > 0.5) {
        col = atmos_apply(lo, in.world_pos);
    }

    return vec4<f32>(col, in.color.a);
}

@fragment
fn fs_id(in: VsOut) -> @location(0) u32 {
    return in.id;
}
