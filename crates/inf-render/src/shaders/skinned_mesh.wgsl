// GPU skinning pass (P11.1): the skinned variant of mesh.wgsl. The vertex stage
// deforms a bind-space vertex by the per-instance joint palette (@group(3),
// weighted across its four joints) BEFORE the model matrix; the fragment stage is
// the same metallic-roughness PBR (duplicated here so mesh.wgsl — and every
// existing golden's pixels — stays byte-stable). `common_view.wgsl` is prepended.

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) joints: vec4<u32>,
    @location(3) weights: vec4<f32>,
    // Instance data (buffer 1), @location(4..=14).
    @location(4) model_0: vec4<f32>,
    @location(5) model_1: vec4<f32>,
    @location(6) model_2: vec4<f32>,
    @location(7) model_3: vec4<f32>,
    @location(8) nrm_0: vec4<f32>,
    @location(9) nrm_1: vec4<f32>,
    @location(10) nrm_2: vec4<f32>,
    @location(11) color: vec4<f32>,
    @location(12) misc: vec4<u32>,
    @location(13) pbr: vec4<f32>,
    @location(14) emissive: vec4<f32>,
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

// The per-instance joint palette: skin[i] = global[i] · inverse_bind[i].
@group(3) @binding(0) var<storage, read> palette: array<mat4x4<f32>>;

@vertex
fn vs(in: VsIn) -> VsOut {
    // Linear blend skinning: weighted sum of the four influencing joint matrices.
    let skin = in.weights.x * palette[in.joints.x]
             + in.weights.y * palette[in.joints.y]
             + in.weights.z * palette[in.joints.z]
             + in.weights.w * palette[in.joints.w];
    let skinned_pos = (skin * vec4<f32>(in.pos, 1.0)).xyz;
    let skin3 = mat3x3<f32>(skin[0].xyz, skin[1].xyz, skin[2].xyz);
    let skinned_normal = skin3 * in.normal;

    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    let nrm = mat3x3<f32>(in.nrm_0.xyz, in.nrm_1.xyz, in.nrm_2.xyz);
    var out: VsOut;
    let wp = model * vec4<f32>(skinned_pos, 1.0);
    out.pos = view.view_proj * wp;
    out.world_pos = wp.xyz;
    out.normal = nrm * skinned_normal;
    out.color = in.color;
    out.id = in.misc.x;
    out.pbr = in.pbr;
    out.emissive = in.emissive.rgb;
    return out;
}

// ── Lights (must match LightsUniform / MAX_LIGHTS in passes/mesh.rs) ──
const MAX_LIGHTS: u32 = 16u;

struct GpuLight {
    color: vec4<f32>,
    pos_dir: vec4<f32>,
    params: vec4<f32>,
};
struct Lights {
    count: vec4<u32>,
    items: array<GpuLight, MAX_LIGHTS>,
};
@group(1) @binding(0) var<uniform> lights: Lights;

// SSAO (P13.3a): group 2 (the material seam slot on the skinned pipeline is
// group 2 empty in the old layout — replaced by the AO bind). 1×1 white when
// disabled → ambient unchanged.
@group(2) @binding(0) var ao_tex: texture_2d<f32>;
@group(2) @binding(1) var ao_smp: sampler;

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
    let n = normalize(in.normal);
    let v = normalize(view.eye.xyz - in.world_pos);
    let albedo = in.color.rgb;
    let metallic = clamp(in.pbr.x, 0.0, 1.0);
    let rough = clamp(in.pbr.y, 0.04, 1.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);
    let count = lights.count.x;
    if (count == 0u) {
        lo += shade_light(n, v, normalize(view.sun_dir.xyz), vec3<f32>(3.0),
                          albedo, metallic, rough, f0);
    } else {
        for (var i = 0u; i < count && i < MAX_LIGHTS; i = i + 1u) {
            let light = lights.items[i];
            let radiance_base = light.color.rgb * light.color.a;
            if (light.pos_dir.w < 0.5) {
                lo += shade_light(n, v, normalize(light.pos_dir.xyz), radiance_base,
                                 albedo, metallic, rough, f0);
            } else {
                let to_light = light.pos_dir.xyz - in.world_pos;
                let dist = length(to_light);
                let l = to_light / max(dist, 1e-4);
                let att = point_attenuation(dist, light.params.x);
                lo += shade_light(n, v, l, radiance_base * att,
                                 albedo, metallic, rough, f0);
            }
        }
    }

    let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    let amb = mix(vec3<f32>(0.03, 0.03, 0.035), vec3<f32>(0.10, 0.13, 0.18), up);
    let ao = textureSampleLevel(ao_tex, ao_smp, in.pos.xy / view.grid_axis_viewport.zw, 0.0).r;
    lo += amb * albedo * (1.0 - metallic) * ao;
    lo += amb * f0 * 0.5 * ao;

    lo += in.emissive;

    // HDR-linear haze; the post tonemap pass runs afterward.
    let dist = length(in.world_pos - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.004);
    let col = mix(lo, vec3<f32>(0.055, 0.081, 0.120), haze * 0.4);

    return vec4<f32>(col, in.color.a);
}
