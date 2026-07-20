// Instanced forward mesh pass: unit meshes with per-instance model matrix,
// normal matrix, color and pick id. `fs` shades; `fs_id` writes the pick id
// for the ID-buffer pass (same vertex path, R32Uint target).

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
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) @interpolate(flat) id: u32,
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
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = view.sun_dir.xyz;

    // Half-lambert key light + cool sky ambient + weak ground bounce.
    let key = pow(max(dot(n, l), 0.0), 1.0) * 0.85;
    let ambient = 0.18 + 0.07 * max(n.y, 0.0);
    let bounce = 0.05 * max(-n.y, 0.0);

    var col = in.color.rgb * (key + ambient + bounce);

    // Cheap distance haze toward the horizon color.
    let dist = length(in.world_pos - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.004);
    col = mix(col, vec3<f32>(0.055, 0.081, 0.120), haze * 0.6);

    return vec4<f32>(col, in.color.a);
}

@fragment
fn fs_id(in: VsOut) -> @location(0) u32 {
    return in.id;
}

// Flat-white variant for the selection/hover mask target (R8Unorm). The
// value distinguishes selection strength: 1.0 selected, 0.5 hovered — the
// composite pass colors edges accordingly. The uniform is set per draw batch.
struct MaskParams {
    value: vec4<f32>, // x = mask value
};
@group(1) @binding(0) var<uniform> mask: MaskParams;

@fragment
fn fs_mask(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(mask.value.x, 0.0, 0.0, 1.0);
}
