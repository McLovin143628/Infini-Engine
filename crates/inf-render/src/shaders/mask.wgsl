// Selection/hover mask fragment (R8Unorm): draws instances as flat silhouettes
// whose value distinguishes selection strength (1.0 selected, 0.5 hovered) —
// the composite pass dilates this into the colored outline. Split out of
// mesh.wgsl so the mesh module can own the lights bind group at @group(1).
//
// The vertex path shares the same instance vertex layout as the mesh pass but
// only consumes position + the model matrix (extra instance attributes are
// simply left unread).

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs(in: VsIn) -> VsOut {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    var out: VsOut;
    out.pos = view.view_proj * (model * vec4<f32>(in.pos, 1.0));
    return out;
}

struct MaskParams {
    value: vec4<f32>, // x = mask value
};
@group(1) @binding(0) var<uniform> mask: MaskParams;

@fragment
fn fs_mask(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(mask.value.x, 0.0, 0.0, 1.0);
}
