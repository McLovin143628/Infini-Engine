// Cascaded shadow-map caster pass (P13.3b): a depth-only render of the rigid mesh
// instances (unit cubes) into one cascade layer of the shadow depth array, using
// that cascade's forward-Z orthographic light view_proj. Vertex-only — no colour,
// no fragment stage. `common_view.wgsl` is NOT prepended (this pass has its own
// per-cascade matrix at group 0, independent of the camera view).
//
// v1 casts the rigid `MeshInstance` geometry only (the golden is boxes on a
// plane); folding terrain + skinned casters into the shadow pass is a documented
// follow-up (mirrors the depth-prepass scope).

struct Cascade {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> cascade: Cascade;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
};

@vertex
fn vs(in: VsIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    return cascade.view_proj * (model * vec4<f32>(in.pos, 1.0));
}
