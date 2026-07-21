// Depth prepass (P13.3a): a depth-only, single-sample render of the rigid mesh
// instances (cubes) into a sampleable `Depth32Float` target. SSAO reconstructs
// position/normal from it and TAA reprojects against it. `common_view.wgsl`
// (group 0 = view) is prepended. Vertex-only — no colour, no fragment stage.
//
// v1 draws the rigid `MeshInstance` geometry only (the SSAO golden is boxes);
// folding terrain + skinned geometry into the prepass is a documented follow-up.

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
    return view.view_proj * (model * vec4<f32>(in.pos, 1.0));
}
