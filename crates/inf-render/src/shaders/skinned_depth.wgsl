// **The skinned depth-prepass vertex stage** (wave VIS1a): `skinned_mesh.wgsl`'s
// `vs` with everything that is not a clip position deleted, and the joint palette
// moved from `@group(3)` to `@group(1)` because a depth-only pipeline binds no
// lights and no environment.
//
// The palette bind group itself is unchanged — a `wgpu::BindGroup` is built
// against a *layout*, not against an index, so `SkinnedMeshNode` binds the very
// same object at slot 1 here and at slot 3 in the colour pass.
//
// There is **no fragment stage**: `skinned_mesh.wgsl`'s `fs` contains no
// `discard` at all (no alpha test, no cutout, no mask), so a fragment-less
// pipeline draws exactly the same silhouette for less.
//
// **The skinning arithmetic is copied character for character from
// `skinned_mesh.wgsl`**, deliberately, and not from `vsm_skinned.wgsl` — the
// shadow caster normalizes the weights and falls back to the bind pose when they
// sum to nothing, which is right for a shadow and wrong here: a prepass depth
// that disagrees with the colour pass's depth by one ulp is a self-occlusion
// pattern in the AO. `common_view.wgsl` is prepended.

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) joints: vec4<u32>,
    @location(3) weights: vec4<f32>,
    @location(15) uv: vec2<f32>,
    // Instance data (buffer 1), @location(4..=14) — the same stream the colour
    // pipeline reads, so one instance buffer serves both. Only the four model
    // columns are used; the rest are declared because the vertex layout is shared.
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

@group(1) @binding(0) var<storage, read> palette: array<mat4x4<f32>>;

@vertex
fn vs(in: VsIn) -> @builtin(position) vec4<f32> {
    let skin = in.weights.x * palette[in.joints.x]
             + in.weights.y * palette[in.joints.y]
             + in.weights.z * palette[in.joints.z]
             + in.weights.w * palette[in.joints.w];
    let skinned_pos = (skin * vec4<f32>(in.pos, 1.0)).xyz;
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    return view.view_proj * (model * vec4<f32>(skinned_pos, 1.0));
}
