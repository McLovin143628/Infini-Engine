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

// R-P5 masked variant: the same depth vertex path, but carrying the packed base
// color alpha + blend/cutoff (pbr.z/.w) so `fs_masked` can discard alpha-tested
// cutout fragments. Used only when the scene contains masked instances; opaque
// scenes keep the fragment-less `vs` fast path above (byte-identical goldens).
//
// **AND ITS ALPHA IS THE INSTANCE'S CONSTANT, NOT A MAP** (wave OUTFIT1 AUDIT,
// carried 176). `mesh.wgsl` and `skinned_mesh.wgsl` now multiply the albedo
// slot's own alpha into their masked test, so a cut-out that lives in a texture
// -- a hair card's coverage atlas -- is a hole in the COLOUR pass and is not one
// here. The price of matching them is stated rather than paid: this shader is
// registered `ShaderKind::Plain`, so it composes no `vt_sample.wgsl` and binds
// no virtual-texture table at all; reading a texel here means a bind group
// layout, a bind group and a per-frame table upload on a depth-only pipeline
// whose whole point is that it binds neither lights nor environment.
//
// What it costs today, measured rather than assumed: this pipeline writes
// `frame.targets.depth_prepass`, which is the SSAO / TAA / SSR depth and NOT the
// main depth buffer -- so a textured cut-out's holes are absent from those three
// effects and present everywhere a player looks. A hair card occludes its own
// ambient occlusion and does not occlude anything else.
struct VsMaskedIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) model_0: vec4<f32>,
    @location(4) model_1: vec4<f32>,
    @location(5) model_2: vec4<f32>,
    @location(6) model_3: vec4<f32>,
    @location(10) color: vec4<f32>,
    @location(12) pbr: vec4<f32>, // z = cutoff, w = blend code
};

struct VsMaskedOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) cutoff: f32,
    @location(1) @interpolate(flat) blend: f32,
    @location(2) @interpolate(flat) alpha: f32,
};

@vertex
fn vs_masked(in: VsMaskedIn) -> VsMaskedOut {
    let model = mat4x4<f32>(in.model_0, in.model_1, in.model_2, in.model_3);
    var out: VsMaskedOut;
    out.pos = view.view_proj * (model * vec4<f32>(in.pos, 1.0));
    out.cutoff = in.pbr.z;
    out.blend = in.pbr.w;
    out.alpha = in.color.a;
    return out;
}

@fragment
fn fs_masked(in: VsMaskedOut) {
    if (in.blend > 0.5 && in.blend < 1.5 && in.alpha < in.cutoff) {
        discard;
    }
}
