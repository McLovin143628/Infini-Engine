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

// R-P5 masked variant: the same caster depth path, carrying base color alpha +
// blend/cutoff so `fs_masked` discards alpha-tested cutout fragments — so a masked
// object casts a matching cut-out shadow. Used only when the scene has masked
// instances; opaque scenes keep the fragment-less `vs` fast path (byte-identical).
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
    out.pos = cascade.view_proj * (model * vec4<f32>(in.pos, 1.0));
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
