// vsm_skinned.wgsl — **the skinned caster's page raster** (P27.2, clause 2).
//
// `vsm_caster.wgsl` with the vertex skinned first. Same page uniform at
// `@group(0)`, same pulled caster record at `@group(1)`, and the instance's own
// joint palette at `@group(2)` — the shape `skinned_mesh.wgsl` already uses,
// moved down two groups because a page raster has no view and no lights.
//
// The skin is applied in the CASTER's space and then by the caster's model
// matrix, exactly as the lit skinned pass does, so a page's silhouette is the one
// the camera sees. A skinned caster that skinned on the CPU instead would be
// O(vertices) of work per frame per instance for a path the GPU already has.

struct VsmPageDraw {
    view_proj: mat4x4<f32>,
    info: vec4<u32>,
};
@group(0) @binding(0) var<uniform> page: VsmPageDraw;

struct VsmCaster {
    model: mat4x4<f32>,
    sphere: vec4<f32>,
    mat: vec4<f32>,
    ids: vec4<u32>,
};
@group(1) @binding(0) var<storage, read> casters: array<VsmCaster>;
@group(1) @binding(1) var<storage, read> visible: array<u32>;

@group(2) @binding(0) var<storage, read> palette: array<mat4x4<f32>>;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) joints: vec4<u32>,
    @location(3) weights: vec4<f32>,
};

@vertex
fn vs(in: VsIn, @builtin(instance_index) iidx: u32) -> @builtin(position) vec4<f32> {
    let c = casters[visible[page.info.x + iidx]];
    let p = vec4<f32>(in.pos, 1.0);
    // Four influences, normalized on the way in by the projector — the same sum
    // `skinned_mesh.wgsl` takes, and an all-zero weight set falls back to the
    // bind pose rather than collapsing the vertex onto the origin.
    var skinned = vec4<f32>(0.0);
    var total = 0.0;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let w = in.weights[i];
        if (w > 0.0) {
            skinned = skinned + w * (palette[in.joints[i]] * p);
            total = total + w;
        }
    }
    if (total <= 1e-5) {
        skinned = p;
    }
    return page.view_proj * (c.model * vec4<f32>(skinned.xyz, 1.0));
}
