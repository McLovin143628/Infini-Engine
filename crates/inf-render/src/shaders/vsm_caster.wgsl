// vsm_caster.wgsl — **the rigid caster's page raster** (P27.2, clause 2).
//
// A depth-only draw of one page's visible casters into that page's slot of the
// `Depth32Float` atlas, through the page's own projection
// (`inf_render::vsm::vsm_page_matrix`). `common_view.wgsl` is NOT prepended: a
// page has nothing to do with the camera, and its matrix arrives in this file's
// own `@group(0)` uniform exactly as a cascade's does in `shadow_depth.wgsl`.
//
// ── the depth convention is the CAMERA's, not the cascade's
//
// `docs/memos/p27-1-depth-convention.md`: pages clear to 0 and compare `Greater`,
// so the nearest caster wins by holding the LARGER value. The cascaded shadow
// map's forward-Z is untouched — `shadow_depth.wgsl` is a different file for a
// different pass, and P27.5 is what demotes it.
//
// ── instance data is PULLED, and that is forced rather than chosen
//
// The draw is `draw_indexed_indirect` with an instance count the GPU wrote, over a
// per-page visible list the GPU compacted. A vertex-step instance buffer cannot
// express that: `instance_index` would walk the buffer from `first_instance`, and
// a non-zero `first_instance` needs `INDIRECT_FIRST_INSTANCE`, which is not
// portable (the `inf_vgeom` ruling). So the instance record is read out of a
// storage buffer at `visible[page.info.x + instance_index]` — `vgeom_mesh.wgsl`'s
// shape, with a per-page base instead of a per-asset one.

struct VsmPageDraw {
    // Render-local world -> this page's clip space.
    view_proj: mat4x4<f32>,
    // x = this (page, group)'s base in the visible list, y = the atlas slot,
    // z = the group, w = flags.
    info: vec4<u32>,
};
@group(0) @binding(0) var<uniform> page: VsmPageDraw;

struct VsmCaster {
    model: mat4x4<f32>,
    sphere: vec4<f32>,
    // x = alpha cutoff, y = blend code (0 opaque, 1 masked, 2 translucent),
    // z = base-colour alpha, w = reserved.
    mat: vec4<f32>,
    ids: vec4<u32>,
};
@group(1) @binding(0) var<storage, read> casters: array<VsmCaster>;
@group(1) @binding(1) var<storage, read> visible: array<u32>;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

fn caster_of(iidx: u32) -> VsmCaster {
    return casters[visible[page.info.x + iidx]];
}

@vertex
fn vs(in: VsIn, @builtin(instance_index) iidx: u32) -> @builtin(position) vec4<f32> {
    let c = caster_of(iidx);
    return page.view_proj * (c.model * vec4<f32>(in.pos, 1.0));
}

// The masked variant: a cut-out material must shadow as a cut-out, so the same
// alpha test the lit pass and the cascade caster run rides the page raster too.
// The predicate is `shadow_depth.wgsl`'s and `mesh.wgsl`'s, spelled identically —
// the R-P5 blend code is 1 for masked, and P26.5 pinned that spelling.
struct VsMaskedOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) cutoff: f32,
    @location(1) @interpolate(flat) blend: f32,
    @location(2) @interpolate(flat) alpha: f32,
};

@vertex
fn vs_masked(in: VsIn, @builtin(instance_index) iidx: u32) -> VsMaskedOut {
    let c = caster_of(iidx);
    var out: VsMaskedOut;
    out.pos = page.view_proj * (c.model * vec4<f32>(in.pos, 1.0));
    out.cutoff = c.mat.x;
    out.blend = c.mat.y;
    out.alpha = c.mat.z;
    return out;
}

@fragment
fn fs_masked(in: VsMaskedOut) {
    if (in.blend > 0.5 && in.blend < 1.5 && in.alpha < in.cutoff) {
        discard;
    }
}
