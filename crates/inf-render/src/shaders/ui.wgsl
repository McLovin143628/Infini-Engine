// The engine's in-game UI pass (island wave I5).
//
// Screen-space quads in VIRTUAL PIXELS, origin at the top left, +y down — the
// space every UI in the world is laid out in, and the space the sprite pass
// deliberately is not (a sprite is a world quad going through the game camera).
//
// Standalone by construction: it owns its `@group(0)` and prepends nothing,
// because a UI has no camera. Exactly the argument `shadow_depth.wgsl` and the
// VSM page rasters carry, and the reason this module is in the *standalone* half
// of the naga gate rather than in `SHADER_TABLE`.

struct UiParams {
    // x, y = the viewport in virtual pixels; zw unused.
    viewport: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: UiParams;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct Instance {
    // xy = the rect's top-left corner in pixels, zw = its extent.
    @location(0) rect: vec4<f32>,
    // xy = uv_min, zw = uv_max.
    @location(1) uv: vec4<f32>,
    @location(2) color: vec4<f32>,
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, inst: Instance) -> VsOut {
    // The unit corner, in the same order `inf_render_2d::unit_corner` uses so a
    // triangle strip of four covers the quad: 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1).
    let corner = vec2<f32>(f32(vi & 1u), f32((vi >> 1u) & 1u));
    let px = inst.rect.xy + corner * inst.rect.zw;
    // Pixels to clip space. `+y` is DOWN in the UI's space and up in NDC, so the
    // y term is flipped here and nowhere else — a flip applied twice is a menu
    // that reads bottom to top.
    let ndc = vec2<f32>(
        px.x / max(params.viewport.x, 1.0) * 2.0 - 1.0,
        1.0 - px.y / max(params.viewport.y, 1.0) * 2.0,
    );
    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = mix(inst.uv.xy, inst.uv.zw, corner);
    out.color = inst.color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // Straight (non-premultiplied) alpha, the same convention the sprite path
    // uses: the texel tints and the pipeline blends.
    return textureSample(tex, samp, in.uv) * in.color;
}
