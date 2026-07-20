// Composite/blit: samples the offscreen scene into the output target
// (swapchain or headless), applying the letterbox/stretch transform and the
// selection-outline overlay from the mask texture.

struct Params {
    // xy = uv scale, zw = uv offset (stretch mode: 1,1,0,0).
    uv_transform: vec4<f32>,
    // Letterbox bar color.
    bg: vec4<f32>,
    // x = outline enabled (>0.5), y = mask texel width, z = mask texel height.
    outline: vec4<f32>,
};
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var scene_smp: sampler;
@group(0) @binding(2) var mask_tex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(p[i], 0.0, 1.0);
    out.uv = vec2<f32>(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
    return out;
}

const SELECT_COLOR: vec3<f32> = vec3<f32>(1.0, 0.42, 0.05); // UE-style orange
const HOVER_COLOR: vec3<f32> = vec3<f32>(0.35, 0.65, 1.0);

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv * params.uv_transform.xy + params.uv_transform.zw;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return params.bg;
    }
    var col = textureSampleLevel(scene_tex, scene_smp, uv, 0.0).rgb;

    if (params.outline.x > 0.5) {
        // 8-neighbour dilate: edge pixels are outside the mask but adjacent
        // to it. Two-texel reach gives a ~2 px outline.
        let px = params.outline.yz;
        let here = textureSampleLevel(mask_tex, scene_smp, uv, 0.0).r;
        if (here < 0.25) {
            var best = 0.0;
            for (var dy = -2; dy <= 2; dy++) {
                for (var dx = -2; dx <= 2; dx++) {
                    if (dx == 0 && dy == 0) { continue; }
                    let s = textureSampleLevel(
                        mask_tex, scene_smp,
                        uv + vec2<f32>(f32(dx), f32(dy)) * px, 0.0).r;
                    best = max(best, s);
                }
            }
            if (best > 0.75) {
                col = SELECT_COLOR;
            } else if (best > 0.25) {
                col = HOVER_COLOR;
            }
        }
    }

    return vec4<f32>(col, 1.0);
}
