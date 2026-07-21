// TAA resolve (P13.3a): blend the current (camera-jittered) HDR frame with the
// reprojected history, clamped to the local 3×3 neighbourhood to kill ghosting.
// `common_view.wgsl` (group 0 = view) is prepended — its jittered `inv_view_proj`
// reconstructs world position from the depth prepass; `prev_view_proj` (group 1)
// projects that into last frame's history. Camera-only motion vectors (v1);
// per-object motion vectors are the documented follow-up. Writes the HDR history
// ping-pong target that bloom + tonemap then consume.

struct TaaParams {
    prev_view_proj: mat4x4<f32>,
    // x = blend (history weight), y = history valid (>0.5), zw = resolution (px).
    cfg: vec4<f32>,
};
@group(1) @binding(0) var scene_tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;
@group(1) @binding(2) var history_tex: texture_2d<f32>;
@group(1) @binding(3) var depth_tex: texture_depth_2d;
@group(1) @binding(4) var<uniform> taa: TaaParams;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let ndc = fullscreen_ndc(i);
    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return out;
}

fn load_depth(uv: vec2<f32>) -> f32 {
    let d = taa.cfg.zw;
    let c = clamp(vec2<i32>(uv * d), vec2<i32>(0), vec2<i32>(d) - vec2<i32>(1));
    return textureLoad(depth_tex, c, 0);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let current = textureSampleLevel(scene_tex, samp, in.uv, 0.0).rgb;

    // First frame / resize: history invalid → take the current frame.
    if (taa.cfg.y < 0.5) {
        return vec4<f32>(current, 1.0);
    }

    // Camera-only reprojection through the depth prepass.
    let depth = load_depth(in.uv);
    var hist_uv = in.uv;
    if (depth > 0.0) {
        let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
        let world = unproject(ndc, depth);
        let pc = taa.prev_view_proj * vec4<f32>(world, 1.0);
        if (pc.w > 0.0) {
            let p = pc.xyz / pc.w;
            hist_uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
        }
    }
    // Off-screen reprojection → no valid history.
    if (hist_uv.x < 0.0 || hist_uv.x > 1.0 || hist_uv.y < 0.0 || hist_uv.y > 1.0) {
        return vec4<f32>(current, 1.0);
    }

    // Neighbourhood min/max clamp (3×3) in the current frame.
    let t = 1.0 / taa.cfg.zw;
    var lo = current;
    var hi = current;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let c = textureSampleLevel(scene_tex, samp, in.uv + vec2<f32>(f32(x), f32(y)) * t, 0.0).rgb;
            lo = min(lo, c);
            hi = max(hi, c);
        }
    }
    var history = textureSampleLevel(history_tex, samp, hist_uv, 0.0).rgb;
    history = clamp(history, lo, hi);

    let out = mix(current, history, taa.cfg.x);
    return vec4<f32>(out, 1.0);
}
