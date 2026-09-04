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
    //
    // **No prepass depth here → no valid history** (audit FIX1). Reverse-Z
    // clears this target to 0.0 = far, so `depth == 0.0` means "nothing this
    // pass can see wrote a depth at this pixel": the sky, and — until VIS-C1b
    // lands — every meshlet, every scattered instance, the water and the
    // translucents, none of which contribute to the prepass.
    //
    // The branch below used to fall through with `hist_uv = in.uv`, which is an
    // IDENTITY reprojection. That is right for a camera that has not moved and
    // wrong for one that has, and the wrongness is not a small one: the pixel
    // then takes 0.9 of a history holding what a DIFFERENT part of the world
    // looked like, bounded only by the 3x3 neighbourhood clamp — which on the
    // high-frequency content this describes is a wide box, so bright values
    // dilate outward a little further every frame. Reproduced headlessly in
    // `tests/taa_motion.rs` as a resolve that moves 4.3x further from its own
    // source under a moving camera while the rigid-mesh control does not move
    // at all.
    //
    // **It is NOT the washed-out island frame**, and the first version of this
    // comment said it was. Measured on the real host afterwards: with `taa`
    // forced off in `shipped_settings`, the showcase island's PIE frame is
    // unchanged (blown-out fraction 0.135 vs 0.134). That frame is dynamic GI —
    // turning `gi` off takes it to 0.000 — and is routed as such. This branch is
    // a defect on its own evidence and fixes a defect only that evidence
    // describes.
    //
    // Refusing the history is the same rule this shader already applies one
    // branch down for a reprojection that lands off-screen, and for the same
    // reason: a pixel whose history cannot be located must take the frame it
    // has. The cost is that those surfaces get no temporal AA, which is what
    // VIS-C1b already says they do not get from SSAO or SSR either — the
    // difference is that they are now merely un-antialiased instead of smeared.
    let depth = load_depth(in.uv);
    if (depth <= 0.0) {
        return vec4<f32>(current, 1.0);
    }
    var hist_uv = in.uv;
    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let world = unproject(ndc, depth);
    let pc = taa.prev_view_proj * vec4<f32>(world, 1.0);
    if (pc.w > 0.0) {
        let p = pc.xyz / pc.w;
        hist_uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
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
