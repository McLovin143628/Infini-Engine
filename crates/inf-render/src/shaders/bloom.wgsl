// Bloom (P13.3a): soft-knee prefilter → downsample mip chain → additive tent
// upsample. `fs_prefilter` extracts the bright part of the HDR scene into mip 0;
// `fs_down` box-downsamples one level; `fs_up` tent-upsamples a lower mip and is
// drawn with an ADDITIVE blend state so each coarser level accumulates upward.
// The final mip 0 is the bloom texture the tonemap pass adds to the scene.

struct Params {
    // x = threshold, y = knee, z = texel.x, w = texel.y (of the SOURCE texture).
    v: vec4<f32>,
};
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> params: Params;

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

// Mirrors `soft_knee_factor` in settings.rs.
fn soft_knee(brightness: f32, threshold: f32, knee: f32) -> f32 {
    let k = max(knee, 1e-5);
    let rq = clamp(brightness - threshold + k, 0.0, 2.0 * k);
    let soft = rq * rq / (4.0 * k + 1e-5);
    let contrib = max(soft, brightness - threshold);
    return clamp(contrib / max(brightness, 1e-5), 0.0, 1.0);
}

@fragment
fn fs_prefilter(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSampleLevel(src, samp, in.uv, 0.0).rgb;
    let brightness = max(c.r, max(c.g, c.b));
    let f = soft_knee(brightness, params.v.x, params.v.y);
    return vec4<f32>(c * f, 1.0);
}

// 4-tap bilinear box downsample (each tap already averages 4 source texels).
@fragment
fn fs_down(in: VsOut) -> @location(0) vec4<f32> {
    let t = params.v.zw;
    var c = textureSampleLevel(src, samp, in.uv + vec2<f32>(-t.x, -t.y), 0.0).rgb;
    c = c + textureSampleLevel(src, samp, in.uv + vec2<f32>(t.x, -t.y), 0.0).rgb;
    c = c + textureSampleLevel(src, samp, in.uv + vec2<f32>(-t.x, t.y), 0.0).rgb;
    c = c + textureSampleLevel(src, samp, in.uv + vec2<f32>(t.x, t.y), 0.0).rgb;
    return vec4<f32>(c * 0.25, 1.0);
}

// 3×3 tent upsample (source = the coarser mip); drawn with additive blend.
@fragment
fn fs_up(in: VsOut) -> @location(0) vec4<f32> {
    let t = params.v.zw;
    var c = textureSampleLevel(src, samp, in.uv, 0.0).rgb * 4.0;
    c = c + textureSampleLevel(src, samp, in.uv + vec2<f32>(t.x, 0.0), 0.0).rgb * 2.0;
    c = c + textureSampleLevel(src, samp, in.uv + vec2<f32>(-t.x, 0.0), 0.0).rgb * 2.0;
    c = c + textureSampleLevel(src, samp, in.uv + vec2<f32>(0.0, t.x), 0.0).rgb * 2.0;
    c = c + textureSampleLevel(src, samp, in.uv + vec2<f32>(0.0, -t.x), 0.0).rgb * 2.0;
    c = c + textureSampleLevel(src, samp, in.uv + vec2<f32>(t.x, t.y), 0.0).rgb;
    c = c + textureSampleLevel(src, samp, in.uv + vec2<f32>(-t.x, t.y), 0.0).rgb;
    c = c + textureSampleLevel(src, samp, in.uv + vec2<f32>(t.x, -t.y), 0.0).rgb;
    c = c + textureSampleLevel(src, samp, in.uv + vec2<f32>(-t.x, -t.y), 0.0).rgb;
    return vec4<f32>(c / 16.0, 1.0);
}
