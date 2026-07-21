// Tonemap post pass (P13.3a): the HDR scene (linear `Rgba16Float`, plus additive
// bloom) → display-referred output. Exposure scale → additive bloom → Narkowicz
// ACES → optional ordered dither. Writes an `Rgba8UnormSrgb` target, so the
// hardware applies the sRGB OETF to this (linear, post-ACES) result — matching
// the old in-shader tonemap exactly at defaults (bloom off, exposure 1).

struct Params {
    // x = exposure, y = bloom intensity, z = dither (>0.5), w unused.
    knobs: vec4<f32>,
    // xy = output resolution (px), zw unused.
    resolution: vec4<f32>,
};
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
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

fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Interleaved-gradient noise (Jimenez): a deterministic per-pixel hash in [0,1),
// so dithering never breaks the golden determinism gate.
fn ign(pix: vec2<f32>) -> f32 {
    let m = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(m.z * fract(dot(pix, m.xy)));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    var hdr = textureSampleLevel(hdr_tex, samp, in.uv, 0.0).rgb;
    let bloom = textureSampleLevel(bloom_tex, samp, in.uv, 0.0).rgb;

    hdr = hdr + bloom * params.knobs.y;
    hdr = hdr * params.knobs.x; // exposure

    var col = tonemap_aces(hdr);

    if (params.knobs.z > 0.5) {
        // ±0.5/255 triangular dither so 8-bit banding disappears.
        let n = ign(in.pos.xy) - 0.5;
        col = col + vec3<f32>(n / 255.0);
    }
    return vec4<f32>(col, 1.0);
}
