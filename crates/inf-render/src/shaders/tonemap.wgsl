// Tonemap post pass (P13.3a): the HDR scene (linear `Rgba16Float`, plus additive
// bloom) → display-referred output. Exposure scale → additive bloom → Narkowicz
// ACES → optional ordered dither. Writes an `Rgba8UnormSrgb` target, so the
// hardware applies the sRGB OETF to this (linear, post-ACES) result — matching
// the old in-shader tonemap exactly at defaults (bloom off, exposure 1).

struct Params {
    // x = UNUSED since wave VIS1b (the exposure moved to its own buffer, because
    // the bloom prefilter needs the same number and a uniform this pass owns
    // could not reach it), y = bloom intensity, z = dither (>0.5), w = the lens
    // flare is on (>0.5).
    knobs: vec4<f32>,
    // xy = output resolution (px), zw unused.
    resolution: vec4<f32>,
};
struct Exposure {
    // x = the frame's linear exposure multiplier; see `inf_render::exposure`.
    v: vec4<f32>,
};
@group(0) @binding(0) var hdr_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<uniform> exposure: Exposure;
// The half-res sun glare / ghost chain (wave VIS1b). Already in exposed units —
// its own bright pass multiplied by the same `exposure.v.x` — so it is added
// here rather than exposed again. Black on every frame the flare is off.
@group(0) @binding(5) var flare_tex: texture_2d<f32>;

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

    // **Exposure first, then bloom** (wave VIS1b — the ordering decision).
    // The prefilter already multiplied by the same `exposure.v.x`, so the bloom
    // texture arrives in exposed units and must NOT be exposed twice.
    //
    // At `exposure == 1.0` this is arithmetically the expression it replaced:
    // `(a + b) * 1.0` and `a * 1.0 + b` are the same bits. Every committed golden
    // renders at exposure 1.0, which is why the reordering re-blessed nothing.
    hdr = hdr * exposure.v.x;
    hdr = hdr + bloom * params.knobs.y;
    if (params.knobs.w > 0.5) {
        // Branched rather than added unconditionally: the target is black when
        // the flare is off, so the add would be exact — but it would still be a
        // full-resolution texture fetch on every shipped frame of every level
        // that never asked for a lens.
        hdr = hdr + textureSampleLevel(flare_tex, samp, in.uv, 0.0).rgb;
    }

    var col = tonemap_aces(hdr);

    if (params.knobs.z > 0.5) {
        // ±0.5/255 triangular dither so 8-bit banding disappears.
        let n = ign(in.pos.xy) - 0.5;
        col = col + vec3<f32>(n / 255.0);
    }
    return vec4<f32>(col, 1.0);
}
