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
    // **The lens trio** (wave VIS1b): x = vignette intensity, y = vignette
    // smoothness, z = lateral chromatic aberration in PIXELS of separation at the
    // corner, w = film-grain intensity.
    film: vec4<f32>,
    // x = grain cell size in px, y = the grain's seed — a LEVEL-CLOCK index, not
    // a frame counter (see `film_grain`). zw unused.
    film2: vec4<f32>,
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

// ── the lens trio (wave VIS1b) ───────────────────────────────────────────────
//
// All three live HERE, in the pass that composites the post chain into a
// display-referred frame, rather than in `composite.wgsl`. The reason is one
// pass later in the graph: the in-game UI draws AFTER the tonemap (island wave
// I5), deliberately, so a menu is not bloomed or reprojected along with the world
// behind it. Putting the grain and the vignette in the composite would put them
// over that menu, and nobody wants a grained health bar.
//
// The cost is two extra texture fetches when chromatic aberration is non-zero
// and arithmetic otherwise. All three are `0.0` at the default, and zero
// strength here is not "off by a branch" — the vignette's and the grain's
// branches are not taken at all, and the aberration's sampling collapses to the
// one fetch the pass already made.

// How far, in NDC radii, this pixel is from the frame's centre: `0` at the
// centre and `1` at a corner, whatever the aspect ratio.
fn film_radius(uv: vec2<f32>) -> f32 {
    return length((uv - vec2<f32>(0.5)) * 2.0) * 0.70710678;
}

// A 32-bit integer avalanche hash. Integer, deliberately: a `fract(sin(dot(...)))`
// hash is exactly the `f32` trig the P14 law bans on any path two machines have
// to agree about, and grain is such a path the moment a level is shared.
fn film_hash(x: u32) -> u32 {
    var h = x;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
}

// Film grain in `[-0.5, 0.5]`, from the pixel's grain CELL and a seed.
//
// The seed is `floor(level_clock * 24)` wrapped into 24 bits, computed on the CPU
// in `f64` — never `frame_index`, and never a wall clock. Two consequences, both
// intended: a paused world has a frozen grain (which is what a paused film frame
// looks like), and two runs of one document at the same time of day have the same
// grain, so a golden with grain on is a deterministic image.
fn film_grain(pix: vec2<f32>, size: f32, seed: f32) -> f32 {
    let cell = vec2<i32>(floor(pix / max(size, 1.0)));
    let k = bitcast<u32>(cell.x) * 73856093u
        ^ bitcast<u32>(cell.y) * 19349663u
        ^ bitcast<u32>(i32(seed)) * 83492791u;
    return f32(film_hash(k)) * (1.0 / 4294967296.0) - 0.5;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let r = film_radius(in.uv);

    // ── chromatic aberration ──
    //
    // A lens refracts the short wavelengths harder, so the three channels land on
    // slightly different radii — zero in the middle, worst at the corner, and
    // ALONG the radius rather than in a fixed screen direction. The authored
    // number is pixels of separation at the corner, so it means the same thing at
    // every resolution.
    var hdr: vec3<f32>;
    if (params.film.z != 0.0) {
        let dir = (in.uv - vec2<f32>(0.5)) * 2.0;
        let px = vec2<f32>(1.0, 1.0) / max(params.resolution.xy, vec2<f32>(1.0));
        let off = dir * r * params.film.z * px;
        hdr = vec3<f32>(
            textureSampleLevel(hdr_tex, samp, in.uv + off, 0.0).r,
            textureSampleLevel(hdr_tex, samp, in.uv, 0.0).g,
            textureSampleLevel(hdr_tex, samp, in.uv - off, 0.0).b,
        );
    } else {
        hdr = textureSampleLevel(hdr_tex, samp, in.uv, 0.0).rgb;
    }
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

    // ── vignette, BEFORE the tonemap ──
    //
    // A lens loses light toward the corner; it does not paint the corner black.
    // Attenuating the radiance and letting ACES answer for it keeps a blown
    // highlight in the corner blown, which is what a real vignetted frame does
    // and what a post-tonemap multiply gets wrong.
    if (params.film.x != 0.0) {
        // `smoothness` 0 puts the ramp's inner edge at 0.9 (a hard ring near the
        // corners); 1 puts it at the centre (a gradual falloff over the whole
        // frame).
        let inner = mix(0.9, 0.0, clamp(params.film.y, 0.0, 1.0));
        let f = smoothstep(inner, 1.0, r);
        hdr = hdr * (1.0 - clamp(params.film.x, 0.0, 1.0) * f);
    }

    var col = tonemap_aces(hdr);

    // ── film grain, AFTER it ──
    //
    // Grain is an artefact of the recording medium, not of the light that hit it,
    // so it belongs in display space where its amplitude means what the author
    // typed. Multiplied by `col` as well as added, so black stays black — a grain
    // that lifts the shadows is a fog, not a grain.
    if (params.film.w != 0.0) {
        let g = film_grain(in.pos.xy, params.film2.x, params.film2.y);
        col = col + col * (g * clamp(params.film.w, 0.0, 1.0));
    }

    if (params.knobs.z > 0.5) {
        // ±0.5/255 triangular dither so 8-bit banding disappears.
        let n = ign(in.pos.xy) - 0.5;
        col = col + vec3<f32>(n / 255.0);
    }
    return vec4<f32>(col, 1.0);
}
