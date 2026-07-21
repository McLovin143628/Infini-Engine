// Heightfield terrain pass (P10.1): geometry-clipmap LOD patches, vertex-shader
// displaced by a per-tile R32Float height texture. One draw per visible tile
// patch (its LOD mesh + its height texture); per-patch data rides the instance
// buffer. LOD morphing (blend toward the coarser grid) kills popping; skirt
// vertices drop the patch boundary down to hide cracks between differing LODs.
//
// Shading is a debug slope+altitude ramp lit by the scene sun — the P10.4 splat
// material hook is marked below. Opaque, reverse-Z, depth-writing.
//
// The height texture is sampled with `textureLoad` + manual bilinear (not a
// filtering sampler), so R32Float works as UnfilterableFloat everywhere —
// no FLOAT32_FILTERABLE feature needed (keeps headless CI adapters happy).

// The per-tile height texture (R32Float; texel = f32 metre offset from the tile
// origin's Y), bound per patch at @group(1).
@group(1) @binding(0) var height_tex: texture_2d<f32>;
// The per-tile splat-weight texture (Rgba8Unorm; texel = the four normalized
// layer weights), bound per patch beside the height texture (P10.4).
@group(1) @binding(1) var weight_tex: texture_2d<f32>;

// Terrain splat material (@group(2)): four layers + macro variation. Mirrors
// `MaterialRaw` in passes/terrain.rs. `params[k].x` = roughness, `.y` = tex_scale.
struct TerrainMaterial {
    albedo: array<vec4<f32>, 4>,
    params: array<vec4<f32>, 4>,
    macro_amp: vec4<f32>,
};
@group(2) @binding(0) var<uniform> material: TerrainMaterial;

struct VIn {
    // Vertex: unit patch coordinates in [0,1]² + skirt flag (z = 1 on the
    // boundary skirt ring, else 0).
    @location(0) uv_skirt: vec3<f32>,
    // Instance (per patch): origin_local.xyz + world tile span.
    @location(1) o_span: vec4<f32>,
    // Instance: morph, grid cells at this LOD, texture resolution, skirt depth (m).
    @location(2) params: vec4<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_local: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) span_res: vec2<f32>, // span, resolution
    @location(3) height: f32,                             // world height (offset)
};

fn load_texel(ij: vec2<i32>, res: f32) -> f32 {
    let m = i32(res) - 1;
    let c = clamp(ij, vec2<i32>(0, 0), vec2<i32>(m, m));
    return textureLoad(height_tex, c, 0).r;
}

// Bilinearly-sampled height offset (metres) at unit patch coord `uv`.
fn sample_height(uv: vec2<f32>, res: f32) -> f32 {
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (res - 1.0);
    let i0 = floor(p);
    let f = p - i0;
    let ii = vec2<i32>(i0);
    let h00 = load_texel(ii, res);
    let h10 = load_texel(ii + vec2<i32>(1, 0), res);
    let h01 = load_texel(ii + vec2<i32>(0, 1), res);
    let h11 = load_texel(ii + vec2<i32>(1, 1), res);
    let hx0 = mix(h00, h10, f.x);
    let hx1 = mix(h01, h11, f.x);
    return mix(hx0, hx1, f.y);
}

// ── splat weights + procedural detail (P10.4) ────────────────────────────────

fn load_weight(ij: vec2<i32>, res: f32) -> vec4<f32> {
    let m = i32(res) - 1;
    let c = clamp(ij, vec2<i32>(0, 0), vec2<i32>(m, m));
    return textureLoad(weight_tex, c, 0);
}

// Bilinearly-sampled RGBA splat weights at unit patch coord `uv`, renormalized so
// the four channels sum to 1 (a defensive guard on hand-authored/interpolated
// weights; a zeroed sample falls back to pure layer 0).
fn sample_weights(uv: vec2<f32>, res: f32) -> vec4<f32> {
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (res - 1.0);
    let i0 = floor(p);
    let f = p - i0;
    let ii = vec2<i32>(i0);
    let w00 = load_weight(ii, res);
    let w10 = load_weight(ii + vec2<i32>(1, 0), res);
    let w01 = load_weight(ii + vec2<i32>(0, 1), res);
    let w11 = load_weight(ii + vec2<i32>(1, 1), res);
    let wx0 = mix(w00, w10, f.x);
    let wx1 = mix(w01, w11, f.x);
    var w = mix(wx0, wx1, f.y);
    let s = w.x + w.y + w.z + w.w;
    if (s > 1e-4) {
        w = w / s;
    } else {
        w = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }
    return w;
}

// Cheap 2D value-noise (hash lattice → smooth interpolation), range [0, 1].
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// 4-octave fBm, range ~[0, 1].
fn fbm2(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var freq = p;
    for (var i = 0; i < 4; i = i + 1) {
        v = v + amp * vnoise(freq);
        freq = freq * 2.0;
        amp = amp * 0.5;
    }
    return v;
}

// Triplanar axis weights: normalized pow(|n|, sharpness) over the YZ/XZ/XY planes
// (mirrors `triplanar_axis_weights` in passes/terrain.rs, which unit-tests it).
fn triplanar_weights(n: vec3<f32>, sharpness: f32) -> vec3<f32> {
    var w = pow(abs(n), vec3<f32>(sharpness));
    let s = w.x + w.y + w.z;
    if (s > 0.0) {
        w = w / s;
    } else {
        w = vec3<f32>(0.0, 1.0, 0.0);
    }
    return w;
}

// A world-space triplanar detail grain in [0, 1]: value-noise projected on the
// three world planes at `1/tex_scale`, blended by the triplanar axis weights so
// steep faces read the vertical (XY/YZ) projections instead of a stretched top.
fn triplanar_grain(world: vec3<f32>, n: vec3<f32>, tex_scale: f32) -> f32 {
    let scale = 1.0 / max(tex_scale, 0.001);
    let gx = vnoise(world.yz * scale);
    let gy = vnoise(world.xz * scale);
    let gz = vnoise(world.xy * scale);
    let tw = triplanar_weights(n, 4.0);
    return gx * tw.x + gy * tw.y + gz * tw.z;
}

@vertex
fn vs(in: VIn) -> VOut {
    let uv = in.uv_skirt.xy;
    let skirt = in.uv_skirt.z;
    let span = in.o_span.w;
    let morph = in.params.x;
    let cells = max(in.params.y, 1.0);
    let res = in.params.z;
    let skirt_depth = in.params.w;

    // LOD morph: blend the fine height toward the height at the next-coarser grid
    // vertex (snap uv to every-other vertex of this LOD). morph 0 → fine, 1 → coarse.
    let coarse_step = 2.0 / cells;
    let coarse_uv = round(uv / coarse_step) * coarse_step;
    let h_fine = sample_height(uv, res);
    let h_coarse = sample_height(coarse_uv, res);
    let h = mix(h_fine, h_coarse, morph);

    // Render-local position: tile origin + planar offset + displaced height.
    var pos = in.o_span.xyz + vec3<f32>(uv.x * span, h, uv.y * span);
    // Skirt: drop the boundary ring straight down to seal cracks.
    pos.y = pos.y - skirt * skirt_depth;

    var out: VOut;
    out.clip = view.view_proj * vec4<f32>(pos, 1.0);
    out.world_local = pos;
    out.uv = uv;
    out.span_res = vec2<f32>(span, res);
    out.height = h;
    return out;
}

// Narkowicz ACES filmic approximation.
fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let span = in.span_res.x;
    let res = max(in.span_res.y, 2.0);
    let world_step = span / (res - 1.0);   // world metres between texels
    let texel = 1.0 / (res - 1.0);

    // Central-difference normal from the height texture (world-space gradient).
    let hl = sample_height(in.uv - vec2<f32>(texel, 0.0), res);
    let hr = sample_height(in.uv + vec2<f32>(texel, 0.0), res);
    let hd = sample_height(in.uv - vec2<f32>(0.0, texel), res);
    let hu = sample_height(in.uv + vec2<f32>(0.0, texel), res);
    let dhdx = (hr - hl) / (2.0 * world_step);
    let dhdz = (hu - hd) / (2.0 * world_step);
    let n = normalize(vec3<f32>(-dhdx, 1.0, -dhdz));

    // ── P10.4 SPLAT MATERIAL HOOK ──────────────────────────────────────────
    // Splat-blended layered material: blend the four layers' albedo/roughness by
    // the per-sample weight texture, add a world-space triplanar detail grain
    // (so steep faces don't stretch), then a large-scale fBm macro variation.
    // The lighting below is the shared PBR-lite path (now roughness-aware).
    let w = sample_weights(in.uv, res);
    var albedo = w.x * material.albedo[0].rgb
        + w.y * material.albedo[1].rgb
        + w.z * material.albedo[2].rgb
        + w.w * material.albedo[3].rgb;
    let roughness = clamp(
        w.x * material.params[0].x + w.y * material.params[1].x
            + w.z * material.params[2].x + w.w * material.params[3].x,
        0.04, 1.0);
    let tex_scale = w.x * material.params[0].y + w.y * material.params[1].y
        + w.z * material.params[2].y + w.w * material.params[3].y;

    // Triplanar detail grain (subtle multiplicative tint, ±15%).
    let grain = triplanar_grain(in.world_local, n, tex_scale);
    albedo = albedo * (0.85 + 0.30 * grain);

    // Macro variation: large-scale fBm brightening/darkening (signed, ±amp).
    let macro_fbm = 2.0 * fbm2(in.world_local.xz * 0.01) - 1.0;
    albedo = albedo * (1.0 + material.macro_amp.x * macro_fbm);
    albedo = clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0));
    // ───────────────────────────────────────────────────────────────────────

    let sun = normalize(view.sun_dir.xyz);
    let ndl = max(dot(n, sun), 0.0);
    // Hemispheric ambient (sky above / ground below).
    let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(vec3<f32>(0.05, 0.06, 0.07), vec3<f32>(0.16, 0.20, 0.26), up);
    // A cheap roughness-aware specular glint (Blinn-ish): smoother layers (lower
    // roughness) get a tighter, brighter highlight, so roughness reads visibly.
    let view_dir = normalize(view.eye.xyz - in.world_local);
    let half_v = normalize(sun + view_dir);
    let gloss = (1.0 - roughness) * (1.0 - roughness);
    let spec_power = mix(8.0, 128.0, gloss);
    let spec = pow(max(dot(n, half_v), 0.0), spec_power) * gloss * 0.4;
    let lo = albedo * (ambient + ndl * vec3<f32>(1.15, 1.10, 1.0)) + vec3<f32>(spec);

    var col = tonemap_aces(lo);
    // Distance haze toward the horizon colour.
    let dist = length(in.world_local - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.0025);
    col = mix(col, vec3<f32>(0.055, 0.081, 0.120), haze * 0.5);
    return vec4<f32>(col, 1.0);
}
