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
    // Debug slope+altitude ramp until splat-blended layered materials land
    // (P10.4). Replace `albedo` here with the splat-weighted layer blend; the
    // lighting below is already the shared PBR-lite path.
    let slope = clamp(1.0 - n.y, 0.0, 1.0);
    let grass = vec3<f32>(0.20, 0.34, 0.14);
    let rock = vec3<f32>(0.33, 0.30, 0.27);
    let snow = vec3<f32>(0.86, 0.89, 0.94);
    let rocky = mix(grass, rock, smoothstep(0.20, 0.55, slope));
    let snow_t = smoothstep(6.0, 14.0, in.height) * (1.0 - slope * 0.6);
    let albedo = mix(rocky, snow, clamp(snow_t, 0.0, 1.0));
    // ───────────────────────────────────────────────────────────────────────

    let sun = normalize(view.sun_dir.xyz);
    let ndl = max(dot(n, sun), 0.0);
    // Hemispheric ambient (sky above / ground below).
    let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    let ambient = mix(vec3<f32>(0.05, 0.06, 0.07), vec3<f32>(0.16, 0.20, 0.26), up);
    let lo = albedo * (ambient + ndl * vec3<f32>(1.15, 1.10, 1.0));

    var col = tonemap_aces(lo);
    // Distance haze toward the horizon colour.
    let dist = length(in.world_local - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.0025);
    col = mix(col, vec3<f32>(0.055, 0.081, 0.120), haze * 0.5);
    return vec4<f32>(col, 1.0);
}
