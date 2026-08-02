// Shared environment lighting (P13.3b, grown in P18.4): AO + cascaded shadows +
// dynamic GI bindings and sampling functions, prepended (after common_view.wgsl) to
// the lit scene shaders by `passes::lit_scene_shader`. The `GROUP_ENV` token is
// substituted with each pipeline's env bind-group index (mesh/skinned/vgeom = 2,
// terrain = 3) so the same source serves every lit pass while the bindings land in
// the right group. Mirrors `EnvBinding` (passes/mod.rs) and the CPU math in
// `crate::csm` / `crate::gi`.
//
// AO stays at bindings 0/1 (declarations moved here from the individual shaders, so
// the existing inline `textureSampleLevel(ao_tex, ao_smp, …)` fragment lines are
// unchanged → byte-stable). Shadows (2,3,4), GI (5,6) and the P18.4 scene-depth
// binding (12, SSR) are appended and only touched when their `enabled` flag is set,
// so the off-path instruction stream is identical.

const SHADOW_RES: f32 = 2048.0; // must equal crate::csm::SHADOW_RESOLUTION
const GI_PI: f32 = 3.14159265359;

struct ShadowData {
    cascade_vp: array<mat4x4<f32>, 3>,
    splits: vec4<f32>,       // xyz = cascade far distances, w = cascade blend fraction
    texel_world: vec4<f32>,  // per-cascade world texel size (xyz)
    params: vec4<f32>,       // x=enabled, y=depth_bias, z=normal_bias, w=cascade_count
};

struct GiData {
    vol_min: vec4<f32>,    // xyz render-local min, w = voxel_size
    probe_min: vec4<f32>,  // xyz render-local probe grid min, w = extent
    dims: vec4<f32>,       // x = gi_dim, yzw = probe dims
    params: vec4<f32>,     // x=enabled, y=intensity, z=rays, w=macro dim
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    // P18.4: x = SH specular on, y = SSR on, z = SSR max distance (m),
    // w = SSR relative thickness.
    params2: vec4<f32>,
    // P18.4 amortization: x = probe start, y = probe count, z = probe total,
    // w = sky source (0 gradient, 1 atmosphere LUT).
    sched: vec4<f32>,
};

@group(GROUP_ENV) @binding(0) var ao_tex: texture_2d<f32>;
@group(GROUP_ENV) @binding(1) var ao_smp: sampler;
@group(GROUP_ENV) @binding(2) var shadow_map: texture_depth_2d_array;
@group(GROUP_ENV) @binding(3) var shadow_smp: sampler_comparison;
@group(GROUP_ENV) @binding(4) var<uniform> shadow: ShadowData;
@group(GROUP_ENV) @binding(5) var<storage, read> gi_sh: array<vec4<f32>>;
@group(GROUP_ENV) @binding(6) var<uniform> gi: GiData;
// P18.4: the single-sample scene depth (the SSAO/TAA prepass target), read by the
// SSR raymarch with `textureLoad` — no sampler, so this costs one binding, not two.
// `RenderSettings::needs_depth_prepass` is true whenever SSR is on, so the texture
// this reads was written by THIS frame's prepass.
@group(GROUP_ENV) @binding(12) var gi_scene_depth: texture_depth_2d;

// 3×3-PCF shadow factor of ONE cascade. Returns `-1.0` when the receiver falls
// outside this cascade's projection (the caller's "try the next one" signal), so
// the selection loop and the P18.4 blend can share the same sampling code instead
// of keeping two copies of the bias + PCF arithmetic in sync.
fn csm_cascade_pcf(world_pos: vec3<f32>, n: vec3<f32>, c: i32) -> f32 {
    let offset_pos = world_pos + n * shadow.texel_world[c] * shadow.params.z;
    let clip = shadow.cascade_vp[c] * vec4<f32>(offset_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let t = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5; // clip → uv (flip y)
    if (!(all(t >= vec2<f32>(0.0)) && all(t <= vec2<f32>(1.0)) && ndc.z >= 0.0 && ndc.z <= 1.0)) {
        return -1.0;
    }
    let compare = ndc.z - shadow.params.y;
    let texel = 1.0 / SHADOW_RES;
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let o = vec2<f32>(f32(dx), f32(dy)) * texel;
            sum = sum + textureSampleCompareLevel(shadow_map, shadow_smp, t + o, c, compare);
        }
    }
    return sum / 9.0;
}

// Cascaded shadow factor for the first directional light. Returns 1.0 (fully lit)
// when shadows are off or the receiver is beyond the last cascade. Selects the
// first cascade whose projected UV/depth are in range, then — P18.4 — **blends
// into the next cascade** across a band at the end of the current one, so the
// resolution change no longer shows up as a hard seam across the ground. The band
// is `splits.w` of the cascade's own range (0 ⇒ the pre-P18.4 hard switch, and the
// branch below is not taken at all).
fn shadow_factor(world_pos: vec3<f32>, n: vec3<f32>) -> f32 {
    if (shadow.params.x < 0.5) {
        return 1.0;
    }
    let cascades = i32(shadow.params.w);
    var ci = -1;
    var factor = 1.0;
    for (var c = 0; c < cascades; c = c + 1) {
        let f = csm_cascade_pcf(world_pos, n, c);
        if (f >= 0.0) {
            ci = c;
            factor = f;
            break;
        }
    }
    if (ci < 0) {
        return 1.0;
    }
    let blend = shadow.splits.w;
    if (blend > 0.0 && ci + 1 < cascades) {
        let far = shadow.splits[ci];
        var near_edge = 0.0;
        if (ci > 0) {
            near_edge = shadow.splits[ci - 1];
        }
        let band = max(far - near_edge, 1e-4) * blend;
        let dist = length(world_pos - view.eye.xyz);
        let w = clamp((dist - (far - band)) / max(band, 1e-4), 0.0, 1.0);
        if (w > 0.0) {
            let nf = csm_cascade_pcf(world_pos, n, ci + 1);
            // A receiver at the very edge can fall outside the NEXT cascade too
            // (it is fit to a bounding sphere, not a slab); keep this cascade's
            // answer rather than fading toward "lit", which would flash.
            if (nf >= 0.0) {
                factor = mix(factor, nf, w);
            }
        }
    }
    return factor;
}

fn sh_basis(d: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(0.282095, 0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x);
}

/// The trilinearly probe-interpolated L1 SH coefficients at `world_pos`, as four
/// RGB triples in a 3×4 matrix (`[c0, c1, c2, c3]` in columns). Both the diffuse
/// irradiance and the P18.4 specular reconstruction start here, so the 8-tap fetch
/// is written once.
fn gi_fetch_sh(world_pos: vec3<f32>) -> mat4x3<f32> {
    let pmin = gi.probe_min.xyz;
    let extent = gi.probe_min.w;
    let pd = vec3<f32>(gi.dims.y, gi.dims.z, gi.dims.w);
    let coord = clamp((world_pos - pmin) / extent, vec3<f32>(0.0), vec3<f32>(1.0)) * (pd - 1.0);
    let base = floor(coord);
    let f = coord - base;

    var c0 = vec3<f32>(0.0);
    var c1 = vec3<f32>(0.0);
    var c2 = vec3<f32>(0.0);
    var c3 = vec3<f32>(0.0);
    let maxc = vec3<i32>(pd) - vec3<i32>(1);
    for (var i = 0; i < 8; i = i + 1) {
        let off = vec3<f32>(f32(i & 1), f32((i >> 1) & 1), f32((i >> 2) & 1));
        let w = mix(1.0 - f, f, off);
        let weight = w.x * w.y * w.z;
        let gc = clamp(vec3<i32>(base + off), vec3<i32>(0), maxc);
        let flat = u32((gc.z * i32(pd.y) + gc.y) * i32(pd.x) + gc.x) * 4u;
        c0 = c0 + weight * gi_sh[flat + 0u].rgb;
        c1 = c1 + weight * gi_sh[flat + 1u].rgb;
        c2 = c2 + weight * gi_sh[flat + 2u].rgb;
        c3 = c3 + weight * gi_sh[flat + 3u].rgb;
    }
    return mat4x3<f32>(c0, c1, c2, c3);
}

// Trilinearly probe-interpolated L1-SH irradiance at `world_pos` for normal `n`.
// The caller only invokes this when GI is enabled, so it needn't re-check the flag.
fn gi_irradiance(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let c = gi_fetch_sh(world_pos);
    let b = sh_basis(n);
    // Cosine-lobe convolution (Ramamoorthi): A0 = π, A1 = 2π/3.
    let a0 = GI_PI;
    let a1 = 2.0943951;
    let e = a0 * c[0] * b.x + a1 * (c[1] * b.y + c[2] * b.z + c[3] * b.w);
    return max(e, vec3<f32>(0.0)) * gi.params.y;
}

// ── P18.4 specular ──────────────────────────────────────────────────────────

// Reconstruct RADIANCE along `d` from an L1 SH set, sharpened by `lobe` (1 = the
// full directional reconstruction, 0 = the DC term alone — what a fully rough
// surface sees). Mirrors `crate::gi::sh_radiance`.
//
// The projection in `gi_probes.wgsl` normalizes by `4π/rays`, which makes this an
// identity on a constant field: a probe that saw uniform radiance L reconstructs
// exactly L. That is why the specular term below reduces to the flat
// `amb * f0 * 0.5` it replaced instead of being a new light source.
fn gi_sh_radiance(c: mat4x3<f32>, d: vec3<f32>, lobe: f32) -> vec3<f32> {
    let b = sh_basis(d);
    return max(c[0] * b.x + lobe * (c[1] * b.y + c[2] * b.z + c[3] * b.w), vec3<f32>(0.0));
}

// The dominant light direction of an L1 SH set (per-channel luma of the linear
// band). Mirrors `crate::gi::sh_dominant_direction`.
fn gi_sh_dominant_dir(c: mat4x3<f32>) -> vec3<f32> {
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    // Basis order is [Y00, y, z, x] → the vector is (c3, c1, c2).
    let v = vec3<f32>(dot(c[3], luma), dot(c[1], luma), dot(c[2], luma));
    if (dot(v, v) > 1e-12) {
        return normalize(v);
    }
    return vec3<f32>(0.0);
}

// Karis' analytic split-sum environment BRDF: `(a, b)` such that a surface with
// reflectance f0 responds `f0 * a + b`. Mirrors `crate::gi::env_brdf_ab`.
fn gi_env_brdf_ab(rough: f32, nov: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

// **SSR v1** — a screen-space raymarch against the single-sample scene depth.
// Returns `vec4(hit_world_pos, 1)` on a hit, `vec4(0, 0, 0, 0)` on a miss.
//
// Honest scope, and the reason this is a *hit finder* rather than a colour fetch:
// the renderer is FORWARD, so at the moment a lit fragment shades, the scene
// colour it would want to reflect does not exist yet — there is no G-buffer to
// defer against and no colour history bound. What screen space CAN answer here is
// *where* the reflection ray lands, and the GI probe field answers what is bright
// there. So a hit **re-anchors the SH reconstruction at the hit point** instead of
// evaluating it at the shading point: the reflection then follows the geometry
// that causes it (parallax, contact darkening) rather than smearing the receiver's
// own probe lobe across the surface. A colour-sourced SSR needs either a deferred
// pass or a reprojected history — the documented follow-up.
//
// Fixed step count, no jitter, no history ⇒ deterministic by construction.
// Reverse-infinite-Z: a LARGER ndc.z is NEARER, and view distance is proportional
// to 1/ndc.z — so the penetration test below is a ratio and needs no near plane.
fn gi_ssr_hit(origin: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    let max_dist = gi.params2.z;
    let thickness = gi.params2.w;
    let vp = view.grid_axis_viewport.zw;
    let steps = 24;
    let dt = max_dist / f32(steps);
    for (var s = 1; s <= steps; s = s + 1) {
        let p = origin + dir * (dt * f32(s));
        let clip = view.view_proj * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) {
            return vec4<f32>(0.0);
        }
        let ndc = clip.xyz / clip.w;
        if (any(abs(ndc.xy) > vec2<f32>(1.0)) || ndc.z <= 0.0) {
            return vec4<f32>(0.0);
        }
        let uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
        let texel = vec2<i32>(clamp(uv * vp, vec2<f32>(0.0), vp - vec2<f32>(1.0)));
        let scene_z = textureLoad(gi_scene_depth, texel, 0);
        if (scene_z <= 0.0) {
            continue; // sky / nothing rasterized here
        }
        // The scene surface is nearer than the ray sample ⇒ the ray went behind
        // it. Accept only a shallow penetration, or a ray passing far behind a
        // foreground object would report a spurious hit on its silhouette.
        let penetration = 1.0 - ndc.z / scene_z;
        if (penetration > 0.0 && penetration < thickness) {
            return vec4<f32>(p, 1.0);
        }
    }
    return vec4<f32>(0.0);
}

// The GI specular term: prefiltered environment radiance along the reflection
// vector, times the split-sum environment BRDF. Replaces the flat
// `ambient * f0 * 0.5` the lit passes used, and is EXACTLY that value in the limit
// of a uniform radiance field with a fully rough surface (see `gi_sh_radiance`) —
// so turning GI specular on adds directionality, not energy.
//
// `rough` is perceptual roughness, `f0` the surface reflectance at normal
// incidence. The caller multiplies by AO.
fn gi_specular(world_pos: vec3<f32>, n: vec3<f32>, v: vec3<f32>, rough: f32, f0: vec3<f32>) -> vec3<f32> {
    let r = reflect(-v, n);
    var probe_pos = world_pos;
    if (gi.params2.y > 0.5) {
        let hit = gi_ssr_hit(world_pos + n * 0.02, r);
        if (hit.w > 0.5) {
            probe_pos = hit.xyz;
        }
    }
    let c = gi_fetch_sh(probe_pos);
    // A rough surface integrates the lobe away; a smooth one keeps it.
    let lobe = clamp(1.0 - rough, 0.0, 1.0);
    let radiance = gi_sh_radiance(c, r, lobe) * gi.params.y;
    let ab = gi_env_brdf_ab(rough, clamp(dot(n, v), 0.0, 1.0));
    // `GI_PI` matches the convention of `gi_irradiance`, which folds the Lambert
    // 1/π into the caller (`E = π·L` on a uniform field) — so the two terms are in
    // the same units, and the retired `ambient × f0 × 0.5` is recovered exactly
    // when `f0·a + b ≈ f0 × 0.5`, which is what the split-sum BRDF gives for a
    // fully rough, face-on surface (`env_brdf_ab(1, 1) ≈ (0.45, 0)`). Pinned by
    // `gi::tests::specular_reduces_to_the_retired_ambient_constant`.
    return radiance * GI_PI * (f0 * ab.x + vec3<f32>(ab.y));
}

// The full P18.4 ambient specular for a lit pass: the GI term when GI **and** its
// specular flag are on, otherwise the pre-P18.4 constant. Written once here so all
// four lit shaders take the identical off-path instruction stream.
fn gi_ambient_specular(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    v: vec3<f32>,
    rough: f32,
    f0: vec3<f32>,
    ambient: vec3<f32>,
) -> vec3<f32> {
    if (gi.params.x > 0.5 && gi.params2.x > 0.5) {
        return gi_specular(world_pos, n, v, rough, f0);
    }
    return ambient * f0 * 0.5;
}
