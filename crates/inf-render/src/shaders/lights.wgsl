// ── THE SHARED LIGHT LIBRARY (wave PAR0) ─────────────────────────────────────
//
// The one place a lit surface reads the frame's lights. Composed into every lit
// shader by `passes::lights_source` with three tokens substituted:
// `LIGHTS_GROUP` (the bind group), `LIGHTS_HDR` (the header uniform's binding)
// and `LIGHTS_DATA` (the storage buffer's binding). The env-bound passes read it
// through the shared environment group; the voxel and water passes, which bind
// no environment group, through a group of their own over the SAME two buffers
// (`crate::lights::LightGrid`).
//
// Before PAR0 this file's job was done six times over — mesh, scatter, skinned,
// vgeom, vis_resolve and voxel each carried a private copy of the BRDF and a
// private `for (i < count && i < MAX_LIGHTS)` loop over a 1 040 B uniform. The
// uniform held sixteen lights for the whole frame, first sixteen in scene order.
//
// The list is now a storage buffer of up to `LIGHTS_PER_FRAME_CEILING` records
// (directional lights first, then every local light that survived the CPU cull,
// both in scene order), and a camera-fitted froxel grid built on the GPU by
// `light_cluster.wgsl` names which local lights can reach each froxel. A
// fragment walks ONLY its own froxel's list.
//
// The three shadow HOOKS — `light_sun_shadow`, `light_cloud_shadow`,
// `light_local_shadow` — are not defined here. The composer appends either the
// environment shim (`lights_env.wgsl`: cascades / virtual pages / cloud map) or
// the bare shim (`lights_bare.wgsl`: 1.0 everywhere) — naga resolves module
// declarations in dependency order, so the call sites below need no forward
// declaration.

const PI: f32 = 3.14159265359;

// Flags a caller passes to `lights_direct` — which shadow terms this surface
// receives. Every pass passes what it applied before PAR0, so no pre-PAR0
// shading term moved: mesh + skinned take all three; vgeom, scatter and the
// visibility resolve never applied cloud shadows to a directional light; the
// voxel pass applies none.
const LIGHT_SUN_SHADOW: u32 = 1u;
const LIGHT_CLOUD_SHADOW: u32 = 2u;
const LIGHT_LOCAL_SHADOW: u32 = 4u;

struct GpuLight {
    color: vec4<f32>,    // rgb = color, a = intensity
    pos_dir: vec4<f32>,  // xyz = dir-to-light (dir) or render-local pos (point/spot); w = kind (0 dir, 1 point, 2 spot)
    params: vec4<f32>,   // x = range, y = spot inner_cos, z = spot outer_cos, w = virtual-shadow slot + 1
    spot_dir: vec4<f32>, // xyz = normalized spot emission direction; w = the cull radius the cluster build tests
};

// Must match `crate::lights::LightHeader` (std140, 144 B).
struct LightHeader {
    // x = records in the list, y = directional records (the first y), z = 1 when
    // the scene carried NO light at all (the fallback editor sun), w = the
    // per-froxel capacity.
    counts: vec4<u32>,
    // xyz = froxel grid dims, w = froxel count.
    grid: vec4<u32>,
    // x = vec4 index of the first froxel count, y = vec4 index of the first
    // froxel index, z = the debug view (1 = paint the froxel's light count),
    // w = reserved.
    bases: vec4<u32>,
    // x = near (end of slice 0), y = far (start of the last slice), z = the
    // log2 slice scale ((Z - 2) / log2(far / near)), w = the depth pad fraction.
    zparams: vec4<f32>,
    // xyz = the camera's unit forward (render-local), w unused.
    fwd: vec4<f32>,
    // xyz = the render-local eye the depth is measured from, w unused.
    eye: vec4<f32>,
    // The froxel build's view basis + tangents (read by `light_cluster.wgsl`).
    right: vec4<f32>,
    up: vec4<f32>,
    proj: vec4<f32>,
};

@group(LIGHTS_GROUP) @binding(LIGHTS_HDR) var<uniform> light_hdr: LightHeader;
@group(LIGHTS_GROUP) @binding(LIGHTS_DATA) var<storage, read> light_words: array<vec4<u32>>;

fn light_at(i: u32) -> GpuLight {
    let b = i * 4u;
    return GpuLight(
        bitcast<vec4<f32>>(light_words[b]),
        bitcast<vec4<f32>>(light_words[b + 1u]),
        bitcast<vec4<f32>>(light_words[b + 2u]),
        bitcast<vec4<f32>>(light_words[b + 3u]),
    );
}

fn light_word(w: u32) -> u32 {
    return light_words[w >> 2u][w & 3u];
}

// The froxel slice of a view depth. Slice 0 is [0, near), the last slice is
// [far, inf), and the Z - 2 between are log-spaced — the same partition
// `light_cluster.wgsl`'s `slice_bounds` builds, padded there so a fragment on a
// boundary can never fall into a froxel that did not test its light.
fn light_slice(depth: f32) -> u32 {
    let z = light_hdr.grid.z;
    if (depth < light_hdr.zparams.x) {
        return 0u;
    }
    if (depth >= light_hdr.zparams.y) {
        return z - 1u;
    }
    let s = 1u + u32(max(floor(log2(depth / light_hdr.zparams.x) * light_hdr.zparams.z), 0.0));
    return clamp(s, 1u, z - 2u);
}

// The froxel a fragment at pixel `frag_xy` and render-local `p` sits in.
fn light_cluster_of(frag_xy: vec2<f32>, p: vec3<f32>) -> u32 {
    let g = light_hdr.grid;
    let vp = max(view.grid_axis_viewport.zw, vec2<f32>(1.0));
    let tx = min(u32(max(frag_xy.x, 0.0) / vp.x * f32(g.x)), g.x - 1u);
    let ty = min(u32(max(frag_xy.y, 0.0) / vp.y * f32(g.y)), g.y - 1u);
    let depth = dot(p - light_hdr.eye.xyz, light_hdr.fwd.xyz);
    let tz = light_slice(depth);
    return (tz * g.y + ty) * g.x + tx;
}

// How many local lights froxel `c` lists (its stored count is the number that
// REACHED it, which may exceed the capacity — the census reads the overflow).
fn light_cluster_len(c: u32) -> u32 {
    return min(light_word(light_hdr.bases.x * 4u + c), light_hdr.counts.w);
}

fn light_cluster_item(c: u32, k: u32) -> u32 {
    return light_word(light_hdr.bases.y * 4u + c * light_hdr.counts.w + k);
}

fn distribution_ggx(n_dot_h: f32, rough: f32) -> f32 {
    let a = rough * rough;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, rough: f32) -> f32 {
    let r = rough + 1.0;
    let k = (r * r) / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Single BRDF term for a light with unit direction `l` and incoming `radiance`.
fn shade_light(
    n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>,
    albedo: vec3<f32>, metallic: f32, rough: f32, f0: vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    if (n_dot_l <= 0.0) {
        return vec3<f32>(0.0);
    }
    let n_dot_v = max(dot(n, v), 1e-4);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    let d = distribution_ggx(n_dot_h, rough);
    let g = geometry_smith(n_dot_v, n_dot_l, rough);
    let f = fresnel_schlick(v_dot_h, f0);

    // Wave VIS1a: multi-scatter energy compensation (`ggx_energy_compensation`
    // lives in `env_lighting.wgsl`; the voxel pass, which binds no environment
    // group, carries its own parity-tested copy).
    let spec = (d * g) * f / max(4.0 * n_dot_v * n_dot_l, 1e-4)
        * ggx_energy_compensation(f0, rough, n_dot_v);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;
    return (diffuse + spec) * radiance * n_dot_l;
}

// UE-style windowed inverse-square point attenuation.
fn point_attenuation(dist: f32, range: f32) -> f32 {
    let inv_sq = 1.0 / max(dist * dist, 1e-4);
    if (range <= 0.0) {
        return inv_sq;
    }
    let t = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
    return inv_sq * t * t;
}

// Every LOCAL light (point + spot) the fragment's froxel lists.
fn lights_local(
    p: vec3<f32>, frag_xy: vec2<f32>, n: vec3<f32>, v: vec3<f32>,
    albedo: vec3<f32>, metallic: f32, rough: f32, f0: vec3<f32>, flags: u32,
) -> vec3<f32> {
    var lo = vec3<f32>(0.0);
    let c = light_cluster_of(frag_xy, p);
    let len = light_cluster_len(c);
    for (var k = 0u; k < len; k = k + 1u) {
        let light = light_at(light_cluster_item(c, k));
        let radiance_base = light.color.rgb * light.color.a;
        // Point (w == 1) / spot (w == 2): shared windowed inverse-square
        // attenuation; a spot additionally masks by its cone.
        let to_light = light.pos_dir.xyz - p;
        let dist = length(to_light);
        let l = to_light / max(dist, 1e-4);
        let att = point_attenuation(dist, light.params.x);
        var cone = 1.0;
        if (light.pos_dir.w > 1.5) {
            let cos_dir = dot(l, -light.spot_dir.xyz);
            cone = smoothstep(light.params.z, light.params.y, cos_dir);
        }
        // P27.4's point/spot virtual shadows, through the hook. `params.w` is
        // 0 on every light without a page tree and the hook returns 1.0 there.
        var shadow = 1.0;
        if ((flags & LIGHT_LOCAL_SHADOW) != 0u) {
            shadow = light_local_shadow(p, n, light.params.w);
        }
        lo += shade_light(n, v, l, radiance_base * att * cone * shadow,
                          albedo, metallic, rough, f0);
    }
    return lo;
}

// The whole direct term: every directional light (the first one receives the
// sun's shadow), then the froxel's local lights — or the fallback editor sun on
// a scene that carried no light at all.
fn lights_direct(
    p: vec3<f32>, frag_xy: vec2<f32>, n: vec3<f32>, v: vec3<f32>, sun_dir: vec3<f32>,
    albedo: vec3<f32>, metallic: f32, rough: f32, f0: vec3<f32>, flags: u32,
) -> vec3<f32> {
    var lo = vec3<f32>(0.0);
    if (light_hdr.counts.z != 0u) {
        var d = shade_light(n, v, normalize(sun_dir), vec3<f32>(3.0),
                            albedo, metallic, rough, f0);
        if ((flags & LIGHT_SUN_SHADOW) != 0u) {
            d = d * light_sun_shadow(p, n);
        }
        if ((flags & LIGHT_CLOUD_SHADOW) != 0u) {
            d = d * light_cloud_shadow(p);
        }
        return d;
    }
    var shadowed = false;
    let dirs = light_hdr.counts.y;
    for (var i = 0u; i < dirs; i = i + 1u) {
        let light = light_at(i);
        let radiance_base = light.color.rgb * light.color.a;
        var d = shade_light(n, v, normalize(light.pos_dir.xyz), radiance_base,
                            albedo, metallic, rough, f0);
        if ((flags & LIGHT_SUN_SHADOW) != 0u && !shadowed) {
            d = d * light_sun_shadow(p, n);
            shadowed = true;
        }
        if ((flags & LIGHT_CLOUD_SHADOW) != 0u) {
            d = d * light_cloud_shadow(p);
        }
        lo += d;
    }
    lo += lights_local(p, frag_xy, n, v, albedo, metallic, rough, f0, flags);
    return lo;
}

// The cluster-grid debug view: a heat ramp over the froxel's light count
// (black = none, blue -> green -> red at 1 / capacity-quarter / capacity).
fn lights_debug_heat(p: vec3<f32>, frag_xy: vec2<f32>) -> vec3<f32> {
    let c = light_cluster_of(frag_xy, p);
    let n = f32(light_word(light_hdr.bases.x * 4u + c));
    if (n < 0.5) {
        return vec3<f32>(0.0);
    }
    let t = clamp(log2(n) / log2(max(f32(light_hdr.counts.w), 2.0)), 0.0, 1.0);
    let cool = vec3<f32>(0.05, 0.15, 1.0);
    let mid = vec3<f32>(0.1, 1.0, 0.2);
    let hot = vec3<f32>(1.0, 0.1, 0.05);
    if (t < 0.5) {
        return mix(cool, mid, t * 2.0);
    }
    return mix(mid, hot, (t - 0.5) * 2.0);
}

fn lights_debug_view() -> bool {
    return light_hdr.bases.z != 0u;
}
