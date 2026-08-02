// scatter_mesh.wgsl — the P18.5 scatter raster: one vertex-pulled indirect draw
// for the full-mesh band and one for the impostor band, sharing a fragment stage.
// Composed by `lit_scene_shader(.., 2)`, so `common_view.wgsl`, the env bind group
// (AO + cascaded shadows + GI + atmosphere + cloud shadows) and the atmosphere
// library are all prepended — scatter receives exactly what every other lit pass
// receives.
//
// ── Why vertex pulling, when a scatter is the textbook case for classic
//    instancing ────────────────────────────────────────────────────────────────
//
// The cull compute produces a *compacted index list*, not a compacted payload.
// Feeding that to classic instancing needs `first_instance` on the indirect args
// to address a sub-range of a shared vertex buffer, and `INDIRECT_FIRST_INSTANCE`
// is a non-portable wgpu feature — the same wall the meshlet path hit in P13.1b.
// Compacting the 48-byte payloads instead would make the compaction pass write 12×
// the bytes it does now. So the list stays indices and the vertex stage reads
// `visible[instance_index]`, which is the vgeom precedent, one array subscript, and
// portable everywhere.
//
// ── The dithered cross-fade, and why it has no holes ─────────────────────────
//
// In the `fade`-metre band before `mesh_end` an instance is in BOTH lists. Its
// mesh keeps the pixels where a screen-position hash falls below the mesh weight
// `m = (mesh_end - d)/fade`, and its impostor keeps exactly the complement
// (`h >= m`). One hash, two complementary tests ⇒ every pixel in the overlap is
// covered exactly once, so the transition never thins the silhouette and never
// double-shades it. Past `mesh_end` the mesh is not in the list at all and the
// impostor keeps everything; in the last `fade` metres before the cull distance
// the impostor thins out against `e = (cull - d)/fade` and vanishes.
//
// The hash is a pure function of the pixel — **no temporal jitter, no frame
// index, no instance salt** — because a golden renders one frame from cold and a
// determinism gate renders two: anything remembered between frames would make a
// fade band a function of history.

struct ScatterInst {
    offset: vec3<f32>,
    scale: f32,
    rotation: vec4<f32>,
    color: vec4<f32>,
};

struct ScatterParams {
    // xyz = the batch anchor in render-local space, w = the impostor/cull radius
    // at unit scale (the primitive's bounding radius).
    anchor: vec4<f32>,
    // x = metallic, y = roughness, zw unused.
    material: vec4<f32>,
    // rgb = emissive, w unused.
    emissive: vec4<f32>,
    // x = full-mesh band end (m), y = cull distance (m), z = fade width (m),
    // w = impostors enabled (1/0).
    bands: vec4<f32>,
    // x = first index of this batch's primitive in the shared index buffer,
    // y = its base vertex, z = pick id, w = the compacted list capacity.
    geom: vec4<u32>,
};

@group(3) @binding(0) var<uniform> sp: ScatterParams;
@group(3) @binding(1) var<storage, read> s_instances: array<ScatterInst>;
@group(3) @binding(2) var<storage, read> s_visible: array<u32>;
// Shared primitive geometry: 6 f32 per vertex (position then normal), and the
// packed u16 index list widened to u32.
@group(3) @binding(3) var<storage, read> s_verts: array<f32>;
@group(3) @binding(4) var<storage, read> s_indices: array<u32>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
    // x = mesh weight `m`, y = impostor far weight `e`, z = impostor flag,
    // w = the billboard's radial coordinate² (impostor only).
    @location(3) @interpolate(flat) fade: vec4<f32>,
    @location(4) disc: vec2<f32>,
};

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn read_vertex(idx: u32) -> array<vec3<f32>, 2> {
    let b = idx * 6u;
    return array<vec3<f32>, 2>(
        vec3<f32>(s_verts[b], s_verts[b + 1u], s_verts[b + 2u]),
        vec3<f32>(s_verts[b + 3u], s_verts[b + 4u], s_verts[b + 5u]),
    );
}

// The two band weights for an instance at distance `d`. `m` is the mesh's, `e`
// the impostor's far ramp; both clamp to [0,1] so the common case (well inside a
// band) is an exact 1.0 and the discard test below is never taken.
fn band_weights(d: f32) -> vec2<f32> {
    let fade = max(sp.bands.z, 1.0e-3);
    return vec2<f32>(
        clamp((sp.bands.x - d) / fade, 0.0, 1.0),
        clamp((sp.bands.y - d) / fade, 0.0, 1.0),
    );
}

@vertex
fn vs_mesh(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iidx: u32) -> VsOut {
    let inst = s_instances[s_visible[iidx]];
    let vert = read_vertex(s_indices[sp.geom.x + vidx] + sp.geom.y);
    let center = sp.anchor.xyz + inst.offset;
    let world = center + qrot(inst.rotation, vert[0] * inst.scale);
    // Uniform scale ⇒ the inverse-transpose of R·S is R·S⁻¹, which normalizes back
    // to R. So a scatter normal needs no normal matrix at all — one of the two
    // reasons the instance record fits in 48 bytes.
    let normal = qrot(inst.rotation, vert[1]);

    var out: VsOut;
    out.pos = view.view_proj * vec4<f32>(world, 1.0);
    out.normal = normal;
    out.color = inst.color;
    out.world_pos = world;
    let w = band_weights(distance(center, view.eye.xyz));
    out.fade = vec4<f32>(w.x, w.y, 0.0, 0.0);
    out.disc = vec2<f32>(0.0, 0.0);
    return out;
}

@vertex
fn vs_impostor(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iidx: u32) -> VsOut {
    let inst = s_instances[s_visible[sp.geom.w + iidx]];
    // Two triangles, corners (-1,-1) (1,-1) (1,1) / (-1,-1) (1,1) (-1,1).
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[vidx];
    let center = sp.anchor.xyz + inst.offset;
    let r = sp.anchor.w * inst.scale;
    let world = center + view.cam_right.xyz * (c.x * r) + view.cam_up.xyz * (c.y * r);

    // A spherical normal over the disc, so the card shades as a blob of the right
    // size rather than as a flat sticker facing the camera. The z term points at
    // the eye, which is what makes the terminator run across the impostor the way
    // it ran across the mesh it replaced.
    let toward_eye = normalize(view.eye.xyz - center);
    let n = normalize(view.cam_right.xyz * c.x + view.cam_up.xyz * c.y
                      + toward_eye * sqrt(max(1.0 - min(dot(c, c), 1.0), 0.0)));

    var out: VsOut;
    out.pos = view.view_proj * vec4<f32>(world, 1.0);
    out.normal = n;
    out.color = inst.color;
    out.world_pos = world;
    let w = band_weights(distance(center, view.eye.xyz));
    out.fade = vec4<f32>(w.x, w.y, 1.0, 0.0);
    out.disc = c;
    return out;
}

// ── Lights (must match LightsUniform / MAX_LIGHTS in passes/mesh.rs) ──
const MAX_LIGHTS: u32 = 16u;

struct GpuLight {
    color: vec4<f32>,
    pos_dir: vec4<f32>, // w = kind (0 dir, 1 point, 2 spot)
    params: vec4<f32>,  // x = range, y = spot inner_cos, z = spot outer_cos
    spot_dir: vec4<f32>,
};
struct Lights {
    count: vec4<u32>,
    items: array<GpuLight, MAX_LIGHTS>,
};
@group(1) @binding(0) var<uniform> lights: Lights;

const PI: f32 = 3.14159265359;

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

    let spec = (d * g) * f / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;
    return (diffuse + spec) * radiance * n_dot_l;
}

fn point_attenuation(dist: f32, range: f32) -> f32 {
    let inv_sq = 1.0 / max(dist * dist, 1e-4);
    if (range <= 0.0) {
        return inv_sq;
    }
    let t = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
    return inv_sq * t * t;
}

// A pure-integer avalanche of the pixel coordinate → [0,1). Wang hash: no trig,
// no float reinterpretation, identical on every backend, and pinned bit-for-bit
// by `scatter::tests::dither_hash_matches_the_rust_side`.
//
// **The result is folded to 24 bits before the divide, and that is a fix.** The
// first cut returned `f32(h) * (1.0 / 4294967296.0)` over the full 32-bit word:
// `h = 0xFFFFFFFF` gives 0.99999999977, which has no f32 neighbour below 1.0 and
// rounds to **exactly 1.0**. Every test below is `h < weight` with `weight <= 1`,
// so those pixels were discarded by both the mesh and the impostor even at full
// weight — a permanent scattering of holes, one pixel in 2^32 per hash draw but
// deterministic and therefore *always in the same places*, and it falsified the
// "the common case is never taken" comment on the discard. `h >> 8` maxes at
// 16777215/16777216 = 0.99999994, which is exactly representable and strictly
// below 1, so a full-weight band now really does keep every pixel. 24 bits is
// ~16.7M dither levels against a fade band that is a few hundred pixels wide.
fn scatter_dither(px: vec2<f32>) -> f32 {
    var h = (u32(px.x) & 0xFFFFu) | ((u32(px.y) & 0xFFFFu) << 16u);
    h = (h ^ 61u) ^ (h >> 16u);
    h = h + (h << 3u);
    h = h ^ (h >> 4u);
    h = h * 0x27d4eb2du;
    h = h ^ (h >> 15u);
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let is_impostor = in.fade.z > 0.5;
    if (is_impostor) {
        // Round silhouette: the one thing a square card gets visibly wrong at the
        // distances an impostor covers.
        if (dot(in.disc, in.disc) > 1.0) {
            discard;
        }
    }

    // The complementary dither. Both tests are skipped entirely outside a fade
    // band (`m == 1` for the mesh, `m == 0 && e == 1` for the impostor), so the
    // common case pays one compare.
    let h = scatter_dither(in.pos.xy);
    if (is_impostor) {
        if (h < in.fade.x || h >= in.fade.y) {
            discard;
        }
    } else if (h >= in.fade.x) {
        discard;
    }

    if (view.flags.x > 0.5) {
        return vec4<f32>(in.color.rgb + sp.emissive.rgb, in.color.a);
    }

    let n = normalize(in.normal);
    let v = normalize(view.eye.xyz - in.world_pos);
    let albedo = in.color.rgb;
    let metallic = clamp(sp.material.x, 0.0, 1.0);
    let rough = clamp(sp.material.y, 0.04, 1.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);
    let count = lights.count.x;
    if (count == 0u) {
        lo += shade_light(n, v, normalize(view.sun_dir.xyz), vec3<f32>(3.0),
                          albedo, metallic, rough, f0);
    } else {
        for (var i = 0u; i < count && i < MAX_LIGHTS; i = i + 1u) {
            let light = lights.items[i];
            let radiance_base = light.color.rgb * light.color.a;
            if (light.pos_dir.w < 0.5) {
                lo += shade_light(n, v, normalize(light.pos_dir.xyz), radiance_base,
                                 albedo, metallic, rough, f0);
            } else {
                let to_light = light.pos_dir.xyz - in.world_pos;
                let dist = length(to_light);
                let l = to_light / max(dist, 1e-4);
                let att = point_attenuation(dist, light.params.x);
                var cone = 1.0;
                if (light.pos_dir.w > 1.5) {
                    let cos_dir = dot(l, -light.spot_dir.xyz);
                    cone = smoothstep(light.params.z, light.params.y, cos_dir);
                }
                lo += shade_light(n, v, l, radiance_base * att * cone,
                                 albedo, metallic, rough, f0);
            }
        }
    }

    let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    var amb = mix(vec3<f32>(0.03, 0.03, 0.035), vec3<f32>(0.10, 0.13, 0.18), up);
    if (gi.params.x > 0.5) {
        amb = gi_irradiance(in.world_pos, n);
    }
    let ao = textureSampleLevel(ao_tex, ao_smp, in.pos.xy / view.grid_axis_viewport.zw, 0.0).r;
    lo += amb * albedo * (1.0 - metallic) * ao;
    lo += gi_ambient_specular(in.world_pos, n, v, rough, f0, amb) * ao;
    lo += sp.emissive.rgb;

    let dist = length(in.world_pos - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.004);
    var col = mix(lo, vec3<f32>(0.055, 0.081, 0.120), haze * 0.4);
    if (atmos.params.x > 0.5) {
        col = atmos_apply(lo, in.world_pos);
    }

    return vec4<f32>(col, in.color.a);
}
