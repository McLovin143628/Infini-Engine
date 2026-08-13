// vgeom_mesh.wgsl — vertex-pulled meshlet raster (P13.1b). `common_view.wgsl` is
// prepended (the shared `View` uniform + helpers).
//
// No vertex buffers: the cull compute produced a `visible` list of (instance,
// meshlet) pairs and an indirect draw of `visible_count` instances, each drawing
// `MAX_TRIS*3` indices from a shared `0,1,2,…` index buffer. This shader pulls
// everything from storage:
//   * @builtin(instance_index) → visible[i] → (global instance, asset-local
//     meshlet id) → v_remap[id] → the meshlet's slot in the SHARED pool (P18.2:
//     meshlet/vertex/micro-index data is suballocated across all assets, so every
//     read goes through the per-asset remap table the cull compute also reads)
//   * @builtin(vertex_index)   → triangle = vidx/3, corner = vidx%3
//   * unused triangles (corner beyond the meshlet's triangle_count) collapse to a
//     zero-area (degenerate) triangle off-screen — the standard fixed-index-count
//     instanced-meshlet technique. Cost: up to (MAX_TRIS - triangle_count)*3
//     degenerate vertices per meshlet (documented in passes/vgeom.rs).
// The fragment stage is the same metallic-roughness PBR as mesh.wgsl (duplicated
// so the rigid pipeline — and every existing golden — stays byte-stable).

struct Meshlet {
    center: vec3<f32>,
    radius: f32,
    cone_axis: vec3<f32>,
    cone_cutoff: f32,
    vertex_offset: u32,
    triangle_offset: u32,
    vertex_count: u32,
    triangle_count: u32,
    error: f32,
    parent_error: f32,
    lod_level: u32,
    pad: u32,
};

struct Instance {
    model: mat4x4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
    color: vec4<f32>,
    emissive: vec4<f32>,
    threshold: f32,
    metallic: f32,
    roughness: f32,
    max_scale: f32,
    pick_id: u32,
    // P26.3: the virtual-texture set (albedo, normal, ORM slots), each a
    // handle + 1. These three words were `p0`/`p1`/`p2` — alignment padding
    // uploaded as zero since P13.1b — so a meshlet instance that samples
    // nothing packs to the bytes it always did.
    vt0: u32,
    vt1: u32,
    vt2: u32,
};

// Group 3: the virtualized-geometry storage buffers.
@group(3) @binding(0) var<storage, read> v_positions: array<f32>; // 8 f32 / vertex
@group(3) @binding(1) var<storage, read> v_meshlets: array<Meshlet>;
@group(3) @binding(2) var<storage, read> v_meshlet_verts: array<u32>;
@group(3) @binding(3) var<storage, read> v_meshlet_tris: array<u32>; // packed u8
@group(3) @binding(4) var<storage, read> v_instances: array<Instance>;
@group(3) @binding(5) var<storage, read> v_visible: array<vec2<u32>>;
// Asset-local meshlet id -> slot in the shared meshlet pool (P18.2).
@group(3) @binding(7) var<storage, read> v_remap: array<u32>;

const NOT_RESIDENT: u32 = 0xFFFFFFFFu;

// Debug flag pushed via the view uniform's spare slot is awkward; instead we bake
// a specialization through a small uniform. Reuse mode: a dedicated tiny uniform.
struct VgeomFlags {
    // x = debug meshlet colouring (0/1), yzw unused.
    flags: vec4<u32>,
};
@group(3) @binding(6) var<uniform> vg: VgeomFlags;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) @interpolate(flat) id: u32,
    @location(4) @interpolate(flat) pbr: vec4<f32>,
    @location(5) @interpolate(flat) emissive: vec3<f32>,
    @location(6) @interpolate(flat) meshlet: u32,
    // P26.3: the meshlet path reads the mesh's REAL uv. `v_positions` has
    // carried 8 floats per vertex since P13.1b — position, normal, and a uv
    // pair nothing read until now.
    @location(7) uv: vec2<f32>,
    @location(8) @interpolate(flat) vt: vec3<u32>,
};

fn read_tri_byte(byte_addr: u32) -> u32 {
    let word = v_meshlet_tris[byte_addr >> 2u];
    let shift = (byte_addr & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

@vertex
fn vs(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iidx: u32) -> VsOut {
    var out: VsOut;
    let pair = v_visible[iidx];
    let inst = v_instances[pair.x];
    let slot = v_remap[pair.y];

    let tri = vidx / 3u;
    let corner = vidx % 3u;

    // The cull compute only ever appends resident pairs, so this cannot fire —
    // but a stale visible list must degenerate rather than read a freed pool slot.
    if (slot == NOT_RESIDENT) {
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0);
        out.pbr = vec4<f32>(0.0);
        return out;
    }
    let m = v_meshlets[slot];

    // Degenerate padding for unused triangles: emit a fixed off-screen point (all
    // three corners of the unused triangle map here ⇒ zero area ⇒ no fragments).
    if (tri >= m.triangle_count) {
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0);
        out.pbr = vec4<f32>(0.0);
        return out;
    }

    let local = read_tri_byte(m.triangle_offset + tri * 3u + corner);
    let global_v = v_meshlet_verts[m.vertex_offset + local];
    let base = global_v * 8u;
    let position = vec3<f32>(v_positions[base], v_positions[base + 1u], v_positions[base + 2u]);
    let normal = vec3<f32>(v_positions[base + 3u], v_positions[base + 4u], v_positions[base + 5u]);
    let uv = vec2<f32>(v_positions[base + 6u], v_positions[base + 7u]);

    let nrm = mat3x3<f32>(inst.n0.xyz, inst.n1.xyz, inst.n2.xyz);
    let wp = inst.model * vec4<f32>(position, 1.0);
    out.pos = view.view_proj * wp;
    out.world_pos = wp.xyz;
    out.normal = nrm * normal;
    out.color = inst.color;
    out.id = inst.pick_id;
    out.pbr = vec4<f32>(inst.metallic, inst.roughness, 0.0, 0.0);
    out.emissive = inst.emissive.rgb;
    out.meshlet = pair.y;
    out.uv = uv;
    out.vt = vec3<u32>(inst.vt0, inst.vt1, inst.vt2);
    return out;
}

// ── Lights (must match LightsUniform / MAX_LIGHTS in passes/mesh.rs) ──
const MAX_LIGHTS: u32 = 16u;

struct GpuLight {
    color: vec4<f32>,
    pos_dir: vec4<f32>, // w = kind (0 dir, 1 point, 2 spot)
    params: vec4<f32>,  // x = range, y = spot inner_cos, z = spot outer_cos
    spot_dir: vec4<f32>, // xyz = normalized spot emission direction (spot only)
};
struct Lights {
    count: vec4<u32>,
    items: array<GpuLight, MAX_LIGHTS>,
};
@group(1) @binding(0) var<uniform> lights: Lights;

// AO + cascaded shadows + dynamic GI + the atmosphere ride the shared env bind
// group at @group(2) (declared in env_lighting.wgsl / atmosphere*.wgsl, prepended
// by `lit_scene_shader` — P17.2 promoted this pass from the AO-only bind to the
// same env bind every other lit pass uses, so aerial perspective reaches meshlet
// geometry too).

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

// A distinct, stable per-meshlet debug colour (hash → hue-ish rgb).
fn meshlet_color(id: u32) -> vec3<f32> {
    let h = (id * 2654435761u) ^ (id << 13u);
    let r = f32((h >> 16u) & 0xFFu) / 255.0;
    let g = f32((h >> 8u) & 0xFFu) / 255.0;
    let b = f32(h & 0xFFu) / 255.0;
    // Lift out of the very-dark corner so clusters read clearly.
    return vec3<f32>(0.15) + 0.85 * vec3<f32>(r, g, b);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    if (vg.flags.x != 0u) {
        // Flat per-meshlet debug colouring (LOD/cluster structure proof).
        return vec4<f32>(meshlet_color(in.meshlet), 1.0);
    }
    // P26.5 RESIDENCY HEAT-MAP (`ViewMode::VtResidency`): every virtual-textured
    // surface painted by how far behind the streamer is at that pixel.
    //
    // **Before the unlit short-circuit, not after.** `VtResidency` sets
    // `flags.x` too (the `Biomes` precedent: everything that is not the overlay
    // renders flat so the ramp is readable), so a heat branch placed below the
    // unlit return would never execute — measured, and it is the whole defect a
    // real-frame arm catches and a source read does not.
    //
    // `flags.z` is 0.0 in every other mode, so this is present-and-false and
    // every golden runs the identical arithmetic. The derivatives are taken
    // here, in uniform control flow: `flags.z` is a uniform, so the return is
    // uniform, and `in.uv` is an interpolated input either way.
    if (view.flags.z > 0.5) {
        return vec4<f32>(vt_heat(in.vt, in.uv, dpdx(in.uv), dpdy(in.uv)), 1.0);
    }
    // Unlit view mode (R-P2): albedo + emissive, no lighting (Unlit + Wireframe).
    if (view.flags.x > 0.5) {
        return vec4<f32>(in.color.rgb + in.emissive, in.color.a);
    }

    var n = normalize(in.normal);
    let v = normalize(view.eye.xyz - in.world_pos);
    var albedo = in.color.rgb;
    var metallic = clamp(in.pbr.x, 0.0, 1.0);
    var rough = clamp(in.pbr.y, 0.04, 1.0);
    // P26.3 VIRTUAL TEXTURES. Zero on every instance that names no texture, so
    // the branch is present-and-false for every scene that predates this batch
    // and the arithmetic below is byte-identical.
    // The screen derivatives, taken in UNIFORM control flow — a fragment shader
    // may only difference against its neighbours outside a divergent branch, and
    // the VT branch below is per instance. Cheap when nothing samples: a
    // derivative of an interpolated value is two subtractions.
    let vt_ddx = dpdx(in.uv);
    let vt_ddy = dpdy(in.uv);
    let vt_dpx = dpdx(in.world_pos);
    let vt_dpy = dpdy(in.world_pos);
    var vt_ao = 1.0;
    if (in.vt.x != 0u || in.vt.y != 0u || in.vt.z != 0u) {
        let s = vt_surface(in.vt, in.uv, vt_ddx, vt_ddy,
                           albedo, in.color.a, metallic, rough);
        albedo = s.albedo;
        metallic = clamp(s.metallic, 0.0, 1.0);
        rough = clamp(s.roughness, 0.04, 1.0);
        vt_ao = s.occlusion;
        if (s.has_normal) {
            n = vt_apply_normal(n, vt_dpx, vt_dpy, vt_ddx, vt_ddy, s.normal_ts);
        }
    }
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
                // Point (w == 1) / spot (w == 2); `cone` stays 1.0 for a point
                // light, so the point path is byte-stable.
                let to_light = light.pos_dir.xyz - in.world_pos;
                let dist = length(to_light);
                let l = to_light / max(dist, 1e-4);
                let att = point_attenuation(dist, light.params.x);
                var cone = 1.0;
                if (light.pos_dir.w > 1.5) {
                    let cos_dir = dot(l, -light.spot_dir.xyz);
                    cone = smoothstep(light.params.z, light.params.y, cos_dir);
                }
                // **P27.4: the engine's FIRST point/spot shadows.** A spot
                // resolves through its single quadtree, a point through the
                // cube-face quadtree its own direction selects. `params.w` is 0
                // on every light without a page tree — which is every light of
                // every scene with virtual shadows off — and `vsm_light_shadow`
                // returns exactly 1.0 there, so this is a present-and-inert
                // `* 1.0` on every committed golden.
                let vsm_f = vsm_light_shadow(in.world_pos, n, light.params.w);
                lo += shade_light(n, v, l, radiance_base * att * cone * vsm_f,
                                 albedo, metallic, rough, f0);
            }
        }
    }

    let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    // P18.4: meshlet geometry now both CONTRIBUTES to the GI volume (its root-page
    // meshlet spheres are voxelized) and RECEIVES from it. Before this it took the
    // hemispheric constant even with GI on — a gap, since the rigid path had read
    // probes since P13.3b. The branch is not taken with GI off, so every vgeom
    // golden stays byte-identical.
    var amb = mix(vec3<f32>(0.03, 0.03, 0.035), vec3<f32>(0.10, 0.13, 0.18), up);
    if (gi.params.x > 0.5) {
        amb = gi_irradiance(in.world_pos, n);
    }
    // The material's own occlusion map multiplies the screen-space AO: both
    // modulate the AMBIENT term only, never the direct light above.
    let ao = textureSampleLevel(ao_tex, ao_smp, in.pos.xy / view.grid_axis_viewport.zw, 0.0).r
        * vt_ao;
    lo += amb * albedo * (1.0 - metallic) * ao;
    lo += gi_ambient_specular(in.world_pos, n, v, rough, f0, amb) * ao;

    lo += in.emissive;

    let dist = length(in.world_pos - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.004);
    var col = mix(lo, vec3<f32>(0.055, 0.081, 0.120), haze * 0.4);
    if (atmos.params.x > 0.5) {
        col = atmos_apply(lo, in.world_pos);
    }

    return vec4<f32>(col, in.color.a);
}

@fragment
fn fs_id(in: VsOut) -> @location(0) u32 {
    return in.id;
}
