// vgeom_cull.wgsl — GPU meshlet culling + LOD selection (P13.1b).
//
// One thread per (instance, meshlet) pair. Each surviving pair is appended to a
// visible list via an atomic counter that *is* the indirect draw's
// `instance_count` (so the raster pass draws exactly the visible meshlets with a
// single `draw_indexed_indirect`). Culls, in order:
//
//   1. LOD cut — draw iff `error <= t < parent_error`, where `t` is a **single
//      per-instance scalar object-space threshold** precomputed on the CPU from
//      the screen-space pixel tolerance (see passes/vgeom.rs). Using one scalar t
//      per instance makes this bit-identical to `VgeomMesh::select(t)` and
//      preserves the DAG cut invariant (which requires one t per mesh instance).
//   2. Frustum — the meshlet's world bounding sphere vs the 6 frustum planes.
//   3. Backface cone — meshopt normal cone: reject iff
//      `dot(normalize(center - eye), cone_axis) >= cone_cutoff`.
//   4. HZB occlusion (optional, v1) — the sphere's screen rect vs a min-depth
//      (reverse-Z ⇒ farthest) hierarchical depth pyramid built from the prepass.

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
    p0: u32,
    p1: u32,
    p2: u32,
};

struct CullParams {
    view_proj: mat4x4<f32>,
    frustum: array<vec4<f32>, 6>,
    eye: vec4<f32>,
    // x = total threads (instance_count * meshlet_count), y = meshlet_count,
    // z = flags bitset, w unused.
    counts: vec4<u32>,
    // x = mip count, y = hzb width, z = hzb height, w unused.
    hzb: vec4<f32>,
};

// Non-indexed indirect draw args (vertex pulling ⇒ no index buffer). The atomic
// `instance_count` IS the visible-meshlet counter: `draw_indirect` then draws
// exactly the appended visible meshlets.
struct DrawArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> params: CullParams;
@group(0) @binding(1) var<storage, read> meshlets: array<Meshlet>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read_write> visible: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> draw_args: DrawArgs;
@group(0) @binding(5) var hzb_tex: texture_2d<f32>;

const FLAG_FRUSTUM: u32 = 1u;
const FLAG_CONE: u32 = 2u;
const FLAG_OCCLUSION: u32 = 4u;

// Reverse-Z HZB occlusion. The pyramid stores the MIN depth (= farthest surface,
// since reverse-Z maps near→1, far→0) over each texel footprint. The sphere is
// occluded iff its nearest point (its LARGEST depth) is still behind the farthest
// occluder across the covered footprint — i.e. every occluder there is closer.
fn occluded(center: vec3<f32>, radius: f32) -> bool {
    let clip = params.view_proj * vec4<f32>(center, 1.0);
    if (clip.w <= 0.0) {
        return false; // straddles / behind the eye — treat as visible
    }
    let inv_w = 1.0 / clip.w;
    let ndc = clip.xyz * inv_w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return false;
    }
    // Sphere's nearest-point depth (reverse-Z ⇒ largest depth value).
    let to_eye = normalize(params.eye.xyz - center);
    let nclip = params.view_proj * vec4<f32>(center + to_eye * radius, 1.0);
    var sphere_depth = ndc.z;
    if (nclip.w > 0.0) {
        sphere_depth = nclip.z / nclip.w;
    }
    // Approximate screen-space pixel radius from a tangent offset projection.
    let w = params.hzb.y;
    let h = params.hzb.z;
    let eclip = params.view_proj * vec4<f32>(center + vec3<f32>(radius, 0.0, 0.0), 1.0);
    var r_px = 2.0;
    if (eclip.w > 0.0) {
        let euv = vec2<f32>((eclip.x / eclip.w) * 0.5 + 0.5, 0.5 - (eclip.y / eclip.w) * 0.5);
        r_px = max(abs(euv.x - uv.x) * w, abs(euv.y - uv.y) * h);
    }
    let mip = clamp(ceil(log2(max(r_px, 1.0))), 0.0, params.hzb.x - 1.0);
    let mlvl = i32(mip);
    let dim = max(vec2<f32>(w, h) / exp2(mip), vec2<f32>(1.0, 1.0));
    let base = vec2<i32>(uv * dim);
    let maxc = vec2<i32>(dim) - vec2<i32>(1, 1);
    var occ = 1.0;
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let c = clamp(base + vec2<i32>(dx, dy), vec2<i32>(0, 0), maxc);
            occ = min(occ, textureLoad(hzb_tex, c, mlvl).r);
        }
    }
    return sphere_depth < occ;
}

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.counts.x;
    if (idx >= total) {
        return;
    }
    let meshlet_count = params.counts.y;
    let inst_i = idx / meshlet_count;
    let ml_i = idx % meshlet_count;
    let inst = instances[inst_i];
    let m = meshlets[ml_i];

    // 1. LOD cut (branchless; exact parity with VgeomMesh::select(threshold)).
    let t = inst.threshold;
    if (!(m.error <= t && t < m.parent_error)) {
        return;
    }

    let flags = params.counts.z;
    let center_world = (inst.model * vec4<f32>(m.center, 1.0)).xyz;
    let radius_world = m.radius * inst.max_scale;

    // 2. Frustum.
    if ((flags & FLAG_FRUSTUM) != 0u) {
        for (var p = 0u; p < 6u; p = p + 1u) {
            let pl = params.frustum[p];
            if (dot(pl.xyz, center_world) + pl.w < -radius_world) {
                return;
            }
        }
    }

    // 3. Backface cone (skip degenerate cones, cone_cutoff >= 1).
    if ((flags & FLAG_CONE) != 0u && m.cone_cutoff < 1.0) {
        let nrm = mat3x3<f32>(inst.n0.xyz, inst.n1.xyz, inst.n2.xyz);
        let axis = normalize(nrm * m.cone_axis);
        let view_dir = normalize(center_world - params.eye.xyz);
        if (dot(view_dir, axis) >= m.cone_cutoff) {
            return;
        }
    }

    // 4. HZB occlusion (optional).
    if ((flags & FLAG_OCCLUSION) != 0u) {
        if (occluded(center_world, radius_world)) {
            return;
        }
    }

    let slot = atomicAdd(&draw_args.instance_count, 1u);
    visible[slot] = vec2<u32>(inst_i, ml_i);
}
