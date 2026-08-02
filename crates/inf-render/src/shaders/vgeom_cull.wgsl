// vgeom_cull.wgsl — GPU meshlet culling + LOD selection (P13.1b, two-pass P18.1).
//
// One thread per (instance, meshlet) pair. Each surviving pair is appended to a
// visible list via an atomic counter that *is* the indirect draw's
// `instance_count` (so the raster pass draws exactly the visible meshlets with a
// single `draw_indirect`). The **base cut**, in order:
//
//   1. LOD cut, CLAMPED TO RESIDENCY (P18.2) — draw iff
//      `eff_error <= t < parent_error`, where `t` is a **single per-instance
//      scalar object-space threshold** precomputed on the CPU from the
//      screen-space pixel tolerance (see passes/vgeom.rs), and
//      `eff_error = (lod_level <= floor_lod) ? 0 : error`. Using one scalar t per
//      instance is what preserves the DAG cut invariant (which requires one t per
//      mesh instance). At full residency `floor_lod == 0`, LOD 0's error is 0
//      anyway, and this is bit-identical to `VgeomMesh::select(t)`.
//   2. Frustum — the meshlet's world bounding sphere vs the 6 frustum planes.
//   3. Backface cone — meshopt normal cone: reject iff
//      `dot(normalize(center - eye), cone_axis) >= cone_cutoff`.
//
// HZB occlusion is applied **after** the base cut and is purely subtractive —
// see `occluded` below for the conservativeness proof.
//
// # Streaming indirection (P18.2)
//
// Meshlet records no longer live in a per-asset buffer: they are suballocated out
// of ONE shared pool, so a thread resolves its asset-local meshlet id through the
// per-asset `remap` table first. `remap[ml] == NOT_RESIDENT` means that meshlet's
// page is not in VRAM — the thread contributes nothing, and the *clamp* above is
// what keeps that from being a hole:
//
//   * pages are ordered coarsest-first with page 0 = every ROOT of the DAG, and
//     residency is always a prefix of that order, so every root-to-leaf path
//     always has a resident meshlet (page 0 is never evicted);
//   * zeroing the error at and below `floor_lod` makes the finest resident
//     meshlet on a path cover `[0, parent_error)` instead of yielding to the finer
//     ones that are not paged in.
//
// So partial residency changes WHICH meshlet on a path draws, never WHETHER one
// does — the same shape of guarantee two-pass occlusion makes about timing.
// `VgeomMesh::select_with_residency` is the CPU twin, and the parity gate asserts
// the two agree under punched-out residency sets.
//
// # Pass modes (P18.1)
//
// `params.counts.w` selects one of three modes; `params.misc.x` is the
// conservative flag (no usable temporal state this frame).
//
// * `MODE_SINGLE` — the v1 / fallback path: base cut, then the optional HZB test,
//   then append. Exactly the pre-P18.1 shader, and the path the CPU-parity
//   readback (`cull_visible`) drives with occlusion forced off.
// * `MODE_EARLY` — append the pairs that pass the base cut **and** were visible
//   last frame (`prev_visible[idx] != 0`, or *everything* when conservative).
//   No occlusion test: these already proved themselves, and drawing them is what
//   gives the HZB its depth.
// * `MODE_LATE` — runs over every pair again, against the HZB built from the early
//   draw. It (a) publishes `cur_visible[idx]` = "passes the base cut and is not
//   HZB-occluded", the early set for the NEXT frame, and (b) appends the pairs
//   that were *not* in this frame's early set and are not occluded — the
//   newly-disoccluded remainder.
//
// So the drawn set is `early ∪ late`, and:
//
//   early ∪ late = { base(i) ∧ E(i) } ∪ { base(i) ∧ ¬E(i) ∧ ¬occluded(i) }
//                = base \ { i : ¬E(i) ∧ occluded(i) }
//
// i.e. the split only ever removes pairs the HZB *proves* invisible. Temporal
// state (`E`) decides WHEN a meshlet draws, never WHETHER the union covers it.

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
    // z = cull flags bitset, w = pass mode (see MODE_*).
    counts: vec4<u32>,
    // x = mip count, y = hzb width, z = hzb height, w unused.
    hzb: vec4<f32>,
    // x = conservative (early set == the whole base cut), y = audit counters on,
    // z = residency floor_lod (the streaming clamp), w unused.
    misc: vec4<u32>,
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
// Last frame's visibility, one u32 per (instance, meshlet) pair.
@group(0) @binding(6) var<storage, read> prev_visible: array<u32>;
// This frame's visibility, published by MODE_LATE for next frame's early set.
@group(0) @binding(7) var<storage, read_write> cur_visible: array<u32>;
// Audit counters: [base_cut, occluded, early_drawn, late_drawn, clamped, ...].
// Eight slots so a sixth costs no layout change. Only touched when
// `params.misc.y != 0` (so the shipping path pays nothing).
@group(0) @binding(8) var<storage, read_write> stats: array<atomic<u32>, 8>;
// Asset-local meshlet id -> slot in the shared meshlet pool, or NOT_RESIDENT.
@group(0) @binding(9) var<storage, read> remap: array<u32>;

const FLAG_FRUSTUM: u32 = 1u;
const FLAG_CONE: u32 = 2u;
const FLAG_OCCLUSION: u32 = 4u;

const MODE_SINGLE: u32 = 0u;
const MODE_EARLY: u32 = 1u;
const MODE_LATE: u32 = 2u;

const AUDIT_BASE: u32 = 0u;
const AUDIT_OCCLUDED: u32 = 1u;
const AUDIT_EARLY: u32 = 2u;
const AUDIT_LATE: u32 = 3u;
// P18.2: base-cut pairs drawn COARSER than the threshold asked for, because the
// finer page is not resident — the "requested-but-missing" signal. Audit only:
// nothing reads it back to decide what to load (see stream.rs for why a GPU
// readback would make residency depend on frame history).
const AUDIT_CLAMPED: u32 = 4u;

const NOT_RESIDENT: u32 = 0xFFFFFFFFu;

// Reverse-Z HZB occlusion — **conservative by construction** (P18.1).
//
// The pyramid stores the MIN depth over each texel footprint, and mip 0 is the
// min over the MSAA subsamples, so `HZB[texel]` is the farthest surface anywhere
// under it. Let `R` be a screen rect that provably contains the meshlet's
// projection and `d_max` an upper bound on its NDC depth (reverse-Z ⇒ its
// *nearest* point). If `d_max < min_{p in R} HZB(p)` then for every pixel p and
// every subsample s covered by the meshlet, `frag_depth <= d_max < D(p, s)`, so
// every fragment fails the `Greater` depth test. The meshlet contributes **zero
// pixels** — culling it cannot change the image.
//
// Both bounds are therefore computed to over-approximate, never under:
//
// * `R` and `d_max` come from the 8 corners of the sphere's **world AABB**. A
//   perspective map takes the AABB (a convex polytope entirely in front of the
//   eye) to the convex hull of its projected corners, and sphere ⊂ AABB, so the
//   corner bbox contains the sphere's projection. Depth is likewise maximised at
//   a vertex. If ANY corner is at/behind the eye plane the projection is not
//   well-defined and we return "visible".
// * the mip is `ceil(log2(span_px))`, so one texel spans at least the rect ⇒ a
//   2×2 block anchored at the rect's min corner always covers it. Clamping to the
//   top mip only widens the footprint (the top mip is 1×1), and a wider footprint
//   has a smaller min ⇒ *less* culling. Same for clamping the rect on-screen:
//   off-screen fragments are clipped by the rasterizer anyway.
fn occluded(center: vec3<f32>, radius: f32) -> bool {
    var uv_min = vec2<f32>(1.0e30, 1.0e30);
    var uv_max = vec2<f32>(-1.0e30, -1.0e30);
    var d_max = -1.0e30;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let o = vec3<f32>(
            select(-radius, radius, (i & 1u) != 0u),
            select(-radius, radius, (i & 2u) != 0u),
            select(-radius, radius, (i & 4u) != 0u),
        );
        let clip = params.view_proj * vec4<f32>(center + o, 1.0);
        if (clip.w <= 1.0e-6) {
            return false; // straddles / behind the eye plane — treat as visible
        }
        let inv = 1.0 / clip.w;
        let uv = vec2<f32>(clip.x * inv * 0.5 + 0.5, 0.5 - clip.y * inv * 0.5);
        uv_min = min(uv_min, uv);
        uv_max = max(uv_max, uv);
        d_max = max(d_max, clip.z * inv);
    }
    // Entirely off-screen: nothing to prove (the rasterizer clips it).
    if (uv_max.x < 0.0 || uv_min.x > 1.0 || uv_max.y < 0.0 || uv_min.y > 1.0) {
        return false;
    }
    let full = vec2<f32>(params.hzb.y, params.hzb.z);
    let ext = (uv_max - uv_min) * full;
    let span = max(max(ext.x, ext.y), 1.0);
    let mip = clamp(ceil(log2(span)), 0.0, params.hzb.x - 1.0);
    let mlvl = i32(mip);
    let dim = vec2<f32>(textureDimensions(hzb_tex, mlvl));
    let maxc = vec2<i32>(dim) - vec2<i32>(1, 1);
    let lo = clamp(uv_min, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let base = clamp(vec2<i32>(floor(lo * dim)), vec2<i32>(0, 0), maxc);
    var occ = 1.0;
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let c = clamp(base + vec2<i32>(dx, dy), vec2<i32>(0, 0), maxc);
            occ = min(occ, textureLoad(hzb_tex, c, mlvl).r);
        }
    }
    return d_max < occ;
}

fn append(inst_i: u32, ml_i: u32) {
    let slot = atomicAdd(&draw_args.instance_count, 1u);
    visible[slot] = vec2<u32>(inst_i, ml_i);
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

    let mode = params.counts.w;
    // Streaming indirection: an asset-local meshlet id resolves to a slot in the
    // shared pool, or to nothing when its page is not resident.
    let slot = remap[ml_i];
    if (slot == NOT_RESIDENT) {
        if (mode == MODE_LATE) {
            cur_visible[idx] = 0u;
        }
        return;
    }
    let m = meshlets[slot];
    let flags = params.counts.z;
    let audit = params.misc.y != 0u;
    // The early set: last frame's visible pairs, or EVERYTHING when the temporal
    // state is not usable (first frame / scene version bump / camera cut), which
    // makes that frame's drawn set exactly the occlusion-off base cut.
    let early_set = params.misc.x != 0u || prev_visible[idx] != 0u;

    // ── 1. LOD cut, clamped to residency (branchless; exact parity with
    //       VgeomMesh::select_with_residency(threshold, floor_lod)).
    let floor_lod = params.misc.z;
    let eff_error = select(m.error, 0.0, m.lod_level <= floor_lod);
    var base = eff_error <= inst.threshold && inst.threshold < m.parent_error;
    // Drawn coarser than asked for: this meshlet only won because the finer page
    // is out. Audit only.
    let clamped = m.error > inst.threshold;

    let center_world = (inst.model * vec4<f32>(m.center, 1.0)).xyz;
    let radius_world = m.radius * inst.max_scale;

    // ── 2. Frustum.
    if (base && (flags & FLAG_FRUSTUM) != 0u) {
        for (var p = 0u; p < 6u; p = p + 1u) {
            let pl = params.frustum[p];
            if (dot(pl.xyz, center_world) + pl.w < -radius_world) {
                base = false;
                break;
            }
        }
    }

    // ── 3. Backface cone (skip degenerate cones, cone_cutoff >= 1).
    if (base && (flags & FLAG_CONE) != 0u && m.cone_cutoff < 1.0) {
        let nrm = mat3x3<f32>(inst.n0.xyz, inst.n1.xyz, inst.n2.xyz);
        let axis = normalize(nrm * m.cone_axis);
        let view_dir = normalize(center_world - params.eye.xyz);
        if (dot(view_dir, axis) >= m.cone_cutoff) {
            base = false;
        }
    }

    if (!base) {
        // MODE_LATE owns `cur_visible` and must publish every slot.
        if (mode == MODE_LATE) {
            cur_visible[idx] = 0u;
        }
        return;
    }

    if (mode == MODE_EARLY) {
        if (early_set) {
            append(inst_i, ml_i);
            if (audit) {
                atomicAdd(&stats[AUDIT_EARLY], 1u);
            }
        }
        return;
    }

    // ── 4. HZB occlusion (MODE_SINGLE and MODE_LATE).
    //
    // Both modes reach here having seen EVERY pair that passed the base cut, so
    // this is where `base_cut` / `occluded` are counted — the audit means the same
    // thing in either shape.
    var occ = false;
    if ((flags & FLAG_OCCLUSION) != 0u) {
        occ = occluded(center_world, radius_world);
    }
    if (audit) {
        atomicAdd(&stats[AUDIT_BASE], 1u);
        if (occ) {
            atomicAdd(&stats[AUDIT_OCCLUDED], 1u);
        }
        if (clamped) {
            atomicAdd(&stats[AUDIT_CLAMPED], 1u);
        }
    }

    if (mode == MODE_SINGLE) {
        if (!occ) {
            append(inst_i, ml_i);
            if (audit) {
                atomicAdd(&stats[AUDIT_EARLY], 1u);
            }
        }
        return;
    }

    // MODE_LATE.
    cur_visible[idx] = select(1u, 0u, occ);
    if (!early_set && !occ) {
        append(inst_i, ml_i);
        if (audit) {
            atomicAdd(&stats[AUDIT_LATE], 1u);
        }
    }
}
