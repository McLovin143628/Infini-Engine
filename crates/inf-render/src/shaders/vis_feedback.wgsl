// vis_feedback.wgsl — PER-FRAGMENT virtual-texture feedback (P28.1), the
// retirement of `docs/memos/p26-4-feedback-mechanism.md`'s deviation.
//
// That memo marks per SURFACE: one thread per (drawn surface x bound map)
// frustum-culls a bounding sphere, derives the mip its projected footprint
// justifies, and marks every tile of that level. It says exactly what that
// costs — occlusion, uv extent, sub-surface variation — and exactly why the
// textbook fragment-side mark was unreachable: it needs a WRITABLE storage
// buffer visible to the FRAGMENT stage, which means
// `DownlevelFlags::FRAGMENT_WRITABLE_STORAGE` becomes a requirement of every lit
// pass on every adapter, on a machine that cannot falsify it.
//
// The visibility buffer dissolves the objection rather than working around it.
// This is a COMPUTE pass with a bind group of its own: nothing it declares
// reaches the shared environment layout, no lit pipeline acquires a writable
// storage binding, and no adapter capability moves. And it is per fragment by
// construction — one thread per PIXEL, reading the id the depth test left
// there — so all three of the memo's losses are recovered for the meshlet set:
//
//   * **occlusion** — a wall in front of a textured floor writes its own id, so
//     the floor's tiles are not asked for at that pixel;
//   * **uv extent** — the tile marked is the tile the pixel's uv lands in, not
//     every tile of the level;
//   * **sub-surface variation** — a 200 m quad marks a different level per
//     screen region, because the mip comes from THIS pixel's analytic gradients.
//
// The floor is untouched: `analytic_floor` still runs at `VT_PRIORITY_FLOOR`, so
// a dropped, late or never-arriving mask still degrades to exactly the analytic
// residency. Everything the determinism argument rests on is unchanged — one
// `atomicOr` of a constant bit into `inf_vt::feedback`'s layout, scanned in
// virtual-address order, read at a pinned latency.

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
    vt0: u32,
    vt1: u32,
    vt2: u32,
};

// `view_proj` is the JITTERED matrix when TAA is on — the renderer writes the
// same `jvp` the visibility buffer was rasterized with, because a reconstruction
// on a different sub-pixel grid than its own depth is the P27.1 defect.
// `counts` = (viewport w, viewport h, mask words, texture count).
struct VisFeedbackParams {
    view_proj: mat4x4<f32>,
    counts: vec4<u32>,
};

@group(0) @binding(0) var<uniform> params: VisFeedbackParams;
@group(0) @binding(1) var vis_buf: texture_2d<u32>;
@group(0) @binding(2) var<storage, read> v_positions: array<f32>;
@group(0) @binding(3) var<storage, read> v_meshlets: array<Meshlet>;
@group(0) @binding(4) var<storage, read> v_meshlet_verts: array<u32>;
@group(0) @binding(5) var<storage, read> v_meshlet_tris: array<u32>;
@group(0) @binding(6) var<storage, read> v_instances: array<Instance>;
@group(0) @binding(7) var<storage, read> vt_table: array<u32>;
// Per texture HANDLE: x = first mask bit (`VtFeedbackLayout::texture_base`).
// The block word offset is already in the table, so only the bit base needs a
// second table — which is why this is one `u32` and not two.
@group(0) @binding(8) var<storage, read> vt_bases: array<u32>;
@group(0) @binding(9) var<storage, read_write> mask: array<atomic<u32>>;

// ── the packing contract (crates/inf-render/src/visbuffer.rs) ────────────────
const VIS_TRI_BITS: u32 = 7u;
const VIS_MESHLET_BITS: u32 = 14u;
const VIS_INSTANCE_BITS: u32 = 11u;
const VIS_EMPTY: u32 = 0u;
const VIS_MESHLET_SHIFT: u32 = VIS_TRI_BITS;
const VIS_INSTANCE_SHIFT: u32 = VIS_TRI_BITS + VIS_MESHLET_BITS;

// `inf_vt::table`'s layout — the same four constants `vt_sample.wgsl` and
// `vt_feedback.wgsl` carry.
const VT_MAGIC: u32 = 0x31425456u;
const VT_HEADER_WORDS: u32 = 4u;
const VT_TEX_HEADER_WORDS: u32 = 4u;
const VT_MIP_REC_WORDS: u32 = 4u;

fn read_tri_byte(byte_addr: u32) -> u32 {
    let word = v_meshlet_tris[byte_addr >> 2u];
    let shift = (byte_addr & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

fn tiles_down(height: u32, tile_size: u32) -> u32 {
    return max((height + tile_size - 1u) / tile_size, 1u);
}

/// The mip a pixel's own gradients justify. Textually `vt_sample.wgsl`'s
/// `vt_mip`, and the arm `the_per_pixel_mark_uses_the_same_mip_rule` pins the
/// two against each other — a feedback pass that asked for a different level
/// than the sampler reads is a streamer chasing tiles nothing samples.
fn vis_vt_mip(block: u32, ddx: vec2<f32>, ddy: vec2<f32>) -> u32 {
    let mip_count = vt_table[block];
    let w = vt_table[block + VT_TEX_HEADER_WORDS];
    let h = vt_table[block + VT_TEX_HEADER_WORDS + 1u];
    let size = vec2<f32>(f32(w), f32(h));
    let dx = ddx * size;
    let dy = ddy * size;
    let rho = max(length(dx), length(dy));
    let lod = max(log2(max(rho, 1e-8)), 0.0);
    return min(u32(lod), max(mip_count, 1u) - 1u);
}

/// Mark the one tile `uv` lands in at level `m`. The bit is
/// `texture_base + entry_index`, and `entry_index` is derived from the tile
/// directory's own offset exactly as `vt_feedback.wgsl` derives it — never
/// re-derived by halving, which `vt_sample.wgsl`'s header measured at 1 322 bad
/// addresses on a 4095-square pyramid.
fn vis_mark_tile(slot: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) {
    if (slot == 0u) {
        return;
    }
    let handle = slot - 1u;
    if (handle >= params.counts.w) {
        return;
    }
    if (arrayLength(&vt_table) < VT_HEADER_WORDS || vt_table[0] != VT_MAGIC) {
        return;
    }
    if (handle >= vt_table[1]) {
        return;
    }
    let b = vt_table[VT_HEADER_WORDS + handle];
    let mip_count = vt_table[b];
    let tile_size = vt_table[b + 1u];
    if (mip_count == 0u || tile_size == 0u) {
        return;
    }
    let m = vis_vt_mip(b, ddx, ddy);
    let rec = b + VT_TEX_HEADER_WORDS + m * VT_MIP_REC_WORDS;
    let mip_w = vt_table[rec];
    let mip_h = vt_table[rec + 1u];
    let tiles_x = vt_table[rec + 2u];
    let tiles_y = tiles_down(mip_h, tile_size);
    if (tiles_x == 0u || mip_w == 0u) {
        return;
    }
    // The same wrap `vt_sample` applies, and after the gradients were taken.
    let w_uv = uv - floor(uv);
    let gp = w_uv * vec2<f32>(f32(mip_w), f32(mip_h));
    let tx = min(u32(gp.x) / tile_size, tiles_x - 1u);
    let ty = min(u32(gp.y) / tile_size, tiles_y - 1u);

    let entries_base = b + VT_TEX_HEADER_WORDS + mip_count * VT_MIP_REC_WORDS;
    let first_entry = vt_table[rec + 3u] - entries_base;
    let bit = vt_bases[handle] + first_entry + ty * tiles_x + tx;
    let word = bit / 32u;
    if (word < params.counts.z) {
        atomicOr(&mask[word], 1u << (bit % 32u));
    }
}

const VIS_DET_EPS: f32 = 1e-20;

fn mat3x3_inverse(m: mat3x3<f32>, det: f32) -> mat3x3<f32> {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let r0 = cross(b, c);
    let r1 = cross(c, a);
    let r2 = cross(a, b);
    let inv_det = 1.0 / det;
    return mat3x3<f32>(
        vec3<f32>(r0.x, r1.x, r2.x) * inv_det,
        vec3<f32>(r0.y, r1.y, r2.y) * inv_det,
        vec3<f32>(r0.z, r1.z, r2.z) * inv_det,
    );
}

@compute @workgroup_size(8, 8, 1)
fn cs_feedback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.counts.x;
    let h = params.counts.y;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let id = textureLoad(vis_buf, vec2<i32>(i32(gid.x), i32(gid.y)), 0).r;
    if (id == VIS_EMPTY) {
        return;
    }
    let instance_id = (id >> VIS_INSTANCE_SHIFT) - 1u;
    let meshlet_id = (id >> VIS_MESHLET_SHIFT) & ((1u << VIS_MESHLET_BITS) - 1u);
    let tri = id & ((1u << VIS_TRI_BITS) - 1u);

    let inst = v_instances[instance_id];
    if (inst.vt0 == 0u && inst.vt1 == 0u && inst.vt2 == 0u) {
        return;
    }
    let m = v_meshlets[meshlet_id];

    var uv3: array<vec2<f32>, 3>;
    var clip: array<vec4<f32>, 3>;
    for (var k = 0u; k < 3u; k = k + 1u) {
        let local = read_tri_byte(m.triangle_offset + tri * 3u + k);
        let global_v = v_meshlet_verts[m.vertex_offset + local];
        let base = global_v * 8u;
        let p = vec3<f32>(v_positions[base], v_positions[base + 1u], v_positions[base + 2u]);
        uv3[k] = vec2<f32>(v_positions[base + 6u], v_positions[base + 7u]);
        clip[k] = params.view_proj * (inst.model * vec4<f32>(p, 1.0));
    }

    let wh = vec2<f32>(f32(w), f32(h));
    let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let ndc = vec2<f32>(2.0 * px.x / wh.x - 1.0, 1.0 - 2.0 * px.y / wh.y);

    let mm = mat3x3<f32>(
        vec3<f32>(clip[0].x, clip[0].y, clip[0].w),
        vec3<f32>(clip[1].x, clip[1].y, clip[1].w),
        vec3<f32>(clip[2].x, clip[2].y, clip[2].w),
    );
    let det = determinant(mm);
    if (abs(det) < VIS_DET_EPS) {
        return;
    }
    let inv = mat3x3_inverse(mm, det);
    let mu = inv * vec3<f32>(ndc.x, ndc.y, 1.0);
    let dmu_du = inv * vec3<f32>(1.0, 0.0, 0.0);
    let dmu_dv = inv * vec3<f32>(0.0, 1.0, 0.0);
    let d = mu.x + mu.y + mu.z;
    if (abs(d) < VIS_DET_EPS) {
        return;
    }
    let inv_d = 1.0 / d;
    let l = mu * inv_d;
    let su = dmu_du.x + dmu_du.y + dmu_du.z;
    let sv = dmu_dv.x + dmu_dv.y + dmu_dv.z;
    let d_dx = ((dmu_du - l * su) * inv_d) * (2.0 / max(wh.x, 1.0));
    let d_dy = ((dmu_dv - l * sv) * inv_d) * (-2.0 / max(wh.y, 1.0));

    let uv = uv3[0] * l.x + uv3[1] * l.y + uv3[2] * l.z;
    let ddx = uv3[0] * d_dx.x + uv3[1] * d_dx.y + uv3[2] * d_dx.z;
    let ddy = uv3[0] * d_dy.x + uv3[1] * d_dy.y + uv3[2] * d_dy.z;

    vis_mark_tile(inst.vt0, uv, ddx, ddy);
    vis_mark_tile(inst.vt1, uv, ddx, ddy);
    vis_mark_tile(inst.vt2, uv, ddx, ddy);
}
