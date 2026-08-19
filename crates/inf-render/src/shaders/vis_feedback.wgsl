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
    // **The material slot** (Wave T, the texture document's section 3 C). This
    // word carries `MeshletRec::group` on disk and was `pad` here because no
    // shader read it; `inf_vgeom::stream::stage_page` now overwrites it with the
    // meshlet's material slot on the way into the pool, unconditionally, so the
    // word means one thing on the GPU for every container version. `0` is the
    // instance's own material, which is what every mesh cooked before Wave T
    // resolves to.
    material_slot: u32,
};

// The flat instance table rides an `Rgba32Float` TEXTURE, one row an instance,
// eleven used texels of sixteen. Not a storage buffer, and the reason is a
// measured device ceiling rather than taste: `Limits::default()` grants eight
// storage buffers a stage, the shared environment group already spends four (GI
// SH, the VT table, the VSM table, its projections) and the four meshlet pools
// are the other four. The ninth is refused by `create_pipeline_layout`. See
// `VIS_FRAGMENT_STORAGE_BINDINGS` in `crates/inf-render/src/passes/visbuffer.rs`.
//
// `textureLoad` is valid in the vertex, fragment AND compute stages, so all
// three visibility passes read the table through one representation.
const VIS_INSTANCE_TEXELS: i32 = 16;

struct Instance {
    model: mat4x4<f32>,
    n0: vec3<f32>,
    n1: vec3<f32>,
    n2: vec3<f32>,
    color: vec4<f32>,
    emissive: vec3<f32>,
    metallic: f32,
    roughness: f32,
    vt: vec3<u32>,
};

fn vis_instance(id: u32) -> Instance {
    let y = i32(id);
    let c0 = textureLoad(v_instance_tex, vec2<i32>(0, y), 0);
    let c1 = textureLoad(v_instance_tex, vec2<i32>(1, y), 0);
    let c2 = textureLoad(v_instance_tex, vec2<i32>(2, y), 0);
    let c3 = textureLoad(v_instance_tex, vec2<i32>(3, y), 0);
    let n0 = textureLoad(v_instance_tex, vec2<i32>(4, y), 0);
    let n1 = textureLoad(v_instance_tex, vec2<i32>(5, y), 0);
    let n2 = textureLoad(v_instance_tex, vec2<i32>(6, y), 0);
    let col = textureLoad(v_instance_tex, vec2<i32>(7, y), 0);
    let emi = textureLoad(v_instance_tex, vec2<i32>(8, y), 0);
    // (threshold, metallic, roughness, max_scale)
    let pbr = textureLoad(v_instance_tex, vec2<i32>(9, y), 0);
    // (pick_id, vt0, vt1, vt2) — u32 words, so they are bitcast rather than
    // converted. `bitcast` is the right call and the FIRST version of this
    // comment gave the wrong reason for it: it said an f32 round-trip breaks
    // "anything past 2^24", which is where the danger is NOT. A virtual-texture
    // slot is `handle + 1`, a small integer, and every u32 below 2^23 has an
    // all-zero f32 exponent field — so as a float every real handle here is
    // SUBNORMAL, and WGSL permits an implementation to flush subnormals to zero.
    // Converting would turn slot 1 into slot 0, which reads as "this surface
    // binds no texture" and shows up as a silently untextured frame rather than
    // as an error (P28.1 audit). A bitcast of a loaded texel is a
    // reinterpretation and takes no arithmetic, which is why it survives;
    // `parity_textured_virtual_texture`'s closing `assert_ne!` is what witnesses
    // that on whatever adapter runs, because a flushed slot makes the textured
    // frame byte-identical to the untextured one.
    let ids = textureLoad(v_instance_tex, vec2<i32>(10, y), 0);
    var out: Instance;
    out.model = mat4x4<f32>(c0, c1, c2, c3);
    out.n0 = n0.xyz;
    out.n1 = n1.xyz;
    out.n2 = n2.xyz;
    out.color = col;
    out.emissive = emi.xyz;
    out.metallic = pbr.y;
    out.roughness = pbr.z;
    out.vt = vec3<u32>(bitcast<u32>(ids.y), bitcast<u32>(ids.z), bitcast<u32>(ids.w));
    return out;
}

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
// **The meshlet vertex record's stride, in floats** (P28.2). One `VgeomVertex`
// is position(3) + normal(3) + uv(2) + a packed tangent word(1) = 9, and this
// constant is the Rust `VERTEX_REC_LEN / 4` read back — `the_shaders_vertex_
// stride_is_the_rusts` fails the day the two stop agreeing, in every shader that
// reads the pool. A wrong stride here does not error: it reads a neighbouring
// vertex's bytes as this one's position, which is a mesh that renders and is
// wrong everywhere.
const VGEOM_VSTRIDE: u32 = 9u;
@group(0) @binding(2) var<storage, read> v_positions: array<f32>;
@group(0) @binding(3) var<storage, read> v_meshlets: array<Meshlet>;
@group(0) @binding(4) var<storage, read> v_meshlet_verts: array<u32>;
@group(0) @binding(5) var<storage, read> v_meshlet_tris: array<u32>;
@group(0) @binding(6) var v_instance_tex: texture_2d<f32>;
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
///
/// **The argument is a per-instance map WORD, not a bare slot** (Wave-T audit).
/// Wave T packed a detail lane into the top half of two of these words, and this
/// function read the whole word — so a surface that bound a detail map computed
/// `handle = (albedo | detail << 16) - 1`, failed the range check below, and
/// **stopped marking its own albedo**. On the visbuffer path this producer is
/// the only per-fragment one there is, so that surface stopped refining
/// entirely. The mask is the same one `vt_sample.wgsl`'s `VT_SLOT_MASK` applies,
/// and it is applied here for the same reason.
fn vis_mark_tile(maps_word: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) {
    let slot = maps_word & 0xFFFFu;
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

    let inst = vis_instance(instance_id);
    if (inst.vt.x == 0u && inst.vt.y == 0u && inst.vt.z == 0u) {
        return;
    }
    let m = v_meshlets[meshlet_id];

    var uv3: array<vec2<f32>, 3>;
    var clip: array<vec4<f32>, 3>;
    for (var k = 0u; k < 3u; k = k + 1u) {
        let local = read_tri_byte(m.triangle_offset + tri * 3u + k);
        let global_v = v_meshlet_verts[m.vertex_offset + local];
        let base = global_v * VGEOM_VSTRIDE;
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
    // The NaN clause the resolve carries, for the same reason: a comparison
    // against NaN is FALSE, so `abs(det) < VIS_DET_EPS` alone would let a NaN
    // triangle reach `vis_mark_tile`, where `u32(gp.x)` of a NaN is an
    // implementation-defined tile index (P28.1 audit).
    if (abs(det) < VIS_DET_EPS || det != det) {
        return;
    }
    let inv = mat3x3_inverse(mm, det);
    let mu = inv * vec3<f32>(ndc.x, ndc.y, 1.0);
    let dmu_du = inv * vec3<f32>(1.0, 0.0, 0.0);
    let dmu_dv = inv * vec3<f32>(0.0, 1.0, 0.0);
    let d = mu.x + mu.y + mu.z;
    if (abs(d) < VIS_DET_EPS || d != d) {
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

    // **The detail map is marked too, at its own tiling** (Wave-T audit). It
    // rides the top half of the first two words — slot in x, 8.8 scale in y,
    // exactly as `vt_sample.wgsl`'s `vt_detail_slot` / `vt_detail_scale` read
    // them — and `vt_apply_detail` samples it at `uv * scale`, so it must be
    // marked at `uv * scale` or the streamer pages the wrong tiles of it. Zero
    // scale (which is every instance written before Wave T) marks nothing,
    // because `vis_mark_tile` refuses slot 0 and the slot is zero with it.
    let detail_scale = f32(inst.vt.y >> 16u) / 256.0;
    if (detail_scale > 0.0) {
        vis_mark_tile(
            inst.vt.x >> 16u,
            uv * detail_scale,
            ddx * detail_scale,
            ddy * detail_scale,
        );
    }
    vis_mark_tile(inst.vt.x, uv, ddx, ddy);
    vis_mark_tile(inst.vt.y, uv, ddx, ddy);
    vis_mark_tile(inst.vt.z, uv, ddx, ddy);
}
