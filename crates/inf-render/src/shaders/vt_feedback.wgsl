// vt_feedback.wgsl — **the deterministic coverage feedback** (P26.4, clause 1).
//
// One thread per (drawn surface × bound texture) request. Each one decides which
// mip level of its texture this camera justifies, and `atomicOr`s a bit for every
// tile of that level into a fixed-layout bitmask. The CPU reads that mask back
// two frames later (`inf_render::readback`, pinned latency) and turns the bits
// into refinement wants in virtual-address order.
//
// ── why an OR-mask, and why that is what reconciles it with the replay doctrine
//
// `inf_vgeom::stream`:52-62 rules that GPU feedback must not drive residency,
// because "a readback is one frame latent, so the want set would depend on the
// frame HISTORY, and two runs differing by a single dropped frame would diverge".
// Three properties answer it, and all three are visible in this file:
//
//  1. **Order independence.** The only write is `atomicOr` of a constant bit.
//     OR is commutative and idempotent, so the mask is a pure function of
//     (camera, request list, table) — not of how many threads ran, in what
//     order, or how often they wrote the same bit. There is no counter to race
//     and no compaction whose output depends on arrival order.
//  2. **Fixed addressing.** A bit's index is `base + entry_index`, the SAME
//     index the indirection table uses for that tile (`inf_vt::feedback`), so
//     the producer and the consumer cannot disagree about what a bit means.
//  3. **Pinned latency**, on the CPU side: the reader takes frame F−2's mask or
//     nothing, never "whatever arrived".
//
// And the fourth, which is what makes a dropped frame harmless: every want this
// produces is a REFINEMENT, served after the whole analytic floor, so residency
// can never fall below the floor because a mask was late.
//
// ── the deviation, stated where it lives
//
// This is a **per-surface** signal, not a per-fragment one: it marks the level a
// surface's screen footprint justifies, for every tile of that level, rather than
// the tiles a fragment actually sampled. `docs/memos/p26-4-feedback-mechanism.md`
// has the measurement and the reasoning — the short version is that a
// fragment-side mark needs a writable storage buffer in the FRAGMENT stage, the
// renderer's four bind groups are all spoken for so it would have to go in the
// SHARED environment layout, and that would make every lit pass on every adapter
// require `FRAGMENT_WRITABLE_STORAGE`. What is lost is occlusion and uv-extent
// precision; what is kept is every determinism property above, and the pipeline
// P28.3 needs.

struct FeedbackParams {
    view_proj: mat4x4<f32>,
    // xyz = render-local eye, w = pixels per world unit at one metre
    // (`0.5 * height / tan(fov_y / 2)`).
    eye: vec4<f32>,
    // x = request count, y = mask words, z = max tiles one request may mark,
    // w = half the viewport height in pixels.
    counts: vec4<u32>,
};
@group(0) @binding(0) var<uniform> params: FeedbackParams;

struct FeedbackRequest {
    // xyz = render-local bounding-sphere centre, w = its radius in metres.
    center: vec4<f32>,
    // x = word offset of this texture's block in `vt_table`, y = its first bit in
    // the mask, zw = reserved.
    tex: vec4<u32>,
};
@group(0) @binding(1) var<storage, read> requests: array<FeedbackRequest>;
// The indirection table, READ ONLY — the same buffer the fragment shader samples
// through, so the pyramid geometry here is the pyramid geometry there.
@group(0) @binding(2) var<storage, read> vt_table: array<u32>;
@group(0) @binding(3) var<storage, read_write> mask: array<atomic<u32>>;

const VT_TEX_HEADER_WORDS: u32 = 4u;
const VT_MIP_REC_WORDS: u32 = 4u;

fn tiles_down(height: u32, tile_size: u32) -> u32 {
    return max((height + tile_size - 1u) / tile_size, 1u);
}

@compute @workgroup_size(64)
fn cs_feedback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.counts.x) {
        return;
    }
    let req = requests[i];
    let centre = req.center.xyz;
    let radius = max(req.center.w, 1e-4);

    // ── is this surface on screen at all? ──
    //
    // The camera term, and it is `inf_render::on_screen`'s — BRANCH FOR BRANCH,
    // margin for margin (the P26.4 audit; `ndc_margin` names the factor on the
    // Rust side and `the_feedback_and_the_floor_agree_about_what_is_on_screen`
    // sweeps the two against each other on a device). A surface the floor keeps
    // and the feedback drops is a surface that pays for its floor pages and is
    // never refined, which no counter in this system reports.
    let clip = params.view_proj * vec4<f32>(centre, 1.0);
    let dist = max(length(centre - params.eye.xyz), 1e-4);
    let r_px = radius * params.eye.w / dist;
    // The sphere's projected EXTENT as an NDC margin on the shorter axis;
    // generous on the longer one, which is the conservative direction.
    let margin = 2.0 * r_px / max(f32(params.counts.w), 1.0);
    if (clip.w <= 0.0) {
        // Behind the eye — unless the sphere still STRADDLES it. That is the
        // ordinary state of the terrain-sized quad the memo names as this
        // signal's worst case: a camera standing on a 200 m ground plane is
        // inside its bounding sphere, and an unconditional return here means the
        // largest surface on screen is the one that never gets marked.
        if (radius <= -clip.w) {
            return;
        }
    } else {
        let ndc = clip.xyz / clip.w;
        if (abs(ndc.x) > 1.0 + margin || abs(ndc.y) > 1.0 + margin) {
            return;
        }
    }

    // ── which level does that footprint justify? ──
    let b = req.tex.x;
    let mip_count = vt_table[b];
    let tile_size = vt_table[b + 1u];
    let rec0 = b + VT_TEX_HEADER_WORDS;
    let extent = f32(max(vt_table[rec0], vt_table[rec0 + 1u]));
    // The surface covers about `2 * r_px` pixels; a level of `extent` texels is
    // the right one when its texels land about one per pixel.
    let want = log2(max(extent, 1.0) / max(2.0 * r_px, 1.0));
    var lv = u32(clamp(ceil(want), 0.0, f32(mip_count - 1u)));

    // …and never more tiles than the budget allows one request to mark. Coarser
    // until it fits, so a 8k texture filling the screen asks for a level it can
    // actually be served rather than for four thousand pages.
    loop {
        let rec = b + VT_TEX_HEADER_WORDS + lv * VT_MIP_REC_WORDS;
        let tx = vt_table[rec + 2u];
        let ty = tiles_down(vt_table[rec + 1u], tile_size);
        if (tx * ty <= params.counts.z || lv + 1u >= mip_count) {
            break;
        }
        lv = lv + 1u;
    }

    // ── mark every tile of it ──
    //
    // A bit is `base + entry_index`, and `entry_index` is derived from the mip
    // record's ABSOLUTE first-entry word minus the block's own entry base — the
    // same arithmetic `VtTextureDesc::entry_index` does on the CPU, read out of
    // the table rather than recomputed from a sum this shader would have to
    // carry.
    let rec = b + VT_TEX_HEADER_WORDS + lv * VT_MIP_REC_WORDS;
    let tiles_x = vt_table[rec + 2u];
    let tiles_y = tiles_down(vt_table[rec + 1u], tile_size);
    let entries_base = b + VT_TEX_HEADER_WORDS + mip_count * VT_MIP_REC_WORDS;
    let first_entry = vt_table[rec + 3u] - entries_base;
    let base_bit = req.tex.y + first_entry;
    let words = params.counts.y;
    for (var t = 0u; t < tiles_x * tiles_y; t = t + 1u) {
        let bit = base_bit + t;
        let word = bit / 32u;
        if (word < words) {
            atomicOr(&mask[word], 1u << (bit % 32u));
        }
    }
}
