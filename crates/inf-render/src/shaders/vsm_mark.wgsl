// vsm_mark.wgsl — **screen-driven shadow-page marking** (P27.1, clause 1).
//
// One thread per screen pixel. Each one reads the frame's depth, reconstructs the
// world position it stands for, projects that into every shadow-casting light's
// page space, decides which level the pixel's footprint justifies, and
// `atomicOr`s one bit into a fixed-layout bitmask. The CPU reads that mask back
// two frames later (`inf_render::readback`, pinned latency) and turns the bits
// into wants in entry order.
//
// ── why an OR-mask, and why that reconciles it with the replay doctrine
//
// `inf_vgeom::stream`:52-62 rules that GPU feedback must not drive residency,
// because "a readback is one frame latent, so the want set would depend on the
// frame HISTORY, and two runs differing by a single dropped frame would diverge".
// Three properties answer it, and all three are visible in this file:
//
//  1. **Order independence.** The only write is `atomicOr` of a constant bit.
//     OR is commutative and idempotent, so the mask is a pure function of
//     (camera, depth buffer, light projections, table) — not of how many threads
//     ran, in what order, or how often they wrote the same bit. There is no
//     counter to race and no compaction whose output depends on arrival order.
//  2. **Fixed addressing.** A bit's index is `light_base + entry_index`, the SAME
//     index the indirection table uses for that page (`inf_vsm::mark`), so the
//     producer and the consumer cannot disagree about what a bit means.
//  3. **Pinned latency**, on the CPU side: the reader takes frame F−2's mask or
//     nothing, never "whatever arrived".
//
// And the fourth: a page nothing marked is a page nothing SAMPLES, so a dropped
// mask costs shadow detail on the pages already resident and can never be a hole
// somewhere else.
//
// ── where the depth buffer binds, and why that is not the VT situation
//
// `docs/memos/p26-4-feedback-mechanism.md` records that virtual texturing's
// coverage had to be per-SURFACE, because a per-fragment mark needs a writable
// storage buffer visible to the FRAGMENT stage, the renderer runs on
// `Limits::default()`'s four bind groups, all four are spoken for, and a writable
// entry in the SHARED environment layout would make `FRAGMENT_WRITABLE_STORAGE` a
// requirement of every lit pass on every adapter.
//
// None of that reaches here. This is a **compute** pass that CONSUMES the depth
// attachment: it owns its own single bind group (`@group(0)`, five entries, this
// file's), it is recorded after the graph has finished writing depth and before
// the frame's submit, and a writable storage buffer in the COMPUTE stage is core
// wgpu on every adapter. So the shadow signal is per-PIXEL — occlusion included
// by construction, since the depth buffer is what it reads — with no new
// capability, no second geometry pass and no change to any lit pipeline.
//
// The depth it reads is the live MSAA scene depth at `@binding(4)`, loaded at
// **sample 0**: the same texture and the same rule `cloud.wgsl` already uses to
// clamp its raymarch (`texture_depth_multisampled_2d` + `textureLoad(..., 0)`),
// so this pass adds no texture and no usage flag to `FrameTargets`.

struct VsmMarkParams {
    // Camera clip -> render-local world, for the depth reconstruction.
    inv_view_proj: mat4x4<f32>,
    // xyz = render-local eye, w = pixels per world unit at one metre.
    eye: vec4<f32>,
    // x = projection count, y = mask words, z = viewport width, w = height.
    counts: vec4<u32>,
};
@group(0) @binding(0) var<uniform> params: VsmMarkParams;

struct VsmProjection {
    // Render-local world -> this light/face's LEVEL-0 clip space.
    view_proj: mat4x4<f32>,
    // x = word offset of this light's block in `vsm_table`, y = its first bit in
    // the mask, z = the face index, w = kind (0 = ortho clipmap, 1 = perspective).
    info: vec4<u32>,
    // xyz = the light's render-local position (perspective only), w = the level-0
    // shadow texel size: world metres for a clipmap, metres per metre of light
    // distance for a perspective light.
    light: vec4<f32>,
};
@group(0) @binding(1) var<storage, read> projections: array<VsmProjection>;
// The indirection table, READ ONLY — the same buffer the P27.4 receiver samples
// through, so the tree geometry here is the tree geometry there.
@group(0) @binding(2) var<storage, read> vsm_table: array<u32>;
@group(0) @binding(3) var<storage, read_write> mask: array<atomic<u32>>;
@group(0) @binding(4) var scene_depth: texture_depth_multisampled_2d;

const VSM_LIGHT_HEADER_WORDS: u32 = 4u;
const VSM_LEVEL_REC_WORDS: u32 = 4u;
const VSM_PROJ_ORTHO: u32 = 0u;

// The level a footprint justifies — `inf_render::vsm::vsm_justified_level`,
// mirrored. One shadow texel per screen pixel, erring COARSE.
fn justified_level(texel0: f32, pixel_world: f32, levels: u32) -> u32 {
    let want = ceil(log2(max(pixel_world, 1e-6) / max(texel0, 1e-9)));
    return u32(clamp(want, 0.0, f32(levels - 1u)));
}

@compute @workgroup_size(8, 8, 1)
fn cs_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.counts.z || gid.y >= params.counts.w) {
        return;
    }
    // Reverse-Z: the clear value is 0 and it means "far / nothing was drawn".
    // A sky pixel therefore marks NOTHING, which is what makes an empty frame's
    // mask empty rather than a clipmap-wide claim — and is the control the
    // marking arms assert on.
    let depth = textureLoad(scene_depth, vec2<i32>(gid.xy), 0);
    if (depth <= 0.0) {
        return;
    }

    // Pixel centre -> NDC -> render-local world.
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) /
             vec2<f32>(f32(params.counts.z), f32(params.counts.w));
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let h = params.inv_view_proj * vec4<f32>(ndc, 1.0);
    if (abs(h.w) < 1e-12) {
        return;
    }
    let world = h.xyz / h.w;

    // The world size of one screen pixel here — the quantity the level rule is
    // against, from the same `projection_scale` the VT floor uses.
    let view_dist = max(length(world - params.eye.xyz), 1e-4);
    let pixel_world = view_dist / max(params.eye.w, 1e-6);

    let n = params.counts.x;
    for (var i = 0u; i < n; i = i + 1u) {
        let p = projections[i];
        let clip = p.view_proj * vec4<f32>(world, 1.0);
        if (clip.w <= 0.0) {
            continue;
        }
        let l = clip.xyz / clip.w;
        // Outside the light's own depth range (reverse-Z: 1 at the near plane,
        // 0 at the far one).
        if (l.z <= 0.0 || l.z > 1.0) {
            continue;
        }

        let b = p.info.x;
        let levels = vsm_table[b];
        if (levels == 0u) {
            continue;
        }

        // The level-0 shadow texel here: absolute for a clipmap, distance-scaled
        // for a spot or a cube face.
        var texel0 = p.light.w;
        if (p.info.w != VSM_PROJ_ORTHO) {
            texel0 = texel0 * max(length(world - p.light.xyz), 1e-4);
        }
        var level = justified_level(texel0, pixel_world, levels);

        var xy = l.xy;
        if (p.info.w == VSM_PROJ_ORTHO) {
            // A clipmap level covers 2^L times level 0 about the SAME centre, so
            // its NDC is level 0's divided by 2^L — one matrix reaches every
            // level. A point must also be INSIDE the level that serves it, which
            // is the other half of the clipmap's level rule
            // (`inf_render::vsm::clipmap_containing_level`): a pixel cannot be
            // served by a level that misses it, however fine its footprint says
            // it deserves.
            let extent = max(abs(l.x), abs(l.y));
            if (extent > 1.0) {
                let contain = ceil(log2(extent));
                if (contain > f32(levels - 1u)) {
                    continue;
                }
                level = max(level, u32(contain));
            }
            xy = l.xy / exp2(f32(level));
        }
        if (abs(xy.x) > 1.0 || abs(xy.y) > 1.0) {
            continue;
        }

        let rec = b + VSM_LIGHT_HEADER_WORDS + level * VSM_LEVEL_REC_WORDS;
        let pages_x = vsm_table[rec];
        let pages_y = vsm_table[rec + 1u];
        let first = vsm_table[rec + 2u];
        let face_stride = vsm_table[rec + 3u];
        // NDC y is up and the page grid's y runs down (the table's `(y, x)`
        // order), so the vertical axis flips here — the same flip
        // `inf_render::vsm::mark_page_for` performs, stated in both places.
        let px = min(u32(clamp(xy.x * 0.5 + 0.5, 0.0, 0.999999) * f32(pages_x)), pages_x - 1u);
        let py = min(u32(clamp(0.5 - xy.y * 0.5, 0.0, 0.999999) * f32(pages_y)), pages_y - 1u);

        // A bit is `light_base + entry_index`, and `entry_index` is derived from
        // the level record's ABSOLUTE first-entry word minus the block's own
        // entry base — the same arithmetic `VsmLightDesc::entry_index` does on
        // the CPU, read out of the table rather than recomputed from a sum this
        // shader would have to carry.
        let entries_base = b + VSM_LIGHT_HEADER_WORDS + levels * VSM_LEVEL_REC_WORDS;
        let entry = (first - entries_base) + p.info.z * face_stride + py * pages_x + px;
        let bit = p.info.y + entry;
        let word = bit / 32u;
        if (word < params.counts.y) {
            atomicOr(&mask[word], 1u << (bit % 32u));
        }
    }
}
