// vsm_cull.wgsl — **per-page caster culling** (P27.2, clause 1).
//
// One thread per (page, caster) pair. Each one tests the caster's bounding sphere
// against the page's own frustum and, if it survives, appends the caster's index
// to that page's visible list and bumps that (page, group)'s indirect draw.
//
// ── the two-buffer precedent, and what it becomes with N pages
//
// `inf_vgeom`'s cull rules that there is no `MULTI_DRAW_INDIRECT` here, so a
// second drawable set means a second `(visible list, args)` pair — a non-zero
// `first_instance` would need `INDIRECT_FIRST_INSTANCE`, which is not portable.
// P18.5's scatter answered the same constraint the other way: **one** args buffer
// holding `array<DrawArgs, N>` and `draw_indirect(args, n * stride)`. With one
// pair per (page × group) the second shape is the only one that scales, so it is
// the one here — and the counter still IS the args word, exactly as vgeom's is:
// there is no separate counter buffer and no compaction pass.
//
// ── the visible list has no capacity to overflow
//
// A page's slice of the list is `casters` entries long and holds only caster
// indices, and the casters are packed **sorted by group** — so group `g`'s slice
// starts at the group's own first caster and can hold every caster of that group
// and no more. `ids.z` carries that base, which is why there is no bounds test in
// `append` and no clamp that could silently drop a caster.
//
// ── the append order is not deterministic, and that is checked rather than assumed
//
// `atomicAdd` hands out slots in arrival order, so the *list* differs run to run
// while the *set* does not. That is safe here for `inf_vgeom::stream`'s reason and
// only for it: the page raster is depth-only, opaque or alpha-tested, and a depth
// test is order-independent. It would NOT be safe under blending or an
// order-dependent dither — `scatter_cull.wgsl` documents that trap and pays a
// prefix sum to avoid it.

struct VsmCullParams {
    // x = casters, y = pages, z = groups, w = reserved.
    counts: vec4<u32>,
};
@group(0) @binding(0) var<uniform> params: VsmCullParams;

struct VsmPage {
    // Render-local world -> this page's own clip space
    // (`inf_render::vsm::vsm_page_matrix`).
    view_proj: mat4x4<f32>,
    // x = the page's base in the visible list, y = slot, z = light, w = the
    // page's DETAIL BUCKET as a one-bit mask (island wave I8c): `1 << level` for
    // a clipmap page, `1 << 31` for a perspective one.
    info: vec4<u32>,
};
@group(0) @binding(1) var<storage, read> pages: array<VsmPage>;

struct VsmCaster {
    model: mat4x4<f32>,
    // xyz = render-local bounding-sphere centre, w = radius.
    sphere: vec4<f32>,
    // x = alpha cutoff, y = blend code, z = base-colour alpha, w = reserved.
    mat: vec4<f32>,
    // x = geometry group, y = the caster's index inside its group, z = the
    // group's first caster (its base in a page's visible list), w = the caster's
    // DETAIL-BUCKET MASK (island wave I8c) — the set of page buckets this record
    // is the right level for. **Zero means every bucket**, which is what every
    // caster that is not a meshlet asset carries and what the pass did before the
    // mask existed.
    ids: vec4<u32>,
};
@group(0) @binding(2) var<storage, read> casters: array<VsmCaster>;
@group(0) @binding(3) var<storage, read_write> visible: array<u32>;
// `DrawIndexedIndirectArgs` is five words — index_count, instance_count,
// first_index, base_vertex, first_instance — and word 1 IS the counter.
@group(0) @binding(4) var<storage, read_write> args: array<atomic<u32>>;
// **The compact draw-slot table** (island wave I4b), `pages * groups` entries:
// the args block a `(page, group)` pair owns, or `0xFFFFFFFF` when the pair does
// not exist this frame because no caster of that group has light-space bounds
// touching that page.
//
// The CPU derives it from the same scatter that folds the page content stamps
// and it is **conservative in this cull's own direction** — a live slot may name
// a pair this cull then finds nothing for, and a dead slot can never hide one it
// would have kept. That is what makes the early return below safe: on a streamed
// world a geometry group is one per resident terrain tile and a 128 m shadow page
// overlaps one or two of them, so the great majority of pairs are dead and the
// CPU no longer writes a 256-byte uniform and a five-word args block for each.
@group(0) @binding(5) var<storage, read> slots: array<u32>;

const VSM_ARG_WORDS: u32 = 5u;

// The six clip-space planes of `vp`, tested against a sphere.
// `inf_render::vsm::vsm_page_sees_sphere` + `page_clip_planes`, mirrored —
// conservative in the same direction (it may keep a sphere the page cannot see,
// never drop one it can), and it SKIPS a degenerate plane rather than trusting it.
//
// **The degenerate row is `r2`, the FAR plane** (`z >= 0` is the far test under
// reverse-Z), and a reverse-infinite perspective makes it `(0, 0, 0, near)` —
// no finite normal, because the far plane is at infinity. `r3 - r2`, the near
// plane, is the one row that is always well conditioned. WGSL leaves division by
// zero INDETERMINATE, so the skip is a real guard here even though the Rust twin
// would compute `+near/0 = +inf` and pass. (This comment said `w - z` and the Rust
// doc said "near plane": both were wrong, and
// `the_only_degenerate_page_plane_is_the_far_one` now measures it.)
fn page_sees_sphere(vp: mat4x4<f32>, c: vec3<f32>, radius: f32) -> bool {
    let r0 = vec4<f32>(vp[0].x, vp[1].x, vp[2].x, vp[3].x);
    let r1 = vec4<f32>(vp[0].y, vp[1].y, vp[2].y, vp[3].y);
    let r2 = vec4<f32>(vp[0].z, vp[1].z, vp[2].z, vp[3].z);
    let r3 = vec4<f32>(vp[0].w, vp[1].w, vp[2].w, vp[3].w);
    var planes = array<vec4<f32>, 6>(r0 + r3, r3 - r0, r1 + r3, r3 - r1, r2, r3 - r2);
    for (var i = 0u; i < 6u; i = i + 1u) {
        let p = planes[i];
        let len = length(p.xyz);
        if (len < 1e-6) {
            continue;
        }
        if ((dot(p.xyz, c) + p.w) / len < -radius) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(64, 1, 1)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let pi = gid.y;
    if (ci >= params.counts.x || pi >= params.counts.y) {
        return;
    }
    let c = casters[ci];
    let p = pages[pi];
    // **The detail bucket** (island wave I8c). A meshlet asset is packed once per
    // classic level its resident page buckets ask for, and the masks PARTITION
    // those buckets — so this test is what makes "every page draws the asset
    // exactly once, at the level that page can show" true rather than "at every
    // level, once each". A zero mask is every bucket, which is every other caster
    // in the pass.
    if (c.ids.w != 0u && (c.ids.w & p.info.w) == 0u) {
        return;
    }
    if (!page_sees_sphere(p.view_proj, c.sphere.xyz, c.sphere.w)) {
        return;
    }
    let pair = slots[pi * params.counts.z + c.ids.x];
    if (pair == 0xFFFFFFFFu) {
        return;
    }
    let arg = pair * VSM_ARG_WORDS + 1u;
    let slot = atomicAdd(&args[arg], 1u);
    visible[pi * params.counts.x + c.ids.z + slot] = ci;
}
