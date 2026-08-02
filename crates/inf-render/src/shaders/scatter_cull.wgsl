// scatter_cull.wgsl — per-instance GPU culling + LOD selection + **deterministic**
// stream compaction for the P18.5 scatter path. `hzb_occlusion.wgsl` is composed
// in ahead of this module (the shared, proven-conservative depth test).
//
// One batch = one dispatch triple. Per instance the classify pass decides two
// independent booleans — "draws as a full mesh" and "draws as an impostor" — and
// the compaction turns them into two dense index lists plus two indirect draw arg
// structs. The raster passes then vertex-pull through those lists.
//
// ── Why the compaction is a PREFIX SUM and not an atomic append ──────────────
//
// The obvious compaction is `visible[atomicAdd(&count, 1u)] = i`, which is what
// the meshlet cull does — and it produces a list whose ORDER is a function of how
// the GPU happened to schedule its workgroups. The meshlet path can afford that
// because its draw order provably cannot change the image (P18.1's subtractive
// proof plus an opaque depth test), so nothing ever compares the list itself.
//
// Scatter cannot afford it, for a reason that is specific to this batch: the
// LOD cross-fade is a **dithered discard**, so an instance in the fade band emits
// a stippled subset of its pixels. Two instances whose fade bands overlap at the
// same depth then resolve by draw order at the sample level, and the house gates
// compare *images* across two runs. Sorting a readback would only make the AUDIT
// deterministic, not the frame.
//
// So the order is fixed by construction instead: a two-level exclusive prefix sum
// (in-workgroup Hillis–Steele scan → per-workgroup partials → global bases) makes
// the compacted list exactly "the surviving instances in ascending instance index",
// on every adapter, every run. It costs one extra `u32` per instance and two extra
// dispatches, and it is entirely integer addition — exactly associative, so even
// the tree scan is bit-reproducible. The audit counters stay atomic, because a
// SUM is order-independent even when the increments are not.

struct CullParams {
    view_proj: mat4x4<f32>,
    frustum: array<vec4<f32>, 6>,
    // xyz = eye (render-local), w unused.
    eye: vec4<f32>,
    // xyz = the batch anchor in render-local space, w = the instance cull radius
    // at unit scale (primitive bounding radius × the batch's max scale).
    anchor: vec4<f32>,
    // x = instance count, y = cull flags bitset, z = list capacity (== instance
    // count), w = audit enabled.
    counts: vec4<u32>,
    // x = full-mesh band end (m), y = cull distance (m), z = fade band width (m),
    // w = impostors enabled (1/0).
    bands: vec4<f32>,
    // x = mip count, y = hzb width, z = hzb height, w unused.
    hzb: vec4<f32>,
};

// One scattered instance, 48 B — mirrors `scene::ScatterInstanceRaw`.
struct ScatterInst {
    offset: vec3<f32>,
    scale: f32,
    rotation: vec4<f32>,
    color: vec4<f32>,
};

// Non-indexed indirect draw args (vertex pulling ⇒ no index buffer). Written by
// the scan pass, never atomically — the totals are already known there.
struct DrawArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0) var<uniform> params: CullParams;
@group(0) @binding(1) var<storage, read> instances: array<ScatterInst>;
// Per instance: bit 31 = draws as mesh, bit 30 = draws as impostor, bits 0..30 =
// the instance's exclusive offset WITHIN its workgroup, for whichever list(s) it
// joins. An instance can join both (the cross-fade band), and then it needs two
// offsets — so the word carries the mesh offset and `slots_imp` the impostor one.
@group(0) @binding(2) var<storage, read_write> slots: array<vec2<u32>>;
// Per workgroup: (mesh_count, impostor_count), rewritten by the scan pass into
// the workgroup's exclusive base in each list. Entry `[nwg]` holds the totals.
@group(0) @binding(3) var<storage, read_write> partials: array<vec2<u32>>;
// The compacted lists: mesh at [0, cap), impostor at [cap, 2*cap).
@group(0) @binding(4) var<storage, read_write> visible: array<u32>;
@group(0) @binding(5) var<storage, read_write> draw_args: array<DrawArgs, 2>;
// Audit counters: [candidates, frustum_culled, occluded, distance_culled, mesh,
// impostor, _, _]. Eight slots so a seventh costs no layout change; only touched
// when `params.counts.w != 0`, so the shipping path pays nothing.
@group(0) @binding(6) var<storage, read_write> stats: array<atomic<u32>, 8>;
@group(0) @binding(7) var hzb_tex: texture_2d<f32>;

const FLAG_FRUSTUM: u32 = 1u;
const FLAG_OCCLUSION: u32 = 4u;

const AUDIT_CANDIDATES: u32 = 0u;
const AUDIT_FRUSTUM: u32 = 1u;
const AUDIT_OCCLUDED: u32 = 2u;
const AUDIT_DISTANCE: u32 = 3u;
const AUDIT_MESH: u32 = 4u;
const AUDIT_IMPOSTOR: u32 = 5u;

const WG: u32 = 256u;

var<workgroup> sm: array<vec2<u32>, WG>;

fn in_frustum(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let p = params.frustum[i];
        if (dot(p.xyz, center) + p.w < -radius) {
            return false;
        }
    }
    return true;
}

// ── Pass 1: classify + in-workgroup exclusive scan ───────────────────────────
@compute @workgroup_size(256)
fn cs_classify(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    let n = params.counts.x;
    let flags = params.counts.y;
    let audit = params.counts.w != 0u;

    var want_mesh = false;
    var want_imp = false;

    if (i < n) {
        let inst = instances[i];
        let center = params.anchor.xyz + inst.offset;
        // One conservative radius for the whole batch: the primitive's unit
        // bounding radius times the batch's LARGEST scale. Per-instance would cull
        // marginally more; a single scalar keeps the record at 48 B and the bound
        // is still an over-approximation, which is all the subtractive proof needs.
        let radius = params.anchor.w;
        let d = distance(center, params.eye.xyz);

        var alive = true;
        if (audit) { atomicAdd(&stats[AUDIT_CANDIDATES], 1u); }

        // Distance first: it is a scalar compare and removes the far field before
        // anything projective runs.
        if (d >= params.bands.y) {
            alive = false;
            if (audit) { atomicAdd(&stats[AUDIT_DISTANCE], 1u); }
        }
        if (alive && (flags & FLAG_FRUSTUM) != 0u && !in_frustum(center, radius)) {
            alive = false;
            if (audit) { atomicAdd(&stats[AUDIT_FRUSTUM], 1u); }
        }
        if (alive && (flags & FLAG_OCCLUSION) != 0u
            && hzb_occluded(params.view_proj, params.hzb, hzb_tex, center, radius)) {
            alive = false;
            if (audit) { atomicAdd(&stats[AUDIT_OCCLUDED], 1u); }
        }

        if (alive) {
            let fade = max(params.bands.z, 1.0e-3);
            let mesh_end = params.bands.x;
            want_mesh = d < mesh_end;
            // The impostor covers the far band AND the near cross-fade, so the two
            // overlap by exactly `fade` metres — the band in which each pixel is
            // resolved by the complementary dither in `scatter_mesh.wgsl`.
            want_imp = params.bands.w != 0.0 && d >= mesh_end - fade;
            if (audit) {
                if (want_mesh) { atomicAdd(&stats[AUDIT_MESH], 1u); }
                if (want_imp) { atomicAdd(&stats[AUDIT_IMPOSTOR], 1u); }
            }
        }
    }

    // Hillis–Steele inclusive scan over the two flag lanes. Integer adds only, a
    // fixed number of barriers, no divergence in the scan itself ⇒ bit-identical
    // on every adapter.
    sm[lid] = vec2<u32>(select(0u, 1u, want_mesh), select(0u, 1u, want_imp));
    workgroupBarrier();
    for (var off = 1u; off < WG; off = off << 1u) {
        var add = vec2<u32>(0u, 0u);
        if (lid >= off) {
            add = sm[lid - off];
        }
        workgroupBarrier();
        sm[lid] = sm[lid] + add;
        workgroupBarrier();
    }
    // Inclusive → exclusive.
    let incl = sm[lid];
    let excl = incl - vec2<u32>(select(0u, 1u, want_mesh), select(0u, 1u, want_imp));

    if (i < n) {
        let tag = select(0u, 0x80000000u, want_mesh) | select(0u, 0x40000000u, want_imp);
        slots[i] = vec2<u32>(tag | (excl.x & 0x3FFFFFFFu), excl.y);
    }
    if (lid == WG - 1u) {
        partials[wid.x] = incl;
    }
}

// ── Pass 2: exclusive-scan the per-workgroup partials, publish the draw args ──
//
// One workgroup, one thread. `nwg` is `ceil(n / 256)`, so at the 1M-instance
// ROADMAP target this is a 4096-iteration integer loop — microseconds, and
// obviously sequential rather than obviously-parallel-and-subtly-not.
@compute @workgroup_size(1)
fn cs_scan(@builtin(local_invocation_index) lid: u32) {
    if (lid != 0u) {
        return;
    }
    let nwg = (params.counts.x + WG - 1u) / WG;
    var run = vec2<u32>(0u, 0u);
    for (var w = 0u; w < nwg; w = w + 1u) {
        let c = partials[w];
        partials[w] = run;
        run = run + c;
    }
    let cap = params.counts.z;
    let n_mesh = min(run.x, cap);
    let n_imp = min(run.y, cap);
    partials[nwg] = vec2<u32>(n_mesh, n_imp);
    // draw_args[0].vertex_count is the batch primitive's index count, written from
    // the CPU (it is a property of the mesh kind, not of the cull); the impostor's
    // is a fixed 6. Only the instance counts come from here.
    draw_args[0].instance_count = n_mesh;
    draw_args[1].instance_count = n_imp;
}

// ── Pass 3: scatter each surviving index into its dense slot ─────────────────
@compute @workgroup_size(256)
fn cs_compact(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let i = gid.x;
    if (i >= params.counts.x) {
        return;
    }
    let s = slots[i];
    let base = partials[wid.x];
    let cap = params.counts.z;
    if ((s.x & 0x80000000u) != 0u) {
        let dst = base.x + (s.x & 0x3FFFFFFFu);
        if (dst < cap) {
            visible[dst] = i;
        }
    }
    if ((s.x & 0x40000000u) != 0u) {
        let dst = base.y + s.y;
        if (dst < cap) {
            visible[cap + dst] = i;
        }
    }
}
