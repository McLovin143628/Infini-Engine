// ── THE FROXEL LIGHT-LIST BUILD (wave PAR0) ──────────────────────────────────
//
// One invocation per froxel of the camera-fitted grid (`crate::lights`'s
// CLUSTER_X × CLUSTER_Y × CLUSTER_Z). Each builds its froxel's bounding box in
// VIEW space (right / up / forward, depth positive) from the tile's tangent
// span and the slice's depth span, then walks every LOCAL light record of the
// frame's list in list order and appends the index of each whose cull sphere,
// moved into the same view space, touches the box. List order in, list order
// out, no atomics: the per-froxel lists are a pure function of the light list
// and the view, which is what lets PIE and shipping agree on them byte for byte.
//
// View space rather than world space, measured: a world-axis box around a
// froxel the camera looks down into at 45° is several times the froxel's own
// volume, and on the PAR0 gate's 2 000-light floor it listed ~300 lights a
// froxel where the froxel reaches a fraction of that.
//
// The header and storage words are declared here read-write; the fragment
// library (`lights.wgsl`) declares the same two buffers read-only.

struct LightHeader {
    counts: vec4<u32>,
    grid: vec4<u32>,
    bases: vec4<u32>,
    zparams: vec4<f32>,
    fwd: vec4<f32>,
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    proj: vec4<f32>,
};

@group(0) @binding(0) var<uniform> light_hdr: LightHeader;
@group(0) @binding(1) var<storage, read_write> light_words: array<vec4<u32>>;

fn light_pos_radius(i: u32) -> vec4<f32> {
    let b = i * 4u;
    let pos = bitcast<vec4<f32>>(light_words[b + 1u]);
    let sd = bitcast<vec4<f32>>(light_words[b + 3u]);
    return vec4<f32>(pos.xyz, sd.w);
}

// [near, far) view depths of slice `k` — the exact partition `light_slice`
// inverts, before padding. The last slice ends at `proj.w`, the deepest point
// any listed light's sphere reaches (CPU-derived), rather than at infinity.
fn slice_bounds(k: u32) -> vec2<f32> {
    let z = light_hdr.grid.z;
    let near = light_hdr.zparams.x;
    let far = light_hdr.zparams.y;
    if (k == 0u) {
        return vec2<f32>(0.0, near);
    }
    if (k >= z - 1u) {
        return vec2<f32>(far, max(light_hdr.proj.w, far));
    }
    let inv = 1.0 / light_hdr.zparams.z;
    return vec2<f32>(near * exp2(f32(k - 1u) * inv), near * exp2(f32(k) * inv));
}

// The view-space extent of `[a, b]` (NDC on one axis) at depths `z0..z1`: a
// perspective tile widens with depth, an orthographic one does not.
fn tile_span(a: f32, b: f32, half: f32, z0: f32, z1: f32) -> vec2<f32> {
    if (light_hdr.proj.z > 0.5) {
        return vec2<f32>(a * half, b * half);
    }
    let p = vec4<f32>(a * half * z0, a * half * z1, b * half * z0, b * half * z1);
    return vec2<f32>(min(min(p.x, p.y), min(p.z, p.w)), max(max(p.x, p.y), max(p.z, p.w)));
}

@compute @workgroup_size(64)
fn cs_cluster(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = light_hdr.grid;
    let c = gid.x;
    if (c >= g.w) {
        return;
    }
    let tx = c % g.x;
    let ty = (c / g.x) % g.y;
    let tz = c / (g.x * g.y);

    // Tile corners in NDC: pixel row 0 is the TOP of the frame, NDC y is up.
    let x0 = f32(tx) / f32(g.x) * 2.0 - 1.0;
    let x1 = f32(tx + 1u) / f32(g.x) * 2.0 - 1.0;
    let y0 = 1.0 - f32(ty + 1u) / f32(g.y) * 2.0;
    let y1 = 1.0 - f32(ty) / f32(g.y) * 2.0;

    var dz = slice_bounds(tz);
    // Pad the slice in depth so a fragment whose `log2` rounds across a
    // boundary still lands in a froxel that tested every light reaching it.
    let pad = light_hdr.zparams.w;
    dz = vec2<f32>(max(dz.x * (1.0 - pad) - 0.01, 0.0), dz.y * (1.0 + pad) + 0.01);

    // A one-percent tile pad covers TAA's sub-pixel jitter at the tile edge.
    let sx = (x1 - x0) * 0.01;
    let sy = (y1 - y0) * 0.01;
    let xs = tile_span(x0 - sx, x1 + sx, light_hdr.proj.x, dz.x, dz.y);
    let ys = tile_span(y0 - sy, y1 + sy, light_hdr.proj.y, dz.x, dz.y);
    let lo = vec3<f32>(xs.x, ys.x, dz.x);
    let hi = vec3<f32>(xs.y, ys.y, dz.y);

    let cap = light_hdr.counts.w;
    let total = light_hdr.counts.x;
    let index_base = light_hdr.bases.y * 4u + c * cap;
    let eye = light_hdr.eye.xyz;
    var n = 0u;
    for (var i = light_hdr.counts.y; i < total; i = i + 1u) {
        let s = light_pos_radius(i);
        let d = s.xyz - eye;
        let v = vec3<f32>(dot(d, light_hdr.right.xyz), dot(d, light_hdr.up.xyz), dot(d, light_hdr.fwd.xyz));
        let q = clamp(v, lo, hi);
        let e = v - q;
        if (dot(e, e) <= s.w * s.w) {
            if (n < cap) {
                let w = index_base + n;
                light_words[w >> 2u][w & 3u] = i;
            }
            n = n + 1u;
        }
    }
    let cw = light_hdr.bases.x * 4u + c;
    light_words[cw >> 2u][cw & 3u] = n;
}
