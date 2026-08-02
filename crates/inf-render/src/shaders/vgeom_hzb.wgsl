// vgeom_hzb.wgsl — hierarchical depth (HZB) pyramid build (P13.1b v1, P18.1 v2).
//
// Reverse-Z: near→1, far→0, so occlusion needs the FARTHEST surface per texel =
// the MIN depth. `cs_copy` seeds mip 0 from the **MSAA scene depth**; `cs_down`
// min-downsamples into the next (coarser) mip. The cull compute then samples the
// mip whose texels cover a meshlet's screen rect (see vgeom_cull.wgsl `occluded`).
//
// # Why the MSAA scene depth, and why min over samples (P18.1)
//
// v1 seeded mip 0 from the single-sample depth prepass. Two problems:
//
//   1. that target only ever held the *classic* rigid-mesh prepass, so meshlets
//      could never occlude meshlets (the whole point of two-pass);
//   2. it is a **different rasterization** from the 4× MSAA target the meshlets
//      actually depth-test against. At a silhouette the prepass pixel centre can
//      be covered while an MSAA subsample is not — culling against it could hide
//      a meshlet that would legitimately have shaded that subsample. The
//      "occlusion is purely subtractive" proof would not hold.
//
// v2 reads `targets.depth` (the live 4× MSAA scene depth, already
// `TEXTURE_BINDING` since P17.3) and takes the **min over all samples**, i.e. the
// farthest subsample. That is the conservative per-pixel occluder depth: a
// meshlet culled against it is behind *every* subsample of that pixel, so it
// could not have shaded any of them. An uncovered subsample still holds
// DEPTH_CLEAR (0.0 = far) and drives the min to 0, which occludes nothing —
// exactly the conservative behaviour we want at silhouettes.
//
// It also means the pyramid sees whatever has been drawn into the scene depth so
// far this frame — the classic mesh pass AND the vgeom early draw — with no extra
// depth write and no resolve.

// ── Pass 0: min-over-samples of the MSAA scene depth → HZB mip 0 ──
@group(0) @binding(0) var src_depth: texture_depth_multisampled_2d;
@group(0) @binding(1) var dst0: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(dst0);
    if (gid.x >= dim.x || gid.y >= dim.y) {
        return;
    }
    // Mip 0 is allocated at exactly the scene-depth size, so this is 1:1; the
    // clamp only guards a hypothetical mismatch.
    let sdim = vec2<i32>(textureDimensions(src_depth)) - vec2<i32>(1, 1);
    let c = min(vec2<i32>(gid.xy), sdim);
    let n = i32(textureNumSamples(src_depth));
    var d = textureLoad(src_depth, c, 0);
    for (var s = 1; s < n; s = s + 1) {
        d = min(d, textureLoad(src_depth, c, s));
    }
    textureStore(dst0, vec2<i32>(gid.xy), vec4<f32>(d, 0.0, 0.0, 0.0));
}

// ── Pass N: min-downsample mip N-1 → mip N ──
//
// Mip dimensions are `max(dim >> n, 1)` (floor), so an ODD source dimension leaves
// a trailing row/column that a plain 2×2 gather would drop — and a dropped source
// texel is a *hole in the coverage*, which would let the pyramid report an
// occluder nearer than the true farthest surface and over-cull. The last
// destination texel in an odd dimension therefore extends its gather to the end
// of the source (`sdim <= 2*ddim + 1`, so the block is at most 3×3).
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ddim = vec2<i32>(textureDimensions(dst));
    let p = vec2<i32>(gid.xy);
    if (p.x >= ddim.x || p.y >= ddim.y) {
        return;
    }
    let sdim = vec2<i32>(textureDimensions(src));
    let lo = p * 2;
    // Inclusive upper bound: the last destination texel absorbs any trailing
    // source texel an odd dimension leaves behind.
    var hi = min(lo + vec2<i32>(1, 1), sdim - vec2<i32>(1, 1));
    if (p.x == ddim.x - 1) {
        hi.x = sdim.x - 1;
    }
    if (p.y == ddim.y - 1) {
        hi.y = sdim.y - 1;
    }
    var m = 1.0;
    for (var y = lo.y; y <= hi.y; y = y + 1) {
        for (var x = lo.x; x <= hi.x; x = x + 1) {
            m = min(m, textureLoad(src, vec2<i32>(x, y), 0).r);
        }
    }
    textureStore(dst, p, vec4<f32>(m, 0.0, 0.0, 0.0));
}
