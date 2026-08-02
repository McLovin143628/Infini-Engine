// hzb_occlusion.wgsl — the reverse-Z hierarchical-depth occlusion test, shared by
// every GPU cull that reads the P18.1 pyramid (P18.5 extracted it from
// `vgeom_cull.wgsl`, where it had been the private `occluded`).
//
// It lives in one file for the reason the P18 portable-fixture batch recorded at
// length: a rule this load-bearing, hand-copied into a second cull, is a rule the
// two copies eventually stop agreeing about — and the disagreement would be
// invisible in a screenshot, because both copies still *cull*, just differently.
// The bindings differ between consumers (different uniforms, different pyramid
// bind slots), so the test takes them as parameters instead of reading globals.
//
// ── Conservative by construction (the P18.1 proof, unchanged) ────────────────
//
// The pyramid stores the MIN depth over each texel footprint, and mip 0 is the
// min over the MSAA subsamples, so `HZB[texel]` is the farthest surface anywhere
// under it. Let `R` be a screen rect that provably contains the object's
// projection and `d_max` an upper bound on its NDC depth (reverse-Z ⇒ its
// *nearest* point). If `d_max < min_{p in R} HZB(p)` then for every pixel p and
// every subsample s the object covers, `frag_depth <= d_max < D(p, s)`, so every
// fragment fails the `Greater` depth test. The object contributes **zero pixels**
// — culling it cannot change the image.
//
// Both bounds therefore over-approximate, never under:
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
//
// `hzb_dims` is `(mip_count, width, height, unused)` — the same `vec4<f32>` both
// cull uniforms already carry.
fn hzb_occluded(
    view_proj: mat4x4<f32>,
    hzb_dims: vec4<f32>,
    hzb: texture_2d<f32>,
    center: vec3<f32>,
    radius: f32,
) -> bool {
    var uv_min = vec2<f32>(1.0e30, 1.0e30);
    var uv_max = vec2<f32>(-1.0e30, -1.0e30);
    var d_max = -1.0e30;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let o = vec3<f32>(
            select(-radius, radius, (i & 1u) != 0u),
            select(-radius, radius, (i & 2u) != 0u),
            select(-radius, radius, (i & 4u) != 0u),
        );
        let clip = view_proj * vec4<f32>(center + o, 1.0);
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
    let full = vec2<f32>(hzb_dims.y, hzb_dims.z);
    let ext = (uv_max - uv_min) * full;
    let span = max(max(ext.x, ext.y), 1.0);
    let mip = clamp(ceil(log2(span)), 0.0, hzb_dims.x - 1.0);
    let mlvl = i32(mip);
    let dim = vec2<f32>(textureDimensions(hzb, mlvl));
    let maxc = vec2<i32>(dim) - vec2<i32>(1, 1);
    let lo = clamp(uv_min, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let base = clamp(vec2<i32>(floor(lo * dim)), vec2<i32>(0, 0), maxc);
    var occ = 1.0;
    for (var dy = 0; dy < 2; dy = dy + 1) {
        for (var dx = 0; dx < 2; dx = dx + 1) {
            let c = clamp(base + vec2<i32>(dx, dy), vec2<i32>(0, 0), maxc);
            occ = min(occ, textureLoad(hzb, c, mlvl).r);
        }
    }
    return d_max < occ;
}
