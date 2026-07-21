// vgeom_hzb.wgsl — hierarchical depth (HZB) pyramid build (P13.1b, v1).
//
// Reverse-Z: near→1, far→0, so occlusion needs the FARTHEST surface per texel =
// the MIN depth. `cs_copy` seeds mip 0 from the single-sample depth prepass;
// `cs_down` min-downsamples each 2×2 block into the next (coarser) mip. The cull
// compute then samples the mip whose texels roughly cover a meshlet's screen rect
// (see vgeom_cull.wgsl `occluded`).

// ── Pass 0: copy prepass depth → HZB mip 0 (r32float storage) ──
@group(0) @binding(0) var src_depth: texture_depth_2d;
@group(0) @binding(1) var dst0: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(dst0);
    if (gid.x >= dim.x || gid.y >= dim.y) {
        return;
    }
    let sdim = vec2<i32>(textureDimensions(src_depth)) - vec2<i32>(1, 1);
    let c = min(vec2<i32>(gid.xy), sdim);
    let d = textureLoad(src_depth, c, 0);
    textureStore(dst0, vec2<i32>(gid.xy), vec4<f32>(d, 0.0, 0.0, 0.0));
}

// ── Pass N: min-downsample mip N-1 → mip N ──
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = textureDimensions(dst);
    if (gid.x >= dim.x || gid.y >= dim.y) {
        return;
    }
    let sdim = vec2<i32>(textureDimensions(src)) - vec2<i32>(1, 1);
    let sc = vec2<i32>(gid.xy) * 2;
    let a = textureLoad(src, min(sc + vec2<i32>(0, 0), sdim), 0).r;
    let b = textureLoad(src, min(sc + vec2<i32>(1, 0), sdim), 0).r;
    let c = textureLoad(src, min(sc + vec2<i32>(0, 1), sdim), 0).r;
    let e = textureLoad(src, min(sc + vec2<i32>(1, 1), sdim), 0).r;
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(min(min(a, b), min(c, e)), 0.0, 0.0, 0.0));
}
