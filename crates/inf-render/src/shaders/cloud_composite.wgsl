// Cloud composite (wave SKY2): the half-res march, upsampled with a
// depth-aware (bilateral) filter and blended into the MSAA scene target.
//
// Composition (`passes::ShaderKind::CloudComposite`): common_view +
// `atmosphere.wgsl` at @group(1) @binding(0). It needs the cloud slab's
// geometry for the depth it writes, and nothing else from the atmosphere — no
// LUTs, no noise volumes, no field.
//
// ## Why a bilateral upsample and not a bilinear one
//
// The tree already had a half-res pass with a full-res consumer: SSAO, whose
// upsample is a 4x4 box BLUR. That is fine for an occlusion term that is
// already low-frequency and multiplies a surface. It is not fine here. A cloud
// tap whose march stopped at a mountain carries almost no cloud; the tap next to
// it, looking past the summit, carries a sky's worth. Bilinear between them
// paints a two-pixel halo of half-cloud along every ridge — the tell of a
// half-res volumetric, and the reason "render it at half res" is usually
// followed by "and it looks like it".
//
// So each of the four taps is weighted by how well the geometry ITS march
// stopped at matches the geometry this full-res pixel sees. The key is the
// distance the march clamped to, published by the march in `cloud_dist.g`, not a
// depth-buffer value: comparing raw reverse-Z is comparing a hyperbola, where a
// two-kilometre summit and infinity are a thousandth apart.
//
// ## Depth
//
// This pass carries the hardware half of the P17.3 depth contract, which the
// march gave up when it stopped rendering into the scene target: `frag_depth` at
// the ray's ENTRY into the slab, `Greater` (reverse-Z) with writes off. That
// rejects — per MSAA sample, so with antialiased silhouettes — every fragment
// whose geometry is entirely in front of the layer. The other half (a summit
// INSIDE the slab must not be veiled by cloud behind it) is the march's `t_far`
// clamp and is unchanged.

@group(1) @binding(1) var cloud_src: texture_2d<f32>;
@group(1) @binding(2) var cloud_dist: texture_2d<f32>;
@group(1) @binding(3) var cloud_scene_depth: texture_depth_multisampled_2d;

// Mirrors `cloud.wgsl`'s sentinel: the distance a march with no geometry in
// front of it publishes.
const CLOUD_NO_GEOMETRY: f32 = 60000.0;
// Softness of the bilateral weight, metres. A tap whose geometry is within this
// of the pixel's is taken at nearly full weight; one much further is nearly
// dropped. 40 m is under a terrain cell and far above the depth buffer's own
// precision at cloud range.
const CLOUD_BILATERAL_M: f32 = 40.0;

struct CompositeOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var out: VsOut;
    let p = fullscreen_ndc(i);
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

// See `cloud.wgsl`: the view uniform carries the floating origin, so world =
// local - those.
fn cloud_to_world(local: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        local.x - view.grid_axis_viewport.x,
        local.y - view.mode_axis.y,
        local.z - view.grid_axis_viewport.y,
    );
}

@fragment
fn fs(in: VsOut) -> CompositeOut {
    var out: CompositeOut;
    out.color = vec4<f32>(0.0);
    out.depth = 0.0;

    let dir = view_ray(in.ndc);
    let eye_world = cloud_to_world(view.eye.xyz);
    let bottom = atmos.cloud_layer.x;
    let top = atmos.cloud_layer.y;

    // ── the slab entry, for `frag_depth` ──
    // The same intersection the march does, and it has to be: a depth that
    // disagreed with the march's `t_near` would reject fragments the march drew
    // and keep ones it did not.
    let dy = dir.y;
    var t_near = 0.0;
    var t_far = 1.0;
    if (abs(dy) > 1e-4) {
        let t0 = (bottom - eye_world.y) / dy;
        let t1 = (top - eye_world.y) / dy;
        t_near = max(min(t0, t1), 0.0);
        t_far = max(t0, t1);
    } else if (eye_world.y < bottom || eye_world.y > top) {
        return out;
    }
    if (t_far <= 0.0) {
        return out;
    }
    let entry_local = view.eye.xyz + dir * max(t_near, 1.0);
    let clip = view.view_proj * vec4<f32>(entry_local, 1.0);
    out.depth = clamp(clip.z / max(clip.w, 1e-6), 0.0, 1.0);

    // ── this pixel's own geometry distance ──
    let full = vec2<i32>(i32(in.pos.x), i32(in.pos.y));
    var my_geo = CLOUD_NO_GEOMETRY;
    let d = textureLoad(cloud_scene_depth, full, 0);
    if (d > 0.0) {
        let hit = unproject(in.ndc, d);
        my_geo = min(dot(hit - view.eye.xyz, dir), CLOUD_NO_GEOMETRY);
    }

    // ── the four taps ──
    // Texel-centre convention: the half-res texel whose centre is nearest is at
    // `full * 0.5 - 0.5`, and the bilinear fractions come off that.
    let half_size = vec2<i32>(textureDimensions(cloud_src));
    let c = vec2<f32>(full) * 0.5 - vec2<f32>(0.25);
    let base = floor(c);
    let f = c - base;
    let bi = vec2<i32>(base);

    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    var best = vec4<f32>(0.0);
    var best_w = -1.0;
    for (var k = 0; k < 4; k = k + 1) {
        let o = vec2<i32>(k & 1, (k >> 1) & 1);
        let t = clamp(bi + o, vec2<i32>(0), half_size - vec2<i32>(1));
        let bilinear = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
        let g = textureLoad(cloud_dist, t, 0).g;
        // The bilateral term. Both at the sentinel (open sky) gives |d| = 0 and
        // full weight, which is the common case and costs nothing.
        let w = bilinear / (1.0 + abs(g - my_geo) / CLOUD_BILATERAL_M);
        let v = textureLoad(cloud_src, t, 0);
        acc = acc + v * w;
        wsum = wsum + w;
        if (w > best_w) {
            best_w = w;
            best = v;
        }
    }
    // A divide-by-zero guard, and NOTHING MORE — stated exactly, because the
    // first draft of this comment claimed the fallback picked "the least wrong
    // tap" when four taps all disagreed, and with these constants that branch
    // cannot be reached. The bilinear weights sum to 1 and the bilateral divisor
    // is at most `1 + CLOUD_NO_GEOMETRY / CLOUD_BILATERAL_M` = 1501, so `wsum` is
    // never below 1/1501 = 6.7e-4. It does not need to be reachable: the weights
    // are NORMALIZED by `wsum`, so even when every tap is wrong the average
    // already leans on whichever is least wrong. The branch exists so that a
    // future edit to either constant — or a weight function that can return a
    // true zero — cannot put a NaN into the scene target.
    if (wsum > 1e-4) {
        out.color = acc / wsum;
    } else {
        out.color = best;
    }
    return out;
}
