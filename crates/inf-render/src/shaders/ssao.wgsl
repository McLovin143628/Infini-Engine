// **GTAO** (wave VIS1a; a hemisphere-kernel SSAO from P13.3a until then) —
// half-res, ambient-only ground-truth ambient occlusion reconstructed from the
// single-sample depth prepass. `common_view.wgsl` (group 0 = view) is prepended,
// giving `unproject()` + the view/inv-view-proj matrices.
//
// **What changed, and why it is a different integral rather than a bigger
// kernel.** The P13.3a pass projected N points of a rotated hemisphere back
// through the depth buffer and counted how many landed behind geometry. That
// estimator is a *fraction of samples occluded*: it has no cosine weighting, so a
// grazing occluder counts as much as one overhead; its variance is the sample
// count, so it needs a blur wide enough to hide the rotation tile; and its answer
// depends on the kernel's own distribution rather than on the scene.
//
// GTAO (Jimenez et al., "Practical Realtime Strategies for Accurate Indirect
// Occlusion", 2016) integrates the *visibility* function directly. For each of a
// few screen-space slices through the pixel it marches out both ways to find the
// **horizon angles** — the highest thing in each direction — and then evaluates
// the cosine-weighted visibility of the wedge between them in closed form. The
// result is the real integral of `V·cos` over the hemisphere, up to the slice
// count; it is correct at grazing angles by construction, and its error shows up
// as a slight underestimate of thin occluders rather than as noise.
//
// The march is **screen-space**: samples walk a straight line of texels, so the
// depth fetches are coherent, unlike the kernel's scattered projections.
//
// `fs_blur` is a **depth-aware bilateral** blur (was a 4x4 box). A box blur over
// AO leaks occlusion across every silhouette in the frame — the halo around a
// character's shoulder against a distant wall is the box, not the AO — and at
// half resolution one leaked texel is four.

const MAX_SLICES: u32 = 8u;
const MAX_STEPS: u32 = 12u;
const PI: f32 = 3.14159265359;

struct SsaoParams {
    // x = slice count, y = radius (m), z = intensity, w = bias (m).
    cfg: vec4<f32>,
    // xy = depth texture size (px), zw = ao target size (px).
    dims: vec4<f32>,
    // x = steps per slice direction, y = thickness heuristic, zw reserved.
    cfg2: vec4<f32>,
};
@group(1) @binding(0) var depth_tex: texture_depth_2d;
@group(1) @binding(1) var<uniform> ssao: SsaoParams;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let ndc = fullscreen_ndc(i);
    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return out;
}

fn load_depth(uv: vec2<f32>) -> f32 {
    let d = ssao.dims.xy;
    let c = clamp(vec2<i32>(uv * d), vec2<i32>(0), vec2<i32>(d) - vec2<i32>(1));
    return textureLoad(depth_tex, c, 0);
}

fn world_at(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return unproject(ndc, depth);
}

// Interleaved-gradient noise → a per-pixel rotation angle for the slice set.
fn ign(pix: vec2<f32>) -> f32 {
    let m = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(m.z * fract(dot(pix, m.xy)));
}

// The GTAO arc integral for one horizon angle `h` about a slice whose projected
// normal sits at angle `n` from the view vector. Jimenez eq. (7), one side.
fn gtao_arc(h: f32, n: f32) -> f32 {
    return 0.25 * (-cos(2.0 * h - n) + cos(n) + 2.0 * h * sin(n));
}

// **What that integral reads on a slice with no horizon in it** — the two arcs
// with `h1 = n - π/2` and `h2 = n + π/2`, which reduces exactly to
// `cos n + n·sin n`.
//
// It is written down and divided by rather than folded into a constant, and that
// is the whole normalization: the unoccluded value depends on `n`, so a fixed
// divisor is right at one normal orientation and wrong at every other. Measured
// before it was written — an un-normalized integral darkened a wall 25 m from the
// nearest occluder by 0.3 % of its luminance, uniformly, with nothing in the
// frame able to explain it.
fn gtao_open(n: f32) -> f32 {
    return cos(n) + n * sin(n);
}

@fragment
fn fs_ssao(in: VsOut) -> @location(0) vec4<f32> {
    let depth = load_depth(in.uv);
    // Reverse-Z: depth 0 == far/background (sky) → fully unoccluded.
    if (depth <= 0.0) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    let p = world_at(in.uv, depth);

    // Normal from the derivative of the reconstructed world position — the same
    // source the P13.3a pass used. A real normal buffer is the alternative and it
    // is a target this renderer does not have; the derivative is exact on a flat
    // surface and wrong only across a silhouette, where the horizon search is
    // already dominated by the occluder.
    var n = normalize(cross(dpdx(p), dpdy(p)));
    let to_eye = view.eye.xyz - p;
    let d_center = length(to_eye);
    let v = to_eye / max(d_center, 1e-6);
    if (dot(n, v) < 0.0) {
        n = -n;
    }

    let radius = max(ssao.cfg.y, 1e-3);
    let bias = ssao.cfg.w;
    let slices = max(u32(ssao.cfg.x), 1u);
    let steps = max(u32(ssao.cfg2.x), 1u);

    // The world-space radius as a screen-space one, so the march covers the same
    // metres at every distance instead of the same pixels. `view_proj`'s x scale
    // over the sample's view depth is the projection's own metres→NDC factor.
    let clip = view.view_proj * vec4<f32>(p, 1.0);
    let screen_radius = clamp(
        radius * abs(view.view_proj[0][0]) / max(clip.w, 1e-4) * 0.5,
        1.5 / ssao.dims.z,
        0.5
    );

    // One rotation per pixel; the slices are then evenly spread over π (a slice
    // and its opposite are the same slice, marched both ways).
    let rot = ign(in.pos.xy) * PI;
    // A per-pixel offset along the march, so the step pattern does not band.
    let jitter = fract(ign(in.pos.xy + vec2<f32>(17.0, 31.0)) + 0.5);

    var visibility = 0.0;
    var weight = 0.0;
    for (var s = 0u; s < slices && s < MAX_SLICES; s = s + 1u) {
        let phi = rot + PI * f32(s) / f32(slices);
        let dir = vec2<f32>(cos(phi), sin(phi));

        // The slice plane is spanned by `v` and the world direction the screen
        // direction maps to. `dir_w` is that direction, orthogonalized against v.
        let right = normalize(cross(v, vec3<f32>(0.0, 1.0, 0.0)) + vec3<f32>(1e-5, 0.0, 0.0));
        let up = cross(right, v);
        var dir_w = normalize(right * dir.x + up * dir.y);
        dir_w = normalize(dir_w - v * dot(dir_w, v));

        // The projection of the surface normal into the slice plane, and its
        // signed angle from the view vector.
        let n_plane = n - dir_w * 0.0; // (kept explicit: n is already 3-space)
        let np = v * dot(n, v) + dir_w * dot(n, dir_w);
        let np_len = length(np);
        if (np_len < 1e-4) {
            continue;
        }
        let np_n = np / np_len;
        // `gamma`: the angle from v to the projected normal, signed by dir_w.
        let gamma = sign(dot(np_n, dir_w)) * acos(clamp(dot(np_n, v), -1.0, 1.0));

        // March both ways, tracking the largest cosine (the highest horizon).
        var cos_h = vec2<f32>(-1.0, -1.0);
        for (var side = 0u; side < 2u; side = side + 1u) {
            let sgn = select(-1.0, 1.0, side == 0u);
            var best = -1.0;
            for (var t = 1u; t <= steps && t <= MAX_STEPS; t = t + 1u) {
                let f = (f32(t) - 1.0 + jitter) / f32(steps);
                // Quadratic spacing: dense at the contact, coarse at the rim.
                let uv = in.uv + dir * (sgn * screen_radius * f * f);
                if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
                    break;
                }
                let sd = load_depth(uv);
                if (sd <= 0.0) {
                    continue; // sky: nothing there to be a horizon
                }
                let sp = world_at(uv, sd);
                let delta = sp - p;
                let dist = length(delta);
                if (dist < 1e-5) {
                    continue;
                }
                let sample_dir = delta / dist;
                // Only the component in this slice's plane is a horizon for it.
                let c = dot(sample_dir, v);
                // Range check: an occluder well outside the radius is a different
                // surface, not a crease. Falls off rather than cutting, so a
                // slowly receding wall does not produce a ring.
                let fall = clamp(1.0 - (dist - radius) / max(radius, 1e-4), 0.0, 1.0);
                let biased = c - bias / max(dist, 1e-4);
                best = max(best, mix(-1.0, biased, fall));
            }
            if (side == 0u) {
                cos_h.x = best;
            } else {
                cos_h.y = best;
            }
        }

        // Horizon angles, measured from v, one each side, then clamped into the
        // hemisphere about the projected normal — the step that makes this a
        // visibility integral rather than a horizon count.
        var h1 = -acos(clamp(cos_h.x, -1.0, 1.0));
        var h2 = acos(clamp(cos_h.y, -1.0, 1.0));
        h1 = gamma + max(h1 - gamma, -0.5 * PI);
        h2 = gamma + min(h2 - gamma, 0.5 * PI);
        let open = gtao_open(gamma);
        if (open > 1e-4) {
            // The slice's own visibility in 0..1, then a weighted mean over the
            // slices with `|n'|` as the weight — a slice whose plane contains the
            // normal says more about this surface than one nearly edge-on to it.
            let vis = clamp((gtao_arc(h1, gamma) + gtao_arc(h2, gamma)) / open, 0.0, 1.0);
            visibility = visibility + np_len * vis;
            weight = weight + np_len;
        }
    }
    if (weight <= 1e-4) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    let ao = clamp(visibility / weight, 0.0, 1.0);
    // `intensity` scales the OCCLUSION, so 0 is "no AO" and 1 is the integral as
    // computed — the same convention the P13.3a pass used.
    return vec4<f32>(clamp(1.0 - (1.0 - ao) * ssao.cfg.z, 0.0, 1.0), 0.0, 0.0, 1.0);
}

@group(1) @binding(2) var ao_src: texture_2d<f32>;
@group(1) @binding(3) var ao_smp: sampler;

// **Depth-aware bilateral blur** (was a 4x4 box). Same 4x4 footprint, but each
// tap is weighted by how close its depth is to the centre's, so occlusion does
// not leak across a silhouette. The weight is on the RATIO of the reverse-Z
// depths rather than on their difference, because reverse-infinite-Z has no far
// plane to linearize against — the same reason SSR's thickness test is relative.
@fragment
fn fs_blur(in: VsOut) -> @location(0) vec4<f32> {
    let t = 1.0 / ssao.dims.zw;
    let centre_z = load_depth(in.uv);
    if (centre_z <= 0.0) {
        // Sky: nothing to blur toward, and every neighbour would drag geometry's
        // occlusion onto the background.
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    var sum = 0.0;
    var weight = 0.0;
    for (var y = -2; y <= 1; y = y + 1) {
        for (var x = -2; x <= 1; x = x + 1) {
            let uv = in.uv + vec2<f32>(f32(x), f32(y)) * t;
            let z = load_depth(uv);
            if (z <= 0.0) {
                continue;
            }
            // 1 at equal depth, falling to 0 at ~4 % relative difference.
            let rel = abs(1.0 - z / centre_z);
            let w = max(1.0 - rel * 25.0, 0.0);
            sum = sum + textureSampleLevel(ao_src, ao_smp, uv, 0.0).r * w;
            weight = weight + w;
        }
    }
    if (weight <= 0.0) {
        return vec4<f32>(textureSampleLevel(ao_src, ao_smp, in.uv, 0.0).r, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(sum / weight, 0.0, 0.0, 1.0);
}
