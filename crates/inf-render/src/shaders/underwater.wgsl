// Underwater post (P20.3) — what the world looks like from inside the water.
//
// ## One absorption story, seen from both sides
//
// The surface shader (`water.wgsl`) already measures a water column and applies
// Beer-Lambert extinction to whatever is behind it:
//
//     behind = scene * exp(-a * column) + deep * (1 - exp(-a * column))
//
// This pass is that expression with the camera moved to the other side of the
// surface. Same `a` (the body's authored extinction, m⁻¹), same `deep`, same
// exponential — nothing is re-tuned for the inside, because there is only one
// medium. What changes is where the column is measured: from the eye to whatever
// is in front of it, capped by the surface itself.
//
// ## The surface caps the column
//
// A ray that rises through the still-water plane LEAVES the medium there, so the
// column it crossed ends at the plane rather than at the sea floor two hundred
// metres behind it. Without that cap the bright disc overhead — Snell's window,
// the whole reason an underwater shot reads as underwater — would be fogged by
// the depth of whatever the ray eventually hit, and there would be nothing left
// for the shafts to come out of.
//
// The cap uses the STILL-WATER plane, not the displaced surface: a per-pixel
// Gerstner inverse in a post pass would be a second surface evaluation, and the
// error it saves is a wave amplitude on a distance that is already tens of
// metres. The v1 ledger in ROADMAP §12 P20.3 names it.
//
// ## Light shafts v1
//
// A fixed 24-tap radial gather toward the sun's screen position — the classic
// screen-space god-ray, deterministic by construction: fixed tap count, no
// jitter, no noise, no history.
//
// What each tap gathers is **not** the frame's own luminance, which is the usual
// recipe. From below, the v1 surface shader renders the deep colour (its Fresnel
// and reflection terms were written for a camera above the water), so a
// luminance gather would have nothing to gather. Instead each tap asks an
// analytic question: *does this pixel's view ray reach the surface without the
// sea floor cutting it off, and how close is it to the sun?* That makes the
// shaft's root a patch of surface, its taper a decay, and its END the silhouette
// of whatever occludes it — which is exactly what a light shaft is, and it does
// not depend on the surface shading being bright.
//
// Its limits are real and are listed in the ROADMAP §12 P20.3 ledger; what it is
// not is a volumetric march.

struct Underwater {
    // x = strength [0,1] (the waterline ramp), y = eye depth below the surface
    // (m), z = render-local still-water level Y, w = fog distance where the depth
    // buffer holds nothing (m).
    params: vec4<f32>,
    // rgb = Beer-Lambert extinction (m⁻¹) — the body's own, the same numbers
    // `water.wgsl` absorbs with; a = shaft intensity.
    absorption: vec4<f32>,
    // rgb = deep-water colour; a = per-tap shaft decay.
    deep: vec4<f32>,
    // x = the sun lobe exponent seen from below, y = shaft reach (fraction of the
    // distance to the sun), z = shafts enabled (0/1), w = shaft tint depth (m).
    shafts: vec4<f32>,
};

@group(1) @binding(0) var<uniform> uw: Underwater;
@group(1) @binding(1) var uw_scene: texture_2d<f32>;
@group(1) @binding(2) var uw_smp: sampler;
@group(1) @binding(3) var uw_depth: texture_depth_multisampled_2d;

// Fixed, and it must stay fixed: the shaft term is a mean over exactly this many
// taps, so changing it changes the picture, and varying it per frame would make
// the frame a function of something other than the scene.
const UW_SHAFT_TAPS: i32 = 24;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let ndc = fullscreen_ndc(vi);
    // No depth attachment on this pass, so the clip Z is free; 1.0 keeps it
    // inside the reverse-Z range rather than at the infinite plane.
    out.pos = vec4<f32>(ndc, 1.0, 1.0);
    out.uv = ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return out;
}

// uv → NDC, the one place that flip lives.
fn uw_ndc(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

// The view ray through an NDC point. One unprojection rather than
// `common_view`'s two: the eye is already in the uniform, so a single finite
// depth is enough — and this runs 24 times per pixel.
fn uw_ray(ndc: vec2<f32>) -> vec3<f32> {
    return normalize(unproject(ndc, 0.8) - view.eye.xyz);
}

// How much sunlight enters the medium through the surface at this pixel, `[0,1]`.
//
// Zero unless the pixel's ray (a) rises, (b) reaches the still-water plane, and
// (c) is not cut off by geometry first — condition (c) is what gives a shaft an
// END, and why a rock underwater casts one.
fn uw_source(uv: vec2<f32>, size: vec2<f32>, sun: vec3<f32>) -> f32 {
    let ndc = uw_ndc(uv);
    let r = uw_ray(ndc);
    if (r.y <= 1e-4) {
        return 0.0;
    }
    let to_surface = (uw.params.z - view.eye.y) / r.y;
    if (to_surface <= 0.0) {
        return 0.0;
    }
    let texel = vec2<i32>(clamp(uv * size, vec2<f32>(0.0, 0.0), size - vec2<f32>(1.0, 1.0)));
    let d = textureLoad(uw_depth, texel, 0);
    if (d > 0.0 && length(unproject(ndc, d) - view.eye.xyz) < to_surface) {
        return 0.0; // occluded before the ray could leave the water
    }
    // The sun, broadened by the surface it came through.
    return pow(max(dot(r, sun), 0.0), uw.shafts.x);
}

// Screen-space sun shafts. Returns an ADDITIVE radiance, already tinted by the
// water; the caller scales it by the downwelling term.
fn uw_shafts(uv: vec2<f32>) -> vec3<f32> {
    let sun = normalize(view.sun_dir.xyz);
    let size = view.grid_axis_viewport.zw;

    // Where the sun is on screen: a point far along the sun direction, projected.
    // Refraction at the surface would bend this (the sun sits inside Snell's
    // window from below); v1 does not, and the ledger says so.
    let clip = view.view_proj * vec4<f32>(view.eye.xyz + sun * 1.0e4, 1.0);
    if (clip.w <= 0.0) {
        // The sun is behind the camera: there is no screen position to gather
        // toward, so there are no shafts. A hard zero rather than a guess.
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let sun_uv = (clip.xy / clip.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    let step = (sun_uv - uv) * (uw.shafts.y / f32(UW_SHAFT_TAPS));

    var pos = uv;
    var decay = 1.0;
    var acc = 0.0;
    for (var i = 0; i < UW_SHAFT_TAPS; i = i + 1) {
        pos = pos + step;
        acc = acc
            + uw_source(clamp(pos, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0)), size, sun) * decay;
        decay = decay * uw.deep.a;
    }
    acc = acc / f32(UW_SHAFT_TAPS);

    // A shaft is sunlight that has already crossed water, so it carries the same
    // extinction the fog does — evaluated at one shaft-length, because the light
    // did not travel the path this view ray did.
    let tint = exp(-uw.absorption.rgb * uw.shafts.w);
    return tint * (acc * uw.absorption.a);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let scene = textureSampleLevel(uw_scene, uw_smp, in.uv, 0.0).rgb;
    let ndc = uw_ndc(in.uv);
    let ray = uw_ray(ndc);

    // How much water this pixel looks through. Reverse-infinite-Z: depth 0 is the
    // far plane / nothing rasterized, and must never be unprojected.
    let d = textureLoad(uw_depth, vec2<i32>(in.pos.xy), 0);
    var column = uw.params.w;
    if (d > 0.0) {
        column = length(unproject(ndc, d) - view.eye.xyz);
    }
    if (ray.y > 1e-4) {
        // The ray leaves the medium at the surface.
        let to_surface = (uw.params.z - view.eye.y) / ray.y;
        if (to_surface > 0.0) {
            column = min(column, to_surface);
        }
    }

    let transmit = exp(-uw.absorption.rgb * column);
    // Downwelling: the light filling the medium at the camera's depth has already
    // crossed that much water on its way down, so a deep camera sits in a darker,
    // bluer medium. Same extinction, applied vertically — which is what makes
    // "deeper is darker" a consequence of the absorption rather than a curve.
    let down = exp(-uw.absorption.rgb * uw.params.y);
    var color = scene * transmit + (uw.deep.rgb * down) * (1.0 - transmit);

    if (uw.shafts.z > 0.5) {
        color = color + uw_shafts(in.uv) * down;
    }

    // The waterline ramp: at zero submersion this returns the scene untouched, so
    // the whole-screen switch has nothing to pop.
    return vec4<f32>(mix(scene, color, clamp(uw.params.x, 0.0, 1.0)), 1.0);
}
