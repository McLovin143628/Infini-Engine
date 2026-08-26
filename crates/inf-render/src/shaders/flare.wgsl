// Sun glare, ghosts, halo and streak (wave VIS1b) — the lens the camera has not
// had.
//
// ## A relocation, not an invention
//
// Three pieces of this pass already existed in `underwater.wgsl`, written for
// light shafts seen from below the sea:
//
//   * the sun's SCREEN POSITION — a point a very long way along the sun
//     direction, projected, with `clip.w <= 0` rejected as a hard zero rather
//     than a guess (`underwater.wgsl`, `uw_shafts`);
//   * a RADIAL GATHER toward that position with an exponential per-tap decay;
//   * the CPU-side sun the whole thing hangs off (`SunParams`), which since
//     P17.1 is projected from the level's time of day rather than a constant.
//
// What changes is the source term. Underwater, each tap asks an analytic
// question ("does this ray reach the surface unobstructed?") because the surface
// shader renders the deep colour from below and a luminance gather would have
// nothing to gather. Above water there IS something to gather: the frame's own
// bright pixels. So the gather is the same kernel over a different question, and
// the occlusion test moves from "did the sea floor cut this ray off" to "is the
// sun itself behind something".
//
// ## Half resolution, and why that is not a compromise
//
// Every term here is low-frequency by construction: a veiling glare is a smooth
// falloff, a ghost is a defocused image of the aperture, a halo is a ring and an
// anamorphic streak is a horizontal smear. There is no detail at the pixel scale
// to lose, and half res is four times fewer of the twenty-odd taps each pixel
// spends. The tonemap upsamples it bilinearly along with the bloom it sits
// beside.
//
// ## Off is a clear
//
// With `FlareSettings::enabled` false the node clears this target to black and
// records nothing else, exactly as the bloom node does — and the tonemap does not
// sample it at all, because its add sits behind a uniform branch. So every golden
// runs the command stream it always did.

struct Flare {
    // x = veiling intensity, y = ghost count, z = halo strength, w = streak
    // strength.
    params: vec4<f32>,
    // x = the bright-pass threshold in exposed units, yzw unused. Neither the
    // exposure nor the sun's visibility is here: the exposure is a binding (in
    // auto mode the CPU does not know it) and the visibility needs the depth
    // buffer, so this pass measures it (see `flare_sun_visibility`).
    tune: vec4<f32>,
    // xy = this pass's target size in px, zw = its texel size.
    dims: vec4<f32>,
};

@group(1) @binding(0) var<uniform> fl: Flare;
@group(1) @binding(1) var fl_scene: texture_2d<f32>;
@group(1) @binding(2) var fl_smp: sampler;
@group(1) @binding(3) var fl_depth: texture_depth_multisampled_2d;
struct FlareExposure {
    // x = the frame's linear exposure multiplier; see `inf_render::exposure`.
    v: vec4<f32>,
};
@group(1) @binding(4) var<uniform> fl_exposure: FlareExposure;

// Fixed, and it must stay fixed for the reason `UW_SHAFT_TAPS` is: the veiling
// term is a mean over exactly this many taps, so changing it changes the
// picture, and varying it per frame would make the frame a function of something
// other than the scene.
const FLARE_VEIL_TAPS: i32 = 16;
// Per-tap decay along the gather. `0.94^16` is 0.37, so the tail contributes a
// third of the head — a glare that fades rather than a bar.
const FLARE_VEIL_DECAY: f32 = 0.94;
// How far along the pixel->sun line the gather reaches, as a fraction.
const FLARE_VEIL_REACH: f32 = 0.6;
// The most ghosts the chain will draw, whatever the record says. A `u32` field
// with no ceiling is an unbounded loop reachable from a level file.
const FLARE_MAX_GHOSTS: i32 = 8;
// The occlusion kernel's half-width: 1 gives 3x3 = nine depth taps. Every pixel
// computes the same nine, which is redundant and is still the cheap answer — the
// alternative is a compute prepass and a fourth buffer for one scalar, and nine
// coherent `textureLoad`s from one cached neighbourhood cost less than that
// costs to schedule.
const FLARE_OCCLUSION_TAPS: i32 = 1;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let ndc = fullscreen_ndc(vi);
    out.pos = vec4<f32>(ndc, 1.0, 1.0);
    out.uv = ndc * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return out;
}

// Where the sun is on screen, and whether it is on screen at all.
//
// `w` is 1.0 when there is a screen position and 0.0 when the sun is behind the
// camera — a hard zero rather than a guess, which is `uw_shafts`' own ruling.
fn flare_sun_uv() -> vec3<f32> {
    let sun = normalize(view.sun_dir.xyz);
    let clip = view.view_proj * vec4<f32>(view.eye.xyz + sun * 1.0e4, 1.0);
    if (clip.w <= 0.0) {
        return vec3<f32>(0.5, 0.5, 0.0);
    }
    let uv = (clip.xy / clip.w) * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return vec3<f32>(uv, 1.0);
}

// Is the sun itself visible, `[0,1]`?
//
// Reverse-infinite Z: a depth of exactly 0 means nothing rasterized there, i.e.
// sky. So the fraction of taps around the sun's screen position that read zero
// IS the fraction of the disc that is unoccluded — no unprojection, no distance
// comparison, and correct for every kind of geometry because it reads the MSAA
// scene depth every pass wrote rather than the rigid-only prepass.
fn flare_sun_visibility(sun_uv: vec2<f32>) -> f32 {
    // The SCENE's size, not this target's: `fl_depth` is the full-resolution MSAA
    // scene depth, so its texels are full-resolution texels.
    let size = view.grid_axis_viewport.zw;
    var open = 0.0;
    var total = 0.0;
    let spread = 6.0 / max(size.x, 1.0);
    for (var y = -FLARE_OCCLUSION_TAPS; y <= FLARE_OCCLUSION_TAPS; y = y + 1) {
        for (var x = -FLARE_OCCLUSION_TAPS; x <= FLARE_OCCLUSION_TAPS; x = x + 1) {
            let uv = sun_uv + vec2<f32>(f32(x), f32(y)) * spread;
            total = total + 1.0;
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                // Off screen is not occluded — the glare of a sun just past the
                // frame edge is the whole reason a veiling term exists.
                open = open + 1.0;
                continue;
            }
            let texel = vec2<i32>(clamp(uv * size, vec2<f32>(0.0), size - vec2<f32>(1.0)));
            if (textureLoad(fl_depth, texel, 0) <= 0.0) {
                open = open + 1.0;
            }
        }
    }
    return open / max(total, 1.0);
}

// The frame's bright part, in EXPOSED units — the same space the bloom prefilter
// keys in, and for the same reason: what glares is what the viewer can see, not
// what the radiance buffer happens to hold.
fn flare_bright(uv: vec2<f32>) -> vec3<f32> {
    let c = textureSampleLevel(fl_scene, fl_smp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb
        * fl_exposure.v.x;
    return max(c - vec3<f32>(fl.tune.x), vec3<f32>(0.0));
}

// A ghost is a defocused image of the aperture, so it is tinted by where in the
// frame it landed — the chromatic spread a real lens gives its ghosts.
fn flare_ghost_tint(t: f32) -> vec3<f32> {
    return vec3<f32>(1.0 - 0.35 * t, 1.0 - 0.15 * abs(t - 0.5), 0.65 + 0.35 * t);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let sun = flare_sun_uv();
    var acc = vec3<f32>(0.0);

    // ── veiling glare: the radial gather, relocated ──
    //
    // Only when the sun has a screen position AND is not fully occluded. The
    // visibility rides on the whole term rather than on the tap, because a
    // veiling glare is light scattered inside the lens by the SOURCE — put the
    // sun behind a building and the glare goes with it.
    //
    // The visibility is measured INSIDE the branch: it costs nine depth taps and
    // a level with the ghosts on and the glare off should not pay for them.
    // Legal in non-uniform control flow because `flare_sun_visibility` takes no
    // derivatives — every fetch is a `textureLoad`.
    if (sun.z > 0.5 && fl.params.x > 0.0) {
        let vis = flare_sun_visibility(sun.xy);
        if (vis > 0.001) {
            let step = (sun.xy - in.uv) * (FLARE_VEIL_REACH / f32(FLARE_VEIL_TAPS));
            var pos = in.uv;
            var decay = 1.0;
            var veil = vec3<f32>(0.0);
            for (var i = 0; i < FLARE_VEIL_TAPS; i = i + 1) {
                pos = pos + step;
                veil = veil + flare_bright(pos) * decay;
                decay = decay * FLARE_VEIL_DECAY;
            }
            acc = acc + veil * (fl.params.x * vis / f32(FLARE_VEIL_TAPS));
        }
    }

    // ── the ghost chain ──
    //
    // Each ghost is the bright pass sampled at the point reflected through the
    // frame centre and scaled — the standard aperture-ghost construction. It
    // does NOT depend on the sun having a screen position: a ghost is an image
    // of whatever is bright, and a window at the edge of frame throws one too.
    let centre = vec2<f32>(0.5, 0.5);
    let to_centre = centre - in.uv;
    let ghosts = min(i32(fl.params.y), FLARE_MAX_GHOSTS);
    for (var g = 1; g <= ghosts; g = g + 1) {
        let t = f32(g) / f32(FLARE_MAX_GHOSTS);
        let scale = 0.4 + 1.6 * t;
        let uv = in.uv + to_centre * (1.0 + scale);
        // A ghost fades toward the frame edge, which is what keeps the chain from
        // laying a hard rectangle over the corners.
        let fade = 1.0 - clamp(length(uv - centre) * 1.6, 0.0, 1.0);
        acc = acc + flare_bright(uv) * flare_ghost_tint(t) * (fade * fade * 0.25);
    }

    // ── the halo ──
    //
    // One ring: the bright pass sampled at a fixed radius along the vector to the
    // centre, weighted by how close this pixel already is to that radius.
    if (fl.params.z > 0.0) {
        let dir = normalize(to_centre + vec2<f32>(1e-6, 0.0));
        let r = length(to_centre);
        let uv = centre - dir * 0.32;
        let band = 1.0 - clamp(abs(r - 0.32) * 7.0, 0.0, 1.0);
        acc = acc + flare_bright(uv) * (band * band * fl.params.z * 0.5);
    }

    // ── the anamorphic streak ──
    //
    // A horizontal-only gather: an anamorphic lens squeezes one axis, so a point
    // source smears along the other. Seven taps rather than the veil's sixteen —
    // it is a line, not a field.
    if (fl.params.w > 0.0) {
        var streak = vec3<f32>(0.0);
        var w = 0.0;
        for (var i = -3; i <= 3; i = i + 1) {
            let o = f32(i) * fl.dims.z * 9.0;
            let k = 1.0 - abs(f32(i)) / 4.0;
            streak = streak + flare_bright(in.uv + vec2<f32>(o, 0.0)) * k;
            w = w + k;
        }
        acc = acc + streak * (fl.params.w / max(w, 1e-5));
    }

    return vec4<f32>(acc, 1.0);
}
