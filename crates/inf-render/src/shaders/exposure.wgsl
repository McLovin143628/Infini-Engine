// Auto exposure (wave VIS1b) — a luminance histogram over the post-HDR buffer,
// reduced to a log-average, turned into a target in stops, and adapted toward at
// a rate the level authors.
//
// ## This file is a transliteration, not a source
//
// The rule lives in `inf_render::settings` as plain Rust — `exposure_bin`,
// `exposure_bin_luminance`, `exposure_log_average`, `exposure_target_ev`,
// `adapt_exposure_ev` — and every one of them is unit-tested without a GPU. What
// runs here has to be the same arithmetic, so the four constants below are read
// back out of this source by `passes::shader_compose_tests` and compared against
// the Rust ones, the idiom the VIS1a audit built for the GGX energy fit. A rule
// that exists only inside a compute shader is a rule nobody can falsify.
//
// ## The clock, and why it is not a frame counter
//
// `params.step.x` is a **level-clock** delta — `ResolvedSky::cloud_time_s`, the
// document's own clock, the same one the wind drifts by and the waves move on.
// It is never a wall clock and never a frame index, which is what makes the
// adaptation a pure function of the document: two runs at the same time of day
// converge through the same sequence of exposures, PIE and shipping included. A
// paused clock hands over `dt = 0`, and a frozen adaptation is the correct
// answer to a frozen world rather than a special case.
//
// ## Determinism inside the dispatch
//
// The histogram's only write is an integer `atomicAdd` of 1, which is
// commutative and associative over the integers, so the bin counts do not depend
// on how the workgroups were scheduled. The reduction is a fixed halving tree
// over exactly 256 lanes, so the float sum has ONE order — not "whatever order
// the hardware retired the lanes in". Both properties are load-bearing: without
// them "the same document renders the same exposure" would be a hope.

struct ExposureParams {
    // x = source width, y = source height (px), z = sample stride, w = unused.
    source: vec4<u32>,
    // x = min scene luminance, y = max scene luminance, z = adaptation speed
    // (stops per second), w = exposure compensation (stops).
    control: vec4<f32>,
    // x = the level-clock delta in seconds, y = history valid (>0.5 ⇒ adapt from
    // `state.v.y`; otherwise snap to the target), zw unused.
    step: vec4<f32>,
};

struct ExposureState {
    // x = the linear exposure multiplier the bloom prefilter and the tonemap
    //     read, compensation already folded in;
    // y = the adapted EV in stops, carried to the next frame (compensation NOT
    //     folded in — an author turning the dial must not make the eye re-adapt);
    // z = the average scene luminance the histogram measured (the gate reads it);
    // w = 1.0 once y is meaningful.
    v: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: ExposureParams;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> state: ExposureState;

// MIRROR: `inf_render::settings::EXPOSURE_BINS` / `EXPOSURE_LOG_MIN` /
// `EXPOSURE_LOG_MAX` / `EXPOSURE_KEY`. Pinned by
// `the_cpu_and_wgsl_exposure_rules_agree`.
const EXPOSURE_BINS: u32 = 256u;
const EXPOSURE_LOG_MIN: f32 = -10.0;
const EXPOSURE_LOG_MAX: f32 = 10.0;
const EXPOSURE_KEY: f32 = 0.18;

var<workgroup> tile: array<atomic<u32>, 256>;
var<workgroup> red_w: array<f32, 256>;
var<workgroup> red_s: array<f32, 256>;

fn luminance(c: vec3<f32>) -> f32 {
    return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

// Which bin a linear luminance falls in. Bin 0 is the BLACK bin — everything at
// or below 2^EXPOSURE_LOG_MIN lands in it, and the reduction ignores it, so an
// unlit backdrop does not vote the exposure open.
fn exposure_bin(l: f32) -> u32 {
    if (!(l > 0.0)) {
        return 0u;
    }
    let l2 = log2(l);
    if (l2 <= EXPOSURE_LOG_MIN) {
        return 0u;
    }
    // Clamped BEFORE the cast, in both copies: an infinity in the HDR buffer
    // would otherwise reach `u32(t * 256)` as an out-of-range float-to-int
    // conversion, which WGSL leaves *indeterminate*.
    let t = clamp((l2 - EXPOSURE_LOG_MIN) / (EXPOSURE_LOG_MAX - EXPOSURE_LOG_MIN), 0.0, 1.0);
    return min(u32(t * f32(EXPOSURE_BINS)), EXPOSURE_BINS - 1u);
}

// The log2 luminance a bin stands for: its centre in LOG space, which is what
// makes the reduction a log-average rather than a biased one.
fn exposure_bin_log(bin: u32) -> f32 {
    let t = (f32(min(bin, EXPOSURE_BINS - 1u)) + 0.5) / f32(EXPOSURE_BINS);
    return EXPOSURE_LOG_MIN + t * (EXPOSURE_LOG_MAX - EXPOSURE_LOG_MIN);
}

@compute @workgroup_size(16, 16, 1)
fn cs_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    atomicStore(&tile[li], 0u);
    workgroupBarrier();

    // One texel per thread, on a fixed stride grid. The stride is what keeps the
    // cost proportional to a QUARTER of the frame rather than to all of it; a
    // regular lattice rather than a jitter because a jittered set would make the
    // frame a function of something other than the scene.
    let stride = max(params.source.z, 1u);
    let coord = gid.xy * stride;
    if (coord.x < params.source.x && coord.y < params.source.y) {
        let c = textureLoad(hdr_tex, vec2<i32>(coord), 0).rgb;
        atomicAdd(&tile[exposure_bin(luminance(c))], 1u);
    }
    workgroupBarrier();

    let n = atomicLoad(&tile[li]);
    if (n > 0u) {
        atomicAdd(&histogram[li], n);
    }
}

@compute @workgroup_size(256, 1, 1)
fn cs_resolve(@builtin(local_invocation_index) li: u32) {
    // Bin 0 contributes nothing — the black bin (see `exposure_bin`).
    var w = 0.0;
    if (li > 0u) {
        w = f32(atomicLoad(&histogram[li]));
    }
    red_w[li] = w;
    red_s[li] = w * exposure_bin_log(li);
    workgroupBarrier();

    // A fixed halving tree: 128, 64, … 1. ONE summation order, so the float
    // result does not depend on lane retirement.
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) {
            red_w[li] = red_w[li] + red_w[li + s];
            red_s[li] = red_s[li] + red_s[li + s];
        }
        workgroupBarrier();
    }

    if (li != 0u) {
        return;
    }

    var avg = 0.0;
    if (red_w[0] > 0.0) {
        avg = exp2(red_s[0] / red_w[0]);
    }

    // The target, in stops. The bounds clamp the SCENE, not the multiplier: below
    // the floor a night frame stays dark instead of being lifted into noise.
    let lo = max(params.control.x, 1e-4);
    let hi = max(params.control.y, lo);
    let l = clamp(avg, lo, hi);
    let target_ev = log2(EXPOSURE_KEY / l);

    // Adapt, linearly in stops — `adaptation_speed` is documented in stops per
    // second and a linear ramp is the only rule that makes that sentence true.
    var ev = target_ev;
    if (params.step.y > 0.5) {
        let step = max(params.control.z, 0.0) * max(params.step.x, 0.0);
        let delta = target_ev - state.v.y;
        ev = state.v.y + clamp(delta, -step, step);
    }

    // Compensation rides on the OUTPUT, not on the adapted value: turning the
    // dial must brighten the frame now, not make the eye re-adapt to it.
    state.v = vec4<f32>(exp2(ev + params.control.w), ev, avg, 1.0);
}
