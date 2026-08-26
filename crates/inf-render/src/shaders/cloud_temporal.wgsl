// Cloud temporal resolve (wave SKY2): blend this frame's jittered half-res
// march with the reprojected history, neighbourhood-clamped, into the cloud
// pass's OWN history ping-pong.
//
// Composition: `passes::ShaderKind::Plain` — common_view and nothing else. It
// needs `view_ray` and `view.view_proj`'s inverse to turn the march's own
// distance back into a position, and no atmosphere at all.
//
// ## Why the cloud cannot ride the scene's TAA
//
// `passes::taa` reprojects through the DEPTH PREPASS. A cloud is not in the
// depth prepass — it is a participating medium and writes no depth — so for
// every cloud pixel `taa.wgsl` reads a cleared reverse-Z depth of 0, takes its
// `depth > 0.0` branch's else, and reprojects the pixel to ITSELF. Under a
// static camera that is correct and free; under a turning one the cloud history
// smears, and it is the reason this pass exists rather than a blend weight being
// added to the one that was already there.
//
// ## What it reprojects against
//
// The march's own coverage-weighted mean distance (`cloud_dist.r`), which is the
// closest thing a volume has to a surface. A cloud is not at one distance, so
// this is an approximation with a name: it is exact for a thin deck, wrong by
// the depth of the cloud for a tower seen edge-on, and bounded in consequence by
// the neighbourhood clamp below.
//
// And it reprojects in RENDER-LOCAL coordinates, which the reconstruction above
// makes plain and which nothing else says out loud: `view.eye.xyz` and
// `prev_view_proj` are both local, so a floating-origin rebase moves the frame
// under the history and invalidates it for exactly one frame. That is the same
// bound `passes::taa` has for the same reason, bounded by the same clamp, and it
// is written here because the alternative was leaving it in a memo. (SKY2 audit.)
//
// ## Determinism
//
// OFF by default (`RenderSettings::cloud_temporal`), exactly like `taa`, and for
// the same reason: an accumulating buffer makes a frame a function of the frames
// before it, which a byte-identical golden cannot be. The march's blue-noise
// jitter is NOT off by default — it is a pure function of the level clock and
// the pixel, so it is deterministic on its own; what this pass adds is
// convergence, and what it costs is single-frame reproducibility.

struct CloudTemporal {
    prev_view_proj: mat4x4<f32>,
    // x = history blend weight, y = history valid (>0.5), zw = half-res size px.
    cfg: vec4<f32>,
};

@group(1) @binding(0) var cloud_cur: texture_2d<f32>;
@group(1) @binding(1) var cloud_hist: texture_2d<f32>;
@group(1) @binding(2) var cloud_dist: texture_2d<f32>;
@group(1) @binding(3) var cloud_smp: sampler;
@group(1) @binding(4) var<uniform> ct: CloudTemporal;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let ndc = fullscreen_ndc(i);
    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.ndc = ndc;
    out.uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = vec2<i32>(i32(in.pos.x), i32(in.pos.y));
    let current = textureLoad(cloud_cur, texel, 0);

    // First frame, a resize, or the knob just turned on: take the frame whole.
    if (ct.cfg.y < 0.5) {
        return current;
    }

    // Reprojection. The march's mean distance is along the view ray, which is
    // unit, so the position is the eye plus that many metres of it — no
    // unprojection and no depth-buffer round trip.
    let dist = textureLoad(cloud_dist, texel, 0).r;
    let world = view.eye.xyz + view_ray(in.ndc) * dist;
    var hist_uv = in.uv;
    let pc = ct.prev_view_proj * vec4<f32>(world, 1.0);
    if (pc.w > 0.0) {
        let p = pc.xyz / pc.w;
        hist_uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    }
    // Off-screen last frame ⇒ there is no history for this pixel. Not a clamp —
    // a rejection, because clamping to the edge would drag the frame's border
    // inward every frame the camera turns.
    if (hist_uv.x < 0.0 || hist_uv.x > 1.0 || hist_uv.y < 0.0 || hist_uv.y > 1.0) {
        return current;
    }

    // Neighbourhood clamp over the 3x3 of THIS frame. It is what makes the
    // accumulation converge rather than ghost: the jittered march's per-pixel
    // values bracket the converged answer, so a history inside that bracket is
    // information and a history outside it is a stale cloud.
    let t = 1.0 / ct.cfg.zw;
    var lo = current;
    var hi = current;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let c = textureSampleLevel(
                cloud_cur, cloud_smp, in.uv + vec2<f32>(f32(x), f32(y)) * t, 0.0);
            lo = min(lo, c);
            hi = max(hi, c);
        }
    }
    var history = textureSampleLevel(cloud_hist, cloud_smp, hist_uv, 0.0);
    history = clamp(history, lo, hi);

    return mix(current, history, ct.cfg.x);
}
