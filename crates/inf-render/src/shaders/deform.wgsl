// deform.wgsl — the P22.1 surface-deformation window, shared by every stage that
// has to agree about where the ground is.
//
// The window is ONE fixed-size R32Float texture that follows the camera in whole
// texels (see `deform.rs`): texel `k` is field sample `k`, always, so panning
// translates the window and never resamples it. Depth is metres BELOW the
// authored surface, always >= 0.
//
// `textureLoad` + manual bilinear, exactly like the terrain height texture beside
// it: R32Float then works as UnfilterableFloat everywhere, no FLOAT32_FILTERABLE
// feature and no sampler — which is what keeps headless CI adapters happy.
//
// `dfm.params.x` is 0.0 in every scene with no deformation, so every branch below
// is present-but-false and the arithmetic around it is instruction-for-instruction
// what it was before P22.1.

struct Deform {
    // x = render-local window min X, y = render-local window min Z,
    // z = 1 / texel metres, w = window texels.
    window: vec4<f32>,
    // x = window enabled, y = max depth (m), z = foliage bend gain,
    // w = foliage wind enabled.
    params: vec4<f32>,
    // xy = wind direction XZ (unit or zero), z = wavenumber (rad/m),
    // w = the WHOLE phase offset, already reduced to [0, 2pi) on the CPU in f64.
    //
    // Reduced there and not here, for two reasons the first cut got wrong:
    // `speed * cloud_time_s * k` in f32 loses the whole fractional phase inside a
    // level's day and the sway FREEZES, and the world->render-local offset has to
    // be folded in or the phase RE-ROLLS on every floating-origin rebase (the
    // grass flicks to a new point of its cycle because the camera crossed a 10 m
    // boundary). See `deform.rs::wind_phase`.
    wind: vec4<f32>,
};

// The window itself. R32Float, `DEFORM_WINDOW_TEXELS²`, created once and never
// resized — which is exactly why it adds no component to `passes::ResourceKey`
// (see `deform.rs`).
@group(DFM_TEX_GROUP) @binding(DFM_TEX_BIND) var deform_tex: texture_2d<f32>;
@group(DFM_UNI_GROUP) @binding(DFM_UNI_BIND) var<uniform> dfm: Deform;

fn deform_enabled() -> bool {
    return dfm.params.x > 0.5;
}

// Foliage sway is INDEPENDENT of whether anything has walked on the ground: it
// is driven by `ScatterSettings::foliage_wind`, not by the presence of a live
// deform cell. (Gating it on the window made an ambient effect flick on and off
// with a footprint expiring half a kilometre away.)
fn deform_wind_enabled() -> bool {
    return dfm.params.w > 0.5;
}

// The window's edge fade, [0, 1] — 1 well inside, ramping to 0 at the rim.
//
// The window is 128 m across, so its edge is 64 m from the camera: INSIDE the
// scatter's 120 m full-mesh band and far inside the terrain's draw distance. An
// un-faded cutoff would draw a straight line across the ground where every rut
// stopped and every blade of grass stood back up. Mirrored (and unit-tested) by
// `deform::rim_fade`; the 32 texels are `DEFORM_RIM_TEXELS`.
fn deform_rim_fade(p: vec2<f32>) -> f32 {
    let n = dfm.window.w - 1.0;
    let edge = min(min(p.x, p.y), min(n - p.x, n - p.y));
    return clamp(edge / 32.0, 0.0, 1.0);
}

fn deform_load(ij: vec2<i32>) -> f32 {
    let m = i32(dfm.window.w) - 1;
    let c = clamp(ij, vec2<i32>(0, 0), vec2<i32>(m, m));
    return textureLoad(deform_tex, c, 0).r;
}

// Depth (m, >= 0) at a RENDER-LOCAL position. Zero outside the window — a hard
// zero, not a clamped edge smear, so the deformation stops at the window rim
// instead of streaking the last row of texels across the rest of the world.
//
// Mirrored on the CPU by `deform::deform_depth_reference`, which unit-tests this
// arithmetic without a GPU.
fn deform_depth(local_xz: vec2<f32>) -> f32 {
    if (!deform_enabled()) {
        return 0.0;
    }
    let n = dfm.window.w;
    let p = (local_xz - dfm.window.xy) * dfm.window.z;
    if (p.x < 0.0 || p.y < 0.0 || p.x > n - 1.0 || p.y > n - 1.0) {
        return 0.0;
    }
    let i0 = floor(p);
    let f = p - i0;
    let ii = vec2<i32>(i0);
    let h00 = deform_load(ii);
    let h10 = deform_load(ii + vec2<i32>(1, 0));
    let h01 = deform_load(ii + vec2<i32>(0, 1));
    let h11 = deform_load(ii + vec2<i32>(1, 1));
    let hx0 = mix(h00, h10, f.x);
    let hx1 = mix(h01, h11, f.x);
    return mix(hx0, hx1, f.y) * deform_rim_fade(p);
}

// The horizontal gradient of the deformation, per metre. Points INTO the dent
// (depth increases toward the centre of a print), so foliage leans along its
// negation — out of the footprint, which is what trampled grass does.
fn deform_gradient(local_xz: vec2<f32>) -> vec2<f32> {
    if (!deform_enabled()) {
        return vec2<f32>(0.0, 0.0);
    }
    let e = 1.0 / dfm.window.z;   // one texel, in metres
    let dx = deform_depth(local_xz + vec2<f32>(e, 0.0))
           - deform_depth(local_xz - vec2<f32>(e, 0.0));
    let dz = deform_depth(local_xz + vec2<f32>(0.0, e))
           - deform_depth(local_xz - vec2<f32>(0.0, e));
    return vec2<f32>(dx, dz) / (2.0 * e);
}
