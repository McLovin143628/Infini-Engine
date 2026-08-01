// Precipitation v1 (P17.4) — a world-anchored rain/snow particle layer.
//
// Composition (`passes::ShaderKind::Precip`): common_view + `atmosphere.wgsl`
// @group(1) @binding(0) + `atmosphere_lut.wgsl` @group(1) bindings 1/2/3 +
// `cloud_noise.wgsl` (for `cloud_hash` / `cloud_hash_unit` — binding-free) +
// this file. It declares NO bindings of its own: the whole parameter block rides
// in the atmosphere uniform, so the pass costs one bind group that already
// existed.
//
// GEOMETRY. There is no vertex buffer and no instance buffer. The pass draws
// `6 * count` vertices with `draw(0..6, 0..count)`, so `instance_index` is the
// particle and `vertex_index` is one of two triangles' corners. A particle's
// entire state is derived from its index and the level's clock, which is what
// makes it deterministic and what makes it free of any CPU-side simulation.
//
// DEPTH. Ordinary hardware depth test against the READ-ONLY scene depth
// attachment (`Greater`, reverse-Z), depth writes off. The cloud pass needed to
// additionally *read* the depth texture because a raymarch has to clamp `t_far`
// at geometry; a particle is a flat quad at a single depth, so the hardware test
// is the whole answer — and it resolves per MSAA sample, which a manual load
// would not.
//
// BLEND ORDER. Alpha compositing is order-dependent and the particles are not
// sorted. That is deliberate rather than overlooked: the draw order is the
// instance index, which is fixed, so the frame is still bit-deterministic — and
// at PRECIP_ALPHA the difference between two orderings of overlapping drops is
// below the low bit of an 8-bit pixel. Sorting 48 000 quads per frame to change
// nothing visible would be the wrong trade.
//
// TIME. Every displacement arrives pre-wrapped from the CPU in `f64` (see
// `precip::PrecipParams::offsets`), so nothing here accumulates and nothing here
// depends on a frame index. Two runs at the same time of day draw the same rain.

// How much of the sky's radiance a droplet scatters toward the eye.
//
// ABOVE 1 on purpose, and the reason is geometry rather than taste: a drop
// gathers light from the whole hemisphere and re-emits it toward the eye, while
// `atmos_sample_skyview` returns the radiance of the ONE direction it is seen
// against. At a gain of exactly 1 a raindrop composites over the sky to precisely
// the sky -- invisible by construction, which is an identity rather than a look.
// Slightly above 1 makes rain a faint brightening against the sky (what a
// downpour does to a horizon) and clearly visible against dark ground.
const PRECIP_SKY_GAIN: f32 = 1.15;
// Weight of the forward-scattering sun highlight — what makes rain visible as
// bright streaks when you look toward a low sun through it.
const PRECIP_SUN_GAIN: f32 = 0.6;
// Sharpness of that highlight. 8 is a narrow lobe: the streaks light up within
// ~30 degrees of the sun and are sky-lit elsewhere.
const PRECIP_SUN_LOBE: f32 = 8.0;
// Minimum on-screen WIDTH of a particle, in pixels.
//
// A raindrop really is sub-pixel: 1.6 cm across at 20 m is a hundredth of a
// pixel at 720p, and a sub-pixel primitive rasterizes to nothing (or, worse, to
// a flickering scatter of hit samples that changes with the camera). So the quad
// is widened until it covers at least this much, and its alpha is divided by the
// same factor — which conserves the integrated coverage, so a drop that is
// widened 40x is 40x fainter and the sheet of rain has the density it should.
// This is the standard treatment for sub-pixel geometry (hair, wires, power
// lines); it is a rasterization fix, not an artistic one.
const PRECIP_MIN_PIXELS: f32 = 1.3;
// Largest alpha compensation the widening is allowed to apply.
//
// Full compensation is right for ONE primitive that happens to be thin. It is
// wrong for a *sample of a population*: the 48 000 particles stand in for the
// millions of drops actually in the air, so a distant particle represents many
// more drops than a near one, and dividing its alpha by the whole widening factor
// under-represents the sheet until it disappears. Capping it is the explicit
// statement that past this point a particle is a stand-in, not a drop.
const PRECIP_MAX_ALPHA_COMPENSATION: f32 = 4.0;
// The hash salt, mirroring `precip::PRECIP_HASH_SALT`.
const PRECIP_SALT: u32 = 0x9e3779b9u;

struct PrecipVsOut {
    @builtin(position) pos: vec4<f32>,
    // Quad-local coordinates in [-1, 1]: x across the particle, y along its
    // velocity. The fragment shape is a pure function of these.
    @location(0) uv: vec2<f32>,
    // Unit direction from the eye to this particle (render-local == world
    // direction), for the sky-radiance lookup.
    @location(1) view_dir: vec3<f32>,
    // Alpha compensation for the sub-pixel widening (see PRECIP_MIN_PIXELS):
    // 1 for a particle that was already at least that wide, smaller for one that
    // had to be inflated.
    @location(2) alpha_scale: f32,
};

// Reduce `v` into [-period/2, period/2). CPU mirror: `precip::wrap_signed`.
// `fract` is `x - floor(x)`, i.e. always non-negative, which is what makes this
// the Euclidean remainder rather than the truncating one.
fn precip_wrap_signed(v: f32, period: f32) -> f32 {
    let half = period * 0.5;
    return fract((v + half) / period) * period - half;
}

// Base position of particle `i` inside the box, metres in [0, box).
// CPU mirror: `precip::precip_base` — the same three `cloud_hash` draws at the
// same salts, so the CPU reference and this agree bit for bit (24-bit unit
// values, no float anywhere in the hash itself).
fn precip_base(i: u32) -> vec3<f32> {
    let bx = atmos.precip_box.x;
    let by = atmos.precip_box.y;
    return vec3<f32>(
        cloud_hash_unit(cloud_hash(i, 0u, 0u, PRECIP_SALT)) * bx,
        cloud_hash_unit(cloud_hash(i, 1u, 0u, PRECIP_SALT)) * by,
        cloud_hash_unit(cloud_hash(i, 2u, 0u, PRECIP_SALT)) * bx,
    );
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PrecipVsOut {
    var out: PrecipVsOut;

    // Two triangles over the corners of a unit square.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );
    let uv = corners[vi];
    out.uv = uv;

    let box_xz = atmos.precip_box.x;
    let box_y = atmos.precip_box.y;
    let base = precip_base(ii);

    // The world-anchoring wrap. `atmos.precip_eye.xyz` is the camera's WORLD
    // position reduced modulo the box (computed in f64 on the CPU), so
    //     p_world = eye_world + wrap_signed(base + drift - eye_mod, box)
    // is congruent to `base + drift` modulo the box: the lattice is locked to
    // world space, the particle does not slide with the camera, and the floating
    // origin never enters the expression, so a rebase cannot pop it.
    //
    // `precip_phase.z` is the distance already FALLEN, hence the minus sign.
    let drift = vec3<f32>(atmos.precip_phase.x, -atmos.precip_phase.z, atmos.precip_phase.y);
    let d = base + drift - atmos.precip_eye.xyz;
    let offset = vec3<f32>(
        precip_wrap_signed(d.x, box_xz),
        precip_wrap_signed(d.y, box_y),
        precip_wrap_signed(d.z, box_xz),
    );
    // Render-local centre: `view.eye.xyz` is already render-local and the offset
    // is smaller than one box, so no precision is lost here.
    let centre = view.eye.xyz + offset;

    // Orient the quad ALONG the particle's velocity — wind horizontally, fall
    // speed downward — and widen it perpendicular to both that and the eye ray.
    // For snow `half_length == radius`, so the very same expression produces a
    // round flake: one code path, no branch, and the rain→snow blend is
    // continuous in geometry as well as in speed.
    let to_eye = view.eye.xyz - centre;
    let eye_dir = to_eye / max(length(to_eye), 1e-4);
    let vel = vec3<f32>(atmos.precip_wind.x, -atmos.precip.w, atmos.precip_wind.y);
    var fall_dir = vec3<f32>(0.0, -1.0, 0.0);
    if (length(vel) > 1e-4) {
        fall_dir = normalize(vel);
    }
    // Degenerate when the drop falls straight at the eye; any perpendicular does
    // as well as another there, and the quad is sub-pixel anyway.
    var axis = cross(fall_dir, eye_dir);
    if (length(axis) < 1e-4) {
        axis = cross(fall_dir, vec3<f32>(1.0, 0.0, 0.0));
    }
    var side = normalize(axis) * atmos.precip_box.w;
    let along = fall_dir * atmos.precip_box.z;

    // Sub-pixel widening. Measure the quad's half-width in PIXELS by projecting
    // its centre and one edge, then inflate until it reaches PRECIP_MIN_PIXELS
    // and divide the alpha by the same factor. Behind the camera (`w <= 0`) there
    // is nothing meaningful to measure and the fragment is clipped anyway, so the
    // widening is skipped rather than fed a negative divisor.
    let clip_c = view.view_proj * vec4<f32>(centre, 1.0);
    let clip_s = view.view_proj * vec4<f32>(centre + side, 1.0);
    var widen = 1.0;
    if (clip_c.w > 1e-4 && clip_s.w > 1e-4) {
        let d_ndc = clip_s.xy / clip_s.w - clip_c.xy / clip_c.w;
        let px = length(d_ndc * 0.5 * view.grid_axis_viewport.zw);
        widen = max(1.0, PRECIP_MIN_PIXELS / max(px, 1e-6));
        side = side * widen;
    }
    out.alpha_scale = 1.0 / min(widen, PRECIP_MAX_ALPHA_COMPENSATION);

    let world = centre + side * uv.x + along * uv.y;
    out.pos = view.view_proj * vec4<f32>(world, 1.0);
    out.view_dir = -eye_dir;
    return out;
}

@fragment
fn fs(in: PrecipVsOut) -> @location(0) vec4<f32> {
    // Shape: a radial falloff in quad-local space. For rain the quad is long and
    // thin, so this is a soft streak; for snow it is square, so it is a soft
    // disc. One expression, two looks, because the aspect ratio does the work.
    let r = length(in.uv);
    let shape = clamp(1.0 - r, 0.0, 1.0);
    if (shape <= 0.0) {
        discard;
    }
    let alpha = shape * shape * atmos.precip_phase.w * in.alpha_scale;

    // Radiance. A drop is a tiny scatterer of the sky it hangs in, so its colour
    // is the sky-view LUT in the view direction — which makes rain grey under
    // overcast, gold at sunset and nearly black at night for free, with no
    // per-time-of-day tuning. The sun term is a forward-scattering highlight,
    // sampled at the CAMERA's radius (unlike the cloud pass, which samples at the
    // layer's, because a raindrop twenty metres away really is in the camera's
    // own air).
    //
    // `params.y` is the sky exposure — the single calibration between the
    // engine's light units and the exposure the renderer is tuned for. It belongs
    // on the sun term for the same reason the cloud march's does: what is being
    // computed is scattered sky-like radiance, not a surface response.
    let sky = atmos_sample_skyview(atmos.planet.z, in.view_dir);
    let sun = normalize(atmos.sun_dir.xyz);
    let sun_t = atmos_sample_transmittance(atmos.planet.z, sun.y);
    let forward = pow(clamp(dot(in.view_dir, -sun), 0.0, 1.0), PRECIP_SUN_LOBE);
    let sun_l = atmos.sun_color.rgb * sun_t * atmos.params.y * forward * PRECIP_SUN_GAIN;
    let color = (sky * PRECIP_SKY_GAIN + sun_l) * atmos.precip_color.rgb;

    // PREMULTIPLIED alpha, matching the cloud pass: the pipeline blends
    // `src + dst*(1-a)`, so the source must already carry its own coverage.
    return vec4<f32>(color * alpha, alpha);
}
