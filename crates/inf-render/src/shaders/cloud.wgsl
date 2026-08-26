// Volumetric cloud raymarch (P17.3).
//
// A fullscreen pass that runs AFTER all opaque geometry and terrain and BEFORE
// the translucent pass, compositing premultiplied-alpha cloud radiance into the
// MSAA scene target. See `passes::cloud` for why it sits there rather than inside
// the sky pass.
//
// Composition (`passes::ShaderKind::Cloud`): common_view + `atmosphere.wgsl`
// @group(1) @binding(0) + `atmosphere_lut.wgsl` @group(1) bindings 1/2/3 +
// `cloud_noise.wgsl` + `cloud_field.wgsl` @group(1) bindings 4/5/6.
//
// DEPTH — two mechanisms, because one is not enough. The pass writes
// `@builtin(frag_depth)` at the ray's ENTRY into the slab and depth-tests
// `Greater` (reverse-Z) with writes off, which rejects — per MSAA sample, so with
// antialiased silhouettes — every fragment whose geometry is entirely in front of
// the layer. That is NOT the whole occlusion: a 2 km summit under a 1.5-4 km deck
// is INSIDE the slab, sits beyond its entry plane, passes the test, and would
// have the whole marched span (including cloud physically behind the mountain)
// composited over it as a veil. So the march also clamps `t_far` to the scene
// depth for its pixel — see `cloud_geometry_distance` below. It costs early-Z,
// which for a pass that already ray-marches is not a cost worth naming.

// The scene depth, bound as a READ-ONLY attachment and as this texture at once —
// the one arrangement WebGPU permits for reading the depth you are also testing
// against. `textureLoad`ed at an integer texel, never sampled, so it needs no
// sampler and no filterability.
@group(1) @binding(7) var cloud_scene_depth: texture_depth_multisampled_2d;

// The blue-noise tile the first sample's position is offset by (SKY2). See
// `crate::bluenoise` for what makes it blue and why it is generated rather than
// shipped. `textureLoad`ed at an integer texel, wrapped by hand — nothing
// filters it, because interpolating a blue-noise tile destroys the property it
// exists for.
@group(1) @binding(8) var cloud_blue_noise: texture_2d<f32>;

// Longest span the primary march will cover, metres. A ray a degree above the
// horizon would otherwise want hundreds of kilometres of slab; past this the
// cloud is aerial-perspective'd into the sky anyway.
const CLOUD_MAX_MARCH_M: f32 = 30000.0;
// Transmittance below which the march stops: the remaining 1 % cannot change an
// 8-bit pixel.
const CLOUD_MIN_TRANSMITTANCE: f32 = 0.01;
// How many empty samples in a row before the march goes back to long strides.
const CLOUD_EMPTY_RUN: u32 = 4u;
// Ratio between the long (empty-air) stride and the fine (in-cloud) one.
const CLOUD_STRIDE_RATIO: f32 = 3.0;
// Octaves of the Hillaire multiple-scattering approximation.
const CLOUD_MS_OCTAVES: u32 = 3u;
// Exponent of the powder (in-scatter probability) term. 2 is the usual value:
// `1 - T^2 == 1 - exp(-2*tau)`, i.e. the Guerrilla `1 - exp(-2d)` written
// against a transmittance the march has already computed.
const CLOUD_POWDER_K: f32 = 2.0;
// Fraction of the overhead sky that still reaches the BASE of the slab, as the
// diffusion approximation standing in for multiple scattering through the layer.
const CLOUD_AMBIENT_BASE: f32 = 0.45;
// Edge of the blue-noise tile, mirroring `bluenoise::BLUE_NOISE_RES`.
const CLOUD_BLUE_NOISE_RES: i32 = 64;
// The golden-ratio conjugate. Advancing a blue-noise value by this per sequence
// element keeps the *set* of offsets a pixel visits low-discrepancy, so the
// temporal average converges in a handful of elements instead of the dozens a
// re-randomization would need (Roberts' R1 sequence, the standard companion to a
// blue-noise tile).
const CLOUD_JITTER_GOLDEN: f32 = 0.6180339887;

struct CloudOut {
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

// The floating-origin offset, recovered from the view uniform: `grid_axis_viewport.xy`
// is (-origin.x, -origin.z) and `mode_axis.y` is -origin.y. So world = local - those.
// This is why the field is locked to the WORLD and does not slide when the origin
// rebases under a moving camera.
fn cloud_to_world(local: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        local.x - view.grid_axis_viewport.x,
        local.y - view.mode_axis.y,
        local.z - view.grid_axis_viewport.y,
    );
}

// Distance along `dir` (metres) to the nearest geometry in this pixel, or a
// negative value where the depth buffer is still at its reverse-Z clear (0.0 =
// infinity) and there is nothing to stop at.
//
// The unprojected point lies on the view ray by construction, so the along-ray
// distance is `dot(hit - eye, dir)` — used rather than `length` because it stays
// signed, which is what distinguishes "geometry behind the eye" (impossible after
// a depth test, but cheap to be safe about) from "geometry at zero range".
fn cloud_geometry_distance(ndc: vec2<f32>, texel: vec2<i32>, dir: vec3<f32>) -> f32 {
    let d = textureLoad(cloud_scene_depth, texel, 0);
    if (d <= 0.0) {
        return -1.0;
    }
    let hit = unproject(ndc, d);
    return dot(hit - view.eye.xyz, dir);
}

// Hillaire's multiple-scattering approximation: N octaves, each with half the
// previous octave's attenuation exponent, scattering weight and phase
// eccentricity. Without it a thick cloud's interior goes to soot — single
// scattering simply has nowhere for the light to come from, while a real cloud's
// interior is lit almost entirely by light that has bounced several times.
// The march's start offset for this pixel, in `[0, 1)` of one base step.
//
// WHY IT IS NOT A FRAME INDEX. Starting every pixel at the same fraction of a
// step makes the integration error coherent across the screen, which is the
// banding this pass shipped with. The standard fix is to offset each pixel by a
// blue-noise value rotated by the frame counter — and a frame counter is exactly
// what a byte-identical determinism gate cannot have, because it counts how long
// the process has been up rather than where the level is. So the rotation is
// driven by `atmos.cloud_color.w`, the **level clock's** jitter phase
// (`clouds::jitter_phase`): a paused clock is a frozen pattern and the same
// document renders the same pixels, while a running clock walks the sequence at
// 240 Hz for the temporal pass to average.
fn cloud_jitter(texel: vec2<i32>) -> f32 {
    let t = vec2<i32>(
        texel.x % CLOUD_BLUE_NOISE_RES,
        texel.y % CLOUD_BLUE_NOISE_RES,
    );
    let b = textureLoad(cloud_blue_noise, t, 0).r;
    return fract(b + atmos.cloud_color.w * CLOUD_JITTER_GOLDEN);
}

fn cloud_sun_energy(sun_t: f32, cos_t: f32, g: f32, sun_y: f32) -> f32 {
    // ── the powder term (SKY2) ──
    //
    // Beer's law alone says a point with nothing between it and the sun is fully
    // lit, and that is the wrong answer for a *volume*: light reaches such a
    // point, but there is almost no material there to scatter it toward the eye,
    // so the point contributes almost nothing. The consequence of leaving it out
    // is the flat, airbrushed look — a cloud whose sunward face and whose shaded
    // face differ only in a shadow, with no dark rim where the medium thins.
    //
    // Written against the transmittance the light march already returned:
    // `1 - T^K == 1 - exp(-K*tau)`, so the in-scatter probability costs one
    // multiply rather than a second march.
    let st = clamp(sun_t, 0.0, 1.0);
    let powder = 1.0 - pow(st, CLOUD_POWDER_K);
    // Blended by view direction, which is not a fudge but the term's own
    // geometry: the deficit is only *visible* from the side that faces away from
    // the sun. Looking INTO the sun the eye sees light forward-scattered by the
    // near surface, which is the silver lining — and applying powder there would
    // erase the one effect the two-lobe phase function exists to produce.
    let facing = clamp(-cos_t * 0.5 + 0.5, 0.0, 1.0);
    // ...and gated on the sun being up, on exactly the threshold
    // `cloud_sun_transmittance` early-outs at. Below it the 1.0 that function
    // returns means "no march was run", not "no material" — reading it as an
    // optical depth of zero would give a powder of zero and take the whole
    // single-scattering term off a night sky, which measured as a 45 % drop in
    // `clouds_night`'s starless cloud brightness before this line existed.
    var single = 1.0;
    if (sun_y > 1e-3) {
        single = mix(1.0, powder, facing);
    }

    var e = 0.0;
    var att = 1.0;
    var sca = 1.0;
    var ecc = 1.0;
    for (var n = 0u; n < CLOUD_MS_OCTAVES; n = n + 1u) {
        // Powder applies to the FIRST octave only. The later octaves stand in for
        // light that has already bounced several times inside the layer, which
        // arrives from every direction rather than along the sun ray — darkening
        // those would take back exactly the term that keeps a thick deck's
        // interior from going to soot.
        var p = 1.0;
        if (n == 0u) {
            p = single;
        }
        e = e + sca * pow(max(sun_t, 0.0), att) * cloud_phase(cos_t, g * ecc) * p;
        att = att * 0.5;
        sca = sca * 0.5;
        ecc = ecc * 0.5;
    }
    return e;
}

@fragment
fn fs(in: VsOut) -> CloudOut {
    var out: CloudOut;
    out.color = vec4<f32>(0.0);
    out.depth = 0.0;

    let dir = view_ray(in.ndc);
    let eye_world = cloud_to_world(view.eye.xyz);
    let bottom = atmos.cloud_layer.x;
    let top = atmos.cloud_layer.y;

    // ── slab intersection ──
    // A ray within a whisker of horizontal never leaves the layer, which would be
    // an unbounded march; the cap below handles it, but the divide must not blow
    // up first.
    let dy = dir.y;
    var t_near = 0.0;
    var t_far = CLOUD_MAX_MARCH_M;
    if (abs(dy) > 1e-4) {
        let t0 = (bottom - eye_world.y) / dy;
        let t1 = (top - eye_world.y) / dy;
        t_near = max(min(t0, t1), 0.0);
        t_far = max(t0, t1);
    } else if (eye_world.y < bottom || eye_world.y > top) {
        // Horizontal ray, outside the slab: it never enters.
        return out;
    }
    if (t_far <= 0.0) {
        return out; // the slab is entirely behind the camera
    }
    t_far = min(t_far, t_near + CLOUD_MAX_MARCH_M);

    // Stop the march at the nearest geometry. This is what makes a mountain
    // INSIDE the cloud deck occlude the cloud behind it instead of being veiled
    // by it — the hardware depth test cannot, because the fragment's depth is the
    // slab's entry plane, which is genuinely in front of the summit.
    //
    // Sample 0 only: on a pixel partially covered by intersecting geometry the
    // march length comes from one sample while the hardware test still resolves
    // coverage per sample, which is a sub-pixel discrepancy at a silhouette and
    // not worth four loads to remove.
    let texel = vec2<i32>(i32(in.pos.x), i32(in.pos.y));
    let geo = cloud_geometry_distance(in.ndc, texel, dir);
    if (geo > 0.0) {
        t_far = min(t_far, geo);
        if (t_far <= t_near) {
            return out; // the geometry is in front of the whole layer
        }
    }
    let span = t_far - t_near;
    if (span <= 0.0) {
        return out;
    }

    // ── depth: the slab entry point ──
    // Held at least a metre ahead of the eye so the projection's w stays finite
    // when the camera is inside the layer.
    let entry_local = view.eye.xyz + dir * max(t_near, 1.0);
    let clip = view.view_proj * vec4<f32>(entry_local, 1.0);
    let entry_depth = clamp(clip.z / max(clip.w, 1e-6), 0.0, 1.0);

    // ── march ──
    let sun = normalize(atmos.sun_dir.xyz);
    let cos_t = dot(dir, sun);
    let g = atmos.cloud_wind.z;
    let r = atmos.planet.z;
    // Sun radiance at cloud altitude, extinguished by the air above the layer.
    // This is what reddens clouds at sunset for free.
    //
    // `atmos.params.y` is the sky exposure (`sky_intensity * SKY_EXPOSURE_CALIBRATION`),
    // and it belongs here for exactly the reason its doc comment gives: it is the
    // single calibration between the engine's arbitrary light units and the
    // exposure the renderer is tuned for, and it multiplies *sky* radiance — which
    // a cloud is — but never the directional light the PBR loop sees. Without it
    // the march produces a phase-function-scaled radiance (~1/4pi of the sun's)
    // and an overcast noon renders as soot, which is the classic tell of a
    // volumetric that has never been calibrated against its own sky. The ambient
    // term needs no such factor: it comes out of the sky-view LUT with the
    // exposure already baked in.
    // The extra `CLOUD_PI` is a unit conversion, not a fudge. The in-scattered
    // source term is `E * p(theta)` where E is IRRADIANCE, while the engine hands
    // the cloud a radiance-like `colour * intensity` — the same number the PBR
    // loop divides by pi for a Lambertian surface. Multiplying by pi undoes that
    // convention, and it is what makes a sunlit cloud top brighter than the
    // Lambertian ground under the same sun, which is the correct relationship and
    // the one a phase-function-only march gets wrong by a factor of ~3.
    //
    // The transmittance is sampled at the LAYER'S radius, not the camera's. That
    // is the difference between a dusk sky that works and one that does not: with
    // the sun two degrees up, the path to a viewer on the ground is opaque in
    // every channel, while the path to a cloud three kilometres up is above a
    // measurable slice of the atmosphere and still carries red. Sampling at the
    // camera makes twilight clouds go grey at exactly the moment they should be
    // the brightest thing in the sky. The layer is thin enough (a couple of km
    // against a hundred) that its mid-altitude serves for the whole slab.
    let layer_r = atmos.planet.x + (bottom + top) * 0.5 * 1e-3;
    let sun_radiance = atmos.sun_color.rgb
        * atmos_sample_transmittance(layer_r, sun.y)
        * atmos.params.y
        * CLOUD_PI;
    // Ambient. A cloud's TOP sees the upper hemisphere, which the zenith stands in
    // for; its BASE sees the lower one, which at any interesting time of day is
    // dominated by the bright band around the horizon — and at dusk that band is
    // orange while the zenith is deep blue. Interpolating between the two by
    // height is what stops a twilight deck from being lit as if it were noon.
    let sky_up = atmos_sample_skyview(r, vec3<f32>(0.0, 1.0, 0.0));
    let sky_horizon = atmos_sample_skyview(r, normalize(vec3<f32>(sun.x, 0.12, sun.z)));
    let albedo = atmos.cloud_color.rgb;
    let ambient_scale = atmos.cloud_wind.w;

    let n = u32(atmos.cloud_march.x);
    let light_steps = u32(atmos.cloud_march.y);
    let base_dt = span / f32(max(n, 1u));

    var transmittance = 1.0;
    var scattered = vec3<f32>(0.0);
    var depth_sum = 0.0;
    var weight_sum = 0.0;
    // The first sample lands a blue-noise fraction of a base step into the slab
    // instead of half of one. Everything downstream inherits the offset — the
    // coarse search, the rewind on contact and the fine march all step from
    // here — so the integration error stops being the same error in every pixel
    // and becomes fine grain instead of concentric shells.
    var t = t_near + base_dt * cloud_jitter(texel);
    var stride = CLOUD_STRIDE_RATIO;
    var empty = 0u;

    for (var i = 0u; i < n; i = i + 1u) {
        if (t >= t_far || transmittance < CLOUD_MIN_TRANSMITTANCE) {
            break;
        }
        let dt = base_dt * stride;
        let p = eye_world + dir * t;
        let sigma = cloud_density(p);
        if (sigma > 0.0) {
            if (stride > 1.5) {
                // Contact while striding: rewind one long step and re-enter at
                // fine resolution. `i` still advances, so the loop terminates
                // whatever the field does.
                t = max(t - dt, t_near);
                stride = 1.0;
                empty = 0u;
                continue;
            }
            let h = clamp(cloud_height_frac(p.y), 0.0, 1.0);
            let sun_t = cloud_sun_transmittance(p, sun, light_steps);
            let energy = cloud_sun_energy(sun_t, cos_t, g, sun.y);
            // The ambient a point inside the cloud sees is the sky ABOVE the
            // layer, which the cloud above it has largely occluded. A
            // single-scattering march has nowhere to get the light back from, so
            // the depth falloff is an explicit diffusion approximation rather
            // than an integral: at the slab's base the deck still keeps
            // `CLOUD_AMBIENT_BASE` of the sky, because in reality that light
            // arrives by multiple scattering through the layer. Take it out and an
            // overcast deck renders as soot — which is the single most common
            // failure of a correct-but-incomplete volumetric.
            let amb = mix(sky_horizon, sky_up, h) * ambient_scale
                * mix(CLOUD_AMBIENT_BASE, 1.0, h);
            let luminance = (sun_radiance * energy + amb) * albedo;

            // Energy-conserving analytic integration of the step (Hillaire): the
            // in-scatter over a segment of constant extinction is
            // `L * (1 - exp(-sigma*dt))`, not `L * sigma * dt`. The difference is
            // visible banding at low step counts, which is precisely the regime a
            // Low tier runs in.
            let step_t = exp(-sigma * dt);
            scattered = scattered + transmittance * luminance * (1.0 - step_t);
            let w = transmittance * (1.0 - step_t);
            depth_sum = depth_sum + t * w;
            weight_sum = weight_sum + w;
            transmittance = transmittance * step_t;
            t = t + dt;
            empty = 0u;
        } else {
            t = t + dt;
            empty = empty + 1u;
            if (empty > CLOUD_EMPTY_RUN) {
                stride = CLOUD_STRIDE_RATIO;
            }
        }
    }

    let alpha = clamp(1.0 - transmittance, 0.0, 1.0);
    if (alpha <= 0.0) {
        return out;
    }

    // ── aerial perspective on distant clouds ──
    // Same v1 model the lit passes use (`atmos_apply`): the eye→cloud segment
    // treated as homogeneous at the camera's local extinction, with the
    // in-scattered colour from the sky-view LUT. Height fog is deliberately NOT
    // applied — fog is a ground-level authored layer and a cloud four kilometres
    // up is above it by construction.
    var color = scattered;
    if (atmos.params.z > 0.0 && weight_sum > 0.0) {
        let dist_km = (depth_sum / weight_sum) * 1e-3;
        let sigma_air = atmos_extinction(max(r - atmos.planet.x, 0.0));
        let t_air = exp(-sigma_air * dist_km * atmos.params.z);
        let scatter_dir = normalize(vec3<f32>(dir.x, max(dir.y, 0.0), dir.z) + vec3<f32>(0.0, 1e-4, 0.0));
        let in_sky = atmos_sample_skyview(r, scatter_dir);
        // Premultiplied: the in-scatter fills only the fraction of the pixel the
        // cloud actually covers.
        color = color * t_air + in_sky * (vec3<f32>(1.0) - t_air) * alpha;
    }

    out.color = vec4<f32>(color, alpha);
    out.depth = entry_depth;
    return out;
}
