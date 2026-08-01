// Sky-view LUT bake (P17.2). One texel per view direction, in the Hillaire
// parameterization: `v` is the zenith angle warped so the horizon lands exactly
// on the 0.5 seam, `u` is the azimuth relative to the sun. The value is the
// in-scattered radiance the camera sees looking that way, already multiplied by
// the sky-intensity exposure.
//
// Composed with `atmosphere.wgsl` at @group(0) @binding(0) by
// `passes::shader_source("atmos_skyview")`.
//
// This LUT depends on the sun direction and the camera's altitude, so it is
// re-baked whenever either moves — but it is deliberately tiny (192x108 at the
// High tier). ~20 k threads x 32 march steps replaces a full per-pixel march for
// every sky pixel on screen, at any output resolution.

@group(0) @binding(1) var atmos_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var atmos_t_lut: texture_2d<f32>;
@group(0) @binding(3) var atmos_t_smp: sampler;

// Isotropic multiple-scattering approximation (v1). A fraction of what
// single-scatters at each sample is treated as re-radiating in all directions.
// Without it the zenith is unnaturally dark and twilight collapses to black too
// fast, because ~all of the sky's blue past the first bounce is missing. A
// proper 32x32 multiple-scattering LUT (Hillaire section 5.3) is the documented
// follow-up; this constant is the honest stand-in for it.
const ATMOS_MULTI_SCATTER: f32 = 0.022;

fn transmittance_to_sun(r: f32, mu: f32) -> vec3<f32> {
    let uv = atmos_transmittance_params_to_uv(r, mu);
    return textureSampleLevel(atmos_t_lut, atmos_t_smp, uv, 0.0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn cs_skyview(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(atmos_out);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
        / vec2<f32>(f32(dims.x), f32(dims.y));

    let r = atmos.planet.z;
    let ground = atmos.planet.x;
    let sun = normalize(atmos.sun_dir.xyz);
    let dir = atmos_skyview_uv_to_dir(r, uv, sun);

    // The march frame: the planet centre is the coordinate origin and the
    // camera's local up is +Y, which is exactly how `dir.y` was built.
    let origin = vec3<f32>(0.0, r, 0.0);
    let t_ground = atmos_ray_sphere_near(origin, dir, ground);
    let t_top = atmos_ray_sphere_far(origin, dir, atmos.planet.y);
    var t_max = t_top;
    if (t_ground > 0.0) {
        t_max = t_ground;
    }

    let cos_theta = dot(dir, sun);
    let phase_r = atmos_phase_rayleigh(cos_theta);
    let phase_m = atmos_phase_mie(cos_theta, atmos.mie.w);
    let sun_irradiance = atmos.sun_color.rgb;

    let steps = max(u32(atmos.ozone_shape.w), 4u);
    let dt = max(t_max, 0.0) / f32(steps);
    var luminance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);

    for (var i = 0u; i < steps; i = i + 1u) {
        let t = (f32(i) + 0.5) * dt;
        let p = origin + dir * t;
        let radius = max(length(p), ground);
        let alt = radius - ground;
        let up = p / radius;

        let sc = atmos_scattering(alt);
        let ext = max(atmos_extinction(alt), vec3<f32>(1e-9));
        let t_sun = transmittance_to_sun(radius, dot(up, sun));

        // Direct single scattering (phase-weighted) + the isotropic
        // multiple-scattering stand-in.
        let direct = (sc.rayleigh * phase_r + vec3<f32>(sc.mie * phase_m)) * t_sun;
        // The multiple-scattering stand-in is driven by the *least attenuated*
        // channel rather than by `t_sun` itself: after several bounces the light
        // has been redistributed across wavelengths, so at sunset — when direct
        // blue is long gone — the sky keeps a warm ambient instead of snapping to
        // black the instant the sun dips.
        let ms_light = vec3<f32>(max(t_sun.r, max(t_sun.g, t_sun.b)));
        let multi = (sc.rayleigh + vec3<f32>(sc.mie)) * ATMOS_MULTI_SCATTER * ms_light;
        let in_scatter = (direct + multi) * sun_irradiance;

        // Hillaire's energy-conserving step integral: the closed-form integral of
        // `in_scatter * exp(-ext * s)` over the step, rather than a midpoint
        // rectangle. This is what keeps a 32-step march from banding.
        let step_transmittance = exp(-ext * dt);
        luminance = luminance
            + throughput * (in_scatter - in_scatter * step_transmittance) / ext;
        throughput = throughput * step_transmittance;
    }

    // Ground bounce for the rows below the horizon: a Lambertian planet lit by
    // the sun. Without it the lower half of the sky-view LUT is pure black and
    // aerial perspective on downhill geometry loses its warm bounce.
    if (t_ground > 0.0) {
        let hit = origin + dir * t_ground;
        let n = normalize(hit);
        let n_dot_l = max(dot(n, sun), 0.0);
        let t_sun = transmittance_to_sun(ground, dot(n, sun));
        luminance = luminance
            + throughput * sun_irradiance * t_sun * (n_dot_l * atmos.ozone_shape.z / ATMOS_PI);
    }

    let out = luminance * atmos.params.y;
    textureStore(atmos_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(out, 1.0));
}
