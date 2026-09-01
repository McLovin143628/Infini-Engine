// Physical atmosphere library (P17.2) — the GPU mirror of `crate::atmosphere`.
//
// Composed into every consumer by `passes::atmosphere_source(group, binding)`,
// which substitutes the ATMOS_GROUP / ATMOS_BIND tokens so the same source serves
// the two LUT compute passes (@group(0)), the sky pass (@group(1)) and the shared
// env bind of every lit pass (@group(2) or @group(3)).
//
// Nothing here samples a texture: this file is the *medium* (densities,
// extinction, the LUT parameterizations, phase functions, the star hash). LUT
// sampling lives in `atmosphere_lut.wgsl`, which only the passes that actually
// bind both LUTs include.
//
// UNITS — kilometres and km^-1 throughout, matching `crate::atmosphere`'s module
// docs. The single exception is the `fog` block, which is SI metres because it is
// authored against world geometry; every fog function below says so.

// Mirrors `passes::sky_lut::AtmosphereGpu` field for field.
struct AtmosphereData {
    // x = enabled, y = sky intensity, z = aerial-perspective strength,
    // w = gradient tint strength [0,1].
    params: vec4<f32>,
    // rgb = Rayleigh scattering (km^-1), w = Rayleigh scale height (km).
    rayleigh: vec4<f32>,
    // x = Mie scattering (km^-1, turbidity-scaled), y = Mie absorption (km^-1),
    // z = Mie scale height (km), w = Mie phase g.
    mie: vec4<f32>,
    // rgb = ozone absorption at the tent peak (km^-1), w = transmittance-bake
    // ray-march step count.
    ozone: vec4<f32>,
    // x = ozone centre (km), y = ozone half-width (km), z = ground albedo,
    // w = sky-view-bake ray-march step count.
    ozone_shape: vec4<f32>,
    // x = ground radius (km), y = top radius (km), z = camera radius (km),
    // w = camera world altitude (m).
    planet: vec4<f32>,
    // xyz = unit direction toward the sun, w = cos(sun disc half-angle).
    sun_dir: vec4<f32>,
    // rgb = sun irradiance (colour x intensity), w = sun disc limb exponent.
    sun_color: vec4<f32>,
    // xyz = unit direction toward the moon, w = cos(moon disc half-angle).
    moon_dir: vec4<f32>,
    // rgb = moon radiance (colour x intensity), w = lunar phase [0,1).
    moon_color: vec4<f32>,
    // x = star intensity, y = star cells per cube-face axis, z = night fade
    // [0,1], w unused.
    stars: vec4<f32>,
    // SI METRES: x = fog density (m^-1), y = fog falloff (m^-1),
    // z = fog reference height (m), w = render-local -> world Y offset (m).
    fog: vec4<f32>,
    // rgb = fog tint, w unused.
    fog_color: vec4<f32>,

    // ── volumetric clouds (P17.3), SI METRES ──
    //
    // The cloud block rides here rather than in a uniform of its own, so the
    // whole feature costs the lit passes exactly ONE new binding (the shadow
    // map). See `passes::sky_lut::AtmosphereGpu`.
    //
    // x = clouds enabled, y = coverage [0,1], z = cloud type [0,1],
    // w = erosion detail strength [0,1].
    clouds: vec4<f32>,
    // x = layer bottom (m), y = layer top (m), z = extinction at full density
    // (m^-1), w = seed as an integer-valued f32 (u32(...) to recover it).
    cloud_layer: vec4<f32>,
    // x = wind offset X (m), y = wind offset Z (m), z = forward phase g,
    // w = ambient multiplier.
    cloud_wind: vec4<f32>,
    // x = primary march steps, y = sun-transmittance steps, z = shadow-bake
    // steps, w = 1 when this tier reads the detail volume.
    cloud_march: vec4<f32>,
    // x = world-shadow strength [0,1], y = shadow-map world extent (m),
    // zw = shadow-map centre in WORLD X/Z metres (texel-quantized).
    cloud_shadow: vec4<f32>,
    // rgb = cloud droplet albedo tint, w unused.
    cloud_color: vec4<f32>,

    // ── precipitation (P17.4), SI METRES ──
    //
    // Rides here for the same reason the cloud block does: the precipitation
    // pass then binds nothing at all beyond what the sky already binds.
    //
    // x = particle count (integer-valued), y = intensity [0,1],
    // z = snowiness [0,1] (0 = rain, 1 = snow), w = fall speed (m/s).
    precip: vec4<f32>,
    // x = horizontal box period (m), y = vertical box period (m),
    // z = particle half-length along its velocity (m), w = particle radius (m).
    precip_box: vec4<f32>,
    // x = wind drift X (m, wrapped into the box), y = drift Z (m, wrapped),
    // z = distance already fallen (m, wrapped), w = alpha scale.
    precip_phase: vec4<f32>,
    // xyz = the camera's WORLD position modulo the box (m) — the world-anchoring
    // term, reduced on the CPU in f64; w unused.
    precip_eye: vec4<f32>,
    // rgb = droplet albedo tint, w unused.
    precip_color: vec4<f32>,
    // x = wind X (m/s), y = wind Z (m/s) — the RAW rates, not the wrapped
    // drift, because the fall direction is `normalize(wind_x, -fall, wind_z)`
    // and a wrapped drift is no longer proportional to the wind. zw unused.
    precip_wind: vec4<f32>,
};

@group(ATMOS_GROUP) @binding(ATMOS_BIND) var<uniform> atmos: AtmosphereData;

const ATMOS_PI: f32 = 3.14159265359;

// ── the medium ──────────────────────────────────────────────────────────────

// Total extinction (scattering + absorption) at `alt` km, per channel, km^-1.
// CPU mirror: `crate::atmosphere::extinction`.
fn atmos_extinction(alt: f32) -> vec3<f32> {
    let dr = exp(-alt / max(atmos.rayleigh.w, 1e-3));
    let dm = exp(-alt / max(atmos.mie.z, 1e-3));
    let doz = max(0.0, 1.0 - abs(alt - atmos.ozone_shape.x) / max(atmos.ozone_shape.y, 1e-3));
    let mie_ext = (atmos.mie.x + atmos.mie.y) * dm;
    return atmos.rayleigh.rgb * dr + vec3<f32>(mie_ext) + atmos.ozone.rgb * doz;
}

// Rayleigh + Mie scattering coefficients at `alt` km (absorption excluded — only
// scattering re-radiates toward the eye).
struct AtmosScatter {
    rayleigh: vec3<f32>,
    mie: f32,
};

fn atmos_scattering(alt: f32) -> AtmosScatter {
    var s: AtmosScatter;
    s.rayleigh = atmos.rayleigh.rgb * exp(-alt / max(atmos.rayleigh.w, 1e-3));
    s.mie = atmos.mie.x * exp(-alt / max(atmos.mie.z, 1e-3));
    return s;
}

// Far root of a ray/sphere intersection (sphere centred at the coordinate
// origin, i.e. the planet centre). Negative when the ray misses.
fn atmos_ray_sphere_far(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> f32 {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    return -b + sqrt(disc);
}

// Near root, or negative when the ray misses or the sphere is behind.
fn atmos_ray_sphere_near(ro: vec3<f32>, rd: vec3<f32>, radius: f32) -> f32 {
    let b = dot(ro, rd);
    let c = dot(ro, ro) - radius * radius;
    let disc = b * b - c;
    if (disc < 0.0) {
        return -1.0;
    }
    let t = -b - sqrt(disc);
    if (t < 0.0) {
        return -1.0;
    }
    return t;
}

// Whether a ray leaving radius `r` at zenith cosine `mu` hits the planet.
fn atmos_hits_ground(r: f32, mu: f32) -> bool {
    let bottom = atmos.planet.x;
    return mu < 0.0 && (r * r * (mu * mu - 1.0) + bottom * bottom) >= 0.0;
}

// ── phase functions ─────────────────────────────────────────────────────────

fn atmos_phase_rayleigh(cos_t: f32) -> f32 {
    return (3.0 / (16.0 * ATMOS_PI)) * (1.0 + cos_t * cos_t);
}

// Cornette-Shanks: the physically-behaved Henyey-Greenstein variant (it keeps the
// Rayleigh-like 1+cos^2 lobe as g -> 0 instead of going isotropic).
fn atmos_phase_mie(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    let num = 3.0 * (1.0 - g2) * (1.0 + cos_t * cos_t);
    let den = 8.0 * ATMOS_PI * (2.0 + g2) * pow(max(1.0 + g2 - 2.0 * g * cos_t, 1e-4), 1.5);
    return num / den;
}

// ── transmittance LUT parameterization (Bruneton; see crate::atmosphere) ────

// uv -> (radius km, zenith cosine).
fn atmos_transmittance_uv_to_params(uv: vec2<f32>) -> vec2<f32> {
    let bottom = atmos.planet.x;
    let top = atmos.planet.y;
    let h = sqrt(max(top * top - bottom * bottom, 0.0));
    let rho = h * clamp(uv.y, 0.0, 1.0);
    let r = sqrt(rho * rho + bottom * bottom);
    let d_min = top - r;
    let d_max = rho + h;
    let d = d_min + clamp(uv.x, 0.0, 1.0) * (d_max - d_min);
    var mu = 1.0;
    if (d > 0.0) {
        mu = clamp((h * h - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
    }
    return vec2<f32>(r, mu);
}

// (radius km, zenith cosine) -> uv.
fn atmos_transmittance_params_to_uv(r_in: f32, mu_in: f32) -> vec2<f32> {
    let bottom = atmos.planet.x;
    let top = atmos.planet.y;
    let r = clamp(r_in, bottom, top);
    let mu = clamp(mu_in, -1.0, 1.0);
    let h = sqrt(max(top * top - bottom * bottom, 0.0));
    let rho = sqrt(max(r * r - bottom * bottom, 0.0));
    let disc = max(r * r * (mu * mu - 1.0) + top * top, 0.0);
    let d = max(-r * mu + sqrt(disc), 0.0);
    let d_min = top - r;
    let d_max = rho + h;
    var u = 0.0;
    if (d_max - d_min > 0.0) {
        u = clamp((d - d_min) / (d_max - d_min), 0.0, 1.0);
    }
    var v = 0.0;
    if (h > 0.0) {
        v = clamp(rho / h, 0.0, 1.0);
    }
    return vec2<f32>(u, v);
}

// ── the planet's own shadow (wave GTA1) ────────────────────────────────────
//
// The parameterization above maps `u` over `d ∈ [d_min, d_max]`, where `d_max`
// is the HORIZON-TANGENT ray — the longest path that still misses the ground.
// Every direction below that tangent is off the end of the axis, so a
// clamp-to-edge sampler (which is itself load-bearing, see `passes::sky_lut`)
// hands back the tangent texel: the reddest entry in the whole table, ~9.7 % of
// the red channel surviving against ~0.004 % of the blue. Read as "the sun's
// light at this sample" that is a lie of the worst kind — it says a sun eight
// degrees underground still delivers sunset red, which is exactly the band that
// used to sit on the midnight horizon.
//
// So every sampler whose `mu` is a cosine toward a BODY (rather than along a
// view ray) multiplies by this: 1 while the body is clear of the sample's own
// local horizon, 0 once it is fully below it, smoothed across the body's angular
// diameter because a disc sets over a finite time rather than instantly.
//
// The local horizon is per-SAMPLE, which is the whole reason this is not a
// global "is the sun up" flag: a parcel of air 60 km up sees the sun until it is
// 7.8° below the ground's horizon, and that difference IS civil twilight. A flag
// on the sun's elevation would snap the whole sky to black at sunset; this keeps
// the glow exactly as high as the lit air actually is.
//
// `disc_cos` is the body's `cos(angular radius)` — `atmos.sun_dir.w` or
// `atmos.moon_dir.w`. Bruneton's `GetTransmittanceToSun` in one function.
fn atmos_horizon_visibility(r: f32, mu: f32, disc_cos: f32) -> f32 {
    let bottom = atmos.planet.x;
    let rr = max(r, bottom + 1e-4);
    let sin_h = min(bottom / rr, 1.0);
    let cos_h = -sqrt(max(1.0 - sin_h * sin_h, 0.0));
    // Half-width in cosine space: the body's angular radius projected onto the
    // `mu` axis, where `d(mu)/d(theta) = -sin_h`. Floored so a degenerate disc
    // still fades over something rather than snapping.
    let ang = sqrt(max(1.0 - disc_cos * disc_cos, 0.0));
    let w = max(sin_h * ang, 1e-4);
    return smoothstep(-w, w, mu - cos_h);
}

// Ray-marched transmittance from radius `r` at zenith cosine `mu` to the top of
// the atmosphere. CPU mirror: `crate::atmosphere::transmittance_to_top`.
fn atmos_transmittance_integral(r: f32, mu: f32, steps: u32) -> vec3<f32> {
    let origin = vec3<f32>(0.0, r, 0.0);
    let sin_t = sqrt(max(1.0 - mu * mu, 0.0));
    let dir = vec3<f32>(sin_t, mu, 0.0);

    if (atmos_ray_sphere_near(origin, dir, atmos.planet.x) > 0.0) {
        return vec3<f32>(0.0);
    }
    let t_max = atmos_ray_sphere_far(origin, dir, atmos.planet.y);
    if (t_max <= 0.0) {
        return vec3<f32>(1.0);
    }
    let n = max(steps, 2u);
    let dt = t_max / f32(n);
    var depth = vec3<f32>(0.0);
    for (var i = 0u; i < n; i = i + 1u) {
        let t = (f32(i) + 0.5) * dt;
        let p = origin + dir * t;
        depth = depth + atmos_extinction(length(p) - atmos.planet.x) * dt;
    }
    return exp(-depth);
}

// ── sky-view LUT parameterization (Hillaire) ───────────────────────────────

// Zenith angle (radians from straight up) at which the horizon sits.
fn atmos_zenith_horizon_angle(r: f32) -> f32 {
    let bottom = atmos.planet.x;
    let rr = max(r, bottom + 1e-4);
    let cos_beta = clamp(sqrt(max(rr * rr - bottom * bottom, 0.0)) / rr, -1.0, 1.0);
    return ATMOS_PI - acos(cos_beta);
}

// Sky-view v for a view zenith cosine, warped so the horizon lands on v = 0.5.
fn atmos_skyview_v(r: f32, view_zenith_cos: f32) -> f32 {
    let horizon = atmos_zenith_horizon_angle(r);
    let beta = ATMOS_PI - horizon;
    let theta = acos(clamp(view_zenith_cos, -1.0, 1.0));
    if (theta < horizon) {
        var coord = 0.0;
        if (horizon > 0.0) {
            coord = theta / horizon;
        }
        coord = 1.0 - coord;
        coord = 1.0 - sqrt(max(coord, 0.0));
        return coord * 0.5;
    }
    var coord = 0.0;
    if (beta > 0.0) {
        coord = (theta - horizon) / beta;
    }
    return sqrt(clamp(coord, 0.0, 1.0)) * 0.5 + 0.5;
}

// Sky-view u from the horizontal-plane cosine between the view and the sun.
fn atmos_skyview_u(light_view_cos: f32) -> f32 {
    return sqrt(max(-clamp(light_view_cos, -1.0, 1.0) * 0.5 + 0.5, 0.0));
}

// Inverse of `atmos_skyview_v`: the view zenith cosine a LUT row stands for.
fn atmos_skyview_v_to_cos(r: f32, v: f32) -> f32 {
    let horizon = atmos_zenith_horizon_angle(r);
    let beta = ATMOS_PI - horizon;
    if (v < 0.5) {
        var coord = 1.0 - 2.0 * v;
        coord = 1.0 - coord * coord;
        return cos(horizon * coord);
    }
    let coord = 2.0 * v - 1.0;
    return cos(horizon + beta * coord * coord);
}

// The sun's horizontal (azimuth) direction — the +X of the frame the sky-view LUT
// is built in. Biased so a sun exactly at the zenith (or nadir) still yields a
// finite, stable axis; at that point every azimuth is equivalent anyway.
fn atmos_sun_azimuth_axis(sun: vec3<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(sun.x, 0.0, sun.z) + vec3<f32>(1e-6, 0.0, 0.0));
}

// The world view direction a sky-view LUT texel stands for. Built in the frame
// whose +Y is the zenith and whose +X is the sun's azimuth, so the LUT is
// azimuthally symmetric about the sun and one texture axis suffices.
fn atmos_skyview_uv_to_dir(r: f32, uv: vec2<f32>, sun: vec3<f32>) -> vec3<f32> {
    let cos_z = atmos_skyview_v_to_cos(r, uv.y);
    let sin_z = sqrt(max(1.0 - cos_z * cos_z, 0.0));
    // u^2 undoes the sqrt warp; the sign flip matches `atmos_skyview_u`.
    let light_cos = -(uv.x * uv.x * 2.0 - 1.0);
    let light_sin = sqrt(max(1.0 - light_cos * light_cos, 0.0));
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let sun_h = atmos_sun_azimuth_axis(sun);
    let side = cross(up, sun_h);
    return normalize(up * cos_z + (sun_h * light_cos + side * light_sin) * sin_z);
}

// World view direction -> sky-view LUT uv, given the sun direction. `up` is the
// local zenith (+Y in engine axes; at game scale the camera never moves far
// enough for the planet's curvature to tilt it).
fn atmos_skyview_uv(r: f32, dir: vec3<f32>, sun: vec3<f32>) -> vec2<f32> {
    let v = atmos_skyview_v(r, dir.y);
    let dh = normalize(vec3<f32>(dir.x, 0.0, dir.z) + vec3<f32>(1e-6, 0.0, 0.0));
    let sh = atmos_sun_azimuth_axis(sun);
    return vec2<f32>(atmos_skyview_u(dot(dh, sh)), v);
}

// ── height fog (SI METRES) ─────────────────────────────────────────────────

// Optical depth of the exponential height fog along a segment from world
// altitude y0 to y1 (metres) of length `distance` (metres). Closed form; CPU
// mirror: `crate::atmosphere::height_fog_optical_depth`.
fn atmos_fog_optical_depth(y0: f32, y1: f32, distance: f32) -> f32 {
    if (atmos.fog.x <= 0.0 || distance <= 0.0) {
        return 0.0;
    }
    let falloff = max(atmos.fog.y, 0.0);
    let base = atmos.fog.x * exp(-falloff * (y0 - atmos.fog.z));
    let dy = y1 - y0;
    let k = falloff * dy;
    var shape = 1.0;
    if (abs(k) >= 1e-4) {
        shape = (1.0 - exp(-k)) / k;
    }
    // Clamp before the caller's exp(): a camera far below a steep fog plane would
    // otherwise produce inf, and one non-finite texel poisons the bloom chain.
    return min(base * shape * distance, 64.0);
}

// ── stars ──────────────────────────────────────────────────────────────────

// A deterministic integer hash. Deliberately trig-free and integer-only: the
// house LAW is that f32 std/WGSL trig is not bit-portable, and a starfield that
// shifts between a golden bless and a CI re-render would be a permanently
// flaky gate. This is pure bit arithmetic, so every adapter agrees exactly.
fn atmos_hash3(a: u32, b: u32, c: u32) -> u32 {
    var h = a * 0x8da6b343u + b * 0xd8163841u + c * 0xcb1ab31fu;
    h = h ^ (h >> 15u);
    h = h * 0x2c1b3c6du;
    h = h ^ (h >> 12u);
    h = h * 0x297a2d39u;
    h = h ^ (h >> 15u);
    return h;
}

fn atmos_hash_unit(h: u32) -> f32 {
    return f32(h & 0x00ffffffu) / 16777216.0;
}

// Procedural starfield in direction `dir` (unit). Cube-face cell grid: the major
// axis picks a face, the remaining two components pick a cell, and one hash per
// cell gives that cell's star a jittered position and a magnitude. Screen-stable
// (a pure function of the world direction), density-scaled by the render tier.
//
// Only the containing cell is tested — stars are jittered into the middle 70 %
// of their cell, so a star never straddles an edge and no 3x3 neighbourhood
// search is needed.
fn atmos_starfield(dir: vec3<f32>, cells: f32) -> f32 {
    let a = abs(dir);
    var face = 0u;
    var uv = vec2<f32>(0.0);
    if (a.x >= a.y && a.x >= a.z) {
        face = select(1u, 0u, dir.x > 0.0);
        uv = vec2<f32>(dir.y, dir.z) / a.x;
    } else if (a.y >= a.z) {
        face = select(3u, 2u, dir.y > 0.0);
        uv = vec2<f32>(dir.x, dir.z) / a.y;
    } else {
        face = select(5u, 4u, dir.z > 0.0);
        uv = vec2<f32>(dir.x, dir.y) / a.z;
    }
    let grid = (uv * 0.5 + 0.5) * cells;
    let cell = floor(grid);
    let frac = grid - cell;

    let h = atmos_hash3(u32(i32(cell.x) + 4096), u32(i32(cell.y) + 4096), face * 7919u);
    // ~1 cell in 22 carries a star; denser than that reads as television static.
    if (atmos_hash_unit(h) > 0.045) {
        return 0.0;
    }
    // Jitter into the middle 20 % of the cell: with a 0.32-cell point-spread
    // radius the star reaches at most 0.08..0.92, so it never straddles an edge
    // and a single-cell test is exact (no 3x3 neighbourhood search).
    let px = 0.4 + 0.2 * atmos_hash_unit(atmos_hash3(h, 1u, 0u));
    let py = 0.4 + 0.2 * atmos_hash_unit(atmos_hash3(h, 2u, 0u));
    let mag = atmos_hash_unit(atmos_hash3(h, 3u, 0u));

    // Distance in cell units. The cubed falloff and the cubed magnitude together
    // give a plausible brightness histogram: a few obvious stars, many faint.
    let d = length(frac - vec2<f32>(px, py));
    let falloff = max(1.0 - d / 0.32, 0.0);
    return falloff * falloff * falloff * (mag * mag * mag);
}
