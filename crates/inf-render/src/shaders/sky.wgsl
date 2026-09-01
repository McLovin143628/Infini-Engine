// Sky pass (P17.2). Fullscreen background, drawn first with depth writes off.
//
// Two paths, selected by the atmosphere master flag:
//
//  * `atmos.params.x < 0.5` — the schema-v11 three-colour gradient, reproduced
//    below in `gradient_sky` character for character. Every scene without a
//    `TimeOfDay` authority takes this branch, which is why every pre-P17.2 golden
//    stays byte-identical (the P13.3 off-path discipline: the *taken* arithmetic
//    is unchanged, not merely the result).
//  * otherwise — the physical sky: the sky-view LUT, the sun and moon discs, and
//    the starfield, with the gradient available on top as an artistic tint.
//
// `atmosphere.wgsl` is composed in at @group(1) @binding(1) and
// `atmosphere_lut.wgsl` at @group(1) bindings 2/3/4 (see `passes::ShaderKind::Sky`).

struct Sky {
    zenith: vec4<f32>,
    horizon: vec4<f32>,
    ground: vec4<f32>,
};
@group(1) @binding(0) var<uniform> sky: Sky;

// Radiance multiplier on the sun disc, relative to the sky exposure. The disc is
// deliberately far above 1.0 so it clips to white through the ACES tonemap and
// drives bloom when bloom is on — a physical sun is ~10^5 times the sky.
const SUN_DISC_GAIN: f32 = 4.0;
// Limb-darkening exponent: the solar disc is measurably dimmer at its rim.
const SUN_LIMB: f32 = 0.35;
// Moon disc gain. Much gentler — the moon is a lit rock, not a light source.
const MOON_DISC_GAIN: f32 = 1.5;
// Star gain, relative to the sky exposure.
const STAR_GAIN: f32 = 0.06;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var out: VsOut;
    let p = fullscreen_ndc(i);
    // Depth 0 = infinity under reverse-Z: the sky is behind everything.
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

// The schema-v11 authored gradient. Kept verbatim: it is the byte-stable path for
// every level without a time of day, AND the artistic tint layer for levels that
// have one.
fn gradient_sky(dir: vec3<f32>) -> vec3<f32> {
    let t = dir.y;

    var col: vec3<f32>;
    if (t >= 0.0) {
        // Slow start near the horizon, deep zenith overhead.
        col = mix(sky.horizon.rgb, sky.zenith.rgb, pow(clamp(t, 0.0, 1.0), 0.55));
    } else {
        // Below the horizon: fade quickly into the ground haze.
        col = mix(sky.horizon.rgb, sky.ground.rgb, clamp(-t * 3.5, 0.0, 1.0));
    }

    // Subtle warm glow around the sun direction.
    let sun = pow(max(dot(dir, view.sun_dir.xyz), 0.0), 48.0);
    col += vec3<f32>(0.30, 0.24, 0.16) * sun * 0.35;

    return col;
}

// The sun disc: a hard-edged circle of angular radius `acos(atmos.sun_dir.w)`,
// softened at the rim by limb darkening and extinguished by the atmosphere along
// the view ray — which is what makes it redden and dim as it sets, and vanish
// entirely once the ray to it passes through the planet.
fn sun_disc(dir: vec3<f32>, r: f32) -> vec3<f32> {
    let sun = normalize(atmos.sun_dir.xyz);
    let cos_a = dot(dir, sun);
    let edge = atmos.sun_dir.w;
    if (cos_a <= edge) {
        return vec3<f32>(0.0);
    }
    // 1 at the centre, 0 at the rim.
    let centre = clamp((cos_a - edge) / max(1.0 - edge, 1e-9), 0.0, 1.0);
    let limb = pow(centre, SUN_LIMB);
    // The body door, not the raw table read: below the horizon the raw read
    // clamps onto the tangent texel and the disc survives its own setting as a
    // red blob hanging under the ground — the exact opposite of the comment
    // above it (wave GTA1).
    let t = atmos_sample_body_transmittance(r, dir.y, edge);
    return atmos.sun_color.rgb * t * (atmos.params.y * SUN_DISC_GAIN * limb);
}

// The moon disc with a v1 phase terminator.
//
// The terminator of a sphere lit from the side projects to an ellipse, so on the
// flat disc the boundary sits at `x = k * sqrt(1 - y^2)` with
// `k = cos(2*pi*phase)`: k = +1 at new moon (nothing lit), k = -1 at full
// (everything lit), k = 0 at the quarters. That is the *shape* right; what this
// v1 does not model is the moon's orbital tilt, so the terminator's roll about
// the view axis is fixed to the local horizontal rather than derived from the
// ecliptic. Documented rather than hidden — a real libration/tilt model is
// follow-up work, and at 0.5 degrees across nobody has ever noticed.
fn moon_disc(dir: vec3<f32>, r: f32) -> vec3<f32> {
    let moon = normalize(atmos.moon_dir.xyz);
    let cos_a = dot(dir, moon);
    let edge = atmos.moon_dir.w;
    if (cos_a <= edge) {
        return vec3<f32>(0.0);
    }
    // Disc-local frame: `mx` is horizontal ("east" on the disc), `my` completes it.
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let mx = normalize(cross(up, moon) + vec3<f32>(1e-6, 0.0, 0.0));
    let my = cross(moon, mx);
    let offset = dir - moon * cos_a;
    let radius = sqrt(max(1.0 - edge * edge, 1e-12));
    let sx = dot(offset, mx) / radius;
    let sy = clamp(dot(offset, my) / radius, -1.0, 1.0);

    let k = cos(6.28318530718 * atmos.moon_color.w);
    let boundary = k * sqrt(max(1.0 - sy * sy, 0.0));
    // Soft terminator; the width is a fraction of the disc, not of the screen.
    let lit = smoothstep(boundary - 0.12, boundary + 0.12, sx);
    // Gentle rim darkening so it reads as a sphere rather than a sticker.
    let rim = sqrt(max(1.0 - (sx * sx + sy * sy), 0.0));
    // Same body door as the sun disc: a moon below the horizon is behind the
    // planet, and the raw table read would keep it visible (wave GTA1).
    let t = atmos_sample_body_transmittance(r, dir.y, edge);
    return atmos.moon_color.rgb * t
        * (atmos.params.y * MOON_DISC_GAIN * lit * (0.35 + 0.65 * rim));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let dir = view_ray(in.ndc);

    // ── the byte-stable pre-P17.2 path ──
    if (atmos.params.x < 0.5) {
        return vec4<f32>(gradient_sky(dir), 1.0);
    }

    // ── the physical sky ──
    let r = atmos.planet.z;
    var col = atmos_sample_skyview(r, dir);

    // The authored gradient as an artistic tint over the physical result. 0 by
    // default: the physics stands on its own and the colours are an override.
    if (atmos.params.w > 0.0) {
        col = mix(col, gradient_sky(dir), clamp(atmos.params.w, 0.0, 1.0));
    }

    // Stars, behind everything, above the horizon only, fading as the sun rises.
    let night = atmos.stars.z;
    if (night > 0.0 && atmos.stars.x > 0.0) {
        let above = smoothstep(-0.02, 0.03, dir.y);
        let s = atmos_starfield(dir, atmos.stars.y);
        col += vec3<f32>(s * night * above * atmos.stars.x * atmos.params.y * STAR_GAIN);
    }

    col += sun_disc(dir, r);
    col += moon_disc(dir, r);

    return vec4<f32>(col, 1.0);
}
