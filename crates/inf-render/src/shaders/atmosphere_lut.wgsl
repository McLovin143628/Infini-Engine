// Atmosphere LUT sampling (P17.2) — the half of the atmosphere library that
// needs both baked LUTs bound. Composed after `atmosphere.wgsl` by
// `passes::atmosphere_lut_source(group, t_bind, s_bind, smp_bind)`; only the
// passes that actually bind both textures include it (the sky pass and, through
// `lit_scene_shader`, every lit pass). The two LUT bake compute shaders do not.

@group(ATMOS_GROUP) @binding(ATMOS_TBIND) var atmos_transmittance_lut: texture_2d<f32>;
@group(ATMOS_GROUP) @binding(ATMOS_SBIND) var atmos_skyview_lut: texture_2d<f32>;
@group(ATMOS_GROUP) @binding(ATMOS_SMPBIND) var atmos_lut_smp: sampler;

// Transmittance from radius `r` (km) at zenith cosine `mu` to space.
fn atmos_sample_transmittance(r: f32, mu: f32) -> vec3<f32> {
    let uv = atmos_transmittance_params_to_uv(r, mu);
    return textureSampleLevel(atmos_transmittance_lut, atmos_lut_smp, uv, 0.0).rgb;
}

// Sky radiance in world direction `dir` for a viewer at radius `r` (km).
// Already includes the sky-intensity exposure (baked in), so callers add only
// their own artistic terms.
fn atmos_sample_skyview(r: f32, dir: vec3<f32>) -> vec3<f32> {
    let uv = atmos_skyview_uv(r, dir, atmos.sun_dir.xyz);
    return textureSampleLevel(atmos_skyview_lut, atmos_lut_smp, uv, 0.0).rgb;
}

// ── aerial perspective + height fog on lit geometry ────────────────────────
//
// v1 model, deliberately LUT-cheap: rather than a 3D froxel volume, the segment
// from the eye to the surface is treated as homogeneous at the *camera's* local
// extinction, which is exact for the ground-level, few-kilometre view distances a
// game renders and errs only for a camera high in a steep density gradient. The
// in-scattered colour comes from the sky-view LUT in the same direction, which is
// what makes distant geometry go blue at noon and orange at sunset for free. A
// full aerial-perspective LUT (Hillaire section 5.4) is the documented follow-up.
//
// Height fog composites on top in the same pass: its extinction multiplies the
// aerial transmittance and its in-scatter uses the same sky colour, tinted.
//
// `world_pos` is RENDER-LOCAL (as every lit shader has it); `view.mode_axis.y` is
// -origin.y, so the world altitude is `world_pos.y - view.mode_axis.y`. That is
// the only place the floating origin enters the atmosphere.
fn atmos_apply(color: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let to_eye = view.eye.xyz - world_pos;
    let dist_m = length(to_eye);
    if (dist_m <= 0.0) {
        return color;
    }
    let dir = -to_eye / dist_m; // eye -> surface
    let r = atmos.planet.z;

    // The in-scattered colour is sampled from the sky-view LUT, but with the view
    // elevation CLAMPED AT THE HORIZON. That is not a fudge, it is the difference
    // between the two things the LUT could mean: below the horizon its texels
    // describe "the planet's surface seen through air", whereas what a surface at
    // 800 m needs is "the air column between here and there" — which, for any
    // horizontal or downhill ray, is the horizon's air column. Sampling the
    // unclamped direction makes distant downhill geometry get *darker* with
    // distance, the exact opposite of aerial perspective. (Getting this right for
    // every elevation is what Hillaire's 3D aerial-perspective froxel LUT is for;
    // that is the documented follow-up.)
    let scatter_dir = normalize(vec3<f32>(dir.x, max(dir.y, 0.0), dir.z) + vec3<f32>(0.0, 1e-4, 0.0));
    let sky = atmos_sample_skyview(r, scatter_dir);

    // ── aerial perspective ──
    let dist_km = dist_m * 1e-3;
    let sigma = atmos_extinction(max(r - atmos.planet.x, 0.0));
    let strength = atmos.params.z;
    let t_air = exp(-sigma * dist_km * strength);
    var out = color * t_air + sky * (vec3<f32>(1.0) - t_air);

    // ── height fog (SI metres) ──
    if (atmos.fog.x > 0.0) {
        let eye_y = view.eye.xyz.y - view.mode_axis.y;
        let surf_y = world_pos.y - view.mode_axis.y;
        let od = atmos_fog_optical_depth(eye_y, surf_y, dist_m);
        let t_fog = exp(-od);
        out = out * t_fog + sky * atmos.fog_color.rgb * (1.0 - t_fog);
    }
    return out;
}
