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
