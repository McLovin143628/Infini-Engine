// The shadow hooks `lights.wgsl` calls, for a pass that binds NO environment
// group (wave PAR0: the voxel and water passes). Neither has a cascade, a page
// atlas or a cloud map to read, so every hook is the identity — and the voxel
// pass passes flags that never call them anyway.

fn light_sun_shadow(p: vec3<f32>, n: vec3<f32>) -> f32 {
    return 1.0;
}

fn light_cloud_shadow(p: vec3<f32>) -> f32 {
    return 1.0;
}

fn light_local_shadow(p: vec3<f32>, n: vec3<f32>, slot: f32) -> f32 {
    return 1.0;
}

// **Multi-scatter energy compensation, and the one copy in the tree** (wave
// VIS1a; moved here from `voxel.wgsl` by wave PAR0). Every env-bound lit shader
// gets `ggx_energy_compensation` from `env_lighting.wgsl`; the passes composed
// with THIS shim (voxel, water) bind no environment group, so the shared
// `shade_light` reaches this verbatim copy instead. Pinned character for
// character against the env one by
// `the_bare_copy_of_the_energy_fit_is_character_for_character_the_env_one`.
fn gi_env_brdf_ab(rough: f32, nov: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
    return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

fn ggx_energy_compensation(f0: vec3<f32>, rough: f32, nov: f32) -> vec3<f32> {
    let ab = gi_env_brdf_ab(rough, nov);
    let e = max(ab.x + ab.y, 1e-3);
    return vec3<f32>(1.0) + f0 * (1.0 / e - 1.0);
}
