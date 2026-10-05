// The shadow hooks `lights.wgsl` calls, for a pass that binds the shared
// environment group (wave PAR0). Each is the exact guard + factor the pass's
// private loop applied before PAR0, returned as a multiplier: `x * 1.0 == x`
// in IEEE arithmetic, so a disabled term is the same value it always was.

// The first directional light: the virtual page atlas when one is bound, the
// cascade otherwise (`shadow_factor` decides), and 1.0 with both paths off.
fn light_sun_shadow(p: vec3<f32>, n: vec3<f32>) -> f32 {
    if (sun_shadowing_enabled()) {
        return shadow_factor(p, n);
    }
    return 1.0;
}

// P17.3's large-scale cloud shadowing of every directional light.
fn light_cloud_shadow(p: vec3<f32>) -> f32 {
    if (atmos.clouds.x > 0.5 && atmos.cloud_shadow.x > 0.0) {
        return cloud_shadow_factor(p);
    }
    return 1.0;
}

// P27.4's point/spot virtual shadow: a spot through its quadtree, a point
// through the cube face its own direction selects; exactly 1.0 for a light
// whose `params.w` is 0 (no page tree).
fn light_local_shadow(p: vec3<f32>, n: vec3<f32>, slot: f32) -> f32 {
    return vsm_light_shadow(p, n, slot);
}
