// Cloud shadows on world geometry (P17.3) — the receiver half.
//
// Composed into every lit scene shader by `passes::lit_scene_shader` (after
// `env_lighting.wgsl` and the atmosphere library), with `GROUP_ENV` substituted
// for the pipeline's env bind-group index. It adds exactly ONE binding to
// `EnvBinding`: the shadow map itself. The sampler is the atmosphere LUT sampler
// already bound at binding 9 — clamp-to-edge, linear — which is what this wants
// anyway, and the parameter block rides inside `AtmosphereData`.
//
// This is deliberately NOT a second cascaded shadow map. A cloud four kilometres
// up casts a shadow whose penumbra is hundreds of metres wide; a crisp one would
// be wrong, and a 512² map over 20 km (39 m per texel, bilinearly filtered) is
// both the right *look* and two orders of magnitude cheaper than making the CSM
// aware of a participating medium.
//
// OFF PATH — every call site is inside `if (atmos.clouds.x > 0.5 && ...)`, so a
// scene without clouds runs the identical instruction stream it ran in P17.2 and
// its goldens stay byte-identical. The function still early-outs on its own, for
// the benefit of any future caller that forgets.

@group(GROUP_ENV) @binding(11) var cloud_shadow_map: texture_2d<f32>;

// Fraction of the sun that survives the cloud layer above `world_local` (a
// RENDER-LOCAL position, as every lit shader has it). 1.0 = fully lit.
fn cloud_shadow_factor(world_local: vec3<f32>) -> f32 {
    let strength = atmos.cloud_shadow.x;
    if (atmos.clouds.x < 0.5 || strength <= 0.0) {
        return 1.0;
    }
    let sun = normalize(atmos.sun_dir.xyz);
    if (sun.y <= 1e-3) {
        // The sun is at or below the horizon: whatever it is contributing is
        // already reddened into nothing, and the projection below would divide by
        // ~zero and sample the far side of the world.
        return 1.0;
    }
    // Render-local → world (see `cloud.wgsl::cloud_to_world` for the derivation).
    let world = vec3<f32>(
        world_local.x - view.grid_axis_viewport.x,
        world_local.y - view.mode_axis.y,
        world_local.z - view.grid_axis_viewport.y,
    );
    // Project up the sun ray to the slab's mid-altitude — the depth at which a
    // layer's shadow is best approximated by a single plane.
    let mid = (atmos.cloud_layer.x + atmos.cloud_layer.y) * 0.5;
    let t = (mid - world.y) / sun.y;
    if (t <= 0.0) {
        return 1.0; // the receiver is above the layer; nothing shades it
    }
    let hit = world.xz + sun.xz * t;

    let extent = max(atmos.cloud_shadow.y, 1.0);
    let uv = (hit - atmos.cloud_shadow.zw) / extent + vec2<f32>(0.5);
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        // Outside the map. Explicitly fully lit rather than whatever
        // clamp-to-edge would hand back — a shadow that smeared to the horizon
        // would be a far worse artefact than one that stops.
        return 1.0;
    }
    let t_cloud = textureSampleLevel(cloud_shadow_map, atmos_lut_smp, uv, 0.0).r;
    return mix(1.0, clamp(t_cloud, 0.0, 1.0), clamp(strength, 0.0, 1.0));
}
