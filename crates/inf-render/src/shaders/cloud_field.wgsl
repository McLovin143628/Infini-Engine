// The cloud density field (P17.3) — the sampled half of the cloud library.
//
// Composed after `cloud_noise.wgsl` by `passes::cloud_field_source(group, shape,
// detail, smp)`, and included ONLY by the passes that bind both baked volumes as
// sampled textures: the raymarch (`cloud.wgsl`) and the cloud-shadow bake
// (`cloud_shadow_bake.wgsl`). The noise bake itself binds them as *storage*
// textures and therefore takes `cloud_noise.wgsl` alone.
//
// Requires `atmosphere.wgsl` to have been composed in first: the cloud parameter
// block rides in `AtmosphereData` rather than a uniform of its own, so the whole
// feature costs the lit passes exactly ONE new binding (the shadow map) instead
// of two.
//
// UNITS — SI metres, m^-1. `p` is always a WORLD position, never render-local:
// the field must be locked to the world, or clouds would slide sideways every
// time the floating origin rebases.

@group(CLOUD_GROUP) @binding(CLOUD_SHAPE_BIND) var cloud_shape_tex: texture_3d<f32>;
@group(CLOUD_GROUP) @binding(CLOUD_DETAIL_BIND) var cloud_detail_tex: texture_3d<f32>;
// Repeat + linear: the volumes are tileable by construction, which is what lets a
// 20 km march wrap an 8 km texture with no seam.
@group(CLOUD_GROUP) @binding(CLOUD_SMP_BIND) var cloud_smp: sampler;

// Slab thickness in metres, always > 0 when clouds are active.
fn cloud_thickness() -> f32 {
    return max(atmos.cloud_layer.y - atmos.cloud_layer.x, 0.0);
}

// Relative height of a world altitude within the slab. Outside [0, 1] ⇒ no cloud.
fn cloud_height_frac(world_y: f32) -> f32 {
    return (world_y - atmos.cloud_layer.x) / max(cloud_thickness(), 1e-3);
}

// Cloud extinction (m^-1) at WORLD position `p` metres. 0 outside the slab.
// CPU mirror: `clouds::CloudVolumes::density`.
fn cloud_density(p: vec3<f32>) -> f32 {
    let thickness = cloud_thickness();
    if (thickness <= 0.0) {
        return 0.0;
    }
    let h = (p.y - atmos.cloud_layer.x) / thickness;
    if (h < 0.0 || h > 1.0) {
        return 0.0;
    }
    let seed = u32(atmos.cloud_layer.w);

    // Weather is sampled at the UNDRIFTED position: coverage is the geography of
    // the weather system, and a system that slid along with its own clouds would
    // never let a gap pass overhead.
    let w = cloud_weather(p.x, p.z, seed, atmos.clouds.y, atmos.clouds.z);
    if (w.x <= 0.0) {
        return 0.0;
    }
    let grad = cloud_height_gradient(h, w.y);
    if (grad <= 0.0) {
        return 0.0;
    }

    let off = atmos.cloud_wind.xy;
    let sp = vec3<f32>(
        (p.x + off.x) / CLOUD_SHAPE_TILE_M,
        p.y / max(thickness, 1.0),
        (p.z + off.y) / CLOUD_SHAPE_TILE_M,
    );
    let s = textureSampleLevel(cloud_shape_tex, cloud_smp, sp, 0.0);
    let low = s.g * 0.625 + s.b * 0.25 + s.a * 0.125;
    let shape = cloud_remap(s.r, low - 1.0, 1.0) * grad;

    // Coverage as a dissolve: raising the floor eats the field from its edges
    // inward, which is how a real cloud thins out. A multiply would just fade the
    // whole sky grey.
    var d = cloud_remap(shape, 1.0 - w.x, 1.0) * w.x;
    if (d <= 0.0) {
        return 0.0;
    }

    // Erosion. Skipped entirely on the Low tier (`cloud_march.w == 0`) — the
    // documented cheap path, which loses the wisps but stays a pure function of
    // the same inputs.
    let detail_strength = clamp(atmos.clouds.w, 0.0, 1.0);
    if (atmos.cloud_march.w > 0.5 && detail_strength > 0.0) {
        // Detail drifts at twice the shape's rate: relative motion inside a cloud
        // is what makes it look alive rather than like a rigid sculpture.
        let dp = vec3<f32>(
            (p.x + off.x * 2.0) / CLOUD_DETAIL_TILE_M,
            p.y / CLOUD_DETAIL_TILE_M,
            (p.z + off.y * 2.0) / CLOUD_DETAIL_TILE_M,
        );
        let e = textureSampleLevel(cloud_detail_tex, cloud_smp, dp, 0.0);
        let fbm = e.r * 0.625 + e.g * 0.25 + e.b * 0.125;
        // Wispy (inverted) at the base, billowy at the shoulders — the standard
        // trick that frays a cumulus downward while keeping its top solid.
        let wispy = fbm + (1.0 - fbm - fbm) * smoothstep(0.0, 0.35, h);
        d = cloud_remap(d, wispy * 0.6 * detail_strength, 1.0);
    }
    return max(d, 0.0) * max(atmos.cloud_layer.z, 0.0);
}

// Beer-Lambert transmittance of the cloud layer from WORLD position `p` toward
// the sun, marched in `steps` steps. Returns 1 (fully lit) for a sun at or below
// the horizon, which is the documented early-out rather than a 70 km march.
// CPU mirror: `clouds::CloudVolumes::sun_transmittance`.
fn cloud_sun_transmittance(p: vec3<f32>, sun: vec3<f32>, steps: u32) -> f32 {
    if (sun.y <= 1e-3) {
        return 1.0;
    }
    let n = max(steps, 1u);
    let span = min(cloud_thickness() / sun.y, CLOUD_MAX_SHADOW_MARCH_M);
    let dt = span / f32(n);
    var depth = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let t = (f32(i) + 0.5) * dt;
        depth = depth + cloud_density(p + sun * t) * dt;
    }
    return exp(-depth);
}
