// Sky gradient: fullscreen background, drawn first with depth writes off.

struct Sky {
    zenith: vec4<f32>,
    horizon: vec4<f32>,
    ground: vec4<f32>,
};
@group(1) @binding(0) var<uniform> sky: Sky;

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

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let dir = view_ray(in.ndc);
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

    return vec4<f32>(col, 1.0);
}
