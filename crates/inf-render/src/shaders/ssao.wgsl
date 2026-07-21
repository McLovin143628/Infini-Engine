// SSAO (P13.3a baseline): half-res hemisphere AO reconstructed from the
// single-sample scene-depth prepass. `common_view.wgsl` (group 0 = view) is
// prepended, giving `unproject()` + the view/inv-view-proj matrices. World-space
// position + a depth-derivative normal are reconstructed per pixel; a rotated
// hemisphere kernel is projected back through the depth buffer to estimate
// occlusion, multiplied into the ambient term by the lit passes. `fs_blur` box-
// blurs the raw AO to hide the 4×4 rotation-noise pattern.

const MAX_KERNEL: u32 = 32u;

struct SsaoParams {
    // xyz = tangent-space hemisphere sample, w unused.
    kernel: array<vec4<f32>, MAX_KERNEL>,
    // x = kernel count, y = radius (m), z = intensity, w = bias (m).
    cfg: vec4<f32>,
    // xy = depth texture size (px), zw = ao target size (px).
    dims: vec4<f32>,
};
@group(1) @binding(0) var depth_tex: texture_depth_2d;
@group(1) @binding(1) var<uniform> ssao: SsaoParams;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let ndc = fullscreen_ndc(i);
    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return out;
}

fn load_depth(uv: vec2<f32>) -> f32 {
    let d = ssao.dims.xy;
    let c = clamp(vec2<i32>(uv * d), vec2<i32>(0), vec2<i32>(d) - vec2<i32>(1));
    return textureLoad(depth_tex, c, 0);
}

fn world_at(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return unproject(ndc, depth);
}

// Interleaved-gradient noise → a per-pixel rotation angle for the kernel.
fn ign(pix: vec2<f32>) -> f32 {
    let m = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(m.z * fract(dot(pix, m.xy)));
}

@fragment
fn fs_ssao(in: VsOut) -> @location(0) vec4<f32> {
    let depth = load_depth(in.uv);
    // Reverse-Z: depth 0 == far/background (sky) → fully unoccluded.
    if (depth <= 0.0) {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    let p = world_at(in.uv, depth);

    // Normal from the derivative of the reconstructed world position.
    var n = normalize(cross(dpdx(p), dpdy(p)));
    let to_eye = view.eye.xyz - p;
    if (dot(n, to_eye) < 0.0) {
        n = -n;
    }

    // Random tangent basis (rotate an arbitrary tangent about n by the IGN angle).
    let ang = ign(in.pos.xy) * 6.2831853;
    let rnd = vec3<f32>(cos(ang), sin(ang), 0.0);
    var t = normalize(rnd - n * dot(rnd, n));
    if (!(dot(t, t) > 0.0)) {
        t = normalize(cross(n, vec3<f32>(0.0, 1.0, 0.0)) + vec3<f32>(1e-4, 0.0, 0.0));
    }
    let b = cross(n, t);
    let tbn = mat3x3<f32>(t, b, n);

    let radius = ssao.cfg.y;
    let bias = ssao.cfg.w;
    let count = u32(ssao.cfg.x);
    let d_center = length(to_eye);

    var occlusion = 0.0;
    for (var i = 0u; i < count && i < MAX_KERNEL; i = i + 1u) {
        let sample_w = p + (tbn * ssao.kernel[i].xyz) * radius;
        // Project the sample into the depth buffer.
        let clip = view.view_proj * vec4<f32>(sample_w, 1.0);
        if (clip.w <= 0.0) {
            continue;
        }
        let s_ndc = clip.xyz / clip.w;
        let s_uv = vec2<f32>(s_ndc.x * 0.5 + 0.5, 0.5 - s_ndc.y * 0.5);
        if (s_uv.x < 0.0 || s_uv.x > 1.0 || s_uv.y < 0.0 || s_uv.y > 1.0) {
            continue;
        }
        let sd = load_depth(s_uv);
        if (sd <= 0.0) {
            continue;
        }
        let scene_w = world_at(s_uv, sd);
        let d_sample = length(view.eye.xyz - sample_w);
        let d_scene = length(view.eye.xyz - scene_w);
        // Geometry in front of the sample point (closer to the eye) occludes it.
        if (d_scene < d_sample - bias) {
            // Range check: ignore occluders far outside the radius.
            let range = smoothstep(0.0, 1.0, radius / max(abs(d_center - d_scene), 1e-4));
            occlusion = occlusion + range;
        }
    }
    let ao = 1.0 - (occlusion / max(f32(count), 1.0)) * ssao.cfg.z;
    return vec4<f32>(clamp(ao, 0.0, 1.0), 0.0, 0.0, 1.0);
}

@group(1) @binding(2) var ao_src: texture_2d<f32>;
@group(1) @binding(3) var ao_smp: sampler;

// 4×4 box blur over the raw AO (matches the noise rotation tile).
@fragment
fn fs_blur(in: VsOut) -> @location(0) vec4<f32> {
    let t = 1.0 / ssao.dims.zw;
    var sum = 0.0;
    for (var y = -2; y <= 1; y = y + 1) {
        for (var x = -2; x <= 1; x = x + 1) {
            let uv = in.uv + vec2<f32>(f32(x), f32(y)) * t;
            sum = sum + textureSampleLevel(ao_src, ao_smp, uv, 0.0).r;
        }
    }
    return vec4<f32>(sum / 16.0, 0.0, 0.0, 1.0);
}
