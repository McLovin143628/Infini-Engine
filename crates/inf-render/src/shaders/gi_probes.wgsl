// Dynamic-GI probe march (P13.3b): one compute thread per probe (16×8×16 grid)
// marches `rays` fixed golden-spiral directions through the voxel volume, gathers
// single-bounce radiance, and projects it to L1 spherical harmonics (4 coeffs ×
// RGB) written to a storage buffer the lit passes sample. Deterministic (fixed
// directions, no temporal jitter). Mirrors `crate::gi`.
//
//   hit  → radiance = albedo × sun_radiance × sun_visibility(hit)   (single bounce)
//   miss → radiance = sky gradient (horizon→zenith by ray.y)

struct GiData {
    vol_min: vec4<f32>,
    probe_min: vec4<f32>,
    dims: vec4<f32>,
    params: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
};
@group(0) @binding(0) var<uniform> gi: GiData;
@group(0) @binding(1) var<storage, read> voxels: array<u32>;
@group(0) @binding(2) var<storage, read_write> sh: array<vec4<f32>>;

const PI: f32 = 3.14159265359;

fn unpack_rgba8(v: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(v & 0xffu),
        f32((v >> 8u) & 0xffu),
        f32((v >> 16u) & 0xffu),
        f32((v >> 24u) & 0xffu),
    ) / 255.0;
}

fn voxel_at(c: vec3<i32>) -> vec4<f32> {
    let dim = i32(gi.dims.x);
    if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(dim))) {
        return vec4<f32>(0.0); // outside → empty
    }
    let d = u32(dim);
    let uc = vec3<u32>(c);
    let idx = (uc.z * d + uc.y) * d + uc.x;
    return unpack_rgba8(voxels[idx]);
}

// Sample occupancy+albedo at render-local point `p`.
fn sample_point(p: vec3<f32>) -> vec4<f32> {
    let coord = (p - gi.vol_min.xyz) / gi.vol_min.w;
    return voxel_at(vec3<i32>(floor(coord)));
}

fn spiral_dir(i: u32, n: u32) -> vec3<f32> {
    let fn_ = f32(max(n, 1u));
    let fi = f32(i);
    let phi = PI * (3.0 - sqrt(5.0));
    let y = 1.0 - 2.0 * (fi + 0.5) / fn_;
    let r = sqrt(max(1.0 - y * y, 0.0));
    let theta = phi * fi;
    return vec3<f32>(cos(theta) * r, y, sin(theta) * r);
}

fn sh_basis(d: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(0.282095, 0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x);
}

// March toward the sun from `p`; 0 if an occluder is hit before leaving the volume.
fn sun_visibility(p: vec3<f32>) -> f32 {
    let vsize = gi.vol_min.w;
    let dir = normalize(gi.sun_dir.xyz);
    let dim = gi.dims.x;
    var pos = p + dir * vsize * 1.5; // step off the surface
    let steps = i32(dim);
    for (var s = 0; s < steps; s = s + 1) {
        if (sample_point(pos).w > 0.5) {
            return 0.0;
        }
        pos = pos + dir * vsize;
    }
    return 1.0;
}

@compute @workgroup_size(64)
fn cs_probes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = u32(gi.dims.y);
    let py = u32(gi.dims.z);
    let pz = u32(gi.dims.w);
    let total = px * py * pz;
    let pi = gid.x;
    if (pi >= total) {
        return;
    }
    let ix = pi % px;
    let iy = (pi / px) % py;
    let iz = pi / (px * py);
    let extent = gi.probe_min.w;
    let frac = vec3<f32>(
        select(f32(ix) / f32(px - 1u), 0.5, px <= 1u),
        select(f32(iy) / f32(py - 1u), 0.5, py <= 1u),
        select(f32(iz) / f32(pz - 1u), 0.5, pz <= 1u),
    );
    let origin = gi.probe_min.xyz + frac * extent;

    let vsize = gi.vol_min.w;
    let dim = gi.dims.x;
    let rays = u32(gi.params.z);

    var c0 = vec3<f32>(0.0);
    var c1 = vec3<f32>(0.0);
    var c2 = vec3<f32>(0.0);
    var c3 = vec3<f32>(0.0);

    for (var r = 0u; r < rays; r = r + 1u) {
        let dir = spiral_dir(r, rays);
        // March the ray through the volume.
        var pos = origin + dir * vsize * 0.5;
        var radiance = vec3<f32>(0.0);
        var hit = false;
        let steps = i32(dim) * 2;
        for (var s = 0; s < steps; s = s + 1) {
            let v = sample_point(pos);
            if (v.w > 0.5) {
                let vis = sun_visibility(pos);
                radiance = v.rgb * gi.sun_color.rgb * vis;
                hit = true;
                break;
            }
            pos = pos + dir * vsize;
        }
        if (!hit) {
            let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
            radiance = mix(gi.sky_horizon.rgb, gi.sky_zenith.rgb, t);
        }
        let b = sh_basis(dir);
        c0 = c0 + radiance * b.x;
        c1 = c1 + radiance * b.y;
        c2 = c2 + radiance * b.z;
        c3 = c3 + radiance * b.w;
    }

    let norm = 4.0 * PI / f32(max(rays, 1u));
    let base = pi * 4u;
    sh[base + 0u] = vec4<f32>(c0 * norm, 0.0);
    sh[base + 1u] = vec4<f32>(c1 * norm, 0.0);
    sh[base + 2u] = vec4<f32>(c2 * norm, 0.0);
    sh[base + 3u] = vec4<f32>(c3 * norm, 0.0);
}
