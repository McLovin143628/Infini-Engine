// Dynamic-GI scene voxelization (P13.3b): one compute thread per voxel of the
// camera-centred GI_DIM³ volume writes a packed RGBA8 (albedo + binary occupancy)
// into a storage buffer. v1 voxelizes rigid mesh instances **analytically**: a
// MeshInstance is a unit cube (±0.5) transformed by its TRS, so a voxel is solid
// when its render-local centre maps inside [-0.5, 0.5]³ in instance-local space
// (via the precomputed inverse model matrix). Terrain/skinned voxelization is a
// documented follow-up. Mirrors `crate::gi` (voxel indexing / volume layout).

struct GiData {
    vol_min: vec4<f32>,    // xyz render-local min corner, w = voxel_size
    probe_min: vec4<f32>,  // xyz render-local probe grid min, w = extent
    dims: vec4<f32>,       // x = gi_dim, yzw = probe dims
    params: vec4<f32>,     // x = enabled, y = intensity, z = rays, w = instance_count
    sun_dir: vec4<f32>,    // xyz toward sun
    sun_color: vec4<f32>,  // rgb = sun radiance
    sky_zenith: vec4<f32>, // rgb
    sky_horizon: vec4<f32>,// rgb
};
@group(0) @binding(0) var<uniform> gi: GiData;

struct GiInstance {
    inv_model: mat4x4<f32>, // render-local world → instance-local
    albedo: vec4<f32>,      // rgb, w unused
};
@group(0) @binding(1) var<storage, read> instances: array<GiInstance>;
@group(0) @binding(2) var<storage, read_write> voxels: array<u32>;

fn pack_rgba8(c: vec4<f32>) -> u32 {
    let q = vec4<u32>(clamp(c, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0 + 0.5);
    return q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
}

@compute @workgroup_size(64)
fn cs_voxelize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = u32(gi.dims.x);
    let total = dim * dim * dim;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let x = idx % dim;
    let y = (idx / dim) % dim;
    let z = idx / (dim * dim);
    let vsize = gi.vol_min.w;
    let center = gi.vol_min.xyz + (vec3<f32>(f32(x), f32(y), f32(z)) + 0.5) * vsize;

    let count = u32(gi.params.w);
    var out = vec4<f32>(0.0);
    for (var i = 0u; i < count; i = i + 1u) {
        let local = (instances[i].inv_model * vec4<f32>(center, 1.0)).xyz;
        if (all(abs(local) <= vec3<f32>(0.5))) {
            out = vec4<f32>(instances[i].albedo.rgb, 1.0);
            break;
        }
    }
    voxels[idx] = pack_rgba8(out);
}
