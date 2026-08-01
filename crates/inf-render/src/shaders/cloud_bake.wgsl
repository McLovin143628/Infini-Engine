// Cloud noise bake (P17.3): the two 3D volumes, written once per seed.
//
// `cloud_noise.wgsl` is composed in ahead of this (see
// `passes::ShaderKind::CloudBake`); the atmosphere uniform rides at
// @group(0) @binding(0) so the bake can read the seed out of the cloud block.
//
// Both volumes are `rgba8unorm` — 8 bits is plenty for a field that is then
// remapped and thresholded, and it is the format whose quantization the CPU
// parity reference mirrors exactly (`clouds::quant`: `v * 255 + 0.5`, which is
// what a normalized `textureStore` performs).

@group(0) @binding(1) var cloud_shape_out: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(2) var cloud_detail_out: texture_storage_3d<rgba8unorm, write>;

// Texel CENTRES, so the CPU reference and the shader agree on *where* they are.
// Sampling corners instead would make the volume non-tileable at the seam, which
// is exactly the bug the `noise_tiles_seamlessly` test exists to catch.
fn cloud_texel_centre(id: vec3<u32>, res: u32) -> vec3<f32> {
    let inv = 1.0 / f32(max(res, 1u));
    return (vec3<f32>(id) + vec3<f32>(0.5)) * inv;
}

@compute @workgroup_size(4, 4, 4)
fn cs_cloud_shape(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(cloud_shape_out);
    if (any(id >= dims)) {
        return;
    }
    let p = cloud_texel_centre(id, dims.x);
    let v = cloud_shape_value(p, u32(atmos.cloud_layer.w));
    textureStore(cloud_shape_out, vec3<i32>(id), v);
}

@compute @workgroup_size(4, 4, 4)
fn cs_cloud_detail(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(cloud_detail_out);
    if (any(id >= dims)) {
        return;
    }
    let p = cloud_texel_centre(id, dims.x);
    let v = cloud_detail_value(p, u32(atmos.cloud_layer.w));
    textureStore(cloud_detail_out, vec3<i32>(id), v);
}
