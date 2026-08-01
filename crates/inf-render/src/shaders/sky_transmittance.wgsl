// Transmittance LUT bake (P17.2). One texel per (altitude, sun/view zenith
// angle) pair; the value is how much of each wavelength survives a ray from that
// altitude at that angle out to space.
//
// Composed with `atmosphere.wgsl` at @group(0) @binding(0) by
// `passes::shader_source("atmos_transmittance")`.
//
// This LUT is a function of the **medium alone** — not of the sun, not of the
// camera — so it is baked once and re-baked only when the atmosphere parameters
// change. At 256x64 that is 16 k threads of a ~40-step march: it happens roughly
// never and costs nothing when it does.

@group(0) @binding(1) var atmos_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_transmittance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(atmos_out);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5))
        / vec2<f32>(f32(dims.x), f32(dims.y));
    let rm = atmos_transmittance_uv_to_params(uv);
    let t = atmos_transmittance_integral(rm.x, rm.y, u32(max(atmos.ozone.w, 2.0)));
    textureStore(atmos_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(t, 1.0));
}
