// Cloud-shadow-map bake (P17.3): the low-resolution transmittance map that lets
// the cloud layer darken the *world*, not just occlude the sky.
//
// Composed as `cloud_noise.wgsl` + `atmosphere.wgsl` @group(0) @binding(0) +
// `cloud_field.wgsl` @group(0) bindings 1/2/3, plus the storage output below
// (see `passes::ShaderKind::CloudShadowBake`).
//
// PARAMETERIZATION — the map is a plain world-XZ square of side
// `atmos.cloud_shadow.y` metres centred on `atmos.cloud_shadow.zw` (world metres,
// quantized to a whole texel on the CPU so a moving camera slides the map without
// making it shimmer). Each texel stores the Beer-Lambert transmittance of the
// cloud slab along the SUN direction, starting from that XZ at the slab's floor.
//
// Why a world-XZ plane rather than a sun-aligned one: the receiver lookup
// (`cloud_shadow.wgsl`) projects a world position UP along the sun to the slab's
// mid-altitude and reads the texel there, so the projection happens at lookup
// time and the map itself stays an ordinary axis-aligned grid. That costs a
// little resolution when the sun is low — where the shadows are so stretched that
// nobody can tell — and buys a bake that never needs a light-space matrix, never
// has to be re-fitted to a frustum, and cannot shear.
//
// The output is `rgba16float`. `r16float` is not a core WebGPU storage format and
// `r32float` is not FILTERABLE without an optional feature — and a
// nearest-sampled cloud shadow is a grid of 39 m squares, not a penumbra. So the
// map spends four channels to get one that is both storable and filterable
// everywhere; only the red channel carries the transmittance.

@group(0) @binding(4) var cloud_shadow_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_cloud_shadow(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(cloud_shadow_out);
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    let res = f32(dims.x);
    let extent = atmos.cloud_shadow.y;
    // Texel centre in world metres.
    let uv = (vec2<f32>(f32(id.x), f32(id.y)) + vec2<f32>(0.5)) / res;
    let world_xz = atmos.cloud_shadow.zw + (uv - vec2<f32>(0.5)) * extent;

    let sun = normalize(atmos.sun_dir.xyz);
    let p = vec3<f32>(world_xz.x, atmos.cloud_layer.x, world_xz.y);
    let t = cloud_sun_transmittance(p, sun, u32(atmos.cloud_march.z));
    textureStore(cloud_shadow_out, vec2<i32>(id.xy), vec4<f32>(t, 0.0, 0.0, 1.0));
}
