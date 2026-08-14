// **The ray-query sun-shadow experiment** (P28.5) — NEVER on the shipped path.
//
// One compute thread per pixel. Two rays: a primary ray into the TLAS to find
// the surface, then a shadow ray from that surface toward the sun. The verdict
// per pixel is one of three values, because "no surface here" and "a surface
// the sun can see" are different facts and one bit cannot say both:
//
//   0 = RT_MISS      the primary ray hit nothing in the TLAS
//   1 = RT_LIT       it hit a surface and the shadow ray escaped
//   2 = RT_SHADOWED  it hit a surface and the shadow ray was occluded
//
// **Primary visibility is traced rather than read from a depth buffer**, and
// that is deliberate: it makes the experiment self-contained (no G-buffer, no
// reprojection, no dependence on any shipped pass) and it makes the coverage
// bound explicit — the TLAS holds exactly the meshlet clusters the host put in
// it, so every pixel this pass has an opinion about is a pixel that geometry
// covers. A comparison against the shipped shadow path is only meaningful over
// those pixels and the gate restricts itself to them.
//
// Geometry is built with `AccelerationStructureGeometryFlags::OPAQUE`, so the
// driver commits candidate triangle intersections itself and the proceed loop
// is a formality rather than an any-hit program.

// The enable extension the type below needs. It has to precede every item, and
// it is what makes `naga::valid::Capabilities::all()` accept this module on a
// CI leg with no ray-tracing adapter anywhere near it.
enable wgpu_ray_query;

struct RtParams {
    // Camera basis, render-local (the floating origin is applied by the host).
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    fwd: vec4<f32>,
    // xyz = unit direction TOWARD the sun; w = the shadow ray's `tmin`, in
    // metres — the surface offset that keeps a surface from shadowing itself.
    sun: vec4<f32>,
    // x = tan(fov_y / 2), y = aspect (w/h), z = primary tmin, w = tmax.
    misc: vec4<f32>,
    // x = width, y = height.
    dims: vec4<u32>,
};

@group(0) @binding(0) var accel: acceleration_structure;
@group(0) @binding(1) var<uniform> params: RtParams;
@group(0) @binding(2) var<storage, read_write> verdict: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn cs_sun_shadow(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dims.x || gid.y >= params.dims.y) {
        return;
    }
    let idx = gid.y * params.dims.x + gid.x;

    // The pixel CENTRE, in a [-1, 1] frame with +y up — the same convention the
    // raster's viewport uses, so a comparison against a rendered frame lines up
    // texel for texel rather than half a pixel off.
    let px = (f32(gid.x) + 0.5) / f32(params.dims.x) * 2.0 - 1.0;
    let py = 1.0 - (f32(gid.y) + 0.5) / f32(params.dims.y) * 2.0;
    let half_h = params.misc.x;
    let dir = normalize(
        params.fwd.xyz
        + params.right.xyz * (px * half_h * params.misc.y)
        + params.up.xyz * (py * half_h)
    );

    var primary: ray_query;
    rayQueryInitialize(
        &primary,
        accel,
        RayDesc(0u, 0xFFu, params.misc.z, params.misc.w, params.eye.xyz, dir),
    );
    while (rayQueryProceed(&primary)) {}
    let hit = rayQueryGetCommittedIntersection(&primary);
    if (hit.kind == RAY_QUERY_INTERSECTION_NONE) {
        verdict[idx] = 0u;
        return;
    }

    let surface = params.eye.xyz + dir * hit.t;
    var shadow: ray_query;
    rayQueryInitialize(
        &shadow,
        accel,
        RayDesc(0u, 0xFFu, params.sun.w, params.misc.w, surface, params.sun.xyz),
    );
    while (rayQueryProceed(&shadow)) {}
    let occluder = rayQueryGetCommittedIntersection(&shadow);
    if (occluder.kind == RAY_QUERY_INTERSECTION_NONE) {
        verdict[idx] = 1u;
    } else {
        verdict[idx] = 2u;
    }
}
