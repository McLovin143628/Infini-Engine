// TSDF integration: the GPU mirror of `tsdf::TsdfGrid::integrate_view`.
//
// One dispatch per view, one invocation per voxel. That arrangement is the
// determinism ruling made executable:
//
//  * It is a GATHER. A voxel reads the depth map; the depth map never scatters
//    into voxels. So each accumulator word is written by exactly ONE invocation
//    in this dispatch, and `acc = acc + w * d` needs no atomic. An `atomicAdd`
//    over floats would make the fused volume depend on the order the GPU
//    happened to schedule its workgroups in, which is the shape of thing the
//    house rules exist to keep out of anything that becomes content.
//  * Views are integrated by SEPARATE dispatches in ascending view order,
//    because the sum across views is a floating-point sum and its value depends
//    on the order it is taken in.
//
// The one atomic here counts votes. Integer `atomicAdd` is associative and
// exact, so a count is order-independent in a way a float sum is not — and it
// is the engagement counter that proves this kernel ran at all rather than
// being dispatched over an empty grid.

struct FuseParams {
    // Camera-space position of voxel (0, 0, 0).
    tx: f32,
    ty: f32,
    tz: f32,
    // Camera-space displacement of one voxel step along each grid axis.
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    cx_: f32,
    cy_: f32,
    cz_: f32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    k1: f32,
    k2: f32,
    truncation: f32,
    width: u32,
    height: u32,
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

const MIN_SAMPLE_Z: f32 = 1.0e-6;

@group(0) @binding(0) var<uniform> params: FuseParams;
@group(0) @binding(1) var<storage, read> depth: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> acc_distance: array<f32>;
@group(0) @binding(4) var<storage, read_write> acc_weight: array<f32>;
@group(0) @binding(5) var<storage, read_write> votes: atomic<u32>;

@compute @workgroup_size(4, 4, 4)
fn cs_fuse(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dim_x || gid.y >= params.dim_y || gid.z >= params.dim_z) {
        return;
    }
    let fi = f32(gid.x);
    let fj = f32(gid.y);
    let fk = f32(gid.z);
    let qx = params.tx + fi * params.ax + fj * params.bx + fk * params.cx_;
    let qy = params.ty + fi * params.ay + fj * params.by + fk * params.cy_;
    let qz = params.tz + fi * params.az + fj * params.bz + fk * params.cz_;
    if (!(qz > MIN_SAMPLE_Z)) {
        return;
    }
    let u = qx / qz;
    let v = qy / qz;
    let r2 = u * u + v * v;
    let r4 = r2 * r2;
    let f = 1.0 + params.k1 * r2 + params.k2 * r4;
    let px = params.fx * (u * f) + params.cx;
    let py = params.fy * (v * f) + params.cy;
    let ix = floor(px + 0.5);
    let iy = floor(py + 0.5);
    if (!(ix >= 0.0) || !(iy >= 0.0) || ix > f32(params.width - 1u) || iy > f32(params.height - 1u)) {
        return;
    }
    let sample = u32(iy) * params.width + u32(ix);
    let observed = depth[sample];
    if (!(observed > 0.0)) {
        return;
    }
    let w = weights[sample];
    if (!(w > 0.0)) {
        return;
    }
    // Positive outside, negative inside: `inf-voxel`'s convention.
    let sdf = observed - qz;
    if (sdf < -params.truncation) {
        return;
    }
    var clamped = sdf;
    if (sdf > params.truncation) {
        clamped = params.truncation;
    }
    let index = (gid.z * params.dim_y + gid.y) * params.dim_x + gid.x;
    acc_distance[index] = acc_distance[index] + w * clamped;
    acc_weight[index] = acc_weight[index] + w;
    atomicAdd(&votes, 1u);
}
