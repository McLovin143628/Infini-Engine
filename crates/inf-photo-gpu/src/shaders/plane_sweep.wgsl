// The plane sweep: the GPU mirror of `sweep::sweep_view`.
//
// Every function below is a transcription of the Rust one it is named after, in
// the same operation order. The two places where "the same order" is
// load-bearing:
//
//  * `floor(px + 0.5)`, never `round(px)`. WGSL's `round` is round-half-to-even
//    and Rust's `f32::round` is round-half-away-from-zero; on a sample landing
//    on a half-integer they disagree systematically. `floor` is exact in both.
//  * `!(qz > MIN_SAMPLE_Z)` rather than `qz <= MIN_SAMPLE_Z`, so a NaN is
//    rejected. Both languages evaluate a comparison against NaN as false.
//
// The one operation that can still break bit-parity is the multiply-add:
// `depth * rx + p.tx`, and `1.0 + k1 * r2 + k2 * r4`. This shader never writes
// `fma()`, and naga emits separate multiply and add instructions — but a
// driver's back end may contract them, and a contracted result differs by up to
// one ULP. One ULP in `u` only matters where `floor(u + 0.5)` is within one ULP
// of an integer, and there it is a whole different census fetch. That is the
// entire parity risk in this file; `tests/mvs_gate.rs` measures it rather than
// assuming it away.
//
// The cost curve is NOT stored. The CPU reference keeps a `Vec<u32>` of it
// because a heap allocation per row is free there; a shader would need a
// per-invocation array of `planes` words, which spills. So the second pass
// recomputes `plane_cost`, which is a pure function of its arguments — the same
// values, arrived at twice.

const COST_INVALID: u32 = 4294967295u;
const COST_SCALE: u32 = 256u;
const MIN_SAMPLE_Z: f32 = 1.0e-6;
const MAX_NEIGHBOURS: u32 = 8u;

struct SweepParams {
    width: u32,
    height: u32,
    planes: u32,
    neighbours: u32,
    min_valid: u32,
    max_cost: u32,
    exclusion: u32,
    _pad0: u32,
    inv_near: f32,
    inv_step: f32,
    min_confidence: f32,
    _pad1: f32,
};

// Scalar members only, matching `sweep::NeighbourParams` field for field. A
// `vec3<f32>` would carry a 16-byte alignment rule with it, and this struct
// exists to be the same bytes on both sides of the seam.
struct NeighbourParams {
    tx: f32,
    ty: f32,
    tz: f32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    k1: f32,
    k2: f32,
    width: u32,
    height: u32,
    census_offset: u32,
};

@group(0) @binding(0) var<uniform> params: SweepParams;
@group(0) @binding(1) var<storage, read> neighbours: array<NeighbourParams>;
@group(0) @binding(2) var<storage, read> rays: array<f32>;
@group(0) @binding(3) var<storage, read> depths: array<f32>;
@group(0) @binding(4) var<storage, read> ref_census: array<u32>;
@group(0) @binding(5) var<storage, read> census: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_depth: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_confidence: array<f32>;

fn neighbour_cost(reference_code: u32, ni: u32, pixel: u32, n_px: u32, depth: f32) -> u32 {
    let p = neighbours[ni];
    let base = (ni * n_px + pixel) * 3u;
    let rx = rays[base];
    let ry = rays[base + 1u];
    let rz = rays[base + 2u];
    let qx = depth * rx + p.tx;
    let qy = depth * ry + p.ty;
    let qz = depth * rz + p.tz;
    if (!(qz > MIN_SAMPLE_Z)) {
        return COST_INVALID;
    }
    let u = qx / qz;
    let v = qy / qz;
    let r2 = u * u + v * v;
    let r4 = r2 * r2;
    let f = 1.0 + p.k1 * r2 + p.k2 * r4;
    let px = p.fx * (u * f) + p.cx;
    let py = p.fy * (v * f) + p.cy;
    let ix = floor(px + 0.5);
    let iy = floor(py + 0.5);
    if (!(ix >= 0.0) || !(iy >= 0.0) || ix > f32(p.width - 1u) || iy > f32(p.height - 1u)) {
        return COST_INVALID;
    }
    let index = p.census_offset + u32(iy) * p.width + u32(ix);
    return countOneBits(reference_code ^ census[index]);
}

fn plane_cost(reference_code: u32, pixel: u32, n_px: u32, depth: f32) -> u32 {
    var sorted: array<u32, 8>;
    var valid: u32 = 0u;
    for (var ni: u32 = 0u; ni < params.neighbours; ni = ni + 1u) {
        let c = neighbour_cost(reference_code, ni, pixel, n_px, depth);
        if (c == COST_INVALID) {
            continue;
        }
        var j: u32 = valid;
        loop {
            if (j == 0u) {
                break;
            }
            if (sorted[j - 1u] <= c) {
                break;
            }
            sorted[j] = sorted[j - 1u];
            j = j - 1u;
        }
        sorted[j] = c;
        valid = valid + 1u;
    }
    if (valid < params.min_valid || valid == 0u) {
        return COST_INVALID;
    }
    let half = (valid + 1u) / 2u;
    var sum: u32 = 0u;
    for (var i: u32 = 0u; i < half; i = i + 1u) {
        sum = sum + sorted[i];
    }
    return sum * COST_SCALE / half;
}

@compute @workgroup_size(8, 8)
fn cs_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let pixel = gid.y * params.width + gid.x;
    let n_px = params.width * params.height;
    let code = ref_census[pixel];

    // Pass one: the winner. Strict `<`, so ties go to the LOWER plane index,
    // which is the nearer surface.
    var best: u32 = COST_INVALID;
    var best_p: u32 = 0u;
    for (var p: u32 = 0u; p < params.planes; p = p + 1u) {
        let c = plane_cost(code, pixel, n_px, depths[p]);
        if (c < best) {
            best = c;
            best_p = p;
        }
    }
    if (best == COST_INVALID || best > params.max_cost) {
        out_depth[pixel] = 0.0;
        out_confidence[pixel] = 0.0;
        return;
    }

    // Pass two: the runner-up, outside the winner's own basin.
    var runner: u32 = COST_INVALID;
    for (var p: u32 = 0u; p < params.planes; p = p + 1u) {
        let diff = select(best_p - p, p - best_p, p > best_p);
        if (diff <= params.exclusion) {
            continue;
        }
        let c = plane_cost(code, pixel, n_px, depths[p]);
        if (c < runner) {
            runner = c;
        }
    }

    var confidence: f32 = 1.0;
    if (runner != COST_INVALID) {
        if (runner <= best) {
            confidence = 0.0;
        } else {
            confidence = clamp(f32(runner - best) / (f32(runner) + 1.0), 0.0, 1.0);
        }
    }
    if (confidence < params.min_confidence) {
        out_depth[pixel] = 0.0;
        out_confidence[pixel] = 0.0;
        return;
    }

    var delta: f32 = 0.0;
    if (best_p > 0u && best_p + 1u < params.planes) {
        let a = plane_cost(code, pixel, n_px, depths[best_p - 1u]);
        let c = plane_cost(code, pixel, n_px, depths[best_p + 1u]);
        if (a != COST_INVALID && c != COST_INVALID) {
            let fa = f32(a);
            let fb = f32(best);
            let fc = f32(c);
            let denom = fa - 2.0 * fb + fc;
            if (denom > 0.0) {
                delta = clamp(0.5 * (fa - fc) / denom, -0.5, 0.5);
            }
        }
    }

    let inv = params.inv_near + params.inv_step * (f32(best_p) + delta);
    if (!(inv > 0.0)) {
        out_depth[pixel] = 0.0;
        out_confidence[pixel] = 0.0;
        return;
    }
    out_depth[pixel] = 1.0 / inv;
    out_confidence[pixel] = confidence;
}
