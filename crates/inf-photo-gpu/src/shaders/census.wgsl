// The census transform: the GPU mirror of `sweep::census_transform`.
//
// This stage is BIT-EXACT against the CPU by construction, not by measurement:
// every operation is an integer comparison of `u8` intensities (carried one per
// `u32` word so the indexing needs no bit shuffling) and an `or` into a mask.
// There is no float in it, so there is no rounding mode, no contraction and no
// adapter in it either.
//
// The tap order is the same raster order the CPU walks — `dy` outer, `dx` inner,
// centre skipped without consuming a bit — because the code is compared as a
// whole word and a permuted tap order would produce a different word for the
// same neighbourhood.

struct CensusParams {
    width: u32,
    height: u32,
    dilation: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: CensusParams;
// One intensity per word. Four times the memory of a packed byte buffer and
// worth it: a packed layout would need a shift-and-mask whose byte order is one
// more thing to keep in step with the CPU for no measurable gain at these sizes.
@group(0) @binding(1) var<storage, read> image: array<u32>;
@group(0) @binding(2) var<storage, read_write> codes: array<u32>;

// Clamp-to-edge addressing, the same rule `GrayImage::clamped` uses.
fn at(x: i32, y: i32) -> u32 {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    return image[u32(cy) * params.width + u32(cx)];
}

@compute @workgroup_size(8, 8)
fn cs_census(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let centre = at(x, y);
    let d = i32(max(params.dilation, 1u));
    var code: u32 = 0u;
    var bit: u32 = 0u;
    for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
        for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            let v = at(x + dx * d, y + dy * d);
            if (v < centre) {
                code = code | (1u << bit);
            }
            bit = bit + 1u;
        }
    }
    codes[gid.y * params.width + gid.x] = code;
}
