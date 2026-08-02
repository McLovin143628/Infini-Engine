// Dynamic-GI scene voxelization (P13.3b, rebuilt in P18.4): one compute thread per
// voxel of the camera-centred `dims.x`³ volume writes TWO packed words into a
// storage buffer —
//
//   voxels[i*2 + 0] = RGBA8 albedo + binary occupancy
//   voxels[i*2 + 1] = RGBA8 emissive (rgb normalized, a = magnitude / GI_EMISSIVE_MAX)
//
// The voxelizer is a **gather** (one thread per voxel, first hit wins) because a
// scatter would race on the voxel word, and a race is nondeterminism. What P18.4
// changed is what it gathers over: instead of a hard-capped linear scan of ≤256
// rigid boxes, each voxel walks only the primitives binned into its **macro cell**
// (`crate::gi::bin_macro_cells`, CSR offsets + items), which is what let the cap
// go. Cell lists are ascending in priority (nearest-volume-centre first), so
// "first hit wins" is a deterministic, priority-respecting choice.
//
// Primitive kinds (`albedo.w`): 0 = oriented box (rigid mesh instances, skinned
// per-joint boxes), 1 = sphere (vgeom per-meshlet spheres of the always-resident
// root page). Both test the voxel centre in instance-local space via the
// precomputed inverse model matrix.
//
// Terrain is NOT an instance: it arrives as one height + albedo per voxel COLUMN
// (`crate::gi::sample_terrain_column`), and a voxel is terrain-solid when its
// centre is at or below that height. A column costs O(1) per voxel however many
// tiles are resident.
//
// Mirrors `crate::gi` (voxel indexing / volume layout / packing).

struct GiData {
    vol_min: vec4<f32>,     // xyz render-local min corner, w = voxel_size
    probe_min: vec4<f32>,   // xyz render-local probe grid min, w = extent
    dims: vec4<f32>,        // x = gi_dim, yzw = probe dims
    params: vec4<f32>,      // x = enabled, y = intensity, z = rays, w = macro dim
    sun_dir: vec4<f32>,     // xyz toward sun
    sun_color: vec4<f32>,   // rgb = sun radiance
    sky_zenith: vec4<f32>,  // rgb
    sky_horizon: vec4<f32>, // rgb
    params2: vec4<f32>,     // x = specular, y = ssr, z = ssr distance, w = ssr thickness
    sched: vec4<f32>,       // x = probe start, y = probe count, z = probe total, w = sky mode
};
@group(0) @binding(0) var<uniform> gi: GiData;

struct GiInstance {
    inv_model: mat4x4<f32>, // render-local world → instance-local
    albedo: vec4<f32>,      // rgb, w = kind (0 box, 1 sphere)
    emissive: vec4<f32>,    // rgb self-emitted radiance, w unused
};
@group(0) @binding(1) var<storage, read> instances: array<GiInstance>;
@group(0) @binding(2) var<storage, read_write> voxels: array<u32>;
// CSR macro-cell bins: cell c owns cell_items[cell_offsets[c] .. cell_offsets[c+1]].
@group(0) @binding(3) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> cell_items: array<u32>;

struct GiTerrainColumn {
    height: f32,   // render-local Y the column is solid up to
    albedo: u32,   // packed RGB8 (splat-blended)
    present: u32,  // 0 = no resident tile covers this column
    pad: u32,
};
@group(0) @binding(5) var<storage, read> terrain: array<GiTerrainColumn>;

/// Largest emissive radiance the volume can carry per channel. Mirrors
/// `crate::gi::EMISSIVE_MAX`.
const GI_EMISSIVE_MAX: f32 = 16.0;

fn pack_rgba8(c: vec4<f32>) -> u32 {
    let q = vec4<u32>(clamp(c, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0 + 0.5);
    return q.x | (q.y << 8u) | (q.z << 16u) | (q.w << 24u);
}

fn unpack_rgb8(v: u32) -> vec3<f32> {
    return vec3<f32>(
        f32(v & 0xffu),
        f32((v >> 8u) & 0xffu),
        f32((v >> 16u) & 0xffu),
    ) / 255.0;
}

// Emissive packing: rgb normalized by the colour's own maximum component, alpha
// = that maximum over GI_EMISSIVE_MAX. Relative rather than absolute
// quantization, so a dim emissive keeps its hue. Mirrors `gi::pack_emissive`.
fn gi_pack_emissive(e: vec3<f32>) -> u32 {
    let maxc = clamp(max(e.x, max(e.y, e.z)), 0.0, GI_EMISSIVE_MAX);
    if (maxc <= 0.0) {
        return 0u;
    }
    let rgb = clamp(e / maxc, vec3<f32>(0.0), vec3<f32>(1.0));
    return pack_rgba8(vec4<f32>(rgb, maxc / GI_EMISSIVE_MAX));
}

fn macro_index(c: vec3<u32>, macro_dim: u32) -> u32 {
    return (c.z * macro_dim + c.y) * macro_dim + c.x;
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

    // ── binned primitives ──
    let macro_dim = max(u32(gi.params.w), 1u);
    let per_cell = max(dim / macro_dim, 1u);
    let cell = min(vec3<u32>(x, y, z) / per_cell, vec3<u32>(macro_dim - 1u));
    let ci = macro_index(cell, macro_dim);
    let lo = cell_offsets[ci];
    let hi = cell_offsets[ci + 1u];

    var out = vec4<f32>(0.0);
    var emissive = 0u;
    for (var i = lo; i < hi; i = i + 1u) {
        let inst = instances[cell_items[i]];
        let local = (inst.inv_model * vec4<f32>(center, 1.0)).xyz;
        var inside = false;
        if (inst.albedo.w > 0.5) {
            // Unit sphere in instance-local space (vgeom meshlet bounds).
            inside = dot(local, local) <= 0.25;
        } else {
            // Unit cube ±0.5 in instance-local space (rigid + skinned joints).
            inside = all(abs(local) <= vec3<f32>(0.5));
        }
        if (inside) {
            out = vec4<f32>(inst.albedo.rgb, 1.0);
            emissive = gi_pack_emissive(inst.emissive.rgb);
            break;
        }
    }

    // ── terrain columns ──
    //
    // Only where no primitive claimed the voxel: a prop standing on the ground
    // should shade as the prop, not as the dirt under it.
    if (out.w < 0.5) {
        let col = terrain[z * dim + x];
        if (col.present != 0u && center.y <= col.height) {
            out = vec4<f32>(unpack_rgb8(col.albedo), 1.0);
        }
    }

    voxels[idx * 2u + 0u] = pack_rgba8(out);
    voxels[idx * 2u + 1u] = emissive;
}
