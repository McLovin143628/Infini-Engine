// vgeom_mesh.wgsl — vertex-pulled meshlet raster (P13.1b). `common_view.wgsl` is
// prepended (the shared `View` uniform + helpers).
//
// No vertex buffers: the cull compute produced a `visible` list of (instance,
// meshlet) pairs and an indirect draw of `visible_count` instances, each drawing
// `MAX_TRIS*3` indices from a shared `0,1,2,…` index buffer. This shader pulls
// everything from storage:
//   * @builtin(instance_index) → visible[i] → (global instance, asset-local
//     meshlet id) → v_remap[id] → the meshlet's slot in the SHARED pool (P18.2:
//     meshlet/vertex/micro-index data is suballocated across all assets, so every
//     read goes through the per-asset remap table the cull compute also reads)
//   * @builtin(vertex_index)   → triangle = vidx/3, corner = vidx%3
//   * unused triangles (corner beyond the meshlet's triangle_count) collapse to a
//     zero-area (degenerate) triangle off-screen — the standard fixed-index-count
//     instanced-meshlet technique. Cost: up to (MAX_TRIS - triangle_count)*3
//     degenerate vertices per meshlet (documented in passes/vgeom.rs).
// The fragment stage is the same metallic-roughness PBR as mesh.wgsl (duplicated
// so the rigid pipeline — and every existing golden — stays byte-stable).

struct Meshlet {
    center: vec3<f32>,
    radius: f32,
    cone_axis: vec3<f32>,
    cone_cutoff: f32,
    vertex_offset: u32,
    triangle_offset: u32,
    vertex_count: u32,
    triangle_count: u32,
    error: f32,
    parent_error: f32,
    lod_level: u32,
    // **The material slot** (Wave T, the texture document's section 3 C). This
    // word carries `MeshletRec::group` on disk and was `pad` here because no
    // shader read it; `inf_vgeom::stream::stage_page` now overwrites it with the
    // meshlet's material slot on the way into the pool, unconditionally, so the
    // word means one thing on the GPU for every container version. `0` is the
    // instance's own material, which is what every mesh cooked before Wave T
    // resolves to.
    material_slot: u32,
};

struct Instance {
    model: mat4x4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
    color: vec4<f32>,
    emissive: vec4<f32>,
    threshold: f32,
    metallic: f32,
    roughness: f32,
    max_scale: f32,
    pick_id: u32,
    // P26.3: the virtual-texture set (albedo, normal, ORM slots), each a
    // handle + 1. These three words were `p0`/`p1`/`p2` — alignment padding
    // uploaded as zero since P13.1b — so a meshlet instance that samples
    // nothing packs to the bytes it always did.
    vt0: u32,
    vt1: u32,
    vt2: u32,
};

// Group 3: the virtualized-geometry storage buffers.
// **The meshlet vertex record's stride, in floats** (P28.2). One `VgeomVertex`
// is position(3) + normal(3) + uv(2) + a packed tangent word(1) = 9, and this
// constant is the Rust `VERTEX_REC_LEN / 4` read back — `the_shaders_vertex_
// stride_is_the_rusts` fails the day the two stop agreeing, in every shader that
// reads the pool. A wrong stride here does not error: it reads a neighbouring
// vertex's bytes as this one's position, which is a mesh that renders and is
// wrong everywhere.
const VGEOM_VSTRIDE: u32 = 9u;
@group(3) @binding(0) var<storage, read> v_positions: array<f32>; // VGEOM_VSTRIDE f32 / vertex
@group(3) @binding(1) var<storage, read> v_meshlets: array<Meshlet>;
@group(3) @binding(2) var<storage, read> v_meshlet_verts: array<u32>;
@group(3) @binding(3) var<storage, read> v_meshlet_tris: array<u32>; // packed u8
@group(3) @binding(4) var<storage, read> v_instances: array<Instance>;
@group(3) @binding(5) var<storage, read> v_visible: array<vec2<u32>>;
// Asset-local meshlet id -> slot in the shared meshlet pool (P18.2).
@group(3) @binding(7) var<storage, read> v_remap: array<u32>;

const NOT_RESIDENT: u32 = 0xFFFFFFFFu;

// Debug flag pushed via the view uniform's spare slot is awkward; instead we bake
// a specialization through a small uniform. Reuse mode: a dedicated tiny uniform.
struct VgeomFlags {
    // x = debug meshlet colouring (0/1), yzw unused.
    flags: vec4<u32>,
};
@group(3) @binding(6) var<uniform> vg: VgeomFlags;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) @interpolate(flat) id: u32,
    @location(4) @interpolate(flat) pbr: vec4<f32>,
    @location(5) @interpolate(flat) emissive: vec3<f32>,
    @location(6) @interpolate(flat) meshlet: u32,
    // P26.3: the meshlet path reads the mesh's REAL uv. `v_positions` has
    // carried 8 floats per vertex since P13.1b — position, normal, and a uv
    // pair nothing read until now.
    @location(7) uv: vec2<f32>,
    @location(8) @interpolate(flat) vt: vec3<u32>,
    // P28.2: the vertex-level tangent frame (`xyz` local-space tangent, `w`
    // handedness), interpolated. `w == 0` means this asset carried none and the
    // fragment falls back to the derivative frame it used before.
    @location(9) tangent: vec4<f32>,
};

fn read_tri_byte(byte_addr: u32) -> u32 {
    let word = v_meshlet_tris[byte_addr >> 2u];
    let shift = (byte_addr & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

@vertex
fn vs(@builtin(vertex_index) vidx: u32, @builtin(instance_index) iidx: u32) -> VsOut {
    var out: VsOut;
    let pair = v_visible[iidx];
    let inst = v_instances[pair.x];
    let slot = v_remap[pair.y];

    let tri = vidx / 3u;
    let corner = vidx % 3u;

    // The cull compute only ever appends resident pairs, so this cannot fire —
    // but a stale visible list must degenerate rather than read a freed pool slot.
    if (slot == NOT_RESIDENT) {
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0);
        out.pbr = vec4<f32>(0.0);
        return out;
    }
    let m = v_meshlets[slot];

    // Degenerate padding for unused triangles: emit a fixed off-screen point (all
    // three corners of the unused triangle map here ⇒ zero area ⇒ no fragments).
    if (tri >= m.triangle_count) {
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0);
        out.pbr = vec4<f32>(0.0);
        return out;
    }

    let local = read_tri_byte(m.triangle_offset + tri * 3u + corner);
    let global_v = v_meshlet_verts[m.vertex_offset + local];
    let base = global_v * VGEOM_VSTRIDE;
    let position = vec3<f32>(v_positions[base], v_positions[base + 1u], v_positions[base + 2u]);
    let normal = vec3<f32>(v_positions[base + 3u], v_positions[base + 4u], v_positions[base + 5u]);
    let uv = vec2<f32>(v_positions[base + 6u], v_positions[base + 7u]);
    let tan4 = vgeom_unpack_tangent(bitcast<u32>(v_positions[base + 8u]));

    let nrm = mat3x3<f32>(inst.n0.xyz, inst.n1.xyz, inst.n2.xyz);
    let wp = inst.model * vec4<f32>(position, 1.0);
    out.pos = view.view_proj * wp;
    out.world_pos = wp.xyz;
    out.normal = nrm * normal;
    out.color = inst.color;
    out.id = inst.pick_id;
    out.pbr = vec4<f32>(inst.metallic, inst.roughness, 0.0, 0.0);
    out.emissive = inst.emissive.rgb;
    // Into world space with the model matrix's LINEAR part, not the normal
    // matrix (P28.2 audit). A tangent is contravariant — it lies IN the surface —
    // so it rides `rotation · scale`, while a normal rides the inverse transpose
    // `rotation · scale⁻¹`. Under uniform scale the two are proportional and
    // normalization erases the difference; under NON-uniform instance scale they
    // differ by more than a positive factor, the transformed tangent is not
    // tangent to the deformed surface, and Gram-Schmidt cannot recover what the
    // transform destroyed — a frame worse than the derivative one it replaced.
    // The handedness rides through untouched, because it is a sign and not a
    // vector. `vis_resolve` does the identical thing, which is what keeps the
    // parity nucleus a comparison rather than two conventions.
    let tmat = mat3x3<f32>(inst.model[0].xyz, inst.model[1].xyz, inst.model[2].xyz);
    out.tangent = vec4<f32>(tmat * tan4.xyz, tan4.w);
    out.meshlet = pair.y;
    out.uv = uv;
    out.vt = vec3<u32>(inst.vt0, inst.vt1, inst.vt2);
    return out;
}


// AO + cascaded shadows + dynamic GI + the atmosphere ride the shared env bind
// group at @group(2) (declared in env_lighting.wgsl / atmosphere*.wgsl, prepended
// by `lit_scene_shader` — P17.2 promoted this pass from the AO-only bind to the
// same env bind every other lit pass uses, so aerial perspective reaches meshlet
// geometry too).







// A distinct, stable per-meshlet debug colour (hash → hue-ish rgb).
fn meshlet_color(id: u32) -> vec3<f32> {
    let h = (id * 2654435761u) ^ (id << 13u);
    let r = f32((h >> 16u) & 0xFFu) / 255.0;
    let g = f32((h >> 8u) & 0xFFu) / 255.0;
    let b = f32(h & 0xFFu) / 255.0;
    // Lift out of the very-dark corner so clusters read clearly.
    return vec3<f32>(0.15) + 0.85 * vec3<f32>(r, g, b);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    if (vg.flags.x != 0u) {
        // Flat per-meshlet debug colouring (LOD/cluster structure proof).
        return vec4<f32>(meshlet_color(in.meshlet), 1.0);
    }
    // P26.5 RESIDENCY HEAT-MAP (`ViewMode::VtResidency`): every virtual-textured
    // surface painted by how far behind the streamer is at that pixel.
    //
    // **Before the unlit short-circuit, not after.** `VtResidency` sets
    // `flags.x` too (the `Biomes` precedent: everything that is not the overlay
    // renders flat so the ramp is readable), so a heat branch placed below the
    // unlit return would never execute — measured, and it is the whole defect a
    // real-frame arm catches and a source read does not.
    //
    // `flags.z` is 0.0 in every other mode, so this is present-and-false and
    // every golden runs the identical arithmetic. The derivatives are taken
    // here, in uniform control flow: `flags.z` is a uniform, so the return is
    // uniform, and `in.uv` is an interpolated input either way.
    if (view.flags.z > 0.5) {
        return vec4<f32>(vt_heat(in.vt, in.uv, dpdx(in.uv), dpdy(in.uv)), 1.0);
    }
    // P27.5 VsmPages: the shadow-page residency ramp, beside the texture one and
    // for the same reason. `flags.w` is 0.0 in every other mode, so this is
    // present-and-false and every golden runs the identical arithmetic. Above
    // the unlit short-circuit, which `VsmPages` also sets (the `VtResidency`
    // precedent, and the defect a source read does not catch).
    if (view.flags.w > 0.5) {
        return vec4<f32>(vsm_heat(in.world_pos, normalize(in.normal)), 1.0);
    }
    // Unlit view mode (R-P2): albedo + emissive, no lighting (Unlit + Wireframe).
    if (view.flags.x > 0.5) {
        return vec4<f32>(in.color.rgb + in.emissive, in.color.a);
    }

    var n = normalize(in.normal);
    let v = normalize(view.eye.xyz - in.world_pos);
    var albedo = in.color.rgb;
    var metallic = clamp(in.pbr.x, 0.0, 1.0);
    var rough = clamp(in.pbr.y, 0.04, 1.0);
    // P26.3 VIRTUAL TEXTURES. Zero on every instance that names no texture, so
    // the branch is present-and-false for every scene that predates this batch
    // and the arithmetic below is byte-identical.
    // The screen derivatives, taken in UNIFORM control flow — a fragment shader
    // may only difference against its neighbours outside a divergent branch, and
    // the VT branch below is per instance. Cheap when nothing samples: a
    // derivative of an interpolated value is two subtractions.
    let vt_ddx = dpdx(in.uv);
    let vt_ddy = dpdy(in.uv);
    let vt_dpx = dpdx(in.world_pos);
    let vt_dpy = dpdy(in.world_pos);
    var vt_ao = 1.0;
    if (in.vt.x != 0u || in.vt.y != 0u || in.vt.z != 0u) {
        let s = vt_surface(in.vt, in.uv, vt_ddx, vt_ddy,
                           albedo, in.color.a, metallic, rough);
        albedo = s.albedo;
        metallic = clamp(s.metallic, 0.0, 1.0);
        rough = clamp(s.roughness, 0.04, 1.0);
        vt_ao = s.occlusion;
        if (s.has_normal) {
            n = vt_apply_normal_t(
                n, in.tangent, vt_dpx, vt_dpy, vt_ddx, vt_ddy, s.normal_ts
            );
        }
    }
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);
    // Wave PAR0: every directional light, then the local lights this
    // fragment's froxel lists — the shared library (`lights.wgsl`).
    lo += lights_direct(in.world_pos, in.pos.xy, n, v, view.sun_dir.xyz,
                        albedo, metallic, rough, f0, LIGHT_SUN_SHADOW | LIGHT_LOCAL_SHADOW);
    // Wave PAR0: the cluster-grid debug view paints the froxel's light count.
    if (lights_debug_view()) {
        return vec4<f32>(lights_debug_heat(in.world_pos, in.pos.xy), 1.0);
    }

    let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    // P18.4: meshlet geometry now both CONTRIBUTES to the GI volume (its root-page
    // meshlet spheres are voxelized) and RECEIVES from it. Before this it took the
    // hemispheric constant even with GI on — a gap, since the rigid path had read
    // probes since P13.3b. The branch is not taken with GI off, so every vgeom
    // golden stays byte-identical.
    // Wave FIX3: one door. The hemispheric constant is the fallback for a level
    // that asked for no computed ambient; otherwise the sky's own irradiance
    // plus the probe field's signed difference from it. See
    // `ambient_irradiance` / `sky_irradiance` in `env_lighting.wgsl`.
    let amb = ambient_irradiance(
        in.world_pos,
        n,
        mix(vec3<f32>(0.03, 0.03, 0.035), vec3<f32>(0.10, 0.13, 0.18), up),
    );
    // The material's own occlusion map multiplies the screen-space AO: both
    // modulate the AMBIENT term only, never the direct light above.
    let ao = textureSampleLevel(ao_tex, ao_smp, in.pos.xy / view.grid_axis_viewport.zw, 0.0).r
        * vt_ao;
    lo += amb * albedo * (1.0 - metallic) * ao;
    lo += gi_ambient_specular(in.world_pos, n, v, rough, f0, amb) * ao;

    lo += in.emissive;

    let dist = length(in.world_pos - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.004);
    var col = mix(lo, vec3<f32>(0.055, 0.081, 0.120), haze * 0.4);
    if (atmos.params.x > 0.5) {
        col = atmos_apply(lo, in.world_pos);
    }

    return vec4<f32>(col, in.color.a);
}

@fragment
fn fs_id(in: VsOut) -> @location(0) u32 {
    return in.id;
}
