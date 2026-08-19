// visbuffer.wgsl — the visibility buffer's raster (P28.1). `common_view.wgsl` is
// prepended (the shared `View` uniform + helpers) and nothing else: this pass
// shades nothing, so it binds no lights, no environment and no textures.
//
// It is a DELIBERATE second construction of `vgeom_mesh.wgsl`'s vertex pull, not
// a shared one. The forward meshlet path is P28.1's fallback AND the parity
// gate's oracle, and an oracle that shares code with its subject can only prove
// they agree with themselves (the P27.5 closing law: a gate cannot see an error
// the subsystems share). What the two DO share is their input — the same pools,
// the same instance table, the same `view.view_proj` — because a parity claim
// about two programs fed different data is about nothing.
//
// Output: one `R32Uint` texel per pixel, `instance ⊕ meshlet ⊕ triangle` packed
// by `crates/inf-render/src/visbuffer.rs`'s contract. Depth-tested last-writer-
// wins into a single-sample depth of its own.
//
// **Determinism.** The colour written is a function of the fragment that wins
// the depth test, and the depth test is `DEPTH_COMPARE` against a buffer cleared
// to a constant — exactly the argument today's opaque raster rests on. Two
// fragments can only tie at a depth if they are coplanar to the last bit of the
// f32, and a tie is broken by primitive order, which is the indirect draw's
// order, which is the cull compute's append order — and *that* is not ordered.
// So the guarantee this pass makes is the one the opaque path makes and no more:
// **a given draw stream produces the same buffer every time it is replayed**,
// because the same appends happen in the same order for the same committed
// input. `a_visbuffer_is_bit_identical_across_two_runs` pins it on a device.

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

// The flat instance table rides an `Rgba32Float` TEXTURE, one row an instance,
// eleven used texels of sixteen. Not a storage buffer, and the reason is a
// measured device ceiling rather than taste: `Limits::default()` grants eight
// storage buffers a stage, the shared environment group already spends four (GI
// SH, the VT table, the VSM table, its projections) and the four meshlet pools
// are the other four. The ninth is refused by `create_pipeline_layout`. See
// `VIS_FRAGMENT_STORAGE_BINDINGS` in `crates/inf-render/src/passes/visbuffer.rs`.
//
// `textureLoad` is valid in the vertex, fragment AND compute stages, so all
// three visibility passes read the table through one representation.
const VIS_INSTANCE_TEXELS: i32 = 16;

struct Instance {
    model: mat4x4<f32>,
    n0: vec3<f32>,
    n1: vec3<f32>,
    n2: vec3<f32>,
    color: vec4<f32>,
    emissive: vec3<f32>,
    metallic: f32,
    roughness: f32,
    vt: vec3<u32>,
};

fn vis_instance(id: u32) -> Instance {
    let y = i32(id);
    let c0 = textureLoad(v_instance_tex, vec2<i32>(0, y), 0);
    let c1 = textureLoad(v_instance_tex, vec2<i32>(1, y), 0);
    let c2 = textureLoad(v_instance_tex, vec2<i32>(2, y), 0);
    let c3 = textureLoad(v_instance_tex, vec2<i32>(3, y), 0);
    let n0 = textureLoad(v_instance_tex, vec2<i32>(4, y), 0);
    let n1 = textureLoad(v_instance_tex, vec2<i32>(5, y), 0);
    let n2 = textureLoad(v_instance_tex, vec2<i32>(6, y), 0);
    let col = textureLoad(v_instance_tex, vec2<i32>(7, y), 0);
    let emi = textureLoad(v_instance_tex, vec2<i32>(8, y), 0);
    // (threshold, metallic, roughness, max_scale)
    let pbr = textureLoad(v_instance_tex, vec2<i32>(9, y), 0);
    // (pick_id, vt0, vt1, vt2) — u32 words, so they are bitcast rather than
    // converted. `bitcast` is the right call and the FIRST version of this
    // comment gave the wrong reason for it: it said an f32 round-trip breaks
    // "anything past 2^24", which is where the danger is NOT. A virtual-texture
    // slot is `handle + 1`, a small integer, and every u32 below 2^23 has an
    // all-zero f32 exponent field — so as a float every real handle here is
    // SUBNORMAL, and WGSL permits an implementation to flush subnormals to zero.
    // Converting would turn slot 1 into slot 0, which reads as "this surface
    // binds no texture" and shows up as a silently untextured frame rather than
    // as an error (P28.1 audit). A bitcast of a loaded texel is a
    // reinterpretation and takes no arithmetic, which is why it survives;
    // `parity_textured_virtual_texture`'s closing `assert_ne!` is what witnesses
    // that on whatever adapter runs, because a flushed slot makes the textured
    // frame byte-identical to the untextured one.
    let ids = textureLoad(v_instance_tex, vec2<i32>(10, y), 0);
    var out: Instance;
    out.model = mat4x4<f32>(c0, c1, c2, c3);
    out.n0 = n0.xyz;
    out.n1 = n1.xyz;
    out.n2 = n2.xyz;
    out.color = col;
    out.emissive = emi.xyz;
    out.metallic = pbr.y;
    out.roughness = pbr.z;
    out.vt = vec3<u32>(bitcast<u32>(ids.y), bitcast<u32>(ids.z), bitcast<u32>(ids.w));
    return out;
}

// Group 1: the shared virtualized-geometry pools + this asset's visible list.
// (Group 0 is the view uniform; there are no lights and no env here, so the
// pools sit one group lower than in the forward path — this pass's layout is its
// own, and reading the forward path's group numbering into it is exactly the
// coupling the file header refuses.)
// **The meshlet vertex record's stride, in floats** (P28.2). One `VgeomVertex`
// is position(3) + normal(3) + uv(2) + a packed tangent word(1) = 9, and this
// constant is the Rust `VERTEX_REC_LEN / 4` read back — `the_shaders_vertex_
// stride_is_the_rusts` fails the day the two stop agreeing, in every shader that
// reads the pool. A wrong stride here does not error: it reads a neighbouring
// vertex's bytes as this one's position, which is a mesh that renders and is
// wrong everywhere.
const VGEOM_VSTRIDE: u32 = 9u;
@group(1) @binding(0) var<storage, read> v_positions: array<f32>; // VGEOM_VSTRIDE f32 / vertex
@group(1) @binding(1) var<storage, read> v_meshlets: array<Meshlet>;
@group(1) @binding(2) var<storage, read> v_meshlet_verts: array<u32>;
@group(1) @binding(3) var<storage, read> v_meshlet_tris: array<u32>; // packed u8
@group(1) @binding(4) var v_instance_tex: texture_2d<f32>;
@group(1) @binding(5) var<storage, read> v_visible: array<vec2<u32>>;
@group(1) @binding(7) var<storage, read> v_remap: array<u32>;

// x = this asset's base in the FLAT instance table, yzw unused.
struct VisFlags {
    flags: vec4<u32>,
};
@group(1) @binding(6) var<uniform> vis: VisFlags;

const NOT_RESIDENT: u32 = 0xFFFFFFFFu;

// ── the packing contract (crates/inf-render/src/visbuffer.rs) ────────────────
const VIS_TRI_BITS: u32 = 7u;
const VIS_MESHLET_BITS: u32 = 14u;
const VIS_INSTANCE_BITS: u32 = 11u;
const VIS_EMPTY: u32 = 0u;
const VIS_MESHLET_SHIFT: u32 = VIS_TRI_BITS;
const VIS_INSTANCE_SHIFT: u32 = VIS_TRI_BITS + VIS_MESHLET_BITS;

// The instance field stores `instance + 1`, which is what makes VIS_EMPTY
// unreachable from a real fragment. Admission (`VisPacking::admit`) has already
// refused any frame whose fields do not fit, so this masks rather than checks —
// a per-fragment branch on a ceiling the frame was admitted under would be a
// second, weaker copy of the door.
fn vis_pack(instance: u32, meshlet: u32, tri: u32) -> u32 {
    return ((instance + 1u) << VIS_INSTANCE_SHIFT)
        | ((meshlet & ((1u << VIS_MESHLET_BITS) - 1u)) << VIS_MESHLET_SHIFT)
        | (tri & ((1u << VIS_TRI_BITS) - 1u));
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) id: u32,
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
    let slot = v_remap[pair.y];

    let tri = vidx / 3u;
    let corner = vidx % 3u;

    // A stale visible list must degenerate rather than read a freed pool slot —
    // the forward path's own guard, and for the same reason.
    if (slot == NOT_RESIDENT) {
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.id = VIS_EMPTY;
        return out;
    }
    let m = v_meshlets[slot];

    // Degenerate padding for the fixed-index-count draw's unused triangles.
    if (tri >= m.triangle_count) {
        out.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.id = VIS_EMPTY;
        return out;
    }

    let local = read_tri_byte(m.triangle_offset + tri * 3u + corner);
    let global_v = v_meshlet_verts[m.vertex_offset + local];
    let base = global_v * VGEOM_VSTRIDE;
    let position = vec3<f32>(v_positions[base], v_positions[base + 1u], v_positions[base + 2u]);

    let inst = vis_instance(vis.flags.x + pair.x);
    // `view.view_proj` is the JITTERED matrix whenever TAA is on (the renderer
    // overwrites the uniform in place). Reading it — rather than rebuilding a
    // projection — is what keeps the resolve's reconstruction on the same
    // sub-pixel grid the buffer was rasterized on: the P27.1 jitter law, one
    // subsystem later.
    out.pos = view.view_proj * (inst.model * vec4<f32>(position, 1.0));
    out.id = vis_pack(vis.flags.x + pair.x, slot, tri);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) u32 {
    return in.id;
}
