// Heightfield terrain pass (P10.1): geometry-clipmap LOD patches, vertex-shader
// displaced by a per-tile R32Float height texture. One draw per visible tile
// patch (its LOD mesh + its height texture); per-patch data rides the instance
// buffer. LOD morphing (blend toward the coarser grid) kills popping; skirt
// vertices drop the patch boundary down to hide cracks between differing LODs.
//
// Shading is a debug slope+altitude ramp lit by the scene sun — the P10.4 splat
// material hook is marked below. Opaque, reverse-Z, depth-writing.
//
// The height texture is sampled with `textureLoad` + manual bilinear (not a
// filtering sampler), so R32Float works as UnfilterableFloat everywhere —
// no FLOAT32_FILTERABLE feature needed (keeps headless CI adapters happy).

// The per-tile height texture (R32Float; texel = f32 metre offset from the tile
// origin's Y), bound per patch at @group(1).
@group(1) @binding(0) var height_tex: texture_2d<f32>;
// The per-tile splat-weight texture (Rgba8Unorm; texel = the four normalized
// layer weights), bound per patch beside the height texture (P10.4).
@group(1) @binding(1) var weight_tex: texture_2d<f32>;
// The per-tile biome-id texture (R8Uint; texel = one categorical biome id, 0 =
// unassigned), bound per patch beside the two above (P19.2). Integer-typed on
// purpose — see `load_biome`.
@group(1) @binding(2) var biome_tex: texture_2d<u32>;
// The per-tile HOLE MASK (P21.2): R32Uint, **row-packed** — texel `(w, j)` holds
// the hole bits for samples `[w*32, w*32+32)` of row `j`, LSB-first. A tile with
// no holes binds a **1×1 zero texture** instead of a `ceil(res/32) × res` one, so
// the un-carved case costs four bytes and needs no shader permutation: every
// `textureLoad` past a 1×1 texture's bounds returns zero by WGSL's own rule, and
// zero means "not holed" everywhere. That is the entire sparse-upload mechanism.
//
// Bits rather than a byte-per-sample R8Uint texture, for the same reason the CPU
// layer packs (see `inf_terrain::hole_mask_bytes`): this is a predicate, and 8×
// the bandwidth to carry seven zero bits per sample buys nothing. Row-packed
// rather than tile-packed so the fragment's index is `(i >> 5, j)` — no division
// by a non-constant, and no dependence on the tile resolution at all.
@group(1) @binding(3) var hole_tex: texture_2d<u32>;

// Terrain splat material (@group(2)): four layers + macro variation. Mirrors
// `MaterialRaw` in passes/terrain.rs. `params[k].x` = roughness, `.y` = tex_scale.
struct TerrainMaterial {
    albedo: array<vec4<f32>, 4>,
    params: array<vec4<f32>, 4>,
    macro_amp: vec4<f32>,
    // **WAVE T (the texture document's §3 B): each splat layer's virtual-texture
    // slots**, in `VtTextureSet::slots()` order — albedo (+ detail in the top
    // half), normal (+ detail scale in the top half), ORM, unused.
    //
    // All zero on every terrain that does not bind layer materials, which is
    // every terrain that has ever existed: `layer_is_textured` is then false for
    // all four layers and the fragment takes the flat-colour path below it,
    // instruction for instruction. That is what keeps the committed terrain
    // goldens byte-stable while the capability ships on by construction.
    slots: array<vec4<u32>, 4>,
};
@group(2) @binding(0) var<uniform> material: TerrainMaterial;

// Biome id → debug colour (P19.2), INDEXED BY ID. Mirrors `BiomePaletteRaw` in
// passes/terrain.rs: a fixed 256 slots so a `u8` id is always in range and the
// fragment needs no bounds logic. Every undefined id (including the reserved 0)
// was padded with the unassigned colour on the CPU side. A separate binding from
// `material` so the layer material's bytes are untouched by this mode.
struct BiomePalette {
    colors: array<vec4<f32>, 256>,
};
@group(2) @binding(1) var<uniform> biome_palette: BiomePalette;

// AO + cascaded shadows + dynamic GI ride the shared env bind group at @group(3)
// (declared in env_lighting.wgsl, prepended by `lit_scene_shader`): `ao_tex`/`ao_smp`
// (SSAO, white when off), `shadow_factor()`, and `gi_irradiance()`.

struct VIn {
    // Vertex: unit patch coordinates in [0,1]² + skirt flag (z = 1 on the
    // boundary skirt ring, else 0).
    @location(0) uv_skirt: vec3<f32>,
    // Instance (per patch): origin_local.xyz + world tile span.
    @location(1) o_span: vec4<f32>,
    // Instance: morph, grid cells at this LOD, texture resolution, skirt depth (m).
    @location(2) params: vec4<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_local: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) span_res: vec2<f32>, // span, resolution
    @location(3) height: f32,                             // world height (offset)
};

fn load_texel(ij: vec2<i32>, res: f32) -> f32 {
    let m = i32(res) - 1;
    let c = clamp(ij, vec2<i32>(0, 0), vec2<i32>(m, m));
    return textureLoad(height_tex, c, 0).r;
}

// Bilinearly-sampled height offset (metres) at unit patch coord `uv`.
fn sample_height(uv: vec2<f32>, res: f32) -> f32 {
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (res - 1.0);
    let i0 = floor(p);
    let f = p - i0;
    let ii = vec2<i32>(i0);
    let h00 = load_texel(ii, res);
    let h10 = load_texel(ii + vec2<i32>(1, 0), res);
    let h01 = load_texel(ii + vec2<i32>(0, 1), res);
    let h11 = load_texel(ii + vec2<i32>(1, 1), res);
    let hx0 = mix(h00, h10, f.x);
    let hx1 = mix(h01, h11, f.x);
    return mix(hx0, hx1, f.y);
}

// ── P22.1 SURFACE DEFORMATION ────────────────────────────────────────────────
//
// The ground height a patch actually draws at: the authored heightfield MINUS
// whatever has been pressed into it. `deform.wgsl` (composed above) supplies
// `deform_depth`, which returns 0 outside the window and 0 in every scene with
// no deformation — so this is the identity function on every pre-P22.1 golden.
//
// **This wrapper is the point.** A vertex-only offset would dent the geometry
// and leave the shading flat, which reads as a texture bug rather than a
// footprint: the fragment's central-difference normal must see the same surface
// the vertex stage moved. Both stages call THIS, and neither computes the offset
// itself, so they cannot drift apart.
//
// `uv` is the patch coordinate (for the heightfield) and `world_xz` the SAME
// point in render-local space (for the window) — two parameters rather than one
// derivation, because the vertex stage has the patch origin to hand and the
// fragment stage has `world_local`, and re-deriving either would be a second
// place for the mapping to be wrong.
//
// Holes are handled by composition order, not by a second test: the fragment
// discards a holed cell before it reaches any of this, and a holed vertex draws
// nothing that survives. Testing again here would only let the two stages
// disagree about which samples are ground.
fn ground_height(uv: vec2<f32>, res: f32, world_xz: vec2<f32>) -> f32 {
    return sample_height(uv, res) - deform_depth(world_xz);
}

// ── splat weights + procedural detail (P10.4) ────────────────────────────────

fn load_weight(ij: vec2<i32>, res: f32) -> vec4<f32> {
    let m = i32(res) - 1;
    let c = clamp(ij, vec2<i32>(0, 0), vec2<i32>(m, m));
    return textureLoad(weight_tex, c, 0);
}

// Bilinearly-sampled RGBA splat weights at unit patch coord `uv`, renormalized so
// the four channels sum to 1 (a defensive guard on hand-authored/interpolated
// weights; a zeroed sample falls back to pure layer 0).
fn sample_weights(uv: vec2<f32>, res: f32) -> vec4<f32> {
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (res - 1.0);
    let i0 = floor(p);
    let f = p - i0;
    let ii = vec2<i32>(i0);
    let w00 = load_weight(ii, res);
    let w10 = load_weight(ii + vec2<i32>(1, 0), res);
    let w01 = load_weight(ii + vec2<i32>(0, 1), res);
    let w11 = load_weight(ii + vec2<i32>(1, 1), res);
    let wx0 = mix(w00, w10, f.x);
    let wx1 = mix(w01, w11, f.x);
    var w = mix(wx0, wx1, f.y);
    let s = w.x + w.y + w.z + w.w;
    if (s > 1e-4) {
        w = w / s;
    } else {
        w = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }
    return w;
}

// ── hole mask (P21.2) ──────────────────────────────────────────────────────

// Is sample `(i, j)` holed? Out-of-range reads return zero (WGSL's rule), which
// is exactly what the 1×1 sentinel texture of an un-carved tile relies on.
fn load_hole(ij: vec2<i32>) -> bool {
    let word = textureLoad(hole_tex, vec2<i32>(ij.x >> 5u, ij.y), 0).r;
    return ((word >> (u32(ij.x) & 31u)) & 1u) != 0u;
}

// **THE POISON RULE, raster half.** A holed sample removes the surface from every
// cell that interpolates it, so one holed corner of the bilinear cell under `uv`
// kills the whole cell — character for character the rule
// `inf_terrain::TerrainData::height_at` applies on the CPU. The two MUST agree:
// if the raster poisoned less than the query, a capsule would stand on ground
// nobody draws; if it poisoned more, the cave's rim would be eaten away.
fn is_holed(uv: vec2<f32>, res: f32) -> bool {
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (res - 1.0);
    let m = i32(res) - 1;
    let i0 = clamp(vec2<i32>(floor(p)), vec2<i32>(0), vec2<i32>(m));
    let i1 = min(i0 + vec2<i32>(1, 1), vec2<i32>(m, m));
    return load_hole(i0)
        || load_hole(vec2<i32>(i1.x, i0.y))
        || load_hole(vec2<i32>(i0.x, i1.y))
        || load_hole(i1);
}

// ── biome ids (P19.2) ────────────────────────────────────────────────────────

// One biome id at integer texel `ij`, clamped to the tile (mirrors `load_weight`).
// `textureLoad` on an integer texture: a biome id is CATEGORICAL, so it must never
// be filtered — the "average" of ids 1 and 3 is not id 2, it is nonsense.
fn load_biome(ij: vec2<i32>, res: f32) -> u32 {
    let m = i32(res) - 1;
    let c = clamp(ij, vec2<i32>(0, 0), vec2<i32>(m, m));
    return textureLoad(biome_tex, c, 0).r;
}

// The biome id at unit patch coord `uv`, resolved NEAREST (round to the closest
// sample) for the same reason: ids are labels, not quantities. This is the one
// terrain input that is deliberately not bilinear, so a painted boundary shows as
// the hard edge it actually is instead of a gradient through ids nobody painted.
fn sample_biome(uv: vec2<f32>, res: f32) -> u32 {
    let p = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (res - 1.0);
    return load_biome(vec2<i32>(round(p)), res);
}

// Cheap 2D value-noise (hash lattice → smooth interpolation), range [0, 1].
fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// 4-octave fBm, range ~[0, 1].
fn fbm2(p: vec2<f32>) -> f32 {
    var v = 0.0;
    var amp = 0.5;
    var freq = p;
    for (var i = 0; i < 4; i = i + 1) {
        v = v + amp * vnoise(freq);
        freq = freq * 2.0;
        amp = amp * 0.5;
    }
    return v;
}

// Triplanar axis weights: normalized pow(|n|, sharpness) over the YZ/XZ/XY planes
// (mirrors `triplanar_axis_weights` in passes/terrain.rs, which unit-tests it).
fn triplanar_weights(n: vec3<f32>, sharpness: f32) -> vec3<f32> {
    var w = pow(abs(n), vec3<f32>(sharpness));
    let s = w.x + w.y + w.z;
    if (s > 0.0) {
        w = w / s;
    } else {
        w = vec3<f32>(0.0, 1.0, 0.0);
    }
    return w;
}

// A world-space triplanar detail grain in [0, 1]: value-noise projected on the
// three world planes at `1/tex_scale`, blended by the triplanar axis weights so
// steep faces read the vertical (XY/YZ) projections instead of a stretched top.
fn triplanar_grain(world: vec3<f32>, n: vec3<f32>, tex_scale: f32) -> f32 {
    let scale = 1.0 / max(tex_scale, 0.001);
    let gx = vnoise(world.yz * scale);
    let gy = vnoise(world.xz * scale);
    let gz = vnoise(world.xy * scale);
    let tw = triplanar_weights(n, 4.0);
    return gx * tw.x + gy * tw.y + gz * tw.z;
}

/// What [`terrain_layers`] resolved: the weight-blended surface of whichever
/// splat layers bind a virtual material, and how much of the fragment they cover.
struct TerrainLayered {
    albedo: vec3<f32>,
    roughness: f32,
    /// Sum of the weights of the layers that were textured, in `[0, 1]`. It is
    /// the mix factor against the flat-colour blend, so a terrain where only two
    /// of four layers carry materials fades correctly into the colours of the
    /// two that do not, rather than darkening toward black.
    coverage: f32,
    any: bool,
};

/// **Weight-blend the four splat layers' virtual materials** (Wave T, §3 B).
///
/// A planar XZ projection, deliberately: the shared material is authored as a
/// tiling ground texture and terrain is a heightfield, so `world.xz / tex_scale`
/// is the parametrization the content is drawn for. It stretches on a cliff
/// face — the honest bound, and the same one every heightfield engine has until
/// it pays for a triplanar variant (three times the fetches for the axis-weighted
/// version). The procedural grain above already runs triplanar, so the *break-up*
/// on a steep face survives; only the material's own pattern stretches. A
/// triplanar layer sample is the named follow-up.
///
/// The gradients are taken **once, at the top, outside every branch**: `dpdx` of
/// a value computed inside a divergent branch has no neighbour to difference
/// against. That is the same rule `vt_surface`'s callers follow and the reason it
/// takes its derivatives as parameters.
fn terrain_layers(world: vec3<f32>, w: vec4<f32>, tex_scale: f32) -> TerrainLayered {
    var out: TerrainLayered;
    out.albedo = vec3<f32>(0.0);
    out.roughness = 0.0;
    out.coverage = 0.0;
    out.any = false;
    if (!vt_active()) {
        return out;
    }
    let inv = 1.0 / max(tex_scale, 0.001);
    // **ABSOLUTE world XZ, not render-local.** `world_local` is rebased every
    // time the floating origin snaps (10 m steps), so a uv derived from it would
    // slide the material across the ground as the player walks — a shift of
    // `10 / tex_scale` tiles, which for a 4 m tiling is two and a half tiles of
    // visible jump. `grid_axis_viewport.xy` is the render-local position of the
    // world X/Z axes, i.e. `-origin.xz`, so subtracting it undoes the rebase.
    // The derivatives are unaffected either way (a constant offset), so this
    // costs one subtraction and buys a material that stays where it was put.
    //
    // The bound, stated: this is an f32 world coordinate, so its resolution is
    // about 4 mm at 50 km from the origin. That is under a texel of a 2K
    // material tiled at 4 m out to roughly that range, and past it the tiling
    // begins to quantise. The procedural grain above it has always been
    // render-local and therefore *does* slide; it is a ±15 % tint and nobody has
    // seen it. A material is not.
    let uv = (world.xz - view.grid_axis_viewport.xy) * inv;
    let ddx = dpdx(uv);
    let ddy = dpdy(uv);
    let weights = array<f32, 4>(w.x, w.y, w.z, w.w);
    for (var k = 0u; k < 4u; k = k + 1u) {
        let wk = weights[k];
        // Below a 256th the layer cannot change a texel of an 8-bit output, and
        // the weights are stored as bytes summing to 255 — so this threshold is
        // the weight's own quantum rather than a tuned epsilon.
        if (wk < 0.0039) {
            continue;
        }
        let slots = material.slots[k].xyz;
        if (!vt_bound(slots.x) && !vt_bound(slots.y) && !vt_bound(slots.z)) {
            continue;
        }
        let s = vt_surface(slots, uv, ddx, ddy,
                           material.albedo[k].rgb, 1.0, 0.0, material.params[k].x);
        out.albedo = out.albedo + s.albedo * wk;
        out.roughness = out.roughness + s.roughness * wk;
        out.coverage = out.coverage + wk;
        out.any = true;
    }
    if (out.coverage > 0.0) {
        // Renormalise INSIDE the covered fraction, so the textured layers'
        // colours are their own rather than scaled by how much of the fragment
        // they happen to own; `coverage` then carries that information to the
        // mix at the call site. Getting this wrong is how a half-textured
        // terrain goes dark.
        out.albedo = out.albedo / out.coverage;
        out.roughness = out.roughness / out.coverage;
    }
    return out;
}

@vertex
fn vs(in: VIn) -> VOut {
    let uv = in.uv_skirt.xy;
    let skirt = in.uv_skirt.z;
    let span = in.o_span.w;
    let morph = in.params.x;
    let cells = max(in.params.y, 1.0);
    let res = in.params.z;
    let skirt_depth = in.params.w;

    // LOD morph: blend the fine height toward the height at the next-coarser grid
    // vertex (snap uv to every-other vertex of this LOD). morph 0 → fine, 1 → coarse.
    let coarse_step = 2.0 / cells;
    let coarse_uv = round(uv / coarse_step) * coarse_step;
    // P22.1: both morph ends are read through `ground_height`, so a footprint
    // survives the LOD blend instead of popping out of existence the moment a
    // patch starts morphing toward its coarser grid.
    let h_fine = ground_height(uv, res, in.o_span.xz + uv * span);
    let h_coarse = ground_height(coarse_uv, res, in.o_span.xz + coarse_uv * span);
    let h = mix(h_fine, h_coarse, morph);

    // Render-local position: tile origin + planar offset + displaced height.
    var pos = in.o_span.xyz + vec3<f32>(uv.x * span, h, uv.y * span);
    // Skirt: drop the boundary ring straight down to seal cracks.
    pos.y = pos.y - skirt * skirt_depth;

    var out: VOut;
    out.clip = view.view_proj * vec4<f32>(pos, 1.0);
    out.world_local = pos;
    out.uv = uv;
    out.span_res = vec2<f32>(span, res);
    out.height = h;
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let span = in.span_res.x;
    let res = max(in.span_res.y, 2.0);
    let world_step = span / (res - 1.0);   // world metres between texels
    let texel = 1.0 / (res - 1.0);

    // ── P21.2 HOLE DISCARD ─────────────────────────────────────────
    // The first statement in the fragment, before anything is computed and
    // before depth is written: a carved cell has no surface, so the clipmap
    // draws nothing there and whatever the voxel volume put behind it shows
    // through. An un-carved tile's 1×1 sentinel makes this a texture fetch that
    // always says "no", so every pre-P21.2 terrain golden stays byte-stable.
    //
    // **This also handles the skirt**, and structurally rather than by a second
    // rule. A skirt vertex carries its boundary `uv`, so a skirt fragment tests
    // the very sample it exists to seal the crack under — which means a skirt
    // wall can never survive inside a hole. The honest consequence, stated
    // because it is a real one: where a hole reaches a patch boundary the crack
    // seal goes with it, so a fine↔coarse LOD gap can open at that edge. That is
    // acceptable and preferable — a gap where there is no ground reads as the
    // cave it is, while a skirt wall hanging in a cave mouth reads as a bug.
    if (is_holed(in.uv, res)) {
        discard;
    }

    // Central-difference normal from the height texture (world-space gradient),
    // through the SAME `ground_height` the vertex stage displaced with (P22.1) —
    // so the shading follows the geometry into a footprint instead of lighting a
    // dent as if it were flat. `uv + texel` is exactly `world_step` metres in x,
    // which is why the neighbours' world positions come off `world_local`
    // directly rather than being re-derived from the patch origin.
    let hl = ground_height(in.uv - vec2<f32>(texel, 0.0), res,
                           in.world_local.xz - vec2<f32>(world_step, 0.0));
    let hr = ground_height(in.uv + vec2<f32>(texel, 0.0), res,
                           in.world_local.xz + vec2<f32>(world_step, 0.0));
    let hd = ground_height(in.uv - vec2<f32>(0.0, texel), res,
                           in.world_local.xz - vec2<f32>(0.0, world_step));
    let hu = ground_height(in.uv + vec2<f32>(0.0, texel), res,
                           in.world_local.xz + vec2<f32>(0.0, world_step));
    let dhdx = (hr - hl) / (2.0 * world_step);
    let dhdz = (hu - hd) / (2.0 * world_step);
    let n = normalize(vec3<f32>(-dhdx, 1.0, -dhdz));

    // ── P10.4 SPLAT MATERIAL HOOK ──────────────────────────────────────────
    // Splat-blended layered material: blend the four layers' albedo/roughness by
    // the per-sample weight texture, add a world-space triplanar detail grain
    // (so steep faces don't stretch), then a large-scale fBm macro variation.
    // The lighting below is the shared PBR-lite path (now roughness-aware).
    let w = sample_weights(in.uv, res);
    var albedo = w.x * material.albedo[0].rgb
        + w.y * material.albedo[1].rgb
        + w.z * material.albedo[2].rgb
        + w.w * material.albedo[3].rgb;
    var roughness = clamp(
        w.x * material.params[0].x + w.y * material.params[1].x
            + w.z * material.params[2].x + w.w * material.params[3].x,
        0.04, 1.0);
    let tex_scale = w.x * material.params[0].y + w.y * material.params[1].y
        + w.z * material.params[2].y + w.w * material.params[3].y;

    // ── WAVE T · MATERIAL LAYERING VIA VIRTUAL TEXTURES (§3 B) ─────────────
    // *"For terrain or large props, store blend weights in a virtual texture
    // mask and sample shared, tileable PBR materials. This delivers infinite
    // visual variety with minimal unique texture storage."*
    //
    // The weights half has shipped since P10.4 — a per-sample RGBA8 mask that
    // renormalises to exactly 255. What was missing is the other half: a layer
    // was a FLAT COLOUR, and the component said so in its own doc comment
    // ("a texture GUID is deliberately absent... per-layer albedo/normal/ORM
    // texture refs are the documented follow-up"). This is that follow-up.
    //
    // Each layer that binds one samples a **shared, tileable** virtual material
    // at `world.xz / tex_scale`, which is the same `tex_scale` the procedural
    // grain already used — so a layer's authored tiling means one thing, not
    // two. Sharing is what makes the storage claim true: `VtTextures` dedupes
    // by GUID, so four terrains using one rock material register it once, and a
    // 2K rock set covers a 50 km world at whatever density `tex_scale` asks for.
    //
    // Weight-gated per layer: a layer whose weight is zero here contributes no
    // texture fetch at all, so the cost is what is actually visible at this
    // fragment (typically one or two layers) rather than four unconditionally.
    let layered = terrain_layers(in.world_local, w, tex_scale);
    if (layered.any) {
        albedo = mix(albedo, layered.albedo, layered.coverage);
        roughness = clamp(mix(roughness, layered.roughness, layered.coverage), 0.04, 1.0);
    }

    // Triplanar detail grain (subtle multiplicative tint, ±15%).
    let grain = triplanar_grain(in.world_local, n, tex_scale);
    albedo = albedo * (0.85 + 0.30 * grain);

    // Macro variation: large-scale fBm brightening/darkening (signed, ±amp).
    let macro_fbm = 2.0 * fbm2(in.world_local.xz * 0.01) - 1.0;
    albedo = albedo * (1.0 + material.macro_amp.x * macro_fbm);
    albedo = clamp(albedo, vec3<f32>(0.0), vec3<f32>(1.0));

    // P22.1 COMPACTION DARKENING. Pressed ground is denser ground: the air is
    // squeezed out of the snow, the mud closes over, and what light gets in
    // scatters fewer times before it is absorbed. A single multiplicative term
    // on the splat blend, at most −25% at full depth — deliberately NOT a second
    // copy of the P20.3 wetness model, which is a different physical story (a
    // film of water on top) with its own roughness term and its own uniform.
    // `deform_depth` is 0 in every scene without deformation, so this is a
    // multiply by 1 and every pre-P22.1 terrain golden stays byte-stable.
    if (deform_enabled()) {
        let pressed = clamp(deform_depth(in.world_local.xz) / max(dfm.params.y, 1e-4), 0.0, 1.0);
        albedo = albedo * (1.0 - 0.25 * pressed);
    }
    // ───────────────────────────────────────────────────────────────────────

    // Biomes view mode (P19.2): tint by the per-sample biome id instead of by the
    // splat material — the terrain reads as a map of the painted vocabulary. This
    // is checked BEFORE the unlit branch because Biomes sets BOTH flags (`flags.x`
    // makes every other kind of geometry in the frame render unlit); testing unlit
    // first would swallow the tint.
    //
    // The colour is the palette entry, shaped by a wrapped N·L so the landform's
    // relief still reads under a flat fill (0.55…1.0 of the tint). It is a pure
    // function of the surface normal and the view's sun direction — no lights, no
    // shadows, no GI, no fog — so the mode is deterministic frame to frame and
    // between adapters, which is what makes it goldenable.
    //
    // Like `flags.x`, `flags.y` is EXACTLY 0.0 in every other mode, so this branch
    // is present-but-false and the arithmetic below runs instruction-for-instruction
    // unchanged — every pre-P19.2 terrain golden stays byte-stable.
    if (view.flags.y > 0.5) {
        let id = sample_biome(in.uv, res);
        let tint = biome_palette.colors[id].rgb;
        let ndl_b = clamp(dot(n, normalize(view.sun_dir.xyz)), 0.0, 1.0);
        return vec4<f32>(tint * (0.55 + 0.45 * ndl_b), 1.0);
    }

    // Unlit view mode (R-P2): return the splat-blended albedo directly, skipping
    // the sun/ambient/spec lighting below. Terrain carries no emissive term. The
    // flag is 0 in the default Lit mode, so the terrain golden stays byte-stable.
    if (view.flags.x > 0.5) {
        return vec4<f32>(albedo, 1.0);
    }

    // P20.3 SHORELINE WETNESS. Terrain is where this reads: a coastline is a
    // terrain crossing a water level, and the darkened, glossier band at that
    // crossing is what makes water look like it is sitting IN the landscape
    // rather than on top of it. A pure function of the fragment's world position
    // and the frame's water bodies — no camera, no screen space, no map (see
    // `shaders/wetness.wgsl`). Placed AFTER the Biomes and Unlit returns so both
    // debug views keep showing the authored data rather than a wetted version of
    // it. `wet.dims.x` is 0 on every scene without water, so the branch is
    // present-but-false and every pre-P20.3 terrain golden is byte-stable.
    if (wet.dims.x > 0u) {
        let wetted = wet_apply(in.world_local, albedo, roughness);
        albedo = wetted.rgb;
        roughness = clamp(wetted.a, 0.04, 1.0);
    }

    let sun = normalize(view.sun_dir.xyz);
    let ndl = max(dot(n, sun), 0.0);
    // Hemispheric ambient (sky above / ground below), or the dynamic-GI probe
    // irradiance when GI is on.
    let up = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    var ambient = mix(vec3<f32>(0.05, 0.06, 0.07), vec3<f32>(0.16, 0.20, 0.26), up);
    if (gi.params.x > 0.5) {
        ambient = gi_irradiance(in.world_local, n);
    }
    // A cheap roughness-aware specular glint (Blinn-ish): smoother layers (lower
    // roughness) get a tighter, brighter highlight, so roughness reads visibly.
    let view_dir = normalize(view.eye.xyz - in.world_local);
    let half_v = normalize(sun + view_dir);
    let gloss = (1.0 - roughness) * (1.0 - roughness);
    let spec_power = mix(8.0, 128.0, gloss);
    let spec = pow(max(dot(n, half_v), 0.0), spec_power) * gloss * 0.4;
    // The direct sun (+ its glint) receives the cascaded shadow factor; SSAO
    // modulates only the ambient term.
    var direct = ndl * vec3<f32>(1.15, 1.10, 1.0);
    var spec_term = vec3<f32>(spec);
    if (sun_shadowing_enabled()) {
        let sf = shadow_factor(in.world_local, n);
        direct = direct * sf;
        spec_term = spec_term * sf;
    }
    // P17.3: the cloud layer's soft, large-scale sun occlusion. Terrain is where
    // this reads most — a kilometre-wide cloud shadow drifting over a valley is
    // the whole point of baking the map. Guarded like the CSM block above, so a
    // cloudless scene is byte-identical.
    if (atmos.clouds.x > 0.5 && atmos.cloud_shadow.x > 0.0) {
        let cf = cloud_shadow_factor(in.world_local);
        direct = direct * cf;
        spec_term = spec_term * cf;
    }
    let ao = textureSampleLevel(ao_tex, ao_smp, in.clip.xy / view.grid_axis_viewport.zw, 0.0).r;
    var lo = albedo * (ambient * ao + direct) + spec_term;
    // P18.4 GI specular. Terrain has no `f0` of its own (it is a dielectric splat
    // blend, and its existing glint is a direct-sun Blinn lobe), so this is an
    // ADDITIVE environment term at the dielectric 0.04 rather than a replacement —
    // which also means a terrain golden with GI off runs the identical arithmetic.
    if (gi.params.x > 0.5 && gi.params2.x > 0.5) {
        lo = lo + gi_specular(in.world_local, n, view_dir, roughness, vec3<f32>(0.04)) * ao;
    }

    // HDR-linear haze; the post tonemap pass (ACES + exposure) runs afterward.
    // P17.2: replaced by physical aerial perspective + height fog when the scene
    // has an atmosphere. Terrain is the pass that shows this off — it is the only
    // geometry that reliably reaches the horizon.
    let dist = length(in.world_local - view.eye.xyz);
    let haze = 1.0 - exp(-dist * 0.0025);
    var col = mix(lo, vec3<f32>(0.055, 0.081, 0.120), haze * 0.5);
    if (atmos.params.x > 0.5) {
        col = atmos_apply(lo, in.world_local);
    }
    return vec4<f32>(col, 1.0);
}
