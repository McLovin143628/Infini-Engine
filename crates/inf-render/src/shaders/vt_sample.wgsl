// vt_sample.wgsl — **sampling a virtual texture** (P26.3).
//
// Composed into every lit shader by `lit_deform_shader`, with `GROUP_ENV`
// substituted for the pass's environment group — the same mechanism
// `env_lighting.wgsl` / `wetness.wgsl` / the atmosphere library use, and for the
// same reason: the bind-group LAYOUT is shared, so declaring the bindings in one
// place is what keeps the declaration and the layout from drifting. A pass that
// never calls `vt_sample` simply carries dead functions (naga drops the unused
// globals, so it does not even acquire the bindings).
//
// ── the three bindings, folded past index 13 ─────────────────────────────────
//
// The ruling in `docs/memos/p26-28-virtualization-direction.md` — fold VT into
// the shared env group rather than raise `max_bind_groups` — executed. 14/15/16
// follow AO (0,1), shadows (2,3,4), GI (5,6), atmosphere (7,8,9,10), the cloud
// shadow (11), the scene depth (12) and wetness (13). `Limits::default()` allows
// 1000 bindings per group and 4 groups; the scarce resource is the GROUP, and
// this spends none.
//
// ── the table, and the walk ──────────────────────────────────────────────────
//
// The layout is `inf_vt::table`'s, verbatim; every offset is a WORD offset so
// there is no byte arithmetic here at all.
//
//   [0] magic  [1] texture_count  [2] slots_x  [3] stored_tile_size
//   [4 + t]    word offset of texture t's block
//   block:     [0] mip_count [1] tile_size [2] border [3] flags (bit 0 = srgb)
//   mip recs:  width · height · tiles_x · first_entry (ABSOLUTE word offset)
//   entries:   slot in bits 0..16, RESOLVED mip in bits 16..24
//
// An entry never names the tile that was asked for: it names the **finest
// resident ancestor** of it, which is why it carries a mip. So the shader
// re-derives its texel position at the mip it was GIVEN — and it must find the
// tile by walking the tile tree DOWN from the address it asked for, one clamped
// `min(t / 2, tiles - 1)` step per level, reading only mip records.
//
// **It must not re-derive the tile from `uv` at the resolved mip.** That is the
// P26.2 audit's finding, and it is measured rather than argued: the container
// halves an extent with `w / 2`, so a 511-texel level sits over a 255-texel one
// and `uv × 255` is not `texel / 2`. Mip 0's texel 256 — the first texel of tile
// 2 — is 127.999 of mip 1, i.e. tile 0, while the entry it read was resolved
// through tile 1, and the sample lands a whole tile (128 texels) away at exactly
// one column of one tile boundary. Measured across a swept pyramid set in
// `inf_vt::address`: 1 address of 511×3, 50 of 1023², 51 of 2047×511, 1 322 of
// 4095². `tests/vt_sampling.rs` runs the correct walk on the GPU against
// `VtTextureDesc::ancestor` on the CPU, on those same extents.
//
// What IS true — the sampling contract — is that the resolved level's texel sits
// inside the named tile's payload EXTENDED BY THE BORDER RING, never more than
// one texel out (`local` never leaves `[-1, tile_size)` across twelve pyramids).
// The ring is filled from the same level's texels under the same clamp, so the
// slack reads a real neighbour. That is what `+ border` below rests on.
//
// ── sRGB is decoded HERE, and that is the P26.2 remainder's answer ───────────
//
// There is one atlas, so its format is a pool-wide decision while `srgb` is a
// per-texture fact — a base colour is sRGB-encoded and a normal/ORM map is not,
// and both live in the same pool. P26.2 left the question open with two
// candidates (two pools, or a `view_formats` reinterpretation). Both are
// rejected here, and a third is taken:
//
// * **two pools** doubles the atlas, the residency, the budget accounting and
//   the binding count (3 → 6) to express one bit;
// * **`view_formats`** needs two views of one atlas (4 bindings) — and it still
//   does not answer the question, because a decode is then a property of the
//   VIEW, so the shader must read the per-texture srgb flag to pick a view
//   anyway. It pays a binding for a `select`.
// * **decoding in the shader**, from the `srgb` flag already sitting in the
//   table's texture header, keeps the atlas linear, keeps `VT_BINDING_COUNT` at
//   the three P26.2 promised, and is exact per texture.
//
// The cost is honest and bounded: the transfer function is applied to the
// FILTERED result rather than per texel before filtering, so a bilinear tap
// across a strong colour edge differs slightly from hardware sRGB decode. This
// is the ordinary "gamma-space filtering" error, it is invisible on the mip
// chain a virtual texture spends its life in, and it is written down rather than
// discovered. The day it matters, the fix is `view_formats` and one more
// binding.

// The physical page atlas — ONE texture, always in the LINEAR variant of the
// pool's format (see above).
@group(GROUP_ENV) @binding(14) var vt_atlas: texture_2d<f32>;
// Linear + clamp. The border ring is what makes clamping correct: a texel at a
// tile's payload edge filters against real neighbours baked into the page, so
// the clamp at the SLOT edge is never reached by a legal sample.
@group(GROUP_ENV) @binding(15) var vt_smp: sampler;
// Every registered texture's indirection table, at `inf_vt::table`'s layout.
@group(GROUP_ENV) @binding(16) var<storage, read> vt_table: array<u32>;

// `b"VTB1"` little-endian — the first word of a real table. A buffer that was
// never written reads zero here, which is what makes `vt_active()` total.
const VT_MAGIC: u32 = 0x31425456u;
const VT_HEADER_WORDS: u32 = 4u;
const VT_TEX_HEADER_WORDS: u32 = 4u;
const VT_MIP_REC_WORDS: u32 = 4u;

/// Whether this frame has a virtual-texture pool at all.
///
/// The renderer binds a 4-byte zero buffer and a 1×1 atlas when it has none, so
/// this is false on every textureless scene and the callers below take the
/// scalar path with no memory traffic — the structural no-op the goldens rest
/// on.
fn vt_active() -> bool {
    return arrayLength(&vt_table) >= VT_HEADER_WORDS && vt_table[0] == VT_MAGIC;
}

/// Whether a per-instance texture slot names a virtual texture.
///
/// **A slot holds `handle + 1`, so 0 is "no texture"** — the three reserved
/// per-instance words this rides in have shipped as zero since P7.1, so an
/// instance that names nothing is byte-identical to what it always was, and the
/// no-texture fallback is structural rather than a flag someone has to set.
fn vt_bound(slot: u32) -> bool {
    return slot != 0u && vt_active() && (slot - 1u) < vt_table[1];
}

/// The exact sRGB→linear transfer function (IEC 61966-2-1), applied to the
/// filtered result — see the header note on what that costs.
fn vt_srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(lo, hi, c > vec3<f32>(0.04045));
}

/// Tiles down a level of `height` texels at `tile_size` payload texels a side.
fn vt_tiles_y(height: u32, tile_size: u32) -> u32 {
    return max((height + tile_size - 1u) / tile_size, 1u);
}

/// **The gradient mip.** `ddx`/`ddy` are the derivatives of the *unwrapped*
/// virtual uv, so a tiling surface picks the mip its footprint deserves rather
/// than mip 0 at every seam.
///
/// The classic `log2(max(|ddx|, |ddy|))` in texels of mip 0, clamped into the
/// pyramid. Deliberately NOT `textureSampleGrad` on the atlas: the atlas has no
/// mip chain — the pyramid lives in the virtual address space, and a page IS a
/// level's tile — so the hardware has nothing to select between.
fn vt_mip(block: u32, ddx: vec2<f32>, ddy: vec2<f32>) -> u32 {
    let mip_count = vt_table[block];
    let rec0 = block + VT_TEX_HEADER_WORDS;
    let size = vec2<f32>(f32(vt_table[rec0]), f32(vt_table[rec0 + 1u]));
    let dx = ddx * size;
    let dy = ddy * size;
    let rho = max(length(dx), length(dy));
    // `max(rho, 1e-8)` keeps `log2` off -inf on a degenerate (zero-area)
    // fragment; the `max(…, 0.0)` keeps a magnified sample on mip 0.
    let lod = max(log2(max(rho, 1e-8)), 0.0);
    return min(u32(lod), mip_count - 1u);
}

/// What an address resolved to: the physical page, the mip it was actually
/// served at, and the TILE of that mip the walk landed on.
struct VtResolved {
    page: u32,
    got: u32,
    gtx: u32,
    gty: u32,
    // The tile at the mip that was ASKED for, kept so a caller can see the walk's
    // start as well as its end.
    tx: u32,
    ty: u32,
};

/// **The address walk** — virtual uv + wanted mip → the page and the tile that
/// serves it (P26.3, factored out in P26.4).
///
/// Its own function so that the compute arm in `tests/vt_sampling.rs` can
/// execute *this* code with a non-zero trip count and compare it against
/// `VtTextureDesc::ancestor` address by address. Before that arm the loop below
/// had never run on a GPU at all: every lit-pixel test makes the whole pyramid
/// resident, so `got == m` and the body never executes. A twin that agrees with
/// a shader nobody ran is a twin that agrees with itself.
///
/// `b` is the texture's block offset, `w_uv` the WRAPPED virtual uv, `m` the mip
/// the gradient asked for.
fn vt_resolve(b: u32, w_uv: vec2<f32>, m: u32) -> VtResolved {
    let tsz = vt_table[b + 1u];
    let tile_sz = f32(tsz);
    let rec = b + VT_TEX_HEADER_WORDS + m * VT_MIP_REC_WORDS;
    let mw = vt_table[rec];
    let mh = vt_table[rec + 1u];
    let tiles_x = vt_table[rec + 2u];
    let tiles_y = vt_tiles_y(mh, tsz);
    // Both axes clamp: a uv reaching exactly 1.0 is off the end of the grid.
    let tx = min(u32(w_uv.x * f32(mw) / tile_sz), tiles_x - 1u);
    let ty = min(u32(w_uv.y * f32(mh) / tile_sz), tiles_y - 1u);
    let entry = vt_table[vt_table[rec + 3u] + ty * tiles_x + tx];

    let page = entry & 0xFFFFu;
    let got = (entry >> 16u) & 0xFFu;

    // Walk the TILE TREE down to the resolved mip — `VtTextureDesc::ancestor`'s
    // clamped chain, one bounded step per LEVEL. See the header for the
    // measurement that forbids re-deriving this from `uv` at `got`.
    var gtx = tx;
    var gty = ty;
    for (var lv = m; lv < got; lv = lv + 1u) {
        let r = b + VT_TEX_HEADER_WORDS + (lv + 1u) * VT_MIP_REC_WORDS;
        gtx = min(gtx / 2u, vt_table[r + 2u] - 1u);
        gty = min(gty / 2u, vt_tiles_y(vt_table[r + 1u], tsz) - 1u);
    }
    return VtResolved(page, got, gtx, gty, tx, ty);
}

/// **Sample a virtual texture.** `slot` is `handle + 1` (0 = none, guarded by
/// [`vt_bound`] at the call site); `uv` is the virtual uv, which may leave
/// `[0,1]` and wraps; `ddx`/`ddy` are its screen derivatives.
///
/// Returns the raw stored value — sRGB is NOT decoded here, because a caller
/// that wants a normal or an ORM triple must not have it decoded and a caller
/// that wants a base colour must. `vt_sample_color` below is the decoded face.
fn vt_sample(slot: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    let b = vt_table[VT_HEADER_WORDS + (slot - 1u)];
    let tsz = vt_table[b + 1u];
    let tile_sz = f32(tsz);
    let border = f32(vt_table[b + 2u]);

    // A virtual texture's address space is [0,1)²; a tiling material wraps into
    // it. `uv - floor(uv)` is `fract` with the sign behaviour repeat wants, and
    // the gradients above were taken BEFORE the wrap, so the seam does not
    // collapse to mip 0.
    let w_uv = uv - floor(uv);

    let m = vt_mip(b, ddx, ddy);
    let r = vt_resolve(b, w_uv, m);

    let grec = b + VT_TEX_HEADER_WORDS + r.got * VT_MIP_REC_WORDS;
    let gp = w_uv * vec2<f32>(f32(vt_table[grec]), f32(vt_table[grec + 1u]));
    // In [-1, tile_size) — the border ring covers the slack four times over.
    let local = gp - vec2<f32>(f32(r.gtx), f32(r.gty)) * tile_sz;

    let slots_x = vt_table[2];
    let stored = f32(vt_table[3]);
    let origin = vec2<f32>(f32(r.page % slots_x), f32(r.page / slots_x)) * stored;
    let dims = vec2<f32>(textureDimensions(vt_atlas, 0));
    let atlas_uv = (origin + vec2<f32>(border) + local) / dims;
    return textureSampleLevel(vt_atlas, vt_smp, atlas_uv, 0.0);
}

/// [`vt_sample`] with the texture's own sRGB flag applied — the base-colour face.
fn vt_sample_color(slot: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    let b = vt_table[VT_HEADER_WORDS + (slot - 1u)];
    let texel = vt_sample(slot, uv, ddx, ddy);
    if ((vt_table[b + 3u] & 1u) != 0u) {
        return vec4<f32>(vt_srgb_to_linear(texel.rgb), texel.a);
    }
    return texel;
}

/// The three maps a surface samples, resolved for one fragment.
///
/// Every field defaults to the value the SCALAR per-instance attribute already
/// carried, and a map that is absent — or whose texture is not yet warm, which
/// the projector expresses by simply not naming it — leaves that value alone.
/// So a textureless instance runs the same arithmetic it always did.
struct VtSurface {
    albedo: vec3<f32>,
    alpha: f32,
    normal_ts: vec3<f32>,
    has_normal: bool,
    metallic: f32,
    roughness: f32,
    occlusion: f32,
};

/// Resolve `maps` = (albedo slot, normal slot, ORM slot) over the scalar
/// fallbacks. **The one call site shape for all three lit paths**, so the rigid,
/// meshlet and skinned fragment stages cannot drift in how they read a material.
///
/// ORM follows the glTF metallic-roughness convention the importer already
/// writes: occlusion in R, roughness in G, metallic in B.
fn vt_surface(
    maps: vec3<u32>,
    uv: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>,
    albedo: vec3<f32>,
    alpha: f32,
    metallic: f32,
    roughness: f32,
) -> VtSurface {
    var out: VtSurface;
    out.albedo = albedo;
    out.alpha = alpha;
    out.normal_ts = vec3<f32>(0.0, 0.0, 1.0);
    out.has_normal = false;
    out.metallic = metallic;
    out.roughness = roughness;
    out.occlusion = 1.0;

    if (vt_bound(maps.x)) {
        let c = vt_sample_color(maps.x, uv, ddx, ddy);
        out.albedo = albedo * c.rgb;
        out.alpha = alpha * c.a;
    }
    if (vt_bound(maps.y)) {
        // Tangent-space normal, stored unsigned. Never sRGB-decoded: the
        // container's `srgb` flag is false for a normal map by import rule, and
        // `vt_sample` (not `vt_sample_color`) is the door that cannot decode.
        let n = vt_sample(maps.y, uv, ddx, ddy).xyz * 2.0 - vec3<f32>(1.0);
        out.normal_ts = normalize(n + vec3<f32>(0.0, 0.0, 1e-6));
        out.has_normal = true;
    }
    if (vt_bound(maps.z)) {
        let orm = vt_sample(maps.z, uv, ddx, ddy);
        out.occlusion = orm.r;
        out.roughness = roughness * orm.g;
        out.metallic = metallic * orm.b;
    }
    return out;
}

/// Re-anchor a tangent-space normal onto an interpolated geometric normal
/// **without a tangent stream**.
///
/// The engine's *render* vertex formats carry no tangent. `MeshVertex` and
/// `SkinnedVertex` are position + normal + **uv** as of P26.5, and
/// `VgeomVertex` has been position + normal + uv since P13.1b — so the basis
/// this derives is now built from a REAL parametrization rather than from a box
/// projection, which is the difference between "first-order correct against the
/// artist's uv" and "first-order correct against a guess".
///
/// A tangent did not come with the uv, and `docs/memos/p26-5-vertex-streams.md`
/// measures why rather than asserting it: the skinned pipeline reached
/// `max_vertex_attributes: 16` exactly with the uv at `@location(15)`, and the
/// two producers that feed the rigid stream are the built-in primitives (which
/// could supply one analytically) and `VgeomVertex` (which has none to give) —
/// so a tangent attribute there would be real data on five shapes and a derived
/// guess on every imported mesh. The data does exist on disk:
/// `inf_mesh::MeshVertex` has carried `tangent: [f32; 4]` (xyz + handedness)
/// since P4. Until a container carries it through the meshlet builder, the basis
/// is derived per fragment from the screen-space derivatives of the world
/// position and the uv, which is the standard cotangent-frame construction:
/// exact for a planar patch, and correct to first order everywhere else.
///
/// A degenerate uv patch (both derivatives zero — a fragment whose uv does not
/// vary) leaves the geometric normal alone rather than producing a NaN basis.
///
/// **The four derivatives are PARAMETERS, not taken here.** `dpdx`/`dpdy` are
/// only meaningful in uniform control flow — a helper lane inside a divergent
/// branch has no neighbour to difference against — so every caller computes
/// them at the top of its fragment stage, outside the "does this instance have
/// a texture" branch, and hands them in. The same reason `vt_surface` takes
/// `ddx`/`ddy` rather than deriving them.
fn vt_apply_normal(
    n: vec3<f32>,
    dp1: vec3<f32>,
    dp2: vec3<f32>,
    duv1: vec2<f32>,
    duv2: vec2<f32>,
    ts: vec3<f32>,
) -> vec3<f32> {
    let det = duv1.x * duv2.y - duv2.x * duv1.y;
    if (abs(det) < 1e-12) {
        return n;
    }
    let t = (dp1 * duv2.y - dp2 * duv1.y) / det;
    let tangent = normalize(t - n * dot(n, t));
    if (!all(tangent == tangent)) {
        return n;
    }
    let bitangent = cross(n, tangent);
    return normalize(tangent * ts.x + bitangent * ts.y + n * ts.z);
}
