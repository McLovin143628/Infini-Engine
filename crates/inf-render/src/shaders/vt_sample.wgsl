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

/// The low half of a per-instance map word is the slot; the high half is Wave T's
/// detail lane (see [`vt_detail_slot`] / [`vt_detail_scale`]).
const VT_SLOT_MASK: u32 = 0xFFFFu;

/// Whether a per-instance texture slot names a virtual texture.
///
/// **A slot holds `handle + 1`, so 0 is "no texture"** — the three reserved
/// per-instance words this rides in have shipped as zero since P7.1, so an
/// instance that names nothing is byte-identical to what it always was, and the
/// no-texture fallback is structural rather than a flag someone has to set.
///
/// Wave T masks to 16 bits on the way in. A handle is bounded by the texture
/// count and the atlas slot field is 16 bits (`inf_vt::table::MAX_SLOT_INDEX`),
/// so the high half of the word was always zero and is now the detail lane —
/// which is why an instance written before Wave T reads identically.
fn vt_bound(word: u32) -> bool {
    let slot = word & VT_SLOT_MASK;
    return slot != 0u && vt_active() && (slot - 1u) < vt_table[1];
}

/// **The detail-map slot** (Wave T, the document's §3 A): `handle + 1` of the
/// high-frequency tileable map, packed into the top 16 bits of the *albedo* word.
///
/// It rides there rather than in a fourth per-instance word because there is no
/// fourth word to be had: `InstanceRaw::misc` is `[pick_id, albedo, normal, orm]`
/// and the skinned pipeline already sits exactly on `max_vertex_attributes: 16`
/// (`docs/memos/p26-5-vertex-streams.md`). Both halves were free by construction,
/// so the whole channel costs zero bytes of instance data and zero bindings.
fn vt_detail_slot(maps: vec3<u32>) -> u32 {
    return maps.x >> 16u;
}

/// **How many times the detail map tiles across one unit of the base uv**
/// (Wave T) — unsigned 8.8 fixed point in the top 16 bits of the *normal* word.
///
/// Fixed point rather than a half-float because the whole useful range is a
/// multiplier of roughly 1 to 256 and 8.8 covers it to 1/256 with an *exact*
/// integer round-trip in both directions: `inf_render::scene::detail_scale_q8`
/// is the CPU half and it is pure integer arithmetic, so the value the author
/// authored is the value the shader reads, on every target, with no float
/// conversion anywhere in the chain. Zero — the pre-Wave-T value of these bits —
/// reads back as `0.0`, which `vt_apply_detail` treats as "no detail", so the
/// disabled state and the never-written state are the same state.
fn vt_detail_scale(maps: vec3<u32>) -> f32 {
    return f32(maps.y >> 16u) / 256.0;
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
    let lod = vt_lod(block, ddx, ddy);
    return min(u32(lod), mip_count - 1u);
}

/// [`vt_mip`] **before the truncation** — the continuous level of detail.
///
/// Split out for Wave T's trilinear arm (T47): the fractional part is exactly
/// the blend weight between two levels, and `vt_mip` used to throw it away, which
/// is what a mip band popping as the camera dollies *is*.
fn vt_lod(block: u32, ddx: vec2<f32>, ddy: vec2<f32>) -> f32 {
    let rec0 = block + VT_TEX_HEADER_WORDS;
    let size = vec2<f32>(f32(vt_table[rec0]), f32(vt_table[rec0 + 1u]));
    let dx = ddx * size;
    let dy = ddy * size;
    let rho = max(length(dx), length(dy));
    // `max(rho, 1e-8)` keeps `log2` off -inf on a degenerate (zero-area)
    // fragment; the `max(…, 0.0)` keeps a magnified sample on mip 0.
    //
    // **This line is byte-pinned** against `vis_feedback.wgsl`'s copy by
    // `inf_render::visbuffer::tests::the_per_pixel_mark_uses_the_same_mip_rule`
    // — the level the streamer is asked for and the level the sampler reads are
    // one rule, and a feedback pass chasing a different one is a streamer paging
    // tiles nothing samples. It stayed spelled exactly this way through Wave T's
    // split of the fractional part out of `vt_mip`.
    let lod = max(log2(max(rho, 1e-8)), 0.0);
    return lod;
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
fn vt_sample(word: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    let slot = word & VT_SLOT_MASK;
    let b = vt_table[VT_HEADER_WORDS + (slot - 1u)];
    let mip_count = vt_table[b];

    // A virtual texture's address space is [0,1)²; a tiling material wraps into
    // it. `uv - floor(uv)` is `fract` with the sign behaviour repeat wants, and
    // the gradients above were taken BEFORE the wrap, so the seam does not
    // collapse to mip 0.
    let w_uv = uv - floor(uv);

    let lod = vt_lod(b, ddx, ddy);
    let m = min(u32(lod), mip_count - 1u);
    let lo = vt_tap(b, w_uv, m);

    // ── WAVE T · TRILINEAR (T47) ────────────────────────────────────────────
    // `vt_mip` truncated the level of detail and sampled one page, so the image
    // changed in a single step as the camera dollied — the mip band this
    // document's §"Trilinear" asks to remove. Off by default (flags bit 2 of the
    // texture's header word, set pool-wide from `VirtualTextureSettings::
    // trilinear`), because turning it on moves pixels in every textured scene
    // and the committed goldens are frozen.
    //
    // The doc's early-outs are kept and they matter: at `blend < 0.01` the
    // second tap changes nothing an 8-bit output can show, and at the pyramid's
    // floor there is no coarser level to blend toward. Both are the common case
    // — a surface at its native scale, and a surface at its coarsest — so the
    // second address walk is paid only where it is visible.
    if ((vt_table[b + 3u] & 4u) != 0u) {
        let blend = lod - floor(lod);
        if (blend > 0.01 && m + 1u < mip_count) {
            return mix(lo, vt_tap(b, w_uv, m + 1u), blend);
        }
    }
    return lo;
}

/// **One tap**: resolve `w_uv` at mip `m` and read the physical page.
///
/// Factored out of [`vt_sample`] for the trilinear arm, on exactly the pattern
/// P26.4 used for [`vt_resolve`] — the address walk and the atlas read are one
/// piece of arithmetic, and two copies of it would be two things that can stop
/// agreeing about the border offset.
fn vt_tap(b: u32, w_uv: vec2<f32>, m: u32) -> vec4<f32> {
    let tile_sz = f32(vt_table[b + 1u]);
    let border = f32(vt_table[b + 2u]);
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
fn vt_sample_color(word: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> vec4<f32> {
    let b = vt_table[VT_HEADER_WORDS + ((word & VT_SLOT_MASK) - 1u)];
    let texel = vt_sample(word, uv, ddx, ddy);
    if ((vt_table[b + 3u] & 1u) != 0u) {
        return vec4<f32>(vt_srgb_to_linear(texel.rgb), texel.a);
    }
    return texel;
}

/// **The tangent-space normal a normal map carries, whichever way it stores it**
/// (Wave T).
///
/// Two storage conventions now reach this function and they must produce the
/// same vector:
///
/// * three channels (RGBA8 / BC1 / BC3, everything imported before Wave T) —
///   `xyz · 2 − 1`, exactly what shipped;
/// * **two** channels (BC5, `reconstruct_z` set in the table's flags bit 1) —
///   `xy · 2 − 1`, with `z = sqrt(max(0, 1 − x² − y²))`.
///
/// The rebuild is not an approximation: a tangent-space normal is a *unit*
/// vector, so its Z is determined by X and Y up to a sign, and the sign is
/// positive by the definition of tangent space (a normal pointing into the
/// surface is not a normal map, it is damage). The `max(0, …)` is the only
/// slack, and it exists because a *filtered* pair of quantised channels can land
/// a hair outside the unit disc; there the honest answer is the flattest normal
/// consistent with the data, which is `z = 0` and then `normalize`.
fn vt_normal_ts(word: u32, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> vec3<f32> {
    let b = vt_table[VT_HEADER_WORDS + ((word & VT_SLOT_MASK) - 1u)];
    let t = vt_sample(word, uv, ddx, ddy);
    if ((vt_table[b + 3u] & 2u) != 0u) {
        let xy = t.xy * 2.0 - vec2<f32>(1.0);
        let z = sqrt(max(1.0 - dot(xy, xy), 0.0));
        return normalize(vec3<f32>(xy, z) + vec3<f32>(0.0, 0.0, 1e-6));
    }
    return normalize(t.xyz * 2.0 - vec3<f32>(1.0) + vec3<f32>(0.0, 0.0, 1e-6));
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
        // `vt_normal_ts` is what knows whether the third channel is stored or
        // rebuilt (Wave T's BC5).
        out.normal_ts = vt_normal_ts(maps.y, uv, ddx, ddy);
        out.has_normal = true;
    }
    if (vt_bound(maps.z)) {
        let orm = vt_sample(maps.z, uv, ddx, ddy);
        out.occlusion = orm.r;
        out.roughness = roughness * orm.g;
        out.metallic = metallic * orm.b;
    }
    // ── WAVE T · DETAIL MAPS (the document's §3 A) ───────────────────────────
    // Applied LAST, so it modifies whatever the three base maps resolved to
    // rather than racing them. Structurally inert without a detail slot: the
    // high half of the albedo word is zero on every instance written before
    // Wave T, `vt_detail_slot` returns 0, `vt_bound(0)` is false, and this is a
    // branch that never taken.
    vt_apply_detail(&out, maps, uv, ddx, ddy);
    return out;
}

/// **Blend a high-frequency tileable detail map over a resolved surface**
/// (Wave T — the document's §3 A, verbatim: *"2K base textures paired with
/// high-frequency 512×512 tileable detail normal/roughness maps"*).
///
/// # What it costs, and what it buys
///
/// The saving the document is after is not subtle. A unique 8K albedo + normal +
/// ORM set is ~340 MB of `.inf_tex`; a 2K set with one *shared* 512² detail map
/// is ~21 MB plus a single 512² page set that **every object referencing it
/// registers once** — because a detail map is a tiling texture and this engine
/// deduplicates virtual textures by GUID (`VtTextures::by_guid`), so the second
/// user of a detail map pays nothing at all, in the atlas or on disk.
///
/// # The blend
///
/// * **Normal** — the UDN blend, `normalize(vec3(base.xy + detail.xy, base.z))`.
///   Cheaper than reoriented normal mapping and, unlike a naive `mix`, it cannot
///   flatten the base normal where the detail is neutral: a neutral detail normal
///   is `(0, 0, 1)`, whose XY is zero, so it adds nothing. That is the property
///   that makes this safe to leave on.
/// * **Roughness** — multiplied by the detail sample's **alpha**. A three- or
///   four-channel detail map can therefore carry a micro-roughness break-up in a
///   channel it was not using; a BC5 detail map samples alpha as exactly `1.0`,
///   so it modulates nothing. One rule, no format branch, and the two storage
///   choices disagree about nothing.
///
/// # The fade, and why it is the mip chain rather than a distance
///
/// A detail map tiled 64× is aliasing noise at any range where its texel is
/// smaller than a pixel, and the classical fix is a distance ramp with two magic
/// numbers in it. This uses the pyramid instead: the detail texture's own
/// gradient mip is already the answer to "how far away is this", and by the time
/// it resolves to the coarsest levels the map has averaged toward neutral
/// anyway. The weight ramps to zero across the last two levels, so the detail
/// leaves smoothly and the fade needs no camera, no uniform and no tuning — and
/// it is correct under a magnifying zoom as well as a walk, which a distance
/// ramp is not.
fn vt_apply_detail(
    s: ptr<function, VtSurface>,
    maps: vec3<u32>,
    uv: vec2<f32>,
    ddx: vec2<f32>,
    ddy: vec2<f32>,
) {
    let word = vt_detail_slot(maps);
    let scale = vt_detail_scale(maps);
    if (!vt_bound(word) || !(scale > 0.0)) {
        return;
    }
    let duv = uv * scale;
    let dx = ddx * scale;
    let dy = ddy * scale;

    let b = vt_table[VT_HEADER_WORDS + ((word & VT_SLOT_MASK) - 1u)];
    let mip_count = vt_table[b];
    let lod = vt_mip(b, dx, dy);
    // Ramp out over the last two levels of the detail texture's own pyramid.
    let w = clamp(f32(mip_count - 1u) - f32(lod), 0.0, 1.0);
    if (!(w > 0.0)) {
        return;
    }

    let d = vt_sample(word, duv, dx, dy);
    let dn = vt_normal_ts(word, duv, dx, dy);
    let base = (*s).normal_ts;
    (*s).normal_ts = normalize(vec3<f32>(base.xy + dn.xy * w, base.z) + vec3<f32>(0.0, 0.0, 1e-6));
    (*s).has_normal = true;
    (*s).roughness = (*s).roughness * mix(1.0, d.a, w);
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

/// **The tangent word** (P28.2): the octahedral direction + handedness a
/// `VgeomVertex` carries, unpacked.
///
/// The mirror of `inf_vgeom::model::unpack_tangent`, and pinned against it by
/// `the_shaders_tangent_unpack_is_the_rusts`, which reads this source. The word
/// is loaded out of the vertex pool as an `f32` and `bitcast` here because the
/// pool has one `array<f32>` view and no room for a second (`vis_resolve` is a
/// FRAGMENT pass and P28.1 measured the storage bindings fully spent) — which is
/// safe *because the packer pins the exponent field*: every real word is a
/// normal float in ±[1, 2), never subnormal, never NaN, so nothing here can be
/// flushed or canonicalized. `0.0` is the "no authored tangent" sentinel and
/// returns `w == 0`, which is how a caller asks for the derivative frame.
fn vgeom_unpack_tangent(word: u32) -> vec4<f32> {
    if (word == 0u) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    // Sign-extend the two 11-bit fields by shifting them to the top and back.
    let ex = f32(i32(word << 21u) >> 21u) / 1023.0;
    let ey = f32(i32(word << 10u) >> 21u) / 1023.0;
    var v = vec3<f32>(ex, ey, 1.0 - abs(ex) - abs(ey));
    if (v.z < 0.0) {
        let sx = select(1.0, -1.0, v.x < 0.0);
        let sy = select(1.0, -1.0, v.y < 0.0);
        v = vec3<f32>((1.0 - abs(v.y)) * sx, (1.0 - abs(v.x)) * sy, v.z);
    }
    let h = select(1.0, -1.0, (word & 0x400000u) != 0u);
    return vec4<f32>(normalize(v), h);
}

/// `vt_apply_normal` with a **vertex-level** tangent frame, where one exists.
///
/// P26.5 routed this here: the derivative frame above is exact for a planar patch
/// and first-order elsewhere, and what it costs is normal-map orientation on
/// curved patches. `tan4` is the interpolated `vgeom_unpack_tangent` — `xyz` the
/// tangent in the same space as `n`, `w` the handedness — and `w == 0` means the
/// vertex carried none, which is every asset cooked before `.inf_vmesh` v3 and
/// every primitive that has no authored parametrization. Those fall straight
/// back, so the channel is additive: a surface without one shades exactly as it
/// did before this batch.
///
/// Gram-Schmidt against `n` rather than trusting the interpolated tangent: linear
/// interpolation of two unit tangents is neither unit nor perpendicular to the
/// interpolated normal, and a frame that is not orthonormal skews the mapped
/// normal in a way that reads as a lighting error rather than as a tangent one.
fn vt_apply_normal_t(
    n: vec3<f32>,
    tan4: vec4<f32>,
    dp1: vec3<f32>,
    dp2: vec3<f32>,
    duv1: vec2<f32>,
    duv2: vec2<f32>,
    ts: vec3<f32>,
) -> vec3<f32> {
    if (tan4.w == 0.0) {
        return vt_apply_normal(n, dp1, dp2, duv1, duv2, ts);
    }
    let t = tan4.xyz - n * dot(n, tan4.xyz);
    let l = length(t);
    if (!(l > 1e-6)) {
        // Degenerate only when the authored tangent is parallel to the normal,
        // which is authoring damage rather than a frame this can rescue.
        return vt_apply_normal(n, dp1, dp2, duv1, duv2, ts);
    }
    let tangent = t / l;
    // The **sign** of `w`, never its magnitude (P28.2 audit). `vis_resolve` takes
    // the handedness from the nearest corner, so its `w` is exactly ±1; the
    // forward path in `vgeom_mesh` carries it as an ordinary interpolated varying,
    // so a triangle straddling a uv-mirror seam delivers a fractional `w` — and
    // `cross(n, tangent) * w` then hands back a bitangent SHORTER than unit,
    // which is precisely the non-orthonormal frame the Gram-Schmidt above exists
    // to prevent. Two consumers of one function must not disagree about that.
    let bitangent = cross(n, tangent) * select(-1.0, 1.0, tan4.w > 0.0);
    return normalize(tangent * ts.x + bitangent * ts.y + n * ts.z);
}

/// **The residency heat-map** (P26.5) — what `ViewMode::VtResidency` paints.
///
/// One question per fragment: *how many levels coarser than the one this pixel
/// asked for is the tile it actually got?* That is the number a streaming bug
/// shows up in and the number a budget decision moves, and it is exactly
/// `vt_resolve`'s `got - m` — the same walk the sampling path takes, so the view
/// mode cannot disagree with the picture it is explaining.
///
/// The ramp is five buckets and deliberately not a gradient: a continuous colour
/// invites reading a number off it, and there is no number here, only "the
/// streamer is N levels behind".
///
///   grey    no virtual texture on this surface (the scalar path)
///   green   exact — the tile this pixel wanted is resident
///   yellow  one level of fallback
///   orange  two
///   red     three or more, or the analytic floor standing alone
///
/// Grey is load-bearing: it is what tells an author that a surface they believed
/// was textured is not bound at all, which no amount of staring at a lit frame
/// distinguishes from a texture that happens to look like its scalar colour.
fn vt_heat(maps: vec3<u32>, uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> vec3<f32> {
    var slot = 0u;
    if (vt_bound(maps.x)) { slot = maps.x; }
    else if (vt_bound(maps.y)) { slot = maps.y; }
    else if (vt_bound(maps.z)) { slot = maps.z; }
    if (slot == 0u) {
        return vec3<f32>(0.06, 0.06, 0.07);
    }
    // Wave T: the map words carry a detail lane in their top half — mask it off
    // before indexing the directory, exactly as `vt_sample` does.
    let b = vt_table[VT_HEADER_WORDS + ((slot & VT_SLOT_MASK) - 1u)];
    let w_uv = uv - floor(uv);
    let m = vt_mip(b, ddx, ddy);
    let r = vt_resolve(b, w_uv, m);
    // `got` is never finer than `m` (an entry names an ANCESTOR), so this
    // subtraction cannot wrap.
    let behind = r.got - m;
    if (behind == 0u) { return vec3<f32>(0.10, 0.85, 0.20); }
    if (behind == 1u) { return vec3<f32>(0.95, 0.85, 0.10); }
    if (behind == 2u) { return vec3<f32>(1.00, 0.45, 0.05); }
    return vec3<f32>(0.95, 0.08, 0.08);
}
