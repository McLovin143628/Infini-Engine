// vsm_receive.wgsl — **sampling a virtual shadow map** (P27.4).
//
// The twin of `inf_render::vsm_receiver`, which carries the whole derivation:
// the clamped-kernel ruling with its four numbers, the two-term bias, the level
// blend and the `VSM_ENTRY_NONE` → *lit* fail direction. Read that module
// first; this file is the arithmetic, not the argument.
//
// Composed into every lit shader by `lit_deform_shader` with `GROUP_ENV`
// substituted — `vt_sample.wgsl`'s mechanism, for its reason: the bind-group
// LAYOUT is shared, so one declaration site is what keeps the declaration and
// the layout from drifting. It is composed **before** `env_lighting.wgsl`,
// because `shadow_factor` calls into here and WGSL has no forward declarations.
//
// ── the four bindings, folded past index 16 ─────────────────────────────────
//
// 17/18/19/20 follow AO (0,1), shadows (2,3,4), GI (5,6), atmosphere (7..10),
// the cloud shadow (11), the scene depth (12), wetness (13) and virtual
// texturing (14,15,16). `Limits::default()` grants 1000 bindings per group and
// 4 groups; the scarce resource is the GROUP and this spends none — the same
// measurement `docs/memos/p26-28-virtualization-direction.md` made for VT.
//
// **There is no sampler**, and that is a ruling rather than an omission. Pages
// are BORDERLESS (`VSM_PAGE_BORDER = 0`, the P27.1 page-geometry memo), so a
// hardware comparison sampler's 2 x 2 footprint at a page edge reads the atlas
// slot NEXT DOOR — an unrelated page of a possibly unrelated light. There is
// nothing correct for the hardware to filter against, so every tap is a
// `textureLoad` at an integer texel and the compare happens here. That also
// makes the clamp exact: a tap is inside its page by integer arithmetic.
//
// ── the storage buffers, and why the fragment stage may read them ───────────
//
// `docs/memos/p26-4-feedback-mechanism.md` records that a per-fragment *write*
// would make `FRAGMENT_WRITABLE_STORAGE` a requirement of every lit pass. These
// are READ-ONLY, which is core wgpu everywhere and which the env group already
// does at binding 5 (the GI SH probes).
//
// ── the projections buffer is THIS frame's ─────────────────────────────────
//
// `VsmSystem::sync` uploads it at the frame's sync point, before the graph, so
// the receiver reads the projections the caster pass rasterized with rather
// than the previous frame's. (`VsmMarker::record` used to write it, after the
// graph — which would have handed the lit passes a frame-stale matrix on top of
// the marking ring's pinned two.)

const VSM_MAGIC: u32 = 0x424D5356u;         // b"VSMB" little-endian
const VSM_HEADER_WORDS: u32 = 4u;
const VSM_LIGHT_HEADER_WORDS: u32 = 4u;
const VSM_LEVEL_REC_WORDS: u32 = 4u;
const VSM_ENTRY_NONE: u32 = 0xFFFFFFFFu;
const VSM_PROJ_ORTHO_KIND: u32 = 0u;
// `inf_render::vsm_receiver`'s constants, mirrored. Each one's derivation is in
// that module's docs and pinned by its arms; the numbers here are checked
// against it character for character by
// `the_receivers_two_halves_spell_one_set_of_constants`.
// P27.5 RETIRED TWO OF THESE, and it is an exchange rather than a loss. The PCF
// radius and its slope bias became a **tier knob** (`VsmSettings::pcf_radius`),
// so they arrive per frame in `vsm.counts.z` and `vsm.params.w` — written by
// `VsmReceiverParams::new`, which derives one from the other so a one-tap kernel
// can never run a three-tap bias. What used to pin them here as constants is
// pinned by `the_kernel_and_its_bias_are_read_from_the_uniform_the_tier_writes`,
// which names the two expressions whose *absence* would be the defect.
const VSM_DEPTH_ULP_BIAS: f32 = 2.3841858e-7;
const VSM_NORMAL_BIAS_TEXELS: f32 = 1.0;
const VSM_MAX_SLOPE: f32 = 8.0;
const VSM_NO_DATA: f32 = -1.0;

struct VsmProjection {
    // Render-local world -> this light/face's LEVEL-0 clip space.
    view_proj: mat4x4<f32>,
    // x = word offset of this light's block, y = its first mask bit (unused
    // here), z = the face index, w = kind (0 = ortho clipmap, 1 = perspective).
    info: vec4<u32>,
    // xyz = the light's render-local position (perspective only), w = the
    // level-0 shadow texel size: world metres for a clipmap, metres per metre of
    // light distance for a perspective light.
    light: vec4<f32>,
    // Per level, finest first: the NDC offset that turns level 0's NDC into that
    // level's (P27.3's per-level snapping). 16 = `inf_vsm::MAX_VSM_LEVELS`.
    level_offset: array<vec2<f32>, 16>,
};

struct VsmReceiver {
    // x = the clipmap level blend band, y = pixels per world unit at one metre,
    // z = the perspective near plane (m), w = the slope bias in page texels
    // (P27.5: `inf_render::vsm_receiver::vsm_slope_bias_texels` of the frame's
    // kernel radius — 2.1213203 at the default, which is what this file's
    // constant used to spell).
    params: vec4<f32>,
    // x = the sun's projection index + 1 (0 = no directional tree),
    // y = the projection count, z = the PCF kernel radius in page texels
    // (P27.5's tier knob), w reserved.
    counts: vec4<u32>,
};

@group(GROUP_ENV) @binding(17) var vsm_atlas: texture_depth_2d;
@group(GROUP_ENV) @binding(18) var<storage, read> vsm_table: array<u32>;
@group(GROUP_ENV) @binding(19) var<storage, read> vsm_proj: array<VsmProjection>;
@group(GROUP_ENV) @binding(20) var<uniform> vsm: VsmReceiver;

// Whether this frame has a virtual shadow system at all. The renderer binds a
// 4-byte zero buffer and a 1x1 depth texture when it has none, so word 0 is not
// the magic and every caller below takes the scalar path with no memory
// traffic — the structural no-op the 50 committed goldens rest on.
fn vsm_active() -> bool {
    return arrayLength(&vsm_table) >= VSM_HEADER_WORDS && vsm_table[0] == VSM_MAGIC;
}

// Whether a light's slot (`GpuLight.params.w`, or the uniform's sun slot) names
// a tree. A slot holds `index + 1`, so 0 is "no virtual shadow for this light".
fn vsm_bound(slot: u32) -> bool {
    return slot != 0u && vsm_active() && (slot - 1u) < vsm.counts.y;
}

// Whether the first shadow-casting DIRECTIONAL light has a tree — what
// `shadow_factor` branches on to choose the virtual path over the cascaded one.
fn vsm_sun_bound() -> bool {
    return vsm_bound(vsm.counts.x);
}

// Which cube face a direction lands on: the dominant axis, in
// `inf_render::vsm::CUBE_FACE_BASES` order (+X, -X, +Y, -Y, +Z, -Z). Exact
// rather than approximate — the six faces are axis-aligned 90 degree frusta, so
// the point is inside face f iff f's axis dominates.
fn vsm_cube_face(d: vec3<f32>) -> u32 {
    let a = abs(d);
    if (a.x >= a.y && a.x >= a.z) {
        return select(1u, 0u, d.x > 0.0);
    }
    if (a.y >= a.z) {
        return select(3u, 2u, d.y > 0.0);
    }
    return select(5u, 4u, d.z > 0.0);
}

// Level L's NDC from level 0's. A clipmap level covers 2^L times level 0 and is
// snapped to its own page stride; a quadtree or a cube face's levels share one
// frustum, so the offset is inert there.
fn vsm_level_ndc(p: VsmProjection, l: vec3<f32>, level: u32) -> vec2<f32> {
    if (p.info.w == VSM_PROJ_ORTHO_KIND) {
        return l.xy / exp2(f32(level)) + p.level_offset[level];
    }
    return l.xy;
}

// The level rule, `inf_render::vsm::vsm_justified_level` mirrored (and the same
// text `vsm_mark.wgsl` carries, so the page a fragment reads is the page its own
// depth marked).
fn vsm_justified_level(texel0: f32, pixel_world: f32, levels: u32) -> u32 {
    let want = ceil(log2(max(pixel_world, 1e-6) / max(texel0, 1e-9)));
    return u32(clamp(want, 0.0, f32(levels - 1u)));
}

// The level a receiver asks for: the justified one, walked UPWARD to the first
// level that contains it (a clipmap's levels are not concentric since P27.3, so
// there is no closed form). Returns `levels` when nothing contains the point,
// which the caller reads as LIT.
fn vsm_receiver_level(p: VsmProjection, levels: u32, l: vec3<f32>, texel0: f32, pixel_world: f32) -> u32 {
    let start = vsm_justified_level(texel0, pixel_world, levels);
    if (p.info.w != VSM_PROJ_ORTHO_KIND) {
        if (abs(l.x) > 1.0 || abs(l.y) > 1.0) {
            return levels;
        }
        return start;
    }
    for (var lv = start; lv < levels; lv = lv + 1u) {
        let q = vsm_level_ndc(p, l, lv);
        if (abs(q.x) <= 1.0 && abs(q.y) <= 1.0) {
            return lv;
        }
    }
    return levels;
}

// **One level's shadow factor**, or `VSM_NO_DATA`.
//
// The comparison is the camera's direction and its sign is the whole of the
// fail story: under reverse-Z a LARGER stored depth is a caster NEARER the
// light, so a receiver is lit exactly when its own biased depth still reaches at
// least as far — `receiver + bias >= stored`.
fn vsm_level_factor(p: VsmProjection, l: vec3<f32>, level: u32, bias: f32) -> f32 {
    let b = p.info.x;
    let levels = vsm_table[b];
    if (level >= levels) {
        return VSM_NO_DATA;
    }
    let q = vsm_level_ndc(p, l, level);
    if (abs(q.x) > 1.0 || abs(q.y) > 1.0) {
        return VSM_NO_DATA;
    }
    let rec = b + VSM_LIGHT_HEADER_WORDS + level * VSM_LEVEL_REC_WORDS;
    let pages_x = vsm_table[rec];
    let pages_y = vsm_table[rec + 1u];
    let first = vsm_table[rec + 2u];
    let face_stride = vsm_table[rec + 3u];
    // NDC y is up and the page grid's rows run down — the flip
    // `vsm_mark.wgsl` and `mark_page_for` both state.
    let u = clamp(q.x * 0.5 + 0.5, 0.0, 0.999999);
    let v = clamp(0.5 - q.y * 0.5, 0.0, 0.999999);
    let px = min(u32(u * f32(pages_x)), pages_x - 1u);
    let py = min(u32(v * f32(pages_y)), pages_y - 1u);
    let entry = vsm_table[first + p.info.z * face_stride + py * pages_x + px];
    // **THE FAIL DIRECTION.** A page with no resident ancestor is not a black
    // hole: it is a light leak. `VSM_ENTRY_NONE` reads LIT, here and nowhere
    // else, and `a_deferred_pages_receiver_is_byte_identical_to_no_shadow_at_all`
    // is the arm that fails if this line is ever written the other way round.
    if (entry == VSM_ENTRY_NONE) {
        return VSM_NO_DATA;
    }
    let slot = entry & 0xFFFFu;
    let got = (entry >> 16u) & 0xFFu;

    // Re-derive the address at the level the table SERVED. A shadow page tree's
    // level is an exact world lattice, so this is identical to walking the
    // ancestor chain and is O(1) instead of O(levels) —
    // `the_re_derived_address_is_the_ancestor_the_chain_walks_to` measures it,
    // and `vt_sample.wgsl` must NOT do the same thing for the reason its own
    // header gives (a `w/2` mip chain has slack this lattice does not).
    let grec = b + VSM_LIGHT_HEADER_WORDS + got * VSM_LEVEL_REC_WORDS;
    let gpx = vsm_table[grec];
    let gpy = vsm_table[grec + 1u];
    let gq = vsm_level_ndc(p, l, got);
    let gu = clamp(gq.x * 0.5 + 0.5, 0.0, 0.999999);
    let gv = clamp(0.5 - gq.y * 0.5, 0.0, 0.999999);
    let gx = min(u32(gu * f32(gpx)), gpx - 1u);
    let gy = min(u32(gv * f32(gpy)), gpy - 1u);

    let page_size = vsm_table[b + 1u];
    let border = vsm_table[b + 2u];
    let side = f32(page_size);
    // The receiver's texel inside the page's payload.
    let local = vec2<f32>(
        gu * f32(gpx) * side - f32(gx) * side,
        gv * f32(gpy) * side - f32(gy) * side,
    );
    let slots_x = max(vsm_table[2], 1u);
    let stored = vsm_table[3];
    let origin = vec2<i32>(
        i32((slot % slots_x) * stored + border),
        i32((slot / slots_x) * stored + border),
    );

    // **The clamped kernel** — taps that leave the page are DROPPED and the
    // weight renormalized, never clamped to the edge (which would double-weight
    // a boundary texel and bias the filter toward whatever the page's own rim
    // holds). See `inf_render::vsm_receiver`'s module docs for the four numbers
    // this ruling rests on. The centre tap is always inside, so the divisor is
    // never zero.
    //
    // **The radius is the frame's, not a constant** (P27.5): `vsm.counts.z`
    // carries `VsmSettings::pcf_radius` after the tier clamp, and at the shipped
    // default it is `VSM_PCF_RADIUS_DEFAULT` — so this loop is P27.4's, iteration
    // for iteration, on every configuration that has not asked for less.
    let radius = i32(vsm.counts.z);
    let base = vec2<i32>(floor(local));
    var sum = 0.0;
    var taps = 0.0;
    for (var dy = -radius; dy <= radius; dy = dy + 1) {
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let t = base + vec2<i32>(dx, dy);
            if (t.x < 0 || t.y < 0 || t.x >= i32(page_size) || t.y >= i32(page_size)) {
                continue;
            }
            let stored_z = textureLoad(vsm_atlas, origin + t, 0);
            sum = sum + select(0.0, 1.0, l.z + bias >= stored_z);
            taps = taps + 1.0;
        }
    }
    if (taps <= 0.0) {
        return VSM_NO_DATA;
    }
    return sum / taps;
}

// The blend weight — the larger of two proximities, both "how close is this
// receiver to falling into the next coarser level". See
// `inf_render::vsm_receiver::vsm_blend_weight`.
fn vsm_blend_weight(band: f32, level: u32, lf: f32, q: vec2<f32>, ortho: bool) -> f32 {
    if (!(band > 0.0)) {
        return 0.0;
    }
    let b = clamp(band, 1e-4, 1.0);
    let t_res = clamp(lf - (f32(level) - 1.0), 0.0, 1.0);
    let w_res = clamp((t_res - (1.0 - b)) / b, 0.0, 1.0);
    if (!ortho) {
        return w_res;
    }
    let m = max(abs(q.x), abs(q.y));
    let w_con = clamp((m - (1.0 - b)) / b, 0.0, 1.0);
    return max(w_res, w_con);
}

// **The receiver.** `slot` is `projection index + 1`; 0 and an inactive system
// both return 1.0, which is what makes every call site a `* 1.0` on a scene
// without virtual shadows.
fn vsm_shadow(world_pos: vec3<f32>, n: vec3<f32>, slot: u32) -> f32 {
    if (!vsm_bound(slot)) {
        return 1.0;
    }
    var pi = slot - 1u;
    var p = vsm_proj[pi];
    let ortho = p.info.w == VSM_PROJ_ORTHO_KIND;
    // A point light owns six consecutive projections; pick the face first, so
    // everything below reads the face's own matrix.
    let faces = vsm_table[p.info.x + 3u] >> 8u;
    if (faces == 6u) {
        let f = vsm_cube_face(world_pos - p.light.xyz);
        if (pi + f >= vsm.counts.y) {
            return 1.0;
        }
        pi = pi + f;
        p = vsm_proj[pi];
    }

    // The world size of one screen pixel here — the quantity the level rule is
    // against, from the same `projection_scale` the marking pass used.
    let view_dist = max(length(world_pos - view.eye.xyz), 1e-4);
    let pixel_world = view_dist / max(vsm.params.y, 1e-6);

    // The level-0 shadow texel here: absolute for a clipmap, distance-scaled for
    // a perspective light.
    var texel0 = p.light.w;
    if (!ortho) {
        texel0 = texel0 * max(length(world_pos - p.light.xyz), 1e-4);
    }

    let levels = vsm_table[p.info.x];
    if (levels == 0u) {
        return 1.0;
    }

    // The direction toward the light: the gradient of `ndc.z` for a clipmap
    // (row 2's xyz — `ndc.z` falls as the receiver moves away), the vector to
    // the position for a perspective light.
    var to_light = normalize(p.light.xyz - world_pos);
    let row2 = vec3<f32>(p.view_proj[0].z, p.view_proj[1].z, p.view_proj[2].z);
    if (ortho) {
        to_light = normalize(row2);
    }

    // **Normal offset, applied BEFORE the projection**: with no page border a
    // displaced lookup may land in a different page, so the displacement has to
    // be part of the address rather than of the tap offsets. Sized at the
    // undisplaced point's level, which is why this probe runs first.
    let probe = p.view_proj * vec4<f32>(world_pos, 1.0);
    if (probe.w <= 0.0) {
        return 1.0;
    }
    let pn = probe.xyz / probe.w;
    if (pn.z <= 0.0 || pn.z > 1.0) {
        return 1.0;
    }
    let level0 = vsm_justified_level(texel0, pixel_world, levels);
    let offset_m = VSM_NORMAL_BIAS_TEXELS * texel0 * exp2(f32(level0));
    let clip = p.view_proj * vec4<f32>(world_pos + n * offset_m, 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    let l = clip.xyz / clip.w;
    // Outside the light's own depth range (reverse-Z: 1 at the near plane, 0 at
    // the far one) — in front of a spot's near plane, or behind a clipmap's
    // finite far plane. Lit.
    if (l.z <= 0.0 || l.z > 1.0) {
        return 1.0;
    }

    let level = vsm_receiver_level(p, levels, l, texel0, pixel_world);
    if (level >= levels) {
        return 1.0;
    }

    // The two derived bias terms: `f32`'s own step (constant in NDC) plus the
    // page's texel density times the slope, converted by the projection's own
    // `d ndc.z / d m`.
    var ndc_per_m = length(row2);
    if (!ortho) {
        ndc_per_m = l.z * l.z / max(vsm.params.z, 1e-3);
    }
    let ndl = clamp(dot(n, to_light), 0.0, 1.0);
    let tan_t = min(sqrt(max(1.0 - ndl * ndl, 0.0)) / max(ndl, 0.05), VSM_MAX_SLOPE);
    // P27.5: `vsm.params.w`, the slope bias for the kernel `vsm.counts.z` names,
    // rather than a constant sized for a kernel that may not be running. At the
    // default it is `VSM_SLOPE_BIAS_TEXELS_DEFAULT` bit for bit — the Rust
    // constant and this literal are the same `f32`, which is what makes the
    // shipped path unchanged.
    let slope = vsm.params.w * tan_t * ndc_per_m;
    let bias = VSM_DEPTH_ULP_BIAS + slope * texel0 * exp2(f32(level));

    var f = vsm_level_factor(p, l, level, bias);
    if (f < 0.0) {
        return 1.0;
    }
    if (level + 1u >= levels) {
        return f;
    }
    let lf = log2(max(pixel_world, 1e-6) / max(texel0, 1e-9));
    let w = vsm_blend_weight(vsm.params.x, level, lf, vsm_level_ndc(p, l, level), ortho);
    if (w <= 0.0) {
        return f;
    }
    let nbias = VSM_DEPTH_ULP_BIAS + slope * texel0 * exp2(f32(level + 1u));
    let nf = vsm_level_factor(p, l, level + 1u, nbias);
    // The coarser level can miss the receiver too — a clipmap's rings are not
    // nested about one centre since P27.3. Keep this level's answer rather than
    // fading toward lit, which would flash: `shadow_factor`'s own ruling for the
    // cascade case, met again.
    if (nf < 0.0) {
        return f;
    }
    return mix(f, nf, w);
}

// The analytic (point / spot) receiver term, for a lit pass's light loop.
// `slot` is `GpuLight.params.w`, which is 0 on every light without a tree — so
// this is exactly `1.0` on every scene that has no virtual shadows, and every
// pre-P27.4 golden runs the identical arithmetic.
// **The shadow-page residency ramp** (P27.5) — the VT heat-map's twin, one
// virtual system over, driven by `view.flags.w` and `ViewMode::VsmPages`.
//
// It answers the question a shadow author actually has and a lit frame cannot
// show: *is this pixel's shadow the resolution it asked for, or is it reading a
// coarser ancestor because the page it wanted did not fit?* A blurry shadow and
// a correctly-coarse one look identical.
//
//   grey    no shadow tree reaches this pixel — no sun, outside the clipmap, or
//           the light was refused at registration (P27.5's ceiling, made
//           visible)
//   green   the page the pixel asked for is resident: full resolution
//   yellow  one level behind
//   orange  two
//   red     three or more
//   blue    NO DATA — no resident ancestor, so the receiver reads LIT. The
//           phase's chosen fail direction, and the one state an author must be
//           able to see, because a missing shadow looks exactly like a surface
//           nothing shadows.
//
// **It is a PAGE view and not a tap view**, deliberately: it re-derives the
// address from the undisplaced world position, with no normal offset and no
// kernel, because what it is about is which page a pixel lands in rather than
// which texels a filter reads. `vsm_shadow` is the sampling door and this is
// not a second one — it reads the same table and the same projection and it
// never returns a shadow factor.
fn vsm_heat(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let grey = vec3<f32>(0.06, 0.06, 0.07);
    let slot = vsm.counts.x;
    if (!vsm_bound(slot)) {
        return grey;
    }
    var pi = slot - 1u;
    var p = vsm_proj[pi];
    let ortho = p.info.w == VSM_PROJ_ORTHO_KIND;
    let faces = vsm_table[p.info.x + 3u] >> 8u;
    if (faces == 6u) {
        let f = vsm_cube_face(world_pos - p.light.xyz);
        if (pi + f >= vsm.counts.y) {
            return grey;
        }
        pi = pi + f;
        p = vsm_proj[pi];
    }
    let levels = vsm_table[p.info.x];
    if (levels == 0u) {
        return grey;
    }
    let clip = p.view_proj * vec4<f32>(world_pos, 1.0);
    if (clip.w <= 0.0) {
        return grey;
    }
    let l = clip.xyz / clip.w;
    if (l.z <= 0.0 || l.z > 1.0) {
        return grey;
    }
    let view_dist = max(length(world_pos - view.eye.xyz), 1e-4);
    let pixel_world = view_dist / max(vsm.params.y, 1e-6);
    var texel0 = p.light.w;
    if (!ortho) {
        texel0 = texel0 * max(length(world_pos - p.light.xyz), 1e-4);
    }
    let level = vsm_receiver_level(p, levels, l, texel0, pixel_world);
    if (level >= levels) {
        return grey;
    }
    let q = vsm_level_ndc(p, l, level);
    if (abs(q.x) > 1.0 || abs(q.y) > 1.0) {
        return grey;
    }
    let b = p.info.x;
    let rec = b + VSM_LIGHT_HEADER_WORDS + level * VSM_LEVEL_REC_WORDS;
    let pages_x = vsm_table[rec];
    let pages_y = vsm_table[rec + 1u];
    let first = vsm_table[rec + 2u];
    let face_stride = vsm_table[rec + 3u];
    let u = clamp(q.x * 0.5 + 0.5, 0.0, 0.999999);
    let v = clamp(0.5 - q.y * 0.5, 0.0, 0.999999);
    let px = min(u32(u * f32(pages_x)), pages_x - 1u);
    let py = min(u32(v * f32(pages_y)), pages_y - 1u);
    let entry = vsm_table[first + p.info.z * face_stride + py * pages_x + px];
    if (entry == VSM_ENTRY_NONE) {
        return vec3<f32>(0.10, 0.25, 0.95);
    }
    let got = (entry >> 16u) & 0xFFu;
    // `got` is never finer than the level asked for (an entry names an
    // ANCESTOR), so this subtraction cannot wrap.
    let behind = got - level;
    if (behind == 0u) { return vec3<f32>(0.10, 0.85, 0.20); }
    if (behind == 1u) { return vec3<f32>(0.95, 0.85, 0.10); }
    if (behind == 2u) { return vec3<f32>(1.00, 0.45, 0.05); }
    return vec3<f32>(0.95, 0.08, 0.08);
}

fn vsm_light_shadow(world_pos: vec3<f32>, n: vec3<f32>, slot: f32) -> f32 {
    return vsm_shadow(world_pos, n, u32(max(slot, 0.0)));
}
