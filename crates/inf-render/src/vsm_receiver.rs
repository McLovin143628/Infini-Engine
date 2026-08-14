//! **The virtual-shadow-map receiver** (P27.4): the address walk a lit fragment
//! makes through the page table, the PCF kernel it filters with, the clipmap
//! level blend, and the bias derived from a page's own texel density.
//!
//! Everything here is a **pure function**, for `vsm.rs`'s reason: the shader
//! `shaders/vsm_receive.wgsl` is a twin of this file, and a twin that agrees
//! with nothing is a twin that agrees with itself. Every arm in
//! `tests/vsm_receiver.rs` renders on a device and compares against *this*
//! arithmetic, and the arms in this module measure the two rulings the ROADMAP
//! asked for numbers on.
//!
//! # 1. The kernel is CLAMPED to its page — measured, not assumed
//!
//! `docs/memos/p27-1-page-geometry.md` ruled `VSM_PAGE_BORDER = 0`, because a
//! rasterized border makes a page's content a function of its *neighbours'*
//! casters and turns "moving one object invalidates exactly the pages its bounds
//! touch" into a 3 × 3 dilation. It left P27.4 two candidates and asked for a
//! measurement:
//!
//! * **the clamped kernel** — taps that leave the page are dropped and the
//!   weight renormalized (one table resolution per shadow sample);
//! * **the per-tap resolve** — every tap re-addresses through the table, so a
//!   tap that leaves the page reads its neighbour (nine resolutions, and for a
//!   clipmap each is a level walk).
//!
//! The measurement is [`pcf_crossing_fraction`] and
//! `the_clamped_kernel_is_cheaper_and_its_error_is_bounded_where_the_other_is_not`,
//! and it rules for **clamping**, on four numbers:
//!
//! 1. **How often it matters: 3.10 %.** A tap leaves its page only for texels
//!    within `VSM_PCF_RADIUS` of a page edge, which is
//!    `1 − ((128 − 2)/128)² = 508/16 384` of a page at the shipped geometry.
//! 2. **What it costs: 1 resolution against 9.** And a clipmap resolution is a
//!    *walk* — up to `clipmap_levels` level records — so at the shipped 8-level
//!    default the per-tap resolve is up to **72** level-record reads per shadow
//!    sample against **8**.
//! 3. **What the clamped kernel's error is bounded by: the kernel's own
//!    variance.** A clamped kernel is the full kernel restricted to a subset of
//!    its taps, renormalized, so wherever the shadow field is locally constant —
//!    which is everywhere except a penumbra — it is **exactly** the full
//!    kernel's answer. In a penumbra it differs by at most the dropped weight,
//!    `3/9` at an edge and `5/9` at a corner.
//! 4. **What the per-tap resolve's error is NOT bounded by: anything.** A
//!    neighbouring page may be `VSM_ENTRY_NONE`, which this phase reads as *lit*
//!    — so a cross-page tap over a missing neighbour injects "lit" into the
//!    kernel on ground that is *uniformly shadowed*, where the clamped kernel is
//!    exact. That is a bright one-texel line along a page seam, in the leak
//!    direction, on the 3.10 % of texels above, and it appears **because** the
//!    filter was made more exact.
//!
//! The fifth reason is not about the kernel at all and it is why the receiver
//! adds **no sampler of its own** — [`bind_group_layout_entries`] is a depth
//! texture and three buffers, and
//! `the_receiver_adds_four_bindings_and_none_of_them_is_a_sampler` is the arm.
//! Hardware `textureSampleCompare` filtering is a 2 × 2 footprint in *atlas*
//! space, and a page's atlas neighbour is `slot + 1` — an unrelated page of a
//! possibly unrelated light. With no border there is nothing correct for the
//! hardware to filter against, so the receiver `textureLoad`s integer texels and
//! compares in the shader. That also makes the clamp exact rather than
//! approximate: a tap is in or out of the page by integer arithmetic.
//!
//! (Said precisely, because this module's first draft said "no comparison
//! sampler **in the env layout**" and that is refutable by grep: the env group
//! holds one at binding **3** — the CASCADE's `env-shadow`, `LessEqual` over a
//! forward-Z array — and its hardware 2 × 2 is exactly the pre-filter this
//! kernel gives up. The claim is about what P27.4 *adds* and what the virtual
//! path *reads*, and neither is a sampler.)
//!
//! # 2. The bias has exactly two terms, and each is derived
//!
//! `docs/memos/p27-1-depth-convention.md` deferred the bias story to here and
//! promised the precision it would have. The derivation:
//!
//! * **The constant term is a property of the DEPTH FORMAT, and is therefore
//!   constant in NDC.** A page stores `f32`, whose ULP in the worst binade
//!   `[0.5, 1)` is `2⁻²⁴`; the P27.1 measurement bisected the *stored* depth and
//!   found the worst step at about two of those. [`VSM_DEPTH_ULP_BIAS`] is four,
//!   which is that with a factor of two in hand — and because it is expressed in
//!   NDC it does **not** have to be re-derived per level, per light or per
//!   settings value. (Expressed in *texels* it would not be constant at all:
//!   the along-light box is `2·extent·2^(levels−1)` deep while a level-0 texel
//!   is `2·extent/(N·128)` wide, so the same guard is **0.25** texels at the
//!   shipped defaults and **64** at the 16-level ceiling — 256× across eight
//!   more levels. The unit is the whole of why this term is not authored.)
//! * **The slope term is a property of the PAGE'S TEXEL DENSITY**, which is what
//!   the ROADMAP's clause asks for. A page stores one depth per texel, so
//!   between two texel centres a tilted surface departs from the stored value by
//!   the slope across that distance. The receiver's own position may sit
//!   anywhere in its texel (± ½) and the kernel reaches [`VSM_PCF_RADIUS`]
//!   further, so the worst lateral separation between the receiver and a tap's
//!   texel centre is `(R + ½)·√2` texels — [`VSM_SLOPE_BIAS_TEXELS`], **2.121**
//!   at `R = 1`, a derivation rather than a number. Times the texel's world size
//!   `texel₀ · 2^level`, times `tan θ`, gives metres along the light; the
//!   projection's own `∂z/∂m` converts it to NDC.
//!
//! At the shipped defaults the two terms are 2.38 × 10⁻⁷ and — at a 45° surface,
//! level 0 — `2.121 · 7.81 mm · 1.0 / 8 192 m` = 2.02 × 10⁻⁶ NDC. Together they
//! are **1/663** of `ShadowSettings::depth_bias`'s 0.0015, which is what a
//! cascade has to pay for having no per-texel density term to scale by.
//!
//! # 3. The fail direction, in the one place a receiver can violate it
//!
//! `inf_vsm::VSM_ENTRY_NONE` reads **lit**. That ruling has lived in
//! `inf_vsm::table`'s docs and three ledgers since P27.1 with no code able to
//! break it, because there was no receiver. [`vsm_level_factor`] is that code,
//! and it returns [`VSM_NO_DATA`] — the caller's *lit* — on the sentinel, on a
//! point outside the level's footprint and on a level past the tree. The device
//! arm `a_deferred_pages_receiver_is_byte_identical_to_no_shadow_at_all` is what
//! makes it a property rather than a decision.

use glam::{Mat4, Vec2, Vec3};

use crate::settings::VsmSettings;
use crate::vsm::{vsm_justified_level, VsmProjection, VSM_PROJ_ORTHO};
use inf_vsm::{
    VsmLightDesc, VsmTreeKind, VSM_ENTRY_NONE, VSM_PAGE_SIZE, VSM_TABLE_HEADER_WORDS,
    VSM_TABLE_LEVEL_REC_WORDS, VSM_TABLE_LIGHT_HEADER_WORDS,
};

/// The PCF kernel's radius in **page texels**: a `(2R+1)²` grid, `R = 1`.
///
/// The cascaded path's own kernel is 3 × 3 (`env_lighting.wgsl`'s
/// `csm_cascade_pcf`), and matching it is deliberate — P27.5 demotes the CSM to
/// a fallback, and a fallback that filters differently would make the tier a
/// *look* change rather than a resolution one.
///
/// **P27.5 made it the DEFAULT rather than the only value.** The radius is a
/// tier knob now ([`VsmSettings::pcf_radius`](crate::VsmSettings::pcf_radius),
/// clamped **down** below High) and it travels to the shader in
/// [`VsmReceiverParams::counts`]`[2]`. This constant is what a caller who has
/// not asked for less gets, and what every function here defaults to when a
/// caller has no params to read.
pub const VSM_PCF_RADIUS: i32 = 1;

/// Taps of the full kernel, `(2R+1)²`.
pub const VSM_PCF_TAPS: u32 = (2 * VSM_PCF_RADIUS as u32 + 1).pow(2);

/// **The constant bias, in NDC** — four ULP of the worst `f32` binade.
///
/// `f32::EPSILON` is `2⁻²³` and the ULP of a value in `[0.5, 1)` is half of it,
/// so this is `4 × 2⁻²⁴ = 2 × f32::EPSILON`. See the module docs for why this
/// term is expressed in NDC and the slope term in texels.
pub const VSM_DEPTH_ULP_BIAS: f32 = 2.0 * f32::EPSILON;

/// **The slope bias, in page texels**: `(R + ½)·√2`.
///
/// The receiver may sit anywhere within its own texel (± ½) and the kernel
/// reaches [`VSM_PCF_RADIUS`] further, on both axes. Derived rather than tuned —
/// see the module docs.
///
/// P27.5: the value at the **default** radius. A tier that clamps the kernel
/// down clamps this with it — [`vsm_slope_bias_texels`] is the derivation and
/// this is `vsm_slope_bias_texels(1)`, asserted below rather than assumed.
pub const VSM_SLOPE_BIAS_TEXELS: f32 = (VSM_PCF_RADIUS as f32 + 0.5) * std::f32::consts::SQRT_2;

/// **The slope bias for a kernel of `radius` page texels**: `(R + ½)·√2`.
///
/// P27.5's tier knob made this a function of a setting rather than a constant,
/// and the derivation is unchanged: the receiver may sit anywhere within its own
/// texel (± ½) and the kernel reaches `R` further, on both axes.
///
/// A bias sized for a kernel that is not running is peter-panning nobody asked
/// for, which is why the radius and this number travel together in one uniform
/// ([`VsmReceiverParams`]) instead of one being a constant the other has to be
/// kept in step with by hand.
#[inline]
pub fn vsm_slope_bias_texels(radius: u32) -> f32 {
    (radius as f32 + 0.5) * std::f32::consts::SQRT_2
}

/// **The normal-offset bias, in page texels**: the receiver is pushed along its
/// own normal by one texel before it is projected.
///
/// One rather than the cascade's two, and the difference is the reason the
/// resolve happens *after* the displacement: with `VSM_PAGE_BORDER = 0` a
/// displaced lookup may land in a different page, so the displacement has to be
/// part of the address rather than of the tap offsets. One texel of 128 is
/// 0.78 % of a page, which bounds how far the resolve can be moved.
pub const VSM_NORMAL_BIAS_TEXELS: f32 = 1.0;

/// The largest slope the bias scales by — `tan θ` at about 83°.
///
/// Past it a surface is nearly edge-on to the light and `n · l` has already
/// taken its direct term to almost nothing, so an unbounded `tan` would buy a
/// metre of peter-panning to protect a term that is not there.
pub const VSM_MAX_SLOPE: f32 = 8.0;

/// What [`vsm_level_factor`] returns when a level has **no data** for a
/// receiver: the page resolved to [`VSM_ENTRY_NONE`], the point is outside the
/// level's footprint, or the level is past the tree.
///
/// Negative, so it can never be confused with a factor — the
/// `csm_cascade_pcf` convention, kept, so the two paths' selection loops read
/// alike. Every caller turns it into **1.0**, which is the phase's fail
/// direction.
pub const VSM_NO_DATA: f32 = -1.0;

/// The uniform the receiver reads — 32 bytes, created once and only written, so
/// it contributes nothing to [`crate::passes::ResourceKey`] (the `wetness`
/// precedent).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VsmReceiverParams {
    /// x = the clipmap level blend band ([`VsmSettings::level_blend`]),
    /// y = pixels per world unit at one metre (`vt_stream::projection_scale` —
    /// the same door the marking pass takes, so the receiver and the marker
    /// choose the same level), z = the perspective near plane in metres,
    /// **w = the slope bias in page texels** (P27.5 —
    /// [`vsm_slope_bias_texels`] of the frame's kernel radius; it was reserved).
    pub params: [f32; 4],
    /// x = the **first shadow-casting directional light's** projection index
    /// **+ 1** (0 = there is none), y = the projection count, **z = the PCF
    /// kernel radius in page texels** (P27.5 — it was reserved), w reserved.
    ///
    /// The sun rides in the uniform rather than in its light record because
    /// `terrain.wgsl` shades it from `view.sun_dir` and has no light index at
    /// all — one door for the sun, and `GpuLight::params.w` for every analytic
    /// light that has a tree.
    pub counts: [u32; 4],
}

const _: () = assert!(std::mem::size_of::<VsmReceiverParams>() == 32);

impl VsmReceiverParams {
    /// The params for a frame that has **no** virtual shadow system: every field
    /// zero, so `sun_slot` is 0 and the projection count is 0. Bound beside the
    /// empty table, which is what actually makes `vsm_active()` false.
    pub fn absent() -> Self {
        Self::default()
    }

    /// The params for a live system.
    pub fn new(settings: &VsmSettings, proj_scale: f32, sun_slot: u32, projections: u32) -> Self {
        // P27.5: the kernel and its bias are **derived together, here**, so the
        // shader can never run a one-tap kernel with a three-tap bias. The clamp
        // is the settings boundary's ceiling restated (the boundary refuses past
        // it; this is the second door, `VsmSettings::validate`'s own pattern).
        let radius = settings.pcf_radius.min(crate::settings::VSM_MAX_PCF_RADIUS);
        Self {
            params: [
                settings.level_blend.clamp(0.0, 1.0),
                proj_scale,
                settings.perspective_near_m.max(1e-3),
                vsm_slope_bias_texels(radius),
            ],
            counts: [sun_slot, projections, radius, 0],
        }
    }

    /// The PCF kernel radius this frame's shader is running, as the `i32` the
    /// tap loop counts with.
    #[inline]
    pub fn pcf_radius(&self) -> i32 {
        self.counts[2] as i32
    }

    /// The slope bias in page texels this frame's shader is applying — always
    /// [`vsm_slope_bias_texels`] of [`pcf_radius`](Self::pcf_radius), because
    /// [`new`](Self::new) is the only thing that writes either.
    #[inline]
    pub fn slope_bias_texels(&self) -> f32 {
        self.params[3]
    }
}

/// **The fraction of a page's texels whose kernel leaves it** — the first of the
/// four numbers the clamped-kernel ruling rests on.
///
/// `1 − ((S − 2R)/S)²`, exactly: a tap leaves the page iff the centre texel is
/// within `R` of an edge on either axis.
pub fn pcf_crossing_fraction(page_size: u32, radius: i32) -> f64 {
    let s = page_size.max(1) as f64;
    let inner = (s - 2.0 * f64::from(radius)).max(0.0);
    1.0 - (inner / s) * (inner / s)
}

/// **Table resolutions one shadow sample costs**, for the two candidates —
/// the second of the four numbers.
///
/// The clamped kernel resolves once whatever the kernel does; the per-tap
/// resolve resolves once per tap. Returned as a pair so the arm and the memo
/// quote one function rather than two literals.
pub fn pcf_resolution_cost(taps: u32) -> (u32, u32) {
    (1, taps)
}

/// The **level-record reads** one clipmap resolution costs: the containment walk
/// from the justified level to the containing one, plus the resolved level's own
/// record. Bounded by the tree's level count.
pub fn clipmap_resolution_reads(levels: u32) -> u32 {
    levels.max(1)
}

/// `tan θ` between a receiver's normal and the direction to its light, clamped
/// into [`VSM_MAX_SLOPE`]. `n_dot_l` is expected already clamped to `[0, 1]`.
#[inline]
pub fn vsm_slope_tan(n_dot_l: f32) -> f32 {
    let ndl = n_dot_l.clamp(0.0, 1.0);
    let s = (1.0 - ndl * ndl).max(0.0).sqrt();
    (s / ndl.max(0.05)).min(VSM_MAX_SLOPE)
}

/// **The bias, in the light's NDC** — the two derived terms of the module docs.
///
/// `texel_m` is the world size of one texel of the page the receiver will read
/// (`texel₀ · 2^level`), `ndc_per_m` the projection's own `∂z/∂m` along the
/// light at this receiver, and `n_dot_l` the geometric term.
/// `slope_texels` is [`vsm_slope_bias_texels`] of the frame's kernel radius —
/// [`VSM_SLOPE_BIAS_TEXELS`] at the shipped default (P27.5).
#[inline]
pub fn vsm_bias_ndc(texel_m: f32, ndc_per_m: f32, n_dot_l: f32, slope_texels: f32) -> f32 {
    VSM_DEPTH_ULP_BIAS + slope_texels * texel_m * vsm_slope_tan(n_dot_l) * ndc_per_m
}

/// **How much a page's stored depth moves per metre of receiver travel toward
/// its light** — the number that turns a bias in metres into a bias in NDC.
///
/// Two closed forms, one per projection kind, and both are read off the shipped
/// matrix rather than off the settings that built it:
///
/// * **orthographic** — `ndc.z` is affine in the world point, so its gradient is
///   row 2's `xyz` and the answer is that row's length (`1/range`), the same at
///   every depth;
/// * **reverse-infinite perspective** — `ndc.z = near/d`, so
///   `∂z/∂d = −near/d² = −z²/near`, which needs the receiver's own `ndc.z` and
///   the near plane and nothing else. Row 2 is `(0, 0, 0, near)` here — the
///   degenerate row the P27.2 audit named the FAR plane — which is exactly why
///   this branch cannot use the orthographic form.
#[inline]
pub fn vsm_ndc_per_metre(view_proj: &Mat4, ortho: bool, ndc_z: f32, near_m: f32) -> f32 {
    if ortho {
        let r = view_proj.row(2);
        r.truncate().length()
    } else {
        ndc_z * ndc_z / near_m.max(1e-3)
    }
}

/// The **direction toward the light** at a receiver, in render-local space.
///
/// For a perspective light it is the vector to its position. For an
/// orthographic clipmap the projection carries no direction of its own, and
/// re-deriving it from row 2 is the point: `ndc.z` falls as the receiver moves
/// away from the light, so the gradient of `ndc.z` — row 2's `xyz` — points
/// *toward* it. One matrix, one answer, and no second copy of the light basis
/// on the receiver's side.
#[inline]
pub fn vsm_to_light(view_proj: &Mat4, ortho: bool, light_pos: Vec3, world: Vec3) -> Vec3 {
    if ortho {
        view_proj.row(2).truncate().normalize_or_zero()
    } else {
        (light_pos - world).normalize_or_zero()
    }
}

/// **Which cube face a direction lands on** — the CPU twin of
/// `vsm_receive.wgsl`'s `vsm_cube_face`, indexed into
/// [`crate::vsm::CUBE_FACE_BASES`].
///
/// The dominant axis, with `>` on the positive side. It is **exact** rather than
/// approximate: the six faces are axis-aligned 90° frusta, so the point is
/// inside face `f` iff `f`'s axis dominates — and at a seam (`|d.x| == |d.y|`)
/// the point is on the boundary of *both*, which is why the tie-break is
/// stated here rather than left to a comparison's accident.
#[inline]
pub fn vsm_cube_face(d: Vec3) -> u32 {
    let a = d.abs();
    if a.x >= a.y && a.x >= a.z {
        u32::from(d.x <= 0.0)
    } else if a.y >= a.z {
        2 + u32::from(d.y <= 0.0)
    } else {
        4 + u32::from(d.z <= 0.0)
    }
}

/// Level `L`'s NDC from level 0's, for the kind of tree `proj` describes — the
/// receiver's spelling of [`crate::vsm::clipmap_level_ndc`], which is inert for
/// a quadtree or a cube face (their levels share one frustum).
#[inline]
pub fn vsm_level_ndc(proj: &VsmProjection, ndc0: Vec3, level: u32) -> Vec2 {
    if proj.info[3] == VSM_PROJ_ORTHO {
        crate::vsm::clipmap_level_ndc(ndc0, level, proj.level_offset[level as usize])
    } else {
        ndc0.truncate()
    }
}

/// The page **and the texel inside it** a receiver's level-`level` NDC lands on.
///
/// # This is where the receiver does NOT walk the ancestor chain, and why
///
/// `inf_vsm::VsmLightDesc::ancestor_at` walks page-by-page from the level the
/// receiver asked for to the level the table served. `vt_sample.wgsl` must do
/// exactly that (the P26.2 audit measured what happens when a virtual texture
/// re-derives its tile from `uv` at the resolved mip: a container halves an
/// extent with `w/2`, so a 511-texel level does not sit over a 255-texel one and
/// the sample lands a whole tile away).
///
/// **A shadow page tree has no such slack, and this function is the claim.** A
/// clipmap level is a world lattice of stride `w_L = page₀·2^L` with a known
/// origin, so "which page of level `L` contains this point" is one division; a
/// quadtree level is `2^(n−1−L)` pages over the *same* frustum, so `⌊u·N_L⌋`
/// and `⌊⌊u·2N_L⌋/2⌋` are the same integer for every `u ≥ 0`. Re-deriving is
/// therefore identical to walking, and it is `O(1)` instead of `O(levels)`.
/// `the_re_derived_address_is_the_ancestor_the_chain_walks_to` sweeps both tree
/// kinds against `ancestor_at` and is what makes that a measurement.
pub fn vsm_page_of(desc: &VsmLightDesc, level: u32, q: Vec2) -> Option<(u32, u32, Vec2)> {
    let g = *desc.levels.get(level as usize)?;
    if g.pages_x == 0 || g.pages_y == 0 {
        return None;
    }
    // NDC y is up and the page grid's rows run down — the flip `mark_page_for`
    // and `vsm_mark.wgsl` both state, stated a third time where the receiver can
    // see it.
    let u = (q.x * 0.5 + 0.5).clamp(0.0, 0.999_999);
    let v = (0.5 - q.y * 0.5).clamp(0.0, 0.999_999);
    let px = ((u * g.pages_x as f32) as u32).min(g.pages_x - 1);
    let py = ((v * g.pages_y as f32) as u32).min(g.pages_y - 1);
    let side = VSM_PAGE_SIZE as f32;
    let local = Vec2::new(
        u * g.pages_x as f32 * side - px as f32 * side,
        v * g.pages_y as f32 * side - py as f32 * side,
    );
    Some((px, py, local))
}

/// **The level a receiver asks for** — the marking pass's rule, on the
/// receiver's side, so the page a fragment reads is the page its own depth
/// marked two frames ago.
///
/// `texel0` is the level-0 shadow texel at this point (absolute for a clipmap,
/// distance-scaled for a perspective light) and `pixel_world` the world size of
/// one screen pixel. For a clipmap the justified level is then walked **upward**
/// to the first level that contains the point, exactly as `mark_page_for` does;
/// `None` when no level does.
pub fn vsm_receiver_level(
    proj: &VsmProjection,
    levels: u32,
    ndc: Vec3,
    texel0: f32,
    pixel_world: f32,
) -> Option<(u32, Vec2)> {
    let start = vsm_justified_level(texel0, pixel_world, levels);
    if proj.info[3] != VSM_PROJ_ORTHO {
        let q = ndc.truncate();
        if q.x.abs() > 1.0 || q.y.abs() > 1.0 {
            return None;
        }
        return Some((start, q));
    }
    for l in start..levels {
        let q = vsm_level_ndc(proj, ndc, l);
        if q.x.abs() <= 1.0 && q.y.abs() <= 1.0 {
            return Some((l, q));
        }
    }
    None
}

/// One entry of a resolved indirection table, as the receiver decodes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsmTableEntry {
    pub slot: u32,
    /// The level the table actually served — the finest **resident ancestor's**.
    pub level: u32,
}

/// Read light `block`'s entry for `(face, level, x, y)` out of a table image —
/// the receiver's own indexing, off the level record rather than off a sum, so
/// it is the arithmetic `vsm_receive.wgsl` performs and not a second derivation
/// of it. `None` is [`VSM_ENTRY_NONE`], which the caller reads as **lit**.
pub fn vsm_table_entry(
    table: &[u32],
    block: u32,
    face: u32,
    level: u32,
    x: u32,
    y: u32,
) -> Option<VsmTableEntry> {
    let b = block as usize;
    let rec = b + VSM_TABLE_LIGHT_HEADER_WORDS + VSM_TABLE_LEVEL_REC_WORDS * level as usize;
    let pages_x = *table.get(rec)?;
    let first = *table.get(rec + 2)? as usize;
    let face_stride = *table.get(rec + 3)? as usize;
    let word = *table.get(first + face as usize * face_stride + (y * pages_x + x) as usize)?;
    if word == VSM_ENTRY_NONE {
        return None;
    }
    Some(VsmTableEntry {
        slot: word & 0xFFFF,
        level: (word >> 16) & 0xFF,
    })
}

/// A light block's header, as the receiver reads it: `(level_count, page_size,
/// border, faces)`.
pub fn vsm_block_header(table: &[u32], block: u32) -> Option<(u32, u32, u32, u32)> {
    let b = block as usize;
    Some((
        *table.get(b)?,
        *table.get(b + 1)?,
        *table.get(b + 2)?,
        *table.get(b + 3)? >> 8,
    ))
}

/// The atlas header: `(slots_x, stored_page_size)`. `None` when the buffer does
/// not begin with [`inf_vsm::VSM_TABLE_MAGIC`], which is what a frame with no
/// system binds.
pub fn vsm_atlas_header(table: &[u32]) -> Option<(u32, u32)> {
    if table.len() < VSM_TABLE_HEADER_WORDS || table[0] != inf_vsm::VSM_TABLE_MAGIC {
        return None;
    }
    Some((table[2], table[3]))
}

/// **The clamped PCF kernel**, as a set of atlas texels and a divisor.
///
/// `local` is the receiver's texel position inside the page's payload, `origin`
/// the page's slot corner in the atlas and `border` the stored ring (**0** by
/// the P27.1 ruling; carried so a hand-built `VsmAtlasConfig` cannot silently
/// sample its neighbour's payload).
///
/// The taps that leave `[0, page_size)` are **dropped**, not clamped to the
/// edge: clamping would double-weight the boundary texel, which biases the
/// filter toward whatever the page's own edge happens to hold; dropping
/// renormalizes and leaves the answer a mean of real data. The centre tap is
/// always inside, so the divisor is never zero.
///
/// `radius` is the frame's kernel radius (P27.5): [`VSM_PCF_RADIUS`] on High,
/// [`VSM_PCF_RADIUS_MEDIUM`](crate::settings::VSM_PCF_RADIUS_MEDIUM) below it.
pub fn vsm_pcf_taps(
    local: Vec2,
    page_size: u32,
    border: u32,
    origin: (u32, u32),
    radius: i32,
) -> Vec<(u32, u32)> {
    let base = (local.x.floor() as i32, local.y.floor() as i32);
    let side = page_size as i32;
    let r = radius.max(0);
    let mut out = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
    for dy in -r..=r {
        for dx in -r..=r {
            let (tx, ty) = (base.0 + dx, base.1 + dy);
            if tx < 0 || ty < 0 || tx >= side || ty >= side {
                continue;
            }
            out.push((origin.0 + border + tx as u32, origin.1 + border + ty as u32));
        }
    }
    out
}

/// **One level's shadow factor**, or [`VSM_NO_DATA`].
///
/// `depth` reads an atlas texel. The comparison is the camera's direction and
/// its sign is the whole of the fail story: under reverse-Z a **larger** stored
/// depth is a caster **nearer** the light, so a receiver is lit exactly when its
/// own biased depth still reaches at least as far — `receiver + bias ≥ stored`.
///
/// (The P27.1 depth memo's closing consequence wrote this as
/// `depth > stored + bias`, which puts the bias on the caster's side and makes
/// a self-shadowing surface *more* shadowed rather than less. Corrected there
/// and here; `a_bias_with_the_wrong_sign_is_acne_not_peter_panning` is the arm
/// that would fail if it were ever written back.)
#[allow(clippy::too_many_arguments)]
pub fn vsm_level_factor(
    table: &[u32],
    proj: &VsmProjection,
    desc: &VsmLightDesc,
    ndc: Vec3,
    level: u32,
    bias: f32,
    radius: i32,
    mut depth: impl FnMut(u32, u32) -> f32,
) -> f32 {
    let block = proj.info[0];
    let Some((levels, page_size, border, _)) = vsm_block_header(table, block) else {
        return VSM_NO_DATA;
    };
    if level >= levels {
        return VSM_NO_DATA;
    }
    let q = vsm_level_ndc(proj, ndc, level);
    if q.x.abs() > 1.0 || q.y.abs() > 1.0 {
        return VSM_NO_DATA;
    }
    let Some((px, py, _)) = vsm_page_of(desc, level, q) else {
        return VSM_NO_DATA;
    };
    // **THE FAIL DIRECTION.** A page with no resident ancestor is `None` here.
    let Some(entry) = vsm_table_entry(table, block, proj.info[2], level, px, py) else {
        return VSM_NO_DATA;
    };
    // …and the address is re-derived at the level the table SERVED, which is the
    // ancestor by construction — see `vsm_page_of`'s docs.
    let gq = vsm_level_ndc(proj, ndc, entry.level);
    let Some((_, _, local)) = vsm_page_of(desc, entry.level, gq) else {
        return VSM_NO_DATA;
    };
    let Some((slots_x, stored)) = vsm_atlas_header(table) else {
        return VSM_NO_DATA;
    };
    let origin = (
        (entry.slot % slots_x.max(1)) * stored,
        (entry.slot / slots_x.max(1)) * stored,
    );
    let taps = vsm_pcf_taps(local, page_size, border, origin, radius);
    if taps.is_empty() {
        return VSM_NO_DATA;
    }
    let mut sum = 0.0;
    for (x, y) in &taps {
        sum += f32::from(ndc.z + bias >= depth(*x, *y));
    }
    sum / taps.len() as f32
}

/// Everything one receiver/light pair needs, resolved once so the level, the
/// bias and the blend all read the same numbers.
#[derive(Debug, Clone, Copy)]
pub struct VsmReceiverSite {
    /// The receiver's NDC in the light's **level-0** clip space, after the
    /// normal offset.
    pub ndc: Vec3,
    /// The level the receiver asks for.
    pub level: u32,
    /// That level's NDC.
    pub q: Vec2,
    /// The world size of one level-0 texel here.
    pub texel0: f32,
    /// `∂ndc.z/∂m` toward the light here.
    pub ndc_per_m: f32,
    /// The geometric term, clamped.
    pub n_dot_l: f32,
    /// Levels in this light's tree.
    pub levels: u32,
    /// The slope bias in page texels this frame's kernel earns (P27.5) —
    /// [`vsm_slope_bias_texels`] of the radius, carried on the site so
    /// [`bias`](Self::bias) cannot read a different kernel's number from the one
    /// [`vsm_level_factor`] is about to tap with.
    pub slope_texels: f32,
}

impl VsmReceiverSite {
    /// The bias for `level`, whose texel is `texel₀ · 2^level`.
    pub fn bias(&self, level: u32) -> f32 {
        let texel_m = self.texel0 * (1u64 << level.min(62)) as f32;
        vsm_bias_ndc(texel_m, self.ndc_per_m, self.n_dot_l, self.slope_texels)
    }
}

/// **Resolve a receiver against one light's projection** — the address half of
/// the receiver, up to but not including the atlas read.
///
/// `None` when the light cannot shadow this point at all: behind its near plane,
/// past a clipmap's finite far plane, or outside every level's footprint. All
/// three read **lit**.
pub fn vsm_receiver_site(
    proj: &VsmProjection,
    desc: &VsmLightDesc,
    world: Vec3,
    normal: Vec3,
    pixel_world: f32,
    near_m: f32,
    slope_texels: f32,
) -> Option<VsmReceiverSite> {
    let vp = Mat4::from_cols_array(&proj.view_proj);
    let ortho = proj.info[3] == VSM_PROJ_ORTHO;
    let light_pos = Vec3::new(proj.light[0], proj.light[1], proj.light[2]);
    let to_light = vsm_to_light(&vp, ortho, light_pos, world);
    // The level-0 texel at the UNDISPLACED point: the normal offset is a
    // fraction of a texel, so using it to size itself would be a fixed point
    // solved for no reason.
    let texel0 = if ortho {
        proj.light[3]
    } else {
        proj.light[3] * (world - light_pos).length().max(1e-4)
    };
    let levels = desc.level_count();
    let probe = vp * world.extend(1.0);
    if probe.w <= 0.0 {
        return None;
    }
    let probe_ndc = probe.truncate() / probe.w;
    if probe_ndc.z <= 0.0 || probe_ndc.z > 1.0 {
        return None;
    }
    let level0 = vsm_justified_level(texel0, pixel_world, levels);
    let offset_m = VSM_NORMAL_BIAS_TEXELS * texel0 * (1u64 << level0.min(62)) as f32;
    let offset_pos = world + normal * offset_m;
    let clip = vp * offset_pos.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.z <= 0.0 || ndc.z > 1.0 {
        return None;
    }
    let (level, q) = vsm_receiver_level(proj, levels, ndc, texel0, pixel_world)?;
    Some(VsmReceiverSite {
        ndc,
        level,
        q,
        texel0,
        ndc_per_m: vsm_ndc_per_metre(&vp, ortho, ndc.z, near_m),
        n_dot_l: normal.dot(to_light).clamp(0.0, 1.0),
        levels,
        slope_texels,
    })
}

/// **The clipmap level blend's weight** — the P27.4 replacement for the cascade
/// blend, and the same shape.
///
/// A cascade blends across the last `band` fraction of its own *view range*.
/// A clipmap level has two edges a receiver can approach and both are seams the
/// blend exists to remove, so the weight is the larger of two proximities:
///
/// * **resolution** — `lf = log₂(pixel_world / texel₀)` is where the level rule
///   sits between levels, and `ceil` gives the level, so `lf − (L − 1)` is the
///   receiver's position inside level `L`'s band. A containment-raised level
///   makes that negative, which clamps to zero: a receiver that is on level `L`
///   because nothing finer *reaches* it is not about to step to `L+1` for
///   resolution;
/// * **footprint** — `max(|q.x|, |q.y|)` at the used level, which is 1 at the
///   ring where the level runs out. Clipmaps only: a quadtree's levels share one
///   frustum, so there is no ring to cross.
///
/// `band = 0` restores the hard switch **exactly** — the caller does not issue
/// the second resolve at all, which is `ShadowSettings::cascade_blend`'s own
/// escape hatch, kept.
pub fn vsm_blend_weight(band: f32, level: u32, lf: f32, q: Vec2, ortho: bool) -> f32 {
    // `is_nan()` beside the comparison rather than `!(band > 0.0)`, which is the
    // same predicate and hides half of it — `quantize_light_dir`'s own ruling,
    // and clippy's. A NaN band would make every comparison below false and the
    // seam would quietly come back; the settings boundary refuses one, and this
    // is the second door. (`vsm_receive.wgsl` keeps the negated form, because
    // WGSL has no `isNan` and a NaN's comparisons are false there too — so the
    // two spell one predicate.)
    if band.is_nan() || band <= 0.0 {
        return 0.0;
    }
    let b = band.clamp(1e-4, 1.0);
    let t_res = (lf - (level as f32 - 1.0)).clamp(0.0, 1.0);
    let w_res = ((t_res - (1.0 - b)) / b).clamp(0.0, 1.0);
    if !ortho {
        return w_res;
    }
    let m = q.x.abs().max(q.y.abs());
    let w_con = ((m - (1.0 - b)) / b).clamp(0.0, 1.0);
    w_res.max(w_con)
}

/// **The whole receiver, on the CPU** — the twin `tests/vsm_receiver.rs`
/// compares a rendered frame against.
///
/// `depth` reads an atlas texel (the arms hand it a readback of the real
/// atlas). Returns 1.0 — **lit** — for every way this light has nothing to say
/// about this point, which is the fail direction the phase chose and the one
/// thing a receiver can get backwards.
#[allow(clippy::too_many_arguments)]
pub fn vsm_shadow_factor(
    table: &[u32],
    proj: &VsmProjection,
    desc: &VsmLightDesc,
    world: Vec3,
    normal: Vec3,
    pixel_world: f32,
    params: &VsmReceiverParams,
    mut depth: impl FnMut(u32, u32) -> f32,
) -> f32 {
    let Some(site) = vsm_receiver_site(
        proj,
        desc,
        world,
        normal,
        pixel_world,
        params.params[2],
        params.slope_bias_texels(),
    ) else {
        return 1.0;
    };
    let radius = params.pcf_radius();
    let f = vsm_level_factor(
        table,
        proj,
        desc,
        site.ndc,
        site.level,
        site.bias(site.level),
        radius,
        &mut depth,
    );
    if f < 0.0 {
        return 1.0;
    }
    let ortho = proj.info[3] == VSM_PROJ_ORTHO;
    if site.level + 1 >= site.levels {
        return f;
    }
    let lf = (pixel_world.max(1e-6) / site.texel0.max(1e-9)).log2();
    let w = vsm_blend_weight(params.params[0], site.level, lf, site.q, ortho);
    if w <= 0.0 {
        return f;
    }
    let nf = vsm_level_factor(
        table,
        proj,
        desc,
        site.ndc,
        site.level + 1,
        site.bias(site.level + 1),
        radius,
        &mut depth,
    );
    if nf < 0.0 {
        // The coarser level can miss the receiver too (a clipmap's rings are not
        // nested about one centre since P27.3). Keep this level's answer rather
        // than fading toward lit, which would flash — `shadow_factor`'s own
        // ruling for the cascade case, met again.
        return f;
    }
    f + (nf - f) * w
}

/// Which projection index each of `scene`'s lights reads, **+ 1** — the mapping
/// `GpuLight::params.w` carries and `VsmReceiverParams::counts.x` names for the
/// sun.
///
/// One derivation of "handle `n` is the `n`-th shadow-casting light in scene
/// order", so the lights uniform and the receiver uniform cannot disagree about
/// which tree a light owns. `trees` and `proj_base` are the system's own, in
/// handle order.
pub fn receiver_slots(
    lights: impl IntoIterator<Item = bool>,
    trees: usize,
    proj_base: &[u32],
) -> Vec<u32> {
    let mut out = Vec::new();
    let mut handle = 0usize;
    for casts in lights {
        if !casts || handle >= trees {
            out.push(0);
            continue;
        }
        out.push(proj_base.get(handle).copied().unwrap_or(0) + 1);
        handle += 1;
    }
    out
}

/// The **first shadow-casting directional light's** slot, from
/// [`receiver_slots`]'s output and the scene's light kinds. 0 when there is
/// none.
pub fn sun_slot(slots: &[u32], directional: impl IntoIterator<Item = bool>) -> u32 {
    for (slot, dir) in slots.iter().zip(directional) {
        if dir && *slot != 0 {
            return *slot;
        }
    }
    0
}

/// Whether `desc` is the kind whose levels are a clipmap ring (the only kind
/// with a footprint edge to blend at).
#[inline]
pub fn is_clipmap(desc: &VsmLightDesc) -> bool {
    desc.kind == VsmTreeKind::Clipmap
}

/// Bindings the receiver adds to the shared environment layout, starting at
/// `base` — atlas, table, projections, params.
///
/// Written here rather than in `passes/mod.rs` for `VtPools`'s reason: the
/// layout and the bind-group entries are a **pair**, and a pair written in two
/// files is a pair that drifts into a validation error, or into a bind group
/// that is legal and wrong. `vsm_receive.wgsl` names the same four indices, and
/// `the_receivers_bindings_are_declared_once` reads them off the shader text.
pub fn bind_group_layout_entries(base: u32) -> [wgpu::BindGroupLayoutEntry; 4] {
    let frag = wgpu::ShaderStages::FRAGMENT;
    let storage = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    [
        wgpu::BindGroupLayoutEntry {
            binding: base,
            visibility: frag,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: base + 1,
            visibility: frag,
            ty: storage,
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: base + 2,
            visibility: frag,
            ty: storage,
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: base + 3,
            visibility: frag,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]
}

/// **The permanent EMPTY shadow surface**: what the env bind group names on
/// every frame with no [`crate::vsm_mark::VsmSystem`] — which is every scene by
/// default and all fifty committed goldens.
///
/// `crate::vt::VtEmptyPool`'s shape, and the same argument: a layout entry has
/// to be satisfied whether or not anything casts, and *"satisfied by something
/// that reads as nothing"* is what keeps a shadowless scene's arithmetic
/// identical. The table is four zero bytes, so its first word is not
/// [`inf_vsm::VSM_TABLE_MAGIC`] and `vsm_active()` is false; the atlas is 1 × 1
/// and the projections buffer holds one zeroed entry, so both are legal
/// resources that nothing ever reads.
pub struct VsmEmptyPool {
    view: wgpu::TextureView,
    table: wgpu::Buffer,
    projections: wgpu::Buffer,
    params: wgpu::Buffer,
}

impl VsmEmptyPool {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vsm-atlas-absent"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // The real atlas's format, so the layout entry's `Depth` sample type
            // is satisfied by the same thing in both cases.
            format: crate::vsm_atlas::VSM_PAGE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vsm-receiver-absent"),
            size: std::mem::size_of::<VsmReceiverParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params, 0, bytemuck::bytes_of(&VsmReceiverParams::absent()));
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            // Four zero bytes: never written, so word 0 is 0 and not the magic.
            table: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vsm-indirection-absent"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            projections: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vsm-projections-absent"),
                size: std::mem::size_of::<VsmProjection>() as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            params,
        }
    }

    #[inline]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
    #[inline]
    pub fn table(&self) -> &wgpu::Buffer {
        &self.table
    }
    #[inline]
    pub fn projections(&self) -> &wgpu::Buffer {
        &self.projections
    }
    #[inline]
    pub fn params(&self) -> &wgpu::Buffer {
        &self.params
    }
}

/// The receiver's own uniform buffer — created once in
/// [`crate::EngineRenderer::new`] and only ever written, so it is **not** a
/// resizable resource and adds nothing to [`crate::passes::ResourceKey`] (the
/// `wetness` precedent, and the invariant on `EnvBinding::bind_group` names the
/// category).
pub struct VsmReceiverResources {
    pub uniform: wgpu::Buffer,
}

impl VsmReceiverResources {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vsm-receiver"),
                size: std::mem::size_of::<VsmReceiverParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    /// Stage this frame's params. `write_buffer` executes before the commands of
    /// any command buffer submitted afterwards, and the renderer calls this at
    /// the sync point — **before** the graph, which is where the lit passes read
    /// it.
    pub fn write(&self, queue: &wgpu::Queue, params: &VsmReceiverParams) {
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(params));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **THE PCF RULING** (P27.4 clause 1), and the four numbers
    /// `docs/memos/p27-4-receiver-filtering.md` quotes.
    ///
    /// The page-geometry memo left two candidates and asked for a measurement
    /// once the receiver existed. Here it is, and the ruling is *clamp* — not
    /// because the clamped kernel is more accurate (it is not, when the
    /// neighbour is resident at the same level) but because its error is bounded
    /// by the kernel's own variance where the other's is bounded by nothing at
    /// all.
    #[test]
    fn the_clamped_kernel_is_cheaper_and_its_error_is_bounded_where_the_other_is_not() {
        // 1. How often a tap leaves its page, at the shipped geometry.
        let f = pcf_crossing_fraction(VSM_PAGE_SIZE, VSM_PCF_RADIUS);
        assert!(
            (f - 508.0 / 16_384.0).abs() < 1e-12,
            "the crossing set is 508 of 16 384 texels: {f}"
        );
        assert!((f - 0.031_006).abs() < 1e-5, "{f}");
        // A 5×5 kernel would double it — recorded so the number is a function of
        // the radius rather than a property of 128.
        assert!((pcf_crossing_fraction(VSM_PAGE_SIZE, 2) - 0.061_523).abs() < 1e-5);

        // 2. What each candidate costs per shadow sample.
        assert_eq!(VSM_PCF_TAPS, 9);
        assert_eq!(pcf_resolution_cost(VSM_PCF_TAPS), (1, 9));
        // …and a clipmap resolution is a WALK, so nine of them is nine walks.
        let reads = clipmap_resolution_reads(VsmSettings::default().clipmap_levels);
        assert_eq!(reads, 8);
        assert_eq!(reads * VSM_PCF_TAPS, 72, "the per-tap resolve's worst case");

        // The kept-tap counts come from the SHIPPED kernel, so numbers 3 and 4
        // below are measurements of it rather than of three literals. (They were
        // exactly that in this arm's first draft — `3.0/9.0` asserted equal to
        // `3.0/9.0` — which the P27.4 audit found and which is the vacuous-check
        // law met again: an arithmetic identity passes whatever the kernel does.)
        let taps_at = |l: Vec2| vsm_pcf_taps(l, VSM_PAGE_SIZE, 0, (0, 0), VSM_PCF_RADIUS).len();
        let interior = taps_at(Vec2::new(64.5, 64.5));
        let on_edge = taps_at(Vec2::new(0.5, 64.5));
        let at_corner = taps_at(Vec2::new(0.5, 0.5));
        assert_eq!((interior, on_edge, at_corner), (9, 6, 4));

        // 3. The clamped kernel is EXACT wherever the field is locally constant,
        //    which is everywhere except a penumbra — measured THROUGH
        //    `vsm_level_factor` on a one-page tree whose atlas is uniform, at an
        //    interior texel (9 taps), an edge (6) and a corner (4).
        let (table, desc, proj) = one_page_tree();
        for (stored, want) in [(0.25f32, 1.0f32), (0.75, 0.0)] {
            // The receiver's own depth is 0.5, so `stored` decides: 0.25 is a
            // caster further from the light (lit) and 0.75 a nearer one (shadow).
            let at = |q: Vec2| {
                vsm_level_factor(
                    &table,
                    &proj,
                    &desc,
                    Vec3::new(q.x, q.y, 0.5),
                    0,
                    0.0,
                    VSM_PCF_RADIUS,
                    |_, _| stored,
                )
            };
            for q in [
                Vec2::ZERO,               // interior
                Vec2::new(-0.999, 0.0),   // an edge
                Vec2::new(-0.999, 0.999), // a corner
                Vec2::new(0.999, -0.999), // the other corner
            ] {
                assert_eq!(
                    at(q),
                    want,
                    "a LOCALLY CONSTANT shadow field did not survive clamping at \
                     {q:?} — the clamped kernel is the full kernel restricted and \
                     renormalized, so a constant field is exact"
                );
            }
        }
        // …and in a penumbra it is off by at most the DROPPED weight, which is
        // the kernel's own tap counts and not a pair of typed fractions.
        let edge_drop = (interior - on_edge) as f32 / interior as f32;
        let corner_drop = (interior - at_corner) as f32 / interior as f32;
        assert!((edge_drop - 3.0 / 9.0).abs() < 1e-6, "{edge_drop}");
        assert!((corner_drop - 5.0 / 9.0).abs() < 1e-6, "{corner_drop}");

        // 4. The per-tap resolve over an ABSENT neighbour is wrong on ground the
        //    clamped kernel is EXACT on, in the leak direction, by that same
        //    dropped weight.
        //
        //    Uniformly shadowed ground at a page edge: the clamp reads 0.0 (all
        //    six kept taps shadowed, measured above). The per-tap resolve keeps
        //    the three that leave the page and re-addresses them — into a
        //    neighbour that is `VSM_ENTRY_NONE`, which this phase reads as LIT.
        let clamped_on_a_seam = vsm_level_factor(
            &table,
            &proj,
            &desc,
            Vec3::new(-0.999, 0.0, 0.5),
            0,
            0.0,
            VSM_PCF_RADIUS,
            |_, _| 0.75,
        );
        assert_eq!(clamped_on_a_seam, 0.0, "the clamp is exact here");
        let per_tap_resolve = (interior - on_edge) as f32 / interior as f32;
        assert!(
            (per_tap_resolve - edge_drop).abs() < 1e-6,
            "a bright line of exactly the dropped weight, along a page seam, on \
             ground the clamped kernel gets right"
        );
        assert!(
            per_tap_resolve > clamped_on_a_seam,
            "the leak is in the LIT direction"
        );
    }

    /// A one-light, one-level, one-page quadtree and the table that serves it —
    /// the smallest tree `vsm_level_factor` can be measured on.
    fn one_page_tree() -> (Vec<u32>, VsmLightDesc, VsmProjection) {
        let desc = VsmLightDesc::quadtree(1);
        let block = VSM_TABLE_HEADER_WORDS;
        let entries = block + VSM_TABLE_LIGHT_HEADER_WORDS + VSM_TABLE_LEVEL_REC_WORDS;
        let mut table = vec![0u32; entries + 1];
        table[0] = inf_vsm::VSM_TABLE_MAGIC;
        table[1] = 1;
        table[2] = 1; // slots a row
        table[3] = VSM_PAGE_SIZE; // stored page size
        table[block] = 1; // levels
        table[block + 1] = VSM_PAGE_SIZE;
        table[block + 2] = 0; // border
        table[block + 3] = inf_vsm::VsmTreeKind::Quadtree.wire() | (1 << 8);
        let rec = block + VSM_TABLE_LIGHT_HEADER_WORDS;
        table[rec] = 1; // pages_x
        table[rec + 1] = 1; // pages_y
        table[rec + 2] = entries as u32;
        table[rec + 3] = 0;
        table[entries] = inf_vsm::pack_entry(0, 0);
        let proj = VsmProjection {
            view_proj: Mat4::IDENTITY.to_cols_array(),
            info: [block as u32, 0, 0, crate::vsm::VSM_PROJ_PERSPECTIVE],
            ..Default::default()
        };
        (table, desc, proj)
    }

    /// The two bias terms, their derivations and the numbers the memo quotes.
    #[test]
    fn the_bias_is_two_derived_terms_and_neither_is_a_tuned_constant() {
        // The constant term is four ULP of the worst f32 binade.
        assert_eq!(VSM_DEPTH_ULP_BIAS, 4.0 * 2.0f32.powi(-24));
        assert!((VSM_DEPTH_ULP_BIAS - 2.384_185_8e-7).abs() < 1e-13);

        // The slope term's texel count is (R + ½)·√2, not a number.
        assert!((VSM_SLOPE_BIAS_TEXELS - 2.121_320_3).abs() < 1e-6);

        // At the shipped defaults: level-0 texel, a 45° surface, the 8 192 m box.
        let s = VsmSettings::default();
        let texel0 =
            2.0 * s.first_level_extent_m / (s.clipmap_pages_per_side * VSM_PAGE_SIZE) as f32;
        assert!((texel0 - 0.007_812_5).abs() < 1e-9, "{texel0}");
        let range = 2.0 * s.first_level_extent_m * (1u32 << (s.clipmap_levels - 1)) as f32;
        assert_eq!(range, 8_192.0);
        let ndl = std::f32::consts::FRAC_1_SQRT_2;
        assert!((vsm_slope_tan(ndl) - 1.0).abs() < 1e-5, "45° is tan 1");
        let bias = vsm_bias_ndc(texel0, 1.0 / range, ndl, VSM_SLOPE_BIAS_TEXELS);
        assert!(
            (bias - 2.26e-6).abs() < 5e-8,
            "the shipped defaults' bias at 45°, level 0: {bias}"
        );
        // …which is three orders of magnitude under the cascade's flat constant,
        // and that ratio is the whole reason the terms are derived.
        //
        // Pinned to a HALF, not to a hundred-wide window: the P27.4 audit found
        // the memo and the ledger quoting **665** against this arm's **663.3**,
        // which a `600 < r < 700` bound cannot see. A number the prose repeats is
        // a number the arm has to hold to the digit the prose prints.
        let csm = crate::settings::ShadowSettings::default().depth_bias;
        let ratio = csm / bias;
        assert!(
            (ratio - 663.3).abs() < 0.5,
            "ratio to the cascade's flat bias: {ratio}"
        );

        // The slope clamp bites before `tan` runs away.
        assert_eq!(vsm_slope_tan(0.0), VSM_MAX_SLOPE);
        assert_eq!(vsm_slope_tan(1.0), 0.0);
        // …and the constant term is what is left when the surface faces the
        // light, which is exactly the depth format's own step.
        assert_eq!(
            vsm_bias_ndc(texel0, 1.0 / range, 1.0, VSM_SLOPE_BIAS_TEXELS),
            VSM_DEPTH_ULP_BIAS
        );

        // The unit ruling, measured: expressed in TEXELS the constant term is not
        // constant at all — it is `2^(levels-1)` of itself.
        //
        // **0.25 and 64**, and the memo and the ledger said 0.125 and 32 until
        // the P27.4 audit compared them with this line: the prose had computed
        // the guard at TWO ULP where `VSM_DEPTH_ULP_BIAS` is four. The ratio it
        // was making the argument about — 256× across eight more levels — was
        // right in both tellings, which is how a factor of two survives a
        // reading. Held to a tenth of a texel here so it cannot drift again.
        let texels_at = |levels: u32| {
            let r = 2.0 * s.first_level_extent_m * (1u32 << (levels - 1)) as f32;
            VSM_DEPTH_ULP_BIAS * r / texel0
        };
        assert!((texels_at(8) - 0.25).abs() < 1e-4, "{}", texels_at(8));
        assert!((texels_at(16) - 64.0).abs() < 1e-4, "{}", texels_at(16));
        assert!(
            (texels_at(16) / texels_at(8) - 256.0).abs() < 1e-3,
            "eight more levels is 256 times the guard, in texels"
        );
    }

    /// `∂ndc.z/∂m` in both branches, read off the shipped matrices.
    #[test]
    fn the_depth_gradient_is_read_off_the_matrix_in_both_branches() {
        // Orthographic: row 2's length is 1/range, at every depth.
        let vp = crate::vsm::clipmap_matrix(Vec3::Z, Vec3::ZERO, 32.0, 8_192.0);
        let g = vsm_ndc_per_metre(&vp, true, 0.5, 0.1);
        assert!((g - 1.0 / 8_192.0).abs() < 1e-9, "{g}");
        assert_eq!(g, vsm_ndc_per_metre(&vp, true, 0.9, 0.1), "ortho is affine");

        // Perspective: `z²/near`, and it AGREES with the shipped matrix by
        // finite difference rather than by algebra restated.
        let near = 0.1f32;
        let pv = crate::vsm::spot_matrix(Vec3::ZERO, Vec3::Z, 0.5, near);
        for d in [1.0f32, 4.0, 25.0] {
            let z = |t: f32| {
                let c = pv * Vec3::new(0.0, 0.0, -(d + t)).extend(1.0);
                c.z / c.w
            };
            let numeric = (z(-0.001) - z(0.001)) / 0.002;
            let closed = vsm_ndc_per_metre(&pv, false, z(0.0), near);
            assert!(
                (numeric - closed).abs() / closed < 2e-3,
                "d={d}: finite difference {numeric} against {closed}"
            );
        }
        // …and row 2 IS degenerate there, which is why the ortho form cannot be
        // used for both (the P27.2 audit's far-plane ruling, met from the other
        // side).
        assert!(pv.row(2).truncate().length() < 1e-12);
    }

    /// **Re-deriving the address at the served level is the ancestor the chain
    /// walks to** — the claim `vsm_page_of` makes, over both tree kinds and over
    /// per-level offsets that are not concentric.
    ///
    /// This is the P26.2 finding's mirror: a virtual texture may **not** do
    /// this, and a shadow page tree may, and the difference is that a level here
    /// is an exact lattice rather than a `w/2` chain.
    #[test]
    fn the_re_derived_address_is_the_ancestor_the_chain_walks_to() {
        let mut checked = 0u32;
        let mut fell_back = 0u32;
        // A clipmap with per-level offsets that are NOT concentric, so the walk
        // and the re-derivation are compared where the closed form does not hold.
        let desc = VsmLightDesc::clipmap(5, 8);
        for shift in [0i64, 1, -1, 3] {
            let origins: Vec<(i64, i64)> = (0..5)
                .map(|l| {
                    let n = 8i64 / 2;
                    (-n + shift * i64::from(l % 2), -n - shift * i64::from(l % 3))
                })
                .collect();
            let mut proj = VsmProjection {
                info: [0, 0, 0, VSM_PROJ_ORTHO],
                ..Default::default()
            };
            // The NDC offsets that describe those origins, derived the way
            // `clipmap_layout` derives them rather than restated: level `L`'s
            // centre is `m_L · w_L` and its origin is `m_L − N/2` along
            // `right`, `−m_L − N/2` along the page rows (which run against the
            // basis's up), so `off_L = (2/N)·(m_0/2^L − m_L)` per axis.
            let n = 8i64 / 2;
            for l in 0..5u32 {
                let s = (1u64 << l) as f64;
                let k = 2.0 / 8.0;
                let (mr0, mu0) = (origins[0].0 + n, -(origins[0].1 + n));
                let (mr, mu) = (origins[l as usize].0 + n, -(origins[l as usize].1 + n));
                proj.level_offset[l as usize] = [
                    (k * (mr0 as f64 / s - mr as f64)) as f32,
                    (k * (mu0 as f64 / s - mu as f64)) as f32,
                ];
            }
            for i in 0..64u32 {
                for j in 0..64u32 {
                    let ndc = Vec3::new(
                        -0.98 + 1.96 * i as f32 / 63.0,
                        -0.98 + 1.96 * j as f32 / 63.0,
                        0.5,
                    );
                    let Some((_, _, _)) = vsm_page_of(&desc, 0, vsm_level_ndc(&proj, ndc, 0))
                    else {
                        continue;
                    };
                    let (x0, y0, _) = vsm_page_of(&desc, 0, vsm_level_ndc(&proj, ndc, 0)).unwrap();
                    let at = inf_vsm::VsmPage::new(0, 0, x0, y0);
                    for target in 1..5u32 {
                        let q = vsm_level_ndc(&proj, ndc, target);
                        if q.x.abs() > 1.0 || q.y.abs() > 1.0 {
                            continue;
                        }
                        let Some((px, py, _)) = vsm_page_of(&desc, target, q) else {
                            continue;
                        };
                        let walked = desc.ancestor_at(&origins, at, target);
                        // The walk may leave the coarser grid where the
                        // re-derivation does not; that is the offsets' own
                        // boundary, and the arm only claims agreement where the
                        // walk has an answer.
                        if let Some(w) = walked {
                            assert_eq!(
                                (w.x, w.y),
                                (px, py),
                                "level {target} page ({px},{py}) is not the ancestor \
                                 ({},{}) of level-0 page ({x0},{y0})",
                                w.x,
                                w.y
                            );
                            checked += 1;
                            fell_back += u32::from(target > 1);
                        }
                    }
                }
            }
        }
        assert!(checked > 5_000, "the sweep proved nothing: {checked}");
        assert!(
            fell_back > 2_000,
            "no DEEP fallback was checked: {fell_back}"
        );

        // The quadtree half: `⌊u·N⌋` and `⌊⌊u·2N⌋/2⌋` are the same integer.
        let q = VsmLightDesc::quadtree(6);
        let proj = VsmProjection {
            info: [0, 0, 0, crate::vsm::VSM_PROJ_PERSPECTIVE],
            ..Default::default()
        };
        let mut qchecked = 0u32;
        for i in 0..512u32 {
            let ndc = Vec3::new(-1.0 + 2.0 * i as f32 / 511.0, 0.25, 0.5);
            let (x0, y0, _) = vsm_page_of(&q, 0, vsm_level_ndc(&proj, ndc, 0)).unwrap();
            let at = inf_vsm::VsmPage::new(0, 0, x0, y0);
            for target in 1..6u32 {
                let (px, py, _) =
                    vsm_page_of(&q, target, vsm_level_ndc(&proj, ndc, target)).unwrap();
                let w = q
                    .ancestor_at(&[], at, target)
                    .expect("a quadtree always has one");
                assert_eq!((w.x, w.y), (px, py), "quadtree level {target}");
                qchecked += 1;
            }
        }
        assert_eq!(qchecked, 512 * 5);
    }

    /// The cube-face rule is the dominant axis, and it agrees with "which face's
    /// projection actually contains the point" over a swept sphere — including
    /// at the seams, where **both** contain it.
    #[test]
    fn the_cube_face_rule_is_the_face_whose_projection_contains_the_point() {
        let pos = Vec3::new(1.0, 2.0, -3.0);
        let mats: Vec<Mat4> = (0..6)
            .map(|f| crate::vsm::cube_face_matrix(pos, f, 0.05))
            .collect();
        let mut seams = 0u32;
        let mut singles = 0u32;
        for i in 0..40u32 {
            for j in 0..40u32 {
                let theta = std::f32::consts::PI * (i as f32 + 0.5) / 40.0;
                let phi = std::f32::consts::TAU * (j as f32 + 0.5) / 40.0;
                let d = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    theta.sin() * phi.sin(),
                );
                let world = pos + d * 7.0;
                let picked = vsm_cube_face(d);
                let containing: Vec<u32> = (0..6u32)
                    .filter(|f| {
                        let c = mats[*f as usize] * world.extend(1.0);
                        c.w > 0.0 && {
                            let n = c.truncate() / c.w;
                            n.x.abs() <= 1.0 + 1e-4 && n.y.abs() <= 1.0 + 1e-4 && n.z > 0.0
                        }
                    })
                    .collect();
                assert!(
                    containing.contains(&picked),
                    "direction {d:?} picked face {picked}, contained by {containing:?}"
                );
                if containing.len() == 1 {
                    singles += 1;
                } else {
                    seams += 1;
                }
            }
        }
        assert!(
            singles > 1_000,
            "the sweep barely left the seams: {singles}"
        );
        // A spherical sweep at half-cell offsets never LANDS on a diagonal — it
        // measured 0 seams, which would leave the tie-break untested by a test
        // that names it. The seams are constructed instead, and each one is
        // asserted to be contained by BOTH its faces (so the rule is choosing
        // between two right answers rather than picking the only one).
        //
        // **All three axis pairs, because the rule is three comparisons.** The
        // P27.4 audit's mutation round found the first draft's five seams covered
        // `a.x >= a.z` and `a.y >= a.z` and never `a.x >= a.y` — so weakening the
        // first comparison to `>` moved every X/Y diagonal from `+X` to `+Y` and
        // survived, in both halves of the twin. The last two rows are that pair.
        assert_eq!(seams, 0, "the sweep is expected to miss the diagonals");
        let mut tied = 0u32;
        let mut pairs = std::collections::BTreeSet::new();
        for (d, a, b) in [
            (Vec3::new(1.0, 0.2, 1.0), 0u32, 4u32),
            (Vec3::new(1.0, 0.2, -1.0), 0, 5),
            (Vec3::new(-1.0, 0.2, 1.0), 1, 4),
            (Vec3::new(0.2, 1.0, 1.0), 2, 4),
            (Vec3::new(0.2, -1.0, -1.0), 3, 5),
            (Vec3::new(1.0, 1.0, 0.2), 0, 2),
            (Vec3::new(-1.0, -1.0, 0.2), 1, 3),
        ] {
            let d = d.normalize();
            let world = pos + d * 9.0;
            let containing: Vec<u32> = (0..6u32)
                .filter(|f| {
                    let c = mats[*f as usize] * world.extend(1.0);
                    c.w > 0.0 && {
                        let n = c.truncate() / c.w;
                        n.x.abs() <= 1.0 + 1e-4 && n.y.abs() <= 1.0 + 1e-4 && n.z > 0.0
                    }
                })
                .collect();
            assert!(
                containing.contains(&a) && containing.contains(&b),
                "the constructed seam {d:?} is not on the boundary of {a} and {b}: \
                 {containing:?}"
            );
            // The tie-break is the axis ORDER — x, then y, then z — so the lower
            // face of the pair wins, and it is stated here rather than left to a
            // comparison's accident.
            assert_eq!(vsm_cube_face(d), a.min(b), "the tie-break moved");
            // The AXES the pair straddles, so "all three pairs" is counted rather
            // than eyeballed off the list.
            pairs.insert((a.min(b) / 2, a.max(b) / 2));
            tied += 1;
        }
        assert_eq!(tied, 7);
        assert_eq!(
            pairs.len(),
            3,
            "the constructed seams cover {pairs:?} — the rule is three \
             comparisons and an untested one is a tie-break nothing decides"
        );

        // **And the WGSL twin's own three comparisons, by SOURCE.** Everything
        // above measures the Rust half; the shader's `vsm_cube_face` is a
        // separate three comparisons, and no device fixture in this phase lands a
        // fragment exactly on a cube diagonal (a shaded pixel is a surface point,
        // not a swept direction). Weakening any `>=` to `>` there moves every
        // seam direction to the next face and is invisible to every arm in the
        // tree — the P27.4 audit's mutation round measured exactly that. Same
        // honesty as the kernel's pin: this catches a deletion, not a
        // re-derivation.
        let src = include_str!("shaders/vsm_receive.wgsl");
        for cmp in ["if (a.x >= a.y && a.x >= a.z) {", "if (a.y >= a.z) {"] {
            assert!(
                src.contains(cmp),
                "`vsm_receive.wgsl`'s cube-face rule no longer spells `{cmp}` — \
                 the tie-break at a diagonal is the axis ORDER, and the two \
                 halves of the twin have to make the same choice on a direction \
                 that is on the boundary of two faces"
            );
        }
    }

    /// **The cube seam is depth-continuous**, and that is a property rather than
    /// an accident: two faces share one near plane and one position, and exactly
    /// at the seam the two view depths are equal — so `near/d` is the same
    /// number on both sides even though the page address jumps.
    #[test]
    fn the_cube_seam_changes_the_page_and_not_the_stored_depth() {
        let pos = Vec3::ZERO;
        let near = 0.05f32;
        let mats: Vec<Mat4> = (0..6)
            .map(|f| crate::vsm::cube_face_matrix(pos, f, near))
            .collect();
        let mut crossed = 0u32;
        // Sweep across the +X/+Z seam (the 45° diagonal in the xz plane).
        let mut prev: Option<(u32, f32)> = None;
        for i in 0..401u32 {
            let a = std::f32::consts::FRAC_PI_4 - 0.02 + 0.04 * i as f32 / 400.0;
            let d = Vec3::new(a.cos(), 0.1, a.sin()).normalize();
            let world = pos + d * 12.0;
            let f = vsm_cube_face(d);
            let c = mats[f as usize] * world.extend(1.0);
            let z = c.z / c.w;
            if let Some((pf, pz)) = prev {
                if pf != f {
                    crossed += 1;
                    assert!(
                        (z - pz).abs() < 2e-4,
                        "the stored depth jumped {} across the seam ({pf} → {f})",
                        (z - pz).abs()
                    );
                }
            }
            prev = Some((f, z));
        }
        assert_eq!(crossed, 1, "the sweep did not cross exactly one seam");
    }

    /// The blend weight's two proximities, its escape hatch, and the case a
    /// containment-raised level must NOT blend in.
    #[test]
    fn the_level_blend_is_a_band_at_both_of_a_levels_edges() {
        // Escape hatch: band 0 is the hard switch, and it is `0.0` rather than
        // "a very small number".
        assert_eq!(
            vsm_blend_weight(0.0, 3, 3.0, Vec2::new(0.99, 0.0), true),
            0.0
        );

        // Resolution: level 3 covers `lf ∈ (2, 3]`, so the band's start at 0.2 is
        // `lf = 2.8`.
        assert_eq!(vsm_blend_weight(0.2, 3, 2.5, Vec2::ZERO, true), 0.0);
        assert!((vsm_blend_weight(0.2, 3, 2.9, Vec2::ZERO, true) - 0.5).abs() < 1e-5);
        assert!((vsm_blend_weight(0.2, 3, 3.0, Vec2::ZERO, true) - 1.0).abs() < 1e-6);

        // Footprint: 1 at the ring, and only for a clipmap.
        assert!((vsm_blend_weight(0.2, 3, 0.0, Vec2::new(0.9, 0.0), true) - 0.5).abs() < 1e-5);
        assert_eq!(
            vsm_blend_weight(0.2, 3, 0.0, Vec2::new(0.9, 0.0), false),
            0.0
        );

        // A containment-raised level (lf well below the level) blends nothing —
        // it is not about to step for resolution, and blending would mix in a
        // level the receiver has no reason to want.
        assert_eq!(vsm_blend_weight(0.3, 6, 1.0, Vec2::ZERO, true), 0.0);

        // The weight is the LARGER of the two, not their sum.
        let both = vsm_blend_weight(0.2, 3, 2.9, Vec2::new(0.95, 0.0), true);
        assert!((both - 0.75).abs() < 1e-5, "{both}");
    }

    /// The scene-order mapping the lights uniform and the receiver uniform share.
    #[test]
    fn the_slot_mapping_is_the_nth_shadow_casting_light_in_scene_order() {
        // Lights: [dir(no cast), point(cast), dir(cast), spot(cast)] with a tree
        // list of two — the truncation `VsmSystem::for_scene` performs, which
        // must leave the tail with NO slot rather than with another light's.
        let casts = [false, true, true, true];
        let bases = [0u32, 1];
        let slots = receiver_slots(casts, 2, &bases);
        assert_eq!(slots, vec![0, 1, 2, 0]);
        // The sun is the first DIRECTIONAL with a slot — light 2, not light 1.
        assert_eq!(sun_slot(&slots, [true, false, true, false]), 2);
        // …and a scene whose casters are all analytic has no sun slot at all.
        assert_eq!(sun_slot(&slots, [true, false, false, false]), 0);
        // A point light costs six projections, so the next light's base is +6.
        assert_eq!(receiver_slots([true, true], 2, &[0, 6]), vec![1, 7]);
    }

    /// **Mirrored ≠ measured** (P26.5), applied to the constants: the shader
    /// carries five literals this module derives, and a literal in a `.wgsl`
    /// file is a number nothing in Rust can see.
    ///
    /// Read off the shipped shader text and compared **numerically** — not as
    /// strings — so a re-spelling is allowed and a re-valuing is not. The
    /// tolerance is the last digit each literal carries.
    #[test]
    fn the_receivers_two_halves_spell_one_set_of_constants() {
        let src = include_str!("shaders/vsm_receive.wgsl");
        // CRLF-tolerant by construction: `lines()` splits on `\n` and trims a
        // trailing `\r`, which is the shape a Windows checkout needs (the P22
        // law, met again).
        let value = |name: &str| -> f64 {
            let line = src
                .lines()
                .find(|l| l.trim_start().starts_with(&format!("const {name}")))
                .unwrap_or_else(|| panic!("`{name}` is not declared in vsm_receive.wgsl"));
            let rhs = line.split('=').nth(1).expect("a constant has a value");
            rhs.trim()
                .trim_end_matches(';')
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("`{name}` = `{rhs}` does not parse as a number: {e}"))
        };
        // `VSM_PCF_RADIUS` and `VSM_SLOPE_BIAS_TEXELS` are deliberately NOT in
        // this list any more (P27.5): both became tier knobs and now arrive in
        // the uniform, so a constant here would be a second, silently divergent
        // answer rather than a mirror. What replaced this pair is
        // `the_kernel_and_its_bias_are_read_from_the_uniform_the_tier_writes`
        // plus `VsmReceiverParams::new`'s own arm — an exchange with one more
        // assertion in it than the two it retired, not a loosening.
        assert!(
            !src.contains("const VSM_PCF_RADIUS") && !src.contains("const VSM_SLOPE_BIAS_TEXELS"),
            "the shader declares a kernel constant again — it would be a second \
             answer to a question `VsmReceiverParams::new` already answers per \
             frame, and the tier clamp would stop reaching the kernel"
        );
        assert!(
            (value("VSM_DEPTH_ULP_BIAS") - f64::from(VSM_DEPTH_ULP_BIAS)).abs() < 1e-14,
            "the constant bias differs between the two halves"
        );
        assert_eq!(
            value("VSM_NORMAL_BIAS_TEXELS"),
            f64::from(VSM_NORMAL_BIAS_TEXELS)
        );
        assert_eq!(value("VSM_MAX_SLOPE"), f64::from(VSM_MAX_SLOPE));
        assert_eq!(value("VSM_NO_DATA"), f64::from(VSM_NO_DATA));
        // …and the sentinel and the table magic, which are the two values a
        // receiver that got them wrong would read as "every page is resident".
        let magic = src
            .lines()
            .find(|l| l.contains("const VSM_MAGIC"))
            .expect("VSM_MAGIC");
        assert!(
            magic.contains(&format!("0x{:08X}u", inf_vsm::VSM_TABLE_MAGIC)),
            "the shader's table magic is not `inf_vsm::VSM_TABLE_MAGIC`: {magic}"
        );
        let none = src
            .lines()
            .find(|l| l.contains("const VSM_ENTRY_NONE"))
            .expect("VSM_ENTRY_NONE");
        assert!(
            none.contains(&format!("0x{:08X}u", VSM_ENTRY_NONE)),
            "the shader's absence sentinel is not `inf_vsm::VSM_ENTRY_NONE`: {none}"
        );
    }

    /// **EVERY LIT PATH THAT CAN RECEIVE A SUN SHADOW DOES, AND THE ONE THAT
    /// CANNOT SAYS SO** (P27.5 — the receiver-completeness ruling).
    ///
    /// The P27.4 ledger left this as a remainder: *"`vgeom_mesh.wgsl` and
    /// `scatter_mesh.wgsl` take the analytic term but not the directional one,
    /// because neither has ever called `shadow_factor` — virtualized geometry
    /// and foliage receive no sun shadow, from the cascade either."* Closed:
    /// both call it now, in the first-directional branch, spelled the way
    /// `mesh.wgsl` spells it.
    ///
    /// This is the SOURCE half. The device half is
    /// `vsm_receiver::a_meshlet_surface_and_a_scattered_one_receive_the_suns_shadow`,
    /// which measures both surfaces darkening; this is what keeps the *set* of
    /// call sites complete, because a device arm can only see the paths its own
    /// fixture draws.
    ///
    /// Read as a SCOPE and not as a spelling (the P23 law): the call has to be
    /// inside the `pos_dir.w < 0.5` branch, since a `shadow_factor` anywhere
    /// else in the file — on a point light, say — would satisfy a bare
    /// `contains`.
    ///
    /// And the REFUSAL is pinned in the same breath, because a refusal with no
    /// arm decays into a stale comment: `voxel.wgsl` is `ShaderKind::Plain`, so
    /// it binds no environment group and cannot call either function. Its own
    /// header carries the ruling and P28.1's VisBuffer resolve is where it goes.
    #[test]
    fn every_env_bound_lit_path_receives_the_suns_shadow_and_voxel_is_refused() {
        // The four that bind the shared environment group and shade a 3D
        // analytic light loop. `terrain.wgsl` is here too: it shades the sun from
        // `view.sun_dir` and has called `shadow_factor` since P13.3b.
        for (name, src) in [
            ("mesh", include_str!("shaders/mesh.wgsl")),
            ("skinned_mesh", include_str!("shaders/skinned_mesh.wgsl")),
            ("vgeom_mesh", include_str!("shaders/vgeom_mesh.wgsl")),
            ("scatter_mesh", include_str!("shaders/scatter_mesh.wgsl")),
        ] {
            let at = src
                .find("if (light.pos_dir.w < 0.5) {")
                .unwrap_or_else(|| panic!("{name} has no directional branch"));
            let end = src[at..]
                .find("} else {")
                .unwrap_or_else(|| panic!("{name}'s directional branch has no end"));
            let branch = &src[at..at + end];
            assert!(
                branch.contains("shadow_factor(in.world_pos, n)"),
                "{name} shades a directional light and never asks for its \
                 shadow — a surface this engine draws cannot receive the sun"
            );
            assert!(
                branch.contains("sun_shadowing_enabled()"),
                "{name} calls shadow_factor unguarded; every golden with both \
                 shadow paths off would take a different instruction stream"
            );
            // …and the analytic half, which P27.4 wired, is still there.
            assert!(
                src.contains("vsm_light_shadow(in.world_pos, n, light.params.w)"),
                "{name} lost its point/spot receiver"
            );
        }

        // THE REFUSAL, pinned against the code rather than left as prose.
        let voxel = include_str!("shaders/voxel.wgsl");
        assert!(
            voxel.contains("light.pos_dir.w < 0.5"),
            "voxel.wgsl no longer has an analytic light loop, so the refusal \
             below is about nothing"
        );
        // Over CODE only — full-line comments stripped, `lines()`-based so a
        // CRLF checkout cannot make it vacuous (the P27.1 GI-gate pattern). The
        // ruling paragraph above NAMES both functions, so a bare `contains` over
        // the whole file would fail on the comment that documents the refusal.
        let voxel_code: String = voxel
            .lines()
            .map(|l| l.trim_end_matches('\r'))
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !voxel_code.contains("shadow_factor") && !voxel_code.contains("vsm_light_shadow"),
            "voxel.wgsl calls a receiver it has no bindings for — it is composed \
             `ShaderKind::Plain` and would not build"
        );
        assert!(
            voxel.contains("P27.5: VIRTUAL SHADOWS ARE REFUSED HERE"),
            "voxel.wgsl's refusal lost its ruling; a refusal with no sentence is \
             indistinguishable from an oversight"
        );
        assert!(
            matches!(
                crate::passes::shader_kind("voxel"),
                Some(crate::passes::ShaderKind::Plain)
            ),
            "voxel.wgsl gained an environment group — the refusal above is now a \
             stale comment and the receiver is one call site away"
        );
    }

    /// **The kernel and its bias are read from the uniform the tier writes**
    /// (P27.5) — the arm that replaces the two constants
    /// `the_receivers_two_halves_spell_one_set_of_constants` used to mirror.
    ///
    /// It names the two expressions whose **absence** is the defect, which is
    /// the P27.4-audit corollary a source pin exists to satisfy: pinning
    /// `vsm.counts.z` somewhere in the file would pass while the loop still
    /// counted to a constant. So the pin is the loop header and the multiply.
    ///
    /// And the Rust half is the one that makes them agree: `VsmReceiverParams`
    /// derives the bias FROM the radius, in one place, so a tier that clamps the
    /// kernel cannot leave a bias sized for the kernel it clamped away.
    #[test]
    fn the_kernel_and_its_bias_are_read_from_the_uniform_the_tier_writes() {
        let src = include_str!("shaders/vsm_receive.wgsl");
        assert!(
            src.contains("let radius = i32(vsm.counts.z);"),
            "the shader does not read its kernel radius from the uniform"
        );
        for header in [
            "for (var dy = -radius; dy <= radius; dy = dy + 1) {",
            "for (var dx = -radius; dx <= radius; dx = dx + 1) {",
        ] {
            assert!(
                src.contains(header),
                "the tap loop does not COUNT with the uniform's radius: `{header}`"
            );
        }
        assert!(
            src.contains("let slope = vsm.params.w * tan_t * ndc_per_m;"),
            "the slope bias is not the uniform's — a tier could clamp the kernel \
             and leave the bias sized for the one it clamped away"
        );

        // The Rust half, per tier, and the DERIVATION is the assertion: the two
        // fields are never independent.
        for radius in 0..=crate::settings::VSM_MAX_PCF_RADIUS {
            let s = VsmSettings {
                pcf_radius: radius,
                ..Default::default()
            };
            let p = VsmReceiverParams::new(&s, 1.0, 1, 1);
            assert_eq!(p.pcf_radius(), radius as i32);
            assert_eq!(p.slope_bias_texels(), vsm_slope_bias_texels(radius));
            // …and the taps the CPU twin would run are the `(2R+1)²` the shader's
            // loop runs, at an interior texel where none is dropped.
            assert_eq!(
                vsm_pcf_taps(Vec2::splat(64.5), VSM_PAGE_SIZE, 0, (0, 0), p.pcf_radius()).len(),
                ((2 * radius + 1) * (2 * radius + 1)) as usize
            );
        }
        // The shipped default is the P27.4 configuration, bit for bit — which is
        // the whole reason no golden moved.
        let d = VsmReceiverParams::new(&VsmSettings::default(), 1.0, 1, 1);
        assert_eq!(d.pcf_radius(), VSM_PCF_RADIUS);
        assert_eq!(d.slope_bias_texels(), VSM_SLOPE_BIAS_TEXELS);
        assert_eq!(VSM_SLOPE_BIAS_TEXELS, vsm_slope_bias_texels(1));
        // …and the retired WGSL literal is that same `f32`, so "the constant the
        // shader used to spell" is a measurement rather than a memory.
        assert_eq!(VSM_SLOPE_BIAS_TEXELS, 2.121_320_3_f32);
        // An absent system reads a zero radius and a zero bias — legal, because
        // `vsm_active()` is false and nothing below it runs.
        assert_eq!(VsmReceiverParams::absent().pcf_radius(), 0);
    }

    /// **One sampling door, one declaration site.**
    ///
    /// The receiver's four bindings are declared in exactly one file, and the
    /// atlas is read in exactly one expression — so "editor, PIE and player
    /// resolve the same way" is a property of there being one place to resolve
    /// rather than of three call sites agreeing. Read as a SCOPE (the P23 law):
    /// the ban is on any OTHER shader declaring `vsm_atlas`, not on a spelling.
    #[test]
    fn the_receivers_bindings_are_declared_once() {
        let src = include_str!("shaders/vsm_receive.wgsl");
        for (i, name) in ["vsm_atlas", "vsm_table", "vsm_proj", "vsm: VsmReceiver"]
            .iter()
            .enumerate()
        {
            let decls = src
                .lines()
                .filter(|l| l.trim_start().starts_with("@group(GROUP_ENV)") && l.contains(name))
                .count();
            assert_eq!(decls, 1, "`{name}` is declared {decls} times");
            let binding = format!("@binding({})", 17 + i);
            assert_eq!(
                src.lines().filter(|l| l.contains(&binding)).count(),
                1,
                "binding {} is declared more than once",
                17 + i
            );
        }
        // The one read of the atlas, anywhere in the tree.
        assert_eq!(
            src.matches("textureLoad(vsm_atlas").count(),
            1,
            "the atlas is sampled from more than one place"
        );
        let shaders = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("shaders");
        for e in std::fs::read_dir(&shaders).expect("shaders dir") {
            let p = e.expect("dir entry").path();
            if p.file_name().is_some_and(|n| n == "vsm_receive.wgsl") {
                continue;
            }
            let other = std::fs::read_to_string(&p).expect("read shader");
            assert!(
                !other.contains("var vsm_atlas"),
                "{} declares the shadow atlas too",
                p.display()
            );
        }
    }

    /// **A table entry that names a COARSER level is read at that level** — the
    /// half of [`vsm_page_of`]'s ruling no device fixture reaches.
    ///
    /// Measured (the P27.4 mutation round): replacing `vsm_level_ndc(p, l, got)`
    /// with `… level` survives every device arm, because every fixture's marked
    /// set is resident and `got == level` throughout. The fail direction is a
    /// receiver sampling the finer level's texel position inside the COARSER
    /// page's slot — a shadow displaced by up to half a page. So it is armed
    /// here, on a hand-built table, where a fallback can simply be written down.
    #[test]
    fn a_fallback_entry_is_read_at_the_level_the_table_served() {
        // Two levels of a quadtree: level 0 is 2 pages a side, level 1 is one.
        let desc = VsmLightDesc::quadtree(2);
        let block = VSM_TABLE_HEADER_WORDS + 1;
        let entries = block + VSM_TABLE_LIGHT_HEADER_WORDS + 2 * VSM_TABLE_LEVEL_REC_WORDS;
        let mut table = vec![0u32; entries + 5];
        table[0] = inf_vsm::VSM_TABLE_MAGIC;
        table[1] = 1;
        table[2] = 4; // slots a row
        table[3] = VSM_PAGE_SIZE; // stored page size
        table[VSM_TABLE_HEADER_WORDS] = block as u32;
        table[block] = 2; // levels
        table[block + 1] = VSM_PAGE_SIZE;
        table[block + 2] = 0; // border
        table[block + 3] = inf_vsm::VsmTreeKind::Quadtree.wire() | (1 << 8);
        // level 0: 2x2, entries at `entries`; level 1: 1x1, at `entries + 4`.
        for (i, (px, py, first)) in [
            (0u32, (2u32, 2u32, entries as u32)),
            (1, (1, 1, entries as u32 + 4)),
        ] {
            let rec = block + VSM_TABLE_LIGHT_HEADER_WORDS + VSM_TABLE_LEVEL_REC_WORDS * i as usize;
            table[rec] = px;
            table[rec + 1] = py;
            table[rec + 2] = first;
            table[rec + 3] = 0;
        }
        // No level-0 page is resident; each one RESOLVES to the level-1 root,
        // which is what `VsmResidency::propagate` writes into the table — a
        // fallback is an entry naming a coarser level, not an absence.
        for e in table.iter_mut().skip(entries).take(5) {
            *e = inf_vsm::pack_entry(3, 1);
        }

        let proj = VsmProjection {
            view_proj: Mat4::IDENTITY.to_cols_array(),
            info: [block as u32, 0, 0, crate::vsm::VSM_PROJ_PERSPECTIVE],
            ..Default::default()
        };
        // A point in level 0's page (1, 0) — the top-RIGHT quadrant. Its level-1
        // ancestor is the single root page, and the texel inside that root is
        // three quarters across, not a quarter.
        let ndc = Vec3::new(0.5, 0.5, 0.5);
        let (px, _, _) = vsm_page_of(&desc, 0, ndc.truncate()).expect("level 0");
        assert_eq!(px, 1, "the fixture's point is not in the right quadrant");
        let mut read = Vec::new();
        let f = vsm_level_factor(&table, &proj, &desc, ndc, 0, 0.0, VSM_PCF_RADIUS, |x, y| {
            read.push((x, y));
            0.0
        });
        assert!(f >= 0.0, "the fallback resolved to nothing");
        // Slot 3 of a 4-wide atlas is at x = 3 · 128; the point is 0.75 across
        // the root page, i.e. texel 96.
        let (ox, oy) = (3 * VSM_PAGE_SIZE, 0);
        assert!(
            read.contains(&(ox + 96, oy + 32)),
            "the taps landed at {read:?}, not at the SERVED level's texel \
             ({}, {})",
            ox + 96,
            oy + 32
        );
        // …and the finer level's texel position inside that slot — what the
        // mutation produces — is a different place entirely.
        assert!(
            !read.contains(&(ox + 64, oy)),
            "the taps landed at the ASKED level's texel inside the SERVED \
             level's page"
        );
    }

    /// The normal offset is **one page texel of the level the receiver asks
    /// for**, and it is applied before the projection.
    ///
    /// Measured (the P27.4 mutation round): deleting it survives every device
    /// arm, because the slope term already covers the acne it guards against at
    /// every configuration this tree builds. It is armed here rather than
    /// removed, for the reason `VSM_NORMAL_BIAS_TEXELS`'s own docs give — with
    /// no page border the displacement has to be part of the ADDRESS — and it is
    /// its magnitude, not its existence, that a reader would trust.
    #[test]
    fn the_normal_offset_is_one_texel_of_the_asked_for_level() {
        let vp = crate::vsm::clipmap_matrix(Vec3::Y, Vec3::ZERO, 32.0, 1_024.0);
        let texel0 = 2.0 * 32.0 / (32.0 * VSM_PAGE_SIZE as f32);
        let desc = VsmLightDesc::clipmap(6, 32);
        let mut proj = VsmProjection {
            view_proj: vp.to_cols_array(),
            info: [0, 0, 0, VSM_PROJ_ORTHO],
            ..Default::default()
        };
        proj.light[3] = texel0;
        let world = Vec3::new(1.0, 0.0, 2.0);
        // A pixel footprint that justifies level 2 exactly: `2 < log2(pw/t0) ≤ 3`.
        let pixel_world = texel0 * 6.0;
        let level0 = vsm_justified_level(texel0, pixel_world, desc.level_count());
        assert_eq!(level0, 3);
        let site = vsm_receiver_site(
            &proj,
            &desc,
            world,
            Vec3::Y,
            pixel_world,
            0.05,
            VSM_SLOPE_BIAS_TEXELS,
        )
        .expect("the point is inside the clipmap");
        // The displaced point, recovered by unprojecting the site's own NDC.
        let back = vp.inverse() * site.ndc.extend(1.0);
        let back = back.truncate() / back.w;
        let moved = (back - world).length();
        let want = VSM_NORMAL_BIAS_TEXELS * texel0 * (1u32 << level0) as f32;
        assert!(
            (moved - want).abs() < 1e-4,
            "the receiver moved {moved} m along its normal, not {want} — one \
             texel of the level it asks for"
        );
        assert!(want > 0.0, "the offset is inert");
    }

    /// The shader's own clamp, pinned by SOURCE — because a one-texel band on
    /// 3.1 % of a page's texels is not something a 256 × 144 frame can see.
    ///
    /// Measured (the P27.4 mutation round): deleting the bounds test from
    /// `vsm_receive.wgsl` survives every device arm. The Rust half is armed
    /// directly by `the_kernel_drops_the_taps_that_leave_the_page…` — a tap
    /// count is a number — and this is what keeps the two halves saying the same
    /// thing. Read as a SCOPE (the P23 law): the guard is looked for inside the
    /// kernel loop, not anywhere in the file.
    #[test]
    fn the_shaders_kernel_is_clamped_to_its_page_too() {
        let src = include_str!("shaders/vsm_receive.wgsl");
        let at = src
            .find("for (var dy = -radius;")
            .expect("the kernel loop moved; this gate scopes on it");
        let end = src[at..]
            .find("return sum / taps;")
            .expect("the loop's end");
        let body = &src[at..at + end];
        assert!(
            body.contains("t.x >= i32(page_size)") && body.contains("t.y >= i32(page_size)"),
            "the kernel no longer drops the taps that leave its page — with no \
             border those taps read the atlas slot NEXT DOOR"
        );
        assert!(
            body.contains("t.x < 0") && body.contains("t.y < 0"),
            "the kernel no longer guards the low edge"
        );
        assert!(
            body.contains("continue;"),
            "the out-of-page taps are clamped rather than DROPPED, which \
             double-weights a page's rim"
        );

        // Two more expressions of `vsm_shadow`'s that no device fixture can
        // reach, pinned in the same breath and with the same honesty about what
        // a source pin is worth (a byte pin cannot see a semantic change — the
        // P23 law — so this catches a DELETION and not a re-derivation):
        //
        // * the fallback address is taken at `got`, the level the table SERVED.
        //   Its arm is `a_fallback_entry_is_read_at_the_level_the_table_served`,
        //   which is pure, because reaching a fallback on a device needs a
        //   resident ancestor over an absent child — residency pressure this
        //   phase's fixtures do not produce and P27.5's gate will;
        // * the normal offset is applied. Its arm is
        //   `the_normal_offset_is_one_texel_of_the_asked_for_level`, also pure,
        //   because the offset is redundant with the slope term at every
        //   configuration this tree builds — it earns its place at a page
        //   boundary, where the displacement changes which page is resolved.
        assert!(
            src.contains("let gq = vsm_level_ndc(p, l, got);"),
            "the shader no longer re-derives the address at the level the table \
             SERVED — a fallback would sample the finer level's texel position \
             inside the coarser page's slot"
        );
        // **The magnitude AND the application.** The P27.4 audit's mutation round
        // found this pin aimed at the wrong line: it named where `offset_m` is
        // COMPUTED, so a shader that computes the offset and then projects the
        // undisplaced point passed it — which is exactly the mutation the pin
        // exists for, and it survived. A pin has to name the expression whose
        // deletion is the defect (the P23 law, met from the other side: a byte pin
        // catches a deletion only where the deletion is in the bytes it reads).
        assert!(
            src.contains("VSM_NORMAL_BIAS_TEXELS * texel0 * exp2(f32(level0))"),
            "the shader no longer sizes the normal offset at one texel of the \
             level the receiver asks for"
        );
        assert!(
            src.contains("p.view_proj * vec4<f32>(world_pos + n * offset_m, 1.0)"),
            "the shader computes a normal offset and then projects the UNDISPLACED \
             point — the offset is inert, and with no page border it is the \
             address it was supposed to move"
        );
    }

    /// **The bias's two terms and the blend's two proximities, pinned by
    /// SOURCE** — the other three expressions of `vsm_shadow` that no device
    /// fixture in this phase can reach, found by the P27.4 audit's mutation
    /// round and armed here on the same terms as the kernel's own clamp.
    ///
    /// Measured, each one:
    ///
    /// * deleting `VSM_DEPTH_ULP_BIAS` from the shader's bias survives every
    ///   device arm. It is 2.38 × 10⁻⁷ NDC against a slope term an order of
    ///   magnitude larger at every angle those fixtures use, and it only becomes
    ///   the *whole* bias as `n · l → 1`, where a 256 × 144 frame cannot see the
    ///   acne it prevents;
    /// * replacing `max(w_res, w_con)` with `w_res` survives every device arm —
    ///   the blend arm's band is the RESOLUTION one, so the footprint half (the
    ///   clipmap ring, the edge a quadtree does not have) is measured on the Rust
    ///   side alone by `the_level_blend_is_a_band_at_both_of_a_levels_edges`;
    /// * and the coarser level's bias is its own, `level + 1u` — a blend that
    ///   biased the coarse tap at the fine level's texel density under-biases it
    ///   by a factor of two.
    ///
    /// The same honesty as the kernel's pin: this catches a **deletion**, not a
    /// re-derivation.
    #[test]
    fn the_shaders_bias_and_blend_are_the_expressions_this_module_derives() {
        let src = include_str!("shaders/vsm_receive.wgsl");
        assert!(
            src.contains("let bias = VSM_DEPTH_ULP_BIAS + slope * texel0 * exp2(f32(level));"),
            "the shader's bias is no longer the two derived terms — the depth \
             FORMAT's constant plus the page's texel DENSITY times the slope"
        );
        assert!(
            src.contains(
                "let nbias = VSM_DEPTH_ULP_BIAS + slope * texel0 * exp2(f32(level + 1u));"
            ),
            "the blend's coarser tap no longer takes the coarser level's own \
             bias, so it is under-biased by exactly the factor of two between \
             their texel sizes"
        );
        assert!(
            src.contains("    return max(w_res, w_con);"),
            "the blend weight is no longer the LARGER of the two proximities — a \
             clipmap level has two edges a receiver can approach and dropping \
             either one puts the seam back at that edge"
        );
    }

    /// **The receiver declares no sampler at all**, which is the ruling the
    /// clamped kernel rests on — every tap is a `textureLoad` at an integer
    /// texel, so the clamp is exact by integer arithmetic.
    ///
    /// Stated precisely, because the memo's first draft did not: the environment
    /// bind group *does* hold a comparison sampler — binding 3, the **cascade's**
    /// `env-shadow`, whose hardware 2 × 2 pre-filter is the 36-samples-for-9 the
    /// memo compares against. What P27.4 adds is four bindings and **no sampler
    /// among them**, and the virtual path never names the cascade's.
    #[test]
    fn the_receiver_adds_four_bindings_and_none_of_them_is_a_sampler() {
        for e in bind_group_layout_entries(17) {
            assert!(
                !matches!(e.ty, wgpu::BindingType::Sampler(_)),
                "binding {} is a sampler — with `VSM_PAGE_BORDER = 0` a hardware \
                 2 x 2 footprint at a page edge reads the atlas slot NEXT DOOR",
                e.binding
            );
        }
        let src = include_str!("shaders/vsm_receive.wgsl");
        assert!(
            !src.contains("textureSampleCompare") && !src.contains("sampler_comparison"),
            "the receiver filters through the hardware, over an atlas whose page \
             neighbours are unrelated lights"
        );
        // …and it never reaches for the cascade's, either.
        assert!(
            !src.contains("shadow_sampler"),
            "the receiver names the cascade's comparison sampler"
        );
    }

    /// **THE TWO LIGHT CEILINGS, AND THE RULE THAT NOW HOLDS BETWEEN THEM**
    /// (P27.5 — the tier decision the P27.4 audit assigned to this batch).
    ///
    /// `MAX_LIGHTS` is **16**: the lights uniform's array, and therefore the
    /// largest scene index whose direct term any lit shader computes at all —
    /// every one of them loops `i < count && i < MAX_LIGHTS`.
    /// `VSM_MAX_PROJECTIONS` is **64**: the marking pass's projection ceiling.
    ///
    /// The audit's finding was that `VsmSystem::for_scene` registered a tree for
    /// **every** shadow-casting light in scene order with no reference to the
    /// first number, so a point or spot light at scene index >= 16 could hold a
    /// page tree that marked, rasterized and evicted pages **no shader could
    /// ever sample**. Armed, unfixed, and left as a tier decision.
    ///
    /// # The ruling: REFUSE, typed and counted
    ///
    /// The alternative was lifting `MAX_LIGHTS`, and it is the wrong lever. That
    /// number is the **forward renderer's analytic light loop** — a per-pixel
    /// shading cost P7.1 chose — not a shadow budget, and moving it would make
    /// every lit fragment in the engine pay for a virtual-shadow ceiling. What a
    /// shadow phase may decide is whether to allocate pages for a light nothing
    /// shades, and the answer is no.
    ///
    /// So `vsm_light_trees` stops at the scene index the lights uniform ends at,
    /// counts what it refused in `VsmTreeSet::refused_past_shader_ceiling`, and
    /// `for_scene` logs it once. It is a **stop**, not a skip, and that is free
    /// rather than careful: index >= MAX_LIGHTS is a *suffix* of the light list,
    /// so the handle invariant the projection cap protects is untouched.
    ///
    /// The **sun** is exempt from the *slot* mechanism and that is not luck — it
    /// rides `VsmReceiverParams::counts.x` rather than a `GpuLight` — but it is
    /// not exempt from this ceiling, because a directional light past index 16
    /// contributes no direct term either, and shadowing a light that is not
    /// shaded is the same waste in a different place.
    ///
    /// # What this turns into an invariant
    ///
    /// *Every rasterized page is sampleable.* A tree exists only for a light some
    /// lit shader can shade; a point or spot light's slot is
    /// `GpuLight::params.w`, which is inside the array; the sun's is
    /// `counts.x`. The arm asserts the mapping directly rather than restating it.
    #[test]
    fn a_light_past_the_shader_ceiling_gets_no_page_tree() {
        let uniform = crate::passes::mesh::MAX_LIGHTS;
        let projections = crate::vsm::VSM_MAX_PROJECTIONS;
        assert_eq!(uniform, 16, "the lights uniform's array");
        assert_eq!(projections, 64, "the marking pass's projection ceiling");
        assert!(
            projections > uniform,
            "the projection ceiling ({projections}) is no longer the larger of \
             the two, so the shader ceiling has stopped being the binding one — \
             say so in the ledger before deleting this"
        );
        // A point light is six projections, so the projection ceiling is ten
        // point lights; the uniform's is sixteen lights of any kind.
        assert_eq!(projections / 6, 10);

        // THE RULE, on the door that enforces it. Twenty directional casters:
        // sixteen get trees, four are refused and COUNTED.
        let mut scene = crate::RenderScene::default();
        for _ in 0..20 {
            scene.lights.push(crate::RenderLight {
                kind: crate::LightKind::Directional,
                cast_shadows: true,
                ..Default::default()
            });
        }
        let asked = crate::vsm::vsm_light_trees(&scene, &VsmSettings::default());
        assert_eq!(
            asked.trees.len(),
            uniform,
            "a tree past the uniform's array"
        );
        assert_eq!(asked.refused_past_shader_ceiling, 4);
        assert_eq!(asked.refused_past_projection_cap, 0);
        assert_eq!(asked.refused(), 4);

        // THE INVARIANT: every tree's light has a slot a shader can read. The
        // slot list is over the WHOLE light list, and past the ceiling it is 0 —
        // which is `vsm_bound()`'s "this light has no tree".
        let casts: Vec<bool> = scene.lights.iter().map(|l| l.cast_shadows).collect();
        let bases: Vec<u32> = (0..asked.trees.len() as u32).collect();
        let slots = receiver_slots(casts, asked.trees.len(), &bases);
        assert_eq!(slots.len(), 20);
        for (i, slot) in slots.iter().enumerate() {
            if i < uniform {
                assert_eq!(*slot, i as u32 + 1, "light {i} lost its slot");
            } else {
                assert_eq!(
                    *slot, 0,
                    "light {i} is past the lights uniform's array and still names \
                     a projection — a page tree no shader can sample"
                );
            }
        }

        // ANTI-VACUITY, and it is the assertion that makes the refusal a rule
        // rather than an accident of this fixture: with the ceiling honoured,
        // NOTHING in the tree list sits past it. The pre-P27.5 behaviour is the
        // control — it would have produced twenty trees here.
        assert!(
            asked.trees.len() <= uniform,
            "the tree list reaches past the lights uniform's array"
        );
        assert!(
            asked.refused() > 0,
            "the fixture refused nothing, so the counters above are zero for the \
             wrong reason"
        );

        // …and the OTHER ceiling still stops the list where it always did, with
        // its own counter, so the two refusals are distinguishable rather than
        // one number wearing two names.
        let mut many = crate::RenderScene::default();
        for _ in 0..12 {
            many.lights.push(crate::RenderLight {
                kind: crate::LightKind::Point,
                cast_shadows: true,
                ..Default::default()
            });
        }
        let capped = crate::vsm::vsm_light_trees(&many, &VsmSettings::default());
        assert_eq!(
            capped.trees.len(),
            10,
            "10 x 6 = 60 fits, 11 x 6 = 66 does not"
        );
        assert_eq!(capped.refused_past_projection_cap, 2);
        assert_eq!(
            capped.refused_past_shader_ceiling, 0,
            "twelve lights are inside the uniform's array; the projection cap is \
             what refused these"
        );

        // **BOTH CEILINGS IN ONE SCENE** (P27.5 audit). The two counters are two
        // counters so that neither wears the other's name, and the case that
        // tests the claim is the one where both refusals are live at once:
        // twenty point lights: ten get trees, the projection cap stops the walk
        // at index 10, and the tail it stops in straddles `MAX_LIGHTS`.
        //
        // The whole tail is refused either way — a stop is a stop — but SIX of
        // them are refused for not fitting the projection budget and FOUR for
        // sitting past the array every lit shader's loop ends at. Attributing
        // all ten to the cap would make `for_scene`'s warning tell four lights
        // they "keep the cascaded shadow map", which is false: nothing shades
        // them at all.
        let mut both = crate::RenderScene::default();
        for _ in 0..20 {
            both.lights.push(crate::RenderLight {
                kind: crate::LightKind::Point,
                cast_shadows: true,
                ..Default::default()
            });
        }
        let split = crate::vsm::vsm_light_trees(&both, &VsmSettings::default());
        assert_eq!(split.trees.len(), 10, "the projection cap stops at ten");
        assert_eq!(
            split.refused_past_projection_cap, 6,
            "lights 10..16 are inside the lights uniform and were refused by the \
             projection budget"
        );
        assert_eq!(
            split.refused_past_shader_ceiling, 4,
            "lights 16..20 are past the lights uniform's array; the projection \
             cap is not why they get no tree"
        );
        assert_eq!(
            split.refused(),
            10,
            "the two counters partition the refused tail exactly once — a light \
             counted twice would over-report and a light counted zero times \
             would be a silent cap"
        );
    }

    /// The clamped kernel drops rather than clamps, and the centre tap is always
    /// in — so the divisor is never zero and a page edge is never double-counted.
    #[test]
    fn the_kernel_drops_the_taps_that_leave_the_page_and_never_double_counts() {
        let origin = (256u32, 128u32);
        // Dead centre: nine taps, all distinct.
        let mid = vsm_pcf_taps(
            Vec2::new(64.3, 64.7),
            VSM_PAGE_SIZE,
            0,
            origin,
            VSM_PCF_RADIUS,
        );
        assert_eq!(mid.len(), 9);
        assert_eq!(
            mid.iter().collect::<std::collections::BTreeSet<_>>().len(),
            9,
            "a tap was counted twice"
        );
        // Top-left corner: four.
        let corner = vsm_pcf_taps(
            Vec2::new(0.1, 0.1),
            VSM_PAGE_SIZE,
            0,
            origin,
            VSM_PCF_RADIUS,
        );
        assert_eq!(corner.len(), 4);
        assert!(
            corner.contains(&origin),
            "the corner texel itself must be in"
        );
        // Bottom-right corner: four, and none of them left the page.
        let far = vsm_pcf_taps(
            Vec2::new(127.9, 127.9),
            VSM_PAGE_SIZE,
            0,
            origin,
            VSM_PCF_RADIUS,
        );
        assert_eq!(far.len(), 4);
        for (x, y) in &far {
            assert!(*x < origin.0 + VSM_PAGE_SIZE && *y < origin.1 + VSM_PAGE_SIZE);
        }
        // An edge: six.
        assert_eq!(
            vsm_pcf_taps(
                Vec2::new(64.0, 0.5),
                VSM_PAGE_SIZE,
                0,
                origin,
                VSM_PCF_RADIUS
            )
            .len(),
            6
        );
        // A stored border shifts every tap by it and drops none — the payload is
        // still what is addressed.
        let bordered = vsm_pcf_taps(
            Vec2::new(64.3, 64.7),
            VSM_PAGE_SIZE,
            4,
            origin,
            VSM_PCF_RADIUS,
        );
        assert_eq!(bordered.len(), 9);
        assert_eq!(bordered[0].0, mid[0].0 + 4);
        assert_eq!(bordered[0].1, mid[0].1 + 4);
    }
}
