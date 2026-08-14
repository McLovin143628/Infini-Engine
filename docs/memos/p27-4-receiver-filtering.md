# P27.4 — the shadow kernel is **clamped to its page**, and the bias has exactly two terms

**Status:** decided 2026-08-13, during P27.4, as the ROADMAP's clause-1 ruling
(*"`shadow_factor` reads through the page table with PCF that respects page
borders (border texels or clamped kernels — **measured**)"*) and its clause-4
one (*"the bias story re-derived for page-resolution shadows and pinned by
goldens"*). Every number below is produced by an arm rather than by this
document: `inf_render::vsm_receiver`'s
`the_clamped_kernel_is_cheaper_and_its_error_is_bounded_where_the_other_is_not`
and `the_bias_is_two_derived_terms_and_neither_is_a_tuned_constant`.

`docs/memos/p27-1-page-geometry.md` left this decision open on purpose — *"the
alternative P27.4 may prefer, and which this ruling does not foreclose, is a
clamped kernel … measurable against the per-tap resolve on goldens when the
receiver exists"*. The receiver exists.

---

## 1. The kernel: **clamped**, and the receiver declares no sampler

Pages are borderless (`VSM_PAGE_BORDER = 0`), so a 3 × 3 PCF kernel at a page
edge has taps that leave the page. Two candidates:

| | clamped kernel | per-tap resolve |
|---|---|---|
| taps that leave the page | **dropped**, weight renormalized | re-addressed through the table |
| table resolutions per shadow sample | **1** | **9** |
| level-record reads per sample (8-level clipmap) | **8** | up to **72** |
| exact when the neighbour is resident at the same level | no | yes |
| exact when the neighbour is **absent** | **yes** | **no** |

### The four numbers

1. **How often it matters: 3.10 %.** A tap leaves its page only for a centre
   texel within `VSM_PCF_RADIUS` of an edge, which is
   `1 − ((128 − 2)/128)² = 508/16 384 = 3.1006 %` of a page. (A 5 × 5 kernel
   would double it to 6.15 %; the number is a function of the radius, not a
   property of 128.)
2. **What it costs: one resolution against nine.** And a clipmap resolution is a
   *walk* — the containment walk from the justified level upward — so at the
   shipped 8-level default the per-tap resolve is up to **72** level-record
   reads per shadow sample against **8**.
3. **The clamped kernel's error is bounded by the kernel's own variance.** It is
   the full kernel restricted to a subset of its taps and renormalized, so
   wherever the shadow field is locally constant — which is everywhere except a
   penumbra — it is **exactly** the full kernel's answer. Inside a penumbra it
   differs by at most the dropped weight: `3/9 = 0.333` at an edge, `5/9 = 0.556`
   at a corner.
4. **The per-tap resolve's error is bounded by nothing.** A neighbouring page may
   be `VSM_ENTRY_NONE`, which this phase reads as *lit*. So a cross-page tap over
   a missing neighbour injects "lit" into the kernel on ground that is
   **uniformly shadowed** — where the clamped kernel is exact — and the result is
   a bright one-texel line along a page seam, in the leak direction, on the
   3.10 % of texels above. It appears *because* the filter was made more exact.

That is the ruling: **clamp**. It is cheaper by a factor of nine, and the case it
is worse in (a resident neighbour at the same level, inside a penumbra) is
bounded by the filter's own width, while the case it is better in (an absent
neighbour) is not bounded at all.

> **Numbers 3 and 4 are measurements now** (P27.4 audit). They shipped as
> arithmetic identities inside the arm that names them — `3.0/9.0` asserted equal
> to `3.0/9.0`, and `shadowed · kept/kept` asserted equal to `shadowed` — which
> pass whatever the kernel does and are the vacuous-check law met again. Number 3
> now runs `vsm_level_factor` on a one-page tree with a uniform atlas at an
> interior texel, an edge and both corners and asserts the factor is *identical*;
> number 4 takes the dropped weights from `vsm_pcf_taps`'s own kept-tap counts
> (9 / 6 / 4) rather than from typed fractions. Mutation-verified: a kernel that
> renormalizes over the full nine taps instead of the kept ones now fails, and
> did not before.

### …and therefore no sampler

The receiver declares **four bindings and no sampler**, and that is part of the
same ruling rather than an omission. Hardware `textureSampleCompare`
filtering is a 2 × 2 footprint in **atlas** space, and a page's atlas neighbour is
slot `s + 1` — an unrelated page of a possibly unrelated light. With no border
there is nothing correct for the hardware to filter against. So every tap is a
`textureLoad` at an integer texel and the compare happens in the shader, which
also makes the clamp *exact*: a tap is inside its page by integer arithmetic
rather than by a texel-centre offset.
`the_receiver_adds_four_bindings_and_none_of_them_is_a_sampler` is the arm, over
`bind_group_layout_entries` and over the shader text.

> **Corrected by the P27.4 audit.** This section's first draft said *"there is no
> comparison sampler in the environment bind group"*, which is refutable by grep:
> there is one, at binding **3** — the CASCADE's `env-shadow`, a `LessEqual`
> comparison sampler over the forward-Z cascade array, and its hardware 2 × 2 is
> precisely the pre-filter the paragraph below says this kernel gives up. The
> claim that survives the check is the narrower and truer one: **P27.4 adds no
> sampler and the virtual path reads none.**

The cost is one binding saved and the hardware's 2 × 2 pre-filter given up. The
kernel is 9 explicit taps where the cascade path gets 9 × 4 = 36 samples' worth
of smoothing for the same instruction count. A future batch that wants the extra
softness has two options and both are written down: a wider explicit kernel
(radius 2 doubles the crossing set to 6.15 % and the taps to 25), or a bordered
page — which `docs/memos/p27-1-page-geometry.md` costs at 121 pages of 1 024 and
a 3 × 3 invalidation dilation that would make P27.3's page-exact clause false by
construction.

### The address walk is `O(1)` here, and it may **not** be in `inf-vt`

`vt_sample.wgsl`'s header records a measurement the P26.2 audit made: a virtual
texture must walk its tile tree down from the address it asked for, because the
container halves an extent with `w / 2` and a 511-texel level does not sit over a
255-texel one — re-deriving the tile from `uv` at the resolved mip lands a whole
tile away at exactly one column of one tile boundary.

**A shadow page tree has no such slack.** A clipmap level is a world lattice of
known stride and origin, and a quadtree level is `2^(n−1−L)` pages over the
*same* frustum — so `⌊u·N⌋` and `⌊⌊u·2N⌋/2⌋` are the same integer for every
`u ≥ 0`. Re-deriving the address at the level the table served is therefore
identical to walking the ancestor chain and is `O(1)` instead of `O(levels)`.
`the_re_derived_address_is_the_ancestor_the_chain_walks_to` sweeps both tree kinds
— including clipmap levels whose per-level offsets are deliberately **not**
concentric — against `VsmLightDesc::ancestor_at`, and
`a_fallback_entry_is_read_at_the_level_the_table_served` drives the fallback case
directly on a hand-built table, because no device fixture in this phase produces
one.

---

## 2. The bias: one term from the depth **format**, one from the page's texel
   **density**

`docs/memos/p27-1-depth-convention.md` deferred the bias story to here and
promised the precision it would have (251× better on perspective lights,
materially unchanged on the directional one). The derivation:

### The constant term is a property of `f32`, and is therefore constant in NDC

A page stores `f32`. The ULP of a value in the worst binade `[0.5, 1)` is `2⁻²⁴`,
and the P27.1 measurement — which bisected the *stored* depth through the shipped
matrices rather than differentiating a formula — found the worst step at about
two of those. `VSM_DEPTH_ULP_BIAS` is **four**, i.e. `2 × f32::EPSILON =
2.384 × 10⁻⁷`, which is that with a factor of two in hand.

**The unit is the whole reason this term is not authored.** Expressed in *texels*
it would not be constant at all: the along-light box is `2·extent·2^(levels−1)`
deep while a level-0 texel is `2·extent/(N·128)` wide, so the same guard is
**0.25 texels** at the shipped defaults and **64 texels** at the 16-level
ceiling — 256× across eight more levels. In NDC it is one number for every
configuration, and it does not have to be re-derived per level, per light or per
settings value.

> **Corrected by the P27.4 audit**, and it is a factor of two. This paragraph
> shipped as *0.125* and *32*, which is the guard computed at **two** ULP where
> `VSM_DEPTH_ULP_BIAS` is **four** — the value the sentence above it states.
> `the_bias_is_two_derived_terms_and_neither_is_a_tuned_constant` had the right
> numbers all along (`texels_at(8)` = 0.25, `texels_at(16)` = 64) and its
> tolerances were wide enough to hold both, so "every number below is produced by
> an arm" was true of the arm and not of the prose. The *ratio* the argument
> actually rests on — 256× — was right in both tellings, which is how a factor of
> two survives a reading. Both numbers are now held to 10⁻⁴ by that arm.

### The slope term is a property of the page's texel density

A page stores one depth per texel, so between two texel centres a tilted surface
departs from the stored value by the slope across that distance. The receiver's
own position may sit anywhere inside its texel (± ½) and the kernel reaches
`VSM_PCF_RADIUS` further, on both axes — so the worst lateral separation between
the receiver and a tap's texel centre is

```
VSM_SLOPE_BIAS_TEXELS = (R + ½)·√2 = 2.1213   texels at R = 1
```

a derivation rather than a number. Times the texel's world size
`texel₀ · 2^level`, times `tan θ` (θ between the surface normal and the direction
to the light, clamped at `VSM_MAX_SLOPE = 8`, about 83°, past which `n · l` has
already taken the direct term to nothing), gives metres along the light. The
projection's own `∂z/∂m` converts it to NDC, and it is read **off the shipped
matrix** in both branches:

* **orthographic** — `ndc.z` is affine in the world point, so its gradient is
  row 2's `xyz` and the answer is that row's length (`1/range`), the same at
  every depth;
* **reverse-infinite perspective** — `ndc.z = near/d`, so `∂z/∂d = −z²/near`,
  which needs the receiver's own `ndc.z` and the near plane and nothing else.
  Row 2 here is `(0, 0, 0, near)` — the degenerate row the P27.2 audit named the
  **far** plane — which is exactly why this branch cannot use the orthographic
  form.

### The numbers, at the shipped defaults

`clipmap_pages_per_side` 64, `first_level_extent_m` 32 m, `clipmap_levels` 8:

| quantity | value |
|---|---|
| level-0 texel | **7.8125 mm** |
| along-light box | **8 192 m** |
| constant term | **2.384 × 10⁻⁷** NDC |
| slope term at 45°, level 0 | **2.02 × 10⁻⁶** NDC (16.5 mm along the light) |
| total at 45°, level 0 | **2.26 × 10⁻⁶** NDC |
| `ShadowSettings::depth_bias` (the cascade's flat constant) | **1.5 × 10⁻³** |
| ratio | **663.3×** (P27.4 audit: the table said 665; the arm now holds ±0.5) |

A cascade pays a flat constant because it has no per-texel density term to scale
by; three orders of magnitude is what that costs, and it is why a cascade's
contact shadows float and a page's do not.

### The sign, and a correction to the P27.1 memo

Under reverse-Z a **larger** stored depth is a caster **nearer** the light, so a
receiver is lit exactly when its own biased depth still reaches at least as far:

```
lit  ⟺  receiver.z + bias ≥ stored.z
```

`docs/memos/p27-1-depth-convention.md`'s closing consequence wrote this as
*"`depth > stored + bias`"*, which puts the bias on the **caster's** side and
makes a self-shadowing surface *more* shadowed rather than less — the opposite of
what a bias is for. Corrected there and here, and the mutation
`sum + select(0.0, 1.0, l.z >= stored_z + bias)` fails three arms.

### Normal offset

`VSM_NORMAL_BIAS_TEXELS = 1.0`: the receiver is pushed along its own normal by
one texel of the level it asks for, **before** it is projected. The order is the
point — with no page border, a displaced lookup may land in a different page, so
the displacement has to be part of the *address* rather than of the tap offsets.
One texel of 128 is 0.78 % of a page, which bounds how far the resolve can move.

Measured honestly: deleting it survives every device arm, because the slope term
already covers the acne it guards against at every configuration this tree
builds. It is kept as the standard second guard, it costs one multiply and one
matrix product, and `the_normal_offset_is_one_texel_of_the_asked_for_level` pins
its magnitude so a reader can trust the number rather than the existence.

> **The source pin was aimed at the wrong line** (P27.4 audit), and this is the
> most instructive of the batch's defects. `the_shaders_kernel_is_clamped_to_its_
> page_too` pinned `VSM_NORMAL_BIAS_TEXELS * texel0 * exp2(f32(level0))` — where
> `offset_m` is **computed** — so a shader that computes the offset and then
> projects `world_pos` undisplaced passed the pin, which is precisely the
> mutation the pin exists to kill, and it survived a second round. The pin now
> names the **application**, `p.view_proj * vec4<f32>(world_pos + n * offset_m,
> 1.0)`. The P23 law says a byte pin catches a deletion and not a re-derivation;
> the corollary it did not spell out is that it catches a deletion only in the
> bytes it actually reads.
>
> Two more expressions were unarmed on the same terms and are now pinned by
> `the_shaders_bias_and_blend_are_the_expressions_this_module_derives`: the
> shader's `VSM_DEPTH_ULP_BIAS +` term (deleting it survives every device arm —
> it is 2.38 × 10⁻⁷ against a slope term an order of magnitude larger at every
> angle those fixtures use) and the blend's `max(w_res, w_con)` (dropping the
> footprint half survives, because the device blend arm's band is the
> *resolution* one).

### What the derivation refuses to do, and the fixture that proved it

The slope bias is a function of the **page's** texel density, so a coarse page
cannot hold a shallow shadow — and the receiver correctly declines to draw one
rather than drawing an acne field. Measured while building
`the_derived_bias_leaves_a_lit_plane_lit_and_a_shadow_attached`: at a 94 mm texel
(a 256 × 144 frame at 60° asks for texels that coarse, because the level rule is
one texel per **screen pixel**) and an 18° sun, the bias is **0.61 m** against a
1.5 m caster's **0.46 m** of depth separation, and the shadow disappears
entirely. That is not a defect; it is a 94 mm shadow map being asked to resolve a
metre-scale box at a grazing angle. No constant would have made it, and a
constant small enough to draw it would stripe every flat surface in the frame.

---

## 3. The level blend

`VsmSettings::level_blend` (default **0.1**, the cascade's own default) is the
clipmap analogue of `ShadowSettings::cascade_blend`, and it is a **second knob**
rather than a reader of the first: the ROADMAP's "clipmap level blend replaces
cascade blend" is about which blend *runs*, and a project that turned the
cascade's seam fix off must not silently turn the clipmap's off with it. The
CSM's own knob is untouched by this batch.

A clipmap level has **two** edges a receiver can approach, and both are seams the
blend exists to remove, so the weight is the larger of two proximities:

* **resolution** — `lf = log₂(pixel_world / texel₀)` is where the level rule sits
  between levels, and `ceil` gives the level, so `lf − (L − 1)` is the receiver's
  position inside level `L`'s band. A containment-raised level makes that
  negative, which clamps to zero: a receiver on level `L` because nothing finer
  *reaches* it is not about to step to `L+1` for resolution;
* **footprint** — `max(|q.x|, |q.y|)` at the used level, which is 1 at the ring
  where the level runs out. Clipmaps only: a quadtree's levels share one frustum,
  so there is no ring to cross.

`0.0` restores the hard switch exactly — the second resolve is not issued at all,
which is the escape hatch `cascade_blend` already ships.

**The bound, written down rather than discovered later.** The blend can only act
where the **coarser level is resident**, and the marking pass asks for exactly one
level per pixel — so the band is populated only where some *other*, farther pixel
happened to want the level this one is blending toward. Along a receding plane
that is common (a far pixel's coarse page covers the near region too), and in a
tight interior it may never happen. A receiver that *wanted* its blend partner —
marking two bits instead of one at a band — is the fix, and it belongs with
P27.5's tier work rather than here, because it changes the want set that P27.3's
caching clause is measured against.

---

## 4. Face culling: **both faces keep casting**, and the reason is the terrain

The P27.2 ledger left *"no depth bias and no face culling"* and the P27.4 brief
asked for the two to be decided together, since front-face culling (storing the
**back** face's depth) is the classic alternative to a slope bias for closed
casters.

**Decided: no face culling, and the raster is not edited.** Front-face culling
removes acne for a closed caster at the cost of peter-panning proportional to the
object's thickness — but this engine's caster set is not closed. `sync_terrain`
builds a **single-sided open surface** from a tile's heights, and a masked cutout
is a **card**; front-face culling makes both cast *nothing at all*, which is a
strictly worse failure than acne and one no bias can recover. The derived slope
term costs 2.26 × 10⁻⁶ NDC at 45° and is measured to leave a flat lit plane
un-striped, so the thing back-face casting would buy is already bought.

The consequence for this batch is worth stating plainly, because it is what the
brief asked to protect: `vsm_raster.rs` is **not edited by P27.4** — `git diff`
over the batch names no line of it — so the content stamp is untouched by
construction and P27.3's `a_cached_pages_texels_are_what_a_fresh_raster_produces`
and `a_static_scene_stops_rasterizing_pages_after_warm_up` are green for the same
reason they were before: nothing changed underneath them.

---

## When to revisit

**P27.5**, for the two bounds above — the blend's resident-partner requirement,
and whether a wider kernel is worth the crossing set at the tiers that can afford
it. **P28.2**, where interleaved cluster pages change what a page *is*, and where
a bordered page becomes cheap enough to re-weigh. And **P28.3**, if the merged
residency ever lets a receiver ask for a page rather than only read one.
