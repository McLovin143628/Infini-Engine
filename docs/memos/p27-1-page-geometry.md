# P27.1 — the shadow page is 128 texels with **no border**

**Status:** decided 2026-08-13, during P27.1, as the ROADMAP's clause-3 ruling
(*"Page size: measure and memo — VT chose 128+4 border for filtering re-gather;
shadow pages want borders for PCF at page edges — decide with P27.4's filtering
in view, write the ruling down"*). The constants are
`inf_vsm::VSM_PAGE_SIZE = 128` and `inf_vsm::VSM_PAGE_BORDER = 0`, and
`atlas.rs`'s `the_ruled_page_geometry_is_the_one_the_memo_measured` fails if
either moves.

## The two questions, separately

### 1. The payload size: **128 texels**

`inf-vt` chose 128 too, and the reasons here are different ones — a shadow page
is not read out of a container, so the tile-directory argument does not apply.
What does:

| page | stored side | bytes (`Depth32Float`) | slots a side at 8 192 | wasted texels a side |
|---|---|---|---|---|
| 64 | 64 | 16 KiB | 128 | 0 |
| **128** | **128** | **64 KiB** | **64** | **0** |
| 128 + 4 border | 136 | 72.25 KiB | 60 | 32 |
| 256 | 256 | 256 KiB | 32 | 0 |

* **128 divides `Limits::default()`'s `max_texture_dimension_2d` of 8 192
  exactly** — 64 slots a side, nothing wasted. `inf-vt`'s 136-texel stored tile
  leaves 32 texels of every axis unused, which is the residual its `plan_pool`
  documents as arithmetic.
* **A 128² `Depth32Float` page is exactly 64 KiB**, so a budget in mebibytes is
  a whole number of pages and the atlas rectangle has **zero** residual for any
  power-of-two page count. The default 64 MiB budget is 1 024 pages laid out
  32 × 32 in a 4 096² texture, and the whole budget is spent.
* 64 would quadruple the page count for the same VRAM and quadruple the number
  of per-page render passes P27.2 has to issue (one `draw_indirect` per dirty
  page, on the two-buffer vgeom precedent — there is no `MULTI_DRAW_INDIRECT`
  here). 256 would quarter the granularity, so a mover that touches one texel
  invalidates four times the depth.

### 2. The border: **zero**, and this is the half that is not obvious

`inf-vt` bakes a 4-texel ring on every side so that a *filtered* sample at a tile
edge reads real neighbours instead of whatever page sits beside it in the atlas.
P27.4 filters too — PCF, and the receiver's normal-offset bias displaces the
lookup by up to `ShadowSettings::normal_bias` (default **2.0**) texels before the
kernel's own ±1 — so the naive transfer of that reasoning asks for a border of at
least 3, rounded to 4.

**The transfer fails, and the reason is P27.3 rather than P27.4.**

A virtual texture's border is baked **once, at cook**, out of the neighbouring
texels of the same mip. A shadow page's border cannot be: its content is
*rasterized*, so a border ring means widening the page's viewport and drawing the
casters that fall in the ring. That is cheap in itself. What is not cheap is what
it does to invalidation:

> A page's border texels are a function of the casters in its **neighbours'**
> footprints. So a cached page must be re-rasterized whenever any of its **eight**
> neighbours' content changes.

Phase 27's own "Done when" says *"moving one object invalidates exactly the pages
its bounds touch"*. With a border, "exactly the pages its bounds touch" becomes
those pages **and their neighbours** — a 3 × 3 dilation of every invalidation, so
a mover that touches one page costs **9**, and P27.3's page-exact invalidation
clause would be false by construction rather than by defect. The engagement
counter would be measuring a number nine times the one the clause names.

Against that, the border's cost side is measurable and small, and the memo
records it so the decision can be reversed on evidence:

| geometry | page bytes | ring as a fraction of the page | pages in 64 MiB | atlas layout |
|---|---|---|---|---|
| 128 + 0 | 65 536 | 0 % | **1 024** | 32 × 32, nothing wasted |
| 128 + 4 | 73 984 | **11.42 %** | 907 (**903** usable) | 21 × 43 — 907 is prime, so four more are lost to the rectangle |

So the border costs **121 pages of 1 024 (11.8 %)** and buys one table lookup per
PCF tap. It costs a 3 × 3 invalidation dilation and buys nothing P27.4 cannot
have another way.

## What P27.4 pays instead

Per-tap page-table resolution at page edges: each PCF tap resolves its own page
through the same indirection read the centre tap does. That is up to 9 storage
reads per shadow sample instead of 1 — and, unlike a border, it is **exact** at
a page boundary where the two pages are at different clipmap levels, which a
border ring cannot represent at all (the ring would have to hold texels from a
level the page does not belong to).

The alternative P27.4 may prefer, and which this ruling does not foreclose, is a
**clamped kernel**: taps that leave the page are dropped and the weight
renormalized. Cheaper, slightly softer at page seams, and measurable against the
per-tap resolve on goldens when the receiver exists.

## When to revisit

**P27.3 first, and this was missing from the list the P27.1 audit read.** The
argument above is *P27.3's* — its clause is the one a border would make false by
construction — so the person who most needs to know a border was rejected is the
one implementing page-exact invalidation, not the one implementing the filter. If
P27.3's engagement counter ever reports a mover invalidating nine pages where its
bounds touch one, this document is where the reason lives, and re-adding a border
is what would put it there.

> **P27.3 landed, and the ruling held exactly as written (2026-08-13).**
> `a_mover_invalidates_exactly_the_pages_its_bounds_touch` asserts the count is
> the pages the mover's own sphere enters — computed independently through the
> cull's `vsm_page_sees_sphere` — with **no** dilation, and
> `a_carve_invalidates_only_the_pages_the_carved_tile_touches` says the same for a
> terrain carve. Both would have to be `3 × 3` statements with a border on, and
> both assert the untouched pages are **bit-identical** afterwards, which a
> bordered page's neighbours could not be. `VSM_PAGE_BORDER` is still 0 and
> nothing in P27.3 wanted it otherwise.

> **P27.4 landed, and the measurement it was asked for is made (2026-08-13).**
> The answer is the **clamped kernel**, and it is in
> `docs/memos/p27-4-receiver-filtering.md` §1 with four numbers: a tap leaves
> its page for **508 of 16 384** texels (3.10 %); the clamped kernel costs
> **one** table resolution against **nine**, and a clipmap resolution is a
> walk, so **8** level-record reads against up to **72**; the clamped
> kernel is **exact** wherever the shadow field is locally constant and off
> by at most the dropped weight (3/9 at an edge) inside a penumbra; and the
> per-tap resolve over an **absent** neighbour is wrong by that same 3/9 in
> the LEAK direction on uniformly shadowed ground, which is the case the
> clamped kernel is exact in. This memo's border rejection therefore costs
> P27.4 nothing it wanted: the receiver pays one resolution per sample and a
> softer seam on 3 % of texels, not the nine-storage-read tap this document
> priced. It also costs a **sampler**: with no border there is nothing
> correct for hardware 2 × 2 comparison filtering to read at a page edge —
> the atlas neighbour is slot `s + 1` — so the receiver `textureLoad`s
> integer texels, which is the one consequence this memo did not predict.

Then **P28.2**, where interleaved cluster pages change
what a page *is*. The numbers above are all in `inf-vsm`'s own arms
(`the_ruled_page_geometry_is_the_one_the_memo_measured`,
`the_default_atlas_is_square_and_spends_its_whole_budget`,
`the_rejected_bordered_geometry_costs_what_the_memo_says`), so re-deciding is a
matter of re-reading them rather than re-deriving this document.
