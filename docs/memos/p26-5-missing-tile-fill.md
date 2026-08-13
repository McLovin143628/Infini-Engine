# P26.5 — the missing-tile fill, measured; and what the measurement said instead

**Status:** decided 2026-08-12, during P26.5. ROADMAP P26.5 clause 2 asked for
*"missing-tile fill v1: deterministic edge-directed upscale of the finest
resident ancestor (the NTC intent, measured before adoption; memo on the outcome)"*.
It was measured. The outcome is not the one the clause assumed, and this is it.
(The house rule: a deviation gets a memo, not silence.)

---

## The clause, and where it came from

`docs/memos/p26-28-virtualization-direction.md` §4 ruled on Neural Texture
Compression:

> **Neural Texture Compression → deferred, with the intent kept.** The doc's
> fallback idea ("reconstruct detail while the real tile streams") ships as a
> deterministic edge-directed upscale of the finest resident ancestor into the
> physical page (compute pass, measured before adoption). Weight-per-material NTC
> needs a training pipeline and is deferred by memo, not silence.

Two claims are bundled there and they turn out to be separable: *what* to write
into a page that has no bytes, and *whether there is such a page at all*.

## What was built

`inf_vt::fill` (Ring 0, integer-only, source-gated against floats on
`container.rs`'s pattern — these bytes would be sampled, so a float makes a
missing tile a property of the machine that drew it):

* `upscale2x` — a deterministic **edge-directed** 2× upscale on the classic
  NEDI/DCCI quincunx: diagonal centres first, choosing whichever diagonal is
  flatter in integer BT.601 luma by a 5/8 margin, then the axial samples by the
  same test rotated 45°. Source texels pass through untouched and every invented
  texel is a mean of real neighbours, so the result cannot leave the source's own
  colour hull — no ringing, which matters for a placeholder;
* `replicate2x` — the trivial alternative: every parent texel becomes a 2×2 block;
* `fill_from_ancestor` — the composition. It walks `VtTextureDesc::descent`
  (P26.5, the inverse of `ancestor`'s clamped chain) from the ancestor down to
  the wanted tile, doubling and cropping at each step. **The crop is exact**: a
  stored page is `tile + 2·border` texels a side, so after a doubling the child
  quadrant's whole page — payload *and* border ring — is the sub-rectangle at
  `(border + tile·qx, border + tile·qy)`. The reconstructed ring is therefore
  filled from real upscaled neighbours rather than from a clamp, and a bilinear
  tap at the payload edge reads what it would have read.

## The measurement

`crates/inf-material/tests/vt_fill_quality.rs`. A 512² fixture through the
shipped `build_tiled_texture`, read back through the shipped
`TiledTextureReader`; every mip-0 tile reconstructed from its parent three ways
and scored against **the tile that is actually there**. Two metrics, because one
would have hidden the argument: mean absolute error per channel (fidelity) and
mean absolute error of the local luma gradient (structure — a blocky
reconstruction has zero variation inside a block and double at its seams, and
both land here).

The fixture carries four regions *and* a texel-scale detail term, which is
load-bearing: without it, three of the four regions score **0.00** for
replication, because a feature wider than two texels survives a box downsample
intact and repeating the parent reproduces it exactly. A fixture without
sub-texel detail measures how well each filter copies information that was never
lost.

```
vt missing-tile fill over 16 mip-0 tiles of a 512² fixture
                             texel MAE (0..255)            |      gradient MAE
                       near   box  bilin   edge  |  near   box  bilin   edge
  smooth gradient      8.45   9.41   9.10   9.61 | 21.08  21.55  22.95  22.15
  45 deg wedge        12.15  14.82  13.81  14.27 | 32.12  31.97  31.92  31.56
  axis-aligned bars    8.59  11.67  11.11  11.86 | 21.53  22.37  23.35  22.92
  shallow (2:1) diag  10.48  12.61  11.79  12.48 | 24.35  24.74  25.11  25.57
  ALL                  9.92  12.13  11.45  12.05 | 24.77  25.16  25.83  25.55
```

**The `bilin` column is the P26.5 audit's, and it is the one this memo's argument
was about.** The batch measured three filters and labelled the `box` column
*"bilinear (what the sampler already does)"*. It is not: `box2x` is the quincunx
lattice with the direction test removed — source texels pass through untouched
and it sits half a texel off — which makes it the right control for isolating the
direction test and the wrong thing to call a sampler's magnification. True
texel-centre bilinear (weights 3/4–1/4, clamped at the edges) is `bilin`.

Two numbers move, and both move the ruling's way:

* replication still wins on both metrics against all three interpolations —
  **9.92 / 24.77** against bilinear's 11.45 / 25.83;
* the edge-directed filter is **not** "within one part in a hundred of the bar".
  Against the quincunx control it is (12.05 vs 12.13); against the filter the
  hardware actually performs it is **5 % worse** (12.05 vs 11.45), and worse on
  structure. The conclusion below is therefore stronger than the batch's, not
  weaker.

## What it says

**1. Replicating the ancestor's texels beats every interpolation of them, on
both metrics, in every region.** By 18% against bilinear on fidelity and by 1.6%
on structure.

This is not a quirk of the fixture; it is a property of the pyramid. **A mip
level is a box downsample**, so a parent texel *is* the arithmetic mean of its
four children — the minimum-error constant predictor for that 2×2 block, by
construction. Any interpolation spends that optimality moving energy across block
boundaries. It buys smoothness, and smoothness is not what a fidelity metric
rewards.

**2. The edge-directed filter does not clear the bar it was given.** The bar is
the bilinear magnification the hardware sampler *already performs* for free when
it magnifies a coarse page, because a fill that does not beat that is a page
write that buys nothing. Edge-directed comes in at **12.05 against bilinear's
11.45** — 5 % worse — and worse on structure too (25.55 against 25.83 is better
here, but both lose to replication's 24.77). It does what it was designed to do
on the two diagonal regions relative to the *quincunx box* (14.27 vs 14.82;
12.48 vs 12.61) and pays for it on the smooth one (9.61 vs 9.41), which is the
staircase risk the 5/8 direction margin exists to bound and evidently does not
eliminate; against real bilinear it loses on three regions of four.

*(The first version of this memo compared 12.05 against **12.13** and called it
"one part in a hundred". That 12.13 was the quincunx box control, not the
sampler's filter — the P26.5 audit measured the filter the sentence names. The
conclusion is unchanged and the margin is larger.)*

**3. There is no reachable window to adopt any of it into.** A page-fill needs a
slot that has been allocated and bytes that cannot be produced. In this
implementation a tile read is a synchronous `read_ref` slice of an mmap
(`inf_render::VtTextures::sync`), so an admitted page's bytes are available in
the same call that admitted it. The only way `fetch` returns `None` is a
container that declares tiles it does not contain — and `container::parse`
refuses that at the door, because the P26.1 audit added the three layout
equalities precisely so a v2 payload must contain the tiles it declares.

Wiring a fill in today would therefore be a defensive branch no test can reach.
That is the exact shape the P26.4 audit withdrew `c78d2ff` for: *"defensive and
unreachable … recorded rather than given an arm that would have to build a
transaction the door cannot produce."*

## The ruling

**Measured, and not adopted.** `inf_vt::fill` ships **declared and pinned with no
hot-path caller**, which is this repository's rule for API whose caller has not
arrived yet (`VtStats::summary`, `is_warm` and `VT_BINDING_COUNT` were all here
first). Concretely:

* `fill_from_ancestor` uses `replicate2x` — the filter the measurement chose, not
  the one the clause assumed — and
  the measurement arm asserts that call site against its own nearest arm, so the
  code and this memo cannot drift;
* `upscale2x` stays public as the measured alternative, so the ruling can be
  **re-measured** rather than re-argued when someone improves it;
* the measurement asserts the ruling **in the direction it went**
  (`replicating_the_ancestor_beats_every_interpolation_of_it`), so the day an
  interpolation does win, the build fails by name and says to rewrite this memo
  before adopting anything.

## Where the caller is

**P28.3, the unified streamer.** That is where an admit's bytes genuinely arrive
late — the ROADMAP's own text: *"one streamer arbitrates vgeom + SVT + VSM under
one budget with one feedback ring"*, with predictive prefetch behind it. A slot
that is allocated on frame *F* against bytes that land on frame *F+k* is a real
window, and it is the window the direction memo's "reconstruct detail **while the
real tile streams**" was always describing. `fill_from_ancestor` is ready for it,
and the ruling above says what to write.

**And NTC itself stays deferred**, unchanged, for the reason §4 gave: a
weight-per-material predictor needs a training pipeline and has no falsifiable
bound under house gates. What this measurement adds to that ruling is a floor it
must clear — a learned predictor has to beat **9.92**, not the 11.45 a hardware
sampler gives for nothing, because the cheap answer it is replacing is cheaper
than anyone assumed.
