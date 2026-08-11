# De-lighting v1: implemented, measured, and refusing (P25.3)

**Date:** 2026-08-11 · **Scope:** `inf_photo_gpu::finish::delight`, `DelightConfig`
· **Numbers re-measured 2026-08-11 by the P25.3 audit** — see "A correction" at the end.

The Phase 25 plan calls P25.3's fourth item "**optional** de-lighting v1". This memo
records what "optional" was allowed to mean, because the licence to defer runs only as
far as the measurement in hand — and the measurement is here.

## What was built

The honest minimum, exactly as specified: estimate one **global ambient-plus-directional**
illumination model from `(baked normal, observed luma)` by least squares over the
well-seen texels, and divide it out.

```
for each candidate direction L in the fixed hemisphere table:
    least squares   luma ≈ a + b · max(0, n · L)      (a 2×2 normal equation)
pick the L with the smallest residual, ties by index
apply  colour × mean(s) / s(n)     with  s(n) = a + b · max(0, n · L)
```

The divide is **mean-preserving**: de-lighting changes the *distribution* of brightness,
not the exposure, because nothing in a photogrammetric capture knows the absolute one.

It is behind `DelightConfig::enabled`, **default off**, and it works when its premise
holds — `delighting_removes_a_shading_gradient_it_was_given` builds an atlas with one
constant albedo and normals sweeping a quarter turn, and the solve removes over 80% of
the gradient's spread and reports over 0.9 explained.

## What it cannot do, and why the premise usually fails

The model removes **shading** — the smooth brightening of surfaces that face the light.
It removes nothing else: not a cast shadow (a shadow is not a function of the normal),
not a specular highlight (a function of the *view*, already averaged away by the blend),
not coloured light (the fit is on luma), and nothing a clipped highlight threw away.

It also rests on an assumption it cannot check: that **albedo and normal are
uncorrelated** over the surface — grey world. That is the assumption a real scan breaks.

## The measurement

On the committed P25.3 fixture — a textured dihedral, a raised block and a floor, each
plane a different colour, lit by an ambient-plus-directional model the renderer applied
*exactly*, so the answer is known rather than estimated:

| quantity | truth | recovered |
|---|---|---|
| ambient (luma) | 0.319 | 0.260 |
| directional (luma) | 0.600 | **0.070** |
| variance explained by the directional term | — | **6.4%** |
| albedo error vs the known albedo, p50 | — | 0.0680 → **0.0700** |
| the same, p90 | — | 0.1884 → **0.1916** |

The truths are the Rec.709 luma of `inf_photo_gpu::fixture::light()`. The directional term
comes back **more than eight times too small**, the fit explains 6.4% of the brightness
variation, and applying it makes the albedo error worse rather than better.

The reason is visible in that 6.4%: the other 93.6% of the brightness variation on this
fixture is the **dot texture and the plane tints**. Albedo variance swamps shading
variance, the least squares has almost nothing to lock onto, and the best direction it
finds is fitting noise.

**And the size of the damage is worth being exact about, because it is small.** Applying
the fit moves p50 from 0.0680 to 0.0700 — three per cent worse, not a ruined texture. That
is a *consequence* of the same defect, not a separate fact: the recovered directional term
is 0.070 against an ambient of 0.260, so `s(n)` ranges over 0.260…0.330 and the
mean-preserving divide scales texels by 0.87…1.11. A fit that has found nothing applies
almost nothing. The case against trusting it is therefore not "it will destroy the
albedo" — it is that the number it would divide out is noise, and the measured direction
of the change is the wrong one every time it has been measured.

## The ruling

**De-lighting v1 ships, off by default, and refuses this fixture.**

`DelightConfig::min_explained` — the fraction of the constant model's residual the
directional term must beat to be believed — is set to **0.25** because of the numbers
above. A threshold under 0.064 would have accepted this fit and shipped a worse texture
than it started with; the gap between 0.064 and 0.25 is the margin, and it is deliberately
wide because nothing here can tell a weak-but-real light from a strong-but-imaginary one
except the fraction it explains. The refusal is a value, not a failure:
`DelightReport::applied` comes back false, the albedo is returned untouched, and
`FinishAdvisory::DelightRefused` carries what it explained against what it needed.

The three things that would make this sound, in the order they would help:

1. **Fit on a low-frequency image, not the raw one.** Downsample the albedo hard before
   fitting, so the dot texture averages out and the shading gradient survives. This is
   the cheapest of the three and probably the one that moves the number.
2. **Fit per channel with a shared direction.** The one global scalar cannot remove a
   colour cast; three (a, b) pairs against one L can, at the cost of nothing.
3. **Use the observations, not the atlas.** The information de-lighting actually wants is
   that *the same surface point looks different from different directions* — which the
   per-view samples carry and the blended atlas has already destroyed. That is a real
   inverse-rendering step and it is a phase of its own, not a v1.

## Why this is not a deferral

The spec's "optional" would have licensed shipping nothing. What is shipped instead is a
working estimator, a guard tuned by a measurement rather than by taste, a refusal that
names its own shortfall, and this memo. The next person to raise `min_explained`'s
question has the number to argue with.

## A correction

The first draft of this memo, and the `DelightConfig::min_explained` doc comment and the
P25.3 gate's header block that quoted it, all carried **8.2% explained, an ambient of
0.253, a directional of 0.078, and an applied albedo error of 0.255**. Only the shape of
those figures survives re-measurement; the applied error does not survive at all.

The cause is an ordering: the memo and the doc comment were written in `dce0655` and
`4d6cb92`, and the mutation pass in `97777df` then **re-saturated the block's five face
tints** to make the occlusion test falsifiable. That changed the albedo variance the
directional term is competing against, and nothing re-ran the de-lighting numbers
afterwards. The gate only ever asserted `explained < min_explained`, which is true at 6.4%
exactly as it was at 8.2%, so no arm noticed.

The lesson is the older one, met again: **a measurement quoted in prose is a measurement
nothing re-derives.** The figures above are now asserted rather than remembered —
`de_lighting_refuses_this_capture_and_names_what_it_found` bounds `explained` from *both*
sides, so a fixture change that moves it materially reds the gate instead of quietly
invalidating this page.
