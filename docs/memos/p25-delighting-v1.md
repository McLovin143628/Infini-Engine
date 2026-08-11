# De-lighting v1: implemented, measured, and refusing (P25.3)

**Date:** 2026-08-11 · **Scope:** `inf_photo_gpu::finish::delight`, `DelightConfig`

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
| ambient (luma) | ≈ 0.32 | 0.253 |
| directional (luma) | ≈ 0.59 | **0.078** |
| variance explained by the directional term | — | **8.2%** |
| albedo error vs the known albedo, p50 | — | 0.068 → **0.255** |

The directional term comes back **seven times too small**, the fit explains 8.2% of the
brightness variation, and applying it makes the albedo error nearly four times worse.

The reason is visible in that 8.2%: the other 91.8% of the brightness variation on this
fixture is the **dot texture and the plane tints**. Albedo variance swamps shading
variance, the least squares has almost nothing to lock onto, and the best direction it
finds is fitting noise. Dividing by that noise stamps it into the texture.

## The ruling

**De-lighting v1 ships, off by default, and refuses this fixture.**

`DelightConfig::min_explained` — the fraction of the constant model's residual the
directional term must beat to be believed — is set to **0.25** because of the numbers
above. A threshold under 0.10 would have accepted an 8.2% fit and shipped a worse
texture than it started with. The refusal is a value, not a failure:
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
