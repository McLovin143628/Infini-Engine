# P27.1 — shadow pages adopt the camera's reverse-Z

**Status:** decided 2026-08-13, during P27.1, as the ROADMAP's clause-4 ruling
(*"pages adopt the camera's reverse-Z unless measurement defends keeping the
forward-Z exception (memo either way) — measured, depth precision distribution
over a shadow frustum, not taste"*). The constants are
`inf_render::vsm::VSM_DEPTH_CLEAR` / `VSM_DEPTH_COMPARE`, defined **as** the
camera's rather than repeated, and asserted equal by
`a_shadow_page_uses_the_cameras_depth_convention`.

The cascaded shadow map keeps its forward-Z **unchanged**. The same arm asserts
that too: P27.5 demotes the CSM path, this batch does not edit it.

## The exception this replaces

`crates/inf-render/src/csm.rs` documents it plainly:

> **Forward-Z shadow depth:** unlike the reverse-Z **camera** projection, the
> shadow ortho uses a conventional forward-Z range (near → 0, far → 1) […]. The
> two projections are independent; keeping the shadow map forward-Z simplifies
> the bias reasoning and the comparison-sampler direction.

That was written for a directional-only, orthographic-only shadow map. P27 adds
**spot and point**, which are perspective, and rasterizes every light kind into
**one** atlas.

## The measurement

`vsm.rs`'s `the_depth_convention_is_decided_by_the_perspective_case` measures
**depth resolution in metres**: at a series of distances along the light, the
smallest world-space step that changes the stored f32 depth. Bisected rather
than differentiated, and taken **through the shipped matrix** rather than
through a formula written beside it — what a shadow map can resolve is literally
"how far must a surface move before the depth buffer notices", and a derivative
would be a statement about the algebra.

Four cases, worst case over 401 samples of each range:

| frustum | convention | worst-case resolution | where |
|---|---|---|---|
| ortho, 200 m box | reverse-Z (`ortho_reverse_z`) | **21.2 µm** | 67.0 m |
| ortho, 200 m box | forward-Z (the CSM's) | **15.2 µm** | 152.9 m |
| perspective, 0.1 → 50 m | reverse-infinite (`perspective_infinite_reverse`) | **7.43 µm** | — |
| perspective, 0.1 → 50 m | forward-Z, finite far | **1.87 mm** | 46.8 m |

### What it says

1. **The orthographic case cannot tell the two apart.** A clipmap's depth is
   linear in view distance, so one f32 step is `ulp(d) × range` whichever way
   round it runs. 21.2 µm against 15.2 µm is a ratio of **1.40** — and for
   scale, the CSM's own `ShadowSettings::depth_bias` default of 0.0015 in NDC
   *is* **0.3 m** over that range, four orders of magnitude larger than either.
   Nothing about bias reasoning is decided here.

2. **The perspective case is decided by a factor of 251.** Forward-Z's worst
   case is 1.87 mm and it sits at **46.8 m of a 50 m cone** — the far end,
   which is exactly where a spot light's shadow covers the most screen. That is
   the classic reverse-Z result and it is the whole reason the camera is
   reverse-Z; P27 is simply the first phase where a shadow projection is
   perspective at all.

3. **One correction the measurement forced on this memo's own draft.** The first
   version asserted that forward-Z is coarsest "at the far plane" and reverse-Z
   "at the near plane", and that the ortho numbers would therefore be identical.
   Both halves are wrong. f32's ULP is a step function of the **stored** depth,
   so a *linear* depth's coarsest step lands wherever the `[0.5, 1)` binade
   does — mid-box, at 67 m and 153 m, not at either end. The perspective case's
   far-end peak survives because a perspective depth is not linear. The arm now
   asserts the peaks are **not** at the ends, so the corrected claim is the one
   under test.

### Tolerances, and why they are ranges

The projections are built with `f32::tan`, which the P14 law records is **not
bit-portable** across platforms. Pinning 21.2 µm exactly would be a bound that
reddens CI on one leg — the P25 "one-platform bounds red CI" law. The arm pins
generous ranges around each number and asserts the two **ratios**, which are what
the ruling rests on and are nowhere near the tolerance.

## The third argument, which is not about precision

Even at a tie, two conventions cost something the CSM never had to pay. P27.2
rasterizes **rigid, meshlet, skinned, terrain and scatter casters for every light
kind into one `Depth32Float` atlas**, with `set_viewport`/`set_scissor` per page.
Two conventions would mean two depth-compare states, two clear values and two
receiver comparison directions, selected per light kind, inside a pass that is
already switching scissor rectangles per page. One convention is one pipeline
state for the whole atlas.

## The consequences, written down

* A page's depth **clears to 0.0** and compares `Greater` — nearest caster wins
  by holding the larger value.
* The directional clipmap uses `camera::ortho_reverse_z`, the camera's own
  function, so there is one implementation of that projection in the tree.
* Spot and cube faces use `perspective_infinite_reverse` — the camera's
  projection function too — so a light's far plane is at infinity and a light's
  `range` culls by distance rather than by a clip plane.
* A receiver's comparison in P27.4 is `depth > stored + bias`, the camera's
  direction; the CSM's `LessEqual` receiver path stays exactly as it is until
  P27.5 demotes it.

  > **P27.4 landed, and this line's SIGN was wrong (2026-08-13).** The
  > direction is right and the side is not. Under reverse-Z a larger stored
  > depth is a caster *nearer* the light, so a receiver is lit exactly when
  > its **own** biased depth still reaches at least as far:
  > `receiver.z + bias >= stored.z`. Written the other way round the bias
  > lands on the caster's side and makes a self-shadowing surface *more*
  > shadowed rather than less — the opposite of what a bias is for. The
  > shipped comparison is `inf_render::vsm_receiver::vsm_level_factor`'s and
  > `vsm_receive.wgsl`'s, and the mutation that writes this memo's version
  > back fails three arms. Corrected here and derived in
  > `docs/memos/p27-4-receiver-filtering.md`.
* **The bias story is re-derived at P27.4** and is not settled here. What this
  memo establishes is that the precision available to it is 251× better on the
  perspective lights and materially unchanged on the directional one.

  > **Done (2026-08-13):** `docs/memos/p27-4-receiver-filtering.md` §2. Two
  > terms, each derived: four ULP of `f32` — constant in NDC, and *not*
  > constant in texels, which is the whole reason for the unit — plus
  > `(R + ½)·√2` texels of the page's own density times `tan θ`, converted
  > by the projection's `∂z/∂m` read off the shipped matrix. Together they
  > are **1/665** of `ShadowSettings::depth_bias` at the shipped defaults,
  > and the perspective branch uses `z²/near` precisely because row 2 is the
  > degenerate one this memo's own convention creates.

## When to revisit

**P27.4**, when the receiver exists and the bias constants are re-derived
against page resolution; and **P28.5**, if the ray-query experiment ever needs to
compare a rasterized page against an analytic hit.
