# P28.1 — the visibility buffer: the packing, the gradients, and the MSAA ruling

**Status:** decided 2026-08-14, during P28.1. Five rulings, each with the
measurement that decided it. The ROADMAP's clause 3 asks for the MSAA ruling
"decided by goldens and frame budget, not taste"; §5 is that, and §1–§4 are the
decisions the pass had to make to get there.

---

## 1. The bit split, and the ceilings it was designed against

One `R32Uint` texel names a triangle. Thirty-two bits, three fields:

| field | bits | range | the measured ceiling it was designed against |
|---|---|---|---|
| triangle | 7 | 0..=127 | `BuildParams::max_triangles` defaults to **124**, and every cooked `.inf_vmesh` in the tree carries that. `meshopt`'s hard cap is 512, so 128 is a *refusal boundary* rather than a format one. |
| meshlet | 14 | 0..=16 383 | a slot in the **shared pool**, not an asset-local id, and **not the DAG's size** — see the correction below. 16 384 slots is 1 MiB of 64-byte records. |
| instance | 11 | 0..=2 046 | `RenderScene::vgeom_instances`. `vgeom-demo` places **324** (an 18 × 18 grid). |

### The meshlet ceiling, restated against the quantity the door reads (P28.1 audit)

The first version of this table justified 14 bits with *"the flagship `vgeom-demo`
DAG is ~500 slots, so 16 384 is 32×"*. That measures the wrong thing.
`VisState::admit` is handed `pool_bytes = self.pools.sizes[1]` — the meshlet
pool's **capacity**, which the streamer grows toward `VgeomStreamBudget::
budget_bytes` as pages arrive — and divides it by `MESHLET_REC_LEN`. The DAG of
any one asset never enters the arithmetic.

So the real headroom is a fraction of the streaming budget, and it is measurable.
Meshlet descriptors are a stable **5.26 – 5.44 %** of a cooked asset's pool bytes
(measured over 16², 48² and 96² displaced grids — 13, 118 and 468 meshlets), the
rest being vertices, vertex-index lists and micro-indices. One MiB of descriptors
is therefore about **18.4 – 19.0 MiB of resident meshlet pool**, against a
`DEFAULT_VGEOM_BUDGET_BYTES` of **256 MiB** — so the visibility path refuses at
roughly **7 % of the default budget**, not at 32× the flagship. In meshlets that
is ~16 000 resident descriptors, which at 124 triangles each is ~2 M resident
triangles; the flagship's *source* count is 10 M and its resident set is far
smaller, which is why the demo is admitted comfortably and why the ceiling is
nonetheless reachable by ordinary streamed content.

That is not a defect — the refusal is typed, counted and falls back — but the
sentence justifying the field has to name the pool, because that is what the door
reads. Widening the field is the frame-derived split below, routed to P28.3.

The instance field stores `instance + 1`. That bias is what makes `VIS_EMPTY`
(zero) **unreachable from a real fragment**, so "cleared" and "no geometry here"
are one statement rather than two that have to be kept in step. It costs one
instance. The alternative — clearing to `u32::MAX` — costs the all-ones
combination, which is a legal triangle of a legal meshlet of a legal instance and
would have to be refused *at pack time, inside the hot loop*.

**Overflow is a typed refusal at registration** (`VisPacking::admit`, the P27.1
grid-overflow law), naming both the measurement and the ceiling, and split into
three counters on the P27.5 ruling that a single number would wear both names. A
refused frame is not a failure: it renders through the forward meshlet raster,
which P28.1 keeps precisely so that there is somewhere to fall back to.
`a_scene_past_a_ceiling_falls_back_to_the_forward_path_and_says_which` asserts
the counter *and* that the refused frame is byte-identical to the forward frame —
a refusal that dropped the geometry would satisfy every counter.

### The alternative, measured and not taken

A **frame-derived** split — `meshlet_bits = ceil(log2(resident_slots))`, the
remaining 25 − that going to instances — is strictly more capacious: the
`vgeom-demo`'s ~500 slots would leave 16 bits for instances (65 535) instead of
11. Not taken because it makes the layout a per-frame value rather than a
constant: the shaders would read a uniform per unpack, the arms could no longer
be a compile-time mirror (`the_shaders_bit_split_is_the_rusts` reads the WGSL
source for four `const`s), and the refusal would move out of registration and
into the frame. **Routed to P28.3**, where one streamer decides pool capacities
and therefore knows the answer before a frame starts.

---

## 2. The gradients are solved, not sampled — and they are the better number

A fragment shader gets `dpdx` by differencing against the pixel next door inside
a 2 × 2 quad. A resolve pass cannot use that: the pixel next door is a *different
triangle*, so the difference is a step across a silhouette and the mip it
justifies is nonsense — one wrong tile per edge pixel, every frame.

So `vis_bary` solves. With `M = [c0.xyw | c1.xyw | c2.xyw]`,
`mu = M⁻¹ · (ndc.x, ndc.y, 1)` is **linear** in the pixel, so its NDC gradients
are two *columns of `M⁻¹`* — nothing is differenced anywhere — and the
perspective-correct weights `lambda = mu / sum(mu)` follow by the quotient rule.
The per-pixel step is `2/width` across and `-2/height` down.

The same solve recovers the **rasterized depth** rather than storing it: `M·mu =
(ndc.x, ndc.y, 1)` holds exactly, so the reconstructed clip point is
`(ndc.x, ndc.y, dot(mu, z), 1)` and its ndc depth is `dot(mu, z)` — which is the
screen-linear interpolation of `z/w` the rasterizer performs. Written as
`@builtin(frag_depth)` into the MSAA depth, so translucency, water and the shadow
marking pass find meshlet depth where they always have.

**Measured against the forward path's `dpdx`**: on the textured parity row,
**8.79 %** of interior pixels differ, worst step **26** of 255. `dpdx(uv)` is a
first-order finite difference *shared by four pixels*; the resolve's is exact and
per pixel. They pick the same mip except at a level boundary, and a different mip
is a different texel.

### The exception, bounded as a class (P28.1 audit)

"8.79 % of pixels differ" is satisfied equally by a mip **boundary** — a curve
one or two pixels thick where the footprint crosses a power of two — and by a
broad smear over whole triangles, which is what a wrong gradient actually
produces. A population bound cannot tell them apart, so
`parity_textured_virtual_texture` now measures the *shape*: **431 of the 451
differing pixels (95.6 %) border an agreeing interior pixel, and ZERO of them are
the centre of a solid 3 × 3 block of disagreement**. It is a boundary set.
Falsified rather than assumed: with the population bound disabled and the uv
x-gradient scaled 4×, the same measurement reads 0 % bordering and 1 320 solid
3 × 3 centres, and the class assertion is what fails.

### "The resolve's number is the better one" — WITHDRAWN on measurement (P28.1 audit)

The first version of this section ruled that the resolve's mip is the better one
and cited no measurement of it. The audit built the one the claim needs — a **16×
supersampled** forward reference (1280 × 720, box-downsampled to 320 × 180 in
linear space) — and it does not support the ruling:

| over 5 132 interior pixels | forward | resolve |
|---|---|---|
| mean abs. error vs the reference | **0.065542** | **0.065580** |
| on the 420 differing pixels only | **0.069580** | **0.070049** |
| differing pixels the path is *closer* on | 52.4 % | **47.6 %** |

A coin flip, with the sign marginally against the resolve. Nor could that design
have found a small effect, and the control says so: running the identical
comparison with **no texture bound at all** — pure resolution-dependent shading,
AO, specular — gives a floor of **0.0868**, larger than the whole textured signal
and roughly 200× the difference between the two paths. Repeating it on the
isolated texture contribution (textured minus untextured, which cancels the
common shading) gives 51.2 %: the same coin flip.

So the claim is demoted to the half that **is** measured, and it is a statement
about the *gradient*, not about the image:

* the resolve's gradient is the exact analytic derivative — asserted by
  `the_analytic_gradient_is_the_limit_of_the_finite_difference` as a convergence
  signature, with an inline control that drops the `2/width` step and must
  disagree by 400×;
* the forward path's is a first-order difference quantized to a 2 × 2 quad;
* **which of the two mips yields a better image is undecided by measurement**,
  and settling it needs a fixture whose reference is not dominated by
  resolution-dependent shading — a flat, unlit, texture-only surface, which this
  suite does not have. Routed to P28.3 with the streamer's own quality work.

The `p26-4-feedback-mechanism.md` item-3 win the first version invoked is a claim
about the *feedback* signal — one level per screen region instead of one per
surface — and that one **is** measured, at 22 resident tiles against 32
(`the_per_fragment_signal_is_finer_and_a_hidden_surface_adds_nothing_to_it`). It
was being spent twice.

**A symmetry worth recording**: transposing `d_dx` and `d_dy` is invisible to
every arm here, and always will be, because `vt_mip` takes
`max(length(dx), length(dy))`. Magnitudes pick the level; their assignment to
axes does not, because this VT path has no anisotropic filtering. The mutation
that *does* bite is a gradient **scale**.

---

## 3. The instance table is a texture, because eight is eight

`Limits::default()` grants **8 storage buffers per shader stage**. The resolve is
a FRAGMENT pass that binds the shared environment group, which already spends
four of them — GI SH probes (binding 5), the VT indirection table (16), the VSM
page table and its projections (18, 19) — and the four meshlet pools are the
other four. A fifth storage binding for the flat instance table is nine against
eight, and the device says so:

> Too many bindings of type StorageBuffers in Stage ShaderStages(FRAGMENT),
> limit is 8, count was 9.

This is the P26.3 scarcity ruling one binding class over: there the scarce
resource was the bind *group* and the answer was to fold; here it is the storage
binding and the answer is to change the resource *kind*. The table rides an
`Rgba32Float` texture, one row an instance, sixteen texels wide so a row is
exactly `COPY_BYTES_PER_ROW_ALIGNMENT`. `textureLoad` is valid in the vertex,
fragment and compute stages, so all three visibility passes read one
representation. `filterable: false`, because nothing samples it — asking for
`FLOAT32_FILTERABLE` to buy a convenience nothing uses is the one-platform class
the P25 audit named.

**The remaining headroom is zero**, and that is stated rather than discovered:
`the_resolve_spends_every_fragment_storage_binding_the_default_limit_grants`
counts them, so the next binding the environment group grows fails a test here
instead of failing `create_pipeline_layout` on a user's machine.

**And that arm did not do it** (P28.1 audit). Its first draft asserted
`4 + 4 == 8` with both fours written as literals, under a comment claiming they
were "read off `passes::mod`'s own constants"; they were not. Mutation-measured:
turning the environment group's binding 6 from a uniform into a storage buffer —
the exact growth the arm exists for — left it green, and the failure would have
arrived as a `create_pipeline_layout` refusal on a user's machine, which is what
the arm's own doc says it prevents. Both entry lists are now hoisted into
functions (`passes::env_bgl_entries`, `passes::visbuffer::resolve_bgl_entries`)
and the arm filters them for FRAGMENT-visible storage buffers. Same mutation now
kills it.

**A precision hazard the first write-up named backwards.** `Rgba32Float` is an
exact 32-bit container and the three shaders `bitcast` the id texel rather than
converting it — right, but the reason given was "an f32 round-trip would be a
DIFFERENT handle for anything past 2^24", which is where the danger is *not*. A
VT slot is `handle + 1`, a small integer, and **every u32 below 2^23 has an
all-zero f32 exponent field** — as a float, every real handle in this table is
*subnormal*, and WGSL permits an implementation to flush subnormals to zero.
Converting would turn slot 1 into slot 0, i.e. "this surface binds no texture":
a silently untextured frame, not an error. The bitcast is a reinterpretation of a
loaded texel and takes no arithmetic, which is why it survives;
`parity_textured_virtual_texture`'s closing `assert_ne!` is what witnesses it on
whatever adapter runs, because a flushed slot makes the textured frame
byte-identical to the untextured one. Corrected in all three shaders.

---

## 4. Screen-space material bins are the id itself

The ROADMAP's clause 2 asks to "shade in screen-space material bins". The bin is
the visibility id: a `u32` is constant across a triangle, so every fragment of one
triangle takes the same branch, and the divergence a resolve pays is bounded by
the number of distinct triangles in a wave — which is what binning buys in a
tiled implementation.

A second pass sorting pixels into **per-material indirect dispatches** is the
classic refinement, and it is measured-and-not-taken: materials here are
per-instance *constants* (colour, metallic, roughness, three VT slots), not
per-instance shader permutations, so there is exactly **one** lit program for
every meshlet instance in the engine. Every bin would name the same pipeline, and
the sort would buy a re-bind of nothing. It becomes real the day a meshlet
instance can carry a *generated* material shader — the P7.2 graph path, which
reaches the thumbnailer and the preview and has never reached the interactive
meshlet raster.

---

## 5. THE MSAA RULING: a setting, not the High-tier default

The ROADMAP: *"VisBuffer+TAA becomes the High-tier default or stays behind a
setting — decided by goldens and frame budget, not taste."*

**Ruling: it stays behind a setting.** `VgeomSettings::visbuffer` is `false` on
every tier and `RenderTier::apply` clears it on Medium and Low explicitly. Three
measurements decided it, in this order.

### (a) Goldens

The visibility buffer is **single-sample by the ROADMAP's own clause 1** and
`SCENE_SAMPLES` is a compile-time `4`. Those are not reconcilable by
configuration: a fullscreen resolve shading from a 1× id buffer writes one colour
to all four samples of its pixel. So the mode changes what a meshlet frame looks
like, and turning it on by default would re-bless committed goldens. Goldens stay
**54**, count and content digest, and `git diff` over `tests/goldens/` across this
batch is empty. A default that re-blessed frozen goldens to ship a mode whose
quality recovery (TAA) is *itself* off by default is not a default.

### (b) What is given up, measured

Two geometric cases, both obligations rather than defects, both measured by
`tests/visbuffer_parity.rs`:

* **Silhouettes.** On the parity fixture, **58.9 %** of covered pixels are edge
  (a pixel whose four neighbours do not all carry its id), and **1 275** of them —
  **2.21 %** of the frame — differ between the two paths. The forward path
  resolves four samples of partial coverage; the resolve writes one colour to all
  four. The mirror holds on the empty side: **128** of 45 274 pixels the buffer
  calls empty differ, all one step off a silhouette, because a pixel whose
  *centre* no meshlet covers can still have samples that one does.
* **Intersection curves.** Where a meshlet surface passes through another
  object, a 4× MSAA depth buffer resolves the intersection **per sample** and a
  single-sample reconstruction cannot. Measured: a rigid slab reaching into a
  meshlet grid's own ±0.66 m displacement made the two paths disagree by up to
  **104 of 255** on **thirteen** pixels. This is the second thing a single-sample
  visibility buffer gives up and it is not the one anybody names first.

Everywhere else the two agree **to the byte**: on 4 320 – 6 465 interior pixels a
row, across untextured, per-instance-set variation, masked alpha,
directional+point+spot, GI on, sun-shadowed and interleaved-with-rigid, the worst
channel step is **1 of 255** and 0.06 % – 0.74 % of pixels sit on a rounding
boundary. `PARITY_MAX_STEP = 1` forbids two.

### (c) Frame budget

Measured on this adapter, 1280 × 720, sixty frames after ten warm-up, occlusion
off (**wall-clock conditions on adapter** — the house law; these numbers are this
machine's and are quoted as a ratio, not as a budget):

| fixture | forward | visbuffer | delta |
|---|---|---|---|
| 64 instances spread over an 8 × 8 × depth grid (low overdraw) | 1.161 ms | 1.325 ms | **+14 %** |
| 64 instances stacked in depth, each filling the frame | 0.199 ms | 0.173 ms | **−13 %** |

The direction is exactly what the architecture predicts — the visibility path
adds a raster pass, a fullscreen resolve and a compute dispatch, and pays for
them by shading each pixel **once** — but neither result is decisive, and the
second fixture is a weaker overdraw test than intended (LOD selection and frustum
culling thin the drawn set at depth, so the stacked instances are not sixty-four
full-screen shading passes). **A mode that is 14 % slower on one frame and 13 %
faster on another, and that gives up silhouettes and intersections to get there,
does not become a default on this evidence.**

**Which leg is load-bearing, stated so a stronger fixture cannot be read as
overturning the ruling** (P28.1 audit). It is (a). The visibility buffer is
single-sample by clause 1 and `SCENE_SAMPLES` is a compile-time 4, so turning the
mode on changes what a meshlet frame looks like and re-blesses fifty-four
byte-frozen goldens — and it does so to ship a mode whose quality recovery is off
by default. That is decisive on its own, at any frame budget. (b) says what the
change costs in quality and (c) says the speed does not buy it back; a rerun of
(c) on a genuinely overdraw-bound fixture could reverse (c) and would still leave
the ruling standing until TAA is on by default. The memo says so here rather than
leaving a future reader to infer it from a table.

### (d) What the frame budget above is partly paying for, named (P28.1 audit)

Two costs the first version of this memo folded into the wall-clock numbers
without naming, because both are structural rather than incidental:

* **The resolve writes `@builtin(frag_depth)`, which forbids early-Z in that
  pass.** It is a real depth-tested, depth-writing pass into the MSAA scene depth
  (`DEPTH_COMPARE`, `depth_write_enabled: true`), and a shader that computes its
  own depth cannot be rejected before it runs. The `discard` for `VIS_EMPTY` is
  at the top of the fragment, so a pixel with no meshlet costs a texel load and
  nothing else — but a pixel *with* one pays the whole lit program before the
  comparison happens.
* **So the visibility path shades pixels that lose to NON-meshlet geometry.** The
  visibility buffer has a depth of its own and knows nothing about the rigid
  pass; a meshlet fragment behind a rigid wall still owns its texel, is still
  shaded by the resolve, and is then thrown away by the depth test. Deferred's
  "shade each pixel once" holds here against meshlet-vs-meshlet overdraw only.
  `parity_interleaved_with_rigid_geometry` proves the *result* is right — the
  cube's pixels agree to the byte — and nothing measures the waste. A depth
  pre-test against the resolved scene depth is the fix and it is P28.3's, where
  the prepass and the visibility buffer stop being two independent depths.

### What would change the ruling

Three things, and P28.1 built none of them:

1. **TAA on by default**, which is the quality recovery the clause's own phrase
   ("VisBuffer+TAA") assumes. It is off for headless determinism and that is a
   separate decision with fifty-four goldens in it.
2. **A frame where deferred shading actually wins** — real overdraw with an
   expensive material, which this engine does not have while every meshlet
   instance runs one lit program (§4).
3. **P28.3's unified streamer**, which is where the visibility buffer stops being
   an alternative shading path and starts being the thing the residency brain
   reads. The cost side of the ledger is not measurable until then.

---

## 6. What P28.1 did NOT close, precisely

* **`voxel.wgsl` still has no shadow receiver, and P27.5's routing to this pass
  does not survive it.** The visibility packing's meshlet field is a slot in the
  shared meshlet pool and a voxel chunk — a Surface-Nets mesh in its own buffers,
  no meshlet structure, no DAG, no page — has none. The resolve cannot shade what
  the packing cannot name, and all thirty-two bits are spent
  (`the_visbuffer_id_space_has_no_room_for_a_second_geometry_kind`). The two real
  doors are **meshletizing voxel chunks** (P28.2, and it closes the three other
  gaps `voxel.wgsl`'s header names — casting, GI, the prepass) and **the env
  group alone** (cheap, and refused for the reason P27.5 gave, which P28.1 did
  not weaken). Corrected in the shader's own header.
* **Occlusion recovery is structural, not measured at residency level.** A
  surface that owns no pixel cannot be marked — the producer reads the id the
  depth test left. But residency is `floor ∪ feedback` and `analytic_floor` is
  deliberately occlusion-blind (that is what makes a dropped mask degrade
  safely), and at the tested viewport its `VT_FLOOR_MAX_TILES` cap already covers
  the level `justified_mip` names, so the per-surface feedback has almost nothing
  extra to waste on a hidden surface: both paths grow by the floor's one tile.
  Isolating it needs **per-consumer want accounting**, which is P28.3's unified
  ring.
* **The tangent channel is untouched**, and P26.5's routing to P28.2 stands
  unchanged. The resolve computes no tangent: it calls `vt_apply_normal` with the
  analytic world-position and uv gradients, which is the same per-fragment
  cotangent frame the forward path builds — better-conditioned inputs, the same
  construction. So P28.2 still owes the whole vertex-level channel:
  `VgeomVertex` is position + normal + uv and has no tangent to give, and the
  skinned path is still at `max_vertex_attributes: 16` exactly.
* **The visibility path does not reach the picker, the depth prepass, the HZB or
  the shadow caster passes.** It is an alternative *shading* path for the meshlet
  raster and nothing more; every other consumer of meshlet geometry is unchanged.

---

## 7. What the P28.1 AUDIT changed, in one place

* **Three cited arms had never existed** — `the_resolves_gradients_match_the_
  devices` (in `visbuffer.rs`), `the_per_pixel_mark_uses_the_same_mip_rule` (in
  `vis_feedback.wgsl`) and `the_visbuffer_edge_cost_against_the_forward_path` (in
  `passes/visbuffer.rs`). The P20 law, three times, in the batch that caught it
  once in P27.5's ledger and wrote the finding up. The second has been **written**
  (the LOD rule is pinned line-for-line between `vt_sample.wgsl` and
  `vis_feedback.wgsl`), the third **renamed** to the arm that does the work, and
  the first **withdrawn**: there is no device-vs-twin gradient comparison, because
  the id names a shared-pool slot and the slot→asset remap lives on the GPU, so a
  host cannot decode a readback id into three vertices. The twin is a mirror, and
  the doc now says so.
* **Two arms could not see their own subject.** The storage-binding ceiling
  (above) and `the_pool_ceiling_converts_bytes_to_slots_at_the_record_length`,
  which re-implemented `VisState::admit`'s division locally — so replacing
  `MESHLET_REC_LEN` with a wrong constant inside `admit` left it green. One
  conversion function now, called by both.
* **The refusal's fallback arm did not falsify.** `a_scene_past_a_ceiling_falls_
  back_to_the_forward_path_and_says_which` asserted `refused.rgba ==
  forward.rgba`, and both frames take the *same* forward draw call — so deleting
  that call leaves them identically empty and the arm green, which is precisely
  the "a refusal that dropped the geometry would satisfy every counter" failure
  the ledger claims it rules out. Mutation-measured. It now compares the refused
  frame against a **geometry-free** frame and requires 5 % of the frame to differ.
* **The degenerate guard was NaN-blind.** Every comparison against NaN is false,
  so `abs(det) < VIS_DET_EPS` let a NaN vertex through the guard and out as
  `λ = (NaN, NaN, NaN)` — from a function whose doc said a degenerate case
  resolves "rather than to a NaN". `|| det.is_nan()` in the twin, `|| det != det`
  in both shaders, and an arm that sweeps a NaN through all twelve clip
  components. Also recorded: the floor is a **finiteness** bound and not a
  conditioning one — a sliver at `ε = 1e-5`, fifteen orders above the floor,
  already gives a gradient of 33.6 per pixel and weights of `(-0.40, 0.80, 0.60)`.
* **The parity oracle's independence is now pinned, and its boundary measured.**
  The two paths share the whole composed `Lit(2)` prelude — `vt_sample`,
  `vsm_receive`, the environment lighting, the atmosphere — and an error in there
  is invisible to all twelve arms: mutation-measured, `+ 1.0` on `vt_sample`'s
  `vt_mip` (a wrong mip for the entire engine) leaves the suite green. What is
  independent is the vertex pull, the barycentrics, the gradients and the BRDF,
  and `the_resolve_derives_its_own_shading_rather_than_borrowing_the_forward_
  paths` fails the day one of those is de-duplicated into the prelude.
* **The textured exception** — classified (§2) and its "better" ruling withdrawn
  (§2).
* **The meshlet ceiling** — restated against the pool capacity the door reads
  (§1).
