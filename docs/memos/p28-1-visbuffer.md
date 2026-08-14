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
| meshlet | 14 | 0..=16 383 | a slot in the **shared pool**, not an asset-local id. The flagship `vgeom-demo` DAG is ~500 slots for a 10 M-triangle scene, because meshlets are stored once per asset and instanced. 16 384 slots is 1 MiB of 64-byte records. |
| instance | 11 | 0..=2 046 | `RenderScene::vgeom_instances`. `vgeom-demo` places **324** (an 18 × 18 grid). |

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
**8.79 %** of interior pixels differ, worst step **26** of 255. That is not a
defect and the arm says so: `dpdx(uv)` is a first-order finite difference *shared
by four pixels*; the resolve's is exact and per pixel. They pick the same mip
except at a level boundary, and a different mip is a different texel. The
resolve's number is the better one — it is exactly the sub-surface-variation win
`docs/memos/p26-4-feedback-mechanism.md` item 3 predicted a visibility buffer
would buy.

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
