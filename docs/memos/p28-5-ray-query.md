# P28.5 — the ray-query shadow experiment: the verdict, and the platform it was measured on

**Status:** decided 2026-08-14, during P28.5. The ROADMAP's last clause asks for
an experiment and a memo with **the measured verdict (quality, cost, coverage)**,
and it says up front what the answer may not be: *"VSM remains the shipped path
on every tier; the experiment lands behind a default-off setting with a `caps.rs`
clamp."* This memo is the verdict.

**Verdict: REFUSED as a shipped path, on coverage and cost, with quality
measured and — narrowly — in the ray query's favour.** Nothing changes about
what the engine draws. The experiment stays, armed, behind
`RaytraceSettings::sun_shadows = false`.

---

## 0. THE PLATFORM BOUND, first, because everything below is inside it

`wgpu::Features::EXPERIMENTAL_RAY_QUERY` is documented **Vulkan-only and
native-only** in the pinned wgpu 30. This tree's instance is `VULKAN | METAL`
(`inf_render::gpu::create_instance`), so:

| target | ray queries |
|---|---|
| Windows / Linux, Vulkan, RTX-class GPU | **yes** — where measured |
| macOS / Metal | **no**, by construction — the backend has no such feature |
| any software adapter (lavapipe, WARP) | **no** |
| wasm32 / WebGPU | **no** — native-only |
| Android | untested here; Vulkan, so possible on some devices |

**Every number in this memo was produced on one machine**: NVIDIA GeForce
RTX 4070 Ti, Vulkan, `DiscreteGpu`, Windows 11. The gate prints that line before
each device arm, and the P25 one-platform law is the reason it does: *say what
ran where.* On every other adapter the probe, the clamp, and the refusal arms
run and the measuring arms print a skip that names the adapter.

## 1. TWO THINGS THAT WOULD HAVE SHIPPED AS A DEFECT

Both were found by wiring the experiment the obvious way and watching the tree
fall over. They are recorded because the obvious way is what a later reader will
try again.

**(a) An `EXPERIMENTAL_*` feature cannot be requested the request-if-available
way.** This tree's standing rule is that optional features are requested only
where the adapter exposes them, so device creation can never fail for one
(`POLYGON_MODE_LINE`, `TEXTURE_COMPRESSION_BC`). Adding `EXPERIMENTAL_RAY_QUERY`
to that mask **broke device creation outright**:

```
request_device: Some experimental features, EXPERIMENTAL_RAY_QUERY, were
requested, but experimental features are not enabled
```

wgpu 30 requires the descriptor to carry `ExperimentalFeatures::enabled()`,
which is an **`unsafe`** token the caller signs to accept that the feature's
implementation may contain undefined behaviour. For one build of this batch,
every headless test in the repo skipped for "no GPU adapter" — on a machine with
a discrete GPU. **Request-if-available is not a safe rule for a feature whose
request can be refused for a reason unrelated to the adapter.**

So the shipped device does not request it, and the experiment builds its own:
`GpuContext::headless_ray_query`, the only place in the tree that signs the
token, called only by the experiment's gate. `an_acceleration_structure_over_
nothing_or_without_the_feature_is_refused` asserts the shipped context does
**not** carry the feature — on a machine whose adapter has it — so the day
somebody moves it into the ordinary mask, that arm goes red.

**(b) The four acceleration-structure limits are `0` in `Limits::default()`.**
`max_blas_primitive_count`, `max_blas_geometry_count`, `max_tlas_instance_count`
and `max_acceleration_structures_per_shader_stage`. A device with the feature
and the default limits fails at `create_blas` with *"Limit
`max_blas_geometry_count` is 0, but the BLAS had 1 geometries"* and at
`create_bind_group_layout` with *"Too many bindings of type
AccelerationStructures … limit is 0"*. Two more failures that read like driver
problems and are descriptor problems.

## 2. WHAT WAS BUILT

* `inf_render::raytrace` — Ring 0, **no new dependency**; the acceleration
  structures come through the pinned wgpu or not at all.
* `RtBlasSource::from_meshlet_level(&VgeomMesh, level)` — **the ROADMAP's
  "BLAS over meshlet clusters"**, literally: it walks one LOD level's meshlets
  and asks each for its triangles through `VgeomMesh::triangle`, over the DAG's
  own welded vertex buffer. One BLAS per level, because a meshlet is a set of
  indices into a shared buffer and not a mesh of its own.
* `RtScene::build` — one BLAS per source, one TLAS over the instances, both
  built in **one** `build_acceleration_structures` call (wgpu requires every
  BLAS a TLAS instance names to be built in that call or an earlier one).
  Geometry is flagged `OPAQUE`, so the driver commits triangle intersections and
  the shader is a visibility query rather than an any-hit program.
* `rt_sun_shadow.wgsl` — one compute thread per pixel, **two** rays: a primary
  ray into the TLAS, then a shadow ray from the hit toward the sun. Three
  verdicts (`RT_MISS` / `RT_LIT` / `RT_SHADOWED`), because "no surface here" and
  "a surface the sun can see" are different facts. Registered in
  `standalone_shaders_validate`, so **every CI leg naga-validates it with no
  device anywhere** — which matters more here than for any other shader in the
  tree, since exactly one adapter class can compile it for real.
* The probe (`GpuContext::supports_ray_query`, `AdapterCaps::ray_query`), the
  clamp (`AdapterCaps::clamp_ray_query`, composed into `clamp_occlusion`), and
  the setting (`RaytraceSettings::sun_shadows`, default false).

**Primary visibility is traced rather than read from a depth buffer.** It costs
a second trace and buys three things: the experiment depends on no shipped pass,
its coverage bound becomes explicit, and the comparison is naturally restricted
to pixels the TLAS actually has an opinion about.

## 3. QUALITY — measured twice, against two different oracles

### 3a. Against a CPU ray caster: **exact**

`a_blas_over_meshlet_clusters_traces_what_a_cpu_ray_caster_traces` compares the
device's verdicts against a Möller–Trumbore intersector written out in the test
file — no acceleration structure, no shared code, same triangles and same rays.

**0 of 36 864 pixels differ.** All three verdict classes present on both sides
(the anti-vacuity, asserted before anything is compared). This is what makes
every other number here mean something: a wrong TLAS transform convention, a
wrong index base, a transposed camera basis or a mis-set field would fail here
and cannot fail by agreement.

### 3b. Against the shipped virtual shadow map: **98.88 % agreement**

`the_ray_queried_sun_shadow_against_the_shipped_virtual_shadow_map`, on a flat
ground with a floating slab casting across it, 256 × 144 at a 25° field of view,
one directional sun:

| | |
|---|---|
| covered by the trace | **36 864 of 36 864** (100 %) |
| agree | **36 450 (98.88 %)** |
| disagree | **414**, of which **410 (99.0 %) at a shadow edge** |
| traced shadowed | 4 180 |
| shipped shadowed | 4 594 |

The shipped mask is the frame with the sun casting against the same frame with
it not casting — the *output* the clause names, rather than a receiver function
called by hand.

**The 414 are the penumbra of a technique difference, not two different
shadows**, and the arm asserts that shape rather than a bare fraction: 99 % of
them border a pixel of the other class in one mask or the other. The shipped
path shadows **414 more pixels** than the trace, which is the shadow map's
filter kernel and its bias widening the umbra by about a pixel.

### 3c. The self-shadowing measurement, which is the honest half

The first fixture used the tree's standard **displaced** grid, and the two masks
agreed on only **56.7 %** of the covered frame — nearly all of the disagreement
in the direction *"the shipped path says shadowed, the trace says lit."* That is
**shadow acne**: a rasterized shadow map's depth bias mis-fires along every
grazing slope, and a ray query has no bias to mis-fire with.

That is a real quality win for ray queries and it is stated here rather than
tuned away. The fixture was flattened so the *comparison* is about the cast
shadow rather than about acne — but the 56.7 % is the number a reader should
carry, because it is where the two techniques actually differ.

### 3d. The surface offset is the trace's own bias, and it is not free

`a_zero_surface_offset_shadows_every_surface_with_itself`. `RtView::shadow_bias`
is a parameter so the gate can take it to zero:

| offset | device shadowed | CPU shadowed | pixels the two disagree on |
|---|---|---|---|
| 1 mm (shipped) | 830 / 19 322 | 830 | **0** |
| 0 | **7 697** | **14 123** | **10 852** |

The interesting part is not that it goes darker — it is that it goes **noisy**,
and differently noisy on each intersector. `eye + dir · t` lands microscopically
above or below its own triangle depending on the last bits, so a zero-offset
shadow is a function of rounding rather than of geometry. Ray queries do not
abolish the bias problem; they move it from the depth buffer to the ray origin.

## 4. COST — one adapter, and the shape matters more than the number

| | |
|---|---|
| BLAS + TLAS build, 1 BLAS / 512 triangles / 2 instances / 9.4 KiB input | **13.6 – 13.9 ms** |
| trace, 73 728 rays (256 × 144, two per pixel) | **3.43 – 3.46 ms** |

**Read the build number as a shape, not as a throughput.** It is a *cold* full
build including buffer creation and a queue submit, on 512 triangles — the
per-triangle cost is not what 13.6 ms measures, and wgpu 30's
`AccelerationStructureUpdateMode::PreferUpdate` (an incremental refit) was
**not** measured. What it does establish is that a *per-frame* full rebuild is
already the order of a whole 16.7 ms frame at a scene size four orders of
magnitude below the flagship's 10 M triangles.

Against that, the shipped path's own measured behaviour: P27's gate asserts a
static scene re-rasterizes **zero** shadow pages after warm-up. The virtual
shadow map's whole design is that it does not pay again for what did not move.
A per-frame TLAS is the opposite trade.

## 5. COVERAGE — the reason on its own

The TLAS holds **the meshlet clusters a host put in it, and nothing else.** This
engine's shadow casters, as of P27.2, are meshlets *and* skinned characters
*and* terrain clipmaps *and* rigid primitives *and* GPU scatter — and a sun
shadow that only knows about meshlets is not a sun shadow, it is a second one
that disagrees with the first everywhere else in the frame.

Bringing the rest in is not a tuning job:

* **terrain** is a clipmap whose geometry is generated per frame from a
  heightfield — a BLAS would have to be rebuilt as the clipmap scrolls;
* **skinned** geometry deforms every step, which is `PreferUpdate`'s use case
  and a per-character refit per frame;
* **voxel** volumes are carved at runtime (P21.4) and have no cooked DAG at all;
* **scatter** is a million GPU-culled instances with no CPU-side transform list.

The experiment's own gate makes the bound structural rather than a caveat: the
comparison is restricted to pixels where the primary ray hit, because that is
what the pass has an opinion about.

## 6. THE RULING

1. **VSM remains the shipped shadow path on every tier**, exactly as the clause
   requires. Nothing in `EngineRenderer::render` reads `RaytraceSettings`; there
   is no render-graph node; the experiment's only caller in the tree is its own
   gate. The 54 committed goldens are byte-frozen across this batch, which is
   the measurement that says the shipped frame did not move.
2. **The setting is off and nothing turns it on.** No tier and no preset assigns
   it — asserted in both directions by
   `the_experiment_is_off_by_default_and_no_tier_or_preset_turns_it_on`, which
   also checks a tier does not *clear* it either, so a host that deliberately
   turned the experiment on to measure it does not silently lose it.
   `AdapterCaps::clamp_ray_query` only ever clears, and an adapter that **can**
   trace does not enable it for a caller who did not ask — that second half is
   the law, and it is the one a table of expected values cannot see.
3. **What would re-open it**: ray queries reaching Metal and WebGPU in a pinned
   wgpu; an incremental TLAS refit measured against the per-frame rebuild; and a
   representation for terrain, skinned and voxel casters. The first is not this
   tree's to decide; the second is a day's work and was out of this batch's
   scope; the third is a phase.
4. **What is worth keeping from it regardless**: the acne measurement (3c). The
   shipped path loses 43 % of a displaced frame's shadow verdict to bias
   mis-fires that a ray query does not have. That is a *VSM* finding, produced
   by building its competitor, and it belongs to whoever next tunes the receiver
   bias.
