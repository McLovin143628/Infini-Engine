# P26–P28 — Virtualize Everything: SVT, VSM, and the unified streamer

**Date:** 2026-08-10. **Status:** approved direction (user decision; source: the external
direction document `../docs/Nanite_VSM_SVT_virtual-textures_implementation.txt`, outside the
repo). **Scope:** three phases appended to the Next-Gen Wave — Phase 26 (Streaming Virtual
Texturing), Phase 27 (Virtual Shadow Maps), Phase 28 (Nanite-native unification).

## The direction in one paragraph

UE5's modern pillars — Nanite, VSM, SVT — are one idea applied three times: map a huge
virtual address space (mesh clusters, shadow texels, texture tiles) onto a small physical
pool, and page in only what lands on screen pixels this frame, so rendering cost scales with
screen resolution rather than scene complexity. UE5's known weaknesses are the CPU↔GPU
feedback round-trip (multi-frame pop-in), three separate page systems that can disagree
(high-poly geometry with a blurry texture), reactive-only streaming, and CPU-bound
decompression. The direction document asks us to implement all three pillars natively in
Rust and beat UE5 on exactly those four weaknesses.

## What exists today (measured 2026-08-10, before P26)

* **Virtual geometry is real and paged**: `inf-vgeom` LOD-DAG meshlets (~64v/~124t) with a
  per-meshlet error cut, seam-safe builder, two-pass HZB occlusion on by default, `.inf_vmesh`
  v2 raw-image paged container (page 0 = all roots, never evicted; residency is always a
  prefix), transactional suballocated GPU pools, 256 MiB default budget, CPU-decided residency.
* **No visibility buffer**: meshlets draw vertex-pulled `draw_indirect` into the 4× MSAA
  forward targets and shade in the fragment shader. A separate `fs_id` path serves only the
  on-demand picker.
* **No SVT — and no material textures at all**: nothing binds `.inf_tex` in the interactive
  renderer; material data is per-instance vertex attributes. `.inf_tex` is a bincode payload
  with no byte-addressable tiles. `TEXTURE_COMPRESSION_BC` is not requested; no BC format is
  uploaded anywhere.
* **No VSM**: shadows are a compile-time 3×2048² forward-Z CSM, off by default, re-rendered
  every frame, rigid+scatter casters only. **vgeom does not cast shadows. Point/spot lights
  have no shadows.** No caching, no scissor use anywhere in the pass tree.
* **No per-frame GPU readback exists**, and `inf-vgeom/src/stream.rs:57-68` records a
  deliberate doctrine: feedback-driven residency was rejected because a readback is a frame
  latent, so the want-set would depend on frame history and replay gates would diverge.
* wgpu 30 (vulkan+metal), `Limits::default()` (4 bind groups), one optional feature requested
  (`POLYGON_MODE_LINE`). Capability tiers (`caps.rs`) gate every heavy feature; the law is
  `RenderTier::apply` never turns a feature on.

## The load-bearing ruling: deterministic feedback

The doc demands GPU-driven page requests; the house demands bit-exact replay. These are
reconcilable, and the reconciliation is the architecture:

1. **Feedback is an order-independent coverage mask, never an append list.** The feedback
   pass writes per-tile/per-page bits (atomicOr into a fixed-layout bitmask buffer keyed by
   virtual address), so its *content* is a pure function of (camera, scene, residency state)
   regardless of rasterization order. No append counters, no sampling jitter.
2. **Readback latency is pinned, not opportunistic.** A small non-blocking readback ring with
   a fixed N=2 frame latency (the codebase's first; built once as a reusable `inf-render`
   primitive). Frame k's requests are always consumed at frame k+2 — never "whenever the map
   resolves" — so the residency trace is a deterministic function of the frame sequence.
3. **The CPU analytic want-set remains the floor.** The existing deterministic wants
   (camera + bounds → conservative tile/page set, the vgeom `plan()` pattern) are still
   computed every frame; feedback only *refines* (adds precision, subtracts waste) and can
   never regress residency below the analytic floor. A dropped feedback frame degrades to
   exactly today's behaviour.
4. **Requests are scanned in fixed order** (virtual-address order, not arrival order) under
   monotone stamps, so two runs of a scripted path produce byte-identical residency traces —
   which is precisely what the phase gates pin.

This is *better than UE5* on its own terms: UE5's feedback is a sparse sampled UAV with
nondeterministic latency; ours is exact per-tile coverage with pinned latency and a
deterministic replay story.

## Honest deviations from the direction document

1. **DirectStorage / io_uring GPU-direct I/O**: not reachable through wgpu 30 — there is no
   API surface for NVMe→VRAM DMA or GPU-initiated I/O. Our equivalent: mmap zero-copy packs
   (`PackReader::read_ref` → aligned borrowed slices → `queue.write_*` upload) + job-pool
   async tile reads. Ledgered as a platform-HAL follow-up the day wgpu (or a console HAL)
   exposes it.
2. **GPU hardware decompression (GDeflate/BCPack)**: not exposed by wgpu. Instead tiles cook
   as raw BC blocks in mmap-sliceable sections, so there is *no* decompression step at all on
   the hot path — the doc's goal (no CPU decompress bottleneck) achieved by format design
   rather than by hardware decode. Optional cold-section zstd is a measure-first follow-up.
3. **Neural motion prediction → deterministic analytic dead-reckoning.** A learned predictor
   is untestable under house gates (training nondeterminism, no falsifiable bound). The
   implemented predictor extrapolates camera velocity + angular momentum over a 200–500 ms
   horizon as a pure function of committed input history — same intent, gate-able, and the
   A/B pop-in reduction is asserted by counters in the phase gate.
4. **Neural Texture Compression → deferred, with the intent kept.** The doc's fallback idea
   ("reconstruct detail while the real tile streams") ships as a deterministic edge-directed
   upscale of the finest resident ancestor into the physical page (compute pass, measured
   before adoption). Weight-per-material NTC needs a training pipeline and is deferred by
   memo, not silence.
5. **Ray-query shadows**: implemented only as an adapter-gated experiment (P28.5) if the
   pinned wgpu exposes ray queries on the running adapter; VSM rasterization is the
   load-bearing path on every tier. Never required, never default-on.

## Feature/limit policy

* `TEXTURE_COMPRESSION_BC` becomes an adapter-probed optional feature (P26); absent → the
  tier fallback is CPU-transcode-to-RGBA8 pages or clamped resident mips through the same
  door (`RenderTier::apply` may only clamp, never enable).
* `max_bind_groups` stays 4 unless measurement at P26.3 forces a probed raise; the default
  plan folds VT bindings (indirection texture, physical atlas, feedback buffer) into the
  shared env group after index 13, via the existing `GROUP_ENV` token substitution.
* No new required features, ever; every new capability gets a `caps.rs` clamp + tier test.

## Order and dependency rationale

**P26 SVT first**: it builds the whole page machinery (tiled container, physical pool,
indirection, feedback ring, residency module) *and* finally binds material textures in the
interactive renderer — the largest visual win. **P27 VSM second**: reuses the page-table/
atlas/feedback patterns, adds the meshlet-into-pages raster (closing "vgeom casts no
shadows") and page caching. **P28 unification last**: visibility-buffer shading, interleaved
mesh+texture cluster pages, one streamer/budget arbiter over all three consumers, predictive
prefetch, and the ray-query experiment — it restructures P26+P27 output and cannot come first.

## Budgets

Schema: `.inf_tex` v2 is an asset-container version (raw-image sectioned, the `.inf_vmesh`
v2 precedent — never through `inf_asset::encode`; v1 payloads keep loading). No scene-schema
bump is expected in P26/P27; if one becomes unavoidable it obeys the one-per-phase law.
Goldens are additive only; the 50 committed files never re-bless. Every new pass is a strict
no-op (encoder untouched, engagement-counter-instrumented) on scenes without its content.
