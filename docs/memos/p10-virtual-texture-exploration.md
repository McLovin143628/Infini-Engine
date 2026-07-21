# Memo: virtual texturing for terrain — explore & defer (P10.4)

**Status:** decided — **defer**; ship splat-blended layers + triplanar/macro now,
revisit streaming VT with `inf-vgeom` in P13/P15.
**Date:** 2026-07-21
**Scope:** how terrain material shading scales beyond the 4-layer splat blend —
whether to build a UE5-class virtual texture (VT) system now, later, or never.

## What P10.4 shipped

- **Splat-blended layered materials.** Each `inf_terrain::TerrainTile` now carries
  a per-sample `[u8; 4]` weight beside its `f32` height (stored sparsely — empty ⇒
  uniform layer 0, so unpainted terrain is byte-identical to pre-P10.4). Four
  `TerrainLayer`s (albedo / roughness / tex-scale) live on the ECS `Terrain`
  component; `shaders/terrain.wgsl` samples the weights at the splat hook and
  blends the four layers.
- **Triplanar + macro variation.** Steep slopes blend a procedural detail grain
  across the XZ/XY/YZ projections weighted by `|normal|^sharpness`; a large-scale
  fBm modulates albedo (`1 + amp·fbm(world_xz)`) to break up tiling.
- **Splat painting** reuses the sculpt machinery one storey up (weights, not
  heights): `apply_paint`/`SplatStroke`/`SplatDelta` mirror the height
  brush/stroke/delta, undoable as one step.

**The honest gap:** layer **texture GUIDs are deferred**. The interactive
viewport still can't upload asset textures (the same documented limitation
sprites, tilemaps, and material previews have — it renders the white fallback).
So layers prove the *blend* with solid albedo colours + procedural grain, not
image textures. Real per-layer albedo/normal/ORM textures — and the desire to
stack *many* of them — are exactly what pushes toward virtual texturing.

## What UE5-class virtual texturing buys terrains

Two related mechanisms:

- **Runtime Virtual Texturing (RVT).** Composite the terrain material (all splat
  layers, plus decals/roads/PCG splatter blended in) **once** into a cached
  physical-texture atlas, then shade the terrain by a cheap lookup into that
  cache. Shading cost decouples from layer count — a 20-layer terrain shades at
  roughly the cost of one textured surface, and expensive per-pixel work (parallax,
  many-layer height-blend) is paid once per cached page, not per frame per pixel.
- **Streaming Virtual Texturing (SVT).** Decouple *resolution* from memory: a GPU
  **feedback** pass records which texture pages are actually visible this frame; a
  page table indirects UVs into a fixed-size physical cache; a streamer pages
  tiles in/out (LRU) from disk. A 256 k² virtual terrain texture costs only the
  handful of pages on screen.

Together they let a terrain carry effectively unbounded material detail and
runtime composition (roads that blend into the ground, PCG that stamps into the
cached albedo) at bounded, near-constant shading and memory cost.

## What our tile-texture cache already gives us

`passes/terrain.rs` already runs a **tile-granular texture cache**: a
`BTreeMap<(i32,i32), _>` of per-tile `R32Float` height textures, uploaded
version-gated, evicted when a tile vanishes. P10.4 adds a **parallel per-tile
`Rgba8Unorm` weight-texture cache** on the same pattern.

This is genuinely the *shape* a VT wants — a coordinate-keyed, gated, evicting
cache of GPU pages — but it is deliberately far short of one:

- **Resident, not streamed.** Every authored tile's textures live on the GPU at
  once; there is no on-screen working set, no LRU eviction under budget.
- **No indirection.** UVs address the tile's texture directly; there is no page
  table, so we can't reface a virtual address space onto a smaller physical cache.
- **No feedback.** Nothing measures which pages/mips are actually needed — upload
  is gated by the document version, not by visibility.
- **Resolution == tile resolution.** Shading detail is capped at the height-sample
  grid; there's no way to shade finer than the heightfield without more tiles.

For the 4-layer, colour-blended terrain P10.4 targets, that's the right amount of
machinery. It becomes the *wrong* amount the moment layers get real textures and
multiply.

## A streaming-VT sketch for P13/P15

When we do build it, the pieces are well understood and compose with what exists:

1. **Physical cache + page table.** One large `Rgba8`/BC-compressed physical
   texture atlas of fixed page size; a page-table texture mapping virtual terrain
   UV → physical page. Terrain UVs run through the page table in the fragment
   shader.
2. **Feedback pass.** A tiny render target (or a storage buffer) where the terrain
   fragment writes the virtual page id + mip it wanted; read back (async), diffed
   against resident pages, drives the streamer.
3. **Page composition.** To fill a page, run the **P7 material codegen** path: the
   splat-blended surface (P10.4) is exactly a material graph output, so composing
   a page = baking that graph over the page's UV rect into the atlas (we already
   bake material graphs to `.inf_tex` via `emit_texture_compute` — the same
   compute-bake seam, retargeted to a physical page). Decals/roads/PCG splatter
   blend into the page here, giving RVT-style runtime composition for free.
4. **Co-design with `inf-vgeom`.** Phase 13's virtualized geometry (meshlet DAG,
   streaming, GPU culling — ROADMAP crate list) has the same feedback+streaming
   spine (visible-cluster feedback, LRU page residency, a budgeted streamer). VT
   and virtualized geometry should share that streaming substrate rather than grow
   two; that is the main reason to wait for P13.

Keep it a sketch, not a spec — the feedback→stream→compose loop and its budget
policy are the hard parts and want real content to tune against.

## Decision — defer

The tile-texture caches + 4-way splat blend cover P10's massive-terrain goal.
Building a feedback-driven streaming VT now would be speculative machinery ahead
of the content and ahead of the geometry-streaming system it should share a spine
with. **Revisit when any of these bite:**

- **Layer count** outgrows what 4-way `Rgba8` weights + a few texture-array slots
  cover (a landscape wanting 8–20+ blended materials).
- **Shading cost** becomes the terrain bottleneck (many-layer per-pixel blend,
  parallax/height-blend, big draw distances) — i.e. caching composited pages would
  actually pay off.
- **Runtime composition** is required — roads, decals, PCG, or gameplay marks that
  must blend *into* the terrain material rather than draw over it.
- **Streaming budget pressure** from very large worlds makes resident per-tile
  textures untenable.
- **`inf-vgeom` (P13) lands**, at which point VT should be built on its streaming
  substrate rather than a separate one.

Until then: real per-layer **texture GUID upload** in the interactive viewport
(closing the shared sprite/tilemap/material texture gap) is the next concrete step
for terrain materials, and needs no VT.

*This memo records a P10.4 decision.*
