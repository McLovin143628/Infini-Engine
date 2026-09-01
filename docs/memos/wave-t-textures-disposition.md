# Wave T — what the texture document asked for, and what happened to every item

**Source:** `Rust_Game_Engine_Textures.md`, supplied 2026-08-17.
**Instruction:** implement its contents; **bolded passages are the highest
priority**; anything that cannot be implemented must be reported with the reason
why. This memo is that report.

**Wave commits:** `65f6c57` (formats + HDR), `1e19d82` (the bold three),
`e3c5509` (the sampler), `3f2815a` (the map-set importer), `ca9559b` (this memo),
`2f4a1ec` (three review fixes), `17a2a5e` (the GPU proofs), plus the audit pass
below.

**Counts:** 10 SHIPPED · 23 ALREADY-HAD · 4 MET-BY-A-DIFFERENT-MECHANISM ·
6 DEFERRED · 12 CANNOT.
Schema: `.inf_tex` v2→**v3**, `.inf_vmesh` v3→**v4**, both stamped only on
payloads that need them. Scene schema: **unmoved**. Goldens: **54, unmoved**
(verified strict, twice, plus the three independent digest gates).
New external dependencies: **zero** — `half` is newly *named* by `inf-material`
but has been pinned in `[workspace.dependencies]` and linked by `inf-terrain`
since P16, so no crate entered the graph (`Cargo.lock` gains one edge, no
package). Frontend untouched.
Battery after the audit: **284 binaries / 5 264 passed / 0 failed / 13 ignored**
(the audit added ten arms and a test binary; the pre-audit tree measured 5 254 on
this machine against the 5 251 recorded above, a three-arm discrepancy nobody has
explained and nothing rests on).

### The audit pass, and what it found

The wave was audited after the fact. **Two defects, seven unproven claims and two
wrong numbers** were found and closed; the ledger below is corrected where it was
wrong. Mutation-measured throughout: eighteen mutations were run against the
wave's tree, breaking one rule each — **nine of them changed no test in the
workspace**. Every one of the nine fails now, and so do the four that falsify the
audit's own fixes. What follows is what a reader of this memo needs to know:

* **A detail map stopped its own surface streaming — on all three producers.**
  `VtTextureSet::slots()` packs the detail lane into the top half of two instance
  words, and every consumer that turns a slot into a texture handle was reading
  that packed word: `camera_wants` and `feedback_requests` on the CPU (the
  analytic and per-surface halves), and `vis_mark_tile` in `vis_feedback.wgsl`
  on the GPU (the per-FRAGMENT half, which on the visbuffer path is the *only*
  producer, because the per-surface one is handed over to it). All three
  computed `handle = (albedo | detail << 16) - 1`, found it past the registry,
  and gave up. So the moment a material bound a detail map, its **albedo and
  normal silently stopped being requested**, and the detail map was never
  requested at all; everything fell back to the pinned floor and stayed at its
  coarsest three levels for ever. Fixed by splitting the two questions into two
  functions (`slots()` for the GPU wire, `handles()` for anything asking which
  textures a surface names) and by masking in the shader; and the per-fragment
  pass now marks the detail lane at `uv * scale`, where `vt_apply_detail`
  samples it, because the analytic lane asks at the base uv and would leave a
  map tiled N times `log2(N)` mips too coarse. Not reachable from any shipped
  content — nothing in either host binds a detail map yet — but it made §0 A a
  capability that did not work the first time it was used.
* **A float import baked infinities and NaNs.** A texel of 70 000 (an ordinary
  sun disc) narrowed to `+inf` and a NaN stayed a NaN, and the box downsample
  then carried a *single* poisoned texel to every coarser level of the pyramid
  up to the 1×1 — which is the level a virtual texture pins resident. The import
  door now clamps: NaN → 0, overflow → ±65 504, said out loud in the advisory.
* **Seven claims had nothing measuring them**, each proven free by mutation: the
  meshlet-material staging arm re-implemented the staging loop instead of calling
  it (so deleting the shipped write failed nothing); the detail-fade fix was
  probed at a level of detail where the truncating bug and the continuous fix
  give the same answer; and `slots()`'s own bit layout, `set_for_maps`'s
  scale-gate, the whole BC5 `reconstruct_z` seam (container → table flag →
  shader branch — mutating it to a constant `false`, which lights every BC5
  normal map backwards, failed no test), and **the entirety of §0 B** had no arm
  at all. All now falsify.
* **Two numbers in the ledger below were wrong**, and both are the kind a reader
  quotes: §4.5 called 12.13 "bilinear" when it is the quincunx control (true
  bilinear is 11.45 — and mislabelling that column is the exact error the memo
  it cites was written to correct), and T44's tripwire was described as firing on
  a registration when it watches two constants. Both corrected in place, with
  what they are instead.
* **§0 B's textured branch has still never run on a GPU** — see the honest bound
  recorded there.

Everything else the memo claimed was checked and held: the goldens (strict,
twice, plus three digest gates), the schema minimum-version rule in both
containers and both refusal arms, BC5's size and quality numbers
(re-measured independently, and widened below), the HDR fixtures' fidelity and
default-off, the CANNOT list's technical reasons, and "zero new crates".

---

## 0. The three bolded passages, and where they landed

The document is a paste from a tool that escaped most of its emphasis, so only
nine lines carry real markdown bold. They are the whole of the document's §3 plus
its §2 closing thesis. All three §3 items shipped.

### A. Detail maps & micro-detail blending — **SHIPPED** (`1e19d82`)

> *"Instead of using unique 8K or 16K textures per object, use 2K base textures
> paired with high-frequency 512×512 tileable detail normal/roughness maps."*

`vt_apply_detail` in `crates/inf-render/src/shaders/vt_sample.wgsl`. A material
may name a fourth virtual texture and a tiling multiplier; the detail sample's XY
is added to the base normal's (the UDN blend, which cannot flatten the base where
the detail is neutral) and its alpha multiplies roughness — one rule, and a BC5
detail map samples alpha as exactly `1.0`, so the two storage choices disagree
about nothing.

**It costs no instance bytes, no vertex attribute and no bind group.** The detail
slot and its 8.8 fixed-point scale ride in the top 16 bits of two per-instance map
words that have been zero on every instance ever uploaded. That mattered: the
skinned pipeline already sits exactly on `max_vertex_attributes: 16`
(`docs/memos/p26-5-vertex-streams.md`), so a fourth word was not available.

*Why those bits really are free, spelled out because the first write-up got the
reason nearly right:* a slot is `texture handle + 1`, and the handle is bounded
not by the atlas slot field but by **root seating** — every registered texture
pins at least one root page, `register_texture` refuses once the pinned roots
exceed the atlas's slot count, and that count is itself `≤ MAX_SLOT_INDEX`
(65 535). So a registry cannot hold more textures than the atlas has slots, and
`handle + 1` cannot leave 16 bits.

*And the cost that was not zero:* a packed word is **not** a texture slot, and
the two functions that turn a coverage set into stream requests were reading one.
See the audit note at the top of this memo, and `VtTextureSet::handles`.

The fade is the detail texture's **own mip pyramid** rather than a distance ramp:
the weight ramps out across the last two levels, which is correct under a zoom as
well as a walk and needs no camera, no uniform and no tuning. *(This was one of
the wave's three late fixes — the weight had been taken from the truncating
`vt_mip`, making the fade a hard cut. The audit found the fix unmeasured: the
existing probe ran at a level of detail where a truncated weight and a continuous
one agree. `the_detail_fade_ramps_across_the_last_two_levels` now probes at a
**fractional** level, where they cannot, and dies if the truncation returns.)*

**Proven on the GPU, not only validated.** `the_detail_map_reaches_the_surface_and_is_inert_without_one` runs the shipped `vt_surface` in a compute pass and asserts both halves: with the lane zero the surface is identical (which is what makes "the 54 goldens did not move" a measurement), and with it set the normal moves in the direction the detail map's XY says and the roughness is multiplied by its alpha. A detail slot with a ZERO SCALE is asserted inert too, because the slot and the scale are one decision.

The saving the document is after is structural here rather than incidental,
because virtual textures are deduplicated by GUID: the *second* object to
reference a shared detail map pays nothing at all, on disk or in the atlas.

**Remainder — the authoring field.** `.inf_mat` (`MaterialAsset`) has no
`detail_texture` reference, so today a detail map is bound through the renderer
API (`VtMaterialMaps::with_detail`) and not from a saved material. Adding the
field is a `.inf_mat` v2→v3 bincode bump; this wave's schema budget was `.inf_tex`
once and `.inf_vmesh` once, with the scene/asset schemas frozen for a later
consolidation. **This is the one field that turns A from a capability into a
feature an artist can use, and it is small.**

### B. Material layering (splatting) via virtual textures — **SHIPPED** (`1e19d82`)

> *"For terrain or large props, store blend weights in a virtual texture mask and
> sample shared, tileable PBR materials."*

The weights half has shipped since P10.4 — a per-sample RGBA8 splat mask,
painted, undoable, renormalised to sum exactly 255. What was missing is the other
half, and the engine had already written it down against itself
(`inf_ecs::TerrainLayer`: *"a texture GUID is deliberately absent… per-layer
albedo/normal/ORM texture refs are the documented follow-up"*).

`terrain_layers()` in `terrain.wgsl` now weight-blends up to four real
`vt_surface` samples at `world.xz / tex_scale` — the same `tex_scale` the
procedural grain already used, so a layer's authored tiling means one thing
rather than two. Two details that are the difference between working and
half-working:

* **weight-gated per layer**, at the splat mask's own quantum (1/256), so the
  texture fetches are the ones that are visible at this fragment — typically one
  or two, not four unconditionally;
* **renormalised inside the covered fraction**, so a terrain where only two of
  four layers carry materials fades into the flat colours of the other two
  instead of darkening toward black.

**Honest bound 1 — the projection:** planar XZ. A shared ground material
stretches on a cliff face. The procedural triplanar grain over it does not, so
the *break-up* on a steep face survives and only the material's own pattern
stretches. A triplanar layer sample is three times the fetches; it is the named
follow-up.

**Honest bound 2 — this branch has never run on a GPU.** `terrain_layers` calls
`dpdx`/`dpdy`, which are fragment-stage builtins, so it cannot be driven from the
compute probe that executes `vt_surface` for §0 A; and nothing in either host
binds a layer material yet, so no golden covers it either. What *is* measured
(`crates/inf-render/tests/terrain_layers.rs`, added by the audit): the
world-anchored uv, with the render-local slide it replaced quantified at
**2.5 tiles per 10 m snap quantum and hundreds of tiles per real rebase**; the
1/256 weight gate; the covered-fraction renormalisation; and the `layered.any`
guard that keeps an untextured terrain instruction-for-instruction what it was —
each as a CPU twin pinned to the shader by a **scope-extracted** source gate,
and each verified to fail when its rule is mutated out of the shader. What is
*not* measured is the sampled result itself. A fragment probe over terrain's four
bind groups is the work that closes it, and it should land with the
`TerrainLayer` authoring field (2.4) rather than before it.

**Remainder — the authoring field, again.** `inf_ecs::TerrainLayer` carries no
texture reference and the scene schema is frozen this wave, so the per-layer
material is bound on the render record (`RenderTerrainLayer::vt`) rather than
saved in the level. Both hosts spell the default identically so they cannot
drift. One `Option<Uuid>` on `TerrainLayer` closes it.

**Not done:** promoting the *mask itself* to a virtual texture so a **prop** can
carry one. Terrain's mask rides the `.inf_terrain` tile stream, which is the
right home for it; a per-prop mask is a genuinely separate feature and is already
on the books at `docs/ROADMAP.md:1240`.

### C. Meshlet-based procedural texturing — **SHIPPED** (`1e19d82`)

> *"consider storing material properties per meshlet cluster or driving material
> variation procedurally via distance fields / ambient occlusion masks baked
> directly into meshlet data."*

`.inf_vmesh` **v4** carries a materials section: one `u32` material slot per
meshlet. `build_vgeom_with_materials` votes each cluster's slot from its
vertices'. This closes `inf-vgeom`'s own oldest follow-up — *"v1 flattens all
submeshes into one geometry; material-slot tagging per meshlet is a follow-up"*.

Two decisions worth reading:

* **Per vertex, not per triangle.** A vertex index is the one identity that
  survives the weld, the cache optimisation and every `simplify` call; a
  triangle's is destroyed by the first one.
* **The record did not grow by a byte.** The GPU meshlet struct's last word was
  named `pad` and carried the CPU's `group`, which no shader read (verified: the
  five WGSL modules declaring the struct name the word and none of them reads
  it). The page-in staging overwrites it with the material slot —
  unconditionally, so a v3 page stages `0` rather than leaking a group index into
  a word the shader now reads. A 6 % meshlet-pool inflation was available and was
  not taken.
  *Audit note:* the arm asserting this originally copied the staging loop into
  its own body, so deleting the write from `stage_page` failed nothing. It now
  runs the shipped streamer (`VgeomStreamer::plan`) and reads the bytes the pool
  is written from, and it dies when the write is removed.

**Remainder — the shading consumer.** A meshlet's slot is not yet *used* to pick
a material, because doing so needs a per-instance material **table** (slot → the
three texture handles) and P28.1 measured `vis_resolve`'s storage bindings fully
spent. The designed route needs no new binding: put the material sets in a second
section of the existing VT indirection buffer, which is the same argument that
put every texture's table in one buffer to begin with. Until then a multi-material
Nanite mesh shades as its instance's material, exactly as before — and now it
*knows* which meshlet wanted which.

### The §2 thesis (T10) — **PART, and a success criterion rather than a task**

> *"GPU-driven rendering, zero-CPU I/O, and data-oriented parallel execution."*

GPU-driven rendering and data-oriented parallel execution: shipped, long before
this document. **Zero-CPU I/O is the one third that is structurally blocked** —
see CANNOT §4.1 and §4.2. What the engine has instead is mmap zero-copy packs
with no decompression step at all on the texture path, which collects most of the
win and leaves only the host→VRAM leg the DMA engine performs anyway.

---

## 1. SHIPPED this wave

| # | Prescription | Where |
|---|---|---|
| T13 | Normal maps as two channels with `Z = √(1 − X² − Y²)` rebuilt in the shader | `PageFormat::Bc5`, `vt_normal_ts` (executed end to end from a real BC5 container by `a_bc5_normal_map_rebuilds_its_z_through_the_shipped_sampler`, added by the audit — before it, the whole container→table-flag→shader-branch seam had no arm) |
| T14 | Pack loose AO / roughness / metallic into one ORM image | `inf_material::mapset::pack_orm` |
| T20 | An importer that consumes a Megascans **set** | `inf_material::mapset::plan_map_set` |
| T20b | Stop flattening EXR/HDR sources to 8 bits | `PageFormat::Rgba16F`, `TextureImportSettings::hdr` |
| T25 | **BOLD A** — detail maps | `vt_apply_detail` |
| T26 | **BOLD B** — layered terrain materials | `terrain_layers()` |
| T27 | **BOLD C** — per-meshlet material slots | `.inf_vmesh` v4 |
| T44 | UV precision at 32K+ | a **tripwire**, not a feature: `an_f32_uv_still_addresses_every_texel_of_the_largest_legal_pyramid`. *Audit correction:* it watches the **constants** (a widened slot field or tile size), not a registration — `VtTextureDesc::validate` has no extent rule at all, so what actually keeps a pyramid inside an `f32` uv today is that no content this project can produce comes near it. The arm now pins that gap open instead of implying a guard |
| T45a | BC5 for normal maps | `inf_material::bc::compress_bc5` |
| T47 | Trilinear | `vt_sample`, opt-in behind `VirtualTextureSettings::trilinear` |

### The trilinear arm, and the fixture that measured nothing

`trilinear_blends_two_levels_and_is_off_by_default` runs the shipped sampler at a
level of detail of exactly **1.5** — the worst case for a truncating sampler — and
asserts it changes the result there and changes *nothing* at an integer level, where
the document's own `blend < 0.01` early-out says it must not.

It is worth recording what its first fixture did. A 32-texel stripe pattern measured
**0 of 24** probes changing, which reads exactly like a dead feature and is not one: a
power-of-two-aligned stripe is very nearly scale-invariant under a box downsample, so
mip 1 *is* mip 0 at half the resolution and blending two levels of it cannot change
anything. The arm now probes mip 1 against mip 2 directly and refuses to certify
anything unless the two levels disagree at half its probes — the anti-vacuity the
first draft would have shipped without.

### The numbers that justify BC5

Normal maps have shipped **uncompressed** in this engine since Phase 4, because
`TextureImportSettings::data`'s own comment said why: *"BC introduces artifacts
in normals; a BC5 path is future work."* It was right — about BC1.

* **Size.** A 136² stored tile is **73 984 B** as RGBA8 and **18 496 B** as BC5:
  exactly 4×. A 4K normal map's whole `.inf_tex` goes from **96.8 MiB to
  24.2 MiB**.
* **Quality, against the alternative that actually existed.** On a swept
  tangent-space normal field, per-channel mean absolute error on X and Y:
  **BC5 = 0.000, BC1 = 2.849**. BC1's 5:6:5 endpoints quantise the red axis to 32
  levels, and the red axis is where a tangent-space X lives.
  **That 0.000 is exact for that fixture and it is not a claim that BC5 is
  lossless** — the qualifier matters, so here is the sweep the audit re-measured
  independently (same encoder, four fixtures, 64² each, MAE on X and Y):

  | fixture | BC5 | BC1 |
  |---|---|---|
  | swept normal field (smooth, ≤4 distinct values a block) | **0.000** | 2.849 |
  | flat block (the degenerate `a0 == a1` mode) | **0.000** | — |
  | linear gradient across x | 0.250 | — |
  | uniform noise, i.e. the worst case for a two-endpoint fit | **7.024** | 51.909 |

  So BC5 is *exact* on data whose per-block range is spanned by its eight
  interpolated levels — which is what a real tangent-space normal map is almost
  everywhere — and is still **7× better than BC1** on data that is not. The
  claim to carry forward is the ratio, not the zero.
* **The rebuild is not an approximation.** A unit normal's Z is determined by X
  and Y. Measured worst-case error of the rebuilt vector against the source:
  **0.021**, and that worst case is on the rim of the unit disc where
  `z = √(1 − x² − y²)` has infinite slope — arithmetic, not a defect. Mean:
  under 0.005.

The encoder is nine lines, because the BC3 alpha block this project already had
*is* a BC4 block. It stays integer-only under the existing source gate.

### The HDR ruling, and what it cost

There was exactly **one** decode on the import path —
`image::load_from_memory(bytes)?.to_rgba8()` — so every OpenEXR and Radiance
source arrived with **everything above 1.0 clipped** and the rest quantised to
256 steps, and the import reported success. That is the shape Megascans and Fab
ship displacement, cavity and bent-normal maps in. Nothing downstream could tell
the difference between that and a PNG.

`hdr: true` now keeps the range as `RGBA16F`. The arm that proves it builds a
Radiance file whose brightest texel is `8.0` and measures both paths against the
source. **Default is `false`**, so no existing import changes.

**What it costs, measured** (`.inf_tex`, full pyramid, per texture):

| extent | RGBA16F (shipped) | BC6H (deferred) | RGBA8 (today's silent flatten) |
|---|---|---|---|
| 2048² | 49.1 MiB | 6.1 MiB | 24.6 MiB |
| 4096² | 193.6 MiB | 24.2 MiB | 96.8 MiB |
| 8192² | 771.6 MiB | 96.5 MiB | 385.8 MiB |

**BC6H is exactly 8× smaller than what shipped**, and it is a **named deferral**
(§2). Until it exists, a float map costs 2× the 8-bit one and is *correct*, and
the importer says so out loud in both directions — `hdr_import_advisory` names the
clipping when the range is being thrown away and names the size when it is being
kept.

RGBA16F deliberately does **not** need `TEXTURE_COMPRESSION_BC`, so an adapter
without BC keeps it whole rather than being handed a transcode that would clamp
exactly the range the format exists to carry; `TiledTextureReader::tile_rgba8`
refuses a float page for the same reason.

**The ceiling, which the first write-up did not state** (audit). A half is not an
`f32`: it stops at **65 504**, and `half::f16::from_f32` is faithful about it —
a texel of 70 000, an ordinary sun disc in a captured sky, became `+inf`, and a
NaN stayed a NaN. Both were stored, and then `rgba16f_mip_chain` averaged them,
so **one bad texel made every coarser level of the pyramid NaN**, up to and
including the 1×1 — which is exactly the level a virtual texture pins resident
and always samples. `TextureAsset::validate` called the result healthy. The
import door now applies the P29.5 rule (*a door clamps or refuses, it never
bakes*): `half_at_the_door` maps NaN to `0.0` and saturates an overflow to
`±65 504`, the mip chain speaks the same function so a level cannot introduce
what the door refused, and the "kept" advisory names both. Finite in-range
samples are untouched, so nothing an honest source carries changed.

### Why neither schema bump moved a byte of existing content

Both containers stamp **the lowest version the payload needs**, not the newest
the build knows. A BC1 albedo still writes `.inf_tex` v2; a mesh with no
per-meshlet materials still writes `.inf_vmesh` v3, byte-identical through the
whole directory and every page section. So no content hash moves, no import cache
is invalidated, and no `.inf_pack` stops reproducing. Only a payload that
actually uses a v3/v4 feature is stamped, and only that payload is refused by an
older build — by name, at the door, instead of being mis-read. A v2 stamp over a
v3 format code, or a v3 stamp over a materials section, is refused: the version
and the feature are one contract.

---

## 2. DEFERRED — wanted, not built this wave, with the reason

**2.1 — BC6H (compressed HDR).** The measurement is in §1: it is exactly 8× the
saving on every float texture, which is the largest single win left on this list.
It is not a *hard* problem in the way BC7 is, but it is a real encoder (two mode
groups, signed and unsigned, delta-coded endpoints) and it is float-domain work
sitting behind a source gate that bans floats from the BC module for reasons that
are load-bearing (a texture's bytes are content-hashed into a reproducible pack).
Doing it right means either an integer half-float endpoint fit or an explicit,
documented exemption with a portability argument. Both are a batch, not an
afternoon, and the wave's priority was the bold set.

**2.2 — BC7 (T45b, and the BC7 half of T12).** Wanted for base colour and ORM, where BC1's 3-colour
endpoint palette bands visibly on Megascans albedo. Bounded rather than guessed
at: BC7 is 8 bpp against BC1's 4, i.e. **twice the page bytes**, for a quality
step that is roughly BC1's error halved on smooth gradients and much better than
that on blocks with two distinct colour clusters (BC1 has one line segment; BC7
has partitions). It is eight modes and sixty-four partition tables of owned
integer code — real work, and the route the document suggests (ISPC /
`intel_tex_2`) is the one this project already refused at Phase 4 as a cross-OS
CI liability. **Own the encoder or do not have it**; that has not changed.

**2.3 — The `.inf_mat` detail-texture field** and **2.4 — the
`TerrainLayer` material field.** See §0 A and B. Both are one optional asset
reference; both are schema bumps this wave was told not to spend. They are the
difference between the two bold items being *capabilities* and being *features an
artist can use*, and they should be first in the consolidated schema wave.

**2.5 — The per-meshlet material shading consumer.** See §0 C.

**2.6 — Anisotropic filtering (T46).** The document says: *"this shader already
supports anisotropic filtering perfectly because `textureSampleArray` is used
with a normal UV, and the border padding handles sampling neighbouring tiles."*
**That claim is false at this engine's border width, and the measurement is
simple enough to state here rather than build a feature around.** A tile stores a
4-texel border ring. An anisotropic footprint at ratio *N* walks up to *N/2*
texels past the isotropic footprint along its major axis, so a 4-texel ring
carries a ratio of about **8:1 at the very edge of a tile and nothing beyond it**
— and past the ring the sampler reads the *neighbouring atlas slot*, which is a
different texture entirely. The honest options are a wider border (a `.inf_tex`
bump, and the border already costs 6.8× at 128² BC1) or clamping the gradient to
the ring and accepting reduced anisotropy near tile edges. Neither is free and
neither was measured against a frame this wave. `anisotropy_clamp` stays at 1,
deliberately; raising it alone would sample another texture's texels.

---

## 3. Prescriptions the engine already satisfied

Roughly two-thirds of this document restates
`docs/Nanite_VSM_SVT_virtual-textures_implementation.txt` (2026-08-10), which was
approved as the P26–P28 direction and shipped. The following were already true
before Wave T began, and are listed so the owner can see they were *checked*
rather than assumed.

| # | Prescription | Already |
|---|---|---|
| T1 | Lock-free async task graphs | `inf_core::job` — rayon + flume, deterministic in-order `parallel_map` |
| T2 | Data-oriented ECS over hierarchies | `inf-ecs` (bevy_ecs), parallel schedule since P9 |
| T3 | Zero-cost typed GPU buffer abstractions | `inf_vt::table`, `inf-vgeom` indirect pools, bytemuck throughout |
| T6 | Unified mesh-texture cluster page | `.inf_vmesh` v3 interleaves tiles per page (P28.2); the tiles section holds **addresses, not texels**, and that was decided by measurement — two slices of one mmap are one read |
| T11 | Never load raw EXR/PNG into GPU memory; bake offline | The shipped player does not link `inf-material`, so no image decoder exists at runtime |
| T15 | Slice each mip into fixed power-of-two tiles | `TILE_SIZE = 128` |
| T16 | A 2–4 pixel border per tile | `TILE_BORDER = 4`, proven 4× over the worst case across twelve swept pyramids |
| T19 | One binary container per texture stack | `.inf_tex`, magic `INFVTEX\0` |
| T21 | GPU feedback pass | Two producers, both compute, both `atomicOr` into a fixed-layout mask |
| T22 | Async CPU readback of the feedback buffer | `readback.rs`, `map_async` + `poll` (never `Wait`) |
| T23 | Physical atlas as an LRU cache | `lru_victim`, free-list first, pinned roots never victims |
| T24 | Indirection page table | One storage buffer holding every texture's table |
| T28 | One archive sorted by mip, spatially clustered | Exactly the shipped layout; the admission sort matches it |
| T30 | Pack contiguous requests into one read | With mmap there is no read to merge; admits are sorted into payload order so the faults walk forward |
| T33 | Prioritise by mip distance, low mips first | Three lanes (floor / feedback / predict), admitted one lane at a time |
| T34 | LRU eviction of the stalest non-visible slot | `residency.rs` |
| T35 | Update the page table only after the transfer completes | A queue-ordering contract instead of a fence, plus a warm gate |
| T37 | Batch a frame's tile updates into one submission | All of a transaction's writes ride the frame's single submit |
| T40 | Zero frame stalls; never `vkQueueWaitIdle` | No blocking poll anywhere on the streaming path |
| T41 | The virtual-UV → page-table → atlas sampling shader | Shipped — **and more correct than the document's version**, see §5 |
| T42 | `compute_required_vt_mip` from `dpdx`/`dpdy` | `vt_lod`, gradients taken *before* the tiling wrap |
| T48 | Non-resident fallback + a pinned coarsest mip | The document's own option 2 (CPU-side hierarchy propagation), which is the better one; the pinned floor is `inf-vt`'s Law 1 |
| T50 | Three-frame ring-buffered readback | Latency **pinned** at 2, not merely typical |

Two survey claims inherited from the pre-wave scouting were re-checked against
the tree and one was wrong:

* *"normal maps ship uncompressed today"* — **confirmed**
  (`TextureImportSettings::data` → `TextureCompression::None`), and it is what
  T13/T45a fixed.
* *"`TEXTURE_COMPRESSION_BC` is already requested"* — **confirmed**
  (`gpu.rs`, masked by `adapter.features()`), which is why BC5 needed no feature
  negotiation, no new capability tier and no caps work.
* *"`VT_ADMITS_PER_FRAME_CEILING` is cited by name and defined nowhere"* —
  **false**. It is defined in `runtime/inf-player/src/budget.rs` and asserted by
  `phase26_gate`. No fix was needed. *(The line number this bullet carried — 368
  — has drifted with the file, which is what line numbers in prose do; the
  constant is now at 593. Named rather than numbered from IASSET1 on.)*
* *"`inf-vt`'s `fill.rs` ships with no hot-path caller"* — **confirmed**; its only
  callers are its own tests and `inf-material/tests/vt_fill_quality.rs`. It
  remains the measured fallback the P26 direction memo promised, awaiting a
  loader with real latency to be worth wiring (see CANNOT §4.4).

---

## 3b. Met by a different mechanism, or whose premise this engine does not have

Four items are neither "done" nor "not done" — the prescription describes a
solution to a problem this engine solved differently or does not have. Listed
separately so the ledger has no silent gaps.

**T33b — *"govern streaming requests with strict frame budgets."*** There is no
per-frame time budget and no per-frame admission or upload throttle in the VT
loop; the only budget is a byte residency ceiling. That is deliberate and it is
**coupled to a measured ruling**: `apply_wants` seats a miss the frame it is
offered, with no latency between admitted and sampleable, and
`docs/memos/p28-5-lead-time-ruling.md` names *"a per-frame admission throttle, or
a loader with real latency"* as the two changes that would reverse the h=0
prefetch decision. Adding one is therefore not a local change — it re-opens a
measurement, and the named tripwire tests are built to go red when it lands. Do
it deliberately or not at all. **Not done this wave, on purpose.**

**T36 — persistent ring staging buffers.** The engine uses `queue.write_texture`
per tile with the **tight** row pitch and lets wgpu's own staging belt be the
ring. The document's `copy_buffer_to_texture` route is *actively wrong here* and
that was measured rather than argued: a 34-block BC1 row is 272 bytes, which is
not a multiple of `COPY_BYTES_PER_ROW_ALIGNMENT` (256), so the call rejects the
write — and padding `bytes_per_row` without padding the data makes wgpu reject it
too. Adopting T36 would require repacking every tile on the CPU, i.e. undoing the
zero-copy property the whole container design exists for.

**T43 — the feedback shader writing `rg32uint` packed page ids.** The engine
writes an order-independent `atomicOr` coverage bitmask instead, one bit per
virtual tile at the same index the indirection table uses. The reason is the same
one that refuses T49 (§4.10): OR is commutative and idempotent, so the mask is a
pure function of (camera, request list, table) and not of how many threads ran or
in what order. The document's packed-ID scheme is fine in itself; it is the
compaction that follows it that cannot ship here.

**T51 — the three-state page lifecycle (Free → Pending In-Flight → Resident)
with a GPU pending-request bitset.** The problem it solves — a "feedback storm"
requesting one tile hundreds of times while it loads — **needs a window between
request and residency, and this loop has none**: `VtTextures::sync` applies and
stages in one synchronous call from mmap slices. It becomes required the day an
async upload path lands, which is the same day T33b's ruling re-opens.

---

## 4. CANNOT — with the specific technical reason

These are the items that cannot be implemented as prescribed. Where the engine
already achieves the underlying goal by another route, that is said.

**4.1 — DirectStorage and GPU-initiated I/O (T4, T31)** *(doc: "Fully GPU-Driven
I/O", and the whole `windows-rs` / `dstorage` code sample).* Not reachable through the
graphics API this engine is built on. DirectStorage's API takes an
`ID3D12Device` and writes into an `ID3D12Resource`, and **neither exists here,
because the D3D12 backend is compiled out of this engine entirely** — the
instance requests `VULKAN | METAL`, so Windows runs Vulkan. `wgpu` exposes no
NVMe→VRAM DMA and no GPU-initiated I/O surface at all, and there is no escape
hatch in use (`as_hal` / `wgpu_hal` appear nowhere in the tree). Reaching
DirectStorage means adopting a raw D3D12 path alongside wgpu — a second renderer,
on one platform. **What the engine does instead:** a tile is a borrowed slice of a
memory-mapped pack, uploaded with no intermediate copy of our own. That collects
the read syscall, the buffer copy and the decompression; only the final host→VRAM
leg remains, and the DMA engine performs that anyway.

**4.2 — GPU hardware decompression, GDeflate / BCPack (T5).** Not exposed by wgpu.
**The underlying goal is nevertheless fully met, by format design rather than by
hardware:** `.inf_tex` tiles are cooked as raw BC blocks and are explicitly
excluded from the pack's zstd compression, so there is *no decompression step at
all* on the hot path. There is nothing for a hardware decompressor to do.

**4.3 — Ray-query shadows replacing shadow maps (T7).** Built, measured and refused,
before this document arrived — the full verdict is
`docs/memos/p28-5-ray-query.md`. Quality was in the ray query's *favour* (exact
against an independent CPU ray caster: 0 of 36 864 pixels differ). It was refused
on **coverage** — the acceleration structure holds only meshlet clusters, while
this engine's shadow casters are meshlets *and* skinned characters *and* terrain
clipmaps *and* voxel volumes carved at runtime *and* a million GPU-culled scatter
instances; "a sun shadow that only knows about meshlets is not a sun shadow, it is
a second one that disagrees with the first everywhere else in the frame" — and on
**cost**: a cold build of 512 triangles measured 13.6–13.9 ms, the order of a
whole frame, at a scene size four orders of magnitude below the flagship's. It is
also Vulkan-only and native-only by wgpu's own documentation, so macOS, the WASM
player and every software adapter are excluded by construction. The experiment is
kept, armed, default-off.

**4.4 — Neural / ML predictive prefetching (T8).** Two separate refusals. *(a)* A
learned predictor is untestable under this project's gates: training
nondeterminism, and no falsifiable bound. It was replaced by a deterministic
analytic dead-reckoner over committed input history, which has the same intent and
can be gated. *(b)* **The document's 200–500 ms horizon was measured and shipped
at zero** (`docs/memos/p28-5-lead-time-ruling.md`): on a 360° whip-pan fixture the
predictor at that horizon produced *more* pop-in than no lead at all — 115 blur
frames against 105 — and against a truth oracle the prediction is 16.6° wrong in
the worst case. The structural reason is that this streamer seats a missing tile
*the frame it is offered*, so "having asked earlier" buys nothing while every want
spent on where the camera *will* be is a slot not spent on where it *is*. Named
tripwire tests re-open the ruling automatically the day an async upload path gives
the loader real latency.

**4.5 — Neural Texture Compression instead of BC7/ASTC (T9).** Deferred by memo, not
by silence. Weight-per-material NTC needs a training pipeline this project does
not have and has no falsifiable bound under the house gates. The document's own
*fallback* idea — reconstruct detail while the real tile streams — was built as a
deterministic, integer-only upscale of the finest resident ancestor
(`inf_vt::fill`) and was **measured before adoption**: plain replication scored a
texel MAE of **9.92**, beating all three interpolations it was measured against —
true texel-centre **bilinear 11.45**, the edge-directed filter **12.05**, and the
quincunx control **12.13** (`docs/memos/p26-5-missing-tile-fill.md`, the `near` /
`bilin` / `edge` / `box` columns). That 9.92 is the floor any future learned
predictor has to clear.

*(Corrected by the audit. The first draft of this row called 12.13 "bilinear",
which is the `box` column — and mislabelling that column as bilinear is the
precise error `p26-5-missing-tile-fill.md` was written to correct: `box2x` is the
quincunx lattice with the direction test removed, not what a sampler's
magnification does. The ruling is unchanged and slightly stronger: replication
beats the filter the hardware actually performs, by 13 %.)*

**4.6 — Per-tile LZ4 / Zstd disk compression (T17).** Declined deliberately, and it is
the stronger position. The `.inf_tex` container has a **uniform tile stride** —
the parser requires every tile to be exactly `tile_bytes` long at exactly
`tile_base + stride × n` — so variable-rate compression is excluded by
construction, and that construction is what makes a tile's byte offset pure
arithmetic and a tile fetch a zero-copy mmap sub-slice. Texture assets are already
excluded from the pack's zstd for the recorded reason that a compressed frame is
*"precisely the CPU-decompression bottleneck the SVT direction memo says to design
out of existence rather than optimise."* Adopting this would reintroduce, on the
hot path, the exact cost the document's own §2 asks us to eliminate.

**4.7 — KTX2 / Basis Universal (T18).** Refused on two grounds. *Build:* the Rust
`basis-universal` crate is an FFI binding over a vendored C++ encoder — the same
class of cross-OS CI liability that caused `intel_tex_2` to be deferred at Phase
4. *Purpose:* Basis's value is a **supercompressed universal** format transcoded
to a native BC format at load time, i.e. a CPU decode step on the streaming path
— see 4.6. The tiles already store the final GPU format with no transcode.

**4.8 — `io_uring` / IOCP / `O_DIRECT` ring-buffered streamer (T32).** Three
independent blockers. *(a)* The prescribed worker loop is built on `tokio`, which
is **Ring 2 by house rule**; the engine crates that would host this streamer are
Ring 0 and may not name it. *(b)* `io_uring` and `O_DIRECT` are Linux-only and the
sample uses `std::os::unix` — a one-platform I/O path is exactly the failure the
Phase 25 law names. *(c)* On the merits it is a regression here: it would replace
a zero-copy mmap slice with a copy through pinned host memory, which is *more*
work per tile, not less, absent the direct-to-VRAM leg 4.1 says is unavailable.

**4.9 — A dedicated Vulkan transfer queue and timeline semaphores (T38, T39).** `wgpu`
exposes exactly one `Queue`, no queue-family selection, and no timeline-semaphore
surface. Implementing this means taking `ash` as a direct dependency and reaching
through `wgpu_hal` escape hatches, which makes the upload path **Vulkan-only** —
macOS/Metal, the WASM player and the null console backend would each need a second
path, and the engine would then have two upload implementations that must be
proven to agree. The ordering guarantee the document wants from timeline
semaphores is already obtained without them: wgpu stages `write_texture` on the
queue such that it executes before the commands of any command buffer submitted
afterwards, so applying a whole transaction at the frame sync point is atomic with
respect to any frame.

**4.10 — The GPU compute reduction into a compacted append buffer (T49).** The *goal* —
never read a screen-resolution feedback texture back to the CPU — is already met,
and by a smaller payload than the document proposes. The *mechanism* is refused
because it is nondeterministic in two distinct places.
`atomicAdd(&request_buffer.count, 1u)` makes both the contents and the order of
the compacted list a function of how many threads ran and in what order; and the
sample's `atomicCompareExchangeWeak(&local_tile_request, 0u, packed_page_id)` —
*"atomically store the first valid tile ID found in this 8×8 workgroup"* — is a
race by design, since a different lane can win on a different run. This engine's
replay gates pin byte-identical residency traces across runs, so a want set that
depends on thread scheduling cannot ship. What is used instead is an `atomicOr`
coverage bitmask at fixed addresses: OR is commutative and idempotent, so the mask
is a pure function of (camera, request list, table) regardless of thread count,
order or duplicate writes — and the CPU reads frame *F−2*'s mask **or nothing**,
never "whatever arrived."

**4.11 — 4096-byte tile alignment (T29).** Alignment is 16 bytes, matching
`inf_asset::BLOB_ALIGN` and `.inf_vmesh`. 4096-byte alignment only buys something
if the reader uses `O_DIRECT` / `FILE_FLAG_NO_BUFFERING`, and **this reader is
mmap** — the kernel maps pages regardless. Padding every 9 248-byte BC1 tile to
12 288 would cost 33 % of the disk and of every mapped page touched, for a benefit
the I/O path cannot collect. Revisit only together with 4.1 and 4.8, which are
themselves refused.

**4.12 — TIFF sources (a T20 sub-case).** Megascans ships some 16-bit maps as `.tif`. The
workspace's `image` pin does not enable the `tiff` feature, so the map-set planner
does not claim to handle them. This is the one CANNOT on the list that is a
*decision away from being possible* rather than structurally blocked: enabling one
Cargo feature would do it, and that is a dependency-surface decision for the owner
rather than an implementation one. Until then, TIFF sources should be converted to
PNG or EXR before import — and the planner refuses them **by name**: a `.tif` in a
set earns its own advisory naming the files and the reason, rather than being
filed under "unrecognised" beside a map that genuinely has no known suffix, which
would tell the author the one thing that is not true about it. *(The advisory was
added by the audit; before it, this sentence described behaviour the planner did
not have.)*

---

## 5. One warning about the document itself

**Do not port its sampling shader.** Its tile derivation is
`fract(uv * pages_at_resident_mip)` — re-deriving the tile from `uv` at the
*resolved* mip. This codebase already paid for that bug and fixed it: the
container halves an extent with `w / 2`, so a 511-texel level sits over a
255-texel one and `uv × 255` is not `texel / 2`. Measured across a swept pyramid
set: 1 address of a 511×3 pyramid lands a whole 128-texel tile away, 50 of 1023²,
51 of 2047×511, **1 322 of 4095²**. The shipped shader walks the tile tree *down*,
one clamped `min(t/2, tiles − 1)` step per level, and a GPU test executes that walk
against the CPU's own `ancestor` address by address.

Its anisotropy claim is false at any realistic border width (§2.6), and its
feedback reduction is nondeterministic (§4.10). The document's *intent* on those
three items is right; its *implementations* are not, and the codebase's are.
