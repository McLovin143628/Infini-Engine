# P27.3 — the page cache, the per-level snap, and the sun's arc as a **quantum**

**Status:** decided 2026-08-13, during P27.3, as the ROADMAP's clause-3 ruling
(*"the sun's slow arc: clipmap-level time-slicing so a moving sun re-rasterizes
coarse levels lazily inside budget — **measured policy, not a magic
constant**"*), plus the two deviations the measurement forced. Every number here
is pinned by an arm, named beside it, so re-deciding is a matter of re-reading
the arms rather than re-deriving this document.

---

## 1. The prerequisite: per-level snapped clipmap centres

P27.1's ledger carried it as the first thing this batch had to answer:

> **The clipmap's levels share one snapped centre**, which is what makes the
> concentric parent rule exact and what makes a *coarse* level's grid shift on
> any camera motion — the first thing P27.3's caching clause has to answer, and
> per-level offsets are the answer it will need.

`ClipmapLayout` (`inf_render::vsm`) is that answer. Each level snaps its centre
**in the light's own basis** to its own page stride `w_L = page0 · 2^L`, so
level `L`'s page boundaries lie on a fixed world lattice of stride `w_L`.

| what | before (P27.1) | after (P27.3) |
|---|---|---|
| level 0 grid moves every | 1 m of camera travel | 1 m |
| level 3 grid moves every | **1 m** | 8 m |
| level 7 grid moves every | **1 m** | 128 m |
| snapping axes | world x/y/z | the light's right/up/forward |

Measured as a *count* rather than as an argument
(`a_coarse_clipmap_level_holds_its_grid_while_a_fine_one_shifts`): over a
40-page camera walk, level `L` takes about `40 / 2^L` distinct grid positions,
and the arrangement this replaces — `clipmap_centre`'s shared snap — moves on
every one of them. That arm keeps `clipmap_centre` alive precisely as the
control.

### Two things the snap does *not* do, and why

**It does not snap in world axes.** The page grid is laid out in the light's
basis, so snapping the world components only lands the grid on the lattice when
the light happens to be axis-aligned. P27.1's `clipmap_centre` did exactly that;
the shipped path projects the **f64 world eye** onto the light's `right`/`up`
and snaps those, which also makes the lattice invariant under a floating-origin
rebase.

**It does not snap the along-light coordinate per level.** A page's stored depth
is `(p − centre) · forward` scaled, so a centre that slides along the light
rewrites **every** page of **every** level. Snapping it per level would make
level 0's one-metre step invalidate the whole atlas — the exact defect the type
exists to remove. It snaps once, at the **coarsest** stride, and the box is `N`
of those deep, so the snap costs `1/N` of the depth range and buys one
whole-atlas invalidation per `w_top` of travel along the light.

### What it costs: the levels stop being concentric

This is not a side effect, it is the price, and it is unavoidable: level `L` must
track the camera at granularity `w_L` and level `L+1` at `2·w_L`, so the two
centres differ by up to `1.5 · w_L` and cannot both be centred on the camera
*and* share a centre. `inf_vsm::VsmLightDesc::parent_at` is the generalization:

```
parent_x = ⌊(g_L + x) / 2⌋ − g_{L+1}
```

where `g_L` is the level's grid origin on the world lattice
(`ClipmapLayout::clip_origins`). Floor division, not truncation — a cell index is
signed. At the concentric origins `(−N/2, −N/2)` the expression **is**
`N/4 + ⌊x/2⌋`, page for page, over every page of every level of four grids
including two that are multiples of four and not powers of two
(`the_general_clipmap_parent_rule_reduces_to_the_concentric_one`). So the shipped
concentric rule is contained rather than replaced, and `parent`, `ancestor` and
the two child ranges keep their meanings as the `origins = &[]` case.

`VsmResidency::set_clip_origins` recomputes one light's fallback chain when the
origins move, and the residency gate's brute-force walk reads the light's own
origins — so the whole-table check after every transaction is against the tree
the light actually has (`moving_a_clipmaps_level_offsets_re_points_the_whole_fallback_chain`).

**The pages keep their slots.** A page whose world cell moved holds stale
*content*, not a stale allocation, and which of them is stale is a question about
content stamps rather than about residency. See §5 for what that costs and what a
*scroll* would buy.

---

## 2. The content stamp

A page is re-rasterized iff its stamp moved. The stamp is two things folded:

1. **The page's own matrix, bit for bit.** Not a proxy for what moves a page's
   content — it *is* the complete list: the quantized sun direction, the level's
   snapped centre, that level's NDC offset, the floating origin, the page's
   sub-rectangle within its level, and the settings that size the box are its
   only inputs and nothing else is one. A rebase moves it; a sun crossing its
   quantum moves it; a camera crossing a level's page stride moves *that* level's
   and not the others'.
2. **A commutative fold of every caster whose light-space bounds touch the
   page** — `model`, `sphere`, `mat` and the identity+version of the geometry:
   the primitive kind, the `.inf_vmesh` id and classic level, the skinned mesh's
   `Arc` and its **whole joint palette**, or the terrain tile's key, version and
   hole-mask length.

`VsmCasterRaw::ids` is deliberately **absent** from the fold. Its group index and
its index inside that group are properties of this frame's packing order, not of
the caster, and folding them in would re-rasterize every page in the atlas
whenever a tree was planted at the other end of the level.

The combiner is `wrapping_add` of a mixed hash: commutative, because a page's
caster set is a *set* and the order `pack_casters` emits two casters in is not a
property of the world — and unlike XOR it does not cancel a caster against its own
duplicate. The mixer is SplitMix64's finalizer written out rather than
`DefaultHasher`, which is documented as unstable across Rust releases: a hash that
changed under the compiler would turn "nothing moved" into "everything moved" on
a toolchain upgrade, silently and only in the frame time.

### Page-exact invalidation costs a scatter, not a product

The obvious implementation tests every caster against every page with the cull's
own `vsm_page_sees_sphere` — `1 024 × 16 384` sphere tests a frame at the
ceilings. Instead a clipmap caster's sphere is projected **once** into the light's
level-0 NDC and each level's page rectangle follows by arithmetic, so the cost is
`casters × levels × pages the caster actually covers`, which for anything smaller
than a page is a constant. `VsmRasterStats::invalidation_touches` is that number,
measured rather than argued. Perspective lights (spot, cube face) have no such
lattice — their levels share one frustum — so their pages take the per-page sphere
test directly, over that light's pages alone.

The rectangle is the sphere's axis-aligned NDC extent, which **contains** the
frustum test the cull performs. So the scatter may fold a caster into a page the
cull then rejects (a page re-rasterized for nothing) and can never miss a page the
cull keeps (a stale page, which is a wrong shadow). The z test is the cull's own
two planes, in the same direction.

---

## 3. The sun's arc: a **quantum**, not a frame counter

### The measurement

Rotating the sun by `δ` moves the shadow of a caster `h` metres above its
receiver by `h · δ` along the ground. The clipmap's level rule puts about one
level-0 shadow texel under each screen pixel, so *"the shadow has moved one
texel"* is *"the shadow has moved one pixel"* — which is exactly what a re-raster
buys back. The quantum is therefore

```
q = texel0 / h_ref,     texel0 = 2 · first_level_extent_m / (pages_per_side · 128)
```

with `h_ref = VSM_SUN_REFERENCE_HEIGHT_M = 2 m`, a standing figure: the shortest
caster whose own shadow a player watches, and therefore the one whose shadow moves
fastest in texels per radian at a given level. A taller caster tolerates *less*
rotation, and its shadow is correspondingly further from its foot.

At the shipped defaults (64 pages a side, 32 m half-extent):

| quantity | measured |
|---|---|
| level-0 shadow texel | **7.8 mm** |
| quantum `q` | **3.91 mrad** (0.224°) |
| quanta per revolution | **1 608** |
| frames per quantum, 20-minute day at 60 fps | **44.8** |
| frames to drain a whole-atlas invalidation (1 024 slots / 256 a frame) | **4** |

`the_sun_quantum_is_one_shadow_texel_at_the_reference_height` pins all five, and
it measures the worst residual angle through the **shipped** `quantize_light_dir`
over a 721-sample sun arc rather than through the formula beside it — and asserts
both bounds: the quantizer must leave at most about one texel of shadow travel
(a quantum that shows) and **more than a tenth of one** (a quantum that caches
nothing and is a constant pretending to be a policy).

### Why quantizing the direction and not counting frames

A clock-driven *"re-raster the coarse levels every N frames"* would make a page's
content a function of the frame **history** — the property this whole phase is
built not to have (`inf_vgeom::stream`'s ruling, and P26.2's "function of state,
not history"). Quantizing the direction is the same policy with none of that: two
runs that reach the same sun angle produce the same quantized direction, the same
page matrices and therefore the same content, whatever they did on the way there.
It is also why the quantizer uses round-to-multiple on the components and a
renormalize rather than trigonometry: `f32::sin`/`cos` are not bit-portable (the
P14 law) and this direction reaches a page's stored depth.

### The per-level half, and why it is **deferred with a number**

The literal clause asks for *clipmap-level* time-slicing — a quantum that doubles
per level, so level 7 tolerates 128× the sun motion level 0 does. The error
argument for that is sound (the shadow displacement `h · δ` is level-independent
in metres, so in *texels* it halves with every level). **The marking pass cannot
follow it**, and the number says so:

> A rotation `δ` moves a point at the rim of level `L` by `R_L · δ` where
> `R_L = N · w_L / 2`, which in **pages of that level** is `N · δ / 2` — level
> independent. At `N = 64` and level 7's quantum of `128 q = 0.5 rad`, that is
> **16 pages**, a quarter of the grid.

(The first draft of this paragraph multiplied it out in prose and wrote 32. The
arm computes it now — `the_sun_quantum_is_one_shadow_texel_at_the_reference_height`
— which is what the P22 law about inference dressed as measurement asks for, and
the ruling is unchanged either way: a quarter of the grid is not a disagreement a
marking pass can carry.)

The marking pass decides which page a pixel needs, from **one** projection per
(light, face) — the per-level variation it can follow is a *translation* in NDC,
which is exactly what `ClipmapLayout::level_offset` is. A per-level **rotation** is
not a translation, so a per-level quantum would need a per-level matrix in
`vsm_mark.wgsl`, and with a shared direction the marker and the raster would
disagree by up to thirty-two pages at the coarsest level.

So: **one quantum per light**, at the finest level's tolerance, and the per-level
laziness lives in the **drain order** instead — dirty pages are rasterized finest
level first, then by slot, under `VSM_MAX_RASTER_PAGES`. When a quantum tick
invalidates everything, the finest pages (the ones a receiver reads at the most
screen pixels) go first and the coarse ones follow over the next frames. The
stability condition is the last two rows of the table above: the drain is **11×
faster** than the thing that fills it, so the dirty queue cannot grow.

`the_dirty_drain_takes_the_finest_levels_first` pins the order, and the drain
condition is asserted inside the quantum arm.

**When to revisit:** P27.4, which rewrites the receiver side and is where a
per-level matrix in the marking pass would be paid for once rather than twice.
The measurement above is what a per-level quantum has to answer.

---

## 4. The deformation window is **not** folded into the caster mesh

The P27.2 ledger carried *"a carved terrain hole still casts: the caster mesh
reads `heights` and ignores `holes` (P21.2) and the deformation window
(P22.1)"*. The holes half is closed. The deformation half is **refused**, and this
is the record of why.

`inf_render::deform`'s window is **camera-following**: 512 texels of 0.25 m
snapped by `window_origin_texels(eye_world_xz)`, i.e. a 128 m square that moves
with the viewer. A caster mesh built through it would be a mesh whose vertices are
a function of *where the camera is* — which is the same failure the P27.2 ledger
rejected the clipmap patch for:

> **terrain** through a static per-tile mesh built from the tile's **own
> heights** rather than from the camera-fitted clipmap patch, because that
> patch's LOD, morph and skirt are all functions of where the camera is and a
> shadow drawn through them would move when the camera moved while nothing in the
> world had.

It is also the P18 law's own shape, and folding it in would make *"static pages
survive frames untouched"* false inside 128 m of the camera — the region where
every page that matters lives.

**What it costs, stated.** `DEFORM_MAX_DEPTH_M` is 1.0 m, so a rut up to a metre
deep casts no shadow of its own and the ground above it shadows as if unpressed.
The fail direction is a **leak** (lit), which is the one this phase already chose
for `VSM_ENTRY_NONE` and the one a depression in the ground wants anyway — the
opposite error would be a dark band following the player.

**When to revisit:** P28.3, where the unified streamer decides what a caster's
residency *is*. A deformation field that is committed rather than camera-paged —
the P22.1 lattice already is; it is only the *window* that follows the camera —
could be folded per tile, keyed on `RenderDeform::epoch`, with no camera term at
all. That is the shape the objection above would go away for.

### The holes half, and the rule it uses

`quad_is_holed` is the terrain raster's own poison rule
(`terrain.wgsl::is_holed`: any holed corner of a bilinear cell removes the whole
cell — the same predicate `RenderTerrain::seam_sample` and
`inf_terrain::TerrainData::height_at` spell) **widened to the caster's
decimation**. A caster quad spans `step` samples, so it is dropped when any sample
of its own block is holed; testing only the four corners would let a tunnel mouth
narrower than the stride go on casting. Erring wide removes shadow, which is the
leak direction again.

---

## 5. Honest bounds, carried

* **There is no clipmap scroll.** When level `L`'s grid shifts, every resident
  page of that level names a different world cell and is re-rasterized. A scroll —
  re-seating page `(x, y)` at `(x − sx, y − sy)` and evicting what falls off —
  would leave only the newly exposed row and column dirty, i.e. `2N − 1` pages
  instead of the level's whole resident set. At the shipped grid that is 127
  against up to 4 096. It is bounded work at a bounded cadence (`w_L` of camera
  travel), it is entirely Ring-0 and adapter-free, and it belongs with P28.3's
  merged residency, which is where re-seating a slot stops being one crate's
  private business.
* **`hole_words` in the terrain caster's cache key is redundant today** — a
  projector that writes a mask also moves the tile's version, and the mutation
  that deletes the term survives the whole file. It is kept for the case the
  P16.3b1 change-stamp doctrine allows (`version: 0` means "no stamp"), on the
  `passes::terrain::CachedTile` precedent, and this sentence is the record.
* **The `is_camera_cut` flush is redundant with the stamps**, measured the same
  way: deleting the flush while keeping its counter changes no verdict, because a
  cut moves every page's matrix and the stamps invalidate the same set.
  `a_camera_cut_flushes_the_cache_and_the_stamps_would_have_too` asserts exactly
  that — *zero* page matrices survive a 194 m cut unchanged — so the day the
  trigger stops being redundant, that arm fails and this ruling is rewritten
  rather than the trigger quietly removed.
* **`VSM_MAX_RASTER_PAGES` is now a cap on WORK, not on residency**, and it stays
  at 256: a cached page costs one stamp comparison and does not consume it. The
  re-measure is the drain row of §3's table — four frames to redraw the default
  1 024-slot atlas against the 44.8 a quantum lasts — so the cap binds only inside
  a burst and the burst drains 11× faster than it refills. It is still counted in
  `deferred_pages` and still never silent.
* **`VsmRasterStats::scatter_casters` counts survivors now** rather than the
  fallback pack's output. The truncating case has no arm: reaching it needs
  16 384 packed casters *and* a scatter batch, which costs a headless fixture more
  than the three lines it would arm — the same ruling the P27.2 audit made about
  the group ceiling's vgeom and terrain call sites. (The *non*-truncating half is
  armed since the P27.3 audit, in `the_caster_ceiling_counts_the_casters_it_refuses`:
  a 16 384-instance fixture with no scatter batch reports **zero**, and a counter
  reading the merged bucket reports all of them.)

---

## 6. What the P27.3 audit changed in this document

**§2's list of stamp inputs was complete and its arms were not.** Of the caster
half's terms, three were armed — `model`, the joint palette, and the terrain
tile's *cache key*. Deleting the vgeom LOD level, `mat`, the terrain tile's
`version`, or the whole perspective branch of the scatter each survived the entire
tree. All four are armed now (`test(vsm)` `555a90f`), and the three terms that
still survive — `sphere`, the terrain fold's `hole_words`, the skinned fold's mesh
key — are rulings with their reasons written beside them in `caster_stamp`'s docs.

**The cache key's aliasing argument is corrected.** §2 and the ledger justified
`(light, page, stamp)` by "a slot re-admitted to a page it was evicted from would
read as a hit while holding the second page's depth". The stamp's geometric half
*is* the page's world footprint, so two pages that share a stamp share the depth
they want, and a stamp-only hit is a **correct** hit. What the two label members
really do is refuse a correct hit when a clipmap level's grid shifts and the world
cell that was page `(x, y)` becomes page `(x − 1, y)` with a bit-identical matrix
— i.e. they charge a re-raster for a re-label. That is §5's *"there is no clipmap
scroll"* bound wearing the cache key, and a stamp-only key would recover part of
it for free. Measured by
`a_clipmap_grid_shift_re_labels_a_page_and_the_cache_key_pays_for_it`, which also
carries the condition that keeps the members honest: every matrix collision is
inside one light and one level.

**§1's "invariant under a floating-origin rebase" now has an arm, and a smaller
claim.** `a_floating_origin_rebase_does_not_move_the_clipmap_lattice` pins it —
but what the invariance buys is the *residency*, not a cached page. A rebase
re-writes every page matrix regardless, because the matrix is render-local, so the
atlas is re-rasterized whole either way; it happens once per
`inf_math::REBASE_DISTANCE` = 1 024 m of travel and drains in four frames. What
would churn without the world-space snap is `clip_origins`, and with it the whole
fallback chain and a table publish, for a world that had not moved. `is_camera_cut`
does **not** fire on a rebase — it reads the f64 world eye — so the flush and the
rebase are two separate triggers, not one.

**§3's drain ratio is asserted rather than described.** The arm's own comment said
8× where this document and the ledger said 11× and the arithmetic says 11.2. It
asserts `10 ≤ frames_per_quantum / drain < 13` now.

**One bound §5 did not carry.** The invalidation scatter's depth envelope
(`ndc.z ∈ [−rz, 1 + rz]`, the cull's own two planes with the sphere's radius) is
**unarmed**: tightening it to the centre alone survives the tree, because every
caster in every fixture sits well inside the clipmap's box and reaching the case
needs a caster at the far face of a kilometres-deep projection. The fail direction
of a tightened test is a *stale page*, which is the wrong one, so this is a
carried gap rather than a redundancy — P27.4's.
