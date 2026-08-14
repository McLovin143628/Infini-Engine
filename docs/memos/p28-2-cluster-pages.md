# P28.2 — interleaved cluster pages: the container decision, the pairing, and the two channels that did not land

**Status:** decided 2026-08-14, during P28.2. Five decisions, each with the
measurement that decided it, and two deferrals with the numbers that make them
deferrals rather than omissions. The ROADMAP's clause 1 asks for
*"`.inf_vmesh` v3 sections **or** a pack-layout extension — decided at cook"*;
§1 is that. Clause 2 asks for partial admission to be *"impossible by
construction"*; §3 is the argument, and §4 is the gate that tries to break it.

---

## 1. THE CONTAINER RULING: `.inf_vmesh` v3 sections, carrying addresses

**Ruling: a v3 section, and it holds tile ADDRESSES rather than tile bytes.**
Three measurements decided it, in this order.

### (a) The pack has no sub-entry to extend

`inf_asset::pack` is a 16-byte header, a flat index of **60-byte fixed-stride**
records, and blobs. One GUID owns one blob. There is no section directory, no
extension table, and `PackReader::read_ref` returns exactly one `Cow` per GUID.
So *"a pack-layout extension"* is not a small change to a table — it is
`PACK_FORMAT_VERSION` 2 → 3, a new per-entry sub-directory, every reader, and
the committed `pack_v1.inf_pack` back-compat fixture. The format's own doc
already records the neighbouring idea (*"sub-entry (page/chunk-granular)
verification … deliberately out of scope"*).

And it would buy nothing the payload cannot already do: a `.inf_vmesh` entry is
**uncompressed and 16-byte aligned** (`compresses_kind(MeshletMesh) == false`),
so a section inside it is already a borrowed slice of the mmap at a known
offset — which is what `VgeomAssetReader::page_sections` has returned since
P18.2. The extension was refused on that, not on effort.

### (b) Bytes or addresses: what a copy would actually cost

The interleaving's *goal* is that a page-in reads one thing. Under mmap it
already does: the geometry page and the texture tile are both slices of **one
mapping**, because `.inf_tex` also cooks uncompressed (P26.1) and `PackTexture::
payload` hands back `Cow::Borrowed`. Two slices of one mapping are one read, and
a copy buys no I/O at all.

What a copy costs is duplication, in three independent directions, and none of
them is small:

* **Across pages.** A page is a whole LOD level of the whole surface, so its uv
  footprint is essentially the whole texture. The pairing gives each page the
  mip its detail justifies, so a copy stores mip `m` once per page that maps to
  it — and the page count and the mip count are unrelated numbers.
* **Across assets.** One material's textures are shared by every mesh drawn with
  it. Each mesh's container would carry its own copy.
* **Against the pack's own copy.** The `.inf_tex` is in the pack regardless —
  the level references the material, the material references the texture — so
  the embedded copies are *additional*.

The cook arm measures the ruling rather than restating it:
`cook_pairs_cluster_pages_with_the_tiles_their_materials_sample` asserts the
derived `.inf_vmesh` stays well under the size of the single `.inf_tex` it pairs
against, which an embedding cannot satisfy.

### (c) What the interleaving therefore *is*

Section order inside the payload: page 0's `indices · meshlets · mlverts ·
mltris · vertices · TILES`, then page 1's six, and so on. A cluster page's
geometry and its texture addresses are contiguous bytes; the page directory
names both. **That is what makes one page-in transaction possible**, and the
transaction — not the byte adjacency — is what makes the two systems unable to
disagree.

### The v3 fields, and why the directory did not grow

`PAGE_ENTRY_LEN` is still **96**. `tile_count` took the `u32` pad at offset 40
and `tiles_off` the `u64` pad at 88, both of which v2 wrote as zeros — so a v2
directory parses as *"no pairing"*, which is the truth about it rather than a
default standing in for one. A v2 image that carries non-zero values in those
lanes is refused (`a_v2_image_claiming_a_tiles_section_is_refused`): the version
says what the lanes mean and the parse holds it to that.

---

## 2. THE PAIRING: where it comes from, and the rule

### The wire, followed in the one direction it runs

A `.inf_mesh` carries material **slot names**, not GUIDs. The binding lives on a
level entity, where a `MeshRef` and a `Material` sit side by side. So:

```
  entity(MeshRef.asset, Material.asset)          the level's binding
    → derive_material_bytes(.inf_mat)            the flattening (one door)
      → DerivedMaterial::texture_dependencies()  albedo → normal → ORM
        → TiledTextureReader::vt_desc()          the tile grid per mip
```

That is a **cross-asset** fact and `cook_one` is a pure function of one asset's
bytes, so `plan_cluster_pairings` is the `plan_fractures` shape: a serial walk
of the levels before the parallel stage, handed in as read-only data. The
texture order is `texture_dependencies`' own, because that order is already the
residency contract; a mesh drawn with two materials gets the union, since a
resident page is resident for every instance of it.

### The mip rule, and the alternative that was measured and refused

> **A LOD level halves a page's triangles; a mip level quarters its texels. So
> one mip is worth two LOD levels.**
>
> `mip(level L) = min(mip_count − 1, L / 2)`, and the root page — which spans
> every level — takes the coarsest mip there is.

Two consequences make it the right rule rather than a plausible one:

* **the finest geometry pairs with mip 0.** The artifact this phase exists to
  make impossible is a high-poly mesh with a blurry texture; a rule that capped
  one level short would leave it reachable at exactly the range it exists to
  close. `cook_pairs_cluster_pages…` asserts the finest page reaches mip 0.
* **the root page's pairing is a no-op.** It pairs with the coarsest mip, which
  is the virtual texture's own mandatory floor, pinned at registration. So the
  always-resident geometry costs the coupling nothing.

**The alternative: a density rule** — pick the finest mip whose
texels-per-triangle stays under a constant `K`. Measured on the 96² fixture
(18 432 triangles at LOD 0) against a 2 048² texture: mip 0 is 4.19 M texels,
i.e. **227 texels per triangle**; at `K = 64` the rule refuses mip 0 and settles
on mip 1 at **57**. So the density rule *never pairs the finest texture level at
all*, and the artifact survives at close range — which is the one thing the
clause is about. Refused, and the number is why.

### The footprint is a bound, and the bound is the safe direction

A page's tiles are the tiles its meshlets' **uv bounding box** covers at that
mip. Conservative on purpose: a bound can name a tile no triangle touches, and
can never miss one a triangle does, and the invariant is a *superset* claim. A
uv outside `[0, 1]` is **wrapped** rather than clamped, because that is what a
sampler does with the default address mode; a non-finite uv contributes nothing,
because `as u32` saturates and a NaN bound would quietly ask for every tile
there is.

### Two hazards are advisories, never silence

`unshippable_material_textures` already reports a missing or v1 `.inf_tex`
against the **material**. P28.2 adds the same two hazards reported against the
**mesh** — `unpairable_cluster_material_advisory`,
`unpairable_cluster_texture_advisory` — because it is the *mesh's pages* that
silently lose the coupling, and a reader chasing a mesh would not think to look
under a material's name. Both say what is lost in the same sentence: that slot
gets no entry in the cluster pages, so nothing couples its tiles to the geometry
and the artifact stays reachable for it.

---

## 3. THE IMPOSSIBILITY ARGUMENT

The clause asks for partial admission to be impossible **by construction, not by
discipline**. Discipline is a comment saying *"call both"*; construction is that
there is nothing else to call. Four facts, and the claim is their conjunction:

1. **`VgeomStreamPlan::uploads` is private.** A `PageUpload` — the geometry half
   — has exactly one public source in the tree.
2. **That source is `VgeomStreamer::pair`, which demands the texture half.** It
   takes a callback that is asked, per page, for the tiles that page's materials
   sample, and it answers with `ClusterPageIn<A>`, whose every `ClusterPage`
   carries a `PageUpload` *and* a `[A]`. `ClusterPage::geometry` is the only way
   to a `PageUpload`, and it is reached through a value that already holds
   `ClusterPage::tiles` — so a consumer cannot loop over geometry with the
   texture half out of scope.
3. **A page whose tiles were refused is gone before the value exists.** `pair`
   releases its pool blocks and truncates the asset's residency *inside itself*,
   so the streamer's residency and the transaction's contents are produced
   together by one function. There is no window — not even a private one — in
   which the streamer believes a page is resident while its tiles are not.
   Residency is a prefix, so a retraction takes everything finer with it, which
   is the correct degradation rather than an approximation of one: without page
   *p* there is no state in which page *p + 1* is drawable.
4. **The one remaining caller goes through the same door.**
   `cull_visible_source`, the standalone readback helper, has no
   virtual-texture library, so every page pairs with the **empty** set — a
   complete page-in, not a partial one. Routing it here rather than around it is
   what keeps `uploads` reachable from exactly one function (the P21 one-door
   law).

### The order is part of the argument

The two halves have to be seated at **one** point of the frame. The meshlet
streamer's sync point used to live inside `VgeomNode::run`, several graph nodes
after the virtual texture's; it is hoisted to `EngineRenderer::render`, and the
sequence is: stage the geometry half → fold the tiles every resident cluster
page samples into the **same want set** as the analytic floor and the previous
frame's feedback → one `apply_wants` → pair or retract.

**One want set is the mechanism, not a convenience.** `VtResidency::apply_wants`
protects every want that is already resident before any miss is offered a slot.
So a tile belonging to a resident cluster page cannot be evicted while the page
is still resident — *between* transactions, which is where the invariant would
otherwise fail. Seat the cluster tiles in a second transaction and the invariant
holds at the sync point and breaks on the very next frame.
`the_renderers_cluster_sync_seats_the_texture_half_between_the_plan_and_the_pair`
reads `renderer.rs` and pins the order, because the churn gate proves the
mechanism and cannot see the caller.

### Reaching the node: `RenderNode: Any`

`RenderGraph::node_mut::<T>()` is how the sync point reaches the streamer. The
alternative considered was a shared `Mutex` around the streamer — a lock over a
value one thread ever touches, standing in for a borrow the compiler can already
prove. Trait upcasting has been stable since 1.86 and the toolchain is 1.97.

### The bound: a texture this level does not bind is not part of the pairing

The cook pairs against the materials the **authored level** binds. A runtime
drawing that mesh with some other material has nothing to keep in step for the
cook's texture — and demanding it would retract every page of the asset for
ever, collapsing the mesh to its root page with no path back. So the runtime
filters the pairing to textures the virtual-texture library actually registered.
The coupling protects what is in play and says nothing about the rest, and that
is a real bound on the guarantee: **a material override at runtime is outside
it.** P28.3, which owns want accounting across all three consumers, is where a
per-instance pairing would live.

---

## 4. THE GATE, AND ITS ORACLE'S INDEPENDENCE

`crates/inf-render/tests/cluster_pages.rs`, four arms, **no adapter** — both
halves are the GPU-free halves by construction, so the churn runs on every CI
leg and can be exhaustive rather than sampled.

The invariant is asserted as **world state**, after *every* transaction of a
fourteen-step churn that admits, evicts and re-admits: for every page the
streamer says is resident, every tile the geometry samples is in the atlas.
1 000+ (page, tile) pairs a run.

**The oracle derives the pairing a different way**, because a gate cannot see an
error two subsystems share:

| | the cook's | the oracle's |
|---|---|---|
| footprint | axis-aligned bound over the page's referenced vertices | point samples: three corners + a centroid, per triangle |
| tile placement | `pair_page_tiles` | arithmetic written out in the test |
| mip | `tile_mip_for_lod` | the rule above, re-derived |
| source | the `.inf_vmesh` tiles section | `to_mesh()` + `VtTextureDesc` |

Re-deriving the mip rule is deliberate. This is an **oracle**, not a unit pin,
so a one-sided edit of the rule fails here — the opposite of the P28.1 audit's
finding, where a *pin* re-implemented its subject and therefore could not see it
move.

**The control exists to fail.** The identical churn with the coupling off — no
cluster wants, and a `pair` that seats unconditionally, which is exactly the
pre-P28.2 arrangement — must **reach** the forbidden state, with more than half
the sampled tiles missing. Without it the invariant arm is satisfied by a
fixture whose analytic floor happens to cover everything.

**The other direction is measured too.** A 1 MiB atlas cannot seat the finest
pages' tiles, and the *geometry* is handed back: softer geometry and softer
texture, together, with the invariant still holding over the reduced residency
and page 0 — the floor that makes "never a hole" true — still there.

Mutation-verified, three mutations, three killed:

| # | mutation | died at |
|---|---|---|
| M1 | the mip rule shifted a level (`lod / 2` → `lod / 2 + 1`) | the churn **and** the tight-budget arm; the control survives, which is what says the control measures the coupling and not the rule |
| M2 | the retraction disabled inside `pair` | the tight-budget arm — the arm that exists for it |
| M3 | the cook pairing only its first texture | the churn, on the second texture's tiles |

M3 is why the fixture carries **two** pyramids (2 048² and 1 024²): with one
texture a mixed-up pairing passes by symmetry.

---

## 5. THE TANGENT CHANNEL: landed on the meshlet path, one word wide

P26.5 routed the tangent here with a precise reason — *"`VgeomVertex` is
position + normal + uv and has no tangent to give … it belongs in **this** edit,
where the container moves once"*. It moves, and the channel lands.

### One `u32`, not four `f32`, and the measurement that decided it

Over 16², 48² and 96² displaced grids, the components of a cooked asset's
meshlet-pool bytes:

| | 16² | 48² | 96² |
|---|---|---|---|
| vertex records | **56.7 %** | **55.4 %** | **54.5 %** |
| meshlet descriptors | 5.50 % | 5.40 % | 5.42 % |
| micro vertex-index lists | 18.96 % | 19.53 % | 19.94 % |
| micro triangle indices | 18.82 % | 19.70 % | 20.12 % |

(The descriptor share reproduces P28.1's independently-measured 5.26 – 5.44 %.)

So the widening options cost, **as a fraction of the whole streaming budget**:

| tangent representation | bytes/vertex | pool cost |
|---|---|---|
| `[f32; 4]` | +16 | **+27.3 – 28.4 %** |
| two `f32` (octahedral) | +8 | +13.6 – 14.2 % |
| **one packed `u32`** | **+4** | **+6.8 – 7.1 %** |

A quarter of the geometry budget is not what a second-order quality feature
costs. `VgeomVertex` is **36 bytes**.

### The exponent is pinned, and that is the P28.1 hazard closed rather than bet on

The vertex pool is bound as `array<f32>` in four shaders, and there is no room
for a second `u32`-typed view of it: `vis_resolve` is a FRAGMENT pass and P28.1
measured `Limits::default()`'s **8 fragment storage bindings fully spent**. So
the tangent word is loaded as an `f32` and `bitcast` back — and an arbitrary
32-bit payload read that way has two hazards, both of which the packing removes
rather than relies on surviving:

* every `u32` below `2^23` has an all-zero exponent field, so as a float it is
  **subnormal**, and WGSL permits an implementation to flush subnormals to zero
  (the hazard `p28-1-visbuffer.md` §3 names, where a flushed slot silently means
  "no texture");
* an all-ones exponent field is a **NaN**, whose payload bits an implementation
  may canonicalize.

Fixing bits 23..=30 at 127 makes every packed word a normal float in ±[1, 2).
The cost is nine bits of payload, which leaves 11 bits per octahedral axis and a
handedness bit — measured worst `1 − cos θ` of **2.03 × 10⁻⁶**, i.e. **0.115°**,
over a 96 × 96 × 2 sweep of both hemispheres.

A useful consequence: `NO_TANGENT` is **zero**, and zero is structurally
unreachable from the packer, so the sentinel costs no direction and needs no
reserved code.

### What consumes it

`vt_apply_normal_t` in `vt_sample.wgsl`: Gram-Schmidt against the interpolated
normal — because linear interpolation of two unit tangents is neither unit nor
perpendicular to the interpolated normal, and a non-orthonormal frame skews the
mapped normal in a way that reads as a *lighting* error rather than a tangent
one — falling straight back to `vt_apply_normal`'s per-fragment cotangent frame
when `w == 0`, which is every asset cooked before v3. The channel is additive:
a surface without one shades exactly as it did before this batch, which is also
why **no committed golden moves** (the shared grid fixture builds without
tangents unless a caller asks).

The resolve interpolates the tangent by its **solved** barycentrics and takes
the handedness from the **nearest corner** rather than blending it: averaging
+1 and −1 across a uv seam gives 0, which this frame reads as "no tangent".

### The DAG does not move

`meshopt` sees positions through a `VertexDataAdapter` at `VERTEX_STRIDE`, and
the stride is the only thing the tangent word changes about what it sees.
Asserted rather than assumed by `the_tangent_stream_does_not_move_the_dag`,
which builds one geometry with two different tangent streams (and one with
none) and compares every meshlet, micro-index and level — mutating a clone
within one run, because `meshopt`'s output is not comparable across platforms
(the P18 law).

---

## 6. THE SKINNED WALL: the format question is answered, the channel is not landed

**Ruling: the skinned path keeps its per-fragment cotangent frame in this
batch, and what changed is that the deferral is no longer a format problem.**

The wall is real and exactly what P26.5 measured: `passes::skinned` asserts
`VERTEX_ATTRIBUTES.len() + INSTANCE_ATTRIBUTES.len() == MAX_VERTEX_ATTRIBUTES`
with `MAX_VERTEX_ATTRIBUTES = 16`, and `@location(15)` is the uv that P26.5 put
there. P26.5 offered two ways past it — raise the limit, or pack uv + an
octahedral tangent into one attribute — and declined to choose *"for a reason
better than 'there was a debt open'"*.

This batch answers the format half:

* **Raising the limit is the wrong door**, and P25's audit named the class: a
  renderer that has never raised a limit and then does fails at
  `create_pipeline_layout` on whatever adapter grants exactly the default, and
  it fails on a user's machine rather than in CI.
* **The pack is now provably safe.** P26.5's own sketch was *"uv.xy + an
  octahedral tangent bitcast into a `Float32x4`"* — and bitcasting an arbitrary
  packed word through a float attribute is exactly the subnormal/NaN class §5
  describes. With the exponent pinned it is not: the word is a normal float, so
  `Float32x3` at `@location(15)` = `(u, v, bitcast(tangent))` carries it exactly,
  at **`SkinnedVertex` 64 → 68 bytes** and three attribute offsets moved, with
  no new attribute and no raised limit.

**What is not done, and it is honest to say it plainly: the edit.** It reaches
`scene::SkinnedVertex`, both mirror-pinned copies of `skinned_mesh_data`,
`deformed_skinned_mesh`, the pipeline's attribute table and
`skinned_mesh.wgsl`; and unlike the meshlet path — where `visbuffer_parity` is a
twelve-arm per-pixel nucleus — the skinned path's only frame-level check is the
golden set, which is byte-frozen. Landing a real tangent frame on skinned
characters is a *visible* change to skinned shading with no gate that could
distinguish an improvement from a regression, and this batch is not the one to
add that gate.

**Routed to P28.3** with the recipe above, which is one edit rather than a
decision. What P28.3 inherits, precisely: the format is `Float32x3` at
`@location(15)`, the packer and unpacker already exist and are mirror-pinned,
the producers are `inf_mesh::MeshVertex::tangent` (real, authored) for skeletal
meshes and `inf_render::box_uv`'s own analytic `∂p/∂u` for deformed garment and
hair geometry, and the missing piece is a skinned-path quality gate.

---

## 7. MESHLETIZING VOXEL CHUNKS: split, with the two facts that split it

P28.1 re-routed `voxel.wgsl`'s missing shadow receiver here, on the finding that
the visibility packing's meshlet field is a slot in the **shared meshlet pool**
and a voxel chunk has none: *"the two real doors are meshletizing voxel chunks
(P28.2 — it also closes casting, GI and the prepass) and the env group alone"*.

**Ruling: it is not a P28.2 clause, and the reason is that a voxel chunk is not
cook-time content.** Two facts, both structural, one of them measured:

1. **The builder is host-only, by design.** `inf_vgeom::build` is
   `#[cfg(not(target_arch = "wasm32"))]` because it compiles `meshoptimizer`'s
   C++ through `cc`; the browser player loads *pre-cooked* DAGs and links no C++
   toolchain (P14.2). A voxel volume with `runtime_carve` enabled (P21.4) has
   no pre-cooked DAG to load: its surface is a function of what the player dug
   thirty milliseconds ago. Meshletizing it means running the builder at
   runtime, on every platform, including the one that cannot have it.
2. **And the cost is not incidental.** Measured on this machine, release build,
   `build_vgeom` on Surface-Nets-chunk-shaped inputs: **0.75 ms** at 128
   triangles, **1.45 ms** at 512, **3.19 ms** at 1 152. P21.4 re-meshes every
   dirty chunk on the fixed step, and a carve dirties several; that is
   milliseconds per chunk per dig against a 16.7 ms frame, on the CPU, on the
   simulation thread. The DAG build is already *"the heaviest single cook
   stage"* (P15.1's own span), and this would move it onto the hot path.

So the honest shape is not "meshletize voxel chunks" but **"give the visibility
packing a second geometry kind"**, which is the frame-derived bit split P28.1
already routed to **P28.3** — where one streamer decides pool capacities and
therefore knows the field widths before a frame starts. A voxel chunk would then
be named by an id space that has room for it, without a DAG and without a
builder on the hot path.

**What P28.3 inherits, precisely:** `voxel.wgsl` still has an analytic 3D light
loop and **no shadow receiver, no GI contribution, no depth prepass entry and no
caster**; the door is the id space, not the mesher; and the cheap door (binding
the env group to `voxel.wgsl` alone) remains refused for the reason P27.5 gave
and P28.1 did not weaken.

---

## 8. WHAT P28.2 DID NOT CLOSE, precisely

* **The editor's derived `.inf_vmesh` is unpaired.** `inf_editor_core::assets::
  vmesh::build_payload` is handed one mesh's geometry with no project in scope,
  and a pairing is a fact about the mesh's *materials*. An unpaired v3 asset is
  not a degraded one — the tiles sections are empty, the coupling is inert, and
  the editor viewport streams exactly as it did — but the editor does **not**
  get the guarantee the shipped build gets. Closing it means giving the editor's
  derivation the same serial material walk the cook has.
* **A runtime material override is outside the guarantee** (§3's bound).
* **The pairing is per PAGE, not per meshlet.** A page is a whole LOD level of
  the whole surface, so its uv footprint is the whole texture and its tile set
  at a fine mip is the whole mip level. That is *correct* — full geometry detail
  costs full texture detail, and the retraction makes the two degrade together —
  but it is coarse: a camera looking at one corner of a large mesh pins the
  whole level. Per-meshlet pairing needs a residency granularity finer than the
  page, which is P28.3's.
* **`resident_bytes` counts geometry only.** A cluster page's tiles are spent out
  of the virtual texture's own budget, by the same transaction; counting them in
  both would spend them twice. What does not exist yet is a *combined* number —
  and one budget over both consumers is literally P28.3's clause 2.
* **Nothing measures the retraction's cost in frames.** `retracted` counts
  pages handed back; what it does not say is how long an asset then sits at a
  coarser cut waiting for atlas room. That is a predictor question (P28.4).
* **The P28.1 audit's two latent shapes are untouched**, as they were left:
  `flat_at[asset_id]` is an indexing panic if the flat-table loop and the draw
  loop ever filter differently, and `VisAudit::frames` increments on an admitted
  frame that goes on to draw nothing. This batch changed neither loop.
* **The textured mip question's fixture** (flat, unlit, texture-only) was not
  built here: the tile pairing work never needed one, so it stays routed to
  P28.3 with the gate criterion the P28.1 audit recorded.
