# P28.3 — one streamer: the arbitration, the budget, and what aging together means

**Status:** decided 2026-08-14, during P28.3. The rulings this batch took, the
measurements that took them, and the debt it did not close, by name.

The direction memo's first paragraph names UE5's third weakness as *"three
separate page systems that can disagree"*. P28.2 made one **pair** of them
unable to disagree, structurally, for one cluster page. This batch is the
general form and it is a smaller change than it sounds: the three consumers keep
their domain brains and surrender arbitration.

| kept by the consumer | surrendered to `inf-stream` |
|---|---|
| which pages/tiles a camera justifies | the order they are admitted in |
| how a slot is seated and a table maintained | which slot may be taken from whom |
| how many bytes a page costs | how many bytes the consumer may spend |
| which counters it reports | the stamp every counter is keyed on |

---

## 1. THE CRATE: Ring 0, thiserror, and the direction of the edge

`inf-stream` is the third crate in this tree whose whole dependency list is
`thiserror`, and the reason is the same one `inf-vt`'s manifest gives twice: it
sits **strictly below** the three residency brains, so anything it names becomes
a dependency of all three and, through `inf-vgeom`, of the cook and the shipped
player.

It names **none of its consumers**. What crosses the seam is a `Lane`, a
`Stamp`, a byte count and a slot index — four scalars — plus one trait
(`SlotPool`) the consumers implement over their own state. That is what lets the
admission order be written once and executed twice without either residency
learning what the other pages, and it is what would let a *fourth* consumer join
without touching the first three.

`inf-vsm`'s own docs said the merge *"is a rewrite of both crates' residency
loops"*. It is not, and the reason it is not is the edge direction: a crate
below both can take the shared half without either of them growing an edge to
the other. The two address spaces are still different shapes — a mip level has
fewer tiles than the level below, a clipmap level has exactly as many — and
nothing here tries to unify them.

### `inf-vgeom` is deliberately NOT a `SlotPool`

Two of three consumers share the admission walk. The third does not, and forcing
it would be a lie the arbiter then reasons on: a meshlet page is a **variable
number of bytes across four suballocated pools** and its residency is a
**prefix**, not a set. There is no slot to take from anybody. What it surrenders
is the byte ceiling, the stamp domain and the coupling; what it keeps is its
worst-error-first auction.

---

## 2. ONE STAMP DOMAIN: what three counters could not answer

Three process-global monotone counters existed, with character-identical
semantics and three doc blocks each naming the other two and pointing at this
batch. They agreed on the property each was built for — never decreasing — and
had **no answer at all** for the one cross-system eviction asks: *was this
cluster page touched more recently than that texture tile?* Two numbers drawn
from two sequences are two clocks started at different times, and comparing them
is not a question with a wrong answer, it is a question with no answer.

`the_three_consumers_draw_their_stamps_from_one_sequence` observes the merge from
outside all three crates: a texture residency, then a shadow residency, then a
geometry residency, then another texture residency, and their four generations
must come out strictly increasing in that order.

### The two counters that did NOT merge, and why

`inf_terrain::data::NEXT_TILE_VERSION`, `inf_voxel`'s two and
`inf_dcc::journal::NEXT_GENERATION` are **content** versions: they stamp *what a
payload is*, so a mirror can ask "are these the bytes I uploaded". A residency
stamp records *when a slot was last wanted*, which is what an LRU orders on. The
two never meet — nothing evicts a terrain tile by comparing its version against a
texture tile's recency — so merging them would couple three more crates to the
streamer to buy an ordering nobody asks for. Five counters existed; three
merged; the split is by the question each one answers.

### The law is easier to break now, in one specific way

A per-crate counter advanced only when that crate did something, so a test that
wrongly printed one saw a small number that looked stable. One shared counter
advances whenever *any* consumer touches anything, so the same mistake now
produces obviously-garbage values. That is the better failure, and it is worth
recording because it is the one behavioural difference the merge makes to a
misuse.

---

## 3. THE PROTECTION ORDER: the defect, and what replaced the measurement

The P28.2 audit's §4(e) found the protection **priority-blind**:

> Step 3 protects every want that is already resident, of any class, before step
> 4 admits any miss, of any class — so a refinement that got there first
> outranks a floor want that has not, and on the audit's fixture a decoy
> feedback class costs the pairing its finest page.

It also found the protection **unfalsifiable in place**: one want class over a
comfortable budget never contests a slot, so an `apply_wants` that protected
*nothing* passed every arm in the invariant gate.

`inf_stream::admit_by_lane` walks the lanes in ascending order and, for each,
**protects that lane's residents and then admits that lane's misses**. A floor
miss may take a resident refinement's slot; a refinement miss may never take a
floor tile's. The guarantee P28.2 needs is strictly stronger than before rather
than merely preserved: a resident cluster page's tiles are floor wants, so they
now outrank every refinement in the pool, resident or not.

### The gate this retires, and how it stays falsifiable

`cluster_pages::a_refinement_class_under_slot_pressure_cannot_cost_a_resident_
page_its_tiles` closed by asserting the **defect** as a measured bound. Its
subject is what this batch was commissioned to remove and what the audit routed
here by name, so it is **retired rather than weakened**: renamed
`…_costs_a_resident_page_nothing`, asserting `contested == alone` on the same
fixture under the same load. The old measurement is the new arm's control —
reverting the lane walk makes `contested` fall strictly below `alone` again and
the equality fails. This is the batch's **only** deliberate gate edit.

### Three properties a plausible implementation loses, armed

* **a transaction never evicts what it just admitted** — an admitted slot joins
  `protected` immediately, *in every lane*, which is the case the single-pass
  walk could not reach at all;
* **the deferral count is exact and O(1) across every remaining lane** — once
  acquisition fails nothing frees a slot;
* **a resident want is touched even after the pool is exhausted** — otherwise a
  frame that over-asks makes its own residents look old and the next frame
  evicts exactly what this one proved was wanted. Reaching this case needs a
  **pinned** slot, and that is a property of the fix rather than of the fixture:
  a miss in an earlier lane can always take a later lane's resident slot, so the
  only resident that outlives an exhausting lane is one nothing may evict.

### The third lane ships without a producer, stated

`LANE_PREDICT` has no producer in this tree and P28.4 supplies one. A sort with
two ranks cannot see a third being mis-ordered, so the lane ships with the walk
that orders it and an arm that exercises it, rather than as a reserved constant
— the `VSM_PRIORITY_SPECULATIVE` treatment, applied to the thing that constant
was reserved *for*. `VSM_PRIORITY_SPECULATIVE` itself becomes `LANE_FEEDBACK`,
not a third lane: a shadow page has no refinement class, the marking mask *is*
the evidence and there is nothing finer to ask for. Its numeric value is
unchanged, so no committed trace moves.

---

## 4. THE BUDGET: even, not proportional, and an identity at the defaults

Before this batch three VRAM ceilings existed and **nothing bounded their sum**.
A host could be handed 344 MiB of streaming residency by a settings struct that
never says so anywhere, and the P28.2 ledger's own remainder says as much:
*"`resident_bytes` counts geometry only … a combined number is literally P28.3's
clause 2."*

`RenderSettings::stream.budget_bytes` is that bound. The three per-consumer
numbers stay and become **requests**; `arbitrate` divides: floors first, refused
**by name** when they do not fit, then an even water-fill clamped at each want.

### Why even and not proportional — measured

A proportional split hands the largest *request* the largest share, and the
meshlet pools' default request is ten times the virtual texture's. Under a
64 MiB ceiling, on the shipped numbers:

| | geometry | texture | shadow |
|---|---|---|---|
| even (shipped) | 23.33 MiB | **21.33** | 19.33 |
| proportional (the control) | 47.24 MiB | **5.78** | 10.98 |

Proportional puts the texture pool **below its own Low-tier ceiling of 6 MiB**
on a machine that asked for High — which is "high-poly mesh with a blurry
texture", produced by the arbiter that exists to prevent it. Even-with-clamping
does the opposite: small requests saturate and stop taking, and what remains
goes to whoever can still use it. The control is written out in
`an_even_split_does_not_starve_the_smallest_consumer` rather than described.

### The identity is what keeps every golden and every gate

The tier constants are each tier's own three shares **summed** — High 256+24+64
= 344 MiB, Medium 128+12+32 = 172, Low 64+6+16 = 86 — and `arbitrate` is an
identity when the requests fit the total. So at the shipped defaults every
consumer is handed exactly the number it had before there was an arbiter. That
is asserted as a `const` block, so a share lowered without the whole fails the
**build** rather than a test.

What the unified number buys is the case that had no bound at all: three 96 MiB
requests under a 96 MiB ceiling now come back as 32/32/32 instead of 288.

### Two floors, two places, and where the refusal has a producer

`RenderSettings::arbitrate_budgets` runs with **zero floors**, deliberately: a
consumer's mandatory floor is a fact about *content* — how many textures a level
registers, how big an asset's page 0 is — and a settings struct has none in
scope. Refusing a budget that cannot hold a floor stays where the numbers are
known (`VtError::MandatoryFloorExceedsBudget` at registration).

`EngineRenderer::stream_report` runs it again per frame with the **live** floors,
and that is where `StreamError::FloorExceedsBudget` has its producer over real
numbers. Two calls to one function with different inputs: the settings-level one
**sizes** the pools, the live one **audits** them.

### The meshlet pools get their first tier knob

`VgeomStreamBudget::budget_bytes` shipped in P18.2 and no tier ever touched it —
a Medium machine was handed the High ceiling by a settings struct that never
said so. It changes nothing today (Medium and Low clear `vgeom.enabled`), which
is exactly why it can land: the clamp exists before anything ships that forgets
it, the shape `vsm.enabled`'s clamp took at the *start* of P27 rather than at
the end.

---

## 5. AGING TOGETHER: what it means, and the bound on what it covers

`VgeomNode::page_tiles` becomes an `inf_stream::Coupling`, and the difference is
ownership rather than shape. Members are wanted together (P28.2's mechanism,
kept verbatim — one want set, so `apply_wants` protects every resident member
before any miss is offered a slot), touched together (one stamp domain), and —
the half a bare map could not do — **dropped together**, in the frame `pair`
retracts the page.

### Aging together is a want, not a stamp copy

The tempting implementation writes one stamp into every member's slot. It is
wrong for a recordable reason: a slot's stamp is the *residency's* bookkeeping
and the arbiter owns no residency, so writing into one from here means two
writers for one field, running at two different points in the frame. Wanting
them together achieves the same thing through the door that already exists, and
it composes with the lane order.

### THE BOUND: a shadow page is not a member, and the reason is not effort

The ROADMAP's clause reads *"a cluster's tiles and pages age together"*, and the
operational content the batch was given is entirely about **tiles** and the
cluster's own geometry pages. A shadow page a cluster casts into is **not** a
member and cannot be one at the arbiter's sync point:

* which pages a caster reaches is decided by a per-page frustum test that runs
  over the pages that are **already resident** (`vsm_raster::scatter_caster_
  stamps`), after the marking mask has been read;
* so producing that membership at the sync point means deriving *next* frame's
  page set from *last* frame's casters — which is a **prediction**, and a
  prediction enters at `LANE_PREDICT`, which is P28.4's lane and has no producer
  here.

Recorded rather than approximated. `Coupling` is generic in its member type, so
the day a predictor produces a shadow-page membership it is a second `couple`
call and not a rewrite.

### Page 0 has no coupling, and the guarantee is over pages 1..N

The P28.2 audit's §1: the root page takes the coarsest mip by fiat, which on a
full pyramid is the 1x1 texel level, so page 0's 544 triangles pair against ONE
texel — 0.002 texels per triangle against a band of 170–477 — and that mip is
pinned at registration, so the pairing demands nothing. The consequence stands
after this batch and is stated rather than closed: **page 0's membership is
vacuous, and the coupling's guarantee is over pages 1..N.** Closing it means the
root page pairing against a mip its own triangle density justifies, which is a
*cook* rule change, and a cook rule change re-derives every committed
`.inf_vmesh`. Not taken here; the ruling is that it is a cook question and not
an arbiter one.

---

## 6. ONE RING, OR ONE DOMAIN — the measurement that decided it

The obvious reading of "one feedback/readback ring" is one buffer. Refused, and
the refusal is **structural rather than about cost**:

* the virtual-texture mask is sized from the texture **registry** and rebuilt
  when a registration grows it;
* the shadow mask is sized from the **light set** and rebuilt when that changes;
* one buffer means either event rebuilds both, so **registering a texture would
  drop an in-flight shadow mask** — turning two independent, counted misses into
  one coupled miss, on a frame where nothing about shadows changed.

The saving would have been one `copy_buffer_to_buffer` and one staging
allocation per frame. So: two buffers, one **domain**. `FEEDBACK_LATENCY_FRAMES`
and the slot arithmetic move into `inf_stream::ring` and `ReadbackRing` indexes
through them; both consumers report into one `RingLedger`, which can state the
property two independent rings cannot — that they read the **same** source
frame, or one of them missed.

---

## 7. THE STALENESS CLASS, CLOSED STRUCTURALLY

The P28.2 audit measured the failure and shipped the *runtime* answer
(`can_address` separates "not paged in" from "no such tile", so the asset
degrades instead of vanishing), and routed the **format** answer here.

`ClusterTileRef`'s pad word becomes `grid`: a 32-bit digest of the tile grid the
address was cooked against. No container version moves — every v3 cook wrote
that word as zero, so a P28.2 image parses as **no claim**, which is the truth
about it rather than a default standing in for one (the argument that let
`tile_count` and `tiles_off` take v2's two zero pads).

**A digest and not a mip count, measured**: a 2 048 x 2 048 pyramid and a
2 048 x 1 024 one have the same mip count and a different grid at every level, so
a mip count calls a re-crop fresh. `grid_digest` folds the level count and every
level's `(tiles_x, tiles_y)` through owned FNV-1a arithmetic — a Ring-0 crate
must not grow a hash dependency to spell four numbers — and never returns 0,
because 0 is the no-claim sentinel. The collision that buys is one grid in 2^32
reading as a different grid, against the alternative of one grid in 2^32 silently
disabling the check for every asset paired with it.

**Checked at load, once per texture**, which is the structural half: a stale
image is detected from its FIRST tile reference rather than discovered address by
address — and in the direction the runtime answer is blind to. An image
re-imported **larger** still has every cooked address, so `can_address` says yes
to all of them and the surface streams the wrong detail level in silence.
Counted as `mismatched_textures`, and it is a *stronger* fact than `stale_tiles`
because it explains every missing address that follows. An unverified pairing is
still not a wrong one: a zero claim matches every descriptor.

---

## 8. THE DEBT LEDGER, item by item

**Landed.**

* `VsmMarkLayout::wants_at`'s dead guard (P27.1 audit → here) — deleted, and it
  was also *wrong* if it had ever fired: `break` leaves the innermost loop and
  the walk carries on into the next row. A `debug_assert_eq!` on the premise
  replaces it.
* The priority-blind protection order (P28.2 audit §4(e)) — fixed, §3 above.
* `stale_tiles` gets a reader (P28.2 audit §8) — `StreamReport`.
* The pairing→image tie, the FORMAT answer (P28.2 audit §8) — §7 above.
* `VtApplyReport`'s upload counter (P26.5 routing) — `VtPopIn::page_uploads` /
  `page_upload_bytes`. `admits` is what residency decided, this is what the queue
  paid for, and the pair diverges exactly when the mirror writes a page residency
  did not newly admit.
* The combined `resident_bytes` (P28.2 ledger) — `StreamReport::resident_bytes`,
  a **sum and not a re-count**: a cluster page's tiles are spent out of the
  texture pool by the same transaction and are counted once, there.
* The two latent shapes (P28.1 audit, carried unchanged through P28.2) —
  `flat_at[asset_id]`'s indexing panic and `VisAudit::frames` on a frame that
  draws nothing. The second was not cosmetic: an empty flat table admits, the
  counter moves, `render` reads that as "the per-pixel producer marked", and the
  streamer reads **no evidence** as *evidence of nothing* on the cold frame.

**Measured and refused.**

* **`inf_vt::fill`'s adoption window** (P26 routing: *"the unified loader may
  finally have one — measure; the ruling was replicate-wins, honor it"*). There
  is still no window. `VtTextures::sync` applies the transaction and stages its
  pages **in the same call, synchronously, from mmap slices**; a slot is never
  allocated with its bytes absent, and an admit whose bytes cannot be produced
  gets a deterministic zero page and is reported. The unified streamer changed
  the *arbitration*, not the loader: `arbitrate` sizes pools before any
  transaction and the coupling adds wants, not asynchrony. `fill` keeps no
  adoption site, and the replicate-wins ruling is honoured by leaving it alone.
* **`VtTransaction::unknown_texture`'s producer** (P26 routing: *"the arbiter is
  P28.3 — the tripwire gets its producer or is retired by name"*). It still has
  none, and it is **kept rather than retired**, because the reason it has none is
  a property that is still true: every want-emitting path filters an unknown
  handle before `apply_wants` sees it, and P28.3 did not introduce a cached want
  set that survives a level switch — the coupling is cleared and rebuilt from the
  container every frame (`cluster_tile_wants` opens with `coupling.clear()`). A
  tripwire whose premise holds is not a dead counter; retiring it would remove
  the only thing that would notice the day a want set does persist, which is
  exactly what a predictor's horizon is.
* **A fully-cached frame still packs its casters on the CPU** (P28.1/P27.5
  routing). Not closable from here, and the reason is circular rather than
  effortful: `pack_casters` computes the content stamps that `scatter_caster_
  stamps` folds into pages, and the dirty set is what those stamps decide — so
  "skip the pack when the frame is fully cached" needs the dirty set before the
  pack that produces it. A second, cheaper derivation of the same quantity is
  two derivations of one fact (the P21 one-door law). What *is* skippable is the
  packing into `VsmCasterRaw` records once the dirty set is known empty, and that
  is a `vsm_raster` restructure. **Re-routed to P28.5.**

**Re-routed, by name.**

* **The clipmap scroll** (127 pages against 4 096) → **P28.4**. The residency
  half already landed in P27.3 (*"the pages keep their slots"*); what remains is
  re-keying a resident page's coordinates under the origin shift so its
  *content* survives, and which pages are then stale is a question about the page
  cache's content stamps, not about who owns a slot. It belongs with the batch
  that owns prediction of where the clipmap is going.
* **The palette-union caster bound** → **P28.4**. It changes what a *mover*
  invalidates, which is the CPU caster cache's contract, and the P27.5 audit's
  correction stands: the 67 %/30 % figures are the margin's own reciprocal on a
  one-joint fixture, so what has to be measured is a real rig, where joints far
  apart can make the union *larger* than the inflated bind sphere.
* **The terrain deformation-window removal** → **P28.5**. P27.3 refused it
  because the window follows the camera; the P22.1 lattice is committed and
  camera-free, so consuming the field directly is the fix. It is a *caster mesh*
  change (`vsm_raster::sync_terrain`) and it moves what a shadow page contains,
  which is a frame-bytes claim with no gate in this batch to hold it.
* **The per-page meshlet cut** → **P28.5**, with its tuning half.
* **The frame-derived visibility bit split** and **a second geometry kind for
  voxels** → **P28.5**. P28.1 measured the split as more capacious and declined
  it because it moves the refusal out of registration; P28.3 supplies the number
  it needed (the streamer now publishes the meshlet pool's live bytes) and does
  not spend it, because moving the refusal out of registration is still the same
  trade and this batch added no frame-level door to take it at.
* **The resolve's occluded overdraw / `frag_depth` early-Z cost** → **P28.5**.
  The fix is a depth pre-test and it belongs with a unified depth pass, which
  this batch did not build.
* **A tangented parity row** → **P28.5**, with the skinned-tangent gate it pairs
  with. `vgeom_unpack_tangent` and `vt_apply_normal_t`'s non-fallback branch
  still never run on a device.
* **The editor's derived `.inf_vmesh` is unpaired** → **P28.5**. Closing it means
  giving the editor's derivation the cook's serial material walk, which is Ring-1
  work in `inf_editor_core::assets::vmesh`.
* **The page-border re-weigh and the wider VSM kernel** (P27.5 audit → P28.2 →
  here) → **P28.5**, and this batch does not repeat P28.2's mistake of arguing
  the premise away: the clauses name *"where interleaved cluster pages change
  what a page is"*, P28.2 satisfied that trigger on its face, and P28.3 satisfies
  nothing of it — it changed **arbitration**, not page geometry. A shadow page is
  still depth-only, still 128², still bordered by four texels. The premise
  arrives when a shadow page carries something a border interpolates.
* **The textured mip question's fixture** (flat, unlit, texture-only) → **P28.5**.
  Not built; the P28.1 audit's gate criterion remains the honest one.

---

## 9. WHAT THIS BATCH DID NOT DO

* **The three residencies are not one residency**, and are not meant to be.
  Nothing here merges an address space, a table image or a page format.
* **The vgeom auction is untouched.** Its worst-error-first heap and its prefix
  invariant are exactly as P18.2 left them; it took a ceiling and a stamp domain.
* **No golden moved, and no gate was updated to pass** except the one named in
  §3, which asserted a defect this batch was commissioned to remove, plus one
  byte pin adjusted for a rustfmt line break and called out per-gate.
* **The arbiter never runs mid-frame against a changed budget.** Re-budgeting a
  pool drops residency, so the division happens where the pools are sized (the
  settings door) and the per-frame call is an **audit**. A host that wants to
  re-divide live re-applies its tier.
