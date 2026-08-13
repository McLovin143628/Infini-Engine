# P27.2 — virtualized geometry casts through its **classic LOD chain**, not through a per-page meshlet cut

**Status:** decided 2026-08-13, during P27.2, as a **deviation** from the second
half of the batch's clause 1 (*"per-page culling on GPU (instances **and
meshlets**) against page frusta"*). The instances half is met as written; the
meshlets half is not, and this is the record of what was built instead, what it
costs, and where the real thing belongs.

## What shipped

`inf_render::vsm_raster` gives every `.inf_vmesh` asset in the scene a caster
geometry entry: the shared vertex buffer and one index buffer per **classic LOD
level**, decoded from the same `VgeomMesh::classic_lods()` chain
`passes::classic_vgeom` draws. A vgeom *instance* becomes a caster record with
the asset's own bounding sphere, its level chosen by `pick_classic_level` against
`passes::vgeom::lod_threshold` — the same two functions, at the same
`VgeomSettings::pixel_error`, that the classic tier picks with.

Those records go through the **same** per-page GPU cull and the **same** page
raster as every rigid caster, because the classic chain hands the pipeline the
same `MeshVertex` a built-in primitive does. One geometry group per (asset,
level); one `draw_indexed_indirect` per (dirty page × group).

So: **vgeom content casts.** The Phase 27 goal's "vgeom's *casts no shadows* hole
closes here" is closed as a property of the atlas —
`a_virtualized_geometry_instance_casts_into_the_pages_it_touches` reads the depth
back and finds the vmesh's surface in front of a control backdrop, with the
instance removed as the control.

## What did not ship, and why

**A per-page meshlet cut.** The literal clause would test each *(instance,
meshlet)* pair against each page frustum and rasterize the surviving cut, which is
what `passes::vgeom` does against the camera. Three things stopped it inside this
batch:

1. **The DAG cut is not a frustum test.** `vgeom_cull.wgsl` selects a cut by
   comparing each meshlet's object-space error against a *projection-derived*
   threshold, then rejects by frustum and by HZB. A page's threshold is not the
   camera's, so a per-page cut is a second, independently-tuned LOD policy — and
   the tuning question ("how much silhouette does a shadow need?") has no answer
   until there is a receiver to look at, which is P27.4.
2. **The meshlet pools are streamed and owned by a render node.** They are paged
   against camera-driven residency with a `remap` table whose entries can say
   `NOT_RESIDENT`. A shadow caster reading them would make a page's depth a
   function of *camera* residency — the P18 law this phase already gates
   (`the_gi_pass_never_reads_a_shadow_page`) pointed the other way round, and the
   same coupling in this direction is what "camera-driven residency never feeds
   lighting" exists to forbid.
3. **The cost is unmeasurable until P27.3.** Without page caching every resident
   page is re-rasterized every frame, so a per-page DAG cut would be
   `pages × meshlets` threshold evaluations per frame with nothing to amortize it
   against. P27.3 is the batch that makes a page's work conditional on the page
   being dirty, and it is the first point where the number means anything.

## What it costs, stated

* **A second copy of every DAG's vertices.** `passes::classic_vgeom` deliberately
  drops its geometry cache when the meshlet path is on (`live_asset_ids` returns
  an empty set), precisely to avoid holding two copies. The page raster holds one
  regardless, because vgeom content has to cast whichever path draws it. For the
  dense-grid fixture that is one vertex buffer per asset; for a real level it is
  the classic chain's vertex set, which is the *same* vertex set the meshlet pools
  page in — so the worst case is 2× the vertices of the vgeom content, not 2× the
  scene.
* **Silhouette fidelity is the classic tier's, not the meshlet tier's.** A shadow
  is drawn from the LOD level the *camera* justifies, which is the level whose
  error the camera already tolerates. It cannot be finer than what is on screen,
  and it can be coarser than a page at a fine clipmap level would justify.
* **No per-meshlet page rejection.** A large asset that overlaps a page by one
  meshlet draws its whole level into that page. The instance-level cull bounds it;
  the meshlet-level one would bound it tighter.

## When to revisit

**P27.3**, and it is not optional there: once a page's raster is conditional on
its content stamp, "which meshlets does this page contain" becomes the *same*
question as "which movers invalidate this page", and the two want one answer. The
per-page cut belongs in the batch that has to compute that set anyway.

Then **P28.3**, where the unified streamer merges the shadow and texture
residencies — if meshlet pages join that merge, the objection in point 2 above
goes away, because there would be one residency rather than a camera's one being
read by a light.

### P27.3 re-read this, and the routing stands — with one reason discharged

**2026-08-13.** P27.3 landed page caching and did **not** build the per-page
meshlet cut. This section is the record of what it read and what it found,
because "revisit at P27.3" is a promise and the promise has to be answered rather
than renewed.

* **Reason 3 is discharged, and the number is smaller than this memo expected.**
  It said a per-page DAG cut would be `pages × meshlets` threshold evaluations a
  frame "with nothing to amortize it against". With caching the second factor is
  the **dirty** page set rather than the resident one, and on a static scene that
  set is empty — the arm `a_static_scene_stops_rasterizing_pages_after_warm_up`
  measures it as zero after warm-up. So the cost is now a number and the number is
  *cheap*. That was the reason this memo said made the decision unmeasurable, and
  it no longer applies.
* **Reasons 1 and 2 are unchanged, and they are the ones that decide.** A per-page
  DAG cut is still a second, independently-tuned LOD policy whose tolerance
  ("how much silhouette does a shadow need?") has no answer until a receiver
  exists — **P27.4**. And the meshlet pools are still camera-driven residency with
  `NOT_RESIDENT` entries, which the P18 law forbids a light to read — until
  **P28.3** makes it one residency rather than a camera's being read by a light.
* **P27.3's own invalidation does not want the meshlet answer either.** The
  scatter that decides which pages a caster touches works from a bounding
  *sphere*, and a per-meshlet cut would refine what is *drawn* into a page rather
  than *which* pages a mover invalidates. The two questions this memo said "want
  one answer" turn out to be different questions: invalidation is about bounds and
  the cut is about detail.

So the revisit point moves to **P27.4** for the tuning question and **P28.3** for
the residency one, with reason 3 struck. The costs in *What it costs, stated*
above are unchanged.
