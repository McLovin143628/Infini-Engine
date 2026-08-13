# P26.5 — the uv streams landed, the tangent stream did not

**Status:** decided 2026-08-12, during P26.5. The carried debt from
`docs/memos/p26-4-carried-debts.md` §2 is **half closed and half deferred**, and
this is the measurement that split it. (The house rule: a deviation gets a memo,
not silence.)

---

## What the debt was

The P26.3 ledger, carried through P26.3b and P26.4:

> *The rigid and skinned paths box-project their uv (`vt_box_uv`), because the
> RENDER vertex streams carry none — and a box projection on a character is
> visibly wrong. … `inf_mesh::MeshVertex` has carried `uv` **and** `tangent`
> since P4, so wiring both through is a widened vertex buffer rather than a new
> idea (`uv` at `@location(2)` / `@location(15)`, plus `skinned_mesh_data` and
> its player twin). … **No tangent stream**, so `vt_apply_normal` derives a
> cotangent frame per fragment from screen-space derivatives.*

P26.4's memo named P26.5 as where it belonged: *"beside the residency heat-map
and the missing-tile fill — both of which want to be looked at on a real textured
character, which is the same session in which a wrong uv is obvious."*

## What landed

**The uv, on both paths, at exactly the two locations the memo predicted.**

| stream | struct | attribute | filled from |
|---|---|---|---|
| rigid | `passes::mesh::MeshVertex` | `@location(2)` | `crate::primitives` (five generated parametrizations) and `passes::classic_vgeom` (the authored `VgeomVertex::uv`, which it had been dropping) |
| skinned | `scene::SkinnedVertex` | `@location(15)` | `skinned_mesh_data` — **both** mirror-pinned copies — reading `inf_mesh::MeshVertex::uv` |
| fracture | `scene::RenderFractureVertex` | `@location(2)` | the `.inf_fracture` chunk's own vertex, because this stream is bound against `mesh.wgsl`'s pipeline and a stride disagreement is not a subtle bug |
| cloth / hair | `scene::deformed_skinned_mesh` | `@location(15)` | `inf_render::box_uv`, the retired projection, computed once per vertex |

`vt_box_uv` is **gone from WGSL**. Its last legitimate consumer is deformed
garment and hair geometry — simulated positions over a topology, with no authored
parametrization to inherit — and that consumer now takes the projection once per
*vertex* where the stream is built rather than once per *fragment* where it was
read. Strictly less work, and it puts the fallback next to the one thing that
needs it. The cube's generated uv is computed by calling `box_uv`, not by
transcribing it, so the one shape a dominant-axis projection was already exactly
right for is provably unmoved (`the_cube_uv_is_the_projection_it_replaces`).

One repair fell out of the widening. The cylinder's side ring was **wrap-shared**
— the last quad's right edge was column 0 again — which is free while there is no
uv and wrong the moment there is one: the seam segment would have run `u` from
`23/24` back to `0`, i.e. the whole texture mirrored into one twenty-fourth of
the barrel. It has a duplicated seam column now, the same one `sphere_geometry`
has always emitted. Positions and normals are unchanged.

## What did not land, and the two numbers that decided it

**No tangent stream.** Two measurements, either of which is sufficient on its own.

### 1. The skinned pipeline is at the attribute wall

`Limits::default()` grants `max_vertex_attributes: 16`, and the renderer has
never raised a limit. The skinned pipeline's addresses:

```
0  position      2  joints       4..=14  the instance block (11 attributes)
1  normal        3  weights      15      uv   ← P26.5
```

Sixteen. The uv took the last address there is. A tangent cannot join it without
either raising the limit or packing two channels into one attribute (uv.xy +
an octahedral tangent bitcast into a `Float32x4`), and both are decisions that
should be taken for a reason better than "there was a debt open".

### 2. The rigid stream has room, and nothing real to put in it

The rigid pipeline uses `0,1,2` + `3..=13` = 14 of 16, so `@location(14)` is
free. But a stream is only as good as its producers, and this one has two:

* **`crate::primitives`** — could supply an analytic tangent for all five shapes;
* **`passes::classic_vgeom`** — reads `inf_vgeom::VgeomVertex`, which is
  `position + normal + uv` and carries **no tangent at all**.

So a tangent attribute on the rigid path would be real data on five built-in
shapes and a derived guess on every *imported* mesh — which is the entire content
a real project draws through it. The meshlet path (`vgeom_mesh.wgsl`), which is
what a real imported mesh draws through on a capable adapter, has the same gap
for the same reason: the tangent would have to reach `VgeomVertex`, which means a
`.inf_vmesh` container revision and a meshlet-builder change.

**That container revision is already scheduled.** ROADMAP P28.2 rewrites
`.inf_vmesh`'s sections to interleave cluster pages with their texture tiles
("`.inf_vmesh` v3 sections or a pack-layout extension — decided at cook"). A
tangent channel belongs in *that* edit, where the container moves once, and not
in a v2.5 that moves it twice.

## What it costs until then

`vt_apply_normal` keeps its per-fragment cotangent frame, derived from the
screen-space derivatives of the world position and the uv. Exact for a planar
patch, first-order elsewhere — unchanged in *kind* from P26.3.

What changed is what it is derived **from**. Through P26.3 and P26.4 the frame
was built from a box-projected uv, so on a character it was first-order correct
against a projection nobody authored; it is now first-order correct against the
artist's own parametrization. The visible defect the P26.3 ledger named — *"the
seams fall on the dominant axis rather than on the artist's, and a face's texture
will not line up with its head"* — is closed by the uv alone. The tangent buys
normal-map orientation accuracy on curved, non-planar patches, which is a second
order of quality on a feature that until this batch had none.

## Where it is routed

**P28.2.** The `.inf_vmesh` section rewrite is where `VgeomVertex` gains a
tangent, and the rigid/meshlet streams gain theirs with it. The skinned path's
attribute wall is a separate decision and is named in the ROADMAP's Phase 26
completion block so it is a remainder with an address rather than a memory.
