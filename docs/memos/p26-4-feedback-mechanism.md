# P26.4 — the feedback pass is per-surface, not per-fragment

**Status:** decided 2026-08-12, during P26.4. A deviation from the spec text's
implied shape, taken deliberately and written down rather than left silent (the
house rule).

## The spec, and what it does not say

ROADMAP P26.4 clause 1: *"the feedback pass: per-tile coverage as an
**order-independent bitmask** (atomicOr, fixed layout — content is a pure
function of camera/scene/residency, the ruling that reconciles GPU feedback with
the replay doctrine in `inf-vgeom/src/stream.rs:52-62`)"*.

(The clause and the first draft of this memo both cited `57-68`, which starts
mid-sentence and runs six lines past the doc block into `use` statements. The
range is **52-62**, as `vt_feedback.wgsl:11` says — the same batch spelled one
citation two ways. Corrected by the P26.4 audit.)

Every property it names is delivered: the bitmask is written with `atomicOr` at a
fixed layout addressed in the same virtual-address order the residency uses, its
content is a pure function of `(camera, scene, residency)`, and it is read back at
a pinned two-frame latency so a trace is a function of committed input at a known
frame rather than of scheduling.

What the clause does **not** say — and what a reader would reasonably assume from
the phrase "per-tile coverage" — is *where the marks come from*. The textbook
virtual-texturing answer is the fragment stage: the shader that samples a tile
marks it, so the mask is exactly "the tiles this frame sampled", occlusion and uv
parametrization included.

**This implementation marks per SURFACE, not per fragment.** One compute thread
per (drawn surface × bound map) frustum-culls the surface's bounding sphere,
derives the mip level its projected footprint justifies, and marks every tile of
that level.

## Why

The fragment-side mark needs a **writable storage buffer visible to the FRAGMENT
stage**, and this renderer cannot give it one cheaply:

* the renderer runs on `Limits::default()`, which is **4 bind groups**, and all
  four are spoken for (view, lights, material, environment). The P26.3 ledger
  records the same wall: *"the renderer runs on `Limits::default()`'s 4 bind
  groups and the shared environment group is already occupied through index
  13"* — which is why the VT surface was folded into the env group at 14/15/16
  rather than given a group of its own;
* so a feedback binding has to go in the **shared environment bind-group
  layout** — the one object every lit pass, and the terrain pass, and the scatter
  pass, all build their pipelines against;
* a `storage` (writable) buffer entry in that layout makes
  `DownlevelFlags::FRAGMENT_WRITABLE_STORAGE` a requirement of **every lit pass
  on every adapter**, not of the VT path. GLES 3.1 and several mobile drivers do
  not expose it. The failure mode is not "no feedback": it is
  `create_bind_group_layout` refusing, on a device where P26.3's read-only table
  binding is fine, and taking the whole renderer with it;
* and this machine cannot falsify that. There is one adapter here and it has the
  flag. Shipping a capability requirement that CI cannot exercise and this
  machine cannot fail is exactly the class the P25 audit named — *"one-platform
  bounds red CI"*.

A second geometry pass into a reduced-resolution `R32Uint` id target (the other
classic answer) was weighed and rejected on cost: it needs a pipeline variant per
lit vertex path — rigid, meshlet, skinned, scatter — and each of those is a
pipeline the golden harness's command stream would have to grow.

## What is lost, exactly

1. **Occlusion.** A wall in front of a textured floor does not stop the floor
   from asking for its tiles. The floor's pages are paid for and not sampled.
   Bounded by the frustum test, which is the larger term in practice.

   The **frustum test itself** was wrong in the first cut of this pass, and the
   P26.4 audit measured it: the shader returned unconditionally on `clip.w <= 0`
   where the floor's `inf_render::on_screen` keeps a sphere that *straddles* the
   eye, and its NDC margin was half the floor's. So a surface the camera stands
   **inside** — which is the ordinary state of the terrain-sized quad in item 3
   below — was never marked at all, and sat at `VT_FLOOR_MAX_TILES` for ever
   while every arm stayed green, because both defects are in the "the floor
   keeps more than the feedback" direction and nothing compared the two. The two
   tests are now the same test, and
   `the_feedback_and_the_floor_agree_about_what_is_on_screen` sweeps five
   positions including one bisected into the disputed margin band.
2. **uv extent.** A surface whose material tiles across only part of its uv space
   marks the whole level rather than the tiles a fragment reached. For a level
   chosen to match the screen footprint this is bounded by
   `screen area / 128²` tiles per surface — a full-screen surface at 1080p is
   about 127 tiles — plus `VT_FEEDBACK_MAX_TILES`, which refuses a level with
   more than 256 tiles in favour of the next coarser one.
3. **Sub-surface variation.** A 200 m terrain-sized quad gets one level for the
   whole thing rather than one per screen region. That is the case where the
   per-fragment signal is genuinely better, and it is the case P28.1's visibility
   buffer makes cheap: with a VisBuffer the material-resolve pass already has
   per-pixel primitive IDs and a **compute** stage to mark from, so the
   fragment-stage capability problem above disappears entirely.

## What is kept

Everything the determinism argument rests on, and it is worth being explicit
because the deviation does not touch any of it:

* the mask is written only with `atomicOr` of a constant bit, so it is
  order-independent and idempotent;
* its layout is `inf_vt::feedback`'s — bit *n* and indirection entry *n* describe
  one tile — so the producer and the consumer cannot disagree about what a bit
  means, and the CPU scan comes out in virtual-address order;
* it is read at a **pinned** `frame − 2` or not at all (`inf_render::readback`);
* every want it produces is `VT_PRIORITY_FEEDBACK`, so a late, dropped or
  never-arriving mask leaves residency at the analytic floor exactly.

## The consequence for the floor

Because the feedback is coarser-grained than a per-fragment signal, the **analytic
floor carries more weight than it otherwise would**, and it is built accordingly:
it is camera-driven (`VT_FLOOR_MAX_TILES` per visible surface at the level the
footprint justifies) rather than the camera-free three-coarsest-levels floor P26.3
shipped. On a 4096² texture those three levels are 4×4, 2×2 and 1×1 *texels* — the
"visibly textured, not sharp" the P26.3 ledger records. The two now share one
level rule (`inf_render::justified_mip`, mirrored by `vt_feedback.wgsl`), and
differ only in the cap: the floor is bounded so it can be claimed unconditionally,
the feedback is not so the budget decides.

They now also share one **camera** rule — `inf_render::on_screen`'s branches and
`inf_render::ndc_margin`'s factor, mirrored by the same shader. That was not true
when this memo was first written; see item 1 above.

## When to revisit

**P28.1.** The visibility buffer puts per-pixel instance⊕meshlet⊕triangle IDs in
an `R32Uint` target and shades in a screen-space pass. Marking from *there* is a
compute-stage write, needs no new adapter capability, needs no second geometry
pass, and is per-fragment by construction — including occlusion, since the
VisBuffer is depth-tested. The layout, the ring, the priority split and the CPU
scan in this batch are all unchanged by that move; only the producer changes.

Until then this is the honest bound, and `tests/vt_feedback.rs` asserts what it
does deliver: the pass marks exactly the level `justified_mip` names, marks
nothing for a surface behind the camera **or beside the frustum** (two fixtures,
because the two are rejected by two different branches — measured: deleting the
NDC test alone left the behind-the-camera case passing), and its wants are
refinements that cannot take a floor tile's page.
