# P26.4 — two debts examined and deliberately carried

**Status:** decided 2026-08-12, during P26.4. Both were named in the batch brief;
both are carried rather than closed, and this is why. (The house rule: a
deviation gets a memo, not silence.)

---

## 1. The three-variant blend enum, spelled four times

The P26.3b ledger: *"The three-variant blend enum is spelled four times —
`inf_ecs::BlendMode`, `inf_material::MatBlend`, `inf_asset::DerivedBlend` and the
Ring-2 string map in `scene_apply_material`. Only the middle pair is pinned
against each other."* The brief's instruction: **one spelling, freeze-pinned, if
the consolidation does not move a wire byte; otherwise memo it.**

It moves wire bytes, and worse than that it cannot be done at all without a new
crate. Measured on the dependency graph:

| spelling | crate | wire it is frozen into | who must read it |
|---|---|---|---|
| `BlendMode` | `inf-ecs` | the `.inf_lvl` scene record (schema v8) | editor, player, cook |
| `MatBlend` | `inf-material` | the authored `.inf_mat` payload | editor, cook — **never the player** |
| `DerivedBlend` | `inf-asset` | the `.inf_matd` pack record | **the player**, which must not link `inf-material` |
| the string map | Ring 2 | the IPC boundary (`scene_apply_material`) | the frontend |

The three enums are in three crates none of which may depend on another:

* `inf-material` must not depend on `inf-ecs` — its own comment says so, and the
  reason is the editor glue, not taste;
* `inf-asset` must not depend on either, because `DerivedBlend` exists precisely
  so a shipped player can read a material record **without** linking
  `inf-material` (the P26.2 dependency inversion, and the whole reason
  `.inf_matd` was invented in P26.3b);
* `inf-ecs` cannot depend on `inf-material` without dragging `image` and `naga`
  into the scene crate.

So "one spelling" means a **new Ring-0 crate below all three** holding a
`Blend` enum, and then three wire formats re-pointed at it. Each of those is a
frozen, append-only discriminant written into shipped files (the P19 law), so
the change is a schema migration on `.inf_lvl`, `.inf_mat` and `.inf_matd` at
once — for a three-variant enum whose middle pair is already pinned against each
other, and whose failure mode (a fourth variant added to one and not the others)
is a compile error at every mapping site because all three matches are
exhaustive.

**Carried, with the pinning extended instead.** The cheap, honest half is to pin
the pairs that are *not* pinned, which costs nothing and catches the real
failure. That belongs with P26.5's gate work, and the ledger says so rather than
this being rediscovered.

---

## 2. Real UV and tangent streams for the rigid and skinned paths

The P26.3 ledger, carried through P26.3b: *"The rigid and skinned paths
box-project their uv (`vt_box_uv`), because the RENDER vertex streams carry none
— and a box projection on a character is visibly wrong. The asset does not have
this problem: `inf_mesh::MeshVertex` has carried `uv` and `tangent` since P4, so
wiring both through is a widened vertex buffer rather than a new idea (`uv` at
`@location(2)` / `@location(15)`, plus `skinned_mesh_data` and its player
twin)."*

**Not done in P26.4, and the reason is scope rather than difficulty.** The change
is exactly as described — no new idea — but it touches, in one indivisible step:

1. `passes::mesh::MeshVertex` and its `VertexBufferLayout` (position + normal
   today);
2. `crate::primitives`, which must now *generate* uvs for the cube, sphere,
   plane, cylinder and cone — five parametrizations, each of which is a visual
   decision an artist will later disagree with;
3. `inf_render::SkinnedVertex` and its layout;
4. `skinned_mesh_data` **twice** — `inf_editor_core::render_assets` and
   `inf_player::skinned` — which `projector_mirror` pins **character for
   character, doc block included**, so both copies move together or the mirror
   gate fails;
5. `mesh.wgsl` and `skinned_mesh.wgsl`, whose `vt_box_uv` call sites become the
   fallback for a streamless primitive rather than the only path;
6. the tangent stream, which is a second widening and is what retires
   `vt_apply_normal`'s per-fragment derivative frame.

Every one of those is a *vertex buffer layout* change, and the golden harness
compares command streams as well as pixels. Landing it beside the feedback loop
would mean one batch in which both the streaming brain and every mesh vertex
format moved, with a single bisect point between them.

**What P26.4 does instead** is leave the box projection exactly where P26.3 put
it — the documented fallback for a streamless primitive, which it remains after
the wiring lands too — and make the thing it feeds correct: the feedback and the
floor both derive their level from a *screen footprint*, not from a uv, so the
streaming loop is unaffected by which uv a fragment ends up sampling with. The
sharpness a character gets is wrong in the same way it was wrong before this
batch, and no more.

**P26.5 is where it belongs**, beside the residency heat-map and the
missing-tile fill — both of which want to be looked at on a real textured
character, which is the same session in which a wrong uv is obvious.
