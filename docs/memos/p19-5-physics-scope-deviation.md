# P19.5 deviation memo — touching `inf-physics` outside the batch's stated scope

**Date:** 2026-08-02
**Batch:** P19.5 (Building & interior grammar + the Phase 19 gate)
**Status:** accepted deviation, recorded per §12's execution doctrine
("deviations require a memo, not silence").

## The stated scope, and what was done instead

P19.5's file boundary was:

> `crates/inf-pcg/**` (the bulk), `crates/inf-ecs` read-mostly (a collider-carrying
> module field only if the physics decision requires it), `runtime/inf-player` +
> `inf-packager`, `editor/crates/inf-editor-core`, `editor/studio`, samples,
> ROADMAP.

`crates/inf-physics` was **not** on that list. The batch modified it anyway:

* `src/d3/ecs.rs` — `pcg_structure_snaps` / `structure_snaps_of` /
  `pcg_structure_guid`, the `structure_stamps` change-stamp map, and
  `sync_retaining`;
* `src/d3/mod.rs` — one re-export;
* `Cargo.toml` — a dev-dependency on `inf-pcg` for the mirror-pinning test;
* `tests/pcg_structures_3d.rs` — new.

`crates/inf-ecs` was also written to rather than read: `ScatteredSolid`,
`PcgVolume::structures`, `PcgVolume::structures_gen` and
`PcgVolume::set_structures`. That one is *within* the stated allowance ("a
collider-carrying module field only if the physics decision requires it"), and
the physics decision did require it — but it is worth listing beside the
excursion, because the two are one change.

## Why the deviation was taken

The batch's headline requirement is **"every building enterable in PIE: openings
are real gaps (no collider blocking a doorway), floors are walkable surfaces,
stairs connect them."** Every clause of that sentence is a statement about
colliders.

The audit that the scope note asked for ("check how `ScatteredInstance`-placed
content gets colliders today") returned an unambiguous answer: **it does not, and
categorically cannot.** A `ScatteredInstance` is not an entity, carries no `Guid`,
and `PhysicsBridge3D::sync_from_world`'s world walk keys on exactly that. PCG
content was render-only by construction, not by omission.

That left three options:

1. **Emit real ECS entities per structural module.** Thousands of derived rows in
   `.inf_lvl`, a despawn-before-re-evaluate pass in two hosts, and undo meaning
   something new — all to express data that is already a pure function of the
   graph and the terrain. Rejected on cost and on schema impact.
2. **Derived state on the volume + one bridge walk** (taken). Zero schema
   movement, zero projector-mirror movement, one site serving the editor and the
   player. Costs ~70 lines in a crate the batch was not scoped to touch.
3. **Ship the buildings render-only and call them enterable.** This is the option
   the scope boundary literally permitted, and it would have made the phase's
   headline claim false. A gate can assert that a `Vec<ScatteredSolid>` has the
   right shape without a single one of those boxes ever reaching a simulation.

Option 3 is the one worth naming, because it is the failure mode a scope
boundary can *cause*. The boundary existed to keep the batch reviewable, not to
license a feature that does not work.

## Why it is a small deviation rather than a large one

* The change is **additive and local**: one extra source of `EntitySync3D` inside
  an existing gather, reconciled by the existing `sync` machinery. No existing
  code path changed behaviour; no existing test changed.
* The dependency direction is **unchanged**: `inf-physics → inf-ecs`, both Ring 0.
  `inf-ecs` still never names physics. The new `inf-pcg` edge is a
  **dev**-dependency only, for the mirror pin.
* It is **covered**: `crates/inf-physics/tests/pcg_structures_3d.rs` (7 tests),
  mutation-verified — commenting out the one new line in `sync_from_world` fails
  three of them.

## What it cost, and what was done about it

The first version of the walk re-described every solid on **every fixed step**.
`sync_from_world` runs at 60 Hz over the whole world, and the Phase 19 town is
~13 000 immovable boxes; the load-time budget arm never saw it. Measured on the
committed sample: **11.62 ms/step** against a 16.7 ms 60 Hz budget.

The fix is a change stamp (`PcgVolume::structures_gen`, bumped by
`set_structures`); the bridge retains an unchanged volume's colliders instead of
rebuilding them. Measured after: **4.94 ms/step** — 6.7 ms of the budget
reclaimed. Both figures are printed by
`phase19_gate::stepping_the_town_stays_cheap_with_thirteen_thousand_colliders`.

That regression is itself an argument for the memo: an excursion into a crate the
batch was not scoped for is exactly where a per-frame cost hides, because the
reviewers of the *scoped* files have no reason to look at a fixed-step loop.

## Follow-ups this opens

* **Phase 22 (destruction) will not want these.** They are static boxes; fracture
  wants chunk hierarchies with a structural-integrity graph. The `structures`
  cache is not the seam destruction should grow from, and P22 should say so
  explicitly rather than inherit it.
* **Evaluation still runs once, at load** (the standing P10.6 remainder), so a
  `PcgVolume` in a streamed partition cell spawns with no building and therefore
  no colliders. The Phase 19 sample marks every volume `AlwaysLoaded` for that
  reason. Closing the remainder is the prerequisite for streamed procedural
  buildings, and the change stamp is already the mechanism a streamed
  re-evaluation would signal through.
* **2D has no equivalent.** `PhysicsBridge2D::sync_from_world` was not touched;
  there is no 2-D PCG structure concept and none is implied.
