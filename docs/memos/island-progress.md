# The island phase — living progress

**Purpose.** The one file an island agent reads first. It says where the phase is,
what has been ruled, what is done, what is next, and what is knowingly still open.

**THE UPDATE RULE.** Every island implementer **updates this file as part of its own
commits** — not afterwards, not in a separate pass. A successor starts here, then reads
`git log` for detail. This is a *working document*: compact, factual, current. The
ledgers live in `docs/ROADMAP.md`; the numbers live in the tests that print them.

**The spec** is `docs/memos/aaa-readiness-certification.md` (the IB-1..IB-16 register and
its recommended ordering). **The doctrine** is ENGINE-FIRST: anything the island needs
that the engine lacks becomes an engine feature, never a level-local hack.

---

## Phase state

| Wave | Scope | Status |
|---|---|---|
| **I1** | foundations — IB-7 layout, IB-1 PCG/streaming, IB-10 schema window, IB-15 multi-terrain | **DONE** (+ `.inf_sm` v3 addendum) |
| I2 | the GIS door — IB-3, IB-4, IB-5, IB-6, IB-14 | not started |
| I3 | city scale — IB-2 (grammar collider LOD/budget, lot subdivision) | not started |
| I4 | the fps instrument + budgets — IB-9, IB-16, the shipping-resolution harness | not started |
| I5 | source data — IB-11 (DEM ingest reality, CRS, LiDAR) | not started |
| I6 | scale seams — IB-8 (2 047 vis instances), IB-12 (floating-origin camera), IB-13 (`SceneDoc::snapshot`) | not started |
| I7 | content — the 50 km² Vancouver map itself | not started |

Wave numbering is this file's; the certification's ordering is what it follows.

---

## Decisions (rulings that bind later waves)

* **Levels are content — `Content/Levels/`** (IB-7). `levels_root()` resolves under the
  *content* root; `levels_dir`'s manifest bytes did not change, so no project-schema move.
  *Why:* a level is an `AssetKind::Level` with a GUID, a sidecar and a dependency closure;
  a parallel root would mean two scan paths and two answers to "what does this project
  contain". Legacy `<root>/Levels/` gets a named cook advisory, never a silent skip.
* **PCG pages its own ground** (IB-1). `inf_terrain::residency::page_region` is the one
  Ring-0 rule; both hosts call a mirrored pre-pass before evaluating. *Why:* a second
  `HeightProvider` that could page would be a second spelling of "what is the ground
  here" — the defect this repo has paid for at four seams. `t.data.is_empty()` was never
  a bug; the fix makes it false.
* **Ground queries are position-aware** (found under IB-15). `ground_height_at` takes
  every terrain and returns the topmost surface that *answers*. *Why:* the old rule picked
  the lowest-`Guid` terrain with no position test, so a character walking onto a second
  terrain fell to sea level.
* **Scene schema v25 was the phase's ONE scene bump.** Carried the vehicle class only.
  Nothing else bumps the scene without an explicit ruling.
* **The blend mode's home is `.inf_sm` v3, not the scene** (IB-10 addendum, coordinator
  ruling). Per-transition `SmTransition::blend: Option<SmBlendMode>`; `None` inherits the
  session default, which `ScenePayload` v11 carries. *Why:* P29's disposition named
  `SmTransition` + a payload slot as one move; a scene field was never its home.
  **Precedence is written in exactly one function** (`PoseBlender::mode_for`) — the P29.3
  two-authorities lesson.
* **`Joint3D::contacts` is RETIRED, not deferred** (IB-10). *Why:* measured — the joint
  flag is worth **0.000000 m** of divergence once the ragdoll layer mask is written, against
  **1.339 m** for the mask, and `Collider3D` has carried the mask since P12.1. The gap was a
  generator gap and cost no schema.
* **Measure a deferral before spending a rung on it.** The window was budgeted for three
  items and carried one; measuring the other two changed their disposition.
* **A version ladder re-enters every hostile-input guard.** A frozen record decodes old
  bytes through a *different* declaration, so the live shape's guards do not run on it.
  Found on `.inf_sm` v2: `decode_wire` dispatches straight into `v2::Motion`, which had no
  depth guard, so a crafted sub-machine chain would have hit the stack before `validate`.
  The guard is now shared, not restated.
* **A payload with a migrating rung must not be diagnosed as "too old".**
  `AssetPayload::migrates_from` exists so a v2 `.inf_sm` that fails for a *structural*
  reason reports that reason. Telling an author to re-create a machine whose problem is
  hostility, not age, is the wrong-diagnosis hazard `peek_schema_version` already warned
  about — one rung before there was a way to break it.

---

## Done — wave I1

**IB-7 · the first run.** All four `inf new` templates scaffold a boot scene under
`Content/Levels/` (payload **and** sidecar); two new committed starter scenes
(`templates/blank-3d`, `templates/2d-platformer`, the latter shipping the actor its level
binds). `stranded_levels_advisory` names the migration and fires on a *successful* cook
too. The editor's no-path Save moved inside the project. CI smoke now runs `inf new`.
Measured: 4/4 templates cook and the shipped player exits 0 over 300 frames.

**IB-1 · PCG over streamed terrain, both halves.** Load path: authored 220 instances
(y 50.068…59.806) vs streamed 220, **identical**, 1 tile paged — against the
certification's 929-of-929-at-sea-level. Footprint: 2 of 16 level-0 tiles for two distant
volumes. Cell streaming: a `PcgVolume` in a streamed cell is now evaluated on activation
(495 instances, y 15.004…16.404, on the hill), restricted to what arrived. Both hosts got
the same pre-pass; `grammar_span_mirror` compares the region rule character-for-character.

**IB-10 · the schema window.** Scene **v24→v25** (the `VehicleClass` entity tail slot: 15
`f64` = `VehicleTuning::names()`, installed through `Vehicle::tune`; measured 20.251 m vs
10.121 m at full throttle). `ScenePayload` **v10→v11**. Re-bless arithmetic: **delta ==
entity_count exactly, all 19 levels**. The record's tail is generic (`V = ()` at zero
bytes, asserted), which retired four field-by-field restatements.

**IB-10 addendum · `.inf_sm` v2→v3** (coordinator-authorized). Per-transition blend mode;
v2 migrates by pure default-fill through a frozen v2 record. `ScenePayload` v11 gained the
session-default slot (folding into the unpushed bump, not a second one). The SM panel and
the text face both author it; the panel's DTO has no ts-rs bindings, so the camelCase key
assertions in `sm.rs` are its only drift guard.

Parity, as required: **12 steps, hosts identical; CrossFade differs from inherit at 12 of
12 steps in both hosts** — plus a precedence arm (per-edge beats session default; inherit
follows it; the two modes really differ).

Re-bless: `Locomotion.inf_sm` +5 B (5 transitions), `Hero Locomotion.inf_sm` +4 B (4), its
`.txt` +4 lines × 18 B = +72 B.

Three things the ladder's own arms found, all fixed:
* The frozen v2 record reached the **live** `SmState`/`Motion`, so nested sub-machine
  transitions were already paying the v3 byte on both sides — caught by
  `v3_costs_one_discriminant_per_transition` (3 transitions, 2 bytes).
* `v2::Motion` had **no depth guard**, and `decode_wire` bypasses the live one.
* A guarded refusal on a v2 payload was reported as `SchemaTooOld` — a confident wrong
  diagnosis. Hence `AssetPayload::migrates_from`.

**IB-15 · multi-terrain.** Heights **closed** (33 border samples bit-identical). Colliders
**closed conditionally** (meet exactly; conditions unenforced). Normals **open, 66.8707°**.
P16 pyramid anchor **stands verbatim**. Plus the position-aware ground-query fix above.

---

## Next — wave I2 (the GIS door)

The certification calls this *"the single largest piece of connective work the island
needs"*. `inf-gis` is 5 732 lines across nine modules with **one dependent crate using one
module** and zero non-test callers for `spawn_layer`, `read_vector`, `RoadGraph::from_layer`,
`build_all_ribbons`, `triangulate_polygon`, `classify_to_ids`.

1. **IB-3 — the door itself.** A Ring-2 command family + a GIS import wizard. Nothing else
   in I2 is reachable without it.
2. **IB-4 — roads become geometry.** `build_ribbon` returns arrays, not a `MeshAsset`, and
   takes a ground closure nothing supplies. No road surface, no collider, no terrain blend.
3. **IB-5 — the two attribute wires.** Land cover → `BiomeSet` (G10: nothing decodes a
   raster, nothing writes biome ids) and GIS attributes → `BuildingParams::floors` (G11:
   *"no code between a GIS attribute and that field in either direction"*).
4. **IB-6 — oriented lots.** `building/pass.rs::lot_of` takes an XZ bounding box, so a real
   footprint becomes axis-aligned. Vancouver's West End and downtown are both rotated.
   Wave G calls the floor-plate slicer's axis-alignment "a deep change… its own sub-phase".
5. **IB-14 — surface the caps.** `SpawnOptions::max_entities` defaults to **4 096**;
   measured 10 000 roads → 4 096 (5 904 truncated), 50 000 footprints → 4 096 (45 904). It
   is reported and never silent — but with no wizard there is nowhere to raise it and
   nowhere to show the report.
6. **IB-11's near half — `.prj` + reprojection.** The CRS is a caller-stated parameter with
   no UI to state it, and a terrain in a different CRS is *refused, not warped*. **Web
   Mercator must stay refused**: its metres are inflated ~1.53× at Vancouver's latitude, so
   accepting it would build the island half again too large with no symptom. Anchor in
   **UTM zone 10N**.

Note for I2: PCG now sees streamed terrain, so GIS-driven scatter over the real DEM is
finally a thing that can be measured rather than a thing that returns zeros.

---

## Open questions / carried bounds

**Measured and bounded (I1 owns the numbers):**
* Adjacent terrains' shading **normals differ by 66.8707°** at a shared border — `normal_at`
  clamps a missing neighbour to the centre height, so an outer edge reads a one-sided
  gradient. Asserted as a bound with an interior control; closing it fails that arm.
* The **`.inf_terrain` header origin still never places a tile** (P16 remainder, verbatim).
  A terrain's world frame is its entity `Transform`; `with_origin` is provenance.
* **Adjacent tile abutment is conditional and unenforced**: equal grids + an origin offset
  that is a whole number of tile spans. Nothing in the engine checks either.
* **`sm_save` does not write the `.inf_sm.txt`** (pre-existing). A machine saved from the
  panel leaves an existing text face stale; the two surfaces do not reconcile through the
  save door. Found during the v3 addendum, not introduced by it.
* **The SM DTOs have no ts-rs bindings** (Ring-2 types; generation runs from Ring 1), so the
  camelCase key assertions in `sm.rs` are the only drift guard for that wire.

**Pending their waves (from the certification, unchanged by I1):**
* **IB-2** — 12 850 static colliders cost 4.663 ms/step (0.363 µs each); 60 fps ceiling
  ≈ 46 000 colliders ≈ **25 archetype buildings**. One `building.plan` node is one building;
  there is no lot-subdivision node.
* **IB-8** — `VIS_MAX_INSTANCES = 2047` per frame, refused beyond; raising it re-cuts a
  packed 32-bit GPU id.
* **IB-9** — `TERRAIN_RESIDENT_BYTES_CEILING = 16 MiB` against `max_resident_tiles = 1024`
  ≈ 264 MiB. **16.4× apart**, and no scene has made them meet.
* **IB-12** — `axis_independent_lag` unrotates absolute world positions, not the delta; not
  floating-origin-safe at partition scale.
* **IB-13** — `SceneDoc::snapshot` is 0.34 µs/entity: a 100 000-entity city ≈ **34 ms per
  snapshot**, past the 33 ms tripwire before anything renders.
* **IB-16** — no per-frame time budget or upload throttle in the VT loop; only a byte
  residency ceiling. Coupled to T51 (no request→residency window).
* **No fps instrument at shipping resolution exists.** The only GPU frame harness renders
  640 × 360, and every wall-clock assertion is disabled on software/paravirtual adapters.
  Nothing may claim "≥60 fps" until I4 builds the instrument.
