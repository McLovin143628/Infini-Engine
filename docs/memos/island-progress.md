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
| **I1** | foundations — IB-7 layout, IB-1 PCG/streaming, IB-10 schema window, IB-15 multi-terrain | **DONE + AUDITED** (+ `.inf_sm` v3 addendum) |
| **I2** | the GIS door — IB-3, IB-4, IB-5, IB-6, IB-14, IB-11's near half | **DONE + AUDITED** — battery 296 / 5 618 / 0 / 13, frontend 702 / 78, goldens 54, clippy 0, rustdoc 443 |
| **I3** | city scale — IB-2a/b/c, IB-8, IB-13, the city fixture | **DONE** — see below |
| I4 | the fps instrument + budgets — IB-9, IB-16, the shipping-resolution harness | not started |
| I5 | source data — IB-11 (DEM ingest reality, CRS, LiDAR) | not started |
| I6 | scale seams — IB-12 (floating-origin camera) *(IB-8 and IB-13 were pulled into I3)* | not started |
| I7 | content — the 50 km² Vancouver map itself | not started |

Wave numbering is this file's; the certification's ordering is what it follows. **I3 pulled
IB-8 and IB-13 forward out of I6**: both are ceilings a thousand-building fixture walks into
on its first frame, and measuring them against a real city was cheaper than measuring them
twice.

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
* **A gate must run the DOOR, not the function behind it** (I1 audit). Three of I1's four
  items were armed by calling the fixed Ring-0 rule directly, so the *wiring* — the one call
  site in each host — was unarmed, and stubbing it left all 64 `inf-player` test binaries
  green. An arm that builds the rule's inputs by hand measures the rule; only an arm that
  goes through the boot path measures the fix.
* **A `contains` needle that is a prefix of a declaration can never fail** (I1 audit).
  `contains("page_terrains_for_pcg(")` reads TRUE off `fn page_terrains_for_pcg(`, so a
  host that defined the pre-pass and stopped calling it satisfied the mirror gate perfectly.
  Use-site pins are **counts**, not substrings.
* **A gate against a cheaper alternative has to price the alternative** (I1 audit). "Not a
  bounding box" was asserted as `paged < 16` over two volumes on the same `z` row, where a
  union costs 4 — so the mutation the gate names passed it. The fixture now puts them on a
  diagonal and *measures* the union at 16.
* **A rule that changes has to change everywhere it is written down.** IB-15 retired the
  lowest-`Guid` ground pick and left ~8 doc sites in five crates still teaching it, plus one
  function (`inf_ecs::deform::ground_terrain`) whose doc asserted an invariant the change had
  just broken. Corrected, and the broken invariant is now a measured bound.
* **One import door means one PLAN, and a plan is a value** (I2). `inf_gis::import`
  owns every import decision — the `.prj` read, the CRS resolution, the naming, the stub
  floor, the entity cap, the stream channel — and hands back a `SpawnPlan`. The editor
  *applies* one; `inf gis` *prints* one. `SpawnPlan::digest()` folds every name, kind and
  coordinate **bit pattern**, so "the wizard and the CLI import the same file identically"
  is a comparison a test makes across a process boundary rather than a sentence a comment
  claims. *Why a digest and not a formatted report:* a rounded comparison passes for two
  imports that differ by a metre in the eighth digit.
* **A road has to be subdivided ACROSS its width, not only along its length** (IB-4).
  Resampling the spine at the terrain's pitch closes the longitudinal gap and does nothing
  for the transverse one. The first builder had one quad across, and the longitudinal fix hid
  the transverse defect perfectly, because both are "the road does not follow the ground" and
  only one was measured. Both alternatives are priced and **printed** now (I2 audit), on a
  14 m arterial at a 1 m step: **0.000750 m** with both axes subdivided, **5.0490 m** on the
  centreline's own vertices, **0.0495 m** with one quad across.
* **Plan in the lot's frame, place in the world's** (IB-6). An oriented `Rect2` would have
  meant an OBB through the slicer, the adjacency test, the wall builder, the roof, the
  stairs and the furniture grid — and `partition::adjacencies`' world-axis `same_line`
  test would have found **zero** doors between rotated rooms, i.e. a building with no way
  through it. Instead the plan is built in lot coordinates, where every existing rule is
  already correct, and one function (`assemble::place_in_frame`) turns the finished output
  into the world. The identity frame is skipped by an exact comparison, so nothing a level
  already contains moves.
* **A lighthouse is not a house** (IB-5b). `kind.contains("house")` classified
  `lighthouse_platform` as a detached dwelling and `contains("mall")` does the same to
  anything *small*. A use code is a phrase of whole words; the table matches TOKENS with a
  prefix.
* **The cap belongs at the door, the report belongs before the import** (IB-14). The
  certification's complaint was never that truncation was silent — it is reported — but
  that there was nowhere to raise it and nowhere to see it. Now: one cap in Ring 0, a
  wizard field backed by `EditorSettings::gis_max_entities`, a `capNote` that says what
  will be dropped *before* the button, and `inf gis plan` **exiting non-zero** when it
  fires so a pipeline stops instead of shipping a city with a hard edge.
* **An arm at datum zero cannot see an ordering** (I2 audit). "The vertical unit is applied
  before the projection, because the anchor subtracts a metric datum height" is a claim about
  `origin_height_m`, and the arm anchored at `origin_height_m = 0` — where converting before
  and converting after are *algebraically the same expression*. Moving the multiplication
  past the anchor passed all 78 tests in the crate. When a claim names the thing that makes
  two orderings differ, the fixture has to contain it.
* **A number that only lives in a ledger drifts** (I2 audit). Two of this wave's reported
  figures did not reproduce — its own commit range, and the road's mid-span pair — and both
  were numbers no test printed. Every claimed measurement now has a `println!` beside its
  assertion, which is this file's own rule ("the numbers live in the tests that print them")
  applied to the ledger as well as to the tests.
* **A radius, not a partition CELL** (IB-2a). The obvious reading of "band by the
  partition" is to admit the colliders of active cells. The measurement refuses it:
  `DEFAULT_CELL_SIZE_M` is 256 m and the activation radius is 256 m, so the active set is at
  least a 3 × 3 block — ≥ 590 000 m², ~840 buildings of interiors, two orders past the
  budget. **The cells decide what EXISTS; the band decides what is SOLID**, and they are
  different questions at different scales. What the band *does* take from P16 is its
  **anchors**: `StreamingSource` entities, exactly the set cell activation reads, so the two
  cannot disagree about where the simulation is.
* **The band fails OPEN.** No streaming source ⇒ no banding, which is every level committed
  before the island and every unit fixture. Dropping colliders is the direction that drops a
  body through the world and keeps it falling; keeping them is merely slow.
* **Anchors are quantized to a 16 m lattice, and the cost is a number.** The band's
  membership rides the P19.5 change stamp, and a stamp that moved every step would
  re-describe the active set 60 times a second — the 11.62 ms regression that memo records.
  Snapping means membership changes on a lattice crossing instead. Worst measured slop:
  **11.180 m against the 11.314 m half-diagonal bound.**
* **Thirty-two bits could not be re-cut** (IB-8). The instance field is the one the
  certification names, but the *meshlet* field was already the binding ceiling — P28.1's own
  measurement puts descriptors at 5.26–5.44 % of pool bytes, so 14 bits refuses past 7.4 % of
  the default streaming budget and 11 bits past **0.9 %**. Shrinking the triangle field is
  worse: it lowers `max_triangles`, which *increases* the meshlet count and spends the bits
  it borrowed. **Every re-cut of 32 bits makes the frame refuse sooner somewhere.** So the id
  widened to 64, WGSL has no `u64` so it is a `vec2<u32>` and no field may straddle the word
  boundary, and that is why the split is 7 + 25 filling word 0 exactly.
* **A refusal that stops firing is the point, not a defect** (IB-8). Two of the three
  ceilings are now unreachable by real content; the binding ceiling moves to
  `budget_bytes` and `MAX_CPU_SCATTER_INSTANCES`, which are *tunable and reported* where a
  packed field was neither. The **triangle** refusal stays reachable on purpose — it is
  meshopt's own configurable cap, and it is what keeps the fallback arm running.
* **Widening the id told two reasons apart.** P28.1 blocked the voxel shadow-receiver routing
  because "all thirty-two bits are spent", and
  `the_visbuffer_id_space_has_no_room_for_a_second_geometry_kind` called itself *"the
  falsifier: it fails the day the id space grows, which is the day the voxel door genuinely
  opens"*. The id grew, the arm failed as designed, and **its prediction was wrong**: eight
  bits are free and the routing stands, because the resolve shades by pulling vertices out of
  the shared meshlet pool and a voxel chunk has none. Exhaustion was the reason *given*; no
  meshlet structure is the reason. Renamed `the_voxel_routing_survives_the_id_widening`, and
  the retired reason is **deleted** from `voxel.wgsl` rather than joined — a stale blocker
  sends the next reader to widen an id that is already wide.
* **A mutation declares what it moved; "I do not know" is legal and costs a full walk**
  (IB-13). `touch` widens the projection scope to everything, `touch_at` names guids. The
  scope is a *conservative union*, so converting a call site is a strict improvement and
  leaving one unconverted is only slow — which is why 41 of 45 sites were not touched.
  `touch_at` may only be used for changes that cannot MOVE an entity in the hierarchy,
  because `children` and `roots` are derived from *other* entities' parent links.
* **The projection's oracle is the thing it replaced.** `diff` is kept — not as a fallback but
  as a second, independent computation — and the gate requires the scoped delta to *contain*
  what the full diff would have said and to be *equal to the current snapshot* where it
  speaks. Containment, not equality: a transform write names an entity whose `SceneNode` did
  not change, so one redundant node ships against a hundred thousand. A fast projection with
  no independent statement of what it should have said would be unfalsifiable.
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

## The I1 audit (adversarial, `2151826..c9ce76a`)

Every claim above **HELD on re-measurement** — the numbers reproduce exactly (220/220
identical, 2 of 16 tiles, 495 on the hill, 33 bit-identical border samples, 127 ground
samples, 66.8707°, delta == entity_count on all 19 levels, `Playground.inf_lvl` 8 839 →
8 839 with 40 bytes differing, `.inf_sm` +5/+4 bytes and +4 lines × 18 B, frontend 691/77,
goldens 54 and unmoved, exactly the three authorized container bumps). What the audit found
was **not wrong claims — it was arms that could not fail**, in three of the four items.

**Mutation-measured gate blindness, all four now closed:**

| mutation | before the audit | now |
|---|---|---|
| `InfSceneWorldBuilder::build` stops paging (the IB-1 fix's one production site) | **all 64 `inf-player` test binaries green** | `a_payload_that_carries_its_terrain_scatters_on_the_hill` |
| the shipped host's ground gather narrows back to the lowest `Guid` (the IB-15 defect) | **all 64 green** | `the_shipped_host_seam_answers_on_both_terrains` + the host mirror |
| the editor host stops calling `page_terrains_for_pcg` | mirror gate green (`contains("page_terrains_for_pcg(")` reads TRUE off `fn page_terrains_for_pcg(`) | occurrence **counts**, both hosts |
| `pcg_regions_of` becomes a union bounding box, **both** mirrors | footprint arm green (both volumes shared a `z` row, so the union was 4 tiles of 16) | volumes on a diagonal, `paged == 2` exact, and the union's own cost measured at 16 |
| `ScenePayload::blend_mode` is never applied by the player | **all 64 green** | `the_payloads_session_blend_mode_reaches_the_built_world` |
| the platformer template's actor sidecar GUID drifts to anything | **nothing red** — the arm asserted the sidecar contained the text `"guid = "` | the GUID's 16 bytes must be in the level payload; the committed sidecar is locked to `COYOTE_ASSET_GUID` |
| `stranded_levels`' guard reverts to `legacy == levels_root()` | its own arm green (`Path` equality normalizes `.` away, so the `content_dir = "."` fixture never discriminated) | a second fixture (`levels_dir = "Content/Levels"`) that does |
| `PoseBlender::mode_for` reads *any* authored mode | nothing red (every parity fixture has one transition) | `the_precedence_reads_the_fired_edge_and_not_its_siblings` |

After the repairs: battery **295 / 5 566 / 0 / 13** (+9 arms, no new test binaries), `fmt`
clean, `clippy --workspace --all-targets` with `-D warnings` **zero**, frontend **691 / 77**
and `tsc --noEmit` clean, rustdoc **442 of 450** (the wave had taken it to 448), goldens
**54** and byte-unchanged.

Nine independent mutations were re-run after the fixes and each dies at exactly the arm that
names it, and at no other. The implementer's own mutations were re-run too and all hold
(template scaffolding to the legacy path trips six arms; deleting the cook advisory trips
two; reverting the Ring-0 ground rule trips the 127-sample arm **and nothing else in three
crates**, which is the proof that single-terrain behaviour is unchanged by construction).

**The one behavioural consequence the wave did not enumerate.** IB-15 made the *height*
query position-aware; `inf_ecs::deform::ground_terrain` — the deformation footprint pass —
did **not** move with it, and its doc claimed the two rules could never disagree. Measured
and now asserted (`a_contact_over_a_second_terrain_leaves_no_footprint`): on a two-terrain
level a body standing on the **second** terrain stands correctly and leaves **zero**
footprints against the control's one, because the pass resolves one terrain for the whole
world and `height_at` answers `None` outside it. Carried below; closing it means resolving
per contact and keying the field by terrain entity.

---

## Done — wave I2 (the GIS door)

The certification called this *"the single largest piece of connective work the island
needs"*: `inf-gis` was 5 732 lines across nine modules with **one dependent crate using one
module** and zero non-test callers for `spawn_layer`, `read_vector`, `RoadGraph::from_layer`,
`build_all_ribbons`, `triangulate_polygon`, `classify_to_ids`. Every one of those now has a
caller on a production path.

**IB-3 · the door.** `inf_gis::import` (Ring 0) owns the whole import: `.prj` resolution,
the probe, naming, the stub floor, the cap, the stream channel, and a `SpawnPlan` as the
result. Three front ends, one decision-maker — the wizard
(`GisImportDialog` + `commands/gis.rs` + `inf_editor_core::gis::run_import`), the CLI
(`inf gis info` / `inf gis plan`), and any future cook-time pipeline. **The proof is
cross-process**: the real `inf` binary prints `SpawnPlan::digest()` and
`the_cli_and_the_library_import_the_same_fixture_identically` compares it against
`import_layer` in-process.

**IB-11's near half.** The `.prj` sidecar is read (last `AUTHORITY["EPSG",…]`, not the
first — a NAD83/UTM WKT names five codes and four describe its ellipsoid, prime meridian,
angular unit and geographic base); an ESRI `.prj` with no authority at all falls back to a
pattern match on the projection NAME and **says it guessed**. The vertical unit is applied
exactly once, in `Transform::to_projected`, **before** the projection — the anchor subtracts
a metric datum height, so scaling afterwards would scale the anchor. Web Mercator stays
refused as an anchor with its 1.53× number and stays legal as a source. LERC / BigTIFF /
LAS are untouched named CANNOTs.

**IB-4 · roads are geometry.** `build_surface` drapes a whole network, merges by class and
fans the junctions; `surface_to_mesh` writes a real `MeshAsset` with a submesh and a
material slot per road class; `gisroad::import_road_surface` supplies the ground from the
level's terrains through the IB-15 rule and spawns through
`edit_create_mesh_asset` — one transaction, one Ctrl+Z. Measured: **3 758 road vertices,
worst deviation 0.000000 m** from `ground + lift`; on a 14 m arterial at a 1 m step, mid-span
**0.000750 m** with both axes subdivided against **5.0490 m** on the centreline's own
vertices (along) and **0.0495 m** with one quad across (across); a road across two terrains
at **west 28.460 m, east 48.460 m**; and the built `.inf_mesh` is bit-identical across two
builds (4 005 vertices / 7 284 triangles, digest `65471d72982b118a`).

**IB-5 · both wires.** Land cover → biome ids through `classify_to_ids` (its first caller
outside its own tests) and a new `inf_terrain::BiomeFill` polygon fill that accumulates any
number of classes into **one** `BiomeDelta`, i.e. one undo step. Footprint attributes →
`BuildingParams::floors`, from the stated count, else DERIVED from a stated height, else a
typed default — with which of the three answered carried per feature and counted per layer.
Measured: 2 496 samples over 2 biomes with 5%/7% canopy in one class and 92% in another;
floors 6/10/2 producing 21.9 m and 5.6 m of baked geometry.

**IB-6 · oriented lots.** `inf_math::obb2` (owned monotone-chain hull + rotating calipers,
trig-free, canonical) and `inf_pcg::building::LotFrame`. Measured: a 30 × 10 lot off the
grid is **300 m² oriented against 780 axis-aligned**; the grammar fixture's 24 × 12 lot is
**288 vs 633.6**. The world proof asserts the population — every placed box square to the
LOT, **none** square to the world axes, and the same lot un-rotated building a
box-for-box identical building.

**IB-14 · the caps.** The certification's numbers reproduced and then retired: 10 000 roads
→ 4 096 (5 904 truncated) → **10 000 whole**; 50 000 footprints → 4 096 (45 904) →
**50 000 whole**.

**Both I1-routed items, closed.** The deformation footprint pass resolves per contact (and
needed no terrain-keyed field — the field is a global world-XZ lattice), and
`ScenePayload::blend_mode` has both halves: a Project Settings writer into `inf.toml` + the
live world, and a cooked-path reader through `manifest.toml` →
`build_world_from_pack`.

## The I2 audit (adversarial, `6990247..d186525`)

**Every measured claim above HELD on re-measurement** — the numbers reproduce from the tests
that print them (3 758 vertices at 0.000000 m; west 28.460 / east 48.460; 581 of 15 178 open
samples paved; 2 496 biome samples at {1: 1456, 2: 1040} with unpainted ground still 0 and
one undo restoring it; floors 6/10/2 → 21.9 m and 5.6 m; 300 vs 780 and 288 vs 633.6;
10 000 and 50 000 whole; goldens 54 and byte-unchanged; schemas unmoved). Both routed items
are genuinely closed — each fix dies alone under the mutation that names it.

**Two reported numbers did not reproduce and are corrected**: the ROADMAP block's commit
range/count, and the road's mid-span pair (see above — "0.0004 m against 3.2 m" was neither).
Both were numbers no test printed. *That is the rule this file already states, met the hard
way: **the numbers live in the tests that print them**, and a number that only lives in a
ledger drifts.*

**Mutation-measured gate blindness, four found and closed:**

| mutation | before the audit | now |
|---|---|---|
| the vertical unit is applied AFTER the anchor's metric datum height instead of before | **all 78 `inf-gis` tests green** — the arm anchors at `origin_height_m = 0`, where the orderings are identical | `the_vertical_unit_is_applied_before_the_anchors_metric_datum_height` (50 m datum: −19.52 m vs the alternative's 15.24 m) |
| `obb2::canonical_dir` becomes the identity | **`inf-math` and `inf-pcg` entire, green** | `a_shape_and_its_half_turn_get_the_same_basis` |
| `SpawnPlan::digest` drops two of its four counts | **all `inf-gis` tests green** | `the_digest_separates_plans_that_differ_by_one_bit` (one ULP, every count, kind, feature, order) |
| `BiomeFill::ring_contains` stops being half-open | `BiomeFill` had **no test in its own crate**; its one caller paints three *disjoint* squares | `adjacent_polygons_tile_a_terrain_…` + `a_fill_honours_holes_…` |

Two more were sound but silent: the **transverse** alternative was not priced though the
longitudinal one was (the wave's own headline law), and the road mesh had a determinism gate
at the *graph* and none at the *asset* it writes. Both are measured and printed now.

Ten new mutations and eight of the implementer's were re-run; each dies at exactly the arm
that names it and at no other, including both blend-mode halves separately (1 of 15 in
`pose_parity` each), the re-aimed portability ban **and its anti-vacuity control**, and
`inf-packager`'s manifest ban (untouched, still biting).

## Done — wave I3 (city scale)

Three certified ceilings killed, and the numbers that killed them all live in tests that
print them.

**The fixture first, because every other number is about it.**
`samples/phase30-city` (262 KB, committed as a *generator*): 100 `PcgVolume` blocks sharing
one `.inf_pcg`, a Driver carrying `StreamingSource`, and a street grid through I2's own
import door — **220 segments, 81 four-way junctions, 4 061 road vertices**. The graph is
`grammar.footprint → building.lots → building.plan`, so **100 blocks × 10 lots = 1 000
buildings / 370 468 solids**.

**IB-2a · the collider band.** The certification: 12 850 colliders at 0.363 µs each is
4.663 ms/step, the town is *seven* buildings, and `STREAMED_STEP_BUDGET_MS` = 4.0 ms buys
~11 000 colliders. The shipped city unbanded is **134.480 ms/step**. Banded it is
**6 067 colliders — 1.64 % — 2.202 ms** at the certification's own rate, and **1.991 ms**
measured here. The radius is a measurement, not a preference:

| near_m | colliders | cert ms | measured ms |
|---|---|---|---|
| 16 | 1 000 | 0.363 | 0.1899 |
| 32 | 1 563 | 0.567 | 0.3821 |
| 48 | 4 241 | 1.539 | 1.4143 |
| **64** | **7 513** | **2.727** | **2.8703** |
| 96 | 14 666 | 5.324 | 6.8114 |
| 128 | 21 544 | 7.820 | 11.0419 |

64 m is the widest radius inside the budget and the arm fails if the shipped default stops
being one. (The sweep is over the programmatic 427 351-solid city in `inf-physics`; the
shipped one is 370 468 solids, hence the two slightly different counts at 64 m.)

**IB-2b · the draw LOD.** Three complementary batches per volume — ungrouped content keeps
`[0, draw)`, the parts take `[0, lod)`, the shells take `[lod, draw)` — where `lod` is
`STRUCTURE_LOD_M = 192`, deliberately **three times** the 64 m collider band so every
building a body can collide with is drawn as its parts. On the shipped city: **14 whole,
788 shells, 198 out** in physics, and through the real `project_scene`, **100 parts batches
(370 468 instances) bounded above at 192 m against 100 shell batches (1 000 instances)
bounded below at it** — a **370×** far-field reduction, with zero ungrouped instances. A
probe dropped on a far building's shell rests at 7.599 m on a 7.200 m box, because a shell
that is not a barrier is a hole rather than a LOD.

**IB-2c · lot subdivision.** 100 × 60 block → 8 lots totalling **6000.0 m² of 6000.0**; a
rotated 120 × 70 block at maximum jitter → 12 lots, worst pairwise overlap **1.4e-12 m²**; a
triangular block keeps **3 of 12** and says so, where a centre-only containment test would
have admitted all 12. On the shipped city: **4 500 lot pairs, worst overlap 0.000e0 m².**

**IB-8 · the instance ceiling.** 2 047 → **16 777 214** (8 196×; the brief asked for 16 384).

**IB-13 · the scene projection.** 100 000 entities, moving one: **52.857 ms → 8.105 ms** for
a drag frame and **0.0006 ms** for a select frame.

### The commits

`cdeb888` lot subdivision + structure groups + the composition door · `29ac631` the sim-side
collider band · `fa4963b` the structure draw LOD + the inner band · `d170651` the scoped
scene projection · `d83b762` the 64-bit visbuffer id · `666d63d` the city fixture + gate.

### Counts

| | after I2's audit | after I3 |
|---|---|---|
| battery blocks / passed / failed / ignored | 296 / 5 618 / 0 / 13 | **298 / 5 663 / 0 / 13** |
| frontend tests / files | 702 / 78 | **702 / 78**, `tsc` and `eslint` clean |
| goldens | 54, byte-frozen | **54, byte-identical under `INF_GOLDEN_STRICT=1`** |
| rustdoc warnings (ceiling 450) | 443 | **445**, and the warning **list** is byte-identical to `866fb55`'s — this wave added zero. (`866fb55` measures 445 today too; the ledger's 443 does not reproduce, which is the I2 law arriving a third time.) |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | **unchanged — no schema moved** |

The battery was measured at `14a5fac` (5 662) plus the one arm that landed after it —
`the_shipped_projection_emits_complementary_parts_and_shell_batches`, green and printed in
the gate's own output. Every other later commit is documentation or `.gitattributes`.

---

## Open questions / carried bounds

**Closed by I2 (were I1's carried bounds):**
* ~~The deformation footprint pass is single-terrain~~ — per-contact resolve, same
  topmost-that-answers rule, same `Guid` tie-break. The predicted "terrain-keyed field" was
  **not needed**: `DeformField` is a global world-XZ lattice.
* ~~`ScenePayload::blend_mode` has a reader and no writer~~ — Project Settings ▸ Animation
  writes `inf.toml` and the live world; the cook copies the name into `manifest.toml` and
  `build_world_from_pack` applies it.

**Measured and bounded (I3 owns these numbers):**
* **A drag frame still costs 8.105 ms at 100 000 entities, and it is not the projection.**
  The select-only column isolates the projection at **0.0006 ms**; the rest is
  `EcsWorld::propagate` — a full DFS over the world with an archetype-touching bundle insert
  per entity, dirty-gated but not *incremental*. Making transform propagation incremental is
  its own item and this wave did not take it.
* **The parts↔shell swap is a hard cut, not a cross-fade.** The scatter path's existing
  `fade_band_m` resolves mesh↔impostor with a complementary dither; the structure LOD's two
  batches meet at `STRUCTURE_LOD_M` with no overlap, because an overlap draws both (a solid
  box inside a building) and the dither is per-pipeline rather than per-band. At 192 m the
  pop is small; closing it needs the fade to become a property of the *band pair*.
* **A far building's shell is a BOX, not its baked mesh.** IB-2b's brief names the P23
  grammar→mesh bake feeding vgeom as the far tier's geometry. What ships is the derived
  oriented shell — correct in silhouette and proportion (the non-uniform instance scale
  exists for exactly that), and consistent with the near tier, which draws placeholder cubes
  too (`kind_index → real mesh` is the standing P19 gap). Wiring `bake_building_in` into a
  runtime vgeom asset per archetype is a project; `gisbuild` already bakes footprints to real
  meshes on the authored path and is where it would start.
* **The city fixture's ground is FLAT.** Deliberate — see the sample's README. It means the
  fixture says nothing about a banded city over streamed terrain, which is I7's.
* **A `PcgVolume` is banded as a whole in the RENDER path.** The collider band is per
  building (per `StructureGroup`); the draw band is per instance on the GPU, but the three
  batches are built per *volume*, so a volume spanning a whole district would put its parts
  batch in one content key. City blocks are 100 × 60 m and this is not reached.
* **The visbuffer costs 8 bytes a pixel now** — 3.7 MB at 1280 × 720, 33 MB at 4K. Paid only
  while `VgeomSettings::visbuffer` is on, which is `false` on every tier.

**Measured and bounded (I2 owns these numbers):**
* **A land-cover tiling drawn FLUSH with the terrain leaves its far row and column
  unpainted** — **15 of 64** samples on an 8 × 8 tile (found by the I2 audit). It is the
  half-open crossing rule behaving correctly: a sample lying exactly on a polygon's
  *outermost* edge has no neighbour on that side to claim it, and the same property is what
  makes an *interior* shared edge belong to exactly one of two neighbours (64 of 64, exactly
  once). Trading it for an inclusive test would double-paint every interior seam instead.
  The remedy is a source ring that runs past the ground, which a clipped published layer
  normally does. Named on `BiomeFill::add_polygon`.
* **A junction fan paves the CORE, not the kerbs.** `fan_at` covers the convex hull of the
  legs' end cross-sections — 581 open samples (5.8 m²) on the acute-fork fixture — and the
  wedges *outside* that hull, between adjacent kerbs, stay open. Closing them needs
  kerb-radius fillets per leg pair, which is a road-modelling project rather than an import.
  Also measured, and worth knowing before building a gate on it: **a symmetric T or `+`
  junction needs no fan at all**, because two opposed legs tile the crossing between them.
* **A road has no collider of its own, by ruling.** It conforms to the terrain, whose
  heightfield collider already answers there, so drawn and collided are two readings of one
  array separated by exactly the 2 cm lift (3 758 vertices, 0.000000 m). Duplicating it as
  per-segment trimeshes is 0.363 µs/step each (IB-2) = **3.63 ms/step** for 10 000 roads,
  for nothing a body can reach. The case that genuinely needs one is a road that **leaves**
  the ground — a bridge — which needs a bridge/tunnel attribute published layers rarely
  carry.
* **A cross-section subdivides at most `MAX_CROSS_STRIPS = 32` times.** A carriageway wider
  than `32 × ground_step` conforms more coarsely across than along. Past every real road at
  every sane step, and reported through the triangle count rather than hidden.
* **A road mesh is f32 and therefore quantised.** Positions are local to the mesh's own
  centre, so the number is an *extent* rather than a coordinate (~3 mm at island scale), and
  `MeshBuildReport::quantisation_m` states it. Past 1 cm the import raises an advisory.
* **Footprint geometry is capped at 512 buildings; the attribute pass is not.** Reading
  50 000 rows is free and the coverage is what an author needs; 50 000 `.inf_mesh` files is
  not. Reported with the number to raise.
* **A GIS import drapes on the heightfield only.** `gis::with_ground` passes no voxel
  volumes, so a road over a carved cave mouth takes the published centreline's elevation.
* **`inf gis` cannot write a level.** The `.inf_lvl` writer is `SceneDoc`, in Ring 1, and
  putting it in the CLI would link wgpu — the same reason `inf-project` exists. So the CLI
  produces plans, reports and derived assets; placing entities is the editor's half. The
  digest is what makes the two provably the same import anyway.
* **`RoadGraph` derives junctions from segment ENDPOINTS.** A street digitised as one
  feature passing *through* a crossing creates no node there. Correct for published layers,
  which are split at intersections — and the reason the fan fixture splits its through-road.
* **The GIS wizard has no attribute-mapping UI.** The field spellings each generator reads
  are constants (`ROAD_CLASS_FIELDS`, `FLOOR_FIELDS`, `CLASS_FIELDS`, …), and a layer that
  spells a column differently is defaulted-and-counted rather than remapped. The counts are
  advisories; the remap is not built.
* **The biome terrain is entered as an entity id.** The wizard has a text field, not a
  picker — a level with several terrains needs the author to paste a GUID.
* **The rustdoc ceiling has 8 of 450 left** — measured at **442** after this audit; the wave
  took it to 448 with six new links (two private ladder aliases, three unqualified
  `SmBlendMode`/`PoseBlender` paths, one private `stranded_levels`), all now item-scoped or
  unlinked. `cargo doc --no-deps --workspace` is the cheapest CI leg to turn red; run it
  before adding an intra-doc link, and prefer `[X](crate::X)` over a bare `[X]`.
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

**Closed by I3 (were pending):**
* ~~**IB-2**~~ — the band, the tiers and the subdivision node; a 1 000-building city is
  6 067 colliders / 2.202 ms against a 4.0 ms budget, and one `building.plan` node is now as
  many buildings as its block has lots.
* ~~**IB-8**~~ — the id is 64 bits; `VIS_MAX_INSTANCES` is 16 777 214. The 32-bit re-cut the
  certification proposed was measured and refused (see the decisions above).
* ~~**IB-13**~~ — `project_delta`; 52.857 ms → 8.105 ms on a drag frame at 100 000 entities
  and 0.0006 ms on a select frame. The residue is `EcsWorld::propagate`, named above.

**Pending their waves (from the certification):**
* **IB-9** — `TERRAIN_RESIDENT_BYTES_CEILING = 16 MiB` against `max_resident_tiles = 1024`
  ≈ 264 MiB. **16.4× apart**, and no scene has made them meet.
* **IB-12** — `axis_independent_lag` unrotates absolute world positions, not the delta; not
  floating-origin-safe at partition scale.
* **IB-16** — no per-frame time budget or upload throttle in the VT loop; only a byte
  residency ceiling. Coupled to T51 (no request→residency window).
* **No fps instrument at shipping resolution exists.** The only GPU frame harness renders
  640 × 360, and every wall-clock assertion is disabled on software/paravirtual adapters.
  Nothing may claim "≥60 fps" until I4 builds the instrument.
