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
| **I3** | city scale — IB-2a/b/c, IB-8, IB-13, the city fixture | **DONE + AUDITED** — see below |
| **I4** | the fps instrument + budgets — IB-9, IB-16, IB-12, the shipping-resolution harness | **DONE + AUDITED** — see below |
| **IP** | **performance** — the cook's PCG evaluation, the sim fixed step, the lighting stack, IB-9's island ceiling | **NEXT** — the wave the instrument makes possible; routed by the I4 audit, see *What IP inherits* below |
| I5 | source data — IB-11 (DEM ingest reality, CRS, LiDAR) | not started |
| ~~I6~~ | ~~scale seams — IB-12~~ *(pulled into I4; IB-8 and IB-13 into I3)* | **absorbed** |
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
* **A sha is only true of the tree it was written in, so the AUDIT states the range**
  (I3 audit). I3 corrected its own commit range twice, correctly, and a rebase onto the I2
  hotfix invalidated every sha in it anyway — the fourth failure of the same field in three
  waves. A wave cannot fix this by being more careful; the wave that *closes* a wave can, by
  re-stating the range from the tree it certifies. The same goes for a battery count: run it
  at the head you are about to write down, not at the commit before the last two.
* **A metric that counts its own summary lines is not a count** (I3 audit). The rustdoc
  figure both previous ledgers carried is `grep -c '^warning'`, which includes one
  "generated N warnings" line per **re-documented** crate — so it moves with the build cache
  and measured 445 then 446 on two consecutive runs of one unchanged tree. The number that
  means something is the individual-warning count from a cold re-documentation: **412**.
* **A mutation that is its own inverse measures nothing** (I3 audit). Swapping
  `VisPacking::words` and `from_words` together leaves every consumer unchanged, because they
  compose the pair. When a mutation passes, ask whether it *could* have failed before
  recording it as evidence.
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
  **11.314 m, which is the 11.314 m half-diagonal bound exactly** — the sweep has a sample
  on a cell corner, so the bound is *reached*, not merely respected. (The wave's ledger said
  11.180 m; the arm that prints it says 11.314. Corrected by the I3 audit.) The lattice's
  own **edge** is the second cost and it is carried below: a source parked on a lattice line
  re-bands every step, and hysteresis is refused because it would make the band a function of
  history rather than of sim state.
* **A per-instance cut cannot be both gap-free and overlap-free, and gap-free wins**
  (IB-2b, I3 audit). The parts/shell bands are complementary in the *group's* distance while
  the GPU cull compares the eye against each *instance*, so the two cuts must differ by the
  group's own reach — one way round a building loses its far parts with no shell behind them
  (a hole), the other way it draws its parts inside its own shell (which contains them).
  `P >= S + reach` and `P <= S - reach` cannot both hold. Any future band pair over
  per-instance culling inherits this, and the number to carry is the widest shell's
  half-diagonal in the batch.
* **The band's quantization is stateless, and its edge is a bound rather than a bug**
  (IB-2a, I3 audit). Hysteresis on the lattice snap is **refused**: it would make the active
  set a function of the *history* of positions, and "a pure function of sim state" is the
  entire licence under which a collider band is allowed to exist at all. A source parked on a
  lattice line re-bands every step, between exactly two sets, for as long as it parks there.
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
  **What a scoped projection may not do is read the hierarchy from somewhere else** (I3
  audit): it asked the world for a parent's children, which is bevy's insertion order, while
  the full projection builds them from creation `order`. The two agree until something
  re-parents. One walk defines creation rank, one cache carries it, and both projections
  answer from it.
  `touch_at` may only be used for changes that cannot MOVE an entity in the hierarchy,
  because `children` and `roots` are derived from *other* entities' parent links.
* **The projection's oracle is the thing it replaced.** `diff` is kept — not as a fallback but
  as a second, independent computation — and the gate requires the scoped delta to *contain*
  what the full diff would have said and to be *equal to the current snapshot* where it
  speaks. Containment, not equality: a transform write names an entity whose `SceneNode` did
  not change, so one redundant node ships against a hundred thousand. A fast projection with
  no independent statement of what it should have said would be unfalsifiable.
* **A budget is a TARGET or a TRIPWIRE and cannot be both** (I4). The instrument's
  `SHIPPING_FRAME_BUDGET_MS = 16.6` is what "≥ 60 fps" *means* and is never asserted,
  because the engine does not meet it and a constant asserted where it fails is a red build
  somebody raises. `SHIPPING_FRAME_CEILING_MS = 58.0` is the ratcheting tripwire beside it,
  and the instrument prints the **distance** between them every run. The day they meet, the
  target becomes the assertion and the ceiling is deleted.
* **The build is not the build, so the instrument reports rather than asserts in `dev`**
  (I4). `[profile.dev]` is `opt-level = 1` with debug assertions for every workspace crate,
  so the CPU half of a frame measured there is a fact about a build nobody ships. That is
  the paravirtual-adapter law one layer down, and it is why the full battery running the
  instrument does not assert it: `cargo test --release` is the run that does. Measured: the
  same frame is **39.8 ms** release and **46.4 ms** dev at 1080p.
* **A MIN over samples of different things is a selection, not a minimum** (I4). The
  instrument's first draft let its camera step run on across rounds; every round replays the
  identical sequence now, which is the right discipline and is kept. *(The **28.3 ms** the
  wave attributed to its absence does not reproduce — the I4 audit re-ran the mutation and
  the p50 moved by about 1 %, to 39.104 ms, because `CITY_DRIVE_STEP_M` is 0.25 m and a
  120-frame round is **thirty metres**. The law stands; the figure is retired.)*
* **A wall clock with no MIN-of-rounds has no sign** (I4 audit). Clause 5's price took one
  30-frame mean per configuration and answered **−2.03, −0.28, +2.00 and −0.86 ms** on four
  runs of byte-identical code. The same three configurations under MIN-of-5-rounds reproduce
  to three decimals. A single mean of an identical scene lands anywhere between **0.54 and
  5.76 ms** on this card, and any comparison built on one is a coin toss with a decimal point.
* **A frame-scale tolerance cannot arm an object-scale claim** (I4 audit). `image_diff`
  averages over a 64 × 36 downscale, so a change confined to a 2 903-pixel building can move
  the frame mean by at most **0.0014 — 43× under `GOLDEN_MEAN_TOLERANCE` at its arithmetic
  maximum**. Clause 6's refusal was armed by a clause that could not fail for any LOD swap
  whatever. Assert a measurement in the units of the thing it is about.
* **A threshold has to sit below the defect it names** (I4 audit). The handoff arm allowed
  3× a steady step and its own comment priced a cut at "0.667 m against a few centimetres" —
  but the steady step is 0.377 m, so a full cut is **1.77×** and passed. Compute the defect,
  then put the threshold under it.
* **A default that is OFF is part of the measurement** (I4 audit). The instrument's frame
  draws no shadows, GI, VSM, TAA, SSAO, bloom or visbuffer, because those are the shipped
  defaults for a level that authors no render block. The number is true; it is a number about
  a configuration, and a constitution has to name the configuration. Measured: the same
  content lit is **p95 92.3-92.9 ms against 43.7-44.0**, GPU frame **35.8-36.1 against 17.3-19.4**.
* **The render cut is O(LEVELS), not O(pages)** (IB-9). A quadtree cut's refine radius
  doubles exactly when its node size doubles, so a level contributes the same ring however
  large the world is, and the pyramid terminates at `min_tiles`. Measured across four world
  sizes whose catalog quadruples each step: **64 → 157 → 250 → 340** pages, a constant ring
  of **93, 93, 90**. That is what makes a residency budget derivable at all — and the first
  draft of the arm asserted the peak was *identical* across world sizes, which is false.
* **The FLOOR is never throttled** (IB-16). The per-frame upload budget bounds *refinement*,
  which is where a burst lives; the analytic floor is the mandatory residency class, bounded
  per visible surface, and a P28.2 cluster's tiles ride its lane. Measured: throttling it
  retracted a cluster page and left the P28.3 load at 3 of 5 pages resident. VSM is exempt
  whole — a missing shadow page has no coarser ancestor to fall back to, so it is a wrong
  answer rather than a blurry one.
* **A throttle takes the TAIL of a lane; it does not delay the HEAD of one** (IB-16).
  `p28-5-lead-time-ruling.md` §3.5 predicted a per-frame admission throttle would reverse
  `DEFAULT_PREDICT_HORIZON_TICKS = 0`. It landed and it did not: at the shipped budget h=0
  still wins (19 542 blur against 19 766), because a page is still sampleable the frame it
  is admitted. Under a budget nothing ships — two pages a frame — **a lead does win**
  (75 928 against 76 074), which is reported and never asserted. The memo named the right
  mechanism and the wrong threshold.
* **The camera lag unrotates the DELTA** (IB-12). Algebraically the same; in floating point
  **1.848e6× better** at half a million metres. The pre-IB-12 spelling is kept as
  `axis_independent_lag_absolute` and pinned at **zero** production call sites, because a fix
  whose predecessor has been deleted cannot be shown to have been necessary.
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
`samples/phase30-city` (**244 974 bytes** across seven files, committed as a *generator*; the wave said 262 KB, which is what the seven files ALLOCATE at a 4 KiB cluster, not what they weigh): 100 `PcgVolume` blocks sharing
one `.inf_pcg`, a Driver carrying `StreamingSource`, and a street grid through I2's own
import door — **220 segments, 81 four-way junctions, 4 061 road vertices**. The graph is
`grammar.footprint → building.lots → building.plan`, so **100 blocks × 10 lots = 1 000
buildings / 370 468 solids**.

**IB-2a · the collider band.** The certification: 12 850 colliders at 0.363 µs each is
4.663 ms/step, the town is *seven* buildings, and `STREAMED_STEP_BUDGET_MS` = 4.0 ms buys
~11 000 colliders. The shipped city unbanded is **134.480 ms/step**. Banded it is
**6 067 colliders — 1.64 % — 2.202 ms** at the certification's own rate. The wall clock is
printed and never asserted, and it is a range rather than a figure: **1.991 ms** when the wave
ran it, **2.13 and 2.06 ms** on the audit's two runs of the same binary. The collider count
is what a machine cannot move.

The radius is a measurement, not a preference:

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

**IB-2b · the draw LOD.** Three batches per volume — ungrouped content keeps `[0, draw)`,
the parts take `[0, lod + reach)`, the shells take `[lod, draw)` — where `lod` is
`STRUCTURE_LOD_M = 192`, deliberately **three times** the 64 m collider band so every
building a body can collide with is drawn as its parts. On the shipped city: **14 whole,
788 shells, 198 out** in physics, and through the real `project_scene`, **100 parts batches
(370 468 instances) bounded above at 192 m + a reach of at most 18.486 m against 100 shell
batches (1 000 instances) bounded below at 192 m** — a **370×** far-field reduction, with
zero ungrouped instances. A probe (a 0.4 m sphere) dropped on a far building's shell rests
at 7.599 m on a 7.200 m box — its own radius above the surface — because a shell that is not
a barrier is a hole rather than a LOD.

`reach` is the I3 audit's correction and the two bands **overlap** by it rather than meeting.
The bands are complementary in the *group's* distance and the cull is per *instance*, so with
both cuts at 192 m a building straddling the line lost the parts outside it and grew no shell
to stand in for them: **196 part-drawn-with-no-shell buildings over 20 eye positions, worst 15
at once**, against **0** once the parts band carries the reach. Gap-freedom and
overlap-freedom cannot both hold for a per-instance cut; the choice is made in favour of the
one that never shows through a building.

**IB-2c · lot subdivision.** 100 × 60 block → 8 lots totalling **6000.0 m² of 6000.0**; a
rotated 120 × 70 block at maximum jitter → 12 lots, worst pairwise overlap **1.4e-12 m²**; a
triangular block keeps **3 of 12** and says so, where a centre-only containment test would
have admitted all 12. On the shipped city: **4 500 lot pairs, worst overlap 0.000e0 m².**

**IB-8 · the instance ceiling.** `VIS_MAX_INSTANCES` 2 047 → **16 777 215** (indices
`0..=16 777 214`; 8 196×, and the brief asked for 16 384). The ledger first said
16 777 214 — the largest index, not the constant — where the arm prints the constant.

**IB-13 · the scene projection.** 100 000 entities, moving one: **52.857 ms → 8.105 ms** for
a drag frame and **0.0006 ms** for a select frame.

### The commits

The wave was **rebased onto the I2 hotfix after it was written**, so every sha its own ledger
named is gone. The range as it stands in this tree: **`0856405..c6ec96e`, thirteen commits**
(the range excludes its base and includes its head), of which six carry the work —
`1ebfbf7` lot subdivision + structure groups + the composition door · `52dcc69` the sim-side
collider band · `7723fb0` the structure draw LOD + the inner band · `37cea4b` the scoped
scene projection · `c48e148` the 64-bit visbuffer id · `98fd35b` the city fixture + gate —
and seven close the ledger (`cdeabab`, `6d593ca`, `09e1a19`, `57d988a`, `b627487`, `e10016e`,
`c6ec96e`).

*A rebase is the fourth way a ledger's own range goes stale*, after a miscount, a wrong end
and two blocks disagreeing: the wave corrected its range twice, correctly, and then the
history moved underneath it. The lesson is not "state it more carefully" — it is that a sha
in a ledger is only true of the tree it was written in, so **the audit that closes a wave
re-states the range from the tree it certifies**.

The audit's own commits follow `c6ec96e` and are listed by subject in the audit section
below, for the same reason.

### Counts

| | after I2's audit | after I3 | after the I3 audit |
|---|---|---|---|
| battery blocks / passed / failed / ignored | 296 / 5 618 / 0 / 13 | 298 / 5 663 / 0 / 13 | **298 / 5 670 / 0 / 13** — the audit adds **exactly five** arms and no test binary, so the wave's head was 5 665 and its ledger says 5 663. One of the two is `6d593ca`, which landed *after* the ledger commit and added an arm; the other is unaccounted. A battery count is a thing to run at the head you are about to write down. |
| frontend tests / files | 702 / 78 | 702 / 78 | **702 / 78**, `tsc` and `eslint` clean |
| goldens | 54, byte-frozen | 54, byte-identical under `INF_GOLDEN_STRICT=1` | **54, byte-identical under `INF_GOLDEN_STRICT=1`**, re-run after the LOD band moved |
| `clippy --workspace --all-targets` `-D warnings` | 0 | 0 | **0** |
| rustdoc warnings (ceiling 450) | 443 | 445 | **the metric was the problem, not either number.** `grep -c '^warning'` counts one *summary* line per **re-documented** crate on top of the warnings themselves, so the total moves with the cache — 445 then 446 on two consecutive runs of one unchanged tree. Re-documented from cold (all 45 crate roots touched): **446 lines − 34 summaries = 412 individual warnings**, over the 34 crates that emit any. 412 is the figure to compare from here; 443 and 445 were never *wrong* so much as ill-defined. |
| `cargo deny` | — | bans / licenses / sources **ok** both times. The wave recorded the advisories leg **FAILING** on a yanked `arrayref 0.3.9` (`winit → sctk-adwaita → tiny-skia`); on the audit's re-run it is **ok**. Nothing in the tree moved between them — the yank check reads the *registry index*, which moves on its own, so that leg is a statement about the day it ran. The pin itself is the I2 hotfix's deliberate one (forward is the dangerous way) and is untouched. |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | **unchanged — no schema moved** |

## The I3 audit (adversarial, `c6ec96e..` this tree)

**Every headline measurement HELD on re-measurement**, from the tests that print them: the
shipped city at 370 468 solids → 6 067 banded colliders (1.64 %) / 2.202 ms at the
certification's rate against 134.480 ms; the programmatic city at 427 351 → 7 513 (1.76 %) /
2.727 ms against 155.128 ms; the whole radius sweep collider-for-collider (1 000 / 1 563 /
4 241 / 7 513 / 14 666 / 21 544); 14 whole / 788 shells / 198 out; 100 + 100 batches at
370 468 + 1 000 instances with 0 ungrouped; 4 500 lot pairs at 0.000e0 m²; 1.4e-12 m² on the
rotated block; 480 steps / 9 distinct active sets / 6 054..=7 538; the probe at 7.599 m on
7.200 m; 12 `visbuffer_parity` + 7 `visbuffer_feedback` + 54 goldens byte-identical under
`INF_GOLDEN_STRICT=1`, **on a real device** (the parity run reads ids back off the GPU).
Schemas and goldens unmoved; the committed sample re-generates byte-for-byte.

**Two defects, both in the wave's own headline claims, both fixed:**

| finding | what shipped | now |
|---|---|---|
| **IB-2b's bands are complementary in the GROUP's distance and the cull is per INSTANCE** | a building straddling 192 m lost its far parts and grew no shell — **196** part-drawn-with-no-shell buildings over 20 eyes, worst 15 at once | the parts band carries the batch's reach; **0**, with the alternative priced in the same run |
| **IB-13's scoped projection read the hierarchy from the WORLD** (bevy's insertion order) while the full one reads creation `order` | a rename after a reparent shipped a re-ordered child list; the Outliner's tree reorders and the next full projection puts it back | one walk defines creation rank, one cache carries it, both projections answer from it |

**Five arms that could not fail, mutation-measured and closed:**

| mutation | before the audit | now |
|---|---|---|
| `DEFAULT_COLLIDER_NEAR_M` narrows to 32 m | the sweep green — it asserted only that the default is *inside* the budget, not that it is the *widest* inside it, which is what its own doc claims | the sweep refuses any row wider than the default and inside the ceiling |
| `vis_feedback.wgsl` masks the meshlet field with the old literal `14u` | **all 7 feedback arms and all 12 parity arms green** — the three-readers pin compared `const` *declarations*, and no fixture resides past the old field | the use sites are pinned as counts, per shader and per constant |
| the band also anchors on a `Camera` | **all 9 city arms green** — the fixture has no camera at all, so the hazard IB-2a's whole argument is about could not be expressed | both hosts get cameras a city apart before the trace; the mutation now kills `pie_equals_shipping_…` and nothing else |
| `project_delta`'s scoped arm stops reporting a removal | all 238 scene tests green — every delete path calls `touch()`, so the branch is unreachable through the document's API | reached directly (despawn behind the document's back, then name the guid), so the branch is covered rather than merely present |
| the band's **loose** branch admits at any tier (the one that decides whether a fence past the radius is walk-through) | all 9 city arms and all 5 physics arms green — neither fixture has a single solid outside a group | two loose posts on a block's tail, one inside the band and one outside, with the tier check first |

**Five ledger numbers that did not reproduce, corrected in place:** the lattice slop
(**11.314 m**, which is the bound exactly, not 11.180); `VIS_MAX_INSTANCES` (**16 777 215**;
16 777 214 is the largest *index*); the visbuffer's cost (**8 B/px = 7.4 MB at 720p**; the
3.7 MB was the *delta*); the sample's size (**244 974 bytes**; 262 KB is what seven files
*allocate* at a 4 KiB cluster); and the commit range, which a rebase invalidated after the
wave had already corrected it twice. Plus three source comments that outlived their own
change (the 48 B scatter record, the `R32Uint` resolve header, and an 8 192× that is 8 196×),
and one doc claim with a missing exception (`structure_admitted` is bounded by the active set
*except* when the band fails open, where the active set is the world).

**Twenty-five mutations were run** — six of the implementer's and nineteen new — and after the
repairs each dies at the arm that names it. Two are recorded as *coverage bounds* rather than
defects and are carried above: `compose_volume`'s instance rebasing (the city has no scatter
prefix, so `shift` is 0 at world scale) and `phase19_gate`'s collider count (the solid count,
not the bridge's). One more was discarded on inspection and is worth writing down:
**a mutation that is its own inverse measures nothing** — swapping `VisPacking::words` and
`from_words` together cancels, because every consumer composes the pair; the real mutation is
to swap the order the *raster* writes the two words, and that kills ten of the twelve parity
arms.

**The audit's commits**, in order: the scoped projection's child order · the LOD bands' reach
· the sweep's own claim + the lattice edge · three visbuffer numbers · the gap sweep reads
the shipped bands · the scoped removal branch's arm · the three-readers use-site pin · the
camera the band had to be tested against · fmt · the ungrouped band + a memory claim's
exception · the lot digest · three sentences IB-13 made false · this ledger. Nine gates were
added or sharpened; `docs/ROADMAP.md`'s audit block tables them.

## Done — wave I4 (the instrument & the budgets)

**THE FIRST HONEST FRAME NUMBERS.** Before this wave the tree had **no GPU timing at all**
— zero `QuerySet`s, sixty literal `timestamp_writes: None` — and its only frame harness
rendered 484 lit cubes at 640 × 360. `inf_render::timing::FrameTimer` writes timestamps
between encoder commands, which needs **one seam** (`RenderGraph::run`, beside the tracing
span that has bracketed every node since Phase 2) and also sees the four segments recorded
*outside* the graph: the VT sync point, the VSM caster raster, the feedback ring copy, the
VSM marking. Off by default, and `timing_changes_no_pixel` is what the 54 frozen goldens
rest on — the same scene through two renderers differing only in the flag, **0 of 230 400
bytes moved**.

**The scene** (`samples::island_frame_scene`, composed, never committed): the phase-30 city
+ a streamed terrain paging beneath it + the phase-29 wizard character, skinned, animating.
The terrain is held flat at **exactly zero** so the composed level is *the same city wave I3
measured* — asserted at **370 468 solids**, not assumed. The settings come from
`inf_player::render::shipped_settings`, extracted out of `PlayerRenderHost::new` so there is
one answer to "what does the player render with".

**AND THE FRAME DRAWS NO SHADOWS** (the I4 audit). `RenderSettingsRecord::default()` ships
shadows / GI / TAA / SSAO / bloom off, `VsmSettings::default().enabled` is `false`
engine-wide, and the visbuffer is off on every tier — so every number below is an honest
measurement of **what a shipped player draws for a level that authors no render block**, and
is not a lit AAA frame. The harness now says so in its own output and measures the
difference through the same `shipped_settings` door: at 1080p, **p95 92.3-92.9 ms (13.4 fps)
lit against 43.7-44.0 as shipped, GPU frame 35.8-36.1 against 17.3-19.4** — the stack roughly doubles
the frame, and none of it is in `SHIPPING_FRAME_CEILING_MS`.

**RTX 4070 Ti, release, tier High, 3 rounds × 120 frames after a discarded pass of 120, MIN
of rounds, every round replaying the same camera sequence:**

| | p50 | p95 | p99 | fps at p50 |
|---|---|---|---|---|
| **1080p** | **39.792 ms** | 45.057 | 46.709 | **25.1** |
| **1440p** | **47.424 ms** | 51.136 | 52.322 | **21.1** |

**and the frame is CPU-BOUND**, which no previous harness could have said:

| | 1080p | 1440p |
|---|---|---|
| sim fixed step | **13.659 ms** | **15.631 ms** |
| stream sync | 0.003 | 0.003 |
| projection | 3.913 | 4.189 |
| render (record) | 3.399 | 3.607 |
| poll (GPU wait) | 16.865 | 23.733 |
| **CPU frame** | **37.839** | **47.164** |
| **GPU frame** | **15.875** | **22.716** |

GPU, dearest first at 1080p: **scatter 10.758 ms (67.8 %)**, terrain 1.402, vgeom 1.125,
resolve 1.117, sky 1.075, everything else under 0.13. At 1440p scatter is 13.034 (57.4 %),
terrain 2.605, resolve 2.400, sky 1.870, vgeom 1.849.

**Distance from 60 fps: p50 +23.2 / +30.8 ms, p95 +28.5 / +34.5 ms.** The harness frame is
*serialized* (it polls every frame, because a frame time with no sync point is a submission
time) while a presenter overlaps the halves, so a **pipelined estimate** is printed beside
it and never asserted: **20.975 ms (47.7 fps)** at 1080p, **23.430 ms (42.7 fps)** at 1440p.

**THE NUMBERS ARE A RANGE, NOT A FIGURE** — the I3 law about a wall clock, applied to the
instrument that produces them. Three independent release runs of the same tree, same
machine, same adapter:

| | 1080p p50 | p95 | p99 | GPU frame | 1440p p50 | p95 | GPU frame |
|---|---|---|---|---|---|---|---|
| run 1 | 39.792 | 45.057 | 46.709 | 15.875 | 47.424 | 51.136 | 22.716 |
| run 2 | 40.955 | 45.165 | 48.877 | 19.776 | 48.218 | 51.587 | — |
| run 3 | 40.517 | 46.096 | 47.259 | 17.615 | 43.281 | 49.268 | 21.436 |
| **audit 4** | **39.917** | 45.173 | 47.449 | **16.949** | 46.439 | 49.909 | 22.413 |
| **audit 5** | **37.803** | 45.450 | 47.074 | **14.395** | 44.397 | 49.263 | 21.862 |
| **audit 6** | **39.561** | 43.679 | 44.657 | **17.323** | 46.285 | 49.917 | 22.121 |

**Four of the wave's five ranges were too narrow** (the I4 audit's three runs): 1080p p50
becomes **37.8–41.0**, the GPU frame **14.4–19.8** (a **37 %** spread, not 20 %), the sim
step **13.0–14.9**, the pipelined estimate **20.3–22.6**. The **CPU-side p50 is stable to
about 5 %**; the GPU frame's swing is boost clocks and nothing the engine did — and is
exactly why `SHIPPING_FRAME_CEILING_MS` is 58.0 and not 47. The *shape* is what reproduces:
the frame is CPU-bound in every run, the sim fixed step is 13.0–14.9 ms in every run, and
the scatter pass is **67.6–68.1 %** of the 1080p GPU frame in every run that printed it.
Quote the shape; treat any single millisecond as ±20 %.

**And the camera moves 30 m per round.** `CITY_DRIVE_STEP_M` is 0.25 m, so the whole
four-pass run covers **120 m of a 1 260 × 900 m city**. Every number above describes one
neighbourhood at one vantage.

**IB-9 · both terrain budgets derived from one measurement.** The certification found
`TERRAIN_RESIDENT_BYTES_CEILING` (16 MiB) and `max_resident_tiles` (1 024 = 264 MiB at 257²)
**16.4× apart** with nothing making them agree. They were two guesses at one quantity —
how many pages a render cut holds — which is now `inf_terrain::stream::cut_page_bound`.
Measured over a flythrough across each world's own diagonal:

| world | levels | catalog | **peak cut** |
|---|---|---|---|
| 8 × 8 | 3 | 84 | 64 (clips) |
| 16 × 16 | 4 | 340 | **157** |
| 32 × 32 (island class) | 5 | 1 364 | **250** |
| 64 × 64 | 6 | 5 460 | **340** |
| 32 × 32 @ 65² page | 5 | 1 364 | **250** (page-size independent) |

`StreamBudget::for_ladder(levels)` is the bound, with **no second multiplier** —
`sync_render` evicts outside the cut it publishes and then loads, so steady residency *is*
the cut, and the bound already exceeds its measurement by 1.75×. Both hosts size the budget
to the terrain in hand. **They meet at this scene's ladder**: `phase16_gate` arm (e) computes 200 pages → **12.70
MiB** at 129² against a **16 MiB** ceiling and a **5.90 MiB** measured peak. Before, the
same arithmetic gave **65 MiB against 16 MiB**. `RENDER_LOD0_RADIUS_TILES`, declared twice
with a `MIRROR of` comment, is Ring 0 now.

**What IB-9 did NOT close** (the I4 audit): the certification's other sentence, *"at island
scale the ratchet fires first"*. `TERRAIN_RESIDENT_BYTES_CEILING` is still a **flat constant
sized for the gate scene** while `for_ladder` grows with the terrain in hand, so at the
certification's own island row — 32 × 32 level-0 pages, five levels, a 257² page — the
derived budget is **464 pages = 116.91 MiB** and the measured peak cut **250 pages =
63.0 MiB**, against the same **16 MiB**: **7.3×** and **3.9×**, where the finding recorded
16.4×. The wave narrowed the disagreement and did not remove it. `phase16_gate` arm (e2)
asserts the gap, so the day the ceiling becomes a function of the terrain the arm goes red.
Routed to **IP**.

**IB-16 · the per-frame VT upload budget.** `inf_stream::AdmitBudget`, in the one admission
walk both page systems run, in **bytes** (an RGBA8 transcode page is 8× BC1's, so a page
budget throttles a BC-less adapter eight times as hard while looking identical). Default
**1 MiB/frame** = 113 BC1 pages. Burst proof: 40 tiles at 4 pages/frame drain in **exactly
10 frames**, worst upload 36 992 B against a 36 992 B budget, 180 deferrals **all flow
control**, every tile arrives. The advisory fires on a **run** of 15 throttled frames and
resets when one drains. Deferrals are counted apart from pool-full, because the two want
opposite fixes.

**IB-12 · the camera at partition scale.** Error against the same relative run at the
origin, 600 steps at 37° yaw:

| anchor | delta form (now) | absolute form (before) |
|---|---|---|
| 1 km | 6.551e-14 m | 2.618e-6 m |
| 50 km (the island) | 8.207e-11 m | 1.309e-4 m |
| 500 km | **7.082e-10 m** | **1.309e-3 m** |

**1.848e6× better.** A rebase mid-lag moves the render-local camera by the world step and
the origin's snap and nothing else (2 rebases over 3 km, worst unexplained 3.7e-5 m = f32 at
1024 m). A partition handoff at speed — 7 crossings of a 256 m cell at 20 m/s, each with a
one-step subject gap — is absorbed by the lag: **0.385 m recovery against a 0.377 m steady
step, 1.02×**, not a cut.

**Clause 6 · the LOD pop, measured, and the cross-fade REFUSED.** At 1080p, noise floor
**zero** (the frame is bit-deterministic across two fresh renderers):

* **as geometry** at 192 m the building is **88 × 33 px** (the I3 ledger's "about thirty
  pixels tall", now measured); the swap moves **63 of its 2 903 px (2.2 %)**, worst channel
  **18/255**, frame-level perceptual mean **0.00000**. Invisible. **Refused**, armed against
  the golden harness's own re-render tolerance.
* **and the finding beside it**: `mesh_distance_m` is **120 m**, so *at* the 192 m swap the
  scatter path is already in its **impostor** band — and a building's impostor silhouette is
  **19.2×** its mesh's (55 868 px against 2 903), of which **93.9 %** moves.

  *The attribution, corrected by the I4 audit.* Both frames in that reading are impostors,
  so the **change is still the band pair's**; what the impostor owns is its **size**, because
  a billboard is sized from the instance's bounding sphere. "The discontinuity belongs to the
  impostor band rather than to the structure band pair" sends the next reader to the wrong
  repair. The refusal is therefore **conditional**: no cross-fade, because as *geometry*
  there is nothing to fade — and the first repair at this distance is the **billboard's
  sizing**, not a fade. Both halves are armed now, the shipped half as the refusal's own
  condition, so fixing the sizing turns the file red and clause 6 is re-decided.

  *And the refusal's armament was itself a clause that could not fail.* It was held against
  `GOLDEN_MEAN_TOLERANCE`, which averages over a 64 × 36 downscale — an arithmetic ceiling of
  **0.0014** for anything a 2 903-pixel object can do, 43× under the 0.06 tolerance. It is
  armed on the object now (≤ 5 % of the building's own pixels, worst channel ≤ 32/255), and
  the "bit-deterministic" noise floor is `assert_eq!`d at zero per configuration rather than
  printed.

**Clause 5 · real building geometry — the path proven, the price re-measured.** bake →
`MeshAsset` → `build_vgeom` → `VgeomAsset` → `VgeomInstance` runs end to end in memory with
no schema and no cook: **424 parts, 10 176 vertices / 5 088 triangles, 144 meshlets**, and
the packed `.inf_vmesh` header reads back an 18.511 m bounding radius.

*(The "and such a frame carries the baked id on every instance with zero placeholder
batches" half was a **fixture tautology** — the scene is hand-built with `vgeom_instances`
and `..Default::default()`, so `scatter.is_empty()` asserted that `Vec::new()` is empty.
Deleted by the I4 audit along with three more identities. The projector still emits
placeholder cubes for every PCG structure; that is the gap, and it is IP's.)*

**The price, corrected.** The wave measured 434 176 cubes at 3.378 ms against 1 024 baked
instances at 4.793 and recorded **+1.416 ms DEARER**. Both halves of that were wrong: the
cube side was **distance-culled at 400 m** (432 046 of 434 176 thrown away, **zero** drawn
as meshes) against a meshlet path with no distance cull at all, so the delta was between two
culling policies; and a 30-frame mean answered with a **different sign on three of four
runs**. With both sides audited, the cube side measured at both band settings, and
MIN-of-5-rounds, the three numbers reproduce to three decimals:

| | GPU frame |
|---|---|
| 434 176 cubes at the shipped bands (nearly all distance-culled) | **0.539 ms** |
| the same cubes with the bands opened past the district | **1.25 ms** |
| 1 024 baked vgeom buildings (146 067 meshlet pairs drawn) | **1.573 ms** |

Real geometry is **dearer by about +0.32 ms** against a comparable configuration (+1.03
against the shipped bands). The direction the wave recorded survives; **1.416 ms does not
and must not be quoted.** It is still a fidelity decision with a cost — a much smaller one.

### Counts

| | after the I3 audit | after I4 | **after the I4 audit** |
|---|---|---|---|
| battery blocks / passed / failed / ignored | 298 / 5 670 / 0 / 13 | 305 / 5 689 / 0 / 14 — 7 new test binaries, **19** new arms (20, less the one clippy turned into a compile-time `const` assertion), and the one new `#[ignore]` is the 257²-page island confirmation | **305 / 5 690 / 0 / 14** — the audit adds **exactly one** arm and no test binary; the rest of its work sharpened arms that already existed and **deleted four that could not fail** |
| frontend tests / files | 702 / 78 | 702 / 78, `tsc` clean | **702 / 78**, `tsc` and `eslint` clean — no UI was touched by either |
| goldens | 54, byte-frozen | 54, byte-identical under `INF_GOLDEN_STRICT=1` | **54, byte-identical under `INF_GOLDEN_STRICT=1`** (100 golden arms, re-run on the audit's head) — and `timing_changes_no_pixel` is the arm that says an attached GPU stopwatch cannot move one, mutation-verified |
| `clippy --workspace --all-targets` `-D warnings` | 0 | 0 | **0** |
| rustdoc individual warnings (cold, all roots touched) | 412 | 412 | **412** — 446 `^warning` lines − 34 per-crate summaries, unmoved |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | unchanged — every struct the wave grew (`VtPoolConfig`, `VtStats`, `VtTransaction`, `AdmitLog`, `VirtualTextureSettings`, `RenderSettings`) carries no serde derive; checked rather than assumed | **unchanged — no schema moved, no golden moved, no committed sample moved** |
| new ratchet constants | — | `SHIPPING_FRAME_CEILING_MS` 58.0, `SHIPPING_FRAME_P99_CEILING_MS` 64.0 (targets 16.6 / 33.2, never asserted); `StreamBudget::default().max_resident_tiles` **1024 → 860**, derived | unchanged — the audit minted none and raised none |
| committed samples | 20 levels | **20** — the instrument's scene is composed at test time and writes nothing into `samples/` | 20 |

## The I4 audit (adversarial, `1d33295..d67e180` audited; the audit's own commits follow it)

**Every headline measurement HELD on re-measurement**, from the tests that print them — the
IB-12 lag pair and its 1.848e6×, 2 rebases at 3.7e-5 m, the handoff at 1.02×, the cut at
64/157/250/340 with a constant ring, 200 pages → 12.70 MiB, the burst at exactly 10 frames /
36 992 B / 180 flow-control deferrals, the advisory at frame 14 of 15, the P28.5 trio
(19 542 / 19 766 · 18 752 / 18 976 · 76 074 / 75 928), the LOD pop at 63 of 2 903 px with a
zero noise floor, the impostor at 19.2× / 93.9 %, the bake at 424 → 5 088 → 144, goldens 54,
schemas unmoved, and the eaten-continuation gate green (the 57 repairs are complete). The
instrument reproduces **in shape** on three independent release runs and outside **four of
the five ranges** the wave tabulated.

**Three findings are large enough to change what the ledger says:**

| finding | what shipped | now |
|---|---|---|
| **The instrument measures a frame with the expensive half of the renderer OFF** — no shadows, GI, VSM, TAA, SSAO, bloom or visbuffer — and nothing said so | a "≥ 60 fps means this" constitution quoting 39.8 ms for an unlit frame | the harness prints what it does not draw and **measures the stack**: p95 92.3-92.9 lit against 43.7-44.0, GPU frame 35.8-36.1 against 17.3-19.4 |
| **Clause 5's price compared two culling policies and its sign moved** | "+1.416 ms dearer", from a cube side that drew **zero** meshes (432 046 of 434 176 distance-culled) against a meshlet path with no distance cull, on a 30-frame mean that answered −2.03 / −0.28 / +2.00 / −0.86 across four runs | both sides audited, the cube side measured at both band settings, MIN-of-5-rounds: **0.539 / 1.25 / 1.573 ms → +0.32 ms** against a comparable configuration, reproducible to three decimals |
| **IB-16's one production door was unarmed** | `build_vt_level`'s `upload_budget_bytes: 0` left the entire tree green with the throttle off in both hosts | `the_registration_door_carries_the_upload_budget` — 1 of 16 seated at a one-page budget, 16 of 16 unthrottled as the control |

**Six arms that could not fail, mutation-measured and closed:**

| arm | why it could not fail | now |
|---|---|---|
| clause 6's refusal (`mean <= GOLDEN_MEAN_TOLERANCE`) | `image_diff` averages over a 64 × 36 downscale, so a 2 903-px object moves the mean by at most **0.0014** — 43× under the tolerance at its arithmetic maximum | armed on the object: ≤ 5 % of its own pixels, worst channel ≤ 32/255 |
| the LOD noise floor | measured with impostors **ON**, compared against a delta taken with them **OFF**; and "bit-deterministic" was printed, never asserted | one floor per configuration, both `assert_eq!`d at zero |
| the shipped LOD configuration's 93.9 % pop | no assertion at all | armed as the refusal's **condition** |
| the handoff ceiling | `3 ×` a steady step, where a full cut is **1.77 ×** — above the defect it named | `1.5 ×`, with the cut computed, printed, and a clause pinning the ceiling below it |
| the rebase arm | rounded the unexplained step to the nearest `ORIGIN_SNAP` on **every** step, forgiving a 10 m teleport | forgiven only on the step `maybe_rebase` reports |
| clause 5's "ZERO placeholder batches" | `scatter.is_empty()` over a scene built with `..Default::default()` and only vgeom pushes — one of four identities | deleted, with the packed asset's own header read back in their place |

**Two claims retracted, one qualified:**

* **"A MIN over different stretches reported 28.3 ms."** Does not reproduce:
  `CITY_DRIVE_STEP_M` is 0.25 m, a round is 30 metres, and removing the `step = 0` reset costs
  about **1 %**. The law and the fix are kept; the figure is retired.
* **"+1.416 ms dearer."** See above — **+0.32 ms**.
* **"The cook does not evaluate PCG volumes"** is accurate and needs one qualification: a
  cooked pack *does* ship its PCG buildings, evaluated at load. What does not exist is an
  evaluated population at **cook time** to bake from.

**A scripted insertion matched three anchors it was not meant to.** The wave's `### Counts`
table was pasted into the Phase 25, wave I1 and wave I2 blocks of `docs/ROADMAP.md`, each
claiming "after the I3 audit / after I4" as though it were theirs. Three deleted. That is the
chr(92) law's cousin and its **thirteenth catch**: a scripted edit anchored on a heading that
occurs more than once, in a file where the gate that catches eaten continuations cannot see a
table in the wrong section.

**Three ledger numbers no test printed, corrected**: "sixty `timestamp_writes: None`" (61 on
the wave's own base tree), `whip_pan`'s "1 220 blur tiles" and "19 972 / 20 196" (790 and
19 542 / 19 766 — the arm beside them prints it), and `SHIPPING_FRAME_CEILING_MS`'s "after 24
warm-up" (a discarded pass of 120). Plus three docs a change made false:
`VtStats::budget_clamped` ("for want of a slot" — a throttled want is deferred too),
`inf_player::terrain_stream`'s `max_resident_tiles` (1024 → derived), and
`island_working_set`'s "island-class" label on the 64 × 64 row.

**Twenty-three mutations were run** — five of the implementer's and eighteen new — and after
the repairs each dies at exactly the arm that names it and at no other. `docs/ROADMAP.md`'s
audit block tables them; the two recorded as *coverage bounds* rather than defects are the
0.015 ms timestamp readback (below any tolerance a wall-clock sum can carry) and
`VirtualTextureSettings::default().upload_budget_bytes`, which `whip_pan` cannot see because
it builds its pool config directly.


**Certification.** Wave I4 is **certified** at this tree. Its four items are closed as
follows: the **fps instrument exists and is honest about its own configuration** (the audit's
addition: it now names what it does not draw and prices it); **IB-12 is closed** and its two
blind arms are repaired; **IB-16 is closed** and its one production door is now armed;
**IB-9 is closed as a derivation and carried as a ceiling**, with the gap asserted so it
cannot be forgotten. Clause 6's refusal stands and is now armed on the object it is about,
conditionally on the billboard sizing. Clause 5's path is proven and its price is corrected
from +1.416 ms to **+0.32 ms** against a comparable configuration.

Battery **305 / 5 690 / 0 / 14** (the audit adds exactly one arm and deletes four that could
not fail), frontend **702 / 78** with `tsc` and `eslint` clean, goldens **54 byte-identical
under `INF_GOLDEN_STRICT=1`**, `clippy --workspace --all-targets -D warnings` **0**, rustdoc
**412** individual warnings from a cold re-documentation, `cargo fmt --all --check` clean,
**no schema moved, no golden moved, no committed sample moved**. Twenty-three mutations were
run and each dies at exactly the arm that names it.

**No blocker for IP.** The one item that could have been — the cook's missing PCG evaluation
— is verified, qualified and routed as IP's first clause rather than left as a sentence.

---

## Open questions / carried bounds

**Closed by I4** (with the I4 audit's amendments applied above):
* ~~**IB-9**~~ *(the derivation; the island-scale ceiling is carried — see IP)* —
  `cut_page_bound` is the one quantity; `StreamBudget::for_ladder` and
  `resident_bytes_bound` are two readings of it, and `phase16_gate` compares them
  (12.70 MiB derived against a 16 MiB ceiling, where the old arithmetic gave 65 MiB).
* ~~**IB-12**~~ — the lag unrotates the delta; 1.848e6× at 500 km, a rebase mid-lag moves
  nothing the origin's snap does not explain, and a handoff at speed costs 1.02× a step.
* ~~**IB-16**~~ — a per-frame upload budget in bytes, floor-exempt, burst-smoothed
  ("late, never never"), with a sustained-demand advisory; the three named tripwires
  re-aimed and the P28.5 ruling re-measured with the throttle on.
* ~~**No fps instrument at shipping resolution exists**~~ — `fps_instrument.rs` at 1080p and
  1440p over composed content, with the per-pass GPU clock. The numbers are above, and they
  are not 60 fps.
* ~~**The parts↔shell swap is a hard cut**~~ — **refused with the measurement**: 63 px of a
  2 903 px building, worst channel 18/255. The pop a player sees at that distance is the
  impostor band's, carried below.

**Measured and bounded (I4 owns these numbers):**
* **THE FRAME IS CPU-BOUND, AND THE SINGLE DEAREST THING IS THE FIXED STEP.** **13.0–14.9 ms**
  at 1080p and 14.3–15.6 at 1440p over six runs, against a 14.4–19.8 / 21.4–22.7 ms GPU frame
  — the *whole* GPU frame at 1080p costs barely more than one sim step. Nothing in this tree
  had measured it over a thousand-building city before, and **no §8 budget covers it**:
  `STREAMED_STEP_BUDGET_MS` is 4.0 ms and is asserted over the phase-16 gate scene, which is a
  walker on a heightfield. Deleting the scatter pass entirely would still leave a ~21 ms
  frame. **This is where the next 60 fps work is**, and it is a sim question rather than a
  render one. What the step is made of has never been broken down: the I3 collider band is
  **~2.0–2.2 ms** of it (1.991 / 2.13 / 2.06 ms measured on the same city, 2.202 computed at
  the certification's rate), and the other **~11.5 ms is unattributed** — movement,
  animation, the physics step over the 6 067 banded colliders, the change-stamp scans.
  *The one cross-check that reconciles: the band IS engaged in the instrument's scene. An
  unbanded city steps in **134.480 ms**, so a 13–15 ms step could not be an unbanded one —
  the city fixture's Driver carries its `StreamingSource` into the composed level and the
  band fails closed on it.*
  **→ IP: a §8 budget for the fixed step over a CITY, and a breakdown that names its parts,
  before anything is optimised.**
* **…and the lighting stack is the other half, unmeasured until the audit.** The frame above
  has no shadows, GI, VSM, TAA, SSAO or bloom in it. Turned on: **p95 92.3-92.9 ms, GPU frame
  35.8-36.1 ms** (scatter 16.124 | gi 6.049 | terrain 4.716 | vgeom 2.495 | vsm-raster 1.411 |
  sky 1.319). Any 60 fps target that includes shadows is starting **56 ms** from 16.6, not
  27. **→ IP.**
* **The instrument's frame is SERIALIZED and a presenter's is not.** It polls to completion
  every frame, because a frame time without a sync point is a submission time; a real
  presenter overlaps the halves. The pipelined estimate (20.975 / 23.430 ms) is arithmetic
  over two measurements and is printed, never asserted. A windowed harness that measured
  present-to-present would be the honest closure and needs a window.
* **The instrument asserts nothing in the `dev` profile**, so the full battery runs it as a
  functional smoke and the ratchets only bite under `cargo test --release`. A regression that
  lands between two release runs is caught late.
* **The composed instrument scene has no virtual textures** (`0 virtual textures` in its own
  report): the city's materials bind no `.inf_tex`, so the `vt-stream` segment costs 0.002 ms
  and the SVT stack is *present but unengaged* in the headline frame. IB-16's own numbers
  come from `whip_pan`'s four 2048² surfaces instead. A textured island fixture would make
  the instrument's VT column mean something.
* **A building's scatter IMPOSTOR is 19.2× its mesh's screen area** (55 868 px against
  2 903 at 192 m), and 93.9 % of it changes at the structure swap. An impostor is sized from
  the instance's bounding *sphere*, and a 20 × 30 × 7.4 m box's sphere is much wider than the
  box — so a non-uniformly-scaled instance gets a billboard far larger than its silhouette.
  Armed as a bound in `structure_lod_pop.rs`, not fixed.
* **The cook does not evaluate PCG volumes, which is the one thing between the engine and a
  city of real geometry.** *(Verified by the I4 audit: `inf-packager` decodes a `.inf_pcg`
  for two advisories and a dependency edge and calls none of `evaluate`,
  `evaluate_grammars`, `evaluate_buildings`, `compose_volume` — its own manifest says so.
  The three production evaluation paths are `InfSceneWorldBuilder::build`, cell activation
  and the editor's `pcg_evaluate` command. **Read it precisely**: a cooked pack DOES ship its
  PCG buildings, evaluated at load; what does not exist is an evaluated population at COOK
  time to bake from.)* The bake→vgeom path works in memory (424 parts → 144 meshlets),
  `bake.rs`'s dependencies are all Ring 0 so `inf_dcc::bake` is acyclic, and `inf-packager`
  taking `inf-dcc` would keep the kernel out of the shipped player — a one-line Cargo change.
  And the measurement says it costs **+0.32 ms** against a comparable configuration rather
  than saving time, so it is a fidelity decision to be taken knowingly.
  **→ IP's first clause.**
* **A streamed cell evaluates its `PcgVolume` and NOT its biome bindings** (I4 audit,
  observation on I1's fix). `evaluate_biome_bindings` has one call site,
  `InfSceneWorldBuilder::build`; `cell_stream::reconcile` mirrors `page_terrains_for_pcg` and
  `evaluate_pcg_volumes_in` and has no biome twin. Pre-existing, not this wave's, and nothing
  in the tree has a partitioned biome-bound terrain yet. **→ IP.**
* **`AdmitBudget` is a seat count derived from bytes, so it is exact only for uniform-size
  pages.** True of every VT pool by construction (one format, one slot size) and *not* true
  of the meshlet pools, which is why `inf-vgeom` deliberately does not implement `SlotPool`.
  A future byte-accurate throttle over variable-size pages is a different mechanism.
* **`resident_bytes` still counts heights and materialized weights only** — not `maps`
  (erosion, `res² × 12 B`), `biomes` (`res²`) or `holes`. So an eroded, painted or carved
  terrain under-reports against both the ceiling and `resident_bytes_bound`, which are now
  derived from the same under-count and therefore still agree with each other.
* **`cut_page_bound` is a bound and not the cut.** It allows 464 pages at five levels where
  250 are measured — 1.75× — and the tightness clause only requires it to be within 2×. A
  budget derived from a bound is generous by construction; the alternative is a budget
  derived from a measurement, which would be a budget that a slightly different ladder
  overruns.
* **The lag's last precision is the accumulator, not the rotation.** `pivot` is still an
  absolute world position, so `current.x + world.x` loses ULPs at the anchor's magnitude —
  7.082e-10 m at 500 km. Closing it means an origin-relative camera, which is a different
  change.

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
  `fade_band_m` resolves mesh↔impostor with a complementary dither, which is per-pipeline
  rather than per-band. At 192 m the pop is small; closing it needs the fade to become a
  property of the *band pair*.
* **…and the two bands now OVERLAP by the batch's own reach** (18.486 m on the city), which
  is the I3 audit's fix and its price. In that annulus a building draws its parts *inside*
  its own shell, and the shell's faces are the parts' outermost faces, so they are coplanar:
  expect z-fighting on the outer walls of the ~27 buildings that are in the band at any
  moment, at 192 m, where a two-storey building is about thirty pixels tall. The alternative
  is the defect it replaced — a hole through the back of the building — and the two cannot
  both be avoided by a cut that compares the eye against each instance. Closing it properly
  is the same item as the cross-fade above: a band pair that dithers.
* **The lattice has an edge and a source can park on it.** The quantization buys "membership
  changes on a crossing, not every step" for a source *travelling*; a source sitting on a
  lattice line with sub-millimetre jitter crosses every step and re-describes the active set
  each time — measured at **59 re-stamps in 60 steps, over exactly 2 bands**. Hysteresis is
  **refused**, not deferred: it would make the band a function of the history of positions
  rather than of the positions, and being a pure function of sim state is what makes
  PIE == shipping hold. The active set is never wrong; it is re-described until the source
  moves off the line.
* **A far building's shell is a BOX, not its baked mesh.** IB-2b's brief names the P23
  grammar→mesh bake feeding vgeom as the far tier's geometry. What ships is the derived
  oriented shell — correct in silhouette and proportion (the non-uniform instance scale
  exists for exactly that), and consistent with the near tier, which draws placeholder cubes
  too (`kind_index → real mesh` is the standing P19 gap). Wiring `bake_building_in` into a
  runtime vgeom asset per archetype is a project; `gisbuild` already bakes footprints to real
  meshes on the authored path and is where it would start.
* **The city fixture's ground is FLAT.** Deliberate — see the sample's README. It means the
  fixture says nothing about a banded city over streamed terrain, which is I7's.
* **The bridge holds one `Uuid` per admitted collider, and fail-open admits the world.**
  97 KB while banded on the city fixture; **5.9 MB** unbanded, because the retain pass cannot
  re-derive the attached set from a count once a band exists. The price of the direction that
  keeps a body on the floor, stated on the field.
* **The city has no SCATTER prefix, so `compose_volume`'s rebasing is armed by its unit test
  alone** (I3 audit). Every block is buildings, so `shift` is 0 at world scale and dropping
  the `inst_start` rebase leaves all nine city arms green. The unit test does catch it. A
  composed fixture — grammar *and* scatter in one volume — is what would arm it end to end.
* **`phase19_gate`'s "13 000 colliders" is the SOLID count, not the bridge's** (I3 audit,
  observation on a prior phase's arm — not changed here). It reads `solids(&built).len()`,
  so a band that dropped every one of the town's colliders leaves all twelve of its arms
  green and its step *faster*. The town has no `StreamingSource` and is therefore unbanded by
  construction, and the fail-open property is armed at the door by both city gates (removing
  the source must price the unbanded alternative), so nothing is wrong today — but the P19
  arm measures what was built rather than what is simulated, and a future wave that bands
  more aggressively should fix that before trusting it.
* **A `PcgVolume` is banded as a whole in the RENDER path.** The collider band is per
  building (per `StructureGroup`); the draw band is per instance on the GPU, but the three
  batches are built per *volume*, so a volume spanning a whole district would put its parts
  batch in one content key. City blocks are 100 × 60 m and this is not reached.
* **The visbuffer costs 8 bytes a pixel now**, four more than before — **7.4 MB** at
  1280 × 720 and **66 MB** at 4K in total, of which the widening added 3.7 MB and 33 MB.
  (The first write-up paired the 8 B/px label with the *delta's* megabytes; corrected by the
  I3 audit, in all three documents that carried it.) Paid only while
  `VgeomSettings::visbuffer` is on, which is `false` on every tier.

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
* ~~**IB-8**~~ — the id is 64 bits; `VIS_MAX_INSTANCES` is 16 777 215. The 32-bit re-cut the
  certification proposed was measured and refused (see the decisions above).
* ~~**IB-13**~~ — `project_delta`; 52.857 ms → 8.105 ms on a drag frame at 100 000 entities
  and 0.0006 ms on a select frame. The residue is `EcsWorld::propagate`, named above.

**Pending their waves (from the certification):**
* **IB-11's far half** — LERC / BigTIFF / JPEG2000 / LAS, reprojection, the geoid. I5's.
* **IB-15's open halves** — the 66.8707° adjacent-terrain normal seam, and the fact that
  abutment is conditional and unenforced.
* ~~IB-12~~, ~~IB-16~~ and ~~the missing fps instrument~~ — **closed by I4**, above.
  **IB-9 is closed as a derivation and open as a ceiling** (I4 audit): the two readings agree
  at the gate scene's ladder, and at island class the derived budget is 116.91 MiB against a
  flat 16 MiB ratchet. The instrument exists, and what it says is that the engine renders
  this content at **25–26 fps at 1080p** and **21–23 fps at 1440p** on an RTX 4070 Ti with
  **no shadows, no GI, no VSM, no TAA, no SSAO and no bloom** — and at **13.3 fps** with the
  authorable half of that stack on. Nothing in this repository may claim ≥ 60 fps; what it
  may now claim is a number, and the number has a configuration attached to it.

---

## What IP inherits (the performance wave), in the order the numbers imply

Routed here by the I4 audit. Every item is a number this tree already prints.

1. **The cook does not evaluate PCG volumes.** The one thing between the engine and a city of
   real geometry, and the only item on this list that is a *feature* rather than a budget.
   `inf-packager` + `inf-dcc` is a one-line Cargo change; what is missing is an evaluated
   population at cook time to bake from. Everything about the shape of the fix is above.
2. **The sim fixed step has no §8 budget and costs 13.0–14.9 ms on the city.** Mint the
   budget over a CITY (`STREAMED_STEP_BUDGET_MS` is 4.0 ms over a walker on a heightfield),
   and break the step down before optimising it — 2.2 ms is the I3 collider band at the
   certification's own rate and the other ~11 ms has never been attributed.
3. **The lighting stack costs +42.9 ms of p95 and is in no ceiling.** Shadows, GI, VSM, TAA,
   SSAO and bloom are off by default, so every frame number this repository carries is for a
   frame without them. Decide what a shipped island level authors, then re-mint the ceilings
   over that configuration.
4. **IB-9's ceiling is a gate-scene constant at island scale** — 116.91 MiB derived and
   63.0 MiB measured against 16 MiB. `phase16_gate` arm (e2) holds the gap open.
5. **The scatter impostor is sized from a bounding SPHERE**, so a 20 × 30 × 7.4 m box gets a
   billboard 19.2× its silhouette and 93.9 % of it changes at the structure swap. This is the
   repair clause 6's cross-fade refusal is conditional on, and `structure_lod_pop.rs` goes red
   the day it lands.
6. **`resident_bytes` counts heights only** — not `maps`, `biomes` or `holes` — so an eroded,
   painted or carved terrain under-reports against both the ceiling and the bound, which are
   now derived from the same under-count.
7. **A streamed cell evaluates its `PcgVolume` and not its biome bindings.**
8. **The instrument's camera covers 120 m of a 1 260 m city**, and its scene has **zero
   virtual textures**. Both bound what the headline number is a number *about*; a textured,
   longer-path fixture is what would make the VT column and the p99 mean something.
