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
| **I4b** | **performance** — the sim fixed step, the lighting stack, the scatter impostor, the pipelining | **DONE + AUDITED** — battery 306 / 5 710 / 0 / 14, frontend 702 / 78, goldens 54 strict, clippy 0, rustdoc 413, no schema moved. Step 12.7 → **1.25 ms**; lit p95 92.3 → **38.9–41.3**; unlit 1080p **inside the 60 fps budget at p50 on every run and at p95 on two of three** (the audit's correction). One defect fixed: the scatter caster cache was blind to the floating origin |
| **IP** | the remainder of the performance list — the cook's PCG evaluation, IB-9's island ceiling, the VSM caster scatter | **carried** — see *What is still open after I4b* below |
| **I5** | **player core** — the owner's binding table, the C key's four verbs, the in-game UI layer, the settings dialog + rebinding, the interaction core | **DONE + AUDITED** — battery 309 / **5 796** / 0 / 14, frontend 702 / 78, goldens 54 strict (101 arms), clippy 0, rustdoc 413, no schema moved. Two pre-existing ragdoll defects found and fixed by the wave; **five more found and fixed by the audit** (A1 the settings-dialog lockout, A2 the half-built Simulate pause mirror, A3 an unarmed determinism sort, A5 three silent dead sliders, A6 two false doc invariants) plus A4 an arm that could not fail and A7 a count that was one low. See below. *(This wave took the I5 slot; **IB-11's far half — LERC / BigTIFF / JPEG2000 / LAS, reprojection, the geoid — is NOT in it** and moves to a later wave.)* |
| ~~I6 (old)~~ | ~~scale seams — IB-12~~ *(pulled into I4; IB-8 and IB-13 into I3)* | **absorbed** |
| **I6** | **gameplay systems** — doors + locks + the kick + crash-through, inventory, weapons v1, health | **DONE + AUDITED** — battery 312 / **5 873** / 0 / 14, frontend 702 / 78, goldens 54 strict (101 arms), clippy 0, rustdoc **404** of a 450 ceiling (447 at the wave's head, re-measured cold; 39 cleared by the audit), **no schema moved**. `NOT_YET_CONSUMED` is empty. Six defects found by the wave's own world-level arms and five more by its gate; one energy door for the kick, the breach and the bullet (mutation-proved across two crates); the city plans **19 790** doorways and the band makes **234** solid. The audit found **five arms that could not fail** (two trace sections, the wheel verb, the corpse guard, the spent attack edge) and **three world defects** (a barged door lost its lock for ever, `door.is_open` walked all 19 790 doorways, a dead block claiming a swing it did not drive). See *Done — wave I6* and *The I6 audit* below |
| **I7** | **the island data build** — the recipe, real Vancouver elevation, the designed coastline, the derived water and biomes, the graded roads, the level | **DONE** — see *Done — wave I7* below. The island exists: **51.38 km² of map, 40.65 km² of land, a 948.7 m peak of real North Shore survey, 25.14 km of designed shore, 50 reaches / 26.32 km, 2 lakes, 33 waterfall sites, 33.74 km of graded road, 342.7 MB of terrain built by one command in 24.7 s.** PIE == shipping over a 900-step drive. Battery 318 / 5 946 / 0 / 16, frontend 702 / 78, goldens 54, clippy 0, rustdoc 404, **no schema moved**. **Then CI went red on macOS and ubuntu** — one ulp of proj4rs latitude in a committed level, and a 2 ms sleep that took 5 on a shared runner; both fixed, recipe schema **1 → 2** (the recipe now *states* its geodetic origin), engine schemas still unmoved. See *The I7 CI-red* at the end of this file |
| **I7b** | **the island lives at 60** — the vegetation through the shipped boot, the VSM caster pack, `render (record)` attributed | **DONE + AUDITED** — see *Done — wave I7b* and *The I7b audit* at the end of this file. The shipped island frame goes **41.5 → ~277 fps** (p50 24.080 → 3.56–3.62 ms, **13 ms inside** the 60 fps budget where it opened 7.5 ms outside) and grows **2 681 instances** of vegetation where it grew none; PIE == shipping on the forest as well as the state fold. Battery 319 / **5 971** / 0 / 16 at the audited head, goldens 54 strict, clippy 0, rustdoc 374 unmoved, **no schema moved**, no new crate or dependency. Clause 2's routed prescription was **measured and retired** and the real fix routed by name. The audit reproduced every headline number on the same machine and found **no HIGH** — four MEDs, all the same shape: *a claim that is true and a gate that cannot tell* |

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
* **A DEFAULT CHOSEN FOR ONE CASE PAYS IN EVERY OTHER** (I4b). `ActiveCollisionTypes::all()`
  was widened so a static *sensor* trigger over static scenery would report, and it charged
  every static *solid* in the world a contact manifold at 60 Hz — **9.2 ms of a 12.7 ms fixed
  step** on a city, and invisible on every hand-authored level it was chosen against. The
  narrowing keeps `all()` for **sensors** (the flags are tested as the pair's union, so the
  sensor's own carry the pair) and drops `FIXED_FIXED` for solids. *Generalize:* when a default
  is widened for a case, widen it **on that case**, not on the type.
* **The dearest half of a lit frame was the RECORDING, not the drawing** (I4b). Two passes cost
  **12 ms of CPU for 1.3 ms of GPU** — a caster re-pack keyed on a version that moves when a
  pose moves, and a `pages × groups` draw loop over a group set that is one per terrain tile.
  A per-pass report with only a GPU column sends the next reader to the wrong processor;
  `PassTime::cpu_ms` costs one `Instant` per mark and is now beside it.
* **A cache key that over-approximates is a cache that never hits** (I4b). `RenderScene::version`
  is a *guess* at any one subsystem's inputs. Key on what the thing reads, and when the content
  has no cheaper identity, hash the bytes the pass just produced — exact, and `O(n)` in the same
  n the pass already walked.
* **Conservative in ONE direction is a licence; conservative in the other is a bug** (I4b). The
  VSM group mask may name a group the GPU cull finds nothing for and can never hide one it would
  keep. That asymmetry — and only that — is what makes skipping a draw on a clear bit move the
  clock and not a pixel.
* **A GPU millisecond is a fact about the device in the state the frame put it in** (I4b). The
  *unlit* GPU frame fell 14.4–19.8 → 2.2–6.0 across this wave, which nothing on the unlit path
  can explain: I4's frame left the card idle two thirds of every frame and an idle card
  downclocks. **GPU columns are comparable only between runs whose CPU frames are comparable**,
  and a ledger that quotes one across a wave that moved the CPU frame by 25 ms is quoting two
  power states.
* **A fix that lands has to re-take the refusal it was conditional on** (I4b). Clause 6's
  cross-fade refusal was armed to go red the day the billboard sizing was repaired. It did. The
  refusal was re-taken on the geometry numbers, kept, and the arms re-aimed at the repaired
  ratio — *not* deleted, because an arm that fired as designed is the most valuable one in the
  file.
* **Price the fidelity fix too** (I4b). The impostor sizing halves a silhouette and moves no
  milliseconds. Saying so is what stops the next reader budgeting for a saving that is not
  there — the same discipline as pricing what you reject, applied to what you accept.
* **A pending list drained by an EVENT is unbounded when the event is optional** (I4b). The
  incremental query tree's marks are drained by the next *query*, and a level with no character,
  no camera subject and no gameplay cast never makes one. Bounded in `step`: the body list by a
  sort-and-dedup, the collider list by the observation that makes it moot — past
  `colliders.len()` pending, **a fresh build is cheaper than re-inserting**, so the list is
  dropped and the next query rebuilds. Measured at **1 pending body and 17 pending colliders
  after 600 unqueried steps of a 17-collider world**, with a query re-asked before and after a
  forced rebuild, because a bound that works by forgetting is only a bound if what it forgot was
  recoverable.
* **The chr(92) law's FOURTEENTH catch was this wave's own** (I4b). Seven user-facing literals —
  two assertion messages and five `println!`s — shipped mid-wave with the P22 shape, and the
  mangled output had already been read into this wave's own notes. The tree's guard
  (`inf_packager`'s workspace sweep) would have caught every one, so the repair is the repair,
  through the Edit tool, which is what the law prescribes. **And the same sweep found an eighth
  that is a collision rather than a defect**: `rustfmt` aligns trailing comments into runs of
  fourteen spaces on lines that carry string literals, which is exactly the shape the sweep
  reads. *A table maintained around a gate is a table that trips it* — the contents moved onto
  the constants they describe. *(And the reason moving it was the only option: the sweep's
  `ALIGNED_ON_PURPOSE` allowlist is keyed on the **enclosing function**, so a module-level
  `const` cannot be excepted at all — I4b audit.)*
* **A CACHED RENDER-LOCAL PACK IS A FUNCTION OF THE ORIGIN, and two lattices that both
  quantize do not agree** (I4b audit). `pack_fallback` writes render-local model matrices, so
  the bytes it produces depend on the `FloatingOrigin` as much as on the batches — and I4b's
  new scatter caster cache keyed on the eye bucket, the bands, the stamp and the content fold,
  and not on that. The two lattices are **1 024 m** (`REBASE_DISTANCE`) and **8 m** (the eye
  bucket over the *world* eye), so the frame a rebase fires in is almost always one where the
  bucket did not tick: the whole-pack key noticed the rebase, re-uploaded, and re-uploaded a
  merge of the stale half. *Generalize:* when a cache holds a value derived through a frame of
  reference, the frame is part of the key.
* **A GATE THAT REBUILDS ITS OWN CONTROL EVERY ITERATION CANNOT ACCUMULATE THE DRIFT IT IS
  LOOKING FOR** (I4b audit). The incremental query tree's equivalence gate stepped one world
  and forced a rebuild between the two halves of each iteration, so the "incremental" answer
  was always **exactly one step** stale — centimetres against a body's own half-metre leaf.
  Deleting the marking *entirely* left it green over all 540 answers. Two worlds, one never
  rebuilt, and the same mutation dies. *And the second half of the same lesson:* a 40 m ray
  cast from 6 m above traverses a stale leaf as happily as a fresh one, because a leaf decides
  only what the narrow phase is **offered** and the narrow phase then reads the real pose. The
  question that reaches a leaf where the leaf *says* the body is, is a **point query at its
  centre**.
* **A stale BVH leaf is a wasted traversal, not a wrong answer** (I4b audit). `BroadPhaseBvh`
  keys a leaf by `handle.into_raw_parts().0` — the raw index, generation discarded — and
  `QueryPipeline` resolves it through `ColliderSet::get_unknown_gen`, so a leaf a removal left
  behind answers `None` or names whatever collider later takes that index (which the attach's
  own mark refreshes). The `query_rebuild` flag on the removal paths therefore buys the
  *invariant* — one leaf per live collider — and not correctness, and **deleting it kills no
  arm**. Where a flag's value is unfalsifiable through the type's own surface, write that down
  instead of inventing an arm that cannot fail.
* **Two spellings of one index must not both exist where a mismatch is a hole** (I4b audit).
  The VSM group mask re-derived each caster's geometry group with a running cursor over
  `groups[..].casters`, while `vsm_cull.wgsl` indexes the compact slot table with the group id
  the record already carries (`VsmCasterRaw::ids.x`). Exact today, and exact only while
  "casters are pushed contiguously per group" holds in four separate push loops — and a
  disagreement would look up one group's slot for a caster the mask registered under another.
  One field, read twice.
* **An exact cache key costs what it measures, and the HIT is where the cost lands** (I4b
  audit). `scene.version` was a `u64` compared *before* the pack ran, so an unchanged frame
  left the shadow node at `O(1)`; a key computed from the content has to pack and hash it, so
  the **hit** went `O(1)` → `O(scene.instances)` and only the **miss** kept its old order. The
  trade is right where it was taken (the city's casters are all scatter; 3.149 → 0.157 ms) and
  it is a cost somewhere else. Price the direction you did not measure.
* **A MIRROR needs its own arm** (I4b audit). `PhysicsWorld2D::active_collision_types` was
  written as "the MIRROR of the 3D one, which carries the full argument and the measurement" —
  which is two declarations agreeing with each other rather than with the world. The P24 law
  ("two hosts agreeing ≠ the world being right") reaches a *mirror comment* too.
* **A PIPELINED ESTIMATE IS A LOWER BOUND, because the player also waits for the DISPLAY**
  (I4b audit). `SurfaceChain::new` sets `PresentMode::AutoVsync`, so what makes the swap chain
  run out of images is the panel as much as the GPU: the presented cadence is
  `max(CPU without the wait, GPU frame, the refresh interval)` and the estimate is the first
  two. That makes it a **lower bound on frame time and an upper bound on fps** — the direction
  a "≥ 60 fps" claim needs, and the reason "the pipelined model IS the player" was a sentence
  short. It also constrains the owed harness: a windowed one measuring `AutoVsync` measures
  the *panel*.
* **"No gameplay could read it" is a claim about every reader, so enumerate them** (I4b
  audit). Dropping `FIXED_FIXED` for solids removes a *manifold*, and the ledger reasoned from
  the solver — but `ActiveEvents::COLLISION_EVENTS` is on every collider this engine builds
  and both hosts turn a `Started` contact into a Blueprint `Collision` event **whether or not
  the pair is a sensor**. Two overlapping static solids fired one and no longer do. The
  narrowing is still right and the sentence was one word too strong; the remedy an author has
  is the mechanism the flags were widened for, which is to make one of them a sensor.
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

## Done — wave I4b (the performance wave)

**THE INSTRUMENT WAS RIGHT AND THE ATTRIBUTION WAS THE WORK.** Every number below is
before → after on I4's own instrument, RTX 4070 Ti, release, MIN of rounds, same scene.

| | wave I4 | **after I4b** |
|---|---|---|
| **the fixed step over the city** | 13.0–14.9 ms | **1.222–1.258 ms** |
| 1080p **unlit** p50 / p95 | 37.8–41.0 / 43.7–46.1 | **10.8–16.6 / 14.2–19.5** |
| 1440p unlit p50 / p95 | 43.3–48.2 / 49.3–51.6 | **18.2–18.9 / 20.2–25.4** |
| 1080p **lit** p50 / p95 | — / 92.3–92.9 | **32.8–33.4 / 38.1–41.8** |
| 1080p lit GPU frame | 35.8–36.1 | **16.1–16.5** |
| 1080p lit **pipelined estimate** | 24.4 (mid-wave, see below) | **16.4–16.9 (59.3–60.9 fps)** |
| **distance from 60 fps, 1080p unlit** | p50 +23.2, p95 +28.5 | **p50 −5.8, p95 −2.4** |

*The lit pipelined "before" is the one number here that is **not** an I4 number: I4's lit block
printed a p95 and a GPU frame and no pipelined estimate at all — closing that is what this
wave's clause 2 began with — so 24.4 was measured **mid-wave**, after the fixed step's repair
and before the record's. It is the right comparison for clauses 2 onward and must not be
quoted as I4's.*

**The shipped-default 1080p frame is inside the 60 fps budget at p50 — and at p95 on two runs
of three.** *(Corrected by the I4b audit, which is this file's own range law applied to the
wave that wrote it. The distance row above quotes the **best** end of each range as though it
were the whole: the audit's three independent release runs measure the 1080p unlit p95 at
**13.6 / 19.6 / 16.3 ms**, so the distance from the 16.6 ms budget is **−3.0 to +3.0 ms** and
not "−2.4". The p50 is inside on all three — 11.2 / 15.3 / 11.3, distance −5.4 to −1.3. Quote
the shape.)* The **lit** frame is at the line on the *pipelined* measure (16.3–16.7 against
16.6) and not on the serialized one (32.8–33.2) — and the serialized number is the one the
instrument asserts, because a present-to-present harness needs a window this battery does not
have. That gap is the wave's honest remainder and it is a **measurement question, not an
engine one**: the shipped player's frame path is four calls with no device poll, and there is
now an arm that says so — at both scopes and over both modules since the audit.

**The fixed step, phase by phase** — the table wave I4 could not print. Two of the three
answers were surprises:

| phase | before | after |
|---|---|---|
| solver | **9.224 ms (72.5 %)** | 0.319 |
| camera (P29.6, one sphere sweep) | **2.258 ms (17.8 %)** | 0.003 |
| physics3d sync (the I3 band's gather) | 0.932 | 0.769 |
| **the step** | **12.715** | **1.222** |

*The I3 ledger's "the band is ~2.0–2.2 ms of the step" was an inference, not a measurement:
2.202 ms is the certification's per-collider rate times the banded count, which predicts the
**solver's** share of those colliders. The band's own gather costs **0.93 ms**, and the
solver's 9.2 was something else entirely.*

**Both fixes, and what they cost before:**

* `ActiveCollisionTypes::all()` includes `FIXED_FIXED`, so **every banded building box resting
  on a streamed terrain heightfield had a contact manifold recomputed at 60 Hz for a pair no
  solver can move**. On the 1 000-building city with 25 heightfield tiles under it:
  **4.259 → 0.319 ms/step**, of which the ground is +3.704 → +0.036. A **sensor** keeps
  `all()`, which is the one case the flags were widened for.
* `ensure_query_pipeline` rebuilt the whole query BVH whenever anything changed and `step`
  declared it stale unconditionally, so one camera sweep paid a 6 000-collider rebuild every
  step. Incremental now, with a **540-answer equivalence gate** against a forced rebuild.

**The lighting stack's dearest half was the RECORDING.** `PassTime::cpu_ms` (the wall clock
between the same two marks that bracket the GPU segment) named it: `render (record)` was
**18.7 ms of a 50.5 ms lit CPU frame**, and two passes owned it —

| pass | GPU | record before | record after |
|---|---|---|---|
| `vsm-raster` | 0.949 ms | **8.847 ms** | 6.048 |
| `shadow` | 0.344 ms | **3.149 ms** | 0.157 |

The shadow node re-packed **eleven thousand scatter casters every frame** because its cache key
was `RenderScene::version`, which moves when a *pose* moves. The VSM caster pass recorded
`pages × groups` indirect draws — **8 426 a frame** → **384** — because a geometry group is one
per resident terrain tile and nothing told the pass which tiles a 128 m page overlaps; a
per-page group mask, derived free from the invalidation scatter, now does, and drives a compact
slot table that also took 2.16 MB of per-pair uniforms a frame down to 98 KB.

**The impostor's sizing, IP item 5, closed with the pop halved and the frame unmoved.** The
card's radius is the instance's own bounding sphere per primitive kind instead of
`unit_radius × max(scale)`: a building's impostor silhouette at 192 m falls **55 868 → 26 792
px**, i.e. **19.2× the mesh's → 9.2×**. `structure_lod_pop` was written by I4 to go red the day
this landed; it did, and clause 6's cross-fade refusal was **re-taken and kept** on the geometry
numbers (63 of 2 903 px, unmoved). The scatter pass did **not** move (2.98 ms unlit against
2.70–3.31 before; 7.44 lit against 7.49) — a fidelity fix, priced.

**Ratchets:** `CITY_STEP_BUDGET_MS` minted 20.0 → **6.0**; `SHIPPING_FRAME_CEILING_MS`
58.0 → **40.0**; `SHIPPING_FRAME_P99_CEILING_MS` 64.0 → **48.0**.

### The commits

**`3c9d87b..4e6e04c`, fourteen commits** — `36d233a` the fixed step · `1967eab` the lit
frame's recording · `cd3f9b0` the impostor sphere + the ratchets · `2a55e11` the VSM scatter's
own number · `724dbe6` seven eaten continuations · `57cbc05` the pending marks' bound · plus
the ledger commits. *Re-stated by the I4b audit, and this is the **fifth** time this field has
gone stale: the wave wrote `5012f4c..` and named six shas the rebase onto the toolchain hotfix
orphaned — all six. `5012f4c` survives as the pre-rebase base and is not the range's start.*

**Verified at the closing head, release, MIN of rounds:** step **1.267 ms** against a 6.0 ms
ceiling; 1080p unlit p50 **12.076** / p95 **15.095** — **−4.524 and −1.505 ms against the
16.6 ms frame**; 1440p p50 18.192 / p95 19.668; lit p50 33.220 / p95 40.539, GPU frame 16.426,
**pipelined 16.531 ms (60.5 fps)**.

### Counts

| | after the I4 audit | **after I4b** |
|---|---|---|
| battery blocks / passed / failed / ignored | 305 / 5 690 / 0 / 14 | **306 / 5 704 / 0 / 14** — one new test binary and fourteen new arms, which is exactly what the wave added |
| frontend tests / files | 702 / 78 | **702 / 78**, `tsc` + `eslint` clean |
| goldens | 54, byte-identical under `INF_GOLDEN_STRICT=1` | **54, byte-identical** — re-run after the VSM group mask, the compact slot table and the impostor re-sizing |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** |
| rustdoc individual warnings (cold) | 412 | **413**, with **no warning at a line this wave wrote** — see the ROADMAP block for the per-file audit and why the `+1` is carried rather than claimed away |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | **unchanged** |
| ratchets | — | three, all **down** |

---

## The I4b audit (adversarial, `3c9d87b..4e6e04c`)

**Every mechanism the wave built HELD on re-measurement, and every headline number
reproduced.** What the audit found was one cache that was wrong, one gate that could not
fail, two doors with no arm at all, and seven claims stated more strongly than their own
evidence.

**THE ONE DEFECT THAT WOULD HAVE SHOWN ON SCREEN.** The new scatter-caster cache holds
**render-local** bytes — `pack_fallback` turns world positions into model matrices through
the `FloatingOrigin` — and its key carried the eye bucket, the bands, the caster stamp and the
content fold, but not the origin. `REBASE_DISTANCE` is **1 024 m**; the eye bucket quantizes
the *world* eye onto **8 m**; the two lattices do not line up, so the frame a rebase fires in
is overwhelmingly one where the bucket did not tick. The whole-pack key noticed the rebase,
re-uploaded — and re-uploaded a merge of the **stale** half. Every scatter shadow caster a
kilometre out of place until the camera left its bucket, which on a 50 km² island is every
1 024 m of travel. The arm asserts the bucket is *identical* across the rebase, so it cannot
pass on the case the defect does not occur in.

**AND THE GATE THAT WAS MEANT TO CATCH THAT CLASS.** The incremental query tree's equivalence
gate — "540 answers, 360 of them hits, all identical" — stepped **one** world and forced a
rebuild between the two halves of every iteration, so its "incremental" tree was the control's
tree from one step earlier: centimetres of drift against a body's own half-metre leaf.
Measured: deleting the marking **entirely** left it green on all 540. It is two worlds now,
the incremental one never rebuilt across 180 steps (**720 answers, 540 hits**), and its fourth
question is a **point query at the moving body's own centre** — because a leaf decides only
what the narrow phase is *offered*, and a 40 m ray reaches a stale leaf as happily as a fresh
one. Two doors of the same marking had no arm at all: `set_body_translation` /
`set_body_rotation` — where `set_body_pose_if_moved` puts **every static and kinematic pose
write both hosts make** — and `remove_body`.

**Seven claims corrected**, each where it is written down:

| the claim | what it is |
|---|---|
| "without `query_rebuild` the tree would answer with a collider that no longer exists" | `BroadPhaseBvh` keys a leaf by the **raw index**, generation discarded, and `QueryPipeline` resolves through `get_unknown_gen` — a stale leaf is a wasted traversal. The flag buys an invariant **unfalsifiable through this type's surface**, and deleting it kills no arm |
| "no gameplay could ever read [a fixed-fixed solid manifold] as an overlap" | `ActiveEvents::COLLISION_EVENTS` is on every collider and both hosts fire a Blueprint `Collision` event for a **non-sensor** pair too: two overlapping static solids fired one and no longer do |
| "the pipelined model IS the player" | the player presents with `PresentMode::AutoVsync`, so its cadence is `max(CPU, GPU, **the refresh interval**)`. The estimate is a **lower bound on frame time and an upper bound on fps** |
| "the CPU segments tile the record phase" | they tile the **marked span**. Measured: the per-pass record column sums to **0.92–0.98 ms of a 2.93–3.26 ms `render (record)` stage** — **two thirds of it is outside**, in the setup before `FrameTimer::begin` and the finish/submit after the last mark. Printed every run now |
| "[the content key] costs the same order as the pack whose guess it replaces" | on a **miss**. The cache **hit** went `O(1)` → `O(scene.instances)` — bounded by `sync` only running with shadows on, and by the city's casters all being scatter |
| the arm "extracts `render`'s body **and the windowed loop's own frame block**" | it extracted one. `PlayerApp::frame` is the second scope, and the `poll(` ban is over both **modules**, because a one-line helper defeats a substring ban inside a scope (the P23 lesson, mutation-confirmed) |
| "the phases tile the step" at a **10 %** tolerance | the residue measures **0.000** — the two numbers print equal to three decimals. A tenth of a 1.25 ms step is more than nineteen of the twenty-two phases put together. **2 %** now, with the residue printed |

**The audit's own three release runs** (RTX 4070 Ti, MIN of rounds, at the head it certifies):

| | run 1 | run 2 | run 3 | the range |
|---|---|---|---|---|
| the fixed step | 1.250 | — | — | **1.250** (rounds 1.250–1.258) against a 6.0 ms ceiling |
| 1080p unlit p50 / p95 | 11.177 / 13.646 | 15.259 / 19.598 | 11.309 / 16.349 | **11.2–15.3 / 13.6–19.6** |
| 1080p unlit GPU frame | 2.469 | 4.758 | 2.828 | 2.5–4.8 |
| 1440p unlit p50 / p95 | 18.276 / 19.883 | 18.483 / 21.541 | 18.700 / 26.052 | **18.3–18.7 / 19.9–26.1** |
| 1080p **lit** p50 / p95 | 32.832 / 38.904 | 33.231 / 39.242 | 33.069 / 41.287 | **32.8–33.2 / 38.9–41.3** |
| 1080p lit GPU frame | 16.012 | 16.353 | 16.445 | **16.0–16.4** |
| 1080p lit **pipelined estimate** | 16.327 | 16.528 | 16.697 | **16.3–16.7 (59.9–61.2 fps)** |
| distance from 60 fps, 1080p unlit | p50 −5.42, p95 −2.95 | p50 −1.34, **p95 +3.00** | p50 −5.29, p95 −0.25 | **p50 −5.4…−1.3, p95 −3.0…+3.0** |

Every one of the wave's ranges contains the audit's, and the **step's own table reproduces
phase for phase**: `physics3d sync` 0.796 (63.7 %), `solver` **0.322**, `character move`
0.062, `camera` **0.003**, everything else under 0.02 — and the wall clock and the sum of the
phases both print **1.250 ms**. On the instrument's real scene the physics world holds **7 443
bodies, 7 346 admitted structure colliders, 20 806 contact pairs tracked and 0 touching**,
which is the `FIXED_FIXED` claim measured where it matters rather than on a fixture. The
VSM numbers reproduce exactly — **387 draws and 8 416 skipped per rastering frame**, **206 399
invalidation touches per frame**, `vsm-raster` record **6.021 ms**, `shadow` record
**0.153 ms** — and so does the impostor: **26 792 px, 9.2× the mesh's 2 903, 91.6 % of it
moving**, with the geometry reading the refusal rests on **unmoved at 63 of 2 903 px, worst
channel 18/255**, over a **zero** noise floor. *(One arithmetic aside: "exactly `0.866 × 30`
against `0.5 × |(20, 30, 7.4)|`, squared" predicts **1.99×** and the pixels measure **2.09×**,
because `silhouette` counts the **union of ~425 impostor cards** — the parts' and the shell's
— and not the shell's card alone. The direction and the magnitude stand; "exactly" does not.)*

**Twenty-one mutations were run** — six of the implementer's and fifteen new — and after the
repairs each dies at exactly the arm that names it and at no other. Three are recorded as
**coverage bounds** rather than defects, in `docs/ROADMAP.md`'s audit block: the two
`active_bodies()` extends are armed as a *pair* and not individually, `query_rebuild` on the
removal paths is unfalsifiable through the type's own surface, and the incremental refit's
crossover on a mostly-awake world is unmeasured. One hardening was **refused and priced** (a
quote-parity reader for the eaten-continuation sweep: thirteen of its fourteen accumulated
exceptions are runs genuinely *inside* a literal, which parity cannot help).

### Counts, after the audit

| | after I4b | **after the I4b audit** |
|---|---|---|
| battery blocks / passed / failed / ignored | 306 / 5 704 / 0 / 14 | **306 / 5 710 / 0 / 14** — the audit adds **exactly six** arms and **no test binary**: two in `passes::shadow`, two in `step_cost_3d`, one in `d2`, one in `step_profile` |
| frontend tests / files | 702 / 78 | **702 / 78**, `tsc` and `eslint` clean — no UI was touched by either |
| goldens | 54, byte-identical under `INF_GOLDEN_STRICT=1` | **54, byte-identical under `INF_GOLDEN_STRICT=1`** (100 arms), re-run on the audit's head after the caster-key fix and the VSM group-id change; **not one golden byte moved across the whole range** `3c9d87b..` this tree |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** (local toolchain 1.97) |
| rustdoc individual warnings (cold, all 41 roots touched) | 413 | **413** — 447 `^warning` lines − 34 summaries, cross-checked against the sum of the summaries' own counts. The audit found and fixed the **one** warning it had itself added (a public item linking a private one) and adds none; the wave's `+1` over I4's 412 stays unattributed, and the per-crate distribution is recorded so the next measure can localize it |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | **unchanged — not one schema constant moved across the wave OR the audit**, checked as a diff over every `.rs` in the range rather than assumed |
| ratchets | three, all down | unchanged — the audit minted none and raised none |
| committed samples | 20 levels | **20**, byte-unmoved |

## What is still open after I4b

1. **The lit frame is at the line on the PIPELINED estimate and not on the serialized one**
   (16.4–16.9 against 32.8–33.4). Both halves of the pipelined number sit at ~16.5, so it is
   genuinely balanced rather than bound by one side. Closing it honestly needs the
   **present-to-present harness**, which needs a window; the player's own pipelining is now
   armed — at **both** scopes since the I4b audit, `PlayerRenderHost::render` and
   `PlayerApp::frame` — so what is missing is the measurement, not the mechanism.
   **→ IP, and it is IP's first *measurement* clause**: until a windowed harness exists, every
   "60 fps lit" sentence in this repository is an **estimate**, `max(CPU without the wait or
   the stopwatch, GPU frame)`, and must be written as one. The serialized number is what the
   instrument asserts.
   **And the estimate is a LOWER bound on the player's frame time** (I4b audit):
   `SurfaceChain::new` sets `PresentMode::AutoVsync`, so the player's presented cadence has a
   third term — `max(CPU, GPU, the refresh interval)` — and what makes `acquire` block is the
   *display* as much as the GPU. An engine that computes a frame in 16.5 ms presents at 60 Hz
   on a 60 Hz panel whatever else is true. The same fact constrains the owed harness: a
   windowed one measuring `AutoVsync` measures the **panel**, so it has to configure
   `Immediate` or `Mailbox` to measure the engine — and then say which it measured.
2. **`vsm-raster` still records 6.05 ms**, and the residue is measured rather than guessed:
   **206 399 invalidation touches per frame over 11 047 casters** — 19 page-cells each, because
   `scatter_caster_stamps` is `casters × levels × pages each covers` and the city's buildings
   all cast into a five-level clipmap. Caching the caster pack itself (the way the shadow node's
   now is) is the next move; it needs a content key over four heterogeneous caster sources.
   **→ IP.** Two conditions the I4b audit attaches to it, both learned from the shadow node's
   cache: the key must carry the **floating origin** (A1 — `pack_casters` writes render-local
   matrices exactly as `pack_fallback` does), and it must be pinned **field by field** against
   what each of the four sources reads (A9), because a caster pack that caches on an
   under-specified key is a shadow that stops moving.
3. **When VSM is bound the receivers ignore the cascades entirely** (`env_lighting.wgsl`: *"VSM
   replaces the cascades rather than adding to them"*), so a lit frame with both on rasters
   three cascades nobody reads — 0.23 ms of GPU and 0.16 of record at this scene's scale, found
   and deliberately not taken.
4. **The unlit GPU frame's 14.4–19.8 → 2.2–6.0 is partly a power state, not the engine.** I4's
   frame left the GPU idle two thirds of every frame and an idle card downclocks. A GPU column
   is comparable only between runs whose CPU frames are comparable, and the instrument's header
   now says so.
5. **The projection is now the dearest CPU stage of an unlit frame** (3.9–5.0 ms of ~13.7), and
   `render (record)` the second (3.0). Neither was touched; both are named with their numbers.

## What IP inherits (the performance wave), in the order the numbers imply

Routed here by the I4 audit. Items 2, 3 and 5 are **closed by I4b** (see above); the rest
stand, with I4b's amendments applied in place.

1. **The cook does not evaluate PCG volumes.** The one thing between the engine and a city of
   real geometry, and the only item on this list that is a *feature* rather than a budget.
   `inf-packager` + `inf-dcc` is a one-line Cargo change; what is missing is an evaluated
   population at cook time to bake from. Everything about the shape of the fix is above.

   **I4b did NOT take it, deliberately, and the reason is a size rather than a preference.**
   The Cargo edge and a cook-time evaluation really are near-one-line. What is not is the half
   after it: the projector still emits **placeholder cubes** for every PCG structure, and
   turning that into real geometry means a runtime vgeom asset per archetype — the standing
   P19 `kind_index → real mesh` gap, which the I4 ledger already calls "a project" and which
   `gisbuild` is the only existing path toward. A performance wave that spent its remaining
   budget starting a geometry project would have shipped neither.

   What the wave *did* change about it is the arithmetic: real geometry costs **+0.32 ms**
   against a comparable configuration (the I4 audit's corrected figure), where the shipped
   1080p p95 was **28.5 ms over** the 60 fps budget and is now within a few milliseconds of it
   either side (see the audit's own re-measurement below — the headroom is a *range* and one
   end of it is negative). **The headroom to spend on it exists on a good run**, which it did
   not when the number was quoted.

   **THE HALF THAT IS THE PROJECT, ROUTED BY NAME** (I4b audit). The Cargo edge and a cook-time
   evaluation are near-one-line; the half after is the **P19 `kind_index → real mesh` gap** —
   a runtime vgeom asset per archetype, one door, so cook, PIE payload and the editor's
   Simulate all resolve an archetype to the same geometry (the P22 "one door for three paths"
   law, which is exactly the shape this will otherwise grow three of). `gisbuild` is the only
   existing path toward it and `inf_dcc::bake` is the only existing archetype→mesh function.
   It is a *phase-sized* item and it belongs at the head of IP as a named project rather than
   as a clause of the cook item it blocks. Nothing in the tree asserts it today; the first
   thing it needs is an arm that says a shipped city draws **zero placeholder batches**, which
   the I4 audit deleted as a fixture tautology and which has to come back as a measurement
   over a real cooked pack.
2. ~~**The sim fixed step has no §8 budget and costs 13.0–14.9 ms on the city.**~~ **CLOSED by
   I4b.** `CITY_STEP_BUDGET_MS` exists, the step is broken down by
   `inf_player::step_profile` into 22 phases that tile it, and it costs **1.222 ms**. The
   "2.2 ms is the I3 collider band" line was an *inference* and is retired: the band's gather
   is 0.93 ms, and 9.2 of the 12.7 was rapier computing static-versus-static manifolds.
3. ~~**The lighting stack costs +42.9 ms of p95 and is in no ceiling.**~~ **Mostly closed by
   I4b**: lit p95 **92.3–92.9 → 38.1–41.8**, and the pipelined estimate is 16.4–16.9 ms.
   What stands is the second half of the sentence — *decide what a shipped island level
   authors, then mint a ceiling over THAT configuration*. The lit numbers are still reported
   and never asserted, because every ceiling in the file is set from the shipped default.
4. **IB-9's ceiling is a gate-scene constant at island scale** — 116.91 MiB derived and
   63.0 MiB measured against 16 MiB. `phase16_gate` arm (e2) holds the gap open. *Untouched by
   I4b.*
5. ~~**The scatter impostor is sized from a bounding SPHERE**~~ **— CLOSED by I4b**, with both
   halves measured: the card is the instance's own bounding sphere per primitive kind, the
   ratio fell **19.2× → 9.2×**, `structure_lod_pop` went red as designed and clause 6's refusal
   was re-taken and kept — and the frame **did not move**, which is recorded rather than hoped
   for. What is left is what an impostor is: a bounding-sphere card is intrinsically about twice
   a box's silhouette, and closing *that* means an oriented card or a real impostor atlas.
6. **`resident_bytes` counts heights only** — not `maps`, `biomes` or `holes` — so an eroded,
   painted or carved terrain under-reports against both the ceiling and the bound, which are
   now derived from the same under-count.
7. **A streamed cell evaluates its `PcgVolume` and not its biome bindings.**
8. **The instrument's camera covers 120 m of a 1 260 m city**, and its scene has **zero
   virtual textures**. Both bound what the headline number is a number *about*; a textured,
   longer-path fixture is what would make the VT column and the p99 mean something.

---

## Done — wave I5 (the player core)

**THE ENGINE HAD NO UI.** Before this wave `inf-render-2d` was a *world*-quad
batcher going through the game camera, depth-tested and tonemapped with the scene; the
shipped player had no HUD, no menu, no toast and no focus model; and a shipped game had no
per-user settings file at all (`player.toml` is a boot config nothing writes at runtime and
`input.toml` is a best-effort read a cooked pack never even looks at). A game with no
settings dialog is a game a player cannot change the resolution of, and one with no
rebinding screen is a game a left-handed player cannot play.

### The shipped binding table

| control | action / axis |
|---|---|
| W / S / A / D (and the arrows) | `move_y` / `move_x` |
| mouse | `look_x` / `look_y` |
| **Tab** | `menu` — the in-game settings dialog |
| **Shift** | `sprint` |
| **Ctrl** | `walk` (the gait default is **RUN**; this is the *slow* modifier) |
| **E** | `interact` |
| **R** | `reload` |
| **C** | `crouch` — **click** crouches or slides, a **long press** goes prone or dives |
| **Space** | `jump`; `move_up` while swimming or flying; `handbrake` while driving |
| **LMB** | `attack` |
| **RMB** | `aim` → `RotationMode::Aiming` |
| **wheel** | `weapon_switch` |
| **I** | `inventory` |
| X · Z · F · V | `prone` · `roll` · `dive` · `fly`, the direct controls the table above folds |

`reload`, `attack`, `inventory` and `weapon_switch` are bound against consumers that arrive
with the weapons and inventory work, and the shipped player **says so** when one is pressed
(`inf_ui::Toasts`, reading `inf_input::actions::NOT_YET_CONSUMED` rather than a second
list). A dead key is indistinguishable from a broken one; a toast is the difference.

**Two bindings moved for a reason rather than for the table.** `KeyW`/`ArrowUp` were bound
to `jump` for the 2D platformer and the 3D intent reads the same name, so **every step
forward on a character was also a jump** — and no scripted trace could have seen it, because
every one of them presses action NAMES. `move_up` was on Q/E, so a swimmer who pressed E to
open a hatch also rose; it is Space/Ctrl now, which is what every swimmer and pilot alive
already expects.

### The C key's four verbs, and where the duration is measured

The discrimination is at **intent** level and the duration is measured on the **sim's fixed
step**. `inf_input::HoldClock` is one accumulator with two instances — the wall clock for
the in-game UI (which runs while the sim is frozen behind a menu) and each sim's fixed step
for the intent — and `inf_ecs::movement::classify_press` is the one rule. A duration
measured in fixed steps is a function of the simulation rather than of the frame rate, so
PIE stays byte-identical to shipping and the edge that crosses into the world is the same
discrete set it has always been.

The matrix, measured end to end through the shipped keys in `player_core_gate`:

| | standing | sprinting |
|---|---|---|
| **click** (release under 250 ms) | crouch | **slide** — 13 steps, 0 dive launches |
| **long press** (held past 250 ms) | prone | **dive** — exactly 1 launch, 0 slide steps |

**The click fires on the RELEASE**, because nothing can know a press is short until it ends.
The cost is bounded by the threshold and is stated on the door: a crouch lands on the frame
the key comes up, at most 250 ms after it went down. A design that fired the crouch on the
press and "upgraded" to prone would do both.

### Run by default, and both sprint gates

Run was already the default tier (`Gait::default()` is `Run` and `desired_gait` falls
through to it); what moved is the *binding* and the *proof*. Measured through the keys, on
the cooked pack: **nothing held 3.750 m/s** (run is 3.750) · **Ctrl 1.783** (walk is 1.650)
· **Shift 6.500** (sprint is 6.500).

Two rulings, both refusals as **values**:

* **A dive needs a sprint**, exactly as a slide does. It was never coherent that one was
  gated and the other free — a dive from a standing start is a belly-flop, and
  `dive_speed_mps` is a *launch* speed that assumes a body already moving. Folded into
  `request_mode`'s condition, so the answer is `ConditionNotMet` and the character does
  whatever it was going to do instead.
* **The slide's own refusal is recorded.** A player holding sprint and pressing crouch has
  asked for a slide; if the body is too slow the crouch below is what it does *instead*, and
  until this wave the two outcomes were indistinguishable — the one entry condition in the
  catalogue whose refusal nothing downstream could ever read.

**Space is a dive when there is water to dive into.** One pure rule
(`inf_ecs::movement::dive_into_water`) fed by a new *place* query
(`PhysicsBridge3D::water_surface_at`), and a dive that reaches the water carries the P29.3
deliberate-dive door for its entry step. Three cases measured: water at a reach while
sprinting launches at `dive_speed_mps` **(0, 2.3365, 5.5)**; the same jump *walking* and the
same sprinting jump on *dry land* both launch at the jump's **(0, 4.3365, 0)**.

### The UI layer

`crates/inf-ui` (Ring 0, **no new external crate** — glam, serde, toml, inf-render-2d,
inf-input, inf-asset, every one already pinned): screen-space rects and text in virtual
pixels; `GameSettings` under the editor's own doctrine; the rebinding table's rows; the
dialog's state, reducer and projection; the toasts. Everything is a pure function of its
inputs, which is what lets a settings dialog exist in a shipped player at all in a
repository whose CI has no GPU.

`inf_render::passes::ui` draws it **after the tonemap**, in pixels, with `LoadOp::Load`,
nearest-filtered and with no depth. The sprite pass draws into the HDR MSAA target before
resolve/TAA/bloom/tonemap, so a menu drawn through it would be bloomed, temporally
reprojected and colour-graded with the world behind it, and depth-tested against the wall
the player is standing next to.

**The goldens are unmoved, and it is measured on both sides**: with an empty list the frame
is byte-identical, and with one quad in it the frame differs *and the corner outside the
quad does not*. All 101 golden arms pass under `INF_GOLDEN_STRICT` with no PNG rewritten.

### The interaction core

`inf_ecs::interact` is one rule: an `Interactable` (verb, label, range, enabled, view cone),
a flat candidate whatever produced it, and a pure `resolve` that answers with the nearest one
in range and in view, ties broken by `Guid` over a sorted walk with a strict `<`.
`Interactable` is a **runtime** component with no scene slot — the shape `MovementRuntime`
and P22's `DeformField` already have — so no schema moved.

P29.7's `vehicle::try_enter` is now a call into `inf_physics::d3::interact` rather than a
second implementation of "nearest thing in reach". **Nothing about its semantics moved**:
the same `ENTER_REACH_M`, the same `Guid`-ordered walk with a strict `<`, the same
occupied-seat skip, and still **no view test** (the seat candidate carries
`NO_VIEW_TEST_DEG`). The seat also keeps its *pose* source — migrating it to the ECS
transform would have moved it by one fixed step, which is a semantic change to a shipped
gate. All 21 P29.7 vehicle arms stay green, and a 22nd says the migration is real: an
authored `Interactable` half a metre away out-ranks the seat, pressing E with it nearest does
**not** put the character in the driving seat, and disabling it makes the same press enter
the car again.

### The gate

`runtime/inf-player/tests/player_core_gate.rs`. Every other scripted replay in this tree
presses **action names** and is blind to the input layer; this one presses **keys**, through
the shipped table, the shipped `InputState`, the shipped dialog and the shipped hold clock,
in `PlayerApp::frame`'s own order — on a cooked pack and on a PIE payload, **byte for byte
over 1 600 steps**.

* the gait ladder above;
* the C-key matrix above;
* **the menu stood open for 137 frames and cost 0 fixed steps**, with the trace record
  byte-identical across every one of them;
* rebinding Sprint from Shift to B *inside the dialog*: B + W reaches **6.500** and Shift + W
  settles at **3.750** — the control that says the claim is about the rebinding and not
  about W;
* no control was live while the menu was open, read off the **resolved input** rather than
  off the world (a paused sim does not integrate, so the pause arm could not have caught a
  leak).

### Counts

| | after the I4b audit | **after I5** |
|---|---|---|
| battery blocks / passed / failed / ignored | 306 / 5 710 / 0 / 14 | **309 / 5 788 / 0 / 14** — three new test binaries (`inf-ui`'s lib and doctest roots, `player_core_gate`) and **78** new arms |
| frontend tests / files | 702 / 78 | **702 / 78**, `tsc` and `eslint` clean |
| goldens | 54, byte-identical under `INF_GOLDEN_STRICT=1` | **54, byte-identical** (101 arms), re-run after the UI node landed |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** (local toolchain 1.97) |
| rustdoc individual warnings (ceiling 450) | 413 | **413** — 447 `^warning` lines − 34 summaries, **unmoved**. The one this wave added (a `[`Interactable`]` link from `inf-physics`, which cannot name `inf-ecs`'s item) was found and fixed |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | **unchanged — no schema moved** |
| committed samples | 20 levels | **20** — `samples/phase29-locomotion/input.toml` regenerated through its generator (see below), no level moved |

**The one committed byte that moved**, through the generator and with the arithmetic:
`samples/phase29-locomotion/input.toml`, **2 072 → 2 423 bytes, +33 / −9 lines**. The plus
side is eight new action-source blocks (aim/LeftTrigger, attack ×2, inventory ×2, menu ×2,
reload) at three lines each = 24, plus the six-line `weapon_switch` axis, plus the four
in-place value changes' new halves = 33. The minus side is `jump`'s two removed keys (five
lines with their headers) and the four in-place changes' old halves = 9. Every line of the
delta is a row of the table above.

## Decisions (I5's, binding on later waves)

* **Tab pauses the single-player simulation, and the pause is SIM state.** Two reasons and
  the second is the one that matters: a menu that did not pause would make the UI part of the
  simulation's input, so the frames a player spends reading a table would be frames the sim
  advanced and a trace that opened the menu would depend on how long they took. `sim_paused`
  lives on `RuntimeSim`/`SimSession` rather than on the host, because a host-level pause is
  invisible to `step_once` — the door every trace in this repository is scripted through. A
  multiplayer session cannot stop a shared world and does not: the host asks
  `MenuState::pauses_sim`, which answers for the session it is given.
* **A press duration is measured on the SIM clock and a UI's on the WALL clock, and they are
  two instances of one implementation.** A gameplay fact measured in wall seconds is one the
  frame rate can change; a menu's key repeat measured in sim seconds stops when the menu
  pauses the sim. `inf_input::HoldClock` is the accumulator and
  `inf_ecs::movement::classify_press` is the rule, and each host runs one instance per clock
  it owns. **Two instances of one implementation is not the two-copies defect; two
  implementations would be.**
* **A RELEASE is never consumed, even by a modal dialog.** Measured: consuming them stranded
  the input state holding the very key that opened the menu — Tab went down, the dialog
  opened, the release was eaten, and `menu` read as *held* for the whole window. It is the
  stuck-key failure `InputState::release_all` exists for, reached through the menu instead of
  through a focus loss. Forwarding a release is safe by construction: the press never reached
  the state, so there is no `just_released` edge for it to fire.
* **An open dialog is MODAL — "consumed" is about who the input belongs to, not about whether
  anything happened.** The reducer shipped for one afternoon with a `_ => consumed = false`
  fallthrough, which reads as "the dialog only claims what it uses" and means the player jumps
  while reading the video page and fires a weapon at whatever is behind the menu.
* **A binding ROW is not an action.** UE lists *Move Forward / Back / Left / Right* as four
  rows and this engine's `move_y` is one axis with a `+1` key and a `−1` key; a table built on
  actions alone could not offer W, A, S or D at all. A row is an action **or one sign of one
  axis**, it has a stable id, and the id is what a settings file stores.
* **A row can have SEVERAL keys, and a conflict must read all of them.** Move Forward is W
  *and* Up in the shipped table. A conflict check that read the displayed cell would let a
  player bind Up to something else and leave two verbs on one key; a swap that cleared the row
  would take the other key away too. Both read and write the **exact token** the dialog named.
* **Only the DIFFERENCE from the shipped table is stored** — the editor keybinding door's
  rule, and it earns its keep the same way: a file that shipped the whole table would freeze a
  player's bindings at the build they first ran, so a control added later would arrive unbound
  and a default corrected later would never reach them. An **empty value means "unbound"**,
  which is the difference between a row a player cleared and one they never touched.
* **Escape and Enter cannot be bound away.** A player who bound Escape to *Attack* would have
  no way to leave a capture. Refused as a value, with the capture left open so a real key
  still works.
* **A shipped game's settings file obeys the editor's doctrine, restated where a player can
  reach it**: absent is the default, corrupt is an **error with the bytes left untouched**,
  newer is **refused**, every write is atomic, and every numeric is guarded — non-finite takes
  the **default** (an infinity is not "very sensitive"), finite **clamps**. And a *preferences*
  file is not a boot config: a corrupt one leaves the player on defaults and **says so**,
  where `player.toml` refuses to boot.
* **The editor's preferences panel is a second SURFACE, not a second rule.**
  `inf_ui::bindings::table` and `apply_row` are what both it and the in-game capture call.
  There is deliberately no binding table in TypeScript: the last copy of one across that
  boundary knew about three of seventeen entries.
* **`EditorSettings.game_bindings` is a different map from `keybindings`.** One is the
  editor's chords (Ctrl+Shift+P) and the other is the game's controls (C). Folding them
  together would put a chord and a key code in one namespace and make "unbind" mean two
  things.
* **A station is at a place, not at a time** — phase 29's own lesson, met from the other side.
  A time-scripted run walks into the committed course's low roof at `z = 11` and
  crouch-shuffles the rest of the way at 1.5 m/s, with the dive station spending all ninety of
  its steps under a ceiling. The clear ground is twenty-one metres and a sprint station is
  nineteen; the script turns the character round at its edge.
* **A station that ACCELERATES into its tier is measured by its peak; one that DECELERATES
  into a slower tier by where it settled.** A peak read **3.47 m/s** for the walk, which is
  the run speed the character entered carrying.
* **A UI quad is not rebased against the floating origin.** The sprite pass's pack calls
  `origin.to_render` because a sprite is in the world; a UI quad is in pixels, and rebasing
  one would move the menu every time the camera crossed a rebase boundary — every kilometre
  of travel on a 50 km island.

## What I5 found in other people's code

Both pre-existing, both in the **ragdoll**, and both fixed with arms that die under the
mutation that names them:

1. **A ragdoll whose hips end inside the floor put the character's FEET a whole body below
   the pelvis** — so it came out of the ragdoll underground, its collider was switched back on
   inside the terrain, and it fell out of the world. Measured on the phase-29 course at
   **y = −132 m and still falling**, from hips that ended 0.8 m under the ground. The line
   above the placement already said *"a pelvis in the floor is on it"*; the placement did not
   agree with it.
2. **Nothing bounded a ragdoll limb's SPEED.** `gravity_enabled` bounds an *acceleration*; an
   articulated body seeded in a pose that violates its joint limits is fed energy by the solver
   every step until the numbers leave the world. Measured: the committed course settles in 46
   steps, and the **same ragdoll entered 2.7 cm further along the same fall**, at a
   bit-identical handoff velocity of `(0, −10.706, 3.750)`, reaches **z = −3.85e13**. The
   jitter is in **both** — the pelvis moves a metre a step from the first step of the committed
   run too — so **the 46-step settle was luck rather than stability**.

   The ceiling is a **BOUND, not a cure**. The instability is upstream in the joint seeding and
   is carried below. What the bound buys is that a simulation which cannot be trusted to settle
   can still be trusted to stay in the level.

   *Two things were ruled out on the way and are worth writing down:* forcing a full query-BVH
   rebuild every step (I4b's incremental marking, which a CI red was being blamed on) changes
   the outcome **not at all**, and the entry pose and handoff velocity are **identical** between
   the run that settles and the run that diverges.

**And the phase-29 course's ragdoll station was luck-dependent.** Its `held > 90` jump-out was
**dead code** in the committed script — the ragdoll always settled at 46 — and the first script
that reached it drove the character to `z = −3.85e13`. A gate whose station passes because the
simulation happened to settle is a gate that has not been falsified.

## The I5 audit (adversarial, `60d6f1d..` this tree)

**Every headline measurement HELD on re-measurement**, from the tests that print them —
`3.750 / 1.783 / 6.500` m/s through the shipped keys; the C matrix at *slide 13 steps, 0
dive launches* against *1 dive launch, 0 slide steps*; the menu at **137 frames, 0 fixed
steps**; the rebinding at B + W **6.500** and Shift + W **3.750**; the dive launch
`(0, 2.3365, 5.5)` against the dry-land jump `(0, 4.3365, 0)`; the ragdoll's *pelvis
y = −0.321 → placed 0.400* (the old rule: −0.321) and its limb ceiling *4 000 m/s in,
8.449 out, against 40.0*; the controls page's 0.15 → 0.225 deg/count, the invert, the
0.25 → 0.30 s threshold and the three buses at 0.8. `phase29_gate` is **16 of 16** with two
probes ignored, `player_core_gate` **6 of 6**, `movement_parity` and `character_demo` green.
The committed `input.toml` arithmetic recomputes **exactly**: 2 072 → 2 423 bytes,
`--numstat` +33 / −9, 178 → 202 lines, and the generator lock is a whole-content compare
against `toml::to_string_pretty(&default_map())` rather than a byte count.

**The merged tree was checked first**, because the wave's physics edits and the
`bridge_sync` hotfix's scratch buffers had never run together: `bridge_sync_scaling` 4/4 and
`step_cost_3d` 8/8, then the whole of `inf-physics` green.

**Five defects, one arm that could not fail, and one ledger number that was one low:**

| finding | what shipped | now |
|---|---|---|
| **A1 · a player could permanently lock themselves out of the settings dialog** | the *Menu* row is a table row like any other: a left arrow on it **cleared** `Tab`, and a capture on any other row could press `Tab`, be told the menu owned it, press Enter and **take** it. Either writes a settings file that outlives the process, so the screen that would undo it is the thing that was unbound. `RESERVED_KEYS`' own doc claimed this was already refused — *"one who bound the menu key to a game verb would have no way back into the settings that let them undo it"* — and only the Escape/Enter half was built. The **editor's** preferences panel reached the same edit through `apply_row`, which would ship a project *nobody* can open the settings of | `bindings::guarded` — one rule, both surfaces, applying an edit and taking the **whole** edit back if it left `menu_is_unreachable`. Plus `restore_menu_if_unreachable` at the boot door for the case an edit door cannot cover: a file, which outlives the build that wrote it |
| **A2 · the editor's pause was a half-built mirror with no arm at all** | `SimSession::sim_paused` shipped as a declared MIRROR of `RuntimeSim`'s. Deleting the check in `step_once` left **every test in the tree green**, and `tick` — the *other* door, the one a host drives by elapsed time — never had the check at all, so a paused Simulate would have advanced *and* banked the frames to spend in one burst on resume. (The same mutation on the player's `step_once` reds **four** arms.) The I4b law: a mirror needs its own arm | the check in `tick`, mirroring `run_frame`'s accumulate-nothing, and `a_paused_session_runs_no_fixed_step_through_either_door` — both doors, the no-banking claim, and a control at each |
| **A3 · the merged candidate walk's `Guid` sort had no falsifier** | `d3::interact::candidates` concatenates two already-sorted lists and sorts the union, and its own comment says why ("a seat and an item at exactly the same distance must resolve the same way in both hosts"). **Deleting the sort killed nothing**: every other arm puts its candidates at different distances, where the tie-break never runs | `a_seat_and_an_item_at_the_same_distance_break_by_guid` puts the item **on the seat** — an exact tie, asserted exact — with a lower `Guid` than the chassis, so "lowest guid wins" and "whichever list came first wins" give different answers |
| **A4 · an arm whose failure message named a defect it could not see** | `the_editors_surface_reports_a_conflict_and_swaps_only_when_asked` swaps `KeyW`, which is Move Forward's **first** key — and "remove the exact token" and "remove whichever desk source is first" are the same expression there. Its message reads *"the swap took Move Forward's other key too"* | the same arm now also swaps `ArrowUp`, the **second** key, where the two rules differ. *(The property was not unheld: `menu`'s own `a_capture_can_bind_the_keys_the_dialog_navigates_with` swaps a second key and dies under the mutation. What was missing was the editor surface's own.)* |
| **A5 · the video page's three dead sliders said nothing to the player** | window mode, resolution and quality are stored and applied by nothing — and until this audit they said so **only in a ledger**, which is not where a player reads. This wave's whole subject is that *a dead key is indistinguishable from a broken one*: it put "not wired to anything yet" on every unwired binding ROW and raised a toast when one is pressed, and then left the three unwired *settings* silent. A player who changes the resolution and sees nothing happen cannot tell a preference from a bug | `STORED_NOT_APPLIED_NOTE` on the three rows, through the same `note` field `not_yet_note` uses. It does **not** promise a restart, because nothing reads these three at boot either — measured: they have no reader outside the dialog and the settings module. The arm counts them: exactly three noted, twenty-plus live, so the day the video half lands it fails and the note comes off |
| **A6 · two doc comments asserted an invariant that is false** | `commands/sim.rs` said Simulate maps keys through "the SAME table the shipped player reads" and that "an `input.toml` beside the level would change both". Measured: that file reaches **one** path — `inf player <level.inf_lvl>` — and neither a cooked build nor Simulate. So a project's own bindings are lost by two of the three paths, which is this wave's own "settings only one host reads" trap a layer down, and the doc sent the next reader at a mechanism that does not connect | both lines say what is true, with the gap and its fix (**move `load_map_beside` into `inf-input`**) carried by name in *What is still open*. Not closed here: moving a loader between rings is a change to a path this audit does not own |
| **A7 · the battery count was one low** (a number, not a defect) | the wave records **5 788** passed. Measured at this head: **5 796**, and the audit adds exactly **seven** arms and no doctest — `git grep -c '#\[test\]'` is `5 801 → 5 808` across the two trees. So the wave's own head was **5 789** | the I3 audit's law met a second time: *run the count at the head you are about to write down.* The figure below is this tree's |

**Fifteen mutations, four of the implementer's and eleven new**, each run to the point of
naming which arm dies:

| mutation | dies at |
|---|---|
| the ragdoll's hips-inside-floor placement reverts to `t.y − (half + radius)` | `a_ragdoll_that_ends_inside_the_floor_leaves_the_character_on_it`, alone in four binaries |
| the limb speed ceiling is deleted | `a_ragdoll_limb_cannot_exceed_the_terminal_speed`, alone |
| the dive drops `&& want_sprint` | `a_dive_needs_a_sprint_and_the_refusal_is_a_value`, alone in the crate |
| the slide's recorded refusal is deleted | `a_slide_needs_sprint_speed_and_ends_in_a_crouch`, alone in the crate |
| `interact::resolve`'s strict `<` becomes `<=` | `a_tie_breaks_by_guid_whichever_order_the_walk_arrived_in` |
| `candidates_in_world` stops filtering on `enabled` | the rule's own arm **and** the 22nd vehicle arm — the other **21 P29.7 arms stay green** |
| `classify_press` drops `prev_hold_s < t` (a long press fires every step) | the rule's arm **and** `the_menu_pauses_the_sim_and_the_rebinding_takes_effect` |
| `HoldClock` drops the release-carry | three arms across `inf-input` and the player gate |
| the conflict swap takes the row instead of the exact token | `a_capture_can_bind_the_keys_the_dialog_navigates_with` — the arm that swaps a **second** key |
| the dialog's `_ => {}` becomes `consumed = false` | two arms, in `inf-ui` and in the player's own routing |
| `PlayerUi::key` consumes releases | `the_dialogs_keys_never_reach_the_resolved_input` |
| `RuntimeSim::step_once` ignores `sim_paused` | **four** arms of `player_core_gate` |
| `SimSession::step_once` ignores `sim_paused` | **nothing** — finding A2 |
| `d3::interact::candidates` stops sorting | **nothing** — finding A3 |
| `KeyW` is bound back to `jump` | **six** arms across three crates, including the committed-sample generator lock |

**What the audit re-measured and did not move**: `inf-ui` is Ring-0 clean (no Tauri/winit/wgpu
name in it) and adds **no external crate**; the `ui` render node sits after the tonemap and
before mask/composite, returns before touching the encoder on an empty list, and the goldens
are **54, byte-identical**; the `Interactable` component is runtime-only and no schema moved;
`phase29_gate`'s anti-vacuity list is intact at **14 forced + 4 reserved = 18 = `ALL_MODES`**,
with `reserved_slot` cross-checked variant by variant; `water_surface_at` is the water index's
own `highest_surface_at` at `O(bodies over the cell)` and `O(1)` with no water, on the sim
clock; and the E key has exactly **one** resolution site in the movement step, which both
hosts run — the prompt asks the same function, so what the player is told and what the press
does are one call.

**The ragdoll bound is labelled a bound in all three places** the mandate asked for — the
`step_ragdoll` comment ("This is a **bound, not a cure**"), the ROADMAP block and the ledger
above — and the carried instability is named with its number (**z = −3.85e13** from a 2.7 cm
entry shift) in *What is still open*, below.

### Counts, at the head this audit certifies

| | after I5 (as recorded) | **after the I5 audit** |
|---|---|---|
| battery blocks / passed / failed / ignored | 309 / 5 788 / 0 / 14 | **309 / 5 796 / 0 / 14** — seven new arms, **no new test binary**, and the wave's own figure was one low (finding A7) |
| frontend tests / files | 702 / 78 | **702 / 78**, `tsc` and `eslint` clean |
| goldens | 54, byte-identical under `INF_GOLDEN_STRICT=1` | **54, byte-identical**, re-run under `INF_GOLDEN_STRICT=1` over **101 arms** with no PNG rewritten — and the 54 files are byte-unchanged across the whole range `60d6f1d..` this tree |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** (local toolchain 1.97) |
| rustdoc individual warnings (ceiling 450) | 413 | **413**, unmoved |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | **unchanged — no schema moved** |
| committed samples | 20 levels, `input.toml` regenerated | **unchanged** — the audit moved no committed byte |

## What is still open after I5

1. **The ragdoll is chaotically unstable and only bounded.** A 2.7 cm difference in entry is
   the difference between settling in 46 steps and diverging without bound; the pelvis moves a
   metre a step from the first step of *every* run, including the committed one. The fix is in
   the joint seeding (limits violated at spawn) and it is a physics project, not a wave clause.
   **The bound is in place and the two defects above are armed; the instability is not fixed.**
2. **A dive off flat ground is 4 cm of clearance in a fixed step and the ground snap is
   entitled to take it back**, so `MovementMode::Dive` lasts one step there. How long the mode
   *lasts* is a fact about a dive with somewhere to go, and `phase29_gate`'s station (which
   dives at `z = 36` with the whole floor ahead of it) is what holds it. The player-core gate
   measures the **launch** instead and says so.
3. **The UI's text is the built-in 8 × 8 bitmap font and nothing else.** A user font is an
   `.inf_tex` grid atlas the player has no asset DB in the render thread to resolve — the gap
   `inf_player::render`'s own header records. The dialog is monospace and will stay monospace
   until that closes.
4. **The settings dialog is keyboard-only.** Gamepad navigation is the brief's own "later" and
   the mouse is not wired to it: the rows have hit-test rectangles (`Rect::contains`) and no
   pointer reaches them. The rebinding *capture* does take a mouse button, because a mouse
   button is a bindable source.
5. **A shipped player's settings directory is resolved from environment variables** —
   `%APPDATA%`, `$XDG_CONFIG_HOME`/`$HOME`, with the exe's own directory as the last resort and
   `INF_PLAYER_SETTINGS_DIR` as an override. Those *are* the platform conventions and a crate
   that resolved them would read the same two variables, but a platform crate would also handle
   the cases they do not (a portable install, a sandboxed store build).
6. **The video page's window mode, resolution and quality tier are stored and not applied.**
   They reach the settings file and the dialog reads them back; nothing resizes the window,
   reconfigures the swap chain or re-detects the tier yet, because that is a winit/surface
   change on a path CI cannot run. The three rows are honest about what they are —
   preferences a later wave connects.

   **And the three rows now say so where the player reads them** (I5 audit, A5):
   `inf_ui::menu::STORED_NOT_APPLIED_NOTE`, through the same `note` field an
   unwired binding row uses. It does not promise a restart — measured, nothing reads
   these three at boot either.

   **The other five ARE live**, in the session the player is in, and
   `the_controls_and_audio_pages_reach_their_consumers` is what says so: the look sensitivity
   multiplies the project's authored degrees-per-count in the live map, invert-Y flips the y
   channel's sign, the hold threshold reaches the sim's own copy, and the three mixer buses
   reach `AudioEngine`. A setting that is stored and read by nothing is a dead slider, which
   is the dead-key defect this wave is about, one level up.
7. **`weapon_switch` reaches a consumer as a RATE.** The wheel is a delta source, so
   `axis_snapshot` divides it by the frame time; a consumer must read its sign or integrate it.
   A notch count would need the wheel to be a button, which it is not on any platform this
   engine speaks to.
8. **The in-game dialog draws in the shipped player and in a windowed PIE preview, and not in
   the editor's own Simulate.** The editor's viewport has its own projector
   (`inf-viewport::host`), which this wave did not touch; an author who wants the menu previews
   it with PIE. The editor's *preferences* panel carries the same bindings table, which is what
   the mandate asked for.
9. **The interaction walk is `O(interactables)`, not the collider band's active set** (I5
   audit, noted rather than fixed). The brief named the band as the mechanism; what shipped is
   `try_query_filtered`, which restricts the walk to archetypes carrying an `Interactable` and
   answers `O(1)` when the component was never inserted. The brief's binding constraint —
   *never `O(entities)`* — is met, and the doc states the bound it actually has rather than the
   one that was asked for, so nothing here is claimed that is not true.
   *Why it is a remainder and not a defect:* P16's partition means an unstreamed cell's entities
   are not in the world at all, so the walk is already over the **resident** set, which is the
   band's own spirit at a coarser grain. It becomes a real cost the day one resident cell holds
   thousands of authored interactables, and the fix then is the band — `IB-2a`'s anchors are
   already the right input. The prompt asks the same `resolve` **every frame**, so that is the
   site to measure first.
10. **A project's `input.toml` reaches exactly ONE of the three paths** (I5 audit, A6 — the
    doc was corrected, the gap is carried). `inf_player::input::load_map_beside` is read on
    `WorldChoice::Level` only, i.e. `inf player <level.inf_lvl>`. A **cooked pack** takes
    `default_map()` (`lib.rs`'s own match, and the cook does not carry the file) and the
    editor's **Simulate** takes it too (`commands/sim.rs`). So an author who ships custom
    bindings gets them running a dev level and loses them in Simulate *and* in the build —
    which is the "settings only one host reads" trap this wave exists to close, one layer
    below where it looked. Two doc comments in `commands/sim.rs` asserted the opposite
    ("the SAME table the shipped player reads"; "an `input.toml` beside the level would change
    both") and now say what is true. **The fix is to move `load_map_beside` down into
    `inf-input`** — Ring 0, beside `default_map`, where a project's binding file belongs — and
    have the cook and this seam both use it. Not done here: moving a loader between rings is a
    change to a path this audit does not own.
    *Nothing is currently wrong on disk*: `samples/phase29-locomotion/input.toml` is the only
    committed one and it is byte-identical to `default_map()`, which is why the phase-29 gate's
    editor-versus-player arm passes without noticing.

---

## Done — wave I6 (gameplay systems)

**THE ENGINE HAD NO GAMEPLAY VERBS.** I5 bound the owner's whole control table and
left four of its keys — `reload`, `attack`, `inventory`, `weapon_switch` — bound
against consumers that did not exist, with the shipped player raising an honest
"not wired up yet" toast when one was pressed. There were no doors (a doorway was
a hole in a wall and nothing else), no inventory, no weapons, and **no health
component anywhere in the workspace** — searched exhaustively before writing one:
every `hp` in the tree was a CSV fixture column or a data-asset codegen test.

`inf_input::actions::NOT_YET_CONSUMED` is now **empty**, and that is the wave's
own summary.

### ONE ENERGY DOOR, and what it cost to make it one

A kick, a crash-through, a bullet and a collapsing wall all meet at
`Destructible::bond_energy_j` — `strength × area × CRACK_OPENING_M`, the P22 rule.
Making that true moved one constant: `CRACK_OPENING_M` left
`inf_physics::d3::fracture` for `inf_ecs::components`, beside `Destructible`,
where `bond_force_n`'s own doc already said the *contract* belongs. The old path
is a `pub use`, so nothing that named it moved. `bond_energies` and
`ground_bond_energies` — which each spelled the multiplication out — now call it,
so the expression exists once.

| quantity | how it is derived | value |
|---|---|---|
| a default lock | 300 MPa (structural steel) × 4 cm² bolt × 1 mm crack | **120.000 J** |
| a kick | half of 15 kg times 4.5 m/s squared — a leg and a hip, not a body | **151.875 J** |
| the kick's margin | one kick opens a house door; doubling the lock takes two | **31.875 J** |
| the breach speed gate | above the run (3.75) and below the sprint (6.5) | **5.0 m/s** |
| a sprint's energy | half of 80 kg times 6.5 m/s squared | **1 690 J** |
| a sprint's exit speed | the lock's joules off exactly, then a 0.85 restitution | **5.325 m/s, 81.9 % kept** |
| a rifle round | the muzzle energy of a real one | **1 700 J** |
| a body | what it absorbs before it stops working | **2 000 J** (two rounds) |

**Health is joules, not hit points**, and that is not a style choice:
`docs/memos/p22-strength.md` §1 refuses damage numbers for walls, and a character
is the one place a bullet, a kick, a fall and a collapsing wall all meet — so a
conversion table there would be the same mistake with more consumers.

### The door system

`inf_ecs::door` is the pure half (a leaf on a hinge, the lock's price, the swing,
the breach arithmetic); `inf_physics::d3::door` is where it meets the world (the
leaf's collider, the blocking probe, the door half of I5's candidate list).

* **The state is a sparse bevy resource** (`DoorField`), keyed by `Guid`, and
  **absent means closed**. Three reasons in the module header, and the binding one
  is that a grammar doorway has no entity to put a component on — a design that
  gave authored doors their state on the component and derived doors theirs in a
  map would be two authorities on "is this door open". It is P22's own split
  (`Destructible` authored, `FractureState` mutable) and it moves no schema.
* **The leaf is a SYNTHETIC kinematic body** under `door_leaf_guid`, for authored
  and derived doors alike — the `pcg_structure_guid` pattern with its own salt.
  A door entity's `Transform` is its **hinge** and is never written.
* **The E key keeps I5's one resolution site.** `step_one`'s verb dispatch is a
  `match` now, so a verb added later is a compile error rather than a silent
  decline. `d3::interact::candidates` gained a `feet` argument for one reason: a
  door's prompt says whether the *lock* verb is on offer, and that is a fact about
  which face the character is standing on — deciding it at the press instead
  would be two calls where I5 built one.
* **The kick lands on the animation's notify**, never on the button, with a
  0.35 s fuse for a character that has no rig to notify it. Both paths are armed.

### What the world-level arms found (and they are the interesting part)

Ten door arms and nine weapon arms against a real `PhysicsBridge3D`. Five real
defects, every one of them invisible to the rule-level tests:

| defect | what it did | how it was found |
|---|---|---|
| **the leaf's box axes were transposed** | `leaf_pose` returned `(width, height, thickness)` while the yaw is applied as `from_rotation_y` and yaw zero is `+Z` — so the box's long axis was its thickness. The blocking probe swept a 0.06 m box along the leaf's length and a 0.9 m box across it, and **a solid standing squarely in a door's arc was never hit** | the wedge arm: 0 blocked steps of 90 against a box the leaf passes through |
| **every door was permanently blocked** | a leaf stands ON the floor and BETWEEN two wall boxes, so a sweep of its exact box begins penetrating and parry reports `toi == 0`. Read as "blocked", it meant **no door in the engine could ever open** | the first fixture that had a floor in it: 0 of 90 steps moved |
| **the door field was not sparse** | `step_doors` called `field.entry` for every door every step, materialising an entry for every door in the world on its first step — so a level's trace bytes became a function of how many doors a player had walked past, and no pre-I6 trace would have stayed byte-identical | the arm that asserts the trace is empty after ten steps of a world with a door in it |
| **a corpse got up** | the ragdoll's get-up fires on settle, so a body handed over by the damage system was re-handed on the next step: a corpse twitching upright and flopping, for ever | 2 handoffs in 30 steps where there should be 1 |
| **`Without<Downed>` answered `None`** | `try_query_filtered` refuses when a component it names has never been inserted — which is the `O(1)` fast path everything relies on and is exactly backwards for a *negative* filter. A world where nobody had died yet had no `Downed` anywhere, so **nothing was ever handed to the ragdoll at all** | 0 handoffs where there should be 1 |

The last one is a law: **`try_query_filtered`'s `None` fast path cannot carry a
`Without`.** Read the latch per entity.

And the **reach budget** was a sixth, found the same way: `interact::resolve`
measures from the character's **feet** and a door's interaction point is at the
leaf's mid-height, so a 2.0 m reach is 1.70 m of floor. It is 2.4 now, which is
2.16 m of floor, and the arithmetic is on the constant.

### THE PERSISTENCE ANSWER (the brief's STOP question, answered without a STOP)

**A runtime `Inventory` needs no wire field**, and the accounting is written where
the next reader will look (`inf_ecs::item`'s module header):

* what a character picks up, carries, equips and drops during a session is
  derived, mutated by gameplay and — like a broken wall, a carved cave and a
  footprint — **not persisted**, because `.inf_lvl` is the author's document and
  this engine has no save-game container;
* an **authored per-entity starting inventory** *would* be a wire field, and the
  exact one is `RuntimeEntityGen::inventory: Option<Inventory>` at the record's
  tail — scene **v26**, its editor mirror, a frozen `EntityRecordV25`, a committed
  downgrade fixture, and `SCENE_PAYLOAD_VERSION` **12** by the envelope's own
  doctrine. I6 does not need it, so I6 does not take it.

There is **no generic component-reflection save path** in this tree: `props` and
`registry` exist for the Details panel and have no writer, and a repo-wide search
for `ReflectSerialize` finds nothing.

**Two catalogue homes were refused and priced.** A `.inf_item` asset needs a scene
field to be named by an entity *and* a `ScenePayload` vector to reach PIE — two
bumps. An `items.toml` beside the level reaches exactly **one** of the three boot
paths (the I5 audit's own finding A6 about `input.toml`), so a catalogue there
would be present in a dev run and absent in the build. What content authors
instead is the **Blueprint kit**, which rides `.inf_act` bytes that a cooked pack
and a PIE payload already both carry — the surface `destruct.*` and `voxel.*`
already are.

### The grammar's own doors, and what a city's doors cost

`inf_pcg::building::doorway` turns each `Opening { kind: Door }` the grammar
already plans into a hinge, a facing and a size, derived in the one pass that has
a `BuildingPlan` in hand (`evaluate_buildings_in` throws it away) and carried out
through `GrammarOutput` → `VolumeOutput` → both hosts' `population_of` → a
`#[serde(skip)]` `PcgVolume::doorways`. **No schema moves.** The leaf swings into
the room its wall serves, which is what `Wall::inside` means.

| the shipped city | number |
|---|---|
| blocks | 100 |
| structural solids | 370 468 |
| **doorways planned** | **19 790** (197.9 a block) |
| **doorways the band makes SOLID** | **234 — 1.18 %** |
| door leaves the physics world actually holds | **234**, equal to the band's list |

1.18 % against the walls' own **1.59 %** (6 213 of 390 258, printed by
`the_banded_city_holds_the_streamed_step_budget`; the *1.64 %* this line first
carried was I3's figure against a smaller solid count, and the I6 audit
re-measured it): the same discipline, at the same radii,
through one new door (`PhysicsBridge3D::sim_band`) so a level's doors and its
walls cannot be solid at different distances. And the two hosts plan the same
doors **byte for byte** — all 19 790 placements compared as raw bit patterns
across the cook and the PIE payload, because two of a doorway's eight mirrored
fields are angles and **swapping them compiles**.

### THE GATE — `phase30_gameplay_gate`, and the fixture under it

`samples/phase30-gameplay` (a grammar-built House, four hand-hung doors, a rifle
on the floor, a destructible target and one hero). Everything the scene cannot
carry is authored by the hero's own Blueprint on `BeginPlay`, through the new
`item.*` / `door.*` / `health.*` kit.

Six arms. The headline runs the same scripted trace on a **cooked pack** and on a
**PIE payload** and compares `state_bytes` step for step, with the coverage check
FIRST so two identical empty worlds cannot agree their way through:

| verb | what the trace measured |
|---|---|
| E picks the rifle up | 1 rifle, floor entity gone |
| I opens the panel | the panel's own `open` |
| F equips from the panel | `equipped_id() == "rifle"` |
| the wheel changes weapon | the sign reached the cycle |
| RMB aims | `RotationMode::Aiming` |
| LMB fires at the destructible | 6 rounds; **1 reached the target owing 1 700 J at the P22 door, and 12 chunks came off** |
| R reloads | magazine 24 → 30 |
| E opens the front door | −80.7 degrees |
| the hero walks through it | z −7.30 → −5.45, past a doorway at −6 |
| E locks it from the inside | the door's own `locked` |
| LMB kicks the locked gate in | **151.875 J against a 120 J lock** |
| a sprint breaches the shed door | **6.500 m/s in, 5.406 m/s out — 83.2 % kept** |
| a dive goes through the hatch | **5.499999691 m/s — the dive's own launch speed, 3.09e-7 off nominal** |

The anti-vacuity list is a `match` with no wildcard (`duty_of`), so a verb
deleted from the enum is a compile error rather than a silently dropped
obligation — `phase29_gate`'s own A3 lesson.

Two more arms are about what the comparison is *of*: `a_rifle_round_spends_its_
joules_at_the_p22_door` refuses a world where nothing can break (forcing `Fire`
only needs a magazine to move), and `the_trace_carries_the_doors_the_bag_the_
magazine_and_the_body` refuses a `state_bytes` that folded none of the four new
sections. Plus a two-cook replay and a fixture-file lock.

### What the gate itself found

* **`sim_from_built` does not attach fractures** — the shipped `--pack` boot
  does, in `lib.rs`. A gate that skipped it ran a world where **nothing can
  break**: six rounds owing 10 200 J at a door that answered `NoFracture` for
  every one, which reads as "the shot missed" and is not.
* **A character standing in a door's own arc BLOCKS it.** The lock station
  pressed E from the middle of the doorway and watched the leaf stop at 77
  degrees against the hero's own capsule. The script steps aside; the system was
  right.
* **E is a TOGGLE**, so a script that pressed it every twenty frames opened the
  door and shut it again — the first draft left the leaf at 11 degrees after two
  hundred steps.
* **Relative turns accumulate.** After the lock station — which faces the *door*
  rather than a cardinal direction — every later "walk east" walked somewhere
  else. `Host::turn_to` takes an absolute heading.
* **The breach is priced before the mode table honours a dive**, so a dive
  requested on the step it reaches a door is still `Grounded` when the breach is
  decided. The script arms it two steps out, and the arm asserts the *speed* —
  which is the dive's own launch speed — rather than the mode, because a dive off
  flat ground lasts one fixed step (I5's carried remainder #2, met again).

### THE FIFTEENTH chr(92) CATCH, and it was this wave's own

Three literals — two assertion messages in `commands/sim.rs` and one in the gate
— shipped mid-wave with an eaten `\` continuation, from scripted edits written
through a shell heredoc. `inf-packager`'s workspace sweep caught all three and
the repair went through the Edit tool, which is what the law prescribes.

### The closing ledger

| | after the I5 audit | **after I6** |
|---|---|---|
| battery blocks / passed / failed / ignored | 309 / 5 796 / 0 / 14 | **312 / 5 867 / 0 / 14** — three new test binaries (`door_3d`, `weapon_3d`, `phase30_gameplay_gate`) and **71** new arms |
| frontend tests / files | 702 / 78 | **702 / 78**, `tsc --noEmit` and `eslint` clean |
| goldens | 54, byte-identical under `INF_GOLDEN_STRICT=1` | **54, byte-identical** over 101 arms — re-run after the inventory panel and the tracer node landed, **no PNG rewritten** |
| `clippy --workspace --all-targets` under `-D warnings` | 0 | **0** |
| rustdoc warnings (ceiling 450) | *recorded as 413* | **447 over 45 documented crates — and 447 is what the base commit's own CI leg printed**, so the wave adds **zero**. See below: the recorded 413 was stale by 34 and the real headroom was **three** |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | **unchanged — no schema moved**, and the accounting for the one that was considered is in `inf_ecs::item`'s module header |
| fixed-step phases | 22 | **23** — `gameplay`, between the character step and the solver, in both hosts |
| committed samples | 20 levels | **21** — `samples/phase30-gameplay` is new; not one byte of the other twenty moved |
| `inf_input::actions::NOT_YET_CONSUMED` | 4 controls | **0** |

### THE RUSTDOC HEADROOM WAS THREE, NOT THIRTY-SEVEN

Every wave since P28.5 has carried "rustdoc 413" forward against a **450**
ceiling, which reads as thirty-seven warnings of room. It is not: the base
commit's own CI leg printed **`rustdoc warnings: 447 (ceiling 450) over 45
documented crates`**. The recorded number was stale by 34 and the true headroom
was **three**.

The way the wrong number survives is worth writing down, because it is the CI
step's own documented trap wearing a local disguise: **`cargo doc` re-emits
warnings only for crates it re-documents.** A warm local run documents whatever
the last interrupted run did not, and counts a fraction of the tree with total
confidence — this wave's first measurement was **449 over 17 crates**, a number
that is neither the baseline nor comparable to it, and it *looked* like the wave
had spent its whole allowance. CI already knows this and does `cargo clean --doc`
before it counts; a local measurement that skips that step is measuring its own
build cache. **Measure rustdoc the way CI measures it, or do not write the number
down.**

Measured properly (`cargo clean --doc` first): **447 over 45 crates**, equal to
the base's CI figure, and the wave's own contribution is **zero** — every
warning's `file:line` was checked against the diff's added-line set and **none**
falls on a line this wave wrote. Two did while the wave was in flight, both in
new code and both repaired here: a `[yaw_dir]` intra-doc link to a private
function in `inf_ecs::door`, and an `[inf_physics::d3::pcg_structure_guid]` link
in `inf_ecs::item` naming a crate that — by the facade rule — `inf-ecs` cannot
and must not depend on.

**Six** commits, `(I6)`-tagged, none pushed (the wave's own line said five and
listed six — the I6 audit counted them): the door core and the P22 energy rule's
one home; the gameplay engine half (doors, kicks, crashes, weapons, health); the
grammar's doors, banded; the gate, the fixture, the Blueprint kit and the tuning
door; the doorway-walk visitor; and the two doc links plus this ledger. *(A sha is only true of the
tree it was written in — the I3 audit's law — so the wave that closes this one
re-states the range rather than trusting a number copied out of this file.)*

## Decisions (I6's, binding on later waves)

* **A door's state is a sparse RESOURCE, not a component.** A grammar doorway has
  no entity to put one on, so a component would be two authorities on "is this
  door open" — and the city's twenty thousand doorways would be twenty thousand
  records nobody had touched. Absent means closed. It is P22's own split
  (`Destructible` authored, `FractureState` mutable) and it moves no schema.
* **Health is JOULES.** `docs/memos/p22-strength.md` §1 refuses damage numbers
  for walls; a character is the one place a bullet, a kick, a fall and a
  collapsing wall all meet, so a conversion table there would be the same mistake
  with more consumers.
* **The Blueprint class is the authoring surface for gameplay content**, and the
  two alternatives are refused with their prices: a `.inf_item` asset needs a
  scene field to be named *and* a `ScenePayload` vector to reach PIE (two bumps);
  an `items.toml` beside the level reaches exactly one of the three boot paths.
  `.inf_act` bytes reach all three with none.
* **`try_query_filtered`'s `None` fast path cannot carry a `Without`.** It
  refuses when a component it names has never been inserted — which is the
  `O(1)` answer everything relies on and is exactly backwards for a negative
  filter. Read the latch per entity.
* **A leaf's box axes follow the yaw convention, not the field order.** Yaw zero
  is `+Z`, so a leaf's length is its local `+Z` and a box written
  `(width, height, thickness)` lies across its own doorway at every angle.
* **A sweep that STARTS penetrating is not a block.** A door stands on the floor
  and between two walls; its own box is in resting contact with three solids at
  all times, so an exact-shape probe reads `toi == 0` and no door in the engine
  can open. The probe is inset by a 2 cm skin and ignores `started_penetrating`.
* **A reach measured from the FEET must budget the thing's height.**
  `interact::resolve` measures from the ground contact and a door's interaction
  point is at the leaf's mid-height, so a 2.0 m reach is 1.70 m of floor. The
  constant carries the arithmetic.
* **The inventory panel does not pause the simulation**, and the settings dialog
  does. A menu that did not pause would make the UI part of the sim's input; a
  bag that DID pause would be a safe place to stand in a firefight. The dialog
  outranks the panel when both are open, because that is what modal means.
* **A kick lands on the animation's notify, never on the button** — with a fuse
  for a character that has no rig to notify it, so the verb is not dead on every
  level committed before this wave. Both paths are armed and the report says
  which ran.
* **A corpse does not get up.** The ragdoll's get-up fires on settle, so a body
  the damage system handed over was re-handed on the next step. `Downed` is the
  latch and `Health::dead` is the guard on the get-up. *(The latch was armed; the
  **guard** was not until the I6 audit — see below.)*
* **The panel's verbs are VALUES the host applies**, never edits the panel makes.
  A UI that reached into the world would move a player's things on the frame
  clock instead of the fixed one.

## Decisions (the I6 audit's, binding on later waves)

* **A LOCK ONLY BREAKS IF IT WAS HOLDING.** `try_break` answers `broke` for a
  door that was merely shut as well as for one that was locked — nothing was
  holding it — and `apply_break` marked both as `lock_broken`. A broken lock
  never re-engages, so one sprint through a house's own unlocked front door
  retired that door's lock for the session, on every door the grammar emits,
  which is every door in the city. Read the **state**, not the price: a lock an
  author gave no area is engaged and costs zero, and should still break.
* **A GATE THAT COMPARES TWO TRACES CANNOT SEE A SECTION MISSING FROM BOTH.**
  Deleting `door_state_bytes` from `RuntimeSim::state_bytes` — and, separately,
  `weapon_state_bytes` — left all sixty-nine `inf-player` test binaries green,
  the PIE-versus-shipping gate included. Only a pin on the **fold itself** can
  see it: `every_trace_section_is_folded_in_its_frozen_order` allowlists all
  eight sections in order, which is P22's "a ban enumerates what you thought of,
  an allowlist what is allowed" at a trace.
* **A COVERAGE ROW MUST NAME A CHANGE, NOT A STATE.** "The wheel left the same
  weapon equipped" is a claim a wheel wired to nothing satisfies perfectly. A
  fixture with **one** of a thing cannot force a verb that cycles between them;
  the fixture carries the second one, or the row is theatre.
* **THE KEY THAT OPENS A SURFACE CLOSES IT**, and the close is decided against
  the **live map**. An open panel takes every key (that is what makes it a
  surface), so the press can never reach the host's own edge and the panel has
  to answer it. A literal — the settings dialog's own `"Escape" | "Tab"` — works
  until somebody rebinds it.
* **A LOCAL RUSTDOC COUNT MEANS NOTHING WITHOUT `cargo clean --doc`.** I6 found
  this and this audit re-measured it from cold to be sure: `cargo doc` re-emits
  warnings only for the crates it re-documents, so a warm tree counts a fraction
  of itself with total confidence. The number to write down is the one produced
  by `cargo clean --doc && CARGO_TERM_COLOR=never cargo doc --no-deps
  --workspace`, over the **45** documented crates CI counts. It is **404** at
  this head, against a ceiling of 450.

## The I6 audit (adversarial, `4155d30..7c0c997`)

**Every number in the wave's energy table re-derives.** 300 MPa × 4 cm² × 1 mm
= **120.000 J**; a kick is `0.5 × 15 × 4.5²` = **151.875 J**, 31.875 J to spare;
a sprint is `0.5 × 80 × 6.5²` = **1 690 J**; the pure exit is
`0.85 × sqrt(6.5² − 2·120/80)` = **5.325 m/s (81.9 %)**, and the gate's
**5.406 m/s / 83.2 %** is that same breach after one step of the movement step's
own friction and acceleration — two different quantities, each printed by the
test that measures it. The city reproduces exactly: **100 blocks, 370 468
solids, 19 790 doorways, 234 banded (1.18 %), 234 leaves**.

**THE ONE DOOR SURVIVES ITS OWN MUTATION.** Multiplying
`Destructible::bond_energy_j` by two fails **three** `inf_ecs::door` arms *and*
**two** `inf-physics::fracture_3d` arms at once. A second pricing site anywhere
in the tree would have left one of those two families green. There is none.

**The `Without` sweep is clean.** All thirteen `try_query_filtered` call sites in
the tree were enumerated: every one carries `With<…>` only. The single negative
filter that ever existed is the one the wave found, and restoring it still costs
the ragdoll its handoff (`0 handoffs where there should be 1`).
`hierarchy.rs:60` uses the **infallible** `query_filtered`, which registers the
component it names and is not subject to the law.

**Mutation-measured gate blindness, five found and closed:**

| mutation | before this audit | now |
|---|---|---|
| `door_state_bytes` deleted from `RuntimeSim::state_bytes` | **all 69 `inf-player` test binaries green** — two hosts that both stop folding a section agree about it perfectly | `every_trace_section_is_folded_in_its_frozen_order` (all eight sections, allowlisted in order) **and** the gate's own sections arm, which now opens a door and equips a rifle before it looks |
| `weapon_state_bytes` deleted from the fold | **all 69 green** — and the arm *named* "…the magazine…" never looked at it | the same two |
| `cycle_equipped`'s call deleted from `step_weapons` | **the whole gate green**: `ScrollSwitch` was forced by "the equipped id is unchanged", which a wheel wired to nothing satisfies | the fixture carries a **pistol**, and the station forces `rifle → pistol → rifle` |
| the `!dead` guard removed from the ragdoll get-up | **all 9 `weapon_3d` arms green** — that fixture's target has no rig, so P29.4's "no rig is coming" branch answers before the settle path is ever reached | `a_dead_body_stays_limp_where_a_live_one_gets_up`, on the rigged `ragdoll_bridge_3d` fixture, with the live body as its control (103 steps to a get-up; 900 and still limp) |
| `press_attack` never consumed in `step_weapons` | `door_3d`, `weapon_3d` and the gate all green — and `apply_intent`'s `\|=` latches the edge for the rest of the session | `an_attack_spent_at_an_unlocked_door_does_not_kick_it_when_it_is_locked_later` |

**Three world-level defects, found by reading and closed:**

* **A door you barged through could never be locked again.** `apply_break` set
  `lock_broken` on *any* door that gave, and a shut-but-unlocked door gives for
  free — so one sprint through a house's own front door retired its lock for the
  session (both `set_locked` and the prompt refuse a broken one). Every grammar
  door starts unlocked, so on the shipped city that was every door in it. Armed
  by `a_sprint_through_an_unlocked_door_leaves_a_lock_that_still_works`, whose
  control is the locked half still breaking and still refusing to re-engage.
* **`door.is_open` walked the unbanded list.** The Blueprint kit's one read is a
  node an author may put on `Tick`, and it collected all 19 790 of the shipped
  city's doorways — a label `String` allocated per door, then a 19 790-element
  sort — to answer a question about one. It band-checks by reach *before* it
  builds a placement now, which is the wave's own visitor discipline applied to
  the caller it missed; `placement_of` (the E key's and the breach's) took the
  same walk and takes the visitor too. It had **no test at all**; it has one.
* **Dead code with a false claim.** `try_breach` re-wrote `target_deg` to the
  value `apply_break` had already written and called it "open it the rest of the
  way **under power**", with `powered` false. Deleting it changed nothing under
  mutation. It is gone, and the claim it stood in for — that the weakest possible
  breach still reaches the frame — is now measured
  (`a_breach_with_nothing_to_spare_still_swings_the_leaf_to_its_stop`: 182 deg/s
  reaches the 95-degree stop in 46 steps, both signs).

**Plus four smaller ones.** `I` opened the inventory and could not close it (an
open panel takes every key, so the host's `just_pressed(INVENTORY)` edge could
never fire again — the owner's table says *open/close*; the close is decided
against the live map, so a rebound key works and the old one is just another key
the panel eats). `doorway_bits` compared eight of `DoorwaySlot`'s ten fields, so
a mirror that dropped `exterior` or `floor` was invisible — it destructures the
struct now and a new field stops the file compiling. `GAMEPLAY_DOORS_TOML`
carried **31 literal newlines** where the const beside it uses `\n` — the
fifteenth `chr(92)` catch's own shape, content-identical only because
`.gitattributes` forces `*.rs text eol=lf` — and its doc said "two" doors where
the fixture hangs four. The wave's own ledger said "five commits" over a list of
six, and quoted the walls' band as 1.64 % (I3's figure) where the test prints
1.59 %.

**THE RUSTDOC MARGIN, re-measured and then widened.** The wave's headline
reproduces exactly, CI-style (`cargo clean --doc`, then
`cargo doc --no-deps --workspace`): **447 warnings over 45 documented crates**,
three under the 450 ceiling. Thirty-nine were then cleared by hand — 31
`redundant explicit link target` and 8 `unclosed HTML tag` in four probe
binaries whose key/value output blocks are now fenced — and four crates fell to
zero and took their summary lines with them: **404 over 45 crates, and the
headroom is 46.**

*Accumulated laws gain a line: **a local rustdoc count means nothing without
`cargo clean --doc`.** `cargo doc` re-emits warnings only for crates it
re-documents, so a warm tree counts a fraction of itself with total confidence.*

**Verdict per claim.** ONE energy door **HELD** (mutation-proved across two
crates). The sparse `DoorField` **HELD** (dense fails two arms; entries persist
once a door is *used*, which is state rather than a walk-past, and the field
never shrinks — remainder 9). The leaf's axes **HELD**. The two kick paths
**HELD**: the gate runs the **fuse** (a fuse of 1e9 s costs it the `KickIn`
verb) and `door_3d` runs the notify with no double. The 19 790 placements
**HELD and genuinely cross-host** — cooked pack against PIE payload — and the
angle-swap hazard is covered better than the byte compare knows, because
`biome_binding_mirror` pins both `population_of` blocks character for character.
The joules arm **HELD** (dropping `attach_fractures` reds it). The persistence
answer **HELD** (no `ReflectSerialize` in the tree, no writer on
`props`/`registry`, the v26 + `EntityRecordV25` + payload-12 price is the real
one). The two-cook replay **HELD**, and it is in-process exactly as
`phase29_gate`'s own is. **CORRECTED**: the trace sections, the wheel verb, the
corpse guard, the barged lock, `I = open/close`, the unbanded probe, the dead
block, the eight-of-ten field pin.

### Counts, at the head this audit certifies

| | after I6 (as recorded) | **after the I6 audit** |
|---|---|---|
| battery blocks / passed / failed / ignored | 312 / 5 867 / 0 / 14 | **312 / 5 873 / 0 / 14** — six new arms, **no new test binary**, and the wave's own figure reproduces exactly |
| frontend tests / files | 702 / 78 | **702 / 78**, `tsc --noEmit` and `eslint` clean (the audit touched no frontend file) |
| goldens | 54, byte-identical under `INF_GOLDEN_STRICT=1` | **54, byte-identical**, re-run under `INF_GOLDEN_STRICT=1` over **101 arms** with no PNG rewritten — and no golden byte moved anywhere in `4155d30..` this tree |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** (local toolchain 1.97) |
| rustdoc individual warnings (ceiling 450) | 447 over 45 crates | **404 over 45 crates** — 39 cleared by hand, headroom 3 → **46** |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | **unchanged — the audit moved no schema** |
| committed samples | 21 levels | **21** — `samples/phase30-gameplay/Gameplay.inf_act` is regenerated (the pistol the wheel station needed); no other committed byte moved |

Five audit commits, `(I6)`-tagged, none pushed: the trace fold's allowlist and
the doorway's field pin; the wheel's second weapon and the inventory key's
close; the barged lock, the corpse guard, the spent edge and the banded probe;
the rustdoc clears; and this ledger. *(A sha is only true of the tree it was
written in — the I3 audit's law — so the wave that closes this one re-states the
range rather than trusting a number copied out of this file.)*

**Six of the implementer's mutations were re-run** and every one still dies at
the arm that names it: `Without<Downed>` (0 handoffs), the dense door field (2
arms), the transposed leaf axes (the wedge arm and the pose arm), `BLOCK_SKIN_M`
at zero (the wedge arm), the corpse's get-up (now armed), and a gate without
`attach_fractures` (the joules arm). One did **not**: removing
`!h.started_penetrating` from the blocking probe changes no test in the tree —
the skin is what carries that fix, and the clause on top of it is defensive
(remainder 10). **Twenty mutations in all**, six of them the implementer's and
fourteen new, and every fix above is recorded with the one that kills it.

## What is still open after I6

1. **A drop lands on the FRAME the key was pressed, not on a fixed step.**
   `RuntimeSim::apply_inventory_verb` is called by the host between frames, so a
   windowed player's own drop is the one place the frame clock touches gameplay.
   Every gate drives `step_once` and presses the verb through the same door, so
   the traces are exact; closing it properly means a queue on `RuntimeSim` that
   the fixed step drains. Stated on the function.
2. **A projectile is resolved as a hitscan.** `ShotKind::Projectile` changes the
   tracer's speed and nothing else in I6 — a body in flight is a wave of its own.
   Stated on `resolve_shot`.
3. **A tracer is a debug-line segment and a muzzle flash is the first 2 % of it.**
   There is no particle system (P22's own carried remainder), so a bullet leaves
   no smoke and a break still makes a sound and no dust.
4. **The editor's Simulate draws neither the settings dialog nor the inventory
   panel.** I5's remainder 8, extended: the editor's viewport has its own
   projector, and an author who wants either previews it with PIE.
5. **The authored-interactable walk is still `O(interactables)`** — I5's carried
   remainder 9. Doors ARE banded now (`placements_near`, the same band the walls
   use), so the largest new source of candidates is bounded; the pre-existing
   `Interactable` walk is not, and the fix is the same band.
   *What the banding cost to get right:* the derived-doorway walk returned a
   `Vec` for one afternoon, and on the shipped city that is **19 790 records,
   about 1.9 MB, copied out on every call** — three or four times a fixed step —
   so that the band could throw 98.8 % of them away.
   `inf_ecs::door::for_each_volume_doorway` is a **visitor**, so a caller
   band-checks before it allocates, and the per-step cost is `O(doorways)`
   pointer bumps with `O(near)` allocations. Determinism costs nothing at that
   shape: the hundred volumes are sorted and each one's slots are visited in
   index order, so the sequence is `(guid, index)` sorted without a sort over
   the doorways themselves.
6. **A dive off flat ground lasts one fixed step**, so the hatch breach is
   measured by the dive's own launch speed rather than by the mode at the end of
   the step. That is I5's remainder 2 met from the other side, and it is why the
   gate asserts a speed.
7. **Nothing persists.** A door a player kicked in, a bag they filled and a body
   they emptied are all runtime state, like a broken wall and a carved cave
   before them. `.inf_lvl` is the author's document and this engine still has no
   save-game container — see `inf_ecs::item`'s header for the exact wire field an
   authored starting inventory would force.
8. **A grammar door's lock is never engaged.** Nothing the building grammar
   builds starts locked, because a city whose every interior door was bolted is a
   city nobody can walk through and there is no authored intent to read one from.
   Locking is a verb a player or a Blueprint uses.
9. **The door field never shrinks.** An entry is written when a door's state
   changes and is never removed, so a door a player opened and shut again keeps
   its 48 bytes in the trace for ever even though its state is exactly the one
   `absent` would have produced. That is bounded by doors a player has **used**
   rather than by doors they have walked past — which is the defect the wave
   found and fixed — and it is stated rather than closed, because pruning would
   need a canonical-form rule that `lock_broken` and `locked_at_spawn` both have
   an opinion about.
10. **`!started_penetrating` in the blocking probe is unarmed.** `BLOCK_SKIN_M`
    is what actually keeps a leaf resting on the floor from reading as blocked —
    setting it to zero fails `a_wedge_in_the_swing_stops_the_leaf_where_it_is` —
    and removing the `started_penetrating` clause on top of it changes no test in
    the tree. It is defensive rather than load-bearing today; a fixture whose
    leaf is inside a solid is what would arm it, and what such a fixture *should*
    assert is a design question rather than a measurement.
11. **`DOOR_REACH_M`'s arithmetic is documented and not armed.** The constant
    carries the derivation (2.4 m of reach is 2.16 m of floor at a leaf's
    1.05 m mid-height) and it has exactly one use site; putting it back to 2.0
    fails nothing, because the gate's script walks to 1.6 m of floor. The
    argument for 2.4 is a player standing a comfortable pace away, which is a
    claim about comfort and not one a trace can hold.
12. **`gameplay::side_of` has no caller.** It is `pub`, documented as the seam
    "the hosts' prompt" reads, and nothing in the tree calls it — the prompt goes
    through `d3::door::candidates`, which decides the side itself. Kept because
    the next host that wants one press's side outside the candidate walk will
    want exactly this; named here so it is not mistaken for coverage.

---

## Done — wave I7 (the island data build)

**THE ISLAND EXISTS, AND ITS ELEVATION IS REAL.** Fifty-one square kilometres of
the North Shore behind Vancouver, sampled out of the AWS terrain-tiles terrarium
pyramid, carved into a landmass by a designed coastline, drained by streams the
ground itself decides, and crossed by a road network routed under a grade
ceiling. Built by **one command** in 24.7 seconds:

```sh
inf island build --recipe samples/island/island.toml
```

### The island in numbers

| | |
|---|---|
| map | 7 168 × 7 168 m = **51.38 km²** |
| land | **40.65 km²** (79.1 %) |
| peak on land | **948.7 m** — real survey |
| sea floor | **−60.0 m** on a 500 m shelf |
| coastline | **25.14 km**, 43 authored vertices |
| terrain | **784** level-0 tiles of 257², **1 064** in the catalog, **5 LOD levels**, **342.7 MB** |
| source | **156** terrarium tiles at z15 = **3.11 m/px**, upsampled **3.11×** onto a 1 m grid, **0 nodata** |
| water | **50 reaches / 26.32 km**, **2 lakes / 0.0708 km²**, **33 waterfall sites** (biggest a **29.5 m** drop), max catchment **2.42 km²** |
| biomes | forest **38.5 %**, plain 20.8 %, meadow 13.5 %, alpine 8.6 %, beach 6.8 %, farmland 6.1 %, urban 5.8 % |
| roads | **33.74 km** over 11 links and 7 junctions; worst grade **0.118** against a 0.080 ceiling, **7 of 2 442** stretches over (0.29 %) |
| build | fetch **156 tiles / 12 MB** · build **24.7 s** · cook **40.7 s** |

**Where the gigabytes live.** Everything heavy is **outside the tree**, at
`<checkout>/../island-build/` — 375 MB of terrain, road mesh and biome set, plus
a 12 MB tile cache. `island-build/` and `samples/island/build/` are gitignored as
belt-and-braces. **Committed: 307 KB** (`samples/island`) + **267 KB**
(`samples/island-fixture`, of which 208 KB is two real terrarium tiles).

### The recipe, and its nine steps

`samples/island/island.toml` is the whole design: where on Earth (UTM 10N, world
`(0,0,0)` at 49.343 N 123.102 W), how fine (28 × 28 tiles of 257² at 1 m), which
source (terrarium z15, cache outside the tree), where the sea is, the grade
ceiling, the hydrology thresholds, the Jenks classes and **what each class
means**, and seven settlement sites. `BuildStep::ALL` is the frozen order:

**plan → fetch → sample → carve → hydrology → biomes → roads → pyramid → write**

* **plan** — which source tiles the extent needs. Pure. A projected square is not
  a lat/lon rectangle, so the plan walks the **perimeter** rather than the four
  corners; measured, the north edge's middle bows **6.5 m** past its corners at
  51 km.
* **fetch** — **the only step that touches a network, and it is not in Ring 0**.
  `inf_island::plan_tiles` decides *which*, `cache_path`/`tile_url` decide *where
  and what*, and the `inf` CLI does the transfer by shelling out to `curl` — the
  Phase-5 `git` ruling, against linking an HTTPS stack whose root-certificate
  crate is off this project's licence allow-list.
* **sample** — **destination-driven**. Every output sample asks where it came
  from, through `inf_gis::Transform::to_source` (this wave's one addition outside
  the island crate: the inverse of a door that only went forward). Forward
  mapping would scatter, leave holes, and write at a density that is a function
  of latitude.
* **carve** — the coastline, the sea shelf, the beaches and the site pads.
* **hydrology** — priority-flood → D8 → accumulation → reaches; the fill's own
  depth is where the lakes are, so one pass answers two questions.
* **biomes** — Fisher-Jenks over the vegetated band, then the design masks.
* **roads** — the committed network draped and **audited**.
* **pyramid** and **write**.

### What is committed and what is not

| committed (307 KB) | not committed |
|---|---|
| `island.toml` — every decision | the `.inf_terrain` (342.7 MB) |
| `layers/coast.geojson` — the designed shore | the road mesh (517 086 vertices) |
| `layers/biomes.geojson` — the design masks | the `.inf_biomes` set |
| `layers/roads.geojson` — routed once, committed as the design | the tile cache (12 MB) |
| `layers/streams.geojson`, `layers/lakes.geojson` — derived, then committed | |
| `VancouverIsland.inf_lvl` + `VancouverIslandCover.inf_pcg` | |

The layers are **GeoJSON in the anchor's own CRS**, so they open in QGIS beside
the survey and the import transform is an **identity** — reading a committed
layer is an exact subtraction rather than a reprojection round trip.

### The CI-scale fixture

`samples/island-fixture` — 2.36 km² of the same ground with its **two real
terrarium tiles committed beside it** (208 KB, z13, fetched 2026-08-21 from the
public keyless endpoint). It runs **every step of the recipe** and never reaches
a network: the plan's tile list and the committed directory are compared **both
ways**, so a change that needed one more tile goes red here rather than reaching
for `curl` on a runner, and a tile committed that the plan does not name fails
too.

Eight arms, of which the gates: every `BuildStep::ALL` matched by a **count** in
the log in frozen order; the **world** asked where the ground is (dry at both
sites, wet off all four edges, every coastline vertex within 2.5 m of the
waterline, the far corner exactly on the shelf floor); streams, lakes and
waterfalls present and read back out of their committed layers with a reach's bed
measured below the ground it was found on; the road audit clean **after** the
corridor is levelled in; every biome reachable, the masks beating the classifier,
a city site reserved on the **terrain**; and two builds byte-identical.

### PIE == shipping, on a drive

`runtime/inf-player/tests/island_gate.rs`, on the CI-scale island: **900 steps of
0.4 m = 360 m, 900 distinct states, byte-identical** between the cooked pack (with
cell streaming) and the loose document (with cell streaming), with the terrain
paging and the partition activating underneath. Coverage first — both hosts must
hold the ground, the water and the hero — so two empty worlds cannot agree their
way through.

The cooked island carries **7 assets**: terrain, level, biome set, pcg, mesh,
meshlet DAG and a `.inf_part` with 1 streamed cell. The shipped player runs it:
`inf-player --pack … --headless --run-frames 300` exits 0.

### The island's own frame numbers

The I4 instrument's camera path is a **parameter** now, so the same frame loop
measures two worlds. RTX 4070 Ti, release, MIN of 3 rounds × 120 frames, 1080p, a
40 m-high flight east from Harbour City. **Reported, never asserted** — the
ceilings in `inf_player::budget` are set from the composed city, and asserting
them over a different world would re-pin a ratchet by accident.

| | p50 | p95 | p99 | GPU frame | pipelined estimate |
|---|---|---|---|---|---|
| **SHIPPED** | **18.209** | 19.287 | 20.887 | 6.657 | **10.994 (91.0 fps)** |
| **LIT** | **48.170** | 53.590 | 56.862 | 31.188 | 31.188 (32.1 fps) |

CPU, shipped: sim step 0.077 · stream sync 0.032 · projection 0.011 ·
**render (record) 10.874** · poll 7.348. GPU, lit, dearest first:
**vsm-raster 29.656 (95.1 %)** · gi 0.777 · vgeom 0.309 · water 0.101 ·
terrain 0.063. Content: 44 terrain tiles, 1 vgeom, **0 instances, 0 scatter
batches** — see the vegetation finding below.

**Distance from 60 fps: shipped p50 +1.609 / p95 +2.687 ms; lit p50 +31.570.**

### What the wave's own arms found

Nine defects, every one measured rather than reasoned about:

| finding | the number |
|---|---|
| **A BLACK PIXEL IS FINITE.** The terrarium codec's floor is `(0,0,0)` → −32 768 m, which every finiteness guard in this engine waves through; inside a coastline it is a 32 km pit | the shipped island's source really contains **56** such samples; now nodata, which the carve already turns into ocean |
| **An eight-neighbour router cannot traverse a uniform slope.** On gradient `g` a D8 step achieves `g` or `g/√2` and nothing else, so an 8 % ceiling on a 15 % hillside answered "no route" | `ROUTE_REACH_CELLS = 4` admits `(1,4)` chords at **0.243 g**; the arm measures both sides of the 8 % line |
| **A switchback's apex must not be cut.** Chaikin averaging at a reversal moves the apex toward its neighbours' midpoint, which is straight up the fall line | **0.1500 — the full gradient — on 24 of 336 stretches**; guarded, and the alternative is re-run inside the arm |
| **`--dry-run` suppressed the layer write the second pass reads.** `inf island route` is two passes and the first is a dry run; with the design unwritten the second audited a corridor never cut | **15.28 % of stretches over the ceiling; 0 % once the design is written whatever `dry_run` says** |
| **A pad does not build land.** A city site's radius reaches past its own shore, and a pad flattening toward the site's datum out there hands the island a rectangular headland | fixed at the door: the pad only levels what the coastline already calls land |
| **Jenks finds the gaps; an author says what grows in them.** A hard-coded "lowest is plain, top is forest" ladder | put **9.8 % of a rain-forest island under canopy** and called a third of it grassy plain; `class_biomes` is the recipe's now, and forest is **38.5 %** |
| **A threshold sized for a continent finds one stream on an island.** The largest catchment here is 2.42 km² — nothing drains far before it reaches the sea | a 1 km² threshold found **ONE** reach; at twelve hectares, **50 reaches, 2 lakes, 33 waterfalls** |
| **A partitioned level's cooked `.inf_lvl` carries no entities**, so a shipping sim without `attach_cell_streaming` holds only what `AlwaysLoaded` kept | **six records against fifteen**; the two agreed for 411 steps and then did not, which reads like a streaming defect and was a gate that forgot to boot the streamer |
| **The instrument's own fixture never attached terrain streaming**, because the composed city did not need it | the island's first measurement was a frame of sky and water — **0 terrain tiles, 0 instances** — and the anti-vacuity assertion is what said so |

And one gate that could not fail, caught by itself: `the_level_is_authored_from_
committed_design_alone` scans `island.rs` for the names of the things the level
must not read — and its own needle list is in that file, so a whole-file scan
matched itself. It stops at `#[cfg(test)]` now, and asserts that it did.

### THE SIXTEENTH AND SEVENTEENTH chr(92) CATCHES

Both this wave's own, both found by `inf-packager`'s workspace sweep, and they are
**different things wearing the same shape**:

* **A collision.** `IslandReport`'s summary is a fixed-column table an author
  reads, and its alignment is runs of spaces *inside* string literals — exactly
  what the sweep looks for. The sanctioned remedy is the `ALIGNED_ON_PURPOSE`
  allowlist, which is keyed on the **enclosing function's name alone** — and
  there are **twenty-four** functions called `summary` in this workspace, so
  exempting the natural name would have exempted all of them. The table moved
  into a uniquely named private helper (`island_summary_table`) instead. *An
  over-broad exemption is the ban-list hazard turned around.*
* **A defect, and the P22 law met head-on.** A scripted edit written through a
  Python heredoc put a `\`-continuation inside an `assert!` message; a lone `\`
  before a newline in a non-raw Python string is a **Python** continuation, so it
  ate the backslash and left eighteen spaces inside the literal. The sweep caught
  it in the final battery. The repair went through the Edit tool, which is what
  the law prescribes — and the law is now three waves old and has caught
  something in every one of them.

### Counts

| | after the I6 audit | **after I7** |
|---|---|---|
| battery blocks / passed / failed / ignored | 312 / 5 873 / 0 / 14 | **318 / 5 946 / 0 / 16** — six new test binaries (`inf-island`'s lib and doctest roots, `island_fixture`, `portable_math_law`, `preview`, and `inf-player`'s `island_gate`) and **73** new arms. The two new `#[ignore]`s are the elevation preview probe and the island's own flythrough, both of which need a fetched cache CI does not have |
| frontend tests / files | 702 / 78 | **702 / 78**, `tsc --noEmit` and `eslint` clean — no UI was touched |
| goldens | 54, byte-identical under `INF_GOLDEN_STRICT=1` | **54, byte-identical** — nothing this wave touched draws a golden scene |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** (local toolchain 1.97). Fourteen findings were cleared on the way, all in this wave's own code |
| rustdoc warnings (ceiling 450) | 404 over 45 crates | **404 over 46 crates** — measured CI-style (`cargo clean --doc` first, the I6 law), and **the wave adds zero**: the one link it introduced (a public constant naming a private `smooth`) was found and removed, which took the count 406 → 404 because a crate that falls to zero takes its summary line with it. Headroom **46** |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | **unchanged — no schema moved**, checked as a diff over the range |
| committed samples | 21 levels | **23** — `samples/island` and `samples/island-fixture`; `EXPECTED_LEVELS` moved in the same commit |
| new crates | — | **`inf-island`**, host-only, **no new external dependency** |

## Decisions (I7's, binding on later waves)

* **An island is a RECIPE, not committed bytes.** Fifty square kilometres of
  one-metre terrain is a quarter of a gigabyte before a pyramid. The repository
  commits the *generator* — the recipe, the coastline, the road network, the
  derived water layers, the masks and the level, 307 KB — and one command builds
  the rest **outside the tree**. That is the samples law read at a scale it had
  not been read at: where the output is too large to commit, the generator is
  what is committed and a CI-scale fixture is what proves it works.
* **The level is authored from the committed design ALONE.** Not from the
  terrain: a level whose numbers came from a build artifact would be a committed
  document only one machine could produce, and nothing could check it had not
  drifted. `inf_island::read_design` opens five small committed files and no
  elevation tile, and a source scan is what keeps it that way. The consequence
  worth carrying: **the player start's elevation comes from the committed road
  layer**, because the roads pass through every settlement and their vertices
  carry the ground each was planned at.
* **Ring 0 decides WHICH bytes; Ring 2 fetches them.** The engine makes no
  network call. `plan_tiles` names the tiles, `cache_path`/`tile_url` name where
  and what, and the CLI shells out to `curl` — so CI runs every other step
  against committed bytes and `--offline` is a refusal rather than a different
  code path. A response that is not a PNG is refused **before it reaches the
  cache**, because the dataset answers `NoSuchKey` above z15 with a 299-byte XML
  body a cache would happily keep.
* **The sampling step is not bit-portable and everything after it is**, and both
  halves are gated. `inf-island` is clean of `std` transcendentals with **no
  exemption**; the derivations are additionally banned from naming a projection
  door at all, and the lattice is asserted to name one so the ban is not vacuous.
  The consequences are stated rather than papered over: the `.inf_terrain` is a
  build artifact of one machine, and the committed derived layers are
  **verified** rather than re-derived (`LayerDrift` prints the comparison every
  run).
* **`sqrt` is portable and the transcendentals are not**, which is why a stream's
  width comes from a square root of an area ratio rather than `powf`, and a slope
  from `inf_math::portable::patan2_64` rather than `.atan()`.
* **An advisory and a blocking finding are two lists.** C4-40 says a report with
  a *blocking* finding exits non-zero, and the word is load-bearing: three of
  this pipeline's four standing advisories are facts about the survey no author
  can act on (the source really is 3.11 m where the grid is 1 m; eight tiles
  really are open ocean; 56 samples really are filled pixels), and a command that
  exited non-zero for those would have an exit code that says nothing.
* **A grade ceiling is audited on the ground the road SITS on, not the one it was
  planned against.** The router works on an 8 m lattice and the audit measures
  the 1 m terrain, which is why `PLAN_GRADE_MARGIN` exists — and it is *not* the
  hairpin, which is a real defect fixed in `smooth`. A margin that stands in for
  a defect is a defect that never gets fixed.
* **`ROAD_OVER_GRADE_CEILING` is a fraction with a named mechanism.** Where two
  designed routes cross at different elevations the corridor can honour only one,
  because this generator builds no grade separation. Eleven routes with seven
  crossings measure **7 of 2 442 stretches**; past 1 % the cause is no longer the
  crossings.
* **The vegetation binding lives on the `.inf_biomes` set**, which is the one
  authority `inf_pcg::BiomeBinding::from_set` reads — so the set this crate
  writes carries it rather than leaving a second place for it to be attached.
  Urban is the one biome that binds nothing, and that is what reserving it is
  *for*: wave I8's generator finds bare ground rather than a forest to clear.
* **In TOML a bare key after a `[table]` header belongs to that table.**
  `content = [...]` written below `[source]` became `source.content`, a key
  nothing reads, and serde's default is to ignore it.
  **`#[serde(deny_unknown_fields)]` on every recipe struct** turns that — and any
  misspelling — into a message.
* **A ScenePayload carries no partition**, so a `--pie` preview of a partitioned
  world builds it whole. Pre-existing, measured rather than described, and it is
  why the island's PIE == shipping arm compares the **loose document** against
  the pack — the pair P16.5's own gate compares, for exactly this reason.

## What is still open after I7

1. **THE VEGETATION SCATTERS NOTHING ON A STREAMED ISLAND, and here is the
   figure.** The wiring is real and measured: six biomes bind the cover graph,
   and with the ground paged the binding produces **4 958 instances over
   2.359 km²** (about **85 000** over the island's own land at the same density).
   Through the shipped boot it produces **0**, because `evaluate_biome_bindings`
   evaluates over the terrain's *resident* `data.xz_bounds()` and a streamed
   terrain ships no tiles. That is the I4 audit's own carried item — *"a streamed
   cell evaluates its `PcgVolume` and NOT its biome bindings"* — met at island
   scale. The fix is `cell_stream::reconcile`'s missing biome twin, the mirror of
   `evaluate_pcg_volumes_in`, in **both** hosts.
   `the_biome_binding_scatters_when_its_ground_is_resident_and_not_before`
   asserts the zero, so the day it is closed the arm goes red and gets rewritten.
2. **The lit island is 95 % VSM raster** — 29.656 of a 31.188 ms GPU frame, on a
   world whose casters are a heightfield and one road mesh. IP item 2 at island
   scale, and a much starker reading than the city's.
3. **The shipped frame's dearest stage is `render (record)` at 10.874 ms for a
   scene with 0 instances.** Whatever it is, it is not the drawing: the GPU frame
   is 6.657 ms. The other half of finding 2, and unattributed.
4. **Seven of 2 442 road stretches exceed the 0.080 ceiling, at 0.118**, where
   two routes cross at different elevations. This generator builds **no bridges
   and no tunnels**: a link with no land route under the ceiling is refused by
   name with both ends' positions rather than routed through the sea.
5. **The source is 3.11 m/px and the grid is 1 m.** Everything below 3.11 m is
   interpolation plus what the design puts there — the carve, the road corridors
   and the stream channels. The build says so every run
   (`[source.upsampled]`). z16 does not exist in this dataset; finer real data
   would be a LiDAR DTM through the GeoTIFF door, which is a different source
   and a different recipe `kind`.
6. **The committed water layers drift from a fresh derivation on the authoring
   machine**: streams 50 vs 51 (0.31 % of length), lakes 2 vs 2 (**16.47 % of
   area**). Not a portability problem — it is the corridor: the layers were
   derived on ground the road levelling has since changed, and the second build
   re-derives on the corridored ground. Reported, and it converges; a third
   `route` would commit the corridored derivation.
7. **The stream channels are cut and the reaches are not re-derived after.** One
   pass: derive → carve. A second derivation over the cut ground would find
   slightly different channels, and the design artifact is the first one.
8. **Only the ten largest reaches are `WaterBody::River` entities.** A
   `RiverPath` holds `segments × 16` frames and `WaterSurface::height_at` walks
   them, so binding all fifty would put tens of thousands of frames behind every
   buoyancy query. The other forty keep their carved channels and are **dry
   beds** — visible geometry with no water surface.
9. **A waterfall is a steep stream segment and nothing else.** Thirty-three sites
   are identified with their drops (the biggest **29.5 m**); what is missing is
   the *look*: there is no particle system in this engine (P22's own carried
   remainder), so a 29.5 m fall is a river surface on a steep bed with P20's foam
   on it, not spray. No new VFX system was built and none is claimed.
10. **`inf island build` cannot write a level**, because the `.inf_lvl` writer is
    `SceneDoc` in Ring 1 and the CLI must not link wgpu — the same ruling I2 took
    for `inf gis`. The level is written by the Ring-1 samples generator and
    **copied** into the project by the build's `[content]` list, which is what
    keeps it one command from an author's point of view.
11. **The hero has no rig.** `AnimStateMachine { sm: None }` and no
    `SkeletalMesh`: the island's character is a capsule that moves, because the
    phase-29 rig is a different sample's asset and copying it into the island's
    committed content would put a megabyte of skeleton in a folder whose whole
    argument is that it is small.
12. **The site pads are terraces, not settlements.** Seven sites, 2.34 km² of
    levelled ground and 5.8 % of the land reserved urban — and nothing standing
    on any of it. That is wave I8's, by the brief.
13. **The derivations run at 8 m.** A channel narrower than the pitch cannot be
    found, and the committed stream layer is a *design* artifact an author may
    edit rather than an oracle.
14. **`inf island` has no editor surface.** Everything is the CLI and the recipe;
    there is no wizard, no preview panel and no in-editor re-route. The ASCII
    elevation probe (`cargo test -p inf-island --test preview -- --ignored`) is
    what stands in for one.
15. **THE CIRCUIT IS DRAWN AND AUDITED, AND NOTHING DRIVES IT.** The brief asked
    for "a drivable road circuit connecting the (empty) city/town sites" and what
    ships is the road: 11 links, 33.74 km, 7 junctions, a surface mesh draped to
    within a 2 cm lift, and a grade audit that says a car could climb it. The
    level's own character is a **walking hero**, not a vehicle — P29.7's
    `Vehicle` exists and the island does not spawn one, because a vehicle needs a
    chassis mesh and a tuning block that would be the fourth committed asset in a
    folder whose whole argument is that it is small. The drive trace moves the
    streaming source at 24 m/s, which is the *streaming* claim; it is not a
    vehicle simulation.
16. **The road network's topology is asserted on a flat fixture, not on the
    island.** `the_network_topology_joins_the_cities_and_strings_the_towns`
    measures one highway, five town-to-city arterials and the closing circuit
    over flat ground; on the real island the same planner produced 11 links and
    7 junctions, which is consistent with it and is not the same as an assertion
    that every site is reachable from every other. A connectivity walk over the
    built `RoadGraph` is what would close it.

---

## The I7 audit (2026-08-25)

Fresh auditor, `7d2d7ba..7790155` read commit by commit, five `(I7) audit:`
commits on top. **Two HIGHs and they are one story**; six MEDs; nine LOWs
carried by name.

### THE ISLAND STOOD HALF A WORLD FROM ITS OWN GROUND

Every other terrain in this repository is built with level-0 tile coordinates
starting at `(0, 0)`, and its entity is translated to `-span/2` to centre the
grid on the world origin — `island_frame_terrain_origin` is the pattern the
composed city uses. **`IslandGrid` does not work that way.** `tile0 =
-(tiles / 2)`, so the `.inf_terrain`'s own tile indices are already centred and
its sample frame **is** the world frame — which is what the entire build assumes:
`CoarseHeights::of(&data, min, max, …)`, the grade audit's `data.height_at(p)`,
the channel carve, the biome stamp and every arm in `island_fixture`.

`island_scene` translated the entity as well. The centring was applied **twice**.
Measured through the shipped host's own `terrain.height_at` seam — the exact
function a Blueprint node, the character's ground snap and the physics
heightfield all dispatch to:

| world position | the simulation | the recipe |
|---|---|---|
| the design's player start, (−420, 380) | **0.000 m** | **129.916 m** |
| the world origin | 80.000 m (a page 768 m away) | 172.801 m |
| (200, −200) | 0.000 m | 191.109 m |
| the second settlement, (430, −300) | 0.000 m | 212.179 m |

Three of the four read the **unauthored default**, because the displaced want set
asked for tiles the asset does not have. On the fixture the displacement is 768 m
on each axis; on the shipped 51 km² island it is **3 584 m**, which puts half the
terrain outside the world. Fixed at `Transform::IDENTITY`, with both committed
levels re-blessed through the generator (`INF_BLESS_SAMPLES=1`) — **no other
sample byte moved**.

### AND THE GATE THAT NEVER STREAMED IS WHY NOBODY KNEW

`island_gate`'s `pack_sim` and `loose_sim` attached `attach_cell_streaming` and
stopped. `run_headless` — which `pack_sim`'s own doc says it is "exactly" — calls
`attach_terrain_streaming` on the next line. So the island's 4.6 MB of pages
never moved: the `Terrain` component kept the empty working set a streamed level
ships, and every height query in all 900 steps answered off nothing. **Two hosts
standing on no ground agree perfectly**, which is how a gate whose subject is
streaming survived a terrain that was 768 m out of place.

**Mutation-measured, and it is the wave's own D8 finding from the other side:**

| mutation | the wave's gate | now |
|---|---|---|
| `attach_cell_streaming` off **one** host | red at the byte compare (D8 found this) | red |
| `attach_cell_streaming` off **both** | **all five arms green** — the coverage check reads `AlwaysLoaded` entities, and the 900 distinct states come from the drive moving the hero itself | `shipping activated 0 cell(s) over 360 m` |
| `attach_terrain_streaming` off both | not attached in the first place | two arms red |
| the double-centring put back on the entity | invisible | red at the first probe |

Three things make it falsify now:

* **Both streamers on both hosts**, and `loose_sim` built the way
  `build_world`'s own `--level` arm builds it — with the PCG payloads, the biome
  sets and the terrain resolver the pack side always had. Two hosts compared for
  byte equality must be given the same world to disagree about; the first draft
  compared one real reading against one impoverished one.
* **`streaming_counters`**, asserted non-zero on both hosts, **equal between
  them**, and taken before and after the drive: **1 cell activation, 20
  sim-resident level-0 pages, 16 page loads at the start and 20 after 360 m** —
  against 0 tiles and 0 loads before.
* **`the_ground_the_simulation_stands_on_is_the_ground_the_recipe_built`** —
  host against the **RECIPE**, at both settlement sites and two points between
  them, with an anti-vacuity arm (61 m of relief across the probes, every one
  above the waterline) so a displaced terrain cannot match them by being flat.

*The wave found this defect one file over and fixed it there.* D9 is *"the
instrument's own fixture never attached terrain streaming"* — corrected in
`fps_instrument.rs` and left standing in the gate beside it. **A fix applied
where it was found is not a fix applied where it belongs.**

### Six MEDs

* **`inf island route` rewrote the water an author had edited.**
  `BuildOptions::planning_pass()` carried `rederive_layers: true`, so a verb
  whose entire subject is the road network silently re-derived and overwrote the
  committed **stream and lake** layers. That is the hazard `rederive_layers`'s
  own doc names two fields up — *"a build that silently rewrote them every run
  would make an author's edit last exactly until the next build"*. The wave's D4
  finding fixed the opposite half (`dry_run` suppressing a write the second pass
  needed) and reached one field too far. Off costs a fresh island nothing: the
  write fires on `rederive_layers || !streams.exists() || !lakes.exists()`, and
  the arm asserts both halves — a renamed reach survives the pass byte for byte,
  and an island with no water gets **9 reaches and 1 lake** written.
* **Two doors onto the player start.** `inf island build` printed
  `IslandBuild::player_start`, which read the **built terrain**; the level's hero
  comes from `IslandDesign::start`, which reads the **committed road layer**. The
  command printed the one nothing spawns at. The road door is the one that
  survives (the level is authored from committed design alone), the build's
  delegates to it, and the terrain's reading is a second function named for what
  it is. The gap, measured: **129.924 m planned against 129.916 m built, 0.008 m
  apart** — structural rather than visible, and closed before it stops being.
* **An odd tile count is not centred, and two doors measured the wrong square.**
  `IslandRecipe::validate` and `plan_tiles` both read `±half_extent_m`; the world
  is `IslandGrid::bounds()`, and the two differ by half a tile span whenever
  `tiles` is odd (`tiles = 5` at 256 m is `[-512, 768]`, not `±640`). So an odd
  recipe admitted a site at `x = −600` on ground the build never makes, refused
  one at `x = 700` that is inside the world, and planned a source band stopping
  128 m short of the east and south edges — a strip with no elevation, which the
  carve turns into ocean with nothing said beyond a nodata count. Both halves
  mutation-verified separately. **No committed byte moves**: both islands have an
  even tile count.
* **The committed-design scan was a ban list.** I7's own headline decision is
  that the level is authored from committed design alone, and
  `the_level_is_authored_from_committed_design_alone` banned five names.
  `inf_island::sample_terrain`, `inf_island::IslandBuild`,
  `inf_terrain::read_terrain_asset` and `TerrainData::height_at` all walked
  through it. Inverted to an **allowlist of nine** `inf_island` doors plus a ban
  on the terrain crate entirely; mutation-verified with a `pub fn` taking an
  `&inf_island::IslandBuild`, which the old needles did not contain. It also had
  **no anti-vacuity arm**, which its sibling
  (`inf-island/tests/portable_math_law.rs`) does have; it has one now.
* **A layer stated a CRS it might not be in.** Every file `layers.rs` writes
  carried the literal `urn:ogc:def:crs:EPSG::32610` — true of the two committed
  islands and of nothing else, because the recipe takes any projected metric CRS.
  An island anchored in another zone wrote layers that told QGIS the wrong one,
  and the symptom is a coastline hundreds of kilometres from the survey it was
  traced off with no error anywhere. Derived from the anchor now; a proj4 string
  is written verbatim rather than given an invented URN. No committed byte moves.
* **The mask count counted one biome.** `BiomeClassification::masked` is printed
  by the report and the build log as "cells the masks overrode" and was computed
  as `id == Farmland` — and **both** committed mask layers name meadow as well.
  Measured on the fixture: **2 460 cells decided by a design mask, of which 1 350
  are farmland**; the figure was 82 % low, and a meadow the classifier chose was
  indistinguishable from one an author drew. The cause is the one-door shape: the
  counting loop carried a second copy of the mask test. `Classifier::mask_at` and
  `Classifier::reserved_at` are the one place each predicate lives now.

Plus two smaller ones in the same commits: **a probe the arm measured and threw
away** (`let _ = off;` after sampling the ground thirty metres off a reach, so
the bed assertion was satisfied by a carve that lowered the whole island — it is
the assertion it was standing next to now), and **the eighteenth chr(92) catch,
which is this audit's own**: a Python heredoc ate two backslash-continuations and
left fourteen spaces inside two literals, one wave after the seventeenth. The
sweep's threshold is eight, so the battery would have caught both; they were
repaired through the Edit tool, which is what the law prescribes and what the
heredoc had bypassed.

### Verdict per claim

The recipe's **nine steps** and their frozen order **HELD** (counted, not
matched). The fixture's **network-free both-ways tile comparison HELD** and is
genuinely both ways. **Two builds byte-identical HELD.** The **committed-design
source scan** was a ban list and is now an allowlist. The **portability law
HELD**: `inf-island` really is clean of `std` transcendentals with no exemption,
the derivations really are banned from naming a projection door, the lattice
really does name one (so the ban is not vacuous), and the "not linked by the cook
or the runtime" arm parses only the shipping sections and refuses to read an
empty dependency list. The **`ROUTE_REACH_CELLS = 4` chord admission HELD** —
the grade audit measures the drape against the 1 m terrain at 20 m steps, so it
is independent of the router's own lattice and of the chord averaging. The
**apex guard HELD** (the alternative is re-run inside the arm). **CORRECTED**:
the terrain's placement, the gate's streaming, the route verb's write, the start
door, the odd-grid square, the design scan, the layer CRS, the mask count, the
discarded probe.

### The three routed items, and whether they have a tripwire

* **The vegetation resident-bounds gap (I7b).**
  `the_biome_binding_scatters_when_its_ground_is_resident_and_not_before` is a
  real zero-assertion arm: it measures **4 958 instances with the ground paged**
  and asserts **0 through the shipped boot**, so the day the twin lands the arm
  goes red and gets rewritten. Verified present and firing.
* **VSM caster-pack caching** and **`render (record)` at 10.874 ms** have **no
  tripwire and cannot have one at the instrument's own ruling**: both numbers
  come from `the_island_at_shipping_resolution`, which is `#[ignore]`d (it needs
  a fetched cache and a real GPU) and **reports, never asserts**, because the
  `inf_player::budget` ceilings are set from the composed city and asserting them
  over a different world would re-pin a ratchet by accident. They are routed
  prose, not armed prose, and this ledger says so rather than implying otherwise.

### Carried LOWs, by name

1. **A lake is a D8 sink.** The priority flood has no epsilon gradient, so every
   cell in a filled depression sits at one level, `drop <= 0.0` for all eight
   neighbours and `down == NO_DOWN`. Flow that enters a lake's interior stops
   there, and a reach below a lake carries only its own local drainage. Bounded
   here by two lakes of 0.0708 km² total; it is a real hydrological
   simplification and it is not stated anywhere in the wave's own remainders.
2. **One channel width for every reach.** `carve_channels` is called with
   `widest` — the largest `width_m()` over the network, floored at 2.0 — as the
   **half**-width, for every reach. A 1.5 m rivulet gets the same trench as the
   biggest river. On the island the largest catchment is 2.42 km², so `widest` is
   about 3.1 m and the trench is ~6 m; the shape is coarse rather than wrong.
3. **The chord land-check truncates toward zero.** `route()`'s intermediate-cell
   walk uses `(dx * t) / span` in integer arithmetic, so a step and its mirror
   sample different cells. The symmetry the step set is asserted to have does not
   reach this walk.
4. **Public doors with no production caller**: `inf_island::terrain::flat_tile`,
   `hydro::lakes_by_id`, `biome::urban_reservations`, `biome::reserves`,
   `build::scattering_biomes`, `layers::write_masks`, `layers::write_coast`, and
   `inf_editor_core::island::cover_volume_guid` (the level binds vegetation
   through the biome set and has no `PcgVolume`, so the guid names something the
   level does not contain). Named so they are not mistaken for coverage.
5. **`slug()`'s `.replace(' ', "")` is dead** — the `is_ascii_alphanumeric`
   filter has already dropped every space.
6. **`cmd_island_plan` parses its arguments and discards them** (`let _ = a;`):
   `--out`, `--offline` and `--dry-run` are accepted and ignored by the plan verb
   rather than refused.
7. **Two recipe fields are unvalidated.** `[roads] shoulder_mult` and
   `[hydro] vertex_stride` skip the finiteness sweep every other number takes: a
   NaN shoulder silently disables the road corridor (`corridor_half > 0.0` is
   false) and a zero stride is saturated to 1 rather than refused.
8. **The wave's committed-size figures were measured with the wrong ruler.**
   307 KB and 267 KB are `du` block totals; the tracked bytes are **287 679**
   (281 KB, eleven files) and **245 966** (240 KB, thirteen files, of which
   **209 258** are the two tiles rather than "208 KB"). The README said "about
   260 KB" and the ROADMAP said 284 KB — three figures for one number. All three
   corrected to the byte counts.
9. **`the_network_topology_joins_the_cities_and_strings_the_towns` is still on
   flat ground** — the wave's own carried remainder 16, re-read and agreed with:
   a connectivity walk over the built `RoadGraph` is what would close it, and
   nothing in this audit changes that.

### Counts, at the head this audit certifies

| | after I7 (as recorded) | **after the I7 audit** |
|---|---|---|
| battery blocks / passed / failed / ignored | 318 / 5 946 / 0 / 16 | **318 / 5 952 / 0 / 16** — six new arms, **no new test binary**, and the wave's own figure reproduces exactly |
| frontend tests / files | 702 / 78 | **702 / 78, not re-run** — the audit touched no file under `editor/studio` |
| goldens | 54, byte-identical under `INF_GOLDEN_STRICT=1` | **54, byte-identical**, re-run under `INF_GOLDEN_STRICT=1` over **101 arms** with no PNG rewritten |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** (local toolchain 1.97, run LAST per the rmeta law) |
| rustdoc individual warnings (ceiling 450) | 404 over 46 crates | **404 over 46 crates** — measured CI-style after `cargo clean --doc`; the audit adds zero. Headroom **46** |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | **unchanged — the audit moved no schema** |
| committed samples | 23 levels | **23** — `VancouverIsland.inf_lvl` and `IslandFixture.inf_lvl` (and their sidecars) regenerated through the generator; no other committed byte moved |

**Six** audit commits, `(I7)`-tagged, none pushed: the terrain's placement and
the gate that never streamed; the route verb's write; the start door and the odd
grid; the design allowlist and the layer CRS; the mask count and the discarded
probe; and this ledger. *(Counted rather than summarised — the I6 audit's own
catch was a wave whose ledger said "five commits" over a list of six. A sha is
only true of the tree it was written in, the I3 audit's law, so the wave that
closes this one re-states the range rather than trusting a number copied out of
this file.)*

**Eight mutations run in all** — the wave's own D8 removal met from the
both-hosts side, and seven new — and every fix above is recorded with the one
that kills it.

## The I7 CI-red (run 32822072658, `7d2d7ba..7059031`)

The wave and its audit went to `main` green on this machine and came back **red on two
platforms of three**. `windows-latest` — the platform every byte in the wave was blessed
on — passed. That asymmetry is the whole story of both failures: each was a claim about
the *machine that ran it* wearing the clothes of a claim about the engine.

### RED 1 — one ulp of latitude, on macOS

`samples::tests::committed_sample_matches_generators` reported
`samples/island/VancouverIsland.inf_lvl` drifted from the island generator. The panic
prints both arrays, so the drift is measurable rather than guessable: **14 820 bytes on
each side, one byte different, at offset 14 788.**

That offset is inside the `GeoAnchor` settings block at the tail of the level, and the
field is `origin_latitude_deg`:

| | value | the f64 |
|---|---|---|
| committed (blessed on Windows) | 49.34307562364773 | `0x4048ABE9E6EBCF97` |
| generated on macOS | 49.34307562364772 | `0x4048ABE9E6EBCF96` |

**One ulp.** The longitude (-123.10187387613468) and the convergence (0.07728924942362428)
sat immediately after it and happened to agree; they came through the same door and were
equally exposed.

The door: `IslandRecipe::anchor` called `inf_gis::anchor_at`, which inverts the recipe's
easting/northing through `proj4rs` — a series over `sin`/`cos`/`atan2`, i.e. the platform's
libm. `read_design` calls it on every read, `island_scene` writes what it returns into the
committed `.inf_lvl`, and the P14 law says a value two machines re-derive independently may
not depend on one. `samples.rs`'s own `terrain_demo_height` carries the same lesson in its
doc comment — *"green where it was blessed and a latent red on any target whose libm rounds
differently"* — written for `sin`, waves before a projection library did it instead.

**The fix is the I7 ruling applied literally: a value that cannot be made portable does not
go in a committed file.** The recipe now **states** its geodetic origin
(`[anchor] latitude_deg` / `longitude_deg` / `convergence_deg`, to 1e-9 deg = 0.11 mm) and
`IslandRecipe::anchor` assembles a `GeoAnchor` out of stated numbers, checking only that the
CRS is projected — a new non-inverting door, `inf_gis::require_projected_crs`, which reads a
string and a table and returns no float. A decimal in a committed TOML is parsed by
`f64::from_str`, which is correctly rounded and therefore the same on every target: the byte
now traces to source. Recipe schema **1 -> 2**.

Rounding the inversion at the door was considered and refused: it would have made the byte
*usually* stable and never *provably* stable, because a value an ulp from a rounding tie
rounds two ways, and "unlikely" is not a property a gate can rest on.

A restatement owes a check, and it has three:

* `crates/inf-island/tests/stated_anchor.rs` inverts each committed recipe through `proj4rs`
  and asserts the stated degrees agree within `ANCHOR_AGREEMENT_DEG` (**1e-8 deg = 1.1 mm**).
  Measured residuals: **3.5e-10 / 1.3e-10 / 4.2e-10** on the island and
  **6.6e-11 / 8.1e-11 / 2.9e-10** on the fixture — twenty-plus times inside the tolerance,
  and six orders of magnitude above the ~7e-15 deg two libms are entitled to differ by. The
  file also carries the anti-vacuity arm (a hundred-metre error is outside it, and the real
  residual has better than 10x headroom) and one that a recipe *missing* the stated origin
  is refused by name — the three fields have no serde default on purpose, because a default
  is a silent equator.
* `crates/inf-island/tests/portable_math_law.rs` bans `anchor_at(` across **every** module
  of the crate, with a vacuity guard that `recipe.rs` still checks its CRS through the
  replacement door and still carries stated degrees.
* `island.rs`'s fixture arm now compares the level's whole anchor against the recipe's
  fields with `assert_eq!` on f64 — a committed byte has to trace to a committed decimal,
  and exact equality is the only comparison that says so.

**The gate that should have caught it, and why it did not.** `portable_math_law.rs` already
claimed "every module in this crate is clean of `std` transcendentals, with no exemption",
and it was *true* — the leak was a crate calling a library that calls a `sin`. Its second
arm, which bans the projection doors, enumerated **four modules** (`hydro`, `roads`,
`biome`, `shape`) and **three needles** (`to_source(`, `lonlat_to_mercator(`,
`mercator_to_lonlat(`). `recipe.rs` was not one of the four and `anchor_at(` was not one of
the three. Both halves are now the other way up: an **allowlist** of the modules permitted
to name a coordinate door, each with its reason (`terrain.rs` the lattice, `source.rs` the
tile plan, `recipe.rs` one longitude-reporting test), so a module added tomorrow is banned
by default rather than unlisted by accident. *A ban enumerates what you thought of; an
allowlist enumerates what is allowed* — met again, and the first time on a door rather than
a spelling.

**The fixture carried the same hazard and was never seen to fail.** `island-fixture`'s
level is compared in the same loop, *after* the full island's, so on the red run the
assertion over its bytes never ran. "It passed on macOS" would have been a reading of an
assertion that did not execute; whether its northing (5 467 400) inverts to the same f64 on
both platforms is **unknown and now moot** — its recipe states its degrees too.

**Byte arithmetic of the re-bless**, through the generator, both levels:

| level | length | bytes changed | where |
|---|---|---|---|
| `VancouverIsland.inf_lvl` | 14 820 -> **14 820** | **9** | latitude @14 788 (3), longitude @14 796 (2), convergence @14 804 (4) |
| `IslandFixture.inf_lvl` | 8 134 -> **8 134** | **8** | latitude @8 102 (2), longitude @8 110 (2), convergence @8 118 (4) |

Nothing else in either file moved — 14 811 and 8 126 bytes byte-identical — which is itself
the measurement that says the rest of the level-authoring path (the coast, the roads, the
streams, the biome masks, the GUIDs, the player start the audit had just rewritten) was
already portable, and that macOS agreed with Windows about all of it. The two `.inf_lvl.toml`
sidecars carry the new content hashes; the two `.inf_pcg` covers are untouched.

### RED 2 — a two-millisecond sleep that took five, on ubuntu

`step_profile::tests::a_phase_marked_twice_sums_rather_than_replaces` asserted that two
2 ms stretches charged to one phase read more than 1.5x the first alone. On the runner the
first stretch measured **4.990 ms** and the pair **6.991 ms**: the ratio came out at 1.40
and a green tree went red with nothing changed but the machine.

The property under test — **a phase marked twice SUMS rather than REPLACES** — is
arithmetic. It has no clock in it. `StepClock::mark` did two separable things (read a clock,
charge an interval), and the arm could only reach the second through the first.

`mark` is now three lines over a private `mark_at(phase, now)` that takes the timestamp as
an argument and holds *the whole of the arithmetic*; `mark` is `mark_at(phase,
Instant::now())` and nothing else, so an arm that drives the seam drives the shipped code
and there is no second copy to drift. The rewritten arm charges **three decided stretches**
— 2 ms, 6 ms, 4 ms, unequal on purpose so "sums" is distinguishable from "keeps the
largest" — and asserts 2 / 8 / 12 ms with a **1e-9 ms** slop that is not a noise budget
(there is no noise) but nine orders of headroom over the one correctly-rounded divide in
`Duration::as_secs_f64`. It also asserts the profile *total* is 12, so a mark that spilled
into a neighbour fails too.

A second arm, `the_live_mark_charges_the_phase_it_names_and_advances_the_clock`, aims at the
real `mark` with the only assertions a wall clock can make that a runner cannot move: the
charge is finite and non-negative, the clock advances, no other phase moved, and a second
mark can only raise the row. *A gate must aim at the thing it names* — the seam arm proves
the arithmetic, this one proves `mark` is the seam.

Swept: `step_profile` had exactly one real-clock arm and it is the one rewritten;
`a_disabled_clock_measures_nothing_and_answers_none`, the two table arms and the mean arm
are pure. The only remaining `Instant`/`sleep` in `inf-player/src` are `main.rs`'s frame
pacing and `window.rs`'s frame timer, both production.

### Mutations run

| mutation | arms that die |
|---|---|
| `IslandRecipe::anchor` restored to `inf_gis::anchor_at` | `no_module_inverts_an_anchor_out_of_a_projection` (names the file and the line), `every_committed_recipe_states_a_true_geodetic_origin`, `a_recipe_round_trips_and_derives_its_own_geometry` — three arms in two binaries |
| a stated latitude moved by 1e-6 deg (11 cm) | `every_committed_recipe_states_a_true_geodetic_origin` prints "1.0003522703527779e-6 degrees apart, past the 1e-8", and the headroom arm goes with it |
| `StepClock::mark_at` charges with `=` instead of `+=` | `a_phase_marked_twice_sums_rather_than_replaces`, at the 8 ms assertion, reading 6 |

### What could not be verified from here

This is a Windows machine and the two reds are on macOS and ubuntu. RED 1's fix is
verifiable *by construction* rather than by re-running the platform: the committed byte is
now a decimal parsed by a correctly-rounded parser, and no libm sits between the recipe and
the file — which is a stronger statement than "it passed on the third runner". RED 2's fix
reads no clock at all. What remains unverified on those platforms is only that the tree
compiles and the rest of the battery is unmoved, which `cargo check --workspace
--all-targets` and the touched crates' suites cover here.

---

## Done — wave I7b (the island lives at 60)

Base `545614f`. Three clauses, each measured before and after on the real island.

### Clause 1 — THE VEGETATION SCATTERS THROUGH THE SHIPPED BOOT (**done**, `e2542343`)

Wave I7's figure was **4 958 instances with the ground paged by hand and 0 through
the shipped boot**. The cause was one function running once: `evaluate_biome_bindings`
evaluated over `TerrainData::xz_bounds()` — the bounding box of whatever is resident —
at load, and a streamed terrain ships no tiles.

**What landed**

* **One Ring-0 door that walks TILES, not a box.** `BiomeBinding::refresh_resident`
  (`crates/inf-pcg/src/binding.rs`) evaluates the resident level-0 tiles in ascending
  coordinate order, memoized per tile in a `BiomeScatterCache`. It is **exact**, not an
  approximation: `scatter_region_in`'s lattice is world-anchored (`floor(x / cell_size)`)
  and its region clip is half-open, so the union of two abutting boxes is the population
  of their union, instance for instance. Measured on the real fixture: the per-tile walk
  over the fully-paged island grows the same **4 958** the whole-bounds walk did.
* **The key carries the NEIGHBOURS.** A candidate near a tile edge reads *across* it —
  `BiomeMask`'s feather search walks up to `MAX_FEATHER_SAMPLES` lattice rings and the
  slope filter's numerical normal probes `FN_HEIGHT_NORMAL_EPS` either side — so a tile
  evaluated alone rejects what the same tile inside its neighbours keeps. `scatter_reach_m`
  bounds that reach for **any** document (the mask's own cap, not the feather in front of
  us) and `neighbour_rings` sizes the neighbourhood from it: **one ring at every grid this
  engine ships** (a 257² tile at 1 m is 256 m across against a 64.1 m reach).
* **The fixed step runs it**, as phase 2 of **24**, straight after cell and terrain
  streaming, because its subject is exactly what those two make resident. One stamp
  comparison per terrain on a step that paged nothing. The tiles are **moved out of the
  component and back**, never cloned: this runs 60 times a second and a `TerrainData`
  clone is a quarter of a megabyte per tile.
* **One door for the boot.** `BuiltWorld::take_pcg_context` → `pcg_context()` (a clone),
  so `sim_from_built` — the one function every boot path goes through — seeds the graphs
  and the palette onto the sim. No new attach for a boot path to forget.
* The editor's `pcg_evaluate_biomes` goes through the same door, and the mirror gate now
  **bans `.xz_bounds()` on both sides** by name.

**Measured, CI-scale island (`island_gate`)**

| | |
|---|---|
| shipped boot | **0 → 2 339 instances** on 16 paged tiles |
| over a 900-step drive out and back | **2 339..3 119** instances on **16..20** sim tiles |
| PIE == shipping | on the state fold **and** on the forest, every step |
| stray instances | **0 of 2 339** — every place the streamed island grows, the fully-paged reading grows too |
| a fully-resident interior tile | **225 against 225**, identical |

**Laws met / paid for**

* **A memo keyed on first sight is keyed on the observer** (P21). Mutation-verified:
  `rings = 0` (the tile's own stamp alone) kills `an_arrival_order_cannot_change_what_grows`
  and nothing else.
* **A fixture that rebuilds the world hides the defect it was built for.** The first draft
  of that arm built a fresh `TerrainData` per arrival, which re-stamps *every* tile, so
  every key moved and the `rings = 0` mutation passed perfectly. It pages into **one**
  `TerrainData` through the streamer's own door now.
* **A `#[serde(skip)]` field reaches no state fold**, so two hosts growing different
  forests compared equal at every step for ever. The drive folds the population separately.
* **Out and back is what makes the ground page OUT.** The drive turns round half way with
  a slow z drift so no two steps stand in one place, and the tile set is asserted to have
  both grown and shrunk.

### Clause 3 — `render (record)` NAMED, and it was one function (**done**)

**THE ANSWER WAS `O(pages²)` IN A LOOP NOBODY HAD MEASURED.** Wave I7 recorded
`render (record)` at **10.874 ms against a 6.657 ms GPU frame** and routed it as
unattributed. The record profile below named it on the first run:

| shipped island, before | ms | % of the record stage |
|---|---|---|
| **cluster plan (+ wants)** | **10.051** | **90.1 %** |
| submit | 0.475 | 4.3 % |
| view uniforms | 0.421 | 3.8 % |
| graph | 0.194 | 1.7 % |
| everything else | < 0.01 | — |

The mechanism, once the phase was split in two: **`VgeomNode::cluster_tile_wants`
asked `VgeomSource::with_page_sections` for one page at a time, and that function
parses the payload — header, bounds checks and the whole page directory, with a
`Vec` allocation — on every call.** So a virtualized mesh with N resident pages
re-parsed its own directory N times a frame to read sections it had already
walked past: `O(N²)` per frame, on a world holding **one** virtualized mesh.

`VgeomSource::for_each_page_sections` parses **once** and walks. Nothing else
changed; the counters are the same additions in the same order (they are applied
outside the walk now because it holds a borrow).

**Measured, RTX 4070 Ti, 1080p, MIN of 3 rounds × 120 frames, the shipped island:**

| | before | after |
|---|---|---|
| p50 / p95 / p99 | 24.080 / 26.827 / 27.634 ms | **3.587 / 3.841 / 4.128 ms** |
| fps at p50 | 41.5 | **278.8** |
| `render (record)` | 11.151 ms | **2.037 ms** |
| of which the cluster plan | 10.051 ms | **0.004 ms plan + 1.313 ms wants** |
| GPU frame | 12.182 ms | **1.133 ms** |
| pipelined estimate | 12.182 ms (82.1 fps) | **2.354 ms (424.8 fps)** |
| distance from 60 fps | p50 **+7.480** | p50 **−13.014** |

**The GPU column moved too, and that is the I4b power-state law, not a second
fix**: a frame whose CPU took 11 ms left the card idle two thirds of the time and
an idle card downclocks. `terrain` reads 5.886 → 0.555 ms across the pair. A GPU
column is comparable only between runs whose CPU frames are comparable — the
instrument's own header says so, and this is the second wave to pay for it.

The lit frame moved by the CPU half only: `render (record)` **17.526 → 10.347 ms**
(cluster wants 12.515 → 3.851), p50 **52.020 → 45.702 ms**, and the GPU frame is
**unchanged at 33.5 ms** because it is 29.9 ms of `vsm-raster` — clause 2's.

### Clause 3's instrument (the thing that named it)

`inf_render::timing::RecordProfile` / `RecordClock`: **15 named phases that tile the whole
of `EngineRenderer::render`**, the record-path twin of `inf_player::step_profile`. It
exists because the per-pass `PassTime::cpu_ms` column tiles only the **marked span** —
`FrameTimer::begin` opens at the frame's first *command*, so the view matrices, the three
per-frame uniform writes, the encoder, the cluster plan and the submit are inside a
caller's `render (record)` clock and inside no segment. The I4b audit measured that residue
at two thirds of a 3.0 ms stage and printed it as one unattributed number; wave I7's island
measured the same stage at **10.874 ms against a 6.657 ms GPU frame**.

The clock carries `mark_at` — the clock-free seam — from the day it was written, and its
arms drive the arithmetic with decided timestamps: the I7 CI-red was exactly this shape,
and it went red on a shared runner rather than in review.

The instrument prints the phases dearest-first and **asserts they tile the record stage**
(the two are the same call measured twice). It also prints `VsmRasterStats::summary()` and
the per-rastering-frame pages / draws / casters / invalidation touches, because a
millisecond count with no page or caster beside it cannot say whether the cost is the
drawing or the asking.

### Clause 2 — the VSM caster pack: THE ROUTED PRESCRIPTION WAS BACKWARDS

I4b routed *"cache the caster pack, keyed on CONTENT, with the floating origin in
the key"* off the composed city, where `vsm-raster` recorded **6.05 ms of CPU**.
The island's `vsm-raster` is **30 ms of GPU**, and the record profile above puts
the whole caster pack — packing, the invalidation scatter and every upload — at
**0.926 ms of CPU record**. Caching it cannot touch a 30 ms GPU pass.

**And the dirty split closes it.** `dirty_pages` alone cannot tell *"the world
moved under the pages"* from *"the pages moved under the world"*, so
`VsmRasterStats` grew three counters that sum to it exactly. Per rastering frame
on the lit island:

| | |
|---|---|
| **re-cast** (something under an unmoved page changed) | **0.0 — every frame** |
| **moved** (the page's own matrix) | 532.0 |
| **re-slotted** (the atlas re-assigned the slot) | 400.8 |
| pages rastered / deferred / cached | **256.0 (the ceiling) / 676.8 / 91.2** |
| draws / casters / invalidation touches | 1 621 / 193 / 168 821 |

**Nothing under a shadow page on this island ever changes.** The 168 821
invalidation touches a frame — the number I4b's routing wanted to spend a cache
on — invalidate **zero** pages. The whole 30 ms is the clipmap grid shifting
under a camera travelling **0.9 m a frame** against a level-0 page **1.0 m**
wide (`2 x first_level_extent_m / clipmap_pages_per_side` = `2 x 32 / 64`). A
page 1 m across is 128 texels over 1 m — **7.8 mm a shadow texel**, four to eight
times finer than any pixel a 40 m-high 1080p camera can show. The cache never
converges: 256 rastered (the cap) and **677 deferred** every frame, for ever.

**The alternative, priced and printed in the instrument** rather than argued —
a third configuration with the first clipmap level widened 4x, which quarters
every level's snap rate:

| lit island, 1080p | shipped ladder | first level x4 |
|---|---|---|
| p50 / p95 | 45.233 / 49.938 ms (22.1 fps) | **25.086 / 44.450 ms (39.9 fps)** |
| GPU frame | 33.563 ms | **19.707 ms** |
| `vsm-raster` | 30.001 ms | **15.931 ms** |
| pages rastered / deferred / cached | 256.0 / 676.8 / 91.2 | **173.2 / 63.5 / 503.2** |
| dirty: re-slotted / moved / re-cast | 400.8 / 532.0 / 0.0 | **11.6 / 225.1 / 0.0** |

The cache *starts working* (91 cached → 503) and the deferral queue *drains*
(677 → 64). **Reported, never shipped**: `first_level_extent_m` is a product
decision about shadow sharpness, and the wave that measures an alternative is not
the wave that gets to pick it.

**ROUTED BY NAME — the real fix is the clipmap SCROLL**, and `vsm_raster.rs`'s
own cache-field doc already names it: *"When a clipmap level's grid shifts, the
world cell that was page `(x, y)` becomes page `(x − 1, y)` with a
**bit-identical** matrix, so the label changes while the content does not… the
'there is no clipmap scroll' remainder wearing the cache key."* A level that
scrolls by one page keeps **63 of its 64 columns** — about 98 % of the "moved"
pages hold depth the atlas already has, in a slot that now answers to a different
label. Relaxing the cache key does **not** reach it (measured reasoning: the
texels are in another slot, and a stamp-only key asks the wrong slot); it needs
residency to keep the slot with the **world cell** rather than with the page
label, which is `inf_vsm`'s to own and touches every P27 invalidation arm. That
is a phase-sized item, and at ~98 % of 933 dirty pages a frame it is the one that
would take `vsm-raster` from 30 ms to single digits.

**Also routed, with its number**: `cluster_tile_wants` is now the dearest record
phase at **1.23 ms shipped / 3.37 ms lit**, and it is the same class the clause-3
fix belonged to — a per-frame rebuild keyed on nothing. It clears and re-couples
every resident page every frame. It is deliberately NOT taken here: the P28.2
ledger records that getting this pairing wrong *retracts an asset for ever*, and
1.2 ms of a 3.5 ms frame that is 13 ms inside budget does not buy that risk.

### Counts

| | after the I7 audit | **after I7b** |
|---|---|---|
| battery blocks / passed / failed / ignored | 318 / 5 952 / 0 / 16 | **319 / 5 968 / 0 / 16** — **+11 arms are this wave's** (7 in `inf-pcg`'s binding, 4 in `inf-render`'s timing) and **no test file or crate was added**, so the +1 block and the other +5 arms cannot be. Recorded as measured and flagged rather than claimed away — the I4b `+1 rustdoc` precedent. The likeliest cause is the recorded 318 having been read off a truncated log, which is a mistake this wave made once itself before re-running for the whole thing |
| frontend tests / files | 702 / 78 | **702 / 78, not re-run** — no file under `editor/studio/src` was touched *(the I7b audit corrects this row: the wave wrote `editor/studio`, and it did touch `editor/studio/src-tauri/src/commands/pcg.rs`. That is Ring-2 **Rust**, which the battery covers; the frontend is what was not re-run, and it is what was not touched)* |
| goldens | 54, byte-identical under `INF_GOLDEN_STRICT=1` | **54, byte-identical**, re-run under `INF_GOLDEN_STRICT=1` over **101 arms** with **no PNG rewritten** (`git status` on `tests/goldens/` is empty) |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** (local toolchain, run LAST per the rmeta law). One finding was cleared on the way and it was this wave's own: `!(span > 0.0)` is a negated comparison on a partially-ordered type |
| rustdoc warnings (ceiling 450) | 404 over 46 crates | **404 `^warning` lines − 30 summaries = 374 individual over 30 crates**, measured after `cargo clean --doc`. **The wave adds zero**: it introduced exactly two (a public doc linking the private `RecordClock::mark_at`, and a `[`OffsetTerrain`](crate::fields::OffsetTerrain)` that became a *redundant* explicit target the moment this wave imported the type into scope — a warning caused by an `use` line, which is worth knowing about) and both were found and removed. The only warnings left pointing at a file this wave touched are `binding.rs`'s four P19.3 **module**-doc links, which predate it. *The recorded 404 and this 404 are the same number by coincidence of method: this measure's `^warning` line count is 404 and its individual count is 374, so whichever convention the earlier figure used, the delta is zero* |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | **unchanged — no schema moved** |
| committed samples | 23 levels | **23**, byte-unmoved |
| new crates / new external deps | — | **none** |

### Decisions (I7b's, binding on later waves)

* **A biome-bound population is a function of the RESIDENT ground, evaluated tile
  by tile.** Not of a bounding box, and not once at load. `BiomeBinding::
  refresh_resident` is the one door all three callers (the load pass, the fixed
  step and the editor's evaluate command) go through, and `.xz_bounds()` is banned
  from both hosts by name. *Why per tile is exact:* the scatter lattice is
  world-anchored and its region clip is half-open, so the union of two abutting
  boxes is the population of their union, instance for instance.
* **A per-tile memo keys on the tile AND its neighbours.** The evaluation reads
  across a tile's edge — up to `MAX_FEATHER_SAMPLES` lattice rings of biome mask
  and one central-difference step of slope — so a tile evaluated alone is not the
  same tile evaluated inside its neighbourhood, and a key that ignores that
  memoizes *the residency the tile first arrived under*. `scatter_reach_m` bounds
  the reach for any document and `neighbour_rings` sizes the ring count from it.
* **A `#[serde(skip)]` field reaches no state fold, so a PIE == shipping gate over
  it must fold it itself.** Two hosts growing different forests compared equal at
  every step for ever. The same sentence applies to every derived cache the
  projector reads and the snapshot does not.
* **A drive that only goes out cannot show anything streaming OUT.** The island
  gate turns round half way, with a drift so no two steps stand in one place.
* **The record path gets phases, and they tile the whole call.** `PassTime::
  cpu_ms` tiles the *marked span*; the marked span starts at the frame's first
  command, so everything before it and after it was unattributed by construction.
  A stage a diagnostic cannot decompose is a stage nobody can optimise, which is
  the shape the certification found on the GPU side and wave I4b found on the sim
  side.
* **A parse in a loop is `O(N²)` and reads as one slow function.** `with_page_
  sections` parses the whole asset to read one page's sections; used per page it
  was 90 % of the shipped island's record stage. Any door of the shape *"give me
  one element, and to find it I will re-derive the index"* needs a plural twin
  before it is called in a loop.
* **A GPU column is comparable only between runs whose CPU frames are comparable**
  (I4b's law, met again in the other direction). Fixing 9 ms of CPU took the
  island's *GPU* frame from 12.182 to 1.133 ms because the card had been idling
  two thirds of every frame and downclocking. Quote both columns or neither.
* **An unmeasured prescription can be backwards, and this wave met one that was.**
  The routed caster-pack cache aims at CPU; the island's `vsm-raster` is GPU, its
  caster pack is 0.926 ms of record, and the content that cache would key on
  invalidates **zero** pages a frame. Measure before implementing a routing,
  including a routing this repository wrote itself.
* **Price an alternative in the tree, not in a memo.** The coarse-clipmap
  configuration is a third row the instrument measures every run, so the number
  cannot go stale and nobody has to trust a paragraph.

### THE NINETEENTH chr(92) CATCH, and it is this wave's own

A scripted edit through a Python heredoc ate a `\`-continuation and left **ten spaces**
inside an `assert!` message — the third wave running, and the second time the *same*
mechanism has done it. Repaired through the Edit tool, which is what the law prescribes and
what the heredoc bypassed. **Scripted edits to Rust string literals go through the Edit
tool, full stop.**

### The island's own frame numbers, RE-RECORDED

RTX 4070 Ti, release, MIN of 3 rounds × 120 frames, 1080p, the same 40 m-high
flight east from Harbour City wave I7 measured. **Reported, never asserted** —
the ceilings in `inf_player::budget` are set from the composed city.

**Three runs, and the ranges are quoted rather than the best end** (the I4b
audit's own law about this file):

| | p50 | p95 | p99 | GPU frame | pipelined estimate |
|---|---|---|---|---|---|
| **SHIPPED**, wave I7 | 18.209 | 19.287 | 20.887 | 6.657 | 10.994 (91.0 fps) |
| **SHIPPED**, I7b's head *(after the audit's terrain fix)* | 24.080 | 26.827 | 27.634 | 12.182 | 12.182 (82.1 fps) |
| **SHIPPED**, after I7b | **3.56–3.62** | 3.83–7.25 | 4.13–7.51 | 1.07–1.43 | **2.26–2.35 (427–443 fps)** |
| **LIT**, wave I7 | 48.170 | 53.590 | 56.862 | 31.188 | 31.188 (32.1 fps) |
| **LIT**, after I7b | **44.48–45.70** | 48.04–50.94 | 49.41–55.11 | 33.56–33.58 | 33.56–33.58 (29.8 fps) |
| **LIT**, first clipmap level ×4 *(priced, not shipped)* | **25.09–25.16** | 44.45–44.52 | 45.65–46.48 | 19.69–19.71 | 19.69–19.71 (50.7–50.8 fps) |

*The wave I7 row and the I7b-head row are **not** the same tree: the I7 audit put
the terrain at `Transform::IDENTITY`, so the head this wave started from draws a
world the I7 row never did. The honest before/after for this wave is the second
row against the third.*

**Distance from 60 fps, shipped: p50 −12.99 to −13.07, p95 −9.35 to −12.87 ms.**
The shipped island frame is **inside the 60 fps budget with about 13 ms to
spare**, where the wave opened at **+7.480**. The lit-with-virtual-shadows
configuration is **not** the shipped default (`VsmSettings::default().enabled` is
`false`); it is at 22.5 fps and its whole cost is the one mechanism clause 2
measures above.

CPU, shipped, after: sim step 0.163 · stream sync 0.099 · projection 0.055 ·
**render (record) 2.026–2.037** · poll 1.256 · readback 0.008. The record stage's
own phases: cluster wants 1.303 · submit 0.408 · graph 0.170 · view uniforms
0.133 · cluster plan 0.004 · everything else under 0.01.

**Content: 0 mesh instances, 1 scatter batch carrying 2 681 SCATTERED
INSTANCES**, 1 vgeom, 192 terrain tiles, 0 virtual textures — against wave I7's
*"0 instances, **0 scatter batches**"*. **Those 2 681 are the vegetation**, and
they are clause 1 arriving in the shipped island's frame:
`Terrain::biome_population` is projected through `push_biome_population`, which
goes through the same `push_scatter` body a `PcgVolume` does and produces one
batch per terrain — never `scene.instances`, which is why the mesh-instance
column stays at 0. The `scatter` GPU pass costs **0.126 ms** of the frame. The
count is ~1.3 km² of cover around the hero at the island's own 0.004 /m²
candidate density, which is what sim residency bounds it to (see below).

### Open / next

* **The instrument's island camera flies away from the vegetation.** The camera
  path is a parameter and the hero is the streaming source, so the *sim* keeps its
  residency around the start while the camera flies east — and the biome-bound
  population is a function of **sim** residency. The frame therefore carries the
  2 681 instances the hero is standing in, wherever the camera has got to.
  Closing it means the flight moving the streaming source with it, which changes
  what every previous island frame number was a number about — so it is stated
  rather than quietly changed.
* **The vegetation is bounded by SIM residency**, which is ±2 tiles around each
  terrain observer (`SIM_MARGIN_TILES`), i.e. a ~1.3 km² carpet around the player
  on the full island. That is the correct home for it — both hosts must agree, and
  sim residency is the only residency both hosts share — but it means the *render*
  cut draws ground with no cover on it past that radius. A render-side population
  off the camera's own cut would be a second authority and is not taken.
* **The EDITOR's authoring viewport still shows no vegetation on a streamed terrain**, and
  that is pre-existing rather than this wave's: the document's `Terrain.data` is empty for
  a streamed terrain (`inf_editor_core::terrain_stream` keeps its own working set) and the
  projector reads `biome_population` off the document component. The fix is the editor
  streamer's twin of `refresh_biome_bindings` over *its* resident set, and it would make
  the mirror gate three-sided. Routed by name, not taken.
* **`inf_editor_core::simulate::SimSession` streams nothing at all** — no cells, no terrain
  — so an editor Simulate of a streamed island stands on no ground and grows nothing.
  Pre-existing and unchanged by this wave; it is why `island_gate`'s "editor side" is the
  loose document through `RuntimeSim`, which is the pair P16.5's own gate compares.

## The I7b audit (2026-08-25)

Adversarial, `545614f..48b4f6e6`, fresh reader, nothing pushed. **No HIGH.** Every
headline number in the wave's ledger reproduced on this machine — including the two
that retire a routing and the one that closes a wave-old zero — and the mutations
the wave claims all reproduce as claimed.

What the audit found instead is a single shape, four times: **a claim that is true
and a gate that cannot tell.** Three of clause 1's and clause 2's load-bearing
properties were unfalsifiable as shipped — the union-is-the-whole argument, the size
of the memo's neighbourhood, and the `0.0 re-cast` that retires I4b's prescription —
and the fourth is a behaviour the `O(N²)` fix changed while its commit message said
*"nothing else moved"*. All four are fixed; nine LOWs are carried by name.

### What reproduced, measured here rather than read

`island_gate`, this tree, `cargo test -p inf-player --test island_gate -- --nocapture`:

| the wave's figure | mine |
|---|---|
| fully paged, by hand | **4 958** instances over 2.359 km² |
| shipped boot | **0 → 2 339** on 16 sim tiles |
| the 900-step drive | **2 339..3 119** on **16..20** sim tiles |
| stray against the fully-paged reading | **0 of 2 339** |
| a fully-resident interior tile | **225 against 225** |
| PIE == shipping | 900 of 900 steps, states **and** forests |
| the drive really streamed | 1 activation / 1 deactivation, 16 → 20 page loads |

`the_island_at_shipping_resolution`, RTX 4070 Ti, release, 3 × 120 frames:

| | the wave | mine |
|---|---|---|
| SHIPPED p50 / GPU frame | 3.56–3.62 / 1.07–1.43 ms | **3.464 / 1.110 ms** |
| SHIPPED `render (record)` | 2.026–2.037 ms | **1.980 ms** (phases sum 1.980) |
| LIT p50 / GPU frame | 44.48–45.70 / 33.56–33.58 ms | **45.245 / 33.461 ms** |
| LIT `vsm-raster` GPU | 30.001 ms | **29.911 ms** |
| LIT dirty per rastering frame | 400.8 / 532.0 / **0.0** | **400.8 / 532.0 / 0.0** |
| LIT pages rastered / deferred / cached | 256.0 / 676.8 / 91.2 | **256.0 / 676.8 / 91.2** |
| scattered instances in the frame | 2 681 | **2 681** |
| LIT-COARSE p50 / GPU / `vsm-raster` | 25.09 / 19.69 / 15.931 ms | **21.995 / 19.714 / 15.925 ms** |

Two rows land **below** their quoted range and both do so on the favourable side:
SHIPPED p50 **3.464** against 3.56–3.62, and LIT-COARSE p50 **21.995** against
25.09–25.16. Both of those configurations' GPU columns are inside their ranges, so
what moved is CPU scheduling on a fourth run rather than anything about the tree —
which is also the honest reading of the wave's own ranges: they are the spread of
*those three* runs, not a bound. The dirty split sums exactly:
`191 562 + 254 296 + 0 = 445 858 = dirty_pages`.

**And the retirement is honest.** The routed caster-pack cache is retired in the
ledger *and* in the ROADMAP, and the item does not vanish — it becomes the named
clipmap-scroll item with its own price. The ×4 clipmap configuration really is
unreachable from any shipping path: `first_level_extent_m` has exactly **one**
assignment in the whole tree and it is `fps_instrument.rs`'s local; `settings.rs`'s
`32.0` default is untouched by the range. The 256-page raster ceiling is **loud in
the shipped path**, not only in the instrument — `tracing::warn!` fires on every
deferral, 677 of them a frame.

### MED 1 — the union-is-the-whole arm never split a scatter cell (fixed, `59f3d70`)

`a_per_tile_walk_places_exactly_what_one_region_places` carries the whole exactness
argument, and that argument is about cells that **straddle a tile edge**. The
fixture's cover document used `cell_size` 8 m against a tile span of 8 m, so every
cell fell whole inside one tile and no straddle ever happened. Mutation-measured: a
**region-anchored** lattice (`cell_x += region.min.x.rem_euclid(cs)`), which breaks
unionality outright, passed **all 227** arms of `inf-pcg`. At 3 m it fails the arm
and nothing else. The shipped island is aligned the same way — 32 m scatter cells
inside a 256 m tile span — so the property the crate has to keep is the one an
author reaches the moment they type any other cell size.

### MED 2 — the neighbours key was armed against `rings = 0` and nothing else (fixed, `59f3d70`)

The wave's own mutation reproduces exactly: `rings = 0` kills
`an_arrival_order_cannot_change_what_grows` and nothing else, in this crate or in
`island_gate`. But `rings = 1`, on a fixture whose own `neighbour_rings` is **9**,
passed every arm in the tree — and an *under-sized* neighbourhood is the likelier
defect (a `ceil` that became a `floor`, a reach read off the wrong spacing, a cap
that shrank), and it puts first sight back silently.
`a_tile_a_whole_reach_away_re_keys_the_tile_it_can_be_read_from` drives the shipped
door over a strip and reads its **engagement counter**: a tile arriving at Chebyshev
distance exactly `neighbour_rings` must re-key the tile it is that far from, so
`tiles_evaluated` goes **1 → 3** and not 1 → 2. Mutation-verified as the only arm
`rings = 1` fails.

*The reach bound itself is sound and was checked rather than taken:*
`BiomeMask::nearest_unlike` is the **only** offset reader among the samplers
(`AltitudeFilter`, `DataMapMask`, `Mask`, `Noise` are point reads) and its radius is
capped at `MAX_FEATHER_SAMPLES` *whatever feather an author writes*, so a feather
wider than a tile cannot escape it; the other reader is `FnHeight`'s central
difference. `ceil(reach / span)` is exactly the Chebyshev radius a candidate at a
tile's far edge can reach, corners included, because the neighbourhood is a **square**
and not a cross.

### MED 3 — the `O(N²)` fix un-declared a coupling group (fixed, `4a4b50c`)

The loop `for_each_page_sections` replaced declared the coupling group **outside**
the `with_page_sections` call, so a page whose sections did not come back was still
coupled with an **empty** member list. The rewrite skips such a page, and declares
nothing at all when the payload itself is unavailable. `inf_stream::Coupling`'s own
`has_group` doc says why that matters in as many words — *"a group with no members
is a legal state … while a group that was never declared is a page the want pass
never saw, which a consumer must refuse"* — and `commit_cluster_pages` refuses,
which is the P28.2 failure whose measured value is **zero pages, the mesh vanished**.
The guard's comment calls itself MEASURED unreachable *because* `cluster_tile_wants`
couples `0..resident_pages()`; after the rewrite it coupled however many pages
parsed, so the sentence had stopped being true by construction. Groups are seeded
before the walk now. Reachability is narrow (a corrupt pack entry), which is why
this is MED and not HIGH — but the invariant is documented, and it was gone.

`the_plural_page_walk_reads_exactly_what_the_singular_one_does` is the
byte-equivalence arm the rewrite went in without: same pages, same order, same tile
references through both doors on a real paired source, plus the skip that makes the
seeding load-bearing. Mutation-verified (reading page `index − 1` fails it).

### MED 4 — the split that retired a routing had no arm (fixed, `79e9792`)

Three counters, zero tests. Nothing in the tree ever drove the `Casters` branch —
the one whose `0.0` is the whole argument for retiring I4b's content-keyed
caster-pack cache. A branch unreachable by construction prints the identical zero,
and inference dressed as measurement is worse than no measurement (P22).
`classify_page` is now the shipped classifier lifted where an arm can reach it, and
`the_dirty_split_names_all_three_reasons_and_tiles_the_dirty_set` drives all four
cache states including the one the routing hangs on — same slot, same label, **same
`geo_key`, a different whole `key`** — plus the sum identity, which `record` also
keeps as a `debug_assert_eq!` over the live split. Mutation-verified: making
`Casters` unreachable fails the new arm and nothing else, which is the state the
counters shipped in.

### The mutations run, and what each killed

| mutation | what went red |
|---|---|
| `rings = 0` in `refresh_resident_in` | `an_arrival_order_cannot_change_what_grows` — **and nothing else, in `inf-pcg` or in `island_gate`** (the wave's claim, reproduced) |
| `rings = 1` (an under-sized neighbourhood) | nothing, before; the new reach arm, after |
| a region-anchored scatter lattice | nothing (227/227 green), before; the union arm, after the fixture's cell size stopped dividing the tile span |
| a closed (`>`) region clip instead of half-open | **nothing, either way** — a jittered candidate landing exactly on a boundary is measure-zero in f64, so the half-open rule is a correctness statement no fixture can reach. Recorded, not armed |
| `for_each_page_sections` reading page `index − 1` | the new plural-walk arm |
| the `Casters` branch made unreachable | nothing, before; the new dirty-split arm, after |
| the `submit` + `epilogue` record marks deleted | `print_record_profile` — *"the record phases sum to 1.610 ms beside a 2.016 ms record stage"* |
| `refresh_biome_scatter()` deleted from `fixed_step` | `pie_equals_shipping_on_an_island_drive` — *"one forest for the whole drive"* (the boot still seeds through `set_terrain_streaming`, so the load half stays green, which is the right division) |
| the drive made one-way at the same distance (`out = step / 2`) | the shrink arm — *"SIM TILES: grew true, shrank false"*. **The wave's justification for the turn stands**: a one-way drive at *twice* the distance passes it, so the turn is what makes the ground page out at **this** drive length |

### Carried LOWs, by name

1. **`island_gate`'s drive samples two distinct forests and four page loads.** 900
   steps × 0.4 m out and back is 180 m each way against the fixture's **256 m** tile
   span, so the hero crosses one tile edge and comes back. The mechanism is proven;
   the sample is thin. Deepening it re-numbers every figure in this wave's ledger, so
   it is named rather than changed.
2. **`island_gate` does not arm the neighbours key at all.** `rings = 0` — the
   first-sight defect the design exists for — passes all six arms, because both hosts
   page in the same order and the boot pages its 16 tiles in one batch. Gate-level
   first-sight coverage is nil and rests entirely on the `inf-pcg` unit arm.
3. **`VgeomNode::cluster_tile_wants` has no end-to-end arm of any kind** (pre-existing).
   `cluster_pages.rs` re-derives the coupling by hand rather than calling it. The
   seeding above is by construction and the skip is armed; the call itself is recorded
   as unarmed rather than counted as covered.
4. **The lit record stage's dearest phase is `vsm sync`, and no routing names it.**
   Measured here: **3.681 ms (36.3 %)**, ahead of `cluster wants` at 3.523 ms — and the
   ledger's routing paragraph quotes only the latter. It is the same mechanism clause 2
   measures, and the third configuration proves it: dirty pages 932.8 → 236.7 (3.94×)
   takes `vsm sync` 3.681 → 0.871 ms (4.23×). So the clipmap scroll costs **~4.6 ms of
   CPU record on top of 29.9 ms of GPU**, and the routed fix would take both.
5. **The editor's Simulate host is not one of the three callers.** Pre-existing, named
   in the wave's own *Open / next*; `island_gate`'s "editor side" is the loose document
   through `RuntimeSim`, which is the pair P16.5's gate compares. The "both hosts"
   claim is about two `RuntimeSim` boot paths and should be read that way.
6. **`EngineRenderer::record_profile()` keeps the last armed frame's values** after
   `set_gpu_timing(false)` rather than zeroing. Harmless in every shipped frame, where
   it is never armed at all.
7. **The mirror gate's new requirement is `>= 1` occurrence of a spelling**, so a doc
   comment naming `refresh_resident(` would satisfy it. The load-bearing half is the
   `.xz_bounds()` ban, which is a real ban; the positive half is a byte pin that cannot
   see semantics (P23's law).
8. **The counts table's "no file under `editor/studio` was touched" is false** —
   `editor/studio/src-tauri/src/commands/pcg.rs` was. The *intent* (no frontend file,
   so no `npm` run) is right, and is corrected in the row below.
9. **The phase-state table's I7b row was orphaned from the table** by a stray blank
   line, so it rendered as a second one-row table. Repaired.
10. **"Committed samples 23" derives from nothing.** `samples/` holds **19**
    `.inf_lvl` files and **20** project directories, and no arm anywhere counts
    either. The figure has been carried unchanged since at least I6 and nobody has
    re-derived it. The load-bearing half *is* verified: `git diff` over `samples/`
    across `545614f..48b4f6e6` **and** across the audit's own commits is empty, so
    "byte-unmoved" is true whatever 23 counts.

### The chr(92) sweep

**No twentieth catch.** Every run of four or more interior spaces on an added line in
`545614f..48b4f6e6` is deliberate: the `//     I7b)` comment block matching the
existing `// 0a.` / `// 0b.` indentation, `print_record_profile`'s four-space row
indent, and the `"  content    "` column alignment that predates the wave. The
workspace sweep
(`no_string_literal_in_the_workspace_carries_an_eaten_continuation`) is green.

### One-platform hazards

None introduced. The range adds no trigonometry, no `libm` route and no `f32`
committed content; `neighbour_rings`'s `ceil` and `scatter_reach_m`'s multiply-add
are exact IEEE operations on `f64`. `FnHeight::new`'s literal now reads
`FN_HEIGHT_NORMAL_EPS`, which is the same `0.1` and therefore moves nothing.

### Counts, at the head this audit certifies

| | after I7b | **after the I7b audit** |
|---|---|---|
| battery blocks / passed / failed / ignored | 319 / 5 968 / 0 / 16 | **319 / 5 971 / 0 / 16** — `cargo test --workspace -j 3`, tallied over all 319 blocks. **The wave's 5 968 is exactly right**: the audit adds three arms and 5 968 + 3 = 5 971, and the block count is unchanged because no test file was added. That settles the wave's flagged "+1 block and +5 arms" the only way it can be settled without re-running the base: **no test file or crate was added between `545614f` and `48b4f6e6`** (checked with `git diff --diff-filter=A`), so the block count *cannot* have moved, and the recorded 318 / 5 952 baseline is what is wrong. The wave's instinct — record it as measured and flag it rather than claim it away — was right, and this run is the confirmation |
| frontend tests / files | 702 / 78 | **702 / 78, not re-run** — the audit touched no file under `editor/studio/src` |
| goldens | 54, byte-identical | **54, byte-identical** — re-run under `INF_GOLDEN_STRICT=1`, **101 arms, 0 failed, no PNG rewritten** (`git status` on `tests/goldens/` empty) |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** (local toolchain, run LAST per the rmeta law). Stated precisely: the run re-checked the **13** crates the audit's seven files reach and answered the rest from clippy's own cache, whose sources are byte-identical to the ones the wave checked green at this head |
| rustdoc warnings (ceiling 450) | 404 `^warning` lines − 30 summaries = 374 individual over 30 crates | **404 − 30 = 374 over 30 crates**, measured after `cargo clean --doc`. **The audit adds zero, proved as a set and not as a count**: diffed line by line against the wave's own cold log, the only differences are the two warnings the wave itself removed and one line-number shift from the audit's `level.rs` doc correction. The warnings still pointing at a file the audit touched are `binding.rs`'s four P19.3 module-doc links, which predate both |
| `cargo fmt --all --check` | clean | **clean** |
| schema versions | scene v25 / payload v11 / `.inf_sm` v3 | **unchanged — no schema moved** |
| committed samples | 23 | **byte-unmoved** across the wave *and* the audit; see carried LOW 10 for what 23 counts |
| new crates / new external deps | none | **none** |

**Five audit commits**, `(I7b) audit:`-tagged — four fixes and the ledger, counted
rather than summarised.

## Done — wave SK1a (the 161-bone skeleton substrate)

Base `9790ff8e`. Five clauses. The wave's headline is a **correction**: the source
document this wave was briefed from is wrong about its own subject, and the right
answer was sitting on this machine.

### THE SOURCE DOCUMENT IS WRONG, AND THE ASSET IS NOT

`character-skeleton-rig-outline.md` (the briefing document, which lives outside
the repository with the rest of the reference material) prints a hierarchy under
the heading
*"the complete hierarchy for the **161 bones** of the full UE5 mannequin"*. Parsed
and counted, that tree is **89 bones** — which is exactly the figure the same
document's own table two sections earlier gives for `SKM_Manny_**Simple**`. It
also states parents the shipped asset does not have.

So the table this wave emits was **measured**, not transcribed. The UE5
reference project named in the anim/island mandate memo ships
`Characters/Mannequins/Meshes/SK_Mannequin.uasset`, whose `FReferenceSkeleton`
was parsed straight out of the package body: a count field of **161**, then 161 records of
`(FName name, int32 parent, FString exportName)`. One root, every parent preceding
its child, no duplicate names. Six disagreements with the document, all printed in
`crates/inf-anim/src/manny.rs`'s module docs rather than quietly resolved:

| | the document | the asset |
|---|---|---|
| bone count of the printed tree | claimed 161 | **89** |
| neck | `neck_01` → `head` | `neck_01` → **`neck_02`** → `head` |
| IK subtree parent | `spine_03` | **`root`** |
| `ik_hand_l` / `ik_hand_r` parent | `ik_hand_root` | **`ik_hand_gun`** |
| `ik_head`, `ik_pelvis`, `ik_spine` | present | **absent** |
| corrective / helper bones | absent | **74**, `weapon_l/r`, `interaction` and `center_of_mass` among them |

**What is Epic's and what is ours.** Names, parent links and emission order are
the interchange contract and are reproduced verbatim — they are the whole point,
since `thigh_l` / `spine_03` / `ik_foot_root` are what every Mixamo clip, every
MetaHuman body and every ALS blueprint addresses. **Every offset is a proportion
multiplied by a `BodyParams` length** — no absolute dimension is copied, so the
rig is proportionally whatever height it was asked for. *Where those proportions
come from is three answers, and the SK1a audit's correction is that this ledger
originally gave two:*

1. **Invented** — the torso, the limbs, the girdles, the hand's overall size.
   These are this engine's own numbers, shared with `template.rs`, and they do
   *not* reproduce the shipped rig's: the asset's hand is 0.096 of its height and
   `HAND_OF_HEIGHT` is 0.105.
2. **Rules read off the asset** — the twist bones sit at **exactly one third and
   two thirds** of their segment, and `_01` is always the one nearest the joint
   that drives it (`upperarm_twist_01_l` at 1/3 from the shoulder;
   `lowerarm_twist_01_l` at 2/3 from the elbow, i.e. 1/3 from the wrist). Both are
   load-bearing for the drive law.
3. **Measured proportions** — the nineteen bones of the **hand**. Their offsets
   are the reference skeleton's own local bind translations, normalized so the
   middle-finger chain sums to 1.0 of hand length (verified: `middle_metacarpal`
   is 3.375833 cm over a 17.117 cm chain = the committed 0.1972). Fifty-seven
   numbers, per-bone rather than per-class. Kept, because there is no *rule* for
   where a pinky metacarpal sits relative to a ring one — it is anatomy, and
   inventing it makes a hand nobody's glove fits — and because a ratio of a length
   this module chooses carries no dimension. Named here rather than left reading
   as though it were derived; `manny.rs`'s module docs carry the same split, and
   `the_two_hands_are_an_exact_mirror_of_each_other` is the arm those 57
   hand-maintained numbers did not have.

The census, asserted: **63 deform + 16 twists + 7 IK handles + 74 helpers + 1
root = 161**.

### Clause 1 — `BodyPlan::Biped` is the mannequin (`878d3169`)

`BodyPlan::BipedCanonical` is the twenty-joint rig `Biped` used to be, and it is
kept *generable* for three reasons that are one reason: it emits exactly
`humanoid_joint_names()`, so it is the rig the canonical vocabulary is defined by;
every committed clip in this repository is index-bound to it; and a small rig is
the right fixture for a test about something other than bone count. Every existing
fixture that pinned the old vocabulary moved to it, one line each.

* The count arm asserts 161 **with** the invariants that make the number mean
  something: one root, every parent preceding its child, no duplicate name, one
  role row per joint.
* Emission order puts every deform bone ahead of every `ik_*` handle, asserted —
  the belt beside the role table's suspenders, because the last site in the engine
  that matches `contains("foot")` and takes the first hit finds `foot_l` before
  `ik_foot_l` *because of this ordering*.
* The bind pose is a **T-pose of pure translations**: identity rotation and unit
  scale on every joint, so an inverse bind is an exact negated translation and no
  `sin`/`cos` appears (the P14 law). That forces the arms out along ±X rather than
  into the shipped mannequin's A-pose — an A-pose is rotation, and rotation in a
  bind pose is what this generator does not do. The consequence is small and good:
  every limb axis is an exact unit basis vector, which is what makes the drive
  pass's twist axes exact.
* Sockets: **both families**. The engine's own six (`hand_l`, `hand_r`, `foot_l`,
  `foot_r`, `head`, `back`) and the ALS six (`hand_l_socket`, `hand_r_socket`,
  `FX_Foot_L`, `FX_Foot_R`, `head_socket`, `root_socket`). Publishing one and not
  the other would make every ALS port a rename.
* The 74 helper bones are emitted **at their parent's origin** and carry
  `BoneRoleKind::Helper`. They exist so an externally authored clip or a retarget
  finds every bone it names; nothing in this engine drives them, and a bone at its
  parent's origin is a bone with no influence.

### Clause 2 — `.inf_skel` v2 → v3, the wave's one bump (`878d3169`)

Spent once, carrying every tail append together: `roles`, `twists`, `ik_follow`,
`grips` (empty on every rig this wave makes — present so the hand solver does not
need a second bump), and `JointLimit::cone`, which that type's own docs had
already named as *"an append behind another bump"*.

**v2 is migrated, not refused**, and the difference from the v1 rung is what makes
that honest: v1 stopped short of a table whose contents could not be invented,
while a v2 file's four empty tables are exactly what a v2 rig *meant*. The frozen
v2 shadow is spelled ladder-locally and independently twice, because a shape
derived from the live encoder pins nothing.

Measured, arithmetic-verified:

| | before | after | delta |
|---|---|---|---|
| `character-demo/Character.inf_skel` (0 limits) | 684 B | 688 B | **+4** = four empty `Vec` prefixes |
| `phase29-locomotion/Hero.inf_skel` (4 limits) | 2 687 B | 2 695 B | **+8** = the same four, plus one `Option<ConeLimit>` tag per limit |

The stamp byte goes 2 → 3. On `character-demo`, which authors no limit, **every
other byte is identical** and the four new prefixes are a pure tail append; on
`Hero` the four `Option` tags are inserted *inside* the limits `Vec`, which is
mid-stream, so the 75 bytes after it shift — the delta is still exactly +8 and no
rig content moved, but "every other byte identical" is true of one file and not
of the other (SK1a audit). Sidecars carry only
their content hash. **No `.inf_anim`, no `.inf_sm`, no `.inf_act`, no `.inf_lvl`
moved** — `phase29_spec` is pinned to `BipedCanonical` so that course keeps the rig
its clips are index-bound to.

A side table naming a joint the rig does not have is **refused at the door, by
name**. Not a panic anywhere (every reader bounds-checks), which is exactly why it
needs catching: the failure it produces downstream is a twist that never drives
and a role lookup that finds nothing, both silent.

The same door refuses a role or IK-follow table that is **not strictly ascending
by joint**, and that check exists because of a defect this wave introduced and
then removed: `RoleIndex` originally *owned* its rows, which made `role_index()` a
161-row clone and a sort — and `foot_states`, `apply_foot_ik` and the pelvis drop
each want one per posed character per fixed step. A per-frame rebuild keyed on
nothing, which is the exact shape wave I7b spent a clause removing from the render
path. The index borrows now; a borrowed index cannot sort what it is handed; so
the invariant moved to the door, where an out-of-order table is a named refusal
rather than a binary search answering `None` for a row that is really there.

### Clause 3 — the procedural drive pass (`878d3169`, `9d22d1b7`)

`inf_anim::drive` is one Ring-0 rule for the two families of bone no clip authors.

**The law, one sentence, both signs:** *the roll along a limb segment is linear in
the position along it.* A twist bone is a child of the segment, so it already
inherits the whole roll. An **upper** segment is rolled by its own joint at the
proximal end, so a bone at fraction `p` gives **back** `1 − p` — a negative
fraction whose source is the segment itself. A **lower** segment is rolled by the
joint at its distal end, which is a child, so a bone at `p` **adds** `p` of that
child's roll. Both are the same mechanism with a sign, and both fall out of the
measured placement.

Portable by construction: swing-twist by projection (arithmetic and one `sqrt`)
then `pslerp` from the identity. **No `sin`, no `cos`, no `atan2`, no `acos`** on
the `f32` path. The `portable_pose` source gate grew `drive.rs`, `manny.rs` and
`roles.rs` and now covers **28** files.

The IK handles follow in **ascending joint order**, recomputing one global column
at a time, because `ik_hand_l` hangs off `ik_hand_gun` which is itself a follow: a
pass that snapshots the globals once puts the child against its parent's *bind*
frame, a whole arm from where the parent just moved to. Armed, with the pairs fed
in deliberately out of order.

**Where it sits, and the bound that costs.** Called from `step_pose_evaluation`
immediately after the layer stack and before every pass that corrects the result,
because this is pose *construction* and those are corrections. It joins
`sample_clip` / `blend_poses` / `apply_layers` / `solve_chain` / `apply_foot_ik` /
the pelvis drop / the ragdoll blend as a pose writer at a fixed place in that
list, since the I6 law makes the ORDER part of the trace. The cost, stated rather
than hidden: **a twist reflects the pose the animation authored, not the pose foot
IK goes on to correct** — a foot IK solve that rolls an ankle 20° leaves
`calf_twist_01_l` showing the pre-solve roll for that frame. Fixing it means
running the pass twice or having each solver re-drive its own chain's twists;
**routed by name to SK1b**, where hand IK gives it a real consumer and a number.

**Absent costs nothing.** A rig with no drive tables takes two early returns and
poses the bytes it posed before this existed — proved end to end by a PIE ==
shipping arm on the canonical biped, which carries no side tables at all.

### Clause 4 — the role table takes the five name-guessing jobs (`878d3169`, `9d22d1b7`)

| site | was | is |
|---|---|---|
| `inf_physics::ragdoll::classify` | a keyword table | the role table; the classifier is the fallback |
| `inf_ecs::pose::foot_joints` | `contains("foot")`, first match | `BoneRoleKind::Foot` by side |
| `inf_ecs::pose::pelvis_joint` | `eq_ignore_ascii_case("pelvis")` else the root | `BoneRoleKind::Pelvis` |
| `inf_anim::derive::foot_joints` | `starts_with("foot_")` | `BoneRoleKind::Foot` |
| `inf_anim::derive::leg_name_of` | `starts_with("upper_leg_")` | `BoneRoleKind::Thigh` |

plus `inf_dcc::autofit`'s ground-plant rule, which was the sixth and was not on the
list.

**A MANNEQUIN RAGDOLLS INTO A CONNECTED BODY.** Measured in one arm against the
same rig with its table stripped, through the real door (the pose step publishes
the rig, the physics side assembles it):

| | role path | name classifier, same rig |
|---|---|---|
| parts | **17** | **92** |
| free capsules | **1** — the pelvis, which is the root | **31** |
| a chest | yes | **never** |
| every part reaches the root, walked | yes | no |
| forearm capsule span | **1.000** of the forearm | built from a twist bone (in *production*; the arm feeds both paths the role-derived spans, so it does not measure this half) |

*(The classifier column read "14" and "4" until the SK1a audit measured it: the
arm asserted only `free.len() > 1`, which two loose capsules and two hundred both
satisfy. Every `upperarm_*` corrective and twist classifies to `UpperArmL` or
`UpperArmR` and every one of them wants a `Chest` that is never produced. Both
numbers are pinned now.)*

The way the classifier fails is the quiet one: `spine_01` … `spine_05` all match
its `spine` keyword and *none* matches its `spine1`/`spine2` chest keywords — the
underscore — so **no `Chest` is ever produced**, and `Chest` is the parent role of
both upper arms and of the head. Those parts name a parent that is not in the index
and spawn with no joint at all.

Two mechanisms made the role path work, and both are worth naming:

* **`build_ragdoll` chains by INDEX**, not by role, when the rig carries a table.
  That is what lets a five-segment spine be five parts in a row instead of a
  collision of labels. Labels remain (a consumer reads `role == Hips`), and are
  documentation.
* **`rig_bones` takes its tail from the first child the table calls a deform
  bone**, not the first child. On the mannequin's own index order the first child
  of `lowerarm_l` is `lowerarm_twist_02_l`, a third of the way to the wrist.

`capsule_part` is now one function both ragdoll paths call, so they cannot
disagree about what a limb weighs — and it guards the zero-length case, because
`from_rotation_arc` on a near-zero direction is a NaN quaternion a solver
propagates into every body jointed to it, and a rig with 74 helper bones at their
parent's origin has plenty of those.

**The retarget map ships, and the silence ends.** `RetargetMap::canonical_to_manny`
(and its reverse, built from one table so they cannot drift). Measured: the
identity map on a mannequin target writes **5 joints of 161** — the two
vocabularies overlap at `head`, `hand_l`, `hand_r`, `foot_l`, `foot_r` — and says
nothing about the other 14 pairs; the pairing writes **19** and
`RetargetReport` names every joint that kept bind. Five is not zero, and that is
the trap: a nearly-vacuous retarget looks like a correct retarget of a still
character.

Two structural choices, stated: canonical `spine` pairs with `spine_01` and
`chest` with `spine_05` — the bottom and the top of the five-segment chain, so a
torso twist arrives at both ends of the back rather than being spent on one
vertebra.

### Clause 5 — the wizard end to end, and three numbers (`187166bb`)

**PIE == shipping on a 161-bone pose trace**, twice from two processes, with the
severed-machine anti-vacuity arm intact. The trace, re-priced and now asserted
(36 B header + 40 B per joint per character per step):

| rig | per character per step |
|---|---|
| canonical biped, 20 joints | **836 B** |
| quadruped, 19 joints | 796 B |
| mannequin, 161 joints | **6 476 B** |

**7.75×, not 8×** — the header does not grow. What is *streamed* between hosts is
a `u64` per step and does not grow at all; what grows is the memcpy and the hash.

**The heat solve, measured before anything was done about it** (release, 386
vertices, min of three): 24 → 249 bone segments is **10.4×**, and the solve grew
about **2.4×**. The per-bone work is real and is not what dominates at this size.

*The SK1a audit corrected the stated cause, and the correction is the interesting
part.* The first write-up said the Laplacian assembly **and the per-vertex top-4
gather** are bone-independent, and that the diffusion is one CG solve per bone.
Both halves are wrong: the gather walks every bone's field for every vertex, which
is O(V·B) and is the part that *does* scale; and a bone that wins no vertex has
`sources == 0` and **skips the solve entirely**. Measured with the count now
reported by the arm: **15 of 24 segments actually solve on the canonical rig and
19 of 249 on the mannequin** — 1.3× the solves for 10.4× the bones, which is the
real explanation of the ratio. (89 of those 249 are also zero-length leaf markers
rather than bones, against 5 of 24 on the small rig, so the segment ratio partly
measures leaf density.) A finding, not a licence: `field` is B × V × 8 bytes, so
161 bones over 50 000 vertices is ~64 MB resident and the balance tips. The arm
asserts the **upper** bound on the clock ratio and the solve count beside it — the
millisecond pair is printed rather than written down, because the wave's code
comment (32.4 → 78.3) and this ledger (31.9 → 75.9) had already drifted into two
readings of one measurement.

**The preview drag, measured** (release, min of five): a cold mannequin preview is
**0.105 ms** against 0.035 ms for the twenty-joint rig; warm, 0.026 against 0.004.
Against one 60 Hz frame. **No cache and no coarser mannequin is owed**, and the
reason is structural rather than lucky: the weight solver is not on this path at
all, because a block mannequin is rigidly bound by construction.

**Two real defects found on the way, neither of them cosmetic:**

* **`fit_template` dropped the side tables.** The fit MOVES joints and never adds,
  removes or reorders one, so every index-keyed table is still true of the result —
  but it rebuilt the asset without them. A fitted mannequin therefore arrived
  role-less, `build_locomotion` fell back to its name rule, and the wizard refused
  the rig it had just generated with a message about `upper_leg_l`.
* **`a_changed_proportion_changes_the_generated_rig` compared joint 0**, which on
  the mannequin is `root` and sits at the origin at every height. Two identical
  zeros, calling the generator broken. It compares the hip girdle by role now.

`block_body_mesh` emits a box only for a segment the table calls a deform bone.
The zero-length guard was already dropping most correctives — *silently*, which is
the problem: "happens to be degenerate" and "is not part of the body" are
different facts and only one of them is stable.

`phase29_gate` stays green on its own rig, untouched, because `phase29_spec` is
pinned to `BipedCanonical`.

### Decisions (SK1a's, binding on later waves)

1. **The asset beats the document.** Where a briefing doc and a shipped asset on
   this machine disagree about a fact the asset contains, the asset wins and the
   disagreement is printed in the code that carries it.
2. **Names, parents and order are an interface; a bind pose is content.** The
   mannequin's vocabulary and topology are reproduced; every offset is a
   proportion of a `BodyParams` length and no absolute dimension is copied.
   *Amended by the SK1a audit*: anything read off the asset must be a **rule**
   (thirds; `_01` nearest the driver) **or a dimensionless proportion that is
   named as measured** — the hand table is nineteen bones of the second kind, and
   the original wording ("not a number") was already false of it when it was
   written. A number read off an asset and presented as derived is the thing this
   decision is actually against.
3. **The role table is the first answer and a name rule is the fallback**, at
   every site that asks what a bone is. A rig with no table behaves exactly as it
   did.
4. **A role kind or side is a wire enum**: append-only, discriminants frozen,
   pinned by an arm.
5. **The drive pass sits between construction and correction**, and its position
   is part of the trace, not an implementation detail.
6. **A ragdoll chains by index and labels by role.** Twelve fixed roles cannot
   describe a five-segment spine; an index can.

### What is open after SK1a (for SK1b and its successors)

* **The twist/IK ordering bound** — a twist reflects the pre-IK pose. Routed to
  SK1b with the measurement it needs (run the pass twice, or re-drive per chain).
* **The corrective bones are inert.** 74 of 161 sit at their parent's origin and
  nothing drives them. They exist so a retarget finds every name; making them
  *correct* anything needs a driver kind this schema does not have (a pose-space
  driver keyed on a parent's angle) and therefore another bump. Not owed until
  something reads them.
* **`grips` is empty.** The type ships; SK1b's finger solver is its first
  consumer, and `JointLimit::cone` is there for it.
* **4 influences per vertex still stands**, and the 16-attribute wall in the
  skinned pipeline is full. A hand with metacarpals and three phalanges per finger
  is where the top-4 truncation starts to bite; state it in SK1b's ledger and
  measure before packing two attributes into one `u32x4`.
* **`Skeleton::index_of` is still a linear scan**, and the engine now hands it
  161-name rigs. Nothing on the fixed step measured badly (the drive pass and the
  role lookups are index-keyed on purpose, and `ik_follow` is a persisted table
  precisely so no per-step name scan exists) — but the *editor* paths still scan,
  and `autofit::symmetrize` does it inside a loop, which is O(J²) per refine
  iteration: 4 × 25 921 string compares on the mannequin against 4 × 400 before.
  Not measured as a problem; named so the next person measures rather than
  discovers.
* **The mannequin's arms are a T-pose**, not the A-pose the shipped rig stands in,
  because a bind pose here carries no rotation. A retarget from an A-pose source
  is bind-relative and unaffected; a *mesh* authored against Epic's A-pose is not,
  and that is SK1b's problem when it builds a real body.
* **The default `arm_length_ratio` is too long, and the T-pose is what made it
  visible.** `BodyParams::default()` carries 0.42 — shoulder to wrist as a
  fraction of height — which on a 1.75 m rig is 0.735 m and gives a **2.24 m
  wingspan** (a real one is about the height). Measured against the same asset the
  hierarchy came from: `upperarm` 27.771 + `lowerarm` 27.251 = 55.0 cm on a rig
  whose pelvis sits at 95.9 cm, which is a ratio of **0.31**, so the default is
  ~37 % long. It always was; arms hanging down put the wrist at mid-thigh and
  nobody looked. **Not changed here**: it is a shared default, `phase29`'s
  committed clips are generated from it, and a proportion change is not a delta
  anyone can verify arithmetically. Routed to SK1b, which builds a real body mesh
  and will care.
* **A driven bone is overwritten, not blended.** Nothing else writes a twist bone
  on a rig this engine generates, so the pass sets it outright. An *imported* clip
  that bakes its own twist values onto a rig that carries a driver table would
  have them replaced. Nothing in the tree does that today (an imported rig arrives
  table-less); the day one does, the answer is a per-driver "authored wins" flag,
  not a guess.
* **No `.inf_retarget`.** The maps are code, not assets; nothing persists a
  pairing an author edits.

### Counts

| | after I7b | **after SK1a** |
|---|---|---|
| battery blocks / passed / failed / ignored | 319 / **5 971** / 0 / 16 | **319 / 6 009 / 0 / 16** — **+38 arms and no new block**. The 6 009 reproduces exactly; the *delta* did not, because the baseline was taken from I7b's **wave** figure (5 968) and not from its **audited head** (5 971, the wave's 5 968 plus that audit's three arms — I7b's own ROADMAP block says so). +38 is also the count of `#[test]` items the wave's diff adds, and it removes none. "No new block" is right for the reason given: every arm went into a file that already existed |
| goldens | 54, byte-identical under `INF_GOLDEN_STRICT=1` | **54, byte-identical**, re-run under `INF_GOLDEN_STRICT=1` over **101 arms** with **no PNG rewritten** (`git status` on `tests/goldens/` empty). Nothing in this wave touches a render path |
| frontend tests / files | 702 / 78 | **702 / 78**, re-run — the wizard, the Skeleton Editor and `BodyPlanName` all moved; `tsc` and `eslint --max-warnings 0` clean |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0**. Four findings were cleared on the way and all four were this wave's own: a `?`-able `else if let` chain in `twist_rule`, a needless borrow in the ladder's arms, a boxed-closure type that wanted a `type` alias, and a needless borrow at the Ring-2 anim door |
| rustdoc warnings (ceiling 450) | 374 individual over 30 crates | **374 individual over 30 crates**, measured after `cargo clean --doc`. **The wave adds zero**: it introduced exactly one (a public doc linking the private `skel_v2::SkeletonAsset`) and it was found and removed |
| `cargo fmt --all --check` | clean | clean |
| schema | `.inf_skel` v2, `.inf_anim` v2, `.inf_sm` v3, `.inf_mesh` v2, scene **v25**, `ScenePayload` v11 | **`.inf_skel` v2 → v3, and NOTHING else moved** |
| committed sample bytes | — | two `.inf_skel` files and their sidecars: **+4 B** and **+8 B**, arithmetic-verified |

**The twentieth chr(92) catch, and it was this wave's own.** The pose-writer pin's
panic literals carried two eaten `\` continuations — written by a non-raw Python
string, in the commit that added a gate about frozen orders, the same afternoon.
Nineteen previous catches are ledgered across two phases;
`no_string_literal_in_the_workspace_carries_an_eaten_continuation` found the
twentieth on the first full battery, which is the gate doing exactly its job.

### Commits

| | |
|---|---|
| `878d3169` | the rig is the mannequin's, bone for bone, measured off the asset |
| `13c5647d` | the downgrade-bless, and the two doors that name a plan by string |
| `9d22d1b7` | the drive pass joins the pose writers, and the name guessing loses its job |
| `187166bb` | the wizard on the new rig, and the three numbers that decide nothing needs mitigating |
| `04a53e0c` | the pose writers get an allowlist, and the fixtures that were about the fit |
| `81d93683` | the index borrows, and the door pays for it |
| `bba02bfc` | the two proportion fields the simple biped never reads |
| `c702c32d` | a driven bone is overwritten, and the census says 74 |
| `5c3bb736` | the twentieth chr(92) catch, and it was mine |
| `e7389fd8` | the last three the gates found |
| `498bc92a` | the ledger's counts, measured rather than carried forward — **eleven**, not the ten this table could name, because a ledger commit cannot list itself |

## The SK1a audit (2026-08-25)

Adversarial, `9790ff8e..498bc92a`, fresh reader, nothing pushed. **The wave's
headline is true and it was re-derived here independently**: the briefing
document's printed tree parses to exactly **89** unique bone names, with `neck_02`
absent and `ik_head`/`ik_pelvis`/`ik_spine` present; the shipped
`SK_Mannequin.uasset` contains exactly one 161-record `FReferenceSkeleton`, and
all 161 `(name, parent)` pairs are byte-for-byte what `manny.rs` emits, in the
same order. The twist bones really do sit at 1/3 and 2/3, and `_01` really is the
one nearest the driver. **No Unreal asset bytes are committed anywhere in the
range** — two `.inf_skel` files and three new `.rs` files, nothing else binary.

What the audit found is one shape and one correction. The shape: **an arm looser
than the sentence it is written under**, six times — a cache assertion six times
its own baseline, a contrast asserted as `> 1` that fed a number seven times wrong
into this ledger, a census that pinned three of eighteen families while claiming
to pin all of them, a deform-only filter no arm could see the removal of, the
wave's own headline fix asserted nowhere in the crate that owns it, and a source
gate that could not enumerate its own directory. The correction: **what came off
the asset was three kinds of thing, not two.**

Three HIGH and sixteen MED, all fixed; twelve LOW carried by name. Every fix is
mutation-verified.

### What reproduced, measured here rather than read

| the wave's figure | mine |
|---|---|
| the document's printed tree is 89, not 161 | **89** unique names, parsed |
| 161 bones, one root, parent < index, no duplicate name | **all four**, against the asset itself |
| census 63 deform + 16 twists + 7 handles + 74 helpers + 1 root | **exactly**, counted off the table |
| twist bones at 1/3 and 2/3, `_01` nearest the driver | **0.3333 / 0.6667** on all four segments, both sides |
| trace 36 B + 40 B/joint, so 836 / 6 476 B, **7.75x** | arithmetic exact |
| `character-demo` 684 to 688 B (+4), `Hero` 2 687 to 2 695 (+8) | **exact**, and byte 0 is 2 to 3 |
| preview drag: mannequin cold well above canonical, both far inside a frame | **0.126 ms vs 0.023 ms** cold, **0.022 vs 0.004** warm |
| goldens 54, byte-identical, 101 arms | **54 / 101**, `INF_GOLDEN_STRICT=1`, no PNG rewritten |
| PIE == shipping on a 161-bone trace | green in the battery, both arms |
| a v2 rig migrates; a v1 rig is refused; a v3 file refuses a v2 reader by stamp | held |

Two the audit re-measured and **corrected** (below): the classifier's shape on a
table-less mannequin, and the cause of the heat solve's 2.4x.

### HIGH

**H1 — `merge_skeletons` dropped all four v3 side tables.** It shifts sockets and
limits by `joint_offset` and left `roles`, `twists`, `ik_follow` and `grips` at
`Vec::new()`: the *same* defect the wave found and fixed in `fit_template`, in the
other door that rebuilds a `SkeletonAsset` field by field, on a shipping path (the
Skeleton Editor's merge, P24.3 modular rigging). Quieter, because nothing fails —
a merged mannequin comes back with no opinion about its own anatomy,
`build_locomotion` refuses it by name for a bone it has, the drive pass drives
nothing, and all five role-first sites fall back to guessing. Fixed with the
shift, and `a_merge_carries_both_sides_side_tables_shifted` asserts both halves,
the offset, the base rows still *answering*, strict ascending order, and that the
merged rig still generates a walk. Mutation: deleting `asset.roles = roles` fails
it at **0 against 163**.

**H2 — the preview cache's arm could not fail.**
`assert!(warm161 <= cold161 * 1.5 + 0.5, "the cache did not help")` is, at the
recorded 0.105 ms cold, a **0.66 ms** bound — six times the cold path — so
deleting `CharacterPreviewSession`'s memoization outright leaves `warm == cold`
and the arm green at every optimization level. It fires only if the cache makes
the path dramatically *slower*, which is the opposite of what its message claims.
The crate already owned the honest instrument (`PreviewBuilds`, whose own docs say
a silently-missing cache "would keep every number in the preview correct"), so the
arm reads counters — six previews, **one** locomotion build, **one** body build,
zero BVHs, on both rigs — and the wall clock is a `println!`. The 250 ms debug
ceiling stays, with its slack named: in the build CI runs it is about 2 400x the
measurement, which is why it is not the assertion.

**H3 — the wave's headline fix was asserted nowhere in the crate that owns it.**
`fit_template`'s side-table carry and its new role-first ground rule had no arm:
every `autofit` fixture moved to `BipedCanonical` *in the same wave*, and a
`BipedCanonical` rig carries **empty** tables, so the carry copied four empty
vectors and the role branch was never taken. The only cover was indirect and two
crates away. `a_fitted_mannequin_keeps_every_table_it_arrived_with` fits
`BodyPlan::Biped` and asserts all four tables, the limits, the sockets, the stamp,
that no joint was renamed or reordered (the premise the carry rests on), that the
ankles and not the `ik_foot_*` markers were planted, and that the fitted rig still
generates a walk. Mutation: restoring the field-by-field rebuild fails it at `[]`
against 161 rows. The rebuild is also gone — `fit_template` moves the generated
asset and replaces its skeleton, so the ninth table rides through without an edit
here, which is the structural half of the fix.

### MED (all fixed)

| | |
|---|---|
| **M1** | `SkeletonAsset::migrate` defined its `ordered` closure **twice**, repeated the six-line doc verbatim, and ran both checks on both tables — a bad scripted merge in the one function the schema bump rests on |
| **M2** | the census comment said "89 deform-or-driven bones and 72 helpers"; the asserted truth is **87 and 74** |
| **M3** | the census arm said "a family cannot be re-labelled without moving a number here" and pinned **three of eighteen** kinds. Mutation: `thigh_l` to `Spine` left the whole arm green and was caught two crates away by `build_locomotion`. All eighteen are pinned now, plus the sum; the same mutation fails at "the Spine census, 6 against 5" |
| **M4** | the ragdoll contrast asserted `free.len() > 1` and fed "14 parts, 4 free" into this ledger. Measured through the same door: **92 parts, 31 free**. Pinned as a pair |
| **M5** | `portable_pose` could not enumerate its own directory: **7 of 31** files in `crates/inf-anim/src` were on neither list, three of them (`retarget.rs`, `template.rs`, `merge.rs`) writing poses or bind poses. `LEDGERED_EXCLUSIONS.len() == 1` was asserted, so recording a gap honestly would have been red. Five files joined the ban (all clean), two are ledgered by name, and a completeness arm now requires every `.rs` under `src` to be on one list or the other |
| **M6** | `retarget_pose`'s doc said the cheap door "never allocates" the report; it delegated and paid four `Vec`s, **161 `String` clones** and four sorts per call. One body with the accumulation as a flag, plus an arm that the two doors write bit-identical poses |
| **M7** | `block_body_mesh`'s deform-only cut was unasserted (`used.len() > 8` is happier with *more* boxes). The skinned set is pinned as an identity — the parents of the deform bones, **51** of them, all deform or root. Mutation: without the cut, `ik_hand_root` carries skin weights |
| **M8** | `skel_set_limit` wrote `cone: None` unconditionally, **erasing an authored cone** on every hinge edit with no way back, since `SkelJointDto` carries no cone. Read-modify-write |
| **M9** | `SkelJointDto::canonical` and `RenameVerdict::left_humanoid_set` asked only the canonical nineteen, so on the engine's own default rig **156 of 161 bones badged as unknown**, renaming `thigh_l` broke `manny_to_canonical` in silence, and renaming `foot_l` warned. `inf_anim::is_interchange_joint_name` is the union both sites read now |
| **M10** | the wizard offered `spineSegments`/`neckSegments` for the mannequin, which reads neither — the `bba02bfc` class, unfinished. Hidden for `biped` only, since `BipedCanonical` does derive its chain from them |
| **M11** | the heat arm's cost model named two causes it does not have (below) |
| **M12** | "the bind pose is not copied" was true of 142 bones and false of nineteen (below) |
| **M13** | the Content Drawer had **no door** to `BipedCanonical` — the rig every committed `.inf_anim` is index-bound to, and the one `SkeletonAsset::UPGRADE_REMEDY` sends a stale project to that very menu to make — and labelled the 161-bone mannequin "Biped" |
| **M14** | `CharacterSpecDto::plan`'s doc omitted `biped-canonical`, on the field the wave widened, in a doc whose own argument is that a stale name list is the hazard |
| **M15** | `phase24_wizard`'s canonical arm is documented as "the same gate" and is one arm of it: one comparison, no second process, no severed-machine control |
| **M16** | this ledger's own arithmetic: baseline, delta, scene version, commit count, byte-identity (see *Counts*) |

### M11 — the heat solve's 2.4x had the wrong explanation

The write-up said the Laplacian assembly **and the per-vertex top-4 gather** are
bone-independent, and that the diffusion is one CG solve per bone. Both halves are
wrong: the gather walks every bone's field for every vertex (**O(V·B)**, and it is
the part that does scale), and a bone that wins no vertex has `sources == 0` and
**skips the solve entirely**. The arm reports the real count now, and it is the
answer: **15 of 24 segments solve on the canonical rig and 19 of 249 on the
mannequin** — 1.3x the actual solves for 10.4x the bones. It asserts that count
beside the clock-ratio upper bound, and the millisecond pair is printed rather
than written down, because the wave's code comment (32.4 to 78.3) and this ledger
(31.9 to 75.9) had already drifted into two readings of one measurement.

Carried: **89 of those 249 segments are zero-length leaf markers** rather than
bones (36 %, against 5 of 24 = 21 % on the small rig), so the 10.4x partly
measures leaf density.

### M12 — three sources, not two

Verified numerically against the asset: the nineteen hand bones' offsets in
`Place::Hand` are the reference skeleton's own local bind translations, normalized
so the middle-finger chain sums to 1.0 of hand length (`middle_metacarpal` is
3.375833 cm over a 17.117 cm chain, which is the committed **0.1972**).
Fifty-seven per-bone numbers, under a `Place` doc that said "a rule per *class* of
bone rather than a number per bone" and a decision that said anything read off the
asset must be "a rule, not a number".

**Nothing was removed**, and the reason is the finding: there is no *rule* for
where a pinky metacarpal sits relative to a ring one — it is anatomy, and
inventing it makes a hand nobody's glove fits — and a ratio of a length this
module chooses carries no dimension (the hand's overall size is authored and does
**not** match the reference rig: 0.105 of height against a measured 0.096). What
was wrong was the sentence. The docs and this ledger now split the three sources
by name, and `the_two_hands_are_an_exact_mirror_of_each_other` is the arm those
fifty-seven hand-maintained numbers never had: nineteen bones per side, `x`
negated and `y`/`z` equal to the bit, plus the normalization itself.

Also on this: "measured off the asset" is a claim a successor cannot check, so
`manny.rs` now carries the recipe — the package path, that it is a **non-cooked**
editor package (which is why `ExportName` is present and the names are ASCII), the
`RawRefBoneInfo` record layout, the acceptance rule that makes the offset unique,
and the `RawRefBonePose` array the thirds and the hand ratios came out of.

### The mutations the audit ran itself

| mutation | expected | result |
|---|---|---|
| move `drive_pose` below the pelvis drop | the pose-writer pin goes red | **red**, naming both passes |
| delete the `drive_pose` call | the fixed-step arm goes red | **red** — "twist_01 took 0 of a full 0.565" |
| `thigh_l` to `Kind::Spine` | the ragdoll connectivity arm catches it | **it does not** — 17 parts, one free capsule, fully connected. Caught by `build_locomotion` two crates away; the census arm now catches it at the table |
| `build_manny` ignores `height_m` for the pelvis | the proportion arm goes red | **red**, with its own message |
| drop `asset.roles` from the merge | H1's arm goes red | **red**, 0 against 163 |
| restore the field-by-field rebuild in `fit_template` | H3's arm goes red | **red**, `[]` against 161 rows |
| delete `block_body_mesh`'s deform cut | M7's arm goes red | **red** — `ik_hand_root` carries skin weights |
| rename one `SIM_PATH` entry | the completeness arm names the orphan | **red**, naming it |

### LOW, carried by name

* **The retarget map has no consumer outside its own tests.** `canonical_to_manny`,
  `manny_to_canonical`, `retarget_pose_reported` and `RetargetReport` appear only
  in `retarget.rs`'s tests and `lib.rs`'s re-exports; `retarget_pose` itself is
  called only from a `template.rs` test. "The silence ends" is true of the library
  and not of the product — nothing in the editor, the wizard, the importer or the
  player retargets anything, and `RetargetReport::summary`'s "for a log or a
  wizard warning" has neither.
* **Role-or-name is decided per *question*, not per limb pair.**
  `locomotion::arms_of` takes the left arm from the table and the right from a
  name if the table happens to name only one; `pose::foot_joints` accepts the role
  answer if *either* side is `Some`; `build_ragdoll` routes to the role path if
  *any* bone carries a role. `leg_by_role`'s own doc says all-or-nothing is the
  rule, for exactly this reason.
* **`derive::foot_joints`' role path is "any `Foot` row wins"**, with no side
  keying, and it returns the *table's* order — right-then-left on the mannequin
  against left-then-right on the canonical rig, so `DeriveReport::plants.first()`
  means a different foot depending on the plan. Deterministic per rig; surprising
  across two.
* **`ConeLimit` is authored and enforced by nothing.** `solve_chain` clamps
  `min_deg`/`max_deg` only, the ragdoll reads neither, no generator produces one —
  so `with_cone` accepts a constraint nothing applies. Documented as such now;
  SK1b's finger solver is the first consumer.
* **A fitted rig no longer satisfies the invariants its side tables were derived
  from.** `ik_foot_*` was emitted *at* the ankle by `Place::Mirrors` and the fit
  moves the ankle without it; a twist bone is refined independently of its segment
  while its `TwistDriver::fraction` still says 1/3. Both are masked at runtime by
  the drive pass overwriting them, which is why nothing is red — and both mean a
  fitted rig's *bind* pose is no longer the pose the tables describe.
* **The fixed-step cost of the drive pass is argued, not measured.** The wave's
  open list says "nothing on the fixed step measured badly" and then gives a
  structural argument; `drive_ik_follow` adds one full `global_transforms` pass and
  a 161-entry allocation per posed character per step. The reasoning is sound and
  the number does not exist. (P22's law: inference dressed as measurement.)
* **`a_changed_proportion_changes_the_generated_rig_and_its_walk` varies only
  `height_m`**, so a generator that ignored `arm_length_ratio` — the wave's own
  carried finding — is invisible to it, and its `.or_else(index_of("hips"))`
  fallback never executes.
* **The Skeleton Editor's tree has no role column**, so a mannequin arrives as 161
  undifferentiated rows and the rig's own answer about what each bone *is* stops
  at the Ring-2 boundary.
* **89 of the heat solve's 249 "bone segments" are zero-length leaf markers.**
* **`Skeleton::index_of` is still a linear scan** and `autofit::symmetrize` still
  calls it inside a loop (the wave carried this; it is unchanged).
* **The `BodyPlan` enum gained `BipedCanonical` at index 1**, shifting
  `Quadruped`/`Hexapod`/`Npedal`. Safe today — the wire is the kebab-case *string*
  at both Ring-2 doors and no bincode payload carries a `BodyPlan` — and worth
  knowing before one does.
* **`ik_hand_root` and `ik_foot_root` never move.** They are deliberately absent
  from `IK_FOLLOW` (an anchor that chases a hand is not an anchor), which is right,
  and it means the handle subtree's *root* stays at the rig origin while the
  character walks away. Nothing reads them yet.

### Decisions (the SK1a audit's, binding on later waves)

1. **A gate's bound must be tighter than the thing it is measuring.** Three of
   this wave's arms were satisfied by the defect they were written against. Where
   a counter exists, assert the counter and print the clock.
2. **A number in a ledger must be a number a test prints.** "14 parts, 4 free" was
   seven times wrong and nothing could see it, because the arm asserted `> 1`.
3. **A hand-maintained list needs an arm that enumerates its domain.** A ban list
   over an enumeration is a ban on what somebody thought of.
4. **A door that rebuilds a struct field by field will drop the next field.** Move
   the value and replace what changed; `fit_template` and `merge_skeletons` were
   the same defect twice in one wave.
5. **A vocabulary question asked of one vocabulary is the wrong question once
   there are two.** The badge, the rename warning and the retarget all ask
   `is_interchange_joint_name` now.

### Counts

| | after the SK1a wave | **after the audit** |
|---|---|---|
| battery blocks / passed / failed / ignored | 319 / 6 009 / 0 / 16 | **319 / 6 015 / 0 / 16** — **six** arms, **no new block**: two in `inf-anim`'s `merge`, one in `retarget`, one in `manny`, one in `inf-dcc`'s `autofit`, one in the `portable_pose` gate |
| goldens | 54, 101 arms | **54, byte-identical under `INF_GOLDEN_STRICT=1`** over 101 arms, no PNG rewritten. Nothing here touches a render path |
| frontend tests / files | 702 / 78 | **702 / 78**, re-run; `tsc` and `eslint --max-warnings 0` clean (the drawer, the wizard dialog, the Skeleton Editor and `assetStore` all moved) |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** — one finding cleared, a `clone` on a `Copy` `JointLimit` in the audit's own merge fix |
| rustdoc warnings (ceiling 450) | 374 individual over 30 crates | **374 individual over 30 crates** after `cargo clean --doc`. **The audit adds zero**: it introduced two (public docs linking the private `Place::Hand` and `template::validate`) and removed both |
| `cargo fmt --all --check` | clean | clean |
| schema | `.inf_skel` v3 | **unmoved** |
| committed sample bytes | — | **unmoved**; `git status` on `samples/` and `tests/goldens/` empty |
| chr(92) | the twentieth was the wave's own | **no twenty-first** — the wave's added lines and the audit's own were swept for both shapes (interior space runs, and bare newlines inside non-raw literals): zero, and the workspace gate is green |

**Ten** audit commits in `498bc92a..`, `(SK1a) audit:`-tagged. Nine carry a
change and are named here; the tenth is the correction to this count, and a
ledger row cannot name the commit that writes it (the same convention the wave’s
own table follows, and the same trap — it said ten against eleven):

| | |
|---|---|
| `637bf93b` | the merge door drops what the fit door stopped dropping |
| `ab6514b7` | three gates that could not see what they claimed |
| `0f226ad3` | the door that checks itself twice, and the cheap door that was not |
| `4b1c97ab` | the fit's carry had no arm, and the cache's arm had no teeth |
| `96c105cd` | the badge, the cone, the two spinners, and a cost model that named the wrong cause |
| `2cbf3a8f` | the drawer had no door to the rig every committed clip is bound to |
| `0e38c084` | what came off the asset was three things, not two |
| `f6d33c24` | how to re-derive the table, and two rustdoc links that were not |
| `27ba1ea3` | the ledger, and the six numbers in it that were not measured |

## Wave SK1b (hands, grips and the starter character)

Base `be85c213`. Five clauses; **four landed, one carried** — clause 5 is not
done and the reason is structural rather than a matter of time (below).

The wave's headline is that giving a rig a *hand* is what finds the things
nobody had looked at. Every one of the four defects below is in code older than
this wave, and every one of them was invisible because nothing in the engine had
ever solved an arm, closed a finger, or generated a mesh in `f64`.

### Clause 1 — hand IK, the finger solver, and the cone (`923b1a31`, `74495241`)

**The cone is enforced.** SK1a spent a schema bump on `ConeLimit` and its audit
recorded honestly that nothing read it. `ik::clamp_to_cone` reads it — swing-twist
about the cone's own axis, the swing *rescaled* to the half-angle in the direction
it was asked for rather than discarded, the twist folded into `(-π, π]` and
clamped, all of it `patan2_64`/`psin64`/`pcos64` in `f64`. `ik::apply_joint_limit`
is the one door every limit now goes through, and **a cone outranks the per-axis
box**: two descriptions of one joint's freedom cannot be made to agree (a 90° cone
and a three-axis box disagree at every diagonal), so the more specific one wins.
That is also what lets `JointLimit::cone_only` author a finger without spelling a
box that `clamp_to_limit` would read as *fully locked*.

`build_manny` authors a cone on **all 38 digit bones**, which gives the type its
first producer as well as its first consumer. Half a degree of margin over the
solver's own maximum, deliberately: a clamp applied exactly at the boundary is a
quaternion rebuild whose result is not bit-identical to its input, so
`GripReport::clamped` would count every fully-closed finger and stop meaning
anything.

**`inf_anim::grip` reads a hand off its own bind pose**, and nothing about it is
authored:

* `along` — the farthest fingertip;
* `spread` — the knuckle line, taken between the two finger roots farthest
  apart and **signed away from the thumb**;
* `palm_in` — where the thumb root sits once `along` and `spread` are projected
  out of it (a thumb is on the palm side of a hand; that is what a thumb *is*,
  and it is the only bone that says which side the palm is on);
* `curl_axis` — `along × palm_in`, **per digit**, so a thumb opposes instead of
  flexing sideways.

Sorted along `spread` the four fingers *are* index, middle, ring, pinky — on the
mannequin, on both hands, with **no string compare**, asserted against the rig's
own vocabulary.

**The measurements:**

| | |
|---|---|
| a closed fist, middle fingertip to wrist | **18.4 cm → 8.1 cm** |
| the palm-inward projection of the same curl | **not monotone** — 0.047 / 0.073 / 0.071 / 0.051 at quarter, half, three-quarter, full |
| halving every cone | pulls back **15 of 19** bones on that hand |
| aperture, fingertip to wrist | open 18.4 cm, 9 cm ball 11.3, 4.5 cm bar 9.6, 3.2 cm grip 9.1 |

The non-monotone projection is the finding the arm was rewritten around: *a real
fist takes the fingertip past the palm and back up toward the knuckles*, so
direction is measured on `palm_in` and **amount** is measured from the wrist,
which is the quantity that stays monotone all the way to a closed hand.

**THE ELBOW WAS NEVER A HINGE.** A hinge's axis is the axis a joint turns
*about*, in its parent's frame, and the mannequin's arms lie along `±X`
(SK1a's law that a bind pose here carries no rotation) — so P24.1's `hinge_x` on
`lowerarm_*` names the forearm's own **roll** axis and permits nothing but roll.
Measured, an arm reaching a point 55 cm in front of the shoulder:

| | reach error |
|---|---|
| no limits at all | **3e-8 m** |
| P24.1's `hinge_x` elbow | **0.484 m**, elbow local rotation pinned to the identity |
| a correct `hinge_y` elbow through `solve_chain` | **0.083 m**, and iterating the pole is a fixed point |
| a correct `hinge_y` elbow through `grip::reach` | **0.00000 m** |

It had been that way since P24.1 and nothing found it because nothing solved an
arm chain. **The canonical biped is untouched** — its arms hang *down* along
`−Y`, so `hinge_x` is a correct elbow there; the same line is right in one
generator and wrong in the other, which is why the fix is a bind-pose property
and not a rename.

**And `clamp_to_limit` did not count the swing it discarded.** The comment said
the discarded swing "is zero for the hinge poses the solver produces" — a claim
about the *solver*, in a function that cannot see one, and false the first time
an arm went through it: the entire 86.7° bend was discarded as swing, the twist
about X was zero before and after, and `IkReport::clamped` reported **0** while
the limit moved that wrist half a metre.

`grip::reach` is the solver an arm wants. A hinge takes the freedom away, so the
answer is closed form: the elbow angle follows from the distance
(`cos θ = (l₁² + l₂² − d²)/2l₁l₂`, and the half-angle identities give the
quaternion from the cosine with **two square roots and no angle ever formed** —
no `acos`, no `sin`, no `cos`), and aiming a rigid two-bone assembly whose end is
already at the right distance puts the end exactly on the target.

**What a gripping hand costs, measured** (release, min of five, on the 161-bone
rig, per hand per fixed step): derive the hand **3.2 µs**, curl it **3.3 µs**,
solve the arm **6.2 µs** — against a 16 667 µs frame. `hand_of` is the one that
could have been a problem: it walks the skeleton once per digit chain and nothing
caches it. The number is taken rather than argued, because *"the fixed-step cost
of the drive pass is argued, not measured"* is on the SK1a audit's carried list
and this wave was not going to add a second.

**Hand IK runs from the fixed step through a resource**, `HandIkRes` — the
runtime half of the pair `IkTargetsRes` already had, so **no schema moves** and
both hosts inherit it with no host-side change. It carries a reach per hand, a
`GunGrip` two-handed hold, and which grip each hand is closed on. Absent until
asked for; an empty request is the same as no request.

**The twist/IK ordering bound is closed** — SK1a routed it here by name. A twist
bone is a statement about the pose that is *finally published*, so it is computed
from the corrected one: `drive_pose` runs at construction as before, and a second
`redrive` runs after every correction, **gated on a correction having happened**
so a character nothing touched poses byte-identical bytes. Re-driving per chain
was the alternative and is worse in the way that matters — it would put the
knowledge of which twists belong to which limb inside every solver, in three
places, which is the shape the role table exists to retire. `redrive` is a named
function precisely so `every_pose_writer_runs_in_its_frozen_order` can see it: a
second `inf_anim::drive_pose(` would be the same needle as the first. The pin is
**8 writers** now, and `apply_hand_ik` sits after the feet because a stance is
decided by the ground and a hand solves against the body that stance produced.

One more ordering finding inside the hand pass: **`ik_hand_gun` is driven at pose
construction**, so reading it for the off-hand target after the holding arm has
just moved reads where the *animation* left the weapon. Measured 0.42 m between
the hands on a weapon 0.30 m long; the handles are re-driven between the reach
and the gun solve, and the two hands are **0.3000 m apart** on a 0.30 m weapon.

### Clause 2 — the weapon is an entity, and the muzzle is its own (`ab68f71c`)

An equipped weapon was an inventory slot index: nothing drew it, nothing could
attach to it, and its shot started 1.4 m above the character's feet because
somebody had to pick a number (the scout's risk 14, whole). It is an entity now —
spawned from `step_gameplay`, the one Ring-0 rule both hosts call, under a
content-derived guid (`equipped_weapon_guid`, the P22 idiom) so two hosts spawn
the same one, attached to the `hand_r` socket via `AttachedTo`, and despawned the
moment nothing is equipped. **No schema moves**: `AttachedTo` is already a scene
component and this one is never saved, and `WeaponDef` carries no `Serialize` at
all, so `muzzle_forward_m` costs nothing.

`MUZZLE_HEIGHT_M` stops being *the* muzzle and becomes the **named fallback** for
a character with no rig to hang a weapon on — which is every level committed
before this wave, the whole `phase30-gameplay` fixture, and every test rig that
steps gameplay without stepping the pose. The control asserts the two agree to
**1e-12** on the legacy capsule hero, on `hit.from`, which nothing in
`weapon_3d.rs` had ever looked at. Mutation: drop the `evaluated_pose` guard in
`weapon_muzzle` and an unposed character reads its weapon entity's origin, which
is its capsule **centre**, 20 cm low.

**Stated rather than hidden**: `step_gameplay` runs before `step_pose_evaluation`
and `update_attachments` in both hosts, so a weapon-derived muzzle is one fixed
step (16.7 ms) behind the hand. Identical in both hosts, so no trace can see it;
moving the gameplay step below the pose would fix it and would move every
committed trace in the tree, so it is named here rather than done quietly.

**`Mask_AimOffset` exists.** `layers.rs` has described an upper-body overlay mask
since P29.2 and the scout found none anywhere in the tree — designed, not built.
`JointMask::upper_body` derives one from the role table's first spine bone;
`BlendProfile::from_mask` is the missing direction of a conversion that could only
read; the wizard authors it onto every generated machine. Measured: **104 of 161
joints** — spine, neck, head, both clavicles, both arms, every finger **in**;
root, pelvis, both legs, both feet and every `ik_*` handle **out**. The arm
asserts the legs are out, not only that the arms are in: a mask covering every
joint satisfies "the hand is masked in" perfectly and confines nothing.

### Clause 3 — the starter character (`943a0028`), and what is not done

`inf_dcc::body_mesh` generates a humanoid from its own rig: limbs are **welded
tapered tubes** swept along a whole chain (shoulder, elbow and wrist are one
surface with a continuous silhouette, not three boxes), hands are a **palm slab
and five tapered digits** following their own metacarpals and phalanges, and the
head is a **cranium with a jaw, a nose and two ears**. Every radius is a fraction
of the rig's *measured* height and every centre is a joint the rig already
placed.

| | |
|---|---|
| kernel | **795 vertices / 727 faces**, 23 welded shells |
| exported | **1247 vertices / 1498 triangles**, one submesh, one `Skin` slot |
| signed volume | **0.0687 m³** — what a 1.75 m human displaces |
| height | **1.749 m on a 1.75 m rig** (1.199 at 1.2, 2.399 at 2.4), sole on the ground the ankle stands on to 0.8 mm |
| generate | **0.57 ms** (debug) |
| generate + bind + masked heat solve | **439.91 ms** (debug), over **63 deform bones of 161** |
| weights | **760 assigned, 35 unreached**, worst residual 0.0000; **1247 skin rows, 0** on a bone that deforms nothing |

**The honest bound: shells, not one skin.** Each chain is welded along its own
length and is a closed manifold shell; the shells **interpenetrate** at the
girdles rather than being stitched into one surface, because stitching them is a
boolean union and this kernel does not have one. A silhouette reads correctly; a
cross-section at the shoulder would show two surfaces.

Three defects found on the way, two of them in code older than this wave:

* **The sweep frame had no semantic axes.** `Ring` names its two radii *width*
  and *depth*, and a frame seeded from "whichever cardinal the direction leans on
  least" put the width along the body's **depth** — a torso 30 cm deep and 22 cm
  across. Seeded from `X` now, except for a shell sweeping along `X`.
* **The foot swept backwards from the ankle**, so its rings tilted with the
  direction and the heel dipped **1.8 cm through the floor** on a 1.2 m
  character. It sweeps heel-to-toe now with every ring centred its own half-depth
  above the sole.
* **The heat solve's visibility oracle was built from `f32`-narrowed geometry
  while its rays start at `f64` kernel positions.** A ray from an exact vertex
  starts an ulp outside the surface the oracle knows and hits its own face:
  **349 of 795** vertices unreached against **35** through `inf_dcc::mesh_soup`,
  which is the same triangles in `f64`. An *imported* mesh is unaffected — its
  kernel positions are widened `f32` and the round trip is exact — which is why
  it took the first mesh this engine generated in `f64` to surface it.
* **A heat solve considers every joint, and a rig is not all deform bones.** The
  nearest visible bone to a palm vertex really is `weapon_r`.
  `solve_heat_weights_for` takes an eligibility mask and the wizard passes the
  role table's deform set — filtered at the *bone* set rather than at the
  assignment, because a bone that wins a vertex and is then dropped leaves that
  vertex with no source at all.

And the generator now reports **which bone made each vertex**, applied as a rigid
prior *before* the solve — because `solve_heat_weights` is documented as
"additive evidence rather than a reset", and without a prior the 35 unreachable
vertices keep `VertWeights::RIGID`, which is all of joint 0, the rig's **root**,
and stay behind when the character walks away.

Also: **the cone axes are normalized in `f64` and narrowed once**. In `f32`,
`-x/sqrt(x*x)` is `-1.0` for some `x` and `-0.99999994` for others, so two rigs
of *different heights* carried axes that differed by an ulp —
`a_fitted_mannequin_keeps_every_table_it_arrived_with` caught exactly that.

**The wizard defaults to it**: `build_character` with no supplied mesh now writes
the generated body, heat-weighted, and `CharacterBuild::mannequin` stays honest —
`true` only when the fallback actually ran, which is a rig with **no role table**
(`BipedCanonical`, and every `.inf_skel` older than v3). That fallback has its own
arm, because the default spec no longer exercises it.

**The skin material ships.** A neutral matte dielectric (`0.62, 0.58, 0.55`,
metallic 0, roughness 0.62) written beside the body and named as one of its
**dependencies**, which is the engine's own material binding for a mesh — the
glTF importer records a resolved material's GUID in the mesh's sidecar the same
way, and the mesh's own `material_slots` carries the slot name. The wizard writes
**eight** assets now, not seven. **Honest bound, stated at the write site**:
`SkeletalMesh` carries a mesh and a skeleton and *no material*, so nothing in the
skinned draw path reads it yet; the binding it would need is a component field.

**NOT DONE, and named:** the clause also asks for the character to be **shipped
as committed engine starter content**. No `samples/starter-character/` exists and
`ProjectTemplate::starter_content` — the committed hook — is untouched. See
*What SK1b did not do*.

### Clause 4 — the grip gate (`24f2fec2`)

`runtime/inf-player/tests/grip_gate.rs`. A character grips a handle, a rifle
two-handed through `ik_hand_gun`, and a thrown prop, and **PIE == shipping byte
for byte** over the whole sequence: 24 steps, **12 distinct poses**, **6476 bytes
a step** — exactly the 161-bone trace SK1a priced (36 B header + 40 B per joint).

The anti-vacuity is most of the gate. The idle pose is settled before anything is
asked for; taking hold moves it; tightening moves it **again** (an eased grip
that snapped to its end state would satisfy "the pose changed" and would not be
an ease); a two-handed hold solves **two** arms and writes more finger bones than
one; a 9 cm ball poses a different hand from a 4.5 cm bar at the same reach. The
**release** is asserted byte-identical to the pose before the hand ever closed —
the claim `apply_grip`'s "a curl is a pose, not a delta" rests on. The engagement
counters are compared between the hosts as well as the bytes, so "both hosts did
nothing identically" is not a pass.

The conformance arm measures the hand in metres rather than inferring it from two
poses differing, and the rifle's **trigger finger is authored straight** and comes
back bit-identical to rest — the one thing a per-finger curl target says that a
per-hand aperture cannot.

### Decisions (SK1b's, binding on later waves)

1. **A hinge's axis is a property of the bind pose, not of the joint's name.** An
   elbow is `hinge_y` on a T-posed rig and `hinge_x` on one whose arms hang down,
   and the same line is correct in one generator and wrong in the other.
2. **A constrained chain gets a solver that knows the constraint.** A pole picks
   a bend plane freely and a clamp then discards whatever is not in the hinge's;
   `grip::reach` sets the constrained joint from the distance and aims the rest.
   Measured 0.083 m against 0.00000 m.
3. **A twist bone is computed from the pose that is finally published**, so the
   drive pass runs at construction and again after every correction — gated on a
   correction having happened, so absent still costs nothing.
4. **A hand's frame is derived from its own bones, never authored.** A grip says
   what to do (aperture, curl); the rig says where the palm is.
5. **A visibility oracle must be built in the space its rays are cast in.** A
   narrowed copy of the geometry is a different surface.
6. **A weight solve needs to be told which bones may carry weight.** A rig is not
   all deform bones, and the nearest visible bone to a palm vertex is a weapon
   marker.
7. **A generated mesh carries a rigid prior**, because the generator is the one
   thing in the world that knows the right answer before the solver runs.

### What SK1b did not do (for its audit and its successor)

* **CLAUSE 5 IS NOT DONE.** The island hero is still `island.rs:429-484`'s
  hand-rolled capsule with `AnimStateMachine { sm: None }` and no `SkeletalMesh`.
  It was stopped deliberately rather than half-done, because the last thing this
  wave should do is leave a 900-step gate red. **The route is scouted, and it is
  narrower than it looks** — writing it down here is most of what the successor
  needs:

  1. **The assets reach the cook through the recipe, not through Ring 0.**
     `island_gate` builds its project with `inf_island::write_content`
     (`crates/inf-island/src/build.rs:852`), which knows nothing of `inf-anim` or
     `inf-dcc` — but its last loop copies **`build.recipe.content`**
     (`:895-909`), a `[content]` list of files that "live beside the recipe". So
     `samples/starter-character/` + six `[content]` entries in both
     `samples/island/island.toml` and `samples/island-fixture/island.toml` puts
     the character on the path the cook walks, with **no new crate edge** and no
     Ring-0 change. That is also exactly the brief's "the assets live with the
     engine samples and the island references them".
  2. **`edit_create_character` mints its own `Uuid::new_v4()`**
     (`scene/doc.rs:1961`) and the island needs `hero_guid(name)` — asserted at
     `island.rs:831`. It needs a guid-accepting variant (three call sites).
  3. **`StreamingSource { radius_m: 256 }` and `AlwaysLoaded` are not part of
     that door** and must be re-inserted after it; both are load-bearing (the
     partition activation anchor and the I3 collider band read the first).
  4. `island.rs:616-626`'s allowlist of `inf_island::` names grows, the level's
     dependency closure at `:867` grows, `INF_BLESS_SAMPLES=1` re-writes two
     `.inf_lvl` files, and the 900-step trace has to be re-priced — a skinned
     hero adds 6 476 B per step to `pose_state_bytes` where the capsule adds 0.
  5. The equivalence gate to copy is `samples.rs:11510`
     `the_showcase_character_matches_the_wizard_door`, field by field.
* **The starter character is not committed content.** Clause 3's other half:
  `samples/starter-character/` does not exist and
  `ProjectTemplate::starter_content` — the committed hook — is untouched. The
  wizard writes a complete character into any project on demand; nothing ships
  one in the repository. It is also clause 5's prerequisite (above).
* **Nothing binds a material to a skeletal mesh.** The body's `.inf_mat` is a
  sidecar dependency and a named slot, and `SkeletalMesh` has no material field,
  so the skinned draw path renders it with the projector's default. A component
  field is a scene-schema move and this wave's ruling was that none happen.
* **A pole-driven `solve_chain` on a hinged chain still loses 8.3 cm.**
  `grip::reach` routes around it for arms; `apply_foot_ik` passes `&[]` and so
  never hits it; an authored `IkTarget` over a chain with a hinge in the middle
  does. The general fix is to solve the parent's roll into the constrained
  joint's hinge plane, inside `solve_chain`.
* **35 of 795 body vertices are unreachable by the visibility oracle** (caps
  buried inside a neighbouring shell). They carry their seed bone, which is
  right; nothing has looked at whether a smarter shell layout would reduce it.
* **The weapon entity is a placeholder cube.** `ItemDef` has no mesh field —
  `item.rs:579` has named it as the next field since I6 — so an equipped weapon
  draws as a scaled primitive. *(SK1b audit: it was not scaled. See H1 below.)*
* **The muzzle is one fixed step behind the hand** (above).
* **The grip catalogue is a test fixture.** `grip_gate.rs` authors its four
  affordances by hand; no generator produces a `GripAffordance` and nothing in
  the editor edits one, so `SkeletonAsset::grips` is still empty on every rig the
  engine writes.
* **`HandIkRes` has no producer outside a test.** Nothing in the movement step,
  the weapon step or the `anim.*` node kit sets a hand request, so hand IK is
  reachable and unreached in the product — the same shape the SK1a audit recorded
  for the retarget map.
* **The aim mask has no consumer either.** It is authored onto the machine and no
  transition names it, because no aim or reload clip exists to name it from.
* **4 influences per vertex still stands** (the 16-attribute wall in the skinned
  pipeline is full). The generated hand puts a metacarpal and three phalanges
  near a palm slab, which is where the top-4 truncation would first bite; nothing
  measured it.
* **The two hands' `palm_in` mirrors in `x`** by a small amount, which is correct
  and was worth an arm: the first version of that assertion demanded they be
  equal and was wrong about its own subject.

### Counts

| | after the SK1a audit | **after SK1b** |
|---|---|---|
| battery blocks / passed / failed / ignored | 319 / 6 015 / 0 / 16 | **320 / 6 048 / 0 / 16** — **+1 block** (`grip_gate`, the wave's only new test file) and **+33 arms** |
| goldens | 54, byte-identical | **54, byte-identical** under `INF_GOLDEN_STRICT=1` over **101 arms**, no PNG rewritten over **101 arms**, no PNG rewritten. Nothing here touches a render path |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** |
| rustdoc warnings (ceiling 450) | 374 individual over 30 crates | **374 individual over 30 crates** after `cargo clean --doc`. **The wave adds zero**: it introduced **9** — four `[`clamp_to_limit`]` links to a private fn, two unresolved `IkError`/`IkReport` links in a new module, `starter_body`'s link to a private `skinned_copy`, `MUZZLE_HEIGHT_M`'s to a private `muzzle_of`, and one a *new public item* re-attributed out of a private doc that had been linking privately all along — and all nine were found and removed |
| `cargo fmt --all --check` | clean | clean |
| frontend tests / files | 702 / 78 | **untouched** — nothing under `editor/studio` moved. `LiveTuning.tsx`'s weapon-tunable gate is one-directional (panel → door), so the door's new `muzzle_forward_m` does not reach it |
| schema | `.inf_skel` v3, `.inf_anim` v2, `.inf_sm` v3, `.inf_mesh` v2, scene v25, `ScenePayload` v11 | **nothing moved** — the wave's whole point. Hand IK rides a resource, a weapon rides `AttachedTo` (already a scene component), and `WeaponDef` carries no `Serialize` at all |
| committed sample bytes | — | **unmoved**; `git status` on `samples/` and `tests/goldens/` empty |

### Commits

| | |
|---|---|
| `923b1a31` | the cone is enforced, and the hand reads its own geometry |
| `74495241` | hands reach through the fixed step, and the elbow was never a hinge |
| `943a0028` | a body, not a pile of boxes |
| `ab68f71c` | the weapon is an entity, the muzzle is its own, and the aim mask exists |
| `24f2fec2` | the grip gate: three things held, one of them let go of |
| `69016b88` | the ledger, and the route clause 5 did not take |
| `7eb912dc` | the twenty-first chr(92) catch, and it was mine |
| `e014b9b3` | the gates the wave has to pass, and the nine doc links it added |

**Nine** commits, not the eight this table names: a ledger row cannot name the
commit that writes it, which is the convention SK1a's own table follows and the
trap it fell into first time.

**The twenty-first chr(92) catch, and it was this wave's own.** Three `format!`
literals in `character.rs` carried eaten continuations — written from a non-raw
Python string, in patches that added a wave about measuring things honestly. The
workspace gate named all three on the first full battery, which is the gate doing
exactly its job for the twenty-first time. Twenty were ledgered across three
waves before it.

## The SK1b audit (2026-08-25)

Adversarial, `be85c213..30e93bcd`, fresh reader, nothing pushed. **Every headline
number this wave prints reproduced**, and the two it asks to be taken on trust
were re-derived rather than read. What the audit found is **one HIGH** — a
one-metre cube welded to every armed character in the tree — and six MED of two
shapes. One is the SK1a audit's: *an arm looser than the sentence it is written
under* (the aim mask's confinement, which was one inequality written twice). The
other is this wave's own, four times over: **a true claim, or a real law, with
nothing arming it** — the muzzle's two answers, the oracle's two spaces, the
portable ban's second crate, and the ROADMAP block that was never written. Two
further claims were true and unasserted and were **armed rather than fixed**: the
two hosts' step order, and the grip gate's distinct-pose count.

One HIGH and six MED, all fixed and all mutation-verified; thirteen LOW carried
by name.

### What reproduced, measured here rather than read

| the wave's figure | mine |
|---|---|
| battery 320 / 6 048 / 0 / 16 | **exact** |
| goldens 54, byte-identical, 101 arms | **54 / 101** under `INF_GOLDEN_STRICT=1`, no PNG rewritten |
| rustdoc 374 individual over 30 crates | **374 / 30** after `cargo clean --doc` |
| 9 commits in range, 8 named | **9**, and the ninth is the ledger's own count |
| +1 block, +33 arms | **exact** against the SK1a audited head |
| body: 795 kernel verts / 727 faces / 23 shells to 1 247 / 1 498 | **exact** |
| 1.749 m on a 1.75 m rig, sole to 0.8 mm, 0.0687 m3 | **1.749 / 0.0008 / 0.06874**; 1.199 at 1.2 and 2.399 at 2.4 |
| 760 assigned, 35 unreached, 1 247 skin rows, 0 non-deforming | **exact** |
| generate 0.62 ms, generate + bind + heat 433 ms (debug), 63 deform bones of 161 | **exact** |
| `Mask_AimOffset` 104 of 161 | **104** |
| grip gate 24 steps / 12 distinct poses / 6 476 B | **exact** |
| aperture ordering open > prop > handle > rifle | held, and the report's per-digit closure with it |
| the 38 digit cones are exactly the digit set | held, asserted as an identity |
| no schema moved; samples byte-unmoved; frontend untouched | **held** — `git status` empty on `samples/` and `tests/goldens/`, and no file under `editor/studio` is in the range |

Two the audit re-derived rather than trusted:

* **349 unreached through the narrowed oracle.** Measured here, exactly 349
  against 35, and now asserted in the crate that owns both halves (M2).
* **The digit derivation with no string compare.** Re-run on **both hands** of
  rigs at 1.2 m and 2.4 m and with non-default `arm_length_ratio` (0.31),
  `upper_limb_ratio` (0.40), `shoulder_width_m` (0.20) and `hip_width_m` (0.40):
  every joint of every chain is the bone the rig names, on all three. The
  spread-sort is a property of the derivation and not of the default proportions.

Also checked and held, because the brief asked: **the elbow-hinge fix on the
OTHER arm** (`a_hinged_arm_reaches_exactly_where_a_pole_solve_misses` runs both
sides, and the sign is asserted per side), and **the legacy 20-joint rig still
solves** — `grip::reach` on `BodyPlan::BipedCanonical`, whose elbow is still
`hinge_x` and whose arms hang down -Y, reaches a point 25 cm below and 35 cm in
front of the shoulder at **0.000000 m** on both arms, through the *name* rule,
with nothing clamped.

### HIGH

**H1 — the equipped weapon drew as a 1 m cube in the character's hand.**
`step_equipped_weapons` sets its placeholder primitive's `Transform::scale` to
the weapon's own barrel — 0.06 x 0.06 x 0.45 — and `inf_ecs::update_attachments`
overwrote it one pass later with the composed affine's scale, which is the
*target's*: `AttachedTo` carries an offset translation and an offset rotation and
**no scale at all**, so there was nothing of the follower's own in the number it
wrote. Every step, for ever.

It had been that way since P11.3 and nothing found it because until this wave the
engine had **no production `AttachedTo`** — every one in the tree is a test
fixture on a unit-scale target, where the destroyed value and the written value
are the same. The first real one is this weapon, and it put a metre cube on every
armed character in the tree, including the whole `phase30-gameplay` course, where
nothing was drawn at all before SK1b. The source comment said "scaled to the
weapon's own length" and no arm looked.

An attachment **places** a follower; it does not resize it. The pass writes
translation and rotation and leaves `scale` alone, with the bound stated at the
site: a follower on a *scaled* target does not inherit that scale — its
*placement* still does, because `socket_local` is model-space and `target_global`
scales it — and making the size follow needs a scale on `AttachedTo`, which is a
scene schema move. Destroying the follower's own size is not a substitute for
one.

Two arms, because "the scale survived" is satisfied perfectly by a pass that
wrote nothing: `an_attachment_places_a_follower_without_resizing_it` asserts the
barrel size on the local **and** the global **and** that the follower still landed
on the animated socket, and the weapon gate asserts the placeholder is the length
of the barrel it stands in for. Mutation: restore the `t.scale` write and the
first goes red at `(1.0000001, 1.0000001, 1.0)` — not even a clean unit, because
the socket matrix is `f32`.

### MED (all fixed)

| | |
|---|---|
| **M1** | the aim mask's confinement assertion was `masked.len() * 3 < rig.skeleton.len() * 3 && masked.len() < rig.skeleton.len()` — one inequality written twice, so the sentence above it ("more than a third of the rig is out") was asserted by nothing and a mask covering **160 of 161** joints passed. Pinned as the number: **104 of 161** |
| **M2** | **`mesh_soup` had no test at all.** The wave's fifth decision — *a visibility oracle must be built in the space its rays are cast in* — landed as a one-line change at one call site; the 349/35 pair lived in this memo, and the narrowing door `inf_editor_core::dcc::triangle_soup` carried no warning, so the next caller that hands a generated mesh to a BVH re-introduces it silently. Armed in the crate that owns both halves, and the door now names the hazard (below) |
| **M3** | the muzzle's fallback had **no tripwire**: a *rigged* hero whose skeleton stops publishing `hand_r` silently returns to 1.4 m, and the only muzzle arm in the tree ran on a rig-less capsule. `GameplayReport::muzzles_without_a_socket` counts a fallback taken by a character that **does** publish a pose, and an arm runs both branches on one mannequin (below) |
| **M4** | the **preview drag** went **0.126 to 4.70 ms cold, 37x**, when the wave put the full body generator on a path SK1a had measured and reasoned from — unremarked, under a 250 ms ceiling that is 53x the new number (below) |
| **M5** | **the portable-math gate does not cover `crates/inf-ecs/src/pose.rs`** — the door every pose writer reaches `pose_state_bytes` through, and where SK1b put `apply_hand_ik` and `solve_arm`. Two of that crate's files (`camera.rs`, `movement.rs`) were added by the P29.6 audit for exactly this reason and this one was not; the SK1a audit's completeness arm could not see the gap, because it enumerates `crates/inf-anim/src` and the file is in another crate. Added (**35** entries), and the file is clean under the ban — the only banned constructors in it are in `#[cfg(test)]` fixtures the stripper already removes. Mutation: a `.sin()` in `redrive` reddens the gate, naming the line |
| **M6** | **the wave wrote no ROADMAP block.** Every wave since P16 carries one in section 12 and SK1a's sits at line 25514; SK1b's diff touches `docs/memos/island-progress.md` and nothing else under `docs/`. Written, with the audit's own block beside it |

Two claims the audit **armed rather than fixed**, because they were true and
unasserted:

* **`both_fixed_steps_settle_the_weapon_after_the_pose`.** The wave states that
  the muzzle is one fixed step behind the hand and that this is acceptable because
  it is *"identical in both hosts, so no trace can see it"* — which is a claim
  about the order of three calls in two files, and a PIE == shipping gate is
  structurally blind to exactly what both hosts do the same way. Pinned on the
  `both_fixed_steps_run_the_cloth_slot` precedent: gameplay < pose < attachments,
  in both `fixed_step`s. Mutation: move `update_attachments` above `step_gameplay`
  in the editor Simulate and it goes red naming the three offsets.
* **The grip gate's "12 distinct poses"** was printed and not asserted, so a
  solver that collapsed every grip onto one pose could keep the handful of
  `assert_ne!` pairs apart and pass. Pinned at 12 of 24, and the byte length with
  it — 6 476 is the 161-bone trace SK1a priced, so a rig that silently lost its
  side tables would otherwise look like a quieter grip.

### M2 — the oracle seam, measured in the crate that owns it

`the_narrowed_oracle_cannot_see_a_third_of_a_generated_body` builds the same
generated body's oracle twice — from `mesh_soup`, and from the exporter's
triangles read back at `f32` and widened, which is precisely the soup
`triangle_soup` hands over — and runs the same masked heat solve against each:

| oracle | unreached, of 795 |
|---|---|
| `mesh_soup` (f64 kernel triangles) | **35** |
| the f32 round trip | **349** |

Asserted as a triple with the ratio beside it, so an arm that stopped
demonstrating the seam fails rather than passing quietly. The only cover before
was an `unreached < 60` bound two crates away — which does catch the regression,
verified by mutation (routing `starter_body` back through the narrowed soup fails
`a_build_writes_six_assets_and_wires_them_together` at 349) — but says nothing
about why. `triangle_soup`'s docs now carry the mechanism, the measurement and
the rule: an author's imported model, yes; anything `body_mesh` or a grammar bake
produced, `mesh_soup`.

### M3 — the muzzle's silent half

`a_rigged_hero_shoots_from_its_weapon_and_says_so_when_it_cannot` runs both
branches on one mannequin, differing only in what the skeleton publishes:

| the rig's right-hand socket | where the shot left | counter |
|---|---|---|
| `hand_r` | **0.957 m** from where the capsule rule would put it | 0 |
| `hand_of_glory` | the capsule rule, to 1e-12 | **1** |

Its fixture also gives `weapon_3d.rs` its first **rigged** hero — the pose slot
and `update_attachments` in the hosts' own order, inert for every other arm in
that file — which is what makes the one-fixed-step latency observable at all: the
shot is taken on the second step, because the first one is what places the
weapon.

A character with **no pose at all** is the legitimate capsule case and is not
counted, so every level committed before this wave still reports zero.

### M4 — a measurement the wave replaced without taking

`body_for` called `block_body_mesh` and calls `inf_dcc::body_mesh` +
`to_mesh_asset` now, so the counts the wizard's preview reports are the counts
the build writes. That is right, and it is 37x the path SK1a measured:

| | SK1a | SK1b |
|---|---|---|
| cold, 20 joints | 0.023 ms | 0.037 ms |
| cold, 161 joints | 0.126 ms | **4.70 ms** |
| warm, 161 joints | 0.022 ms | 0.024 ms |

(4.66 / 4.69 / 4.71 over three unloaded runs; **8.4 ms** with the rest of the
battery running beside it.) **Nothing is owed** — the sliders debounce at 250 ms,
so 4.7 ms is under 2 % of the interval between two previews and the warm path did
not move — but SK1a's stated *reason* ("the weight solver is not on this path at
all") stopped being the whole of it, and the ceiling was 53x the new number.
It is 100 ms now: about 21x the measurement, about 4x a runner five times slower
than this machine, and 12x the loaded reading. The counters stay the real
assertion.

### The mutations the audit ran itself

| mutation | expected | result |
|---|---|---|
| `swing_deg: 0.0` on every finger cone | the curl arm catches it | **red**, and so do the cone arm, the census arm and **both** grip-gate arms |
| swap `redrive` above `apply_hand_ik` | the pose-writer pin goes red | **red** |
| disable the `redrive` call | the twist arm goes red | **red** — "the twists were driven from the pose before the IK corrected it" |
| flip the sign of `spread` in `hand_of` | the digit derivation catches it | **red**, and the pinky/aperture arm with it |
| drop the generator's rigid prior | the skin-stream arm goes red | **red** — **62** vertices on a bone that deforms nothing, against 0 |
| route `starter_body` back through the narrowed soup | the weights arm goes red | **red** at **349** unreached |
| restore `t.scale` in `update_attachments` | H1's arm goes red | **red** at `(1.0000001, 1.0000001, 1.0)` |
| move `update_attachments` above `step_gameplay` in Simulate | the new host-order pin goes red | **red**, naming the three offsets |
| a `.sin()` in `inf_ecs::pose::redrive` | the portable gate goes red | **red** once `inf_ecs::pose` is on `SIM_PATH`; **green** before, which is the finding |
| truncate a `Pose` below its skeleton and call `grip::reach` | a refusal, or a panic | **panic** — "len is 10 but the index is 58" (LOW, and `solve_chain`'s own convention) |

### LOW, carried by name

* **`HandIkRes` has no producer outside a test**, and the ledger says so loudly.
  Verified by sweep: `set_hand_ik` appears only in `grip_gate.rs` and `pose.rs`'s
  own tests; nothing in the movement step, the weapon step or the `anim.*` node
  kit sets a hand request. Hand IK is reachable and unreached in the product —
  the same shape the SK1a audit recorded for the retarget map, one wave later.
  Neither `phase29_gate` nor `phase30_gameplay_gate` implies otherwise; the only
  gate that exercises it is the one this wave wrote.
* **The aim mask has no consumer either.** `AIM_MASK` appears at its definition,
  at the wizard's authoring call and in tests. Nothing reads it.
* **The two hosts run `apply_root_motion` and `advance_state_machines` in
  OPPOSITE order**, and the editor propagates between them where the player does
  not. Pre-existing (not SK1b's), and **unmeasured** — the P29 traces are green,
  so either it does not reach the traced state or those courses do not exercise
  root motion off an `AnimPlayer`. Found while writing the host-order pin, which
  deliberately does not cover it: extending a pin to a divergence nobody has
  measured is how a gate goes red for a reason nobody can name.
* **A despawned character orphans its weapon entity.** `step_equipped_weapons`
  iterates `movement_targets`, so a character that leaves the world takes its
  owner row with it and the weapon stays, attached to a guid
  `update_attachments` skips — frozen in place, visible, and in the trace.
  Unequipping is covered; dying is not (a ragdolled character keeps
  `CharacterMovement` and so keeps its weapon, which is right).
* **The re-drive doubles a cost SK1a had already carried as unmeasured.** The
  SK1a audit's list says *"the fixed-step cost of the drive pass is argued, not
  measured"* — `drive_ik_follow` adds a full `global_transforms` pass and a
  161-entry allocation per posed character per step. `redrive` runs the whole
  pass a second time, and its gate (`corrected`) opens for **any** correction,
  not only a hand one: a mannequin with foot IK or a pelvis drop pays it every
  step whether or not anything touched its hands. The wave measured what a
  *gripping hand* costs (3.2 / 3.3 / 6.2 µs) and did not measure this. Correct,
  and still a number that does not exist.
* **`grip::reach` indexes `pose.locals` without a bounds check** and panics on a
  pose shorter than its skeleton (measured: "len is 10 but the index is 58").
  `solve_chain` does the same, so this is the crate's convention rather than a
  new hazard — but `apply_grip` guards, and three doors in one module do not
  agree.
* **`drive_ik_follow` inside `apply_hand_ik` is a pose writer the frozen-order
  pin cannot see** (the pin reads `step_pose_evaluation`'s own body). Its
  position between the reach and the gun solve is load-bearing and *is* covered
  behaviourally — `the_off_hand_follows_the_weapon...` asserts 0.30 +/- 0.06 and
  the un-re-driven answer is 0.42 — so this is a note about the pin's reach, not
  an uncovered writer.
* **`clamp_to_cone` does not check `twist_deg` for finiteness** where it checks
  the axis and `swing_deg`. Safe by accident: a NaN twist range produces a NaN
  quaternion that the closing `is_finite` guard turns into "leave the joint
  alone". Safe, and not by the argument the function's own docs give.
* **`apply_hand_ik` uses the `ik_hand_gun` frame only when the holding hand is
  the RIGHT one** (the handle follows `hand_r`, so this is correct), and the
  function's doc says "the `ik_hand_gun` handle's frame when the rig publishes
  one" without the side condition.
* **`step_equipped_weapons` re-inserts four components on the weapon entity every
  fixed step**, so change detection fires for it on every step of every armed
  character. Inert-looking and not measured.
* **`the_new_muzzle_agrees_with_the_old_one_on_a_capsule_hero`'s named mutation
  is imprecise**: it says dropping the `evaluated_pose` guard makes an unposed
  character read "its capsule **centre**, 20 cm low", but that fixture never runs
  `update_attachments`, so the weapon entity is at the **world origin**. The
  mutation still reddens the arm; the number in the sentence is not the one it
  would print.
* **The grip gate's two hosts are two host IMPLEMENTATIONS in one process**, not
  two processes. `player_trace` builds a `RuntimeSim` and `editor_trace` a
  `SimSession`, both in the test binary; the module doc says "across two
  processes' worth of hosts", which is a hedge doing some work. The claim it
  makes — PIE == shipping — is honest and is what the phrase means; what is
  absent is the *real `--pie` subprocess* arm phase21 and phase22 both carry, and
  which is what catches a host that only agrees because a `OnceLock` was already
  warm. No such state is on this path today.
* **The preview body cache is keyed on the whole `BodyParams`**, so a proportion
  drag misses on counts that (on the mannequin) do not depend on any dimension —
  `the_body_follows_its_rig_and_is_reproducible` asserts exactly that invariance.
  A topology-shaped key would make the drag warm. Named, not taken: "correct for
  both generators" is a claim that needs its own arm.

### Decisions (the SK1b audit's, binding on later waves)

1. **An attachment places a follower; it does not resize it.** A door that
   composes a transform onto something else's must write only the parts it
   actually knows about — `AttachedTo` knows a position and a rotation, so those
   are what it writes.
2. **A defect that only a production caller can show is a defect that waits.**
   `update_attachments` had six arms and every one of them attached a unit-scale
   fixture to a unit-scale target, where the bug is the identity.
3. **A law is not enforced by the call site that obeys it.** SK1b's oracle
   decision was implemented at one caller and left the narrowing door silent; a
   decision needs an arm in the crate that owns it and a warning at the door that
   breaks it.
4. **A claim about two hosts' source order cannot be checked by comparing two
   hosts.** A PIE == shipping gate is blind by construction to whatever both
   hosts do identically, which is precisely the class "identical in both hosts,
   so no trace can see it" belongs to.
5. **Re-measure the number your predecessor reasoned from.** SK1a's preview
   conclusion rested on *which code was on the path*; SK1b changed the path and
   kept the conclusion.
6. **A completeness arm covers the directory it enumerates, and no more.** The
   portable-math gate's domain is two crates and its enumeration is one, so the
   half it does not walk is the half a new file lands in unnoticed.

### Counts

| | after the SK1b wave | **after the audit** |
|---|---|---|
| battery blocks / passed / failed / ignored | 320 / 6 048 / 0 / 16 | **320 / 6 052 / 0 / 16** — **four** arms, **no new block**: one in `inf-ecs`'s `attach`, one in `inf-physics`'s `weapon_3d`, one in `inf-dcc`'s `body`, one in the `projector_mirror` host mirror |
| goldens | 54, byte-identical, 101 arms | **54, byte-identical under `INF_GOLDEN_STRICT=1`** over 101 arms, no PNG rewritten. Nothing here touches a render path |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** |
| rustdoc warnings (ceiling 450) | 374 individual over 30 crates | **374 individual over 30 crates** after `cargo clean --doc`. **The audit adds zero** |
| `cargo fmt --all --check` | clean | clean |
| frontend tests / files | untouched | **untouched** — nothing under `editor/studio` moved here either |
| schema | `.inf_skel` v3, scene v25, `ScenePayload` v11 | **unmoved** |
| committed sample bytes | unmoved | **unmoved**; `git status` on `samples/` and `tests/goldens/` empty |
| chr(92) | the twenty-first was the wave's own | **no twenty-second** — the wave's added lines and the audit's own were swept for both shapes (interior space runs at and below the gate's own 8-space threshold, and bare newlines inside non-raw literals): zero, and the workspace gate is green |

## Wave SK1c (the hero is a character, and the hands have a reason)

Base `2452ce20`. Five clauses; **four landed and the fifth is priced and
stopped**, which is what its own brief asked for.

The wave's headline is that SK1b's two loudest carried items — *`HandIkRes` has
no producer outside a test* and *the grip catalogue is a test fixture* — are
closed together, because they are one item: a catalogue with no producer and a
consumer with no caller are the same feature missing its two ends. The second
headline is a **measurement**: the two hosts' opposite pass order, which SK1b's
audit deliberately left unmeasured rather than pinning, does not commute, and it
took ten minutes to build the trace that says so.

### Clause 4 — the opposite-order landmine, measured (`d6f57f49`)

The SK1b audit recorded, as a LOW it would not pin, that the shipped player runs
`advance_state_machines` then `apply_root_motion` while the editor Simulate runs
them the other way round — and that the two agreed on every committed trace,
which says nothing about whether the passes *commute*.

**They do not.** `step_pose_evaluation` reads the entity's `GlobalTransform`
twice — `authored_ik_goals` inverts it to bring a world-space goal into model
space (`pose.rs:592`, `:610`), and `model_to_world` feeds the foot pass, the hand
pass and the feet it publishes (`pose.rs:1479`) — so the order decides which
step's placement all of those are computed in.

The fixture is three things, none of them exotic: `RootMotion` +
`AnimPlayer` (the only shape `apply_root_motion` acts on at all, and
`samples/character-demo` already carries the component), an `AnimStateMachine`
(so the pose pass does work), and an authored world-space `IkTarget` (so the pose
depends on where the entity *is*). Measured at the SK1b head:

| step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| bytes equal | no | no | no | no | no | no | no | no |
| worst pose component | 0.027 | 0.035 | **0.060** | 0.055 | 0.049 | 0.043 | 0.037 | 0.031 |

Different pose bytes on **every one of eight steps**, while the transform itself
agreed to the bit — a divergence entirely inside the pose, which is exactly where
a `RootMotion` component looks inert.

**Nothing found it because no gate fixture in the tree carries `RootMotion` at
all**, and the one committed sample that does has no `AnimPlayer`, so
`apply_root_motion` returns at its first line and the two hosts perform an
identical no-op in two different places.

Unified on the editor's order, so the **shipped player moves**: root motion is
*movement*, and every other movement in this engine happens before the pose —
`step_character_movement`, `step_gameplay`, and the one-step latency
`anim_bridge`'s own doc rests on. The propagate between them is not decoration:
`apply_root_motion` writes `Transform` and the pose reads `GlobalTransform`, so
without it the reorder is inert and the two hosts agree only by both being wrong.

Two arms, because a source pin and a trace catch different things:
`both_fixed_steps_move_the_root_before_the_pose` asserts play-heads < root motion
< pose **and a propagate between the middle two**, in both files; and
`both_hosts_pose_a_root_motion_driven_character_the_same_way` is the trace, with
its travel asserted so an unregistered clip cannot make it vacuous. **The
mutation is the measurement**: at the SK1b head the trace arm is red on step 0.

### Clause 3c — the grip catalogue is generated (`0b10fc4c`)

`SkeletonAsset::grips` shipped in SK1a empty on every rig, got its runtime
consumer in SK1b, and still had no producer: the only two catalogues in the tree
were a fixture in `grip_gate.rs` and a second, differently-named one in
`pose.rs`'s own tests — two hand-written tables for one idea, with different names
and different apertures.

`inf_anim::grip::grip_catalogue` writes it, and the reason it *can* be generated
is the type's own doctrine: **a `GripAffordance` is a property of the hand**. An
aperture is how wide the fingers open and a curl set is how far each closes;
neither is a property of the door. So the only rig-dependent thing in the table is
which joint is a hand, and that comes off the role table rather than off a name.

Four names, exported as consts so the three sites that spell them cannot
disagree: `handle` (4.5 cm bar, thumb wrapped), `rifle` (0.032, **trigger finger
straight**), `rifle_fore` (the *off* hand — both on one wrist and a two-handed
hold closes the same fingers twice), `prop` (a 9 cm ball).

The four numbers moved **unchanged**, and that is asserted rather than asserted by
me: the whole grip gate is byte-green on the generated table, and
`a_generated_rig_is_the_catalogue_this_gate_takes` pins each aperture — because
the gate's aperture arm asserts an *order* and its PIE == shipping arm compares
two hosts, and both would survive a generator that moved every number together.

**No committed byte moved**: the two `.inf_skel` files in the tree are a
`BipedCanonical` rig and a six-joint placeholder, neither of which goes through
`build_manny`.

### Clause 1 — the engine ships a character (`315d01c8`)

`samples/starter-character/` exists: **nineteen files**, the exact output of
pressing New Character and accepting every default, on the 161-bone mannequin.

| | |
|---|---|
| the rig | `Starter.inf_skel`, **23 026 B** — 161 bones, role table, twists, IK follows, hand cones, grip catalogue |
| the body | `Starter_Body.inf_mesh`, **95 932 B** — generated and heat-weighted |
| the skin | `Starter_Skin.inf_mat`, 50 B |
| the cycles | `Starter_Idle/Walk/Run.inf_anim`, 8 810 / 4 387 / 3 366 B — **derived** |
| the machine | `Starter_Locomotion.inf_sm` 837 B + its 4 654 B text face |
| the controller | `Starter_Controller.inf_act`, 9 359 B |
| the tables | `camera.toml` 3 486 B, `input.toml` 2 423 B |
| total committed | **~155 KB** over 19 files + a README |

**It is generated by the wizard's own door, not by a mirror of it.** Every other
sample in `samples.rs` is a pure generator with an equivalence arm standing
between it and the tool; a character is eight assets, a heat solve, a derivation
and a proposal, and mirroring that is how two generators end up disagreeing.
`starter_character_files` runs `build_character_with_ids` into a scratch project
and copies the result out, so the committed bytes **are** `build_character`'s
output and the byte lock is a lock on the wizard.

That needed one new thing. A character's eight assets name each other by GUID —
the body's skin binding names the rig, every clip carries the rig's id, the
machine names three clips, the controller names the machine — so a build with
minted ids is a different file set every time and nothing committed could ever be
re-blessed. `AssetProject::write_asset_with_id` and `CharacterIds` supply them,
and every interactive path still mints, because two assets sharing a GUID is a
database whose reverse edges lie.

**Two defects the committed output found**, both older than this wave and both
invisible until something wrote the wizard's files down and looked at them:

* **The machine's text face sat next to nothing.** `AssetProject::write_asset`
  runs a display name through `sanitize`, so "My Hero" writes
  `My_Hero_Locomotion.inf_sm` — and the wizard placed its `.txt` at a path
  rebuilt from the *display* name. `sm_text`'s whole convention is
  `<payload>.txt`, so `read_text` on the real payload answered `None` and the
  reviewable face pillar S1 argues for was silently absent. **Every
  wizard-default name has a space in it.**
* **The controller wrote its own file with a raw `dir.join`**, so a character came
  out as six `My_Hero_*` assets and one `My Hero Controller.inf_act` — a content
  folder that looks like two authors were in it.

`every_file_the_wizard_writes_is_named_the_same_way` is the regression arm, and
its fixture is called "My Hero" deliberately: one called "Hero" passes with the
defect in place.

The advisory the build reports is **handed back rather than swallowed or refused
on** — exactly one, SK1b's carried 35-of-795 unreachable cap vertices — and pinned
by content, because "no warnings" could only be bought by silencing it and a count
of one says nothing about which one.

`ProjectTemplate::starter_content` scaffolds **seventeen** of the nineteen into
every 3D project's `Content/Characters/` (`camera.toml` and `input.toml` are
project tables with no home under `Content/`). The 2D platformer keeps Coyote: a
161-bone biped in a side-scroller is 155 KB an author has to notice in order to
delete. `every_3d_template_ships_the_whole_starter_character` reads the sample
folder **off disk** and asserts the `include_bytes!` table is the whole of it,
because a character missing one sidecar is a project whose rig resolves and whose
clips do not.

The wizard references it three ways: the plan step says so in the dialog, the
templates scaffold it, and `the_starter_character_is_what_the_wizard_opens_with`
asserts the spec is `CharacterSpec::default()` field by field with the name as the
single deliberate difference — so a moved default keeps that arm green and turns
the byte lock red, which is the right pair.

**`EXPECTED_LEVELS` is untouched**: this folder is content, not a level.

### Clause 2 — the island hero is that character (`19cf44be`)

`island.rs` spawned forty lines of hand-rolled components ending in
`AnimStateMachine { sm: None }` and no `SkeletalMesh` — a hero that walked, drew
nothing and posed nothing — because the one door that knows how to build a
character minted its own entity GUID and the island derives every one of its own.
`edit_create_character_with_guid` takes one.

The scouted route held **exactly**, and was narrower than the scout thought: the
assets reach a built project through the recipe's `[content]` list, which
`write_content` already copies, so there is no new crate edge, no Ring-0 change
and — the one place the scout over-estimated — **the `island.rs:616` allowlist did
not have to grow at all**, because this file names GUIDs rather than
`inf_island::` items.

`HERO_HEIGHT_M` is gone. It was 1.8 while there was nothing inside the capsule;
the wizard's default is 1.75, and a 1.75 m body in a 1.8 m capsule floats 5 cm off
the ground it stands on. The height is read from the starter character's own spec.

**The bless, arithmetic-verified:**

| | before | after | delta |
|---|---|---|---|
| `samples/island/VancouverIsland.inf_lvl` | 14 820 B | 14 890 B | **+70** |
| `samples/island-fixture/IslandFixture.inf_lvl` | 8 134 B | 8 204 B | **+70** |
| both `.inf_lvl.toml` dependency lists | 3 | 6 | +3 |

**+70 is four GUIDs and two Option tags.** bincode writes a `Uuid` as a
length-prefixed 16-byte string (17 B); the rig, the body, the machine and the
controller are 4 × 17 = 68, and `SkeletalMesh`'s two inner `Option` tags go from
absent-inside-a-`None` to present, +2. 68 + 2 = 70, twice, one hero each. The
`.inf_act` is deliberately **not** in the dependency list: `level_dependencies`
walks asset references on components and `ActorClass` is not one, which is why
`samples/phase29-locomotion`'s own sidecar lists three and not four. Asserted in
both directions, so the day it changes the comment fails rather than rots.

**The 900-step trace, re-priced honestly:**

| | before | after |
|---|---|---|
| bytes a state | **403** | **6 879** |
| the hero's pose section | 0 | **6 476** |
| distinct states of 900 | 900 | 900 |
| PIE == shipping | green | **green** |

+6 476 a step, exactly the figure the brief carried, and exactly SK1a's
arithmetic for a 161-bone rig (36 B header + 40 B a joint). A **17×** growth in
what the gate compares. Pinned as the number and not as `> 0`: a byte equality is
blind to two hosts posing nothing identically, which is what this gate did for its
whole life until now.

**Two gate-side gaps the swap found**, both the same shape — a fixture host poorer
than the one it is compared against:

* `loose_sim` never called `with_anim_assets`, so the editor-side host had no
  skeletons, no clips and no machines where the cooked host had all three. Its own
  doc already carried the rule — *two hosts compared for byte equality must be
  given the same world to disagree about, or the equality is between one real
  reading and one impoverished one* — and the anim index was the third thing it
  was missing. Invisible while the hero had nothing to pose; **step 0 red** the
  moment it did.
* `pie_sim` passed `|_| None` for the blueprint-class and anim resolvers. It reads
  the **sidecars** in the content root now — the same index `AssetDb`'s own scan
  reads — rather than a name table, which would be a second place the character's
  identity is written down. Asserted: **1 skeleton, 1 machine, 3 clips** (through
  the transitive machine→clip hop), **1 class**.

### Clause 3a/3b — hand IK gets its producers (`86e0bfec`)

**One producer**, in `step_gameplay`, which is the one Ring-0 rule both hosts call
and already sits before the pose by a pinned order. One and not two: a weapon
wants both hands and a grab wants one, and two producers writing into the same
two-slot array would race every step with the winner decided by call order. The
rule between them is written down rather than emergent — **the weapon owns the
hand it is in and a grab takes the other one** — so a character reaching for a
door handle with a rifle in its right hand reaches with its left.

* **EQUIP** puts the `GunGrip` two-handed hold on the rig (`ik_hand_gun`) and
  closes each hand on its own affordance. The fore-grip offset is two thirds of
  the weapon's own length, clamped to `0.12..=0.60` m, so a pistol does not ask
  the off hand to occupy the on hand's space.
* **AIM** drives the reach, and only aiming does. A carried weapon hangs where the
  animation puts it; RMB brings it to a point on the aim line at 0.82 of the
  character's own stand height, 0.42 m in front. That line is
  `inf_ecs::weapon::aim_forward`, **factored out of `shot_direction`** so the hand
  and the bullet cannot point in different directions.
* **E-GRAB**: `Interactable` carries a grip *name*, `InteractCandidate` and
  `InteractHit` carry it through, and the press latches
  `interact::HandGrabRes` — 0.25 s in, 0.5 s held, 0.25 s out. Recorded **before**
  the verb match, because the hand is orthogonal to the consequence — which is
  also how `InteractVerb::Grab`, the one verb this engine has and does not
  consume, stops doing literally nothing.

Two producers of grip *names*, so the catalogue has consumers as well as a
generator: a door's `Interactable` names `handle` (`d3/door.rs:254`), a dropped
item's names `prop` (`item.rs:591`). A kick names none (a kick is a leg); a
vehicle seat names none (a seat is a whole-body choreography).

**The gate**: `runtime/inf-player/tests/weapon_hands_gate.rs` — equip → aim →
fire → reload → unequip → E-grab on a **rigged** hero.

| | |
|---|---|
| steps | **140** (a grab is a whole second at 60 Hz, and the point is that it opens again) |
| distinct poses | **18**, pinned as the number |
| bytes a step | **6 476** |
| PIE == shipping | on the pose **and** on `(holds, grabs, shots, reloads)` |
| shots / reloads over the course | 1 / 1, magazine 5 → 4 → 5 |

Its anti-vacuity is most of it. **Carrying and aiming are separate bands**,
because "the pose changed since idle" is satisfied by the equip alone — the aim
arm compares two settled bands and the settling of each is asserted. The grab is
asserted to **ease** (closing further moves the pose again) rather than snap. And
the release is asserted **byte-identical** to the pose before anything was picked
up — the claim `apply_grip`'s "a curl is a pose, not a delta" rests on, which a
drifting solver would fail after passing every `assert_ne!` above.
`the_hand_pass_costs_an_unarmed_character_nothing` runs two worlds identical but
for a rifle in the bag, with the armed one asserted to diverge so the loop is not
a statement about the fixture.

**The ground in that fixture is load-bearing and says so.**
`RotationMode::Aiming` is set on the *grounded* movement branch, so a hero
standing on nothing never aims and the aim half of the course would have been
measuring an unpressed button. The first run did exactly that.

`inf_physics::d3::gameplay` joins the portable-math ban list (**36** entries): it
computes a point that lands in the solved pose and therefore in
`pose_state_bytes`. The SK1b audit added `inf_ecs::pose` for this reason and made
it a law; this is the same law meeting a **third** crate. Mutation: a `.sin()` in
`aim_hold_point` reddens it, naming the line.

`SimSession::gameplay()` is the mirror of `RuntimeSim::gameplay()` — the field had
been kept since I6 with no reader outside its own file, so a two-host gate could
read the shipped player's counters and had nothing to compare them against.

**The gates that watched the capsule hero are untouched and still green**:
`phase30_gameplay_gate` (3), `weapon_3d` (12), `door_3d` (13), `grip_gate` (3),
`phase29_gate` (6). The rigged course is a *new* gate beside them, which is the
reading this wave took of "must still pass on a rigged hero": the capsule fixtures
keep certifying the capsule path — which is every level committed before SK1b —
and the rigged one certifies the hands.

### Clause 5 — the weapon's visible mesh: PRICED, and STOPPED

The brief's condition was "take it **only** if it rides an existing mechanism with
no schema move". It does not. The schema half is free and every other half is not.

**What is free.** `ItemDef` and `WeaponDef` carry no `Serialize`, ride no wire and
are built from TOML, so a `mesh: Option<Uuid>` field costs **no schema bump** —
the same slot `muzzle_forward_m` used in SK1b. `MeshRef` already carries
`asset: Option<Uuid>`. Setting it on the spawned weapon entity is four lines.

**What is not.** Four blockers, none of them inside this wave's boundary:

| # | the blocker | evidence |
|---|---|---|
| 1 | `RenderScene` has **one** door for non-primitive geometry — virtualized geometry — so a mesh with no derived `.inf_vmesh` draws as a placeholder cube | `assets/vmesh.rs:63-75` |
| 2 | the **cook** derives one only above `VgeomCookOptions::min_triangles` = **2048**; a rifle is one or two hundred | `cook.rs:126,135`; the advisory at `cook.rs:1604` says the words "renders it as a PLACEHOLDER CUBE" |
| 3 | **PIE streams no vmesh assets at all**, so a rigid `MeshRef.asset` is a cube in PIE whatever the cook did | `window.rs:1029-1032` |
| 4 | the cook's `asset_deps` walks Level, Material, StateMachine, AnimClip, BiomeSet and Pcg — **not Blueprint** — and an item catalogue is authored in a `.inf_act`, so a mesh the catalogue names would never enter the closure and would not be packed at all | `cook.rs:1849-1998` |

So taking it would ship: a cube in PIE (3), a cube in the shipped build (1+2), and
most likely **no asset in the pack** (4) — three cubes and a dangling reference,
for four lines that look like a feature. The editor viewport would draw the rifle
correctly (its own threshold is 1 triangle, `vmesh.rs:75`), which is the worst
shape a defect can have and is precisely what `sub_threshold_advisory` was written
to shout about.

**The honest one-line fix is `min_triangles`**, and it is not this wave's: it
changes shipped bytes for every sample in the tree, it is a cook default with a
stated cost rationale, and the P18.3 audit already ledgered lowering it as a
follow-up. Carried by name, with the four blockers above, so the wave that takes
it takes all of them at once.

### Decisions (SK1c's, binding on later waves)

1. **Two passes that agree on every trace have not been shown to commute.** They
   have been shown that no trace exercises both. Build the trace before pinning
   the order — and if it diverges, the order is a fact, not a preference.
2. **Root motion is movement.** It happens before the pose, with a propagate
   between, in both hosts. A pose is a statement about where the character *is*.
3. **A committed generator output must be reproducible, which means its GUIDs are
   an input.** An asset set that names itself by GUID cannot be byte-locked
   against the door that wrote it otherwise.
4. **A generated catalogue beats a fixture, and the fixture's numbers are what it
   generates.** Moving the numbers and the producer in one step is how the
   measurements survive the move.
5. **One producer per resource slot.** Two writers into one two-slot array is a
   race whose winner is call order; compose them once and write the precedence
   down.
6. **The interaction and its consequence are separate.** A hand goes on a thing
   whatever the verb then does with it, which is what gives a verb with no
   consumer something real to do.
7. **A fixture host must be as rich as the host it is compared against.** The
   island gate's loose side was missing its anim index for as long as the hero had
   nothing to pose, and no arm could have said so.

### What SK1c did not do (for its audit and its successor)

* **The weapon is still a placeholder cube.** Priced above, four blockers, none of
  them a schema move and none of them in this wave's boundary.
* **The aim mask still has no consumer.** `Mask_AimOffset` is authored onto every
  generated machine (including the committed starter character's) and no
  transition names it, because no aim or reload clip exists to name it from.
  Carried by name for the weapons wave, as SK1b's brief said.
* **The grab is a gesture, not a carry.** The hand reaches, closes, holds and
  opens; the engine has no "carrying" mode and inventing one here would be a mode
  nothing can leave. A `PickUp` still moves the item into the bag on the press —
  the hand and the inventory are two things that happen, not one.
* **`GripAffordance::palm` is still read by nothing.** `apply_hand_ik` uses
  `name`, `hand`, `aperture_m` and `curl`; the palm frame would refine where the
  weapon entity sits relative to the socket, which is `AttachedTo`'s offset and
  needs the rig at a step that does not have it.
* **The island hero has no locomotion clips bound to *its* rig.** The starter
  character's three cycles are generated for the mannequin and committed, and the
  island's machine plays them — but nothing in the 900-step drive checks that the
  hero's feet match its motion, because the drive writes the transform directly.
* **`camera.toml` and `input.toml` are not scaffolded** into a new project by
  `starter_content`: both are project tables with no home under `Content/`. An
  author who wants to tune either copies them out of the sample.
* **The starter character is 155 KB in every binary that links `inf-project`**,
  which includes the shipped player. Measured, not argued: `Starter_Body.inf_mesh`
  is 95 932 B of it. A feature-gate is the obvious lever and was not pulled.
* **35 of 795 body vertices are still unreachable by the visibility oracle** —
  SK1b's carried item, now pinned by content as the starter character's one
  advisory rather than left as a number in a memo.
* **`Interactable::grip` is authored by two generators and by no editor.** A door
  and a pickup name a grip; an author placing an `Interactable` by hand gets
  `None`, and there is no Details field for it (the component is runtime, so there
  could not be one without a schema move).
* **The E-grab reaches the interaction's own point, not a handle.** A door's is
  the middle of its closed opening at mid-height (`door::prompt_position`), which
  is where the prompt is measured from and is not where a handle is. The engine
  has no handle: a door is a hinge and a spec box.

### Counts

| | after the SK1b audit | **after SK1c** |
|---|---|---|
| battery blocks / passed / failed / ignored | 320 / 6 052 / 0 / 16 | **321 / 6 067 / 0 / 16** — **+1 block** (`weapon_hands_gate`, the wave's only new test file) and **+15 arms** |
| goldens | 54, byte-identical | **54, byte-identical** under `INF_GOLDEN_STRICT=1` — no render path is touched, and no golden can see the hero: `inf-render` names neither `inf-island` nor `inf-editor-core`, and its only `inf_anim` use is a hand-built three-joint cylinder |
| `clippy --workspace --all-targets` `-D warnings` | 0 | **0** |
| rustdoc warnings (ceiling 450) | 374 over 30 crates | **374 over 30 crates** after `cargo clean --doc`. **The wave adds zero**: it introduced **one** — an intra-doc link from `SimSession::gameplay` to `RuntimeSim::gameplay`, which is downstream of this crate and therefore unlinkable — and it was found and named instead |
| `cargo fmt --all --check` | clean | clean |
| frontend tests / files | 702 / 78 | **702 / 78**, unchanged — one dialog line added, `tsc` and `eslint` clean |
| schema | `.inf_skel` v3, `.inf_anim` v2, `.inf_sm` v3, `.inf_mesh` v2, scene v25, `ScenePayload` v11 | **nothing moved.** `Interactable`, `HandGrabRes` and `HandIkRes` are runtime; `ItemDef`/`WeaponDef` carry no `Serialize`; the starter character is content |
| committed sample bytes | unmoved | **two `.inf_lvl` +70 B each** (arithmetic above) and **19 new files** under `samples/starter-character/`. Nothing else moved |
| `EXPECTED_LEVELS` | 23 | **23** — the new folder is content, not a level |
| chr(92) | the twenty-first was the wave's own | **the twenty-SECOND and twenty-THIRD, and both were mine** — two eaten continuations in `island.rs` and `samples.rs`, written from a *raw* Python string where a single backslash is what a Rust continuation needs and a doubled one is what a raw string preserves: the mirror image of the mistake the law was written about. Caught by the workspace gate on the first full battery. Every `.rs` file this wave touched was then swept for both shapes: one hit, pre-existing, in a literal that spells its own indentation on purpose |

### Commits

| | |
|---|---|
| `d6f57f49` | the two hosts did not commute, and it took a trace to say so |
| `0b10fc4c` | the grip catalogue is generated, not authored in a test |
| `315d01c8` | the engine ships a character, and it is the wizard's own |
| `19cf44be` | the island's hero stops being a capsule |
| `86e0bfec` | hand IK gets its producers, and the hands go on the weapon |
| `f0613bb3` | the gate's imports, without the hack that silenced them |
| `516d4e4b` | a grab is session state too |

**Nine** commits, not the seven this table names: the ledger commit cannot name
itself, and neither can the one that closes the counts after it — SK1b's own
convention, and the trap SK1a fell into first time.
