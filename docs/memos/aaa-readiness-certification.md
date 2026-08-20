# AAA-readiness certification — can a studio build the island on this engine today?

**Audit date** 2026-08-19 · **Measured against** `c023b73`; the audit's three commits were
later rebased onto `2181417` (a CI-red fix on two test arms) without conflict — no file
overlap · **Scope** the whole engine,
against one question: *can a AAA team — and specifically the 50 km² Vancouver island
starter map — develop on this engine today, and what would stop them?*

---

## The verdict

**Conditionally yes for a AAA team; not yet for the island as specified.**

The engine is, by an unusually wide margin, a real one. Thirty-five crates, 293 test blocks
and 5 516 green arms, a determinism doctrine that has caught its own violations twelve times, a
PIE-equals-shipping gate on every phase since P21, budget ratchets, a cook pipeline, an
embedded DCC that can be edited during Simulate, virtualized geometry and textures, and a
level of documentary honesty in the ledgers that most commercial engines do not attempt.
A studio dropped into it today could model, texture, script, animate, simulate, cook and
ship a game of conventional scope. Nothing found in this audit threatens that.

The island is a different question, and the answer is no — **not because the engine is
too slow or too small, but because the specific chain the island depends on is not
connected.** Three findings carry the verdict, and all three are measured, not inferred:

1. **PCG cannot see a streamed terrain.** Over an asset-backed (streamed) terrain — which
   a 50 km² world must use — every scattered instance lands at exactly `y = 0`. Measured:
   **220 instances following an authored hill vs 929 instances, 929 of 929 at sea level**,
   from the same volume and the same graph. The instance *count* differs too, because
   slope and height masks evaluate against a plane, so this is not a world that needs
   nudging downward — it is a different world. Now pinned by
   `runtime/inf-player/tests/pcg_over_streamed_terrain.rs`.

2. **The GIS half is library code with no door.** `inf-gis` is nine well-built,
   well-tested modules with **one dependent crate in the workspace, using one of them**.
   `spawn_layer`, `read_vector`, `RoadGraph::from_layer`, `build_all_ribbons`,
   `triangulate_polygon` and `classify_to_ids` have **zero callers outside their own
   tests**. There is no IPC command, no wizard, no CLI. "Vector-data-driven" is not a
   setting to switch on; it is a sub-phase to build.

3. **Grammar buildings do not fit in a frame at city scale.** Measured on the committed
   P19 town: **12 850 static colliders cost 4.663 ms/step — 28% of a 60 Hz frame — for
   seven buildings.** That is 0.363 µs per collider per step, putting the 60 fps ceiling
   near 46 000 colliders, i.e. **about 25 archetype buildings**. Two cities of ≥1 km² with
   districts need hundreds to thousands. And streaming does not relieve it, because
   finding 1 means PCG content is evaluated once at load and never by the cell streamer.

None of the three is a defect of craftsmanship; all three are *seams that were never
required to meet*. The engine's own ledgers predicted two of them by name (P29 routed the
floating-origin camera and the vehicle fleet "to the island"; Wave G called the missing
GIS wizard "the wave's largest honest limit"). What this audit adds is the numbers, and
one seam nobody had named: the PCG/streaming collapse.

**Recommendation.** The island is a *phase*, not a *map*. Budget it as one — with a schema
window, because two already-deferred items require one — and sequence the connective work
before any content is authored. The itemized list is below with numbers attached. Nothing
found re-opens the wave plan; nothing found suggests the architecture is wrong.

---

## How this was measured

Everything labelled MEASURED below was run on this machine at `83ba8e7` and the command is
given. Everything labelled RELAYED is quoted from a wave ledger or memo and is sourced;
where a ledger's claim was checked and found wrong, that is said so. Nothing in the
island-blocker list rests on inference alone — the P25 law ("unmeasured prescriptions can
be backwards") is the reason.

Two ledger claims were checked and **corrected**:

* The Wave-D audit re-carried the libm gap as *"its three `.sin()`/`.cos()` sites"* in
  `dcc.rs`. There are **five** — the two it missed are `Quat::from_axis_angle` and
  `from_rotation_arc`, which reach `sin_cos` inside glam where no grep of the crate sees
  them. A gate written to that sentence would have shipped with both uncovered.
* The Terrain & GIS wave's ROADMAP block says its named remainders are *"all six … in the
  memo"*. The memo carries **eleven**; five of the missing ones (land cover → biomes, GIS
  attributes → building floors, the road ribbon not being a `MeshAsset`, OSM protobuf,
  LERC/GeoPackage/WMS) are the most island-relevant items in the whole register. Anyone
  planning from the ROADMAP alone would miss them.

---

## Island blockers, with numbers

Ordered by what stops the island soonest. **IB** = island blocker.

### IB-1 · PCG scatters at sea level over any streamed terrain — MEASURED

`evaluate_pcg_volumes` picks its height source with `if t.data.is_empty() { return None }`;
an asset-backed terrain ships no inline tiles (the streamed-terrain sample
`debug_assert!`s this about itself); the `None` arm is `FnHeight::new(|_, _| Some(0.0))`.
And `attach_terrain_streaming` runs *after* `RuntimeSim::new`, so there is no arrangement
of a level in which a streamed terrain has pages resident when PCG asks.

| terrain | instances | y range |
|---|---|---|
| authored (inline tiles) | 220 | 50.068 … 59.813 — follows the hill |
| asset-backed (streamed) | **929** | **0.000 … 0.000 — all 929 at sea level** |

The editor mirror (`commands::pcg::pcg_evaluate`) carries the identical fallback, so the
two hosts *agree* and PIE == shipping still holds. They agree on the wrong answer, which
is exactly why no existing gate caught it.

Second half of the same finding, VERIFIED: `evaluate_pcg_volumes` has exactly **one**
production caller in the tree — `level.rs:578`, inside the world builder — and
`cell_stream.rs` calls `spawn_entities` in six places and **never** calls it. So a
`PcgVolume` inside a streamed partition cell is spawned and **never evaluated at all**.
The engine documents this about itself (`samples.rs`: "**PCG evaluation is a load-time
pass.** `evaluate_pcg_volumes` runs once"), which is correct for a hand-authored level and
is exactly the assumption a 50 km² streamed world breaks.

**Fix shape:** defer PCG evaluation until after streaming attaches, or give PCG a
`TerrainSource`-backed `HeightProvider` that can page. Either is a behaviour change to the
load path with its own arms — a sub-phase, not a patch.
**Gate:** `cargo test -p inf-player --test pcg_over_streamed_terrain` (asserts the defect;
delete it when fixed).

### IB-2 · Grammar buildings saturate the frame at ~25 buildings — MEASURED

```
cargo test -p inf-player --test phase19_gate -- --nocapture
cargo test -p inf-physics --test bridge_sync_scaling -- --nocapture
```

* P19 town: **15 097 instances + 12 850 solids built in 14.24 ms**; steady
  **4.663 ms/step** with 12 850 colliders (60 Hz frame = 16.7 ms; tripwire 33 ms).
* Per collider: **0.363 µs/step**. 60 fps ceiling ≈ **46 000 colliders**.
* The town is **seven** buildings → ~1 836 colliders each → ceiling ≈ **25 buildings**.
* Physics bridge reconcile: **58.9–61.7 / 78.5–82.0 / 89.7–96.0 ns per entity** at 1k / 5k /
  13k entities against an asserted ceiling of 200 ns — two runs, before and after the
  rebase onto `2181417`, whose fix makes this arm calibration-normalize. Quoted as a range
  because the spread between runs (±7% at 13k) is larger than the difference the fix made,
  and a single figure would imply a precision this bench does not have. At 1.8 M colliders
  the reconcile alone is ~160–175 ms/sync.

Compounding it: **one `building.plan` node is one building.** There is no block- or
lot-subdivision node; `building/partition.rs` splits *rooms inside a floor plate*, not a
city block into lots. Two cities implies thousands of graph nodes or thousands of
`PcgVolume` entities, authored by hand.

**Fix shape:** LOD/streaming for grammar colliders (they are static — most need not be
resident), a lot-subdivision node, and a collider budget. All three are features.

> **CLOSED by island wave I3 (2026-08-20)**, all three, and measured on a committed
> thousand-building city (`samples/phase30-city`, 370 468 solids):
>
> * **IB-2a** — `inf_ecs::SimBand` bands static structural colliders off the simulation's own
>   `StreamingSource` entities (never a camera). **6 067 colliders, 1.64 %, 2.202 ms/step** at
>   the 0.363 µs above, against **134.480 ms** unbanded and a 4.0 ms
>   `STREAMED_STEP_BUDGET_MS`. The 64 m radius is the widest on a printed six-row sweep that
>   stays inside the budget.
> * **IB-2b** — `StructureGroup` + `ScatterBatch::near_distance`: a far building is one shell
>   box, drawn and collided. 14 whole / 788 shells / 198 out on the fixture. The draw bands
>   **overlap** by the batch's widest shell half-diagonal (≤ 18.486 m) rather than meeting at
>   192 m: the bands are complementary per *group* and the cull is per *instance*, and the
>   I3 audit measured 196 buildings drawn in pieces with no shell before that reach was
>   carried.
> * **IB-2c** — `building.lots`, so one `building.plan` node is as many buildings as its block
>   has lots. 4 500 shipped lot pairs at **0.000e0 m²** of overlap.
>
> `runtime/inf-player/tests/city_scale.rs` — nine arms through the cooked pack, including
> **PIE == shipping at every one of 480 drive-through steps**, with a `Camera` planted in each
> host a city apart so that the "never a camera" clause has something to fail against.
> `docs/ROADMAP.md`'s I3 block and its audit block.

### IB-3 · The GIS vector path has no caller — RELAYED, verified by grep

`inf-gis` is 5 732 lines across nine modules. **One** workspace crate depends on it, using
**one** module. Zero non-test callers for `spawn_layer`, `read_vector`, `read_shapefile`,
`read_geojson`, `RoadGraph::from_layer`, `build_ribbon`, `build_all_ribbons`,
`triangulate_polygon`, `classify_to_ids`, and the terrarium codec. No `#[tauri::command]`,
no `ipc.ts` wrapper, no panel, no CLI subcommand.

The Wave G memo says this about itself, and it is the plainest sentence in the register:

> every claim in this memo that reads as "the engine now does X" should be read as "the
> engine now *can* do X, from Rust, when something calls it".

**Fix shape:** a GIS import wizard + Ring-2 command family. This is the single largest
piece of connective work the island needs.

### IB-4 · Roads never become geometry — RELAYED

`build_ribbon` returns **vertex, UV and index arrays, not a `MeshAsset`**, and takes the
ground as a caller-supplied closure that **nothing in the tree supplies**. A GIS road
layer today spawns bare `Spline` entities with linear interpolation (`gis.rs::spawn_spline`);
polygons spawn their boundary as a closed spline, and **polygon interiors are explicitly
deferred**. There is no road surface, no road collider, no road-to-terrain blend.

### IB-5 · Vector attributes cannot drive PCG or buildings — RELAYED

Two named Wave G deferrals, neither of which reached the ROADMAP's remainder list:

* **G10 — land cover → biomes.** "there is still **no path from a raster to a
  `BiomeSet`** — nothing decodes a land-cover image, nothing writes biome ids, and the
  classifier has no caller outside its own tests."
* **G11 — GIS attributes → building floors.** "there is **no code between a GIS attribute
  and that field** [`BuildingParams::floors`] **in either direction**. 'Maps onto'
  described a shape, not a wire."

Together with IB-3 these are what "vector-data-driven" means; none of it is wired.

### IB-6 · Building footprints collapse to axis-aligned boxes — RELAYED

`building/pass.rs::lot_of` takes the **XZ bounding box** of a span, so a real footprint
polygon becomes an axis-aligned `Rect2`. Wave G defers oriented lots with: "**The building
floor-plate slicer assumes axis alignment throughout.** That is a deep change and deserves
its own sub-phase." A city whose streets are not on a compass grid — Vancouver's West End
and downtown are both rotated — will not sit on its own parcels.

### IB-7 · `inf new` → `inf cook` is a dead end — MEASURED

The first thing anyone does with this engine fails.

| `inf new --template` | levels scaffolded | `inf cook` |
|---|---|---|
| `blank-3d` | 0 | blocked — "no levels in cook — the build has no boot scene" |
| `2d-platformer` | 0 | blocked, same |
| `first-person` | 1 (`Levels/Main.inf_lvl`) | **blocked, same** |
| `hybrid-2.5d` | 1 (`Levels/Main.inf_lvl`) | **blocked, same** |

**Mechanism, proven:** templates scaffold the starter level into `Levels/`
(`manifest.levels_dir`, default `"Levels"`), and the cook only ever opens
`<project>/Content`. `levels_root()` has two callers in the whole tree — a `create_dir_all`
at project creation, and a test. Copying `Levels/Main.inf_lvl` into `Content/` makes the
same project cook (386-byte pack, root level resolved) and the shipped player run
**300 frames, exit 0**.

**Why CI is green:** the cook-and-run smoke does **not** use `inf new`. It hand-writes an
`inf.toml` and copies a committed sample level into `Content/`. The real first-run path is
uncovered.

Compounding it: the editor's `scene_save` with no explicit path falls back to
`<app_data>/quicksave.inf_lvl` — **outside the project entirely**. There is no guided route
from "new project" to "shippable build".

**Fix shape:** a product decision — either the cook also scans `levels_root()`, or the
templates scaffold into `content_dir` and `Levels/` is retired. Left unfixed deliberately:
which directory is authoritative for levels is a layout decision that bakes into every
project a studio ever makes, and an audit should not make it unannounced.

### IB-8 · 2 047 virtualized-geometry instances per frame — VERIFIED constant

`VIS_INSTANCE_BITS = 11` ⇒ `VIS_MAX_INSTANCES = 2047`, and `VisPacking::admit` **refuses**
beyond it (`VisPackError::Instances`). A city of thousands of Nanite-class building meshes
cannot all enter the visbuffer path in one frame. Raising it means re-cutting a packed
32-bit GPU id (`VIS_TRI_BITS + VIS_MESHLET_BITS + VIS_INSTANCE_BITS == 32`, asserted at
compile time) — a format change, and the meshlet slot field would have to give up bits.

Related caps, verified: `MAX_CPU_SCATTER_INSTANCES = 65 536` (CPU fallback drops beyond),
GI `instance_budget = 4096`.

> **CLOSED by island wave I3 (2026-08-20).** The id is **sixty-four bits** —
> triangle 7 / meshlet 25 in word 0, instance 24 + 8 reserved in word 1 — and
> `VIS_MAX_INSTANCES` is **16 777 215** (indices `0..=16 777 214`), 8 196× the
> number above. The re-cut this
> entry proposes was measured and refused: the meshlet field addresses pool
> *capacity* and was already the binding ceiling at 7.4 % of the default streaming
> budget, so buying instance bits from it would have made an already-firing
> refusal fire at 0.9 %. Cost: four more bytes a pixel — eight in total, 7.4 MB at
> 720p — paid only while the mode is on, which is off on every tier. All twelve
> P28.1 parity arms, the seven feedback arms, the seven `phase28_gate` arms and
> the 54 goldens under `INF_GOLDEN_STRICT=1` pass unchanged.
> `docs/memos/p28-1-visbuffer.md` §1.1.
>
> *The I3 audit adds one thing to that list:* the three shaders' bit-split pin
> compared `const` **declarations**, so `vis_feedback.wgsl` masking the meshlet
> field with the old literal `14u` passed all twelve parity and all seven feedback
> arms. It is invisible for the same reason it was invisible to the arms — no
> fixture in the tree resides past the old 16 384-slot field — and the use sites
> are now pinned as counts.

### IB-9 · The terrain ratchet and the terrain budget are 16× apart — VERIFIED

`TERRAIN_RESIDENT_BYTES_CEILING = 16 MiB` (asserted by `phase16_gate`) against
`StreamBudget::default().max_resident_tiles = 1024`. At the default 257² tile that is
257² × 4 B ≈ 258 KiB per tile, so 1 024 tiles ≈ **264 MiB — 16.4× the ratchet**. The gate
scene never approaches either, so the two have never had to agree. At island scale the
ratchet fires first, and whoever is holding it will not know which number is the real one.

> **CLOSED by island wave I4 (2026-08-20).** They were two guesses at one quantity: how many
> pages a render cut holds. `inf_terrain::stream::cut_page_bound` is that quantity, and
> `StreamBudget::for_ladder` and `resident_bytes_bound` are two readings of it.
>
> The cut is **O(levels), not O(pages)** — measured across four world sizes whose catalog
> quadruples each step (84 → 340 → 1 364 → 5 460 pages), the peak cut goes **64 → 157 → 250
> → 340**: a constant ring of **93, 93, 90** per LOD level, and identical at two page
> resolutions. That invariance is what makes a budget derivable; the finding's own premise —
> that the working set grows with the map — is false in the direction that matters.
>
> **They meet now**: `phase16_gate` arm (e) computes what the budget would let its own scene
> hold and asserts it inside the ratchet — **200 pages → 12.70 MiB** at 129² against a
> **16 MiB** ceiling and a **5.90 MiB** measured peak. The same arithmetic before this wave
> gave **65 MiB against 16 MiB**. `crates/inf-terrain/tests/island_working_set.rs`.

### IB-10 · The island needs a schema window on day one — RELAYED

P29's disposition table defers two items **to "the island's schema window"** and says they
must ride one bump: `set_blend_mode` per-transition (deferred five waves) and the persisted
ragdoll's missing `contacts`/`layers`. P29.7 adds a third: a vehicle *class* with its own
tuning "is the island's", and needs a scene field. This audit ran under a no-schema
mandate, so nothing here moved — but the island phase's first act is a schema bump, and it
should carry all three at once.

### IB-11 · Source-data reality for a real Vancouver DEM — RELAYED

Refused **by name** at the GeoTIFF door, each with an external remedy: **BigTIFF**,
**LERC** ("which ArcGIS produces by default and most government portals run on"),
**JPEG2000**, rotated/sheared rasters, non-square pixels, single-strip TIFFs (which is what
the project's own test encoder writes by default — 4.2 GiB otherwise). Also: `.prj` is not
read so the CRS is a caller-stated parameter with no UI to state it; **no reprojection** (a
terrain in a different CRS is refused, not warped); **no geoid model** (orthometric vs
ellipsoidal sources differ by "tens of metres"); datum shifts are analytic with
"metre-class error"; **LAS/LAZ LiDAR deferred** — which is Vancouver's best elevation data.

And one that will bite silently: **Web Mercator is refused** because "its 'metres' are
inflated by about **1.53× at Vancouver's latitude**, which would build the island half
again too large with no symptom other than everything being wrong." The island must be
anchored in UTM zone 10N.

Every one of these is survivable with a `gdal_translate`/`gdalwarp` pre-pass on the
author's machine. They are listed because "vector- and raster-data-driven" implies an
ingest pipeline, and today that pipeline's first stage is a human with GDAL installed.

### IB-12 · The locomotion camera is not floating-origin-safe — RELAYED

P29 disposition row 23: `axis_independent_lag` "unrotates absolute world positions rather
than the delta — algebraically origin-independent, not so in floating point **at partition
scale**. It wants a floating-origin-aware camera, which is a streaming-scale question and
**belongs with the island's 50 km²**." Routed to the island by name, by the wave that wrote
the camera.

> **CLOSED by island wave I4 (2026-08-20).** The delta goes into the yaw frame and the anchor
> never does — the Wave-T terrain-UV precedent applied to a camera. Error against the same
> relative run at the origin, 600 steps at 37° yaw: **6.551e-14 / 8.207e-11 / 7.082e-10 m** at
> 1 km / 50 km / 500 km, against the old form's **2.618e-6 / 1.309e-4 / 1.309e-3** —
> **1.848e6× better**. A rebase mid-lag moves the render-local camera by the world step and
> the origin's own snap and nothing else; a partition handoff at speed (7 crossings of a 256 m
> cell at 20 m/s, each with a one-step subject gap) costs **1.02×** a steady step rather than
> a cut. The pre-IB-12 spelling is kept and pinned at **zero** production call sites.
> `crates/inf-ecs/tests/camera_at_scale.rs`.

### IB-13 · The editor's own scene projection at city scale — MEASURED

`SceneDoc::snapshot`, which every `world://delta` pays:

| entities | snapshot |
|---|---|
| 4 096 | 1.02 ms |
| 10 000 | 3.23 ms |
| 50 000 | 16.89 ms |

Linear at ≈ **0.34 µs/entity**. A 100 000-entity city document costs ≈ **34 ms per
snapshot** — at the 33 ms frame tripwire, before anything renders. Two cities of roads and
buildings plausibly exceed that.

Beside it, the vector spawn path itself is *fine*: 10 000 road polylines spawn in 39.0 ms,
50 000 footprints in 513.5 ms. It is the document projection that does not scale, and it is
already a known hot spot (`SceneDoc::snapshot` was 3 277.5 ms at 15 000 entities before its
P-wave fix; it is 3.232 ms now).

> **CLOSED by island wave I3 (2026-08-20)** — and the table above **understates** it, because
> it counts only the snapshot half. Every `world://delta` also paid a full `diff`: the round
> trip at 100 000 entities, moving one entity, measures **52.857 ms**.
>
> `SceneDoc::project_delta` replaces both. A mutation declares what it moved (`touch_at`), and
> `touch` still means "everything" for the 41 of 45 call sites this wave did not narrow — a
> conservative union, so an unconverted site is slow and never wrong. Measured after:
> **8.105 ms** for a drag frame and **0.0006 ms** for a select frame, at the same 100 000.
>
> The residue is named rather than absorbed: the 8.105 ms is `EcsWorld::propagate`, which the
> select-only column isolates at 0.0006 ms. Incremental transform propagation is its own item.
> `docs/ROADMAP.md`'s I3 block.

### IB-14 · The default vector import cap silently amputates a city — MEASURED

`SpawnOptions::max_entities` defaults to **4 096**, documented as "a guard, not a
preference: a county road layer is ~10⁵ features".

| layer | features | spawned | truncated |
|---|---|---|---|
| roads | 10 000 | 4 096 | **5 904** |
| footprints | 50 000 | 4 096 | **45 904** |

It is *reported* (`SpawnReport::truncated`) and never silent, which is the right design —
but with no wizard (IB-3) there is nowhere for an author to raise it and nowhere for the
report to be shown.

### IB-15 · Multi-terrain and the distant island — RELAYED, needs verification

Two items the island is the first content to need, neither confirmed closed:

* **The multi-anchor pyramid seam** (P16 remainder, ROADMAP:1235): "every terrain is
  anchored at zero, so two terrains cannot yet share one pyramid across their boundary."
  Wave G shipped a non-zero `.inf_terrain` origin, so the premise may be partly retired —
  **no source read in this audit claims the seam itself is closed.** Verify before
  planning a multi-tile island.
* **No global silhouette LOD** (Wave G G20/G22/G23): "a distinct always-resident global
  silhouette does not [exist]", and the ~10 km cross-fade "is not [built], because there is
  no silhouette." An island seen from across the water has no LOD answer.

### IB-16 · No per-frame streaming budget in the VT loop — RELAYED

Wave T's T33b, refused on purpose: "There is **no per-frame time budget and no per-frame
admission or upload throttle in the VT loop; the only budget is a byte residency
ceiling.** … Adding one is not a local change — it re-opens a measurement, and the named
tripwire tests are built to go red when it lands. Do it deliberately or not at all."
Coupled to T51 (no request→residency window, so no async upload path). This is the most
60 fps-relevant carried item in the texture stack.

> **CLOSED by island wave I4 (2026-08-20), deliberately.** `inf_stream::AdmitBudget` goes into
> the one admission walk both page systems run, in **bytes** rather than pages (an RGBA8
> transcode page is 8× BC1's). Default **1 MiB/frame**. A burst is **smoothed, never dropped**:
> 40 tiles at 4 pages/frame drain in exactly 10 frames, worst upload 36 992 B against a
> 36 992 B budget, every tile arrives. **The floor is never throttled** — measured: throttling
> it retracted a P28.2 cluster page and left the P28.3 load at 3 of 5 pages resident.
>
> The named tripwires were re-aimed and **none of them went red**.
> `p28-5-lead-time-ruling.md` §3.5 predicted this would reverse
> `DEFAULT_PREDICT_HORIZON_TICKS = 0`; at the shipped budget h=0 still wins (19 542 blur
> against 19 766), because a throttle takes the *tail* of a lane and does not delay the *head*
> of one. Under a budget nothing ships — two pages a frame — **a lead does win** (75 928
> against 76 074), reported and never asserted. T51's half of the condition, a loader with
> real latency, is still not built.
> `crates/inf-vt/tests/upload_budget.rs`, `crates/inf-render/tests/whip_pan.rs`.

---

## What is NOT a blocker — measured, and better than expected

**Terrain import survives island scale comfortably.** A synthetic ridged DEM through the
real chunked importer (`terrain_import::build`), 257² tiles, 1 m/sample, peak working set
sampled externally:

| source | world | import | peak RSS | `.inf_terrain` | tiles / LODs |
|---|---|---|---|---|---|
| 8 192² (67 M samples) | 8.19 × 8.19 km ≈ **67 km²** | **746 ms** | **839 MiB** | 343.8 MiB | 1 364 / 5 |
| 16 384² (268 M samples) | 16.4 × 16.4 km ≈ 268 km² | **3 163 ms** | **3 215 MiB** | 1 376.1 MiB | 5 460 / 6 |

Scaling is clean: 4× the samples costs 4.24× the time, 3.83× the memory, 4.0× the payload.
Peak memory runs at **≈ 2.34× the finished payload**, which is the whole-image writer
(`write_terrain_asset` assembles the payload in RAM before its atomic rename — the
spill-to-temp writer is a named Wave G follow-up). Extrapolating that ratio, the ceiling on
a 16 GB workstation is around **32 768² (~1 000 km², ≈ 12.9 GiB peak)**.

**The island's own size class costs 746 ms and 839 MiB.** Import is not the problem.

Also healthy: the **soak harness** (10 000 edit/undo/save cycles, 10 save-reload
round-trips, peak 300 entities, peak ~996 KiB, final undo depth 0 — bounded, no leak); the
**crash harness** (writes a plain-text report with engine version, OS, message, location
and the log tail); **determinism** (`parallel_map` is an in-order pure map, so PCG output
is byte-identical for any pool size, pinned by `portable_placement`); and the **atomic-save
doctrine**, which held at every door examined.

---

## The open-items register, triaged

Sourced to the ledger that carries each item. **FN** = FIX-NOW (closed by this audit),
**IB** = island blocker (above), **SP** = studio papercut, **CO** = cosmetic.

### Closed by this audit (FIX-NOW)

| # | Item | Source | Disposition |
|---|---|---|---|
| FN-1 | "A `libm`/trig source gate does not cover `editor/.../dcc.rs`" | Wave D audit re-carry | **CLOSED** — `tests/portable_math_law.rs`, walking `src/` recursively. Found **five** sites, not the three the re-carry named. |
| FN-2 | **Committed sample levels depended on the platform libm** | *found by this audit* | **CLOSED** — `terrain_demo_height`/`character_demo_height` byte-lock `TerrainDemo.inf_lvl`/`Character.inf_lvl`; with `std` trig that lock asserted a property of *the machine running the test*. Now portable; two samples re-blessed. |
| FN-3 | `SpawnKind::Terrain`'s starter hill used `std` trig on a path that writes committed content | *found by this audit* | **CLOSED** — `psin64`/`pcos64`. |
| FN-4 | "The lossy detach has no severity and no threshold" | Wave D audit re-carry | **CLOSED** — `inf_dcc::DetachSeverity` bands the share (None / ≤1% / 1–10% / >10%); panel verdict reads the band, not `=== 0`. |

### Deferred by this audit, with the reason

| # | Item | Source | Class | Reason not fixed here |
|---|---|---|---|---|
| D-1 | `preventDefault()` runs before the "did anything run" check; nine bare-letter chords (`1 2 3 G R S A L F`) swallowed app-wide for non-editable targets, and `closest("input, textarea, [contenteditable]")` does not match `<select>` (ModelEditor has six) | Wave D audit re-carry | **SP** | Reordering the guard changes dispatch for every chord in the shell. Editor-UX scope, and a real risk of silent regressions across the whole app. |
| D-2 | A non-orientable surface (Möbius band) is torn and the message blames the author's file | Wave D audit re-carry | **CO** | The kernel cannot represent non-orientability; the honest message is about *our* limit. A wording fix, but it wants the kernel decision written down first. |
| D-3 | `welded_positions` is snapshotted before the repair, so it understates a repaired import | Wave D audit re-carry | **CO** | Documented as such in the field's own doc comment. Correcting it changes a reported number authors may already read. |
| D-4 | `ImportError::NonManifoldEdge`'s convergence guard is unreachable from `from_mesh_asset`, making a property arm dead code | Wave D audit re-carry | **CO** | Kept deliberately: "unreachable" is a claim about code that will change. |
| D-5 | Sculpt `[`/`]` listener still outside the keybinding registry | P23 → Wave E remainder | **SP** | Named in three ledgers; a raw `window` handler on the terrain sculpt radius. |
| D-6 | Blueprint: one document at a time; no graph is written back into a `.inf_act`; the "code of actor X" tab has no on-disk Rust | Wave E remainders 2–4 | **SP** | Multi-document store + a transpiler-workflow decision about where generated Rust lives. |
| D-7 | No marquee select; no `dcc_new` (a primitive cannot be converted to an editable mesh); **the grammar bake has no Ring-2 door** | Wave E remainder 5 | **SP → IB** | The grammar-bake door is island-relevant: buildings cannot be baked from the editor. |
| D-8 | macOS has no viewport mouse input at all | Wave E remainder 6 | **SP** | Hardware pass. Every batch-C gesture is Windows-only. |
| D-9 | Keybinding conflict resolution is a toast, not a dialog | Wave E remainder 7 | **CO** | — |
| D-10 | Content Drawer menu guarded but not migrated onto `ContextMenuSurface` | Wave E remainder 1 | **CO** | Has an inline expanding submenu the surface does not model. |
| D-11 | Blend-space triangulation runs on every call — 0.94 / 1.74 / **6.84 µs** at 5 / 9 / 25 samples; ≈3.5 µs per entity per fixed step; **≈2% of a core per 100 characters at 60 Hz** | P29.2 | **IB (crowds)** | Wants a caller-held triangulation cache — an API change reaching both fixed steps. Scales linearly with crowd size, so it is the island's the moment the island has a population. |
| D-12 | f32 play-head reduction: markers stop separating at **t > 131 072 s ≈ 36 h** in one unbroken state | P29.5 audit A7/A9 | **SP** | Real for a persistent open world; the honest fixes are an f64 `resolve_time` every caller pays for, or a second reduction rule. |
| D-13 | Ragdoll self-collision off wholesale; two ragdolls pass through each other | P29.6 | **SP** | Needs per-ragdoll group ids. Its current bound is asserted, so closing it *fails* an arm rather than outliving it. |
| D-14 | A wheel is a visual and a radius, not a rolling body — no wheel collider in rapier; the suspension ray is the whole contact model. **The vehicle has never driven on a heightfield.** | P29.7 + closing audit | **IB (vehicles)** | Named by the closing audit as "the day this rig drives on a heightfield is the island's". |
| D-15 | `fov_deg` documented vertical, valued like UE's horizontal — the shipped window renders **1.46× the intended field** | P29.6 audit, corrected by P29.7 A2 | **SP** | A rendering decision with its own before/after; converted constants already computed (43.0 / 49.0 / 32.6 / 58.7). |
| D-16 | No live tuning of an engine-published parameter (`Tune::Param` drains before the movement step publishes) | P29.6 | **SP** | Needs an author-override layer over the parameter overlay. |
| D-17 | Sub-machines are one level deep; an interruption carry is one deep (measured: **40.5°** across a second interruption vs `Snap`'s 63°); triggers capped at **64** parameters | P29.1 | **SP** | Structural bounds, all three refusing rather than silently wrong. |
| D-18 | BC6H missing (**8× the memory on every float texture** — 771.6 MiB vs 96.5 MiB at 8192²); BC7 missing | Wave T 2.1/2.2 | **SP → IB (memory)** | Both are owned-encoder work. BC6H is called "the largest single win left". |
| D-19 | No anisotropic filtering; `anisotropy_clamp = 1`. Border ring gives ≈**8:1 at a tile edge and nothing beyond** | Wave T 2.6 | **IB (quality)** | Grazing-angle ground over a 7 km view is exactly this case. Fixing it needs a wider border (a `.inf_tex` bump) or clamping to the ring. |
| D-20 | Terrain layer materials are planar XZ — **they stretch on cliffs**; triplanar is 3× the fetches | Wave T §0 B | **IB (quality)** | Vancouver's terrain is the cliff/mountain case this fails on. |
| D-21 | Only **four** terrain layers; splat weights still not virtualized (pyramid is heights-only, coarse rings read the level-0 weight page) | Wave T / ROADMAP:1240 | **SP** | Memory-relevant at 50 km². |
| D-22 | T44: `VtTextureDesc::validate` has **no extent rule at all** — what keeps a pyramid inside an f32 uv is that no content this project can produce comes near 32K | Wave T | **IB (verify)** | A 1:1 island's VT extents are precisely the content that could come near it. Re-measure before authoring. |
| D-23 | 12 Wave-T **CANNOT**s (DirectStorage, GPU decompression, ray-query shadows, neural prefetch/NTC, per-tile LZ4, KTX2/Basis, io_uring, transfer queues, GPU compaction, 4 KiB alignment, TIFF sources) | Wave T §4 | **CO** | Each refused with a measured or structural reason. Only **TIFF sources** is "a decision away" — one Cargo feature. |
| D-24 | 3 Wave-G **CANNOT**s (`gdal`, TIN terrain, `rkyv`) | Wave G §2 | **CO** | Refused on licence/CI/architecture grounds, all reasoned. Consequence is IB-11. |
| D-25 | Rustdoc warnings at **438** against a pinned ceiling of 450 | Wave E audit counts | **CO** | 12 of headroom. Worth a sweep before it forces a re-pin. |
| D-26 | Running the battery on Windows **dirties eight `bindings/*.ts` files** with pure CRLF/LF churn (the ts-rs generator writes LF; `.gitattributes` normalizes to CRLF) | *found by this audit* | **CO** | Every Windows contributor finishes a green battery with a dirty tree and has to know the diff is empty. `git diff --numstat` reports zero changed lines, which is the tell. |

### Ledger-integrity findings

Recorded because a register is only as good as the ledgers it is built from:

* **P29's disposition table claims completeness it does not have.** It says "**Every** item
  named as carried, deferred or routed by any P29 ledger or audit" and has 26 rows. At
  least **fifteen** named carries are absent — including P29.7's own six honest remainders
  written immediately above it, the P29.7 audit's three carries, P29.3's slope-limit
  `mover_for` item and its `MovementRuntime`-not-in-`state_bytes` item, P29.3's
  re-ledgered cloth remainders, P29.5's mixed-rig proposal gap, and P29.1's five structural
  bounds. The table's own audit spot-checked **13 of 26** rows and says so.
* **The Terrain & GIS ROADMAP block undercounts its own memo** (six vs eleven) — see "How
  this was measured".
* **Wave G's summary tally says 5 PARTIAL; §3 marks ten rows PARTIAL.** The counts do not
  reconcile.
* Every P29 sub-phase STATUS block ends "local gates green; **NOT PUSHED**".
* The `chr(92)` law took **five more catches** during P29 (catches 4, 6, 8, 9, 10), twice
  by auditors on their own edits — and a twelfth in the Wave D audit. This audit's own
  edits were routed through the `Write` tool or raw-string Python for exactly that reason,
  and the workspace gate is still the thing that would catch it.

---

## The studio-workflow soak

The loop a studio hits daily: new project → import → author → Simulate → PIE → cook → run
shipped.

| step | result |
|---|---|
| `inf new` (4 templates) | **DEAD END — IB-7.** 4 of 4 produce a project `inf cook` refuses. |
| Level into `Content/` → `inf cook` | Green — 386-byte pack, root level resolved, 1 level rewritten for runtime |
| `inf-player --pack --headless --run-frames 300 --assert-exit` | Green — 300 frames, `final-state-hash` printed, exit 0 |
| GeoTIFF / DEM import (8 192², 16 384²) | Green — see "What is NOT a blocker" |
| Vector import (10k roads, 50k footprints) | **No door — IB-3.** Library path measured green; capped at 4 096 by default (IB-14) |
| Terrain sculpt / paint / PCG authoring | Green in-editor; **PCG over streamed terrain is IB-1** |
| DCC model edit, incl. **edit during Simulate** | Green — proven end-to-end since P23.6 |
| Blueprint author → interpret → transpile | Green; **no write-back to `.inf_act`** (D-6) |
| Autosave / crash recovery / atomic save | Green — atomic temp+rename held at every door examined; recover-on-boot loads a surviving recovery file |
| Soak: 10 000 edit/undo/save cycles | Green — peak 300 entities, ~996 KiB, undo depth returns to 0 |
| Crash report | Green — plain text with version/OS/message/location/log tail. **No minidump** (SP: AAA crash triage expects symbolicated dumps) |

---

## The numbers

| | before (`c023b73`) | after (this audit's three commits) |
|---|---|---|
| Rust battery | 291 blocks / 5 507 passed / 0 failed / 13 ignored | **293 / 5 516 / 0 / 13** |
| Frontend | 690 tests / 77 files | **690 / 77** (unchanged) |
| Goldens | 54 | **54, byte-untouched** |
| `cargo clippy --workspace --all-targets -- -D warnings` | 0 | **0 warnings, 0 errors** |
| `cargo fmt --all --check` | clean | **clean** |
| Schema | — | **unmoved** |

**+2 blocks, +9 arms**, and the arithmetic is exactly the four additions: `portable_math_law`
(5), `pcg_over_streamed_terrain` (1), the two detach-band arms in `inf-dcc`, and the wire-enum
pin in `inf-studio`. `cargo test --workspace` exit status **0**, taken directly rather than
through a pipe — a `cargo test | tail` pipeline reports *`tail`'s* status and would call a red
battery green, which is how the first run of this audit's battery nearly went unchecked.

**Budget ratchets (P15) — they run, and they are `const`s, not data.** There is no ratchet
data file, no loader, no bless path: every arm *prints* its measured value and the ceiling
is a committed Rust `const` edited downward by hand (`runtime/inf-player/src/budget.rs`).
The four §8 budget gates all run in CI:

```
cargo test -p inf-runtime     --test sim_budget
cargo test -p inf-render      --test frame_budget      # GPU; skips without an adapter
cargo test -p inf-editor-core --test startup_budget
cargo test -p inf-player      --test startup_budget
```

Two honest bounds on that machinery, both material for the island:

* **The only GPU frame harness renders 640 × 360.** `frame_budget.rs` measures 484 lit
  cubes at that size. **No test in this repo measures fps at a shipping resolution**, so
  "≥60 fps" for the island has no existing instrument.
  > **CLOSED by island wave I4 (2026-08-20)** — `runtime/inf-player/tests/fps_instrument.rs`,
  > 1920 × 1080 and 2560 × 1440 over the phase-30 city + a streamed terrain + the phase-29
  > wizard character, with per-pass GPU timings from `inf_render::timing` (the repo had **no**
  > GPU timing at all before this wave). On an RTX 4070 Ti in release: **p50 39.792 ms
  > (25.1 fps) at 1080p** and **47.424 ms (21.1 fps) at 1440p**, p95 45.057 / 51.136. The
  > frame is **CPU-bound** — the sim fixed step alone is 13.659 ms against a 15.875 ms GPU
  > frame. `SHIPPING_FRAME_BUDGET_MS = 16.6` is what "≥ 60 fps" now MEANS, it is a target and
  > not an assertion, and the instrument prints the distance from it every run.
* **Every wall-clock assertion is disabled on software/paravirtual adapters** (they print
  and return), and `create_instance()` hard-codes `VULKAN | METAL` with no DX12/GL and no
  `WGPU_BACKEND` override honoured. On a Windows box without a Vulkan ICD every GPU test
  silently skips.

**Tracy** is wired (`--features tracy`, spans on `render_frame`, `sim_step`, five sim
phases, `cook`, `derive_vmesh`, `build_vgeom`, `import_textures`) and **never built in
CI**. **`cargo bench`** has two targets (`job_pool`, `schedule`), neither in CI, no
committed baselines.

**Disk.** The build tree remains the operational hazard the house law says it is:
`target/debug/incremental` alone was **19 GB** at the start of this audit on a volume with
23 GB free. Deleting just that directory freed it without a cold rebuild, per the Wave-G
law.

---

## Honest — human-verified, not arm-covered

Stated plainly because the island will rely on some of it:

* **Anything needing a window or a GPU on Windows**: the right-click menu appearing at the
  cursor, double-click opening the Model Editor, the flycam's feel, the editor's play
  capture grab (`Capture::SimLook` is `wnd_proc` code), DPI matrix behaviour.
* **macOS viewport input does not exist** (D-8), so every viewport gesture is Windows-only.
* **The Terrain Import wizard has never had a UX pass with a real multi-gigabyte source**,
  and the literal 16 k × 16 k import test is `#[ignore]`d — this audit is, as far as the
  ledgers show, the first time the at-scale import path has been *measured* rather than
  asserted structurally.
* **`inf-gis`'s nine modules have never run against a real government data file** — every
  arm is synthetic.
* **The `.inf_gis` → island chain has never been run end to end**, because IB-3 means there
  is no end to run it to.
* ~~**No fps measurement at shipping resolution exists**~~ — closed by island wave I4; the
  numbers are above, and they are not 60 fps.
* LSP, terminal and git runtime remain human-verified (CI spawns none of them).

---

## What the island phase should do first

Not a plan — an ordering the numbers imply.

1. ~~**Decide the project layout** and close IB-7.~~ *(done — wave I1)*
2. ~~**Take the schema window** (IB-10) once, carrying the three deferred items.~~ *(I1)*
3. ~~**Connect PCG to streamed terrain** (IB-1).~~ *(I1)*
4. ~~**Build the GIS door** (IB-3), then roads-as-geometry (IB-4) and the two attribute
   wires (IB-5).~~ *(wave I2 — with IB-6, IB-14 and IB-11's near half)*
5. ~~**Give grammar buildings an LOD/streaming story and a collider budget** (IB-2), and a
   lot-subdivision node; then re-measure against the 0.363 µs/collider figure.~~
   *(wave I3 — re-measured on a thousand-building city: 2.202 ms/step banded against
   134.480 ms unbanded. I3 also pulled **IB-8** and **IB-13** forward out of their wave,
   because both are ceilings a city walks into on its first frame.)*
6. ~~**Build an fps instrument at shipping resolution** before claiming 60 fps, and settle
   the terrain ratchet-vs-budget disagreement (IB-9).~~ *(wave I4 — with **IB-12** and
   **IB-16**, both pulled forward. The instrument exists and says 25.1 fps at 1080p; the
   dearest single thing in the frame is the **sim fixed step at 13.659 ms**, which no §8
   budget covers and which is where the next 60 fps work is.)*
7. Verify IB-15 (pyramid seam, global silhouette) before committing to multi-terrain. ←
   **next**, with IB-11's far half (I5).

Only then author content.
