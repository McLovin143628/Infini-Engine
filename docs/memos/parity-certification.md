# Parity certification — does the showcase island stand beside the reference?

**Wave** CERT1 · **Date** 2026-09-03 · **Base** `f70d5fca` · **Measured on** an
NVIDIA GeForce RTX 4070 Ti (DiscreteGpu), tier High, Windows/Vulkan, `--release`
where a clock is quoted · **Reference** the curated GTA 6 frame index at
`<parent>/docs/reference_videos/frames/NOTES.md` — leaked footage, local only,
never committed and never cited by path from inside this repository.

Its precedent is `docs/memos/aaa-readiness-certification.md`: rows tagged
MEASURED or REFUSED with the instrument named, a FIX-NOW register closed
in-wave, a deferred register with numbers, and an honest section for what only
eyes can judge.

---

## The verdict

**The engine draws the island; it does not yet dress it, light it or fill it the
way the reference does — and after this wave every one of those three is a
number rather than an impression.**

Six things were true when this wave opened and are not true now. The showcase
**played unlit** — every one of the twenty-four committed levels shipped
`RenderSettingsRecord::default()`, which is shadows, GI, VSM, TAA, SSAO and
bloom all off. **The Play button rendered a different level from the one that
ships**, because every windowed PIE path substituted a default render block for
the level's own. **The application had no answer at all to "which project"**, so
the showcase was a file dialog away on a fresh profile. **The terrain cracked**,
by up to 3.8262 m along a seam the island's own tile pitch guarantees. **Every
tile edge carried a one-texel band of half-strength shading** across 7.2 km. And
**a grammar building had two distance tiers where the owner asked for three.**
All six are closed, each with an arm that goes red if it comes back.

What is *not* closed is the half the owner already routed to its own arc, and
this certification's job there was to price it rather than to ship a cheap
version. The numbers are worse than the prose was. There are **no street lamps,
no poles, no cables, no hydrants, no signs, no awnings, no billboards and no
traffic signals** anywhere in the engine — twenty-two nouns, twenty-two empty
greps. There is **no particle system**. There is no subsurface term, so a face
under a red stage wash is Lambert. And the lighting is the one that surprised
me: the island's entire resident world is lit by **one directional light and
zero real fixtures**, against **2 122 glowing window panes** — which is exactly
the substitution the owner rejected in their own words ("I don't like the
glowing window panes"). Eleven of fourteen building archetypes hang no light at
all.

Two of the five gap areas turned out not to be gaps, and that correction matters
more than any of the confirmations: **aerial perspective and analytic height fog
already ship and already apply to seven lit shaders**, and **a four-term lens
flare pass already ships and its ghosts, halo and streak already fire off any
bright pixel**. PAR3 and PAR4 are narrower than the brief assumed.

The frame: the shipped 51.38 km² island at 1080p, with the record its own level
now authors, is **p50 39.672 ms — 25.2 fps** on an RTX 4070 Ti, and **7.2 fps**
with its town's own thousand-agent society at the rush hour. That is the honest
distance from the reference, and it is a CPU distance, not a GPU one.

---

## How to reproduce every number here

| number | command |
|---|---|
| the island censuses | `cargo test -p inf-player --test parity_cert -- --nocapture` |
| …at shipped scale | `INF_CERT_ISLAND_PACK=<...>/island-build/project/Build cargo test --release -p inf-player --test parity_cert -- --ignored --nocapture` |
| the controls table | `cargo test -p inf-player --test controls_cert -- --nocapture` |
| what a level draws | `cargo test -p inf-player --test lit_stack -- --nocapture` |
| the terrain | `cargo test -p inf-render --test terrain_continuity -- --nocapture` |
| the frame | `cargo test --release -p inf-player --test fps_instrument -- --include-ignored --nocapture` |
| the boot rule | `cargo test -p inf-project boot` |

Everything below is printed by one of those. Nothing here is arithmetic done by
hand except where it says so.

---

## The rows

### CP-C1 · The application opens on the island — MEASURED, FIXED

Before this wave `ProjectState::current` started `None` on every cold launch,
nothing persisted what had been open, and the only project knowledge that
survived a restart was a recent list the start screen renders as *buttons a human
presses*.

`inf_project::boot::resolve` is the rule, and it is a pure function of three
arguments plus `Path::is_file`, so the whole ordering is unit-tested against a
temp directory rather than against this machine:

1. `INF_BOOT_PROJECT` — the `INF_PLAYER_BIN` precedent, one variable, one path;
2. `EditorSettings::boot_project` — written by every successful open, so its
   plain meaning is *the last project you opened*;
3. **the showcase** — `island-build/project`, DISCOVERED by walking up at most
   `SHOWCASE_SEARCH_DEPTH = 8` ancestors of the running executable;
4. nothing: the start screen, exactly as before.

**The showcase is discovered and never hard-coded.** A dev executable sits at
`<checkout>/target/debug`, so the showcase's holder is four ancestors up. **On a
machine where `inf island build` has never run, no rung resolves and the editor
opens the start screen exactly as it always has** — and `inf island build` now
prints, beside the project it wrote, that the editor will open it and what the
override is.

The bound is asserted from **both** sides: a start eight levels below the holder
reaches, nine does not. The first version of that arm was off by one and said so
by failing.

Arms: 8 in `inf-project`, 2 in `inf-studio` (the pin lands; an unwritable
settings directory is not an error), 3 in the frontend (asked on a cold launch,
NOT asked with a project open, a null answer says nothing).

### CP-C2 · Play, from a cold start, on the island — MEASURED, FIXED

| measurement | fixture (2.36 km²) | shipped island (51.38 km², release) |
|---|---|---|
| open the pack + attach both streamers | 28.0 ms | **285.7 ms** |
| the first fixed step | 221.9 ms | **256.2 ms** |
| **total to a stepped world** | **249.9 ms** | **541.8 ms** |
| against `LOAD_BUDGET_MS` | 5 000 ms | 5 000 ms |

That is the headless twin of time-to-first-frame: it excludes the window and the
GPU, which is the half a test on this machine can hold to a number. The other
half is `fps_instrument.rs`'s and it is a frame rather than a boot.

**The pawn comes to rest on the island.** GTA1's closing audit found every
starter level giving its pawn a plane with nothing physical on it — 4.9868 m of
fall in one second, still accelerating. The island's ground is a streamed
heightfield, so the question is asked again over the real boot path:

* fixture: spawned at y 131.799, settles at y 130.811 — a **0.9883 m** drop over
  300 steps, then **+0.00000 m in its last second**;
* shipped island: **0.9769 m** of settle from y 17.56;
* the control, the same hero lifted 50 m, falls **4.9840 m** in its first second.

The first version of this arm asserted *"it barely moves"*, measured the same
0.9883 m with the terrain streamer switched OFF, and was therefore measuring
gravity rather than ground. The property is **rest**, and the control is what
makes it one.

The ~1 m of settle is real and it is the honest remainder: **the island spawns
its hero about a metre above its own ground**. It is a settle rather than a
plunge, no player would see it, and it is carried.

**And Play now renders what ships** — see CP-A1.

### CP-A1 / CP-B5 · The lit stack — MEASURED, RULED, FIXED

`fps_instrument.rs` has documented since island wave I4 that a level authoring no
render block ships with shadows, GI, VSM, TAA, SSAO, bloom and the visbuffer all
off. **No arm anywhere said WHICH levels those were.** All twenty-four were.

**The ruling, and it is a measurement.** The brief offered two routes:

* *"the default becomes lit for 3D levels"* — **reaches nothing that exists.**
  `RenderSettingsRecord` is persisted POSITIONALLY inside `RuntimeSettings`, so
  every committed `.inf_lvl` carries the values current when it was written;
  moving `Default` relights only levels created afterwards, and it breaks the
  standing both-hosts pin `apply_record(&default()) == RenderSettings::default()`.
  It would have left the island exactly as dark as it was.
* *"the levels author the lit stack"* — **taken.** Five committed levels move, at
  an **unchanged byte length on every one** (144 755 / 21 316 / 1 651 / 1 595 /
  2 150), because what moved is six bools and one f32 in place.

`RenderSettingsRecord::lit_showcase()` is the one definition: shadows, GI, bloom,
SSAO and TAA — exactly the five the fps instrument has priced as THE STACK'S
PRICE for four waves — plus flare, because the reference's daytime street frames
are sun-glare frames and the pass has existed since VIS1b. VSM is absent because
it has no authorable field; SSR is absent because it is a real cost that a
certification should route rather than switch on.

The island takes one knob beyond it: `ISLAND_SHADOW_DISTANCE_M = 250.0` against a
60 m default, because on a settlement of 100 m city blocks a 60 m cascade means
the building across the street casts nothing.

**The lit stack's price, over two runs on the same machine and content:**

| run | lit p95 | shipped p95 | delta | lit GPU | shipped GPU |
|---|---|---|---|---|---|
| whole suite | 16.862 | 16.733 | **+0.129 ms** | 5.506 | 7.575 (**−2.069**) |
| this arm alone | 20.499 | 14.209 | **+6.290 ms** | 5.439 | 3.409 (**+2.029**) |

Both are quoted because either alone is a claim. The lit frame's own GPU cost
barely moved (5.506 / 5.439); what moved is the *shipped* frame's, which is the
instrument's own warning about device state paying out. Against island wave I4's
**+48.6 ms**, the stack is now somewhere between a seventh of a millisecond and
six of them.

**AND PIE WAS RENDERING A DIFFERENT LEVEL.** `window::run_pie` built its
`PlayerApp` with `RenderSettingsRecord::default()` under a comment saying the
scene payload carried no settings. It carried them all along: `level_bytes` **is**
the live document's `.inf_lvl`, and `build_world_from_payload` has decoded the
record into `BuiltWorld::render` since R-P4 — `sim_from_payload` then dropped it.
So the moment the island went lit, the editor's Play button would have previewed
it unlit while the shipped build rendered it lit: **PIE ≠ shipping on the one half
of the frame no `state_bytes` fold can see**, and therefore the one half no
determinism gate could ever have caught. `run_web` and `run_android` had the
identical substitution. Closed with **no schema move** — the bytes were already
in the envelope.

Four arms, each red for a different reason: the five lit levels decode lit (six
bits read one at a time); three levels that were *not* ruled lit still decode to
the default (the control, without which arm 1 would pass on a flipped default); a
PIE payload's record is the level's and not the default, with the complement; and
a source gate that `window.rs` names `RenderSettingsRecord::default()` zero times
in code — stripping comments first, because the fix's own doc quotes the
expression it removed.

### CP-C3 / CP-A2 / CP-B9 · The LOD ladder — CENSUSED, ONE GAP CLOSED, ONE ROUTED

The census is per **draw path**, not per file, because four of the island's five
meshes have no meshlet DAG and three of those four do not need one.

| asset kind | tiers | the ladder |
|---|---|---|
| virtualized geometry (the road mesh) | **continuous** | 5 meshlet levels, 9 898 → 858 triangles; per-meshlet selection at a 1.0 px projected error |
| scatter / vegetation | **3** | mesh 0–120 m, impostor 100–400 m (20 m dithered fade), culled beyond 400 m |
| grammar buildings | **3** *(was 2)* | fit-out 0–64 m, fabric 0–96 m + reach, shell 96 m–draw |
| terrain | **4 rings** | 64/32/16/8 cells, per-vertex morph over the last 35 % of a band |
| textures (SVT) | **10–11** | see CP-C5 |
| **characters** | **1** | `Starter_Body` is 1 498 triangles and has no geometry LOD at any distance |
| **materials** | **1 fade** | see CP-C4 |

**The mid tier for grammar buildings, and the measurement that earned it.**
`INTERIOR_LOD_M = 64.0`: a building's FIT-OUT stops drawing where the building
stops having colliders (`inf_ecs::band::DEFAULT_COLLIDER_NEAR_M`, tied by an
assertion because `inf-render` cannot name `inf-ecs`), since past that you cannot
be inside it and the window between you and it is an opaque box.

The prescription was measured before it was landed, and **the first measurement
said not to land it**: over the fixture's resident set, fit-out is **1.0 %** of a
building's instances. That number is an artefact — the fixture's resident blocks
are Office, Apartment and Industrial zones and `settlement::furnishes` turns
furniture off for all three. Re-aimed at the subject, one furnished build of each
of the fourteen archetypes:

| archetype | instances | fit-out | | archetype | instances | fit-out |
|---|---|---|---|---|---|---|
| Apartment | 1 018 | 5.7 % | | PoliceStation | 676 | 13.3 % |
| House | 1 142 | 6.3 % | | Industrial | 351 | 17.9 % |
| Hotel | 884 | 6.8 % | | StripClub | 577 | 18.0 % |
| Shop | 857 | 8.1 % | | Nightclub | 639 | 18.5 % |
| Hospital | 654 | 8.7 % | | FireHall | 672 | 22.3 % |
| Estate | 683 | 10.2 % | | **Bar** | 774 | **23.6 %** |
| Clinic | 795 | 11.9 % | | | | |
| Office | 706 | 12.0 % | | **all fourteen** | **10 428** | **12.2 %** |

Twelve per cent of a settlement's building instances and a quarter of a bar's.
It costs no new data, no schema and no second pass — `push_scatter` has bucketed
by mesh GUID since I8b — and it is proven by the shipped projector on the
smallest world that can tell three complementary bands apart:

```
[   0.0,   64.0)  2 instances   fit-out
[   0.0,  102.9)  2 instances   fabric (96 m + the shell's own reach)
[  96.0, 1000.0)  1 instance    the shell box
```

Shutter, Sign, Festoon and Grille are classified as **fabric, deliberately**: a
roller shutter and a cell front are wall, and a sign plate and a string-light run
are the two families whose whole purpose is to emit.

**The character is the remaining gap and it is routed, with its number.** The
starter body is 1 498 triangles, one geometry tier at every distance. The crowd
has four *simulation* tiers (32 / 96 / 512 m) and three *shadow* tiers
(own silhouette < 32 m, shared box proxy < 96 m, nothing past 96 m), and no
geometry LOD and no impostors at all — the last is named in three places in the
tree as the future lever and implemented in none.

### CP-C4 · Material LOD — MEASURED, REFUSED WITH THE ARGUMENT

There is one distance-dependent material path and it is **in mip space, by
explicit ruling**. `vt_apply_detail` ramps the detail normal and the detail
roughness out over the last two levels of the pyramid:

```wgsl
let lodf = vt_lod(b, dx, dy);
let w = clamp((f32(mip_count - 1u) - lodf) * 0.5, 0.0, 1.0);
```

Its own header states the case against what this row was asked to price: *"the
classical fix is a distance ramp with two magic numbers in it. This uses the
pyramid instead… the fade needs no camera, no uniform and no tuning — and it is
correct under a magnifying zoom as well as a walk, which a distance ramp is
not."*

**Refused.** A far-band material path would be a second, weaker answer to a
question the mip chain already answers, and it would put the camera into a
projection that is deliberately camera-free. There is **no parallax and no POM
anywhere in the tree** (two hits, neither a material feature), so there is no
detail-mapping cost to shed either.

The honest remainder is not a band: it is that a material has no shading-model
switch at all (CP-B8), and that `SkinnedInstance` carries no material handle, so
the character's own `.inf_mat` reaches the GPU through nothing.

### CP-C5 · Texture LOD — MEASURED

Textures have been virtual since P26 and the ladder is continuous. The census, at
1080p / 70° fov over a 1 m-radius footprint, through the same `justified_mip` the
CPU floor, the GPU feedback shader and the visbuffer feedback all use:

| texture | extent | mips | 1 m | 4 m | 16 m | 64 m | 256 m | 1 024 m |
|---|---|---|---|---|---|---|---|---|
| 4 albedos | 1 024 | 11 | 0 | 2 | 4 | 6 | 8 | 10 |
| 10 others | 512 | 10 | 0 | 1 | 3 | 5 | 7 | 9 |

Ten and eleven rungs against the three that were asked for. The coarsest **3**
levels of every texture are resident unconditionally (`VT_FLOOR_LEVELS`); a tile
is 128 × 128 with a 4-texel border. Residency is decided in three lanes — an
analytic CPU floor capped at 16 tiles, a two-frame-latent GPU feedback OR-mask
capped at 256, and a speculative lane capped at 256 — and the arm asserts the
ladder MOVES, so a `justified_mip` that answered a constant would be red.

### CP-C6 · The controls — MEASURED, ONE ARM PER BINDING

Twenty arms, `runtime/inf-player/tests/controls_cert.rs`. Each presses a **literal
key code**, hands it to the shipped `PlayerUi` first (a key a dialog takes never
reaches the game), folds it through the shipped `InputMap` and `InputState`,
reduces it with `inf_player::input::held_actions`, and steps the shipped
`RuntimeSim`. Nothing writes an action name, an intent field or a component.

| binding | asserted world quantity | measured |
|---|---|---|
| W | forward velocity in the aim frame | **+3.750 m/s** forward, +0.000 lateral |
| S | same axis, opposite sign | **−3.750 m/s** |
| A | **sign** of lateral velocity, and opposition with D | **−3.750 m/s** |
| D | **sign** of lateral velocity, and opposition with A | **+3.750 m/s** |
| Shift | `Gait` variant + settled speed | `Sprint` at **6.500 m/s** |
| Ctrl | `Gait` variant + settled speed | `Walk` at **1.650 m/s** |
| *(nothing held)* | `Gait::default()` **by name** | **`Run` at 3.750 m/s** |
| E | bag count + the pickup entity's existence | bag **0 → 1**, entity gone |
| R | magazine count | 30 → 24 fired → **30 reloaded** |
| LMB | magazine **and** the step's own shot counter | 30 → 24, **6 shots** |
| RMB | `RotationMode` | VelocityDirection → **Aiming** → LookingDirection |
| scroll | equipped item id, moved **and back** | rifle → **pistol** → rifle |
| C (click) | `MovementMode` | **`Crouch` 30/30 steps** |
| C (long press) | `MovementMode`, against the click control | **`Prone` 29/45** (threshold 0.250 s) |
| C (click, sprinting) | `MovementMode` | **`Slide` 13/60** from 6.500 m/s |
| C (long press, sprinting) | vertical launch speed | **+2.336500 m/s**; the click peaked +0.000 |
| Space | launch + `MovementMode` + height gained | **+4.336500 m/s**, `FallFree`, rose **0.995 m** |
| Space (at water) | launch components, wet vs dry | wet fwd **+5.500000** / up +2.336500; dry fwd +0.000000 / up +4.336500 |
| I | panel open flag + the slots it shows | **open true (2 filled) → false** |
| Tab | `sim.steps()` | 30 frames = 30 steps; **60 frames with the dialog open = 0**; Tab closed it |

The keys are **literals and not read from the binding table**, deliberately: an
arm that asked the table which key is `move_x−` would press the swapped key
after a swap and stay green. The vertical numbers are exact to 1e-6 because the
expected launch is derived as `tuning + GRAVITY.y·DT` — the verb writes the speed
and the same fixed step then integrates one step of the fall.

**Falsification, five mutations:** swapping the two strafe arms' keys → 2 red;
pressing `KeyD` in both → 2 red (the sign clause *and* the opposite-signs
clause); raising the hold threshold past every press → 2 red and **both click
arms stay green**, so the discrimination is real; the dialog not pausing the sim
→ 1 red; the character not `player_controlled` → **18 of 20 red**. The two
survivors are `I` and `Tab`, correctly: they are decisions about the *session*,
not about the character.

Every binding in the owner's table reached its verb from a key. None had to be
driven by action name; none was missing an implementation.

### CP-C7 · Terrain quality — MEASURED, THREE DEFECTS FIXED, FOUR PRICED

The owner's sentence was *"not all jagged and sharp and glitched out"*, and three
of the four causes were mechanisms.

| defect | before | after | bound |
|---|---|---|---|
| the morph factor was one number per 256 m patch | **3.8262 m** shared-edge crack | **0.0000 m** | < 1 mm |
| the fragment normal ignored the morph | **10.586°** at morph 1.0 | **0.000°** at every morph | < 0.5° |
| the fragment normal clamped at every tile edge | **14.688°** step, gradient **0.4875×** the truth | **0.707°**, **0.9751×** | < 3.5°, ±10 % |

**Why the crack was guaranteed rather than occasional.** `TERRAIN_MORPH_REGION`
is 0.35 of a band = 134.4 m in rings 0 and 1, and adjacent tile centres on this
island are **256 m** apart — so two neighbours can never both be inside the ramp:
one is pinned at 0 and the other at 1, every time, and the gap is the full chord
deviation.

**A fourth defect the brief did not name, and it is the one that made the crack
so large.** The morph target was `round(uv/step)*step` — a nearest-vertex snap,
and WGSL `round` is ties-to-even, so odd vertices snapped in alternating
directions. At full morph the grid never became the coarser grid, and surviving
segments spanned 4 m where the coarse mesh spans 8: **double the local slope**,
literally the thing this row was called for. It is now a bilinear on the coarse
lattice — the chord the coarser mesh actually rasterizes. Fixing the fragment
normal over a snapped target would have made it *dead flat* at full morph, so
this had to be found before the next one could be fixed at all.

**Priced, not fixed:**

* **the height pyramid is point-decimated, not filtered** — level 1 differs from
  a 3×3 tent by **RMS 0.1748 / max 0.3205 m**, level 2 by **0.5877 / 1.1189**.
  Deliberate (`pyramid.rs`: anything else cracks the LOD mesh at a shared edge)
  and it is the distance-shimmer source.
* **the streamer's ladder and the renderer's do not change gear together.** Render
  rings at 384/768/1536 m, asset switches at 1088/2176/4352 m, so texels per mesh
  cell runs **4 / 8 / 16 / 8 / 16 / 8 / 4 — a 4× spread**. `ring_source_lod`'s
  claim that this stays constant is **false and the doc is corrected**, with an
  arm that asserts the claim is false so it cannot drift back.
* the asset-LOD switch is un-morphed: **0.6359 m** at 1088 m, **2.1737 m** at
  2176 m.
* ring 0's mesh is 4 m a vertex over 1 m data — **max 0.2843 m** off its own
  height field, **0.0720 m at 128 cells**, which closes 75 % of it at the 4×
  triangles this file's own 32→64 note prices at ~22.6 % more terrain time.

**Resolution, honestly.** The grid is 1 m per sample over 7 168 m — 51.38 km²,
51.4 M samples — which is competitive with any shipping landscape. The *content*
is 3.113 m (z15 terrarium at 49.34 N), the band from 6.226 m down to 3.113 m is
a 2-octave synthetic fill capped at 1.5 m (mean 0.093, worst 1.493 over 37.7 M of
51.8 M samples), and the band from 3.113 m to 2 m is deliberately empty. So:
**high-resolution storage, mid-resolution content, and a rendered silhouette of
4 m at best and 16–32 m past 768 m.**

**Goldens: 62 files, 121 arms, green under `INF_GOLDEN_STRICT=1`, none blessed** —
and the honest reading is in the arms, not the harness. `INF_GOLDEN_STRICT`
downscales to 64 × 36 with a 6 % mean tolerance, so a one-texel line every 256 m
is below its resolution by construction. What makes the result mean something is
the arm that asserts at morph 0 — every golden's near ground — the interior is
**bit-identical over 65 025 of 66 049 fragments** and only the **1-texel edge
ring, 1.55 %, moves**.

### CP-B2 · The streets are never empty — MEASURED

Shipped island, 7 000 m of resident street, after the plan queues settle:

| local hour | parked | driving | pedestrians | per 100 m |
|---|---|---|---|---|
| 08:30 | 292 (4.17/100 m) | **109 (1.56/100 m)** | 1 000 (14.29/100 m) | **20.01** |
| 14:00 | 363 (5.19/100 m) | 38 (0.54/100 m) | 1 000 (14.29/100 m) | 20.01 |
| 21:00 | 363 (5.19/100 m) | 38 (0.54/100 m) | 1 000 (14.29/100 m) | 20.01 |

**The street is never empty at any hour, and the density is the same at all
three** — because the population is capped (`SOCIETY_MAX_AGENTS = 1 000`,
`MAX_COMMUTERS = 128`) and a car is either parked or driving. What moves is the
split: **2.87× as many cars driving at the rush hour as at midday.**

**And 14:00 and 21:00 are identical, which is the honest limit.** The traffic
draw has a night-circuit band (6 %) and a day-circuit band (14 %), but the
schedule that decides who is *out* does not distinguish an afternoon from an
evening on this measurement. Against the reference — a full hotel lot at midday
(driving/0030) and a parked row under streetlights at night (steal-car/0028) —
the counts are plausible and the *diurnal texture* is not there.

The compounding bound is VEH2b's carried item and it is a wall in the literal
sense: a `Full` crowd agent carries a **kinematic** capsule, so a car among the
eighty-one at the fixture's crossroads is not slowed, it is walled in —
**4.7 m covered in ten seconds at full throttle, against 87.4 m at 18.14 m/s on
an empty street.**

### CP-B6 · Carjack and the wanted chain — CERTIFIED BY TRACE, WITH THE GAP NAMED

The chain exists end to end and is armed in **two halves that have never met in
one test**, which is itself the finding:

* `island_gate::pie_equals_shipping_at_rush_hour_with_cars_on_the_streets` drives
  a **real** carjack on the island — approach, `INTERACT`, the resist draw
  (`RESIST_CHANCE` 0.25), the eject, `mark_taken`, the drive-off — measured at
  **four presses**, the victim **4.8 m** from the seat, **631 digested steps
  byte-identical on both hosts**. It observes no wanted level.
* `ems3_crime_gate` drives the **whole** wanted system — profile, evidence
  channels (`Outfit` 0.60, `Vehicle` 0.85), recognition ranges (16.7 m outfit /
  23.5 m vehicle / 25.1 m both in daylight; 1.1 / 12.5 / 15.2 at night), heat,
  stars (`WANTED_STARS = [1, 3, 6, 10, 15]`), dispatch, decay — but it raises the
  act by calling `witness::raise_act` directly, because `try_carjack` needs a
  `PhysicsBridge3D` the editor's `SimSession` does not expose.

So a carjack is one heat point, one star, `Response::Patrol`, one unit. **No
single arm carjacks a real traffic driver and then reads `crime::heat_of`.**
Carried, with the ingredients named: `island_gate::rush_hour` already has the
target selection, the door stand and the press loop.

**The animation gap, plainly.** There is **no yank-door clip and no carjack clip
of any kind** — zero hits in `inf-anim`, in any `.inf_sm`, in any clip table. The
eject is `finish_driving`'s one-frame transform write to a computed door point;
the victim's only "animation" is `MovementMode::FallControlled` and then an
ordinary crowd walk of `FLEE_M = 40 m`, after which it stands (it is given a
route, not a schedule). Against the reference's carjack beats (steal-car
0010–0035: approach → yank → resist → drive off) this is the whole gap.

### CP-B10 · Building illumination — MEASURED AND PRICED (the PAR arc builds it)

The owner's ruling is that a glowing window pane is not illumination: *"We should
have actual lights in this game and game engine."* The census says how far that
is from true.

**The island's resident world, counted by the arm:**

| | |
|---|---|
| authored `Light` entities | **1** (the sun; the sky adds one key light) |
| `PcgVolume` blocks resident | 4 |
| …carrying **any** real fixture | **0** |
| real `PcgLight` fixtures | **0** |
| scattered instances | 22 989 |
| …that merely **glow** (window panes) | **2 122** |

**Eleven of fourteen archetypes hang zero lights.** `Assembler::rig` is the only
producer of a `PcgLight` in the whole engine, it is gated on
`arch.rig.is_some()`, and only Stage, DanceFloor and BarRoom rooms match — which
is Bar (1 fixture), Nightclub (4) and StripClub (4). Every house, apartment,
office, hotel, shop, factory, police station, fire hall, hospital and clinic is
lit by the sun, the moon and its own emissive panes. Anchors are ground-floor
only, so **every light in the engine is on floor 0**.

**The wall is not sixteen; it is four, and it is upstream.**
`MAX_LIGHTS = 16` per frame, first-N in scene order, `.take(count)` — no distance
sort, no cull, no priority anywhere between the ECS and the uniform. But
`VOLUME_LIGHT_CAP = 4` truncates *per city block*, in the content layer, before
the renderer ever sees them. A city Bar block is 15 bars → 15 fixtures → **4
kept**; a Nightclub block is 9 clubs × 4 → 36 → **4 kept**; a StripClub block 12
× 4 → 48 → **4 kept**. **About 92 of the 99 fixtures a nightlife strip already
builds are deleted before the frame.** Three blocks at the cap fit beside the
sun; the fourth overflows.

**The many-lights number.** Harbour City is 750 buildings over 20.88 km of
street, ~22 956 interior rooms, 329 residents of which ≥66 % are home at 21:00.
A defensible *minimum* for one lit settlement — a porch light per building, a
fixture per occupied room, a street lamp every 30 m — is **≈ 1 670 lights**;
one-per-room is **22 956**. Island-wide, 1 814 buildings and 60.88 km of street
give **≈ 5 000** at the porch-plus-street floor and **≈ 61 000** one-per-room.
Against a ceiling of 16, that is **104× at the floor and 1 435× at the room.**

**Three findings that shape PAR0/PAR1 more than the count does:**

1. **A real interior light would look wrong today, and not because of the count.**
   Point and spot lights are unshadowed in every shipped configuration —
   `VsmSettings::enabled` defaults false and nothing in `apply_record` or the tier
   clamp turns it on; every `PcgLight` is pinned `cast_shadows: false`; and
   `RenderLight::cast_shadows` is documented as inert for point/spot. A fixture
   placed in a room today lights **through** the walls, the floor and the ceiling,
   windowed only by `1/d²` and a hard `range` cutoff.
2. **The window is a real hole with an opaque box in it.** The wall grammar
   expands only the gaps *between* openings — a doorway is a genuine, unobstructed
   void with no leaf and no jamb — but a window's pane is a **box instance filling
   the void exactly**, on the decoration tail (so no collider), drawn through the
   scatter path, and `ScatterBatch` has **no blend-mode field at all**. So window
   glass does **not** transmit: an interior light would be invisible from the
   street, and light escapes only through doorways. The reference's own frame for
   this is steal-car/0016, where a gallery's window shows a lit room with
   paintings on the wall behind it.
3. **There is no night schedule.** Every volume light is pushed unconditionally
   in both hosts; the clock is read only by `swept_colour`, which changes the hue
   and never the intensity. A nightclub's stage rig burns at intensity 26.0 at
   11 a.m.

**Exterior lights do not exist in any form.** No porch lights, no signage lights,
no street lamps, and **no vehicle headlights** — one grep hit repo-wide and it is
a comment. EMS light bars are emissive cubes pulsed at 2 Hz. Terrain and water do
not read the lights array at all, so a street lamp would not light the road it
stands on.

**Seams a clustered pass (PAR0) must touch:** `MAX_LIGHTS` and its six WGSL
mirrors; `LightsUniform` (1 040 B uniform → a storage buffer, which changes the
binding *type* and therefore nine pipeline layouts and nine bind groups); the six
identical shader loops; `gi.rs`'s single-directional read; and — the hard one —
`params.w` is the **VSM slot**, and both `vsm_light_trees` and `receiver_slots`
are written against *"handle n is the n-th shadow-casting light in scene order"*,
so **a reorder must land the slot re-derivation in the same change**. Content-side
caps that relax with it: `VOLUME_LIGHT_CAP`, `MAX_RIG_SPOTS`. Goldens that would
move: `spot_lights`, `venue_interior`, the four VSM frames, `csm`, `2d_lit`.

**There is no measured per-frame cost for the light loop anywhere**, and with
`count` at 2–6 against a bound of 16 there would not be one to find. **PAR0's
justification is capability, not throughput**, and any performance claim about it
needs an instrument that does not exist yet.

### CP-B3 · Street furniture and street lighting — MEASURED, ZERO, PRICED

Twenty-two nouns, twenty-two empty greps across `crates/`, `runtime/`, `editor/`:
`hydrant`, `awning`, `billboard`, `streetlight`, `street_light`,
`traffic_signal`, `lamppost`, `utility_pole`, `power_line`, `overhead_line`,
`manhole`, `bus_stop`, `parking_meter`, `mailbox`, `litter_bin`, `trash_can`, and
the adjacent `decal`, `road_marking`, `crosswalk`, `zebra`, `lane marking`,
`kerb_mesh`. The `lamp` / `pole` / `bench` hits are all VEN1a **interior** venue
furniture — `ModuleShape::Pole` is a chrome dance pole.

**The pavement is never drawn either.** `PAVEMENT_M = 2.0` is a nav ring of eight
nodes; there is no kerb geometry, no pavement mesh and no road marking.

**The scatter is cheap and the lighting is the wave.** The precedent is
`inf_ecs::traffic::kerb_slots` — a deterministic, world-lattice, geometry-derived
scatter along mitred lane centrelines with a hash-driven occupancy draw and
content-addressed identity (`KERB_SLOT_M = 14.0`, `KERB_PARK_OFFSET_M = 5.0`,
`KERB_OCCUPANCY = 0.45`, `JUNCTION_CLEAR_M = 2.0`). The mesh vocabulary is two
functions — `push_box` and `push_prism_y`, an eight-sided prism, the only
non-box primitive in the tree. A lamp post is a prism and two boxes.

**But fifty lamps down a street cannot be sixteen lights.** PAR1 is blocked on
PAR0, and the codebase already says so against itself, at `MAX_RIG_SPOTS`:
*"`MAX_LIGHTS` is 16 for the whole FRAME, so a palette that asked for twenty
would not merely be extravagant — it would push the sun out of the uniform."*

Against driving/0006 and steal-car/0028 the delta is total: the reference frames
carry six or more overhead catenary spans, poles every ~30 m, a hydrant, signage,
and streetlight pools on the asphalt. The island has none of them, and its night
streets are lit by neon and venue rigs only.

### CP-B1 · Haze and volumetrics — **THE FRAMING WAS WRONG**; MEASURED, NARROWED

**Aerial perspective and analytic height fog already ship, and both apply to lit
scene geometry.** `atmos_apply(color, world_pos)` is called from **seven** lit
shaders — mesh, scatter, skinned, terrain, vgeom, vis_resolve and water. Aerial
perspective is `color·t_air + sky·(1 − t_air)` with `t_air = exp(−σ·km·strength)`;
height fog is a closed-form analytic optical depth, `σ(y) = density·exp(−falloff·(y − height))`,
clamped at 64. No ray march. Fog is fully authorable through
`SkyAtmosphere::fog_density` (scene v12/v13) with weather presets from Clear 0.0
to Fog 6.0e-3.

**And there is already a depth-aware volumetric raymarch**: the P17.3 cloud pass
marches a world-space slab in SI metres, clamps `t_far` at geometry inside the
slab, and reads MSAA depth. Its limit is that the slab is an *altitude band*, not
a camera-fitted frustum volume — so the machinery transfers to PAR3 and the
geometry does not. Cloud shadow reaches the ground as a single mid-altitude
plane, 512² over 20 km ≈ 39 m/texel.

**What is actually missing** is narrower than the brief assumed: (i) the 3D
aerial-perspective froxel LUT, named as the follow-up in-source twice; (ii)
**shadowed local-light in-scatter — the beam — which has zero prior art**, and
the cloud march uses zero local lights; (iii) one real hole, `voxel.wgsl` has no
`atmos_apply` and no haze at all, so a cave surface does not fade with distance
while the terrain beside it does.

**The seam.** The render graph is a flat ordered list of 34 nodes, and the
insertion point is between `scatter` (the last opaque) and `cloud`: MSAA depth is
complete and `TEXTURE_BINDING`-capable, CSM is ready, VSM has been rastered, and
tonemap is 66 lines downstream so volumetric radiance blooms and tonemaps for
free. The half-res precedent is one line (`(width/2, height/2)`) and the cloud
stack's depth-aware bilateral upsample is the template.

**And (ii) inherits the 16-light wall, harder than a surface does**: a fog
volume's scattering from a light is visible far outside that light's own lit
footprint, so the seventeenth street lamp is exactly the one that cannot be
dropped.

Against trailer/0088 — a dusty sunset road where the low sun is a visible volume
— the engine has the extinction and none of the in-scatter.

### CP-B4 · Coronas and flare — **THE FRAMING WAS WRONG**; MEASURED, NARROWED

**A four-term lens flare pass already ships** (`passes/flare.rs`, half-res, after
bloom and before tonemap): veiling glare (16-tap radial gather, 0.94^i decay),
a ghost chain (8 scales, chromatic tint), a halo ring at radius 0.32, and a 7-tap
anamorphic streak.

**Three of the four already fire off any bright pixel, and the file says so**:
*"a ghost is an image of whatever is bright, and a window at the edge of frame
throws one too"* — measured in `golden.rs` on an `emissive = [30.0; 3]` slab.
Only the **veiling glare** is sun-locked, and for four concrete reasons: it reads
one `view.sun_dir`, it places the source 10 km down a *direction* (a point light
has a position), the pass binds **no lights uniform at all** (five entries:
uniform, scene, sampler, depth, exposure), and its occlusion test is a 3×3 depth
gather keyed on reverse-Z "sky".

**Would a bright point light already bloom? No for a light, yes for emissive
geometry.** `RenderLight` has no mesh, no proxy and no radius geometry — a point
light is never rasterised as anything. But `InstanceRaw.emissive` is unclamped
into `Rgba16Float`, and `GLAZING_GLOW = 1.6` at full night emits
`[1.6, 1.376, 0.992]`, which clears both the bloom threshold (1.0, soft-knee
f = 0.375) and `FLARE_THRESHOLD`.

So PAR4 is two things, not five: a veiling term around a non-sun source (which
needs the per-light loop and the lights binding the pass deliberately lacks), and
any on-screen representation of an analytic point light at all. **Both
`bloom.enabled` and `flare.enabled` default `false`** — the island now authors
both on.

### CP-B7 · Particles — MEASURED, ZERO, PRICED

`ParticleSystem`, `ParticleEmitter`, `spawn_rate`: three empty greps. The
codebase states the absence **five times in its own source**.

What exists is a `Sprite` billboard on the P8 2D pipeline. EMS2's smoke is one
ECS entity per puff — `PUFF_PERIOD = 20` steps, `PUFF_LIFETIME_S = 3.5`,
`PUFF_RISE_MPS = 1.6`, `MAX_PUFFS = 64` **for the whole level**, ~11 alive per
fire — whose motion is **rise only** (no drift, no wind, no collision, rotation
hard-coded 0.0), whose fade is linear, and which is **never culled** (puffs carry
no `Visibility`, so all 64 draw every frame). The muzzle flash is not a sprite at
all: it is two **debug line** segments for one frame.

`precip.rs` is a real GPU quad-particle renderer — 6 verts × up to 48 000
instances — but a **stateless procedural field**: three integer hashes on
`instance_index`, no buffer, no spawn, no lifetime, no despawn, no sorting, no
emitters. It donates the draw side and nothing else, and it disqualifies itself
in-source.

**Seams:** a node in `renderer.rs`'s 34-call registration list (order *is* pass
order); a row in `SHADER_TABLE`, naga-gated; a sim step inside the
`MIRROR-BEGIN` fences of **both** `runtime_sim.rs` and `simulate.rs`; an
`AssetKind` plus five paired functions plus `compresses_kind`; and three cook
sites. The right donor for the GPU half is the P18.5 scatter cull (workgroup 256,
frustum + HZB, in-workgroup prefix sum, GPU-side compaction into an indirect
draw). **The P22 debris path is a trap**: its fragments deliberately never move,
because `ScatterData::key` is a content hash and a moving instance re-uploads the
whole buffer every step.

Against driving/0006 (tyre smoke), steal-car/0040 (exhaust flames) and
venues/0004 (a cigarette ember) the delta is total. Tyre smoke has its input
ready and unread — VEH2a publishes slip and nothing consumes it.

### CP-B8 · Skin and subsurface — MEASURED, ZERO, PRICED

`sss`, `subsurface`, `wrap_light`, `burley`, `diffusion_profile`, `transmission`:
empty. One hit, a doc comment on `MapKind::Translucency`, an import-time texture
slot with no consumer.

The BRDF is GGX + **plain Lambert**, and it is duplicated **sixfold** with no
shared include — `distribution_ggx` / `geometry_smith` / `fresnel_schlick` /
`shade_light` appear verbatim in six shaders. The diffuse line, from the
character's own shader: `let diffuse = kd * albedo / PI;`. Six lines, one grep.

`Starter_Skin.inf_mat` is `base_color [0.62, 0.58, 0.55]`, `metallic 0.0`,
`roughness 0.62`, three textures `None`. **`MaterialAsset` has no shading-model
field at all**, and — the structural blocker — **`SkinnedInstance` carries no
material handle**, so the starter skin material reaches the GPU through nothing.
The character crate says so about itself: *"`SkeletalMesh` carries a mesh and a
skeleton and no material, so nothing in the skinned draw path reads this yet."*

**The cheapest honest route** is to hoist a `diffuse_term()` into
`env_lighting.wgsl` — the only shared-shader-library precedent — before adding a
wrap or Burley term, or the edit is sixfold. There is no permutation system (8
greps, all prose, several arguing against needing one); the mechanism is one
uber-shader per pass branched on uniform flags, and there are **two spare `f32`
words**: `emissive.w` on the rigid path and `pbr.w` on the skinned one, both
uploaded as a literal 0.0 today, which is the exact byte-stability shape every
prior flag used.

Against venues/0068 and all-clips/0060 — a face under a red stage wash with a
blue rim — the engine has the lights and none of the skin.

---

## The numbers

### The frame, on an RTX 4070 Ti at 1080p, MIN of 3 rounds × 120 frames

| scene / configuration | p50 | p95 | p99 | fps @ p50 |
|---|---|---|---|---|
| the composed city, shipped | 15.353 | 16.733 | 17.067 | 65.1 |
| the composed city, LIT | 15.735 | 16.862 | 17.592 | 63.6 |
| **the island, as its level now ships** | **39.672** | **42.299** | 43.188 | **25.2** |
| the island, LIT (+VSM) | 34.237 | 37.671 | 39.299 | 29.2 |
| the island, LIT + SSR | 34.659 | 36.775 | 38.074 | 28.9 |
| the island, LIT + visbuffer | 37.023 | 60.045 | 71.198 | 27.0 |
| the island, LIT, coarse clipmap | 33.990 | 36.188 | 37.932 | 29.4 |
| the island, LIT+VIS, town cleared | 42.719 | 47.068 | 59.410 | 23.4 |
| the island, LIT+VIS, crowd cleared | 34.543 | 37.956 | 43.875 | 28.9 |
| the island, LIT+VIS + 1 000 NPCs in a 320 m block | 54.922 | 79.940 | 83.699 | 18.2 |
| **the island, LIT+VIS, its own society at the rush hour** | **138.701** | **153.814** | 169.246 | **7.2** |

The island's frame is **CPU-bound**: crowd cleared, the pipelined estimate is
21.065 ms of CPU against a 12.834 ms GPU frame. The fixed step alone is
**7.664 ms/step** against a 6.0 ms ratchet, and the dearest CPU stages are the
solver (3.295 ms) and the physics3d sync (3.035 ms). The dearest GPU pass is
scatter, at 5.916 ms of 12.834.

The island builds in **54.0 s** and cooks in **43.1 s**; the cooked pack is
**249 125 340 B** for 55 assets, and its `.inf_terrain` alone is 549 879 456 B.

`SHIPPING_FRAME_CEILING_MS` is **re-minted 40.0 → 38.0** and
`SHIPPING_FRAME_P99_CEILING_MS` **48.0 → 46.0**, and the instrument now asserts
them over the **lit** configuration as well as the shipped one — which is the
discharge of the constant's own standing clause, *"the day a shipped level turns
the stack on, this ceiling is measuring a different renderer and has to be
re-minted, not raised."*

### The content

| | |
|---|---|
| island extent | 7 168 m square, 51.38 km², 51.4 M level-0 height samples |
| terrain resolution | 1.0 m/sample stored, 3.113 m surveyed, 6.226 → 3.113 m synthesised |
| settlements | 2 cities + 5 towns, 172 blocks, **60.88 km** of street, ~1 814 buildings |
| inter-settlement road | 33.74 km over 11 links and 7 junctions — **carries no traffic** |
| meshes in the cooked island | **5** (roads, the starter body, three ground-cover props) |
| meshlet DAGs derived | **1** (the road mesh) |
| virtual textures | 14, 512²–1024², 10–11 mips |
| the hero | **1 498 triangles**, one geometry tier |

---

## Honest — what only eyes on the frames can judge, and what I saw

I looked at ten frames across seven of the twelve folders. This is what they say
that no arm in this repository can.

**driving/0006 · a sun-glare street.** Six or more overhead catenary spans
crossing the frame, poles at roughly every third of it, a billboard, palms, a
weathered asphalt with lane paint and patch scars, tyre smoke behind a ute, and a
low sun with real veiling glare. The engine has the sun and the glare. It has
**none of the poles, none of the cables, no billboard, no lane markings, no
kerbs, no pavement and no smoke.** This single frame is the whole of CP-B3 and
CP-B7 in one image.

**driving/0030 · a hotel forecourt.** A *full* car park — I count ten-plus parked
cars — a fire hydrant, flowering shrubs with per-leaf variation, individual
rocks, peds at the entrance, and a yellow stucco building with a real sign and a
logo. Our parked lattice would populate that lot; our hydrant, shrubs, rocks and
sign would not exist.

**steal-car/0028 · night, red signals.** Traffic signals with visible coronas,
streetlight pools on the asphalt, a parked row of white sedans, a wet-ish
specular road, headlights throwing real cones, brake lights pooling red. We have
the parked row and the black-blue sky (GTA1 closed the red band). We have **no
signals, no lamps, no pools, no headlights** — vehicle headlights do not exist in
this engine in any form, not even as an emissive quad.

**steal-car/0016 · a gallery shopfront.** This is the frame the owner's ruling is
about. The window is **transparent**, and behind it is a lit room with paintings
on the wall and furniture on the floor. A street tree is uplit from the ground and
throws a shadow onto the building. A pink neon sign has a corona. Our window is an
**opaque emissive box**; our tree has no uplight; our neon does bloom. What the
owner asked for is literally this frame, and the census above says what stands
between us and it — glass that transmits, a light that can be inside a room, and a
shadow policy for point lights.

**prologue/0032 · a cell interior.** A ceiling fluorescent fixture visibly
casting a pool of light down the wall, a transparent window with an exterior
behind it, posters, bunks, props on a table. One real fixture, in one ordinary
room, in a building that is not a nightclub. Eleven of our fourteen archetypes
have none.

**venues/0020 · the strip-club stage.** Near-black ambient, a deep red stage wash
on a wooden catwalk, blue rim, twenty-plus individually visible string-light
bulbs, a chrome pole with a bright specular, benches, patrons. Our
`venue_interior` golden has the *idea* — black ambient, magenta and red pools, a
string of small emitters, an emissive sign — and it is a **box**: no benches, no
patrons, no wood grain, no beam haze. The lighting recipe is right; the room is
not built.

**all-clips/0060 · a bar.** Faces lit by blue neon and red practicals at the same
time. Ours would be Lambert.

**beach/0012 · a crowded beach.** Sixty to a hundred distinct people at every
distance, clearly LOD'd — near ones with clothing, far ones as simplified
silhouettes — plus footprint decals, seaweed, umbrellas, a kiosk, boats, and a
hazy skyline. We can put a thousand agents on the ground and it costs us
**7.2 fps**; and we have no crowd impostors at all, which is the named lever we
have never pulled.

**all-clips/0100 · an aerial vista.** Towers a kilometre away that still have
window mullions and balconies in silhouette; a marina with hundreds of individual
boats; aerial haze doing real work on the skyline. Our aerial perspective is
there. Our distant buildings are one shell box each past 96 m, and the census
says why: there is nothing between the parts and the box except the fit-out band
this wave added, which is a *near* band, not a far one.

**trailer/0088 · a dusty sunset road.** The air is a volume. We have the
extinction that dims the road and none of the in-scatter that makes the light
visible.

**What I could not judge and no arm here does:** whether the island *looks*
right at 1080p in motion. There is no committed capture of the island's own frame
at any resolution — every golden in this repository is 320 × 180, and
`INF_GOLDEN_STRICT` compares at 64 × 36 with a 6 % mean tolerance. The terrain
row is armed at the CPU-twin level precisely because the harness cannot see a
one-texel line, and the fixes are asserted in degrees and metres rather than in
pixels. Someone has to open the editor and fly it.

Also human-verified, and unchanged from the AAA certification: the right-click
menu, the flycam's feel, the DPI matrix, macOS viewport input (which does not
exist), and LSP/terminal/git runtime.

---

## FIX-NOW — closed by this wave

| # | what | evidence |
|---|---|---|
| FN-1 | The application had no boot project at all | `inf_project::boot`, 8 + 2 + 3 arms |
| FN-2 | Every committed level shipped unlit | 5 levels re-blessed; `lit_stack.rs` |
| FN-3 | **PIE, web and Android substituted a default render block for the level's own** | `PayloadSim::render`; a source gate on `window.rs` |
| FN-4 | The terrain cracked by up to 3.8262 m along a guaranteed seam | 0.0000 m; `terrain_continuity.rs` |
| FN-5 | The fragment normal lit a surface the vertex had moved away from | 10.586° → 0.000° |
| FN-6 | Every tile edge carried a half-gradient shading band, world-wide | 14.688° → 0.707°, 0.4875× → 0.9751× |
| FN-7 | The morph target was a ties-to-even SNAP, doubling local slope at full morph | replaced with the coarse-lattice bilinear |
| FN-8 | Grammar buildings had two distance tiers | three, at 12.2 % of a furnished building's instances |
| FN-9 | `ring_source_lod` claimed constant texels-per-cell; it is a 4× spread | doc corrected, arm asserts the claim is false |
| FN-10 | `PlayerRenderHost::tier`'s doc claimed the `--pie` loop builds no host | false for the windowed branch; corrected |
| FN-11 | No arm anywhere asserted what a committed level draws | `lit_stack.rs`, with a control |
| FN-12 | Six binding rows had no key-level arm (`S`, `A`, `D`, Space×2, C-long) | `controls_cert.rs`, 20 arms |

---

## Deferred, with the reason and the number

| # | item | routed to | the number |
|---|---|---|---|
| D-1 | Street furniture and street lighting | **PAR1**, behind PAR0 | 22 nouns, 22 empty greps; a lamp is a prism and two boxes; 50 lamps ≫ 16 lights |
| D-2 | A real particle system | **PAR2** | absent, stated 5× in source; `MAX_PUFFS = 64` for a level; the muzzle flash is a debug line |
| D-3 | Volumetric in-scatter (the beam) | **PAR3**, narrowed | aerial perspective + height fog already ship on 7 shaders; the froxel LUT and shadowed local in-scatter do not; `voxel.wgsl` has no fog at all |
| D-4 | A veiling corona around a non-sun light | **PAR4**, narrowed | ghosts, halo and streak already fire off any bright pixel; only the veil is sun-locked, for 4 concrete reasons |
| D-5 | Skin / subsurface | **PAR5** | Lambert, sixfold; `SkinnedInstance` has no material handle — that blocker first |
| D-6 | Clustered light culling | **PAR0**, blocks PAR1 | 16 per frame, first-N, no sort; `VOLUME_LIGHT_CAP = 4` deletes ~92 of 99 on a nightlife strip; a lit settlement wants ≈1 670 |
| D-7 | Point/spot shadows | **PAR0** | VSM off in every shipped configuration; `cast_shadows` inert for point/spot; a room light leaks through its walls |
| D-8 | Window glass that transmits | **PAR0/PAR1** | the pane is an opaque box on a `ScatterBatch` with no blend mode |
| D-9 | A night schedule for lights | PAR1 | every volume light pushed unconditionally in both hosts; a stage rig burns at 11 a.m. |
| D-10 | Character geometry LOD + crowd impostors | anim/crowd | 1 498 triangles at every distance; impostors named in 3 places, built in none |
| D-11 | One arm that carjacks a real driver **and** reads the wanted level | EMS/VEH | the chain is armed in two halves that never meet |
| D-12 | A yank-door animation | VEH | zero carjack clips anywhere; the eject is a one-frame transform write |
| D-13 | The height pyramid is point-decimated | terrain | RMS 0.1748 / 0.5877 m against a tent at levels 1 / 2 |
| D-14 | Streamer and renderer LOD ladders disagree | terrain | 4 / 8 / 16 / 8 / 16 / 8 / 4 texels per mesh cell |
| D-15 | The asset-LOD switch is un-morphed | terrain | 0.6359 m at 1088 m, 2.1737 m at 2176 m |
| D-16 | Ring 0 is 4 m a vertex over 1 m data | terrain | max 0.2843 m; 128 cells closes 75 % at ~22.6 % more terrain time |
| D-17 | A tile-edge apron for continuous normals | terrain | the residual is 1.318° across a seam; needs a `.inf_terrain` change |
| D-18 | The hero spawns ~1 m above its ground | island | 0.9883 m (fixture), 0.9769 m (shipped) |
| D-19 | 14:00 and 21:00 are the same street | NPC | identical counts; the schedule has no evening |
| D-20 | A crowd is a wall | NPC | 4.7 m in 10 s against 87.4 m on an empty street |
| D-21 | No committed capture of the island's frame at any resolution | — | every golden is 320 × 180; strict compares at 64 × 36 |

---

## What this wave did not touch

Scene schema **v27**, `ScenePayload` **v12**, `EXPECTED_LEVELS` **24**, goldens
**62 files / 121 arms** with none blessed, `Cargo.toml` / `Cargo.lock` /
`deny.toml` unmoved, no new dependency. Five committed `.inf_lvl` files changed
content hash at an unchanged byte length, for the stated cause in CP-A1.
