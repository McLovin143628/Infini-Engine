# Wave GTA1 — the Play button plays, and the night sky is night

**Mandate:** 2026-08-31, wave 1 of 6. Six clauses: the night sky's red horizon,
PIE carrying terrain by path so the island plays, starter levels that ship a
pawn, an honest no-pawn Play dialog, menus that reach the levels, and this
ledger.

**Base:** `bf1954e6` (main, CI green). **Commits, in order:**

| | |
|---|---|
| `3cd6dbd0` | clause 1 — the planet's own shadow gates every body term |
| `1af795f3` | clause 2 — the island's terrain rides as a PATH |
| `496914a6` | clause 3 — every 3D starter level ships a pawn |
| `82cc9a15` | clause 2 follow-up — the retired byte-reader |
| `117b9ed0` | clause 5 — the menus reach the levels |
| `101c2a07` | clause 4 — Play with no pawn asks a question |
| `ec261c7c` | clause 1 follow-up — the GI probes' night arm |
| `158ca5d4` | the five pins this wave's committed images and content fired |
| `a8060ca4` | the island terrain figure, corrected (549.9 MB, not 342.7) |
| `544e2711` | the rustdoc link that pointed up the ring order |
| `3f8a0cc4` | the digest pin's note is one doc comment, not two |
| (this memo) | clause 6 — the ledger |

**Schemas:** the scene schema is **unmoved** (v26). `ScenePayload` **v11 → v12**,
one append-only tail field — the wave's one version move, priced below.
**Goldens 59 → 60** (one new; two re-blessed). **Content re-bless:** three
committed template `.inf_lvl`s and their sidecars.

---

## Clause 1 — the night sky loses its red horizon

### The defect, and what it actually was

The transmittance LUT is parameterised the Bruneton way: `u` runs over the
distance to the top of the atmosphere, from straight up (`d_min`) to the
**horizon-tangent ray** (`d_max`) — the longest path that still misses the
planet. Every direction *below* that tangent is off the end of the axis, and the
sampler is `ClampToEdge` (deliberately, and correctly: a repeating sampler would
wrap a grazing ray onto an overhead one). So a below-horizon sun cosine read the
tangent texel: **the reddest entry in the table**, ~9.7 % of red surviving
against ~0.004 % of blue.

Four passes read that value as "the sun's light at this sample": the sky-view
march (`sky_view.wgsl`), the cloud march (`cloud.wgsl`), the precipitation
highlight (`precip.wgsl`) and the sun disc itself (`sky.wgsl`, whose own comment
promised the disc "vanishes entirely once the ray to it passes through the
planet" — it did not). The multi-scatter stand-in made it worse by construction:
it is driven by `max(t_sun.rgb)`, which at night is always the red channel.

### The fix, and the ruling that shaped it

`atmos_horizon_visibility(r, mu, disc_cos)` in `atmosphere.wgsl` — Bruneton's
`GetTransmittanceToSun` in one function: a smoothstep on `mu` against the
sample's own local horizon cosine `-sqrt(1 - (bottom/r)²)`, with a half-width of
the body's angular radius projected onto the cosine axis. Applied in
`transmittance_to_sun` (the sky-view bake, which has its own sampler bindings and
cannot call the shared door) and through a new `atmos_sample_body_transmittance`
everywhere `mu` is a cosine toward the sun or the moon.
`atmos_sample_transmittance` stays the raw table read for the passes that need
extinction along a *view* ray.

**The brief prescribed a different fix** — fade the sun term to zero across
`sun.y ∈ [+0.02, −0.02]`, the house's existing `night_fade` pattern — and it was
measured before it was landed (the P23 law). It would have passed every red-band
assertion and **deleted civil twilight**: ±0.02 in sine is ±1.15°, so the sky
would snap to black about nine minutes after sunset. The per-sample local horizon
is the physically correct version of the same idea and keeps twilight exactly as
high as the lit air actually is — a parcel 60 km up sees the sun until it is 7.8°
below the ground's horizon, and *that difference is twilight*.

`golden_sky_night_horizon`'s second half is the arm that says which fix this is,
and it is built to fail for the prescribed one: at civil twilight the same band
must still be lit and still be warm.

### Measured

Facing the sun at 23:30 UTC, the band just below the horizon line:

| | band | R/G | ÷ zenith red |
|---|---|---|---|
| before | `[0.1174, 0.0281, 0.0318]` | **4.18** | **10.7×** |
| after | `[0.0068, 0.0068, 0.0068]` | 1.00 | 1.01× |
| civil twilight (sun.y −0.032), after | `[0.398, 0.223, 0.079]` | 1.79 | — |

**Day is untouched, and it is measured rather than argued:** `sky_noon`,
`sky_dawn` and `sky_dusk` all compare at **mean 0.000000 / max 0.000000** against
their committed frames. The visibility factor is exactly `1.0` for every sample
while the sun is up, and `x * 1.0` is bit-exact. 118 of 119 golden arms unchanged.

### The goldens

* **`sky_night_horizon` — NEW.** The arm `golden_sky_night` could not have: that
  one pitches 35° up, looks *away* from the sun and bounds only blue, so the one
  place a below-horizon sun can still be seen was outside its frame.
* **`clouds_night` — re-blessed.** Stated purpose: the committed frame depicted
  the defect. Top-half mean `[0.27609, 0.01323, 0.01039]`, **R/G 20.88** →
  `[0.00656, 0.00656, 0.00656]`, R/G 1.00. Image diff mean 0.0835, max 0.5064.
  Every structural arm still passes (stars survive the gaps: peak 0.365 against a
  starless 0.027; the clouds still occlude 42 px).
* **`sky_night` — re-blessed.** Same stated purpose, and it was **inside
  tolerance** (mean 0.0057, max 0.0500) — re-blessed anyway, because a committed
  frame that quietly depicts an engine that no longer exists is the thing the
  strict-mode printout exists to catch. `[0.00975, 0.00802, 0.00941]` (a
  red-biased black) → `[0.00672, 0.00672, 0.00672]` (a colourless one).

### Cause 2 — the ruling

`passes/sky_lut.rs:302` publishes `sun_color` unconditionally, and the brief
asked whether a Rust-side fade is also needed. **It is not**, and the evidence is
an enumeration rather than an opinion: every consumer of `atmos.sun_color` in the
shader tree — the sky-view march, the cloud march, the precipitation highlight
and the sun disc — now multiplies by a horizon-gated transmittance.
(`gi_probes.wgsl:185` reads `gi.sun_color`, a *different* uniform fed from the
ECS key light, which `ResolvedSky::key_light` already swaps to the moon at night.
Fading the atmosphere's sun colour would not have touched it, and darkening the
moonlit ambient was the failure mode the brief warned against.) A Rust-side fade
would also be a **global elevation gate wearing another hat** and would take
twilight with it.

`atmosphere_apply.wgsl`'s in-scatter is pinned to the horizon row of the sky-view
LUT — which is exactly the band `golden_sky_night_horizon` measures at 0.0068
grey. Distant terrain composites what that band contains, so it no longer
composites red at night; the fix is one LUT upstream of both.

The GI probes are the same story one indirection further on, and the brief asked
for them to be **verified rather than assumed**: the probe march's miss term is
`atmos_sample_skyview`, so `gi_sky_radiance_comes_from_the_atmosphere` gained a
midnight arm. It reads **blue/red 1.000** (colourless) against a floor of 0.6,
and the pre-wave LUT's midnight band measured 0.27, so any of that band reaching
the bounce fails it.

---

## Clause 2 — PIE carries terrain by path

The island's `.inf_terrain` is **549.9 MB** against a `MAX_FRAME_LEN` of
**268 435 456 B**, so `write_msg` refused the frame and Play on 50 km² of ground
produced one line in the status bar and no player at all.

**That figure is a correction, and it is the third time this number has gone
wrong the same way.** The wave brief scouted `342 742 272 B`, which is *wave I7's*
measurement; wave TER2b's detail band moved it and the I8a audit re-measured it
at 549.9 MB off `build.report.summary()` — a correction that memo already records
under its own item 3. This wave restated the stale number in new prose exactly as
I8a did, in nine source comments and in its first five commit messages, and
`a8060ca4` corrects the comments (a commit message cannot be corrected, so this
paragraph is where the record lives). **Nothing about the conclusion changes**:
both figures are more than double the cap.

**The cap did not move**, and the reasoning is worth keeping: an unbounded frame
length is the thing standing between a desynced pipe and a 4 GiB allocation, and
a protocol whose limits follow the author's content is not a protocol. What moved
is what the frame has to hold.

`ScenePayload` **v12** appends `terrain_paths: Vec<(Uuid, String)>` at the tail
(the envelope's own documented growth pattern, with `check_version`'s exact-match
refusal behind it). `TerrainRef::{Path, Bytes}` makes the choice the *resolver's*,
because only the caller knows whether the asset has a file: the editor always
does, the in-memory fixtures do not and keep the byte route. A terrain rides
exactly one of the two, and both consumers — the world builder's PCG resolver and
`TerrainContent::Payload` — prefer the path, so the two cannot page different
ground (the IB-1 defect's shape).

**PIE == shipping is not weakened.** The player opens the file through
`terrain_source_from_file`, which is the `--level` boot's door and the same shape
the shipped build takes (a cooked pack is read from disk, never streamed). The
wire was the odd path out for this one asset kind. What it costs, stated rather
than discovered later: a terrain edited between the payload being built and the
player opening it previews the newer bytes — the same window `--level` has always
had, bounded by process startup.

Measured on the CI-scale island fixture (`island_gate`):

```
PAYLOAD: 1 terrain path(s), 0 inline terrain(s), 1 biome set(s), 1 pcg(s), 1 mesh(es)
PAYLOAD FRAME: 6 183 889 B against a 268 435 456 B cap; the terrain it names is 7 043 328 B
```

The assertion is that the frame is **smaller than the terrain file it names**,
not merely under the cap: the cap alone is passed by a fixture of any size, while
the comparison is false for any payload carrying the bytes at any island size.
`inf-editor-core`'s own unit arm proves both halves (path present *and* bytes
absent) over a 4 MiB fixture and measures the two frames through `write_msg`
itself.

---

## Clause 3 — the starter levels ship a pawn

`inf_ecs::movement::camera_subject` returns `None` on a level with no
`player_controlled` character, and every host then keeps its own view. No level
the editor booted had one, so pressing Play showed an author their furniture from
an overhead orthographic camera with nothing that answered a key — while the
templates had been scaffolding the seventeen files of
`samples/starter-character` into `Content/Characters/` since SK1c and nothing
ever spawned them.

The three 3D starter levels (blank-3d, hybrid-2.5d, first-person) and the
editor's own boot document now place it, through
`SceneDoc::edit_create_character_with_guid` — the door the New Character wizard
and the island's hero take. The editor's boot content root is seeded with the
same seventeen files **whenever they are absent**, not only on a fresh root: an
editor run before this wave has three materials and no character, and "first run
only" would have left every such machine with a boot level of placeholder cubes.

**first-person got the smallest true fix rather than a character.** Its Player
already had a `CharacterController3D` — the *physics* half — and no
`CharacterMovement`, which is the half `movement_targets` and `camera_subject`
key on. So the template that exists to say "here is your player" shipped a level
with no player in it. The missing component, sized to the capsule that was
already there (`stand_half_height_m` 0.9 against `half_extents.y` 0.9), and no
skeleton: a first-person player has no visible body. Its README now says what is
true, including what is still missing (below).

**The content re-bless** (no schema move; scene stays v26):

| | |
|---|---|
| `templates/blank-3d/Blank.inf_lvl` | 768 → 1 557 B (3 → 4 entities) |
| `templates/hybrid-2.5d/Hybrid.inf_lvl` | 1 267 → 2 056 B (5 → 6) |
| `templates/first-person/FirstPerson.inf_lvl` | 1 072 → 1 501 B (the component) |

and the two 3D sidecars now declare the rig, the body and the machine as
dependencies where they declared nothing.

**Re-measured, because a doc table said otherwise:** a scaffolded blank-3d
project's cooked `content.inf_pack` goes **6 480 B → 33 625 B**. The old figure's
explanation ("the boot level spawns no character, so the closure never reaches
the body or its material") was exactly right and is now exactly wrong;
`template.rs`'s table carries the new number and the reason.

The gate asks through the runtime's own door — `camera_subject` must answer with
the level's character on all four documents — and the three that spawn the
committed character must name its **committed** asset guids, because a pawn wired
to minted ids draws as a placeholder cube in the project it ships in.

---

## Clause 4 — Play with no pawn is a question

A dialog, before anything starts: *"This level has no player-controlled
character"*, with **[Place Starter Character & Play]** and **[Play Overhead]**.

The first button performs the **real level edit** (`character_place_starter`, one
undo step, the clause-3 door) and then plays. **There is no PIE-only auto-spawn
and there must never be**: a preview that spawns a player the build does not is
precisely the divergence PIE == shipping exists to forbid. This way the level
that comes back is the level that ships.

The question is asked through `scene_player_pawn`, which *is*
`camera_subject` — so it predicts what the player process is about to decide
rather than guessing from components. A check that cannot be made does not block
Play: a failed diagnostic gets out of the way, and that is asserted rather than
assumed.

A dialog and not a toast, because the two answers are different acts: one edits
the document the build will be made from.

---

## Clause 5 — the editor can reach its levels

Four doors that named things they did not do.

* **`File ▸ Open Level…`** called `scene_open` with no path, and no path is the
  **quicksave fallback** — so the row that says "open a level" silently replaced
  the document with `quicksave.inf_lvl`. It gets a real file dialog, seeded like
  Save As. The test asserts the **argument**, because "it called open" was true
  before the fix too.
* **`Place Actor ▸ Starter Character`** — a Place Actor row, not a second wizard.
  `character_place_starter` writes **nothing**; `actor.newCharacter` mints six
  assets on every click, which is the wrong verb for "give this level a player".
* **Opening a project opens its boot level.** `project_boot_level` applies **the
  cook's own rule** (the lowest-GUID level, `levels.sort(); levels.first()`) to
  the same asset database, so what the editor opens is what a build would start
  in. A filename sort would have agreed by luck on a one-level project and
  disagreed the day a second arrived. **A dirty document is never replaced** —
  the offer becomes a status line, because opening a project is not consent to
  discard an edit.
* **A missing player names its remedy.** `find_player_bin` returns a path whether
  or not anything is at it, so Play with no `inf-player.exe` reported the
  operating system's "cannot find the file specified" beside a path. It now says
  `cargo build -p inf-player`.

---

## The pins that fired (and what that says about them)

Five arms went red on the first full battery, and **every one of them was right**
— which is the report worth making about a wave that changes committed images and
committed content:

* `phase18_gate::the_golden_inventory_is_exactly_the_committed_set` — the golden
  set is pinned **by name**, not by count, so the new frame had to be named.
* the three `phase2{6,7,8}_gate` count-and-CONTENT digest pins — 59 → 60 and
  `1db3dd0b…` → `e4ef4624…`. The count alone would have accepted a swap; the
  digest is what makes a re-bless visible. This wave hits **both branches of that
  pin's own rule at once** (one frame added, two re-blessed for the look), which
  is why the note beside the digest says which is which, and why the two acts are
  in different commits.
* `inf-scene::decodes_the_committed_hybrid_template` — counted five entities in a
  template that now has six. The count moved and the arm gained the assertion a
  count cannot make: that one of those entities is a **pawn**.

Nothing was silenced. The five fixes are in `158ca5d4`, on their own, after the
work that earned them.

Two more arrived from the gates the battery does not run: `cargo doc` went
373 → **374** on an intra-doc link that pointed *up* the ring order
(`inf-runtime` → `inf-player`), and clippy's `empty_line_after_doc_comments`
caught the blank line a scripted insertion left between two halves of the digest
pin's doc block — which would have silently split it into two comments, one of
them documenting nothing. Both fixed (`544e2711`, `3f8a0cc4`); the rustdoc total
is back to the base's 373.

## Findings this wave made and did NOT fix

* **The cook's vgeom advisory is over-broad for a skinned mesh.** Cooking a
  scaffolded blank-3d project prints *"`Starter_Body` has 1 498 triangles, below
  the virtualized-geometry threshold of 2 048 … the shipped build renders it as a
  PLACEHOLDER CUBE"*. For a `SkeletalMesh` binding that is **false on every
  native target**: `SkinnedRegistry::from_pack` loads the `.inf_mesh` payload
  directly and never consults `.inf_vmesh`. It *is* true on wasm32, which carries
  no `inf-mesh`. The advisory should exempt meshes reached only through a
  skeletal binding, and the message should name the platform where the
  consequence is real.
* **`level_dependencies` does not enumerate `ActorClass`.** The new starter
  levels declare the rig, the body and the machine and **not** the controller
  `.inf_act` they bind — the same silence the 2D platformer's sidecar has carried
  since I1 (`dependencies = []` on a level whose player is a Coyote class). The
  cook finds blueprints by another walk, so nothing is broken today; the delete
  guard and the Content Drawer's reference view are the ones that cannot see it.
  Fixing it re-blesses every committed level sidecar, which is a wave of its own.
* **Night clouds are unlit, not merely un-red.** The cloud march's only light is
  the sun; moonlight scattering is not modelled anywhere in this engine. After
  this wave a midnight deck is black and reads as holes in the starfield. That is
  the honest depiction of the model we have, and a moon term in the cloud and
  sky-view marches is the feature that would change it.

## Carried

* **`ExposureMode::Auto`'s `min_luminance` amplifier** (`settings.rs:113–123`) —
  named by the brief, untouched here. Now that the night sky is genuinely dark,
  auto-exposure has less to work with, which makes this more likely to be visible
  rather than less.
* ~~GI probe night tint~~ — **closed**, with an arm: see clause 1 above.
* **A per-level camera view mode.** `ViewMode` is camera-side only and never
  crosses the sim wire (`inf_ecs::camera`'s ruling 4), so the first-person
  template plays in the third-person locomotion camera. Its README says so.
* **The society empties at 18:00** → VEN1's schedule work.
* **The island build on a dev machine is stale** (v25 / 17-entity, Aug 21;
  predates the SK1c hero and VEH1a). `cargo run -p inf-cli -- island build
  --recipe samples/island/island.toml` before any island verification. This
  wave's island evidence is the **CI-scale fixture** that `island_gate` builds
  from its own recipe, which is the shipped path at a smaller size.
* **Vehicle enter/exit animations** → VEH2.

## House gates at head

| | base `bf1954e6` | this tree |
|---|---|---|
| battery (`-j 3`, `--no-fail-fast`) | 356 / 6 604 / 0 / 19 | **356 / 6 607 / 0 / 19**, exit 0 |
| goldens, `INF_GOLDEN_STRICT=1` | 59 files, 118 arms | **60 files, 119 arms**, 200 s, none blessed by the run |
| rustdoc after `cargo clean --doc` | 373 over 30 crates | **373 over 30 crates** (403 `^warning` lines − 30 summaries, cross-checked against the summaries' own sum); headroom 77 against the 450 ceiling |
| clippy `-D warnings`, run LAST | 0 | **0** |
| `cargo fmt --all --check` | clean | clean |
| frontend | — | typecheck + eslint clean, **85 files / 775 tests** (was 84 / 764) |

The battery is **+3 arms, exactly this wave's new `#[test]` count**:
`golden_sky_night_horizon`, `a_terrain_with_a_file_rides_as_a_path_and_the_frame_stays_small`
and `every_starter_level_has_a_player_controlled_character`. The rustdoc total
went to 374 first — the wave's own unresolved intra-doc link, `inf-runtime`
pointing *up* at `inf-player`, fixed in `544e2711`.

## What this wave did NOT verify

* **The real island's Play button was not pressed.** No *fresh* `.inf_terrain`
  for the shipped island was built here (it is not committed), so neither the
  549.9 MB nor the 342.7 MB figure was measured by this wave — see the correction
  under clause 2. What is measured here is stronger than one more file size would
  be: the payload no longer contains terrain bytes **at any size**, which the unit
  arm shows by building the same document both ways (the byte route grows by the
  whole file, the path route by a filename) and `island_gate` shows by the frame
  being smaller than the terrain it names.
  **Closed by the audit below**, which rebuilt the island and measured both.
* **Every GPU claim in this memo is a headless-render measurement**, like every
  golden in this repository. Nobody looked at the sky in the editor.

---

# The adversarial audit (2026-08-31, over `bf1954e6..b41ab78e`)

Clause 1, clause 2's envelope and clause 5's rulings stand as written and are
re-derived below. Clause 3 **shipped a pawn with no ground under it**, which is
the finding this audit exists to have caught, and clause 3's seeding wrote files
into projects it does not own.

## THE ISLAND CLOSURE — both numbers, measured

The wave could not press the island's Play button because it had no island. This
audit built one: `inf island build --recipe samples/island/island.toml`, **43.3 s**
on a warm cache, all four standing advisories and nothing blocking. (The invocation
is `cargo run -p inf-cli --release --bin inf -- island build …` — there is no `inf`
*package*; the brief's `-p inf` fails with *"package(s) `inf` not found"*.)

| | |
|---|---|
| `VancouverIsland.inf_terrain` | **549 879 456 B** |
| `MAX_FRAME_LEN` | 268 435 456 B |
| ratio | **2.05×** — the frame refusal was real |
| the whole built project | 595 333 570 B |

**549.9 MB is right and 342.7 MB was wave I7's**, settled by measurement rather
than by a fourth restatement. `samples/island/README.md` (twice) and
`samples/island-fixture/README.md` still carried I7's figure after `a8060ca4`
corrected the nine source comments; they carry the measured one now, which is the
place a reader actually quotes it from.

And the payload the editor really builds for it — the real `build_scene_payload`
over the built project's own asset sidecars, the real `TerrainRef::Path`, the
real `write_msg`:

```
TERRAIN ASSETS: 1
  bbb9b175-… -> …/island-build/project/Content/VancouverIsland.inf_terrain (549 879 456 B)
PAYLOAD: 1 terrain path(s), 0 inline terrain(s)
REAL ISLAND PAYLOAD FRAME: 6 250 534 B against a 268 435 456 B cap;
                           the terrain it names is 549 879 456 B
```

**6 250 534 B is 2.3 % of the cap and 1.1 % of the ground it names.** Carried
inline the same frame is ~556 MB, which is what `write_msg` refused. The
measurement ran through a throwaway `inf-player` integration test and is not
committed: an arm that needs an uncommitted 550 MB artifact skips in CI, and a
skipping arm is a vacuous one.

## Findings

### 1 — THE PAWN HAD NO GROUND (fixed)

`blank3d_scene`, `hybrid_scene`, `firstperson_scene` and the editor's boot
document all put their character on a ground plane carrying a `MeshRef`, a
`Material` and **nothing physical**. `inf_physics::d3::ecs`'s sync walks the
entities that carry a body **or** a collider and `continue`s on the rest
(`ecs.rs:800`), so those planes reached the solver as nothing at all — which cost
nothing while the levels held only furniture, and became the whole story the
moment this wave put a *gravity-driven* pawn on them.

Measured, one second of Simulate at 60 Hz, through `SimSession` — the editor's own
door — on all four documents:

| | falls in 1 s |
|---|---|
| the plane as the wave shipped it | **4.9868 m**, still accelerating toward the 53 m/s terminal velocity |
| with `samples::ground_slab` under it | **−0.0201 m** (a 2 cm *rise*: the kinematic controller's ground snap) |

`templates/first-person/README.md`, written by this wave, says *"Press Play and
WASD moves the Player"*. What it moved was a body in free fall, and the wave's own
gate could not see it: `every_starter_level_has_a_player_controlled_character`
asks `camera_subject` and stops, which is exactly what a pawn with nothing under
it passes.

The fix is one component per ground entity — a box `Collider3D` with **no**
`RigidBody3D`, which is this engine's documented way to say *static world*, and an
offset that puts the slab's top face **exactly on the visual plane** (a unit
`Primitive::Plane` spans ±0.5, so a ground scaled by 20 is ±10 m at `y = 0`).
Second half of the same finding: **first-person's Player was authored at
`y = 1.0`** while a character's transform is its capsule centre and
`feet_offset_m` is `half_extents.y + radius` = 1.2 — so its feet sat 20 cm under
the floor the moment there was a floor. Now 1.2, which is what
`edit_create_character_with_guid` computes for the other three.

The new arm is `every_starter_level_gives_its_pawn_something_to_stand_on`, and its
**control** is what makes it an assertion about the ground rather than about
gravity: the same document with every non-pawn collider removed must drop the
pawn, measured in the same run, plus an anti-vacuity assertion that the document
authors gravity at all.

**Content re-bless, stated purpose: the committed levels had no ground under
their new pawn.** No schema move (scene stays v26), no entity added, no dependency
added — 94 B of `Collider3D` per level:

| | |
|---|---|
| `templates/blank-3d/Blank.inf_lvl` | 1 557 → **1 651 B** (4 entities, unchanged) |
| `templates/hybrid-2.5d/Hybrid.inf_lvl` | 2 056 → **2 150 B** (6, unchanged) |
| `templates/first-person/FirstPerson.inf_lvl` | 1 501 → **1 595 B** (4, unchanged; includes the 1.0 → 1.2 spawn) |

and `template.rs`'s doc table re-measured a second time by the same door the wave
used (`inf new` → `inf cook`): **33 625 → 33 641 B**. The 16 B is the collider.

### 2 — THE SEEDER WROTE INTO OTHER PEOPLE'S PROJECTS (fixed)

`seed_starter_content`'s doc says it seeds *"the editor's own boot content root —
`<app_data>/Content`"*. `build_inner` runs it on **every re-root as well**, and
`AssetState::reroot` is what opening a project calls. Before this wave the
`if !proj.db().is_empty() { return; }` guard made that harmless; the wave moved
the character branch *above* the guard and keyed it on absence, so opening any
project without the starter skeleton copied seventeen files into its
`Content/Characters/`. That is every 2D project (`Platformer2d` scaffolds no
character) and every project whose author deleted theirs — which would then come
back on each open.

`build_inner` takes a `boot: bool` now: true from `init_assets_on_boot`, false
from `reroot`. The materials branch is untouched (it was already fresh-root only).

### 3 — THE HYBRID DECODER'S "PAWN" WAS NOT A PAWN (fixed)

`158ca5d4` says the arm *"gained the assertion the count cannot make — that one of
those entities is a pawn"*. What it asserted was `skeletal_mesh.is_some() &&
character_movement.is_some()`. `camera_subject` filters on
`CharacterMovement::player_controlled`, so a character with the flag **false**
passes that arm while the level goes straight back to the overhead camera — the
whole defect the count moved for. The arm reads the flag now.

### 4 — THE PLACE ACTOR ROW'S THREE ARMS COULD NOT SEE IT UNWIRED (fixed)

`shellCommands.test.ts`'s new arms check that the row is in `MENU_BAR`, that it is
*not* in `ACTOR_PLACE_KINDS`, and that `stubHint` has nothing for it. Deleting the
`setCommandHandler` call passes all three, and the row then dispatches into the
unhandled hook and toasts *"is not implemented yet"* — which that file's own header
says is the failure it exists to catch. **Mutation-verified**: renaming the handler
id leaves all three green and fails only the arm this audit added to
`objectCommands.test.ts`, which is the suite that actually calls
`bootstrapShellCommands`.

### 5 — THE ISLAND READMEs (fixed)

Above, under the closure.

## What was re-derived and stands

* **The horizon gate is Bruneton's, and it is right.** `sin_h = bottom/r`,
  `cos_h = -sqrt(1 - sin_h²)`, `smoothstep(-w, w, mu - cos_h)` with
  `w = sin_h·sin(angular radius)` — `GetTransmittanceToSun` with `sin(θ)` where the
  reference has `θ`, which is the same number to five decimals at 0.27°. Per-sample
  `r`, so twilight survives; **no double-gate** with the ground-bounce block
  (`n_dot_l` and the visibility reach zero at the same `mu`, and their product is
  zero either way) and none with the cloud paths (`cloud_shadow_factor` and
  `cloud_sun_transmittance` return **1.0** below the horizon — they are occlusion
  terms, not light terms, and 1.0 × 0 is 0). Every call site passes a radius in km
  and `atmos.{sun,moon}_dir.w` really is `p.{sun,moon}_disc_cos()` at the only
  place the uniform is filled (`sky_lut.rs:302`); the other two `sun_dir` sites are
  cache keys. `atmosphere_apply.wgsl` samples the sky-view LUT, so the fix reaches
  aerial perspective one table upstream, as claimed.
* **The twilight arm falsifies the brief's prescription.** Its window is
  `sun.y ∈ (−0.09, −0.03)`, entirely past the ±0.02 fade, and it demands the band
  be 4× the midnight red and warm. A global fade renders it black and fails.
* **`ScenePayload` v12 is append-only and honest in both hosts.** One field at the
  tail, the wire-order pin extended to 21 fields with a string length unique among
  the tail (bincode encodes `String` exactly as `Vec<u8>`, so that uniqueness is
  the pin), `check_version` exact equality in `build_world_from_payload`,
  `Cargo.lock` unmoved, and the **scene schema did not move** — v26 in all three
  sidecars, and `crates/inf-scene`'s only diff is inside `mod tests`.
* **`TerrainRef::Path` cannot carry a relative path.** It is `AssetEntry::path`,
  documented absolute and canonicalized by `AssetDb::normalize`, kind-checked to
  `AssetKind::Terrain`, and the player opens it with `terrain_source_from_file` —
  byte-for-byte the door `TerrainContent::Dir` (the `--level` boot) uses, with no
  editor-only decode anywhere. Both consumers prefer the path, so the IB-1 shape
  cannot recur.
* **The boot-level rule is the cook's.** `cook.rs:1283` declares
  `levels: Vec<AssetId>` and `cook.rs:1448` does `levels.sort(); levels.first()`;
  `project_boot_level` sorts the same `AssetId`s off `by_kind(Level)`. The
  dirty-document guard is a real branch with a real arm (`openLevel.test.ts`
  asserts `scene_open` is **not called** and the status line says *unsaved*).

## House gates on the closing tree

| | wave `b41ab78e` | audit `4817e67b`+ |
|---|---|---|
| battery (`-j 3`, `--no-fail-fast`, `INF_GOLDEN_STRICT=1`) | 356 / 6 607 / 0 / 19 | **356 / 6 608 / 0 / 19**, exit 0 |
| goldens, strict | 60 files, 119 arms | **60 files, 119 arms**, all green, **none blessed by the run** |
| clippy `-D warnings`, `CARGO_INCREMENTAL=0`, run LAST | 0 | **0**, exit 0 |
| rustdoc after `cargo clean --doc` | 373 over 30 crates | **373 over 30 crates** (403 `^warning` lines − 30 summaries, cross-checked against the summaries' own sum of 373); headroom 77 |
| `cargo fmt --all --check` | clean | clean |
| frontend | 85 files / 775 tests | **85 files / 776 tests**, typecheck + eslint clean |
| `Cargo.lock` | unmoved | unmoved |

**+1 battery arm and +1 frontend arm, which is exactly this audit's `#[test]` and
`it()` diff.** The three template `.inf_lvl`s moved and no golden did: the
`phase2{6,7,8}_gate` count-and-CONTENT digests and `phase18_gate`'s inventory all
pass unchanged, which is the check that a content re-bless did not reach the
images.

## Carried (found here, not fixed here)

* **`seed_starter_character` checks only the skeleton guid.** A boot root holding
  the rig but missing one of the other sixteen files is not re-seeded. Checking all
  seventeen would also mean overwriting sixteen files an author may have edited, so
  the narrow check is defensible and the gap is recorded rather than closed.
* **The cook's `levels` set is post-`Skipped`.** A level that fails to cook is not
  in it, so a project whose lowest-GUID level does not cook would have the editor
  open one level and the build boot another. Pathological, and no gate names it.
* **`inf-ecs/src/item.rs:25`** prices a hypothetical inventory field at
  `SCENE_PAYLOAD_VERSION` **12**, which this wave has now spent on terrain paths
  (and at scene **v26**, which is already the current one). Prose drift on a
  feature nobody is building; it would mislead the person who does.
* The wave's own three: the cook's vgeom advisory over-claiming a placeholder cube
  for a skeletal mesh, `level_dependencies` not enumerating `ActorClass`, and night
  clouds now being unlit rather than merely un-red.
