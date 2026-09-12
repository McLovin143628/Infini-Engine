# `tools/demo` — end a wave with something a person can judge

A wave that ends in a green battery has proved that the tests agree with the
code. It has not proved that the editor opens, that the Play button plays, or
that the character walks — and in wave FIX1 the author found all three false at
a head whose battery was green (`7 019 passed, 0 failed`).

So every wave from FIX1 onward ends here: the real editor, built the way it
ships, launched, driven through its own Play button, and photographed.

## Run it

```powershell
pwsh -NoProfile -File tools/demo/demo.ps1
```

That builds the editor (`npx tauri build --no-bundle`), launches
`target/release/inf-studio.exe` **from its own directory** — which is how the
boot ladder discovers the showcase island by walking up from the executable —
presses Play, waits for `inf-player.exe`, holds `W` on the running game, and
writes three PNGs plus a CSV of where the hero was.

Useful switches:

| | |
|---|---|
| `-SkipBuild` | use whatever is already in `target/release` |
| `-OutDir <path>` | where the PNGs and the CSV go (default: a timestamped folder under the system temp dir) |
| `-KeepOpen` | leave the editor running at the end |
| `-Port <n>` | the WebView2 debug port (default 9222) |
| `-PlayMode window` | **the default since the CHAR1b.2 audit** — drive "Play in New Window" instead of the embedded viewport. Needs the CDP path, because a menu item has no coordinate to fall back to; `-PlayMode embedded` is the other way and is what the roadmap calls the preview |

## What it produces

```
01-editor.png     the editor as it booted, on the showcase island
02-pie-a.png      the running game
03-pie-b.png      the same, two seconds later, with W held throughout
A1..A6-*.png      the SHOOTOUT (wave WPN2e, section 5e, last): the brass, an
                  officer aiming at the hero, the hero under fire, the hero
                  returning it, cover under fire, and a blind shot. Every one is
                  triggered on a column the sim writes -- `engaged` and
                  `incoming` are the two the wave appended for exactly this --
                  and every one says so in the log when it does not fire
hero.csv          t,frame,x,y,z,mode,speed,camera_clip,aim_yaw,head_yaw,head_pitch,
                  state,foot_mm,boom_m,body_fade,whisker_steer,camera_holder,
                  cover_class,cover_side,cover_peek,rounds,last_hit_m,equipped,
                  recoil_mm,aim_recoil_deg,spread_deg,ads,casings,tail,
                  class,attach,lock,engaged,incoming,heat,on_scene,responder_m,
                  tyre_temp_fl,tyre_temp_fr,tyre_temp_rl,tyre_temp_rr,surface,
                  slip_ratio,slip_lat,mu,rpm,clutch,boost,cut,
                  car_health,engine_scale,flats,panes_broken,parts_shed
                  — FIFTY-FOUR columns, four rows a second, and NO header line:
                  every consumer filters on `^[0-9]`, and the `#` lines are the
                  driver's own notes. Columns are only ever APPENDED, so every
                  index a script already reads keeps its meaning: 14-17 are wave
                  CHAR1c's (boom / fade / steer / holder), 18-20 are COV1's,
                  21-23 are WPN2a's (rounds in flight, the last impact's flight
                  distance, and the equipped id), 38-45 are VEH3a's (four tyre
                  temperatures in Celsius, the surface the most wheels are on,
                  the driven axle's slip ratio and lateral slip, and the µ the
                  contact is worth — all `0`/`-` when nobody is driving, and
                  `air`/`0.000` when the car has NO WHEEL ON THE GROUND, which
                  is a different thing from `asphalt` and used to read as one),
                  46-49 are VEH3b's (the crank's own rpm, the clutch's
                  engagement in `[0, 1]`, the turbo's boost in `[0, 1]` and `1`
                  while the limiter is cutting fuel — the four a launch flare, a
                  downshift blip and a limiter bounce have to be triggered on,
                  because none of them is visible in `speed`), 50-54 are
                  VEH3c's (the hull's remaining percent, the engine's remaining
                  percent, how many tyres are flat, how many panes have gone and
                  how many parts have left the car — all `0` when nobody is
                  driving, which is the same convention the eight tyre columns
                  before them use, and the five a shed bumper, a shattered
                  window, a flat and a burning car have to be triggered on),
                  24-27 are WPN2b's (the
                  hold-point spring in millimetres, the aim's own recoil offset
                  in degrees, the whole cone the next round would leave through,
                  and the aim-down-sights blend), 28-29 are WPN2c's (how many
                  shell casings exist right now, and which TAIL the last loud
                  shot chose — `indoor`, `outdoor`, or `-` before anything has
                  been fired) and 30-32 are WPN2d's:

                    class   what KIND of gun is in the hand — `pistol`, `smg`,
                            `ar`, `dmr`, `sniper`, `shotgun`, `launcher` — or
                            `-`. Column 23 carries the row NAME; a leg that
                            rotates one weapon per class waits on this.
                    attach  `n@x.xx`: how many attachments are bolted on, and
                            the fold's own loudness multiplier. `0@1.00` is a
                            bare rail; `1@0.35` is a suppressor fitted and
                            WORKING, which is the difference between a table row
                            and an effect.
                    lock    how far into a lock-on the launcher is, `[0, 1]`,
                            with a `+` once it has completed. A lock takes
                            1.2-1.6 s to acquire and releases the instant the
                            cone loses it, so a frame of the indicator has to be
                            triggered on it.

                  and 33-37 are WPN2e's and its audit's:

                    engaged  how many responding units are pointing a weapon at
                             somebody RIGHT NOW. The police arrive over tens of
                             seconds and the aim is a ray-gated decision that can
                             go away between two screenshots, so a frame of an
                             officer aiming is triggered on this and not on a
                             wall clock.
                    incoming rounds in the air the hero did NOT fire — "somebody
                             is shooting at me", as a number. Zero on every level
                             nobody has fired at the hero on.
                    heat     the hero's own criminal heat (`CrimeRes`), which is
                             what `Response::for_heat` reads and therefore the
                             whole rung ladder: 0 cold, 1-2 a patrol, 3-5 two
                             cars, 6+ SWAT. It bleeds off one point per
                             `HEAT_DECAY_STEPS` (15 s) since the last sighting.
                    on_scene units standing at an incident right now.
                    responder_m how far the NEAREST responding crew is from the
                             hero, or -1 before one exists. `ENGAGE_RANGE_M` is
                             35 m and a unit stops within `ON_SCENE_M` of an
                             incident OR where its road runs out, which on a
                             street can be a long way further — so "on scene"
                             and "near you" are two different facts.

                  The last two are the WPN2e AUDIT's, and they exist because two
                  waves failed to diagnose this loop's own shootout leg with the
                  columns in front of them: `engaged 0` is the same number for
                  "nobody heard the shot", "the file went cold before the car
                  arrived", "the car never arrived" and "the officer arrived and
                  could not see you", and those are four different bugs in four
                  different crates. `heat` and `on_scene` are the two that tell
                  them apart.

                  The numbers above are ONE-based, which is how a person counts
                  columns; `demo.ps1`'s own predicates index `$c[..]` ZERO-based,
                  so `boom_m` is `$c[13]`, `ads` is `$c[26]`, `tail` is `$c[28]`,
                  `lock` is `$c[31]`, `engaged` is `$c[32]`, `incoming` is
                  `$c[33]`, `heat` is `$c[34]`, `on_scene` is `$c[35]`,
                  `responder_m` is `$c[36]`, `car_health` is `$c[49]`,
                  `engine_scale` is `$c[50]`, `flats` is `$c[51]`,
                  `panes_broken` is `$c[52]` and `parts_shed` is `$c[53]`.
demo.log          every step the driver took, with timings
```

and it prints the hero's first and last position and the distance between them.
**That distance is the point.** Two screenshots cannot tell a character that
walked from a camera that drifted; the CSV can.

`hero.csv` also carries the player's own `#` notes — its keyboard-focus report
and every focus handover, with the window and process that took it. That is not
decoration: the editor's Output Log is behind the game's window, so a scripted
session has nowhere else to read the player's stderr, and this file is how wave
FIX1 found out that its own synthetic click was what pushed the new-window
session out of the foreground.

## How it presses Play

By name, over the Chrome DevTools Protocol: `INF_WEBVIEW_DEBUG_PORT` makes the
editor's WebView2 listen, `play.mjs` finds `[data-tour="play-cluster"] button`
and clicks it. If node is not on the PATH or the port never opens, the driver
falls back to clicking the button's screen coordinate on a maximized 1080p
window and says so in the log.

The same port drives the other CDP scripts beside it: `place.mjs` adds the
committed female body to the open document, `portrait.mjs` frames the hero's
face, `undo.mjs` puts both back before Play, and `car.mjs` (wave VEH3c) frames a
CAR -- it finds a part named `door_fl` or `door_l` in the open document, moves
its PARENT onto the editor camera's own opening view ray, and exits 4 if the
level's vehicles have no doors, so a level that lost its parts is a refusal
rather than a photograph of a box. None of the four writes to disk: the document
is left dirty and nothing presses Ctrl+S.

Setting `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS` yourself does **not** work and
the reason is in `inf_studio_lib::debuggable_context`: WebView2 reads that
variable only when the embedder passes no arguments of its own, and Tauri always
passes some.

## `-TuneVehicle "doors_open=1"` — the one name that is not a tunable

`-TuneVehicle` is a `;`-separated `name=value` list that reaches
`VehicleClass::set` and `Vehicle::tune`, the same by-name doors an authored
catalogue row uses. **`doors_open` is the exception.** It is a DIRECTIVE: it is
filtered out of the pairs before either tuner sees them — so it is never reported
refused — and it reaches `RuntimeSim::open_vehicle_doors`, which swings every
hinged part of every chassis onto its motor (`=0` shuts them again).

It exists because **the shipped input map has no key for a car door.** An input
path to one is wave VEH3d's. Without this directive the joints wave VEH3c built —
a door past `LATCH_POP_FRAC` of its mount is a rapier body on a real revolute
with a motor and its contacts against its own chassis off — could be *measured*
and never *seen*. Nothing else about the door is faked: the hinge, its limit, the
impulse the joint carries and the tear are the shipping path.

Two frames in section 6z3 come from it, and the second needs it:

```
105-veh3c-bumper-run-over.png   a car back over what it just shed. NOT triggered
                                — there is no column for "a wheel is on a
                                bumper" — so it is a manoeuvre (reverse, then
                                return at PART throttle) and a photograph.
                                `veh3c_gate::a_shed_bumper_in_the_road_is_run_over`
                                is the measurement: a cast over the panel stops
                                18.5 mm short of the road beside it and a car
                                crossing at walking pace rides 2.7 mm higher and
                                keeps going at 3.00 m/s. At FULL throttle it
                                reads −1.3 mm, because a car crosses a 70 mm
                                panel in a fifth of a fixed step — which is a
                                fact about the sample rate, not about the panel.
106-veh3c-door-torn-off.png     triggered on `parts_shed` rising after the doors
                                are on their hinges. A hinge carries only what
                                the PART brings (a 22 kg door at 30 km/h delivers
                                183 N·s), so the door has to MEET something;
                                `an_open_door_is_torn_off_by_a_lamp_post`
                                measures 515 N·s and 14 → 13 parts against 2 N·s
                                and 14 → 14 in open air.
```

A session that did not ask for `doors_open` says so in the log and cites the arm,
rather than photographing a shut car and calling it a hinge.

## Before you run it

The driver refuses to build while an `inf-studio` or `inf-player` is running —
the island's pack is memory-mapped and a build that tries to replace a running
executable fails as a sharing violation, which surfaces as `LNK1104` and looks
like a disk problem. Close the editor first, or pass `-SkipBuild`.

## `island-refresh.ps1` — refresh the island WITHOUT reverting its hero

**Never run `inf island build` on its own.** It is safe for the terrain, the
roads, the biomes and the level, and destructive to the character:
`samples/island/island.toml`'s `content` list copies the committed starter
character into the project BY NAME, and the island's hero is not the committed
starter — it is a MetaHuman rebound at those GUIDs by
`inf-import --rebind-character`, local-only content this repository does not
carry and CI never sees. A plain build puts the **161-joint wizard rig and its
five-track clips** back over the **342-joint MetaHuman body and its 150-track
ALS clips**, and reddens four gates that have nothing to do with the wave that
ran it:

| gate | after a plain `inf island build` | after `island-refresh.ps1` |
|---|---|---|
| `char1a3_gate` | 11 / **11 failed** | **14 / 0** |
| `char1b_gate` | 21 / **11 failed** | **32 / 0** |
| `cov1_gate` | 13 / **3 failed** | **16 / 0** |
| `outfit1_gate` | 14 / **3 failed** | **17 / 0** |

The first symptom is `hero's rig has 161 joints … right: 342`, which reads like
a character regression and is a file copy.

```
pwsh tools/demo/island-refresh.ps1                # build, then restore
pwsh tools/demo/island-refresh.ps1 -SkipBuild     # restore only
```

It runs `inf island build` and then the three imports **in the order that makes
them work** — a clip's coupling to a skeleton is POSITIONAL, so the mannequin
and its 164 ALS clips go to the starter GUIDs first, and the MetaHuman body swap
second, where `retarget_committed_clips` re-retargets by NAME (**150 of 161
tracks kept, 11 dropped**, every one an IK or attachment helper). Run the body
swap alone and the retarget runs on the wizard's five-track clips and keeps
**four**. The clips go under `Content/UE/Mannequins/` because that is where the
original import put them, and a second copy under `Content/UE/` makes every ALS
name AMBIGUOUS — 74 of 74 sequences come back "unbound" with all of them on disk.

## It is a gate, not a screenshot service (audit FIX1)

`demo.ps1` **exits non-zero (7)** when the hero moved less than `-MinMetres`
(default **5 m**), when the player wrote no positions, or when there is no
`hero.csv` at all. The first version printed `HERO MOVED` and exited 0 whatever
the number was — including the runs the wave later found had moved **0.000 m**,
which were caught by a person reading a log rather than by the gate whose whole
purpose is to catch them.

Five metres is the bound because a held `W` buys twelve in the seconds this
script allows, and because a settle, a slide or a camera drift is centimetres.

It also echoes the player's own `keyboard focus …` line beside the number, so a
session that moved is read next to the reason it could:

```
player: keyboard focus hwnd=0x250406 parent=0x0 fg=0x250406 focus 0x250406 ->
        0x250406 attached=false landed=true
```

`parent=0x0` there means the player's window was still **top-level** when it
asked — the editor had not reparented it yet — so the embedded child branch of
`win_host::take_keyboard_focus` did not run. In 28 recorded sessions across two
machines' worth of runs it has never run, and `attached` has never once been
`true`. Read that line before believing a story about which branch made a
session work.

## Exit codes

| code | meaning |
|---|---|
| 0 | the hero moved, the frames are in `-OutDir` |
| 2 | an `inf-studio` or `inf-player` was already running |
| 3 | a build failed, or there is no editor/player to run |
| 4 | the editor exited while we waited |
| 5 | no `inf-player` appeared within `-PieWaitS` |
| 6 | `-PlayMode window` without node on the PATH (a menu item has no coordinate) |
| 7 | **Play did not play**: the hero moved less than `-MinMetres` |
