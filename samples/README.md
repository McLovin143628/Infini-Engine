# Infini Engine samples

These are dogfood projects that double as integration-test fixtures — each is generated
deterministically by `inf_editor_core::samples` and its committed bytes are drift-checked in CI
(`cargo test -p inf-editor-core samples`; regenerate with `INF_BLESS_SAMPLES=1`). They are the
phase-gate scenes: every one is exercised by a runtime test that asserts real behavior.

## How to open / cook / run a sample

Most samples are a folder of committed `.inf_*` assets (a bincode payload + a git-diffable TOML
sidecar each). To try one:

```sh
# Open in the editor: File ▸ Open, point at the sample folder (or its Content dir).

# Cook a shippable pack and run it headless (the make→cook→play loop):
inf cook  --project samples/<name> --out /tmp/<name>-build
inf-player --pack /tmp/<name>-build --headless --run-frames 300 --assert-exit

# Or export a runnable, double-clickable folder:
inf export --project samples/<name> --out /tmp/<name>-dist
```

(The `platformer-2d` sample is exactly what CI's cook-and-run smoke test ships — see
`.github/workflows/ci.yml`.)

## Index

| Sample | Gate phase | Demonstrates |
|--------|-----------|--------------|
| [`platformer-2d`](platformer-2d/) | P8 (2D) | A Blueprint **coyote-time jump** on a tilemap; the 2D physics + interpreter path. |
| [`terrain-demo`](terrain-demo/) | P10 (Terrain & PCG) | A sculpted + splat-painted **heightfield** with a **PCG scatter** volume. |
| [`character-demo`](character-demo/) | P11 (Animation) | An idle/run/jump **state-machine character** driven by a Blueprint across terrain. |
| [`physics-playground`](physics-playground/) | P12 (Physics) | Box stacks, motors, ropes, CCD, ragdolls, collision layers + spatial audio. |
| [`vgeom-demo`](vgeom-demo/) | P13 (Virtual geometry) | A 10M+ source-triangle scene via one instanced dense mesh + meshlet culling. |
| [`mods/`](mods/) | P14.5 (Modding) | A sandboxed **WASM mod** (`spinner`) — safe, no-recompile end-user extensibility. |
| [`streamed-terrain`](streamed-terrain/) | P16.3 (Terrain streaming) | A 256 m heightfield living entirely in a `.inf_terrain`, paging by camera — and the sim/render want split that keeps the camera out of the fixed step. |
| [`partitioned-world`](partitioned-world/) | P16.5 (World partition) | A 4×4 grid of 128 m cells whose entities spawn/despawn around a `StreamingSource`, plus the persistent cell. |
| [`phase16-world`](phase16-world/) | **P16 (the phase gate)** | The composed world: a **wizard-imported** streamed terrain over 8.2 km, a partition on top of it, a **second inline terrain**, and the residency/step budgets. |
| [`phase18-scatter`](phase18-scatter/) | **P18 (the phase gate)** | The composed frame: standing **meshlet slabs** (sharing vgeom-demo's mesh by GUID) under two-pass HZB occlusion + a bound streaming budget, **GI v2** on a running clock, and **102 400 GPU-scattered instances** with LOD fade. |
| [`phase19-town`](phase19-town/) | **P19 (the phase gate)** | The composed town: a **biome-painted** terrain, a **spline road** with a solid grammar fence, twelve streamed street lamps on a 128 m partition, and **seven fully enterable, furnished, three-storey buildings — one per archetype** (office, apartment, industrial, house, estate, hotel, shop). |
| [`phase20-coastal`](phase20-coastal/) | **P20 (the phase gate)** | The composed coast: an **ocean** with swell, a **head lake** in a dug basin 33.6 m up, a **spline river** running the valley between them, eight buoyant crates at ascending densities, and a swimmer that surfaces. |
| [`phase21-cavern`](phase21-cavern/) | **P21 (the phase gate)** | The composed workings: a **carved cave system** whose mouth is a real hole in an **asset-backed** heightfield, an **excavated foundation pit** with its exactly-conserved **spoil heap**, an **underground room** under the pit, and a Blueprint **borer** that keeps digging at runtime. |
| [`starter-character`](starter-character/) | SK1c | **The engine's starter character** — the eight assets the New Character wizard writes for its own defaults, on the 161-bone mannequin. Scaffolded into every 3D project by `ProjectTemplate::starter_content`, and the island's hero. Not a level: it is content. |

## Project templates

The New Project gallery (and `inf new --template <slug>`) scaffolds from four templates:
`blank-3d`, `2d-platformer`, `first-person`, and `hybrid-2.5d`. The three 3D templates also
scaffold [`starter-character`](starter-character/) into `Content/Characters/`, so a new project
has a rigged, animated character in it before its author has opened a wizard. The `hybrid-2.5d` starter scene
is committed under [`../templates/hybrid-2.5d/`](../templates/hybrid-2.5d/); the others scaffold a
clean cargo crate + `inf.toml` + empty `Content`/`Levels`.

## The "three polished sample games" — honest status

The roadmap (P15.3.4) targets **three polished sample games**: a 3D exploration game, a 2D
platformer, and a top-down shooter. Where we actually stand:

- **2D platformer — DONE.** `platformer-2d` is a real, playable, gate-tested platformer with a
  Blueprint coyote-time jump. This is one of the three.
- **3D exploration — PARTIAL, as building blocks.** `terrain-demo` (sculpted/eroded terrain +
  PCG scatter) and `character-demo` (a state-machine character that walks/jumps across terrain)
  together contain everything an exploration game needs — a traversable world and a controllable
  animated character — but they are two separate gate scenes, not yet composed into one polished
  "walk around and explore" game. Composing them into a single `exploration-demo` project (drop
  the character controller into the terrain, add a follow camera + points of interest) is the
  remaining polish work.
- **Top-down shooter — NOT STARTED.** No sample covers a top-down shooter (twin-stick movement,
  projectile spawning, enemy waves). This is a dedicated follow-up. The pieces exist — 2D physics,
  the input action map, spawn-prefab-on-event Blueprints, the sprite pipeline — so it is
  assembly, not new engine work.

So: **one of three is a polished game today; the second is one composition step away from its
parts; the third is a green-field follow-up.** The rest of the samples above are focused
feature-gate scenes rather than "games", and are documented as such.
