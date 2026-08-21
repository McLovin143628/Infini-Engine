# The CI-scale island (wave I7)

**2.36 km² of the same ground as `samples/island`, with its elevation committed
beside it.** This is the only island CI builds, and it exercises **every step of
the recipe** without ever reaching a network.

```sh
cargo test -p inf-island --test island_fixture
```

## Why it exists

The real island is 51 km² of fetched elevation and 342.7 MB of terrain. CI
cannot build it and must not fetch it. A fixture that stubbed the elevation with
a synthetic heightfield would certify a decoder against itself, so this one
commits **two real terrarium tiles** — the same public, keyless bytes the full
recipe reads — and runs the whole pipeline over them.

| | |
|---|---|
| map | 1 536 × 1 536 m = 2.36 km² |
| land | 1.54 km² (65.3 %) |
| source | **2 tiles at z13, 208 KB committed**, 12.45 m/px |
| terrain | 36 level-0 tiles of 129², 56 in the catalog, 3 LOD levels, 4.6 MB |
| water | 9 reaches / 1.25 km, 1 lake, **4 waterfall sites** |
| roads | 1.37 km, worst grade 0.099 against a 0.100 ceiling, **0 stretches over** |

## The provenance of the committed bytes

`tiles/terrarium/13/1294/2800.png` and `.../2801.png`, fetched 2026-08-21 from
`s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`. Public,
keyless, no attribution required. They are ordinary 256² RGB PNGs and this
engine already decodes PNG a row at a time, which is why real-Earth elevation
costs the tree no dependency at all.

## What the gate asserts

* **CI never fetches.** The plan's tile list and the committed directory are
  compared **both ways**, so a change that needed one more tile goes red here
  rather than reaching for `curl` on a runner — and a tile committed that the
  plan does not name fails too.
* **Every step ran**, counted per `BuildStep::ALL` in its frozen order.
* **The world, not the report**: the built terrain is asked where the ground is,
  at the sites (dry), off all four edges (wet), and at every coastline vertex
  (within 2.5 m of the waterline).
* The sea floor reaches the recipe's own shelf depth — and it is a *different*
  quantity from the island's lowest point, because land inside the shore whose
  survey elevation is under the waterline stays there.
* Streams, lakes and **waterfalls** all exist and read back out of their
  committed layers; a reach's bed really is cut below the ground it was found on.
* The road network holds its grade ceiling **after** the corridor is levelled in.
* Every biome in the palette is reachable; the masks beat the classifier; a city
  site is reserved on the terrain and not merely in a report.
* **Two builds are byte-identical.** Same machine, not cross-platform: the
  sampling step goes through the projection modules the portability law exempts
  by name, and `crates/inf-island/tests/portable_math_law.rs` is where that line
  is drawn.

Regenerate the derived layers with:

```sh
inf island route --recipe samples/island-fixture/island.toml --offline --out ../island-build/fixture
```
