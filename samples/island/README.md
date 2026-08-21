# The Vancouver island (wave I7)

**Fifty square kilometres of real North Shore elevation, carved into an island.**
This folder is the *generator*, not the world: the recipe, the designed
coastline, the biome masks, the road network and the derived water layers —
about 260 KB. The world it describes is 342.7 MB of terrain and is not committed.

```sh
inf island build --recipe samples/island/island.toml
```

That is the whole command. It plans the source tiles, fetches the ones the cache
lacks, samples real elevation onto the world grid, carves the coastline, derives
the water and the biomes, drapes and audits the roads, builds the pyramid, and
writes the heavy halves into a project at `<checkout>/../island-build/project`.
About forty seconds on a warm cache, and about ten megabytes of tiles on a cold
one.

## What is committed and what is not

| committed | why |
|---|---|
| `island.toml` | every decision: where on Earth, how fine, which source, where the sea is, where the settlements are |
| `layers/coast.geojson` | **the designed coastline** — 43 vertices that turn a piece of a mountain range into a landmass |
| `layers/biomes.geojson` | the design masks: the farmland belts and the meadows the classifier is never allowed to invent |
| `layers/roads.geojson` | the road network, routed once under an 8 % grade ceiling and committed as the design |
| `layers/streams.geojson`, `layers/lakes.geojson` | derived from flow accumulation over the carved ground, then committed — the derivation is re-runnable, the layer is the artifact |

| NOT committed | why |
|---|---|
| the `.inf_terrain` | 342.7 MB |
| the road mesh | 517 086 vertices |
| the `.inf_biomes` set | derived from the palette |
| the tile cache | 10 MB of somebody else's bytes |

Everything in the second table is rebuilt by the one command above and lives
**outside the tree**, at `<checkout>/../island-build/`.

## The island in numbers

| | |
|---|---|
| map | 7 168 × 7 168 m = **51.38 km²** |
| land | **40.65 km²** (79.1 %) |
| peak | **948.7 m** — real North Shore ground |
| sea floor | −60.0 m, on a 500 m shelf |
| coastline | **25.14 km** |
| terrain | 784 level-0 tiles of 257², 1 064 in the catalog, 5 LOD levels |
| source | 156 terrarium tiles at z15 = **3.11 m/px**, upsampled 3.11× onto a 1 m grid |
| water | **50 reaches / 26.32 km**, 2 lakes, **33 waterfall sites** (biggest a 29.5 m drop) |
| biomes | forest 38.5 %, plain 20.8 %, meadow 13.5 %, alpine 8.6 %, beach 6.8 %, farmland 6.1 %, urban 5.8 % |
| roads | **33.74 km** over 11 links and 7 junctions; worst grade 0.118 against a 0.080 ceiling, 7 of 2 442 stretches over |

## The elevation is real and the shape is designed

The source is the AWS terrain-tiles terrarium pyramid — a keyless, worldwide DEM
— over the ground behind Ambleside, with Grouse and Hollyburn to the north.
World `(0, 0, 0)` is 49.343 N, 123.102 W, in UTM zone 10N. Remember the frame:
**+X east, +Y up, −Z north**.

**What the survey gives is the relief.** What the design gives is everything
else: the coastline (there is no island there), the sea shelf and the beaches,
the seven settlement sites and their terraces, the road network, and the biome
masks. The build says so every run — `[source.upsampled]` is a standing
advisory, because a 1 m grid over a 3.11 m survey is 3.11× of interpolation and
pretending otherwise would be the most flattering lie this folder could tell.

## Three standing advisories, and why none of them blocks

* **`source.upsampled`** — above.
* **`source.sea_level_tiles`** — 8 of 156 source tiles are uniformly sea level,
  and a missing tile decodes exactly the same way. Nothing here can tell them
  apart; the extent can.
* **`source.implausible`** — 56 source samples carry −32 768 m, which is the
  terrarium codec's floor and means "the provider filled this pixel". It is
  *finite*, so every finiteness guard in this engine waves it through; it is
  nodata here, and nodata becomes ocean.

None of the three is something an author can fix, so none of them stops the
build. What does: a mask that names no biome, and a road network more than 1 %
of which is over its own ceiling.

## The two-pass route

`inf island route` plans the network and then **re-builds against it**, because
the corridor levelling is part of the carve and a road audited before its
corridor is cut is a road nobody has built. Measured: **8.11 % of stretches over
the ceiling before, 0.29 % after.** The seven that remain are places two routes
cross at different elevations, which this generator does not grade-separate.

## What CI runs instead

`samples/island-fixture` — 2.36 km² of the same ground with its two source tiles
committed beside it, exercising every step of the recipe and never touching a
network. See `crates/inf-island/tests/island_fixture.rs`.
