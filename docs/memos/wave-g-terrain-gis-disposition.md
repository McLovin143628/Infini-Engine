# Wave G — terrain, height maps, vector data: what happened to every item

**Owner's document:** `Rust_Game_Engine_Terrain_Height_Maps_Vector_Data.md` (five chat turns,
467 lines). This memo is the answer to it, item by item, plus the GIS capability plan the
owner asked for alongside it.

**Read this first — there is no bold in the document.** The brief said "bolded passages are
highest priority". Measured against the file: zero `**bold**`, zero `__bold__`, zero
`<b>`/`<strong>`, zero Unicode math-bold. All 34 `*` characters are list bullets. The emphasis
almost certainly existed in the chat UI it was copied out of and was flattened on paste, so
"bold = highest priority" could not be honoured literally. Priority below follows the
document's own emphasis devices instead — its section headings, its "👑 The Hybrid Winner"
row, its numbered "Ideal Rust Pipeline", and the two blocks it calls "complete,
production-ready". **If you re-paste the document with its bold intact, say so and the
priority will be re-derived from it.**

---

## The short version

| verdict | count | meaning |
|---|---|---|
| **SHIPPED** | 11 | built this wave |
| **ALREADY-HAD** | 11 | the engine already did this, sometimes by a different mechanism |
| **PARTIAL** | 5 | the substance exists; a named piece does not |
| **DEFERRED** | 8 | agreed, not built, with a reason and a landing site |
| **CANNOT** | 3 | will not be done as written, with the reason in full |

> **Corrected after audit.** The first version of this table said 14 SHIPPED. An
> adversarial pass over the tree found three rows whose code exists and whose
> *claim* does not: land cover → biome ids (G10) has a classifier with nothing on
> either end of it, GIS attributes → building floor counts (G11) has no
> connecting code at all, and the road ribbon (G29) produces vertex arrays that
> are not a mesh asset and conforms to a terrain callback that no caller supplies.
> Each is now marked for what it is. The rest of this memo is corrected in place;
> §7 records what the audit changed and why, because a memo that quietly improves
> its own scorecard is worth less than one that says where it was wrong.

The document's Part 3 (streaming architecture) is, almost line for line, a description of a
system this engine shipped in Phases 9 and 16. Its genuinely new content is Part 1's pivot to
vector data, Part 4's GeoTIFF recipe, and Part 5's road network — which is what this wave
built.

---

## 1. What the engine gained

### Real-world elevation goes in (Batch G-A)

A GeoTIFF DEM — the format every government elevation portal publishes — now imports through
the same door PNG and EXR heightmaps already used. The document's read recipe (raster size,
geotransform, band 1, no-data value, a windowed read, pixel → world by `t[0] + x·t[1]`) is
exactly right about **what to read**; only the library was wrong (see CANNOT §1).

What that meant in practice, beyond "add a decoder":

* **The memory bound had to survive the new format.** A 16 k × 16 k source imports inside
  ~100 MB rather than the 1 GB its sample grid would occupy, because the importer reads one
  row at a time. The `tiff` crate was chosen over every alternative for one property: its
  `read_chunk` API is the same chunk-at-a-time seam PNG and EXR already gave us. The claim is
  not made in prose — the existing byte-identity determinism gate now has a GeoTIFF twin over
  six source sizes chosen against the *strip* grid, because a chunk-scattering decoder is a
  genuinely different shape of bug from a row walk.
  *(Audit correction: the bound is a property of the **file's** chunk grid, not of the reader.
  A TIFF that omits `RowsPerStrip` — or names one at or past its own height, which is what the
  encoder used in our own tests writes by default — is a single strip, and then "one chunk row"
  is the whole image: 4.2 GiB for the 16 k source this claim is stated against. That layout is
  now refused by name with the `gdal_translate -co TILED=YES` remedy, which makes it the fifth
  thing the importer declines to guess at rather than a silent 4 GiB allocation.)*
* **Four things are refused rather than guessed**, each naming its remedy: a rotated or
  sheared raster (honouring one means resampling, which changes the elevations rather than
  placing them); non-square pixels (which would stretch the world along one axis); an
  unrecognised vertical unit; and BigTIFF, LERC and JPEG2000 by name.
* **Feet become metres exactly once.** Many published DEMs are in feet and the file says so.
  Reading one as metres builds a landscape 3.28× too flat — and nothing about the result looks
  broken, it just looks like somebody scaled it wrong.
  *(Audit correction: true on its own and false in combination. `GDAL_NODATA` states its
  sentinel in the file's **own** vertical unit, and the conversion happened before the no-data
  policy saw the row — so a feet DEM declaring `-9999` handed the policy `-3047.6952`, matched
  nothing, and shipped its voids as finite three-kilometre craters that the finiteness door
  cannot see and the report counted as zero. Measured: 0 of 3 substituted. The sentinel is now
  moved into the sample domain at the same seam the samples are.)*
* **The `.inf_terrain` header's `origin` finally carries a value.** It has been in the format
  since 2026 and every importer wrote zero into it. A georeferenced DEM is the reason it
  existed: the terrain now lands where the survey says it does, so roads and rivers from the
  same source line up with it. This costs **no format bump** — the bytes were always there.
  *(Audit correction: at the end of the wave this was true of the code and not of the running
  editor. `build_anchored` threaded an anchor all the way to the header and its only caller
  passed `None`, so every asset the editor could actually produce still carried a zero origin —
  and turning the wizard's placement switch **on** suppressed the "imported at the world
  origin" advisory while still importing there, which is worse than leaving it off. The anchor
  is now read off the open level and captured into the import job, a queue-level arm asserts a
  non-zero origin in the written payload, and asking for placement and not getting it is
  reported with which of the three halves was missing.)*

### The no-data policy door (new, and not in the document)

This is the thing that would have stopped the very first real GeoTIFF import dead, and it is
worth explaining because it is a change of behaviour.

Real DEMs are **full** of no-data: ocean beyond the survey, the ragged edge of a flight line,
LiDAR voids under canopy and water, cloud shadow. It arrives as `NaN`, as a declared sentinel
(`-9999`, `-32768`, `-3.4e38`), or as an undeclared sentinel the publisher simply knows about.

Before this wave the engine refused every one of them, and the reasoning was right:

> *a no-data pixel means the author's source does not cover that ground, and this engine has
> no representation for a hole in a heightfield.*

But "refuse" as the *only* behaviour means an author's only recourse is to go and edit the
DEM. So the refusal became the **default of a policy**:

| policy | what it does |
|---|---|
| `refuse` | today's behaviour, naming the offending sample. Still the default. |
| `clamp:<metres>` | substitute a stated elevation. `clamp:0` — sea level — is the usual answer for a coastal DEM whose no-data is ocean. |
| `fill-row:<samples>` | fill a void from the valid samples either side of it, for runs up to a stated width. Wider runs refuse rather than smear. |

Two rules fall out of that and are both tested. The policy runs **before** the finiteness
check rather than instead of it, so a non-finite sample the policy did not explain still
refuses. And the choice is recorded in the asset's sidecar, so **re-importing the same file
reproduces the same terrain** rather than depending on what somebody clicked six months ago.

`fill-row` is honest about being row-wise: a true nearest-neighbour fill needs the rows above
and below, which means giving up the streaming memory bound. A 2-D fill is a named remainder,
not a pretence.

### Where on Earth the world is (the geo-anchor)

A level can now say where its origin sits on Earth and in what coordinate system. Without that
one fact, a DEM, a road centreline and a coastline each arrive in their own numbers and
nothing can put them on the same ground.

Two decisions inside it are worth recording:

**The world is a plane in exactly one projected, metric CRS.** This was not a preference —
the engine's own rules decided it. World coordinates are metres with no scale factors allowed,
and a *geographic* CRS is in degrees, which are not metres and are not even a constant number
of metres (a degree of longitude shrinks to nothing at the poles). So a project picks one
projected CRS and every import is transformed into it at the door. Anchoring a world in
EPSG:4326 is refused, and so is anchoring one in Web Mercator — its "metres" are inflated by
about 1.53× at Vancouver's latitude, which would build the island half again too large with no
symptom other than everything being wrong.

**The compass was already decided, and the anchor had to agree with it.** It would have been
easy to treat the sign of the northing as a free choice. It was not: the sky system pinned
+X east / +Y up / +Z south in Phase 17, because that is the frame the sun is placed in. An
anchor that disagreed would mirror every imported map east-for-west while leaving the sun
where it was — a defect with no visual signature until somebody notices the shadows fall the
wrong way at noon. The gate asserts the anchor's axes against the solar module's own sun
direction rather than against a restatement of it.

The anchor also carries the **grid convergence** — the angle between grid north and true north,
which reaches a degree or two at the edge of a UTM zone. The sun is placed from true north, so
without it a world near a zone edge carries that error into its shadows.

### Vector data in (Batch G-B)

Shapefile and GeoJSON now import into one normalized feature type, already reprojected into
world metres, so a new source format is a new reader rather than a new case in every consumer.

*Audit correction: the first version of this paragraph said "everything downstream — roads,
hydrology, biomes, the grammar — consumes that one type". One of those four is wired. A stream
layer becomes the same `WaterBody` + `Spline` pair the hydrology tool creates by hand, and
that path is real. The road graph consumes a layer inside its own tests and nothing calls it;
the grammar does not reference this crate at all; the biome path does not exist (G10). The
`GeoFeature` type is the right shape for all four — it is one consumer today.*

Three hazards this makes loud rather than silent:

* **Axis order.** EPSG:4326's authority definition is *latitude, longitude*; essentially every
  real file stores *longitude, latitude*. This engine takes file order, always, at one door —
  and a transposed record is refused **by name** rather than landing in the Indian Ocean.
* **The projection library reports failure by returning NaN**, not by erroring. That default
  suits a browser map, where a failed reprojection should not abort a pan; at an import door it
  is exactly backwards, because a NaN easting becomes a NaN vertex whose bounds then look
  perfectly healthy. Every transform is finiteness-checked at the point of production.
* **A skipped feature is a reported feature.** A published layer of 4 000 roads routinely
  contains a handful of unusable records. Those are skipped *and named*, because the
  alternative is an author spending an afternoon looking for a road the importer silently
  declined to build.

### Roads, polygons and the rest (Batch G-C)

The document's road model — intersections as nodes, road stretches as edges — was adopted
almost verbatim, with two required changes:

1. **`HashMap` had to become `BTreeMap`.** Anything reaching a cooked asset must iterate in a
   deterministic order, or two cooks of the same Shapefile produce different bytes while the
   geometry looks identical every time. This is a standing rule here, paid for previously.
2. **Road spines had to carry a height.** The document's `Vec<[f64; 2]>` cannot describe a road
   that crosses a valley.

Intersections are *derived*, not read: published layers carry segments whose endpoints happen
to coincide to within digitising precision, so the graph snaps endpoints onto a lattice. That
tolerance is a real modelling decision — too small leaves a city of disconnected stubs, too
large welds an overpass to the road beneath it — and it is documented where it is defined.

Roads extrude into quad ribbons with arc-length UVs, mitred corners and an upward winding.
*Audit correction: the first version said "terrain-conforming" and "closes a follow-up the
engine's own spline module had written down as unbuilt". The ground comes from a caller-supplied
closure that no caller supplies, and the result is vertex/UV/index arrays rather than a
`MeshAsset` — so the ribbon is built and is not yet attached to a terrain or to the asset
system. `inf-math`'s spline module still lists that follow-up as open, correctly. The audit
also found the corner mitre was **aimed but not scaled**: offsetting by the half-width along
the bisector pinches a 10 m road to 7.07 m through every right-angle bend, and the arm that
should have caught it asserted `(10.0..20.0).contains(&w)` — which the pinched answer, 10.0
exactly, satisfies. Both are fixed and the arm now asserts 10·√2.*

Polygons — parcels, building footprints, lake surfaces, land-cover regions — now **triangulate**
as a library operation. **The engine previously had no polygon primitive at all**; every one of
those collapsed to an axis-aligned bounding box, which is what blocked the document's most
attractive promise ("align a house mesh inside a property boundary"). Constrained Delaunay
handles holes, nested holes and concave boundaries. *Audit correction: "now triangulate" is
true of the operation and not of the scene — the layer spawner still turns an area into a
closed spline, because putting a polygon SURFACE in a level needs the polygon-interior work
this wave deliberately deferred. The audit also found that `spade`'s `add_constraint` **panics**
on a self-intersecting ring, which published parcel and land-cover layers carry routinely; that
is now a refusal naming the remedy, and a face-culling pass that was quadratic in vertex count
now rejects against per-ring bounds first.*

> **A real bug in the document's sample code, worth naming.** Its `generate_terrain_mesh`
> pushes into a parallel `heights` vector only on a *successful* insert, then indexes that
> vector by the triangulation library's internal vertex index. Those two numberings diverge
> the moment a duplicate point is rejected — and a DEM resampled onto a lattice produces
> duplicates routinely. The result is a mesh whose vertices wear other vertices' elevations,
> silently. Our triangulator carries the elevation *inside* the vertex type so there is no
> parallel array to fall out of step, and a test pins it on input containing duplicates.

### Ported from GeoCanvas

Three things crossed, adapted into our conventions rather than copied:

* **XYZ tile arithmetic**, keeping the one habit that earns it: each tile edge is computed from
  its own index rather than as `min + size`, so adjacent tiles share bit-identical edges and a
  terrain built from them has no hairline seam.
* **The terrarium elevation codec** — which makes real Earth elevation importable with **zero
  new dependencies**, because the tiles are ordinary PNGs we already decode. One caution is
  recorded rather than papered over: a missing tile and genuine open ocean decode identically,
  so the decoder reports the ambiguity and leaves the policy to the caller.
* **Jenks natural breaks**, for turning a land-cover raster into biome ids. Equal-interval
  classing is usually wrong for geographic data — real distributions are lumpy, so equal-width
  classes put nine tenths of the map in one biome.

---

## 2. The three CANNOTs, in full

These are written to be read directly.

### CANNOT 1 — the `gdal` crate (and `proj-sys`, and `geos`)

> We will not take the `gdal` crate. It is a binding over the GDAL C++ library, which drags in
> PROJ, GEOS, libtiff, libcurl, OpenSSL, SQLite and roughly forty more shared libraries, and
> our CI builds and tests on **three** operating systems.
>
> The evidence that this is a real liability rather than a theoretical one is sitting in your
> other repository. GeoCanvas solves the Windows half by **vendoring a 52-DLL GDAL runtime**,
> a vendored import library and headers, an environment variable, a runtime DLL-search-path
> shim, and a PE-import-table diagnostic for when loading fails with a misleading error. There
> is **no Linux or macOS resource tree in GeoCanvas at all** — it is Windows-only. That is the
> honest cost of GDAL in a Rust desktop app, measured from our own code.
>
> This repository has also already refused this exact class of dependency **twice**, in
> writing: a BC7 texture compressor whose ISPC build was called "a cross-OS CI risk we decline
> to take on now", and a QUIC crypto backend for the same reason. Taking GDAL would reverse
> both rulings for a capability we can serve with pure Rust.
>
> `geos` is refused twice over: it is native C **and** LGPL-2.1, which is not on this
> project's licence allow-list at all.
>
> **What we genuinely lose, stated plainly:** the 200-odd exotic raster drivers (JPEG2000,
> HDF, netCDF, MrSID, ECW); LERC-compressed files, which ArcGIS produces by default and most
> government portals run on; on-the-fly warping between coordinate systems; the universal
> vector driver set (we hand-picked Shapefile and GeoJSON instead); GEOS geometry predicates
> and buffering; and PROJ's full EPSG database with its NTv2 datum grids.
>
> Each of those has a stated workaround. The important ones: a LERC file needs one
> `gdal_translate -co COMPRESS=DEFLATE` on your machine and then imports unchanged — and the
> importer's error message says exactly that rather than "unsupported compression". A rotated
> or non-square-pixel raster needs one `gdalwarp`. A coordinate system outside our curated
> table can be pasted in as a proj4 string, which is accepted verbatim and has no such limit.

### CANNOT 2 — a TIN terrain

> We will not convert the DEM into a triangulated irregular network for the terrain surface.
> The reason is not that TINs are bad — it is that the engine already has a complete, shipped,
> gated answer to the same problem, and a TIN would be a *second* terrain renderer running
> beside it rather than an improvement to the first.
>
> Terrain is a regular grid end to end: a fixed sample lattice per tile, a 2:1 decimation
> pyramid, uniform-grid patch meshes with morphing and skirts, a heightfield collider, a hole
> mask, a GPU sculpt and erosion path, a height query four other systems call, and a streaming
> residency budget. Every one of those assumes the lattice. An irregular network invalidates
> all of them at once.
>
> The specific benefit the document asks for — "flat ground costs 4 points, ridges cost many" —
> is already delivered by the pyramid plus the clipmap ring selection: a flat region streams
> and draws at a coarse level and never pays for its interior samples.
>
> One caveat worth recording even so. Where genuinely irregular, error-driven simplification
> *is* the right tool, this engine already uses quadric error metrics — but that simplifier is
> **not cross-platform deterministic**, and terrain that gets cooked and shipped must be
> bit-identical across machines. So a TIN could not use it and would need its own portable
> simplifier. That is a second reason this is a large project rather than a small one.
>
> Delaunay triangulation was **redirected**, not discarded: it now triangulates GIS *polygons*,
> which is a job the engine genuinely could not do.

### CANNOT 3 — `rkyv` / `zerocopy` zero-copy archives

> `rkyv`'s licence is fine. We are refusing it on architecture.
>
> This engine has one asset container doctrine and it is load-bearing: a binary payload plus a
> deterministic text sidecar, with a schema version and a migration path on every schema
> struct. Every rung of every ladder is a frozen previous-version record; there are more than a
> dozen such ladders in the tree. `rkyv` archives are byte-layout-pinned to the Rust struct
> definition with no migration story of that kind. Adopting it would mean a second container
> discipline with weaker version guarantees, for assets that must load forever.
>
> And the property the document actually wants — "map the file bytes directly into memory
> arrays, no allocate, no parse" — **we already have**, without it. Pack reads return borrowed
> slices pointing straight into a memory mapping, and the streaming-class asset kinds are
> deliberately cooked *uncompressed* with aligned offsets precisely so that works. The document
> is prescribing a mechanism for a property Phase 16 already delivered.

---

## 3. Every item in the document

Numbered in document order.

### Part 1 — raster versus vector

| # | prescription | verdict |
|---|---|---|
| G1 | A 16K RGBA layer costs ~530 MB; ten stacked layers ~5.3 GB, so giant rasters are the wrong authoring substrate | **ALREADY-HAD.** The engine never holds a whole heightmap; a 16 k² import runs in ~100 MB, asserted by a test that measures the live set rather than claiming it. |
| G2 | Pivot roads / rails / parcels from raster to **vector**; ingest GeoJSON / Shapefile | **SHIPPED.** Both readers, one normalized feature type. |
| G3 | Use high-resolution rasters **only** for elevation | **ALREADY-HAD** in principle, **SHIPPED** in fact — the division was already the engine's; the missing half was the GeoTIFF reader. |
| G4 | Feed vector coordinates for roads, utilities, parcels; align building meshes inside parcel polygons | **SHIPPED (polylines + polygon triangulation)** / **DEFERRED (oriented lots)** — see the remainders. |
| G5 | A data-driven ECS spawns the objects; a "Residential Parcel" gets Building + Mesh + Collision components | **ALREADY-HAD.** This describes what the procedural-generation phase already does. |

### Part 2 — point data and sources

| # | prescription | verdict |
|---|---|---|
| G6 | Do **not** store a raw dense point cloud | **ALREADY-HAD.** Import goes row → tile → asset; no point cloud is ever materialised. |
| G7 | Adaptive density — flat ground collapses, ridges densify | **PARTIAL, by different means.** The pyramid + clipmap deliver the benefit; a TIN is refused (CANNOT 2). |
| G8 | Use `spade` to turn the point cloud into a mesh | **PARTIAL / redirected.** Adopted for GIS polygons, not for terrain. |
| G9 | Source 1 m bare-earth DTM GeoTIFF; also raw LAS LiDAR | **SHIPPED (GeoTIFF)** / **DEFERRED (LAS)** — the LiDAR crates are clean and pure Rust, but a point→raster gridding step does not exist. |
| G10 | Ingest roads, parcels, utilities, zoning / land cover → biome painting | **SHIPPED (vector)** / **DEFERRED (land cover → biomes)**. Roads, parcels and utilities import as features. The land-cover half does **not** ship: Jenks classification is built and tested (`classify_breaks`, `classify_to_ids`), and there is still **no path from a raster to a `BiomeSet`** — nothing decodes a land-cover image, nothing writes biome ids, and the classifier has no caller outside its own tests. The claim that this was "the cleanest fit in the document" stands; the claim that it was built does not. |
| G11 | County datasets — building footprints, sewer, zoning height limits | **PARTIAL.** Footprints and utility lines import as features like any other layer. Height limits do **not** map onto `BuildingParams::floors` — there is no code between a GIS attribute and that field in either direction. "Maps onto" described a shape, not a wire. |
| G12 | Use the **`gdal` crate** | **CANNOT 1.** Replaced by the pure-Rust `tiff` crate. |

### Part 3 — binary format and streaming

This whole section describes a system shipped in Phases 9 and 16.

| # | prescription | verdict |
|---|---|---|
| G13 | Pre-cook GIS data at a bake stage; never parse GeoTIFF/GeoJSON at runtime | **ALREADY-HAD**, and it is this engine's standing cook doctrine. Every GIS dependency added this wave is host-only and never linked by the shipped player. |
| G14 | Zero-copy deserialization via `rkyv` / `zerocopy` | **CANNOT 3** as written; the *property* is already had. |
| G15 | Memory-map the world files | **ALREADY-HAD.** |
| G16 | Continuous spatial quadtree / R-tree; 1 km chunks each carrying terrain + road + parcel data | **ALREADY-HAD** — a uniform partition grid plus a terrain quadtree, rather than one R-tree; functionally the requirement, already streaming. |
| G17 | Background threads look ahead and stream chunks in, dropping far ones | **ALREADY-HAD**, with the decode fan-out deterministic and in-order so streaming can never move a simulation trace. |
| G18 | The "Hybrid Winner": multi-threaded loader + memory-mapped pre-cooked binary + LOD hierarchies | **ALREADY-HAD.** This is the composition of G13 + G15 + G16 + G17. |
| G19 | A separate lightweight CLI asset compiler | **ALREADY-HAD (the tool)** / **SHIPPED (its GIS front end)**. |

### Part 4 — distant LOD and the GDAL boilerplate

| # | prescription | verdict |
|---|---|---|
| G20 | Hierarchical quadtree: one global silhouette, per-island mid-res, 1 km chunks | **PARTIAL.** The lower two layers exist; a distinct always-resident global silhouette does not. |
| G21 | `WorldManifest` / `GlobalSilhouette` structs with per-island bounds for GPU occlusion cull | **CANNOT 3 as written / PARTIAL in substance** — per-tile height bounds already feed culling. |
| G22 | Boot-load the global silhouette and keep it in VRAM as a skybox replacement | **DEFERRED, low priority.** Nothing blocks it; nothing needs it until a multi-island world exists. |
| G23 | Raycast the quadtree; fade the silhouette out at ~10 km and stream real chunks | **PARTIAL.** The distance-banded streaming machinery is exactly what exists; the cross-fade is not, because there is no silhouette. Content-shaped, not engine-shaped. |
| G24 | The GeoTIFF read recipe | **SHIPPED — the single highest-value item in the document.** The recipe is right; only the library was wrong. |
| G25 | Pipe `(x, y, elevation)` into a simplification pass inside the read loop | **CANNOT 2 for terrain.** Superseded by the pyramid and clipmap. |

### Part 5 — meshing and the road network

| # | prescription | verdict |
|---|---|---|
| G26 | `spade` Delaunay: insert points, keep a parallel height table, emit buffers | **PARTIAL / redirected** — and the sample code has a real indexing bug, described above. |
| G27 | Model the road network as a directed graph | **SHIPPED.** No road type of any kind existed in the repository before this wave. |
| G28 | `RoadNetworkGraph` / `RoadSegment` / `RoadType` struct shapes | **SHIPPED, with two required changes** (`BTreeMap`, and spines carry a height). |
| G29 | Procedural extrusion: width from lane count, perpendicular cross-sections, snap to terrain, stitch a quad ribbon with tiling UVs | **PARTIAL.** The generator ships and is correct — mitred corners, arc-length UVs, upward winding, refusals for every degenerate input — but it returns **vertex, UV and index arrays, not a `MeshAsset`**, and it takes the ground as a caller-supplied closure that nothing in the tree supplies. So it is a ribbon builder that is not yet wired to a terrain or to the asset system. `inf-math`'s spline module still lists "baking a spline to a renderable mesh (tube / ribbon)" as unbuilt, and it is right to. |
| G30 | Spawn a building in a parcel, query the road graph, rotate the house to face the street | **PARTIAL.** The nearest-segment query ships; oriented lots are deferred — see the remainders. |
| G31 | (Offered) A shader for blending textures where roads meet terrain | **PARTIAL.** The four-layer splat path and the deformation field are what such a blend would ride; no prescription to implement — the document only offers it. |
| G32 | (Offered) Structure the asset compiler to process DEM + vector simultaneously | **ALREADY-HAD.** |

---

## 4. Deferred, with reasons

| item | why, and where it would land |
|---|---|
| **LAS / LAZ LiDAR** | The crates are pure Rust and licence-clean; the missing piece is a point-cloud→raster gridding step, which is real work rather than a reader. |
| **Polygon interiors and oriented building lots** | Scattering *inside* an arbitrary boundary, and rotating a lot to face a street, both need an oriented (or concave) lot. The building floor-plate slicer assumes axis alignment throughout. That is a deep change and deserves its own sub-phase rather than being smuggled into a GIS wave. |
| **A global silhouette LOD layer** | Nothing needs it until a multi-island world exists, and it is content-shaped rather than engine-shaped. |
| **2-D no-data fill** | The row-wise fill is what a streaming decoder can honestly do; a true nearest-neighbour fill means buffering the image and giving up the memory bound. |
| **OSM protobuf** | The government portals the document names serve Shapefile and GeoJSON, which are covered. |
| **GeoPackage, cloud-optimised GeoTIFF over HTTP, WMS/WMTS, LERC** | Each is a named gap with a stated author-side conversion step. |

## 5. Things that were measured and turned out otherwise

Recorded because each one was believed before it was checked, and each would
have shipped as a plausible-looking mistake.

**A range measures the scene, not the texture.** The first version of the
terrain fragment probe compared the textured terrain's red-channel *range*
against the same terrain untextured, expecting the control to be narrow. The
control spans 206 of 255 levels all by itself — from lighting and the terrain's
silhouette against the sky. The second version counted sharp descents instead
(a ramp texture falls off a cliff once per repeat; smooth shading does not fall
at all) and failed in the *opposite* direction: the untextured terrain scores 335
descents on its own, because it carries a procedural triplanar grain that the
texture then largely replaces. The control that works holds the **code path**
fixed and varies only the texels — same pools, same slot, same gradients, a
constant-colour texture instead of a ramp.

**`vt_engaged_frames` counts pool *bindings*, not samples.** The probe was
written asserting that a terrain naming no textures would not engage the virtual-
texture path even with pools bound. It measured one. The counter's honest meaning
is "frames drawn with a pool bound", which is what it was introduced to measure —
evidence about the command stream — and it is never evidence that a sample
happened. That is now pinned in a test with the distinction spelled out, because
the name reads like the other question.

**`+datum=NAD27` cannot be built at all.** The pure-Rust projection library asks
for an NTv2 grid it does not ship and refuses the projection outright, so a NAD27
source would have been unimportable. It is spelled with its ellipsoid and an
explicit shift instead, and the residual is advised rather than hidden.

**A finiteness check must run before deduplication, not after.** Every comparison
against a NaN is false, so a ring-cleaning pass that removes coincident points
silently *drops* a non-finite vertex — and the polygon is then refused for the
wrong reason, or quietly triangulated without it.

**The upward-facing winding wants the shoelace sign negated.** For a triangle on
the ground plane, the 3-D cross product's vertical component is the negation of
the usual 2-D orientation term. The first version had it the other way round and
every face pointed at the ground.

**Field names need separator folding, not just case folding.** `ROAD_TYPE` and
`RoadType` differ by an underscore, not by case, and they are exactly the pair a
real attribute table will hand you.

## 6. Honest limits of what shipped

Stated plainly, because these are the things that will surprise somebody:

* **A Shapefile's coordinate system is not read automatically.** It lives in a sidecar `.prj`
  holding WKT, a format we have no parser for, so the CRS is a parameter the caller states.
  *(Audit correction: the first version of this bullet said "the wizard shows the author the
  `.prj` text so the choice is informed". Nothing reads the `.prj` — and, as the bullet three
  below says, there is no GIS wizard at all. Showing it is the design; none of it is built.)*
* **Datum shifts are analytic.** Without NTv2 grids, shifting between old national datums
  carries metre-class error — a uniform offset of the whole import, not a distortion of it,
  and invisible in a game. It is **reported as an advisory** rather than hidden, following
  GeoCanvas's own precedent of warning and continuing.
* **Coordinate systems outside the curated table are refused by name**, with the proj4-string
  escape hatch in the message. UTM zones are derived from their code rule (180 definitions,
  no table); everything else is a short list.
* **The engine applies no geoid model.** A source in orthometric heights and one in
  ellipsoidal heights differ by tens of metres in places, and nothing here corrects that. The
  vertical datum name is recorded so a mixed import can at least be *noticed*.
* **Road intersection snapping is planar.** Two endpoints at the same plan position but
  different heights — a bridge and its underpass — are snapped together. Separating them needs
  a bridge/tunnel attribute that published layers rarely carry.
* **A terrain in a different coordinate system from the level is refused, not reprojected.**
  Reprojecting a raster means resampling it, which changes the elevations rather than placing
  them.
* **The geo-anchor has commands and typed bindings but no panel yet.** It can be read and
  written over the editor's typed IPC, and the terrain import wizard shows a source's
  georeferencing and offers to place against it; a World Settings row for typing the anchor
  by hand is not built. The import path sets it, which is the flow that matters, but an
  author who wants to change it afterwards currently needs the API.
* **The vector import doors are library code, not a wizard.** Shapefile and GeoJSON read into
  layers, layers become road graphs, ribbons and triangulated polygons — all callable and all
  tested — but the GIS Import dialog that would walk an author through picking a file, naming
  its CRS and choosing a layer kind is not built. That is the next batch's shape, not a
  defect in this one. **This is the wave's largest honest limit and it deserves the plainest
  wording:** `inf-gis` has exactly one dependent crate in the workspace, and that crate uses
  exactly one of its nine modules. The road graph, the ribbon builder, the triangulator, the
  Jenks classifier, the terrarium codec and the tile math have **no caller outside their own
  tests**. They are correct, gated library code waiting for a door, and every claim in this
  memo that reads as "the engine now does X" should be read as "the engine now *can* do X, from
  Rust, when something calls it".
* **The road graph is derived at bake, never persisted.** Deliberate, following the
  procedural-generation precedent: the vector layer is the source of truth and a derived thing
  that is also stored is a thing that can disagree with its own source. It also means the road
  system costs no schema ladder.

---

## 7. What the audit changed

This memo was audited adversarially after the wave closed. Nine defects reached the tree and
were fixed in the audit commit; four claims in this document were wrong and are corrected
above, in place, with the correction visible rather than the sentence quietly rewritten.

**Defects found and fixed.** Each was measured, not reasoned about, and each now has an arm
that fails without the fix:

| what | why it mattered |
|---|---|
| A feet DEM never matched its own no-data sentinel | `GDAL_NODATA` is stated in the file's units and the row was already in metres; 0 of 3 voids substituted, and the terrain shipped with finite three-kilometre craters that the finiteness door cannot see |
| A single-strip TIFF materialised the whole image | "one chunk row" is the whole image for a file with no `RowsPerStrip`; 4.2 GiB for the 16 k source the memory bound is stated against. Now refused with its remedy |
| The level's anchor never reached the importer | `use_georeference` was a no-op that also **suppressed** the advisory saying so — strictly worse than leaving it off |
| The loader's own anchor gate was blind | `SceneDoc::reset` did not clear the field, and the arm seeded the value it then asserted; deleting the confessed fix left all 666 tests in the crate green |
| A self-intersecting ring panicked the triangulator | `spade::add_constraint` panics on a crossing constraint, and published parcel layers carry them routinely — one bad record took the whole import down from inside a dependency |
| Every road corner pinched to 70.7% width | the mitre was aimed along the bisector and not scaled by `1/cos(θ/2)`; the arm that named 14.1 in its failure message accepted the pinched 10.0 |
| A refused polygon hole was silently dropped | `filter(is_ok_and(..))` removed the `Err` before `collect::<Result<_,_>>` could see it, so the `?` was dead code and a NaN hole became a polygon with one fewer hole |
| Non-finite inputs became plausible answers | a NaN longitude became tile (0, 0); a NaN lift and a NaN ground query became mesh vertices; a non-finite DBF number became a stored NaN |
| `.log2()` was not on the libm ban list | the canonical portability list had a hole the first new crate walked through, on a path that selects which source tiles get cooked. The whole logarithm/exponential family, the hyperbolics and `hypot` are named now, and `inf-gis` has the source gate its seven sibling crates already had — with its two projection modules exempt **by name**, with the reason and the condition that retires the exemption |

**Claims corrected**, each marked in place above: the SHIPPED count (14 → 11), the
`.inf_terrain` origin, the memory bound, the feet conversion, the `.prj` text, the list of
downstream consumers, "roads extrude into terrain-conforming ribbons", and "polygons now
triangulate".

**One claim in §5 was recorded as MEASURED with no surviving measurement**, and now has one.
"`+datum=NAD27` cannot be built at all" justified spelling EPSG:4267 by ellipsoid instead —
but every NAD27 test drove the *table's* spelling, so the claim about the spelling it avoids
rested on a comment. It is re-derived in four lines
(`the_datum_name_this_table_avoids_genuinely_cannot_be_built`), and it holds: the projection
library refuses `+datum=NAD27` outright, the `+ellps=clrk66 +towgs84=…` form builds, and a
NAD27 source imports with its advisory. The day the library gains NTv2 grids that test fails
and tells somebody the table's note is stale, which is what a measurement is for.
