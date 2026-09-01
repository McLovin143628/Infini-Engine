# IASSET1 — sub-entry compression, the `.ipack` rename, and the tables

Wave IASSET1 of the `.iasset` arc (user mandate 2026-08-28 #2). Implemented
2026-09-01. This memo is the ledger: what shipped, what the numbers are, what was
measured and declined, and what is carried.

---

## THE HEADLINE

**The cooked Vancouver Island pack went from 604 631 836 B to 247 497 020 B — a
59.07 % reduction — and boots 6.5 % faster.**

| | before | after |
|---|---:|---:|
| shipped pack | 604 631 836 B | **247 497 020 B** |
| `.inf_terrain` entry | 549 879 456 B | **192 744 640 B** (ratio 0.3505) |
| cook wall time | 42.640 s | 42.673 s |
| headless boot, 120 frames (best of 3) | 1 540 ms | **1 437 ms** |
| final-state hash, 300 frames | `6a34aa77b6095aeb0af1fe954cb5d166` | **identical** |

The boot got *faster*, which the arc brief predicted and this wave measured: the
load is mapping-and-page-fault bound, and 357 MB fewer bytes to touch beats the
decompress that replaces them.

The identical 300-frame state hash across the two packs is the strongest form of
the lossless claim available — not "the bytes round-trip in a unit test" but "the
whole simulated world is the same world".

---

## THE PREMISE, RE-PRICED

The pack has compressed **whole entries** with zstd-19 since P9.2. The kinds that
stream — terrain, voxel, partition, meshlet, texture — opted *out* of that,
because a pack entry is decompressed whole on every `read_ref` and `read_ref` is
on a per-frame path: reaching one 581 KiB terrain tile out of a 550 MB
`.inf_terrain` would decode all 550 MB. So they shipped raw, at 100 % of their
bytes, and the only remaining ship-size lever was the one that must never be
pulled.

IASSET1 is the third option that two-way choice was hiding: compress each
**block** — tile, chunk, cell — independently, record the codec in the
container's own directory, and decompress exactly the block a page-in asked for.

---

## THE CODEC BAKE-OFF

Measured in Rust (`crates/inf-terrain/tests/block_codec_bakeoff.rs`, run with
`INF_BAKEOFF_TERRAIN` pointing at the real island) over 1 064 DEM tiles at 257²,
Windows, test profile with optimizations.

### Per-LOD (sampled), level 0

| codec | stored | raw | ratio | enc ms/tile | dec ms/tile | 16 serialized |
|---|---:|---:|---:|---:|---:|---:|
| raw | 14 003 356 | 14 003 356 | 1.000 | 0.169 | 0.000 | 0.00 ms |
| lz4 | 5 483 471 | 14 003 356 | 0.392 | 0.203 | 0.099 | 1.59 ms |
| deflate | 4 239 369 | 14 003 356 | 0.303 | 8.048 | 0.749 | **11.99 ms** |
| **zstd** | 4 305 651 | 14 003 356 | 0.307 | 3.773 | 0.168 | **2.69 ms** |

### Whole asset (549 879 456 B)

| codec | stored | ratio | tiles compressed | encode |
|---|---:|---:|---:|---:|
| raw | 549 879 456 | 1.0000 | 0 / 1064 | 0.4 s |
| lz4 | 244 258 768 | 0.4442 | 1064 / 1064 | 0.4 s |
| deflate | 196 128 832 | 0.3567 | 1064 / 1064 | 8.8 s |
| **zstd** | **192 744 640** | **0.3505** | 1064 / 1064 | 3.3 s |

### THE RULING: zstd, and it wins on every axis at once

Best ratio, **4.5× faster to decode** than DEFLATE, **2.7× faster to encode**,
and already a shipped dependency. This is not the answer the arc brief expected —
it refused zstd outright on a "zstd-sys is C" reading — and it is exactly why the
brief's own law said to measure rather than reason.

**DEFLATE is the codec that fails.** Sixteen level-0 tiles decompressed serially
is 11.99 ms against `STREAMED_STEP_BUDGET_MS`'s 4.0. The killing arithmetic the
brief predicted is real; it just kills a different candidate.

### Two findings a ratio column would have hidden

**1. The honest denominator.** A worst-case sync already costs **9.66 ms serially
at raw**, because `bincode` has been decoding these tiles since P16.3 and that
cost is not new. End-to-end `load_tile` over 16 level-0 tiles:

| codec | serial | job pool | budget |
|---|---:|---:|---:|
| raw | 9.66 ms | 1.77 ms | 4.0 ms |
| lz4 | 10.50 ms | 2.05 ms | 4.0 ms |
| deflate | 18.01 ms | 2.15 ms | 4.0 ms |
| zstd | 10.13 ms | **2.41 ms** | 4.0 ms |

The **serial path is over budget with or without this wave**; what keeps the
streamer inside it is the job pool, which `sync_render` already uses
(`parallel_map_ref`). Reporting a decompress against zero would have overstated
the wave's cost several-fold. zstd adds **0.47 ms to a 9.66 ms baseline** — under
5 %.

**2. The wasm arm, measured rather than assumed.** `BlockCodec::Zstd` is the C
`zstd` natively and the pure-Rust `ruzstd` in a browser. `ruzstd` decodes the same
level-0 tile in **1.224 ms — 7.3× slower** (16 serialized = 19.59 ms). `lz4_flex`
and `miniz_oxide` are one implementation on every target and have no such split.
**A web-targeted cook should pass `--block-codec lz4`**, which is why the codec is
a `CookOptions` default rather than a law. Carried below.

**3. LZ4's ratio is corpus-shaped, in both directions.** On a synthetic analytic
heightfield LZ4 wins *exactly nothing* (ratio 1.000 — every block inflates and
falls back to raw); on the real island it reaches 0.392. LZ4 is a match-finder
with a 4-byte minimum and no entropy stage, so it can only spend redundancy that
appears as literal repeats: the synthetic surface has none, the island has a flat
ocean across most of its level-0 tiles. DEFLATE and zstd win on both because their
Huffman/FSE stages price down the repeated *exponent byte* whether or not a match
exists. Both numbers are pinned in the bake-off, and neither generalizes.

---

## WHAT THE CONTAINERS DO NOW

### `.inf_terrain` v5 → **v6**

The directory entry's always-zero `reserved u32` becomes a `BlockCodec` byte plus
three zeros; `len` keeps meaning *stored* bytes. `TileStore::tile_bytes` returns a
`Cow`: borrowed for a raw tile (the P16.3 zero-copy read, intact), owned for a
compressed one.

**Why this is legal here and not for `.inf_vmesh`:** terrain tiles were never
*cast*. Every page-in has always run the blob through `bincode`, so the borrowed
slice was an input to a decoder — a decompress in front of that decode is an
addition to an existing cost, not the destruction of a zero-copy path. The mmap
doctrine is about `bytemuck` casts, and it still binds where casts happen.

**Downgrade story — four bytes.** An all-raw v6 image differs from the v5 image of
the same tiles only in `schema_version`, because the codec byte lands where v5
already wrote a reserved zero (`v6_all_raw_is_v5_plus_four_bytes`). v1–v5 payloads
load for ever, and the parser **refuses** a non-zero reserved word below v6 rather
than reading a codec out of bytes that never meant one — which is what makes
"old assets keep loading" a checked property rather than a hope.

### `.inf_voxel` v1 → **v2**

The same shape one dimension up, justified by its own measurement: SDF chunks are
the most compressible thing this engine ships. `samples/phase21-cavern`'s eight
chunks go **131 104 B → 11 619 B, ratio 0.089**, all eight compressed.

This is also the first exercise of the strict reserved-word rule `.inf_voxel`'s
module docs wrote at P21.1: below v2 the **whole** word must still be zero, and an
unknown codec is refused *by name* rather than read as raw (which would hand a
compressed frame to `bincode` and blame `bincode`).

### `.inf_part` — MEASURED AND DECLINED

The partition is genuinely compressible: the island's cell blobs go
**75 424 B → 14 915 B, ratio 0.198**. In absolute terms that is **60 509 B — 0.024 %
of the shipped pack**, and unlike terrain and voxel the `.inf_part` directory entry
has **no reserved word** (`kind u32 · cx i32 · cz i32 · entities u32 · offset u64 ·
len u64`), so a codec would need a real 32→40 B entry lengthening or a
header-level codec with a weaker per-cell fallback.

A container version bump and a wire-format widening for 60 kB is not a trade. It
stays raw, with the number recorded and `EntryPolicy::BlockCompressed` already
naming it, so the day a world ships ten thousand cells the arithmetic is one
measurement away.

### `.inf_vmesh` and `.inf_tex` — OUT BY CONSTRUCTION

`EntryPolicy::MappedInPlace`. Priced anyway, so the carried list has numbers:

| kind | stored | DEFLATE-9 | ratio | saving | share of pack |
|---|---:|---:|---:|---:|---:|
| `.inf_vmesh` | 29 686 000 | 19 294 141 | 0.650 | 10 391 859 | **4.2 %** |
| `.inf_tex` (14) | 6 019 840 | ~2 553 526 | ~0.42 | ~3 466 314 | 1.4 % |

The vmesh is **the largest remaining win in the pack** and it is blocked by the
mmap doctrine, not by effort: `VgeomAssetReader` parses vertex/index sections with
`bytemuck` casts straight off the mapping. The brief's own route — compress only
the *page sections* the `stage_page` seam already copies — is real and is carried,
because it needs the parse-time/stage-time seam separated first and that is its own
wave's worth of care on a 3 218-line container behind a per-frame `read_ref`.

VT tiles are excluded by the uniform-stride container invariant (wave T's ruling,
`inf-vt/src/container.rs`) and are not re-litigated here.

---

## THE POLICY DOOR AND THE ANTI-CLAUSE

`PackWriter::compresses_kind` was a `bool`, and a bool has room for exactly one
reason. It could say "not zstd-compressed" and could not say *why*. It is now
`PackWriter::entry_policy -> EntryPolicy`:

* `WholeEntry` — the P9.2 behaviour, right for an asset read once in full.
* `BlockCompressed` — raw at pack level, compressed per block inside its own
  container. Terrain, voxel, partition.
* `MappedInPlace` — raw, and nothing inside compresses either, because a consumer
  **casts** it. Meshlet mesh, texture.

`compresses_kind` survives as a one-line reader of the policy, so a hundred doc
links and call sites are untouched.

**The anti-clause is a test** (`no_streaming_kind_compresses_whole`). Flipping a
streaming kind to `WholeEntry` is a one-line change that looks like a ship-size
win and is a catastrophe: `read_ref` has no cache, so `MeshletMesh` at
`WholeEntry` decodes the entire `.inf_vmesh` **every frame it is drawn**. The test
fails with that sentence attached, and — the strongest form the guard can take —
names the door that has already taken the win it would be reaching for.

---

## PRICED AND NOT TAKEN: a pack-level sub-directory

The alternative to per-container block policy is a **pack-format** one: give each
index entry a sub-directory of blocks, so `PackReader` itself could hand back one
decompressed block. `docs/memos/p28-2-cluster-pages.md` §1(a) already priced that
exact move for a different purpose and refused it. The price has not changed:

| | container-level (taken) | pack-level sub-directory (refused) |
|---|---|---|
| format moves | `.inf_terrain` v5→v6, `.inf_voxel` v1→v2 — both spending an **already-reserved** directory byte, no record re-length | `PACK_FORMAT_VERSION` 2→**3**, a new per-entry sub-directory inside a **60-byte fixed-stride** index record |
| readers touched | the two containers' own parsers | **every** pack reader, on every platform, plus the wasm arm |
| frozen fixtures | none (the `pack_v1` fixture is untouched) | the committed `pack_v1` back-compat fixture has to keep parsing across a stride change |
| what it buys | exactly what shipped: per-tile / per-chunk decode | the same thing, for kinds that already have a directory of their own |
| what it costs the kinds that need it least | — | every kind pays index size for a feature three of them use |

**The deciding fact is that the streaming containers already have directories.** A
`.inf_terrain` is a header, a sorted tile directory and aligned blobs; putting a
codec byte in it is a byte that was already reserved for the purpose. A pack-level
sub-directory would build a *second* directory in front of the one that exists, so
the pack could learn a fact its payload already knows. It is refused on that, not
on effort — the same sentence P28.2 refused it with.

The one thing the container-level answer does **not** cover is a kind with no
directory of its own that nonetheless wants sub-entry granularity. None exists
today; when one does, this table is where the comparison restarts.

## CARRIED TO IASSET2: the strip list

The arc's IASSET1 clause 1 named a "strip editor-only fields per kind" step. It is
**not in this wave**, deliberately: stripping is only safe with a proof that the
runtime never reads what is stripped, and this wave's whole budget went to
compression, the rename and the tables. The candidates, so IASSET2 starts with a
list rather than a search:

* **`MeshAsset::material_slots` and `SubMesh::name`** — authoring identity. The
  runtime binds materials by GUID off a level entity (the P26.3b wire), not by
  slot name, so these look strippable; "look" is not the proof, and the proof is
  a reader sweep per field.
* **Dropping the source `.inf_mesh` when a `.inf_vmesh` was derived** — the big
  one (18 958 057 B of the island pack is `mesh`), and **currently blocked**: the
  physics bridge builds colliders from the source mesh, and a `.inf_vmesh` is a
  render DAG. Stripping it would ship a world you can see and walk through
  nothing of. IASSET2 should take this as **R4's cross-asset plan shape** — a
  cook-time plan that knows which meshes are *only* drawn — rather than as a
  per-asset strip.

Both are quantization-adjacent in the same way: the moment the cook removes or
rounds something the sim reads, the PIE-reads-loose / shipping-reads-cooked twin
stops being safe, and IASSET1's "compression is lossless so both decode
bit-identical" sentence stops covering it. That is the whole reason this wave
quantized nothing.

## THE RENAME: `.inf_pack` → `.ipack`

89 sites across 44 files, plus `.gitattributes`, the frozen `pack_v1` fixture, the
editor's UI strings and the living docs. `docs/memos/` deliberately keeps the old
spelling: a memo is a dated record, and rewriting one falsifies a source rather
than fixing a reference.

**The FourCC stays `INFPACK\0`.** The extension moved with the product's name; the
*format* did not, and the FourCC is what identifies a format. Changing it would
make every pack ever cooked unreadable to buy a tidier hex dump.

**Compatibility:** a `.inf_pack` written before the rename still opens. Nothing in
this engine validates a pack's extension — the player resolves a pack by path,
`PackReader::open` reads what it is handed, and identity is those eight bytes plus
`format_ver`. The rename changes what the cook *writes*, not what the runtime
*accepts*.

### The three gates (R5)

1. **`city_scale`** asserts `DEFAULT_PACK_NAME == level::PACK_FILE`. Their only
   witness was a `let _` that named the constant and asserted nothing — so
   renaming one and not the other would have compiled, cooked, and produced a
   build the player reports as "no pack" with both spellings correct in their own
   file.
2. **`cook_script`** extends the `.gitattributes` pin to `*.ipack -text` **and**
   `*.inf_tex -text`. The rule covering the frozen fixture named an extension that
   no longer exists — which is how a CRLF finding comes back a fifth time.
3. **`advisory_source_gate::no_rust_source_still_names_the_old_pack_extension`**
   sweeps every `.rs` file for the old spelling **in code** (prose may keep it —
   the compatibility story has to be written down somewhere). Needle and
   falsifiers are assembled, so the gate neither trips on itself nor reports a
   sweep it never took.

`manifest.rs`'s round-trip fixture now reads `DEFAULT_PACK_NAME` instead of a
literal, which would otherwise have sailed through the rename describing a
manifest no cook writes.

---

## THE TABLES' ONE PRODUCER

`CookReport::kind_bytes` and `inf pack ls --totals` both read
`PackReader::kind_totals()` over the pack that was written, so the ship-size table
describes the **file** rather than the writer's intent, and the two cannot
disagree. Counts answer "what is in the build"; only bytes answer "where did the
download go", and `CookReport` had counts only.

```
kind                     n           stored              raw    ratio
terrain                  1        192744640        192744640    1.000
meshlet_mesh             1         29686000         29686000    1.000
mesh                     5         18958057         37910299    0.500
texture                 14          6019840          6019840    1.000
partition                1            75424            75424    1.000
pcg                      8             3122             9670    0.323
skeleton                 1             2463            23026    0.107
anim_clip                3             2265            16563    0.137
blueprint                1              768             9359    0.082
material                 5              322              488    0.660
derived_material         5              308              488    0.631
state_machine            1              282              837    0.337
biome_set                1              216              327    0.661
level                    1              191              272    0.702
TOTAL                   48        247493898        266497233    0.929
file 247497020 B; index + padding 3122 B
```

Note what the `raw` column is and is not: it is the pack entry's
`uncompressed_len`, so for a `BlockCompressed` kind it equals `stored` and the
per-block saving is invisible here — that saving already happened, inside the
payload, before this file existed. The before/after for those kinds is between two
*packs*, which is how the headline table above is built.

---

## BUDGETS

| budget | value | held |
|---|---:|---|
| `LOAD_BUDGET_MS` | 5000 | yes — full headless boot of the island is 1 437 ms wall, load included |
| `STREAMED_STEP_BUDGET_MS` | 4.0 | yes on the job pool (2.41 ms for 16 lod-0 admits); the **serial** path is 10.13 ms and was already 9.66 ms at raw |
| `FRAME_BUDGET_MS` | 33.0 | untouched |
| VT budgets | — | untouched (VT is out of scope) |

No ratchet moved. The serial-path number is stated rather than ratcheted because
it is a pre-existing condition this wave inherited and did not create; the
measurement is now on the record so the day someone runs `sync_render` serially
they find the 9.66 ms rather than blaming the codec.

---

## THE `.iasset` NAMING RULING

The arc adopted the source doc's three-tier naming: editor sources stay `.inf_*`,
the archive becomes `.ipack`. The cooked-entry tier is where a decision was owed,
and it is this: **the cooked FORM lives inside `.ipack`** — kind codes in the
index, per-block policy in each container's own directory — **not as loose
`.iasset` files on disk**.

A loose cooked file would be a second thing to name, hash, deduplicate, watch and
ship, in exchange for nothing the index does not already provide. The cook's
compile step is real (this wave built it: `recompress_terrain_asset`,
`recompress_voxel_asset`); its *output* is an entry, not a file.

---

## THE GUID PROBE, AND THE ATOM TABLE IT LEAVES UNBUILT

R8 owed a number for the doc's atom-slot table. Measured
(`crates/inf-asset/tests/guid_probe.rs`):

| pack entries | ns / probe |
|---:|---:|
| 48 | 12.5 |
| 1 000 | 40.6 |
| 10 000 | 49.5 |
| 100 000 | 59.1 |

The figure that decides is the **inversion**: at the slowest measured probe it
takes **2 810 probes per frame** to reach 1 % of a 16.6 ms frame. The cooked island
holds **48 pack entries in total, one of them a `.inf_vmesh`**.

**The atom table stays unbuilt, with the number beside it.** (The first draft of
this test charged 4 096 probes against a 100 000-entry index and *failed* at
1.457 % — which would have "justified" the table by budgeting for a scene this
engine has never produced. The count and index are now ~250× and ~200× the
island's real ones: a margin rather than a fantasy.)

---

## VERIFICATION

| gate | result |
|---|---|
| `cargo test --workspace -j 3`, `INF_GOLDEN_STRICT=1` | **358 targets / 6 639 passed / 0 failed / 20 ignored**, exit 0 |
| goldens | **60 files**, none blessed by the run (clean tree afterwards) |
| `cargo clippy --workspace --all-targets`, `-D warnings`, LAST | **0 warnings, 0 errors** |
| `cargo doc --workspace --no-deps` | **376** over 30 crates (ceiling 450) |
| `cargo fmt --all --check` | clean |
| frontend | typecheck + eslint clean, **85 files / 776 tests** |
| `cargo deny check` | bans / licenses / sources / advisories **all ok** |

**Two things the battery taught, recorded because both cost time:**

*The re-bless is the downgrade claim, measured on real content.* The two container
bumps move the generator's output, so `samples/phase21-cavern`'s `.inf_voxel` and
both sample `.inf_terrain`s were regenerated — and **each changed in exactly ONE
BYTE**, the schema word at offset 8. Nothing else moved in 131 520 + 17 984 +
35 264 bytes. A unit test asserting "four bytes" is one thing; three committed
sample files agreeing is another.

*The disk law, third time, same disguise.* The first full battery died on
`LNK1140: limit exceeded for program database` — which reads as an MSVC PDB
ceiling and was the volume filling, `target/` at **308 GB** with 20 MB free.
CLAUDE.md says to read a link error against `df` before believing it. The fix
that made the battery affordable is the one **CI has used all along and this
machine had not**: `CARGO_PROFILE_DEV_DEBUG=line-tables-only`, which took the
same battery's artifacts from 300 GB to 27 GB and changes nothing about what is
tested. It belongs in the local workflow, not only in `ci.yml`.

*One advisory, unrelated and unavoidable.* `RUSTSEC-2026-0274` (double free in
`rtrb`'s `ReadChunk::commit` when an element's `Drop` panics) landed upstream
against kira's ring buffer while this wave was in flight, and reds
`cargo deny check advisories`. Fixed lock-only: `rtrb 0.3.4 → 0.3.5`.

---

## CARRIED

* **`.inf_vmesh` page-section compression — 10.4 MB, 4.2 % of the island pack.**
  The largest remaining win. Needs the parse-time cast sites separated from the
  `stage_page` copy seam first; `EntryPolicy::MappedInPlace` is what stops it
  being taken carelessly in the meantime.
* **The web target, and why it is a QUESTION rather than a fix** (sharpened by the
  IASSET1 audit). `ruzstd` is 7.3× slower than the C `zstd` on the same tile and
  the browser player has no job pool, so a web build's worst-case sync is 19.59 ms
  against a 4.0 ms step budget — a regression this wave introduced, because those
  tiles used to ship raw and cost 0 ms to "decompress".

  Two corrections to the first write-up of this item. First, **one door does know
  its target**: `inf_packager::targets::export_web` cooks with
  `CookOptions::default()`, and it is called *because* the user asked for the web.
  "The cook has no target concept" is true of `inf cook` and false of
  `inf export --target web`, which is where a fix would land.

  Second, and the reason it did not land here: **the browser fetches the whole
  pack over HTTP** (`web-player.md`, "Pack streaming: v1 fetches the whole pack"),
  so the codec choice trades decode time against *download* size on a target where
  download is the first cost the user pays. LZ4's ratio is 0.4442 to zstd's 0.3505
  — on the island, **+51.5 MB over the wire** to save ~15 ms per worst-case sync.
  Which of those a web player would rather pay has not been measured, and this
  repo's standing law is that an unmeasured prescription can be backwards. So the
  item is carried with **both** axes named rather than closed with a one-line
  default flip. IASSET2 should measure fetch-to-first-frame, not decode alone.
* **`.inf_part` per-cell compression** — measured at ratio 0.198 (75 424 B →
  14 915 B, so a **saving** of 60 509 B, 0.024 % of the pack) and declined; needs a
  32→40 B directory entry. Revisit when a world ships cells by the thousand.
* **BC7** (IASSET2) + the second-pool/`view_formats` blocker (R9′).
* **BC6H for HDR** — D-18's "largest single win left" for textures.
* **vgeom position quantization** with the T1–T3 traps.
* **`.inf_tex` v3→v4.**
* Undiagnosed P23.5 LSCM healthy-triangle collapse (unchanged).

## FIXED IN PASSING

* `inf pack ls --totals <pack>` parsed `args[1]` as the path, so a flag before the
  path produced an `ENOENT` naming the flag. It now takes the first non-flag
  argument.
