# IASSET2 — per-format page pools, BC7 done honestly, and the formats the measurement kept

Wave IASSET2 of the `.iasset` arc (user mandate 2026-08-28 #2), the texture half.
Implemented 2026-09-01. This memo is the ledger: what shipped, what the numbers
are, what was measured and declined, and what is carried.

---

## THE HEADLINE

**A level that mixes page formats stopped losing 9.2× of its refinement, and the
first thing that freedom bought was normal maps that point the right way.**

| | before | after |
|---|---:|---:|
| a BC1 + BC5 level at the ground library's 4 : 1 tile ratio, pages resident at 24 MiB | **340** (demoted to RGBA8) | **2 445** (7.2×) |
| ground normal map, worst error per channel | **122** / 255 (BC1) | **17** / 255 (BC5) |
| ground normal map, mean error per channel | 10.4 / 255 | **1.7** / 255 |
| `samples/ground` on disk — **the wave's one cost** | 7 394 656 B | 9 207 264 B (**+24.51 %**) |
| `.inf_tex` container version | v3 | **v4** (BC7's format code, nothing else) |
| goldens | 60 | 60 (one re-blessed, digest moved) |

The ordering was the wave's law and it earned its place: the demotion had to fall
*before* a new format could be added, because until it did, adding one made every
level that used it worse.

---

## THE BLOCKER, AND WHY IT WAS FIRST

`build_vt_level` picked **one** atlas format for a whole level by format
equality. A level whose textures disagreed fell back to `Rgba8` — 73 984 B a page
against BC1's 9 248 — so the same 24 MiB budget held 340 pages instead of 2 721.

`inf_material::ground` paid that price rather than take it. All seventeen
committed ground maps were BC1, normal maps included, and the module said why:

> Wave T shipped `PageFormat::Bc5` precisely for normal maps and it has no
> consumer, which is exactly why: **the first content to use it alongside a BC1
> albedo demotes the whole atlas.** … The fix is a second pool or a
> `view_formats` reinterpretation … and it is a wave, not a clause.

So a BC7 texture beside that content would have been a *net regression*, and
building the encoder first would have shipped a format nothing could use.

### RULING: per-format arms, not `view_formats`

`view_formats` was priced as the brief asked and it **cannot express this at
all**. WebGPU's texture-view compatibility rule admits only a format's sRGB
counterpart; BC1 and BC5 are neither the same format nor an sRGB pair of one. It
covers **0** of the mixed-format cases against per-format pools' **all** of them.
(The P26.3 ruling it *did* settle — sRGB decoded in the shader, atlas always
linear — is untouched, and the shader header now says so explicitly so the two
questions are not confused again.)

### What an arm is

`VtResidency` holds `Vec<Arm>`: each arm is one atlas, one format, one slot size,
one share of the budget. A texture records its arm at registration, from its
container's own header, and never leaves it.

The **indirection table stays one buffer with one directory**, so a handle is
still a handle and the per-instance word is byte-identical to what it always was.
Which atlas a texture is in is a word in its own block: `slots_x` and `pool`
joined `mip_count`/`tile_size`/`border`/`flags`, and `TABLE_TEXTURE_HEADER_WORDS`
moved 4 → 6. Reading the pool-wide `slots_x` would have been right for arm 0 and
silently wrong for every other arm — right page size, wrong atlas coordinate, no
error anywhere — so `vt_sample.wgsl` reads both from the block and a source gate
bans the old spelling.

### The admission walk runs once per arm, and that is not a filter

`inf_stream::admit_by_lane` gives up on the first failed acquisition — *"once
acquisition fails nothing in this transaction frees a slot"*. That reasoning is
true over one array of interchangeable slots and **false across two**. One walk
over both arms would let a full BC1 atlas defer every BC5 want in the same frame
with slots free, so the walk runs per arm, in arm order, with per-arm protected
sets (a slot index is arm-local). `a_full_arm_does_not_defer_another_arms_wants`
is the arm that fails if it is ever folded back.

### The budget is SPLIT, never multiplied

`split_pool_budget` divides one total by virtual-tile weight, floors each arm to
whole pages of its own format, and never rounds an arm to zero while another has
two. The sum is `<=` the total, always
(`an_arm_split_never_spends_more_than_the_whole_budget` sweeps five shapes).
`DEFAULT_VT_BUDGET_BYTES` (24 MiB) and `DEFAULT_VT_UPLOAD_BUDGET_BYTES`
(1 MiB/frame) **did not move** — the ratchet law holds, and a single-format level
takes one arm, the whole budget and the pool geometry it had before this wave.

---

## RULING: `VT_MAX_POOLS = 3`, and the GPU chose the number

Each arm past the first is a **sampled-texture binding**, and that is the scarce
resource — not the bind group `ENV_VT_ATLAS`'s folding argument was about.

`Limits::default().max_sampled_textures_per_shader_stage` is **16**. The fattest
lit pipeline is terrain, whose fragment stage carries the shared environment
group plus its own five per-tile textures. Measured by walking into it — asking
for four extra atlases produced:

```
In Device::create_pipeline_layout, label = 'terrain'
  Too many bindings of type SampledTextures in Stage ShaderStages(FRAGMENT),
  limit is 16, count was 18
```

So the budget grants **two** further atlases: three arms, and the lit path then
sits at **exactly 16** (11 environment + 5 per-tile) with no headroom.
`the_lit_pipelines_fit_the_default_sampled_texture_limit` counts off the two
entry lists the layouts are actually built from and pins that, so the seventeenth
sampled texture is a failing test naming the ceiling rather than a driver-shaped
refusal on a user's machine. (The `terrain-tile` entries were hoisted out of
`TerrainNode::new` for the same reason `env_bgl_entries` was in P28.1: a gate
must be able to count the thing it is about.)

Three arms covers every format combination this engine's content can currently
produce. A level with a fourth folds its **lightest transcodable** formats into
the RGBA8 arm and `VtLevelReport::demoted` names them — the old whole-level
demotion, reduced to the maps that caused it. A float format is never folded
(`tile_rgba8` refuses an `Rgba16F` page by name, so folding one would page
black).

Raising it is a device-limit decision with a wasm player behind it, not an edit.

---

## THE IMPORT DOOR GETS A POLICY

`inf_material::mapset` wrote the PBR suffix table down at Wave T and **nothing
called it from the editor**. Every loose image took
`TextureImportSettings::default()` — sRGB, `Auto` — so `rock_2K_Normal.png`
became an sRGB BC1 texture with its X axis quantised onto five bits, and
`rock_2K_Displacement.exr` arrived with everything above 1.0 clipped. That
module's own opening paragraph named the artifact and could not fix it.

Now `image_import_policy` routes a loose image through `classify_map` →
`MapKind::settings(source_is_float)`, and the glTF path routes by **slot** rather
than only by colour space: an image a material binds through `normal_texture`
takes `normal_map()` (BC5) instead of `data()` (uncompressed RGBA8, four times
the page bytes for a signal BC5 carries better). `TextureImportSettings::hdr`
gained its first production caller.

Advisories fire only where the import **departed** from what it always did. An
albedo and an unnamed file say nothing: an advisory on every icon is the noise
that makes the real ones unreadable, which is `tail_cost_advisory`'s own ruling.

### FINDING: the import cache key ignored the settings

`ImportKey::new(source_bytes, settings_bytes)` has always taken two halves and
this call site passed the **extension** for the second — harmless while every
image ran one default, and not harmless the moment the settings depend on the
filename.

Measured the moment the policy landed: `Rock_2K_Normal.png` and
`Rock_2K_Roughness.png` with identical pixels hashed to **one key**, so the
second import returned the first's asset and a roughness map came back as a BC5
normal, with nothing erroring anywhere. The tag now carries the chosen settings
field by field — not through `Debug`, which is not a wire contract and this one
is hashed into a manifest that outlives the process. The gate arm imports exactly
that pair.

---

## THE BC7 ENCODER

Pure Rust, integer-only, and **inside `bc.rs`** so
`the_encoder_never_touches_a_float` covers it: that gate scopes on this file's
`#[cfg(test)]` marker, so a new file would have inherited only the weaker
crate-wide gate.

**THE MODE SUBSET, STATED: mode 6 only.** One subset, 7-bit-plus-p-bit RGBA
endpoints, 4-bit indices. It is the only mode carrying full RGBA at maximum index
precision, so one encoder covers albedo, ORM and masks; every other useful mode
needs the 64 + 64 partition tables and a search over them.

Two CPU decoders were owed and both are paid by one function:
`inf_vt::decode_bc7` serves `TiledTextureReader::tile_rgba8` (the transcode tier,
every adapter without `TEXTURE_COMPRESSION_BC`) and `TextureAsset::level_rgba8`
(the thumbnailer). It decodes mode 6 and states its scope: the container has
exactly one writer, so it is complete for every payload that can exist here, and
a block in another mode decodes to zeros rather than to a guess.

### FINDING: the bounding box was on the wrong diagonal

A naive box takes `(max R, max G, …)` as one endpoint, which is wrong when two
channels are **anti-correlated** — the line then misses every texel in the block.
Measured before the fix: a colinear RGBA gradient with one falling channel came
back **27** off per channel at worst, on content mode 6 fits exactly.

The corner is now chosen per channel by the sign of its covariance against the
widest channel, in exact integers (each term scaled by 16 so no mean needs a
division). It also took a red-against-cyan two-cluster fixture from 2 660 to
**0** — which is why that arm now uses three primaries instead: *a fixture a fix
makes free measures the fix, not the limit.*

**`encode_color_block` (BC1/BC3) has the same defect and is deliberately NOT
fixed here.** It would move the bytes of every `.inf_tex` this repository has
committed, which is a content re-bless and belongs with one. Carried.

---

## THE TABLES

### BC7 against BC1 and BC3, on the committed ground library

Not a synthetic ramp — `inf_material::ground::synthesize`, the same door
`samples/ground/` is written from. Errors are per channel out of 255, RGB only.

**Albedo (5 maps, 1 024² each):**

| | BC1 | BC3 | BC7 |
|---|---:|---:|---:|
| MAE ×1000 | 1 551 | 1 551 | **236** |
| **worst** | **11** | 11 | **5** |
| bytes/block | 8 | 16 | 16 |
| encode ms/Mtexel (release) | 6 | 10 | **95** |

**ORM (5 maps, 512² each):**

| | BC1 | BC3 | BC7 |
|---|---:|---:|---:|
| MAE ×1000 | 1 465 | 1 465 | **481** |
| **worst** | **45** | 45 | **33** |

### BC5 against BC1, on the committed ground normal + detail maps

X and Y only — they are the whole signal, Z is rebuilt from them, and scoring
blue would credit BC1 for carrying a redundancy.

| (7 maps, 512² each) | BC1 | BC5 |
|---|---:|---:|
| MAE ×1000 | 10 383 | **1 710** |
| **worst** | **122** | **17** |
| bytes/block | 8 | 16 |

### The subset's own cost

| fixture | BC1 | BC7 mode 6 |
|---|---:|---:|
| non-colinear block (three primaries), total abs err | 4 242 | **2 660** |
| cutout mask, alpha error | — (no alpha) | **1 024** vs BC3's **0** |

### The pool split, as rates

| a level of… | pages resident at 24 MiB |
|---|---:|
| one BC1 format (unchanged by this wave) | 2 714 |
| **BC1 + BC5, two arms, at the ground library's 4 : 1 tile ratio** | **2 445** (2 173 + 272) |
| BC1 + BC5, two arms, at **equal** tile weight | 2 040 (1 360 + 680) |
| the same level before this wave (demoted to RGBA8) | 340 |

**The split depends on the weighting, so the weighting is named on every row**
(IASSET2 audit). The first draft of this memo carried 2 040 — the equal-weight
figure — under a headline that pointed at the ground library's 4 : 1 case, whose
arm measures 2 445; the `>= 5×` assertion in
`a_mixed_level_pages_at_bc_rates_instead_of_demoting_to_rgba8` was satisfied by
both, so nothing caught the mismatch. That arm now pins **2 173 + 272 = 2 445**
by value.

### The re-priced budgets

Nothing rose. `DEFAULT_VT_BUDGET_BYTES` = 24 MiB and
`DEFAULT_VT_UPLOAD_BUDGET_BYTES` = 1 MiB/frame are **totals** now rather than
per-pool ceilings, divided by tile weight. A two-arm BC1+BC5 level at equal
weight takes 12 MiB each: 1 360 BC1 pages and 680 BC5 pages, and 512 KiB/frame of
upload each. An arm never gets a **zero** upload share, because zero means
UNLIMITED in `AdmitBudget::from_bytes` and would make the smallest arm the one
arm with no throttle at all.

---

## THE CONTENT RULINGS

| slot | format | why, measured |
|---|---|---|
| normal, detail | **BC5** (changed) | BC1's worst is **122 of 255** — the normal does not shade approximately wrong, it faces somewhere else |
| albedo | **BC1** (kept) | BC7 takes worst 11 → 5, for twice the page bytes: half of what an arm holds |
| ORM | **BC1** (kept) | BC7 takes worst 45 → 33, same doubling; mode 6 fits ONE line and O/R/M are three independent signals |
| `Auto`'s alpha branch | **BC3** (kept) | BC7 mode 6 shares one index set between colour and alpha; on a cutout mask BC3's dedicated alpha block scores **0** against BC7's **1 024** |

**"BC1 stays" for the albedo is the honest outcome the brief allowed, and it is
the one the numbers give.** The ground albedos are low-contrast noisy surfaces —
exactly the content BC1 handles well, because a small per-block colour range puts
its 5:6:5 endpoints close together.

**BC7 therefore ships measured and author-selectable, and nothing defaults to
it.** That is a different state from BC5's before this wave, and the difference
is the point: BC5 was *impossible* to use (any use demoted the level), while BC7
is *possible* and simply not the best choice for the content that exists.
`TextureCompression::Bc7` selects it, the encoder is exercised on real content by
three arms, and the day a high-contrast photogrammetric albedo lands the table
above is the one to re-read.

### RULING: BC5 IS A TANGENT-SPACE PRESET, and the wave found that out the hard way

`TextureImportSettings::normal_map()` was routed at the photogrammetry finish's
normal map for the four-times page saving — an uncompressed normal map is
**73 984 B a page** against BC5's 18 496 — and the gate refused it in the same
run.

BC5 stores X and Y; its reader rebuilds `z = sqrt(1 - x² - y²)`, which is
**positive by definition**. That definition is a *tangent-space* law: a tangent
normal pointing into the surface is not a normal map, it is damage. It is simply
**false in object or world space**, where the sign of Z is real data held by
every texel on the far side of the asset — and the photogrammetry map is
object-space (`FinishedAsset::normal`, *"Object-space normals, linear"*).

Measured: **93.00 degrees** of median angular error against the analytic truth,
on a mesh whose own normals are 34.65 out. Reverted; the ruling is written into
`normal_map()`'s doc, at the call site and in the gate's own assertion.

The two callers that *do* take the preset are safe by convention and by
specification respectively: `MapKind::Normal` reads the `_Normal` / `_NormalGL` /
`_NormalDX` suffixes, all of which ship tangent space, and a glTF `normalTexture`
is tangent space per the glTF specification.

### FINDING: a CPU reader of a normal map had nowhere to ask whether Z is stored

`vt_sample.wgsl` has had the rebuild since Wave T, off the indirection table's
`reconstruct_z` flag; the **CPU had no twin**, which was harmless while nothing
shipped as BC5. First measurement of its absence, on the same gate: a reader
taking `xyz * 2 - 1` off a two-channel decode was **45.59 degrees** out at the
median. `inf_material::normal_from_rgba8` is now the door and
`TextureFormat::is_two_channel` is the question, mirroring `PageFormat`'s.

### FINDING: a gate's size proxy, broken by legitimate content growth

`island_gate::a_scene_payload_carries_no_partition` asserted *"the frame is
smaller than the terrain file it names"*, on the argument that this is false for
any payload carrying the terrain bytes. True — and also false for a payload that
carries nothing of the kind and simply got bigger. The BC5 ground normals added
1 812 608 B of texture to a payload that legitimately carries textures, and the
frame crossed a terrain file it has nothing to do with (**7 737 553 B against
7 043 328**) while printing `0 inline terrain(s)` two lines above.

It now searches the frame for a 64-byte window from the middle of the terrain
file — the thing the proxy stood in for — with an anti-vacuity arm proving the
window is findable in the file it came from. *A gate must aim at the thing it
names* (P23), applied to a gate that had been aiming next to it since GTA1.

### Ship-size delta — this wave made the content BIGGER, and that is the cost

The `.iasset` arc's own mandate is **supercompression**, and IASSET1 delivered
it: `.ipack` is 59.07 % smaller on the island. This wave spends part of that
back. It is a trade, not a saving, and it is stated as a trade: **quality for
normals, paid in bytes on disk.**

`samples/ground`: **7 394 656 B → 9 207 264 B**, +1 812 608 B (**+24.51 %**) —
seven maps at 16 bytes a block instead of 8. What buys it is a normal map whose
worst per-channel error falls from 122 of 255 to 17, i.e. texels that were
pointing somewhere else. The same reasoning is what **declined** BC7 for the
albedo and the ORM, where the quality win did not justify the same doubling —
so the increase is confined to the one slot that could show it.

GUIDs are frozen constants rather
than content-derived, so the blast radius was exactly seven `.inf_tex`, seven
sidecars and the README: no `.inf_mat`, no island level, no `.inf_pcg`.

### The golden, re-blessed with its numbers

`ground_close.png` is the one committed frame that depicts those maps. Its
fixture hard-coded `TextureCompression::Bc1` for all four slots; it now calls
`inf_material::ground`'s own per-slot settings, so *"the real committed content"*
stays true rather than remembered.

The frame moved by **mean 0.001212 / max 0.024941** — inside the harness's
tolerance, and re-blessed anyway on the phase26 precedent quoted verbatim in the
gate: *"a frame that quietly depicts an engine that no longer exists is one
nobody can read a regression off"*. `GOLDENS` stays **60**;
`GOLDEN_SET_DIGEST` moved `e4ef462477c82c533c31d92fba74cf71` →
`9460b53af708ab4f1b633e188014c3d9` in all three gates that carry it, each with
the reason beside it.

---

## `.INF_TEX` v3 → v4: ONE WINDOW, ONE FORMAT CODE

v4 appends `PageFormat::Bc7` (code 5) and **not one byte else** — same 128-byte
header, same two directories, same uniform stride. Every pre-BC7 texture still
stamps its own lowest version through `min_schema_version`, so no committed
content moved and no `.ipack` stopped reproducing.

**The per-tile compression IASSET1 carried was weighed for the same window and
declined, with its number.** A `.inf_tex` tile is *already* the compression unit
— a page is a fixed-size slot in an atlas — so a variable-length tile would have
to be decompressed into that slot on the way in: the transcode tier's cost on
**every** adapter rather than only on the ones without BC. What it would save is
the ~2 % a general-purpose codec finds in already block-compressed data
(`.ipack`'s own measurement on `.inf_tex` entries). The ruling lives in
`TEX_ASSET_SCHEMA_VERSION`'s doc rather than only here, so it is read where it is
needed.

---

## RULING: BC6H PRICED AND DECLINED

BC6H is the larger arithmetic win than BC7 was — **16 bytes a block against
RGBA16F's 128 per 4×4, an 8× saving**, the biggest single-format reduction the
enum has left. It is buildable at this wave's bar: its decode is integer
(endpoints unquantise to 17 bits, interpolate through the same weight table BC7
uses, scale by 31/64, reinterpret as a half) and its one-subset mode is the exact
analogue of the mode 6 that shipped.

**It is declined because nothing would encode a byte through it**, and that is
measured rather than assumed. This repository holds:

* **zero** `.hdr` or `.exr` sources,
* **zero** `.inf_tex` stamped with format code 4,
* **zero** sidecars with `hdr = true`.

The only consumer surface is `TextureImportSettings::hdr`, which this wave gave
its first production caller and which has no content to route. A second price the
GPU names: `VT_MAX_POOLS` is 3 and BC6H would not retire `Rgba16F` (existing float
content would keep it), so an HDR level would want a fourth arm the
sampled-texture budget does not grant.

Carried, with the number to beat: **8×**. The paragraph lives in
`PageFormat::Rgba16F`'s own doc.

---

## THE ARC'S COMPARISON TABLE, RE-SCORED

The source doc closes with *"Unreal Engine (`.uasset`/Zen) vs Your Custom
`.iasset` Pipeline"*. Scored against what the arc actually built, rather than
against what it proposed:

| the doc's row | what it prescribed | what shipped, and the score |
|---|---|---|
| **Memory mapping** | *"Direct `mmap` zero-copy binary read (`rkyv`/FlatBuffers)"* | **MET, without the framework.** `mmap` zero-copy has been the pack doctrine since P16; `rkyv`/FlatBuffers were REFUSED (a serialization framework re-buys a layout proven through 40 waves and adds a dep). IASSET1 added `EntryPolicy` so mapped-in-place and copy-to-pool are a per-entry fact rather than a convention. |
| **DCC workflow** | *"Integrated In-Memory Bake: edits hot-swap VRAM directly"* | **MET before the arc** (P23/P24: edit-during-Simulate proven end to end). Untouched by it. |
| **Asset handles** | *"64-bit String Atoms / BLAKE3 hashes"* | **DECLINED WITH NUMBERS.** IASSET1's `guid_probe`: 59.1 ns at 100 000 entries, so **2 810 probes per frame** to reach 1 % of a 16.6 ms frame — against a cooked island of **48** pack entries. BLAKE3 refused separately (xxh3-128 is faster and sufficient for dedup). |
| **Scripting bridge** | *"Direct `mlua` 64-bit handle bindings"* | **DECLINED, replaced.** The SCRIPT arc shipped InfiniScript over the P6 IR; `mlua` was refused in its own rulings. |
| **Cooked binary layer** | *"pre-quantized vertex buffers, interleaved float data, GPU-compressed textures"* | **PART MET.** GPU-compressed textures: BC1/BC3/BC5 shipped and BC7 lands here, with the formats chosen per map by measurement. Vertex quantization is **carried** (this memo's list) with the traps named. |
| **Archive layer** | *"bundled, chunked by world region, async streaming"* | **MET.** `.ipack` (IASSET1), per-block policy headers, 59.07 % smaller on the island and a 6.5 % faster boot. |

The honest summary of the arc: the doc's *architecture* was already this engine's,
its *naming* was adopted, and three of its four "your pipeline" cells were
answered with a measurement rather than with the prescription — two of them
declining it.

---

## VERIFICATION

* `cargo fmt --all --check` — clean.
* `cargo test --workspace -j 3 --no-fail-fast` with `INF_GOLDEN_STRICT=1` —
  **358 targets, 6 659 passed, 0 failed, 20 ignored**, exit 0. Goldens **60**,
  and the tree is clean afterwards: **none blessed by the run**.
* `cargo doc --workspace --no-deps` after `cargo clean --doc` — **409 warnings
  over 48 documented crates**, ceiling 450, and **exactly IASSET1's count**: the
  four this wave's links added were resolved (`crate::container::format_code`) or
  de-linked (`tests::…` items rustdoc cannot see) rather than carried.
* `cargo clippy --workspace --all-targets` with `-D warnings` — **0**, run LAST.
  Three it caught in this wave's own code: two `needless_range_loop` in the BC7
  bit packer and decoder, one `unnecessary_cast`.
* `cargo deny check` — **advisories / bans / licenses / sources all ok**.
  **No new dependency**: the BC7 encoder and decoder are this repository's own
  integer code, like every encoder beside them, and `Cargo.lock` did not move.
* Frontend untouched (no TypeScript moved).

Two batteries were run — one before the hardening pass and one after — and both
report the same 358 / 6 659 / 0 / 20. The first produced the three findings above;
the second confirmed the fixes moved nothing else.

---

## THE LAWS THIS WAVE PAID FOR

1. **A gate scopes on the file it is in.** The no-float gate reads `bc.rs`'s own
   source up to its `#[cfg(test)]` marker, so a BC7 encoder in a new file would
   have inherited only the weaker crate-wide gate. It went in `bc.rs`.
2. **A fixture a fix makes free measures the fix, not the limit.** The
   two-cluster block went from 2 660 to 0 the moment the corner rule landed; the
   arm now uses three primaries.
3. **Measure the prescription before landing it** — twice, in opposite
   directions. BC7 "should" replace BC3 under `Auto` at the same 16 bytes: it
   loses 1 024 to 0 on a cutout mask because mode 6 shares one index set between
   colour and alpha. BC5 "should" replace an uncompressed normal map at a quarter
   of the page: it loses 93 degrees on an **object-space** one, because the Z it
   rebuilds is positive by definition.
4. **A rule the GPU has and the CPU does not is a rule written once and needed
   twice.** The Z rebuild existed in `vt_sample.wgsl` and nowhere on the CPU.
5. **A size proxy is a hostage to content.** A gate that stands in for a
   structural property with an inequality between two unrelated numbers fails the
   day either number moves for its own reasons.

---

## CARRIED

* **`encode_color_block`'s wrong-diagonal bounding box** (BC1/BC3). The
  per-channel corner rule BC7 shipped applies verbatim and is measured to matter;
  taking it would move every committed `.inf_tex` byte, so it belongs in a commit
  whose purpose is that re-bless.
* **BC7's partitioned modes (1, 2, 3, 7) and its separate-alpha modes (4, 5).**
  The first is what a non-colinear block needs — measured cost above, 2 660
  against a perfect 0. The second is what would let `Auto`'s alpha branch move
  off BC3 — measured cost above, 1 024 against 0 on a cutout mask.
* **BC6H** — priced above, declined, 8× on the table when float content exists.
* **A fourth page-pool arm** — blocked by
  `max_sampled_textures_per_shader_stage`, not by this crate. Freeing one sampled
  texture in the lit path buys it.
* **vgeom position quantization** with the T1–T3 traps (IASSET1's list,
  unchanged).
* **`.inf_vmesh` page-section compression** — 10.4 MB, 4.2 % of the island pack
  (IASSET1's list, unchanged).
* **The web codec question** — `ruzstd` decode time against LZ4's +51.5 MB over
  the wire; IASSET1 asked for fetch-to-first-frame rather than decode alone, and
  this wave did not measure it.
* **`.inf_part` per-cell compression** — declined with its number (IASSET1).
* Undiagnosed P23.5 LSCM healthy-triangle collapse (unchanged).
