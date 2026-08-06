# P23.1 — Embedded DCC v1: the design decisions

**Status:** decided, 2026-08-05. Implemented enablers: P23.2a (this batch).
**Scope:** answers the DCC spec's open questions *against this engine's
infrastructure*, and records the rulings the P23.3–P23.6 batches build on.

The one-line summary: **the DCC is an asset editor, not a second world.** It
edits a mesh keyed by asset id, saves through the asset database, and lets the
machinery that already exists — the watcher, the content-hash render key, the
`assets://changed` fan-out — carry the result into every view that references
it. Nothing about the level document is involved at any point.

---

## 1. What we reuse, and what we are not building

**Standing decision (nextgen plan, 2026-07-31), reaffirmed: no Blender DNA/RNA
clone.** Blender's DNA/RNA is a hand-rolled reflection + versioning system built
because C had none. We have three of its jobs solved already, and building a
fourth mechanism to sit beside them would be the largest source of drift in the
codebase:

| Blender solves with DNA/RNA | We already have |
| --- | --- |
| Type introspection for the property editor | `bevy_reflect` behind `inf_ecs::props` — the Details panel is *already* reflection-driven, and `bevy_reflect` never leaves `inf-ecs` (P3 architecture rule) |
| File versioning / forward-compat | `schema_version` + `migrate()` on every payload, the dual-format rule (bincode + deterministic TOML sidecar), and the frozen-record + downgrade-bless gates |
| Datablock identity and linking | `AssetId` GUIDs + `AssetDb`'s forward *and reverse* dependency edges, content-hash dedupe, and delete-with-references |
| Undo | `GraphJournal` (whole-snapshot) for graph documents; `EditCommand` + transactions for the scene |

So P23.3's `inf-dcc` is a **half-edge mesh kernel and an op journal**, and
nothing else. It imports from and exports to `inf_mesh::MeshAsset` — the
exporter `inf-mesh` has never had — and it derives `Reflect` where the Details
panel needs to see a value. It does not get its own type system, its own file
format, or its own undo stack.

## 2. The edit session is STRICTLY ASSET-SCOPED

**Ruling.** The Model Editor (P23.4) edits a `DccDoc` keyed by asset id and
**never touches the scene document.**

This is the single most load-bearing decision in the phase, and it is what makes
everything downstream cheap:

* No schema move. `.inf_lvl` is untouched, so no version bump, no frozen-record
  update, no downgrade-bless.
* No lock-order question. The DCC session takes the asset project's lock; the
  viewport thread takes the scene document's. They never meet, so the
  document-then-volumes ordering discipline (P21.2) does not grow a third rung.
* No "the render store IS the save's staging source" hazard (P21.4). A mesh edit
  is written to a mesh asset. There is no camera-paged store in the path and
  nothing the author did not ask for can reach the level.
* The scene *sees* the edit for free, through machinery that already exists.

### The state machine

```
  open asset (Content Drawer ▸ "Edit Mesh", or a double-click)
        │
        ▼
  DccDoc in DccState  ── the GraphDoc/GraphJournal four-store template:
        │                { registry, docs: BTreeMap<Id, Doc>,
        │                  journals: BTreeMap<Id, Journal>, counter }
        │                exactly as commands/{graph,material,pcg,sm}.rs already do
        ▼
  apply ops  ─────────  each op is journalled (§4); the panel re-renders its
        │               preview from the edited kernel, offscreen (§5)
        ▼
  save  ──────────────  AssetProject::rewrite_payload(mesh_id, &MeshAsset, deps)
        │                        ↓  SYNCHRONOUSLY
        │               vmesh::ensure_vmesh(project, mesh_id)
        │                        ↓
        │               thumbnail: nothing to do — the cache is keyed by content
        │               hash, so a rewritten payload MISSES and re-renders itself
        ▼
  assets://changed ───  every referencing view re-resolves
```

`DccState` is a fourth instance of a template that has been right three times
(`GraphState`, `MaterialState`, `PcgState`), which is why it is named here
rather than designed: a registry, a document map, a journal map keyed the same
way, and a counter for minting ids. `graph_close` frees both maps together; so
will `dcc_close`.

### The save is `rewrite_payload` + a SYNCHRONOUS `ensure_vmesh`, and that is a fix

`AssetProject::rewrite_payload` (`assets/mod.rs:338`) writes the bytes, recomputes
the `ContentHash`, rewrites the sidecar's dependencies, re-inserts the entry and
bumps the DB version. That is the correct and only door for editing an existing
asset's payload in place.

But **a `.inf_mesh` rewrite regenerates nothing today**, and that is a real
standing defect this phase must not walk into. `ensure_vmesh`
(`assets/vmesh.rs:254`) is the synchronous door that derives a mesh's
`.inf_vmesh` meshlet DAG, and it has exactly **one** non-test caller: the import
orchestrator (`assets/import.rs:329`). The project-open sweep does not use it —
it goes through the split `plan_sweep` / `build_vmesh` / `commit_vmesh` trio so
the long build phase runs outside the project lock.

Neither path is triggered by `rewrite_payload`. So today, a hypothetical
`.inf_mesh` rewrite would leave the derived `.inf_vmesh` describing the **old**
geometry, and the editor viewport — which renders real meshes through vgeom
since P18.3 — would keep drawing the stale surface with complete confidence. No
error, no warning, no visual tell: the exact failure shape this codebase keeps
paying for.

Hence: **P23.6's save calls `ensure_vmesh` synchronously, in the same unit of
work as the rewrite.** Synchronous, not queued, because the author pressed Save
and the next thing they will do is look at the viewport; a background derivation
means a window in which the level draws the previous mesh. `ensure_vmesh` is
already documented as "cheap and idempotent on the hit path" — `plan_vmesh`
answers from the database alone — so the cost when nothing changed is a hash
comparison.

Thumbnails need no invalidation call at all: `ThumbnailCache` keys on the
asset's content hash, so a rewritten payload is a cache **miss** by
construction. The cache self-invalidates. (It does need the orphan sweep it
already has, or the old PNG lingers on disk — that is a disk-space question, not
a correctness one.)

### Live-in-scene editing rides the existing pipeline

Nothing new is required for a scene that references the edited mesh to update:

```
rewrite_payload → AssetWatcher (debounced notify) → index_stale
   → ViewportState::refresh_asset_index(Target::All)      [P23.2a: broadcast]
   → EditorRenderAssets re-keys on the payload's CONTENT HASH
   → viewport redraws
```

The render key being the content hash (`render_assets.rs`, module docs and the
`the render key is the payload's content hash` comment) is what makes this work
without a single explicit invalidation: new bytes are a new key, the old entry
is unreachable, and a stale draw is impossible rather than merely unlikely.

## 3. Editing during play — what actually works, honestly

**Simulate: YES, and it works for a reason worth writing down.**
`SimSession::exit` (`simulate.rs:461`) reverts *the document* — it applies the
entry snapshot back and clears the deformation resource. **Asset edits are not
in the snapshot**, because they are not in the document. So a mesh edited while
Simulate runs survives Stop, exactly as an author would expect: they edited an
asset, not the running world. This is not a happy accident of the asset-scoped
ruling, it is the ruling's main payoff.

**Embedded PIE: IMPOSSIBLE today, and the memo says so rather than promising
it.** Two independent blockers, both documented in the code:

1. **The embedded player does not render asset meshes at all.**
   `runtime/inf-player/src/window.rs:539-542`: *"PIE streams no vmesh assets yet
   (a documented follow-up); asset meshes render as placeholder cubes in PIE
   until the payload carries the vmesh index."* Editing a mesh and watching a
   cube not change is not a feature.
2. **Embedding hides the editor viewport.** `embed_foreign` reparents the
   player's HWND into the viewport slot and `ShowWindow(hwnd, SW_HIDE)`s our own
   child (`win32.rs`). While embedded PIE is running there is no editor viewport
   on screen to update.

So the supported live-edit paths for P23 are **Simulate** and **"Play in New
Window" PIE** (which leaves the editor viewport visible and running, though the
player window itself still draws cubes for asset meshes until the vmesh-in-payload
follow-up lands). This is a ledger item, not a design.

### And one more the author will meet: Ctrl+Z in a torn-off editor does nothing

P23.2a routes undo by focused panel, and a detached panel window reports its
pointer-downs to the main window (`panel://focus`) so the routing can aim at it.
But a detached window is a **lean bootstrap** — by design it skips everything
one-window-assuming in the main `App`, including `installKeybindingListener`.

So an author who tears the Model Editor (or the Material, Blueprint or PCG
editor) into its own OS window and presses Ctrl+Z there gets **nothing at all**:
no undo, no message, no clue. What the focus report fixes is the *main* window's
aim — press Ctrl+Z back in the main window and it reaches the torn-off panel's
document, which is the improvement — but the shortcut does not work where the
author's hands are.

Recorded here because it is a **shipped limitation, not an implementation
note**. Fixing it means giving `PanelWindowApp` its own keybinding listener with
a scope narrowed to the panel it hosts, which is a real design question (which
of the shell's global chords should a panel window own? Ctrl+S? the palette?)
and not a line of code.

## 4. Undo for meshes — and why meshopt never enters the journal

`GraphJournal` is a **whole-snapshot** journal: it stores complete `Graph`
documents and undo is "restore the previous one". That is right for a node graph
(a few hundred nodes of `BTreeMap` + `Vec`) and wrong for a mesh, where a bevel
on a 100k-vertex model would snapshot megabytes per click.

**Plan for P23.3: an OP journal.** Each modelling operation is a value
(`Extrude { faces, distance }`, `LoopCut { edge_ring, cuts }`, …), the journal is
a `Vec<Op>` plus a cursor, and undo is *replay from the last checkpoint*. This
is the same shape the roadmap already names ("deterministic replay is both the
undo/redo story and the test story, mirroring `GraphJournal`"), and it buys
three things at once: bounded memory, a property-test harness for free (replay
must be a pure function of ops), and a serialization story if a session ever
needs to survive a crash.

Checkpoints keep replay bounded: a full snapshot every N ops, so undo cost is
O(N) rather than O(session).

**LAW, stated before it can be violated: `meshopt` must NEVER appear in the op
journal.** The P18 finding stands — *meshopt is NOT cross-platform*; its output
differs between `x86_64-msvc` and `aarch64-apple-darwin` from provably identical
input (ROADMAP §12, P18: 138 176 B vs a different byte count, meshlet counts
apart by several). An op journal is a **deterministic replay** structure: if any
op's effect depends on meshopt, then replaying the same journal on two machines
produces two different meshes, and every property test that says "replay is a
pure function of ops" becomes a test that passes on one developer's machine.

Concretely:

* **Optimize at EXPORT only.** `meshopt` runs where it already runs — inside
  `build_vmesh`, deriving the `.inf_vmesh` from a finished `.inf_mesh` — and
  never inside the kernel, never as a modelling op, never as an implicit cleanup
  pass after an op.
* The half-edge kernel's own operations (weld, dissolve, triangulate) are
  hand-written and deterministic, or they do not go in.
* P23.3's property tests state this positively: replay of a journal must produce
  a byte-identical `MeshAsset`, which is exactly the assertion meshopt would
  break.

## 5. The viewport decision — offscreen-PNG interactive first, **with measurements**

**Ruling: the Model Editor's preview is offscreen-rendered PNG, driven
interactively, and the native second viewport is a fast-follow (P23.2b).**

The alternative — a second `EngineHost`, i.e. a second wgpu `Device` on a second
thread with a second native child window — is not rejected, it is *sequenced*.
It carries device-loss, focus and airspace risk that a modeling kernel batch
should not be paying for at the same time as it is inventing a half-edge
structure. P23.2a shipped the enablers that make it a fast-follow rather than a
redesign (§6).

The ruling needed a number, and **no prior measurement existed anywhere in this
repository**. P23.2a extracted `PreviewSession` from `thumbnail/scene_render.rs`
— a reusable renderer that owns its target, depth buffer, sphere buffers, camera
uniform and a hash-keyed pipeline cache, so a camera move writes 144 bytes
instead of rebuilding six GPU objects — and measured it.

### Measured, on this machine

RTX 4070 Ti / Windows 11, `cargo test -p inf-editor-core --lib
preview_session_cold -- --nocapture`, best-of-5 per figure, three runs, spread
under 10%:

| | 256² | 512² |
| --- | --- | --- |
| process-cold (first render in the process) | ~19 ms | — |
| session-cold (new session, warm process) | ~1.5 ms | ~1.7 ms |
| **warm re-render (camera only)** | **~0.09 ms** | **~0.34 ms** |
| PNG encode, default deflate | ~8.5 ms | ~22.9 ms |
| PNG encode, `encode_png_fast` | ~0.34 ms | ~1.98 ms |

**The ruling is validated, with room to spare.** The stated bar was ~10 ms for a
warm 512² re-render; the measurement is **0.34 ms — thirty times under it**. An
orbit is not remotely GPU-bound.

**The finding that matters is what the frame is NOT.** At 512² the *render* is
0.34 ms and the *default PNG encode* is 22.9 ms: **98% of an offscreen frame is
deflate.** A measurement that stopped at `read_rgba` would have reported a
gloriously fast preview and shipped a panel that orbits at 43 fps for reasons
nobody could find. `encode_png_fast` (Fast compression + `NoFilter`, identical
pixels, ~12× the bytes) takes 512² to 1.98 ms, so warm + encode is ~2.3 ms.

Consequences recorded as decisions:

* The **disk thumbnail cache keeps default compression** — it encodes once and
  the file lives for ever under its content hash, so the bytes are worth
  compressing hard.
* **`material_compile` also keeps default compression.** It fires at *edit*
  rate, and at 256² the trade is 8 ms of CPU against ~230 kB more base64 through
  the webview bridge. The fast door exists for *orbit* rate, where that trade
  inverts.
* The Model Editor (P23.4) uses `encode_png_fast`, at 256² by default, and
  512² is affordable if the panel is large.
* If a future preview ever needs more than this, the next lever is **not** a
  faster encoder — it is skipping the encode entirely (raw RGBA over a Tauri
  channel into an `ImageData`), and after that the native second host.

There is also the honest caveat the numbers cannot cover: base64 + IPC +
`<img>` decode in WebView2 are not measured here, because they are not
measurable from a Rust test. They are bounded by the payload size, which is why
the byte counts are in the table — and at 512² the bound is **not comfortable**:
`encode_png_fast` produces 1 049 236 B, which base64 inflates to ~1.4 MB per
frame, so a 30 fps orbit pushes **~42 MB/s of string through the webview
bridge**. The GPU cost says the panel is free; the transport says a 512² orbit
is the thing to measure first when P23.4 has something to measure. 256²
(262 488 B → ~350 kB, ~10 MB/s at 30 fps) is the default for that reason, and
if 512² is wanted the fix is the one already named: raw RGBA over a Tauri
channel into an `ImageData`, skipping both the encode and the base64.

`PreviewSession` is deliberately not a Model-Editor-only object: `Thumbnailer`
holds one, so `material_compile`'s preview got the caching for free and the seam
has a real consumer *before* the panel that was designed for it exists.

## 6. Multi-viewport: 2a is shipped, 2b is deliberately not

**P23.2a (shipped in this batch)** is the pure-refactor half:

* `ViewportState` is a keyed `BTreeMap<String, ViewportHandle>` with a
  well-known `PRIMARY_VIEWPORT`. All 31 resolution points — 17 commands and 14
  cross-module pushes — now name `Target::Primary` or `Target::All`
  **explicitly**, and which one is a decision with a reason at each site
  (broadcast for facts about the project or the level; Primary for the PIE embed
  slot).
* **The store hoist**, which is the real refactor and fixes a latent defect: the
  shared carve store and the Simulate fracture map were created *inside*
  `inf_viewport::spawn` and held by the `ViewportHandle`, making them
  per-viewport. A carve made through a second viewport would have landed in a
  store `scene_save` never reads, and the level would have saved without it,
  silently. They are now `commands::SharedStores`, created once per process and
  passed into `spawn`; the save path, the autosave note and both Simulate
  publishes resolve them with no viewport in hand. This also retires the P21.2
  poisoned-outer-handle hazard outright — there is no outer handle left to
  poison, so a panicked viewport thread can no longer make unsaved carves *look
  absent*.
* Events carry a viewport id; `ViewportPanel` is a registered panel type (it was
  hard-mounted at `App.tsx:229` — survey breaker #7).
* **The airspace refcount's default acquisition is window-wide** — every
  attached viewport, not one. This was the audit's find and it is the same
  mistake as the store hoist, one layer up: `Target::All` existed on the Rust
  side, but the frontend primitive could only ever name a single viewport, so
  the moment viewport #2 existed every menu, dialog and drag ghost in the shell
  would have been painted over by it while the scene viewport politely hid.
  `acquireViewportOverlayFor(id)` remains for the panel-local overlay that does
  not exist yet, and a viewport attaching *while* an overlay is open now comes
  up hidden (opening a second viewport with the palette up must not punch a hole
  through it).
* **One lock rule, no exception**: never hold the scene document and the carve
  store at the same time. The rule is *no overlap*, deliberately **not** an
  acquisition order — the three sites that touch both genuinely differ in which
  they take first, and an earlier comment calling it "document first, volumes
  second" described one site and would have led a future author to "fix"
  `scene_autosave` into the very overlap the rule forbids.
  `sim::overlay_sim_carves` was the one real exception (it held the *store*
  across the *document*, both live for a whole loop — the classic two-lock
  deadlock shape). It survived because the store used to hang off a
  `ViewportHandle` and was awkward to reach; the hoist makes it one `try_state`
  from any command already holding the document, so it now snapshots its
  entity→asset bindings under the document and releases before touching the
  store.

**P23.2b (fast-follow, deliberately NOT this batch)** is the native second
`EngineHost`. Two reasons for the sequencing, and one hard constraint:

1. **A second host is a second wgpu `Device`.** Device loss, adapter selection,
   and the P2 `is_lost()` recovery path all become two-instance problems, and
   the editor's floating-origin rebase is currently driven by one camera.
2. **The DCC preview does not need it** (§5): 0.34 ms at 512².

3. **HARD CONSTRAINT — the DCC projector must never enter the projector mirror
   set.** `tests/projector_mirror.rs` pins nine `EngineHost` functions
   character-for-character against their `inf-player` twins
   (`project_voxel`, `project_water`, `project_sky`, `project_fracture`,
   `project_deform`, `project_debris`, `push_scatter`,
   `push_biome_population`, `skinned_mesh_data`), plus positional pins on
   `sync_voxels` and `fixed_step`. That gate exists because the editor and the
   shipped player must project the same world the same way — it is the
   structural half of "PIE == shipping".

   A DCC viewport projects **an edit mesh, not a world**. There is no player
   twin for it, and there never will be. If a second host's projection were
   written into `host.rs` beside the mirrored functions, the mirror set would
   either have to grow an exemption list (which is how a mirror stops mirroring)
   or gain a phantom "player" side that exists only to satisfy a test. So: the
   DCC projector lives in its own module with its own entry point, and
   `host.rs`'s mirrored functions are not touched by P23 at all.

## 7. The user requirements this has to satisfy

From the product side, three things were asked for. Each is answered by the
asset-scoped ruling rather than by new machinery:

* **Drag-and-drop modular assembly.** Kit pieces are mesh assets; the Content
  Drawer already drags them to the viewport and `scene_spawn_asset` places a
  real, selectable, saveable entity. Modelling a kit piece and assembling with
  it are two different documents, which is precisely why the session is
  asset-scoped: the piece can be re-edited without touching any of the levels
  that placed it, and every placement updates.
* **Live edit during play.** Answered in §3 — Simulate yes (because
  `SimSession::exit` reverts the document and an asset edit is not in it), new-
  window PIE yes, embedded PIE no and here is why.
* **Zero import/export friction.** The kernel imports from and exports to
  `MeshAsset`, which is the same type the glTF importer produces — so "model in
  engine" and "import from Blender" converge on one representation, and there is
  no in-engine format an author could get trapped in. `inf-mesh` gaining a
  writer is P23.3's first deliverable for exactly this reason.

## 7a. Addendum, P23.3 — the kernel's answers

The kernel landed in `crates/inf-dcc` (2026-08-05). Two things this memo left
open are now decided, and one of its rulings got sharper:

* **Seams live on the half-edge** (§8's first ledger item, answered). Positions
  are on vertices; UV and an *optional authored* normal are on **corners**, i.e.
  on face-side half-edges; edge sharpness is on the twin pair. No side table.
  That is what lets `from_mesh_asset` weld a `MeshAsset`'s split vertices back
  into topology — with a tolerance of **exactly zero**, because an epsilon weld
  is a modelling operation wearing a reader's clothes — without averaging away
  the attributes the split existed to carry. P23.5's UV work inherits corners
  that already exist rather than inventing a parallel store.
* **The op journal is `base + Vec<Op> + cursor`** (§4, built). Checkpoint every
  32 ops, at most 8 retained *nearest the cursor*, and an undo that lands on a
  boundary stores the mesh it just computed so walking backwards stays cheap.
  `SessionSave` persists in bincode and JSON; `restore` replays the redo tail as
  well as the applied prefix, which is what makes `undo`/`redo` infallible
  afterwards rather than optimistic.
* **§4's meshopt LAW held, and is now enforced by reading the source.**
  `tests/determinism_law.rs` greps the crate for `meshopt::`, `std`
  transcendentals, hash containers and stray `f32`, because the claim is about
  *two machines* and nothing in a test process compares two machines. The one
  sanctioned call is `inf_mesh::optimize` in `export.rs`, behind
  `ExportOptions::optimize`, off by default.

One consequence the memo did not anticipate, recorded here because P23.6's save
path is the thing that meets it: **a kernel mesh can be legal and still not
survive a write/read round trip.** Two distinct vertices at the same position
(the kernel distinguishes them; the exact weld fuses them) and an n-gon
triangulation diagonal that duplicates an edge elsewhere in the mesh both
produce an asset the reader refuses as non-manifold. The writer avoids the
second where it can and *counts* both as advisories
(`ExportReport::coincident_vertices` / `reused_diagonals`) — the P16 doctrine,
because the alternatives are nudging the author's geometry or refusing to save a
legal intermediate state.

## 7b. Addendum, P23.4 — the panel's answers

The modelling ops, the selection model and the Model Editor landed
(2026-08-05). Four things this memo left to the batch are now decided:

* **The overlay is CPU-composited, not a second GPU pipeline** (§5 named the
  preview path but not what draws on it). The reason is not cost: picking has to
  be CPU — there is no sub-object id buffer and §5 rules the viewport's ID pass
  out of this path deliberately — so a GPU line pass would compute the
  *highlight* in a vertex shader and the *hit* on the CPU, two answers to one
  question differing exactly at the sub-pixel margins a user complains about.
  Composited through the same `Projector`, what lights up is what `pick` would
  have returned, by construction. Occlusion is a bonus: a half-edge mesh knows
  both faces of every edge, so an edge whose two faces point away is culled with
  a dot product and no depth buffer at all. **Honest limit**: that is back-face
  culling, not depth testing — a near edge behind another *part* of the same
  model still draws, and fixing it means reading the depth buffer back beside the
  colour.
* **The preview draws through the WRITER.** `tessellate` calls
  `inf_dcc::to_mesh_asset`, so the picture is the geometry the save will produce —
  its ear clipping, its corner splits, its derived normals. A private
  triangulator would be a second answer to "what is this mesh", and the two would
  disagree exactly where an n-gon is interesting.
* **The camera, the selection and the mesh all live in the backend.** The panel
  is a thin client because all three are questions the *generation stamp* has to
  arbitrate, and only the side that can compare stamps may answer them. Every
  command returns the document; the store replaces its state rather than patching
  it.
* **Soft select measures geodesic distance in metres**, through
  `inf_terrain::Falloff` rather than a second copy of the same five curves. A hop
  count would be unitless and mesh-density-dependent; a Euclidean ball grabs the
  far side of a thin wall.

And one consequence §7a predicted in outline and P23.4 met head on: **the third
face of the coincidence hazard**. §7a recorded that a legal kernel mesh can fail
to survive a write/read round trip, and named two shapes (the reader refuses it;
a diagonal duplicates an edge). The modelling ops make a third ordinary, because
they place new vertices a parameter away from existing ones: two `f64` vertices a
hair apart round to the **same `f32`**, the writer emits them at one place, the
exact weld fuses them, the triangles that used both are dropped as degenerate —
and the asset comes back *legal, smaller, and not the mesh that was saved*. Same
ruling, same counter: `ExportReport::coincident_vertices` is what the save path
surfaces, because nudging the author's geometry falsifies their model and
refusing the export makes extrude-then-drag unsaveable.

## 7c. Addendum, the P23.4 audit — the save's failure contract

§2 said the save is `rewrite_payload` plus a **synchronous** `ensure_vmesh`, and
gave the reason: a background derivation is a window in which the level draws
the previous mesh. What it did not say is what happens when the derivation
*fails*, and the first implementation answered that question by accident —
leaving new payload and stale DAG on disk **permanently**, which is the same
failure the memo opens by naming, with no window at all because it never closes.

**Ruling.** `AssetProject` has a lock, not a transaction, so the pair cannot be
made atomic. It can be made to have only good failure states:

* the derivation succeeds → (new payload, new DAG);
* the derivation fails → **the stale DAG is removed** → (new payload, no DAG).

"No DAG" is a state the renderer already handles: `resolve_vgeom` misses and the
entity falls back to a placeholder. Visibly wrong beats confidently wrong, and
the next save or the project-open sweep repairs it. If the removal *also* fails —
the only reachable state in which something can still draw the previous geometry
— the error names it (`SaveError::Torn`) rather than hiding it.

**And the verdict is checked against the filesystem, not the database.** This is
the whole of the re-audit's M-1 and it is worth stating as a rule rather than a
fix: `AssetProject::delete` drops both `remove_file` results and returns `Ok`
unconditionally, so the removal helper's `is_ok()` was **always true** — a
database condition that every caller read as a filesystem one. The save's error
therefore told an author whose `.inf_vmesh` was held mapped (a real Windows case)
that the stale DAG had been removed and the mesh would draw as a placeholder,
while the file sat there being found by `resolve_vgeom` and drawing the previous
geometry. `Torn` is now reachable, and tested with a handle opened without
`FILE_SHARE_DELETE`. LAW: **a bool that reports on the filesystem must ask the
filesystem.**

Two consequences worth recording:

* **`VmeshDerivation::Skipped` is not an error, and was the leak.** A mesh edited
  below the virtualization threshold derives nothing, correctly, and kept the DAG
  describing the mesh it used to be. The save removes it.
* **The save lives in Ring 1** (`inf_editor_core::dcc::save_mesh_session`), not in
  the command. A `#[tauri::command]` cannot be driven from a test, and the first
  gate for this proved the *pattern* by inlining the same two calls — so deleting
  the derivation from the product failed nothing at all. Both the command and the
  gate now go through one function. LAW: **a gate that inlines the code it is
  gating is a copy, not a gate.**

## 7d. Addendum, P23.5 — the brush, the gizmo and the UV half

Sculpt, the component gizmo and the UV pipeline landed (2026-08-05). Six things
this memo left open are now decided.

* **A stroke is an op, not a transaction.** §8 left "UV/sculpt data model" to
  this batch, and the sculpt half's answer is structural: `inf_terrain`'s brush
  needs `Stroke::begin`/`add_dab`/`finish` because *its* undo unit is a
  `HeightDelta` several dabs accumulate into. This journal's atom is already an
  `Op`, so the adaptation is to put the whole gesture **inside** one —
  `Op::Sculpt` carries every dab centre of one mouse-down→up drag. One journal
  entry, one undo step, and the replay story comes free because a stroke is
  data. The dabs are arc-length resampled *before* they reach the op, so replay
  does not depend on the resampler either.
* **The gizmo is the same widget, not a second one** (§8's first ledger item,
  answered). `inf_render::gizmo` is pure interaction math over a `view_proj`, so
  the DCC reuses `pick_axis`'s analytic 11-pixel hit test and
  `GizmoDrag::update`'s deltas verbatim. What is *not* reused is
  `build_geometry`, which emits `DebugDraw` lines into a GPU pass this panel does
  not have — so the handles are drawn by the CPU compositor, and a gate asserts
  every painted pixel is a pixel the picker answers for. §6's hard constraint
  holds unchanged: nothing in `host.rs` was touched and the DCC projector is not
  in the mirror set.
* **One door for the numeric tool and the dragged handle.** Both produce a
  `VertTransform` and both go through `dcc::transform_ops`. The alternative —
  two code paths kept in step by a test — is the shape §7c's LAW already
  condemned ("a gate that inlines the code it is gating is a copy, not a gate"),
  one layer up.
* **The orphan-settler doctrine reaches gestures.** A pointer-up is not
  guaranteed: the panel can close, the tool can change, a detached window can
  lose capture. Every journal-touching command settles the pending drag first.
  **`dcc_close` abandons, deliberately** — it frees the `MeshSession` in the same
  call, so an op applied first could never be undone, saved or seen; a settle
  there is the same loss with a wasted `Mesh::transact` in front of it. Escape
  abandons too, because Escape means "no" and not "commit then undo". Both
  directions are tested, so neither can quietly become the other.
* **The live-drag side channel is a scratch clone, and it was measured.** §5's
  `PreviewCache` keys on the journal generation, and an uncommitted drag
  deliberately does not move it — so a drag needs a second channel. v1 applies
  the pending ops to a clone and re-tessellates, keyed on the drag's own shape.
  Debug build: 26 v → 0.17 ms committed / 0.11 ms scratch; 1 538 v → 8.6 / 9.1.
  **The clone and the stroke are free; the tessellation is the whole cost.** The
  stated limit is that this will not hold an interactive rate at a hundred
  thousand vertices, and the next lever is displacing the cached vertex buffer in
  place — *not* displacing on the GPU, which would put the drawn surface and the
  pickable surface back into disagreement (§7b's first ruling).
* **The unwrap journals its RESULT, and the solver is not part of the op.**
  `Op::Unwrap` carries the computed per-corner UVs; replay does not re-solve. The
  solver is deterministic today — `BTreeMap`-ordered `f64` over `+ - * /` and
  `sqrt`, a fixed-iteration CG, no transcendental — and that is *still* not a
  reason to define an op as "whatever this build's solver says". §4's meshopt LAW
  is about a journal being replayed by a different build than wrote it; applying
  the same reasoning to a solver that has not yet been improved is the cheap
  version of learning it twice.

And one number this memo should carry, because it constrains a whole class of
future work: **`inf_math::psin64` is accurate to ~5.7e-8, and that is enough to
make a rotation not a rotation.** The raw polynomial's `s² + c²` is not 1, so
Rodrigues built from it is a rotation composed with a slight scale — a
quarter-turn of a 1 m vertex came back 56 nanometres short, and repeated gizmo
drags would have shrunk a selection with nothing to tell the author why.
Renormalizing the pair (`sqrt` is exactly specified, so still bit-portable) buys
an exact isometry and leaves an *angle* error under 6e-8 rad. Any future feature
that composes many portable-trig rotations needs the same treatment.

## 7e. Addendum, the P23.5 audit — three things worth keeping

The audit's fixes are in the ROADMAP's completion block. Three of them changed a
*rule* rather than a line, and belong here.

* **A gate that reads source must read a SCOPE, not a file.** The
  replay-the-result gate banned the string `uv::unwrap` anywhere in `ops.rs` and
  checked its positive halves with `.contains()` on the whole file. Both halves
  were defeatable and one was defeated: a `pub fn recompute` wrapper re-solved on
  replay with every test green, because the ban never saw the new spelling and the
  positives were satisfied by unrelated lines. The rule that comes out of it is
  narrow and reusable — **scope the read to the construct (brace-balance from its
  own signature) and ban the MODULE rather than the function** — and it is now how
  both source gates in this phase are written.
* **A doctrine spread across N call sites needs a table, not N sentences.** The
  settle/abandon rule was eleven hand-written statements over twenty commands, and
  its cited test did not exist. What replaced it is a hand-written policy table
  plus a source read that fails when a command is missing from it — so the
  *default for a new door is "fails the build"* rather than "does not settle".
  The same shape fits any rule of the form "every X must do Y unless it says
  otherwise", and this codebase has several.
* **A measurement can refuse a prescribed fix, and should.** The audit prescribed
  reducing the rotation angle mod 2π. Measured across fifteen decades it improves
  neither the collapse (the bound and the degenerate-pair refusal close that) nor
  the accuracy (at 1e12 it is *worse*: 5.0e-5 against 3.4e-5), because at those
  magnitudes the error is the input's own resolution and no reduction recovers a
  digit that was never stored. It is not in the tree, and the table is in the code
  at the point where someone will next be tempted to add it. **`inf_math::psin64`
  is a degree-11 polynomial; below 2^52 its pair never degenerates (worst
  `|s, c|` = 0.968, swept), and past ~2e16 it is exactly zero — so the honest
  interface is a bound plus a refusal, not a fold.**

## 7f. Addendum, P23.6 — the chain, executed, and what it cost

The asset round-trip closed the phase (2026-08-06). Three things this memo asserted
are now *measurements* rather than arguments, and one thing it did not
anticipate is the phase's most valuable finding.

* **§3's claim about Simulate is executed, not reasoned about.** A save spliced
  into the middle of a live `SimSession` leaves the step trace **bit-identical to
  a control run**, the editor re-keys mid-run, `exit` restores the document byte
  for byte, and the asset keeps the edit
  (`inf-editor-core/tests/dcc_edit_during_simulate.rs`). The chain was also
  *read* rather than assumed: the watcher's indexed-extension set covers a
  `.inf_mesh` **rewrite** as well as an insert, and neither the background asset
  tick nor `dcc_save` names any play state. The one link a test cannot execute —
  a `#[tauri::command]` — is held by a source-scope gate that requires the
  refresh push at **statement level**, so a `if !sim_is_running()` around it
  fails that gate and nothing else. §2's "the scene sees the edit for free"
  turned out to be exactly true, which is worth recording precisely because it
  was the kind of claim that is usually not.
* **The bake exists, and §7's "kit pieces are mesh assets" now has a producer.**
  `inf_editor_core::bake` collapses a P19 scattered building into ONE
  `MeshAsset`, which is what lets a `Destructible` fracture it (the P22 ledger
  item). The finding that shaped the module: a `ScatteredSolid` is **always an
  oriented box** and carries **no material identity** — the kind lives on the
  parallel instance list, and the two lists are index-aligned *only* for
  buildings. So there are two doors and the generic one **refuses an unaligned
  pair as a value** rather than guessing.
* **§7a's round-trip hazard is not hypothetical, and the ordinary case is a
  bevel.** §7a recorded that a legal kernel mesh can fail to survive a write/read
  round trip and named the worst case ("an edge used twice, and the read is
  refused"). P23.6's gate models the prop the phase's own sentence asks for and
  hits it: `Op::BevelEdges` on a cap that a `MeshAsset` carries as two triangles
  sharing a diagonal leaves collinear boundaries and a coincident pair per
  corner, and the saved prop **cannot be re-opened**. Attributed op by op — the
  extrude and the loop cut are clean — and pinned three ways (6 un-earable faces,
  8 coincident vertices, 101 of 106 vertices with no usable tangent, plus the one
  chart of eighteen that does not converge). The advisory doctrine did its job:
  the save *says* the vertices will fuse. The bevel is the defect, and it is the
  first thing to fix in the next DCC batch.

And one consequence of the whole phase that belongs beside §5's measurements
because it is about what *ships* rather than what is authored: **a hand-modelled
prop is a few dozen triangles, and the cook's `[vgeom] min_triangles` is 2048.**
`RenderScene` has one door for non-primitive geometry, the editor's
`ensure_vmesh` derives from one triangle, and the result is a prop that looks
right for the whole time it is being modelled and ships as a placeholder cube.
The advisory exists and names the fix; the DCC's entire output class lives in
that gap, and the phase gate asserts it rather than dodging it.

## 8. Ledger — what this memo does NOT decide

* ~~**The gizmo on component selections.**~~ **Built in P23.5** (§7d): the same
  widget, the same `pick_axis`, one `transform_ops` door shared with the numeric
  tool.
* ~~**UV/sculpt data model** (P23.5).~~ **Answered in full**: seams on the
  half-edge twin pair (P23.3 §7a for the storage discipline, P23.5 for the flag
  itself), the stroke as one op, the unwrap as its own result (§7d).
* **UV *editing* in the 2D view** is still open. P23.5's UV panel is
  read-mostly: it draws the charts, the seams and the shared selection, and
  seam marking happens in the 3D view. Dragging a vertex in UV space needs a
  pick in UV pixel space and a per-corner move op, and it is a remainder rather
  than a design question.
* **Multi-object edit.** v1 edits one mesh. Whether a session can hold several
  is a UI question that the `docs: BTreeMap<Id, Doc>` shape already permits.
* **The vmesh-in-PIE-payload follow-up** (blocker 1 in §3) is not P23 work; it
  is what would make embedded-PIE live editing meaningful, and it is tracked
  where it was raised.
* **Whether `encode_png_fast` is enough**, or the preview eventually needs raw
  RGBA over a channel. Measured when a panel exists to measure.
* **The State Machine editor still has no undo** — surfaced by P23.2a's routing
  registry, which claims the scope and says so rather than silently undoing the
  scene. It is now visible to the user, which is the first step to fixing it.
* **Ctrl+Z inside a detached panel window does nothing** (§3, last part). Needs
  a scoped keybinding listener in `PanelWindowApp` and a decision about which
  global chords a panel window owns.
* **A lost preview device stays lost** (P23.2a audit — M4). `Thumbnailer`
  resolves its `GpuContext` once and caches the result in `GpuState`; there is
  no `is_lost()` check and no rebuild, so after a driver TDR — a real event on a
  machine that also runs the interactive viewport — every material preview
  reads "No preview" for the rest of the session, and only a restart fixes it.
  The lenient error handler installed in P23.2a keeps the editor *alive* through
  that, which is the difference between a degraded panel and a lost level, but
  it does not recover. The fix shape is the one the viewport host already uses:
  check `GpuContext::is_lost()` before a render, drop the context and the
  `PreviewSession` together on a loss, and let `ensure_gpu` rebuild both on the
  next call (the session holds nothing that outlives its device, so a drop-and-
  recreate is the whole of it).
* **There is no `viewport_detach`.** `viewport_attach` inserts into the keyed
  map and nothing ever removes an entry: a `ViewportHandle` lives until the
  process exits, holding its thread, its native child window and its wgpu
  surface. Harmless while the only viewport is the shell's permanent one — it is
  attached once and wanted for ever — and an unbounded native-window factory the
  moment P23.4 opens and closes Model Editor tabs. The command is the missing
  half of the keyed map, and `ViewportHandle::destroy` (which already exists on
  all three platforms) is what it calls.
