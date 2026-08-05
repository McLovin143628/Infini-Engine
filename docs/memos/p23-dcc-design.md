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
the byte counts are in the table.

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
* Events carry a viewport id; the airspace refcount is keyed per viewport;
  `ViewportPanel` is a registered panel type (it was hard-mounted at
  `App.tsx:229` — survey breaker #7).

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

## 8. Ledger — what this memo does NOT decide

* **UV/sculpt data model** (P23.5). Whether seams live on the half-edge or in a
  side table is a kernel question and belongs with the kernel.
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
