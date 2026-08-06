# Infinity Engine — Engineering Roadmap

**Version 1.0 · Supersedes the preliminary blueprint (v4.0)**

Infinity Engine is a commercial-grade, next-generation game engine: a native Rust core with a
built-in Tauri v2 + React editor. It targets 2D, 2.5D, and 3D games of every genre,
with a professional pipeline from first asset import to shipped, packaged builds. Developers write
gameplay in **real Rust** (built-in IDE) or in **Infinity Blueprints** (node graphs) —
interchangeably, because graphs transpile to real Rust source and stay bidirectionally in sync.

---

## Table of contents

1. [Vision & product principles](#1-vision--product-principles)
2. [Architecture overview](#2-architecture-overview)
3. [Asset system & file formats](#3-asset-system--file-formats)
4. [UI design language](#4-ui-design-language)
5. [Risk register & Phase 0 spikes](#5-risk-register--phase-0-spikes)
6. [Phase roadmap](#6-phase-roadmap) — 16 phases → sub-phases → batches
7. [Technology matrix](#7-technology-matrix)
8. [Verification & CI strategy](#8-verification--ci-strategy)
9. [Platform strategy](#9-platform-strategy)
10. [Porting inventory](#10-porting-inventory)
11. [Post-plan status — UE-Parity Wave 1](#11-post-plan-status--ue-parity-wave-1-2026-07-22-complete)
12. [Next-Gen Wave — Phases 16–25](#12-next-gen-wave--phases-1625-planned-2026-07-31) — 10 phases

---

## 1. Vision & product principles

1. **Commercial-grade, not a demo.** Every phase ends in a demonstrable, shippable vertical
   slice. Quality bars (perf budgets, test suites, UX polish) are enforced from Phase 0.
2. **Two ways to code, one truth.** Blueprints transpile to real Rust; hand-written Rust lifts
   back into graphs where possible. Preview (interpreter) and shipped behavior (compiled Rust)
   are parity-tested in CI.
3. **Data-Oriented Design.** ECS world, archetype storage, GPU-driven rendering, zero-copy
   asset loading. No OO scene-graph bloat.
4. **Planetary scale by construction.** 64-bit world coordinates with floating-origin rebasing
   from the first line of renderer code — retrofitting f64 is a rewrite, so it is never deferred.
5. **The editor is a product.** Infinity Engine must feel like a next-generation refinement of
   UE5 — familiar mental model, modernized execution — from the first milestone.
6. **Honest engineering.** Research-grade features (virtualized geometry) are sequenced so the
   engine ships without them; known platform constraints (Wayland embedding, console NDAs) are
   documented, not hand-waved.

## 2. Architecture overview

### 2.1 The three-ring rule

| Ring | Contents | May depend on |
|------|----------|---------------|
| **Ring 0 — Engine** (`crates/`) | Simulation, rendering, assets, scripting runtime | Nothing UI- or Tauri-related. Compiles for shipped games and (later) consoles. |
| **Ring 1 — Editor core** (`editor/crates/`) | Project model, asset DB queries, undo/redo, thumbnailer, build orchestration, native viewport host | Ring 0. Still Tauri-free → headless-testable. |
| **Ring 2 — Apps** (`editor/studio/`, `runtime/`, `tools/`) | Tauri Studio app, standalone player, packager, CLI | Rings 0–1. The only ring that names Tauri/winit. |

### 2.2 Workspace layout

```
infinity_engine/
├── Cargo.toml                # single workspace; [workspace.dependencies] pins all versions
├── crates/                   # ─ Ring 0
│   ├── inf-core        ids, errors, tracing, frame clock, job system (rayon pool + flume; §2.5, built P7.0)
│   ├── inf-math        glam-based; WorldPos(DVec3) f64 world / f32 render split, floating-origin
│   ├── inf-platform    HAL traits: window/surface, vfs, time, threads (console seam)
│   ├── inf-ecs         facade over bevy_ecs + bevy_reflect (sole crate naming bevy_ecs)
│   ├── inf-asset       .inf_* schemas, bincode payload + TOML sidecar, GUIDs, xxh3, asset DB, importers
│   ├── inf-scene       .inf_lvl format, transform hierarchy, .inf_act prefab instantiation
│   ├── inf-render      wgpu device/surface, render graph, pipeline cache, GPU-driven draws, ID picking
│   ├── inf-render-2d   sprite batcher, tilemaps, 2D lights
│   ├── inf-mesh        import processing, meshopt; later meshlet building
│   ├── inf-vgeom       virtualized geometry (meshlet DAG, streaming, GPU culling) — Phase 13
│   ├── inf-material    material model; .inf_mat → WGSL codegen (naga-validated); .inf_tex → compute
│   ├── inf-terrain     quadtree/clipmap heightfield, GPU erosion, splat layers
│   ├── inf-pcg         .inf_pcg runtime: samplers, rules, scatter → GPU instance buffers
│   ├── inf-physics     rapier3d-f64 / rapier2d facade
│   ├── inf-audio       kira facade
│   ├── inf-anim        skeletal runtime, clips, blend spaces, state machines
│   ├── inf-input       action/axis mapping shared by editor and runtime
│   ├── inf-graph       generic node-graph DAG: model/compile/exec/derive/cache/registry
│   ├── inf-blueprint   blueprint semantics + tree-walking interpreter (editor preview)
│   ├── inf-transpile   syn/quote/prettyplease bidirectional graph ↔ Rust
│   ├── inf-hotreload   dylib host, #[repr(C)] vtables, state snapshot/migration
│   └── inf-runtime     the game loop; consumed by play-in-editor and the player
├── editor/
│   ├── crates/
│   │   ├── inf-editor-core   Ring 1 editor logic (headless-testable)
│   │   └── inf-viewport      native child-window host per OS, input capture, surface lifecycle
│   └── studio/               Tauri v2 app: src-tauri (commands) + src (React 18/19 + TS frontend)
├── runtime/
│   ├── inf-player            standalone winit binary; doubles as the PIE subprocess
│   └── inf-packager          cook + bundle per platform
├── tools/inf-cli             `inf new` / `inf cook` / `inf bindings`
├── templates/                blank-3D, 2D-platformer, first-person project templates
├── samples/                  dogfood projects — double as integration-test fixtures
└── docs/                     this roadmap, decision memos, per-subsystem design docs
```

### 2.3 Decided architecture (settled; do not relitigate casually)

1. **Viewport = native wgpu child window.** The engine renders into a real native swapchain
   window embedded in the Studio window, sitting *above* the webview inside a rectangular hole
   the React layout reserves. Win32 `WS_CHILD` HWND first, macOS `NSView`+`CAMetalLayer` second,
   Linux X11 reparenting third; native Wayland uses a frame-streaming fallback until a
   subsurface approach is proven. Consequence (the "airspace" rule): HTML can never draw on top
   of the viewport — all in-viewport overlays (gizmos, selection outlines, HUD) are
   engine-rendered.
2. **Editor-shell first.** The complete Studio shell (docking, content drawer, IDE, graphs,
   viewport) arrives early with a simple-but-real engine underneath; engine systems deepen phase
   by phase.
3. **Blueprints transpile to Rust.** One execution model in shipped games: compiled Rust. The
   editor uses a graph interpreter for instant preview; interpreter-vs-compiled parity is a CI
   gate. Generated code lives in the *user project's* cargo workspace so rust-analyzer treats it
   as first-class.
4. **Console-ready HAL.** `inf-platform` traits are the seam; console backends are private
   out-of-tree crates (see §9).

### 2.4 IPC & bindings conventions (adopted from reference projects)

- Every backend capability is a `#[tauri::command] async fn … -> Result<T, String>` in a
  per-domain module; the frontend calls only typed wrappers in `src/lib/ipc.ts` (never raw
  `invoke`). Events use namespaced channels: `viewport://rect`, `assets://changed/{id}`,
  `log://line`, `play://state`, `graph://sync/{id}`.
- All shared types derive `serde` + `ts-rs`; a `cargo test`-driven export writes committed
  TypeScript bindings; CI fails on drift.
- Tauri capability ACL stays minimal: no raw fs/shell plugin permissions — everything routes
  through audited commands.

### 2.5 Concurrency & parallelism model *(a first-class concern, not an afterthought)*

Multi-core parallelism is a headline reason to build the engine in Rust, so it is an owned,
scheduled deliverable — not an emergent property. The doctrine:

- **`inf-core` owns the compute job system** (Ring 0): a **rayon** worker pool sized to the
  machine plus **`flume`** channels for work hand-off and results. This is the single place
  gameplay/render/asset code reaches for data-parallelism (`par_iter`, scoped fan-out, task
  graphs). It is built as **P7.0** (its first real consumer is material-graph compile), replacing
  the placeholder facade the workspace layout has long described.
- **`tokio` stays in Ring 2** (editor IO: LSP, pty, file watching, dialogs). It never enters a
  Ring-0 hot loop — async is for IO concurrency, rayon is for compute parallelism.
- **The ECS runs a real parallel schedule.** `inf-ecs` enables `bevy_ecs`'s `multi_threaded`
  feature and drives systems through a genuine `Schedule` whose non-conflicting systems run on the
  job pool — the report's "systems are parallel functions over contiguous memory" made real. Stood
  up in **P9.1** with the runtime game loop.
- **Parallel *and* deterministic.** The fixed-step sim keeps its bit-determinism guarantee by
  requiring within-step systems to be order-independent (disjoint component access, or explicit
  ordering edges where they conflict) and by resolving structural changes at deterministic sync
  points. Determinism is proven by the replay harness (§8) running under the parallel scheduler —
  parallelism must never change the result of a step.
- **Data-race freedom is free.** Rust's borrow checker enforces at compile time the thread-safety
  the report describes; there is no runtime cost and no class of use-after-free/data-race bugs to
  chase.

Named hot loops to parallelize (each owned by the phase where it lives, not deferred wholesale to
Phase 15): ECS transform propagation + gameplay systems (P9), asset import + cook (P7.0 / P9.2),
material/texture graph compile (P7), meshlet DAG build (P13.1), GPU draw-command prep (P13), PCG
scatter (P10.5). Phase 15 keeps only the final *tuning/profiling* pass, not the initial build-out.

## 3. Asset system & file formats

Every asset = **bincode binary payload** (fast runtime load) + **TOML sidecar** (human-readable,
git-diffable metadata: GUID, schema version, dependencies, tags, import settings). Sidecar
emission is byte-deterministic. Every schema struct carries `schema_version` and a migration
function; old-version fixtures are load-tested forever.

| Extension | Asset | Contents |
|-----------|-------|----------|
| `.inf_lvl` | Level/Scene | ECS world snapshot, hierarchy, world settings |
| `.inf_act` | Actor Assembly (blueprint class) | component layout, physics bounds, graph document, links to generated Rust |
| `.inf_fn` | Function Library | reusable graph functions ↔ Rust methods |
| `.inf_mat` | Material Graph | PBR/layered shader graph → WGSL |
| `.inf_tex` | Texture Graph | procedural texture graph → GPU compute passes |
| `.inf_pcg` | PCG Graph | sampler/filter/scatter rules |
| `.inf_enum` | Enum | strongly-typed Rust enum with editor dropdown bindings |
| `.inf_struct` | Struct | strongly-typed Rust struct for data grouping |
| `.inf_table` | Data Table | tabular data, CSV/JSON import-backed |
| `.inf_anim` / `.inf_skel` / `.inf_sm` | Animation clip / skeleton / state machine | Phase 11 |

Asset database: GUID-keyed, xxh3 content-hashed, dependency-graphed, `notify`-watched. Imports
(glTF, PNG/EXR/HDR, WAV/OGG) are cached by content hash and processed in parallel.

## 4. UI design language

**Goal: "UE5, next generation."** Same mental model — users who know Unreal are instantly at
home — but visibly more modern and more disciplined.

- **Layout parity:** menu bar (File · Edit · Window · Tools · Build · Platforms · Select · Actor
  · Help), main toolbar (save, selection mode, add-actor, play/pause/stop/step), tabbed
  level/asset editors, viewport with its own toolbar (transform tools, snapping toggles, camera
  speed, Perspective/Lit dropdowns, view-mode flyouts), right-hand Outliner (search, type
  column, visibility eyes) over a Details panel, bottom status bar (Content Drawer toggle,
  Output Log, command console, save state, source control), slide-up **Content Drawer**
  (Favorites + folder tree | filter column | breadcrumb bar, virtualized thumbnail grid,
  Add/Import/Save All, "Dock in Layout").
- **Node editors:** dedicated tabs per asset (Blueprint, Material, Texture, PCG) with left
  preview/details column, center canvas (zoom badge, grid), right collapsible palette;
  searchable sectioned context menus with fly-out submenus; comment boxes; reroute nodes.
- **Next-gen refinements (where we surpass UE5):** an 8-pt spacing system and real typographic
  hierarchy (UE5 is cramped); one accent hue with semantic pin/wire colors; global command
  palette (Ctrl+Shift+P) covering every menu action; instant-filter everything; GPU-accelerated
  node canvas that stays 60 fps at 1,000+ nodes; first-run layouts per discipline (3D, 2D,
  scripting); animated-but-fast panel transitions (<120 ms); dark theme default with a real
  theme system (JSON themes → CSS variables) rather than one hardcoded skin.
- **Viewport navigation parity (zero learning curve):** RMB+WASD/QE flycam with scroll speed
  adjust; LMB-drag move/orbit behavior; MMB pan; Alt+LMB orbit / Alt+RMB dolly; F focus;
  Ctrl+number bookmarks. Matches UE5 hotkeys exactly.
- **Every panel dockable and detachable** to floating native windows across monitors, with
  zero-latency state sync over the IPC store bridge.

## 5. Risk register & Phase 0 spikes

Four go/no-go spikes run in Phase 0. Each is timeboxed, throwaway-quality, and ends in a
committed decision memo under `docs/memos/`.

### Spike A — wgpu child window inside Tauri v2 *(the hardest problem)*
Windows: `CreateWindowExW(WS_CHILD | WS_CLIPSIBLINGS)` parented to the Tauri HWND; wgpu surface
via raw-window-handle 0.6; pick DX12 vs Vulkan empirically. macOS: `NSView` + `CAMetalLayer`
(objc2), `contentsScale` synced to backing scale. Linux: X11/XWayland reparent via x11rb;
native Wayland → offscreen + frame-streaming fallback (documented, not parity). Rect sync:
React `ResizeObserver` → rAF-throttled `viewport://rect` IPC → `SetWindowPos` + debounced
swapchain reconfigure (letterbox between). Input: viewport child receives native input;
RMB-down → `SetCapture` + raw deltas + WASD polling; focus handoff events so React renders
focus chrome. Drag-drop: HTML ghost dies over the hole → IPC handoff, engine renders native
drop preview and raycasts on drop.
**Exit:** embedded spinning triangle on Windows + macOS; 60 fps splitter resize with no white
flash; correct at 100/150/200 % DPI and cross-monitor drag; flycam capture works.

### Spike B — bidirectional graph ↔ Rust
Doctrine: **the graph is the source of truth; Rust is a projection with a defined liftable
subset; hand edits outside the subset become opaque *snippet nodes*** — never data loss, never
a failed sync, and we never promise arbitrary-edit round-tripping. Item-level
`#[infinity::blueprint(id = …)]` attributes only (statement attrs aren't stable Rust). Node
identity is carried in generated *identifiers* (`let node_a3f2_out = …`) because identifiers
survive syn/prettyplease and comments do not. The `.inf_act` sidecar stores per-fn xxh3 of the
last-generated body; the file watcher parses changed fns and lifts them by canonical-shape
pattern matching; unmatched regions become snippet nodes holding verbatim source (original byte
spans, so comments survive inside snippets).
**Exit:** proptest suite — random graph → code → lift → graph isomorphism; regenerate
idempotence; ~30-case hand-edit corpus with expected outcomes.

### Spike C — hot-reload of game-logic dylibs
`#[repr(C)]` fn-pointer vtables; no Rust types cross the boundary. Components register as
(name, schema_hash, serialize, deserialize, tick). Reload = old dylib serializes state → new
one deserializes; mismatched schema hashes default new fields. **Never unload** old dylibs
(copy to hashed path to dodge Windows file locks; unloading Rust dylibs is UB-adjacent).
`catch_unwind` at every entry: a panicking script disables its system and reports to the Output
Log. Scoping: the interpreter is the primary iteration loop; hot reload is the compiled-preview
tier and may slip without blocking the roadmap.

### Spike D — play-in-editor process model
**Subprocess-first**: `inf-player` spawned with a cooked in-memory snapshot over a local
channel; crash-isolated (a script panic cannot destroy unsaved editor state); exercises the
real cook path so previewing never diverges from shipping. Its window embeds into the viewport
slot with Spike A machinery; "Play in New Window" is the v1 fallback. In-process **Simulate**
mode (physics/PCG tick, no game logic) arrives cheap and early.

### Standing risk register
| # | Risk | Mitigation |
|---|------|-----------|
| 1 | Wayland viewport embedding unsolved | X11 first; streaming fallback; no parity promise |
| 2 | Users expect perfect AST round-trip | Snippet-node model surfaced *in the product UI* |
| 3 | Virtualized geometry is research-grade | Sequenced last; engine ships without it (classic LOD fallback) |
| 4 | Hot-reload ABI fragility | Interpreter + subprocess PIE are the safety net |
| 5 | Console SDKs are NDA'd | HAL seams + private crates; no SDK code in this repo |
| 6 | Scope. This is a multi-year effort | Strict phase gates; every phase independently demonstrable |

## 6. Phase roadmap

Notation: **P4.2** = phase 4, sub-phase 2; batches are the numbered lists inside each sub-phase
— each batch is roughly one reviewable commit/PR. Critical path: **P0 → P1 → P2 → P3 → P4 → P6
→ P9**. P5 runs parallel to P3–P4; P7 overlaps P6; P10/P11 overlap after P9.

---

### Phase 0 — Foundation & risk spikes *(gate phase)*

**Goal:** a compiling monorepo, green CI on three OSes, and go/fallback decisions on the four
spikes. **Done when:** CI green (Win/mac/Linux); Spike A triangle demo recorded; Spike B
proptests green; memos A–D committed.

> **Status: COMPLETE (2026-07-19).** CI green on all three OSes (Rust matrix,
> cargo-deny, Frontend, bindings-drift jobs); all four spikes GO with memos in
> `docs/memos/`. Deferred to human/hardware, tracked but non-blocking: the
> 150/200% + cross-monitor DPI manual pass, a Spike A runtime pass on real
> macOS hardware (the mac port is compile-verified), and the demo recording.

- **P0.1 Repository & workspace scaffold** *(done)*
  1. Git repo, `.gitignore`, README, this roadmap. *(done)*
  2. Cargo workspace with every Ring 0/1 crate as an empty compiling `lib.rs` + rustfmt/clippy/deny config. *(done)*
  3. Studio app scaffold: Tauri v2 + Vite + React + TS + Tailwind; window opens with placeholder shell. *(done)*
  4. `tools/inf-cli` skeleton (`inf --version`); `runtime/` crate stubs. *(done)*
  5. ts-rs binding export harness + committed bindings directory. *(done — `inf_editor_core::ipc`
     types exported by `cargo test -p inf-editor-core --test bindings` into
     `editor/studio/src/bindings/`)*
- **P0.2 CI & quality gates** *(done)*
  1. GitHub Actions: fmt, clippy `-D warnings`, nextest, cargo-deny on 3 OSes. *(done —
     deny runs as its own ubuntu job; results are OS-independent since `deny.toml`
     pins the four target triples)*
  2. Frontend job: tsc, eslint, vitest. *(done)*
  3. Bindings-drift job (regenerate + `git diff --exit-code`). *(done — also fails on
     exported-but-uncommitted files)*
  4. PR template, branch protection notes, conventional-commit convention doc. *(done —
     `.github/PULL_REQUEST_TEMPLATE.md`, `docs/CONTRIBUTING.md`)*
- **P0.3 Spike A — embedded viewport** (see §5) — batches: Win32 child HWND + wgpu clear;
  rect-sync IPC + resize; DPI matrix; flycam capture; macOS port; memo. *(GO — memo
  committed; remaining manual passes: DPI matrix at 150/200% + cross-monitor drag, and a
  macOS run on real hardware — the mac port is compile-verified only)*
- **P0.4 Spike B — transpiler round-trip** — batches: node model fixture; codegen for
  Branch/Sequence/var/call/math kit; syn lifter; proptest harness; hand-edit corpus; memo.
  *(GO — memo committed)*
- **P0.5 Spike C — hot reload** — batches: vtable host + sample plugin; state snapshot/restore;
  panic containment; Windows file-lock dance; memo. *(GO — memo committed)*
- **P0.6 Spike D — PIE process model** — batches: player stub + local-channel snapshot
  handoff; crash-isolation test; embed-window experiment; memo. *(GO — memo committed)*

### Phase 1 — Studio Shell v1

**Goal:** the UE-refined shell, fully dockable, themed, with nothing real behind it yet.
**Done when:** dark-theme shell with dockable Outliner/Details/Content-Drawer/Viewport/Output-Log
placeholder panels, persisted layouts, detachable windows, command palette — demo-recordable.

> **Status: COMPLETE (2026-07-19).** All four sub-phases shipped and verified
> live on Windows (screenshots of shell boot, Content Drawer push-up with
> native-viewport resize, docked Output Log streaming real tracing output,
> command palette over a hidden viewport, layout persistence across restart).
> Notable decisions along the way: the Content Drawer *pushes the workspace
> up* instead of overlaying it, and shell overlays (menus, palette, dialogs,
> drag ghost) hide the native viewport via a refcounted `viewport_set_visible`
> guard — both consequences of the airspace rule; P2.1 explores flash-free
> alternatives (window-region cutouts / last-frame freeze). Human-only
> remainder: the phase demo recording.

- **P1.1 Shell chrome** *(done)* — 1. custom window chrome (decorations off, min/max/close, drag
  regions); 2. menu bar with full UE-parity menu tree (actions stubbed but enumerated);
  3. main toolbar + status bar; 4. theme system port (JSON themes → CSS variables, dark default).
- **P1.2 Docking system port** *(done)* — 1. port GeoCanvas dock core (workspace/group/region/splitter);
  2. drag layer + drop targets + tab strips; 3. floating panels; 4. detachable native windows
  with IPC store bridge; 5. layout persistence + named layout presets.
- **P1.3 IPC backbone** *(done)* — 1. typed `ipc.ts` + command registry convention; 2. namespaced event
  helpers; 3. zustand store architecture; 4. ts-rs pipeline wired to real shared types.
- **P1.4 Core panels (placeholder data)** *(done)* — 1. Outliner tree w/ search + type column; 2. Details
  panel w/ property-row primitives; 3. Content Drawer slide-up shell (tree + grid + filters);
  4. Output Log (tracing subscriber → `log://` events, severity filters, ANSI); 5. command
  palette + keybinding registry.

### Phase 2 — Viewport & renderer bring-up

**Goal:** productionize Spike A into `inf-viewport` + `inf-render`; UE-parity camera; picking
and gizmos. **Done when:** 10k cubes at vsync; select + translate/rotate/scale with gizmos;
correct on three DPI configs; golden-image tests running in CI.

> **STATUS: Phase 2 COMPLETE** (2026-07-20), verified live on Windows/Vulkan (RTX 4070 Ti).
> The Spike A viewport is productionized into `inf-render` (Ring 0 forward renderer) driven
> by `inf-viewport` through a shared `EngineHost`. Live demo: 10 407 shaded cubes at vsync,
> infinite dual grid + sky, RMB flycam / Alt-orbit / dolly, F-focus, camera bookmarks,
> click-select via ID buffer, orange selection outline, translate/rotate/scale gizmos,
> command-palette focus handoff from the native viewport. See the CLAUDE.md status block.
> Key decisions: reverse-infinite-Z depth for large-world precision; floating origin in
> `inf-math` (10 m-snapped rebase); analytic gizmo hit-testing + custom overlays over egui
> (`docs/memos/p2-overlay-egui-vs-custom.md`). Golden harness runs determinism + structural
> gates in CI (adapter-robust); strict pixel diffing is opt-in (`INF_GOLDEN_STRICT`) to avoid
> cross-adapter flakiness — a pinned-adapter (lavapipe) strict job is the follow-up.
> Human-only remainders (non-blocking): DPI matrix manual pass (150/200 %, cross-monitor),
> demo recording, Spike A macOS hardware run.

- **P2.1 Surface & device management** *(done)* — 1. `inf-render` device/queue/surface init +
  swapchain lifecycle; 2. production rect-sync (debounced reconfigure, letterboxing); 3. multi-DPI
  + monitor-change handling; 4. crash-safe device-lost recovery.
- **P2.2 Forward renderer core** *(done)* — 1. render-graph skeleton + WGSL pipeline cache;
  2. depth + MSAA + sRGB; 3. camera/view uniforms with **f64 world / f32 render split and
  floating-origin rebase**; 4. infinite editor grid + sky gradient; 5. debug-primitive layer
  (lines, wireframes, bounds).
- **P2.3 Editor camera** *(done)* — 1. RMB+WASD/QE flycam + scroll speed; 2. Alt-orbit/pan/dolly
  + MMB pan + LMB behaviors; 3. F-focus + camera bookmarks; 4. input capture/focus handoff polish.
- **P2.4 Picking & gizmos** *(done)* — 1. ID-buffer picking pass; 2. hover/selection outlines;
  3. translate/rotate/scale gizmos (engine-rendered, axis constraints, snapping); 4. overlay
  layer decision memo (egui vs custom) + implementation.
- **P2.5 Render test harness** *(done)* — 1. headless-wgpu golden-image runner (perceptual
  downscale tolerance, skip-if-no-adapter for WARP/lavapipe CI); 2. first golden scenes
  (grid+sky, cubes, selection+gizmo).

### Phase 3 — ECS & scene model

**Goal:** a real world: hierarchy, live Outliner, reflection-driven Details, undo/redo,
save/load. **Done when:** author a primitive+lights scene, 50-step undo/redo clean, save,
restart, byte-identical reload; sidecars readable in git diffs.

> **STATUS: Phase 3 COMPLETE** (2026-07-20). The mock Outliner/Details are now a live
> editor↔world binding over a real `bevy_ecs` world. The authoritative `SceneDoc`
> (`inf-editor-core::scene`) owns the world + selection + undo + serialization; the frontend
> is a projection kept in sync by a full snapshot + incremental `world://delta` events; the
> native viewport renders and picks against the *same* document (single source of truth).
> Gate met: primitives+lights authored via the Outliner Add menu, 50-step undo/redo clean
> (tested), Ctrl+S quicksave → reload is byte-identical (tested), TOML sidecars are
> git-diffable. Key decisions: `bevy_reflect` runs **without** its glam feature (it pins glam
> 0.32; the renderer is 0.33) — editable math is f64 value types that derive Reflect + serde,
> and reflection stays sealed inside `inf-ecs` (the facade rule) behind plain PropValue
> descriptors; euler-degree rotation in the Transform (UE-style Details); `.inf_lvl` =
> deterministic bincode payload + TOML metadata sidecar. Gizmo writeback writes world-space
> TRS as local (correct for roots/identity-parent objects; full parent-relative solve is a
> follow-up). Human-only remainder: a live demo recording + the macOS viewport hardware pass.

- **P3.1 ECS foundation** *(done)* — 1. `inf-ecs` facade (bevy_ecs+bevy_reflect), component registration
  macro; 2. transform hierarchy (f64 roots, propagation, dirty tracking); 3. core components
  (Name, Transform, Visibility, MeshRef, Light, Camera); 4. spawn/despawn command buffers.
- **P3.2 Editor↔world binding** *(done)* — 1. world-snapshot IPC protocol (diff-based, not full dumps);
  2. Outliner bound live (create/rename/reparent/delete/multi-select); 3. selection service
  (viewport picking ↔ Outliner ↔ Details single source of truth).
- **P3.3 Details via reflection** *(done)* — 1. reflection walker → property-tree IPC schema; 2. editor
  widgets per type (numeric drag, vec3, color, enum dropdown, asset ref); 3. multi-object edit;
  4. per-property reset + revert.
- **P3.4 Undo/redo** *(done)* — 1. command stack in `inf-editor-core` (every mutation is a command);
  2. transaction grouping (gizmo drag = one entry); 3. reflection-diff property tests.
- **P3.5 Scene serialization** *(done)* — 1. `.inf_lvl` bincode + TOML sidecar; 2. stable entity GUIDs;
  3. schema-version + migration harness; 4. autosave + crash recovery file.

### Phase 4 — Asset system & Content Drawer

**Goal:** the asset backbone with a production-feel Content Drawer. **Done when:** import a
glTF character with textures, browse thumbnails, drag into the scene, delete-with-references
warns correctly.

> **STATUS: Phase 4 COMPLETE** (2026-07-20). A real asset backbone under a live Content Drawer.
> `inf-asset` (Ring 0) is the GUID + xxh3 content-hash registry with a live dependency graph
> (forward *and* reverse edges), a debounced `notify` watcher, and a content-addressed import
> cache; `inf-mesh`/`inf-material` add the glTF (+meshopt) and texture (image decode + mips +
> hand-rolled BC1/BC3) importers. `inf-editor-core` orchestrates a glTF import into
> textures→materials→meshes wired by dependency edges (async queue + progress events), renders
> content-hash-cached thumbnails on headless wgpu (mesh 3/4 view, material sphere, texture
> flat), and owns the data-asset (`.inf_struct`/`.inf_enum`/`.inf_table`) schemas, Rust codegen,
> and CSV/JSON table import. The frontend Content Drawer is a live projection (virtualized grid,
> folder tree, breadcrumbs, filters, fuzzy search, favorites, lazy thumbnails, import strip,
> context menus) with the delete-with-references warning (the gate), drag-to-viewport placement,
> and inline struct/enum/table editors + asset-ref pickers. Key decisions: **BC7/intel_tex_2 is
> deferred** (its ISPC build is a cross-OS CI liability) in favor of a pure-Rust BC1/BC3 encoder;
> dropped-asset placement spawns a real named placeholder entity (rendering the *imported
> geometry* in the interactive viewport is the documented Phase 4→7 follow-up — the thumbnailer
> already renders it headlessly). Human-only remainder: the phase demo recording.

- **P4.1 Asset database** *(done)* — 1. GUID + xxh3 content-hash registry; 2. dependency graph +
  reverse-reference queries; 3. `notify` watcher with debounced rescan; 4. import cache.
- **P4.2 Importers** *(done)* — 1. glTF meshes (+meshopt optimization); 2. textures PNG/EXR/HDR +
  BCn (pure-Rust BC1/BC3; BC7/intel_tex_2 deferred) + mip generation; 3. import-settings sidecars
  (reimport respects them); 4. async import queue with progress events.
- **P4.3 Thumbnails** *(done)* — 1. headless-wgpu thumbnailer in `inf-editor-core`; 2. disk cache
  keyed by content hash; 3. type-specific renderers (mesh turntable frame, material sphere,
  texture flat).
- **P4.4 Content Drawer** *(done)* — 1. virtualized grid (@tanstack/react-virtual) + folder tree +
  breadcrumbs; 2. filters/type chips + fuzzy search + favorites; 3. drag-drop to viewport via
  the native handoff; 4. context menus (create/rename/duplicate/delete/show-references);
  5. Favorites + Collections (favorites shipped; named collections are a follow-up).
- **P4.5 Data assets** *(done)* — 1. `.inf_struct`/`.inf_enum` form editors → generated Rust types;
  2. `.inf_table` grid editor with CSV/JSON import; 3. asset-ref pickers everywhere.

### Phase 5 — IDE integration *(parallel with P3–P4)*

**Goal:** the CodeR IDE stack inside Studio panels; user projects are real cargo workspaces.
**Done when:** open a sample project, completions/diagnostics live in a hand-written system,
`cargo check` runs in the embedded terminal, git panel stages and commits.

> **STATUS: Phase 5 COMPLETE** (2026-07-20). The CodeR IDE stack ported into Studio dock panels.
> A lightweight Ring-0 `inf-project` crate (shared by editor/CLI/runtime) owns `inf.toml`,
> templates + scaffolding, and the recent list; `inf new` scaffolds a real user cargo crate;
> opening a project re-roots the asset database to its Content dir. Dockable panels: a
> **CodeMirror 6 editor** (per-file tabs with preserved undo/cursor, syntax highlighting,
> minimap, `--ink`-bridged theme, Ctrl+S save), an **xterm.js Terminal** (portable-pty / ConPTY,
> `cargo check`/`test`/`build` presets — the gate), a **Source Control** panel (shells out to
> `git`: status/stage/unstage/commit/diff/branch), **Search** (ignore-walk content search +
> fuzzy Go-to-File), a **File Explorer** (tree + git status letters), a **Problems** panel, and
> **LSP language intelligence** (a hand-rolled JSON-RPC-over-stdio rust-analyzer client → live
> completions/hover/diagnostics via a CodeMirror compartment seam). Files open through one
> `infinity:open-file` event; every panel is a `registerPanelType` entry. Key decisions: git
> **shells out to the `git` CLI** (no git2 native linkage, per the CodeR map); the rust-analyzer
> **HTTPS auto-installer is deferred** (the `ureq` TLS stack pulls `webpki-roots`, CDLA-Permissive-2.0,
> off the deny allow-list) in favor of PATH/cache resolution — RA runtime is human-verified like
> the GPU paths. Human-only remainders: split editors, diff/merge-view, stash/blame/log, inlay
> hints/code-lens/rename, the SHA-verified multi-server installer, and the phase demo recording.

- **P5.1 Editor panel** *(done)* — 1. CodeMirror 6 port (compartments, themes bridged to Studio
  theme); 2. multi-tab editors in dock panels (split editors: follow-up); 3. minimap, indentation
  guides (breadcrumbs: follow-up); 4. diff view (in Source Control; merge view: follow-up).
- **P5.2 Language intelligence** *(done)* — 1. LSP client (rust-analyzer via PATH/cache;
  SHA-verified auto-install: follow-up); 2. diagnostics/Problems panel + squiggles (quick fixes:
  follow-up); 3. hover + completion + go-to-definition (refs/rename/outline: follow-up);
  4. inlay hints, code lens, signature help: follow-up.
- **P5.3 Terminal & shell** *(done)* — 1. xterm.js + portable-pty port; 2. tabbed terminals
  (one per panel instance; split: follow-up); 3. task runner presets (`cargo check`, `test`,
  `build`).
- **P5.4 Git & search** *(done)* — 1. git panel (status/stage/commit/branch/diff via the `git`
  CLI; stash/log: follow-up); 2. file-tree git status letters; 3. global search panel
  (ignore-walk, fuzzy file open).
- **P5.5 Project system** *(done)* — 1. `inf new` (templates → user cargo workspace + `inf.toml`);
  2. project open/recent/switcher (Start Screen); 3. per-project content re-rooting.

### Phase 6 — Infinity Blueprints & transpiler v1 *(signature feature)*

> **STATUS: Phase 6 COMPLETE** (2026-07-20). The signature feature's full stack is in and
> CI-green. The graph ↔ IR ↔ Rust four-way sync is proven end-to-end and interpreter-vs-compiled
> parity holds. `inf-graph` (de-geo'd DAG substrate), `inf-blueprint` (interpreter + semantics +
> `.inf_act`/`.inf_fn` + node kit + graph↔IR lower/raise), `inf-transpile` (production round-trip),
> the Ring-2 `graph_*` command surface, and the `@xyflow/react` Blueprint canvas all landed. The
> "rotate on tick" gate authors in-graph, runs via the interpreter, and generates round-tripping
> Rust. Human-verified remainders (the Phase-5 bar): live canvas interaction, and compile-on-save
> dylib hot-swap in Simulate (the `inf-hotreload` mechanism is proven; wiring it to Simulate is a
> follow-up alongside the P9 PIE/Simulate loop). Deferred niceties: comment/reroute nodes,
> alignment/copy-paste, 1k-node perf pass, and idiomatic `self.field` variable sugar in the IR.

**Goal:** author gameplay visually, watch it run live, read/edit the generated Rust, stay in
sync. **Done when:** "rotate on tick + spawn prefab on click" authored purely in-graph runs via
interpreter; the generated Rust is hand-edited and the graph updates; round-trip proptests in CI.

- **P6.1 Graph core port** — 1. `inf-graph` port of the GeoCanvas DAG (model/registry/compile/
  exec/derive/cache, de-geo'd); 2. param metadata + pin typing (incl. `.inf_enum`/`.inf_struct`
  types); 3. xxh3 subgraph caching.
- **P6.2 Graph canvas** — 1. @xyflow/react canvas core (custom nodes, exec+data pins, wire
  styling per type); 2. node search menu (right-click, sectioned, searchable); 3. comment boxes,
  reroute nodes, alignment tools, copy/paste; 4. canvas perf pass (1k nodes @ 60 fps).
- **P6.3 Blueprint semantics** — 1. event model (BeginPlay/Tick/input/collision/custom);
  2. variables + functions + macros panels; 3. `.inf_act` asset (components + graph + defaults);
  4. `.inf_fn` libraries.
- **P6.4 Interpreter** — 1. tree-walking evaluator over compiled graph IR; 2. in-editor live
  ticking (Simulate integration); 3. wire-value debugging (hover to inspect, exec-flow pulse
  visualization); 4. breakpoints on nodes.
- **P6.5 Transpiler productionization** — 1. Spike B → `inf-transpile` production quality;
  2. full node kit coverage (flow control, math, ECS queries, spawn/destroy, timers, events);
  3. watcher-driven bidirectional sync + snippet nodes + in-UI "snippet" affordance;
  4. round-trip proptests as permanent CI; 5. "Open generated Rust" jump from any node.
- **P6.6 Hot-reload tier** — 1. Spike C → `inf-hotreload` integration; 2. compile-on-save of
  user crate → dylib swap in Simulate; 3. interpreter-vs-compiled parity test fixture in CI.

### Phase 7 — Materials & texture graphs *(overlaps P6 back half)*

**Goal:** `.inf_mat` graphs to live WGSL. **Done when:** layered PBR material with a
procedural mask authored in-graph, applied to the P4 character, edits visible in <1 s.

> **STATUS: Phase 7 COMPLETE** (2026-07-20), CI-green on all three OSes. The concurrency
> foundation, PBR renderer, material node graph → WGSL, texture bake, and material instances
> all landed. **P7.0** built the `inf-core` job system (rayon pool + named workers + flume;
> `parallel_map` is a deterministic in-order pure map — the P7.0 guard) and wired parallel glTF
> texture decode; a criterion bench proves scaling. **P7.1** replaced flat shading with a
> Cook-Torrance GGX metallic-roughness BRDF lit by a projected scene-lights buffer (directional
> + point, hemispheric ambient, ACES tonemap); `Material` gained metallic/roughness/emissive
> (Details-editable via reflection) and Content-Drawer apply-by-drag. **P7.2** is the flagship:
> `inf-material::graph` compiles a pure node DAG (inputs/const/math/vector/procedural/texture +
> an `output.surface` sink) to a naga-validated `material_surface` WGSL fn (node-anchored
> diagnostics, shared-node `let` hoisting), a Ring-2 `material_*` command surface, an offscreen
> live-preview sphere (PBR-lit), and an `@xyflow` Material editor panel opened from a `.inf_mat`.
> **P7.3** emits a `@compute cs_bake` shader that evaluates the graph per texel into a storage
> texture and bakes it (headless) to a new `.inf_tex` asset (usable as a `tex.sample` input).
> **P7.4** adds `.inf_mati` material instances (sparse overrides over a parent chain, resolved on
> apply) + a PBR golden scene. Key decisions: naga standalone (wgpu 30's pin) validates codegen
> in Ring-0 without pulling wgpu; the material graph reuses the blueprint substrate + frontend
> wholesale (only the registry, node theme, and compile/preview differ). Documented follow-ups:
> binding *real* referenced textures in the preview/bake (white today), per-material pipelines in
> the *interactive* viewport (preview + thumbnailer already render generated shaders), the
> instance override-parameter editor UI, and persisting a material's graph into its `.inf_mat`.

- **P7.0 Concurrency foundation** *(the job system, finally real — see §2.5)* — 1. `inf-core`
  job system: a rayon compute pool (sized to cores, named threads) + `flume` channels + a thin
  `parallel_for` / scoped-fan-out API; 2. retire the placeholder facade — this is the single
  Ring-0 data-parallelism entry point; 3. first consumers wired here: parallel asset import and
  the material/texture graph compile; 4. a criterion bench proving scaling + a determinism guard
  (same input → same output regardless of pool size).
- **P7.1 Material model** — 1. PBR BRDF + lighting pass in `inf-render` (directional + point,
  IBL basic); 2. material parameter blocks + per-material bind groups; 3. material assignment in
  Details + Content Drawer apply-by-drag.
- **P7.2 Material graph editor** — 1. material node kit (textures, UVs, math, lerp/mask, PBR
  output); 2. WGSL codegen with naga validation + error mapping back onto nodes; 3. live preview
  viewport (sphere/cube/mesh) via secondary surface; 4. hot pipeline recompile.
- **P7.3 Texture graphs** — 1. `.inf_tex` compute-pass codegen (noise, gradients, blends,
  filters); 2. bake-to-texture + live-procedural modes; 3. use as material inputs.
- **P7.4 Material instances** — 1. instance assets overriding parent parameters; 2. instance
  editor UI; 3. golden-image material test scenes.

### Phase 8 — 2D pipeline

**Goal:** first-class 2D/2.5D, not an afterthought. **Done when:** a small platformer scene
with Blueprint coyote-time jump plays in-viewport via interpreter.

> **STATUS: Phase 8 COMPLETE** (2026-07-21), CI-green on all three OSes, eight commits.
> First-class 2D built on the shared 3D substrate. **P8.1**: `inf-render-2d` (pure-CPU
> batcher — stable `(layer, order, texture)` sort + prebatched runs) + a GPU sprite pass
> (vertex-shader quad expansion, GUID-keyed texture cache, straight alpha, depth-tested/
> not-writing); chunked `Tilemap` (sparse 32×32 chunks, deterministic serde, chunk-level
> cull, per-chunk draw runs — no per-tile sort at 100k tiles); `NineSlice`; `Text2D` with
> an embedded public-domain font8x8 debug font (zero-asset text); minimal `Light2D`
> (`@group(2)`, ambient defaults white — provably invisible until used, strict-golden
> byte-stability held across every batch). **P8.2**: sprite-sheet slicing panel (sidecar
> grid + named rects) + sorting-layer manager; `.inf_lvl` schema v2 (five 2D slots,
> frozen-v1 decode + forever-load fixture); tile-paint panel (cell-level undo strokes,
> bounded fill); orthographic 2D editor mode (shared reverse-Z convention, per-mode
> cameras, zoom-to-cursor, XY grid, Z-less 2D gizmos, pixel snap with per-project PPU in
> `.infinity/settings.toml`). **P8.3**: `rapier2d-f64` facade (`enhanced-determinism`,
> rapier's rayon off — `inf-core` owns parallelism; 300-step byte-identical replay) +
> Guid-ordered `PhysicsBridge2D` (sync/step/write-back; handle allocation never follows
> entity-id churn) + selected-collider debug outlines + the `physics2d.*` blueprint kit
> via a `Host::physics()` accessor (zero IR change; every node transpile-round-trips).
> **P8.4 (the gate)**: spherical/cylindrical billboards; `SimSession` (enter snapshots →
> per fixed step: sync → Guid-ordered blueprint tick over a real `Physics2dHost` adapter →
> step → write-back → exit restores byte-for-byte); `input.is_down`/`just_pressed` nodes;
> `samples/platformer-2d` with the coyote-time jump graph — CI proves the coyote window
> (jump 3 steps after the ledge fires; post-window doesn't), deterministic re-runs, and
> interpreter-vs-compiled parity over a live physics host; `hybrid-2.5d` template
> (`inf new`-scaffolded); Play/Pause/Stop/Step toolbar + Alt+P driving `sim_start/tick/
> stop` with layout-independent key routing. Key decisions: tilemaps/9-slice/text ride
> the sprite pipeline as prebatched runs (one batching path, one painter order);
> physics is f64 with rapier's math *being* workspace glam (via glamx — no conversion
> layer); `.inf_act` stores JSON (bincode cannot round-trip `skip_serializing_if` —
> documented latent issue for asset-DB integration). Human-verified remainders: the live
> in-viewport play pass + demo recording, native-viewport held-key forwarding during
> Simulate, a fixed-dt Step command (`SimSession::step_once` exists, not yet surfaced),
> 2D click-picking (Outliner selects; ID buffer is mesh-only), macOS input (pre-existing).
> Deferred niceties: live-viewport texture upload for 2D content (goldens prove the
> textured path headlessly), atlas-image tile preview, Vec2 blueprint value type,
> `Light2D` layer masks, camera-delta tilemap re-expansion caching.

- **P8.1 2D rendering** — 1. sprite batcher (atlases, sorting layers, alpha); 2. tilemap
  renderer + tile editor panel; 3. 9-slice + text quads; 4. 2D lights (later batch, optional).
- **P8.2 2D editing** — 1. orthographic camera + 2D editor mode toggle; 2. grid/pixel snapping,
  sorting-layer UI; 3. sprite-sheet import + slicing editor.
- **P8.3 2D physics** — 1. rapier2d integration (fixed step); 2. collider components + Details
  editors + debug draw; 3. 2D character-controller node kit.
- **P8.4 2.5D** — 1. sprites-in-3D (billboards, sorting vs depth); 2. hybrid scene template.

### Phase 9 — Play-in-editor, player & desktop packaging *(close the loop early)*

**Goal:** the full make→play→ship loop, deliberately before advanced rendering. **Done when:**
the platformer packages to a double-clickable exe on Windows/macOS/Linux; a deliberate PIE
panic loses no editor state.

> **STATUS: Phase 9 COMPLETE** (2026-07-21), CI-green on all three OSes. The make→play→ship
> loop is closed. **P9.1**: the parallel `bevy_ecs` `Schedule` behind the facade
> (`multi_threaded`; phase-set sync points; command-buffer-only structural changes;
> `conflict_report()` discipline check) with the §8 replay gate REAL — serial == parallel
> byte-identical across pool sizes 1/2/8 via subprocess probes, verified on CI on all three
> OSes; headless `GameLoop` (fixed step + interpolation alpha, no wall clock); rapier3d-f64
> facade mirroring d2 (incl. trimesh) + ECS wiring + 3D debug outlines; kira audio (device
> backend feature-gated so CI never links ALSA/dbus; headless-consistent fallback tested);
> inf-input action/axis maps (pure edge/deadzone core; gilrs behind a feature). **P9.2**:
> deterministic `.inf_pack` (GUID-sorted index, xxh3-verified reads, zstd, byte-identical
> rebuilds); dependency-closure cook with handler-anchored blueprint IR validation; the
> runtime `.inf_lvl` reader lives in `inf-scene` (byte-lockstep with the editor codec,
> proven on committed bytes — rings intact); `inf cook`/`inf pack ls`; Build ▸ Package
> dialog. **Schema v3**: physics components + actor bindings + level settings persist
> (frozen-v2 fixtures, v1→v2→v3 lift in both codecs); cook follows actor dep edges;
> **gameplay runs off a real cooked pack** (trace-proven: differs from a no-actor bake,
> scripted input moves the character). **P9.3**: standalone winit player (full render
> stack, editor-free RuntimeSim, InputMap); `--headless --run-frames N --assert-exit` with
> crash.txt capture + xxh3 determinism; **cooked == uncooked trace identity**. **P9.5**:
> `inf export` → runnable folder (renamed player + pack + player.toml boot config + honest
> PACKAGING.txt — no faked installers/notarization); the exported exe boots its own content
> with zero args (tested); the §8 cook-and-run smoke runs on every push on all three OSes.
> **P9.4**: PIE protocol v2 (versioned LE frames) streams the live unsaved doc + bound
> classes to a spawned player that builds the world through the same code as the pack path
> — **PIE == shipping proven byte-identical over a real subprocess**; Pause/Resume/Step/
> Stop/Eject; deliberate panic loses no editor state (tested); zombie-free reaping;
> Windows embedded window via main-window-parent SetParent (sidestepping the Spike-D
> teardown deadlock), Play-in-New-Window as the sanctioned fallback elsewhere; split-button
> play cluster (Embedded / New Window / Simulate). Human-verified remainders: embedded-PIE
> visual pass + demo recording, camera possession & Eject hand-back, input routing from the
> embedded window, flash-free embed, macOS/Wayland embedded PIE, native macOS/Linux export
> runs. Deferred: per-OS installers/signing (P15), mmap pack reader, shared Ring-0
> scene→render projection (player duplicates the viewport's), 3D solver in the player
> actor loop (2D bridge today), compiled-dylib cook tier (interpreter IR ships; dylib is
> the hot-reload tier).

- **P9.1 Runtime assembly** — 1. `inf-runtime` game loop (fixed-step sim + interpolated
  render); 2. **parallel ECS schedule** — enable `bevy_ecs` `multi_threaded`, run systems through
  a real `Schedule` on the P7.0 job pool, structural changes resolved at deterministic sync points;
  the fixed-step result is **identical to the serial baseline** (replay-harness gate, §8);
  3. rapier3d + kira baseline integration; 4. `inf-input` action mapping + gamepad.
- **P9.2 Cook pipeline** — 1. asset pack format (content-addressed, compressed); 2. cook =
  resolve deps → compile blueprints → release-build user crate → bundle; 3. `inf cook` CLI +
  editor Package dialog.
- **P9.3 Player** — 1. `inf-player` winit binary loading packs; 2. headless mode
  (`--run-frames N --assert-exit`) for CI; 3. crash/log capture.
- **P9.4 PIE** — 1. subprocess spawn with in-memory snapshot handoff; 2. Play/Pause/Stop/Eject
  + possess camera; 3. embedded PIE window via Spike A machinery ("new window" fallback);
  4. Simulate mode in-process.
- **P9.5 Desktop export** — 1. Windows (exe + installer), macOS (.app + notarization docs),
  Linux (AppImage); 2. icon/branding injection; 3. CI cook-and-run smoke test.

### Phase 10 — Terrain & PCG

**Goal:** the Gaea-heritage differentiator: sculptable, erodible, planet-ready terrain with
massive scattering. **Done when:** 16 km² terrain sculpted + interactively eroded, 1M+
instances scattered by rules, 60 fps flythrough, works in PIE.

> **STATUS: Phase 10 COMPLETE** (2026-07-21), CI-green on all three OSes. The proprietary
> terrain + PCG stack. **P10.1**: sparse f64-anchored tile heightfield (tile-local f32 —
> the precision doctrine applied to ground; shared edges, seamless), bilinear/normal
> queries, PNG16/EXR import, chunked-LOD clipmap renderer (4 LODs, smoothstep morph, skirt
> rings, per-patch cull, R32Float tile textures via textureLoad so CI adapters work).
> **P10.2**: brush core (raise/lower/smooth/flatten/world-anchored-noise; falloff
> profiles; HeightDelta = dense per-tile patches + created_tiles → byte-identical undo)
> + in-viewport sculpting (ray-march picking, gizmo-style stroke transactions, engine-
> rendered brush ring, toolbar controls) + splat painting one storey up (SplatDelta,
> exact-255 renormalization, per-layer flow). **P10.3**: virtual-pipes hydraulic + thermal
> erosion — deterministic mass-accounted CPU reference (closed-box conservation, monotonic
> talus relaxation, byte-identical with stochastic rain; 13.6M cell-steps/s) and a WGSL
> port that is bit-exact at early steps (two-tier parity gates with a documented cross-
> adapter envelope — Metal fast-math drifts chaotic long-runs; the 8-step per-cell gate is
> the precision spec); bakes are ONE undo step; eroded terrain is data (adapter variance
> confined to the bake action). **P10.4**: 4-layer splat materials (sparse RGBA8 weights
> beside heights, byte-compatible serde both directions), triplanar procedural grain on
> steep faces, macro fBm variation; VT exploration memo (defer to P13/P15 with revisit
> criteria). **P10.5**: deterministic massive-scale scattering (counter-hashed, pool-size-
> invariant, 100k instances/0.06s; density/slope/altitude/mask samplers + combinators;
> hand-rolled seeded fBm) + the PCG node editor on the shared canvas (graph→PcgDocument
> lowering with node-anchored diagnostics, .inf_pcg stores graph-as-JSON-in-bincode —
> the third dodge of the skip_serializing_if/bincode trap, now joined by format-aware
> TerrainTile serde) + PcgVolume instances rendered through the instanced path with
> volume-select picking. **Schema v4** persists Terrain (heights + weights) and PcgVolume;
> guard tests flipped to byte-identical round-trips; cook follows PCG graph dep edges;
> the player evaluates volumes on load and renders terrain; `samples/terrain-demo` gate
> scene passes all three gates — byte-identical save/reload, cooked headless population
> with deterministic instances, and **PIE == shipping for terrain + PCG content**.
> Human-verified remainders: the 16 km²/1M-instance/60 fps flythrough perf pass (the
> machinery exists; scale numbers need eyes + Tracy), live sculpt/erode/paint visual
> pass, demo recording. Deferred: per-layer texture GUIDs + real mesh/texture upload in
> the interactive viewport (the one shared gap), GPU-instanced scatter + impostor/LOD
> fade (P13 pairs), erosion sub-region drag-select, streaming/paging of terrain tiles,
> multi-terrain merge, PCG mask-image node + multi-rule lowering, moving-camera re-eval.

- **P10.1 Heightfield core** — 1. quadtree/clipmap terrain in f64 world space; 2. LOD morphing,
  skirts, frustum culling; 3. heightmap import (EXR/PNG16) + export.
- **P10.2 Sculpt & paint** — 1. brush framework (raise/lower/smooth/flatten/noise); 2. GPU
  brush application + undo integration; 3. splat-layer painting (weights → material blend).
- **P10.3 Erosion (GPU)** — 1. hydraulic erosion compute pipeline (WGSL, tiled); 2. thermal +
  sediment passes; 3. interactive preview + bake; 4. erosion-graph nodes in `.inf_pcg`-style
  terrain graph.
- **P10.4 Terrain materials** — 1. splat-blended layered materials (P7 integration);
  2. triplanar + macro variation; 3. virtual-texture exploration memo.
- **P10.5 PCG runtime & editor** — 1. `inf-pcg` samplers (density/slope/altitude/mask/noise);
  2. rule graph (filter → transform → spawn) + `.inf_pcg` editor on the shared canvas; 3. GPU
  instanced scattering + per-instance culling; 4. LOD/impostor fade; 5. PCG debug visualization.

### Phase 11 — Animation & character

**Goal:** things that move and feel alive. **Done when:** idle/run/jump state-machine character
driven by Blueprint input runs across P10 terrain in PIE.

> **STATUS: Phase 11 COMPLETE** (2026-07-21), CI-green on all three OSes. **P11.1**:
> `inf-anim` pure pose math (validated topological skeletons, step/linear clips with
> cubic-resampled import, shortest-path slerp sampling, blend_poses, skinning matrices);
> `.inf_skel`/`.inf_anim` assets (pack codes 13/14); glTF skins/clips/JOINTS+WEIGHTS import
> proven on an in-test-constructed glTF; GPU skinning as an additive pipeline (joints/
> weights attributes + @group(3) palette buffer; unskinned path byte-stable, strict-golden
> proven; golden_skinned_mesh bends a procedural cylinder); AnimPlayer ticks in both sim
> loops. **P11.2**: 1D/2D blend spaces (bracketing/IDW-k3 v1); `.inf_sm` state machines as
> a PLAIN serde model (states/transitions aren't a DAG — and no bincode escape hatch
> needed), AND-condition transitions, exit-time gating, crossfade; the SmContext seam reads
> Blueprint vars for conditions AND blend params; state-machine canvas panel (states as
> nodes, transitions as edges, condition inspector). **P11.3**: sockets on SkeletonAsset +
> AttachedTo following posed sockets; loop-safe root-motion extraction applied through the
> 3D mover; the `physics3d.*` kit via Host::physics3d() (zero lowering changes — the
> generic namespace path absorbed it; full transpile round-trips); both sim ticks run the
> 2D+3D bridges; humanoid retarget v1 (bind-relative rotation copy). **P11.4**: sequencer
> seed — scalar property tracks, non-dirtying scrub-with-restore (proven by test),
> capture-key workflow, timeline panel. **Schema v5** persists all five animation
> components (frozen-v4 fixtures both codecs; delete→undo restores); cook closes over
> skeleton/clip/sm refs incl. SM→clip edges; ScenePayload v3 carries anim assets into PIE.
> **The gate** (`samples/character-demo`): a 6-joint procedural character with programmatic
> idle/run/jump clips and a state machine driven by Blueprint vars crosses sine-hill
> terrain — x advances, Y tracks the heightfield (terrain.height_at host seam), jump
> lifts and lands, SM transitions idle→run→jump in order, deterministic across two
> independent cooks, and **PIE == shipping on the (x, y, sm-state) trace**. Human-verified
> remainders: live skinned rendering in the interactive viewport (headless-golden-proven;
> the shared mesh/texture-upload gap), the visual play pass + demo recording. Deferred:
> in-panel blend-space authoring, Delaunay 2D blending, IK/foot fix-up, retarget scaling,
> Vec3 blueprint value, pose-driven socket tracking, blend-space root motion.

- **P11.1 Skeletal foundation** — 1. glTF skin/skeleton import (`.inf_skel`); 2. GPU skinning;
  3. clip playback + blending (`.inf_anim`).
- **P11.2 Animation graphs** — 1. blend spaces (1D/2D); 2. state machine asset (`.inf_sm`) +
  editor on the shared canvas; 3. transitions/conditions bound to Blueprint variables.
- **P11.3 Character tools** — 1. sockets/attachments; 2. root motion; 3. kinematic character
  controller (slopes/steps/jumps); 4. retarget v1 (humanoid rig mapping).
- **P11.4 Sequencer seed** — 1. simple timeline panel for cutscene/property tracks (full
  sequencer deferred).

### Phase 12 — Physics & audio depth

**Goal:** production-grade simulation. **Done when:** physics playground sample (joints,
ragdoll, spatial audio) runs deterministically under replay test.

> **STATUS: Phase 12 COMPLETE** (2026-07-21), CI-green on all three OSes. **P12.1**:
> impulse joints in both facades (Fixed/Revolute/Prismatic/Spherical/Distance, motors +
> limits, sorted deterministic ids) with flat Joint2D/3D components reconciled in a
> second Guid-keyed bridge pass; CollisionLayers → InteractionGroups + a named 32-bit
> registry (.infinity/collision_layers.toml); friction/restitution CombineRule; CCD with
> tunnels-without/stops-with proofs in both dims; build_ragdoll (humanoid name-map →
> capsule chain with limited joints; settles bounded); joint debug draw. **P12.2**: the
> composed playground determinism gate (stacks + every joint kind + motorized hinge +
> CCD bullet + layer ghost + sensor sweep → byte-identical 300-step runs, 2D and 3D);
> the Jolt-vs-rapier memo — VERDICT: stay on rapier through P14 (f64 world fit,
> cross-platform enhanced-determinism, pure-Rust CI), named revisit triggers recorded.
> **P12.3**: .inf_audio assets (original compressed bytes, decode-on-load); the
> command-queue doctrine — the audio command stream is a pure function of sim state,
> the device is never sim state (headless tests assert the stream verbatim in editor
> AND player); exponential attenuation + min/max clamps; raycast occlusion (−12 dB)
> behind AudioSource.occlusion; .infinity/mixer.toml bus tree (Gain engine-side,
> Lowpass modeled — kira TrackBuilder wiring is the device follow-up); audio.* nodes
> with full transpile round-trips. **Schema v6** persists joints + audio source/listener
> (frozen v5 both codecs; the downgrade-bless mechanism now standard); cook follows
> AudioSource.clip edges. **The gate** (`samples/physics-playground`): slab, settling
> stack, motorized spinner (+looping occluded spatial source), rope pendulum, prismatic
> slider, CCD bullet vs thin wall, layer-ghost pair, sensor plate (+source), an
> 8-bone/7-joint spawned ragdoll, camera listener — 300-step cooked replays are
> byte-identical in BOTH pose trace and audio command stream, and **PIE == shipping on
> both**. Pre-1.0 policy codified this phase: frozen records embed live component types;
> layout changes re-bless fixtures (INF_BLESS_FIXTURES / downgrade-bless); true
> loads-forever = frozen component snapshots at 1.0. Human-verified remainders: audible
> playback pass (device builds), live joint/ragdoll visual pass, demo recording.
> Deferred: doppler, reverb/sends + audible per-bus lowpass DSP, waveform thumbnails +
> preview UI, ragdoll editor UI, named-bitmask Details widget, joint entity-ref picker,
> contact-point debug draw, PIE audio-byte streaming (command-stream parity needs none).

- **P12.1 Physics depth** — 1. joints/motors/CCD/queries; 2. collision layers + filtering UI;
  3. physics materials; 4. ragdoll setup tool; 5. debug-draw overlay.
- **P12.2 Determinism** — 1. fixed-step replay harness; 2. Jolt-vs-rapier benchmark memo
  (decision gate for a backend swap).
- **P12.3 Audio depth** — 1. kira spatialization/attenuation/occlusion basic; 2. mixer buses +
  effects; 3. audio assets + import; 4. Blueprint audio node kit.

### Phase 13 — Virtualized geometry & advanced rendering *(flagship; deliberately late)*

**Goal:** Nanite-class geometry and Substrate-class materials. **Done when:** a 10M+ triangle
scene streams and culls at interactive rates; classic-LOD fallback documented for older GPUs.

> **STATUS: Phase 13 COMPLETE** (2026-07-21), CI-green on all three OSes. The flagship —
> all original implementations. **P13.1 (virtualized geometry)**: offline meshlet DAG
> builder (64v/124t clusterize → greedy shared-edge grouping → border-locked group
> simplify (parallel, pool-size-invariant) → recluster → link; strictly-monotonic error
> intervals tile [0,∞)) with THE proof — watertightness by edge bookkeeping across 400+
> threshold sweeps (crack-free, non-overlapping at any cut); `.inf_vmesh` (pack code 16,
> coarse-first for streaming) cook-derived for meshes ≥2048 tris; the GPU path: cull
> compute (LOD cut → frustum → cone → optional HZB (reverse-Z MIN-depth pyramid, wired,
> two-pass re-projection deferred)) with the parity trick — the pixel tolerance projects
> to ONE per-instance CPU scalar so the branchless GPU cut is BIT-IDENTICAL to
> VgeomMesh::select (parity test exact); vertex-pulled single draw_indirect per asset.
> **P13.2**: Substrate-class slabs over the material graph (MatSlab WGSL structs,
> factor/mask blends, byte-identical legacy back-compat, node-anchored both-wired errors)
> + advisory complexity budgets in the panel. **P13.3**: HDR (Rgba16Float, tonemap
> hoisted to post), bloom, TAA (Halton jitter, history reprojection, convergence smoke),
> half-res SSAO (ambient-only); 3-cascade CSM (texel-snapped bounding-sphere fits, PCF);
> and the **dynamic GI** — camera-centred 64³ analytic voxelization, 16×8×16 probes ×48
> deterministic golden-spiral rays with marched sun visibility, L1 SH, trilinear ambient
> evaluation — proven by golden_gi_bleed (near-wall red/green 1.135 vs far 1.040, byte-
> identical double-render); AO+shadow+GI consolidated into one EnvBinding under the
> 4-bind-group ceiling, off-paths instruction-stream-identical (19 prior goldens byte-
> stable). **P13.4**: MeshRef.asset (schema v7 both codecs; frozen MeshRefV6 repointing
> kept all v1–v5 fixtures decoding unchanged); TRUE discrete-LOD classic fallback
> (per-level index buffers over the shared vmesh vertices, picked by the SAME threshold
> rule — parity, not distance-cull); adapter capability probe → High/Medium/Low
> auto-tiering (clamps down only, override + memo). **The gate** (`samples/vgeom-demo`):
> 18×18 instances of a 33k-tri mesh = **10,616,832 source triangles**, ground-camera GPU
> cull to **2.4% visible**, deterministic across runs; the same pack renders classically
> with far-camera picking strictly coarser LODs; forced-Low auto-disables vgeom.
> Human-verified remainders: interactive-rate flythrough on hardware (the machinery +
> ratios are proven; frame-rate needs eyes), live visual passes, demo recording.
> Deferred: two-pass HZB occlusion, meshlet streaming/paging, editor-viewport real-mesh
> rendering + PIE vmesh streaming (player renders real geometry; viewport placeholders),
> per-slab textures/normals, temporal probe amortization, specular GI, cascade blending,
> quantized vgeom vertices, per-material meshlet tagging.

- **P13.1 Meshlet pipeline** — 1. meshopt clusterization + simplification DAG builder (offline,
  in cook); 2. meshlet pack format + streaming; 3. GPU-driven culling (two-pass occlusion,
  HZB); 4. meshlet LOD selection compute.
- **P13.2 Substrate-class materials** — 1. layered material model (slabs/blends) over P7;
  2. codegen for layered BSDFs; 3. material complexity budget tooling.
- **P13.3 Lighting & post** — 1. cascaded shadow maps → virtual shadow map exploration;
  2. HDR pipeline, bloom, tonemap, TAA; 3. SSAO/SSR baseline; 4. GI exploration memo
  (DDGI-style probes vs baked).
- **P13.4 Fallbacks** — 1. classic LOD path parity; 2. GPU-capability detection + auto-tier.

### Phase 14 — Platform expansion & networking

**Goal:** widen targets on the HAL; prove the console seam. **Done when:** platformer sample
runs on Android and in Chrome; two desktop instances replicate transforms; console-port design
review passes against the HAL audit.

> **STATUS: Phase 14 COMPLETE (engineering scope)** (2026-07-21), CI-green on all three
> OSes. The honest-scope doctrine applies: device-dependent verification (a phone in
> hand, a browser frame on screen) is human-only; everything CI can prove, it proves.
> **P14.1/14.2**: the ENTIRE player dep tree cross-compiles for wasm32-unknown-unknown
> with -D warnings — gated by the new CI wasm-check job (uuid js, getrandom wasm_js,
> notify gated, zstd→ruzstd decode on wasm, the meshopt C++ builder cook-only);
> `web::start_player(canvas, pack_url)` fetches packs over HTTP into the shared world
> builder; Android android-native-activity entry (NDK-gated locally, non-blocking CI
> check); pure TouchControls virtual gamepad feeding the existing InputMap; mobile
> render-tier preset; `inf export --target web|android` with real builds when tools are
> present, honest instructions otherwise. Honest remainder: the in-browser frame needs
> the async-adapter seam (GpuContext block_on cannot run on the web main thread).
> **P14.3**: inf-net sans-io reliability Endpoint — exactly-once-in-order proven under
> 16 seeds of 30% loss/20% dup/40% reorder; Guid-keyed delta snapshots; RPC registry;
> quinn QUIC transport behind an off-by-default feature (ring crypto; localhost
> integration: 100 transforms + RPC round-trip); two-GameLoop replication proof; the
> net-model memo (snapshot default; deterministic lockstep documented VIABLE because
> §2.5 bit-determinism is CI-proven). **P14.4**: inf-platform HAL traits + desktop
> backend; platforms/inf-platform-null implements all six seams (the out-of-tree
> private-repo pattern demonstrated); the audit memo's honest gap table (file IO direct
> in five crates, wgpu inherent, save-data greenfield) + cook-target plumbing + generic
> TRC checklist. **P14.5**: the WASM modding tier per the crossref memo — inf-wasm-host
> (wasmtime 47, core modules only): capability-scoped deny-by-default imports
> (ungranted import = capability-anchored instantiation error), deterministic fuel
> limits + opt-in epoch interruption (a deliberately hung mod traps, is disabled, the
> host survives — tested), memory caps; the blueprint→Rust→wasm cook shim reuses the
> EXISTING transpiler (one truth preserved; `inf cook --mods` generated AND compiled a
> real mod); the spinner sample runs end-to-end in a real RuntimeSim (filename-ordered
> deterministic ticking); editor ModsSession hot-reload proven by test (SimSession
> glue = documented follow-up); the security-posture memo. Human-verified remainders:
> browser/device runs, two-desktop live replication session, demo recordings.
> Deferred: async-adapter web frame, range-request pack streaming, iOS xcodeproj/
> signing, client prediction/interest management, mod-state migration across reload,
> per-capability rate limits, HAL file-IO routing.

- **P14.1 Mobile** — 1. Android export (wgpu Vulkan/GLES paths, touch in `inf-input`, APK
  packaging); 2. iOS export (Metal, xcodeproj generation, docs for signing); 3. mobile perf
  tier presets.
- **P14.2 Web** — 1. wasm + WebGPU player build; 2. decision: interpreter vs wasm-compiled
  blueprints on web; 3. pack streaming over HTTP.
- **P14.3 Networking seed** — 1. quinn (QUIC) transport; 2. transform replication + RPC
  skeleton; 3. lockstep-vs-snapshot decision memo per genre template.
- **P14.4 Console readiness** — 1. HAL audit: every OS/GPU/file/input dependency behind
  `inf-platform` traits, verified by a mock "null console" backend crate; 2. cook-target
  plumbing for out-of-tree platform crates; 3. private-repo pattern + docs for PS5/Xbox/Switch
  backends (NDA SDKs and devkits required; cannot live in this public repo); 4. controller/TRC
  compliance checklist drafts.
- **P14.5 Sandboxed extensibility / modding** *(reuses one authoring model — see the crossref
  memo)* — 1. a sandboxed **WASM** runtime (e.g. `wasmtime`) in `inf-runtime` loading mods/plugins
  as `.wasm`; 2. a **WASM cook target** so a mod authored in the *same* Blueprints/Rust is compiled
  (blueprint → Rust → wasm) — no new scripting language, "two ways to code, one truth" preserved;
  3. a stable, **capability-scoped host API** (explicit, deny-by-default surface for entities,
  assets, input, events) so untrusted mods stay safe; 4. hot-reload of WASM modules in the editor +
  a sample "moddable" game. Rationale: the blueprint interpreter already gives fast no-recompile
  *iteration*; the capability WASM adds is **safe, no-compiler, end-user extensibility** that dylib
  (ABI-fragile/unsafe) and blueprints (not runtime-user-facing) cannot. This resolves the P14.2
  "interpreter vs wasm-compiled blueprints" decision toward a shared WASM path.

### Phase 15 — Polish, optimization, docs & samples

**Goal:** commercial-grade finish. **Done when:** a newcomer installs Studio, follows the
tutorial, and ships a small game in a weekend.

> **STATUS: Phase 15 COMPLETE (engineering scope)** (2026-07-21), CI-green on all three
> OSes — and with it, **every phase of this roadmap is implemented**. **P15.1**:
> tracing-tracy behind an off-by-default feature (spans unconditional: per-frame/
> per-pass render, per-phase sim, cook/import/vgeom stages; docs/profiling.md); the §8
> budget ratchets live in the normal test run (FRAME_BUDGET_MS 33 vs 0.185 measured;
> SIM_STEP_BUDGET_MS 2 vs ~0.02; 5s open/startup tripwires vs 5.6ms measured — the
> constants only go down); the cook's per-asset CPU work fans across the job pool with
> byte-determinism gates still green (zstd serial, documented). **P15.2**: structured
> crash reports (player: engine/OS/adapter/log-tail + crashes/ dir; editor: src-tauri
> panic hook + log ring; the viewport thread is catch_unwind-wrapped — the editor
> survives its render loop panicking); upload is a documented opt-in stub, never silent
> telemetry; autosave recovery never panics on corrupt files (.corrupt quarantine +
> fallback, tested); an ignored deterministic 10k-cycle soak with bounded-memory
> asserts. **P15.3**: template gallery with SVG previews + 3D/2D/Scripting first-run
> layouts; the 7-step anchored first-run tour (deferred to first project open,
> replayable from Help); docs/book mdBook with nine real content pages + a CI build;
> first-person template upgraded to a real starter scene; samples index with the honest
> three-games status (platformer done; exploration = parts exist; shooter = follow-up).
> **P15.4**: branding sweep clean; docs/LICENSING.md presents options for the OWNER's
> decision (deliberately not picked); one version source + build-time git hash → About;
> docs/release-channels.md (honest updater spec). Human-verified remainders: Tracy
> profiling sessions on real content, the newcomer-weekend test itself, demo recordings,
> docs-site deploy (ops), the signed updater (ops). Deferred: the third sample game +
> composing the exploration demo, frontend adoption of the diagnostics command,
> renderer/runtime byte-counters in MemoryReport, parallel zstd, mouse-look input seams.

- **P15.1 Performance** — 1. Tracy-instrumented frame-budget passes (editor + runtime);
  2. startup-time budget; 3. parallel import/cook tuning; 4. memory profiling + budgets.
- **P15.2 Stability** — 1. crash reporter (minidump + opt-in upload); 2. autosave/recovery
  hardening; 3. long-session soak tests.
- **P15.3 Onboarding** — 1. project-template picker with previews; 2. interactive first-run
  tour; 3. docs site (mdBook/Docusaurus) + API docs; 4. three polished sample games (3D
  exploration, 2D platformer, top-down shooter).
- **P15.4 Product** — 1. branding/licensing pass; 2. versioned release channels + updater;
  3. roadmap v2 planning from telemetry of our own dogfooding.

## 7. Technology matrix

| Subsystem | Choice | Rationale |
|---|---|---|
| ECS | bevy_ecs (standalone, behind `inf-ecs`) | best archetype ECS + change detection without adopting Bevy-the-engine |
| Reflection | bevy_reflect | powers Details panel + scene serde; building this from scratch costs a phase |
| Math | glam (DVec3/DQuat world, f32 render) | only mainstream Rust math lib with real f64 SIMD types |
| Rendering | wgpu + naga | portable modern API; naga validates generated WGSL at author time |
| Meshlets | meshopt (meshopt-rs) | industry-baseline clusterization/simplification |
| Physics | rapier3d-f64 / rapier2d | pure Rust (console-portable), f64 builds; Jolt re-evaluated P12 |
| Audio | kira | game-oriented (clocks, tweens, spatial); rodio is playback-only |
| Import | gltf, image, intel_tex_2 (BC7); FBX via ufbx later | glTF-first modern pipeline |
| Serialization | serde + bincode 2 + toml + xxhash-rust | the dual-format asset design |
| TS bindings | ts-rs | proven pattern; CI drift-checked |
| Transpiler | syn 2 + quote + prettyplease | the only serious option; deterministic formatting |
| Hot reload | libloading + hand-rolled #[repr(C)] vtables | abi_stable judged too invasive (Spike C re-checks) |
| Concurrency | rayon compute pool + flume (Ring 0 job system) + tokio (Ring 2 editor IO) | first-class; see §2.5, built P7.0, parallel ECS P9.1 |
| Profiling | tracing + tracing-tracy | doubles as Output Log source |
| OS interop | windows, objc2(+app-kit), x11rb, raw-window-handle | Spike A requirements |
| Watching | notify + notify-debouncer-full | asset DB + transpiler sync |
| Frontend | React + TS + Vite + Tailwind + Radix | reference-project stack; ports cleanly |
| UI state | zustand | both reference projects use it; fits the dock IPC bridge |
| Docking | GeoCanvas custom dock (ported) | off-the-shelf libs can't do detachable native windows w/ IPC sync |
| Node canvas | @xyflow/react 12 | proven with a Rust DAG backend in GeoCanvas |
| Code editor | CodeMirror 6 | entire CodeR LSP/editor investment is CM6 |
| Terminal | xterm.js + portable-pty | direct CodeR port |
| Git | git2 | direct CodeR port |
| Virtual lists | @tanstack/react-virtual | 100k-asset Content Drawer |
| Networking | quinn (QUIC) | deferred to P14 by design |

Deferred decisions: egui vs custom viewport overlay (P2.4 memo); FBX support (demand-driven);
GI technique (P13.3 memo); text rendering in-engine (glyphon likely, P11+); web blueprint
execution mode (P14.2).

Deliberate non-goals (decided, not gaps): **no separate embedded scripting language** (Rhai/Lua/
Rune) — the blueprint interpreter already delivers no-recompile iteration over the *same* IR that
ships as Rust, so a third language would fracture "two ways to code, one truth"; safe end-user
extensibility is served by the P14.5 WASM tier instead. WGSL over `rust-gpu` (naga validates at
author time). Native wgpu viewport over immediate-mode egui (§2.3.1). See
`docs/memos/rust-report-crossref.md`.

## 8. Verification & CI strategy

- **Every commit, 3 OSes:** `cargo fmt --check` · `cargo clippy --workspace -D warnings` ·
  `cargo nextest run` · `cargo deny check` · `tsc --noEmit` · eslint · vitest · ts-rs drift check.
- **Renderer:** headless-wgpu golden images (dssim perceptual diff; WARP on Windows CI,
  lavapipe on Linux) from P2 onward; per-backend goldens.
- **Transpiler:** permanent proptest suite — graph→code→graph isomorphism, regeneration
  idempotence, hand-edit corpus; every new node type must ship with its round-trip case.
- **Assets:** serde round-trip proptests per `.inf_*` type; schema-migration fixtures load
  forever; sidecar byte-determinism test.
- **Undo:** property test — any command sequence + inverses restores reflection-diff equality.
- **Parity gate:** each blueprint fixture runs under interpreter *and* compiled dylib; outputs
  must match (the preview == shipped guarantee).
- **Concurrency determinism (from P9.1):** the fixed-step replay harness runs a sample under the
  **parallel** ECS schedule and asserts a byte-identical trace to the serial baseline across pool
  sizes — parallelism may never change a step's result (§2.5).
- **Cook/PIE:** CI cooks a sample and runs `inf-player --headless --run-frames 300 --assert-exit`.
- **Performance:** criterion benches (transform propagation, sprite batcher, scatter kernels);
  nightly frame-budget smoke on a reference scene with a hard ms budget that only ratchets down.
  From P16.6 the ratchet also covers a **streamed scene**: a fixed-step ms budget while cell +
  terrain streaming are live, plus residency **memory ceilings** (terrain page bytes, cell blob
  bytes, active cells), all asserted headless over `samples/phase16-world`
  (`inf_player::budget`; the table lives in `docs/profiling.md` §2).
- **Process:** every phase ends with a recorded demo against its written checklist; each
  phase's sample content is committed under `samples/` and becomes the next phase's regression
  fixture.

## 9. Platform strategy

| Tier | Platforms | Status in roadmap |
|------|-----------|-------------------|
| 1 | Windows, macOS, Linux (X11; Wayland-fallback editor) | first-class from P9 |
| 2 | Android, iOS, Web (WebGPU) | P14.1–P14.2 |
| 3 | PS5, Xbox Series, Nintendo Switch | P14.4 architecture + private backend crates |

**Console reality, stated plainly:** shipping on PS5/Xbox/Switch requires registered developer
status with each platform holder, NDA'd SDKs, and physical devkits. None of that can exist in
this public repository. What this roadmap delivers is the *engineering seam*: every platform
dependency isolated behind `inf-platform` traits, a null-console mock backend proving the seam
holds, cook-target plumbing for out-of-tree crates, and compliance checklists — so a console
port is a bounded, additive project (a private crate + toolchain) rather than a refactor.

## 10. Porting inventory

From **CodeR** (`Tauri v2 IDE`): CodeMirror 6 editor stack (compartments, minimap, diff/merge),
LSP client + rust-analyzer auto-installer (SHA-256-verified), xterm.js + portable-pty terminal,
git2 panel + status index, JSON theme system, typed-IPC convention (`ipc.ts` mirror, namespaced
events), command palette, global search, custom window chrome, multi-window management.

From **GeoCanvas** (`Tauri v2 GIS`): React Flow node-graph editor + custom Rust DAG backend
(model/compile/exec/derive/cache with xxh3 caching, registry, param metadata), hand-rolled
docking system with detachable native windows + IPC store bridge, serde+ts-rs binding
generation, the Tauri-free-core crate discipline, progress/event job system.

Built fresh (no donor exists): the entire wgpu renderer, ECS integration, asset database,
terrain/erosion, PCG runtime, animation, physics/audio integration, transpiler, packager.

---

## 11. Post-plan status — UE-Parity Wave 1 (2026-07-22, COMPLETE)

Six CI-gated waves after the master plan closed the tracked follow-ups and lifted the most
user-visible "simplified" subsystems toward Unreal parity. Schema bumped v7→v8 once (frozen V7
records in both codecs; downgrade-blessed).

**Shipped:** per-kind primitive geometry (sphere/plane/cylinder/cone across mesh/depth/shadow/
mask/pick, editor+player mirrored); entity duplicate/copy/cut/paste; drag-spawn at the cursor's
world point; two-way gizmo-mode sync + local/world space + nested-transform writeback fix +
configurable snap; Reveal in Explorer; Save Level As + current-level path; live Recent Projects;
editable World Settings (gravity/sim-rate + scene-persisted exposure/bloom/SSAO/TAA/shadows/GI,
applied in-editor and in the player); real spot lights (cone falloff, per-light cast_shadows,
range bugfix); lit/unlit/wireframe view modes; translucent + masked materials (sorted pass,
opacity pin on the material graph); blueprint math palette completion + While/For with IR-level
runaway guard + DoOnce/FlipFlop/Gate; input/collision/custom events firing in both sims + event
dispatchers; blueprint debugger (breakpoints, wire values, debug runs, Simulate seams,
sim_step_fixed); Details deep editing (lists/structs/EntityRef pickers — joints authorable —
add/remove component with undo); material-instance override editor; trigger/blocking volumes
(overlap events parity-gated); splines (Catmull-Rom math + viewport polyline); foliage painting
(deterministic stroke-seeded brush, sparse undo, player-mirrored); OBJ import; named content
collections; Audio Mixer panel.

**Deferred (next parity waves):** particles/VFX, FBX/USD, level streaming/world partition,
decal render pass (component slot shipped), point/spot shadow maps, Delay node + ForEach/vector
pins, graph→.inf_act authoring (prerequisite for live-Simulate breakpoint highlighting),
spline meshes + per-point gizmo, mesh-asset foliage palettes, vgeom/skinned translucency,
material functions, multi-viewport, PIE player options.

---

## 12. Next-Gen Wave — Phases 16–25 (planned 2026-07-31; **P16–P20 COMPLETE**)

The 16-phase master plan (§6) and UE-Parity Wave 1 (§11) are complete and CI-green: the engine
now does what a mature traditional engine does. This wave pushes past that workflow —
planet-scale streaming worlds, a living sky, Lumen/Nanite-class completion, biome-driven PCG
with a structure grammar and fully enterable furnished buildings, realistic water, volumetric
(diggable) terrain, deformation and destruction, an embedded DCC suite, and in-engine
photogrammetry. Foundation-first order, one phase in flight at a time, each sub-phase ≈ one
reviewable commit.

**Execution doctrine:** every phase keeps the house gates — determinism (replay traces),
goldens, PIE == shipping, CI on three OSes; schema bumps land at most one per phase via the
frozen-record + downgrade-bless pattern.

### Phase 16 — World scale & streaming foundation

**Goal:** planet-scale becomes an engineering fact, not an aspiration. **Done when:** a
≥ 16k×16k source heightmap imports through the wizard (`meters_per_sample` ≥ 8 → tens of km),
flies in PIE with tiles paging under a bounded residency budget, cooked == uncooked, and
streamed-scene ms budgets ratchet in CI (120 fps-class headless; real fps human-verified).

> **STATUS: Phase 16 COMPLETE** (2026-08-01) — **local gates green; CI pending push.** (This
> block is written with the commit rather than after the CI run, unlike the earlier phases';
> it says so rather than claiming a green it has not seen.) The whole phase is
> pinned by one composed gate — `samples/phase16-world` + `runtime/inf-player/tests/phase16_gate.rs`:
> a wizard-imported streamed terrain, a partitioned world on top of it, a **second inline
> terrain**, and a scripted `StreamingSource` walk with a diverging render camera.
>
> **P16.1 mmap zero-copy packs** — `memmap2` backing in `inf-asset`'s pack reader with a
> borrowed `read_ref() -> &[u8]` for uncompressed, 16-byte-aligned, xxh3-verified-once entries;
> streaming-class kinds (terrain tiles, `.inf_part`, vmesh) cook **uncompressed** so a page is a
> sub-slice of the mapping; `PACK_FORMAT_VERSION` bump with back-compat read; wasm keeps
> whole-file reads. **P16.2 Units doctrine** — `docs/memos/units-doctrine.md` + the
> CLAUDE.md/ROADMAP rule (1 world unit = 1 metre, SI everywhere, no unit-scale factor ever),
> and the loose constants named as their files were touched. **P16.3 Terrain tile streaming** —
> terrain leaves the `.inf_lvl` blob for the `.inf_terrain` asset (header + sorted tile
> directory + 16-byte-aligned blobs across a cook-time LOD pyramid); `TerrainData` becomes the
> *resident working set* over a `TileStore` (pack or loose file); per-tile change stamps replace
> the whole-terrain counter; and **the determinism doctrine** — sim wants (level-0 pages around
> terrain *observers*) load synchronously at the fixed-step boundary into the ECS component,
> camera wants (a quadtree cut with hysteresis) land in a second working set the world holds no
> reference to. **P16.4 Huge-heightmap import** — banded PNG16/EXR row decode + a chunked
> importer that emits tiles and pyramid straight into the payload (`O(one tile row)` live pages,
> byte-identical to the whole-image path and to any job-pool size), the Terrain Import wizard,
> and **sculpt/paint on a streamed terrain** with save-time write-back that is byte-equal to a
> full rebuild. **P16.5 World partition / level streaming** — cook-time grid-cell partition into
> a derived `.inf_part`, a runtime cell manager that spawns/despawns at the deterministic sync
> point only (loading may be early; activation may not be late), a `--debug-cells` overlay, and
> PIE == shipping on the partitioned sample. **P16.6 Multi-terrain & budgets** — see below.
>
> **P16.6, in detail.** (1) **Multi-terrain**: `RenderScene.terrains` is a `Vec`, the terrain
> pass loops it with per-terrain patch assembly, per-terrain splat-material uniforms and a GPU
> tile cache keyed by `(RenderTerrain::id, TerrainTileKey)` — two terrains routinely share tile
> coordinates, so the coordinate-only key was a live collision; both projector MIRRORs
> (`inf_viewport::host::rebuild_scene`, `inf_player::render::project_scene`) emit **every**
> visible terrain — the editor in document order, the player in `Guid` order, each deterministic
> for its own side — and both stamp `terrain_id_from_guid`, which is what makes a PIE-vs-shipping
> comparison of the projected scene match terrains up by *identity* rather than by index;
> `EditorTerrainStreams::retain_only` keeps *all* live streamed GUIDs (it previously evicted a
> second terrain's whole payload every frame). Every cursor path — sculpt, paint, drag-drop spawn,
> foliage — now resolves through one `terrain_probes` seam, so a **streamed** terrain answers from
> the pages the streamer has paged in rather than from the document's (by-design empty) set:
> picks take the **nearest** hit along the ray (a stroke in progress restricts to the terrain it
> started on), scatters take the **topmost** surface, since foliage falls from above. Existing
> single-terrain goldens are byte-identical under `INF_GOLDEN_STRICT`; a two-terrain structural
> scene joins the harness.
> (2) **`.inf_terrain` header v2**: the pyramid options are recorded (128-byte header, 56 bytes
> reserved); a v1 payload keeps loading forever and reports its options as **`None` — unknown,
> not "the defaults"**, which is what let `warn_on_pyramid_reshape` narrow to the one case it can
> still honestly describe. (3) **Streamed-scene budgets** in `inf_player::budget` on the §8
> ratchet, asserted headless over the gate scene: `STREAMED_STEP_BUDGET_MS` 4.0 (measured 0.18),
> terrain residency ≤ 16 MiB (measured 5.65), cell residency ≤ 256 KiB / 8 active cells
> (measured 2.8 KiB / 4). (4) **`CellStreamStats::unresolved_refs`** — the live count of
> cross-cell `AttachedTo`/joint references that are currently disconnected, logged once per
> referrer; the runtime face of the cook's `cross_cell_refs` warning. (5) A third **cook
> advisory**, `partition::streamed_terrains` — see the terrain × partition boundary below.
>
> **Schema:** scene **v9** (terrain becomes an asset ref + streaming settings) and **v10**
> (world-partition metadata) — a deliberate second bump in one phase, because v9 shipped before
> the partition design existed and retro-fitting it would have meant re-blessing bytes that were
> already committed and already load. The `.inf_terrain` **v2** header is asset-internal and
> costs the scene schema nothing.
>
> **The terrain × partition boundary, stated.** A `Terrain` "occupies space", so the partitioner
> bins it by its entity **origin** — while its heightfield spans kilometres from that origin. An
> unmarked terrain in a partitioned level therefore lands in one cell and **the ground despawns
> under the player** when a streaming source leaves it; enabling partitioning on a level that
> already has terrain produces this every time, with no symptom until somebody walks far enough.
> v1 does not auto-promote (a fixup would make residency depend on which *components* an entity
> carries, and a cell-local patch of inline ground is a legitimate thing to author), so the cook
> **warns** — naming the entity, its cell, its `.inf_terrain` and the remedy, mark it
> `AlwaysLoaded` — exactly as it already does for a streamed Blueprint actor. Both committed
> partitioned samples author it that way. Binning a terrain by its real **footprint** (or
> splitting it across cells) so that only a genuinely cell-local terrain streams is the deferred
> fix; it is tracked on `inf_scene::partition::streamed_terrains`, whose docs say the diagnostic
> narrows the day it lands.
>
> **Human-verified remainders (not asserted by CI, and honestly so):** the real-hardware frame
> rate — CI can only bound the CPU-side streaming work (see `inf_player::budget` on why a
> millisecond on a shared runner is not a millisecond on a target machine), so the 120 fps-class
> claim itself needs a GPU and a stopwatch; the *visual* pass on streamed terrain (LOD pops,
> fine↔coarse skirt slivers at grazing angles, splat continuity across a cut); a **UX pass on the
> Terrain Import wizard** with a real multi-gigabyte source; the literal **16k × 16k import**
> (`terrain_import::huge_heightmap_16k_imports`, `#[ignore]`d — ~268 M samples and ~1 GB of
> payload cannot run in a CI job that also cooks it twice, so the gate scene runs the same
> pipeline at 1025² / 8 m / 8.2 km and the 16k pass is run by hand when the importer is
> profiled); and the phase demo recording.
>
> **Deferred, with where each is tracked:** a **disk-to-disk streaming rewriter** so a
> write-back does not stage two whole payloads in RAM (`inf_terrain::writeback` module docs —
> the same follow-up P16.4a's chunked import documents from the other side); **PIE-wire streamed
> terrain**, i.e. the editor's in-process Simulate paging like the subprocess player does
> (`inf_editor_core::terrain_stream` — the editor computes no sim wants today); a **dynamic actor
> map** so an entity that streams *in* gains a ticking Blueprint (`inf_player::cell_stream`
> module docs — the actor map is fixed at `RuntimeSim` construction; content marks such entities
> `AlwaysLoaded` meanwhile, and both partitioned samples' READMEs say so); **footprint-aware
> terrain binning**, so a terrain need not be `AlwaysLoaded` to survive partitioning
> (`inf_scene::partition::streamed_terrains`, and the cook advisory it feeds); a **multi-anchor
> pyramid seam** — `TerrainAssetBuilder::with_origin` exists but every terrain is anchored at
> zero, so two terrains cannot yet share one pyramid across their boundary; a **background spawn
> pool** for cell activation (today a cell that reaches its activation step unloaded blocks the
> step — the documented v1 semantic, mitigated by the prefetch margin); and a **virtual-texture
> revisit** for terrain splat weights at 256² pages (§6 P10 deferral, unchanged — the pyramid is
> heights-only and coarse rings read the level-0 weight page).

- **P16.1 mmap zero-copy packs** — 1. `memmap2` backing in `inf-asset`'s pack reader with a
  borrowed `read_ref() -> &[u8]` for uncompressed, 16-byte-aligned, xxh3-verified-once entries;
  2. streaming-class kinds (terrain tiles, vmesh) cook uncompressed — zstd path unchanged, wasm
  keeps whole-file reads; 3. `PACK_FORMAT_VERSION` bump with back-compat read.
- **P16.2 Units doctrine** — 1. `docs/memos/units-doctrine.md` + the CLAUDE.md/ROADMAP rule:
  1 world unit = 1 metre, SI everywhere, no unit-scale factor ever; 2. name the loose constants
  (editor grid spacing, snap increments) as those files are next touched.
- **P16.3 Terrain tile streaming** — 1. terrain leaves the `.inf_lvl` blob for a new
  `.inf_terrain` asset (per-tile blobs + index, new `AssetKind` + pack code); 2. `TerrainData`
  gains residency — a resident map + a cold-store trait over the asset, camera-radius
  load/evict on the job pool; 3. per-tile versions replace the whole-terrain version counter;
  4. a cook-time tile LOD pyramid feeds the outer clipmap rings so far terrain never needs
  full-res residency; 5. sculpt/paint keeps the live-stroke + `EditCommand` contract against
  resident tiles.
- **P16.4 Huge-heightmap import** — 1. banded/tiled decode (PNG16 row bands, tiled EXR,
  float-metre EXR) replaces the whole-image decode; 2. tiles + pyramid emitted straight to
  `.inf_terrain` on the job pool with progress; 3. a Terrain Import wizard — drop a file, set
  the real-world extent, import in the background, walk it. The "import a terrain and get
  developing" UX gate.
- **P16.5 World partition / level streaming** — 1. cook-time grid-cell partition of `.inf_lvl`
  (entities binned by position, `AlwaysLoaded` flag); 2. a runtime streaming manager loading
  cells by camera radius through the existing `LevelSource`/`WorldBuilder` seam, extended for
  incremental spawn/despawn; 3. a streaming debug overlay; 4. PIE == shipping on a partitioned
  scene (the editor stays single-document in v1).
- **P16.6 Multi-terrain & budgets** — 1. lift "first visible terrain wins" in the viewport host
  and the player render mirror (`RenderScene.terrains`, per-terrain GPU cache keys, nearest-hit
  terrain picking); 2. `.inf_terrain` header v2 records the pyramid options; 3. streamed-scene
  frame budgets + residency memory ceilings on the ratchet; 4. `CellStreamStats::unresolved_refs`;
  5. the composed **Phase 16 gate** (`samples/phase16-world`).

Schema **v9**: Terrain becomes an asset reference plus streaming settings (P16.3).
Schema **v10**: world-partition metadata — the `StreamingSource` / `AlwaysLoaded`
components plus a `PartitionSettings` block on the level (P16.5). A deliberate second
bump in one phase: v9 shipped before the partition design existed, and retro-fitting it
would mean re-blessing bytes that are already committed and already load.

### Phase 17 — Ultra Dynamic Sky & atmosphere

**Goal:** the engine's default look becomes a living sky. **Done when:** a time-of-day sweep
holds goldens (dawn/noon/dusk/night), clouds are deterministic under the psin/pcos doctrine,
PIE == shipping on the sky-state trace, and new levels default to the dynamic sky.

Starting point: the sky is a 44-line three-colour gradient shader, `SkyParams` has zero
writers, and the sun is a `SUN_DIR` constant consumed by the sky, GI, and shadow passes — no
atmosphere, fog, cloud, or TOD code exists anywhere.

> **STATUS: Phase 17 COMPLETE** (2026-08-01) — **local gates green; CI pending push.** (Written
> with the commit rather than after the CI run, like Phase 16's, and saying so rather than
> claiming a green it has not seen.) The engine's default look is a living sky: a new level
> boots at 10:00 on the June solstice under a Hillaire-class physical atmosphere with a
> raymarched cumulus deck drifting on a 6 m/s westerly, and every part of it — the sun, the
> clouds, the weather, the rain — is a pure function of the document rather than of a wall
> clock, a frame index or a machine.
>
> The whole phase is pinned by one composed gate,
> `runtime/inf-player/tests/phase17_gate.rs`: a 600× time-of-day ramp with **two weather
> transitions driven through the Blueprint host seams** (Clear → Storm → Snow), traced across
> the full sky state — clock, sun, projected sun, cloud wind, cloud field, fog, the weather
> block, the blend countdown, the precipitation offsets and count, and the P22 snow-accumulation
> rate — asserted bit-identical across two runs, bit-identical between a cooked pack and a PIE
> payload, and bit-identical across ECS pool sizes.
>
> **P17.1 Sun & time of day** — `SUN_DIR` **deleted**; a `TimeOfDay` + `SkyAtmosphere` component
> pair projected into `RenderScene.sun` by both scene builders; deterministic solar math in
> `inf_math::solar` (Spencer fits, a vector topocentric transform on `psin64`/`pcos64` so no
> `asin`/`atan2` is ever needed); the **singleton problem solved once** in Ring 0 by
> `sky_authority` (lowest `(Guid, Entity)`, archetype-scoped); the clock advanced in both fixed
> steps; a `sky.*` Blueprint namespace and zero-code sequencer keying. **P17.2 Physical
> atmosphere** — transmittance + sky-view LUT compute passes on Bruneton/Hillaire
> parameterizations, a rewritten `sky.wgsl` drawing sun and moon discs and an integer-hashed
> starfield, aerial perspective + exponential height fog in every lit pass, and the
> `EnvBinding` generation key extended to `(targets, atmosphere)` — the first resizable resource
> in that bind group, exactly as P13's comment predicted. **P17.3 Volumetric clouds** — a
> Perlin–Worley shape volume and a Worley detail volume baked from a pure-integer hash, marched
> in a dedicated pass between the opaque and translucent ones (two occlusion mechanisms: the
> hardware test on the slab-entry depth, plus a march clamp against the read-only depth
> texture), a 512² cloud-shadow map sampled beside the CSM, and CPU/GPU parity gated at 1 LSB.
> **P17.4 Weather states + the phase gate** — below.
>
> **P17.4, in detail.** (1) **The weather state's shape** was the batch's real design call, and
> the note is worth keeping: the alternative was a pure `from → to → t` state machine, which is
> smaller and is wrong here for two reasons. The sequencer keys **reflected component fields**,
> and a blend fraction is not something a curve can be drawn through — P17.1 bought "zero
> sequencer code" by making the clock a plain reflected field, and a state machine would have
> spent it. And it leaves the Details grid with nothing authorable between presets, i.e. dead
> controls. So the **live values are the component** (coverage, type, wind X/Z, fog density,
> precipitation, snowiness) and the transition is two extra fields beside them
> (`weather_target`, `weather_blend_remaining`). `advance_weather` closes `dt / remaining` of
> the gap each fixed step, which is linear **in exact arithmetic** — after `n` steps the
> surviving gap is `∏(1 − dt/(T − i·dt)) = (T − n·dt)/T` — and needs **no `from` snapshot** to
> be. In f32 it is linear to within a measured **17 ULP over a 240-step (4 s at 60 Hz) blend**,
> i.e. ~1e-6 of a unit range, which is why the gate asserts linearity against a 1e-3 envelope
> (loose enough for the accumulation, tight enough that a smoothstep — off by 0.09 of the range
> at `t = 0.25` — fails at once). The **endpoint** is exact rather than merely close: the last
> step *assigns* the target instead of closing the final `k = 1`, so "a settled blend equals its
> preset" is a bit-identity, not a tolerance. A blend also takes `ceil(T/dt)` steps and
> occasionally one more, because `1/60` is not representable and the countdown accumulates —
> the gate asserts the count is 240–242 rather than pretending it is 240.
>
> A related sharp edge, found in audit: the countdown is `f32`, so a large enough
> `weather_blend_remaining` makes `remaining − dt` round back to `remaining` — the blend freezes
> and `advance_weather` writes the component every step forever, voiding the "settled ⇒ the
> blender never writes" contract everything else rests on. `MAX_WEATHER_BLEND_S` (3600 s) is
> that ceiling, defined once in Ring 0 and referenced — never re-spelled — by both doors into
> the field (`sky.set_weather` and `WeatherDto::to_component`), plus a **no-progress backstop**
> inside `advance_weather` for the hand-edited sidecar that bypasses both. Two conditions, not
> one: the arithmetic test (`remaining − dt >= remaining`) catches the true freeze for any `dt`,
> and the ceiling test catches the band just below it where the subtraction still moves but by
> one ULP a step — sixteen million steps to settle, i.e. frozen in practice. Once settled the blender never writes those fields again, which is precisely what
> lets a sequencer track or a Details edit own them.
> (2) **Driven, not duplicated.** `ResolvedSky::weather()` decides, in Ring 0, whether the
> `weather_*` block or the authored `cloud_*`/`fog_density` fields are in force — the
> `cloud_time_s` precedent, for the same reason: it is exactly the sort of one-line derivation
> two byte-identical MIRROR bodies eventually stop agreeing about. With weather off the
> projection reproduces v13 field for field (asserted over a whole day-ramp), which is the
> byte-stability promise for every existing level; with it on, coverage, type, the wind that
> drifts the clouds *and* slants the rain, and the fog density all come from the weather block.
> (3) **Precipitation v1** is a dedicated ~200-line pass, not a VFX system, and says so. No
> vertex buffer: `draw(0..6, 0..count)` with a particle's entire state derived from
> `cloud_hash(i, salt)` — the same pure-integer avalanche the cloud field uses, already pinned
> bit-for-bit — plus three displacements the CPU wraps in `f64` (`wind·t`, `fall·t`) exactly as
> `wind_offset` does, because a day of 9 m/s fall is 780 km and an `f32` metre there has 6 cm of
> resolution. **World-anchoring without a world coordinate**: the shader gets the camera's world
> position *modulo the box*, so `p = eye + wrap_signed(base + drift − eye_mod, box)` is congruent
> to `base + drift` in world space — the lattice is locked to the world, the rain does not slide
> with the camera, and the floating origin never enters the expression so a rebase cannot pop
> it. Depth is the **hardware test alone** (read-only attachment, `Greater`, writes off): the
> cloud pass needed a manual depth read because a raymarch has to clamp `t_far` at geometry
> *inside* the slab, whereas a particle is a flat quad at one depth, and the hardware test
> resolves per MSAA sample where a manual load would not.
> (4) **The honest calibrations, and the batch's only supra-physical constants.** A raindrop is
> ~2 mm across and genuinely sub-pixel at any resolution, so a physically-sized quad rasterizes
> to nothing (or to a flickering scatter of hit samples). Three consequences, disclosed together
> in `precip.rs` rather than one at a time: the drawn **sizes** are all larger than the
> hydrometeor they stand for — `RAIN_RADIUS_M` 20× a drop's radius, `SNOW_RADIUS_M` ~15× a
> flake's, `RAIN_STREAK_M` ~7× the 0.15 m a 9 m/s drop smears in one 60 Hz frame; the quad is
> **widened** to at least 1.3 px with its alpha divided by the same factor (the standard
> sub-pixel-geometry treatment, energy-conserving) with that compensation **capped at 4×**; and
> the **sky gain is above 1** (1.15). The first two share one reason — 48 000 particles stand in
> for millions, so a sample can only read as a sheet by being drawn bigger than what it
> represents, and full alpha compensation makes the far half of the sheet vanish. The third is
> geometry: a drop gathers light from the whole hemisphere while `atmos_sample_skyview` returns
> the one direction it is seen against, so at a gain of exactly 1 rain composites over the sky
> to precisely the sky — an identity, not a look. Everything else in the layer (fall speeds, box
> extents, wind, the accumulation rate) is real.
> (5) **The P22 hook.** `ResolvedSky::snow_accumulation_rate()` returns
> `intensity × snowiness × SNOW_ACCUMULATION_MAX_M_PER_S` (1.4e-5 m/s ≈ 5 cm/hour at full
> blast) in Ring 0, so P22.1's deformation code consumes a rate rather than re-deriving one.
> Nothing renders accumulation yet; the gate asserts it is exactly zero through the rain,
> reaches the documented value once the Snow preset settles, and passes through **continuous
> intermediate values** during the blend, which is what an integrator needs.
> (6) **Blueprint + sequencer + editor.** `sky.set_weather(preset, blend_seconds)` plus three
> getters; the preset crosses as a **`Str`** (the `input.is_down` precedent — no new `PortType`,
> no lowering special case, and a typo is a documented **no-op** rather than a different sky),
> and `blend_seconds` is read literally so an unwired pin (`0.0`) changes the weather *now*,
> which is what the node looks like it will do. `set_weather` also **enables** the block: a
> script naming a preset means it, and doing nothing because a checkbox was clear is the worst
> of the available behaviours (asserted, not assumed). Zero IR change and zero transpiler
> change again. The eleven `weather_*` fields are ordinary reflected `f32`s, so the sequencer
> keys them with zero sequencer code. World Settings grew a Weather section — five preset
> buttons that **snap** (an idle editor runs no fixed step, so a blend there would simply never
> advance), a blend time, the explicit params, and `precipLabel` / `cloudCoverLabel` /
> `fogVisibility` readbacks. `WeatherDto` is a block of its own rather than more fields on
> `SkyAtmosphereDto`, because it is a different question with a different UI, and its own
> `edit_weather` transaction means clicking "Storm" reads **Weather** in the undo history.
>
> **Schema v14 — the third bump of the phase, for the third time for the same reason.**
> `.inf_lvl` payloads are bincode, which reads a fixed field count positionally, so growing
> `SkyAtmosphere` by 11 fields makes a v13 record **stop short** of what the decoder expects and
> read on into the next entity's bytes, silently. The v13 shape freezes as `SkyAtmosphereV13`
> inside `EntityRecordV13` / `SceneFileV13` in **both** codecs, and `into_current` lifts a v13
> level with **weather disabled** — which leaves the authored cloud and fog fields driving the
> sky exactly as they did, i.e. what a v13 level meant.
> `v14_weather_is_wider_on_the_wire_than_v13` pins the delta at exactly **38 bytes**, priced
> field by field: 1 bool + `varint_len(default_variant_index)` + 9 × f32. That middle term is a
> named computation rather than a literal for the same reason the v13 test spelled out
> `cloud_seed`: `WeatherPreset` is a fieldless serde enum written as its **variant index**, and
> `bincode::config::standard()` is varint, so its cost is a function of the *default preset's
> position* rather than of its type — a preset inserted before `Clear` would otherwise fail
> saying "the weather block grew", which would be a lie. A `scene_v13.inf_lvl` fixture is
> blessed in both crates from a frozen writer and **byte-compared against each other** (the
> mirror lock); the downgrade direction is asserted as a property, not left to a bless-only
> path. Every committed sample and template changed by **exactly one byte, at offset 0**
> (`13` → `14`) — verified programmatically, every file's length unchanged and offset 0 the only
> differing byte — because no committed content carries a `SkyAtmosphere`.
>
> **Goldens: three added, none re-blessed.** `weather_storm_noon` (the Storm preset over a low
> storm deck — asserting the ceiling covers >90 % of the sky band *and* darkens it, that the
> rain perceptibly changes >1 % of the frame, and that it reaches **6 of 8** vertical bands, so
> a single bright artefact cannot pass), `weather_fog_dawn` (the Fog preset's 6e-3 m⁻¹ ≈ 500 m
> visibility, measured as the frame's *darkest* pixel climbing — two identical-albedo,
> identical-screen-size walls at 30 m and 900 m, the `aerial_fog` construction reused), and
> `weather_snow_dusk` (flakes measurably **warmer at dusk than at noon**, measured only over the
> pixels the precipitation actually occupies, which is the single assertion that would catch
> precipitation shaded by a constant instead of by the sky-view LUT). Beside them, four
> non-image gates: the off path is byte-identical three ways (absent / disabled / zero
> intensity), a 2.5 m wall cuts the rain behind it to under a third of the open-air rate
> (mutation-verified by dropping `depth_compare`, which takes the ratio to ~1.0), the field
> follows the level clock, and the tier clamp reduces the particle count — that last one caught
> its own first draft, which compared a Low-tier frame against a High-tier control and reported
> Low drawing **thirty times** the rain of High, all of it the LUT-size difference. **All 33
> pre-P17.4 goldens are byte-identical**, verified the P17.2/P17.3 way: `INF_BLESS_GOLDENS=1`
> over the whole suite, `git status` reporting zero changed PNGs.
>
> **Cost.** Measured GPU-fenced at 640×360 on an RTX 4070 Ti, with the *heaviest* state a preset
> can produce (Storm: solid cover, full rain), against the same frame with the whole sky off
> (0.299 ms): **+0.036 ms Low, +0.180 ms Medium, +0.393 ms High.** Deliberately **no new ratchet
> constant**: §8's rule is that a budget tripwire only ratchets down, which makes each one a
> standing obligation, and paying it for an off-by-default feature whose worst measured cost is
> four tenths of a millisecond would be paying it for nothing. `sky_stack_cost_per_tier` asserts
> the composed sky-on frame stays inside the existing `FRAME_BUDGET_MS` — the same tripwire,
> now exercised with every Phase-17 layer on at once, which was previously untested — and
> prints the per-tier numbers rather than asserting them, because an absolute millisecond on one
> machine is not a contract.
>
> **Key decisions, in one place.** *The `SUN_DIR` retirement mechanism*: `SunParams::default()`
> keeps the retired constant **un-normalized** (`Vec3::new(0.45, 0.75, 0.3)`) because all three
> of its call sites wrote `.normalize()`, and reproducing the *arithmetic* cannot drift by an
> ULP the way transcribing the *result* can — which is why all 23 goldens survived P17.1
> untouched. *The v12/v13/v14 ladder*: three bumps in one phase, each a component growing inside
> an unmoved entity slot (the v8→v9 `TerrainV8` shape), each with a frozen record carrying its
> **own literal defaults** so a frozen record can never move when the live component is re-tuned,
> and each with a fixture blessed twice and byte-compared. The two ladders lift by different
> routes — `inf-scene` in one hop from `SkyAtmosphere::default()`, the editor's through its own
> `into_v13` literals — and `weather_defaults_are_the_documented_ones` in **both** crates is what
> makes those routes equivalent rather than coincidentally equal. *GenCache / ResourceKey
> discipline*: every bind group is keyed on exactly what it embeds — `EnvBinding` and the cloud
> raymarch on `(targets, atmosphere)`, the LUT bake, the cloud bake and the precipitation pass on
> the atmosphere generation alone, because nothing they bind is viewport-size-dependent and
> rebuilding on a window resize would be a lie about what invalidates them. The failure this
> guards is **quiet**: wgpu keeps the old texture alive as long as a bind group references it, so
> a stale key produces no validation error and no black frame, just last tier's pixels that a
> determinism gate would happily call stable. *The depth clamp*: the cloud march's `t_far` clamp
> against the scene depth is what stops a summit *inside* the deck from being veiled by the cloud
> behind it — the hardware test cannot, because the fragment's depth is the slab's entry plane,
> which is genuinely in front of the summit. Measured 0.275 vs 0.588 composited alpha with and
> without it.
>
> **Honest human-verified remainders.** Every GPU path in this phase is human-verified, as they
> all are: that the four TOD sweep images, the four cloud images, the three weather images and
> the default level actually **look good** on hardware is a subjective bar no assertion reaches,
> and "gorgeous" in the phase's own framing is the one criterion this ledger cannot check off.
> Specifically outstanding: a visual pass on dawn/dusk colour and the night gradient at real
> resolution; whether the precipitation reads as rain rather than as noise at 1080p and 4K (the
> goldens are 320×180, where a widened drop is a large fraction of a pixel); whether the
> weather blend *feels* like weather changing at the default 8 s; and the Phase 17 demo
> recording.
>
> **Deferred, with where each is tracked.** Hillaire's 32×32 **multiple-scattering LUT** (the
> single `ATMOS_MULTI_SCATTER` constant stands in — `crates/inf-render/src/atmosphere.rs`) and
> his 3D **froxel aerial-perspective LUT** (the homogeneous v1 with a horizon-clamped in-scatter
> is documented in `atmosphere_lut.wgsl`). **TAA-integrated temporal cloud jitter** — the
> standard way to buy back march steps, and a per-frame-index dependence, which is exactly what a
> byte-identical determinism gate forbids; it needs a jitter that is a function of the *level's*
> clock rather than the frame counter (`crates/inf-render/src/shaders/cloud.wgsl`). The **moon's
> roll about the view axis** is pinned to the local horizontal rather than derived from the
> ecliptic (`sky.wgsl`; invisible at 0.5° across). **GI ↔ sky coupling → P18.4**: cloud shadows
> reach the *direct* lighting of every lit pass but not the probes, which still read the gradient
> constants, so a heavy overcast dims the sun on the ground but not the bounce. Precipitation has
> **no collision and no audio** — drops fall through roofs, and rain is silent (the audio command
> queue is the seam, `docs/memos/` audio doctrine); it also has no splashes, no wetness/roughness
> response on materials, and no lightning. **Weather zones / volumes** — one weather state per
> level, so a valley cannot be foggy while the ridge is clear; the `Volume` component is the
> obvious carrier and this is future scope, not a P17 remainder. Clouds remain a **flat slab**
> (no orographic lift, no terrain-varying base), and precipitation is a fixed-size box around the
> camera rather than a projected frustum volume.

> **STATUS: P17.1 COMPLETE** (2026-08-01) — **local gates green; CI pending push.** The sun
> stopped being a compile-time constant: `inf_render::camera::SUN_DIR` is **deleted**, and every
> consumer (the sky glow, the terrain shader, the mesh/skinned/vgeom no-light fallback, the CSM
> caster fallback, the GI sun fallback) now reads a `RenderScene.sun` block projected from a
> `TimeOfDay` + `SkyAtmosphere` component pair. Pinned by
> `runtime/inf-player/tests/sun_tod.rs`: a cooked pack and a PIE payload run the same scripted
> clock ramp and produce **bit-identical** clock + sun + projected-`RenderScene.sun` traces.
>
> **Deterministic solar math** lives in `inf_math::solar` — not `inf-render` — because the
> editor projector, the shipped-player projector, the fixed-step clock, the `sky.*` Blueprint
> host and the sequencer all need it and none of them should pay for `wgpu` to get it;
> `inf-math` is the leaf every one of them already reaches, and it is where the bit-portable
> trigonometry lives. Declination and the equation of time are Spencer (1971) fits (≈0.03° /
> ≈0.3 min); the topocentric transform is written **directly as a vector** (`East = −cos δ·sin H`,
> `Up = sin δ·sin φ + cos δ·cos φ·cos H`, `North = sin δ·cos φ − cos δ·sin φ·cos H`) so no
> `asin`/`atan2` is ever needed and the whole module stays on `psin64`/`pcos64` + `sqrt`/`floor`.
> Equinox and solstice noon elevations land within **0.02°** of exact spherical geometry at
> latitudes from the equator to 78° N; midnight sun and polar night hold above the Arctic circle.
> The psin/pcos law does not strictly *bind* here — a sun direction feeds uniforms, not committed
> bytes; the committed bytes are `TimeOfDay`'s plain `f64` fields, which `advance` touches with
> nothing but IEEE add/mul/floor — but it is followed anyway, because two gates compare numbers
> across process boundaries (PIE vs shipping, replay), and portable polynomials make those traces
> portable artefacts rather than machine-local ones. `elevation_deg`/`azimuth_deg` are the only
> `std`-trig functions and are documented **display-only**. The moon is an explicit v1
> approximation (ecliptic-longitude model: hour angle lags the sun by `phase·2π`, declination
> from `sin δ = sin ε·sin λ`; no lunar inclination, eccentricity or nodal precession — errors of
> a few degrees, and the phase repeats each simulated year because the engine has no year field).
>
> **The singleton problem, solved once.** The two scene projectors walk the world in *different*
> orders (the editor in document order, the player in `Guid` order), so "the first `TimeOfDay` I
> meet" would silently resolve to different entities. `inf_ecs::sky::sky_authority` answers it in
> Ring 0 by **lowest `(Guid, Entity)`** — properties of the data, not of the traversal — and
> `resolve_sky`/`advance_time_of_day`/the four `sky.*` host seams all go through it. The `Entity`
> tie-break matters for the one shape a level can still reach: duplicate `Guid`s from a mangled
> merge, where a `Guid`-only comparison would keep whichever the walk happened to hit first. The
> lookup is `O(clocks)`, not `O(entities)` — `try_query_filtered` restricts it to archetypes that
> actually hold a clock, which is what keeps a Blueprint calling `sky.get_time_of_day()` several
> times a tick from scanning a 50k-entity world (asserted by a relative-scale test: an 8 000-entity
> world must not cost 20× a 100-entity one). What is left in the two MIRRORs is a ~30-line mapping
> into renderer types (neither Ring-0 crate can host it: `inf-render` does not depend on `inf-ecs`,
> and `inf-ecs` must not depend on `inf-render`), compared character for character by
> `inf-editor-core/tests/projector_mirror.rs` — deliberately **not** inside
> `inf_viewport::host`, which is `#[cfg(windows/macos)]` and therefore invisible to the Linux CI leg.
>
> **Diagnostics.** A `SkyAtmosphere` is read *from the authority entity*, so one with no
> `TimeOfDay` beside it is silently inert — the level looks configured and renders as if it were
> not. Both shapes (no clock anywhere; a clock elsewhere) emit a one-shot `tracing::warn!` naming
> the entity and the remedy, latched because `resolve_sky` runs every frame in two projectors.
>
> **Byte-stability, deliberately.** `SunParams::default()` is the retired constant kept
> **un-normalized** (`Vec3::new(0.45, 0.75, 0.3)`), because every one of `SUN_DIR`'s three call
> sites wrote `.normalize()` and reproducing the *arithmetic* cannot drift by an ULP the way
> transcribing the *result* can; `unit_direction()` does the same multiplication on the same
> bits, pinned by a `to_bits()` comparison. A level with no clock projects exactly that, so
> **all 23 goldens are byte-identical under `INF_GOLDEN_STRICT` — nothing was re-blessed.** The
> `SkyAtmosphere` gradient defaults are `SkyParams::default()` verbatim, and `sky_dim()` is
> exactly `1.0` whenever the sun is more than ~9° up, so a daytime scene draws the authored
> colours untouched; the smoothstep only bites at dusk. The component defaults (10:00 UTC, day
> 172, 48.9° N) put the sun within **1.6°** of the retired constant, so a scene that opts in
> keeps essentially the look it had.
>
> **Where time advances:** once per fixed step, at the top, in **both** `SimSession::fixed_step`
> and `RuntimeSim::fixed_step`, so blueprints, the projected sun, shadows, GI and audio all
> observe one consistent clock for the step. `rate` defaults to **0** (frozen) — a level opts
> into a moving sun explicitly — and nothing outside a fixed step ever calls it, so an idle
> editor never moves the sun or dirties the document. Simulate's enter/exit snapshot restores the
> authored clock on Stop for free.
>
> **Schema v11** — the entity record appends `time_of_day` / `sky_atmosphere` in **both** codecs;
> `LevelSettings`/`RuntimeSettings` are untouched, so only the entity record freezes
> (`EntityRecordV10` in each, with `into_current` + `from_current`), the v10 file record is
> repointed at it, and every older decode arm gains one `.into_v10()` hop. A `scene_v10.inf_lvl`
> fixture is blessed in both crates from a frozen writer, byte-compared **against each other**
> (the mirror lock), and loads forever; the downgrade direction is asserted as a property, not
> left to a bless-only path. Every committed sample and template grew by exactly
> `2 bytes × entity_count` — two `Option::None` tags — plus the one schema byte: the single-byte
> standard, and the arithmetic *is* the contract.
>
> **Blueprint + sequencer:** a `sky.*` namespace (`get`/`set_time_of_day`, `get`/`set_rate`) as
> four **single-purpose** nodes — a pure node with more than one data output would fan into
> `sky::get::<field>` and force three-segment arms in both hosts. Zero IR change, zero transpiler
> change (`crates/inf-transpile/tests/sky_roundtrip.rs` proves it). `TimeOfDay.seconds` keys from
> the sequencer with **zero** sequencer code — the payoff for making the clock a reflected
> component; `ComponentRegistry::type_path_for` gained a short-name fallback so both
> `"Time of Day"` and `"TimeOfDay"` resolve, and a stored track path survives a display rename.
>
> **World Settings** grew a Time of Day section (time slider with `HH:MM` readback, day-of-year,
> lat/long, rate, plus a live sun elevation/compass readback computed in Rust rather than
> re-implemented in TypeScript). The clock is **not** in `LevelSettings`: the DTO projects the
> authority entity's components, and writing any row calls
> `SceneDoc::edit_time_of_day(tod, create)`, which creates the `Sky` actor + both components +
> five field writes inside **one** undo transaction — so a single Ctrl+Z takes the whole opt-in
> back out. `create` is the DTO's `present` flag pushed down into Ring 1 (where it is testable):
> the panel sends the *whole* settings block on every edit, so without it, editing gravity would
> conjure a sun out of the previewed defaults. `create: false` on a clockless level is a total
> no-op — no entity, no undo entry, no version bump, no dirty flag.
>
> **Honest scope:** the sky is still the 44-line three-colour gradient — P17.2 replaces it with
> the Hillaire LUTs. The moon has a direction and an intensity but nothing draws a moon *disc*
> (P17.2). New levels do **not** yet default to a dynamic sky (that is the Phase-17 done-when,
> not P17.1's); a level opts in from World Settings or by adding the components in Details. The
> GI fallback tracks TOD but probe sky-radiance is still the gradient constants (P18.4). The
> visual pass — dawn/dusk colour, the night gradient, the sun glow at grazing angles — is
> human-verified, as every GPU path here is.

> **STATUS: P17.2 COMPLETE** (2026-08-01) — **local gates green; CI pending push.** The
> 44-line gradient is gone. The sky is a Hillaire-2020-class **transmittance + sky-view LUT
> pair** baked by two compute passes, sampled by a rewritten `sky.wgsl` that also draws the sun
> and moon discs and a procedural starfield, and re-used by every lit pass for **aerial
> perspective + exponential height fog**. `SkyAtmosphere` grew the physical block, and a **new
> level now boots with a real sky** — the Phase-17 done-when.
>
> **Where the physics lives.** `inf_render::atmosphere` is pure and GPU-free: the medium
> (Rayleigh `exp(-h/8 km)`, Mie `exp(-h/1.2 km)` with Cornette-Shanks `g = 0.8`, and the ozone
> tent at 25 ± 15 km that is the whole reason a twilight sky stays blue instead of going
> muddy), both LUT parameterizations, and the closed-form height-fog integral. So the physics
> is unit-tested on **every** CI leg, including the ones with no adapter where the golden
> harness skips: transmittance is monotone with altitude per channel, a ray into the planet is
> opaque, turbidity scales the aerosol terms and nothing else, and a 0.5°-elevation sun is
> **> 10× redder** (r/b) than an overhead one with blue down under 2 % — the single assertion
> that catches a swapped wavelength triple. Every function has a WGSL mirror named in its doc
> comment.
>
> **Units, stated once and enforced.** Atmospheric optics is kilometres and km⁻¹ and fighting
> that would be worse than documenting it, so `inf_render::atmosphere` — and only it — works in
> km. `HeightFog` is **SI metres** throughout (m⁻¹ extinction and falloff, metre altitudes),
> because fog is authored against level geometry rather than against a planet; the single
> conversion happens in `camera_radius_km`. Architecture rule 6 is satisfied by saying which
> unit each quantity is in, not by pretending the engine has only one.
>
> **The LUT parameterizations are the load-bearing choice.** Transmittance uses Bruneton's
> `(u = distance-to-atmosphere-top between its two extremes, v = distance-to-horizon)` mapping
> rather than a linear `mu`, because a linear mapping spends most of a 256×64 texture on the
> ~identical overhead directions and bands the sunset. Sky view uses Hillaire's horizon-pivoted
> warp: `v = 0.5` is **exactly** the horizon and both halves are sqrt-compressed toward it,
> where the gradient is steepest; `u` is the azimuth relative to the sun, sqrt-warped toward it
> where the Mie halo lives. Both mappings round-trip in unit tests, and the sky-view warp is
> asserted monotone with the horizon on the seam — a bad parameterization shows up as a black
> line across the sky, and that is a test, not a bug report.
>
> **Version gating, two different keys.** The transmittance LUT is a function of the **medium
> and the shell alone** — not the sun, not the camera — so it is baked on the first enabled
> frame and then essentially never again. The sky-view LUT additionally keys on the sun
> direction, the sun's radiance, the exposure and the camera radius **quantized to whole
> metres**, so a hovering flycam does not re-bake 60 times a second over sub-millimetre jitter
> that cannot change anything at a 6360 km radius. Both keys are `to_bits()` tuples, so a `NaN`
> parameter cannot make the cache thrash. Unit-tested as a property: moving the sun must not
> touch the medium key, a 248 m climb must move the view key, a 0.4 m one must not, and a fog
> change — a *receiver* parameter — must move neither.
>
> **The `EnvBinding` invariant, extended as its own comment predicted.** P13 wrote: *"If
> shadow/GI resources ever become resizable, this key MUST incorporate their generation."* The
> atmosphere LUTs are the first resizable resource in that bind group — `AtmosphereQuality` is
> clamped **down** by `RenderTier` at runtime, which recreates both textures at a new size — so
> the cache key is now the pair `(targets.generation, atmosphere.generation)` and
> `AtmosphereResources` carries a monotonic generation the renderer bumps on every recreation.
> The failure mode is **quiet**, which is what makes the gate's shape matter: wgpu keeps the old
> texture alive as long as a bind group references it, so a stale key does not validate-error and
> does not blank the frame — the pass simply keeps sampling the previous quality's LUT. So
> `atmosphere_quality_switch_rebuilds_the_env_bind` asserts on the *lit region* of a distant
> fogged wall (near geometry has no aerial term to speak of, and a whole-frame compare is
> satisfied by the separately-keyed sky pass), and the adapter-free
> `pointer_identity_changes_only_when_the_key_does` pins the same rule on a refcounted stand-in
> payload. Both were mutation-verified: dropping the atmosphere generation from the key, and
> making the cache ignore its key entirely, each fail. Shadow/GI stay out of the key, still
> created once, and the comment now says so explicitly rather than by omission.
>
> **vgeom joined the lit family.** `vgeom_mesh.wgsl` was the one lit shader still on the
> AO-only bind (`ShaderKind::Plain` + `AoBinding`), which meant `lit_scene_shader` did *not*
> in fact reach it — so meshlet geometry would have been the only surface in the engine with no
> aerial perspective. It is now `ShaderKind::Lit(2)` on the shared `EnvBinding`, `AoBinding` is
> deleted as its last user, and both vgeom goldens are byte-identical.
>
> **Byte-stability of the off path — measured, not asserted.** The atmosphere is **disabled by
> default** and every consumer branches on one uniform flag; the gradient body survives verbatim
> inside `gradient_sky()`, and each lit shader still computes its historic fixed haze and only
> then overwrites it when the flag is set. Verified the hard way: `INF_BLESS_GOLDENS=1` was run
> over the whole suite and `git status` reported **zero** changes to the 23 pre-P17.2 golden
> PNGs — byte-identical, not within-tolerance. That covers the recomposed shader table, the four
> new `EnvBinding` bindings and the vgeom bind-group swap.
>
> **Goldens added (6, none re-blessed):** `sky_dawn` / `sky_noon` / `sky_dusk` / `sky_night` —
> the TOD sweep P17.4 extends, built from `inf_math::solar` at the *component defaults* so they
> picture what a level actually gets rather than a tuned demo; `aerial_fog` — two walls of
> identical albedo and identical screen size at 50 m and 1500 m (the far one is the near one
> scaled 30× about the eye), so every pixel of difference between them **is** the atmosphere;
> `editor_default` — the default clock's look. The structural assertions are ratios and
> region comparisons, not absolute pixels: the horizon out-brightens and de-saturates against
> the zenith at noon, the twilight band is measurably redder than the zenith at both dawn *and*
> dusk (kept as two goldens because their azimuths differ by ~100°), the night field carries
> star **contrast** rather than a raised black level, and the far wall converges on the sky in
> RGB distance. That last one is deliberate: "gets bluer" would be *wrong* and fails — a hazy
> noon horizon is whiter than the blue hemispheric ambient, so distant geometry gets less blue
> while becoming more sky-coloured.
>
> **Determinism.** `atmosphere_luts_are_deterministic` reads both baked textures back and
> byte-compares two independent bakes — one level below the frame, so a nondeterministic march
> surfaces here instead of as a flaky pixel three passes downstream. The starfield is
> **integer-hash only, no trig anywhere**, per the spirit of the psin/pcos law: it is a pure
> function of world direction, byte-identical across renders, and yaws with the camera rather
> than sticking to the screen (both asserted).
>
> **The one honest approximation.** Aerial perspective is a v1 without a froxel volume: the
> eye→surface segment is treated as homogeneous at the camera's local extinction, and the
> in-scattered colour is the sky-view LUT **with the elevation clamped at the horizon**. That
> clamp is not a fudge — below the horizon the LUT means "the planet seen through air", whereas
> a surface 800 m away needs "the air column between here and there", which for any horizontal
> or downhill ray *is* the horizon's column. Without the clamp distant downhill geometry gets
> **darker** with distance, the exact opposite of aerial perspective (caught by the
> `aerial_fog` golden, which is why it exists). Hillaire's 3D aerial-perspective LUT is the
> documented follow-up, as is the 32×32 multiple-scattering LUT that the single
> `ATMOS_MULTI_SCATTER` constant currently stands in for.
>
> **Schema v12 — the bump the batch was told to avoid, and why it could not be.** `#[serde(default)]`
> rescues *self-describing* formats. `.inf_lvl` payloads are **bincode**, which reads a fixed
> field count positionally, so after growing `SkyAtmosphere` by 13 fields a v11 record **stops
> short** of what the decoder now expects — and the decoder, having no length to stop at, reads
> straight on into the next entity's bytes. Silently. That is the
> same root cause as the standing `skip_serializing_if` law, and it is exactly what the
> frozen-record scheme exists for. The v8→v9 `TerrainV8` precedent applies verbatim: only a
> component's shape changed, so no entity slot moved, `SkyAtmosphereV11` + `EntityRecordV11` +
> `SceneFileV11` are frozen in **both** codecs, and `into_current` lifts a v11 level with the
> physical block at its defaults — a gradient sky and no fog, which is precisely what a v11
> level meant. `v12_atmosphere_is_wider_on_the_wire_than_v11` pins the delta at exactly
> **61 bytes** (1 bool + 11 f32 + a 4×f32 `Color`), which is the mechanical proof that no serde
> attribute could have covered it. Every sample and template changed by **one byte, at offset 0**
> (`11` → `12`); no committed content carries a `SkyAtmosphere`, so the 61-byte block never
> appears in them.
>
> **The default level's sun is now the time of day.** `demo::build` gains the `Sky` authority
> pair — named and shaped exactly like what World Settings creates, so a user who learns it once
> recognises it everywhere — and **loses its authored `DirectionalLight`**, because `project_sky`
> pushes the sun as a directional light in both hosts and a level carrying both would be lit
> twice from two directions and cast two sets of shadows. The defaults are the *component*
> defaults untouched, which is worth more than a tuned demo: 10:00 UTC on day 172 at 48.9° N
> puts the sun ≈ 55° up — high enough for a saturated blue zenith, low enough to keep a
> direction, where a noon sun would flatten exactly the geometry the scene exists to show;
> `rate = 0` so an idle editor never moves the sun and never dirties the document under a user
> who touched nothing; and no height fog, because at a 20 m scene's scale any density that reads
> at distance is invisible here and would be a knob that appears to do nothing.
>
> **World Settings** grew an Atmosphere section — physical-sky and sun-lights-scene toggles,
> sky intensity, turbidity, haze anisotropy, disc sizes, stars, gradient tint, aerial-perspective
> strength and the three fog rows, plus a live **visibility readback** (Koschmieder `3/σ`), because
> "0.0004 per metre" means nothing to an author and "~7.5 km" is checkable against their level.
> `SkyAtmosphereDto` is deliberately numeric-and-boolean only: the five `Color` fields stay in
> the Details grid, which already has a colour widget. It therefore **overlays** rather than
> rebuilds — a fog-slider drag must not silently reset an authored sun colour, which is a test.
> `edit_sky_atmosphere` mirrors `edit_time_of_day` exactly, including the `create` guard that
> stops a gravity edit from conjuring a sky, and creating goes *through* `edit_time_of_day` so
> there is one definition of what a sky authority is and the clock always lands first.
>
> **Tiers + cost.** `AtmosphereQuality` (256×64 / 192×108 at High down to 128×32 / 96×54 at Low,
> with 40/32 down to 24/16 march steps and the star density scaled with it) is clamped down by
> `RenderTier::apply` and `clamp_mobile` — never up, like every other capability knob. A tier
> never turns the sky **off**: a sky the level authored must still be a sky on a weak GPU. The
> sky-view bake is ~20 k threads × 32 steps at High and only runs when the sun or the camera's
> altitude actually moved; it replaces a per-pixel march for every sky pixel at any output
> resolution. Frame budget measured **0.17 ms/frame** (33 ms budget) and the sim step **0.040
> ms** (2 ms) — untouched, as expected: this is all render-side.
>
> **Honest scope:** no clouds, no weather, no precipitation (P17.3/4). The moon's phase
> terminator is the right *shape* (an ellipse at `cos 2πφ`) but its roll about the view axis is
> pinned to the local horizontal rather than derived from the ecliptic — documented in the
> shader, and invisible at 0.5° across. GI probe radiance still reads the gradient constants
> rather than the sky-view LUT (P18.4 owns that). The `night_darkening` / gradient-tint path
> remains the artistic override at strength 0 by default. The visual pass — that the four sweep
> images and the default level actually look good — is human-verified, as every GPU path here is.

> **STATUS: P17.3 COMPLETE** (2026-08-01) — **local gates green; CI pending push.** The sky
> has clouds. A **raymarched volumetric layer** between two authored altitudes, shaped by
> hand-rolled seeded 3D noise, drifting with the level's clock, lit by the P17.2 sun through
> the P17.2 atmosphere, occluding the sky/sun/moon/stars behind it, occluded by the terrain in
> front of it, and casting a soft kilometre-scale shadow on the world. A **new level boots
> with it on**.
>
> **Where the field lives.** `inf_render::clouds` is pure and GPU-free — the parameter block,
> the noise generators, the weather field, the height profile, the two-lobe phase, and a CPU
> **reference implementation of the density function**. So the field is unit-tested on every CI
> leg, including the ones with no adapter where the golden harness skips: the noise is
> seed-stable and bake-order-independent, it tiles seamlessly on all three axes, coverage is
> monotone and genuinely reaches both ends, the height profile closes at the slab's floor *and*
> its ceiling for every species, and the wind wraps into one tile however long the level has
> been running. Every function names its WGSL mirror in its doc comment.
>
> **The noise is integer-hashed, and that is the whole portability story.** Lattice values come
> from `cloud_hash`, a pure-integer avalanche — no trigonometry anywhere in the field, per the
> psin/pcos law the P17.2 starfield established. Perlin's improved-noise gradients are ±1
> cube-edge directions, so a gradient dot product is two adds and cannot round differently on a
> different adapter. The shape volume is a Perlin–Worley base (Schneider's remap: dissolve
> Perlin by inverted Worley fBm, which keeps Perlin's connected topology and Worley's rounded
> billows) plus three single-octave Worley channels; the detail volume is three more. **Single
> octaves, not fBm, is a real constraint and not a simplification**: three Worley fBms would
> each reach 4× their base frequency, putting the alpha channel at 128 cells — past what even a
> 128³ volume can store — so what got baked would be aliasing rather than detail. The tier
> table is checked against that Nyquist relation by a test rather than by hand.
>
> **Committed values are pinned.** `committed_noise_values_are_bit_stable` asserts four
> specific RGBA8 texels and three specific hash outputs as literals. If any of them ever moves,
> a level's sky moved under it and a golden has to be re-blessed **on purpose** — which is the
> difference between a deterministic field and one that merely happens to be stable today.
>
> **CPU/GPU parity, in two gates with two different envelopes.** The bake gate compares every
> texel of both baked volumes against the CPU reference: measured on Windows/Vulkan, **88.0 %
> of shape texels and 91.1 % of detail texels are bit-exact, with a worst case of exactly
> 1 LSB**. The envelope is `≤ 1 LSB everywhere` plus a floor on the exact fraction, and the
> reasoning is stated rather than fudged — WGSL explicitly permits contracting `a*b + c` into an
> FMA, which shifts a result by ~1 ULP and, after the `×255 + 0.5` quantization, by at most one
> LSB, while everything that *could* diverge structurally (the hash, the gradient table, the
> lattice wrap) is pure integer arithmetic and would move whole texels, failing both halves at
> once. The **density** gate is measured end-to-end through the cloud-shadow map, because a
> shadow texel is a Beer–Lambert march of `cloud_density` along the sun and agreeing on it
> means agreeing on the weather bias, the height gradient, the Perlin–Worley remap, the
> coverage dissolve and the erosion, in the right order. It evaluates the CPU reference against
> the **read-back** volumes rather than re-baking them, so a disagreement is attributable to the
> density function and not to the separately-gated bake. Measured **mean |Δ| 4 × 10⁻⁵, worst
> 1.1 × 10⁻³** over 2704 taps against a 2 % envelope; the envelope is wide because hardware
> trilinear filtering carries only ~8 bits of sub-texel precision while the reference filters in
> full f32, so exact agreement is not available at any price.
>
> **Pass placement, and why the obvious home is the wrong one.** Clouds are **not** in the sky
> pass. `SkyNode` is the *first* scene pass — it clears colour and depth — so at the moment it
> runs there is no geometry in the depth buffer to occlude anything, and a cloud drawn there
> hangs in front of the terrain it is supposed to be behind. `CloudNode` is therefore its own
> pass, after every opaque pass (mesh, vgeom, skinned, terrain) and before the translucent one:
> after opaque so the hardware can reject cloud fragments behind the world, before translucent
> so glass composites over cloud as it would over any other background, and inside the MSAA
> scene target so the resolve → TAA → bloom → tonemap chain treats cloud radiance as ordinary
> scene radiance — a cloud edge against the sun blooms without a line of code for it.
>
> **Occlusion takes two mechanisms, because one is not enough.** The fragment writes
> `frag_depth` at the ray's **entry** into the slab with depth writes off and `Greater`
> comparison, which rejects — per MSAA sample, so with antialiased silhouettes — every fragment
> whose geometry is entirely in front of the layer. That is the common case, and
> `clouds_are_occluded_by_geometry` proves it by asserting a walled frame is **byte-identical**
> with and without clouds behind the wall. It is **not** the whole occlusion, and the first
> draft's claim that it was is exactly the sort of thing that ships: a 2 km summit under a
> 1.5–4 km deck is *inside* the slab, sits beyond the entry plane, passes a `Greater` test, and
> would have the entire marched span — including the cloud physically behind the mountain —
> composited over it as a veil. On an 8 km terrain that is not exotic, it is Tuesday. So the
> shader also reads the scene depth for its pixel and clamps `t_far` to the nearest geometry,
> stopping the march at the surface; the depth attachment is bound **read-only** and the same
> view is additionally bound as a `texture_depth_multisampled_2d`, which is the one arrangement
> WebGPU permits for reading the depth you are also testing against.
> `clouds_do_not_veil_geometry_inside_the_slab` measures the composited **alpha** over a mesa
> whose top pokes into a thin deck — an RGB-delta metric was tried first and reported 97 %
> either way, because the cloud sits over a dark mesa in one frame and a bright sky in the
> other and the same alpha then gives wildly different deltas. Mutation-verified: disabling the
> clamp moves the measured alpha from **0.275 to 0.588**.
>
> **What stays approximate**, now that both are in place: the clamp reads MSAA sample 0 only, so
> a pixel *partially* covered by intersecting geometry takes its march length from one sample
> while the hardware test still resolves coverage per sample — a sub-pixel discrepancy at a
> silhouette. And the pass runs before the translucent one, so cloud behind glass composites in
> the right order but cloud *inside* a translucent volume does not; there is no pass ordering
> that would fix that without a per-fragment sort.
>
> **Two calibrations that are unit conversions, not fudges.** A phase-function-only march
> produces radiance ~1/4π of the sun's and renders an overcast noon as soot. The cloud's
> in-scattered sun term is multiplied by the **sky exposure** (`params.y`) for exactly the
> reason `SKY_EXPOSURE_CALIBRATION`'s own doc gives — it is the single calibration between the
> engine's arbitrary light units and the exposure the renderer is tuned for, and it multiplies
> sky radiance, which a cloud is — and by **π**, because the in-scatter source is `E · p(θ)`
> where `E` is *irradiance* while the engine hands over the radiance-like number the PBR loop
> divides by π for a Lambertian. Undoing that convention is what makes a sunlit cloud top
> brighter than the ground under the same sun, which is the correct relationship.
>
> **The dusk fix that made the feature work.** Sun transmittance is sampled at the **layer's**
> radius, not the camera's. With the sun two degrees up the path to a viewer on the ground is
> opaque in every channel, while the path to a cloud three kilometres up is above a measurable
> slice of the atmosphere and still carries red. Sampling at the camera made twilight clouds go
> grey at exactly the moment they should be the brightest thing in the sky — measured r/b 0.33
> before, **1.81** after, against a clear dusk sky's 0.67. The ambient is likewise interpolated
> between the zenith (which a cloud's top sees) and the bright band around the horizon (which
> its base sees), so a twilight deck is not lit as if it were noon. Both are pinned by
> `golden_clouds_dusk`, which compares the same clouds under a noon sun.
>
> **The one approximation that is a diffusion model rather than an integral.** The ambient
> inside a cloud is the sky *above* the layer, which the cloud above it has largely occluded — a
> single-scattering march has nowhere to get that light back from. `CLOUD_AMBIENT_BASE` keeps
> 45 % of the sky at the slab's base as an explicit stand-in for the multiple scattering that
> really carries it there; on the sun side, Hillaire's three-octave multiple-scattering
> approximation does the same job. Take either out and an overcast deck renders as soot, which
> is the single most common failure of a correct-but-incomplete volumetric.
>
> **World shadowing, for one binding.** A low-resolution `rgba16float` cloud-shadow map (512²
> over 20 km = 39 m/texel at High) stores the sun transmittance of the slab at each world-XZ
> point, baked by compute and sampled in the lit passes beside the CSM. Deliberately *not* a
> second cascaded shadow map: a cloud four kilometres up casts a penumbra hundreds of metres
> wide, so a crisper map would only store detail that is physically wrong. The map's centre is
> **snapped to a whole texel** — unsnapped it would both re-bake every frame a flycam breathes
> and slide the pattern by a fraction of a texel each frame, making a *static* scene shimmer.
> The receiver projects up the sun ray to the slab's mid-altitude and reads there, so the
> projection happens at lookup time and the map itself never needs a light-space matrix and
> cannot shear. `rgba16float` costs three wasted channels and buys the one thing that matters:
> `r16float` is not a core WebGPU storage format and `r32float` is not *filterable* without an
> optional feature, and a nearest-sampled cloud shadow is a grid of 39 m squares rather than a
> penumbra. (That was found the way such things are found — a validation error on the first
> golden run.)
>
> **The `EnvBinding` invariant, and why clouds add no key component.** The cloud shadow map is
> the third resizable resource in that bind group, and the cache key is still the P17.2 pair.
> That is correct rather than an oversight, and it takes **two** tests to say
> why, because it is two claims — the first draft had only the first, and only the first is
> about the mapping. `cloud_quality_is_a_total_function_of_atmosphere_quality` pins the
> mapping: injective on the texture sizes (no two tiers share a cloud resource set) and
> deterministic, so the atmosphere generation distinguishes every distinct set of cloud
> textures that can exist. `cloud_quality_is_only_assigned_at_construction` pins the other half
> by scanning the crate's own source: the field is written in exactly one place, `new`, which is
> also the only place that takes a fresh generation — an injective mapping assigned in two
> places would still let the textures change under an *unchanged* generation. Mutation-verified
> (a `probe.cloud_quality = …` anywhere in the crate fails it).
>
> The golden `cloud_quality_switch_rebuilds_the_cloud_binds` pins the rule from outside, and its
> first draft **did not bite**: every whole-frame assertion in it survived dropping the
> generation from the bake's cache key, because with a stale bind group the bake keeps writing
> into the previous tier's views while the frames still differ by tier (the march step counts
> come from the uniform, not from the bind group). The fix is a content assertion — after each
> switch the freshly-created volume is read back and required to match `shape_texel` at that
> tier's resolution; zeros mean the bake wrote somewhere else, and dropping the generation now
> fails on the second tier. Mutation-verified, and said so in the test. A GPU-free
> `the_cloud_bake_key_rebuilds_on_a_generation_bump` mirrors the P17.2 pointer-identity property
> at the cloud key type, so the contract is pinned on CI legs with no adapter too.
>
> **Version gating, two keys again.** The two noise volumes are a function of the **seed
> alone**: coverage, type, wind, altitude and the sun all shape the field at *sample* time,
> which is the entire reason it is worth baking, so dragging a coverage slider re-bakes nothing
> and 8.4 MB of 3D noise is written once. The shadow map keys on everything that can move a
> shadow — the layer geometry, the weather knobs, the wind's current displacement, the sun
> direction, the march budget, the snapped centre — and on nothing that cannot: the camera's
> *altitude* is absent, because the map is a property of the world and not of the viewer. Both
> keys are `to_bits()` tuples, so a `NaN` parameter cannot make the cache thrash.
>
> **Determinism, in three places.** The wind drift is a **deterministic function of the level's
> clock** (`ResolvedSky::cloud_time_s`, defined once in Ring 0 so the two projector MIRRORs
> cannot disagree about it), never a wall clock or a frame counter — two runs at the same time
> of day see the same sky, and `cloud_wind_follows_the_level_clock` asserts that a whole tile of
> drift is a *no-op*, which is simultaneously the tileability proof and what keeps an all-day
> session from quantizing into stair-steps. `cloud_bakes_are_deterministic` byte-compares two
> independent bakes of both volumes and the shadow map, one level below the frame, so a
> nondeterministic bake surfaces there instead of as a flaky pixel three passes downstream.
> And **temporal jitter is off**: blue-noise-offsetting the first sample and letting TAA resolve
> it is the standard way to buy back march steps and is the documented follow-up, but it is a
> per-frame-index dependence, which is exactly what a byte-identical determinism gate forbids.
>
> **Byte-stability of the off path — measured, not asserted.** Clouds are **disabled by
> default**, the bake and raymarch nodes dispatch nothing at all, and every lit shader's
> cloud-shadow multiply sits inside a guarded branch beside the CSM's. Verified the P17.2 way:
> `INF_BLESS_GOLDENS=1` was run over the whole suite and `git status` reported **zero** changes
> to the 29 pre-P17.3 golden PNGs — byte-identical, not within-tolerance. That covers the three
> new `SHADER_TABLE` entries, the six new `AtmosphereData` vec4s, the new `EnvBinding` binding
> and the recomposed lit shaders. `cloud_shadows_darken_lit_geometry` pins the finer-grained
> version: `shadow_strength = 0` is byte-identical *on the ground band* to a scene with no
> clouds at all, while the sky above still has clouds in it.
>
> **Goldens added (4); `editor_default` re-blessed ONCE.** `clouds_overcast` (solid low
> stratus — the case single scattering renders as soot, so the assertion is on absolute
> luminance and on the collapse of the sky's blue excess), `clouds_scattered` (the *component
> default* coverage, asserting both that the luma spread is far above a clear sky's smooth
> gradient and that real gaps survive — "scattered" is a claim with two ends), `clouds_dusk`
> (warmer than the same clouds at noon, which is the single assertion that would catch clouds
> lit by a hard-coded white sun), `clouds_night` (stars still visible through the gaps, asserted
> by *removing* the starfield and watching the peak drop — a contrast-against-the-mean test
> would also be satisfied by a bright cloud edge). **`editor_default` is the one re-bless, and
> the reason is one boolean**: `demo::build` sets `clouds_enabled = true` while the *component*
> default stays false. Those two must disagree in exactly that direction — a `true` component
> default would silently grow clouds on every existing v12 level the next time it loaded, which
> is the one thing the frozen-record scheme cannot undo — and `the_default_scene_opts_into_
> clouds_while_the_component_does_not` asserts both halves plus that nothing *else* in the block
> was privately tuned. The golden itself asserts the clouds are actually visible in it (12.2 %
> of the upper frame), so the re-bless is justified by a measurement rather than by a comment.
>
> **The coverage slider was calibrated against what it does, not what it says.** As first
> written, 0.30 realised 13 % sky cover and 0.45 realised 97 % — nine tenths of the slider did
> nothing and a tenth did everything, because two octaves of interpolated hash pile up around
> 0.5 and the authored bias slid that narrow field across the density threshold in one go.
> Stretching the weather field to fill `[0, 1]` and recalibrating the bias slope/offset against
> *realised* cover spreads the transition over the range an author actually drags (0.2 → clear,
> 0.35 → 60 %, 0.55 → solid). The component default moved to **0.35** as a result, and World
> Settings shows the aviation word for what the number will look like (clear / few / scattered /
> broken / overcast) rather than pretending the slider is an area fraction.
>
> **Tiers + cost.** `CloudQuality` derives from `AtmosphereQuality` — one knob, not two, because
> a machine that can afford a 256×64 transmittance LUT can afford a 128³ cloud volume and
> letting them disagree would only produce combinations nobody tests. High/Medium/Low are 128³ /
> 96³ / 64³ shape, 32³ / 24³ / 16³ detail, 512² / 384² / 256² shadow, 96 / 64 / 32 primary march
> steps, 6 / 5 / 4 sun steps and 16 / 12 / 8 shadow-bake steps. The **Low-tier cheat is
> documented and deterministic**: it skips the erosion volume entirely, losing the wisps but
> staying a pure function of the same inputs. (A billboard fallback would not — it needs a
> screen-space fade, and screen-space is where determinism goes to die.) The march itself is
> adaptive — long strides through empty air, rewind and refine on contact — so the step count is
> a *ceiling*, not a per-pixel cost. Measured at 1280×720 with the default coverage, GPU-fenced:
> **+0.09 ms Low, +0.20 ms Medium, +0.29 ms High** over the same frame without clouds. The
> always-on VRAM is ~9.6 MB at High (8.4 shape + 0.13 detail + 2 shadow, allocated
> unconditionally like `ShadowResources`' 48 MB cascade array); the bake runs once per seed.
>
> **Honest scope:** cloud shadows reach the **direct** lighting of every lit pass, not the GI
> probes — P18.4 owns GI↔sky coupling and the probes still read the gradient constants, so a
> heavy overcast dims the sun on the ground but not the bounce. No weather presets, no
> precipitation, no lightning (P17.4). Aerial perspective on clouds reuses P17.2's homogeneous
> v1 rather than a froxel volume, and height fog is deliberately *not* applied to them (fog is a
> ground-level authored layer and a cloud four kilometres up is above it by construction). The
> cloud layer does not self-shadow onto the terrain's *own* CSM cascades — it multiplies the sun
> after them, which is right for a soft occluder and would be wrong for a hard one. Clouds are
> flat-slab: no orographic lift over mountains, no cloud-height variation with terrain. The
> march clamp reads MSAA sample 0, so a pixel partially covered by geometry *inside* the deck
> takes its march length from one sample (see the occlusion paragraph). The visual pass — that
> the four new images and the new default level actually look good — is human-verified, as every
> GPU path here is.
>
> **Schema v13 — the same bump for the same reason, one phase later.** `.inf_lvl` payloads are
> bincode, which reads a fixed field count positionally, so growing `SkyAtmosphere` by 14 fields
> makes a v12 record **stop short** of what the decoder expects and read straight on into the
> next entity's bytes. Silently. `#[serde(default)]` rescues self-describing formats and this is
> not one. So the v12 shape freezes as `SkyAtmosphereV12` inside `EntityRecordV12` /
> `SceneFileV12` in **both** codecs — the v8→v9 `TerrainV8` and v11→v12 precedents verbatim, a
> component's layout changing while no entity slot moves — and `into_current` lifts a v12 level
> with clouds **disabled**, which is exactly what a v12 level meant.
> `v13_clouds_are_wider_on_the_wire_than_v12` pins the delta at exactly **62 bytes**, priced
> field by field rather than as one number: 1 bool + 11 × f32 (44) + 16 for the `Color` +
> `varint_len(SkyAtmosphere::default().cloud_seed)`. That last term is a named computation and
> not a literal for a reason — the workspace's `bincode::config::standard()` is **varint**, so
> the seed's cost is a function of the *default seed's value* rather than of its type, and a
> future change to that default would otherwise fail saying "the cloud block grew", which would
> be a lie. The default is asserted to be 0 beside it, so the failure names the field that
> actually moved. That is the
> mechanical proof no serde attribute could have covered it. Every sample and template changed
> by **one byte, at offset 0** (`12` → `13`) — verified programmatically, every file's length
> unchanged and offset 0 the only differing byte — because no committed content carries a
> `SkyAtmosphere`, so the 62-byte block never appears in any of them.
>
> One thing the editor's ladder needed that inf-scene's did not: `EntityRecordV12` carries the
> *frozen* atmosphere, so `EntityRecordV11::into_v12` cannot reach the live component and
> `SkyAtmosphereV11::into_current` became `::into_v12`, filling the physical block from **this
> ladder's own** `v12_*` literals rather than from `SkyAtmosphere::default()`. That is
> doctrinally right for a frozen record — a frozen record must never reach into Ring 0, or it
> stops being frozen — and it is byte-identical to inf-scene's one-hop lift *only while* the
> two agree about the v12 defaults, which is precisely what
> `cloud_defaults_are_the_documented_ones` asserts over all 22 fields rather than leaving
> implicit.

- **P17.1 Sun & time of day** — 1. kill the `SUN_DIR` const: a `SkyAtmosphere` + `TimeOfDay`
  world-component set (components + registry + schema records) projected in both scene
  builders; 2. deterministic solar math (date/latitude → sun and moon position); 3. TOD
  animatable from Blueprints and the sequencer; 4. World Settings rows through the existing
  settings commands.
- **P17.2 Physical atmosphere** — 1. Hillaire-style transmittance / sky-view LUT compute passes
  replacing the gradient (new shader entries validate through the SHADER_TABLE naga gate);
  2. aerial perspective + height fog in the lit passes, extending the generation-keyed
  `EnvBinding` cache invariant; 3. stars and moon at night.
- **P17.3 Volumetric clouds** — 1. raymarched coverage/type-driven clouds on hand-rolled
  deterministic noise; 2. wind drift; 3. sun occlusion feeding the shadow and GI inputs.
- **P17.4 Weather states** — 1. coverage/precipitation/wind parameter blocks with blendable
  presets, Blueprint-drivable; 2. minimal precipitation VFX v1 (a full VFX system stays future
  scope); 3. accumulation hooks consumed by P22 deformation (snowfall).

Schema **v11**: the sky-authority pair — a `TimeOfDay` clock (UTC seconds, day of year,
latitude/longitude, rate) and a `SkyAtmosphere` block (sun/moon colour + intensity, the
three gradient colours, night darkening) as entity slots in both codecs (P17.1). The file
settings are untouched, so only the entity record froze.

Schema **v12**: `SkyAtmosphere` grows the physical-atmosphere block (P17.2) — physical-sky
flag, sky intensity, turbidity, Mie anisotropy, sun/moon disc diameters, star intensity,
gradient-tint strength, aerial-perspective strength, and the SI-metre height-fog quartet
(density, falloff, height, colour). Bincode is positional, so growing a component **is** a
wire-format change: the v11 shape freezes as `SkyAtmosphereV11` inside `EntityRecordV11` /
`SceneFileV11` in both codecs, exactly the v8→v9 `TerrainV8` precedent (a component's layout
changed; no entity slot moved). `into_current` lifts a v11 level with the new block at its
defaults — a gradient sky and no fog, which is what a v11 level meant.

Schema **v13**: `SkyAtmosphere` grows the volumetric-cloud block (P17.3) — enable flag,
coverage, cloud type, the SI-metre layer bottom/top, extinction (m⁻¹), erosion detail, field
seed, the m/s wind pair, forward phase `g`, ground-shadow strength, ambient multiplier, and the
droplet colour. Same mechanism as v12 for the same reason (bincode is positional, so growing a
component **is** a wire-format change): `SkyAtmosphereV12` freezes inside `EntityRecordV12` /
`SceneFileV12` in both codecs, and `into_current` lifts a v12 level with **clouds disabled** —
what a v12 level meant. The delta is exactly **62 bytes** (1 bool + 11 f32 + a 4×f32 `Color` +
**1** for the `u32` seed under `bincode`'s varint `standard()` config). The clock stays
untouched: the wind's drift is *derived* from `TimeOfDay`, not authored, so nothing about it
crosses the wire.

Schema **v14**: `SkyAtmosphere` grows the **weather block** (P17.4) — the enable flag, the
target `WeatherPreset`, the authored blend length and the in-flight remainder, and the seven
live blendable parameters (coverage, cloud type, the m/s wind pair, fog density in m⁻¹,
precipitation intensity, snowiness). Same mechanism as v12 and v13 for the same reason:
`SkyAtmosphereV13` freezes inside `EntityRecordV13` / `SceneFileV13` in both codecs, and
`into_current` lifts a v13 level with **weather disabled** — leaving the authored cloud and fog
fields driving the sky exactly as they did, which is what a v13 level meant. The delta is
exactly **38 bytes** (1 bool + `varint_len` of the default preset's variant index + 9 × f32);
the preset's cost is priced as a varint over its *index* rather than as a fixed width, because
a fieldless serde enum under bincode's varint `standard()` config costs what its position
costs. Third bump in one phase, deliberately: each of v12/v13/v14 shipped before the next
block's design existed, and retro-fitting any of them would mean re-blessing bytes that are
already committed and already load.

### Phase 18 — Lumen-class GI & virtualized-geometry completion

**Goal:** the flagship rendering wave — finish Nanite, finish Lumen. **Done when:** a
vgeom-heavy scene streams meshlets from an mmap pack under two-pass occlusion, the editor
viewport shows real imported meshes, GI responds to TOD/sky and has a specular term, and
goldens + parity + budget ratchets all hold.

Starting point: HZB is a single-pass v1, off by default; whole vmeshes upload at once with no
eviction; the editor viewport still draws primitive placeholders instead of `MeshRef.asset`;
GI revoxelizes every frame, caps at 256 instances, sees only rigid meshes, and has no specular.

> **STATUS: Phase 18 COMPLETE** (2026-08-02) — **local gates green; CI pending push.**
> (Written with the commit rather than after the CI run, like Phase 16's and Phase
> 17's, and saying so rather than claiming a green it has not seen.) Nanite is
> finished and Lumen is finished, to the engineering scope this repository can hold:
> meshlets occlude meshlets and stream a page at a time out of an mmap'd pack under a
> VRAM budget; the editor viewport draws the same real geometry the shipped player
> does; global illumination sees the whole scene, reads the P17 sky, has a specular
> term and can amortize without giving up determinism; and a hundred thousand
> scattered instances are culled, LOD-banded and faded on the GPU in a fifth of a
> millisecond.
>
> The whole phase is pinned by one composed gate,
> `runtime/inf-player/tests/phase18_gate.rs`: a scene that streams meshlets from a
> cooked pack under a binding budget, with two-pass occlusion on, GI v2 running
> against a live time-of-day clock, and 100k+ GPU-scattered instances fading through
> their impostor band — asserted deterministic across two runs on a composed trace
> (pixels, residency, GI audit, instance-cull counters), identical between a cooked
> pack and a PIE payload, still byte-identical between occlusion on and occlusion off
> at 10.6M source triangles with everything else enabled, and inside `FRAME_BUDGET_MS`
> with the per-system costs printed rather than merely asserted — **3.038 ms of a
> 33 ms budget with everything on**, of which GI is 2.658 ms and the 102 416-instance
> scatter is 0.036 ms. That split is the honest headline of the phase: what is left to
> optimise in the flagship frame is Lumen, not Nanite and not the ground cover.
>
> **P18.1 Two-pass HZB occlusion** — the persisted visible list ping-ponged on the
> GPU and never read back; `early cull → early draw → HZB build → late cull → late
> draw`; the pyramid re-sourced from the **MSAA scene depth** rather than the
> single-sample prepass, which is what made a proof possible at all; on by default at
> High, provably free below it. **P18.2 Meshlet streaming** — `.inf_vmesh` v2 as a
> paged container in the `.inf_terrain` shape, four suballocated pools under one
> shared budget with transactional page reservation, a residency-clamped cut, a lazy
> `VmeshRegistry`, and parse-time validation of every payload that arrives from disk.
> **P18.3 Editor real meshes** — the oldest documented gap in the engine closed: an
> imported glTF placed in a scene streams its meshlet DAG in the *editor*, derived on
> import into a content-hash-cached sidecar, resolved by computing its id; a bound
> `SkeletalMesh` draws its skinned mesh in rest pose; eviction, mirror discipline and
> selection all follow the geometry. **P18.4 GI v2 (Lumen-class)** — the 256-instance
> cap deleted in favour of a prioritized, macro-cell-binned gather; terrain, skinned
> characters and meshlets both occluding and receiving; the ray-miss term reading the
> P17.2 sky-view LUT; emissive injection; an SH-derived ambient specular plus a
> screen-space *hit finder*; cascade blending; quality tiers on the existing clamp-down
> system. **P18.5 GPU-instanced scatter** — below.
>
> **The five decisions worth carrying forward.**
>
> 1. **A per-frame conservativeness proof beats a convergence argument.** Two-pass
>    occlusion is temporal, and the house gates assume a frame is a pure function of
>    (scene, view, settings). The resolution was not "it converges" — it was to prove
>    the HZB test is **purely subtractive** from the eight corners of a bounding
>    sphere's world AABB, a `ceil(log2(span))` mip and a min-corner-anchored gather,
>    so temporal state can only decide *when* a meshlet draws, never *whether* the
>    union covers it. That turned a temporal feature into a byte-equality gate instead
>    of a tolerance, and it is why all 36 goldens survived P18.1 untouched. P18.5
>    inherited the proof wholesale for a second, entirely different consumer, which is
>    the strongest evidence it was the right shape.
>
> 2. **The paging unit is a page, not a LOD level — and the allocator is
>    transactional.** A group that fails to simplify leaves its members as *roots* at
>    whatever level they reached, so roots live at many levels and "evict everything
>    finer than F" would punch a hole. Page 0 is therefore every root from every level
>    and is never evicted; residency is always a prefix, which makes ancestor closure
>    structural rather than checked. The four pools share **one** budget (a fixed
>    per-pool split makes some pool rather than the byte budget binding, and it binds
>    hardest on the coarse pages, which is exactly backwards), and a page's four
>    sections are reserved atomically with rollback — without which an asset whose
>    root page was smaller than the growth overshoot ended with **zero** resident
>    pages and *silently vanished from the frame*.
>
> 3. **An editor id must be content-addressed; a cooked one need not be.** A cooked
>    pack is immutable, so a GUID names one sequence of bytes forever. A content root
>    is not — and both render nodes cache GPU state by that id, so a content change
>    under a stable id is a **stale render**, not a reload. Keying the editor's derived
>    `.inf_vmesh` by its payload hash makes that unrepresentable and deduplicates for
>    free. P18.5 applied the same rule to a payload that is rebuilt on *every*
>    projection, which is the harder case: the hash is folded while packing, so it
>    costs one pass over bytes the projector was writing anyway.
>
> 4. **Voxelize by gathering, and quantize whatever a clock touches.** A scatter
>    voxelizer would race on the voxel word, and a race is nondeterminism; a gather
>    over an unbounded instance list is `O(voxels × instances)`. Keeping the gather and
>    shortening it — nearest-surface-first priority, a per-frame budget, macro-cell CSR
>    bins — lifted the cap *and* made "first hit wins" a choice rather than a race.
>    Separately: the moment a running `TimeOfDay` clock entered the probe-sweep key by
>    raw bits, amortization silently paid a full update's cost for one slice of
>    freshness. `sun_bucket` quantizes it (≈0.50° per bucket, ~2 sim-minutes at
>    `rate = 1`), following P17.2's LUT-radius bucketing — and P18.5's CPU fallback
>    buckets the camera for the same reason.
>
> 5. **Compaction order is a design decision, and the right answer differs per pass.**
>    The meshlet cull appends atomically because its draw order provably cannot reach
>    the image. The scatter cull cannot, because its LOD cross-fade is a dithered
>    discard — so it pays a two-level prefix sum to make the compacted list "the
>    survivors in ascending index" on every adapter. Two passes, two answers, both
>    argued from what the frame actually depends on rather than from a house style.
>
> **Honest remainders — human-verified, not gated.** The visual bar is the one thing
> no test in this repository can hold. That emissive bounce, blended cascades and
> screen-space reflections *look* right; that a meshlet LOD transition is invisible in
> motion; that an impostor cross-fade reads as depth rather than as a screen door;
> that a streamed flythrough never shows a pop — all human-verified on one machine
> (RTX 4070 Ti, Windows/Vulkan), and the subjective "Lumen-class" claim in this
> phase's goal is exactly that: subjective. So is the frame rate: every cost figure
> here is a headless measurement at 320×180 or 640×360 on one GPU, and a real-hardware
> fps pass at shipping resolution across the tier ladder — plus a Tracy capture of the
> composed gate scene — remains outstanding, as does the phase demo recording.
>
> **Deferred, with tracking.** Swept from all **seven** blocks the phase carries —
> P18.1 through P18.5 plus the two fixture trips (portable trig, then the
> `meshopt`-is-not-cross-platform re-import fixture) — so the list is in one place.
> The watertight-builder fix that landed beside them added no block of its own and no
> remainder: it corrected `build.rs`'s seam handling with the default-parameter DAGs
> byte-identical, so nothing downstream of it moved.
> *Rendering* — SSR is a screen-space **hit finder**, not a colour fetch; a
> colour-sourced SSR needs a deferred pass or a reprojected history, and the depth
> prepass it marches covers rigid meshes only. The GI volume still revoxelizes every
> frame at the defaults (a **voxel cache keyed on the volume's snapped origin** is the
> temporal half that remains), occupancy is binary and single-bounce, every primitive
> voxelizes as a box or a sphere, the 40 m volume is near-field only (a cascaded or
> world-space clipmap is the next structural step), and emissive is quantized to RGBA8
> against a shared 16.0 ceiling. **The phase gate found one live defect and this is
> where it was recorded — ~~open~~ FIXED 2026-08-02, after Phase 20, see the note
> below:** `GiAudit::probe_cursor` did not advance across a shipped run, so an
> amortized sweep restarted at probe 0 every frame and the probes past the first
> slice were never re-integrated. The cause was upstream of GI —
> `inf_player::render::project_scene` calls `RenderScene::mark_dirty()`
> *unconditionally* at the end of every projection, so `scene.version` moves whether
> or not any content did, and `GiSweepKey` reset the cursor. `gi::sun_bucket` exists
> to stop exactly this happening via the **sun**; the version had the same shape of
> problem and no equivalent guard. At the shipping default (`probe_budget = 0`, a
> full update) it was unreachable. Scatter does not enter the voxelizer at all, and
> the P18.5 block says why that is a decision rather than a gap.
>
> > **THE FIX (2026-08-02).** Not the projection-level "did anything actually
> > change?" question this block anticipated — that ripples through every
> > version-gated upload in the renderer and is a far larger change than the defect
> > warrants. Instead **`scene.version` left `GiSweepKey` entirely**, because on
> > inspection it never belonged there. A content change does not invalidate the
> > *integration*; it makes some probes stale, and the sweep already bounds staleness
> > by construction — the cursor wraps, so every probe is revisited within
> > `ceil(total / budget)` frames and an unvisited probe lags by at most one sweep.
> > That is the identical guarantee, in the identical words, that the sun bucket's
> > own doc already accepts. Resetting on top of it bought nothing and cost the whole
> > feature wherever content moves every frame — which is every frame of a game that
> > is running, not merely of the shipped player. A reset is still what the
> > *irreconcilable* changes get: probe geometry, GI settings, the volume generation,
> > the sky source and the (bucketed) sun. Gated in `inf-render` by
> > `gi_amortization_survives_the_shipped_players_scene_version_churn` (the cursor
> > sweeps `256 → … → 1792 → 0` under the player's own per-frame `mark_dirty`, with a
> > moving-volume anti-vacuity arm that must still reset), and
> > `gi_amortization_survives_a_running_time_of_day_clock` was strengthened at the
> > same time: it used to pin `scene.version` to isolate the sun and no longer needs
> > to, so it now churns the version too and is a second witness.
> >
> > **The player-side witness is a new arm of the Phase 18 gate**,
> > `the_amortized_sweep_advances_when_only_the_scene_version_churns`: the composed
> > level, the real `project_scene`, the sim advanced one block to the composed run's
> > opening pose and then **never stepped again**, so the clock, the sun and the
> > content are bit-frozen (both the frozen sun and the still-rising version are
> > asserted) and `scene.version` is the only moving input — and the cursor then
> > sweeps and wraps. **Gate (a)'s own cursor stays a printed observation rather than
> > an assertion**, which is correct and not a leftover: that arm advances the clock
> > ten sim-minutes per rendered frame (60 steps of the sample's 600× clock), so the
> > sun moves ≈2.5° per frame and crosses `gi::sun_bucket`'s ≈0.50° bucket every
> > frame — the documented, legitimate reset. Asserting an advancing cursor there
> > would be asserting *against* the sun doctrine, not for the fix.
> >
> > **What is still open is the churn itself**, now stated as its own item rather
> > than as GI's: `project_scene` marks the scene dirty every frame, so *every other*
> > version-gated cache in the shipped player — the mesh instance buffer, the depth
> > prepass packing, shadow casters, skinned uploads, sprite batching, the mask id
> > map, vgeom and scatter — re-uploads or re-packs every frame regardless of whether
> > anything moved. That is a pure throughput cost with no correctness face, and the
> > honest fix is a projection fingerprint (or an ECS-side content version) that both
> > hosts can compute; the editor is already correct here, because `sync_from_doc` is
> > document-version-gated.
> *Virtualized geometry* — streaming wants are per **asset**, driven by its closest
> instance, so a hundred distant instances page in the near one's detail;
> per-instance/per-region residency needs a second remap level. A pool that grows
> re-stages every resident page rather than copying (correct, and worth knowing before
> someone "optimizes" it). `AssetResidency::floor_lod()` returns `max_lod` for both
> "roots only" and "nothing resident", and should be an `Option<u32>`. The vgeom HZB
> still sees only what precedes its node — terrain, skinned meshes and translucency
> cannot occlude meshlets — while P18.5's scatter pyramid, built last of the opaque
> passes, shows what a shared frame-level pyramid would look like; consolidating the
> two is both a saving and a fidelity win. `late_drawn` disocclusion is exercised by
> construction and by camera-cut re-convergence rather than by a dedicated
> moving-camera trace.
> *Editor* — the **web** player still draws a `SkeletalMesh` as a placeholder (the
> `meshopt` C++ build script does not cross-compile to wasm32, so `inf-mesh` is
> `cfg`-gated off it); an optional-`meshopt` feature on `inf-mesh` closes it. No
> selection **outline** on real geometry (the mask pass draws `PrimMesh`
> batches); scatter is likewise absent from the ID pass and reaches picking through the
> analytic fallback — extending the ID pass to vgeom, skinned and scatter is one piece
> of work that closes all of it. `VgeomCookOptions::min_triangles` is still 2048 while
> the editor derives from one triangle, so a sub-2048-triangle imported mesh still
> ships as a placeholder; the cook now raises a per-asset advisory naming the mesh,
> and changing the default itself changes shipped bytes. Derived `.inf_vmesh` assets
> are visible in the Content Drawer (a filter chip is the fix); a first project open
> still derives one mesh at a time in the background; multi-material imported meshes
> render with the entity's single `Material` (`build_vgeom` flattens submeshes, a
> documented v1 limitation of the format). The skinned pass caches GPU geometry by
> `Arc` pointer identity, which is a convention the projectors follow rather than
> something the renderer can enforce — **both** hosts follow it now and both are
> gated, but it is still a convention. The player's `render.rs` still calls
> `detect_tier(..).apply(..)` rather than the one-call `detect_and_clamp` seam the
> editor adopted in P18.3.
> *Scatter* — impostors are shaded albedo discs; a **baked snapshot atlas** is what
> textured scatter will need, and is the natural P19 companion to biome populations.
> Scatter draws built-in **primitives** only: a `MeshRef.asset` scatter, routing
> batches through the meshlet path, is the other P19 prerequisite. One cull radius per
> batch (the primitive's bound × the batch's largest scale) rather than per instance.
> The CPU fallback has no impostors and no per-instance occlusion, and re-packs on a
> bucketed camera lattice. Scatter does not enter the GI voxelizer (a decision, argued
> in the P18.5 block beside the shadow-caster budget it is contrasted with), and only
> its full-mesh band casts shadows.
>
> **The golden ledger for the phase.** Thirty-six goldens entered Phase 18 and
> **forty-one** leave it. Every movement, in one place, because "all N are
> byte-identical" is a claim that only means something if the exceptions are
> enumerated:
>
> | Golden | Batch | What moved, and why |
> |---|---|---|
> | `vgeom_dense.png` | P18 fixture fix | **Re-blessed.** The displaced-grid fixture went from `std` f32 trig to `psin64`/`pcos64` (the P14 LAW), so `meshopt` cooked a different — now *portable* — DAG on every platform. Strict mode passed *without* the re-bless (mean 2.5e-4, max 3.4e-2 against 6e-2 / 3.5e-1); re-blessed anyway so the reference is what today's generator produces. |
> | `vgeom_far.png` | P18 fixture fix | **Re-blessed**, same cause (mean 1.5e-5, max 2.1e-2). |
> | `csm.png` | P18.4 | **Changed.** Cascade blending landed with `cascade_blend` defaulting to 0.1. |
> | `gi_bleed.png` | P18.4 | **Changed.** SH ambient specular replaced the flat `ambient × f0 × 0.5`, plus the voxelizer's priority-ordered upload. |
> | `gi_emissive.png` | P18.4 | **New** (emissive injection). |
> | `gi_specular.png` | P18.4 | **New** (the SH specular term). |
> | `gi_terrain.png` | P18.4 | **New** (terrain in the voxel volume). |
> | `scatter.png` | P18.5 | **New** — the full-mesh band: cull, prefix-sum compaction, vertex-pulled indirect draw, PBR. |
> | `scatter_impostors.png` | P18.5 | **New** — the second indirect draw and the dithered cross-fade. |
>
> Nothing else in the suite moved at any point in the phase: P18.1, P18.2 and P18.3
> each shipped with **all 36 byte-identical, verified strict**, and P18.5's five
> changes to the shader stack (including the dither's 24-bit fold) left all 39
> pre-existing images untouched. Each batch's claim was checked the same way — a full
> `INF_BLESS_GOLDENS=1` sweep followed by `git status`, so "only these files" is an
> observation rather than an intention.
>
> **No schema bump, and that is worth a line.** Phase 17 spent three
> (`v12`/`v13`/`v14`) and §12's execution doctrine allows one per phase; Phase 18
> spends **zero** and leaves the level format at **v14**. It is not luck. Everything
> the phase added is either a *derived artifact* (the paged `.inf_vmesh` v2 container
> and the editor's content-hash-keyed sidecars, neither of which is a level record), a
> *renderer cost knob* (`VgeomSettings::stream`, `GiQuality`, `ScatterSettings` — all
> on `RenderSettings`, which is a property of the machine and has never been
> persisted), or a *reinterpretation of fields that already existed*
> (`PcgVolume::draw_distance`, authored since P10.5 and merely honoured in a new
> place). The one place a bump would have been forced — LOD band thresholds as
> authored content — was deliberately answered by putting the bands on the renderer
> beside `AtmosphereSettings` and letting the existing content knob clamp them down.
> A phase that changes this much of the renderer without touching a byte of committed
> level format is the payoff for the `.inf_vmesh` / `RenderSettings` split, and the
> reason every pre-P18 level still loads unmigrated.

> **P18.1 Two-pass HZB occlusion — COMPLETE** (2026-08-01, local gates green; CI pending push).
> Meshlets now occlude meshlets, and occlusion ships **on by default**. Per frame the vgeom node
> records `early cull → early draw` (last frame's visible set) → `HZB build` → `late cull → late
> draw` (the newly-disoccluded remainder), all on the existing vertex-pulled
> single-`draw_indirect`-per-pass shape. The persisted visible list is a GPU-resident `u32` per
> `(instance, meshlet)` pair, ping-ponged between two buffers — **never read back**; the late
> dispatch publishes next frame's early set as a side effect of the work it already does.
>
> **The determinism problem, and how it is actually solved.** Two-pass occlusion is temporal, and
> the house gates assume a frame is a pure function of (scene, view, settings). The resolution is
> **not** a convergence argument — it is a per-frame proof that occlusion is **purely
> subtractive**, so the temporal state can only ever decide *when* a meshlet draws, never
> *whether* the union covers it:
>
> * the meshlet's screen rect and its maximum reverse-Z depth are bounded from the **8 corners of
>   its bounding sphere's world AABB** (a projective map takes that polytope to the convex hull of
>   its projected corners, and sphere ⊂ AABB), bailing to "visible" if any corner is at or behind
>   the eye plane;
> * the mip is `ceil(log2(span_px))`, so one texel spans at least the whole rect and a 2×2 gather
>   anchored at the rect's **min** corner provably covers it (v1 anchored at the *centre* with an
>   approximate tangent-projected radius — which is not a covering, and is why v1 could not have
>   carried this proof);
> * every clamp rounds the safe way (top mip is 1×1; a wider footprint has a smaller min ⇒ *less*
>   culling), and `cs_down` extends its gather on odd mip dimensions so the floor'd chain never
>   drops a trailing row/column — a coverage hole would over-cull;
> * mip 0 is the **min over the 4 MSAA subsamples**, so a culled meshlet is behind *every*
>   subsample of every pixel it touches.
>
> Therefore `d_max < min_R HZB` ⇒ every fragment fails the `Greater` depth test ⇒ the meshlet
> contributes zero pixels, and `image(occlusion on) == image(occlusion off)` **byte for byte**.
> That is the gate, not a tolerance: `crates/inf-render/tests/vgeom_occlusion.rs` asserts exact
> equality on every frame of the convergence, in both occlusion modes, at two viewport sizes
> (one deliberately odd, 211×97), and `vgeom_gate.rs::gate_b2` asserts it on the 10.6M-triangle
> flagship scene.
>
> **Conservative-when-there-is-nothing-to-inherit.** With no usable state — first frame, a
> `scene.version` bump, an instance/meshlet count change, a frame-target reallocation, or a
> **camera cut** (`is_camera_cut`: >50 m in one frame, >60° snap, or an fov/viewport change) —
> the early set is the *whole* base cut, so that frame's drawn set is exactly the pre-P18.1
> single-pass, occlusion-off set. This is a **cost/quality heuristic and is explicitly not a
> correctness dependency** (the proof above already rules holes out); the test suite pins both
> halves separately so a future threshold change cannot quietly become load-bearing. It also
> means every golden — rendered one frame from cold — takes the conservative branch: **all 36
> goldens are byte-identical with occlusion now on by default, no re-bless**, verified strict.
>
> **The HZB source changed, and that is the load-bearing decision.** v1 seeded from the
> single-sample depth *prepass*. That is a **different rasterization** from the 4× MSAA target
> meshlets actually depth-test against — at a silhouette the prepass pixel centre can be covered
> while an MSAA subsample is not — so nothing could be proven from it. The pyramid now
> min-reduces `targets.depth` (the live MSAA scene depth, already `TEXTURE_BINDING` since P17.3)
> over its subsamples. Consequences: the classic mesh pass still occludes meshlets exactly as
> before (it runs earlier into that same target), the early vgeom draw *adds* to it for **no
> extra depth write and no resolve**, and `needs_depth_prepass` **drops its vgeom clause** — a
> full-res depth-only pass is no longer forced just to enable occlusion, which is a net *saving*
> against v1. Honest limitation: passes after this node (terrain, skinned, translucent) are not
> in the pyramid and so cannot occlude meshlets — that costs culls, never correctness.
>
> **Tiers.** `VgeomSettings::occlusion` and the new `two_pass` both default `true`;
> `RenderTier::apply` clears them on Medium/Low (as does `clamp_mobile`), so **no tier below
> High pays anything** — asserted as a pure check rather than measured. `AdapterCaps` gained
> `max_storage_textures_per_stage` + `supports_vgeom_occlusion()` + `clamp_occlusion()` (a strict
> superset of `supports_vgeom`, so it can only further restrict). The cull compute now binds
> **7 storage buffers** (was 4), still inside the portable 8 the High tier already demanded.
> `two_pass = false` keeps the single-pass v1 shape as a settings-selectable fallback, and
> `cull_visible` — the CPU-parity readback — always runs it with occlusion forced off, so the
> LOD+frustum+cone cut the CPU reference mirrors is untouched by this batch.
>
> **Cost** (RTX 4070 Ti, 640×360, 64 meshlet instances × 206 meshlets, steady state): occlusion
> off **0.212 ms**, single-pass **0.331 ms (+0.119)**, two-pass **0.369 ms (+0.157)**. The delta
> is dominated by the resolution-bound HZB build, not by the second cull dispatch. Measured by
> `frame_budget.rs::vgeom_two_pass_cost`, which — following P17.4's precedent — adds **no new
> ratchet constant** (§8 makes each one a standing obligation) and instead asserts the heaviest
> configuration stays inside the existing `FRAME_BUDGET_MS`. Culling yield: **5.6 %** of the
> base cut proven occluded on the 10.6M ground-camera view, **14.7 %** on the wall fixture.
>
> **What the second pass actually buys, measured.** On the wall fixture — pure vgeom, no classic
> geometry — the single-pass v1 shape proves **zero** meshlets occluded, because its pyramid can
> only ever hold depth written *before* the node runs. Two-pass proves 14.7 % on the identical
> scene. That comparison is asserted, not just printed (`defaults_and_fallback`), so "meshlets
> occlude meshlets" is a gated claim rather than a description.
>
> **New instrument.** `EngineRenderer::set_vgeom_audit(bool)` / `vgeom_audit(&gpu)` expose four
> GPU counters (`base_cut`, `occluded`, `early_drawn`, `late_drawn`) aggregated over the frame's
> assets. **Off by default and free when off** — the shader skips the atomics and no readback
> copy is recorded. It exists so the tests can prove the culling is *real* rather than a no-op
> that trivially satisfies the pixel equality. The HZB's own bind groups are `GenCache`d on
> `(targets.generation, hzb.generation)` — the P17.2 `ResourceKey` discipline, with the pyramid
> as a second resizable resource in the key.
>
> **Honest remainders.** (1) `detect_and_clamp(gpu, settings)` is the new one-call host seam
> (tier clamp **and** occlusion floor, so neither can be applied without the other), but the two
> live hosts — the player's `render.rs` and the editor viewport's `apply_render_settings` — still
> call `detect_tier(...).apply(...)` and are unchanged by this batch (out of its file boundary).
> The practical exposure is nil, because `supports_vgeom()` already demands 8 storage buffers per
> stage and no real adapter clears that while reporting zero storage textures; migrating both
> call sites is a one-line follow-up, not a fix. (2) The HZB sees only what precedes the vgeom
> node — the classic mesh pass and the early vgeom draw. Terrain, skinned meshes and translucency
> draw after it and therefore cannot occlude meshlets; folding them in means either reordering the
> graph or a second pyramid, and is a cost opportunity rather than a correctness gap.
> (3) `late_drawn` is 0 on every static converged frame by construction, so the *disocclusion*
> path (a meshlet re-entering the visible set mid-motion) is exercised by the camera-cut
> re-convergence loop and by construction of the shader, not yet by a dedicated moving-camera
> trace; P18.5's GPU-instanced scatter is the natural place to add one.

> **P18.2 Meshlet streaming — COMPLETE** (2026-08-01, local gates green; CI pending push).
> Nothing uploads a whole vmesh any more. A `.inf_vmesh` reaches the GPU one **page** at a
> time, sliced out of the mmap'd pack, into four suballocated pools under a VRAM budget, with
> the LOD cut clamped to what is resident.
>
> **The format: v2, paged — and the paging unit is not the LOD level.** v1 was
> `inf_asset::encode(&VgeomMesh)`: one bincode stream whose `Vec` fields are varint-packed runs,
> so there is no byte offset for "level 3's meshlets" that does not require decoding everything
> before it. v2 is the `.inf_terrain` shape — 128-byte header, a 96-byte-per-entry directory,
> 16-byte-aligned sections, cooked **uncompressed** so a page is a borrowed sub-slice of the
> mapping. `crates/inf-vgeom/src/asset.rs` owns it; the cook emits the raw image (never
> `inf_asset::encode`, which would shift every section off its boundary), and a **v1 payload
> keeps loading forever** — the magic is sniffed and an old payload is lifted into the paged form
> at open, so a pack cooked before this batch still runs with no second code path downstream.
>
> The load-bearing design decision is that **the paging unit is a page, not a LOD level**. The
> obvious unit is the level, and it is wrong: a group that fails to simplify leaves its members
> as *roots* (`parent_error == +inf`) at whatever level they reached (`build.rs`:
> `if !res.progressed { continue }`), so roots live at many levels. Evicting "everything finer
> than level F" would evict a level-2 root whose path has nothing coarser at all — a hole, the
> exact failure this design exists to make unreachable. So **page 0 is every root, from every
> level, and is never evicted**; page `p >= 1` is the *non-root* meshlets of one level, coarse to
> fine. Residency is always a prefix of that order, which makes ancestor closure structural and
> "every root-to-leaf path always has a resident meshlet" true by construction rather than by
> check.
>
> **Vertices are stored once.** `VgeomMesh::vertices` is one welded buffer and a coarse meshlet's
> vertices are a *subset* of the finer ones', so naive per-page blocks would store a vertex once
> per page that touches it. The image instead **permutes** the vertex buffer into page order
> (`page(v)` = the coarsest page referencing `v`) and gives each page the *increment*, so page `p`
> references exactly `[0, prefix(p))` and a prefix residency holds a complete, contiguous vertex
> prefix. The permutation is internal to the container; `to_mesh()` hands back the same geometry.
>
> **The clamp is one scalar, and it is provably free at the wanted floor.** The cut becomes
> `eff_error <= t < parent_error` with `eff_error = (lod_level <= floor_lod) ? 0 : error`, where
> `floor_lod` is the finest resident page's. `VgeomMesh::select_with_residency` is the CPU twin
> and `vgeom_cull.wgsl` applies the identical rule. The surprising and load-bearing part is that
> at the streamer's *own* want — `ideal_page_count(t)`, the pages whose `max_parent_error` still
> exceeds `t` — the clamped cut is **identical to the unclamped one**: a page past that bound
> holds only meshlets that fail `t < parent_error` anyway, and the error-zeroing cannot add one
> because a floor-level meshlet's `error` *is* its children's `parent_error`, which is past the
> bound. So streaming costs VRAM and never detail the camera asked for — only a *budget* clamp
> shorter than the want actually coarsens anything. That theorem, not "everything happens to fit",
> is why **all 36 goldens are byte-identical, verified strict, with no re-bless**, while the
> streamer declines to page in most of the asset. It is asserted, not assumed
> (`golden.rs::vgeom_cpu_gpu_cut_parity` compares the clamped cut against the unclamped one).
>
> **Determinism.** Streaming is where a renderer usually stops being reproducible, because loads
> land when IO says so. Here the plan is a pure function of `(wants, residency, budget)` at one
> sync point per frame: the want comes from the **same** per-instance screen-error scalar `t` the
> cut uses (never from a GPU readback, which is a frame latent and would make residency depend on
> frame history); grants are auctioned worst-error-first through a total order tie-broken on the
> asset id; fetches are `read_ref` slices of an mmap and the staging runs through
> `parallel_map_ref`, the deterministic in-order pure map — terrain's B2 precedent exactly; and
> `max_loads_per_sync` bounds the batch while leaving every asset holding a prefix. The
> "requested-but-missing" cull feedback survives as an **audit counter only**
> (`VgeomAudit::clamped`, off by default), with no path from it to what gets loaded — stated in
> `stream.rs` so a future edit cannot quietly wire it up.
>
> **Suballocation, and the per-pool share that had to go.** Four pools (vertices / meshlet
> records / micro vertex indices / micro triangle indices), each a first-fit free-list allocator
> that is deterministic by construction (sorted, coalesced free list; growth only ever *appends*,
> so a live block never moves). They **share one budget**: a pool may grow only into the headroom
> the other three have not claimed, so total capacity can never exceed `budget_bytes` and there is
> no second ceiling to drift from it. A page's four sections are reserved **transactionally**
> (`VgeomPools::alloc_page`): the first attempt lets growth overshoot — doubling, so the
> reallocate-and-restage a growth costs is amortized — and if any section then fails, the partial
> reservation is rolled back, every pool's *untaken* tail slack is returned to the budget (growth
> only appends, so the tail is exactly the speculation nothing claimed), and the page is retried
> with growth that takes precisely what each section needs.
>
> **That retry is a fix, not a flourish, and the audit is why it exists.** Without it a first
> growth rounded every section up to 1 024 units and charged the slack to the shared budget, so an
> asset whose root page was smaller than the overshoot ended with **zero** resident pages — and
> `VgeomNode` skips an asset with no residency, so the object *silently vanished from the frame*
> instead of degrading to coarser detail. It was not an edge: the band covers ordinary meshlet
> counts (the regression sweep reproduces it at n = 14). Two tests pin it now —
> `a_page_fits_whenever_the_budget_equals_its_bytes` sweeps 1..=2048 units x five section shapes at
> the allocator in microseconds, and `the_root_page_survives_every_budget_that_can_hold_it` proves
> it survives the streamer, the want rules and the budget auction on a **~1 300-meshlet** fixture
> that straddles the 1 024-unit boundary in both directions (the narrow fixture the rest of the
> module uses never reaches it, which is how the defect got past the first round of tests). The
> guarantee is now exactly statable: **if the budget can hold the page's bytes, the page is
> resident.**
>
> The first design gave each pool a fixed *share* of the budget, and it was wrong in a way worth
> recording. The format suggests a split, but the ratio is not constant across an asset: a coarse
> page carries a handful of meshlets with very few vertices each, so its descriptor share climbs
> past 7 % where a fine page's is ~3 %. Any fixed split therefore makes *some* pool, rather than
> the byte budget, the binding constraint — and it binds hardest on the coarse pages, which is
> exactly backwards, because **page 0 is the always-resident floor that makes "never a hole"
> true**. Measured on the render-side fixture: page 0 cost 7 828 B, of which 6 176 B were
> vertices, and the 60 % vertex share meant any budget under 10 293 B left the asset with **zero**
> resident pages drawing nothing — a silent empty frame, not softer detail. The GPU test agent
> found it by bisection while writing the sweep. Sharing one budget deletes the failure mode
> rather than tuning around it, and
> `the_root_page_survives_every_budget_that_can_hold_it` pins the property with no GPU: if the
> budget can hold the roots, the roots are resident, whatever their mix.
>
> A meshlet id resolves through a per-asset **remap** table both the cull compute and the raster
> read; `NOT_RESIDENT` is how a page that is out disappears from the cut, and the sentinel is
> pinned against both shaders by `shader_constants_match_the_rust_side`.
>
> **Reading a payload is a trust boundary.** The build side validated micro-index ranges for a mesh
> *this process* was about to write, which says nothing about bytes that arrived from disk — and
> `to_mesh` slices with those offsets directly, so a doctored `vertex_offset` panics on a 64-bit
> host and yields a wrong slice on wasm32, both reachable from a shipped pack through
> `classic_vgeom`'s `to_mesh().ok()?`, which cannot catch a panic. Every stored record's ranges are
> now checked against its page's sections **once, at parse** (`validate_records`, `O(meshlets)`),
> and the header's `vertex_count` / `meshlet_count` are bounded from **above** by the payload that
> would have to store them — `u32::MAX` vertices previously passed parse and turned into a
> `Vec::with_capacity` request for ~127 GiB, an abort rather than an error. Same discipline
> `inf_asset::pack` applies to blob offsets. Owned payload backings are `inf_asset::AlignedBytes`
> (now public) rather than `Vec<u8>`, so a section's 16-byte alignment inside the file is an aligned
> *address* on every backing — the P16.1 reasoning, and the difference between a `cast_slice` that
> works and one that panics only in the browser. The cull compute now
> binds **eight** storage buffers — exactly the portable floor the High tier already demands, with
> **no headroom left**: a ninth needs a capability bump or a merge.
>
> **`VmeshRegistry` is lazy.** It used to decode every `.inf_vmesh` in the pack at load; it now
> holds a `VgeomSource` per asset — header and page directory, a few hundred bytes — over one
> shared `Arc<PackReader>`, so a level with a thousand virtualized meshes costs a thousand
> directory parses instead of a thousand full decodes. The loose-file path (dev-dir `--level`,
> and the editor path P18.3 will use) is identical after open. `RenderScene`'s `VgeomAsset`
> carries the source, not a decoded DAG. The one consumer that genuinely still needs the whole
> mesh is the **classic discrete-LOD fallback**, which builds a self-contained index buffer per
> level — it materializes once per asset, never per frame.
>
> **Gates.** `crates/inf-vgeom/tests/streaming.rs` brute-forces never-a-hole over every residency
> floor x a threshold sweep (non-empty, resident-only, watertight, exactly one meshlet per
> root-to-leaf chain), plus full-residency equivalence, monotone coarsening, streamer-vs-CPU-clamp
> agreement, cross-asset pool disjointness, the corrupt-page blocked set, and v1-vs-v2 trace
> equality. `crates/inf-render/tests/vgeom_streaming.rs` extends CPU/GPU cut parity to
> **punched-out** residency and pins the pixel + residency traces of a scripted flythrough.
> `vgeom_gate.rs::gate_b3` runs the 10.6M-triangle flagship pack under a hard budget: the ceiling
> holds, the frame is coarser but covers the same ground, and two fresh renderers agree byte for
> byte on both the image and the residency trace. A **frozen v1 `.inf_vmesh`** is committed
> (`tests/fixtures/v1_dense12.inf_vmesh`, the `pack.rs` fixture standard: provenance recorded, never
> re-blessed from the current writer) so "v1 loads forever" gates yesterday's bytes rather than
> today's round-trip.
>
> **Cost** (RTX 4070 Ti, 320x180, a 115-meshlet / 6-page fixture, budget 34 585 B forcing 3 of 6
> pages resident): the cold frame — where the whole wanted prefix pages in — is **4.79 ms**, and
> the steady state, where the sync point re-derives the same wants and does nothing, is
> **0.498 ms/frame** over 60 frames against a 33 ms budget. Following the P17.4 / P18.1
> precedent this adds **no new ratchet constant** (§8 makes each one a standing obligation) and
> instead asserts the streamed configuration stays inside the existing `FRAME_BUDGET_MS`. The
> VRAM side is gated rather than measured: the budget is a hard bound by construction and
> `gate_b3` asserts it on the flagship pack.
>
> **Honest remainders.** (1) The want is per **asset**, driven by its closest instance, so a scene
> where one instance of a mesh is at arm's length and a hundred are on the horizon pages in the
> near one's detail for all of them. Per-instance or per-region residency is the natural follow-up
> and needs a second remap level. (2) A pool that grows re-stages every resident page rather than
> copying the old buffer, because `queue.write_buffer` is ordered *before* an encoder's commands
> in a submit and a same-frame copy would clobber the uploads it was meant to preserve. Growth is
> O(log budget) times per session, so this is cheap — but it is a re-upload, not a copy, and worth
> knowing before someone "optimizes" it. (3) `AssetResidency::floor_lod()` reports `max_lod` when
> nothing is resident at all, which is the *same* value a roots-only residency reports — only
> `resident_pages == 0` distinguishes the two, and it is now only reachable with a budget smaller
> than a single root page. The node skips such an asset rather than drawing it, and both test
> files say so; making the state unrepresentable (`floor_lod() -> Option<u32>`) is the tidier
> follow-up. (4) The editor viewport still does
> not carry vgeom content at all — that is P18.3, and it inherits this path unchanged.

> **P18.3 Editor real meshes — COMPLETE** (2026-08-01, local gates green; CI pending push).
> The oldest documented gap in the engine is closed. Since P4 an imported glTF/OBJ placed in a
> scene drew a **placeholder cube** in the interactive viewport while the shipped player drew its
> real geometry — P4's own status note called it "the documented Phase 4→7 follow-up". A
> `MeshRef.asset` now streams its meshlet DAG in the editor on the same P18.2 path the player
> uses, and a bound `SkeletalMesh` draws its skinned mesh instead of a slate cube.
>
> **The editor has to ASK for that path, and the first cut of this batch forgot.**
> `VgeomSettings::default()` is `enabled: false`; the player opts in explicitly and the editor
> never did, so every asset it carried would have gone through `ClassicVgeomNode` — the same
> geometry, at discrete LODs, with none of P18.2's streaming, budget or eviction. The failure is
> invisible in a screenshot, which is exactly why it is now gated on both sides
> (`projector_mirror::both_hosts_request_the_meshlet_path`) and unit-tested as a pure decision
> (`requested_render_settings`). Requesting is not forcing: `RenderTier::apply` still drops the
> meshlet path below High and `AdapterCaps::clamp_occlusion` still applies the storage-texture
> floor — the editor now applies **both**, which also closes P18.1's honest remainder (1) for this
> host. On Medium/Low the classic fallback draws the same content, so the tier decides the
> mechanism and never the pixels.
>
> **The structural reason it took this long, and what actually unblocked it.** `RenderScene` has
> exactly **one** door for non-primitive geometry — `VgeomAsset` + `VgeomInstance` — and
> `.inf_vmesh` was *cook-only*, so the editor had nothing to put through it. Two halves therefore
> had to land together: the editor derives its own DAGs (`inf_editor_core::assets::vmesh`), and it
> has a loose-file store to resolve them from (`inf_editor_core::render_assets`). Both live in
> **Ring 1**, not in `inf_viewport::host`, for the P16.3b2 reason — the host is
> `#[cfg(any(windows, target_os = "macos"))]`, so logic placed there is invisible to Linux CI. The
> host is left with call sites; the policy is unit-tested on all three OSes.
>
> **Derivation.** On import (and via a project-open sweep that covers content imported before this
> batch) `build_vgeom` → `build_vgeom_asset` writes the v2 paged image **beside the mesh**, under
> the GUID `inf_vgeom::derived_vmesh_id(mesh)` — so the viewport finds a mesh's DAG by *computing*
> the id, with no side index, exactly as the pack does. It is **content-hash cached** the way
> `ImportCache` and `ThumbnailCache` are: the sidecar's `import` table records the **source mesh's**
> hash (not the payload's — the question is "would rebuilding produce the same bytes?", which is a
> property of the input), so an unchanged mesh never rebuilds and a re-import always does.
> `build_vgeom` is pool-size-invariant and `build_vgeom_asset` is a pure function of the DAG, so two
> derivations are byte-identical — the property the cook already relied on, now relied on twice.
> The image is written **atomically** (temp + rename), the `write_terrain_asset` discipline this path
> already cited for its sidecar but not for its own multi-megabyte payload: the content watcher fires
> on that write, so a plain `fs::write` leaves a wide window in which a reader sees a truncated image
> — and `VgeomSource::from_payload` rejects one, turning a race into a mesh that silently stops
> rendering. Gated the decisive way: a *failed* write must not create or touch the target at all,
> which is exactly what separates rename-over from write-in-place.
> The salt that ties the two ids together **moved to Ring 0** (`inf_vgeom::VMESH_ID_SALT`): it was
> hand-copied in the cook and in the player with a drift test holding them together, and a third
> copy for the editor is one past the point where that is defensible. Both now delegate.
>
> **The one deliberate difference from the player, and why it is not a divergence.** The player
> keys `VgeomAsset::id` by the derived **GUID**; the editor keys it by the derived payload's
> **content hash**. The reason is a real asymmetry: a cooked pack is immutable, so an id names one
> sequence of bytes forever — a content root is not. Both render nodes cache GPU state by that id
> (`ClassicVgeomNode::geom` never evicts; `VgeomStreamer` holds pool blocks staged from the source
> it registered), so a content change under a stable id is a **stale render**, not a reload. A
> content-addressed id makes that unrepresentable: changed bytes are a different asset, the old one
> leaves `wants` and is fully evicted, the new one pages in. It also deduplicates — two mesh assets
> with identical geometry share one upload. `AssetResidency::matches` was strengthened alongside
> (whole page directory, not just its counts), and is documented as the cheap discriminator it is:
> a directory-identical payload still passes, which is precisely why the *id* carries the guarantee.
>
> **Mirror discipline.** `tests/projector_mirror.rs` gained a second gate. `project_sky` is still
> compared character for character; the `MeshRef` branch cannot be (it is inline in two loops with
> different iteration orders and id bookkeeping), so it is pinned **field for field**: every field
> of the `VgeomInstance` both hosts construct, in order, with identical value expressions for all
> but the two documented host-local ones (`asset`, `id`). A second test asserts the surrounding
> rules — resolution through the derived id, per-frame asset dedup, the paged source rather than a
> decoded DAG, the primitive fallback — exist on **both** sides. The failure this catches ("the
> editor forgot to project `emissive`") reads as *the shipped game looks different from the
> preview*, which is found by a player, not by a compiler.
>
> **Skinned geometry is shared, not copied.** `RenderScene::skinned_meshes` holds
> `Arc<SkinnedMeshData>` since this batch, and the skinned pass keys its GPU upload on **pointer
> identity** rather than on `scene.version`. Before that, a host re-projecting on every document
> change paid a full CPU copy *and* a full GPU re-upload of a character's bind-space stream — for an
> editor, on every gizmo tick of an unrelated entity (~2.3 MB each, twice). Palettes are still
> rebuilt and re-uploaded per projection, which is correct: they are the part that actually changes.
> The cache holds the `Arc` alongside its buffers, which is what makes the pointer a sound key — the
> allocation cannot be freed and recycled under a live entry — and entries a sync does not touch are
> dropped, which frees them.
>
> **Skinned meshes are not a mirror, and the block says so.** The shipped player has no
> `SkeletalMesh` branch at all, so there was nothing to keep in sync: this is the first host to
> drive `RenderScene::skinned` from real assets rather than from a golden's hand-built fixture. The
> pose rule is: no skeleton ⇒ keep the placeholder; no `AnimPlayer`, no clip, or an unresolvable
> clip ⇒ **rest pose**; otherwise the clip sampled at the play-head through `inf_anim::sample_clip`.
> Rest-pose-by-default is what makes a freshly dropped character visible instead of
> invisible-until-you-press-play. To be exact about the ladder, since "rest pose" is
> doing two different jobs in that sentence: **no `Skeleton`** keeps the placeholder
> cube (there is no bind pose to draw); a resolved skeleton with **no `AnimPlayer`, no
> clip, or an unresolvable clip** draws the skinned mesh at its *bind* pose, i.e. an
> identity palette; and only a resolved clip is sampled at the play-head. Giving the player the same branch was the matching follow-up, and **P18.5 landed it** — see that block.
>
> **The placeholder stays, and that is not a hedge.** A primitive `MeshRef` (Cube/Sphere/Plane/
> Cylinder/Cone) is legitimate authored content, not a stand-in; an unresolved or dangling asset
> degrades to it rather than vanishing; an unbound `SkeletalMesh` keeps its slate cube down to the
> tint. Every golden scene is primitives, so **all 36 goldens are byte-identical, verified strict,
> no re-bless.**
>
> **Lifecycle.** Re-import invalidates through `assets://changed` → `refresh_asset_index`, which
> drops opened `.inf_vmesh` payloads (a vmesh is opened once and then only sliced, so a rewritten
> payload would otherwise be served from the stale mapping forever) while *keeping* terrain streams
> (their tiles are re-read per page, and re-pointing the root would re-page terrain the user is
> flying over). Deleting a mesh takes its derived artifact with it — no dependency edge, on purpose,
> because an edge would make every mesh undeletable by its own derived form. A level switch releases
> everything (`clear_streams`), and each projection releases what the document no longer references
> (`retain_only` — the P16.4b lesson, in mesh form, tested rather than asserted).
>
> **`ClassicVgeomNode` had no eviction at all, and content-addressed ids made that a leak.** It
> materializes a whole DAG per asset into GPU buffers and nothing ever removed an entry — bounded by
> the pack in a shipped build, unbounded in an editor where a re-import mints a *new* id (~3.6 MB
> per 100k-triangle mesh, per re-import, until device-lost). It now retains to the frame's live set,
> mirroring `VgeomNode`'s `plan.dropped` eviction, and does so **before** its early-out so it also
> releases everything when the meshlet path takes over or the scene stops carrying vgeom — the two
> transitions the early-out otherwise hid. The rule is a free function over plain maps, so it is
> pinned without an adapter.
>
> **Selection had to follow the geometry.** A `MeshRef.asset` is no longer a `MeshInstance`, and
> every selection-driven affordance searched that one list: the gizmo snapshot, focus framing, the
> Local-space basis, the transform write-back. They all read through one `instance_xform` /
> `set_instance_xform` pair now, so **an imported mesh is exactly as manipulable as a cube** by
> construction rather than by remembering to add a third branch. Picking needed the same care: the
> GPU id-buffer pass rasterizes `instances` only, so real geometry would have been *unclickable*.
> The stopgap is the technique the gizmo already uses — analytic picking: on an id-buffer miss,
> ray-test the cursor against each vgeom/skinned instance's world bounding sphere, nearest hit wins,
> ties broken by id. It is a fallback, never a first choice, so nothing about picking a primitive
> changes.
>
> **Gates.** `inf_editor_core::assets::vmesh` (derivation determinism, cache hit, re-import
> invalidation, delete sweep, degenerate-mesh skip, project sweep idempotence);
> `inf_editor_core::render_assets` (resolution, cross-store determinism, cheap dangling miss,
> re-import key change, delete-degrades, level-switch release, skeletal rest pose, `AnimPlayer`-driven
> palette, unresolvable-clip fallback, dot-directory exclusion);
> `tests/editor_real_meshes.rs` (the whole chain over a real glTF import: real geometry projected,
> determinism across two full runs, delete degrades to the entity's **own** primitive kind,
> primitives unaffected); `tests/projector_mirror.rs` (the two mirror gates above);
> `inf-vgeom::stream` (re-imported content under one id resets residency, on a fixture asserted
> count-identical so it cannot go vacuous); `inf-viewport::host` (the analytic pick rule, and the
> pure render-settings request); `inf-render::passes::classic_vgeom` (the eviction rule, including a
> re-import replacing rather than accumulating); `inf-packager` (the sub-threshold advisory's rule
> and wording, plus a cook-level test that it reaches `CookReport::warnings`);
> `inf_editor_core::assets::queue` (the sweep's contention gate and its uncontended behaviour).
> **All 36 goldens re-verified byte-identical with the editor opt-in in place** — they render through
> the golden harness's own `RenderScene`, never through the editor host, so the opt-in cannot reach
> them; the point of re-verifying is that this is *checked* rather than assumed.
>
> **Honest remainders.** (1) **No selection outline on real geometry.** The mask pass draws
> `PrimMesh` batches, so a selected imported mesh gets no highlight where its placeholder used to.
> Covering vgeom/skinned there is a renderer change and belongs with extending the ID pass — the
> two are the same piece of work, and the analytic pick above is explicitly the stopgap for half of
> it. (2) **The cook still declines small meshes — but it now says so.**
> `VgeomCookOptions::min_triangles` is 2048; the editor derives from **one** triangle, because vgeom
> is its only real-geometry path. So a *shipped* build of a scene with a sub-2048-triangle imported
> mesh still draws the placeholder the editor no longer does. That asymmetry is the worst shape a
> defect can have — invisible until the build, and invisible in the build until someone walks up to
> the prop — so the cook now raises a per-asset advisory naming the mesh, its triangle count, the
> threshold and the remedy (the `partition::streamed_actors` precedent: the rule and the wording
> unit-tested, the delivery gated at cook level, and a *derived* mesh asserted to raise nothing so it
> cannot become background noise). Changing the default itself is still a follow-up — it changes
> shipped bytes — and the advisory is what makes leaving it a decision rather than an oversight.
> (3) Derived `.inf_vmesh` assets are registered in the DB and therefore
> visible in the Content Drawer; hiding derived artifacts behind a filter chip is a UI follow-up.
> (5) The project-open sweep is **cancellable and never holds the project across a build**: it plans
> under a short lock, builds holding nothing, and commits through a `try_lock` loop that re-checks
> the cancel flag — the P16.4a shape, because a worker parked on the mutex cannot see a cancellation,
> and dropping the `ImportQueue` (project close/switch) joins that thread. Pinned by a contention
> gate that holds the project for the whole sweep and asserts a build still completes and Cancel
> still lands. What remains is only that a first open of a large project still *does* the work, one
> mesh at a time, in the background. (6) The editor derives DAGs; it does **not** yet
> honour per-submesh material slots on them (`build_vgeom` flattens submeshes, a documented v1
> limitation of the format itself), so a multi-material imported mesh renders with the entity's
> single `Material` — the same behaviour the shipped player has always had.

> **P18 fix — portable vgeom fixture meshes, and the one sanctioned re-bless** (2026-08-01).
> Two macOS-only CI failures in P18.2/P18.3 had a single root cause, and it was never in the code
> under test: the displaced-grid **fixture** was built with f32 `sin`/`cos`. Std trig is not
> bit-portable (the P14 LAW), so `meshopt` was handed different vertices on each platform and
> simplified them into a genuinely different meshlet DAG — 138 340 B of resident pages on
> x86_64-msvc against 138 176 B on aarch64-apple-darwin, meshlet counts apart by several, per-page
> `max_parent_error` moving by percent rather than by ULPs. The visible symptoms were a flythrough
> tuned clear of an error boundary on Windows landing on the far side of it on macOS, and
> `reimported_content_under_one_id_resets_residency`'s anti-vacuity premise — two fixture builds
> agreeing on `(meshlet_count, page_count)` — holding at (41, 5) here and failing there.
>
> **The generator is now one function, not nine copies.** It had been hand-copied into
> `inf-vgeom`'s unit tests, its integration tests, `inf-render`'s three vgeom suites and its
> frame-budget gate, and `inf-player`'s activation gate — and the copies were *not* textually
> identical (two normal conventions, one with the constant folded by hand), which is how a fixture
> quietly stops exercising what its tests claim. It lives in `inf_vgeom::test_support` behind a
> test-only `test-support` feature that the other crates' **dev**-dependencies switch on (a
> dev-dependency on self reaches `inf-vgeom`'s own `tests/`); the shipping crate is unchanged. Its
> displacement runs on `psin64`/`pcos64` in f64 and casts once per vertex, and the grid coordinates
> stay in f32 because divide/subtract/multiply were already IEEE-exact. Every platform now cooks
> these fixtures to byte-identical **vertices**. (This memo originally drew one further inference —
> that identical vertices give identical `(meshlet_count, page_count, page_bytes)`, on which the
> anti-vacuity guard's amplitude pair was re-picked to 0.30 / 0.90. **That inference is false**; see
> the follow-up memo below.)
> `dag.rs`'s local wavy grid and the packager's cook fixture were ported for the same reason, as
> were two synthetic threshold walks.
>
> **Two goldens re-blessed, on purpose, once: `vgeom_dense.png` and `vgeom_far.png`.** They are the
> only goldens that draw this fixture, and the fixture's DAG legitimately changed on *every*
> platform, not just macOS. The move is small — mean 2.5e-4 / max 3.4e-2 for `vgeom_dense`, mean
> 1.5e-5 / max 2.1e-2 for `vgeom_far`, against tolerances of 6e-2 / 3.5e-1 — so strict mode
> actually passed *without* the re-bless; they were re-blessed anyway so the reference is what
> today's generator produces rather than a stale image quietly spending the tolerance budget. **The
> other 34 goldens are byte-identical, verified by decoding the pre-change PNGs against the
> post-change ones (mean = max = 0 for all 34), and all 36 pass `INF_GOLDEN_STRICT=1` afterwards.**
> The frozen v1 fixture `v1_dense12.inf_vmesh` was deliberately **not** touched: it pins committed
> *bytes*, not the generator, and its test only ever reads the committed file — its provenance note
> now records that the generator has since changed and that this is intended.

> **P18 fix, second trip — `meshopt` is not cross-platform, so the re-import fixture stopped
> pretending otherwise** (2026-08-02). The memo above ended one inference too far. Byte-identical
> vertices do **not** imply an identical meshlet DAG: `build_vgeom` runs `meshopt`'s native C++
> clusterizer and simplifier, whose float paths and SIMD differ between arm64 and x86_64.
> `reimported_content_under_one_id_resets_residency` failed on macos-arm64 a second time — the
> re-picked (0.30, 0.90) pair, both (38, 5) on x86_64-msvc, measures (37, 5) and (39, 5) on
> aarch64-apple-darwin, from vertices that were by then *provably* bit-identical between the two.
> The first fix made the **input** portable, which was right and stays; the **builder** is not
> portable and cannot be asked to be. Nor should it: cross-platform DAG identity was never a
> requirement of the format — a `.inf_vmesh` is cook-derived on one machine and shipped, and only
> the cook's own same-platform determinism is promised (and gated, in `dag.rs` and `asset.rs`).
>
> So no amplitude pair can carry that premise, and hunting for a better one is the trap.
> **Variant B is now a mutation of a built DAG, never a second build:** A is built once, and B is a
> clone of A's `VgeomMesh` with vertex heights, cluster bounds and object-space errors scaled 3×.
> The page partition is a pure function of `is_root`/`lod_level` and every section length a pure
> function of the counts, so `(meshlet_count, page_count)` agree **by construction, on every
> platform and under every future `meshopt`** — while the scaled errors move the directory's
> `max_parent_error`/`min_error` and the payload bytes genuinely differ, which is exactly what a
> re-import looks like to the staleness check. Errors are *scaled*, not shifted, so LOD 0 stays at
> error 0, a root's `parent_error` stays `+∞` (page 0 must remain unconditionally wanted) and the
> monotone error chain survives; the test now also asserts the two payloads differ.
> `test_support::displaced_mesh` — the amplitude-varied second build that invited the fragile
> premise — is **deleted**, and the module header carries the standing rule: **no test may assert a
> count, a page ladder or a byte total that was measured on somebody's machine.** Derive it from the
> build at runtime, or, where two structurally identical DAGs are genuinely needed, build once and
> mutate a clone. A workspace sweep found no other test resting on cross-build meshopt count
> equality — the counts quoted in `inf-render`'s vgeom suites are already declared illustration and
> their assertions read the live page directory. Gates: `inf-vgeom` (42 unit + 11 dag + 7
> streaming), `inf-render` `vgeom_streaming` + `vgeom_occlusion`, and `INF_GOLDEN_STRICT=1` all
> green with **no golden re-bless** — the mutation touches a test fixture only, and nothing the
> goldens draw.

> **P18.4 GI v2 (Lumen-class) — COMPLETE** (2026-08-01, local gates green; CI pending push).
> The probe GI stops being a rigid-only, gradient-lit, 256-instance demo. Terrain, skinned
> characters and meshlet geometry now occlude and bounce; the ray-miss term reads the P17.2
> sky-view LUT; emissive surfaces inject radiance; a specular term lands; probe updates can
> amortize without giving up determinism; the instance cap is gone; and cascades blend.
>
> **What the cap actually was.** `MAX_GI_INSTANCES = 256` did not "limit quality" — it took
> `scene.instances.iter().take(256)` and threw the rest away in **scene order**, so which
> geometry lit a room was a function of the outliner. It existed because the voxelizer is a
> *gather* (one thread per voxel, first hit wins) and a gather over an unbounded list is
> `O(voxels × instances)`; a scatter would be `O(instances × their voxels)` but would race on the
> voxel word, and a race is nondeterminism. The fix keeps the gather and shortens it:
> primitives are ordered nearest-**surface**-first (`priority_order`, `f32::total_cmp` with the
> source index as tie-break, so a degenerate transform still yields *an* order), clipped to a
> per-frame budget, and binned into 8³-voxel macro cells as CSR offsets+items
> (`bin_macro_cells`) so each voxel walks only the primitives that can reach it. Cell lists are
> ascending in priority, which is what makes "first hit wins" a *choice* rather than a race. The
> overflow is now **reported**, not swallowed: `EngineRenderer::gi_audit()` publishes candidates
> / voxelized / dropped / cell entries / terrain columns / probes updated, free and always on
> like the P18.2 streaming report. Measured: 600 lattice boxes voxelize with zero drops, and a
> deliberately tight budget of 100 reports 500 dropped rather than pretending.
>
> **Coverage, and the fidelity/cost argument for each shape.** Rigid instances stay oriented
> boxes. **Skinned** instances become *per-joint* boxes — the bind-space AABB of each joint's
> dominant vertices (weight ≥ 0.35; a lower threshold inflates every joint toward its neighbours
> until a character voxelizes as one slab), cached per mesh by the `Arc` pointer identity P18.3
> introduced, then carried by the live palette. So a character costs one AABB pass *ever* and a
> handful of matrix multiplies per frame, and it bends. **vgeom** instances become the per-meshlet
> spheres of the **root page** — the coarsest cut, which P18.2's residency floor guarantees is
> always resident. Reading the *live* resident cut would have been higher fidelity and was
> rejected deliberately: it would make GI a function of what the streamer happened to have paged
> in, i.e. of frame history, which is the opposite of what a determinism gate can hold. At the
> volume's 0.6 m voxels the coarse cluster spheres are at or below the grid's own resolution
> anyway. **Terrain** is not an instance at all: it arrives as one height + splat-blended albedo
> per voxel *column*, so it costs `O(1)` per voxel however many tiles are resident.
>
> **Residency interplay — and the same trap, caught a second time.** The first cut of this batch
> sampled *the finest resident tile*, which reads well and is wrong for exactly the reason the
> vgeom decision one paragraph up avoids: `RenderTerrain::tiles` is the streamer's
> **camera-driven** working set, so GI occupancy and albedo would have become a function of where
> the camera had *been*. Invisible to CI, too — every golden renders a fully-resident terrain.
> The voxelizer now reads only the projection's **coarsest asset level**
> (`gi::voxelization_tiles`), the terrain analogue of the always-resident root page: small by
> construction (`build_pyramid` stops at `PyramidOptions::min_tiles`), covering the whole terrain,
> and seeded/reseeded into `TerrainStreamer`'s published cut whatever the camera does. It settles
> the *albedo* half for free — coarse pyramid pages are heights-only, so they project the uniform
> default weight and GI cannot see one splat blend near the camera and another far from it.
> **Fidelity tradeoff, stated plainly:** GI voxels are 0.63 m at the default 40 m volume while a
> level-`n` terrain sample is `mps · 2ⁿ`, so on a deep pyramid the coarse lattice is the coarser
> of the two and near-field terrain occupancy is blockier than the drawn surface. That is the
> right way round — a slightly wrong occluder everywhere beats a differently wrong one depending
> on where the player walked — and an inline (non-streamed) terrain has `max_lod() == 0`, so it
> voxelizes at full authored detail with its painted weights and pays nothing. Gated by
> `gi_terrain_voxelization_is_independent_of_residency`, which drives one scene through a
> fully-resident and a punched-out residency and byte-compares the **voxel volume and probe
> buffer** rather than the pixels (the two states legitimately *draw* different detail — the test
> asserts that too, so it cannot pass on a fixture that isn't exercising anything). `GiResources`
> grew `COPY_SRC` + `read_voxels`/`read_sh` for it; mutation-checked by reverting the level filter,
> which fails the gate. A column with no covering tile is a hole, exactly as an unauthored tile has
> always drawn as nothing. vgeom also now *receives* GI, which it never did: it took the
> hemispheric constant even with GI on, a gap since P13.3b. To be exact, since the
> sentence has been read as broader than it is: the gap was in the **meshlet raster
> specifically** (`vgeom_mesh.wgsl` gained the `gi_irradiance` /
> `gi_ambient_specular` branch the rigid path had carried since P13.3b). The classic
> discrete-LOD fallback already shares the rigid shader and so already received —
> which means a machine that dropped *below* High had been getting more GI on the same
> content than one at High.
>
> **The sky term closes the tracked P17 deferral.** `gi_probes.wgsl` is now a *composed* module
> (it joined `SHADER_TABLE`, so the naga gate covers it): the atmosphere medium at
> `@group(0) @binding(3)` and the LUT samplers at 4/5/6, and its ray-miss term is
> `atmos_sample_skyview` — the same LUT the sky pass draws. That required splitting
> `atmosphere_lut.wgsl`: `atmos_apply` reads the `View` uniform and the probe march is a compute
> pass with none, so aerial perspective moved to `atmosphere_apply.wgsl`, included by the four
> composers that *have* a view. A scene with no atmosphere still takes the authored-gradient
> path, byte-identically — which is why `gi_bleed`'s sky term did not move. Time of day
> propagates for free at the default full-probe update, and restarts the sweep when amortizing
> (the sun direction is part of the sweep key).
>
> **Amortization is a schedule, not an approximation.** `ProbeSchedule` is a renderer-side
> cursor, **never a frame index** — deriving the slice from the frame counter would desync two
> renders the moment a host drew a warm-up frame. Three properties are gated on the GPU: two
> *cold* renders of the same content agree; a converged static scene reproduces across runs; and
> a converged amortized frame is **byte-identical to the full-update frame**. The default is
> `probe_budget = 0` (full update), which is what every golden and every determinism gate
> renders with — a full update makes a frame a pure function of the scene with no convergence
> transient to reason about. A moving camera trades probe latency for the saving, which is why
> it is opt-in rather than default. The sweep resets on GI settings, probe geometry, GI
> generation and the (bucketed) sun; camera *motion* is deliberately absent, because the volume
> follows the camera and resetting on that would mean never amortizing at all. **Scene version
> was in that list until 2026-08-02 and is not any more** — it churns every frame in the shipped
> player, which made amortization a no-op there; the sweep's own wrap-around bounds staleness
> after a content change without it. See the P18.4 defect note above for the whole argument.
>
> **The sun enters that key QUANTIZED, and the first cut got it wrong.** With raw `f32::to_bits()`
> a running `TimeOfDay` clock moves the projected direction in the low bits every frame, so the
> sweep would reset every frame and the cursor would never leave its first slice — amortization
> paying a full update's CPU cost for one slice of freshness, precisely where it was meant to pay
> off, and silently. `gi::sun_bucket` quantizes the direction onto a `1/200` component lattice
> (≈ **0.50°** in-bucket, `√3/200 rad`) and the radiance onto a `1/64` one, following the P17.2
> precedent where the sky-view LUT's camera radius is bucketed so a walking camera does not re-bake
> the sky. Sized against the clock, not the shader: the sun sweeps 15°/hour, so at `rate = 1` a
> bucket lasts ≈ **2 sim-minutes** — an 8-frame sweep always completes inside one. **Bounded
> staleness:** within a bucket the probes are integrated against a sun up to 0.50° stale, and an
> unvisited probe lags by at most one further sweep; a bucket crossing restarts the sweep, so the
> lag never accumulates. At the default full update none of it is reachable. `GiAudit` gained
> `probe_cursor` because the sweep is otherwise unobservable from outside — the bug pins it at
> `probe_budget` forever and no rendered frame would say so — and
> `gi_amortization_survives_a_running_time_of_day_clock` drives a rate-1 sun (cursor sweeps
> `256 → … → 1792 → 0`, a full wrap in 8 frames) against a 2°/frame one (pinned at `256`, i.e. the
> key still resets on real motion — only the resolution changed).
>
> **Specular, and what SSR v1 honestly is.** (a) The ambient specular becomes L1-SH radiance
> reconstructed along the reflection vector, sharpened by `1 − roughness`, times Karis' analytic
> split-sum BRDF. It **reduces to the flat `ambient × f0 × 0.5` it replaced** in the
> rough/uniform limit (pinned in the pure half, where a uniform field can actually be
> constructed), so turning it on adds directionality rather than energy — which is why it can
> default to on. (b) **SSR v1 is a screen-space *hit finder*, not a colour fetch, and the code
> says so.** The renderer is forward: when a lit fragment shades, the scene colour it would want
> to reflect does not exist yet, there is no G-buffer to defer against and no colour history
> bound. What screen space *can* answer is where the reflection ray lands, so a hit re-anchors
> the SH probe fetch at the hit point instead of at the shading point — the reflection then
> follows the geometry causing it rather than smearing the receiver's own probe lobe. Fixed
> 24-step march, no jitter, no history ⇒ deterministic by construction; the penetration test is a
> **ratio** (`1 − ndc.z / scene_z`) rather than a metric thickness, because reverse-infinite-Z
> has no far plane to linearize against and a relative tolerance works at every scale. It is
> **off by default** (it forces the depth prepass on), and with it off the lit shaders run the
> identical instruction stream. Two limitations, both real: the depth prepass covers rigid meshes
> only (P13.3a's own scope), so SSR does not see terrain/skinned/vgeom surfaces; and a
> colour-sourced SSR needs either a deferred pass or a reprojected history — the documented
> follow-up.
>
> **Cascade blending closes the P13 deferral.** `shadow_factor` was refactored so the per-cascade
> bias+PCF exists once (`csm_cascade_pcf`, returning `-1` for "not in this cascade"), and across
> the last `cascade_blend ×` of a cascade's range the receiver additionally samples the next one
> and lerps. `0` restores the hard switch *exactly* — the branch is not taken and the second PCF
> never issues. The continuity property is a pure function (`csm::cascade_blend_weight`), so it
> is unit-tested on every CI leg including the adapter-free ones; the GPU gate proves the effect
> is **localized**: with a 14 m shadow range the differing pixels sit only in the two blend
> bands, with the last cascade (nothing to blend into) and the middle of cascade 0 byte-identical.
>
> **Resizable resources spend the exclusion the P13 comment granted.** `GiQuality` (Low 32³/8×4×8,
> Medium 48³/12×6×12, High 64³/16×8×16 = exactly the pre-P18.4 geometry) makes `GiResources`
> recreatable, so `ResourceKey` is now the **three**-tuple `(targets, atmosphere, gi)` — the case
> `EnvBinding::bind_group`'s doc comment predicted would arrive "if either ever becomes
> resizable". A stale key here is silent: wgpu keeps the old, *larger* SH buffer alive, so a Low
> frame would come back byte-identical to the High one instead of erroring. Both halves are
> gated — pointer identity in the adapter-free `GenCache` test, and a High→Low→Medium→High sweep
> on the GPU. `HzbChain` got a key type of its own rather than borrowing `ResourceKey`: it embeds
> no GI resource, and rebuilding its bind groups on a GI-quality clamp would be a lie about what
> invalidates them.
>
> **Cost** (RTX 4070 Ti, 640×360, 484-cube field, GPU-fenced, over a 0.2–0.4 ms GI-less frame,
> against the 33 ms budget): **Low +0.72 ms, Medium +1.14 ms, High +1.85 ms**; High with
> amortization at 256 probes/frame **+1.29 ms**; High + SSR **+1.74 ms** (SSR is inside the noise
> of High here — the depth prepass is cheap at this resolution and most reflection rays leave the
> screen in a few steps). Below High the tier clamp governs by construction, not by luck:
> `RenderTier::apply` lowers `GiQuality` on Medium and turns GI **off** on Low, as does
> `clamp_mobile`. Always-on VRAM at High is 2 MB of voxels (two words per voxel now — albedo +
> occupancy, then emissive) + 128 KB of SH.
>
> **Golden evolution — two changed, three new, thirty-four untouched.** A full
> `INF_BLESS_GOLDENS=1` sweep reports exactly: `csm.png` (cascade blending, `cascade_blend`
> defaults to 0.1) and `gi_bleed.png` (the SH specular replacing the flat ambient specular, plus
> the voxelizer's priority-ordered upload) modified; `gi_emissive.png`, `gi_specular.png`,
> `gi_terrain.png` added — the suite goes 36 → **39**. Every one of the other 34 is byte-identical,
> which is the off-path discipline made checkable, and is itself gated by
> `gi_v2_off_path_is_byte_identical`, which winds every new knob to a non-default value on a
> GI-off, shadow-off scene and demands the same bytes. That 34 **includes `vgeom_dense.png` and
> `vgeom_far.png`**, which the portable-fixture batch above re-blessed for its own reasons: the two
> evolutions are disjoint, and a full bless on the rebased branch reports exactly this batch's five
> files and nothing else. The two audit fixes moved **no** golden either: `gi_terrain`'s fixture is
> a level-0-only terrain, so `max_lod() == 0` selects exactly the tiles it always did, and the sun
> bucket is unreachable at the default full probe update.
>
> **Honest scope.** The volume still **revoxelizes every frame** at the defaults; probe
> amortization is the opt-in half of the temporal story and a *voxel* cache keyed on the volume's
> snapped origin is the remaining follow-up. Occupancy is binary and single-bounce — no
> multi-bounce feedback, no distance field, no cone tracing. Every primitive is a box or a
> sphere, so a `PrimMesh::Sphere` instance still voxelizes as its bounding box exactly as in v1.
> The 40 m default volume is near-field only; a cascaded or world-space clipmap volume is the
> next structural step. Emissive is quantized to an RGBA8 word with a shared 16.0 ceiling
> (relative, so hue survives; anything brighter clamps). SSR's two limits are above.
> **The residency fix has a residual the auditor named and this batch does not close:**
> `RenderTerrain::max_lod()` is a max over the *resident* set, so a terrain small enough to sit
> entirely inside the finest refine ring publishes no root tile at all, and voxelization detail
> flips between two stable regimes at that camera distance — camera-independent *within* each
> regime, which is strictly better than the per-tile drift it replaced, but not yet one regime;
> the honest fix is a **streamer residency floor pinning the root level**, which lives in
> `inf-terrain` and is outside this batch's file boundary. And the
> P18.3 remainder the auditor flagged is untouched by this batch and still stands: the skinned
> pass caches its GPU geometry by `Arc` pointer identity, so a host that rebuilds a
> `SkinnedMeshData` rather than sharing it re-uploads megabytes per projection — the sharing is a
> convention the projectors follow, not something the renderer can enforce. The visual pass —
> that emissive bounce, reflections and blended cascades actually *look* right — is
> human-verified, as every GPU path here is.

> **P18.5 GPU-instanced scatter + the Phase 18 gate — COMPLETE** (2026-08-02, local
> gates green; CI pending push). PCG and foliage instances stop being CPU-pushed
> `MeshInstance`s. They live in GPU instance buffers, are culled per instance on the
> GPU against the frustum and the P18.1 HZB, and fade through distance bands to
> impostors — the three P10.5 deferrals, landed together because they are one path.
>
> **What died, and the size of it.** Since P10.5 both projectors expanded
> `PcgVolume::evaluated` and `Foliage::instances` into one `MeshInstance` *each* and
> pushed them onto `RenderScene::instances`. A 100k-instance scatter therefore cost
> 100k CPU structs per projection, 100k 176-byte `InstanceRaw`s per pack, and a
> ~17 MB vertex-buffer upload — before the GPU had culled one of them. Both hosts
> carried the same admission in a `warn!`: *">50k instances — instanced-draw perf
> path is a follow-up"*. This is that follow-up; the warning is deleted, and
> `projector_mirror::neither_projector_warns_about_fifty_thousand_instances` keeps it
> deleted.
>
> **The payload is content-keyed AND origin-independent, and the second half is the
> load-bearing one.** `ScatterData::build` packs 48 bytes per instance and folds an
> `xxh3`-128 over the packed bytes *as it packs*; that hash is the renderer's
> `GenCache` key, so identical content is one upload (two foliage entities painted
> from the same stroke; the editor and the player rendering the same level) and
> changed content is a *different* asset rather than a stale one under a reused id —
> P18.3's derived-vmesh-id argument, applied to a payload that is rebuilt on every
> projection. The obvious pack would store render-local positions, and it is wrong:
> a render-local buffer is invalidated by every **floating-origin rebase**, so a
> camera flying across the world would re-upload every instance buffer it can see.
> Offsets are therefore relative to the batch's own **anchor**, which rides in a
> per-frame uniform; the buffer is a pure function of the content and a rebase costs
> 16 bytes. Precision is stated rather than assumed: f32 against the anchor resolves
> to ~6e-5 m over a 1 km batch, and `PcgVolume::extent` defaults to 50 m.
>
> **The compaction is a PREFIX SUM, and that is the batch's real design call.** The
> obvious compaction is `visible[atomicAdd(&count, 1u)] = i` — which is exactly what
> the meshlet cull does. The meshlet path can afford it because P18.1's subtractive
> proof plus an opaque depth test make its draw order provably unable to reach the
> image, so nothing ever compares the list. **Scatter cannot afford it**, for a
> reason specific to this batch: the LOD cross-fade is a *dithered discard*, so an
> instance in the fade band emits a stippled subset of its pixels and two overlapping
> instances at the same depth resolve by draw order at sample granularity. Sorting a
> readback would make the *audit* deterministic and leave the *frame* nondeterministic
> — the wrong half. So the order is fixed by construction: an in-workgroup
> Hillis–Steele scan over two flag lanes, then a one-thread exclusive scan of the
> per-workgroup partials, then a scatter into dense slots, which makes the compacted
> list exactly *"the surviving instances in ascending index"* on every adapter, every
> run. It costs one `u32` per instance and two extra dispatches, and it is entirely
> integer addition — exactly associative, so even the tree scan is bit-reproducible.
> The **audit counters stay atomic**, because a sum is order-independent even when its
> increments are not. Three dispatches per batch, and
> `shader_constants_match_the_rust_side` asserts the source has not grown an
> `atomicAdd` into the visible list — the one regression that would still pass every
> count-based assertion. (That guard was itself too narrow on the first pass: it
> watched two buffer *names*, and the obvious atomic append touches neither. It now
> strips comments from the three entry points' bodies and demands that the **only**
> atomic anywhere in them is on the audit counters.)
>
> **Content addressing dedupes the payload and NOTHING else, and the first cut got
> that wrong.** Keying a batch's whole GPU state by content is the natural reading of
> "content-addressed", and it is a defect: the compacted list, the indirect args and
> the two uniforms are per-frame state for **one draw**, so two batches sharing a
> payload at different anchors wrote the same uniform in sequence and the last anchor
> won — one of the two fields silently vanished, on precisely the duplicated-stroke
> case the design cites as its own justification (reproduced at 615 painted pixels
> against a control's 1223). The upload is now keyed by content and the scratch by
> `(content key, batch pick id)`, a pair that is unique by construction for
> everything the projectors emit: a foliage entity's several batches share a pick id
> but differ in *mesh kind*, which is part of the content key.
> `two_batches_of_the_same_content_at_different_anchors_both_draw` pins it with the
> control that makes it non-vacuous — the two fields must **sum**, so a regression
> that draws one of them twice as densely fails too.
>
> **Vertex-pulled, not classic instancing — and a scatter is the textbook case for
> classic instancing.** Feeding a compacted list to classic instancing needs
> `first_instance` on the indirect args to address a sub-range of a shared buffer, and
> `INDIRECT_FIRST_INSTANCE` is a non-portable wgpu feature — the wall P13.1b already
> hit. Compacting the 48-byte payloads instead of the 4-byte indices would make the
> compaction write 12× the bytes. So the list stays indices and the vertex stage reads
> `visible[instance_index]`, which is the vgeom precedent, one array subscript, and
> portable everywhere. Uniform scale buys a second economy: the inverse-transpose of
> `R·S` normalizes back to `R`, so a scatter instance needs **no normal matrix** —
> which is half the reason the record fits in 48 bytes against `InstanceRaw`'s 176.
>
> **The impostor is a shaded disc, not a baked snapshot, and the choice is a scope
> decision rather than a shortcut.** One camera-facing quad per instance out of the
> second indirect draw, alpha-cut to a circle (a square card gets the silhouette
> visibly wrong at exactly the distances an impostor covers) and shaded with a
> **spherical normal** over that disc, so the terminator runs across it the way it ran
> across the mesh it replaced instead of reading as a flat sticker. Its albedo is the
> instance's own. That IS the mesh's average albedo here — scatter v1 draws untextured
> primitives with one flat colour per instance — so a baked snapshot would reproduce
> the same constant plus a silhouette the disc already approximates within a pixel.
> What a snapshot would buy is exactly nothing until scatter carries *textured*
> meshes, and what it would cost is a bake pass, an octahedral view set, an atlas
> allocator, a new asset kind and committed bytes to bless. Tracked as a remainder,
> for P19.
>
> **The cross-fade has no holes, by complementarity.** In the `fade`-metre band before
> `mesh_end` an instance is in **both** lists; the mesh keeps the pixels where a
> screen-position hash falls below `m = (mesh_end − d)/fade` and the impostor keeps
> exactly the complement. One hash, two complementary tests, so every pixel in the
> overlap is covered exactly once — the transition never thins the silhouette and
> never double-shades it. The hash is a pure integer avalanche of the pixel
> coordinate: **no frame index, no temporal jitter, no instance salt**, because a
> golden renders one frame from cold and a determinism gate renders two, and anything
> remembered between frames would make a fade band a function of history.
> `dither_hash_matches_the_rust_side` pins the avalanche against a Rust twin and
> checks the *function body*, not the file, for temporal inputs — so the header's own
> prose about not jittering can neither satisfy nor trip its own gate.
>
> The hash returns 24 bits, not 32, and the missing eight were a bug: `f32(h) / 2^32`
> at `h = 0xFFFFFFFF` is 0.99999999977, which has no `f32` neighbour below 1.0 and
> rounds to **exactly 1.0** — so every test of the form `h < weight` discarded those
> pixels even at full weight, scattering deterministic, permanently-located holes
> through geometry nowhere near a fade band, and falsifying the "never taken outside a
> band" comment on the discard itself. `h >> 8` maxes at 0.99999994, exactly
> representable and strictly below 1; the gate now checks the saturating input rather
> than sampling, because one pixel in 2³² is not something a sweep finds. Neither
> scatter golden moved (at 320×180 the odds of hitting it are ~1e-5), which is the
> honest reason it survived review.
>
> **Both consumers converge, and a real PIE-vs-shipping divergence dies with them.**
> A PCG volume becomes one batch; a foliage entity becomes one batch per palette
> *primitive kind* actually used, emitted in `PrimMesh::ALL` order so the grouping is
> deterministic. Foliage packs against a **zero** build-anchor while carrying the
> entity translation on `ScatterBatch::anchor`, which is worth a sentence: foliage
> instances are already entity-local, so a zero anchor makes "the packed offsets *are*
> the authored positions" a bit-identity rather than an `x + t − t` round trip — and
> because the anchor is deliberately outside the content key, two identical strokes
> painted a thousand kilometres apart share one GPU upload. PCG positions are world,
> so that batch converts against its entity translation normally. A whole scatter
> *entity* carries one pick id, so a multi-kind foliage stroke's several batches share
> it and a click anywhere in the stroke selects the entity that owns it.
> `PcgVolume::draw_distance` — authored since P10.5 — now rides on the
> batch and is honoured **inside the cull compute**, which is what finally makes the
> two hosts agree about it: the editor used to cull against its own camera eye on the
> CPU while the player ignored the field entirely, so a shipped build drew strictly
> more scatter than its preview. The content knob clamps the tier's band **down**,
> never up (`the_authored_draw_distance_only_pulls_the_band_in`), so no schema change
> was needed to land LOD banding — the cost knobs live on `RenderSettings::scatter`
> beside `AtmosphereSettings`, for the reason that one has no enable flag either:
> whether a level has scatter is a property of the content, and what a host owns is
> how far it is willing to pay to draw it.
>
> **The shipped player learned to draw skeletal characters, which closes a second
> live divergence.** P18.3 gave the *editor viewport* a real `SkeletalMesh`
> projection and left the player with **no branch at all** — `project_scene` never
> touched `RenderScene::skinned_meshes` — so a level with a skeletal character
> previewed correctly in PIE and shipped as nothing. `inf_player::skinned` is the
> player's half: the `VmeshRegistry` shape applied to skeletal assets, reading a
> cooked pack (or a `--level` dev dir) where the editor walks a mutable content root.
>
> Two functions are kept **character for character** identical to the editor's and
> pinned as source text by `projector_mirror.rs`: the **pose rule** (no skeleton, or
> a skeleton with no joints ⇒ keep the placeholder; no `AnimPlayer`, no clip, or an
> unresolvable clip ⇒ the rest pose; otherwise the clip sampled at the play-head,
> honouring `looping`) and the **bind-space rebuild** (submeshes concatenated,
> indices rebased, an unskinned submesh pinned to joint 0 at weight 1). What is
> host-local is only *where the bytes come from*. Notably the P18.3 **content-hash
> vs GUID** asymmetry does **not** apply here, and the reason is worth keeping: that
> one exists because both vgeom nodes cache GPU state by `VgeomAsset::id`, so a
> re-import under a stable id renders stale — the skinned pass caches by the
> **pointer identity** of the `Arc<SkinnedMeshData>` instead, so a re-import that
> produced a new `Arc` is already a new cache entry and an id needs to carry nothing.
> The player shares one `Arc` per mesh across projections, which is the convention
> the P18.3 remainder says the renderer cannot enforce — so it is gated instead.
> `phase18_gate::pie_equals_shipping_on_the_projected_skinned_pose` steps a cooked
> sim and a PIE sim the same number of fixed steps and compares the projected joint
> palettes **bit for bit**, with the anti-vacuity guards that matter: the palette
> must not be all-identity, it must have *moved* between two step counts, and the
> old 4-argument door must still yield exactly the one placeholder cube that shipped
> before — so the fixture is provably exercising the new path.
>
> **One honest platform carve-out.** The bind-space geometry lives in the authoring
> `.inf_mesh`, so the player needs `inf_mesh::MeshAsset` — and `inf-mesh` pulls
> `meshopt`, whose build script compiles C++ through `cc`, which does not
> cross-compile to `wasm32-unknown-unknown` (the same wall `inf-vgeom` already gates
> around). The dependency therefore sits under `cfg(not(target_arch = "wasm32"))` and
> the **browser player keeps drawing a `SkeletalMesh` as its placeholder** —
> unchanged from before this batch, while every native target draws the real thing.
> Hand-copying the asset struct instead was the alternative and is exactly the
> bincode positional-decode desync the schema LAW forbids. Making `meshopt` an
> optional feature of `inf-mesh` is the ~5-line follow-up that removes the carve-out.
>
> **A batch is one object for selection.** It carries one pick id, not one per
> instance — so the editor's `id_to_guid` map holds one row where it held 100k, and a
> click on a scattered rock still selects the owning volume. The GPU id pass rasters
> `instances` only, so scatter reaches picking through P18.3's **analytic** fallback,
> which now has a second consumer; extending the ID pass covers both at once and is
> the tracked follow-up.
>
> **The shadow regression, found and closed rather than documented.** Before this
> batch a scatter's instances *were* `scene.instances`, so they cast cascaded shadows
> for free. Moving them to the GPU path silently deletes every one of those shadows —
> invisible to the compiler, invisible to a unit test, invisible to a pixel comparison
> with no shadow in it, and obvious the moment anyone looks at the ground. The shadow
> node now packs scatter casters beside the rigid ones through the same
> `pack_fallback`, clipped to `shadows.max_distance × 1.5` (a caster outside the last
> cascade can still cast *into* it when the sun is low) and to the same bucketed
> camera lattice, so the caster set stays a pure function of its key.
> `scatter_casts_cascaded_shadows` isolates the claim by toggling *shadows* rather
> than the scatter — the ground is flat and the only thing standing on it is scatter,
> so every ground pixel that darkens is a scatter shadow — and it carries an
> anti-vacuity control that earned its place: the first draft put the sun 20° above
> the horizon and found **1** shadowed pixel, which reads exactly like "scatter casts
> no shadows"; eight rigid cubes in the same fixture found **6**, i.e. the fixture was
> blind, not the code. The control now fails first and says so.
>
> **The caster set escaped every clamp in the renderer, and that is the more
> interesting half.** The first cut *synthesized* a `ScatterSettings` for the shadow
> pack by overwriting `cull_distance_m` with the shadow range — and since that was
> the only field the packer read, the tier's band ceilings, `clamp_scatter` and
> `mesh_distance_m` all became inert for shadows: a Medium-tier machine that had just
> been told to draw 240 m of foliage still rasterized full primitive meshes for 600 m
> of it into three cascades, unbounded in count, and a legal `shadows.max_distance =
> 0` packed **every instance in the world** because the packer read a zero band as
> "unlimited" while `scatter_cull.wgsl` reads it as "cull everything". Every one of
> those is now a `min` against the settings the host already handed the renderer
> (`shadow_caster_settings`), the zero sentinel agrees with the shader, the pack is
> bounded by `MAX_CPU_SCATTER_INSTANCES` degrading **nearest-first** through the
> P18.4 `priority_order` total order, and `ScatterAudit::shadow_casters` publishes
> what actually got in — the number nobody was in a position to ask for. Measured on
> the budget scene: **18 313** casters of 99 856 instances under a 60 m shadow range,
> asserted to be both nonzero and under a quarter of the field.
>
> **Only the full-mesh band casts**, which is a decision rather than an optimisation.
> An impostor is a camera-facing card; from the sun's point of view it is a sliver or
> a disc depending on the angle and never the object's silhouette, so rasterizing one
> into a shadow map is geometrically wrong rather than approximate — and casting the
> *full mesh* for something the camera draws as a disc is exactly the cost the LOD
> band exists to avoid. Pulling `mesh_distance_m` in below the shadow range therefore
> stops the far half of a field casting: a bounded softening the tier explicitly asked
> for, and the reason the CPU fallback's own cost fell from 0.403 ms to 0.095 ms when
> the same clamp reached it.
>
> **That rule was documented one revision before it was implemented**, which is worth
> recording because the shape recurs. Turning impostors off makes `effective_bands`
> report `mesh_end == cull`, so `pack_fallback` reads **`cull_distance_m`** as the
> band — and a `shadow_caster_settings` that clamped `mesh_distance_m` alone produced
> settings that *read* exactly right and packed to the cull distance anyway. Dead
> code that looks like the rule. It only bites past
> `mesh_distance_m / SHADOW_CASTER_MARGIN` (80 m of shadow range at the defaults) and
> every fixture in the suite sat at 60 m or below, so nothing crossed the line: the
> settings-level sweep passed, and 307 instances packed where 128 fit. The clamp now
> lands on `cull_distance_m` too, the band is stated once as
> `min(mesh_distance_m, cull_distance_m, range × margin)` at both doc sites, and
> `the_packed_caster_band_stops_at_the_mesh_distance` asserts it on the **packed
> set** at a 200 m range — mutation-verified, and the only test in the file that
> crosses that boundary at all.
>
> **GI is the other half of that question, and the answer is different on purpose.**
> Scatter does **not** enter the P18.4 voxelizer, and that is not an oversight: the
> voxelizer is a budgeted gather ordered nearest-surface-first, so 100k ground-cover
> instances would have consumed the entire per-frame budget and evicted the actual
> architecture — the failure `MAX_GI_INSTANCES` used to cause silently, arriving by a
> different road. At the volume's 0.63 m voxels a 0.8 m tuft is at the lattice's own
> resolution anyway.
>
> **The asymmetry with shadows is deliberate and is worth stating in one place**,
> because "scatter casts shadows but does not bounce light" reads like an oversight
> until the two budgets are put side by side. A shadow caster is *bounded by the
> shadow range* — 18 313 of 99 856 instances at the defaults, clamped again by the
> tier and again by a hard ceiling — and it produces something a player looks
> straight at. A GI candidate is bounded by a **global** per-frame budget shared with
> the whole scene, ordered nearest-surface-first, and produces a 0.63 m occupancy
> voxel that ground cover is already below. So the same content is affordable and
> visible in one and unaffordable and invisible in the other. Both halves are now
> *bounded and measured* rather than merely asserted, which is what makes this a
> decision instead of a gap.
>
> **This node builds its own HZB, and that is a cost decision, not a correctness
> one.** It runs **last of the opaque passes** — after the rigid mesh pass, both vgeom
> paths, the skinned pass and terrain — and builds a pyramid from the depth all of
> them have already written, which is the richest occluder set in the engine and
> closes, for this consumer, P18.1's honest remainder (2). Sharing the meshlet pyramid
> would have been cheaper and strictly worse: it is built before the late vgeom draw
> and before terrain, and it would couple scatter's culling to
> `VgeomSettings::two_pass` — a setting about a different subsystem. A pyramid is a
> pure function of the depth target at the moment it is built, so a second one is a
> cost and never a correctness question, and it costs *nothing* on a scene with no
> scatter because the node returns before building it. Correctness is inherited rather
> than re-argued: the test is P18.1's, and
> `occlusion_on_is_pixel_identical_to_occlusion_off` gates the inheritance with
> `occluded > 0` asserted separately so the equality cannot be vacuous.
>
> **The test moved into one file so the second copy never existed.**
> `hzb_occlusion.wgsl` now holds the rule and its proof, taking the uniform, the
> dimensions and the pyramid as *parameters* rather than reading globals, so it serves
> two different bind layouts; `vgeom_cull.wgsl`'s `occluded` is a one-line forwarder
> and `vgeom_cull` joined `SHADER_TABLE` as a composed module (so the naga gate covers
> the composition, not just the fragment). Extracting *before* the second copy exists
> is the P18-fixture lesson applied prospectively — that batch paid for nine
> hand-copied generators that were not textually identical.
>
> **The tier decides the mechanism, never whether there is any foliage.**
> `ScatterSettings::gpu` defaults on; `RenderTier::apply` clears it on Medium and Low,
> as does `clamp_mobile`, and `AdapterCaps::clamp_scatter` clears it without compute
> or indirect execution. With it off the same batches draw through the rigid mesh
> pipeline with `InstanceRaw` — no impostors, no per-instance occlusion, same content
> — CPU-culled against a **bucketed** camera position (the P17.2 sky-view-LUT
> precedent: the eye is snapped to an 8 m lattice, the snapped value is part of the
> re-pack key so a walking camera does not re-pack every frame, and the cull radius is
> widened by the cell's own half-diagonal so a snapped eye can only ever keep *more*
> instances than the true one). The scatter capability floor is deliberately **6**
> storage buffers against the meshlet path's 8: ground cover is something a mid-range
> GPU is expected to carry, and it should not be held back by a meshlet floor it has
> nothing to do with. The band clamps are **absolute metres, not scale factors**,
> because the tier clamps have to be idempotent and order-independent —
> `apply_clamps_down_never_up` applies them twice and demands the same settings, and
> repeated multiplication is neither. (The first cut multiplied, and that test caught
> it.)
>
> **The residency floor — the P18.4 auditor residual, now CLOSED.**
> `RenderTerrain::max_lod()` is a max over the *projected* set, and `render_wants`
> replaces a node by its children whenever the camera is inside the refine radius — so
> a terrain small enough to sit entirely inside the finest ring refined every root
> away, published no root tile, and flipped `max_lod` between two regimes at that
> camera distance, taking the GI voxelizer's level choice with it. `TerrainStreamer`
> now carries a `floor` — the catalog's coarsest level, minus blocked keys — which is
> never evicted, always loaded, and charged against the tile ceiling through the same
> `pin_ceiling` the editor's pins already used. **The published cut is unchanged**:
> the floor is *residency*, not the cut, which is what keeps `wants.rs`'s "no parent
> coexists with a child" contract and leaves all eighteen of its tests untouched. It
> is cheap by construction — `build_pyramid` stops at `min_tiles`, so the coarsest
> level is a handful of pages.
>
> That fix needed a second half to be safe, and the second half took two goes.
> `superseded`'s "fine wins" clause tested whether a coarse tile's **immediate**
> children were resident. With the floor pinned, a root can be resident while the cut
> refined *two* levels past it — its immediate children are not resident, the clause
> does not fire, and the root draws over the fine terrain and z-fights. So coverage
> became **transitive**: `covered_tiles` computes, bottom-up in `O(resident × levels)`,
> the set of tiles whose four children are each resident *or themselves covered*. That
> also caught its own test helper: the brute-force hole-free sweep's `footprint_drawn`
> had independently encoded the immediate-children rule and failed at
> `l1 mask 1110, l0 mask 1111` — a footprint fully covered by grandchildren with
> nothing resident at level 1. The helper was wrong; the sweep still brute-forces all
> 256 residency shapes × 4 rings.
>
> **And it was still wrong, because both clauses read the same set.** Coverage was
> computed over the tiles that survived the **frustum cull**, which is right for one
> clause and a reproduced z-fight for the other. Put the camera *inside* the terrain:
> the descendants behind it are culled, the pinned root — whose AABB spans the whole
> grid — is not, so post-cull coverage reads it as uncovered and its decimated surface
> draws straight over the fine pages in front (root drawn beside eight fine patches,
> exactly the artefact the transitive test was added to prevent). The two clauses ask
> different questions and now read different sets:
>
> * **"Is this tile redundant?"** is a property of the **projection**, so it reads the
>   *pre-cull* set. No hole is reachable: a covered tile's ground is drawn by the
>   descendant whose AABB contains it, and if that descendant was culled then that
>   ground is off screen — the parent surviving the cull only means its own AABB
>   clipped the frustum, and that box is one bound over a footprint `4^k` times
>   wider, i.e. mostly **air**. (Not the *union* of its children's boxes: a decimated
>   page's height bounds can be vertically narrower than its children's, since
>   decimation can miss a spike a child keeps. The conclusion needs only the single
>   wide box, and the no-hole argument needs neither — it runs through the
>   descendant's AABB.) Dropping it removes overdraw of empty space.
> * **"Is this tile suppressed by an ancestor?"** is the opposite operation, and an
>   ancestor that was culled draws nothing — suppressing a visible child in its favour
>   is sky through the ground. It keeps the *post-cull* set: the P16.3b1 invariant,
>   unchanged.
>
> With the split in place the property is exact and gated rather than asserted: the
> drawn patch set is **unchanged** for every residency the streamer could already
> produce (`adding_covered_root_tiles_does_not_move_a_single_patch`), it stays
> unchanged when the frustum cuts the root
> (`a_frustum_that_cuts_the_root_still_silences_it`, mutation-verified — reverting
> clause 1 to the post-cull set fails it with the reproduced z-fight), and `max_lod()`
> becomes a stable property of the **asset** instead of of where the camera has been.
>
> **Cost** (RTX 4070 Ti, 640×360, **99 856 instances** over 400 m — one tenth of the
> ROADMAP's 1M target, on one batch), measured over a 0.090 ms scatter-free baseline:
> the full GPU path **+0.147 ms**; the GPU path with the HZB build off **+0.080 ms**,
> i.e. the pyramid is again the resolution-bound majority of the delta and the three
> cull dispatches are ~0.08 ms for 100k instances; the CPU fallback **+0.095 ms**;
> the GPU path **with cascaded shadows +0.381 ms** (18 313 casters admitted by the
> clamps) and the fallback with shadows **+0.306 ms**. The shape matters more than the
> numbers: the GPU path's cost is bounded by the **screen**, the fallback's by the
> **instance count** it is allowed to pack — which is why the fallback looks cheap
> here (the caster clamps cut it to the 120 m mesh band, where the GPU path is drawing
> 400 m of impostors) and why the tier that loses the compute path also loses two
> thirds of its draw distance. Following the P17.4 / P18.1 / P18.2 / P18.4 precedent
> this adds **no new ratchet constant** (§8 makes each one a standing obligation) and
> asserts the heaviest configuration stays inside the existing `FRAME_BUDGET_MS`.
>
> **Goldens: 39 → 41, two added, thirty-nine byte-identical.** `scatter.png` (the
> full-mesh band: cull, prefix-sum compaction, vertex-pulled indirect draw, PBR) and
> `scatter_impostors.png` (the second indirect draw and the dithered cross-fade) are
> new; a full `INF_BLESS_GOLDENS=1` sweep reports **exactly those two files and
> nothing else**, and the whole suite passes `INF_GOLDEN_STRICT=1` afterwards. The
> impostor golden is a separate image rather than an extra assertion on the first
> because an impostor is *different geometry* out of a *different draw*, and a golden
> that never rendered one would leave half the path unpinned.
> `scatter_off_path_is_byte_identical` is the machine-checked half of the claim: every
> scatter knob wound to a non-default value on a scatter-free scene must return the
> same bytes.
>
> **Gates.** `crates/inf-render/tests/scatter.rs` (determinism of the image *and* the
> counters across two fresh renderers; occlusion-on == occlusion-off with `occluded >
> 0` asserted separately; monotone mesh → impostor → culled banding; the authored draw
> distance clamping down and provably not up; impostors-off still covering the ground;
> the CPU fallback drawing the same field and being reproducible; the cast-shadow
> claim with its fixture-blindness control; two same-content batches at different
> anchors both drawing, against a control that demands they *sum*; the inert-off-path
> claim);
> `passes::scatter::tests` (the WGSL constant wire contract, the anti-atomic-append
> guard over the entry-point bodies, the dither avalanche pinned bit-for-bit with no
> GPU *and* proven strictly below 1.0, the band rule, the tier clamps, the shadow
> caster band's three clamps as a swept property over ranges that straddle the
> mesh-band crossover, the packed-band assertion at a range above it, the zero-band
> sentinel, the nearest-first pack ceiling, the fallback's eye lattice);
> `primitives::tests::bounding_radius_bounds_every_vertex` (the cull sphere really
> bounds the geometry, and is *tight* — a loose bound would cull nothing and make the
> whole occlusion suite vacuous); `inf-terrain::stream` (a fully-refined tiny terrain
> still publishing its root level, `max_lod` stable across the regime boundary, a
> pyramid-less store unaffected, the floor charged against the budget);
> `passes::terrain` (transitive coverage silencing a root two levels above the cut,
> the patch-set-unchanged proof, and the frustum-cut case with its own anti-vacuity
> premise); `projector_mirror` (the `ScatterBatch` literal
> field for field, the shared projection rules on both sides, and the deleted 50k
> warning); and `runtime/inf-player/tests/phase18_gate.rs` — **the phase gate**.
>
> **THE PHASE 18 GATE.** `runtime/inf-player/tests/phase18_gate.rs` over a new
> committed sample, `samples/phase18-scatter` — a **12 133-byte** level (12 137 after the
> P19.1 schema bump: the version byte plus one sparse-data-map count per terrain tile) carrying a
> four-tile terrain with ~40 m of relief, a 5×5 grid of **standing** meshlet slabs
> (rotated 90° about X: laid flat, as in the vgeom demo, they occlude nothing, and
> occluders are what make a subtractive proof non-vacuous), a `PcgVolume` evaluating
> to **102 400** instances, a 16-instance painted `Foliage` patch on two palette
> slots, and a `TimeOfDay` + `SkyAtmosphere` authority on a 600× clock. The slabs
> reference `samples/vgeom-demo/Dense.inf_mesh` **by GUID** rather than duplicating a
> megabyte of committed binary, and the gate copies both sample directories into its
> throwaway project. The scatter is split PCG-bulk + small-foliage for a reason the
> README states and a test enforces: a volume's `evaluated` cache is **derived and
> never persisted**, so 102 400 instances cost the level nothing, while
> `Foliage::instances` *is* persisted and 100k of those would be a megabyte of
> committed level.
>
> Eleven tests. **(a)** the composed frame trace — pixels, meshlet residency and floor,
> the GI audit, the instance-cull audit, and the clock and sun as **bits** — reproduces
> byte for byte across two fresh renderers over six frames under a *binding* VRAM
> budget, with every layer asserted to have actually moved (the clock advanced, the sun
> moved, the budget clamped, the meshlet path activated, GI saw the ground, the sweep
> really amortized at 256 of 2048 probes, the cull saw all 102 416 instances); its GI
> probe cursor is *printed*, because this arm's ten-minutes-per-frame clock crosses the
> sun bucket every frame and legitimately restarts the sweep — the advancing-cursor
> claim is the sun-still arm added with the P18.4 amortization fix
> (`the_amortized_sweep_advances_when_only_the_scene_version_churns`: project every
> frame, step the sim never, and the cursor sweeps `256 → … → 1792 → 0`). **(b)**
> the cooked pack and the editor's PIE payload project the same scene, step for step,
> with the clock running; a companion asserts the *scene-level* editor-parity claim
> (vgeom assets and instances, scatter keys/anchors/draw-distance, terrain ids), since
> the field-for-field source mirror is already `projector_mirror.rs`'s job. **(c)** the
> 10.6M-triangle flagship still renders **byte-identically** with two-pass occlusion on
> and off — *with GI, 102 400 scattered instances and a bound streaming budget all
> enabled*. That is a genuinely different claim from `vgeom_gate::gate_b2`, because
> scatter builds its HZB from the depth target the meshlet draw wrote: a
> non-conservative meshlet cull would not merely tint a pixel, it would change the
> pyramid the scatter cull reads and a different 100k instances would draw. Two
> individually conservative optimisations composing into a non-conservative one is
> exactly what a per-feature suite cannot see, so the gate also asserts both audits and
> the streamer's residency are *identical* either way. **(d)** the instance-cull
> counters are real and reproducible — over six frames the peaks are 46 961 frustum,
> 70 284 occluded, 34 946 distance, 17 742 mesh and 16 149 impostor out of 102 416. **(e)**
> the composed frame is inside `FRAME_BUDGET_MS` with per-system costs measured. **(f)**
> the golden inventory is exactly the 41 committed PNGs, **listed by name** so a swap —
> one scene deleted, another added — fails too; nothing is re-blessed here, because a
> comparison that is no longer run cannot fail. Plus the **skinned-pose parity** arm
> described above, and two invariants on the committed sample itself (it stays under
> 48 KB and really carries all four features; the cooked world really evaluates
> 102 400 instances, deterministically, on the terrain rather than a flat fallback).
>
> **Composed cost** (RTX 4070 Ti, 320×180, 25 meshlet instances and **102 416**
> scattered ones, under a 73 482-byte meshlet budget): baseline (meshlets, unbound)
> **0.652 ms**; + streaming under the bound budget **1.107 ms (+0.455)**; + GI v2
> at a 256-probe slice **3.310 ms (+2.658)**; + GPU scatter **0.688 ms (+0.036)**;
> **everything on 3.038 ms (+2.386)** against the 33 ms budget. GI dominates the
> composed frame by nearly two orders of magnitude over the scatter, which is the
> honest headline: what is left to optimise in the flagship frame is Lumen, not
> Nanite and not the ground cover.

- **P18.1 Two-pass HZB occlusion** — 1. persist the last-frame visible list; 2. early draw →
  HZB from its depth → late cull and draw of the remainder; 3. on by default where supported.
- **P18.2 Meshlet streaming** — 1. a residency/page table over the existing coarse-first level
  layout; 2. partial pack decode / range reads on P16.1 mmap; 3. suballocated GPU buffers with
  eviction; 4. cull feedback for missing meshlets; 5. the cut clamps to resident levels — never
  a hole, only softer detail.
- **P18.3 Editor real meshes** — 1. the asset DB / vmesh cache reachable from the viewport
  thread (in-editor `build_vgeom` on import; `.inf_vmesh` stops being cook-only);
  2. `MeshRef.asset` + `SkeletalMesh` rendered in the interactive viewport — the oldest
  documented gap, closed.
- **P18.4 GI v2 (Lumen-class)** — 1. terrain and skinned geometry into voxelization; 2. the
  instance cap lifted via chunked, prioritized voxelization; 3. temporal probe amortization;
  4. a specular term (SH-derived + screen-space reflections v1); 5. sky radiance from P17
  replacing the gradient constants; 6. emissive injection; 7. cascade blending in CSM;
  8. quality tiers on the existing `RenderTier` clamp-down system.
- **P18.5 GPU-instanced scatter** — 1. PCG/foliage instances onto GPU instance buffers;
  2. per-instance frustum + HZB culling; 3. impostor/LOD fade — the P10.5 deferrals, landed
  here so P19 biome populations scale.

### Phase 19 — Biomes, PCG grammar & enterable structures

**Goal:** paint where the world *is*; let a grammar build what stands on it — including
interiors. **Done when:** a multi-biome streamed terrain grows a grammar-built neighbourhood
along a spline road with at least one fully enterable, multi-floor, furnished building per
palette, deterministic across cooks, PIE == shipping.

Starting point: no biome, grammar, or interior concept exists; erosion discards its flow and
sediment state; the `MaskImage` sampler has no graph node; lowering is single-rule even though
`PcgDocument` already models layers × rules × weighted kinds.

> **STATUS: Phase 19 COMPLETE** (2026-08-02) — **local gates green; CI pending push.** (Written
> with the commit rather than after the CI run, like Phases 16–18's, and saying so rather than
> implying a green run that has not happened.)
>
> **The five batches, in one line each.** **P19.1** stopped erosion discarding its story —
> flow / deposition / wear accumulators through the CPU reference, the WGSL mirror, tile
> persistence, both scene codecs and a PNG16 export, with `Σ deposition − Σ wear == Σ Δb` as
> the conservation gate. **P19.2** let authors paint *where the world is* — a `BiomeSet` asset,
> per-sample biome ids sparse on the splat-weight pattern, a paint tool cloning the splat seam.
> **P19.3** made the two halves do something — per-biome `.inf_pcg` dispatch over the region an
> id owns, the deferred `mask.image` node, multi-rule × multi-layer lowering, and data-map
> samplers over P19.1. **P19.4** answered *what sequence of pieces goes along this line* — a
> rule-rewriting grammar DSL, spline and footprint spans, and an exact-fill layout that turns
> them into placed modular-mesh instances on scatter's own instancing path. **P19.5** answered
> *what stands on this lot* — a footprint becomes a floor stack, rooms, walls with real
> openings, stairs and furniture, in seven archetypes, and you can walk into all of them.
>
> **THE SCHEMA ANSWER FOR P19.5: NO — nothing bumps. Scene stays v16, `.inf_pcg` stays v2, no
> MIRROR moved.** The batch needed two new pieces of per-volume state — the solid boxes a
> building's structure is, and the building passes a graph lowers to — and neither reached the
> bytes. `PcgVolume::structures` is `#[serde(skip)] #[reflect(ignore)]` on the exact
> `PcgVolume::evaluated` precedent: derived from the graph and the terrain, both of which every
> loading host already has, and pinned by the component's own round-trip guard (the serialized
> form must not contain the word). `LoweredPcg::buildings` rides beside the document, exactly
> where P19.4 put `grammars`, because `PcgDocument` is the frozen v2 wire and bincode is
> positional. **Only what reaches the bytes can force a bump, and this reaches none.** Phase 19
> spent its one permitted bump on P19.1/P19.2 (v15, then v16) and P19.3/P19.4/P19.5 each spent
> nothing.
>
> **THE DIMENSIONAL SPLIT — the design decision the whole batch turns on.** P19.4's grammar is
> **one-dimensional**: a rule text expands along an arc length. A building is **two-dimensional
> in plan and one-dimensional per wall**, and `crates/inf-pcg/src/building/` is exactly that
> split rather than a second grammar. The 2-D half is a deterministic slice tree
> (`partition.rs`); the 1-D half is P19.4 **verbatim** — a wall *is* a
> `Span`, `assemble.rs` hands each run to `expand_span`, and everything already proved about
> that path (the exact fill, the counter-hashed alternatives, the truncation policy, the
> `atan2`-free orientation) applies because it is the same function. Nothing here
> re-implements a layout. The archetype's wall palette is *authored in the same DSL a user
> types*, parsed by the same parser, so a palette is not a privileged dialect — asserted for
> all seven.
>
> **THE ENTERABILITY INVARIANT, in three parts, all machine-checked.** "Fully enterable" is not
> a screenshot; it is three statements a `BuildingPlan` answers about itself.
> 1. **Rooms connect.** Every room on a floor is reachable from every other through doors.
> 2. **Floors connect to the OUTSIDE.** Exactly one exterior door is placed on the ground
>    floor, and the gate's walk starts there — through interior doors, up stair cores. Without
>    this a building can have a perfectly connected room graph and still be sealed, which is
>    the failure a "rooms are connected" assertion alone would miss. The gate also *severs* the
>    flights and asserts the upper floors become unreachable, so the walk cannot be passing for
>    some other reason.
> 3. **Openings are clear.** No collider — wall module, slab, lintel, parapet, stair tread or
>    piece of furniture — intrudes into a door's void.
>
> **AN OPENING IS THE ABSENCE OF A WALL RUN, NEVER A BOOLEAN CUT.** A wall carrying openings is
> emitted as the intervals *between* them, each expanded independently, plus a lintel above and
> (for a window) a parapet below. There is no subtraction and no "delete the modules that
> overlap the door" pass, so a collider cannot survive inside a doorway by accident: part 3
> above is a check on arithmetic, not a hopeful assertion about a mesh operation.
>
> **AND PART 3 HAS AN ANTI-TAUTOLOGY CONTROL, because the first implementation was vacuous.**
> `opening_is_clear` built its void from a `thickness` of `0.0` — a 2 µm band — and then shrank
> it by the caller's margin on **every** axis. That inverts the thin axis, and
> `Rect2::intersection`'s `max > min` test can never succeed on an inverted rectangle, so *every*
> solid read as clear: the enterability arm passed for a building that was one solid block.
> Three things came out of that:
> * the margin now shrinks the void **along the run and in Y only** — the two axes where a
>   legitimate touch happens (a wall run ends exactly at the jamb; a slab's top face is exactly
>   the sill) — and never across the wall, where anything present is by definition *in the
>   doorway*;
> * the void is the **full wall thickness**, taken from the archetype rather than passed in, so
>   no caller can pass a degenerate one;
> * both the unit suite and the gate now drop a **slab through the whole building and require
>   every opening to report blocked**. An assertion of the form "nothing is here" is worthless
>   if the predicate cannot say *no*, and that control is what keeps it from silently disarming
>   again. Mutation-verified: restoring the old arithmetic fails it.
>
> Making the void real also surfaced two genuine intrusions the vacuous version had hidden, and
> both are fixed at the source rather than papered over. **A stair flight ran wall-to-wall**, so
> a tread stood in the doorway of every room opening onto the stairwell — the flight is now inset
> from the core by the wall thickness, which is also what a real stair has. **Furniture was
> placed by testing its CENTRE**: a bed is a metre deep, so its centre could clear a doorway by
> more than the door is wide while its back stood squarely in the hole. Placement now tests the
> piece's **footprint** against the opening voids themselves — the same rectangles the invariant
> tests — so a rejected piece cannot later fail the assertion. The `DOOR_JAMB` widened from
> 0.15 m to **0.4 m** for the related reason that a module on the *perpendicular* wall straddles
> its own line and reaches `collider.x` along this run; 0.4 is wider than the widest cross
> half-extent any palette declares, and the palette gate now asserts both halves of that
> sentence.
>
> **THE CONNECTIVITY PROOF, and what it does *not* claim.** Every floor is cut the same way —
> `[slab | core strip | slab]`, the strip split into `[stair | corridor]`, the slabs split
> recursively on their **longer** axis (never a hashed axis: hashing it makes slivers) at a
> counter-hashed fraction clamped so both halves clear `min_room`. Then: across any split, both
> sides' leaves tile the *same* interval, and the first leaf on each side starts at the same
> coordinate, so they overlap by at least `min_room`; `min_room ≥ 2 · door_width` is a palette
> invariant, so that overlap always hosts a **full-width** door; by induction the door-capable
> adjacency graph is connected. What it deliberately does **not** claim is that every pair of
> touching rooms shares a door-width wall — two leaves whose boundaries nearly align share a
> sliver, and `connect` simply does not treat a sliver as an edge rather than narrowing a door
> to fit one. A proof that is never machine-checked is a comment, so the finished plan is
> re-checked anyway, per floor, for every archetype.
>
> **THE CORE CARVE HAS THREE REGIMES, because halving a room that will not fit is always
> wrong.** The strip's *position* gives as the plate narrows, never the slabs' size:
> `slack ≥ 2·min_room` is double-loaded (a slab either side); `min_room ≤ slack < 2·min_room` is
> **single-loaded** (the strip hard against a hashed edge, one slab carrying the whole slack);
> below that the strip swallows the leftover and the floor is just `[stair | hall]`. The first
> implementation padded by `min(min_room, slack/2)`, which put *two* sub-minimum slivers on any
> ordinary narrow lot — a 12 × 6 m house gave two 1.90 m slabs against a 2.6 m minimum. Two
> related corrections came with it: the strip is sized by **what its leftover becomes** (a
> corridor may be narrower than a room; a non-corridor archetype's hall may not, and an
> industrial floor was getting a 3.2 m "workshop" against an 8 m minimum), and a **single-storey
> building merges its stair rectangle into the hall** rather than demoting it to an ordinary room
> that `stair_size` may legitimately have sized below the minimum. `min_room` is now swept over
> nine lot shapes × five seeds × seven archetypes, with the reproduced cases named in the test.
> The two structural rooms — stair and corridor — are **exempt by design**, and
> `BuildingArchetype::min_room` says so: a 2.2 m stairwell in a house is right, they are not
> products of a split, and the connectivity proof (which is about split leaves) does not rest on
> them.
>
> **STAIRS ALIGN BY CONSTRUCTION, NOT BY SEARCH.** The obvious implementation — partition each
> floor, then look for the room above that overlaps the stair below — needs a tolerance, fails
> when the partitions disagree, and answers differently per seed. So the core strip is drawn
> from the **building** hash with no floor index folded in, and every storey is partitioned
> around the same rectangle. Alignment is then a property of the arithmetic; the gate asserts
> the stair room's rect is bit-identical on every storey, and the corridor and risers are
> vertically continuous for free. A single-storey building has no core at all — its stair
> rectangle becomes an ordinary room — and a lot too small to host a core is forced to one
> storey, because a floor you cannot reach is not a floor.
>
> **THE COLLIDER DECISION: scattered content had none, and buildings needed them.** The audit
> was unambiguous — `PcgVolume.evaluated` is consumed at three render projectors and by nothing
> in `inf-physics`; a `ScatteredInstance` is not an entity, has no `Guid`, and the bridge's
> world walk keys on exactly that. Scattered content was, categorically, walk-through. Three
> options were weighed and the middle one taken:
> * **Real ECS entities per wall panel** — thousands of derived rows in `.inf_lvl`, a
>   despawn-before-re-evaluate pass in two hosts, and undo meaning something new, all to express
>   data that is already a pure function of the graph and the terrain. Rejected.
> * **`PcgVolume::structures`, derived and unserialized** (taken): one `ScatteredSolid` per
>   solid box, and `PhysicsBridge3D::sync_from_world` walks them into static box colliders under
>   synthetic content-derived GUIDs (`pcg_structure_guid` — a 128-bit mix, not a XOR, so two
>   volumes cannot alias each other's structures). One bridge site serves the editor and the
>   player. Zero schema movement, zero projector-mirror movement.
> * **Doing nothing and calling the buildings "enterable"** — a diagram, not a feature.
>
> **And the decision is gated in the simulation, not only in the data.**
> `crates/inf-physics/tests/pcg_structures_3d.rs` is where "the buildings have colliders" stops
> being unfalsifiable: two derived solids become two **static** rapier colliders at the
> transforms they declare; a dropped solid despawns its body (so re-evaluating a volume cannot
> leak); the synthetic GUIDs are pure and do not alias across volumes; a dynamic body dropped on
> a structure **lands on it** instead of falling through — the pre-P19.5 behaviour, stated as a
> test. Mutation-verified: deleting the one new line in `sync_from_world` fails three of them.
> Without this suite the batch's 13 000 "colliders" were ECS records nothing had shown reaching
> rapier.
>
> **The per-fixed-step cost, found and fixed.** `sync_from_world` runs every fixed step at 60 Hz
> over the whole world, and the first version re-described and re-sorted all ~13 000 immovable
> boxes each time — a regression the *load-time* budget arm could never see. Measured on the
> committed town: **11.62 ms/step against a 16.7 ms 60 Hz frame.** (That 16.7 is the engineering
> claim; the *assertion* is against the imported `FRAME_BUDGET_MS` — a per-step measurement is the
> right class for it, but a 60 fps literal is a hardware claim a shared runner cannot make.)
> The fix is a change stamp
> (`PcgVolume::structures_gen`, bumped by `set_structures`; the bridge retains an unchanged
> volume's colliders rather than rebuilding them) — the same version-stamp shape `SceneDoc` and
> the terrain tiles already use. Measured after: **4.94 ms/step**, 6.7 ms of the budget
> reclaimed. Both figures are printed by the gate arm that measures them, and the retention is
> pinned by a test that requires the *same handles* to survive twenty no-change syncs.
>
> The mechanism is the DSL's one P19.5 addition: `collider hx,hy,hz` on a module declaration,
> **opt-in** (a P19.4 fence that never asked to be solid must not silently start blocking the
> road it follows), rejected at parse time if any half-extent is non-positive, and oriented by
> the **slot yaw only** — half-extents are stated in the slot frame, so rotating them by an
> authored euler would mean the numbers no longer describe what the author typed. Furniture is
> solid too, deliberately: a furnished building whose desks you walk through is worse than one
> whose doors you cannot, and it makes the door-clearance rule *gated* rather than cosmetic.
>
> **The process note: this crossed the batch's stated file boundary, and there is a memo.**
> P19.5 was scoped to `inf-pcg` + `inf-ecs` (read-mostly) + the runtime/editor/sample surface;
> `crates/inf-physics` was not on the list and was modified anyway (~70 lines). §12's doctrine is
> that deviations require a memo rather than silence, so
> **`docs/memos/p19-5-physics-scope-deviation.md`** records what was done, the three options
> weighed, and — the part worth naming — that the option the boundary *literally permitted* was
> to ship the buildings render-only and call them enterable. A scope boundary exists to keep a
> batch reviewable, not to license a feature that does not work. The memo also records the
> per-fixed-step regression the excursion introduced and the change stamp that fixed it, because
> a crate the batch was not scoped for is exactly where a per-frame cost hides.
>
> **The `PcgCollider` carries a quaternion, not an angle, and that is the P14 law again.**
> Recovering a yaw from a span frame would need `atan2` on committed placement data. Instead
> every orientation query is a rational function of the stored components: for a yaw-only
> `(0, s, 0, c)`, `cos θ = 1 − 2s²` and `sin θ = 2sc` — exactly `1` and `0` at the identity,
> which is every wall a v1 building has, so a clearance predicate compares exact bounds.
>
> **The seven palettes are code-shipped constants declaring PRIMITIVES, and that is the honest
> v1.** Office, apartment, industrial (factory/warehouse), house, estate, hotel, shop: each a
> grammar text (module palette + wall rules) plus a table of plan parameters, room weights and
> furniture sets. Every module declares **no mesh GUID**, so a building needs no imported art to
> exist — it is boxes with honest dimensions and honest colliders, which is what "enterable"
> requires and what an engine can ship without a licence question. A palette entry gains
> `mesh <guid>` the day a project has one, with no code change. They are constants rather than
> assets because an archetype has no identity a user edits yet; making it an asset kind would
> buy a `.inf_barch` format, a sidecar, a migration ladder and a Content Drawer glyph before
> anybody has asked to author one. The seam is ready (plain data, one lookup) and the deferral
> is stated rather than discovered. Their *consistency* is gated, not just their existence:
> `min_room ≥ 2 · door_width`, a door fits under a storey, a window band fits, a corridor takes
> a door — the preconditions the connectivity proof rests on.
>
> **The node family is the grammar kit's shape one level up.** `building.archetype` (a
> definition node) → `building.plan` (an expander), joined by one `Named("building")` wire —
> the same *definition + expander* pair `grammar.rules → grammar.expand` already is, and for the
> same reason: one archetype can feed several planners. `building.plan` **outputs a SCATTER**,
> so it joins the existing `scatter.merge` / `layer.layer` chains with no third merge node, no
> third sink input, and a disabled layer disables its buildings for free. **No new `PortType`
> variant** — P19.4's argument, unchanged.
>
> **The lot pin closes P19.4's own remainder.** P19.4 noted that a biome is a painted *region*
> and a grammar needs a *span*, and named "footprints-from-a-region" as the natural closure.
> `building.plan`'s `lot` pin takes the **same SPAN wire** `grammar.footprint` and
> `grammar.spline` produce, and uses its XZ bounding box — so a spline-derived lot works with no
> new concept. Unconnected, it falls back to the node's own size, then to the evaluating
> volume's extent (the P19.4 footprint default). P19.2's `structure_hint` — declared then as
> inert data "because it is what P19.5 will ask a biome for" — is answered too: it names a real
> `ArchetypeId`, gated, and is **advisory**, because a biome owns a region and a building needs
> a lot.
>
> **A building levels its site.** Every Y derives from one `base_y`, sampled once at the
> footprint centre by the evaluation site. Sampling the terrain per module — the scatter idiom,
> and the right one for a fence — would make a building's floors follow the hill, which is not a
> building. `Ground::Terrain` fails **closed**: no ground under the footprint centre means no
> building, not a building at y = 0.
>
> **THE STREAMING FINDING, stated rather than hidden.** The sample's seven lots and its road all
> carry `AlwaysLoaded`, and the reason is a real engine property: **PCG evaluation is a
> load-time pass.** `evaluate_pcg_volumes` runs once over the world the level builder produced;
> cell streaming spawns entities *afterwards* and nothing re-runs evaluation for them, so a
> `PcgVolume` binned into a grid cell would stream in and stay empty — a building lot with no
> building on it. That is the standing P10.6 remainder, restated by P19.4 and unchanged here; a
> batch that hid it by never streaming a volume would have hidden it. The level's *streamed*
> content is therefore twelve street lamps, which bin by position and give the partition arm
> something real to be about, and what the gate asserts about the buildings is the property that
> survives the gap: every instance a lot places lies inside the lot's own footprint, so the day
> evaluation follows streaming the content is already in the right cell. A building larger than
> a cell would still be one entity in one cell; at the sample's deliberately small 128 m cell a
> 44 m lot fits comfortably, and a 400 m megastructure would want a larger cell or the same
> declaration — a content decision the engine should not make.
>
> **THE PHASE 19 GATE** (`runtime/inf-player/tests/phase19_gate.rs`, over the committed
> `samples/phase19-town`: a partitioned 128 m-cell level on a biome-painted four-tile terrain, a
> spline road with a **solid** grammar fence, twelve streamed lamps, and **seven fully furnished
> three-storey buildings, one per archetype**, 15 451 instances and 13 204 colliders).
> **(a) determinism** — two fresh loads agree on the whole trace: the population *and the
> solids* (a wall that renders in the same place and collides in a different one is the specific
> failure this batch could introduce) *and* the partition's cell directory; plus the shipped
> content's own building passes are invariant at 1/2/4/8 workers through the real
> `evaluate_buildings_in` seam. **(b) cooked == uncooked** — pack and dev-dir, bit for bit on
> both halves; and the cook reaches all eight `.inf_pcg` graphs plus the biome set from the level
> alone. **(c) PIE == shipping** — bit for bit on the P19.4 `bits()` standard, extended to a
> `solid_bits()` sibling. **(d) ENTERABILITY** — the headline, for **every** archetype: rooms
> connected per floor, every door and window void clear of the colliders the *pack actually
> shipped*, every floor reachable from outside, the core bit-identical per storey, and the
> severed-stairs control. **(e) partition** — every lamp is in exactly the cell `cell_of` puts
> it in, *exactly* the lamps stream (a lot that lost its marker shows up here), and every lot's
> instances stay inside its footprint. **(f) budget** — the whole town builds in ~8 ms against
> the **load-class** ceiling `inf_player::budget::LOAD_BUDGET_MS` (5 000 ms, the P15.1 startup
> tripwire's number, now shared by both one-shot-load arms rather than copied). It deliberately
> does *not* use the 33 ms composed-frame budget: a load happens once, a frame recurs, and
> holding the first against the second is a hardware claim that duly went red at 34.77 ms on a
> shared `windows-latest` runner while measuring ~8 ms locally. No new ratchet constant.
>
> Beside it, `grammar_span_mirror.rs` grew two needles — `inf_pcg::evaluate_buildings(` and
> `vol.structures =` — because a host that called the first and skipped the second would draw
> the building and leave it walk-through, which is the exact failure "enterable" names.
>
> **Tests.** `inf-pcg::building::palettes` (all seven parse with the *author's* parser; every
> module the assembler and the furniture tables look up by name resolves; the plan parameters
> are self-consistent — the proof's own preconditions); `::building::partition` (a partition
> **tiles** its plate exactly with no overlaps, over every archetype × four plate shapes; no room
> falls below `min_room`; the adjacency graph is connected and **so is the door-capable
> subgraph**; corner contact is not adjacency; walls cover interiors and the plate boundary with
> the perimeter conserved; purity and seed-sensitivity; the core fraction does not depend on a
> floor); `::building::plan` (**every archetype plans a connected, enterable building** over four
> seeds; the core is bit-identical on every storey; each floor tiles the footprint; a lot too
> small for a core is single-storey; a degenerate lot plans nothing rather than panicking;
> openings are disjoint, inside their runs, and under the slab above; the storey override and
> the drawn range); `::building::assemble` (**no solid ever blocks a doorway**, and none blocks a
> window band; floors are slabs and the stairwell is a void; a flight lands **exactly** on the
> floor above; purity + pool-size invariance; unfurnished is a subset of furnished; furniture
> keeps out of door swings; everything stands inside the footprint; a building has substance);
> `::building::pass` (the lot's three-way fallback; a **spline** span becomes a lot; terrain
> datum and fail-closed over a hole; pool-size invariance; the volume seed decorrelates two
> volumes and is a different space from the grammar's; `plans_of` matches what evaluation
> builds); `::grammar::dsl` (a collider is optional and round-trips; a degenerate one is rejected
> **at its value**, with the anchor asserted); `::grammar::expand` (a collider declaration places
> a solid beside its instance, scaled, at the slot yaw and **not** the authored euler; a
> collider-free grammar places none); `::graph` (the kit's wires and the SCATTER output; a
> building lowers to a pass beside an **empty** document; a span on the lot pin; merge and layer
> chains with the toggle; diagnostics anchored and failing closed, including the unknown-palette
> warning driven the only way it can occur — a hand-edited document, since the edit door's
> `sanitize` resets an out-of-set choice; the payload round trip). Frontend: `pcgPinTheme.test.ts`
> gained the `building` wire and category, and now pins all three generator families as visually
> distinct. **42 goldens byte-identical, none re-blessed** — a building is instances and boxes,
> not pixels.
>
> **Human-verified remainders (the honest ledger).** Everything below is *engineering-complete
> and machine-gated*; what a machine cannot check is stated as itself.
> * **The subjective bar: "do the buildings look good?"** They are boxes. Every dimension is
>   honest, every opening is where a door goes, and the plans read as plans — but until the
>   kind→mesh gap closes (below) a viewer sees placeholder cubes, and nobody has walked one of
>   these interiors on a screen and said "yes, that is a hotel". That judgement is a human pass,
>   and it has not happened.
> * **Nobody has actually walked into one in PIE.** The colliders are real, the bridge builds
>   them, and the geometry-vs-collider agreement is gated — but a character controller has not
>   been driven through a doorway and up a flight by a person. CI does not walk.
> * **The visual passes for P19.1–P19.3** — the erosion data-map overlays, the biome paint tool's
>   feel, the biome-overlay view mode — are unchanged human remainders from those batches.
> * **The Phase 19 demo recording.**
>
> **THE DEFERRED LEDGER, swept from all five batches.** Nothing below is a bug; each is a
> decision with a stated reason.
> * **P19.1** — a **thermal channel** as a fourth data map (excluded so the conservation
>   identity stays exact); the **coarse-page data maps read zeros above level 0** (P19.3 landed
>   the decimation rules; the levels themselves are still unpopulated for legacy pyramids); the
>   **bless-guard asymmetry** (`INF_BLESS_FIXTURES` means *any value* in the editor codec and
>   *exactly `"1"`* in `inf-scene` — always use `=1`); **`export_data_map` writes to a derived
>   path** (`Content/DataMaps/<Entity>_<map>.png`, because the editor's Tauri capability set has
>   no save dialog — a user-chosen destination needs `dialog:allow-save` and is deferred with it).
> * **P19.2** — **coarse LOD pages carry no biome ids**, so a zoomed-out clipmap ring reads
>   *unassigned*; **`splat_layer` is declared but never applied** (a biome knows which terrain
>   layer it shades as and nothing reads it; making it automatic would let painting a biome
>   destroy authored splat work, so an explicit "apply biome splat" action is the follow-up);
>   **a shipped player draws the biome overlay neutral** — a **code gap, not a visual pass**:
>   `inf_player::render::project_terrain` gets the component and its data but no asset database
>   (the same reason per-layer textures never reached it), so it passes an empty palette and the
>   renderer pads; the ids project correctly and only the colours are missing. Wiring the
>   player's pack lookup into the projection is the fix; the **feather knob at v17** (the
>   per-biome feather width wants a schema slot and Phase 19's bump was already spent).
> * **P19.3** — the **`mask.image` player chain** (the editor resolves textures, the player
>   cannot; the cook advises, and closing it needs a texture fetch on the player's load path); a
>   **kinds palette node** (weighted kinds are still authored as one `mesh` + `weight` pair per
>   scatter node); **`float_roundtrip`** on params with full 17-digit mantissas, latent since
>   P10.5b; the **coarse-sampler LOD division** for data maps.
> * **P19.4** — a flexible slot changes **spacing, not mesh size** (`ScatteredInstance::scale`
>   is one `f64`; a stretch needs a per-instance scale *vector*, i.e. an ECS field and both
>   projector mirrors); **repeats resolve greedily left to right** (each reserves the *nominal*
>   length of everything after it, so two `*` in one sequence means the first wins); the
>   **scatter half of the cook's mesh edge** and a `PcgKind.mesh` dangling advisory; a
>   **document-only `.inf_pcg` carries no grammar**; **neither grammars nor buildings are
>   dispatched by the biome binding** — `BiomeBinding` is untouched and a biome's `.inf_pcg`
>   still contributes only its *scatter*. P19.5's `lot` pin closes half of what P19.4 named (a
>   span can now become a footprint), and the other half is open: nothing turns a painted
>   **region** into the set of lots that should stand in it. That generator is the real closure
>   and it is not written.
> * **P19.5 — the shape of the sample, stated as a deviation rather than a design claim.**
>   **One `PcgVolume` is one lot**, so the seven archetypes are seven small `.inf_pcg` graphs
>   rather than one canvas. A building's footprint is its volume's own box (the same default a
>   `grammar.footprint` span has), and putting seven of them on one canvas would need a per-node
>   lot offset — which is the volume's transform spelt a second way. It reads as what it is
>   (seven plots on a street) and it costs seven files; a "lots from a region" generator is what
>   would make one canvas right, and that is the open half of the biome-dispatch remainder above.
> * **P19.5** — **evaluation still runs once, at load** (inherited from P10.6, and now the
>   reason every `PcgVolume` in the sample is `AlwaysLoaded`; closing it is the prerequisite for
>   streamed procedural buildings); **the viewport still draws every scattered instance as a
>   placeholder cube**, buildings included, so a structural part's *collider* is exact and its
>   *render* is a cube — the kind→mesh upload gap sprites and tilemaps also have; **footprints
>   are axis-aligned rectangles** (a rotated or L-shaped lot needs an oriented rect type
>   throughout); **archetypes are code constants, not assets**; **buildings are not dispatched by
>   the biome binding** (a biome owns a region, a building needs a lot — the `lot` pin is the
>   seam, and a "lots from a region" generator is the natural next step); **no destruction
>   coupling** (Phase 22 will want fracture chunks, not these static boxes); the **adjacency scan
>   is O(n²)** in a floor's room count, which no shipped palette can reach (the area cap bites at
>   a few dozen rooms) but a very large lot with a very small `max_room_area` would — recorded
>   rather than pre-optimized against a case nothing produces, with the depth cap itself now
>   tested at the cap.

> **STATUS — P19.1 Erosion data maps: COMPLETE (2026-08-02).** Erosion stops discarding its
> story. Three per-cell accumulators now ride the whole pipeline — CPU reference, WGSL mirror,
> tile persistence, both scene codecs, the `.inf_terrain` container, undo, and a PNG16 export.
>
> **The accumulators, stated in SI** (`inf_terrain::DataMapKind` is the one definition):
> **flow** = `Σ_steps dt · (Σ_pipes outflow)` after the flux clamp — the time-integrated water
> volume the cell shipped, **m³** (the pipe flux is a volume *rate*); **deposition** =
> `Σ_steps` material settled, **metres** of height gained; **wear** = `Σ_steps` material
> dissolved, **metres** lost. All three are **raw and monotone**: nothing normalizes on the way
> in, so the **accumulators** sum across bakes and a normalized view (the export, P19.3's mask
> samplers) is always derived. **Thermal is deliberately excluded** — it relaxes slopes and
> conserves mass exactly, so folding it in would count relaxation as wear *and* add equal
> amounts to both totals; leaving it out makes the accounting identity **exact**, and
> `Σ deposition − Σ wear == Σ Δb` is the conservation gate.
>
> **Determinism is inherited, not re-argued.** The accumulators live in the same fixed-order
> loops as the fields they are computed from, so `erode`'s existing byte-identical guarantee
> covers them verbatim — machine-checked across repeated runs *and* under four job-pool sizes
> (the loop is serial by construction; the sweep pins that it stays so).
>
> **The unit is gated absolutely, not just relatively.** Every other flow assertion compares one
> flow number to another — a profile shape, a CPU/GPU agreement, a run-to-run hash — and a
> coherent mutation that drops the `dt` factor from *both* paths satisfies all of them: the maps
> would count steps instead of integrating seconds, and nothing relative would notice. What
> pins the m³ is that flow is a **time integral of a volume rate**, so at a fixed simulated
> duration it must not depend on how finely that duration is sliced: halving `dt` while doubling
> `steps` moves the total 2.6 % and then 1.3 % (converging), where the drop-`dt` mutation moves
> it **105 %**. A second arm pins the *volume* half — the same run on 2 m cells ships ~3.7× the
> water per cell. Mutation-verified in both directions.
>
> **The GPU mirror needed a binding.** The compute stage is hard-capped at **8 storage
> buffers** (wgpu's `Limits::default`, which `GpuContext` requests) and every slot was already
> spoken for, so the maps' slot was made by merging the water depth and the velocity into one
> `hydro` vec4 — same values, same order, same bits, a layout change and nothing else. The maps
> are **write-only** in the simulation, which is why adding them cannot move a height;
> `data_maps_do_not_feed_back_into_the_heights` proves it on both paths by eroding a
> map-seeded region and a clean one to identical terrain. Both parity tiers extend to the maps
> under the **same envelope discipline as the heights**: at 8 steps max|Δ| is
> 1.05e-6 (flow) / 6.1e-4 (deposition) / 2.1e-4 (wear) against a 1e-3·peak tolerance — the
> per-pass arithmetic is exact; at 50/100 steps the *totals* agree to 3.7e-3 / 2.7e-3 / 1.3e-3
> and 7.7e-3 / 5.2e-3 / 3.1e-3 relative, inside the same 5e-2 cross-adapter envelope the mass
> check uses, because a chaotic channel that moved one cell carries its flow with it.
>
> **Persistence: sparse on the splat-weights rule.** A tile stores `Vec<[f32; 3]>` — empty
> means *never eroded*. The human-readable form skips the field entirely; bincode always
> encodes it (the `skip_serializing_if` law), so an un-eroded tile costs **exactly one byte**,
> priced by test rather than asserted.
>
> **THE SCHEMA ANSWER: YES — scene v14 → v15, and `.inf_terrain` header v2 → v3.** The change
> is one level *below* the component (no `Terrain` field was added and none moved), but it is
> the same law as v12/v13 for the same reason: bincode is positional, so an extra
> length-prefixed layer **inside a tile** is a wire-format change, and a v14 payload fed to the
> grown tile would read past the end of its heights and into the next tile. The
> format-aware split localizes *where* the frozen record lives, not *whether* one is needed.
> So: the pre-P19.1 tile and heightfield are frozen once, in `inf-terrain`, as
> `TerrainTileV14` / `TerrainDataV14` (both codecs already share `inf_terrain::TerrainData`, so
> sharing its frozen twin is the existing seam, not a new one); each codec then freezes
> `TerrainV14` + `EntityRecordV14` + `SceneFileV14` and repoints v4..v14's terrain slots at
> them. v1..v14 payloads load unchanged with every tile's maps **empty** — never eroded, which
> is exactly what a v14 level meant. The `.inf_terrain` header did **not** change length; the
> *version* selects the tile wire type (`decode_tile_at`), which is also what makes a
> write-back over a v1/v2 source **transcode** its passed-through blobs instead of copying
> bytes — a v3 header over v2 blobs would pass every structural check and surface only as a
> corrupt tile on some later load.
>
> **Committed re-bless, priced.** Eleven `.inf_lvl` samples/templates move: the terrain-free
> ones by **exactly the version byte**, and the terrain-carrying ones by that plus **one byte
> per tile** (phase16-world +16, terrain-demo +9, phase18-scatter +4). Two new committed
> fixtures, `scene_v14.inf_lvl`, written twice and byte-compared across the codecs.
>
> **Undo is one step, two layers.** The bake's `DataMapDelta` is a *sibling* of its
> `HeightDelta`, not a field on it — every sculpt brush produces the first and only erosion
> produces the second, so merging them would put an always-empty map buffer inside every stroke
> on the undo stack. Both are recorded inside one `EditHistory` transaction, so one Ctrl+Z
> restores heights **and** maps byte-identically (including dropping a materialized buffer back
> to the sparse default) — gated at the `SceneDoc` layer, not just at the delta types. The
> **record order is load-bearing and documented where it is chosen**: `undo` reverts a
> transaction in reverse, so heights-then-maps means maps are restored while every tile is still
> present. (`revert_delta` may remove tiles a stroke authored and `revert_data_map_delta` skips a
> patch whose tile is gone; benign for erosion specifically, because `HeightRegion::write_back`
> never creates a tile — but the ordering must not depend on that.) The P16.4b save write-back
> needs no change: writing a map dirties the tile through the same `get_tile_mut` seam the
> heights use, asserted by a map-only-edit test that follows it into `TerrainEdits::from_dirty`.
>
> **Export** is `TerrainData::export_data_map(kind, region)` → PNG16, normalized over the
> region's own measured `[min, max]` and **reporting that range**, plus a
> `terrain_export_data_map` command writing under `Content/DataMaps/` (a derived destination —
> the editor's capability set has no save dialog) and three buttons in the erosion dialog.
>
> **Tests.** `inf-terrain::maps` (sparse-is-free, delta round-trip + byte-identical undo,
> export normalization/determinism, region clipping); `tests/erosion.rs` (accumulator
> determinism across runs and pool sizes; the conservation identity against both the measured
> heights and `ErosionStats`; **known values** — a V-valley's flow profile peaks at the floor,
> is unimodal, and beats each crest 2×, while the steep flank wears hardest and the valley sits
> above it on net balance; **flow is a time integral, not a step count** — halving `dt` at fixed
> simulated time moves the total 2.6 % then 1.3 %, where a dropped-`dt` accumulator moves 105 %, and
> the same run on 2 m cells ships ~3.7× the volume; a second bake adds to the first's totals and
> un-composes exactly, while moving a measured 34 % *less* than one long run; holes never
> accumulate; a map-only edit still dirties its tile for write-back; both
> codecs round-trip); `inf-terrain::tile` (the legacy-decode contract — see below);
> `inf-terrain::asset` (a v2 payload loads forever with default maps; a v2 source transcodes on
> write-back); `erosion_gpu` (both parity tiers over all three maps, plus the no-feedback
> guard); `SceneDoc` — **the undo gate**: an `edit_erode_region` bake is exactly one history
> entry, and one `undo` returns the terrain's *saved bytes* (heights + weights + maps in one
> encode) to what they were, with the materialized buffers dropped back to sparse, redo
> reproducing both layers; and the v15 ladder in both codecs (fixture provenance, cross-codec
> byte identity, old-bytes-load-forever, the downgrade's single documented loss, and the wire
> cost priced per tile). **41 goldens byte-identical** — data maps are data, not pixels.
>
> **Legacy reads keep their teeth.** The frozen record is now the *only* path a pre-P19.1
> payload takes, so it carries the live decoder's length contract verbatim, in its own
> `Deserialize`, as **hard errors** — a corrupt height buffer is a decode failure, never a
> terrain full of holes that gets saved back, and a short-but-non-empty weight buffer is
> rejected at the door rather than becoming an out-of-bounds index the first time the paint
> path touches the tile. That hazard is pinned by a `#[should_panic]` test on the hand-built
> shape and a paint-path sweep over every buffer a legacy payload can legally carry; deleting
> the check makes the sweep fail with the index panic. The frozen type is named
> `TerrainTileFrozenV1` / `TerrainDataFrozenV1` rather than `…V14`, because one tile layout
> backs **two** independently-versioned containers (`.inf_lvl` ≤ 14 *and* `.inf_terrain` ≤ 2)
> and neither owns the other's numbering; three tripwire tests — one per codec plus the asset
> half in `inf-terrain` — fail if either ladder bumps past its row without a new generation.
>
> **Remainders, stated (P19.1).**
> * **Thermal wear is invisible in the wear map.** Thermal erosion is excluded so the
>   conservation identity stays exact (above), which means a talus-relaxed cliff face reads as
>   *unworn*. For a texturing mask that is arguably right — thermal does not carry material away,
>   it slumps it — but a consumer that wants "how much did this face move" needs a fourth channel.
>   Deferred until something asks for it; the channel count is one constant.
> * **Coarse LOD pages carry no data maps.** Like the splat weights, a decimated tile is a
>   streaming page rather than authored content, so `downsample_block` leaves the maps sparse
>   (asserted in the write-back gate). **P19.3's samplers therefore read zeros above level 0** —
>   fine for the editor and for a level-0 evaluation, and a real gap for any future
>   coarse-resolution PCG pass. The fix is a decimation rule for accumulators (sum, not average,
>   for flow; mean for the metre channels) and it belongs with the first consumer.
> * **The bless-guard asymmetry.** The three bless switches do not agree on what "on" means:
>   `INF_BLESS_SAMPLES` and the editor codec's `bless_v14_fixture` accept **any** value
>   (`is_ok()` / `is_err()`), while `inf-scene`'s `bless_scene_v14_fixture` demands **exactly
>   `"1"`** (`as_deref() != Ok("1")`). So `INF_BLESS_FIXTURES=true cargo test --workspace`
>   rewrites the editor's `scene_v14.inf_lvl` and silently leaves the runtime's alone — the two
>   diverge, and the only thing that says so is
>   `v14_fixture_matches_the_runtime_codecs_copy` failing on the next run. It *is* caught (that
>   is exactly the mirror test's job, and it is why the test exists), but a one-line predicate
>   mismatch shouldn't be what stands between a bless and a corrupt fixture pair. Inherited, not
>   introduced here; unifying the three predicates is a cheap follow-up.
> * **`export_data_map` writes to a derived path.** The editor's Tauri capability set has no save
>   dialog, so the PNG lands under `Content/DataMaps/<Entity>_<map>.png`. A user-chosen
>   destination needs `dialog:allow-save` and is deferred with it.

> **STATUS — P19.2 Biome painting: COMPLETE (2026-08-02).** Authors now paint *where the world
> is* the way they paint what it is made of. A `.inf_biomes` asset defines the level's biomes,
> a per-sample id layer rides the tile beside the heights / weights / data maps, a brush writes
> it with full undo, and a Biomes view mode shows the result.
>
> **THE HARD-EDGE DECISION, and what it cost to make it honest.** A biome id is **categorical**:
> there is no "half forest, half desert" id, so the brush writes a crisp boundary and the falloff
> and strength decide *where that boundary falls* —
> `claimed ⇔ weight(d, r) > 0 ∧ weight(d, r) ≥ 1 − strength`. Strength 1 stamps the whole disk;
> strength ½ claims the half-weight contour (a smaller disk under a soft falloff, the whole disk
> under `Plateau(1.0)`); strength 0 claims nothing. Monotone in both parameters, pure, and every
> slider position does something visible — machine-checked, including that a **flat** falloff is
> strength-independent, which is what proves the curve (not the number) is doing the work.
> **The briefed "soft radius writes a majority vote" was considered and rejected**, and the
> reason is not aesthetic: a majority vote over each sample's neighbourhood makes a dab depend on
> ids the *previous* dabs wrote, so a stroke stops being a function of its path (the editor
> resamples a drag into dabs at ~⅓ radius — a timing-dependent count), and re-applying the same
> dab keeps changing the terrain, which breaks the `apply(revert(apply)) == apply` identity every
> delta in the crate rests on. Boundary *feathering* is a real requirement and it belongs at the
> **consumer**: P19.3 blends adjacent biomes' PCG graphs across a feather width, reading the crisp
> ids stored here. Feathering the storage would throw away exactly the information that blend needs.
>
> **THE SCHEMA ANSWER: YES — scene v15 → v16, and `.inf_terrain` header v3 → v4.** Same law as
> v15, one generation on: bincode is positional, so an extra length-prefixed layer inside a tile
> is a wire-format change. The generation table now has three rows and it lives, stated once, on
> `TerrainTileFrozenV1`:
>
> | tile generation | layout | `.inf_lvl` | `.inf_terrain` |
> |---|---|---|---|
> | `TerrainTileFrozenV1` | origin + heights + weights | v1 … v14 | v1, v2 |
> | `TerrainTileFrozenV2` | + erosion data maps (P19.1) | **v15** | **v3** |
> | `TerrainTile` (live) | + biome ids (P19.2) | v16 | v4 |
>
> Three tripwires (one per codec plus the asset half in `inf-terrain`) fail if any container bumps
> past its row without a new generation, and the asset-half one now proves the *whole* mapping —
> each generation's bytes decode at exactly its own header versions and produce the same tile.
> The `Terrain` component also gained `biome_set: Option<Uuid>` (additive, `serde(default)`,
> `reflect(ignore)` — an asset reference like `MeshRef::asset`), but the **tile** change is what
> forced the bump; the field would have ridden a `serde(default)` for free.
>
> **The wire cost, measured rather than assumed.** `biome_set: None` costs **exactly 1 byte**
> (the bare bincode `Option` discriminant) — measured on a *tile-less* terrain so the per-tile
> counts cannot mask it. An unpainted tile costs **exactly 1 byte** (the zero-length count of the
> sparse `biomes` sequence); a painted one costs **exactly `res²`** (one `u8` per sample — *not*
> ×4, which is the mistake a copy of the data-map pricing test would have made). Eleven `.inf_lvl`
> samples/templates re-blessed: the terrain-free ones move **zero bytes** (v15 and v16 are the same
> varint width), and the terrain-carrying ones by the biome_set byte plus one per tile —
> streamed-terrain +1 (no inline tiles), phase18-scatter +5, character-demo +7, terrain-demo +10,
> phase16-world +18. Two new committed fixtures, `scene_v15.inf_lvl` (1243 B), written twice and
> byte-compared across the codecs.
>
> **The `BiomeSet` asset (`.inf_biomes`, kind code 19).** `Vec<BiomeDef { id, name, colour,
> splat_layer, pcg_graph, water_hint, structure_hint }>`, deterministic bincode, **compressed**
> (it is a short list of names, not a streaming page). **Id 0 is reserved** and undefinable —
> that reservation is *what makes the storage sparse*, because an unpainted tile's default has to
> mean something coherent, and it is enforced on validate **and on decode**, so a hand-edited file
> fails at the door rather than becoming an ambiguous lookup. `water_hint` / `structure_hint` are
> inert plain data, declared now because they are what P20 and P19.5 will ask a biome for and
> adding a bincode field later costs a migration on every `.inf_biomes` in every project; storing
> them as *hints* rather than references keeps them honest (an empty hint is not a dangling edge).
> `pcg_graph` **is** a real reference: it becomes a sidecar dependency edge, so a set that names a
> graph protects it through delete-with-references and pulls it into the cook closure.
> `splat_layer` is **advisory** — painting a biome deliberately does *not* rewrite the splat
> weights, because silently overwriting one authored layer from another makes a paint stroke
> unpredictably destructive.
>
> **The paint tool is the splat seam, one layer over.** `BiomeDelta`/`BiomePatch`/`BiomeStroke`
> mirror `SplatDelta`/`SplatPatch`/`SplatStroke` exactly, `materialized_tiles` on the P19.1
> `is_empty` standard (both halves, so a stroke that only materialized a buffer still has undo
> work). `EditCommand::PaintBiome` is a **sibling** of `PaintSplat`, not a field on it, for the
> reason `PaintSplat` is a sibling of `SculptTerrain`: merging them would put an always-empty
> buffer inside every stroke on the undo stack. `ToolMode::Biome` is likewise its own tool rather
> than a `SculptOp` sub-mode (the Foliage precedent) — folding it in would have made
> `SculptSettings::strength` mean a third thing depending on the op. It rides the **same** stroke
> machinery: one terrain pick, one footprint page per dab, the streamed read-only gate, one
> command per stroke. **Ctrl erases** (writes the reserved id), and the undo entry is labelled
> "Erase Biome" so the menu distinguishes it. Streamed terrain needed **zero** new write-back
> code — ids live inside the tile blob, so `TerrainEdits::from_dirty` carries them for free —
> which is exactly why it got a test: "for free" is a claim about the `get_tile_mut` seam, and a
> future optimization writing ids through an immutable back door would drop every biome stroke on
> save with nothing else noticing.
>
> **The overlay is a uniform flag, not a pipeline.** `ViewMode::Biomes` sets `view.flags.y`;
> `unlit_flag()` returns 1.0 for it too, so `mesh`/`skinned_mesh`/`vgeom_mesh`/`scatter_mesh` need
> **no edit at all** and non-terrain geometry simply renders unlit — the smallest honest
> treatment. Terrain gains an `R8Uint` per-tile id texture at `@group(1) @binding(2)`, sampled with
> `textureLoad` and **never interpolated** (an id is categorical; the midpoint of biome 3 and
> biome 7 is one of the two), and a 256-slot palette uniform at `@group(2) @binding(1)` in its own
> buffer, so `MaterialRaw`'s bytes are untouched. Terrain already spent all four bind groups, which
> is why both are new *bindings* rather than a `@group(4)`. The tint is palette × wrapped N·L, so
> the landform's relief still reads. **Off-path byte-stability**: `flags.y` is exactly 0.0 in every
> other mode, the branch is present-but-false, and a full `INF_BLESS_GOLDENS=1` sweep moved
> **nothing** — `git status` showed only the new `biomes.png`. **42 goldens**, 41 byte-identical.
>
> **Cook.** `Terrain.biome_set` is a real level→asset edge (found only by walking the persisted
> component — the fixture's sidecar declares no dependency), a dangling one is a **deduplicated
> advisory** rather than a failure (the level is still valid; its ids just resolve to nothing), and
> an **invalid** set is a hard `CookError::BiomeSet`: ambiguous ids cannot be recovered from at
> runtime because the per-sample values are already baked into the tiles.
>
> **Tests.** `inf-terrain::biome` (starter round-trip + determinism, id-0 rejected on validate
> *and* on decode, duplicate/blank/bad-layer, a full 255-biome set and the 256th refused,
> `next_free_id` gap-skipping, the id-indexed palette's padding and purity, schema-too-new, the
> kind wiring); `inf-terrain::biomepaint` (the claim rule's monotonicity/totality/zero-strength
> no-op and its documented ½ contour; **strength shrinks the disk instead of fading it** on a fine
> lattice; a flat falloff is strength-independent; paint determinism; byte-identical undo + exact
> redo; a stroke merges dabs so one revert walks all the way back; the eraser is a first-class
> stroke; no tile authoring; a repeat dab is a no-op that materializes nothing; seam continuity;
> **sparse-is-free priced in bytes**; a biome edit dirties its tile for write-back; `biome_at` is
> nearest, never interpolated); `inf-terrain::tile` (generation-2 wire shape both codecs, the
> three-row tripwire, clamp/ignore parity with the other layers); `inf-terrain::asset` (a **v3**
> payload loads forever *keeping its maps*, a v3 source transcodes on write-back, the v4 per-tile
> price); `inf-editor-core::assets::biome_set` (create/get/save/list, **saving validates** and
> leaves the file untouched when it refuses, `pcg_graph` becomes a dependency edge, wrong kind
> refused); `terrain_edit` (a biome-only edit writes back, a fresh streamer reads the ids, coarse
> pages stay unpainted, and an undone stroke restores byte-identical tile blobs); `SceneDoc` —
> **the undo gate**: one stroke is one entry and one Ctrl+Z returns the terrain's *saved bytes*
> (heights + weights + maps + ids in one encode) with the materialized buffers dropped back to
> sparse, plus the eraser's label and the idempotent biome-set binding; both codecs' v16 ladders;
> `inf-packager` (the edge is followed, a dangling one advises, an invalid set fails the build);
> `inf-render` (`golden_biomes` + GPU-free palette unit tests + the naga gate).
>
> **Remainders, stated (P19.2).**
> * **A shipped player draws the overlay neutral.** `inf_player::render::project_terrain` receives
>   only the component and its data — there is no asset database on the player's projection path
>   (the same reason per-layer *textures* never reached it) — so it passes an empty palette and the
>   renderer pads. The ids themselves project correctly; only the colours are missing, and only in
>   a mode that is an authoring aid. Wiring the player's pack lookup into the projection is the fix.
> * **Coarse LOD pages carry no biome ids**, exactly like the splat weights and the P19.1 data
>   maps: a decimated tile is a streaming page, not authored content (asserted in the write-back
>   test). So a zoomed-out clipmap ring reads *unassigned*, and P19.3's samplers see zeros above
>   level 0. The fix is a decimation rule for a categorical channel — majority vote, not average —
>   and it belongs with the first consumer.
> * **`splat_layer` is declared but never applied.** A biome knows which terrain layer it "shades
>   as" and nothing reads it. An "apply biome splat" action is a small, obvious follow-up; it was
>   left out because making it automatic would make painting a biome destroy authored splat work.
> * **The bless-guard asymmetry is still there.** Inherited from P19.1 and deliberately not
>   "fixed" mid-batch: `INF_BLESS_FIXTURES` means *any value* in the editor codec and *exactly
>   `"1"`* in `inf-scene`. Always use `=1`.

> **STATUS — P19.3 Biome → PCG binding & node-kit completion: COMPLETE (2026-08-02).** The two
> halves P19.1 and P19.2 laid down finally do something: a painted terrain **grows** what belongs
> on it, and the node kit stops having documented holes.
>
> **THE SCHEMA ANSWER: NO — nothing bumps. Scene stays v16, `.inf_terrain` stays header v4,
> `.inf_biomes` stays v1, `.inf_pcg` stays v2.** Three separate arguments, one per thing that
> looked like it might:
> * **The coarse pages already had a wire form for the layers.** Header v4 selects the *live*
>   `TerrainTile`, and bincode encodes its `maps` and `biomes` sequences unconditionally (the
>   `skip_serializing_if` law). A coarse page that now carries a reduction is the same wire form
>   with a non-empty vector — no new field, no new type, no builder change. The asset builder
>   needed **nothing**; `encode_tile` already wrote whatever the tile held, and P19.2 simply
>   never handed it a coarse tile that held anything. Determined by reading the encoder, then
>   pinned by the write-back test that now asserts the coarse pages *do* carry the reduction.
> * **`Terrain.biome_population` is `#[serde(skip)]`**, exactly like `PcgVolume.evaluated` — the
>   established precedent for a derived cache. Priced rather than asserted: encoding a `Terrain`
>   with a 1000-instance population produces **byte-identical** output to an empty one, and a
>   decode leaves it empty. No frozen `TerrainVn` record was touched.
> * **`ScenePayload` v3 → v4** is the one version that moved, and it is the **PIE envelope's own**
>   — a transient editor↔player IPC frame, not a scene or asset schema. Same append-a-field,
>   bump-the-envelope move v2 (pcgs) and v3 (anim) already made.
>
> **THE ORDERING LAW, paid for in this batch.** Adding `SamplerDef::DataMap` / `::Biome` beside
> the other *sources* — where they read best — shifted `Multiply/Max/Min/Invert` by two
> declaration indices, and bincode writes an externally-tagged enum as its **declaration index**.
> The committed `terrain-demo` `.inf_pcg` changed bytes and `committed_sample_matches_generators`
> caught it. The fix is the rule, not the patch: **new variants are appended, never inserted**,
> and `sampler_variant_discriminants_are_frozen` now pins all eleven indices so the next person
> fails at `cargo test` instead of in somebody's project. (This is the *same* law as the tile's,
> one type down: positional encodings have no names.)
>
> **And the law applies to the enum NESTED inside it.** `SamplerDef::DataMap` carries an
> `inf_terrain::DataMapKind`, which until now was persisted nowhere and so carried no ordering
> constraint at all — while P19.1's own remainders contemplate adding a fourth *thermal* channel.
> Inserting one would silently turn every committed `mask.wear` into a `mask.deposition`. The
> kind now carries the append-only note where it is declared, its three indices are pinned in the
> same freeze test (including one spelled out as raw bytes — `DataMap(Wear)` starts `[9, 2, …]`),
> and `channel()`'s storage order is asserted **separately** so the two can never drift into
> agreeing by accident.
>
> **A SECOND LAW, found by the property and not by a report: `serde_json`'s default parser can
> land 1 ULP off.** A `.inf_pcg` stores its authored graph as a **JSON string** (`graph_json` —
> the `inf_graph` model's `skip_serializing_if` fields are not bincode-safe), and the shipped/PIE
> player **re-lowers that graph** rather than trusting the stored document mirror. So every
> authored `f64` param crosses JSON on the way to the player — and serde_json's fast float path
> is not exact. A `base_density` or a noise `gain` that came back a bit light hands the player a
> different `ScatterParams` from the editor's, which surfaces only as *the shipped world's foliage
> sits somewhere else*, on somebody's machine, months later. The workspace pin is now
> `serde_json = { features = ["float_roundtrip"] }`, stated with its reason at the pin, and
> `an_authored_graph_re_lowers_bit_identically_through_the_payload` compares the editor's and the
> player's documents **bit for bit** (`to_bits()`, so a future sloppy `PartialEq` cannot hide it)
> on params with full 17-digit mantissas. This is latent since P10.5b; P19.3 is simply where a
> round-trip property was pointed at it.
>
> **And PCG is not the biggest exposure** — the audit's catch. A `.inf_act` is JSON too, and the
> player decodes one on **every** boot path (dev dir, cooked pack, and the PIE
> `ScenePayload.classes` list) — far more traffic than `.inf_pcg`, and a literal that came back a
> bit light makes the shipped actor compute imperceptibly differently from the preview, with both
> sides internally consistent so nothing below the simulation notices. `inf-blueprint` now carries
> the twin test: every `Lit::Float` in a class — nested inside a `BinOp`, and on a member
> variable's default — round-trips `to_bits()`-identically and re-serializes byte-identically, on
> the same 17-digit values (including the one the property actually found). Neither crate is now
> the sole reason the pin exists.
>
> **The coarse-page decimation rules (the tracked prerequisite from both P19.1 and P19.2).** A
> tile has four layers and they do not reduce the same way, because they do not mean the same
> thing. Stated once, on `pyramid`:
>
> | layer | rule | why |
> |---|---|---|
> | heights | **decimate** (unchanged) | a point value on a lattice; anything else cracks the LOD mesh |
> | splat weights | **not carried** (unchanged) | authoring-resolution paint below a screen texel at LOD *n* |
> | data map **flow** (m³) | **sum** the footprint | *extensive* — a coarse cell covers 4 cells' area, so it shipped their total water |
> | data maps **deposition / wear** (m) | **mean** the footprint | *intensive* — metres of height change is per-area; the mean is what survives a level |
> | biome ids | **majority vote**, ties to the **lowest id** | categorical: the midpoint of biome 3 and biome 7 is one of the two |
>
> **The reduction follows the DIMENSION, not the word "accumulator" — and P19.3 got that wrong
> for one commit.** All three maps are raw monotone accumulators, so the first cut summed all
> three; the audit caught that this 4×-inflates the metre channels every level, and it was
> already writing into `.inf_terrain` pages. Flow is **m³** of water shipped — extensive, so a
> coarse cell holds its four children's *total* and `Σ over a world region` survives the level
> (its per-level value scales ×4, which is correct: so did the area). Deposition and wear are
> **metres** of height moved — a per-area *intensity*, which P19.1's own definition says to
> multiply by `l²` to get a volume — so doubling the area does not double a height, and the
> **mean** is the value that survives; it preserves the volume integral too
> (`mean(h)·4A == Σ(h)·A`). Summing them would make a mask thresholded at level 0 stop matching
> at level 1, which is exactly backwards from the reason flow sums. P19.1's own remainder note had
> stated this split (*"sum, not average, for flow; mean for the metre channels"*) and it was not
> read closely enough. `DataMapKind::is_extensive()` is now the single place the dimension lives,
> so the next area-changing consumer branches on a property rather than re-deriving a rule.
> **The tie-break is the lowest id, and id 0 votes like any other value** — a coarse texel over
> mostly-unpainted ground honestly reads *unassigned*, rather than being rescued by a special case.
>
> **Shared-edge continuity for categorical data, and what it cost.** The footprint is the coarse
> sample's own 2 × 2 fine block `{2I, 2I+1} × {2J, 2J+1}` — **except on the shared-edge ring**
> (`I` or `J` ∈ `{0, res−1}`), where it degenerates to the single sample, i.e. the ring
> *decimates, exactly like the heights*. That is the seam guarantee, not an optimization: a coarse
> tile's far edge and its `+X` neighbour's near edge are the **same fine world sample**, so they
> reduce to bit-identical bytes only if neither aggregates. Every non-degenerate window at the
> ring was tried on paper and every one of them needs fine samples from *outside* the 2 × 2 block
> — which terrain sparsity forbids (the block may be all that exists) and which would break the
> invariant the partial write-back rebuild rests on, since `downsample_block` is fed only the
> block's four members. Widening the write-back to a 3 × 3 fine neighbourhood would also have to
> widen its staleness propagation *and* `chunked.rs`'s memory-bounded row-band importer — a large
> blast radius for a streaming page's edge ring. So the ring is decimated, and the price is
> **only paid by flow**, priced exactly: the ring's window is not uniformly one sample (a corner
> reduces 1 child, a non-corner edge 2, the interior 4), and a *mean* of 1 or 2 equal-area
> children is still the right per-area value, so the metre channels and the categorical ids lose
> nothing at all. What flow loses is the single combined index **`1`** per axis, which falls in
> no window — every other odd index *is* covered (`2res−3` belongs to `I = res−2`'s window, which
> an earlier draft of this block got wrong) — for an uncovered fraction of
> `1 − ((2res−2)/(2res−1))²` = **0.39 % at `res = 256`**. Asserted both ways: X- and Z-seam
> equality for **ids and maps** at every pyramid level, and the four-member block still reducing
> **bit-identically** to a whole-level reduction. The seam guarantee is also **conditional on
> coverage**, and that is now tested rather than implied: a block member that is absent
> contributes nothing to its window, so where two coarse tiles have different coverage the shared
> sample reads the sparse default on the side that cannot reach it — exactly the heights' own
> hole behaviour, with a full-coverage positive control beside it. **Sparsity survives on both ends** — a block
> whose members carry no maps / no ids skips the reduction wholesale, and a reduction that comes
> out all-default is dropped back to the sparse default rather than materialized, so an
> un-eroded, unpainted terrain's pyramid costs exactly what it did before P19.3.
>
> **The binding.** `BiomeBinding` is a **terrain-level sibling** of the volume path, not a
> replacement: a volume scatters one graph over a box an author placed; a binding scatters many
> graphs over the regions an author *painted*. Both run at the same two moments — the editor's
> `pcg_evaluate_biomes` command and the player's load-time pass — and neither reads the other's
> output. Three determinism rules, all machine-checked: **dispatch order is ascending biome id**
> (the constructor sorts, so a set's declaration order cannot leak into placement); **the biome id
> is folded into the counter-hash** (`biome_seed(seed, id)`) — the tuple every draw is a pure
> function of simply grew the biome; and **masking is multiplicative, so it only ever removes** — the scatter kernel's
> acceptance draw does not depend on the density value, so a bound rule's output is provably a
> *subset* of the unbound rule's, which is what makes the region property testable at all. The
> whole GUID-onward half is `BiomeBinding::from_set`, shared **verbatim** between the editor
> command and the player; the two paths differ only in how they fetch a `.inf_pcg`.
>
> **What actually prevents two biomes co-placing today is DISJOINTNESS, not the seed fold** — a
> correction the audit forced, because the first draft credited the wrong mechanism.
> `TerrainFields::biome_id` answers with exactly one id per position and `BiomeMask` scores `0`
> unless it matches, so at most one biome's mask is positive anywhere and the feather only *thins*
> a biome inside its own region; there is no both-masks-positive band in P19.3, and a
> two-biomes-on-a-split-terrain test would have passed with `biome_seed` deleted. The fold is a
> **forward guard** for the moment anything makes the masks overlap (the deferred per-level
> feather blending across a border, a soft id field), and it is now gated on its own terms: one
> document run as id 1 and as id 2 against terrains that are *entirely* those ids — the same
> region, fully-overlapping masks, nothing different but the id — must produce different position
> sets. Mutation-verified: deleting the fold fails that test, and disjointness itself is asserted
> separately by sweeping both masks across the border and requiring their product to be zero
> everywhere (which is also the tripwire for the day the feather starts blending across).
>
> **The border blend.** `BiomeMask` feathers by **distance to the nearest unlike sample** —
> including off-terrain, so a terrain edge is a border like any other — found by an expanding
> ring search that stops as soon as no further ring can beat the best hit and is capped at
> `MAX_FEATHER_SAMPLES = 64` (the search is `O(k²)` per candidate in the worst case, so the radius
> is bounded rather than trusted). The boundary sits about half a sample inside the nearest unlike
> point, so that half-spacing is subtracted before the `smoothstep` — otherwise every mask would
> read `smoothstep(spacing)` at its own edge instead of 0. Monotone in depth by construction and
> tested as such, `feather <= 0` costs no search at all, and the width is *metric*: the same 4 m
> blend saturates 4 m in on a 1 m and a 2 m lattice alike.
>
> **Where the feather is authored, honestly**: nowhere persistent. A per-level value needs either a
> `Terrain` field (scene v17 in both mirrors) or a `.inf_biomes` bump with a frozen record, and
> this batch buys no schema bump — so the engine default is `DEFAULT_BIOME_FEATHER = 8.0 m` and the
> *authored* path is per-graph, through the `mask.biome` node's own `feather` param. Stated as a
> remainder below.
>
> **The multi-rule graph shape: merge nodes, and why.** The substrate enforces one link per input
> pin, so several scatter chains cannot meet at one port. A variadic/indexed sink
> (`layer0…layerN`) is an arbitrary cap and a shape no other domain here uses; an explicit `rule`
> node duplicates what `scatter.scatter` already is. A **binary combinator that associates left**
> is instead the convention this very registry already has three times over — `combine.multiply` /
> `max` / `min` join two densities exactly that way, and the lowerer already flattens *those*
> recursively. So `scatter.merge` (SCATTER × SCATTER) and `layer.merge` (LAYER × LAYER) are the
> same idea one and two wire types up: no new concept, no cap, and the sink keeps its single-pin
> shape. `layer.layer` wraps a scatter chain with `name`/`enabled`; `output.pcg` **keeps** its
> `scatter` input and gains `layers`, so every `.inf_pcg` authored before P19.3 lowers
> byte-identically (one implicit layer named `layer`) and connecting both is a node-anchored
> warning with `layers` winning. Merge trees flatten `a`-then-`b` depth-first, so the rule list
> reads the way the canvas does. `scatter.scatter` gained a `name` param (sparse `ParamMap`, so
> old graphs resolve the default) because N rules need distinguishable names.
>
> **The node kit's holes are closed.** `mask.image` resolves its texture GUID through a new
> `MaskSource` seam — the lowerer holds no asset database, so the pixels come in from the caller —
> and **fails closed**: a blank, malformed or unloadable GUID lowers to a `0 × 0` mask that scores
> `0` everywhere with a node-anchored *warning*, never "place everywhere" and never a hard error.
> `mask.flow` / `mask.deposition` / `mask.wear` each carry their own `min`/`max` **because the
> stored data is raw** — normalization is a view the reader chooses (the P19.1 doctrine), and two
> masks over one terrain may legitimately want different windows. `mask.biome` matches an id with
> the same feathering seam the binding uses.
>
> **The layer seam.** The masks read a new `TerrainFields` trait — `data_map` / `biome_id` /
> `sample_spacing` — deliberately *not* folded into `HeightProvider`, because a purely procedural
> height function has no data maps and no biomes and forcing it to answer would make every
> `FnHeight` in the codebase lie. `NoFields` is the honest empty implementation and is what the
> old `evaluate` / `SamplerDef::build` entry points pass, so a mask without a terrain places
> nothing rather than everything. `OffsetTerrain` is the world-offset wrapper both evaluation
> sites share, so "the terrain is at `origin`" cannot mean two things.
>
> **The editor surface.** `pcg_evaluate_biomes` is the terrain-level sibling of `pcg_evaluate` —
> resolve the terrain, load its `.inf_biomes`, resolve each biome's graph, evaluate, write the
> population, emit `world://delta` — and a "🌿 Evaluate Biomes" button beside the existing
> "⚡ Evaluate" on the PCG canvas. The palette gains the `masks` and `layer` categories and the
> new LAYER wire colour; the frontend needed no new concepts because the registry is served from
> Rust and the canvas is registry-driven. **Both projector MIRRORs** push the population through
> one shared `push_scatter` helper — the volume path and the terrain path cannot drift, which is
> now asserted by the extended mirror gate rather than hoped for.
>
> **Gates.** `runtime/inf-player/tests/biome_pcg.rs` builds a two-biome painted terrain whose
> `.inf_biomes` names two **distinct** `.inf_pcg` graphs — one single-rule, one a real
> `scatter.merge` two-rule graph, so the new lowering rides the shipping pipeline rather than only
> its unit tests — and proves: the cook follows the **whole chain** (level → `Terrain.biome_set` →
> `.inf_biomes` → each `BiomeDef.pcg_graph` → `.inf_pcg`) with only the level as an explicit root
> (P19.2 landed the first hop; this is the first test of the second); **cooked == uncooked**
> instance for instance; **PIE == shipping** on the same population; both biomes contribute and
> every instance lands on ground its own biome owns; an **unpainted** terrain grows nothing; and
> two loads of one pack are identical. The fixture is built in-test rather than committed, because
> it exists to pin the *chain* and a temp-dir fixture cannot drift from a `.inf_lvl` nobody
> regenerated.
>
> **Tests.** `inf-terrain::pyramid` (the tie-break's totality and order-independence; the
> footprint's degeneration on the ring; a deterministic majority vote that never invents an id;
> **the dimensional split** — a uniform *metre* field surviving a level unchanged at
> interior/edge/corner while a uniform *flow* field scales ×4/×2/×1, plus each channel's own
> conservation statement (flow's volume total; deposition's height × area); X/Z seam equality for
> ids and maps at every level; **the coverage-hole seam** — an unreachable shared sample reads the
> sparse default rather than a guess — with a full-coverage positive control beside it; a clean
> terrain still pyramiding to sparse layers; the block-vs-level
> bit-identity **with layers live**); `inf-terrain::maps` (`data_map_at` nearest/raw/bounded,
> `xz_bounds` incl. negative coordinates and the empty terrain); `inf-pcg::fields` (the empty
> source, both layers off a real terrain, the offset wrapper shifting every layer by one origin);
> `inf-pcg::sampler` (known-value normalization windows incl. the degenerate one, crisp-then-
> monotone biome feathering, the metric feather across two lattice spacings, the capped radius);
> `inf-pcg::rules` (**the frozen discriminant table** + both new variants round-tripping in both
> codecs, nested); `inf-pcg/tests/sampler_roundtrip.rs` — **the round-trip discipline made a
> property**: arbitrary recursive sampler trees and whole layers × rules documents survive bincode
> *canonically* (re-encode is byte-identical, which is what keeps a `.inf_pcg`'s content hash
> stable) and survive JSON, because the failure mode here is "a variant nobody wrote a case for",
> which is exactly what a hand-written corpus misses; `inf-pcg::graph` (every mask node's lowering + round-trip, param clamping,
> `mask.image` resolved / unresolved / blank / malformed, ordered `scatter.merge` flattening,
> N rules × M layers through the payload codec and back, the legacy sink shape unchanged, the
> both-inputs tie, anchored diagnostics on every new walk, and a shared density subgraph *not*
> being a cycle); `inf-pcg::binding` (the region property, an unpainted terrain, monotone feather
> blending with fewer instances than a crisp mask, ascending-id dispatch, the id-in-the-seed
> no-co-placement proof, subset-not-move, `from_set` skipping unresolvable and reserved ids, the
> rewrite stated in one place, disjointness swept across the border, **and pool-size invariance
> through the REAL path** — `BiomeBinding::evaluate_in`, a new caller-supplied-pool seam mirroring
> `scatter_region_in` with `evaluate_with_in` beneath it, run at 1/2/4/8 workers over two biomes ×
> two layers × three rules, so the per-biome seed fold, the mask wrapping, the weighted kind picks
> and the concatenation order all participate; the first version lifted one rule out of a document
> by hand and therefore covered none of them); `inf-blueprint` (the `.inf_act` float twin);
> `inf-ecs` (the population costs zero
> bytes); `inf-editor-core` (the projector MIRROR gate extended; the coarse-page assertion flipped
> to "never invents an id"; and a **new `biome_binding_mirror` gate** reading both evaluation
> paths' source text — same Ring-0 seams by name, same `.inf_pcg`→document resolution rule, and
> the two passes' *seed* rules kept apart, since a copy-paste of the volume's `wrapping_add` into
> the biome pass would make two biomes sharing one graph co-place, which is the exact bug
> `biome_seed` exists to prevent). **42 goldens byte-identical** — a population is instances, not
> pixels, and no golden scene builds a pyramid.
>
> **Remainders, stated (P19.3).**
> * **The border feather has no persisted per-level knob.** `DEFAULT_BIOME_FEATHER` (8 m) is the
>   engine default and `mask.biome`'s param is the authored path; a per-level override needs a
>   schema bump this batch deliberately did not take. First thing to add when a `.inf_biomes`
>   bump is warranted for another reason.
> * **`mask.image` resolves in the editor and NOT in the player**, and this is the one place
>   P19.3 knowingly ships a preview/shipping difference. The editor lowers through
>   `AssetMasks` (the thumbnailer's existing CPU BC decode → Rec.709 luma), so an image mask is
>   real pixels there; the shipped and PIE players have no asset database on the evaluation path
>   — the same reason per-layer terrain *textures* never reached the player's projection — so
>   `NoMasks` gives them a `0 × 0` mask that scores `0`. The divergence **fails closed** (less
>   content, never wrong content) and is diagnosed on the node, but nothing downstream can tell
>   "masked out" from "authored empty" — so the **cook now advises** on any `.inf_pcg` in the
>   closure that uses the node, which is the last moment before it becomes a shipped build.
>   Closing it properly needs a texture map through the builder, a `LevelSource` lookup, a
>   **seventh** `ScenePayload` list, *and* a new cook edge (`.inf_pcg` → texture, which does not exist today,
>   so the texture is not even packed) — a chain, not a patch, and deliberately not taken in a
>   batch that had already grown a schema-free surface this wide.
> * **A rule still lowers to a one-entry kind palette.** `PcgDocument` models weighted `PcgKind`
>   lists and always has; the graph still says `mesh` + `weight` on the scatter node. A `kind`
>   node plus a third merge would close it, and it was left out because the deliverable was
>   layers × rules and three merge nodes start to look like a pattern nobody asked for.
> * **Evaluation still runs once, at load.** Inherited from P10.6 and unchanged: a streamed
>   terrain that pages in a region after load does not re-evaluate its binding for it. The binding
>   is regional by construction (`evaluate` takes a `Region`), so the fix is a caller, not a
>   redesign.
> * **The coarse pages' ring under-counts its data-map sums** (above). Bounded, stated, and only
>   fixable by widening the reduction's input beyond one block.
> * **The bless-guard asymmetry is *still* there.** Third batch running. `INF_BLESS_FIXTURES`
>   means any value in the editor codec and exactly `"1"` in `inf-scene`. Always use `=1`.

> **STATUS — P19.4 PCG grammar core: COMPLETE (2026-08-02).** Scatter answers *how many of
> these, over this area*. A grammar answers *what sequence of pieces goes along this line* — and
> that is the difference between a meadow and a fence, a colonnade, a wall. A rule text expands
> along a spline's arc length or a footprint's edges into placed modular-mesh instances, under
> the same counter-hash determinism as scatter, on the same GPU-instanced path, through the same
> cook.
>
> **THE SCHEMA ANSWER: NO — nothing bumps. `.inf_pcg` stays v2, scene stays v16, no MIRROR
> moved.** Three statements, each load-bearing:
> * **A grammar pass is not in `PcgDocument`, deliberately.** That type is the frozen v2 wire and
>   bincode is positional, so growing it by one field makes every committed `.inf_pcg` fail to
>   *decode* — a real schema bump with a frozen record, bought for a value nothing needs on disk.
>   Since P19.3 the **authored graph JSON is the source of truth** and every evaluation site
>   re-lowers it (the player included; the stored `document` is explicitly "a convenience
>   mirror"). So `lower_graph` now returns **two** things — the document and
>   `LoweredPcg::grammars` — and the serialized surface is byte-for-byte what it was. Pinned:
>   the payload of a grammar-only graph encodes identically to an empty one-layer document.
> * **Grammar instances ARE `ScatteredInstance`s.** They ride `PcgVolume.evaluated`, which means
>   P18.5's GPU instancing, the draw-distance cull, picking and both projector MIRRORs consumed
>   the feature for free. Zero projection change was the design target and it was met.
> * **No new `PortType` variant.** The two new wires (`SPAN`, `RULES`) are `Named`, like
>   `density`/`scatter`/`layer` before them. The substrate is domain-free on purpose; a real
>   variant would have to be threaded through the blueprint, material and state-machine editors'
>   type tables, colour maps and hand-written TS mirrors to buy a wire nothing else can see.
>   The **one** `inf-graph` change is `UiHint::Multiline` (+ `ParamDef::multiline`), and it is
>   free by inspection: `UiHint` serializes as its *variant name*, appears only on a `NodeDef`
>   the backend serves fresh each session, and is persisted nowhere — a graph stores
>   `ParamValue`s, which are untouched. None of the append-only discriminant reasoning applies,
>   and that argument now lives on the enum.
>
> **THE EXACT-FILL ALGORITHM, stated as an identity rather than a tolerance.** The requirement is
> that slot intervals **partition the span exactly**, because "to within an epsilon" is a
> tolerance somebody later widens. Fixed sizes are authored; flexible ones (`Gap[0.5..1.5m]`) are
> drawn from the counter hash and water-filled toward the span length. Then the desired lengths
> are **apportioned into integers**: `LAYOUT_UNITS = 2^40` ticks by largest-remainder (Hamilton),
> `q_i = floor(d_i/Σd · UNITS)`, leftovers to the biggest fractional remainders, ties by
> ascending index. **The tick sum is exactly `UNITS` — an integer identity, not a float
> comparison** (the degenerate all-zero-weight branch has to hand its own remainder to the last
> slot to keep that true: `UNITS / n · n` undershoots for every `n` that is not a power of two,
> and `n = 3/5/7/100` are now enumerated cases). A boundary is then `b_k = L · (T_k / UNITS)`
> from the *prefix* of ticks: `UNITS` is a power of two and `T_k ≤ 2^40 < 2^53`, so the fraction
> is an exact `f64`. Every interior boundary is **one multiply off the span length, never a
> running sum** — the P17.4 exact-linear lesson (derive the k-th value from k; do not add k
> times). **The exactness itself comes from the ENDPOINTS being assigned rather than computed**
> — `b_0` is the literal `0.0` and `b_n` the literal `span_len`; the apportionment's job is to
> place the *interior* boundaries proportionally and monotonically, and no float identity is
> relied on for the ends. The price is priced, not hidden: a fixed size is honoured to
> within one tick, `L/2^40` ≈ **0.9 nanometres on a kilometre span**. Asserted on `to_bits()`
> over a corpus and as a proptest over arbitrary legal grammars × arbitrary span lengths.
>
> **Overflow truncates; underflow tails.** If the minimum lengths exceed the span, slots are
> dropped **from the end** — a wall that cannot fit its last post is short a post, never squeezed,
> because squeezing silently shrinks authored module sizes. If even the maxima under-fill, the
> slack folds into the final interval as a **tail**; a module is anchored at its interval's
> *start*, so an oversized last interval is invisible and the span simply ends with empty space,
> which is the honest result of a greedy fill.
>
> **The DSL, and the three decisions inside it.** Statements end at a `;` **or a newline**, so
> the ordinary one-per-line layout needs no punctuation; `#` and `//` comment to end of line.
> `module X = mesh <guid> offset x,y,z rot x,y,z scale s size m` declares the palette;
> `Sym -> A B* C | D@0.5` declares a rule; `*` `+` `?` `{n}` repeat, `[2m]` / `[1..3m]` size,
> `@w` weights an alternative, `( … | … )` groups. Then:
> * **A terminal with no module is a gap** — it consumes its size and places nothing. That is the
>   `Gap[0.5..1.5m]` idiom, so a spacer needs no palette entry. Because a typo would otherwise
>   vanish, `Grammar::gaps()` lists every such symbol and the lowerer emits **one node-anchored
>   warning naming them all**.
> * **Sizes are mandatory on terminals and forbidden on rules.** A layout cannot be computed
>   without one, and defaulting would build a wall of 1 m panels nobody asked for; a rule's length
>   is whatever it rewrites to, so a size there is a misunderstanding, not a no-op.
> * **v1 rejects recursion, by construction.** The rule reference graph must be acyclic and a
>   cycle is a parse error naming the loop — tested in **fifteen shapes** (through groups, later
>   elements, non-first and weighted alternatives, every repetition operator, indirect chains,
>   and rules unreachable from the axiom) with acyclic twins beside them, so a future `refs_of`
>   fast path cannot reintroduce non-termination green. Self-similarity is expressed with
>   repetition, which the span bounds. (The cycle check is an explicit-stack DFS: a 2000-deep
>   chain must not blow the native stack.)
>
> **THREE NATIVE-STACK GUARDS, and the one that was missing.** "Terminates by construction" was
> true of the *grammar* and false of the *text*: `parse_alternatives` → `parse_alternative` →
> `parse_element` → group → `parse_alternatives` is native recursion driven straight by authored
> characters, so ~10 KB of `(((…` aborted the process — uncatchably, and in the editor, the cook
> **and** the shipped player alike, since every one of them re-lowers a stored `graph_json`. The
> acyclic-rule check that the termination claim rests on sits one layer *downstream* and never
> saw the input. So the parser now counts its own nesting (`MAX_NESTING = 64`, far past anything
> readable) and returns an ordinary anchored error; a depth-5000 input, balanced and unbalanced,
> is a direct test, and the never-panics property grew a deep-nesting arm past its 400-char cap
> with a long-**flat** twin so the cap cannot pass by rejecting size. The other two guards are
> stated beside it because neither subsumes the others: `MAX_DEPTH = 128` bounds the
> *derivation's* recursion, which a long **acyclic** chain (`R0 -> R1 -> … -> R2000`, perfectly
> legal) drives once per link; and `MAX_SLOTS` bounds the output. Each has its own test.
>
> **And a fourth hazard the same reading found: the nominal-length walk was exponential.** A
> rule's nominal length needs its references', so `A -> B B`, `B -> C C`, … — thirty readable
> lines — is 2³⁰ walks by recursive descent, i.e. a hang in the cook and in the player on
> content that parses and validates cleanly. `rule_nominals` now evaluates every rule **once,
> bottom-up, on an explicit stack**, and the derivation reads a table. Linear, and the diamond is
> a test.
>
> **THE PORTABILITY ANSWER: the whole grammar path is bit-portable, and one short-circuit was
> the price.** The span math is `inf_math::spline` — Catmull-Rom and arc length in `+ − × ÷` and
> `sqrt`, all exactly specified by IEEE-754, no `std` trig anywhere. Orientation deliberately
> avoids `atan2`: `Frame::rotation` builds the yaw quaternion from the half-angle identity
> `cos(θ/2) = √((1+cosθ)/2)` off the unit tangent's own components (unit by construction, since
> `s² + c² = (1−cosθ)/2 + (1+cosθ)/2`). The only trigonometry is a module's authored euler
> rotation, which goes through `inf_math::portable::psin64/pcos64` — the P14 law applied to
> committed placement data. **And that surfaced a real bug the law did not predict:** `pcos64(0)`
> is the polynomial's `0.999_999_943_741_051_1` — short of `1` by **5.63e-8** — so an unrotated
> module carried a residual tilt. A zero angle now short-circuits to the exact identity and each
> axis quaternion is normalized. The shortfall is *asserted* rather than described, so the prose
> cannot drift from the polynomial.
>
> **The span IS the arc-length LUT, not a second opinion of it.** Arc length on a cubic has no
> closed form, so *every* implementation samples. Making the sampled polyline the domain — rather
> than sampling to build a table and then evaluating the curve again — means the length a slot is
> measured against and the position it is placed at come from the **same points**. Pinned:
> `Span::from_spline` uses `arc_length_lut`'s own `t` sequence and its length matches
> `lut_length` **bit for bit**, for both interpolations, open and closed.
>
> **The merge shape: `grammar.expand` outputs a SCATTER.** It joins the P19.3 `scatter.merge` and
> `layer.layer` chains untouched — no third merge node (P19.3's own remainder warned that three
> would start to look like a pattern nobody asked for), no third sink input, and a grammar
> inherits its layer's name and `enabled` flag for free, which is what makes a disabled layer
> disable its grammars. What it lowers *to* is not a `PcgRule`, so it appends to
> `LoweredPcg::grammars` and contributes an empty rule tail. Four nodes: `grammar.rules` (the
> rule text as a **multiline node param** — the node is its editor, there is no rule-text panel),
> `grammar.spline`, `grammar.footprint`, `grammar.expand`.
>
> **The corner rule, v1, stated.** A perimeter is **four independent edge spans**, not one loop:
> `Wall -> Post Panel* Post` should put a post at each end of each side, not run one expansion
> around the ring and land its only two posts on an arbitrary corner. `corner_size` insets every
> edge by half of it at both ends, and a named `corner` module is stamped on each vertex facing
> along the *outgoing* edge. `corner_size = 0` runs the edges corner to corner; an edge too short
> to host its own insets degenerates rather than folding back on itself. Perimeter conservation
> (4 edges + 4 corners = the rectangle) is asserted.
>
> **`grammar.spline` has no preview/shipping divergence, and that is the contrast worth drawing.**
> P19.3 shipped `mask.image` knowing the editor resolves textures and the player cannot, and had
> to buy a cook advisory for it. A `Spline` is a **persisted scene component in both codecs**, so
> the editor, PIE and a shipped build all resolve it from the world they already built. The
> `SplineSource` seam has the same shape as `MaskSource` and none of its asymmetry. A blank
> entity ref means *the evaluating entity's own spline*, so a spline and a volume on one actor
> need no GUID typed anywhere.
>
> **The cook grew one edge and one advisory.** `.inf_pcg` → each grammar module's `.inf_mesh`,
> so an explicit-roots cook of just a level ships the pieces its walls are made of; a module
> naming a mesh the project does not have is a **deduplicated, sorted warning** naming both,
> because a grammar fails *quietly* — the derivation runs, the slot consumes its span, and the
> wall simply has a piece missing. **The edge is read from the `grammar.rules` nodes, NOT from
> the lowered passes**, and that distinction is load-bearing: lowering has five ways to give up
> before a pass exists and every one of them is an ordinary mid-authoring state — most obviously
> a Span pin nobody has dragged yet — so a pass-driven edge would ship a wall missing its pieces
> *and* stay silent about the dangling one. It is over-inclusive in exchange (an unwired rules
> node still declares its meshes), which is the right asymmetry: bytes, not a hole in a wall.
> Gated by a fixture whose Span pin is deliberately unconnected. **The edge is grammar-only, deliberately:** a scatter rule's
> `PcgKind.mesh` is an older hole with a different shape (blank-tolerant since P10.5, thousands
> per document, still drawn as a placeholder cube), and closing it would change what every
> existing project packs for bytes nothing reads. Stated as a remainder rather than smuggled in.
>
> **Gates.** `runtime/inf-player/tests/grammar_pcg.rs` builds a volume whose `.inf_pcg` carries
> **three passes on one canvas** — a scatter rule, a spline grammar following a real `Spline`
> entity, and a footprint perimeter grammar around the volume's own box, merged through
> `scatter.merge` — so P19.3's multi-pass lowering carries a P19.4 generator through the shipping
> pipeline rather than only its unit tests. It proves: the cook follows level → `PcgVolume.graph`
> → `.inf_pcg` → module `.inf_mesh` with only the level as an explicit root; the dangling module
> is advised by name; **cooked == uncooked** and **PIE == shipping**, each arm instance for
> instance *and* **bit for bit** (`to_bits()` on every position, rotation and scale — the PIE arm
> to the same standard as its sibling, since a wall three nanometres out on one host is still a
> wall that moved when you shipped it); the structure is real
> (instances on the spline's own line, on the footprint's perimeter, exactly on the terrain);
> two loads of one pack agree; **the shipped content's own passes are invariant at 1/2/4/8
> workers** through `evaluate_grammars_in`, read back out of the pack; and a graph with no
> grammar still evaluates to exactly its scatter; and the cook's module edge survives a graph
> that does **not** lower, with a grammar-free negative control so the advisory is evidence
> rather than noise. `inf-editor-core`'s new `grammar_span_mirror.rs` compares the editor's and
> the player's spline fetch **character for character** (whitespace-collapsed), because that
> function — the ECS walk — is the only part of the grammar path each host writes for itself;
> everything downstream is one Ring-0 `evaluate_grammars`. **It is a source-text gate, not a
> behavioural one** — the same technique and the same admission as `projector_mirror.rs` and
> `biome_binding_mirror.rs`, because Ring 1 cannot link the `inf-studio` binary; what proves the
> two hosts actually agree is the PIE == shipping arm above.
>
> **Tests.** `inf-pcg::grammar::dsl` (the worked example; newline-or-semicolon termination;
> comments; fixed/flexible/unit-suffixed sizes; every repeat and group; epsilon productions;
> canonical-text round trip; **a 30-case error corpus asserting the message and BOTH anchor
> coordinates** — exact columns, including indented statements, because an `err.col >= 1`
> assertion on a `u32` cannot fail and the caret position is the product; **the parser's own
> depth cap** (at the cap, past it, 5000 balanced and unbalanced, nested inside an alternative,
> and a 20 000-element flat sequence that must still parse); **fifteen cycle shapes with acyclic
> twins**; exact GUID lexing; the 2000-deep chain; negative and exponent numbers);
> `::grammar::span`
> (exact endpoints via the `a(1−t)+bt` form; the LUT bit-identity; arc length against an analytic
> circle; NaN-free degenerates; closed wrap; zero-length segments borrowing a direction; the yaw
> against `from_rotation_y` over 720 angles plus the antiparallel case; portable euler against
> glam's `YXZ`; perimeter order/corners/insets/conservation; row centring and axes; the sample
> cap); `::grammar::expand` (**the exact-fill identity over a corpus of grammars × lengths ×
> seeds**; fixed sizes surviving quantization; apportionment exactness for degenerate weight
> vectors including all-zero, NaN, negative and sub-tick, with the even-split check at counts
> that do not divide 2⁴⁰; **the diamond rule graph that would be 2³⁰ walks without the nominal
> table**; **a 4000-link chain capped rather than overflowing**; a known grammar's **exact
> expected sequence**;
> truncation from the end; every repeat operator; weighted alternatives; gaps consuming span;
> flexible slots absorbing the remainder in range; purity in the seed; the slot cap; a
> zero-size `*` not looping; anchoring and the slot frame; ground modes; corner stamping;
> footprint-from-extent; spline self-reference; **pool-size invariance**; cross-run bit
> identity); `inf-pcg/tests/grammar_dsl.rs` (**properties**: any legal grammar round-trips
> through its own text and printing is idempotent; arbitrary text and DSL-alphabet text never
> panic the parser and every rejection is 1-based-anchored; **deeply nested text errors instead
> of aborting, at depths up to 6000 and under three different prefixes, with a long-flat twin**;
> any layout partitions its span exactly and every slot stays in range); `inf-pcg::graph` (the
> kit's wires and the multiline
> hint; the shipped default rule text *parsing*; the pass carrying every authored param beside an
> **empty** document; spline and rows lowering; **the parse error anchored on the rules node with
> the DSL's own `line:col`**; missing rules/span; wrong node types; the gaps warning; an
> undeclared corner; an unknown axiom; merging into layers and inheriting the toggle; canvas
> order; the payload round trip re-lowering to identical passes; **`grammar_mesh_refs` reading
> the palette rather than the lowered passes** — wired, half-wired, unwired, deduplicated across
> nodes, sorted, and silent for a grammar-free graph). Frontend:
> `pcgPinTheme.test.ts` pins every wire the Rust registry declares to a distinct colour — the one
> place a new backend wire silently renders grey.
> **42 goldens byte-identical** — a grammar is instances, not pixels.
>
> **Remainders, stated (P19.4).**
> * **A flexible slot changes SPACING, not mesh size.** `ScatteredInstance::scale` is one `f64`,
>   so there is no non-uniform stretch on the instancing path a grammar rides. A palette entry
>   recentres itself with `offset 0,0,<half>`; stretching to fit needs a per-instance scale
>   *vector*, i.e. an ECS field and both projector MIRRORs, and was not taken for a v1.
> * **Grammar passes are not dispatched by the biome binding.** A biome is a painted *region* and
>   a grammar needs a *span*; `BiomeBinding` is untouched and a biome's `.inf_pcg` contributes
>   only its scatter. The natural closure is P19.5's footprints-from-a-region, not a patch here.
> * **Repeats resolve greedily left to right**, each reserving the *nominal* length of everything
>   after it (a `*` reserves nothing, an optional reserves its full body — over-reserving leaves a
>   tail, which is gentler than the truncation under-reserving causes). Two `*` in one sequence
>   means the first wins. Documented on `Deriver`, not discovered.
> * **The scatter half of the cook's mesh edge is still open** (above), as is a `PcgKind.mesh`
>   dangling advisory.
> * **A document-only (v1) `.inf_pcg` carries no grammar** — the passes live in the authored
>   graph. Same shape as a v1 payload having no image mask and no merge tree.
> * **Evaluation still runs once, at load.** Inherited from P10.6 and unchanged by this batch.
> * **The viewport still draws every scattered instance as a placeholder cube**, grammar modules
>   included. Kind→real-mesh upload is the same documented gap sprites and tilemaps have; it is
>   why a module's mesh GUID matters to the cook and not yet to the renderer.

> **STATUS — P19.5 Building & interior grammar + THE PHASE 19 GATE: COMPLETE (2026-08-02).**
> The headline requirement: a footprint becomes a floor stack, the floors become rooms, the
> rooms become walls with real openings, and **you can walk in**. Seven archetypes — office,
> apartment, industrial, house, estate, hotel, shop — as code-shipped primitive palettes, a
> `building.archetype → building.plan` node family on the SCATTER wire, `collider` as the
> grammar DSL's one addition, `PcgVolume::structures` as derived (unserialized) solid state that
> the 3-D physics bridge turns into static box colliders, and `samples/phase19-town` +
> `phase19_gate.rs` composing biomes × grammar × buildings on a partitioned world.
>
> **The full argument — the dimensional split, the enterability invariant, the connectivity
> proof, the collider decision, the streaming finding, the schema answer (nothing bumps), the
> gate inventory and the deferred ledger for all five batches — is the Phase 19 status block at
> the top of this section**, because it is the phase's record as much as the batch's. Files:
> `crates/inf-pcg/src/building/{mod,palettes,partition,plan,assemble,pass}.rs` (the layer),
> `grammar/dsl.rs` + `grammar/expand.rs` (the `collider` attribute and `GrammarOutput`),
> `graph.rs` (the node family + lowering), `crates/inf-ecs/src/components.rs`
> (`ScatteredSolid` + `PcgVolume::structures`), `crates/inf-physics/src/d3/ecs.rs`
> (`pcg_structure_snaps`), both evaluation sites, `samples.rs`, and the frontend pin theme.

- **P19.1 Erosion data maps** — 1. accumulate flow / deposition / wear maps in the erosion
  passes (CPU reference + WGSL mirror, parity-gated exactly like heights); 2. persist them
  per-tile and sparse on the format-aware serde pattern; 3. export as mask images (Gaea-style
  data maps).
- **P19.2 Biome painting** — 1. a `BiomeSet` asset — named biomes with colour, splat mapping,
  PCG graph ref, water/structure hints; 2. per-sample biome ids on tiles, sparse, mirroring the
  splat weights; 3. a paint tool cloning the splat seam (`BiomeDelta` beside `SplatDelta`,
  `EditCommand::PaintBiome`, toolbar controls + a biome-overlay view mode).
- **P19.3 Biome → PCG binding & node-kit completion** — 1. evaluation dispatches per-biome
  graphs with feathered border blending; 2. the deferred `mask.image` node and multi-rule
  lowering; 3. data-map samplers (`mask.flow`, `mask.wear`, …) over P19.1.
- **P19.4 PCG grammar core** — 1. a deterministic rule-rewriting grammar in `inf-pcg`,
  counter-hashed like scatter; 2. token rules expanding along splines and footprint volumes
  into placed modular-mesh instances (fences, walls, façades); 3. a grammar node namespace in
  `.inf_pcg` + a rule text editor; 4. transpile/PIE parity.
- **P19.5 Building & interior grammar** — the enterable-buildings requirement: 1. footprint →
  floor stack → room partitioning by an interior grammar; 2. staircase/connector placement,
  door and window cutting; 3. per-room-type furniture population (scatter with wall-align and
  clearance placement rules); 4. building-type palettes — **office, apartment, industrial
  (factory/warehouse), house, estate, hotel, shop** — shipped as sample module sets
  (primitive/procedural + CC0); 5. interiors respect the P16 streaming cells. Phase 23's DCC
  later makes palette authoring in-engine.

### Phase 20 — Water & hydrology

**Goal:** realistic lakes, rivers, and oceans. **Done when:** a coastal scene — ocean plus a
spline river fed by a lake — carries buoyant physics objects, replays deterministically on the
physics/audio trace, holds water goldens, and PIE == shipping.

> **STATUS — P20.1 Water surfaces: COMPLETE (2026-08-02).**
> Oceans, lakes and spline rivers, on **one** wave model, **one** shader and **one** pass.
> A lake is a bounded ocean with small numbers; a river is the same evaluator run in the
> river's own `(arc length, lateral)` frame so its ripple travels *downstream* rather than
> with the wind. Schema **v17** (one appended entity slot). All 42 pre-P20.1 goldens
> byte-identical; three new ones blessed once.
>
> **The wave model, and where it is evaluated.** `crates/inf-water` (Ring 0, new) derives the
> Gerstner components — direction, wavenumber `k = 2π/λ`, amplitude, `ω = √(g k)` (deep-water
> gravity dispersion, so `T = √(2πλ/g)` and long swells outrun short chop), steepness `Q` and
> phase — as a **pure function of `(seed, wind)`** through an integer SplitMix64 hash, in
> bit-portable `f64`, with every trig call going through `inf_math::portable::psin64/pcos64`
> (the P14 LAW). Amplitudes are renormalized so `Σ Aᵢ` is exactly the authored bound, and `Qᵢ`
> is solved so `Σ Qᵢ Aᵢ kᵢ` is exactly the authored steepness — which is what makes
> `|height − level| ≤ amplitude_m` and "the trochoid never self-intersects" *properties*
> rather than hopes. Wind response: direction from the wind vector (never `atan2` — not
> bit-portable), amplitude from a monotone gain saturating at 12 m/s with a 0.25 floor,
> because an ocean whose wind has just dropped is swell, not glass.
>
> **The CPU derives, the GPU only evaluates.** The parameters are uploaded already solved, so
> there is no second, `f32`, platform-dependent copy of the wave model in WGSL beside the one
> P20.2's buoyancy will sample — the terrain-parity class of drift, avoided by not creating it.
> Two consequences fall out. **Time never reaches the GPU**: a wave arrives with its phase
> already reduced (`wrap(φ − ωt)`) in `f64`, the `CloudParams::wind_offset` trick, so a level
> clock in the millions of seconds does not quantise into visible steps. And the **floating
> origin rides in the same reduction** (`+ k·(d·origin_xz)`), so the shader evaluates at
> render-local coordinates, gets the world-space phase, and a rebase moves no wave —
> `a_rebase_moves_no_wave` pins it.
>
> **The height query is designed for the fixed step, now.** `WaterSurface::height_at(p, t)` is
> pure `f64`, allocation-free, camera-free and frame-free; `t` is the *document's* clock
> (`ResolvedSky::cloud_time_s`). A Gerstner surface is parametric, so "how high is the water at
> my boat" is an **inverse** problem: it is solved by a fixed 6-iteration fixed point
> (`pₙ₊₁ = p − Δxz(pₙ)`), never a convergence test, so the operation count — and the answer —
> is identical on every machine. Measured residual ≤ 3 mm on the steepest sea the tests author;
> the honest worst-case bound is ~17 cm and quoting it would overstate the error by two orders
> of magnitude, so both numbers are written down. `submersion_m` and `flow_at` are the other
> two P20.2 seams. Ocean/lake current is **zero**, and that is a decision: a Gerstner orbit
> averages to no net transport, and reporting the instantaneous orbital velocity as a current
> would push a boat across the sea at wave speed. Stokes drift is P20.2's question.
>
> **Rivers.** `RiverPath` samples the P19.4 arc-length machinery at even **distance** (never at
> even spline `t`, which bunches on curves), giving frames with a centre, a flow tangent, a
> horizontal across-vector and the width/depth profile interpolated along arc length. The
> across-vector is `normalize(tangent × up)` recomputed per frame, **not** parallel-transported:
> transport accumulates roll and a closed loop generally does not return to the frame it
> started from (the curve's holonomy), which reads as a ribbon visibly banked at its own seam.
> The price is that a river cannot bank — which water does not do — and the purchase is exact
> continuity on closed splines, asserted by `a_closed_river_closes_exactly`. A hairpin keeps
> its frames (`a_hairpin_keeps_its_frames`).
>
> **The river's centreline is the `Spline` on the SAME ENTITY**, not a reference. Composition,
> the way `Terrain` and `Transform` already relate: it cannot dangle, it needs no cook edge and
> no dangling-reference advisory, and "select the river, drag its points" is the obvious
> gesture. **So the cook's dependency closure is unchanged** — stated because the batch brief
> asked. The cost is that a river cannot share a centreline with a road; an `EntityRef` field
> is the additive change if that is ever wanted, and it would be the thing that introduced a
> reference to keep alive.
>
> **The reflection source is the sky-view LUT, not SSR — and that is a decision, not a
> shortcut.** A wave-perturbed normal at the grazing angles that dominate a water surface
> reflects *toward the horizon*, which is exactly where a screen-space march has nothing to
> hit: the ray leaves the frame in a few steps and essentially every pixel takes the miss path.
> The miss path IS the sky, so v1 asks the sky directly — one fetch, no march, no per-pixel
> failure mode, and the same authority the sky pass samples, so water and sky agree by
> construction. With the atmosphere off it falls back to the authored three-colour gradient, so
> a level with no clock still gets a plausible reflection. Reflecting *scene* geometry (a boat,
> a cliff) needs the P18.4 SSR machinery running after the opaque resolve; named as the P20.3
> follow-up.
>
> **The pass sits after opaques, after clouds and rain, before translucency** — the `CloudNode`
> placement argument applied to a surface rather than a medium. After opaque because every
> interesting thing water does reads *what is behind it*; after clouds so a sea reflects the sky
> it is under; before translucency so glass composites over water like any other surface. It is
> **not inside** the translucent pass: that pass is a back-to-front sort of `MeshInstance`s
> through the `mesh` shader, and water is one procedural surface per body with its own vertex
> generation and a fragment stage that reads the frame buffer. Depth is bound **read-only**
> (`depth_ops: None`) with the same view additionally bound as
> `texture_depth_multisampled_2d` — the P17.3 arrangement, and the only one WebGPU permits.
>
> **Screen-space refraction costs one extra resolve, and only when it is used.** The colour
> behind the water is the MSAA target being rendered into, which cannot be sampled while it is
> an attachment — so the node first records a **resolve-only render pass** (`color_msaa` in,
> `scene_hdr` out, zero draws; wgpu resolves at pass end regardless). `scene_hdr` is
> overwritten by the real `ResolveNode` later, so nothing downstream sees it. Skipped entirely
> at `WaterQuality::Low`.
>
> **Geometry.** One index-buffer-only grid (no vertex buffer — the position comes from
> `vertex_index`), mapped three ways in the vertex stage: an ocean's **graded** (`sign(q)·q²`),
> camera-following, 4 m-**snapped** 8 km patch (a uniform 8 km grid at 64 quads would put three
> wavelengths in a cell — the waves would not exist; snapping makes the vertex set
> piecewise-constant in camera position so the tessellation stops crawling); a lake's uniform
> rectangle; and a river's ribbon interpolated across a storage buffer of frames. Per-body
> uniforms ride one buffer at a 512-byte dynamic-offset stride (the uniform is 448 bytes —
> `the_uniform_matches_the_shader_struct` pins that it cannot outgrow its slot and silently
> overlap the next body).
>
> **Shading**: Fresnel with `F0 = 0.02` (water's IOR 1.333, derived not tuned) → sky;
> Beer-Lambert absorption `exp(−σ·d)` over the screen-space water column with the default `σ`
> at the clear-water ratio (red absorbed ~13× faster than blue), which is why deep water is blue
> without anyone painting it blue; a smoothstep shore fade on the same depth difference, so it
> works against terrain, a jetty and a boat hull in one expression; and foam from **three**
> sources combined by `max` (foam is a coverage fraction, and two causes do not make more than
> white): the wave **crest factor** — `Σ Q A k sin θ`, the surface-folding measure the Gerstner
> model already contains, so "how close to breaking" rather than a dial with no referent — the
> shallow-water band, and river flow speed (plus a bank term).
>
> **Shore is computed twice, deliberately, and it is written down.** In the shader it is a
> screen-space depth difference; on the CPU (`inf_water::shore`) it is a world-space question
> about terrain heights, for the cook, gameplay, P20.2 and P20.4. Neither is derived from the
> other. The CPU `shore_distance` is honestly labelled **SDF-class**, not an SDF: a 16-direction
> bounded radial probe with fixed bisection — never under-reports, exact for a straight shore
> probed head-on, error bounded by the angular step on a concave one.
>
> **Schema v17** — the five-step dance, both codec mirrors. One appended entity slot
> (`water_body`), so this is the `EntityRecordV10` *shape* of bump, not the `EntityRecordV14`
> one: `EntityRecordV16` / `SceneFileV16` frozen in both codecs with `into_current` +
> `from_current`, the v15 rung repointed at v16, a `17 =>` decode arm, and `scene_v16.inf_lvl`
> blessed in **both** fixture dirs and byte-compared (`v16_fixture_matches_the_runtime_codecs_copy`).
> The v16 fixture carries what only v16 could express — a painted biome id and a `biome_set` —
> so the v17 hop is proven to *preserve* v16 content rather than merely to produce defaults.
> **Priced:** the slot costs exactly **one discriminant byte per entity**
> (`v17_costs_one_byte_per_water_free_entity`, measured as a delta against the frozen v16 shape
> of the same record). The 12 samples/templates re-blessed accordingly, and every delta is
> exactly its entity count: character-demo +4, streamed-terrain +4, terrain-demo +4,
> first-person +4, platformer +5, hybrid +5, partitioned-world +20, phase16-world +22,
> phase19-town +25, physics-playground +28, phase18-scatter +31, vgeom-demo +325. The
> **wire-enum law** applies to the first new scene enum since P19.2 wrote it down:
> `water_kind_discriminants_are_frozen` pins Ocean=0 / Lake=1 / River=2 and their string ids.
> The two-ladder tile tripwire moves to 17 with the note that v17 is the *scene-only* case.
>
> **Dependency ledger.** `inf-render` gained `inf-water` (Ring 0 → Ring 0) — it consumes the
> *derived* `Wave`/`WaveField`/`RiverFrame`, never re-deriving one, which is what keeps the
> wave model single-sourced. `inf-packager` gained `inf-water` + `glam` outright and
> **promoted `inf-ecs` and `inf-math` from dev-dependencies to real ones**: the river advisory
> reads `WaterBody`/`Spline`/`Transform` off a decoded level and maps the ECS spline
> interpolation onto `inf_math::spline`'s. Both former dev-dep comments were rewritten in place
> rather than deleted, so the reason each crate is there is still readable. No new third-party
> crate enters the tree, and `cargo deny` is unmoved.
>
> **The component is flat, and that is a deviation with a reason.** The brief asked for an
> enum-shaped `WaterBody`; it ships as a flat `WaterKind` enum + flat fields, exactly like
> `Light`/`LightKind` and `Volume`/`VolumeKind`. Two concrete reasons: the Details reflection
> walker surfaces flat scalars and enum *dropdowns*, so struct variants would have no widget at
> all; and a variant-dependent field set makes a bincode record's length depend on its variant,
> which is a worse wire contract than one record of `serde(default)` fields.
>
> **The MIRRORs.** `project_water` is byte-identical in both projectors and compared
> character-for-character (`project_water_is_identical_in_both_projectors`), like
> `project_sky`. The two things that *could* silently diverge are in Ring 0 instead:
> `inf_ecs::sky::water_environment` decides what clock and wind a body sees, and
> `WaterBody::effective_wind` decides whether *this* body follows them — a lake sets
> `wind_from_weather: false` because a lake has no fetch, and a gale must not raise a swell on
> it. The PIE-vs-shipping gate proves the pairing: over a 120-step storm blend the ocean's
> derived components change every step while the lake's stay **bit-identical**.
>
> **The downhill advisory** reads the spline's own arc-length elevation profile through the
> same `RiverPath` the renderer and the sim build. **No terrain is queried on this path**, so
> the 0.5 m **merged-span** tolerance is absorbing Catmull-Rom overshoot between knots plus
> arc-length resampling — centimetres — not heightfield noise; the constant's doc says exactly
> that, because a justification describing a measurement the code never performs is worse than
> none. It composes the parent chain, honours a **negative** `river_flow_m_s` by reading the
> profile backwards, and skips closed loops (a loop cannot help regaining what it loses —
> `a_closed_river_is_never_reported`).
>
> **THE ORDERING LAW, paid for in the P20.1 audit.** The advisory runs **before** the
> partition branch, because partitioning MOVES a level's entities into the derived `.inf_part`
> and clears `level.entities` in place. The first version ran after it, and therefore reported
> **nothing at all** on partitioned levels — silently, and on exactly the level type most likely
> to hold a kilometre of river. It also had to become `advisories.extend(notes)` rather than
> `advisories = notes`, or the partition's own advisories would have clobbered the water one.
> *Every future per-entity advisory on this path belongs above that branch.* Pinned from the
> outside by `an_uphill_river_is_reported_on_a_partitioned_level_too` and
> `the_partition_advisories_survive_the_water_one`, both verified to fail against the old
> ordering.
>
> **Goldens: 42 → 45.** `water_ocean_noon`, `water_lake_dusk`, `water_river` — one per body
> kind, because three tessellations and three shading regimes share one shader. All 42
> pre-existing PNGs are byte-identical, verified the house way (the whole suite under
> `INF_BLESS_GOLDENS=1`, `git status` reporting only the three new files) **and** pinned from
> the inside by `water_off_path_is_byte_identical`, which winds every water knob on a
> water-free scene *and* on a scene whose only body is undrawable, with an anti-vacuity guard
> that a real body does move the frame.
>
> **Remainders, stated.**
> * **An ocean is a finite 8 km patch**, so a camera on a flat horizon can see it end. A
>   projected-grid (screen-space, truly infinite) ocean is different work, named rather than
>   half-built. In practice the P17 aerial-perspective term has washed the surface into the sky
>   well before the edge.
> * **No SSR on water** (above), and therefore **no reflected geometry** — a boat reflects the
>   sky, not itself. P20.3.
> * **No underwater view.** Looking up from below draws the surface's back face with the same
>   shading as the front; underwater fog and light shafts are P20.3's whole subject.
> * **The advisory checks the water surface, not the BED.** "Is the river's water above the
>   ground under it?" is the natural companion and is not implemented: the answer lives in tile
>   payloads inside a `.inf_terrain` the cook validates structurally and never pages in.
>   `RiverPath::bed_profile` is the seam (already written and tested, including the
>   terrain-hole skip); P20.4's tools, which have the terrain resident, are where it belongs.
> * **The uphill check has a documented escape: a sawtooth.** A climb broken into sub-tolerance
>   rises by intervening falls is reported by nothing, because each merged span closes at the
>   fall. The alternative — a net-elevation test — fires on every river that crosses a ridge on
>   its way down a valley, i.e. on correct content, so this is the lesser gap. Pinned as a
>   *property* by `a_sawtooth_climb_escapes_the_per_span_tolerance` so it stays known.
> * **P19.1 flow maps are not yet an input.** The plumbing a river needs is the terrain's
>   `data_map_at(DataMapKind::Flow, ..)`, and the shader term it would drive (foam and speed
>   modulation) exists — what is missing is the projector edge from a terrain's maps to a
>   river's frames, which wants the authoring story P20.4 brings.
> * **The refraction resolve is per-frame, not per-region.** Water in 2 % of the frame still
>   pays a full-resolution MSAA resolve. A scissored resolve needs the bodies' screen bounds,
>   which the CPU does not currently compute.
> * **A river's depth profile is linear in arc length** (two ends, interpolated). A keyframed
>   profile is a `Vec<(s, width, depth)>` and needs no change beyond the interpolator.
> * **`MAX_BODIES = 32` per frame**, extra bodies skipped deterministically in projection order.
> * **No authoring tools** — a `WaterBody` is added through the Details "Add Component" menu and
>   its rows come free through reflection. Placement tools, per-biome water-level hints (the
>   P19.2 `water_hint` field is still unread) and the erosion→water pipeline are P20.4.
>
> **Two environment artifacts, both now hit twice — recorded so the third time costs nothing.**
> * **"crate `X` required to be available in rlib format, but was not found in this form" /
>   "can't find crate for `inf_player`" is a DISK-FULL symptom**, not a code failure. Seen on
>   `gimli`, `wasmtime_environ`, `wgpu_hal`, `windows_sys`, `inf_mesh`, then on our own libs.
>   The P20.1 occurrence was diagnosed properly the second time: `df` showed **5.1 GB free on a
>   1.9 TB volume**, and `target/` was holding 121 GB. rustc had been writing truncated or
>   partial artifacts, so the rlibs existed on disk but no longer matched the fingerprints
>   cargo was asking for — which is why a plain retry sometimes appears to clear it and
>   sometimes does not. **Check `df -h .` FIRST.** `cargo clean -p <the named packages>` (121 GB
>   freed here, 159 green legs immediately after) is the fix; the phase-11 law — *clean between
>   phases* — is the prevention, and this is its second citation. Do not "fix" it by editing the
>   crates named.
> * **Blessing fixtures/goldens with a multi-package `-p A -p B` invocation can produce phantom
>   churn**, because cargo unifies features across the selected set and a crate can then bless
>   under a feature set it does not ship with. **Bless one package at a time**, and always
>   confirm with `git status --porcelain` on the artifact directory rather than trusting the
>   test's exit code. The P20.1 bless-diff was run exactly that way: the whole golden suite
>   under `INF_BLESS_GOLDENS=1`, then `git status` showing only the three new PNGs.
>   (A related third: a test that *writes* a committed file can leave pure line-ending churn on
>   Windows — `git diff --numstat` reporting zero changed lines is the tell, and
>   `git checkout --` on those paths is the fix. It hit the ts-rs bindings here.)
>
> Files: `crates/inf-water/**` (new), `crates/inf-ecs/src/{components,registry,sky}.rs`,
> `crates/inf-scene/src/lib.rs` + `editor/crates/inf-editor-core/src/scene/serialize.rs` (v17),
> `crates/inf-render/src/{water.rs,passes/water.rs,shaders/water.wgsl,passes/mod.rs,scene.rs,
> settings.rs,caps.rs,renderer.rs,lib.rs}`, both projectors, `runtime/inf-packager/src/cook.rs`,
> and the gates `runtime/inf-player/tests/water_projection.rs`,
> `runtime/inf-packager/tests/cook_water.rs`,
> `editor/crates/inf-editor-core/tests/projector_mirror.rs`.

> **STATUS — P20.2 Water volumes & physics: COMPLETE (2026-08-02).**
> Water became **physical**: bodies float, water pushes back, Blueprints hear it, and a
> character swims. Schema **v18** (one appended entity slot, `buoyancy`). All 45 goldens
> byte-identical — this batch draws nothing.
>
> **The force is a pure function of `(WaterBody params, body pose, level clock)`.** It samples
> P20.1's `WaterSurface::height_at` — `f64`, allocation-free, bit-portable, camera-free,
> frame-free — from inside the fixed step, and the clock is the *document's*
> (`ResolvedSky::cloud_time_s`). That is the P16.3 sim/render split at its sharpest so far: a
> buoyant force computed from anything the renderer owns would float a boat at a different
> height in a shipped build than in the preview, and *it would still look like a sea*, so
> nothing would notice. **There is one wave model** — the Gerstner sum the shader evaluates and
> the one the buoyancy samples are the same `inf_water::WaveField`, built by the same
> `from_spec` from the same component — and `the_sim_and_the_renderer_derive_the_same_waves`
> pins it as **bits**, comparing the two projections rather than trusting two copies of source.
>
> **THE ORDERING LAW.** The pass runs strictly `sync → apply_water_forces(dt) → step`, and its
> events drain in the **collision slot**. After the sync because a body has to be sampled where
> it *is*; before the step because rapier clears force accumulators every step; in the collision
> slot because a crossing is sensed from the same pre-step poses a contact is, and two "when did
> this happen" answers in one fixed step is one too many.
>
> **TWO RAPIER LAWS, both paid for in this batch, both on `apply_force_at_point`'s doc.**
> (1) **A rapier force is persistent** — it re-applies every step until `reset_forces` clears
> it. A per-step buoyant "force" re-added each step accumulates without bound, and the first
> floating box left the atmosphere at 13 km. (2) **A force is not an impulse of `F · dt`, for
> POSITION.** rapier substeps; a front-loaded impulse conserves the velocity *exactly* and still
> drifts the position by `g · dt² · (N−1)/2N` per step — the second attempt had a neutrally
> buoyant box rising a millimetre per step with a measured velocity of `1.4e-17`. Buoyancy is a
> force; applying it as one is what makes it cancel gravity substep for substep. The pass
> therefore `reset_forces`es every buoyant body first, and that ownership is total (there is no
> `apply_force` Blueprint node, and both hosts use impulses) rather than shared.
>
> **The model, and its named errors.** Buoyancy is Archimedes: `submerged_fraction × displaced
> volume × ρ_fluid × g` against gravity, split across **four fixed sample points** on the
> shape's mid-plane at its quadrant midpoints and weighted by each point's own submersion —
> four because one point has no lever arm and therefore no righting moment, and two can only
> right about one axis. The displaced volume is `mass / density_kg_m3`, i.e. **rapier's own
> exact per-shape volume read back through the mass it already computed**, rather than a second
> hand-written volume table that could disagree with the one the solver uses (a box, a ball and
> a capsule all have closed-form volumes in rapier already). That is a deviation from the brief's
> "per-shape exact volumes, AABB fallback otherwise" with a strictly better property: there is
> no second formula to drift. The **AABB fallback survives where it is really needed** — the
> sample *layout* for a trimesh, whose vertices are reduced to their bounding box.
> Per-column submersion is **linear in depth over the shape's vertical extent**: exact for a box
> at any depth, exact for a sphere or a capsule at the symmetric half-submerged point (which is
> the equilibrium the statics tests assert), a shape factor away from exact elsewhere — written
> down because "approximate" with a named error is engineering and "approximate" without one is
> a guess. The centre of buoyancy is not computed as an offset at all: applying each sample's
> force **where it is generated** produces the righting moment as a consequence, and a raft
> rolled 25° is pushed back (`a_tilted_body_gets_a_righting_moment`), while a level one gets a
> torque of exactly zero.
>
> **Drag is LINEAR and still-water, and both halves are decisions.** Linear in the velocity
> relative to the water's *flow*, with the coefficient in **s⁻¹** so it means exactly what
> `RigidBody3D::linear_damping` means, scaled by the submerged fraction, and clamped so
> `k·dt ≤ 0.9` — a hostile authored coefficient is a mistake, not an instruction to explode the
> sim. Quadratic drag is the physically right law for a hull and is deferred: it needs a
> reference area and a shape-dependent drag coefficient, neither of which v1 has anywhere honest
> to get, and the honest consequence is that a hull's terminal speed cannot depend on its shape.
> **The water's velocity is `flow_at`, not the wave orbit** — zero for an ocean or a lake by
> P20.1's explicit decision (a Gerstner orbit averages to no net transport), the tangent flow
> for a river. Using the *orbital* velocity as the drag reference rather than as a current is
> defensible and is deferred: it needs a new `WaveField` seam and it oscillates at wave
> frequency, which is a stiff term at 60 Hz. **Angular drag needs an inertia**, and rather than
> rotate rapier's local-frame tensor into world space every step it uses the sample points'
> own second moment `m · mean(|r|²)` — isotropic, exact for a sphere, an approximation for a long
> hull, and it keeps the coefficient a plain s⁻¹ rate like the linear one.
>
> **Schema v18, and the argument for it.** `Buoyancy` ships as an **opt-in component**, and the
> no-bump alternative — buoyancy on for every `RigidBody3D`, derived from the collider's existing
> `density` — was considered and rejected for two concrete reasons. First, it changes what
> committed levels *mean*: nothing in a pre-v18 `.inf_lvl` says "this crate floats", so adding a
> lake to an existing level would silently rewrite the physics of every dynamic body in it and a
> replay recorded before the lake would diverge. Second, **`Collider3D::density` defaults to
> `1.0`, which is not a material density** — it is rapier's placeholder, and it has never
> mattered because a rigid body's fall is mass-independent. Buoyancy is the first system that
> reads it as physics: at 1 kg/m³ against water's 1000, *every* default body would float like a
> cork on a millimetre of draught, so a default-on rule would be wrong for essentially all
> existing content and the fix would be "go author a density on every collider in your level".
> The same trap is why flotation reads `Buoyancy::density_kg_m3` (default 600 — seasoned wood)
> rather than the collider's: the collider's density keeps doing its own job (it is what rapier
> turns into **mass and inertia**, i.e. how *hard* the body is to move) while the component says
> how *high* it rides. When the two agree the model is exactly Archimedes; the equilibrium
> submerged fraction is always `density / fluid_density`, stated once as
> `Buoyancy::equilibrium_fraction` so the statics tests assert against a **contract** rather than
> against the pass's own arithmetic. The bump is the `EntityRecordV10` *shape* — one appended
> slot, `EntityRecordV17`/`SceneFileV17` frozen in both codecs with `into_current` +
> `from_current`, the v16 rung repointed at v17, an `18 =>` decode arm, and `scene_v17.inf_lvl`
> blessed in **both** fixture dirs and byte-compared. The v17 fixture carries what only v17 could
> express (a spline **river** with an authored width/depth taper), so the v18 hop is proven to
> *preserve* v17 content rather than merely to produce defaults. **Priced:** exactly **one
> discriminant byte per buoyancy-free entity**, measured as a delta against the frozen v17 shape
> of the same record, and every sample's delta is exactly its entity count — character-demo +4,
> streamed-terrain +4, terrain-demo +4, first-person +4, platformer +5, hybrid +5,
> partitioned-world +20, phase16-world +22, phase19-town +25, physics-playground +28,
> phase18-scatter +31, vgeom-demo +325. The two-ladder tile tripwire moves to 18.
>
> **The spatial index, and why the town is free.** `WaterIndex` is a uniform grid over the union
> of the **bounded** bodies' XZ bounds (cell target 64 m, side capped at 48 so it is `O(1)`
> memory however far apart two lakes are authored), plus a separate list of the **unbounded**
> ones — an ocean is over every point and belongs in no cell. Its two lists are **merged** in
> ascending body index rather than concatenated, which is load-bearing: it makes
> `highest_surface_at` give the identical answer a full scan would, tie rule included
> (`inf_water::highest_surface`: topmost wins, ties to the earlier body), and a spatial structure
> that changed the answer would only be visible in a level too big to debug.
> `the_index_answers_exactly_what_a_full_scan_would` compares 14 400 query points against a
> linear scan. The index is rebuilt only when its **stamp** moves — the P19.5 change-stamp
> pattern, over `(guid, WaterBody, spline hash, world affine)` and deliberately **not** the
> clock, because a wave field is a function of the *wind* and time is an argument to `height_at`.
> A river's arc-length resample is not a 60 Hz cost. The spline is folded to 64 bits rather than
> cloned, so the steady state does not allocate per point per step.
> **But the real perf statement is one level up:** the pass iterates the *buoyant set*, not the
> rigid bodies, and that set is built from `Buoyancy` components inside `sync_from_world`'s
> **existing** entity walk — no second pass over the world. A furnished town is ~13 000 static
> colliders and zero `Buoyancy` components, so adding a lake to it costs **one `is_empty()`
> branch per step**, and `a_town_of_colliders_with_a_lake_in_it_costs_nothing` asserts that the
> only way it can honestly be asserted from outside: 3 000 colliders traced with and without the
> lake are **bit-identical**, with an anti-vacuity guard that the lake really is indexed and that
> the same body *with* a `Buoyancy` really does float.
>
> **Events.** `Enter` / `Exit` / `Splash`, on the `EventKind::Collision` precedent: three
> appended `EventKind` variants (the wire-enum law — `EventKind` is externally tagged for
> bincode, so a variant's tag is its declaration index and inserting one in the middle would turn
> every committed `.inf_act`'s `Collision` handler into a `Custom` one), three `event.water_*`
> nodes carrying `water: Int` + `speed: Float`, and handler ids `water_enter`/`water_exit`/
> `water_splash` frozen by a test. The latch is **depth hysteresis, not fraction hysteresis**:
> a floating body always has its underside wet, so "is it in the water" has to be a question
> about the *lowest point* clearing the surface (by 5 % of the body's own height, floor 1 cm) —
> a fraction test would report a bobbing cork as dry, which is how the first version failed.
> **Splash is its own event, not a float to test**, because "play a sound when something hits the
> water hard" should not have to run on every quiet entry and then branch, and because a splash
> fires on a fast *exit* too, which a threshold inside `On Enter Water` could not express at all;
> it fires *in addition to* the enter/exit it accompanies, never instead of it, at 2 m/s of
> vertical speed (the speed a body reaches after a 20 cm fall — a dropped crate splashes, a boat
> on a swell does not).
> **The audio hook is the existing `audio.*` kit called from those handlers, and no new command
> type** — that is the P12.3 doctrine working as designed: the audio stream is a pure function of
> sim state, a water crossing *is* sim state, so a `Play` queued from an `On Splash` handler
> lands in the command queue in the same deterministic order the crossings were produced in.
> An engine-authored `AudioCommand::Splash` would have meant the engine picking the sound.
>
> **Swim mode, and the asymmetry that makes it work.** A character swims at 60 % submerged and
> stops at 45 % (hysteresis, so standing at chest depth in a rippling lake does not flicker
> between two locomotion modes every step); it is pulled toward 80 % submerged — head out — at
> 4 m/s per unit of fraction error; horizontal speed is capped at 2.5 m/s. The **vertical is
> read as a rate, and downward requests are honoured at a quarter strength**
> (`SWIM_SINK_AUTHORITY`), and that asymmetry is the whole trick: the host cannot tell a
> deliberate dive from an accumulated fall, because a character controller integrates gravity
> into its own velocity every step and has no way to know the water should have stopped it. The
> first version clamped the incoming vertical symmetrically to 2 m/s and the swimmer sank at
> 1.4 m/s forever — the balance term could not out-push a clamped free fall. At a quarter
> strength the balance wins by default (a character that only integrates gravity **surfaces**)
> while a player really holding "dive" still sinks. `move_and_slide` honours it in **both** hosts
> through one Ring-0 pair (`water::swim_latch`, `water::swim_motion`), so the thresholds exist
> once and there is nothing to mirror; `the_same_character_with_no_water_just_falls` is the
> anti-vacuity twin, and it keeps its authored 6 m/s run.
>
> **Gates.** `runtime/inf-player/tests/water_physics.rs` (10): determinism, **pool-size
> invariance**, **PIE == shipping** on a cooked pack, the wave-model bit comparison, the
> off-path/anti-vacuity pair, swim + its dry twin, the event/audio determinism pair, the splash
> threshold, and the **editor↔runtime parity** case.
>
> **The pool-size leg runs in SUBPROCESSES, and the first attempt did not.** `bevy_ecs`'s
> `ComputeTaskPool` is a process-global `OnceLock`: the first `init_ecs_task_pool` wins and later
> calls are no-ops reporting the count already chosen, so an in-process "matrix" runs every leg on
> one pool and is a duplicate of the two-run determinism arm wearing a stronger name. That is
> exactly what shipped in the first cut of this batch, and the audit caught it — the same trap
> `crates/inf-runtime/src/bin/replay_probe.rs` was built to avoid for the §8 replay gate.
> `runtime/inf-player/src/bin/water_probe.rs` is its water twin: it pins the pool, runs the
> floating-stack trace over a **bare `EcsWorld`** (a `[[bin]]` cannot reach dev-dependencies, so
> the scene lives once, in the probe) and prints `threads=` / `first=` / `last=` / `trace=`; the
> test spawns it at 1/2/4/8 and compares the outputs to each other. The `threads=` line is
> asserted too, so the harness proves each process really got the pool it asked for.
> **Honestly stated, as `phase17_gate.rs` states it:** the water pass is serial today, so this leg
> is expected to pass trivially — it is asserted rather than reasoned about because the schedule
> mode is a startup choice and the first change that moves the buoyancy loop onto the pool is the
> one that would introduce an ordering dependency nobody was watching for.
>
> The parity case is paired with
> `editor/crates/inf-editor-core/tests/simulate_water.rs`: the two hosts each own a copy of the
> fixed-step ordering, of the `Host::call` match arms and of the `move_and_slide` path, so "they
> run the same water physics" is a claim about two files and is checked from the outside — by
> pinning **integers** (the step a `WaterEnter` lands on, 58; the enter/exit/splash counts,
> 2/1/3) rather than a settled `f64`, which would pin the solver's last bit on this machine and
> fail on someone else's for a reason that is not a bug.
> `crates/inf-physics/tests/water_buoyancy_3d.rs` (14) carries the statics — density 0.5 floats
> half submerged, density 2 sinks, neutral hovers to `1e-6`, a body *placed* at its equilibrium
> does not drift — plus drag direction/magnitude/clamp, the event thresholds, the swim latch, a
> floating-stack determinism run and the town. **Tolerances are sized against the P20.1
> height-query bound consciously**: every statics case authors `wave_amplitude_m = 0`, which
> makes the Gerstner displacement identically zero and the query exact, so what is being absorbed
> is the solver's own settling; the one wave case budgets `amplitude + 0.17 m`, the **honest**
> worst case rather than the ≤3 mm typical, because tuning a tolerance to a measurement instead
> of to a guarantee is how a bound stops meaning anything.
>
> **Remainders, stated.**
> * **Quadratic drag** (above) — v1 cannot make a hull's terminal speed depend on its shape.
> * **Orbital-velocity drag / Stokes drift** (above) — a body near a crest feels still water.
> * **No water→body reaction.** A boat displaces water in the model and not in the render: there
>   is no wake, no bow spray and no local depression of the surface. The surface is authored, and
>   nothing a body does moves it.
> * **The submerged fraction is a slab, not an integral.** A sphere at 20 % submerged displaces
>   less than the linear model says (the true spherical cap is `O(d²)`, not `O(d)`), so a ball
>   floats slightly high off its equilibrium and slightly low above it. Exact at the symmetric
>   point, which is where the tests assert.
> * **`Buoyancy` on a body with no collider does nothing** — there is no shape to displace with,
>   and rapier gives it no mass. Silent rather than an advisory; P20.4's tools are where an
>   authoring warning belongs.
> * **Static and kinematic bodies never float.** A `Buoyancy` on a moving platform is ignored,
>   which is right (it is script-driven by definition) and undiscoverable.
> * **A trimesh floats by its AABB.** rapier cannot give a trimesh well-defined mass properties,
>   so the path is nearly dead in practice; it is written rather than `unreachable!()`d.
> * **The swim mode has no animation.** Flipping the latch changes the motion, not the pose;
>   binding it to an `AnimStateMachine` parameter is a `water.is_swimming` node away and is not
>   in this batch (the three shipped nodes are `is_in_water`, `surface_height`,
>   `submerged_fraction`).
> * **The `water.*` queries are INSTANTANEOUS; the events are LATCHED.** `is_in_water` answers
>   from this step's raw probe, while enter/exit fire off the 5 %-of-body-height hysteresis, so
>   the two disagree inside the band — a body bobbing at the waterline can make the poll flicker
>   while no event fires. Deliberate (a poll wants the truth now, an event wants a debounced
>   edge) and documented at both the node kit and `PhysicsBridge3D::water_probe`, because the
>   natural reading is that they are the same predicate.
> * **`water.surface_height` answers `0.0` where there is no water** — the `terrain.height_at`
>   precedent, because the IR has no optional Float. `0` is a plausible sea level and is
>   deliberately not a sentinel; pair it with `submerged_fraction` when the question is really
>   "is there water here".
>
> Files: `crates/inf-physics/src/d3/{water.rs (new),ecs.rs,world.rs,mod.rs}` + `Cargo.toml`,
> `crates/inf-ecs/src/{components,registry,lib}.rs`, `crates/inf-blueprint/src/{semantics,nodekit,
> lower,raise}.rs`, `crates/inf-scene/src/lib.rs` + `editor/crates/inf-editor-core/src/scene/
> serialize.rs` (v18) + both fixture dirs, `runtime/inf-player/src/{runtime_sim,level,
> cell_stream}.rs` + `runtime/inf-player/src/bin/water_probe.rs` (new),
> `editor/crates/inf-editor-core/src/simulate.rs`, the 12 re-blessed
> samples/templates, and the gates `crates/inf-physics/tests/water_buoyancy_3d.rs`,
> `runtime/inf-player/tests/water_physics.rs`,
> `editor/crates/inf-editor-core/tests/simulate_water.rs`,
> `crates/inf-transpile/tests/water_roundtrip.rs`.

> **STATUS — P20.3 Underwater & wetness: COMPLETE (2026-08-02).**
>
> Two render-only features, **no schema bump** (Phase 20 has already spent v17 and v18, and
> neither of these needed a field): the camera inside the medium, and the band a water level
> leaves on the ground it meets. Goldens 45 → **47**.
>
> **ONE ABSORPTION STORY, SEEN FROM BOTH SIDES.** The underwater fog is the *same expression*
> `water.wgsl` already applies to whatever is behind the surface — `scene·exp(−a·column) +
> deep·(1 − exp(−a·column))` — with the camera moved to the other side of it. `a` and `deep`
> are the submerging **body's own** authored fields (`WaterBody`'s P20.1 absorption/colour), not
> a second set of underwater constants that would have to be kept in step; the
> `the_fog_absorbs_with_the_body_it_is_inside` unit test pins that byte for byte, and
> `golden_water_underwater_ocean`'s hue assertion pins it in pixels (wetness is a scalar albedo
> multiply and cannot move a hue; per-channel absorption is the only thing that can).
> **The surface caps the column**: a ray that rises through the still-water plane leaves the
> medium there, which is what keeps the bright disc overhead from being fogged by the sea floor
> two hundred metres behind it. A **downwelling** term `exp(−a·eye_depth)` dims the medium with
> depth — the same extinction applied vertically, so "deeper is darker" is a consequence of the
> absorption rather than a curve someone drew.
>
> **THE CAMERA-UNDERWATER TEST REUSES THE RING-0 EVALUATOR — it does not re-derive a surface.**
> `RenderWater::surface()` reconstructs the `inf_water::WaterSurface` the record describes (a
> river's `RiverPath` from the frames it already carries; `length_m` is the last frame's arc
> length and `closed` is unread by `sample`, both pinned by
> `a_reconstructed_river_answers_like_the_path_it_came_from`), and `camera_underwater` asks
> `WaterSurface::height_at` + `inf_water::highest_surface` — the same functions P20.2's buoyancy
> samples and `the_sim_and_the_renderer_derive_the_same_waves` pins. Only `drawable()` bodies
> count, which is the filter the water pass applies: a body that draws nothing must not fog a
> camera either, or `water_off_path_is_byte_identical` would be lying. A conservative,
> allocation-free early-out (level + max wave amplitude; plus the rectangle for a lake) runs in
> front of the query so a camera far above the water does not build a river's centreline every
> frame — `the_cheap_reject_never_drops_a_submerging_body` pins the one direction that matters,
> and caught the first version of it: a river gets **no XZ reject**, because
> `RiverSample::inside` tests only the *lateral* offset, so the Ring-0 evaluator answers for
> points beyond a river's mouth and a box around its frames would have dropped them.
>
> **The off path is pinned by a counter, not by pixels.** `UnderwaterReport` (the house
> `SharedStreamReport` pattern) is bumped at the point in `run` past which the encoder will be
> touched, and `underwater_off_path_never_engages` asserts it stays at zero for a scene with
> **drawable** water and a camera above it, at every quality tier — then moves the camera under
> and asserts it increments. A pixel comparison could not make that claim: a pass that engaged
> and wrote the scene back unchanged is byte-identical from outside. (The P20.1 water pass's
> module doc cited an equivalent inside test, `a_scene_without_water_records_nothing`, that was
> never written; the citation is corrected in the same commit and the instrument above is what
> such a test would use.)
>
> **WETNESS IS CONTENT, NOT A CAMERA EFFECT.** The band is a pure function of a fragment's world
> position and the frame's water bodies: an ocean's level (unbounded), a lake's authored
> rectangle, and a **river's centreline** — its surface follows its spline, so the band does too
> (the nearest of up to 32 decimated segments gives the local level; taking the body's reference
> level instead would wet a whole hillside at the height of the river's source). The P18.2
> camera-residency law is what makes this delicate: an ocean's *drawn* patch is snapped to the
> camera, and sourcing the footprint from that patch instead of from the body's level would slide
> a shoreline under a moving player. `wetness_is_a_pure_function_of_the_water` (exact, GPU-free)
> and `the_wet_band_does_not_follow_the_camera` (pixels) pin both halves.
> The response is `albedo × 0.55` and `roughness × 0.35` inside a `0.75 m` band above the level
> (everything at or below it is submerged, hence fully wet), with a `2 m` footprint dilation so
> the band does not stop dead on the authored polygon. **All four are engine constants with the
> argument in their doc comments** (`crates/inf-render/src/wetness.rs`), not authored knobs —
> which is what let P20.3 ship without a schema bump. **P22's material response is where a
> per-material wetness curve belongs**, and it will find the band already computed.
>
> **THE GOLDEN DELTA — deliberate, bounded, and re-blessed in one single-package pass.**
> Wetness is default-on and the three P20.1 water goldens all carry `hill_terrain`, so ground at
> and below their water levels is now darker. Measured old-vs-new: `water_ocean_noon` 6.3 % of
> pixels moved (mean luma −0.95/255), `water_river` 2.5 % (−1.04), `water_lake_dusk` 42.6 %
> (−9.19) — the lake is the big one because its level (5 m) submerges most of a terrain spanning
> [−0.5, 7.5] m, so most of its visible ground is *fully* wet rather than banded. All three stayed
> **inside** the strict perceptual tolerance (worst mean 0.030 of 0.06, worst max 0.173 of 0.35),
> so the strict gate would have passed either way; they were re-blessed anyway because the images
> no longer showed what the renderer produces. **The other 42 goldens are byte-identical** —
> the underwater node returns before touching the encoder above the waterline, and `wet.dims.x`
> is 0 on a scene with no water, so every call site is a present-but-false branch.
>
> **NO NEW PROJECTOR STATE — the strongest form the mirror rule takes.** Both hosts already
> publish `RenderScene::waters` through the character-for-character-gated `project_water`;
> `pack_wetness` and `camera_underwater` are derivations *over that list*, performed once in
> `EngineRenderer::render`. Two hosts cannot disagree about a derivation neither of them
> performs, so `host.rs::rebuild_scene` and `inf-player/src/render.rs::build_scene` are untouched
> and the P20.1 mirror gate stands unchanged. Likewise **no sim change**: the underwater pass is
> view-dependent post-processing and nothing under `crates/inf-physics` was touched, so the
> replay and PIE trace gates ran untouched.
>
> **`EnvBinding` grew a binding and NOT a key component.** Wetness rides at `@binding(13)` of the
> shared env group (so terrain, mesh, skinned, vgeom and scatter all declare it and the layout
> cannot drift from the declaration), backed by one fixed-size uniform buffer created in
> `EngineRenderer::new` and only ever written. The `ResourceKey` invariant exists for resources
> that get **recreated** — a buffer that never is cannot go stale behind a cached bind group —
> so it is the second entry, after `frame.shadow.*`, in the documented exclusion, and the
> invariant comment now says so with the same "if it ever becomes resizable" clause.
>
> **DEFERRED / v1 LEDGER.**
> * **KNOWN DIVERGENCE — hidden water bodies fog nobody but still float boats.** The render
>   projectors skip a `WaterBody` on a hidden entity (`host.rs`'s `if visible`, mirrored in the
>   player); `PhysicsBridge3D`'s gather walks **every** `WaterBody` with no visibility test
>   (`inf-physics/src/d3/ecs.rs`). Hide a lake in the outliner and a swimmer keeps swimming, a
>   boat keeps floating and the water events keep firing while the camera stays dry and the
>   surface is gone. Which side is wrong is a genuine design question — visibility is an *editor*
>   concept and arguably has no business reaching the fixed step — and P20.3 is render-only, so
>   it is **named, not papered over**: the doc on `RenderWater::surface()` states it at the seam.
>   Deciding it (and, if the sim is the one to change, doing so behind the replay gate) is P20.4's.
> * **A boat thirty metres past a river's mouth still floats.** `RiverSample::inside` tests only
>   the *lateral* offset against the local half-width; `RiverPath::sample` clamps to the end
>   segment, so the Ring-0 evaluator answers "inside, at the mouth's level" for any point beyond
>   either end of an **open** river. P20.3 only *met* this designing the cheap reject (a box
>   around the frames would have dropped points the evaluator accepts) — but the consequence is
>   P20.2's, and user-visible: buoyancy, drag, swim and the water events all fire past the mouth,
>   over dry land, for as far as the lateral test keeps passing. Closing it means an arc-length
>   bound in `inside` for open paths — a **sim change**, so it belongs behind the replay gate
>   with the other hydrology work: **P20.4's**, alongside the visibility divergence above. The usual screen-space
>   god-ray gathers bright pixels toward the sun; from below, the v1 surface shader renders the
>   deep colour (its Fresnel and reflection terms were written for a camera *above* the water),
>   so there is nothing bright in the frame to gather. Each of the 24 fixed taps instead asks
>   whether that pixel's ray reaches the surface unoccluded and how close it is to the sun
>   (`cos^24` — a ≈12° lobe, because a shaft's root is a patch of roughened surface, not a
>   point). That gives beams with real *ends* (a rock cuts one off) at a fixed cost, but it is
>   **not** volumetric: no density variation along a beam, no caustic banding from the wave
>   field, no shafts from geometry *above* the water, and the sun's screen position is
>   **unrefracted** (from below the sun really sits inside Snell's window). A wave-modulated or
>   marched version is the follow-up. Shafts are gated on `WaterQuality::light_shafts()`
>   (= the refraction tier); the **fog is never gated** — absorption is the content.
> * **Shafts are faded out by SUN ELEVATION, and that fade is the only time-of-day coupling they
>   have.** `uw_source` asks whether a ray rises to an unoccluded surface and how close it points
>   to `view.sun_dir` — nothing in that question knows whether the sun is *up*, and `sun_dir`
>   genuinely goes below the horizon (P17.1's clock swings it; a projector may hand over
>   straight-down). Unfaded, a sun 10° below with a ray rising 2° at the same azimuth gives
>   `pow(0.978, 24) ≈ 0.59` — 59 %-strength god rays at civil twilight. `shaft_sun_fade`
>   smoothsteps from **zero at 2° below** the horizon (refraction lifts the disc ~0.57° and it is
>   ~0.27° in radius, so a geometrically set sun is still lighting the water) to **full at 5°
>   above**, folded into the packed intensity and clearing the enable flag once it reaches zero
>   (so the 24-tap loop is skipped outright at night). What it is NOT: moonlit shafts, or any
>   coupling to the sun's *colour* or *intensity*.
> * **The fog's depth path is not antialiased.** The colour path loses nothing (the full-screen
>   write puts the resolved colour in all four samples, and the final resolve reproduces it), but
>   the column is `textureLoad`ed at sample 0, so a pixel straddling a silhouette is fogged
>   entirely at the near depth or entirely at the far one. Inherited from the water pass's
>   arrangement rather than introduced here, and sub-pixel where the fog is strong.
> * **"One absorption story" does not hold at `WaterQuality::Low`.** With no resolved scene colour
>   to refract, the *surface* shader composites `mix(deep, shallow, T)` instead of
>   `scene·T + deep·(1−T)`. The extinction is the same `exp(−a·d)` on both sides; the composite is
>   not, so at Low the two sides of the interface stop agreeing pixel for pixel. A Low-tier
>   surface that also takes the scene-colour form is the follow-up.
> * **Partial submersion is a whole-screen switch, softened rather than split.** The treatment is
>   all-or-nothing per frame, with strength ramped over the first `UNDERWATER_RAMP_M` (0.25 m)
>   so crossing the line has nothing to pop. A camera *straddling* the waterline still gets one
>   answer for the whole frame; a near-plane waterline split is the named follow-up. The switch
>   itself uses the **displaced** surface, so a passing crest genuinely submerges you.
> * **The column cap is ONE PLANE, placed at the displaced surface over the camera.**
>   `Underwater::surface_y` is what `WaterSurface::height_at` answered at the eye's own XZ (wave
>   included), so the plane sits at the right height *for the camera*; it is then treated as flat
>   for every pixel, because a per-pixel Gerstner inverse in a post pass would be a second surface
>   evaluation for an error of one wave amplitude on a distance already measured in tens of
>   metres. The derivation is pinned by `the_column_cap_follows_the_displaced_surface`, which
>   feeds a body whose `surface_y != level_m` — the forwarding test alone would not have caught a
>   swap to the still-water level.
> * **Wetness is applied by `terrain.wgsl` and `mesh.wgsl` only.** `skinned_mesh`, `vgeom_mesh`
>   and `scatter_mesh` declare the binding (they share the env group) but do not call
>   `wet_apply` yet — characters, meshlet geometry and scattered foliage do not darken at a
>   shoreline. One-line additions when P22 arrives.
> * **The band is a shading-time loop, not a map.** Up to 8 bodies per frame and 32 shared river
>   segments, evaluated per fragment; a level with rivers pays a bounded inner loop in every lit
>   fragment. A baked distance field is the optimisation if it ever shows up in a budget ratchet.
>   Bodies past the eighth are dropped deterministically, in projection order.
> * **The water surface seen from BELOW is still P20.1's shading.** Fresnel, the sky reflection
>   and total internal reflection are all wrong-side-of-the-interface; the pass fogs what the
>   surface draws rather than re-deriving it. Named here because the underwater golden shows it.
>
> Files: `crates/inf-render/src/{water.rs,wetness.rs (new),lib.rs,renderer.rs}`,
> `crates/inf-render/src/passes/{underwater.rs (new),water.rs (one corrected citation),mod.rs}`,
> `crates/inf-render/src/shaders/{underwater.wgsl (new),wetness.wgsl (new),terrain.wgsl,
> mesh.wgsl}`, `crates/inf-render/tests/golden.rs` + two new PNGs + three re-blessed ones,
> `runtime/inf-player/tests/phase18_gate.rs` (the golden inventory, 45 → 47), and — the one
> projector touch, made character-identically on both sides so the P20.1 mirror gate stands —
> `RenderWater::spline_closed` forwarded by `editor/crates/inf-viewport/src/host.rs` and
> `runtime/inf-player/src/render.rs`, so a river's loop flag reaches the Ring-0 `RiverPath` the
> reconstruction rebuilds instead of being silently guessed.

> **STATUS — P20.4 Hydrology authoring: COMPLETE (2026-08-02).**
>
> The batch that closes Phase 20: two inherited **sim defects** fixed behind the replay gate,
> the water tools an author needs, the P19.2 biome hint's first reader, the P19.1 flow map's
> first water consumer, and the phase gate. **No schema bump** — v17 and v18 are still the last
> two, and every field the tools write already existed on `WaterBody` from P20.1. All 47 goldens
> byte-identical.
>
> ---
>
> **THE RIVER-MOUTH BUG, CLOSED — and it was a sim change, so it went behind the gates.**
> `RiverSample::inside` tested only the *lateral* offset against the local half-width, and
> `RiverPath::sample` clamps its projection to the end segment, so for an **open** river every
> point on the centreline's extension — to infinity — answered "inside, at the mouth's level".
> A boat thirty metres past the mouth floated; buoyancy, drag, swim and the enter/exit/splash
> events all fired over dry land. The fix is a second bound: `sample` keeps the *unclamped*
> parameter of the winning segment and reports `RiverSample::beyond_m`, the overshoot past the
> first or last segment of an open path, and `inside()` now tests **both** bounds. A ribbon is a
> bounded surface; testing one of its two bounds was the bug.
>
> Three details are load-bearing. **The bound applies only on the two END segments**, so a
> hairpin's return arm — nearest an interior segment, and a genuinely wet stretch of river — is
> untouched (`a_hairpin_stays_wet_beside_its_own_far_arm`). **A closed path is exempt by
> construction**: it has no ends, so `beyond_m` is identically `0` and the loop's seam is not a
> wall. And the mouth plane is **inclusive**, which needs a tolerance — `RIVER_END_TOLERANCE_M`
> (1 µm) — because the frames are a *resampling* and the last one lands on the authored endpoint
> to within the arc-length LUT's inversion error, so without one, whether the water reaches its
> own mouth would depend on the last bit of that inversion. The same reasoning
> `bank_fraction() <= 1.0` already embodies, made explicit because this edge is not exact.
>
> **No existing test was asserting the bug.** Every P20.1/P20.2 case queries inside the banks or
> laterally outside them; the P20.2 physics gates (determinism, pool-size invariance, PIE ==
> shipping, swim, events, the editor↔runtime parity case) were re-run and are green untouched,
> because none of their fixtures put a body past a river's mouth. The one place the old
> behaviour was *documented* was P20.3's cheap-reject comment in `could_submerge`, which said an
> XZ reject for a river would be unsound; that comment is now corrected — the reject has become
> *possible* and is still *absent*, and the difference is written down rather than left to be
> rediscovered.
>
> ---
>
> **THE VISIBILITY DIVERGENCE, DECIDED: `Visibility` filters what is DRAWN, never what is
> SIMULATED.**
>
> P20.3 named a KNOWN DIVERGENCE — the render projectors skip a `WaterBody` on a hidden entity,
> `PhysicsBridge3D`'s gather walks every one of them — and left which side was wrong open. It is
> not a divergence. It is the engine's existing hidden-entity law showing through the first
> feature that has both a render half and a sim half, and the evidence that settled it is that
> **nothing in the simulation has ever read visibility**: the 2D and 3D bridges gather rigid
> bodies, colliders and joints on component presence alone (a hidden wall still blocks); P19.5's
> `ScatteredSolid` colliders likewise (hiding a `PcgVolume` removes its instances from the frame
> and leaves every building collider standing); `terrain.height_at` picks the lowest-`Guid`
> non-empty terrain with no visibility test; `AudioSource`s keep playing; sensors keep
> triggering; `partition::occupies_space` bins a hidden entity like any other. Across the whole
> repository `ComputedVisibility` has exactly **three** readers: the two render projectors and
> the Outliner's DTO.
>
> So **neither side changed**, and the decision is the interesting output. Teaching the fixed
> step to read visibility — for water alone or for everything — would make an *editor authoring
> toggle*, one that cooks into the pack and is restored on load, change physics; the alternative
> (water alone honouring it) would make water the single exception to a rule every other system
> follows silently. The rationale now lives where P20.3 put the divergence, on
> `RenderWater::surface()`, and the law is **pinned from both sides** so a future "fix" trips a
> test rather than shipping: `crates/inf-physics/tests/water_visibility_3d.rs` asserts a hidden
> lake floats a boat **bit-identically** to a visible one over 900 steps (with the entity's
> `ComputedVisibility` asserted false, the box asserted to actually float, and a
> water-removed control that really does diverge) *and* that a hidden **collider** still blocks —
> the consistency evidence, asserted rather than claimed; `water_projection.rs` asserts the same
> body reaches no `RenderScene::waters` while `water.surface_height` still answers the lake's
> level and not the ocean's beneath it. Because no sim behaviour moved, no replay gate needed
> re-blessing — the gates were re-run and are green as they stood.
>
> The authoring answer to "make this lake go away for a moment" is to remove or retune the
> component. A per-body `enabled` switch that *does* reach the sim is an additive field and is
> ledgered below, not built.
>
> ---
>
> **THE TOOLS.** A `ToolMode::Water` in the terrain toolbar with two sub-modes, following the
> P19.2 biome-paint pattern (one tool mode, a sub-mode picker, brush state in `viewportStore`,
> pushed over `viewport_set_water`). It is **not** a brush and does not pretend to be: a
> **river** click *appends a control point* (the first click on empty space starts a river and
> lays two points a metre apart, so the author sees a ribbon immediately rather than a
> degenerate path), and a **lake** press-drag-release *defines a rectangle*. **The water tool's world pick REFUSES rather than guessing.** Every other terrain tool resolves
> a click through `pick_world_point`, which falls through to the `y = 0` ground plane on a terrain
> miss. That is right for a drag-drop and wrong here, because a water click **commits geometry**:
> over a coarsely-paged streamed terrain, or a hole, the fallback would silently plant a control
> point or a lake corner at sea level — and two authors at different camera distances would commit
> *different geometry from the same click*. `water_pick` hits terrain or rejects, on the same
> `reject_tool` seam the sculpt brush already guards its own commits with; a level with **no
> terrain at all** is not a miss (there the ground plane *is* the ground). A sub-metre lake drag
> now says so too, instead of refusing silently. Found in the audit — it is the P16.6 "reading the
> document's own terrain set was the bug" lesson applying one tool over. Every mutation goes through a
> `SceneDoc::edit_*` — `edit_create_river`, `edit_create_lake`, `edit_append_spline_point`,
> `edit_set_river_profile`, `edit_set_water_level` — so each is exactly **one undo step**, taken
> around the complete change on the `edit_create_streamed_terrain` pattern (components attached
> *before* the record is snapshotted, so redo restores a lake rather than an empty entity). The
> point edits ride the existing `SwapComponents` command rather than a new variant, because a
> `Vec<Vec3d>` is not a reflection-addressable scalar and the P19-era mechanism already covers
> exactly this.
>
> **`edit_append_spline_point` CREATES the `Spline` when the entity has none**, and that is not a
> convenience. A `WaterKind::River` added through the Details "Add Component" menu is exactly the
> state "I have declared a river and not drawn it yet", and it is the state the tool meets most
> often; refusing it *wedged* the tool — every click resolved to the same spline-less selection,
> did nothing, and said nothing (audit). `SwapComponents` round-trips a component *addition*
> natively, so undo removes the spline again exactly as `edit_add_component` does.
>
> **There was no spline editor to reuse, and this batch did not build a second one.** The audit
> question the brief asked was answered by looking: `Spline` is authored today through the
> Details `ListField` over `points`, the viewport *draws* splines (cyan polyline, control-point
> crosses when selected) and picks nothing, and P19.4's grammar binds a spline by **GUID text
> param**. So the river tool is the engine's first spline *gesture*, and it deliberately writes
> the same `Spline` component the Details grid edits — the two are the same data, and the
> polyline the viewport already draws is the river's preview for free. A general per-control-
> point drag gizmo is ledgered.
>
> **BOTH HALVES OF THE REPORTING REACH THE AUTHOR — and that was an audit fix.** The river verdict
> is a toolbar readout on the selected river: "✓ 210 m, falls 32 m", or
> "⚠ 2 issues: buried in the ground for 40 m (worst 3.1 m)" — the worst span of each kind, ground
> problems before surface ones because that is the order an author fixes them, the full list on
> hover, a click to re-check. It re-reads on entering the tool and on every `world://delta`,
> subscribed at the store like every other live projection in that file (so no component carries
> the banned set-state-in-effect shape). The lake drag's **coverage, max/mean depth and sample
> count** ride the existing `viewport://tool-status` seam as a live readout while the rectangle is
> being dragged, on a new `report_tool` beside `reject_tool` — split because a rejection belongs
> in the Output Log and a per-frame measurement does not, and because "there is no ground here"
> and "the lake is empty" must not both render as 0 %. Both were computed and dropped in the
> first cut of this batch: a capability the ledger claims and no author can reach is a false
> ledger entry, which is why the audit called it a blocker.
>
> **The lake's fill-level preview is a real marching-squares waterline**, not a rectangle with a
> number beside it. `inf_water::hydro::fill_preview` samples the ground on a clamped
> `(n+1)²` grid and returns the coverage fraction, the max and mean depth, how many samples the
> terrain **answered for** (a preview whose `known` is 0 says "there is no ground here", which is
> not the same statement as "the lake is empty"), and the contour as world-XZ segments the
> viewport draws as debug lines. Holes are **excluded rather than defaulted**, so a rectangle
> half off the terrain does not report itself half dry and no waterline is drawn along the edge
> of the *data*. The two ambiguous saddle cases are resolved the same way every time, so the
> contour is a function of the heights alone.
>
> **BED VALIDATION, SPLIT BY WHAT IS KNOWABLE.** The brief asked for a bed advisory; there turned
> out to be two different questions, with two different remedies, and only one of them is
> answerable at cook time:
>
> * **The authored bed climbs** — `surface(s) − depth(s)` gains elevation in the flow direction.
>   Needs no terrain at all, so it is a **cook advisory**, a sibling of P20.1's surface check
>   rather than a stronger version of it. A river descending 2 m while its depth tapers from 5 m
>   to 0.5 m has a bed 2.5 m *higher* at the mouth than at the source: that is a basin, and
>   nothing at runtime says so, because the surface still slopes the right way and the water
>   still renders. It is a second advisory because the remedies differ — one moves spline points,
>   the other moves depths — and an author told only "your river is wrong" would fix the wrong
>   one. It honours a negative `river_flow_m_s` exactly as the surface check does, skips closed
>   loops, and rides **above the partition branch** with the P20.1 ORDERING LAW.
> * **The river is buried in, or perched over, the ground** — needs a heightfield, which the cook
>   validates structurally and never pages in. So `inf_water::hydro::bed_conflicts` lives in
>   Ring 0 and the **editor** runs it, where `Terrain::data` is resident under the author's
>   cursor. Adjacent offending frames merge into spans carrying their worst frame's world
>   position, holes close a span rather than bridging it, and the report says how many frames the
>   terrain answered for — a verdict over 3 of 200 frames is not a clean bill of health, and it
>   says so instead of implying one.
>
> **The tool re-runs both cook advisories itself**, at the cook's own tolerance, so it says what
> the build will say. That tolerance moved to Ring 0 the moment it acquired a second reader
> (`inf_water::UPHILL_TOLERANCE_M`, imported by both the cook and the editor command): a tool
> nagging about rivers the build accepts, or a build advisory arriving as a surprise at package
> time, are the two failure modes of two copies, and neither is worth a "keep these in sync"
> comment. The terrain-aware checks use their own, larger `BED_TOLERANCE_M` (1 m), because they
> additionally sample a bilinear heightfield along a curve that crosses tile diagonals.
>
> **ONE TERRAIN AUTHORITY FOR AUTHORING, AND ONE RIVER-PROFILE SANITIZER.** Two audit smalls of
> the same shape — a doc claiming an agreement the code did not have.
>
> (1) The viewport resolved the biome hint through the **topmost** ground while Ring 1's
> `water_defaults` resolved it through the **lowest-`Guid` first answer**, under a comment saying
> they matched. Ring 1 now uses topmost too (`hydro::topmost_ground`, ties to the lower `Guid`),
> which is what the brush ring and the foliage drop height beside it already use: the author's
> question is "what am I pointing at", and on overlapping terrains that is the surface they can
> see. The **simulation's** authority stays lowest-`Guid`-first, deliberately, and now says so —
> a stable owner matters more to a fixed step than a visible surface. The biome and the height now
> come off the *same* terrain by construction, which they did not before.
>
> (2) The **cook** built its `RiverProfile` from the raw authored fields while both projectors,
> `PhysicsBridge3D` and the editor all clamped, so a negative authored depth tapered the cook's
> bed differently from everyone else's — breaking the one thing the tool's re-run of the cook
> checks is for. `RiverProfile::authored` is now the single sanitizer and all five call sites go
> through it: widths and depths floor at zero, and the flow speed keeps its sign, because a
> negative one reverses the river.
>
> **THE BIOME WATER HINT, READ AT LAST.** `BiomeDef::water_hint` has been inert plain data since
> P19.2 — declared, round-tripped through the DTO and the editor, and consumed by nothing. It is
> now the **default provider** for a new body's level: `water_defaults(x, z)` answers the painted
> biome's hint if it has one, else the ground under the point, else `0` (a plausible sea level,
> and the same answer `terrain.height_at` gives a terrain-less world), and reports **which** so
> the toolbar can say why. Kept v1-honest and said so at the seam: it does not place water, it
> does not fill basins, and painting a biome makes nothing appear — a hint that spawned geometry
> would be a generator wearing a hint's name. River width and depth are *not* derived from it,
> because a biome carries a still-water **level** and inventing per-biome river dimensions from
> one would be making data up.
>
> **Entering the Water tool ARMS the hint table**, and the first cut did not — an audit blocker,
> because the consequence reached *committed content*. The table is filled by the same Ring-2 push
> that answers `terrain_biomes`, so a fresh session that went straight to Water committed a lake
> at the picked **ground**, while the identical click after a detour through the Biome tool
> committed it at the **hint**. A committed level is not allowed to depend on which tools the
> session happened to visit. The `Water` arm of `setToolMode` now re-reads the vocabulary exactly
> as the `Biome` arm does. The same fix deleted a dead `WaterSettings::biome_level_hint_m` that
> was *documented* as pushed from Ring 2 and passed `None` unconditionally — the hint is resolved
> per click out of the id-indexed table, so a single pre-resolved one had nothing to say.
>
> The hint reaches the native viewport the way the biome *palette* does: id-indexed, resolved
> once in Ring 2 and pushed (`BiomeSet::water_hints`, the exact shape and length rule of
> `BiomeSet::palette`, with the reserved id 0 always `None`). The viewport thread holds a
> document, not an asset database, so it cannot resolve a `.inf_biomes` itself — and the tool
> needs the hint *per click*, from the biome under the cursor, which is why the toolbar does not
> send one.
>
> **P19.1 FLOW MAPS REACH THE WATER — additively, which is why no golden moved.**
> `inf_ecs::hydro::TerrainFlow` gathers the level's terrains once per projection (ascending
> `Guid`, first answer wins — the same rule the height query uses, stated in Ring 0 so the two
> MIRROR projectors cannot each invent one) and turns `DataMapKind::Flow` into a per-frame foam
> gain through `inf_water::flow_foam_gain`. The curve **can only ever add**: it returns exactly
> `1.0` over terrain that was never eroded, over a hole, and in a level with no terrain, and
> saturates at `1 + 0.6` at 1000 m³ — the same ceiling `mask.flow` already uses, because a flow
> value that reads as "a real channel" to a scatter mask should read as one to a river.
> A *subtracting* coupling ("a river off-channel is glassy") was the other candidate and was
> rejected: it makes the absence of a bake — the default state of every terrain in the engine —
> into a visible change to every river already authored, which is a migration disguised as a
> feature.
>
> The gain rides on **`inf_render::WaterFrame` and deliberately not on `inf_water::RiverFrame`**:
> the Ring-0 frame is what the fixed step samples, foam is not a force, and keeping the gain on
> the render mirror is what makes "the sim and the renderer derive the same waves" still
> literally true. `FrameGpu` grew a fourth `vec4` rather than stealing a mantissa (all twelve
> existing lanes are load-bearing), `water.wgsl`'s `VsOut.profile` became a `vec3`, and the
> fragment stage multiplies the flow-foam *speed* by it — so on an unmapped terrain the
> expression is bit-identical to P20.1's. Pinned by
> `the_flow_map_modulates_a_rivers_foam_and_nothing_else`, which asserts the **exact** identity
> frame-for-frame with no bake and the saturated gain with one, and that the two vectors differ.
>
> **Goldens stay at 47, and that is a decision.** The only render surface P20.4 touches is the
> flow-foam gain, which is provably the identity on every existing golden (none of them carries
> an eroded terrain), so nothing moved. A new coastal golden would differ from `water_river` by a
> foam intensity — a claim a PNG makes weakly and a projection test makes as `1.0` vs `1.6`,
> which is the form it ships in. The re-bless was run the house way regardless (the whole suite
> under `INF_BLESS_GOLDENS=1`, `git status` on the golden directory) and reported nothing.
>
> **THE PHASE 20 GATE.** `samples/phase20-coastal` is the plan's own done-when sentence built as
> committed content: a 512 m coast (a 42 m headland falling to −10 m, a meandering valley, a dug
> basin), an **ocean** at sea level, a **head lake** in the basin, a **spline river** running the
> valley from the lake to the shore with a 6→14 m width taper and a 1.2→2.0 m depth taper,
> **eight buoyant crates** (six at sea, two on the lake) and a **swimmer** driven by a committed
> `Swimmer.inf_act` whose Tick asks for a brisk swim *and a full second of accumulated free
> fall* — so the P20.2 asymmetric sink authority is exercised by shipped content and not only by
> a unit test. The height function is **polynomial throughout** (a cubic meander, quadratic
> valley walls, a paraboloid basin): this is committed content and `std` trigonometry is not
> bit-portable, so a `sin` here would have made the `.inf_lvl` machine-dependent.
>
> `runtime/inf-player/tests/phase20_gate.rs`, six arms: **determinism** (two fresh loads of one
> cooked pack, 900 steps, bit-identical), **PIE == shipping** (the editor payload vs the pack,
> bit-identical), **anti-vacuity** (crates settle at draughts that *depend on their densities* —
> the lightest rides higher than the heaviest **by at least half the 0.25 m Archimedes predicts**,
> because a bare `>` would pass on a millimetre — the lake crates settle at 33.6 m rather than at
> sea level, and the swimmer *rises*), **the river's mouth is finite in the shipped pack** (a
> probe 30 m past the mouth gets the sea's level, not the river's; the P20.4 fix asserted through
> the shipped `water.surface_height` seam rather than only in Ring 0), **the cook is silent**
> (neither water advisory fires on the flagship sample — an advisory that fires on correct
> content is one nobody reads), and **budget** (the composed scene builds inside `LOAD_BUDGET_MS`
> and steps inside `FRAME_BUDGET_MS`, both imported from their homes, each arm taking the ceiling
> of its own class — a load measured against the frame budget is the category error that cost a
> CI failure earlier in this phase).
>
> **Remainders, stated.**
> * **The river tool has no control-point DRAG.** Points are appended by clicking and edited
>   through the Details list; moving one in the viewport needs per-point picking, which the
>   viewport has never had for splines (they live in `scene.debug` and consume no pick id). That
>   is the general spline-gizmo work, and it belongs to whichever batch wants it for roads and
>   rails too, not to water alone.
> * **No basin solver.** The lake tool takes its level from the click (or the biome hint) and
>   shows where that lands; it does not find the level that fills a depression to its rim. The
>   preview makes the manual search cheap, which is the v1 trade.
> * **No river→terrain carve.** Placing a river does not sculpt the channel under it; the tool
>   *reports* the bed conflict — in the toolbar, on the selected river, as "⚠ 1 issue: buried in
>   the ground for 40 m (worst 3.1 m)", with the full verdict on hover — and leaves the sculpting
>   to the sculpt brush. Carving is a terrain edit with its own undo record and its own erosion
>   interaction, and doing it silently inside a water placement would be the worst of both.
> * **The flow coupling is foam only.** Flow does not modulate a river's *speed*, its width, or
>   its absorption, and it never reaches the sim — a rapid looks faster and is not.
> * **`bed_conflicts` samples the centreline, not the banks.** A river whose left bank is buried
>   in a cliff while its centre is clear reports nothing. Sampling the ribbon's cross-section is
>   the obvious v2 and is `frames × width` work rather than `frames`.
> * **The lake readout is a STATUS LINE, not a panel.** Coverage, max/mean depth and the
>   ground-sample count arrive on `viewport://tool-status` while the rectangle is being dragged,
>   which is where the shell shows one line. A hydrology *panel* — the verdict, the fill preview
>   and the profile editor in one place, with a jump-to-conflict button off `worst_x/worst_z`
>   (which the DTO already carries and nothing yet reads) — is the obvious follow-up and is not
>   in this batch.
> * **The river verdict follows the FIRST selected entity.** A multi-select reports on one river;
>   a level with several is checked one at a time. There is no "check every river in this level"
>   sweep short of cooking it, which the cook advisories then answer for the two terrain-free
>   halves.
> * **The tools are terrain-only.** Water placed over a mesh floor, a scattered solid or a
>   grammar building answers from the ground plane, exactly as the sculpt and foliage brushes do.
> * **The biome hint resolves through the level's FIRST bound terrain**, matching how
>   `terrain_biomes` resolves the paint tool's vocabulary — so the two tools agree about which
>   `.inf_biomes` is in play. A multi-terrain level binding *different* sets would take the first
>   one's hints everywhere; per-terrain resolution is a `water_defaults` signature change away.
> * **Viewport interaction is human-verified**, like every other native-viewport gesture in this
>   repository: CI does not create a window. The *logic* is not — every edit, every report and
>   every preview is a Ring-0/Ring-1 function with its own tests, and the win32 layer is the
>   press/drag/release plumbing over them.
> * **A hidden water body still simulates** — the decided law above, not an omission. There is no
>   per-body `enabled` field yet; adding one is additive and would need its own replay-gated
>   batch, because it *would* change what a level means.

> **STATUS: Phase 20 COMPLETE** (2026-08-02) — **local gates green; CI pending push.** (Written
> with the commit rather than after the CI run, like Phases 16–19's, and saying so rather than
> implying a green run that has not happened.)
>
> **The four batches, in one line each.** **P20.1** gave the engine water at all — oceans, lakes
> and spline rivers on **one** Gerstner model, **one** shader and **one** pass, with the CPU
> deriving and the GPU only evaluating so there is no second `f32` copy of the wave model to
> drift, and `WaterSurface::height_at` designed for the fixed step before there was a simulation
> to design it for. **P20.2** made water **physical** — Archimedes over four sample points on
> rapier's own exact volumes, linear still-water drag, enter/exit/splash into Blueprints and the
> audio command queue, and a swim mode whose asymmetric sink authority is the whole trick.
> **P20.3** put the camera *inside* the medium and the waterline *on the ground* — one absorption
> story seen from both sides, and a wetness band that is content rather than a camera effect.
> **P20.4** fixed the two sim defects the earlier batches had named, decided the visibility
> question, and shipped the authoring: river and lake tools, the bed advisories, the biome hint's
> first reader, the flow map's first water consumer, and the phase gate.
>
> **Schema v16 → v18, in two bumps, both the `EntityRecordV10` *shape*.** v17 appended
> `water_body`, v18 appended `buoyancy`; each cost exactly **one discriminant byte per entity
> that does not carry it**, measured as a delta against the frozen previous shape of the same
> record, with every sample's delta equal to its entity count. P20.3 and P20.4 added no field at
> all — the underwater constants are engine constants with the argument in their doc comments,
> the wetness response likewise, and every value the water tools write already existed on
> `WaterBody` from P20.1. Two bumps in a phase is the ceiling the house rule sets, and the phase
> spent exactly two.
>
> **The gate is `samples/phase20-coastal` + `runtime/inf-player/tests/phase20_gate.rs`**: the
> plan's done-when sentence as committed content, asserted deterministic across two loads,
> identical between a cooked pack and a PIE payload, non-vacuous on the physics (density-
> dependent draughts, a lake 33.6 m above the sea, a swimmer that surfaces), silent in the cook,
> and inside both budget classes. Beside it stand P20.1's projection gate, P20.2's ten-arm
> physics gate (including the subprocess pool-size leg and the editor↔runtime parity case),
> P20.3's engagement-counter off-path gate, the two cook-advisory suites, the projector MIRROR
> gate and 47 goldens.
>
> **THE PHASE'S REMAINDER LEDGER, swept across all four batches.** Each batch's own block carries
> its full list; this is the consolidated carry-forward, in the shape Phases 18 and 19 close with.
>
> *Surfaces and shading (P20.1, P20.3).* An ocean is a **finite 8 km patch** — a camera on a flat
> horizon can see it end, and a projected-grid ocean is different work. There is **no SSR on
> water**, so a boat reflects the sky and not itself. The surface seen **from below** is still
> P20.1's above-water shading (Fresnel, the sky reflection and total internal reflection are all
> wrong-side-of-the-interface). "One absorption story" **does not hold at `WaterQuality::Low`**,
> where the surface composites `mix(deep, shallow, T)` instead of `scene·T + deep·(1−T)`. The
> fog's **depth path is not antialiased** (the column is `textureLoad`ed at sample 0). The
> **column cap is one plane** at the displaced surface over the camera, treated as flat for every
> pixel. Partial submersion is a **whole-screen switch**, ramped rather than split at the near
> plane. Light shafts are **not volumetric** and have **no moonlit variant** — they fade to zero
> by sun elevation and stay there all night. Wetness is applied by `terrain.wgsl` and `mesh.wgsl`
> **only**: `skinned_mesh`, `vgeom_mesh` and `scatter_mesh` declare the binding and do not call
> `wet_apply`, so characters, meshlet geometry and scattered foliage do not darken at a shoreline
> (one line each when P22 arrives). The wetness band is a **shading-time loop**, not a baked
> field, bounded at 8 bodies and 32 shared river segments. FFT v2, a keyframed depth profile, a
> scissored (per-region) refraction resolve and `MAX_BODIES > 32` are all named and unbuilt.
>
> *Physics (P20.2).* Drag is **linear and still-water**: quadratic drag is deferred, so a hull's
> terminal speed cannot depend on its shape, and a body near a crest feels still water because the
> reference is `flow_at` and not the wave orbit (**no Stokes drift**). There is **no water→body
> reaction** — no wake, no bow spray, no local depression; the surface is authored and nothing a
> body does moves it. The submerged fraction is a **slab, not an integral** (exact at the
> symmetric point, where the tests assert). `Buoyancy` on a body with **no collider** does nothing
> and says nothing; **static and kinematic bodies never float**; a **trimesh floats by its AABB**.
> Swim mode has **no animation** — flipping the latch changes the motion, not the pose. The
> `water.*` **queries are instantaneous while the events are latched**, so a body bobbing at the
> waterline can make `is_in_water` flicker while no enter/exit fires — deliberate (a poll wants
> the truth now, an event wants a debounced edge) and documented at both ends, but it is a real
> asymmetry an author can trip over. `water.surface_height` answers **`0.0` where there is no
> water**, which is a plausible sea level and deliberately not a sentinel.
>
> *Authoring (P20.4).* No control-point **drag** (points are appended by click and edited in the
> Details list; per-point picking is the general spline-gizmo work). No **basin solver**. No
> **river→terrain carve** — the tool reports, the sculpt brush carves. The flow coupling is
> **foam only** and never reaches the sim. `bed_conflicts` samples the **centreline, not the
> banks**. The tools are **terrain-only**. The biome hint resolves through the level's **first
> bound terrain**. Viewport interaction is **human-verified**, as every native-viewport gesture in
> this repository is. A **hidden water body still simulates** — the decided law, with no per-body
> `enabled` field yet.
>
> *Dependency hygiene.* P20.1 promoted `inf-ecs` and `inf-math` from dev-dependencies to real ones
> in `inf-packager` and added `inf-water` + `glam` there outright; P20.4 added `inf-water` to
> `inf-ecs` (for the one flow-gain curve) and to `inf-editor-core`, plus a dev-dependency on it in
> `inf-player`. Every edge is Ring-0 → Ring-0 or Ring-1 → Ring-0, every one carries its reason in
> the manifest, and `cargo deny` is unmoved across the whole phase — **no new third-party crate
> entered the tree for water**.
>
> **Laws this phase paid for.** *One wave model, derived on the CPU* — the terrain-parity class
> of drift avoided by not creating it. *Time never reaches the GPU* — a wave arrives with its
> phase already reduced in `f64`, and the floating origin rides in the same reduction. *A rapier
> force is persistent, and a force is not an impulse of `F·dt` for POSITION* — both on
> `apply_force_at_point`'s doc, both paid for by a box that left the atmosphere at 13 km and a
> neutrally-buoyant one that rose a millimetre per step. *The advisory runs above the partition
> branch* — partitioning clears `level.entities` in place, so every future per-entity advisory
> belongs there. *Bless one package at a time* — feature unification across a multi-package
> selection produces phantom churn. *Check `df` FIRST* — "crate X required to be available in
> rlib format" is a disk-full symptom, now cited three times, and `target/debug/incremental`
> alone held 44.5 GB in this batch. And, new here: **visibility filters what is drawn, never what
> is simulated.**

- **P20.1 Water surfaces** — 1. a new `inf-water` (Ring 0) + render passes: ocean (Gerstner v1
  → FFT spectrum v2, deterministic seeds), lake volumes (flat + ripple), and spline rivers
  (flow along the parity-wave splines, width/depth profiles, downhill validation against
  terrain, P19.1 flow maps as inputs); 2. shading — depth-tinted absorption, screen-space
  refraction, shore blending against terrain height, foam from flow/wave data; 3. integration
  with the translucent pass.
- **P20.2 Water volumes & physics** — 1. a `WaterVolume` component; 2. buoyancy and drag forces
  in the rapier bridges, deterministic and replay-gated; 3. splash/enter/exit events into
  Blueprints and the audio command queue; 4. a swim-capable character-controller mode.
- **P20.3 Underwater & wetness** — 1. underwater post (tinted fog, surface light shafts v1);
  2. wetness darkening near waterlines, feeding P22's material response.
- **P20.4 Hydrology authoring** — 1. river and lake placement tools in the terrain toolbar;
  2. per-biome water-level hints; 3. the erosion → water pipeline: carve with P10 erosion, fill
  with P20 water. *(Shipped 1 and 2 in full, plus the two inherited sim fixes and the phase gate.
  For 3, the P19.1 flow map now drives a river's foam — the erosion→water edge exists and is
  gated — while the **carve** half is ledgered: placing a river *reports* the bed conflict in the
  toolbar and leaves the sculpting to the sculpt brush. See the P20.4 status block.)*

### Phase 21 — Volumetric terrain: caves, tunnels & excavation

**Goal:** terrain stops being a heightfield-only illusion — dig it, tunnel it, build under it.
**Done when:** on a streamed terrain you can carve a cave system and excavate a foundation pit
with displaced soil piles, build an underground room in the pit, save and reload
byte-identical, and it works in PIE.

**Design stance (honest):** the planet-scale base **stays a heightfield** — the streaming
clipmap economics from P16 are unbeatable at that scale. Volumetric capability arrives as
**SDF voxel chunk volumes that locally override and extend it**, the hybrid serious open-world
tech uses. We are not voxelizing the planet.

- **P21.1 Voxel chunk core** — 1. `inf-voxel` (Ring 0): a sparse SDF chunk store, f64-anchored
  like terrain tiles, format-aware serde; 2. deterministic meshing (dual-contouring /
  transvoxel class); 3. material channels aligned with the terrain splat layers; 4. per-chunk
  versions + residency riding the P16 streaming machinery.
- **P21.2 Terrain integration** — 1. volumes punch **holes** in the heightfield (per-sample
  hole mask on tiles → clipmap discards; collision switches to the voxel mesh inside a volume);
  2. seamless material and normal blending at the seams; 3. carve-brush and spline-tunnel
  tools.
- **P21.3 Excavation & soil displacement** — 1. dig tools (box/spline/brush cuts for
  foundations, parking garages, underground malls); 2. material accounting — excavated volume
  becomes displaced spoil (voxel additions or instanced debris), conservation-tested like the
  erosion mass gates; 3. undo via chunk deltas on the `EditCommand` pattern. *(All three shipped,
  plus the two carried ledger items M11 and N2. Spoil is **voxel additions**; instanced debris
  stays a P22 concern. See the P21.3 status block below.)*

> **P21.3 STATUS (complete).** Three cuts, one ledger, one undo step.
>
> *The cuts.* `VoxelToolKind` grew **BoxCut** (press-drag a rectangle, release excavates it) and
> **Trench** (waypoints → a swept rectangular cut, Ctrl+click commits), beside the P21.2 Brush
> and Tunnel; the brush gained a **dig-to-grade** mode whose dabs are columns to daylight rather
> than balls at depth. Their shapes are Ring-1 pure functions
> (`inf_editor_core::voxel_tool::{box_cut_plan, trench_shapes, brush_dab_shape}`), so every CI
> leg tests what a click commits. The shared rule: **a cut is open to the sky** — its top clears
> the *highest* ground it spans and its floor is `depth` below the *lowest*, so a pit dragged
> across a slope has no lid of surviving hillside and "3 m deep" means below grade everywhere.
> A new Ring-0 primitive, `VoxelShape::Trench`, is the swept rectangle the axis-aligned `Box`'s
> own doc comment said would be added beside it — yaw free, roll not, exact SDF, no `std` trig.
>
> *The conservation centrepiece.* `spoiled[m] == removed[m]`, per material, as **integers**, with
> **no bulking factor** (a documented non-goal: a 1:1 identity is a gate, a 1.25× fudge is a
> number nobody can test). Since no analytic mound holds an arbitrary integer count, the pile is
> an **order** rather than a solid: `rank = d·tan 35° + height`, the apex of the smallest repose
> cone containing a cell, taken ascending with `(height, x, z)` ties. Three rules make it exact —
> an already-solid cell is not a placement (so a heap on a hillside conserves as exactly as one
> on a plain), the search region **grows until it holds the count**, and materials are laid down
> in ascending index order so a two-strata dig builds a visibly layered mound. The default site
> is stated so an author can predict it and a test can pin it: *centred on the cut's +X face,
> offset by the pile's own footprint radius plus 1 m, dropped onto the ground there* — a function
> of the cut and the count and of nothing the session happened to page in. `cbrt_det` replaces
> `f64::cbrt` for the reason `psin64` exists.
>
> *The transaction.* `SceneDoc::edit_dig` judges **size, store, volume and the inline-terrain
> verdict before a single sample moves** — the size gate in particular cannot be discovered any
> other way, since you find out a dig is too big by doing it — then cuts the whole chain into one
> `CarveStroke` and displaces the soil into the same stroke. Pit, cave mouths and heap are one
> `EditCommand` labelled "Excavate", and one Ctrl+Z takes back all three, byte-identically, in
> the chunks *and* in the terrain tiles. `edit_carve_path` is now this with spoil discarded.
>
> *The audit round (four blockers, all measured).* **B1** — the spoil growth loop multiplied three
> `i32`-clamped spans unchecked, so a count past ~4.08 M panicked *while holding the
> shared-volumes guard, mid-transaction, with the cut already in the world and no `EditCommand`
> describing it*: the exact `a4e5844` worst case. Now `checked_mul`, the loop **stops** the moment
> a step outgrows the cap, and `edit_commit_dig` carries the size gate the brush path never had
> (a stroke's dabs accumulate across frames with no ceiling, so the only thing left to refuse is
> the heap — `SPOIL_TOO_LARGE_REFUSAL` says so instead of claiming nothing was cut). **B2** — the
> trench had three doc blocks and a tooltip promising the sky rule and no `surface` closure at
> all; a 1.5 m ridge mid-run left 11 of 51 ground samples outside the cut. Legs now read the
> ground over their own rotated footprint and are **horizontal**, which also deletes the
> diving-leg frame bug (a vertical shift on a pitched leg leaked into the along-run axis and put
> the first waypoint outside its own leg). **B3** — *the dig path pages its footprint first*, now
> a rule in three places: the box drag pages before it probes (it was probing `height_at`, whose
> `None` on a non-resident tile reads as "no ground", so two cameras dug two pits), the Auto spoil
> site's ground is paged before the rule reads it, and `carve_into`/`spoil_into` page the voxel
> chunks — rock in an unpaged chunk was not removed, not counted and not spoiled, **and
> conservation balanced anyway**, which is what hid it. **B4** — the sky-rule gate was vacuous:
> its fixture was monotone over its own rectangle, so the entire probe loop could be deleted and
> the test still passed. The fixture now has an interior ridge and an interior hollow, and both
> mutations (probe loop no-op; pitch coarsened) fail it.
>
> *Also this round.* The heap could be placed **back into its own hole** — the default clearance
> is the pile's *analytic* radius and the real footprint is 81 % wider, which refilled 59 of 729
> excavated samples with conservation balancing perfectly throughout. `SpoilPlan::exclude` makes
> it structural: the cut's cells are not candidates, exactly as solid cells are not. The dab cap
> became a real bound (it was `.take(32)` on a fully-materialized list — 2 M points, 80 MB, 27 ms
> per frame at the spacing floor). `cbrt_det(∞)` no longer spins. Probe semantics are now a
> **pitch** (0.5 m, ceiling 129/axis) rather than a fixed count documented as a pitch.
>
> *Honest scope.* Spoil is **voxel additions only** — no instanced debris, which is P22's
> fracture/destruction concern and is written into the bullet above rather than implied.
> `SpoilMode::Site` **falls back to the default site** when no marker has been picked, and says
> so on the readout. The sky rule's probe pitch is 0.5 m up to 64 m on an axis, past which it
> coarsens in proportion — a feature narrower than the coarsened pitch can still be missed, which
> is where the guarantee stops and `MAX_GROUND_PROBES` says so. A long trench leg over a big
> elevation change becomes a tall box and over-digs its low end, exactly as a box cut does; the
> fix is another waypoint. **A dig at the `MAX_DIG_SAMPLES` ceiling is not interactive**: mouse-up
> on 2 M samples spends ≈1.3 s under the volumes lock (4 M ≈3.1 s) because the cut, the spoil
> search and the re-mesh all run there. Decided rather than hidden — the ceiling stays, the
> number is on the constant, and *make a big dig incremental (re-mesh off-lock)* is ledgered for
> P21.4. Viewport interaction is **human-verified**, as every native-viewport gesture in this
> repository is. No schema bump (v19 stands), no `.inf_terrain` bump, no new dependency, no new
> golden.
>
> *The settlement discipline, finished (P21.3 audit ruling).* The P21.2 carve-brush fix and
> P21.3's sculpt sibling were two thirds of a rule; the audit found the rest. **A gizmo drag opens
> `"Move"` and commits it on release, both inside the tool-gated select branch** — so *hold a
> translate handle → Ctrl+Shift+P → `tool.sculpt` → release* left one unmatched
> `begin_transaction`, after which every later begin/commit pair bounces the nesting depth
> 1 → 2 → 1 without closing, every edit folds into the stranded entry, `undo_len()` stops growing
> and **Ctrl+Z is silently dead for the rest of the session** while edits land, the document goes
> dirty and saves work. Four settlers now exist and the pump calls all four
> (`settle_orphaned_{carve,sculpt,foliage,transaction}`), plus two rules that are not settlement:
>
> * **a document swap ABANDONS rather than settles.** `scene_open`/`scene_new` replace `*doc`
>   under the lock and only `clear_streams` is notified; a settler waking up afterwards would
>   faithfully commit the *old* level's terrain stroke into the *new* document, where one Ctrl+Z
>   applies it. `SceneDoc::doc_id` (a monotone per-instance counter) is how the viewport thread
>   notices, and the abandon is gated to run **before** every settler.
> * **the render loop's panic exits settle on the way out.** A caught panic ends the loop but not
>   the process — the document survives with the mid-drag edits in it and no `EditCommand`
>   describing them.
>
> *M11, actually finished.* `spoil_choice` moved to `SpoilMode::choice` in `camera.rs` (not
> `#[cfg]`-gated, so every CI leg runs it), the tunnel's inline capsule chain became
> `voxel_tool::tunnel_shapes`, and the terrain brushes' resampler became
> `voxel_tool::dab_centers_2d_capped` — **which also gave them the per-frame cap they never had**;
> only the carve brush was bounded. And `orphaned_strokes.rs` is honest now: it passed with
> `settle_orphaned_sculpt` deleted, because it gates the recorders rather than the settler, so
> `viewport_pump_mirror.rs` gained `every_cross_frame_gesture_has_a_settler_and_the_pump_calls_it`
> (verified: deleting the settler fails it) and the file says which half it is.
>
> *Preview == commit.* Two divergences closed rather than ledgered: the box drag's rubber band is
> now drawn from the **resolved plan** (its floor is below the lowest ground it spans, not between
> the two corner picks), and the dig-to-grade brush draws its column instead of a sphere at depth.
> Both read the same Ring-1 function the commit does.
>
> *Laws this batch paid for.* **Conservation can hide a bug rather than catch one** — a heap in
> its own pit and a cut over unpaged rock both balance perfectly, so an identity gate needs a
> *placement* gate beside it. **A `.take(n)` is a filter, not a bound.** **The dig path pages its
> footprint before it reads it** — camera residency never decides what a committed edit contains,
> now in its third phase. And **a fixture whose extremes are its corners cannot test a probe**:
> the seeded picks already found them. And, from the ruling: **an unmatched `begin_transaction`
> is not a leak, it is session-wide undo death** — every gesture that holds state across frames
> owes a settler, and every settler owes a pump call above the branch that would have run it.
- **P21.4 Runtime carving** — 1. the same ops as Blueprint nodes, deterministic and
  replay-gated, so games can dig at runtime; 2. physics and nav updates on carve.
  *(1 shipped in full — the `voxel.*` kit over one Ring-0 rule, gated by
  `VoxelVolume::runtime_carve`, with the heightfield coupling sim-local. For 2, the
  **physics** half shipped as per-chunk trimesh colliders on the P19.5 change-stamp
  pattern; the **nav** half has nothing to update, because this engine has no
  navigation system — see the P21.4 status block, which says so rather than
  quietly dropping the word.)*

> **P21.4 owes (carried from the P21.3 audit rounds).**
>
> * ~~**The `phase21_gate` must assert a dig over a COLD region counts everything.**~~
>   **CLOSED in P21.4** — `phase21_gate::a_dig_over_a_never_paged_region_counts_and_conserves_everything`,
>   against a genuinely cold `MemoryChunkStore` over the shipped `.inf_voxel`.
>   Original text: P21.3 fixed the
>   paging (`carve_into`/`spoil_into` page before they write) and gated it at call-site level, but
>   the *end-to-end* claim — that `removed[m]` over an unpaged chunk equals `removed[m]` over a
>   paged one, through a real cook and a real player — is verified in Ring 1 only. The gate P21.4
>   owes anyway (M9: voxel ground has zero PIE == shipping coverage) is where that belongs.
> * ~~**Make a big dig incremental.**~~ **PARTLY CLOSED in P21.4** — the re-mesh is
>   off the lock (12–16 % of a big dig's lock time, measured; see the P21.4 status
>   block for the numbers and the bench). The cut and the spoil search stay under
>   the guard because they *are* the edit, so a dig at the ceiling is still not
>   interactive. Original text: A dig at the `MAX_DIG_SAMPLES` ceiling spends ≈1.3 s under the
>   shared-volumes lock (4 M samples ≈3.1 s) because the cut, the spoil search and the re-mesh all
>   run there. Moving the re-mesh off the lock is the fix; it is a change to the store's threading
>   and did not belong in P21.3.
> * **`LNK1102` / "crate X required to be available in rlib format" now has THREE causes**, and
>   `df` only explains one of them: disk-full (P4, P20, cited three times), a corrupted incremental
>   cache after hand-editing `target/debug/build/*` (P21.3 — fixed by
>   `rm -rf target/debug/incremental` plus a targeted `cargo clean -p`), and **linker/rustc OOM
>   under parallel jobs** (P21.3 re-audit at `-j2`, this batch at `-j4`, both with >100 GB free —
>   fixed by lowering `-j`). Check free space *first*, but do not stop there.

> **P21.2 CARRIED LEDGER (from the P21.2 audit round — deferred, not lost).** Each item below
> is a named obligation on a later batch of this phase, recorded here at the moment it was
> found rather than at the moment someone trips over it.
>
> * ~~**M9 — voxel ground has ZERO PIE == shipping coverage.**~~ **CLOSED in P21.4**
>   — `ScenePayload` v5 carries the `.inf_voxel` **and** the `.inf_terrain` (the
>   P16.3b2 deferral, closed with it because a hole mask only persists on an
>   asset-backed terrain), and `phase21_gate::pie_equals_shipping_on_the_runtime_carve`
>   compares the traces as `f64::to_bits` over `samples/phase21-cavern`. Original
>   text: **voxel ground has ZERO PIE == shipping coverage, and P21.4's gate must cover it
>   explicitly.** Every phase since P9 has closed on a gate that runs the same scene in PIE and
>   in the shipped player and compares the traces; P21 does not have one yet. The combined
>   ground query (`inf_voxel::ground_height_at`, the "terrain where solid, topmost voxel surface
>   where holed" rule) is wired into **both** `terrain.height_at` host arms precisely so preview
>   and shipped agree — and nothing anywhere asserts that they do. So P21.4 owes a
>   `phase21_gate` plus a committed **cave sample**: a level with a carved mouth a character
>   walks into, driven through Simulate and through the cooked pack, with the ground-height
>   trace compared. Until that exists, "preview == shipped" for voxel ground is a design
>   intention and not a checked property, and this line says so rather than letting the mirrored
>   `ground_height_at` call sites imply otherwise.
> * ~~**M11 — `voxel_target` / `voxel_pick` / dab resampling belong in Ring 1.**~~ **CLOSED in
>   P21.3.** All three moved into `inf_editor_core::voxel_tool` — `voxel_target` (selection first,
>   an unloaded volume is not a target), `cut_center` (the surface → depth rule) and `dab_centers`
>   (which now **wraps `inf_terrain::dab_positions`** rather than carrying a hand copy of its
>   spacing/carry semantics). The host keeps the pick and the store lookup, which are the two
>   halves that genuinely need the platform and the mutex. Tested in Ring 1, so every CI leg runs
>   them.
> * **Coarse-LOD holes do not propagate into the pyramid** (the M7 remainder).
>   `inf_terrain::pyramid::downsample_block` reduces heights, biome ids and erosion data maps
>   and carries **no hole mask** upward, so a coarse page is hole-free however carved the
>   level-0 block under it is. Two shipped consequences: a clipmap ring far enough out to draw a
>   coarse page **draws ground over a cave**, and `RenderTerrain::seam_sample` — which reads the
>   residency floor so that lighting cannot depend on camera history — cannot apply the poison
>   rule on a streamed terrain either (hence `seam_holes_are_known` and `apply_seam`'s mask-free
>   veto). Pinned by `a_coarse_page_carries_no_hole_mask`, so flipping it is deliberate. It is a
>   **decision**, not an oversight: a coarse sample covering four fine ones could be holed if
>   *any* child is (a mouth grows by a whole coarse sample), if *all* are (it vanishes until it
>   is 2ⁿ samples wide), or by majority like biome ids — and each answer draws a different
>   distant silhouette.
> * ~~**N2 — the three terrain brushes can still strand a stroke, exactly as the carve brush
>   could.**~~ **CLOSED in P21.3.** `EngineHost::settle_orphaned_sculpt` is
>   `settle_orphaned_carve`'s sibling: the pump calls it above the tool-gated branches with the
>   document in hand, and it commits an in-flight `DragStroke` — height, splat *or* biome — when
>   the branch that would have finished it no longer runs. Two gates, because a host cannot be
>   constructed in CI: `tests/orphaned_strokes.rs` pins that a mid-drag commit of each of the
>   three kinds is one undo entry that Ctrl+Z fully reverts and Ctrl+Y fully replays (plus a
>   non-vacuity test that the three really move three different tile layers), and
>   `viewport_pump_mirror.rs` pins both that the pump calls both settlers and — positionally —
>   that they run **before** `if sculpting {`, which is the one way the fix regresses.


> **P21.4 STATUS (complete).** Gameplay digs, and the ground opens.
>
> *The kit.* `voxel.*` joins the palette: three exec actions — `carve_sphere`,
> `carve_box`, `fill_sphere` — and two pure queries, `is_solid` and
> `ground_height`. A carve reports the volume it moved in **cubic metres**, an
> exact integer sample count times `voxel_size_m³`, because the units doctrine
> says SI everywhere and a raw voxel count means something different the moment an
> author re-authors the same cave on a finer grid; divide by `voxel_size_m³` to get
> the count back. `voxel.ground_height` is the **voxel half alone** and says so on
> the node — `terrain.height_at` remains the combined query a character controller
> wants, and two names for one number would have been the worse choice.
>
> *One rule, in Ring 0.* `inf_voxel::runtime_carve` is the whole gameplay carve:
> permission, shape validity, the per-step ceiling, then `apply_op_into`. Both
> hosts call it, for the reason `ground_height_at` is one function. The editor's
> dig transaction does not survive into a fixed step — there is no undo entry, no
> author to refuse to, no toolbar to report a verdict on — so the smaller rule is:
> a **pure function of the op and the volume**, no camera and no session history;
> every refusal decided **before the first sample moves**; and **idempotent**,
> because `VoxelOp` is, which is what makes a replayed step after a rollback land
> on the same bytes.
>
> *Four refusals, all answering `0.0` and all logged.* No volume on the entity;
> `runtime_carve` off; a degenerate shape; more than `MAX_RUNTIME_CARVE_SAMPLES =
> 65_536` samples in one call (the editor's ceiling is 2 M and costs ≈78 fixed
> steps, so a gameplay carve gets one two orders of magnitude smaller — a grenade
> crater rather than a quarry, stated as a **sample count** because the same radius
> is a different bill on a different grid). A blueprint node is not a transaction:
> failing the handler would take down the rest of the Tick body — the movement, the
> animation, the sound — for an op the author fixes by typing a smaller radius. The
> flag frozen into scene schema v19 finally has its reader, and it behaves the way
> its doc comment promised: **refused and reported, never silently applied**, so a
> replay cannot diverge on whether some node happened to run.
>
> *The heightfield half is sim-local, and that changes which refusals apply.* A
> carve that crosses the surface opens a mouth through the same
> `apply_surface_cut` the editor brush runs, over every terrain in `Guid` order.
> But **nothing is persisted**: the editor's Simulate world is a
> `ScenePersist::Memory` snapshot and the player's is a loaded pack, and both die
> with the session. So the inline-terrain refusal — which exists to stop an author
> *saving* a document whose mask cannot survive — does not apply at runtime, and a
> game may carve any terrain. Craters last exactly as long as the play session; a
> save system is P22-and-later and is not half-built here.
>
> *Physics on carve, and nav honestly.* `PhysicsBridge3D::sync_from_world_with_voxels`
> gives every chunk of `mesh_keys_for` a static trimesh collider built by the
> **same** `mesh_chunk` the renderer draws with, stamped per `(entity, chunk key)`
> on `inf_voxel::source_key` — the P19.5 `structure_stamps` pattern one level
> finer, because a cave is hundreds of chunks and a gameplay carve moves two. A
> buried chunk meshes to nothing and costs no collider; a chunk carved hollow
> *loses* its collider on the next sync, which is how a runtime carve becomes a
> hole a body falls through.
>
> **Both of those nouns were wrong in the first cut, and both failed silently.**
> The walk was `resident_keys()` while the renderer meshes `mesh_keys_for` — the
> resident set *closed downward by one chunk on each axis*, which `inf_voxel::mesh`
> calls "a correctness requirement, not defensive padding" — so 37 % of the
> flagship sample's drawn surface had no collider at all (10 936 triangles drawn
> against 6 874 collidable; the whole −X/−Y/−Z faces were walk-through). And the
> stamp was the chunk's own `chunk_version`, which cannot see a neighbour being
> carved or evicted: **the exact M3 defect P21.1 paid for in the renderer,
> re-introduced in the physics bridge**, and reproduced as a phantom wall at
> x = 15.50 over a doorway carved wholly inside the next chunk. Using the mesher's
> own key set and the mesher's own stamp is also what makes the `inf-physics`
> manifest's "the SAME mesher the renderer draws with" true rather than
> aspirational.
>
> The cost of doing it correctly is worth a number: `source_key` is a max over the
> 3×3×3 neighbourhood, so carving **one** chunk moves the key of that chunk and its
> 26 neighbours, and each of the 27 is re-meshed and re-described. That is the price
> of a mesh being a function of its neighbourhood — the alternative is the stale
> seam — and it is bounded and local: 27 against a cave system of hundreds, and
> zero on every step that digs nothing.
>
> **There is no navigation system in this engine** — not a nav mesh, not a nav
> volume, not a path query — so the plan bullet's "nav updates on carve" has
> nothing to update. Said here rather than quietly dropped.
>
> *The v5 fields live at the TAIL, and the envelope is version-checked.* They were
> first written **between `biome_sets` and `tick_hz`**, which is the bincode
> positional law broken in the one way that does not announce itself: two empty
> `Vec`s are two zero bytes, and two zero bytes are a perfectly valid `u32` + `bool`
> — so a stale player decoded `tick_hz = 0`, reported `Loaded`, and Play silently
> did nothing. Every envelope test in the tree used a payload whose new vectors were
> **empty**, which is exactly the shape that cannot tell a mid-struct insertion from
> a tail one. The fields are now after `windowed` with an append-only comment on
> the seam, `ScenePayload::check_version` refuses a mismatch at the one place every
> consumer passes through (nothing read `schema_version` at all before this), and
> the round-trip test carries **non-empty** voxels and terrains. `write_msg` also
> stops truncating its `u32` length: a payload now carries whole assets
> uncompressed, so frame size is a function of the author's content and
> `MAX_FRAME_LEN` is a bound a real level can reach.
>
> *The PIE voxel source, and the P16.3b2 deferral closed with it.* `ScenePayload`
> v4 → v5 appends **`voxels`** and **`terrains`**. The first was the M9
> prerequisite: before it the PIE player had no `.inf_voxel` at all, so a carved
> cave answered `terrain.height_at` with the seam's `0.0` in preview and with the
> cave floor in the shipped build — and any gate comparing them would have compared
> **two empty maps agreeing**. The second was forced by P21.2's own design: a hole
> mask only persists on an asset-backed terrain (v19 pins `TerrainTileFrozenV3`,
> which has no hole rows), and `strip_streamed_terrain` blanks a streamed terrain's
> working set on the way to the wire — so "a level with cave mouths" and "a level
> PIE can preview" were mutually exclusive until the `.inf_terrain` bytes crossed
> too. `terrain_source_from_bytes` + `TerrainContent::Memory` is the whole
> mechanism; the store is the one the dev-dir path already used, with the read
> already done.
>
> *The shipped player must SEE what it carves (the sim→render fold).* Two seams
> were missing, and they fail together: the render voxel store never read
> `sim.voxel_volumes()`, and `TerrainStreamer::pin_tile`'s only production caller
> was the **editor**. So a game dug, the collider opened, gameplay walked in — and
> the screen kept drawing the rock, with the mouth sealed over on any asset-backed
> terrain, which is the only kind that can carry a hole mask and therefore the
> configuration every carved level ships in.
>
> `VoxelVolumes::overlay_sim` and `TerrainStreaming::overlay_sim_edits` close it,
> both **`sim → render` only**. That direction is the legal one and it is worth
> saying why, because the phase is full of the opposite rule: camera residency must
> never reach the simulation or the lighting, and nothing here goes that way — the
> simulation is authoritative and the renderer projects from it, exactly as it does
> for every `Transform` in the world.
>
> **The first version of the fold was wrong three times, and the third audit caught
> all three.** Each is worth the space, because each is a shape rather than a typo.
>
> *It baselined away the first carve.* The rule was "a key seen for the first time
> records the sim's stamp and copies nothing — both sides came from the same asset
> and already agree". They do not, because *first sight happens after the first
> step*: both hosts bind their store after a frame has run (the editor folds after
> `session.tick`, the player binds inside a render sync that runs after the first
> frame), so the stamps being recorded were stamps a carve had **already moved**. A
> **one-shot dig on the first Tick** — the pattern the node kit itself prescribes,
> since `BeginPlay` cannot see a volume — was therefore invisible for the rest of
> the session: the sim had the hole, the colliders had the hole, gameplay walked
> through it, and the screen drew solid rock. A *continuous* borer masked it (tick 2
> repaired tick 1), which is exactly why the flagship sample did not catch it. The
> rule is now **dirty-driven**: `sim_volume` clears the dirty set on load and the
> simulation never writes back, so a dirty sim chunk means precisely "gameplay
> carved this". An undug level still copies nothing — the property the baseline
> existed for — without inventing a moment before which carves do not count.
>
> *It dirtied what it copied.* The copy went in through `insert_chunk`, leaning on
> the dirty flag as a free eviction pin. In the editor **this store is the one a
> save stages from** (`SceneDoc::save` → `stage_voxel_edits` → `write_voxel_edits`),
> so Simulate → dig → Stop → Ctrl+S would have written a player's runtime craters
> into the author's `.inf_voxel` — flatly contradicting the rule three paragraphs
> down, that nothing a game digs is persisted — and `has_unsaved_edits` would have
> stayed true after a clean save, telling the author they had unsaved caves they
> had never carved and never releasing the crash-recovery file. It now copies
> through `insert_resident_chunk`, which stamps and does **not** dirty.
>
> *And the pin never released.* Holding carved chunks resident for ever grew the
> resident set without bound: a camera a thousand kilometres away still kept every
> carved chunk meshed and uploaded for the life of the session. There is now **no
> pin at all.** The overlay runs *after* the camera pass and simply runs again: a
> chunk residency evicted and later paged back in arrives as the asset's pre-carve
> bytes with a fresh stamp, which is exactly what the overlay re-copies on. So
> residency stays entirely the camera's business and the carve is re-applied on top
> of whatever it decided, as often as it takes. `overlaid` records **both** stamps —
> the sim's and the one this slot held afterwards — because recording only the sim's
> would leave the overlay convinced it had already done its work while the store
> drew rock.
>
> It also checks the **whole lattice and the asset id**, not just `voxel_size_m`: an
> author can re-point `VoxelVolume.asset` in the Details panel mid-Simulate, and
> without the check asset A's chunks were copied into a slot bound to asset B (and,
> before the dirty fix, written into B's file).
>
> The terrain half is the editor's `overlay_document_edits` with the sim world in
> place of the document, and it needed the same lesson. The editor pins every dirty
> tile and releases them all on save; a **player never saves**, and a runtime hole
> mask is never cleared, so the pin set only grew. Past
> `StreamBudget::max_resident_tiles` that is not a memory cost but a **stall**:
> `pin_ceiling` clamps the camera's cut to `.max(1)` once pins fill the budget, and
> the terrain silently stops streaming around the player. The player's pin set now
> follows its **cut** — pin a dirty tile inside it, release one that leaves — so it
> is bounded by the cut, which is bounded by the budget. `inf_terrain::stream`'s
> claim that "pins only ever exist in the editor; the shipped player never pins" was
> falsified by this batch and is corrected in place, since it is the sentence that
> made the unbounded case unthinkable.
>
> **PIE ships the SAVED cave, and the reason is stronger than symmetry.** Editor
> *Simulate* does carry unsaved carves (`overlay_unsaved_carves` folds the editor
> store's **dirty** chunks over the resolved map, safe because dirty is a function
> of edit history and `sync_residency` refuses to evict a dirty chunk). Shipping
> that store *as a volume* over the wire is a different act: the store is
> **camera-paged**, so a PIE session built from it would preview a cave truncated
> by where the author happened to be looking — precisely the dependency every seam
> in this phase is shaped to forbid. The fix that would work is the dirty-chunk
> **overlay** (ship the saved bytes plus `(entity, chunk key, chunk bytes)` and let
> the player apply the same rule), and it is ledgered below rather than half-built.
>
> *`sim_from_payload`, and a drift it closed on the way past.* Every PIE boot path
> now goes through one function that makes every attachment a session needs. It had
> to, because the failure is silent — a path that forgets an attach does not crash
> and does not warn, it runs a world whose caves are not there. Building it found
> that the real `--pie` subprocess had been constructing its sim with a bare
> `RuntimeSim::new` rather than `sim_from_built`, so it was **also** dropping the
> state machines, the root-motion clips and the audio clips.
>
> **What that did and did not mean, stated precisely** (the first draft of this
> block overclaimed it, and the audit was right to say so). It did *not* mean every
> gate since P11 was comparing two different worlds: those gates build **both**
> sides in process, and the levels they run carry no state machine or audio that
> the subprocess would have had to drop — the subprocess had nothing to lose, so
> nothing diverged and nothing was hidden. What it meant is that the *capability*
> was missing from the shipped `--pie` path and **no gate could have told us**,
> because none of them ran it. That is the same shape as this batch's own law, one
> level up: a boot path nobody exercises does not fail, it simply is not there. The
> fix is `the_real_pie_subprocess_matches_the_in_process_reference`, which spawns
> the actual binary — and which is mutation-proved: reverting the seam fails it.
>
> *The gate is `samples/phase21-cavern` + `runtime/inf-player/tests/phase21_gate.rs`.*
> The plan's done-when sentence as committed content — a 128 m ridge on an
> asset-backed terrain, a carved cave system with a real mouth, an excavated
> foundation pit with its exactly-conserved spoil heap, an underground room under
> the pit joined by a shaft, a borer that keeps digging, and a **boulder** resting
> on the rock the borer is about to remove — with **thirteen** arms: two loads
> bit-identical including the field and the hole mask; **the carve is real** (the
> post-run field differs from the seeded asset where the drift runs and is
> byte-identical everywhere else); **the collider world opens** (a ray down the
> bore stops metres lower after it, and the boulder rests on rock, then falls into
> the trench and lands on its floor); **the shipped player sees it** (the render
> voxel store mirrors the carved chunks and the render terrain streamer pins the
> holed tiles); **the real `--pie` subprocess** matches the in-process reference;
> cooked == uncooked *on the field as well as the trace*; **PIE == shipping on the
> runtime-carve trace, as `f64::to_bits`** (the M9 debt); the workings surviving a
> round trip byte-identical with the room reachable through the combined query
> *and* through `voxel.ground_height`; `runtime_carve` gated **both ways** on one
> world with one flag flipped; a **silent** cook; both budget classes; a subprocess
> pool gate at 1/2/4/8 comparing the trace **and the resulting field**; and the
> cold-region count below.
>
> **Six of those arms exist because the first version of this gate certified a
> no-op.** The audit mutated `runtime_carve` to apply the op to a *clone* — correct
> cubic metres reported, real field never touched — and the gate stayed 10/10
> green (and ran ten times faster, because nothing was being dug). A second
> mutation deleted `gather_voxels` outright and it stayed green again. Nothing
> compared the **field**, and nothing asked the **solver** what it contained; every
> number the gate read was downstream of a report rather than of the world. Both
> mutations now fail — the first on five arms, the second on four — and so does
> reverting the `--pie` boot seam. The lesson is recorded in the laws below.
>
> It caught its own first defect too: the borer's IR used `math::add`, which is a
> **binary op** and not a `dispatch_math` builtin, so the handler errored and the
> trace was 160 ticks of zero. The anti-vacuity arm said so in one line.
>
> *The cold-region count, paid.* P21.3 fixed the paging and gated it in Ring 1;
> the end-to-end claim — that a dig over a region **no camera has ever paged**
> removes and conserves exactly what the same dig over a warm region does — is now
> asserted on the shipped `.inf_voxel`, against a genuinely cold `MemoryChunkStore`
> whose volume starts at zero resident chunks. The failure it exists to catch is
> invisible from inside: a non-resident chunk reads as air, so the cut removes
> nothing there, counts nothing, spoils nothing — **and conservation balances
> perfectly**.
>
> *Off-lock re-mesh, with honest numbers.* `SceneDoc::edit_dig` — the one door the
> box cut, the trench, the tunnel and the brush's committed stroke all arrive at —
> now commits with the re-mesh **deferred**: the chunk versions move inside the
> transaction (they *are* the edit), and the meshes are rebuilt by the viewport's
> next projection, which runs whenever the document version moves and which a carve
> always bumps. Measured (release, one machine, 108 chunks at 0.5 m, a 405 000-sample
> box cut, `editor/crates/inf-editor-core/tests/dig_stall_bench.rs`): **86.1 ms →
> 72.1 ms** under the shared-volumes guard with spoil discarded, and **≈224 ms →
> 197.8 ms** with `SpoilChoice::Auto`; the 11.3 ms / 26.8 ms of meshing moves to
> the render thread. So the re-mesh is **12–16 %** of a big dig's lock time and is
> the only part with no reason to be there — the remaining 84–88 % is the cut and
> the spoil search, which *are* the edit and cannot leave. **A dig at the ceiling
> is still not interactive.** That is the honest conclusion, recorded rather than
> rounded up, and the `#[ignore]`d bench that produced the numbers is committed
> beside it. The live **brush** is untouched: `CarveStroke::begin` still re-meshes
> every dab, which is what makes a drag look like digging.
>
> *Laws this batch paid for.* **A refusal must be a value, not a failure** — a
> node that errors takes its whole handler down, so a gameplay op's refusals are
> zero-and-a-log. **A camera-paged store may be read but never shipped**: the
> difference between the Simulate overlay (dirty-gated, edit-history-driven) and
> the PIE payload (the whole volume, residency-driven) is the whole reason one is
> legal and the other is not — while `sim → render` is the *permitted* direction
> and had to be built, because a player that cannot see what it carves is not
> shipping the feature. **`math.add` is a binary op, not a math builtin** —
> hand-written IR that says `math::add` fails its handler silently, and a
> zero-forever trace is what that looks like from outside.
>
> And the four this batch's audit paid for, which are one law seen from four sides:
>
> * **A boot path that forgets an attachment does not crash — it agrees with
>   itself.** A gate whose two sides are both built in process compares one function
>   against itself; at least one arm has to cross the real process boundary.
> * **Assert the WORLD, not the report.** Every number the first gate read — cubic
>   metres, ground height, entity count — was downstream of a report, so a carve
>   applied to a clone satisfied all of them. The field is what a carve *is*, and a
>   gate that never compares it certifies a no-op.
> * **A mesh is a function of its neighbourhood, so its identity is `source_key`
>   and its key set is `mesh_keys_for`.** Written down in `inf_voxel::mesh` after
>   P21.1 paid for it, and re-learned in the physics bridge five months later —
>   which is the argument for using the mesher's own two functions rather than
>   re-deriving "which chunks, and when did they change" per consumer.
> * **An `#[serde(default)]` on a bincode struct buys nothing.** Positional means
>   positional: new fields go at the tail, the version is checked at decode, and the
>   round-trip test carries a **non-empty** value of the new field — an empty `Vec`
>   is two zero bytes and will decode as whatever follows it.
>
> And three more from the round after that, all about the same fold:
>
> * **"First sight" is not "the beginning" in any system that binds lazily.** A
>   baseline recorded when a cache first sees a key is a baseline recorded after
>   however many events already happened — and it silently swallows every one of
>   them. Key on a fact that *means* what you need ("this was carved") rather than
>   on the observer's own history.
> * **In the editor, the render store IS the save's staging source.** Anything
>   written into it for display is a candidate for the author's asset file on the
>   next Ctrl+S. `insert_resident_chunk` stamps; `insert_chunk` stamps *and*
>   schedules a write-back, and the difference is a player's craters in an author's
>   cave.
> * **A pin with no release is a leak with a deadline.** The voxel one grew the
>   resident set for the life of a session; the terrain one grew until it hit
>   `pin_ceiling` and silently stopped the world streaming. Re-applying from the
>   authoritative side is cheaper and bounded — and needs no policy at all.

> **PHASE 21 SHIPPED LEDGER.**
>
> * **P21.1** — `ccbc348` (SDF chunk store, deterministic Surface-Nets mesher,
>   `.inf_voxel`, scene schema **v19** + `VoxelVolume`, the renderer's voxel path),
>   `a027513` (audit: engagement-counted off-path, editor bind parity,
>   eviction-aware mesh keys).
> * **P21.2** — `bab8596` (per-sample hole mask; `.inf_terrain` **v5**),
>   `65d2876` (clipmap discard, seam blending, the combined ground query, camera
>   residency), `f6a60fb`, `f7548b0` (hole-coupled carve as one undo step + the
>   inline-terrain refusal), `b7260a4` (carve brush + spline tunnel),
>   `a4e5844`, `a4fc002` (write-back + reload), `2ee5b0b` (toolbar + verdict),
>   `8504e75`, `b3fe2a8`, `c33bde8` (three audit rounds).
> * **P21.3** — `bd1e3fa` (excavation: box/trench cuts, exact-conservation spoil),
>   `1ec70b5` (tool policy to Ring 1; orphaned strokes settle), `cf2ed56` (audit:
>   paged digs, sky-rule trenches, bounded spoil growth), `8ab04e0` (stranded
>   transactions settle), `cdd356d` (a document swap abandons rather than settles).
> * **P21.4** — `31144dc` (runtime carving: the `voxel.*` kit, the Ring-0 rule,
>   chunk colliders on carve, the PIE voxel source), plus this batch's gate +
>   sample + completion block.
>
> **Schema: one bump, v18 → v19**, spent in P21.1 on `VoxelVolume`'s three fields
> and never touched again — `.inf_voxel` and `.inf_terrain` version themselves, and
> `ScenePayload` v5 is a **wire envelope** bump that changes nothing on disk and
> re-blesses no golden. Goldens stand at **49** (P21.1's `voxel` and P21.2's
> `cave_mouth`); this batch added none, because every claim it makes is structural
> and a screenshot could not have carried one of them.
>
> **THE PHASE'S REMAINDER LEDGER, swept across all four batches.**
>
> *Carried from P21.2, still open.* **Coarse-LOD holes do not propagate into the
> pyramid** (`pyramid::downsample_block` carries no hole mask upward), so a clipmap
> ring far enough out draws ground over a cave and `RenderTerrain::seam_sample`
> cannot apply the poison rule on a streamed terrain. Pinned by
> `a_coarse_page_carries_no_hole_mask` so flipping it is deliberate. It is a
> **decision, not an oversight**: a coarse sample covering four fine ones could be
> holed if *any* child is, if *all* are, or by majority, and each answer draws a
> different distant silhouette.
>
> *New, carried out of P21.4.* **The render fold is stamp-gated, not free**: a
> carve copies its own chunks into the render store the frame it changes them and
> whenever residency has paged the asset back over them, and a volume nobody digs
> costs one lookup per **dirty** sim chunk per frame — which is zero until something
> digs. **A carved chunk is re-copied after every eviction round trip**, so a camera
> oscillating across a carved region pays the copy repeatedly; that is the price of
> not pinning, and it is bounded by the camera's own residency churn.
> **The editor's Simulate fold runs from Ring 2** (`commands/sim.rs` overlays the
> session's map into the viewport store after each tick) rather than from the
> viewport pump, because the pump has no reference to a `SimSession`; it is
> correct and it is not the same call site as the player's. **Windowed PIE now
> carries its voxel assets** (it bound none before, so an embedded session drew no
> caves at all while the headless one the gate drives drew them correctly) — and
> the windowed path remains **human-verified**, as every windowed path here is.
> **A PIE payload carries whole `.inf_voxel` / `.inf_terrain` assets
> uncompressed**, against a 256 MiB frame cap that is now a real refusal rather
> than a truncated length; compressing them, or streaming them out of band, is the
> open question a level near the roadmap's ~1 GB target will force. **PIE previews
> the last SAVED cave** — the dirty-chunk overlay described above is the fix and is
> not built. **`BeginPlay` cannot see a voxel volume**: both hosts seed their map after constructing the sim
> (which is where `BeginPlay` runs), so a carve or a hole-query there refuses; the
> workaround is deferring `BeginPlay`, which changes when *every* handler in the
> engine runs, and the node kit says "put the first dig on Tick" instead. **A
> `voxel.*` node can only name the volume on its own actor**, because
> `vars::get("entity")` is the only entity reference the blueprint IR has — the
> same limit the audio and physics kits live under, and the reason the sample is a
> cavern that bores itself. **A runtime carve has no spoil**: gameplay deletes
> rock, and conservation stays an authoring guarantee about a document.
> **Nothing a game digs is persisted** (see above). **Chunk colliders are built for
> every resident chunk at load**, which is a real one-off cost on a large cave and
> is not amortized or budgeted — the change stamp makes the *steady state* cheap
> and says nothing about the first frame.
>
> *Standing, from earlier batches.* Instanced debris for spoil is **P22**'s
> (fracture/destruction), not this phase's. There is **no voxel raycast picking
> from inside a cave** — the viewport picks against the heightfield and the ID
> buffer. **macOS viewport input is unwired**, so every voxel gesture there is
> compile-checked and not driven. The sculpt-side items P21.2 named
> (per-material brush falloff, a hole-aware sculpt undo preview) are unbuilt.
> Viewport interaction is **human-verified**, as every native-viewport gesture in
> this repository is: CI creates no window. The *logic* is not — every edit, every
> verdict and every preview is a Ring-0/Ring-1 function with its own tests.
>
> *Dependency hygiene.* P21.4 added `inf-voxel` to `inf-physics` (the chunk
> colliders) — Ring 0 → Ring 0, with its reason in the manifest — and nothing else.
> **No new third-party crate entered the tree for the whole phase**, and
> `cargo deny` is unmoved across all four batches.
>
> *The `LNK1102` note, since this phase is where it grew.* "crate X required to be
> available in rlib format" now has **three** causes and `df` explains only one:
> disk-full (P4, P20), a corrupted incremental cache after hand-editing
> `target/debug/build/*` (P21.3 — `rm -rf target/debug/incremental` plus a targeted
> `cargo clean -p`), and **linker/rustc OOM under parallel jobs** (P21.3 at `-j2`,
> its re-audit at `-j4`, both with >100 GB free — fixed by lowering `-j`). Check
> free space *first*, but do not stop there.

> **STATUS: Phase 21 COMPLETE** (2026-08-04) — **local gates green; CI pending
> push.** (Written with the commit rather than after the CI run, like Phases
> 16–20's, and saying so rather than implying a green run that has not happened.)
>
> **The four batches, in one line each.** **P21.1** gave the engine geometry a
> heightfield cannot express — sparse SDF chunks, `f64`-anchored like terrain
> tiles, meshed by a Surface-Nets pass whose seams are watertight by construction
> and whose output is byte-identical however the chunks arrived. **P21.2** made
> them part of the *world*: a per-sample hole mask on the tiles so a cave has a
> mouth, one combined ground query wired into both hosts so gameplay stands on a
> cave floor, and the carve brush and spline tunnel that author it. **P21.3** made
> them **excavatable** — box and trench cuts under a sky rule, and material
> accounting whose spoil heap holds the excavated count exactly, per material, as
> integers, with no bulking factor. **P21.4** handed the whole thing to gameplay:
> Blueprint carve nodes gated by `runtime_carve`, chunk colliders that rebuild on
> a carve, the two byte sources PIE never had, and the phase gate that compares
> preview against shipping on a trace of digging.

### Phase 22 — Dynamic world: deformation & destruction

**Goal:** the world reacts — surfaces deform, assets and buildings break. **Done when:** a
playground scene shows footprints and tyre tracks in snow and sand, bending grass, and a car
and a multi-storey building destroyed by Blueprint-triggered explosions with debris physics —
deterministic on the replay trace, PIE == shipping.

- **P22.1 Surface deformation (snow/sand/grass/mud)** — 1. world-space deformation
  height/offset maps around active actors, compute-written into a camera-following ring buffer;
  2. sampled by the terrain displacement and foliage bend shaders; 3. per-splat-layer response
  params (depth, recovery time, hardness); 4. accumulation hooks from P17 weather (snowfall
  refills the surface); 5. deterministic fixed-step writes for committed content.
- **P22.2 Fracture pipeline** — 1. cook-time (later DCC-time) Voronoi pre-fracture of meshes
  into chunk hierarchies — `.inf_fracture` beside the mesh, deterministic seeds;
  2. material-derived strength params (density, Young's-modulus-class scalars — the DCC spec's
  derivation, simplified) on physics materials.
- **P22.3 Runtime destruction** — 1. damage events and a Blueprint kit swapping the intact mesh
  for chunk bodies through the existing rapier bridges; 2. a **structural-integrity graph** for
  buildings — support-chunk removal drives progressive collapse, solved deterministically at
  fixed step; 3. debris lifetime and budget caps with despawn; 4. audio/VFX event hooks.

> **P22.3 STATUS: runtime destruction is BUILT** (2026-08-05) — local battery
> green (191 test binaries, `clippy -D warnings`, `wasm32` player), CI pending
> push.
>
> **The prerequisite nobody had noticed.** The terrain heightfield had **no
> physics representation at all**: `terrain.height_at` answered a query and the
> character mover read it, so a scripted character stayed grounded while a
> *dynamic* body fell through the world. Destruction cannot be built on that —
> debris that never lands is not debris. So a fourth gather in `PhysicsBridge3D`
> turns every **sim-resident** level-0 tile into a static
> `ColliderShape3D::Heightfield`, change-stamped so an unsculpted level pays one
> build at load. Holes are honoured on **exactly** `TerrainData::height_at`'s
> poison rule (any holed corner removes the whole cell), stated at both sites and
> compared cell-by-cell by a test — so the visible gap, the queryable gap and the
> walkable gap are one gap, and what falls through a cave mouth lands on the P21
> voxel floor.
>
> **The destruction itself.** `inf_physics::d3::fracture` owns the intact→chunks
> swap with **no new entities**: chunks reach the solver as synthetic
> content-derived guids, the third time this repository has taken that decision
> and the third time it was right. Damage is an **energy in joules** spent
> breaking bonds at `strength × area × 1 mm` — the strength memo's force contract
> times one crack-opening displacement, `Pa·m²·m = J` with no invented constant.
> Support is decided exactly as `docs/memos/p22-strength.md` §4.3 committed:
> runtime contact with static geometry, `AlwaysLoaded` as the explicit per-entity
> override, propagation through undetached neighbours. **Scene schema v20 did not
> move**, which is what that memo existed to buy.
>
> **Two design errors the tests found, and they are worth recording.**
> (1) *The obvious bond-area measurement is wrong and fails silently.* "Sum the
> triangles on the bisector of the two chunk centres" is not the Voronoi face: a
> face bisects the two **sites**, and a cell clipped against the source hull has a
> centroid that is not its site. Every bond of a real cook asset found nothing on
> the plane and fell through to the fallback estimate — a whole building's
> energies quietly replaced, with nothing to notice it. The face is now the
> corners the two `f64` hull sets agree on. (2) *A tower's base and its top are
> equally cheap* if a chunk is only bonded to its neighbours — both have exactly
> one — so a low-index tie-break knocked the foundation out for one bond and
> dropped the lot. A chunk standing on static geometry now pays a **ground bond**
> too, over its own `volume^(2/3)`, which is both the fix and the honest physics: a
> wall is mortared to its footing.
>
> **PIE == shipping, and what it cost.** A `.inf_fracture` is *derived at cook*,
> so it is in no content root and `ScenePayload` v5 could not carry it. v6 appends
> `fractures` at the tail — the only payload entry that is **computed** rather than
> resolved: the editor runs the same `inf_mesh::fracture_mesh` the cook runs, with
> the same authored seed, keyed by the same `derived_fracture_id`. The twin gate
> runs one fixture through both hosts and compares the fracture state as **raw
> bits**, with an anti-vacuity arm so it cannot pass by comparing two intact walls
> (the P21.4 lesson, applied before it could be re-learned).
>
> **The audit round (2026-08-05), and the ruling it turned on.** A fix-first audit
> of the three commits found one blocker and six majors, all fixed here. The
> blocker was the one that would have been felt by every player: a height field is
> two triangles per cell and, without `FIX_INTERNAL_EDGES`, a body crossing a cell
> boundary is answered with an *edge* normal — so a sphere sliding on **flat**
> ground is kicked upward with no lateral force applied to it. One flag, and the
> same fix for the P21 voxel trimesh floors that debris lands on when it falls
> through a cave mouth. (The first write-up of this quoted "46 kicks, 12 cm hop,
> 0.72 m drift" from a fixture that did not exist; the committed gates measure
> **5 kicks at 0.105 m/s** on terrain and **11 at 0.188 m/s** on a voxel floor,
> both zero once flagged, and those are the numbers that stand.) The rest: the editor's
> Simulate seeder resolved a shared mesh's chunking in ECS **archetype** order
> while the cook and PIE used document order (Simulate shattered a wall into 24
> pieces the shipped pack shattered into 8); PIE skipped the cook's own
> `convex_hull_is_buildable` refusal; a dynamic body inside the 2 cm support skin
> **hid** the ground under it, so a tower collapsed because its own rubble had
> landed beside it; the placement was frozen at seed time, so a wall moved after
> load shattered where it used to be; a one-sided adjacency edge priced at 0 J;
> and the fracture-production path's cited gate was a phantom — all 17
> `build_scene_payload` call sites passed `|_| None`, so deleting the payload's
> fractures entirely left the tree green.
>
> **THE RULING on cheapest-to-liberate.** It stands: there is no impact point in
> `destruct.apply_damage`'s signature or in scene schema v20, an origin proxy
> would detach a wall's *core* first, and fracture minimising new surface energy is
> the physics. But the consequence belongs in this ledger and not only in a doc
> block: **damage is NON-LOCAL.** A wall struck anywhere sheds its cheapest chunk,
> which may be at the far end of it. Revisit when a hit position arrives — the
> ordering is a two-line change once there is something to sort by.
>
> *Honest remainders, carried into P22.4.* **No sample scene and no `phase22_gate`
> yet** — the cross-host comparison above is a unit-level twin, not a
> subprocess `--pie` arm, and the playground the phase's "done when" describes is
> unbuilt. **Destruction is not persisted** (a save game is P22.4's), so rubble
> dies with the session — and the editor clears its published states on
> `sim_stop` precisely so a broken wall is never drawn over an intact document.
> **No VFX**: this engine has no particle system, so a break makes no dust; the
> audio hook is real and the visual one is not, and that is ledgered rather than
> faked. **Debris budgets are a `DebrisBudget` the host sets**, not a tier mapping
> — physics must never name `RenderTier`, or a fixed step becomes a function of
> the graphics settings; P22.4's "per-tier debris budgets" fills it in.
> **Instanced debris through the GPU scatter path is P22.4's**: today each chunk
> is one draw against its own buffer, which is right for tens of chunks and not
> for thousands. **Streaming and destruction do not meet yet**, on the P21 voxel
> ledger's precedent: a `Destructible` that arrives with a partition cell after
> the level was seeded answers `NoFracture` for ever (the seed is a one-shot walk,
> not a subscription), and a *broken* actor whose cell streams out leaves its
> static chunk colliders behind (the gather keys on the fracture map, which cell
> streaming does not touch). **A `destruct.*` node can only name a whole actor**, never one
> chunk (chunks are not entities), and `apply_damage` has **no impact point** —
> the order is cheapest-to-liberate first, which is the same physics without an
> invented input, and it is documented rather than papered over.
>
> *Dependency hygiene.* P22.3 added `inf-terrain` and `inf-mesh` to `inf-physics`
> (the hole rule and the chunk adjacency, read from the crates that define them
> rather than copied) and `inf-physics` to `inf-viewport`. **No new third-party
> crate.** The one knock-on: `inf-mesh` now confines its whole **import** path
> (`meshopt`, `gltf`, `image`) to `cfg(not(target_arch = "wasm32"))`, on the rule
> `inf-vgeom` already applies — measured with
> `cargo check --target wasm32-unknown-unknown -p inf-player`, not assumed.

- **P22.4 Destructible environments at scale** — 1. instanced debris through the P18.5 GPU
  instance path; 2. per-tier debris budgets; 3. destruction state persisted in the save and
  replication seams, with net-relevant events documented for the P14 net layer.


> **PHASE 22 COMPLETE — the dynamic world reacts** (2026-08-05). The phase's own
> done-when sentence is built and gated: *a playground scene shows footprints and
> tyre tracks in snow and sand, bending grass, and a car and a multi-storey
> building destroyed by Blueprint-triggered explosions with debris physics —
> deterministic on the replay trace, PIE == shipping.*
>
> **The shipped ledger, by batch.**
>
> * **P22.1 surface deformation** — `c93d974` (the sim-authoritative field:
>   `inf_terrain::deform`'s sparse dense-cell `BTreeMap` over a global lattice,
>   per-layer response archetypes, `MAX_DEFORM_CELLS` with least-recently-stamped
>   eviction, and `inf_ecs::deform`'s one Ring-0 contact rule both fixed steps
>   call), `1169f55` (the camera-following render window, terrain displacement and
>   normal perturbation, scatter bend + wind), `5b6bb0f` (the audit round: the
>   field exits with the session at BOTH ends, the wrapper halves pinned, real
>   gates). The field is a **bevy resource**, so no schema moved and nothing here
>   can be saved. CPU at fixed step, not a compute-written ring buffer: Ring-1's
>   GPU erosion is already documented as not bit-identical across adapters, and a
>   GPU-authoritative field would make a committed level's ground depend on the
>   player's card.
> * **P22.2 the fracture pipeline** — `6f85fd8` (deterministic Voronoi chunking +
>   the `.inf_fracture` asset, derived at cook under `derived_fracture_id`),
>   `c5da477` (scene schema **v20**: the `Destructible` component, and the cook's
>   `plan_fractures` derivation), `7c0a433` (watertight hulls under f32 reality,
>   chunk-indexed adjacency, honest budgets), `f8a15a1` (the `ConvexHull` collider
>   + the parry pin). Schema **v20 once**, and never again.
> * **P22.3 runtime destruction** — `be430b8` (terrain tile heightfield colliders —
>   the prerequisite nobody had noticed — plus the fracture runtime), `9158841`
>   (the `destruct.*` kit, both host arms, `Destroyed`, `ScenePayload` **v6**),
>   `07965e4` (the chunk render projection, its upload pass and the atomic swap),
>   `57c5f4e` + `93293d5` (two audit rounds). See the P22.3 status block above for
>   the full account; its own ledger is carried forward below.
> * **P22.4 destruction at scale** — `8d60548` (a P22.3 leftover: the macOS spawn's
>   shared fracture store, without which `inf-viewport` did not build on macOS at
>   all), `82bb7b7` (sub-chunk debris through the P18.5 GPU scatter path, the
>   per-tier debris budget, the persistence twin test and the net-events memo),
>   and this batch's sample + gate commit.
>
> **What P22.4 added, and the one design decision worth reading.** Rubble is
> **render-only dressing** laid by `inf_render::debris` and shipped as one
> `ScatterBatch` per broken actor. `ScatterData::key` is a content hash over the
> packed instance bytes, so a batch whose instances move re-uploads its whole
> buffer every step — the exact cost the scatter path exists to avoid. The answer
> is not a stamp: each fragment is placed around its chunk's **rest centre**, which
> `FractureState` freezes at the first detach, so the batch's bytes are a pure
> function of *which* chunks are live and re-upload exactly once per break. There
> is no stamp to get wrong because the content genuinely does not change. Every
> fragment is a pure function of `(entity, chunk, detach order, fragment index)`
> through SplitMix64 — so two hosts and two machines lay the same rubble with
> nothing sent between them, which is also why the net memo can say the rubble is
> not replication-relevant.
>
> `inf_render::debris_budget_for` is the per-tier mapping and
> `RuntimeSim::set_debris_budget`'s first real caller. It lives in the **render**
> crate as plain numbers so physics never learns the word `RenderTier`, and the
> **windowed player** — the one place with both a tier and a session — converts and
> applies it. High is the physics default *exactly*, pinned by a test in the crate
> that can see both, so every headless gate, replay and `--pie` comparison runs the
> unclamped numbers. The editor's Simulate deliberately does **not** apply it: it
> is the preview half of PIE == shipping, and a preview whose rubble count came
> from the author's graphics card would fail the house gate on any other machine.
>
> **LAWS this phase paid for.**
>
> * **`FIX_INTERNAL_EDGES` or a heightfield is a trampoline.** A heightfield is two
>   triangles per cell, and without the flag a body crossing a cell boundary is
>   answered with an *edge* normal — so a sphere sliding on flat ground is kicked
>   upward with no lateral force applied to it. Measured: 5 kicks at 0.105 m/s on
>   terrain, 11 at 0.188 m/s on a P21 voxel floor, both zero once flagged.
> * **Inference dressed as measurement is worse than no measurement.** The bond
>   "estimated" residue was a *tolerance* on the strength of one observation — and
>   that observation was an artefact of the test that made it, which INFERRED "this
>   bond took the fallback" by comparing the priced area against the fallback value
>   within 1%. A sweep of 24 cook configurations (2 959 bonds) found **zero**.
>   `FractureState::estimated_bonds` is now a **record**, asserted EMPTY, and the
>   phase-22 gate asserts it on the shipped block and its control.
> * **A pool matrix over a serial program measures nothing.** `phase22_gate`'s
>   `destruct_probe` arm shipped claiming the `destruct.*` calls "run from a
>   Blueprint tick, which runs inside the ECS schedule". They do not:
>   `RuntimeSim::run_all_with_args` is a serial `for` loop over a `Vec` of
>   `BTreeMap` keys, no `SimSchedule` runs on the player's fixed-step path, and
>   rapier's `parallel` feature is off by rule — so 1/2/4/8 workers execute the
>   same program and the arm cannot *discover* a race. It is kept with the claim
>   corrected (a regression tripwire for the day the tick pass parallelizes, plus
>   a pin that nothing on that path reached for the process-global pool), and the
>   same false premise was inherited from `voxel_probe.rs` (P21.4) and corrected
>   there too. This is the "inference dressed as measurement" law again, one level
>   up: four green runs proving something that was never at risk.
> * **A preview must run what it previews.** The per-tier debris budget was applied
>   by the windowed player — and **embedded PIE is windowed**, so it built a real
>   host, detected a real tier and clamped. On any Medium or Low machine the
>   editor's Simulate (which never clamps) and the PIE session it had just spawned
>   therefore stepped **different simulations**: the budget is read by
>   `step_fractures`, and a reclaim removes a solver body. Nothing failed; PIE just
>   silently stopped being the preview half of PIE == shipping, on exactly the
>   machines least likely to be the author's. The door is now
>   `debris_budget_for_session(tier, pie)` and the exemption is pinned at the
>   source.
> * **A gate must be built to falsify, not to confirm.** Every world arm in
>   `phase22_gate` carries its own control: a debris cap of 2 that really reclaims,
>   a cooked-vs-uncooked comparison with a stated horizon *and* an asserted
>   divergence past it, zero depth off the roller's lane, a `runtime_destruct`
>   twin that differs in one flag. Mutation-measured: making `attach_fractures` a
>   no-op fails **7 of 14** arms; dropping the payload's fractures inside
>   `main.rs`'s own `LoadScene` handler fails the real-`--pie`-subprocess arm and
>   nothing else, which is precisely the seam that arm exists for.
> * **A lone `\` before a newline inside a non-raw Python string is a *Python*
>   continuation.** It eats the Rust one and leaves the literal's indentation in
>   the string. Nine user-facing messages were mangled that way, including a
>   Blueprint refusal; scripted edits to Rust string literals must use raw strings
>   or a heredoc.
> * **One door for three paths.** The cook, the PIE payload builder and the
>   editor's Simulate seeder all derive fractures; two of them were separate code
>   carrying comments claiming they agreed "by construction". They did not — one
>   walked ECS **archetype** order while the others walked document order, and one
>   skipped the collidability refusal entirely. `fracture_equivalence.rs` now
>   compares the **bytes** out of a real pack against a real editor derivation.
>
> **New this batch, and worth recording.**
>
> * **`FractureAudit::collapsed` counts only out-of-damage collapses.**
>   `runtime_destruct` runs the same structural solve *inside* the damage call, so
>   a collapse the charge triggered is reported in `DamageReport::detached` and
>   never reaches the audit. Reading the audit as "did the solve run" shows zero and
>   looks like a dead subsystem. The gate measures it in **joules** instead: the car
>   absorbs one chunk's worth of bonds (25 136 J) and twelve chunks come off.
> * **"Once per break" was wrong; it is once per GENERATION.** The rubble's site set
>   is `detached && !gone`, which moves on every detach *and* every reclaim — so a
>   collapsing actor re-keys and re-uploads once per chunk that comes off, bounded
>   by `2 × chunk_count` per actor per session. The CPU pack was worse: it ran
>   every *frame*, for an answer that changes when the live set does.
>   `DebrisCache` memoizes it on `(entity, generation)`.
> * **A ban enumerates what you thought of; an allowlist enumerates what is
>   allowed.** The mirror gate banned `c.translation`/`c.rotation` from the debris
>   projection — and `age_s`, a per-step field added one batch earlier, sailed
>   through: the auditor compiled `entity: id ^ age_s.to_bits()` into both hosts
>   and all three debris gates stayed green. The whole `DebrisSite` literal is now
>   pinned field-by-field against five exact expressions.
> * **A charge with slack in it cascades.** Breaking a chunk makes its neighbours
>   cheaper — their bond to it is gone — so 40 kJ took a whole car that 30 kJ opens
>   one chunk of. Tuning a demolition charge is tuning against the *cheapest* chunk,
>   and both neighbours of the right value are instructive: 25 000 J spent nothing
>   at all (damage is not banked, and a Blueprint reports 0 J as a legal value).
> * **Size a blast against the LIGHTEST body in its radius — and count how often
>   it fires.** 60 kN·s is reasonable for 5-tonne chunks and is 64 m/s on a 295 kg
>   wheel; fired on every tick of a six-tick charge window it put a wheel past
>   **60 m and still climbing**, which is what the gate measured. (The first
>   write-up of this said "13 km up", which was neither the measurement nor
>   consistent with its own 60 m/s — corrected here, because the lesson was paid
>   for and is worth keeping accurate.) The fix was both halves: the car bomb
>   became one instant, and the constant became the right size for one.
> * **`Collider3D::density` defaults to 1.0, which is rapier's mass placeholder and
>   not a material density** — the P20.2 buoyancy finding, met again: a 0.4 m wheel
>   at the default weighs 268 grams.
> * **`.rs` files are READ BY TESTS, so they need `text eol=lf` too.** The trig-law
>   gate indexed `include_str!("debris.rs")`, searching for a newline-brace-newline
>   to find the end of an item, and never
>   normalized; with `core.autocrlf = true` and no attribute for `*.rs`, that
>   substring occurs nowhere on a Windows checkout, and the *sole* enforcement of
>   the law on that path aborted with a message about braces. The mirror gates
>   defend themselves in twenty places; `.gitattributes` now pins `*.rs text
>   eol=lf` so the class dies at the source. Its first paragraph records the
>   identical incident for `.inf_act`.
> * **The libm law is not only about trigonometry.** `f64::cbrt` was on the rubble
>   placement path in two crates — and on `wasm32` the standard library routes it
>   through the `libm` crate, so a browser client and a native one derive different
>   fragments from the same detach set. `inf_math::pcbrt` (a deliberate duplicate
>   of P21.3's `inf_voxel::cbrt_det`, held to it by a bit-equality sweep in the
>   crate that sees both) replaces it, and the grep gates now ban `.cbrt()` in
>   `inf-render` **and** in `inf-physics`, which the first one could not see.
> * **The P14 trig LAW reaches further than serialization.** The rubble is never
>   written to a file, so `sin`/`cos` in its placement could never fail a gate —
>   and would quietly make the net memo's "every client re-derives the rubble byte
>   for byte with nothing sent" false, because nothing in this repository compares
>   two *machines*. `unit_dir`/`unit_quat` are rejection samplers over `sqrt` and
>   arithmetic instead, and a grep test keeps them that way.
>
> **Honest remainders, carried out of the phase.**
>
> * **Damage is NON-LOCAL.** `destruct.apply_damage` has no impact point — there is
>   none in its signature or in scene schema v20 — so a structure sheds its
>   cheapest-to-liberate chunk, which may be at the far end of it. The consequence,
>   now measured on the flagship sample: a charge on a monolithic block standing on
>   solid ground **peels** it rather than toppling it, and the structural solve has
>   nothing to say. The gate asserts that outcome (zero out-of-damage collapses)
>   precisely so that the day a hit position arrives, the test fails and this
>   paragraph gets rewritten. The car is the sample's solve witness instead, and
>   honestly so: a car body is not supported by static geometry.
> * **No save-game container**, so destruction is **not persisted** — the phase's
>   ruling. `.inf_lvl` is the author's document, not a player's progress, and
>   `simulate_destruction_not_persisted` plus the gate's arm (h) keep a
>   save-after-damage byte-identical to a save-before. The net memo's late-joiner
>   payload is the shape a save game should take when one exists.
> * **No VFX.** This engine has no particle system, so a break makes a *sound* and
>   no dust. The audio hook is real and fires once; the visual one does not exist.
>   P22.4's instanced rubble is the nearest thing and it is dressing, not effects.
> * **No vehicle.** The sample's car is a prop: a destructible chassis on four
>   revolute wheels that settles and is then blown up. There is no vehicle
>   controller in this engine and none was built here.
> * **Streaming and destruction still do not meet.** A `Destructible` that arrives
>   with a partition cell after the level was seeded answers `NoFracture` for ever
>   (the seed is a one-shot walk, not a subscription), and a *broken* actor whose
>   cell streams out leaves its static chunk colliders behind.
> * **Wind is off by default** and foliage shadows do not bend — the P22.1 render
>   remainders, unchanged.
> * **Instanced debris: what shipped and what did not.** The GPU instance path
>   shipped, keyed by content and bounded by the budget. What did not: an impostor
>   or LOD band of its own (the rubble rides `ScatterSettings`' global bands), any
>   *physical* sub-chunk debris (the fragments are visual only), and per-fragment
>   lifetime (a fragment lives exactly as long as the chunk it came off).
> * **Coarse-LOD holes still do not propagate into the terrain pyramid** — standing
>   from P21, untouched here.
>
> **Goldens stay 50.** The playground has no golden of its own, deliberately: every
> claim it makes is structural and is asserted as state (field bytes, chunk poses,
> detach events, audit counters, ground probes) rather than as pixels, so a golden
> would add a re-bless liability and no coverage.

### Phase 23 — Embedded DCC v1: modeling core

**Goal:** create and edit meshes inside the engine. **Done when:** you can model a usable prop
(extrude / bevel / loop-cut), unwrap it, save it as a standard mesh asset, and watch a scene
that references it live-update — with clean undo and deterministic op replay.

Starting point: no editing code exists anywhere (`inf-mesh` is flat, import-only, with no
exporter), the frontend has no 3D library so in-panel 3D must be Rust-rendered, and the
viewport is explicitly single-instance. The material editor is the proven new-panel template
and the headless preview render is the proven offscreen-PNG path.

- **P23.1 Design memo** — 1. the DCC spec's open questions answered against our infrastructure
  — reuse `bevy_reflect` + the asset DB + undo; **no Blender DNA/RNA clone**; 2. the
  edit-session model; 3. the viewport decision below, with measured latency.
  **DONE 2026-08-05** — `docs/memos/p23-dcc-design.md`. Rulings: the edit session is
  **strictly asset-scoped** (a `DccDoc` keyed by asset id; the scene document is never
  touched, so no schema move and no third rung on the lock order); saves go through
  `AssetProject::rewrite_payload` **plus a synchronous `ensure_vmesh`** (a `.inf_mesh`
  rewrite regenerates nothing today, and the viewport would redraw stale `.inf_vmesh`
  geometry with full confidence); live edit works under **Simulate** because
  `SimSession::exit` reverts only the *document* and an asset edit is not in it, and is
  **impossible under embedded PIE** (the player draws placeholder cubes for asset meshes,
  and `embed_foreign` hides the editor viewport) — ledgered rather than promised;
  **`meshopt` NEVER enters the op journal** (the P18 non-portability law makes replay
  machine-dependent) — optimize at EXPORT only.
- **P23.2 Multi-viewport enabler** — 1. promote `ViewportState` to a keyed map with
  id-parameterized `viewport_*` commands and events; 2. a second `EngineHost` on its own thread
  with its own scene projector; 3. per-viewport airspace refcounts. Fallback if hostile:
  offscreen-PNG interactive preview first, native second viewport as fast-follow. Also unlocks
  the standing editor multi-viewport deferral.
  **P23.2a DONE 2026-08-05** — the pure-refactor half: keyed `ViewportState` with
  `Target::{Primary,Named,All}` named explicitly at all 31 resolution points (17 commands +
  14 cross-module pushes); **the store hoist** (the shared carve store and Simulate fracture
  map move out of `ViewportHandle` into a process-wide `commands::SharedStores`, fixing a
  latent defect where a second viewport's carves would have been saved by nobody, and
  retiring the P21.2 poisoned-outer-handle hazard); id-namespaced `viewport://` events;
  `ViewportPanel` a registered panel type; per-viewport airspace refcounts; **panel-focus
  undo routing** (Ctrl+Z inside the Material panel undid the SCENE — a shipping bug); and
  `PreviewSession`, the cached-pipeline offscreen renderer, **measured**: warm 512²
  re-render **0.34 ms** (thirty times under the ~10 ms bar, so the offscreen-interactive
  ruling holds) while the *default PNG encode* is 22.9 ms — 98% of an offscreen frame is
  deflate, and `encode_png_fast` takes it to 1.98 ms. **P23.2b (the native second
  `EngineHost`) is deliberately a fast-follow**; its projector must never enter
  `projector_mirror.rs`'s set (it projects an edit mesh, not a world, so it has no player
  twin and an exemption there is how a mirror stops mirroring).
  LAW: **the document/volumes rule is NO OVERLAP, not an acquisition order** — the three
  sites that touch both genuinely differ in which they take first, and calling it
  "document first" (as an earlier comment did) would have led a future author to "fix"
  `scene_autosave` into the very overlap the rule forbids. `overlay_sim_carves` was the one
  real exception (store held ACROSS the document, both live for a whole loop — a genuine
  two-lock deadlock shape, survivable only while the store was awkward to reach behind a
  `ViewportHandle`, which the hoist ended); it now snapshots its bindings under the document
  and releases before touching the store, so there is no exception left.
  **Shipped limitations, ledgered:** (1) a **detached panel window installs no keybinding
  listener of its own**, so Ctrl+Z pressed *inside* a torn-off Material/Blueprint/PCG editor
  does nothing at all — the `panel://focus` report fixes the MAIN window's aim, not the
  detached window's own shortcuts; (2) the **State Machine editor has no undo**, and now
  says so instead of silently undoing the scene. Closed during the audit rather than
  ledgered: the airspace refcount's default acquisition is now **window-wide** (every
  attached viewport), because `Target::All` existed in Rust while the frontend primitive
  could only name one viewport — the moment viewport #2 existed, every menu, dialog and
  drag ghost would have been painted over by it; a viewport attaching *while* an overlay is
  open now comes up hidden. LAW: **a gate must aim at the thing it names** — the
  pipeline-cache test called `program()` and never `render()`, so replacing `render`'s
  cache lookup with an unconditional rebuild left all nine preview tests green while warm
  latency degraded tenfold; and the framing test pinned the eye while target/up/fov/near/far
  were literals inside `render`, so 40°→55° passed. Both now go through the real entry
  point, and every camera parameter is a `PreviewView` field with a guard test proving each
  one moves the projection. **Carried remainders**: a lost preview device stays lost (no
  `is_lost()` check on the `Thumbnailer`'s cached context — after a TDR every material
  preview reads "No preview" until restart; the lenient handler keeps the editor alive but
  does not recover); there is **no `viewport_detach`**, so the keyed map only ever grows —
  harmless for the shell's one permanent viewport, an unbounded native-window factory the
  moment P23.4 opens and closes Model Editor tabs; and a 512² offscreen orbit pushes
  ~1.4 MB of base64 per frame (**~42 MB/s at 30 fps**) through the webview bridge, which is
  why 256² is the default and raw RGBA over a channel is the named next lever.
- **P23.3 Mesh kernel** — 1. `inf-dcc` (Ring 0): a half-edge structure importing from and
  exporting to `inf-mesh`'s `MeshAsset` (the missing writer); 2. validity invariants
  property-tested; 3. an op journal — deterministic replay is both the undo/redo story and the
  test story, mirroring `GraphJournal`.
  **DONE 2026-08-05** — `crates/inf-dcc` (Ring 0; `inf-mesh` + `inf-math` + glam + serde +
  bincode + thiserror, **no new external crate**). 107 tests after the two audit rounds below,
  no
  schema move (`MeshAsset` stays v2 — this batch adds a *writer*, not a version; the session
  journal gets its OWN v1 ladder, which is a new format, not a bump of an existing one). The
  six decisions:
  **(1) boundaries are real half-edges** — `twin` is TOTAL and `face` is the `Option`, so an
  open mesh is not a special case and no traversal branches; **(2) attributes live where seams
  live** — positions on vertices, UV + *optional authored* normal on **corners**, so importing
  a `MeshAsset` position-welds the topology back together (tolerance **exactly 0**, because an
  epsilon weld is a modelling op wearing a reader's clothes: not transitive, order-dependent,
  and irreversible) while every split attribute survives; a bowtie created by welding is
  **split back into fans**, not refused — refusing to open a file because two boxes touch at a
  corner is not a DCC; **(3) edge- AND vertex-manifold**, because a vertex's boundary loop must
  be unique for the boundary relink to be defined at all; **(4) refusals are values and they
  are INERT** — structural ops run on a clone inside `Mesh::transact` and commit only on
  success, so a refused op leaves the mesh **byte-identical** (property-tested; the honest cost
  is `O(|mesh|)` per structural op and `transact` is the single seam a slot-level undo log
  would land in); **(5) `f64` throughout** — `f32` exists only inside `build.rs`/`export.rs`,
  pinned by a source-grep gate; **(6) `meshopt` NEVER in the journal** (P18 law) — one call
  site, in `export.rs`, behind `ExportOptions::optimize`, off by default.
  **The writer**: one submesh per material slot, ear clipping (deterministic — the *first*
  valid ear, never the "best"), MikkTSpace-**class** tangents written here in pure `f64` with
  no dependency (`sqrt` is IEEE-exact and therefore bit-portable, unlike `sin`/`cbrt`), and
  derived normals from the corner's **smooth fan** summed in ascending face id so every corner
  of a fan lands on identical bits and collapses to one written vertex. `MeshSession` is the
  op journal: `base + Vec<Op> + cursor`, checkpoint every 32 ops, at most 8 retained nearest
  the cursor, `SessionSave` in bincode **and** JSON, and `restore` replays the redo tail as
  well as the applied prefix — which is what makes `undo`/`redo` infallible afterwards.
  LAWS: **a gate must be built to falsify, and a winding gate has to be a VOLUME** — every
  count, every invariant and every round trip in this crate is winding-agnostic, so the cube
  primitive shipped uniformly **inside-out** through 63 green tests until a
  divergence-theorem signed-volume assertion existed; **an exported asset must be readable by
  its own reader** — two independent ways it was not (an ear diagonal that duplicates an edge
  the mesh already has elsewhere → four faces on one edge in the flattened soup, now avoided by
  preferring an unused diagonal and *counted* when unavoidable; and two distinct kernel
  vertices at one position, which the exact weld fuses, now counted as an advisory since both
  "fixes" — nudging geometry or refusing the save — are worse); **first-use order is part of
  the format** — interning corners in face-loop order instead of *index-buffer* order cost a
  byte-identical `export∘import∘export`, because the reader welds in index order; **"keep the
  first of the run" is direction-dependent** — collapsing a merged vertex kept the survivor's
  UV on one incident face and the vanishing vertex's on the other, so a collapse that should
  undo a split left one UV at the midpoint (the rule is "drop the corner that came from
  `merge`"); **relink every vertex the patch TOUCHED, not every vertex it rebuilt** — a face
  that loses its surface in a merge is not rebuilt, but its edges were still freed and other
  boundary half-edges still pointed at them (`DeadLink`, found by the validity property);
  **validity is audited, never enforced** — `validate()` is deliberately *not* called by
  `apply`, because "every op leaves the mesh valid" would then be asserting a call the op just
  made, and vacuous checks hide real intrusions (P19); the same reasoning put a
  `the_generator_reaches_both_applied_and_refused_ops` coverage guard on the property
  generator. **Mutation-verified** (each gate fails under the defect it names, and only that
  gate): dropping the `prev` fix-up in `add_face_raw` fails 60 of 67 unit tests and
  `validity_holds_after_every_op`; making `SetCornerUv` mutate without journalling fails
  `replay_is_a_pure_function_of_the_ops` and `replay_reproduces_the_session_byte_for_byte` and
  nothing else; interning export corners in face-loop order instead of index-buffer order
  fails `export_is_a_fixed_point` and `one_round_trip_reaches_a_fixed_point` while the other
  six properties stay green. The source-grep determinism gate was falsified too (a planted
  `HashMap<u8, f32>` + `.sin()` trips three of its six arms). **Carried remainders**: no
  **AUDIT ROUND (fix-first, one commit).** Three blockers, four majors, all landed.
  **B1 — ear clipping emitted geometry OUTSIDE the polygon on ordinary rectilinear
  n-gons**: `strictly_inside` required `> 0` on all three edges, so a reflex vertex lying
  exactly *on* a candidate diagonal — the normal case wherever an L, a T or a staircase puts
  three corners on a line — did not block the ear. Measured on a 3.0 m² L-hexagon: **4.0 m² of
  triangles emitted**, one outside the L and one wound backwards, with `fan_fallbacks` reading
  zero. The textbook rule is now enforced (only **reflex** vertices block, and they block by
  lying inside **or on**), and the reason nothing caught it is itself a LAW: **the winding gate
  is blind to this by construction** — an escaped triangle and an inverted one cancel exactly,
  so signed volume reads correct. Hence `every_ngon_triangulation_tiles_its_polygon`, which
  measures **unsigned** area (`Σ|tri| == |polygon|`) over six rectilinear shapes, plus the same
  shapes through the real writer. **B2 — the session had no version ladder and `Op` had no
  discriminant freeze**: `Op::CollapseEdge{half:7}` encodes `[5, 7]`, and against an enum with
  one plausible P23.4 op inserted at index 5 that decodes as a *different edit* with no error.
  `SessionSave` now carries `schema_version` as its **first field** (positional bincode: a
  version that is not first cannot guard what follows, and adding it later would have made
  today's bytes read their leading `Mesh` arena count as a version — free only while zero
  sessions exist), `restore` refuses a mismatch, and `frozen_discriminant` is a **`match`, not a
  table**, so adding a variant stops the crate compiling until an author appends it
  consciously. **B3 — `restore` never validated `save.base`**: a mangled `next` restored `Ok`
  and failed later; a mangled `twin` **panicked** inside `split_edge`'s `expect` chain, in an op
  whose own contract says a refusal is a value. `restore` now validates in full and returns a
  typed `SessionError::{UnsupportedSchema, InvalidBase, Op}`. The **arena-indexing contract** is
  stated with it: the internal accessors assert *the kernel's own invariants*, not input, and a
  stale id is always either dead (→ typed refusal) or live (→ the documented generation hazard)
  — it never dangles, so the only door that needed closing was the one that accepts a mesh from
  outside. **M3** `validate` was blind to a half-edge naming a **dead face** — on a plane it
  returned a flat `Ok(())` while every half-edge pointed at a face that did not exist, and
  since `validate` is the independent auditor every property leans on, that was a blind spot in
  all of them at once. **M1** `ImportReport::boundary_edges` (the exact-0 weld ruling **upheld**
  — an epsilon weld smuggles non-transitivity back in — with the consequence *measured* instead
  of argued: a solid the author believes is closed arriving with boundary edges is
  self-evidently fragmented, no tolerance required). **M2** `coincident_vertices` moved into the
  **f32 domain the reader actually welds in** and restricted to what was written (an f64
  comparison read zero at exactly the moment the reader was about to fuse two vertices, and
  counted isolated vertices export never emits). **M4** the generation stamp is now
  process-monotone across `new` *and* `restore`. **M5** UV handedness joined the corner-split
  key — and had to be taken **per triangle and from the f32 UVs**, because per-face broke the
  export fixed point the moment an n-gon's UV loop summed to zero while its triangles did not,
  and f64-computed signs disagree with the f32 ones the reread recomputes. **M6** the write path
  counts non-finite and non-unit values while **`ops` refuse them outright** — the door is
  closed where closing it is free, and an author who opened a bad glTF is not locked out of
  saving their own work. **M7** the `Recompute` exception named the wrong fixture: measured, it
  **is** a fixed point on plane/cube and is **not** on cylinder/torus, and the mechanism (the
  fan sums n-gons on the first pass and triangles on the second) is now stated and pinned.
  99 tests. LAWS added: **a winding gate must be a VOLUME and a tiling gate must be UNSIGNED** —
  they catch disjoint classes and signed area cancels the one that matters; **a version that is
  not the first field cannot guard the fields after it**, and the only free moment to add one is
  before the first byte is written; **a computed value that goes into a split key must be
  computed in the domain the reader will recompute it in** (twice: coincidence in f32,
  handedness in f32).
  **AUDIT ROUND 2 (one commit).** Two new blockers, three smaller items, 99 → 107 tests.
  **NB1 — a regression the previous audit predicted in writing.** Adding `DeadFace` (M3 above)
  made the Euler test's fixture fail a *structural* check, and `check_euler` runs only on a
  structurally clean mesh — so the check stopped executing while its test kept passing on a
  violation from somewhere else. `EulerInconsistent` was constructed at one site and asserted
  nowhere. The fixture is now one only Euler can catch (a triangle whose **boundary** loop also
  claims the live face: every structural check satisfied, χ = 1, no integer genus), `check_euler`
  is called **directly** so no earlier check can short-circuit the gate, and the variant is
  asserted. LAW: **a gate whose fixture is caught by an earlier check is a gate that no longer
  runs** — when a new check lands, every existing gate downstream of it has to be re-proven, not
  assumed. **NB2 — M4 was not actually fixed**: `new`/`restore` drew from the process counter
  while every *mutation* did `+= 1`, two interleaved schemes. Two live sessions collided after a
  single edit (measured: A=1, B=2, one op on A → A=2 == B), a restored long session reissued
  stamps it had already used, and `#[derive(Clone)]` copied a live stamp verbatim. One scheme
  now: every mutation, every constructor and a hand-written `Clone` all draw from
  `fresh_generation()`. The named consumer is P23.4's `(generation, HalfId)` selection cache,
  which would otherwise accept one document's stamp for another's ids.
  **Also**: the non-finite gate moved onto the **stored** value — two finite operands can add to
  an infinity, so `TranslateVerts` and `split_edge`'s midpoint were writing infinities under a
  test that claimed the kernel "cannot be made to hold a NaN" (three doc absolutes corrected to
  match); `restore` **refuses** an out-of-range cursor instead of clamping it (a truncated write
  restored fully consistent, having silently dropped edits, and the next `save()` wrote the loss
  back as history — inside the function documented as the trust boundary) and now validates the
  **replayed** mesh as well as the base, because a valid base plus a replay is only a valid mesh
  if the *replaying build's* ops preserve the invariants; and the tiling gate's "iff" was
  **false** — unsigned area cancels an overlap against a gap exactly as signed area cancels an
  escape against an inversion (witness: `(v0,v1,v2)+(v0,v1,v3)` on a unit square has the right
  count, the right winding and *exactly* the right area while overlapping 0.25 and gapping 0.25),
  so the check is now the exact combinatorial one — each boundary edge used once, each of the
  n−3 diagonals twice in opposite directions, `O(n)` and tolerance-free. Finally, **all nine of
  `validate`'s checks are now individually falsified**: six could previously be deleted outright
  with the suite still green, and neutralizing any one of the nine now fails it (measured,
  one at a time). LAWS: **an auditor nobody audits is an `Ok(())` generator**; **two counters for
  one quantity is one counter and one bug** (the generation stamp had a global scheme and a local
  scheme, and the test that passed applied exactly one op — the single count at which the
  arithmetic accidentally agrees); **gate the value you STORE, not the value you were handed**.
  Carried, named rather than closed: `SessionError::InvalidResult` is constructed and asserted
  nowhere, deliberately — reaching it requires an op that breaks the invariants, i.e. a bug in
  this crate, and it exists for the cross-build case (an old journal replayed by a newer op set)
  that cannot be constructed today.
  **Carried remainders**: no
  panel, no commands, no modelling ops beyond the core
  set (P23.4); collapsing a tetrahedron edge is *permitted* and leaves a legal two-face
  degenerate surface (topology kernel, geometric rules excluded on purpose); import **refuses**
  a skinned submesh rather than dropping weights (P24 gives the kernel somewhere to put them)
  and refuses a genuinely non-manifold *edge* rather than inventing geometry.
- **P23.4 Modeling ops & tools** — 1. extrude / inset / bevel / loop-cut / knife / merge /
  subdivide / mirror; 2. a vertex/edge/face selection model with soft-select; 3. gizmo reuse
  extended to component selections; 4. a Model Editor panel on the material-editor template.
  **DONE 2026-08-05** — the modelling set, the selection model and the DCC's first visible
  surface. No new external dependency, `MeshAsset` stays v2, scene v20 untouched, goldens
  stay 50 (the panel preview is a PNG in a DOM panel; nothing here reaches the real renderer,
  so a golden would add a re-bless liability and no coverage).
  **The ops** (`inf_dcc::model`, nine **appended** `Op` variants at 13..21): extrude
  (faces along the region normal; edges by an **explicit delta**, because an edge has no
  canonical direction and both obvious candidates are wrong half the time), inset
  (region + individual, miter-offset corners), bevel, loop cut, knife, merge, subdivide,
  mirror. **Region-border detection** is computed once and shared by extrude and inset: an
  edge with a selected face on both sides is interior and gets no wall, so two faces extrude
  as one block rather than two boxes sharing a membrane. Every op runs inside
  `Mesh::transact`, so a refusal is a typed value and the mesh is byte-identical.
  **The v1 scopes, refused rather than approximated**: bevel is **one segment**, and its
  construction *keeps* both endpoints and caps each end of the strip with a triangle — which
  is what makes it work at **any valence** (proven on a cylinder pole) where a
  vertex-dissolving bevel needs a case per valence; knife is a path of vertices and points on
  edges applied **atomically** (a path that cannot finish cuts nothing), and a segment that is
  already an edge is *skipped*, not refused; subdivide is simple midpoint and rebuilds the
  faces around the region (a shared edge split on one side is not a mesh); mirror is
  **axis-aligned, and that is a correctness requirement rather than laziness** — the seam
  weld's tolerance is exactly zero, so a point on the plane has to reflect back
  BIT-IDENTICAL, `2d - d == d` exactly, and an arbitrary plane lands one ULP away and hands
  the author two shells with a crack between them.
  **The selection** (`inf_dcc::select`) is keyed by the journal **generation**, which is the
  whole correctness story: ids are arena slots with a LIFO free list, so a structural op hands
  the SAME ids back naming different polygons — measured, since after a `SplitEdge` on a cube
  the face-id set is *identical* and every loop changed, so a liveness check could never catch
  it. Two doors decided by `op_preserves_ids`, an exhaustive match with no wildcard:
  restructuring ops REPLACE the selection from their `OpOutcome` (so an extrude leaves the new
  cap selected), attribute ops may carry it. Plus grow/shrink, edge loop + edge **ring** (the
  ring is what loop cut walks — one traversal, two consumers), linked, invert, the six
  conversions, and **soft-select over GEODESIC distance in metres** through
  `inf_terrain::Falloff` (a hop count is unitless and density-dependent; a Euclidean ball
  grabs the far side of a thin wall; a `BTreeSet<(distance bits, VertId)>` frontier makes
  Dijkstra order-independent).
  **The panel** (`commands/dcc.rs` + `dccStore` + `panels/model/ModelEditor.tsx`, instance
  `"model:<assetId>"`, wired into the P23.2a undo-scope registry so Ctrl+Z undoes the MESH):
  the preview is `PreviewSession` with a **swapped geometry buffer**, keyed by the generation
  stamp so an orbit re-renders without re-uploading; tessellation goes through the real
  **writer**, so the picture is the geometry that gets saved. The overlay is
  **CPU-composited**, and that is the decision: picking has to be CPU (there is no sub-object
  id buffer and the memo rules the viewport's ID pass out of this path), so a GPU line pass
  would give two answers to one question — composited here, what lights up is what `pick`
  would return because it is the same `Projector`, and occlusion comes free from the topology
  (an edge whose two faces both point away is culled with a dot product, which a line pipeline
  cannot do at any price). Honest limit: that is back-face culling, not depth testing, so a
  near edge sitting behind another *part* of the same model still draws.
  Save is `rewrite_payload` **plus a synchronous `ensure_vmesh`, under one project lock** —
  the P23.1 rule — with the `ExportReport`'s two unroundtrippable counters surfaced as
  sentences and the `ImportReport`'s `boundary_edges` as the open/closed verdict on how the
  asset arrived. Content Drawer: "Edit Mesh" plus double-click on `.inf_mesh`.
  **Live in scene, proven**: `tests/dcc_live_in_scene.rs` opens a written mesh, extrudes its
  lid *as a region*, saves, and asserts the `EditorRenderAssets` content-hash key **changed**
  AND that the derived `.inf_vmesh`, decoded off disk, tops out at 2.5 m rather than the 1 m
  of the cube it was built from — because the id moving only proves a rewrite happened, and
  only the second proves it went through the derivation. Its twin asserts that an
  open-then-save is a **no-op**, so the key cannot be trusted to move for the wrong reason.
  FOUND: **the third face of the coincidence hazard.** P23.3 documented two ways a legal
  kernel mesh fails to round-trip; the property battery found a third, and these ops make it
  ordinary. Two `f64` vertices a hair apart round to the SAME `f32`, so the writer emits them
  at one place, the reader's exact weld fuses them, and the triangles that used both are
  dropped as degenerate — the asset comes back legal, smaller, and **not the mesh that was
  saved** (81 indices out, 75 back). Not repaired (nudging geometry falsifies the model;
  refusing the export makes extrude-then-drag unsaveable) and now *entitled* by the writer's
  own `coincident_vertices` counter, over the whole random battery and in a named
  deterministic test.
  LAWS: **an append-only wire enum is append-only in the FROZEN TEST too** — the discriminant
  pin is a `match`, not a table, precisely so nine new ops had to be numbered by hand;
  `CollapseEdge{7}` still encodes `[5, 7]`, so `SessionSave::CURRENT_VERSION` stays 1, and the
  three enums *nested* inside an `Op` are now frozen as well (a swapped `MergeTarget` replays
  a saved session as a different edit with no decode error anywhere). **A gate that names a
  region must select a region** — the selection property passed under a table that lied about
  `SplitEdge`, because selecting only the *first* face missed the two or three a structural op
  rebuilds. **A mark has a radius, so "is this pixel hot" is not "is it drawn here"** — the
  overlay-agrees-with-pick test passed with the overlay drawing three pixels off, and now
  measures the drawn mark's centroid. **A shrink test must not put its block on a mesh
  boundary** — at a boundary there is nothing outside the selection, so those faces correctly
  stay, and a test expecting otherwise asserts a bug into existence.
  **Mutation-verified** — but see the audit block below: **one of these seven counts was
  false and two were measured without `--no-fail-fast`**, which is not a count but "the
  first target that failed". The re-measured table is there.
  Tests: inf-dcc **157** (138 lib + 10 property + 7 determinism-law + 2 fracture; the property
  generator now reaches all 22 op kinds, with a coverage guard that fails if any of the nine
  never applies, and `determinism_law` gained a gate on *itself* that reads `src/` and fails
  when its ban list falls behind), inf-editor-core +15 (13 `dcc` unit + 2 live-in-scene),
  inf-studio +8, frontend 347 (+14).
  **AUDIT ROUND (fix-first, one commit).** One blocker, six majors, three blind gates — all
  landed. What held, verified independently: region borders genuinely handle holes (annulus and
  three-border fixtures, inner rims sealed); bevel stays manifold at adjacent and three-way
  corners; loop cut is exact-once on torus rings; the knife is genuinely atomic (a self-crossing
  path refuses byte-identical, including the nested rollback); all 22 refusals are inert; the
  one-`Projector` claim is TRUE; the freeze pin is append-only by diff; the phantom sweep is
  clean (320 citations, zero); and the coincidence-hazard ruling is upheld — the advisory chain
  reaches the panel with count, consequence and remedy, and refusing the save would be worse.
  **B1 — ONE GLOBAL STORE UNDER A MULTI-INSTANCE PANEL.** `dccStore` held a single `doc` while
  the panel is registered `singleton: false` and the dock keeps inactive tabs **mounted**. Open
  A then B and both tabs render B; every tool press in A edits B; A's backend session leaks for
  the process's life; and closing A reads the shared doc and closes **B**, blanking a document
  with unsaved work in it. The undo scope was one global too. The backend was multi-document
  from the first commit, so this was purely the frontend collapsing it: the store is now keyed
  by asset id with a per-panel selector, the preview queue is per document (a global gate would
  have made two panels take turns), and the undo registry hands the scope the focused panel's
  `params` so Ctrl+Z reaches the document under the cursor.
  **M1 — THE SAVE WAS NOT ATOMIC AND ITS GATE DID NOT EXIST AS CLAIMED.** `AssetProject` has a
  lock, not a transaction: `rewrite_payload` completes and `ensure_vmesh` can then fail, leaving
  new payload + stale DAG **permanently** — the watcher re-keys on the new hash and the viewport
  draws the old surface with complete confidence, which is the exact failure the design memo
  opens by naming, while the module docs claimed "no window in which the two disagree". And the
  claim that dropping `ensure_vmesh` failed a gate was **false**: independently measured **ZERO**,
  because the gate was Ring-1 and *inlined* the pattern instead of calling the product. The save
  is now `inf_editor_core::dcc::save_mesh_session` — Ring 1, so a test can reach it — the command
  is four lines that call it, the gate calls it, and the mutation now fails **2**. Its failure
  contract is decided rather than assumed: on a failed derivation the **stale DAG is removed**, so
  the pair is always (new payload, new DAG) or (new payload, no DAG), and "no DAG" is a state the
  renderer already handles (a placeholder — visibly wrong beats confidently wrong) that the next
  save or project-open repairs. If the removal *also* fails, `SaveError::Torn` says exactly what
  disk holds. Writing that gate immediately found a third state nobody had named: **`Skipped` is
  not an error**, so a mesh edited below the virtualization threshold derived nothing and kept the
  DAG describing the mesh it used to be. Closed.
  **M2 — SUCCESSOR SELECTIONS WERE THE WHOLE PATCH.** `OpOutcome` reported everything an op
  created, so an extrude handed back cap **and walls** and the successor selection dragged the
  base vertices with it: extrude→drag *flattened* the extrude, extrude→extrude moved the whole
  box, and a loop cut returned 20 of 20 strip edges where the new loop is 4. The test encoded the
  defect ("the cap plus four walls"). `OpOutcome` is now documented as the op's **successor**, with
  a table: extrude → the cap, inset → the inner faces, bevel → the strip, loop cut → the new
  loop's edges, knife → the cut edges, subdivide → its own faces and never the fringe. `adopt`
  takes it literally and widens **downward only**. The gate is the workflow:
  `extrude_then_translate_moves_only_what_the_extrude_made`.
  **M3 — derived corner UVs escaped the finiteness contract.** `lerp_corner`/`average_corner` are
  arithmetic on values the reader preserves verbatim, so `SubdivideFaces`/`LoopCut`/`SplitEdge`
  returned `Ok` leaving a mesh `validate` rejects — and the session then **failed its own
  `restore`**, so the editor could save a document it could never reopen. Same law as the
  positions (gate the value you STORE); the corners had simply never been brought under it.
  **M4 — `MergeVerts` was not a function of its vertex set** (permuting three ids flipped `Ok` and
  a refusal). Canonicalized; `Last` therefore means the highest id in the set, which is what the
  only caller already passes. **M5 — bevel and inset accepted overshoots that INVERT the solid**
  with `validate` blind to winding (a 2 m cube negative at ~2.83, an inset of −0.5 folding four
  faces). Refused, not clamped, with the amount named — and it took **two** criteria, which is
  the interesting part: a square face inset past its own centre maps its corners through a 180°
  rotation, which is *orientation preserving*, so every normal agrees and what is actually broken
  is that each ring quad became a bowtie. A negative inset reverses the ring's **normals**; a
  positive overshoot reverses its **edges**. **M6 — the preview re-tessellated and cloned the whole
  mesh EVERY ORBIT FRAME**, contradicting the module's own contract ("never on a camera orbit").
  `PreviewCache` keys on the generation stamp and counts its own runs, so the contract is now
  measurable: ten orbit frames, one tessellation.
  **Blind gates, each now biting**: the classifier's arm list could be replaced by `_ => false`
  suite-green (a source gate now proves it has no wildcard and names every variant, read out of
  the enum itself); the geodesic step could be replaced by a constant 1.0 (every fixture was
  unit-spaced and the thin-wall case used *disconnected* sheets, which a hop count also passes —
  there is now a 0.1/0.1/0.1/2.7 strip where the two answers cannot be confused); the facing gate
  asserted `visible == 9`, which is inversion-symmetric (it now asserts **which** faces, one per
  axis, all on the camera's side); `frozen_nested`'s outer wildcard gave silent zero coverage to
  any future nested enum (exhaustive now); the save advisory was cleared by the next action
  rather than the next **save**, so an author who pressed Save and moved the mouse never read the
  one sentence saying disk ≠ session; the facing rule was written three times in a file whose
  docs claim "no second one to drift" (one `faces_eye` now); the overlay rasterizer overflowed
  `i32` on extreme projections (saturating, with a clamp window — **and that one was listed
  here without a gate**: reverting it failed nothing, which the re-audit caught and a test over
  ±3e30, infinities and NaN now closes); nine `cargo doc` warnings; and
  the determinism self-gate did not recurse into subdirectories — the day `src/uv/` arrives every
  law above would have stopped applying to it silently.
  **Also found while writing the gates**: the property battery turned up a **third symptom** of
  the coincidence hazard — a face whose triangles *all* collapse in `f32` makes the reader report
  `NoGeometry` on a mesh that has faces. Same entitlement, now asserted.
  **RE-AUDIT (one commit).** B1/M1/M2/M5 and the preview queue confirmed closed by independent
  mutation. Four majors remained. **M-1 — `SaveError::Torn` was unreachable by any filesystem
  failure, and `Derived` asserted a disk state nobody had checked.**
  `AssetProject::delete` discards both `remove_file` results with `let _ =` and returns
  `Ok(Vec::new())` unconditionally, so `remove_derived_vmesh`'s `.is_ok()` was **always true**
  past its early returns — a database condition wearing a filesystem's clothes. When the delete
  genuinely fails (a renderer holding the `.inf_vmesh` mapped — a real Windows case since P16)
  the author was told "the stale DAG has been removed, the mesh will draw as a placeholder";
  it had not been, `resolve_vgeom` found it, and the viewport drew the PREVIOUS geometry
  confidently — the exact pair the save's headline invariant forbids, delivered by the error
  path that claims to handle it. The verdict is now on-disk truth
  (`delete(..).is_ok() && !path.exists()`), `Skipped`'s removal is checked like any other
  (its answer had been dropped, so no path could observe a failed removal), and `Torn` is
  reachable and **tested**: a handle opened without `FILE_SHARE_DELETE` blocks the unlink and
  the save reports it, with a companion test proving the check is not simply inverted. LAW:
  **a bool that reports on the filesystem must ask the filesystem.**
  **M-2/M-3 — the B1 leak survived in a narrower window.** `close` read the entry's `doc`,
  found it null because `open` had not resolved, and sent nothing — leaking the backend
  document, its `MeshSession` and its journal for the process — and then `patch` created the
  entry unconditionally, so every late-resolving `open`/`apply`/`save`/`mergeAsset`
  **resurrected** a deleted document and `docs` never shrank. React StrictMode walks this on
  every dev mount. Fixed with per-asset tombstones: `close` marks, deletes and sends
  `dcc_close` **by asset id** (the command is now symmetric with `dcc_open` and idempotent, so
  it can always be sent); `patch` never creates, so a late reply for a closed document is
  dropped; and an `open` that resolves into a tombstone sends the close again, because the
  session it just created is the one the first close could not name.
  **M-4 — the i32 overflow fix was ungated** (reverting it failed nothing while the ledger
  listed it under gates that bite). Promoted to a test over ±3e30, infinities and NaN.
  **Re-measured mutation table** (every count with `--no-fail-fast`, run **serialized** — both are
  house law now, since one earlier count was taken without the flag and a parallel run shares a
  target directory): wall every region edge → **3**; claim `SplitEdge` preserves ids → **3**;
  drop one of the bevel's end caps → **4**; skip the derivation *inside `save_mesh_session`* →
  **2** (was **0** through the inlined gate); draw the overlay 3 px off → **1**; invert the facing
  rule → **3**; remove the preview's in-flight gate → **1**; report the extrude's walls again →
  **4**; collapse the store to one document → **4**; drop the inversion gate → **1**; drop the
  derived-corner finiteness gate → **2**; drop the canonical merge order → **1**; re-tessellate
  every frame → **1**. Re-audit round: trust the database for the removal verdict → **1**; drop
  `Skipped`'s removal answer → **1**; unclamp the rasterizer's pixel cast → **1**; require a
  document before sending `dcc_close` → **1**; let `patch` create entries again → **2**.
  LAWS added: **a Ring-1 gate that INLINES a Ring-2 pattern proves the pattern, never the
  product** — the save's gate and the save's code were two copies of the same two calls, so the
  product could lose one and nothing failed; the fix is that the logic lives where the test can
  reach it and both go through the door. **A lock is not a transaction**, and a failure contract
  that is not written down is the claim that failures do not happen. **An outcome is a successor,
  not an inventory.** **A rotation is orientation-preserving, so a winding check cannot see a
  fold-through** — the criterion for an offset is the direction of the edge it moved.
  Tests after both rounds: inf-dcc **166** (146 lib + 10 property + 8 determinism-law + 2
  fracture), inf-editor-core **+20** (15 `dcc` unit + 5 live-in-scene, one of them
  Windows-only), inf-studio +8, frontend **355**.
  **Carried remainders**: bevel has no segment count and no true multi-edge *vertex* bevel
  (several edges meeting at one vertex get a strip each with a wedge between them); the knife
  is a vertex/edge-point path, not free-form (that needs a ray-vs-face solve belonging with the
  panel); subdivide does not smooth; mirror is axis-aligned only; the wireframe is back-face
  culled, not depth tested; **the gizmo is not extended to component selections** — deliverable
  3 of this batch, deferred to P23.5, which wants the same drag plumbing for its brush, so the
  panel translates through a numeric tool rather than a dragged handle; there is still **no
  `viewport_detach`** (harmless here only because the panel is DOM, not a native viewport);
  and the drop-merge copies geometry rather than referencing it, loses material slots, and
  does not weld. Added by the audit: the wireframe's occlusion is **back-face culling, not
  depth testing** (fixing it means reading the depth buffer back beside the colour);
  and a **detached** Model Editor
  window still installs no keybinding listener, so Ctrl+Z inside a torn-off one does nothing
  (the standing P23.2a limitation, unchanged and now with one more panel behind it).
  **`MergeVerts::Last` changed meaning** in the audit round — from "the last vertex in the
  caller's list" to "the highest id in the set" — with **no version bump**, and that is a
  deliberate call worth writing down rather than a detail. It is harmless today because no
  session has ever been written to disk (`MeshSession` lives in `DccState` for the process),
  and the only caller already passed a `BTreeSet`-ordered list, so nothing observable moved.
  What it exposes is a real limit of the freeze pin: `Op`'s discriminants are pinned as
  **bytes**, and a byte pin cannot see a *semantic* change — `[19, 2, 1]` encoded the same
  edit before and after and means something different. The day sessions persist
  (`SessionSave` already exists and already has a v1 ladder), this line is the warning that
  the ladder has to move for a reason the byte test will never notice.
- **P23.5 Sculpt & UV** — 1. brush sculpt on the edit mesh, reusing the terrain brush/falloff
  doctrine; 2. UV seams + an LSCM-class unwrap + a 2D UV panel; 3. normals and tangents
  recompute.
  **DONE 2026-08-05** — the brush, the component gizmo (P23.4's deferral) and the UV half.
  No new external dependency (the CG solver is written here), scene v20 and `MeshAsset` v2
  untouched, goldens stay 50. **`SessionSave` v1 → v2**, and for a reason no `Op` pin could
  see: `Mesh`'s `HalfEdge` grew a `seam: bool`, so the `base` a save carries is a different
  *shape* on the wire. No migration is written and that is the honest answer — `MeshSession`
  has only ever lived in `DccState` for the life of a process, so **zero v1 sessions exist**,
  and a decoder for a file that has never been produced is a claim, not a ladder.
  **Sculpt** (`inf_dcc::sculpt`, `Op::Sculpt` at 22): the terrain `Stroke::begin`/`add_dab`/
  `finish` doctrine adapted to a journal that has no transactions — **the op IS the
  transaction**, carrying every dab centre of one mouse-down→up gesture, so a stroke is one
  journal entry and one undo step. Dabs are arc-length resampled by `stroke_dabs` (the 3D
  mirror of `inf_terrain::dab_positions`) **before** they reach the op, so replay does not
  depend on the resampler. Four modes: draw along the influenced region's averaged normal,
  smooth as a **Jacobi** Laplacian sweep (Gauss-Seidel would relax along ascending vertex id,
  an arena artefact), flatten toward the plane the stroke *started* on (a per-dab plane
  chases the geometry and converges on nothing), and grab, whose influence set is fixed at the
  first dab so the grabbed region cannot walk across the model. Influence is the **existing**
  geodesic machinery seeded at the vertex nearest each dab, through `inf_terrain::Falloff`.
  Honest limit: geodesic distance lives on the *edge graph*, so the felt centre of the brush
  snaps to the nearest vertex.
  **The component gizmo** reuses `inf_render::gizmo` wholesale — `pick_axis`'s analytic
  11-pixel hit test, `GizmoDrag::update`'s delta math, `gizmo_world_size`'s screen-constant
  rule — and the DCC projector stays out of the `projector_mirror` set (the P23.1 §6 hard
  constraint, unchanged). New `Op::RotateVerts` / `Op::ScaleVerts` (23, 24) carry a **pivot**
  and no weight table: the caller emits one op per distinct soft-select weight, the
  `SoftTranslate` shape. Both the number box and the dragged handle go through **one**
  function, `inf_editor_core::dcc::transform_ops`, and the equivalence is tested at the
  product boundary rather than asserted. Rodrigues is built from `inf_math::psin64`/`pcos64`
  and **the sine/cosine pair is renormalized**: the raw degree-11 polynomial is a rotation
  composed with a slight scale, so a quarter-turn of a 1 m vertex came back 56 nm short and
  repeated drags would have shrunk a selection with nothing to tell the author why. What
  remains is an *angle* error under 6e-8 rad, which is the honest price of the portability
  law.
  **The orphan-settler doctrine** (P21.3, applied to gestures): a drag lives in ONE `pending`
  slot on the document, and every journal-touching command settles it first — so a stroke
  whose pointer-up never arrives becomes a real, undoable edit. **`dcc_close` is the one
  deliberate abandon** (it frees the journal in the same call, so a settle there is the same
  loss with a wasted `transact` in front of it) and `dcc_drag_cancel` is the other (Escape
  means "no", not "commit then undo"). Both directions have a test.
  **Live feedback, measured** rather than assumed: `PreviewCache`'s key is the journal
  generation and an uncommitted drag deliberately does not move it, so the drag renders from a
  scratch clone keyed on the drag's own shape. Debug build: 26 v → 0.17 ms committed / 0.11 ms
  scratch; 1 538 v → 8.6 / 9.1. **The clone and the stroke are free; the tessellation is the
  whole cost**, which is the finding — and the stated limit is that this path will not hold an
  interactive rate at a hundred thousand vertices.
  **UV** (`inf_dcc::uv`, `Op::SetEdgeSeam` at 25 and `Op::Unwrap` at 26): seams live on the
  half-edge twin pair, the `sharp` storage discipline exactly (`capture_edge_flags`/
  `apply_edge_flags` widened to carry both, because a second capture/apply pair would have
  been a dozen more places to remember one flag and forget the other). LSCM per chart, pure
  Rust, **no external dep**: the sparse system is assembled in `BTreeMap` order and solved by
  a **fixed-iteration** conjugate gradient on the normal equations — fixed because a
  `while residual > tol` loop makes the answer depend on rounding, and the price is paid by
  **reporting** the residual (`‖Ax − b‖/‖b‖` of the ORIGINAL system, not of the normal
  equations, because the number an author reads has to be about the thing they can see).
  **The pin rule**: `p0` is the chart's lowest `VertId`, `p1` the vertex farthest from it,
  ties by lowest id. **The packing rule**: shelf by decreasing height, ties by decreasing
  width, ties by chart index — a total order, so the layout cannot depend on the sort's
  stability — into a bin of width `√(Σ area)`, then normalized into `[0,1]²`.
  **Replay does not re-solve**: `Op::Unwrap` carries the computed per-corner UVs. The solver
  *is* deterministic today; that is still not a reason to make an op mean "whatever this
  build's solver says" — the meshopt lesson applied before it can bite, since a journal is
  replayed by a different build than wrote it.
  **The 2D UV view** is a second pane in the Model Editor, CPU-composited and PNG-encoded down
  the same path as the 3D overlay, for the same reason: seams, charts and the selection are
  backend facts and a `<canvas>` renderer would be a second answer to them. The selection is
  literally shared — one `SelectionSet`, one document. Edges are drawn per **corner pair**, so
  a vertex on a seam draws twice and the charts come apart on screen the way they do in the
  file. Draw order is the priority order (wires → frame → seams → selection), which the first
  version got wrong twice: a chart packed against `u = 0` swallowed the border, and a selected
  seam read as a seam.
  FOUND, and worth keeping: the reachability battery could **run away to 6.6 million
  vertices** on the plane — `Mirror` doubles a mesh the new transform ops have pushed off its
  own mirror plane — taking a 0.7 s property suite to 93 s. Bounded by restarting from the
  base past 4 000 vertices.
  Tests: inf-dcc **204** (184 lib + 10 property + 8 determinism-law + 2 fracture),
  inf-editor-core `dcc` **31**, inf-studio **14**, frontend **363**.
  **AUDIT FIXES (2026-08-05)** — one blocker and eight majors, all measured:
  **B1, the solver under-converged at panel-reachable sizes and silently folded geometry.**
  A plane plus five Subdivide clicks is 1 089 vertices; at the flat 256 iterations its residual
  was 5.7e-2, its edges were wrong by 361%, and **two triangles were folded on a flat square**
  whose exact conformal map is a similarity — six clicks folded 338. The count is now
  `cg_iterations(free) = (4·free).clamp(256, 8192)`, integer arithmetic on a count so the
  determinism argument is untouched; measured after: ×5 residual **3.3e-14** and ×6 **1.2e-13**,
  both with **zero** folds.
  **The rotation collapse.** Past ~2e16 `psin64` and `pcos64` both return exactly zero and
  Rodrigues becomes an **axis projection** — finite, accepted, `validate` green (it audits
  topology, not geometry), and a quad returned collinear from a public op that rides in a
  session save. Closed by `MAX_ROTATION_RADIANS` (2^52, past which an `f64` has less than one
  radian of resolution) **and** by making the degenerate `s²+c²` pair a refusal rather than a
  fallthrough. The audit also prescribed a `mod 2π` fold, and **the measurement refused it**:
  across `[6.5, 4.5e15]` the fold moves the angle error around without removing it (1e12:
  3.4e-5 raw vs 5.0e-5 folded) because at those magnitudes the error *is* the input's own
  resolution. It improves only `|s²+c²|−1`, which the renormalization already fixes — so it is
  not in the tree, and the reasoning is in the code where the next reader will meet it.
  **The dab explosion.** `stroke_dabs` had no radius floor and sat *ahead* of the kernel's
  guards: a 1 m drag at 0.1 mm asked for 40 000 dabs (a ~3 MB journal entry inside the document
  lock) and at 1e-12 m for 1.2e13 pushes — an OOM abort taking the unsaved session. Fixed with
  `MIN_BRUSH_RADIUS_M` (1 mm, a refusal-as-value at the door) and `MAX_STROKE_DABS` (4096,
  bounding the **path** before the resampler allocates — the P21.3 rule). The two constants are
  one decision and a test pins their relationship: at the floor the cap covers 1.02 m of drag.
  **The dab_positions duplicate is gone.** `inf_dcc` and `inf_editor_core::voxel_tool` had each
  written their own lift of the terrain resampler — "deliberately the same algorithm", no gate,
  and **already drifted** on a non-finite spacing. The 3D resampler is now
  `inf_terrain::dab_positions_3d{,_capped}` and both callers are one line.
  **Gates that did not reach.** The replay-the-result source gate banned one *spelling* and
  checked its positives against the whole file — a `pub fn recompute` wrapper re-solved on
  replay with 208/208 green; it now reads the **arm** (brace-balanced) and bans the **module**.
  The settle/abandon doctrine was eleven prose statements over twenty commands and its cited
  test **did not exist**; there is now a source-read policy table where a new door fails until
  it chooses, and the cited test is real. The radius floor lived in a `#[tauri::command]`, where
  deleting it failed nothing — it moved to Ring 1 (`dcc::begin_stroke`), the same finding and
  the same fix as P23.4's save. Grab's influence-at-first-dab was ungated (the old test dragged
  *perpendicular*, so the nearest vertex never changed); a lateral-drag gate now catches the
  region walking.
  **Honest verdicts.** `worstResidual` conflated "the solver stopped early" with "this shape is
  not developable" — a failed flat plane read 5.7e-2 and a converged saddle 4.1e-2, same number,
  same advice, opposite causes. Split into `residual` (distortion) and `convergence` (the normal
  equations, zero iff CG finished), plus a per-chart **fold count**: nothing detected a
  non-disk chart, and with this crate's own seam recipe a cylinder folds 16 of 44 triangles and
  a torus 285 of 576. All three are on the wire and in the panel with advice that differs.
  **M6, corroborated and capped.** One op per distinct weight was 105 journal ops per drag on a
  289-vertex plane — about three full mesh clones at `CHECKPOINT_INTERVAL = 32`, evicting the
  entire eight-slot checkpoint history — and the gate asserted `ops.len() > 1`, naming the defect
  as a feature. Weights quantize to 1/64, capping a drag at 64 ops (measured 45 on a jittered
  289-vertex mesh, 24 on a regular grid), and the gate asserts the **cap**. The real fix — a
  weight table on a `Sculpt`-shaped transform op — is ledgered for the day sessions persist and
  the wire has to move anyway.
  **M3.** The UV pane keyed on the *journal* generation, which a selection change does not move,
  so picking a different face never refreshed it — and `selected` is a count, so A→B both read
  `1`. `DccDocDto::selectionRev` is a content **hash** (a counter can be forgotten by a new
  mutation path; a hash cannot), and the store re-renders on it.
  **Mirror ruling**: `Op::Mirror` does **not** refuse doubling a mesh that crosses its plane.
  The runaway was a generator artefact — an author presses it once and gets one undo step — a
  straddling mesh is the ordinary case to mirror, and the check would have to be geometric
  beside a weld whose whole correctness rests on `2d − d == d` being exact. The battery stays
  bounded; the op is unchanged.
  Tests after the audit: inf-dcc **219**, inf-editor-core `dcc` **36**, inf-studio **15**,
  inf-terrain **258**, frontend **367**.
  **Carried remainders**: the brush ring is drawn only *during* a stroke (a hover ring needs a
  pointer-move round trip the panel does not make); the sculpt seed is the nearest **vertex**,
  not the exact surface point; **UV editing in the 2D view is read-only** — vertex dragging in
  UV space is not built, and seam marking happens in the 3D view; the unwrap has no per-chart
  re-solve, no angle-based auto-seaming and no rotation-to-minimal-bbox before packing; **a
  soft drag is still one op per weight-bucket** (capped at 64, not collapsed to one — the
  weight-table op is the real fix); a rotate gizmo has no on-screen angle readout; and P23.4's
  remainders all still stand. **Worktree constraint**: `determinism_law` bakes
  `CARGO_MANIFEST_DIR` at compile time (as `include_str!` must), so this crate's tests have to
  be built and run in the same worktree — documented on the gate rather than worked around.
- **P23.6 Asset round-trip** — 1. edited meshes save through the asset DB (dependency events
  already live-update referencing scenes); 2. "Edit Mesh" from the Content Drawer context menu;
  3. vgeom rebuild on save via the P18.3 machinery.

### Phase 24 — DCC v2: characters

**Goal:** new characters stop being painful — template body plans become auto-rigged,
animatable characters. **Done when:** a biped and a quadruped generate from templates, the
biped rig auto-fits an imported humanoid mesh, weights solve, cloth and hair attach, and both
run under existing state machines in PIE.

Starting point: `inf-anim` already provides validated skeletons, poses and skinning palettes,
clips, blend spaces, state machines, retarget v1, sockets and root motion, with a GPU skinning
pass. Missing for rigging: IK solvers, weight painting, in-viewport bone manipulation, and a
skeleton editor.

- **P24.1 Template body plans** — 1. a parametric N-pedal skeleton generator (biped, quadruped,
  hexapod, arbitrary; proportions as params) emitting standard `.inf_skel`; 2. a template
  library UI.
- **P24.2 Auto-fit & weight solve** — 1. SDF-based template-to-mesh fitting (bounding analysis
  → joint-placement optimization); 2. heat / voxel-diffusion weight solve; 3. a manual
  weight-paint brush applying the terrain-paint doctrine to the P23 kernel; 4. IK solvers
  (Two-Bone + FABRIK) landing in `inf-anim` for both rig authoring and runtime use.
- **P24.3 Modular rigging** — 1. append limbs, tails and extras without breaking IK chains or
  weight tables; 2. sockets integration.
- **P24.4 Cloth & hair authoring** — 1. XPBD cloth (garment meshes against character SDF
  collision, deterministic fixed-step, replay-gated); 2. strand hair (guide curves +
  interpolation, clump/curl params, card generation for lower tiers); 3. authored in the Model
  Editor, run by runtime systems, quality-tiered.
- **P24.5 Character pipeline UX** — 1. a "New Character from Template" wizard: pick a plan →
  shape it → auto-rig → a default locomotion set wired to a state machine. The anti-pain
  headline.

### Phase 25 — Photogrammetry: photos → asset

**Goal:** photos in, game-ready asset out, entirely in-engine. **Done when:** a
synthetic-render dataset with known poses reconstructs within error bounds deterministically, a
real photo set produces a textured, retopologized, LOD-ready asset through the wizard, and the
result imports as a standard asset.

Decided 2026-07-31: **native classical SfM + GPU MVS** in Rust/WGSL — deterministic and
in-engine, with no external reconstruction dependency.

- **P25.1 SfM core** — 1. feature extraction and matching with deterministic ordering;
  2. incremental SfM + bundle adjustment on the job pool, pool-size-invariant per house rules.
- **P25.2 GPU MVS** — 1. WGSL plane-sweep depth maps; 2. TSDF fusion → dense mesh extraction.
- **P25.3 Finish pipeline** — 1. decimation/retopo (meshopt-based v1); 2. UV unwrap via P23.5;
  3. normal / AO / albedo bake from dense → retopo on the existing headless bake machinery;
  4. optional de-lighting v1.
- **P25.4 Capture wizard** — 1. drop photos → reconstruct with progress → preview through the
  offscreen path → import; 2. failure diagnostics (coverage and overlap warnings).

---

*This roadmap is a living document. Each phase completion updates it; decision memos land in
`docs/memos/`; deviations require a memo, not silence.*
