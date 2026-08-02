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

## 12. Next-Gen Wave — Phases 16–25 (planned 2026-07-31; **P16–P17 COMPLETE**)

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
  with P20 water.

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
  erosion mass gates; 3. undo via chunk deltas on the `EditCommand` pattern.
- **P21.4 Runtime carving** — 1. the same ops as Blueprint nodes, deterministic and
  replay-gated, so games can dig at runtime; 2. physics and nav updates on carve.

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
- **P22.4 Destructible environments at scale** — 1. instanced debris through the P18.5 GPU
  instance path; 2. per-tier debris budgets; 3. destruction state persisted in the save and
  replication seams, with net-relevant events documented for the P14 net layer.

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
- **P23.2 Multi-viewport enabler** — 1. promote `ViewportState` to a keyed map with
  id-parameterized `viewport_*` commands and events; 2. a second `EngineHost` on its own thread
  with its own scene projector; 3. per-viewport airspace refcounts. Fallback if hostile:
  offscreen-PNG interactive preview first, native second viewport as fast-follow. Also unlocks
  the standing editor multi-viewport deferral.
- **P23.3 Mesh kernel** — 1. `inf-dcc` (Ring 0): a half-edge structure importing from and
  exporting to `inf-mesh`'s `MeshAsset` (the missing writer); 2. validity invariants
  property-tested; 3. an op journal — deterministic replay is both the undo/redo story and the
  test story, mirroring `GraphJournal`.
- **P23.4 Modeling ops & tools** — 1. extrude / inset / bevel / loop-cut / knife / merge /
  subdivide / mirror; 2. a vertex/edge/face selection model with soft-select; 3. gizmo reuse
  extended to component selections; 4. a Model Editor panel on the material-editor template.
- **P23.5 Sculpt & UV** — 1. brush sculpt on the edit mesh, reusing the terrain brush/falloff
  doctrine; 2. UV seams + an LSCM-class unwrap + a 2D UV panel; 3. normals and tangents
  recompute.
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
