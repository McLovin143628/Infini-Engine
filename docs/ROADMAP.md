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
│   ├── inf-core        ids, errors, tracing, frame clock, job system (rayon+tokio facade)
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

- **P9.1 Runtime assembly** — 1. `inf-runtime` game loop (fixed-step sim + interpolated
  render); 2. rapier3d + kira baseline integration; 3. `inf-input` action mapping + gamepad.
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

- **P12.1 Physics depth** — 1. joints/motors/CCD/queries; 2. collision layers + filtering UI;
  3. physics materials; 4. ragdoll setup tool; 5. debug-draw overlay.
- **P12.2 Determinism** — 1. fixed-step replay harness; 2. Jolt-vs-rapier benchmark memo
  (decision gate for a backend swap).
- **P12.3 Audio depth** — 1. kira spatialization/attenuation/occlusion basic; 2. mixer buses +
  effects; 3. audio assets + import; 4. Blueprint audio node kit.

### Phase 13 — Virtualized geometry & advanced rendering *(flagship; deliberately late)*

**Goal:** Nanite-class geometry and Substrate-class materials. **Done when:** a 10M+ triangle
scene streams and culls at interactive rates; classic-LOD fallback documented for older GPUs.

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

### Phase 15 — Polish, optimization, docs & samples

**Goal:** commercial-grade finish. **Done when:** a newcomer installs Studio, follows the
tutorial, and ships a small game in a weekend.

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
| Concurrency | rayon (frame) + tokio (editor IO) + flume | keep tokio out of Ring 0 hot loops |
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
- **Cook/PIE:** CI cooks a sample and runs `inf-player --headless --run-frames 300 --assert-exit`.
- **Performance:** criterion benches (transform propagation, sprite batcher, scatter kernels);
  nightly frame-budget smoke on a reference scene with a hard ms budget that only ratchets down.
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

*This roadmap is a living document. Each phase completion updates it; decision memos land in
`docs/memos/`; deviations require a memo, not silence.*
