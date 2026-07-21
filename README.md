# Infinity Engine

A next-generation, commercial-grade game engine built in **Rust**, with a built-in
Tauri v2 + React editor designed as a modern evolution of the industry-standard editor UX.
It targets 2D, 2.5D, and 3D games of every genre, with a professional pipeline from first
asset import to a shipped, packaged build.

## What makes it different

- **Two ways to code, one truth.** Write gameplay in real Rust (built-in IDE with
  rust-analyzer) or in **Infinity Blueprints** (node graphs) — interchangeably. Graphs
  transpile to real Rust source and stay bidirectionally in sync; the in-editor interpreter is
  parity-tested against the compiled Rust so preview never diverges from shipped behavior.
- **Native-core editor.** The engine core is pure Rust (wgpu/WGSL, data-oriented `bevy_ecs`);
  the viewport is a native swapchain embedded in the editor, not a browser canvas.
- **Planetary scale by construction.** 64-bit world coordinates with floating-origin rebasing,
  designed in from the first line of renderer code.
- **2D, 2.5D, and 3D as first-class citizens**, for every genre.
- **Gaea-heritage terrain & PCG.** Sculptable, GPU-erodible, planet-ready heightfields with
  million-instance rule-based scattering.
- **Professional pipeline.** Typed `.inf_*` assets (fast binary + git-friendly TOML sidecars),
  content-addressed cooking, per-platform packaging, play-in-editor with crash isolation, and
  sandboxed WASM mods for safe end-user extensibility.

## Feature status

Infinity Engine is built in 16 phases (see [`docs/ROADMAP.md`](docs/ROADMAP.md)). The following
are complete and CI-green on Windows, macOS, and Linux:

| Area | Status |
|------|--------|
| Studio shell — docking, detachable windows, themes, command palette | Phase 1 ✅ |
| Native embedded viewport, UE-parity camera, gizmos, picking | Phase 2 ✅ |
| ECS & scene model — reflection Details, undo/redo, `.inf_lvl` save/load | Phase 3 ✅ |
| Asset system & Content Drawer — import, thumbnails, dependency graph | Phase 4 ✅ |
| IDE integration — CodeMirror, terminal, git, search, LSP | Phase 5 ✅ |
| Infinity Blueprints & transpiler — graph ↔ Rust, interpreter parity | Phase 6 ✅ |
| Materials & texture graphs — node graph → WGSL, PBR, bake | Phase 7 ✅ |
| 2D pipeline — sprites, tilemaps, 2D physics, 2.5D | Phase 8 ✅ |
| Play-in-editor, standalone player & desktop packaging | Phase 9 ✅ |
| Terrain & PCG — sculpt, erode, scatter | Phase 10 ✅ |
| Animation — skeletal, blend spaces, state machines | Phase 11 ✅ |
| Physics — rapier3d-f64, joints, CCD, ragdolls, audio | Phase 12 ✅ |
| Virtualized geometry (meshlet DAG, GPU culling) + LOD fallback | Phase 13 ✅ |
| Networking, WASM modding, console HAL seams | Phase 14 ✅ |
| Polish, optimization, docs & samples | Phase 15 🚧 |

The interactive documentation lives in [`docs/book/`](docs/book/) (build with `mdbook build`).

## Repository layout

```
crates/            Engine core (Ring 0) — Tauri-free, console-portable
editor/crates/     Editor core + native viewport host (Ring 1) — Tauri-free
editor/studio/     The Infinity Engine editor — Tauri v2 app (React + TypeScript frontend)
runtime/           Standalone player + cook/packaging pipeline
tools/             `inf` CLI (new / cook / pack / export)
templates/         Project templates (scaffolded by `inf new`)
samples/           Sample projects (double as integration fixtures)
docs/              Engineering roadmap, the mdBook docs site, decision memos
```

Start with [`docs/ROADMAP.md`](docs/ROADMAP.md) — the full engineering plan: architecture, the
16-phase roadmap, technology matrix, and verification strategy.

## Building

Prerequisites: Rust (stable, 1.97+), Node 22+, and the
[Tauri v2 platform prerequisites](https://v2.tauri.app/start/prerequisites/) for your OS.

```sh
# Engine + all crates
cargo check --workspace

# Run the test suite (all three OSes in CI)
cargo nextest run --workspace

# The editor application
cd editor/studio
npm install
npm run tauri dev
```

Convenience wrappers at the repo root: `run_dev.cmd`, `run_build.cmd`, `run_clean.cmd`.

### The `inf` CLI

```sh
inf new MyGame --template blank-3d    # scaffold a project (real cargo workspace + inf.toml)
inf cook --project MyGame --out Build  # cook a shippable content pack
inf export --project MyGame --out Dist # produce a runnable folder (renamed player + pack)
inf --version                          # engine version + git hash
```

## Documentation

- **[Docs site](docs/book/)** — Introduction, Getting Started, Your First Scene, Blueprints 101,
  Terrain & PCG, Materials, Animation, Packaging & Shipping, Modding. Build with `mdbook build docs/book`.
- **[`docs/ROADMAP.md`](docs/ROADMAP.md)** — the authoritative engineering plan.
- **[`docs/memos/`](docs/memos/)** — decision memos (viewport embedding, transpiler, hot reload, PIE, …).
- **[`samples/`](samples/)** — runnable dogfood projects; see [`samples/README.md`](samples/README.md).

## License

The engine is intended to ship dual-licensed under **MIT OR Apache-2.0**. The final decision
(dual-permissive vs. source-available) is pending — see [`docs/LICENSING.md`](docs/LICENSING.md).
Contributions are accepted under the same terms; see [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).
</content>
