# Infinity Engine

A next-generation, commercial-grade game engine built in **Rust**, with **Infinity Studio** —
a Tauri v2 + React editor designed as a modern evolution of the industry-standard editor UX.

## What makes it different

- **Two ways to code, one truth.** Write gameplay in real Rust (built-in IDE with
  rust-analyzer) or in **Infinity Blueprints** (node graphs) — interchangeably. Graphs
  transpile to real Rust source and stay bidirectionally in sync; shipped games run compiled
  native code only.
- **Native-core editor.** The engine core is pure Rust (wgpu/WGSL, data-oriented ECS); the
  viewport is a native swapchain embedded in the editor, not a browser canvas.
- **Planetary scale by construction.** 64-bit world coordinates with floating-origin rebasing,
  designed in from the first line of renderer code.
- **2D, 2.5D, and 3D as first-class citizens**, for every genre.
- **Professional pipeline.** Typed `.inf_*` assets (fast binary + git-friendly TOML sidecars),
  content-addressed cooking, per-platform packaging, play-in-editor with crash isolation.

## Repository layout

```
crates/            Engine core (Ring 0) — Tauri-free, console-portable
editor/crates/     Editor core + native viewport host (Ring 1) — Tauri-free
editor/studio/     Infinity Studio — Tauri v2 app (React + TypeScript frontend)
runtime/           Standalone player + cook/packaging pipeline
tools/             `inf` CLI
templates/         Project templates
samples/           Sample projects (double as integration fixtures)
docs/              Engineering roadmap, decision memos, design docs
```

Start with [`docs/ROADMAP.md`](docs/ROADMAP.md) — the full engineering plan: architecture,
16-phase roadmap, technology matrix, and verification strategy.

## Building

Prerequisites: Rust (stable, 1.97+), Node 22+, and the
[Tauri v2 platform prerequisites](https://v2.tauri.app/start/prerequisites/) for your OS.

```sh
# Engine + all crates
cargo check --workspace

# Infinity Studio (editor)
cd editor/studio
npm install
npm run tauri dev
```

## Status

Early development — Phase 0 (foundation & risk spikes) of the
[roadmap](docs/ROADMAP.md). The workspace scaffold, CI, and the four
de-risking spikes (native viewport embedding, graph↔Rust transpiler,
hot reload, play-in-editor) are in progress.

## License

MIT OR Apache-2.0
