# Introduction

**Infini Engine** is a commercial-grade game engine built in Rust, with a built-in editor that
runs as a Tauri v2 + React desktop app. It targets 2D, 2.5D, and 3D games of every genre, and
carries a project from the first imported asset all the way to a shipped, packaged build.

Two things make it distinct. First, **two ways to code, one truth**: you can write gameplay in
real Rust — the editor embeds a full IDE with rust-analyzer completions, a terminal, and source
control — or you can author it visually with **Infinity Blueprints**, node graphs that *transpile
to real Rust source* and stay bidirectionally in sync. The in-editor interpreter that previews
your graphs is parity-tested in CI against the compiled Rust, so what you see while iterating is
what ships. Second, the **viewport is a native window**: the engine renders your scene into a
real wgpu swapchain embedded in the editor, not into a browser canvas, so the editor feels and
performs like a native tool.

Under the hood, Infini Engine is data-oriented. The world is a `bevy_ecs` archetype ECS driven
by a real parallel schedule; rendering is GPU-driven wgpu/WGSL; and world coordinates are 64-bit
with floating-origin rebasing designed in from the first line of renderer code, so planetary-scale
worlds keep their precision. Assets are a dual format — a fast binary payload plus a git-diffable
TOML sidecar — with content-hashing, a live dependency graph, and content-addressed cooking.

The editor's mental model will be immediately familiar if you have used Unreal Engine 5: a menu
bar (File · Edit · Window · Tools · Build · Platforms · Select · Actor · Help), a main toolbar, an
Outliner over a Details panel, a slide-up Content Drawer, and dockable, detachable panels
everywhere. Where it differs, it aims to be *more* disciplined — an 8-point spacing system, a real
JSON theme system, a global command palette (Ctrl+Shift+P), and first-run layouts per discipline.

This guide walks you from installing the engine through building your first scene, scripting it
with Blueprints, sculpting terrain, authoring materials, animating a character, and finally
packaging and shipping your game. For the full engineering plan — architecture, the 16-phase
roadmap, and the technology matrix — see [`docs/ROADMAP.md`](https://github.com/McLovin143628/Infini-Engine/blob/main/docs/ROADMAP.md)
in the repository.
