# Getting Started

## Prerequisites

- **Rust** (stable, 1.97 or newer) via [rustup](https://rustup.rs).
- **Node 22+** and npm (the editor frontend is React + TypeScript + Vite).
- The **Tauri v2 platform prerequisites** for your OS (WebView2 on Windows; Xcode command-line
  tools on macOS; `libwebkit2gtk-4.1-dev`, `libgtk-3-dev`, and friends on Linux). See the
  [Tauri prerequisites guide](https://v2.tauri.app/start/prerequisites/).

## Build and run the editor

Clone the repository, then build the whole engine workspace and launch the editor:

```sh
# Compile every crate (Ring 0 engine, Ring 1 editor core, Ring 2 apps).
cargo check --workspace

# Launch the editor (Vite dev server on port 1440 + the Tauri shell).
cd editor/studio
npm install
npm run tauri dev
```

On Windows you can also use the convenience wrappers at the repo root: `run_dev.cmd` (launch),
`run_build.cmd` (release build), and `run_clean.cmd`.

## Run the tests

Infini Engine holds a hard quality bar — CI runs the same checks on Windows, macOS, and Linux
on every commit. To run them locally:

```sh
cargo fmt --all --check          # formatting
cargo clippy --workspace         # lints (CI treats warnings as errors)
cargo nextest run --workspace    # the Rust test suite

cd editor/studio
npm run typecheck                # tsc --noEmit
npm run lint                     # eslint
npm test                         # vitest
```

## Create a project with the CLI

The `inf` CLI scaffolds a real cargo workspace plus an `inf.toml` project manifest:

```sh
inf new MyGame --template blank-3d   # blank-3d | 2d-platformer | first-person | hybrid-2.5d
```

You can also create projects from inside the editor: the **Start Screen** (shown when no project
is open, or via File ▸ New Project…) presents a template gallery with a preview of each template
and a first-run layout choice — **3D**, **2D**, or **Scripting** — that arranges the panels for
that discipline. Pick a template, name the project, and click **Create Project…**.

## Get your bearings

When a project opens for the first time, an **interactive tour** highlights the core panels — the
viewport, Outliner, Details, Content Drawer, the Play cluster, and the command palette. You can
skip it at any time and replay it later from **Help ▸ Interactive Tour**. The single most useful
shortcut to remember is **Ctrl+Shift+P**, the command palette, which fuzzy-searches every menu
action and command in the editor.
