# Memo: viewport overlay — egui vs. custom (P2.4.4)

**Status:** decided — engine-rendered custom overlays, no egui.
**Date:** 2026-07-20
**Scope:** how in-viewport 2D/3D affordances (gizmos, selection outlines,
measurement text, viewport HUD, debug draws) are drawn.

## The question

The viewport needs a growing set of overlays: transform gizmos, selection and
hover outlines, the world grid, debug primitives, and later a small HUD (camera
speed, gizmo mode, stats) plus billboarded labels. Two ways to draw them:

1. **egui** on the wgpu surface — mature immediate-mode UI, batteries-included
   widgets and text, one dependency.
2. **Custom** — geometry emitted into our own render passes (debug lines, the
   composite outline dilate, a future text/billboard pass).

## Why custom wins here

- **We already have the machinery.** The debug-line layer, the reverse-Z depth
  buffer, the ID-buffer picker, and the composite outline dilate are all in
  `inf-render`. Gizmos are just debug lines with screen-constant sizing;
  outlines are a mask + dilate. egui would duplicate a renderer we already run.
- **Depth-correct 3D affordances.** Gizmos and debug volumes must interleave
  with scene depth (an axis handle behind a cube is occluded). That is native
  in our passes and awkward in egui, which is fundamentally a 2D screen-space
  layer.
- **Airspace rule.** The native viewport child window draws *over* the webview
  (Phase 1 memo). Anything egui rendered would live on the wgpu surface anyway —
  so egui buys us no compositing advantage over drawing directly, while adding a
  second UI paradigm next to our React/Tauri shell.
- **One interaction model.** Picking already routes through the ID buffer
  (objects) and analytic screen-space tests (gizmo handles). egui would
  introduce a parallel event/hit-test path for the same clicks.
- **Dependency weight.** egui + egui-wgpu + winit glue is a large surface for
  one feature; our custom path adds a handful of WGSL lines per overlay.

## What egui would have been good for

Rich *panels* (histograms, node editors, property grids). Those live in the
React shell (Details, Outliner, Content Drawer, the future graph canvas), not on
the viewport surface — so egui's strength doesn't apply where this decision bites.

## Consequences / follow-ups

- **Text in the viewport** (labels, measurements, HUD) needs a small SDF-glyph
  billboard pass in `inf-render`. Not needed for P2; scheduled with P2.4's HUD
  polish / P3 selection labels. Until then the shell's status bar carries gizmo
  mode and camera speed.
- Gizmo hit-testing stays analytic (screen-space distance) rather than tagging
  gizmo parts into the ID buffer — thin handles pick far more reliably, and it
  keeps picking on the CPU next to the drag math. `ID_GIZMO_BASE` in
  `scene.rs` is reserved so a future ID-buffer gizmo pass remains possible.
- If a full in-viewport tool UI ever appears (unlikely given the React shell),
  revisit — but that would be a new requirement, not a reversal of this one.
