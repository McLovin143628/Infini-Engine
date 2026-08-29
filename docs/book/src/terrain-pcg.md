# Terrain & PCG

Infini Engine carries a Gaea-heritage terrain and procedural-content-generation (PCG) stack:
sculptable, GPU-erodible, planet-ready heightfields with rule-based scattering of millions of
instances. The `samples/terrain-demo` project is the reference scene for everything below.

## Heightfield terrain

Terrain is a sparse, f64-anchored tile heightfield: each tile stores tile-local f32 heights (the
same precision doctrine the renderer uses — a global f64 anchor with local f32 detail), with shared
edges so tiles stay seamless. You can import heights from PNG16 or EXR. The renderer draws terrain
through a chunked-LOD clipmap with four LOD rings, smoothstep morphing between them, and skirt
geometry to hide cracks, with per-patch culling so a large world stays at frame rate.

## Sculpting

Switch to a sculpt brush and shape the terrain directly in the viewport. The brush core offers
**raise**, **lower**, **smooth**, **flatten**, and a world-anchored **noise** mode, each with
falloff profiles. Sculpting ray-marches the heightfield under the cursor, renders an engine-drawn
brush ring, and records each stroke as one undo transaction (a dense per-tile height patch), so
edits are byte-identically undoable. One layer up, **splat painting** blends terrain material
layers with exact-255 renormalization so weights always sum correctly.

## Erosion

Terrain supports virtual-pipes **hydraulic** and **thermal** erosion. There is a deterministic,
mass-accounted CPU reference (a closed-box simulation that conserves material), and a GPU path for
interactive iteration. Open the erosion bake dialog from the sculpt toolbar (**Erode…**), choose
the number of iterations, and bake — the result is a new set of heights you can keep sculpting on.

## PCG scatter

A **PCG graph** (`.inf_pcg`) describes how to populate the world procedurally: samplers generate
candidate points, filters cull them by rule (for example, by height or slope), and a scatter stage
turns survivors into GPU instance buffers. In a scene, a **PcgVolume** references a `.inf_pcg` graph
and evaluates it on load; the resulting instances are a derived cache, never persisted into the
level, so the `.inf_lvl` stays small no matter how many instances the rules produce. The
`terrain-demo` sample scatters via a noise-plus-slope rule, and the same evaluation runs in the
editor, in play-in-editor, and in the shipped player, so what you author is what ships.
