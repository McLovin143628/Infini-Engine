# Your First Scene

A *scene* (also called a level) is an ECS world snapshot saved as a `.inf_lvl` asset — a fast
binary payload plus a git-diffable TOML sidecar. This chapter builds a tiny scene and saves it.

## Add actors with the Outliner

The **Outliner** (right-hand dock by default) lists every actor in the level as a live tree. Use
its **Add** menu — or the **+** button on the main toolbar, or **Actor ▸ Place Actor** — to add
primitives and lights: try an **Empty Actor**, a **Cube**, a **Plane**, and a **Point Light**.
Each new actor appears in the Outliner immediately; double-click to rename it, drag one onto
another to reparent, and toggle the eye icon to hide or show it. Selection is a single source of
truth: selecting in the Outliner selects in the viewport, and vice versa.

## Navigate the viewport

The viewport uses UE-parity camera controls, so there is zero learning curve if you have flown a
camera in Unreal:

- **Right-mouse + WASD / QE** — flycam; scroll to change speed.
- **Alt + LMB** — orbit; **Alt + RMB** — dolly; **MMB** — pan.
- **F** — focus the selected actor; **Ctrl + 1..9** — camera bookmarks.

Click an actor to select it; an orange outline marks the selection. The translate/rotate/scale
**gizmos** are engine-rendered directly in the native viewport (the "airspace" rule: HTML never
draws over the 3D view). Drag a gizmo handle to transform the selection — hold **Shift** to snap.

## Edit properties in the Details panel

With an actor selected, the **Details** panel (below the Outliner) shows its components as editable
properties, driven by reflection over the real ECS components. Transforms use UE-style euler-degree
rotation; you get numeric drag fields, vec3 editors, color pickers, enum dropdowns, and asset-ref
pickers per type. Every edit is undoable (**Ctrl+Z** / **Ctrl+Y**), a gizmo drag counts as one
undo step, and you can multi-select actors to edit shared properties at once. Use per-property
**reset** to return a value to its default.

## Place assets from the Content Drawer

Press **Ctrl+Space** (or click **Content Drawer** in the status bar) to slide up your project's
assets — meshes, materials, textures, and more — in a virtualized thumbnail grid with a folder
tree, breadcrumbs, filters, and fuzzy search. Import new assets by dragging files in (glTF/GLB
meshes, PNG/EXR/HDR textures). To place an asset, drag it from the drawer onto the viewport.

## Save, and trust the reload

Press **Ctrl+S** to save the level. The `.inf_lvl` write is deterministic: save, restart, and
reload, and you get a byte-identical world — and the TOML sidecar is readable in a `git diff`. The
editor also autosaves a crash-recovery file every few seconds, so an unexpected exit never costs
you unsaved work.
