import type { DockSide } from "./dockTypes";

/**
 * Axis math for the four dock regions — the single home for side-dependent
 * geometry so no component special-cases an edge (a side is just data).
 */

/** True for the vertical sidebars (left/right): cross-axis = width, groups
 *  stack vertically. False for top/bottom bands: cross-axis = height,
 *  groups stack horizontally. */
export function isHorizontalSide(side: DockSide): boolean {
  return side === "left" || side === "right";
}

/** CSS flex-direction for a region's group stack. */
export function groupStackDirection(side: DockSide): "column" | "row" {
  return isHorizontalSide(side) ? "column" : "row";
}

/** The CSS size property the region's `size` drives. */
export function crossSizeProp(side: DockSide): "width" | "height" {
  return isHorizontalSide(side) ? "width" : "height";
}

/** Resize-cursor for the region's center-facing splitter. */
export function splitterCursor(side: DockSide): "col-resize" | "row-resize" {
  return isHorizontalSide(side) ? "col-resize" : "row-resize";
}

/** +1 when dragging the splitter toward the workspace center grows the
 *  region, -1 when it shrinks it (right/bottom regions grow leftward/upward). */
export function splitterGrowSign(side: DockSide): 1 | -1 {
  return side === "left" || side === "top" ? 1 : -1;
}
