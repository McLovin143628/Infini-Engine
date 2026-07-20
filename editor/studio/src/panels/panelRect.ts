/**
 * Pure rect math for floating workspace panels (ported from GeoCanvas).
 * Panel rects are top-left based (`x`/`y` = offset from the workspace
 * container's top-left, CSS px). No React, no stores.
 */

export interface PanelRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface ContainerSize {
  w: number;
  h: number;
}

/** Resize-handle ids: compass points + corners. */
export type HandleId = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

export const HANDLE_IDS: HandleId[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

export const HANDLE_CURSORS: Record<HandleId, string> = {
  n: "ns-resize",
  s: "ns-resize",
  e: "ew-resize",
  w: "ew-resize",
  nw: "nwse-resize",
  se: "nwse-resize",
  ne: "nesw-resize",
  sw: "nesw-resize",
};

/** Minimum open-panel size (panel bodies scroll internally). */
export const MIN_PANEL_W = 220;
export const MIN_PANEL_H = 160;

/** Height of a panel's header strip — the drag handle that clamping keeps
 *  reachable. Doesn't need to be pixel-perfect; used for clamp bounds. */
export const PANEL_HEADER_H = 32;

/** Horizontal sliver of the header that must stay inside the container so
 *  an off-screen panel can always be grabbed back. */
export const MIN_VISIBLE_PX = 48;

/**
 * Apply a resize-handle drag to a start rect. Top-left math: the opposite
 * edge/corner stays anchored, the driven edges follow the pointer, clamped
 * to the container bounds and the minimum size.
 */
export function resizePanelRect(
  start: PanelRect,
  handle: HandleId,
  dx: number,
  dy: number,
  min: { w: number; h: number },
  container: ContainerSize,
): PanelRect {
  let left = start.x;
  let top = start.y;
  let right = start.x + start.w;
  let bottom = start.y + start.h;

  if (handle.includes("w")) {
    left = clamp(start.x + dx, 0, right - min.w);
  }
  if (handle.includes("e")) {
    right = clamp(start.x + start.w + dx, left + min.w, container.w);
  }
  if (handle.includes("n")) {
    top = clamp(start.y + dy, 0, bottom - min.h);
  }
  if (handle.includes("s")) {
    bottom = clamp(start.y + start.h + dy, top + min.h, container.h);
  }

  return { x: left, y: top, w: right - left, h: bottom - top };
}

/**
 * Pull a rect's position back so its header strip stays reachable inside
 * the container: at least `MIN_VISIBLE_PX` of the header remains visible
 * horizontally, and the header row can never leave the container
 * vertically. Only the position moves — size is left alone (re-growing
 * the window restores un-clamped positions from the persisted values).
 */
export function clampPanelRect(
  rect: PanelRect,
  container: ContainerSize,
  headerH: number = PANEL_HEADER_H,
): PanelRect {
  const x = clamp(
    rect.x,
    -(rect.w - MIN_VISIBLE_PX),
    Math.max(-(rect.w - MIN_VISIBLE_PX), container.w - MIN_VISIBLE_PX),
  );
  const y = clamp(rect.y, 0, Math.max(0, container.h - headerH));
  return { x, y, w: rect.w, h: rect.h };
}

function clamp(n: number, lo: number, hi: number): number {
  if (!Number.isFinite(n)) return lo;
  if (n < lo) return lo;
  if (n > hi) return hi;
  return n;
}
