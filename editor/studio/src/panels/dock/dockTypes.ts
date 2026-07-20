import type { ContainerSize, PanelRect } from "../panelRect";

/**
 * Runtime model + persisted document for the dock layout (ported from
 * GeoCanvas, ROADMAP P1.2). Panels are a `Record` keyed by id at runtime
 * (O(1) lookup everywhere) and an array in the persisted JSON document
 * (`DockLayoutDoc`) so closed panels actually disappear on write.
 *
 * Design rules (each fixes a documented pain point in the donor project):
 *  - Transient gesture state (drag ghost, hover target) lives ONLY in
 *    `dockDragStore` — this model is 100% persistable.
 *  - A region's size is a dedicated field, never derived from a member
 *    panel's rect.
 *  - All four sides share one code path; axis math lives in
 *    `dockGeometry.ts` — no per-side special cases.
 *  - Tab groups are first-class; a region stacks 1..n groups by weight.
 */

export type DockSide = "left" | "right" | "top" | "bottom";
export const DOCK_SIDES: readonly DockSide[] = ["left", "right", "top", "bottom"];

/** Persisted dock-layout schema version. Layouts saved with an older
 *  version are discarded and re-seeded to the default at hydrate. */
export const DOCK_LAYOUT_VERSION = 1;

export type PanelLocation =
  | { kind: "docked"; side: DockSide; groupId: string }
  | { kind: "floating" }
  | { kind: "window" }
  | { kind: "hidden" };

export interface PanelState {
  /** Registered panel type (`panelRegistry`). For singletons, id === type. */
  type: string;
  /** Opaque per-instance parameter (dynamic panel types). */
  params: string | null;
  location: PanelLocation;
  /** Last/current floating rect, workspace-relative CSS px. */
  floatRect: PanelRect;
  /** Stacking among floating panels only (0 = back-most). */
  floatZ: number;
  /** Floating mode only: collapsed to the header strip. */
  minimized: boolean;
  /** Side to restore to when re-shown from hidden; null = restore floating. */
  lastDock: DockSide | null;
  /** Detached OS-window outer geometry, physical px. */
  windowRect: { x: number; y: number; w: number; h: number } | null;
}

export interface DockGroupState {
  /** Stable frontend-generated id (`newGroupId()`). */
  id: string;
  /** Panel ids in tab order. Never empty — empty groups are pruned. */
  tabs: string[];
  activeTab: string;
  /** Flex share of the region's main axis among the region's groups. */
  weight: number;
}

export interface DockRegionState {
  /** Cross-axis size in CSS px (width for left/right, height for top/bottom). */
  size: number;
  groups: DockGroupState[];
}

export interface DockLayout {
  panels: Record<string, PanelState>;
  docks: Record<DockSide, DockRegionState>;
}

/** Where a drop (or an explicit dock command) puts a panel. */
export type DockTarget =
  | {
      kind: "dock";
      side: DockSide;
      /** Existing group to join (tab insert) … */
      groupId: string;
      /** … at this tab index. */
      index: number;
    }
  | {
      kind: "new-group";
      side: DockSide;
      /** Insert position among the region's groups. */
      index: number;
    }
  | { kind: "float"; rect: PanelRect }
  | { kind: "window" };

/** Region size bounds (px / fraction of the workspace's relevant axis). */
export const MIN_DOCK_W = 200;
export const MIN_DOCK_H = 140;
export const MAX_DOCK_W_FRAC = 0.45;
export const MAX_DOCK_H_FRAC = 0.6;

export function clampRegionSize(
  side: DockSide,
  size: number,
  workspace: ContainerSize,
): number {
  const horizontal = side === "left" || side === "right";
  const min = horizontal ? MIN_DOCK_W : MIN_DOCK_H;
  const max = horizontal
    ? Math.max(min, workspace.w * MAX_DOCK_W_FRAC)
    : Math.max(min, workspace.h * MAX_DOCK_H_FRAC);
  return Math.round(Math.min(max, Math.max(min, size)));
}

let groupCounter = 0;
/** Stable-enough unique group id (persists as an opaque string). */
export function newGroupId(): string {
  groupCounter += 1;
  return `grp-${Date.now().toString(36)}-${groupCounter}`;
}

// =============================================================================
// Persisted document (JSON stored via the `layouts` IPC domain)
// =============================================================================

export interface DockPanelEntryDoc {
  id: string;
  panelType: string;
  params: string | null;
  location: PanelLocation;
  floatRect: PanelRect;
  floatZ: number;
  minimized: boolean;
  lastDock: DockSide | null;
  windowRect: { x: number; y: number; w: number; h: number } | null;
}

export interface DockRegionDoc {
  size: number;
  groups: DockGroupState[];
}

export interface DockLayoutDoc {
  version: number;
  panels: DockPanelEntryDoc[];
  left: DockRegionDoc;
  right: DockRegionDoc;
  top: DockRegionDoc;
  bottom: DockRegionDoc;
}

export function toDockDoc(layout: DockLayout): DockLayoutDoc {
  const panels: DockPanelEntryDoc[] = Object.entries(layout.panels).map(([id, p]) => ({
    id,
    panelType: p.type,
    params: p.params,
    location: p.location,
    floatRect: {
      x: Math.round(p.floatRect.x),
      y: Math.round(p.floatRect.y),
      w: Math.max(1, Math.round(p.floatRect.w)),
      h: Math.max(1, Math.round(p.floatRect.h)),
    },
    floatZ: Math.max(0, Math.round(p.floatZ)),
    minimized: p.minimized,
    lastDock: p.lastDock,
    windowRect: p.windowRect
      ? {
          x: Math.round(p.windowRect.x),
          y: Math.round(p.windowRect.y),
          w: Math.max(1, Math.round(p.windowRect.w)),
          h: Math.max(1, Math.round(p.windowRect.h)),
        }
      : null,
  }));
  const region = (r: DockRegionState): DockRegionDoc => ({
    size: Math.max(1, Math.round(r.size)),
    groups: r.groups.map((g) => ({
      id: g.id,
      tabs: [...g.tabs],
      activeTab: g.activeTab,
      weight: g.weight,
    })),
  });
  return {
    version: DOCK_LAYOUT_VERSION,
    panels,
    left: region(layout.docks.left),
    right: region(layout.docks.right),
    top: region(layout.docks.top),
    bottom: region(layout.docks.bottom),
  };
}

export function fromDockDoc(doc: DockLayoutDoc): DockLayout {
  const panels: Record<string, PanelState> = {};
  for (const e of doc.panels) {
    panels[e.id] = {
      type: e.panelType,
      params: e.params,
      location: e.location,
      floatRect: { ...e.floatRect },
      floatZ: e.floatZ,
      minimized: e.minimized,
      lastDock: e.lastDock,
      windowRect: e.windowRect ? { ...e.windowRect } : null,
    };
  }
  const region = (r: DockRegionDoc): DockRegionState => ({
    size: r.size,
    groups: r.groups.map((g) => ({
      id: g.id,
      tabs: [...g.tabs],
      activeTab: g.activeTab,
      weight: g.weight,
    })),
  });
  return {
    panels,
    docks: {
      left: region(doc.left),
      right: region(doc.right),
      top: region(doc.top),
      bottom: region(doc.bottom),
    },
  };
}
