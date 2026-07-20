import type { PanelRect } from "../panelRect";
import type { ActiveDropTarget } from "./dockDragStore";
import { isHorizontalSide } from "./dockGeometry";
import { useDockLayout } from "./dockLayoutStore";
import { DOCK_SIDES, type DockLayout, type DockSide } from "./dockTypes";

/**
 * Drop-target hit testing for the unified panel drag. Every rect is
 * measured ONCE at drag start (`captureDropZones`) — pointermove does pure
 * math against the snapshot, zero layout reads per frame. Safe because a
 * panel's location never changes mid-gesture (nothing remounts or reflows
 * until release).
 *
 * Priority: tab strips (insert) > group bodies (join / split) > workspace
 * edge bands (dock as new group) > float.
 */

/** Edge band thickness inside the workspace edges. */
const EDGE_BAND = 48;
/** Fraction of a group body reserved for the before/after split zones. */
const SPLIT_FRAC = 0.25;

interface TabZone {
  panelId: string;
  rect: PanelRect; // viewport px
}

interface GroupZone {
  side: DockSide;
  groupId: string;
  groupIndex: number;
  stripRect: PanelRect;
  bodyRect: PanelRect;
  tabs: TabZone[];
}

export interface DropZones {
  workspace: PanelRect;
  /** The centre column (between full-height sidebars) — top/bottom edge-band
   *  previews inset to this so they don't visually overlap the sidebars. */
  center: PanelRect;
  groups: GroupZone[];
  /** Region sizes snapshotted at capture time (edge-band previews). */
  regionSizes: Record<DockSide, number>;
  regionGroupCounts: Record<DockSide, number>;
}

function rectOf(el: Element): PanelRect {
  const r = el.getBoundingClientRect();
  return { x: r.left, y: r.top, w: r.width, h: r.height };
}

function inRect(r: PanelRect, x: number, y: number): boolean {
  return x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h;
}

/** Snapshot every drop zone's geometry (call once, at drag start). */
export function captureDropZones(
  workspaceEl: HTMLElement,
  layoutIn?: DockLayout,
): DropZones {
  const layout = layoutIn ?? useDockLayout.getState().layout;
  const groups: GroupZone[] = [];
  for (const side of DOCK_SIDES) {
    layout.docks[side].groups.forEach((g, groupIndex) => {
      const section = workspaceEl.querySelector(`[data-dock-group="${CSS.escape(g.id)}"]`);
      if (!section) return;
      const strip = section.querySelector('[role="tablist"]');
      const tabs: TabZone[] = g.tabs.flatMap((panelId) => {
        const el = section.querySelector(`[data-dock-tab="${CSS.escape(panelId)}"]`);
        return el ? [{ panelId, rect: rectOf(el) }] : [];
      });
      groups.push({
        side,
        groupId: g.id,
        groupIndex,
        stripRect: strip ? rectOf(strip) : rectOf(section),
        bodyRect: rectOf(section),
        tabs,
      });
    });
  }
  const centerEl = workspaceEl.querySelector("[data-dock-center]");
  return {
    workspace: rectOf(workspaceEl),
    center: centerEl ? rectOf(centerEl) : rectOf(workspaceEl),
    groups,
    regionSizes: {
      left: layout.docks.left.size,
      right: layout.docks.right.size,
      top: layout.docks.top.size,
      bottom: layout.docks.bottom.size,
    },
    regionGroupCounts: {
      left: layout.docks.left.groups.length,
      right: layout.docks.right.groups.length,
      top: layout.docks.top.groups.length,
      bottom: layout.docks.bottom.groups.length,
    },
  };
}

/**
 * Resolve the pointer position to a drop target (or null = float).
 * `draggedId` never matches itself as a tab-insert reference.
 */
export function resolveDropTarget(
  zones: DropZones,
  x: number,
  y: number,
  draggedId: string,
): ActiveDropTarget | null {
  if (!inRect(zones.workspace, x, y)) return null;

  // 1. Tab strips — insert at the caret position.
  for (const g of zones.groups) {
    if (!inRect(g.stripRect, x, y)) continue;
    let index = 0;
    let caretX = g.stripRect.x + 4;
    for (const t of g.tabs) {
      if (t.panelId === draggedId) continue;
      const mid = t.rect.x + t.rect.w / 2;
      if (x > mid) {
        index += 1;
        caretX = t.rect.x + t.rect.w;
      }
    }
    return {
      target: { kind: "dock", side: g.side, groupId: g.groupId, index },
      preview: g.stripRect,
      tabCaret: { x: caretX, y: g.stripRect.y + 3, h: g.stripRect.h - 6 },
    };
  }

  // 2. Group bodies — before/after split bands along the region's main
  //    axis; the central area joins the group as a new tab.
  for (const g of zones.groups) {
    if (!inRect(g.bodyRect, x, y)) continue;
    const vertical = isHorizontalSide(g.side); // sidebars stack vertically
    const pos = vertical ? (y - g.bodyRect.y) / g.bodyRect.h : (x - g.bodyRect.x) / g.bodyRect.w;
    if (pos < SPLIT_FRAC) {
      return {
        target: { kind: "new-group", side: g.side, index: g.groupIndex },
        preview: splitPreview(g.bodyRect, vertical, "before"),
      };
    }
    if (pos > 1 - SPLIT_FRAC) {
      return {
        target: { kind: "new-group", side: g.side, index: g.groupIndex + 1 },
        preview: splitPreview(g.bodyRect, vertical, "after"),
      };
    }
    return {
      target: {
        kind: "dock",
        side: g.side,
        groupId: g.groupId,
        index: Number.MAX_SAFE_INTEGER,
      },
      preview: g.bodyRect,
    };
  }

  // 3. Workspace edge bands — dock as a new group on that side.
  const ws = zones.workspace;
  const bands: Array<[DockSide, boolean]> = [
    ["left", x < ws.x + EDGE_BAND],
    ["right", x > ws.x + ws.w - EDGE_BAND],
    ["top", y < ws.y + EDGE_BAND],
    ["bottom", y > ws.y + ws.h - EDGE_BAND],
  ];
  for (const [side, hit] of bands) {
    if (!hit) continue;
    const size = zones.regionSizes[side];
    const c = zones.center;
    // Sidebars span the full height; top/bottom bands inset to the centre
    // column (between the sidebars) so previews match the real layout.
    const preview: PanelRect =
      side === "left"
        ? { x: ws.x, y: ws.y, w: size, h: ws.h }
        : side === "right"
          ? { x: ws.x + ws.w - size, y: ws.y, w: size, h: ws.h }
          : side === "top"
            ? { x: c.x, y: c.y, w: c.w, h: size }
            : { x: c.x, y: c.y + c.h - size, w: c.w, h: size };
    return {
      target: { kind: "new-group", side, index: zones.regionGroupCounts[side] },
      preview,
    };
  }

  return null;
}

function splitPreview(body: PanelRect, vertical: boolean, where: "before" | "after"): PanelRect {
  if (vertical) {
    const h = body.h / 2;
    return { x: body.x, y: where === "before" ? body.y : body.y + h, w: body.w, h };
  }
  const w = body.w / 2;
  return { x: where === "before" ? body.x : body.x + w, y: body.y, w, h: body.h };
}
