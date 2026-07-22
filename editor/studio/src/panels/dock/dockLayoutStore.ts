import { create } from "zustand";
import { layouts } from "../../lib/ipc";
import type { Discipline } from "../../lib/disciplines";
import { clampPanelRect, type ContainerSize, type PanelRect } from "../panelRect";
import { panelDefFor, panelTypeOf } from "../panelRegistry";
import { notifyPanelClosed } from "./panelLifecycle";
import {
  clampRegionSize,
  fromDockDoc,
  newGroupId,
  toDockDoc,
  DOCK_LAYOUT_VERSION,
  DOCK_SIDES,
  type DockLayout,
  type DockLayoutDoc,
  type DockRegionState,
  type DockSide,
  type DockTarget,
  type PanelState,
} from "./dockTypes";

/**
 * The dock layout store (ported from GeoCanvas, ROADMAP P1.2). Owns where
 * every panel lives (docked / floating / detached window / hidden), the
 * four regions' sizes + tab groups, and floating rects/z.
 *
 * Persistence contract:
 *  - The ACTIVE layout auto-persists under the reserved preset name
 *    (`ACTIVE_LAYOUT_NAME`) through the `layouts` IPC domain. Named presets
 *    (Window ▸ Save/Load Layout) use the same domain with user names.
 *  - Live gestures (`setFloatRect`, `setRegionSize`, `setGroupWeights`)
 *    never persist; the matching `commit*` writes the whole dock tree once
 *    per gesture. Raise-only changes idle-flush after 2 s.
 */

/** Reserved preset name for the auto-persisted active layout. */
export const ACTIVE_LAYOUT_NAME = "__active";

interface DockLayoutStore {
  layout: DockLayout;
  hydrated: boolean;
  /** Live size of the workspace area hosting docks + floats. */
  container: ContainerSize | null;

  setContainer: (size: ContainerSize) => void;
  /** Hydrate from a saved doc (or seed the default when null/stale). */
  hydrate: (doc: DockLayoutDoc | null, container: ContainerSize) => void;

  /** Create-or-focus a panel instance; returns the instance id. */
  openPanel: (type: string, params?: string | null) => string;
  /** Remove a dynamic instance entirely; singletons are hidden instead. */
  closePanel: (id: string) => void;

  /** The one location-transition transaction (drag release, menu action). */
  applyDrop: (id: string, target: DockTarget) => void;
  hidePanel: (id: string) => void;
  /** Restore a hidden panel to `lastDock` (or floating). */
  showPanel: (id: string) => void;
  activateTab: (id: string) => void;
  setMinimized: (id: string, minimized: boolean) => void;

  /** Live float move/resize — no persist. */
  setFloatRect: (id: string, rect: PanelRect) => void;
  commitFloat: (id: string) => void;
  bringToFront: (id: string) => void;
  /** Detached-window geometry report (physical px) — idle-persisted. */
  setWindowRect: (id: string, rect: { x: number; y: number; w: number; h: number }) => void;

  /** Live region resize — no persist. */
  setRegionSize: (side: DockSide, px: number) => void;
  /** Live group-weight change within a region — no persist. */
  setGroupWeights: (side: DockSide, weights: number[]) => void;
  commitRegion: () => void;

  /** Pull floating headers back into the container (window shrink). */
  clampAll: (container: ContainerSize) => void;
  resetLayout: () => void;
}

// =============================================================================
// Persistence (active layout ⇄ `layouts` IPC domain)
// =============================================================================

let idleFlushTimer: ReturnType<typeof setTimeout> | null = null;

/** Fire-and-forget whole-tree write (one per completed gesture). */
function persist(layout: DockLayout): void {
  if (idleFlushTimer) {
    clearTimeout(idleFlushTimer);
    idleFlushTimer = null;
  }
  void layouts.save(ACTIVE_LAYOUT_NAME, JSON.stringify(toDockDoc(layout))).catch(() => {});
}

/** Deferred persist for raise-only changes: flushes after 2 s idle (or is
 *  superseded by the next real gesture's `persist`). */
function persistIdle(): void {
  if (idleFlushTimer) clearTimeout(idleFlushTimer);
  idleFlushTimer = setTimeout(() => {
    idleFlushTimer = null;
    persist(useDockLayout.getState().layout);
  }, 2000);
}

/** Load the persisted active layout doc (null when none / unparsable). */
export async function loadActiveLayoutDoc(): Promise<DockLayoutDoc | null> {
  try {
    const json = await layouts.load(ACTIVE_LAYOUT_NAME);
    if (!json) return null;
    return JSON.parse(json) as DockLayoutDoc;
  } catch {
    return null;
  }
}

// =============================================================================
// Pure layout surgery (all return new objects; callers persist)
// =============================================================================

function emptyRegions(): Record<DockSide, DockRegionState> {
  return {
    left: { size: 300, groups: [] },
    right: { size: 320, groups: [] },
    top: { size: 200, groups: [] },
    bottom: { size: 240, groups: [] },
  };
}

/** Remove `id` from whatever group holds it; prune empty groups. */
function detachFromGroups(layout: DockLayout, id: string): DockLayout {
  const docks = { ...layout.docks };
  for (const side of DOCK_SIDES) {
    const region = docks[side];
    if (!region.groups.some((g) => g.tabs.includes(id))) continue;
    const groups = region.groups
      .map((g) =>
        g.tabs.includes(id)
          ? {
              ...g,
              tabs: g.tabs.filter((t) => t !== id),
              activeTab:
                g.activeTab === id ? (g.tabs.filter((t) => t !== id)[0] ?? "") : g.activeTab,
            }
          : g,
      )
      .filter((g) => g.tabs.length > 0);
    docks[side] = { ...region, groups };
  }
  return { ...layout, docks };
}

/** Insert `id` per the target; assumes it's already detached from groups. */
function attach(layout: DockLayout, id: string, target: DockTarget): DockLayout {
  const panel = layout.panels[id];
  if (!panel) return layout;
  const panels = { ...layout.panels };
  const docks = { ...layout.docks };

  switch (target.kind) {
    case "dock": {
      const region = docks[target.side];
      let groups = region.groups;
      const gi = groups.findIndex((g) => g.id === target.groupId);
      if (gi === -1) {
        // Group vanished mid-gesture — degrade to a new group at the end.
        return attach(layout, id, {
          kind: "new-group",
          side: target.side,
          index: groups.length,
        });
      }
      const g = groups[gi]!;
      const tabs = [...g.tabs];
      tabs.splice(Math.max(0, Math.min(target.index, tabs.length)), 0, id);
      groups = [...groups];
      groups[gi] = { ...g, tabs, activeTab: id };
      docks[target.side] = { ...region, groups };
      panels[id] = {
        ...panel,
        location: { kind: "docked", side: target.side, groupId: g.id },
        lastDock: target.side,
      };
      break;
    }
    case "new-group": {
      const region = docks[target.side];
      const groups = [...region.groups];
      const group = {
        id: newGroupId(),
        tabs: [id],
        activeTab: id,
        weight: groups.length > 0 ? avgWeight(groups) : 1,
      };
      groups.splice(Math.max(0, Math.min(target.index, groups.length)), 0, group);
      docks[target.side] = { ...region, groups };
      panels[id] = {
        ...panel,
        location: { kind: "docked", side: target.side, groupId: group.id },
        lastDock: target.side,
      };
      break;
    }
    case "float": {
      panels[id] = {
        ...panel,
        location: { kind: "floating" },
        floatRect: target.rect,
        minimized: false,
      };
      break;
    }
    case "window": {
      // Fresh detach spawns at the drop cursor — clearing the remembered
      // geometry is what makes the window appear under the pointer.
      panels[id] = { ...panel, location: { kind: "window" }, windowRect: null };
      break;
    }
  }
  return raiseFloat({ panels, docks }, id);
}

function avgWeight(groups: { weight: number }[]): number {
  const sum = groups.reduce((a, g) => a + g.weight, 0);
  return groups.length > 0 && sum > 0 ? sum / groups.length : 1;
}

/** Raise a floating panel's z to the top; normalize 0..n-1. No-op for
 *  non-floating panels. */
function raiseFloat(layout: DockLayout, id: string): DockLayout {
  const p = layout.panels[id];
  if (!p || p.location.kind !== "floating") return layout;
  const floats = Object.entries(layout.panels)
    .filter(([, s]) => s.location.kind === "floating")
    .sort(([, a], [, b]) => a.floatZ - b.floatZ)
    .map(([pid]) => pid)
    .filter((pid) => pid !== id);
  floats.push(id);
  const panels = { ...layout.panels };
  floats.forEach((pid, i) => {
    if (panels[pid]!.floatZ !== i) panels[pid] = { ...panels[pid]!, floatZ: i };
  });
  return { ...layout, panels };
}

/**
 * Enforce every structural invariant after hydrate (or any external data):
 * docked panels point at real groups that contain them, groups only carry
 * real docked panels, no empty groups, activeTab ∈ tabs, weights positive,
 * unknown panel types dropped (registry must be populated first).
 */
export function repairLayout(layout: DockLayout): DockLayout {
  const panels: Record<string, PanelState> = {};
  for (const [id, p] of Object.entries(layout.panels)) {
    const def = panelDefFor(id);
    if (!def) continue; // unregistered type → drop
    if (def.transient) continue; // session-only → never restore
    panels[id] = p;
  }

  const docks = { ...layout.docks };
  // Pass 1: groups keep only tabs whose panel is docked to exactly them.
  for (const side of DOCK_SIDES) {
    const region = docks[side];
    const groups = region.groups
      .map((g) => {
        const tabs = g.tabs.filter((t) => {
          const p = panels[t];
          return (
            p &&
            p.location.kind === "docked" &&
            p.location.side === side &&
            p.location.groupId === g.id
          );
        });
        const activeTab = tabs.includes(g.activeTab) ? g.activeTab : (tabs[0] ?? "");
        return { ...g, tabs, activeTab, weight: g.weight > 0 ? g.weight : 1 };
      })
      .filter((g) => g.tabs.length > 0);
    docks[side] = { ...region, groups };
  }
  // Pass 2: docked panels whose group evaporated fall back to floating.
  for (const [id, p] of Object.entries(panels)) {
    if (p.location.kind !== "docked") continue;
    const loc = p.location;
    const ok = docks[loc.side].groups.some((g) => g.id === loc.groupId && g.tabs.includes(id));
    if (!ok) panels[id] = { ...p, location: { kind: "floating" } };
  }
  // Pass 3: normalize float z.
  let repaired: DockLayout = { panels, docks };
  const floats = Object.entries(panels)
    .filter(([, s]) => s.location.kind === "floating")
    .sort(([, a], [, b]) => a.floatZ - b.floatZ);
  floats.forEach(([pid], i) => {
    if (repaired.panels[pid]!.floatZ !== i) {
      repaired = {
        ...repaired,
        panels: {
          ...repaired.panels,
          [pid]: { ...repaired.panels[pid]!, floatZ: i },
        },
      };
    }
  });
  return repaired;
}

// =============================================================================
// Defaults
// =============================================================================

/**
 * First-run layout, UE-parity (ROADMAP §4): right sidebar with the Outliner
 * group stacked over the Details group; everything else hidden but
 * pre-seeded with a sensible `lastDock` so the Window-menu toggles restore
 * each panel where it belongs (Output Log → bottom, Place Actors → left,
 * World Settings → right).
 */
export function defaultDockLayout(): DockLayout {
  const outlinerGroup = newGroupId();
  const detailsGroup = newGroupId();

  const base = (
    type: string,
    location: PanelState["location"],
    lastDock: DockSide | null,
    size: { w: number; h: number },
  ): PanelState => ({
    type,
    params: null,
    location,
    floatRect: { x: 24, y: 24, w: size.w, h: size.h },
    floatZ: 0,
    minimized: false,
    lastDock,
    windowRect: null,
  });

  const docks = emptyRegions();
  docks.right = {
    size: 320,
    groups: [
      { id: outlinerGroup, tabs: ["outliner"], activeTab: "outliner", weight: 0.42 },
      { id: detailsGroup, tabs: ["details"], activeTab: "details", weight: 0.58 },
    ],
  };

  return {
    panels: {
      outliner: base(
        "outliner",
        { kind: "docked", side: "right", groupId: outlinerGroup },
        "right",
        { w: 320, h: 420 },
      ),
      details: base(
        "details",
        { kind: "docked", side: "right", groupId: detailsGroup },
        "right",
        { w: 320, h: 480 },
      ),
      outputLog: base("outputLog", { kind: "hidden" }, "bottom", { w: 720, h: 260 }),
      placeActors: base("placeActors", { kind: "hidden" }, "left", { w: 300, h: 460 }),
      worldSettings: base("worldSettings", { kind: "hidden" }, "right", { w: 320, h: 420 }),
    },
    docks,
  };
}

// =============================================================================
// Discipline layout presets (P15.3): first-run layouts per discipline
// =============================================================================

/** A single docked panel state with a sensible float fallback. */
function basePanel(
  type: string,
  location: PanelState["location"],
  lastDock: DockSide | null,
): PanelState {
  const size = panelDefFor(type)?.defaultSize ?? { w: 340, h: 440 };
  return {
    type,
    params: null,
    location,
    floatRect: { x: 24, y: 24, w: size.w, h: size.h },
    floatZ: 0,
    minimized: false,
    lastDock,
    windowRect: null,
  };
}

interface RegionSpec {
  side: DockSide;
  size: number;
  groups: Array<{ weight: number; tabs: string[] }>;
}

/** Build a layout from a docked-region spec + a list of pre-seeded hidden panels. */
function buildLayout(
  regions: RegionSpec[],
  hidden: Array<{ type: string; lastDock: DockSide | null }>,
): DockLayout {
  const docks = emptyRegions();
  const panels: Record<string, PanelState> = {};
  for (const rs of regions) {
    const groups = rs.groups
      .filter((g) => g.tabs.length > 0)
      .map((g) => {
        const id = newGroupId();
        for (const t of g.tabs) {
          panels[t] = basePanel(t, { kind: "docked", side: rs.side, groupId: id }, rs.side);
        }
        return { id, tabs: [...g.tabs], activeTab: g.tabs[0]!, weight: g.weight };
      });
    docks[rs.side] = { size: rs.size, groups };
  }
  for (const h of hidden) {
    if (panels[h.type]) continue;
    panels[h.type] = basePanel(h.type, { kind: "hidden" }, h.lastDock);
  }
  return { panels, docks };
}

/**
 * The three first-run discipline layouts (ROADMAP §4). `repairLayout` runs on
 * every one so an unregistered panel type degrades gracefully.
 *  - **3d**: Place Actors (left) · Outliner/Details (right) · Output Log (bottom).
 *  - **2d**: Outliner/Details (right) · Output Log (bottom) — viewport-forward.
 *  - **scripting**: Explorer (left) · Outliner/Details (right) · Editor/Terminal/
 *    Problems tabs (bottom) — IDE-forward.
 */
export function disciplineLayout(kind: Discipline): DockLayout {
  let layout: DockLayout;
  switch (kind) {
    case "2d":
      layout = buildLayout(
        [
          {
            side: "right",
            size: 320,
            groups: [
              { weight: 0.42, tabs: ["outliner"] },
              { weight: 0.58, tabs: ["details"] },
            ],
          },
          { side: "bottom", size: 220, groups: [{ weight: 1, tabs: ["outputLog"] }] },
        ],
        [
          { type: "placeActors", lastDock: "left" },
          { type: "worldSettings", lastDock: "right" },
        ],
      );
      break;
    case "scripting":
      layout = buildLayout(
        [
          { side: "left", size: 300, groups: [{ weight: 1, tabs: ["explorer"] }] },
          {
            side: "right",
            size: 300,
            groups: [
              { weight: 0.5, tabs: ["outliner"] },
              { weight: 0.5, tabs: ["details"] },
            ],
          },
          {
            side: "bottom",
            size: 300,
            groups: [{ weight: 1, tabs: ["editor", "terminal", "problems"] }],
          },
        ],
        [
          { type: "outputLog", lastDock: "bottom" },
          { type: "placeActors", lastDock: "left" },
          { type: "worldSettings", lastDock: "right" },
        ],
      );
      break;
    case "3d":
    default:
      layout = buildLayout(
        [
          { side: "left", size: 280, groups: [{ weight: 1, tabs: ["placeActors"] }] },
          {
            side: "right",
            size: 320,
            groups: [
              { weight: 0.42, tabs: ["outliner"] },
              { weight: 0.58, tabs: ["details"] },
            ],
          },
          { side: "bottom", size: 220, groups: [{ weight: 1, tabs: ["outputLog"] }] },
        ],
        [{ type: "worldSettings", lastDock: "right" }],
      );
      break;
  }
  return repairLayout(layout);
}

/**
 * Swap the live layout to a discipline preset and persist it as the active
 * layout. Used by the New Project gallery (first-run) and Window ▸ Layout.
 */
export function applyDisciplineLayout(kind: Discipline): void {
  const store = useDockLayout.getState();
  const container = store.container ?? { w: 1280, h: 720 };
  store.hydrate(toDockDoc(disciplineLayout(kind)), container);
  void layouts.save(ACTIVE_LAYOUT_NAME, JSON.stringify(toDockDoc(useDockLayout.getState().layout)));
}

// =============================================================================
// The store
// =============================================================================

export const useDockLayout = create<DockLayoutStore>((set, get) => {
  /** Apply a transaction + persist in one step. */
  const commitTx = (next: DockLayout) => {
    set({ layout: next });
    persist(next);
  };

  return {
    layout: defaultDockLayout(),
    hydrated: false,
    container: null,

    setContainer: (container) => set({ container }),

    hydrate: (doc, container) => {
      let layout: DockLayout;
      // Version-gated: a layout saved before the current DOCK_LAYOUT_VERSION
      // is discarded and re-seeded so schema bumps reach existing installs.
      if (doc && (doc.version ?? 0) >= DOCK_LAYOUT_VERSION) {
        layout = repairLayout(fromDockDoc(doc));
        const docks = { ...layout.docks };
        for (const side of DOCK_SIDES) {
          docks[side] = {
            ...docks[side],
            size: clampRegionSize(side, docks[side].size, container),
          };
        }
        const panels = { ...layout.panels };
        for (const [id, p] of Object.entries(panels)) {
          panels[id] = { ...p, floatRect: clampPanelRect(p.floatRect, container) };
        }
        layout = { panels, docks };
      } else {
        layout = defaultDockLayout();
      }
      set({ layout, hydrated: true, container });
    },

    openPanel: (type, params = null) => {
      const def = panelDefFor(type);
      const id = def?.singleton !== false ? type : `${type}:${params ?? ""}`;
      const { layout, container } = get();
      const existing = layout.panels[id];
      if (existing) {
        if (existing.location.kind === "hidden") get().showPanel(id);
        else if (existing.location.kind === "window") {
          // Re-opening a detached panel = bring its OS window forward.
          void import("../window/panelWindows").then(({ focusPanelWindow }) =>
            focusPanelWindow(id),
          );
        } else get().activateTab(id);
        return id;
      }
      const size = def?.defaultSize ?? { w: 340, h: 420 };
      const c = container ?? { w: 1280, h: 720 };
      const panel: PanelState = {
        type: panelTypeOf(id),
        params,
        location: { kind: "hidden" },
        floatRect: {
          x: Math.max(12, (c.w - size.w) / 2),
          y: Math.max(12, (c.h - size.h) / 3),
          w: size.w,
          h: size.h,
        },
        floatZ: 0,
        minimized: false,
        lastDock: null,
        windowRect: null,
      };
      let next: DockLayout = {
        ...layout,
        panels: { ...layout.panels, [id]: panel },
      };
      const loc = def?.defaultLocation ?? "float";
      if (loc === "float") {
        next = attach(next, id, { kind: "float", rect: panel.floatRect });
      } else if (loc === "window") {
        next = attach(next, id, { kind: "window" });
      } else if (loc !== "hidden") {
        next = dockOntoSide(next, id, loc);
      }
      commitTx(next);
      return id;
    },

    closePanel: (id) => {
      const { layout } = get();
      const p = layout.panels[id];
      if (!p) return;
      const def = panelDefFor(id);
      if (def?.singleton !== false) {
        get().hidePanel(id);
        return;
      }
      let next = detachFromGroups(layout, id);
      const panels = { ...next.panels };
      delete panels[id];
      next = { ...next, panels };
      commitTx(next);
      // Explicit close (instance destroyed) → let resource owners free up
      // (e.g. the terminal's PTY). A location move goes through `applyDrop`,
      // which never notifies, so a drag keeps the resource alive.
      notifyPanelClosed(id);
    },

    applyDrop: (id, target) => {
      const { layout } = get();
      if (!layout.panels[id]) return;
      commitTx(attach(detachFromGroups(layout, id), id, target));
    },

    hidePanel: (id) => {
      const { layout } = get();
      const p = layout.panels[id];
      if (!p || p.location.kind === "hidden") return;
      // Docked → remember this side. Floating → restore floating (drop the
      // memory). Detached window → KEEP the previous dock memory: closing a
      // window should bring the panel back where it lived before detaching.
      const lastDock =
        p.location.kind === "docked"
          ? p.location.side
          : p.location.kind === "window"
            ? p.lastDock
            : null;
      let next = detachFromGroups(layout, id);
      next = {
        ...next,
        panels: {
          ...next.panels,
          [id]: { ...next.panels[id]!, location: { kind: "hidden" }, lastDock },
        },
      };
      commitTx(next);
      // Hiding (the singleton "close" path) is an explicit close from the
      // user's view — free panel-owned resources. Moves use `applyDrop`.
      notifyPanelClosed(id);
    },

    showPanel: (id) => {
      const { layout } = get();
      const p = layout.panels[id];
      if (!p || p.location.kind !== "hidden") return;
      let next: DockLayout;
      if (p.lastDock) {
        next = dockOntoSide(layout, id, p.lastDock);
      } else if (panelDefFor(id)?.defaultLocation === "window") {
        // A panel that LIVES in its own OS window restores as one.
        next = attach(layout, id, { kind: "window" });
      } else {
        const rect = get().container
          ? clampPanelRect(p.floatRect, get().container!)
          : p.floatRect;
        next = attach(layout, id, { kind: "float", rect });
      }
      commitTx(next);
    },

    activateTab: (id) => {
      const { layout } = get();
      const p = layout.panels[id];
      if (!p) return;
      if (p.location.kind === "floating") {
        const next = raiseFloat(layout, id);
        if (next !== layout) {
          set({ layout: next });
          persistIdle();
        }
        return;
      }
      if (p.location.kind !== "docked") return;
      const { side, groupId } = p.location;
      const region = layout.docks[side];
      const gi = region.groups.findIndex((g) => g.id === groupId);
      if (gi === -1 || region.groups[gi]!.activeTab === id) return;
      const groups = [...region.groups];
      groups[gi] = { ...groups[gi]!, activeTab: id };
      const next = {
        ...layout,
        docks: { ...layout.docks, [side]: { ...region, groups } },
      };
      set({ layout: next });
      persistIdle();
    },

    setMinimized: (id, minimized) => {
      const { layout } = get();
      const p = layout.panels[id];
      if (!p || p.location.kind !== "floating" || p.minimized === minimized) {
        return;
      }
      commitTx({
        ...layout,
        panels: { ...layout.panels, [id]: { ...p, minimized } },
      });
    },

    setFloatRect: (id, rect) =>
      set((s) => {
        const p = s.layout.panels[id];
        if (!p) return s;
        return {
          layout: {
            ...s.layout,
            panels: { ...s.layout.panels, [id]: { ...p, floatRect: rect } },
          },
        };
      }),

    commitFloat: () => persist(get().layout),

    setWindowRect: (id, rect) => {
      const { layout } = get();
      const p = layout.panels[id];
      if (!p) return;
      const next = {
        ...layout,
        panels: { ...layout.panels, [id]: { ...p, windowRect: rect } },
      };
      set({ layout: next });
      persistIdle();
    },

    bringToFront: (id) => {
      const { layout } = get();
      const next = raiseFloat(layout, id);
      if (next === layout) return;
      set({ layout: next });
      persistIdle();
    },

    setRegionSize: (side, px) =>
      set((s) => {
        const c = s.container ?? { w: 1920, h: 1080 };
        return {
          layout: {
            ...s.layout,
            docks: {
              ...s.layout.docks,
              [side]: {
                ...s.layout.docks[side],
                size: clampRegionSize(side, px, c),
              },
            },
          },
        };
      }),

    setGroupWeights: (side, weights) =>
      set((s) => {
        const region = s.layout.docks[side];
        if (weights.length !== region.groups.length) return s;
        const groups = region.groups.map((g, i) => ({
          ...g,
          weight: Math.max(0.1, weights[i] ?? g.weight),
        }));
        return {
          layout: {
            ...s.layout,
            docks: { ...s.layout.docks, [side]: { ...region, groups } },
          },
        };
      }),

    commitRegion: () => persist(get().layout),

    clampAll: (container) =>
      set((s) => {
        let changed = false;
        const panels = { ...s.layout.panels };
        for (const [id, p] of Object.entries(panels)) {
          if (p.location.kind !== "floating") continue;
          const rect = clampPanelRect(p.floatRect, container);
          if (rect.x !== p.floatRect.x || rect.y !== p.floatRect.y) {
            panels[id] = { ...p, floatRect: rect };
            changed = true;
          }
        }
        const docks = { ...s.layout.docks };
        for (const side of DOCK_SIDES) {
          const size = clampRegionSize(side, docks[side].size, container);
          if (size !== docks[side].size) {
            docks[side] = { ...docks[side], size };
            changed = true;
          }
        }
        return changed ? { layout: { panels, docks }, container } : { container };
      }),

    resetLayout: () => {
      const layout = defaultDockLayout();
      set({ layout, hydrated: true });
      persist(layout);
    },
  };
});

/** Dock `id` onto `side`: join the side's first group, or open a new one. */
function dockOntoSide(layout: DockLayout, id: string, side: DockSide): DockLayout {
  const detached = detachFromGroups(layout, id);
  const first = detached.docks[side].groups[0];
  return attach(
    detached,
    id,
    first
      ? { kind: "dock", side, groupId: first.id, index: first.tabs.length }
      : { kind: "new-group", side, index: 0 },
  );
}

// =============================================================================
// Named layout presets (Window ▸ Save/Load Layout)
// =============================================================================

/** Persist the current layout under a user-chosen preset name. */
export async function saveLayoutPreset(name: string): Promise<void> {
  await layouts.save(name, JSON.stringify(toDockDoc(useDockLayout.getState().layout)));
}

/** Load a preset into the live layout (and re-persist it as active). */
export async function loadLayoutPreset(name: string): Promise<boolean> {
  const json = await layouts.load(name);
  if (!json) return false;
  const store = useDockLayout.getState();
  const container = store.container ?? { w: 1280, h: 720 };
  try {
    store.hydrate(JSON.parse(json) as DockLayoutDoc, container);
  } catch {
    return false;
  }
  persist(useDockLayout.getState().layout);
  return true;
}

// =============================================================================
// Selector helpers
// =============================================================================

/** Panel is on screen (docked / floating / window). */
export function isPanelVisible(layout: DockLayout, id: string): boolean {
  const p = layout.panels[id];
  return !!p && p.location.kind !== "hidden";
}

/** Ordered floating panel ids, back-most first. */
export function floatingPanelIds(layout: DockLayout): string[] {
  return Object.entries(layout.panels)
    .filter(([, p]) => p.location.kind === "floating")
    .sort(([, a], [, b]) => a.floatZ - b.floatZ)
    .map(([id]) => id);
}
