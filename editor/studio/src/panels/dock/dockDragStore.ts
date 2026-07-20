import { create } from "zustand";
import type { PanelRect } from "../panelRect";
import type { DockTarget } from "./dockTypes";

/**
 * Transient panel-drag state. NEVER persisted — the persisted model lives
 * in `dockLayoutStore`; this store only exists for the duration of one
 * gesture, driving the drag ghost + drop previews in `DragLayer`.
 *
 * All rects are VIEWPORT CSS px (the drag layer portals into
 * `document.body`); `dockLayoutStore` converts to workspace-relative on
 * release using the workspace element registered here.
 */

export interface ActiveDropTarget {
  target: DockTarget;
  /** Preview rect to render (viewport px). */
  preview: PanelRect;
  /** Extra hint for the tab-insert caret (viewport px x-position). */
  tabCaret?: { x: number; y: number; h: number };
}

export interface DragState {
  panelId: string;
  /** Where the gesture started: a dock tab or a floating header. */
  origin: "tab" | "float";
  /** Ghost card top-left (viewport px). */
  ghost: { x: number; y: number };
  /** Pointer offset inside the ghost. */
  grabOffset: { x: number; y: number };
  /** Current pointer position (viewport px). */
  pointer: { x: number; y: number };
  target: ActiveDropTarget | null;
  /** True while the floating panel itself is being live-moved (no ghost). */
  liveMove: boolean;
}

interface DockDragStore {
  dragging: DragState | null;
  /** The workspace host element (set by DockWorkspace) — the coordinate
   *  anchor for float rects and edge bands. */
  workspaceEl: HTMLElement | null;
  setWorkspaceEl: (el: HTMLElement | null) => void;
  start: (state: DragState) => void;
  update: (patch: Partial<DragState>) => void;
  end: () => void;
}

export const useDockDrag = create<DockDragStore>((set) => ({
  dragging: null,
  workspaceEl: null,
  setWorkspaceEl: (workspaceEl) => set({ workspaceEl }),
  start: (dragging) => set({ dragging }),
  update: (patch) => set((s) => (s.dragging ? { dragging: { ...s.dragging, ...patch } } : s)),
  end: () => set({ dragging: null }),
}));
