/**
 * Live scene store (Phase 3): the frontend mirror of the backend ECS world.
 *
 * The authoritative world lives in Rust (`inf-editor-core::scene::SceneDoc`).
 * This store is a projection: it loads a full `SceneSnapshot`, then applies
 * incremental `world://delta` events. Every mutation is a command call — the
 * store never edits the tree locally except optimistically for selection. The
 * Outliner and Details panels (and the viewport, via the shared document) read
 * a single source of truth.
 */
import { create } from "zustand";

import type { DetailsDto } from "../bindings/DetailsDto";
import type { PropValueDto } from "../bindings/PropValueDto";
import type { SceneDelta } from "../bindings/SceneDelta";
import type { SceneNode } from "../bindings/SceneNode";
import type { SceneSnapshot } from "../bindings/SceneSnapshot";
import type { SpawnKind } from "../bindings/SpawnKind";
import { getCommand, setCommandHandler } from "../lib/commands";
import { listenTo, type UnlistenFn } from "../lib/events";
import { scene as sceneIpc } from "../lib/ipc";
import { registerBridgedStore } from "../panels/window/storeBridge";
import { useShellStore } from "./shellStore";

export type { SceneNode };

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/**
 * Monotonic token guarding async Details fetches. `refreshDetails`,
 * `setProperty`, and `resetProperty` all bump it before their await and only
 * apply their result if still newest — so a slow response for a stale
 * selection can't clobber the Details pane for the current one.
 */
let detailsToken = 0;

interface SceneState {
  nodes: Record<string, SceneNode>;
  roots: string[];
  selection: string[];
  /** Collapsed Outliner rows (guids). */
  collapsed: string[];
  version: number;
  dirty: boolean;
  title: string;
  canUndo: boolean;
  canRedo: boolean;
  undoLabel: string | null;
  redoLabel: string | null;
  details: DetailsDto | null;
  /** True once the first snapshot has loaded. */
  ready: boolean;

  applySnapshot: (s: SceneSnapshot) => void;
  applyDelta: (d: SceneDelta) => void;
  toggleCollapsed: (guid: string) => void;
  refreshDetails: () => Promise<void>;

  select: (guids: string[], additive?: boolean) => void;
  createEntity: (kind: SpawnKind, parent?: string | null) => Promise<string>;
  deleteSelected: () => void;
  rename: (guid: string, name: string) => void;
  reparent: (guid: string, parent: string | null) => void;
  toggleVisible: (guid: string) => void;
  setProperty: (typePath: string, field: string, value: PropValueDto) => Promise<void>;
  resetProperty: (typePath: string, field: string) => Promise<void>;
  undo: () => void;
  redo: () => void;
}

function nodeMap(nodes: SceneNode[]): Record<string, SceneNode> {
  const map: Record<string, SceneNode> = {};
  for (const n of nodes) map[n.guid] = n;
  return map;
}

function sameSet(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  const s = new Set(a);
  return b.every((x) => s.has(x));
}

export const useSceneStore = create<SceneState>((set, get) => ({
  nodes: {},
  roots: [],
  selection: [],
  collapsed: [],
  version: 0,
  dirty: false,
  title: "Untitled",
  canUndo: false,
  canRedo: false,
  undoLabel: null,
  redoLabel: null,
  details: null,
  ready: false,

  applySnapshot: (s) => {
    set({
      nodes: nodeMap(s.nodes),
      roots: s.roots,
      selection: s.selection,
      version: Number(s.version),
      dirty: s.dirty,
      title: s.title,
      canUndo: s.can_undo,
      canRedo: s.can_redo,
      undoLabel: s.undo_label,
      redoLabel: s.redo_label,
      ready: true,
    });
    void get().refreshDetails();
  },

  applyDelta: (d) => {
    const prevSelection = get().selection;
    set((state) => {
      const nodes = { ...state.nodes };
      for (const guid of d.removed) delete nodes[guid];
      for (const n of d.added) nodes[n.guid] = n;
      for (const n of d.updated) nodes[n.guid] = n;
      return {
        nodes,
        roots: d.roots,
        selection: d.selection,
        version: Number(d.version),
        dirty: d.dirty,
        title: d.title,
        canUndo: d.can_undo,
        canRedo: d.can_redo,
        undoLabel: d.undo_label,
        redoLabel: d.redo_label,
      };
    });
    // Re-fetch Details only when the selection set changed (property edits
    // update it directly from their command result).
    if (!sameSet(prevSelection, d.selection)) void get().refreshDetails();
  },

  toggleCollapsed: (guid) =>
    set((s) => ({
      collapsed: s.collapsed.includes(guid)
        ? s.collapsed.filter((c) => c !== guid)
        : [...s.collapsed, guid],
    })),

  refreshDetails: async () => {
    const token = ++detailsToken;
    try {
      const details = await sceneIpc.details();
      if (token !== detailsToken) return; // superseded by a newer selection
      set({ details });
    } catch (e) {
      // Kept as console-only: this fires on every selection change, so a
      // persistent failure would spam the status bar.
      if (token === detailsToken) console.error("scene.details failed", e);
    }
  },

  select: (guids, additive = false) => {
    // Optimistic local update; the delta reconciles authoritative state.
    set((s) => ({
      selection: additive
        ? s.selection.includes(guids[0] ?? "")
          ? s.selection.filter((g) => !guids.includes(g))
          : [...new Set([...s.selection, ...guids])]
        : guids,
    }));
    void sceneIpc.select(guids, additive);
  },

  createEntity: async (kind, parent = null) => sceneIpc.create(kind, parent),

  deleteSelected: () => {
    const sel = get().selection;
    if (sel.length) void sceneIpc.delete(sel);
  },

  rename: (guid, name) => void sceneIpc.rename(guid, name),
  reparent: (guid, parent) => void sceneIpc.reparent(guid, parent),

  toggleVisible: (guid) => {
    const node = get().nodes[guid];
    if (node) void sceneIpc.setVisible(guid, !node.visible);
  },

  setProperty: async (typePath, field, value) => {
    const sel = get().selection;
    if (!sel.length) return;
    const token = ++detailsToken;
    try {
      const details = await sceneIpc.setProperty(sel, typePath, field, value);
      if (token !== detailsToken) return; // selection moved on
      set({ details });
    } catch (e) {
      // Details-panel edit, not command-initiated → surface here (commands.ts
      // handles command-initiated failures, so no double-toast).
      console.error("scene.setProperty failed", e);
      useShellStore.getState().pushStatus(`Property edit failed: ${errText(e)}`);
    }
  },

  resetProperty: async (typePath, field) => {
    const sel = get().selection;
    if (!sel.length) return;
    const token = ++detailsToken;
    try {
      const details = await sceneIpc.resetProperty(sel, typePath, field);
      if (token !== detailsToken) return; // selection moved on
      set({ details });
    } catch (e) {
      console.error("scene.resetProperty failed", e);
      useShellStore.getState().pushStatus(`Property reset failed: ${errText(e)}`);
    }
  },

  undo: () => void sceneIpc.undo(),
  redo: () => void sceneIpc.redo(),
}));

registerBridgedStore("scene", useSceneStore);

/** A flattened Outliner row. */
export interface OutlinerRow {
  node: SceneNode;
  depth: number;
}

/** Depth-first flatten of the tree honoring collapse state. */
export function outlinerRows(
  nodes: Record<string, SceneNode>,
  roots: string[],
  collapsed: string[],
): OutlinerRow[] {
  const out: OutlinerRow[] = [];
  const walk = (guids: string[], depth: number) => {
    for (const guid of guids) {
      const node = nodes[guid];
      if (!node) continue;
      out.push({ node, depth });
      if (!collapsed.includes(guid)) walk(node.children, depth + 1);
    }
  };
  walk(roots, 0);
  return out;
}

/** An entity is dimmed in the Outliner when its effective visibility is off. */
export function isEffectivelyVisible(nodes: Record<string, SceneNode>, guid: string): boolean {
  return nodes[guid]?.effective_visible ?? true;
}

/** Attach scene handlers to the enumerated Edit/File menu commands (P3.4/P3.5). */
export function registerSceneCommands(): void {
  const s = () => useSceneStore.getState();
  const wire = (id: string, run: () => void | Promise<void>) => {
    if (getCommand(id)) setCommandHandler(id, run);
  };
  wire("edit.undo", () => s().undo());
  wire("edit.redo", () => s().redo());
  wire("edit.delete", () => s().deleteSelected());
  wire("file.saveLevel", () => sceneIpc.save());
  wire("file.saveAll", () => sceneIpc.save());
  wire("file.newLevel", async () => s().applySnapshot(await sceneIpc.newScene()));
  wire("file.openLevel", async () => s().applySnapshot(await sceneIpc.open()));
}

let unlisten: UnlistenFn | null = null;

/**
 * Load the initial snapshot and subscribe to `world://delta`. Idempotent
 * (React StrictMode double-mounts); returns a disposer.
 */
export async function initSceneSync(): Promise<() => void> {
  if (unlisten) return () => {};
  unlisten = await listenTo("world://delta", (delta) => useSceneStore.getState().applyDelta(delta));
  try {
    useSceneStore.getState().applySnapshot(await sceneIpc.snapshot());
  } catch (e) {
    console.error("scene.snapshot failed", e);
  }
  const dispose = unlisten;
  return () => {
    dispose?.();
    unlisten = null;
  };
}
