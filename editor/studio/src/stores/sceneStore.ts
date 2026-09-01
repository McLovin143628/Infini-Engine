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
import type { LevelSettingsDto } from "../bindings/LevelSettingsDto";
import type { PropValueDto } from "../bindings/PropValueDto";
import type { SaveResultDto } from "../bindings/SaveResultDto";
import type { SceneDelta } from "../bindings/SceneDelta";
import type { SceneNode } from "../bindings/SceneNode";
import type { SceneSnapshot } from "../bindings/SceneSnapshot";
import type { SpawnKind } from "../bindings/SpawnKind";
import { getCommand, setCommandHandler } from "../lib/commands";
import { listenTo, refCountedInit } from "../lib/events";
import { scene as sceneIpc } from "../lib/ipc";
import { dispatchRedo, dispatchUndo } from "../lib/undoScopes";
import { registerBridgedStore } from "../panels/window/storeBridge";
import { useProjectStore } from "./projectStore";
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

/**
 * Debounce + in-flight guard for World Settings writes (R-P4). The panel edits
 * fire rapidly (typing / dragging); we coalesce them into one `scene.setSettings`
 * ~300 ms after the last change. While a local edit is pending, `world://delta`
 * refetches are suppressed so an echoing delta can't clobber the value the user
 * is still editing.
 */
let settingsWriteTimer: ReturnType<typeof setTimeout> | null = null;
let settingsWritePending = false;
const SETTINGS_WRITE_DEBOUNCE_MS = 300;

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
  /** The level's file-level settings (gravity / sim rate / render block), or null
   * until the first `loadLevelSettings` resolves (World Settings panel, R-P4). */
  levelSettings: LevelSettingsDto | null;
  /** True once the first snapshot has loaded. */
  ready: boolean;

  applySnapshot: (s: SceneSnapshot) => void;
  applyDelta: (d: SceneDelta) => void;
  toggleCollapsed: (guid: string) => void;
  refreshDetails: () => Promise<void>;
  /** Fetch the level settings from the backend (skipped while a local edit is pending). */
  loadLevelSettings: () => Promise<void>;
  /** Optimistically set the level settings + debounce a `scene.setSettings` write. */
  setLevelSettings: (next: LevelSettingsDto) => void;

  select: (guids: string[], additive?: boolean) => void;
  createEntity: (kind: SpawnKind, parent?: string | null) => Promise<string>;
  deleteSelected: () => void;
  duplicateSelected: () => void;
  copySelected: () => void;
  cutSelected: () => void;
  paste: () => void;
  saveLevelAs: () => Promise<void>;
  /**
   * File > Open Level... -- a real file dialog (wave GTA1). Before it, the menu
   * row called `scene_open` with no path, and no path means the QUICKSAVE
   * fallback: choosing "Open Level" silently replaced the document with
   * `quicksave.inf_lvl`, whatever that happened to be.
   */
  openLevelViaDialog: () => Promise<void>;
  /**
   * Open a level by path, replacing the document. Resolves to true when the
   * level opened.
   */
  openLevel: (path: string) => Promise<boolean>;
  rename: (guid: string, name: string) => void;
  reparent: (guid: string, parent: string | null) => void;
  toggleVisible: (guid: string) => void;
  setProperty: (typePath: string, field: string, value: PropValueDto) => Promise<void>;
  resetProperty: (typePath: string, field: string) => Promise<void>;
  /** Add a Default component to the current selection (E-P1). */
  addComponent: (typePath: string) => Promise<void>;
  /** Remove a component from a single entity (E-P1). */
  removeComponent: (guid: string, typePath: string) => Promise<void>;
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
  levelSettings: null,
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
    void get().loadLevelSettings();
  },

  applyDelta: (d) => {
    const prevSelection = get().selection;
    const prevVersion = get().version;
    set((state) => {
      const nodes = { ...state.nodes };
      for (const guid of d.removed) delete nodes[guid];
      for (const n of d.added) nodes[n.guid] = n;
      for (const n of d.updated) nodes[n.guid] = n;
      return {
        nodes,
        // IB-13: `roots` is omitted when it did not move. A root list can only
        // change on a create, a delete or a reparent, and re-shipping 100 000
        // strings on every drag frame measured 3.496 ms of an otherwise 0.03 ms
        // delta. The reducer stays trivially correct.
        roots: d.roots ?? state.roots,
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
    // Re-fetch level settings on any real document change (undo/redo of a
    // settings edit, a fresh load) — cheap, and guarded against clobbering an
    // in-flight local edit inside `loadLevelSettings`.
    if (Number(d.version) !== prevVersion) void get().loadLevelSettings();
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

  loadLevelSettings: async () => {
    // Don't clobber a value the user is actively editing (a debounced write is
    // queued/in-flight); the write's own delta will refetch once settled.
    if (settingsWritePending) return;
    try {
      const s = await sceneIpc.getSettings();
      if (settingsWritePending) return; // an edit started during the await
      set({ levelSettings: s });
    } catch (e) {
      console.error("scene.getSettings failed", e);
    }
  },

  setLevelSettings: (next) => {
    set({ levelSettings: next }); // optimistic — the panel reflects it instantly
    settingsWritePending = true;
    if (settingsWriteTimer) clearTimeout(settingsWriteTimer);
    settingsWriteTimer = setTimeout(() => {
      settingsWriteTimer = null;
      void sceneIpc
        .setSettings(next)
        .catch((e) =>
          useShellStore.getState().pushStatus(`World settings edit failed: ${errText(e)}`),
        )
        .finally(() => {
          settingsWritePending = false;
        });
    }, SETTINGS_WRITE_DEBOUNCE_MS);
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

  duplicateSelected: () => {
    const sel = get().selection;
    if (!sel.length) return;
    void sceneIpc.duplicate(sel).catch((e) =>
      useShellStore.getState().pushStatus(`Duplicate failed: ${errText(e)}`),
    );
  },

  copySelected: () => {
    const sel = get().selection;
    if (!sel.length) return;
    void sceneIpc
      .copy(sel)
      .then((n) => {
        if (n > 0) useShellStore.getState().pushStatus(`Copied ${n} object${n > 1 ? "s" : ""}.`);
      })
      .catch((e) => useShellStore.getState().pushStatus(`Copy failed: ${errText(e)}`));
  },

  cutSelected: () => {
    const sel = get().selection;
    if (!sel.length) return;
    void sceneIpc
      .cut(sel)
      .then((n) => {
        if (n > 0) useShellStore.getState().pushStatus(`Cut ${n} object${n > 1 ? "s" : ""}.`);
      })
      .catch((e) => useShellStore.getState().pushStatus(`Cut failed: ${errText(e)}`));
  },

  paste: () => {
    void sceneIpc
      .paste()
      .then((roots) =>
        useShellStore
          .getState()
          .pushStatus(
            roots.length ? `Pasted ${roots.length} object${roots.length > 1 ? "s" : ""}.` : "Clipboard is empty.",
          ),
      )
      .catch((e) => useShellStore.getState().pushStatus(`Paste failed: ${errText(e)}`));
  },

  saveLevelAs: async () => {
    const { save } = await import("@tauri-apps/plugin-dialog");
    // Seed the dialog from the current level path, else the project's Levels dir.
    let defaultPath: string | undefined;
    try {
      defaultPath = (await sceneIpc.currentPath()) ?? undefined;
    } catch {
      // No current path — fall through to the project dir.
    }
    if (!defaultPath) {
      const proj = useProjectStore.getState().current;
      // IB-7: `levels_dir` is relative to the CONTENT root, not the project
      // root. Seeding the dialog at `<root>/Levels` pointed authors at a
      // directory the cook never opens.
      if (proj) defaultPath = `${proj.root}/${proj.content_dir}/${proj.levels_dir}`;
    }
    const path = await save({
      title: "Save Current Level As",
      defaultPath,
      filters: [{ name: "Infini Level", extensions: ["inf_lvl"] }],
    });
    if (!path) return; // cancelled
    try {
      reportSaveResult(await sceneIpc.save(path), "Level");
    } catch (e) {
      useShellStore.getState().pushStatus(`Save As failed: ${errText(e)}`);
    }
  },

  openLevelViaDialog: async () => {
    const { open } = await import("@tauri-apps/plugin-dialog");
    // Seeded exactly like Save As: the current level's own directory, else the
    // project's Levels dir (which is under the CONTENT root -- the IB-7 ruling).
    let defaultPath: string | undefined;
    try {
      defaultPath = (await sceneIpc.currentPath()) ?? undefined;
    } catch {
      // No current path -- fall through to the project dir.
    }
    if (!defaultPath) {
      const proj = useProjectStore.getState().current;
      if (proj) defaultPath = `${proj.root}/${proj.content_dir}/${proj.levels_dir}`;
    }
    const picked = await open({
      title: "Open Level",
      defaultPath,
      multiple: false,
      directory: false,
      filters: [{ name: "Infini Level", extensions: ["inf_lvl"] }],
    });
    const path = typeof picked === "string" ? picked : null;
    if (!path) return; // cancelled
    await get().openLevel(path);
  },

  openLevel: async (path) => {
    try {
      get().applySnapshot(await sceneIpc.open(path));
      return true;
    } catch (e) {
      useShellStore.getState().pushStatus(`Open Level failed: ${errText(e)}`, 12000);
      return false;
    }
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

  addComponent: async (typePath) => {
    const sel = get().selection;
    if (!sel.length) return;
    const token = ++detailsToken;
    try {
      const details = await sceneIpc.addComponent(typePath, sel);
      if (token !== detailsToken) return;
      set({ details });
    } catch (e) {
      console.error("scene.addComponent failed", e);
      useShellStore.getState().pushStatus(`Add component failed: ${errText(e)}`);
    }
  },

  removeComponent: async (guid, typePath) => {
    const token = ++detailsToken;
    try {
      const details = await sceneIpc.removeComponent(guid, typePath);
      if (token !== detailsToken) return;
      set({ details });
    } catch (e) {
      console.error("scene.removeComponent failed", e);
      useShellStore.getState().pushStatus(`Remove component failed: ${errText(e)}`);
    }
  },

  // `void`-ing a promise drops its REJECTION too (F-lens L7.M4). `scene_undo`
  // rejects whenever the backend refuses — an empty history is the common one —
  // and Ctrl+Z reaches here through `void scope.undo()`, so the failure used to
  // land as an unhandled promise rejection with nothing in the status bar. Every
  // other command in this store already reports through `pushStatus`.
  undo: () =>
    void sceneIpc.undo().catch((e) => {
      console.error("scene.undo failed", e);
      useShellStore.getState().pushStatus(`Undo failed: ${errText(e)}`);
    }),
  redo: () =>
    void sceneIpc.redo().catch((e) => {
      console.error("scene.redo failed", e);
      useShellStore.getState().pushStatus(`Redo failed: ${errText(e)}`);
    }),
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

/**
 * Report what a save actually accomplished (P16.4b).
 *
 * A level save is no longer all-or-nothing: the `.inf_lvl` and the
 * `.inf_terrain` assets its streamed terrains reference are separate files, and
 * a terrain write can fail while the level lands. Those edits are NOT lost —
 * they stay dirty and the next save retries them — but the user has to be told,
 * or a failed terrain write reads as a clean save.
 */
export function reportSaveResult(res: SaveResultDto, what = "Level"): void {
  const status = useShellStore.getState().pushStatus;
  // Failures first, and the voxel ones beside the terrain ones (P21.2): both mean
  // "the level landed, this asset did not", and an author who carved a cave and a
  // hillside in one session has to hear about both in one message rather than
  // about whichever the code checked first.
  const failures = [...res.terrain_failures, ...res.voxel_warnings];
  if (failures.length > 0) {
    status(
      `${what} saved, but terrain/carve edits were NOT fully written: ${failures.join("; ")}. ` +
        "Fix the cause and save again.",
      10000,
    );
    return;
  }
  if (res.terrain_assets_written > 0 || res.voxel_assets_written > 0) {
    const parts: string[] = [];
    if (res.terrain_assets_written > 0) {
      parts.push(
        `${res.terrain_tiles_written} terrain tile(s) to ${res.terrain_assets_written} asset(s)`,
      );
    }
    if (res.voxel_assets_written > 0) {
      parts.push(
        `${res.voxel_chunks_written} voxel chunk(s) to ${res.voxel_assets_written} asset(s)`,
      );
    }
    status(`${what} saved (${parts.join(", ")}).`);
    return;
  }
  status(`${what} saved.`);
}

/** Attach scene handlers to the enumerated Edit/File menu commands (P3.4/P3.5). */
export function registerSceneCommands(): void {
  const s = () => useSceneStore.getState();
  const wire = (id: string, run: () => void | Promise<void>) => {
    if (getCommand(id)) setCommandHandler(id, run);
  };
  // **Undo routes to the focused panel first, the scene otherwise** (P23.2a).
  //
  // The scene is the DEFAULT, not the only answer: with no panel focused, or a
  // focused panel that claimed no scope, `dispatchUndo` returns false and this
  // runs, which is exactly what Ctrl+Z did before. What changed is that a graph
  // editor with its own journal now gets its own Ctrl+Z instead of quietly
  // moving an actor.
  //
  // The level-editing surface — outliner, details, and **the viewport** — is on
  // the default path because none of them registers a scope. For the viewport
  // that is only half the story and the other half was a bug (P23.2a audit —
  // B1): claiming no scope is useless if focus can never REACH the viewport,
  // and it could not, because the native child swallows DOM pointer events.
  // `lib/viewportFocus.ts` is what puts it back on this path.
  wire("edit.undo", () => {
    if (!dispatchUndo()) s().undo();
  });
  wire("edit.redo", () => {
    if (!dispatchRedo()) s().redo();
  });
  wire("edit.delete", () => s().deleteSelected());
  wire("edit.duplicate", () => s().duplicateSelected());
  wire("actor.duplicate", () => s().duplicateSelected());
  wire("edit.copy", () => s().copySelected());
  wire("edit.cut", () => s().cutSelected());
  wire("edit.paste", () => s().paste());
  wire("file.saveLevel", async () => reportSaveResult(await sceneIpc.save()));
  wire("file.saveLevelAs", () => s().saveLevelAs());
  wire("file.saveAll", async () => reportSaveResult(await sceneIpc.save(), "All"));
  wire("file.newLevel", async () => s().applySnapshot(await sceneIpc.newScene()));
  wire("file.openLevel", () => s().openLevelViaDialog());
}

/**
 * Load the initial snapshot and subscribe to `world://delta`. Returns a
 * disposer.
 *
 * **Refcounted** (`refCountedInit`, F-lens L7.M1): the guard used to be set
 * after the first `await`, so StrictMode's double mount subscribed twice — and
 * `world://delta` is the hottest channel in the editor, so every scene change
 * was reduced into the store twice over.
 */
export const initSceneSync = refCountedInit(async (sink) => {
  const unlisten = await listenTo("world://delta", (delta) =>
    useSceneStore.getState().applyDelta(delta),
  );
  sink(unlisten);
  try {
    useSceneStore.getState().applySnapshot(await sceneIpc.snapshot());
  } catch (e) {
    console.error("scene.snapshot failed", e);
  }
  return () => unlisten();
});
