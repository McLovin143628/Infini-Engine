/**
 * Code editor tab model (P5.1). Open files become tabs; each tab's CodeMirror
 * `EditorState` lives OUTSIDE React in a module map so switching tabs preserves
 * undo history + cursor. The panel owns the single live `EditorView` and swaps
 * states. Files are read/written through the P5.4 `files` IPC domain; opens
 * arrive via the `infinity:open-file` event (Explorer/Search/Git).
 */
import { create } from "zustand";

import type { EditorState } from "@codemirror/state";
import type { EditorView, ViewUpdate } from "@codemirror/view";

import { createEditorState, registerSaveHandler } from "../lib/editor/setup";
import { files } from "../lib/ipc";
import { onOpenFile } from "../lib/openFile";

export interface EditorTab {
  id: string;
  path: string;
  name: string;
  dirty: boolean;
}

// Per-tab EditorState, kept off the React tree (immutable CM state references).
const tabStates = new Map<string, EditorState>();
export function tabStateFor(id: string): EditorState | undefined {
  return tabStates.get(id);
}
export function setTabState(id: string, state: EditorState): void {
  tabStates.set(id, state);
}

// The panel's single live EditorView, registered here so the LSP bridge (P5.2)
// can reconfigure the extra compartment + push diagnostics without prop-drilling.
let activeView: EditorView | null = null;
export function setActiveView(v: EditorView | null): void {
  activeView = v;
}
export function getActiveView(): EditorView | null {
  return activeView;
}

/** The active tab's absolute path, or null. */
export function activeTabPath(): string | null {
  const { activeId, tabs } = useEditorStore.getState();
  return tabs.find((t) => t.id === activeId)?.path ?? null;
}

let counter = 0;
function baseName(p: string): string {
  return p.split(/[\\/]/).pop() ?? p;
}

interface EditorStore {
  tabs: EditorTab[];
  activeId: string | null;
  openFile: (absPath: string) => Promise<void>;
  closeTab: (id: string) => void;
  activate: (id: string) => void;
  markDirty: (id: string, dirty: boolean) => void;
  saveActive: () => Promise<void>;
}

export const useEditorStore = create<EditorStore>((set, get) => ({
  tabs: [],
  activeId: null,

  openFile: async (absPath) => {
    const existing = get().tabs.find((t) => t.path === absPath);
    if (existing) {
      set({ activeId: existing.id });
      return;
    }
    let content: string;
    try {
      content = await files.read(absPath);
    } catch (e) {
      console.error("open file failed", absPath, e);
      return;
    }
    const id = `tab-${counter++}`;
    const onUpdate = (u: ViewUpdate) => {
      tabStates.set(id, u.state);
      // Only user edits (dispatched transactions) mark the tab dirty — a
      // programmatic setState on tab-switch has no transactions.
      if (u.docChanged && u.transactions.length > 0) get().markDirty(id, true);
    };
    tabStates.set(id, createEditorState(content, absPath, onUpdate));
    set((s) => ({
      tabs: [...s.tabs, { id, path: absPath, name: baseName(absPath), dirty: false }],
      activeId: id,
    }));
  },

  closeTab: (id) => {
    tabStates.delete(id);
    set((s) => {
      const tabs = s.tabs.filter((t) => t.id !== id);
      const activeId = s.activeId === id ? (tabs[tabs.length - 1]?.id ?? null) : s.activeId;
      return { tabs, activeId };
    });
  },

  activate: (id) => set({ activeId: id }),

  markDirty: (id, dirty) =>
    set((s) => ({ tabs: s.tabs.map((t) => (t.id === id ? { ...t, dirty } : t)) })),

  saveActive: async () => {
    const { activeId, tabs } = get();
    if (!activeId) return;
    const tab = tabs.find((t) => t.id === activeId);
    const state = tabStates.get(activeId);
    if (!tab || !state) return;
    try {
      await files.write(tab.path, state.doc.toString());
      get().markDirty(activeId, false);
    } catch (e) {
      console.error("save failed", tab.path, e);
    }
  },
}));

// Ctrl/Cmd+S in the editor keymap routes here (avoids an import cycle).
registerSaveHandler(() => void useEditorStore.getState().saveActive());

let unlisten: (() => void) | null = null;

/**
 * Subscribe to `infinity:open-file`. Idempotent; returns a disposer.
 *
 * **The panel is opened too** (Wave E). `requestOpenFile` added a tab and
 * stopped there, so a request made while the Code Editor was hidden — which is
 * the default layout, and every request from the Wave-E object routing — put the
 * file somewhere invisible and looked like nothing happened. `openPanel` is
 * create-or-focus, so an already-visible editor is merely activated.
 */
export function initEditorSync(): () => void {
  if (unlisten) return () => {};
  unlisten = onOpenFile((path) => {
    void import("../panels/dock/dockLayoutStore").then(({ useDockLayout }) =>
      useDockLayout.getState().openPanel("editor"),
    );
    void useEditorStore.getState().openFile(path);
  });
  const dispose = unlisten;
  return () => {
    dispose();
    unlisten = null;
  };
}
