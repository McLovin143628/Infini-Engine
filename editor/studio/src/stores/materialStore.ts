/**
 * Material editor store (Phase 7.2). Mirrors the blueprint store: an optimistic
 * local edit path for instant canvas feedback plus authoritative `material_apply`
 * for invariant-checking + undo. After every edit it recompiles the graph to
 * WGSL and re-renders the live preview sphere. One active document in v1 (the
 * per-`.inf_mat`-asset binding is a follow-up, matching the blueprint editor).
 */
import { create } from "zustand";

import { registerUndoScope } from "../lib/undoScopes";

import { material as materialIpc } from "../lib/ipc";
import type { BpDoc, BpEdit, BpIssue, NodeDef } from "../lib/blueprintTypes";
import type { MaterialCompileResult } from "../lib/materialTypes";
import { applyEditsLocal } from "../panels/blueprint/reducer";

interface MaterialState {
  registry: NodeDef[];
  registryById: Record<string, NodeDef>;
  doc: BpDoc | null;
  issues: BpIssue[];
  canUndo: boolean;
  canRedo: boolean;
  ready: boolean;
  compiling: boolean;
  compileResult: MaterialCompileResult | null;
  showWgsl: boolean;

  init: () => Promise<void>;
  close: () => Promise<void>;
  apply: (edits: BpEdit[], label: string) => Promise<void>;
  previewMove: (id: number, x: number, y: number) => void;
  nextId: () => number;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  compile: () => Promise<void>;
  bake: (size: number, name: string) => Promise<string | null>;
  toggleWgsl: () => void;
}

export const useMaterialStore = create<MaterialState>((set, get) => ({
  registry: [],
  registryById: {},
  doc: null,
  issues: [],
  canUndo: false,
  canRedo: false,
  ready: false,
  compiling: false,
  compileResult: null,
  showWgsl: false,

  init: async () => {
    if (get().ready) return;
    try {
      const registry = await materialIpc.registry();
      const registryById: Record<string, NodeDef> = {};
      for (const d of registry) registryById[d.typeId] = d;
      const existing = await materialIpc.list();
      const doc = existing[0] ?? (await materialIpc.create("Main"));
      set({ registry, registryById, doc, ready: true });
      void get().compile();
    } catch (e) {
      console.error("material.init failed", e);
    }
  },

  // Discard the editing surface: free the backend document (+ journal) and reset
  // to an un-inited state so a later re-open starts fresh instead of leaking the
  // old doc for the session. Called when the canvas panel unmounts (panel close).
  close: async () => {
    const doc = get().doc;
    set({
      doc: null,
      ready: false,
      issues: [],
      canUndo: false,
      canRedo: false,
      compileResult: null,
    });
    if (doc) {
      try {
        await materialIpc.close(doc.id);
      } catch (e) {
        console.error("material.close failed", e);
      }
    }
  },

  nextId: () => get().doc?.graph.nextId ?? 1,

  // Optimistic drag: reflect a node's position locally without a round-trip.
  previewMove: (id, x, y) => {
    const doc = get().doc;
    const node = doc?.graph.nodes[String(id)];
    if (!doc || !node) return;
    const nodes = { ...doc.graph.nodes, [String(id)]: { ...node, ui: { ...node.ui, x, y } } };
    set({ doc: { ...doc, graph: { ...doc.graph, nodes } } });
  },

  apply: async (edits, label) => {
    const doc = get().doc;
    if (!doc || edits.length === 0) return;
    const graph = structuredClone(doc.graph);
    applyEditsLocal(graph, edits);
    set({ doc: { ...doc, graph } });
    try {
      const res = await materialIpc.apply(doc.id, edits, label);
      set({ issues: res.issues, canUndo: res.canUndo, canRedo: res.canRedo });
      // Position-only moves don't change the shader; skip the recompile.
      if (edits.some((e) => e.kind !== "move-node")) void get().compile();
    } catch (e) {
      console.error("material.apply failed", e);
    }
  },

  compile: async () => {
    const doc = get().doc;
    if (!doc) return;
    set({ compiling: true });
    try {
      const compileResult = await materialIpc.compile(doc.id);
      set({ compileResult });
    } catch (e) {
      console.error("material.compile failed", e);
    } finally {
      set({ compiling: false });
    }
  },

  undo: async () => {
    const doc = get().doc;
    if (!doc) return;
    const restored = await materialIpc.undo(doc.id);
    if (restored) {
      set({ doc: restored, canRedo: true });
      void get().compile();
    }
  },

  redo: async () => {
    const doc = get().doc;
    if (!doc) return;
    const restored = await materialIpc.redo(doc.id);
    if (restored) {
      set({ doc: restored, canUndo: true });
      void get().compile();
    }
  },

  bake: async (size, name) => {
    const doc = get().doc;
    if (!doc) return null;
    try {
      return await materialIpc.bake(doc.id, size, name);
    } catch (e) {
      console.error("material.bake failed", e);
      return null;
    }
  },

  toggleWgsl: () => set((s) => ({ showWgsl: !s.showWgsl })),
}));

// **Ctrl+Z inside the Material panel undoes the MATERIAL** (P23.2a).
//
// Before this it undid the scene: `edit.undo` was wired straight to the scene
// store, so an author editing a graph moved an actor they could not see while
// the graph in front of them did nothing. The journal these call has existed
// since P7.2 — it simply had no way to be reached from the keyboard.
registerUndoScope("material", {
  undo: () => useMaterialStore.getState().undo(),
  redo: () => useMaterialStore.getState().redo(),
});
