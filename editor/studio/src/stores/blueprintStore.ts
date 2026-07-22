/**
 * Blueprint editor store (Phase 6). Mirrors the backend graph document with an
 * optimistic local edit path (instant canvas feedback) plus authoritative
 * `graph_apply` for validation + undo. One active document in v1.
 */
import { create } from "zustand";

import { graph as graphIpc } from "../lib/ipc";
import type {
  BpDoc,
  BpEdit,
  BpIssue,
  GraphRunResult,
  NodeDef,
} from "../lib/blueprintTypes";
import { applyEditsLocal } from "../panels/blueprint/reducer";

interface BlueprintState {
  registry: NodeDef[];
  registryById: Record<string, NodeDef>;
  doc: BpDoc | null;
  issues: BpIssue[];
  canUndo: boolean;
  canRedo: boolean;
  running: boolean;
  runResult: GraphRunResult | null;
  generated: string | null;
  ready: boolean;

  init: () => Promise<void>;
  close: () => Promise<void>;
  apply: (edits: BpEdit[], label: string) => Promise<void>;
  previewMove: (id: number, x: number, y: number) => void;
  nextId: () => number;
  run: () => Promise<void>;
  generate: () => Promise<void>;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  clearOutput: () => void;
}

export const useBlueprintStore = create<BlueprintState>((set, get) => ({
  registry: [],
  registryById: {},
  doc: null,
  issues: [],
  canUndo: false,
  canRedo: false,
  running: false,
  runResult: null,
  generated: null,
  ready: false,

  init: async () => {
    if (get().ready) return;
    try {
      const registry = await graphIpc.registry();
      const registryById: Record<string, NodeDef> = {};
      for (const d of registry) registryById[d.typeId] = d;
      const existing = await graphIpc.list();
      const doc = existing[0] ?? (await graphIpc.create("Main"));
      set({ registry, registryById, doc, ready: true });
    } catch (e) {
      console.error("blueprint.init failed", e);
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
      runResult: null,
      generated: null,
    });
    if (doc) {
      try {
        await graphIpc.close(doc.id);
      } catch (e) {
        console.error("graph.close failed", e);
      }
    }
  },

  nextId: () => get().doc?.graph.nextId ?? 1,

  // Optimistic drag: reflect a node's position locally without a backend
  // round-trip or an undo entry (the final move is committed on drag end).
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
      const res = await graphIpc.apply(doc.id, edits, label);
      set({ issues: res.issues, canUndo: res.canUndo, canRedo: res.canRedo });
    } catch (e) {
      console.error("graph.apply failed", e);
    }
  },

  run: async () => {
    const doc = get().doc;
    if (!doc) return;
    set({ running: true });
    try {
      const runResult = await graphIpc.run(doc.id);
      set({ runResult });
    } catch (e) {
      console.error("graph.run failed", e);
      set({ runResult: { logs: [], vars: {}, handlers: [], error: String(e) } });
    } finally {
      set({ running: false });
    }
  },

  generate: async () => {
    const doc = get().doc;
    if (!doc) return;
    try {
      const generated = await graphIpc.generate(doc.id);
      set({ generated });
    } catch (e) {
      set({ generated: `// generation failed: ${String(e)}` });
    }
  },

  undo: async () => {
    const doc = get().doc;
    if (!doc) return;
    const restored = await graphIpc.undo(doc.id);
    if (restored) set({ doc: restored, canRedo: true });
  },

  redo: async () => {
    const doc = get().doc;
    if (!doc) return;
    const restored = await graphIpc.redo(doc.id);
    if (restored) set({ doc: restored, canUndo: true });
  },

  clearOutput: () => set({ runResult: null, generated: null }),
}));
