/**
 * Material editor store (Phase 7.2). Mirrors the blueprint store: an optimistic
 * local edit path for instant canvas feedback plus authoritative `material_apply`
 * for invariant-checking + undo. After every edit it recompiles the graph to
 * WGSL and re-renders the live preview sphere. One active document in v1 (the
 * per-`.inf_mat`-asset binding is a follow-up, matching the blueprint editor).
 */
import { create } from "zustand";

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
  apply: (edits: BpEdit[], label: string) => Promise<void>;
  previewMove: (id: number, x: number, y: number) => void;
  nextId: () => number;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  compile: () => Promise<void>;
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

  toggleWgsl: () => set((s) => ({ showWgsl: !s.showWgsl })),
}));
