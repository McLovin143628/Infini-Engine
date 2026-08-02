/**
 * PCG editor store (Phase 10.5b). Mirrors the material store: an optimistic local
 * edit path for instant canvas feedback plus authoritative `pcg_apply` for
 * invariant-checking + undo. After every structural edit it recompiles the graph
 * (lower + node-anchored diagnostics). One active document in v1 (the per-`.inf_pcg`
 * binding is a follow-up, matching the blueprint/material editors). The `evaluate`
 * action scatters the graph over the scene terrain into the selected volume.
 */
import { create } from "zustand";

import { pcg as pcgIpc } from "../lib/ipc";
import type { BpDoc, BpEdit, BpIssue, NodeDef } from "../lib/blueprintTypes";
import type { PcgBiomeResult, PcgCompileResult, PcgEvaluateResult } from "../lib/pcgTypes";
import { applyEditsLocal } from "../panels/blueprint/reducer";

interface PcgState {
  registry: NodeDef[];
  registryById: Record<string, NodeDef>;
  doc: BpDoc | null;
  issues: BpIssue[];
  canUndo: boolean;
  canRedo: boolean;
  ready: boolean;
  compiling: boolean;
  compileResult: PcgCompileResult | null;
  evaluating: boolean;
  lastEval: PcgEvaluateResult | null;
  /** The last biome→PCG binding run (P19.3) — the terrain-level sibling of
   *  `lastEval`, kept beside it because the two populate different caches and
   *  neither supersedes the other. */
  lastBiomeEval: PcgBiomeResult | null;

  init: () => Promise<void>;
  close: () => Promise<void>;
  apply: (edits: BpEdit[], label: string) => Promise<void>;
  previewMove: (id: number, x: number, y: number) => void;
  nextId: () => number;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
  compile: () => Promise<void>;
  evaluate: () => Promise<void>;
  evaluateBiomes: () => Promise<void>;
  save: (name: string) => Promise<string | null>;
}

export const usePcgStore = create<PcgState>((set, get) => ({
  registry: [],
  registryById: {},
  doc: null,
  issues: [],
  canUndo: false,
  canRedo: false,
  ready: false,
  compiling: false,
  compileResult: null,
  evaluating: false,
  lastEval: null,
  lastBiomeEval: null,

  init: async () => {
    if (get().ready) return;
    try {
      const registry = await pcgIpc.registry();
      const registryById: Record<string, NodeDef> = {};
      for (const d of registry) registryById[d.typeId] = d;
      const existing = await pcgIpc.list();
      const doc = existing[0] ?? (await pcgIpc.create("Main"));
      set({ registry, registryById, doc, ready: true });
      void get().compile();
    } catch (e) {
      console.error("pcg.init failed", e);
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
      lastEval: null,
      lastBiomeEval: null,
    });
    if (doc) {
      try {
        await pcgIpc.close(doc.id);
      } catch (e) {
        console.error("pcg.close failed", e);
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
      const res = await pcgIpc.apply(doc.id, edits, label);
      set({ issues: res.issues, canUndo: res.canUndo, canRedo: res.canRedo });
      // Position-only moves don't change the lowered document; skip the recompile.
      if (edits.some((e) => e.kind !== "move-node")) void get().compile();
    } catch (e) {
      console.error("pcg.apply failed", e);
    }
  },

  compile: async () => {
    const doc = get().doc;
    if (!doc) return;
    set({ compiling: true });
    try {
      const compileResult = await pcgIpc.compile(doc.id);
      set({ compileResult });
    } catch (e) {
      console.error("pcg.compile failed", e);
    } finally {
      set({ compiling: false });
    }
  },

  evaluate: async () => {
    const doc = get().doc;
    if (!doc) return;
    set({ evaluating: true });
    try {
      const lastEval = await pcgIpc.evaluate(doc.id, null);
      set({ lastEval });
    } catch (e) {
      console.error("pcg.evaluate failed", e);
      set({ lastEval: null });
    } finally {
      set({ evaluating: false });
    }
  },

  // The terrain-level sibling of `evaluate`: no document id, because the graphs
  // come from the terrain's biome set rather than the open canvas. Shares the
  // `evaluating` flag so the two buttons can't be fired at once.
  evaluateBiomes: async () => {
    set({ evaluating: true });
    try {
      const lastBiomeEval = await pcgIpc.evaluateBiomes(null);
      set({ lastBiomeEval });
    } catch (e) {
      console.error("pcg.evaluateBiomes failed", e);
      set({ lastBiomeEval: null });
    } finally {
      set({ evaluating: false });
    }
  },

  undo: async () => {
    const doc = get().doc;
    if (!doc) return;
    const restored = await pcgIpc.undo(doc.id);
    if (restored) {
      set({ doc: restored, canRedo: true });
      void get().compile();
    }
  },

  redo: async () => {
    const doc = get().doc;
    if (!doc) return;
    const restored = await pcgIpc.redo(doc.id);
    if (restored) {
      set({ doc: restored, canUndo: true });
      void get().compile();
    }
  },

  save: async (name) => {
    const doc = get().doc;
    if (!doc) return null;
    try {
      return await pcgIpc.save(doc.id, name);
    } catch (e) {
      console.error("pcg.save failed", e);
      return null;
    }
  },
}));
