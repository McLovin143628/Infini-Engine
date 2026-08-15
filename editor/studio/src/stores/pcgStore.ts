/**
 * PCG editor store (Phase 10.5b). Mirrors the material store: an optimistic local
 * edit path for instant canvas feedback plus authoritative `pcg_apply` for
 * invariant-checking + undo. After every structural edit it recompiles the graph
 * (lower + node-anchored diagnostics). One active document in v1 (the per-`.inf_pcg`
 * binding is a follow-up, matching the blueprint/material editors). The `evaluate`
 * action scatters the graph over the scene terrain into the selected volume.
 */
import { create } from "zustand";

import { registerUndoScope } from "../lib/undoScopes";

import { pcg as pcgIpc } from "../lib/ipc";
import type { BpDoc, BpEdit, BpIssue, NodeDef } from "../lib/blueprintTypes";
import type { PcgBiomeResult, PcgCompileResult, PcgEvaluateResult } from "../lib/pcgTypes";
import { applyEditsLocal } from "../panels/blueprint/reducer";

// ── The tombstone (F-lens L7.M3) ─────────────────────────────────────────────
//
// Ported from `dccStore` by way of `blueprintStore`, which carries the long
// version. Short form: `init` guarded on `ready`, which is set three awaits in,
// so React StrictMode's mount → cleanup → mount ran both mounts before
// `pcg_list` returned, both found no document, and both called
// `pcg_create("Main")` — two backend documents, two journals, one adopted and
// one leaked for the process. And `close` reads `get().doc`, which is null while
// the create is in flight, so it sent no `pcg_close` at all.
//
// A tombstoned document accepts no state and its `init` reply is answered with a
// `close`; a fresh `init` clears the tombstone first, so a StrictMode remount
// adopts rather than tears down.
let opening: Promise<void> | null = null;
let openGen = 0;
let tombstoned = false;

/** Test-only: forget any in-flight init and clear the tombstone. */
export function __resetPcgInitForTest(): void {
  opening = null;
  openGen += 1;
  tombstoned = false;
}

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
    // Both synchronous, before any await — see the tombstone note above.
    tombstoned = false;
    if (opening) return opening;
    const gen = ++openGen;
    opening = (async () => {
      try {
        const registry = await pcgIpc.registry();
        const registryById: Record<string, NodeDef> = {};
        for (const d of registry) registryById[d.typeId] = d;
        const existing = await pcgIpc.list();
        const doc = existing[0] ?? (await pcgIpc.create("Main"));
        if (tombstoned) {
          // Closed while this was in flight. The document exists NOW, and the
          // `close` that already ran had no id to name — so it is closed here.
          try {
            await pcgIpc.close(doc.id);
          } catch (e) {
            console.error("pcg.close failed", e);
          }
          return;
        }
        set({ registry, registryById, doc, ready: true });
        void get().compile();
      } catch (e) {
        console.error("pcg.init failed", e);
      } finally {
        // Only the LATEST init clears the memo — an older one resolving late
        // must not free a newer one's slot.
        if (openGen === gen) opening = null;
      }
    })();
    return opening;
  },

  // Discard the editing surface: free the backend document (+ journal) and reset
  // to an un-inited state so a later re-open starts fresh instead of leaking the
  // old doc for the session. Called when the canvas panel unmounts (panel close).
  //
  // `opening` is deliberately left alone: an init still in flight has to keep
  // its memo so a StrictMode remount JOINS it rather than minting a second
  // document, and it clears the memo itself in its own `finally`.
  close: async () => {
    tombstoned = true;
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

  // Undo/redo carry the same `try`/`catch` every other backend call in this
  // store has (F-lens L7.M4) — their callers are `void undo()` (the toolbar) and
  // `void scope.undo()` (Ctrl+Z), so a rejection was an unhandled promise
  // rejection with nothing shown to the author.
  undo: async () => {
    const doc = get().doc;
    if (!doc) return;
    try {
      const restored = await pcgIpc.undo(doc.id);
      if (restored) {
        set({ doc: restored, canRedo: true });
        void get().compile();
      }
    } catch (e) {
      console.error("pcg.undo failed", e);
    }
  },

  redo: async () => {
    const doc = get().doc;
    if (!doc) return;
    try {
      const restored = await pcgIpc.redo(doc.id);
      if (restored) {
        set({ doc: restored, canUndo: true });
        void get().compile();
      }
    } catch (e) {
      console.error("pcg.redo failed", e);
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

// Ctrl+Z inside the PCG panel undoes the PCG graph, not the scene (P23.2a).
registerUndoScope("pcg", {
  undo: () => usePcgStore.getState().undo(),
  redo: () => usePcgStore.getState().redo(),
});
