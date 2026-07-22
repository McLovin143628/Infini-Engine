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

// ── B-P4 debugger: breakpoints persist per graph in localStorage. ────────────

/** The localStorage key holding a graph's breakpoint node ids. */
function bpKey(graphId: string): string {
  return `bp:debug:${graphId}`;
}

/** Load a graph's persisted breakpoints (best-effort; empty on any failure). */
function loadBreakpoints(graphId: string): Set<number> {
  try {
    const raw = localStorage.getItem(bpKey(graphId));
    if (!raw) return new Set();
    const arr = JSON.parse(raw) as unknown;
    return new Set(Array.isArray(arr) ? arr.filter((n): n is number => typeof n === "number") : []);
  } catch {
    return new Set();
  }
}

/** Persist a graph's breakpoints (best-effort). */
function saveBreakpoints(graphId: string, bps: Set<number>): void {
  try {
    localStorage.setItem(bpKey(graphId), JSON.stringify([...bps]));
  } catch {
    /* storage unavailable — breakpoints stay in-memory only */
  }
}

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

  // ── B-P4 debugger slice ──
  /** Breakpoint node ids for the active graph (persisted per graph). */
  debugBreakpoints: Set<number>;
  /** Node ids a breakpoint paused on in the last debug run (pulse highlight). */
  debugHits: Set<number>;
  /** Latest captured wire values: nodeId → port → stringified value. */
  debugWireValues: Record<number, Record<string, string>>;

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

  /** Toggle a node's breakpoint (Alt-click on the node header) + persist. */
  toggleBreakpoint: (id: number) => void;
  /** Run the graph under debug lowering with the current breakpoints. */
  debugRun: () => Promise<void>;
  /** Clear captured wire values + hit highlights. */
  clearDebugValues: () => void;
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
  debugBreakpoints: new Set(),
  debugHits: new Set(),
  debugWireValues: {},

  init: async () => {
    if (get().ready) return;
    try {
      const registry = await graphIpc.registry();
      const registryById: Record<string, NodeDef> = {};
      for (const d of registry) registryById[d.typeId] = d;
      const existing = await graphIpc.list();
      const doc = existing[0] ?? (await graphIpc.create("Main"));
      set({
        registry,
        registryById,
        doc,
        ready: true,
        // Restore this graph's persisted breakpoints.
        debugBreakpoints: loadBreakpoints(doc.id),
      });
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
      debugHits: new Set(),
      debugWireValues: {},
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

  toggleBreakpoint: (id) => {
    const doc = get().doc;
    const next = new Set(get().debugBreakpoints);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    set({ debugBreakpoints: next });
    if (doc) saveBreakpoints(doc.id, next);
  },

  debugRun: async () => {
    const doc = get().doc;
    if (!doc) return;
    set({ running: true });
    try {
      const res = await graphIpc.debugRun(doc.id, [...get().debugBreakpoints], true);
      // Fold captured wires into nodeId → port → value.
      const wireValues: Record<number, Record<string, string>> = {};
      for (const w of res.wires) {
        (wireValues[w.node] ??= {})[w.port] = w.value;
      }
      set({
        debugHits: new Set(res.hits),
        debugWireValues: wireValues,
        // Reuse the run-output drawer for logs / vars / error.
        runResult: {
          logs: res.logs,
          vars: res.vars,
          handlers: res.handlers,
          error: res.error ?? null,
        },
      });
    } catch (e) {
      console.error("graph.debugRun failed", e);
      set({ runResult: { logs: [], vars: {}, handlers: [], error: String(e) } });
    } finally {
      set({ running: false });
    }
  },

  clearDebugValues: () => set({ debugHits: new Set(), debugWireValues: {} }),
}));
