/**
 * Blueprint editor store (Phase 6). Mirrors the backend graph document with an
 * optimistic local edit path (instant canvas feedback) plus authoritative
 * `graph_apply` for validation + undo. One active document in v1.
 */
import { create } from "zustand";

import { registerUndoScope } from "../lib/undoScopes";

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

/**
 * The localStorage key holding a graph's breakpoint node ids.
 *
 * Keyed on the document's **name**, not its id (a round-2 LOW). A `GraphDoc`
 * id is `bp:{counter}` where the counter is per PROCESS and restarts at 1 every
 * session, so `bp:debug:bp:1` named "whichever graph happened to be created
 * first this run" — last session's breakpoints reloaded onto a different graph,
 * silently, and pausing somewhere the author never asked for is worse than not
 * pausing at all. The name is what the author sees and what identifies the
 * document to them; it is the best identity a session-scoped document has,
 * since `graph_create` takes no asset key.
 */
function bpKey(graphName: string): string {
  return `bp:debug:name:${graphName}`;
}

/** Load a graph's persisted breakpoints (best-effort; empty on any failure). */
function loadBreakpoints(graphName: string): Set<number> {
  try {
    const raw = localStorage.getItem(bpKey(graphName));
    if (!raw) return new Set();
    const arr = JSON.parse(raw) as unknown;
    return new Set(Array.isArray(arr) ? arr.filter((n): n is number => typeof n === "number") : []);
  } catch {
    return new Set();
  }
}

/** Persist a graph's breakpoints (best-effort). */
function saveBreakpoints(graphName: string, bps: Set<number>): void {
  try {
    localStorage.setItem(bpKey(graphName), JSON.stringify([...bps]));
  } catch {
    /* storage unavailable — breakpoints stay in-memory only */
  }
}

// ── The tombstone (F-lens L7.M3) ─────────────────────────────────────────────
//
// A deliberate port of `dccStore`'s pattern, for a store that holds exactly ONE
// document. `dccStore.ts` carries the full reasoning; what it costs here is the
// same two module-level slots, and what it buys is two distinct defects:
//
//  1. **The double create.** `init` guarded on `get().ready`, which is set after
//     three awaits. React StrictMode's mount → cleanup → mount runs both mounts
//     before the first `graph_list` returns, so both saw `ready === false`, both
//     found no existing document, and both called `graph_create("Main")` —
//     minting two backend documents (each with its own journal) and adopting
//     one. The other lived for the process. Unlike `dcc_open`, `graph_create`
//     takes no asset key, so it cannot be made idempotent on the backend: a
//     second `init` must JOIN the first instead of racing it.
//  2. **The close that had nothing to name.** `close` reads `get().doc`, which
//     is null while the create is still in flight — so it sent no `graph_close`
//     at all, and the document the reply was about to hand over leaked.
//
// The rule, exactly as `dccStore` states it: a tombstoned document accepts no
// state, and its `init` reply is answered with a `close` — because the document
// it just created is exactly the one the first close could not name. And, also
// exactly as `dccStore.open` does it, a fresh `init` CLEARS the tombstone first:
// a re-open supersedes a close that has not landed, which is what makes the
// StrictMode remount adopt rather than tear down.
let opening: Promise<void> | null = null;
let openGen = 0;
let tombstoned = false;

/** Test-only: forget any in-flight init and clear the tombstone. */
export function __resetBlueprintInitForTest(): void {
  opening = null;
  openGen += 1;
  tombstoned = false;
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
    // Both synchronous, before any await — see the tombstone note above.
    tombstoned = false;
    if (opening) return opening;
    const gen = ++openGen;
    opening = (async () => {
      try {
        const registry = await graphIpc.registry();
        const registryById: Record<string, NodeDef> = {};
        for (const d of registry) registryById[d.typeId] = d;
        const existing = await graphIpc.list();
        const doc = existing[0] ?? (await graphIpc.create("Main"));
        if (tombstoned) {
          // Closed while this was in flight. The document exists NOW, and the
          // `close` that already ran had no id to name — so it is closed here.
          try {
            await graphIpc.close(doc.id);
          } catch (e) {
            console.error("graph.close failed", e);
          }
          return;
        }
        set({
          registry,
          registryById,
          doc,
          ready: true,
          // Restore this graph's persisted breakpoints.
          debugBreakpoints: loadBreakpoints(doc.name),
        });
      } catch (e) {
        console.error("blueprint.init failed", e);
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
      // **R2.F1.** The backend refused `rejected` of the edits — a `connect`
      // that would cycle, a node that is not there, an unknown param — and the
      // optimistic clone above is now a picture of a graph that does not exist.
      // `issues` cannot report it: a wire that never entered the graph has
      // nothing to be invalid about. So the canvas re-derives, which is what
      // "fully controlled" is supposed to mean, and the wire visibly snaps back.
      if (res.rejected > 0) {
        console.warn(`graph.apply: the backend refused ${res.rejected} edit(s); re-deriving`);
        const authoritative = await graphIpc.get(doc.id);
        if (!tombstoned && get().doc?.id === doc.id) set({ doc: authoritative });
      }
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

  // Undo/redo carry the same `try`/`catch` every other backend call in this
  // store has (F-lens L7.M4). They used to be the exception: a bare `await` in a
  // path whose callers are `void undo()` (the toolbar button) and
  // `void scope.undo()` (the Ctrl+Z routing), so a rejected `graph_undo` — a
  // closed document, a backend panic, a serialization mismatch — surfaced only
  // as an unhandled promise rejection with nothing shown to the author.
  undo: async () => {
    const doc = get().doc;
    if (!doc) return;
    try {
      const restored = await graphIpc.undo(doc.id);
      if (restored) set({ doc: restored, canRedo: true });
    } catch (e) {
      console.error("graph.undo failed", e);
    }
  },

  redo: async () => {
    const doc = get().doc;
    if (!doc) return;
    try {
      const restored = await graphIpc.redo(doc.id);
      if (restored) set({ doc: restored, canUndo: true });
    } catch (e) {
      console.error("graph.redo failed", e);
    }
  },

  clearOutput: () => set({ runResult: null, generated: null }),

  toggleBreakpoint: (id) => {
    const doc = get().doc;
    const next = new Set(get().debugBreakpoints);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    set({ debugBreakpoints: next });
    if (doc) saveBreakpoints(doc.name, next);
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

// Ctrl+Z inside the Blueprint panel undoes the GRAPH, not the scene (P23.2a).
registerUndoScope("blueprint", {
  undo: () => useBlueprintStore.getState().undo(),
  redo: () => useBlueprintStore.getState().redo(),
});
