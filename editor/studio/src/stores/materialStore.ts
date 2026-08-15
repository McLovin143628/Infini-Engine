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

// ── The tombstone (F-lens L7.M3) ─────────────────────────────────────────────
//
// Ported from `dccStore` by way of `blueprintStore`, which carries the long
// version. Short form: `init` guarded on `ready`, which is set three awaits in,
// so React StrictMode's mount → cleanup → mount ran both mounts before
// `material_list` returned, both found no document, and both called
// `material_create("Main")` — two backend documents, two journals, one adopted
// and one leaked for the process. And `close` reads `get().doc`, which is null
// while the create is in flight, so it sent no `material_close` at all.
//
// A tombstoned document accepts no state and its `init` reply is answered with a
// `close`; a fresh `init` clears the tombstone first, so a StrictMode remount
// adopts rather than tears down.
let opening: Promise<void> | null = null;
let openGen = 0;
let tombstoned = false;

/** Test-only: forget any in-flight init and clear the tombstone. */
export function __resetMaterialInitForTest(): void {
  opening = null;
  openGen += 1;
  tombstoned = false;
}

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
    // Both synchronous, before any await — see the tombstone note above.
    tombstoned = false;
    if (opening) return opening;
    const gen = ++openGen;
    opening = (async () => {
      try {
        const registry = await materialIpc.registry();
        const registryById: Record<string, NodeDef> = {};
        for (const d of registry) registryById[d.typeId] = d;
        const existing = await materialIpc.list();
        const doc = existing[0] ?? (await materialIpc.create("Main"));
        if (tombstoned) {
          // Closed while this was in flight. The document exists NOW, and the
          // `close` that already ran had no id to name — so it is closed here.
          try {
            await materialIpc.close(doc.id);
          } catch (e) {
            console.error("material.close failed", e);
          }
          return;
        }
        set({ registry, registryById, doc, ready: true });
        void get().compile();
      } catch (e) {
        console.error("material.init failed", e);
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
      // **R2.F1.** A refused edit means the canvas is drawing a graph the
      // backend does not have, and `issues` cannot say so — a wire that never
      // entered the graph has nothing to be invalid about. Re-derive before the
      // recompile, so the shader is compiled from what actually exists.
      if (res.rejected > 0) {
        console.warn(`material.apply: the backend refused ${res.rejected} edit(s); re-deriving`);
        const authoritative = await materialIpc.get(doc.id);
        if (!tombstoned && get().doc?.id === doc.id) set({ doc: authoritative });
      }
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

  // Undo/redo carry the same `try`/`catch` every other backend call in this
  // store has (F-lens L7.M4) — their callers are `void undo()` (the toolbar) and
  // `void scope.undo()` (Ctrl+Z), so a rejection was an unhandled promise
  // rejection with nothing shown to the author.
  undo: async () => {
    const doc = get().doc;
    if (!doc) return;
    try {
      const restored = await materialIpc.undo(doc.id);
      if (restored) {
        set({ doc: restored, canRedo: true });
        void get().compile();
      }
    } catch (e) {
      console.error("material.undo failed", e);
    }
  },

  redo: async () => {
    const doc = get().doc;
    if (!doc) return;
    try {
      const restored = await materialIpc.redo(doc.id);
      if (restored) {
        set({ doc: restored, canUndo: true });
        void get().compile();
      }
    } catch (e) {
      console.error("material.redo failed", e);
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
