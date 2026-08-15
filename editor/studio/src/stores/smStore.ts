/**
 * Animation State Machine editor store (P11.2). A state machine is a plain typed
 * model (states + transitions), not an `inf-graph` dataflow document — so unlike
 * the blueprint/material/PCG stores there is no optimistic-apply / server-undo
 * round-trip. The frontend **owns** the document: every gesture mutates the local
 * `doc` in place, and `save` pushes the whole document to the backend to persist
 * a `.inf_sm` asset. One active document in v1 (per-`.inf_sm` binding is a
 * follow-up, matching the other graph editors).
 */
import { create } from "zustand";

import { sm as smIpc } from "../lib/ipc";
import { registerUndoScope } from "../lib/undoScopes";
import { useShellStore } from "./shellStore";
import { newTransition } from "../lib/smTypes";
import type {
  SmClipDto,
  SmConditionDto,
  SmDoc,
  SmMachineDto,
  SmOp,
  SmStateDto,
} from "../lib/smTypes";

interface SmStore {
  doc: SmDoc | null;
  clips: SmClipDto[];
  /** Index of the selected transition (for the inspector), or null. */
  selectedTransition: number | null;
  ready: boolean;
  saving: boolean;

  init: () => Promise<void>;
  close: () => Promise<void>;
  clipName: (id: string | null) => string;

  // ── state edits (all local) ──
  addState: (x: number, y: number) => void;
  renameState: (index: number, name: string) => void;
  deleteState: (index: number) => void;
  setEntry: (index: number) => void;
  moveState: (index: number, x: number, y: number) => void;
  setStateClip: (index: number, clip: string | null) => void;
  setStateSpeed: (index: number, speed: number) => void;
  setStateLooping: (index: number, looping: boolean) => void;

  // ── transition edits (all local) ──
  addTransition: (from: number, to: number) => void;
  deleteTransition: (index: number) => void;
  selectTransition: (index: number | null) => void;
  setTransitionDuration: (index: number, duration: number) => void;
  setTransitionExitTime: (index: number, exitTime: number | null) => void;
  addCondition: (index: number) => void;
  updateCondition: (index: number, ci: number, patch: Partial<SmConditionDto>) => void;
  removeCondition: (index: number, ci: number) => void;

  save: (name: string) => Promise<string | null>;
}

// ── The tombstone (F-lens L7.M3) ─────────────────────────────────────────────
//
// Ported from `dccStore` by way of `blueprintStore`, which carries the long
// version. Short form: `init` guarded on `ready`, which is set after the list
// (and the clip list), so React StrictMode's mount → cleanup → mount ran both
// mounts before `sm_list` returned, both found no document, and both called
// `sm_create("Main")` — two backend documents, one adopted and one leaked for
// the process. And `close` reads `get().doc`, which is null while the create is
// in flight, so it sent no `sm_close` at all.
//
// A tombstoned document accepts no state and its `init` reply is answered with a
// `close`; a fresh `init` clears the tombstone first, so a StrictMode remount
// adopts rather than tears down. The tombstone is re-checked after the CLIP
// list too — this store has an await after the document exists, so a close can
// land in that window as well.
let opening: Promise<void> | null = null;
let openGen = 0;
let tombstoned = false;

/** Test-only: forget any in-flight init and clear the tombstone. */
export function __resetSmInitForTest(): void {
  opening = null;
  openGen += 1;
  tombstoned = false;
}

/** Produce a new machine from `doc` via `mut`, keeping the reducer pure. */
function editMachine(doc: SmDoc, mut: (m: SmMachineDto) => void): SmDoc {
  const machine = structuredClone(doc.machine);
  mut(machine);
  return { ...doc, machine };
}

export const useSmStore = create<SmStore>((set, get) => ({
  doc: null,
  clips: [],
  selectedTransition: null,
  ready: false,
  saving: false,

  init: async () => {
    if (get().ready) return;
    // Both synchronous, before any await — see the tombstone note above.
    tombstoned = false;
    if (opening) return opening;
    const gen = ++openGen;
    opening = (async () => {
      try {
        const existing = await smIpc.list();
        const doc = existing[0] ?? (await smIpc.create("Main"));
        if (tombstoned) {
          // Closed while this was in flight. The document exists NOW, and the
          // `close` that already ran had no id to name — so it is closed here.
          try {
            await smIpc.close(doc.id);
          } catch (e) {
            console.error("sm.close failed", e);
          }
          return;
        }
        let clips: SmClipDto[] = [];
        try {
          clips = await smIpc.listClips();
        } catch {
          clips = []; // no open project yet
        }
        if (tombstoned) {
          try {
            await smIpc.close(doc.id);
          } catch (e) {
            console.error("sm.close failed", e);
          }
          return;
        }
        set({ doc, clips, ready: true });
      } catch (e) {
        console.error("sm.init failed", e);
      } finally {
        // Only the LATEST init clears the memo — an older one resolving late
        // must not free a newer one's slot.
        if (openGen === gen) opening = null;
      }
    })();
    return opening;
  },

  // Discard the editing surface: free the backend document and reset to an
  // un-inited state so a later re-open starts fresh instead of leaking the old
  // doc for the session. Called when the canvas panel unmounts (panel close).
  //
  // `opening` is deliberately left alone: an init still in flight has to keep
  // its memo so a StrictMode remount JOINS it rather than minting a second
  // document, and it clears the memo itself in its own `finally`.
  close: async () => {
    tombstoned = true;
    const doc = get().doc;
    set({ doc: null, ready: false, selectedTransition: null });
    if (doc) {
      try {
        await smIpc.close(doc.id);
      } catch (e) {
        console.error("sm.close failed", e);
      }
    }
  },

  clipName: (id) => {
    if (!id) return "(no clip)";
    const c = get().clips.find((c) => c.id === id);
    return c ? c.name : id.slice(0, 8);
  },

  addState: (x, y) => {
    const doc = get().doc;
    if (!doc) return;
    const next = editMachine(doc, (m) => {
      const st: SmStateDto = {
        name: `State ${m.states.length + 1}`,
        motion: { kind: "clip", clip: null },
        looping: true,
        speed: 1,
        x,
        y,
        onEnter: [],
        onExit: [],
      };
      m.states.push(st);
    });
    set({ doc: next });
  },

  renameState: (index, name) => {
    const doc = get().doc;
    if (!doc) return;
    set({ doc: editMachine(doc, (m) => void (m.states[index] && (m.states[index].name = name))) });
  },

  deleteState: (index) => {
    const doc = get().doc;
    if (!doc) return;
    const next = editMachine(doc, (m) => {
      m.states.splice(index, 1);
      // Drop transitions touching the removed state; reindex the survivors.
      // An any-state transition has NO source, so it survives a state being
      // deleted (it left "any state", and there are still states). Only its
      // target can strand it.
      m.transitions = m.transitions
        .filter((t) => t.from !== index && t.to !== index)
        .map((t) => ({
          ...t,
          from: t.from !== null && t.from > index ? t.from - 1 : t.from,
          to: t.to > index ? t.to - 1 : t.to,
        }));
      if (m.entry === index) m.entry = 0;
      else if (m.entry > index) m.entry -= 1;
    });
    set({ doc: next, selectedTransition: null });
  },

  setEntry: (index) => {
    const doc = get().doc;
    if (!doc) return;
    set({ doc: editMachine(doc, (m) => void (m.entry = index)) });
  },

  moveState: (index, x, y) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, (m) => {
        if (m.states[index]) {
          m.states[index].x = x;
          m.states[index].y = y;
        }
      }),
    });
  },

  setStateClip: (index, clip) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, (m) => {
        const st = m.states[index];
        if (st) st.motion = { kind: "clip", clip };
      }),
    });
  },

  setStateSpeed: (index, speed) => {
    const doc = get().doc;
    if (!doc) return;
    set({ doc: editMachine(doc, (m) => void (m.states[index] && (m.states[index].speed = speed))) });
  },

  setStateLooping: (index, looping) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, (m) => void (m.states[index] && (m.states[index].looping = looping))),
    });
  },

  addTransition: (from, to) => {
    const doc = get().doc;
    if (!doc || from === to) return;
    // Skip an exact duplicate edge.
    if (doc.machine.transitions.some((t) => t.from === from && t.to === to)) return;
    const next = editMachine(doc, (m) => {
      m.transitions.push(newTransition(from, to));
    });
    set({ doc: next, selectedTransition: next.machine.transitions.length - 1 });
  },

  deleteTransition: (index) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, (m) => void m.transitions.splice(index, 1)),
      selectedTransition: null,
    });
  },

  selectTransition: (index) => set({ selectedTransition: index }),

  setTransitionDuration: (index, duration) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(
        doc,
        (m) => void (m.transitions[index] && (m.transitions[index].duration = duration)),
      ),
    });
  },

  setTransitionExitTime: (index, exitTime) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(
        doc,
        (m) => void (m.transitions[index] && (m.transitions[index].exitTime = exitTime)),
      ),
    });
  },

  // The three flat-condition editors are no-ops on a transition whose tree the
  // flat view cannot represent (`conditions === null`). Refusing is the point:
  // the alternative is materialising a list, which is the flattening the whole
  // two-field design exists to prevent. The inspector renders those transitions
  // read-only, so this guard is a second lock rather than the only one.
  addCondition: (index) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, (m) => {
        m.transitions[index]?.conditions?.push({ var: "speed", op: ">" as SmOp, value: 0 });
      }),
    });
  },

  updateCondition: (index, ci, patch) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, (m) => {
        const c = m.transitions[index]?.conditions?.[ci];
        if (c) Object.assign(c, patch);
      }),
    });
  },

  removeCondition: (index, ci) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, (m) => void m.transitions[index]?.conditions?.splice(ci, 1)),
    });
  },

  save: async (name) => {
    const doc = get().doc;
    if (!doc) return null;
    set({ saving: true });
    try {
      return await smIpc.save(doc.id, doc, name);
    } catch (e) {
      console.error("sm.save failed", e);
      return null;
    } finally {
      set({ saving: false });
    }
  },
}));

// **The State Machine editor claims Ctrl+Z and says it has no undo** (P23.2a).
//
// It genuinely has none: the frontend owns the `.inf_sm` document outright, so
// there is no journal on either side (see the module doc). Claiming the scope
// anyway is the honest choice — the alternative is falling through to the scene
// default, which would undo an ACTOR MOVE while the author is looking at a state
// machine. That is precisely the bug this routing exists to fix, and "nothing
// happened, and here is why" beats "something happened somewhere else".
registerUndoScope("stateMachine", {
  undo: () =>
    useShellStore
      .getState()
      .pushStatus("The State Machine editor has no undo yet (P23 ledger)."),
  redo: () =>
    useShellStore
      .getState()
      .pushStatus("The State Machine editor has no redo yet (P23 ledger)."),
});
