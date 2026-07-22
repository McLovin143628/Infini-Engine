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
    try {
      const existing = await smIpc.list();
      const doc = existing[0] ?? (await smIpc.create("Main"));
      let clips: SmClipDto[] = [];
      try {
        clips = await smIpc.listClips();
      } catch {
        clips = []; // no open project yet
      }
      set({ doc, clips, ready: true });
    } catch (e) {
      console.error("sm.init failed", e);
    }
  },

  // Discard the editing surface: free the backend document and reset to an
  // un-inited state so a later re-open starts fresh instead of leaking the old
  // doc for the session. Called when the canvas panel unmounts (panel close).
  close: async () => {
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
      m.transitions = m.transitions
        .filter((t) => t.from !== index && t.to !== index)
        .map((t) => ({
          ...t,
          from: t.from > index ? t.from - 1 : t.from,
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
      m.transitions.push({ from, to, duration: 0.2, conditions: [], exitTime: null });
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

  addCondition: (index) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, (m) => {
        m.transitions[index]?.conditions.push({ var: "speed", op: ">" as SmOp, value: 0 });
      }),
    });
  },

  updateCondition: (index, ci, patch) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, (m) => {
        const c = m.transitions[index]?.conditions[ci];
        if (c) Object.assign(c, patch);
      }),
    });
  },

  removeCondition: (index, ci) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, (m) => void m.transitions[index]?.conditions.splice(ci, 1)),
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
