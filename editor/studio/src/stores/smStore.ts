/**
 * Animation State Machine editor store (P11.2, v2 authoring P29.5). A state
 * machine is a plain typed model (states + transitions), not an `inf-graph`
 * dataflow document — so unlike the blueprint/material/PCG stores there is no
 * optimistic-apply / server-undo round-trip. The frontend **owns** the document:
 * every gesture mutates the local `doc` in place, and `save` pushes the whole
 * document to the backend to persist a `.inf_sm` asset. One active document
 * (per-`.inf_sm` binding is a follow-up, matching the other graph editors).
 *
 * ## The v2 authoring surface (P29.5)
 *
 * P29.1 shipped a model the canvas could carry and not edit; this store closes
 * that. Everything `.inf_sm` v2 decodes is now authorable: typed parameters
 * (including `Trigger`), condition **trees**, transition priority and
 * interruption, blend curves and per-joint blend profiles, `exit_time`,
 * any-state sources, one level of nested sub-machines, and state enter/exit
 * events.
 *
 * Two structural decisions are worth naming, because both are load-bearing:
 *
 * * **`path` is how a sub-machine is edited.** `[]` is the root machine and
 *   `[i]` is the machine inside state `i`. Every state/transition action reads
 *   it, so there is one set of editing verbs rather than a nested copy of them —
 *   and the engine refuses a sub-machine inside a sub-machine, so the path is
 *   never longer than one.
 * * **Parameters and profiles are always the ROOT's.** `StateMachine::validate`
 *   refuses a nested machine that declares its own parameters (a sub-machine
 *   shares its parent's table), so a UI that let you add one inside a
 *   sub-machine would be authoring a file the reader rejects — which is the
 *   P29.2 A1 rule this whole editor is built on.
 *
 * The **validator remains the door**: `sm_save` calls `StateMachine::validate`
 * and refuses, and the refusal is surfaced inline rather than swallowed.
 */
import { create } from "zustand";

import { sm as smIpc } from "../lib/ipc";
import { registerUndoScope } from "../lib/undoScopes";
import { useShellStore } from "./shellStore";
import { newTransition } from "../lib/smTypes";
import type {
  SmClipDto,
  SmCondDto,
  SmConditionDto,
  SmDoc,
  SmMachineDto,
  SmOp,
  SmParamDto,
  SmParamKind,
  SmProfileDto,
  SmStateDto,
  SmTransitionDto,
} from "../lib/smTypes";

interface SmStore {
  doc: SmDoc | null;
  clips: SmClipDto[];
  /** Index of the selected transition (for the inspector), or null. */
  selectedTransition: number | null;
  /** Which machine is being edited: `[]` the root, `[i]` state `i`'s
   *  sub-machine. See the module docs. */
  path: number[];
  /** The last proposal's reasoning, shown beside the canvas until dismissed. */
  proposalNotes: string[];
  /** The last save refusal, verbatim from the validator. */
  refusal: string | null;
  ready: boolean;
  saving: boolean;
  proposing: boolean;

  init: () => Promise<void>;
  close: () => Promise<void>;
  clipName: (id: string | null) => string;
  /** The machine `path` names, or null when there is no document. */
  activeMachine: () => SmMachineDto | null;
  /** Drill into state `i`'s sub-machine, or back out with `[]`. */
  setPath: (path: number[]) => void;

  // ── state edits (all local) ──
  addState: (x: number, y: number) => void;
  renameState: (index: number, name: string) => void;
  deleteState: (index: number) => void;
  setEntry: (index: number) => void;
  moveState: (index: number, x: number, y: number) => void;
  setStateClip: (index: number, clip: string | null) => void;
  /** Replace a state's motion with a 2D blend space (P29.2's authoring panel).
   *  Returns `false` when no document is open, so the caller can say so. */
  setStateBlend2d: (
    index: number,
    paramX: string,
    paramY: string,
    entries: { x: number; y: number; clip: string | null }[],
  ) => boolean;
  setStateSpeed: (index: number, speed: number) => void;
  setStateLooping: (index: number, looping: boolean) => void;
  /** Enter/exit notify names (v2). */
  setStateEvents: (index: number, onEnter: string[], onExit: string[]) => void;
  /** Turn a state into a **sub-machine** (v2), seeded with one empty state so
   *  drilling in has something to show. */
  makeSubMachine: (index: number) => void;

  // ── transition edits (all local) ──
  addTransition: (from: number, to: number) => void;
  /** An **any-state** transition into `to` (v2). */
  addAnyTransition: (to: number) => void;
  deleteTransition: (index: number) => void;
  selectTransition: (index: number | null) => void;
  setTransition: (index: number, patch: Partial<SmTransitionDto>) => void;
  setTransitionDuration: (index: number, duration: number) => void;
  setTransitionExitTime: (index: number, exitTime: number | null) => void;
  /** Replace a transition's whole condition **tree** — the rule builder's one
   *  write door. It computes the new tree with the pure helpers in `smTypes`
   *  and hands it over, so the store never has to know what a tree is. */
  setCondition: (index: number, condition: SmCondDto) => void;
  addCondition: (index: number) => void;
  updateCondition: (index: number, ci: number, patch: Partial<SmConditionDto>) => void;
  removeCondition: (index: number, ci: number) => void;

  // ── parameters + profiles (root machine only — see the module docs) ──
  addParam: (kind: SmParamKind) => void;
  updateParam: (index: number, patch: Partial<SmParamDto>) => void;
  removeParam: (index: number) => void;
  addProfile: () => void;
  updateProfile: (index: number, patch: Partial<SmProfileDto>) => void;
  removeProfile: (index: number) => void;
  setProfileWeight: (index: number, wi: number, joint: number, weight: number) => void;
  addProfileWeight: (index: number) => void;
  removeProfileWeight: (index: number, wi: number) => void;

  /** Propose a machine over `clips` (P29.5, pillar S3) and adopt it as the
   *  document. Returns the refusal, if there was one. */
  propose: (clips: string[]) => Promise<string | null>;
  dismissNotes: () => void;

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

/** The machine `path` names inside `m`, or `null` when the path is stale. */
function machineAt(m: SmMachineDto, path: number[]): SmMachineDto | null {
  let cur: SmMachineDto | null = m;
  for (const i of path) {
    const st: SmStateDto | undefined = cur?.states[i];
    if (!st || st.motion.kind !== "subMachine") return null;
    cur = st.motion.machine;
  }
  return cur;
}

/** Produce a new machine from `doc` by mutating the one at `path`, keeping the
 *  reducer pure. A stale path mutates nothing — a **value**, because a path can
 *  go stale under an author who deleted the state it named. */
function editMachine(doc: SmDoc, path: number[], mut: (m: SmMachineDto) => void): SmDoc {
  const machine = structuredClone(doc.machine);
  const target = machineAt(machine, path);
  if (target) mut(target);
  return { ...doc, machine };
}

export const useSmStore = create<SmStore>((set, get) => ({
  doc: null,
  clips: [],
  selectedTransition: null,
  path: [],
  proposalNotes: [],
  refusal: null,
  ready: false,
  saving: false,
  proposing: false,

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
    set({
      doc: null,
      ready: false,
      selectedTransition: null,
      path: [],
      proposalNotes: [],
      refusal: null,
    });
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

  activeMachine: () => {
    const doc = get().doc;
    return doc ? machineAt(doc.machine, get().path) : null;
  },

  setPath: (path) => set({ path, selectedTransition: null }),

  addState: (x, y) => {
    const doc = get().doc;
    if (!doc) return;
    const next = editMachine(doc, get().path, (m) => {
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
    set({
      doc: editMachine(
        doc,
        get().path,
        (m) => void (m.states[index] && (m.states[index].name = name)),
      ),
    });
  },

  deleteState: (index) => {
    const doc = get().doc;
    if (!doc) return;
    const next = editMachine(doc, get().path, (m) => {
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
    set({ doc: editMachine(doc, get().path, (m) => void (m.entry = index)) });
  },

  moveState: (index, x, y) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, get().path, (m) => {
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
      doc: editMachine(doc, get().path, (m) => {
        const st = m.states[index];
        if (st) st.motion = { kind: "clip", clip };
      }),
    });
  },

  // The blend-space panel's write door (P29.2). It lives here rather than in
  // `blendSpaceStore` because the `.inf_sm` document has one owner, and a second
  // writer would be a second thing `sm_save` has to agree with.
  setStateBlend2d: (index, paramX, paramY, entries) => {
    const doc = get().doc;
    const active = get().activeMachine();
    if (!doc || !active?.states[index]) return false;
    set({
      doc: editMachine(doc, get().path, (m) => {
        const st = m.states[index];
        if (st) {
          st.motion = {
            kind: "blend2d",
            paramX,
            paramY,
            entries: entries.map((e) => ({ x: e.x, y: e.y, clip: e.clip })),
          };
        }
      }),
    });
    return true;
  },

  setStateSpeed: (index, speed) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(
        doc,
        get().path,
        (m) => void (m.states[index] && (m.states[index].speed = speed)),
      ),
    });
  },

  setStateLooping: (index, looping) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(
        doc,
        get().path,
        (m) => void (m.states[index] && (m.states[index].looping = looping)),
      ),
    });
  },

  setStateEvents: (index, onEnter, onExit) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, get().path, (m) => {
        const st = m.states[index];
        if (st) {
          st.onEnter = onEnter.filter((n) => n.trim() !== "");
          st.onExit = onExit.filter((n) => n.trim() !== "");
        }
      }),
    });
  },

  // One level only, and the guard is not cosmetic: `StateMachine::validate`
  // refuses a sub-machine inside a sub-machine (the runtime's nested play state
  // is one inline slot, because `SmRuntime` is `Copy`), so offering it here
  // would author a file the reader rejects.
  makeSubMachine: (index) => {
    const doc = get().doc;
    if (!doc || get().path.length > 0) return;
    set({
      doc: editMachine(doc, [], (m) => {
        const st = m.states[index];
        if (!st || st.motion.kind === "subMachine") return;
        st.motion = {
          kind: "subMachine",
          machine: {
            states: [
              {
                name: "State 1",
                motion: { kind: "clip", clip: null },
                looping: true,
                speed: 1,
                x: 0,
                y: 0,
                onEnter: [],
                onExit: [],
              },
            ],
            transitions: [],
            entry: 0,
            params: [],
            profiles: [],
          },
        };
      }),
    });
  },

  addTransition: (from, to) => {
    const doc = get().doc;
    const active = get().activeMachine();
    if (!doc || !active || from === to) return;
    // Skip an exact duplicate edge.
    if (active.transitions.some((t) => t.from === from && t.to === to)) return;
    const next = editMachine(doc, get().path, (m) => {
      m.transitions.push(newTransition(from, to));
    });
    const m = machineAt(next.machine, get().path);
    set({ doc: next, selectedTransition: m ? m.transitions.length - 1 : null });
  },

  addAnyTransition: (to) => {
    const doc = get().doc;
    if (!doc) return;
    const next = editMachine(doc, get().path, (m) => {
      m.transitions.push({ ...newTransition(0, to), from: null, excludeSelf: true });
    });
    const m = machineAt(next.machine, get().path);
    set({ doc: next, selectedTransition: m ? m.transitions.length - 1 : null });
  },

  deleteTransition: (index) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, get().path, (m) => void m.transitions.splice(index, 1)),
      selectedTransition: null,
    });
  },

  selectTransition: (index) => set({ selectedTransition: index }),

  setTransition: (index, patch) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, get().path, (m) => {
        const t = m.transitions[index];
        if (t) Object.assign(t, patch);
      }),
    });
  },

  setTransitionDuration: (index, duration) => get().setTransition(index, { duration }),

  setTransitionExitTime: (index, exitTime) => get().setTransition(index, { exitTime }),

  // **The rule builder's write door.** It replaces the tree AND clears the flat
  // view, because the two must not disagree: `dto_to_transition` prefers
  // `conditions` when it is non-null, so leaving a stale flat list beside a new
  // tree would save the list and silently discard the tree the author just
  // built. `null` is exactly what the backend sends for a tree the flat view
  // cannot represent, so this is the same state, reached from the other side.
  setCondition: (index, condition) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, get().path, (m) => {
        const t = m.transitions[index];
        if (t) {
          t.condition = condition;
          t.conditions = null;
        }
      }),
    });
  },

  // The three flat-condition editors are no-ops on a transition whose tree the
  // flat view cannot represent (`conditions === null`). Refusing is the point:
  // the alternative is materialising a list, which is the flattening the whole
  // two-field design exists to prevent. The inspector offers the rule builder
  // for those, so this guard is a second lock rather than the only one.
  addCondition: (index) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, get().path, (m) => {
        m.transitions[index]?.conditions?.push({ var: "speed", op: ">" as SmOp, value: 0 });
      }),
    });
  },

  updateCondition: (index, ci, patch) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, get().path, (m) => {
        const c = m.transitions[index]?.conditions?.[ci];
        if (c) Object.assign(c, patch);
      }),
    });
  },

  removeCondition: (index, ci) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, get().path, (m) => void m.transitions[index]?.conditions?.splice(ci, 1)),
    });
  },

  // ── parameters and profiles: the ROOT machine's, always (module docs) ──
  addParam: (kind) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, [], (m) => {
        let n = m.params.length + 1;
        while (m.params.some((p) => p.name === `param${n}`)) n += 1;
        m.params.push({ name: `param${n}`, kind, default: 0 });
      }),
    });
  },

  updateParam: (index, patch) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, [], (m) => {
        const p = m.params[index];
        if (p) Object.assign(p, patch);
      }),
    });
  },

  removeParam: (index) => {
    const doc = get().doc;
    if (!doc) return;
    set({ doc: editMachine(doc, [], (m) => void m.params.splice(index, 1)) });
  },

  addProfile: () => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, [], (m) => {
        m.profiles.push({ name: `Profile ${m.profiles.length + 1}`, weights: [] });
      }),
    });
  },

  updateProfile: (index, patch) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, [], (m) => {
        const p = m.profiles[index];
        if (p) Object.assign(p, patch);
      }),
    });
  },

  // A profile a transition points at cannot simply vanish: `profile` is an
  // INDEX, so removing one silently re-points every later reference at its
  // neighbour and `validate` accepts the result. Every reference is repaired
  // here — cleared for the removed profile, decremented past it — at every
  // depth, because a sub-machine's transitions index the root's profile table
  // too.
  removeProfile: (index) => {
    const doc = get().doc;
    if (!doc) return;
    const repair = (m: SmMachineDto) => {
      for (const t of m.transitions) {
        if (t.profile === index) t.profile = null;
        else if (t.profile !== null && t.profile > index) t.profile -= 1;
      }
      for (const s of m.states) {
        if (s.motion.kind === "subMachine") repair(s.motion.machine);
      }
    };
    set({
      doc: editMachine(doc, [], (m) => {
        m.profiles.splice(index, 1);
        repair(m);
      }),
    });
  },

  addProfileWeight: (index) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, [], (m) => {
        m.profiles[index]?.weights.push({ joint: 0, weight: 1 });
      }),
    });
  },

  setProfileWeight: (index, wi, joint, weight) => {
    const doc = get().doc;
    if (!doc) return;
    set({
      doc: editMachine(doc, [], (m) => {
        const w = m.profiles[index]?.weights[wi];
        if (w) {
          w.joint = Math.max(0, Math.round(joint));
          // The engine refuses a weight outside [0,1] (`BadProfileWeight`), so
          // the clamp is here rather than in a toast after the save fails.
          w.weight = Math.min(1, Math.max(0, weight));
        }
      }),
    });
  },

  removeProfileWeight: (index, wi) => {
    const doc = get().doc;
    if (!doc) return;
    set({ doc: editMachine(doc, [], (m) => void m.profiles[index]?.weights.splice(wi, 1)) });
  },

  propose: async (clips) => {
    const doc = get().doc;
    if (!doc || clips.length === 0) return "pick at least one clip to propose from";
    set({ proposing: true });
    try {
      const p = await smIpc.propose(clips);
      if (p.refusal) {
        set({ proposalNotes: p.notes, refusal: p.refusal });
        return p.refusal;
      }
      set({
        doc: { ...doc, machine: p.machine },
        proposalNotes: p.notes,
        refusal: null,
        selectedTransition: null,
        path: [],
      });
      return null;
    } catch (e) {
      const why = String(e);
      set({ refusal: why });
      return why;
    } finally {
      set({ proposing: false });
    }
  },

  dismissNotes: () => set({ proposalNotes: [] }),

  save: async (name) => {
    const doc = get().doc;
    if (!doc) return null;
    set({ saving: true, refusal: null });
    try {
      return await smIpc.save(doc.id, doc, name);
    } catch (e) {
      // **The validator is the door, and its refusal is shown.** `sm_save` calls
      // `StateMachine::validate` and rejects with the reason; swallowing it into
      // a console line would leave an author looking at a canvas that silently
      // does not persist.
      const why = String(e);
      set({ refusal: why });
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
