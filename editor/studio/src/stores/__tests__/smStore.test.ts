// @vitest-environment jsdom
//
// Animation State Machine store (P11.2). The machine is a plain typed model owned
// by the frontend, so this verifies the *local* reducers (add/rename/delete state,
// entry reindexing, connect/delete transitions, condition editing) and that `save`
// pushes the whole document. IPC is mocked.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  sm: {
    list: vi.fn(),
    create: vi.fn(),
    get: vi.fn(),
    close: vi.fn(),
    save: vi.fn(),
    listClips: vi.fn(),
    propose: vi.fn(),
  },
}));

import { sm } from "../../lib/ipc";
import type { SmDoc } from "../../lib/smTypes";
import { useSmStore } from "../smStore";

function makeDoc(): SmDoc {
  return {
    id: "sm1",
    name: "Main",
    machine: {
      states: [
        {
          name: "Idle",
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
}

const mockSm = sm as unknown as {
  list: ReturnType<typeof vi.fn>;
  create: ReturnType<typeof vi.fn>;
  close: ReturnType<typeof vi.fn>;
  save: ReturnType<typeof vi.fn>;
  listClips: ReturnType<typeof vi.fn>;
};

beforeEach(async () => {
  vi.clearAllMocks();
  mockSm.list.mockResolvedValue([makeDoc()]);
  mockSm.listClips.mockResolvedValue([{ id: "c1", name: "Walk" }]);
  mockSm.save.mockResolvedValue("Main.inf_sm");
  mockSm.close.mockResolvedValue(undefined);
  // Reset the store to a fresh, un-inited state.
  useSmStore.setState({
    doc: null,
    clips: [],
    selectedTransition: null,
    ready: false,
    saving: false,
    path: [],
    proposalNotes: [],
    refusal: null,
    proposing: false,
  });
  await useSmStore.getState().init();
});

afterEach(() => {
  useSmStore.setState({ ready: false });
});

describe("smStore", () => {
  it("init seeds the first document + clips", () => {
    const s = useSmStore.getState();
    expect(s.ready).toBe(true);
    expect(s.doc?.machine.states.length).toBe(1);
    expect(s.clips).toEqual([{ id: "c1", name: "Walk" }]);
    expect(s.clipName("c1")).toBe("Walk");
  });

  it("adds and connects states into a transition", () => {
    const s = useSmStore.getState();
    s.addState(100, 50);
    expect(useSmStore.getState().doc?.machine.states.length).toBe(2);
    s.addTransition(0, 1);
    const m = useSmStore.getState().doc!.machine;
    expect(m.transitions).toHaveLength(1);
    expect(m.transitions[0]).toMatchObject({ from: 0, to: 1, duration: 0.2, conditions: [] });
    // A duplicate edge and a self-edge are rejected.
    s.addTransition(0, 1);
    s.addTransition(1, 1);
    expect(useSmStore.getState().doc!.machine.transitions).toHaveLength(1);
  });

  it("deleting a state reindexes transitions and the entry", () => {
    const s = useSmStore.getState();
    s.addState(1, 1); // index 1
    s.addState(2, 2); // index 2
    s.addTransition(1, 2);
    s.setEntry(2);
    // Delete state 0 → transition (1→2) becomes (0→1); entry 2 → 1.
    s.deleteState(0);
    const m = useSmStore.getState().doc!.machine;
    expect(m.states.length).toBe(2);
    expect(m.transitions[0]).toMatchObject({ from: 0, to: 1 });
    expect(m.entry).toBe(1);
  });

  it("edits transition conditions", () => {
    const s = useSmStore.getState();
    s.addState(1, 1);
    s.addTransition(0, 1);
    s.addCondition(0);
    s.updateCondition(0, 0, { var: "moving", op: ">", value: 0.5 });
    expect(useSmStore.getState().doc!.machine.transitions[0].conditions![0]).toEqual({
      var: "moving",
      op: ">",
      value: 0.5,
    });
    s.removeCondition(0, 0);
    expect(useSmStore.getState().doc!.machine.transitions[0].conditions).toHaveLength(0);
  });

  // v2: a transition whose condition tree the flat inspector cannot draw comes
  // back with `conditions: null`, and the three flat editors must then be
  // NO-OPS -- materialising a list is exactly the flattening that would delete
  // the tree on the next save.
  it("refuses to flatten a condition tree the flat editors cannot represent", () => {
    const s = useSmStore.getState();
    s.addState(1, 1);
    s.addTransition(0, 1);
    // The shape Ring 2 sends for an Or/Not/trigger/typed tree.
    useSmStore.setState((st) => {
      const doc = structuredClone(st.doc!);
      doc.machine.transitions[0].conditions = null;
      doc.machine.transitions[0].condition = {
        kind: "or",
        terms: [
          { kind: "trigger", param: "jump" },
          { kind: "compare", param: "stance", op: "==", value: 2, valueKind: "int" },
        ],
      };
      return { doc };
    });
    s.addCondition(0);
    s.updateCondition(0, 0, { var: "hacked" });
    s.removeCondition(0, 0);
    const tr = useSmStore.getState().doc!.machine.transitions[0];
    expect(tr.conditions).toBeNull();
    expect(tr.condition).toEqual({
      kind: "or",
      terms: [
        { kind: "trigger", param: "jump" },
        { kind: "compare", param: "stance", op: "==", value: 2, valueKind: "int" },
      ],
    });
  });

  it("sets exit time and clip motion", () => {
    const s = useSmStore.getState();
    s.setStateClip(0, "c1");
    expect(useSmStore.getState().doc!.machine.states[0].motion).toEqual({ kind: "clip", clip: "c1" });
    s.addState(1, 1);
    s.addTransition(0, 1);
    s.setTransitionExitTime(0, 0.8);
    expect(useSmStore.getState().doc!.machine.transitions[0].exitTime).toBe(0.8);
  });

  it("save pushes the whole document", async () => {
    const s = useSmStore.getState();
    const file = await s.save("Hero");
    expect(file).toBe("Main.inf_sm");
    expect(mockSm.save).toHaveBeenCalledWith("sm1", useSmStore.getState().doc, "Hero");
  });

  it("close frees the backend document and resets to an un-inited state", async () => {
    await useSmStore.getState().close();
    expect(mockSm.close).toHaveBeenCalledWith("sm1");
    const s = useSmStore.getState();
    expect(s.doc).toBeNull();
    expect(s.ready).toBe(false);
  });
});
