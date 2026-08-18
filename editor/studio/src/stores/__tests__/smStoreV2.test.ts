// @vitest-environment jsdom
//
// **The `.inf_sm` v2 authoring surface** (P29.5). P29.1 shipped a model the
// canvas could carry and not edit, and wrote down that the editing half is this
// wave's. These are the arms for it: every v2 shape the reader decodes has a
// store action that writes it, the round-trip through `save` is byte-stable, and
// the two rules that are easy to get silently wrong — a stale flat condition
// beside a new tree, and a blend-profile index that shifts under a delete — are
// asserted rather than described.
import { beforeEach, describe, expect, it, vi } from "vitest";

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
import type { SmDoc, SmMachineDto } from "../../lib/smTypes";
import { useSmStore, __resetSmInitForTest } from "../smStore";

function state(name: string): SmMachineDto["states"][number] {
  return {
    name,
    motion: { kind: "clip", clip: null },
    looping: true,
    speed: 1,
    x: 0,
    y: 0,
    onEnter: [],
    onExit: [],
  };
}

function makeDoc(): SmDoc {
  return {
    id: "sm1",
    name: "Main",
    machine: {
      states: [state("Idle"), state("Walk")],
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
  propose: ReturnType<typeof vi.fn>;
};

beforeEach(async () => {
  vi.clearAllMocks();
  __resetSmInitForTest();
  mockSm.list.mockResolvedValue([makeDoc()]);
  mockSm.listClips.mockResolvedValue([{ id: "c1", name: "Walk" }]);
  mockSm.save.mockResolvedValue("Main.inf_sm");
  mockSm.close.mockResolvedValue(undefined);
  useSmStore.setState({
    doc: null,
    clips: [],
    selectedTransition: null,
    ready: false,
    saving: false,
    proposing: false,
    path: [],
    proposalNotes: [],
    refusal: null,
  });
  await useSmStore.getState().init();
});

describe("v2 parameters", () => {
  it("declares each kind, edits it and takes it away", () => {
    const s = useSmStore.getState();
    s.addParam("float");
    s.addParam("trigger");
    s.addParam("bool");
    s.addParam("int");
    let m = useSmStore.getState().doc!.machine;
    expect(m.params.map((p) => p.kind)).toEqual(["float", "trigger", "bool", "int"]);
    // Names are unique out of the box, so a fresh table validates.
    expect(new Set(m.params.map((p) => p.name)).size).toBe(4);

    s.updateParam(0, { name: "speed", default: 1.5 });
    m = useSmStore.getState().doc!.machine;
    expect(m.params[0]).toEqual({ name: "speed", kind: "float", default: 1.5 });

    s.removeParam(1);
    expect(useSmStore.getState().doc!.machine.params.map((p) => p.kind)).toEqual([
      "float",
      "bool",
      "int",
    ]);
  });
});

describe("v2 transitions", () => {
  it("authors priority, interruption, curve, profile and any-state", () => {
    const s = useSmStore.getState();
    s.addTransition(0, 1);
    s.setTransition(0, {
      priority: 7,
      interruptSource: "sourceOrDestination",
      interruptBlend: "snap",
      curve: "easeInOut",
    });
    const t = useSmStore.getState().doc!.machine.transitions[0];
    expect(t).toMatchObject({
      priority: 7,
      interruptSource: "sourceOrDestination",
      interruptBlend: "snap",
      curve: "easeInOut",
    });

    s.addAnyTransition(1);
    const any = useSmStore.getState().doc!.machine.transitions[1];
    expect(any.from).toBeNull();
    expect(any.excludeSelf).toBe(true);
    expect(any.to).toBe(1);
  });

  // **The rule that is easy to get silently wrong.** `dto_to_transition` prefers
  // the flat list when it is non-null, so a builder that set the tree and left a
  // stale list behind would save the LIST and discard the tree the author just
  // drew. The store clears it in the same write.
  it("setting a tree clears the flat view so the save cannot prefer the stale one", () => {
    const s = useSmStore.getState();
    s.addTransition(0, 1);
    s.addCondition(0);
    expect(useSmStore.getState().doc!.machine.transitions[0].conditions).toHaveLength(1);

    s.setCondition(0, {
      kind: "or",
      terms: [
        { kind: "trigger", param: "jump" },
        { kind: "compare", param: "speed", op: ">", value: 2, valueKind: "float" },
      ],
    });
    const t = useSmStore.getState().doc!.machine.transitions[0];
    expect(t.conditions).toBeNull();
    expect(t.condition.kind).toBe("or");

    // …and the flat editors are inert on it afterwards, which is the second lock.
    s.addCondition(0);
    expect(useSmStore.getState().doc!.machine.transitions[0].conditions).toBeNull();
  });
});

describe("v2 blend profiles", () => {
  it("clamps a weight into the range the engine accepts", () => {
    const s = useSmStore.getState();
    s.addProfile();
    s.addProfileWeight(0);
    s.setProfileWeight(0, 0, 3.7, 4.2);
    expect(useSmStore.getState().doc!.machine.profiles[0].weights[0]).toEqual({
      joint: 4,
      weight: 1,
    });
    s.setProfileWeight(0, 0, 2, -1);
    expect(useSmStore.getState().doc!.machine.profiles[0].weights[0].weight).toBe(0);
  });

  // **A profile is an INDEX.** Deleting one silently re-points every later
  // reference at its neighbour, and `validate` accepts the result — so the
  // machine would keep saving and quietly mask the wrong joints.
  it("deleting a profile repairs every transition that referenced one", () => {
    const s = useSmStore.getState();
    s.addProfile(); // 0
    s.addProfile(); // 1
    s.addProfile(); // 2
    s.addTransition(0, 1);
    s.addTransition(1, 0);
    s.setTransition(0, { profile: 1 });
    s.setTransition(1, { profile: 2 });

    s.removeProfile(1);
    const m = useSmStore.getState().doc!.machine;
    expect(m.profiles).toHaveLength(2);
    expect(m.transitions[0].profile).toBeNull();
    expect(m.transitions[1].profile).toBe(1);
  });
});

describe("v2 states", () => {
  it("authors enter/exit notifies, dropping the empties a comma list leaves", () => {
    const s = useSmStore.getState();
    s.setStateEvents(0, ["idle_begin", "  ", ""], ["idle_end"]);
    const st = useSmStore.getState().doc!.machine.states[0];
    expect(st.onEnter).toEqual(["idle_begin"]);
    expect(st.onExit).toEqual(["idle_end"]);
  });

  it("nests ONE sub-machine and edits inside it through the path", () => {
    const s = useSmStore.getState();
    s.makeSubMachine(1);
    let st = useSmStore.getState().doc!.machine.states[1];
    expect(st.motion.kind).toBe("subMachine");

    // Drill in: the same verbs now edit the nested machine.
    s.setPath([1]);
    expect(useSmStore.getState().activeMachine()!.states).toHaveLength(1);
    useSmStore.getState().addState(10, 10);
    useSmStore.getState().addTransition(0, 1);
    const inner = useSmStore.getState().activeMachine()!;
    expect(inner.states).toHaveLength(2);
    expect(inner.transitions).toHaveLength(1);
    // …and the ROOT is untouched by all of it.
    expect(useSmStore.getState().doc!.machine.states).toHaveLength(2);

    // A sub-machine inside a sub-machine is refused: the engine's runtime holds
    // one inline nested slot, and `validate` rejects a second level.
    useSmStore.getState().makeSubMachine(0);
    st = useSmStore.getState().activeMachine()!.states[0];
    expect(st.motion.kind).toBe("clip");

    // Parameters stay the ROOT's even while drilled in — a nested machine that
    // declared its own is a file the reader refuses.
    useSmStore.getState().addParam("float");
    expect(useSmStore.getState().activeMachine()!.params).toHaveLength(0);
    expect(useSmStore.getState().doc!.machine.params).toHaveLength(1);
  });
});

describe("the save door", () => {
  // **The S1 property**: author → save → reopen is byte-stable. The backend is
  // mocked, so what this asserts is the half the frontend owns — the document
  // that goes over the wire is exactly the document the store holds, with every
  // v2 field on it, and pushing it twice pushes the same bytes.
  it("round-trips every v2 shape it authored, byte-stable", async () => {
    const s = useSmStore.getState();
    s.addParam("trigger");
    s.updateParam(0, { name: "jump" });
    s.addProfile();
    s.addProfileWeight(0);
    s.setProfileWeight(0, 0, 3, 0.25);
    s.addTransition(0, 1);
    s.setTransition(0, { priority: 3, curve: "easeOut", profile: 0, exitTime: 0.75 });
    s.setCondition(0, { kind: "not", term: { kind: "trigger", param: "jump" } });
    s.setStateEvents(0, ["begin"], ["end"]);
    s.makeSubMachine(1);

    await useSmStore.getState().save("Hero");
    const [, pushed] = mockSm.save.mock.calls[0];
    const before = JSON.stringify(pushed);
    expect(before).toBe(JSON.stringify(useSmStore.getState().doc));

    await useSmStore.getState().save("Hero");
    const [, again] = mockSm.save.mock.calls[1];
    expect(JSON.stringify(again)).toBe(before);

    // Every v2 shape really is in what went over the wire — the anti-vacuity
    // half, because a document with none of them would round-trip perfectly.
    expect(before).toContain('"kind":"trigger"');
    expect(before).toContain('"kind":"not"');
    expect(before).toContain('"kind":"subMachine"');
    expect(before).toContain('"priority":3');
    expect(before).toContain('"curve":"easeOut"');
    expect(before).toContain('"exitTime":0.75');
    expect(before).toContain('"onEnter":["begin"]');
  });

  // **The validator is the door and its refusal is shown**, not swallowed —
  // otherwise an author looks at a canvas that silently does not persist.
  it("surfaces the validator's refusal inline", async () => {
    mockSm.save.mockRejectedValueOnce(
      "this state machine cannot be saved: parameter `speed` is declared twice",
    );
    const file = await useSmStore.getState().save("Hero");
    expect(file).toBeNull();
    expect(useSmStore.getState().refusal).toContain("declared twice");

    // A later successful save clears it.
    mockSm.save.mockResolvedValueOnce("Main.inf_sm");
    await useSmStore.getState().save("Hero");
    expect(useSmStore.getState().refusal).toBeNull();
  });
});

describe("proposing a machine", () => {
  it("adopts the proposal as the document and keeps its reasoning", async () => {
    mockSm.propose.mockResolvedValue({
      machine: {
        states: [state("idle"), state("walk")],
        transitions: [],
        entry: 0,
        params: [{ name: "speed", kind: "float", default: 0 }],
        profiles: [],
      },
      notes: ["`walk` depicts 1.60 m/s (gait 1.00) and is the whole of `walk`"],
      triggers: ["play_jump"],
      refusal: null,
    });
    const why = await useSmStore.getState().propose(["c1"]);
    expect(why).toBeNull();
    expect(mockSm.propose).toHaveBeenCalledWith(["c1"]);
    const s = useSmStore.getState();
    expect(s.doc!.machine.states.map((x) => x.name)).toEqual(["idle", "walk"]);
    expect(s.doc!.machine.params[0].name).toBe("speed");
    expect(s.proposalNotes).toHaveLength(1);
    // The document id is the SAME one — a proposal replaces the machine, not the
    // backend document, so `save` still targets what `init` opened.
    expect(s.doc!.id).toBe("sm1");
    s.dismissNotes();
    expect(useSmStore.getState().proposalNotes).toEqual([]);
  });

  it("a refusal keeps the open document and says why", async () => {
    mockSm.propose.mockResolvedValue({
      machine: { states: [], transitions: [], entry: 0, params: [], profiles: [] },
      notes: [],
      triggers: [],
      refusal: "a proposal needs at least one clip",
    });
    const before = useSmStore.getState().doc!.machine.states.length;
    const why = await useSmStore.getState().propose(["c1"]);
    expect(why).toContain("at least one clip");
    expect(useSmStore.getState().doc!.machine.states).toHaveLength(before);
    expect(useSmStore.getState().refusal).toContain("at least one clip");
  });

  it("proposing from nothing is refused before the wire", async () => {
    const why = await useSmStore.getState().propose([]);
    expect(why).toBeTruthy();
    expect(mockSm.propose).not.toHaveBeenCalled();
  });
});
