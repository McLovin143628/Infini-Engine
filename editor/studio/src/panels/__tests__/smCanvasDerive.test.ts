// The State Machine canvas's two pure projections (P11.2, extended P29.5).
//
// `deriveNodes`/`deriveEdges` are what makes the canvas fully controlled: the
// ReactFlow model is a `useMemo` over the store document and nothing else, so
// everything the canvas *decides* lives in these two functions and can be
// asserted without a DOM. The P29.5 decisions are the ones tested here — that an
// any-state transition is drawn as such rather than silently as an ordinary edge
// out of its own target, and that priority reaches the label.
import { describe, expect, it } from "vitest";

import { deriveEdges, deriveNodes } from "../sm/StateMachineCanvas";
import { newTransition, type SmMachineDto } from "../../lib/smTypes";

function state(name: string, x = 0): SmMachineDto["states"][number] {
  return {
    name,
    motion: { kind: "clip", clip: "c1" },
    looping: true,
    speed: 1,
    x,
    y: 0,
    onEnter: [],
    onExit: [],
  };
}

function machine(): SmMachineDto {
  return {
    states: [state("idle"), state("walk", 240), state("jump", 480)],
    transitions: [
      { ...newTransition(0, 1), conditions: [{ var: "speed", op: ">", value: 0.4 }] },
      {
        ...newTransition(0, 2),
        from: null,
        excludeSelf: true,
        priority: 10,
        exitTime: null,
        conditions: null,
        condition: { kind: "trigger", param: "play_jump" },
      },
      { ...newTransition(2, 0), exitTime: 0.9, conditions: null, condition: { kind: "always" } },
    ],
    entry: 0,
    params: [],
    profiles: [],
  };
}

const name = (id: string | null) => (id ? "Walk" : "(no clip)");

describe("deriveNodes", () => {
  it("carries the entry flag and the motion summary", () => {
    const n = deriveNodes(machine(), name);
    expect(n).toHaveLength(3);
    expect(n[0].data).toMatchObject({ name: "idle", isEntry: true, summary: "Walk" });
    expect(n[1].data).toMatchObject({ isEntry: false });
    expect(n[1].position).toEqual({ x: 240, y: 0 });
  });
});

describe("deriveEdges", () => {
  it("draws an any-state transition as one, and an ordinary one as one", () => {
    const e = deriveEdges(machine(), null);
    // The ordinary edge: solid, from its real source, labelled with its rule.
    expect(e[0]).toMatchObject({ source: "0", target: "1" });
    expect(e[0].style?.strokeDasharray).toBeUndefined();
    expect(e[0].label).toBe("1 cond");

    // The any-state edge: dashed, labelled `any`, and drawn on its own target
    // because there is no node to leave from.
    expect(e[1]).toMatchObject({ source: "2", target: "2" });
    expect(e[1].style?.strokeDasharray).toBe("6 3");
    expect(String(e[1].label)).toContain("any");
    // Priority reaches the label — it is the field that decides which of two
    // ready transitions fires, and an author cannot see it anywhere else.
    expect(String(e[1].label)).toContain("p10");
    // …and the condition tree is summarised rather than counted, because the
    // flat view cannot represent it.
    expect(String(e[1].label)).toContain("play_jump");
  });

  it("labels an exit-time gate and marks the selected edge", () => {
    const e = deriveEdges(machine(), 2);
    expect(String(e[2].label)).toContain("exit 0.9");
    expect(e[2].animated).toBe(true);
    expect(e[0].animated).toBe(false);
    expect(e[2].style?.strokeWidth).toBeGreaterThan(e[0].style!.strokeWidth as number);
  });

  it("an unconditional edge with nothing to say carries no label", () => {
    const m = machine();
    m.transitions = [newTransition(0, 1)];
    expect(deriveEdges(m, null)[0].label).toBeUndefined();
  });
});
