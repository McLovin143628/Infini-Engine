// The condition-tree helpers the P29.5 rule builder is built on.
//
// These are the whole reason the builder can be fully controlled: every gesture
// is `helper(tree) -> new tree`, so the component holds no copy of the document
// and nothing has to be kept in sync. A bug here is a transition that fires when
// it should not, which is why the mutation-shaped cases (removing the root,
// removing a `not`'s only operand, a stale path) are asserted as **values**
// rather than left to crash.
import { describe, expect, it } from "vitest";

import {
  addTermAt,
  condAt,
  condChildren,
  conditionSummary,
  defaultCond,
  removeCondAt,
  setCondAt,
  type SmCondDto,
} from "../smTypes";

function tree(): SmCondDto {
  return {
    kind: "and",
    terms: [
      { kind: "compare", param: "speed", op: ">", value: 1, valueKind: "float" },
      {
        kind: "or",
        terms: [
          { kind: "trigger", param: "jump" },
          { kind: "not", term: { kind: "compare", param: "grounded", op: "==", value: 0, valueKind: "bool" } },
        ],
      },
    ],
  };
}

describe("condition tree helpers", () => {
  it("addresses every node by path, and answers null past the end", () => {
    const t = tree();
    expect(condAt(t, [])).toBe(t);
    expect(condAt(t, [0])?.kind).toBe("compare");
    expect(condAt(t, [1])?.kind).toBe("or");
    expect(condAt(t, [1, 0])?.kind).toBe("trigger");
    expect(condAt(t, [1, 1, 0])?.kind).toBe("compare");
    expect(condAt(t, [9])).toBeNull();
    expect(condAt(t, [1, 9, 0])).toBeNull();
    // A leaf has no children — the recursion's base case, asserted.
    expect(condChildren({ kind: "always" })).toEqual([]);
  });

  it("replaces a node without touching the original tree", () => {
    const t = tree();
    const next = setCondAt(t, [1, 0], { kind: "always" });
    expect(condAt(next, [1, 0])?.kind).toBe("always");
    // Immutable: the input is exactly what it was.
    expect(condAt(t, [1, 0])?.kind).toBe("trigger");
    expect(t).toEqual(tree());
    // A stale path changes nothing, and is a VALUE.
    expect(setCondAt(t, [7, 7], { kind: "always" })).toEqual(t);
  });

  it("adds a term only to a group", () => {
    const t = tree();
    const next = addTermAt(t, [1], defaultCond("compare", "aim"));
    expect(condChildren(condAt(next, [1])!)).toHaveLength(3);
    expect(condAt(next, [1, 2])).toMatchObject({ param: "aim", op: ">", valueKind: "float" });
    // A leaf is not a group: adding to one is refused, as a value.
    expect(addTermAt(t, [0], { kind: "always" })).toEqual(t);
    expect(addTermAt(t, [1, 1], { kind: "always" })).toEqual(t);
  });

  it("removes a term from its group and refuses the two holes", () => {
    const t = tree();
    const next = removeCondAt(t, [0]);
    expect(condChildren(next)).toHaveLength(1);
    expect(condAt(next, [0])?.kind).toBe("or");

    // The ROOT cannot be removed — there would be no tree left.
    expect(removeCondAt(t, [])).toEqual(t);
    // Nor can a `not`'s single operand: `not` of nothing is not a condition.
    // The builder offers "change kind" for that instead.
    expect(removeCondAt(t, [1, 1, 0])).toEqual(t);
  });

  it("every kind has a default that is immediately meaningful", () => {
    for (const k of ["always", "compare", "trigger", "and", "or", "not"] as const) {
      const c = defaultCond(k);
      expect(c.kind).toBe(k);
      // It renders — which is the only thing "meaningful" can mean for a node
      // nobody has typed into yet.
      expect(conditionSummary(c).length).toBeGreaterThan(0);
    }
    expect(defaultCond("compare", "gait")).toMatchObject({ param: "gait" });
  });

  it("summarises a tree the flat view could never draw", () => {
    expect(conditionSummary(tree())).toBe(
      "speed > 1 and (jump (trigger) or not grounded == 0)",
    );
  });
});
