import { describe, expect, it } from "vitest";
import type { SceneDelta } from "../../bindings/SceneDelta";
import type { SceneNode } from "../../bindings/SceneNode";
import { isEffectivelyVisible, outlinerRows, useSceneStore } from "../sceneStore";

function node(guid: string, children: string[] = [], extra: Partial<SceneNode> = {}): SceneNode {
  return {
    guid,
    name: guid,
    kind: "Actor",
    visible: true,
    effective_visible: true,
    parent: null,
    children,
    ...extra,
  };
}

describe("outlinerRows", () => {
  const nodes: Record<string, SceneNode> = {
    a: node("a", ["b"]),
    b: node("b", ["c"], { parent: "a" }),
    c: node("c", [], { parent: "b" }),
  };

  it("walks depth-first with depths", () => {
    const rows = outlinerRows(nodes, ["a"], []);
    expect(rows.map((r) => r.node.guid)).toEqual(["a", "b", "c"]);
    expect(rows.find((r) => r.node.guid === "c")!.depth).toBe(2);
  });

  it("honors collapsed guids", () => {
    const rows = outlinerRows(nodes, ["a"], ["b"]);
    expect(rows.some((r) => r.node.guid === "b")).toBe(true);
    expect(rows.some((r) => r.node.guid === "c")).toBe(false);
  });
});

describe("isEffectivelyVisible", () => {
  it("reads the node's computed flag", () => {
    const nodes = { a: node("a", [], { effective_visible: false }) };
    expect(isEffectivelyVisible(nodes, "a")).toBe(false);
    expect(isEffectivelyVisible(nodes, "missing")).toBe(true);
  });
});

describe("applyDelta reducer", () => {
  it("adds, updates, and removes nodes", () => {
    const store = useSceneStore.getState();
    store.applySnapshot({
      version: 1,
      roots: ["a"],
      nodes: [node("a")],
      selection: [],
      dirty: false,
      title: "T",
      can_undo: false,
      can_redo: false,
      undo_label: null,
      redo_label: null,
    });
    const delta: SceneDelta = {
      version: 2,
      added: [node("b")],
      removed: [],
      updated: [node("a", [], { name: "renamed" })],
      roots: ["a", "b"],
      selection: ["b"],
      dirty: true,
      title: "T",
      can_undo: true,
      can_redo: false,
      undo_label: "Create",
      redo_label: null,
    };
    store.applyDelta(delta);
    const s = useSceneStore.getState();
    expect(s.nodes.b?.guid).toBe("b");
    expect(s.nodes.a?.name).toBe("renamed");
    expect(s.roots).toEqual(["a", "b"]);
    expect(s.selection).toEqual(["b"]);
    expect(s.canUndo).toBe(true);

    // IB-13: a delta with no root list keeps the one the store holds — the
    // backend omits it on every frame that cannot have moved it (a drag, a
    // select), because re-shipping 100 000 strings measured 3.496 ms.
    store.applyDelta({
      ...delta,
      version: 3,
      added: [],
      updated: [node("a", [], { name: "moved" })],
      roots: null,
      selection: ["b"],
    });
    const t = useSceneStore.getState();
    expect(t.roots).toEqual(["a", "b"]);
    expect(t.nodes.a?.name).toBe("moved");
    expect(t.version).toBe(3);

    // …and a delta that DOES carry one replaces it wholesale, including a
    // shrink — an `??` that had been written `||` would keep the old list on an
    // empty one, which is a delete of the last root the Outliner never sees.
    store.applyDelta({
      ...delta,
      version: 4,
      added: [],
      updated: [],
      removed: ["a", "b"],
      roots: [],
      selection: [],
    });
    expect(useSceneStore.getState().roots).toEqual([]);
  });
});
