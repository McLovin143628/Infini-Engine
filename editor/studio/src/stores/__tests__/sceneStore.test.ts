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
  });
});
