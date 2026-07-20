import { describe, expect, it } from "vitest";
import { findActor, flattenTree, useSceneStore } from "../sceneStore";

describe("sceneStore helpers", () => {
  it("flattenTree walks depth-first with depths", () => {
    const roots = useSceneStore.getState().roots;
    const rows = flattenTree(roots, []);
    expect(rows[0].actor.id).toBe("world");
    expect(rows[0].depth).toBe(0);
    const sun = rows.find((r) => r.actor.id === "sun")!;
    expect(sun.depth).toBe(2);
  });

  it("flattenTree honors collapsed ids", () => {
    const roots = useSceneStore.getState().roots;
    const rows = flattenTree(roots, ["lighting"]);
    expect(rows.some((r) => r.actor.id === "lighting")).toBe(true);
    expect(rows.some((r) => r.actor.id === "sun")).toBe(false);
  });

  it("findActor locates nested actors", () => {
    const roots = useSceneStore.getState().roots;
    expect(findActor(roots, "cube-2")?.name).toBe("Cube2");
    expect(findActor(roots, "nope")).toBeNull();
  });

  it("toggleVisible flips exactly one actor", () => {
    useSceneStore.getState().toggleVisible("sun");
    const roots = useSceneStore.getState().roots;
    expect(findActor(roots, "sun")!.visible).toBe(false);
    expect(findActor(roots, "sky")!.visible).toBe(true);
    useSceneStore.getState().toggleVisible("sun");
  });

  it("select replaces or extends the selection", () => {
    useSceneStore.getState().select(["cube-1"]);
    expect(useSceneStore.getState().selectedIds).toEqual(["cube-1"]);
    useSceneStore.getState().select(["sphere"], true);
    expect(useSceneStore.getState().selectedIds).toEqual(["cube-1", "sphere"]);
    useSceneStore.getState().select([]);
  });
});
