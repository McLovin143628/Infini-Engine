import { describe, expect, it } from "vitest";
import { resolveDropTarget, type DropZones } from "../dropTargets";

/** Synthetic zone snapshot: 1600×900 workspace, one right-sidebar group
 *  with two tabs, center column between x=0 and x=1280. */
const zones: DropZones = {
  workspace: { x: 0, y: 0, w: 1600, h: 900 },
  center: { x: 0, y: 0, w: 1280, h: 900 },
  groups: [
    {
      side: "right",
      groupId: "g1",
      groupIndex: 0,
      stripRect: { x: 1280, y: 0, w: 320, h: 32 },
      bodyRect: { x: 1280, y: 0, w: 320, h: 450 },
      tabs: [
        { panelId: "outliner", rect: { x: 1284, y: 4, w: 90, h: 26 } },
        { panelId: "details", rect: { x: 1378, y: 4, w: 90, h: 26 } },
      ],
    },
  ],
  regionSizes: { left: 300, right: 320, top: 200, bottom: 240 },
  regionGroupCounts: { left: 0, right: 1, top: 0, bottom: 0 },
};

describe("resolveDropTarget", () => {
  it("outside the workspace → null (float)", () => {
    expect(resolveDropTarget(zones, -10, 50, "x")).toBeNull();
    expect(resolveDropTarget(zones, 200, 950, "x")).toBeNull();
  });

  it("tab strip hit inserts at the caret index, skipping the dragged tab", () => {
    // Pointer right of the second tab's midpoint → index 2 for a foreign panel…
    const t = resolveDropTarget(zones, 1450, 16, "other")!;
    expect(t.target).toEqual({ kind: "dock", side: "right", groupId: "g1", index: 2 });
    expect(t.tabCaret).toBeTruthy();
    // …but index 1 when the dragged panel is one of the two tabs.
    const t2 = resolveDropTarget(zones, 1450, 16, "details")!;
    expect(t2.target).toEqual({ kind: "dock", side: "right", groupId: "g1", index: 1 });
  });

  it("group body: top quarter splits before, bottom quarter after, middle joins", () => {
    const before = resolveDropTarget(zones, 1400, 60, "x")!; // 60/450 < 0.25
    expect(before.target).toEqual({ kind: "new-group", side: "right", index: 0 });
    const after = resolveDropTarget(zones, 1400, 430, "x")!; // > 0.75
    expect(after.target).toEqual({ kind: "new-group", side: "right", index: 1 });
    const join = resolveDropTarget(zones, 1400, 225, "x")!;
    expect(join.target).toEqual({
      kind: "dock",
      side: "right",
      groupId: "g1",
      index: Number.MAX_SAFE_INTEGER,
    });
  });

  it("workspace edge bands dock as a new group; previews match region sizes", () => {
    const left = resolveDropTarget(zones, 20, 450, "x")!;
    expect(left.target).toEqual({ kind: "new-group", side: "left", index: 0 });
    expect(left.preview).toEqual({ x: 0, y: 0, w: 300, h: 900 });

    const bottom = resolveDropTarget(zones, 640, 880, "x")!;
    expect(bottom.target).toEqual({ kind: "new-group", side: "bottom", index: 0 });
    // Bottom band insets to the center column, not the full workspace width.
    expect(bottom.preview).toEqual({ x: 0, y: 900 - 240, w: 1280, h: 240 });
  });

  it("center area with no band/group hit → null (float)", () => {
    expect(resolveDropTarget(zones, 640, 450, "x")).toBeNull();
  });
});
