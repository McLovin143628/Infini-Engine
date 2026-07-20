import { describe, expect, it } from "vitest";
import {
  clampRegionSize,
  fromDockDoc,
  MAX_DOCK_W_FRAC,
  MIN_DOCK_H,
  MIN_DOCK_W,
  toDockDoc,
  type DockLayout,
} from "../dockTypes";

const layout: DockLayout = {
  panels: {
    outliner: {
      type: "outliner",
      params: null,
      location: { kind: "docked", side: "right", groupId: "g1" },
      floatRect: { x: 10.4, y: 20.6, w: 320, h: 420 },
      floatZ: 0,
      minimized: false,
      lastDock: "right",
      windowRect: null,
    },
    details: {
      type: "details",
      params: null,
      location: { kind: "window" },
      floatRect: { x: 24, y: 24, w: 320, h: 480 },
      floatZ: 1,
      minimized: false,
      lastDock: "right",
      windowRect: { x: 100, y: 120, w: 400, h: 500 },
    },
  },
  docks: {
    left: { size: 300, groups: [] },
    right: {
      size: 320,
      groups: [{ id: "g1", tabs: ["outliner"], activeTab: "outliner", weight: 1 }],
    },
    top: { size: 200, groups: [] },
    bottom: { size: 240, groups: [] },
  },
};

describe("dock doc round-trip", () => {
  it("survives toDockDoc → fromDockDoc structurally", () => {
    const doc = toDockDoc(layout);
    const back = fromDockDoc(doc);
    expect(Object.keys(back.panels).sort()).toEqual(["details", "outliner"]);
    expect(back.panels.outliner.location).toEqual(layout.panels.outliner.location);
    expect(back.panels.details.windowRect).toEqual(layout.panels.details.windowRect);
    expect(back.docks.right.groups).toEqual(layout.docks.right.groups);
    // Rects are rounded at the persistence edge.
    expect(back.panels.outliner.floatRect.x).toBe(10);
    expect(back.panels.outliner.floatRect.y).toBe(21);
  });

  it("JSON-serializes cleanly (the layouts IPC payload)", () => {
    const doc = toDockDoc(layout);
    const parsed = fromDockDoc(JSON.parse(JSON.stringify(doc)));
    expect(parsed.docks.right.groups[0].activeTab).toBe("outliner");
  });
});

describe("clampRegionSize", () => {
  const ws = { w: 2000, h: 1000 };
  it("enforces minimums", () => {
    expect(clampRegionSize("left", 10, ws)).toBe(MIN_DOCK_W);
    expect(clampRegionSize("bottom", 10, ws)).toBe(MIN_DOCK_H);
  });
  it("caps sidebars at the width fraction", () => {
    expect(clampRegionSize("right", 5000, ws)).toBe(Math.round(ws.w * MAX_DOCK_W_FRAC));
  });
});
