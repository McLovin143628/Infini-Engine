import { describe, expect, it } from "vitest";
import { clampPanelRect, MIN_VISIBLE_PX, resizePanelRect } from "../panelRect";

const container = { w: 1000, h: 800 };
const min = { w: 200, h: 150 };

describe("resizePanelRect", () => {
  const start = { x: 100, y: 100, w: 300, h: 200 };

  it("east drag grows width, anchored left", () => {
    const r = resizePanelRect(start, "e", 50, 999, min, container);
    expect(r).toEqual({ x: 100, y: 100, w: 350, h: 200 });
  });

  it("north-west drag moves origin and shrinks, clamped to min size", () => {
    const r = resizePanelRect(start, "nw", 500, 500, min, container);
    expect(r.w).toBe(min.w);
    expect(r.h).toBe(min.h);
    expect(r.x + r.w).toBe(start.x + start.w);
    expect(r.y + r.h).toBe(start.y + start.h);
  });

  it("never exceeds the container", () => {
    const r = resizePanelRect(start, "se", 5000, 5000, min, container);
    expect(r.x + r.w).toBeLessThanOrEqual(container.w);
    expect(r.y + r.h).toBeLessThanOrEqual(container.h);
  });
});

describe("clampPanelRect", () => {
  it("keeps a header sliver reachable on the right edge", () => {
    const r = clampPanelRect({ x: 5000, y: 100, w: 300, h: 200 }, container);
    expect(r.x).toBe(container.w - MIN_VISIBLE_PX);
  });

  it("keeps the header row inside vertically", () => {
    const r = clampPanelRect({ x: 100, y: -50, w: 300, h: 200 }, container);
    expect(r.y).toBe(0);
    const r2 = clampPanelRect({ x: 100, y: 5000, w: 300, h: 200 }, container);
    expect(r2.y).toBeLessThanOrEqual(container.h);
  });

  it("leaves size untouched", () => {
    const r = clampPanelRect({ x: -900, y: 0, w: 300, h: 200 }, container);
    expect(r.w).toBe(300);
    expect(r.h).toBe(200);
  });
});
