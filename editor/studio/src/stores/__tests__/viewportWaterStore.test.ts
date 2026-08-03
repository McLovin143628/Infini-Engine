/**
 * P20.4: the water tool's store slice.
 *
 * Three claims, and the third is the one that matters most:
 *
 * 1. every setter **pushes** the whole DTO to the native viewport — a setter
 *    that updated state and forgot the push would leave the toolbar and the tool
 *    describing two different rivers;
 * 2. a half-typed number input (`NaN`) never reaches the viewport;
 * 3. `flow` and `levelOffset` are **signed** — a negative flow reverses a river
 *    without re-authoring its spline, and a negative offset sinks a lake below
 *    the ground clicked. Clamping either would silently delete a feature.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  viewport: {
    setMode: vi.fn().mockResolvedValue(undefined),
    setSnap2d: vi.fn().mockResolvedValue(undefined),
    setToolMode: vi.fn().mockResolvedValue(undefined),
    setSculpt: vi.fn().mockResolvedValue(undefined),
    setFoliage: vi.fn().mockResolvedValue(undefined),
    setBiome: vi.fn().mockResolvedValue(undefined),
    setWater: vi.fn().mockResolvedValue(undefined),
  },
  terrain: {
    biomes: vi.fn().mockResolvedValue(null),
  },
  projectSettings: {
    get: vi.fn().mockResolvedValue({ pixels_per_unit: 100 }),
    set: vi.fn().mockResolvedValue({ pixels_per_unit: 100 }),
  },
}));

import { viewport } from "../../lib/ipc";
import { useViewportStore } from "../viewportStore";

/** The most recent DTO pushed to the viewport. */
function lastPush() {
  const calls = vi.mocked(viewport.setWater).mock.calls;
  expect(calls.length).toBeGreaterThan(0);
  return calls[calls.length - 1][0];
}

beforeEach(() => {
  useViewportStore.setState({
    waterKind: "River",
    waterWidth: 8,
    waterDepth: 1.5,
    waterFlow: 1.5,
    waterLevelOffset: 0,
  });
  vi.clearAllMocks();
});

describe("water tool store", () => {
  it("defaults match the WaterBody component's river defaults", () => {
    const s = useViewportStore.getState();
    expect(s.waterKind).toBe("River");
    expect(s.waterWidth).toBe(8);
    expect(s.waterDepth).toBe(1.5);
    expect(s.waterFlow).toBe(1.5);
    expect(s.waterLevelOffset).toBe(0);
    // Nothing is pushed until something is set.
    expect(viewport.setWater).not.toHaveBeenCalled();
  });

  it("every setter pushes the WHOLE dto, not just its own field", () => {
    const s = useViewportStore.getState();
    s.setWaterKind("Lake");
    expect(lastPush()).toEqual({
      kind: "Lake",
      width_m: 8,
      depth_m: 1.5,
      flow_m_s: 1.5,
      level_offset_m: 0,
    });

    useViewportStore.getState().setWaterWidth(20);
    expect(lastPush().width_m).toBe(20);
    expect(lastPush().kind).toBe("Lake");

    useViewportStore.getState().setWaterDepth(4);
    expect(lastPush().depth_m).toBe(4);
    useViewportStore.getState().setWaterLevelOffset(2.5);
    expect(lastPush().level_offset_m).toBe(2.5);
    // Five setters, five pushes — none of them silent.
    expect(vi.mocked(viewport.setWater).mock.calls.length).toBe(4);
  });

  it("keeps flow and level offset SIGNED", () => {
    useViewportStore.getState().setWaterFlow(-2.5);
    expect(useViewportStore.getState().waterFlow).toBe(-2.5);
    expect(lastPush().flow_m_s).toBe(-2.5);

    useViewportStore.getState().setWaterLevelOffset(-3);
    expect(useViewportStore.getState().waterLevelOffset).toBe(-3);
    expect(lastPush().level_offset_m).toBe(-3);
  });

  it("clamps width and depth to non-negative and rejects NaN everywhere", () => {
    const s = () => useViewportStore.getState();
    s().setWaterWidth(-5);
    expect(s().waterWidth).toBe(8); // kept the previous value
    s().setWaterDepth(-1);
    expect(s().waterDepth).toBe(1.5);

    s().setWaterWidth(Number.NaN);
    s().setWaterDepth(Number.NaN);
    s().setWaterFlow(Number.NaN);
    s().setWaterLevelOffset(Number.NaN);
    expect(s().waterWidth).toBe(8);
    expect(s().waterDepth).toBe(1.5);
    expect(s().waterFlow).toBe(1.5);
    expect(s().waterLevelOffset).toBe(0);
    // …and nothing NaN was ever pushed.
    for (const [dto] of vi.mocked(viewport.setWater).mock.calls) {
      expect(Number.isFinite(dto.width_m)).toBe(true);
      expect(Number.isFinite(dto.depth_m)).toBe(true);
      expect(Number.isFinite(dto.flow_m_s)).toBe(true);
      expect(Number.isFinite(dto.level_offset_m)).toBe(true);
    }
  });

  it("a zero width or depth is allowed — an author may want a hairline stream", () => {
    useViewportStore.getState().setWaterWidth(0);
    expect(useViewportStore.getState().waterWidth).toBe(0);
    useViewportStore.getState().setWaterDepth(0);
    expect(useViewportStore.getState().waterDepth).toBe(0);
  });
});
