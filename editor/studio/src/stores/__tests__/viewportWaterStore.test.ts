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

import type { RiverReportDto } from "../../bindings/RiverReportDto";

vi.mock("../../lib/ipc", () => ({
  water: {
    riverReport: vi.fn(),
  },
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

import { terrain, water, viewport } from "../../lib/ipc";
import { riverIssues } from "../../viewport/ViewportToolbar";
import { useSceneStore } from "../sceneStore";
import { useViewportStore } from "../viewportStore";

/** A clean report over a fully-sampled river. */
function report(over: Partial<RiverReportDto> = {}): RiverReportDto {
  return {
    entity: "aaaaaaaa-0000-4000-8000-000000000001",
    length_m: 210,
    points: 5,
    fall_m: 32,
    surface_climbs: [],
    bed_climbs: [],
    bed_conflicts: [],
    sampled_frames: 49,
    total_frames: 49,
    ...over,
  };
}

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
    waterRiverReport: null,
    toolMode: "Select",
  });
  useSceneStore.setState({ selection: [] });
  vi.clearAllMocks();
  vi.mocked(water.riverReport).mockResolvedValue(report());
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

describe("the river verdict", () => {
  it("is null with nothing selected, and holds a real river's report", async () => {
    await useViewportStore.getState().refreshRiverReport();
    expect(useViewportStore.getState().waterRiverReport).toBeNull();
    expect(water.riverReport).not.toHaveBeenCalled();

    useSceneStore.setState({ selection: ["aaaaaaaa-0000-4000-8000-000000000001"] });
    await useViewportStore.getState().refreshRiverReport();
    expect(water.riverReport).toHaveBeenCalledWith("aaaaaaaa-0000-4000-8000-000000000001");
    expect(useViewportStore.getState().waterRiverReport?.length_m).toBe(210);
  });

  it("treats a frameless report as 'not a river' rather than as a clean bill", async () => {
    useSceneStore.setState({ selection: ["bbbbbbbb-0000-4000-8000-000000000001"] });
    vi.mocked(water.riverReport).mockResolvedValue(
      report({ total_frames: 0, points: 0, length_m: 0 }),
    );
    await useViewportStore.getState().refreshRiverReport();
    expect(useViewportStore.getState().waterRiverReport).toBeNull();
  });

  it("survives a failing command without wedging on a stale verdict", async () => {
    useSceneStore.setState({ selection: ["cccccccc-0000-4000-8000-000000000001"] });
    await useViewportStore.getState().refreshRiverReport();
    expect(useViewportStore.getState().waterRiverReport).not.toBeNull();
    vi.mocked(water.riverReport).mockRejectedValue(new Error("no such entity"));
    await useViewportStore.getState().refreshRiverReport();
    expect(useViewportStore.getState().waterRiverReport).toBeNull();
  });

  /**
   * THE ARMING ARM (P20.4 audit, Blocker 2). Entering the Water tool must
   * re-read the biome vocabulary — that is the call that makes the backend push
   * the id-indexed water-level hints — or a fresh session that goes straight to
   * Water commits a DIFFERENT level from one that visited Biome first.
   */
  it("arms the hint table and the verdict on entering the tool", () => {
    useViewportStore.getState().setToolMode("Water");
    expect(viewport.setToolMode).toHaveBeenCalledWith("Water");
    expect(viewport.setWater).toHaveBeenCalled();
    expect(terrain.biomes).toHaveBeenCalled();
  });

  it("says nothing about a clean river and names the worst problem otherwise", () => {
    expect(riverIssues(report())).toEqual([]);

    const dirty = report({
      bed_conflicts: [
        { issue: "buried", from_s: 10, to_s: 50, worst_m: 3.1, worst_x: 1, worst_z: 2 },
        { issue: "buried", from_s: 80, to_s: 90, worst_m: 0.9, worst_x: 3, worst_z: 4 },
        { issue: "perched", from_s: 120, to_s: 140, worst_m: 12, worst_x: 5, worst_z: 6 },
      ],
      surface_climbs: [{ from_s: 0, to_s: 20, rise_m: 2.5, gradient: 0.125 }],
      bed_climbs: [{ from_s: 0, to_s: 20, rise_m: 1.5, gradient: 0.075 }],
    });
    const issues = riverIssues(dirty);
    expect(issues).toHaveLength(4);
    // The GROUND problems come first — they are what an author fixes first.
    expect(issues[0]).toContain("buried");
    // …and each names the WORST span, not the first one found.
    expect(issues[0]).toContain("3.1");
    expect(issues[1]).toContain("perched");
    expect(issues[2]).toContain("UPHILL");
    expect(issues[3]).toContain("BED");
    // The two cook advisories say so, because the tool is quoting the build.
    expect(issues[2]).toContain("the cook will say so");
    expect(issues[3]).toContain("the cook will say so");
  });
});
