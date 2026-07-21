import { beforeEach, describe, expect, it, vi } from "vitest";

// The store setters push to the native viewport over typed IPC; stub it so the
// tests exercise state + the push calls without a Tauri backend.
vi.mock("../../lib/ipc", () => ({
  viewport: {
    setMode: vi.fn().mockResolvedValue(undefined),
    setSnap2d: vi.fn().mockResolvedValue(undefined),
    setToolMode: vi.fn().mockResolvedValue(undefined),
    setSculpt: vi.fn().mockResolvedValue(undefined),
  },
  projectSettings: {
    get: vi.fn().mockResolvedValue({ pixels_per_unit: 100 }),
    set: vi.fn().mockResolvedValue({ pixels_per_unit: 100 }),
  },
}));

import { viewport } from "../../lib/ipc";
import {
  clampSculptRadius,
  SCULPT_RADIUS_MAX,
  SCULPT_RADIUS_MIN,
  useViewportStore,
} from "../viewportStore";

beforeEach(() => {
  useViewportStore.setState({
    toolMode: "Select",
    sculptOp: "Raise",
    sculptRadius: 8,
    sculptStrength: 0.5,
    sculptFalloff: "Smooth",
  });
  vi.clearAllMocks();
});

describe("clampSculptRadius", () => {
  it("clamps to the supported range and rejects NaN", () => {
    expect(clampSculptRadius(100)).toBe(100);
    expect(clampSculptRadius(0)).toBe(SCULPT_RADIUS_MIN);
    expect(clampSculptRadius(1e9)).toBe(SCULPT_RADIUS_MAX);
    expect(clampSculptRadius(Number.NaN)).toBe(SCULPT_RADIUS_MIN);
  });
});

describe("viewportStore sculpt actions", () => {
  it("switches tool and pushes the brush config when entering Sculpt", () => {
    useViewportStore.getState().setToolMode("Sculpt");
    expect(useViewportStore.getState().toolMode).toBe("Sculpt");
    expect(viewport.setToolMode).toHaveBeenCalledWith("Sculpt");
    // Entering Sculpt arms the viewport with the current brush.
    expect(viewport.setSculpt).toHaveBeenCalledTimes(1);
    expect(viewport.setSculpt).toHaveBeenCalledWith({
      op: "Raise",
      radius: 8,
      strength: 0.5,
      falloff: "Smooth",
    });
  });

  it("does not push the brush when switching back to Select", () => {
    useViewportStore.getState().setToolMode("Select");
    expect(viewport.setSculpt).not.toHaveBeenCalled();
  });

  it("setSculptOp updates state and pushes", () => {
    useViewportStore.getState().setSculptOp("Lower");
    expect(useViewportStore.getState().sculptOp).toBe("Lower");
    expect(viewport.setSculpt).toHaveBeenCalledWith(
      expect.objectContaining({ op: "Lower" }),
    );
  });

  it("setSculptRadius clamps and pushes", () => {
    useViewportStore.getState().setSculptRadius(0.1);
    expect(useViewportStore.getState().sculptRadius).toBe(SCULPT_RADIUS_MIN);
    expect(viewport.setSculpt).toHaveBeenCalledWith(
      expect.objectContaining({ radius: SCULPT_RADIUS_MIN }),
    );
  });

  it("nudgeSculptRadius grows / shrinks multiplicatively and clamps at bounds", () => {
    const s = useViewportStore.getState();
    s.nudgeSculptRadius(1); // ] → grow
    const grown = useViewportStore.getState().sculptRadius;
    expect(grown).toBeGreaterThan(8);
    useViewportStore.getState().nudgeSculptRadius(-1); // [ → shrink
    expect(useViewportStore.getState().sculptRadius).toBeCloseTo(8, 5);

    // Shrinking hard bottoms out at the minimum, never below.
    for (let i = 0; i < 50; i++) useViewportStore.getState().nudgeSculptRadius(-1);
    expect(useViewportStore.getState().sculptRadius).toBe(SCULPT_RADIUS_MIN);
  });

  it("setSculptStrength clamps negatives to zero", () => {
    useViewportStore.getState().setSculptStrength(-5);
    expect(useViewportStore.getState().sculptStrength).toBe(0);
    useViewportStore.getState().setSculptStrength(2.5);
    expect(useViewportStore.getState().sculptStrength).toBe(2.5);
  });

  it("setSculptFalloff updates state and pushes", () => {
    useViewportStore.getState().setSculptFalloff("Sphere");
    expect(useViewportStore.getState().sculptFalloff).toBe("Sphere");
    expect(viewport.setSculpt).toHaveBeenCalledWith(
      expect.objectContaining({ falloff: "Sphere" }),
    );
  });
});
