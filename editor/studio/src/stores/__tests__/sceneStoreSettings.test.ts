import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { LevelSettingsDto } from "../../bindings/LevelSettingsDto";

// Mock the IPC layer so `setLevelSettings`'s debounced write is observable and
// no real Tauri `invoke` is attempted in the node test env.
vi.mock("../../lib/ipc", () => ({
  scene: {
    getSettings: vi.fn(),
    setSettings: vi.fn().mockResolvedValue(undefined),
  },
}));

import { scene as sceneIpc } from "../../lib/ipc";
import { useSceneStore } from "../sceneStore";

const sample: LevelSettingsDto = {
  gravity_2d: [0, 0],
  gravity_3d: [0, -9.81, 0],
  sim_hz: 60,
  render: {
    exposure: 1.0,
    dither: true,
    bloom_enabled: false,
    bloom_threshold: 1.0,
    bloom_knee: 0.5,
    bloom_intensity: 0.06,
    ssao_enabled: false,
    ssao_radius: 0.6,
    ssao_intensity: 1.0,
    ssao_bias: 0.025,
    taa: false,
    shadows_enabled: false,
    shadows_max_distance: 60.0,
    gi_enabled: false,
    gi_intensity: 1.0,
  },
  partition: {
    enabled: false,
    cell_size_m: 256,
    activation_radius_m: 256,
    prefetch_margin_m: 256,
  },
};

describe("sceneStore level settings slice", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.clearAllMocks();
  });
  afterEach(() => vi.useRealTimers());

  it("optimistically updates state and coalesces rapid edits into one debounced write", () => {
    const store = useSceneStore.getState();

    store.setLevelSettings(sample);
    // Optimistic: the panel reflects the value synchronously.
    expect(useSceneStore.getState().levelSettings).toEqual(sample);
    // Nothing written yet — the write is debounced.
    expect(sceneIpc.setSettings).not.toHaveBeenCalled();

    // A rapid second edit resets the debounce; only the latest survives.
    const edited: LevelSettingsDto = { ...sample, render: { ...sample.render, exposure: 2.5 } };
    store.setLevelSettings(edited);
    expect(useSceneStore.getState().levelSettings).toEqual(edited);

    vi.advanceTimersByTime(300);
    expect(sceneIpc.setSettings).toHaveBeenCalledTimes(1);
    expect(sceneIpc.setSettings).toHaveBeenCalledWith(edited);
  });
});
