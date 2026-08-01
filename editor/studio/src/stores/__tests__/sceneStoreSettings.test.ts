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
  time_of_day: {
    present: false,
    seconds: 36000,
    day_of_year: 172,
    latitude_deg: 48.9,
    longitude_deg: 0,
    rate: 0,
    sun_elevation_deg: 55.1,
    sun_azimuth_deg: 125.4,
  },
  atmosphere: {
    present: false,
    enabled: true,
    physical: true,
    sky_intensity: 1,
    turbidity: 1,
    mie_anisotropy: 0.8,
    sun_disc_deg: 0.545,
    moon_disc_deg: 0.52,
    star_intensity: 1,
    tint_strength: 0,
    aerial_perspective: 1,
    fog_density: 0,
    fog_falloff: 0.002,
    fog_height: 0,
    clouds_enabled: false,
    cloud_coverage: 0.35,
    cloud_type: 0.7,
    cloud_bottom: 1500,
    cloud_top: 4000,
    cloud_density: 0.04,
    cloud_detail: 0.6,
    cloud_seed: 0,
    cloud_wind_x: 6,
    cloud_wind_z: 2,
    cloud_phase_g: 0.8,
    cloud_shadow: 1,
    cloud_ambient: 1,
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
