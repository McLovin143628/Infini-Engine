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
    // Schema v26 (wave VIS1a). Every value is the record's own default, so this
    // fixture stays what it has always been: a level with nothing authored.
    ssr_enabled: false,
    ssr_distance: 24.0,
    ssr_thickness: 0.15,
    ssr_quality: 1,
    ssr_intensity: 1.0,
    ssr_roughness_cutoff: 0.4,
    exposure_mode: 0,
    exposure_compensation_ev: 0.0,
    exposure_min_luminance: 0.03,
    exposure_max_luminance: 8.0,
    exposure_adaptation_speed: 1.5,
    bloom_karis: false,
    flare_enabled: false,
    flare_intensity: 0.25,
    flare_ghost_count: 4,
    flare_halo: 0.3,
    flare_streak: 0.0,
    vignette_intensity: 0.0,
    vignette_smoothness: 0.0,
    chromatic_aberration: 0.0,
    grain_intensity: 0.0,
    grain_size: 0.0,
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
  weather: {
    present: true,
    enabled: false,
    preset: "clear",
    blend_seconds: 8,
    blend_remaining: 0,
    coverage: 0.08,
    cloud_type: 0.75,
    wind_x: 4,
    wind_z: 1.5,
    fog_density: 0,
    precipitation: 0,
    snowiness: 0,
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

  // **A v26 edit carries the WHOLE block across** (wave VIS1a).
  //
  // The World Settings panel only draws controls for the fields that have a
  // consumer this slice — the SSR block. Everything else v26 added (exposure,
  // flare, the lens trio) is invisible in the UI, and the failure mode that
  // creates is silent: a panel that wrote only what it drew would reset an
  // authored exposure compensation to zero the first time somebody ticked a
  // reflection checkbox. The DTO carries every field for exactly this reason,
  // and this is the arm that says so.
  it("an SSR edit does not reset the v26 fields the panel does not draw", () => {
    const store = useSceneStore.getState();
    const authored: LevelSettingsDto = {
      ...sample,
      render: {
        ...sample.render,
        exposure_mode: 1,
        exposure_compensation_ev: -1.5,
        flare_enabled: true,
        grain_intensity: 0.2,
      },
    };
    store.setLevelSettings(authored);

    // What the panel's `patchRender({ ssr_enabled: true })` produces.
    const patched: LevelSettingsDto = {
      ...authored,
      render: { ...authored.render, ssr_enabled: true },
    };
    store.setLevelSettings(patched);
    vi.advanceTimersByTime(300);

    const sent = vi.mocked(sceneIpc.setSettings).mock.calls.at(-1)?.[0] as LevelSettingsDto;
    expect(sent.render.ssr_enabled).toBe(true);
    expect(sent.render.exposure_mode).toBe(1);
    expect(sent.render.exposure_compensation_ev).toBe(-1.5);
    expect(sent.render.flare_enabled).toBe(true);
    expect(sent.render.grain_intensity).toBe(0.2);
  });
});
