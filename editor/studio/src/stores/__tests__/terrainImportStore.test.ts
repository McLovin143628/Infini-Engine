// @vitest-environment jsdom
//
// Terrain Import wizard (P16.4a): the pure state machine + the readback
// formatters + the pre-flight settings check, and the store actions that drive
// them. IPC is mocked — nothing here decodes a heightmap; the byte-level
// behaviour is Ring 0's (`inf_terrain::chunked`) and the job's is Ring 1's.
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  terrain: {
    probeHeightmap: vi.fn(),
    importPlan: vi.fn(),
    import: vi.fn(),
    cancelImport: vi.fn(),
    assetInfo: vi.fn(),
    spawnStreamed: vi.fn(),
  },
}));

import type { HeightmapProbeDto } from "../../bindings/HeightmapProbeDto";
import type { ImportEventDto } from "../../bindings/ImportEventDto";
import type { TerrainImportSettingsDto } from "../../bindings/TerrainImportSettingsDto";
import { terrain as terrainIpc } from "../../lib/ipc";
import {
  formatBytes,
  formatKm,
  initialMachine,
  progressPercent,
  settingsIssue,
  terrainImportReducer,
  useTerrainImportStore,
  type TerrainImportMachine,
} from "../terrainImportStore";

const SETTINGS: TerrainImportSettingsDto = {
  tile_resolution: 256,
  meters_per_sample: 8,
  min_height: 0,
  max_height: 1000,
  float_meters: false,
  center: true,
  max_pyramid_levels: 8,
  min_pyramid_tiles: 4,
};

const PROBE: HeightmapProbeDto = {
  path: "C:/heights/World.png",
  format: "PNG",
  width: 16385,
  height: 16385,
  bit_depth: 16,
  float_samples: false,
  channel: "gray",
  suggested: SETTINGS,
};

function event(partial: Partial<ImportEventDto>): ImportEventDto {
  return {
    job: 7,
    source: PROBE.path,
    phase: "progress",
    produced: [],
    primary: null,
    cached: false,
    error: null,
    done: null,
    total: null,
    stage: null,
    advisories: [],
    ...partial,
  };
}

function machine(patch: Partial<TerrainImportMachine> = {}): TerrainImportMachine {
  return { ...initialMachine(), job: 7, ...patch };
}

beforeEach(() => {
  vi.clearAllMocks();
  useTerrainImportStore.getState().reset();
});

describe("terrainImportReducer", () => {
  it("ignores events for other jobs and when no job is owned", () => {
    const state = machine();
    expect(terrainImportReducer(state, event({ job: 99 }))).toBe(state);
    const idle = initialMachine();
    expect(terrainImportReducer(idle, event({}))).toBe(idle);
  });

  it("folds progress ticks and keeps the last known totals", () => {
    let s = terrainImportReducer(machine(), event({ phase: "started" }));
    expect(s.step).toBe("importing");
    s = terrainImportReducer(s, event({ done: 64, total: 4225, stage: "tiles" }));
    expect([s.done, s.total, s.stage]).toEqual([64, 4225, "tiles"]);
    // A tick with no total (shouldn't happen, but the reducer is total) keeps
    // the previous one rather than resetting the bar to indeterminate.
    s = terrainImportReducer(s, event({ done: 128, total: null, stage: "lod1" }));
    expect([s.done, s.total, s.stage]).toEqual([128, 4225, "lod1"]);
  });

  it("finishing lands on the done step and releases the job", () => {
    const s = terrainImportReducer(
      machine({ step: "importing", done: 4000, total: 4225 }),
      event({ phase: "finished", primary: "asset-guid" }),
    );
    expect(s.step).toBe("done");
    expect(s.job).toBeNull();
    expect(s.done).toBe(4225);
    expect(s.error).toBeNull();
  });

  // **R2.F7.** L7.H7 stopped the sidecar recording an absolute source path;
  // the ADVISORY half reached the mesh and texture importers and not this one
  // — the door where an outside-the-project source is the norm. `queue.rs`
  // hard-coded `advisories: Vec::new()` under a comment reasoning only about
  // texture advisories, so the channel to the drawer's badge posted empty and
  // the wizard had no surface of its own. The author found out later, from
  // `reimport` refusing with "no import source".
  it("carries a finished import's advisories to the done step", () => {
    const note =
      '"heights.exr" was imported from outside the project, so no source path is recorded';
    const s = terrainImportReducer(
      machine({ step: "importing", done: 4000, total: 4225 }),
      event({ phase: "finished", primary: "asset-guid", advisories: [note] }),
    );
    expect(s.step).toBe("done");
    expect(s.advisories).toEqual([note]);
  });

  it("clears advisories when a new import starts", () => {
    // Otherwise the second import inherits the first one's notice and the
    // author reads a stale warning about a file they did not choose.
    const s = terrainImportReducer(
      machine({ step: "importing", advisories: ["stale"] }),
      event({ phase: "started" }),
    );
    expect(s.advisories).toEqual([]);
  });

  it("a failure surfaces the error, a cancellation steps back to settings", () => {
    const failed = terrainImportReducer(
      machine({ step: "importing" }),
      event({ phase: "failed", error: "png ended after 3 of 9 rows" }),
    );
    expect(failed.step).toBe("importing");
    expect(failed.error).toMatch(/png ended/);
    expect(failed.job).toBeNull();

    const cancelled = terrainImportReducer(
      machine({ step: "importing" }),
      event({ phase: "failed", error: "import: import cancelled" }),
    );
    // The user asked for it: no error banner, and the settings they filled in
    // are still there to retry from.
    expect(cancelled.step).toBe("configure");
    expect(cancelled.error).toBeNull();
    expect(cancelled.job).toBeNull();
  });
});

describe("readbacks", () => {
  it("progressPercent clamps and treats an unknown total as indeterminate", () => {
    expect(progressPercent(0, 100)).toBe(0);
    expect(progressPercent(50, 200)).toBe(25);
    expect(progressPercent(4225, 4225)).toBe(100);
    expect(progressPercent(9999, 100)).toBe(100);
    expect(progressPercent(-5, 100)).toBe(0);
    expect(progressPercent(10, 0)).toBe(0);
    expect(progressPercent(10, Number.NaN)).toBe(0);
  });

  it("formatKm is a DISPLAY conversion of metres and stays honest below 1 km", () => {
    expect(formatKm(0)).toBe("0 m");
    expect(formatKm(132)).toBe("132 m");
    expect(formatKm(1024)).toBe("1.02 km");
    // 16385 samples at 8 m = 131 072 m — the P16 "tens of km" gate.
    expect(formatKm(131072)).toBe("131 km");
    expect(formatKm(Number.NaN)).toBe("—");
  });

  it("formatBytes scales through the binary units", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(1024 * 1024 * 3.5)).toBe("3.5 MB");
    expect(formatBytes(-1)).toBe("—");
  });
});

describe("settingsIssue", () => {
  it("accepts sane settings", () => {
    expect(settingsIssue(SETTINGS, PROBE)).toBeNull();
  });

  it("mirrors the backend's validation rules", () => {
    expect(settingsIssue({ ...SETTINGS, tile_resolution: 1 }, PROBE)).toMatch(/at least 2/);
    expect(settingsIssue({ ...SETTINGS, meters_per_sample: 0 }, PROBE)).toMatch(/positive/);
    expect(settingsIssue({ ...SETTINGS, max_height: 0 }, PROBE)).toMatch(/greater than/);
    expect(settingsIssue({ ...SETTINGS, min_height: Number.NaN }, PROBE)).toMatch(/finite/);
  });

  it("refuses float-metres on an integer source", () => {
    expect(settingsIssue({ ...SETTINGS, float_meters: true }, PROBE)).toMatch(
      /float source/,
    );
    const exr = { ...PROBE, format: "EXR", float_samples: true };
    expect(settingsIssue({ ...SETTINGS, float_meters: true }, exr)).toBeNull();
    // With no probe yet the check cannot know — it must not block.
    expect(settingsIssue({ ...SETTINGS, float_meters: true }, null)).toBeNull();
  });
});

describe("the store's flow", () => {
  it("picking probes the header and opens the settings step with a plan", async () => {
    vi.mocked(terrainIpc.probeHeightmap).mockResolvedValue(PROBE);
    vi.mocked(terrainIpc.importPlan).mockResolvedValue({
      extent_x_m: 131072,
      extent_z_m: 131072,
      tiles_x: 65,
      tiles_z: 65,
      tiles: 4225,
    });

    await useTerrainImportStore.getState().pick(PROBE.path);

    const s = useTerrainImportStore.getState();
    expect(s.step).toBe("configure");
    expect(s.probe).toEqual(PROBE);
    expect(s.settings).toEqual(SETTINGS);
    expect(s.plan?.tiles).toBe(4225);
    expect(terrainIpc.importPlan).toHaveBeenCalledWith(16385, 16385, SETTINGS);
  });

  it("editing settings re-plans without a re-probe", async () => {
    vi.mocked(terrainIpc.probeHeightmap).mockResolvedValue(PROBE);
    vi.mocked(terrainIpc.importPlan).mockResolvedValue({
      extent_x_m: 16384,
      extent_z_m: 16384,
      tiles_x: 65,
      tiles_z: 65,
      tiles: 4225,
    });
    await useTerrainImportStore.getState().pick(PROBE.path);
    vi.mocked(terrainIpc.importPlan).mockClear();

    useTerrainImportStore.getState().patchSettings({ meters_per_sample: 1 });
    await Promise.resolve();
    await Promise.resolve();

    expect(useTerrainImportStore.getState().settings?.meters_per_sample).toBe(1);
    expect(terrainIpc.probeHeightmap).toHaveBeenCalledTimes(1);
    expect(terrainIpc.importPlan).toHaveBeenCalledWith(
      16385,
      16385,
      expect.objectContaining({ meters_per_sample: 1 }),
    );
  });

  it("refuses to queue an import the backend would only reject", async () => {
    vi.mocked(terrainIpc.probeHeightmap).mockResolvedValue(PROBE);
    vi.mocked(terrainIpc.importPlan).mockResolvedValue({
      extent_x_m: 1,
      extent_z_m: 1,
      tiles_x: 1,
      tiles_z: 1,
      tiles: 1,
    });
    await useTerrainImportStore.getState().pick(PROBE.path);
    useTerrainImportStore.getState().patchSettings({ max_height: -10 });

    await useTerrainImportStore.getState().start();

    expect(terrainIpc.import).not.toHaveBeenCalled();
    expect(useTerrainImportStore.getState().error).toMatch(/greater than/);
    expect(useTerrainImportStore.getState().step).toBe("configure");
  });

  it("starting owns a job, and the finished event fetches the asset numbers", async () => {
    vi.mocked(terrainIpc.probeHeightmap).mockResolvedValue(PROBE);
    vi.mocked(terrainIpc.importPlan).mockResolvedValue({
      extent_x_m: 131072,
      extent_z_m: 131072,
      tiles_x: 65,
      tiles_z: 65,
      tiles: 4225,
    });
    vi.mocked(terrainIpc.import).mockResolvedValue(7);
    vi.mocked(terrainIpc.assetInfo).mockResolvedValue({
      asset: "asset-guid",
      name: "World",
      width: 16385,
      height: 16385,
      tiles_x: 65,
      tiles_z: 65,
      tiles: 5619,
      lod_levels: 6,
      extent_x_m: 131072,
      extent_z_m: 131072,
      bytes: 1_100_000_000,
    });

    await useTerrainImportStore.getState().pick(PROBE.path);
    await useTerrainImportStore.getState().start();
    expect(useTerrainImportStore.getState().job).toBe(7);
    expect(useTerrainImportStore.getState().step).toBe("importing");

    // Another job's events must not move this wizard.
    useTerrainImportStore.getState().applyImportEvent(event({ job: 8, phase: "finished" }));
    expect(useTerrainImportStore.getState().step).toBe("importing");

    useTerrainImportStore
      .getState()
      .applyImportEvent(event({ done: 100, total: 5619, stage: "tiles" }));
    expect(progressPercent(...tally())).toBe(2);

    useTerrainImportStore
      .getState()
      .applyImportEvent(event({ phase: "finished", primary: "asset-guid" }));
    await Promise.resolve();
    await Promise.resolve();

    const s = useTerrainImportStore.getState();
    expect(s.step).toBe("done");
    expect(terrainIpc.assetInfo).toHaveBeenCalledWith("asset-guid");
    expect(s.result?.name).toBe("World");
    expect(formatKm(s.result!.extent_x_m)).toBe("131 km");
  });

  it("adding to the scene spawns a streamed terrain and returns its guid", async () => {
    vi.mocked(terrainIpc.spawnStreamed).mockResolvedValue("entity-guid");
    useTerrainImportStore.setState({
      step: "done",
      result: {
        asset: "asset-guid",
        name: "World",
        width: 129,
        height: 129,
        tiles_x: 4,
        tiles_z: 4,
        tiles: 21,
        lod_levels: 2,
        extent_x_m: 1024,
        extent_z_m: 1024,
        bytes: 4096,
      },
    });

    await expect(useTerrainImportStore.getState().addToScene()).resolves.toBe("entity-guid");
    expect(terrainIpc.spawnStreamed).toHaveBeenCalledWith("asset-guid");
    expect(useTerrainImportStore.getState().busy).toBe(false);
  });

  it("cancel asks the backend to stop the owned job", async () => {
    vi.mocked(terrainIpc.cancelImport).mockResolvedValue(true);
    useTerrainImportStore.setState({ step: "importing", job: 7 });
    await useTerrainImportStore.getState().cancel();
    expect(terrainIpc.cancelImport).toHaveBeenCalledWith(7);
  });
});

/** The store's current (done, total) pair, for a percent assertion. */
function tally(): [number, number] {
  const s = useTerrainImportStore.getState();
  return [s.done, s.total];
}
