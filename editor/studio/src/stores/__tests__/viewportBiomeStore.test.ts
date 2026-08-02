import { beforeEach, describe, expect, it, vi } from "vitest";

import type { BiomeDefDto } from "../../bindings/BiomeDefDto";
import type { TerrainBiomesDto } from "../../bindings/TerrainBiomesDto";

// The store setters push to the native viewport over typed IPC; stub it so the
// tests exercise state + the push calls without a Tauri backend.
vi.mock("../../lib/ipc", () => ({
  viewport: {
    setMode: vi.fn().mockResolvedValue(undefined),
    setSnap2d: vi.fn().mockResolvedValue(undefined),
    setToolMode: vi.fn().mockResolvedValue(undefined),
    setSculpt: vi.fn().mockResolvedValue(undefined),
    setFoliage: vi.fn().mockResolvedValue(undefined),
    setBiome: vi.fn().mockResolvedValue(undefined),
  },
  terrain: {
    biomes: vi.fn().mockResolvedValue(null),
  },
  projectSettings: {
    get: vi.fn().mockResolvedValue({ pixels_per_unit: 100 }),
    set: vi.fn().mockResolvedValue({ pixels_per_unit: 100 }),
  },
}));

import { terrain, viewport } from "../../lib/ipc";
import { SCULPT_RADIUS_MAX, SCULPT_RADIUS_MIN, useViewportStore } from "../viewportStore";

function def(id: number, name = `b${id}`): BiomeDefDto {
  return {
    id,
    name,
    color: [0.5, 0.5, 0.5, 1],
    splat_layer: null,
    pcg_graph: null,
    water_hint: null,
    structure_hint: null,
  };
}

function vocabulary(biomes: BiomeDefDto[]): TerrainBiomesDto {
  return {
    entity: "11111111-1111-1111-1111-111111111111",
    biome_set: biomes.length ? "22222222-2222-2222-2222-222222222222" : null,
    biome_set_name: biomes.length ? "World Biomes" : "",
    biomes,
    available: [],
  };
}

beforeEach(() => {
  useViewportStore.setState({
    toolMode: "Select",
    biomeRadius: 8,
    biomeStrength: 1,
    biomeFalloff: "Smooth",
    biomeId: 0,
    terrainBiomes: null,
  });
  vi.clearAllMocks();
  vi.mocked(terrain.biomes).mockResolvedValue(null);
});

describe("viewportStore biome brush", () => {
  it("arms the brush and re-reads the vocabulary when entering the Biome tool", async () => {
    useViewportStore.getState().setToolMode("Biome");
    expect(useViewportStore.getState().toolMode).toBe("Biome");
    expect(viewport.setToolMode).toHaveBeenCalledWith("Biome");
    // The defaults are a full-strength stamp of the eraser id, which is what the
    // viewport must be holding before the first drag.
    expect(viewport.setBiome).toHaveBeenCalledTimes(1);
    expect(viewport.setBiome).toHaveBeenCalledWith({
      radius: 8,
      strength: 1,
      falloff: "Smooth",
      biome: 0,
    });
    expect(terrain.biomes).toHaveBeenCalled();
    await vi.waitFor(() => expect(useViewportStore.getState().terrainBiomes).toBeNull());
  });

  it("does not arm the biome brush when switching to another tool", () => {
    useViewportStore.getState().setToolMode("Select");
    expect(viewport.setBiome).not.toHaveBeenCalled();
    expect(terrain.biomes).not.toHaveBeenCalled();
  });

  it("setBiomeId clamps to 0..255 and pushes the exact id", () => {
    useViewportStore.getState().setBiomeId(7);
    expect(useViewportStore.getState().biomeId).toBe(7);
    expect(viewport.setBiome).toHaveBeenCalledWith(expect.objectContaining({ biome: 7 }));

    useViewportStore.getState().setBiomeId(999);
    expect(useViewportStore.getState().biomeId).toBe(255);
    useViewportStore.getState().setBiomeId(-4);
    expect(useViewportStore.getState().biomeId).toBe(0);
    useViewportStore.getState().setBiomeId(Number.NaN);
    expect(useViewportStore.getState().biomeId).toBe(0);
  });

  it("setBiomeStrength clamps to [0,1] — it selects a contour, not a rate", () => {
    useViewportStore.getState().setBiomeStrength(0.25);
    expect(useViewportStore.getState().biomeStrength).toBe(0.25);
    expect(viewport.setBiome).toHaveBeenCalledWith(expect.objectContaining({ strength: 0.25 }));

    useViewportStore.getState().setBiomeStrength(4);
    expect(useViewportStore.getState().biomeStrength).toBe(1);
    useViewportStore.getState().setBiomeStrength(-1);
    expect(useViewportStore.getState().biomeStrength).toBe(0);
    useViewportStore.getState().setBiomeStrength(Number.NaN);
    expect(useViewportStore.getState().biomeStrength).toBe(1);
  });

  it("setBiomeRadius shares the sculpt clamp and pushes", () => {
    useViewportStore.getState().setBiomeRadius(0.1);
    expect(useViewportStore.getState().biomeRadius).toBe(SCULPT_RADIUS_MIN);
    expect(viewport.setBiome).toHaveBeenCalledWith(
      expect.objectContaining({ radius: SCULPT_RADIUS_MIN }),
    );
    useViewportStore.getState().setBiomeRadius(1e9);
    expect(useViewportStore.getState().biomeRadius).toBe(SCULPT_RADIUS_MAX);
  });

  it("setBiomeFalloff updates state and pushes", () => {
    useViewportStore.getState().setBiomeFalloff("Sharp");
    expect(useViewportStore.getState().biomeFalloff).toBe("Sharp");
    expect(viewport.setBiome).toHaveBeenCalledWith(expect.objectContaining({ falloff: "Sharp" }));
  });
});

describe("viewportStore refreshBiomes", () => {
  it("stores the vocabulary and keeps a still-defined selection", async () => {
    useViewportStore.setState({ biomeId: 2 });
    const dto = vocabulary([def(1, "Grassland"), def(2, "Rock")]);
    vi.mocked(terrain.biomes).mockResolvedValue(dto);

    await useViewportStore.getState().refreshBiomes();
    expect(useViewportStore.getState().terrainBiomes).toEqual(dto);
    expect(useViewportStore.getState().biomeId).toBe(2);
    // Nothing changed, so nothing was re-pushed.
    expect(viewport.setBiome).not.toHaveBeenCalled();
  });

  it("falls back to the first defined id when the selection disappears", async () => {
    useViewportStore.setState({ biomeId: 9 });
    vi.mocked(terrain.biomes).mockResolvedValue(vocabulary([def(4, "Marsh"), def(5, "Dune")]));

    await useViewportStore.getState().refreshBiomes();
    expect(useViewportStore.getState().biomeId).toBe(4);
    // A repaired selection MUST reach the viewport, or the next drag paints the
    // id the toolbar no longer shows.
    expect(viewport.setBiome).toHaveBeenCalledWith(expect.objectContaining({ biome: 4 }));
  });

  it("falls back to 0 (the eraser) when the set defines nothing", async () => {
    useViewportStore.setState({ biomeId: 3 });
    vi.mocked(terrain.biomes).mockResolvedValue(vocabulary([]));

    await useViewportStore.getState().refreshBiomes();
    expect(useViewportStore.getState().biomeId).toBe(0);
  });

  it("keeps the eraser selected — id 0 is always valid", async () => {
    useViewportStore.setState({ biomeId: 0 });
    vi.mocked(terrain.biomes).mockResolvedValue(vocabulary([def(1)]));

    await useViewportStore.getState().refreshBiomes();
    expect(useViewportStore.getState().biomeId).toBe(0);
  });

  it("swallows a failed read and leaves the selection alone", async () => {
    useViewportStore.setState({ biomeId: 6, terrainBiomes: vocabulary([def(6)]) });
    vi.mocked(terrain.biomes).mockRejectedValue(new Error("no project open"));

    await expect(useViewportStore.getState().refreshBiomes()).resolves.toBeUndefined();
    expect(useViewportStore.getState().biomeId).toBe(6);
  });
});
