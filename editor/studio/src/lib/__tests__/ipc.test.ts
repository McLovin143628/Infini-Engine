// @vitest-environment jsdom
//
// Locks the typed IPC wrappers to the backend command names and payload
// shapes they were written against (Tauri's mockIPC intercepts `invoke`).
// The payload shapes themselves are the ts-rs generated bindings, so a
// backend type change shows up here as a compile error before it ships.
import { clearMocks, mockIPC } from "@tauri-apps/api/mocks";
import { afterEach, expect, test } from "vitest";

import { app, assets, layouts, pcg, sim, terrain, viewport } from "../ipc";
import type { BiomeSetDto } from "../ipc";

afterEach(() => {
  clearMocks();
});

test("every wrapper invokes its registered command with the typed payload", async () => {
  const calls: Array<[string, unknown]> = [];
  mockIPC((cmd, args) => {
    calls.push([cmd, args]);
    if (cmd === "app_version") return "0.1.0";
  });

  await expect(app.version()).resolves.toBe("0.1.0");
  await viewport.attach();
  await viewport.setRect({ x: 1, y: 2, width: 3, height: 4 });
  await viewport.drop({ x: 5, y: 6, payload: "TestActor" });

  expect(calls.map(([cmd]) => cmd)).toEqual([
    "app_version",
    "viewport_attach",
    "viewport_set_rect",
    "viewport_drop",
  ]);
  expect(calls[2][1]).toEqual({ rect: { x: 1, y: 2, width: 3, height: 4 } });
  expect(calls[3][1]).toEqual({ drop: { x: 5, y: 6, payload: "TestActor" } });
});

test("layout wrappers invoke the layout_* commands with typed payloads", async () => {
  const calls: Array<[string, unknown]> = [];
  mockIPC((cmd, args) => {
    calls.push([cmd, args]);
    if (cmd === "layout_load") return '{"v":1}';
    if (cmd === "layout_list") return [{ name: "Default", modified_ms: 0 }];
    if (cmd === "layout_delete") return true;
  });

  await layouts.save("Default", '{"v":1}');
  await expect(layouts.load("Default")).resolves.toBe('{"v":1}');
  await expect(layouts.list()).resolves.toEqual([{ name: "Default", modified_ms: 0 }]);
  await expect(layouts.delete("Default")).resolves.toBe(true);

  expect(calls.map(([cmd]) => cmd)).toEqual([
    "layout_save",
    "layout_load",
    "layout_list",
    "layout_delete",
  ]);
  expect(calls[0][1]).toEqual({ name: "Default", json: '{"v":1}' });
});

test("sim wrappers invoke the sim_* commands with typed payloads", async () => {
  const calls: Array<[string, unknown]> = [];
  mockIPC((cmd, args) => {
    calls.push([cmd, args]);
    if (cmd === "sim_tick") return true;
    if (cmd === "sim_is_running") return false;
  });

  await sim.start();
  await expect(sim.tick(["left", "jump"])).resolves.toBe(true);
  await sim.stop();
  await expect(sim.isRunning()).resolves.toBe(false);

  expect(calls.map(([cmd]) => cmd)).toEqual([
    "sim_start",
    "sim_tick",
    "sim_stop",
    "sim_is_running",
  ]);
  expect(calls[1][1]).toEqual({ keys: ["left", "jump"] });
});

test("pcg wrappers invoke the pcg_* commands with typed payloads", async () => {
  const calls: Array<[string, unknown]> = [];
  mockIPC((cmd, args) => {
    calls.push([cmd, args]);
    if (cmd === "pcg_compile")
      return { ok: true, issues: [], ruleCount: 1, summary: "1 rule(s)" };
    if (cmd === "pcg_evaluate")
      return { entity: "abc", placed: 7, issues: [], ok: true };
    if (cmd === "pcg_save") return "PCG_Graph.inf_pcg";
  });

  await pcg.apply("pcg:1", [], "Edit");
  await expect(pcg.compile("pcg:1")).resolves.toMatchObject({ ok: true, ruleCount: 1 });
  await expect(pcg.evaluate("pcg:1", null)).resolves.toMatchObject({ placed: 7 });
  await expect(pcg.save("pcg:1", "PCG Graph")).resolves.toBe("PCG_Graph.inf_pcg");

  expect(calls.map(([cmd]) => cmd)).toEqual([
    "pcg_apply",
    "pcg_compile",
    "pcg_evaluate",
    "pcg_save",
  ]);
  expect(calls[0][1]).toEqual({ id: "pcg:1", edits: [], label: "Edit" });
  expect(calls[2][1]).toEqual({ id: "pcg:1", entity: null });
  expect(calls[3][1]).toEqual({ id: "pcg:1", name: "PCG Graph" });
});

test("biome wrappers invoke the P19.2 commands with typed payloads", async () => {
  const calls: Array<[string, unknown]> = [];
  const set: BiomeSetDto = {
    id: "bs-1",
    name: "World Biomes",
    biomes: [
      {
        id: 1,
        name: "Grassland",
        color: [0.29, 0.45, 0.19, 1],
        splat_layer: 0,
        pcg_graph: null,
        water_hint: null,
        structure_hint: null,
      },
    ],
  };
  mockIPC((cmd, args) => {
    calls.push([cmd, args]);
    if (cmd === "terrain_biomes")
      return {
        entity: "e-1",
        biome_set: "bs-1",
        biome_set_name: "World Biomes",
        biomes: set.biomes,
        available: [["bs-1", "World Biomes"]],
      };
    if (cmd === "terrain_set_biome_set") return true;
    if (cmd === "asset_get_biome_set") return set;
  });

  await viewport.setBiome({ radius: 8, strength: 1, falloff: "Smooth", biome: 3 });
  await expect(terrain.biomes()).resolves.toMatchObject({ biome_set: "bs-1" });
  await terrain.biomes("e-1");
  await expect(terrain.setBiomeSet("e-1", null)).resolves.toBe(true);
  await expect(assets.getBiomeSet("bs-1")).resolves.toEqual(set);
  await assets.saveBiomeSet("bs-1", set);

  expect(calls.map(([cmd]) => cmd)).toEqual([
    "viewport_set_biome",
    "terrain_biomes",
    "terrain_biomes",
    "terrain_set_biome_set",
    "asset_get_biome_set",
    "asset_save_biome_set",
  ]);
  expect(calls[0][1]).toEqual({
    biome: { radius: 8, strength: 1, falloff: "Smooth", biome: 3 },
  });
  // An omitted entity must cross as an explicit null (the backend's `Option`),
  // never as `undefined` — Tauri would drop the key entirely.
  expect(calls[1][1]).toEqual({ entity: null });
  expect(calls[2][1]).toEqual({ entity: "e-1" });
  expect(calls[3][1]).toEqual({ entity: "e-1", asset: null });
  expect(calls[4][1]).toEqual({ id: "bs-1" });
  expect(calls[5][1]).toEqual({ id: "bs-1", set });
});
