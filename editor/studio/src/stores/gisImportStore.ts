/**
 * GIS Import wizard state (IB-3).
 *
 * A three-step machine — pick a file, choose what it becomes, see what it
 * produced — over the one import door. Every *decision* is in Ring 0
 * (`inf_gis::import`); this store holds the author's answers and the report.
 *
 * The pure functions below (`gisSettingsIssue`, `capNote`, `describeCrs`,
 * `kindsFor`) are the whole of the wizard's judgement and are unit-tested
 * without a backend, following `terrainImportStore`'s precedent.
 *
 * UNITS: every length that crosses IPC is metres (the units doctrine). Nothing
 * here scales anything.
 */
import { create } from "zustand";

import type { GisImportResultDto } from "../bindings/GisImportResultDto";
import type { GisImportSettingsDto } from "../bindings/GisImportSettingsDto";
import type { GisProbeDto } from "../bindings/GisProbeDto";
import { gis as gisIpc } from "../lib/ipc";
import { useSettingsStore } from "./settingsStore";

/** Where the wizard is. */
export type GisImportStep = "pick" | "configure" | "done";

/**
 * The default entity cap, mirrored from `inf_gis::DEFAULT_MAX_ENTITIES`.
 *
 * A mirror, and it is allowed to be one because the backend NEVER reads it: the
 * cap that runs is the one in the settings the wizard sends, and the settings
 * the wizard opens on come from `gis_suggested_settings`. This constant only
 * decides what the field shows before the first probe answers.
 */
export const GIS_DEFAULT_MAX_ENTITIES = 4096;
/** Mirrored from `inf_gis::ISLAND_MAX_ENTITIES` — the ceiling the field allows. */
export const GIS_ISLAND_MAX_ENTITIES = 262144;

/** The layer kinds the door offers, in the order the wizard lists them. */
export const GIS_LAYER_KINDS = [
  "generic",
  "roads",
  "streams",
  "lakes",
  "biomes",
  "buildings",
  "parcels",
] as const;
export type GisLayerKind = (typeof GIS_LAYER_KINDS)[number];

/** Which kinds make sense for a layer's dominant geometry. */
export function kindsFor(dominant: string): GisLayerKind[] {
  if (dominant === "polygon") return ["buildings", "parcels", "biomes", "lakes", "generic"];
  if (dominant === "polyline") return ["roads", "streams", "generic", "parcels", "lakes"];
  return ["generic"];
}

/**
 * What is wrong with these settings, or `null`.
 *
 * The wizard refuses to send a request the door would refuse, and says the same
 * sentence the door would — an author should not have to press Import to find
 * out that the level has no anchor.
 */
export function gisSettingsIssue(
  probe: GisProbeDto | null,
  s: GisImportSettingsDto | null,
): string | null {
  if (!probe || !s) return "pick a vector file first";
  if (!probe.level_anchor_crs) {
    return "this level has no geo-anchor, so there is no answer to where its origin is on Earth. Set one in World Settings first — a GIS import has to be transformed into something.";
  }
  if (probe.features === 0) return "this file contains no features";
  if (!Number.isFinite(s.max_entities) || s.max_entities < 1) {
    return "the entity cap must be at least 1";
  }
  if (s.max_entities > GIS_ISLAND_MAX_ENTITIES) {
    return `the entity cap tops out at ${GIS_ISLAND_MAX_ENTITIES}`;
  }
  if (!Number.isFinite(s.min_length_m) || s.min_length_m < 0) {
    return "the minimum feature length must be zero or more metres";
  }
  if (s.road_surface && s.kind !== "roads") {
    return "a road surface needs the layer imported as roads";
  }
  if (s.buildings && s.kind !== "buildings" && s.kind !== "parcels") {
    return "buildings are built from footprint or parcel polygons";
  }
  if (s.biome_terrain.trim() !== "" && s.kind !== "biomes") {
    return "land cover is painted from a layer imported as biomes";
  }
  return null;
}

/**
 * What the cap will do to this layer — shown BEFORE the import, which is the
 * half IB-14 says was missing.
 */
export function capNote(probe: GisProbeDto | null, s: GisImportSettingsDto | null): string | null {
  if (!probe || !s || !Number.isFinite(s.max_entities)) return null;
  if (probe.features <= s.max_entities) return null;
  const lost = probe.features - s.max_entities;
  return `${lost} of ${probe.features} features will NOT be imported at a cap of ${s.max_entities}. Raise it to ${probe.features} to take the whole layer.`;
}

/** One line describing where a source's CRS came from. */
export function describeCrs(probe: GisProbeDto | null): string {
  if (!probe) return "";
  const { spec, origin, name } = probe.crs;
  const from =
    origin === "prj"
      ? "from its .prj"
      : origin === "prj-name"
        ? "GUESSED from the .prj's projection name"
        : "as stated";
  return name ? `${spec} (${from} — ${name})` : `${spec} (${from})`;
}

/** The part of the wizard a test needs. */
export interface GisImportMachine {
  step: GisImportStep;
  path: string;
  probe: GisProbeDto | null;
  settings: GisImportSettingsDto | null;
  result: GisImportResultDto | null;
  busy: boolean;
  error: string | null;
}

export function initialGisMachine(): GisImportMachine {
  return {
    step: "pick",
    path: "",
    probe: null,
    settings: null,
    result: null,
    busy: false,
    error: null,
  };
}

interface GisImportState extends GisImportMachine {
  /** Probe a file and move to the settings step. */
  pick: (path: string, sourceCrs?: string) => Promise<void>;
  patchSettings: (patch: Partial<GisImportSettingsDto>) => void;
  /** Re-probe with a stated CRS — the escape hatch for an unreadable `.prj`. */
  restateCrs: (crs: string) => Promise<void>;
  start: () => Promise<void>;
  back: () => void;
  reset: () => void;
}

export const useGisImportStore = create<GisImportState>((set, get) => ({
  ...initialGisMachine(),

  pick: async (path, sourceCrs = "") => {
    set({ busy: true, error: null, path });
    try {
      const probe = await gisIpc.probe(path, sourceCrs);
      // **IB-14: the wizard opens on the author's OWN cap**, not on the guard.
      // An author who raised it once for a county road layer should not raise
      // it again for the county's footprints.
      const remembered =
        useSettingsStore.getState().settings.gis_max_entities || GIS_DEFAULT_MAX_ENTITIES;
      const settings = await gisIpc.suggestedSettings(probe, remembered);
      set({ probe, settings, step: "configure", result: null, busy: false });
    } catch (e) {
      set({ busy: false, error: String(e) });
    }
  },

  patchSettings: (patch) => {
    const settings = get().settings;
    if (!settings) return;
    set({ settings: { ...settings, ...patch } });
    // …and raising it here is what "remembered" means. Written through the
    // settings store's debounced door, so dragging the field does not write a
    // file per keystroke.
    if (
      patch.max_entities !== undefined &&
      Number.isFinite(patch.max_entities) &&
      patch.max_entities >= 1
    ) {
      useSettingsStore.getState().patch({ gis_max_entities: patch.max_entities });
    }
  },

  restateCrs: async (crs) => {
    const { path } = get();
    if (!path) return;
    await get().pick(path, crs);
    const s = get().settings;
    if (s) set({ settings: { ...s, source_crs: crs } });
  },

  start: async () => {
    const { path, probe, settings } = get();
    const issue = gisSettingsIssue(probe, settings);
    if (issue || !settings) {
      set({ error: issue ?? "the wizard has no settings" });
      return;
    }
    set({ busy: true, error: null });
    try {
      const result = await gisIpc.import(path, settings);
      set({ result, step: "done", busy: false });
    } catch (e) {
      set({ busy: false, error: String(e) });
    }
  },

  back: () => set({ step: "configure", result: null, error: null }),
  reset: () => set(initialGisMachine()),
}));
