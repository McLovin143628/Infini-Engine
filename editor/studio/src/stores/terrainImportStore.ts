/**
 * Terrain Import wizard state (P16.4a).
 *
 * The wizard is a four-step machine — pick a file, configure the world, watch it
 * import, add it to the scene — and every transition is driven either by a
 * command result or by an `assets://import` event for the job it owns.
 *
 * The transition logic lives in the PURE functions below
 * (`terrainImportReducer`, `progressPercent`, `formatKm`, `settingsIssue`) and
 * the store is a thin shell over them, so the machine is unit-testable without a
 * backend (`stores/__tests__/terrainImportStore.test.ts`).
 *
 * UNITS: every length that crosses IPC is metres (units doctrine). `formatKm` is
 * a DISPLAY conversion — nothing stored or sent is ever scaled.
 */
import { create } from "zustand";

import type { HeightmapProbeDto } from "../bindings/HeightmapProbeDto";
import type { ImportEventDto } from "../bindings/ImportEventDto";
import type { TerrainImportPlanDto } from "../bindings/TerrainImportPlanDto";
import type { TerrainImportResultDto } from "../bindings/TerrainImportResultDto";
import type { TerrainImportSettingsDto } from "../bindings/TerrainImportSettingsDto";
import { listenTo, refCountedInit } from "../lib/events";
import { terrain as terrainIpc } from "../lib/ipc";

/** Where the wizard is. */
export type TerrainImportStep = "pick" | "configure" | "importing" | "done";

/** The part of the wizard the reducer owns (everything a test needs). */
export interface TerrainImportMachine {
  step: TerrainImportStep;
  probe: HeightmapProbeDto | null;
  settings: TerrainImportSettingsDto | null;
  plan: TerrainImportPlanDto | null;
  /** The queued import job id, while one is in flight. */
  job: number | null;
  /** Tiles written / expected, from the last progress tick. */
  done: number;
  total: number;
  /** The stage label of the last tick ("tiles", "lod2", …). */
  stage: string;
  result: TerrainImportResultDto | null;
  /**
   * **Non-fatal advisories from the finished import** (round-2 finding R2.F7).
   *
   * The wizard is the one importer where the source normally lives OUTSIDE the
   * project, so L7.H7's "no source path is recorded, re-import is unavailable"
   * note is the common case here — and it had no surface at all. The author
   * learned about it later, from `reimport` refusing. The Content Drawer's
   * advisory badge shows it too; this is the wizard saying it while the author
   * is still looking at the import they just made.
   */
  advisories: string[];
  error: string | null;
  /** A command is in flight (probe / spawn) — disables the buttons. */
  busy: boolean;
}

/** The wizard's initial (nothing picked) state. */
export function initialMachine(): TerrainImportMachine {
  return {
    step: "pick",
    probe: null,
    settings: null,
    plan: null,
    advisories: [],
    job: null,
    done: 0,
    total: 0,
    stage: "",
    result: null,
    error: null,
    busy: false,
  };
}

/**
 * Fold one `assets://import` event into the machine.
 *
 * Events for other jobs are ignored (the same channel carries every import in
 * the app), and a terminal event always leaves a step the UI can act on —
 * `done` with a result, or `importing` with an error the user can retry from.
 * A cancellation arrives as `failed` with a cancellation message, which is
 * deliberately not special-cased: the user asked for it, so it is not an error
 * banner, just a return to the settings step.
 */
export function terrainImportReducer(
  state: TerrainImportMachine,
  event: ImportEventDto,
): TerrainImportMachine {
  if (state.job === null || Number(event.job) !== state.job) return state;
  switch (event.phase) {
    case "started":
      return { ...state, step: "importing", error: null, advisories: [], done: 0 };
    case "progress":
      return {
        ...state,
        step: "importing",
        done: event.done == null ? state.done : Number(event.done),
        total: event.total == null ? state.total : Number(event.total),
        stage: event.stage ?? state.stage,
      };
    case "finished":
      return {
        ...state,
        step: "done",
        job: null,
        done: state.total || state.done,
        advisories: event.advisories,
        error: null,
      };
    case "failed": {
      const message = event.error ?? "import failed";
      const cancelled = message.toLowerCase().includes("cancel");
      return {
        ...state,
        // A cancelled import drops back to the settings the user already
        // filled in, so "Cancel" is a step back rather than a dead end.
        step: cancelled ? "configure" : "importing",
        job: null,
        error: cancelled ? null : message,
      };
    }
    default:
      return state;
  }
}

/** Progress as a 0–100 integer; an unknown total reads as 0 (indeterminate). */
export function progressPercent(done: number, total: number): number {
  if (!Number.isFinite(total) || total <= 0) return 0;
  const pct = (done / total) * 100;
  return Math.max(0, Math.min(100, Math.round(pct)));
}

/**
 * Metres → a short kilometre string for the extent readback. Display only.
 * Under a kilometre it stays in metres, because "0.13 km" helps nobody.
 */
export function formatKm(metres: number): string {
  if (!Number.isFinite(metres)) return "—";
  const m = Math.abs(metres);
  if (m < 1000) return `${Math.round(metres)} m`;
  const km = metres / 1000;
  return `${km >= 100 ? km.toFixed(0) : km.toFixed(2)} km`;
}

/** Bytes → a short human string (the done state's payload size). */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${i === 0 ? v : v.toFixed(v >= 100 ? 0 : 1)} ${units[i]}`;
}

/**
 * The first reason these settings cannot be imported, or `null`.
 *
 * Mirrors `HeightmapImport::validate` so the wizard says no *before* queueing a
 * job that would only fail on the worker. The backend still validates — this is
 * the fast path, not the authority.
 */
export function settingsIssue(
  settings: TerrainImportSettingsDto,
  probe: HeightmapProbeDto | null,
): string | null {
  if (!Number.isFinite(settings.tile_resolution) || settings.tile_resolution < 2) {
    return "Tile resolution must be at least 2 samples.";
  }
  if (!Number.isFinite(settings.meters_per_sample) || settings.meters_per_sample <= 0) {
    return "Metres per sample must be a positive length.";
  }
  if (settings.float_meters) {
    if (probe && !probe.float_samples) {
      return `Float-metres mode needs a float source; ${probe.format} carries integer samples.`;
    }
    return null;
  }
  if (!Number.isFinite(settings.min_height) || !Number.isFinite(settings.max_height)) {
    return "The height range must be finite.";
  }
  if (settings.max_height <= settings.min_height) {
    return "Max height must be greater than min height.";
  }
  return null;
}

interface TerrainImportState extends TerrainImportMachine {
  /** Probe a picked file and move to the settings step. */
  pick: (path: string) => Promise<void>;
  /** Edit the settings block (recomputes the plan). */
  patchSettings: (patch: Partial<TerrainImportSettingsDto>) => void;
  /** Re-fetch the plan for the current settings. */
  refreshPlan: () => Promise<void>;
  /** Queue the import. */
  start: () => Promise<void>;
  /** Ask the in-flight import to stop. */
  cancel: () => Promise<void>;
  /** Spawn a streamed Terrain entity for the finished asset. */
  addToScene: () => Promise<string | null>;
  /** Fold an `assets://import` event in. */
  applyImportEvent: (event: ImportEventDto) => void;
  /** Back to the settings step, keeping the probe. */
  back: () => void;
  /** Reset everything (dialog closed). */
  reset: () => void;
}

export const useTerrainImportStore = create<TerrainImportState>((set, get) => ({
  ...initialMachine(),

  pick: async (path) => {
    set({ busy: true, error: null });
    try {
      const probe = await terrainIpc.probeHeightmap(path);
      set({
        probe,
        settings: probe.suggested,
        step: "configure",
        result: null,
        advisories: [],
        job: null,
        done: 0,
        total: 0,
        busy: false,
      });
      await get().refreshPlan();
    } catch (e) {
      set({ busy: false, error: String(e) });
    }
  },

  patchSettings: (patch) => {
    const settings = get().settings;
    if (!settings) return;
    set({ settings: { ...settings, ...patch } });
    void get().refreshPlan();
  },

  refreshPlan: async () => {
    const { probe, settings } = get();
    if (!probe || !settings) return;
    try {
      const plan = await terrainIpc.importPlan(probe.width, probe.height, settings);
      set({ plan });
    } catch {
      // A plan is a readback, not a gate — leave the previous one rather than
      // blocking the wizard on a transient failure.
      set({ plan: null });
    }
  },

  start: async () => {
    const { probe, settings } = get();
    if (!probe || !settings) return;
    const issue = settingsIssue(settings, probe);
    if (issue) {
      set({ error: issue });
      return;
    }
    set({ busy: true, error: null, done: 0, total: 0, stage: "" });
    try {
      const job = await terrainIpc.import(probe.path, settings);
      set({ job: Number(job), step: "importing", busy: false });
    } catch (e) {
      set({ busy: false, error: String(e) });
    }
  },

  cancel: async () => {
    const job = get().job;
    if (job === null) return;
    try {
      await terrainIpc.cancelImport(job);
    } catch (e) {
      set({ error: String(e) });
    }
  },

  addToScene: async () => {
    const result = get().result;
    if (!result) return null;
    set({ busy: true, error: null });
    try {
      const guid = await terrainIpc.spawnStreamed(result.asset);
      set({ busy: false });
      return guid;
    } catch (e) {
      set({ busy: false, error: String(e) });
      return null;
    }
  },

  applyImportEvent: (event) => {
    const next = terrainImportReducer(get(), event);
    if (next === get()) return;
    set(next);
    // A finished job's headline asset carries the numbers the done state shows.
    if (next.step === "done" && event.primary) {
      void terrainIpc
        .assetInfo(event.primary)
        .then((result) => set({ result }))
        .catch((e) => set({ error: String(e) }));
    }
  },

  back: () => set({ step: "configure", error: null, result: null, advisories: [] }),

  reset: () => set(initialMachine()),
}));

/**
 * Subscribe the wizard to `assets://import`. Returns a disposer.
 *
 * **Refcounted** (`refCountedInit`, F-lens L7.M1): the guard used to be set
 * after the first `await`, so StrictMode's double mount subscribed twice and
 * orphaned the first handle.
 */
export const initTerrainImportSync = refCountedInit(async (sink) => {
  const unlisten = await listenTo("assets://import", (e) =>
    useTerrainImportStore.getState().applyImportEvent(e),
  );
  sink(unlisten);
  return () => unlisten();
});
