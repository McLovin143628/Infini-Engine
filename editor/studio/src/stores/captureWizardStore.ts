/**
 * Capture wizard state (P25.4) — "drop photos, get an asset".
 *
 * Five steps, one dialog: pick the photographs, configure the camera and the
 * scale, watch the four solve stages, look at what came out, import it. Every
 * transition is driven either by a command result (the whole
 * `CaptureStatusDto`, which the backend owns) or by a
 * `photogrammetry://progress` event for the run this store is watching.
 *
 * The transition logic lives in the PURE functions below (`captureReducer`,
 * `overallPercent`, `stepFor`, `issuesAt`, `scaleForLongestSide`,
 * `formatDuration`) and the store is a thin shell over them, so the machine is
 * unit-testable without a backend — the `terrainImportStore` pattern.
 *
 * NOTHING here predicts what the backend will produce. The status DTO is
 * authoritative and is replaced wholesale on every command; the reducer only
 * moves the progress bar and the step between status reads, because a run that
 * takes minutes cannot be shown by polling alone.
 *
 * UNITS: `metresPerUnit` is metres per reconstruction unit — the scale step. A
 * reconstruction is scale-ambiguous, so `1.0` means "baseline units", not
 * metres. `extentUnits` is in those units and does NOT move when the scale
 * does, which is what lets a second correction be computed against the same
 * number as the first.
 */
import { create } from "zustand";

import type { CaptureImportDto } from "../bindings/CaptureImportDto";
import type { CaptureIssueDto } from "../bindings/CaptureIssueDto";
import type { CapturePreviewDto } from "../bindings/CapturePreviewDto";
import type { CaptureProgressDto } from "../bindings/CaptureProgressDto";
import type { CaptureSettingsDto } from "../bindings/CaptureSettingsDto";
import type { CaptureStatusDto } from "../bindings/CaptureStatusDto";
import { listenTo, type UnlistenFn } from "../lib/events";
import { photogrammetry as photoIpc } from "../lib/ipc";

/** Where the wizard is. */
export type CaptureStep = "pick" | "configure" | "running" | "review";

/** The part of the wizard the reducer owns (everything a test needs). */
export interface CaptureMachine {
  step: CaptureStep;
  /** The backend's own view, replaced wholesale by every command. */
  status: CaptureStatusDto | null;
  /** The last progress event for the run being watched. */
  progress: CaptureProgressDto | null;
  /** What an import wrote, once one has. */
  result: CaptureImportDto | null;
  /** The two preview images. */
  preview: CapturePreviewDto | null;
  /** Orbit angles for the offscreen preview, in degrees. */
  yaw: number;
  pitch: number;
  /** A command failed outright (a pre-flight refusal, or IPC itself). */
  error: string | null;
  /** A command is in flight — disables the buttons. */
  busy: boolean;
}

/** The wizard's initial (nothing picked) state. */
export function initialMachine(): CaptureMachine {
  return {
    step: "pick",
    status: null,
    progress: null,
    result: null,
    preview: null,
    yaw: 30,
    pitch: 20,
    error: null,
    busy: false,
  };
}

/**
 * Which step a backend state belongs on.
 *
 * A function of the status alone, so the step can never disagree with what the
 * backend thinks is happening — the failure mode a separately-tracked step has
 * is a dialog stuck on "running" over a session that finished minutes ago.
 *
 * `failed` and `cancelled` return to **configure**, deliberately: both leave the
 * photographs loaded and the settings the user typed, so the next thing they
 * want is the button that starts it again — not a dead end and not a wizard
 * that has forgotten its own inputs.
 */
export function stepFor(status: CaptureStatusDto | null): CaptureStep {
  if (!status) return "pick";
  switch (status.state) {
    case "running":
      return "running";
    case "ready":
    case "imported":
      return "review";
    default:
      return status.photos.length > 0 ? "configure" : "pick";
  }
}

/**
 * Fold one `photogrammetry://progress` event into the machine.
 *
 * Events for other runs are ignored (one channel, one session, but a cancelled
 * run's last event can arrive after the next one starts). A terminal phase
 * always leaves a step the UI can act on, and never invents a status the
 * backend has not reported — `status` is refreshed by the store's action, not
 * by this function, because the event carries a stage and the status carries
 * everything else.
 */
export function captureReducer(
  state: CaptureMachine,
  event: CaptureProgressDto,
): CaptureMachine {
  const run = state.status ? Number(state.status.run) : null;
  if (run !== null && Number(event.run) !== run) return state;
  switch (event.phase) {
    case "started":
    case "progress":
      return { ...state, progress: event, step: "running", error: null };
    case "finished":
      // The LAST stage finishing is not the run finishing: `write` is a stage
      // too, and the four automatic ones are followed by a status read. So the
      // step is left where it is and only the bar moves.
      return { ...state, progress: event };
    case "failed":
      return { ...state, progress: event, error: event.error ?? "the reconstruction failed" };
    case "cancelled":
      // A cancellation is not an error banner: the user asked for it.
      return { ...state, progress: event, error: null };
    default:
      return state;
  }
}

/**
 * Overall progress as a 0-100 integer, across the whole pipeline.
 *
 * Stages are weighted equally, which is a lie about the clock (the dense solve
 * and the finish dominate) and the truth about what a bar can promise: the
 * event carries `stageIndex` and `stages`, so the bar is derived rather than
 * holding a second copy of the stage order. Within a stage, `done/total`
 * refines the position when the stage reports it and the bar sits at the
 * stage's start when it does not — the two Ring-0 solves, honestly.
 */
export function overallPercent(event: CaptureProgressDto | null): number {
  if (!event || event.stages <= 0) return 0;
  const span = 100 / event.stages;
  const base = event.stageIndex * span;
  const total = Number(event.total);
  const done = Number(event.done);
  let within = 0;
  if (event.phase === "finished") within = span;
  else if (total > 0 && Number.isFinite(done)) within = span * Math.min(1, done / total);
  return Math.max(0, Math.min(100, Math.round(base + within)));
}

/** The findings raised at one stage, in the order the backend sorted them. */
export function issuesAt(
  issues: CaptureIssueDto[],
  stage: CaptureIssueDto["stage"],
): CaptureIssueDto[] {
  return issues.filter((i) => i.stage === stage);
}

/** Whether anything blocks a run from starting. */
export function blockingIssues(issues: CaptureIssueDto[]): CaptureIssueDto[] {
  return issues.filter((i) => i.severity === "blocking");
}

/**
 * The multiplier that makes a reconstruction's longest side measure
 * `knownMetres` — the scale step's arithmetic, mirroring
 * `inf_editor_core::capture::scale_for_longest_side`.
 *
 * A mirror rather than a round trip because it runs on every keystroke of a
 * number field, and because the backend still refuses everything this refuses:
 * `null` here only decides whether the button is enabled.
 */
export function scaleForLongestSide(
  extentUnits: number,
  knownMetres: number,
): number | null {
  if (!Number.isFinite(extentUnits) || extentUnits <= 0) return null;
  if (!Number.isFinite(knownMetres) || knownMetres <= 0) return null;
  return knownMetres / extentUnits;
}

/** Milliseconds as a short human duration. Display only. */
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m} m ${Math.round(s - m * 60)} s`;
}

/** A file name out of a path, for the photograph table. */
export function fileNameOf(path: string): string {
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

interface CaptureState extends CaptureMachine {
  /** Re-read the backend's state. */
  refresh: () => Promise<void>;
  /** Load a photograph set (replacing whatever was loaded). */
  loadPhotos: (paths: string[]) => Promise<void>;
  /** Edit the settings block. */
  patchSettings: (patch: Partial<CaptureSettingsDto>) => Promise<void>;
  /** Edit the assumed camera. */
  patchCamera: (patch: Partial<CaptureSettingsDto["camera"]>) => Promise<void>;
  /** Start a reconstruction. */
  start: () => Promise<void>;
  /** Re-run the finish stage alone (how the scale step is applied). */
  refinish: () => Promise<void>;
  /** Ask an in-flight run to stop. */
  cancel: () => Promise<void>;
  /** Re-render the preview at the current orbit. */
  refreshPreview: () => Promise<void>;
  /** Orbit the preview. */
  orbit: (yaw: number, pitch: number) => void;
  /** Write the five assets. */
  importScan: (name: string) => Promise<CaptureImportDto | null>;
  /** Fold a progress event in. */
  applyProgress: (event: CaptureProgressDto) => void;
  /** Forget everything (dialog closed). */
  reset: () => Promise<void>;
}

/** Apply a status answer, moving the step with it. */
function applyStatus(status: CaptureStatusDto): Partial<CaptureMachine> {
  return { status, step: stepFor(status), busy: false };
}

export const useCaptureStore = create<CaptureState>((set, get) => ({
  ...initialMachine(),

  refresh: async () => {
    try {
      const status = await photoIpc.status();
      set(applyStatus(status));
      // A finished run gets a preview, from here rather than from a mount
      // effect: the review step is reached three different ways (a run
      // finishing, a re-finish finishing, and re-opening the dialog over a
      // session that already has a product) and an effect per way is three
      // places to forget.
      const ready = status.state === "ready" || status.state === "imported";
      if (ready && !get().preview) void get().refreshPreview();
    } catch (e) {
      set({ error: String(e), busy: false });
    }
  },

  loadPhotos: async (paths) => {
    set({ busy: true, error: null, result: null, preview: null, progress: null });
    try {
      set(applyStatus(await photoIpc.load(paths)));
    } catch (e) {
      set({ busy: false, error: String(e) });
    }
  },

  patchSettings: async (patch) => {
    const settings = get().status?.settings;
    if (!settings) return;
    set({ busy: true, error: null });
    try {
      set(applyStatus(await photoIpc.setSettings({ ...settings, ...patch })));
    } catch (e) {
      set({ busy: false, error: String(e) });
    }
  },

  patchCamera: async (patch) => {
    const settings = get().status?.settings;
    if (!settings) return;
    await get().patchSettings({ camera: { ...settings.camera, ...patch } });
  },

  start: async () => {
    set({ busy: true, error: null, progress: null, result: null, preview: null });
    try {
      set(applyStatus(await photoIpc.start()));
    } catch (e) {
      set({ busy: false, error: String(e) });
    }
  },

  refinish: async () => {
    set({ busy: true, error: null, progress: null, preview: null });
    try {
      set(applyStatus(await photoIpc.refinish()));
    } catch (e) {
      set({ busy: false, error: String(e) });
    }
  },

  cancel: async () => {
    try {
      await photoIpc.cancel();
    } catch (e) {
      set({ error: String(e) });
    }
  },

  refreshPreview: async () => {
    const { yaw, pitch, status } = get();
    if (!status || (status.state !== "ready" && status.state !== "imported")) return;
    try {
      set({ preview: await photoIpc.preview(yaw, pitch) });
    } catch (e) {
      // A preview is a readback, not a gate — a failure here must never stop an
      // import, so it is shown in the preview's own slot and nowhere else.
      set({ preview: { geometry: null, albedo: null, error: String(e), size: 0 } });
    }
  },

  orbit: (yaw, pitch) => set({ yaw, pitch: Math.max(-89, Math.min(89, pitch)) }),

  importScan: async (name) => {
    set({ busy: true, error: null });
    try {
      const result = await photoIpc.import(name);
      set({ result });
      set(applyStatus(await photoIpc.status()));
      return result;
    } catch (e) {
      set({ busy: false, error: String(e) });
      return null;
    }
  },

  applyProgress: (event) => {
    const next = captureReducer(get(), event);
    if (next === get()) return;
    set(next);
    // A terminal phase means the backend's own state has moved; re-read it
    // rather than guessing what it moved to.
    if (event.phase === "finished" && event.stage === "finish") void get().refresh();
    if (event.phase === "failed" || event.phase === "cancelled") void get().refresh();
  },

  reset: async () => {
    set(initialMachine());
    try {
      await photoIpc.reset();
    } catch {
      /* a reset that cannot reach the backend still clears the dialog */
    }
  },
}));

let unlisten: UnlistenFn | null = null;

/**
 * Subscribe the wizard to `photogrammetry://progress`. Idempotent (StrictMode
 * double-mounts); returns a disposer.
 */
export async function initCaptureSync(): Promise<() => void> {
  if (unlisten) return () => {};
  unlisten = await listenTo("photogrammetry://progress", (e) =>
    useCaptureStore.getState().applyProgress(e),
  );
  return () => {
    unlisten?.();
    unlisten = null;
  };
}
