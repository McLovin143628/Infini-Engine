// @vitest-environment jsdom
//
// Capture wizard (P25.4): the pure state machine, the progress arithmetic, the
// scale helper, and the store actions that drive them. IPC is mocked — nothing
// here reconstructs anything; that is Ring 0's (`inf_photo`, `inf_photo_gpu`)
// and Ring 1's (`inf_editor_core::capture`), and both have their own batteries
// including an eleven-arm gate over the session door.
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  photogrammetry: {
    status: vi.fn(),
    load: vi.fn(),
    setSettings: vi.fn(),
    start: vi.fn(),
    refinish: vi.fn(),
    cancel: vi.fn(),
    reset: vi.fn(),
    preview: vi.fn(),
    import: vi.fn(),
  },
}));

import type { CaptureIssueDto } from "../../bindings/CaptureIssueDto";
import type { CaptureProgressDto } from "../../bindings/CaptureProgressDto";
import type { CaptureSettingsDto } from "../../bindings/CaptureSettingsDto";
import type { CaptureStatusDto } from "../../bindings/CaptureStatusDto";
import { photogrammetry as photoIpc } from "../../lib/ipc";
import {
  blockingIssues,
  captureReducer,
  fileNameOf,
  formatDuration,
  initialMachine,
  issuesAt,
  overallPercent,
  scaleForLongestSide,
  stepFor,
  useCaptureStore,
  type CaptureMachine,
} from "../captureWizardStore";

const SETTINGS: CaptureSettingsDto = {
  camera: { focalRatio: 1.2, k1: 0, k2: 0 },
  metresPerUnit: 1,
  targetTriangles: 20000,
  atlasSize: 1024,
  aoRays: 32,
  trimUnseen: true,
  delight: false,
  roughness: 0.8,
  metallic: 0,
};

const ISSUES: CaptureIssueDto[] = [
  { severity: "blocking", stage: "load", message: "photogrammetry needs at least 3" },
  { severity: "warning", stage: "sfm", message: "b.jpg never got a pose" },
  { severity: "warning", stage: "finish", message: "seen by exactly ONE camera" },
  { severity: "note", stage: "write", message: "draws as a PLACEHOLDER CUBE" },
];

function status(partial: Partial<CaptureStatusDto> = {}): CaptureStatusDto {
  return {
    state: "idle",
    stage: null,
    run: 0,
    photos: [],
    settings: SETTINGS,
    issues: [],
    result: null,
    error: null,
    folder: "Scans",
    ...partial,
  };
}

function photo(name: string) {
  return { path: `C:/shoot/${name}`, name, width: 4000, height: 3000, error: null };
}

function event(partial: Partial<CaptureProgressDto> = {}): CaptureProgressDto {
  return {
    run: 1,
    stage: "dense",
    stageIndex: 2,
    stages: 5,
    phase: "progress",
    done: 0,
    total: 0,
    detail: "",
    error: null,
    ...partial,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  useCaptureStore.setState(initialMachine());
});

describe("stepFor", () => {
  it("is a function of the backend's state and nothing else", () => {
    expect(stepFor(null)).toBe("pick");
    expect(stepFor(status())).toBe("pick");
    expect(stepFor(status({ photos: [photo("a.jpg")] }))).toBe("configure");
    expect(stepFor(status({ state: "running", stage: "dense" }))).toBe("running");
    expect(stepFor(status({ state: "ready" }))).toBe("review");
    expect(stepFor(status({ state: "imported" }))).toBe("review");
  });

  it("returns a failed or cancelled run to the settings it was started from", () => {
    // Both leave the photographs and the settings loaded, so the next thing a
    // user wants is the button that starts it again — not a dead end.
    const loaded = { photos: [photo("a.jpg"), photo("b.jpg"), photo("c.jpg")] };
    expect(stepFor(status({ state: "failed", ...loaded }))).toBe("configure");
    expect(stepFor(status({ state: "cancelled", ...loaded }))).toBe("configure");
    // …but a failure with nothing loaded goes back to the picker rather than to
    // an empty settings page.
    expect(stepFor(status({ state: "failed" }))).toBe("pick");
  });
});

describe("captureReducer", () => {
  const running: CaptureMachine = {
    ...initialMachine(),
    status: status({ state: "running", run: 1 }),
  };

  it("moves the bar and the step on a start or a tick", () => {
    const started = captureReducer(running, event({ phase: "started", stage: "load" }));
    expect(started.step).toBe("running");
    expect(started.progress?.stage).toBe("load");
    const ticked = captureReducer(started, event({ done: 3, total: 6, stage: "load" }));
    expect(Number(ticked.progress?.done)).toBe(3);
    expect(ticked.error).toBeNull();
  });

  it("ignores events from a run it is not watching", () => {
    const other = captureReducer(running, event({ run: 99, phase: "failed" }));
    expect(other).toBe(running);
    expect(other.error).toBeNull();
  });

  it("does not move the step when a STAGE finishes", () => {
    // A stage finishing is not the run finishing — `write` is a stage too — so
    // only the bar moves and the backend's status decides the rest.
    const next = captureReducer(running, event({ phase: "finished", stage: "finish" }));
    expect(next.step).toBe(running.step);
    expect(next.progress?.phase).toBe("finished");
  });

  it("carries a failure's own words and leaves a cancellation silent", () => {
    const failed = captureReducer(
      running,
      event({ phase: "failed", stage: "sfm", error: "no viable initial pair" }),
    );
    expect(failed.error).toBe("no viable initial pair");
    // A cancellation is not an error banner: the user asked for it.
    const cancelled = captureReducer(failed, event({ phase: "cancelled", stage: "dense" }));
    expect(cancelled.error).toBeNull();
    expect(cancelled.progress?.phase).toBe("cancelled");
  });

  it("takes a first event before any status has landed", () => {
    // The run id is unknown until `start` resolves, and the first event can beat
    // it: filtering on a null run would drop the beginning of every capture.
    const fresh = captureReducer(initialMachine(), event({ phase: "started", stage: "load" }));
    expect(fresh.step).toBe("running");
  });
});

describe("overallPercent", () => {
  it("weights the five stages equally and refines within one that reports", () => {
    expect(overallPercent(null)).toBe(0);
    expect(overallPercent(event({ stageIndex: 0, phase: "started" }))).toBe(0);
    // Half-way through the third of five stages.
    expect(overallPercent(event({ stageIndex: 2, done: 1, total: 2 }))).toBe(50);
    // A stage that reports no progress sits at its own start — the honest
    // position for the two Ring-0 solves.
    expect(overallPercent(event({ stageIndex: 3, done: 0, total: 0 }))).toBe(60);
    // …and reaches its end when it finishes.
    expect(overallPercent(event({ stageIndex: 3, phase: "finished" }))).toBe(80);
    expect(overallPercent(event({ stageIndex: 4, phase: "finished" }))).toBe(100);
  });

  it("never leaves 0-100 whatever it is handed", () => {
    expect(overallPercent(event({ stages: 0 }))).toBe(0);
    expect(overallPercent(event({ stageIndex: 9, done: 99, total: 1 }))).toBe(100);
    expect(overallPercent(event({ stageIndex: 0, done: -5, total: 10 }))).toBe(0);
  });
});

describe("the diagnostics projections", () => {
  it("groups by stage and finds what blocks", () => {
    expect(issuesAt(ISSUES, "sfm")).toHaveLength(1);
    expect(issuesAt(ISSUES, "finish")[0].message).toMatch(/ONE camera/);
    expect(issuesAt(ISSUES, "dense")).toHaveLength(0);
    const blocking = blockingIssues(ISSUES);
    expect(blocking).toHaveLength(1);
    expect(blocking[0].stage).toBe("load");
    // A note never blocks — the placeholder-cube one is the reason that matters.
    expect(blockingIssues([ISSUES[3]])).toHaveLength(0);
  });
});

describe("scaleForLongestSide", () => {
  it("mirrors the Ring-1 refusals exactly", () => {
    expect(scaleForLongestSide(4, 2)).toBe(0.5);
    expect(scaleForLongestSide(0, 2)).toBeNull();
    expect(scaleForLongestSide(4, 0)).toBeNull();
    expect(scaleForLongestSide(-4, 2)).toBeNull();
    expect(scaleForLongestSide(4, -2)).toBeNull();
    expect(scaleForLongestSide(Number.NaN, 2)).toBeNull();
    expect(scaleForLongestSide(Number.POSITIVE_INFINITY, 2)).toBeNull();
  });
});

describe("the display helpers", () => {
  it("format a duration and a file name", () => {
    expect(formatDuration(450)).toBe("450 ms");
    expect(formatDuration(1500)).toBe("1.5 s");
    expect(formatDuration(125000)).toBe("2 m 5 s");
    expect(formatDuration(-1)).toBe("—");
    expect(fileNameOf("C:\\shoot\\IMG_0001.jpg")).toBe("IMG_0001.jpg");
    expect(fileNameOf("/home/a/b.png")).toBe("b.png");
    expect(fileNameOf("plain.png")).toBe("plain.png");
  });
});

describe("the store actions", () => {
  it("loads photographs and lands on the step the backend's answer names", async () => {
    const loaded = status({ photos: [photo("a.jpg"), photo("b.jpg"), photo("c.jpg")] });
    vi.mocked(photoIpc.load).mockResolvedValue(loaded);
    await useCaptureStore.getState().loadPhotos(["C:/shoot/a.jpg"]);
    expect(photoIpc.load).toHaveBeenCalledWith(["C:/shoot/a.jpg"]);
    expect(useCaptureStore.getState().step).toBe("configure");
    expect(useCaptureStore.getState().busy).toBe(false);
  });

  it("sends the whole settings block on a patch, so the backend never sees a gap", async () => {
    useCaptureStore.setState({ status: status({ photos: [photo("a.jpg")] }) });
    vi.mocked(photoIpc.setSettings).mockResolvedValue(status());
    await useCaptureStore.getState().patchSettings({ metresPerUnit: 0.25 });
    expect(photoIpc.setSettings).toHaveBeenCalledWith({ ...SETTINGS, metresPerUnit: 0.25 });
    // A camera patch is a settings patch over the camera it already has.
    vi.mocked(photoIpc.setSettings).mockClear();
    await useCaptureStore.getState().patchCamera({ focalRatio: 0.9375 });
    expect(photoIpc.setSettings).toHaveBeenCalledWith({
      ...SETTINGS,
      camera: { focalRatio: 0.9375, k1: 0, k2: 0 },
    });
  });

  it("shows a pre-flight refusal as an error rather than a step change", async () => {
    useCaptureStore.setState({ status: status({ photos: [photo("a.jpg")] }), step: "configure" });
    vi.mocked(photoIpc.start).mockRejectedValue(new Error("at least 3 photographs"));
    await useCaptureStore.getState().start();
    const s = useCaptureStore.getState();
    expect(s.error).toMatch(/at least 3/);
    expect(s.step).toBe("configure");
    expect(s.busy).toBe(false);
  });

  it("never lets a failed preview stop an import", async () => {
    useCaptureStore.setState({ status: status({ state: "ready" }) });
    vi.mocked(photoIpc.preview).mockRejectedValue(new Error("no adapter"));
    await useCaptureStore.getState().refreshPreview();
    const s = useCaptureStore.getState();
    expect(s.preview?.error).toMatch(/no adapter/);
    // The top-level error slot — the one that gates the buttons — is untouched.
    expect(s.error).toBeNull();
  });

  it("does not ask for a preview before there is anything to preview", async () => {
    useCaptureStore.setState({ status: status({ state: "running" }) });
    await useCaptureStore.getState().refreshPreview();
    expect(photoIpc.preview).not.toHaveBeenCalled();
  });

  it("re-reads the status after an import and keeps what was written", async () => {
    useCaptureStore.setState({ status: status({ state: "ready" }) });
    const written = {
      mesh: "m",
      albedo: "a",
      normal: "n",
      orm: "o",
      material: "mat",
      folder: "Scans",
      name: "Ridge",
      notes: ["draws as a PLACEHOLDER CUBE"],
    };
    vi.mocked(photoIpc.import).mockResolvedValue(written);
    vi.mocked(photoIpc.status).mockResolvedValue(status({ state: "imported" }));
    vi.mocked(photoIpc.preview).mockResolvedValue({
      geometry: null,
      albedo: null,
      error: null,
      size: 256,
    });
    const result = await useCaptureStore.getState().importScan("Ridge");
    expect(result).toEqual(written);
    const s = useCaptureStore.getState();
    expect(s.result?.notes[0]).toMatch(/PLACEHOLDER CUBE/);
    expect(s.step).toBe("review");
    expect(s.busy).toBe(false);
  });

  it("asks for a preview once a run lands on ready, and only once", async () => {
    vi.mocked(photoIpc.status).mockResolvedValue(status({ state: "ready" }));
    vi.mocked(photoIpc.preview).mockResolvedValue({
      geometry: "data:image/png;base64,AA",
      albedo: null,
      error: null,
      size: 256,
    });
    await useCaptureStore.getState().refresh();
    // The preview lands on a microtask the action kicked; drain it.
    await Promise.resolve();
    await Promise.resolve();
    expect(photoIpc.preview).toHaveBeenCalledTimes(1);
    await useCaptureStore.getState().refresh();
    await Promise.resolve();
    expect(photoIpc.preview).toHaveBeenCalledTimes(1);
  });

  it("clears everything on reset even when the backend cannot be reached", async () => {
    useCaptureStore.setState({ status: status({ state: "ready" }), step: "review" });
    vi.mocked(photoIpc.reset).mockRejectedValue(new Error("gone"));
    await useCaptureStore.getState().reset();
    const s = useCaptureStore.getState();
    expect(s.step).toBe("pick");
    expect(s.status).toBeNull();
    expect(s.error).toBeNull();
  });
});
