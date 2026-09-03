// @vitest-environment jsdom
/**
 * **The application opens on a project** (wave CERT1, CP-C1).
 *
 * Before this wave a cold launch had no answer to "which project": Rust's
 * `ProjectState::current` started `None` every time, nothing persisted what had
 * been open, and the showcase the engine exists to show off was a file dialog
 * away. `initProjectSync` now asks the backend once, on mount, when nothing is
 * open — `project_boot_default`, which resolves `INF_BOOT_PROJECT` → the
 * `boot_project` pin → the showcase island beside the checkout → nothing.
 *
 * The three arms are the three things that can go wrong in the FRONTEND half,
 * and each one passed before only because nothing asked:
 *
 *  * it must be asked at all on a cold launch (the store used to stop at
 *    `refresh`, which sets `current` and opens nothing);
 *  * it must NOT be asked when a project is already open — a re-mount that
 *    re-rooted a live asset database is the defect this guard exists for;
 *  * a `null` answer must leave the start screen alone and raise no status
 *    line, because "no showcase on this machine" is the ordinary case for
 *    everyone but the author of the island.
 */
import { afterEach, beforeEach, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  scene: {
    open: vi.fn(() => Promise.resolve(null)),
    currentPath: vi.fn(() => Promise.resolve<string | null>(null)),
  },
  project: {
    templates: vi.fn(() => Promise.resolve([])),
    recent: vi.fn(() => Promise.resolve([])),
    current: vi.fn(() => Promise.resolve(null)),
    bootLevel: vi.fn(() => Promise.resolve<string | null>(null)),
    bootDefault: vi.fn(() => Promise.resolve(null)),
  },
}));

vi.mock("../../lib/events", () => ({
  listenTo: vi.fn(() => Promise.resolve(() => {})),
  refCountedInit: (fn: (sink: (d: () => void) => void) => Promise<() => void>) => {
    let disposer: (() => void) | null = null;
    return async () => {
      disposer = await fn(() => {});
      return () => disposer?.();
    };
  },
}));

vi.mock("../assetStore", () => ({
  useAssetStore: { getState: () => ({ refresh: vi.fn(() => Promise.resolve()) }) },
}));

import type { ProjectBootDto } from "../../bindings/ProjectBootDto";
import type { ProjectInfoDto } from "../../bindings/ProjectInfoDto";
import { project as projectIpc } from "../../lib/ipc";
import { initProjectSync, useProjectStore } from "../projectStore";
import { useShellStore } from "../shellStore";

const ISLAND: ProjectInfoDto = {
  name: "Vancouver Island",
  root: "C:/Users/x/island-build/project",
  content_dir: "Content",
  levels_dir: "Levels",
  template: "blank-3d",
};

const BOOT: ProjectBootDto = { project: ISLAND, source: "the showcase island" };

beforeEach(() => {
  vi.mocked(projectIpc.current).mockResolvedValue(null);
  vi.mocked(projectIpc.bootDefault).mockResolvedValue(null);
  useProjectStore.setState({ current: null, recent: [], templates: [], ready: false });
  useShellStore.setState({ statusMessage: null });
});

afterEach(() => {
  vi.clearAllMocks();
});

it("a cold launch asks the backend which project to open", async () => {
  vi.mocked(projectIpc.bootDefault).mockResolvedValue(BOOT);

  const dispose = await initProjectSync();
  try {
    expect(projectIpc.bootDefault).toHaveBeenCalledTimes(1);
    // The RUNG reaches the author, not just the name: "you pinned this" and
    // "the engine found the showcase beside your checkout" are two different
    // sentences and the status line has to be able to say which.
    expect(useShellStore.getState().statusMessage).toContain("Vancouver Island");
    expect(useShellStore.getState().statusMessage).toContain("the showcase island");
  } finally {
    dispose();
  }
});

it("a launch with a project already open does not ask", async () => {
  vi.mocked(projectIpc.current).mockResolvedValue(ISLAND);

  const dispose = await initProjectSync();
  try {
    expect(projectIpc.bootDefault).not.toHaveBeenCalled();
  } finally {
    dispose();
  }
});

it("no boot project leaves the start screen alone and says nothing", async () => {
  const dispose = await initProjectSync();
  try {
    expect(projectIpc.bootDefault).toHaveBeenCalledTimes(1);
    expect(useProjectStore.getState().current).toBeNull();
    expect(useShellStore.getState().statusMessage).toBeNull();
  } finally {
    dispose();
  }
});
