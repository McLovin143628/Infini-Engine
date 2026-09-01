// @vitest-environment jsdom
/**
 * **The editor can reach its levels** (wave GTA1, clause 5).
 *
 * Two defects, both of which passed every previous test because nothing asked
 * these two questions:
 *
 *  * `File ▸ Open Level…` called `scene_open` with NO PATH, and no path means
 *    the quicksave fallback — so the row that says "open a level" silently
 *    replaced the document with `quicksave.inf_lvl`. The assertion is on the
 *    ARGUMENT, because "it called open" was true before the fix too.
 *  * opening a project left the document alone, so a fresh `inf new` put an
 *    author in a project whose level they had to go and find. It opens the
 *    project's boot level now — unless the current document has unsaved
 *    changes, which is the arm that keeps a menu click from discarding work.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const dialogOpen = vi.fn();
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: dialogOpen }));

vi.mock("../../lib/ipc", () => ({
  scene: {
    open: vi.fn(() => Promise.resolve(snapshot())),
    currentPath: vi.fn(() => Promise.resolve<string | null>(null)),
  },
  project: {
    templates: vi.fn(() => Promise.resolve([])),
    recent: vi.fn(() => Promise.resolve([])),
    current: vi.fn(() => Promise.resolve(null)),
    bootLevel: vi.fn(() => Promise.resolve<string | null>("C:/proj/Content/Levels/Blank.inf_lvl")),
  },
}));

vi.mock("../assetStore", () => ({
  useAssetStore: { getState: () => ({ refresh: vi.fn(() => Promise.resolve()) }) },
}));

import type { ProjectInfoDto } from "../../bindings/ProjectInfoDto";
import type { SceneSnapshot } from "../../bindings/SceneSnapshot";
import { project as projectIpc, scene as sceneIpc } from "../../lib/ipc";
import { useProjectStore } from "../projectStore";
import { useSceneStore } from "../sceneStore";
import { useShellStore } from "../shellStore";

function snapshot(dirty = false): SceneSnapshot {
  return {
    version: 1,
    roots: [],
    nodes: [],
    selection: [],
    dirty,
    title: "Opened",
    can_undo: false,
    can_redo: false,
    undo_label: null,
    redo_label: null,
  };
}

const PROJECT: ProjectInfoDto = {
  name: "proj",
  root: "C:/proj",
  content_dir: "Content",
  levels_dir: "Levels",
  template: "blank-3d",
};

beforeEach(() => {
  vi.mocked(sceneIpc.open).mockResolvedValue(snapshot());
  vi.mocked(sceneIpc.currentPath).mockResolvedValue(null);
  vi.mocked(projectIpc.bootLevel).mockResolvedValue("C:/proj/Content/Levels/Blank.inf_lvl");
  useSceneStore.getState().applySnapshot(snapshot());
});

afterEach(() => {
  useShellStore.getState().clearStatus();
  vi.clearAllMocks();
});

describe("File ▸ Open Level…", () => {
  it("opens the level the author PICKED, not the quicksave", async () => {
    dialogOpen.mockResolvedValueOnce("C:/proj/Content/Levels/Second.inf_lvl");
    await useSceneStore.getState().openLevelViaDialog();
    expect(vi.mocked(sceneIpc.open)).toHaveBeenCalledWith(
      "C:/proj/Content/Levels/Second.inf_lvl",
    );
  });

  it("opens nothing when the dialog is cancelled", async () => {
    dialogOpen.mockResolvedValueOnce(null);
    await useSceneStore.getState().openLevelViaDialog();
    expect(vi.mocked(sceneIpc.open)).not.toHaveBeenCalled();
  });

  it("reports a failed open instead of leaving the menu silent", async () => {
    vi.mocked(sceneIpc.open).mockRejectedValueOnce(new Error("bad schema"));
    const ok = await useSceneStore.getState().openLevel("C:/proj/x.inf_lvl");
    expect(ok).toBe(false);
    expect(useShellStore.getState().statusMessage).toContain("bad schema");
  });
});

describe("opening a project opens its boot level", () => {
  it("opens the level a cooked build would boot into", async () => {
    useProjectStore.getState().applyChanged(PROJECT);
    await vi.waitFor(() =>
      expect(vi.mocked(sceneIpc.open)).toHaveBeenCalledWith(
        "C:/proj/Content/Levels/Blank.inf_lvl",
      ),
    );
  });

  it("never replaces a document with unsaved changes", async () => {
    useSceneStore.getState().applySnapshot(snapshot(true));
    useProjectStore.getState().applyChanged(PROJECT);
    await vi.waitFor(() => expect(useShellStore.getState().statusMessage).toContain("unsaved"));
    expect(vi.mocked(sceneIpc.open)).not.toHaveBeenCalled();
  });

  it("says nothing when the project has no level at all", async () => {
    vi.mocked(projectIpc.bootLevel).mockResolvedValueOnce(null);
    useProjectStore.getState().applyChanged(PROJECT);
    await vi.waitFor(() => expect(vi.mocked(projectIpc.bootLevel)).toHaveBeenCalled());
    expect(vi.mocked(sceneIpc.open)).not.toHaveBeenCalled();
  });
});
