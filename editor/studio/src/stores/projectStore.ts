/**
 * Project store (Phase 5.5): the open project + recent list + templates.
 *
 * The authoritative project lives in Rust (`ProjectState`). Opening/creating a
 * project re-roots the asset database and emits `project://changed`; this store
 * mirrors the current project and drives the start screen. When no project is
 * open (or the user asks for the start screen) the `StartScreen` overlay shows.
 */
import { create } from "zustand";

import type { ProjectInfoDto } from "../bindings/ProjectInfoDto";
import type { ProjectTemplateDto } from "../bindings/ProjectTemplateDto";
import type { RecentProjectDto } from "../bindings/RecentProjectDto";
import { getCommand, registerCommand, setCommandHandler } from "../lib/commands";
import { listenTo, refCountedInit } from "../lib/events";
import { project as projectIpc } from "../lib/ipc";
import { RECENT_PROJECTS_MAX } from "../lib/menus";
import { useAssetStore } from "./assetStore";
import { useSceneStore } from "./sceneStore";
import { useShellStore } from "./shellStore";

export type { ProjectInfoDto, RecentProjectDto, ProjectTemplateDto };

interface ProjectState {
  current: ProjectInfoDto | null;
  recent: RecentProjectDto[];
  templates: ProjectTemplateDto[];
  ready: boolean;
  /** Force the start screen even while a project is open (File → New/Open). */
  showStartScreen: boolean;
  /** Last error surfaced in the start screen. */
  error: string | null;
  /**
   * **Which rung of `inf_project::boot::resolve` will answer next launch**, as
   * its own human phrase (CERT1 audit ruling), or `null` before anything has
   * asked.
   *
   * Set on a cold launch by {@link bootDefault} and re-set by the two
   * Preferences actions, so the row that offers those actions can say what it
   * is changing rather than making the author restart to find out. It is not
   * derived in TypeScript: the phrases come from `BootSource::phrase`, and a
   * second copy of the rung ladder across a language boundary is the defect
   * `inf_input::default_map` records against itself.
   */
  bootSource: string | null;

  refresh: () => Promise<void>;
  applyChanged: (info: ProjectInfoDto) => void;
  setShowStartScreen: (v: boolean) => void;

  /**
   * Open the application's boot project, if it has one (wave CERT1). Resolves
   * to the rung that answered, or null when the start screen should show.
   */
  bootDefault: () => Promise<string | null>;
  /**
   * Make the open project the deliberate default, or forget it (CERT1 audit
   * ruling). Both refresh {@link bootSource} from the backend's answer.
   */
  setBootDefault: (deliberate: boolean) => Promise<void>;

  newProject: (name: string, template: string, parentDir: string) => Promise<void>;
  openProject: (root: string) => Promise<void>;
  openViaDialog: () => Promise<void>;
  /** Pick a parent folder, then scaffold + open `name`/`template` inside it. */
  newViaDialog: (name: string, template: string) => Promise<void>;
  close: () => Promise<void>;
}

async function pickDirectory(title: string): Promise<string | null> {
  const { open } = await import("@tauri-apps/plugin-dialog");
  const picked = await open({ directory: true, multiple: false, title });
  return typeof picked === "string" ? picked : null;
}

export const useProjectStore = create<ProjectState>((set, get) => ({
  current: null,
  recent: [],
  templates: [],
  ready: false,
  showStartScreen: false,
  error: null,
  bootSource: null,

  refresh: async () => {
    try {
      const [templates, recent, current] = await Promise.all([
        projectIpc.templates(),
        projectIpc.recent(),
        projectIpc.current(),
      ]);
      set({ templates, recent, current, ready: true });
    } catch (e) {
      console.error("project.refresh failed", e);
      set({ ready: true });
    }
  },

  applyChanged: (info) => {
    set({ current: info, showStartScreen: false, error: null });
    // `project_recent` can now reject: a recent-projects list that exists but
    // cannot be read is an error rather than an empty list (C4-38/F14), so that
    // a corrupt one is never silently rewritten as empty. Without this catch the
    // new arm would surface as an unhandled rejection.
    void projectIpc
      .recent()
      .then((recent) => set({ recent }))
      .catch((e) => console.error("project.recent failed", e));
    // The asset DB re-rooted to the new project; re-sync the Content Drawer.
    void useAssetStore.getState().refresh();
    // ...and OPEN THE PROJECT'S BOOT LEVEL (wave GTA1).
    //
    // Opening a project used to leave the document alone, so a fresh `inf new`
    // put an author in a project whose level they had to go and find -- and the
    // scratch document they were left looking at is not in the project at all,
    // so a Ctrl+S would write it to `Levels/quicksave.inf_lvl`.
    //
    // The level is the cook's own root level (lowest guid), so what opens is
    // what a build would start in. UNSAVED WORK IS NEVER REPLACED: a dirty
    // document keeps the viewport and the offer becomes a status line, because
    // "open a project" is not consent to discard an edit.
    void openBootLevel();
  },

  setShowStartScreen: (v) => set({ showStartScreen: v, error: null }),

  bootDefault: async () => {
    try {
      const boot = await projectIpc.bootDefault();
      if (!boot) return null;
      // `project://changed` already applied the project; this line only says
      // WHICH rung chose it, because "your last project" and "the showcase the
      // engine found beside your checkout" are two different sentences for an
      // author who did not expect either.
      set({ bootSource: boot.source });
      useShellStore
        .getState()
        .pushStatus(`Opened ${boot.project.name} — ${boot.source}.`, 8000);
      return boot.source;
    } catch (e) {
      console.error("project.bootDefault failed", e);
      return null;
    }
  },

  setBootDefault: async (deliberate) => {
    try {
      // One store action for both directions, because they are one decision
      // seen from two sides and the row that calls them has to end up in the
      // same place either way: a phrase the BACKEND resolved.
      const phrase = deliberate
        ? await projectIpc.setDefault()
        : await projectIpc.clearDefault();
      set({ bootSource: phrase });
      useShellStore.getState().pushStatus(`On the next launch: ${phrase}.`, 8000);
    } catch (e) {
      console.error("project.setBootDefault failed", e);
      set({ error: String(e) });
    }
  },

  newProject: async (name, template, parentDir) => {
    if (!name.trim()) {
      set({ error: "Enter a project name." });
      return;
    }
    try {
      const info = await projectIpc.newProject(parentDir, name.trim(), template);
      get().applyChanged(info);
    } catch (e) {
      set({ error: String(e) });
    }
  },

  openProject: async (root) => {
    try {
      const info = await projectIpc.open(root);
      get().applyChanged(info);
    } catch (e) {
      set({ error: String(e) });
    }
  },

  openViaDialog: async () => {
    const dir = await pickDirectory("Open Infini Engine Project");
    if (dir) await get().openProject(dir);
  },

  newViaDialog: async (name, template) => {
    if (!name.trim()) {
      set({ error: "Enter a project name." });
      return;
    }
    const parent = await pickDirectory("Choose where to create the project");
    if (parent) await get().newProject(name, template, parent);
  },

  close: async () => {
    try {
      await projectIpc.close();
      set({ current: null, showStartScreen: false });
    } catch (e) {
      console.error("project.close failed", e);
    }
  },
}));

/**
 * Open the newly-opened project's boot level, unless the current document has
 * unsaved changes (in which case the author is TOLD rather than overruled).
 *
 * Separate from the store action so the "which level" question has exactly one
 * answer -- `project_boot_level`, which applies the cook's own rule to the same
 * asset database -- and so the dirty-document branch is readable.
 */
async function openBootLevel(): Promise<void> {
  let path: string | null;
  try {
    path = await projectIpc.bootLevel();
  } catch (e) {
    console.error("project.bootLevel failed", e);
    return;
  }
  if (!path) return; // A project with no level: the cook refuses it too.
  const scene = useSceneStore.getState();
  const name = path.split("/").pop() ?? path;
  if (scene.dirty) {
    useShellStore
      .getState()
      .pushStatus(
        `Project opened. ${name} was not opened because this level has unsaved changes — save, then File ▸ Open Level….`,
        12000,
      );
    return;
  }
  if (await scene.openLevel(path)) {
    useShellStore.getState().pushStatus(`Opened ${name}.`);
  }
}

/** Attach project handlers to the enumerated File-menu commands. */
export function registerProjectCommands(): void {
  const wire = (id: string, run: () => void | Promise<void>) => {
    if (getCommand(id)) setCommandHandler(id, run);
  };
  wire("file.newProject", () => useProjectStore.getState().setShowStartScreen(true));
  wire("file.openProject", () => useProjectStore.getState().openViaDialog());

  // Dynamic File ▸ Recent Projects entries. The submenu (built in MenuBar from
  // the live recent list) dispatches `file.recentProjects.{i}`; register one
  // handler per slot that opens the recent entry at that index by path, reusing
  // the standard open flow (start-screen guard + error surfacing).
  for (let i = 0; i < RECENT_PROJECTS_MAX; i++) {
    registerCommand({
      id: `file.recentProjects.${i}`,
      title: `Open Recent Project ${i + 1}`,
      category: "File",
      run: () => {
        const entry = useProjectStore.getState().recent[i];
        if (entry) void useProjectStore.getState().openProject(entry.path);
      },
    });
  }
}

/**
 * Load templates + recent + current and subscribe to `project://changed`.
 * Returns a disposer.
 *
 * **Refcounted** (`refCountedInit`, F-lens L7.M1): the guard used to be set
 * after the first `await`, so StrictMode's double mount subscribed twice and
 * orphaned the first handle.
 */
export const initProjectSync = refCountedInit(async (sink) => {
  const unlisten = await listenTo("project://changed", (info) =>
    useProjectStore.getState().applyChanged(info),
  );
  // The `refresh` below can reject, and without the sink that stranded the
  // subscription above for the life of the process (round-2 finding R2-7).
  sink(unlisten);
  await useProjectStore.getState().refresh();
  // ...AND OPEN THE APPLICATION'S BOOT PROJECT (wave CERT1).
  //
  // `refresh` sets `current` and does NOT open anything -- so a cold launch
  // used to land on the start screen with the showcase one file dialog away.
  // The backend resolves which project (`inf_project::boot::resolve`) and
  // opens it through the same `apply_open` the start screen uses, so the
  // `project://changed` subscribed above is what actually applies it.
  //
  // Only when nothing is open: a re-mount must not re-root a live session,
  // and the backend refuses that case too.
  if (useProjectStore.getState().current === null) {
    await useProjectStore.getState().bootDefault();
  }
  return () => unlisten();
});
