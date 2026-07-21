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
import { getCommand, setCommandHandler } from "../lib/commands";
import { listenTo, type UnlistenFn } from "../lib/events";
import { project as projectIpc } from "../lib/ipc";
import { useAssetStore } from "./assetStore";

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

  refresh: () => Promise<void>;
  applyChanged: (info: ProjectInfoDto) => void;
  setShowStartScreen: (v: boolean) => void;

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
    void projectIpc.recent().then((recent) => set({ recent }));
    // The asset DB re-rooted to the new project; re-sync the Content Drawer.
    void useAssetStore.getState().refresh();
  },

  setShowStartScreen: (v) => set({ showStartScreen: v, error: null }),

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
    const dir = await pickDirectory("Open Infinity Engine Project");
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

/** Attach project handlers to the enumerated File-menu commands. */
export function registerProjectCommands(): void {
  const wire = (id: string, run: () => void | Promise<void>) => {
    if (getCommand(id)) setCommandHandler(id, run);
  };
  wire("file.newProject", () => useProjectStore.getState().setShowStartScreen(true));
  wire("file.openProject", () => useProjectStore.getState().openViaDialog());
}

let unlisten: UnlistenFn | null = null;

/**
 * Load templates + recent + current and subscribe to `project://changed`.
 * Idempotent (StrictMode-safe); returns a disposer.
 */
export async function initProjectSync(): Promise<() => void> {
  if (unlisten) return () => {};
  unlisten = await listenTo("project://changed", (info) =>
    useProjectStore.getState().applyChanged(info),
  );
  await useProjectStore.getState().refresh();
  const dispose = unlisten;
  return () => {
    dispose?.();
    unlisten = null;
  };
}
