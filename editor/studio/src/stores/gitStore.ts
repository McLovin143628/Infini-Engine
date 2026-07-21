/**
 * Git status store (P5.4). Reflects the open project's working tree; every
 * mutation refreshes. No project / not-a-repo are represented, not errors.
 */
import { create } from "zustand";

import type { GitStatusDto } from "../bindings/GitStatusDto";
import { git } from "../lib/ipc";
import { useProjectStore } from "./projectStore";

interface GitState {
  status: GitStatusDto | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  stage: (paths: string[]) => Promise<void>;
  unstage: (paths: string[]) => Promise<void>;
  discard: (paths: string[]) => Promise<void>;
  commit: (message: string) => Promise<void>;
  init: () => Promise<void>;
}

function repoRoot(): string | null {
  return useProjectStore.getState().current?.root ?? null;
}

export const useGitStore = create<GitState>((set, get) => ({
  status: null,
  loading: false,
  error: null,

  refresh: async () => {
    const repo = repoRoot();
    if (!repo) {
      set({ status: null, error: null });
      return;
    }
    set({ loading: true });
    try {
      set({ status: await git.status(repo), error: null });
    } catch (e) {
      set({ error: String(e), status: null });
    } finally {
      set({ loading: false });
    }
  },

  stage: async (paths) => {
    const repo = repoRoot();
    if (repo) await git.stage(repo, paths);
    await get().refresh();
  },
  unstage: async (paths) => {
    const repo = repoRoot();
    if (repo) await git.unstage(repo, paths);
    await get().refresh();
  },
  discard: async (paths) => {
    const repo = repoRoot();
    if (repo) await git.discard(repo, paths);
    await get().refresh();
  },
  commit: async (message) => {
    const repo = repoRoot();
    if (repo && message.trim()) await git.commit(repo, message.trim());
    await get().refresh();
  },
  init: async () => {
    const repo = repoRoot();
    if (repo) await git.init(repo);
    await get().refresh();
  },
}));
