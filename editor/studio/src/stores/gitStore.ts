/**
 * Git status store (P5.4). Reflects the open project's working tree; every
 * mutation refreshes.
 *
 * # Every mutation answers whether it worked (round-2 finding R2.F10)
 *
 * `stage`/`unstage`/`discard`/`commit`/`init` each bare-`await`ed the IPC, the
 * `error` slot was written only by `refresh`, and every call site is
 * `void action(...)`. So a commit that failed — no `user.email` on a fresh
 * machine, a pre-commit hook, an `index.lock` — was an unhandled promise
 * rejection: nothing shown, the `await refresh()` after the throw never ran so
 * the file list did not even move, and `GitPanel` cleared the typed message
 * unconditionally. The user's evidence said it committed.
 *
 * Each mutation now returns `true` only if it really happened, records the
 * error, and refreshes either way (a failed `git` still leaves a working tree
 * worth re-reading). The panel keeps the message on a `false`.
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
  /** Each returns whether the operation actually happened. */
  stage: (paths: string[]) => Promise<boolean>;
  unstage: (paths: string[]) => Promise<boolean>;
  discard: (paths: string[]) => Promise<boolean>;
  commit: (message: string) => Promise<boolean>;
  init: () => Promise<boolean>;
  /** Dismiss the error banner. */
  clearError: () => void;
}

function repoRoot(): string | null {
  return useProjectStore.getState().current?.root ?? null;
}

/**
 * Run one mutation against the open repo: `true` only if it happened.
 *
 * The refresh runs on **both** paths. A `git` that failed part-way still moved
 * the working tree (a partial `add`, a hook that wrote files), and the old
 * shape's `await refresh()` sat after the throw — so a failure left the panel
 * showing a tree that no longer existed.
 */
async function run(
  set: (partial: Partial<GitState>) => void,
  get: () => GitState,
  op: (repo: string) => Promise<unknown>,
): Promise<boolean> {
  const repo = repoRoot();
  if (!repo) return false;
  let failure: string | null = null;
  try {
    await op(repo);
  } catch (e) {
    failure = String(e);
  }
  await get().refresh();
  // The message is captured BEFORE the refresh and restated after it: a
  // successful `git status` clears `error`, which would erase the only account
  // of why the mutation did not happen.
  set({ error: failure ?? get().error });
  return failure === null;
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

  stage: async (paths) => run(set, get, (repo) => git.stage(repo, paths)),
  unstage: async (paths) => run(set, get, (repo) => git.unstage(repo, paths)),
  discard: async (paths) => run(set, get, (repo) => git.discard(repo, paths)),
  commit: async (message) => {
    const text = message.trim();
    // An empty message is not a failure to report — the button is disabled for
    // it — but it is not a commit either, so it must not clear the box.
    if (!text) return false;
    return run(set, get, (repo) => git.commit(repo, text));
  },
  init: async () => run(set, get, (repo) => git.init(repo)),

  clearError: () => set({ error: null }),
}));
