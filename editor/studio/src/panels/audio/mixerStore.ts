/**
 * Audio Mixer panel store (E-P9): the working draft + last-saved baseline for
 * the project's `inf_audio::MixerConfig`.
 *
 * State lives in a zustand store (not component `useState`) so the async load /
 * save actions can call `set` without tripping the `set-state-in-effect` lint —
 * the same pattern as `gitStore` / `sequencerStore`. The panel subscribes to
 * slices and calls actions; dirty is derived from `draft` vs `saved`.
 */
import { create } from "zustand";

import type { MixerConfigDto } from "../../bindings/MixerConfigDto";
import { audio as audioIpc } from "../../lib/ipc";
import { validateMixer } from "./mixerModel";

function eq(a: MixerConfigDto | null, b: MixerConfigDto | null): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

interface MixerStore {
  /** The last config loaded from / written to disk (dirty baseline). */
  saved: MixerConfigDto | null;
  /** The live working copy the panel edits. */
  draft: MixerConfigDto | null;
  /** Last load/save error message, or null. */
  error: string | null;
  /** A save is in flight. */
  busy: boolean;

  /**
   * (Re)load from the backend. `force` replaces the working draft outright (mount
   * / project switch); otherwise a fresh load only adopts the draft when the user
   * has no unsaved edits, so an `audio://mixer-changed` echo of our own save
   * doesn't clobber ongoing typing.
   */
  load: (force: boolean) => Promise<void>;
  /** Apply a pure edit to the draft. */
  mutate: (fn: (cfg: MixerConfigDto) => MixerConfigDto) => void;
  /** Validate + persist the draft (no-op when clean/invalid); updates `saved`. */
  save: () => Promise<void>;
  /** Discard edits back to the saved baseline. */
  revert: () => void;
}

export const useMixerStore = create<MixerStore>((set, get) => ({
  saved: null,
  draft: null,
  error: null,
  busy: false,

  load: async (force) => {
    try {
      const cfg = await audioIpc.mixerGet();
      const dirty = !eq(get().draft, get().saved);
      set(force || !dirty ? { saved: cfg, draft: cfg, error: null } : { saved: cfg, error: null });
    } catch (e) {
      set({ saved: null, draft: null, error: String(e) });
    }
  },

  mutate: (fn) => set((s) => (s.draft ? { draft: fn(s.draft) } : {})),

  save: async () => {
    const draft = get().draft;
    if (!draft || validateMixer(draft) !== null) return;
    set({ busy: true });
    try {
      await audioIpc.mixerSave(draft);
      set({ saved: draft, error: null });
    } catch (e) {
      set({ error: String(e) });
    } finally {
      set({ busy: false });
    }
  },

  revert: () => set((s) => ({ draft: s.saved, error: null })),
}));

/** Whether the draft differs from the saved baseline (call as a selector). */
export function isDirty(s: Pick<MixerStore, "draft" | "saved">): boolean {
  return !eq(s.draft, s.saved);
}
