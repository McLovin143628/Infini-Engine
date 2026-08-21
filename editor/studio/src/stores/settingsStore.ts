/**
 * App-level editor preferences (Wave E, batch A) — the frontend half of
 * `<app config>/editor-settings.toml` (`inf_editor_core::editor_settings`).
 *
 * This module is deliberately **leaf**: it imports the typed IPC wrapper and
 * nothing else from the app. The *apply doors* — theme, keybindings, autosave
 * period, the viewport brush defaults — live in `lib/settingsApply.ts`, which
 * subscribes to this store. That is what keeps the graph acyclic while letting
 * `viewportStore` write its snap/foliage settings back through `patch`.
 *
 * Two rules the backend owns and this store obeys:
 *
 * 1. **The reply is the truth.** `set` returns the normalized settings, so a
 *    clamped or NaN-guarded value reaches the UI instead of the UI and the file
 *    disagreeing. `patch` hydrates from the reply.
 * 2. **Corrupt is refused.** A failed `get` leaves `settings` at the defaults
 *    *and sets `error`*; it does NOT write anything back, so the unreadable file
 *    survives for the user to repair. Only an explicit later edit overwrites it.
 */
import { create } from "zustand";

import type { EditorSettings } from "../bindings/EditorSettings";
import { editorSettings as settingsIpc } from "../lib/ipc";

/**
 * The frontend's copy of `EditorSettings::default()`. Kept in step with the
 * Rust defaults by `settingsStore.test.ts`'s doc note and used only until the
 * first `get` resolves — the file is always the authority once loaded.
 */
export const DEFAULT_EDITOR_SETTINGS: EditorSettings = {
  schema_version: 1,
  theme_id: "infinity-dark",
  tour_seen: false,
  autosave_interval_s: 5,
  camera_fly_speed_mps: 8,
  camera_look_sensitivity: 1,
  rmb_click_travel_px: 4,
  rmb_click_ms: 250,
  // IB-14: the GIS vector-import entity cap, remembered per user.
  gis_max_entities: 4096,
  snap_3d: { translate: 1, rotate_deg: 15, scale: 0.1, always_on: false },
  foliage: {
    radius: 3,
    density: 0.4,
    erase: false,
    kind: 0,
    scale_jitter: 0.2,
    align_to_normal: false,
    seed: 1,
  },
  keybindings: {},
  // Island wave I5: the GAME's binding overrides, in the same format the shipped
  // player writes. Only what differs from the shipped table lives here.
  game_bindings: {},
};

/** How long `patch` waits before writing, so a dragged slider writes once. */
export const SETTINGS_WRITE_DEBOUNCE_MS = 300;

interface SettingsState {
  settings: EditorSettings;
  /** True once a `get` has resolved (or failed) — the tour waits on this. */
  loaded: boolean;
  /** Last load/save failure, surfaced by the Preferences dialog. */
  error: string | null;
  /** Replace in memory WITHOUT writing (the load path). */
  hydrate: (settings: EditorSettings, error?: string | null) => void;
  /** Merge + schedule a write. The merged value is live immediately. */
  patch: (partial: Partial<EditorSettings>) => EditorSettings;
  /** Restore every preference to the shipped defaults (writes immediately). */
  resetToDefaults: () => Promise<void>;
  /** Await any pending debounced write (tests + "Close" on the dialog). */
  flush: () => Promise<void>;
  setError: (error: string | null) => void;
}

let writeTimer: ReturnType<typeof setTimeout> | undefined;
let pending: Promise<void> | null = null;

/**
 * Cancel a scheduled write and drop the promise it was going to settle.
 *
 * **Both halves, always** (Wave E audit, A4): clearing the timer without
 * dropping `pending` leaves a promise nobody will ever resolve, and the next
 * `flush()` — the Preferences dialog's Close button — takes its `await pending`
 * branch and never returns. Nothing throws; the await simply hangs.
 */
function cancelScheduledWrite(): void {
  if (writeTimer !== undefined) clearTimeout(writeTimer);
  writeTimer = undefined;
  pending = null;
}

/**
 * Call the backend, converting a SYNCHRONOUS throw into a rejection.
 *
 * `invoke` is not available outside a Tauri webview (headless tests, a detached
 * window during teardown) and can throw before it ever returns a promise. A
 * throw inside the debounce's `setTimeout` callback has nobody to catch it, so
 * it is turned into a value here — the same reason refusals are values.
 */
function writeSettings(next: EditorSettings): Promise<EditorSettings> {
  try {
    return settingsIpc.set(next);
  } catch (e) {
    return Promise.reject(e instanceof Error ? e : new Error(String(e)));
  }
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  settings: DEFAULT_EDITOR_SETTINGS,
  loaded: false,
  error: null,

  hydrate: (settings, error = null) => set({ settings, loaded: true, error }),
  setError: (error) => set({ error }),

  patch: (partial) => {
    const settings = { ...get().settings, ...partial };
    set({ settings });
    if (writeTimer !== undefined) clearTimeout(writeTimer);
    pending = new Promise<void>((resolve) => {
      writeTimer = setTimeout(() => {
        writeTimer = undefined;
        // Always write the LATEST value, not the one captured at schedule time.
        writeSettings(get().settings)
          .then((saved) => set({ settings: saved, error: null }))
          .catch((e: unknown) => set({ error: String(e) }))
          .finally(() => {
            pending = null;
            resolve();
          });
      }, SETTINGS_WRITE_DEBOUNCE_MS);
    });
    return settings;
  },

  resetToDefaults: async () => {
    cancelScheduledWrite();
    set({ settings: DEFAULT_EDITOR_SETTINGS });
    try {
      const saved = await writeSettings(DEFAULT_EDITOR_SETTINGS);
      set({ settings: saved, error: null });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  flush: async () => {
    if (writeTimer !== undefined) {
      cancelScheduledWrite();
      try {
        const saved = await writeSettings(get().settings);
        set({ settings: saved, error: null });
      } catch (e) {
        set({ error: String(e) });
      }
      return;
    }
    if (pending) await pending;
  },
}));

/** Test-only: drop the debounce timer so cases cannot leak into each other. */
export function __resetSettingsStoreForTest(): void {
  cancelScheduledWrite();
  useSettingsStore.setState({
    settings: DEFAULT_EDITOR_SETTINGS,
    loaded: false,
    error: null,
  });
}
