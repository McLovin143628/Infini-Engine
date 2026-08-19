/**
 * The apply doors for the app-level editor preferences (Wave E, batch A).
 *
 * `stores/settingsStore.ts` is the *record*; this module is what makes a
 * preference **do** something. It is the only place that knows both the
 * settings file and the subsystems it drives, which is what keeps
 * `settingsStore` a leaf (`viewportStore` writes back through it without a
 * cycle).
 *
 * ## What applies live, and where
 *
 * | Setting | Door |
 * |---|---|
 * | `theme_id` | `lib/theme.ts` `setTheme` — repaints via CSS vars, no re-render |
 * | `keybindings` | `lib/keybindings.ts` `applyKeybindingOverrides` |
 * | `snap_3d` | `viewportStore.applySnap3dFromSettings` → `viewport_set_snap3d` |
 * | `foliage` | `viewportStore.applyFoliageFromSettings` → `viewport_set_foliage` |
 * | `autosave_interval_s` | `stores/autosave.ts` re-arms its interval |
 * | `tour_seen` | read by `stores/tourStore.ts` |
 * | camera / RMB feel | pushed to the native viewport (Wave E batch C) |
 *
 * ## The localStorage migration, and why it runs exactly once
 *
 * Before this wave the editor's preferences were ad-hoc `localStorage` keys.
 * The file is now authoritative, so on the **first** load after upgrading, the
 * old keys are folded in — otherwise the user's current theme is silently lost
 * the moment the file wins. "First" is recorded by its own marker key rather
 * than inferred from the settings' values, because a saved file that happens to
 * hold the defaults is indistinguishable from no file at all.
 *
 * `infinity:theme` keeps being written by `setTheme` afterwards, deliberately:
 * it is the **pre-paint cache** `main.tsx` reads before React mounts, so the
 * shell does not flash the default theme while the async load is in flight.
 * The other three legacy keys are removed once folded in.
 */
import type { EditorSettings } from "../bindings/EditorSettings";
import type { FoliageSettingsDto } from "../bindings/FoliageSettingsDto";
import type { Snap3DDto } from "../bindings/Snap3DDto";
import { editorSettings as settingsIpc, viewport as viewportIpc } from "./ipc";
import { applyKeybindingOverrides } from "./keybindings";
import { setTheme } from "./theme";
import {
  applyFoliageFromSettings,
  applySnap3dFromSettings,
} from "../stores/viewportStore";
import { DEFAULT_EDITOR_SETTINGS, useSettingsStore } from "../stores/settingsStore";

/** Marks that the one-time localStorage → file fold-in has happened. */
export const PREFS_MIGRATED_KEY = "infinity:prefsMigrated";
/** The legacy keys, named here and nowhere else. */
const LEGACY_THEME_KEY = "infinity:theme";
const LEGACY_TOUR_KEY = "infinity:tourSeen";
const LEGACY_SNAP3D_KEY = "inf.viewport.snap3d";
const LEGACY_FOLIAGE_KEY = "inf.viewport.foliage";

function readLocal(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function dropLocal(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    /* storage denied — the marker below still stops a second fold-in */
  }
}

function parseJson(raw: string | null): Record<string, unknown> | null {
  if (!raw) return null;
  try {
    const v: unknown = JSON.parse(raw);
    return typeof v === "object" && v !== null ? (v as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

const num = (v: unknown, fallback: number): number =>
  typeof v === "number" && Number.isFinite(v) ? v : fallback;
const bool = (v: unknown, fallback: boolean): boolean =>
  typeof v === "boolean" ? v : fallback;

/**
 * Fold the legacy localStorage preferences into `base`. Returns `null` when
 * there is nothing to do (already migrated, or no legacy value differed).
 *
 * Pure apart from reading storage, so the test can drive it directly.
 */
export function foldLegacyPreferences(base: EditorSettings): EditorSettings | null {
  if (readLocal(PREFS_MIGRATED_KEY) === "1") return null;

  const next: EditorSettings = { ...base };
  let changed = false;

  const theme = readLocal(LEGACY_THEME_KEY);
  if (theme && theme !== next.theme_id) {
    next.theme_id = theme;
    changed = true;
  }

  if (readLocal(LEGACY_TOUR_KEY) === "1" && !next.tour_seen) {
    next.tour_seen = true;
    changed = true;
  }

  const snap = parseJson(readLocal(LEGACY_SNAP3D_KEY));
  if (snap) {
    const d = DEFAULT_EDITOR_SETTINGS.snap_3d;
    const merged: Snap3DDto = {
      translate: num(snap.translate, d.translate),
      rotate_deg: num(snap.rotate_deg, d.rotate_deg),
      scale: num(snap.scale, d.scale),
      always_on: bool(snap.always_on, d.always_on),
    };
    next.snap_3d = merged;
    changed = true;
  }

  const foliage = parseJson(readLocal(LEGACY_FOLIAGE_KEY));
  if (foliage) {
    const d = DEFAULT_EDITOR_SETTINGS.foliage;
    const merged: FoliageSettingsDto = {
      radius: num(foliage.radius, d.radius),
      density: num(foliage.density, d.density),
      erase: bool(foliage.erase, d.erase),
      kind: num(foliage.kind, d.kind),
      scale_jitter: num(foliage.scale_jitter, d.scale_jitter),
      align_to_normal: bool(foliage.align_to_normal, d.align_to_normal),
      seed: num(foliage.seed, d.seed),
    };
    next.foliage = merged;
    changed = true;
  }

  // The marker is set even when nothing differed: a fresh install has no legacy
  // keys and must not re-scan every launch.
  try {
    localStorage.setItem(PREFS_MIGRATED_KEY, "1");
  } catch {
    /* storage denied — the fold-in is idempotent anyway (same values in) */
  }
  // `infinity:theme` STAYS (the pre-paint cache); the rest are done.
  dropLocal(LEGACY_TOUR_KEY);
  dropLocal(LEGACY_SNAP3D_KEY);
  dropLocal(LEGACY_FOLIAGE_KEY);

  return changed ? next : null;
}

/**
 * Push `next` into the subsystems it drives. `prev` is `null` on the first
 * apply (everything is applied); afterwards only what changed is touched, which
 * is what stops `viewportStore → settingsStore → here → viewportStore` from
 * chasing its own tail.
 */
export function applySettings(next: EditorSettings, prev: EditorSettings | null): void {
  if (!prev || prev.theme_id !== next.theme_id) setTheme(next.theme_id);
  if (!prev || !sameMap(prev.keybindings, next.keybindings)) {
    applyKeybindingOverrides(next.keybindings);
  }
  // Compared BY VALUE, not by reference: `viewportStore` patches a freshly
  // built DTO on every snap change, so a reference test would push the same
  // numbers back at the native viewport on every keystroke.
  if (!prev || !sameSnap(prev.snap_3d, next.snap_3d)) applySnap3dFromSettings(next.snap_3d);
  if (!prev || !sameFoliage(prev.foliage, next.foliage)) applyFoliageFromSettings(next.foliage);
  // The pointer feel reaches the native input state machine (Wave E batch C):
  // flycam speed, look sensitivity, and the right-button click-vs-drag
  // thresholds the context-menu gesture is discriminated on.
  if (!prev || !sameInteraction(prev, next)) {
    void viewportIpc
      .setInteraction({
        fly_speed_mps: next.camera_fly_speed_mps,
        look_sensitivity: next.camera_look_sensitivity,
        rmb_click_travel_px: next.rmb_click_travel_px,
        rmb_click_ms: next.rmb_click_ms,
      })
      .catch(() => {
        // No native viewport (headless test, detached window) — nothing to feel.
      });
  }
}

const sameInteraction = (a: EditorSettings, b: EditorSettings): boolean =>
  a.camera_fly_speed_mps === b.camera_fly_speed_mps &&
  a.camera_look_sensitivity === b.camera_look_sensitivity &&
  a.rmb_click_travel_px === b.rmb_click_travel_px &&
  a.rmb_click_ms === b.rmb_click_ms;

const sameSnap = (a: Snap3DDto, b: Snap3DDto): boolean =>
  a.translate === b.translate &&
  a.rotate_deg === b.rotate_deg &&
  a.scale === b.scale &&
  a.always_on === b.always_on;

const sameFoliage = (a: FoliageSettingsDto, b: FoliageSettingsDto): boolean =>
  a.radius === b.radius &&
  a.density === b.density &&
  a.erase === b.erase &&
  a.kind === b.kind &&
  a.scale_jitter === b.scale_jitter &&
  a.align_to_normal === b.align_to_normal &&
  a.seed === b.seed;

const sameMap = (a: Record<string, string>, b: Record<string, string>): boolean => {
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => a[k] === b[k]);
};

/**
 * Load the settings file, migrate the legacy keys once, apply everything, and
 * keep applying on every later change. Returns a disposer (StrictMode-safe,
 * matching the other `init*Sync` helpers).
 *
 * A **failed** load does not write: the corrupt file is left for the user to
 * repair, the in-memory defaults stay in force and the message is surfaced by
 * the Preferences dialog.
 */
export function initSettingsSync(): () => void {
  const store = useSettingsStore.getState();
  let last: EditorSettings | null = null;
  const unsub = useSettingsStore.subscribe((s) => {
    if (s.settings === last) return;
    const prev = last;
    last = s.settings;
    applySettings(s.settings, prev);
  });

  void settingsIpc
    .get()
    .then((loaded) => {
      const folded = foldLegacyPreferences(loaded);
      if (folded) {
        // One write, and only when something actually moved.
        store.hydrate(folded);
        void settingsIpc
          .set(folded)
          .then((saved) => store.hydrate(saved))
          .catch((e: unknown) => store.setError(String(e)));
      } else {
        store.hydrate(loaded);
      }
    })
    .catch((e: unknown) => {
      // Corrupt / unavailable: keep the defaults, say so, write NOTHING.
      useSettingsStore.setState({ loaded: true, error: String(e) });
      applySettings(useSettingsStore.getState().settings, null);
    });

  return unsub;
}
