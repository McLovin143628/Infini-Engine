/**
 * First-run tour (P15.3): a dismissable, step-by-step overlay that introduces
 * the shell's core panels. The state machine lives here (headless-testable);
 * `shell/FirstRunTour.tsx` renders it.
 *
 * Persistence is the app-level settings file (`tour_seen` in
 * `editor-settings.toml`) since Wave E; the old `infinity:tourSeen`
 * localStorage key is folded in once by `lib/settingsApply.ts`. No implicit
 * persist middleware, so what's saved (and when) stays auditable per the
 * `stores/index.ts` convention.
 *
 * When does it run? It DEFERS until the first project is open: before that the
 * StartScreen overlay covers the shell and none of the tour's anchors
 * (Outliner, Details, viewport hole, Content Drawer) are meaningful. On the
 * first project open with the flag unset, `maybeAutostart()` fires it once.
 */
import { create } from "zustand";

import { useSettingsStore } from "./settingsStore";

export interface TourStep {
  id: string;
  title: string;
  body: string;
  /**
   * CSS selector for the element the callout points at (highlighted with a
   * ring). `null` centers the callout (welcome / closing steps).
   */
  anchor: string | null;
  /** Preferred callout placement relative to the anchor. */
  placement: "top" | "bottom" | "left" | "right" | "center";
}

/** The tour script — 7 steps across the shell's core surfaces. */
export const TOUR_STEPS: TourStep[] = [
  {
    id: "welcome",
    title: "Welcome to Infinity Engine",
    body: "A quick tour of the core panels. You can skip any time — this only shows on first launch. Reopen it from Help ▸ Interactive Tour.",
    anchor: null,
    placement: "center",
  },
  {
    id: "viewport",
    title: "The Viewport",
    body: "Your scene renders here in a native wgpu window. RMB + WASD to fly, Alt to orbit, F to focus, click to select. Gizmos translate/rotate/scale the selection.",
    anchor: "[data-viewport-hole]",
    placement: "left",
  },
  {
    id: "outliner",
    title: "The Outliner",
    body: "Every actor in the level, as a live tree. Add via the + menu, double-click to rename, drag to reparent, and toggle visibility with the eye. It stays in sync with the viewport selection.",
    anchor: '[data-tab-id="outliner"]',
    placement: "left",
  },
  {
    id: "details",
    title: "The Details panel",
    body: "Reflection-driven properties for the selection — Transform, Material, Light, and any Blueprint variables. Edits are undoable (Ctrl+Z) and support multi-object editing.",
    anchor: '[data-tab-id="details"]',
    placement: "left",
  },
  {
    id: "contentDrawer",
    title: "The Content Drawer",
    body: "Slide up your assets (or press Ctrl+Space): meshes, materials, textures, blueprints and more. Import with drag-and-drop, then drag an asset into the viewport to place it.",
    anchor: '[data-tour="content-drawer"]',
    placement: "top",
  },
  {
    id: "play",
    title: "Play & Simulate",
    body: "Play-in-Editor runs your game in a crash-isolated subprocess (Shift+Alt+P). The dropdown picks Embedded, New Window, or in-process Simulate. Pause, Step, and Stop drive the live session.",
    anchor: '[data-tour="play-cluster"]',
    placement: "bottom",
  },
  {
    id: "commandPalette",
    title: "The Command Palette",
    body: "Press Ctrl+Shift+P for fuzzy access to every menu action and command — the fastest way to reach anything in the editor.",
    anchor: '[data-tour="command-palette"]',
    placement: "top",
  },
];

/**
 * Has the user already seen (or dismissed) the first-run tour?
 *
 * Since Wave E the flag lives in the app-level settings FILE, not in
 * `localStorage` (the old `infinity:tourSeen` key is folded in once by
 * `lib/settingsApply.ts`). **Unloaded settings read as "seen"**: the tour is
 * suppressed until the file has answered, so a slow load can never flash the
 * tour at someone who dismissed it a year ago. `maybeAutostart` is re-checked
 * when the settings load, so the genuine first run still gets it.
 */
export function tourSeen(): boolean {
  const s = useSettingsStore.getState();
  return !s.loaded || s.settings.tour_seen;
}

function markSeen(): void {
  useSettingsStore.getState().patch({ tour_seen: true });
}

interface TourState {
  active: boolean;
  step: number;
  /** Force-start at step 0 (Help ▸ Interactive Tour), regardless of the flag. */
  start: () => void;
  /** Start once if never seen (first project open). No-op otherwise. */
  maybeAutostart: () => void;
  next: () => void;
  prev: () => void;
  /** Dismiss early; marks seen. */
  skip: () => void;
  /** Complete the tour; marks seen. */
  finish: () => void;
}

export const useTourStore = create<TourState>((set, get) => ({
  active: false,
  step: 0,

  start: () => set({ active: true, step: 0 }),

  maybeAutostart: () => {
    if (get().active || tourSeen()) return;
    set({ active: true, step: 0 });
  },

  next: () => {
    const { step } = get();
    if (step >= TOUR_STEPS.length - 1) {
      get().finish();
      return;
    }
    set({ step: step + 1 });
  },

  prev: () => set((s) => ({ step: Math.max(0, s.step - 1) })),

  skip: () => {
    markSeen();
    set({ active: false, step: 0 });
  },

  finish: () => {
    markSeen();
    set({ active: false, step: 0 });
  },
}));

/** Test hook: reset the store and clear the persisted seen flag. */
export function __resetTourForTest(): void {
  useSettingsStore.setState((s) => ({
    loaded: true,
    settings: { ...s.settings, tour_seen: false },
  }));
  useTourStore.setState({ active: false, step: 0 });
}
