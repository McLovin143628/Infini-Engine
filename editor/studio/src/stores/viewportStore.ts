/**
 * Viewport UI state (P8.2c): the active projection (Perspective ↔ 2D ortho) and
 * the 2D-mode grid/pixel snapping settings. Each setter pushes the change to the
 * native viewport over the typed IPC channel (`viewport.setMode` /
 * `viewport.setSnap2d`); the per-project pixels-per-unit persists through
 * `projectSettings` (see stores/index.ts for the zustand conventions).
 */
import { create } from "zustand";

import { getCommand, registerCommands, setCommandHandler } from "../lib/commands";
import { listenTo } from "../lib/events";
import { projectSettings, viewport } from "../lib/ipc";
import type { SculptFalloffDto } from "../bindings/SculptFalloffDto";
import type { SculptOpDto } from "../bindings/SculptOpDto";
import type { SculptSettingsDto } from "../bindings/SculptSettingsDto";
import type { Snap2DDto } from "../bindings/Snap2DDto";
import type { ToolModeDto } from "../bindings/ToolModeDto";
import type { ViewportModeDto } from "../bindings/ViewportModeDto";

/** Radius clamp (world metres) and the multiplicative `[` / `]` nudge factor. */
export const SCULPT_RADIUS_MIN = 0.5;
export const SCULPT_RADIUS_MAX = 4096;
const SCULPT_RADIUS_NUDGE = 1.25;

/** Clamp a sculpt radius into the supported range. */
export function clampSculptRadius(r: number): number {
  if (!Number.isFinite(r)) return SCULPT_RADIUS_MIN;
  return Math.min(Math.max(r, SCULPT_RADIUS_MIN), SCULPT_RADIUS_MAX);
}

interface ViewportUiState {
  /** Active projection. */
  mode: ViewportModeDto;
  /** Snap a 2D translate to `gridSnapSize` world units. */
  gridSnapEnabled: boolean;
  gridSnapSize: number;
  /** Snap a 2D translate to `1/pixelsPerUnit` world units (finer; wins). */
  pixelSnapEnabled: boolean;
  /** Per-project pixels-per-unit (persisted). */
  pixelsPerUnit: number;

  /** Active tool: pick/gizmo (`Select`) or terrain sculpt (`Sculpt`). (P10.2b) */
  toolMode: ToolModeDto;
  /** Sculpt brush operation. */
  sculptOp: SculptOpDto;
  /** Brush radius (world metres). */
  sculptRadius: number;
  /** Brush strength (metres/dab for Raise/Lower/Noise; blend for Smooth/Flatten). */
  sculptStrength: number;
  /** Brush falloff curve. */
  sculptFalloff: SculptFalloffDto;

  setMode: (mode: ViewportModeDto) => void;
  toggleMode: () => void;
  setGridSnapEnabled: (v: boolean) => void;
  setGridSnapSize: (v: number) => void;
  setPixelSnapEnabled: (v: boolean) => void;
  /** Update + persist the per-project pixels-per-unit. */
  setPixelsPerUnit: (v: number) => void;

  setToolMode: (mode: ToolModeDto) => void;
  setSculptOp: (op: SculptOpDto) => void;
  setSculptRadius: (r: number) => void;
  /** Multiplicatively grow (`+1`) or shrink (`-1`) the radius — the `[`/`]` keys. */
  nudgeSculptRadius: (dir: number) => void;
  setSculptStrength: (s: number) => void;
  setSculptFalloff: (f: SculptFalloffDto) => void;
}

/** Send the current sculpt brush settings to the native viewport. */
function pushSculpt(s: ViewportUiState): void {
  const dto: SculptSettingsDto = {
    op: s.sculptOp,
    radius: s.sculptRadius,
    strength: s.sculptStrength,
    falloff: s.sculptFalloff,
  };
  void viewport.setSculpt(dto).catch(() => {});
}

/** Send the current snap settings to the native viewport. */
function pushSnap(s: ViewportUiState): void {
  const dto: Snap2DDto = {
    grid_enabled: s.gridSnapEnabled,
    grid_size: s.gridSnapSize,
    pixel_enabled: s.pixelSnapEnabled,
    pixels_per_unit: s.pixelsPerUnit,
  };
  void viewport.setSnap2d(dto).catch(() => {});
}

export const useViewportStore = create<ViewportUiState>((set, get) => ({
  mode: "Perspective",
  gridSnapEnabled: false,
  gridSnapSize: 1,
  pixelSnapEnabled: false,
  pixelsPerUnit: 100,
  toolMode: "Select",
  sculptOp: "Raise",
  sculptRadius: 8,
  sculptStrength: 0.5,
  sculptFalloff: "Smooth",

  setMode: (mode) => {
    set({ mode });
    void viewport.setMode(mode).catch(() => {});
  },
  toggleMode: () => {
    get().setMode(get().mode === "TwoD" ? "Perspective" : "TwoD");
  },
  setGridSnapEnabled: (gridSnapEnabled) => {
    set({ gridSnapEnabled });
    pushSnap(get());
  },
  setGridSnapSize: (gridSnapSize) => {
    set({ gridSnapSize: Math.max(gridSnapSize, 0) });
    pushSnap(get());
  },
  setPixelSnapEnabled: (pixelSnapEnabled) => {
    set({ pixelSnapEnabled });
    pushSnap(get());
  },
  setPixelsPerUnit: (v) => {
    const pixelsPerUnit = v > 0 ? v : 100;
    set({ pixelsPerUnit });
    pushSnap(get());
    void projectSettings.set({ pixels_per_unit: pixelsPerUnit }).catch(() => {});
  },

  setToolMode: (toolMode) => {
    set({ toolMode });
    void viewport.setToolMode(toolMode).catch(() => {});
    // Sync the brush config whenever we enter Sculpt so the viewport is armed.
    if (toolMode === "Sculpt") pushSculpt(get());
  },
  setSculptOp: (sculptOp) => {
    set({ sculptOp });
    pushSculpt(get());
  },
  setSculptRadius: (r) => {
    set({ sculptRadius: clampSculptRadius(r) });
    pushSculpt(get());
  },
  nudgeSculptRadius: (dir) => {
    const factor = dir >= 0 ? SCULPT_RADIUS_NUDGE : 1 / SCULPT_RADIUS_NUDGE;
    set({ sculptRadius: clampSculptRadius(get().sculptRadius * factor) });
    pushSculpt(get());
  },
  setSculptStrength: (s) => {
    set({ sculptStrength: Number.isFinite(s) ? Math.max(s, 0) : 0 });
    pushSculpt(get());
  },
  setSculptFalloff: (sculptFalloff) => {
    set({ sculptFalloff });
    pushSculpt(get());
  },
}));

/** Register the View-menu/palette commands for the mode + tool toggles. */
export function registerViewportCommands(): void {
  registerCommands([
    { id: "view.toggle2D", title: "Toggle 2D / Perspective Viewport", category: "View" },
    { id: "view.perspective", title: "Viewport: Perspective", category: "View" },
    { id: "view.2d", title: "Viewport: 2D (Orthographic)", category: "View" },
    { id: "tool.select", title: "Tool: Select", category: "Tools" },
    { id: "tool.sculpt", title: "Tool: Sculpt Terrain", category: "Tools" },
  ]);
  if (getCommand("view.toggle2D")) {
    setCommandHandler("view.toggle2D", () => useViewportStore.getState().toggleMode());
  }
  if (getCommand("view.perspective")) {
    setCommandHandler("view.perspective", () => useViewportStore.getState().setMode("Perspective"));
  }
  if (getCommand("view.2d")) {
    setCommandHandler("view.2d", () => useViewportStore.getState().setMode("TwoD"));
  }
  if (getCommand("tool.select")) {
    setCommandHandler("tool.select", () => useViewportStore.getState().setToolMode("Select"));
  }
  if (getCommand("tool.sculpt")) {
    setCommandHandler("tool.sculpt", () => useViewportStore.getState().setToolMode("Sculpt"));
  }
}

/**
 * `[` / `]` adjust the sculpt radius while the Sculpt tool is active and the
 * webview has focus (a plain-key global shortcut). NOTE: when the native
 * viewport child window holds OS focus it only forwards Ctrl/F-key chords
 * (win32.rs `vk_name`), so bracket keys pressed *over the viewport* do not reach
 * here — radius is then adjusted from the toolbar. Widening the forwarded-key set
 * to include the brackets is the documented follow-up. Returns a disposer.
 */
export function initSculptKeybindings(): () => void {
  const onKey = (e: KeyboardEvent) => {
    if (e.key !== "[" && e.key !== "]") return;
    if (useViewportStore.getState().toolMode !== "Sculpt") return;
    // Ignore while typing in a field.
    const t = e.target as HTMLElement | null;
    const tag = t?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || t?.isContentEditable) return;
    e.preventDefault();
    useViewportStore.getState().nudgeSculptRadius(e.key === "]" ? 1 : -1);
  };
  window.addEventListener("keydown", onKey);
  return () => window.removeEventListener("keydown", onKey);
}

/**
 * Load the per-project pixels-per-unit and apply the current snap settings to
 * the viewport, on boot and whenever the open project changes. Returns a
 * disposer (StrictMode-safe, matching the other store sync helpers).
 */
export function initViewportSync(): () => void {
  const reload = () => {
    projectSettings
      .get()
      .then((s) => {
        useViewportStore.setState({ pixelsPerUnit: s.pixels_per_unit });
        pushSnap(useViewportStore.getState());
      })
      .catch(() => {});
  };
  reload();

  let disposed = false;
  let unlisten: (() => void) | undefined;
  listenTo("project://changed", () => reload()).then((fn) => {
    if (disposed) fn();
    else unlisten = fn;
  });
  return () => {
    disposed = true;
    unlisten?.();
  };
}
