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
import type { Snap2DDto } from "../bindings/Snap2DDto";
import type { ViewportModeDto } from "../bindings/ViewportModeDto";

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

  setMode: (mode: ViewportModeDto) => void;
  toggleMode: () => void;
  setGridSnapEnabled: (v: boolean) => void;
  setGridSnapSize: (v: number) => void;
  setPixelSnapEnabled: (v: boolean) => void;
  /** Update + persist the per-project pixels-per-unit. */
  setPixelsPerUnit: (v: number) => void;
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
}));

/** Register the View-menu/palette commands for the mode toggle (P8.2c). */
export function registerViewportCommands(): void {
  registerCommands([
    { id: "view.toggle2D", title: "Toggle 2D / Perspective Viewport", category: "View" },
    { id: "view.perspective", title: "Viewport: Perspective", category: "View" },
    { id: "view.2d", title: "Viewport: 2D (Orthographic)", category: "View" },
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
