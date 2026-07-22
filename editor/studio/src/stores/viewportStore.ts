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
import type { FoliageSettingsDto } from "../bindings/FoliageSettingsDto";
import type { GizmoModeDto } from "../bindings/GizmoModeDto";
import type { GizmoSpaceDto } from "../bindings/GizmoSpaceDto";
import type { SculptFalloffDto } from "../bindings/SculptFalloffDto";
import type { SculptOpDto } from "../bindings/SculptOpDto";
import type { SculptSettingsDto } from "../bindings/SculptSettingsDto";
import type { Snap2DDto } from "../bindings/Snap2DDto";
import type { Snap3DDto } from "../bindings/Snap3DDto";
import type { ToolModeDto } from "../bindings/ToolModeDto";
import type { ViewModeDto } from "../bindings/ViewModeDto";
import type { ViewportModeDto } from "../bindings/ViewportModeDto";

/** localStorage key for the persisted 3D gizmo snap settings (Wave 2). */
const SNAP3D_KEY = "inf.viewport.snap3d";

/** localStorage key for the persisted foliage brush settings (E-P6). */
const FOLIAGE_KEY = "inf.viewport.foliage";

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

  /** Shading view mode (Lit / Unlit / Wireframe), perspective only (R-P2). */
  viewMode: ViewModeDto;

  /** Active tool: pick/gizmo (`Select`) or terrain sculpt (`Sculpt`). (P10.2b) */
  toolMode: ToolModeDto;
  /** Sculpt brush operation. */
  sculptOp: SculptOpDto;
  /** Brush radius (world metres). */
  sculptRadius: number;
  /** Brush strength (metres/dab for Raise/Lower/Noise; blend for Smooth/Flatten; flow for Paint). */
  sculptStrength: number;
  /** Brush falloff curve. */
  sculptFalloff: SculptFalloffDto;
  /** Target splat layer `0..=3` for the Paint op (P10.4). */
  sculptPaintLayer: number;

  /** Foliage brush (E-P6). Radius (world m); density (instances per m²); erase
   * removes within the radius; kind = palette slot; scaleJitter = ± scale spread;
   * seed makes a stroke's scatter deterministic. */
  foliageRadius: number;
  foliageDensity: number;
  foliageErase: boolean;
  foliageKind: number;
  foliageScaleJitter: number;
  foliageSeed: number;

  /** Transform-gizmo mode (translate/rotate/scale), two-way synced (Wave 2). */
  gizmoMode: GizmoModeDto;
  /** Gizmo orientation frame (World ↔ Local). */
  gizmoSpace: GizmoSpaceDto;
  /** 3D snap: when enabled, drags always snap; else Shift-gated in the viewport. */
  snap3dEnabled: boolean;
  snap3dTranslate: number;
  snap3dRotate: number;
  snap3dScale: number;

  setMode: (mode: ViewportModeDto) => void;
  toggleMode: () => void;
  /** Set the shading view mode + push to the viewport (R-P2). */
  setViewMode: (mode: ViewModeDto) => void;
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
  setSculptPaintLayer: (n: number) => void;

  setFoliageRadius: (r: number) => void;
  setFoliageDensity: (d: number) => void;
  setFoliageErase: (v: boolean) => void;
  setFoliageKind: (n: number) => void;
  setFoliageScaleJitter: (v: number) => void;
  setFoliageSeed: (n: number) => void;

  /** Set the gizmo mode + push to the viewport (Wave 2). */
  setGizmoMode: (mode: GizmoModeDto) => void;
  /** Toggle World ↔ Local gizmo space + push. */
  toggleGizmoSpace: () => void;
  setGizmoSpace: (space: GizmoSpaceDto) => void;
  /** Enable/disable always-on 3D snap + persist + push. */
  setSnap3dEnabled: (v: boolean) => void;
  setSnap3dTranslate: (v: number) => void;
  setSnap3dRotate: (v: number) => void;
  setSnap3dScale: (v: number) => void;
}

/** Number of splat layers (matches inf_terrain::SPLAT_LAYERS / TERRAIN_LAYERS). */
export const SPLAT_LAYER_COUNT = 4;

/** Send the current sculpt brush settings to the native viewport. */
function pushSculpt(s: ViewportUiState): void {
  const dto: SculptSettingsDto = {
    op: s.sculptOp,
    radius: s.sculptRadius,
    strength: s.sculptStrength,
    falloff: s.sculptFalloff,
    paint_layer: s.sculptPaintLayer,
  };
  void viewport.setSculpt(dto).catch(() => {});
}

/** Build the foliage brush DTO from the store state (E-P6). `align_to_normal` is
 * off in v1 (yaw-only) — the seam carries it for the forthcoming normal-align. */
function foliageDto(s: ViewportUiState): FoliageSettingsDto {
  return {
    radius: s.foliageRadius,
    density: s.foliageDensity,
    erase: s.foliageErase,
    kind: s.foliageKind,
    scale_jitter: s.foliageScaleJitter,
    align_to_normal: false,
    seed: s.foliageSeed,
  };
}

/** Send + persist the current foliage brush settings (E-P6). */
function pushFoliage(s: ViewportUiState): void {
  const dto = foliageDto(s);
  void viewport.setFoliage(dto).catch(() => {});
  try {
    localStorage.setItem(FOLIAGE_KEY, JSON.stringify(dto));
  } catch {
    // ignore quota / privacy-mode failures
  }
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

/** Build the 3D snap DTO from the store state (Wave 2). */
function snap3dDto(s: ViewportUiState): Snap3DDto {
  return {
    translate: s.snap3dTranslate,
    rotate_deg: s.snap3dRotate,
    scale: s.snap3dScale,
    always_on: s.snap3dEnabled,
  };
}

/** Push the 3D snap settings to the viewport and persist them (Wave 2). */
function pushSnap3d(s: ViewportUiState): void {
  const dto = snap3dDto(s);
  void viewport.setSnap3d(dto).catch(() => {});
  try {
    localStorage.setItem(SNAP3D_KEY, JSON.stringify(dto));
  } catch {
    // ignore quota / privacy-mode failures
  }
}

export const useViewportStore = create<ViewportUiState>((set, get) => ({
  mode: "Perspective",
  gridSnapEnabled: false,
  gridSnapSize: 1,
  pixelSnapEnabled: false,
  pixelsPerUnit: 100,
  viewMode: "Lit",
  toolMode: "Select",
  sculptOp: "Raise",
  sculptRadius: 8,
  sculptStrength: 0.5,
  sculptFalloff: "Smooth",
  sculptPaintLayer: 0,
  foliageRadius: 3,
  foliageDensity: 0.4,
  foliageErase: false,
  foliageKind: 0,
  foliageScaleJitter: 0.2,
  foliageSeed: 1,
  gizmoMode: "Translate",
  gizmoSpace: "World",
  snap3dEnabled: false,
  snap3dTranslate: 1,
  snap3dRotate: 15,
  snap3dScale: 0.1,

  setMode: (mode) => {
    set({ mode });
    void viewport.setMode(mode).catch(() => {});
  },
  toggleMode: () => {
    get().setMode(get().mode === "TwoD" ? "Perspective" : "TwoD");
  },
  setViewMode: (viewMode) => {
    set({ viewMode });
    void viewport.setViewMode(viewMode).catch(() => {});
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
    // Sync the brush config whenever we enter a brush tool so it's armed.
    if (toolMode === "Sculpt") pushSculpt(get());
    if (toolMode === "Foliage") pushFoliage(get());
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
  setSculptPaintLayer: (n) => {
    const sculptPaintLayer = Number.isFinite(n)
      ? Math.min(Math.max(Math.round(n), 0), SPLAT_LAYER_COUNT - 1)
      : 0;
    set({ sculptPaintLayer });
    pushSculpt(get());
  },

  setFoliageRadius: (r) => {
    set({ foliageRadius: Number.isFinite(r) ? Math.max(r, 0.1) : 0.1 });
    pushFoliage(get());
  },
  setFoliageDensity: (d) => {
    set({ foliageDensity: Number.isFinite(d) ? Math.max(d, 0) : 0 });
    pushFoliage(get());
  },
  setFoliageErase: (foliageErase) => {
    set({ foliageErase });
    pushFoliage(get());
  },
  setFoliageKind: (n) => {
    set({ foliageKind: Number.isFinite(n) ? Math.max(Math.round(n), 0) : 0 });
    pushFoliage(get());
  },
  setFoliageScaleJitter: (v) => {
    set({ foliageScaleJitter: Number.isFinite(v) ? Math.min(Math.max(v, 0), 1) : 0 });
    pushFoliage(get());
  },
  setFoliageSeed: (n) => {
    set({ foliageSeed: Number.isFinite(n) ? Math.max(Math.round(n), 0) : 0 });
    pushFoliage(get());
  },

  setGizmoMode: (gizmoMode) => {
    set({ gizmoMode });
    void viewport.setGizmoMode(gizmoMode).catch(() => {});
  },
  setGizmoSpace: (gizmoSpace) => {
    set({ gizmoSpace });
    void viewport.setGizmoSpace(gizmoSpace).catch(() => {});
  },
  toggleGizmoSpace: () => {
    get().setGizmoSpace(get().gizmoSpace === "Local" ? "World" : "Local");
  },
  setSnap3dEnabled: (snap3dEnabled) => {
    set({ snap3dEnabled });
    pushSnap3d(get());
  },
  setSnap3dTranslate: (v) => {
    set({ snap3dTranslate: Number.isFinite(v) ? Math.max(v, 0) : 0 });
    pushSnap3d(get());
  },
  setSnap3dRotate: (v) => {
    set({ snap3dRotate: Number.isFinite(v) ? Math.max(v, 0) : 0 });
    pushSnap3d(get());
  },
  setSnap3dScale: (v) => {
    set({ snap3dScale: Number.isFinite(v) ? Math.max(v, 0) : 0 });
    pushSnap3d(get());
  },
}));

/** Register the View-menu/palette commands for the mode + tool toggles. */
export function registerViewportCommands(): void {
  registerCommands([
    { id: "view.toggle2D", title: "Toggle 2D / Perspective Viewport", category: "View" },
    { id: "view.perspective", title: "Viewport: Perspective", category: "View" },
    { id: "view.2d", title: "Viewport: 2D (Orthographic)", category: "View" },
    { id: "view.lit", title: "View Mode: Lit", category: "View" },
    { id: "view.unlit", title: "View Mode: Unlit", category: "View" },
    { id: "view.wireframe", title: "View Mode: Wireframe", category: "View" },
    { id: "tool.select", title: "Tool: Select", category: "Tools" },
    { id: "tool.sculpt", title: "Tool: Sculpt Terrain", category: "Tools" },
    { id: "tool.foliage", title: "Tool: Paint Foliage", category: "Tools" },
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
  if (getCommand("view.lit")) {
    setCommandHandler("view.lit", () => useViewportStore.getState().setViewMode("Lit"));
  }
  if (getCommand("view.unlit")) {
    setCommandHandler("view.unlit", () => useViewportStore.getState().setViewMode("Unlit"));
  }
  if (getCommand("view.wireframe")) {
    setCommandHandler("view.wireframe", () =>
      useViewportStore.getState().setViewMode("Wireframe"),
    );
  }
  if (getCommand("tool.select")) {
    setCommandHandler("tool.select", () => useViewportStore.getState().setToolMode("Select"));
  }
  if (getCommand("tool.sculpt")) {
    setCommandHandler("tool.sculpt", () => useViewportStore.getState().setToolMode("Sculpt"));
  }
  if (getCommand("tool.foliage")) {
    setCommandHandler("tool.foliage", () => useViewportStore.getState().setToolMode("Foliage"));
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

  // Restore the persisted 3D gizmo snap settings and push them to the viewport
  // (Wave 2). Malformed / absent storage falls back to the store defaults.
  try {
    const raw = localStorage.getItem(SNAP3D_KEY);
    if (raw) {
      const p = JSON.parse(raw) as Partial<Snap3DDto>;
      useViewportStore.setState({
        snap3dEnabled: typeof p.always_on === "boolean" ? p.always_on : false,
        snap3dTranslate: typeof p.translate === "number" ? p.translate : 1,
        snap3dRotate: typeof p.rotate_deg === "number" ? p.rotate_deg : 15,
        snap3dScale: typeof p.scale === "number" ? p.scale : 0.1,
      });
    }
  } catch {
    // ignore parse / access failures
  }
  void viewport.setSnap3d(snap3dDto(useViewportStore.getState())).catch(() => {});

  // Restore the persisted foliage brush settings (E-P6). Malformed / absent
  // storage falls back to the store defaults.
  try {
    const raw = localStorage.getItem(FOLIAGE_KEY);
    if (raw) {
      const p = JSON.parse(raw) as Partial<FoliageSettingsDto>;
      useViewportStore.setState({
        foliageRadius: typeof p.radius === "number" ? p.radius : 3,
        foliageDensity: typeof p.density === "number" ? p.density : 0.4,
        foliageErase: typeof p.erase === "boolean" ? p.erase : false,
        foliageKind: typeof p.kind === "number" ? p.kind : 0,
        foliageScaleJitter: typeof p.scale_jitter === "number" ? p.scale_jitter : 0.2,
        foliageSeed: typeof p.seed === "number" ? p.seed : 1,
      });
    }
  } catch {
    // ignore parse / access failures
  }
  void viewport.setFoliage(foliageDto(useViewportStore.getState())).catch(() => {});

  let disposed = false;
  const unlisteners: Array<() => void> = [];
  const track = (p: Promise<() => void>) => {
    p.then((fn) => {
      if (disposed) fn();
      else unlisteners.push(fn);
    }).catch(() => {});
  };
  track(listenTo("project://changed", () => reload()));
  // The viewport echoes gizmo-mode changes (W/E/R keypress or an IPC set); mirror
  // them into the store WITHOUT calling back into IPC (no loop).
  track(
    listenTo("viewport://gizmo", (mode) => {
      if (useViewportStore.getState().gizmoMode !== mode) {
        useViewportStore.setState({ gizmoMode: mode });
      }
    }),
  );
  return () => {
    disposed = true;
    for (const fn of unlisteners) fn();
  };
}
