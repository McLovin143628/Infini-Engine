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
import { projectSettings, terrain, viewport, water } from "../lib/ipc";
import { isPrimaryViewport } from "../lib/viewportIds";
import type { BiomeSettingsDto } from "../bindings/BiomeSettingsDto";
import type { FoliageSettingsDto } from "../bindings/FoliageSettingsDto";
import type { GizmoModeDto } from "../bindings/GizmoModeDto";
import type { GizmoSpaceDto } from "../bindings/GizmoSpaceDto";
import type { SculptFalloffDto } from "../bindings/SculptFalloffDto";
import type { SculptOpDto } from "../bindings/SculptOpDto";
import type { SculptSettingsDto } from "../bindings/SculptSettingsDto";
import type { Snap2DDto } from "../bindings/Snap2DDto";
import type { Snap3DDto } from "../bindings/Snap3DDto";
import type { TerrainBiomesDto } from "../bindings/TerrainBiomesDto";
import type { RiverReportDto } from "../bindings/RiverReportDto";
import type { ToolModeDto } from "../bindings/ToolModeDto";
import type { WaterSettingsDto } from "../bindings/WaterSettingsDto";
import type { WaterToolKindDto } from "../bindings/WaterToolKindDto";
import type { SpoilModeDto } from "../bindings/SpoilModeDto";
import type { VoxelOpModeDto } from "../bindings/VoxelOpModeDto";
import type { VoxelSettingsDto } from "../bindings/VoxelSettingsDto";
import type { VoxelStatusDto } from "../bindings/VoxelStatusDto";
import type { VoxelToolKindDto } from "../bindings/VoxelToolKindDto";
import type { ViewModeDto } from "../bindings/ViewModeDto";
import type { ViewportModeDto } from "../bindings/ViewportModeDto";
import { useSceneStore } from "./sceneStore";
import { useShellStore } from "./shellStore";

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

  /** Shading view mode (Lit / Unlit / Wireframe / Biomes), perspective only
   * (R-P2, P19.2). */
  viewMode: ViewModeDto;

  /** Active tool: pick/gizmo (`Select`) or terrain sculpt (`Sculpt`). (P10.2b) */
  toolMode: ToolModeDto;
  /**
   * The terrain the viewport is drawing streams from a `.inf_terrain` (P16.4a).
   * Read-only, pushed on `viewport://tool-status`.
   */
  terrainStreamed: boolean;
  /**
   * That `.inf_terrain` is writable, so Sculpt/Paint may edit it (P16.4b). Only
   * meaningful with `terrainStreamed`: streamed && !editable is the one case the
   * brush tools are disabled.
   */
  terrainEditable: boolean;
  /**
   * The terrain carries sculpt/paint edits not yet written back to its asset
   * (P16.4b). Terrain edits are only written by an explicit save — autosave does
   * not touch assets — so this is the reminder that Ctrl+S is owed.
   */
  terrainUnsavedEdits: boolean;
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

  /** Biome brush (P19.2). Radius is world metres; `biomeStrength` is NOT a blend
   * — it picks which falloff contour the painted biome's hard boundary lands on
   * (`1` stamps the whole disk); `biomeId` is the painted id, `0` = the reserved
   * *unassigned* value, which erases. */
  biomeRadius: number;
  biomeStrength: number;
  biomeFalloff: SculptFalloffDto;
  biomeId: number;
  /**
   * The vocabulary the Biome tool paints with, for the terrain the toolbar is
   * pointed at (P19.2). `null` until the first `refreshBiomes`, and again
   * whenever the level has no terrain — the toolbar shows a hint rather than an
   * empty swatch row.
   */
  terrainBiomes: TerrainBiomesDto | null;

  /** Water tool (P20.4). `waterKind` picks River or Lake; the rest are the
   * dimensions a NEW body takes. `waterLevelOffset` is added to whatever level
   * the click suggests (the biome `water_hint`, else the ground), so "a lake 2 m
   * above the ground I clicked" needs no arithmetic. Units are SI: metres, m/s.
   * A NEGATIVE flow is meaningful — it reverses a river without re-authoring its
   * spline — so it is deliberately not clamped. */
  waterKind: WaterToolKindDto;
  waterWidth: number;
  waterDepth: number;
  waterFlow: number;
  waterLevelOffset: number;
  /**
   * The river tool's verdict on the selected river (P20.4): the two COOK
   * advisories re-run plus the terrain-aware bed conflicts only the editor can
   * make. `null` when the Water tool is off, nothing is selected, or the
   * selection is not a river — which the toolbar shows as "select a river",
   * not as a clean bill of health.
   */
  waterRiverReport: RiverReportDto | null;

  /**
   * Voxel carve tool (P21.2). `voxelKind` picks the brush or the spline tunnel;
   * `voxelMode` picks carve or fill; both lengths are world metres.
   *
   * `voxelDepth` is the one that needs explaining: it is how far BELOW the picked
   * surface the cut's centre sits. At 0 the cut breaks the ground where you point
   * (a cave mouth); past the radius it hollows rock with no mouth at all, which is
   * legal on any terrain because no hole has to be persisted. It exists because
   * the pick is the heightfield under the cursor — camera-independent, like the
   * water tool's — so saying "how deep" is the only way to reach under the surface
   * without making the result depend on where the camera happened to be.
   */
  voxelKind: VoxelToolKindDto;
  voxelRadius: number;
  voxelDepth: number;
  voxelMode: VoxelOpModeDto;
  /** Splat index a FILL paints. Ignored by a carve — an emptied voxel has no
   * material. A voxel material index IS a terrain splat index, which is what
   * makes a cave wall shade like the hillside it opens out of. */
  voxelMaterial: number;
  /**
   * **Dig to grade** (P21.3): a brush dab becomes a column from `voxelDepth`
   * below the surface up to daylight instead of a ball at depth, so a freehand
   * stroke leaves an open cut rather than a string of buried bubbles. Ignored by
   * the other three sub-modes, which are open to the sky by construction.
   */
  voxelDigToDepth: boolean;
  /**
   * Where the excavated material goes (P21.3): discarded, piled at the
   * deterministic default site (east of the cut), or piled where the author
   * picked.
   */
  voxelSpoil: SpoilModeDto;
  /**
   * While true, a viewport click MOVES the spoil site instead of digging.
   *
   * A sticky mode and not a one-shot arm: the button stays lit, every click
   * drags the marker, and the author turns it off when the heap is where they
   * want it. A one-shot would have to disarm itself on the native side and
   * re-sync a flag this store owns — a desync waiting to happen.
   */
  voxelPickSpoilSite: boolean;
  /**
   * The Voxel tool's verdict on the open level (P21.2): what can be carved, what
   * a surface-crossing cut would be refused for, and how much is unsaved. `null`
   * until the first `refreshVoxelStatus` — "not asked yet", which the toolbar
   * deliberately does not render the same as "a level with nothing to carve".
   */
  voxelStatus: VoxelStatusDto | null;

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
  /** Set the shading view mode + push to the viewport (R-P2, P19.2). */
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

  setBiomeRadius: (r: number) => void;
  setBiomeStrength: (s: number) => void;
  setBiomeFalloff: (f: SculptFalloffDto) => void;
  setBiomeId: (id: number) => void;
  /**
   * Re-read the terrain's biome vocabulary into `terrainBiomes` (P19.2). Also
   * repairs the selection: an id that the (re)bound set no longer defines would
   * paint an invisible biome, so it falls back to the first defined id, or to 0
   * (the eraser) when the set is empty.
   */
  refreshBiomes: () => Promise<void>;

  setWaterKind: (kind: WaterToolKindDto) => void;
  setWaterWidth: (w: number) => void;
  setWaterDepth: (d: number) => void;
  setWaterFlow: (f: number) => void;
  setWaterLevelOffset: (o: number) => void;
  /**
   * Re-read the river report for the first selected entity (P20.4).
   *
   * Called on entering the tool, on selection change and after each placement,
   * because all three change the answer. A non-river selection reports
   * `total_frames == 0`, which is how "not a river" and "a river with no
   * centreline yet" both arrive as "nothing to say".
   */
  refreshRiverReport: () => Promise<void>;

  setVoxelKind: (kind: VoxelToolKindDto) => void;
  setVoxelRadius: (r: number) => void;
  setVoxelDepth: (d: number) => void;
  setVoxelMode: (mode: VoxelOpModeDto) => void;
  setVoxelMaterial: (m: number) => void;
  setVoxelDigToDepth: (on: boolean) => void;
  setVoxelSpoil: (mode: SpoilModeDto) => void;
  /** Toggle the sticky "click places the spoil site" mode (P21.3). */
  toggleVoxelPickSpoilSite: () => void;
  /**
   * Re-read the level's carve verdict (P21.2).
   *
   * Called on entering the tool and after every `world://delta`, because both
   * change the answer: a delta is emitted by every carve (so `unsaved_chunks`
   * moves), by every save, and by anything that adds or removes a terrain or a
   * volume — which is exactly what decides whether the next cut is refused.
   */
  refreshVoxelStatus: () => Promise<void>;

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

/** Send the current biome brush settings to the native viewport (P19.2).
 *  Deliberately NOT persisted: the ids are only meaningful against whichever
 *  `.inf_biomes` the open level's terrain binds, so a remembered id would arm the
 *  brush with a biome the next project has never heard of. */
function pushBiome(s: ViewportUiState): void {
  const dto: BiomeSettingsDto = {
    radius: s.biomeRadius,
    strength: s.biomeStrength,
    falloff: s.biomeFalloff,
    biome: s.biomeId,
  };
  void viewport.setBiome(dto).catch(() => {});
}

/**
 * Send the current water-tool settings to the native viewport (P20.4).
 *
 * NOT persisted, for the biome brush's reason turned around: these are metres
 * and m/s that mean something about the level being authored, and a remembered
 * 40 m-wide river from another project would place a lake-sized stream on the
 * first click.
 */
function pushWater(s: ViewportUiState): void {
  const dto: WaterSettingsDto = {
    kind: s.waterKind,
    width_m: s.waterWidth,
    depth_m: s.waterDepth,
    flow_m_s: s.waterFlow,
    level_offset_m: s.waterLevelOffset,
  };
  void viewport.setWater(dto).catch(() => {});
}

/**
 * Send the current voxel carve-tool settings to the native viewport (P21.2).
 *
 * NOT persisted, for the water tool's reason: a radius is a statement about the
 * level being authored, and a remembered 20 m brush from another project would
 * make the first click of a new session excavate a room.
 */
function pushVoxel(s: ViewportUiState): void {
  const dto: VoxelSettingsDto = {
    kind: s.voxelKind,
    radius_m: s.voxelRadius,
    depth_m: s.voxelDepth,
    mode: s.voxelMode,
    material: s.voxelMaterial,
    dig_to_depth: s.voxelDigToDepth,
    spoil: s.voxelSpoil,
    pick_spoil_site: s.voxelPickSpoilSite,
  };
  void viewport.setVoxel(dto).catch(() => {});
}

/** A finite, non-negative metre value, or the fallback. Shared by the water
 *  tool's width/depth setters — a NaN from a half-typed number input must not
 *  reach the viewport. */
function positiveMetres(v: number, fallback: number): number {
  return Number.isFinite(v) && v >= 0 ? v : fallback;
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
  terrainStreamed: false,
  terrainEditable: false,
  terrainUnsavedEdits: false,
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
  biomeRadius: 8,
  // Full strength by default (mirrors `inf_viewport::camera::BiomeSettings`): a
  // stamp is what an author reaches for first, and a partial-strength default
  // reads as a broken brush rather than a smaller one.
  biomeStrength: 1,
  biomeFalloff: "Smooth",
  biomeId: 0,
  terrainBiomes: null,
  // The `WaterBody` component's own river defaults, so a river drawn with the
  // tool and one added through Add Component start identical.
  waterKind: "River",
  waterWidth: 8,
  waterDepth: 1.5,
  waterFlow: 1.5,
  waterLevelOffset: 0,
  waterRiverReport: null,
  // Mirrors `inf_viewport::camera::VoxelSettings::default()` and the DTO's own
  // `Default`, so the toolbar and the tool start out describing the same cut.
  voxelKind: "Brush",
  voxelRadius: 2,
  voxelDepth: 0,
  voxelMode: "Carve",
  voxelMaterial: 0,
  voxelDigToDepth: false,
  // Off, so the carve brush behaves exactly as it did in P21.2 until an author
  // asks for an excavation — a first click that filled the hillside beside it
  // with spoil would be a surprise, and the point of a default is that nothing
  // surprising happens.
  voxelSpoil: "Off",
  voxelPickSpoilSite: false,
  voxelStatus: null,
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
    if (toolMode === "Biome") {
      pushBiome(get());
      // …and re-read the vocabulary: the bound set may have changed (or been
      // edited) since the toolbar last showed its swatches.
      void get().refreshBiomes();
    }
    if (toolMode === "Water") {
      pushWater(get());
      // …and re-read the vocabulary TOO (P20.4 audit). `terrain.biomes` is what
      // makes the backend push the id-indexed water-level hints beside the
      // overlay palette, and the water tool resolves the biome `water_hint` per
      // click out of that table. Without this arm the table stayed empty until
      // the author happened to visit the Biome tool — so a fresh session that
      // went straight to Water committed a lake at the picked GROUND, while the
      // same click after a detour through Biome committed it at the HINT. A
      // committed level is not allowed to depend on which tools the session
      // visited.
      void get().refreshBiomes();
      // The river report follows the selection, so arm it on entry too.
      void get().refreshRiverReport();
    }
    if (toolMode === "Voxel") {
      pushVoxel(get());
      // …and read the verdict on entry, for the reason the water tool reads its
      // hint table: an author has to be able to see that a cut here WILL be
      // refused before making it, not after. A carve commits geometry.
      void get().refreshVoxelStatus();
    }
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

  setBiomeRadius: (r) => {
    set({ biomeRadius: clampSculptRadius(r) });
    pushBiome(get());
  },
  setBiomeStrength: (s) => {
    // A contour selector, not a rate: outside [0,1] it names no contour at all.
    set({ biomeStrength: Number.isFinite(s) ? Math.min(Math.max(s, 0), 1) : 1 });
    pushBiome(get());
  },
  setBiomeFalloff: (biomeFalloff) => {
    set({ biomeFalloff });
    pushBiome(get());
  },
  setBiomeId: (id) => {
    const biomeId = Number.isFinite(id) ? Math.min(Math.max(Math.round(id), 0), 255) : 0;
    set({ biomeId });
    pushBiome(get());
  },
  setWaterKind: (waterKind) => {
    set({ waterKind });
    pushWater(get());
  },
  setWaterWidth: (w) => {
    set({ waterWidth: positiveMetres(w, get().waterWidth) });
    pushWater(get());
  },
  setWaterDepth: (d) => {
    set({ waterDepth: positiveMetres(d, get().waterDepth) });
    pushWater(get());
  },
  setWaterFlow: (f) => {
    // Signed on purpose: a negative flow reverses the river.
    set({ waterFlow: Number.isFinite(f) ? f : get().waterFlow });
    pushWater(get());
  },
  setWaterLevelOffset: (o) => {
    // Also signed: a lake BELOW the clicked ground is how you author a sinkhole.
    set({ waterLevelOffset: Number.isFinite(o) ? o : get().waterLevelOffset });
    pushWater(get());
  },
  setVoxelKind: (voxelKind) => {
    set({ voxelKind });
    pushVoxel(get());
  },
  setVoxelRadius: (r) => {
    set({ voxelRadius: positiveMetres(r, get().voxelRadius) });
    pushVoxel(get());
  },
  setVoxelDepth: (d) => {
    // Non-negative: depth is measured DOWN from the picked surface, so a negative
    // one names a cut above the ground that was clicked. The backend clamps it to
    // 0 anyway, and a field that silently disagrees with the tool is worse than
    // one that cannot express the value at all.
    set({ voxelDepth: positiveMetres(d, get().voxelDepth) });
    pushVoxel(get());
  },
  setVoxelMode: (voxelMode) => {
    set({ voxelMode });
    pushVoxel(get());
  },
  setVoxelMaterial: (m) => {
    const voxelMaterial = Number.isFinite(m) ? Math.min(Math.max(Math.round(m), 0), 255) : 0;
    set({ voxelMaterial });
    pushVoxel(get());
  },
  setVoxelDigToDepth: (voxelDigToDepth) => {
    set({ voxelDigToDepth });
    pushVoxel(get());
  },
  setVoxelSpoil: (voxelSpoil) => {
    // Leaving Site mode also leaves the picking mode: a lit "Set spoil site"
    // button over a tool that no longer uses one is a control that does nothing
    // and explains nothing.
    set(voxelSpoil === "Site" ? { voxelSpoil } : { voxelSpoil, voxelPickSpoilSite: false });
    pushVoxel(get());
  },
  toggleVoxelPickSpoilSite: () => {
    const on = !get().voxelPickSpoilSite;
    // Arming the picker implies the mode that uses the site — otherwise the
    // author places a marker and the dig ignores it.
    set(on ? { voxelPickSpoilSite: true, voxelSpoil: "Site" } : { voxelPickSpoilSite: false });
    pushVoxel(get());
  },
  refreshVoxelStatus: async () => {
    try {
      set({ voxelStatus: await viewport.voxelStatus() });
    } catch {
      // Swallowed like the other polled reads: no viewport attached is a normal
      // state, not an error to surface. `null` reads as "nothing known", which is
      // what the toolbar then says.
      set({ voxelStatus: null });
    }
  },
  refreshRiverReport: async () => {
    const selection = useSceneStore.getState().selection;
    const entity = selection[0];
    if (!entity) {
      set({ waterRiverReport: null });
      return;
    }
    try {
      const report = await water.riverReport(entity);
      // A report with no frames is not a river (or is a river with no
      // centreline yet); either way there is nothing to report ON.
      set({ waterRiverReport: report.total_frames > 0 ? report : null });
    } catch {
      set({ waterRiverReport: null });
    }
  },
  refreshBiomes: async () => {
    try {
      const terrainBiomes = await terrain.biomes();
      // Repair the selection before it can paint: an id the (re)bound set no
      // longer defines writes samples nothing can name. 0 (erase) is always
      // valid, so it is the floor.
      const defined = terrainBiomes?.biomes ?? [];
      const current = get().biomeId;
      const biomeId =
        current === 0 || defined.some((b) => b.id === current) ? current : (defined[0]?.id ?? 0);
      set({ terrainBiomes, biomeId });
      if (biomeId !== current) pushBiome(get());
    } catch {
      // Swallowed like the other pushes — the toolbar polls this, and a level
      // with no terrain is a normal state, not an error to surface.
    }
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
    { id: "view.biomes", title: "View Mode: Biomes", category: "View" },
    { id: "tool.select", title: "Tool: Select", category: "Tools" },
    { id: "tool.sculpt", title: "Tool: Sculpt Terrain", category: "Tools" },
    { id: "tool.foliage", title: "Tool: Paint Foliage", category: "Tools" },
    { id: "tool.biome", title: "Tool: Paint Biomes", category: "Tools" },
    { id: "tool.voxel", title: "Tool: Carve Voxels", category: "Tools" },
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
  if (getCommand("view.biomes")) {
    setCommandHandler("view.biomes", () => useViewportStore.getState().setViewMode("Biomes"));
  }
  if (getCommand("tool.foliage")) {
    setCommandHandler("tool.foliage", () => useViewportStore.getState().setToolMode("Foliage"));
  }
  if (getCommand("tool.biome")) {
    setCommandHandler("tool.biome", () => useViewportStore.getState().setToolMode("Biome"));
  }
  if (getCommand("tool.voxel")) {
    setCommandHandler("tool.voxel", () => useViewportStore.getState().setToolMode("Voxel"));
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
  // P20.4: while the Water tool is active, keep the river verdict in step with
  // the document. Every water placement emits a delta, and so does every
  // selection change — the two things that change the answer. Subscribed at the
  // store rather than in a React effect, matching every other live projection in
  // this file (and keeping the components free of the banned set-state-in-effect
  // shape). Off the Water tool this costs one enum comparison per delta.
  track(
    listenTo("world://delta", () => {
      if (useViewportStore.getState().toolMode === "Water") {
        void useViewportStore.getState().refreshRiverReport();
      }
      // P21.2, the same shape: a carve, a save and any terrain/volume change all
      // emit a delta, and all three change what the next cut is allowed to do.
      if (useViewportStore.getState().toolMode === "Voxel") {
        void useViewportStore.getState().refreshVoxelStatus();
      }
    }),
  );
  // The viewport echoes gizmo-mode changes (W/E/R keypress or an IPC set); mirror
  // them into the store WITHOUT calling back into IPC (no loop).
  //
  // Filtered to the SCENE viewport (P23.2a): this store is the scene toolbar's
  // state, so a second viewport's W/E/R must not move it. The channel is shared
  // by every viewport, which is why the id has to be read rather than assumed.
  track(
    listenTo("viewport://gizmo", (ev) => {
      if (!isPrimaryViewport(ev.viewport)) return;
      if (useViewportStore.getState().gizmoMode !== ev.mode) {
        useViewportStore.setState({ gizmoMode: ev.mode });
      }
    }),
  );
  // Tool rejections + the standing terrain flags (P16.4a/b). The message is a
  // one-shot toast; the flags are standing state the toolbar reads.
  track(
    listenTo("viewport://tool-status", (status) => {
      // Scene viewport only — the flags below are the scene toolbar's (P23.2a).
      if (!isPrimaryViewport(status.viewport)) return;
      const s = useViewportStore.getState();
      if (
        s.terrainStreamed !== status.terrain_streamed ||
        s.terrainEditable !== status.terrain_editable ||
        s.terrainUnsavedEdits !== status.terrain_unsaved_edits
      ) {
        useViewportStore.setState({
          terrainStreamed: status.terrain_streamed,
          terrainEditable: status.terrain_editable,
          terrainUnsavedEdits: status.terrain_unsaved_edits,
        });
      }
      if (status.message) useShellStore.getState().pushStatus(status.message, 6000);
    }),
  );
  return () => {
    disposed = true;
    for (const fn of unlisteners) fn();
  };
}
