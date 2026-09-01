/**
 * Shell-level UI state (zustand; see stores/index.ts for the architecture
 * conventions). Carries the transient status-bar message and the layout
 * save/load dialog state. Bridged across windows so detached panels can
 * surface status messages in the main status bar.
 */
import { create } from "zustand";
import { registerBridgedStore } from "../panels/window/storeBridge";

type LayoutDialogKind = "save" | "load" | null;

interface ShellState {
  /** Transient message shown in the status bar (null = show defaults). */
  statusMessage: string | null;
  /** Which layout dialog is open (Window ▸ Save/Load Layout). */
  layoutDialog: LayoutDialogKind;
  /** Sorting-layer manager dialog open (Window ▸ Sorting Layers…, P8.2a). */
  sortingLayersOpen: boolean;
  /** Package/cook dialog open (Build ▸ Package Project…, P9.2). */
  packageDialogOpen: boolean;
  /** Terrain erosion bake dialog open (Sculpt toolbar ▸ Erode…, P10.3b). */
  erodeOpen: boolean;
  /** Terrain Import wizard open (File ▸ Import Terrain…, P16.4a). */
  terrainImportOpen: boolean;
  /** GIS Import wizard open (File ▸ Import GIS Data…, IB-3). */
  gisImportOpen: boolean;
  /** New Character wizard open (Actor ▸ New Character from Template…, P24.5). */
  characterWizardOpen: boolean;
  /** Capture wizard open (File ▸ Capture from Photographs…, P25.4). */
  captureWizardOpen: boolean;
  /** Editor Preferences dialog open (Edit ▸ Editor Preferences… / gear, Wave E). */
  preferencesOpen: boolean;
  /** Project Settings dialog open (Edit ▸ Project Settings…, Wave E). */
  projectSettingsOpen: boolean;
  /** Content Drawer slide-up (P1.4.3). */
  drawerOpen: boolean;
  /** Command palette overlay (P1.4.5). */
  paletteOpen: boolean;
  /**
   * The **no-pawn Play** dialog (wave GTA1), holding the mode Play was asked
   * for so the choice can start it. `null` = closed.
   *
   * A whole dialog rather than a status line because the two answers are
   * genuinely different acts: one EDITS THE LEVEL (places the starter
   * character, an undoable document change that the shipped build will also
   * see) and the other plays a level with no player in it on purpose. A toast
   * cannot ask that, and pressing Play and getting a camera looking at nothing
   * answered it silently and wrongly.
   */
  noPawnPlay: "embedded" | "window" | null;
  pushStatus: (message: string, ttlMs?: number) => void;
  clearStatus: () => void;
  openLayoutDialog: (kind: Exclude<LayoutDialogKind, null>) => void;
  closeLayoutDialog: () => void;
  setSortingLayersOpen: (open: boolean) => void;
  setPackageDialogOpen: (open: boolean) => void;
  setErodeOpen: (open: boolean) => void;
  setTerrainImportOpen: (open: boolean) => void;
  setGisImportOpen: (open: boolean) => void;
  setCharacterWizardOpen: (open: boolean) => void;
  setCaptureWizardOpen: (open: boolean) => void;
  setPreferencesOpen: (open: boolean) => void;
  setProjectSettingsOpen: (open: boolean) => void;
  setDrawerOpen: (open: boolean) => void;
  toggleDrawer: () => void;
  setPaletteOpen: (open: boolean) => void;
  setNoPawnPlay: (mode: "embedded" | "window" | null) => void;
}

let statusTimer: ReturnType<typeof setTimeout> | undefined;

export const useShellStore = create<ShellState>((set) => ({
  statusMessage: null,
  layoutDialog: null,
  sortingLayersOpen: false,
  packageDialogOpen: false,
  erodeOpen: false,
  terrainImportOpen: false,
  gisImportOpen: false,
  characterWizardOpen: false,
  captureWizardOpen: false,
  preferencesOpen: false,
  projectSettingsOpen: false,
  drawerOpen: false,
  paletteOpen: false,
  noPawnPlay: null,
  pushStatus: (message, ttlMs = 4000) => {
    set({ statusMessage: message });
    if (statusTimer !== undefined) clearTimeout(statusTimer);
    statusTimer = setTimeout(() => set({ statusMessage: null }), ttlMs);
  },
  clearStatus: () => {
    if (statusTimer !== undefined) clearTimeout(statusTimer);
    set({ statusMessage: null });
  },
  openLayoutDialog: (kind) => set({ layoutDialog: kind }),
  closeLayoutDialog: () => set({ layoutDialog: null }),
  setSortingLayersOpen: (sortingLayersOpen) => set({ sortingLayersOpen }),
  setPackageDialogOpen: (packageDialogOpen) => set({ packageDialogOpen }),
  setErodeOpen: (erodeOpen) => set({ erodeOpen }),
  setTerrainImportOpen: (terrainImportOpen) => set({ terrainImportOpen }),
  setGisImportOpen: (gisImportOpen) => set({ gisImportOpen }),
  setCharacterWizardOpen: (characterWizardOpen) => set({ characterWizardOpen }),
  setCaptureWizardOpen: (captureWizardOpen) => set({ captureWizardOpen }),
  setPreferencesOpen: (preferencesOpen) => set({ preferencesOpen }),
  setProjectSettingsOpen: (projectSettingsOpen) => set({ projectSettingsOpen }),
  setDrawerOpen: (drawerOpen) => set({ drawerOpen }),
  toggleDrawer: () => set((s) => ({ drawerOpen: !s.drawerOpen })),
  setPaletteOpen: (paletteOpen) => set({ paletteOpen }),
  setNoPawnPlay: (noPawnPlay) => set({ noPawnPlay }),
}));

registerBridgedStore("shell", useShellStore);
