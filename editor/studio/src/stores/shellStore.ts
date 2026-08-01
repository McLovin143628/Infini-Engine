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
  /** Content Drawer slide-up (P1.4.3). */
  drawerOpen: boolean;
  /** Command palette overlay (P1.4.5). */
  paletteOpen: boolean;
  pushStatus: (message: string, ttlMs?: number) => void;
  clearStatus: () => void;
  openLayoutDialog: (kind: Exclude<LayoutDialogKind, null>) => void;
  closeLayoutDialog: () => void;
  setSortingLayersOpen: (open: boolean) => void;
  setPackageDialogOpen: (open: boolean) => void;
  setErodeOpen: (open: boolean) => void;
  setTerrainImportOpen: (open: boolean) => void;
  setDrawerOpen: (open: boolean) => void;
  toggleDrawer: () => void;
  setPaletteOpen: (open: boolean) => void;
}

let statusTimer: ReturnType<typeof setTimeout> | undefined;

export const useShellStore = create<ShellState>((set) => ({
  statusMessage: null,
  layoutDialog: null,
  sortingLayersOpen: false,
  packageDialogOpen: false,
  erodeOpen: false,
  terrainImportOpen: false,
  drawerOpen: false,
  paletteOpen: false,
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
  setDrawerOpen: (drawerOpen) => set({ drawerOpen }),
  toggleDrawer: () => set((s) => ({ drawerOpen: !s.drawerOpen })),
  setPaletteOpen: (paletteOpen) => set({ paletteOpen }),
}));

registerBridgedStore("shell", useShellStore);
