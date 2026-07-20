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
  pushStatus: (message: string, ttlMs?: number) => void;
  clearStatus: () => void;
  openLayoutDialog: (kind: Exclude<LayoutDialogKind, null>) => void;
  closeLayoutDialog: () => void;
}

let statusTimer: ReturnType<typeof setTimeout> | undefined;

export const useShellStore = create<ShellState>((set) => ({
  statusMessage: null,
  layoutDialog: null,
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
}));

registerBridgedStore("shell", useShellStore);
