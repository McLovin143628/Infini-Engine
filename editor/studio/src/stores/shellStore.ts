/**
 * Shell-level UI state (zustand). Grows into the P1.3 store architecture;
 * for now it carries the transient status-bar message used to surface
 * stubbed commands ("coming in Phase N") and general shell notices.
 */
import { create } from "zustand";

interface ShellState {
  /** Transient message shown in the status bar (null = show defaults). */
  statusMessage: string | null;
  pushStatus: (message: string, ttlMs?: number) => void;
  clearStatus: () => void;
}

let statusTimer: ReturnType<typeof setTimeout> | undefined;

export const useShellStore = create<ShellState>((set) => ({
  statusMessage: null,
  pushStatus: (message, ttlMs = 4000) => {
    set({ statusMessage: message });
    if (statusTimer !== undefined) clearTimeout(statusTimer);
    statusTimer = setTimeout(() => set({ statusMessage: null }), ttlMs);
  },
  clearStatus: () => {
    if (statusTimer !== undefined) clearTimeout(statusTimer);
    set({ statusMessage: null });
  },
}));
