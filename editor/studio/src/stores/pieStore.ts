/**
 * Play-In-Editor (PIE) store (P9.4) — the frontend half of the subprocess play
 * loop.
 *
 * Unlike Simulate (in-process, the editor drives the tick), PIE spawns
 * `inf-player` as a **separate, crash-isolated process** that runs the live
 * scene itself. So this store is thin: it issues `pie_*` commands and mirrors
 * the authoritative session state broadcast on `pie://state`. There is no rAF
 * tick loop and no input tracker here — the player owns its own loop and input.
 *
 * A deliberate PIE panic kills only the player; the backend monitor surfaces the
 * crash on `pie://state` with an `error`, which we toast before returning the
 * toolbar to stopped. Unsaved editor state is never at risk.
 */
import { create } from "zustand";

import { listenTo, refCountedInit, type PieStateEvent } from "../lib/events";
import { pie as pieIpc } from "../lib/ipc";
import { bindKey } from "../lib/keybindings";
import { registerCommands } from "../lib/commands";
import { useShellStore } from "./shellStore";

/** The PIE window mode: reparent into the viewport, or a separate window. */
export type PieMode = "embedded" | "window";

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

interface PieState {
  /** A player subprocess is live (playing or paused). */
  running: boolean;
  /** Live but the player's fixed step is frozen. */
  paused: boolean;
  /** `"embedded"`, `"window"`, or `""` when stopped. */
  mode: string;
  /** The player's current fixed-step frame count (telemetry). */
  frame: number;

  /** Start PIE in `mode` (or Resume when already paused). */
  play: (mode: PieMode) => Promise<void>;
  /** Pause the running player. */
  pause: () => Promise<void>;
  /** Resume a paused player. */
  resume: () => Promise<void>;
  /** Advance exactly one fixed step (implies Pause). */
  step: () => Promise<void>;
  /** Release input possession back to the editor (v1 Eject). */
  eject: () => Promise<void>;
  /** Stop PIE, restoring the viewport. */
  stop: () => Promise<void>;
  /** Reconcile from a `pie://state` event (or the mount sync). */
  syncState: (ev: PieStateEvent) => void;
  /** Reconcile from the boolean mount sync (`pie_is_running`). */
  syncRunning: (running: boolean) => void;
}

export const usePieStore = create<PieState>((set, get) => ({
  running: false,
  paused: false,
  mode: "",
  frame: 0,

  play: async (mode) => {
    if (get().running) {
      if (get().paused) await get().resume();
      return;
    }
    try {
      await pieIpc.start(mode);
    } catch (e) {
      useShellStore.getState().pushStatus(`Play could not start: ${errText(e)}`);
      return;
    }
    // Optimistic; the pie://state event confirms idempotently.
    set({ running: true, paused: false, mode });
  },

  pause: async () => {
    if (!get().running || get().paused) return;
    set({ paused: true });
    try {
      await pieIpc.pause();
    } catch (e) {
      console.error("pie.pause failed", e);
    }
  },

  resume: async () => {
    if (!get().running || !get().paused) return;
    set({ paused: false });
    try {
      await pieIpc.resume();
    } catch (e) {
      console.error("pie.resume failed", e);
    }
  },

  step: async () => {
    if (!get().running) return;
    if (!get().paused) set({ paused: true });
    try {
      await pieIpc.step();
    } catch (e) {
      console.error("pie.step failed", e);
    }
  },

  eject: async () => {
    if (!get().running) return;
    try {
      await pieIpc.eject();
    } catch (e) {
      console.error("pie.eject failed", e);
    }
  },

  stop: async () => {
    if (!get().running) return;
    try {
      await pieIpc.stop();
    } catch (e) {
      useShellStore.getState().pushStatus(`Play stop failed: ${errText(e)}`);
      return;
    }
    set({ running: false, paused: false, mode: "", frame: 0 });
  },

  syncState: (ev) => {
    // A crash / load error surfaces as a toast, then returns to stopped.
    if (ev.error) {
      useShellStore.getState().pushStatus(ev.error);
    }
    set({
      running: ev.running,
      paused: ev.paused,
      mode: ev.running ? ev.mode : "",
      frame: ev.frame,
    });
  },

  syncRunning: (running) => {
    if (running) {
      if (!get().running) set({ running: true });
    } else {
      set({ running: false, paused: false, mode: "", frame: 0 });
    }
  },
}));

/** Register the PIE play-cluster commands + the Shift+Alt+P Play binding. */
export function registerPieCommands(): void {
  registerCommands([
    {
      id: "pie.play",
      title: "Play (Embedded)",
      category: "Play",
      shortcut: "Shift+Alt+P",
      run: () => void usePieStore.getState().play("embedded"),
    },
    {
      id: "pie.playWindow",
      title: "Play in New Window",
      category: "Play",
      run: () => void usePieStore.getState().play("window"),
    },
    {
      id: "pie.pause",
      title: "Play: Pause",
      category: "Play",
      run: () => void usePieStore.getState().pause(),
    },
    {
      id: "pie.resume",
      title: "Play: Resume",
      category: "Play",
      run: () => void usePieStore.getState().resume(),
    },
    {
      id: "pie.step",
      title: "Play: Step Frame",
      category: "Play",
      run: () => void usePieStore.getState().step(),
    },
    {
      id: "pie.eject",
      title: "Play: Eject (release input)",
      category: "Play",
      run: () => void usePieStore.getState().eject(),
    },
    {
      id: "pie.stop",
      title: "Play: Stop",
      category: "Play",
      run: () => void usePieStore.getState().stop(),
    },
  ]);
  bindKey({ chord: "Shift+Alt+P", command: "pie.play" });
}

/**
 * Sync running state from `pie_is_running()` on mount and subscribe to
 * `pie://state`. Returns a disposer.
 *
 * **Refcounted** (`refCountedInit`, F-lens L7.M1): the guard used to be set
 * after the first `await`, so StrictMode's double mount subscribed twice and
 * orphaned the first handle.
 */
export const initPieSync = refCountedInit(async () => {
  const unlisten = await listenTo("pie://state", (ev) => usePieStore.getState().syncState(ev));
  try {
    usePieStore.getState().syncRunning(await pieIpc.isRunning());
  } catch (e) {
    console.error("pie.isRunning failed", e);
  }
  return () => unlisten();
});

/** Test-only: reset the store between cases. */
export function __resetPieForTest(): void {
  usePieStore.setState({ running: false, paused: false, mode: "", frame: 0 });
}
