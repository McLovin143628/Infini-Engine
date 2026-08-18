/**
 * Simulate (Play) store (P8.4) — the frontend half of the in-editor play loop.
 *
 * The backend owns the authoritative `SimSession` (start snapshots the world,
 * stop restores it) and fixed-step accumulation; this store drives it:
 *
 * 1. **Running state** — mirrors the backend, synced from the `sim://state`
 *    event and `sim_is_running()` on mount (see [`initSimSync`]).
 * 2. **Input tracker** — while running, a window keydown/keyup listener keeps
 *    the set of held **physical keys** (`KeyboardEvent.code`) and sends it
 *    verbatim. It does NOT map keys to action names: that table is the engine's
 *    (`inf_input::default_map`, Ring 0, shared with the shipped player), and the
 *    version of it that used to live here knew three of its fourteen entries —
 *    the two-copies-across-a-language-boundary defect the campaign's Wave I
 *    found at this same seam. `Space` still `preventDefault`s so the page never
 *    scrolls while playing.
 * 3. **Tick driver** — a `requestAnimationFrame` loop feeds real frame cadence +
 *    the held actions to `sim_tick`. The backend clamps/accumulates into fixed
 *    steps, so the rAF loop only supplies the wall-clock delta (implicitly, via
 *    the backend's `Instant`) and the keys. Overlapping ticks are guarded: a new
 *    `sim_tick` is never issued until the previous one has resolved (`inFlight`).
 *
 * Pause stops ticking WITHOUT `sim_stop` (the session stays live); Resume
 * restarts the loop. Step advances exactly one `sim_tick` (see [`step`] for the
 * dt limitation). Stop calls `sim_stop`.
 */
import { create } from "zustand";

import { listenTo, refCountedInit } from "../lib/events";
import { sim as simIpc } from "../lib/ipc";
import { bindKey } from "../lib/keybindings";
import { registerCommands } from "../lib/commands";
import { useShellStore } from "./shellStore";

/**
 * The physical keys Simulate forwards, as `KeyboardEvent.code` values.
 *
 * A denylist rather than an allowlist, and that is the point: the backend owns
 * the binding table, so this side must not decide which keys are "controls".
 * The modifiers are excluded because they arrive as their own events and the
 * engine's map spells them `Shift` / `AltLeft` / `Control` (see
 * `keycode_to_code` in the player), which the backend resolves; forwarding the
 * raw `ShiftLeft` alongside would be a second name for one key.
 */
export function simKeyCode(e: KeyboardEvent): string | null {
  switch (e.code) {
    case "ShiftLeft":
    case "ShiftRight":
      return "Shift";
    case "ControlLeft":
    case "ControlRight":
      return "Control";
    default:
      return e.code;
  }
}

// ── Module-local driver state (kept out of the store to avoid re-render churn
//    on every keypress / frame; the store carries only what the UI reads). ────

/** Currently-held physical keys, as the engine's `KeyboardEvent.code` names. */
const held = new Set<string>();
/** Active rAF handle, or null when the tick loop is stopped/paused. */
let rafId: number | null = null;
/** True while a `sim_tick` call is awaiting — the overlap guard. */
let inFlight = false;
/** Whether the window key listeners are currently installed. */
let inputInstalled = false;

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function onKeyDown(e: KeyboardEvent): void {
  if (!useSimStore.getState().running) return;
  // Don't hijack typing in a focused text field.
  const el = e.target instanceof HTMLElement ? e.target : null;
  if (el?.closest("input, textarea, [contenteditable]")) return;
  const code = simKeyCode(e);
  if (!code) return;
  // Stop Space from scrolling the shell while jumping.
  if (e.code === "Space") e.preventDefault();
  held.add(code);
}

function onKeyUp(e: KeyboardEvent): void {
  const code = simKeyCode(e);
  if (code) held.delete(code);
}

function installInput(): void {
  if (inputInstalled) return;
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  inputInstalled = true;
}

function removeInput(): void {
  if (!inputInstalled) return;
  window.removeEventListener("keydown", onKeyDown);
  window.removeEventListener("keyup", onKeyUp);
  inputInstalled = false;
  held.clear();
}

/** One animation frame: schedule the next, then issue a tick unless one is in flight. */
function frame(): void {
  rafId = requestAnimationFrame(frame);
  if (inFlight) return;
  inFlight = true;
  void simIpc
    .tick([...held])
    .catch((e) => console.error("sim.tick failed", e))
    .finally(() => {
      inFlight = false;
    });
}

function startLoop(): void {
  if (rafId != null) return;
  rafId = requestAnimationFrame(frame);
}

function stopLoop(): void {
  if (rafId != null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
}

interface SimState {
  /** A session is live in the backend (playing or paused). */
  running: boolean;
  /** Running but the tick loop is stopped (session preserved). */
  paused: boolean;

  /** Enter Simulate, or Resume when already paused. */
  play: () => Promise<void>;
  /** Stop ticking without ending the session. */
  pause: () => void;
  /** Restart the tick loop after a Pause. */
  resume: () => void;
  /** Exit Simulate, restoring the pre-play world. */
  stop: () => Promise<void>;
  /** Advance exactly one frame (implies Pause). */
  step: () => Promise<void>;
  /** Reconcile to the backend's running state (from the event / mount sync). */
  syncRunning: (running: boolean) => void;
}

export const useSimStore = create<SimState>((set, get) => ({
  running: false,
  paused: false,

  play: async () => {
    if (get().running) {
      if (get().paused) get().resume();
      return;
    }
    try {
      await simIpc.start();
    } catch (e) {
      useShellStore.getState().pushStatus(`Simulate could not start: ${errText(e)}`);
      return;
    }
    // Drive the effects immediately; the sim://state event confirms idempotently.
    get().syncRunning(true);
  },

  pause: () => {
    if (get().running && !get().paused) {
      set({ paused: true });
      stopLoop();
    }
  },

  resume: () => {
    if (get().running && get().paused) {
      set({ paused: false });
      startLoop();
    }
  },

  stop: async () => {
    if (!get().running) return;
    try {
      await simIpc.stop();
    } catch (e) {
      useShellStore.getState().pushStatus(`Simulate stop failed: ${errText(e)}`);
      return;
    }
    get().syncRunning(false);
  },

  step: async () => {
    if (!get().running) return;
    // Step is single-frame: pause the free-running loop first.
    if (!get().paused) {
      set({ paused: true });
      stopLoop();
    }
    // B-P4: `sim_step_fixed` advances EXACTLY one fixed step (bypassing the
    // wall-clock accumulator), so Step is now a guaranteed single step — fixing
    // the former "one frame's worth (0..N steps)" limitation.
    if (inFlight) return;
    inFlight = true;
    try {
      await simIpc.stepFixed([...held]);
    } catch (e) {
      console.error("sim.step stepFixed failed", e);
    } finally {
      inFlight = false;
    }
  },

  syncRunning: (running) => {
    if (running) {
      if (!get().running) set({ running: true, paused: false });
      installInput();
      if (!get().paused) startLoop();
    } else {
      stopLoop();
      removeInput();
      set({ running: false, paused: false });
    }
  },
}));

/** Register the Simulate play-cluster commands + the Alt+P Play binding (P8.4). */
export function registerSimCommands(): void {
  registerCommands([
    {
      id: "sim.play",
      title: "Simulate: Play / Resume",
      category: "Play",
      shortcut: "Alt+P",
      run: () => void useSimStore.getState().play(),
    },
    {
      id: "sim.pause",
      title: "Simulate: Pause",
      category: "Play",
      run: () => useSimStore.getState().pause(),
    },
    {
      id: "sim.stop",
      title: "Simulate: Stop",
      category: "Play",
      run: () => void useSimStore.getState().stop(),
    },
    {
      id: "sim.step",
      title: "Simulate: Step Frame",
      category: "Play",
      run: () => void useSimStore.getState().step(),
    },
  ]);
  bindKey({ chord: "Alt+P", command: "sim.play" });
}

/**
 * Sync running state from `sim_is_running()` on mount and subscribe to
 * `sim://state` + `sim://debug`. Returns a disposer.
 *
 * The `sim://debug` listener pauses Simulate whenever a step hit a breakpoint
 * (B-P4 tier A′). Node-level hit highlighting on the canvas activates once
 * `.inf_act` classes carry graph provenance (the events currently carry IR
 * `LocalId`s, not canvas `NodeId`s) — pausing already works and is harmless.
 *
 * **Refcounted** (`refCountedInit`, F-lens L7.M1): the guard used to be set
 * after the first `await`, so StrictMode's double mount subscribed both
 * channels twice — and a doubled `sim://debug` handler pauses Simulate twice
 * per breakpoint hit.
 */
export const initSimSync = refCountedInit(async (sink) => {
  const unlisten = await listenTo("sim://state", (running) =>
    useSimStore.getState().syncRunning(running),
  );
  // The second subscribe can reject; without the sink the first is orphaned
  // (round-2 finding R2-7).
  sink(unlisten);
  const unlistenDebug = await listenTo("sim://debug", (events) => {
    if (events.some((e) => e.hits.length > 0)) {
      useSimStore.getState().pause();
    }
  });
  try {
    useSimStore.getState().syncRunning(await simIpc.isRunning());
  } catch (e) {
    console.error("sim.isRunning failed", e);
  }
  sink(unlistenDebug);
  return () => {
    unlisten();
    unlistenDebug();
  };
});

/** Test-only: tear down the driver + reset global state between cases. */
export function __resetSimForTest(): void {
  stopLoop();
  removeInput();
  held.clear();
  inFlight = false;
  useSimStore.setState({ running: false, paused: false });
}

/** Test-only: the currently-held engine actions. */
export function __heldActions(): string[] {
  return [...held];
}
