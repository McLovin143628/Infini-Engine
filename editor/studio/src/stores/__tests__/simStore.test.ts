// @vitest-environment jsdom
//
// Simulate store (P8.4): play/pause/resume/stop state transitions, the
// key→action mapping (incl. edges), the input tracker, and the rAF tick loop's
// in-flight guard. IPC is mocked; requestAnimationFrame is stubbed so frames run
// deterministically under test control.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  sim: {
    start: vi.fn(() => Promise.resolve()),
    tick: vi.fn(() => Promise.resolve(true)),
    stepFixed: vi.fn(() => Promise.resolve(true)),
    stop: vi.fn(() => Promise.resolve()),
    isRunning: vi.fn(() => Promise.resolve(false)),
  },
}));

import { sim } from "../../lib/ipc";
import { useShellStore } from "../shellStore";
import {
  __heldActions,
  __resetSimForTest,
  simKeyCode,
  useSimStore,
} from "../simStore";

/** The currently-scheduled animation-frame callback (single-slot rAF stub). */
let nextFrame: FrameRequestCallback | null = null;

function runFrame(): void {
  const cb = nextFrame;
  nextFrame = null;
  cb?.(0);
}

/** Let queued microtasks (the tick's `.finally`) settle. */
const flush = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

beforeEach(() => {
  nextFrame = null;
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    nextFrame = cb;
    return 1;
  });
  vi.stubGlobal("cancelAnimationFrame", () => {
    nextFrame = null;
  });
  vi.mocked(sim.start).mockResolvedValue(undefined);
  vi.mocked(sim.stop).mockResolvedValue(undefined);
  vi.mocked(sim.tick).mockResolvedValue(true);
  vi.mocked(sim.stepFixed).mockResolvedValue(true);
  vi.mocked(sim.isRunning).mockResolvedValue(false);
});

afterEach(() => {
  __resetSimForTest();
  useShellStore.getState().clearStatus();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("simKeyCode", () => {
  const ev = (code: string): KeyboardEvent => new KeyboardEvent("keydown", { code });

  it("passes a physical key through unmapped", () => {
    // P29.6: the binding table is `inf_input::default_map`'s, Ring 0, shared
    // with the shipped player. This side must not have an opinion about which
    // keys are controls, or it will have a smaller opinion than the engine's.
    for (const code of ["KeyA", "KeyD", "KeyW", "KeyS", "KeyC", "KeyX", "KeyR", "KeyF", "Space"]) {
      expect(simKeyCode(ev(code))).toBe(code);
    }
  });

  it("folds the left/right modifiers onto the engine's single name", () => {
    expect(simKeyCode(ev("ShiftLeft"))).toBe("Shift");
    expect(simKeyCode(ev("ShiftRight"))).toBe("Shift");
    expect(simKeyCode(ev("ControlLeft"))).toBe("Control");
    expect(simKeyCode(ev("ControlRight"))).toBe("Control");
    // …and `AltLeft` is NOT folded: the engine binds `walk` to that exact name.
    expect(simKeyCode(ev("AltLeft"))).toBe("AltLeft");
  });
});

describe("play / pause / resume / stop transitions", () => {
  it("play() starts a session and marks it running", async () => {
    await useSimStore.getState().play();
    expect(vi.mocked(sim.start)).toHaveBeenCalledOnce();
    expect(useSimStore.getState().running).toBe(true);
    expect(useSimStore.getState().paused).toBe(false);
  });

  it("pause() stops the loop but keeps the session; resume() restarts it", async () => {
    await useSimStore.getState().play();
    useSimStore.getState().pause();
    expect(useSimStore.getState().running).toBe(true);
    expect(useSimStore.getState().paused).toBe(true);
    expect(nextFrame).toBeNull(); // loop stopped

    useSimStore.getState().resume();
    expect(useSimStore.getState().paused).toBe(false);
    expect(nextFrame).not.toBeNull(); // loop rescheduled
  });

  it("play() while paused resumes instead of restarting the session", async () => {
    await useSimStore.getState().play();
    useSimStore.getState().pause();
    await useSimStore.getState().play();
    expect(vi.mocked(sim.start)).toHaveBeenCalledOnce(); // not started twice
    expect(useSimStore.getState().paused).toBe(false);
  });

  it("stop() ends the session and clears state", async () => {
    await useSimStore.getState().play();
    await useSimStore.getState().stop();
    expect(vi.mocked(sim.stop)).toHaveBeenCalledOnce();
    expect(useSimStore.getState().running).toBe(false);
    expect(useSimStore.getState().paused).toBe(false);
  });

  it("surfaces a start failure via the shell status and stays stopped", async () => {
    vi.mocked(sim.start).mockRejectedValueOnce(new Error("no scene"));
    await useSimStore.getState().play();
    expect(useSimStore.getState().running).toBe(false);
    expect(useShellStore.getState().statusMessage).toContain("no scene");
  });
});

describe("input tracker", () => {
  it("tracks held physical keys from window key events while running", async () => {
    await useSimStore.getState().play(); // installs listeners
    window.dispatchEvent(new KeyboardEvent("keydown", { code: "ArrowLeft" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { code: "Space" }));
    expect(__heldActions().sort()).toEqual(["ArrowLeft", "Space"]);

    window.dispatchEvent(new KeyboardEvent("keyup", { code: "ArrowLeft" }));
    expect(__heldActions()).toEqual(["Space"]);
  });

  it("ignores keys while stopped", () => {
    window.dispatchEvent(new KeyboardEvent("keydown", { code: "ArrowLeft" }));
    expect(__heldActions()).toEqual([]);
  });

  it("preventDefault on Space stops the page scrolling", async () => {
    await useSimStore.getState().play();
    const ev = new KeyboardEvent("keydown", { code: "Space", cancelable: true });
    window.dispatchEvent(ev);
    expect(ev.defaultPrevented).toBe(true);
  });
});

describe("tick loop", () => {
  it("guards against overlapping in-flight ticks", async () => {
    let resolveTick: (v: boolean) => void = () => {};
    vi.mocked(sim.tick).mockImplementation(
      () => new Promise<boolean>((res) => (resolveTick = res)),
    );
    await useSimStore.getState().play(); // schedules the first frame

    runFrame(); // issues tick #1 (pending)
    expect(vi.mocked(sim.tick)).toHaveBeenCalledTimes(1);
    runFrame(); // guarded: previous tick still in flight → no new call
    expect(vi.mocked(sim.tick)).toHaveBeenCalledTimes(1);

    resolveTick(true);
    await flush(); // let the `.finally` clear the in-flight flag

    runFrame(); // now the next tick may issue
    expect(vi.mocked(sim.tick)).toHaveBeenCalledTimes(2);
  });

  it("feeds the held PHYSICAL keys to sim_tick, unmapped", async () => {
    await useSimStore.getState().play();
    window.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyD" }));
    runFrame();
    // `KeyD`, not `"right"`: the binding table is the engine's (Ring 0), and a
    // copy of it here would know a fraction of it (P29.6).
    expect(vi.mocked(sim.tick)).toHaveBeenCalledWith(["KeyD"]);
  });

  it("forwards a key the old three-action map had never heard of", async () => {
    await useSimStore.getState().play();
    window.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyC" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { code: "ShiftLeft" }));
    runFrame();
    // Crouch and sprint reach the backend now. Under the old mapping both were
    // dropped here and no control in the engine could ever have seen them.
    expect(vi.mocked(sim.tick)).toHaveBeenCalledWith(["KeyC", "Shift"]);
  });

  it("step() advances exactly one fixed step and pauses", async () => {
    await useSimStore.getState().play();
    await useSimStore.getState().step();
    expect(useSimStore.getState().paused).toBe(true);
    // Step uses the fixed-step command (B-P4), not the accumulating tick.
    expect(vi.mocked(sim.stepFixed)).toHaveBeenCalledTimes(1);
  });
});
