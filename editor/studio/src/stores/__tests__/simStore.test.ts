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
    stop: vi.fn(() => Promise.resolve()),
    isRunning: vi.fn(() => Promise.resolve(false)),
  },
}));

import { sim } from "../../lib/ipc";
import { useShellStore } from "../shellStore";
import {
  __heldActions,
  __resetSimForTest,
  codeToAction,
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
  vi.mocked(sim.isRunning).mockResolvedValue(false);
});

afterEach(() => {
  __resetSimForTest();
  useShellStore.getState().clearStatus();
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("codeToAction", () => {
  it("maps arrows and WASD to engine actions", () => {
    expect(codeToAction("ArrowLeft")).toBe("left");
    expect(codeToAction("KeyA")).toBe("left");
    expect(codeToAction("ArrowRight")).toBe("right");
    expect(codeToAction("KeyD")).toBe("right");
    expect(codeToAction("Space")).toBe("jump");
    expect(codeToAction("KeyW")).toBe("jump");
  });

  it("returns null for unmapped / empty keys (edges)", () => {
    expect(codeToAction("KeyS")).toBeNull();
    expect(codeToAction("Enter")).toBeNull();
    expect(codeToAction("")).toBeNull();
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
  it("tracks held actions from window key events while running", async () => {
    await useSimStore.getState().play(); // installs listeners
    window.dispatchEvent(new KeyboardEvent("keydown", { code: "ArrowLeft" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { code: "Space" }));
    expect(__heldActions().sort()).toEqual(["jump", "left"]);

    window.dispatchEvent(new KeyboardEvent("keyup", { code: "ArrowLeft" }));
    expect(__heldActions()).toEqual(["jump"]);
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

  it("feeds the held actions to sim_tick", async () => {
    await useSimStore.getState().play();
    window.dispatchEvent(new KeyboardEvent("keydown", { code: "KeyD" }));
    runFrame();
    expect(vi.mocked(sim.tick)).toHaveBeenCalledWith(["right"]);
  });

  it("step() advances exactly one tick and pauses", async () => {
    await useSimStore.getState().play();
    vi.mocked(sim.tick).mockClear();
    await useSimStore.getState().step();
    expect(useSimStore.getState().paused).toBe(true);
    expect(vi.mocked(sim.tick)).toHaveBeenCalledTimes(1);
  });
});
