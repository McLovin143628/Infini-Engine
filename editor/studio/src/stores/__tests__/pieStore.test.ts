// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  pie: {
    start: vi.fn(() => Promise.resolve()),
    pause: vi.fn(() => Promise.resolve()),
    resume: vi.fn(() => Promise.resolve()),
    step: vi.fn(() => Promise.resolve()),
    eject: vi.fn(() => Promise.resolve()),
    stop: vi.fn(() => Promise.resolve()),
    isRunning: vi.fn(() => Promise.resolve(false)),
  },
}));

import { pie } from "../../lib/ipc";
import { useShellStore } from "../shellStore";
import { __resetPieForTest, usePieStore } from "../pieStore";
import type { PieStateEvent } from "../../lib/events";

beforeEach(() => {
  vi.mocked(pie.start).mockResolvedValue(undefined);
  vi.mocked(pie.pause).mockResolvedValue(undefined);
  vi.mocked(pie.resume).mockResolvedValue(undefined);
  vi.mocked(pie.step).mockResolvedValue(undefined);
  vi.mocked(pie.eject).mockResolvedValue(undefined);
  vi.mocked(pie.stop).mockResolvedValue(undefined);
  vi.mocked(pie.isRunning).mockResolvedValue(false);
});

afterEach(() => {
  __resetPieForTest();
  useShellStore.getState().clearStatus();
  vi.clearAllMocks();
});

function stateEvent(over: Partial<PieStateEvent>): PieStateEvent {
  return { running: false, paused: false, mode: "", frame: 0, error: null, ...over };
}

describe("pieStore", () => {
  it("play(embedded) starts a session and marks it running", async () => {
    await usePieStore.getState().play("embedded");
    expect(vi.mocked(pie.start)).toHaveBeenCalledWith("embedded");
    expect(usePieStore.getState().running).toBe(true);
    expect(usePieStore.getState().mode).toBe("embedded");
    expect(usePieStore.getState().paused).toBe(false);
  });

  it("play('window') passes the window mode through", async () => {
    await usePieStore.getState().play("window");
    expect(vi.mocked(pie.start)).toHaveBeenCalledWith("window");
    expect(usePieStore.getState().mode).toBe("window");
  });

  it("surfaces a start failure via the shell status and stays stopped", async () => {
    vi.mocked(pie.start).mockRejectedValueOnce(new Error("no player binary"));
    await usePieStore.getState().play("embedded");
    expect(usePieStore.getState().running).toBe(false);
    expect(useShellStore.getState().statusMessage).toContain("no player binary");
  });

  it("play() while running resumes when paused, else no-ops", async () => {
    await usePieStore.getState().play("embedded");
    await usePieStore.getState().pause();
    expect(usePieStore.getState().paused).toBe(true);
    vi.mocked(pie.start).mockClear();

    await usePieStore.getState().play("embedded");
    expect(vi.mocked(pie.start)).not.toHaveBeenCalled();
    expect(vi.mocked(pie.resume)).toHaveBeenCalledOnce();
    expect(usePieStore.getState().paused).toBe(false);
  });

  it("pause/resume/step drive the backend and toggle paused", async () => {
    await usePieStore.getState().play("embedded");

    await usePieStore.getState().pause();
    expect(vi.mocked(pie.pause)).toHaveBeenCalledOnce();
    expect(usePieStore.getState().paused).toBe(true);

    await usePieStore.getState().step();
    expect(vi.mocked(pie.step)).toHaveBeenCalledOnce();
    expect(usePieStore.getState().paused).toBe(true);

    await usePieStore.getState().resume();
    expect(vi.mocked(pie.resume)).toHaveBeenCalledOnce();
    expect(usePieStore.getState().paused).toBe(false);
  });

  it("stop ends the session and clears mode", async () => {
    await usePieStore.getState().play("embedded");
    await usePieStore.getState().stop();
    expect(vi.mocked(pie.stop)).toHaveBeenCalledOnce();
    expect(usePieStore.getState().running).toBe(false);
    expect(usePieStore.getState().mode).toBe("");
  });

  it("a crash on pie://state surfaces a toast and returns to stopped", () => {
    usePieStore.getState().syncState(stateEvent({ running: true, mode: "embedded" }));
    expect(usePieStore.getState().running).toBe(true);

    usePieStore
      .getState()
      .syncState(stateEvent({ running: false, error: "PIE player crashed: panic" }));
    expect(usePieStore.getState().running).toBe(false);
    expect(usePieStore.getState().mode).toBe("");
    expect(useShellStore.getState().statusMessage).toContain("crashed");
  });

  it("syncState mirrors running/paused/frame from the backend", () => {
    usePieStore
      .getState()
      .syncState(stateEvent({ running: true, paused: true, mode: "window", frame: 42 }));
    const s = usePieStore.getState();
    expect(s.running).toBe(true);
    expect(s.paused).toBe(true);
    expect(s.mode).toBe("window");
    expect(s.frame).toBe(42);
  });
});
