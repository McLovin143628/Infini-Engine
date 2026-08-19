// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../lib/ipc", () => ({
  scene: { autosave: vi.fn() },
  editorSettings: { get: vi.fn(), set: vi.fn() },
}));

import { editorSettings as settingsIpc, scene as sceneIpc } from "../../lib/ipc";
import { autosaveIntervalMs, initAutosave } from "../autosave";
import { __resetSettingsStoreForTest, useSettingsStore } from "../settingsStore";
import { useShellStore } from "../shellStore";

const autosaveMock = vi.mocked(sceneIpc.autosave);

describe("autosaveIntervalMs — the door a preference reaches a timer through", () => {
  it("converts seconds to ms and clamps", () => {
    expect(autosaveIntervalMs(5)).toBe(5000);
    expect(autosaveIntervalMs(0.1)).toBe(1000);
    expect(autosaveIntervalMs(100000)).toBe(3_600_000);
  });

  it("refuses NaN, ±Infinity and zero — setInterval(NaN) spins the event loop", () => {
    expect(autosaveIntervalMs(Number.NaN)).toBe(5000);
    expect(autosaveIntervalMs(Number.POSITIVE_INFINITY)).toBe(5000);
    expect(autosaveIntervalMs(0)).toBe(5000);
    expect(autosaveIntervalMs(-5)).toBe(5000);
  });
});

describe("initAutosave — the setting APPLIES, it does not merely persist", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    autosaveMock.mockReset();
    autosaveMock.mockResolvedValue(undefined);
    vi.mocked(settingsIpc.set).mockReset();
    vi.mocked(settingsIpc.set).mockImplementation((s) => Promise.resolve(s));
    __resetSettingsStoreForTest();
    useShellStore.setState({ statusMessage: null });
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("saves at the default period, then RE-ARMS when the preference changes", async () => {
    const dispose = initAutosave();

    // Default 5 s: one save in 6 s.
    await vi.advanceTimersByTimeAsync(6_000);
    expect(autosaveMock).toHaveBeenCalledTimes(1);

    // Shorten it to 1 s. The next 3 s must produce THREE more saves — at the
    // old period they would have produced zero.
    autosaveMock.mockClear();
    useSettingsStore.getState().patch({ autosave_interval_s: 1 });
    await vi.advanceTimersByTimeAsync(3_100);
    expect(autosaveMock.mock.calls.length).toBeGreaterThanOrEqual(3);

    dispose();
  });

  it("stops on dispose", async () => {
    const dispose = initAutosave();
    dispose();
    await vi.advanceTimersByTimeAsync(30_000);
    expect(autosaveMock).not.toHaveBeenCalled();
  });

  it("does not re-arm when the value is unchanged (a patch of something else)", async () => {
    const dispose = initAutosave();
    await vi.advanceTimersByTimeAsync(4_000); // 4 s into the 5 s period
    useSettingsStore.getState().patch({ theme_id: "midnight" });
    await vi.advanceTimersByTimeAsync(1_500); // crosses 5 s
    // Re-arming would have reset the countdown and produced NO save here.
    expect(autosaveMock).toHaveBeenCalledTimes(1);
    dispose();
  });

  it("a persistent failure toasts once, not every tick", async () => {
    autosaveMock.mockRejectedValue(new Error("disk full"));
    const pushStatus = vi.spyOn(useShellStore.getState(), "pushStatus");
    const dispose = initAutosave();
    useSettingsStore.getState().patch({ autosave_interval_s: 1 });

    await vi.advanceTimersByTimeAsync(1_100);
    expect(pushStatus).toHaveBeenCalledTimes(1);
    expect(pushStatus.mock.calls[0][0]).toContain("disk full");

    // Nine more failures with the SAME message: still one toast (the rate limit
    // re-toasts only on a changed message or every 12th tick).
    await vi.advanceTimersByTimeAsync(9_000);
    expect(autosaveMock.mock.calls.length).toBeGreaterThanOrEqual(9);
    expect(pushStatus).toHaveBeenCalledTimes(1);

    pushStatus.mockRestore();
    dispose();
  });
});
