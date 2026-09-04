// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { EditorSettings } from "../../bindings/EditorSettings";

// The typed IPC layer is mocked so the debounced write is observable and no
// real Tauri `invoke` is attempted (`sceneStoreSettings.test.ts` precedent).
vi.mock("../../lib/ipc", () => ({
  editorSettings: {
    get: vi.fn(),
    set: vi.fn(),
  },
}));

import { editorSettings as settingsIpc } from "../../lib/ipc";
import {
  DEFAULT_EDITOR_SETTINGS,
  SETTINGS_WRITE_DEBOUNCE_MS,
  __resetSettingsStoreForTest,
  useSettingsStore,
} from "../settingsStore";

const store = () => useSettingsStore.getState();
const setMock = vi.mocked(settingsIpc.set);

/** Echo whatever was sent, as a well-behaved backend would. */
const echo = () => setMock.mockImplementation((s: EditorSettings) => Promise.resolve(s));

describe("settingsStore", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    setMock.mockReset();
    echo();
    __resetSettingsStoreForTest();
  });
  afterEach(() => {
    __resetSettingsStoreForTest();
    vi.useRealTimers();
  });

  it("patch is live immediately and writes once after the debounce", async () => {
    store().patch({ theme_id: "midnight" });
    store().patch({ theme_id: "graphite" });
    store().patch({ autosave_interval_s: 30 });

    // Live in memory before any write has happened.
    expect(store().settings.theme_id).toBe("graphite");
    expect(store().settings.autosave_interval_s).toBe(30);
    expect(setMock).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(SETTINGS_WRITE_DEBOUNCE_MS + 1);

    // THREE edits, ONE write — and the write carries the LATEST values, not the
    // ones captured when the first timer was scheduled.
    expect(setMock).toHaveBeenCalledTimes(1);
    const sent = setMock.mock.calls[0][0];
    expect(sent.theme_id).toBe("graphite");
    expect(sent.autosave_interval_s).toBe(30);
  });

  it("hydrates from the REPLY, so a clamped value reaches the UI", async () => {
    // A backend that normalizes: the store must show 3600, not the 1e9 it sent.
    setMock.mockImplementation((s: EditorSettings) =>
      Promise.resolve({ ...s, autosave_interval_s: 3600 }),
    );
    store().patch({ autosave_interval_s: 1e9 });
    await vi.advanceTimersByTimeAsync(SETTINGS_WRITE_DEBOUNCE_MS + 1);
    expect(store().settings.autosave_interval_s).toBe(3600);
  });

  it("a failed write surfaces an error and keeps the edited values", async () => {
    setMock.mockRejectedValue(new Error("disk full"));
    store().patch({ theme_id: "midnight" });
    await vi.advanceTimersByTimeAsync(SETTINGS_WRITE_DEBOUNCE_MS + 1);
    expect(store().error).toContain("disk full");
    expect(store().settings.theme_id).toBe("midnight");
  });

  it("a SYNCHRONOUS invoke throw is a value, not an uncaught timer error", async () => {
    setMock.mockImplementation(() => {
      throw new Error("not a tauri window");
    });
    store().patch({ theme_id: "midnight" });
    await vi.advanceTimersByTimeAsync(SETTINGS_WRITE_DEBOUNCE_MS + 1);
    expect(store().error).toContain("not a tauri window");
  });

  it("flush writes a pending edit immediately", async () => {
    store().patch({ theme_id: "midnight" });
    expect(setMock).not.toHaveBeenCalled();
    await store().flush();
    expect(setMock).toHaveBeenCalledTimes(1);
    expect(setMock.mock.calls[0][0].theme_id).toBe("midnight");
  });

  it("resetToDefaults writes the shipped defaults and cancels a pending edit", async () => {
    store().patch({ theme_id: "midnight", autosave_interval_s: 99 });
    await store().resetToDefaults();
    expect(setMock).toHaveBeenCalledTimes(1);
    expect(setMock.mock.calls[0][0]).toEqual(DEFAULT_EDITOR_SETTINGS);
    expect(store().settings).toEqual(DEFAULT_EDITOR_SETTINGS);
    // The cancelled debounce must not fire afterwards and re-write the old edit.
    await vi.advanceTimersByTimeAsync(SETTINGS_WRITE_DEBOUNCE_MS * 4);
    expect(setMock).toHaveBeenCalledTimes(1);
  });

  /**
   * **A cancelled debounce must not leave an un-resolvable promise behind**
   * (Wave E audit, A4). `resetToDefaults` cleared the timer but not the promise
   * the timer was going to resolve, so the next `flush()` — which the
   * Preferences dialog's Close button calls — took the `await pending` branch on
   * a promise nobody would ever settle. Nothing throws; the await simply never
   * returns.
   */
  it("flush after resetToDefaults settles instead of awaiting a dead promise", async () => {
    store().patch({ theme_id: "midnight" });
    await store().resetToDefaults();
    await Promise.race([
      store().flush(),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("flush never settled")), 1000),
      ),
    ]);
    // The reset's write is the only one; flush had nothing pending to send.
    expect(setMock).toHaveBeenCalledTimes(1);
  });

  it("the frontend defaults mirror the Rust defaults field for field", () => {
    // `EditorSettings::default()` in `editor/crates/inf-editor-core/src/
    // editor_settings.rs` is asserted there by `defaults_match_the_shipped_feel`;
    // this is the other half of the same claim, so the two cannot drift silently.
    expect(DEFAULT_EDITOR_SETTINGS).toEqual({
      schema_version: 1,
      theme_id: "infinity-dark",
      tour_seen: false,
      autosave_interval_s: 5,
      camera_fly_speed_mps: 8,
      camera_look_sensitivity: 1,
      rmb_click_travel_px: 4,
      rmb_click_ms: 250,
      // IB-14: `inf_gis::DEFAULT_MAX_ENTITIES`.
      gis_max_entities: 4096,
      // CERT1: the boot project, empty meaning "not pinned". This arm is what
      // caught the frontend default missing the field at all -- `toEqual` is a
      // DEEP equality, so an added Rust field fails here as well as in tsc.
      boot_project: "",
      snap_3d: { translate: 1, rotate_deg: 15, scale: 0.1, always_on: false },
      foliage: {
        radius: 3,
        density: 0.4,
        erase: false,
        kind: 0,
        scale_jitter: 0.2,
        align_to_normal: false,
        seed: 1,
      },
      keybindings: {},
      // Island wave I5: the GAME's binding overrides, a different map from the
      // editor's chords above — that one is Ctrl+Shift+P, this one is C.
      game_bindings: {},
    });
  });
});
