// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { EditorSettings } from "../../bindings/EditorSettings";

vi.mock("../ipc", () => ({
  editorSettings: { get: vi.fn(), set: vi.fn() },
}));

// The two viewport doors are mocked: `viewportStore` pulls in the whole native
// viewport IPC surface, and what this file is testing is that the doors are
// CALLED with the file's values — not what the viewport does with them.
vi.mock("../../stores/viewportStore", () => ({
  applySnap3dFromSettings: vi.fn(),
  applyFoliageFromSettings: vi.fn(),
}));

import { editorSettings as settingsIpc } from "../ipc";
import {
  applyFoliageFromSettings,
  applySnap3dFromSettings,
} from "../../stores/viewportStore";
import {
  DEFAULT_EDITOR_SETTINGS,
  __resetSettingsStoreForTest,
  useSettingsStore,
} from "../../stores/settingsStore";
import { __resetKeybindingsForTest, bindingFor, registerDefaultKeybindings } from "../keybindings";
import {
  PREFS_MIGRATED_KEY,
  applySettings,
  foldLegacyPreferences,
  initSettingsSync,
} from "../settingsApply";

const getMock = vi.mocked(settingsIpc.get);
const setMock = vi.mocked(settingsIpc.set);

const base = (): EditorSettings => structuredClone(DEFAULT_EDITOR_SETTINGS);

describe("foldLegacyPreferences — the localStorage migration runs exactly once", () => {
  beforeEach(() => {
    localStorage.clear();
    __resetSettingsStoreForTest();
  });

  it("preserves a non-default theme, then never looks again", () => {
    localStorage.setItem("infinity:theme", "midnight");

    const first = foldLegacyPreferences(base());
    expect(first).not.toBeNull();
    expect(first!.theme_id).toBe("midnight");
    expect(localStorage.getItem(PREFS_MIGRATED_KEY)).toBe("1");

    // The user later picks graphite; the stale legacy key must NOT win again.
    localStorage.setItem("infinity:theme", "midnight");
    const second = foldLegacyPreferences({ ...base(), theme_id: "graphite" });
    expect(second).toBeNull();
  });

  it("folds the tour flag and both viewport blobs, and removes those keys", () => {
    localStorage.setItem("infinity:tourSeen", "1");
    localStorage.setItem(
      "inf.viewport.snap3d",
      JSON.stringify({ translate: 0.25, rotate_deg: 5, scale: 0.05, always_on: true }),
    );
    localStorage.setItem(
      "inf.viewport.foliage",
      JSON.stringify({ radius: 9, density: 0.75, erase: true, kind: 3, scale_jitter: 0.4, seed: 7 }),
    );

    const folded = foldLegacyPreferences(base());
    expect(folded).not.toBeNull();
    expect(folded!.tour_seen).toBe(true);
    expect(folded!.snap_3d).toEqual({
      translate: 0.25,
      rotate_deg: 5,
      scale: 0.05,
      always_on: true,
    });
    expect(folded!.foliage.radius).toBe(9);
    expect(folded!.foliage.kind).toBe(3);
    expect(folded!.foliage.seed).toBe(7);
    // `align_to_normal` was absent from the legacy blob — it defaults, not NaN.
    expect(folded!.foliage.align_to_normal).toBe(false);

    expect(localStorage.getItem("infinity:tourSeen")).toBeNull();
    expect(localStorage.getItem("inf.viewport.snap3d")).toBeNull();
    expect(localStorage.getItem("inf.viewport.foliage")).toBeNull();
    // The theme key STAYS — it is the pre-paint cache `main.tsx` reads.
    expect(localStorage.getItem(PREFS_MIGRATED_KEY)).toBe("1");
  });

  it("a fresh install has nothing to fold and still marks itself migrated", () => {
    expect(foldLegacyPreferences(base())).toBeNull();
    expect(localStorage.getItem(PREFS_MIGRATED_KEY)).toBe("1");
  });

  it("a corrupt legacy blob does not produce NaN fields", () => {
    localStorage.setItem("inf.viewport.snap3d", "{not json");
    localStorage.setItem("inf.viewport.foliage", JSON.stringify({ radius: "wide" }));
    const folded = foldLegacyPreferences(base());
    // The unreadable snap blob is skipped; the foliage blob's bad field defaults.
    expect(folded!.snap_3d).toEqual(DEFAULT_EDITOR_SETTINGS.snap_3d);
    expect(folded!.foliage.radius).toBe(DEFAULT_EDITOR_SETTINGS.foliage.radius);
  });
});

describe("applySettings — what actually applies live", () => {
  beforeEach(() => {
    localStorage.clear();
    __resetKeybindingsForTest();
    registerDefaultKeybindings();
    vi.mocked(applySnap3dFromSettings).mockClear();
    vi.mocked(applyFoliageFromSettings).mockClear();
  });

  it("the theme flips the document, live", () => {
    applySettings({ ...base(), theme_id: "infinity-light" }, null);
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    applySettings(
      { ...base(), theme_id: "midnight" },
      { ...base(), theme_id: "infinity-light" },
    );
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
  });

  it("keybinding overrides replace a default chord and unbind on an empty command", () => {
    expect(bindingFor("Ctrl+Shift+P")?.command).toBe("tools.commandPalette");

    applySettings(
      { ...base(), keybindings: { "Ctrl+Alt+P": "tools.commandPalette", "Ctrl+Shift+P": "" } },
      null,
    );
    expect(bindingFor("Ctrl+Alt+P")?.command).toBe("tools.commandPalette");
    // `allowInInputs` is inherited from the default binding of the same command.
    expect(bindingFor("Ctrl+Alt+P")?.allowInInputs).toBe(true);
    expect(bindingFor("Ctrl+Shift+P")).toBeUndefined();

    // Removing the override restores the shipped default — the point of storing
    // only overrides rather than the whole table.
    applySettings({ ...base(), keybindings: {} }, {
      ...base(),
      keybindings: { "Ctrl+Alt+P": "tools.commandPalette", "Ctrl+Shift+P": "" },
    });
    expect(bindingFor("Ctrl+Shift+P")?.command).toBe("tools.commandPalette");
    expect(bindingFor("Ctrl+Alt+P")).toBeUndefined();
  });

  it("snap + foliage reach the viewport doors, and only when they CHANGE", () => {
    const prev = base();
    const next = { ...prev, snap_3d: { ...prev.snap_3d, translate: 0.5 } };
    applySettings(next, prev);
    expect(applySnap3dFromSettings).toHaveBeenCalledWith(next.snap_3d);
    expect(applyFoliageFromSettings).not.toHaveBeenCalled();

    // Same VALUES in a fresh object (what `viewportStore.patch` produces on every
    // keystroke) must not re-push to the native viewport.
    vi.mocked(applySnap3dFromSettings).mockClear();
    applySettings({ ...next, snap_3d: { ...next.snap_3d } }, next);
    expect(applySnap3dFromSettings).not.toHaveBeenCalled();
  });
});

describe("initSettingsSync", () => {
  beforeEach(() => {
    localStorage.clear();
    getMock.mockReset();
    setMock.mockReset();
    setMock.mockImplementation((s: EditorSettings) => Promise.resolve(s));
    __resetSettingsStoreForTest();
  });
  afterEach(() => {
    __resetSettingsStoreForTest();
  });

  it("loads the file, applies it, and writes nothing when there is no migration", async () => {
    getMock.mockResolvedValue({ ...base(), theme_id: "graphite" });
    const dispose = initSettingsSync();
    await vi.waitFor(() => expect(useSettingsStore.getState().loaded).toBe(true));
    expect(useSettingsStore.getState().settings.theme_id).toBe("graphite");
    expect(setMock).not.toHaveBeenCalled();
    dispose();
  });

  it("writes exactly once when a legacy preference is folded in", async () => {
    localStorage.setItem("infinity:theme", "midnight");
    getMock.mockResolvedValue(base());
    const dispose = initSettingsSync();
    await vi.waitFor(() => expect(setMock).toHaveBeenCalledTimes(1));
    expect(setMock.mock.calls[0][0].theme_id).toBe("midnight");
    expect(useSettingsStore.getState().settings.theme_id).toBe("midnight");
    dispose();
  });

  /**
   * The corruption law, at the seam where it can actually be broken: a refused
   * load must not be followed by a write, or the first launch after a typo in
   * the file replaces the user's preferences with defaults.
   */
  it("a REFUSED load surfaces the message and writes NOTHING", async () => {
    getMock.mockRejectedValue(new Error("editor-settings.toml exists but cannot be read"));
    const dispose = initSettingsSync();
    await vi.waitFor(() => expect(useSettingsStore.getState().loaded).toBe(true));
    expect(useSettingsStore.getState().error).toContain("cannot be read");
    expect(useSettingsStore.getState().settings).toEqual(DEFAULT_EDITOR_SETTINGS);
    expect(setMock).not.toHaveBeenCalled();
    dispose();
  });
});
