// @vitest-environment jsdom
/**
 * Wave E batch A's wiring arm: the two settings entries in the Edit menu, and
 * the Window ▸ Audio Mixer entry, are LIVE commands.
 *
 * The failure this catches is the exact state the wave found: a menu item and a
 * toolbar gear enumerated from `menuCommandDefs()` with no handler, so
 * `executeCommand` fell through to the unhandled hook and toasted "…is not
 * implemented yet." Asserting only "the command exists" would have passed in
 * that state — the assertion has to be that the hook is NOT reached.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

import { executeCommand, setUnhandledCommandHook, getCommand } from "../../lib/commands";
import { MENU_BAR } from "../../lib/menus";
import { useShellStore } from "../../stores/shellStore";
import { bootstrapShellCommands, stubHint } from "../shellCommands";

const unhandled = vi.fn();

describe("settings commands are wired end to end", () => {
  beforeEach(() => {
    bootstrapShellCommands();
    unhandled.mockReset();
    setUnhandledCommandHook(unhandled);
    useShellStore.setState({ preferencesOpen: false, projectSettingsOpen: false });
  });

  it("edit.editorPreferences opens the dialog and never reaches the unhandled hook", () => {
    expect(executeCommand("edit.editorPreferences")).toBe(true);
    expect(useShellStore.getState().preferencesOpen).toBe(true);
    expect(unhandled).not.toHaveBeenCalled();
  });

  it("edit.projectSettings opens the project dialog, not the preferences one", () => {
    expect(executeCommand("edit.projectSettings")).toBe(true);
    expect(useShellStore.getState().projectSettingsOpen).toBe(true);
    expect(useShellStore.getState().preferencesOpen).toBe(false);
    expect(unhandled).not.toHaveBeenCalled();
  });

  it("both are reachable from the Edit menu (the palette lists what the menu holds)", () => {
    const edit = MENU_BAR.find((m) => m.id === "edit");
    const ids = (edit?.items ?? [])
      .filter((n): n is Extract<typeof n, { kind: "action" }> => n.kind === "action")
      .map((n) => n.command);
    expect(ids).toContain("edit.editorPreferences");
    expect(ids).toContain("edit.projectSettings");
  });

  it("window.audioMixer is in the Window menu AND has a handler", () => {
    const win = MENU_BAR.find((m) => m.id === "window");
    const ids = (win?.items ?? [])
      .filter((n): n is Extract<typeof n, { kind: "action" }> => n.kind === "action")
      .map((n) => n.command);
    expect(ids).toContain("window.audioMixer");
    expect(getCommand("window.audioMixer")?.run).toBeTypeOf("function");
  });

  it("the two genuinely-unbuilt settings entries say what exists instead", () => {
    // `edit.plugins` and `platforms.packagingSettings` are NOT built. They must
    // not pretend to be — but the generic "is not implemented yet" is replaced
    // by a message that names the door that does exist.
    expect(getCommand("edit.plugins")?.run).toBeUndefined();
    expect(stubHint("edit.plugins")).toMatch(/cargo crates/);
    expect(stubHint("platforms.packagingSettings")).toMatch(/Package Project/);
    // And the two that ARE built have no stub hint at all.
    expect(stubHint("edit.editorPreferences")).toBeUndefined();
    expect(stubHint("edit.projectSettings")).toBeUndefined();
  });
});
