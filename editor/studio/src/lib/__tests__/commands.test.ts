import { afterEach, describe, expect, it, vi } from "vitest";
import {
  __resetCommandsForTest,
  allCommands,
  executeCommand,
  getCommand,
  onCommandsChanged,
  registerCommand,
  registerCommands,
  setCommandHandler,
  setUnhandledCommandHook,
} from "../commands";
import { MENU_BAR, menuCommandDefs } from "../menus";

afterEach(() => {
  __resetCommandsForTest();
});

describe("command registry", () => {
  it("registers and executes commands", () => {
    const run = vi.fn();
    registerCommand({ id: "t.hello", title: "Hello", category: "Test", run });
    expect(executeCommand("t.hello")).toBe(true);
    expect(run).toHaveBeenCalledOnce();
  });

  it("returns false for unknown ids", () => {
    expect(executeCommand("t.missing")).toBe(false);
  });

  it("routes handler-less commands to the unhandled hook", () => {
    const unhandled = vi.fn();
    setUnhandledCommandHook(unhandled);
    registerCommand({ id: "t.stub", title: "Stub", category: "Test" });
    expect(executeCommand("t.stub")).toBe(true);
    expect(unhandled).toHaveBeenCalledWith(expect.objectContaining({ id: "t.stub" }));
  });

  it("setCommandHandler upgrades a stub in place", () => {
    registerCommand({ id: "t.later", title: "Later", category: "Test" });
    const run = vi.fn();
    setCommandHandler("t.later", run);
    executeCommand("t.later");
    expect(run).toHaveBeenCalledOnce();
    expect(getCommand("t.later")!.title).toBe("Later");
  });

  it("notifies listeners on registration", () => {
    const listener = vi.fn();
    const off = onCommandsChanged(listener);
    registerCommands([{ id: "t.a", title: "A", category: "Test" }]);
    expect(listener).toHaveBeenCalled();
    off();
  });
});

describe("menu tree", () => {
  it("covers the full UE-parity top-level set", () => {
    expect(MENU_BAR.map((m) => m.label)).toEqual([
      "File",
      "Edit",
      "Window",
      "Tools",
      "Build",
      "Platforms",
      "Select",
      "Actor",
      "Help",
    ]);
  });

  it("flattens to command defs with unique ids", () => {
    const defs = menuCommandDefs();
    expect(defs.length).toBeGreaterThan(60);
    const ids = defs.map((d) => d.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const def of defs) {
      expect(def.title).toBeTruthy();
      expect(def.category).toBeTruthy();
    }
  });

  it("registers every menu command into the registry", async () => {
    const { registerMenuCommands } = await import("../menus");
    registerMenuCommands();
    for (const def of menuCommandDefs()) {
      expect(getCommand(def.id), `missing ${def.id}`).toBeDefined();
    }
    // Theme commands are live (have handlers).
    expect(getCommand("theme.infinity-dark")!.run).toBeDefined();
    expect(allCommands().length).toBeGreaterThanOrEqual(menuCommandDefs().length);
  });
});
