// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";
import { __resetCommandsForTest, registerCommand } from "../commands";
import {
  __resetKeybindingsForTest,
  bindKey,
  chordOf,
  installKeybindingListener,
  registerDefaultKeybindings,
  bindingFor,
} from "../keybindings";

afterEach(() => {
  __resetKeybindingsForTest();
  __resetCommandsForTest();
});

const kbd = (init: KeyboardEventInit) => new KeyboardEvent("keydown", { bubbles: true, ...init });

describe("chordOf", () => {
  it("normalizes modifier order and key names", () => {
    expect(chordOf(kbd({ key: "p", ctrlKey: true, shiftKey: true }))).toBe("Ctrl+Shift+P");
    expect(chordOf(kbd({ key: " ", ctrlKey: true }))).toBe("Ctrl+Space");
    expect(chordOf(kbd({ key: "F11" }))).toBe("F11");
  });

  it("ignores bare modifier presses", () => {
    expect(chordOf(kbd({ key: "Control", ctrlKey: true }))).toBeNull();
    expect(chordOf(kbd({ key: "Shift", shiftKey: true }))).toBeNull();
  });
});

describe("listener dispatch", () => {
  it("executes the bound command and preventDefaults", () => {
    const run = vi.fn();
    registerCommand({ id: "t.palette", title: "Palette", category: "T", run });
    bindKey({ chord: "Ctrl+Shift+P", command: "t.palette", allowInInputs: true });
    const off = installKeybindingListener();
    const ev = kbd({ key: "P", ctrlKey: true, shiftKey: true, cancelable: true });
    window.dispatchEvent(ev);
    expect(run).toHaveBeenCalledOnce();
    expect(ev.defaultPrevented).toBe(true);
    off();
  });

  it("skips editable targets unless the binding opts in", () => {
    const run = vi.fn();
    registerCommand({ id: "t.save", title: "Save", category: "T", run });
    bindKey({ chord: "Ctrl+S", command: "t.save" });
    const off = installKeybindingListener();

    const input = document.createElement("input");
    document.body.appendChild(input);
    input.dispatchEvent(kbd({ key: "s", ctrlKey: true, cancelable: true }));
    expect(run).not.toHaveBeenCalled();

    window.dispatchEvent(kbd({ key: "s", ctrlKey: true, cancelable: true }));
    expect(run).toHaveBeenCalledOnce();
    off();
    input.remove();
  });
});

describe("defaults", () => {
  it("registers the Phase 1 chords", () => {
    registerDefaultKeybindings();
    expect(bindingFor("Ctrl+Shift+P")?.command).toBe("tools.commandPalette");
    expect(bindingFor("Ctrl+Space")?.command).toBe("window.contentDrawer");
    expect(bindingFor("F11")?.command).toBe("window.fullscreen");
  });
});
