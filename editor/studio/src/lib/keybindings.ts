/**
 * Keybinding registry (ROADMAP P1.4.5). Chords are normalized strings —
 * `"Ctrl+Shift+P"`, `"F11"`, `"Ctrl+Space"` — mapped to command ids and
 * dispatched through the command registry. Rebinding UI + persistence
 * arrive with editor preferences (P5); Phase 1 ships the registry and the
 * default map.
 */
import { executeCommand } from "./commands";

export interface Keybinding {
  chord: string;
  command: string;
  /** Fire even when focus is in an input/textarea (default false). */
  allowInInputs?: boolean;
}

const bindings = new Map<string, Keybinding>();

/** Normalize a KeyboardEvent to a chord string (or null for bare mods). */
export function chordOf(e: KeyboardEvent): string | null {
  const key = e.key;
  if (key === "Control" || key === "Shift" || key === "Alt" || key === "Meta") return null;
  const parts: string[] = [];
  if (e.ctrlKey) parts.push("Ctrl");
  if (e.altKey) parts.push("Alt");
  if (e.shiftKey) parts.push("Shift");
  if (e.metaKey) parts.push("Meta");
  const name =
    key === " " ? "Space" : key.length === 1 ? key.toUpperCase() : key;
  parts.push(name);
  return parts.join("+");
}

export function bindKey(binding: Keybinding): void {
  bindings.set(binding.chord, binding);
}

export function unbindKey(chord: string): void {
  bindings.delete(chord);
}

export function bindingFor(chord: string): Keybinding | undefined {
  return bindings.get(chord);
}

export function allKeybindings(): Keybinding[] {
  return [...bindings.values()];
}

/** The Phase 1 defaults (grows with each phase's commands). */
export function registerDefaultKeybindings(): void {
  bindKey({ chord: "Ctrl+Shift+P", command: "tools.commandPalette", allowInInputs: true });
  bindKey({ chord: "Ctrl+Space", command: "window.contentDrawer" });
  bindKey({ chord: "F11", command: "window.fullscreen" });
  bindKey({ chord: "Ctrl+S", command: "file.saveLevel" });
  bindKey({ chord: "Ctrl+Alt+S", command: "file.saveLevelAs" });
  bindKey({ chord: "Ctrl+Shift+S", command: "file.saveAll" });
  bindKey({ chord: "Ctrl+Z", command: "edit.undo" });
  bindKey({ chord: "Ctrl+Y", command: "edit.redo" });
  // Scene clipboard + duplicate (editor seams). The listener skips editable
  // targets, so these never steal Ctrl+C/X/V from text fields.
  bindKey({ chord: "Ctrl+D", command: "edit.duplicate" });
  bindKey({ chord: "Ctrl+C", command: "edit.copy" });
  bindKey({ chord: "Ctrl+X", command: "edit.cut" });
  bindKey({ chord: "Ctrl+V", command: "edit.paste" });
}

/**
 * Global keydown handler. Installed once by the shell (`App`); returns the
 * uninstaller. Skips editable targets unless the binding opts in.
 */
export function installKeybindingListener(target: Window = window): () => void {
  const onKeyDown = (e: KeyboardEvent) => {
    const chord = chordOf(e);
    if (!chord) return;
    const binding = bindings.get(chord);
    if (!binding) return;
    const el = e.target instanceof HTMLElement ? e.target : null;
    const editable = el?.closest("input, textarea, [contenteditable]") != null;
    if (editable && !binding.allowInInputs) return;
    e.preventDefault();
    executeCommand(binding.command);
  };
  target.addEventListener("keydown", onKeyDown);
  return () => target.removeEventListener("keydown", onKeyDown);
}

/**
 * Dispatch a chord that arrived from outside the DOM — the native viewport
 * forwards global shortcuts here when it holds OS focus (`viewport://key`,
 * P2.3.4), since those key events never reach the webview. Returns true if a
 * binding fired. No editable-target check: the viewport is never a text field.
 */
export function dispatchChord(chord: string): boolean {
  const binding = bindings.get(chord);
  if (!binding) return false;
  executeCommand(binding.command);
  return true;
}

/** Test-only: reset global state between cases. */
export function __resetKeybindingsForTest(): void {
  bindings.clear();
}
