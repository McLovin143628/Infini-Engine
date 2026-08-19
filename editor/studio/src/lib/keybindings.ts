/**
 * Keybinding registry (ROADMAP P1.4.5). Chords are normalized strings —
 * `"Ctrl+Shift+P"`, `"F11"`, `"Ctrl+Space"` — mapped to command ids and
 * dispatched through the command registry.
 *
 * The Phase-1 promise that "rebinding UI + persistence arrive with editor
 * preferences" is kept by Wave E: {@link DEFAULT_KEYBINDINGS} is the shipped
 * table, `editor-settings.toml` stores only the user's OVERRIDES, and
 * {@link applyKeybindingOverrides} composes the two. The live map is still a
 * process-lifetime `Map` — it is a projection of the file, never the record.
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

/**
 * The shipped default chords (grows with each phase's commands).
 *
 * Data, not code, since Wave E: the Preferences dialog renders this list beside
 * the user's overrides and "Reset" means "delete the override", so the day a
 * default changes every user gets it — an override file that copied the whole
 * table would freeze the defaults of the version that wrote it.
 */
export const DEFAULT_KEYBINDINGS: readonly Keybinding[] = [
  { chord: "Ctrl+Shift+P", command: "tools.commandPalette", allowInInputs: true },
  { chord: "Ctrl+Space", command: "window.contentDrawer" },
  { chord: "F11", command: "window.fullscreen" },
  { chord: "Ctrl+S", command: "file.saveLevel" },
  { chord: "Ctrl+Alt+S", command: "file.saveLevelAs" },
  { chord: "Ctrl+Shift+S", command: "file.saveAll" },
  { chord: "Ctrl+Z", command: "edit.undo" },
  { chord: "Ctrl+Y", command: "edit.redo" },
  // Scene clipboard + duplicate (editor seams). The listener skips editable
  // targets, so these never steal Ctrl+C/X/V from text fields.
  { chord: "Ctrl+D", command: "edit.duplicate" },
  { chord: "Ctrl+C", command: "edit.copy" },
  { chord: "Ctrl+X", command: "edit.cut" },
  { chord: "Ctrl+V", command: "edit.paste" },
  // Wave E: the object-editor routing (Actor ▸ Edit …) and rename.
  { chord: "F2", command: "actor.rename" },
] as const;

/** Install the defaults over whatever is bound now. */
export function registerDefaultKeybindings(): void {
  for (const b of DEFAULT_KEYBINDINGS) bindKey({ ...b });
}

/**
 * The chords the last {@link applyKeybindingOverrides} installed.
 *
 * **The live map holds chords no settings file knows about.** `simStore` binds
 * `Alt+P` and `pieStore` binds `Shift+Alt+P` at bootstrap, before the
 * preferences file has resolved; nothing else registers them, and no menu id
 * dispatches them. The Wave-E audit found that rebuilding the map by CLEARING
 * it deleted both the instant `initSettingsSync` applied its first (usually
 * empty) override set — Simulate and PIE silently lost their shortcuts on every
 * launch. So an apply undoes exactly what a previous apply did, and owns nothing
 * it did not install.
 */
let appliedOverrideChords: string[] = [];

/**
 * Rebuild the live map as `defaults + overrides` (Wave E).
 *
 * `overrides` is chord → command id, straight out of `editor-settings.toml`.
 * An **empty** command id means "this chord is unbound" — that is how a user
 * removes a default without the file having to enumerate the defaults it keeps.
 *
 * `allowInInputs` is inherited from the default binding of the SAME COMMAND, so
 * rebinding the palette to another chord keeps working inside a text field.
 *
 * Idempotent without being destructive: the previous override set is unbound
 * first, then the defaults are re-installed (which restores anything a previous
 * override had unbound), then the new set is applied. Applying twice, or
 * applying a smaller set after a larger one, always lands on exactly
 * `defaults + overrides + whatever other subsystems bound` — and that last term
 * is why this is not a `bindings.clear()`.
 */
export function applyKeybindingOverrides(overrides: Record<string, string>): void {
  for (const chord of appliedOverrideChords) unbindKey(chord);
  appliedOverrideChords = [];
  registerDefaultKeybindings();
  for (const [chord, command] of Object.entries(overrides)) {
    if (!chord) continue;
    appliedOverrideChords.push(chord);
    if (!command) {
      unbindKey(chord);
      continue;
    }
    const inherited = DEFAULT_KEYBINDINGS.find((b) => b.command === command)?.allowInInputs;
    bindKey({ chord, command, ...(inherited ? { allowInInputs: true } : {}) });
  }
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
  appliedOverrideChords = [];
}
