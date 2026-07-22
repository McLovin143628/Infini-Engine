/**
 * Command registry — the single dispatch point for every user-invokable
 * action (menu items, toolbar buttons, the P1.4 command palette, and the
 * keybinding registry all route through here).
 *
 * Commands are registered by id (`"file.saveAll"`, `"play.start"`, …).
 * A command may exist without a handler — the full UE-parity menu tree is
 * enumerated from Phase 1 but most actions land in later phases; executing
 * an unimplemented command routes to the `unhandled` hook so the shell can
 * surface "coming in Phase N" instead of silently doing nothing.
 */

export interface CommandDef {
  id: string;
  /** Palette / tooltip title, e.g. "Save All". */
  title: string;
  /** Palette grouping, e.g. "File", "Build", "View". */
  category: string;
  /** Display-only accelerator, e.g. "Ctrl+Shift+S". Binding wiring is P1.4. */
  shortcut?: string;
  /** Missing ⇒ enumerated-but-stubbed; execution goes to the unhandled hook. */
  run?: () => void | Promise<void>;
}

type Listener = () => void;

const registry = new Map<string, CommandDef>();
const listeners = new Set<Listener>();

function errText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

let unhandledHook: (cmd: CommandDef) => void = (cmd) => {
  console.warn(`command not implemented yet: ${cmd.id}`);
};

/** Shell installs a hook to surface unimplemented commands (status toast). */
export function setUnhandledCommandHook(hook: (cmd: CommandDef) => void): void {
  unhandledHook = hook;
}

export function registerCommand(def: CommandDef): void {
  registry.set(def.id, def);
  emit();
}

export function registerCommands(defs: CommandDef[]): void {
  for (const def of defs) registry.set(def.id, def);
  emit();
}

/** Attach/replace the handler of an already-enumerated command. */
export function setCommandHandler(id: string, run: CommandDef["run"]): void {
  const existing = registry.get(id);
  if (!existing) {
    console.warn(`setCommandHandler: unknown command ${id}`);
    return;
  }
  registry.set(id, { ...existing, run });
  emit();
}

export function getCommand(id: string): CommandDef | undefined {
  return registry.get(id);
}

export function allCommands(): CommandDef[] {
  return [...registry.values()];
}

/** Run a command. Returns false when the id is unknown. */
export function executeCommand(id: string): boolean {
  const cmd = registry.get(id);
  if (!cmd) {
    console.warn(`executeCommand: unknown command ${id}`);
    return false;
  }
  if (!cmd.run) {
    unhandledHook(cmd);
    return true;
  }
  // Run SYNCHRONOUSLY (callers + keybindings rely on it), but surface a
  // rejected/throwing handler that used to vanish silently (e.g. a failed scene
  // save) in the status bar with the command title. Handles both a synchronous
  // throw and an async rejection.
  const surface = (e: unknown) => {
    console.error(`command ${cmd.id} failed`, e);
    // Lazy import avoids a module-load cycle (shellStore ⇄ command wiring).
    void import("../stores/shellStore").then(({ useShellStore }) =>
      useShellStore.getState().pushStatus(`${cmd.title} failed: ${errText(e)}`),
    );
  };
  try {
    const result = cmd.run();
    if (result && typeof (result as Promise<void>).then === "function") {
      void (result as Promise<void>).catch(surface);
    }
  } catch (e) {
    surface(e);
  }
  return true;
}

/** Subscribe to registry changes (the palette re-lists on registration). */
export function onCommandsChanged(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function emit(): void {
  for (const l of listeners) l();
}

/** Test-only: reset global registry state between test cases. */
export function __resetCommandsForTest(): void {
  registry.clear();
  listeners.clear();
  unhandledHook = (cmd) => {
    console.warn(`command not implemented yet: ${cmd.id}`);
  };
}
