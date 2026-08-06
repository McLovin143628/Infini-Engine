/**
 * **Undo/redo scope routing** (P23.2a) — which document Ctrl+Z talks to.
 *
 * Until now `edit.undo` was wired straight to the scene store, so Ctrl+Z inside
 * the Material, Blueprint or PCG editor undid **the scene**: it moved an actor
 * the author could not see, in a panel they were not looking at, while the graph
 * in front of them did nothing. That is a shipping bug, not a missing feature —
 * the material graph has had a server-side journal since P7.2.
 *
 * The fix is a registry keyed by PANEL TYPE. The dock store records the last
 * panel to receive a pointer-down (`focusedPanel`); `dispatchUndo` looks up that
 * panel's type here and routes to whatever registered for it. **The default is
 * the scene** — nothing registered, no panel focused, or a panel that is part of
 * the level-editing surface (viewport / outliner / details) all mean "undo the
 * level", which is exactly today's behaviour for today's flows.
 *
 * Deliberately narrow: undo and redo only. Per-command scoping (copy, paste,
 * delete) is a bigger design — a panel would have to declare a whole command
 * surface — and this registry can grow into it if that day comes. Widening it
 * speculatively would mean guessing at semantics for a dozen commands nobody has
 * asked about.
 */
import { useDockLayout } from "../panels/dock/dockLayoutStore";
import { panelTypeOf } from "../panels/panelRegistry";

export interface UndoScope {
  /**
   * `params` is the focused panel instance's parameter — for `model:<assetId>`,
   * the asset id.
   *
   * A registry keyed by panel TYPE is one entry per editor, and that is right
   * until an editor is multi-instance: two Model Editors are two `"model"`
   * panels with one registration between them, and without the parameter the
   * chord would reach whichever document the store happened to hold. Optional in
   * practice — the singleton editors ignore it (a zero-argument function is a
   * valid one-argument function in TypeScript), so nothing else had to change.
   */
  undo: (params: string | null) => void | Promise<void>;
  redo: (params: string | null) => void | Promise<void>;
}

const scopes = new Map<string, UndoScope>();

/**
 * Claim `edit.undo` / `edit.redo` for a panel type.
 *
 * Called at the module scope of the owning store, like `registerBridgedStore` —
 * so the registry stays ignorant of the domain stores and there is no import
 * cycle to manage.
 */
export function registerUndoScope(panelType: string, scope: UndoScope): void {
  scopes.set(panelType, scope);
}

/** The scope registered for `panelType`, or `undefined` (⇒ the scene default). */
export function undoScopeFor(panelType: string | null | undefined): UndoScope | undefined {
  return panelType ? scopes.get(panelType) : undefined;
}

/**
 * The scope the focused panel claims, if any.
 *
 * Reads the dock store at call time rather than subscribing: a keybinding fires
 * once and wants the state as it is now, and a subscription here would make an
 * unrelated layout change re-run this module's consumers.
 */
export function activeUndoScope(): UndoScope | undefined {
  const focused = useDockLayout.getState().focusedPanel;
  return undoScopeFor(focused ? panelTypeOf(focused) : null);
}

/**
 * The focused panel instance's parameter, read from the dock layout at call time.
 *
 * Read from the STORE rather than parsed out of the instance id: the id's shape
 * (`"<type>:<params>"`) is `openPanel`'s business, and a second place that knows
 * it is a second place to get it wrong the day a parameter contains a colon.
 */
export function activeUndoParams(): string | null {
  const layout = useDockLayout.getState();
  const focused = layout.focusedPanel;
  if (!focused) return null;
  return layout.layout.panels[focused]?.params ?? null;
}

/**
 * Route undo to the focused panel's scope; returns false when nothing claimed
 * it and the caller should fall back to the scene.
 */
export function dispatchUndo(): boolean {
  const scope = activeUndoScope();
  if (!scope) return false;
  void scope.undo(activeUndoParams());
  return true;
}

/** The [`dispatchUndo`] twin. */
export function dispatchRedo(): boolean {
  const scope = activeUndoScope();
  if (!scope) return false;
  void scope.redo(activeUndoParams());
  return true;
}

/** Test-only: reset global registry state between cases. */
export function __resetUndoScopesForTest(): void {
  scopes.clear();
}
