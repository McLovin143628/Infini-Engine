/**
 * **The viewport takes focus back** (P23.2a audit — B1).
 *
 * The panel-focus undo routing had a hole big enough to lose an author's work
 * in. `focusedPanel` is set by a pointer-down on a docked panel, a float raise,
 * a tab activation, or a detached window's report — and **none of those can
 * ever name the viewport**:
 *
 * * `ViewportPanel` is mounted in the dock's CENTRE cell, outside any
 *   `DockGroup`, so the body-level `onPointerDownCapture` that focuses every
 *   other panel does not exist for it;
 * * and the hole itself is a native child window that draws over the webview
 *   and swallows DOM pointer events entirely, so a click on the 3D view is not
 *   a DOM event at all.
 *
 * The consequence was repeatable and silent: click the Material editor, click
 * into the viewport, press Ctrl+Z — the chord is forwarded natively
 * (`viewport://key`), replayed through `dispatchChord`, and routed to the
 * *material* scope, because `focusedPanel` never stopped saying "material". The
 * graph rewinds; the actor the author was looking at does not move.
 *
 * Two entry points close it, matching the two ways focus can land on the
 * viewport:
 *
 * 1. [`handleViewportChord`] — **a forwarded chord is proof the native viewport
 *    holds OS focus.** Win32 only forwards a chord it declined to consume, and
 *    it only receives keys at all when the child window has focus, so the
 *    arrival of one is a stronger focus signal than any DOM event.
 * 2. [`focusViewport`] — a pointer-down on the centre cell's margin (the
 *    toolbar strip and the padding around the hole), which IS a DOM event.
 *
 * `panelTypeOf("viewport")` has no registered undo scope, so routing falls
 * through to the scene default — which is what Ctrl+Z over the 3D view has
 * always meant.
 */
import { useDockLayout } from "../panels/dock/dockLayoutStore";
import { dispatchChord } from "./keybindings";

/**
 * The panel id the scene viewport is registered under (`registerPanels.tsx`).
 * `App.tsx` mounts it with this id, and it is what focus routing names.
 */
export const VIEWPORT_PANEL_ID = "viewport";

/** Make the viewport the focused panel. */
export function focusViewport(): void {
  useDockLayout.getState().setFocusedPanel(VIEWPORT_PANEL_ID);
}

/**
 * Handle a key chord forwarded from a native viewport: take focus, then
 * replay. Returns whether a binding fired.
 *
 * Order is the whole content of this function. Focus must move BEFORE the
 * dispatch, or the very first Ctrl+Z after clicking into the viewport still
 * lands on the previously focused panel — which is the bug, one keypress
 * narrower.
 */
export function handleViewportChord(chord: string): boolean {
  focusViewport();
  return dispatchChord(chord);
}
