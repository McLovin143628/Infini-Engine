/**
 * Viewport-overlay refcount. The native engine viewport draws OVER the
 * webview (airspace rule), so any HTML surface that can cross the hole —
 * menu dropdowns, the command palette, dialogs, the panel-drag ghost —
 * must hide the native window for its lifetime. Overlays acquire on open
 * and release on close; the viewport hides while the count is > 0.
 *
 * Phase 2 (production rect-sync, P2.1) explores flash-free alternatives:
 * SetWindowRgn cutouts or freezing the last rendered frame into an <img>.
 */
import { useEffect } from "react";
import { viewport } from "./ipc";

let count = 0;

function setVisible(visible: boolean): void {
  viewport.setVisible(visible).catch(() => {
    // No native viewport in this window (detached panels) — nothing to hide.
  });
}

/** Increment the overlay count; returns a release-once function. */
export function acquireViewportOverlay(): () => void {
  count += 1;
  if (count === 1) setVisible(false);
  let released = false;
  return () => {
    if (released) return;
    released = true;
    count -= 1;
    if (count === 0) setVisible(true);
  };
}

/** Hook form: holds an overlay acquisition while `active` is true. */
export function useViewportOverlay(active: boolean): void {
  useEffect(() => {
    if (!active) return;
    return acquireViewportOverlay();
  }, [active]);
}

/** Test-only. */
export function __overlayCountForTest(): number {
  return count;
}
