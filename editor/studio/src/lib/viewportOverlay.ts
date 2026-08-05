/**
 * Viewport-overlay refcount. The native engine viewport draws OVER the
 * webview (airspace rule), so any HTML surface that can cross the hole —
 * menu dropdowns, the command palette, dialogs, the panel-drag ghost —
 * must hide the native window for its lifetime. Overlays acquire on open
 * and release on close; the viewport hides while the count is > 0.
 *
 * **Keyed per viewport** (P23.2a). It used to be one module-level counter,
 * which was the same implicit-identity assumption the backend's single-slot
 * `ViewportState` made: with a second viewport, one overlay's release would
 * have un-hidden a window it never hid, or left one hidden for ever. Each id
 * now has its own count, and every existing caller defaults to
 * `PRIMARY_VIEWPORT` — which is exactly today's behaviour, because today the
 * scene viewport is the only one.
 *
 * Phase 2 (production rect-sync, P2.1) explores flash-free alternatives:
 * SetWindowRgn cutouts or freezing the last rendered frame into an <img>.
 */
import { useEffect } from "react";
import { viewport } from "./ipc";
import { PRIMARY_VIEWPORT } from "./viewportIds";

const counts = new Map<string, number>();

function setVisible(id: string, visible: boolean): void {
  viewport.setVisible(visible, id).catch(() => {
    // No native viewport in this window (detached panels) — nothing to hide.
  });
}

/**
 * Increment the overlay count for `id`; returns a release-once function.
 *
 * The release closes over the id, so a caller cannot release the wrong
 * viewport's count by acquiring one and releasing another.
 */
export function acquireViewportOverlay(id: string = PRIMARY_VIEWPORT): () => void {
  const next = (counts.get(id) ?? 0) + 1;
  counts.set(id, next);
  if (next === 1) setVisible(id, false);
  let released = false;
  return () => {
    if (released) return;
    released = true;
    const left = (counts.get(id) ?? 1) - 1;
    if (left <= 0) {
      counts.delete(id);
      setVisible(id, true);
    } else {
      counts.set(id, left);
    }
  };
}

/** Hook form: holds an overlay acquisition while `active` is true. */
export function useViewportOverlay(active: boolean, id: string = PRIMARY_VIEWPORT): void {
  useEffect(() => {
    if (!active) return;
    return acquireViewportOverlay(id);
  }, [active, id]);
}

/** Test-only. */
export function __overlayCountForTest(id: string = PRIMARY_VIEWPORT): number {
  return counts.get(id) ?? 0;
}
