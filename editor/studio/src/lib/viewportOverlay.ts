/**
 * Viewport-overlay refcount. The native engine viewport draws OVER the
 * webview (airspace rule), so any HTML surface that can cross the hole —
 * menu dropdowns, the command palette, dialogs, the panel-drag ghost —
 * must hide the native window for its lifetime. Overlays acquire on open
 * and release on close; a viewport hides while its count is > 0.
 *
 * **Two kinds of acquisition** (P23.2a audit). Every overlay the shell has
 * today is *window-wide*: a menu, the palette, a modal dialog and the drag
 * ghost are drawn over the whole workspace and can cross ANY viewport's hole.
 * So the default — [`acquireViewportOverlay`] / [`useViewportOverlay`] — hides
 * **every attached viewport**, and [`acquireViewportOverlayFor`] exists for the
 * panel-local case (a popover inside one editor) that does not exist yet.
 *
 * That distinction is why this is not simply a per-id counter. `Target::All`
 * has existed on the Rust side since the keyed-viewport refactor, but the
 * frontend primitive could only ever name ONE viewport — so the moment a second
 * one existed, every menu and dialog in the shell would have been painted over
 * by it while the scene viewport politely hid. Latent today (nothing creates a
 * non-primary viewport until P23.2b) and closed before it can bite.
 *
 * A viewport that attaches **while** a window-wide overlay is open comes up
 * hidden ([`registerViewport`]), which is the other half of the same bug:
 * opening a second viewport with the command palette up must not punch a hole
 * through it.
 *
 * Phase 2 (production rect-sync, P2.1) explores flash-free alternatives:
 * SetWindowRgn cutouts or freezing the last rendered frame into an <img>.
 */
import { useEffect } from "react";
import { viewport } from "./ipc";
import { PRIMARY_VIEWPORT } from "./viewportIds";

/** Window-wide acquisitions in flight. */
let allCount = 0;
/** Per-viewport acquisitions in flight, by id. */
const scoped = new Map<string, number>();
/** Viewports that exist. Seeded with the scene viewport, which the shell always
 *  mounts — so behaviour with no `registerViewport` call is exactly as before. */
const attached = new Set<string>([PRIMARY_VIEWPORT]);
/** Last visibility pushed per id, so an unchanged state costs no IPC. */
const shown = new Map<string, boolean>();

function hidden(id: string): boolean {
  return allCount > 0 || (scoped.get(id) ?? 0) > 0;
}

/** Push `id`'s visibility if it changed. */
function sync(id: string): void {
  const visible = !hidden(id);
  if (shown.get(id) === visible) return;
  shown.set(id, visible);
  viewport.setVisible(visible, id).catch(() => {
    // No native viewport in this window (detached panels) — nothing to hide.
  });
}

/** Every id that could need a push: attached viewports, plus any with a scoped
 *  count (a caller may address a viewport it owns before/after registration). */
function syncAll(): void {
  for (const id of new Set([...attached, ...scoped.keys()])) sync(id);
}

/**
 * Hide **every attached viewport** for the lifetime of a window-wide overlay;
 * returns a release-once function.
 */
export function acquireViewportOverlay(): () => void {
  allCount += 1;
  if (allCount === 1) syncAll();
  let released = false;
  return () => {
    if (released) return;
    released = true;
    allCount = Math.max(0, allCount - 1);
    if (allCount === 0) syncAll();
  };
}

/**
 * Hide ONE viewport — for an overlay confined to a single panel. The release
 * closes over the id, so a caller cannot release a viewport it did not acquire.
 */
export function acquireViewportOverlayFor(id: string): () => void {
  scoped.set(id, (scoped.get(id) ?? 0) + 1);
  sync(id);
  let released = false;
  return () => {
    if (released) return;
    released = true;
    const left = (scoped.get(id) ?? 1) - 1;
    if (left <= 0) scoped.delete(id);
    else scoped.set(id, left);
    sync(id);
  };
}

/**
 * Announce that a native viewport exists (called by `ViewportPanel` on attach);
 * returns a disposer for unmount.
 *
 * Registering while a window-wide overlay is open hides the new viewport
 * immediately — otherwise it would come up drawing over an open palette.
 */
export function registerViewport(id: string): () => void {
  attached.add(id);
  sync(id);
  return () => {
    attached.delete(id);
    scoped.delete(id);
    shown.delete(id);
  };
}

/** Hook form: holds a window-wide overlay acquisition while `active` is true. */
export function useViewportOverlay(active: boolean): void {
  useEffect(() => {
    if (!active) return;
    return acquireViewportOverlay();
  }, [active]);
}

/** Test-only: the scoped count for one viewport. */
export function __overlayCountForTest(id: string = PRIMARY_VIEWPORT): number {
  return scoped.get(id) ?? 0;
}

/** Test-only: window-wide acquisitions in flight. */
export function __overlayAllCountForTest(): number {
  return allCount;
}

/** Test-only: reset module state between cases. */
export function __resetViewportOverlayForTest(): void {
  allCount = 0;
  scoped.clear();
  shown.clear();
  attached.clear();
  attached.add(PRIMARY_VIEWPORT);
}
