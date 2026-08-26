/**
 * Viewport-overlay refcount. The native engine viewport draws OVER the
 * webview (airspace rule), so any HTML surface that can cross the hole —
 * menu dropdowns, the command palette, dialogs, the panel-drag ghost —
 * must get the native window out of its way for its lifetime. Overlays acquire
 * on open and release on close.
 *
 * **Two ways out of the way** (UX2). An overlay that can MEASURE ITSELF supplies
 * its rectangles and the native child is *cut out* around them
 * (`viewport_set_region` → `SetWindowRgn`): the webview shows through exactly
 * where the menu is and the 3D view keeps rendering everywhere else. An overlay
 * that cannot — the command palette, the modal dialogs, the drag ghost, all of
 * which are drawn over the whole workspace — still hides it outright, which is
 * what every overlay did from Phase 1 until this wave.
 *
 * The rule is deliberately pessimistic: a viewport is cut out only when EVERY
 * hold on it carries rects. One rect-less hold anywhere and the whole thing
 * hides, because a surface that could not say where it is could be anywhere.
 * The same fallback covers the platforms with no cutout backend — Windows
 * carves, macOS's layer-mask twin is not built and Linux has no embedding at
 * all, so [`viewport.cutoutSupported`] is asked once and the answer is `false`
 * until it is known.
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
 */
import { useEffect, useLayoutEffect, useRef, type RefObject } from "react";
import { viewport } from "./ipc";
import { toPhysicalRect } from "./viewportRect";
import { PRIMARY_VIEWPORT } from "./viewportIds";

/** A CSS-pixel rectangle relative to the window — what an overlay covers. */
export interface OverlayRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** One live acquisition. `rects === null` means "I cannot say where I am". */
interface Hold {
  rects: OverlayRect[] | null;
}

/**
 * What [`acquireViewportOverlay`] returns: the release function it always was,
 * with a way to keep a measured overlay's rectangles up to date as it moves,
 * grows a submenu, or the window resizes under it.
 */
export type OverlayRelease = (() => void) & {
  /** Replace this hold's cutout rects; `null` falls back to the full hide. */
  setRects(rects: OverlayRect[] | null): void;
};

/** Window-wide acquisitions in flight. */
const wide = new Set<Hold>();
/** Per-viewport acquisitions in flight, by id. */
const scoped = new Map<string, number>();
/** Viewports that exist. Seeded with the scene viewport, which the shell always
 *  mounts — so behaviour with no `registerViewport` call is exactly as before. */
const attached = new Set<string>([PRIMARY_VIEWPORT]);
/** Last visibility pushed per id, so an unchanged state costs no IPC. */
const shown = new Map<string, boolean>();
/** Last cutout set pushed per id, same purpose (a menu redraws constantly). */
const carved = new Map<string, string>();

/** Whether the native backend can cut a hole; see the module docs. */
let cutoutSupported = false;
let probeStarted = false;

/**
 * Ask the backend once whether cutouts exist on this platform (UX2).
 *
 * Until the answer arrives the guard behaves exactly as it did before this
 * wave, which is why the probe needs no await anywhere: the fallback is the old
 * behaviour, not a broken one. It is kicked off both when a viewport registers
 * (app start, long before any menu) and on the first acquisition, so the answer
 * is in hand by the time a user opens anything.
 */
function probeCutoutSupport(): void {
  if (probeStarted) return;
  probeStarted = true;
  viewport
    .cutoutSupported()
    .then((supported) => {
      const yes = supported === true;
      if (yes === cutoutSupported) return;
      cutoutSupported = yes;
      syncAll();
    })
    .catch(() => {
      // No backend in this window (a detached panel) — the full hide stands.
    });
}

/** What one viewport should be doing right now. */
interface OverlayState {
  visible: boolean;
  cutouts: OverlayRect[];
}

function stateFor(id: string): OverlayState {
  const scopedCount = scoped.get(id) ?? 0;
  if (wide.size === 0 && scopedCount === 0) return { visible: true, cutouts: [] };
  // A panel-local hold is the popover case that does not exist yet and has no
  // rect to offer; and with no cutout backend there is nothing to offer it to.
  if (scopedCount > 0 || !cutoutSupported) return { visible: false, cutouts: [] };
  const cutouts: OverlayRect[] = [];
  for (const hold of wide) {
    // ONE unmeasured overlay hides everything: it could be anywhere, and
    // guessing is how a menu ends up drawn under the 3D view.
    if (!hold.rects || hold.rects.length === 0) return { visible: false, cutouts: [] };
    cutouts.push(...hold.rects);
  }
  return { visible: true, cutouts };
}

/**
 * A cutout set's identity, so an unchanged region costs no IPC.
 *
 * **The scale is part of the identity** (UX2 audit). What crosses is PHYSICAL
 * pixels; what is compared here would otherwise be CSS ones. Drag the window to
 * a monitor at a different scale with a menu open and the CSS rectangle is
 * unchanged, so the dedupe swallows the push — while the hole itself moved,
 * because `viewport_set_rect` DID re-push (`ViewportPanel` watches the scale
 * explicitly for exactly this case) and the native side re-applied the same
 * stale physical rectangle against the new one.
 */
function regionKey(cutouts: OverlayRect[]): string {
  const dpr = window.devicePixelRatio || 1;
  return `${dpr}|${cutouts.map((r) => `${r.x},${r.y},${r.width},${r.height}`).join(";")}`;
}

/** The display-scale watch, armed once; see [`regionKey`]. */
let dprArmed = false;
let dprQuery: MediaQueryList | null = null;

function onDprChange(): void {
  // Re-arm for the NEW ratio (a media query matches one value), then re-push:
  // the key above has changed even though nothing was re-measured.
  armDprQuery();
  syncAll();
}

function armDprQuery(): void {
  dprQuery?.removeEventListener("change", onDprChange);
  dprQuery = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
  dprQuery.addEventListener("change", onDprChange);
}

/**
 * Notice a display-scale change that arrives with no resize and no re-measure —
 * a cross-monitor drag with a menu open (UX2 audit). Same mechanism, and the
 * same reason, as `ViewportPanel`'s: the CSS layout is identical, so nothing
 * else fires.
 */
function watchDevicePixelRatio(): void {
  if (dprArmed) return;
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
  dprArmed = true;
  armDprQuery();
}

function pushVisible(id: string, visible: boolean): void {
  viewport.setVisible(visible, id).catch(() => {
    // No native viewport in this window (detached panels) — nothing to hide.
  });
}

function pushRegion(id: string, cutouts: OverlayRect[]): void {
  const dpr = window.devicePixelRatio || 1;
  viewport.setRegion(cutouts.map((r) => toPhysicalRect(r, dpr)), id).catch(() => {
    // Same: a window with no native child has nothing to carve.
  });
}

/**
 * Push `id`'s state if it changed.
 *
 * **The order is load-bearing.** Coming back into view, the region goes first so
 * the child never appears whole for a frame under an open menu; going away, the
 * visibility goes first so the child is gone before its region is released.
 * Either way round the wrong way is a one-frame flash of exactly the artefact
 * the mechanism exists to remove.
 */
function sync(id: string): void {
  const { visible, cutouts } = stateFor(id);
  const key = regionKey(cutouts);
  const visibleChanged = shown.get(id) !== visible;
  const regionChanged = (carved.get(id) ?? "") !== key;
  if (!visibleChanged && !regionChanged) return;
  shown.set(id, visible);
  carved.set(id, key);
  if (visible) {
    if (regionChanged) pushRegion(id, cutouts);
    if (visibleChanged) pushVisible(id, true);
  } else {
    if (visibleChanged) pushVisible(id, false);
    if (regionChanged) pushRegion(id, cutouts);
  }
}

/** Every id that could need a push: attached viewports, plus any with a scoped
 *  count (a caller may address a viewport it owns before/after registration). */
function syncAll(): void {
  for (const id of new Set([...attached, ...scoped.keys()])) sync(id);
}

/**
 * Get **every attached viewport** out of the way for the lifetime of a
 * window-wide overlay; returns a release-once function.
 *
 * Pass `rects` (CSS pixels, window-relative) to be cut out of the viewport
 * rather than hiding it — see the module docs. Passing nothing is the old
 * behaviour and stays the right answer for a surface that covers the workspace.
 */
export function acquireViewportOverlay(rects: OverlayRect[] | null = null): OverlayRelease {
  probeCutoutSupport();
  watchDevicePixelRatio();
  const hold: Hold = { rects };
  wide.add(hold);
  syncAll();
  let released = false;
  const release = (() => {
    if (released) return;
    released = true;
    wide.delete(hold);
    syncAll();
  }) as OverlayRelease;
  release.setRects = (next: OverlayRect[] | null) => {
    // After release this hold is not in the set; writing to it would be a quiet
    // no-op that reads like it worked.
    if (released) return;
    hold.rects = next;
    syncAll();
  };
  return release;
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
  probeCutoutSupport();
  watchDevicePixelRatio();
  attached.add(id);
  sync(id);
  return () => {
    attached.delete(id);
    scoped.delete(id);
    shown.delete(id);
    carved.delete(id);
  };
}

/** Hook form: holds a window-wide overlay acquisition while `active` is true. */
export function useViewportOverlay(active: boolean): void {
  useEffect(() => {
    if (!active) return;
    return acquireViewportOverlay();
  }, [active]);
}

/**
 * The mark on every element a measured overlay wants cut out — the menu panel
 * itself, and separately any panel that flies out of it (a menu-bar dropdown's
 * submenu is absolutely positioned OUTSIDE its parent's box, so its parent's
 * rectangle does not contain it).
 *
 * Nothing is measured implicitly. An overlay that marks nothing measures as
 * `null` and hides the viewport outright, which is the pre-UX2 behaviour and
 * therefore the right thing to fall back to: the alternative — assuming some
 * container is the overlay — punches the hole in the wrong place and leaves the
 * menu drawn under the 3D view, which is worse than the blackout.
 */
export const CUTOUT_ATTR = "data-viewport-cutout";

/**
 * The rectangles the marked elements at or under `root` cover, or `null` if
 * there is nothing measurable — which the guard reads as "hide it all".
 *
 * Overlapping rectangles are fine and are not merged: the native side subtracts
 * them from the child's region one after another, and subtracting the same
 * pixels twice is the same region.
 */
export function measureCutout(root: HTMLElement | null): OverlayRect[] | null {
  if (!root) return null;
  const marked = root.matches(`[${CUTOUT_ATTR}]`) ? [root] : [];
  const rects: OverlayRect[] = [];
  for (const el of [...marked, ...root.querySelectorAll<HTMLElement>(`[${CUTOUT_ATTR}]`)]) {
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) {
      rects.push({ x: r.left, y: r.top, width: r.width, height: r.height });
    }
  }
  return rects.length > 0 ? rects : null;
}

/**
 * Hook form for a **measured** overlay (UX2): holds the acquisition while
 * `active`, supplying `root`'s rectangles so the 3D view keeps rendering
 * around it.
 *
 * The acquisition is taken in a LAYOUT effect with the rect already measured,
 * so the native side never receives a hide it then has to undo — a hide
 * followed a millisecond later by a show is the blackout flash this wave
 * removes, just shorter.
 *
 * Re-measuring is deliberately generous, because it is three `getBoundingClientRect`
 * calls behind an IPC dedupe and being one frame stale is a visibly wrong hole:
 * every render, on any resize of the root, and on any DOM change under it (a
 * fly-out submenu is a child component's state, so it re-renders the menu and
 * not its host).
 */
export function useViewportCutout(active: boolean, root: RefObject<HTMLElement | null>): void {
  const hold = useRef<OverlayRelease | null>(null);

  useLayoutEffect(() => {
    if (!active) return;
    const held = acquireViewportOverlay(measureCutout(root.current));
    hold.current = held;
    return () => {
      held();
      hold.current = null;
    };
  }, [active, root]);

  // No dependency array on purpose: a menu that re-rendered may have moved.
  useLayoutEffect(() => {
    hold.current?.setRects(measureCutout(root.current));
  });

  useEffect(() => {
    if (!active) return;
    const el = root.current;
    if (!el) return;
    const remeasure = () => hold.current?.setRects(measureCutout(root.current));
    const mo = new MutationObserver(remeasure);
    mo.observe(el, { childList: true, subtree: true, attributes: true });
    // jsdom has no ResizeObserver; the mutation half still runs there.
    const ro = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(remeasure);
    ro?.observe(el);
    return () => {
      mo.disconnect();
      ro?.disconnect();
    };
  }, [active, root]);
}

/** Test-only: the scoped count for one viewport. */
export function __overlayCountForTest(id: string = PRIMARY_VIEWPORT): number {
  return scoped.get(id) ?? 0;
}

/** Test-only: window-wide acquisitions in flight. */
export function __overlayAllCountForTest(): number {
  return wide.size;
}

/** Test-only: pretend the platform does (or does not) support cutouts. */
export function __setCutoutSupportedForTest(supported: boolean): void {
  probeStarted = true;
  cutoutSupported = supported;
  syncAll();
}

/** Test-only: reset module state between cases. */
export function __resetViewportOverlayForTest(): void {
  dprQuery?.removeEventListener("change", onDprChange);
  dprQuery = null;
  dprArmed = false;
  wide.clear();
  scoped.clear();
  shown.clear();
  carved.clear();
  attached.clear();
  attached.add(PRIMARY_VIEWPORT);
  cutoutSupported = false;
  probeStarted = false;
}
