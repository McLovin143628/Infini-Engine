import { useEffect } from "react";
import { listen } from "@tauri-apps/api/event";
import { PhysicalPosition, PhysicalSize } from "@tauri-apps/api/dpi";
import {
  cursorPosition,
  currentMonitor,
  getCurrentWindow,
  monitorFromPoint,
  primaryMonitor,
} from "@tauri-apps/api/window";
import { getAllWebviewWindows, WebviewWindow } from "@tauri-apps/api/webviewWindow";
import { useShallow } from "zustand/react/shallow";
import { panelTitle } from "../panelRegistry";
import { useDockDrag } from "../dock/dockDragStore";
import { useDockLayout } from "../dock/dockLayoutStore";
import { captureDropZones, resolveDropTarget, type DropZones } from "../dock/dropTargets";
import { setPanelWindowCount, startStoreBridgeHost } from "./storeBridge";

/**
 * Main-window manager for detached panel windows (ported from GeoCanvas,
 * ROADMAP P1.2.4). Declarative: the dock layout is the source of truth —
 * every panel whose `location.kind === "window"` gets a real
 * `WebviewWindow` (label `panel:<panelId>`, same bundle at
 * `#/panel/<panelId>`), and windows whose panel left that state are
 * destroyed. User-initiated closes come back as `panel://closed` events →
 * the panel returns to hidden (restorable via its Window-menu toggle).
 * Geometry reports persist into the layout's `windowRect`.
 *
 * Drag-to-dock-back: a detached window's titlebar drag emits
 * `panel://drag/*` events with SCREEN coordinates; this manager converts
 * them into main-viewport coordinates, drives the same drop-zone previews
 * as an in-window drag (via dockDragStore), and commits the drop —
 * destroying the window through the declarative sync.
 */

const spawned = new Map<string, WebviewWindow>();

interface PhysRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Clamp a desired PHYSICAL window rect so the WHOLE window fits inside a
 * monitor's work area: cap the size to the work area first, then pull the
 * top-left in so no edge spills past the screen.
 */
function clampWindowRect(desired: PhysRect, wa: PhysRect): PhysRect {
  const w = Math.max(1, Math.min(desired.w, wa.w));
  const h = Math.max(1, Math.min(desired.h, wa.h));
  const x = Math.round(Math.min(Math.max(desired.x, wa.x), wa.x + wa.w - w));
  const y = Math.round(Math.min(Math.max(desired.y, wa.y), wa.y + wa.h - h));
  return { x, y, w: Math.round(w), h: Math.round(h) };
}

export function panelWindowLabel(panelId: string): string {
  // Window labels allow [a-zA-Z0-9-/:_]; panel ids are type[:param] which
  // fits. Defensive replace for anything exotic.
  return `panel:${panelId.replace(/[^a-zA-Z0-9\-/:_]/g, "_")}`;
}

async function spawnPanelWindow(panelId: string): Promise<void> {
  const layout = useDockLayout.getState().layout;
  const panel = layout.panels[panelId];
  if (!panel) return;
  const label = panelWindowLabel(panelId);
  if (spawned.has(label)) return;

  const title = panelTitle(panelId, panel.params);
  const size = panel.windowRect
    ? { w: panel.windowRect.w, h: panel.windowRect.h }
    : { w: panel.floatRect.w + 16, h: panel.floatRect.h + 48 };

  const win = new WebviewWindow(label, {
    url: `index.html#/panel/${encodeURIComponent(panelId)}`,
    title,
    width: size.w,
    height: size.h,
    minWidth: 260,
    minHeight: 180,
    decorations: false,
    shadow: true,
    // Shown by the window itself once its React root mounted (no white
    // flash); positioned below before that.
    visible: false,
  });
  spawned.set(label, win);
  setPanelWindowCount(spawned.size);

  void win.once("tauri://created", async () => {
    try {
      // Resolve the desired PHYSICAL top-left + size, then clamp both to
      // the work area of the monitor the window lands on. `windowRect` is
      // already physical (from `outerSize`/`outerPosition`); a first detach
      // uses the cursor and a scale-converted `floatRect` (logical CSS px).
      let anchorX: number;
      let anchorY: number;
      let physW: number;
      let physH: number;
      if (panel.windowRect) {
        anchorX = panel.windowRect.x;
        anchorY = panel.windowRect.y;
        physW = panel.windowRect.w;
        physH = panel.windowRect.h;
      } else {
        const cur = await cursorPosition();
        anchorX = Math.round(cur.x - 60);
        anchorY = Math.round(cur.y - 16);
        physW = panel.floatRect.w + 16;
        physH = panel.floatRect.h + 48;
      }

      const monitor =
        (await monitorFromPoint(anchorX, anchorY)) ??
        (await currentMonitor()) ??
        (await primaryMonitor());

      if (monitor && !panel.windowRect) {
        // floatRect is logical CSS px — convert to physical for this monitor.
        physW = Math.round(physW * monitor.scaleFactor);
        physH = Math.round(physH * monitor.scaleFactor);
      }

      let rect: PhysRect = { x: anchorX, y: anchorY, w: physW, h: physH };
      if (monitor) {
        rect = clampWindowRect(rect, {
          x: monitor.workArea.position.x,
          y: monitor.workArea.position.y,
          w: monitor.workArea.size.width,
          h: monitor.workArea.size.height,
        });
      }

      await win.setPosition(new PhysicalPosition(rect.x, rect.y));
      await win.setSize(new PhysicalSize(rect.w, rect.h));
    } catch (e) {
      console.warn("[panelWindows] position failed:", e);
    }
  });
  void win.once("tauri://error", (e) => {
    console.error(`[panelWindows] window ${label} failed:`, e);
    spawned.delete(label);
    setPanelWindowCount(spawned.size);
    // Degrade: the panel comes back as a floating card.
    const st = useDockLayout.getState();
    const p = st.layout.panels[panelId];
    if (p?.location.kind === "window") {
      st.applyDrop(panelId, { kind: "float", rect: p.floatRect });
    }
  });
}

async function destroyPanelWindow(label: string): Promise<void> {
  const win = spawned.get(label);
  spawned.delete(label);
  setPanelWindowCount(spawned.size);
  try {
    await (win ?? (await WebviewWindow.getByLabel(label)))?.destroy();
  } catch {
    /* already gone */
  }
}

/** Bring a detached panel's OS window to the foreground. */
export function focusPanelWindow(panelId: string): void {
  const win = spawned.get(panelWindowLabel(panelId));
  if (!win) return;
  void win.unminimize().catch(() => {});
  void win.setFocus().catch(() => {});
}

/** Destroy every `panel:*` window (app quit / layout reset). */
export async function destroyAllPanelWindows(): Promise<void> {
  const all = await getAllWebviewWindows();
  await Promise.all(
    all.filter((w) => w.label.startsWith("panel:")).map((w) => w.destroy().catch(() => {})),
  );
  spawned.clear();
  setPanelWindowCount(0);
}

// =============================================================================
// The manager hook (mounted once by DockWorkspace, main window only)
// =============================================================================

export function usePanelWindowManager(): void {
  const hydrated = useDockLayout((s) => s.hydrated);
  const windowPanelIds = useDockLayout(
    useShallow((s) =>
      Object.entries(s.layout.panels)
        .filter(([, p]) => p.location.kind === "window")
        .map(([id]) => id)
        .sort(),
    ),
  );

  // Host side of the store bridge, once.
  useEffect(() => startStoreBridgeHost(), []);

  // Declarative window ↔ layout sync.
  useEffect(() => {
    if (!hydrated) return;
    const want = new Set(windowPanelIds.map(panelWindowLabel));
    for (const id of windowPanelIds) {
      const label = panelWindowLabel(id);
      if (!spawned.has(label)) void spawnPanelWindow(id);
    }
    for (const label of [...spawned.keys()]) {
      if (!want.has(label)) void destroyPanelWindow(label);
    }
  }, [hydrated, windowPanelIds]);

  // Startup reconciler: orphaned panel windows from a crashed session.
  useEffect(() => {
    if (!hydrated) return;
    void (async () => {
      const all = await getAllWebviewWindows();
      const layout = useDockLayout.getState().layout;
      for (const w of all) {
        if (!w.label.startsWith("panel:")) continue;
        if (spawned.has(w.label)) continue;
        const panelId = w.label.slice("panel:".length);
        const p = layout.panels[panelId];
        if (p?.location.kind === "window") {
          spawned.set(w.label, w as WebviewWindow);
          setPanelWindowCount(spawned.size);
        } else {
          void w.destroy().catch(() => {});
        }
      }
    })();
  }, [hydrated]);

  // Panel-window events. Same async-listen vs sync-cleanup race as the
  // store bridge host — `disposed` keeps StrictMode double-mounts safe.
  useEffect(() => {
    let disposed = false;
    const unlistens: Array<() => void> = [];
    const track = (p: Promise<() => void>) => {
      void p.then((u) => {
        if (disposed) u();
        else unlistens.push(u);
      });
    };
    let dragZones: DropZones | null = null;

    // A detached window reports its own pointer-downs so undo routing can aim
    // at it (P23.2a) — the dock layout store lives here and is not bridged.
    track(
      listen<{ panelId: string }>("panel://focus", (e) => {
        useDockLayout.getState().setFocusedPanel(e.payload.panelId);
      }),
    );

    track(
      listen<{ panelId: string }>("panel://closed", (e) => {
        const st = useDockLayout.getState();
        if (st.layout.panels[e.payload.panelId]?.location.kind === "window") {
          st.hidePanel(e.payload.panelId);
        }
      }),
    );

    track(
      listen<{ panelId: string; x: number; y: number; w: number; h: number }>(
        "panel://geometry",
        (e) => {
          useDockLayout.getState().setWindowRect(e.payload.panelId, {
            x: e.payload.x,
            y: e.payload.y,
            w: e.payload.w,
            h: e.payload.h,
          });
        },
      ),
    );

    // Detached-window titlebar drag → dock-back previews on the main
    // workspace. Screen px → main-viewport CSS px via the main window's
    // inner position + scale factor.
    const toViewport = async (sx: number, sy: number) => {
      const main = getCurrentWindow();
      const [pos, scale] = await Promise.all([main.innerPosition(), main.scaleFactor()]);
      return { x: (sx - pos.x) / scale, y: (sy - pos.y) / scale };
    };

    track(
      listen<{ panelId: string }>("panel://drag/start", () => {
        const el = useDockDrag.getState().workspaceEl;
        dragZones = el ? captureDropZones(el) : null;
      }),
    );

    track(
      listen<{ panelId: string; screenX: number; screenY: number }>(
        "panel://drag/move",
        (e) => {
          void (async () => {
            if (!dragZones) return;
            const { x, y } = await toViewport(e.payload.screenX, e.payload.screenY);
            const target = resolveDropTarget(dragZones, x, y, e.payload.panelId);
            const drag = useDockDrag.getState();
            if (!drag.dragging) {
              drag.start({
                panelId: e.payload.panelId,
                origin: "tab",
                ghost: { x: -1000, y: -1000 },
                grabOffset: { x: 0, y: 0 },
                pointer: { x, y },
                target,
                liveMove: true, // no in-window ghost — the OS window IS the ghost
              });
            } else {
              drag.update({ pointer: { x, y }, target });
            }
          })();
        },
      ),
    );

    track(
      listen<{ panelId: string; screenX: number; screenY: number }>(
        "panel://drag/drop",
        (e) => {
          void (async () => {
            const drag = useDockDrag.getState();
            const target = drag.dragging?.target?.target ?? null;
            drag.end();
            dragZones = null;
            if (target) {
              useDockLayout.getState().applyDrop(e.payload.panelId, target);
              try {
                await getCurrentWindow().setFocus();
              } catch {
                /* focus is a nicety */
              }
            }
          })();
        },
      ),
    );

    track(
      listen<{ panelId: string }>("panel://drag/cancel", () => {
        useDockDrag.getState().end();
        dragZones = null;
      }),
    );

    return () => {
      disposed = true;
      for (const u of unlistens) u();
      unlistens.length = 0;
    };
  }, []);
}
