import type { PointerEvent as ReactPointerEvent } from "react";
import { emitTo } from "@tauri-apps/api/event";
import { PhysicalPosition } from "@tauri-apps/api/dpi";
import { cursorPosition, getCurrentWindow } from "@tauri-apps/api/window";

/**
 * Titlebar drag for a DETACHED panel window: we own the move (rAF-paced
 * `setPosition` from the global cursor) instead of `startDragging()`,
 * because the native modal move loop would freeze JS for the whole drag —
 * no `panel://drag/move` events, no dock-back previews on the main window.
 * ~1 frame of trailing lag is the accepted trade (plus no Windows Snap for
 * these windows).
 *
 * Event contract (screen/physical px, consumed by `usePanelWindowManager`):
 *   panel://drag/start  { panelId }
 *   panel://drag/move   { panelId, screenX, screenY }
 *   panel://drag/drop   { panelId, screenX, screenY }   (may dock the panel)
 *   panel://drag/cancel { panelId }                      (Escape)
 */
export function beginDetachedWindowDrag(
  panelId: string,
  e: ReactPointerEvent<HTMLElement>,
): void {
  if (e.button !== 0) return;
  e.preventDefault();
  e.stopPropagation();

  const el = e.currentTarget;
  try {
    el.setPointerCapture(e.pointerId);
  } catch {
    /* gesture still works while over the element */
  }

  const win = getCurrentWindow();
  let grab: { dx: number; dy: number } | null = null;
  let raf = 0;
  let live = true;
  let moving = false;
  let started = false;

  // Establish the grab offset between cursor and window origin once.
  void (async () => {
    try {
      const [cur, pos] = await Promise.all([cursorPosition(), win.outerPosition()]);
      grab = { dx: cur.x - pos.x, dy: cur.y - pos.y };
    } catch {
      grab = { dx: 40, dy: 12 };
    }
  })();

  const step = async () => {
    raf = 0;
    if (!live || !grab || moving) return;
    moving = true;
    try {
      const cur = await cursorPosition();
      await win.setPosition(
        new PhysicalPosition(Math.round(cur.x - grab.dx), Math.round(cur.y - grab.dy)),
      );
      void emitTo("main", "panel://drag/move", {
        panelId,
        screenX: cur.x,
        screenY: cur.y,
      });
    } catch {
      /* window mid-teardown */
    } finally {
      moving = false;
      if (live && !raf) raf = requestAnimationFrame(() => void step());
    }
  };

  const onMove = () => {
    if (!started) {
      started = true;
      void emitTo("main", "panel://drag/start", { panelId });
    }
    if (!raf && !moving) raf = requestAnimationFrame(() => void step());
  };

  const cleanup = (ev?: PointerEvent) => {
    live = false;
    el.removeEventListener("pointermove", onMove);
    el.removeEventListener("pointerup", finish);
    el.removeEventListener("pointercancel", onCancel);
    window.removeEventListener("keydown", onKey, true);
    if (raf) cancelAnimationFrame(raf);
    if (ev) {
      try {
        el.releasePointerCapture(ev.pointerId);
      } catch {
        /* already released */
      }
    }
  };

  const finish = (ev: PointerEvent) => {
    cleanup(ev);
    if (!started) return;
    void (async () => {
      try {
        const cur = await cursorPosition();
        void emitTo("main", "panel://drag/drop", {
          panelId,
          screenX: cur.x,
          screenY: cur.y,
        });
      } catch {
        void emitTo("main", "panel://drag/cancel", { panelId });
      }
    })();
  };

  const onCancel = (ev: PointerEvent) => {
    cleanup(ev);
    if (started) void emitTo("main", "panel://drag/cancel", { panelId });
  };

  const onKey = (ev: KeyboardEvent) => {
    if (ev.key === "Escape") {
      cleanup();
      if (started) void emitTo("main", "panel://drag/cancel", { panelId });
    }
  };

  el.addEventListener("pointermove", onMove);
  el.addEventListener("pointerup", finish);
  el.addEventListener("pointercancel", onCancel);
  window.addEventListener("keydown", onKey, true);
}
