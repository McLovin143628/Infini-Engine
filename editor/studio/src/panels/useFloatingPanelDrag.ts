import { useCallback, type PointerEvent as ReactPointerEvent } from "react";
import { beginPanelDrag } from "./dock/dragController";
import { useDockLayout } from "./dock/dockLayoutStore";
import {
  MIN_PANEL_H,
  MIN_PANEL_W,
  resizePanelRect,
  type HandleId,
  type PanelRect,
} from "./panelRect";

export interface FloatingPanelDragHandlers {
  /** Pointer-down on the panel header — starts a move gesture. */
  startHeaderDrag: (e: ReactPointerEvent<HTMLElement>) => void;
  /** Pointer-down on one of the 8 resize handles. */
  startResize: (handle: HandleId, e: ReactPointerEvent<HTMLElement>) => void;
}

/**
 * Gestures for one FLOATING panel. The MOVE path goes through the unified
 * drag controller (`beginPanelDrag`) — live-moving with no dock target,
 * ghost + dock previews once one activates, tear-in to any region on
 * release. RESIZE stays a local pointer-capture gesture on the 8 handles
 * (resizing never changes location, so it needs none of that machinery).
 */
export function useFloatingPanelDrag(id: string): FloatingPanelDragHandlers {
  const startResize = useCallback(
    (handle: HandleId, e: ReactPointerEvent<HTMLElement>) => {
      if (e.button !== 0) return;
      const store = useDockLayout.getState();
      const container = store.container;
      const panel = store.layout.panels[id];
      if (!container || !panel) return;

      store.bringToFront(id);
      e.stopPropagation();
      e.preventDefault();

      const el = e.currentTarget;
      try {
        el.setPointerCapture(e.pointerId);
      } catch {
        /* capture unavailable — gesture still works while over the element */
      }

      const startRect: PanelRect = { ...panel.floatRect };
      const startX = e.clientX;
      const startY = e.clientY;
      let moved = false;

      const onMove = (ev: PointerEvent) => {
        moved = true;
        const st = useDockLayout.getState();
        const c = st.container ?? container;
        st.setFloatRect(
          id,
          resizePanelRect(
            startRect,
            handle,
            ev.clientX - startX,
            ev.clientY - startY,
            { w: MIN_PANEL_W, h: MIN_PANEL_H },
            c,
          ),
        );
      };

      const finish = (ev: PointerEvent) => {
        el.removeEventListener("pointermove", onMove);
        el.removeEventListener("pointerup", finish);
        el.removeEventListener("pointercancel", finish);
        try {
          el.releasePointerCapture(ev.pointerId);
        } catch {
          /* pointer already released */
        }
        // Write-on-release: one persisted write per completed gesture.
        if (moved) useDockLayout.getState().commitFloat(id);
      };

      el.addEventListener("pointermove", onMove);
      el.addEventListener("pointerup", finish);
      el.addEventListener("pointercancel", finish);
    },
    [id],
  );

  const startHeaderDrag = useCallback(
    (e: ReactPointerEvent<HTMLElement>) => {
      // Header buttons (chevron, X) keep their own clicks; the controller
      // applies its own 4px click threshold.
      if ((e.target as HTMLElement).closest("button") != null) return;
      beginPanelDrag(id, e, "float");
    },
    [id],
  );

  return { startHeaderDrag, startResize };
}
