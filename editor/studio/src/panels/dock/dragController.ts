import type { PointerEvent as ReactPointerEvent } from "react";
import { panelDefFor } from "../panelRegistry";
import { clampPanelRect, type PanelRect } from "../panelRect";
import { useDockDrag } from "./dockDragStore";
import { useDockLayout } from "./dockLayoutStore";
import { captureDropZones, resolveDropTarget, type DropZones } from "./dropTargets";

/** Pointer travel below which a press stays a click. */
const MOVE_THRESHOLD_PX = 4;
/** Ghost card footprint (matches DragLayer's chip). */
export const GHOST_W = 180;
export const GHOST_H = 34;

/**
 * The unified panel-drag controller (ported from GeoCanvas). ONE code path
 * serves every origin — a dock tab, a floating panel header, and a
 * detached window's titlebar — because the dragged panel's location NEVER
 * changes mid-gesture: pointer capture stays on the originating element
 * (which stays mounted), a portal ghost follows the cursor, and the single
 * `applyDrop` transaction runs on release.
 *
 * Feel details:
 *  - A floating panel with NO active dock target live-moves for real; the
 *    ghost takes over the moment a dock preview activates. Safe:
 *    floating→floating never remounts.
 *  - A dock tab shows the ghost as soon as the threshold is passed.
 *  - Escape cancels the whole gesture (float position restored).
 *  - Moves are rAF-latched; drop zones are measured once at drag start.
 */
export function beginPanelDrag(
  panelId: string,
  e: ReactPointerEvent<HTMLElement>,
  origin: "tab" | "float",
): void {
  if (e.button !== 0) return;
  if ((e.target as HTMLElement).closest("button") != null && origin === "float") {
    return; // header buttons keep their clicks
  }
  const layoutStore = useDockLayout.getState();
  const dragStore = useDockDrag.getState();
  const panel = layoutStore.layout.panels[panelId];
  const workspaceEl = dragStore.workspaceEl;
  if (!panel || !workspaceEl) return;

  e.stopPropagation();
  if (origin === "float") e.preventDefault();

  const el = e.currentTarget;
  try {
    el.setPointerCapture(e.pointerId);
  } catch {
    /* capture unavailable — gesture still works while over the element */
  }

  const startX = e.clientX;
  const startY = e.clientY;
  const startRect: PanelRect = { ...panel.floatRect };
  const wsRect = workspaceEl.getBoundingClientRect();
  let zones: DropZones | null = null;
  let started = false;
  let raf = 0;
  let lastEv: { x: number; y: number } | null = null;

  const ensureStarted = () => {
    if (started) return;
    started = true;
    zones = captureDropZones(workspaceEl);
    if (origin === "float") useDockLayout.getState().bringToFront(panelId);
    useDockDrag.getState().start({
      panelId,
      origin,
      ghost: { x: startX - GHOST_W / 2, y: startY - GHOST_H / 2 },
      grabOffset: { x: GHOST_W / 2, y: GHOST_H / 2 },
      pointer: { x: startX, y: startY },
      target: null,
      liveMove: origin === "float",
    });
  };

  const canDetach = panelDefFor(panelId)?.canDetach !== false;

  const applyMove = () => {
    raf = 0;
    const ev = lastEv;
    if (!ev || !zones) return;
    let target = resolveDropTarget(zones, ev.x, ev.y, panelId);
    // Outside the viewport entirely (pointer capture keeps events flowing
    // past the window edge on Windows) → tear out into a real OS window.
    if (
      !target &&
      canDetach &&
      (ev.x < 0 || ev.y < 0 || ev.x > window.innerWidth || ev.y > window.innerHeight)
    ) {
      target = { target: { kind: "window" }, preview: { x: -1, y: -1, w: 0, h: 0 } };
    }
    const st = useDockLayout.getState();
    const dragging = useDockDrag.getState().dragging;
    if (!dragging) return;

    const liveMove = origin === "float" && target === null;
    if (liveMove) {
      // Real panel follows the cursor (workspace-relative coords).
      const c = st.container ?? { w: wsRect.width, h: wsRect.height };
      st.setFloatRect(
        panelId,
        clampPanelRect(
          {
            ...startRect,
            x: startRect.x + (ev.x - startX),
            y: startRect.y + (ev.y - startY),
          },
          c,
        ),
      );
    }
    useDockDrag.getState().update({
      ghost: { x: ev.x - GHOST_W / 2, y: ev.y - GHOST_H / 2 },
      pointer: { x: ev.x, y: ev.y },
      target,
      liveMove,
    });
  };

  const onMove = (ev: PointerEvent) => {
    if (
      !started &&
      Math.abs(ev.clientX - startX) < MOVE_THRESHOLD_PX &&
      Math.abs(ev.clientY - startY) < MOVE_THRESHOLD_PX
    ) {
      return;
    }
    ensureStarted();
    lastEv = { x: ev.clientX, y: ev.clientY };
    if (!raf) raf = requestAnimationFrame(applyMove);
  };

  const cleanup = (ev?: PointerEvent) => {
    el.removeEventListener("pointermove", onMove);
    el.removeEventListener("pointerup", finish);
    el.removeEventListener("pointercancel", cancel);
    window.removeEventListener("keydown", onKey, true);
    if (raf) {
      cancelAnimationFrame(raf);
      raf = 0;
    }
    if (ev) {
      try {
        el.releasePointerCapture(ev.pointerId);
      } catch {
        /* already released */
      }
    }
    useDockDrag.getState().end();
  };

  const finish = (ev: PointerEvent) => {
    const dragging = useDockDrag.getState().dragging;
    cleanup(ev);
    if (!started || !dragging) return;
    const st = useDockLayout.getState();
    const target = dragging.target?.target ?? null;
    if (target) {
      st.applyDrop(panelId, target);
      return;
    }
    if (origin === "float") {
      // Plain float move — persist the release position.
      st.commitFloat(panelId);
      return;
    }
    // Tab torn out with no dock target → float where it was dropped.
    const c = st.container ?? { w: wsRect.width, h: wsRect.height };
    const rect = clampPanelRect(
      {
        x: ev.clientX - wsRect.left - startRect.w / 2,
        y: ev.clientY - wsRect.top - 18,
        w: startRect.w,
        h: startRect.h,
      },
      c,
    );
    st.applyDrop(panelId, { kind: "float", rect });
  };

  const cancel = (ev?: PointerEvent) => {
    const wasStarted = started;
    cleanup(ev);
    if (wasStarted && origin === "float") {
      // Restore the pre-drag position.
      useDockLayout.getState().setFloatRect(panelId, startRect);
    }
  };

  const onKey = (ev: KeyboardEvent) => {
    if (ev.key === "Escape") {
      ev.stopPropagation();
      cancel();
    }
  };

  el.addEventListener("pointermove", onMove);
  el.addEventListener("pointerup", finish);
  el.addEventListener("pointercancel", cancel);
  window.addEventListener("keydown", onKey, true);
}
