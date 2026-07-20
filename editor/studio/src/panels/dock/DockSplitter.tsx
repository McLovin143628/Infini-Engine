import { useCallback, useRef, type PointerEvent as ReactPointerEvent } from "react";
import { cn } from "../../lib/utils";
import { isHorizontalSide, splitterCursor, splitterGrowSign } from "./dockGeometry";
import { useDockLayout } from "./dockLayoutStore";
import type { DockSide } from "./dockTypes";

/** Keyboard resize step (px); Shift = 4×. */
const KEY_STEP = 16;

/**
 * The region-edge splitter: a slim strip on a dock region's center-facing
 * edge that resizes the region's dedicated `size` field. Pointer-capture +
 * rAF-latched moves; one persist on release (`commitRegion`). Keyboard
 * accessible (`role="separator"`, arrow keys).
 */
export function RegionSplitter({ side }: { side: DockSide }) {
  const size = useDockLayout((s) => s.layout.docks[side].size);
  const horizontal = isHorizontalSide(side);
  const raf = useRef(0);

  const onPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();
      const el = e.currentTarget;
      try {
        el.setPointerCapture(e.pointerId);
      } catch {
        /* gesture still works while over the element */
      }
      const store = useDockLayout.getState();
      const startSize = store.layout.docks[side].size;
      const start = horizontal ? e.clientX : e.clientY;
      const sign = splitterGrowSign(side);
      let pending: number | null = null;

      const onMove = (ev: PointerEvent) => {
        pending = startSize + sign * ((horizontal ? ev.clientX : ev.clientY) - start);
        if (raf.current) return;
        raf.current = requestAnimationFrame(() => {
          raf.current = 0;
          if (pending != null) {
            useDockLayout.getState().setRegionSize(side, pending);
          }
        });
      };
      const finish = (ev: PointerEvent) => {
        el.removeEventListener("pointermove", onMove);
        el.removeEventListener("pointerup", finish);
        el.removeEventListener("pointercancel", finish);
        if (raf.current) {
          cancelAnimationFrame(raf.current);
          raf.current = 0;
        }
        if (pending != null) {
          useDockLayout.getState().setRegionSize(side, pending);
          useDockLayout.getState().commitRegion();
        }
        try {
          el.releasePointerCapture(ev.pointerId);
        } catch {
          /* already released */
        }
      };
      el.addEventListener("pointermove", onMove);
      el.addEventListener("pointerup", finish);
      el.addEventListener("pointercancel", finish);
    },
    [side, horizontal],
  );

  const onKeyDown = (e: React.KeyboardEvent) => {
    const step = (e.shiftKey ? 4 : 1) * KEY_STEP;
    const sign = splitterGrowSign(side);
    let delta = 0;
    if (horizontal) {
      if (e.key === "ArrowRight") delta = step;
      else if (e.key === "ArrowLeft") delta = -step;
    } else {
      if (e.key === "ArrowDown") delta = step;
      else if (e.key === "ArrowUp") delta = -step;
    }
    if (!delta) return;
    e.preventDefault();
    const store = useDockLayout.getState();
    store.setRegionSize(side, store.layout.docks[side].size + sign * delta);
    store.commitRegion();
  };

  return (
    <div
      role="separator"
      aria-orientation={horizontal ? "vertical" : "horizontal"}
      aria-label={`Resize ${side} dock`}
      aria-valuenow={Math.round(size)}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onKeyDown={onKeyDown}
      style={{ cursor: splitterCursor(side), touchAction: "none" }}
      className={cn(
        "absolute z-10 shrink-0 outline-none transition-colors",
        "hover:bg-(--ink-accent)/50 active:bg-(--ink-accent) focus-visible:bg-(--ink-accent)/50",
        horizontal ? "inset-y-0 w-1.5" : "inset-x-0 h-1.5",
        side === "left" && "-right-0.5",
        side === "right" && "-left-0.5",
        side === "top" && "-bottom-0.5",
        side === "bottom" && "-top-0.5",
      )}
    />
  );
}

/**
 * The weight splitter between two adjacent groups in the same region:
 * redistributes flex weight between group `index` and `index + 1`.
 */
export function GroupSplitter({
  side,
  index,
  containerRef,
}: {
  side: DockSide;
  index: number;
  containerRef: React.RefObject<HTMLElement | null>;
}) {
  const horizontal = isHorizontalSide(side);
  // Groups stack along the region's MAIN axis: vertically for sidebars,
  // horizontally for top/bottom bands — the opposite of the region splitter.
  const stackVertical = horizontal;
  const raf = useRef(0);

  const onPointerDown = useCallback(
    (e: ReactPointerEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();
      const el = e.currentTarget;
      try {
        el.setPointerCapture(e.pointerId);
      } catch {
        /* fine */
      }
      const store = useDockLayout.getState();
      const startWeights = store.layout.docks[side].groups.map((g) => g.weight);
      const total = startWeights.reduce((a, b) => a + b, 0) || 1;
      const stackPx = stackVertical
        ? (containerRef.current?.getBoundingClientRect().height ?? 600)
        : (containerRef.current?.getBoundingClientRect().width ?? 600);
      const start = stackVertical ? e.clientY : e.clientX;
      let pending: number[] | null = null;

      const onMove = (ev: PointerEvent) => {
        const deltaPx = (stackVertical ? ev.clientY : ev.clientX) - start;
        const deltaW = (deltaPx / Math.max(1, stackPx)) * total;
        const a = startWeights[index] ?? 1;
        const b = startWeights[index + 1] ?? 1;
        const min = total * 0.1;
        const na = Math.max(min, Math.min(a + b - min, a + deltaW));
        const nb = a + b - na;
        pending = [...startWeights];
        pending[index] = na;
        pending[index + 1] = nb;
        if (raf.current) return;
        raf.current = requestAnimationFrame(() => {
          raf.current = 0;
          if (pending) useDockLayout.getState().setGroupWeights(side, pending);
        });
      };
      const finish = (ev: PointerEvent) => {
        el.removeEventListener("pointermove", onMove);
        el.removeEventListener("pointerup", finish);
        el.removeEventListener("pointercancel", finish);
        if (raf.current) {
          cancelAnimationFrame(raf.current);
          raf.current = 0;
        }
        if (pending) {
          useDockLayout.getState().setGroupWeights(side, pending);
          useDockLayout.getState().commitRegion();
        }
        try {
          el.releasePointerCapture(ev.pointerId);
        } catch {
          /* fine */
        }
      };
      el.addEventListener("pointermove", onMove);
      el.addEventListener("pointerup", finish);
      el.addEventListener("pointercancel", finish);
    },
    [side, index, stackVertical, containerRef],
  );

  return (
    <div
      role="separator"
      aria-orientation={stackVertical ? "horizontal" : "vertical"}
      aria-label="Resize panel group"
      onPointerDown={onPointerDown}
      style={{
        cursor: stackVertical ? "row-resize" : "col-resize",
        touchAction: "none",
      }}
      className={cn(
        "shrink-0 bg-(--ink-border) transition-colors hover:bg-(--ink-accent)/60 active:bg-(--ink-accent)",
        stackVertical ? "h-1 w-full" : "h-full w-1",
      )}
    />
  );
}
