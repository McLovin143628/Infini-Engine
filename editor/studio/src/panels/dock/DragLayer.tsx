import { createPortal } from "react-dom";
import { panelDefFor, panelTitle } from "../panelRegistry";
import { useDockDrag } from "./dockDragStore";
import { useDockLayout } from "./dockLayoutStore";
import { GHOST_H, GHOST_W } from "./dragController";

/**
 * Portal layer rendering the panel-drag ghost + drop previews over
 * everything HTML (document.body). Pure presentation: all state comes from
 * `dockDragStore`; transform-only updates so a 120 Hz drag stays cheap.
 * (Over the native viewport hole the ghost is occluded — airspace rule.)
 */
export function DragLayer() {
  const dragging = useDockDrag((s) => s.dragging);
  const panel = useDockLayout((s) => (dragging ? s.layout.panels[dragging.panelId] : undefined));
  if (!dragging || !panel) return null;

  const def = panelDefFor(dragging.panelId);
  const Icon = def?.icon;
  const title = panelTitle(dragging.panelId, panel.params);
  const showGhost = !dragging.liveMove;
  const t = dragging.target;
  const windowTarget = t?.target.kind === "window";

  return createPortal(
    <div className="pointer-events-none fixed inset-0 z-[80]" aria-hidden data-panel-drag-layer>
      {/* Drop preview */}
      {t && !windowTarget && (
        <div
          className="absolute rounded-md border-2 border-(--ink-accent) bg-(--ink-accent-muted) transition-[left,top,width,height] duration-100 ease-out"
          style={{
            left: t.preview.x,
            top: t.preview.y,
            width: t.preview.w,
            height: t.preview.h,
          }}
        />
      )}
      {/* Tab-insert caret */}
      {t?.tabCaret && (
        <div
          className="absolute w-0.5 rounded-full bg-(--ink-accent)"
          style={{ left: t.tabCaret.x, top: t.tabCaret.y, height: t.tabCaret.h }}
        />
      )}
      {/* Ghost chip */}
      {showGhost && (
        <div
          className="absolute flex items-center gap-2 rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-2) px-3 text-xs font-semibold text-(--ink-text)"
          style={{
            width: windowTarget ? GHOST_W + 40 : GHOST_W,
            height: GHOST_H,
            transform: `translate3d(${Math.min(dragging.ghost.x, window.innerWidth - GHOST_W)}px, ${Math.min(dragging.ghost.y, window.innerHeight - GHOST_H)}px, 0) scale(0.98)`,
            willChange: "transform",
            boxShadow: `0 10px 32px var(--ink-shadow)`,
          }}
        >
          {Icon && <Icon size={15} className="shrink-0 text-(--ink-accent)" />}
          <span className="truncate">{windowTarget ? `${title} → new window` : title}</span>
        </div>
      )}
    </div>,
    document.body,
  );
}
