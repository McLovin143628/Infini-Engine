import type { CSSProperties } from "react";
import { ChevronDown, ChevronUp, X } from "lucide-react";
import { cn } from "../lib/utils";
import { useDockDrag } from "./dock/dockDragStore";
import { useDockLayout } from "./dock/dockLayoutStore";
import { PanelContextMenu } from "./dock/PanelContextMenu";
import { panelDefFor, panelTitle } from "./panelRegistry";
import { HANDLE_CURSORS, HANDLE_IDS, type HandleId } from "./panelRect";
import { useFloatingPanelDrag } from "./useFloatingPanelDrag";

/** Edge strips are 6px thick, corner pads 10px square, hanging half
 *  outside the frame for an easier grab. */
function handleStyle(h: HandleId): CSSProperties {
  const base: CSSProperties = {
    position: "absolute",
    touchAction: "none",
    cursor: HANDLE_CURSORS[h],
  };
  switch (h) {
    case "n":
      return { ...base, top: -3, left: 8, right: 8, height: 6 };
    case "s":
      return { ...base, bottom: -3, left: 8, right: 8, height: 6 };
    case "e":
      return { ...base, right: -3, top: 8, bottom: 8, width: 6 };
    case "w":
      return { ...base, left: -3, top: 8, bottom: 8, width: 6 };
    case "nw":
      return { ...base, top: -4, left: -4, width: 10, height: 10 };
    case "ne":
      return { ...base, top: -4, right: -4, width: 10, height: 10 };
    case "se":
      return { ...base, bottom: -4, right: -4, width: 10, height: 10 };
    case "sw":
      return { ...base, bottom: -4, left: -4, width: 10, height: 10 };
  }
}

/**
 * Absolute-positioned card for one FLOATING panel: owns position, size,
 * stacking, header chrome (drag handle, minimize chevron, hide ✕), and the
 * drag/resize gestures. Width always applies; height only while open (a
 * minimized panel is its header strip at width `w`). The 8 resize handles
 * render only while open.
 *
 * Airspace note: floating panels are HTML — over the native viewport hole
 * they are occluded by the engine's child window (decided architecture
 * §2.3). Docked regions and detached OS windows are unaffected; that is
 * one reason "Move to new window" exists on every panel.
 */
export function FloatingPanelFrame({
  id,
  children,
}: {
  id: string;
  children: React.ReactNode;
}) {
  const panel = useDockLayout((s) => s.layout.panels[id]);
  const setMinimized = useDockLayout((s) => s.setMinimized);
  const hidePanel = useDockLayout((s) => s.hidePanel);
  const bringToFront = useDockLayout((s) => s.bringToFront);
  const setFocusedPanel = useDockLayout((s) => s.setFocusedPanel);
  // Dim the source panel while its drag ghost is over a dock target.
  const ghosting = useDockDrag((s) => s.dragging?.panelId === id && !s.dragging.liveMove);
  const { startHeaderDrag, startResize } = useFloatingPanelDrag(id);

  if (!panel || panel.location.kind !== "floating") return null;
  const open = !panel.minimized;
  const def = panelDefFor(id);
  const Icon = def?.icon;
  const title = panelTitle(id, panel.params);

  return (
    <div
      className="pointer-events-auto absolute flex flex-col overflow-visible rounded-md border border-(--ink-border-strong) bg-(--ink-bg-1)"
      data-float-panel={id}
      style={{
        left: panel.floatRect.x,
        top: panel.floatRect.y,
        width: panel.floatRect.w,
        ...(open ? { height: panel.floatRect.h } : {}),
        zIndex: 20 + panel.floatZ,
        boxShadow: `0 10px 32px var(--ink-shadow)`,
        ...(ghosting ? { opacity: 0.4 } : {}),
      }}
      // Clicking anywhere in a panel raises it (no-op when already front) and
      // makes it the focused panel (P23.2a) — raising and focusing are the same
      // gesture for a float, so they share a handler.
      onPointerDown={() => {
        bringToFront(id);
        setFocusedPanel(id);
      }}
    >
      <PanelContextMenu panelId={id}>
        <header
          className="flex h-8 shrink-0 cursor-grab select-none items-center gap-1.5 rounded-t-md border-b border-(--ink-border) bg-(--ink-bg-2) px-2 active:cursor-grabbing"
          onPointerDown={startHeaderDrag}
          onDoubleClick={() => setMinimized(id, open)}
        >
          {Icon && <Icon size={14} className="shrink-0 text-(--ink-accent)" />}
          <span className="min-w-0 flex-1 truncate text-xs font-semibold">{title}</span>
          <button
            type="button"
            aria-label={open ? "Minimize panel" : "Restore panel"}
            className="inline-flex size-5 items-center justify-center rounded-sm text-(--ink-text-faint) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
            onClick={() => setMinimized(id, open)}
          >
            {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>
          <button
            type="button"
            aria-label={`Hide ${title}`}
            className="inline-flex size-5 items-center justify-center rounded-sm text-(--ink-text-faint) hover:bg-(--ink-error)/15 hover:text-(--ink-error)"
            onClick={() => hidePanel(id)}
          >
            <X size={13} />
          </button>
        </header>
      </PanelContextMenu>

      <div className={cn("min-h-0 flex-1 flex-col overflow-hidden", open ? "flex" : "hidden")}>
        {children}
      </div>

      {open &&
        HANDLE_IDS.map((h) => (
          <div
            key={h}
            role="presentation"
            aria-hidden
            style={handleStyle(h)}
            onPointerDown={(e) => startResize(h, e)}
          />
        ))}
    </div>
  );
}
