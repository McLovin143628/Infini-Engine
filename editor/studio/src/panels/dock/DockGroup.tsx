import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { cn } from "../../lib/utils";
import { panelDefFor, panelTitle } from "../panelRegistry";
import { useDockDrag } from "./dockDragStore";
import { useDockLayout } from "./dockLayoutStore";
import { beginPanelDrag } from "./dragController";
import { PanelChromeContext } from "./panelChromeContext";
import { dockPanelToSide, PanelContextMenu } from "./PanelContextMenu";
import type { DockGroupState, DockSide } from "./dockTypes";

/**
 * One tab group inside a dock region: a compact tab strip (the group's
 * header/drag surface) over the active panel's body. Every tab's body stays
 * mounted (`hidden` when inactive) so panel state — scroll positions, form
 * inputs, virtualized rows — survives tab switches, VS Code-style.
 *
 * A group holding a single panel renders as a full-width panel HEADER
 * (UE-style), not a browser tab. Multi-panel groups keep the tab strip.
 * The drag engine is unaffected: `role="tablist"`, `data-dock-tab`, and
 * `beginPanelDrag` are preserved in both modes.
 *
 * Keyboard: Left/Right move focus, Enter/Space activate, Delete/middle-
 * click close, Ctrl+Shift+Arrow docks to that side.
 */
export function DockGroup({ side, group }: { side: DockSide; group: DockGroupState }) {
  const activateTab = useDockLayout((s) => s.activateTab);
  const closePanel = useDockLayout((s) => s.closePanel);
  const setFocusedPanel = useDockLayout((s) => s.setFocusedPanel);
  const panels = useDockLayout((s) => s.layout.panels);
  const draggingId = useDockDrag((s) => s.dragging?.panelId ?? null);
  const stripRef = useRef<HTMLDivElement | null>(null);

  const single = group.tabs.length === 1;

  // Keep the active tab visible when the strip overflows.
  useEffect(() => {
    const strip = stripRef.current;
    if (!strip) return;
    const active = strip.querySelector<HTMLElement>('[data-active="true"]');
    active?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [group.activeTab]);

  const focusTab = (delta: number, fromId: string) => {
    const i = group.tabs.indexOf(fromId);
    const next = group.tabs[(i + delta + group.tabs.length) % group.tabs.length];
    if (!next) return;
    stripRef.current?.querySelector<HTMLElement>(`[data-tab-id="${CSS.escape(next)}"]`)?.focus();
  };

  return (
    <section
      className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
      aria-label={`${side} dock group`}
      data-dock-group={group.id}
    >
      <div
        ref={stripRef}
        role="tablist"
        aria-orientation="horizontal"
        className={cn(
          "flex h-8 shrink-0 gap-0.5 border-b border-(--ink-border) bg-(--ink-bg-2)",
          single
            ? "items-center px-1"
            : "items-end overflow-x-auto px-1 pt-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden",
        )}
      >
        {group.tabs.map((panelId) => {
          const def = panelDefFor(panelId);
          const p = panels[panelId];
          if (!def || !p) return null;
          const active = group.activeTab === panelId;
          const Icon = def.icon;
          const title = panelTitle(panelId, p.params);
          return (
            <PanelContextMenu key={panelId} panelId={panelId}>
              <button
                type="button"
                role="tab"
                data-tab-id={panelId}
                data-active={active || undefined}
                data-dock-tab={panelId}
                aria-selected={active}
                tabIndex={active ? 0 : -1}
                title={title}
                onPointerDown={(e) => {
                  if (e.button !== 0) return;
                  activateTab(panelId);
                  // The unified drag controller takes over past the 4px
                  // threshold (tear out → float / re-dock / new group).
                  beginPanelDrag(panelId, e, "tab");
                }}
                onAuxClick={(e) => {
                  if (e.button === 1) closePanel(panelId);
                }}
                onKeyDown={(e) => {
                  // Ctrl+Shift+Arrow moves the panel to that side's dock —
                  // keyboard parity with the drag gesture.
                  if (e.ctrlKey && e.shiftKey) {
                    const toSide = (
                      {
                        ArrowLeft: "left",
                        ArrowRight: "right",
                        ArrowUp: "top",
                        ArrowDown: "bottom",
                      } as const
                    )[e.key];
                    if (toSide) {
                      e.preventDefault();
                      dockPanelToSide(panelId, toSide);
                      return;
                    }
                  }
                  if (e.key === "ArrowRight") focusTab(1, panelId);
                  else if (e.key === "ArrowLeft") focusTab(-1, panelId);
                  else if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    activateTab(panelId);
                  } else if (e.key === "Delete") closePanel(panelId);
                }}
                className={cn(
                  "group/tab relative flex min-w-0 items-center gap-1.5 text-xs outline-none transition-colors",
                  draggingId === panelId && "opacity-40",
                  single
                    ? // Panel header: full-width, grab cursor, no tab shape.
                      "flex-1 cursor-grab rounded-md px-2 py-1 font-semibold text-(--ink-text) hover:bg-(--ink-bg-3) active:cursor-grabbing"
                    : cn(
                        "max-w-40 shrink rounded-t-md px-2.5 py-1.5",
                        active
                          ? "bg-(--ink-bg-3) font-semibold text-(--ink-text)"
                          : "text-(--ink-text-dim) hover:bg-(--ink-bg-3)/60 hover:text-(--ink-text)",
                      ),
                )}
              >
                <Icon
                  size={14}
                  className={cn("shrink-0", (active || single) && "text-(--ink-accent)")}
                />
                <span className={cn("truncate", single && "flex-1 text-left")}>{title}</span>
                {/* A tab closes from inside itself; a single-panel HEADER
                    closes from the strip-level button below instead. */}
                {!single && (
                  <span
                    role="button"
                    tabIndex={-1}
                    aria-label={`Close ${title}`}
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => {
                      e.stopPropagation();
                      closePanel(panelId);
                    }}
                    className={cn(
                      "-mr-1 size-4 shrink-0 items-center justify-center rounded-sm text-(--ink-text-faint) hover:bg-(--ink-error)/15 hover:text-(--ink-error)",
                      active ? "inline-flex" : "hidden group-hover/tab:inline-flex",
                    )}
                  >
                    <X size={12} />
                  </span>
                )}
                {active && !single && (
                  <span
                    aria-hidden
                    className="absolute inset-x-1.5 -bottom-px h-0.5 rounded-full bg-(--ink-accent)"
                  />
                )}
              </button>
            </PanelContextMenu>
          );
        })}

        {single && group.activeTab && (
          <button
            type="button"
            aria-label={`Close ${panelTitle(group.activeTab, panels[group.activeTab]?.params ?? null)}`}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={() => closePanel(group.activeTab)}
            className="inline-flex size-5 shrink-0 cursor-pointer items-center justify-center rounded-sm text-(--ink-text-faint) outline-none transition-colors hover:bg-(--ink-error)/15 hover:text-(--ink-error)"
          >
            <X size={12} />
          </button>
        )}
      </div>

      <PanelChromeContext.Provider value="docked">
        {group.tabs.map((panelId) => {
          const def = panelDefFor(panelId);
          const p = panels[panelId];
          if (!def || !p) return null;
          const Body = def.component;
          const active = group.activeTab === panelId;
          return (
            <div
              key={panelId}
              role="tabpanel"
              hidden={!active}
              // Focus follows the pointer INTO the body, not just onto the tab
              // (P23.2a): clicking inside the material canvas is the gesture
              // that has to make Ctrl+Z mean "the material". Capture phase, so
              // a child that stops propagation cannot swallow it.
              onPointerDownCapture={() => setFocusedPanel(panelId)}
              className={cn(
                "min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-(--ink-bg-1)",
                active ? "flex" : "hidden",
              )}
            >
              <Body panelId={panelId} params={p.params} />
            </div>
          );
        })}
      </PanelChromeContext.Provider>
    </section>
  );
}
