import { useEffect, useRef, type ReactNode } from "react";
// Side-effect: the built-in panel types must be registered before the
// first hydrate (repairLayout drops unregistered types).
import "../registerPanels";
import { usePanelWindowManager } from "../window/panelWindows";
import { useDockDrag } from "./dockDragStore";
import { DockRegion } from "./DockRegion";
import { loadActiveLayoutDoc, useDockLayout } from "./dockLayoutStore";
import { DragLayer } from "./DragLayer";
import { FloatLayer } from "./FloatLayer";

/**
 * The dock workspace shell (ported from GeoCanvas, ROADMAP P1.2): four
 * always-mounted dock regions around the center content (the native
 * viewport), plus the floating layer over everything. The regions collapse
 * to zero size when empty but never leave the tree, so the center
 * container's position in the React tree is invariant — the native
 * viewport RESIZES, never remounts (Spike A invariant).
 *
 * Also owns the workspace ResizeObserver (hydrate-once + clamp-on-shrink)
 * and the startup hydrate from the persisted active layout.
 */
export function DockWorkspace({ children }: { children: ReactNode }) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Detached OS windows: layout-declarative spawn/destroy + store bridge
  // host + dock-back drag listeners (main window only — this component
  // never mounts in a panel window).
  usePanelWindowManager();

  // Register the workspace element with the drag store — the coordinate
  // anchor for the unified drag controller's drop zones + edge bands.
  useEffect(() => {
    useDockDrag.getState().setWorkspaceEl(containerRef.current);
    return () => useDockDrag.getState().setWorkspaceEl(null);
  }, []);

  // Track the workspace size: feeds hydrate (once) and clampAll (resizes).
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onSize = () => {
      const rect = el.getBoundingClientRect();
      if (rect.width < 100 || rect.height < 100) return;
      const size = { w: rect.width, h: rect.height };
      const store = useDockLayout.getState();
      if (store.hydrated) store.clampAll(size);
      else store.setContainer(size);
    };
    const ro = new ResizeObserver(onSize);
    ro.observe(el);
    onSize();
    return () => ro.disconnect();
  }, []);

  // Startup hydrate: once a container size lands, load the persisted
  // active layout (or seed the UE-parity default). Guarded so StrictMode's
  // double effect run hydrates exactly once.
  const container = useDockLayout((s) => s.container);
  const hydrated = useDockLayout((s) => s.hydrated);
  const hydrating = useRef(false);
  useEffect(() => {
    if (!container || hydrated || hydrating.current) return;
    hydrating.current = true;
    void loadActiveLayoutDoc()
      .then((doc) => {
        const store = useDockLayout.getState();
        if (!store.hydrated) store.hydrate(doc, store.container ?? container);
      })
      .finally(() => {
        hydrating.current = false;
      });
  }, [container, hydrated]);

  // CSS-grid dock shell: left/right sidebars span the full height (all
  // three rows); top/bottom regions inset only into the CENTER column,
  // between the sidebars. The center (`children`) stays a single,
  // always-rendered grid child in the same tree position.
  return (
    <div
      ref={containerRef}
      className="relative grid min-h-0 min-w-0 flex-1"
      style={{
        gridTemplateColumns: "min-content 1fr min-content",
        gridTemplateRows: "min-content 1fr min-content",
        gridTemplateAreas: `"left top right" "left center right" "left bottom right"`,
      }}
    >
      <DockRegion side="left" />
      <DockRegion side="top" />
      <div data-dock-center className="relative min-h-0 min-w-0" style={{ gridArea: "center" }}>
        {children}
      </div>
      <DockRegion side="bottom" />
      <DockRegion side="right" />
      <FloatLayer />
      <DragLayer />
    </div>
  );
}
