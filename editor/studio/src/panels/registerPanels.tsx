import { useCallback, useState } from "react";
import { Globe2, ListTree, PlusSquare, ScrollText, SlidersHorizontal } from "lucide-react";
import { viewport } from "../lib/ipc";
import { registerPanelType } from "./panelRegistry";

/**
 * Built-in panel registrations. Imported for its side effect by
 * `DockWorkspace` and `PanelWindowApp` BEFORE the first hydrate
 * (`repairLayout` drops unregistered types).
 *
 * Phase 1 ships placeholder bodies; P1.4 upgrades Outliner / Details /
 * Output Log to their real (still mock-data) implementations, and later
 * phases bind them to the engine.
 */

/** The Spike A drop-handoff demo chip, kept alive inside the Outliner
 *  placeholder until the Content Drawer takes over drag-to-viewport (P4.4). */
function TestActorChip() {
  const [ghost, setGhost] = useState<{ x: number; y: number } | null>(null);

  const onDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setGhost({ x: e.clientX, y: e.clientY });
  }, []);
  const onMove = useCallback((e: React.PointerEvent) => {
    setGhost((g) => (g ? { x: e.clientX, y: e.clientY } : g));
  }, []);
  const onUp = useCallback((e: React.PointerEvent) => {
    setGhost(null);
    const hole = document.querySelector("[data-viewport-hole]");
    if (!hole) return;
    const r = hole.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) {
      return;
    }
    const s = window.devicePixelRatio;
    viewport
      .drop({ x: (e.clientX - r.left) * s, y: (e.clientY - r.top) * s, payload: "TestActor" })
      .catch((err) => console.error("viewport drop failed:", err));
  }, []);

  return (
    <>
      <div
        className="inline-flex cursor-grab touch-none select-none items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-0.5 hover:border-(--ink-accent)"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
      >
        <span className="text-(--ink-accent)">⬢</span> TestActor
      </div>
      {ghost && (
        <div
          className="pointer-events-none fixed z-50 rounded border border-(--ink-accent) bg-(--ink-bg-2) px-2 py-0.5 opacity-80"
          style={{ left: ghost.x + 10, top: ghost.y + 6 }}
        >
          ⬢ TestActor
        </div>
      )}
    </>
  );
}

function OutlinerPlaceholder() {
  return (
    <div className="min-h-0 flex-1 overflow-auto p-2">
      <TestActorChip />
      <div className="mt-2 text-(--ink-text-dim)">
        Drag the actor into the viewport (drop handoff stub). The real Outliner tree lands with
        P1.4 / P3.2.
      </div>
    </div>
  );
}

function DetailsPlaceholder() {
  return (
    <div className="min-h-0 flex-1 overflow-auto p-2 text-(--ink-text-dim)">
      Select an object to view details
    </div>
  );
}

function OutputLogPlaceholder() {
  return (
    <div className="min-h-0 flex-1 overflow-auto p-2 font-mono text-xs text-(--ink-text-dim)">
      Output Log — tracing stream lands with P1.4.
    </div>
  );
}

function WorldSettingsPlaceholder() {
  return (
    <div className="min-h-0 flex-1 overflow-auto p-2 text-(--ink-text-dim)">
      World Settings arrive with the scene model (Phase 3).
    </div>
  );
}

function PlaceActorsPlaceholder() {
  return (
    <div className="min-h-0 flex-1 overflow-auto p-2 text-(--ink-text-dim)">
      Place Actors palette arrives with the scene model (Phase 3).
    </div>
  );
}

registerPanelType({
  type: "outliner",
  title: () => "Outliner",
  icon: ListTree,
  component: OutlinerPlaceholder,
  singleton: true,
  defaultLocation: "right",
  defaultSize: { w: 320, h: 420 },
});

registerPanelType({
  type: "details",
  title: () => "Details",
  icon: SlidersHorizontal,
  component: DetailsPlaceholder,
  singleton: true,
  defaultLocation: "right",
  defaultSize: { w: 320, h: 480 },
});

registerPanelType({
  type: "outputLog",
  title: () => "Output Log",
  icon: ScrollText,
  component: OutputLogPlaceholder,
  singleton: true,
  defaultLocation: "bottom",
  defaultSize: { w: 720, h: 260 },
});

registerPanelType({
  type: "worldSettings",
  title: () => "World Settings",
  icon: Globe2,
  component: WorldSettingsPlaceholder,
  singleton: true,
  defaultLocation: "right",
  defaultSize: { w: 320, h: 420 },
});

registerPanelType({
  type: "placeActors",
  title: () => "Place Actors",
  icon: PlusSquare,
  component: PlaceActorsPlaceholder,
  singleton: true,
  defaultLocation: "left",
  defaultSize: { w: 300, h: 460 },
});
