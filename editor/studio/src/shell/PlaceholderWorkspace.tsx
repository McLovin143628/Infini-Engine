/**
 * Interim center workspace — the Phase 0 splitter + viewport + right stack,
 * extracted verbatim from the old App.tsx. P1.2 replaces this with the real
 * docking workspace; the drag chip keeps exercising the Spike A drop
 * handoff until the Content Drawer takes over that job (P4.4).
 */
import { useCallback, useRef, useState } from "react";
import { viewport } from "../lib/ipc";
import ViewportPanel from "../viewport/ViewportPanel";

export default function PlaceholderWorkspace() {
  const [rightWidth, setRightWidth] = useState(288);
  const [dragGhost, setDragGhost] = useState<{ x: number; y: number } | null>(null);
  const splitDrag = useRef(false);

  const onSplitterDown = useCallback((e: React.PointerEvent) => {
    splitDrag.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);
  const onSplitterMove = useCallback((e: React.PointerEvent) => {
    if (!splitDrag.current) return;
    const width = document.documentElement.clientWidth - e.clientX;
    setRightWidth(Math.min(640, Math.max(160, width)));
  }, []);
  const onSplitterUp = useCallback(() => {
    splitDrag.current = false;
  }, []);

  // Drag-drop handoff stub: the HTML ghost dies over the native hole by
  // design (airspace rule); the drop point crosses via IPC instead.
  const onChipDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setDragGhost({ x: e.clientX, y: e.clientY });
  }, []);
  const onChipMove = useCallback((e: React.PointerEvent) => {
    setDragGhost((g) => (g ? { x: e.clientX, y: e.clientY } : g));
  }, []);
  const onChipUp = useCallback((e: React.PointerEvent) => {
    setDragGhost(null);
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
    <div className="flex min-h-0 flex-1">
      {/* The native wgpu child window mirrors this element's rectangle. */}
      <div className="m-1 mr-0 flex min-w-0 flex-1">
        <ViewportPanel />
      </div>

      <div
        className="my-1 w-1.5 shrink-0 cursor-col-resize rounded hover:bg-(--ink-accent)"
        onPointerDown={onSplitterDown}
        onPointerMove={onSplitterMove}
        onPointerUp={onSplitterUp}
      />

      <div className="m-1 ml-0 flex shrink-0 flex-col gap-1" style={{ width: rightWidth }}>
        <div className="flex-1 rounded border border-(--ink-border) bg-(--ink-bg-1)">
          <div className="border-b border-(--ink-border) bg-(--ink-bg-2) px-2 py-1">Outliner</div>
          <div className="p-2">
            <div
              className="inline-flex cursor-grab touch-none select-none items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-0.5 hover:border-(--ink-accent)"
              onPointerDown={onChipDown}
              onPointerMove={onChipMove}
              onPointerUp={onChipUp}
            >
              <span className="text-(--ink-accent)">⬢</span> TestActor
            </div>
            <div className="mt-2 text-(--ink-text-dim)">
              Drag the actor into the viewport (drop handoff stub)
            </div>
          </div>
        </div>
        <div className="flex-1 rounded border border-(--ink-border) bg-(--ink-bg-1)">
          <div className="border-b border-(--ink-border) bg-(--ink-bg-2) px-2 py-1">Details</div>
          <div className="p-2 text-(--ink-text-dim)">Select an object to view details</div>
        </div>
      </div>

      {dragGhost && (
        <div
          className="pointer-events-none fixed z-50 rounded border border-(--ink-accent) bg-(--ink-bg-2) px-2 py-0.5 opacity-80"
          style={{ left: dragGhost.x + 10, top: dragGhost.y + 6 }}
        >
          ⬢ TestActor
        </div>
      )}
    </div>
  );
}
