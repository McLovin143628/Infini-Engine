import { useCallback, useState } from "react";
import {
  Code2,
  FolderTree,
  GitBranch,
  Globe2,
  Grid2x2,
  Grid3x3,
  ListChecks,
  ListTree,
  Palette,
  PlusSquare,
  ScrollText,
  Search,
  SlidersHorizontal,
  SquareTerminal,
  Workflow,
} from "lucide-react";
import { viewport } from "../lib/ipc";
import DetailsPanel from "./DetailsPanel";
import EditorPanel from "./EditorPanel";
import FileExplorerPanel from "./FileExplorerPanel";
import GitPanel from "./GitPanel";
import OutlinerPanel from "./OutlinerPanel";
import OutputLogPanel from "./OutputLogPanel";
import ProblemsPanel from "./ProblemsPanel";
import SearchPanel from "./SearchPanel";
import SpriteSheetPanel from "./SpriteSheetPanel";
import TerminalPanel from "./TerminalPanel";
import TilemapPanel from "./TilemapPanel";
import { BlueprintCanvas } from "./blueprint/BlueprintCanvas";
import { MaterialCanvas } from "./material/MaterialCanvas";
import { registerPanelType } from "./panelRegistry";

/**
 * Built-in panel registrations. Imported for its side effect by
 * `DockWorkspace` and `PanelWindowApp` BEFORE the first hydrate
 * (`repairLayout` drops unregistered types).
 *
 * Outliner/Details/Output Log are the real P1.4 implementations on mock
 * data; later phases bind them to the engine (P3 world, tracing already
 * live for the log).
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

/** Outliner body: the real tree (P1.4.1) + the Spike A demo chip footer. */
function OutlinerBody() {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <OutlinerPanel />
      <div className="border-t border-(--ink-border) p-1.5">
        <TestActorChip />
      </div>
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
  component: OutlinerBody,
  singleton: true,
  defaultLocation: "right",
  defaultSize: { w: 320, h: 420 },
});

registerPanelType({
  type: "details",
  title: () => "Details",
  icon: SlidersHorizontal,
  component: DetailsPanel,
  singleton: true,
  defaultLocation: "right",
  defaultSize: { w: 320, h: 480 },
});

registerPanelType({
  type: "outputLog",
  title: () => "Output Log",
  icon: ScrollText,
  component: OutputLogPanel,
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

registerPanelType({
  type: "terminal",
  title: () => "Terminal",
  icon: SquareTerminal,
  component: TerminalPanel,
  singleton: true,
  defaultLocation: "bottom",
  defaultSize: { w: 720, h: 260 },
  // A session's PTY dies with the app; don't try to restore it across restarts.
  transient: true,
});

// ── IDE panels (P5.4) ──────────────────────────────────────────────────────

registerPanelType({
  type: "explorer",
  title: () => "Explorer",
  icon: FolderTree,
  component: FileExplorerPanel,
  singleton: true,
  defaultLocation: "left",
  defaultSize: { w: 280, h: 480 },
});

registerPanelType({
  type: "git",
  title: () => "Source Control",
  icon: GitBranch,
  component: GitPanel,
  singleton: true,
  defaultLocation: "left",
  defaultSize: { w: 300, h: 480 },
});

registerPanelType({
  type: "editor",
  title: () => "Code Editor",
  icon: Code2,
  component: EditorPanel,
  singleton: true,
  defaultLocation: "bottom",
  defaultSize: { w: 900, h: 480 },
});

registerPanelType({
  type: "blueprint",
  title: () => "Blueprint",
  icon: Workflow,
  component: BlueprintCanvas,
  singleton: true,
  defaultLocation: "bottom",
  defaultSize: { w: 1000, h: 560 },
});

registerPanelType({
  type: "material",
  title: () => "Material",
  icon: Palette,
  component: MaterialCanvas,
  singleton: true,
  defaultLocation: "bottom",
  defaultSize: { w: 1040, h: 560 },
});

registerPanelType({
  type: "spriteSheet",
  title: () => "Sprite Sheet",
  icon: Grid3x3,
  component: SpriteSheetPanel,
  singleton: true,
  defaultLocation: "bottom",
  defaultSize: { w: 900, h: 560 },
});

registerPanelType({
  type: "tilemap",
  title: () => "Tilemap",
  icon: Grid2x2,
  component: TilemapPanel,
  singleton: true,
  defaultLocation: "bottom",
  defaultSize: { w: 900, h: 560 },
});

registerPanelType({
  type: "search",
  title: () => "Search",
  icon: Search,
  component: SearchPanel,
  singleton: true,
  defaultLocation: "left",
  defaultSize: { w: 300, h: 480 },
  transient: true,
});

registerPanelType({
  type: "problems",
  title: () => "Problems",
  icon: ListChecks,
  component: ProblemsPanel,
  singleton: true,
  defaultLocation: "bottom",
  defaultSize: { w: 720, h: 220 },
  transient: true,
});
