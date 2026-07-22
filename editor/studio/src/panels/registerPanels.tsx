import {
  Clapperboard,
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
  Sprout,
  SquareTerminal,
  Workflow,
} from "lucide-react";
import DetailsPanel from "./DetailsPanel";
import EditorPanel from "./EditorPanel";
import FileExplorerPanel from "./FileExplorerPanel";
import GitPanel from "./GitPanel";
import OutlinerPanel from "./OutlinerPanel";
import OutputLogPanel from "./OutputLogPanel";
import ProblemsPanel from "./ProblemsPanel";
import SearchPanel from "./SearchPanel";
import SequencerPanel from "./SequencerPanel";
import SpriteSheetPanel from "./SpriteSheetPanel";
import TerminalPanel from "./TerminalPanel";
import TilemapPanel from "./TilemapPanel";
import { BlueprintCanvas } from "./blueprint/BlueprintCanvas";
import { MaterialCanvas } from "./material/MaterialCanvas";
import { PcgCanvas } from "./pcg/PcgCanvas";
import { StateMachineCanvas } from "./sm/StateMachineCanvas";
import PlaceActorsPanel from "./placeActors/PlaceActorsPanel";
import WorldSettingsPanel from "./worldSettings/WorldSettingsPanel";
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

registerPanelType({
  type: "outliner",
  title: () => "Outliner",
  icon: ListTree,
  component: OutlinerPanel,
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
  component: WorldSettingsPanel,
  singleton: true,
  defaultLocation: "right",
  defaultSize: { w: 320, h: 420 },
});

registerPanelType({
  type: "placeActors",
  title: () => "Place Actors",
  icon: PlusSquare,
  component: PlaceActorsPanel,
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
  type: "pcg",
  title: () => "PCG",
  icon: Sprout,
  component: PcgCanvas,
  singleton: true,
  defaultLocation: "bottom",
  defaultSize: { w: 1040, h: 560 },
});

registerPanelType({
  type: "stateMachine",
  title: () => "State Machine",
  icon: Workflow,
  component: StateMachineCanvas,
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
  type: "sequencer",
  title: () => "Sequencer",
  icon: Clapperboard,
  component: SequencerPanel,
  singleton: true,
  defaultLocation: "bottom",
  defaultSize: { w: 960, h: 340 },
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
