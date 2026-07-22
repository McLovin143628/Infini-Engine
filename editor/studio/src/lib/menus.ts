/**
 * The UE-parity menu tree (ROADMAP §4 "layout parity", P1.1 batch 2).
 *
 * Data only — `MenuBar.tsx` renders it, and every item dispatches through
 * the command registry (`lib/commands.ts`). Actions that land in later
 * phases are enumerated now (with the phase noted) so the menu structure
 * is complete from the first milestone; executing one surfaces a
 * "coming in Phase N" status message via the unhandled-command hook.
 */

import type { RecentProjectDto } from "../bindings/RecentProjectDto";
import type { CommandDef } from "./commands";
import { registerCommands } from "./commands";
import { BUILTIN_THEMES, setTheme } from "./theme";

export interface MenuAction {
  kind: "action";
  command: string; // command registry id
  label: string;
  shortcut?: string;
  /** Secondary text (tooltip) — e.g. the full path on a Recent Projects entry. */
  detail?: string;
  /** Rendered but non-interactive (e.g. the "(none yet)" placeholder). */
  disabled?: boolean;
}

export interface MenuSubmenu {
  kind: "submenu";
  label: string;
  items: MenuNode[];
}

export interface MenuSeparator {
  kind: "separator";
}

export type MenuNode = MenuAction | MenuSubmenu | MenuSeparator;

export interface TopMenu {
  id: string;
  label: string;
  items: MenuNode[];
}

const sep: MenuSeparator = { kind: "separator" };
const act = (command: string, label: string, shortcut?: string): MenuAction => ({
  kind: "action",
  command,
  label,
  shortcut,
});
const sub = (label: string, items: MenuNode[]): MenuSubmenu => ({ kind: "submenu", label, items });

export const MENU_BAR: TopMenu[] = [
  {
    id: "file",
    label: "File",
    items: [
      act("file.newLevel", "New Level…", "Ctrl+N"),
      act("file.openLevel", "Open Level…", "Ctrl+O"),
      act("file.openAsset", "Open Asset…", "Ctrl+P"),
      sep,
      act("file.saveLevel", "Save Current Level", "Ctrl+S"),
      act("file.saveLevelAs", "Save Current Level As…", "Ctrl+Alt+S"),
      act("file.saveAll", "Save All", "Ctrl+Shift+S"),
      sep,
      act("file.importIntoLevel", "Import Into Level…"),
      act("file.exportAll", "Export All…"),
      act("file.exportSelected", "Export Selected…"),
      sep,
      act("file.newProject", "New Project…"),
      act("file.openProject", "Open Project…"),
      sub("Recent Projects", [act("file.recentProjects.none", "(none yet)")]),
      sep,
      act("app.exit", "Exit", "Alt+F4"),
    ],
  },
  {
    id: "edit",
    label: "Edit",
    items: [
      act("edit.undo", "Undo", "Ctrl+Z"),
      act("edit.redo", "Redo", "Ctrl+Y"),
      act("edit.undoHistory", "Undo History"),
      sep,
      act("edit.cut", "Cut", "Ctrl+X"),
      act("edit.copy", "Copy", "Ctrl+C"),
      act("edit.paste", "Paste", "Ctrl+V"),
      act("edit.duplicate", "Duplicate", "Ctrl+D"),
      act("edit.delete", "Delete", "Del"),
      sep,
      act("edit.editorPreferences", "Editor Preferences…"),
      act("edit.projectSettings", "Project Settings…"),
      act("edit.plugins", "Plugins…"),
    ],
  },
  {
    id: "window",
    label: "Window",
    items: [
      act("window.viewport", "Viewport"),
      act("window.outliner", "Outliner"),
      act("window.details", "Details"),
      act("window.contentDrawer", "Content Drawer", "Ctrl+Space"),
      act("window.outputLog", "Output Log"),
      act("window.terminal", "Terminal"),
      act("window.explorer", "Explorer"),
      act("window.editor", "Code Editor"),
      act("window.problems", "Problems"),
      act("window.git", "Source Control"),
      act("window.search", "Search"),
      act("window.worldSettings", "World Settings"),
      act("window.placeActors", "Place Actors"),
      act("window.tilemap", "Tilemap"),
      act("window.sequencer", "Sequencer"),
      act("window.sortingLayers", "Sorting Layers…"),
      sep,
      sub("Load Layout", [
        act("window.layout.default", "Default Editor Layout"),
        act("window.layout.discipline.3d", "3D Layout"),
        act("window.layout.discipline.2d", "2D Layout"),
        act("window.layout.discipline.scripting", "Scripting Layout"),
        act("window.layout.load", "Load Layout…"),
      ]),
      act("window.layout.save", "Save Layout As…"),
      act("window.layout.reset", "Reset Layout"),
      sep,
      act("window.fullscreen", "Enable Fullscreen", "F11"),
      sub("Theme", themeMenuItems()),
    ],
  },
  {
    id: "tools",
    label: "Tools",
    items: [
      act("tools.commandPalette", "Command Palette…", "Ctrl+Shift+P"),
      sep,
      act("tools.newRustSystem", "New Rust System…"),
      act("tools.openIde", "Open IDE"),
      act("tools.refreshIdeProject", "Refresh IDE Project"),
      sep,
      act("tools.findInBlueprints", "Find in Blueprints…"),
      act("tools.blueprintDebugger", "Blueprint Debugger"),
      sep,
      act("tools.profiler", "Profiler"),
      act("tools.outputLog", "Output Log"),
    ],
  },
  {
    id: "build",
    label: "Build",
    items: [
      act("build.buildAll", "Build All Levels"),
      sep,
      act("build.lighting", "Build Lighting"),
      act("build.navigation", "Build Navigation"),
      act("build.geometry", "Build Geometry"),
      sep,
      act("build.cook", "Cook Content"),
      act("build.package", "Package Project…"),
    ],
  },
  {
    id: "platforms",
    label: "Platforms",
    items: [
      act("platforms.windows", "Windows"),
      act("platforms.macos", "macOS"),
      act("platforms.linux", "Linux"),
      sep,
      act("platforms.deviceManager", "Device Manager…"),
      act("platforms.packagingSettings", "Packaging Settings…"),
    ],
  },
  {
    id: "select",
    label: "Select",
    items: [
      act("select.all", "Select All", "Ctrl+A"),
      act("select.none", "Deselect All", "Esc"),
      act("select.invert", "Invert Selection", "Ctrl+I"),
      sep,
      act("select.sameClass", "Select All Actors of Same Class"),
      act("select.children", "Select All Children"),
      act("select.descendants", "Select All Descendants"),
    ],
  },
  {
    id: "actor",
    label: "Actor",
    items: [
      sub("Place Actor", [
        act("actor.place.empty", "Empty Actor"),
        act("actor.place.cube", "Cube"),
        act("actor.place.sphere", "Sphere"),
        act("actor.place.cylinder", "Cylinder"),
        act("actor.place.plane", "Plane"),
        sep,
        act("actor.place.pointLight", "Point Light"),
        act("actor.place.directionalLight", "Directional Light"),
        act("actor.place.camera", "Camera"),
      ]),
      sep,
      act("actor.duplicate", "Duplicate", "Ctrl+W"),
      act("actor.delete", "Delete", "Del"),
      act("actor.rename", "Rename", "F2"),
      sep,
      act("actor.attachTo", "Attach To…"),
      act("actor.detach", "Detach"),
      act("actor.group", "Group", "Ctrl+G"),
      act("actor.ungroup", "Ungroup", "Shift+Ctrl+G"),
      sep,
      act("actor.snapToFloor", "Snap to Floor", "End"),
    ],
  },
  {
    id: "help",
    label: "Help",
    items: [
      act("help.documentation", "Documentation"),
      act("help.roadmap", "Engineering Roadmap"),
      act("help.tour", "Interactive Tour"),
      act("help.reportBug", "Report a Bug…"),
      sep,
      act("help.about", "About Infinity Engine"),
    ],
  },
];

function themeMenuItems(): MenuNode[] {
  return BUILTIN_THEMES.map((t) => act(`theme.${t.id}`, t.name));
}

/** How many recent projects the File ▸ Recent Projects submenu lists. */
export const RECENT_PROJECTS_MAX = 10;

/**
 * Build the File ▸ Recent Projects submenu items from the live recent list:
 * `file.recentProjects.{i}` per entry (label = name, tooltip = path, capped),
 * or a single disabled "(none yet)" placeholder when the list is empty. The
 * per-index command ids are wired to open-by-path in `projectStore`.
 */
export function recentProjectsItems(recent: RecentProjectDto[]): MenuNode[] {
  if (recent.length === 0) {
    return [{ kind: "action", command: "file.recentProjects.none", label: "(none yet)", disabled: true }];
  }
  return recent.slice(0, RECENT_PROJECTS_MAX).map((entry, i) => ({
    kind: "action",
    command: `file.recentProjects.${i}`,
    label: entry.name,
    detail: entry.path,
  }));
}

/**
 * The menu bar with the (otherwise static) File ▸ Recent Projects submenu filled
 * in from `recent`. Every other menu passes through untouched — this is the one
 * dynamic section (`MenuBar` recomputes it from the project store's recent list).
 */
export function buildMenuBar(recent: RecentProjectDto[]): TopMenu[] {
  return MENU_BAR.map((menu) => {
    if (menu.id !== "file") return menu;
    return {
      ...menu,
      items: menu.items.map((node) =>
        node.kind === "submenu" && node.label === "Recent Projects"
          ? { ...node, items: recentProjectsItems(recent) }
          : node,
      ),
    };
  });
}

/** Category shown in the command palette per top-level menu. */
const CATEGORY_BY_MENU: Record<string, string> = {
  file: "File",
  edit: "Edit",
  window: "Window",
  tools: "Tools",
  build: "Build",
  platforms: "Platforms",
  select: "Select",
  actor: "Actor",
  help: "Help",
};

/** Flatten the menu tree into palette-ready command definitions. */
export function menuCommandDefs(): CommandDef[] {
  const defs: CommandDef[] = [];
  const walk = (nodes: MenuNode[], category: string, prefix: string) => {
    for (const node of nodes) {
      if (node.kind === "action") {
        defs.push({
          id: node.command,
          title: prefix ? `${prefix}: ${node.label}` : node.label,
          category,
          shortcut: node.shortcut,
        });
      } else if (node.kind === "submenu") {
        walk(node.items, category, node.label.replace(/…$/, ""));
      }
    }
  };
  for (const menu of MENU_BAR) {
    walk(menu.items, CATEGORY_BY_MENU[menu.id] ?? menu.label, "");
  }
  return defs;
}

/** Register the whole menu tree (stubs) + the handlers that are live now. */
export function registerMenuCommands(): void {
  registerCommands(menuCommandDefs());
  // Live from Phase 1: theme switching.
  for (const theme of BUILTIN_THEMES) {
    registerCommands([
      {
        id: `theme.${theme.id}`,
        title: `Theme: ${theme.name}`,
        category: "Window",
        run: () => {
          setTheme(theme.id);
        },
      },
    ]);
  }
}
