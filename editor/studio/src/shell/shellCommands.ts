/**
 * Shell command bootstrap: enumerates the full menu tree, registers
 * toolbar-only commands, and implements the handful of actions that are
 * live in Phase 1 (theme switching lives in menus.ts; window controls and
 * fullscreen here). Everything else surfaces "coming in Phase N" through
 * the unhandled-command hook.
 */
import { getCurrentWindow } from "@tauri-apps/api/window";
import { setCommandHandler, setUnhandledCommandHook } from "../lib/commands";
import { registerMenuCommands } from "../lib/menus";
import { useDockLayout } from "../panels/dock/dockLayoutStore";
import { useShellStore } from "../stores/shellStore";

/** Which phase delivers each stubbed command family (status-toast hint). */
const PHASE_HINTS: [RegExp, string][] = [
  [/^(file\.(newLevel|openLevel|saveLevel|saveLevelAs|saveAll)|edit\.(undo|redo|undoHistory))/, "Phase 3 (scene model & undo)"],
  [/^file\.(openAsset|importIntoLevel|exportAll|exportSelected)/, "Phase 4 (asset system)"],
  [/^file\.(newProject|openProject|recentProjects)/, "Phase 5 (project system)"],
  [/^edit\.(cut|copy|paste|duplicate|delete)/, "Phase 3 (scene model)"],
  [/^edit\.(editorPreferences|projectSettings|plugins)/, "Phase 5"],
  [/^window\.viewport/, "Phase 2–3"],
  [/^tools\.(newRustSystem|openIde|refreshIdeProject)/, "Phase 5 (IDE integration)"],
  [/^tools\.(findInBlueprints|blueprintDebugger)/, "Phase 6 (Blueprints)"],
  [/^tools\.profiler/, "Phase 15"],
  [/^build\./, "Phase 9 (cook & package)"],
  [/^platforms\./, "Phase 9 (desktop export)"],
  [/^select\./, "Phase 3 (selection service)"],
  [/^actor\./, "Phase 3 (scene model)"],
  [/^play\./, "Phase 9 (play-in-editor)"],
  [/^help\.(documentation|reportBug)/, "Phase 15 (docs site)"],
];

function phaseHint(id: string): string | undefined {
  return PHASE_HINTS.find(([re]) => re.test(id))?.[1];
}

export function bootstrapShellCommands(): void {
  registerMenuCommands();

  // The Simulate play cluster (sim.play/pause/stop/step) is a live command
  // family registered by `registerSimCommands()` (stores/simStore) — P8.4.

  setUnhandledCommandHook((cmd) => {
    const hint = phaseHint(cmd.id);
    useShellStore
      .getState()
      .pushStatus(hint ? `“${cmd.title}” arrives with ${hint}.` : `“${cmd.title}” is not implemented yet.`);
  });

  // Live now: window/application controls.
  setCommandHandler("app.exit", () => {
    void getCurrentWindow().close();
  });

  // Panel toggles (P1.2): hidden → restore to lastDock; visible → focus.
  // Detached windows come to the foreground via openPanel's window path.
  const panelToggle = (type: string) => () => {
    useDockLayout.getState().openPanel(type);
  };
  setCommandHandler("window.outliner", panelToggle("outliner"));
  setCommandHandler("window.details", panelToggle("details"));
  setCommandHandler("window.outputLog", panelToggle("outputLog"));
  setCommandHandler("tools.outputLog", panelToggle("outputLog"));
  setCommandHandler("window.terminal", panelToggle("terminal"));
  setCommandHandler("window.explorer", panelToggle("explorer"));
  setCommandHandler("window.git", panelToggle("git"));
  setCommandHandler("window.search", panelToggle("search"));
  setCommandHandler("window.editor", panelToggle("editor"));
  setCommandHandler("window.problems", panelToggle("problems"));
  setCommandHandler("window.worldSettings", panelToggle("worldSettings"));
  setCommandHandler("window.placeActors", panelToggle("placeActors"));
  setCommandHandler("window.tilemap", panelToggle("tilemap"));

  // Sorting-layer manager dialog (P8.2a).
  setCommandHandler("window.sortingLayers", () => {
    useShellStore.getState().setSortingLayersOpen(true);
  });

  // Cook / Package dialog (P9.2). Both Build entries open the same dialog — the
  // backend command IS the cook (a `.inf_pack` + manifest); per-platform bundling
  // layers on later (P9.5).
  const openPackageDialog = () => useShellStore.getState().setPackageDialogOpen(true);
  setCommandHandler("build.package", openPackageDialog);
  setCommandHandler("build.cook", openPackageDialog);

  // Command palette + Content Drawer (P1.4).
  setCommandHandler("tools.commandPalette", () => {
    useShellStore.getState().setPaletteOpen(true);
  });
  setCommandHandler("window.contentDrawer", () => {
    useShellStore.getState().toggleDrawer();
  });

  // Layout persistence (P1.2.5).
  setCommandHandler("window.layout.default", () => {
    useDockLayout.getState().resetLayout();
    useShellStore.getState().pushStatus("Default editor layout restored.");
  });
  setCommandHandler("window.layout.reset", () => {
    useDockLayout.getState().resetLayout();
    useShellStore.getState().pushStatus("Layout reset to default.");
  });
  setCommandHandler("window.layout.save", () => {
    useShellStore.getState().openLayoutDialog("save");
  });
  setCommandHandler("window.layout.load", () => {
    useShellStore.getState().openLayoutDialog("load");
  });
  setCommandHandler("window.fullscreen", async () => {
    const win = getCurrentWindow();
    const fs = await win.isFullscreen();
    await win.setFullscreen(!fs);
  });
  setCommandHandler("help.about", () => {
    useShellStore
      .getState()
      .pushStatus("Infinity Engine — native Rust core, Tauri v2 editor. See docs/ROADMAP.md.", 6000);
  });
  setCommandHandler("help.roadmap", () => {
    useShellStore.getState().pushStatus("The roadmap lives at docs/ROADMAP.md in the repository.", 6000);
  });
}
