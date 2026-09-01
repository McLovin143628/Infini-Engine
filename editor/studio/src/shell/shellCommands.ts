/**
 * Shell command bootstrap: enumerates the full menu tree, registers
 * toolbar-only commands, and implements the handful of actions that are
 * live in Phase 1 (theme switching lives in menus.ts; window controls and
 * fullscreen here). Everything else surfaces "coming in Phase N" through
 * the unhandled-command hook.
 */
import { getCurrentWindow } from "@tauri-apps/api/window";
import type { SpawnKind } from "../bindings/SpawnKind";
import { setCommandHandler, setUnhandledCommandHook } from "../lib/commands";
import { app, character as characterIpc } from "../lib/ipc";
import { DISCIPLINE_LABEL } from "../lib/disciplines";
import { registerMenuCommands } from "../lib/menus";
import { applyDisciplineLayout, useDockLayout } from "../panels/dock/dockLayoutStore";
import { useProjectStore } from "../stores/projectStore";
import { useSceneStore } from "../stores/sceneStore";
import { useShellStore } from "../stores/shellStore";
import { useTourStore } from "../stores/tourStore";

/**
 * Honest, phase-free status messages for command families that are enumerated
 * (the full UE-parity menu tree) but genuinely NOT implemented yet. Anything not
 * matched here falls through to a generic "… is not implemented yet."
 *
 * The old "arrives with Phase N" hints were removed once those phases shipped —
 * most of those families are now live (wired by the scene/project/viewport
 * stores or below) and never reach the unhandled hook at all. Cut/copy/paste and
 * duplicate are now live too (scene clipboard, editor seams). This table is kept
 * as the mechanism for the handful that remain genuinely unbuilt.
 */
const STUB_HINTS: [RegExp, string][] = [
  // Wave E: the two settings surfaces that ARE built now say so by being live.
  // These two are genuinely unbuilt, and the honest message names what exists
  // instead of "is not implemented yet".
  [
    /^edit\.plugins$/,
    "There is no plugin manager yet — game modules are cargo crates in the project workspace (see the Explorer panel).",
  ],
  [
    /^platforms\.packagingSettings$/,
    "Packaging options live in the Package Project dialog (Build ▸ Package Project…); a persisted per-platform settings page is not built yet.",
  ],
];

export function stubHint(id: string): string | undefined {
  return STUB_HINTS.find(([re]) => re.test(id))?.[1];
}

/**
 * Maps each `actor.place.*` menu command (and the toolbar "Add" button, which
 * dispatches `actor.place.empty`) to the `SpawnKind` the scene store creates.
 * The set mirrors `lib/spawnables` — asserted by the wiring-table test.
 */
export const ACTOR_PLACE_KINDS: Record<string, SpawnKind> = {
  "actor.place.empty": "empty",
  "actor.place.cube": "cube",
  "actor.place.sphere": "sphere",
  "actor.place.cylinder": "cylinder",
  "actor.place.plane": "plane",
  "actor.place.pointLight": "point_light",
  "actor.place.directionalLight": "directional_light",
  "actor.place.camera": "camera",
};

export function bootstrapShellCommands(): void {
  registerMenuCommands();

  // The Simulate play cluster (sim.play/pause/stop/step) is a live command
  // family registered by `registerSimCommands()` (stores/simStore) — P8.4.

  setUnhandledCommandHook((cmd) => {
    useShellStore.getState().pushStatus(stubHint(cmd.id) ?? `“${cmd.title}” is not implemented yet.`);
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
  setCommandHandler("window.sequencer", panelToggle("sequencer"));
  setCommandHandler("window.stateMachine", panelToggle("stateMachine"));
  setCommandHandler("window.blendSpace", panelToggle("blendSpace"));

  // Sorting-layer manager dialog (P8.2a).
  setCommandHandler("window.sortingLayers", () => {
    useShellStore.getState().setSortingLayersOpen(true);
  });

  // Settings, end to end (Wave E). Both of these were enumerated with no
  // handler since Phase 1 — the toolbar gear and Edit ▸ Editor Preferences…
  // reached the unhandled hook and toasted "is not implemented yet".
  setCommandHandler("edit.editorPreferences", () => {
    useShellStore.getState().setPreferencesOpen(true);
  });
  setCommandHandler("edit.projectSettings", () => {
    useShellStore.getState().setProjectSettingsOpen(true);
  });

  // The audio mixer panel has been registered since P12.4 and was reachable
  // from nowhere (no menu entry, no command, no drawer path).
  setCommandHandler("window.audioMixer", panelToggle("audioMixer"));

  // Cook / Package dialog (P9.2). Both Build entries open the same dialog — the
  // backend command IS the cook (a `.inf_pack` + manifest); per-platform bundling
  // layers on later (P9.5).
  const openPackageDialog = () => useShellStore.getState().setPackageDialogOpen(true);
  setCommandHandler("build.package", openPackageDialog);
  setCommandHandler("build.cook", openPackageDialog);

  // ── Scene editing: selection + actor placement ─────────────────────────────
  // The scene store already owns edit.undo/redo/delete + file new/open/save
  // (registerSceneCommands) and project new/open (registerProjectCommands); the
  // handlers below wire the remaining enumerated commands to the same live
  // capabilities. Genuinely-unbuilt actions (duplicate, clipboard, attach/group,
  // rename-via-menu, etc.) stay stubbed and surface an honest message.
  const sceneState = () => useSceneStore.getState();

  // Select menu → the selection service (client-side over the live scene tree).
  setCommandHandler("select.all", () => {
    const s = sceneState();
    s.select(Object.keys(s.nodes), false);
  });
  setCommandHandler("select.none", () => sceneState().select([], false));
  setCommandHandler("select.invert", () => {
    const s = sceneState();
    const sel = new Set(s.selection);
    s.select(
      Object.keys(s.nodes).filter((g) => !sel.has(g)),
      false,
    );
  });

  // Actor ▸ Place Actor (and the toolbar "Add" button, which dispatches
  // actor.place.empty) → scene_create the mapped kind at the world origin.
  for (const [id, kind] of Object.entries(ACTOR_PLACE_KINDS)) {
    setCommandHandler(id, () => void sceneState().createEntity(kind, null));
  }
  // Actor ▸ Place Actor ▸ Starter Character (wave GTA1) — its own handler
  // rather than a `SpawnKind`, because it is not a primitive: it is the
  // committed character, placed through the character door so it arrives with a
  // capsule, a movement component and a controller. `character.placeStarter`
  // writes nothing; `actor.newCharacter` (the wizard) is the one that generates
  // assets, and confusing the two is how a project ends up with six sets of
  // near-identical rigs.
  setCommandHandler("actor.place.starterCharacter", () => {
    void characterIpc
      .placeStarter()
      .then(() =>
        useShellStore.getState().pushStatus("Placed the starter character."),
      )
      .catch((e) =>
        useShellStore
          .getState()
          .pushStatus(`Could not place the starter character: ${String(e)}`, 12000),
      );
  });

  // Actor ▸ Delete mirrors Edit ▸ Delete (both drop the current selection).
  setCommandHandler("actor.delete", () => sceneState().deleteSelected());

  // File ▸ Recent Projects placeholder → open the Start screen (lists recents).
  setCommandHandler("file.recentProjects.none", () =>
    useProjectStore.getState().setShowStartScreen(true),
  );

  // Command palette + Content Drawer (P1.4).
  setCommandHandler("tools.commandPalette", () => {
    useShellStore.getState().setPaletteOpen(true);
  });
  setCommandHandler("window.contentDrawer", () => {
    useShellStore.getState().toggleDrawer();
  });

  // The Terrain Import wizard (P16.4a): File ▸ Import Terrain… — deliberately
  // its own entry rather than a branch of "Import Into Level…", because a .png
  // is far more often a texture than a heightmap.
  setCommandHandler("file.importTerrain", () => {
    useShellStore.getState().setTerrainImportOpen(true);
  });

  // The GIS Import wizard (IB-3): File ▸ Import GIS Data… — the one door for
  // roads, watercourses, land cover, footprints and parcels. Its own entry for
  // the Terrain Import wizard's reason: a `.geojson` is never a texture, and a
  // vector layer needs a target kind that no generic importer can guess.
  setCommandHandler("file.importGis", () => {
    useShellStore.getState().setGisImportOpen(true);
  });

  // The capture wizard (P25.4): File ▸ Capture from Photographs… — photographs
  // in, a textured, retopologized asset out, entirely in-engine.
  setCommandHandler("file.captureFromPhotos", () => {
    useShellStore.getState().setCaptureWizardOpen(true);
  });

  // The New Character wizard (P24.5): Actor ▸ New Character from Template… — its
  // own entry rather than a Place Actor row, because it GENERATES six assets and
  // then places an actor wearing them.
  setCommandHandler("actor.newCharacter", () => {
    useShellStore.getState().setCharacterWizardOpen(true);
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

  // Discipline first-run layout presets (P15.3).
  (["3d", "2d", "scripting"] as const).forEach((d) => {
    setCommandHandler(`window.layout.discipline.${d}`, () => {
      applyDisciplineLayout(d);
      useShellStore.getState().pushStatus(`${DISCIPLINE_LABEL[d]} layout applied.`);
    });
  });

  // Interactive first-run tour (P15.3) — replayable from Help ▸ Interactive Tour.
  setCommandHandler("help.tour", () => {
    useTourStore.getState().start();
  });
  setCommandHandler("help.documentation", () => {
    useShellStore
      .getState()
      .pushStatus("Docs: build the mdBook at docs/book (mdbook serve docs/book), or see docs/ROADMAP.md.", 6000);
  });
  setCommandHandler("window.fullscreen", async () => {
    const win = getCurrentWindow();
    const fs = await win.isFullscreen();
    await win.setFullscreen(!fs);
  });
  setCommandHandler("help.about", async () => {
    // Surface the build banner (version + git hash) embedded at build time.
    let info = "Infini Engine — native Rust core, Tauri v2 editor.";
    try {
      info = (await app.buildInfo()).replace(/\n/g, " · ");
    } catch {
      // Backend unavailable (e.g. detached-window context) — keep the fallback.
    }
    useShellStore.getState().pushStatus(info, 6000);
  });
  setCommandHandler("help.roadmap", () => {
    useShellStore.getState().pushStatus("The roadmap lives at docs/ROADMAP.md in the repository.", 6000);
  });
}
