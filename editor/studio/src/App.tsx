/**
 * Infinity Engine shell (Phase 1). Vertical composition per ROADMAP §4:
 * custom title bar (chrome + UE-parity menus) → main toolbar → dock
 * workspace (native viewport center + dockable panels) → Content Drawer
 * (push-up, airspace-safe) → status bar.
 */
import { useEffect } from "react";
import TitleBar from "./shell/TitleBar";
import MainToolbar from "./shell/MainToolbar";
import StatusBar from "./shell/StatusBar";
import LayoutDialog from "./shell/LayoutDialog";
import PreferencesDialog from "./shell/PreferencesDialog";
import ProjectSettingsDialog from "./shell/ProjectSettingsDialog";
import SortingLayersDialog from "./shell/SortingLayersDialog";
import PackageDialog from "./shell/PackageDialog";
import ErodeDialog from "./shell/ErodeDialog";
import CaptureWizardDialog from "./shell/CaptureWizardDialog";
import CharacterWizardDialog from "./shell/CharacterWizardDialog";
import TerrainImportDialog from "./shell/TerrainImportDialog";
import ContentDrawer from "./shell/ContentDrawer";
import CommandPalette from "./shell/CommandPalette";
import StartScreen from "./shell/StartScreen";
import FirstRunTour from "./shell/FirstRunTour";
import { DockWorkspace } from "./panels/dock/DockWorkspace";
import ViewportPanel from "./viewport/ViewportPanel";
import ViewportContextMenu from "./viewport/ViewportContextMenu";
import { bootstrapShellCommands } from "./shell/shellCommands";
import { installKeybindingListener, registerDefaultKeybindings } from "./lib/keybindings";
import { listenTo } from "./lib/events";
import { PRIMARY_VIEWPORT, isPrimaryViewport } from "./lib/viewportIds";
import { focusViewport, handleViewportChord, VIEWPORT_PANEL_ID } from "./lib/viewportFocus";
import { startLogListener } from "./stores/logStore";
import { initSceneSync, registerSceneCommands } from "./stores/sceneStore";
import { initAssetSync, registerAssetCommands } from "./stores/assetStore";
import { initCaptureSync } from "./stores/captureWizardStore";
import { initTerrainImportSync } from "./stores/terrainImportStore";
import { initProjectSync, registerProjectCommands, useProjectStore } from "./stores/projectStore";
import { useTourStore } from "./stores/tourStore";
import {
  initSculptKeybindings,
  initViewportSync,
  registerViewportCommands,
} from "./stores/viewportStore";
import { initSimSync, registerSimCommands } from "./stores/simStore";
import { registerDccCommands } from "./lib/dccCommands";
import { openObject, registerObjectEditorCommands } from "./stores/objectEditorCommands";
import { initPieSync, registerPieCommands } from "./stores/pieStore";
import { initEditorSync } from "./stores/editorStore";
import { initLsp } from "./lib/editor/lspBridge";
import { initSettingsSync } from "./lib/settingsApply";
import { initAutosave } from "./stores/autosave";
import { useSettingsStore } from "./stores/settingsStore";

bootstrapShellCommands();
registerDefaultKeybindings();
registerSceneCommands();
registerAssetCommands();
registerProjectCommands();
registerViewportCommands();
registerObjectEditorCommands();
// Wave D: the Model Editor's keyboard, through the same registry as every other
// chord — so a modeller's 1/2/3 and G/R/S are rebindable for free.
registerDccCommands();
registerSimCommands();
registerPieCommands();

export default function App() {
  useEffect(() => {
    // Suppress the browser context menu everywhere except text inputs —
    // panels provide their own context menus.
    const onContextMenu = (e: MouseEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, [contenteditable]")) return;
      e.preventDefault();
    };
    window.addEventListener("contextmenu", onContextMenu);
    return () => window.removeEventListener("contextmenu", onContextMenu);
  }, []);

  // Global keybindings + the backend log stream (main window only —
  // detached windows mirror the log store over the bridge instead).
  useEffect(() => installKeybindingListener(), []);
  useEffect(() => startLogListener(), []);

  // Load the world + subscribe to incremental deltas (P3.2). StrictMode-safe.
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let disposed = false;
    void initSceneSync()
      .then((fn) => (disposed ? fn() : (dispose = fn)))
      .catch((e) => console.error("initSceneSync failed", e));
    return () => {
      disposed = true;
      dispose?.();
    };
  }, []);

  // Load the asset database + subscribe to content-change / import events
  // (P4.4). StrictMode-safe.
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let disposed = false;
    void initAssetSync()
      .then((fn) => (disposed ? fn() : (dispose = fn)))
      .catch((e) => console.error("initAssetSync failed", e));
    return () => {
      disposed = true;
      dispose?.();
    };
  }, []);

  // The Terrain Import wizard folds its own job's `assets://import` events
  // (P16.4a). StrictMode-safe.
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let disposed = false;
    void initTerrainImportSync()
      .then((fn) => (disposed ? fn() : (dispose = fn)))
      .catch((e) => console.error("initTerrainImportSync failed", e));
    return () => {
      disposed = true;
      dispose?.();
    };
  }, []);

  // The capture wizard folds `photogrammetry://progress` (P25.4). Subscribed at
  // the shell rather than in the dialog, so a reconstruction started and left
  // running keeps reporting while the dialog is closed. StrictMode-safe.
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let disposed = false;
    void initCaptureSync()
      .then((fn) => (disposed ? fn() : (dispose = fn)))
      .catch((e) => console.error("initCaptureSync failed", e));
    return () => {
      disposed = true;
      dispose?.();
    };
  }, []);

  // Load the project state + subscribe to project://changed (P5.5).
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let disposed = false;
    void initProjectSync()
      .then((fn) => (disposed ? fn() : (dispose = fn)))
      .catch((e) => console.error("initProjectSync failed", e));
    return () => {
      disposed = true;
      dispose?.();
    };
  }, []);

  // Load per-project pixels-per-unit + apply 2D snap settings; reload on
  // project change (P8.2c). StrictMode-safe.
  useEffect(() => initViewportSync(), []);

  // Load the app-level editor preferences, fold in the legacy localStorage keys
  // once, and keep theme / keybindings / snap / foliage applied as they change
  // (Wave E). StrictMode-safe (the disposer drops the subscription).
  useEffect(() => initSettingsSync(), []);

  // `[` / `]` adjust the sculpt brush radius while the Sculpt tool is active
  // (P10.2b). StrictMode-safe (disposer removes the listener).
  useEffect(() => initSculptKeybindings(), []);

  // Sync Simulate running state + subscribe to sim://state (P8.4). StrictMode-safe.
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let disposed = false;
    void initSimSync()
      .then((fn) => (disposed ? fn() : (dispose = fn)))
      .catch((e) => console.error("initSimSync failed", e));
    return () => {
      disposed = true;
      dispose?.();
    };
  }, []);

  // Sync PIE session state + subscribe to pie://state (P9.4). StrictMode-safe.
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let disposed = false;
    void initPieSync()
      .then((fn) => (disposed ? fn() : (dispose = fn)))
      .catch((e) => console.error("initPieSync failed", e));
    return () => {
      disposed = true;
      dispose?.();
    };
  }, []);

  // First-run tour (P15.3): deferred until the first project is open (before
  // that the StartScreen overlay covers the shell). `maybeAutostart` no-ops if
  // the tour was already seen/dismissed, so re-firing on every project change
  // is safe.
  //
  // It also waits on the SETTINGS now (Wave E): `tourSeen` reads the file, and
  // an unloaded file reads as "seen" so a slow load cannot flash the tour at
  // someone who dismissed it. Subscribing to both stores is what keeps the
  // genuine first run — whichever of the two resolves last re-checks.
  useEffect(() => {
    const check = () => {
      const s = useProjectStore.getState();
      if (s.current !== null && !s.showStartScreen) {
        useTourStore.getState().maybeAutostart();
      }
    };
    check();
    const unsubProject = useProjectStore.subscribe(check);
    const unsubSettings = useSettingsStore.subscribe(check);
    return () => {
      unsubProject();
      unsubSettings();
    };
  }, []);

  // Subscribe to infinity:open-file so the Code Editor opens tabs (P5.1).
  useEffect(() => initEditorSync(), []);

  // LSP bridge: rust-analyzer diagnostics/completions for open Rust files (P5.2).
  useEffect(() => {
    let dispose: (() => void) | undefined;
    let disposed = false;
    void initLsp()
      .then((fn) => (disposed ? fn() : (dispose = fn)))
      .catch((e) => console.error("initLsp failed", e));
    return () => {
      disposed = true;
      dispose?.();
    };
  }, []);

  // Crash-recovery autosave (P3.5.4), at the period the editor preferences
  // hold — `initAutosave` re-arms its interval whenever the setting changes
  // (Wave E). Failure toasts keep their rate limit. StrictMode-safe.
  useEffect(() => initAutosave(), []);

  // Focus handoff (P2.3.4): the native viewport forwards global-shortcut
  // chords it doesn't consume so the palette/save/etc. keep working while the
  // 3D view holds OS focus. StrictMode double-invoke → `disposed` guard so a
  // late-resolving unlisten still fires.
  //
  // The channel is shared by every viewport and the payload carries the id
  // (P23.2a); replaying a chord is a shell-global act, so this accepts them
  // all — but it reads the id rather than assuming, and says why.
  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    // NOT filtered by `payload.viewport`, deliberately: what is forwarded here
    // is a GLOBAL shortcut the viewport declined to consume (Ctrl+S, the
    // palette), and those belong to the shell whichever viewport had focus.
    //
    // `handleViewportChord` takes panel focus BEFORE replaying (P23.2a audit —
    // B1): the arrival of a forwarded chord is proof the native viewport holds
    // OS focus, and it is the only signal available, because the native child
    // swallows DOM pointer events so a click on the 3D view is not a DOM event.
    // Without it, Ctrl+Z over the viewport undid whichever panel was clicked
    // last.
    listenTo("viewport://key", (payload) => handleViewportChord(payload.chord)).then((fn) => {
      if (disposed) fn();
      else unlisten = fn;
    });
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  // **Double-click in the 3D view opens the object** (Wave E). The native
  // child window swallows pointer events, so this event is the only way a
  // double-click there can reach the shell; it routes through the SAME resolver
  // the Outliner's double-click uses, so the two gestures cannot mean different
  // things. StrictMode-safe (`disposed` guard, as the chord listener above).
  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    listenTo("viewport://activate", (payload) => {
      if (!isPrimaryViewport(payload.viewport)) return;
      focusViewport();
      void openObject(payload.guid);
    }).then((fn) => {
      if (disposed) fn();
      else unlisten = fn;
    });
    return () => {
      disposed = true;
      unlisten?.();
    };
  }, []);

  return (
    <div className="flex h-full flex-col">
      <TitleBar />
      <MainToolbar />
      <DockWorkspace>
        {/* The native wgpu child window mirrors this element's rectangle.
            `ViewportPanel` is a registered panel type (P23.2a) but is mounted
            HERE rather than in a dock region: the centre cell's position in the
            React tree is invariant, so the native child resizes and never
            remounts (the Spike A invariant). `PRIMARY_VIEWPORT` is the id it
            attaches under; the props are the registry's component signature. */}
        <div
          className="absolute inset-0 flex p-1"
          // The DOM half of B1: the hole is a native child window and swallows
          // pointer events, but this wrapper covers the viewport toolbar and the
          // padding around it — real DOM surface where a click means "I am
          // working in the viewport now". The chord path above covers clicks on
          // the 3D view itself.
          onPointerDownCapture={focusViewport}
        >
          <ViewportPanel panelId={VIEWPORT_PANEL_ID} params={PRIMARY_VIEWPORT} />
        </div>
      </DockWorkspace>
      <ContentDrawer />
      <ViewportContextMenu />
      <StatusBar />
      <LayoutDialog />
      <PreferencesDialog />
      <ProjectSettingsDialog />
      <SortingLayersDialog />
      <PackageDialog />
      <ErodeDialog />
      <TerrainImportDialog />
      <CharacterWizardDialog />
      <CaptureWizardDialog />
      <CommandPalette />
      <StartScreen />
      <FirstRunTour />
    </div>
  );
}
