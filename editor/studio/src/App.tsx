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
import ContentDrawer from "./shell/ContentDrawer";
import CommandPalette from "./shell/CommandPalette";
import { DockWorkspace } from "./panels/dock/DockWorkspace";
import ViewportPanel from "./viewport/ViewportPanel";
import { bootstrapShellCommands } from "./shell/shellCommands";
import {
  dispatchChord,
  installKeybindingListener,
  registerDefaultKeybindings,
} from "./lib/keybindings";
import { listenTo } from "./lib/events";
import { startLogListener } from "./stores/logStore";
import { initSceneSync, registerSceneCommands } from "./stores/sceneStore";
import { initAssetSync, registerAssetCommands } from "./stores/assetStore";
import { scene as sceneIpc } from "./lib/ipc";

bootstrapShellCommands();
registerDefaultKeybindings();
registerSceneCommands();
registerAssetCommands();

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
    initSceneSync().then((fn) => (disposed ? fn() : (dispose = fn)));
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
    initAssetSync().then((fn) => (disposed ? fn() : (dispose = fn)));
    return () => {
      disposed = true;
      dispose?.();
    };
  }, []);

  // Debounced crash-recovery autosave (P3.5.4): flush unsaved work every 5 s.
  useEffect(() => {
    const id = window.setInterval(() => void sceneIpc.autosave(), 5000);
    return () => window.clearInterval(id);
  }, []);

  // Focus handoff (P2.3.4): the native viewport forwards global-shortcut
  // chords it doesn't consume so the palette/save/etc. keep working while the
  // 3D view holds OS focus. StrictMode double-invoke → `disposed` guard so a
  // late-resolving unlisten still fires.
  useEffect(() => {
    let disposed = false;
    let unlisten: (() => void) | undefined;
    listenTo("viewport://key", (payload) => dispatchChord(payload.chord)).then((fn) => {
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
        {/* The native wgpu child window mirrors this element's rectangle. */}
        <div className="absolute inset-0 flex p-1">
          <ViewportPanel />
        </div>
      </DockWorkspace>
      <ContentDrawer />
      <StatusBar />
      <LayoutDialog />
      <CommandPalette />
    </div>
  );
}
