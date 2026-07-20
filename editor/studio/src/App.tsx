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
import { installKeybindingListener, registerDefaultKeybindings } from "./lib/keybindings";
import { startLogListener } from "./stores/logStore";

bootstrapShellCommands();
registerDefaultKeybindings();

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
