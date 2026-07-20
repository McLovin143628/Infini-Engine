/**
 * Infinity Engine shell (Phase 1). Vertical composition per ROADMAP §4:
 * custom title bar (chrome + UE-parity menus) → main toolbar → workspace
 * (docking arrives with P1.2) → status bar.
 */
import { useEffect } from "react";
import TitleBar from "./shell/TitleBar";
import MainToolbar from "./shell/MainToolbar";
import StatusBar from "./shell/StatusBar";
import PlaceholderWorkspace from "./shell/PlaceholderWorkspace";
import { bootstrapShellCommands } from "./shell/shellCommands";

bootstrapShellCommands();

export default function App() {
  useEffect(() => {
    // Suppress the browser context menu everywhere except text inputs —
    // panels provide their own context menus (P1.2+).
    const onContextMenu = (e: MouseEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, [contenteditable]")) return;
      e.preventDefault();
    };
    window.addEventListener("contextmenu", onContextMenu);
    return () => window.removeEventListener("contextmenu", onContextMenu);
  }, []);

  return (
    <div className="flex h-full flex-col">
      <TitleBar />
      <MainToolbar />
      <PlaceholderWorkspace />
      <StatusBar />
    </div>
  );
}
