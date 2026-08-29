/**
 * Bottom status bar (P1.1 batch 3): Content Drawer toggle, Output Log,
 * command console stub, transient status messages, save state, source
 * control placeholder, engine version.
 */
import { useEffect, useState } from "react";
import { ChevronUp, GitBranch, ScrollText, TerminalSquare } from "lucide-react";
import { app } from "../lib/ipc";
import { executeCommand } from "../lib/commands";
import { useSceneStore } from "../stores/sceneStore";
import { useShellStore } from "../stores/shellStore";

export default function StatusBar() {
  const [version, setVersion] = useState("dev");
  const statusMessage = useShellStore((s) => s.statusMessage);
  /**
   * **Discoverability for multi-select** (Wave E). Ctrl+click in the native
   * viewport has toggled the selection since Wave 2 and NOTHING said so — the
   * outline is the only feedback, and it is easy to miss on a small object. A
   * count in the status bar is the cheapest honest signal that the gesture did
   * something.
   */
  const selectionCount = useSceneStore((s) => s.selection.length);

  useEffect(() => {
    app
      .version()
      .then(setVersion)
      .catch(() => setVersion("dev"));
  }, []);

  return (
    <div className="flex h-7 shrink-0 items-center gap-1 border-t border-(--ink-border) bg-(--ink-bg-2) px-2 text-(--ink-text-dim)">
      <button
        data-tour="content-drawer"
        className="flex h-full items-center gap-1 px-2 hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
        onClick={() => executeCommand("window.contentDrawer")}
      >
        <ChevronUp size={13} />
        Content Drawer
      </button>
      <button
        className="flex h-full items-center gap-1 px-2 hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
        onClick={() => executeCommand("window.outputLog")}
      >
        <ScrollText size={13} />
        Output Log
      </button>
      <button
        data-tour="command-palette"
        className="flex h-full items-center gap-1 px-2 hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
        onClick={() => executeCommand("tools.commandPalette")}
      >
        <TerminalSquare size={13} />
        Cmd
      </button>

      {statusMessage && (
        <span className="ml-2 truncate text-(--ink-text)" role="status">
          {statusMessage}
        </span>
      )}

      <div className="flex-1" />

      {selectionCount > 1 && (
        <span className="px-2 text-(--ink-text)">{selectionCount} selected</span>
      )}
      <span className="px-2">All changes saved</span>
      <button className="flex h-full items-center gap-1 px-2 hover:bg-(--ink-bg-3) hover:text-(--ink-text)">
        <GitBranch size={13} />
        Source Control: Off
      </button>
      <span className="px-2">Infini Engine v{version}</span>
    </div>
  );
}
