/**
 * Main toolbar (P1.1 batch 3) — UE-parity: save, selection mode, add-actor,
 * play controls, platform target. Everything dispatches through the command
 * registry; play/build actions are enumerated stubs until P2/P9.
 */
import { useState } from "react";
import {
  ChevronDown,
  Hammer,
  Mountain,
  MousePointer2,
  Pause,
  Play,
  Plus,
  Save,
  Settings,
  SkipForward,
  Square,
} from "lucide-react";
import { executeCommand } from "../lib/commands";
import { useSimStore } from "../stores/simStore";
import { usePieStore } from "../stores/pieStore";
import { useViewportStore } from "../stores/viewportStore";

function ToolButton(props: {
  label: string;
  command: string;
  children: React.ReactNode;
  accent?: boolean;
}) {
  return (
    <button
      title={props.label}
      aria-label={props.label}
      className={`flex h-7 items-center gap-1 rounded px-2 hover:bg-(--ink-bg-3) ${
        props.accent ? "text-(--ink-success)" : "text-(--ink-text)"
      }`}
      onClick={() => executeCommand(props.command)}
    >
      {props.children}
    </button>
  );
}

function Divider() {
  return <div className="mx-1 h-5 w-px bg-(--ink-border)" />;
}

/** One item in the Play split-button dropdown. */
function MenuItem(props: { label: string; command: string; onPick: () => void }) {
  return (
    <button
      className="flex w-full items-center gap-2 whitespace-nowrap px-3 py-1.5 text-left text-(--ink-text) hover:bg-(--ink-bg-3)"
      onClick={() => {
        props.onPick();
        executeCommand(props.command);
      }}
    >
      {props.label}
    </button>
  );
}

/**
 * The unified play cluster (P9.4): Play-In-Editor (subprocess) is primary, with
 * a split-button dropdown for the mode — "Play (Embedded)", "Play in New Window",
 * and the in-process "Simulate". While a session runs, the transport buttons
 * (Pause/Resume/Stop/Step) drive whichever kind is live; Eject sits in the menu.
 */
function PlayCluster() {
  const pieRunning = usePieStore((s) => s.running);
  const piePaused = usePieStore((s) => s.paused);
  const simRunning = useSimStore((s) => s.running);
  const simPaused = useSimStore((s) => s.paused);
  const [menuOpen, setMenuOpen] = useState(false);

  const running = pieRunning || simRunning;
  const paused = pieRunning ? piePaused : simPaused;
  // Which command family drives the transport buttons.
  const p = pieRunning ? "pie" : "sim";

  return (
    <div
      data-tour="play-cluster"
      className={`relative flex items-center gap-0.5 rounded bg-(--ink-bg-1) p-0.5 ${
        running ? "ring-1 ring-(--ink-success)" : ""
      }`}
    >
      {!running ? (
        <>
          <ToolButton label="Play in Editor (Shift+Alt+P)" command="pie.play" accent>
            <Play size={16} />
            <span className="text-sm">Play</span>
          </ToolButton>
          <button
            title="Play options"
            aria-label="Play options"
            className="flex h-7 items-center rounded px-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
            onClick={() => setMenuOpen((o) => !o)}
          >
            <ChevronDown size={14} />
          </button>
        </>
      ) : (
        <>
          <ToolButton
            label={paused ? "Resume" : "Pause"}
            command={paused ? `${p}.resume` : `${p}.pause`}
            accent={paused}
          >
            {paused ? <Play size={15} /> : <Pause size={15} />}
          </ToolButton>
          <ToolButton label="Stop" command={`${p}.stop`}>
            <Square size={13} />
          </ToolButton>
          <ToolButton label="Step Frame" command={`${p}.step`}>
            <SkipForward size={15} />
          </ToolButton>
          {pieRunning && (
            <button
              title="Play options"
              aria-label="Play options"
              className="flex h-7 items-center rounded px-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
              onClick={() => setMenuOpen((o) => !o)}
            >
              <ChevronDown size={14} />
            </button>
          )}
        </>
      )}

      {menuOpen && (
        <>
          {/* Click-away backdrop. */}
          <div className="fixed inset-0 z-40" onClick={() => setMenuOpen(false)} />
          <div className="absolute right-0 top-9 z-50 min-w-44 overflow-hidden rounded border border-(--ink-border) bg-(--ink-bg-2) py-1 shadow-lg">
            {!running ? (
              <>
                <MenuItem
                  label="Play (Embedded)"
                  command="pie.play"
                  onPick={() => setMenuOpen(false)}
                />
                <MenuItem
                  label="Play in New Window"
                  command="pie.playWindow"
                  onPick={() => setMenuOpen(false)}
                />
                <div className="my-1 h-px bg-(--ink-border)" />
                <MenuItem label="Simulate" command="sim.play" onPick={() => setMenuOpen(false)} />
              </>
            ) : (
              <>
                <MenuItem
                  label="Eject (release input)"
                  command="pie.eject"
                  onPick={() => setMenuOpen(false)}
                />
                <MenuItem label="Stop" command="pie.stop" onPick={() => setMenuOpen(false)} />
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default function MainToolbar() {
  // Editor tool mode, two-way synced with the native viewport via
  // `viewport_set_tool_mode` (Select = pick/gizmo, Sculpt = terrain/"Landscape").
  // The W/E/R translate/rotate/scale gizmo switch is handled inside the native
  // viewport and is not (yet) settable from the webview — see report.
  const toolMode = useViewportStore((s) => s.toolMode);
  const setToolMode = useViewportStore((s) => s.setToolMode);
  return (
    <div className="flex h-10 shrink-0 items-center gap-1 border-b border-(--ink-border) bg-(--ink-bg-2) px-2">
      <ToolButton label="Save Current Level (Ctrl+S)" command="file.saveLevel">
        <Save size={16} />
      </ToolButton>
      <Divider />
      <div className="flex items-center rounded bg-(--ink-bg-1) p-0.5">
        {(
          [
            ["Select", "Select"],
            ["Sculpt", "Landscape"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            title={id === "Sculpt" ? "Landscape (terrain sculpt)" : "Select / gizmo"}
            className={`flex h-6 items-center gap-1 rounded px-2 ${
              toolMode === id
                ? "bg-(--ink-bg-3) text-(--ink-text)"
                : "text-(--ink-text-dim) hover:text-(--ink-text)"
            }`}
            onClick={() => setToolMode(id)}
          >
            {id === "Select" ? <MousePointer2 size={13} /> : <Mountain size={13} />}
            {label}
          </button>
        ))}
      </div>
      <Divider />
      <ToolButton label="Add Actor" command="actor.place.empty">
        <Plus size={16} />
        <span>Add</span>
        <ChevronDown size={12} className="text-(--ink-text-dim)" />
      </ToolButton>
      <ToolButton label="Build All Levels" command="build.buildAll">
        <Hammer size={15} />
      </ToolButton>

      <div className="flex-1" />

      {/* Play cluster (P9.4): PIE subprocess (embedded / new window) + Simulate,
          via a split-button dropdown. A success ring marks a live session. */}
      <PlayCluster />
      <button
        className="flex h-7 items-center gap-1 rounded px-2 text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
        onClick={() => executeCommand("platforms.windows")}
        title="Platform target"
      >
        Windows
        <ChevronDown size={12} />
      </button>

      <div className="flex-1" />

      <ToolButton label="Editor Preferences" command="edit.editorPreferences">
        <Settings size={15} />
      </ToolButton>
    </div>
  );
}
