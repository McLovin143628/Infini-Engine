/**
 * Main toolbar (P1.1 batch 3) — UE-parity: save, selection mode, add-actor,
 * play controls, platform target. Everything dispatches through the command
 * registry; play/build actions are enumerated stubs until P2/P9.
 */
import { useState } from "react";
import {
  ChevronDown,
  Hammer,
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

export default function MainToolbar() {
  // Selection-mode segmented control is a visual stub until P2.4 gizmos.
  const [mode, setMode] = useState<"select" | "landscape" | "foliage">("select");
  // Simulate (P8.4): the play cluster reflects the live session state.
  const running = useSimStore((s) => s.running);
  const paused = useSimStore((s) => s.paused);

  return (
    <div className="flex h-10 shrink-0 items-center gap-1 border-b border-(--ink-border) bg-(--ink-bg-2) px-2">
      <ToolButton label="Save Current Level (Ctrl+S)" command="file.saveLevel">
        <Save size={16} />
      </ToolButton>
      <Divider />
      <div className="flex items-center rounded bg-(--ink-bg-1) p-0.5">
        {(
          [
            ["select", "Select"],
            ["landscape", "Landscape"],
            ["foliage", "Foliage"],
          ] as const
        ).map(([id, label]) => (
          <button
            key={id}
            className={`flex h-6 items-center gap-1 rounded px-2 ${
              mode === id
                ? "bg-(--ink-bg-3) text-(--ink-text)"
                : "text-(--ink-text-dim) hover:text-(--ink-text)"
            }`}
            onClick={() => setMode(id)}
          >
            {id === "select" && <MousePointer2 size={13} />}
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

      {/* Simulate controls (P8.4): Play when stopped; Pause/Stop/Step when
          running. A success ring marks a live session. */}
      <div
        className={`flex items-center gap-0.5 rounded bg-(--ink-bg-1) p-0.5 ${
          running ? "ring-1 ring-(--ink-success)" : ""
        }`}
      >
        {!running ? (
          <ToolButton label="Play (Alt+P)" command="sim.play" accent>
            <Play size={16} />
          </ToolButton>
        ) : (
          <>
            <ToolButton
              label={paused ? "Resume (Alt+P)" : "Pause"}
              command={paused ? "sim.play" : "sim.pause"}
              accent={paused}
            >
              {paused ? <Play size={15} /> : <Pause size={15} />}
            </ToolButton>
            <ToolButton label="Stop" command="sim.stop">
              <Square size={13} />
            </ToolButton>
            <ToolButton label="Step Frame" command="sim.step">
              <SkipForward size={15} />
            </ToolButton>
          </>
        )}
      </div>
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
