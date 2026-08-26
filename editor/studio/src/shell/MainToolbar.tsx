/**
 * Main toolbar (P1.1 batch 3) — UE-parity: save, selection mode, add-actor,
 * play controls, platform target. Everything dispatches through the command
 * registry; play/build actions are enumerated stubs until P2/P9.
 */
import { useRef, useState } from "react";
import {
  Box,
  ChevronDown,
  Globe,
  Hammer,
  Magnet,
  Maximize2,
  Mountain,
  MousePointer2,
  Move,
  Pause,
  Play,
  Plus,
  RotateCw,
  Save,
  Settings,
  SkipForward,
  Square,
} from "lucide-react";
import type { GizmoModeDto } from "../bindings/GizmoModeDto";
import { executeCommand } from "../lib/commands";
import { CUTOUT_ATTR, useViewportCutout } from "../lib/viewportOverlay";
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
 *
 * # Airspace (UX2 audit)
 *
 * The dropdown is `absolute top-9` under a toolbar that sits directly above the
 * workspace, so it opens ACROSS the native viewport hole — and it had no
 * airspace guard of any kind. UX2 found it and recorded it as PIE-only, on the
 * reasoning that hiding our own child could not uncover an embedded foreign
 * player window anyway; that reasoning is sound and the premise was wrong. The
 * chevron is rendered in the `!running` branch too, where the hole is our own
 * child window drawing OVER the dropdown: a menu you can open, cannot see the
 * lower half of, and whose clicks land in the 3D view. Exported for its arm.
 */
export function PlayCluster() {
  const pieRunning = usePieStore((s) => s.running);
  const piePaused = usePieStore((s) => s.paused);
  const simRunning = useSimStore((s) => s.running);
  const simPaused = useSimStore((s) => s.paused);
  const [menuOpen, setMenuOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  // Cut the dropdown out of the viewport rather than blank it (UX2). Only the
  // PANEL is marked: the click-away backdrop is `fixed inset-0` and marking it
  // would clip the whole child away, which is the full hide with extra steps.
  // It is transparent — no scrim — so a cutout is the right treatment, unlike
  // the palette and the dialogs.
  useViewportCutout(menuOpen, rootRef);

  const running = pieRunning || simRunning;
  const paused = pieRunning ? piePaused : simPaused;
  // Which command family drives the transport buttons.
  const p = pieRunning ? "pie" : "sim";

  return (
    <div
      ref={rootRef}
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
          <div
            {...{ [CUTOUT_ATTR]: "" }}
            className="absolute right-0 top-9 z-50 min-w-44 overflow-hidden rounded border border-(--ink-border) bg-(--ink-bg-2) py-1 shadow-lg"
          >
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

/** Per-mode glyph for the transform-gizmo segmented control (static component,
 *  a switch — same idiom as the Place Actors panel — to satisfy react-hooks). */
function GizmoModeGlyph({ mode }: { mode: GizmoModeDto }) {
  switch (mode) {
    case "Translate":
      return <Move size={13} />;
    case "Rotate":
      return <RotateCw size={13} />;
    case "Scale":
      return <Maximize2 size={13} />;
  }
}

/**
 * Transform-gizmo controls (Wave 2): a Translate/Rotate/Scale segmented control
 * (two-way synced with the native viewport — W/E/R over the viewport updates it
 * here, via `viewport://gizmo`), a World/Local space toggle, and the 3D snap
 * toggle + increment dropdowns (persisted through `viewportStore`).
 */
function GizmoCluster() {
  const gizmoMode = useViewportStore((s) => s.gizmoMode);
  const setGizmoMode = useViewportStore((s) => s.setGizmoMode);
  const gizmoSpace = useViewportStore((s) => s.gizmoSpace);
  const toggleGizmoSpace = useViewportStore((s) => s.toggleGizmoSpace);
  const snapEnabled = useViewportStore((s) => s.snap3dEnabled);
  const setSnapEnabled = useViewportStore((s) => s.setSnap3dEnabled);
  const snapT = useViewportStore((s) => s.snap3dTranslate);
  const setSnapT = useViewportStore((s) => s.setSnap3dTranslate);
  const snapR = useViewportStore((s) => s.snap3dRotate);
  const setSnapR = useViewportStore((s) => s.setSnap3dRotate);
  const snapS = useViewportStore((s) => s.snap3dScale);
  const setSnapS = useViewportStore((s) => s.setSnap3dScale);

  const selectCls =
    "h-6 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1 text-xs text-(--ink-text) outline-none hover:border-(--ink-accent) disabled:opacity-40";

  return (
    <>
      {/* Gizmo mode: Translate / Rotate / Scale (W / E / R). */}
      <div className="flex items-center rounded bg-(--ink-bg-1) p-0.5">
        {(["Translate", "Rotate", "Scale"] as const).map((mode, i) => (
          <button
            key={mode}
            title={`${mode} (${["W", "E", "R"][i]})`}
            aria-label={mode}
            className={`flex h-6 items-center rounded px-2 ${
              gizmoMode === mode
                ? "bg-(--ink-bg-3) text-(--ink-text)"
                : "text-(--ink-text-dim) hover:text-(--ink-text)"
            }`}
            onClick={() => setGizmoMode(mode)}
          >
            <GizmoModeGlyph mode={mode} />
          </button>
        ))}
      </div>
      {/* World / Local space toggle. */}
      <button
        title={`Gizmo space: ${gizmoSpace} — click to toggle World / Local`}
        aria-label={`Gizmo space: ${gizmoSpace}`}
        className="flex h-6 items-center gap-1 rounded px-2 text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
        onClick={() => toggleGizmoSpace()}
      >
        {gizmoSpace === "Local" ? <Box size={13} /> : <Globe size={13} />}
        <span className="text-xs">{gizmoSpace}</span>
      </button>
      {/* Snap toggle + increment dropdowns. */}
      <button
        title={`Snapping ${snapEnabled ? "on" : "off (hold Shift to snap)"}`}
        aria-label="Toggle snapping"
        aria-pressed={snapEnabled}
        className={`flex h-6 items-center rounded px-2 ${
          snapEnabled
            ? "bg-(--ink-bg-3) text-(--ink-accent)"
            : "text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
        }`}
        onClick={() => setSnapEnabled(!snapEnabled)}
      >
        <Magnet size={13} />
      </button>
      <select
        title="Move snap (metres)"
        aria-label="Move snap"
        className={selectCls}
        value={String(snapT)}
        onChange={(e) => setSnapT(Number(e.target.value))}
      >
        {[0.1, 0.5, 1, 5].map((v) => (
          <option key={v} value={v}>
            {v} m
          </option>
        ))}
      </select>
      <select
        title="Rotate snap (degrees)"
        aria-label="Rotate snap"
        className={selectCls}
        value={String(snapR)}
        onChange={(e) => setSnapR(Number(e.target.value))}
      >
        {[5, 15, 45, 90].map((v) => (
          <option key={v} value={v}>
            {v}°
          </option>
        ))}
      </select>
      <select
        title="Scale snap (ratio)"
        aria-label="Scale snap"
        className={selectCls}
        value={String(snapS)}
        onChange={(e) => setSnapS(Number(e.target.value))}
      >
        {[0.05, 0.1, 0.25].map((v) => (
          <option key={v} value={v}>
            {v}
          </option>
        ))}
      </select>
    </>
  );
}

export default function MainToolbar() {
  // Editor tool mode, two-way synced with the native viewport via
  // `viewport_set_tool_mode` (Select = pick/gizmo, Sculpt = terrain/"Landscape").
  // The transform-gizmo mode/space + 3D snap controls (GizmoCluster below) are
  // two-way synced too (`viewport_set_gizmo_*` / `viewport://gizmo`, Wave 2).
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
      {/* Transform gizmo: mode + space + 3D snap (Wave 2). */}
      <GizmoCluster />
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
