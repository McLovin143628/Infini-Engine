/**
 * Editor Preferences (Wave E, batch A) — the dialog behind Edit ▸ Editor
 * Preferences… and the main toolbar's gear, both of which dispatched a
 * handler-less command until this wave.
 *
 * Every control writes through `settingsStore.patch`, which persists to
 * `editor-settings.toml` (debounced) and re-applies live via
 * `lib/settingsApply.ts` — there is no "Save" button because there is nothing
 * to save: the theme flips, the autosave timer re-arms and the snap steps reach
 * the native viewport as you type. "Close" flushes the pending write.
 *
 * Airspace: a modal overlays the native viewport hole, so `useViewportOverlay`
 * holds it hidden for the dialog's lifetime (`SortingLayersDialog` precedent).
 */
import { useEffect, useState } from "react";
import { RotateCcw, X } from "lucide-react";

import type { GameBindingRowDto } from "../bindings/GameBindingRowDto";
import { getCommand } from "../lib/commands";
import { gameBindings } from "../lib/ipc";
import { chordOf, DEFAULT_KEYBINDINGS, allKeybindings } from "../lib/keybindings";
import { BUILTIN_THEMES } from "../lib/theme";
import { useViewportOverlay } from "../lib/viewportOverlay";
import { useProjectStore } from "../stores/projectStore";
import { useSettingsStore } from "../stores/settingsStore";
import { useShellStore } from "../stores/shellStore";

type TabId = "general" | "appearance" | "viewport" | "keyboard" | "gameBindings";

const TABS: { id: TabId; label: string }[] = [
  { id: "general", label: "General" },
  { id: "appearance", label: "Appearance" },
  { id: "viewport", label: "Viewport" },
  { id: "keyboard", label: "Keyboard" },
  { id: "gameBindings", label: "Game Bindings" },
];

/**
 * **The GAME's binding table** (island wave I5) — the controls a player uses,
 * not the editor's chords.
 *
 * Every row, every token and every conflict comes from the backend
 * (`inf_ui::bindings`), which is the same source the in-game settings dialog
 * reads. There is deliberately **no table in TypeScript**: a second one across a
 * language boundary is the defect `inf_input::default_map`'s own note records,
 * and the last copy of it knew about three of seventeen entries.
 *
 * The capture is the same flow the in-game dialog runs: click a row, press a
 * key, and a key that is already taken names its owner and waits for an answer
 * rather than being stolen.
 */
function GameBindingsTab() {
  const settings = useSettingsStore((s) => s.settings);
  const patch = useSettingsStore((s) => s.patch);
  const [rows, setRows] = useState<GameBindingRowDto[]>([]);
  const [capturing, setCapturing] = useState<string | null>(null);
  const [conflict, setConflict] = useState<{ row: string; token: string; owners: string[] } | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const overrides = settings.game_bindings;

  useEffect(() => {
    let live = true;
    gameBindings
      .table(overrides)
      .then((t) => live && setRows(t))
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [overrides]);

  /** Apply a token to a row, asking first when it is already taken. */
  const apply = (row: string, token: string, swap: boolean) => {
    gameBindings
      .apply(overrides, row, token, swap)
      .then((out) => {
        if (out.conflicts.length > 0) {
          setConflict({ row, token, owners: out.conflicts });
          return;
        }
        setConflict(null);
        patch({ game_bindings: out.overrides });
      })
      .catch((e) => setError(String(e)));
  };

  return (
    <>
      <p className="mb-3 text-xs text-(--ink-text-faint)">
        The controls a player uses. Click a row and press a key; a key that is already taken names
        the control that has it. Only your changes are stored, so a control added by a later build
        still arrives bound — and Simulate honours these, not just a shipped build.
      </p>
      {error && <p className="mb-2 text-xs text-(--ink-danger)">{error}</p>}
      {conflict && (
        <div className="mb-2 rounded border border-(--ink-accent) px-2 py-1 text-xs">
          <span className="mr-2">
            {conflict.token} is already used by {conflict.owners.join(", ")}.
          </span>
          <button
            onClick={() => apply(conflict.row, conflict.token, true)}
            className="mr-2 rounded bg-(--ink-accent) px-2 py-0.5 text-(--ink-text-onaccent)"
          >
            Swap
          </button>
          <button onClick={() => setConflict(null)} className="rounded px-2 py-0.5">
            Cancel
          </button>
        </div>
      )}
      <div className="mb-1 flex gap-2 px-1 text-xs text-(--ink-text-faint)">
        <span className="flex-1">Control</span>
        <span className="w-40">Key</span>
        <span className="w-6" />
      </div>
      {rows.map((r) => (
        <div key={r.id} className="mb-1 flex items-center gap-2 text-xs">
          <span className="min-w-0 flex-1 truncate" title={r.id}>
            {r.label}
            {!r.wired && (
              <span className="ml-1 text-(--ink-text-faint)">(not wired to anything yet)</span>
            )}
          </span>
          {capturing === r.id ? (
            <input
              autoFocus
              readOnly
              value="press a key…"
              onKeyDown={(e) => {
                e.preventDefault();
                setCapturing(null);
                if (e.code === "Escape") return;
                apply(r.id, e.code, false);
              }}
              onBlur={() => setCapturing(null)}
              className="w-40 rounded border border-(--ink-accent) bg-(--ink-bg-2) px-2 py-1 outline-none"
            />
          ) : (
            <button
              onClick={() => {
                setConflict(null);
                setCapturing(r.id);
              }}
              className={`w-40 rounded border px-2 py-1 text-left ${
                r.overridden
                  ? "border-(--ink-accent) text-(--ink-accent)"
                  : "border-(--ink-border) text-(--ink-text)"
              } hover:bg-(--ink-bg-3)`}
            >
              {r.token || "--"}
            </button>
          )}
          <button
            aria-label={`Reset ${r.label}`}
            disabled={!r.overridden}
            onClick={() => {
              const next = { ...overrides };
              delete next[r.id];
              patch({ game_bindings: next });
            }}
            className="rounded p-1 text-(--ink-text-faint) disabled:opacity-30 hover:bg-(--ink-bg-3)"
          >
            <RotateCcw size={12} />
          </button>
        </div>
      ))}
      <button
        onClick={() => patch({ game_bindings: {} })}
        disabled={Object.keys(overrides).length === 0}
        className="mt-2 rounded border border-(--ink-border) px-2 py-1 text-xs disabled:opacity-30 hover:bg-(--ink-bg-3)"
      >
        Reset all
      </button>
    </>
  );
}

/** A labelled numeric row. Clamps at the door; the backend clamps again. */
function NumberRow({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="mb-2 flex items-center gap-3 text-xs">
      <span className="w-56 shrink-0">
        {label}
        {hint && <span className="ml-1 text-(--ink-text-faint)">{hint}</span>}
      </span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const n = Number(e.target.value);
          // NaN at the door: an empty or garbage field must never reach a timer
          // or a divisor. The value is left alone until it parses.
          if (!Number.isFinite(n)) return;
          onChange(Math.min(Math.max(n, min), max));
        }}
        className="w-28 rounded border border-(--ink-border) bg-(--ink-bg-2) px-2 py-1 text-right outline-none focus:border-(--ink-accent)"
      />
    </label>
  );
}

function CheckRow({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label className="mb-2 flex items-center gap-3 text-xs">
      <span className="w-56 shrink-0">{label}</span>
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    </label>
  );
}

/**
 * **Which project the application boots on, and the two ways to change it**
 * (CERT1 audit ruling).
 *
 * The rung ladder is `INF_BOOT_PROJECT` → a DELIBERATE pin → the showcase
 * island → the last project opened → the start screen, and before this row an
 * author could neither see which rung had answered nor undo the pin, which the
 * audit measured as: the first other project you open takes the showcase's
 * place for ever.
 *
 * The phrase is **not composed here**. It is `BootSource::phrase`, carried out
 * of `project_boot_default` on launch and re-answered by each of the two
 * commands, so a ladder that changes changes in one place. "Reset to the
 * showcase" therefore reports what will *actually* happen — on a machine where
 * `inf island build` never ran there is no showcase to reach, and the row says
 * so instead of promising one.
 */
function BootProjectRow() {
  const current = useProjectStore((s) => s.current);
  const bootSource = useProjectStore((s) => s.bootSource);
  const setBootDefault = useProjectStore((s) => s.setBootDefault);
  const deliberate = useSettingsStore((s) => s.settings.boot_project_deliberate);

  return (
    <div className="mb-3 border-t border-(--ink-border) pt-3">
      <div className="mb-2 flex items-start gap-3 text-xs">
        <span className="w-56 shrink-0">Opens on launch</span>
        <span className="text-(--ink-text-dim)">
          {bootSource ?? "not asked this session"}
        </span>
      </div>
      <div className="ml-59 flex gap-2">
        <button
          disabled={!current}
          onClick={() => void setBootDefault(true)}
          title={
            current
              ? `Boot on ${current.name} from now on, whatever else you open.`
              : "Open a project first."
          }
          className="flex h-7 items-center rounded border border-(--ink-border) px-2 text-xs hover:border-(--ink-accent) disabled:opacity-40"
        >
          Make this project the default
        </button>
        <button
          disabled={!deliberate}
          onClick={() => void setBootDefault(false)}
          title="Forget the default, so the showcase island answers again."
          className="flex h-7 items-center rounded border border-(--ink-border) px-2 text-xs hover:border-(--ink-accent) disabled:opacity-40"
        >
          Reset to the showcase
        </button>
      </div>
    </div>
  );
}

export default function PreferencesDialog() {
  const open = useShellStore((s) => s.preferencesOpen);
  const setOpen = useShellStore((s) => s.setPreferencesOpen);
  const pushStatus = useShellStore((s) => s.pushStatus);
  const settings = useSettingsStore((s) => s.settings);
  const error = useSettingsStore((s) => s.error);
  const patch = useSettingsStore((s) => s.patch);
  const resetToDefaults = useSettingsStore((s) => s.resetToDefaults);

  const [tab, setTab] = useState<TabId>("general");
  /** The chord row currently listening for a key press ("" = none). */
  const [capturing, setCapturing] = useState<string | null>(null);

  useViewportOverlay(open);

  if (!open) return null;

  const close = () => {
    void useSettingsStore.getState().flush();
    setCapturing(null);
    setOpen(false);
  };

  const snap = settings.snap_3d;
  const foliage = settings.foliage;

  /** Live rows: what the registry holds right now (defaults + overrides). */
  const live = allKeybindings();
  const rows = [...live].sort((a, b) => a.command.localeCompare(b.command));
  const defaultChordFor = (command: string) =>
    DEFAULT_KEYBINDINGS.find((b) => b.command === command)?.chord;
  /**
   * "Default" means **not overridden by this file**, not "listed in
   * `DEFAULT_KEYBINDINGS`" (Wave E audit, A1). The live map also holds chords
   * registered by other subsystems — `Alt+P` for Simulate, `Shift+Alt+P` for
   * PIE — which are shipped defaults this table does not enumerate. Keying off
   * the table painted both as user overrides, with a Reset button that had
   * nothing to reset.
   */
  const isOverridden = (chord: string) =>
    Object.prototype.hasOwnProperty.call(settings.keybindings, chord);

  const rebind = (command: string, oldChord: string, e: React.KeyboardEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const chord = chordOf(e.nativeEvent);
    if (!chord) return; // a bare modifier — keep listening
    if (chord === "Escape") {
      setCapturing(null);
      return;
    }
    const clash = live.find((b) => b.chord === chord && b.command !== command);
    const overrides = { ...settings.keybindings };
    // Unbind the previous chord for this command unless it IS the default and
    // the new chord is the default too (i.e. a no-op).
    if (oldChord !== chord) overrides[oldChord] = "";
    overrides[chord] = command;
    // Re-binding to the command's own default chord means "no override".
    if (defaultChordFor(command) === chord) delete overrides[chord];
    patch({ keybindings: overrides });
    setCapturing(null);
    pushStatus(
      clash
        ? `${chord} bound to ${command} — it was ${clash.command}.`
        : `${chord} bound to ${command}.`,
    );
  };

  const resetChord = (command: string, chord: string) => {
    const overrides = { ...settings.keybindings };
    delete overrides[chord];
    const def = defaultChordFor(command);
    if (def) delete overrides[def];
    patch({ keybindings: overrides });
  };

  return (
    <div
      className="fixed inset-0 z-[85] flex items-start justify-center bg-black/40 pt-16"
      onPointerDown={(e) => {
        if (e.target === e.currentTarget) close();
      }}
      onKeyDown={(e) => {
        if (e.key === "Escape" && capturing === null) close();
      }}
    >
      <div
        className="flex h-[560px] max-h-[80vh] w-[660px] flex-col rounded-lg border border-(--ink-border-strong) bg-(--ink-bg-1)"
        style={{ boxShadow: `0 16px 48px var(--ink-shadow)` }}
      >
        <div className="flex items-center border-b border-(--ink-border) px-3 py-2">
          <span className="flex-1 font-semibold">Editor Preferences</span>
          <button
            aria-label="Close dialog"
            className="rounded p-1 text-(--ink-text-dim) hover:bg-(--ink-bg-3) hover:text-(--ink-text)"
            onClick={close}
          >
            <X size={14} />
          </button>
        </div>

        <div className="flex min-h-0 flex-1">
          <nav className="w-36 shrink-0 border-r border-(--ink-border) p-2">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`mb-1 block w-full rounded px-2 py-1 text-left text-xs ${
                  tab === t.id
                    ? "bg-(--ink-selection) text-(--ink-text)"
                    : "text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>

          <div className="min-h-0 flex-1 overflow-auto p-4">
            {error && (
              <div className="mb-3 rounded border border-(--ink-error) px-2 py-1 text-xs text-(--ink-error)">
                {error}
              </div>
            )}

            {tab === "general" && (
              <>
                <NumberRow
                  label="Autosave interval"
                  hint="(seconds)"
                  value={settings.autosave_interval_s}
                  min={1}
                  max={3600}
                  step={1}
                  onChange={(autosave_interval_s) => patch({ autosave_interval_s })}
                />
                <CheckRow
                  label="First-run tour already seen"
                  checked={settings.tour_seen}
                  onChange={(tour_seen) => patch({ tour_seen })}
                />
                <BootProjectRow />
                <p className="mt-4 text-xs text-(--ink-text-faint)">
                  Preferences are stored per user in <code>editor-settings.toml</code> beside the
                  saved layouts. Changes apply immediately.
                </p>
                <button
                  onClick={() => {
                    void resetToDefaults();
                    pushStatus("Editor preferences restored to defaults.");
                  }}
                  className="mt-3 flex h-7 items-center gap-1 rounded border border-(--ink-border) px-2 text-xs hover:border-(--ink-accent)"
                >
                  <RotateCcw size={12} /> Restore all defaults
                </button>
              </>
            )}

            {tab === "appearance" && (
              <>
                <div className="mb-2 text-xs text-(--ink-text-dim)">Theme</div>
                {BUILTIN_THEMES.map((t) => (
                  <label key={t.id} className="mb-1 flex items-center gap-2 text-xs">
                    <input
                      type="radio"
                      name="theme"
                      checked={settings.theme_id === t.id}
                      onChange={() => patch({ theme_id: t.id })}
                    />
                    <span>{t.name}</span>
                    <span className="text-(--ink-text-faint)">({t.type})</span>
                  </label>
                ))}
              </>
            )}

            {tab === "viewport" && (
              <>
                <div className="mb-2 text-xs text-(--ink-text-dim)">Camera</div>
                {/* 0.2 … 250 m/s is `EditorCamera`'s own clamp
                    (`inf_viewport::camera::{FLY_SPEED_MIN, FLY_SPEED_MAX}`), not
                    a range of the dialog's choosing — offering more would show a
                    number the camera silently refuses (Wave E audit, A2). */}
                <NumberRow
                  label="Fly speed"
                  hint="(m/s)"
                  value={settings.camera_fly_speed_mps}
                  min={0.2}
                  max={250}
                  step={0.5}
                  onChange={(camera_fly_speed_mps) => patch({ camera_fly_speed_mps })}
                />
                <NumberRow
                  label="Mouse-look sensitivity"
                  hint="(×)"
                  value={settings.camera_look_sensitivity}
                  min={0.05}
                  max={10}
                  step={0.05}
                  onChange={(camera_look_sensitivity) => patch({ camera_look_sensitivity })}
                />

                <div className="mt-4 mb-2 text-xs text-(--ink-text-dim)">
                  Right-click vs. fly (a right-button gesture shorter and smaller than both
                  thresholds opens the context menu; anything longer flies)
                </div>
                <NumberRow
                  label="Click travel threshold"
                  hint="(px)"
                  value={settings.rmb_click_travel_px}
                  min={0}
                  max={64}
                  step={1}
                  onChange={(rmb_click_travel_px) => patch({ rmb_click_travel_px })}
                />
                <NumberRow
                  label="Click time threshold"
                  hint="(ms)"
                  value={settings.rmb_click_ms}
                  min={16}
                  max={2000}
                  step={10}
                  onChange={(rmb_click_ms) => patch({ rmb_click_ms: Math.round(rmb_click_ms) })}
                />

                {/* **Wave EDIT1, clause 1.** The editor camera evaluates the
                    PCG volumes it comes near, so the viewport shows the city the
                    player will see instead of the empty boxes it used to. The
                    scale multiplies the LEVEL's own activation/prefetch radii,
                    so 1.0 is exactly what a player standing there would have
                    loaded; the bounds are `pcg_stream`'s own, not the dialog's
                    (the Wave E audit A2 rule). */}
                <div className="mt-4 mb-2 text-xs text-(--ink-text-dim)">World streaming</div>
                <CheckRow
                  label="Evaluate PCG volumes near the camera"
                  checked={settings.pcg_stream}
                  onChange={(pcg_stream) => patch({ pcg_stream })}
                />
                <NumberRow
                  label="Streaming radius"
                  hint="(× the level's own)"
                  value={settings.pcg_stream_radius_scale}
                  min={0.1}
                  max={8}
                  step={0.1}
                  onChange={(pcg_stream_radius_scale) => patch({ pcg_stream_radius_scale })}
                />

                <div className="mt-4 mb-2 text-xs text-(--ink-text-dim)">Gizmo snap (3D)</div>
                <CheckRow
                  label="Snap always on (otherwise Shift-gated)"
                  checked={snap.always_on}
                  onChange={(always_on) => patch({ snap_3d: { ...snap, always_on } })}
                />
                <NumberRow
                  label="Translate step"
                  hint="(m)"
                  value={snap.translate}
                  min={0.0001}
                  max={1000000}
                  step={0.1}
                  onChange={(translate) => patch({ snap_3d: { ...snap, translate } })}
                />
                <NumberRow
                  label="Rotate step"
                  hint="(°)"
                  value={snap.rotate_deg}
                  min={0.001}
                  max={360}
                  step={1}
                  onChange={(rotate_deg) => patch({ snap_3d: { ...snap, rotate_deg } })}
                />
                <NumberRow
                  label="Scale step"
                  value={snap.scale}
                  min={0.0001}
                  max={1000}
                  step={0.01}
                  onChange={(scale) => patch({ snap_3d: { ...snap, scale } })}
                />

                <div className="mt-4 mb-2 text-xs text-(--ink-text-dim)">Foliage brush</div>
                <NumberRow
                  label="Radius"
                  hint="(m)"
                  value={foliage.radius}
                  min={0.05}
                  max={4096}
                  step={0.5}
                  onChange={(radius) => patch({ foliage: { ...foliage, radius } })}
                />
                <NumberRow
                  label="Density"
                  hint="(instances/m²)"
                  value={foliage.density}
                  min={0}
                  max={1000}
                  step={0.1}
                  onChange={(density) => patch({ foliage: { ...foliage, density } })}
                />
                <NumberRow
                  label="Scale jitter"
                  hint="(±fraction)"
                  value={foliage.scale_jitter}
                  min={0}
                  max={1}
                  step={0.05}
                  onChange={(scale_jitter) => patch({ foliage: { ...foliage, scale_jitter } })}
                />
              </>
            )}

            {tab === "keyboard" && (
              <>
                <p className="mb-3 text-xs text-(--ink-text-faint)">
                  Click a chord to rebind it, then press the new combination (Esc cancels). Only
                  your changes are stored, so future default shortcuts still reach you.
                </p>
                <div className="mb-1 flex gap-2 px-1 text-xs text-(--ink-text-faint)">
                  <span className="w-40">Chord</span>
                  <span className="flex-1">Command</span>
                  <span className="w-6" />
                </div>
                {rows.map((b) => {
                  const cmd = getCommand(b.command);
                  const isDefault = !isOverridden(b.chord);
                  return (
                    <div key={b.chord} className="mb-1 flex items-center gap-2 text-xs">
                      {capturing === b.chord ? (
                        <input
                          autoFocus
                          readOnly
                          value="press a key…"
                          onKeyDown={(e) => rebind(b.command, b.chord, e)}
                          onBlur={() => setCapturing(null)}
                          className="w-40 rounded border border-(--ink-accent) bg-(--ink-bg-2) px-2 py-1 outline-none"
                        />
                      ) : (
                        <button
                          onClick={() => setCapturing(b.chord)}
                          className={`w-40 rounded border px-2 py-1 text-left ${
                            isDefault
                              ? "border-(--ink-border) text-(--ink-text)"
                              : "border-(--ink-accent) text-(--ink-accent)"
                          } hover:bg-(--ink-bg-3)`}
                        >
                          {b.chord}
                        </button>
                      )}
                      <span className="min-w-0 flex-1 truncate" title={b.command}>
                        {cmd?.title ?? b.command}
                      </span>
                      <button
                        aria-label={`Reset ${b.command}`}
                        disabled={isDefault}
                        onClick={() => resetChord(b.command, b.chord)}
                        className="rounded p-1 text-(--ink-text-faint) disabled:opacity-30 hover:bg-(--ink-bg-3)"
                      >
                        <RotateCcw size={12} />
                      </button>
                    </div>
                  );
                })}
              </>
            )}

            {tab === "gameBindings" && <GameBindingsTab />}
          </div>
        </div>

        <div className="flex justify-end gap-2 border-t border-(--ink-border) px-3 py-2">
          <button
            onClick={close}
            className="rounded bg-(--ink-accent) px-3 py-1 text-xs text-(--ink-text-onaccent) hover:bg-(--ink-accent-hover)"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
