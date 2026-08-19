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
import { useState } from "react";
import { RotateCcw, X } from "lucide-react";

import { getCommand } from "../lib/commands";
import { chordOf, DEFAULT_KEYBINDINGS, allKeybindings } from "../lib/keybindings";
import { BUILTIN_THEMES } from "../lib/theme";
import { useViewportOverlay } from "../lib/viewportOverlay";
import { useSettingsStore } from "../stores/settingsStore";
import { useShellStore } from "../stores/shellStore";

type TabId = "general" | "appearance" | "viewport" | "keyboard";

const TABS: { id: TabId; label: string }[] = [
  { id: "general", label: "General" },
  { id: "appearance", label: "Appearance" },
  { id: "viewport", label: "Viewport" },
  { id: "keyboard", label: "Keyboard" },
];

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
                <NumberRow
                  label="Fly speed"
                  hint="(m/s)"
                  value={settings.camera_fly_speed_mps}
                  min={0.05}
                  max={10000}
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
                  const isDefault = defaultChordFor(b.command) === b.chord;
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
