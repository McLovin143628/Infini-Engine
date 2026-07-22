/**
 * Audio Mixer panel (E-P9): edit the project's named-bus mixer
 * (`inf_audio::MixerConfig`, persisted at `.infinity/mixer.toml`).
 *
 * The mixer is pure data folded into per-bus gains (the audio command-queue
 * doctrine). The panel edits a local draft (in `mixerStore`) and writes it back
 * via `mixer_save` — which validates, persists, live-applies to a running
 * Simulate session, and emits `audio://mixer-changed`. It reloads on open, on
 * `audio://mixer-changed`, and on `project://changed`.
 *
 * Bus routing: `master` is the undeletable root; every other bus picks a
 * non-cyclic parent. `Effect::Gain` is exposed as an editable dB trim; `Lowpass`
 * is shown read-only (its cutoff is folded, but audible filtering needs the cpal
 * sub-track wiring — a follow-up). An entity plays on a bus via the free-form
 * `AudioSource.bus` name.
 */
import { useEffect, useMemo } from "react";
import { Plus, Trash2, Volume2 } from "lucide-react";

import type { MixerBusDto } from "../../bindings/MixerBusDto";
import { NumberField, PropertyRow, PropertySection, TextField } from "../../components/propertyRows";
import { listenTo, type UnlistenFn } from "../../lib/events";
import {
  MASTER,
  addBus,
  deleteBus,
  gainDb,
  lowpassCutoffs,
  renameBus,
  setGainDb,
  updateBus,
  validParents,
  validateMixer,
} from "./mixerModel";
import { isDirty, useMixerStore } from "./mixerStore";

export default function AudioMixerPanel() {
  const draft = useMixerStore((s) => s.draft);
  const error = useMixerStore((s) => s.error);
  const busy = useMixerStore((s) => s.busy);
  const dirty = useMixerStore(isDirty);
  const load = useMixerStore((s) => s.load);
  const mutate = useMixerStore((s) => s.mutate);
  const save = useMixerStore((s) => s.save);
  const revert = useMixerStore((s) => s.revert);

  useEffect(() => {
    void load(true);
    let unlistenMixer: UnlistenFn | undefined;
    let unlistenProject: UnlistenFn | undefined;
    let disposed = false;
    void listenTo("audio://mixer-changed", () => void load(false)).then((fn) =>
      disposed ? fn() : (unlistenMixer = fn),
    );
    void listenTo("project://changed", () => void load(true)).then((fn) =>
      disposed ? fn() : (unlistenProject = fn),
    );
    return () => {
      disposed = true;
      unlistenMixer?.();
      unlistenProject?.();
    };
  }, [load]);

  const validationError = useMemo(() => (draft ? validateMixer(draft) : null), [draft]);

  if (error && !draft) {
    return (
      <div className="flex h-full items-center justify-center p-4 text-center text-xs text-(--ink-text-faint)">
        {error.includes("no project") ? "Open a project to edit its audio mixer." : error}
      </div>
    );
  }
  if (!draft) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-(--ink-text-faint)">
        Loading…
      </div>
    );
  }

  const patchBusAt = (index: number, next: MixerBusDto) => mutate((d) => updateBus(d, index, next));

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex items-center gap-2 border-b border-(--ink-border) bg-(--ink-bg-2) px-2 py-1">
        <Volume2 size={13} className="text-(--ink-text-dim)" />
        <span className="text-xs font-semibold">Audio Mixer</span>
        <div className="flex-1" />
        <button
          onClick={() => mutate(addBus)}
          className="flex h-6 items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-1) px-2 text-xs hover:bg-(--ink-bg-3)"
          title="Add a new bus (parented to master)"
        >
          <Plus size={12} /> Add Bus
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {draft.buses.map((bus, index) => {
          const parents = validParents(draft.buses, bus.name);
          const isMaster = bus.name === MASTER;
          const db = gainDb(bus);
          const cutoffs = lowpassCutoffs(bus);
          return (
            <PropertySection key={index} title={bus.name || "(unnamed)"}>
              <PropertyRow label="Name">
                {isMaster ? (
                  <span className="min-w-0 flex-1 truncate text-xs text-(--ink-text-dim)">
                    master <span className="text-(--ink-text-faint)">(root · locked)</span>
                  </span>
                ) : (
                  <TextField value={bus.name} onChange={(v) => mutate((d) => renameBus(d, index, v))} />
                )}
              </PropertyRow>

              {!isMaster && (
                <PropertyRow label="Parent">
                  <select
                    value={bus.parent ?? MASTER}
                    onChange={(e) => patchBusAt(index, { ...bus, parent: e.target.value })}
                    className="h-6 w-full rounded border border-(--ink-border) bg-(--ink-bg-2) px-1 text-xs outline-none focus:border-(--ink-accent)"
                  >
                    {parents.map((p) => (
                      <option key={p} value={p}>
                        {p}
                      </option>
                    ))}
                  </select>
                </PropertyRow>
              )}

              <PropertyRow label="Volume">
                <div className="flex min-w-0 flex-1 items-center gap-2">
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={Math.min(1, Math.max(0, bus.volume))}
                    onChange={(e) => patchBusAt(index, { ...bus, volume: Number(e.target.value) })}
                    className="min-w-0 flex-1 accent-(--ink-accent)"
                  />
                  <div className="w-14 shrink-0">
                    <NumberField
                      value={bus.volume}
                      step={0.05}
                      onChange={(v) => patchBusAt(index, { ...bus, volume: v })}
                    />
                  </div>
                </div>
              </PropertyRow>

              <PropertyRow label="Gain (dB)">
                <div className="flex min-w-0 flex-1 items-center gap-1">
                  {db === null ? (
                    <button
                      onClick={() => patchBusAt(index, setGainDb(bus, 0))}
                      className="flex h-6 items-center gap-1 rounded border border-(--ink-border) bg-(--ink-bg-1) px-2 text-xs text-(--ink-text-dim) hover:bg-(--ink-bg-3)"
                    >
                      <Plus size={11} /> Add Gain
                    </button>
                  ) : (
                    <>
                      <div className="min-w-0 flex-1">
                        <NumberField
                          value={db}
                          step={0.5}
                          onChange={(v) => patchBusAt(index, setGainDb(bus, v))}
                        />
                      </div>
                      <button
                        aria-label="Remove gain"
                        onClick={() => patchBusAt(index, setGainDb(bus, null))}
                        className="flex size-6 shrink-0 items-center justify-center rounded text-(--ink-text-faint) hover:text-(--ink-error)"
                      >
                        <Trash2 size={12} />
                      </button>
                    </>
                  )}
                </div>
              </PropertyRow>

              {cutoffs.length > 0 && (
                <PropertyRow label="Lowpass">
                  <div
                    className="flex min-w-0 flex-1 flex-wrap gap-1"
                    title="Cutoff is folded deterministically, but audible filtering needs cpal sub-track wiring (follow-up)"
                  >
                    {cutoffs.map((hz, i) => (
                      <span
                        key={i}
                        className="rounded bg-(--ink-bg-3) px-1.5 py-0.5 text-[11px] text-(--ink-text-dim)"
                      >
                        {hz} Hz · read-only
                      </span>
                    ))}
                  </div>
                </PropertyRow>
              )}

              {!isMaster && (
                <div className="px-2 py-1">
                  <button
                    onClick={() => mutate((d) => deleteBus(d, bus.name))}
                    className="flex h-6 items-center gap-1 rounded border border-(--ink-border) px-2 text-xs text-(--ink-text-dim) hover:border-(--ink-error) hover:text-(--ink-error)"
                    title="Delete this bus (its children reparent to master)"
                  >
                    <Trash2 size={12} /> Delete Bus
                  </button>
                </div>
              )}
            </PropertySection>
          );
        })}

        <p className="px-3 py-2 text-[11px] leading-relaxed text-(--ink-text-faint)">
          An entity plays on a bus by its free-form <code>AudioSource.bus</code> name — it need not
          exist here (an unknown bus folds through master). Both Simulate and the shipped runtime
          load this mixer from <code>.infinity/mixer.toml</code>.
        </p>
      </div>

      <div className="flex items-center gap-2 border-t border-(--ink-border) bg-(--ink-bg-2) px-2 py-1.5">
        {validationError ? (
          <span className="min-w-0 flex-1 truncate text-[11px] text-(--ink-error)">
            {validationError}
          </span>
        ) : error ? (
          <span className="min-w-0 flex-1 truncate text-[11px] text-(--ink-error)">{error}</span>
        ) : (
          <span className="min-w-0 flex-1 truncate text-[11px] text-(--ink-text-faint)">
            {dirty ? "Unsaved changes" : "Saved"}
          </span>
        )}
        <button
          onClick={revert}
          disabled={!dirty || busy}
          className="h-6 rounded border border-(--ink-border) px-2 text-xs disabled:opacity-40 enabled:hover:bg-(--ink-bg-3)"
        >
          Revert
        </button>
        <button
          onClick={() => void save()}
          disabled={!dirty || busy || !!validationError}
          className="h-6 rounded bg-(--ink-accent) px-3 text-xs text-white disabled:opacity-40"
        >
          {busy ? "Saving…" : "Save"}
        </button>
      </div>
    </div>
  );
}
