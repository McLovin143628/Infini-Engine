/**
 * Sequencer panel (P11.4) — the seed timeline.
 *
 * A simple property-track timeline for cutscene / property animation. Pick or
 * create a project-local sequence, scrub the playhead (a NON-undoable live
 * preview in the viewport — it never dirties the scene and is disabled while
 * Simulate runs), and author scalar keys: pick the selected entity + a property
 * path and Capture the live value at the playhead. Keys show as draggable
 * diamonds on each track's lane; drag retimes (rounded to `1 / fps_hint`), and a
 * selected key can flip Step/Linear or be deleted. Play runs a client-side rAF
 * advancing the playhead.
 *
 * A normal DOM panel (not the native viewport) — no airspace concern.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Diamond, Pause, Play, Plus, Trash2 } from "lucide-react";

import type { SeqInterpDto } from "../bindings/SeqInterpDto";
import { listenTo } from "../lib/events";
import { cn } from "../lib/utils";
import { useSceneStore } from "../stores/sceneStore";
import { advancePlayhead, useSequencerStore } from "../stores/sequencerStore";
import { useSimStore } from "../stores/simStore";

/** Horizontal scale: pixels per timeline second. */
const PX_PER_SEC = 100;
/** Left column width for track labels. */
const LABEL_W = 176;
const ROW_H = 28;

/** The curated property paths the Add-Key dropdown offers (the reflection grid's
 *  Transform vector components). Free text covers anything else. */
const CURATED_PATHS = [
  "Transform.translation.x",
  "Transform.translation.y",
  "Transform.translation.z",
  "Transform.rotation.x",
  "Transform.rotation.y",
  "Transform.rotation.z",
  "Transform.scale.x",
  "Transform.scale.y",
  "Transform.scale.z",
];

function shortGuid(g: string): string {
  return g.length > 8 ? `${g.slice(0, 8)}…` : g;
}

export default function SequencerPanel() {
  const sequences = useSequencerStore((s) => s.sequences);
  const current = useSequencerStore((s) => s.current);
  const doc = useSequencerStore((s) => s.doc);
  const playhead = useSequencerStore((s) => s.playhead);
  const playing = useSequencerStore((s) => s.playing);
  const selectedKey = useSequencerStore((s) => s.selectedKey);
  const error = useSequencerStore((s) => s.error);

  const refresh = useSequencerStore((s) => s.refresh);
  const open = useSequencerStore((s) => s.open);
  const create = useSequencerStore((s) => s.create);
  const remove = useSequencerStore((s) => s.remove);
  const setDuration = useSequencerStore((s) => s.setDuration);
  const scrubTo = useSequencerStore((s) => s.scrubTo);
  const endScrub = useSequencerStore((s) => s.endScrub);
  const setPlaying = useSequencerStore((s) => s.setPlaying);
  const selectKey = useSequencerStore((s) => s.selectKey);
  const captureKey = useSequencerStore((s) => s.captureKey);
  const setKeyInterp = useSequencerStore((s) => s.setKeyInterp);
  const deleteKey = useSequencerStore((s) => s.deleteKey);
  const retimeKey = useSequencerStore((s) => s.retimeKey);

  const selection = useSceneStore((s) => s.selection);
  const nodes = useSceneStore((s) => s.nodes);
  const simRunning = useSimStore((s) => s.running);

  const [path, setPath] = useState(CURATED_PATHS[0]);
  const [freePath, setFreePath] = useState("");
  const [dragKey, setDragKey] = useState<{ track: number; index: number; t: number } | null>(null);

  const laneRef = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastTsRef = useRef<number>(0);

  // Initial load + re-sync when any sequence file changes.
  useEffect(() => {
    void refresh();
    const un = listenTo("seq://changed", (name) => {
      void refresh();
      if (name === useSequencerStore.getState().current) void open(name);
    });
    return () => {
      void un.then((f) => f());
    };
  }, [refresh, open]);

  // On unmount, make sure any live preview is restored.
  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      void useSequencerStore.getState().endScrub();
    };
  }, []);

  const selected = selection.length === 1 ? nodes[selection[0]] : null;

  const timelineW = (doc?.duration ?? 5) * PX_PER_SEC;

  const timeFromEvent = useCallback((clientX: number): number => {
    const el = laneRef.current;
    if (!el) return 0;
    const rect = el.getBoundingClientRect();
    const x = clientX - rect.left + el.scrollLeft;
    return Math.max(0, x / PX_PER_SEC);
  }, []);

  // ── playhead scrub (drag on the ruler / lanes) ────────────────────────────
  const onRulerDown = useCallback(
    (e: React.PointerEvent) => {
      if (simRunning) return;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      void scrubTo(timeFromEvent(e.clientX), { force: true });
    },
    [scrubTo, timeFromEvent, simRunning],
  );
  const onRulerMove = useCallback(
    (e: React.PointerEvent) => {
      if (e.buttons !== 1 || simRunning) return;
      void scrubTo(timeFromEvent(e.clientX));
    },
    [scrubTo, timeFromEvent, simRunning],
  );
  const onRulerUp = useCallback(() => {
    void endScrub();
  }, [endScrub]);

  // ── key diamond drag (retime) ─────────────────────────────────────────────
  const onKeyDown = useCallback(
    (e: React.PointerEvent, track: number, index: number, t: number) => {
      e.stopPropagation();
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      selectKey([track, index]);
      setDragKey({ track, index, t });
    },
    [selectKey],
  );
  const onKeyMove = useCallback(
    (e: React.PointerEvent) => {
      if (!dragKey) return;
      setDragKey({ ...dragKey, t: timeFromEvent(e.clientX) });
    },
    [dragKey, timeFromEvent],
  );
  const onKeyUp = useCallback(() => {
    if (!dragKey) return;
    void retimeKey(dragKey.track, dragKey.index, dragKey.t);
    setDragKey(null);
  }, [dragKey, retimeKey]);

  // ── Play (client rAF) ─────────────────────────────────────────────────────
  const stopPlay = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    void endScrub();
  }, [endScrub]);

  const startPlay = useCallback(() => {
    if (simRunning || !current) return;
    setPlaying(true);
    lastTsRef.current = performance.now();
    const step = (ts: number) => {
      const dt = (ts - lastTsRef.current) / 1000;
      lastTsRef.current = ts;
      const store = useSequencerStore.getState();
      const dur = store.doc?.duration ?? 0;
      const { t, done } = advancePlayhead(store.playhead, dt, dur);
      void store.scrubTo(t);
      if (done) {
        stopPlay();
        return;
      }
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
  }, [current, simRunning, setPlaying, stopPlay]);

  const effPath = freePath.trim() || path;
  const canCapture = !!current && !!selected && !!effPath && !simRunning;

  const selKeyData = useMemo(() => {
    if (!doc || !selectedKey) return null;
    const [ti, ki] = selectedKey;
    const track = doc.tracks[ti];
    const k = track?.keys[ki];
    if (!track || !k) return null;
    return { track, k, ti, ki };
  }, [doc, selectedKey]);

  return (
    <div className="flex min-h-0 flex-1 flex-col text-xs text-(--ink-text)">
      {/* ── header: sequence picker + transport ── */}
      <div className="flex flex-wrap items-center gap-2 border-b border-(--ink-border) p-2">
        <select
          className="rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-1"
          value={current ?? ""}
          onChange={(e) => void open(e.target.value)}
        >
          <option value="" disabled>
            {sequences.length ? "Select a sequence…" : "No sequences"}
          </option>
          {sequences.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <button
          type="button"
          title="New sequence"
          className="inline-flex items-center gap-1 rounded border border-(--ink-border) px-2 py-1 hover:border-(--ink-accent)"
          onClick={() => {
            const name = window.prompt("New sequence name");
            if (name) void create(name);
          }}
        >
          <Plus size={13} /> New
        </button>
        <button
          type="button"
          title="Delete sequence"
          disabled={!current}
          className="inline-flex items-center gap-1 rounded border border-(--ink-border) px-2 py-1 hover:border-(--ink-danger) disabled:opacity-40"
          onClick={() => current && void remove(current)}
        >
          <Trash2 size={13} />
        </button>

        {doc && (
          <>
            <button
              type="button"
              title={playing ? "Stop" : "Play"}
              disabled={simRunning}
              className="inline-flex items-center gap-1 rounded border border-(--ink-border) px-2 py-1 hover:border-(--ink-accent) disabled:opacity-40"
              onClick={() => (playing ? stopPlay() : startPlay())}
            >
              {playing ? <Pause size={13} /> : <Play size={13} />}
            </button>
            <label className="flex items-center gap-1 text-(--ink-text-dim)">
              Duration
              <input
                type="number"
                min={0.1}
                step={0.5}
                value={doc.duration}
                className="w-16 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1 py-0.5"
                onChange={(e) => void setDuration(Number(e.target.value))}
              />
              s
            </label>
            <span className="tabular-nums text-(--ink-text-dim)">t = {playhead.toFixed(3)}s</span>
          </>
        )}
      </div>

      {simRunning && (
        <div className="border-b border-(--ink-border) bg-(--ink-bg-2) px-2 py-1 text-(--ink-text-dim)">
          Scrubbing is disabled while Simulate is running.
        </div>
      )}

      {/* ── add-key row ── */}
      {doc && (
        <div className="flex flex-wrap items-center gap-2 border-b border-(--ink-border) p-2">
          <span className="text-(--ink-text-dim)">Add key:</span>
          <span className="rounded bg-(--ink-bg-2) px-1.5 py-0.5">
            {selected ? selected.name : "select one entity"}
          </span>
          <select
            className="rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-1"
            value={path}
            onChange={(e) => setPath(e.target.value)}
          >
            {CURATED_PATHS.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
          <input
            type="text"
            placeholder="or a custom path…"
            value={freePath}
            onChange={(e) => setFreePath(e.target.value)}
            className="w-44 rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-1"
          />
          <button
            type="button"
            disabled={!canCapture}
            title="Capture the live value at the playhead"
            className="inline-flex items-center gap-1 rounded border border-(--ink-border) px-2 py-1 hover:border-(--ink-accent) disabled:opacity-40"
            onClick={() => selected && void captureKey(selected.guid, effPath)}
          >
            <Diamond size={12} /> Capture @ playhead
          </button>
        </div>
      )}

      {/* ── selected-key inspector ── */}
      {selKeyData && (
        <div className="flex flex-wrap items-center gap-2 border-b border-(--ink-border) p-2">
          <span className="text-(--ink-text-dim)">
            Key @ {selKeyData.k.t.toFixed(3)}s = {selKeyData.k.value.toFixed(3)}
          </span>
          <select
            className="rounded border border-(--ink-border) bg-(--ink-bg-2) px-1.5 py-0.5"
            value={selKeyData.k.interp}
            onChange={(e) =>
              void setKeyInterp(selKeyData.ti, selKeyData.ki, e.target.value as SeqInterpDto)
            }
          >
            <option value="linear">Linear</option>
            <option value="step">Step</option>
          </select>
          <button
            type="button"
            className="inline-flex items-center gap-1 rounded border border-(--ink-border) px-2 py-0.5 hover:border-(--ink-danger)"
            onClick={() => void deleteKey(selKeyData.ti, selKeyData.ki)}
          >
            <Trash2 size={12} /> Delete key
          </button>
        </div>
      )}

      {/* ── timeline ── */}
      {doc ? (
        <div className="flex min-h-0 flex-1">
          {/* label column */}
          <div
            className="shrink-0 border-r border-(--ink-border)"
            style={{ width: LABEL_W }}
          >
            <div className="border-b border-(--ink-border) bg-(--ink-bg-2)" style={{ height: ROW_H }} />
            {doc.tracks.map((tr) => (
              <div
                key={`${tr.target}:${tr.path}`}
                className="flex items-center gap-1 overflow-hidden border-b border-(--ink-border) px-2"
                style={{ height: ROW_H }}
                title={`${nodes[tr.target]?.name ?? tr.target} · ${tr.path}`}
              >
                <span className="truncate">{nodes[tr.target]?.name ?? shortGuid(tr.target)}</span>
                <span className="truncate text-(--ink-text-dim)">{tr.path}</span>
              </div>
            ))}
            {doc.tracks.length === 0 && (
              <div
                className="px-2 text-(--ink-text-dim)"
                style={{ lineHeight: `${ROW_H}px` }}
              >
                No tracks yet.
              </div>
            )}
          </div>

          {/* scrollable lanes */}
          <div ref={laneRef} className="relative min-h-0 flex-1 overflow-auto">
            <div style={{ width: timelineW, minWidth: "100%" }}>
              {/* ruler */}
              <div
                className="relative cursor-ew-resize border-b border-(--ink-border) bg-(--ink-bg-2)"
                style={{ height: ROW_H }}
                onPointerDown={onRulerDown}
                onPointerMove={onRulerMove}
                onPointerUp={onRulerUp}
              >
                {Array.from({ length: Math.floor(doc.duration) + 1 }, (_, s) => (
                  <div
                    key={s}
                    className="absolute top-0 h-full border-l border-(--ink-border) pl-1 text-(--ink-text-dim)"
                    style={{ left: s * PX_PER_SEC }}
                  >
                    {s}s
                  </div>
                ))}
              </div>

              {/* track lanes */}
              {doc.tracks.map((tr, ti) => (
                <div
                  key={`${tr.target}:${tr.path}`}
                  className="relative border-b border-(--ink-border)"
                  style={{ height: ROW_H }}
                  onPointerDown={onRulerDown}
                  onPointerMove={onRulerMove}
                  onPointerUp={onRulerUp}
                >
                  {tr.keys.map((k, ki) => {
                    const isDragging = dragKey && dragKey.track === ti && dragKey.index === ki;
                    const t = isDragging ? dragKey.t : k.t;
                    const isSel = selectedKey?.[0] === ti && selectedKey?.[1] === ki;
                    return (
                      <div
                        key={ki}
                        role="button"
                        tabIndex={0}
                        aria-label={`key at ${k.t}s`}
                        title={`${k.t.toFixed(3)}s = ${k.value.toFixed(3)} (${k.interp})`}
                        className={cn(
                          "absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rotate-45 cursor-ew-resize border",
                          isSel
                            ? "border-(--ink-accent) bg-(--ink-accent)"
                            : "border-(--ink-text-dim) bg-(--ink-bg)",
                          k.interp === "step" ? "rounded-none" : "rounded-[2px]",
                        )}
                        style={{ left: t * PX_PER_SEC }}
                        onPointerDown={(e) => onKeyDown(e, ti, ki, k.t)}
                        onPointerMove={onKeyMove}
                        onPointerUp={onKeyUp}
                      />
                    );
                  })}
                </div>
              ))}

              {/* playhead */}
              <div
                className="pointer-events-none absolute top-0 bottom-0 w-px bg-(--ink-accent)"
                style={{ left: playhead * PX_PER_SEC }}
              />
            </div>
          </div>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 items-center justify-center p-4 text-center text-(--ink-text-dim)">
          {sequences.length ? "Select a sequence to edit." : "Create a sequence to begin."}
        </div>
      )}

      {error && (
        <div className="border-t border-(--ink-border) px-2 py-1 text-(--ink-danger)">{error}</div>
      )}
    </div>
  );
}
